//===---- MachineCombiner.cpp - Instcombining on SSA form machine code ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// The machine combiner pass uses machine trace metrics to ensure the combined
// instructions does not lengthen the critical path or the resource depth.
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "machine-combiner"
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/MachineTraceMetrics.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetSchedule.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"

using namespace llvm;

STATISTIC(NumInstCombined, "Number of machineinst combined");

namespace {
class MachineCombiner : public MachineFunctionPass {
  const TargetInstrInfo *TII;
  const TargetRegisterInfo *TRI;
  MCSchedModel SchedModel;
  MachineRegisterInfo *MRI;
  MachineTraceMetrics *Traces;
  MachineTraceMetrics::Ensemble *MinInstr;

  TargetSchedModel TSchedModel;

  /// True if optimizing for code size.
  bool OptSize;

public:
  static char ID;
  MachineCombiner() : MachineFunctionPass(ID) {
    initializeMachineCombinerPass(*PassRegistry::getPassRegistry());
  }
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  bool runOnMachineFunction(MachineFunction &MF) override;
  StringRef getPassName() const override { return "Machine InstCombiner"; }

private:
  bool doSubstitute(unsigned NewSize, unsigned OldSize);
  bool combineInstructions(MachineBasicBlock *);
  MachineInstr *getOperandDef(const MachineOperand &MO);
  unsigned getDepth(SmallVectorImpl<MachineInstr *> &InsInstrs,
                    DenseMap<unsigned, unsigned> &InstrIdxForVirtReg,
                    MachineTraceMetrics::Trace BlockTrace);
  unsigned getLatency(MachineInstr *Root, MachineInstr *NewRoot,
                      MachineTraceMetrics::Trace BlockTrace);
  bool
  improvesCriticalPathLen(MachineBasicBlock *MBB, MachineInstr *Root,
                           MachineTraceMetrics::Trace BlockTrace,
                           SmallVectorImpl<MachineInstr *> &InsInstrs,
                           DenseMap<unsigned, unsigned> &InstrIdxForVirtReg,
                           bool NewCodeHasLessInsts);
  bool preservesResourceLen(MachineBasicBlock *MBB,
                            MachineTraceMetrics::Trace BlockTrace,
                            SmallVectorImpl<MachineInstr *> &InsInstrs,
                            SmallVectorImpl<MachineInstr *> &DelInstrs);
  void instr2instrSC(SmallVectorImpl<MachineInstr *> &Instrs,
                     SmallVectorImpl<const MCSchedClassDesc *> &InstrsSC);
};
}

char MachineCombiner::ID = 0;
char &llvm::MachineCombinerID = MachineCombiner::ID;

INITIALIZE_PASS_BEGIN(MachineCombiner, "machine-combiner",
                      "Machine InstCombiner", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineTraceMetrics)
INITIALIZE_PASS_END(MachineCombiner, "machine-combiner", "Machine InstCombiner",
                    false, false)

void MachineCombiner::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  AU.addPreserved<MachineDominatorTree>();
  AU.addPreserved<MachineLoopInfo>();
  AU.addRequired<MachineTraceMetrics>();
  AU.addPreserved<MachineTraceMetrics>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

MachineInstr *MachineCombiner::getOperandDef(const MachineOperand &MO) {
  MachineInstr *DefInstr = nullptr;
  // We need a virtual register definition.
  if (MO.isReg() && TargetRegisterInfo::isVirtualRegister(MO.getReg()))
    DefInstr = MRI->getUniqueVRegDef(MO.getReg());
  // PHI's have no depth etc.
  if (DefInstr && DefInstr->isPHI())
    DefInstr = nullptr;
  return DefInstr;
}

/// Computes depth of instructions in vector \InsInstr.
///
/// \param InsInstrs is a vector of machine instructions
/// \param InstrIdxForVirtReg is a dense map of virtual register to index
/// of defining machine instruction in \p InsInstrs
/// \param BlockTrace is a trace of machine instructions
///
/// \returns Depth of last instruction in \InsInstrs ("NewRoot")
unsigned
MachineCombiner::getDepth(SmallVectorImpl<MachineInstr *> &InsInstrs,
                          DenseMap<unsigned, unsigned> &InstrIdxForVirtReg,
                          MachineTraceMetrics::Trace BlockTrace) {

  SmallVector<unsigned, 16> InstrDepth;
  assert(TSchedModel.hasInstrSchedModel() && "Missing machine model\n");

  // For each instruction in the new sequence compute the depth based on the
  // operands. Use the trace information when possible. For new operands which
  // are tracked in the InstrIdxForVirtReg map depth is looked up in InstrDepth
  for (auto *InstrPtr : InsInstrs) { // for each Use
    unsigned IDepth = 0;
    DEBUG(dbgs() << "NEW INSTR "; InstrPtr->dump(); dbgs() << "\n";);
    for (const MachineOperand &MO : InstrPtr->operands()) {
      // Check for virtual register operand.
      if (!(MO.isReg() && TargetRegisterInfo::isVirtualRegister(MO.getReg())))
        continue;
      if (!MO.isUse())
        continue;
      unsigned DepthOp = 0;
      unsigned LatencyOp = 0;
      DenseMap<unsigned, unsigned>::iterator II =
          InstrIdxForVirtReg.find(MO.getReg());
      if (II != InstrIdxForVirtReg.end()) {
        // Operand is new virtual register not in trace
        assert(II->second < InstrDepth.size() && "Bad Index");
        MachineInstr *DefInstr = InsInstrs[II->second];
        assert(DefInstr &&
               "There must be a definition for a new virtual register");
        DepthOp = InstrDepth[II->second];
        LatencyOp = TSchedModel.computeOperandLatency(
            DefInstr, DefInstr->findRegisterDefOperandIdx(MO.getReg()),
            InstrPtr, InstrPtr->findRegisterUseOperandIdx(MO.getReg()));
      } else {
        MachineInstr *DefInstr = getOperandDef(MO);
        if (DefInstr) {
          DepthOp = BlockTrace.getInstrCycles(DefInstr).Depth;
          LatencyOp = TSchedModel.computeOperandLatency(
              DefInstr, DefInstr->findRegisterDefOperandIdx(MO.getReg()),
              InstrPtr, InstrPtr->findRegisterUseOperandIdx(MO.getReg()));
        }
      }
      IDepth = std::max(IDepth, DepthOp + LatencyOp);
    }
    InstrDepth.push_back(IDepth);
  }
  unsigned NewRootIdx = InsInstrs.size() - 1;
  return InstrDepth[NewRootIdx];
}

/// Computes instruction latency as max of latency of defined operands.
///
/// \param Root is a machine instruction that could be replaced by NewRoot.
/// It is used to compute a more accurate latency information for NewRoot in
/// case there is a dependent instruction in the same trace (\p BlockTrace)
/// \param NewRoot is the instruction for which the latency is computed
/// \param BlockTrace is a trace of machine instructions
///
/// \returns Latency of \p NewRoot
unsigned MachineCombiner::getLatency(MachineInstr *Root, MachineInstr *NewRoot,
                                     MachineTraceMetrics::Trace BlockTrace) {

  assert(TSchedModel.hasInstrSchedModel() && "Missing machine model\n");

  // Check each definition in NewRoot and compute the latency
  unsigned NewRootLatency = 0;

  for (const MachineOperand &MO : NewRoot->operands()) {
    // Check for virtual register operand.
    if (!(MO.isReg() && TargetRegisterInfo::isVirtualRegister(MO.getReg())))
      continue;
    if (!MO.isDef())
      continue;
    // Get the first instruction that uses MO
    MachineRegisterInfo::reg_iterator RI = MRI->reg_begin(MO.getReg());
    RI++;
    MachineInstr *UseMO = RI->getParent();
    unsigned LatencyOp = 0;
    if (UseMO && BlockTrace.isDepInTrace(Root, UseMO)) {
      LatencyOp = TSchedModel.computeOperandLatency(
          NewRoot, NewRoot->findRegisterDefOperandIdx(MO.getReg()), UseMO,
          UseMO->findRegisterUseOperandIdx(MO.getReg()));
    } else {
      LatencyOp = TSchedModel.computeInstrLatency(NewRoot->getOpcode());
    }
    NewRootLatency = std::max(NewRootLatency, LatencyOp);
  }
  return NewRootLatency;
}

/// True when the new instruction sequence does not lengthen the critical path
/// and the new sequence has less instructions or the new sequence improves the
/// critical path.
/// The DAGCombine code sequence ends in MI (Machine Instruction) Root.
/// The new code sequence ends in MI NewRoot. A necessary condition for the new
/// sequence to replace the old sequence is that it cannot lengthen the critical
/// path. This is decided by the formula:
/// (NewRootDepth + NewRootLatency) <= (RootDepth + RootLatency + RootSlack)).
/// If the new sequence has an equal length critical path but does not reduce
/// the number of instructions (NewCodeHasLessInsts is false), then it is not
/// considered an improvement. The slack is the number of cycles Root can be
/// delayed before the critical patch becomes longer.
bool MachineCombiner::improvesCriticalPathLen(
    MachineBasicBlock *MBB, MachineInstr *Root,
    MachineTraceMetrics::Trace BlockTrace,
    SmallVectorImpl<MachineInstr *> &InsInstrs,
    DenseMap<unsigned, unsigned> &InstrIdxForVirtReg,
    bool NewCodeHasLessInsts) {

  assert(TSchedModel.hasInstrSchedModel() && "Missing machine model\n");
  // NewRoot is the last instruction in the \p InsInstrs vector.
  // Get depth and latency of NewRoot.
  unsigned NewRootIdx = InsInstrs.size() - 1;
  MachineInstr *NewRoot = InsInstrs[NewRootIdx];
  unsigned NewRootDepth = getDepth(InsInstrs, InstrIdxForVirtReg, BlockTrace);
  unsigned NewRootLatency = getLatency(Root, NewRoot, BlockTrace);

  // Get depth, latency and slack of Root.
  unsigned RootDepth = BlockTrace.getInstrCycles(Root).Depth;
  unsigned RootLatency = TSchedModel.computeInstrLatency(Root);
  unsigned RootSlack = BlockTrace.getInstrSlack(Root);

  DEBUG(dbgs() << "DEPENDENCE DATA FOR " << Root << "\n";
        dbgs() << " NewRootDepth: " << NewRootDepth
               << " NewRootLatency: " << NewRootLatency << "\n";
        dbgs() << " RootDepth: " << RootDepth << " RootLatency: " << RootLatency
               << " RootSlack: " << RootSlack << "\n";
        dbgs() << " NewRootDepth + NewRootLatency "
               << NewRootDepth + NewRootLatency << "\n";
        dbgs() << " RootDepth + RootLatency + RootSlack "
               << RootDepth + RootLatency + RootSlack << "\n";);

  unsigned NewCycleCount = NewRootDepth + NewRootLatency;
  unsigned OldCycleCount = RootDepth + RootLatency + RootSlack;
  
  if (NewCodeHasLessInsts)
    return NewCycleCount <= OldCycleCount;
  else
    return NewCycleCount < OldCycleCount;
}

/// helper routine to convert instructions into SC
void MachineCombiner::instr2instrSC(
    SmallVectorImpl<MachineInstr *> &Instrs,
    SmallVectorImpl<const MCSchedClassDesc *> &InstrsSC) {
  for (auto *InstrPtr : Instrs) {
    unsigned Opc = InstrPtr->getOpcode();
    unsigned Idx = TII->get(Opc).getSchedClass();
    const MCSchedClassDesc *SC = SchedModel.getSchedClassDesc(Idx);
    InstrsSC.push_back(SC);
  }
}
/// True when the new instructions do not increase resource length
bool MachineCombiner::preservesResourceLen(
    MachineBasicBlock *MBB, MachineTraceMetrics::Trace BlockTrace,
    SmallVectorImpl<MachineInstr *> &InsInstrs,
    SmallVectorImpl<MachineInstr *> &DelInstrs) {

  // Compute current resource length

  //ArrayRef<const MachineBasicBlock *> MBBarr(MBB);
  SmallVector <const MachineBasicBlock *, 1> MBBarr;
  MBBarr.push_back(MBB);
  unsigned ResLenBeforeCombine = BlockTrace.getResourceLength(MBBarr);

  // Deal with SC rather than Instructions.
  SmallVector<const MCSchedClassDesc *, 16> InsInstrsSC;
  SmallVector<const MCSchedClassDesc *, 16> DelInstrsSC;

  instr2instrSC(InsInstrs, InsInstrsSC);
  instr2instrSC(DelInstrs, DelInstrsSC);

  ArrayRef<const MCSchedClassDesc *> MSCInsArr = makeArrayRef(InsInstrsSC);
  ArrayRef<const MCSchedClassDesc *> MSCDelArr = makeArrayRef(DelInstrsSC);

  // Compute new resource length.
  unsigned ResLenAfterCombine =
      BlockTrace.getResourceLength(MBBarr, MSCInsArr, MSCDelArr);

  DEBUG(dbgs() << "RESOURCE DATA: \n";
        dbgs() << " resource len before: " << ResLenBeforeCombine
               << " after: " << ResLenAfterCombine << "\n";);

  return ResLenAfterCombine <= ResLenBeforeCombine;
}

/// \returns true when new instruction sequence should be generated
/// independent if it lengthens critical path or not
bool MachineCombiner::doSubstitute(unsigned NewSize, unsigned OldSize) {
  if (OptSize && (NewSize < OldSize))
    return true;
  if (!TSchedModel.hasInstrSchedModel())
    return true;
  return false;
}

/// Substitute a slow code sequence with a faster one by
/// evaluating instruction combining pattern.
/// The prototype of such a pattern is MUl + ADD -> MADD. Performs instruction
/// combining based on machine trace metrics. Only combine a sequence of
/// instructions  when this neither lengthens the critical path nor increases
/// resource pressure. When optimizing for codesize always combine when the new
/// sequence is shorter.
bool MachineCombiner::combineInstructions(MachineBasicBlock *MBB) {
  bool Changed = false;
  DEBUG(dbgs() << "Combining MBB " << MBB->getName() << "\n");

  auto BlockIter = MBB->begin();

  while (BlockIter != MBB->end()) {
    auto &MI = *BlockIter++;

    DEBUG(dbgs() << "INSTR "; MI.dump(); dbgs() << "\n";);
    SmallVector<MachineCombinerPattern::MC_PATTERN, 16> Patterns;
    // The motivating example is:
    //
    //     MUL  Other        MUL_op1 MUL_op2  Other
    //      \    /               \      |    /
    //      ADD/SUB      =>        MADD/MSUB
    //      (=Root)                (=NewRoot)

    // The DAGCombine code always replaced MUL + ADD/SUB by MADD. While this is
    // usually beneficial for code size it unfortunately can hurt performance
    // when the ADD is on the critical path, but the MUL is not. With the
    // substitution the MUL becomes part of the critical path (in form of the
    // MADD) and can lengthen it on architectures where the MADD latency is
    // longer than the ADD latency.
    //
    // For each instruction we check if it can be the root of a combiner
    // pattern. Then for each pattern the new code sequence in form of MI is
    // generated and evaluated. When the efficiency criteria (don't lengthen
    // critical path, don't use more resources) is met the new sequence gets
    // hooked up into the basic block before the old sequence is removed.
    //
    // The algorithm does not try to evaluate all patterns and pick the best.
    // This is only an artificial restriction though. In practice there is
    // mostly one pattern, and getMachineCombinerPatterns() can order patterns
    // based on an internal cost heuristic.

    if (TII->getMachineCombinerPatterns(MI, Patterns)) {
      for (auto P : Patterns) {
        SmallVector<MachineInstr *, 16> InsInstrs;
        SmallVector<MachineInstr *, 16> DelInstrs;
        DenseMap<unsigned, unsigned> InstrIdxForVirtReg;
        if (!MinInstr)
          MinInstr = Traces->getEnsemble(MachineTraceMetrics::TS_MinInstrCount);
        MachineTraceMetrics::Trace BlockTrace = MinInstr->getTrace(MBB);
        Traces->verifyAnalysis();
        TII->genAlternativeCodeSequence(MI, P, InsInstrs, DelInstrs,
                                        InstrIdxForVirtReg);
        unsigned NewInstCount = InsInstrs.size();
        unsigned OldInstCount = DelInstrs.size();
        // Found pattern, but did not generate alternative sequence.
        // This can happen e.g. when an immediate could not be materialized
        // in a single instruction.
        if (!NewInstCount)
          continue;
        // Substitute when we optimize for codesize and the new sequence has
        // fewer instructions OR
        // the new sequence neither lengthens the critical path nor increases
        // resource pressure.
        if (doSubstitute(NewInstCount, OldInstCount) ||
            (improvesCriticalPathLen(MBB, &MI, BlockTrace, InsInstrs,
                                      InstrIdxForVirtReg,
                                      NewInstCount < OldInstCount) &&
             preservesResourceLen(MBB, BlockTrace, InsInstrs, DelInstrs))) {
          for (auto *InstrPtr : InsInstrs)
            MBB->insert((MachineBasicBlock::iterator) &MI, InstrPtr);
          for (auto *InstrPtr : DelInstrs)
            InstrPtr->eraseFromParentAndMarkDBGValuesForRemoval();

          Changed = true;
          ++NumInstCombined;

          Traces->invalidate(MBB);
          Traces->verifyAnalysis();
          // Eagerly stop after the first pattern fires.
          break;
        } else {
          // Cleanup instructions of the alternative code sequence. There is no
          // use for them.
          MachineFunction *MF = MBB->getParent();
          for (auto *InstrPtr : InsInstrs)
            MF->DeleteMachineInstr(InstrPtr);
        }
        InstrIdxForVirtReg.clear();
      }
    }
  }

  return Changed;
}

bool MachineCombiner::runOnMachineFunction(MachineFunction &MF) {
  const TargetSubtargetInfo &STI = MF.getSubtarget();
  TII = STI.getInstrInfo();
  TRI = STI.getRegisterInfo();
  SchedModel = STI.getSchedModel();
  TSchedModel.init(SchedModel, &STI, TII);
  MRI = &MF.getRegInfo();
  Traces = &getAnalysis<MachineTraceMetrics>();
  MinInstr = 0;

  OptSize = MF.getFunction()->hasFnAttribute(Attribute::OptimizeForSize);

  DEBUG(dbgs() << getPassName() << ": " << MF.getName() << '\n');
  if (!TII->useMachineCombiner()) {
    DEBUG(dbgs() << "  Skipping pass: Target does not support machine combiner\n");
    return false;
  }

  bool Changed = false;

  // Try to combine instructions.
  for (auto &MBB : MF)
    Changed |= combineInstructions(&MBB);

  return Changed;
}
