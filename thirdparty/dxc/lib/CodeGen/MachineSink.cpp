//===-- MachineSink.cpp - Sinking for machine instructions ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass moves instructions into successor blocks when possible, so that
// they aren't executed on paths where their results aren't needed.
//
// This pass is not intended to be a replacement or a complete alternative
// for an LLVM-IR-level sinking pass. It is only designed to sink simple
// constructs that are not exposed before lowering and instruction selection.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/Passes.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SparseBitVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachinePostDominators.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"
using namespace llvm;

#define DEBUG_TYPE "machine-sink"

static cl::opt<bool>
SplitEdges("machine-sink-split",
           cl::desc("Split critical edges during machine sinking"),
           cl::init(true), cl::Hidden);

static cl::opt<bool>
UseBlockFreqInfo("machine-sink-bfi",
           cl::desc("Use block frequency info to find successors to sink"),
           cl::init(true), cl::Hidden);


STATISTIC(NumSunk,      "Number of machine instructions sunk");
STATISTIC(NumSplit,     "Number of critical edges split");
STATISTIC(NumCoalesces, "Number of copies coalesced");

namespace {
  class MachineSinking : public MachineFunctionPass {
    const TargetInstrInfo *TII;
    const TargetRegisterInfo *TRI;
    MachineRegisterInfo  *MRI;     // Machine register information
    MachineDominatorTree *DT;      // Machine dominator tree
    MachinePostDominatorTree *PDT; // Machine post dominator tree
    MachineLoopInfo *LI;
    const MachineBlockFrequencyInfo *MBFI;
    AliasAnalysis *AA;

    // Remember which edges have been considered for breaking.
    SmallSet<std::pair<MachineBasicBlock*,MachineBasicBlock*>, 8>
    CEBCandidates;
    // Remember which edges we are about to split.
    // This is different from CEBCandidates since those edges
    // will be split.
    SetVector<std::pair<MachineBasicBlock*,MachineBasicBlock*> > ToSplit;

    SparseBitVector<> RegsToClearKillFlags;

    typedef std::map<MachineBasicBlock *, SmallVector<MachineBasicBlock *, 4>>
        AllSuccsCache;

  public:
    static char ID; // Pass identification
    MachineSinking() : MachineFunctionPass(ID) {
      initializeMachineSinkingPass(*PassRegistry::getPassRegistry());
    }

    bool runOnMachineFunction(MachineFunction &MF) override;

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.setPreservesCFG();
      MachineFunctionPass::getAnalysisUsage(AU);
      AU.addRequired<AliasAnalysis>();
      AU.addRequired<MachineDominatorTree>();
      AU.addRequired<MachinePostDominatorTree>();
      AU.addRequired<MachineLoopInfo>();
      AU.addPreserved<MachineDominatorTree>();
      AU.addPreserved<MachinePostDominatorTree>();
      AU.addPreserved<MachineLoopInfo>();
      if (UseBlockFreqInfo)
        AU.addRequired<MachineBlockFrequencyInfo>();
    }

    void releaseMemory() override {
      CEBCandidates.clear();
    }

  private:
    bool ProcessBlock(MachineBasicBlock &MBB);
    bool isWorthBreakingCriticalEdge(MachineInstr *MI,
                                     MachineBasicBlock *From,
                                     MachineBasicBlock *To);
    /// \brief Postpone the splitting of the given critical
    /// edge (\p From, \p To).
    ///
    /// We do not split the edges on the fly. Indeed, this invalidates
    /// the dominance information and thus triggers a lot of updates
    /// of that information underneath.
    /// Instead, we postpone all the splits after each iteration of
    /// the main loop. That way, the information is at least valid
    /// for the lifetime of an iteration.
    ///
    /// \return True if the edge is marked as toSplit, false otherwise.
    /// False can be returned if, for instance, this is not profitable.
    bool PostponeSplitCriticalEdge(MachineInstr *MI,
                                   MachineBasicBlock *From,
                                   MachineBasicBlock *To,
                                   bool BreakPHIEdge);
    bool SinkInstruction(MachineInstr *MI, bool &SawStore,
                         AllSuccsCache &AllSuccessors);
    bool AllUsesDominatedByBlock(unsigned Reg, MachineBasicBlock *MBB,
                                 MachineBasicBlock *DefMBB,
                                 bool &BreakPHIEdge, bool &LocalUse) const;
    MachineBasicBlock *FindSuccToSinkTo(MachineInstr *MI, MachineBasicBlock *MBB,
               bool &BreakPHIEdge, AllSuccsCache &AllSuccessors);
    bool isProfitableToSinkTo(unsigned Reg, MachineInstr *MI,
                              MachineBasicBlock *MBB,
                              MachineBasicBlock *SuccToSinkTo,
                              AllSuccsCache &AllSuccessors);

    bool PerformTrivialForwardCoalescing(MachineInstr *MI,
                                         MachineBasicBlock *MBB);

    SmallVector<MachineBasicBlock *, 4> &
    GetAllSortedSuccessors(MachineInstr *MI, MachineBasicBlock *MBB,
                           AllSuccsCache &AllSuccessors) const;
  };
} // end anonymous namespace

char MachineSinking::ID = 0;
char &llvm::MachineSinkingID = MachineSinking::ID;
INITIALIZE_PASS_BEGIN(MachineSinking, "machine-sink",
                "Machine code sinking", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTree)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfo)
INITIALIZE_AG_DEPENDENCY(AliasAnalysis)
INITIALIZE_PASS_END(MachineSinking, "machine-sink",
                "Machine code sinking", false, false)

bool MachineSinking::PerformTrivialForwardCoalescing(MachineInstr *MI,
                                                     MachineBasicBlock *MBB) {
  if (!MI->isCopy())
    return false;

  unsigned SrcReg = MI->getOperand(1).getReg();
  unsigned DstReg = MI->getOperand(0).getReg();
  if (!TargetRegisterInfo::isVirtualRegister(SrcReg) ||
      !TargetRegisterInfo::isVirtualRegister(DstReg) ||
      !MRI->hasOneNonDBGUse(SrcReg))
    return false;

  const TargetRegisterClass *SRC = MRI->getRegClass(SrcReg);
  const TargetRegisterClass *DRC = MRI->getRegClass(DstReg);
  if (SRC != DRC)
    return false;

  MachineInstr *DefMI = MRI->getVRegDef(SrcReg);
  if (DefMI->isCopyLike())
    return false;
  DEBUG(dbgs() << "Coalescing: " << *DefMI);
  DEBUG(dbgs() << "*** to: " << *MI);
  MRI->replaceRegWith(DstReg, SrcReg);
  MI->eraseFromParent();

  // Conservatively, clear any kill flags, since it's possible that they are no
  // longer correct.
  MRI->clearKillFlags(SrcReg);

  ++NumCoalesces;
  return true;
}

/// AllUsesDominatedByBlock - Return true if all uses of the specified register
/// occur in blocks dominated by the specified block. If any use is in the
/// definition block, then return false since it is never legal to move def
/// after uses.
bool
MachineSinking::AllUsesDominatedByBlock(unsigned Reg,
                                        MachineBasicBlock *MBB,
                                        MachineBasicBlock *DefMBB,
                                        bool &BreakPHIEdge,
                                        bool &LocalUse) const {
  assert(TargetRegisterInfo::isVirtualRegister(Reg) &&
         "Only makes sense for vregs");

  // Ignore debug uses because debug info doesn't affect the code.
  if (MRI->use_nodbg_empty(Reg))
    return true;

  // BreakPHIEdge is true if all the uses are in the successor MBB being sunken
  // into and they are all PHI nodes. In this case, machine-sink must break
  // the critical edge first. e.g.
  //
  // BB#1: derived from LLVM BB %bb4.preheader
  //   Predecessors according to CFG: BB#0
  //     ...
  //     %reg16385<def> = DEC64_32r %reg16437, %EFLAGS<imp-def,dead>
  //     ...
  //     JE_4 <BB#37>, %EFLAGS<imp-use>
  //   Successors according to CFG: BB#37 BB#2
  //
  // BB#2: derived from LLVM BB %bb.nph
  //   Predecessors according to CFG: BB#0 BB#1
  //     %reg16386<def> = PHI %reg16434, <BB#0>, %reg16385, <BB#1>
  BreakPHIEdge = true;
  for (MachineOperand &MO : MRI->use_nodbg_operands(Reg)) {
    MachineInstr *UseInst = MO.getParent();
    unsigned OpNo = &MO - &UseInst->getOperand(0);
    MachineBasicBlock *UseBlock = UseInst->getParent();
    if (!(UseBlock == MBB && UseInst->isPHI() &&
          UseInst->getOperand(OpNo+1).getMBB() == DefMBB)) {
      BreakPHIEdge = false;
      break;
    }
  }
  if (BreakPHIEdge)
    return true;

  for (MachineOperand &MO : MRI->use_nodbg_operands(Reg)) {
    // Determine the block of the use.
    MachineInstr *UseInst = MO.getParent();
    unsigned OpNo = &MO - &UseInst->getOperand(0);
    MachineBasicBlock *UseBlock = UseInst->getParent();
    if (UseInst->isPHI()) {
      // PHI nodes use the operand in the predecessor block, not the block with
      // the PHI.
      UseBlock = UseInst->getOperand(OpNo+1).getMBB();
    } else if (UseBlock == DefMBB) {
      LocalUse = true;
      return false;
    }

    // Check that it dominates.
    if (!DT->dominates(MBB, UseBlock))
      return false;
  }

  return true;
}

bool MachineSinking::runOnMachineFunction(MachineFunction &MF) {
  if (skipOptnoneFunction(*MF.getFunction()))
    return false;

  DEBUG(dbgs() << "******** Machine Sinking ********\n");

  TII = MF.getSubtarget().getInstrInfo();
  TRI = MF.getSubtarget().getRegisterInfo();
  MRI = &MF.getRegInfo();
  DT = &getAnalysis<MachineDominatorTree>();
  PDT = &getAnalysis<MachinePostDominatorTree>();
  LI = &getAnalysis<MachineLoopInfo>();
  MBFI = UseBlockFreqInfo ? &getAnalysis<MachineBlockFrequencyInfo>() : nullptr;
  AA = &getAnalysis<AliasAnalysis>();

  bool EverMadeChange = false;

  while (1) {
    bool MadeChange = false;

    // Process all basic blocks.
    CEBCandidates.clear();
    ToSplit.clear();
    for (auto &MBB: MF)
      MadeChange |= ProcessBlock(MBB);

    // If we have anything we marked as toSplit, split it now.
    for (auto &Pair : ToSplit) {
      auto NewSucc = Pair.first->SplitCriticalEdge(Pair.second, this);
      if (NewSucc != nullptr) {
        DEBUG(dbgs() << " *** Splitting critical edge:"
              " BB#" << Pair.first->getNumber()
              << " -- BB#" << NewSucc->getNumber()
              << " -- BB#" << Pair.second->getNumber() << '\n');
        MadeChange = true;
        ++NumSplit;
      } else
        DEBUG(dbgs() << " *** Not legal to break critical edge\n");
    }
    // If this iteration over the code changed anything, keep iterating.
    if (!MadeChange) break;
    EverMadeChange = true;
  }

  // Now clear any kill flags for recorded registers.
  for (auto I : RegsToClearKillFlags)
    MRI->clearKillFlags(I);
  RegsToClearKillFlags.clear();

  return EverMadeChange;
}

bool MachineSinking::ProcessBlock(MachineBasicBlock &MBB) {
  // Can't sink anything out of a block that has less than two successors.
  if (MBB.succ_size() <= 1 || MBB.empty()) return false;

  // Don't bother sinking code out of unreachable blocks. In addition to being
  // unprofitable, it can also lead to infinite looping, because in an
  // unreachable loop there may be nowhere to stop.
  if (!DT->isReachableFromEntry(&MBB)) return false;

  bool MadeChange = false;

  // Cache all successors, sorted by frequency info and loop depth.
  AllSuccsCache AllSuccessors;

  // Walk the basic block bottom-up.  Remember if we saw a store.
  MachineBasicBlock::iterator I = MBB.end();
  --I;
  bool ProcessedBegin, SawStore = false;
  do {
    MachineInstr *MI = I;  // The instruction to sink.

    // Predecrement I (if it's not begin) so that it isn't invalidated by
    // sinking.
    ProcessedBegin = I == MBB.begin();
    if (!ProcessedBegin)
      --I;

    if (MI->isDebugValue())
      continue;

    bool Joined = PerformTrivialForwardCoalescing(MI, &MBB);
    if (Joined) {
      MadeChange = true;
      continue;
    }

    if (SinkInstruction(MI, SawStore, AllSuccessors))
      ++NumSunk, MadeChange = true;

    // If we just processed the first instruction in the block, we're done.
  } while (!ProcessedBegin);

  return MadeChange;
}

bool MachineSinking::isWorthBreakingCriticalEdge(MachineInstr *MI,
                                                 MachineBasicBlock *From,
                                                 MachineBasicBlock *To) {
  // FIXME: Need much better heuristics.

  // If the pass has already considered breaking this edge (during this pass
  // through the function), then let's go ahead and break it. This means
  // sinking multiple "cheap" instructions into the same block.
  if (!CEBCandidates.insert(std::make_pair(From, To)).second)
    return true;

  if (!MI->isCopy() && !TII->isAsCheapAsAMove(MI))
    return true;

  // MI is cheap, we probably don't want to break the critical edge for it.
  // However, if this would allow some definitions of its source operands
  // to be sunk then it's probably worth it.
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg() || !MO.isUse())
      continue;
    unsigned Reg = MO.getReg();
    if (Reg == 0)
      continue;

    // We don't move live definitions of physical registers,
    // so sinking their uses won't enable any opportunities.
    if (TargetRegisterInfo::isPhysicalRegister(Reg))
      continue;

    // If this instruction is the only user of a virtual register,
    // check if breaking the edge will enable sinking
    // both this instruction and the defining instruction.
    if (MRI->hasOneNonDBGUse(Reg)) {
      // If the definition resides in same MBB,
      // claim it's likely we can sink these together.
      // If definition resides elsewhere, we aren't
      // blocking it from being sunk so don't break the edge.
      MachineInstr *DefMI = MRI->getVRegDef(Reg);
      if (DefMI->getParent() == MI->getParent())
        return true;
    }
  }

  return false;
}

bool MachineSinking::PostponeSplitCriticalEdge(MachineInstr *MI,
                                               MachineBasicBlock *FromBB,
                                               MachineBasicBlock *ToBB,
                                               bool BreakPHIEdge) {
  if (!isWorthBreakingCriticalEdge(MI, FromBB, ToBB))
    return false;

  // Avoid breaking back edge. From == To means backedge for single BB loop.
  if (!SplitEdges || FromBB == ToBB)
    return false;

  // Check for backedges of more "complex" loops.
  if (LI->getLoopFor(FromBB) == LI->getLoopFor(ToBB) &&
      LI->isLoopHeader(ToBB))
    return false;

  // It's not always legal to break critical edges and sink the computation
  // to the edge.
  //
  // BB#1:
  // v1024
  // Beq BB#3
  // <fallthrough>
  // BB#2:
  // ... no uses of v1024
  // <fallthrough>
  // BB#3:
  // ...
  //       = v1024
  //
  // If BB#1 -> BB#3 edge is broken and computation of v1024 is inserted:
  //
  // BB#1:
  // ...
  // Bne BB#2
  // BB#4:
  // v1024 =
  // B BB#3
  // BB#2:
  // ... no uses of v1024
  // <fallthrough>
  // BB#3:
  // ...
  //       = v1024
  //
  // This is incorrect since v1024 is not computed along the BB#1->BB#2->BB#3
  // flow. We need to ensure the new basic block where the computation is
  // sunk to dominates all the uses.
  // It's only legal to break critical edge and sink the computation to the
  // new block if all the predecessors of "To", except for "From", are
  // not dominated by "From". Given SSA property, this means these
  // predecessors are dominated by "To".
  //
  // There is no need to do this check if all the uses are PHI nodes. PHI
  // sources are only defined on the specific predecessor edges.
  if (!BreakPHIEdge) {
    for (MachineBasicBlock::pred_iterator PI = ToBB->pred_begin(),
           E = ToBB->pred_end(); PI != E; ++PI) {
      if (*PI == FromBB)
        continue;
      if (!DT->dominates(ToBB, *PI))
        return false;
    }
  }

  ToSplit.insert(std::make_pair(FromBB, ToBB));
  
  return true;
}

static bool AvoidsSinking(MachineInstr *MI, MachineRegisterInfo *MRI) {
  return MI->isInsertSubreg() || MI->isSubregToReg() || MI->isRegSequence();
}

/// collectDebgValues - Scan instructions following MI and collect any
/// matching DBG_VALUEs.
static void collectDebugValues(MachineInstr *MI,
                               SmallVectorImpl<MachineInstr *> &DbgValues) {
  DbgValues.clear();
  if (!MI->getOperand(0).isReg())
    return;

  MachineBasicBlock::iterator DI = MI; ++DI;
  for (MachineBasicBlock::iterator DE = MI->getParent()->end();
       DI != DE; ++DI) {
    if (!DI->isDebugValue())
      return;
    if (DI->getOperand(0).isReg() &&
        DI->getOperand(0).getReg() == MI->getOperand(0).getReg())
      DbgValues.push_back(DI);
  }
}

/// isProfitableToSinkTo - Return true if it is profitable to sink MI.
bool MachineSinking::isProfitableToSinkTo(unsigned Reg, MachineInstr *MI,
                                          MachineBasicBlock *MBB,
                                          MachineBasicBlock *SuccToSinkTo,
                                          AllSuccsCache &AllSuccessors) {
  assert (MI && "Invalid MachineInstr!");
  assert (SuccToSinkTo && "Invalid SinkTo Candidate BB");

  if (MBB == SuccToSinkTo)
    return false;

  // It is profitable if SuccToSinkTo does not post dominate current block.
  if (!PDT->dominates(SuccToSinkTo, MBB))
    return true;

  // It is profitable to sink an instruction from a deeper loop to a shallower
  // loop, even if the latter post-dominates the former (PR21115).
  if (LI->getLoopDepth(MBB) > LI->getLoopDepth(SuccToSinkTo))
    return true;

  // Check if only use in post dominated block is PHI instruction.
  bool NonPHIUse = false;
  for (MachineInstr &UseInst : MRI->use_nodbg_instructions(Reg)) {
    MachineBasicBlock *UseBlock = UseInst.getParent();
    if (UseBlock == SuccToSinkTo && !UseInst.isPHI())
      NonPHIUse = true;
  }
  if (!NonPHIUse)
    return true;

  // If SuccToSinkTo post dominates then also it may be profitable if MI
  // can further profitably sinked into another block in next round.
  bool BreakPHIEdge = false;
  // FIXME - If finding successor is compile time expensive then cache results.
  if (MachineBasicBlock *MBB2 =
          FindSuccToSinkTo(MI, SuccToSinkTo, BreakPHIEdge, AllSuccessors))
    return isProfitableToSinkTo(Reg, MI, SuccToSinkTo, MBB2, AllSuccessors);

  // If SuccToSinkTo is final destination and it is a post dominator of current
  // block then it is not profitable to sink MI into SuccToSinkTo block.
  return false;
}

/// Get the sorted sequence of successors for this MachineBasicBlock, possibly
/// computing it if it was not already cached.
SmallVector<MachineBasicBlock *, 4> &
MachineSinking::GetAllSortedSuccessors(MachineInstr *MI, MachineBasicBlock *MBB,
                                       AllSuccsCache &AllSuccessors) const {

  // Do we have the sorted successors in cache ?
  auto Succs = AllSuccessors.find(MBB);
  if (Succs != AllSuccessors.end())
    return Succs->second;

  SmallVector<MachineBasicBlock *, 4> AllSuccs(MBB->succ_begin(),
                                               MBB->succ_end());

  // Handle cases where sinking can happen but where the sink point isn't a
  // successor. For example:
  //
  //   x = computation
  //   if () {} else {}
  //   use x
  //
  const std::vector<MachineDomTreeNode *> &Children =
    DT->getNode(MBB)->getChildren();
  for (const auto &DTChild : Children)
    // DomTree children of MBB that have MBB as immediate dominator are added.
    if (DTChild->getIDom()->getBlock() == MI->getParent() &&
        // Skip MBBs already added to the AllSuccs vector above.
        !MBB->isSuccessor(DTChild->getBlock()))
      AllSuccs.push_back(DTChild->getBlock());

  // Sort Successors according to their loop depth or block frequency info.
  std::stable_sort(
      AllSuccs.begin(), AllSuccs.end(),
      [this](const MachineBasicBlock *L, const MachineBasicBlock *R) {
        uint64_t LHSFreq = MBFI ? MBFI->getBlockFreq(L).getFrequency() : 0;
        uint64_t RHSFreq = MBFI ? MBFI->getBlockFreq(R).getFrequency() : 0;
        bool HasBlockFreq = LHSFreq != 0 && RHSFreq != 0;
        return HasBlockFreq ? LHSFreq < RHSFreq
                            : LI->getLoopDepth(L) < LI->getLoopDepth(R);
      });

  auto it = AllSuccessors.insert(std::make_pair(MBB, AllSuccs));

  return it.first->second;
}

/// FindSuccToSinkTo - Find a successor to sink this instruction to.
MachineBasicBlock *MachineSinking::FindSuccToSinkTo(MachineInstr *MI,
                                   MachineBasicBlock *MBB,
                                   bool &BreakPHIEdge,
                                   AllSuccsCache &AllSuccessors) {

  assert (MI && "Invalid MachineInstr!");
  assert (MBB && "Invalid MachineBasicBlock!");

  // Loop over all the operands of the specified instruction.  If there is
  // anything we can't handle, bail out.

  // SuccToSinkTo - This is the successor to sink this instruction to, once we
  // decide.
  MachineBasicBlock *SuccToSinkTo = nullptr;
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg()) continue;  // Ignore non-register operands.

    unsigned Reg = MO.getReg();
    if (Reg == 0) continue;

    if (TargetRegisterInfo::isPhysicalRegister(Reg)) {
      if (MO.isUse()) {
        // If the physreg has no defs anywhere, it's just an ambient register
        // and we can freely move its uses. Alternatively, if it's allocatable,
        // it could get allocated to something with a def during allocation.
        if (!MRI->isConstantPhysReg(Reg, *MBB->getParent()))
          return nullptr;
      } else if (!MO.isDead()) {
        // A def that isn't dead. We can't move it.
        return nullptr;
      }
    } else {
      // Virtual register uses are always safe to sink.
      if (MO.isUse()) continue;

      // If it's not safe to move defs of the register class, then abort.
      if (!TII->isSafeToMoveRegClassDefs(MRI->getRegClass(Reg)))
        return nullptr;

      // Virtual register defs can only be sunk if all their uses are in blocks
      // dominated by one of the successors.
      if (SuccToSinkTo) {
        // If a previous operand picked a block to sink to, then this operand
        // must be sinkable to the same block.
        bool LocalUse = false;
        if (!AllUsesDominatedByBlock(Reg, SuccToSinkTo, MBB,
                                     BreakPHIEdge, LocalUse))
          return nullptr;

        continue;
      }

      // Otherwise, we should look at all the successors and decide which one
      // we should sink to. If we have reliable block frequency information
      // (frequency != 0) available, give successors with smaller frequencies
      // higher priority, otherwise prioritize smaller loop depths.
      for (MachineBasicBlock *SuccBlock :
           GetAllSortedSuccessors(MI, MBB, AllSuccessors)) {
        bool LocalUse = false;
        if (AllUsesDominatedByBlock(Reg, SuccBlock, MBB,
                                    BreakPHIEdge, LocalUse)) {
          SuccToSinkTo = SuccBlock;
          break;
        }
        if (LocalUse)
          // Def is used locally, it's never safe to move this def.
          return nullptr;
      }

      // If we couldn't find a block to sink to, ignore this instruction.
      if (!SuccToSinkTo)
        return nullptr;
      if (!isProfitableToSinkTo(Reg, MI, MBB, SuccToSinkTo, AllSuccessors))
        return nullptr;
    }
  }

  // It is not possible to sink an instruction into its own block.  This can
  // happen with loops.
  if (MBB == SuccToSinkTo)
    return nullptr;

  // It's not safe to sink instructions to EH landing pad. Control flow into
  // landing pad is implicitly defined.
  if (SuccToSinkTo && SuccToSinkTo->isLandingPad())
    return nullptr;

  return SuccToSinkTo;
}

/// SinkInstruction - Determine whether it is safe to sink the specified machine
/// instruction out of its current block into a successor.
bool MachineSinking::SinkInstruction(MachineInstr *MI, bool &SawStore,
                                     AllSuccsCache &AllSuccessors) {
  // Don't sink insert_subreg, subreg_to_reg, reg_sequence. These are meant to
  // be close to the source to make it easier to coalesce.
  if (AvoidsSinking(MI, MRI))
    return false;

  // Check if it's safe to move the instruction.
  if (!MI->isSafeToMove(AA, SawStore))
    return false;

  // Convergent operations may only be moved to control equivalent locations.
  if (MI->isConvergent())
    return false;

  // FIXME: This should include support for sinking instructions within the
  // block they are currently in to shorten the live ranges.  We often get
  // instructions sunk into the top of a large block, but it would be better to
  // also sink them down before their first use in the block.  This xform has to
  // be careful not to *increase* register pressure though, e.g. sinking
  // "x = y + z" down if it kills y and z would increase the live ranges of y
  // and z and only shrink the live range of x.

  bool BreakPHIEdge = false;
  MachineBasicBlock *ParentBlock = MI->getParent();
  MachineBasicBlock *SuccToSinkTo =
      FindSuccToSinkTo(MI, ParentBlock, BreakPHIEdge, AllSuccessors);

  // If there are no outputs, it must have side-effects.
  if (!SuccToSinkTo)
    return false;


  // If the instruction to move defines a dead physical register which is live
  // when leaving the basic block, don't move it because it could turn into a
  // "zombie" define of that preg. E.g., EFLAGS. (<rdar://problem/8030636>)
  for (unsigned I = 0, E = MI->getNumOperands(); I != E; ++I) {
    const MachineOperand &MO = MI->getOperand(I);
    if (!MO.isReg()) continue;
    unsigned Reg = MO.getReg();
    if (Reg == 0 || !TargetRegisterInfo::isPhysicalRegister(Reg)) continue;
    if (SuccToSinkTo->isLiveIn(Reg))
      return false;
  }

  DEBUG(dbgs() << "Sink instr " << *MI << "\tinto block " << *SuccToSinkTo);

  // If the block has multiple predecessors, this is a critical edge.
  // Decide if we can sink along it or need to break the edge.
  if (SuccToSinkTo->pred_size() > 1) {
    // We cannot sink a load across a critical edge - there may be stores in
    // other code paths.
    bool TryBreak = false;
    bool store = true;
    if (!MI->isSafeToMove(AA, store)) {
      DEBUG(dbgs() << " *** NOTE: Won't sink load along critical edge.\n");
      TryBreak = true;
    }

    // We don't want to sink across a critical edge if we don't dominate the
    // successor. We could be introducing calculations to new code paths.
    if (!TryBreak && !DT->dominates(ParentBlock, SuccToSinkTo)) {
      DEBUG(dbgs() << " *** NOTE: Critical edge found\n");
      TryBreak = true;
    }

    // Don't sink instructions into a loop.
    if (!TryBreak && LI->isLoopHeader(SuccToSinkTo)) {
      DEBUG(dbgs() << " *** NOTE: Loop header found\n");
      TryBreak = true;
    }

    // Otherwise we are OK with sinking along a critical edge.
    if (!TryBreak)
      DEBUG(dbgs() << "Sinking along critical edge.\n");
    else {
      // Mark this edge as to be split.
      // If the edge can actually be split, the next iteration of the main loop
      // will sink MI in the newly created block.
      bool Status =
        PostponeSplitCriticalEdge(MI, ParentBlock, SuccToSinkTo, BreakPHIEdge);
      if (!Status)
        DEBUG(dbgs() << " *** PUNTING: Not legal or profitable to "
              "break critical edge\n");
      // The instruction will not be sunk this time.
      return false;
    }
  }

  if (BreakPHIEdge) {
    // BreakPHIEdge is true if all the uses are in the successor MBB being
    // sunken into and they are all PHI nodes. In this case, machine-sink must
    // break the critical edge first.
    bool Status = PostponeSplitCriticalEdge(MI, ParentBlock,
                                            SuccToSinkTo, BreakPHIEdge);
    if (!Status)
      DEBUG(dbgs() << " *** PUNTING: Not legal or profitable to "
            "break critical edge\n");
    // The instruction will not be sunk this time.
    return false;
  }

  // Determine where to insert into. Skip phi nodes.
  MachineBasicBlock::iterator InsertPos = SuccToSinkTo->begin();
  while (InsertPos != SuccToSinkTo->end() && InsertPos->isPHI())
    ++InsertPos;

  // collect matching debug values.
  SmallVector<MachineInstr *, 2> DbgValuesToSink;
  collectDebugValues(MI, DbgValuesToSink);

  // Move the instruction.
  SuccToSinkTo->splice(InsertPos, ParentBlock, MI,
                       ++MachineBasicBlock::iterator(MI));

  // Move debug values.
  for (SmallVectorImpl<MachineInstr *>::iterator DBI = DbgValuesToSink.begin(),
         DBE = DbgValuesToSink.end(); DBI != DBE; ++DBI) {
    MachineInstr *DbgMI = *DBI;
    SuccToSinkTo->splice(InsertPos, ParentBlock,  DbgMI,
                         ++MachineBasicBlock::iterator(DbgMI));
  }

  // Conservatively, clear any kill flags, since it's possible that they are no
  // longer correct.
  // Note that we have to clear the kill flags for any register this instruction
  // uses as we may sink over another instruction which currently kills the
  // used registers.
  for (MachineOperand &MO : MI->operands()) {
    if (MO.isReg() && MO.isUse())
      RegsToClearKillFlags.set(MO.getReg()); // Remember to clear kill flags.
  }

  return true;
}
