//===-- SelectionDAGISel.cpp - Implement the SelectionDAGISel class -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements the SelectionDAGISel class.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GCStrategy.h"
#include "ScheduleDAGSDNodes.h"
#include "SelectionDAGBuilder.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/CodeGen/Analysis.h"
#include "llvm/CodeGen/FastISel.h"
#include "llvm/CodeGen/FunctionLoweringInfo.h"
#include "llvm/CodeGen/GCMetadata.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/ScheduleHazardRecognizer.h"
#include "llvm/CodeGen/SchedulerRegistry.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/CodeGen/WinEHFuncInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetIntrinsicInfo.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include <algorithm>
using namespace llvm;

#define DEBUG_TYPE "isel"

STATISTIC(NumFastIselFailures, "Number of instructions fast isel failed on");
STATISTIC(NumFastIselSuccess, "Number of instructions fast isel selected");
STATISTIC(NumFastIselBlocks, "Number of blocks selected entirely by fast isel");
STATISTIC(NumDAGBlocks, "Number of blocks selected using DAG");
STATISTIC(NumDAGIselRetries,"Number of times dag isel has to try another path");
STATISTIC(NumEntryBlocks, "Number of entry blocks encountered");
STATISTIC(NumFastIselFailLowerArguments,
          "Number of entry blocks where fast isel failed to lower arguments");

#ifndef NDEBUG
static cl::opt<bool>
EnableFastISelVerbose2("fast-isel-verbose2", cl::Hidden,
          cl::desc("Enable extra verbose messages in the \"fast\" "
                   "instruction selector"));

  // Terminators
STATISTIC(NumFastIselFailRet,"Fast isel fails on Ret");
STATISTIC(NumFastIselFailBr,"Fast isel fails on Br");
STATISTIC(NumFastIselFailSwitch,"Fast isel fails on Switch");
STATISTIC(NumFastIselFailIndirectBr,"Fast isel fails on IndirectBr");
STATISTIC(NumFastIselFailInvoke,"Fast isel fails on Invoke");
STATISTIC(NumFastIselFailResume,"Fast isel fails on Resume");
STATISTIC(NumFastIselFailUnreachable,"Fast isel fails on Unreachable");

  // Standard binary operators...
STATISTIC(NumFastIselFailAdd,"Fast isel fails on Add");
STATISTIC(NumFastIselFailFAdd,"Fast isel fails on FAdd");
STATISTIC(NumFastIselFailSub,"Fast isel fails on Sub");
STATISTIC(NumFastIselFailFSub,"Fast isel fails on FSub");
STATISTIC(NumFastIselFailMul,"Fast isel fails on Mul");
STATISTIC(NumFastIselFailFMul,"Fast isel fails on FMul");
STATISTIC(NumFastIselFailUDiv,"Fast isel fails on UDiv");
STATISTIC(NumFastIselFailSDiv,"Fast isel fails on SDiv");
STATISTIC(NumFastIselFailFDiv,"Fast isel fails on FDiv");
STATISTIC(NumFastIselFailURem,"Fast isel fails on URem");
STATISTIC(NumFastIselFailSRem,"Fast isel fails on SRem");
STATISTIC(NumFastIselFailFRem,"Fast isel fails on FRem");

  // Logical operators...
STATISTIC(NumFastIselFailAnd,"Fast isel fails on And");
STATISTIC(NumFastIselFailOr,"Fast isel fails on Or");
STATISTIC(NumFastIselFailXor,"Fast isel fails on Xor");

  // Memory instructions...
STATISTIC(NumFastIselFailAlloca,"Fast isel fails on Alloca");
STATISTIC(NumFastIselFailLoad,"Fast isel fails on Load");
STATISTIC(NumFastIselFailStore,"Fast isel fails on Store");
STATISTIC(NumFastIselFailAtomicCmpXchg,"Fast isel fails on AtomicCmpXchg");
STATISTIC(NumFastIselFailAtomicRMW,"Fast isel fails on AtomicRWM");
STATISTIC(NumFastIselFailFence,"Fast isel fails on Frence");
STATISTIC(NumFastIselFailGetElementPtr,"Fast isel fails on GetElementPtr");

  // Convert instructions...
STATISTIC(NumFastIselFailTrunc,"Fast isel fails on Trunc");
STATISTIC(NumFastIselFailZExt,"Fast isel fails on ZExt");
STATISTIC(NumFastIselFailSExt,"Fast isel fails on SExt");
STATISTIC(NumFastIselFailFPTrunc,"Fast isel fails on FPTrunc");
STATISTIC(NumFastIselFailFPExt,"Fast isel fails on FPExt");
STATISTIC(NumFastIselFailFPToUI,"Fast isel fails on FPToUI");
STATISTIC(NumFastIselFailFPToSI,"Fast isel fails on FPToSI");
STATISTIC(NumFastIselFailUIToFP,"Fast isel fails on UIToFP");
STATISTIC(NumFastIselFailSIToFP,"Fast isel fails on SIToFP");
STATISTIC(NumFastIselFailIntToPtr,"Fast isel fails on IntToPtr");
STATISTIC(NumFastIselFailPtrToInt,"Fast isel fails on PtrToInt");
STATISTIC(NumFastIselFailBitCast,"Fast isel fails on BitCast");

  // Other instructions...
STATISTIC(NumFastIselFailICmp,"Fast isel fails on ICmp");
STATISTIC(NumFastIselFailFCmp,"Fast isel fails on FCmp");
STATISTIC(NumFastIselFailPHI,"Fast isel fails on PHI");
STATISTIC(NumFastIselFailSelect,"Fast isel fails on Select");
STATISTIC(NumFastIselFailCall,"Fast isel fails on Call");
STATISTIC(NumFastIselFailShl,"Fast isel fails on Shl");
STATISTIC(NumFastIselFailLShr,"Fast isel fails on LShr");
STATISTIC(NumFastIselFailAShr,"Fast isel fails on AShr");
STATISTIC(NumFastIselFailVAArg,"Fast isel fails on VAArg");
STATISTIC(NumFastIselFailExtractElement,"Fast isel fails on ExtractElement");
STATISTIC(NumFastIselFailInsertElement,"Fast isel fails on InsertElement");
STATISTIC(NumFastIselFailShuffleVector,"Fast isel fails on ShuffleVector");
STATISTIC(NumFastIselFailExtractValue,"Fast isel fails on ExtractValue");
STATISTIC(NumFastIselFailInsertValue,"Fast isel fails on InsertValue");
STATISTIC(NumFastIselFailLandingPad,"Fast isel fails on LandingPad");

// Intrinsic instructions...
STATISTIC(NumFastIselFailIntrinsicCall, "Fast isel fails on Intrinsic call");
STATISTIC(NumFastIselFailSAddWithOverflow,
          "Fast isel fails on sadd.with.overflow");
STATISTIC(NumFastIselFailUAddWithOverflow,
          "Fast isel fails on uadd.with.overflow");
STATISTIC(NumFastIselFailSSubWithOverflow,
          "Fast isel fails on ssub.with.overflow");
STATISTIC(NumFastIselFailUSubWithOverflow,
          "Fast isel fails on usub.with.overflow");
STATISTIC(NumFastIselFailSMulWithOverflow,
          "Fast isel fails on smul.with.overflow");
STATISTIC(NumFastIselFailUMulWithOverflow,
          "Fast isel fails on umul.with.overflow");
STATISTIC(NumFastIselFailFrameaddress, "Fast isel fails on Frameaddress");
STATISTIC(NumFastIselFailSqrt, "Fast isel fails on sqrt call");
STATISTIC(NumFastIselFailStackMap, "Fast isel fails on StackMap call");
STATISTIC(NumFastIselFailPatchPoint, "Fast isel fails on PatchPoint call");
#endif

static cl::opt<bool>
EnableFastISelVerbose("fast-isel-verbose", cl::Hidden,
          cl::desc("Enable verbose messages in the \"fast\" "
                   "instruction selector"));
static cl::opt<int> EnableFastISelAbort(
    "fast-isel-abort", cl::Hidden,
    cl::desc("Enable abort calls when \"fast\" instruction selection "
             "fails to lower an instruction: 0 disable the abort, 1 will "
             "abort but for args, calls and terminators, 2 will also "
             "abort for argument lowering, and 3 will never fallback "
             "to SelectionDAG."));

static cl::opt<bool>
UseMBPI("use-mbpi",
        cl::desc("use Machine Branch Probability Info"),
        cl::init(true), cl::Hidden);

#ifndef NDEBUG
static cl::opt<std::string>
FilterDAGBasicBlockName("filter-view-dags", cl::Hidden,
                        cl::desc("Only display the basic block whose name "
                                 "matches this for all view-*-dags options"));
static cl::opt<bool>
ViewDAGCombine1("view-dag-combine1-dags", cl::Hidden,
          cl::desc("Pop up a window to show dags before the first "
                   "dag combine pass"));
static cl::opt<bool>
ViewLegalizeTypesDAGs("view-legalize-types-dags", cl::Hidden,
          cl::desc("Pop up a window to show dags before legalize types"));
static cl::opt<bool>
ViewLegalizeDAGs("view-legalize-dags", cl::Hidden,
          cl::desc("Pop up a window to show dags before legalize"));
static cl::opt<bool>
ViewDAGCombine2("view-dag-combine2-dags", cl::Hidden,
          cl::desc("Pop up a window to show dags before the second "
                   "dag combine pass"));
static cl::opt<bool>
ViewDAGCombineLT("view-dag-combine-lt-dags", cl::Hidden,
          cl::desc("Pop up a window to show dags before the post legalize types"
                   " dag combine pass"));
static cl::opt<bool>
ViewISelDAGs("view-isel-dags", cl::Hidden,
          cl::desc("Pop up a window to show isel dags as they are selected"));
static cl::opt<bool>
ViewSchedDAGs("view-sched-dags", cl::Hidden,
          cl::desc("Pop up a window to show sched dags as they are processed"));
static cl::opt<bool>
ViewSUnitDAGs("view-sunit-dags", cl::Hidden,
      cl::desc("Pop up a window to show SUnit dags after they are processed"));
#else
static const bool ViewDAGCombine1 = false,
                  ViewLegalizeTypesDAGs = false, ViewLegalizeDAGs = false,
                  ViewDAGCombine2 = false,
                  ViewDAGCombineLT = false,
                  ViewISelDAGs = false, ViewSchedDAGs = false,
                  ViewSUnitDAGs = false;
#endif

//===---------------------------------------------------------------------===//
///
/// RegisterScheduler class - Track the registration of instruction schedulers.
///
//===---------------------------------------------------------------------===//
MachinePassRegistry RegisterScheduler::Registry;

//===---------------------------------------------------------------------===//
///
/// ISHeuristic command line option for instruction schedulers.
///
//===---------------------------------------------------------------------===//
static cl::opt<RegisterScheduler::FunctionPassCtor, false,
               RegisterPassParser<RegisterScheduler> >
ISHeuristic("pre-RA-sched",
            cl::init(&createDefaultScheduler), cl::Hidden,
            cl::desc("Instruction schedulers available (before register"
                     " allocation):"));

static RegisterScheduler
defaultListDAGScheduler("default", "Best scheduler for the target",
                        createDefaultScheduler);

namespace llvm {
  //===--------------------------------------------------------------------===//
  /// \brief This class is used by SelectionDAGISel to temporarily override
  /// the optimization level on a per-function basis.
  class OptLevelChanger {
    SelectionDAGISel &IS;
    CodeGenOpt::Level SavedOptLevel;
    bool SavedFastISel;

  public:
    OptLevelChanger(SelectionDAGISel &ISel,
                    CodeGenOpt::Level NewOptLevel) : IS(ISel) {
      SavedOptLevel = IS.OptLevel;
      if (NewOptLevel == SavedOptLevel)
        return;
      IS.OptLevel = NewOptLevel;
      IS.TM.setOptLevel(NewOptLevel);
      SavedFastISel = IS.TM.Options.EnableFastISel;
      if (NewOptLevel == CodeGenOpt::None)
        IS.TM.setFastISel(true);
      DEBUG(dbgs() << "\nChanging optimization level for Function "
            << IS.MF->getFunction()->getName() << "\n");
      DEBUG(dbgs() << "\tBefore: -O" << SavedOptLevel
            << " ; After: -O" << NewOptLevel << "\n");
    }

    ~OptLevelChanger() {
      if (IS.OptLevel == SavedOptLevel)
        return;
      DEBUG(dbgs() << "\nRestoring optimization level for Function "
            << IS.MF->getFunction()->getName() << "\n");
      DEBUG(dbgs() << "\tBefore: -O" << IS.OptLevel
            << " ; After: -O" << SavedOptLevel << "\n");
      IS.OptLevel = SavedOptLevel;
      IS.TM.setOptLevel(SavedOptLevel);
      IS.TM.setFastISel(SavedFastISel);
    }
  };

  //===--------------------------------------------------------------------===//
  /// createDefaultScheduler - This creates an instruction scheduler appropriate
  /// for the target.
  ScheduleDAGSDNodes* createDefaultScheduler(SelectionDAGISel *IS,
                                             CodeGenOpt::Level OptLevel) {
    const TargetLowering *TLI = IS->TLI;
    const TargetSubtargetInfo &ST = IS->MF->getSubtarget();

    if (OptLevel == CodeGenOpt::None ||
        (ST.enableMachineScheduler() && ST.enableMachineSchedDefaultSched()) ||
        TLI->getSchedulingPreference() == Sched::Source)
      return createSourceListDAGScheduler(IS, OptLevel);
    if (TLI->getSchedulingPreference() == Sched::RegPressure)
      return createBURRListDAGScheduler(IS, OptLevel);
    if (TLI->getSchedulingPreference() == Sched::Hybrid)
      return createHybridListDAGScheduler(IS, OptLevel);
    if (TLI->getSchedulingPreference() == Sched::VLIW)
      return createVLIWDAGScheduler(IS, OptLevel);
    assert(TLI->getSchedulingPreference() == Sched::ILP &&
           "Unknown sched type!");
    return createILPListDAGScheduler(IS, OptLevel);
  }
}

// EmitInstrWithCustomInserter - This method should be implemented by targets
// that mark instructions with the 'usesCustomInserter' flag.  These
// instructions are special in various ways, which require special support to
// insert.  The specified MachineInstr is created but not inserted into any
// basic blocks, and this method is called to expand it into a sequence of
// instructions, potentially also creating new basic blocks and control flow.
// When new basic blocks are inserted and the edges from MBB to its successors
// are modified, the method should insert pairs of <OldSucc, NewSucc> into the
// DenseMap.
MachineBasicBlock *
TargetLowering::EmitInstrWithCustomInserter(MachineInstr *MI,
                                            MachineBasicBlock *MBB) const {
#ifndef NDEBUG
  dbgs() << "If a target marks an instruction with "
          "'usesCustomInserter', it must implement "
          "TargetLowering::EmitInstrWithCustomInserter!";
#endif
  llvm_unreachable(nullptr);
}

void TargetLowering::AdjustInstrPostInstrSelection(MachineInstr *MI,
                                                   SDNode *Node) const {
  assert(!MI->hasPostISelHook() &&
         "If a target marks an instruction with 'hasPostISelHook', "
         "it must implement TargetLowering::AdjustInstrPostInstrSelection!");
}

//===----------------------------------------------------------------------===//
// SelectionDAGISel code
//===----------------------------------------------------------------------===//

SelectionDAGISel::SelectionDAGISel(TargetMachine &tm,
                                   CodeGenOpt::Level OL) :
  MachineFunctionPass(ID), TM(tm),
  FuncInfo(new FunctionLoweringInfo()),
  CurDAG(new SelectionDAG(tm, OL)),
  SDB(new SelectionDAGBuilder(*CurDAG, *FuncInfo, OL)),
  GFI(),
  OptLevel(OL),
  DAGSize(0) {
    initializeGCModuleInfoPass(*PassRegistry::getPassRegistry());
    initializeAliasAnalysisAnalysisGroup(*PassRegistry::getPassRegistry());
    initializeBranchProbabilityInfoPass(*PassRegistry::getPassRegistry());
    initializeTargetLibraryInfoWrapperPassPass(
        *PassRegistry::getPassRegistry());
  }

SelectionDAGISel::~SelectionDAGISel() {
  delete SDB;
  delete CurDAG;
  delete FuncInfo;
}

void SelectionDAGISel::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<AliasAnalysis>();
  AU.addPreserved<AliasAnalysis>();
  AU.addRequired<GCModuleInfo>();
  AU.addPreserved<GCModuleInfo>();
  AU.addRequired<TargetLibraryInfoWrapperPass>();
  if (UseMBPI && OptLevel != CodeGenOpt::None)
    AU.addRequired<BranchProbabilityInfo>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

/// SplitCriticalSideEffectEdges - Look for critical edges with a PHI value that
/// may trap on it.  In this case we have to split the edge so that the path
/// through the predecessor block that doesn't go to the phi block doesn't
/// execute the possibly trapping instruction.
///
/// This is required for correctness, so it must be done at -O0.
///
static void SplitCriticalSideEffectEdges(Function &Fn, AliasAnalysis *AA) {
  // Loop for blocks with phi nodes.
  for (Function::iterator BB = Fn.begin(), E = Fn.end(); BB != E; ++BB) {
    PHINode *PN = dyn_cast<PHINode>(BB->begin());
    if (!PN) continue;

  ReprocessBlock:
    // For each block with a PHI node, check to see if any of the input values
    // are potentially trapping constant expressions.  Constant expressions are
    // the only potentially trapping value that can occur as the argument to a
    // PHI.
    for (BasicBlock::iterator I = BB->begin(); (PN = dyn_cast<PHINode>(I)); ++I)
      for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
        ConstantExpr *CE = dyn_cast<ConstantExpr>(PN->getIncomingValue(i));
        if (!CE || !CE->canTrap()) continue;

        // The only case we have to worry about is when the edge is critical.
        // Since this block has a PHI Node, we assume it has multiple input
        // edges: check to see if the pred has multiple successors.
        BasicBlock *Pred = PN->getIncomingBlock(i);
        if (Pred->getTerminator()->getNumSuccessors() == 1)
          continue;

        // Okay, we have to split this edge.
        SplitCriticalEdge(
            Pred->getTerminator(), GetSuccessorNumber(Pred, BB),
            CriticalEdgeSplittingOptions(AA).setMergeIdenticalEdges());
        goto ReprocessBlock;
      }
  }
}

bool SelectionDAGISel::runOnMachineFunction(MachineFunction &mf) {
  // Do some sanity-checking on the command-line options.
  assert((!EnableFastISelVerbose || TM.Options.EnableFastISel) &&
         "-fast-isel-verbose requires -fast-isel");
  assert((!EnableFastISelAbort || TM.Options.EnableFastISel) &&
         "-fast-isel-abort > 0 requires -fast-isel");

  const Function &Fn = *mf.getFunction();
  MF = &mf;

  // Reset the target options before resetting the optimization
  // level below.
  // FIXME: This is a horrible hack and should be processed via
  // codegen looking at the optimization level explicitly when
  // it wants to look at it.
  TM.resetTargetOptions(Fn);
  // Reset OptLevel to None for optnone functions.
  CodeGenOpt::Level NewOptLevel = OptLevel;
  if (Fn.hasFnAttribute(Attribute::OptimizeNone))
    NewOptLevel = CodeGenOpt::None;
  OptLevelChanger OLC(*this, NewOptLevel);

  TII = MF->getSubtarget().getInstrInfo();
  TLI = MF->getSubtarget().getTargetLowering();
  RegInfo = &MF->getRegInfo();
  AA = &getAnalysis<AliasAnalysis>();
  LibInfo = &getAnalysis<TargetLibraryInfoWrapperPass>().getTLI();
  GFI = Fn.hasGC() ? &getAnalysis<GCModuleInfo>().getFunctionInfo(Fn) : nullptr;

  DEBUG(dbgs() << "\n\n\n=== " << Fn.getName() << "\n");

  SplitCriticalSideEffectEdges(const_cast<Function&>(Fn), AA);

  CurDAG->init(*MF);
  FuncInfo->set(Fn, *MF, CurDAG);

  if (UseMBPI && OptLevel != CodeGenOpt::None)
    FuncInfo->BPI = &getAnalysis<BranchProbabilityInfo>();
  else
    FuncInfo->BPI = nullptr;

  SDB->init(GFI, *AA, LibInfo);

  MF->setHasInlineAsm(false);

  SelectAllBasicBlocks(Fn);

  // If the first basic block in the function has live ins that need to be
  // copied into vregs, emit the copies into the top of the block before
  // emitting the code for the block.
  MachineBasicBlock *EntryMBB = MF->begin();
  const TargetRegisterInfo &TRI = *MF->getSubtarget().getRegisterInfo();
  RegInfo->EmitLiveInCopies(EntryMBB, TRI, *TII);

  DenseMap<unsigned, unsigned> LiveInMap;
  if (!FuncInfo->ArgDbgValues.empty())
    for (MachineRegisterInfo::livein_iterator LI = RegInfo->livein_begin(),
           E = RegInfo->livein_end(); LI != E; ++LI)
      if (LI->second)
        LiveInMap.insert(std::make_pair(LI->first, LI->second));

  // Insert DBG_VALUE instructions for function arguments to the entry block.
  for (unsigned i = 0, e = FuncInfo->ArgDbgValues.size(); i != e; ++i) {
    MachineInstr *MI = FuncInfo->ArgDbgValues[e-i-1];
    bool hasFI = MI->getOperand(0).isFI();
    unsigned Reg =
        hasFI ? TRI.getFrameRegister(*MF) : MI->getOperand(0).getReg();
    if (TargetRegisterInfo::isPhysicalRegister(Reg))
      EntryMBB->insert(EntryMBB->begin(), MI);
    else {
      MachineInstr *Def = RegInfo->getVRegDef(Reg);
      if (Def) {
        MachineBasicBlock::iterator InsertPos = Def;
        // FIXME: VR def may not be in entry block.
        Def->getParent()->insert(std::next(InsertPos), MI);
      } else
        DEBUG(dbgs() << "Dropping debug info for dead vreg"
              << TargetRegisterInfo::virtReg2Index(Reg) << "\n");
    }

    // If Reg is live-in then update debug info to track its copy in a vreg.
    DenseMap<unsigned, unsigned>::iterator LDI = LiveInMap.find(Reg);
    if (LDI != LiveInMap.end()) {
      assert(!hasFI && "There's no handling of frame pointer updating here yet "
                       "- add if needed");
      MachineInstr *Def = RegInfo->getVRegDef(LDI->second);
      MachineBasicBlock::iterator InsertPos = Def;
      const MDNode *Variable = MI->getDebugVariable();
      const MDNode *Expr = MI->getDebugExpression();
      DebugLoc DL = MI->getDebugLoc();
      bool IsIndirect = MI->isIndirectDebugValue();
      unsigned Offset = IsIndirect ? MI->getOperand(1).getImm() : 0;
      assert(cast<DILocalVariable>(Variable)->isValidLocationForIntrinsic(DL) &&
             "Expected inlined-at fields to agree");
      // Def is never a terminator here, so it is ok to increment InsertPos.
      BuildMI(*EntryMBB, ++InsertPos, DL, TII->get(TargetOpcode::DBG_VALUE),
              IsIndirect, LDI->second, Offset, Variable, Expr);

      // If this vreg is directly copied into an exported register then
      // that COPY instructions also need DBG_VALUE, if it is the only
      // user of LDI->second.
      MachineInstr *CopyUseMI = nullptr;
      for (MachineRegisterInfo::use_instr_iterator
           UI = RegInfo->use_instr_begin(LDI->second),
           E = RegInfo->use_instr_end(); UI != E; ) {
        MachineInstr *UseMI = &*(UI++);
        if (UseMI->isDebugValue()) continue;
        if (UseMI->isCopy() && !CopyUseMI && UseMI->getParent() == EntryMBB) {
          CopyUseMI = UseMI; continue;
        }
        // Otherwise this is another use or second copy use.
        CopyUseMI = nullptr; break;
      }
      if (CopyUseMI) {
        // Use MI's debug location, which describes where Variable was
        // declared, rather than whatever is attached to CopyUseMI.
        MachineInstr *NewMI =
            BuildMI(*MF, DL, TII->get(TargetOpcode::DBG_VALUE), IsIndirect,
                    CopyUseMI->getOperand(0).getReg(), Offset, Variable, Expr);
        MachineBasicBlock::iterator Pos = CopyUseMI;
        EntryMBB->insertAfter(Pos, NewMI);
      }
    }
  }

  // Determine if there are any calls in this machine function.
  MachineFrameInfo *MFI = MF->getFrameInfo();
  for (const auto &MBB : *MF) {
    if (MFI->hasCalls() && MF->hasInlineAsm())
      break;

    for (const auto &MI : MBB) {
      const MCInstrDesc &MCID = TII->get(MI.getOpcode());
      if ((MCID.isCall() && !MCID.isReturn()) ||
          MI.isStackAligningInlineAsm()) {
        MFI->setHasCalls(true);
      }
      if (MI.isInlineAsm()) {
        MF->setHasInlineAsm(true);
      }
    }
  }

  // Determine if there is a call to setjmp in the machine function.
  MF->setExposesReturnsTwice(Fn.callsFunctionThatReturnsTwice());

  // Replace forward-declared registers with the registers containing
  // the desired value.
  MachineRegisterInfo &MRI = MF->getRegInfo();
  for (DenseMap<unsigned, unsigned>::iterator
       I = FuncInfo->RegFixups.begin(), E = FuncInfo->RegFixups.end();
       I != E; ++I) {
    unsigned From = I->first;
    unsigned To = I->second;
    // If To is also scheduled to be replaced, find what its ultimate
    // replacement is.
    for (;;) {
      DenseMap<unsigned, unsigned>::iterator J = FuncInfo->RegFixups.find(To);
      if (J == E) break;
      To = J->second;
    }
    // Make sure the new register has a sufficiently constrained register class.
    if (TargetRegisterInfo::isVirtualRegister(From) &&
        TargetRegisterInfo::isVirtualRegister(To))
      MRI.constrainRegClass(To, MRI.getRegClass(From));
    // Replace it.


    // Replacing one register with another won't touch the kill flags.
    // We need to conservatively clear the kill flags as a kill on the old
    // register might dominate existing uses of the new register.
    if (!MRI.use_empty(To))
      MRI.clearKillFlags(From);
    MRI.replaceRegWith(From, To);
  }

  // Freeze the set of reserved registers now that MachineFrameInfo has been
  // set up. All the information required by getReservedRegs() should be
  // available now.
  MRI.freezeReservedRegs(*MF);

  // Release function-specific state. SDB and CurDAG are already cleared
  // at this point.
  FuncInfo->clear();

  DEBUG(dbgs() << "*** MachineFunction at end of ISel ***\n");
  DEBUG(MF->print(dbgs()));

  return true;
}

void SelectionDAGISel::SelectBasicBlock(BasicBlock::const_iterator Begin,
                                        BasicBlock::const_iterator End,
                                        bool &HadTailCall) {
  // Lower the instructions. If a call is emitted as a tail call, cease emitting
  // nodes for this block.
  for (BasicBlock::const_iterator I = Begin; I != End && !SDB->HasTailCall; ++I)
    SDB->visit(*I);

  // Make sure the root of the DAG is up-to-date.
  CurDAG->setRoot(SDB->getControlRoot());
  HadTailCall = SDB->HasTailCall;
  SDB->clear();

  // Final step, emit the lowered DAG as machine code.
  CodeGenAndEmitDAG();
}

void SelectionDAGISel::ComputeLiveOutVRegInfo() {
  SmallPtrSet<SDNode*, 128> VisitedNodes;
  SmallVector<SDNode*, 128> Worklist;

  Worklist.push_back(CurDAG->getRoot().getNode());

  APInt KnownZero;
  APInt KnownOne;

  do {
    SDNode *N = Worklist.pop_back_val();

    // If we've already seen this node, ignore it.
    if (!VisitedNodes.insert(N).second)
      continue;

    // Otherwise, add all chain operands to the worklist.
    for (const SDValue &Op : N->op_values())
      if (Op.getValueType() == MVT::Other)
        Worklist.push_back(Op.getNode());

    // If this is a CopyToReg with a vreg dest, process it.
    if (N->getOpcode() != ISD::CopyToReg)
      continue;

    unsigned DestReg = cast<RegisterSDNode>(N->getOperand(1))->getReg();
    if (!TargetRegisterInfo::isVirtualRegister(DestReg))
      continue;

    // Ignore non-scalar or non-integer values.
    SDValue Src = N->getOperand(2);
    EVT SrcVT = Src.getValueType();
    if (!SrcVT.isInteger() || SrcVT.isVector())
      continue;

    unsigned NumSignBits = CurDAG->ComputeNumSignBits(Src);
    CurDAG->computeKnownBits(Src, KnownZero, KnownOne);
    FuncInfo->AddLiveOutRegInfo(DestReg, NumSignBits, KnownZero, KnownOne);
  } while (!Worklist.empty());
}

void SelectionDAGISel::CodeGenAndEmitDAG() {
  std::string GroupName;
  if (TimePassesIsEnabled)
    GroupName = "Instruction Selection and Scheduling";
  std::string BlockName;
  int BlockNumber = -1;
  (void)BlockNumber;
  bool MatchFilterBB = false; (void)MatchFilterBB;
#ifndef NDEBUG
  MatchFilterBB = (FilterDAGBasicBlockName.empty() ||
                   FilterDAGBasicBlockName ==
                       FuncInfo->MBB->getBasicBlock()->getName().str());
#endif
#ifdef NDEBUG
  if (ViewDAGCombine1 || ViewLegalizeTypesDAGs || ViewLegalizeDAGs ||
      ViewDAGCombine2 || ViewDAGCombineLT || ViewISelDAGs || ViewSchedDAGs ||
      ViewSUnitDAGs)
#endif
  {
    BlockNumber = FuncInfo->MBB->getNumber();
    BlockName =
        (MF->getName() + ":" + FuncInfo->MBB->getBasicBlock()->getName()).str();
  }
  DEBUG(dbgs() << "Initial selection DAG: BB#" << BlockNumber
        << " '" << BlockName << "'\n"; CurDAG->dump());

  if (ViewDAGCombine1 && MatchFilterBB)
    CurDAG->viewGraph("dag-combine1 input for " + BlockName);

  // Run the DAG combiner in pre-legalize mode.
  {
    NamedRegionTimer T("DAG Combining 1", GroupName, TimePassesIsEnabled);
    CurDAG->Combine(BeforeLegalizeTypes, *AA, OptLevel);
  }

  DEBUG(dbgs() << "Optimized lowered selection DAG: BB#" << BlockNumber
        << " '" << BlockName << "'\n"; CurDAG->dump());

  // Second step, hack on the DAG until it only uses operations and types that
  // the target supports.
  if (ViewLegalizeTypesDAGs && MatchFilterBB)
    CurDAG->viewGraph("legalize-types input for " + BlockName);

  bool Changed;
  {
    NamedRegionTimer T("Type Legalization", GroupName, TimePassesIsEnabled);
    Changed = CurDAG->LegalizeTypes();
  }

  DEBUG(dbgs() << "Type-legalized selection DAG: BB#" << BlockNumber
        << " '" << BlockName << "'\n"; CurDAG->dump());

  CurDAG->NewNodesMustHaveLegalTypes = true;

  if (Changed) {
    if (ViewDAGCombineLT && MatchFilterBB)
      CurDAG->viewGraph("dag-combine-lt input for " + BlockName);

    // Run the DAG combiner in post-type-legalize mode.
    {
      NamedRegionTimer T("DAG Combining after legalize types", GroupName,
                         TimePassesIsEnabled);
      CurDAG->Combine(AfterLegalizeTypes, *AA, OptLevel);
    }

    DEBUG(dbgs() << "Optimized type-legalized selection DAG: BB#" << BlockNumber
          << " '" << BlockName << "'\n"; CurDAG->dump());

  }

  {
    NamedRegionTimer T("Vector Legalization", GroupName, TimePassesIsEnabled);
    Changed = CurDAG->LegalizeVectors();
  }

  if (Changed) {
    {
      NamedRegionTimer T("Type Legalization 2", GroupName, TimePassesIsEnabled);
      CurDAG->LegalizeTypes();
    }

    if (ViewDAGCombineLT && MatchFilterBB)
      CurDAG->viewGraph("dag-combine-lv input for " + BlockName);

    // Run the DAG combiner in post-type-legalize mode.
    {
      NamedRegionTimer T("DAG Combining after legalize vectors", GroupName,
                         TimePassesIsEnabled);
      CurDAG->Combine(AfterLegalizeVectorOps, *AA, OptLevel);
    }

    DEBUG(dbgs() << "Optimized vector-legalized selection DAG: BB#"
          << BlockNumber << " '" << BlockName << "'\n"; CurDAG->dump());
  }

  if (ViewLegalizeDAGs && MatchFilterBB)
    CurDAG->viewGraph("legalize input for " + BlockName);

  {
    NamedRegionTimer T("DAG Legalization", GroupName, TimePassesIsEnabled);
    CurDAG->Legalize();
  }

  DEBUG(dbgs() << "Legalized selection DAG: BB#" << BlockNumber
        << " '" << BlockName << "'\n"; CurDAG->dump());

  if (ViewDAGCombine2 && MatchFilterBB)
    CurDAG->viewGraph("dag-combine2 input for " + BlockName);

  // Run the DAG combiner in post-legalize mode.
  {
    NamedRegionTimer T("DAG Combining 2", GroupName, TimePassesIsEnabled);
    CurDAG->Combine(AfterLegalizeDAG, *AA, OptLevel);
  }

  DEBUG(dbgs() << "Optimized legalized selection DAG: BB#" << BlockNumber
        << " '" << BlockName << "'\n"; CurDAG->dump());

  if (OptLevel != CodeGenOpt::None)
    ComputeLiveOutVRegInfo();

  if (ViewISelDAGs && MatchFilterBB)
    CurDAG->viewGraph("isel input for " + BlockName);

  // Third, instruction select all of the operations to machine code, adding the
  // code to the MachineBasicBlock.
  {
    NamedRegionTimer T("Instruction Selection", GroupName, TimePassesIsEnabled);
    DoInstructionSelection();
  }

  DEBUG(dbgs() << "Selected selection DAG: BB#" << BlockNumber
        << " '" << BlockName << "'\n"; CurDAG->dump());

  if (ViewSchedDAGs && MatchFilterBB)
    CurDAG->viewGraph("scheduler input for " + BlockName);

  // Schedule machine code.
  ScheduleDAGSDNodes *Scheduler = CreateScheduler();
  {
    NamedRegionTimer T("Instruction Scheduling", GroupName,
                       TimePassesIsEnabled);
    Scheduler->Run(CurDAG, FuncInfo->MBB);
  }

  if (ViewSUnitDAGs && MatchFilterBB) Scheduler->viewGraph();

  // Emit machine code to BB.  This can change 'BB' to the last block being
  // inserted into.
  MachineBasicBlock *FirstMBB = FuncInfo->MBB, *LastMBB;
  {
    NamedRegionTimer T("Instruction Creation", GroupName, TimePassesIsEnabled);

    // FuncInfo->InsertPt is passed by reference and set to the end of the
    // scheduled instructions.
    LastMBB = FuncInfo->MBB = Scheduler->EmitSchedule(FuncInfo->InsertPt);
  }

  // If the block was split, make sure we update any references that are used to
  // update PHI nodes later on.
  if (FirstMBB != LastMBB)
    SDB->UpdateSplitBlock(FirstMBB, LastMBB);

  // Free the scheduler state.
  {
    NamedRegionTimer T("Instruction Scheduling Cleanup", GroupName,
                       TimePassesIsEnabled);
    delete Scheduler;
  }

  // Free the SelectionDAG state, now that we're finished with it.
  CurDAG->clear();
}

namespace {
/// ISelUpdater - helper class to handle updates of the instruction selection
/// graph.
class ISelUpdater : public SelectionDAG::DAGUpdateListener {
  SelectionDAG::allnodes_iterator &ISelPosition;
public:
  ISelUpdater(SelectionDAG &DAG, SelectionDAG::allnodes_iterator &isp)
    : SelectionDAG::DAGUpdateListener(DAG), ISelPosition(isp) {}

  /// NodeDeleted - Handle nodes deleted from the graph. If the node being
  /// deleted is the current ISelPosition node, update ISelPosition.
  ///
  void NodeDeleted(SDNode *N, SDNode *E) override {
    if (ISelPosition == SelectionDAG::allnodes_iterator(N))
      ++ISelPosition;
  }
};
} // end anonymous namespace

void SelectionDAGISel::DoInstructionSelection() {
  DEBUG(dbgs() << "===== Instruction selection begins: BB#"
        << FuncInfo->MBB->getNumber()
        << " '" << FuncInfo->MBB->getName() << "'\n");

  PreprocessISelDAG();

  // Select target instructions for the DAG.
  {
    // Number all nodes with a topological order and set DAGSize.
    DAGSize = CurDAG->AssignTopologicalOrder();

    // Create a dummy node (which is not added to allnodes), that adds
    // a reference to the root node, preventing it from being deleted,
    // and tracking any changes of the root.
    HandleSDNode Dummy(CurDAG->getRoot());
    SelectionDAG::allnodes_iterator ISelPosition (CurDAG->getRoot().getNode());
    ++ISelPosition;

    // Make sure that ISelPosition gets properly updated when nodes are deleted
    // in calls made from this function.
    ISelUpdater ISU(*CurDAG, ISelPosition);

    // The AllNodes list is now topological-sorted. Visit the
    // nodes by starting at the end of the list (the root of the
    // graph) and preceding back toward the beginning (the entry
    // node).
    while (ISelPosition != CurDAG->allnodes_begin()) {
      SDNode *Node = --ISelPosition;
      // Skip dead nodes. DAGCombiner is expected to eliminate all dead nodes,
      // but there are currently some corner cases that it misses. Also, this
      // makes it theoretically possible to disable the DAGCombiner.
      if (Node->use_empty())
        continue;

      SDNode *ResNode = Select(Node);

      // FIXME: This is pretty gross.  'Select' should be changed to not return
      // anything at all and this code should be nuked with a tactical strike.

      // If node should not be replaced, continue with the next one.
      if (ResNode == Node || Node->getOpcode() == ISD::DELETED_NODE)
        continue;
      // Replace node.
      if (ResNode) {
        ReplaceUses(Node, ResNode);
      }

      // If after the replacement this node is not used any more,
      // remove this dead node.
      if (Node->use_empty()) // Don't delete EntryToken, etc.
        CurDAG->RemoveDeadNode(Node);
    }

    CurDAG->setRoot(Dummy.getValue());
  }

  DEBUG(dbgs() << "===== Instruction selection ends:\n");

  PostprocessISelDAG();
}

/// PrepareEHLandingPad - Emit an EH_LABEL, set up live-in registers, and
/// do other setup for EH landing-pad blocks.
bool SelectionDAGISel::PrepareEHLandingPad() {
  MachineBasicBlock *MBB = FuncInfo->MBB;

  const TargetRegisterClass *PtrRC =
      TLI->getRegClassFor(TLI->getPointerTy(CurDAG->getDataLayout()));

  // Add a label to mark the beginning of the landing pad.  Deletion of the
  // landing pad can thus be detected via the MachineModuleInfo.
  MCSymbol *Label = MF->getMMI().addLandingPad(MBB);

  // Assign the call site to the landing pad's begin label.
  MF->getMMI().setCallSiteLandingPad(Label, SDB->LPadToCallSiteMap[MBB]);

  const MCInstrDesc &II = TII->get(TargetOpcode::EH_LABEL);
  BuildMI(*MBB, FuncInfo->InsertPt, SDB->getCurDebugLoc(), II)
    .addSym(Label);

  // If this is an MSVC-style personality function, we need to split the landing
  // pad into several BBs.
  const BasicBlock *LLVMBB = MBB->getBasicBlock();
  const LandingPadInst *LPadInst = LLVMBB->getLandingPadInst();
  MF->getMMI().addPersonality(MBB, cast<Function>(LPadInst->getParent()
                                                      ->getParent()
                                                      ->getPersonalityFn()
                                                      ->stripPointerCasts()));
  EHPersonality Personality = MF->getMMI().getPersonalityType();

  if (isMSVCEHPersonality(Personality)) {
    SmallVector<MachineBasicBlock *, 4> ClauseBBs;
    const IntrinsicInst *ActionsCall =
        dyn_cast<IntrinsicInst>(LLVMBB->getFirstInsertionPt());
    // Get all invoke BBs that unwind to this landingpad.
    SmallVector<MachineBasicBlock *, 4> InvokeBBs(MBB->pred_begin(),
                                                  MBB->pred_end());
    if (ActionsCall && ActionsCall->getIntrinsicID() == Intrinsic::eh_actions) {
      // If this is a call to llvm.eh.actions followed by indirectbr, then we've
      // run WinEHPrepare, and we should remove this block from the machine CFG.
      // Mark the targets of the indirectbr as landingpads instead.
      for (const BasicBlock *LLVMSucc : successors(LLVMBB)) {
        MachineBasicBlock *ClauseBB = FuncInfo->MBBMap[LLVMSucc];
        // Add the edge from the invoke to the clause.
        for (MachineBasicBlock *InvokeBB : InvokeBBs)
          InvokeBB->addSuccessor(ClauseBB);

        // Mark the clause as a landing pad or MI passes will delete it.
        ClauseBB->setIsLandingPad();
      }
    }

    // Remove the edge from the invoke to the lpad.
    for (MachineBasicBlock *InvokeBB : InvokeBBs)
      InvokeBB->removeSuccessor(MBB);

    // Don't select instructions for the landingpad.
    return false;
  }

  // Mark exception register as live in.
  if (unsigned Reg = TLI->getExceptionPointerRegister())
    FuncInfo->ExceptionPointerVirtReg = MBB->addLiveIn(Reg, PtrRC);

  // Mark exception selector register as live in.
  if (unsigned Reg = TLI->getExceptionSelectorRegister())
    FuncInfo->ExceptionSelectorVirtReg = MBB->addLiveIn(Reg, PtrRC);

  return true;
}

/// isFoldedOrDeadInstruction - Return true if the specified instruction is
/// side-effect free and is either dead or folded into a generated instruction.
/// Return false if it needs to be emitted.
static bool isFoldedOrDeadInstruction(const Instruction *I,
                                      FunctionLoweringInfo *FuncInfo) {
  return !I->mayWriteToMemory() && // Side-effecting instructions aren't folded.
         !isa<TerminatorInst>(I) && // Terminators aren't folded.
         !isa<DbgInfoIntrinsic>(I) &&  // Debug instructions aren't folded.
         !isa<LandingPadInst>(I) &&    // Landingpad instructions aren't folded.
         !FuncInfo->isExportedInst(I); // Exported instrs must be computed.
}

#ifndef NDEBUG
// Collect per Instruction statistics for fast-isel misses.  Only those
// instructions that cause the bail are accounted for.  It does not account for
// instructions higher in the block.  Thus, summing the per instructions stats
// will not add up to what is reported by NumFastIselFailures.
static void collectFailStats(const Instruction *I) {
  switch (I->getOpcode()) {
  default: assert (0 && "<Invalid operator> ");

  // Terminators
  case Instruction::Ret:         NumFastIselFailRet++; return;
  case Instruction::Br:          NumFastIselFailBr++; return;
  case Instruction::Switch:      NumFastIselFailSwitch++; return;
  case Instruction::IndirectBr:  NumFastIselFailIndirectBr++; return;
  case Instruction::Invoke:      NumFastIselFailInvoke++; return;
  case Instruction::Resume:      NumFastIselFailResume++; return;
  case Instruction::Unreachable: NumFastIselFailUnreachable++; return;

  // Standard binary operators...
  case Instruction::Add:  NumFastIselFailAdd++; return;
  case Instruction::FAdd: NumFastIselFailFAdd++; return;
  case Instruction::Sub:  NumFastIselFailSub++; return;
  case Instruction::FSub: NumFastIselFailFSub++; return;
  case Instruction::Mul:  NumFastIselFailMul++; return;
  case Instruction::FMul: NumFastIselFailFMul++; return;
  case Instruction::UDiv: NumFastIselFailUDiv++; return;
  case Instruction::SDiv: NumFastIselFailSDiv++; return;
  case Instruction::FDiv: NumFastIselFailFDiv++; return;
  case Instruction::URem: NumFastIselFailURem++; return;
  case Instruction::SRem: NumFastIselFailSRem++; return;
  case Instruction::FRem: NumFastIselFailFRem++; return;

  // Logical operators...
  case Instruction::And: NumFastIselFailAnd++; return;
  case Instruction::Or:  NumFastIselFailOr++; return;
  case Instruction::Xor: NumFastIselFailXor++; return;

  // Memory instructions...
  case Instruction::Alloca:        NumFastIselFailAlloca++; return;
  case Instruction::Load:          NumFastIselFailLoad++; return;
  case Instruction::Store:         NumFastIselFailStore++; return;
  case Instruction::AtomicCmpXchg: NumFastIselFailAtomicCmpXchg++; return;
  case Instruction::AtomicRMW:     NumFastIselFailAtomicRMW++; return;
  case Instruction::Fence:         NumFastIselFailFence++; return;
  case Instruction::GetElementPtr: NumFastIselFailGetElementPtr++; return;

  // Convert instructions...
  case Instruction::Trunc:    NumFastIselFailTrunc++; return;
  case Instruction::ZExt:     NumFastIselFailZExt++; return;
  case Instruction::SExt:     NumFastIselFailSExt++; return;
  case Instruction::FPTrunc:  NumFastIselFailFPTrunc++; return;
  case Instruction::FPExt:    NumFastIselFailFPExt++; return;
  case Instruction::FPToUI:   NumFastIselFailFPToUI++; return;
  case Instruction::FPToSI:   NumFastIselFailFPToSI++; return;
  case Instruction::UIToFP:   NumFastIselFailUIToFP++; return;
  case Instruction::SIToFP:   NumFastIselFailSIToFP++; return;
  case Instruction::IntToPtr: NumFastIselFailIntToPtr++; return;
  case Instruction::PtrToInt: NumFastIselFailPtrToInt++; return;
  case Instruction::BitCast:  NumFastIselFailBitCast++; return;

  // Other instructions...
  case Instruction::ICmp:           NumFastIselFailICmp++; return;
  case Instruction::FCmp:           NumFastIselFailFCmp++; return;
  case Instruction::PHI:            NumFastIselFailPHI++; return;
  case Instruction::Select:         NumFastIselFailSelect++; return;
  case Instruction::Call: {
    if (auto const *Intrinsic = dyn_cast<IntrinsicInst>(I)) {
      switch (Intrinsic->getIntrinsicID()) {
      default:
        NumFastIselFailIntrinsicCall++; return;
      case Intrinsic::sadd_with_overflow:
        NumFastIselFailSAddWithOverflow++; return;
      case Intrinsic::uadd_with_overflow:
        NumFastIselFailUAddWithOverflow++; return;
      case Intrinsic::ssub_with_overflow:
        NumFastIselFailSSubWithOverflow++; return;
      case Intrinsic::usub_with_overflow:
        NumFastIselFailUSubWithOverflow++; return;
      case Intrinsic::smul_with_overflow:
        NumFastIselFailSMulWithOverflow++; return;
      case Intrinsic::umul_with_overflow:
        NumFastIselFailUMulWithOverflow++; return;
      case Intrinsic::frameaddress:
        NumFastIselFailFrameaddress++; return;
      case Intrinsic::sqrt:
          NumFastIselFailSqrt++; return;
      case Intrinsic::experimental_stackmap:
        NumFastIselFailStackMap++; return;
      case Intrinsic::experimental_patchpoint_void: // fall-through
      case Intrinsic::experimental_patchpoint_i64:
        NumFastIselFailPatchPoint++; return;
      }
    }
    NumFastIselFailCall++;
    return;
  }
  case Instruction::Shl:            NumFastIselFailShl++; return;
  case Instruction::LShr:           NumFastIselFailLShr++; return;
  case Instruction::AShr:           NumFastIselFailAShr++; return;
  case Instruction::VAArg:          NumFastIselFailVAArg++; return;
  case Instruction::ExtractElement: NumFastIselFailExtractElement++; return;
  case Instruction::InsertElement:  NumFastIselFailInsertElement++; return;
  case Instruction::ShuffleVector:  NumFastIselFailShuffleVector++; return;
  case Instruction::ExtractValue:   NumFastIselFailExtractValue++; return;
  case Instruction::InsertValue:    NumFastIselFailInsertValue++; return;
  case Instruction::LandingPad:     NumFastIselFailLandingPad++; return;
  }
}
#endif

void SelectionDAGISel::SelectAllBasicBlocks(const Function &Fn) {
  // Initialize the Fast-ISel state, if needed.
  FastISel *FastIS = nullptr;
  if (TM.Options.EnableFastISel)
    FastIS = TLI->createFastISel(*FuncInfo, LibInfo);

  // Iterate over all basic blocks in the function.
  ReversePostOrderTraversal<const Function*> RPOT(&Fn);
  for (ReversePostOrderTraversal<const Function*>::rpo_iterator
       I = RPOT.begin(), E = RPOT.end(); I != E; ++I) {
    const BasicBlock *LLVMBB = *I;

    if (OptLevel != CodeGenOpt::None) {
      bool AllPredsVisited = true;
      for (const_pred_iterator PI = pred_begin(LLVMBB), PE = pred_end(LLVMBB);
           PI != PE; ++PI) {
        if (!FuncInfo->VisitedBBs.count(*PI)) {
          AllPredsVisited = false;
          break;
        }
      }

      if (AllPredsVisited) {
        for (BasicBlock::const_iterator I = LLVMBB->begin();
             const PHINode *PN = dyn_cast<PHINode>(I); ++I)
          FuncInfo->ComputePHILiveOutRegInfo(PN);
      } else {
        for (BasicBlock::const_iterator I = LLVMBB->begin();
             const PHINode *PN = dyn_cast<PHINode>(I); ++I)
          FuncInfo->InvalidatePHILiveOutRegInfo(PN);
      }

      FuncInfo->VisitedBBs.insert(LLVMBB);
    }

    BasicBlock::const_iterator const Begin = LLVMBB->getFirstNonPHI();
    BasicBlock::const_iterator const End = LLVMBB->end();
    BasicBlock::const_iterator BI = End;

    FuncInfo->MBB = FuncInfo->MBBMap[LLVMBB];
    FuncInfo->InsertPt = FuncInfo->MBB->getFirstNonPHI();

    // Setup an EH landing-pad block.
    FuncInfo->ExceptionPointerVirtReg = 0;
    FuncInfo->ExceptionSelectorVirtReg = 0;
    if (LLVMBB->isLandingPad())
      if (!PrepareEHLandingPad())
        continue;

    // Before doing SelectionDAG ISel, see if FastISel has been requested.
    if (FastIS) {
      FastIS->startNewBlock();

      // Emit code for any incoming arguments. This must happen before
      // beginning FastISel on the entry block.
      if (LLVMBB == &Fn.getEntryBlock()) {
        ++NumEntryBlocks;

        // Lower any arguments needed in this block if this is the entry block.
        if (!FastIS->lowerArguments()) {
          // Fast isel failed to lower these arguments
          ++NumFastIselFailLowerArguments;
          if (EnableFastISelAbort > 1)
            report_fatal_error("FastISel didn't lower all arguments");

          // Use SelectionDAG argument lowering
          LowerArguments(Fn);
          CurDAG->setRoot(SDB->getControlRoot());
          SDB->clear();
          CodeGenAndEmitDAG();
        }

        // If we inserted any instructions at the beginning, make a note of
        // where they are, so we can be sure to emit subsequent instructions
        // after them.
        if (FuncInfo->InsertPt != FuncInfo->MBB->begin())
          FastIS->setLastLocalValue(std::prev(FuncInfo->InsertPt));
        else
          FastIS->setLastLocalValue(nullptr);
      }

      unsigned NumFastIselRemaining = std::distance(Begin, End);
      // Do FastISel on as many instructions as possible.
      for (; BI != Begin; --BI) {
        const Instruction *Inst = std::prev(BI);

        // If we no longer require this instruction, skip it.
        if (isFoldedOrDeadInstruction(Inst, FuncInfo)) {
          --NumFastIselRemaining;
          continue;
        }

        // Bottom-up: reset the insert pos at the top, after any local-value
        // instructions.
        FastIS->recomputeInsertPt();

        // Try to select the instruction with FastISel.
        if (FastIS->selectInstruction(Inst)) {
          --NumFastIselRemaining;
          ++NumFastIselSuccess;
          // If fast isel succeeded, skip over all the folded instructions, and
          // then see if there is a load right before the selected instructions.
          // Try to fold the load if so.
          const Instruction *BeforeInst = Inst;
          while (BeforeInst != Begin) {
            BeforeInst = std::prev(BasicBlock::const_iterator(BeforeInst));
            if (!isFoldedOrDeadInstruction(BeforeInst, FuncInfo))
              break;
          }
          if (BeforeInst != Inst && isa<LoadInst>(BeforeInst) &&
              BeforeInst->hasOneUse() &&
              FastIS->tryToFoldLoad(cast<LoadInst>(BeforeInst), Inst)) {
            // If we succeeded, don't re-select the load.
            BI = std::next(BasicBlock::const_iterator(BeforeInst));
            --NumFastIselRemaining;
            ++NumFastIselSuccess;
          }
          continue;
        }

#ifndef NDEBUG
        if (EnableFastISelVerbose2)
          collectFailStats(Inst);
#endif

        // Then handle certain instructions as single-LLVM-Instruction blocks.
        if (isa<CallInst>(Inst)) {

          if (EnableFastISelVerbose || EnableFastISelAbort) {
            dbgs() << "FastISel missed call: ";
            Inst->dump();
          }
          if (EnableFastISelAbort > 2)
            // FastISel selector couldn't handle something and bailed.
            // For the purpose of debugging, just abort.
            report_fatal_error("FastISel didn't select the entire block");

          if (!Inst->getType()->isVoidTy() && !Inst->use_empty()) {
            unsigned &R = FuncInfo->ValueMap[Inst];
            if (!R)
              R = FuncInfo->CreateRegs(Inst->getType());
          }

          bool HadTailCall = false;
          MachineBasicBlock::iterator SavedInsertPt = FuncInfo->InsertPt;
          SelectBasicBlock(Inst, BI, HadTailCall);

          // If the call was emitted as a tail call, we're done with the block.
          // We also need to delete any previously emitted instructions.
          if (HadTailCall) {
            FastIS->removeDeadCode(SavedInsertPt, FuncInfo->MBB->end());
            --BI;
            break;
          }

          // Recompute NumFastIselRemaining as Selection DAG instruction
          // selection may have handled the call, input args, etc.
          unsigned RemainingNow = std::distance(Begin, BI);
          NumFastIselFailures += NumFastIselRemaining - RemainingNow;
          NumFastIselRemaining = RemainingNow;
          continue;
        }

        bool ShouldAbort = EnableFastISelAbort;
        if (EnableFastISelVerbose || EnableFastISelAbort) {
          if (isa<TerminatorInst>(Inst)) {
            // Use a different message for terminator misses.
            dbgs() << "FastISel missed terminator: ";
            // Don't abort unless for terminator unless the level is really high
            ShouldAbort = (EnableFastISelAbort > 2);
          } else {
            dbgs() << "FastISel miss: ";
          }
          Inst->dump();
        }
        if (ShouldAbort)
          // FastISel selector couldn't handle something and bailed.
          // For the purpose of debugging, just abort.
          report_fatal_error("FastISel didn't select the entire block");

        NumFastIselFailures += NumFastIselRemaining;
        break;
      }

      FastIS->recomputeInsertPt();
    } else {
      // Lower any arguments needed in this block if this is the entry block.
      if (LLVMBB == &Fn.getEntryBlock()) {
        ++NumEntryBlocks;
        LowerArguments(Fn);
      }
    }

    if (Begin != BI)
      ++NumDAGBlocks;
    else
      ++NumFastIselBlocks;

    if (Begin != BI) {
      // Run SelectionDAG instruction selection on the remainder of the block
      // not handled by FastISel. If FastISel is not run, this is the entire
      // block.
      bool HadTailCall;
      SelectBasicBlock(Begin, BI, HadTailCall);
    }

    FinishBasicBlock();
    FuncInfo->PHINodesToUpdate.clear();
  }

  delete FastIS;
  SDB->clearDanglingDebugInfo();
  SDB->SPDescriptor.resetPerFunctionState();
}

/// Given that the input MI is before a partial terminator sequence TSeq, return
/// true if M + TSeq also a partial terminator sequence.
///
/// A Terminator sequence is a sequence of MachineInstrs which at this point in
/// lowering copy vregs into physical registers, which are then passed into
/// terminator instructors so we can satisfy ABI constraints. A partial
/// terminator sequence is an improper subset of a terminator sequence (i.e. it
/// may be the whole terminator sequence).
static bool MIIsInTerminatorSequence(const MachineInstr *MI) {
  // If we do not have a copy or an implicit def, we return true if and only if
  // MI is a debug value.
  if (!MI->isCopy() && !MI->isImplicitDef())
    // Sometimes DBG_VALUE MI sneak in between the copies from the vregs to the
    // physical registers if there is debug info associated with the terminator
    // of our mbb. We want to include said debug info in our terminator
    // sequence, so we return true in that case.
    return MI->isDebugValue();

  // We have left the terminator sequence if we are not doing one of the
  // following:
  //
  // 1. Copying a vreg into a physical register.
  // 2. Copying a vreg into a vreg.
  // 3. Defining a register via an implicit def.

  // OPI should always be a register definition...
  MachineInstr::const_mop_iterator OPI = MI->operands_begin();
  if (!OPI->isReg() || !OPI->isDef())
    return false;

  // Defining any register via an implicit def is always ok.
  if (MI->isImplicitDef())
    return true;

  // Grab the copy source...
  MachineInstr::const_mop_iterator OPI2 = OPI;
  ++OPI2;
  assert(OPI2 != MI->operands_end()
         && "Should have a copy implying we should have 2 arguments.");

  // Make sure that the copy dest is not a vreg when the copy source is a
  // physical register.
  if (!OPI2->isReg() ||
      (!TargetRegisterInfo::isPhysicalRegister(OPI->getReg()) &&
       TargetRegisterInfo::isPhysicalRegister(OPI2->getReg())))
    return false;

  return true;
}

/// Find the split point at which to splice the end of BB into its success stack
/// protector check machine basic block.
///
/// On many platforms, due to ABI constraints, terminators, even before register
/// allocation, use physical registers. This creates an issue for us since
/// physical registers at this point can not travel across basic
/// blocks. Luckily, selectiondag always moves physical registers into vregs
/// when they enter functions and moves them through a sequence of copies back
/// into the physical registers right before the terminator creating a
/// ``Terminator Sequence''. This function is searching for the beginning of the
/// terminator sequence so that we can ensure that we splice off not just the
/// terminator, but additionally the copies that move the vregs into the
/// physical registers.
static MachineBasicBlock::iterator
FindSplitPointForStackProtector(MachineBasicBlock *BB, DebugLoc DL) {
  MachineBasicBlock::iterator SplitPoint = BB->getFirstTerminator();
  //
  if (SplitPoint == BB->begin())
    return SplitPoint;

  MachineBasicBlock::iterator Start = BB->begin();
  MachineBasicBlock::iterator Previous = SplitPoint;
  --Previous;

  while (MIIsInTerminatorSequence(Previous)) {
    SplitPoint = Previous;
    if (Previous == Start)
      break;
    --Previous;
  }

  return SplitPoint;
}

void
SelectionDAGISel::FinishBasicBlock() {

  DEBUG(dbgs() << "Total amount of phi nodes to update: "
               << FuncInfo->PHINodesToUpdate.size() << "\n";
        for (unsigned i = 0, e = FuncInfo->PHINodesToUpdate.size(); i != e; ++i)
          dbgs() << "Node " << i << " : ("
                 << FuncInfo->PHINodesToUpdate[i].first
                 << ", " << FuncInfo->PHINodesToUpdate[i].second << ")\n");

  // Next, now that we know what the last MBB the LLVM BB expanded is, update
  // PHI nodes in successors.
  for (unsigned i = 0, e = FuncInfo->PHINodesToUpdate.size(); i != e; ++i) {
    MachineInstrBuilder PHI(*MF, FuncInfo->PHINodesToUpdate[i].first);
    assert(PHI->isPHI() &&
           "This is not a machine PHI node that we are updating!");
    if (!FuncInfo->MBB->isSuccessor(PHI->getParent()))
      continue;
    PHI.addReg(FuncInfo->PHINodesToUpdate[i].second).addMBB(FuncInfo->MBB);
  }

  // Handle stack protector.
  if (SDB->SPDescriptor.shouldEmitStackProtector()) {
    MachineBasicBlock *ParentMBB = SDB->SPDescriptor.getParentMBB();
    MachineBasicBlock *SuccessMBB = SDB->SPDescriptor.getSuccessMBB();

    // Find the split point to split the parent mbb. At the same time copy all
    // physical registers used in the tail of parent mbb into virtual registers
    // before the split point and back into physical registers after the split
    // point. This prevents us needing to deal with Live-ins and many other
    // register allocation issues caused by us splitting the parent mbb. The
    // register allocator will clean up said virtual copies later on.
    MachineBasicBlock::iterator SplitPoint =
      FindSplitPointForStackProtector(ParentMBB, SDB->getCurDebugLoc());

    // Splice the terminator of ParentMBB into SuccessMBB.
    SuccessMBB->splice(SuccessMBB->end(), ParentMBB,
                       SplitPoint,
                       ParentMBB->end());

    // Add compare/jump on neq/jump to the parent BB.
    FuncInfo->MBB = ParentMBB;
    FuncInfo->InsertPt = ParentMBB->end();
    SDB->visitSPDescriptorParent(SDB->SPDescriptor, ParentMBB);
    CurDAG->setRoot(SDB->getRoot());
    SDB->clear();
    CodeGenAndEmitDAG();

    // CodeGen Failure MBB if we have not codegened it yet.
    MachineBasicBlock *FailureMBB = SDB->SPDescriptor.getFailureMBB();
    if (!FailureMBB->size()) {
      FuncInfo->MBB = FailureMBB;
      FuncInfo->InsertPt = FailureMBB->end();
      SDB->visitSPDescriptorFailure(SDB->SPDescriptor);
      CurDAG->setRoot(SDB->getRoot());
      SDB->clear();
      CodeGenAndEmitDAG();
    }

    // Clear the Per-BB State.
    SDB->SPDescriptor.resetPerBBState();
  }

  for (unsigned i = 0, e = SDB->BitTestCases.size(); i != e; ++i) {
    // Lower header first, if it wasn't already lowered
    if (!SDB->BitTestCases[i].Emitted) {
      // Set the current basic block to the mbb we wish to insert the code into
      FuncInfo->MBB = SDB->BitTestCases[i].Parent;
      FuncInfo->InsertPt = FuncInfo->MBB->end();
      // Emit the code
      SDB->visitBitTestHeader(SDB->BitTestCases[i], FuncInfo->MBB);
      CurDAG->setRoot(SDB->getRoot());
      SDB->clear();
      CodeGenAndEmitDAG();
    }

    uint32_t UnhandledWeight = 0;
    for (unsigned j = 0, ej = SDB->BitTestCases[i].Cases.size(); j != ej; ++j)
      UnhandledWeight += SDB->BitTestCases[i].Cases[j].ExtraWeight;

    for (unsigned j = 0, ej = SDB->BitTestCases[i].Cases.size(); j != ej; ++j) {
      UnhandledWeight -= SDB->BitTestCases[i].Cases[j].ExtraWeight;
      // Set the current basic block to the mbb we wish to insert the code into
      FuncInfo->MBB = SDB->BitTestCases[i].Cases[j].ThisBB;
      FuncInfo->InsertPt = FuncInfo->MBB->end();
      // Emit the code
      if (j+1 != ej)
        SDB->visitBitTestCase(SDB->BitTestCases[i],
                              SDB->BitTestCases[i].Cases[j+1].ThisBB,
                              UnhandledWeight,
                              SDB->BitTestCases[i].Reg,
                              SDB->BitTestCases[i].Cases[j],
                              FuncInfo->MBB);
      else
        SDB->visitBitTestCase(SDB->BitTestCases[i],
                              SDB->BitTestCases[i].Default,
                              UnhandledWeight,
                              SDB->BitTestCases[i].Reg,
                              SDB->BitTestCases[i].Cases[j],
                              FuncInfo->MBB);


      CurDAG->setRoot(SDB->getRoot());
      SDB->clear();
      CodeGenAndEmitDAG();
    }

    // Update PHI Nodes
    for (unsigned pi = 0, pe = FuncInfo->PHINodesToUpdate.size();
         pi != pe; ++pi) {
      MachineInstrBuilder PHI(*MF, FuncInfo->PHINodesToUpdate[pi].first);
      MachineBasicBlock *PHIBB = PHI->getParent();
      assert(PHI->isPHI() &&
             "This is not a machine PHI node that we are updating!");
      // This is "default" BB. We have two jumps to it. From "header" BB and
      // from last "case" BB.
      if (PHIBB == SDB->BitTestCases[i].Default)
        PHI.addReg(FuncInfo->PHINodesToUpdate[pi].second)
           .addMBB(SDB->BitTestCases[i].Parent)
           .addReg(FuncInfo->PHINodesToUpdate[pi].second)
           .addMBB(SDB->BitTestCases[i].Cases.back().ThisBB);
      // One of "cases" BB.
      for (unsigned j = 0, ej = SDB->BitTestCases[i].Cases.size();
           j != ej; ++j) {
        MachineBasicBlock* cBB = SDB->BitTestCases[i].Cases[j].ThisBB;
        if (cBB->isSuccessor(PHIBB))
          PHI.addReg(FuncInfo->PHINodesToUpdate[pi].second).addMBB(cBB);
      }
    }
  }
  SDB->BitTestCases.clear();

  // If the JumpTable record is filled in, then we need to emit a jump table.
  // Updating the PHI nodes is tricky in this case, since we need to determine
  // whether the PHI is a successor of the range check MBB or the jump table MBB
  for (unsigned i = 0, e = SDB->JTCases.size(); i != e; ++i) {
    // Lower header first, if it wasn't already lowered
    if (!SDB->JTCases[i].first.Emitted) {
      // Set the current basic block to the mbb we wish to insert the code into
      FuncInfo->MBB = SDB->JTCases[i].first.HeaderBB;
      FuncInfo->InsertPt = FuncInfo->MBB->end();
      // Emit the code
      SDB->visitJumpTableHeader(SDB->JTCases[i].second, SDB->JTCases[i].first,
                                FuncInfo->MBB);
      CurDAG->setRoot(SDB->getRoot());
      SDB->clear();
      CodeGenAndEmitDAG();
    }

    // Set the current basic block to the mbb we wish to insert the code into
    FuncInfo->MBB = SDB->JTCases[i].second.MBB;
    FuncInfo->InsertPt = FuncInfo->MBB->end();
    // Emit the code
    SDB->visitJumpTable(SDB->JTCases[i].second);
    CurDAG->setRoot(SDB->getRoot());
    SDB->clear();
    CodeGenAndEmitDAG();

    // Update PHI Nodes
    for (unsigned pi = 0, pe = FuncInfo->PHINodesToUpdate.size();
         pi != pe; ++pi) {
      MachineInstrBuilder PHI(*MF, FuncInfo->PHINodesToUpdate[pi].first);
      MachineBasicBlock *PHIBB = PHI->getParent();
      assert(PHI->isPHI() &&
             "This is not a machine PHI node that we are updating!");
      // "default" BB. We can go there only from header BB.
      if (PHIBB == SDB->JTCases[i].second.Default)
        PHI.addReg(FuncInfo->PHINodesToUpdate[pi].second)
           .addMBB(SDB->JTCases[i].first.HeaderBB);
      // JT BB. Just iterate over successors here
      if (FuncInfo->MBB->isSuccessor(PHIBB))
        PHI.addReg(FuncInfo->PHINodesToUpdate[pi].second).addMBB(FuncInfo->MBB);
    }
  }
  SDB->JTCases.clear();

  // If we generated any switch lowering information, build and codegen any
  // additional DAGs necessary.
  for (unsigned i = 0, e = SDB->SwitchCases.size(); i != e; ++i) {
    // Set the current basic block to the mbb we wish to insert the code into
    FuncInfo->MBB = SDB->SwitchCases[i].ThisBB;
    FuncInfo->InsertPt = FuncInfo->MBB->end();

    // Determine the unique successors.
    SmallVector<MachineBasicBlock *, 2> Succs;
    Succs.push_back(SDB->SwitchCases[i].TrueBB);
    if (SDB->SwitchCases[i].TrueBB != SDB->SwitchCases[i].FalseBB)
      Succs.push_back(SDB->SwitchCases[i].FalseBB);

    // Emit the code. Note that this could result in FuncInfo->MBB being split.
    SDB->visitSwitchCase(SDB->SwitchCases[i], FuncInfo->MBB);
    CurDAG->setRoot(SDB->getRoot());
    SDB->clear();
    CodeGenAndEmitDAG();

    // Remember the last block, now that any splitting is done, for use in
    // populating PHI nodes in successors.
    MachineBasicBlock *ThisBB = FuncInfo->MBB;

    // Handle any PHI nodes in successors of this chunk, as if we were coming
    // from the original BB before switch expansion.  Note that PHI nodes can
    // occur multiple times in PHINodesToUpdate.  We have to be very careful to
    // handle them the right number of times.
    for (unsigned i = 0, e = Succs.size(); i != e; ++i) {
      FuncInfo->MBB = Succs[i];
      FuncInfo->InsertPt = FuncInfo->MBB->end();
      // FuncInfo->MBB may have been removed from the CFG if a branch was
      // constant folded.
      if (ThisBB->isSuccessor(FuncInfo->MBB)) {
        for (MachineBasicBlock::iterator
             MBBI = FuncInfo->MBB->begin(), MBBE = FuncInfo->MBB->end();
             MBBI != MBBE && MBBI->isPHI(); ++MBBI) {
          MachineInstrBuilder PHI(*MF, MBBI);
          // This value for this PHI node is recorded in PHINodesToUpdate.
          for (unsigned pn = 0; ; ++pn) {
            assert(pn != FuncInfo->PHINodesToUpdate.size() &&
                   "Didn't find PHI entry!");
            if (FuncInfo->PHINodesToUpdate[pn].first == PHI) {
              PHI.addReg(FuncInfo->PHINodesToUpdate[pn].second).addMBB(ThisBB);
              break;
            }
          }
        }
      }
    }
  }
  SDB->SwitchCases.clear();
}


/// Create the scheduler. If a specific scheduler was specified
/// via the SchedulerRegistry, use it, otherwise select the
/// one preferred by the target.
///
ScheduleDAGSDNodes *SelectionDAGISel::CreateScheduler() {
  RegisterScheduler::FunctionPassCtor Ctor = RegisterScheduler::getDefault();

  if (!Ctor) {
    Ctor = ISHeuristic;
    RegisterScheduler::setDefault(Ctor);
  }

  return Ctor(this, OptLevel);
}

//===----------------------------------------------------------------------===//
// Helper functions used by the generated instruction selector.
//===----------------------------------------------------------------------===//
// Calls to these methods are generated by tblgen.

/// CheckAndMask - The isel is trying to match something like (and X, 255).  If
/// the dag combiner simplified the 255, we still want to match.  RHS is the
/// actual value in the DAG on the RHS of an AND, and DesiredMaskS is the value
/// specified in the .td file (e.g. 255).
bool SelectionDAGISel::CheckAndMask(SDValue LHS, ConstantSDNode *RHS,
                                    int64_t DesiredMaskS) const {
  const APInt &ActualMask = RHS->getAPIntValue();
  const APInt &DesiredMask = APInt(LHS.getValueSizeInBits(), DesiredMaskS);

  // If the actual mask exactly matches, success!
  if (ActualMask == DesiredMask)
    return true;

  // If the actual AND mask is allowing unallowed bits, this doesn't match.
  if (ActualMask.intersects(~DesiredMask))
    return false;

  // Otherwise, the DAG Combiner may have proven that the value coming in is
  // either already zero or is not demanded.  Check for known zero input bits.
  APInt NeededMask = DesiredMask & ~ActualMask;
  if (CurDAG->MaskedValueIsZero(LHS, NeededMask))
    return true;

  // TODO: check to see if missing bits are just not demanded.

  // Otherwise, this pattern doesn't match.
  return false;
}

/// CheckOrMask - The isel is trying to match something like (or X, 255).  If
/// the dag combiner simplified the 255, we still want to match.  RHS is the
/// actual value in the DAG on the RHS of an OR, and DesiredMaskS is the value
/// specified in the .td file (e.g. 255).
bool SelectionDAGISel::CheckOrMask(SDValue LHS, ConstantSDNode *RHS,
                                   int64_t DesiredMaskS) const {
  const APInt &ActualMask = RHS->getAPIntValue();
  const APInt &DesiredMask = APInt(LHS.getValueSizeInBits(), DesiredMaskS);

  // If the actual mask exactly matches, success!
  if (ActualMask == DesiredMask)
    return true;

  // If the actual AND mask is allowing unallowed bits, this doesn't match.
  if (ActualMask.intersects(~DesiredMask))
    return false;

  // Otherwise, the DAG Combiner may have proven that the value coming in is
  // either already zero or is not demanded.  Check for known zero input bits.
  APInt NeededMask = DesiredMask & ~ActualMask;

  APInt KnownZero, KnownOne;
  CurDAG->computeKnownBits(LHS, KnownZero, KnownOne);

  // If all the missing bits in the or are already known to be set, match!
  if ((NeededMask & KnownOne) == NeededMask)
    return true;

  // TODO: check to see if missing bits are just not demanded.

  // Otherwise, this pattern doesn't match.
  return false;
}

/// SelectInlineAsmMemoryOperands - Calls to this are automatically generated
/// by tblgen.  Others should not call it.
void SelectionDAGISel::
SelectInlineAsmMemoryOperands(std::vector<SDValue> &Ops, SDLoc DL) {
  std::vector<SDValue> InOps;
  std::swap(InOps, Ops);

  Ops.push_back(InOps[InlineAsm::Op_InputChain]); // 0
  Ops.push_back(InOps[InlineAsm::Op_AsmString]);  // 1
  Ops.push_back(InOps[InlineAsm::Op_MDNode]);     // 2, !srcloc
  Ops.push_back(InOps[InlineAsm::Op_ExtraInfo]);  // 3 (SideEffect, AlignStack)

  unsigned i = InlineAsm::Op_FirstOperand, e = InOps.size();
  if (InOps[e-1].getValueType() == MVT::Glue)
    --e;  // Don't process a glue operand if it is here.

  while (i != e) {
    unsigned Flags = cast<ConstantSDNode>(InOps[i])->getZExtValue();
    if (!InlineAsm::isMemKind(Flags)) {
      // Just skip over this operand, copying the operands verbatim.
      Ops.insert(Ops.end(), InOps.begin()+i,
                 InOps.begin()+i+InlineAsm::getNumOperandRegisters(Flags) + 1);
      i += InlineAsm::getNumOperandRegisters(Flags) + 1;
    } else {
      assert(InlineAsm::getNumOperandRegisters(Flags) == 1 &&
             "Memory operand with multiple values?");

      unsigned TiedToOperand;
      if (InlineAsm::isUseOperandTiedToDef(Flags, TiedToOperand)) {
        // We need the constraint ID from the operand this is tied to.
        unsigned CurOp = InlineAsm::Op_FirstOperand;
        Flags = cast<ConstantSDNode>(InOps[CurOp])->getZExtValue();
        for (; TiedToOperand; --TiedToOperand) {
          CurOp += InlineAsm::getNumOperandRegisters(Flags)+1;
          Flags = cast<ConstantSDNode>(InOps[CurOp])->getZExtValue();
        }
      }

      // Otherwise, this is a memory operand.  Ask the target to select it.
      std::vector<SDValue> SelOps;
      if (SelectInlineAsmMemoryOperand(InOps[i+1],
                                       InlineAsm::getMemoryConstraintID(Flags),
                                       SelOps))
        report_fatal_error("Could not match memory address.  Inline asm"
                           " failure!");

      // Add this to the output node.
      unsigned NewFlags =
        InlineAsm::getFlagWord(InlineAsm::Kind_Mem, SelOps.size());
      Ops.push_back(CurDAG->getTargetConstant(NewFlags, DL, MVT::i32));
      Ops.insert(Ops.end(), SelOps.begin(), SelOps.end());
      i += 2;
    }
  }

  // Add the glue input back if present.
  if (e != InOps.size())
    Ops.push_back(InOps.back());
}

/// findGlueUse - Return use of MVT::Glue value produced by the specified
/// SDNode.
///
static SDNode *findGlueUse(SDNode *N) {
  unsigned FlagResNo = N->getNumValues()-1;
  for (SDNode::use_iterator I = N->use_begin(), E = N->use_end(); I != E; ++I) {
    SDUse &Use = I.getUse();
    if (Use.getResNo() == FlagResNo)
      return Use.getUser();
  }
  return nullptr;
}

/// findNonImmUse - Return true if "Use" is a non-immediate use of "Def".
/// This function recursively traverses up the operand chain, ignoring
/// certain nodes.
static bool findNonImmUse(SDNode *Use, SDNode* Def, SDNode *ImmedUse,
                          SDNode *Root, SmallPtrSetImpl<SDNode*> &Visited,
                          bool IgnoreChains) {
  // The NodeID's are given uniques ID's where a node ID is guaranteed to be
  // greater than all of its (recursive) operands.  If we scan to a point where
  // 'use' is smaller than the node we're scanning for, then we know we will
  // never find it.
  //
  // The Use may be -1 (unassigned) if it is a newly allocated node.  This can
  // happen because we scan down to newly selected nodes in the case of glue
  // uses.
  if ((Use->getNodeId() < Def->getNodeId() && Use->getNodeId() != -1))
    return false;

  // Don't revisit nodes if we already scanned it and didn't fail, we know we
  // won't fail if we scan it again.
  if (!Visited.insert(Use).second)
    return false;

  for (const SDValue &Op : Use->op_values()) {
    // Ignore chain uses, they are validated by HandleMergeInputChains.
    if (Op.getValueType() == MVT::Other && IgnoreChains)
      continue;

    SDNode *N = Op.getNode();
    if (N == Def) {
      if (Use == ImmedUse || Use == Root)
        continue;  // We are not looking for immediate use.
      assert(N != Root);
      return true;
    }

    // Traverse up the operand chain.
    if (findNonImmUse(N, Def, ImmedUse, Root, Visited, IgnoreChains))
      return true;
  }
  return false;
}

/// IsProfitableToFold - Returns true if it's profitable to fold the specific
/// operand node N of U during instruction selection that starts at Root.
bool SelectionDAGISel::IsProfitableToFold(SDValue N, SDNode *U,
                                          SDNode *Root) const {
  if (OptLevel == CodeGenOpt::None) return false;
  return N.hasOneUse();
}

/// IsLegalToFold - Returns true if the specific operand node N of
/// U can be folded during instruction selection that starts at Root.
bool SelectionDAGISel::IsLegalToFold(SDValue N, SDNode *U, SDNode *Root,
                                     CodeGenOpt::Level OptLevel,
                                     bool IgnoreChains) {
  if (OptLevel == CodeGenOpt::None) return false;

  // If Root use can somehow reach N through a path that that doesn't contain
  // U then folding N would create a cycle. e.g. In the following
  // diagram, Root can reach N through X. If N is folded into into Root, then
  // X is both a predecessor and a successor of U.
  //
  //          [N*]           //
  //         ^   ^           //
  //        /     \          //
  //      [U*]    [X]?       //
  //        ^     ^          //
  //         \   /           //
  //          \ /            //
  //         [Root*]         //
  //
  // * indicates nodes to be folded together.
  //
  // If Root produces glue, then it gets (even more) interesting. Since it
  // will be "glued" together with its glue use in the scheduler, we need to
  // check if it might reach N.
  //
  //          [N*]           //
  //         ^   ^           //
  //        /     \          //
  //      [U*]    [X]?       //
  //        ^       ^        //
  //         \       \       //
  //          \      |       //
  //         [Root*] |       //
  //          ^      |       //
  //          f      |       //
  //          |      /       //
  //         [Y]    /        //
  //           ^   /         //
  //           f  /          //
  //           | /           //
  //          [GU]           //
  //
  // If GU (glue use) indirectly reaches N (the load), and Root folds N
  // (call it Fold), then X is a predecessor of GU and a successor of
  // Fold. But since Fold and GU are glued together, this will create
  // a cycle in the scheduling graph.

  // If the node has glue, walk down the graph to the "lowest" node in the
  // glueged set.
  EVT VT = Root->getValueType(Root->getNumValues()-1);
  while (VT == MVT::Glue) {
    SDNode *GU = findGlueUse(Root);
    if (!GU)
      break;
    Root = GU;
    VT = Root->getValueType(Root->getNumValues()-1);

    // If our query node has a glue result with a use, we've walked up it.  If
    // the user (which has already been selected) has a chain or indirectly uses
    // the chain, our WalkChainUsers predicate will not consider it.  Because of
    // this, we cannot ignore chains in this predicate.
    IgnoreChains = false;
  }


  SmallPtrSet<SDNode*, 16> Visited;
  return !findNonImmUse(Root, N.getNode(), U, Root, Visited, IgnoreChains);
}

SDNode *SelectionDAGISel::Select_INLINEASM(SDNode *N) {
  SDLoc DL(N);

  std::vector<SDValue> Ops(N->op_begin(), N->op_end());
  SelectInlineAsmMemoryOperands(Ops, DL);

  const EVT VTs[] = {MVT::Other, MVT::Glue};
  SDValue New = CurDAG->getNode(ISD::INLINEASM, DL, VTs, Ops);
  New->setNodeId(-1);
  return New.getNode();
}

SDNode
*SelectionDAGISel::Select_READ_REGISTER(SDNode *Op) {
  SDLoc dl(Op);
  MDNodeSDNode *MD = dyn_cast<MDNodeSDNode>(Op->getOperand(1));
  const MDString *RegStr = dyn_cast<MDString>(MD->getMD()->getOperand(0));
  unsigned Reg =
      TLI->getRegisterByName(RegStr->getString().data(), Op->getValueType(0),
                             *CurDAG);
  SDValue New = CurDAG->getCopyFromReg(
                        Op->getOperand(0), dl, Reg, Op->getValueType(0));
  New->setNodeId(-1);
  return New.getNode();
}

SDNode
*SelectionDAGISel::Select_WRITE_REGISTER(SDNode *Op) {
  SDLoc dl(Op);
  MDNodeSDNode *MD = dyn_cast<MDNodeSDNode>(Op->getOperand(1));
  const MDString *RegStr = dyn_cast<MDString>(MD->getMD()->getOperand(0));
  unsigned Reg = TLI->getRegisterByName(RegStr->getString().data(),
                                        Op->getOperand(2).getValueType(),
                                        *CurDAG);
  SDValue New = CurDAG->getCopyToReg(
                        Op->getOperand(0), dl, Reg, Op->getOperand(2));
  New->setNodeId(-1);
  return New.getNode();
}



SDNode *SelectionDAGISel::Select_UNDEF(SDNode *N) {
  return CurDAG->SelectNodeTo(N, TargetOpcode::IMPLICIT_DEF,N->getValueType(0));
}

/// GetVBR - decode a vbr encoding whose top bit is set.
LLVM_ATTRIBUTE_ALWAYS_INLINE static uint64_t
GetVBR(uint64_t Val, const unsigned char *MatcherTable, unsigned &Idx) {
  assert(Val >= 128 && "Not a VBR");
  Val &= 127;  // Remove first vbr bit.

  unsigned Shift = 7;
  uint64_t NextBits;
  do {
    NextBits = MatcherTable[Idx++];
    Val |= (NextBits&127) << Shift;
    Shift += 7;
  } while (NextBits & 128);

  return Val;
}


/// UpdateChainsAndGlue - When a match is complete, this method updates uses of
/// interior glue and chain results to use the new glue and chain results.
void SelectionDAGISel::
UpdateChainsAndGlue(SDNode *NodeToMatch, SDValue InputChain,
                    const SmallVectorImpl<SDNode*> &ChainNodesMatched,
                    SDValue InputGlue,
                    const SmallVectorImpl<SDNode*> &GlueResultNodesMatched,
                    bool isMorphNodeTo) {
  SmallVector<SDNode*, 4> NowDeadNodes;

  // Now that all the normal results are replaced, we replace the chain and
  // glue results if present.
  if (!ChainNodesMatched.empty()) {
    assert(InputChain.getNode() &&
           "Matched input chains but didn't produce a chain");
    // Loop over all of the nodes we matched that produced a chain result.
    // Replace all the chain results with the final chain we ended up with.
    for (unsigned i = 0, e = ChainNodesMatched.size(); i != e; ++i) {
      SDNode *ChainNode = ChainNodesMatched[i];

      // If this node was already deleted, don't look at it.
      if (ChainNode->getOpcode() == ISD::DELETED_NODE)
        continue;

      // Don't replace the results of the root node if we're doing a
      // MorphNodeTo.
      if (ChainNode == NodeToMatch && isMorphNodeTo)
        continue;

      SDValue ChainVal = SDValue(ChainNode, ChainNode->getNumValues()-1);
      if (ChainVal.getValueType() == MVT::Glue)
        ChainVal = ChainVal.getValue(ChainVal->getNumValues()-2);
      assert(ChainVal.getValueType() == MVT::Other && "Not a chain?");
      CurDAG->ReplaceAllUsesOfValueWith(ChainVal, InputChain);

      // If the node became dead and we haven't already seen it, delete it.
      if (ChainNode->use_empty() &&
          !std::count(NowDeadNodes.begin(), NowDeadNodes.end(), ChainNode))
        NowDeadNodes.push_back(ChainNode);
    }
  }

  // If the result produces glue, update any glue results in the matched
  // pattern with the glue result.
  if (InputGlue.getNode()) {
    // Handle any interior nodes explicitly marked.
    for (unsigned i = 0, e = GlueResultNodesMatched.size(); i != e; ++i) {
      SDNode *FRN = GlueResultNodesMatched[i];

      // If this node was already deleted, don't look at it.
      if (FRN->getOpcode() == ISD::DELETED_NODE)
        continue;

      assert(FRN->getValueType(FRN->getNumValues()-1) == MVT::Glue &&
             "Doesn't have a glue result");
      CurDAG->ReplaceAllUsesOfValueWith(SDValue(FRN, FRN->getNumValues()-1),
                                        InputGlue);

      // If the node became dead and we haven't already seen it, delete it.
      if (FRN->use_empty() &&
          !std::count(NowDeadNodes.begin(), NowDeadNodes.end(), FRN))
        NowDeadNodes.push_back(FRN);
    }
  }

  if (!NowDeadNodes.empty())
    CurDAG->RemoveDeadNodes(NowDeadNodes);

  DEBUG(dbgs() << "ISEL: Match complete!\n");
}

enum ChainResult {
  CR_Simple,
  CR_InducesCycle,
  CR_LeadsToInteriorNode
};

/// WalkChainUsers - Walk down the users of the specified chained node that is
/// part of the pattern we're matching, looking at all of the users we find.
/// This determines whether something is an interior node, whether we have a
/// non-pattern node in between two pattern nodes (which prevent folding because
/// it would induce a cycle) and whether we have a TokenFactor node sandwiched
/// between pattern nodes (in which case the TF becomes part of the pattern).
///
/// The walk we do here is guaranteed to be small because we quickly get down to
/// already selected nodes "below" us.
static ChainResult
WalkChainUsers(const SDNode *ChainedNode,
               SmallVectorImpl<SDNode*> &ChainedNodesInPattern,
               SmallVectorImpl<SDNode*> &InteriorChainedNodes) {
  ChainResult Result = CR_Simple;

  for (SDNode::use_iterator UI = ChainedNode->use_begin(),
         E = ChainedNode->use_end(); UI != E; ++UI) {
    // Make sure the use is of the chain, not some other value we produce.
    if (UI.getUse().getValueType() != MVT::Other) continue;

    SDNode *User = *UI;

    if (User->getOpcode() == ISD::HANDLENODE)  // Root of the graph.
      continue;

    // If we see an already-selected machine node, then we've gone beyond the
    // pattern that we're selecting down into the already selected chunk of the
    // DAG.
    unsigned UserOpcode = User->getOpcode();
    if (User->isMachineOpcode() ||
        UserOpcode == ISD::CopyToReg ||
        UserOpcode == ISD::CopyFromReg ||
        UserOpcode == ISD::INLINEASM ||
        UserOpcode == ISD::EH_LABEL ||
        UserOpcode == ISD::LIFETIME_START ||
        UserOpcode == ISD::LIFETIME_END) {
      // If their node ID got reset to -1 then they've already been selected.
      // Treat them like a MachineOpcode.
      if (User->getNodeId() == -1)
        continue;
    }

    // If we have a TokenFactor, we handle it specially.
    if (User->getOpcode() != ISD::TokenFactor) {
      // If the node isn't a token factor and isn't part of our pattern, then it
      // must be a random chained node in between two nodes we're selecting.
      // This happens when we have something like:
      //   x = load ptr
      //   call
      //   y = x+4
      //   store y -> ptr
      // Because we structurally match the load/store as a read/modify/write,
      // but the call is chained between them.  We cannot fold in this case
      // because it would induce a cycle in the graph.
      if (!std::count(ChainedNodesInPattern.begin(),
                      ChainedNodesInPattern.end(), User))
        return CR_InducesCycle;

      // Otherwise we found a node that is part of our pattern.  For example in:
      //   x = load ptr
      //   y = x+4
      //   store y -> ptr
      // This would happen when we're scanning down from the load and see the
      // store as a user.  Record that there is a use of ChainedNode that is
      // part of the pattern and keep scanning uses.
      Result = CR_LeadsToInteriorNode;
      InteriorChainedNodes.push_back(User);
      continue;
    }

    // If we found a TokenFactor, there are two cases to consider: first if the
    // TokenFactor is just hanging "below" the pattern we're matching (i.e. no
    // uses of the TF are in our pattern) we just want to ignore it.  Second,
    // the TokenFactor can be sandwiched in between two chained nodes, like so:
    //     [Load chain]
    //         ^
    //         |
    //       [Load]
    //       ^    ^
    //       |    \                    DAG's like cheese
    //      /       \                       do you?
    //     /         |
    // [TokenFactor] [Op]
    //     ^          ^
    //     |          |
    //      \        /
    //       \      /
    //       [Store]
    //
    // In this case, the TokenFactor becomes part of our match and we rewrite it
    // as a new TokenFactor.
    //
    // To distinguish these two cases, do a recursive walk down the uses.
    switch (WalkChainUsers(User, ChainedNodesInPattern, InteriorChainedNodes)) {
    case CR_Simple:
      // If the uses of the TokenFactor are just already-selected nodes, ignore
      // it, it is "below" our pattern.
      continue;
    case CR_InducesCycle:
      // If the uses of the TokenFactor lead to nodes that are not part of our
      // pattern that are not selected, folding would turn this into a cycle,
      // bail out now.
      return CR_InducesCycle;
    case CR_LeadsToInteriorNode:
      break;  // Otherwise, keep processing.
    }

    // Okay, we know we're in the interesting interior case.  The TokenFactor
    // is now going to be considered part of the pattern so that we rewrite its
    // uses (it may have uses that are not part of the pattern) with the
    // ultimate chain result of the generated code.  We will also add its chain
    // inputs as inputs to the ultimate TokenFactor we create.
    Result = CR_LeadsToInteriorNode;
    ChainedNodesInPattern.push_back(User);
    InteriorChainedNodes.push_back(User);
    continue;
  }

  return Result;
}

/// HandleMergeInputChains - This implements the OPC_EmitMergeInputChains
/// operation for when the pattern matched at least one node with a chains.  The
/// input vector contains a list of all of the chained nodes that we match.  We
/// must determine if this is a valid thing to cover (i.e. matching it won't
/// induce cycles in the DAG) and if so, creating a TokenFactor node. that will
/// be used as the input node chain for the generated nodes.
static SDValue
HandleMergeInputChains(SmallVectorImpl<SDNode*> &ChainNodesMatched,
                       SelectionDAG *CurDAG) {
  // Walk all of the chained nodes we've matched, recursively scanning down the
  // users of the chain result. This adds any TokenFactor nodes that are caught
  // in between chained nodes to the chained and interior nodes list.
  SmallVector<SDNode*, 3> InteriorChainedNodes;
  for (unsigned i = 0, e = ChainNodesMatched.size(); i != e; ++i) {
    if (WalkChainUsers(ChainNodesMatched[i], ChainNodesMatched,
                       InteriorChainedNodes) == CR_InducesCycle)
      return SDValue(); // Would induce a cycle.
  }

  // Okay, we have walked all the matched nodes and collected TokenFactor nodes
  // that we are interested in.  Form our input TokenFactor node.
  SmallVector<SDValue, 3> InputChains;
  for (unsigned i = 0, e = ChainNodesMatched.size(); i != e; ++i) {
    // Add the input chain of this node to the InputChains list (which will be
    // the operands of the generated TokenFactor) if it's not an interior node.
    SDNode *N = ChainNodesMatched[i];
    if (N->getOpcode() != ISD::TokenFactor) {
      if (std::count(InteriorChainedNodes.begin(),InteriorChainedNodes.end(),N))
        continue;

      // Otherwise, add the input chain.
      SDValue InChain = ChainNodesMatched[i]->getOperand(0);
      assert(InChain.getValueType() == MVT::Other && "Not a chain");
      InputChains.push_back(InChain);
      continue;
    }

    // If we have a token factor, we want to add all inputs of the token factor
    // that are not part of the pattern we're matching.
    for (const SDValue &Op : N->op_values()) {
      if (!std::count(ChainNodesMatched.begin(), ChainNodesMatched.end(),
                      Op.getNode()))
        InputChains.push_back(Op);
    }
  }

  if (InputChains.size() == 1)
    return InputChains[0];
  return CurDAG->getNode(ISD::TokenFactor, SDLoc(ChainNodesMatched[0]),
                         MVT::Other, InputChains);
}

/// MorphNode - Handle morphing a node in place for the selector.
SDNode *SelectionDAGISel::
MorphNode(SDNode *Node, unsigned TargetOpc, SDVTList VTList,
          ArrayRef<SDValue> Ops, unsigned EmitNodeInfo) {
  // It is possible we're using MorphNodeTo to replace a node with no
  // normal results with one that has a normal result (or we could be
  // adding a chain) and the input could have glue and chains as well.
  // In this case we need to shift the operands down.
  // FIXME: This is a horrible hack and broken in obscure cases, no worse
  // than the old isel though.
  int OldGlueResultNo = -1, OldChainResultNo = -1;

  unsigned NTMNumResults = Node->getNumValues();
  if (Node->getValueType(NTMNumResults-1) == MVT::Glue) {
    OldGlueResultNo = NTMNumResults-1;
    if (NTMNumResults != 1 &&
        Node->getValueType(NTMNumResults-2) == MVT::Other)
      OldChainResultNo = NTMNumResults-2;
  } else if (Node->getValueType(NTMNumResults-1) == MVT::Other)
    OldChainResultNo = NTMNumResults-1;

  // Call the underlying SelectionDAG routine to do the transmogrification. Note
  // that this deletes operands of the old node that become dead.
  SDNode *Res = CurDAG->MorphNodeTo(Node, ~TargetOpc, VTList, Ops);

  // MorphNodeTo can operate in two ways: if an existing node with the
  // specified operands exists, it can just return it.  Otherwise, it
  // updates the node in place to have the requested operands.
  if (Res == Node) {
    // If we updated the node in place, reset the node ID.  To the isel,
    // this should be just like a newly allocated machine node.
    Res->setNodeId(-1);
  }

  unsigned ResNumResults = Res->getNumValues();
  // Move the glue if needed.
  if ((EmitNodeInfo & OPFL_GlueOutput) && OldGlueResultNo != -1 &&
      (unsigned)OldGlueResultNo != ResNumResults-1)
    CurDAG->ReplaceAllUsesOfValueWith(SDValue(Node, OldGlueResultNo),
                                      SDValue(Res, ResNumResults-1));

  if ((EmitNodeInfo & OPFL_GlueOutput) != 0)
    --ResNumResults;

  // Move the chain reference if needed.
  if ((EmitNodeInfo & OPFL_Chain) && OldChainResultNo != -1 &&
      (unsigned)OldChainResultNo != ResNumResults-1)
    CurDAG->ReplaceAllUsesOfValueWith(SDValue(Node, OldChainResultNo),
                                      SDValue(Res, ResNumResults-1));

  // Otherwise, no replacement happened because the node already exists. Replace
  // Uses of the old node with the new one.
  if (Res != Node)
    CurDAG->ReplaceAllUsesWith(Node, Res);

  return Res;
}

/// CheckSame - Implements OP_CheckSame.
LLVM_ATTRIBUTE_ALWAYS_INLINE static bool
CheckSame(const unsigned char *MatcherTable, unsigned &MatcherIndex,
          SDValue N,
          const SmallVectorImpl<std::pair<SDValue, SDNode*> > &RecordedNodes) {
  // Accept if it is exactly the same as a previously recorded node.
  unsigned RecNo = MatcherTable[MatcherIndex++];
  assert(RecNo < RecordedNodes.size() && "Invalid CheckSame");
  return N == RecordedNodes[RecNo].first;
}

/// CheckChildSame - Implements OP_CheckChildXSame.
LLVM_ATTRIBUTE_ALWAYS_INLINE static bool
CheckChildSame(const unsigned char *MatcherTable, unsigned &MatcherIndex,
             SDValue N,
             const SmallVectorImpl<std::pair<SDValue, SDNode*> > &RecordedNodes,
             unsigned ChildNo) {
  if (ChildNo >= N.getNumOperands())
    return false;  // Match fails if out of range child #.
  return ::CheckSame(MatcherTable, MatcherIndex, N.getOperand(ChildNo),
                     RecordedNodes);
}

/// CheckPatternPredicate - Implements OP_CheckPatternPredicate.
LLVM_ATTRIBUTE_ALWAYS_INLINE static bool
CheckPatternPredicate(const unsigned char *MatcherTable, unsigned &MatcherIndex,
                      const SelectionDAGISel &SDISel) {
  return SDISel.CheckPatternPredicate(MatcherTable[MatcherIndex++]);
}

/// CheckNodePredicate - Implements OP_CheckNodePredicate.
LLVM_ATTRIBUTE_ALWAYS_INLINE static bool
CheckNodePredicate(const unsigned char *MatcherTable, unsigned &MatcherIndex,
                   const SelectionDAGISel &SDISel, SDNode *N) {
  return SDISel.CheckNodePredicate(N, MatcherTable[MatcherIndex++]);
}

LLVM_ATTRIBUTE_ALWAYS_INLINE static bool
CheckOpcode(const unsigned char *MatcherTable, unsigned &MatcherIndex,
            SDNode *N) {
  uint16_t Opc = MatcherTable[MatcherIndex++];
  Opc |= (unsigned short)MatcherTable[MatcherIndex++] << 8;
  return N->getOpcode() == Opc;
}

LLVM_ATTRIBUTE_ALWAYS_INLINE static bool
CheckType(const unsigned char *MatcherTable, unsigned &MatcherIndex, SDValue N,
          const TargetLowering *TLI, const DataLayout &DL) {
  MVT::SimpleValueType VT = (MVT::SimpleValueType)MatcherTable[MatcherIndex++];
  if (N.getValueType() == VT) return true;

  // Handle the case when VT is iPTR.
  return VT == MVT::iPTR && N.getValueType() == TLI->getPointerTy(DL);
}

LLVM_ATTRIBUTE_ALWAYS_INLINE static bool
CheckChildType(const unsigned char *MatcherTable, unsigned &MatcherIndex,
               SDValue N, const TargetLowering *TLI, const DataLayout &DL,
               unsigned ChildNo) {
  if (ChildNo >= N.getNumOperands())
    return false;  // Match fails if out of range child #.
  return ::CheckType(MatcherTable, MatcherIndex, N.getOperand(ChildNo), TLI,
                     DL);
}

LLVM_ATTRIBUTE_ALWAYS_INLINE static bool
CheckCondCode(const unsigned char *MatcherTable, unsigned &MatcherIndex,
              SDValue N) {
  return cast<CondCodeSDNode>(N)->get() ==
      (ISD::CondCode)MatcherTable[MatcherIndex++];
}

LLVM_ATTRIBUTE_ALWAYS_INLINE static bool
CheckValueType(const unsigned char *MatcherTable, unsigned &MatcherIndex,
               SDValue N, const TargetLowering *TLI, const DataLayout &DL) {
  MVT::SimpleValueType VT = (MVT::SimpleValueType)MatcherTable[MatcherIndex++];
  if (cast<VTSDNode>(N)->getVT() == VT)
    return true;

  // Handle the case when VT is iPTR.
  return VT == MVT::iPTR && cast<VTSDNode>(N)->getVT() == TLI->getPointerTy(DL);
}

LLVM_ATTRIBUTE_ALWAYS_INLINE static bool
CheckInteger(const unsigned char *MatcherTable, unsigned &MatcherIndex,
             SDValue N) {
  int64_t Val = MatcherTable[MatcherIndex++];
  if (Val & 128)
    Val = GetVBR(Val, MatcherTable, MatcherIndex);

  ConstantSDNode *C = dyn_cast<ConstantSDNode>(N);
  return C && C->getSExtValue() == Val;
}

LLVM_ATTRIBUTE_ALWAYS_INLINE static bool
CheckChildInteger(const unsigned char *MatcherTable, unsigned &MatcherIndex,
                  SDValue N, unsigned ChildNo) {
  if (ChildNo >= N.getNumOperands())
    return false;  // Match fails if out of range child #.
  return ::CheckInteger(MatcherTable, MatcherIndex, N.getOperand(ChildNo));
}

LLVM_ATTRIBUTE_ALWAYS_INLINE static bool
CheckAndImm(const unsigned char *MatcherTable, unsigned &MatcherIndex,
            SDValue N, const SelectionDAGISel &SDISel) {
  int64_t Val = MatcherTable[MatcherIndex++];
  if (Val & 128)
    Val = GetVBR(Val, MatcherTable, MatcherIndex);

  if (N->getOpcode() != ISD::AND) return false;

  ConstantSDNode *C = dyn_cast<ConstantSDNode>(N->getOperand(1));
  return C && SDISel.CheckAndMask(N.getOperand(0), C, Val);
}

LLVM_ATTRIBUTE_ALWAYS_INLINE static bool
CheckOrImm(const unsigned char *MatcherTable, unsigned &MatcherIndex,
           SDValue N, const SelectionDAGISel &SDISel) {
  int64_t Val = MatcherTable[MatcherIndex++];
  if (Val & 128)
    Val = GetVBR(Val, MatcherTable, MatcherIndex);

  if (N->getOpcode() != ISD::OR) return false;

  ConstantSDNode *C = dyn_cast<ConstantSDNode>(N->getOperand(1));
  return C && SDISel.CheckOrMask(N.getOperand(0), C, Val);
}

/// IsPredicateKnownToFail - If we know how and can do so without pushing a
/// scope, evaluate the current node.  If the current predicate is known to
/// fail, set Result=true and return anything.  If the current predicate is
/// known to pass, set Result=false and return the MatcherIndex to continue
/// with.  If the current predicate is unknown, set Result=false and return the
/// MatcherIndex to continue with.
static unsigned IsPredicateKnownToFail(const unsigned char *Table,
                                       unsigned Index, SDValue N,
                                       bool &Result,
                                       const SelectionDAGISel &SDISel,
                 SmallVectorImpl<std::pair<SDValue, SDNode*> > &RecordedNodes) {
  switch (Table[Index++]) {
  default:
    Result = false;
    return Index-1;  // Could not evaluate this predicate.
  case SelectionDAGISel::OPC_CheckSame:
    Result = !::CheckSame(Table, Index, N, RecordedNodes);
    return Index;
  case SelectionDAGISel::OPC_CheckChild0Same:
  case SelectionDAGISel::OPC_CheckChild1Same:
  case SelectionDAGISel::OPC_CheckChild2Same:
  case SelectionDAGISel::OPC_CheckChild3Same:
    Result = !::CheckChildSame(Table, Index, N, RecordedNodes,
                        Table[Index-1] - SelectionDAGISel::OPC_CheckChild0Same);
    return Index;
  case SelectionDAGISel::OPC_CheckPatternPredicate:
    Result = !::CheckPatternPredicate(Table, Index, SDISel);
    return Index;
  case SelectionDAGISel::OPC_CheckPredicate:
    Result = !::CheckNodePredicate(Table, Index, SDISel, N.getNode());
    return Index;
  case SelectionDAGISel::OPC_CheckOpcode:
    Result = !::CheckOpcode(Table, Index, N.getNode());
    return Index;
  case SelectionDAGISel::OPC_CheckType:
    Result = !::CheckType(Table, Index, N, SDISel.TLI,
                          SDISel.CurDAG->getDataLayout());
    return Index;
  case SelectionDAGISel::OPC_CheckChild0Type:
  case SelectionDAGISel::OPC_CheckChild1Type:
  case SelectionDAGISel::OPC_CheckChild2Type:
  case SelectionDAGISel::OPC_CheckChild3Type:
  case SelectionDAGISel::OPC_CheckChild4Type:
  case SelectionDAGISel::OPC_CheckChild5Type:
  case SelectionDAGISel::OPC_CheckChild6Type:
  case SelectionDAGISel::OPC_CheckChild7Type:
    Result = !::CheckChildType(
                 Table, Index, N, SDISel.TLI, SDISel.CurDAG->getDataLayout(),
                 Table[Index - 1] - SelectionDAGISel::OPC_CheckChild0Type);
    return Index;
  case SelectionDAGISel::OPC_CheckCondCode:
    Result = !::CheckCondCode(Table, Index, N);
    return Index;
  case SelectionDAGISel::OPC_CheckValueType:
    Result = !::CheckValueType(Table, Index, N, SDISel.TLI,
                               SDISel.CurDAG->getDataLayout());
    return Index;
  case SelectionDAGISel::OPC_CheckInteger:
    Result = !::CheckInteger(Table, Index, N);
    return Index;
  case SelectionDAGISel::OPC_CheckChild0Integer:
  case SelectionDAGISel::OPC_CheckChild1Integer:
  case SelectionDAGISel::OPC_CheckChild2Integer:
  case SelectionDAGISel::OPC_CheckChild3Integer:
  case SelectionDAGISel::OPC_CheckChild4Integer:
    Result = !::CheckChildInteger(Table, Index, N,
                     Table[Index-1] - SelectionDAGISel::OPC_CheckChild0Integer);
    return Index;
  case SelectionDAGISel::OPC_CheckAndImm:
    Result = !::CheckAndImm(Table, Index, N, SDISel);
    return Index;
  case SelectionDAGISel::OPC_CheckOrImm:
    Result = !::CheckOrImm(Table, Index, N, SDISel);
    return Index;
  }
}

namespace {

struct MatchScope {
  /// FailIndex - If this match fails, this is the index to continue with.
  unsigned FailIndex;

  /// NodeStack - The node stack when the scope was formed.
  SmallVector<SDValue, 4> NodeStack;

  /// NumRecordedNodes - The number of recorded nodes when the scope was formed.
  unsigned NumRecordedNodes;

  /// NumMatchedMemRefs - The number of matched memref entries.
  unsigned NumMatchedMemRefs;

  /// InputChain/InputGlue - The current chain/glue
  SDValue InputChain, InputGlue;

  /// HasChainNodesMatched - True if the ChainNodesMatched list is non-empty.
  bool HasChainNodesMatched, HasGlueResultNodesMatched;
};

/// \\brief A DAG update listener to keep the matching state
/// (i.e. RecordedNodes and MatchScope) uptodate if the target is allowed to
/// change the DAG while matching.  X86 addressing mode matcher is an example
/// for this.
class MatchStateUpdater : public SelectionDAG::DAGUpdateListener
{
      SmallVectorImpl<std::pair<SDValue, SDNode*> > &RecordedNodes;
      SmallVectorImpl<MatchScope> &MatchScopes;
public:
  MatchStateUpdater(SelectionDAG &DAG,
                    SmallVectorImpl<std::pair<SDValue, SDNode*> > &RN,
                    SmallVectorImpl<MatchScope> &MS) :
    SelectionDAG::DAGUpdateListener(DAG),
    RecordedNodes(RN), MatchScopes(MS) { }

  void NodeDeleted(SDNode *N, SDNode *E) override {
    // Some early-returns here to avoid the search if we deleted the node or
    // if the update comes from MorphNodeTo (MorphNodeTo is the last thing we
    // do, so it's unnecessary to update matching state at that point).
    // Neither of these can occur currently because we only install this
    // update listener during matching a complex patterns.
    if (!E || E->isMachineOpcode())
      return;
    // Performing linear search here does not matter because we almost never
    // run this code.  You'd have to have a CSE during complex pattern
    // matching.
    for (auto &I : RecordedNodes)
      if (I.first.getNode() == N)
        I.first.setNode(E);

    for (auto &I : MatchScopes)
      for (auto &J : I.NodeStack)
        if (J.getNode() == N)
          J.setNode(E);
  }
};
}

SDNode *SelectionDAGISel::
SelectCodeCommon(SDNode *NodeToMatch, const unsigned char *MatcherTable,
                 unsigned TableSize) {
  // FIXME: Should these even be selected?  Handle these cases in the caller?
  switch (NodeToMatch->getOpcode()) {
  default:
    break;
  case ISD::EntryToken:       // These nodes remain the same.
  case ISD::BasicBlock:
  case ISD::Register:
  case ISD::RegisterMask:
  case ISD::HANDLENODE:
  case ISD::MDNODE_SDNODE:
  case ISD::TargetConstant:
  case ISD::TargetConstantFP:
  case ISD::TargetConstantPool:
  case ISD::TargetFrameIndex:
  case ISD::TargetExternalSymbol:
  case ISD::MCSymbol:
  case ISD::TargetBlockAddress:
  case ISD::TargetJumpTable:
  case ISD::TargetGlobalTLSAddress:
  case ISD::TargetGlobalAddress:
  case ISD::TokenFactor:
  case ISD::CopyFromReg:
  case ISD::CopyToReg:
  case ISD::EH_LABEL:
  case ISD::LIFETIME_START:
  case ISD::LIFETIME_END:
    NodeToMatch->setNodeId(-1); // Mark selected.
    return nullptr;
  case ISD::AssertSext:
  case ISD::AssertZext:
    CurDAG->ReplaceAllUsesOfValueWith(SDValue(NodeToMatch, 0),
                                      NodeToMatch->getOperand(0));
    return nullptr;
  case ISD::INLINEASM: return Select_INLINEASM(NodeToMatch);
  case ISD::READ_REGISTER: return Select_READ_REGISTER(NodeToMatch);
  case ISD::WRITE_REGISTER: return Select_WRITE_REGISTER(NodeToMatch);
  case ISD::UNDEF:     return Select_UNDEF(NodeToMatch);
  }

  assert(!NodeToMatch->isMachineOpcode() && "Node already selected!");

  // Set up the node stack with NodeToMatch as the only node on the stack.
  SmallVector<SDValue, 8> NodeStack;
  SDValue N = SDValue(NodeToMatch, 0);
  NodeStack.push_back(N);

  // MatchScopes - Scopes used when matching, if a match failure happens, this
  // indicates where to continue checking.
  SmallVector<MatchScope, 8> MatchScopes;

  // RecordedNodes - This is the set of nodes that have been recorded by the
  // state machine.  The second value is the parent of the node, or null if the
  // root is recorded.
  SmallVector<std::pair<SDValue, SDNode*>, 8> RecordedNodes;

  // MatchedMemRefs - This is the set of MemRef's we've seen in the input
  // pattern.
  SmallVector<MachineMemOperand*, 2> MatchedMemRefs;

  // These are the current input chain and glue for use when generating nodes.
  // Various Emit operations change these.  For example, emitting a copytoreg
  // uses and updates these.
  SDValue InputChain, InputGlue;

  // ChainNodesMatched - If a pattern matches nodes that have input/output
  // chains, the OPC_EmitMergeInputChains operation is emitted which indicates
  // which ones they are.  The result is captured into this list so that we can
  // update the chain results when the pattern is complete.
  SmallVector<SDNode*, 3> ChainNodesMatched;
  SmallVector<SDNode*, 3> GlueResultNodesMatched;

  DEBUG(dbgs() << "ISEL: Starting pattern match on root node: ";
        NodeToMatch->dump(CurDAG);
        dbgs() << '\n');

  // Determine where to start the interpreter.  Normally we start at opcode #0,
  // but if the state machine starts with an OPC_SwitchOpcode, then we
  // accelerate the first lookup (which is guaranteed to be hot) with the
  // OpcodeOffset table.
  unsigned MatcherIndex = 0;

  if (!OpcodeOffset.empty()) {
    // Already computed the OpcodeOffset table, just index into it.
    if (N.getOpcode() < OpcodeOffset.size())
      MatcherIndex = OpcodeOffset[N.getOpcode()];
    DEBUG(dbgs() << "  Initial Opcode index to " << MatcherIndex << "\n");

  } else if (MatcherTable[0] == OPC_SwitchOpcode) {
    // Otherwise, the table isn't computed, but the state machine does start
    // with an OPC_SwitchOpcode instruction.  Populate the table now, since this
    // is the first time we're selecting an instruction.
    unsigned Idx = 1;
    while (1) {
      // Get the size of this case.
      unsigned CaseSize = MatcherTable[Idx++];
      if (CaseSize & 128)
        CaseSize = GetVBR(CaseSize, MatcherTable, Idx);
      if (CaseSize == 0) break;

      // Get the opcode, add the index to the table.
      uint16_t Opc = MatcherTable[Idx++];
      Opc |= (unsigned short)MatcherTable[Idx++] << 8;
      if (Opc >= OpcodeOffset.size())
        OpcodeOffset.resize((Opc+1)*2);
      OpcodeOffset[Opc] = Idx;
      Idx += CaseSize;
    }

    // Okay, do the lookup for the first opcode.
    if (N.getOpcode() < OpcodeOffset.size())
      MatcherIndex = OpcodeOffset[N.getOpcode()];
  }

  while (1) {
    assert(MatcherIndex < TableSize && "Invalid index");
#ifndef NDEBUG
    unsigned CurrentOpcodeIndex = MatcherIndex;
#endif
    BuiltinOpcodes Opcode = (BuiltinOpcodes)MatcherTable[MatcherIndex++];
    switch (Opcode) {
    case OPC_Scope: {
      // Okay, the semantics of this operation are that we should push a scope
      // then evaluate the first child.  However, pushing a scope only to have
      // the first check fail (which then pops it) is inefficient.  If we can
      // determine immediately that the first check (or first several) will
      // immediately fail, don't even bother pushing a scope for them.
      unsigned FailIndex;

      while (1) {
        unsigned NumToSkip = MatcherTable[MatcherIndex++];
        if (NumToSkip & 128)
          NumToSkip = GetVBR(NumToSkip, MatcherTable, MatcherIndex);
        // Found the end of the scope with no match.
        if (NumToSkip == 0) {
          FailIndex = 0;
          break;
        }

        FailIndex = MatcherIndex+NumToSkip;

        unsigned MatcherIndexOfPredicate = MatcherIndex;
        (void)MatcherIndexOfPredicate; // silence warning.

        // If we can't evaluate this predicate without pushing a scope (e.g. if
        // it is a 'MoveParent') or if the predicate succeeds on this node, we
        // push the scope and evaluate the full predicate chain.
        bool Result;
        MatcherIndex = IsPredicateKnownToFail(MatcherTable, MatcherIndex, N,
                                              Result, *this, RecordedNodes);
        if (!Result)
          break;

        DEBUG(dbgs() << "  Skipped scope entry (due to false predicate) at "
                     << "index " << MatcherIndexOfPredicate
                     << ", continuing at " << FailIndex << "\n");
        ++NumDAGIselRetries;

        // Otherwise, we know that this case of the Scope is guaranteed to fail,
        // move to the next case.
        MatcherIndex = FailIndex;
      }

      // If the whole scope failed to match, bail.
      if (FailIndex == 0) break;

      // Push a MatchScope which indicates where to go if the first child fails
      // to match.
      MatchScope NewEntry;
      NewEntry.FailIndex = FailIndex;
      NewEntry.NodeStack.append(NodeStack.begin(), NodeStack.end());
      NewEntry.NumRecordedNodes = RecordedNodes.size();
      NewEntry.NumMatchedMemRefs = MatchedMemRefs.size();
      NewEntry.InputChain = InputChain;
      NewEntry.InputGlue = InputGlue;
      NewEntry.HasChainNodesMatched = !ChainNodesMatched.empty();
      NewEntry.HasGlueResultNodesMatched = !GlueResultNodesMatched.empty();
      MatchScopes.push_back(NewEntry);
      continue;
    }
    case OPC_RecordNode: {
      // Remember this node, it may end up being an operand in the pattern.
      SDNode *Parent = nullptr;
      if (NodeStack.size() > 1)
        Parent = NodeStack[NodeStack.size()-2].getNode();
      RecordedNodes.push_back(std::make_pair(N, Parent));
      continue;
    }

    case OPC_RecordChild0: case OPC_RecordChild1:
    case OPC_RecordChild2: case OPC_RecordChild3:
    case OPC_RecordChild4: case OPC_RecordChild5:
    case OPC_RecordChild6: case OPC_RecordChild7: {
      unsigned ChildNo = Opcode-OPC_RecordChild0;
      if (ChildNo >= N.getNumOperands())
        break;  // Match fails if out of range child #.

      RecordedNodes.push_back(std::make_pair(N->getOperand(ChildNo),
                                             N.getNode()));
      continue;
    }
    case OPC_RecordMemRef:
      MatchedMemRefs.push_back(cast<MemSDNode>(N)->getMemOperand());
      continue;

    case OPC_CaptureGlueInput:
      // If the current node has an input glue, capture it in InputGlue.
      if (N->getNumOperands() != 0 &&
          N->getOperand(N->getNumOperands()-1).getValueType() == MVT::Glue)
        InputGlue = N->getOperand(N->getNumOperands()-1);
      continue;

    case OPC_MoveChild: {
      unsigned ChildNo = MatcherTable[MatcherIndex++];
      if (ChildNo >= N.getNumOperands())
        break;  // Match fails if out of range child #.
      N = N.getOperand(ChildNo);
      NodeStack.push_back(N);
      continue;
    }

    case OPC_MoveParent:
      // Pop the current node off the NodeStack.
      NodeStack.pop_back();
      assert(!NodeStack.empty() && "Node stack imbalance!");
      N = NodeStack.back();
      continue;

    case OPC_CheckSame:
      if (!::CheckSame(MatcherTable, MatcherIndex, N, RecordedNodes)) break;
      continue;

    case OPC_CheckChild0Same: case OPC_CheckChild1Same:
    case OPC_CheckChild2Same: case OPC_CheckChild3Same:
      if (!::CheckChildSame(MatcherTable, MatcherIndex, N, RecordedNodes,
                            Opcode-OPC_CheckChild0Same))
        break;
      continue;

    case OPC_CheckPatternPredicate:
      if (!::CheckPatternPredicate(MatcherTable, MatcherIndex, *this)) break;
      continue;
    case OPC_CheckPredicate:
      if (!::CheckNodePredicate(MatcherTable, MatcherIndex, *this,
                                N.getNode()))
        break;
      continue;
    case OPC_CheckComplexPat: {
      unsigned CPNum = MatcherTable[MatcherIndex++];
      unsigned RecNo = MatcherTable[MatcherIndex++];
      assert(RecNo < RecordedNodes.size() && "Invalid CheckComplexPat");

      // If target can modify DAG during matching, keep the matching state
      // consistent.
      std::unique_ptr<MatchStateUpdater> MSU;
      if (ComplexPatternFuncMutatesDAG())
        MSU.reset(new MatchStateUpdater(*CurDAG, RecordedNodes,
                                        MatchScopes));

      if (!CheckComplexPattern(NodeToMatch, RecordedNodes[RecNo].second,
                               RecordedNodes[RecNo].first, CPNum,
                               RecordedNodes))
        break;
      continue;
    }
    case OPC_CheckOpcode:
      if (!::CheckOpcode(MatcherTable, MatcherIndex, N.getNode())) break;
      continue;

    case OPC_CheckType:
      if (!::CheckType(MatcherTable, MatcherIndex, N, TLI,
                       CurDAG->getDataLayout()))
        break;
      continue;

    case OPC_SwitchOpcode: {
      unsigned CurNodeOpcode = N.getOpcode();
      unsigned SwitchStart = MatcherIndex-1; (void)SwitchStart;
      unsigned CaseSize;
      while (1) {
        // Get the size of this case.
        CaseSize = MatcherTable[MatcherIndex++];
        if (CaseSize & 128)
          CaseSize = GetVBR(CaseSize, MatcherTable, MatcherIndex);
        if (CaseSize == 0) break;

        uint16_t Opc = MatcherTable[MatcherIndex++];
        Opc |= (unsigned short)MatcherTable[MatcherIndex++] << 8;

        // If the opcode matches, then we will execute this case.
        if (CurNodeOpcode == Opc)
          break;

        // Otherwise, skip over this case.
        MatcherIndex += CaseSize;
      }

      // If no cases matched, bail out.
      if (CaseSize == 0) break;

      // Otherwise, execute the case we found.
      DEBUG(dbgs() << "  OpcodeSwitch from " << SwitchStart
                   << " to " << MatcherIndex << "\n");
      continue;
    }

    case OPC_SwitchType: {
      MVT CurNodeVT = N.getSimpleValueType();
      unsigned SwitchStart = MatcherIndex-1; (void)SwitchStart;
      unsigned CaseSize;
      while (1) {
        // Get the size of this case.
        CaseSize = MatcherTable[MatcherIndex++];
        if (CaseSize & 128)
          CaseSize = GetVBR(CaseSize, MatcherTable, MatcherIndex);
        if (CaseSize == 0) break;

        MVT CaseVT = (MVT::SimpleValueType)MatcherTable[MatcherIndex++];
        if (CaseVT == MVT::iPTR)
          CaseVT = TLI->getPointerTy(CurDAG->getDataLayout());

        // If the VT matches, then we will execute this case.
        if (CurNodeVT == CaseVT)
          break;

        // Otherwise, skip over this case.
        MatcherIndex += CaseSize;
      }

      // If no cases matched, bail out.
      if (CaseSize == 0) break;

      // Otherwise, execute the case we found.
      DEBUG(dbgs() << "  TypeSwitch[" << EVT(CurNodeVT).getEVTString()
                   << "] from " << SwitchStart << " to " << MatcherIndex<<'\n');
      continue;
    }
    case OPC_CheckChild0Type: case OPC_CheckChild1Type:
    case OPC_CheckChild2Type: case OPC_CheckChild3Type:
    case OPC_CheckChild4Type: case OPC_CheckChild5Type:
    case OPC_CheckChild6Type: case OPC_CheckChild7Type:
      if (!::CheckChildType(MatcherTable, MatcherIndex, N, TLI,
                            CurDAG->getDataLayout(),
                            Opcode - OPC_CheckChild0Type))
        break;
      continue;
    case OPC_CheckCondCode:
      if (!::CheckCondCode(MatcherTable, MatcherIndex, N)) break;
      continue;
    case OPC_CheckValueType:
      if (!::CheckValueType(MatcherTable, MatcherIndex, N, TLI,
                            CurDAG->getDataLayout()))
        break;
      continue;
    case OPC_CheckInteger:
      if (!::CheckInteger(MatcherTable, MatcherIndex, N)) break;
      continue;
    case OPC_CheckChild0Integer: case OPC_CheckChild1Integer:
    case OPC_CheckChild2Integer: case OPC_CheckChild3Integer:
    case OPC_CheckChild4Integer:
      if (!::CheckChildInteger(MatcherTable, MatcherIndex, N,
                               Opcode-OPC_CheckChild0Integer)) break;
      continue;
    case OPC_CheckAndImm:
      if (!::CheckAndImm(MatcherTable, MatcherIndex, N, *this)) break;
      continue;
    case OPC_CheckOrImm:
      if (!::CheckOrImm(MatcherTable, MatcherIndex, N, *this)) break;
      continue;

    case OPC_CheckFoldableChainNode: {
      assert(NodeStack.size() != 1 && "No parent node");
      // Verify that all intermediate nodes between the root and this one have
      // a single use.
      bool HasMultipleUses = false;
      for (unsigned i = 1, e = NodeStack.size()-1; i != e; ++i)
        if (!NodeStack[i].hasOneUse()) {
          HasMultipleUses = true;
          break;
        }
      if (HasMultipleUses) break;

      // Check to see that the target thinks this is profitable to fold and that
      // we can fold it without inducing cycles in the graph.
      if (!IsProfitableToFold(N, NodeStack[NodeStack.size()-2].getNode(),
                              NodeToMatch) ||
          !IsLegalToFold(N, NodeStack[NodeStack.size()-2].getNode(),
                         NodeToMatch, OptLevel,
                         true/*We validate our own chains*/))
        break;

      continue;
    }
    case OPC_EmitInteger: {
      MVT::SimpleValueType VT =
        (MVT::SimpleValueType)MatcherTable[MatcherIndex++];
      int64_t Val = MatcherTable[MatcherIndex++];
      if (Val & 128)
        Val = GetVBR(Val, MatcherTable, MatcherIndex);
      RecordedNodes.push_back(std::pair<SDValue, SDNode*>(
                              CurDAG->getTargetConstant(Val, SDLoc(NodeToMatch),
                                                        VT), nullptr));
      continue;
    }
    case OPC_EmitRegister: {
      MVT::SimpleValueType VT =
        (MVT::SimpleValueType)MatcherTable[MatcherIndex++];
      unsigned RegNo = MatcherTable[MatcherIndex++];
      RecordedNodes.push_back(std::pair<SDValue, SDNode*>(
                              CurDAG->getRegister(RegNo, VT), nullptr));
      continue;
    }
    case OPC_EmitRegister2: {
      // For targets w/ more than 256 register names, the register enum
      // values are stored in two bytes in the matcher table (just like
      // opcodes).
      MVT::SimpleValueType VT =
        (MVT::SimpleValueType)MatcherTable[MatcherIndex++];
      unsigned RegNo = MatcherTable[MatcherIndex++];
      RegNo |= MatcherTable[MatcherIndex++] << 8;
      RecordedNodes.push_back(std::pair<SDValue, SDNode*>(
                              CurDAG->getRegister(RegNo, VT), nullptr));
      continue;
    }

    case OPC_EmitConvertToTarget:  {
      // Convert from IMM/FPIMM to target version.
      unsigned RecNo = MatcherTable[MatcherIndex++];
      assert(RecNo < RecordedNodes.size() && "Invalid EmitConvertToTarget");
      SDValue Imm = RecordedNodes[RecNo].first;

      if (Imm->getOpcode() == ISD::Constant) {
        const ConstantInt *Val=cast<ConstantSDNode>(Imm)->getConstantIntValue();
        Imm = CurDAG->getConstant(*Val, SDLoc(NodeToMatch), Imm.getValueType(),
                                  true);
      } else if (Imm->getOpcode() == ISD::ConstantFP) {
        const ConstantFP *Val=cast<ConstantFPSDNode>(Imm)->getConstantFPValue();
        Imm = CurDAG->getConstantFP(*Val, SDLoc(NodeToMatch),
                                    Imm.getValueType(), true);
      }

      RecordedNodes.push_back(std::make_pair(Imm, RecordedNodes[RecNo].second));
      continue;
    }

    case OPC_EmitMergeInputChains1_0:    // OPC_EmitMergeInputChains, 1, 0
    case OPC_EmitMergeInputChains1_1: {  // OPC_EmitMergeInputChains, 1, 1
      // These are space-optimized forms of OPC_EmitMergeInputChains.
      assert(!InputChain.getNode() &&
             "EmitMergeInputChains should be the first chain producing node");
      assert(ChainNodesMatched.empty() &&
             "Should only have one EmitMergeInputChains per match");

      // Read all of the chained nodes.
      unsigned RecNo = Opcode == OPC_EmitMergeInputChains1_1;
      assert(RecNo < RecordedNodes.size() && "Invalid EmitMergeInputChains");
      ChainNodesMatched.push_back(RecordedNodes[RecNo].first.getNode());

      // FIXME: What if other value results of the node have uses not matched
      // by this pattern?
      if (ChainNodesMatched.back() != NodeToMatch &&
          !RecordedNodes[RecNo].first.hasOneUse()) {
        ChainNodesMatched.clear();
        break;
      }

      // Merge the input chains if they are not intra-pattern references.
      InputChain = HandleMergeInputChains(ChainNodesMatched, CurDAG);

      if (!InputChain.getNode())
        break;  // Failed to merge.
      continue;
    }

    case OPC_EmitMergeInputChains: {
      assert(!InputChain.getNode() &&
             "EmitMergeInputChains should be the first chain producing node");
      // This node gets a list of nodes we matched in the input that have
      // chains.  We want to token factor all of the input chains to these nodes
      // together.  However, if any of the input chains is actually one of the
      // nodes matched in this pattern, then we have an intra-match reference.
      // Ignore these because the newly token factored chain should not refer to
      // the old nodes.
      unsigned NumChains = MatcherTable[MatcherIndex++];
      assert(NumChains != 0 && "Can't TF zero chains");

      assert(ChainNodesMatched.empty() &&
             "Should only have one EmitMergeInputChains per match");

      // Read all of the chained nodes.
      for (unsigned i = 0; i != NumChains; ++i) {
        unsigned RecNo = MatcherTable[MatcherIndex++];
        assert(RecNo < RecordedNodes.size() && "Invalid EmitMergeInputChains");
        ChainNodesMatched.push_back(RecordedNodes[RecNo].first.getNode());

        // FIXME: What if other value results of the node have uses not matched
        // by this pattern?
        if (ChainNodesMatched.back() != NodeToMatch &&
            !RecordedNodes[RecNo].first.hasOneUse()) {
          ChainNodesMatched.clear();
          break;
        }
      }

      // If the inner loop broke out, the match fails.
      if (ChainNodesMatched.empty())
        break;

      // Merge the input chains if they are not intra-pattern references.
      InputChain = HandleMergeInputChains(ChainNodesMatched, CurDAG);

      if (!InputChain.getNode())
        break;  // Failed to merge.

      continue;
    }

    case OPC_EmitCopyToReg: {
      unsigned RecNo = MatcherTable[MatcherIndex++];
      assert(RecNo < RecordedNodes.size() && "Invalid EmitCopyToReg");
      unsigned DestPhysReg = MatcherTable[MatcherIndex++];

      if (!InputChain.getNode())
        InputChain = CurDAG->getEntryNode();

      InputChain = CurDAG->getCopyToReg(InputChain, SDLoc(NodeToMatch),
                                        DestPhysReg, RecordedNodes[RecNo].first,
                                        InputGlue);

      InputGlue = InputChain.getValue(1);
      continue;
    }

    case OPC_EmitNodeXForm: {
      unsigned XFormNo = MatcherTable[MatcherIndex++];
      unsigned RecNo = MatcherTable[MatcherIndex++];
      assert(RecNo < RecordedNodes.size() && "Invalid EmitNodeXForm");
      SDValue Res = RunSDNodeXForm(RecordedNodes[RecNo].first, XFormNo);
      RecordedNodes.push_back(std::pair<SDValue,SDNode*>(Res, nullptr));
      continue;
    }

    case OPC_EmitNode:
    case OPC_MorphNodeTo: {
      uint16_t TargetOpc = MatcherTable[MatcherIndex++];
      TargetOpc |= (unsigned short)MatcherTable[MatcherIndex++] << 8;
      unsigned EmitNodeInfo = MatcherTable[MatcherIndex++];
      // Get the result VT list.
      unsigned NumVTs = MatcherTable[MatcherIndex++];
      SmallVector<EVT, 4> VTs;
      for (unsigned i = 0; i != NumVTs; ++i) {
        MVT::SimpleValueType VT =
          (MVT::SimpleValueType)MatcherTable[MatcherIndex++];
        if (VT == MVT::iPTR)
          VT = TLI->getPointerTy(CurDAG->getDataLayout()).SimpleTy;
        VTs.push_back(VT);
      }

      if (EmitNodeInfo & OPFL_Chain)
        VTs.push_back(MVT::Other);
      if (EmitNodeInfo & OPFL_GlueOutput)
        VTs.push_back(MVT::Glue);

      // This is hot code, so optimize the two most common cases of 1 and 2
      // results.
      SDVTList VTList;
      if (VTs.size() == 1)
        VTList = CurDAG->getVTList(VTs[0]);
      else if (VTs.size() == 2)
        VTList = CurDAG->getVTList(VTs[0], VTs[1]);
      else
        VTList = CurDAG->getVTList(VTs);

      // Get the operand list.
      unsigned NumOps = MatcherTable[MatcherIndex++];
      SmallVector<SDValue, 8> Ops;
      for (unsigned i = 0; i != NumOps; ++i) {
        unsigned RecNo = MatcherTable[MatcherIndex++];
        if (RecNo & 128)
          RecNo = GetVBR(RecNo, MatcherTable, MatcherIndex);

        assert(RecNo < RecordedNodes.size() && "Invalid EmitNode");
        Ops.push_back(RecordedNodes[RecNo].first);
      }

      // If there are variadic operands to add, handle them now.
      if (EmitNodeInfo & OPFL_VariadicInfo) {
        // Determine the start index to copy from.
        unsigned FirstOpToCopy = getNumFixedFromVariadicInfo(EmitNodeInfo);
        FirstOpToCopy += (EmitNodeInfo & OPFL_Chain) ? 1 : 0;
        assert(NodeToMatch->getNumOperands() >= FirstOpToCopy &&
               "Invalid variadic node");
        // Copy all of the variadic operands, not including a potential glue
        // input.
        for (unsigned i = FirstOpToCopy, e = NodeToMatch->getNumOperands();
             i != e; ++i) {
          SDValue V = NodeToMatch->getOperand(i);
          if (V.getValueType() == MVT::Glue) break;
          Ops.push_back(V);
        }
      }

      // If this has chain/glue inputs, add them.
      if (EmitNodeInfo & OPFL_Chain)
        Ops.push_back(InputChain);
      if ((EmitNodeInfo & OPFL_GlueInput) && InputGlue.getNode() != nullptr)
        Ops.push_back(InputGlue);

      // Create the node.
      SDNode *Res = nullptr;
      if (Opcode != OPC_MorphNodeTo) {
        // If this is a normal EmitNode command, just create the new node and
        // add the results to the RecordedNodes list.
        Res = CurDAG->getMachineNode(TargetOpc, SDLoc(NodeToMatch),
                                     VTList, Ops);

        // Add all the non-glue/non-chain results to the RecordedNodes list.
        for (unsigned i = 0, e = VTs.size(); i != e; ++i) {
          if (VTs[i] == MVT::Other || VTs[i] == MVT::Glue) break;
          RecordedNodes.push_back(std::pair<SDValue,SDNode*>(SDValue(Res, i),
                                                             nullptr));
        }

      } else if (NodeToMatch->getOpcode() != ISD::DELETED_NODE) {
        Res = MorphNode(NodeToMatch, TargetOpc, VTList, Ops, EmitNodeInfo);
      } else {
        // NodeToMatch was eliminated by CSE when the target changed the DAG.
        // We will visit the equivalent node later.
        DEBUG(dbgs() << "Node was eliminated by CSE\n");
        return nullptr;
      }

      // If the node had chain/glue results, update our notion of the current
      // chain and glue.
      if (EmitNodeInfo & OPFL_GlueOutput) {
        InputGlue = SDValue(Res, VTs.size()-1);
        if (EmitNodeInfo & OPFL_Chain)
          InputChain = SDValue(Res, VTs.size()-2);
      } else if (EmitNodeInfo & OPFL_Chain)
        InputChain = SDValue(Res, VTs.size()-1);

      // If the OPFL_MemRefs glue is set on this node, slap all of the
      // accumulated memrefs onto it.
      //
      // FIXME: This is vastly incorrect for patterns with multiple outputs
      // instructions that access memory and for ComplexPatterns that match
      // loads.
      if (EmitNodeInfo & OPFL_MemRefs) {
        // Only attach load or store memory operands if the generated
        // instruction may load or store.
        const MCInstrDesc &MCID = TII->get(TargetOpc);
        bool mayLoad = MCID.mayLoad();
        bool mayStore = MCID.mayStore();

        unsigned NumMemRefs = 0;
        for (SmallVectorImpl<MachineMemOperand *>::const_iterator I =
               MatchedMemRefs.begin(), E = MatchedMemRefs.end(); I != E; ++I) {
          if ((*I)->isLoad()) {
            if (mayLoad)
              ++NumMemRefs;
          } else if ((*I)->isStore()) {
            if (mayStore)
              ++NumMemRefs;
          } else {
            ++NumMemRefs;
          }
        }

        MachineSDNode::mmo_iterator MemRefs =
          MF->allocateMemRefsArray(NumMemRefs);

        MachineSDNode::mmo_iterator MemRefsPos = MemRefs;
        for (SmallVectorImpl<MachineMemOperand *>::const_iterator I =
               MatchedMemRefs.begin(), E = MatchedMemRefs.end(); I != E; ++I) {
          if ((*I)->isLoad()) {
            if (mayLoad)
              *MemRefsPos++ = *I;
          } else if ((*I)->isStore()) {
            if (mayStore)
              *MemRefsPos++ = *I;
          } else {
            *MemRefsPos++ = *I;
          }
        }

        cast<MachineSDNode>(Res)
          ->setMemRefs(MemRefs, MemRefs + NumMemRefs);
      }

      DEBUG(dbgs() << "  "
                   << (Opcode == OPC_MorphNodeTo ? "Morphed" : "Created")
                   << " node: "; Res->dump(CurDAG); dbgs() << "\n");

      // If this was a MorphNodeTo then we're completely done!
      if (Opcode == OPC_MorphNodeTo) {
        // Update chain and glue uses.
        UpdateChainsAndGlue(NodeToMatch, InputChain, ChainNodesMatched,
                            InputGlue, GlueResultNodesMatched, true);
        return Res;
      }

      continue;
    }

    case OPC_MarkGlueResults: {
      unsigned NumNodes = MatcherTable[MatcherIndex++];

      // Read and remember all the glue-result nodes.
      for (unsigned i = 0; i != NumNodes; ++i) {
        unsigned RecNo = MatcherTable[MatcherIndex++];
        if (RecNo & 128)
          RecNo = GetVBR(RecNo, MatcherTable, MatcherIndex);

        assert(RecNo < RecordedNodes.size() && "Invalid MarkGlueResults");
        GlueResultNodesMatched.push_back(RecordedNodes[RecNo].first.getNode());
      }
      continue;
    }

    case OPC_CompleteMatch: {
      // The match has been completed, and any new nodes (if any) have been
      // created.  Patch up references to the matched dag to use the newly
      // created nodes.
      unsigned NumResults = MatcherTable[MatcherIndex++];

      for (unsigned i = 0; i != NumResults; ++i) {
        unsigned ResSlot = MatcherTable[MatcherIndex++];
        if (ResSlot & 128)
          ResSlot = GetVBR(ResSlot, MatcherTable, MatcherIndex);

        assert(ResSlot < RecordedNodes.size() && "Invalid CompleteMatch");
        SDValue Res = RecordedNodes[ResSlot].first;

        assert(i < NodeToMatch->getNumValues() &&
               NodeToMatch->getValueType(i) != MVT::Other &&
               NodeToMatch->getValueType(i) != MVT::Glue &&
               "Invalid number of results to complete!");
        assert((NodeToMatch->getValueType(i) == Res.getValueType() ||
                NodeToMatch->getValueType(i) == MVT::iPTR ||
                Res.getValueType() == MVT::iPTR ||
                NodeToMatch->getValueType(i).getSizeInBits() ==
                    Res.getValueType().getSizeInBits()) &&
               "invalid replacement");
        CurDAG->ReplaceAllUsesOfValueWith(SDValue(NodeToMatch, i), Res);
      }

      // If the root node defines glue, add it to the glue nodes to update list.
      if (NodeToMatch->getValueType(NodeToMatch->getNumValues()-1) == MVT::Glue)
        GlueResultNodesMatched.push_back(NodeToMatch);

      // Update chain and glue uses.
      UpdateChainsAndGlue(NodeToMatch, InputChain, ChainNodesMatched,
                          InputGlue, GlueResultNodesMatched, false);

      assert(NodeToMatch->use_empty() &&
             "Didn't replace all uses of the node?");

      // FIXME: We just return here, which interacts correctly with SelectRoot
      // above.  We should fix this to not return an SDNode* anymore.
      return nullptr;
    }
    }

    // If the code reached this point, then the match failed.  See if there is
    // another child to try in the current 'Scope', otherwise pop it until we
    // find a case to check.
    DEBUG(dbgs() << "  Match failed at index " << CurrentOpcodeIndex << "\n");
    ++NumDAGIselRetries;
    while (1) {
      if (MatchScopes.empty()) {
        CannotYetSelect(NodeToMatch);
        return nullptr;
      }

      // Restore the interpreter state back to the point where the scope was
      // formed.
      MatchScope &LastScope = MatchScopes.back();
      RecordedNodes.resize(LastScope.NumRecordedNodes);
      NodeStack.clear();
      NodeStack.append(LastScope.NodeStack.begin(), LastScope.NodeStack.end());
      N = NodeStack.back();

      if (LastScope.NumMatchedMemRefs != MatchedMemRefs.size())
        MatchedMemRefs.resize(LastScope.NumMatchedMemRefs);
      MatcherIndex = LastScope.FailIndex;

      DEBUG(dbgs() << "  Continuing at " << MatcherIndex << "\n");

      InputChain = LastScope.InputChain;
      InputGlue = LastScope.InputGlue;
      if (!LastScope.HasChainNodesMatched)
        ChainNodesMatched.clear();
      if (!LastScope.HasGlueResultNodesMatched)
        GlueResultNodesMatched.clear();

      // Check to see what the offset is at the new MatcherIndex.  If it is zero
      // we have reached the end of this scope, otherwise we have another child
      // in the current scope to try.
      unsigned NumToSkip = MatcherTable[MatcherIndex++];
      if (NumToSkip & 128)
        NumToSkip = GetVBR(NumToSkip, MatcherTable, MatcherIndex);

      // If we have another child in this scope to match, update FailIndex and
      // try it.
      if (NumToSkip != 0) {
        LastScope.FailIndex = MatcherIndex+NumToSkip;
        break;
      }

      // End of this scope, pop it and try the next child in the containing
      // scope.
      MatchScopes.pop_back();
    }
  }
}



void SelectionDAGISel::CannotYetSelect(SDNode *N) {
  std::string msg;
  raw_string_ostream Msg(msg);
  Msg << "Cannot select: ";

  if (N->getOpcode() != ISD::INTRINSIC_W_CHAIN &&
      N->getOpcode() != ISD::INTRINSIC_WO_CHAIN &&
      N->getOpcode() != ISD::INTRINSIC_VOID) {
    N->printrFull(Msg, CurDAG);
    Msg << "\nIn function: " << MF->getName();
  } else {
    bool HasInputChain = N->getOperand(0).getValueType() == MVT::Other;
    unsigned iid =
      cast<ConstantSDNode>(N->getOperand(HasInputChain))->getZExtValue();
    if (iid < Intrinsic::num_intrinsics)
      Msg << "intrinsic %" << Intrinsic::getName((Intrinsic::ID)iid);
    else if (const TargetIntrinsicInfo *TII = TM.getIntrinsicInfo())
      Msg << "target intrinsic %" << TII->getName(iid);
    else
      Msg << "unknown intrinsic #" << iid;
  }
  report_fatal_error(Msg.str());
}

char SelectionDAGISel::ID = 0;
