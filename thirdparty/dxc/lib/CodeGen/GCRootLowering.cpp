//===-- GCRootLowering.cpp - Garbage collection infrastructure ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the lowering for the gc.root mechanism.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GCMetadata.h"
#include "llvm/CodeGen/GCStrategy.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"

using namespace llvm;

namespace {

/// LowerIntrinsics - This pass rewrites calls to the llvm.gcread or
/// llvm.gcwrite intrinsics, replacing them with simple loads and stores as
/// directed by the GCStrategy. It also performs automatic root initialization
/// and custom intrinsic lowering.
class LowerIntrinsics : public FunctionPass {
  bool PerformDefaultLowering(Function &F, GCStrategy &Coll);

public:
  static char ID;

  LowerIntrinsics();
  StringRef getPassName() const override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;

  bool doInitialization(Module &M) override;
  bool runOnFunction(Function &F) override;
};

/// GCMachineCodeAnalysis - This is a target-independent pass over the machine
/// function representation to identify safe points for the garbage collector
/// in the machine code. It inserts labels at safe points and populates a
/// GCMetadata record for each function.
class GCMachineCodeAnalysis : public MachineFunctionPass {
  GCFunctionInfo *FI;
  MachineModuleInfo *MMI;
  const TargetInstrInfo *TII;

  void FindSafePoints(MachineFunction &MF);
  void VisitCallPoint(MachineBasicBlock::iterator MI);
  MCSymbol *InsertLabel(MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
                        DebugLoc DL) const;

  void FindStackOffsets(MachineFunction &MF);

public:
  static char ID;

  GCMachineCodeAnalysis();
  void getAnalysisUsage(AnalysisUsage &AU) const override;

  bool runOnMachineFunction(MachineFunction &MF) override;
};
}

// -----------------------------------------------------------------------------

INITIALIZE_PASS_BEGIN(LowerIntrinsics, "gc-lowering", "GC Lowering", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(GCModuleInfo)
INITIALIZE_PASS_END(LowerIntrinsics, "gc-lowering", "GC Lowering", false, false)

FunctionPass *llvm::createGCLoweringPass() { return new LowerIntrinsics(); }

char LowerIntrinsics::ID = 0;

LowerIntrinsics::LowerIntrinsics() : FunctionPass(ID) {
  initializeLowerIntrinsicsPass(*PassRegistry::getPassRegistry());
}

const char *LowerIntrinsics::getPassName() const {
  return "Lower Garbage Collection Instructions";
}

void LowerIntrinsics::getAnalysisUsage(AnalysisUsage &AU) const {
  FunctionPass::getAnalysisUsage(AU);
  AU.addRequired<GCModuleInfo>();
  AU.addPreserved<DominatorTreeWrapperPass>();
}

static bool NeedsDefaultLoweringPass(const GCStrategy &C) {
  // Default lowering is necessary only if read or write barriers have a default
  // action. The default for roots is no action.
  return !C.customWriteBarrier() || !C.customReadBarrier() ||
         C.initializeRoots();
}

/// doInitialization - If this module uses the GC intrinsics, find them now.
bool LowerIntrinsics::doInitialization(Module &M) {
  GCModuleInfo *MI = getAnalysisIfAvailable<GCModuleInfo>();
  assert(MI && "LowerIntrinsics didn't require GCModuleInfo!?");
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
    if (!I->isDeclaration() && I->hasGC())
      MI->getFunctionInfo(*I); // Instantiate the GC strategy.

  return false;
}

/// CouldBecomeSafePoint - Predicate to conservatively determine whether the
/// instruction could introduce a safe point.
static bool CouldBecomeSafePoint(Instruction *I) {
  // The natural definition of instructions which could introduce safe points
  // are:
  //
  //   - call, invoke (AfterCall, BeforeCall)
  //   - phis (Loops)
  //   - invoke, ret, unwind (Exit)
  //
  // However, instructions as seemingly inoccuous as arithmetic can become
  // libcalls upon lowering (e.g., div i64 on a 32-bit platform), so instead
  // it is necessary to take a conservative approach.

  if (isa<AllocaInst>(I) || isa<GetElementPtrInst>(I) || isa<StoreInst>(I) ||
      isa<LoadInst>(I))
    return false;

  // llvm.gcroot is safe because it doesn't do anything at runtime.
  if (CallInst *CI = dyn_cast<CallInst>(I))
    if (Function *F = CI->getCalledFunction())
      if (Intrinsic::ID IID = F->getIntrinsicID())
        if (IID == Intrinsic::gcroot)
          return false;

  return true;
}

static bool InsertRootInitializers(Function &F, AllocaInst **Roots,
                                   unsigned Count) {
  // Scroll past alloca instructions.
  BasicBlock::iterator IP = F.getEntryBlock().begin();
  while (isa<AllocaInst>(IP))
    ++IP;

  // Search for initializers in the initial BB.
  SmallPtrSet<AllocaInst *, 16> InitedRoots;
  for (; !CouldBecomeSafePoint(IP); ++IP)
    if (StoreInst *SI = dyn_cast<StoreInst>(IP))
      if (AllocaInst *AI =
              dyn_cast<AllocaInst>(SI->getOperand(1)->stripPointerCasts()))
        InitedRoots.insert(AI);

  // Add root initializers.
  bool MadeChange = false;

  for (AllocaInst **I = Roots, **E = Roots + Count; I != E; ++I)
    if (!InitedRoots.count(*I)) {
      StoreInst *SI = new StoreInst(
          ConstantPointerNull::get(cast<PointerType>(
              cast<PointerType>((*I)->getType())->getElementType())),
          *I);
      SI->insertAfter(*I);
      MadeChange = true;
    }

  return MadeChange;
}

/// runOnFunction - Replace gcread/gcwrite intrinsics with loads and stores.
/// Leave gcroot intrinsics; the code generator needs to see those.
bool LowerIntrinsics::runOnFunction(Function &F) {
  // Quick exit for functions that do not use GC.
  if (!F.hasGC())
    return false;

  GCFunctionInfo &FI = getAnalysis<GCModuleInfo>().getFunctionInfo(F);
  GCStrategy &S = FI.getStrategy();

  bool MadeChange = false;

  if (NeedsDefaultLoweringPass(S))
    MadeChange |= PerformDefaultLowering(F, S);

  return MadeChange;
}

bool LowerIntrinsics::PerformDefaultLowering(Function &F, GCStrategy &S) {
  bool LowerWr = !S.customWriteBarrier();
  bool LowerRd = !S.customReadBarrier();
  bool InitRoots = S.initializeRoots();

  SmallVector<AllocaInst *, 32> Roots;

  bool MadeChange = false;
  for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB) {
    for (BasicBlock::iterator II = BB->begin(), E = BB->end(); II != E;) {
      if (IntrinsicInst *CI = dyn_cast<IntrinsicInst>(II++)) {
        Function *F = CI->getCalledFunction();
        switch (F->getIntrinsicID()) {
        case Intrinsic::gcwrite:
          if (LowerWr) {
            // Replace a write barrier with a simple store.
            Value *St =
                new StoreInst(CI->getArgOperand(0), CI->getArgOperand(2), CI);
            CI->replaceAllUsesWith(St);
            CI->eraseFromParent();
          }
          break;
        case Intrinsic::gcread:
          if (LowerRd) {
            // Replace a read barrier with a simple load.
            Value *Ld = new LoadInst(CI->getArgOperand(1), "", CI);
            Ld->takeName(CI);
            CI->replaceAllUsesWith(Ld);
            CI->eraseFromParent();
          }
          break;
        case Intrinsic::gcroot:
          if (InitRoots) {
            // Initialize the GC root, but do not delete the intrinsic. The
            // backend needs the intrinsic to flag the stack slot.
            Roots.push_back(
                cast<AllocaInst>(CI->getArgOperand(0)->stripPointerCasts()));
          }
          break;
        default:
          continue;
        }

        MadeChange = true;
      }
    }
  }

  if (Roots.size())
    MadeChange |= InsertRootInitializers(F, Roots.begin(), Roots.size());

  return MadeChange;
}

// -----------------------------------------------------------------------------

char GCMachineCodeAnalysis::ID = 0;
char &llvm::GCMachineCodeAnalysisID = GCMachineCodeAnalysis::ID;

INITIALIZE_PASS(GCMachineCodeAnalysis, "gc-analysis",
                "Analyze Machine Code For Garbage Collection", false, false)

GCMachineCodeAnalysis::GCMachineCodeAnalysis() : MachineFunctionPass(ID) {}

void GCMachineCodeAnalysis::getAnalysisUsage(AnalysisUsage &AU) const {
  MachineFunctionPass::getAnalysisUsage(AU);
  AU.setPreservesAll();
  AU.addRequired<MachineModuleInfo>();
  AU.addRequired<GCModuleInfo>();
}

MCSymbol *GCMachineCodeAnalysis::InsertLabel(MachineBasicBlock &MBB,
                                             MachineBasicBlock::iterator MI,
                                             DebugLoc DL) const {
  MCSymbol *Label = MBB.getParent()->getContext().createTempSymbol();
  BuildMI(MBB, MI, DL, TII->get(TargetOpcode::GC_LABEL)).addSym(Label);
  return Label;
}

void GCMachineCodeAnalysis::VisitCallPoint(MachineBasicBlock::iterator CI) {
  // Find the return address (next instruction), too, so as to bracket the call
  // instruction.
  MachineBasicBlock::iterator RAI = CI;
  ++RAI;

  if (FI->getStrategy().needsSafePoint(GC::PreCall)) {
    MCSymbol *Label = InsertLabel(*CI->getParent(), CI, CI->getDebugLoc());
    FI->addSafePoint(GC::PreCall, Label, CI->getDebugLoc());
  }

  if (FI->getStrategy().needsSafePoint(GC::PostCall)) {
    MCSymbol *Label = InsertLabel(*CI->getParent(), RAI, CI->getDebugLoc());
    FI->addSafePoint(GC::PostCall, Label, CI->getDebugLoc());
  }
}

void GCMachineCodeAnalysis::FindSafePoints(MachineFunction &MF) {
  for (MachineFunction::iterator BBI = MF.begin(), BBE = MF.end(); BBI != BBE;
       ++BBI)
    for (MachineBasicBlock::iterator MI = BBI->begin(), ME = BBI->end();
         MI != ME; ++MI)
      if (MI->isCall()) {
        // Do not treat tail or sibling call sites as safe points.  This is
        // legal since any arguments passed to the callee which live in the
        // remnants of the callers frame will be owned and updated by the
        // callee if required.
        if (MI->isTerminator())
          continue;
        VisitCallPoint(MI);
      }
}

void GCMachineCodeAnalysis::FindStackOffsets(MachineFunction &MF) {
  const TargetFrameLowering *TFI = MF.getSubtarget().getFrameLowering();
  assert(TFI && "TargetRegisterInfo not available!");

  for (GCFunctionInfo::roots_iterator RI = FI->roots_begin();
       RI != FI->roots_end();) {
    // If the root references a dead object, no need to keep it.
    if (MF.getFrameInfo()->isDeadObjectIndex(RI->Num)) {
      RI = FI->removeStackRoot(RI);
    } else {
      RI->StackOffset = TFI->getFrameIndexOffset(MF, RI->Num);
      ++RI;
    }
  }
}

bool GCMachineCodeAnalysis::runOnMachineFunction(MachineFunction &MF) {
  // Quick exit for functions that do not use GC.
  if (!MF.getFunction()->hasGC())
    return false;

  FI = &getAnalysis<GCModuleInfo>().getFunctionInfo(*MF.getFunction());
  MMI = &getAnalysis<MachineModuleInfo>();
  TII = MF.getSubtarget().getInstrInfo();

  // Find the size of the stack frame.  There may be no correct static frame
  // size, we use UINT64_MAX to represent this.
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  const TargetRegisterInfo *RegInfo = MF.getSubtarget().getRegisterInfo();
  const bool DynamicFrameSize = MFI->hasVarSizedObjects() ||
    RegInfo->needsStackRealignment(MF);
  FI->setFrameSize(DynamicFrameSize ? UINT64_MAX : MFI->getStackSize());

  // Find all safe points.
  if (FI->getStrategy().needsSafePoints())
    FindSafePoints(MF);

  // Find the concrete stack offsets for all roots (stack slots)
  FindStackOffsets(MF);

  return false;
}
