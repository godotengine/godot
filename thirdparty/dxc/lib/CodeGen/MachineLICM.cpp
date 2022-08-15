//===-- MachineLICM.cpp - Machine Loop Invariant Code Motion Pass ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass performs loop invariant code motion on machine instructions. We
// attempt to remove as much code from the body of a loop as possible.
//
// This pass is not intended to be a replacement or a complete alternative
// for the LLVM-IR-level LICM pass. It is only designed to hoist simple
// constructs that are not exposed before lowering and instruction selection.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/CodeGen/TargetSchedule.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"
using namespace llvm;

#define DEBUG_TYPE "machine-licm"

static cl::opt<bool>
AvoidSpeculation("avoid-speculation",
                 cl::desc("MachineLICM should avoid speculation"),
                 cl::init(true), cl::Hidden);

static cl::opt<bool>
HoistCheapInsts("hoist-cheap-insts",
                cl::desc("MachineLICM should hoist even cheap instructions"),
                cl::init(false), cl::Hidden);

static cl::opt<bool>
SinkInstsToAvoidSpills("sink-insts-to-avoid-spills",
                       cl::desc("MachineLICM should sink instructions into "
                                "loops to avoid register spills"),
                       cl::init(false), cl::Hidden);

STATISTIC(NumHoisted,
          "Number of machine instructions hoisted out of loops");
STATISTIC(NumLowRP,
          "Number of instructions hoisted in low reg pressure situation");
STATISTIC(NumHighLatency,
          "Number of high latency instructions hoisted");
STATISTIC(NumCSEed,
          "Number of hoisted machine instructions CSEed");
STATISTIC(NumPostRAHoisted,
          "Number of machine instructions hoisted out of loops post regalloc");

namespace {
  class MachineLICM : public MachineFunctionPass {
    const TargetInstrInfo *TII;
    const TargetLoweringBase *TLI;
    const TargetRegisterInfo *TRI;
    const MachineFrameInfo *MFI;
    MachineRegisterInfo *MRI;
    TargetSchedModel SchedModel;
    bool PreRegAlloc;

    // Various analyses that we use...
    AliasAnalysis        *AA;      // Alias analysis info.
    MachineLoopInfo      *MLI;     // Current MachineLoopInfo
    MachineDominatorTree *DT;      // Machine dominator tree for the cur loop

    // State that is updated as we process loops
    bool         Changed;          // True if a loop is changed.
    bool         FirstInLoop;      // True if it's the first LICM in the loop.
    MachineLoop *CurLoop;          // The current loop we are working on.
    MachineBasicBlock *CurPreheader; // The preheader for CurLoop.

    // Exit blocks for CurLoop.
    SmallVector<MachineBasicBlock*, 8> ExitBlocks;

    bool isExitBlock(const MachineBasicBlock *MBB) const {
      return std::find(ExitBlocks.begin(), ExitBlocks.end(), MBB) !=
        ExitBlocks.end();
    }

    // Track 'estimated' register pressure.
    SmallSet<unsigned, 32> RegSeen;
    SmallVector<unsigned, 8> RegPressure;

    // Register pressure "limit" per register pressure set. If the pressure
    // is higher than the limit, then it's considered high.
    SmallVector<unsigned, 8> RegLimit;

    // Register pressure on path leading from loop preheader to current BB.
    SmallVector<SmallVector<unsigned, 8>, 16> BackTrace;

    // For each opcode, keep a list of potential CSE instructions.
    DenseMap<unsigned, std::vector<const MachineInstr*> > CSEMap;

    enum {
      SpeculateFalse   = 0,
      SpeculateTrue    = 1,
      SpeculateUnknown = 2
    };

    // If a MBB does not dominate loop exiting blocks then it may not safe
    // to hoist loads from this block.
    // Tri-state: 0 - false, 1 - true, 2 - unknown
    unsigned SpeculationState;

  public:
    static char ID; // Pass identification, replacement for typeid
    MachineLICM() :
      MachineFunctionPass(ID), PreRegAlloc(true) {
        initializeMachineLICMPass(*PassRegistry::getPassRegistry());
      }

    explicit MachineLICM(bool PreRA) :
      MachineFunctionPass(ID), PreRegAlloc(PreRA) {
        initializeMachineLICMPass(*PassRegistry::getPassRegistry());
      }

    bool runOnMachineFunction(MachineFunction &MF) override;

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.addRequired<MachineLoopInfo>();
      AU.addRequired<MachineDominatorTree>();
      AU.addRequired<AliasAnalysis>();
      AU.addPreserved<MachineLoopInfo>();
      AU.addPreserved<MachineDominatorTree>();
      MachineFunctionPass::getAnalysisUsage(AU);
    }

    void releaseMemory() override {
      RegSeen.clear();
      RegPressure.clear();
      RegLimit.clear();
      BackTrace.clear();
      CSEMap.clear();
    }

  private:
    /// CandidateInfo - Keep track of information about hoisting candidates.
    struct CandidateInfo {
      MachineInstr *MI;
      unsigned      Def;
      int           FI;
      CandidateInfo(MachineInstr *mi, unsigned def, int fi)
        : MI(mi), Def(def), FI(fi) {}
    };

    /// HoistRegionPostRA - Walk the specified region of the CFG and hoist loop
    /// invariants out to the preheader.
    void HoistRegionPostRA();

    /// HoistPostRA - When an instruction is found to only use loop invariant
    /// operands that is safe to hoist, this instruction is called to do the
    /// dirty work.
    void HoistPostRA(MachineInstr *MI, unsigned Def);

    /// ProcessMI - Examine the instruction for potentai LICM candidate. Also
    /// gather register def and frame object update information.
    void ProcessMI(MachineInstr *MI,
                   BitVector &PhysRegDefs,
                   BitVector &PhysRegClobbers,
                   SmallSet<int, 32> &StoredFIs,
                   SmallVectorImpl<CandidateInfo> &Candidates);

    /// AddToLiveIns - Add register 'Reg' to the livein sets of BBs in the
    /// current loop.
    void AddToLiveIns(unsigned Reg);

    /// IsLICMCandidate - Returns true if the instruction may be a suitable
    /// candidate for LICM. e.g. If the instruction is a call, then it's
    /// obviously not safe to hoist it.
    bool IsLICMCandidate(MachineInstr &I);

    /// IsLoopInvariantInst - Returns true if the instruction is loop
    /// invariant. I.e., all virtual register operands are defined outside of
    /// the loop, physical registers aren't accessed (explicitly or implicitly),
    /// and the instruction is hoistable.
    ///
    bool IsLoopInvariantInst(MachineInstr &I);

    /// HasLoopPHIUse - Return true if the specified instruction is used by any
    /// phi node in the current loop.
    bool HasLoopPHIUse(const MachineInstr *MI) const;

    /// HasHighOperandLatency - Compute operand latency between a def of 'Reg'
    /// and an use in the current loop, return true if the target considered
    /// it 'high'.
    bool HasHighOperandLatency(MachineInstr &MI, unsigned DefIdx,
                               unsigned Reg) const;

    bool IsCheapInstruction(MachineInstr &MI) const;

    /// CanCauseHighRegPressure - Visit BBs from header to current BB,
    /// check if hoisting an instruction of the given cost matrix can cause high
    /// register pressure.
    bool CanCauseHighRegPressure(const DenseMap<unsigned, int> &Cost,
                                 bool Cheap);

    /// UpdateBackTraceRegPressure - Traverse the back trace from header to
    /// the current block and update their register pressures to reflect the
    /// effect of hoisting MI from the current block to the preheader.
    void UpdateBackTraceRegPressure(const MachineInstr *MI);

    /// IsProfitableToHoist - Return true if it is potentially profitable to
    /// hoist the given loop invariant.
    bool IsProfitableToHoist(MachineInstr &MI);

    /// IsGuaranteedToExecute - Check if this mbb is guaranteed to execute.
    /// If not then a load from this mbb may not be safe to hoist.
    bool IsGuaranteedToExecute(MachineBasicBlock *BB);

    void EnterScope(MachineBasicBlock *MBB);

    void ExitScope(MachineBasicBlock *MBB);

    /// ExitScopeIfDone - Destroy scope for the MBB that corresponds to given
    /// dominator tree node if its a leaf or all of its children are done. Walk
    /// up the dominator tree to destroy ancestors which are now done.
    void ExitScopeIfDone(MachineDomTreeNode *Node,
                DenseMap<MachineDomTreeNode*, unsigned> &OpenChildren,
                DenseMap<MachineDomTreeNode*, MachineDomTreeNode*> &ParentMap);

    /// HoistOutOfLoop - Walk the specified loop in the CFG (defined by all
    /// blocks dominated by the specified header block, and that are in the
    /// current loop) in depth first order w.r.t the DominatorTree. This allows
    /// us to visit definitions before uses, allowing us to hoist a loop body in
    /// one pass without iteration.
    ///
    void HoistOutOfLoop(MachineDomTreeNode *LoopHeaderNode);
    void HoistRegion(MachineDomTreeNode *N, bool IsHeader);

    /// SinkIntoLoop - Sink instructions into loops if profitable. This
    /// especially tries to prevent register spills caused by register pressure
    /// if there is little to no overhead moving instructions into loops.
    void SinkIntoLoop();

    /// InitRegPressure - Find all virtual register references that are liveout
    /// of the preheader to initialize the starting "register pressure". Note
    /// this does not count live through (livein but not used) registers.
    void InitRegPressure(MachineBasicBlock *BB);

    /// calcRegisterCost - Calculate the additional register pressure that the
    /// registers used in MI cause.
    ///
    /// If 'ConsiderSeen' is true, updates 'RegSeen' and uses the information to
    /// figure out which usages are live-ins.
    /// FIXME: Figure out a way to consider 'RegSeen' from all code paths.
    DenseMap<unsigned, int> calcRegisterCost(const MachineInstr *MI,
                                             bool ConsiderSeen,
                                             bool ConsiderUnseenAsDef);

    /// UpdateRegPressure - Update estimate of register pressure after the
    /// specified instruction.
    void UpdateRegPressure(const MachineInstr *MI,
                           bool ConsiderUnseenAsDef = false);

    /// ExtractHoistableLoad - Unfold a load from the given machineinstr if
    /// the load itself could be hoisted. Return the unfolded and hoistable
    /// load, or null if the load couldn't be unfolded or if it wouldn't
    /// be hoistable.
    MachineInstr *ExtractHoistableLoad(MachineInstr *MI);

    /// LookForDuplicate - Find an instruction amount PrevMIs that is a
    /// duplicate of MI. Return this instruction if it's found.
    const MachineInstr *LookForDuplicate(const MachineInstr *MI,
                                     std::vector<const MachineInstr*> &PrevMIs);

    /// EliminateCSE - Given a LICM'ed instruction, look for an instruction on
    /// the preheader that compute the same value. If it's found, do a RAU on
    /// with the definition of the existing instruction rather than hoisting
    /// the instruction to the preheader.
    bool EliminateCSE(MachineInstr *MI,
           DenseMap<unsigned, std::vector<const MachineInstr*> >::iterator &CI);

    /// MayCSE - Return true if the given instruction will be CSE'd if it's
    /// hoisted out of the loop.
    bool MayCSE(MachineInstr *MI);

    /// Hoist - When an instruction is found to only use loop invariant operands
    /// that is safe to hoist, this instruction is called to do the dirty work.
    /// It returns true if the instruction is hoisted.
    bool Hoist(MachineInstr *MI, MachineBasicBlock *Preheader);

    /// InitCSEMap - Initialize the CSE map with instructions that are in the
    /// current loop preheader that may become duplicates of instructions that
    /// are hoisted out of the loop.
    void InitCSEMap(MachineBasicBlock *BB);

    /// getCurPreheader - Get the preheader for the current loop, splitting
    /// a critical edge if needed.
    MachineBasicBlock *getCurPreheader();
  };
} // end anonymous namespace

char MachineLICM::ID = 0;
char &llvm::MachineLICMID = MachineLICM::ID;
INITIALIZE_PASS_BEGIN(MachineLICM, "machinelicm",
                "Machine Loop Invariant Code Motion", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfo)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTree)
INITIALIZE_AG_DEPENDENCY(AliasAnalysis)
INITIALIZE_PASS_END(MachineLICM, "machinelicm",
                "Machine Loop Invariant Code Motion", false, false)

/// LoopIsOuterMostWithPredecessor - Test if the given loop is the outer-most
/// loop that has a unique predecessor.
static bool LoopIsOuterMostWithPredecessor(MachineLoop *CurLoop) {
  // Check whether this loop even has a unique predecessor.
  if (!CurLoop->getLoopPredecessor())
    return false;
  // Ok, now check to see if any of its outer loops do.
  for (MachineLoop *L = CurLoop->getParentLoop(); L; L = L->getParentLoop())
    if (L->getLoopPredecessor())
      return false;
  // None of them did, so this is the outermost with a unique predecessor.
  return true;
}

bool MachineLICM::runOnMachineFunction(MachineFunction &MF) {
  if (skipOptnoneFunction(*MF.getFunction()))
    return false;

  Changed = FirstInLoop = false;
  const TargetSubtargetInfo &ST = MF.getSubtarget();
  TII = ST.getInstrInfo();
  TLI = ST.getTargetLowering();
  TRI = ST.getRegisterInfo();
  MFI = MF.getFrameInfo();
  MRI = &MF.getRegInfo();
  SchedModel.init(ST.getSchedModel(), &ST, TII);

  PreRegAlloc = MRI->isSSA();

  if (PreRegAlloc)
    DEBUG(dbgs() << "******** Pre-regalloc Machine LICM: ");
  else
    DEBUG(dbgs() << "******** Post-regalloc Machine LICM: ");
  DEBUG(dbgs() << MF.getName() << " ********\n");

  if (PreRegAlloc) {
    // Estimate register pressure during pre-regalloc pass.
    unsigned NumRPS = TRI->getNumRegPressureSets();
    RegPressure.resize(NumRPS);
    std::fill(RegPressure.begin(), RegPressure.end(), 0);
    RegLimit.resize(NumRPS);
    for (unsigned i = 0, e = NumRPS; i != e; ++i)
      RegLimit[i] = TRI->getRegPressureSetLimit(MF, i);
  }

  // Get our Loop information...
  MLI = &getAnalysis<MachineLoopInfo>();
  DT  = &getAnalysis<MachineDominatorTree>();
  AA  = &getAnalysis<AliasAnalysis>();

  SmallVector<MachineLoop *, 8> Worklist(MLI->begin(), MLI->end());
  while (!Worklist.empty()) {
    CurLoop = Worklist.pop_back_val();
    CurPreheader = nullptr;
    ExitBlocks.clear();

    // If this is done before regalloc, only visit outer-most preheader-sporting
    // loops.
    if (PreRegAlloc && !LoopIsOuterMostWithPredecessor(CurLoop)) {
      Worklist.append(CurLoop->begin(), CurLoop->end());
      continue;
    }

    CurLoop->getExitBlocks(ExitBlocks);

    if (!PreRegAlloc)
      HoistRegionPostRA();
    else {
      // CSEMap is initialized for loop header when the first instruction is
      // being hoisted.
      MachineDomTreeNode *N = DT->getNode(CurLoop->getHeader());
      FirstInLoop = true;
      HoistOutOfLoop(N);
      CSEMap.clear();

      if (SinkInstsToAvoidSpills)
        SinkIntoLoop();
    }
  }

  return Changed;
}

/// InstructionStoresToFI - Return true if instruction stores to the
/// specified frame.
static bool InstructionStoresToFI(const MachineInstr *MI, int FI) {
  for (MachineInstr::mmo_iterator o = MI->memoperands_begin(),
         oe = MI->memoperands_end(); o != oe; ++o) {
    if (!(*o)->isStore() || !(*o)->getPseudoValue())
      continue;
    if (const FixedStackPseudoSourceValue *Value =
        dyn_cast<FixedStackPseudoSourceValue>((*o)->getPseudoValue())) {
      if (Value->getFrameIndex() == FI)
        return true;
    }
  }
  return false;
}

/// ProcessMI - Examine the instruction for potentai LICM candidate. Also
/// gather register def and frame object update information.
void MachineLICM::ProcessMI(MachineInstr *MI,
                            BitVector &PhysRegDefs,
                            BitVector &PhysRegClobbers,
                            SmallSet<int, 32> &StoredFIs,
                            SmallVectorImpl<CandidateInfo> &Candidates) {
  bool RuledOut = false;
  bool HasNonInvariantUse = false;
  unsigned Def = 0;
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    if (MO.isFI()) {
      // Remember if the instruction stores to the frame index.
      int FI = MO.getIndex();
      if (!StoredFIs.count(FI) &&
          MFI->isSpillSlotObjectIndex(FI) &&
          InstructionStoresToFI(MI, FI))
        StoredFIs.insert(FI);
      HasNonInvariantUse = true;
      continue;
    }

    // We can't hoist an instruction defining a physreg that is clobbered in
    // the loop.
    if (MO.isRegMask()) {
      PhysRegClobbers.setBitsNotInMask(MO.getRegMask());
      continue;
    }

    if (!MO.isReg())
      continue;
    unsigned Reg = MO.getReg();
    if (!Reg)
      continue;
    assert(TargetRegisterInfo::isPhysicalRegister(Reg) &&
           "Not expecting virtual register!");

    if (!MO.isDef()) {
      if (Reg && (PhysRegDefs.test(Reg) || PhysRegClobbers.test(Reg)))
        // If it's using a non-loop-invariant register, then it's obviously not
        // safe to hoist.
        HasNonInvariantUse = true;
      continue;
    }

    if (MO.isImplicit()) {
      for (MCRegAliasIterator AI(Reg, TRI, true); AI.isValid(); ++AI)
        PhysRegClobbers.set(*AI);
      if (!MO.isDead())
        // Non-dead implicit def? This cannot be hoisted.
        RuledOut = true;
      // No need to check if a dead implicit def is also defined by
      // another instruction.
      continue;
    }

    // FIXME: For now, avoid instructions with multiple defs, unless
    // it's a dead implicit def.
    if (Def)
      RuledOut = true;
    else
      Def = Reg;

    // If we have already seen another instruction that defines the same
    // register, then this is not safe.  Two defs is indicated by setting a
    // PhysRegClobbers bit.
    for (MCRegAliasIterator AS(Reg, TRI, true); AS.isValid(); ++AS) {
      if (PhysRegDefs.test(*AS))
        PhysRegClobbers.set(*AS);
      PhysRegDefs.set(*AS);
    }
    if (PhysRegClobbers.test(Reg))
      // MI defined register is seen defined by another instruction in
      // the loop, it cannot be a LICM candidate.
      RuledOut = true;
  }

  // Only consider reloads for now and remats which do not have register
  // operands. FIXME: Consider unfold load folding instructions.
  if (Def && !RuledOut) {
    int FI = INT_MIN;
    if ((!HasNonInvariantUse && IsLICMCandidate(*MI)) ||
        (TII->isLoadFromStackSlot(MI, FI) && MFI->isSpillSlotObjectIndex(FI)))
      Candidates.push_back(CandidateInfo(MI, Def, FI));
  }
}

/// HoistRegionPostRA - Walk the specified region of the CFG and hoist loop
/// invariants out to the preheader.
void MachineLICM::HoistRegionPostRA() {
  MachineBasicBlock *Preheader = getCurPreheader();
  if (!Preheader)
    return;

  unsigned NumRegs = TRI->getNumRegs();
  BitVector PhysRegDefs(NumRegs); // Regs defined once in the loop.
  BitVector PhysRegClobbers(NumRegs); // Regs defined more than once.

  SmallVector<CandidateInfo, 32> Candidates;
  SmallSet<int, 32> StoredFIs;

  // Walk the entire region, count number of defs for each register, and
  // collect potential LICM candidates.
  const std::vector<MachineBasicBlock *> &Blocks = CurLoop->getBlocks();
  for (unsigned i = 0, e = Blocks.size(); i != e; ++i) {
    MachineBasicBlock *BB = Blocks[i];

    // If the header of the loop containing this basic block is a landing pad,
    // then don't try to hoist instructions out of this loop.
    const MachineLoop *ML = MLI->getLoopFor(BB);
    if (ML && ML->getHeader()->isLandingPad()) continue;

    // Conservatively treat live-in's as an external def.
    // FIXME: That means a reload that're reused in successor block(s) will not
    // be LICM'ed.
    for (MachineBasicBlock::livein_iterator I = BB->livein_begin(),
           E = BB->livein_end(); I != E; ++I) {
      unsigned Reg = *I;
      for (MCRegAliasIterator AI(Reg, TRI, true); AI.isValid(); ++AI)
        PhysRegDefs.set(*AI);
    }

    SpeculationState = SpeculateUnknown;
    for (MachineBasicBlock::iterator
           MII = BB->begin(), E = BB->end(); MII != E; ++MII) {
      MachineInstr *MI = &*MII;
      ProcessMI(MI, PhysRegDefs, PhysRegClobbers, StoredFIs, Candidates);
    }
  }

  // Gather the registers read / clobbered by the terminator.
  BitVector TermRegs(NumRegs);
  MachineBasicBlock::iterator TI = Preheader->getFirstTerminator();
  if (TI != Preheader->end()) {
    for (unsigned i = 0, e = TI->getNumOperands(); i != e; ++i) {
      const MachineOperand &MO = TI->getOperand(i);
      if (!MO.isReg())
        continue;
      unsigned Reg = MO.getReg();
      if (!Reg)
        continue;
      for (MCRegAliasIterator AI(Reg, TRI, true); AI.isValid(); ++AI)
        TermRegs.set(*AI);
    }
  }

  // Now evaluate whether the potential candidates qualify.
  // 1. Check if the candidate defined register is defined by another
  //    instruction in the loop.
  // 2. If the candidate is a load from stack slot (always true for now),
  //    check if the slot is stored anywhere in the loop.
  // 3. Make sure candidate def should not clobber
  //    registers read by the terminator. Similarly its def should not be
  //    clobbered by the terminator.
  for (unsigned i = 0, e = Candidates.size(); i != e; ++i) {
    if (Candidates[i].FI != INT_MIN &&
        StoredFIs.count(Candidates[i].FI))
      continue;

    unsigned Def = Candidates[i].Def;
    if (!PhysRegClobbers.test(Def) && !TermRegs.test(Def)) {
      bool Safe = true;
      MachineInstr *MI = Candidates[i].MI;
      for (unsigned j = 0, ee = MI->getNumOperands(); j != ee; ++j) {
        const MachineOperand &MO = MI->getOperand(j);
        if (!MO.isReg() || MO.isDef() || !MO.getReg())
          continue;
        unsigned Reg = MO.getReg();
        if (PhysRegDefs.test(Reg) ||
            PhysRegClobbers.test(Reg)) {
          // If it's using a non-loop-invariant register, then it's obviously
          // not safe to hoist.
          Safe = false;
          break;
        }
      }
      if (Safe)
        HoistPostRA(MI, Candidates[i].Def);
    }
  }
}

/// AddToLiveIns - Add register 'Reg' to the livein sets of BBs in the current
/// loop, and make sure it is not killed by any instructions in the loop.
void MachineLICM::AddToLiveIns(unsigned Reg) {
  const std::vector<MachineBasicBlock *> &Blocks = CurLoop->getBlocks();
  for (unsigned i = 0, e = Blocks.size(); i != e; ++i) {
    MachineBasicBlock *BB = Blocks[i];
    if (!BB->isLiveIn(Reg))
      BB->addLiveIn(Reg);
    for (MachineBasicBlock::iterator
           MII = BB->begin(), E = BB->end(); MII != E; ++MII) {
      MachineInstr *MI = &*MII;
      for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
        MachineOperand &MO = MI->getOperand(i);
        if (!MO.isReg() || !MO.getReg() || MO.isDef()) continue;
        if (MO.getReg() == Reg || TRI->isSuperRegister(Reg, MO.getReg()))
          MO.setIsKill(false);
      }
    }
  }
}

/// HoistPostRA - When an instruction is found to only use loop invariant
/// operands that is safe to hoist, this instruction is called to do the
/// dirty work.
void MachineLICM::HoistPostRA(MachineInstr *MI, unsigned Def) {
  MachineBasicBlock *Preheader = getCurPreheader();

  // Now move the instructions to the predecessor, inserting it before any
  // terminator instructions.
  DEBUG(dbgs() << "Hoisting to BB#" << Preheader->getNumber() << " from BB#"
               << MI->getParent()->getNumber() << ": " << *MI);

  // Splice the instruction to the preheader.
  MachineBasicBlock *MBB = MI->getParent();
  Preheader->splice(Preheader->getFirstTerminator(), MBB, MI);

  // Add register to livein list to all the BBs in the current loop since a
  // loop invariant must be kept live throughout the whole loop. This is
  // important to ensure later passes do not scavenge the def register.
  AddToLiveIns(Def);

  ++NumPostRAHoisted;
  Changed = true;
}

// IsGuaranteedToExecute - Check if this mbb is guaranteed to execute.
// If not then a load from this mbb may not be safe to hoist.
bool MachineLICM::IsGuaranteedToExecute(MachineBasicBlock *BB) {
  if (SpeculationState != SpeculateUnknown)
    return SpeculationState == SpeculateFalse;

  if (BB != CurLoop->getHeader()) {
    // Check loop exiting blocks.
    SmallVector<MachineBasicBlock*, 8> CurrentLoopExitingBlocks;
    CurLoop->getExitingBlocks(CurrentLoopExitingBlocks);
    for (unsigned i = 0, e = CurrentLoopExitingBlocks.size(); i != e; ++i)
      if (!DT->dominates(BB, CurrentLoopExitingBlocks[i])) {
        SpeculationState = SpeculateTrue;
        return false;
      }
  }

  SpeculationState = SpeculateFalse;
  return true;
}

void MachineLICM::EnterScope(MachineBasicBlock *MBB) {
  DEBUG(dbgs() << "Entering: " << MBB->getName() << '\n');

  // Remember livein register pressure.
  BackTrace.push_back(RegPressure);
}

void MachineLICM::ExitScope(MachineBasicBlock *MBB) {
  DEBUG(dbgs() << "Exiting: " << MBB->getName() << '\n');
  BackTrace.pop_back();
}

/// ExitScopeIfDone - Destroy scope for the MBB that corresponds to the given
/// dominator tree node if its a leaf or all of its children are done. Walk
/// up the dominator tree to destroy ancestors which are now done.
void MachineLICM::ExitScopeIfDone(MachineDomTreeNode *Node,
                DenseMap<MachineDomTreeNode*, unsigned> &OpenChildren,
                DenseMap<MachineDomTreeNode*, MachineDomTreeNode*> &ParentMap) {
  if (OpenChildren[Node])
    return;

  // Pop scope.
  ExitScope(Node->getBlock());

  // Now traverse upwards to pop ancestors whose offsprings are all done.
  while (MachineDomTreeNode *Parent = ParentMap[Node]) {
    unsigned Left = --OpenChildren[Parent];
    if (Left != 0)
      break;
    ExitScope(Parent->getBlock());
    Node = Parent;
  }
}

/// HoistOutOfLoop - Walk the specified loop in the CFG (defined by all
/// blocks dominated by the specified header block, and that are in the
/// current loop) in depth first order w.r.t the DominatorTree. This allows
/// us to visit definitions before uses, allowing us to hoist a loop body in
/// one pass without iteration.
///
void MachineLICM::HoistOutOfLoop(MachineDomTreeNode *HeaderN) {
  MachineBasicBlock *Preheader = getCurPreheader();
  if (!Preheader)
    return;

  SmallVector<MachineDomTreeNode*, 32> Scopes;
  SmallVector<MachineDomTreeNode*, 8> WorkList;
  DenseMap<MachineDomTreeNode*, MachineDomTreeNode*> ParentMap;
  DenseMap<MachineDomTreeNode*, unsigned> OpenChildren;

  // Perform a DFS walk to determine the order of visit.
  WorkList.push_back(HeaderN);
  while (!WorkList.empty()) {
    MachineDomTreeNode *Node = WorkList.pop_back_val();
    assert(Node && "Null dominator tree node?");
    MachineBasicBlock *BB = Node->getBlock();

    // If the header of the loop containing this basic block is a landing pad,
    // then don't try to hoist instructions out of this loop.
    const MachineLoop *ML = MLI->getLoopFor(BB);
    if (ML && ML->getHeader()->isLandingPad())
      continue;

    // If this subregion is not in the top level loop at all, exit.
    if (!CurLoop->contains(BB))
      continue;

    Scopes.push_back(Node);
    const std::vector<MachineDomTreeNode*> &Children = Node->getChildren();
    unsigned NumChildren = Children.size();

    // Don't hoist things out of a large switch statement.  This often causes
    // code to be hoisted that wasn't going to be executed, and increases
    // register pressure in a situation where it's likely to matter.
    if (BB->succ_size() >= 25)
      NumChildren = 0;

    OpenChildren[Node] = NumChildren;
    // Add children in reverse order as then the next popped worklist node is
    // the first child of this node.  This means we ultimately traverse the
    // DOM tree in exactly the same order as if we'd recursed.
    for (int i = (int)NumChildren-1; i >= 0; --i) {
      MachineDomTreeNode *Child = Children[i];
      ParentMap[Child] = Node;
      WorkList.push_back(Child);
    }
  }

  if (Scopes.size() == 0)
    return;

  // Compute registers which are livein into the loop headers.
  RegSeen.clear();
  BackTrace.clear();
  InitRegPressure(Preheader);

  // Now perform LICM.
  for (unsigned i = 0, e = Scopes.size(); i != e; ++i) {
    MachineDomTreeNode *Node = Scopes[i];
    MachineBasicBlock *MBB = Node->getBlock();

    EnterScope(MBB);

    // Process the block
    SpeculationState = SpeculateUnknown;
    for (MachineBasicBlock::iterator
         MII = MBB->begin(), E = MBB->end(); MII != E; ) {
      MachineBasicBlock::iterator NextMII = MII; ++NextMII;
      MachineInstr *MI = &*MII;
      if (!Hoist(MI, Preheader))
        UpdateRegPressure(MI);
      MII = NextMII;
    }

    // If it's a leaf node, it's done. Traverse upwards to pop ancestors.
    ExitScopeIfDone(Node, OpenChildren, ParentMap);
  }
}

void MachineLICM::SinkIntoLoop() {
  MachineBasicBlock *Preheader = getCurPreheader();
  if (!Preheader)
    return;

  SmallVector<MachineInstr *, 8> Candidates;
  for (MachineBasicBlock::instr_iterator I = Preheader->instr_begin();
       I != Preheader->instr_end(); ++I) {
    // We need to ensure that we can safely move this instruction into the loop.
    // As such, it must not have side-effects, e.g. such as a call has.  
    if (IsLoopInvariantInst(*I) && !HasLoopPHIUse(I))
      Candidates.push_back(I);
  }

  for (MachineInstr *I : Candidates) {
    const MachineOperand &MO = I->getOperand(0);
    if (!MO.isDef() || !MO.isReg() || !MO.getReg())
      continue;
    if (!MRI->hasOneDef(MO.getReg()))
      continue;
    bool CanSink = true;
    MachineBasicBlock *B = nullptr;
    for (MachineInstr &MI : MRI->use_instructions(MO.getReg())) {
      // FIXME: Come up with a proper cost model that estimates whether sinking
      // the instruction (and thus possibly executing it on every loop
      // iteration) is more expensive than a register.
      // For now assumes that copies are cheap and thus almost always worth it.
      if (!MI.isCopy()) {
        CanSink = false;
        break;
      }
      if (!B) {
        B = MI.getParent();
        continue;
      }
      B = DT->findNearestCommonDominator(B, MI.getParent());
      if (!B) {
        CanSink = false;
        break;
      }
    }
    if (!CanSink || !B || B == Preheader)
      continue;
    B->splice(B->getFirstNonPHI(), Preheader, I);
  }
}

static bool isOperandKill(const MachineOperand &MO, MachineRegisterInfo *MRI) {
  return MO.isKill() || MRI->hasOneNonDBGUse(MO.getReg());
}

/// InitRegPressure - Find all virtual register references that are liveout of
/// the preheader to initialize the starting "register pressure". Note this
/// does not count live through (livein but not used) registers.
void MachineLICM::InitRegPressure(MachineBasicBlock *BB) {
  std::fill(RegPressure.begin(), RegPressure.end(), 0);

  // If the preheader has only a single predecessor and it ends with a
  // fallthrough or an unconditional branch, then scan its predecessor for live
  // defs as well. This happens whenever the preheader is created by splitting
  // the critical edge from the loop predecessor to the loop header.
  if (BB->pred_size() == 1) {
    MachineBasicBlock *TBB = nullptr, *FBB = nullptr;
    SmallVector<MachineOperand, 4> Cond;
    if (!TII->AnalyzeBranch(*BB, TBB, FBB, Cond, false) && Cond.empty())
      InitRegPressure(*BB->pred_begin());
  }

  for (const MachineInstr &MI : *BB)
    UpdateRegPressure(&MI, /*ConsiderUnseenAsDef=*/true);
}

/// UpdateRegPressure - Update estimate of register pressure after the
/// specified instruction.
void MachineLICM::UpdateRegPressure(const MachineInstr *MI,
                                    bool ConsiderUnseenAsDef) {
  auto Cost = calcRegisterCost(MI, /*ConsiderSeen=*/true, ConsiderUnseenAsDef);
  for (const auto &RPIdAndCost : Cost) {
    unsigned Class = RPIdAndCost.first;
    if (static_cast<int>(RegPressure[Class]) < -RPIdAndCost.second)
      RegPressure[Class] = 0;
    else
      RegPressure[Class] += RPIdAndCost.second;
  }
}

DenseMap<unsigned, int>
MachineLICM::calcRegisterCost(const MachineInstr *MI, bool ConsiderSeen,
                              bool ConsiderUnseenAsDef) {
  DenseMap<unsigned, int> Cost;
  if (MI->isImplicitDef())
    return Cost;
  for (unsigned i = 0, e = MI->getDesc().getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg() || MO.isImplicit())
      continue;
    unsigned Reg = MO.getReg();
    if (!TargetRegisterInfo::isVirtualRegister(Reg))
      continue;

    // FIXME: It seems bad to use RegSeen only for some of these calculations.
    bool isNew = ConsiderSeen ? RegSeen.insert(Reg).second : false;
    const TargetRegisterClass *RC = MRI->getRegClass(Reg);

    RegClassWeight W = TRI->getRegClassWeight(RC);
    int RCCost = 0;
    if (MO.isDef())
      RCCost = W.RegWeight;
    else {
      bool isKill = isOperandKill(MO, MRI);
      if (isNew && !isKill && ConsiderUnseenAsDef)
        // Haven't seen this, it must be a livein.
        RCCost = W.RegWeight;
      else if (!isNew && isKill)
        RCCost = -W.RegWeight;
    }
    if (RCCost == 0)
      continue;
    const int *PS = TRI->getRegClassPressureSets(RC);
    for (; *PS != -1; ++PS) {
      if (Cost.find(*PS) == Cost.end())
        Cost[*PS] = RCCost;
      else
        Cost[*PS] += RCCost;
    }
  }
  return Cost;
}

/// isLoadFromGOTOrConstantPool - Return true if this machine instruction
/// loads from global offset table or constant pool.
static bool isLoadFromGOTOrConstantPool(MachineInstr &MI) {
  assert (MI.mayLoad() && "Expected MI that loads!");
  for (MachineInstr::mmo_iterator I = MI.memoperands_begin(),
         E = MI.memoperands_end(); I != E; ++I) {
    if (const PseudoSourceValue *PSV = (*I)->getPseudoValue()) {
      if (PSV == PSV->getGOT() || PSV == PSV->getConstantPool())
        return true;
    }
  }
  return false;
}

/// IsLICMCandidate - Returns true if the instruction may be a suitable
/// candidate for LICM. e.g. If the instruction is a call, then it's obviously
/// not safe to hoist it.
bool MachineLICM::IsLICMCandidate(MachineInstr &I) {
  // Check if it's safe to move the instruction.
  bool DontMoveAcrossStore = true;
  if (!I.isSafeToMove(AA, DontMoveAcrossStore))
    return false;

  // If it is load then check if it is guaranteed to execute by making sure that
  // it dominates all exiting blocks. If it doesn't, then there is a path out of
  // the loop which does not execute this load, so we can't hoist it. Loads
  // from constant memory are not safe to speculate all the time, for example
  // indexed load from a jump table.
  // Stores and side effects are already checked by isSafeToMove.
  if (I.mayLoad() && !isLoadFromGOTOrConstantPool(I) &&
      !IsGuaranteedToExecute(I.getParent()))
    return false;

  return true;
}

/// IsLoopInvariantInst - Returns true if the instruction is loop
/// invariant. I.e., all virtual register operands are defined outside of the
/// loop, physical registers aren't accessed explicitly, and there are no side
/// effects that aren't captured by the operands or other flags.
///
bool MachineLICM::IsLoopInvariantInst(MachineInstr &I) {
  if (!IsLICMCandidate(I))
    return false;

  // The instruction is loop invariant if all of its operands are.
  for (unsigned i = 0, e = I.getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = I.getOperand(i);

    if (!MO.isReg())
      continue;

    unsigned Reg = MO.getReg();
    if (Reg == 0) continue;

    // Don't hoist an instruction that uses or defines a physical register.
    if (TargetRegisterInfo::isPhysicalRegister(Reg)) {
      if (MO.isUse()) {
        // If the physreg has no defs anywhere, it's just an ambient register
        // and we can freely move its uses. Alternatively, if it's allocatable,
        // it could get allocated to something with a def during allocation.
        if (!MRI->isConstantPhysReg(Reg, *I.getParent()->getParent()))
          return false;
        // Otherwise it's safe to move.
        continue;
      } else if (!MO.isDead()) {
        // A def that isn't dead. We can't move it.
        return false;
      } else if (CurLoop->getHeader()->isLiveIn(Reg)) {
        // If the reg is live into the loop, we can't hoist an instruction
        // which would clobber it.
        return false;
      }
    }

    if (!MO.isUse())
      continue;

    assert(MRI->getVRegDef(Reg) &&
           "Machine instr not mapped for this vreg?!");

    // If the loop contains the definition of an operand, then the instruction
    // isn't loop invariant.
    if (CurLoop->contains(MRI->getVRegDef(Reg)))
      return false;
  }

  // If we got this far, the instruction is loop invariant!
  return true;
}


/// HasLoopPHIUse - Return true if the specified instruction is used by a
/// phi node and hoisting it could cause a copy to be inserted.
bool MachineLICM::HasLoopPHIUse(const MachineInstr *MI) const {
  SmallVector<const MachineInstr*, 8> Work(1, MI);
  do {
    MI = Work.pop_back_val();
    for (const MachineOperand &MO : MI->operands()) {
      if (!MO.isReg() || !MO.isDef())
        continue;
      unsigned Reg = MO.getReg();
      if (!TargetRegisterInfo::isVirtualRegister(Reg))
        continue;
      for (MachineInstr &UseMI : MRI->use_instructions(Reg)) {
        // A PHI may cause a copy to be inserted.
        if (UseMI.isPHI()) {
          // A PHI inside the loop causes a copy because the live range of Reg is
          // extended across the PHI.
          if (CurLoop->contains(&UseMI))
            return true;
          // A PHI in an exit block can cause a copy to be inserted if the PHI
          // has multiple predecessors in the loop with different values.
          // For now, approximate by rejecting all exit blocks.
          if (isExitBlock(UseMI.getParent()))
            return true;
          continue;
        }
        // Look past copies as well.
        if (UseMI.isCopy() && CurLoop->contains(&UseMI))
          Work.push_back(&UseMI);
      }
    }
  } while (!Work.empty());
  return false;
}

/// HasHighOperandLatency - Compute operand latency between a def of 'Reg'
/// and an use in the current loop, return true if the target considered
/// it 'high'.
bool MachineLICM::HasHighOperandLatency(MachineInstr &MI,
                                        unsigned DefIdx, unsigned Reg) const {
  if (MRI->use_nodbg_empty(Reg))
    return false;

  for (MachineInstr &UseMI : MRI->use_nodbg_instructions(Reg)) {
    if (UseMI.isCopyLike())
      continue;
    if (!CurLoop->contains(UseMI.getParent()))
      continue;
    for (unsigned i = 0, e = UseMI.getNumOperands(); i != e; ++i) {
      const MachineOperand &MO = UseMI.getOperand(i);
      if (!MO.isReg() || !MO.isUse())
        continue;
      unsigned MOReg = MO.getReg();
      if (MOReg != Reg)
        continue;

      if (TII->hasHighOperandLatency(SchedModel, MRI, &MI, DefIdx, &UseMI, i))
        return true;
    }

    // Only look at the first in loop use.
    break;
  }

  return false;
}

/// IsCheapInstruction - Return true if the instruction is marked "cheap" or
/// the operand latency between its def and a use is one or less.
bool MachineLICM::IsCheapInstruction(MachineInstr &MI) const {
  if (TII->isAsCheapAsAMove(&MI) || MI.isCopyLike())
    return true;

  bool isCheap = false;
  unsigned NumDefs = MI.getDesc().getNumDefs();
  for (unsigned i = 0, e = MI.getNumOperands(); NumDefs && i != e; ++i) {
    MachineOperand &DefMO = MI.getOperand(i);
    if (!DefMO.isReg() || !DefMO.isDef())
      continue;
    --NumDefs;
    unsigned Reg = DefMO.getReg();
    if (TargetRegisterInfo::isPhysicalRegister(Reg))
      continue;

    if (!TII->hasLowDefLatency(SchedModel, &MI, i))
      return false;
    isCheap = true;
  }

  return isCheap;
}

/// CanCauseHighRegPressure - Visit BBs from header to current BB, check
/// if hoisting an instruction of the given cost matrix can cause high
/// register pressure.
bool MachineLICM::CanCauseHighRegPressure(const DenseMap<unsigned, int>& Cost,
                                          bool CheapInstr) {
  for (const auto &RPIdAndCost : Cost) {
    if (RPIdAndCost.second <= 0)
      continue;

    unsigned Class = RPIdAndCost.first;
    int Limit = RegLimit[Class];

    // Don't hoist cheap instructions if they would increase register pressure,
    // even if we're under the limit.
    if (CheapInstr && !HoistCheapInsts)
      return true;

    for (const auto &RP : BackTrace)
      if (static_cast<int>(RP[Class]) + RPIdAndCost.second >= Limit)
        return true;
  }

  return false;
}

/// UpdateBackTraceRegPressure - Traverse the back trace from header to the
/// current block and update their register pressures to reflect the effect
/// of hoisting MI from the current block to the preheader.
void MachineLICM::UpdateBackTraceRegPressure(const MachineInstr *MI) {
  // First compute the 'cost' of the instruction, i.e. its contribution
  // to register pressure.
  auto Cost = calcRegisterCost(MI, /*ConsiderSeen=*/false,
                               /*ConsiderUnseenAsDef=*/false);

  // Update register pressure of blocks from loop header to current block.
  for (auto &RP : BackTrace)
    for (const auto &RPIdAndCost : Cost)
      RP[RPIdAndCost.first] += RPIdAndCost.second;
}

/// IsProfitableToHoist - Return true if it is potentially profitable to hoist
/// the given loop invariant.
bool MachineLICM::IsProfitableToHoist(MachineInstr &MI) {
  if (MI.isImplicitDef())
    return true;

  // Besides removing computation from the loop, hoisting an instruction has
  // these effects:
  //
  // - The value defined by the instruction becomes live across the entire
  //   loop. This increases register pressure in the loop.
  //
  // - If the value is used by a PHI in the loop, a copy will be required for
  //   lowering the PHI after extending the live range.
  //
  // - When hoisting the last use of a value in the loop, that value no longer
  //   needs to be live in the loop. This lowers register pressure in the loop.

  bool CheapInstr = IsCheapInstruction(MI);
  bool CreatesCopy = HasLoopPHIUse(&MI);

  // Don't hoist a cheap instruction if it would create a copy in the loop.
  if (CheapInstr && CreatesCopy) {
    DEBUG(dbgs() << "Won't hoist cheap instr with loop PHI use: " << MI);
    return false;
  }

  // Rematerializable instructions should always be hoisted since the register
  // allocator can just pull them down again when needed.
  if (TII->isTriviallyReMaterializable(&MI, AA))
    return true;

  // FIXME: If there are long latency loop-invariant instructions inside the
  // loop at this point, why didn't the optimizer's LICM hoist them?
  for (unsigned i = 0, e = MI.getDesc().getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI.getOperand(i);
    if (!MO.isReg() || MO.isImplicit())
      continue;
    unsigned Reg = MO.getReg();
    if (!TargetRegisterInfo::isVirtualRegister(Reg))
      continue;
    if (MO.isDef() && HasHighOperandLatency(MI, i, Reg)) {
      DEBUG(dbgs() << "Hoist High Latency: " << MI);
      ++NumHighLatency;
      return true;
    }
  }

  // Estimate register pressure to determine whether to LICM the instruction.
  // In low register pressure situation, we can be more aggressive about
  // hoisting. Also, favors hoisting long latency instructions even in
  // moderately high pressure situation.
  // Cheap instructions will only be hoisted if they don't increase register
  // pressure at all.
  auto Cost = calcRegisterCost(&MI, /*ConsiderSeen=*/false,
                               /*ConsiderUnseenAsDef=*/false);

  // Visit BBs from header to current BB, if hoisting this doesn't cause
  // high register pressure, then it's safe to proceed.
  if (!CanCauseHighRegPressure(Cost, CheapInstr)) {
    DEBUG(dbgs() << "Hoist non-reg-pressure: " << MI);
    ++NumLowRP;
    return true;
  }

  // Don't risk increasing register pressure if it would create copies.
  if (CreatesCopy) {
    DEBUG(dbgs() << "Won't hoist instr with loop PHI use: " << MI);
    return false;
  }

  // Do not "speculate" in high register pressure situation. If an
  // instruction is not guaranteed to be executed in the loop, it's best to be
  // conservative.
  if (AvoidSpeculation &&
      (!IsGuaranteedToExecute(MI.getParent()) && !MayCSE(&MI))) {
    DEBUG(dbgs() << "Won't speculate: " << MI);
    return false;
  }

  // High register pressure situation, only hoist if the instruction is going
  // to be remat'ed.
  if (!TII->isTriviallyReMaterializable(&MI, AA) &&
      !MI.isInvariantLoad(AA)) {
    DEBUG(dbgs() << "Can't remat / high reg-pressure: " << MI);
    return false;
  }

  return true;
}

MachineInstr *MachineLICM::ExtractHoistableLoad(MachineInstr *MI) {
  // Don't unfold simple loads.
  if (MI->canFoldAsLoad())
    return nullptr;

  // If not, we may be able to unfold a load and hoist that.
  // First test whether the instruction is loading from an amenable
  // memory location.
  if (!MI->isInvariantLoad(AA))
    return nullptr;

  // Next determine the register class for a temporary register.
  unsigned LoadRegIndex;
  unsigned NewOpc =
    TII->getOpcodeAfterMemoryUnfold(MI->getOpcode(),
                                    /*UnfoldLoad=*/true,
                                    /*UnfoldStore=*/false,
                                    &LoadRegIndex);
  if (NewOpc == 0) return nullptr;
  const MCInstrDesc &MID = TII->get(NewOpc);
  if (MID.getNumDefs() != 1) return nullptr;
  MachineFunction &MF = *MI->getParent()->getParent();
  const TargetRegisterClass *RC = TII->getRegClass(MID, LoadRegIndex, TRI, MF);
  // Ok, we're unfolding. Create a temporary register and do the unfold.
  unsigned Reg = MRI->createVirtualRegister(RC);

  SmallVector<MachineInstr *, 2> NewMIs;
  bool Success =
    TII->unfoldMemoryOperand(MF, MI, Reg,
                             /*UnfoldLoad=*/true, /*UnfoldStore=*/false,
                             NewMIs);
  (void)Success;
  assert(Success &&
         "unfoldMemoryOperand failed when getOpcodeAfterMemoryUnfold "
         "succeeded!");
  assert(NewMIs.size() == 2 &&
         "Unfolded a load into multiple instructions!");
  MachineBasicBlock *MBB = MI->getParent();
  MachineBasicBlock::iterator Pos = MI;
  MBB->insert(Pos, NewMIs[0]);
  MBB->insert(Pos, NewMIs[1]);
  // If unfolding produced a load that wasn't loop-invariant or profitable to
  // hoist, discard the new instructions and bail.
  if (!IsLoopInvariantInst(*NewMIs[0]) || !IsProfitableToHoist(*NewMIs[0])) {
    NewMIs[0]->eraseFromParent();
    NewMIs[1]->eraseFromParent();
    return nullptr;
  }

  // Update register pressure for the unfolded instruction.
  UpdateRegPressure(NewMIs[1]);

  // Otherwise we successfully unfolded a load that we can hoist.
  MI->eraseFromParent();
  return NewMIs[0];
}

void MachineLICM::InitCSEMap(MachineBasicBlock *BB) {
  for (MachineBasicBlock::iterator I = BB->begin(),E = BB->end(); I != E; ++I) {
    const MachineInstr *MI = &*I;
    unsigned Opcode = MI->getOpcode();
    CSEMap[Opcode].push_back(MI);
  }
}

const MachineInstr*
MachineLICM::LookForDuplicate(const MachineInstr *MI,
                              std::vector<const MachineInstr*> &PrevMIs) {
  for (unsigned i = 0, e = PrevMIs.size(); i != e; ++i) {
    const MachineInstr *PrevMI = PrevMIs[i];
    if (TII->produceSameValue(MI, PrevMI, (PreRegAlloc ? MRI : nullptr)))
      return PrevMI;
  }
  return nullptr;
}

bool MachineLICM::EliminateCSE(MachineInstr *MI,
          DenseMap<unsigned, std::vector<const MachineInstr*> >::iterator &CI) {
  // Do not CSE implicit_def so ProcessImplicitDefs can properly propagate
  // the undef property onto uses.
  if (CI == CSEMap.end() || MI->isImplicitDef())
    return false;

  if (const MachineInstr *Dup = LookForDuplicate(MI, CI->second)) {
    DEBUG(dbgs() << "CSEing " << *MI << " with " << *Dup);

    // Replace virtual registers defined by MI by their counterparts defined
    // by Dup.
    SmallVector<unsigned, 2> Defs;
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      const MachineOperand &MO = MI->getOperand(i);

      // Physical registers may not differ here.
      assert((!MO.isReg() || MO.getReg() == 0 ||
              !TargetRegisterInfo::isPhysicalRegister(MO.getReg()) ||
              MO.getReg() == Dup->getOperand(i).getReg()) &&
             "Instructions with different phys regs are not identical!");

      if (MO.isReg() && MO.isDef() &&
          !TargetRegisterInfo::isPhysicalRegister(MO.getReg()))
        Defs.push_back(i);
    }

    SmallVector<const TargetRegisterClass*, 2> OrigRCs;
    for (unsigned i = 0, e = Defs.size(); i != e; ++i) {
      unsigned Idx = Defs[i];
      unsigned Reg = MI->getOperand(Idx).getReg();
      unsigned DupReg = Dup->getOperand(Idx).getReg();
      OrigRCs.push_back(MRI->getRegClass(DupReg));

      if (!MRI->constrainRegClass(DupReg, MRI->getRegClass(Reg))) {
        // Restore old RCs if more than one defs.
        for (unsigned j = 0; j != i; ++j)
          MRI->setRegClass(Dup->getOperand(Defs[j]).getReg(), OrigRCs[j]);
        return false;
      }
    }

    for (unsigned i = 0, e = Defs.size(); i != e; ++i) {
      unsigned Idx = Defs[i];
      unsigned Reg = MI->getOperand(Idx).getReg();
      unsigned DupReg = Dup->getOperand(Idx).getReg();
      MRI->replaceRegWith(Reg, DupReg);
      MRI->clearKillFlags(DupReg);
    }

    MI->eraseFromParent();
    ++NumCSEed;
    return true;
  }
  return false;
}

/// MayCSE - Return true if the given instruction will be CSE'd if it's
/// hoisted out of the loop.
bool MachineLICM::MayCSE(MachineInstr *MI) {
  unsigned Opcode = MI->getOpcode();
  DenseMap<unsigned, std::vector<const MachineInstr*> >::iterator
    CI = CSEMap.find(Opcode);
  // Do not CSE implicit_def so ProcessImplicitDefs can properly propagate
  // the undef property onto uses.
  if (CI == CSEMap.end() || MI->isImplicitDef())
    return false;

  return LookForDuplicate(MI, CI->second) != nullptr;
}

/// Hoist - When an instruction is found to use only loop invariant operands
/// that are safe to hoist, this instruction is called to do the dirty work.
///
bool MachineLICM::Hoist(MachineInstr *MI, MachineBasicBlock *Preheader) {
  // First check whether we should hoist this instruction.
  if (!IsLoopInvariantInst(*MI) || !IsProfitableToHoist(*MI)) {
    // If not, try unfolding a hoistable load.
    MI = ExtractHoistableLoad(MI);
    if (!MI) return false;
  }

  // Now move the instructions to the predecessor, inserting it before any
  // terminator instructions.
  DEBUG({
      dbgs() << "Hoisting " << *MI;
      if (Preheader->getBasicBlock())
        dbgs() << " to MachineBasicBlock "
               << Preheader->getName();
      if (MI->getParent()->getBasicBlock())
        dbgs() << " from MachineBasicBlock "
               << MI->getParent()->getName();
      dbgs() << "\n";
    });

  // If this is the first instruction being hoisted to the preheader,
  // initialize the CSE map with potential common expressions.
  if (FirstInLoop) {
    InitCSEMap(Preheader);
    FirstInLoop = false;
  }

  // Look for opportunity to CSE the hoisted instruction.
  unsigned Opcode = MI->getOpcode();
  DenseMap<unsigned, std::vector<const MachineInstr*> >::iterator
    CI = CSEMap.find(Opcode);
  if (!EliminateCSE(MI, CI)) {
    // Otherwise, splice the instruction to the preheader.
    Preheader->splice(Preheader->getFirstTerminator(),MI->getParent(),MI);

    // Update register pressure for BBs from header to this block.
    UpdateBackTraceRegPressure(MI);

    // Clear the kill flags of any register this instruction defines,
    // since they may need to be live throughout the entire loop
    // rather than just live for part of it.
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      MachineOperand &MO = MI->getOperand(i);
      if (MO.isReg() && MO.isDef() && !MO.isDead())
        MRI->clearKillFlags(MO.getReg());
    }

    // Add to the CSE map.
    if (CI != CSEMap.end())
      CI->second.push_back(MI);
    else
      CSEMap[Opcode].push_back(MI);
  }

  ++NumHoisted;
  Changed = true;

  return true;
}

MachineBasicBlock *MachineLICM::getCurPreheader() {
  // Determine the block to which to hoist instructions. If we can't find a
  // suitable loop predecessor, we can't do any hoisting.

  // If we've tried to get a preheader and failed, don't try again.
  if (CurPreheader == reinterpret_cast<MachineBasicBlock *>(-1))
    return nullptr;

  if (!CurPreheader) {
    CurPreheader = CurLoop->getLoopPreheader();
    if (!CurPreheader) {
      MachineBasicBlock *Pred = CurLoop->getLoopPredecessor();
      if (!Pred) {
        CurPreheader = reinterpret_cast<MachineBasicBlock *>(-1);
        return nullptr;
      }

      CurPreheader = Pred->SplitCriticalEdge(CurLoop->getHeader(), this);
      if (!CurPreheader) {
        CurPreheader = reinterpret_cast<MachineBasicBlock *>(-1);
        return nullptr;
      }
    }
  }
  return CurPreheader;
}
