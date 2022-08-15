//===-- TailDuplication.cpp - Duplicate blocks into predecessors' tails ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass duplicates basic blocks ending in unconditional branches into
// the tails of their predecessors.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/Passes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineBranchProbabilityInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/MachineSSAUpdater.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"
using namespace llvm;

#define DEBUG_TYPE "tailduplication"

STATISTIC(NumTails     , "Number of tails duplicated");
STATISTIC(NumTailDups  , "Number of tail duplicated blocks");
STATISTIC(NumInstrDups , "Additional instructions due to tail duplication");
STATISTIC(NumDeadBlocks, "Number of dead blocks removed");
STATISTIC(NumAddedPHIs , "Number of phis added");

// Heuristic for tail duplication.
static cl::opt<unsigned>
TailDuplicateSize("tail-dup-size",
                  cl::desc("Maximum instructions to consider tail duplicating"),
                  cl::init(2), cl::Hidden);

static cl::opt<bool>
TailDupVerify("tail-dup-verify",
              cl::desc("Verify sanity of PHI instructions during taildup"),
              cl::init(false), cl::Hidden);

static cl::opt<unsigned>
TailDupLimit("tail-dup-limit", cl::init(~0U), cl::Hidden);

typedef std::vector<std::pair<MachineBasicBlock*,unsigned> > AvailableValsTy;

namespace {
  /// TailDuplicatePass - Perform tail duplication.
  class TailDuplicatePass : public MachineFunctionPass {
    const TargetInstrInfo *TII;
    const TargetRegisterInfo *TRI;
    const MachineBranchProbabilityInfo *MBPI;
    MachineModuleInfo *MMI;
    MachineRegisterInfo *MRI;
    std::unique_ptr<RegScavenger> RS;
    bool PreRegAlloc;

    // SSAUpdateVRs - A list of virtual registers for which to update SSA form.
    SmallVector<unsigned, 16> SSAUpdateVRs;

    // SSAUpdateVals - For each virtual register in SSAUpdateVals keep a list of
    // source virtual registers.
    DenseMap<unsigned, AvailableValsTy> SSAUpdateVals;

  public:
    static char ID;
    explicit TailDuplicatePass() :
      MachineFunctionPass(ID), PreRegAlloc(false) {}

    bool runOnMachineFunction(MachineFunction &MF) override;

    void getAnalysisUsage(AnalysisUsage &AU) const override;

  private:
    void AddSSAUpdateEntry(unsigned OrigReg, unsigned NewReg,
                           MachineBasicBlock *BB);
    void ProcessPHI(MachineInstr *MI, MachineBasicBlock *TailBB,
                    MachineBasicBlock *PredBB,
                    DenseMap<unsigned, unsigned> &LocalVRMap,
                    SmallVectorImpl<std::pair<unsigned,unsigned> > &Copies,
                    const DenseSet<unsigned> &UsedByPhi,
                    bool Remove);
    void DuplicateInstruction(MachineInstr *MI,
                              MachineBasicBlock *TailBB,
                              MachineBasicBlock *PredBB,
                              MachineFunction &MF,
                              DenseMap<unsigned, unsigned> &LocalVRMap,
                              const DenseSet<unsigned> &UsedByPhi);
    void UpdateSuccessorsPHIs(MachineBasicBlock *FromBB, bool isDead,
                              SmallVectorImpl<MachineBasicBlock *> &TDBBs,
                              SmallSetVector<MachineBasicBlock*, 8> &Succs);
    bool TailDuplicateBlocks(MachineFunction &MF);
    bool shouldTailDuplicate(const MachineFunction &MF,
                             bool IsSimple, MachineBasicBlock &TailBB);
    bool isSimpleBB(MachineBasicBlock *TailBB);
    bool canCompletelyDuplicateBB(MachineBasicBlock &BB);
    bool duplicateSimpleBB(MachineBasicBlock *TailBB,
                           SmallVectorImpl<MachineBasicBlock *> &TDBBs,
                           const DenseSet<unsigned> &RegsUsedByPhi,
                           SmallVectorImpl<MachineInstr *> &Copies);
    bool TailDuplicate(MachineBasicBlock *TailBB,
                       bool IsSimple,
                       MachineFunction &MF,
                       SmallVectorImpl<MachineBasicBlock *> &TDBBs,
                       SmallVectorImpl<MachineInstr *> &Copies);
    bool TailDuplicateAndUpdate(MachineBasicBlock *MBB,
                                bool IsSimple,
                                MachineFunction &MF);

    void RemoveDeadBlock(MachineBasicBlock *MBB);
  };

  char TailDuplicatePass::ID = 0;
}

char &llvm::TailDuplicateID = TailDuplicatePass::ID;

INITIALIZE_PASS(TailDuplicatePass, "tailduplication", "Tail Duplication",
                false, false)

bool TailDuplicatePass::runOnMachineFunction(MachineFunction &MF) {
  if (skipOptnoneFunction(*MF.getFunction()))
    return false;

  TII = MF.getSubtarget().getInstrInfo();
  TRI = MF.getSubtarget().getRegisterInfo();
  MRI = &MF.getRegInfo();
  MMI = getAnalysisIfAvailable<MachineModuleInfo>();
  MBPI = &getAnalysis<MachineBranchProbabilityInfo>();

  PreRegAlloc = MRI->isSSA();
  RS.reset();
  if (MRI->tracksLiveness() && TRI->trackLivenessAfterRegAlloc(MF))
    RS.reset(new RegScavenger());

  bool MadeChange = false;
  while (TailDuplicateBlocks(MF))
    MadeChange = true;

  return MadeChange;
}

void TailDuplicatePass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<MachineBranchProbabilityInfo>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

static void VerifyPHIs(MachineFunction &MF, bool CheckExtra) {
  for (MachineFunction::iterator I = ++MF.begin(), E = MF.end(); I != E; ++I) {
    MachineBasicBlock *MBB = I;
    SmallSetVector<MachineBasicBlock*, 8> Preds(MBB->pred_begin(),
                                                MBB->pred_end());
    MachineBasicBlock::iterator MI = MBB->begin();
    while (MI != MBB->end()) {
      if (!MI->isPHI())
        break;
      for (SmallSetVector<MachineBasicBlock *, 8>::iterator PI = Preds.begin(),
             PE = Preds.end(); PI != PE; ++PI) {
        MachineBasicBlock *PredBB = *PI;
        bool Found = false;
        for (unsigned i = 1, e = MI->getNumOperands(); i != e; i += 2) {
          MachineBasicBlock *PHIBB = MI->getOperand(i+1).getMBB();
          if (PHIBB == PredBB) {
            Found = true;
            break;
          }
        }
        if (!Found) {
          dbgs() << "Malformed PHI in BB#" << MBB->getNumber() << ": " << *MI;
          dbgs() << "  missing input from predecessor BB#"
                 << PredBB->getNumber() << '\n';
          llvm_unreachable(nullptr);
        }
      }

      for (unsigned i = 1, e = MI->getNumOperands(); i != e; i += 2) {
        MachineBasicBlock *PHIBB = MI->getOperand(i+1).getMBB();
        if (CheckExtra && !Preds.count(PHIBB)) {
          dbgs() << "Warning: malformed PHI in BB#" << MBB->getNumber()
                 << ": " << *MI;
          dbgs() << "  extra input from predecessor BB#"
                 << PHIBB->getNumber() << '\n';
          llvm_unreachable(nullptr);
        }
        if (PHIBB->getNumber() < 0) {
          dbgs() << "Malformed PHI in BB#" << MBB->getNumber() << ": " << *MI;
          dbgs() << "  non-existing BB#" << PHIBB->getNumber() << '\n';
          llvm_unreachable(nullptr);
        }
      }
      ++MI;
    }
  }
}

/// TailDuplicateAndUpdate - Tail duplicate the block and cleanup.
bool
TailDuplicatePass::TailDuplicateAndUpdate(MachineBasicBlock *MBB,
                                          bool IsSimple,
                                          MachineFunction &MF) {
  // Save the successors list.
  SmallSetVector<MachineBasicBlock*, 8> Succs(MBB->succ_begin(),
                                              MBB->succ_end());

  SmallVector<MachineBasicBlock*, 8> TDBBs;
  SmallVector<MachineInstr*, 16> Copies;
  if (!TailDuplicate(MBB, IsSimple, MF, TDBBs, Copies))
    return false;

  ++NumTails;

  SmallVector<MachineInstr*, 8> NewPHIs;
  MachineSSAUpdater SSAUpdate(MF, &NewPHIs);

  // TailBB's immediate successors are now successors of those predecessors
  // which duplicated TailBB. Add the predecessors as sources to the PHI
  // instructions.
  bool isDead = MBB->pred_empty() && !MBB->hasAddressTaken();
  if (PreRegAlloc)
    UpdateSuccessorsPHIs(MBB, isDead, TDBBs, Succs);

  // If it is dead, remove it.
  if (isDead) {
    NumInstrDups -= MBB->size();
    RemoveDeadBlock(MBB);
    ++NumDeadBlocks;
  }

  // Update SSA form.
  if (!SSAUpdateVRs.empty()) {
    for (unsigned i = 0, e = SSAUpdateVRs.size(); i != e; ++i) {
      unsigned VReg = SSAUpdateVRs[i];
      SSAUpdate.Initialize(VReg);

      // If the original definition is still around, add it as an available
      // value.
      MachineInstr *DefMI = MRI->getVRegDef(VReg);
      MachineBasicBlock *DefBB = nullptr;
      if (DefMI) {
        DefBB = DefMI->getParent();
        SSAUpdate.AddAvailableValue(DefBB, VReg);
      }

      // Add the new vregs as available values.
      DenseMap<unsigned, AvailableValsTy>::iterator LI =
        SSAUpdateVals.find(VReg);
      for (unsigned j = 0, ee = LI->second.size(); j != ee; ++j) {
        MachineBasicBlock *SrcBB = LI->second[j].first;
        unsigned SrcReg = LI->second[j].second;
        SSAUpdate.AddAvailableValue(SrcBB, SrcReg);
      }

      // Rewrite uses that are outside of the original def's block.
      MachineRegisterInfo::use_iterator UI = MRI->use_begin(VReg);
      while (UI != MRI->use_end()) {
        MachineOperand &UseMO = *UI;
        MachineInstr *UseMI = UseMO.getParent();
        ++UI;
        if (UseMI->isDebugValue()) {
          // SSAUpdate can replace the use with an undef. That creates
          // a debug instruction that is a kill.
          // FIXME: Should it SSAUpdate job to delete debug instructions
          // instead of replacing the use with undef?
          UseMI->eraseFromParent();
          continue;
        }
        if (UseMI->getParent() == DefBB && !UseMI->isPHI())
          continue;
        SSAUpdate.RewriteUse(UseMO);
      }
    }

    SSAUpdateVRs.clear();
    SSAUpdateVals.clear();
  }

  // Eliminate some of the copies inserted by tail duplication to maintain
  // SSA form.
  for (unsigned i = 0, e = Copies.size(); i != e; ++i) {
    MachineInstr *Copy = Copies[i];
    if (!Copy->isCopy())
      continue;
    unsigned Dst = Copy->getOperand(0).getReg();
    unsigned Src = Copy->getOperand(1).getReg();
    if (MRI->hasOneNonDBGUse(Src) &&
        MRI->constrainRegClass(Src, MRI->getRegClass(Dst))) {
      // Copy is the only use. Do trivial copy propagation here.
      MRI->replaceRegWith(Dst, Src);
      Copy->eraseFromParent();
    }
  }

  if (NewPHIs.size())
    NumAddedPHIs += NewPHIs.size();

  return true;
}

/// TailDuplicateBlocks - Look for small blocks that are unconditionally
/// branched to and do not fall through. Tail-duplicate their instructions
/// into their predecessors to eliminate (dynamic) branches.
bool TailDuplicatePass::TailDuplicateBlocks(MachineFunction &MF) {
  bool MadeChange = false;

  if (PreRegAlloc && TailDupVerify) {
    DEBUG(dbgs() << "\n*** Before tail-duplicating\n");
    VerifyPHIs(MF, true);
  }

  for (MachineFunction::iterator I = ++MF.begin(), E = MF.end(); I != E; ) {
    MachineBasicBlock *MBB = I++;

    if (NumTails == TailDupLimit)
      break;

    bool IsSimple = isSimpleBB(MBB);

    if (!shouldTailDuplicate(MF, IsSimple, *MBB))
      continue;

    MadeChange |= TailDuplicateAndUpdate(MBB, IsSimple, MF);
  }

  if (PreRegAlloc && TailDupVerify)
    VerifyPHIs(MF, false);

  return MadeChange;
}

static bool isDefLiveOut(unsigned Reg, MachineBasicBlock *BB,
                         const MachineRegisterInfo *MRI) {
  for (MachineInstr &UseMI : MRI->use_instructions(Reg)) {
    if (UseMI.isDebugValue())
      continue;
    if (UseMI.getParent() != BB)
      return true;
  }
  return false;
}

static unsigned getPHISrcRegOpIdx(MachineInstr *MI, MachineBasicBlock *SrcBB) {
  for (unsigned i = 1, e = MI->getNumOperands(); i != e; i += 2)
    if (MI->getOperand(i+1).getMBB() == SrcBB)
      return i;
  return 0;
}


// Remember which registers are used by phis in this block. This is
// used to determine which registers are liveout while modifying the
// block (which is why we need to copy the information).
static void getRegsUsedByPHIs(const MachineBasicBlock &BB,
                              DenseSet<unsigned> *UsedByPhi) {
  for (const auto &MI : BB) {
    if (!MI.isPHI())
      break;
    for (unsigned i = 1, e = MI.getNumOperands(); i != e; i += 2) {
      unsigned SrcReg = MI.getOperand(i).getReg();
      UsedByPhi->insert(SrcReg);
    }
  }
}

/// AddSSAUpdateEntry - Add a definition and source virtual registers pair for
/// SSA update.
void TailDuplicatePass::AddSSAUpdateEntry(unsigned OrigReg, unsigned NewReg,
                                          MachineBasicBlock *BB) {
  DenseMap<unsigned, AvailableValsTy>::iterator LI= SSAUpdateVals.find(OrigReg);
  if (LI != SSAUpdateVals.end())
    LI->second.push_back(std::make_pair(BB, NewReg));
  else {
    AvailableValsTy Vals;
    Vals.push_back(std::make_pair(BB, NewReg));
    SSAUpdateVals.insert(std::make_pair(OrigReg, Vals));
    SSAUpdateVRs.push_back(OrigReg);
  }
}

/// ProcessPHI - Process PHI node in TailBB by turning it into a copy in PredBB.
/// Remember the source register that's contributed by PredBB and update SSA
/// update map.
void TailDuplicatePass::ProcessPHI(
    MachineInstr *MI, MachineBasicBlock *TailBB, MachineBasicBlock *PredBB,
    DenseMap<unsigned, unsigned> &LocalVRMap,
    SmallVectorImpl<std::pair<unsigned, unsigned> > &Copies,
    const DenseSet<unsigned> &RegsUsedByPhi, bool Remove) {
  unsigned DefReg = MI->getOperand(0).getReg();
  unsigned SrcOpIdx = getPHISrcRegOpIdx(MI, PredBB);
  assert(SrcOpIdx && "Unable to find matching PHI source?");
  unsigned SrcReg = MI->getOperand(SrcOpIdx).getReg();
  const TargetRegisterClass *RC = MRI->getRegClass(DefReg);
  LocalVRMap.insert(std::make_pair(DefReg, SrcReg));

  // Insert a copy from source to the end of the block. The def register is the
  // available value liveout of the block.
  unsigned NewDef = MRI->createVirtualRegister(RC);
  Copies.push_back(std::make_pair(NewDef, SrcReg));
  if (isDefLiveOut(DefReg, TailBB, MRI) || RegsUsedByPhi.count(DefReg))
    AddSSAUpdateEntry(DefReg, NewDef, PredBB);

  if (!Remove)
    return;

  // Remove PredBB from the PHI node.
  MI->RemoveOperand(SrcOpIdx+1);
  MI->RemoveOperand(SrcOpIdx);
  if (MI->getNumOperands() == 1)
    MI->eraseFromParent();
}

/// DuplicateInstruction - Duplicate a TailBB instruction to PredBB and update
/// the source operands due to earlier PHI translation.
void TailDuplicatePass::DuplicateInstruction(MachineInstr *MI,
                                     MachineBasicBlock *TailBB,
                                     MachineBasicBlock *PredBB,
                                     MachineFunction &MF,
                                     DenseMap<unsigned, unsigned> &LocalVRMap,
                                     const DenseSet<unsigned> &UsedByPhi) {
  MachineInstr *NewMI = TII->duplicate(MI, MF);
  for (unsigned i = 0, e = NewMI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = NewMI->getOperand(i);
    if (!MO.isReg())
      continue;
    unsigned Reg = MO.getReg();
    if (!TargetRegisterInfo::isVirtualRegister(Reg))
      continue;
    if (MO.isDef()) {
      const TargetRegisterClass *RC = MRI->getRegClass(Reg);
      unsigned NewReg = MRI->createVirtualRegister(RC);
      MO.setReg(NewReg);
      LocalVRMap.insert(std::make_pair(Reg, NewReg));
      if (isDefLiveOut(Reg, TailBB, MRI) || UsedByPhi.count(Reg))
        AddSSAUpdateEntry(Reg, NewReg, PredBB);
    } else {
      DenseMap<unsigned, unsigned>::iterator VI = LocalVRMap.find(Reg);
      if (VI != LocalVRMap.end()) {
        MO.setReg(VI->second);
        // Clear any kill flags from this operand.  The new register could have
        // uses after this one, so kills are not valid here.
        MO.setIsKill(false);
        MRI->constrainRegClass(VI->second, MRI->getRegClass(Reg));
      }
    }
  }
  PredBB->insert(PredBB->instr_end(), NewMI);
}

/// UpdateSuccessorsPHIs - After FromBB is tail duplicated into its predecessor
/// blocks, the successors have gained new predecessors. Update the PHI
/// instructions in them accordingly.
void
TailDuplicatePass::UpdateSuccessorsPHIs(MachineBasicBlock *FromBB, bool isDead,
                                  SmallVectorImpl<MachineBasicBlock *> &TDBBs,
                                  SmallSetVector<MachineBasicBlock*,8> &Succs) {
  for (SmallSetVector<MachineBasicBlock*, 8>::iterator SI = Succs.begin(),
         SE = Succs.end(); SI != SE; ++SI) {
    MachineBasicBlock *SuccBB = *SI;
    for (MachineBasicBlock::iterator II = SuccBB->begin(), EE = SuccBB->end();
         II != EE; ++II) {
      if (!II->isPHI())
        break;
      MachineInstrBuilder MIB(*FromBB->getParent(), II);
      unsigned Idx = 0;
      for (unsigned i = 1, e = II->getNumOperands(); i != e; i += 2) {
        MachineOperand &MO = II->getOperand(i+1);
        if (MO.getMBB() == FromBB) {
          Idx = i;
          break;
        }
      }

      assert(Idx != 0);
      MachineOperand &MO0 = II->getOperand(Idx);
      unsigned Reg = MO0.getReg();
      if (isDead) {
        // Folded into the previous BB.
        // There could be duplicate phi source entries. FIXME: Should sdisel
        // or earlier pass fixed this?
        for (unsigned i = II->getNumOperands()-2; i != Idx; i -= 2) {
          MachineOperand &MO = II->getOperand(i+1);
          if (MO.getMBB() == FromBB) {
            II->RemoveOperand(i+1);
            II->RemoveOperand(i);
          }
        }
      } else
        Idx = 0;

      // If Idx is set, the operands at Idx and Idx+1 must be removed.
      // We reuse the location to avoid expensive RemoveOperand calls.

      DenseMap<unsigned,AvailableValsTy>::iterator LI=SSAUpdateVals.find(Reg);
      if (LI != SSAUpdateVals.end()) {
        // This register is defined in the tail block.
        for (unsigned j = 0, ee = LI->second.size(); j != ee; ++j) {
          MachineBasicBlock *SrcBB = LI->second[j].first;
          // If we didn't duplicate a bb into a particular predecessor, we
          // might still have added an entry to SSAUpdateVals to correcly
          // recompute SSA. If that case, avoid adding a dummy extra argument
          // this PHI.
          if (!SrcBB->isSuccessor(SuccBB))
            continue;

          unsigned SrcReg = LI->second[j].second;
          if (Idx != 0) {
            II->getOperand(Idx).setReg(SrcReg);
            II->getOperand(Idx+1).setMBB(SrcBB);
            Idx = 0;
          } else {
            MIB.addReg(SrcReg).addMBB(SrcBB);
          }
        }
      } else {
        // Live in tail block, must also be live in predecessors.
        for (unsigned j = 0, ee = TDBBs.size(); j != ee; ++j) {
          MachineBasicBlock *SrcBB = TDBBs[j];
          if (Idx != 0) {
            II->getOperand(Idx).setReg(Reg);
            II->getOperand(Idx+1).setMBB(SrcBB);
            Idx = 0;
          } else {
            MIB.addReg(Reg).addMBB(SrcBB);
          }
        }
      }
      if (Idx != 0) {
        II->RemoveOperand(Idx+1);
        II->RemoveOperand(Idx);
      }
    }
  }
}

/// shouldTailDuplicate - Determine if it is profitable to duplicate this block.
bool
TailDuplicatePass::shouldTailDuplicate(const MachineFunction &MF,
                                       bool IsSimple,
                                       MachineBasicBlock &TailBB) {
  // Only duplicate blocks that end with unconditional branches.
  if (TailBB.canFallThrough())
    return false;

  // Don't try to tail-duplicate single-block loops.
  if (TailBB.isSuccessor(&TailBB))
    return false;

  // Set the limit on the cost to duplicate. When optimizing for size,
  // duplicate only one, because one branch instruction can be eliminated to
  // compensate for the duplication.
  unsigned MaxDuplicateCount;
  if (TailDuplicateSize.getNumOccurrences() == 0 &&
      MF.getFunction()->hasFnAttribute(Attribute::OptimizeForSize))
    MaxDuplicateCount = 1;
  else
    MaxDuplicateCount = TailDuplicateSize;

  // If the target has hardware branch prediction that can handle indirect
  // branches, duplicating them can often make them predictable when there
  // are common paths through the code.  The limit needs to be high enough
  // to allow undoing the effects of tail merging and other optimizations
  // that rearrange the predecessors of the indirect branch.

  bool HasIndirectbr = false;
  if (!TailBB.empty())
    HasIndirectbr = TailBB.back().isIndirectBranch();

  if (HasIndirectbr && PreRegAlloc)
    MaxDuplicateCount = 20;

  // Check the instructions in the block to determine whether tail-duplication
  // is invalid or unlikely to be profitable.
  unsigned InstrCount = 0;
  for (MachineBasicBlock::iterator I = TailBB.begin(); I != TailBB.end(); ++I) {
    // Non-duplicable things shouldn't be tail-duplicated.
    if (I->isNotDuplicable())
      return false;

    // Do not duplicate 'return' instructions if this is a pre-regalloc run.
    // A return may expand into a lot more instructions (e.g. reload of callee
    // saved registers) after PEI.
    if (PreRegAlloc && I->isReturn())
      return false;

    // Avoid duplicating calls before register allocation. Calls presents a
    // barrier to register allocation so duplicating them may end up increasing
    // spills.
    if (PreRegAlloc && I->isCall())
      return false;

    if (!I->isPHI() && !I->isDebugValue())
      InstrCount += 1;

    if (InstrCount > MaxDuplicateCount)
      return false;
  }

  if (HasIndirectbr && PreRegAlloc)
    return true;

  if (IsSimple)
    return true;

  if (!PreRegAlloc)
    return true;

  return canCompletelyDuplicateBB(TailBB);
}

/// isSimpleBB - True if this BB has only one unconditional jump.
bool
TailDuplicatePass::isSimpleBB(MachineBasicBlock *TailBB) {
  if (TailBB->succ_size() != 1)
    return false;
  if (TailBB->pred_empty())
    return false;
  MachineBasicBlock::iterator I = TailBB->getFirstNonDebugInstr();
  if (I == TailBB->end())
    return true;
  return I->isUnconditionalBranch();
}

static bool
bothUsedInPHI(const MachineBasicBlock &A,
              SmallPtrSet<MachineBasicBlock*, 8> SuccsB) {
  for (MachineBasicBlock::const_succ_iterator SI = A.succ_begin(),
         SE = A.succ_end(); SI != SE; ++SI) {
    MachineBasicBlock *BB = *SI;
    if (SuccsB.count(BB) && !BB->empty() && BB->begin()->isPHI())
      return true;
  }

  return false;
}

bool
TailDuplicatePass::canCompletelyDuplicateBB(MachineBasicBlock &BB) {
  for (MachineBasicBlock::pred_iterator PI = BB.pred_begin(),
       PE = BB.pred_end(); PI != PE; ++PI) {
    MachineBasicBlock *PredBB = *PI;

    if (PredBB->succ_size() > 1)
      return false;

    MachineBasicBlock *PredTBB = nullptr, *PredFBB = nullptr;
    SmallVector<MachineOperand, 4> PredCond;
    if (TII->AnalyzeBranch(*PredBB, PredTBB, PredFBB, PredCond, true))
      return false;

    if (!PredCond.empty())
      return false;
  }
  return true;
}

bool
TailDuplicatePass::duplicateSimpleBB(MachineBasicBlock *TailBB,
                                    SmallVectorImpl<MachineBasicBlock *> &TDBBs,
                                    const DenseSet<unsigned> &UsedByPhi,
                                    SmallVectorImpl<MachineInstr *> &Copies) {
  SmallPtrSet<MachineBasicBlock*, 8> Succs(TailBB->succ_begin(),
                                           TailBB->succ_end());
  SmallVector<MachineBasicBlock*, 8> Preds(TailBB->pred_begin(),
                                           TailBB->pred_end());
  bool Changed = false;
  for (SmallSetVector<MachineBasicBlock *, 8>::iterator PI = Preds.begin(),
       PE = Preds.end(); PI != PE; ++PI) {
    MachineBasicBlock *PredBB = *PI;

    if (PredBB->getLandingPadSuccessor())
      continue;

    if (bothUsedInPHI(*PredBB, Succs))
      continue;

    MachineBasicBlock *PredTBB = nullptr, *PredFBB = nullptr;
    SmallVector<MachineOperand, 4> PredCond;
    if (TII->AnalyzeBranch(*PredBB, PredTBB, PredFBB, PredCond, true))
      continue;

    Changed = true;
    DEBUG(dbgs() << "\nTail-duplicating into PredBB: " << *PredBB
                 << "From simple Succ: " << *TailBB);

    MachineBasicBlock *NewTarget = *TailBB->succ_begin();
    MachineBasicBlock *NextBB = std::next(MachineFunction::iterator(PredBB));

    // Make PredFBB explicit.
    if (PredCond.empty())
      PredFBB = PredTBB;

    // Make fall through explicit.
    if (!PredTBB)
      PredTBB = NextBB;
    if (!PredFBB)
      PredFBB = NextBB;

    // Redirect
    if (PredFBB == TailBB)
      PredFBB = NewTarget;
    if (PredTBB == TailBB)
      PredTBB = NewTarget;

    // Make the branch unconditional if possible
    if (PredTBB == PredFBB) {
      PredCond.clear();
      PredFBB = nullptr;
    }

    // Avoid adding fall through branches.
    if (PredFBB == NextBB)
      PredFBB = nullptr;
    if (PredTBB == NextBB && PredFBB == nullptr)
      PredTBB = nullptr;

    TII->RemoveBranch(*PredBB);

    if (PredTBB)
      TII->InsertBranch(*PredBB, PredTBB, PredFBB, PredCond, DebugLoc());

    uint32_t Weight = MBPI->getEdgeWeight(PredBB, TailBB);
    PredBB->removeSuccessor(TailBB);
    unsigned NumSuccessors = PredBB->succ_size();
    assert(NumSuccessors <= 1);
    if (NumSuccessors == 0 || *PredBB->succ_begin() != NewTarget)
      PredBB->addSuccessor(NewTarget, Weight);

    TDBBs.push_back(PredBB);
  }
  return Changed;
}

/// TailDuplicate - If it is profitable, duplicate TailBB's contents in each
/// of its predecessors.
bool
TailDuplicatePass::TailDuplicate(MachineBasicBlock *TailBB,
                                 bool IsSimple,
                                 MachineFunction &MF,
                                 SmallVectorImpl<MachineBasicBlock *> &TDBBs,
                                 SmallVectorImpl<MachineInstr *> &Copies) {
  DEBUG(dbgs() << "\n*** Tail-duplicating BB#" << TailBB->getNumber() << '\n');

  DenseSet<unsigned> UsedByPhi;
  getRegsUsedByPHIs(*TailBB, &UsedByPhi);

  if (IsSimple)
    return duplicateSimpleBB(TailBB, TDBBs, UsedByPhi, Copies);

  // Iterate through all the unique predecessors and tail-duplicate this
  // block into them, if possible. Copying the list ahead of time also
  // avoids trouble with the predecessor list reallocating.
  bool Changed = false;
  SmallSetVector<MachineBasicBlock*, 8> Preds(TailBB->pred_begin(),
                                              TailBB->pred_end());
  for (SmallSetVector<MachineBasicBlock *, 8>::iterator PI = Preds.begin(),
       PE = Preds.end(); PI != PE; ++PI) {
    MachineBasicBlock *PredBB = *PI;

    assert(TailBB != PredBB &&
           "Single-block loop should have been rejected earlier!");
    // EH edges are ignored by AnalyzeBranch.
    if (PredBB->succ_size() > 1)
      continue;

    MachineBasicBlock *PredTBB, *PredFBB;
    SmallVector<MachineOperand, 4> PredCond;
    if (TII->AnalyzeBranch(*PredBB, PredTBB, PredFBB, PredCond, true))
      continue;
    if (!PredCond.empty())
      continue;
    // Don't duplicate into a fall-through predecessor (at least for now).
    if (PredBB->isLayoutSuccessor(TailBB) && PredBB->canFallThrough())
      continue;

    DEBUG(dbgs() << "\nTail-duplicating into PredBB: " << *PredBB
                 << "From Succ: " << *TailBB);

    TDBBs.push_back(PredBB);

    // Remove PredBB's unconditional branch.
    TII->RemoveBranch(*PredBB);

    if (RS && !TailBB->livein_empty()) {
      // Update PredBB livein.
      RS->enterBasicBlock(PredBB);
      if (!PredBB->empty())
        RS->forward(std::prev(PredBB->end()));
      for (MachineBasicBlock::livein_iterator I = TailBB->livein_begin(),
             E = TailBB->livein_end(); I != E; ++I) {
        if (!RS->isRegUsed(*I, false))
          // If a register is previously livein to the tail but it's not live
          // at the end of predecessor BB, then it should be added to its
          // livein list.
          PredBB->addLiveIn(*I);
      }
    }

    // Clone the contents of TailBB into PredBB.
    DenseMap<unsigned, unsigned> LocalVRMap;
    SmallVector<std::pair<unsigned,unsigned>, 4> CopyInfos;
    // Use instr_iterator here to properly handle bundles, e.g.
    // ARM Thumb2 IT block.
    MachineBasicBlock::instr_iterator I = TailBB->instr_begin();
    while (I != TailBB->instr_end()) {
      MachineInstr *MI = &*I;
      ++I;
      if (MI->isPHI()) {
        // Replace the uses of the def of the PHI with the register coming
        // from PredBB.
        ProcessPHI(MI, TailBB, PredBB, LocalVRMap, CopyInfos, UsedByPhi, true);
      } else {
        // Replace def of virtual registers with new registers, and update
        // uses with PHI source register or the new registers.
        DuplicateInstruction(MI, TailBB, PredBB, MF, LocalVRMap, UsedByPhi);
      }
    }
    MachineBasicBlock::iterator Loc = PredBB->getFirstTerminator();
    for (unsigned i = 0, e = CopyInfos.size(); i != e; ++i) {
      Copies.push_back(BuildMI(*PredBB, Loc, DebugLoc(),
                               TII->get(TargetOpcode::COPY),
                               CopyInfos[i].first).addReg(CopyInfos[i].second));
    }

    // Simplify
    TII->AnalyzeBranch(*PredBB, PredTBB, PredFBB, PredCond, true);

    NumInstrDups += TailBB->size() - 1; // subtract one for removed branch

    // Update the CFG.
    PredBB->removeSuccessor(PredBB->succ_begin());
    assert(PredBB->succ_empty() &&
           "TailDuplicate called on block with multiple successors!");
    for (MachineBasicBlock::succ_iterator I = TailBB->succ_begin(),
           E = TailBB->succ_end(); I != E; ++I)
      PredBB->addSuccessor(*I, MBPI->getEdgeWeight(TailBB, I));

    Changed = true;
    ++NumTailDups;
  }

  // If TailBB was duplicated into all its predecessors except for the prior
  // block, which falls through unconditionally, move the contents of this
  // block into the prior block.
  MachineBasicBlock *PrevBB = std::prev(MachineFunction::iterator(TailBB));
  MachineBasicBlock *PriorTBB = nullptr, *PriorFBB = nullptr;
  SmallVector<MachineOperand, 4> PriorCond;
  // This has to check PrevBB->succ_size() because EH edges are ignored by
  // AnalyzeBranch.
  if (PrevBB->succ_size() == 1 &&
      !TII->AnalyzeBranch(*PrevBB, PriorTBB, PriorFBB, PriorCond, true) &&
      PriorCond.empty() && !PriorTBB && TailBB->pred_size() == 1 &&
      !TailBB->hasAddressTaken()) {
    DEBUG(dbgs() << "\nMerging into block: " << *PrevBB
          << "From MBB: " << *TailBB);
    if (PreRegAlloc) {
      DenseMap<unsigned, unsigned> LocalVRMap;
      SmallVector<std::pair<unsigned,unsigned>, 4> CopyInfos;
      MachineBasicBlock::iterator I = TailBB->begin();
      // Process PHI instructions first.
      while (I != TailBB->end() && I->isPHI()) {
        // Replace the uses of the def of the PHI with the register coming
        // from PredBB.
        MachineInstr *MI = &*I++;
        ProcessPHI(MI, TailBB, PrevBB, LocalVRMap, CopyInfos, UsedByPhi, true);
        if (MI->getParent())
          MI->eraseFromParent();
      }

      // Now copy the non-PHI instructions.
      while (I != TailBB->end()) {
        // Replace def of virtual registers with new registers, and update
        // uses with PHI source register or the new registers.
        MachineInstr *MI = &*I++;
        assert(!MI->isBundle() && "Not expecting bundles before regalloc!");
        DuplicateInstruction(MI, TailBB, PrevBB, MF, LocalVRMap, UsedByPhi);
        MI->eraseFromParent();
      }
      MachineBasicBlock::iterator Loc = PrevBB->getFirstTerminator();
      for (unsigned i = 0, e = CopyInfos.size(); i != e; ++i) {
        Copies.push_back(BuildMI(*PrevBB, Loc, DebugLoc(),
                                 TII->get(TargetOpcode::COPY),
                                 CopyInfos[i].first)
                           .addReg(CopyInfos[i].second));
      }
    } else {
      // No PHIs to worry about, just splice the instructions over.
      PrevBB->splice(PrevBB->end(), TailBB, TailBB->begin(), TailBB->end());
    }
    PrevBB->removeSuccessor(PrevBB->succ_begin());
    assert(PrevBB->succ_empty());
    PrevBB->transferSuccessors(TailBB);
    TDBBs.push_back(PrevBB);
    Changed = true;
  }

  // If this is after register allocation, there are no phis to fix.
  if (!PreRegAlloc)
    return Changed;

  // If we made no changes so far, we are safe.
  if (!Changed)
    return Changed;


  // Handle the nasty case in that we duplicated a block that is part of a loop
  // into some but not all of its predecessors. For example:
  //    1 -> 2 <-> 3                 |
  //          \                      |
  //           \---> rest            |
  // if we duplicate 2 into 1 but not into 3, we end up with
  // 12 -> 3 <-> 2 -> rest           |
  //   \             /               |
  //    \----->-----/                |
  // If there was a "var = phi(1, 3)" in 2, it has to be ultimately replaced
  // with a phi in 3 (which now dominates 2).
  // What we do here is introduce a copy in 3 of the register defined by the
  // phi, just like when we are duplicating 2 into 3, but we don't copy any
  // real instructions or remove the 3 -> 2 edge from the phi in 2.
  for (SmallSetVector<MachineBasicBlock *, 8>::iterator PI = Preds.begin(),
       PE = Preds.end(); PI != PE; ++PI) {
    MachineBasicBlock *PredBB = *PI;
    if (std::find(TDBBs.begin(), TDBBs.end(), PredBB) != TDBBs.end())
      continue;

    // EH edges
    if (PredBB->succ_size() != 1)
      continue;

    DenseMap<unsigned, unsigned> LocalVRMap;
    SmallVector<std::pair<unsigned,unsigned>, 4> CopyInfos;
    MachineBasicBlock::iterator I = TailBB->begin();
    // Process PHI instructions first.
    while (I != TailBB->end() && I->isPHI()) {
      // Replace the uses of the def of the PHI with the register coming
      // from PredBB.
      MachineInstr *MI = &*I++;
      ProcessPHI(MI, TailBB, PredBB, LocalVRMap, CopyInfos, UsedByPhi, false);
    }
    MachineBasicBlock::iterator Loc = PredBB->getFirstTerminator();
    for (unsigned i = 0, e = CopyInfos.size(); i != e; ++i) {
      Copies.push_back(BuildMI(*PredBB, Loc, DebugLoc(),
                               TII->get(TargetOpcode::COPY),
                               CopyInfos[i].first).addReg(CopyInfos[i].second));
    }
  }

  return Changed;
}

/// RemoveDeadBlock - Remove the specified dead machine basic block from the
/// function, updating the CFG.
void TailDuplicatePass::RemoveDeadBlock(MachineBasicBlock *MBB) {
  assert(MBB->pred_empty() && "MBB must be dead!");
  DEBUG(dbgs() << "\nRemoving MBB: " << *MBB);

  // Remove all successors.
  while (!MBB->succ_empty())
    MBB->removeSuccessor(MBB->succ_end()-1);

  // Remove the block.
  MBB->eraseFromParent();
}
