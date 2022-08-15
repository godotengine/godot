//===-- BranchFolding.cpp - Fold machine code branch instructions ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass forwards branches to unconditional branches to make them branch
// directly to the target block.  This pass often results in dead MBB's, which
// it then removes.
//
// Note that this pass must be run after register allocation, it cannot handle
// SSA form.
//
//===----------------------------------------------------------------------===//

#include "BranchFolding.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineBranchProbabilityInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include <algorithm>
using namespace llvm;

#define DEBUG_TYPE "branchfolding"

STATISTIC(NumDeadBlocks, "Number of dead blocks removed");
STATISTIC(NumBranchOpts, "Number of branches optimized");
STATISTIC(NumTailMerge , "Number of block tails merged");
STATISTIC(NumHoist     , "Number of times common instructions are hoisted");

static cl::opt<cl::boolOrDefault> FlagEnableTailMerge("enable-tail-merge",
                              cl::init(cl::BOU_UNSET), cl::Hidden);

// Throttle for huge numbers of predecessors (compile speed problems)
static cl::opt<unsigned>
TailMergeThreshold("tail-merge-threshold",
          cl::desc("Max number of predecessors to consider tail merging"),
          cl::init(150), cl::Hidden);

// Heuristic for tail merging (and, inversely, tail duplication).
// TODO: This should be replaced with a target query.
static cl::opt<unsigned>
TailMergeSize("tail-merge-size",
          cl::desc("Min number of instructions to consider tail merging"),
                              cl::init(3), cl::Hidden);

namespace {
  /// BranchFolderPass - Wrap branch folder in a machine function pass.
  class BranchFolderPass : public MachineFunctionPass {
  public:
    static char ID;
    explicit BranchFolderPass(): MachineFunctionPass(ID) {}

    bool runOnMachineFunction(MachineFunction &MF) override;

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.addRequired<MachineBlockFrequencyInfo>();
      AU.addRequired<MachineBranchProbabilityInfo>();
      AU.addRequired<TargetPassConfig>();
      MachineFunctionPass::getAnalysisUsage(AU);
    }
  };
}

char BranchFolderPass::ID = 0;
char &llvm::BranchFolderPassID = BranchFolderPass::ID;

INITIALIZE_PASS(BranchFolderPass, "branch-folder",
                "Control Flow Optimizer", false, false)

bool BranchFolderPass::runOnMachineFunction(MachineFunction &MF) {
  if (skipOptnoneFunction(*MF.getFunction()))
    return false;

  TargetPassConfig *PassConfig = &getAnalysis<TargetPassConfig>();
  // TailMerge can create jump into if branches that make CFG irreducible for
  // HW that requires structurized CFG.
  bool EnableTailMerge = !MF.getTarget().requiresStructuredCFG() &&
      PassConfig->getEnableTailMerge();
  BranchFolder Folder(EnableTailMerge, /*CommonHoist=*/true,
                      getAnalysis<MachineBlockFrequencyInfo>(),
                      getAnalysis<MachineBranchProbabilityInfo>());
  return Folder.OptimizeFunction(MF, MF.getSubtarget().getInstrInfo(),
                                 MF.getSubtarget().getRegisterInfo(),
                                 getAnalysisIfAvailable<MachineModuleInfo>());
}

BranchFolder::BranchFolder(bool defaultEnableTailMerge, bool CommonHoist,
                           const MachineBlockFrequencyInfo &FreqInfo,
                           const MachineBranchProbabilityInfo &ProbInfo)
    : EnableHoistCommonCode(CommonHoist), MBBFreqInfo(FreqInfo),
      MBPI(ProbInfo) {
  switch (FlagEnableTailMerge) {
  case cl::BOU_UNSET: EnableTailMerge = defaultEnableTailMerge; break;
  case cl::BOU_TRUE: EnableTailMerge = true; break;
  case cl::BOU_FALSE: EnableTailMerge = false; break;
  }
}

/// RemoveDeadBlock - Remove the specified dead machine basic block from the
/// function, updating the CFG.
void BranchFolder::RemoveDeadBlock(MachineBasicBlock *MBB) {
  assert(MBB->pred_empty() && "MBB must be dead!");
  DEBUG(dbgs() << "\nRemoving MBB: " << *MBB);

  MachineFunction *MF = MBB->getParent();
  // drop all successors.
  while (!MBB->succ_empty())
    MBB->removeSuccessor(MBB->succ_end()-1);

  // Avoid matching if this pointer gets reused.
  TriedMerging.erase(MBB);

  // Remove the block.
  MF->erase(MBB);
}

/// OptimizeImpDefsBlock - If a basic block is just a bunch of implicit_def
/// followed by terminators, and if the implicitly defined registers are not
/// used by the terminators, remove those implicit_def's. e.g.
/// BB1:
///   r0 = implicit_def
///   r1 = implicit_def
///   br
/// This block can be optimized away later if the implicit instructions are
/// removed.
bool BranchFolder::OptimizeImpDefsBlock(MachineBasicBlock *MBB) {
  SmallSet<unsigned, 4> ImpDefRegs;
  MachineBasicBlock::iterator I = MBB->begin();
  while (I != MBB->end()) {
    if (!I->isImplicitDef())
      break;
    unsigned Reg = I->getOperand(0).getReg();
    for (MCSubRegIterator SubRegs(Reg, TRI, /*IncludeSelf=*/true);
         SubRegs.isValid(); ++SubRegs)
      ImpDefRegs.insert(*SubRegs);
    ++I;
  }
  if (ImpDefRegs.empty())
    return false;

  MachineBasicBlock::iterator FirstTerm = I;
  while (I != MBB->end()) {
    if (!TII->isUnpredicatedTerminator(I))
      return false;
    // See if it uses any of the implicitly defined registers.
    for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i) {
      MachineOperand &MO = I->getOperand(i);
      if (!MO.isReg() || !MO.isUse())
        continue;
      unsigned Reg = MO.getReg();
      if (ImpDefRegs.count(Reg))
        return false;
    }
    ++I;
  }

  I = MBB->begin();
  while (I != FirstTerm) {
    MachineInstr *ImpDefMI = &*I;
    ++I;
    MBB->erase(ImpDefMI);
  }

  return true;
}

/// OptimizeFunction - Perhaps branch folding, tail merging and other
/// CFG optimizations on the given function.
bool BranchFolder::OptimizeFunction(MachineFunction &MF,
                                    const TargetInstrInfo *tii,
                                    const TargetRegisterInfo *tri,
                                    MachineModuleInfo *mmi) {
  if (!tii) return false;

  TriedMerging.clear();

  TII = tii;
  TRI = tri;
  MMI = mmi;
  RS = nullptr;

  // Use a RegScavenger to help update liveness when required.
  MachineRegisterInfo &MRI = MF.getRegInfo();
  if (MRI.tracksLiveness() && TRI->trackLivenessAfterRegAlloc(MF))
    RS = new RegScavenger();
  else
    MRI.invalidateLiveness();

  // Fix CFG.  The later algorithms expect it to be right.
  bool MadeChange = false;
  for (MachineFunction::iterator I = MF.begin(), E = MF.end(); I != E; I++) {
    MachineBasicBlock *MBB = I, *TBB = nullptr, *FBB = nullptr;
    SmallVector<MachineOperand, 4> Cond;
    if (!TII->AnalyzeBranch(*MBB, TBB, FBB, Cond, true))
      MadeChange |= MBB->CorrectExtraCFGEdges(TBB, FBB, !Cond.empty());
    MadeChange |= OptimizeImpDefsBlock(MBB);
  }

  bool MadeChangeThisIteration = true;
  while (MadeChangeThisIteration) {
    MadeChangeThisIteration    = TailMergeBlocks(MF);
    MadeChangeThisIteration   |= OptimizeBranches(MF);
    if (EnableHoistCommonCode)
      MadeChangeThisIteration |= HoistCommonCode(MF);
    MadeChange |= MadeChangeThisIteration;
  }

  // See if any jump tables have become dead as the code generator
  // did its thing.
  MachineJumpTableInfo *JTI = MF.getJumpTableInfo();
  if (!JTI) {
    delete RS;
    return MadeChange;
  }

  // Walk the function to find jump tables that are live.
  BitVector JTIsLive(JTI->getJumpTables().size());
  for (MachineFunction::iterator BB = MF.begin(), E = MF.end();
       BB != E; ++BB) {
    for (MachineBasicBlock::iterator I = BB->begin(), E = BB->end();
         I != E; ++I)
      for (unsigned op = 0, e = I->getNumOperands(); op != e; ++op) {
        MachineOperand &Op = I->getOperand(op);
        if (!Op.isJTI()) continue;

        // Remember that this JT is live.
        JTIsLive.set(Op.getIndex());
      }
  }

  // Finally, remove dead jump tables.  This happens when the
  // indirect jump was unreachable (and thus deleted).
  for (unsigned i = 0, e = JTIsLive.size(); i != e; ++i)
    if (!JTIsLive.test(i)) {
      JTI->RemoveJumpTable(i);
      MadeChange = true;
    }

  delete RS;
  return MadeChange;
}

//===----------------------------------------------------------------------===//
//  Tail Merging of Blocks
//===----------------------------------------------------------------------===//

/// HashMachineInstr - Compute a hash value for MI and its operands.
static unsigned HashMachineInstr(const MachineInstr *MI) {
  unsigned Hash = MI->getOpcode();
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &Op = MI->getOperand(i);

    // Merge in bits from the operand if easy. We can't use MachineOperand's
    // hash_code here because it's not deterministic and we sort by hash value
    // later.
    unsigned OperandHash = 0;
    switch (Op.getType()) {
    case MachineOperand::MO_Register:
      OperandHash = Op.getReg();
      break;
    case MachineOperand::MO_Immediate:
      OperandHash = Op.getImm();
      break;
    case MachineOperand::MO_MachineBasicBlock:
      OperandHash = Op.getMBB()->getNumber();
      break;
    case MachineOperand::MO_FrameIndex:
    case MachineOperand::MO_ConstantPoolIndex:
    case MachineOperand::MO_JumpTableIndex:
      OperandHash = Op.getIndex();
      break;
    case MachineOperand::MO_GlobalAddress:
    case MachineOperand::MO_ExternalSymbol:
      // Global address / external symbol are too hard, don't bother, but do
      // pull in the offset.
      OperandHash = Op.getOffset();
      break;
    default:
      break;
    }

    Hash += ((OperandHash << 3) | Op.getType()) << (i & 31);
  }
  return Hash;
}

/// HashEndOfMBB - Hash the last instruction in the MBB.
static unsigned HashEndOfMBB(const MachineBasicBlock *MBB) {
  MachineBasicBlock::const_iterator I = MBB->getLastNonDebugInstr();
  if (I == MBB->end())
    return 0;

  return HashMachineInstr(I);
}

/// ComputeCommonTailLength - Given two machine basic blocks, compute the number
/// of instructions they actually have in common together at their end.  Return
/// iterators for the first shared instruction in each block.
static unsigned ComputeCommonTailLength(MachineBasicBlock *MBB1,
                                        MachineBasicBlock *MBB2,
                                        MachineBasicBlock::iterator &I1,
                                        MachineBasicBlock::iterator &I2) {
  I1 = MBB1->end();
  I2 = MBB2->end();

  unsigned TailLen = 0;
  while (I1 != MBB1->begin() && I2 != MBB2->begin()) {
    --I1; --I2;
    // Skip debugging pseudos; necessary to avoid changing the code.
    while (I1->isDebugValue()) {
      if (I1==MBB1->begin()) {
        while (I2->isDebugValue()) {
          if (I2==MBB2->begin())
            // I1==DBG at begin; I2==DBG at begin
            return TailLen;
          --I2;
        }
        ++I2;
        // I1==DBG at begin; I2==non-DBG, or first of DBGs not at begin
        return TailLen;
      }
      --I1;
    }
    // I1==first (untested) non-DBG preceding known match
    while (I2->isDebugValue()) {
      if (I2==MBB2->begin()) {
        ++I1;
        // I1==non-DBG, or first of DBGs not at begin; I2==DBG at begin
        return TailLen;
      }
      --I2;
    }
    // I1, I2==first (untested) non-DBGs preceding known match
    if (!I1->isIdenticalTo(I2) ||
        // FIXME: This check is dubious. It's used to get around a problem where
        // people incorrectly expect inline asm directives to remain in the same
        // relative order. This is untenable because normal compiler
        // optimizations (like this one) may reorder and/or merge these
        // directives.
        I1->isInlineAsm()) {
      ++I1; ++I2;
      break;
    }
    ++TailLen;
  }
  // Back past possible debugging pseudos at beginning of block.  This matters
  // when one block differs from the other only by whether debugging pseudos
  // are present at the beginning.  (This way, the various checks later for
  // I1==MBB1->begin() work as expected.)
  if (I1 == MBB1->begin() && I2 != MBB2->begin()) {
    --I2;
    while (I2->isDebugValue()) {
      if (I2 == MBB2->begin())
        return TailLen;
      --I2;
    }
    ++I2;
  }
  if (I2 == MBB2->begin() && I1 != MBB1->begin()) {
    --I1;
    while (I1->isDebugValue()) {
      if (I1 == MBB1->begin())
        return TailLen;
      --I1;
    }
    ++I1;
  }
  return TailLen;
}

void BranchFolder::MaintainLiveIns(MachineBasicBlock *CurMBB,
                                   MachineBasicBlock *NewMBB) {
  if (RS) {
    RS->enterBasicBlock(CurMBB);
    if (!CurMBB->empty())
      RS->forward(std::prev(CurMBB->end()));
    for (unsigned int i = 1, e = TRI->getNumRegs(); i != e; i++)
      if (RS->isRegUsed(i, false))
        NewMBB->addLiveIn(i);
  }
}

/// ReplaceTailWithBranchTo - Delete the instruction OldInst and everything
/// after it, replacing it with an unconditional branch to NewDest.
void BranchFolder::ReplaceTailWithBranchTo(MachineBasicBlock::iterator OldInst,
                                           MachineBasicBlock *NewDest) {
  MachineBasicBlock *CurMBB = OldInst->getParent();

  TII->ReplaceTailWithBranchTo(OldInst, NewDest);

  // For targets that use the register scavenger, we must maintain LiveIns.
  MaintainLiveIns(CurMBB, NewDest);

  ++NumTailMerge;
}

/// SplitMBBAt - Given a machine basic block and an iterator into it, split the
/// MBB so that the part before the iterator falls into the part starting at the
/// iterator.  This returns the new MBB.
MachineBasicBlock *BranchFolder::SplitMBBAt(MachineBasicBlock &CurMBB,
                                            MachineBasicBlock::iterator BBI1,
                                            const BasicBlock *BB) {
  if (!TII->isLegalToSplitMBBAt(CurMBB, BBI1))
    return nullptr;

  MachineFunction &MF = *CurMBB.getParent();

  // Create the fall-through block.
  MachineFunction::iterator MBBI = &CurMBB;
  MachineBasicBlock *NewMBB =MF.CreateMachineBasicBlock(BB);
  CurMBB.getParent()->insert(++MBBI, NewMBB);

  // Move all the successors of this block to the specified block.
  NewMBB->transferSuccessors(&CurMBB);

  // Add an edge from CurMBB to NewMBB for the fall-through.
  CurMBB.addSuccessor(NewMBB);

  // Splice the code over.
  NewMBB->splice(NewMBB->end(), &CurMBB, BBI1, CurMBB.end());

  // NewMBB inherits CurMBB's block frequency.
  MBBFreqInfo.setBlockFreq(NewMBB, MBBFreqInfo.getBlockFreq(&CurMBB));

  // For targets that use the register scavenger, we must maintain LiveIns.
  MaintainLiveIns(&CurMBB, NewMBB);

  return NewMBB;
}

/// EstimateRuntime - Make a rough estimate for how long it will take to run
/// the specified code.
static unsigned EstimateRuntime(MachineBasicBlock::iterator I,
                                MachineBasicBlock::iterator E) {
  unsigned Time = 0;
  for (; I != E; ++I) {
    if (I->isDebugValue())
      continue;
    if (I->isCall())
      Time += 10;
    else if (I->mayLoad() || I->mayStore())
      Time += 2;
    else
      ++Time;
  }
  return Time;
}

// CurMBB needs to add an unconditional branch to SuccMBB (we removed these
// branches temporarily for tail merging).  In the case where CurMBB ends
// with a conditional branch to the next block, optimize by reversing the
// test and conditionally branching to SuccMBB instead.
static void FixTail(MachineBasicBlock *CurMBB, MachineBasicBlock *SuccBB,
                    const TargetInstrInfo *TII) {
  MachineFunction *MF = CurMBB->getParent();
  MachineFunction::iterator I = std::next(MachineFunction::iterator(CurMBB));
  MachineBasicBlock *TBB = nullptr, *FBB = nullptr;
  SmallVector<MachineOperand, 4> Cond;
  DebugLoc dl;  // FIXME: this is nowhere
  if (I != MF->end() &&
      !TII->AnalyzeBranch(*CurMBB, TBB, FBB, Cond, true)) {
    MachineBasicBlock *NextBB = I;
    if (TBB == NextBB && !Cond.empty() && !FBB) {
      if (!TII->ReverseBranchCondition(Cond)) {
        TII->RemoveBranch(*CurMBB);
        TII->InsertBranch(*CurMBB, SuccBB, nullptr, Cond, dl);
        return;
      }
    }
  }
  TII->InsertBranch(*CurMBB, SuccBB, nullptr,
                    SmallVector<MachineOperand, 0>(), dl);
}

bool
BranchFolder::MergePotentialsElt::operator<(const MergePotentialsElt &o) const {
  if (getHash() < o.getHash())
    return true;
  if (getHash() > o.getHash())
    return false;
  if (getBlock()->getNumber() < o.getBlock()->getNumber())
    return true;
  if (getBlock()->getNumber() > o.getBlock()->getNumber())
    return false;
  // _GLIBCXX_DEBUG checks strict weak ordering, which involves comparing
  // an object with itself.
#ifndef _GLIBCXX_DEBUG
  llvm_unreachable("Predecessor appears twice");
#else
  return false;
#endif
}

BlockFrequency
BranchFolder::MBFIWrapper::getBlockFreq(const MachineBasicBlock *MBB) const {
  auto I = MergedBBFreq.find(MBB);

  if (I != MergedBBFreq.end())
    return I->second;

  return MBFI.getBlockFreq(MBB);
}

void BranchFolder::MBFIWrapper::setBlockFreq(const MachineBasicBlock *MBB,
                                             BlockFrequency F) {
  MergedBBFreq[MBB] = F;
}

/// CountTerminators - Count the number of terminators in the given
/// block and set I to the position of the first non-terminator, if there
/// is one, or MBB->end() otherwise.
static unsigned CountTerminators(MachineBasicBlock *MBB,
                                 MachineBasicBlock::iterator &I) {
  I = MBB->end();
  unsigned NumTerms = 0;
  for (;;) {
    if (I == MBB->begin()) {
      I = MBB->end();
      break;
    }
    --I;
    if (!I->isTerminator()) break;
    ++NumTerms;
  }
  return NumTerms;
}

/// ProfitableToMerge - Check if two machine basic blocks have a common tail
/// and decide if it would be profitable to merge those tails.  Return the
/// length of the common tail and iterators to the first common instruction
/// in each block.
static bool ProfitableToMerge(MachineBasicBlock *MBB1,
                              MachineBasicBlock *MBB2,
                              unsigned minCommonTailLength,
                              unsigned &CommonTailLen,
                              MachineBasicBlock::iterator &I1,
                              MachineBasicBlock::iterator &I2,
                              MachineBasicBlock *SuccBB,
                              MachineBasicBlock *PredBB) {
  CommonTailLen = ComputeCommonTailLength(MBB1, MBB2, I1, I2);
  if (CommonTailLen == 0)
    return false;
  DEBUG(dbgs() << "Common tail length of BB#" << MBB1->getNumber()
               << " and BB#" << MBB2->getNumber() << " is " << CommonTailLen
               << '\n');

  // It's almost always profitable to merge any number of non-terminator
  // instructions with the block that falls through into the common successor.
  if (MBB1 == PredBB || MBB2 == PredBB) {
    MachineBasicBlock::iterator I;
    unsigned NumTerms = CountTerminators(MBB1 == PredBB ? MBB2 : MBB1, I);
    if (CommonTailLen > NumTerms)
      return true;
  }

  // If one of the blocks can be completely merged and happens to be in
  // a position where the other could fall through into it, merge any number
  // of instructions, because it can be done without a branch.
  // TODO: If the blocks are not adjacent, move one of them so that they are?
  if (MBB1->isLayoutSuccessor(MBB2) && I2 == MBB2->begin())
    return true;
  if (MBB2->isLayoutSuccessor(MBB1) && I1 == MBB1->begin())
    return true;

  // If both blocks have an unconditional branch temporarily stripped out,
  // count that as an additional common instruction for the following
  // heuristics.
  unsigned EffectiveTailLen = CommonTailLen;
  if (SuccBB && MBB1 != PredBB && MBB2 != PredBB &&
      !MBB1->back().isBarrier() &&
      !MBB2->back().isBarrier())
    ++EffectiveTailLen;

  // Check if the common tail is long enough to be worthwhile.
  if (EffectiveTailLen >= minCommonTailLength)
    return true;

  // If we are optimizing for code size, 2 instructions in common is enough if
  // we don't have to split a block.  At worst we will be introducing 1 new
  // branch instruction, which is likely to be smaller than the 2
  // instructions that would be deleted in the merge.
  MachineFunction *MF = MBB1->getParent();
  if (EffectiveTailLen >= 2 &&
      MF->getFunction()->hasFnAttribute(Attribute::OptimizeForSize) &&
      (I1 == MBB1->begin() || I2 == MBB2->begin()))
    return true;

  return false;
}

/// ComputeSameTails - Look through all the blocks in MergePotentials that have
/// hash CurHash (guaranteed to match the last element).  Build the vector
/// SameTails of all those that have the (same) largest number of instructions
/// in common of any pair of these blocks.  SameTails entries contain an
/// iterator into MergePotentials (from which the MachineBasicBlock can be
/// found) and a MachineBasicBlock::iterator into that MBB indicating the
/// instruction where the matching code sequence begins.
/// Order of elements in SameTails is the reverse of the order in which
/// those blocks appear in MergePotentials (where they are not necessarily
/// consecutive).
unsigned BranchFolder::ComputeSameTails(unsigned CurHash,
                                        unsigned minCommonTailLength,
                                        MachineBasicBlock *SuccBB,
                                        MachineBasicBlock *PredBB) {
  unsigned maxCommonTailLength = 0U;
  SameTails.clear();
  MachineBasicBlock::iterator TrialBBI1, TrialBBI2;
  MPIterator HighestMPIter = std::prev(MergePotentials.end());
  for (MPIterator CurMPIter = std::prev(MergePotentials.end()),
                  B = MergePotentials.begin();
       CurMPIter != B && CurMPIter->getHash() == CurHash; --CurMPIter) {
    for (MPIterator I = std::prev(CurMPIter); I->getHash() == CurHash; --I) {
      unsigned CommonTailLen;
      if (ProfitableToMerge(CurMPIter->getBlock(), I->getBlock(),
                            minCommonTailLength,
                            CommonTailLen, TrialBBI1, TrialBBI2,
                            SuccBB, PredBB)) {
        if (CommonTailLen > maxCommonTailLength) {
          SameTails.clear();
          maxCommonTailLength = CommonTailLen;
          HighestMPIter = CurMPIter;
          SameTails.push_back(SameTailElt(CurMPIter, TrialBBI1));
        }
        if (HighestMPIter == CurMPIter &&
            CommonTailLen == maxCommonTailLength)
          SameTails.push_back(SameTailElt(I, TrialBBI2));
      }
      if (I == B)
        break;
    }
  }
  return maxCommonTailLength;
}

/// RemoveBlocksWithHash - Remove all blocks with hash CurHash from
/// MergePotentials, restoring branches at ends of blocks as appropriate.
void BranchFolder::RemoveBlocksWithHash(unsigned CurHash,
                                        MachineBasicBlock *SuccBB,
                                        MachineBasicBlock *PredBB) {
  MPIterator CurMPIter, B;
  for (CurMPIter = std::prev(MergePotentials.end()),
      B = MergePotentials.begin();
       CurMPIter->getHash() == CurHash; --CurMPIter) {
    // Put the unconditional branch back, if we need one.
    MachineBasicBlock *CurMBB = CurMPIter->getBlock();
    if (SuccBB && CurMBB != PredBB)
      FixTail(CurMBB, SuccBB, TII);
    if (CurMPIter == B)
      break;
  }
  if (CurMPIter->getHash() != CurHash)
    CurMPIter++;
  MergePotentials.erase(CurMPIter, MergePotentials.end());
}

/// CreateCommonTailOnlyBlock - None of the blocks to be tail-merged consist
/// only of the common tail.  Create a block that does by splitting one.
bool BranchFolder::CreateCommonTailOnlyBlock(MachineBasicBlock *&PredBB,
                                             MachineBasicBlock *SuccBB,
                                             unsigned maxCommonTailLength,
                                             unsigned &commonTailIndex) {
  commonTailIndex = 0;
  unsigned TimeEstimate = ~0U;
  for (unsigned i = 0, e = SameTails.size(); i != e; ++i) {
    // Use PredBB if possible; that doesn't require a new branch.
    if (SameTails[i].getBlock() == PredBB) {
      commonTailIndex = i;
      break;
    }
    // Otherwise, make a (fairly bogus) choice based on estimate of
    // how long it will take the various blocks to execute.
    unsigned t = EstimateRuntime(SameTails[i].getBlock()->begin(),
                                 SameTails[i].getTailStartPos());
    if (t <= TimeEstimate) {
      TimeEstimate = t;
      commonTailIndex = i;
    }
  }

  MachineBasicBlock::iterator BBI =
    SameTails[commonTailIndex].getTailStartPos();
  MachineBasicBlock *MBB = SameTails[commonTailIndex].getBlock();

  // If the common tail includes any debug info we will take it pretty
  // randomly from one of the inputs.  Might be better to remove it?
  DEBUG(dbgs() << "\nSplitting BB#" << MBB->getNumber() << ", size "
               << maxCommonTailLength);

  // If the split block unconditionally falls-thru to SuccBB, it will be
  // merged. In control flow terms it should then take SuccBB's name. e.g. If
  // SuccBB is an inner loop, the common tail is still part of the inner loop.
  const BasicBlock *BB = (SuccBB && MBB->succ_size() == 1) ?
    SuccBB->getBasicBlock() : MBB->getBasicBlock();
  MachineBasicBlock *newMBB = SplitMBBAt(*MBB, BBI, BB);
  if (!newMBB) {
    DEBUG(dbgs() << "... failed!");
    return false;
  }

  SameTails[commonTailIndex].setBlock(newMBB);
  SameTails[commonTailIndex].setTailStartPos(newMBB->begin());

  // If we split PredBB, newMBB is the new predecessor.
  if (PredBB == MBB)
    PredBB = newMBB;

  return true;
}

static bool hasIdenticalMMOs(const MachineInstr *MI1, const MachineInstr *MI2) {
  auto I1 = MI1->memoperands_begin(), E1 = MI1->memoperands_end();
  auto I2 = MI2->memoperands_begin(), E2 = MI2->memoperands_end();
  if ((E1 - I1) != (E2 - I2))
    return false;
  for (; I1 != E1; ++I1, ++I2) {
    if (**I1 != **I2)
      return false;
  }
  return true;
}

static void
removeMMOsFromMemoryOperations(MachineBasicBlock::iterator MBBIStartPos,
                               MachineBasicBlock &MBBCommon) {
  // Remove MMOs from memory operations in the common block
  // when they do not match the ones from the block being tail-merged.
  // This ensures later passes conservatively compute dependencies.
  MachineBasicBlock *MBB = MBBIStartPos->getParent();
  // Note CommonTailLen does not necessarily matches the size of
  // the common BB nor all its instructions because of debug
  // instructions differences.
  unsigned CommonTailLen = 0;
  for (auto E = MBB->end(); MBBIStartPos != E; ++MBBIStartPos)
    ++CommonTailLen;

  MachineBasicBlock::reverse_iterator MBBI = MBB->rbegin();
  MachineBasicBlock::reverse_iterator MBBIE = MBB->rend();
  MachineBasicBlock::reverse_iterator MBBICommon = MBBCommon.rbegin();
  MachineBasicBlock::reverse_iterator MBBIECommon = MBBCommon.rend();

  while (CommonTailLen--) {
    assert(MBBI != MBBIE && "Reached BB end within common tail length!");
    (void)MBBIE;

    if (MBBI->isDebugValue()) {
      ++MBBI;
      continue;
    }

    while ((MBBICommon != MBBIECommon) && MBBICommon->isDebugValue())
      ++MBBICommon;

    assert(MBBICommon != MBBIECommon &&
           "Reached BB end within common tail length!");
    assert(MBBICommon->isIdenticalTo(&*MBBI) && "Expected matching MIIs!");

    if (MBBICommon->mayLoad() || MBBICommon->mayStore())
      if (!hasIdenticalMMOs(&*MBBI, &*MBBICommon))
        MBBICommon->clearMemRefs();

    ++MBBI;
    ++MBBICommon;
  }
}

// See if any of the blocks in MergePotentials (which all have a common single
// successor, or all have no successor) can be tail-merged.  If there is a
// successor, any blocks in MergePotentials that are not tail-merged and
// are not immediately before Succ must have an unconditional branch to
// Succ added (but the predecessor/successor lists need no adjustment).
// The lone predecessor of Succ that falls through into Succ,
// if any, is given in PredBB.

bool BranchFolder::TryTailMergeBlocks(MachineBasicBlock *SuccBB,
                                      MachineBasicBlock *PredBB) {
  bool MadeChange = false;

  // Except for the special cases below, tail-merge if there are at least
  // this many instructions in common.
  unsigned minCommonTailLength = TailMergeSize;

  DEBUG(dbgs() << "\nTryTailMergeBlocks: ";
        for (unsigned i = 0, e = MergePotentials.size(); i != e; ++i)
          dbgs() << "BB#" << MergePotentials[i].getBlock()->getNumber()
                 << (i == e-1 ? "" : ", ");
        dbgs() << "\n";
        if (SuccBB) {
          dbgs() << "  with successor BB#" << SuccBB->getNumber() << '\n';
          if (PredBB)
            dbgs() << "  which has fall-through from BB#"
                   << PredBB->getNumber() << "\n";
        }
        dbgs() << "Looking for common tails of at least "
               << minCommonTailLength << " instruction"
               << (minCommonTailLength == 1 ? "" : "s") << '\n';
       );

  // Sort by hash value so that blocks with identical end sequences sort
  // together.
  array_pod_sort(MergePotentials.begin(), MergePotentials.end());

  // Walk through equivalence sets looking for actual exact matches.
  while (MergePotentials.size() > 1) {
    unsigned CurHash = MergePotentials.back().getHash();

    // Build SameTails, identifying the set of blocks with this hash code
    // and with the maximum number of instructions in common.
    unsigned maxCommonTailLength = ComputeSameTails(CurHash,
                                                    minCommonTailLength,
                                                    SuccBB, PredBB);

    // If we didn't find any pair that has at least minCommonTailLength
    // instructions in common, remove all blocks with this hash code and retry.
    if (SameTails.empty()) {
      RemoveBlocksWithHash(CurHash, SuccBB, PredBB);
      continue;
    }

    // If one of the blocks is the entire common tail (and not the entry
    // block, which we can't jump to), we can treat all blocks with this same
    // tail at once.  Use PredBB if that is one of the possibilities, as that
    // will not introduce any extra branches.
    MachineBasicBlock *EntryBB = MergePotentials.begin()->getBlock()->
                                 getParent()->begin();
    unsigned commonTailIndex = SameTails.size();
    // If there are two blocks, check to see if one can be made to fall through
    // into the other.
    if (SameTails.size() == 2 &&
        SameTails[0].getBlock()->isLayoutSuccessor(SameTails[1].getBlock()) &&
        SameTails[1].tailIsWholeBlock())
      commonTailIndex = 1;
    else if (SameTails.size() == 2 &&
             SameTails[1].getBlock()->isLayoutSuccessor(
                                                     SameTails[0].getBlock()) &&
             SameTails[0].tailIsWholeBlock())
      commonTailIndex = 0;
    else {
      // Otherwise just pick one, favoring the fall-through predecessor if
      // there is one.
      for (unsigned i = 0, e = SameTails.size(); i != e; ++i) {
        MachineBasicBlock *MBB = SameTails[i].getBlock();
        if (MBB == EntryBB && SameTails[i].tailIsWholeBlock())
          continue;
        if (MBB == PredBB) {
          commonTailIndex = i;
          break;
        }
        if (SameTails[i].tailIsWholeBlock())
          commonTailIndex = i;
      }
    }

    if (commonTailIndex == SameTails.size() ||
        (SameTails[commonTailIndex].getBlock() == PredBB &&
         !SameTails[commonTailIndex].tailIsWholeBlock())) {
      // None of the blocks consist entirely of the common tail.
      // Split a block so that one does.
      if (!CreateCommonTailOnlyBlock(PredBB, SuccBB,
                                     maxCommonTailLength, commonTailIndex)) {
        RemoveBlocksWithHash(CurHash, SuccBB, PredBB);
        continue;
      }
    }

    MachineBasicBlock *MBB = SameTails[commonTailIndex].getBlock();

    // Recompute commont tail MBB's edge weights and block frequency.
    setCommonTailEdgeWeights(*MBB);

    // MBB is common tail.  Adjust all other BB's to jump to this one.
    // Traversal must be forwards so erases work.
    DEBUG(dbgs() << "\nUsing common tail in BB#" << MBB->getNumber()
                 << " for ");
    for (unsigned int i=0, e = SameTails.size(); i != e; ++i) {
      if (commonTailIndex == i)
        continue;
      DEBUG(dbgs() << "BB#" << SameTails[i].getBlock()->getNumber()
                   << (i == e-1 ? "" : ", "));
      // Remove MMOs from memory operations as needed.
      removeMMOsFromMemoryOperations(SameTails[i].getTailStartPos(), *MBB);
      // Hack the end off BB i, making it jump to BB commonTailIndex instead.
      ReplaceTailWithBranchTo(SameTails[i].getTailStartPos(), MBB);
      // BB i is no longer a predecessor of SuccBB; remove it from the worklist.
      MergePotentials.erase(SameTails[i].getMPIter());
    }
    DEBUG(dbgs() << "\n");
    // We leave commonTailIndex in the worklist in case there are other blocks
    // that match it with a smaller number of instructions.
    MadeChange = true;
  }
  return MadeChange;
}

bool BranchFolder::TailMergeBlocks(MachineFunction &MF) {
  bool MadeChange = false;
  if (!EnableTailMerge) return MadeChange;

  // First find blocks with no successors.
  MergePotentials.clear();
  for (MachineFunction::iterator I = MF.begin(), E = MF.end();
       I != E && MergePotentials.size() < TailMergeThreshold; ++I) {
    if (TriedMerging.count(I))
      continue;
    if (I->succ_empty())
      MergePotentials.push_back(MergePotentialsElt(HashEndOfMBB(I), I));
  }

  // If this is a large problem, avoid visiting the same basic blocks
  // multiple times.
  if (MergePotentials.size() == TailMergeThreshold)
    for (unsigned i = 0, e = MergePotentials.size(); i != e; ++i)
      TriedMerging.insert(MergePotentials[i].getBlock());

  // See if we can do any tail merging on those.
  if (MergePotentials.size() >= 2)
    MadeChange |= TryTailMergeBlocks(nullptr, nullptr);

  // Look at blocks (IBB) with multiple predecessors (PBB).
  // We change each predecessor to a canonical form, by
  // (1) temporarily removing any unconditional branch from the predecessor
  // to IBB, and
  // (2) alter conditional branches so they branch to the other block
  // not IBB; this may require adding back an unconditional branch to IBB
  // later, where there wasn't one coming in.  E.g.
  //   Bcc IBB
  //   fallthrough to QBB
  // here becomes
  //   Bncc QBB
  // with a conceptual B to IBB after that, which never actually exists.
  // With those changes, we see whether the predecessors' tails match,
  // and merge them if so.  We change things out of canonical form and
  // back to the way they were later in the process.  (OptimizeBranches
  // would undo some of this, but we can't use it, because we'd get into
  // a compile-time infinite loop repeatedly doing and undoing the same
  // transformations.)

  for (MachineFunction::iterator I = std::next(MF.begin()), E = MF.end();
       I != E; ++I) {
    if (I->pred_size() < 2) continue;
    SmallPtrSet<MachineBasicBlock *, 8> UniquePreds;
    MachineBasicBlock *IBB = I;
    MachineBasicBlock *PredBB = std::prev(I);
    MergePotentials.clear();
    for (MachineBasicBlock::pred_iterator P = I->pred_begin(),
           E2 = I->pred_end();
         P != E2 && MergePotentials.size() < TailMergeThreshold; ++P) {
      MachineBasicBlock *PBB = *P;
      if (TriedMerging.count(PBB))
        continue;

      // Skip blocks that loop to themselves, can't tail merge these.
      if (PBB == IBB)
        continue;

      // Visit each predecessor only once.
      if (!UniquePreds.insert(PBB).second)
        continue;

      // Skip blocks which may jump to a landing pad. Can't tail merge these.
      if (PBB->getLandingPadSuccessor())
        continue;

      MachineBasicBlock *TBB = nullptr, *FBB = nullptr;
      SmallVector<MachineOperand, 4> Cond;
      if (!TII->AnalyzeBranch(*PBB, TBB, FBB, Cond, true)) {
        // Failing case: IBB is the target of a cbr, and we cannot reverse the
        // branch.
        SmallVector<MachineOperand, 4> NewCond(Cond);
        if (!Cond.empty() && TBB == IBB) {
          if (TII->ReverseBranchCondition(NewCond))
            continue;
          // This is the QBB case described above
          if (!FBB)
            FBB = std::next(MachineFunction::iterator(PBB));
        }

        // Failing case: the only way IBB can be reached from PBB is via
        // exception handling.  Happens for landing pads.  Would be nice to have
        // a bit in the edge so we didn't have to do all this.
        if (IBB->isLandingPad()) {
          MachineFunction::iterator IP = PBB;  IP++;
          MachineBasicBlock *PredNextBB = nullptr;
          if (IP != MF.end())
            PredNextBB = IP;
          if (!TBB) {
            if (IBB != PredNextBB)      // fallthrough
              continue;
          } else if (FBB) {
            if (TBB != IBB && FBB != IBB)   // cbr then ubr
              continue;
          } else if (Cond.empty()) {
            if (TBB != IBB)               // ubr
              continue;
          } else {
            if (TBB != IBB && IBB != PredNextBB)  // cbr
              continue;
          }
        }

        // Remove the unconditional branch at the end, if any.
        if (TBB && (Cond.empty() || FBB)) {
          DebugLoc dl;  // FIXME: this is nowhere
          TII->RemoveBranch(*PBB);
          if (!Cond.empty())
            // reinsert conditional branch only, for now
            TII->InsertBranch(*PBB, (TBB == IBB) ? FBB : TBB, nullptr,
                              NewCond, dl);
        }

        MergePotentials.push_back(MergePotentialsElt(HashEndOfMBB(PBB), *P));
      }
    }

    // If this is a large problem, avoid visiting the same basic blocks multiple
    // times.
    if (MergePotentials.size() == TailMergeThreshold)
      for (unsigned i = 0, e = MergePotentials.size(); i != e; ++i)
        TriedMerging.insert(MergePotentials[i].getBlock());

    if (MergePotentials.size() >= 2)
      MadeChange |= TryTailMergeBlocks(IBB, PredBB);

    // Reinsert an unconditional branch if needed. The 1 below can occur as a
    // result of removing blocks in TryTailMergeBlocks.
    PredBB = std::prev(I);     // this may have been changed in TryTailMergeBlocks
    if (MergePotentials.size() == 1 &&
        MergePotentials.begin()->getBlock() != PredBB)
      FixTail(MergePotentials.begin()->getBlock(), IBB, TII);
  }

  return MadeChange;
}

void BranchFolder::setCommonTailEdgeWeights(MachineBasicBlock &TailMBB) {
  SmallVector<BlockFrequency, 2> EdgeFreqLs(TailMBB.succ_size());
  BlockFrequency AccumulatedMBBFreq;

  // Aggregate edge frequency of successor edge j:
  //  edgeFreq(j) = sum (freq(bb) * edgeProb(bb, j)),
  //  where bb is a basic block that is in SameTails.
  for (const auto &Src : SameTails) {
    const MachineBasicBlock *SrcMBB = Src.getBlock();
    BlockFrequency BlockFreq = MBBFreqInfo.getBlockFreq(SrcMBB);
    AccumulatedMBBFreq += BlockFreq;

    // It is not necessary to recompute edge weights if TailBB has less than two
    // successors.
    if (TailMBB.succ_size() <= 1)
      continue;

    auto EdgeFreq = EdgeFreqLs.begin();

    for (auto SuccI = TailMBB.succ_begin(), SuccE = TailMBB.succ_end();
         SuccI != SuccE; ++SuccI, ++EdgeFreq)
      *EdgeFreq += BlockFreq * MBPI.getEdgeProbability(SrcMBB, *SuccI);
  }

  MBBFreqInfo.setBlockFreq(&TailMBB, AccumulatedMBBFreq);

  if (TailMBB.succ_size() <= 1)
    return;

  auto MaxEdgeFreq = *std::max_element(EdgeFreqLs.begin(), EdgeFreqLs.end());
  uint64_t Scale = MaxEdgeFreq.getFrequency() / UINT32_MAX + 1;
  auto EdgeFreq = EdgeFreqLs.begin();

  for (auto SuccI = TailMBB.succ_begin(), SuccE = TailMBB.succ_end();
       SuccI != SuccE; ++SuccI, ++EdgeFreq)
    TailMBB.setSuccWeight(SuccI, EdgeFreq->getFrequency() / Scale);
}

//===----------------------------------------------------------------------===//
//  Branch Optimization
//===----------------------------------------------------------------------===//

bool BranchFolder::OptimizeBranches(MachineFunction &MF) {
  bool MadeChange = false;

  // Make sure blocks are numbered in order
  MF.RenumberBlocks();

  for (MachineFunction::iterator I = std::next(MF.begin()), E = MF.end();
       I != E; ) {
    MachineBasicBlock *MBB = I++;
    MadeChange |= OptimizeBlock(MBB);

    // If it is dead, remove it.
    if (MBB->pred_empty()) {
      RemoveDeadBlock(MBB);
      MadeChange = true;
      ++NumDeadBlocks;
    }
  }
  return MadeChange;
}

// Blocks should be considered empty if they contain only debug info;
// else the debug info would affect codegen.
static bool IsEmptyBlock(MachineBasicBlock *MBB) {
  return MBB->getFirstNonDebugInstr() == MBB->end();
}

// Blocks with only debug info and branches should be considered the same
// as blocks with only branches.
static bool IsBranchOnlyBlock(MachineBasicBlock *MBB) {
  MachineBasicBlock::iterator I = MBB->getFirstNonDebugInstr();
  assert(I != MBB->end() && "empty block!");
  return I->isBranch();
}

/// IsBetterFallthrough - Return true if it would be clearly better to
/// fall-through to MBB1 than to fall through into MBB2.  This has to return
/// a strict ordering, returning true for both (MBB1,MBB2) and (MBB2,MBB1) will
/// result in infinite loops.
static bool IsBetterFallthrough(MachineBasicBlock *MBB1,
                                MachineBasicBlock *MBB2) {
  // Right now, we use a simple heuristic.  If MBB2 ends with a call, and
  // MBB1 doesn't, we prefer to fall through into MBB1.  This allows us to
  // optimize branches that branch to either a return block or an assert block
  // into a fallthrough to the return.
  MachineBasicBlock::iterator MBB1I = MBB1->getLastNonDebugInstr();
  MachineBasicBlock::iterator MBB2I = MBB2->getLastNonDebugInstr();
  if (MBB1I == MBB1->end() || MBB2I == MBB2->end())
    return false;

  // If there is a clear successor ordering we make sure that one block
  // will fall through to the next
  if (MBB1->isSuccessor(MBB2)) return true;
  if (MBB2->isSuccessor(MBB1)) return false;

  return MBB2I->isCall() && !MBB1I->isCall();
}

/// getBranchDebugLoc - Find and return, if any, the DebugLoc of the branch
/// instructions on the block.
static DebugLoc getBranchDebugLoc(MachineBasicBlock &MBB) {
  MachineBasicBlock::iterator I = MBB.getLastNonDebugInstr();
  if (I != MBB.end() && I->isBranch())
    return I->getDebugLoc();
  return DebugLoc();
}

/// OptimizeBlock - Analyze and optimize control flow related to the specified
/// block.  This is never called on the entry block.
bool BranchFolder::OptimizeBlock(MachineBasicBlock *MBB) {
  bool MadeChange = false;
  MachineFunction &MF = *MBB->getParent();
ReoptimizeBlock:

  MachineFunction::iterator FallThrough = MBB;
  ++FallThrough;

  // If this block is empty, make everyone use its fall-through, not the block
  // explicitly.  Landing pads should not do this since the landing-pad table
  // points to this block.  Blocks with their addresses taken shouldn't be
  // optimized away.
  if (IsEmptyBlock(MBB) && !MBB->isLandingPad() && !MBB->hasAddressTaken()) {
    // Dead block?  Leave for cleanup later.
    if (MBB->pred_empty()) return MadeChange;

    if (FallThrough == MF.end()) {
      // TODO: Simplify preds to not branch here if possible!
    } else if (FallThrough->isLandingPad()) {
      // Don't rewrite to a landing pad fallthough.  That could lead to the case
      // where a BB jumps to more than one landing pad.
      // TODO: Is it ever worth rewriting predecessors which don't already
      // jump to a landing pad, and so can safely jump to the fallthrough?
    } else {
      // Rewrite all predecessors of the old block to go to the fallthrough
      // instead.
      while (!MBB->pred_empty()) {
        MachineBasicBlock *Pred = *(MBB->pred_end()-1);
        Pred->ReplaceUsesOfBlockWith(MBB, FallThrough);
      }
      // If MBB was the target of a jump table, update jump tables to go to the
      // fallthrough instead.
      if (MachineJumpTableInfo *MJTI = MF.getJumpTableInfo())
        MJTI->ReplaceMBBInJumpTables(MBB, FallThrough);
      MadeChange = true;
    }
    return MadeChange;
  }

  // Check to see if we can simplify the terminator of the block before this
  // one.
  MachineBasicBlock &PrevBB = *std::prev(MachineFunction::iterator(MBB));

  MachineBasicBlock *PriorTBB = nullptr, *PriorFBB = nullptr;
  SmallVector<MachineOperand, 4> PriorCond;
  bool PriorUnAnalyzable =
    TII->AnalyzeBranch(PrevBB, PriorTBB, PriorFBB, PriorCond, true);
  if (!PriorUnAnalyzable) {
    // If the CFG for the prior block has extra edges, remove them.
    MadeChange |= PrevBB.CorrectExtraCFGEdges(PriorTBB, PriorFBB,
                                              !PriorCond.empty());

    // If the previous branch is conditional and both conditions go to the same
    // destination, remove the branch, replacing it with an unconditional one or
    // a fall-through.
    if (PriorTBB && PriorTBB == PriorFBB) {
      DebugLoc dl = getBranchDebugLoc(PrevBB);
      TII->RemoveBranch(PrevBB);
      PriorCond.clear();
      if (PriorTBB != MBB)
        TII->InsertBranch(PrevBB, PriorTBB, nullptr, PriorCond, dl);
      MadeChange = true;
      ++NumBranchOpts;
      goto ReoptimizeBlock;
    }

    // If the previous block unconditionally falls through to this block and
    // this block has no other predecessors, move the contents of this block
    // into the prior block. This doesn't usually happen when SimplifyCFG
    // has been used, but it can happen if tail merging splits a fall-through
    // predecessor of a block.
    // This has to check PrevBB->succ_size() because EH edges are ignored by
    // AnalyzeBranch.
    if (PriorCond.empty() && !PriorTBB && MBB->pred_size() == 1 &&
        PrevBB.succ_size() == 1 &&
        !MBB->hasAddressTaken() && !MBB->isLandingPad()) {
      DEBUG(dbgs() << "\nMerging into block: " << PrevBB
                   << "From MBB: " << *MBB);
      // Remove redundant DBG_VALUEs first.
      if (PrevBB.begin() != PrevBB.end()) {
        MachineBasicBlock::iterator PrevBBIter = PrevBB.end();
        --PrevBBIter;
        MachineBasicBlock::iterator MBBIter = MBB->begin();
        // Check if DBG_VALUE at the end of PrevBB is identical to the
        // DBG_VALUE at the beginning of MBB.
        while (PrevBBIter != PrevBB.begin() && MBBIter != MBB->end()
               && PrevBBIter->isDebugValue() && MBBIter->isDebugValue()) {
          if (!MBBIter->isIdenticalTo(PrevBBIter))
            break;
          MachineInstr *DuplicateDbg = MBBIter;
          ++MBBIter; -- PrevBBIter;
          DuplicateDbg->eraseFromParent();
        }
      }
      PrevBB.splice(PrevBB.end(), MBB, MBB->begin(), MBB->end());
      PrevBB.removeSuccessor(PrevBB.succ_begin());
      assert(PrevBB.succ_empty());
      PrevBB.transferSuccessors(MBB);
      MadeChange = true;
      return MadeChange;
    }

    // If the previous branch *only* branches to *this* block (conditional or
    // not) remove the branch.
    if (PriorTBB == MBB && !PriorFBB) {
      TII->RemoveBranch(PrevBB);
      MadeChange = true;
      ++NumBranchOpts;
      goto ReoptimizeBlock;
    }

    // If the prior block branches somewhere else on the condition and here if
    // the condition is false, remove the uncond second branch.
    if (PriorFBB == MBB) {
      DebugLoc dl = getBranchDebugLoc(PrevBB);
      TII->RemoveBranch(PrevBB);
      TII->InsertBranch(PrevBB, PriorTBB, nullptr, PriorCond, dl);
      MadeChange = true;
      ++NumBranchOpts;
      goto ReoptimizeBlock;
    }

    // If the prior block branches here on true and somewhere else on false, and
    // if the branch condition is reversible, reverse the branch to create a
    // fall-through.
    if (PriorTBB == MBB) {
      SmallVector<MachineOperand, 4> NewPriorCond(PriorCond);
      if (!TII->ReverseBranchCondition(NewPriorCond)) {
        DebugLoc dl = getBranchDebugLoc(PrevBB);
        TII->RemoveBranch(PrevBB);
        TII->InsertBranch(PrevBB, PriorFBB, nullptr, NewPriorCond, dl);
        MadeChange = true;
        ++NumBranchOpts;
        goto ReoptimizeBlock;
      }
    }

    // If this block has no successors (e.g. it is a return block or ends with
    // a call to a no-return function like abort or __cxa_throw) and if the pred
    // falls through into this block, and if it would otherwise fall through
    // into the block after this, move this block to the end of the function.
    //
    // We consider it more likely that execution will stay in the function (e.g.
    // due to loops) than it is to exit it.  This asserts in loops etc, moving
    // the assert condition out of the loop body.
    if (MBB->succ_empty() && !PriorCond.empty() && !PriorFBB &&
        MachineFunction::iterator(PriorTBB) == FallThrough &&
        !MBB->canFallThrough()) {
      bool DoTransform = true;

      // We have to be careful that the succs of PredBB aren't both no-successor
      // blocks.  If neither have successors and if PredBB is the second from
      // last block in the function, we'd just keep swapping the two blocks for
      // last.  Only do the swap if one is clearly better to fall through than
      // the other.
      if (FallThrough == --MF.end() &&
          !IsBetterFallthrough(PriorTBB, MBB))
        DoTransform = false;

      if (DoTransform) {
        // Reverse the branch so we will fall through on the previous true cond.
        SmallVector<MachineOperand, 4> NewPriorCond(PriorCond);
        if (!TII->ReverseBranchCondition(NewPriorCond)) {
          DEBUG(dbgs() << "\nMoving MBB: " << *MBB
                       << "To make fallthrough to: " << *PriorTBB << "\n");

          DebugLoc dl = getBranchDebugLoc(PrevBB);
          TII->RemoveBranch(PrevBB);
          TII->InsertBranch(PrevBB, MBB, nullptr, NewPriorCond, dl);

          // Move this block to the end of the function.
          MBB->moveAfter(--MF.end());
          MadeChange = true;
          ++NumBranchOpts;
          return MadeChange;
        }
      }
    }
  }

  // Analyze the branch in the current block.
  MachineBasicBlock *CurTBB = nullptr, *CurFBB = nullptr;
  SmallVector<MachineOperand, 4> CurCond;
  bool CurUnAnalyzable= TII->AnalyzeBranch(*MBB, CurTBB, CurFBB, CurCond, true);
  if (!CurUnAnalyzable) {
    // If the CFG for the prior block has extra edges, remove them.
    MadeChange |= MBB->CorrectExtraCFGEdges(CurTBB, CurFBB, !CurCond.empty());

    // If this is a two-way branch, and the FBB branches to this block, reverse
    // the condition so the single-basic-block loop is faster.  Instead of:
    //    Loop: xxx; jcc Out; jmp Loop
    // we want:
    //    Loop: xxx; jncc Loop; jmp Out
    if (CurTBB && CurFBB && CurFBB == MBB && CurTBB != MBB) {
      SmallVector<MachineOperand, 4> NewCond(CurCond);
      if (!TII->ReverseBranchCondition(NewCond)) {
        DebugLoc dl = getBranchDebugLoc(*MBB);
        TII->RemoveBranch(*MBB);
        TII->InsertBranch(*MBB, CurFBB, CurTBB, NewCond, dl);
        MadeChange = true;
        ++NumBranchOpts;
        goto ReoptimizeBlock;
      }
    }

    // If this branch is the only thing in its block, see if we can forward
    // other blocks across it.
    if (CurTBB && CurCond.empty() && !CurFBB &&
        IsBranchOnlyBlock(MBB) && CurTBB != MBB &&
        !MBB->hasAddressTaken()) {
      DebugLoc dl = getBranchDebugLoc(*MBB);
      // This block may contain just an unconditional branch.  Because there can
      // be 'non-branch terminators' in the block, try removing the branch and
      // then seeing if the block is empty.
      TII->RemoveBranch(*MBB);
      // If the only things remaining in the block are debug info, remove these
      // as well, so this will behave the same as an empty block in non-debug
      // mode.
      if (IsEmptyBlock(MBB)) {
        // Make the block empty, losing the debug info (we could probably
        // improve this in some cases.)
        MBB->erase(MBB->begin(), MBB->end());
      }
      // If this block is just an unconditional branch to CurTBB, we can
      // usually completely eliminate the block.  The only case we cannot
      // completely eliminate the block is when the block before this one
      // falls through into MBB and we can't understand the prior block's branch
      // condition.
      if (MBB->empty()) {
        bool PredHasNoFallThrough = !PrevBB.canFallThrough();
        if (PredHasNoFallThrough || !PriorUnAnalyzable ||
            !PrevBB.isSuccessor(MBB)) {
          // If the prior block falls through into us, turn it into an
          // explicit branch to us to make updates simpler.
          if (!PredHasNoFallThrough && PrevBB.isSuccessor(MBB) &&
              PriorTBB != MBB && PriorFBB != MBB) {
            if (!PriorTBB) {
              assert(PriorCond.empty() && !PriorFBB &&
                     "Bad branch analysis");
              PriorTBB = MBB;
            } else {
              assert(!PriorFBB && "Machine CFG out of date!");
              PriorFBB = MBB;
            }
            DebugLoc pdl = getBranchDebugLoc(PrevBB);
            TII->RemoveBranch(PrevBB);
            TII->InsertBranch(PrevBB, PriorTBB, PriorFBB, PriorCond, pdl);
          }

          // Iterate through all the predecessors, revectoring each in-turn.
          size_t PI = 0;
          bool DidChange = false;
          bool HasBranchToSelf = false;
          while(PI != MBB->pred_size()) {
            MachineBasicBlock *PMBB = *(MBB->pred_begin() + PI);
            if (PMBB == MBB) {
              // If this block has an uncond branch to itself, leave it.
              ++PI;
              HasBranchToSelf = true;
            } else {
              DidChange = true;
              PMBB->ReplaceUsesOfBlockWith(MBB, CurTBB);
              // If this change resulted in PMBB ending in a conditional
              // branch where both conditions go to the same destination,
              // change this to an unconditional branch (and fix the CFG).
              MachineBasicBlock *NewCurTBB = nullptr, *NewCurFBB = nullptr;
              SmallVector<MachineOperand, 4> NewCurCond;
              bool NewCurUnAnalyzable = TII->AnalyzeBranch(*PMBB, NewCurTBB,
                      NewCurFBB, NewCurCond, true);
              if (!NewCurUnAnalyzable && NewCurTBB && NewCurTBB == NewCurFBB) {
                DebugLoc pdl = getBranchDebugLoc(*PMBB);
                TII->RemoveBranch(*PMBB);
                NewCurCond.clear();
                TII->InsertBranch(*PMBB, NewCurTBB, nullptr, NewCurCond, pdl);
                MadeChange = true;
                ++NumBranchOpts;
                PMBB->CorrectExtraCFGEdges(NewCurTBB, nullptr, false);
              }
            }
          }

          // Change any jumptables to go to the new MBB.
          if (MachineJumpTableInfo *MJTI = MF.getJumpTableInfo())
            MJTI->ReplaceMBBInJumpTables(MBB, CurTBB);
          if (DidChange) {
            ++NumBranchOpts;
            MadeChange = true;
            if (!HasBranchToSelf) return MadeChange;
          }
        }
      }

      // Add the branch back if the block is more than just an uncond branch.
      TII->InsertBranch(*MBB, CurTBB, nullptr, CurCond, dl);
    }
  }

  // If the prior block doesn't fall through into this block, and if this
  // block doesn't fall through into some other block, see if we can find a
  // place to move this block where a fall-through will happen.
  if (!PrevBB.canFallThrough()) {

    // Now we know that there was no fall-through into this block, check to
    // see if it has a fall-through into its successor.
    bool CurFallsThru = MBB->canFallThrough();

    if (!MBB->isLandingPad()) {
      // Check all the predecessors of this block.  If one of them has no fall
      // throughs, move this block right after it.
      for (MachineBasicBlock::pred_iterator PI = MBB->pred_begin(),
           E = MBB->pred_end(); PI != E; ++PI) {
        // Analyze the branch at the end of the pred.
        MachineBasicBlock *PredBB = *PI;
        MachineFunction::iterator PredFallthrough = PredBB; ++PredFallthrough;
        MachineBasicBlock *PredTBB = nullptr, *PredFBB = nullptr;
        SmallVector<MachineOperand, 4> PredCond;
        if (PredBB != MBB && !PredBB->canFallThrough() &&
            !TII->AnalyzeBranch(*PredBB, PredTBB, PredFBB, PredCond, true)
            && (!CurFallsThru || !CurTBB || !CurFBB)
            && (!CurFallsThru || MBB->getNumber() >= PredBB->getNumber())) {
          // If the current block doesn't fall through, just move it.
          // If the current block can fall through and does not end with a
          // conditional branch, we need to append an unconditional jump to
          // the (current) next block.  To avoid a possible compile-time
          // infinite loop, move blocks only backward in this case.
          // Also, if there are already 2 branches here, we cannot add a third;
          // this means we have the case
          // Bcc next
          // B elsewhere
          // next:
          if (CurFallsThru) {
            MachineBasicBlock *NextBB =
                std::next(MachineFunction::iterator(MBB));
            CurCond.clear();
            TII->InsertBranch(*MBB, NextBB, nullptr, CurCond, DebugLoc());
          }
          MBB->moveAfter(PredBB);
          MadeChange = true;
          goto ReoptimizeBlock;
        }
      }
    }

    if (!CurFallsThru) {
      // Check all successors to see if we can move this block before it.
      for (MachineBasicBlock::succ_iterator SI = MBB->succ_begin(),
           E = MBB->succ_end(); SI != E; ++SI) {
        // Analyze the branch at the end of the block before the succ.
        MachineBasicBlock *SuccBB = *SI;
        MachineFunction::iterator SuccPrev = SuccBB; --SuccPrev;

        // If this block doesn't already fall-through to that successor, and if
        // the succ doesn't already have a block that can fall through into it,
        // and if the successor isn't an EH destination, we can arrange for the
        // fallthrough to happen.
        if (SuccBB != MBB && &*SuccPrev != MBB &&
            !SuccPrev->canFallThrough() && !CurUnAnalyzable &&
            !SuccBB->isLandingPad()) {
          MBB->moveBefore(SuccBB);
          MadeChange = true;
          goto ReoptimizeBlock;
        }
      }

      // Okay, there is no really great place to put this block.  If, however,
      // the block before this one would be a fall-through if this block were
      // removed, move this block to the end of the function.
      MachineBasicBlock *PrevTBB = nullptr, *PrevFBB = nullptr;
      SmallVector<MachineOperand, 4> PrevCond;
      if (FallThrough != MF.end() &&
          !TII->AnalyzeBranch(PrevBB, PrevTBB, PrevFBB, PrevCond, true) &&
          PrevBB.isSuccessor(FallThrough)) {
        MBB->moveAfter(--MF.end());
        MadeChange = true;
        return MadeChange;
      }
    }
  }

  return MadeChange;
}

//===----------------------------------------------------------------------===//
//  Hoist Common Code
//===----------------------------------------------------------------------===//

/// HoistCommonCode - Hoist common instruction sequences at the start of basic
/// blocks to their common predecessor.
bool BranchFolder::HoistCommonCode(MachineFunction &MF) {
  bool MadeChange = false;
  for (MachineFunction::iterator I = MF.begin(), E = MF.end(); I != E; ) {
    MachineBasicBlock *MBB = I++;
    MadeChange |= HoistCommonCodeInSuccs(MBB);
  }

  return MadeChange;
}

/// findFalseBlock - BB has a fallthrough. Find its 'false' successor given
/// its 'true' successor.
static MachineBasicBlock *findFalseBlock(MachineBasicBlock *BB,
                                         MachineBasicBlock *TrueBB) {
  for (MachineBasicBlock::succ_iterator SI = BB->succ_begin(),
         E = BB->succ_end(); SI != E; ++SI) {
    MachineBasicBlock *SuccBB = *SI;
    if (SuccBB != TrueBB)
      return SuccBB;
  }
  return nullptr;
}

/// findHoistingInsertPosAndDeps - Find the location to move common instructions
/// in successors to. The location is usually just before the terminator,
/// however if the terminator is a conditional branch and its previous
/// instruction is the flag setting instruction, the previous instruction is
/// the preferred location. This function also gathers uses and defs of the
/// instructions from the insertion point to the end of the block. The data is
/// used by HoistCommonCodeInSuccs to ensure safety.
static
MachineBasicBlock::iterator findHoistingInsertPosAndDeps(MachineBasicBlock *MBB,
                                                  const TargetInstrInfo *TII,
                                                  const TargetRegisterInfo *TRI,
                                                  SmallSet<unsigned,4> &Uses,
                                                  SmallSet<unsigned,4> &Defs) {
  MachineBasicBlock::iterator Loc = MBB->getFirstTerminator();
  if (!TII->isUnpredicatedTerminator(Loc))
    return MBB->end();

  for (unsigned i = 0, e = Loc->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = Loc->getOperand(i);
    if (!MO.isReg())
      continue;
    unsigned Reg = MO.getReg();
    if (!Reg)
      continue;
    if (MO.isUse()) {
      for (MCRegAliasIterator AI(Reg, TRI, true); AI.isValid(); ++AI)
        Uses.insert(*AI);
    } else {
      if (!MO.isDead())
        // Don't try to hoist code in the rare case the terminator defines a
        // register that is later used.
        return MBB->end();

      // If the terminator defines a register, make sure we don't hoist
      // the instruction whose def might be clobbered by the terminator.
      for (MCRegAliasIterator AI(Reg, TRI, true); AI.isValid(); ++AI)
        Defs.insert(*AI);
    }
  }

  if (Uses.empty())
    return Loc;
  if (Loc == MBB->begin())
    return MBB->end();

  // The terminator is probably a conditional branch, try not to separate the
  // branch from condition setting instruction.
  MachineBasicBlock::iterator PI = Loc;
  --PI;
  while (PI != MBB->begin() && PI->isDebugValue())
    --PI;

  bool IsDef = false;
  for (unsigned i = 0, e = PI->getNumOperands(); !IsDef && i != e; ++i) {
    const MachineOperand &MO = PI->getOperand(i);
    // If PI has a regmask operand, it is probably a call. Separate away.
    if (MO.isRegMask())
      return Loc;
    if (!MO.isReg() || MO.isUse())
      continue;
    unsigned Reg = MO.getReg();
    if (!Reg)
      continue;
    if (Uses.count(Reg))
      IsDef = true;
  }
  if (!IsDef)
    // The condition setting instruction is not just before the conditional
    // branch.
    return Loc;

  // Be conservative, don't insert instruction above something that may have
  // side-effects. And since it's potentially bad to separate flag setting
  // instruction from the conditional branch, just abort the optimization
  // completely.
  // Also avoid moving code above predicated instruction since it's hard to
  // reason about register liveness with predicated instruction.
  bool DontMoveAcrossStore = true;
  if (!PI->isSafeToMove(nullptr, DontMoveAcrossStore) || TII->isPredicated(PI))
    return MBB->end();


  // Find out what registers are live. Note this routine is ignoring other live
  // registers which are only used by instructions in successor blocks.
  for (unsigned i = 0, e = PI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = PI->getOperand(i);
    if (!MO.isReg())
      continue;
    unsigned Reg = MO.getReg();
    if (!Reg)
      continue;
    if (MO.isUse()) {
      for (MCRegAliasIterator AI(Reg, TRI, true); AI.isValid(); ++AI)
        Uses.insert(*AI);
    } else {
      if (Uses.erase(Reg)) {
        for (MCSubRegIterator SubRegs(Reg, TRI); SubRegs.isValid(); ++SubRegs)
          Uses.erase(*SubRegs); // Use sub-registers to be conservative
      }
      for (MCRegAliasIterator AI(Reg, TRI, true); AI.isValid(); ++AI)
        Defs.insert(*AI);
    }
  }

  return PI;
}

/// HoistCommonCodeInSuccs - If the successors of MBB has common instruction
/// sequence at the start of the function, move the instructions before MBB
/// terminator if it's legal.
bool BranchFolder::HoistCommonCodeInSuccs(MachineBasicBlock *MBB) {
  MachineBasicBlock *TBB = nullptr, *FBB = nullptr;
  SmallVector<MachineOperand, 4> Cond;
  if (TII->AnalyzeBranch(*MBB, TBB, FBB, Cond, true) || !TBB || Cond.empty())
    return false;

  if (!FBB) FBB = findFalseBlock(MBB, TBB);
  if (!FBB)
    // Malformed bcc? True and false blocks are the same?
    return false;

  // Restrict the optimization to cases where MBB is the only predecessor,
  // it is an obvious win.
  if (TBB->pred_size() > 1 || FBB->pred_size() > 1)
    return false;

  // Find a suitable position to hoist the common instructions to. Also figure
  // out which registers are used or defined by instructions from the insertion
  // point to the end of the block.
  SmallSet<unsigned, 4> Uses, Defs;
  MachineBasicBlock::iterator Loc =
    findHoistingInsertPosAndDeps(MBB, TII, TRI, Uses, Defs);
  if (Loc == MBB->end())
    return false;

  bool HasDups = false;
  SmallVector<unsigned, 4> LocalDefs;
  SmallSet<unsigned, 4> LocalDefsSet;
  MachineBasicBlock::iterator TIB = TBB->begin();
  MachineBasicBlock::iterator FIB = FBB->begin();
  MachineBasicBlock::iterator TIE = TBB->end();
  MachineBasicBlock::iterator FIE = FBB->end();
  while (TIB != TIE && FIB != FIE) {
    // Skip dbg_value instructions. These do not count.
    if (TIB->isDebugValue()) {
      while (TIB != TIE && TIB->isDebugValue())
        ++TIB;
      if (TIB == TIE)
        break;
    }
    if (FIB->isDebugValue()) {
      while (FIB != FIE && FIB->isDebugValue())
        ++FIB;
      if (FIB == FIE)
        break;
    }
    if (!TIB->isIdenticalTo(FIB, MachineInstr::CheckKillDead))
      break;

    if (TII->isPredicated(TIB))
      // Hard to reason about register liveness with predicated instruction.
      break;

    bool IsSafe = true;
    for (unsigned i = 0, e = TIB->getNumOperands(); i != e; ++i) {
      MachineOperand &MO = TIB->getOperand(i);
      // Don't attempt to hoist instructions with register masks.
      if (MO.isRegMask()) {
        IsSafe = false;
        break;
      }
      if (!MO.isReg())
        continue;
      unsigned Reg = MO.getReg();
      if (!Reg)
        continue;
      if (MO.isDef()) {
        if (Uses.count(Reg)) {
          // Avoid clobbering a register that's used by the instruction at
          // the point of insertion.
          IsSafe = false;
          break;
        }

        if (Defs.count(Reg) && !MO.isDead()) {
          // Don't hoist the instruction if the def would be clobber by the
          // instruction at the point insertion. FIXME: This is overly
          // conservative. It should be possible to hoist the instructions
          // in BB2 in the following example:
          // BB1:
          // r1, eflag = op1 r2, r3
          // brcc eflag
          //
          // BB2:
          // r1 = op2, ...
          //    = op3, r1<kill>
          IsSafe = false;
          break;
        }
      } else if (!LocalDefsSet.count(Reg)) {
        if (Defs.count(Reg)) {
          // Use is defined by the instruction at the point of insertion.
          IsSafe = false;
          break;
        }

        if (MO.isKill() && Uses.count(Reg))
          // Kills a register that's read by the instruction at the point of
          // insertion. Remove the kill marker.
          MO.setIsKill(false);
      }
    }
    if (!IsSafe)
      break;

    bool DontMoveAcrossStore = true;
    if (!TIB->isSafeToMove(nullptr, DontMoveAcrossStore))
      break;

    // Remove kills from LocalDefsSet, these registers had short live ranges.
    for (unsigned i = 0, e = TIB->getNumOperands(); i != e; ++i) {
      MachineOperand &MO = TIB->getOperand(i);
      if (!MO.isReg() || !MO.isUse() || !MO.isKill())
        continue;
      unsigned Reg = MO.getReg();
      if (!Reg || !LocalDefsSet.count(Reg))
        continue;
      for (MCRegAliasIterator AI(Reg, TRI, true); AI.isValid(); ++AI)
        LocalDefsSet.erase(*AI);
    }

    // Track local defs so we can update liveins.
    for (unsigned i = 0, e = TIB->getNumOperands(); i != e; ++i) {
      MachineOperand &MO = TIB->getOperand(i);
      if (!MO.isReg() || !MO.isDef() || MO.isDead())
        continue;
      unsigned Reg = MO.getReg();
      if (!Reg)
        continue;
      LocalDefs.push_back(Reg);
      for (MCRegAliasIterator AI(Reg, TRI, true); AI.isValid(); ++AI)
        LocalDefsSet.insert(*AI);
    }

    HasDups = true;
    ++TIB;
    ++FIB;
  }

  if (!HasDups)
    return false;

  MBB->splice(Loc, TBB, TBB->begin(), TIB);
  FBB->erase(FBB->begin(), FIB);

  // Update livein's.
  for (unsigned i = 0, e = LocalDefs.size(); i != e; ++i) {
    unsigned Def = LocalDefs[i];
    if (LocalDefsSet.count(Def)) {
      TBB->addLiveIn(Def);
      FBB->addLiveIn(Def);
    }
  }

  ++NumHoist;
  return true;
}
