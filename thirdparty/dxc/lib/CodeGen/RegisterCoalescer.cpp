//===- RegisterCoalescer.cpp - Generic Register Coalescing Interface -------==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the generic RegisterCoalescer interface which
// is used as the common interface used by all clients and
// implementations of register coalescing.
//
//===----------------------------------------------------------------------===//

#include "RegisterCoalescer.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/LiveRangeEdit.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/IR/Value.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include <algorithm>
#include <cmath>
using namespace llvm;

#define DEBUG_TYPE "regalloc"

STATISTIC(numJoins    , "Number of interval joins performed");
STATISTIC(numCrossRCs , "Number of cross class joins performed");
STATISTIC(numCommutes , "Number of instruction commuting performed");
STATISTIC(numExtends  , "Number of copies extended");
STATISTIC(NumReMats   , "Number of instructions re-materialized");
STATISTIC(NumInflated , "Number of register classes inflated");
STATISTIC(NumLaneConflicts, "Number of dead lane conflicts tested");
STATISTIC(NumLaneResolves,  "Number of dead lane conflicts resolved");

static cl::opt<bool>
EnableJoining("join-liveintervals",
              cl::desc("Coalesce copies (default=true)"),
              cl::init(true));

static cl::opt<bool> UseTerminalRule("terminal-rule",
                                     cl::desc("Apply the terminal rule"),
                                     cl::init(false), cl::Hidden);

/// Temporary flag to test critical edge unsplitting.
static cl::opt<bool>
EnableJoinSplits("join-splitedges",
  cl::desc("Coalesce copies on split edges (default=subtarget)"), cl::Hidden);

/// Temporary flag to test global copy optimization.
static cl::opt<cl::boolOrDefault>
EnableGlobalCopies("join-globalcopies",
  cl::desc("Coalesce copies that span blocks (default=subtarget)"),
  cl::init(cl::BOU_UNSET), cl::Hidden);

static cl::opt<bool>
VerifyCoalescing("verify-coalescing",
         cl::desc("Verify machine instrs before and after register coalescing"),
         cl::Hidden);

namespace {
  class RegisterCoalescer : public MachineFunctionPass,
                            private LiveRangeEdit::Delegate {
    MachineFunction* MF;
    MachineRegisterInfo* MRI;
    const TargetMachine* TM;
    const TargetRegisterInfo* TRI;
    const TargetInstrInfo* TII;
    LiveIntervals *LIS;
    const MachineLoopInfo* Loops;
    AliasAnalysis *AA;
    RegisterClassInfo RegClassInfo;

    /// A LaneMask to remember on which subregister live ranges we need to call
    /// shrinkToUses() later.
    unsigned ShrinkMask;

    /// True if the main range of the currently coalesced intervals should be
    /// checked for smaller live intervals.
    bool ShrinkMainRange;

    /// \brief True if the coalescer should aggressively coalesce global copies
    /// in favor of keeping local copies.
    bool JoinGlobalCopies;

    /// \brief True if the coalescer should aggressively coalesce fall-thru
    /// blocks exclusively containing copies.
    bool JoinSplitEdges;

    /// Copy instructions yet to be coalesced.
    SmallVector<MachineInstr*, 8> WorkList;
    SmallVector<MachineInstr*, 8> LocalWorkList;

    /// Set of instruction pointers that have been erased, and
    /// that may be present in WorkList.
    SmallPtrSet<MachineInstr*, 8> ErasedInstrs;

    /// Dead instructions that are about to be deleted.
    SmallVector<MachineInstr*, 8> DeadDefs;

    /// Virtual registers to be considered for register class inflation.
    SmallVector<unsigned, 8> InflateRegs;

    /// Recursively eliminate dead defs in DeadDefs.
    void eliminateDeadDefs();

    /// LiveRangeEdit callback for eliminateDeadDefs().
    void LRE_WillEraseInstruction(MachineInstr *MI) override;

    /// Coalesce the LocalWorkList.
    void coalesceLocals();

    /// Join compatible live intervals
    void joinAllIntervals();

    /// Coalesce copies in the specified MBB, putting
    /// copies that cannot yet be coalesced into WorkList.
    void copyCoalesceInMBB(MachineBasicBlock *MBB);

    /// Tries to coalesce all copies in CurrList. Returns true if any progress
    /// was made.
    bool copyCoalesceWorkList(MutableArrayRef<MachineInstr*> CurrList);

    /// Attempt to join intervals corresponding to SrcReg/DstReg, which are the
    /// src/dst of the copy instruction CopyMI.  This returns true if the copy
    /// was successfully coalesced away. If it is not currently possible to
    /// coalesce this interval, but it may be possible if other things get
    /// coalesced, then it returns true by reference in 'Again'.
    bool joinCopy(MachineInstr *TheCopy, bool &Again);

    /// Attempt to join these two intervals.  On failure, this
    /// returns false.  The output "SrcInt" will not have been modified, so we
    /// can use this information below to update aliases.
    bool joinIntervals(CoalescerPair &CP);

    /// Attempt joining two virtual registers. Return true on success.
    bool joinVirtRegs(CoalescerPair &CP);

    /// Attempt joining with a reserved physreg.
    bool joinReservedPhysReg(CoalescerPair &CP);

    /// Add the LiveRange @p ToMerge as a subregister liverange of @p LI.
    /// Subranges in @p LI which only partially interfere with the desired
    /// LaneMask are split as necessary. @p LaneMask are the lanes that
    /// @p ToMerge will occupy in the coalescer register. @p LI has its subrange
    /// lanemasks already adjusted to the coalesced register.
    /// @returns false if live range conflicts couldn't get resolved.
    bool mergeSubRangeInto(LiveInterval &LI, const LiveRange &ToMerge,
                           unsigned LaneMask, CoalescerPair &CP);

    /// Join the liveranges of two subregisters. Joins @p RRange into
    /// @p LRange, @p RRange may be invalid afterwards.
    /// @returns false if live range conflicts couldn't get resolved.
    bool joinSubRegRanges(LiveRange &LRange, LiveRange &RRange,
                          unsigned LaneMask, const CoalescerPair &CP);

    /// We found a non-trivially-coalescable copy. If the source value number is
    /// defined by a copy from the destination reg see if we can merge these two
    /// destination reg valno# into a single value number, eliminating a copy.
    /// This returns true if an interval was modified.
    bool adjustCopiesBackFrom(const CoalescerPair &CP, MachineInstr *CopyMI);

    /// Return true if there are definitions of IntB
    /// other than BValNo val# that can reach uses of AValno val# of IntA.
    bool hasOtherReachingDefs(LiveInterval &IntA, LiveInterval &IntB,
                              VNInfo *AValNo, VNInfo *BValNo);

    /// We found a non-trivially-coalescable copy.
    /// If the source value number is defined by a commutable instruction and
    /// its other operand is coalesced to the copy dest register, see if we
    /// can transform the copy into a noop by commuting the definition.
    /// This returns true if an interval was modified.
    bool removeCopyByCommutingDef(const CoalescerPair &CP,MachineInstr *CopyMI);

    /// If the source of a copy is defined by a
    /// trivial computation, replace the copy by rematerialize the definition.
    bool reMaterializeTrivialDef(const CoalescerPair &CP, MachineInstr *CopyMI,
                                 bool &IsDefCopy);

    /// Return true if a copy involving a physreg should be joined.
    bool canJoinPhys(const CoalescerPair &CP);

    /// Replace all defs and uses of SrcReg to DstReg and update the subregister
    /// number if it is not zero. If DstReg is a physical register and the
    /// existing subregister number of the def / use being updated is not zero,
    /// make sure to set it to the correct physical subregister.
    void updateRegDefsUses(unsigned SrcReg, unsigned DstReg, unsigned SubIdx);

    /// Handle copies of undef values.
    /// Returns true if @p CopyMI was a copy of an undef value and eliminated.
    bool eliminateUndefCopy(MachineInstr *CopyMI);

    /// Check whether or not we should apply the terminal rule on the
    /// destination (Dst) of \p Copy.
    /// When the terminal rule applies, Copy is not profitable to
    /// coalesce.
    /// Dst is terminal if it has exactly one affinity (Dst, Src) and
    /// at least one interference (Dst, Dst2). If Dst is terminal, the
    /// terminal rule consists in checking that at least one of
    /// interfering node, say Dst2, has an affinity of equal or greater
    /// weight with Src.
    /// In that case, Dst2 and Dst will not be able to be both coalesced
    /// with Src. Since Dst2 exposes more coalescing opportunities than
    /// Dst, we can drop \p Copy.
    bool applyTerminalRule(const MachineInstr &Copy) const;

    /// Check whether or not \p LI is composed by multiple connected
    /// components and if that is the case, fix that.
    void splitNewRanges(LiveInterval *LI) {
      ConnectedVNInfoEqClasses ConEQ(*LIS);
      unsigned NumComps = ConEQ.Classify(LI);
      if (NumComps <= 1)
        return;
      SmallVector<LiveInterval*, 8> NewComps(1, LI);
      for (unsigned i = 1; i != NumComps; ++i) {
        unsigned VReg = MRI->createVirtualRegister(MRI->getRegClass(LI->reg));
        NewComps.push_back(&LIS->createEmptyInterval(VReg));
      }

      ConEQ.Distribute(&NewComps[0], *MRI);
    }

    /// Wrapper method for \see LiveIntervals::shrinkToUses.
    /// This method does the proper fixing of the live-ranges when the afore
    /// mentioned method returns true.
    void shrinkToUses(LiveInterval *LI,
                      SmallVectorImpl<MachineInstr * > *Dead = nullptr) {
      if (LIS->shrinkToUses(LI, Dead))
        // We may have created multiple connected components, split them.
        splitNewRanges(LI);
    }

  public:
    static char ID; ///< Class identification, replacement for typeinfo
    RegisterCoalescer() : MachineFunctionPass(ID) {
      initializeRegisterCoalescerPass(*PassRegistry::getPassRegistry());
    }

    void getAnalysisUsage(AnalysisUsage &AU) const override;

    void releaseMemory() override;

    /// This is the pass entry point.
    bool runOnMachineFunction(MachineFunction&) override;

    /// Implement the dump method.
    void print(raw_ostream &O, const Module* = nullptr) const override;
  };
} // end anonymous namespace

char &llvm::RegisterCoalescerID = RegisterCoalescer::ID;

INITIALIZE_PASS_BEGIN(RegisterCoalescer, "simple-register-coalescing",
                      "Simple Register Coalescing", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervals)
INITIALIZE_PASS_DEPENDENCY(SlotIndexes)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfo)
INITIALIZE_AG_DEPENDENCY(AliasAnalysis)
INITIALIZE_PASS_END(RegisterCoalescer, "simple-register-coalescing",
                    "Simple Register Coalescing", false, false)

char RegisterCoalescer::ID = 0;

static bool isMoveInstr(const TargetRegisterInfo &tri, const MachineInstr *MI,
                        unsigned &Src, unsigned &Dst,
                        unsigned &SrcSub, unsigned &DstSub) {
  if (MI->isCopy()) {
    Dst = MI->getOperand(0).getReg();
    DstSub = MI->getOperand(0).getSubReg();
    Src = MI->getOperand(1).getReg();
    SrcSub = MI->getOperand(1).getSubReg();
  } else if (MI->isSubregToReg()) {
    Dst = MI->getOperand(0).getReg();
    DstSub = tri.composeSubRegIndices(MI->getOperand(0).getSubReg(),
                                      MI->getOperand(3).getImm());
    Src = MI->getOperand(2).getReg();
    SrcSub = MI->getOperand(2).getSubReg();
  } else
    return false;
  return true;
}

/// Return true if this block should be vacated by the coalescer to eliminate
/// branches. The important cases to handle in the coalescer are critical edges
/// split during phi elimination which contain only copies. Simple blocks that
/// contain non-branches should also be vacated, but this can be handled by an
/// earlier pass similar to early if-conversion.
static bool isSplitEdge(const MachineBasicBlock *MBB) {
  if (MBB->pred_size() != 1 || MBB->succ_size() != 1)
    return false;

  for (const auto &MI : *MBB) {
    if (!MI.isCopyLike() && !MI.isUnconditionalBranch())
      return false;
  }
  return true;
}

bool CoalescerPair::setRegisters(const MachineInstr *MI) {
  SrcReg = DstReg = 0;
  SrcIdx = DstIdx = 0;
  NewRC = nullptr;
  Flipped = CrossClass = false;

  unsigned Src, Dst, SrcSub, DstSub;
  if (!isMoveInstr(TRI, MI, Src, Dst, SrcSub, DstSub))
    return false;
  Partial = SrcSub || DstSub;

  // If one register is a physreg, it must be Dst.
  if (TargetRegisterInfo::isPhysicalRegister(Src)) {
    if (TargetRegisterInfo::isPhysicalRegister(Dst))
      return false;
    std::swap(Src, Dst);
    std::swap(SrcSub, DstSub);
    Flipped = true;
  }

  const MachineRegisterInfo &MRI = MI->getParent()->getParent()->getRegInfo();

  if (TargetRegisterInfo::isPhysicalRegister(Dst)) {
    // Eliminate DstSub on a physreg.
    if (DstSub) {
      Dst = TRI.getSubReg(Dst, DstSub);
      if (!Dst) return false;
      DstSub = 0;
    }

    // Eliminate SrcSub by picking a corresponding Dst superregister.
    if (SrcSub) {
      Dst = TRI.getMatchingSuperReg(Dst, SrcSub, MRI.getRegClass(Src));
      if (!Dst) return false;
    } else if (!MRI.getRegClass(Src)->contains(Dst)) {
      return false;
    }
  } else {
    // Both registers are virtual.
    const TargetRegisterClass *SrcRC = MRI.getRegClass(Src);
    const TargetRegisterClass *DstRC = MRI.getRegClass(Dst);

    // Both registers have subreg indices.
    if (SrcSub && DstSub) {
      // Copies between different sub-registers are never coalescable.
      if (Src == Dst && SrcSub != DstSub)
        return false;

      NewRC = TRI.getCommonSuperRegClass(SrcRC, SrcSub, DstRC, DstSub,
                                         SrcIdx, DstIdx);
      if (!NewRC)
        return false;
    } else if (DstSub) {
      // SrcReg will be merged with a sub-register of DstReg.
      SrcIdx = DstSub;
      NewRC = TRI.getMatchingSuperRegClass(DstRC, SrcRC, DstSub);
    } else if (SrcSub) {
      // DstReg will be merged with a sub-register of SrcReg.
      DstIdx = SrcSub;
      NewRC = TRI.getMatchingSuperRegClass(SrcRC, DstRC, SrcSub);
    } else {
      // This is a straight copy without sub-registers.
      NewRC = TRI.getCommonSubClass(DstRC, SrcRC);
    }

    // The combined constraint may be impossible to satisfy.
    if (!NewRC)
      return false;

    // Prefer SrcReg to be a sub-register of DstReg.
    // FIXME: Coalescer should support subregs symmetrically.
    if (DstIdx && !SrcIdx) {
      std::swap(Src, Dst);
      std::swap(SrcIdx, DstIdx);
      Flipped = !Flipped;
    }

    CrossClass = NewRC != DstRC || NewRC != SrcRC;
  }
  // Check our invariants
  assert(TargetRegisterInfo::isVirtualRegister(Src) && "Src must be virtual");
  assert(!(TargetRegisterInfo::isPhysicalRegister(Dst) && DstSub) &&
         "Cannot have a physical SubIdx");
  SrcReg = Src;
  DstReg = Dst;
  return true;
}

bool CoalescerPair::flip() {
  if (TargetRegisterInfo::isPhysicalRegister(DstReg))
    return false;
  std::swap(SrcReg, DstReg);
  std::swap(SrcIdx, DstIdx);
  Flipped = !Flipped;
  return true;
}

bool CoalescerPair::isCoalescable(const MachineInstr *MI) const {
  if (!MI)
    return false;
  unsigned Src, Dst, SrcSub, DstSub;
  if (!isMoveInstr(TRI, MI, Src, Dst, SrcSub, DstSub))
    return false;

  // Find the virtual register that is SrcReg.
  if (Dst == SrcReg) {
    std::swap(Src, Dst);
    std::swap(SrcSub, DstSub);
  } else if (Src != SrcReg) {
    return false;
  }

  // Now check that Dst matches DstReg.
  if (TargetRegisterInfo::isPhysicalRegister(DstReg)) {
    if (!TargetRegisterInfo::isPhysicalRegister(Dst))
      return false;
    assert(!DstIdx && !SrcIdx && "Inconsistent CoalescerPair state.");
    // DstSub could be set for a physreg from INSERT_SUBREG.
    if (DstSub)
      Dst = TRI.getSubReg(Dst, DstSub);
    // Full copy of Src.
    if (!SrcSub)
      return DstReg == Dst;
    // This is a partial register copy. Check that the parts match.
    return TRI.getSubReg(DstReg, SrcSub) == Dst;
  } else {
    // DstReg is virtual.
    if (DstReg != Dst)
      return false;
    // Registers match, do the subregisters line up?
    return TRI.composeSubRegIndices(SrcIdx, SrcSub) ==
           TRI.composeSubRegIndices(DstIdx, DstSub);
  }
}

void RegisterCoalescer::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  AU.addRequired<AliasAnalysis>();
  AU.addRequired<LiveIntervals>();
  AU.addPreserved<LiveIntervals>();
  AU.addPreserved<SlotIndexes>();
  AU.addRequired<MachineLoopInfo>();
  AU.addPreserved<MachineLoopInfo>();
  AU.addPreservedID(MachineDominatorsID);
  MachineFunctionPass::getAnalysisUsage(AU);
}

void RegisterCoalescer::eliminateDeadDefs() {
  SmallVector<unsigned, 8> NewRegs;
  LiveRangeEdit(nullptr, NewRegs, *MF, *LIS,
                nullptr, this).eliminateDeadDefs(DeadDefs);
}

void RegisterCoalescer::LRE_WillEraseInstruction(MachineInstr *MI) {
  // MI may be in WorkList. Make sure we don't visit it.
  ErasedInstrs.insert(MI);
}

bool RegisterCoalescer::adjustCopiesBackFrom(const CoalescerPair &CP,
                                             MachineInstr *CopyMI) {
  assert(!CP.isPartial() && "This doesn't work for partial copies.");
  assert(!CP.isPhys() && "This doesn't work for physreg copies.");

  LiveInterval &IntA =
    LIS->getInterval(CP.isFlipped() ? CP.getDstReg() : CP.getSrcReg());
  LiveInterval &IntB =
    LIS->getInterval(CP.isFlipped() ? CP.getSrcReg() : CP.getDstReg());
  SlotIndex CopyIdx = LIS->getInstructionIndex(CopyMI).getRegSlot();

  // We have a non-trivially-coalescable copy with IntA being the source and
  // IntB being the dest, thus this defines a value number in IntB.  If the
  // source value number (in IntA) is defined by a copy from B, see if we can
  // merge these two pieces of B into a single value number, eliminating a copy.
  // For example:
  //
  //  A3 = B0
  //    ...
  //  B1 = A3      <- this copy
  //
  // In this case, B0 can be extended to where the B1 copy lives, allowing the
  // B1 value number to be replaced with B0 (which simplifies the B
  // liveinterval).

  // BValNo is a value number in B that is defined by a copy from A.  'B1' in
  // the example above.
  LiveInterval::iterator BS = IntB.FindSegmentContaining(CopyIdx);
  if (BS == IntB.end()) return false;
  VNInfo *BValNo = BS->valno;

  // Get the location that B is defined at.  Two options: either this value has
  // an unknown definition point or it is defined at CopyIdx.  If unknown, we
  // can't process it.
  if (BValNo->def != CopyIdx) return false;

  // AValNo is the value number in A that defines the copy, A3 in the example.
  SlotIndex CopyUseIdx = CopyIdx.getRegSlot(true);
  LiveInterval::iterator AS = IntA.FindSegmentContaining(CopyUseIdx);
  // The live segment might not exist after fun with physreg coalescing.
  if (AS == IntA.end()) return false;
  VNInfo *AValNo = AS->valno;

  // If AValNo is defined as a copy from IntB, we can potentially process this.
  // Get the instruction that defines this value number.
  MachineInstr *ACopyMI = LIS->getInstructionFromIndex(AValNo->def);
  // Don't allow any partial copies, even if isCoalescable() allows them.
  if (!CP.isCoalescable(ACopyMI) || !ACopyMI->isFullCopy())
    return false;

  // Get the Segment in IntB that this value number starts with.
  LiveInterval::iterator ValS =
    IntB.FindSegmentContaining(AValNo->def.getPrevSlot());
  if (ValS == IntB.end())
    return false;

  // Make sure that the end of the live segment is inside the same block as
  // CopyMI.
  MachineInstr *ValSEndInst =
    LIS->getInstructionFromIndex(ValS->end.getPrevSlot());
  if (!ValSEndInst || ValSEndInst->getParent() != CopyMI->getParent())
    return false;

  // Okay, we now know that ValS ends in the same block that the CopyMI
  // live-range starts.  If there are no intervening live segments between them
  // in IntB, we can merge them.
  if (ValS+1 != BS) return false;

  DEBUG(dbgs() << "Extending: " << PrintReg(IntB.reg, TRI));

  SlotIndex FillerStart = ValS->end, FillerEnd = BS->start;
  // We are about to delete CopyMI, so need to remove it as the 'instruction
  // that defines this value #'. Update the valnum with the new defining
  // instruction #.
  BValNo->def = FillerStart;

  // Okay, we can merge them.  We need to insert a new liverange:
  // [ValS.end, BS.begin) of either value number, then we merge the
  // two value numbers.
  IntB.addSegment(LiveInterval::Segment(FillerStart, FillerEnd, BValNo));

  // Okay, merge "B1" into the same value number as "B0".
  if (BValNo != ValS->valno)
    IntB.MergeValueNumberInto(BValNo, ValS->valno);

  // Do the same for the subregister segments.
  for (LiveInterval::SubRange &S : IntB.subranges()) {
    VNInfo *SubBValNo = S.getVNInfoAt(CopyIdx);
    S.addSegment(LiveInterval::Segment(FillerStart, FillerEnd, SubBValNo));
    VNInfo *SubValSNo = S.getVNInfoAt(AValNo->def.getPrevSlot());
    if (SubBValNo != SubValSNo)
      S.MergeValueNumberInto(SubBValNo, SubValSNo);
  }

  DEBUG(dbgs() << "   result = " << IntB << '\n');

  // If the source instruction was killing the source register before the
  // merge, unset the isKill marker given the live range has been extended.
  int UIdx = ValSEndInst->findRegisterUseOperandIdx(IntB.reg, true);
  if (UIdx != -1) {
    ValSEndInst->getOperand(UIdx).setIsKill(false);
  }

  // Rewrite the copy. If the copy instruction was killing the destination
  // register before the merge, find the last use and trim the live range. That
  // will also add the isKill marker.
  CopyMI->substituteRegister(IntA.reg, IntB.reg, 0, *TRI);
  if (AS->end == CopyIdx)
    shrinkToUses(&IntA);

  ++numExtends;
  return true;
}

bool RegisterCoalescer::hasOtherReachingDefs(LiveInterval &IntA,
                                             LiveInterval &IntB,
                                             VNInfo *AValNo,
                                             VNInfo *BValNo) {
  // If AValNo has PHI kills, conservatively assume that IntB defs can reach
  // the PHI values.
  if (LIS->hasPHIKill(IntA, AValNo))
    return true;

  for (LiveRange::Segment &ASeg : IntA.segments) {
    if (ASeg.valno != AValNo) continue;
    LiveInterval::iterator BI =
      std::upper_bound(IntB.begin(), IntB.end(), ASeg.start);
    if (BI != IntB.begin())
      --BI;
    for (; BI != IntB.end() && ASeg.end >= BI->start; ++BI) {
      if (BI->valno == BValNo)
        continue;
      if (BI->start <= ASeg.start && BI->end > ASeg.start)
        return true;
      if (BI->start > ASeg.start && BI->start < ASeg.end)
        return true;
    }
  }
  return false;
}

/// Copy segements with value number @p SrcValNo from liverange @p Src to live
/// range @Dst and use value number @p DstValNo there.
static void addSegmentsWithValNo(LiveRange &Dst, VNInfo *DstValNo,
                                 const LiveRange &Src, const VNInfo *SrcValNo)
{
  for (const LiveRange::Segment &S : Src.segments) {
    if (S.valno != SrcValNo)
      continue;
    Dst.addSegment(LiveRange::Segment(S.start, S.end, DstValNo));
  }
}

bool RegisterCoalescer::removeCopyByCommutingDef(const CoalescerPair &CP,
                                                 MachineInstr *CopyMI) {
  assert(!CP.isPhys());

  LiveInterval &IntA =
      LIS->getInterval(CP.isFlipped() ? CP.getDstReg() : CP.getSrcReg());
  LiveInterval &IntB =
      LIS->getInterval(CP.isFlipped() ? CP.getSrcReg() : CP.getDstReg());

  // We found a non-trivially-coalescable copy with IntA being the source and
  // IntB being the dest, thus this defines a value number in IntB.  If the
  // source value number (in IntA) is defined by a commutable instruction and
  // its other operand is coalesced to the copy dest register, see if we can
  // transform the copy into a noop by commuting the definition. For example,
  //
  //  A3 = op A2 B0<kill>
  //    ...
  //  B1 = A3      <- this copy
  //    ...
  //     = op A3   <- more uses
  //
  // ==>
  //
  //  B2 = op B0 A2<kill>
  //    ...
  //  B1 = B2      <- now an identity copy
  //    ...
  //     = op B2   <- more uses

  // BValNo is a value number in B that is defined by a copy from A. 'B1' in
  // the example above.
  SlotIndex CopyIdx = LIS->getInstructionIndex(CopyMI).getRegSlot();
  VNInfo *BValNo = IntB.getVNInfoAt(CopyIdx);
  assert(BValNo != nullptr && BValNo->def == CopyIdx);

  // AValNo is the value number in A that defines the copy, A3 in the example.
  VNInfo *AValNo = IntA.getVNInfoAt(CopyIdx.getRegSlot(true));
  assert(AValNo && !AValNo->isUnused() && "COPY source not live");
  if (AValNo->isPHIDef())
    return false;
  MachineInstr *DefMI = LIS->getInstructionFromIndex(AValNo->def);
  if (!DefMI)
    return false;
  if (!DefMI->isCommutable())
    return false;
  // If DefMI is a two-address instruction then commuting it will change the
  // destination register.
  int DefIdx = DefMI->findRegisterDefOperandIdx(IntA.reg);
  assert(DefIdx != -1);
  unsigned UseOpIdx;
  if (!DefMI->isRegTiedToUseOperand(DefIdx, &UseOpIdx))
    return false;
  unsigned Op1, Op2, NewDstIdx;
  if (!TII->findCommutedOpIndices(DefMI, Op1, Op2))
    return false;
  if (Op1 == UseOpIdx)
    NewDstIdx = Op2;
  else if (Op2 == UseOpIdx)
    NewDstIdx = Op1;
  else
    return false;

  MachineOperand &NewDstMO = DefMI->getOperand(NewDstIdx);
  unsigned NewReg = NewDstMO.getReg();
  if (NewReg != IntB.reg || !IntB.Query(AValNo->def).isKill())
    return false;

  // Make sure there are no other definitions of IntB that would reach the
  // uses which the new definition can reach.
  if (hasOtherReachingDefs(IntA, IntB, AValNo, BValNo))
    return false;

  // If some of the uses of IntA.reg is already coalesced away, return false.
  // It's not possible to determine whether it's safe to perform the coalescing.
  for (MachineOperand &MO : MRI->use_nodbg_operands(IntA.reg)) {
    MachineInstr *UseMI = MO.getParent();
    unsigned OpNo = &MO - &UseMI->getOperand(0);
    SlotIndex UseIdx = LIS->getInstructionIndex(UseMI);
    LiveInterval::iterator US = IntA.FindSegmentContaining(UseIdx);
    if (US == IntA.end() || US->valno != AValNo)
      continue;
    // If this use is tied to a def, we can't rewrite the register.
    if (UseMI->isRegTiedToDefOperand(OpNo))
      return false;
  }

  DEBUG(dbgs() << "\tremoveCopyByCommutingDef: " << AValNo->def << '\t'
               << *DefMI);

  // At this point we have decided that it is legal to do this
  // transformation.  Start by commuting the instruction.
  MachineBasicBlock *MBB = DefMI->getParent();
  MachineInstr *NewMI = TII->commuteInstruction(DefMI);
  if (!NewMI)
    return false;
  if (TargetRegisterInfo::isVirtualRegister(IntA.reg) &&
      TargetRegisterInfo::isVirtualRegister(IntB.reg) &&
      !MRI->constrainRegClass(IntB.reg, MRI->getRegClass(IntA.reg)))
    return false;
  if (NewMI != DefMI) {
    LIS->ReplaceMachineInstrInMaps(DefMI, NewMI);
    MachineBasicBlock::iterator Pos = DefMI;
    MBB->insert(Pos, NewMI);
    MBB->erase(DefMI);
  }

  // If ALR and BLR overlaps and end of BLR extends beyond end of ALR, e.g.
  // A = or A, B
  // ...
  // B = A
  // ...
  // C = A<kill>
  // ...
  //   = B

  // Update uses of IntA of the specific Val# with IntB.
  for (MachineRegisterInfo::use_iterator UI = MRI->use_begin(IntA.reg),
                                         UE = MRI->use_end();
       UI != UE; /* ++UI is below because of possible MI removal */) {
    MachineOperand &UseMO = *UI;
    ++UI;
    if (UseMO.isUndef())
      continue;
    MachineInstr *UseMI = UseMO.getParent();
    if (UseMI->isDebugValue()) {
      // FIXME These don't have an instruction index.  Not clear we have enough
      // info to decide whether to do this replacement or not.  For now do it.
      UseMO.setReg(NewReg);
      continue;
    }
    SlotIndex UseIdx = LIS->getInstructionIndex(UseMI).getRegSlot(true);
    LiveInterval::iterator US = IntA.FindSegmentContaining(UseIdx);
    assert(US != IntA.end() && "Use must be live");
    if (US->valno != AValNo)
      continue;
    // Kill flags are no longer accurate. They are recomputed after RA.
    UseMO.setIsKill(false);
    if (TargetRegisterInfo::isPhysicalRegister(NewReg))
      UseMO.substPhysReg(NewReg, *TRI);
    else
      UseMO.setReg(NewReg);
    if (UseMI == CopyMI)
      continue;
    if (!UseMI->isCopy())
      continue;
    if (UseMI->getOperand(0).getReg() != IntB.reg ||
        UseMI->getOperand(0).getSubReg())
      continue;

    // This copy will become a noop. If it's defining a new val#, merge it into
    // BValNo.
    SlotIndex DefIdx = UseIdx.getRegSlot();
    VNInfo *DVNI = IntB.getVNInfoAt(DefIdx);
    if (!DVNI)
      continue;
    DEBUG(dbgs() << "\t\tnoop: " << DefIdx << '\t' << *UseMI);
    assert(DVNI->def == DefIdx);
    BValNo = IntB.MergeValueNumberInto(DVNI, BValNo);
    for (LiveInterval::SubRange &S : IntB.subranges()) {
      VNInfo *SubDVNI = S.getVNInfoAt(DefIdx);
      if (!SubDVNI)
        continue;
      VNInfo *SubBValNo = S.getVNInfoAt(CopyIdx);
      assert(SubBValNo->def == CopyIdx);
      S.MergeValueNumberInto(SubDVNI, SubBValNo);
    }

    ErasedInstrs.insert(UseMI);
    LIS->RemoveMachineInstrFromMaps(UseMI);
    UseMI->eraseFromParent();
  }

  // Extend BValNo by merging in IntA live segments of AValNo. Val# definition
  // is updated.
  BumpPtrAllocator &Allocator = LIS->getVNInfoAllocator();
  if (IntB.hasSubRanges()) {
    if (!IntA.hasSubRanges()) {
      unsigned Mask = MRI->getMaxLaneMaskForVReg(IntA.reg);
      IntA.createSubRangeFrom(Allocator, Mask, IntA);
    }
    SlotIndex AIdx = CopyIdx.getRegSlot(true);
    for (LiveInterval::SubRange &SA : IntA.subranges()) {
      VNInfo *ASubValNo = SA.getVNInfoAt(AIdx);
      assert(ASubValNo != nullptr);

      unsigned AMask = SA.LaneMask;
      for (LiveInterval::SubRange &SB : IntB.subranges()) {
        unsigned BMask = SB.LaneMask;
        unsigned Common = BMask & AMask;
        if (Common == 0)
          continue;

        DEBUG(
            dbgs() << format("\t\tCopy+Merge %04X into %04X\n", BMask, Common));
        unsigned BRest = BMask & ~AMask;
        LiveInterval::SubRange *CommonRange;
        if (BRest != 0) {
          SB.LaneMask = BRest;
          DEBUG(dbgs() << format("\t\tReduce Lane to %04X\n", BRest));
          // Duplicate SubRange for newly merged common stuff.
          CommonRange = IntB.createSubRangeFrom(Allocator, Common, SB);
        } else {
          // We van reuse the L SubRange.
          SB.LaneMask = Common;
          CommonRange = &SB;
        }
        LiveRange RangeCopy(SB, Allocator);

        VNInfo *BSubValNo = CommonRange->getVNInfoAt(CopyIdx);
        assert(BSubValNo->def == CopyIdx);
        BSubValNo->def = ASubValNo->def;
        addSegmentsWithValNo(*CommonRange, BSubValNo, SA, ASubValNo);
        AMask &= ~BMask;
      }
      if (AMask != 0) {
        DEBUG(dbgs() << format("\t\tNew Lane %04X\n", AMask));
        LiveRange *NewRange = IntB.createSubRange(Allocator, AMask);
        VNInfo *BSubValNo = NewRange->getNextValue(CopyIdx, Allocator);
        addSegmentsWithValNo(*NewRange, BSubValNo, SA, ASubValNo);
      }
    }
  }

  BValNo->def = AValNo->def;
  addSegmentsWithValNo(IntB, BValNo, IntA, AValNo);
  DEBUG(dbgs() << "\t\textended: " << IntB << '\n');

  LIS->removeVRegDefAt(IntA, AValNo->def);

  DEBUG(dbgs() << "\t\ttrimmed:  " << IntA << '\n');
  ++numCommutes;
  return true;
}

/// Returns true if @p MI defines the full vreg @p Reg, as opposed to just
/// defining a subregister.
static bool definesFullReg(const MachineInstr &MI, unsigned Reg) {
  assert(!TargetRegisterInfo::isPhysicalRegister(Reg) &&
         "This code cannot handle physreg aliasing");
  for (const MachineOperand &Op : MI.operands()) {
    if (!Op.isReg() || !Op.isDef() || Op.getReg() != Reg)
      continue;
    // Return true if we define the full register or don't care about the value
    // inside other subregisters.
    if (Op.getSubReg() == 0 || Op.isUndef())
      return true;
  }
  return false;
}

bool RegisterCoalescer::reMaterializeTrivialDef(const CoalescerPair &CP,
                                                MachineInstr *CopyMI,
                                                bool &IsDefCopy) {
  IsDefCopy = false;
  unsigned SrcReg = CP.isFlipped() ? CP.getDstReg() : CP.getSrcReg();
  unsigned SrcIdx = CP.isFlipped() ? CP.getDstIdx() : CP.getSrcIdx();
  unsigned DstReg = CP.isFlipped() ? CP.getSrcReg() : CP.getDstReg();
  unsigned DstIdx = CP.isFlipped() ? CP.getSrcIdx() : CP.getDstIdx();
  if (TargetRegisterInfo::isPhysicalRegister(SrcReg))
    return false;

  LiveInterval &SrcInt = LIS->getInterval(SrcReg);
  SlotIndex CopyIdx = LIS->getInstructionIndex(CopyMI);
  VNInfo *ValNo = SrcInt.Query(CopyIdx).valueIn();
  assert(ValNo && "CopyMI input register not live");
  if (ValNo->isPHIDef() || ValNo->isUnused())
    return false;
  MachineInstr *DefMI = LIS->getInstructionFromIndex(ValNo->def);
  if (!DefMI)
    return false;
  if (DefMI->isCopyLike()) {
    IsDefCopy = true;
    return false;
  }
  if (!TII->isAsCheapAsAMove(DefMI))
    return false;
  if (!TII->isTriviallyReMaterializable(DefMI, AA))
    return false;
  if (!definesFullReg(*DefMI, SrcReg))
    return false;
  bool SawStore = false;
  if (!DefMI->isSafeToMove(AA, SawStore))
    return false;
  const MCInstrDesc &MCID = DefMI->getDesc();
  if (MCID.getNumDefs() != 1)
    return false;
  // Only support subregister destinations when the def is read-undef.
  MachineOperand &DstOperand = CopyMI->getOperand(0);
  unsigned CopyDstReg = DstOperand.getReg();
  if (DstOperand.getSubReg() && !DstOperand.isUndef())
    return false;

  // If both SrcIdx and DstIdx are set, correct rematerialization would widen
  // the register substantially (beyond both source and dest size). This is bad
  // for performance since it can cascade through a function, introducing many
  // extra spills and fills (e.g. ARM can easily end up copying QQQQPR registers
  // around after a few subreg copies).
  if (SrcIdx && DstIdx)
    return false;

  const TargetRegisterClass *DefRC = TII->getRegClass(MCID, 0, TRI, *MF);
  if (!DefMI->isImplicitDef()) {
    if (TargetRegisterInfo::isPhysicalRegister(DstReg)) {
      unsigned NewDstReg = DstReg;

      unsigned NewDstIdx = TRI->composeSubRegIndices(CP.getSrcIdx(),
                                              DefMI->getOperand(0).getSubReg());
      if (NewDstIdx)
        NewDstReg = TRI->getSubReg(DstReg, NewDstIdx);

      // Finally, make sure that the physical subregister that will be
      // constructed later is permitted for the instruction.
      if (!DefRC->contains(NewDstReg))
        return false;
    } else {
      // Theoretically, some stack frame reference could exist. Just make sure
      // it hasn't actually happened.
      assert(TargetRegisterInfo::isVirtualRegister(DstReg) &&
             "Only expect to deal with virtual or physical registers");
    }
  }

  MachineBasicBlock *MBB = CopyMI->getParent();
  MachineBasicBlock::iterator MII =
    std::next(MachineBasicBlock::iterator(CopyMI));
  TII->reMaterialize(*MBB, MII, DstReg, SrcIdx, DefMI, *TRI);
  MachineInstr *NewMI = std::prev(MII);

  // In a situation like the following:
  //     %vreg0:subreg = instr              ; DefMI, subreg = DstIdx
  //     %vreg1        = copy %vreg0:subreg ; CopyMI, SrcIdx = 0
  // instead of widening %vreg1 to the register class of %vreg0 simply do:
  //     %vreg1 = instr
  const TargetRegisterClass *NewRC = CP.getNewRC();
  if (DstIdx != 0) {
    MachineOperand &DefMO = NewMI->getOperand(0);
    if (DefMO.getSubReg() == DstIdx) {
      assert(SrcIdx == 0 && CP.isFlipped()
             && "Shouldn't have SrcIdx+DstIdx at this point");
      const TargetRegisterClass *DstRC = MRI->getRegClass(DstReg);
      const TargetRegisterClass *CommonRC =
        TRI->getCommonSubClass(DefRC, DstRC);
      if (CommonRC != nullptr) {
        NewRC = CommonRC;
        DstIdx = 0;
        DefMO.setSubReg(0);
      }
    }
  }

  LIS->ReplaceMachineInstrInMaps(CopyMI, NewMI);
  CopyMI->eraseFromParent();
  ErasedInstrs.insert(CopyMI);

  // NewMI may have dead implicit defs (E.g. EFLAGS for MOV<bits>r0 on X86).
  // We need to remember these so we can add intervals once we insert
  // NewMI into SlotIndexes.
  SmallVector<unsigned, 4> NewMIImplDefs;
  for (unsigned i = NewMI->getDesc().getNumOperands(),
         e = NewMI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = NewMI->getOperand(i);
    if (MO.isReg() && MO.isDef()) {
      assert(MO.isImplicit() && MO.isDead() &&
             TargetRegisterInfo::isPhysicalRegister(MO.getReg()));
      NewMIImplDefs.push_back(MO.getReg());
    }
  }

  if (TargetRegisterInfo::isVirtualRegister(DstReg)) {
    unsigned NewIdx = NewMI->getOperand(0).getSubReg();

    if (DefRC != nullptr) {
      if (NewIdx)
        NewRC = TRI->getMatchingSuperRegClass(NewRC, DefRC, NewIdx);
      else
        NewRC = TRI->getCommonSubClass(NewRC, DefRC);
      assert(NewRC && "subreg chosen for remat incompatible with instruction");
    }
    MRI->setRegClass(DstReg, NewRC);

    updateRegDefsUses(DstReg, DstReg, DstIdx);
    NewMI->getOperand(0).setSubReg(NewIdx);
  } else if (NewMI->getOperand(0).getReg() != CopyDstReg) {
    // The New instruction may be defining a sub-register of what's actually
    // been asked for. If so it must implicitly define the whole thing.
    assert(TargetRegisterInfo::isPhysicalRegister(DstReg) &&
           "Only expect virtual or physical registers in remat");
    NewMI->getOperand(0).setIsDead(true);
    NewMI->addOperand(MachineOperand::CreateReg(CopyDstReg,
                                                true  /*IsDef*/,
                                                true  /*IsImp*/,
                                                false /*IsKill*/));
    // Record small dead def live-ranges for all the subregisters
    // of the destination register.
    // Otherwise, variables that live through may miss some
    // interferences, thus creating invalid allocation.
    // E.g., i386 code:
    // vreg1 = somedef ; vreg1 GR8
    // vreg2 = remat ; vreg2 GR32
    // CL = COPY vreg2.sub_8bit
    // = somedef vreg1 ; vreg1 GR8
    // =>
    // vreg1 = somedef ; vreg1 GR8
    // ECX<def, dead> = remat ; CL<imp-def>
    // = somedef vreg1 ; vreg1 GR8
    // vreg1 will see the inteferences with CL but not with CH since
    // no live-ranges would have been created for ECX.
    // Fix that!
    SlotIndex NewMIIdx = LIS->getInstructionIndex(NewMI);
    for (MCRegUnitIterator Units(NewMI->getOperand(0).getReg(), TRI);
         Units.isValid(); ++Units)
      if (LiveRange *LR = LIS->getCachedRegUnit(*Units))
        LR->createDeadDef(NewMIIdx.getRegSlot(), LIS->getVNInfoAllocator());
  }

  if (NewMI->getOperand(0).getSubReg())
    NewMI->getOperand(0).setIsUndef();

  // CopyMI may have implicit operands, transfer them over to the newly
  // rematerialized instruction. And update implicit def interval valnos.
  for (unsigned i = CopyMI->getDesc().getNumOperands(),
         e = CopyMI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = CopyMI->getOperand(i);
    if (MO.isReg()) {
      assert(MO.isImplicit() && "No explicit operands after implict operands.");
      // Discard VReg implicit defs.
      if (TargetRegisterInfo::isPhysicalRegister(MO.getReg())) {
        NewMI->addOperand(MO);
      }
    }
  }

  SlotIndex NewMIIdx = LIS->getInstructionIndex(NewMI);
  for (unsigned i = 0, e = NewMIImplDefs.size(); i != e; ++i) {
    unsigned Reg = NewMIImplDefs[i];
    for (MCRegUnitIterator Units(Reg, TRI); Units.isValid(); ++Units)
      if (LiveRange *LR = LIS->getCachedRegUnit(*Units))
        LR->createDeadDef(NewMIIdx.getRegSlot(), LIS->getVNInfoAllocator());
  }

  DEBUG(dbgs() << "Remat: " << *NewMI);
  ++NumReMats;

  // The source interval can become smaller because we removed a use.
  shrinkToUses(&SrcInt, &DeadDefs);
  if (!DeadDefs.empty()) {
    // If the virtual SrcReg is completely eliminated, update all DBG_VALUEs
    // to describe DstReg instead.
    for (MachineOperand &UseMO : MRI->use_operands(SrcReg)) {
      MachineInstr *UseMI = UseMO.getParent();
      if (UseMI->isDebugValue()) {
        UseMO.setReg(DstReg);
        DEBUG(dbgs() << "\t\tupdated: " << *UseMI);
      }
    }
    eliminateDeadDefs();
  }

  return true;
}

bool RegisterCoalescer::eliminateUndefCopy(MachineInstr *CopyMI) {
  // ProcessImpicitDefs may leave some copies of <undef> values, it only removes
  // local variables. When we have a copy like:
  //
  //   %vreg1 = COPY %vreg2<undef>
  //
  // We delete the copy and remove the corresponding value number from %vreg1.
  // Any uses of that value number are marked as <undef>.

  // Note that we do not query CoalescerPair here but redo isMoveInstr as the
  // CoalescerPair may have a new register class with adjusted subreg indices
  // at this point.
  unsigned SrcReg, DstReg, SrcSubIdx, DstSubIdx;
  isMoveInstr(*TRI, CopyMI, SrcReg, DstReg, SrcSubIdx, DstSubIdx);

  SlotIndex Idx = LIS->getInstructionIndex(CopyMI);
  const LiveInterval &SrcLI = LIS->getInterval(SrcReg);
  // CopyMI is undef iff SrcReg is not live before the instruction.
  if (SrcSubIdx != 0 && SrcLI.hasSubRanges()) {
    unsigned SrcMask = TRI->getSubRegIndexLaneMask(SrcSubIdx);
    for (const LiveInterval::SubRange &SR : SrcLI.subranges()) {
      if ((SR.LaneMask & SrcMask) == 0)
        continue;
      if (SR.liveAt(Idx))
        return false;
    }
  } else if (SrcLI.liveAt(Idx))
    return false;

  DEBUG(dbgs() << "\tEliminating copy of <undef> value\n");

  // Remove any DstReg segments starting at the instruction.
  LiveInterval &DstLI = LIS->getInterval(DstReg);
  SlotIndex RegIndex = Idx.getRegSlot();
  // Remove value or merge with previous one in case of a subregister def.
  if (VNInfo *PrevVNI = DstLI.getVNInfoAt(Idx)) {
    VNInfo *VNI = DstLI.getVNInfoAt(RegIndex);
    DstLI.MergeValueNumberInto(VNI, PrevVNI);

    // The affected subregister segments can be removed.
    unsigned DstMask = TRI->getSubRegIndexLaneMask(DstSubIdx);
    for (LiveInterval::SubRange &SR : DstLI.subranges()) {
      if ((SR.LaneMask & DstMask) == 0)
        continue;

      VNInfo *SVNI = SR.getVNInfoAt(RegIndex);
      assert(SVNI != nullptr && SlotIndex::isSameInstr(SVNI->def, RegIndex));
      SR.removeValNo(SVNI);
    }
    DstLI.removeEmptySubRanges();
  } else
    LIS->removeVRegDefAt(DstLI, RegIndex);

  // Mark uses as undef.
  for (MachineOperand &MO : MRI->reg_nodbg_operands(DstReg)) {
    if (MO.isDef() /*|| MO.isUndef()*/)
      continue;
    const MachineInstr &MI = *MO.getParent();
    SlotIndex UseIdx = LIS->getInstructionIndex(&MI);
    unsigned UseMask = TRI->getSubRegIndexLaneMask(MO.getSubReg());
    bool isLive;
    if (UseMask != ~0u && DstLI.hasSubRanges()) {
      isLive = false;
      for (const LiveInterval::SubRange &SR : DstLI.subranges()) {
        if ((SR.LaneMask & UseMask) == 0)
          continue;
        if (SR.liveAt(UseIdx)) {
          isLive = true;
          break;
        }
      }
    } else
      isLive = DstLI.liveAt(UseIdx);
    if (isLive)
      continue;
    MO.setIsUndef(true);
    DEBUG(dbgs() << "\tnew undef: " << UseIdx << '\t' << MI);
  }
  return true;
}

void RegisterCoalescer::updateRegDefsUses(unsigned SrcReg,
                                          unsigned DstReg,
                                          unsigned SubIdx) {
  bool DstIsPhys = TargetRegisterInfo::isPhysicalRegister(DstReg);
  LiveInterval *DstInt = DstIsPhys ? nullptr : &LIS->getInterval(DstReg);

  SmallPtrSet<MachineInstr*, 8> Visited;
  for (MachineRegisterInfo::reg_instr_iterator
       I = MRI->reg_instr_begin(SrcReg), E = MRI->reg_instr_end();
       I != E; ) {
    MachineInstr *UseMI = &*(I++);

    // Each instruction can only be rewritten once because sub-register
    // composition is not always idempotent. When SrcReg != DstReg, rewriting
    // the UseMI operands removes them from the SrcReg use-def chain, but when
    // SrcReg is DstReg we could encounter UseMI twice if it has multiple
    // operands mentioning the virtual register.
    if (SrcReg == DstReg && !Visited.insert(UseMI).second)
      continue;

    SmallVector<unsigned,8> Ops;
    bool Reads, Writes;
    std::tie(Reads, Writes) = UseMI->readsWritesVirtualRegister(SrcReg, &Ops);

    // If SrcReg wasn't read, it may still be the case that DstReg is live-in
    // because SrcReg is a sub-register.
    if (DstInt && !Reads && SubIdx)
      Reads = DstInt->liveAt(LIS->getInstructionIndex(UseMI));

    // Replace SrcReg with DstReg in all UseMI operands.
    for (unsigned i = 0, e = Ops.size(); i != e; ++i) {
      MachineOperand &MO = UseMI->getOperand(Ops[i]);

      // Adjust <undef> flags in case of sub-register joins. We don't want to
      // turn a full def into a read-modify-write sub-register def and vice
      // versa.
      if (SubIdx && MO.isDef())
        MO.setIsUndef(!Reads);

      // A subreg use of a partially undef (super) register may be a complete
      // undef use now and then has to be marked that way.
      if (SubIdx != 0 && MO.isUse() && MRI->shouldTrackSubRegLiveness(DstReg)) {
        if (!DstInt->hasSubRanges()) {
          BumpPtrAllocator &Allocator = LIS->getVNInfoAllocator();
          unsigned Mask = MRI->getMaxLaneMaskForVReg(DstInt->reg);
          DstInt->createSubRangeFrom(Allocator, Mask, *DstInt);
        }
        unsigned Mask = TRI->getSubRegIndexLaneMask(SubIdx);
        bool IsUndef = true;
        SlotIndex MIIdx = UseMI->isDebugValue()
          ? LIS->getSlotIndexes()->getIndexBefore(UseMI)
          : LIS->getInstructionIndex(UseMI);
        SlotIndex UseIdx = MIIdx.getRegSlot(true);
        for (LiveInterval::SubRange &S : DstInt->subranges()) {
          if ((S.LaneMask & Mask) == 0)
            continue;
          if (S.liveAt(UseIdx)) {
            IsUndef = false;
            break;
          }
        }
        if (IsUndef) {
          MO.setIsUndef(true);
          // We found out some subregister use is actually reading an undefined
          // value. In some cases the whole vreg has become undefined at this
          // point so we have to potentially shrink the main range if the
          // use was ending a live segment there.
          LiveQueryResult Q = DstInt->Query(MIIdx);
          if (Q.valueOut() == nullptr)
            ShrinkMainRange = true;
        }
      }

      if (DstIsPhys)
        MO.substPhysReg(DstReg, *TRI);
      else
        MO.substVirtReg(DstReg, SubIdx, *TRI);
    }

    DEBUG({
        dbgs() << "\t\tupdated: ";
        if (!UseMI->isDebugValue())
          dbgs() << LIS->getInstructionIndex(UseMI) << "\t";
        dbgs() << *UseMI;
      });
  }
}

bool RegisterCoalescer::canJoinPhys(const CoalescerPair &CP) {
  // Always join simple intervals that are defined by a single copy from a
  // reserved register. This doesn't increase register pressure, so it is
  // always beneficial.
  if (!MRI->isReserved(CP.getDstReg())) {
    DEBUG(dbgs() << "\tCan only merge into reserved registers.\n");
    return false;
  }

  LiveInterval &JoinVInt = LIS->getInterval(CP.getSrcReg());
  if (JoinVInt.containsOneValue())
    return true;

  DEBUG(dbgs() << "\tCannot join complex intervals into reserved register.\n");
  return false;
}

bool RegisterCoalescer::joinCopy(MachineInstr *CopyMI, bool &Again) {

  Again = false;
  DEBUG(dbgs() << LIS->getInstructionIndex(CopyMI) << '\t' << *CopyMI);

  CoalescerPair CP(*TRI);
  if (!CP.setRegisters(CopyMI)) {
    DEBUG(dbgs() << "\tNot coalescable.\n");
    return false;
  }

  if (CP.getNewRC()) {
    auto SrcRC = MRI->getRegClass(CP.getSrcReg());
    auto DstRC = MRI->getRegClass(CP.getDstReg());
    unsigned SrcIdx = CP.getSrcIdx();
    unsigned DstIdx = CP.getDstIdx();
    if (CP.isFlipped()) {
      std::swap(SrcIdx, DstIdx);
      std::swap(SrcRC, DstRC);
    }
    if (!TRI->shouldCoalesce(CopyMI, SrcRC, SrcIdx, DstRC, DstIdx,
                            CP.getNewRC())) {
      DEBUG(dbgs() << "\tSubtarget bailed on coalescing.\n");
      return false;
    }
  }

  // Dead code elimination. This really should be handled by MachineDCE, but
  // sometimes dead copies slip through, and we can't generate invalid live
  // ranges.
  if (!CP.isPhys() && CopyMI->allDefsAreDead()) {
    DEBUG(dbgs() << "\tCopy is dead.\n");
    DeadDefs.push_back(CopyMI);
    eliminateDeadDefs();
    return true;
  }

  // Eliminate undefs.
  if (!CP.isPhys() && eliminateUndefCopy(CopyMI)) {
    LIS->RemoveMachineInstrFromMaps(CopyMI);
    CopyMI->eraseFromParent();
    return false;  // Not coalescable.
  }

  // Coalesced copies are normally removed immediately, but transformations
  // like removeCopyByCommutingDef() can inadvertently create identity copies.
  // When that happens, just join the values and remove the copy.
  if (CP.getSrcReg() == CP.getDstReg()) {
    LiveInterval &LI = LIS->getInterval(CP.getSrcReg());
    DEBUG(dbgs() << "\tCopy already coalesced: " << LI << '\n');
    const SlotIndex CopyIdx = LIS->getInstructionIndex(CopyMI);
    LiveQueryResult LRQ = LI.Query(CopyIdx);
    if (VNInfo *DefVNI = LRQ.valueDefined()) {
      VNInfo *ReadVNI = LRQ.valueIn();
      assert(ReadVNI && "No value before copy and no <undef> flag.");
      assert(ReadVNI != DefVNI && "Cannot read and define the same value.");
      LI.MergeValueNumberInto(DefVNI, ReadVNI);

      // Process subregister liveranges.
      for (LiveInterval::SubRange &S : LI.subranges()) {
        LiveQueryResult SLRQ = S.Query(CopyIdx);
        if (VNInfo *SDefVNI = SLRQ.valueDefined()) {
          VNInfo *SReadVNI = SLRQ.valueIn();
          S.MergeValueNumberInto(SDefVNI, SReadVNI);
        }
      }
      DEBUG(dbgs() << "\tMerged values:          " << LI << '\n');
    }
    LIS->RemoveMachineInstrFromMaps(CopyMI);
    CopyMI->eraseFromParent();
    return true;
  }

  // Enforce policies.
  if (CP.isPhys()) {
    DEBUG(dbgs() << "\tConsidering merging " << PrintReg(CP.getSrcReg(), TRI)
                 << " with " << PrintReg(CP.getDstReg(), TRI, CP.getSrcIdx())
                 << '\n');
    if (!canJoinPhys(CP)) {
      // Before giving up coalescing, if definition of source is defined by
      // trivial computation, try rematerializing it.
      bool IsDefCopy;
      if (reMaterializeTrivialDef(CP, CopyMI, IsDefCopy))
        return true;
      if (IsDefCopy)
        Again = true;  // May be possible to coalesce later.
      return false;
    }
  } else {
    // When possible, let DstReg be the larger interval.
    if (!CP.isPartial() && LIS->getInterval(CP.getSrcReg()).size() >
                           LIS->getInterval(CP.getDstReg()).size())
      CP.flip();

    DEBUG({
      dbgs() << "\tConsidering merging to "
             << TRI->getRegClassName(CP.getNewRC()) << " with ";
      if (CP.getDstIdx() && CP.getSrcIdx())
        dbgs() << PrintReg(CP.getDstReg()) << " in "
               << TRI->getSubRegIndexName(CP.getDstIdx()) << " and "
               << PrintReg(CP.getSrcReg()) << " in "
               << TRI->getSubRegIndexName(CP.getSrcIdx()) << '\n';
      else
        dbgs() << PrintReg(CP.getSrcReg(), TRI) << " in "
               << PrintReg(CP.getDstReg(), TRI, CP.getSrcIdx()) << '\n';
    });
  }

  ShrinkMask = 0;
  ShrinkMainRange = false;

  // Okay, attempt to join these two intervals.  On failure, this returns false.
  // Otherwise, if one of the intervals being joined is a physreg, this method
  // always canonicalizes DstInt to be it.  The output "SrcInt" will not have
  // been modified, so we can use this information below to update aliases.
  if (!joinIntervals(CP)) {
    // Coalescing failed.

    // If definition of source is defined by trivial computation, try
    // rematerializing it.
    bool IsDefCopy;
    if (reMaterializeTrivialDef(CP, CopyMI, IsDefCopy))
      return true;

    // If we can eliminate the copy without merging the live segments, do so
    // now.
    if (!CP.isPartial() && !CP.isPhys()) {
      if (adjustCopiesBackFrom(CP, CopyMI) ||
          removeCopyByCommutingDef(CP, CopyMI)) {
        LIS->RemoveMachineInstrFromMaps(CopyMI);
        CopyMI->eraseFromParent();
        DEBUG(dbgs() << "\tTrivial!\n");
        return true;
      }
    }

    // Otherwise, we are unable to join the intervals.
    DEBUG(dbgs() << "\tInterference!\n");
    Again = true;  // May be possible to coalesce later.
    return false;
  }

  // Coalescing to a virtual register that is of a sub-register class of the
  // other. Make sure the resulting register is set to the right register class.
  if (CP.isCrossClass()) {
    ++numCrossRCs;
    MRI->setRegClass(CP.getDstReg(), CP.getNewRC());
  }

  // Removing sub-register copies can ease the register class constraints.
  // Make sure we attempt to inflate the register class of DstReg.
  if (!CP.isPhys() && RegClassInfo.isProperSubClass(CP.getNewRC()))
    InflateRegs.push_back(CP.getDstReg());

  // CopyMI has been erased by joinIntervals at this point. Remove it from
  // ErasedInstrs since copyCoalesceWorkList() won't add a successful join back
  // to the work list. This keeps ErasedInstrs from growing needlessly.
  ErasedInstrs.erase(CopyMI);

  // Rewrite all SrcReg operands to DstReg.
  // Also update DstReg operands to include DstIdx if it is set.
  if (CP.getDstIdx())
    updateRegDefsUses(CP.getDstReg(), CP.getDstReg(), CP.getDstIdx());
  updateRegDefsUses(CP.getSrcReg(), CP.getDstReg(), CP.getSrcIdx());

  // Shrink subregister ranges if necessary.
  if (ShrinkMask != 0) {
    LiveInterval &LI = LIS->getInterval(CP.getDstReg());
    for (LiveInterval::SubRange &S : LI.subranges()) {
      if ((S.LaneMask & ShrinkMask) == 0)
        continue;
      DEBUG(dbgs() << "Shrink LaneUses (Lane "
                   << format("%04X", S.LaneMask) << ")\n");
      LIS->shrinkToUses(S, LI.reg);
    }
    LI.removeEmptySubRanges();
  }
  if (ShrinkMainRange) {
    LiveInterval &LI = LIS->getInterval(CP.getDstReg());
    shrinkToUses(&LI);
  }

  // SrcReg is guaranteed to be the register whose live interval that is
  // being merged.
  LIS->removeInterval(CP.getSrcReg());

  // Update regalloc hint.
  TRI->updateRegAllocHint(CP.getSrcReg(), CP.getDstReg(), *MF);

  DEBUG({
    dbgs() << "\tSuccess: " << PrintReg(CP.getSrcReg(), TRI, CP.getSrcIdx())
           << " -> " << PrintReg(CP.getDstReg(), TRI, CP.getDstIdx()) << '\n';
    dbgs() << "\tResult = ";
    if (CP.isPhys())
      dbgs() << PrintReg(CP.getDstReg(), TRI);
    else
      dbgs() << LIS->getInterval(CP.getDstReg());
    dbgs() << '\n';
  });

  ++numJoins;
  return true;
}

bool RegisterCoalescer::joinReservedPhysReg(CoalescerPair &CP) {
  unsigned DstReg = CP.getDstReg();
  assert(CP.isPhys() && "Must be a physreg copy");
  assert(MRI->isReserved(DstReg) && "Not a reserved register");
  LiveInterval &RHS = LIS->getInterval(CP.getSrcReg());
  DEBUG(dbgs() << "\t\tRHS = " << RHS << '\n');

  assert(RHS.containsOneValue() && "Invalid join with reserved register");

  // Optimization for reserved registers like ESP. We can only merge with a
  // reserved physreg if RHS has a single value that is a copy of DstReg.
  // The live range of the reserved register will look like a set of dead defs
  // - we don't properly track the live range of reserved registers.

  // Deny any overlapping intervals.  This depends on all the reserved
  // register live ranges to look like dead defs.
  for (MCRegUnitIterator UI(DstReg, TRI); UI.isValid(); ++UI)
    if (RHS.overlaps(LIS->getRegUnit(*UI))) {
      DEBUG(dbgs() << "\t\tInterference: " << PrintRegUnit(*UI, TRI) << '\n');
      return false;
    }

  // Skip any value computations, we are not adding new values to the
  // reserved register.  Also skip merging the live ranges, the reserved
  // register live range doesn't need to be accurate as long as all the
  // defs are there.

  // Delete the identity copy.
  MachineInstr *CopyMI;
  if (CP.isFlipped()) {
    CopyMI = MRI->getVRegDef(RHS.reg);
  } else {
    if (!MRI->hasOneNonDBGUse(RHS.reg)) {
      DEBUG(dbgs() << "\t\tMultiple vreg uses!\n");
      return false;
    }

    MachineInstr *DestMI = MRI->getVRegDef(RHS.reg);
    CopyMI = &*MRI->use_instr_nodbg_begin(RHS.reg);
    const SlotIndex CopyRegIdx = LIS->getInstructionIndex(CopyMI).getRegSlot();
    const SlotIndex DestRegIdx = LIS->getInstructionIndex(DestMI).getRegSlot();

    // We checked above that there are no interfering defs of the physical
    // register. However, for this case, where we intent to move up the def of
    // the physical register, we also need to check for interfering uses.
    SlotIndexes *Indexes = LIS->getSlotIndexes();
    for (SlotIndex SI = Indexes->getNextNonNullIndex(DestRegIdx);
         SI != CopyRegIdx; SI = Indexes->getNextNonNullIndex(SI)) {
      MachineInstr *MI = LIS->getInstructionFromIndex(SI);
      if (MI->readsRegister(DstReg, TRI)) {
        DEBUG(dbgs() << "\t\tInterference (read): " << *MI);
        return false;
      }

      // We must also check for clobbers caused by regmasks.
      for (const auto &MO : MI->operands()) {
        if (MO.isRegMask() && MO.clobbersPhysReg(DstReg)) {
          DEBUG(dbgs() << "\t\tInterference (regmask clobber): " << *MI);
          return false;
        }
      }
    }

    // We're going to remove the copy which defines a physical reserved
    // register, so remove its valno, etc.
    DEBUG(dbgs() << "\t\tRemoving phys reg def of " << DstReg << " at "
          << CopyRegIdx << "\n");

    LIS->removePhysRegDefAt(DstReg, CopyRegIdx);
    // Create a new dead def at the new def location.
    for (MCRegUnitIterator UI(DstReg, TRI); UI.isValid(); ++UI) {
      LiveRange &LR = LIS->getRegUnit(*UI);
      LR.createDeadDef(DestRegIdx, LIS->getVNInfoAllocator());
    }
  }

  LIS->RemoveMachineInstrFromMaps(CopyMI);
  CopyMI->eraseFromParent();

  // We don't track kills for reserved registers.
  MRI->clearKillFlags(CP.getSrcReg());

  return true;
}

//===----------------------------------------------------------------------===//
//                 Interference checking and interval joining
//===----------------------------------------------------------------------===//
//
// In the easiest case, the two live ranges being joined are disjoint, and
// there is no interference to consider. It is quite common, though, to have
// overlapping live ranges, and we need to check if the interference can be
// resolved.
//
// The live range of a single SSA value forms a sub-tree of the dominator tree.
// This means that two SSA values overlap if and only if the def of one value
// is contained in the live range of the other value. As a special case, the
// overlapping values can be defined at the same index.
//
// The interference from an overlapping def can be resolved in these cases:
//
// 1. Coalescable copies. The value is defined by a copy that would become an
//    identity copy after joining SrcReg and DstReg. The copy instruction will
//    be removed, and the value will be merged with the source value.
//
//    There can be several copies back and forth, causing many values to be
//    merged into one. We compute a list of ultimate values in the joined live
//    range as well as a mappings from the old value numbers.
//
// 2. IMPLICIT_DEF. This instruction is only inserted to ensure all PHI
//    predecessors have a live out value. It doesn't cause real interference,
//    and can be merged into the value it overlaps. Like a coalescable copy, it
//    can be erased after joining.
//
// 3. Copy of external value. The overlapping def may be a copy of a value that
//    is already in the other register. This is like a coalescable copy, but
//    the live range of the source register must be trimmed after erasing the
//    copy instruction:
//
//      %src = COPY %ext
//      %dst = COPY %ext  <-- Remove this COPY, trim the live range of %ext.
//
// 4. Clobbering undefined lanes. Vector registers are sometimes built by
//    defining one lane at a time:
//
//      %dst:ssub0<def,read-undef> = FOO
//      %src = BAR
//      %dst:ssub1<def> = COPY %src
//
//    The live range of %src overlaps the %dst value defined by FOO, but
//    merging %src into %dst:ssub1 is only going to clobber the ssub1 lane
//    which was undef anyway.
//
//    The value mapping is more complicated in this case. The final live range
//    will have different value numbers for both FOO and BAR, but there is no
//    simple mapping from old to new values. It may even be necessary to add
//    new PHI values.
//
// 5. Clobbering dead lanes. A def may clobber a lane of a vector register that
//    is live, but never read. This can happen because we don't compute
//    individual live ranges per lane.
//
//      %dst<def> = FOO
//      %src = BAR
//      %dst:ssub1<def> = COPY %src
//
//    This kind of interference is only resolved locally. If the clobbered
//    lane value escapes the block, the join is aborted.

namespace {
/// Track information about values in a single virtual register about to be
/// joined. Objects of this class are always created in pairs - one for each
/// side of the CoalescerPair (or one for each lane of a side of the coalescer
/// pair)
class JoinVals {
  /// Live range we work on.
  LiveRange &LR;
  /// (Main) register we work on.
  const unsigned Reg;

  /// Reg (and therefore the values in this liverange) will end up as
  /// subregister SubIdx in the coalesced register. Either CP.DstIdx or
  /// CP.SrcIdx.
  const unsigned SubIdx;
  /// The LaneMask that this liverange will occupy the coalesced register. May
  /// be smaller than the lanemask produced by SubIdx when merging subranges.
  const unsigned LaneMask;

  /// This is true when joining sub register ranges, false when joining main
  /// ranges.
  const bool SubRangeJoin;
  /// Whether the current LiveInterval tracks subregister liveness.
  const bool TrackSubRegLiveness;

  /// Values that will be present in the final live range.
  SmallVectorImpl<VNInfo*> &NewVNInfo;

  const CoalescerPair &CP;
  LiveIntervals *LIS;
  SlotIndexes *Indexes;
  const TargetRegisterInfo *TRI;

  /// Value number assignments. Maps value numbers in LI to entries in
  /// NewVNInfo. This is suitable for passing to LiveInterval::join().
  SmallVector<int, 8> Assignments;

  /// Conflict resolution for overlapping values.
  enum ConflictResolution {
    /// No overlap, simply keep this value.
    CR_Keep,

    /// Merge this value into OtherVNI and erase the defining instruction.
    /// Used for IMPLICIT_DEF, coalescable copies, and copies from external
    /// values.
    CR_Erase,

    /// Merge this value into OtherVNI but keep the defining instruction.
    /// This is for the special case where OtherVNI is defined by the same
    /// instruction.
    CR_Merge,

    /// Keep this value, and have it replace OtherVNI where possible. This
    /// complicates value mapping since OtherVNI maps to two different values
    /// before and after this def.
    /// Used when clobbering undefined or dead lanes.
    CR_Replace,

    /// Unresolved conflict. Visit later when all values have been mapped.
    CR_Unresolved,

    /// Unresolvable conflict. Abort the join.
    CR_Impossible
  };

  /// Per-value info for LI. The lane bit masks are all relative to the final
  /// joined register, so they can be compared directly between SrcReg and
  /// DstReg.
  struct Val {
    ConflictResolution Resolution;

    /// Lanes written by this def, 0 for unanalyzed values.
    unsigned WriteLanes;

    /// Lanes with defined values in this register. Other lanes are undef and
    /// safe to clobber.
    unsigned ValidLanes;

    /// Value in LI being redefined by this def.
    VNInfo *RedefVNI;

    /// Value in the other live range that overlaps this def, if any.
    VNInfo *OtherVNI;

    /// Is this value an IMPLICIT_DEF that can be erased?
    ///
    /// IMPLICIT_DEF values should only exist at the end of a basic block that
    /// is a predecessor to a phi-value. These IMPLICIT_DEF instructions can be
    /// safely erased if they are overlapping a live value in the other live
    /// interval.
    ///
    /// Weird control flow graphs and incomplete PHI handling in
    /// ProcessImplicitDefs can very rarely create IMPLICIT_DEF values with
    /// longer live ranges. Such IMPLICIT_DEF values should be treated like
    /// normal values.
    bool ErasableImplicitDef;

    /// True when the live range of this value will be pruned because of an
    /// overlapping CR_Replace value in the other live range.
    bool Pruned;

    /// True once Pruned above has been computed.
    bool PrunedComputed;

    Val() : Resolution(CR_Keep), WriteLanes(0), ValidLanes(0),
            RedefVNI(nullptr), OtherVNI(nullptr), ErasableImplicitDef(false),
            Pruned(false), PrunedComputed(false) {}

    bool isAnalyzed() const { return WriteLanes != 0; }
  };

  /// One entry per value number in LI.
  SmallVector<Val, 8> Vals;

  /// Compute the bitmask of lanes actually written by DefMI.
  /// Set Redef if there are any partial register definitions that depend on the
  /// previous value of the register.
  unsigned computeWriteLanes(const MachineInstr *DefMI, bool &Redef) const;

  /// Find the ultimate value that VNI was copied from.
  std::pair<const VNInfo*,unsigned> followCopyChain(const VNInfo *VNI) const;

  bool valuesIdentical(VNInfo *Val0, VNInfo *Val1, const JoinVals &Other) const;

  /// Analyze ValNo in this live range, and set all fields of Vals[ValNo].
  /// Return a conflict resolution when possible, but leave the hard cases as
  /// CR_Unresolved.
  /// Recursively calls computeAssignment() on this and Other, guaranteeing that
  /// both OtherVNI and RedefVNI have been analyzed and mapped before returning.
  /// The recursion always goes upwards in the dominator tree, making loops
  /// impossible.
  ConflictResolution analyzeValue(unsigned ValNo, JoinVals &Other);

  /// Compute the value assignment for ValNo in RI.
  /// This may be called recursively by analyzeValue(), but never for a ValNo on
  /// the stack.
  void computeAssignment(unsigned ValNo, JoinVals &Other);

  /// Assuming ValNo is going to clobber some valid lanes in Other.LR, compute
  /// the extent of the tainted lanes in the block.
  ///
  /// Multiple values in Other.LR can be affected since partial redefinitions
  /// can preserve previously tainted lanes.
  ///
  ///   1 %dst = VLOAD           <-- Define all lanes in %dst
  ///   2 %src = FOO             <-- ValNo to be joined with %dst:ssub0
  ///   3 %dst:ssub1 = BAR       <-- Partial redef doesn't clear taint in ssub0
  ///   4 %dst:ssub0 = COPY %src <-- Conflict resolved, ssub0 wasn't read
  ///
  /// For each ValNo in Other that is affected, add an (EndIndex, TaintedLanes)
  /// entry to TaintedVals.
  ///
  /// Returns false if the tainted lanes extend beyond the basic block.
  bool taintExtent(unsigned, unsigned, JoinVals&,
                   SmallVectorImpl<std::pair<SlotIndex, unsigned> >&);

  /// Return true if MI uses any of the given Lanes from Reg.
  /// This does not include partial redefinitions of Reg.
  bool usesLanes(const MachineInstr *MI, unsigned, unsigned, unsigned) const;

  /// Determine if ValNo is a copy of a value number in LR or Other.LR that will
  /// be pruned:
  ///
  ///   %dst = COPY %src
  ///   %src = COPY %dst  <-- This value to be pruned.
  ///   %dst = COPY %src  <-- This value is a copy of a pruned value.
  bool isPrunedValue(unsigned ValNo, JoinVals &Other);

public:
  JoinVals(LiveRange &LR, unsigned Reg, unsigned SubIdx, unsigned LaneMask,
           SmallVectorImpl<VNInfo*> &newVNInfo, const CoalescerPair &cp,
           LiveIntervals *lis, const TargetRegisterInfo *TRI, bool SubRangeJoin,
           bool TrackSubRegLiveness)
    : LR(LR), Reg(Reg), SubIdx(SubIdx), LaneMask(LaneMask),
      SubRangeJoin(SubRangeJoin), TrackSubRegLiveness(TrackSubRegLiveness),
      NewVNInfo(newVNInfo), CP(cp), LIS(lis), Indexes(LIS->getSlotIndexes()),
      TRI(TRI), Assignments(LR.getNumValNums(), -1), Vals(LR.getNumValNums())
  {}

  /// Analyze defs in LR and compute a value mapping in NewVNInfo.
  /// Returns false if any conflicts were impossible to resolve.
  bool mapValues(JoinVals &Other);

  /// Try to resolve conflicts that require all values to be mapped.
  /// Returns false if any conflicts were impossible to resolve.
  bool resolveConflicts(JoinVals &Other);

  /// Prune the live range of values in Other.LR where they would conflict with
  /// CR_Replace values in LR. Collect end points for restoring the live range
  /// after joining.
  void pruneValues(JoinVals &Other, SmallVectorImpl<SlotIndex> &EndPoints,
                   bool changeInstrs);

  /// Removes subranges starting at copies that get removed. This sometimes
  /// happens when undefined subranges are copied around. These ranges contain
  /// no usefull information and can be removed.
  void pruneSubRegValues(LiveInterval &LI, unsigned &ShrinkMask);

  /// Erase any machine instructions that have been coalesced away.
  /// Add erased instructions to ErasedInstrs.
  /// Add foreign virtual registers to ShrinkRegs if their live range ended at
  /// the erased instrs.
  void eraseInstrs(SmallPtrSetImpl<MachineInstr*> &ErasedInstrs,
                   SmallVectorImpl<unsigned> &ShrinkRegs);

  /// Remove liverange defs at places where implicit defs will be removed.
  void removeImplicitDefs();

  /// Get the value assignments suitable for passing to LiveInterval::join.
  const int *getAssignments() const { return Assignments.data(); }
};
} // end anonymous namespace

unsigned JoinVals::computeWriteLanes(const MachineInstr *DefMI, bool &Redef)
  const {
  unsigned L = 0;
  for (const MachineOperand &MO : DefMI->operands()) {
    if (!MO.isReg() || MO.getReg() != Reg || !MO.isDef())
      continue;
    L |= TRI->getSubRegIndexLaneMask(
           TRI->composeSubRegIndices(SubIdx, MO.getSubReg()));
    if (MO.readsReg())
      Redef = true;
  }
  return L;
}

std::pair<const VNInfo*, unsigned> JoinVals::followCopyChain(
    const VNInfo *VNI) const {
  unsigned Reg = this->Reg;

  while (!VNI->isPHIDef()) {
    SlotIndex Def = VNI->def;
    MachineInstr *MI = Indexes->getInstructionFromIndex(Def);
    assert(MI && "No defining instruction");
    if (!MI->isFullCopy())
      return std::make_pair(VNI, Reg);
    unsigned SrcReg = MI->getOperand(1).getReg();
    if (!TargetRegisterInfo::isVirtualRegister(SrcReg))
      return std::make_pair(VNI, Reg);

    const LiveInterval &LI = LIS->getInterval(SrcReg);
    const VNInfo *ValueIn;
    // No subrange involved.
    if (!SubRangeJoin || !LI.hasSubRanges()) {
      LiveQueryResult LRQ = LI.Query(Def);
      ValueIn = LRQ.valueIn();
    } else {
      // Query subranges. Pick the first matching one.
      ValueIn = nullptr;
      for (const LiveInterval::SubRange &S : LI.subranges()) {
        // Transform lanemask to a mask in the joined live interval.
        unsigned SMask = TRI->composeSubRegIndexLaneMask(SubIdx, S.LaneMask);
        if ((SMask & LaneMask) == 0)
          continue;
        LiveQueryResult LRQ = S.Query(Def);
        ValueIn = LRQ.valueIn();
        break;
      }
    }
    if (ValueIn == nullptr)
      break;
    VNI = ValueIn;
    Reg = SrcReg;
  }
  return std::make_pair(VNI, Reg);
}

bool JoinVals::valuesIdentical(VNInfo *Value0, VNInfo *Value1,
                               const JoinVals &Other) const {
  const VNInfo *Orig0;
  unsigned Reg0;
  std::tie(Orig0, Reg0) = followCopyChain(Value0);
  if (Orig0 == Value1)
    return true;

  const VNInfo *Orig1;
  unsigned Reg1;
  std::tie(Orig1, Reg1) = Other.followCopyChain(Value1);

  // The values are equal if they are defined at the same place and use the
  // same register. Note that we cannot compare VNInfos directly as some of
  // them might be from a copy created in mergeSubRangeInto()  while the other
  // is from the original LiveInterval.
  return Orig0->def == Orig1->def && Reg0 == Reg1;
}

JoinVals::ConflictResolution
JoinVals::analyzeValue(unsigned ValNo, JoinVals &Other) {
  Val &V = Vals[ValNo];
  assert(!V.isAnalyzed() && "Value has already been analyzed!");
  VNInfo *VNI = LR.getValNumInfo(ValNo);
  if (VNI->isUnused()) {
    V.WriteLanes = ~0u;
    return CR_Keep;
  }

  // Get the instruction defining this value, compute the lanes written.
  const MachineInstr *DefMI = nullptr;
  if (VNI->isPHIDef()) {
    // Conservatively assume that all lanes in a PHI are valid.
    unsigned Lanes = SubRangeJoin ? 1 : TRI->getSubRegIndexLaneMask(SubIdx);
    V.ValidLanes = V.WriteLanes = Lanes;
  } else {
    DefMI = Indexes->getInstructionFromIndex(VNI->def);
    assert(DefMI != nullptr);
    if (SubRangeJoin) {
      // We don't care about the lanes when joining subregister ranges.
      V.WriteLanes = V.ValidLanes = 1;
      if (DefMI->isImplicitDef()) {
        V.ValidLanes = 0;
        V.ErasableImplicitDef = true;
      }
    } else {
      bool Redef = false;
      V.ValidLanes = V.WriteLanes = computeWriteLanes(DefMI, Redef);

      // If this is a read-modify-write instruction, there may be more valid
      // lanes than the ones written by this instruction.
      // This only covers partial redef operands. DefMI may have normal use
      // operands reading the register. They don't contribute valid lanes.
      //
      // This adds ssub1 to the set of valid lanes in %src:
      //
      //   %src:ssub1<def> = FOO
      //
      // This leaves only ssub1 valid, making any other lanes undef:
      //
      //   %src:ssub1<def,read-undef> = FOO %src:ssub2
      //
      // The <read-undef> flag on the def operand means that old lane values are
      // not important.
      if (Redef) {
        V.RedefVNI = LR.Query(VNI->def).valueIn();
        assert((TrackSubRegLiveness || V.RedefVNI) &&
               "Instruction is reading nonexistent value");
        if (V.RedefVNI != nullptr) {
          computeAssignment(V.RedefVNI->id, Other);
          V.ValidLanes |= Vals[V.RedefVNI->id].ValidLanes;
        }
      }

      // An IMPLICIT_DEF writes undef values.
      if (DefMI->isImplicitDef()) {
        // We normally expect IMPLICIT_DEF values to be live only until the end
        // of their block. If the value is really live longer and gets pruned in
        // another block, this flag is cleared again.
        V.ErasableImplicitDef = true;
        V.ValidLanes &= ~V.WriteLanes;
      }
    }
  }

  // Find the value in Other that overlaps VNI->def, if any.
  LiveQueryResult OtherLRQ = Other.LR.Query(VNI->def);

  // It is possible that both values are defined by the same instruction, or
  // the values are PHIs defined in the same block. When that happens, the two
  // values should be merged into one, but not into any preceding value.
  // The first value defined or visited gets CR_Keep, the other gets CR_Merge.
  if (VNInfo *OtherVNI = OtherLRQ.valueDefined()) {
    assert(SlotIndex::isSameInstr(VNI->def, OtherVNI->def) && "Broken LRQ");

    // One value stays, the other is merged. Keep the earlier one, or the first
    // one we see.
    if (OtherVNI->def < VNI->def)
      Other.computeAssignment(OtherVNI->id, *this);
    else if (VNI->def < OtherVNI->def && OtherLRQ.valueIn()) {
      // This is an early-clobber def overlapping a live-in value in the other
      // register. Not mergeable.
      V.OtherVNI = OtherLRQ.valueIn();
      return CR_Impossible;
    }
    V.OtherVNI = OtherVNI;
    Val &OtherV = Other.Vals[OtherVNI->id];
    // Keep this value, check for conflicts when analyzing OtherVNI.
    if (!OtherV.isAnalyzed())
      return CR_Keep;
    // Both sides have been analyzed now.
    // Allow overlapping PHI values. Any real interference would show up in a
    // predecessor, the PHI itself can't introduce any conflicts.
    if (VNI->isPHIDef())
      return CR_Merge;
    if (V.ValidLanes & OtherV.ValidLanes)
      // Overlapping lanes can't be resolved.
      return CR_Impossible;
    else
      return CR_Merge;
  }

  // No simultaneous def. Is Other live at the def?
  V.OtherVNI = OtherLRQ.valueIn();
  if (!V.OtherVNI)
    // No overlap, no conflict.
    return CR_Keep;

  assert(!SlotIndex::isSameInstr(VNI->def, V.OtherVNI->def) && "Broken LRQ");

  // We have overlapping values, or possibly a kill of Other.
  // Recursively compute assignments up the dominator tree.
  Other.computeAssignment(V.OtherVNI->id, *this);
  Val &OtherV = Other.Vals[V.OtherVNI->id];

  // Check if OtherV is an IMPLICIT_DEF that extends beyond its basic block.
  // This shouldn't normally happen, but ProcessImplicitDefs can leave such
  // IMPLICIT_DEF instructions behind, and there is nothing wrong with it
  // technically.
  //
  // WHen it happens, treat that IMPLICIT_DEF as a normal value, and don't try
  // to erase the IMPLICIT_DEF instruction.
  if (OtherV.ErasableImplicitDef && DefMI &&
      DefMI->getParent() != Indexes->getMBBFromIndex(V.OtherVNI->def)) {
    DEBUG(dbgs() << "IMPLICIT_DEF defined at " << V.OtherVNI->def
                 << " extends into BB#" << DefMI->getParent()->getNumber()
                 << ", keeping it.\n");
    OtherV.ErasableImplicitDef = false;
  }

  // Allow overlapping PHI values. Any real interference would show up in a
  // predecessor, the PHI itself can't introduce any conflicts.
  if (VNI->isPHIDef())
    return CR_Replace;

  // Check for simple erasable conflicts.
  if (DefMI->isImplicitDef()) {
    // We need the def for the subregister if there is nothing else live at the
    // subrange at this point.
    if (TrackSubRegLiveness
        && (V.WriteLanes & (OtherV.ValidLanes | OtherV.WriteLanes)) == 0)
      return CR_Replace;
    return CR_Erase;
  }

  // Include the non-conflict where DefMI is a coalescable copy that kills
  // OtherVNI. We still want the copy erased and value numbers merged.
  if (CP.isCoalescable(DefMI)) {
    // Some of the lanes copied from OtherVNI may be undef, making them undef
    // here too.
    V.ValidLanes &= ~V.WriteLanes | OtherV.ValidLanes;
    return CR_Erase;
  }

  // This may not be a real conflict if DefMI simply kills Other and defines
  // VNI.
  if (OtherLRQ.isKill() && OtherLRQ.endPoint() <= VNI->def)
    return CR_Keep;

  // Handle the case where VNI and OtherVNI can be proven to be identical:
  //
  //   %other = COPY %ext
  //   %this  = COPY %ext <-- Erase this copy
  //
  if (DefMI->isFullCopy() && !CP.isPartial()
      && valuesIdentical(VNI, V.OtherVNI, Other))
    return CR_Erase;

  // If the lanes written by this instruction were all undef in OtherVNI, it is
  // still safe to join the live ranges. This can't be done with a simple value
  // mapping, though - OtherVNI will map to multiple values:
  //
  //   1 %dst:ssub0 = FOO                <-- OtherVNI
  //   2 %src = BAR                      <-- VNI
  //   3 %dst:ssub1 = COPY %src<kill>    <-- Eliminate this copy.
  //   4 BAZ %dst<kill>
  //   5 QUUX %src<kill>
  //
  // Here OtherVNI will map to itself in [1;2), but to VNI in [2;5). CR_Replace
  // handles this complex value mapping.
  if ((V.WriteLanes & OtherV.ValidLanes) == 0)
    return CR_Replace;

  // If the other live range is killed by DefMI and the live ranges are still
  // overlapping, it must be because we're looking at an early clobber def:
  //
  //   %dst<def,early-clobber> = ASM %src<kill>
  //
  // In this case, it is illegal to merge the two live ranges since the early
  // clobber def would clobber %src before it was read.
  if (OtherLRQ.isKill()) {
    // This case where the def doesn't overlap the kill is handled above.
    assert(VNI->def.isEarlyClobber() &&
           "Only early clobber defs can overlap a kill");
    return CR_Impossible;
  }

  // VNI is clobbering live lanes in OtherVNI, but there is still the
  // possibility that no instructions actually read the clobbered lanes.
  // If we're clobbering all the lanes in OtherVNI, at least one must be read.
  // Otherwise Other.RI wouldn't be live here.
  if ((TRI->getSubRegIndexLaneMask(Other.SubIdx) & ~V.WriteLanes) == 0)
    return CR_Impossible;

  // We need to verify that no instructions are reading the clobbered lanes. To
  // save compile time, we'll only check that locally. Don't allow the tainted
  // value to escape the basic block.
  MachineBasicBlock *MBB = Indexes->getMBBFromIndex(VNI->def);
  if (OtherLRQ.endPoint() >= Indexes->getMBBEndIdx(MBB))
    return CR_Impossible;

  // There are still some things that could go wrong besides clobbered lanes
  // being read, for example OtherVNI may be only partially redefined in MBB,
  // and some clobbered lanes could escape the block. Save this analysis for
  // resolveConflicts() when all values have been mapped. We need to know
  // RedefVNI and WriteLanes for any later defs in MBB, and we can't compute
  // that now - the recursive analyzeValue() calls must go upwards in the
  // dominator tree.
  return CR_Unresolved;
}

void JoinVals::computeAssignment(unsigned ValNo, JoinVals &Other) {
  Val &V = Vals[ValNo];
  if (V.isAnalyzed()) {
    // Recursion should always move up the dominator tree, so ValNo is not
    // supposed to reappear before it has been assigned.
    assert(Assignments[ValNo] != -1 && "Bad recursion?");
    return;
  }
  switch ((V.Resolution = analyzeValue(ValNo, Other))) {
  case CR_Erase:
  case CR_Merge:
    // Merge this ValNo into OtherVNI.
    assert(V.OtherVNI && "OtherVNI not assigned, can't merge.");
    assert(Other.Vals[V.OtherVNI->id].isAnalyzed() && "Missing recursion");
    Assignments[ValNo] = Other.Assignments[V.OtherVNI->id];
    DEBUG(dbgs() << "\t\tmerge " << PrintReg(Reg) << ':' << ValNo << '@'
                 << LR.getValNumInfo(ValNo)->def << " into "
                 << PrintReg(Other.Reg) << ':' << V.OtherVNI->id << '@'
                 << V.OtherVNI->def << " --> @"
                 << NewVNInfo[Assignments[ValNo]]->def << '\n');
    break;
  case CR_Replace:
  case CR_Unresolved: {
    // The other value is going to be pruned if this join is successful.
    assert(V.OtherVNI && "OtherVNI not assigned, can't prune");
    Val &OtherV = Other.Vals[V.OtherVNI->id];
    // We cannot erase an IMPLICIT_DEF if we don't have valid values for all
    // its lanes.
    if ((OtherV.WriteLanes & ~V.ValidLanes) != 0 && TrackSubRegLiveness)
      OtherV.ErasableImplicitDef = false;
    OtherV.Pruned = true;
  }
    // Fall through.
  default:
    // This value number needs to go in the final joined live range.
    Assignments[ValNo] = NewVNInfo.size();
    NewVNInfo.push_back(LR.getValNumInfo(ValNo));
    break;
  }
}

bool JoinVals::mapValues(JoinVals &Other) {
  for (unsigned i = 0, e = LR.getNumValNums(); i != e; ++i) {
    computeAssignment(i, Other);
    if (Vals[i].Resolution == CR_Impossible) {
      DEBUG(dbgs() << "\t\tinterference at " << PrintReg(Reg) << ':' << i
                   << '@' << LR.getValNumInfo(i)->def << '\n');
      return false;
    }
  }
  return true;
}

bool JoinVals::
taintExtent(unsigned ValNo, unsigned TaintedLanes, JoinVals &Other,
            SmallVectorImpl<std::pair<SlotIndex, unsigned> > &TaintExtent) {
  VNInfo *VNI = LR.getValNumInfo(ValNo);
  MachineBasicBlock *MBB = Indexes->getMBBFromIndex(VNI->def);
  SlotIndex MBBEnd = Indexes->getMBBEndIdx(MBB);

  // Scan Other.LR from VNI.def to MBBEnd.
  LiveInterval::iterator OtherI = Other.LR.find(VNI->def);
  assert(OtherI != Other.LR.end() && "No conflict?");
  do {
    // OtherI is pointing to a tainted value. Abort the join if the tainted
    // lanes escape the block.
    SlotIndex End = OtherI->end;
    if (End >= MBBEnd) {
      DEBUG(dbgs() << "\t\ttaints global " << PrintReg(Other.Reg) << ':'
                   << OtherI->valno->id << '@' << OtherI->start << '\n');
      return false;
    }
    DEBUG(dbgs() << "\t\ttaints local " << PrintReg(Other.Reg) << ':'
                 << OtherI->valno->id << '@' << OtherI->start
                 << " to " << End << '\n');
    // A dead def is not a problem.
    if (End.isDead())
      break;
    TaintExtent.push_back(std::make_pair(End, TaintedLanes));

    // Check for another def in the MBB.
    if (++OtherI == Other.LR.end() || OtherI->start >= MBBEnd)
      break;

    // Lanes written by the new def are no longer tainted.
    const Val &OV = Other.Vals[OtherI->valno->id];
    TaintedLanes &= ~OV.WriteLanes;
    if (!OV.RedefVNI)
      break;
  } while (TaintedLanes);
  return true;
}

bool JoinVals::usesLanes(const MachineInstr *MI, unsigned Reg, unsigned SubIdx,
                         unsigned Lanes) const {
  if (MI->isDebugValue())
    return false;
  for (const MachineOperand &MO : MI->operands()) {
    if (!MO.isReg() || MO.isDef() || MO.getReg() != Reg)
      continue;
    if (!MO.readsReg())
      continue;
    if (Lanes & TRI->getSubRegIndexLaneMask(
                  TRI->composeSubRegIndices(SubIdx, MO.getSubReg())))
      return true;
  }
  return false;
}

bool JoinVals::resolveConflicts(JoinVals &Other) {
  for (unsigned i = 0, e = LR.getNumValNums(); i != e; ++i) {
    Val &V = Vals[i];
    assert (V.Resolution != CR_Impossible && "Unresolvable conflict");
    if (V.Resolution != CR_Unresolved)
      continue;
    DEBUG(dbgs() << "\t\tconflict at " << PrintReg(Reg) << ':' << i
                 << '@' << LR.getValNumInfo(i)->def << '\n');
    if (SubRangeJoin)
      return false;

    ++NumLaneConflicts;
    assert(V.OtherVNI && "Inconsistent conflict resolution.");
    VNInfo *VNI = LR.getValNumInfo(i);
    const Val &OtherV = Other.Vals[V.OtherVNI->id];

    // VNI is known to clobber some lanes in OtherVNI. If we go ahead with the
    // join, those lanes will be tainted with a wrong value. Get the extent of
    // the tainted lanes.
    unsigned TaintedLanes = V.WriteLanes & OtherV.ValidLanes;
    SmallVector<std::pair<SlotIndex, unsigned>, 8> TaintExtent;
    if (!taintExtent(i, TaintedLanes, Other, TaintExtent))
      // Tainted lanes would extend beyond the basic block.
      return false;

    assert(!TaintExtent.empty() && "There should be at least one conflict.");

    // Now look at the instructions from VNI->def to TaintExtent (inclusive).
    MachineBasicBlock *MBB = Indexes->getMBBFromIndex(VNI->def);
    MachineBasicBlock::iterator MI = MBB->begin();
    if (!VNI->isPHIDef()) {
      MI = Indexes->getInstructionFromIndex(VNI->def);
      // No need to check the instruction defining VNI for reads.
      ++MI;
    }
    assert(!SlotIndex::isSameInstr(VNI->def, TaintExtent.front().first) &&
           "Interference ends on VNI->def. Should have been handled earlier");
    MachineInstr *LastMI =
      Indexes->getInstructionFromIndex(TaintExtent.front().first);
    assert(LastMI && "Range must end at a proper instruction");
    unsigned TaintNum = 0;
    for(;;) {
      assert(MI != MBB->end() && "Bad LastMI");
      if (usesLanes(MI, Other.Reg, Other.SubIdx, TaintedLanes)) {
        DEBUG(dbgs() << "\t\ttainted lanes used by: " << *MI);
        return false;
      }
      // LastMI is the last instruction to use the current value.
      if (&*MI == LastMI) {
        if (++TaintNum == TaintExtent.size())
          break;
        LastMI = Indexes->getInstructionFromIndex(TaintExtent[TaintNum].first);
        assert(LastMI && "Range must end at a proper instruction");
        TaintedLanes = TaintExtent[TaintNum].second;
      }
      ++MI;
    }

    // The tainted lanes are unused.
    V.Resolution = CR_Replace;
    ++NumLaneResolves;
  }
  return true;
}

bool JoinVals::isPrunedValue(unsigned ValNo, JoinVals &Other) {
  Val &V = Vals[ValNo];
  if (V.Pruned || V.PrunedComputed)
    return V.Pruned;

  if (V.Resolution != CR_Erase && V.Resolution != CR_Merge)
    return V.Pruned;

  // Follow copies up the dominator tree and check if any intermediate value
  // has been pruned.
  V.PrunedComputed = true;
  V.Pruned = Other.isPrunedValue(V.OtherVNI->id, *this);
  return V.Pruned;
}

void JoinVals::pruneValues(JoinVals &Other,
                           SmallVectorImpl<SlotIndex> &EndPoints,
                           bool changeInstrs) {
  for (unsigned i = 0, e = LR.getNumValNums(); i != e; ++i) {
    SlotIndex Def = LR.getValNumInfo(i)->def;
    switch (Vals[i].Resolution) {
    case CR_Keep:
      break;
    case CR_Replace: {
      // This value takes precedence over the value in Other.LR.
      LIS->pruneValue(Other.LR, Def, &EndPoints);
      // Check if we're replacing an IMPLICIT_DEF value. The IMPLICIT_DEF
      // instructions are only inserted to provide a live-out value for PHI
      // predecessors, so the instruction should simply go away once its value
      // has been replaced.
      Val &OtherV = Other.Vals[Vals[i].OtherVNI->id];
      bool EraseImpDef = OtherV.ErasableImplicitDef &&
                         OtherV.Resolution == CR_Keep;
      if (!Def.isBlock()) {
        if (changeInstrs) {
          // Remove <def,read-undef> flags. This def is now a partial redef.
          // Also remove <def,dead> flags since the joined live range will
          // continue past this instruction.
          for (MachineOperand &MO :
               Indexes->getInstructionFromIndex(Def)->operands()) {
            if (MO.isReg() && MO.isDef() && MO.getReg() == Reg) {
              MO.setIsUndef(EraseImpDef);
              MO.setIsDead(false);
            }
          }
        }
        // This value will reach instructions below, but we need to make sure
        // the live range also reaches the instruction at Def.
        if (!EraseImpDef)
          EndPoints.push_back(Def);
      }
      DEBUG(dbgs() << "\t\tpruned " << PrintReg(Other.Reg) << " at " << Def
                   << ": " << Other.LR << '\n');
      break;
    }
    case CR_Erase:
    case CR_Merge:
      if (isPrunedValue(i, Other)) {
        // This value is ultimately a copy of a pruned value in LR or Other.LR.
        // We can no longer trust the value mapping computed by
        // computeAssignment(), the value that was originally copied could have
        // been replaced.
        LIS->pruneValue(LR, Def, &EndPoints);
        DEBUG(dbgs() << "\t\tpruned all of " << PrintReg(Reg) << " at "
                     << Def << ": " << LR << '\n');
      }
      break;
    case CR_Unresolved:
    case CR_Impossible:
      llvm_unreachable("Unresolved conflicts");
    }
  }
}

void JoinVals::pruneSubRegValues(LiveInterval &LI, unsigned &ShrinkMask)
{
  // Look for values being erased.
  bool DidPrune = false;
  for (unsigned i = 0, e = LR.getNumValNums(); i != e; ++i) {
    if (Vals[i].Resolution != CR_Erase)
      continue;

    // Check subranges at the point where the copy will be removed.
    SlotIndex Def = LR.getValNumInfo(i)->def;
    for (LiveInterval::SubRange &S : LI.subranges()) {
      LiveQueryResult Q = S.Query(Def);

      // If a subrange starts at the copy then an undefined value has been
      // copied and we must remove that subrange value as well.
      VNInfo *ValueOut = Q.valueOutOrDead();
      if (ValueOut != nullptr && Q.valueIn() == nullptr) {
        DEBUG(dbgs() << "\t\tPrune sublane " << format("%04X", S.LaneMask)
                     << " at " << Def << "\n");
        LIS->pruneValue(S, Def, nullptr);
        DidPrune = true;
        // Mark value number as unused.
        ValueOut->markUnused();
        continue;
      }
      // If a subrange ends at the copy, then a value was copied but only
      // partially used later. Shrink the subregister range apropriately.
      if (Q.valueIn() != nullptr && Q.valueOut() == nullptr) {
        DEBUG(dbgs() << "\t\tDead uses at sublane "
                     << format("%04X", S.LaneMask) << " at " << Def << "\n");
        ShrinkMask |= S.LaneMask;
      }
    }
  }
  if (DidPrune)
    LI.removeEmptySubRanges();
}

void JoinVals::removeImplicitDefs() {
  for (unsigned i = 0, e = LR.getNumValNums(); i != e; ++i) {
    Val &V = Vals[i];
    if (V.Resolution != CR_Keep || !V.ErasableImplicitDef || !V.Pruned)
      continue;

    VNInfo *VNI = LR.getValNumInfo(i);
    VNI->markUnused();
    LR.removeValNo(VNI);
  }
}

void JoinVals::eraseInstrs(SmallPtrSetImpl<MachineInstr*> &ErasedInstrs,
                           SmallVectorImpl<unsigned> &ShrinkRegs) {
  for (unsigned i = 0, e = LR.getNumValNums(); i != e; ++i) {
    // Get the def location before markUnused() below invalidates it.
    SlotIndex Def = LR.getValNumInfo(i)->def;
    switch (Vals[i].Resolution) {
    case CR_Keep: {
      // If an IMPLICIT_DEF value is pruned, it doesn't serve a purpose any
      // longer. The IMPLICIT_DEF instructions are only inserted by
      // PHIElimination to guarantee that all PHI predecessors have a value.
      if (!Vals[i].ErasableImplicitDef || !Vals[i].Pruned)
        break;
      // Remove value number i from LR.
      VNInfo *VNI = LR.getValNumInfo(i);
      LR.removeValNo(VNI);
      // Note that this VNInfo is reused and still referenced in NewVNInfo,
      // make it appear like an unused value number.
      VNI->markUnused();
      DEBUG(dbgs() << "\t\tremoved " << i << '@' << Def << ": " << LR << '\n');
      // FALL THROUGH.
    }

    case CR_Erase: {
      MachineInstr *MI = Indexes->getInstructionFromIndex(Def);
      assert(MI && "No instruction to erase");
      if (MI->isCopy()) {
        unsigned Reg = MI->getOperand(1).getReg();
        if (TargetRegisterInfo::isVirtualRegister(Reg) &&
            Reg != CP.getSrcReg() && Reg != CP.getDstReg())
          ShrinkRegs.push_back(Reg);
      }
      ErasedInstrs.insert(MI);
      DEBUG(dbgs() << "\t\terased:\t" << Def << '\t' << *MI);
      LIS->RemoveMachineInstrFromMaps(MI);
      MI->eraseFromParent();
      break;
    }
    default:
      break;
    }
  }
}

bool RegisterCoalescer::joinSubRegRanges(LiveRange &LRange, LiveRange &RRange,
                                         unsigned LaneMask,
                                         const CoalescerPair &CP) {
  SmallVector<VNInfo*, 16> NewVNInfo;
  JoinVals RHSVals(RRange, CP.getSrcReg(), CP.getSrcIdx(), LaneMask,
                   NewVNInfo, CP, LIS, TRI, true, true);
  JoinVals LHSVals(LRange, CP.getDstReg(), CP.getDstIdx(), LaneMask,
                   NewVNInfo, CP, LIS, TRI, true, true);

  // Compute NewVNInfo and resolve conflicts (see also joinVirtRegs())
  // We should be able to resolve all conflicts here as we could successfully do
  // it on the mainrange already. There is however a problem when multiple
  // ranges get mapped to the "overflow" lane mask bit which creates unexpected
  // interferences.
  if (!LHSVals.mapValues(RHSVals) || !RHSVals.mapValues(LHSVals)) {
    DEBUG(dbgs() << "*** Couldn't join subrange!\n");
    return false;
  }
  if (!LHSVals.resolveConflicts(RHSVals) ||
      !RHSVals.resolveConflicts(LHSVals)) {
    DEBUG(dbgs() << "*** Couldn't join subrange!\n");
    return false;
  }

  // The merging algorithm in LiveInterval::join() can't handle conflicting
  // value mappings, so we need to remove any live ranges that overlap a
  // CR_Replace resolution. Collect a set of end points that can be used to
  // restore the live range after joining.
  SmallVector<SlotIndex, 8> EndPoints;
  LHSVals.pruneValues(RHSVals, EndPoints, false);
  RHSVals.pruneValues(LHSVals, EndPoints, false);

  LHSVals.removeImplicitDefs();
  RHSVals.removeImplicitDefs();

  LRange.verify();
  RRange.verify();

  // Join RRange into LHS.
  LRange.join(RRange, LHSVals.getAssignments(), RHSVals.getAssignments(),
              NewVNInfo);

  DEBUG(dbgs() << "\t\tjoined lanes: " << LRange << "\n");
  if (EndPoints.empty())
    return true;

  // Recompute the parts of the live range we had to remove because of
  // CR_Replace conflicts.
  DEBUG(dbgs() << "\t\trestoring liveness to " << EndPoints.size()
               << " points: " << LRange << '\n');
  LIS->extendToIndices(LRange, EndPoints);
  return true;
}

bool RegisterCoalescer::mergeSubRangeInto(LiveInterval &LI,
                                          const LiveRange &ToMerge,
                                          unsigned LaneMask, CoalescerPair &CP) {
  BumpPtrAllocator &Allocator = LIS->getVNInfoAllocator();
  for (LiveInterval::SubRange &R : LI.subranges()) {
    unsigned RMask = R.LaneMask;
    // LaneMask of subregisters common to subrange R and ToMerge.
    unsigned Common = RMask & LaneMask;
    // There is nothing to do without common subregs.
    if (Common == 0)
      continue;

    DEBUG(dbgs() << format("\t\tCopy+Merge %04X into %04X\n", RMask, Common));
    // LaneMask of subregisters contained in the R range but not in ToMerge,
    // they have to split into their own subrange.
    unsigned LRest = RMask & ~LaneMask;
    LiveInterval::SubRange *CommonRange;
    if (LRest != 0) {
      R.LaneMask = LRest;
      DEBUG(dbgs() << format("\t\tReduce Lane to %04X\n", LRest));
      // Duplicate SubRange for newly merged common stuff.
      CommonRange = LI.createSubRangeFrom(Allocator, Common, R);
    } else {
      // Reuse the existing range.
      R.LaneMask = Common;
      CommonRange = &R;
    }
    LiveRange RangeCopy(ToMerge, Allocator);
    if (!joinSubRegRanges(*CommonRange, RangeCopy, Common, CP))
      return false;
    LaneMask &= ~RMask;
  }

  if (LaneMask != 0) {
    DEBUG(dbgs() << format("\t\tNew Lane %04X\n", LaneMask));
    LI.createSubRangeFrom(Allocator, LaneMask, ToMerge);
  }
  return true;
}

bool RegisterCoalescer::joinVirtRegs(CoalescerPair &CP) {
  SmallVector<VNInfo*, 16> NewVNInfo;
  LiveInterval &RHS = LIS->getInterval(CP.getSrcReg());
  LiveInterval &LHS = LIS->getInterval(CP.getDstReg());
  bool TrackSubRegLiveness = MRI->shouldTrackSubRegLiveness(*CP.getNewRC());
  JoinVals RHSVals(RHS, CP.getSrcReg(), CP.getSrcIdx(), 0, NewVNInfo, CP, LIS,
                   TRI, false, TrackSubRegLiveness);
  JoinVals LHSVals(LHS, CP.getDstReg(), CP.getDstIdx(), 0, NewVNInfo, CP, LIS,
                   TRI, false, TrackSubRegLiveness);

  DEBUG(dbgs() << "\t\tRHS = " << RHS
               << "\n\t\tLHS = " << LHS
               << '\n');

  // First compute NewVNInfo and the simple value mappings.
  // Detect impossible conflicts early.
  if (!LHSVals.mapValues(RHSVals) || !RHSVals.mapValues(LHSVals))
    return false;

  // Some conflicts can only be resolved after all values have been mapped.
  if (!LHSVals.resolveConflicts(RHSVals) || !RHSVals.resolveConflicts(LHSVals))
    return false;

  // All clear, the live ranges can be merged.
  if (RHS.hasSubRanges() || LHS.hasSubRanges()) {
    BumpPtrAllocator &Allocator = LIS->getVNInfoAllocator();

    // Transform lanemasks from the LHS to masks in the coalesced register and
    // create initial subranges if necessary.
    unsigned DstIdx = CP.getDstIdx();
    if (!LHS.hasSubRanges()) {
      unsigned Mask = DstIdx == 0 ? CP.getNewRC()->getLaneMask()
                                  : TRI->getSubRegIndexLaneMask(DstIdx);
      // LHS must support subregs or we wouldn't be in this codepath.
      assert(Mask != 0);
      LHS.createSubRangeFrom(Allocator, Mask, LHS);
    } else if (DstIdx != 0) {
      // Transform LHS lanemasks to new register class if necessary.
      for (LiveInterval::SubRange &R : LHS.subranges()) {
        unsigned Mask = TRI->composeSubRegIndexLaneMask(DstIdx, R.LaneMask);
        R.LaneMask = Mask;
      }
    }
    DEBUG(dbgs() << "\t\tLHST = " << PrintReg(CP.getDstReg())
                 << ' ' << LHS << '\n');

    // Determine lanemasks of RHS in the coalesced register and merge subranges.
    unsigned SrcIdx = CP.getSrcIdx();
    bool Abort = false;
    if (!RHS.hasSubRanges()) {
      unsigned Mask = SrcIdx == 0 ? CP.getNewRC()->getLaneMask()
                                  : TRI->getSubRegIndexLaneMask(SrcIdx);
      if (!mergeSubRangeInto(LHS, RHS, Mask, CP))
        Abort = true;
    } else {
      // Pair up subranges and merge.
      for (LiveInterval::SubRange &R : RHS.subranges()) {
        unsigned Mask = TRI->composeSubRegIndexLaneMask(SrcIdx, R.LaneMask);
        if (!mergeSubRangeInto(LHS, R, Mask, CP)) {
          Abort = true;
          break;
        }
      }
    }
    if (Abort) {
      // This shouldn't have happened :-(
      // However we are aware of at least one existing problem where we
      // can't merge subranges when multiple ranges end up in the
      // "overflow bit" 32. As a workaround we drop all subregister ranges
      // which means we loose some precision but are back to a well defined
      // state.
      assert(TargetRegisterInfo::isImpreciseLaneMask(
             CP.getNewRC()->getLaneMask())
             && "SubRange merge should only fail when merging into bit 32.");
      DEBUG(dbgs() << "\tSubrange join aborted!\n");
      LHS.clearSubRanges();
      RHS.clearSubRanges();
    } else {
      DEBUG(dbgs() << "\tJoined SubRanges " << LHS << "\n");

      LHSVals.pruneSubRegValues(LHS, ShrinkMask);
      RHSVals.pruneSubRegValues(LHS, ShrinkMask);
    }
  }

  // The merging algorithm in LiveInterval::join() can't handle conflicting
  // value mappings, so we need to remove any live ranges that overlap a
  // CR_Replace resolution. Collect a set of end points that can be used to
  // restore the live range after joining.
  SmallVector<SlotIndex, 8> EndPoints;
  LHSVals.pruneValues(RHSVals, EndPoints, true);
  RHSVals.pruneValues(LHSVals, EndPoints, true);

  // Erase COPY and IMPLICIT_DEF instructions. This may cause some external
  // registers to require trimming.
  SmallVector<unsigned, 8> ShrinkRegs;
  LHSVals.eraseInstrs(ErasedInstrs, ShrinkRegs);
  RHSVals.eraseInstrs(ErasedInstrs, ShrinkRegs);
  while (!ShrinkRegs.empty())
    shrinkToUses(&LIS->getInterval(ShrinkRegs.pop_back_val()));

  // Join RHS into LHS.
  LHS.join(RHS, LHSVals.getAssignments(), RHSVals.getAssignments(), NewVNInfo);

  // Kill flags are going to be wrong if the live ranges were overlapping.
  // Eventually, we should simply clear all kill flags when computing live
  // ranges. They are reinserted after register allocation.
  MRI->clearKillFlags(LHS.reg);
  MRI->clearKillFlags(RHS.reg);

  if (!EndPoints.empty()) {
    // Recompute the parts of the live range we had to remove because of
    // CR_Replace conflicts.
    DEBUG(dbgs() << "\t\trestoring liveness to " << EndPoints.size()
                 << " points: " << LHS << '\n');
    LIS->extendToIndices((LiveRange&)LHS, EndPoints);
  }

  return true;
}

bool RegisterCoalescer::joinIntervals(CoalescerPair &CP) {
  return CP.isPhys() ? joinReservedPhysReg(CP) : joinVirtRegs(CP);
}

namespace {
/// Information concerning MBB coalescing priority.
struct MBBPriorityInfo {
  MachineBasicBlock *MBB;
  unsigned Depth;
  bool IsSplit;

  MBBPriorityInfo(MachineBasicBlock *mbb, unsigned depth, bool issplit)
    : MBB(mbb), Depth(depth), IsSplit(issplit) {}
};
}

/// C-style comparator that sorts first based on the loop depth of the basic
/// block (the unsigned), and then on the MBB number.
///
/// EnableGlobalCopies assumes that the primary sort key is loop depth.
// HLSL Change: changed calling convention to __cdecl
static int __cdecl compareMBBPriority(const MBBPriorityInfo *LHS,
                              const MBBPriorityInfo *RHS) {
  // Deeper loops first
  if (LHS->Depth != RHS->Depth)
    return LHS->Depth > RHS->Depth ? -1 : 1;

  // Try to unsplit critical edges next.
  if (LHS->IsSplit != RHS->IsSplit)
    return LHS->IsSplit ? -1 : 1;

  // Prefer blocks that are more connected in the CFG. This takes care of
  // the most difficult copies first while intervals are short.
  unsigned cl = LHS->MBB->pred_size() + LHS->MBB->succ_size();
  unsigned cr = RHS->MBB->pred_size() + RHS->MBB->succ_size();
  if (cl != cr)
    return cl > cr ? -1 : 1;

  // As a last resort, sort by block number.
  return LHS->MBB->getNumber() < RHS->MBB->getNumber() ? -1 : 1;
}

/// \returns true if the given copy uses or defines a local live range.
static bool isLocalCopy(MachineInstr *Copy, const LiveIntervals *LIS) {
  if (!Copy->isCopy())
    return false;

  if (Copy->getOperand(1).isUndef())
    return false;

  unsigned SrcReg = Copy->getOperand(1).getReg();
  unsigned DstReg = Copy->getOperand(0).getReg();
  if (TargetRegisterInfo::isPhysicalRegister(SrcReg)
      || TargetRegisterInfo::isPhysicalRegister(DstReg))
    return false;

  return LIS->intervalIsInOneMBB(LIS->getInterval(SrcReg))
    || LIS->intervalIsInOneMBB(LIS->getInterval(DstReg));
}

bool RegisterCoalescer::
copyCoalesceWorkList(MutableArrayRef<MachineInstr*> CurrList) {
  bool Progress = false;
  for (unsigned i = 0, e = CurrList.size(); i != e; ++i) {
    if (!CurrList[i])
      continue;
    // Skip instruction pointers that have already been erased, for example by
    // dead code elimination.
    if (ErasedInstrs.erase(CurrList[i])) {
      CurrList[i] = nullptr;
      continue;
    }
    bool Again = false;
    bool Success = joinCopy(CurrList[i], Again);
    Progress |= Success;
    if (Success || !Again)
      CurrList[i] = nullptr;
  }
  return Progress;
}

/// Check if DstReg is a terminal node.
/// I.e., it does not have any affinity other than \p Copy.
static bool isTerminalReg(unsigned DstReg, const MachineInstr &Copy,
                          const MachineRegisterInfo *MRI) {
  assert(Copy.isCopyLike());
  // Check if the destination of this copy as any other affinity.
  for (const MachineInstr &MI : MRI->reg_nodbg_instructions(DstReg))
    if (&MI != &Copy && MI.isCopyLike())
      return false;
  return true;
}

bool RegisterCoalescer::applyTerminalRule(const MachineInstr &Copy) const {
  assert(Copy.isCopyLike());
  if (!UseTerminalRule)
    return false;
  unsigned DstReg, DstSubReg, SrcReg, SrcSubReg;
  isMoveInstr(*TRI, &Copy, SrcReg, DstReg, SrcSubReg, DstSubReg);
  // Check if the destination of this copy has any other affinity.
  if (TargetRegisterInfo::isPhysicalRegister(DstReg) ||
      // If SrcReg is a physical register, the copy won't be coalesced.
      // Ignoring it may have other side effect (like missing
      // rematerialization). So keep it.
      TargetRegisterInfo::isPhysicalRegister(SrcReg) ||
      !isTerminalReg(DstReg, Copy, MRI))
    return false;

  // DstReg is a terminal node. Check if it inteferes with any other
  // copy involving SrcReg.
  const MachineBasicBlock *OrigBB = Copy.getParent();
  const LiveInterval &DstLI = LIS->getInterval(DstReg);
  for (const MachineInstr &MI : MRI->reg_nodbg_instructions(SrcReg)) {
    // Technically we should check if the weight of the new copy is
    // interesting compared to the other one and update the weight
    // of the copies accordingly. However, this would only work if
    // we would gather all the copies first then coalesce, whereas
    // right now we interleave both actions.
    // For now, just consider the copies that are in the same block.
    if (&MI == &Copy || !MI.isCopyLike() || MI.getParent() != OrigBB)
      continue;
    unsigned OtherReg, OtherSubReg, OtherSrcReg, OtherSrcSubReg;
    isMoveInstr(*TRI, &Copy, OtherSrcReg, OtherReg, OtherSrcSubReg,
                OtherSubReg);
    if (OtherReg == SrcReg)
      OtherReg = OtherSrcReg;
    // Check if OtherReg is a non-terminal.
    if (TargetRegisterInfo::isPhysicalRegister(OtherReg) ||
        isTerminalReg(OtherReg, MI, MRI))
      continue;
    // Check that OtherReg interfere with DstReg.
    if (LIS->getInterval(OtherReg).overlaps(DstLI)) {
      DEBUG(dbgs() << "Apply terminal rule for: " << PrintReg(DstReg) << '\n');
      return true;
    }
  }
  return false;
}

void
RegisterCoalescer::copyCoalesceInMBB(MachineBasicBlock *MBB) {
  DEBUG(dbgs() << MBB->getName() << ":\n");

  // Collect all copy-like instructions in MBB. Don't start coalescing anything
  // yet, it might invalidate the iterator.
  const unsigned PrevSize = WorkList.size();
  if (JoinGlobalCopies) {
    SmallVector<MachineInstr*, 2> LocalTerminals;
    SmallVector<MachineInstr*, 2> GlobalTerminals;
    // Coalesce copies bottom-up to coalesce local defs before local uses. They
    // are not inherently easier to resolve, but slightly preferable until we
    // have local live range splitting. In particular this is required by
    // cmp+jmp macro fusion.
    for (MachineBasicBlock::iterator MII = MBB->begin(), E = MBB->end();
         MII != E; ++MII) {
      if (!MII->isCopyLike())
        continue;
      bool ApplyTerminalRule = applyTerminalRule(*MII);
      if (isLocalCopy(&(*MII), LIS)) {
        if (ApplyTerminalRule)
          LocalTerminals.push_back(&(*MII));
        else
          LocalWorkList.push_back(&(*MII));
      } else {
        if (ApplyTerminalRule)
          GlobalTerminals.push_back(&(*MII));
        else
          WorkList.push_back(&(*MII));
      }
    }
    // Append the copies evicted by the terminal rule at the end of the list.
    LocalWorkList.append(LocalTerminals.begin(), LocalTerminals.end());
    WorkList.append(GlobalTerminals.begin(), GlobalTerminals.end());
  }
  else {
    SmallVector<MachineInstr*, 2> Terminals;
     for (MachineBasicBlock::iterator MII = MBB->begin(), E = MBB->end();
          MII != E; ++MII)
       if (MII->isCopyLike()) {
        if (applyTerminalRule(*MII))
          Terminals.push_back(&(*MII));
        else
          WorkList.push_back(MII);
       }
     // Append the copies evicted by the terminal rule at the end of the list.
     WorkList.append(Terminals.begin(), Terminals.end());
  }
  // Try coalescing the collected copies immediately, and remove the nulls.
  // This prevents the WorkList from getting too large since most copies are
  // joinable on the first attempt.
  MutableArrayRef<MachineInstr*>
    CurrList(WorkList.begin() + PrevSize, WorkList.end());
  if (copyCoalesceWorkList(CurrList))
    WorkList.erase(std::remove(WorkList.begin() + PrevSize, WorkList.end(),
                               (MachineInstr*)nullptr), WorkList.end());
}

void RegisterCoalescer::coalesceLocals() {
  copyCoalesceWorkList(LocalWorkList);
  for (unsigned j = 0, je = LocalWorkList.size(); j != je; ++j) {
    if (LocalWorkList[j])
      WorkList.push_back(LocalWorkList[j]);
  }
  LocalWorkList.clear();
}

void RegisterCoalescer::joinAllIntervals() {
  DEBUG(dbgs() << "********** JOINING INTERVALS ***********\n");
  assert(WorkList.empty() && LocalWorkList.empty() && "Old data still around.");

  std::vector<MBBPriorityInfo> MBBs;
  MBBs.reserve(MF->size());
  for (MachineFunction::iterator I = MF->begin(), E = MF->end();I != E;++I){
    MachineBasicBlock *MBB = I;
    MBBs.push_back(MBBPriorityInfo(MBB, Loops->getLoopDepth(MBB),
                                   JoinSplitEdges && isSplitEdge(MBB)));
  }
  array_pod_sort(MBBs.begin(), MBBs.end(), compareMBBPriority);

  // Coalesce intervals in MBB priority order.
  unsigned CurrDepth = UINT_MAX;
  for (unsigned i = 0, e = MBBs.size(); i != e; ++i) {
    // Try coalescing the collected local copies for deeper loops.
    if (JoinGlobalCopies && MBBs[i].Depth < CurrDepth) {
      coalesceLocals();
      CurrDepth = MBBs[i].Depth;
    }
    copyCoalesceInMBB(MBBs[i].MBB);
  }
  coalesceLocals();

  // Joining intervals can allow other intervals to be joined.  Iteratively join
  // until we make no progress.
  while (copyCoalesceWorkList(WorkList))
    /* empty */ ;
}

void RegisterCoalescer::releaseMemory() {
  ErasedInstrs.clear();
  WorkList.clear();
  DeadDefs.clear();
  InflateRegs.clear();
}

bool RegisterCoalescer::runOnMachineFunction(MachineFunction &fn) {
  MF = &fn;
  MRI = &fn.getRegInfo();
  TM = &fn.getTarget();
  const TargetSubtargetInfo &STI = fn.getSubtarget();
  TRI = STI.getRegisterInfo();
  TII = STI.getInstrInfo();
  LIS = &getAnalysis<LiveIntervals>();
  AA = &getAnalysis<AliasAnalysis>();
  Loops = &getAnalysis<MachineLoopInfo>();
  if (EnableGlobalCopies == cl::BOU_UNSET)
    JoinGlobalCopies = STI.enableJoinGlobalCopies();
  else
    JoinGlobalCopies = (EnableGlobalCopies == cl::BOU_TRUE);

  // The MachineScheduler does not currently require JoinSplitEdges. This will
  // either be enabled unconditionally or replaced by a more general live range
  // splitting optimization.
  JoinSplitEdges = EnableJoinSplits;

  DEBUG(dbgs() << "********** SIMPLE REGISTER COALESCING **********\n"
               << "********** Function: " << MF->getName() << '\n');

  if (VerifyCoalescing)
    MF->verify(this, "Before register coalescing");

  RegClassInfo.runOnMachineFunction(fn);

  // Join (coalesce) intervals if requested.
  if (EnableJoining)
    joinAllIntervals();

  // After deleting a lot of copies, register classes may be less constrained.
  // Removing sub-register operands may allow GR32_ABCD -> GR32 and DPR_VFP2 ->
  // DPR inflation.
  array_pod_sort(InflateRegs.begin(), InflateRegs.end());
  InflateRegs.erase(std::unique(InflateRegs.begin(), InflateRegs.end()),
                    InflateRegs.end());
  DEBUG(dbgs() << "Trying to inflate " << InflateRegs.size() << " regs.\n");
  for (unsigned i = 0, e = InflateRegs.size(); i != e; ++i) {
    unsigned Reg = InflateRegs[i];
    if (MRI->reg_nodbg_empty(Reg))
      continue;
    if (MRI->recomputeRegClass(Reg)) {
      DEBUG(dbgs() << PrintReg(Reg) << " inflated to "
                   << TRI->getRegClassName(MRI->getRegClass(Reg)) << '\n');
      LiveInterval &LI = LIS->getInterval(Reg);
      unsigned MaxMask = MRI->getMaxLaneMaskForVReg(Reg);
      if (MaxMask == 0) {
        // If the inflated register class does not support subregisters anymore
        // remove the subranges.
        LI.clearSubRanges();
      } else {
#ifndef NDEBUG
        // If subranges are still supported, then the same subregs should still
        // be supported.
        for (LiveInterval::SubRange &S : LI.subranges()) {
          assert ((S.LaneMask & ~MaxMask) == 0);
        }
#endif
      }
      ++NumInflated;
    }
  }

  DEBUG(dump());
  if (VerifyCoalescing)
    MF->verify(this, "After register coalescing");
  return true;
}

void RegisterCoalescer::print(raw_ostream &O, const Module* m) const {
   LIS->print(O, m);
}
