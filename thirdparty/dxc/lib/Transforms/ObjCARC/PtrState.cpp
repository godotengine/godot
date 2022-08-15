//===--- PtrState.cpp -----------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "PtrState.h"
#include "DependencyAnalysis.h"
#include "ObjCARC.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::objcarc;

#define DEBUG_TYPE "objc-arc-ptr-state"

//===----------------------------------------------------------------------===//
//                                  Utility
//===----------------------------------------------------------------------===//

raw_ostream &llvm::objcarc::operator<<(raw_ostream &OS, const Sequence S) {
  switch (S) {
  case S_None:
    return OS << "S_None";
  case S_Retain:
    return OS << "S_Retain";
  case S_CanRelease:
    return OS << "S_CanRelease";
  case S_Use:
    return OS << "S_Use";
  case S_Release:
    return OS << "S_Release";
  case S_MovableRelease:
    return OS << "S_MovableRelease";
  case S_Stop:
    return OS << "S_Stop";
  }
  llvm_unreachable("Unknown sequence type.");
}

//===----------------------------------------------------------------------===//
//                                  Sequence
//===----------------------------------------------------------------------===//

static Sequence MergeSeqs(Sequence A, Sequence B, bool TopDown) {
  // The easy cases.
  if (A == B)
    return A;
  if (A == S_None || B == S_None)
    return S_None;

  if (A > B)
    std::swap(A, B);
  if (TopDown) {
    // Choose the side which is further along in the sequence.
    if ((A == S_Retain || A == S_CanRelease) &&
        (B == S_CanRelease || B == S_Use))
      return B;
  } else {
    // Choose the side which is further along in the sequence.
    if ((A == S_Use || A == S_CanRelease) &&
        (B == S_Use || B == S_Release || B == S_Stop || B == S_MovableRelease))
      return A;
    // If both sides are releases, choose the more conservative one.
    if (A == S_Stop && (B == S_Release || B == S_MovableRelease))
      return A;
    if (A == S_Release && B == S_MovableRelease)
      return A;
  }

  return S_None;
}

//===----------------------------------------------------------------------===//
//                                   RRInfo
//===----------------------------------------------------------------------===//

void RRInfo::clear() {
  KnownSafe = false;
  IsTailCallRelease = false;
  ReleaseMetadata = nullptr;
  Calls.clear();
  ReverseInsertPts.clear();
  CFGHazardAfflicted = false;
}

bool RRInfo::Merge(const RRInfo &Other) {
  // Conservatively merge the ReleaseMetadata information.
  if (ReleaseMetadata != Other.ReleaseMetadata)
    ReleaseMetadata = nullptr;

  // Conservatively merge the boolean state.
  KnownSafe &= Other.KnownSafe;
  IsTailCallRelease &= Other.IsTailCallRelease;
  CFGHazardAfflicted |= Other.CFGHazardAfflicted;

  // Merge the call sets.
  Calls.insert(Other.Calls.begin(), Other.Calls.end());

  // Merge the insert point sets. If there are any differences,
  // that makes this a partial merge.
  bool Partial = ReverseInsertPts.size() != Other.ReverseInsertPts.size();
  for (Instruction *Inst : Other.ReverseInsertPts)
    Partial |= ReverseInsertPts.insert(Inst).second;
  return Partial;
}

//===----------------------------------------------------------------------===//
//                                  PtrState
//===----------------------------------------------------------------------===//

void PtrState::SetKnownPositiveRefCount() {
  DEBUG(dbgs() << "        Setting Known Positive.\n");
  KnownPositiveRefCount = true;
}

void PtrState::ClearKnownPositiveRefCount() {
  DEBUG(dbgs() << "        Clearing Known Positive.\n");
  KnownPositiveRefCount = false;
}

void PtrState::SetSeq(Sequence NewSeq) {
  DEBUG(dbgs() << "            Old: " << GetSeq() << "; New: " << NewSeq << "\n");
  Seq = NewSeq;
}

void PtrState::ResetSequenceProgress(Sequence NewSeq) {
  DEBUG(dbgs() << "        Resetting sequence progress.\n");
  SetSeq(NewSeq);
  Partial = false;
  RRI.clear();
}

void PtrState::Merge(const PtrState &Other, bool TopDown) {
  Seq = MergeSeqs(GetSeq(), Other.GetSeq(), TopDown);
  KnownPositiveRefCount &= Other.KnownPositiveRefCount;

  // If we're not in a sequence (anymore), drop all associated state.
  if (Seq == S_None) {
    Partial = false;
    RRI.clear();
  } else if (Partial || Other.Partial) {
    // If we're doing a merge on a path that's previously seen a partial
    // merge, conservatively drop the sequence, to avoid doing partial
    // RR elimination. If the branch predicates for the two merge differ,
    // mixing them is unsafe.
    ClearSequenceProgress();
  } else {
    // Otherwise merge the other PtrState's RRInfo into our RRInfo. At this
    // point, we know that currently we are not partial. Stash whether or not
    // the merge operation caused us to undergo a partial merging of reverse
    // insertion points.
    Partial = RRI.Merge(Other.RRI);
  }
}

//===----------------------------------------------------------------------===//
//                              BottomUpPtrState
//===----------------------------------------------------------------------===//

bool BottomUpPtrState::InitBottomUp(ARCMDKindCache &Cache, Instruction *I) {
  // If we see two releases in a row on the same pointer. If so, make
  // a note, and we'll cicle back to revisit it after we've
  // hopefully eliminated the second release, which may allow us to
  // eliminate the first release too.
  // Theoretically we could implement removal of nested retain+release
  // pairs by making PtrState hold a stack of states, but this is
  // simple and avoids adding overhead for the non-nested case.
  bool NestingDetected = false;
  if (GetSeq() == S_Release || GetSeq() == S_MovableRelease) {
    DEBUG(dbgs() << "        Found nested releases (i.e. a release pair)\n");
    NestingDetected = true;
  }

  MDNode *ReleaseMetadata =
      I->getMetadata(Cache.get(ARCMDKindID::ImpreciseRelease));
  Sequence NewSeq = ReleaseMetadata ? S_MovableRelease : S_Release;
  ResetSequenceProgress(NewSeq);
  SetReleaseMetadata(ReleaseMetadata);
  SetKnownSafe(HasKnownPositiveRefCount());
  SetTailCallRelease(cast<CallInst>(I)->isTailCall());
  InsertCall(I);
  SetKnownPositiveRefCount();
  return NestingDetected;
}

bool BottomUpPtrState::MatchWithRetain() {
  SetKnownPositiveRefCount();

  Sequence OldSeq = GetSeq();
  switch (OldSeq) {
  case S_Stop:
  case S_Release:
  case S_MovableRelease:
  case S_Use:
    // If OldSeq is not S_Use or OldSeq is S_Use and we are tracking an
    // imprecise release, clear our reverse insertion points.
    if (OldSeq != S_Use || IsTrackingImpreciseReleases())
      ClearReverseInsertPts();
  // FALL THROUGH
  case S_CanRelease:
    return true;
  case S_None:
    return false;
  case S_Retain:
    llvm_unreachable("bottom-up pointer in retain state!");
  }
  llvm_unreachable("Sequence unknown enum value");
}

bool BottomUpPtrState::HandlePotentialAlterRefCount(Instruction *Inst,
                                                    const Value *Ptr,
                                                    ProvenanceAnalysis &PA,
                                                    ARCInstKind Class) {
  Sequence S = GetSeq();

  // Check for possible releases.
  if (!CanAlterRefCount(Inst, Ptr, PA, Class))
    return false;

  DEBUG(dbgs() << "            CanAlterRefCount: Seq: " << S << "; " << *Ptr
               << "\n");
  switch (S) {
  case S_Use:
    SetSeq(S_CanRelease);
    return true;
  case S_CanRelease:
  case S_Release:
  case S_MovableRelease:
  case S_Stop:
  case S_None:
    return false;
  case S_Retain:
    llvm_unreachable("bottom-up pointer in retain state!");
  }
  llvm_unreachable("Sequence unknown enum value");
}

void BottomUpPtrState::HandlePotentialUse(BasicBlock *BB, Instruction *Inst,
                                          const Value *Ptr,
                                          ProvenanceAnalysis &PA,
                                          ARCInstKind Class) {
  // Check for possible direct uses.
  switch (GetSeq()) {
  case S_Release:
  case S_MovableRelease:
    if (CanUse(Inst, Ptr, PA, Class)) {
      DEBUG(dbgs() << "            CanUse: Seq: " << GetSeq() << "; " << *Ptr
                   << "\n");
      assert(!HasReverseInsertPts());
      // If this is an invoke instruction, we're scanning it as part of
      // one of its successor blocks, since we can't insert code after it
      // in its own block, and we don't want to split critical edges.
      if (isa<InvokeInst>(Inst))
        InsertReverseInsertPt(BB->getFirstInsertionPt());
      else
        InsertReverseInsertPt(std::next(BasicBlock::iterator(Inst)));
      SetSeq(S_Use);
    } else if (Seq == S_Release && IsUser(Class)) {
      DEBUG(dbgs() << "            PreciseReleaseUse: Seq: " << GetSeq() << "; "
                   << *Ptr << "\n");
      // Non-movable releases depend on any possible objc pointer use.
      SetSeq(S_Stop);
      assert(!HasReverseInsertPts());
      // As above; handle invoke specially.
      if (isa<InvokeInst>(Inst))
        InsertReverseInsertPt(BB->getFirstInsertionPt());
      else
        InsertReverseInsertPt(std::next(BasicBlock::iterator(Inst)));
    }
    break;
  case S_Stop:
    if (CanUse(Inst, Ptr, PA, Class)) {
      DEBUG(dbgs() << "            PreciseStopUse: Seq: " << GetSeq() << "; "
                   << *Ptr << "\n");
      SetSeq(S_Use);
    }
    break;
  case S_CanRelease:
  case S_Use:
  case S_None:
    break;
  case S_Retain:
    llvm_unreachable("bottom-up pointer in retain state!");
  }
}

//===----------------------------------------------------------------------===//
//                              TopDownPtrState
//===----------------------------------------------------------------------===//

bool TopDownPtrState::InitTopDown(ARCInstKind Kind, Instruction *I) {
  bool NestingDetected = false;
  // Don't do retain+release tracking for ARCInstKind::RetainRV, because
  // it's
  // better to let it remain as the first instruction after a call.
  if (Kind != ARCInstKind::RetainRV) {
    // If we see two retains in a row on the same pointer. If so, make
    // a note, and we'll cicle back to revisit it after we've
    // hopefully eliminated the second retain, which may allow us to
    // eliminate the first retain too.
    // Theoretically we could implement removal of nested retain+release
    // pairs by making PtrState hold a stack of states, but this is
    // simple and avoids adding overhead for the non-nested case.
    if (GetSeq() == S_Retain)
      NestingDetected = true;

    ResetSequenceProgress(S_Retain);
    SetKnownSafe(HasKnownPositiveRefCount());
    InsertCall(I);
  }

  SetKnownPositiveRefCount();
  return NestingDetected;
}

bool TopDownPtrState::MatchWithRelease(ARCMDKindCache &Cache,
                                       Instruction *Release) {
  ClearKnownPositiveRefCount();

  Sequence OldSeq = GetSeq();

  MDNode *ReleaseMetadata =
      Release->getMetadata(Cache.get(ARCMDKindID::ImpreciseRelease));

  switch (OldSeq) {
  case S_Retain:
  case S_CanRelease:
    if (OldSeq == S_Retain || ReleaseMetadata != nullptr)
      ClearReverseInsertPts();
  // FALL THROUGH
  case S_Use:
    SetReleaseMetadata(ReleaseMetadata);
    SetTailCallRelease(cast<CallInst>(Release)->isTailCall());
    return true;
  case S_None:
    return false;
  case S_Stop:
  case S_Release:
  case S_MovableRelease:
    llvm_unreachable("top-down pointer in bottom up state!");
  }
  llvm_unreachable("Sequence unknown enum value");
}

bool TopDownPtrState::HandlePotentialAlterRefCount(Instruction *Inst,
                                                   const Value *Ptr,
                                                   ProvenanceAnalysis &PA,
                                                   ARCInstKind Class) {
  // Check for possible releases.
  if (!CanAlterRefCount(Inst, Ptr, PA, Class))
    return false;

  DEBUG(dbgs() << "            CanAlterRefCount: Seq: " << GetSeq() << "; " << *Ptr
               << "\n");
  ClearKnownPositiveRefCount();
  switch (GetSeq()) {
  case S_Retain:
    SetSeq(S_CanRelease);
    assert(!HasReverseInsertPts());
    InsertReverseInsertPt(Inst);

    // One call can't cause a transition from S_Retain to S_CanRelease
    // and S_CanRelease to S_Use. If we've made the first transition,
    // we're done.
    return true;
  case S_Use:
  case S_CanRelease:
  case S_None:
    return false;
  case S_Stop:
  case S_Release:
  case S_MovableRelease:
    llvm_unreachable("top-down pointer in release state!");
  }
  llvm_unreachable("covered switch is not covered!?");
}

void TopDownPtrState::HandlePotentialUse(Instruction *Inst, const Value *Ptr,
                                         ProvenanceAnalysis &PA,
                                         ARCInstKind Class) {
  // Check for possible direct uses.
  switch (GetSeq()) {
  case S_CanRelease:
    if (!CanUse(Inst, Ptr, PA, Class))
      return;
    DEBUG(dbgs() << "             CanUse: Seq: " << GetSeq() << "; " << *Ptr
                 << "\n");
    SetSeq(S_Use);
    return;
  case S_Retain:
  case S_Use:
  case S_None:
    return;
  case S_Stop:
  case S_Release:
  case S_MovableRelease:
    llvm_unreachable("top-down pointer in release state!");
  }
}
