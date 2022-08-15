//===-- LiveIntervalUnion.h - Live interval union data struct --*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// LiveIntervalUnion is a union of live segments across multiple live virtual
// registers. This may be used during coalescing to represent a congruence
// class, or during register allocation to model liveness of a physical
// register.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_LIVEINTERVALUNION_H
#define LLVM_CODEGEN_LIVEINTERVALUNION_H

#include "llvm/ADT/IntervalMap.h"
#include "llvm/CodeGen/LiveInterval.h"

namespace llvm {

class TargetRegisterInfo;

#ifndef NDEBUG
// forward declaration
template <unsigned Element> class SparseBitVector;
typedef SparseBitVector<128> LiveVirtRegBitSet;
#endif

/// Compare a live virtual register segment to a LiveIntervalUnion segment.
inline bool
overlap(const LiveInterval::Segment &VRSeg,
        const IntervalMap<SlotIndex, LiveInterval*>::const_iterator &LUSeg) {
  return VRSeg.start < LUSeg.stop() && LUSeg.start() < VRSeg.end;
}

/// Union of live intervals that are strong candidates for coalescing into a
/// single register (either physical or virtual depending on the context).  We
/// expect the constituent live intervals to be disjoint, although we may
/// eventually make exceptions to handle value-based interference.
class LiveIntervalUnion {
  // A set of live virtual register segments that supports fast insertion,
  // intersection, and removal.
  // Mapping SlotIndex intervals to virtual register numbers.
  typedef IntervalMap<SlotIndex, LiveInterval*> LiveSegments;

public:
  // SegmentIter can advance to the next segment ordered by starting position
  // which may belong to a different live virtual register. We also must be able
  // to reach the current segment's containing virtual register.
  typedef LiveSegments::iterator SegmentIter;

  // LiveIntervalUnions share an external allocator.
  typedef LiveSegments::Allocator Allocator;

  class Query;

private:
  unsigned Tag;           // unique tag for current contents.
  LiveSegments Segments;  // union of virtual reg segments

public:
  explicit LiveIntervalUnion(Allocator &a) : Tag(0), Segments(a) {}

  // Iterate over all segments in the union of live virtual registers ordered
  // by their starting position.
  SegmentIter begin() { return Segments.begin(); }
  SegmentIter end() { return Segments.end(); }
  SegmentIter find(SlotIndex x) { return Segments.find(x); }
  bool empty() const { return Segments.empty(); }
  SlotIndex startIndex() const { return Segments.start(); }

  // Provide public access to the underlying map to allow overlap iteration.
  typedef LiveSegments Map;
  const Map &getMap() { return Segments; }

  /// getTag - Return an opaque tag representing the current state of the union.
  unsigned getTag() const { return Tag; }

  /// changedSince - Return true if the union change since getTag returned tag.
  bool changedSince(unsigned tag) const { return tag != Tag; }

  // Add a live virtual register to this union and merge its segments.
  void unify(LiveInterval &VirtReg, const LiveRange &Range);
  void unify(LiveInterval &VirtReg) {
    unify(VirtReg, VirtReg);
  }

  // Remove a live virtual register's segments from this union.
  void extract(LiveInterval &VirtReg, const LiveRange &Range);
  void extract(LiveInterval &VirtReg) {
    extract(VirtReg, VirtReg);
  }

  // Remove all inserted virtual registers.
  void clear() { Segments.clear(); ++Tag; }

  // Print union, using TRI to translate register names
  void print(raw_ostream &OS, const TargetRegisterInfo *TRI) const;

#ifndef NDEBUG
  // Verify the live intervals in this union and add them to the visited set.
  void verify(LiveVirtRegBitSet& VisitedVRegs);
#endif

  /// Query interferences between a single live virtual register and a live
  /// interval union.
  class Query {
    LiveIntervalUnion *LiveUnion;
    LiveInterval *VirtReg;
    LiveInterval::iterator VirtRegI; // current position in VirtReg
    SegmentIter LiveUnionI;          // current position in LiveUnion
    SmallVector<LiveInterval*,4> InterferingVRegs;
    bool CheckedFirstInterference;
    bool SeenAllInterferences;
    bool SeenUnspillableVReg;
    unsigned Tag, UserTag;

  public:
    Query(): LiveUnion(), VirtReg(), Tag(0), UserTag(0) {}

    Query(LiveInterval *VReg, LiveIntervalUnion *LIU):
      LiveUnion(LIU), VirtReg(VReg), CheckedFirstInterference(false),
      SeenAllInterferences(false), SeenUnspillableVReg(false)
    {}

    void clear() {
      LiveUnion = nullptr;
      VirtReg = nullptr;
      InterferingVRegs.clear();
      CheckedFirstInterference = false;
      SeenAllInterferences = false;
      SeenUnspillableVReg = false;
      Tag = 0;
      UserTag = 0;
    }

    void init(unsigned UTag, LiveInterval *VReg, LiveIntervalUnion *LIU) {
      assert(VReg && LIU && "Invalid arguments");
      if (UserTag == UTag && VirtReg == VReg &&
          LiveUnion == LIU && !LIU->changedSince(Tag)) {
        // Retain cached results, e.g. firstInterference.
        return;
      }
      clear();
      LiveUnion = LIU;
      VirtReg = VReg;
      Tag = LIU->getTag();
      UserTag = UTag;
    }

    LiveInterval &virtReg() const {
      assert(VirtReg && "uninitialized");
      return *VirtReg;
    }

    // Does this live virtual register interfere with the union?
    bool checkInterference() { return collectInterferingVRegs(1); }

    // Count the virtual registers in this union that interfere with this
    // query's live virtual register, up to maxInterferingRegs.
    unsigned collectInterferingVRegs(unsigned MaxInterferingRegs = UINT_MAX);

    // Was this virtual register visited during collectInterferingVRegs?
    bool isSeenInterference(LiveInterval *VReg) const;

    // Did collectInterferingVRegs collect all interferences?
    bool seenAllInterferences() const { return SeenAllInterferences; }

    // Did collectInterferingVRegs encounter an unspillable vreg?
    bool seenUnspillableVReg() const { return SeenUnspillableVReg; }

    // Vector generated by collectInterferingVRegs.
    const SmallVectorImpl<LiveInterval*> &interferingVRegs() const {
      return InterferingVRegs;
    }

  private:
    Query(const Query&) = delete;
    void operator=(const Query&) = delete;
  };

  // Array of LiveIntervalUnions.
  class Array {
    unsigned Size;
    LiveIntervalUnion *LIUs;
  public:
    Array() : Size(0), LIUs(nullptr) {}
    ~Array() { clear(); }

    // Initialize the array to have Size entries.
    // Reuse an existing allocation if the size matches.
    void init(LiveIntervalUnion::Allocator&, unsigned Size);

    unsigned size() const { return Size; }

    void clear();

    LiveIntervalUnion& operator[](unsigned idx) {
      assert(idx <  Size && "idx out of bounds");
      return LIUs[idx];
    }

    const LiveIntervalUnion& operator[](unsigned Idx) const {
      assert(Idx < Size && "Idx out of bounds");
      return LIUs[Idx];
    }
  };
};

} // end namespace llvm

#endif // !defined(LLVM_CODEGEN_LIVEINTERVALUNION_H)
