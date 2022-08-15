//===-- RegisterPressure.h - Dynamic Register Pressure -*- C++ -*-------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the RegisterPressure class which can be used to track
// MachineInstr level register pressure.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_REGISTERPRESSURE_H
#define LLVM_CODEGEN_REGISTERPRESSURE_H

#include "llvm/ADT/SparseSet.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/Target/TargetRegisterInfo.h"

namespace llvm {

class LiveIntervals;
class LiveRange;
class RegisterClassInfo;
class MachineInstr;

/// Base class for register pressure results.
struct RegisterPressure {
  /// Map of max reg pressure indexed by pressure set ID, not class ID.
  std::vector<unsigned> MaxSetPressure;

  /// List of live in virtual registers or physical register units.
  SmallVector<unsigned,8> LiveInRegs;
  SmallVector<unsigned,8> LiveOutRegs;

  void dump(const TargetRegisterInfo *TRI) const;
};

/// RegisterPressure computed within a region of instructions delimited by
/// TopIdx and BottomIdx.  During pressure computation, the maximum pressure per
/// register pressure set is increased. Once pressure within a region is fully
/// computed, the live-in and live-out sets are recorded.
///
/// This is preferable to RegionPressure when LiveIntervals are available,
/// because delimiting regions by SlotIndex is more robust and convenient than
/// holding block iterators. The block contents can change without invalidating
/// the pressure result.
struct IntervalPressure : RegisterPressure {
  /// Record the boundary of the region being tracked.
  SlotIndex TopIdx;
  SlotIndex BottomIdx;

  void reset();

  void openTop(SlotIndex NextTop);

  void openBottom(SlotIndex PrevBottom);
};

/// RegisterPressure computed within a region of instructions delimited by
/// TopPos and BottomPos. This is a less precise version of IntervalPressure for
/// use when LiveIntervals are unavailable.
struct RegionPressure : RegisterPressure {
  /// Record the boundary of the region being tracked.
  MachineBasicBlock::const_iterator TopPos;
  MachineBasicBlock::const_iterator BottomPos;

  void reset();

  void openTop(MachineBasicBlock::const_iterator PrevTop);

  void openBottom(MachineBasicBlock::const_iterator PrevBottom);
};

/// Capture a change in pressure for a single pressure set. UnitInc may be
/// expressed in terms of upward or downward pressure depending on the client
/// and will be dynamically adjusted for current liveness.
///
/// Pressure increments are tiny, typically 1-2 units, and this is only for
/// heuristics, so we don't check UnitInc overflow. Instead, we may have a
/// higher level assert that pressure is consistent within a region. We also
/// effectively ignore dead defs which don't affect heuristics much.
class PressureChange {
  uint16_t PSetID; // ID+1. 0=Invalid.
  int16_t  UnitInc;
public:
  PressureChange(): PSetID(0), UnitInc(0) {}
  PressureChange(unsigned id): PSetID(id+1), UnitInc(0) {
    assert(id < UINT16_MAX && "PSetID overflow.");
  }

  bool isValid() const { return PSetID > 0; }

  unsigned getPSet() const {
    assert(isValid() && "invalid PressureChange");
    return PSetID - 1;
  }
  // If PSetID is invalid, return UINT16_MAX to give it lowest priority.
  unsigned getPSetOrMax() const { return (PSetID - 1) & UINT16_MAX; }

  int getUnitInc() const { return UnitInc; }

  void setUnitInc(int Inc) { UnitInc = Inc; }

  bool operator==(const PressureChange &RHS) const {
    return PSetID == RHS.PSetID && UnitInc == RHS.UnitInc;
  }
};

template <> struct isPodLike<PressureChange> {
   static const bool value = true;
};

/// List of PressureChanges in order of increasing, unique PSetID.
///
/// Use a small fixed number, because we can fit more PressureChanges in an
/// empty SmallVector than ever need to be tracked per register class. If more
/// PSets are affected, then we only track the most constrained.
class PressureDiff {
  // The initial design was for MaxPSets=4, but that requires PSet partitions,
  // which are not yet implemented. (PSet partitions are equivalent PSets given
  // the register classes actually in use within the scheduling region.)
  enum { MaxPSets = 16 };

  PressureChange PressureChanges[MaxPSets];
public:
  typedef PressureChange* iterator;
  typedef const PressureChange* const_iterator;
  iterator begin() { return &PressureChanges[0]; }
  iterator end() { return &PressureChanges[MaxPSets]; }
  const_iterator begin() const { return &PressureChanges[0]; }
  const_iterator end() const { return &PressureChanges[MaxPSets]; }

  void addPressureChange(unsigned RegUnit, bool IsDec,
                         const MachineRegisterInfo *MRI);

  LLVM_DUMP_METHOD void dump(const TargetRegisterInfo &TRI) const;
};

/// Array of PressureDiffs.
class PressureDiffs {
  PressureDiff *PDiffArray;
  unsigned Size;
  unsigned Max;
public:
  PressureDiffs(): PDiffArray(nullptr), Size(0), Max(0) {}
  ~PressureDiffs() { delete[] PDiffArray; } // HLSL Change: Use overridable operator delete

  void clear() { Size = 0; }

  void init(unsigned N);

  PressureDiff &operator[](unsigned Idx) {
    assert(Idx < Size && "PressureDiff index out of bounds");
    return PDiffArray[Idx];
  }
  const PressureDiff &operator[](unsigned Idx) const {
    return const_cast<PressureDiffs*>(this)->operator[](Idx);
  }
};

/// Store the effects of a change in pressure on things that MI scheduler cares
/// about.
///
/// Excess records the value of the largest difference in register units beyond
/// the target's pressure limits across the affected pressure sets, where
/// largest is defined as the absolute value of the difference. Negative
/// ExcessUnits indicates a reduction in pressure that had already exceeded the
/// target's limits.
///
/// CriticalMax records the largest increase in the tracker's max pressure that
/// exceeds the critical limit for some pressure set determined by the client.
///
/// CurrentMax records the largest increase in the tracker's max pressure that
/// exceeds the current limit for some pressure set determined by the client.
struct RegPressureDelta {
  PressureChange Excess;
  PressureChange CriticalMax;
  PressureChange CurrentMax;

  RegPressureDelta() {}

  bool operator==(const RegPressureDelta &RHS) const {
    return Excess == RHS.Excess && CriticalMax == RHS.CriticalMax
      && CurrentMax == RHS.CurrentMax;
  }
  bool operator!=(const RegPressureDelta &RHS) const {
    return !operator==(RHS);
  }
};

/// \brief A set of live virtual registers and physical register units.
///
/// Virtual and physical register numbers require separate sparse sets, but most
/// of the RegisterPressureTracker handles them uniformly.
struct LiveRegSet {
  SparseSet<unsigned> PhysRegs;
  SparseSet<unsigned, VirtReg2IndexFunctor> VirtRegs;

  bool contains(unsigned Reg) const {
    if (TargetRegisterInfo::isVirtualRegister(Reg))
      return VirtRegs.count(Reg);
    return PhysRegs.count(Reg);
  }

  bool insert(unsigned Reg) {
    if (TargetRegisterInfo::isVirtualRegister(Reg))
      return VirtRegs.insert(Reg).second;
    return PhysRegs.insert(Reg).second;
  }

  bool erase(unsigned Reg) {
    if (TargetRegisterInfo::isVirtualRegister(Reg))
      return VirtRegs.erase(Reg);
    return PhysRegs.erase(Reg);
  }
};

/// Track the current register pressure at some position in the instruction
/// stream, and remember the high water mark within the region traversed. This
/// does not automatically consider live-through ranges. The client may
/// independently adjust for global liveness.
///
/// Each RegPressureTracker only works within a MachineBasicBlock. Pressure can
/// be tracked across a larger region by storing a RegisterPressure result at
/// each block boundary and explicitly adjusting pressure to account for block
/// live-in and live-out register sets.
///
/// RegPressureTracker holds a reference to a RegisterPressure result that it
/// computes incrementally. During downward tracking, P.BottomIdx or P.BottomPos
/// is invalid until it reaches the end of the block or closeRegion() is
/// explicitly called. Similarly, P.TopIdx is invalid during upward
/// tracking. Changing direction has the side effect of closing region, and
/// traversing past TopIdx or BottomIdx reopens it.
class RegPressureTracker {
  const MachineFunction     *MF;
  const TargetRegisterInfo  *TRI;
  const RegisterClassInfo   *RCI;
  const MachineRegisterInfo *MRI;
  const LiveIntervals       *LIS;

  /// We currently only allow pressure tracking within a block.
  const MachineBasicBlock *MBB;

  /// Track the max pressure within the region traversed so far.
  RegisterPressure &P;

  /// Run in two modes dependending on whether constructed with IntervalPressure
  /// or RegisterPressure. If requireIntervals is false, LIS are ignored.
  bool RequireIntervals;

  /// True if UntiedDefs will be populated.
  bool TrackUntiedDefs;

  /// Register pressure corresponds to liveness before this instruction
  /// iterator. It may point to the end of the block or a DebugValue rather than
  /// an instruction.
  MachineBasicBlock::const_iterator CurrPos;

  /// Pressure map indexed by pressure set ID, not class ID.
  std::vector<unsigned> CurrSetPressure;

  /// Set of live registers.
  LiveRegSet LiveRegs;

  /// Set of vreg defs that start a live range.
  SparseSet<unsigned, VirtReg2IndexFunctor> UntiedDefs;
  /// Live-through pressure.
  std::vector<unsigned> LiveThruPressure;

public:
  RegPressureTracker(IntervalPressure &rp) :
    MF(nullptr), TRI(nullptr), RCI(nullptr), LIS(nullptr), MBB(nullptr), P(rp),
    RequireIntervals(true), TrackUntiedDefs(false) {}

  RegPressureTracker(RegionPressure &rp) :
    MF(nullptr), TRI(nullptr), RCI(nullptr), LIS(nullptr), MBB(nullptr), P(rp),
    RequireIntervals(false), TrackUntiedDefs(false) {}

  void reset();

  void init(const MachineFunction *mf, const RegisterClassInfo *rci,
            const LiveIntervals *lis, const MachineBasicBlock *mbb,
            MachineBasicBlock::const_iterator pos,
            bool ShouldTrackUntiedDefs = false);

  /// Force liveness of virtual registers or physical register
  /// units. Particularly useful to initialize the livein/out state of the
  /// tracker before the first call to advance/recede.
  void addLiveRegs(ArrayRef<unsigned> Regs);

  /// Get the MI position corresponding to this register pressure.
  MachineBasicBlock::const_iterator getPos() const { return CurrPos; }

  // Reset the MI position corresponding to the register pressure. This allows
  // schedulers to move instructions above the RegPressureTracker's
  // CurrPos. Since the pressure is computed before CurrPos, the iterator
  // position changes while pressure does not.
  void setPos(MachineBasicBlock::const_iterator Pos) { CurrPos = Pos; }

  /// \brief Get the SlotIndex for the first nondebug instruction including or
  /// after the current position.
  SlotIndex getCurrSlot() const;

  /// Recede across the previous instruction.
  bool recede(SmallVectorImpl<unsigned> *LiveUses = nullptr,
              PressureDiff *PDiff = nullptr);

  /// Advance across the current instruction.
  bool advance();

  /// Finalize the region boundaries and recored live ins and live outs.
  void closeRegion();

  /// Initialize the LiveThru pressure set based on the untied defs found in
  /// RPTracker.
  void initLiveThru(const RegPressureTracker &RPTracker);

  /// Copy an existing live thru pressure result.
  void initLiveThru(ArrayRef<unsigned> PressureSet) {
    LiveThruPressure.assign(PressureSet.begin(), PressureSet.end());
  }

  ArrayRef<unsigned> getLiveThru() const { return LiveThruPressure; }

  /// Get the resulting register pressure over the traversed region.
  /// This result is complete if either advance() or recede() has returned true,
  /// or if closeRegion() was explicitly invoked.
  RegisterPressure &getPressure() { return P; }
  const RegisterPressure &getPressure() const { return P; }

  /// Get the register set pressure at the current position, which may be less
  /// than the pressure across the traversed region.
  std::vector<unsigned> &getRegSetPressureAtPos() { return CurrSetPressure; }

  void discoverLiveOut(unsigned Reg);
  void discoverLiveIn(unsigned Reg);

  bool isTopClosed() const;
  bool isBottomClosed() const;

  void closeTop();
  void closeBottom();

  /// Consider the pressure increase caused by traversing this instruction
  /// bottom-up. Find the pressure set with the most change beyond its pressure
  /// limit based on the tracker's current pressure, and record the number of
  /// excess register units of that pressure set introduced by this instruction.
  void getMaxUpwardPressureDelta(const MachineInstr *MI,
                                 PressureDiff *PDiff,
                                 RegPressureDelta &Delta,
                                 ArrayRef<PressureChange> CriticalPSets,
                                 ArrayRef<unsigned> MaxPressureLimit);

  void getUpwardPressureDelta(const MachineInstr *MI,
                              /*const*/ PressureDiff &PDiff,
                              RegPressureDelta &Delta,
                              ArrayRef<PressureChange> CriticalPSets,
                              ArrayRef<unsigned> MaxPressureLimit) const;

  /// Consider the pressure increase caused by traversing this instruction
  /// top-down. Find the pressure set with the most change beyond its pressure
  /// limit based on the tracker's current pressure, and record the number of
  /// excess register units of that pressure set introduced by this instruction.
  void getMaxDownwardPressureDelta(const MachineInstr *MI,
                                   RegPressureDelta &Delta,
                                   ArrayRef<PressureChange> CriticalPSets,
                                   ArrayRef<unsigned> MaxPressureLimit);

  /// Find the pressure set with the most change beyond its pressure limit after
  /// traversing this instruction either upward or downward depending on the
  /// closed end of the current region.
  void getMaxPressureDelta(const MachineInstr *MI,
                           RegPressureDelta &Delta,
                           ArrayRef<PressureChange> CriticalPSets,
                           ArrayRef<unsigned> MaxPressureLimit) {
    if (isTopClosed())
      return getMaxDownwardPressureDelta(MI, Delta, CriticalPSets,
                                         MaxPressureLimit);

    assert(isBottomClosed() && "Uninitialized pressure tracker");
    return getMaxUpwardPressureDelta(MI, nullptr, Delta, CriticalPSets,
                                     MaxPressureLimit);
  }

  /// Get the pressure of each PSet after traversing this instruction bottom-up.
  void getUpwardPressure(const MachineInstr *MI,
                         std::vector<unsigned> &PressureResult,
                         std::vector<unsigned> &MaxPressureResult);

  /// Get the pressure of each PSet after traversing this instruction top-down.
  void getDownwardPressure(const MachineInstr *MI,
                           std::vector<unsigned> &PressureResult,
                           std::vector<unsigned> &MaxPressureResult);

  void getPressureAfterInst(const MachineInstr *MI,
                            std::vector<unsigned> &PressureResult,
                            std::vector<unsigned> &MaxPressureResult) {
    if (isTopClosed())
      return getUpwardPressure(MI, PressureResult, MaxPressureResult);

    assert(isBottomClosed() && "Uninitialized pressure tracker");
    return getDownwardPressure(MI, PressureResult, MaxPressureResult);
  }

  bool hasUntiedDef(unsigned VirtReg) const {
    return UntiedDefs.count(VirtReg);
  }

  void dump() const;

protected:
  const LiveRange *getLiveRange(unsigned Reg) const;

  void increaseRegPressure(ArrayRef<unsigned> Regs);
  void decreaseRegPressure(ArrayRef<unsigned> Regs);

  void bumpUpwardPressure(const MachineInstr *MI);
  void bumpDownwardPressure(const MachineInstr *MI);
};

void dumpRegSetPressure(ArrayRef<unsigned> SetPressure,
                        const TargetRegisterInfo *TRI);
} // end namespace llvm

#endif
