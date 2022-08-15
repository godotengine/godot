//===-- LiveIntervalAnalysis.h - Live Interval Analysis ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the LiveInterval analysis pass.  Given some numbering of
// each the machine instructions (in this implemention depth-first order) an
// interval [i, j) is said to be a live interval for register v if there is no
// instruction with number j' > j such that v is live at j' and there is no
// instruction with number i' < i such that v is live at i'. In this
// implementation intervals can have holes, i.e. an interval might look like
// [1,20), [50,65), [1000,1001).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_LIVEINTERVALANALYSIS_H
#define LLVM_CODEGEN_LIVEINTERVALANALYSIS_H

#include "llvm/ADT/IndexedMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/LiveInterval.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include <cmath>
#include <iterator>

namespace llvm {

extern cl::opt<bool> UseSegmentSetForPhysRegs;

  class AliasAnalysis;
  class BitVector;
  class BlockFrequency;
  class LiveRangeCalc;
  class LiveVariables;
  class MachineDominatorTree;
  class MachineLoopInfo;
  class TargetRegisterInfo;
  class MachineRegisterInfo;
  class TargetInstrInfo;
  class TargetRegisterClass;
  class VirtRegMap;
  class MachineBlockFrequencyInfo;

  class LiveIntervals : public MachineFunctionPass {
    MachineFunction* MF;
    MachineRegisterInfo* MRI;
    const TargetRegisterInfo* TRI;
    const TargetInstrInfo* TII;
    AliasAnalysis *AA;
    SlotIndexes* Indexes;
    MachineDominatorTree *DomTree;
    LiveRangeCalc *LRCalc;

    /// Special pool allocator for VNInfo's (LiveInterval val#).
    ///
    VNInfo::Allocator VNInfoAllocator;

    /// Live interval pointers for all the virtual registers.
    IndexedMap<LiveInterval*, VirtReg2IndexFunctor> VirtRegIntervals;

    /// RegMaskSlots - Sorted list of instructions with register mask operands.
    /// Always use the 'r' slot, RegMasks are normal clobbers, not early
    /// clobbers.
    SmallVector<SlotIndex, 8> RegMaskSlots;

    /// RegMaskBits - This vector is parallel to RegMaskSlots, it holds a
    /// pointer to the corresponding register mask.  This pointer can be
    /// recomputed as:
    ///
    ///   MI = Indexes->getInstructionFromIndex(RegMaskSlot[N]);
    ///   unsigned OpNum = findRegMaskOperand(MI);
    ///   RegMaskBits[N] = MI->getOperand(OpNum).getRegMask();
    ///
    /// This is kept in a separate vector partly because some standard
    /// libraries don't support lower_bound() with mixed objects, partly to
    /// improve locality when searching in RegMaskSlots.
    /// Also see the comment in LiveInterval::find().
    SmallVector<const uint32_t*, 8> RegMaskBits;

    /// For each basic block number, keep (begin, size) pairs indexing into the
    /// RegMaskSlots and RegMaskBits arrays.
    /// Note that basic block numbers may not be layout contiguous, that's why
    /// we can't just keep track of the first register mask in each basic
    /// block.
    SmallVector<std::pair<unsigned, unsigned>, 8> RegMaskBlocks;

    /// Keeps a live range set for each register unit to track fixed physreg
    /// interference.
    SmallVector<LiveRange*, 0> RegUnitRanges;

  public:
    static char ID; // Pass identification, replacement for typeid
    LiveIntervals();
    ~LiveIntervals() override;

    // Calculate the spill weight to assign to a single instruction.
    static float getSpillWeight(bool isDef, bool isUse,
                                const MachineBlockFrequencyInfo *MBFI,
                                const MachineInstr *Instr);

    LiveInterval &getInterval(unsigned Reg) {
      if (hasInterval(Reg))
        return *VirtRegIntervals[Reg];
      else
        return createAndComputeVirtRegInterval(Reg);
    }

    const LiveInterval &getInterval(unsigned Reg) const {
      return const_cast<LiveIntervals*>(this)->getInterval(Reg);
    }

    bool hasInterval(unsigned Reg) const {
      return VirtRegIntervals.inBounds(Reg) && VirtRegIntervals[Reg];
    }

    // Interval creation.
    LiveInterval &createEmptyInterval(unsigned Reg) {
      assert(!hasInterval(Reg) && "Interval already exists!");
      VirtRegIntervals.grow(Reg);
      VirtRegIntervals[Reg] = createInterval(Reg);
      return *VirtRegIntervals[Reg];
    }

    LiveInterval &createAndComputeVirtRegInterval(unsigned Reg) {
      LiveInterval &LI = createEmptyInterval(Reg);
      computeVirtRegInterval(LI);
      return LI;
    }

    // Interval removal.
    void removeInterval(unsigned Reg) {
      delete VirtRegIntervals[Reg];
      VirtRegIntervals[Reg] = nullptr;
    }

    /// Given a register and an instruction, adds a live segment from that
    /// instruction to the end of its MBB.
    LiveInterval::Segment addSegmentToEndOfBlock(unsigned reg,
                                                 MachineInstr* startInst);

    /// shrinkToUses - After removing some uses of a register, shrink its live
    /// range to just the remaining uses. This method does not compute reaching
    /// defs for new uses, and it doesn't remove dead defs.
    /// Dead PHIDef values are marked as unused.
    /// New dead machine instructions are added to the dead vector.
    /// Return true if the interval may have been separated into multiple
    /// connected components.
    bool shrinkToUses(LiveInterval *li,
                      SmallVectorImpl<MachineInstr*> *dead = nullptr);

    /// Specialized version of
    /// shrinkToUses(LiveInterval *li, SmallVectorImpl<MachineInstr*> *dead)
    /// that works on a subregister live range and only looks at uses matching
    /// the lane mask of the subregister range.
    void shrinkToUses(LiveInterval::SubRange &SR, unsigned Reg);

    /// extendToIndices - Extend the live range of LI to reach all points in
    /// Indices. The points in the Indices array must be jointly dominated by
    /// existing defs in LI. PHI-defs are added as needed to maintain SSA form.
    ///
    /// If a SlotIndex in Indices is the end index of a basic block, LI will be
    /// extended to be live out of the basic block.
    ///
    /// See also LiveRangeCalc::extend().
    void extendToIndices(LiveRange &LR, ArrayRef<SlotIndex> Indices);


    /// If @p LR has a live value at @p Kill, prune its live range by removing
    /// any liveness reachable from Kill. Add live range end points to
    /// EndPoints such that extendToIndices(LI, EndPoints) will reconstruct the
    /// value's live range.
    ///
    /// Calling pruneValue() and extendToIndices() can be used to reconstruct
    /// SSA form after adding defs to a virtual register.
    void pruneValue(LiveRange &LR, SlotIndex Kill,
                    SmallVectorImpl<SlotIndex> *EndPoints);

    SlotIndexes *getSlotIndexes() const {
      return Indexes;
    }

    AliasAnalysis *getAliasAnalysis() const {
      return AA;
    }

    /// isNotInMIMap - returns true if the specified machine instr has been
    /// removed or was never entered in the map.
    bool isNotInMIMap(const MachineInstr* Instr) const {
      return !Indexes->hasIndex(Instr);
    }

    /// Returns the base index of the given instruction.
    SlotIndex getInstructionIndex(const MachineInstr *instr) const {
      return Indexes->getInstructionIndex(instr);
    }

    /// Returns the instruction associated with the given index.
    MachineInstr* getInstructionFromIndex(SlotIndex index) const {
      return Indexes->getInstructionFromIndex(index);
    }

    /// Return the first index in the given basic block.
    SlotIndex getMBBStartIdx(const MachineBasicBlock *mbb) const {
      return Indexes->getMBBStartIdx(mbb);
    }

    /// Return the last index in the given basic block.
    SlotIndex getMBBEndIdx(const MachineBasicBlock *mbb) const {
      return Indexes->getMBBEndIdx(mbb);
    }

    bool isLiveInToMBB(const LiveRange &LR,
                       const MachineBasicBlock *mbb) const {
      return LR.liveAt(getMBBStartIdx(mbb));
    }

    bool isLiveOutOfMBB(const LiveRange &LR,
                        const MachineBasicBlock *mbb) const {
      return LR.liveAt(getMBBEndIdx(mbb).getPrevSlot());
    }

    MachineBasicBlock* getMBBFromIndex(SlotIndex index) const {
      return Indexes->getMBBFromIndex(index);
    }

    void insertMBBInMaps(MachineBasicBlock *MBB) {
      Indexes->insertMBBInMaps(MBB);
      assert(unsigned(MBB->getNumber()) == RegMaskBlocks.size() &&
             "Blocks must be added in order.");
      RegMaskBlocks.push_back(std::make_pair(RegMaskSlots.size(), 0));
    }

    SlotIndex InsertMachineInstrInMaps(MachineInstr *MI) {
      return Indexes->insertMachineInstrInMaps(MI);
    }

    void InsertMachineInstrRangeInMaps(MachineBasicBlock::iterator B,
                                       MachineBasicBlock::iterator E) {
      for (MachineBasicBlock::iterator I = B; I != E; ++I)
        Indexes->insertMachineInstrInMaps(I);
    }

    void RemoveMachineInstrFromMaps(MachineInstr *MI) {
      Indexes->removeMachineInstrFromMaps(MI);
    }

    void ReplaceMachineInstrInMaps(MachineInstr *MI, MachineInstr *NewMI) {
      Indexes->replaceMachineInstrInMaps(MI, NewMI);
    }

    bool findLiveInMBBs(SlotIndex Start, SlotIndex End,
                        SmallVectorImpl<MachineBasicBlock*> &MBBs) const {
      return Indexes->findLiveInMBBs(Start, End, MBBs);
    }

    VNInfo::Allocator& getVNInfoAllocator() { return VNInfoAllocator; }

    void getAnalysisUsage(AnalysisUsage &AU) const override;
    void releaseMemory() override;

    /// runOnMachineFunction - pass entry point
    bool runOnMachineFunction(MachineFunction&) override;

    /// print - Implement the dump method.
    void print(raw_ostream &O, const Module* = nullptr) const override;

    /// intervalIsInOneMBB - If LI is confined to a single basic block, return
    /// a pointer to that block.  If LI is live in to or out of any block,
    /// return NULL.
    MachineBasicBlock *intervalIsInOneMBB(const LiveInterval &LI) const;

    /// Returns true if VNI is killed by any PHI-def values in LI.
    /// This may conservatively return true to avoid expensive computations.
    bool hasPHIKill(const LiveInterval &LI, const VNInfo *VNI) const;

    /// addKillFlags - Add kill flags to any instruction that kills a virtual
    /// register.
    void addKillFlags(const VirtRegMap*);

    /// handleMove - call this method to notify LiveIntervals that
    /// instruction 'mi' has been moved within a basic block. This will update
    /// the live intervals for all operands of mi. Moves between basic blocks
    /// are not supported.
    ///
    /// \param UpdateFlags Update live intervals for nonallocatable physregs.
    void handleMove(MachineInstr* MI, bool UpdateFlags = false);

    /// moveIntoBundle - Update intervals for operands of MI so that they
    /// begin/end on the SlotIndex for BundleStart.
    ///
    /// \param UpdateFlags Update live intervals for nonallocatable physregs.
    ///
    /// Requires MI and BundleStart to have SlotIndexes, and assumes
    /// existing liveness is accurate. BundleStart should be the first
    /// instruction in the Bundle.
    void handleMoveIntoBundle(MachineInstr* MI, MachineInstr* BundleStart,
                              bool UpdateFlags = false);

    /// repairIntervalsInRange - Update live intervals for instructions in a
    /// range of iterators. It is intended for use after target hooks that may
    /// insert or remove instructions, and is only efficient for a small number
    /// of instructions.
    ///
    /// OrigRegs is a vector of registers that were originally used by the
    /// instructions in the range between the two iterators.
    ///
    /// Currently, the only only changes that are supported are simple removal
    /// and addition of uses.
    void repairIntervalsInRange(MachineBasicBlock *MBB,
                                MachineBasicBlock::iterator Begin,
                                MachineBasicBlock::iterator End,
                                ArrayRef<unsigned> OrigRegs);

    // Register mask functions.
    //
    // Machine instructions may use a register mask operand to indicate that a
    // large number of registers are clobbered by the instruction.  This is
    // typically used for calls.
    //
    // For compile time performance reasons, these clobbers are not recorded in
    // the live intervals for individual physical registers.  Instead,
    // LiveIntervalAnalysis maintains a sorted list of instructions with
    // register mask operands.

    /// getRegMaskSlots - Returns a sorted array of slot indices of all
    /// instructions with register mask operands.
    ArrayRef<SlotIndex> getRegMaskSlots() const { return RegMaskSlots; }

    /// getRegMaskSlotsInBlock - Returns a sorted array of slot indices of all
    /// instructions with register mask operands in the basic block numbered
    /// MBBNum.
    ArrayRef<SlotIndex> getRegMaskSlotsInBlock(unsigned MBBNum) const {
      std::pair<unsigned, unsigned> P = RegMaskBlocks[MBBNum];
      return getRegMaskSlots().slice(P.first, P.second);
    }

    /// getRegMaskBits() - Returns an array of register mask pointers
    /// corresponding to getRegMaskSlots().
    ArrayRef<const uint32_t*> getRegMaskBits() const { return RegMaskBits; }

    /// getRegMaskBitsInBlock - Returns an array of mask pointers corresponding
    /// to getRegMaskSlotsInBlock(MBBNum).
    ArrayRef<const uint32_t*> getRegMaskBitsInBlock(unsigned MBBNum) const {
      std::pair<unsigned, unsigned> P = RegMaskBlocks[MBBNum];
      return getRegMaskBits().slice(P.first, P.second);
    }

    /// checkRegMaskInterference - Test if LI is live across any register mask
    /// instructions, and compute a bit mask of physical registers that are not
    /// clobbered by any of them.
    ///
    /// Returns false if LI doesn't cross any register mask instructions. In
    /// that case, the bit vector is not filled in.
    bool checkRegMaskInterference(LiveInterval &LI,
                                  BitVector &UsableRegs);

    // Register unit functions.
    //
    // Fixed interference occurs when MachineInstrs use physregs directly
    // instead of virtual registers. This typically happens when passing
    // arguments to a function call, or when instructions require operands in
    // fixed registers.
    //
    // Each physreg has one or more register units, see MCRegisterInfo. We
    // track liveness per register unit to handle aliasing registers more
    // efficiently.

    /// getRegUnit - Return the live range for Unit.
    /// It will be computed if it doesn't exist.
    LiveRange &getRegUnit(unsigned Unit) {
      LiveRange *LR = RegUnitRanges[Unit];
      if (!LR) {
        // Compute missing ranges on demand.
        // Use segment set to speed-up initial computation of the live range.
        RegUnitRanges[Unit] = LR = new LiveRange(UseSegmentSetForPhysRegs);
        computeRegUnitRange(*LR, Unit);
      }
      return *LR;
    }

    /// getCachedRegUnit - Return the live range for Unit if it has already
    /// been computed, or NULL if it hasn't been computed yet.
    LiveRange *getCachedRegUnit(unsigned Unit) {
      return RegUnitRanges[Unit];
    }

    const LiveRange *getCachedRegUnit(unsigned Unit) const {
      return RegUnitRanges[Unit];
    }

    /// Remove value numbers and related live segments starting at position
    /// @p Pos that are part of any liverange of physical register @p Reg or one
    /// of its subregisters.
    void removePhysRegDefAt(unsigned Reg, SlotIndex Pos);

    /// Remove value number and related live segments of @p LI and its subranges
    /// that start at position @p Pos.
    void removeVRegDefAt(LiveInterval &LI, SlotIndex Pos);

  private:
    /// Compute live intervals for all virtual registers.
    void computeVirtRegs();

    /// Compute RegMaskSlots and RegMaskBits.
    void computeRegMasks();

    /// Walk the values in @p LI and check for dead values:
    /// - Dead PHIDef values are marked as unused.
    /// - Dead operands are marked as such.
    /// - Completely dead machine instructions are added to the @p dead vector
    ///   if it is not nullptr.
    /// Returns true if any PHI value numbers have been removed which may
    /// have separated the interval into multiple connected components.
    bool computeDeadValues(LiveInterval &LI,
                           SmallVectorImpl<MachineInstr*> *dead);

    static LiveInterval* createInterval(unsigned Reg);

    void printInstrs(raw_ostream &O) const;
    void dumpInstrs() const;

    void computeLiveInRegUnits();
    void computeRegUnitRange(LiveRange&, unsigned Unit);
    void computeVirtRegInterval(LiveInterval&);


    /// Helper function for repairIntervalsInRange(), walks backwards and
    /// creates/modifies live segments in @p LR to match the operands found.
    /// Only full operands or operands with subregisters matching @p LaneMask
    /// are considered.
    void repairOldRegInRange(MachineBasicBlock::iterator Begin,
                             MachineBasicBlock::iterator End,
                             const SlotIndex endIdx, LiveRange &LR,
                             unsigned Reg, unsigned LaneMask = ~0u);

    class HMEditor;
  };
} // End llvm namespace

#endif
