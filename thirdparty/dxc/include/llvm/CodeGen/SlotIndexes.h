//===- llvm/CodeGen/SlotIndexes.h - Slot indexes representation -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements SlotIndex and related classes. The purpose of SlotIndex
// is to describe a position at which a register can become live, or cease to
// be live.
//
// SlotIndex is mostly a proxy for entries of the SlotIndexList, a class which
// is held is LiveIntervals and provides the real numbering. This allows
// LiveIntervals to perform largely transparent renumbering.
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_SLOTINDEXES_H
#define LLVM_CODEGEN_SLOTINDEXES_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/IntervalMap.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/ilist.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBundle.h"
#include "llvm/Support/Allocator.h"

namespace llvm {

  /// This class represents an entry in the slot index list held in the
  /// SlotIndexes pass. It should not be used directly. See the
  /// SlotIndex & SlotIndexes classes for the public interface to this
  /// information.
  class IndexListEntry : public ilist_node<IndexListEntry> {
    MachineInstr *mi;
    unsigned index;

  public:

    IndexListEntry(MachineInstr *mi, unsigned index) : mi(mi), index(index) {}

    MachineInstr* getInstr() const { return mi; }
    void setInstr(MachineInstr *mi) {
      this->mi = mi;
    }

    unsigned getIndex() const { return index; }
    void setIndex(unsigned index) {
      this->index = index;
    }

#ifdef EXPENSIVE_CHECKS
    // When EXPENSIVE_CHECKS is defined, "erased" index list entries will
    // actually be moved to a "graveyard" list, and have their pointers
    // poisoned, so that dangling SlotIndex access can be reliably detected.
    void setPoison() {
      intptr_t tmp = reinterpret_cast<intptr_t>(mi);
      assert(((tmp & 0x1) == 0x0) && "Pointer already poisoned?");
      tmp |= 0x1;
      mi = reinterpret_cast<MachineInstr*>(tmp);
    }

    bool isPoisoned() const { return (reinterpret_cast<intptr_t>(mi) & 0x1) == 0x1; }
#endif // EXPENSIVE_CHECKS

  };

  template <>
  struct ilist_traits<IndexListEntry> : public ilist_default_traits<IndexListEntry> {
  private:
    mutable ilist_half_node<IndexListEntry> Sentinel;
  public:
    IndexListEntry *createSentinel() const {
      return static_cast<IndexListEntry*>(&Sentinel);
    }
    void destroySentinel(IndexListEntry *) const {}

    IndexListEntry *provideInitialHead() const { return createSentinel(); }
    IndexListEntry *ensureHead(IndexListEntry*) const { return createSentinel(); }
    static void noteHead(IndexListEntry*, IndexListEntry*) {}
    void deleteNode(IndexListEntry *N) {}

  private:
    void createNode(const IndexListEntry &);
  };

  /// SlotIndex - An opaque wrapper around machine indexes.
  class SlotIndex {
    friend class SlotIndexes;

    enum Slot {
      /// Basic block boundary.  Used for live ranges entering and leaving a
      /// block without being live in the layout neighbor.  Also used as the
      /// def slot of PHI-defs.
      Slot_Block,

      /// Early-clobber register use/def slot.  A live range defined at
      /// Slot_EarlyCLobber interferes with normal live ranges killed at
      /// Slot_Register.  Also used as the kill slot for live ranges tied to an
      /// early-clobber def.
      Slot_EarlyClobber,

      /// Normal register use/def slot.  Normal instructions kill and define
      /// register live ranges at this slot.
      Slot_Register,

      /// Dead def kill point.  Kill slot for a live range that is defined by
      /// the same instruction (Slot_Register or Slot_EarlyClobber), but isn't
      /// used anywhere.
      Slot_Dead,

      Slot_Count
    };

    PointerIntPair<IndexListEntry*, 2, unsigned> lie;

    SlotIndex(IndexListEntry *entry, unsigned slot)
      : lie(entry, slot) {}

    IndexListEntry* listEntry() const {
      assert(isValid() && "Attempt to compare reserved index.");
#ifdef EXPENSIVE_CHECKS
      assert(!lie.getPointer()->isPoisoned() &&
             "Attempt to access deleted list-entry.");
#endif // EXPENSIVE_CHECKS
      return lie.getPointer();
    }

    unsigned getIndex() const {
      return listEntry()->getIndex() | getSlot();
    }

    /// Returns the slot for this SlotIndex.
    Slot getSlot() const {
      return static_cast<Slot>(lie.getInt());
    }

  public:
    enum {
      /// The default distance between instructions as returned by distance().
      /// This may vary as instructions are inserted and removed.
      InstrDist = 4 * Slot_Count
    };

    /// Construct an invalid index.
    SlotIndex() : lie(nullptr, 0) {}

    // Construct a new slot index from the given one, and set the slot.
    SlotIndex(const SlotIndex &li, Slot s) : lie(li.listEntry(), unsigned(s)) {
      assert(lie.getPointer() != nullptr &&
             "Attempt to construct index with 0 pointer.");
    }

    /// Returns true if this is a valid index. Invalid indicies do
    /// not point into an index table, and cannot be compared.
    bool isValid() const {
      return lie.getPointer();
    }

    /// Return true for a valid index.
    explicit operator bool() const { return isValid(); }

    /// Print this index to the given raw_ostream.
    void print(raw_ostream &os) const;

    /// Dump this index to stderr.
    void dump() const;

    /// Compare two SlotIndex objects for equality.
    bool operator==(SlotIndex other) const {
      return lie == other.lie;
    }
    /// Compare two SlotIndex objects for inequality.
    bool operator!=(SlotIndex other) const {
      return lie != other.lie;
    }

    /// Compare two SlotIndex objects. Return true if the first index
    /// is strictly lower than the second.
    bool operator<(SlotIndex other) const {
      return getIndex() < other.getIndex();
    }
    /// Compare two SlotIndex objects. Return true if the first index
    /// is lower than, or equal to, the second.
    bool operator<=(SlotIndex other) const {
      return getIndex() <= other.getIndex();
    }

    /// Compare two SlotIndex objects. Return true if the first index
    /// is greater than the second.
    bool operator>(SlotIndex other) const {
      return getIndex() > other.getIndex();
    }

    /// Compare two SlotIndex objects. Return true if the first index
    /// is greater than, or equal to, the second.
    bool operator>=(SlotIndex other) const {
      return getIndex() >= other.getIndex();
    }

    /// isSameInstr - Return true if A and B refer to the same instruction.
    static bool isSameInstr(SlotIndex A, SlotIndex B) {
      return A.lie.getPointer() == B.lie.getPointer();
    }

    /// isEarlierInstr - Return true if A refers to an instruction earlier than
    /// B. This is equivalent to A < B && !isSameInstr(A, B).
    static bool isEarlierInstr(SlotIndex A, SlotIndex B) {
      return A.listEntry()->getIndex() < B.listEntry()->getIndex();
    }

    /// Return the distance from this index to the given one.
    int distance(SlotIndex other) const {
      return other.getIndex() - getIndex();
    }

    /// Return the scaled distance from this index to the given one, where all
    /// slots on the same instruction have zero distance.
    int getInstrDistance(SlotIndex other) const {
      return (other.listEntry()->getIndex() - listEntry()->getIndex())
        / Slot_Count;
    }

    /// isBlock - Returns true if this is a block boundary slot.
    bool isBlock() const { return getSlot() == Slot_Block; }

    /// isEarlyClobber - Returns true if this is an early-clobber slot.
    bool isEarlyClobber() const { return getSlot() == Slot_EarlyClobber; }

    /// isRegister - Returns true if this is a normal register use/def slot.
    /// Note that early-clobber slots may also be used for uses and defs.
    bool isRegister() const { return getSlot() == Slot_Register; }

    /// isDead - Returns true if this is a dead def kill slot.
    bool isDead() const { return getSlot() == Slot_Dead; }

    /// Returns the base index for associated with this index. The base index
    /// is the one associated with the Slot_Block slot for the instruction
    /// pointed to by this index.
    SlotIndex getBaseIndex() const {
      return SlotIndex(listEntry(), Slot_Block);
    }

    /// Returns the boundary index for associated with this index. The boundary
    /// index is the one associated with the Slot_Block slot for the instruction
    /// pointed to by this index.
    SlotIndex getBoundaryIndex() const {
      return SlotIndex(listEntry(), Slot_Dead);
    }

    /// Returns the register use/def slot in the current instruction for a
    /// normal or early-clobber def.
    SlotIndex getRegSlot(bool EC = false) const {
      return SlotIndex(listEntry(), EC ? Slot_EarlyClobber : Slot_Register);
    }

    /// Returns the dead def kill slot for the current instruction.
    SlotIndex getDeadSlot() const {
      return SlotIndex(listEntry(), Slot_Dead);
    }

    /// Returns the next slot in the index list. This could be either the
    /// next slot for the instruction pointed to by this index or, if this
    /// index is a STORE, the first slot for the next instruction.
    /// WARNING: This method is considerably more expensive than the methods
    /// that return specific slots (getUseIndex(), etc). If you can - please
    /// use one of those methods.
    SlotIndex getNextSlot() const {
      Slot s = getSlot();
      if (s == Slot_Dead) {
        return SlotIndex(listEntry()->getNextNode(), Slot_Block);
      }
      return SlotIndex(listEntry(), s + 1);
    }

    /// Returns the next index. This is the index corresponding to the this
    /// index's slot, but for the next instruction.
    SlotIndex getNextIndex() const {
      return SlotIndex(listEntry()->getNextNode(), getSlot());
    }

    /// Returns the previous slot in the index list. This could be either the
    /// previous slot for the instruction pointed to by this index or, if this
    /// index is a Slot_Block, the last slot for the previous instruction.
    /// WARNING: This method is considerably more expensive than the methods
    /// that return specific slots (getUseIndex(), etc). If you can - please
    /// use one of those methods.
    SlotIndex getPrevSlot() const {
      Slot s = getSlot();
      if (s == Slot_Block) {
        return SlotIndex(listEntry()->getPrevNode(), Slot_Dead);
      }
      return SlotIndex(listEntry(), s - 1);
    }

    /// Returns the previous index. This is the index corresponding to this
    /// index's slot, but for the previous instruction.
    SlotIndex getPrevIndex() const {
      return SlotIndex(listEntry()->getPrevNode(), getSlot());
    }

  };

  template <> struct isPodLike<SlotIndex> { static const bool value = true; };

  inline raw_ostream& operator<<(raw_ostream &os, SlotIndex li) {
    li.print(os);
    return os;
  }

  typedef std::pair<SlotIndex, MachineBasicBlock*> IdxMBBPair;

  inline bool operator<(SlotIndex V, const IdxMBBPair &IM) {
    return V < IM.first;
  }

  inline bool operator<(const IdxMBBPair &IM, SlotIndex V) {
    return IM.first < V;
  }

  struct Idx2MBBCompare {
    bool operator()(const IdxMBBPair &LHS, const IdxMBBPair &RHS) const {
      return LHS.first < RHS.first;
    }
  };

  /// SlotIndexes pass.
  ///
  /// This pass assigns indexes to each instruction.
  class SlotIndexes : public MachineFunctionPass {
  private:

    typedef ilist<IndexListEntry> IndexList;
    IndexList indexList;

#ifdef EXPENSIVE_CHECKS
    IndexList graveyardList;
#endif // EXPENSIVE_CHECKS

    MachineFunction *mf;

    typedef DenseMap<const MachineInstr*, SlotIndex> Mi2IndexMap;
    Mi2IndexMap mi2iMap;

    /// MBBRanges - Map MBB number to (start, stop) indexes.
    SmallVector<std::pair<SlotIndex, SlotIndex>, 8> MBBRanges;

    /// Idx2MBBMap - Sorted list of pairs of index of first instruction
    /// and MBB id.
    SmallVector<IdxMBBPair, 8> idx2MBBMap;

    // IndexListEntry allocator.
    BumpPtrAllocator ileAllocator;

    IndexListEntry* createEntry(MachineInstr *mi, unsigned index) {
      IndexListEntry *entry =
        static_cast<IndexListEntry*>(
          ileAllocator.Allocate(sizeof(IndexListEntry),
          alignOf<IndexListEntry>()));

      new (entry) IndexListEntry(mi, index);

      return entry;
    }

    /// Renumber locally after inserting curItr.
    void renumberIndexes(IndexList::iterator curItr);

  public:
    static char ID;

    SlotIndexes() : MachineFunctionPass(ID) {
      initializeSlotIndexesPass(*PassRegistry::getPassRegistry());
    }

    void getAnalysisUsage(AnalysisUsage &au) const override;
    void releaseMemory() override;

    bool runOnMachineFunction(MachineFunction &fn) override;

    /// Dump the indexes.
    void dump() const;

    /// Renumber the index list, providing space for new instructions.
    void renumberIndexes();

    /// Repair indexes after adding and removing instructions.
    void repairIndexesInRange(MachineBasicBlock *MBB,
                              MachineBasicBlock::iterator Begin,
                              MachineBasicBlock::iterator End);

    /// Returns the zero index for this analysis.
    SlotIndex getZeroIndex() {
      assert(indexList.front().getIndex() == 0 && "First index is not 0?");
      return SlotIndex(&indexList.front(), 0);
    }

    /// Returns the base index of the last slot in this analysis.
    SlotIndex getLastIndex() {
      return SlotIndex(&indexList.back(), 0);
    }

    /// Returns true if the given machine instr is mapped to an index,
    /// otherwise returns false.
    bool hasIndex(const MachineInstr *instr) const {
      return mi2iMap.count(instr);
    }

    /// Returns the base index for the given instruction.
    SlotIndex getInstructionIndex(const MachineInstr *MI) const {
      // Instructions inside a bundle have the same number as the bundle itself.
      Mi2IndexMap::const_iterator itr = mi2iMap.find(getBundleStart(MI));
      assert(itr != mi2iMap.end() && "Instruction not found in maps.");
      return itr->second;
    }

    /// Returns the instruction for the given index, or null if the given
    /// index has no instruction associated with it.
    MachineInstr* getInstructionFromIndex(SlotIndex index) const {
      return index.isValid() ? index.listEntry()->getInstr() : nullptr;
    }

    /// Returns the next non-null index, if one exists.
    /// Otherwise returns getLastIndex().
    SlotIndex getNextNonNullIndex(SlotIndex Index) {
      IndexList::iterator I = Index.listEntry();
      IndexList::iterator E = indexList.end();
      while (++I != E)
        if (I->getInstr())
          return SlotIndex(I, Index.getSlot());
      // We reached the end of the function.
      return getLastIndex();
    }

    /// getIndexBefore - Returns the index of the last indexed instruction
    /// before MI, or the start index of its basic block.
    /// MI is not required to have an index.
    SlotIndex getIndexBefore(const MachineInstr *MI) const {
      const MachineBasicBlock *MBB = MI->getParent();
      assert(MBB && "MI must be inserted inna basic block");
      MachineBasicBlock::const_iterator I = MI, B = MBB->begin();
      for (;;) {
        if (I == B)
          return getMBBStartIdx(MBB);
        --I;
        Mi2IndexMap::const_iterator MapItr = mi2iMap.find(I);
        if (MapItr != mi2iMap.end())
          return MapItr->second;
      }
    }

    /// getIndexAfter - Returns the index of the first indexed instruction
    /// after MI, or the end index of its basic block.
    /// MI is not required to have an index.
    SlotIndex getIndexAfter(const MachineInstr *MI) const {
      const MachineBasicBlock *MBB = MI->getParent();
      assert(MBB && "MI must be inserted inna basic block");
      MachineBasicBlock::const_iterator I = MI, E = MBB->end();
      for (;;) {
        ++I;
        if (I == E)
          return getMBBEndIdx(MBB);
        Mi2IndexMap::const_iterator MapItr = mi2iMap.find(I);
        if (MapItr != mi2iMap.end())
          return MapItr->second;
      }
    }

    /// Return the (start,end) range of the given basic block number.
    const std::pair<SlotIndex, SlotIndex> &
    getMBBRange(unsigned Num) const {
      return MBBRanges[Num];
    }

    /// Return the (start,end) range of the given basic block.
    const std::pair<SlotIndex, SlotIndex> &
    getMBBRange(const MachineBasicBlock *MBB) const {
      return getMBBRange(MBB->getNumber());
    }

    /// Returns the first index in the given basic block number.
    SlotIndex getMBBStartIdx(unsigned Num) const {
      return getMBBRange(Num).first;
    }

    /// Returns the first index in the given basic block.
    SlotIndex getMBBStartIdx(const MachineBasicBlock *mbb) const {
      return getMBBRange(mbb).first;
    }

    /// Returns the last index in the given basic block number.
    SlotIndex getMBBEndIdx(unsigned Num) const {
      return getMBBRange(Num).second;
    }

    /// Returns the last index in the given basic block.
    SlotIndex getMBBEndIdx(const MachineBasicBlock *mbb) const {
      return getMBBRange(mbb).second;
    }

    /// Returns the basic block which the given index falls in.
    MachineBasicBlock* getMBBFromIndex(SlotIndex index) const {
      if (MachineInstr *MI = getInstructionFromIndex(index))
        return MI->getParent();
      SmallVectorImpl<IdxMBBPair>::const_iterator I =
        std::lower_bound(idx2MBBMap.begin(), idx2MBBMap.end(), index);
      // Take the pair containing the index
      SmallVectorImpl<IdxMBBPair>::const_iterator J =
        ((I != idx2MBBMap.end() && I->first > index) ||
         (I == idx2MBBMap.end() && idx2MBBMap.size()>0)) ? (I-1): I;

      assert(J != idx2MBBMap.end() && J->first <= index &&
             index < getMBBEndIdx(J->second) &&
             "index does not correspond to an MBB");
      return J->second;
    }

    bool findLiveInMBBs(SlotIndex start, SlotIndex end,
                        SmallVectorImpl<MachineBasicBlock*> &mbbs) const {
      SmallVectorImpl<IdxMBBPair>::const_iterator itr =
        std::lower_bound(idx2MBBMap.begin(), idx2MBBMap.end(), start);
      bool resVal = false;

      while (itr != idx2MBBMap.end()) {
        if (itr->first >= end)
          break;
        mbbs.push_back(itr->second);
        resVal = true;
        ++itr;
      }
      return resVal;
    }

    /// Returns the MBB covering the given range, or null if the range covers
    /// more than one basic block.
    MachineBasicBlock* getMBBCoveringRange(SlotIndex start, SlotIndex end) const {

      assert(start < end && "Backwards ranges not allowed.");

      SmallVectorImpl<IdxMBBPair>::const_iterator itr =
        std::lower_bound(idx2MBBMap.begin(), idx2MBBMap.end(), start);

      if (itr == idx2MBBMap.end()) {
        itr = std::prev(itr);
        return itr->second;
      }

      // Check that we don't cross the boundary into this block.
      if (itr->first < end)
        return nullptr;

      itr = std::prev(itr);

      if (itr->first <= start)
        return itr->second;

      return nullptr;
    }

    /// Insert the given machine instruction into the mapping. Returns the
    /// assigned index.
    /// If Late is set and there are null indexes between mi's neighboring
    /// instructions, create the new index after the null indexes instead of
    /// before them.
    SlotIndex insertMachineInstrInMaps(MachineInstr *mi, bool Late = false) {
      assert(!mi->isInsideBundle() &&
             "Instructions inside bundles should use bundle start's slot.");
      assert(mi2iMap.find(mi) == mi2iMap.end() && "Instr already indexed.");
      // Numbering DBG_VALUE instructions could cause code generation to be
      // affected by debug information.
      assert(!mi->isDebugValue() && "Cannot number DBG_VALUE instructions.");

      assert(mi->getParent() != nullptr && "Instr must be added to function.");

      // Get the entries where mi should be inserted.
      IndexList::iterator prevItr, nextItr;
      if (Late) {
        // Insert mi's index immediately before the following instruction.
        nextItr = getIndexAfter(mi).listEntry();
        prevItr = std::prev(nextItr);
      } else {
        // Insert mi's index immediately after the preceding instruction.
        prevItr = getIndexBefore(mi).listEntry();
        nextItr = std::next(prevItr);
      }

      // Get a number for the new instr, or 0 if there's no room currently.
      // In the latter case we'll force a renumber later.
      unsigned dist = ((nextItr->getIndex() - prevItr->getIndex())/2) & ~3u;
      unsigned newNumber = prevItr->getIndex() + dist;

      // Insert a new list entry for mi.
      IndexList::iterator newItr =
        indexList.insert(nextItr, createEntry(mi, newNumber));

      // Renumber locally if we need to.
      if (dist == 0)
        renumberIndexes(newItr);

      SlotIndex newIndex(&*newItr, SlotIndex::Slot_Block);
      mi2iMap.insert(std::make_pair(mi, newIndex));
      return newIndex;
    }

    /// Remove the given machine instruction from the mapping.
    void removeMachineInstrFromMaps(MachineInstr *mi) {
      // remove index -> MachineInstr and
      // MachineInstr -> index mappings
      Mi2IndexMap::iterator mi2iItr = mi2iMap.find(mi);
      if (mi2iItr != mi2iMap.end()) {
        IndexListEntry *miEntry(mi2iItr->second.listEntry());
        assert(miEntry->getInstr() == mi && "Instruction indexes broken.");
        // FIXME: Eventually we want to actually delete these indexes.
        miEntry->setInstr(nullptr);
        mi2iMap.erase(mi2iItr);
      }
    }

    /// ReplaceMachineInstrInMaps - Replacing a machine instr with a new one in
    /// maps used by register allocator.
    void replaceMachineInstrInMaps(MachineInstr *mi, MachineInstr *newMI) {
      Mi2IndexMap::iterator mi2iItr = mi2iMap.find(mi);
      if (mi2iItr == mi2iMap.end())
        return;
      SlotIndex replaceBaseIndex = mi2iItr->second;
      IndexListEntry *miEntry(replaceBaseIndex.listEntry());
      assert(miEntry->getInstr() == mi &&
             "Mismatched instruction in index tables.");
      miEntry->setInstr(newMI);
      mi2iMap.erase(mi2iItr);
      mi2iMap.insert(std::make_pair(newMI, replaceBaseIndex));
    }

    /// Add the given MachineBasicBlock into the maps.
    void insertMBBInMaps(MachineBasicBlock *mbb) {
      MachineFunction::iterator nextMBB =
        std::next(MachineFunction::iterator(mbb));

      IndexListEntry *startEntry = nullptr;
      IndexListEntry *endEntry = nullptr;
      IndexList::iterator newItr;
      if (nextMBB == mbb->getParent()->end()) {
        startEntry = &indexList.back();
        endEntry = createEntry(nullptr, 0);
        newItr = indexList.insertAfter(startEntry, endEntry);
      } else {
        startEntry = createEntry(nullptr, 0);
        endEntry = getMBBStartIdx(nextMBB).listEntry();
        newItr = indexList.insert(endEntry, startEntry);
      }

      SlotIndex startIdx(startEntry, SlotIndex::Slot_Block);
      SlotIndex endIdx(endEntry, SlotIndex::Slot_Block);

      MachineFunction::iterator prevMBB(mbb);
      assert(prevMBB != mbb->getParent()->end() &&
             "Can't insert a new block at the beginning of a function.");
      --prevMBB;
      MBBRanges[prevMBB->getNumber()].second = startIdx;

      assert(unsigned(mbb->getNumber()) == MBBRanges.size() &&
             "Blocks must be added in order");
      MBBRanges.push_back(std::make_pair(startIdx, endIdx));
      idx2MBBMap.push_back(IdxMBBPair(startIdx, mbb));

      renumberIndexes(newItr);
      std::sort(idx2MBBMap.begin(), idx2MBBMap.end(), Idx2MBBCompare());
    }

    /// \brief Free the resources that were required to maintain a SlotIndex.
    ///
    /// Once an index is no longer needed (for instance because the instruction
    /// at that index has been moved), the resources required to maintain the
    /// index can be relinquished to reduce memory use and improve renumbering
    /// performance. Any remaining SlotIndex objects that point to the same
    /// index are left 'dangling' (much the same as a dangling pointer to a
    /// freed object) and should not be accessed, except to destruct them.
    ///
    /// Like dangling pointers, access to dangling SlotIndexes can cause
    /// painful-to-track-down bugs, especially if the memory for the index
    /// previously pointed to has been re-used. To detect dangling SlotIndex
    /// bugs, build with EXPENSIVE_CHECKS=1. This will cause "erased" indexes to
    /// be retained in a graveyard instead of being freed. Operations on indexes
    /// in the graveyard will trigger an assertion.
    void eraseIndex(SlotIndex index) {
      IndexListEntry *entry = index.listEntry();
#ifdef EXPENSIVE_CHECKS
      indexList.remove(entry);
      graveyardList.push_back(entry);
      entry->setPoison();
#else
      indexList.erase(entry);
#endif
    }

  };


  // Specialize IntervalMapInfo for half-open slot index intervals.
  template <>
  struct IntervalMapInfo<SlotIndex> : IntervalMapHalfOpenInfo<SlotIndex> {
  };

}

#endif // LLVM_CODEGEN_SLOTINDEXES_H
