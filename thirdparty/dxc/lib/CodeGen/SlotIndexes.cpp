//===-- SlotIndexes.cpp - Slot Indexes Pass  ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"

using namespace llvm;

#define DEBUG_TYPE "slotindexes"

char SlotIndexes::ID = 0;
INITIALIZE_PASS(SlotIndexes, "slotindexes",
                "Slot index numbering", false, false)

STATISTIC(NumLocalRenum,  "Number of local renumberings");
STATISTIC(NumGlobalRenum, "Number of global renumberings");

void SlotIndexes::getAnalysisUsage(AnalysisUsage &au) const {
  au.setPreservesAll();
  MachineFunctionPass::getAnalysisUsage(au);
}

void SlotIndexes::releaseMemory() {
  mi2iMap.clear();
  MBBRanges.clear();
  idx2MBBMap.clear();
  indexList.clear();
  ileAllocator.Reset();
}

bool SlotIndexes::runOnMachineFunction(MachineFunction &fn) {

  // Compute numbering as follows:
  // Grab an iterator to the start of the index list.
  // Iterate over all MBBs, and within each MBB all MIs, keeping the MI
  // iterator in lock-step (though skipping it over indexes which have
  // null pointers in the instruction field).
  // At each iteration assert that the instruction pointed to in the index
  // is the same one pointed to by the MI iterator. This

  // FIXME: This can be simplified. The mi2iMap_, Idx2MBBMap, etc. should
  // only need to be set up once after the first numbering is computed.

  mf = &fn;

  // Check that the list contains only the sentinal.
  assert(indexList.empty() && "Index list non-empty at initial numbering?");
  assert(idx2MBBMap.empty() &&
         "Index -> MBB mapping non-empty at initial numbering?");
  assert(MBBRanges.empty() &&
         "MBB -> Index mapping non-empty at initial numbering?");
  assert(mi2iMap.empty() &&
         "MachineInstr -> Index mapping non-empty at initial numbering?");

  unsigned index = 0;
  MBBRanges.resize(mf->getNumBlockIDs());
  idx2MBBMap.reserve(mf->size());

  indexList.push_back(createEntry(nullptr, index));

  // Iterate over the function.
  for (MachineFunction::iterator mbbItr = mf->begin(), mbbEnd = mf->end();
       mbbItr != mbbEnd; ++mbbItr) {
    MachineBasicBlock *mbb = &*mbbItr;

    // Insert an index for the MBB start.
    SlotIndex blockStartIndex(&indexList.back(), SlotIndex::Slot_Block);

    for (MachineBasicBlock::iterator miItr = mbb->begin(), miEnd = mbb->end();
         miItr != miEnd; ++miItr) {
      MachineInstr *mi = miItr;
      if (mi->isDebugValue())
        continue;

      // Insert a store index for the instr.
      indexList.push_back(createEntry(mi, index += SlotIndex::InstrDist));

      // Save this base index in the maps.
      mi2iMap.insert(std::make_pair(mi, SlotIndex(&indexList.back(),
                                                  SlotIndex::Slot_Block)));
    }

    // We insert one blank instructions between basic blocks.
    indexList.push_back(createEntry(nullptr, index += SlotIndex::InstrDist));

    MBBRanges[mbb->getNumber()].first = blockStartIndex;
    MBBRanges[mbb->getNumber()].second = SlotIndex(&indexList.back(),
                                                   SlotIndex::Slot_Block);
    idx2MBBMap.push_back(IdxMBBPair(blockStartIndex, mbb));
  }

  // Sort the Idx2MBBMap
  std::sort(idx2MBBMap.begin(), idx2MBBMap.end(), Idx2MBBCompare());

  DEBUG(mf->print(dbgs(), this));

  // And we're done!
  return false;
}

void SlotIndexes::renumberIndexes() {
  // Renumber updates the index of every element of the index list.
  DEBUG(dbgs() << "\n*** Renumbering SlotIndexes ***\n");
  ++NumGlobalRenum;

  unsigned index = 0;

  for (IndexList::iterator I = indexList.begin(), E = indexList.end();
       I != E; ++I) {
    I->setIndex(index);
    index += SlotIndex::InstrDist;
  }
}

// Renumber indexes locally after curItr was inserted, but failed to get a new
// index.
void SlotIndexes::renumberIndexes(IndexList::iterator curItr) {
  // Number indexes with half the default spacing so we can catch up quickly.
  const unsigned Space = SlotIndex::InstrDist/2;
  static_assert((Space & 3) == 0, "InstrDist must be a multiple of 2*NUM");

  IndexList::iterator startItr = std::prev(curItr);
  unsigned index = startItr->getIndex();
  do {
    curItr->setIndex(index += Space);
    ++curItr;
    // If the next index is bigger, we have caught up.
  } while (curItr != indexList.end() && curItr->getIndex() <= index);

  DEBUG(dbgs() << "\n*** Renumbered SlotIndexes " << startItr->getIndex() << '-'
               << index << " ***\n");
  ++NumLocalRenum;
}

// Repair indexes after adding and removing instructions.
void SlotIndexes::repairIndexesInRange(MachineBasicBlock *MBB,
                                       MachineBasicBlock::iterator Begin,
                                       MachineBasicBlock::iterator End) {
  // FIXME: Is this really necessary? The only caller repairIntervalsForRange()
  // does the same thing.
  // Find anchor points, which are at the beginning/end of blocks or at
  // instructions that already have indexes.
  while (Begin != MBB->begin() && !hasIndex(Begin))
    --Begin;
  while (End != MBB->end() && !hasIndex(End))
    ++End;

  bool includeStart = (Begin == MBB->begin());
  SlotIndex startIdx;
  if (includeStart)
    startIdx = getMBBStartIdx(MBB);
  else
    startIdx = getInstructionIndex(Begin);

  SlotIndex endIdx;
  if (End == MBB->end())
    endIdx = getMBBEndIdx(MBB);
  else
    endIdx = getInstructionIndex(End);

  // FIXME: Conceptually, this code is implementing an iterator on MBB that
  // optionally includes an additional position prior to MBB->begin(), indicated
  // by the includeStart flag. This is done so that we can iterate MIs in a MBB
  // in parallel with SlotIndexes, but there should be a better way to do this.
  IndexList::iterator ListB = startIdx.listEntry();
  IndexList::iterator ListI = endIdx.listEntry();
  MachineBasicBlock::iterator MBBI = End;
  bool pastStart = false;
  while (ListI != ListB || MBBI != Begin || (includeStart && !pastStart)) {
    assert(ListI->getIndex() >= startIdx.getIndex() &&
           (includeStart || !pastStart) &&
           "Decremented past the beginning of region to repair.");

    MachineInstr *SlotMI = ListI->getInstr();
    MachineInstr *MI = (MBBI != MBB->end() && !pastStart) ? MBBI : nullptr;
    bool MBBIAtBegin = MBBI == Begin && (!includeStart || pastStart);

    if (SlotMI == MI && !MBBIAtBegin) {
      --ListI;
      if (MBBI != Begin)
        --MBBI;
      else
        pastStart = true;
    } else if (MI && mi2iMap.find(MI) == mi2iMap.end()) {
      if (MBBI != Begin)
        --MBBI;
      else
        pastStart = true;
    } else {
      --ListI;
      if (SlotMI)
        removeMachineInstrFromMaps(SlotMI);
    }
  }

  // In theory this could be combined with the previous loop, but it is tricky
  // to update the IndexList while we are iterating it.
  for (MachineBasicBlock::iterator I = End; I != Begin;) {
    --I;
    MachineInstr *MI = I;
    if (!MI->isDebugValue() && mi2iMap.find(MI) == mi2iMap.end())
      insertMachineInstrInMaps(MI);
  }
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void SlotIndexes::dump() const {
  for (IndexList::const_iterator itr = indexList.begin();
       itr != indexList.end(); ++itr) {
    dbgs() << itr->getIndex() << " ";

    if (itr->getInstr()) {
      dbgs() << *itr->getInstr();
    } else {
      dbgs() << "\n";
    }
  }

  for (unsigned i = 0, e = MBBRanges.size(); i != e; ++i)
    dbgs() << "BB#" << i << "\t[" << MBBRanges[i].first << ';'
           << MBBRanges[i].second << ")\n";
}
#endif

// Print a SlotIndex to a raw_ostream.
void SlotIndex::print(raw_ostream &os) const {
  if (isValid())
    os << listEntry()->getIndex() << "Berd"[getSlot()];
  else
    os << "invalid";
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
// Dump a SlotIndex to stderr.
void SlotIndex::dump() const {
  print(dbgs());
  dbgs() << "\n";
}
#endif

