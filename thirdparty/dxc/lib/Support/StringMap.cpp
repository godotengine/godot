//===--- StringMap.cpp - String Hash table map implementation -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the StringMap class.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Compiler.h"
#include <cassert>
using namespace llvm;

StringMapImpl::StringMapImpl(unsigned InitSize, unsigned itemSize) {
  ItemSize = itemSize;
  
  // If a size is specified, initialize the table with that many buckets.
  if (InitSize) {
    init(InitSize);
    return;
  }
  
  // Otherwise, initialize it with zero buckets to avoid the allocation.
  TheTable = nullptr;
  NumBuckets = 0;
  NumItems = 0;
  NumTombstones = 0;
}

void StringMapImpl::init(unsigned InitSize) {
  assert((InitSize & (InitSize-1)) == 0 &&
         "Init Size must be a power of 2 or zero!");
  NumBuckets = InitSize ? InitSize : 16;
  NumItems = 0;
  NumTombstones = 0;
  
  // HLSL Change Begin: Use overridable operator new
  unsigned NewNumBuckets = NumBuckets;
  NumBuckets = 0; // If allocation fails, we should have zero buckets.
  size_t TableSize = (NewNumBuckets + 1) * (sizeof(StringMapEntryBase *) + sizeof(unsigned));
  TheTable = (StringMapEntryBase **)::operator new(TableSize);
  std::memset(TheTable, 0, TableSize);
  NumBuckets = NewNumBuckets;
  // HLSL Change End

  // Allocate one extra bucket, set it to look filled so the iterators stop at
  // end.
  TheTable[NumBuckets] = (StringMapEntryBase*)2;
}


/// LookupBucketFor - Look up the bucket that the specified string should end
/// up in.  If it already exists as a key in the map, the Item pointer for the
/// specified bucket will be non-null.  Otherwise, it will be null.  In either
/// case, the FullHashValue field of the bucket will be set to the hash value
/// of the string.
unsigned StringMapImpl::LookupBucketFor(StringRef Name) {
  unsigned HTSize = NumBuckets;
  if (HTSize == 0) {  // Hash table unallocated so far?
    init(16);
    HTSize = NumBuckets;
  }
  unsigned FullHashValue = HashString(Name);
  unsigned BucketNo = FullHashValue & (HTSize-1);
  unsigned *HashTable = (unsigned *)(TheTable + NumBuckets + 1);

  unsigned ProbeAmt = 1;
  int FirstTombstone = -1;
  while (1) {
    StringMapEntryBase *BucketItem = TheTable[BucketNo];
    // If we found an empty bucket, this key isn't in the table yet, return it.
    if (LLVM_LIKELY(!BucketItem)) {
      // If we found a tombstone, we want to reuse the tombstone instead of an
      // empty bucket.  This reduces probing.
      if (FirstTombstone != -1) {
        HashTable[FirstTombstone] = FullHashValue;
        return FirstTombstone;
      }
      
      HashTable[BucketNo] = FullHashValue;
      return BucketNo;
    }
    
    if (BucketItem == getTombstoneVal()) {
      // Skip over tombstones.  However, remember the first one we see.
      if (FirstTombstone == -1) FirstTombstone = BucketNo;
    } else if (LLVM_LIKELY(HashTable[BucketNo] == FullHashValue)) {
      // If the full hash value matches, check deeply for a match.  The common
      // case here is that we are only looking at the buckets (for item info
      // being non-null and for the full hash value) not at the items.  This
      // is important for cache locality.
      
      // Do the comparison like this because Name isn't necessarily
      // null-terminated!
      char *ItemStr = (char*)BucketItem+ItemSize;
      if (Name == StringRef(ItemStr, BucketItem->getKeyLength())) {
        // We found a match!
        return BucketNo;
      }
    }
    
    // Okay, we didn't find the item.  Probe to the next bucket.
    BucketNo = (BucketNo+ProbeAmt) & (HTSize-1);
    
    // Use quadratic probing, it has fewer clumping artifacts than linear
    // probing and has good cache behavior in the common case.
    ++ProbeAmt;
  }
}


/// FindKey - Look up the bucket that contains the specified key. If it exists
/// in the map, return the bucket number of the key.  Otherwise return -1.
/// This does not modify the map.
int StringMapImpl::FindKey(StringRef Key) const {
  unsigned HTSize = NumBuckets;
  if (HTSize == 0) return -1;  // Really empty table?
  unsigned FullHashValue = HashString(Key);
  unsigned BucketNo = FullHashValue & (HTSize-1);
  unsigned *HashTable = (unsigned *)(TheTable + NumBuckets + 1);

  unsigned ProbeAmt = 1;
  while (1) {
    StringMapEntryBase *BucketItem = TheTable[BucketNo];
    // If we found an empty bucket, this key isn't in the table yet, return.
    if (LLVM_LIKELY(!BucketItem))
      return -1;
    
    if (BucketItem == getTombstoneVal()) {
      // Ignore tombstones.
    } else if (LLVM_LIKELY(HashTable[BucketNo] == FullHashValue)) {
      // If the full hash value matches, check deeply for a match.  The common
      // case here is that we are only looking at the buckets (for item info
      // being non-null and for the full hash value) not at the items.  This
      // is important for cache locality.
      
      // Do the comparison like this because NameStart isn't necessarily
      // null-terminated!
      char *ItemStr = (char*)BucketItem+ItemSize;
      if (Key == StringRef(ItemStr, BucketItem->getKeyLength())) {
        // We found a match!
        return BucketNo;
      }
    }
    
    // Okay, we didn't find the item.  Probe to the next bucket.
    BucketNo = (BucketNo+ProbeAmt) & (HTSize-1);
    
    // Use quadratic probing, it has fewer clumping artifacts than linear
    // probing and has good cache behavior in the common case.
    ++ProbeAmt;
  }
}

/// RemoveKey - Remove the specified StringMapEntry from the table, but do not
/// delete it.  This aborts if the value isn't in the table.
void StringMapImpl::RemoveKey(StringMapEntryBase *V) {
  const char *VStr = (char*)V + ItemSize;
  StringMapEntryBase *V2 = RemoveKey(StringRef(VStr, V->getKeyLength()));
  (void)V2;
  assert(V == V2 && "Didn't find key?");
}

/// RemoveKey - Remove the StringMapEntry for the specified key from the
/// table, returning it.  If the key is not in the table, this returns null.
StringMapEntryBase *StringMapImpl::RemoveKey(StringRef Key) {
  int Bucket = FindKey(Key);
  if (Bucket == -1) return nullptr;
  
  StringMapEntryBase *Result = TheTable[Bucket];
  TheTable[Bucket] = getTombstoneVal();
  --NumItems;
  ++NumTombstones;
  assert(NumItems + NumTombstones <= NumBuckets);

  return Result;
}



/// RehashTable - Grow the table, redistributing values into the buckets with
/// the appropriate mod-of-hashtable-size.
unsigned StringMapImpl::RehashTable(unsigned BucketNo) {
  unsigned NewSize;
  unsigned *HashTable = (unsigned *)(TheTable + NumBuckets + 1);

  // If the hash table is now more than 3/4 full, or if fewer than 1/8 of
  // the buckets are empty (meaning that many are filled with tombstones),
  // grow/rehash the table.
  if (LLVM_UNLIKELY(NumItems * 4 > NumBuckets * 3)) {
    NewSize = NumBuckets*2;
  } else if (LLVM_UNLIKELY(NumBuckets - (NumItems + NumTombstones) <=
                           NumBuckets / 8)) {
    NewSize = NumBuckets;
  } else {
    return BucketNo;
  }

  unsigned NewBucketNo = BucketNo;
  // Allocate one extra bucket which will always be non-empty.  This allows the
  // iterators to stop at end.
  // HLSL Change Begin: Use overridable operator new
  size_t NewTableSize = (NewSize + 1) * (sizeof(StringMapEntryBase*) + sizeof(unsigned));
  StringMapEntryBase **NewTableArray = (StringMapEntryBase **)::operator new(NewTableSize);
  std::memset(NewTableArray, 0, NewTableSize);
  // HLSL Change End
  unsigned *NewHashArray = (unsigned *)(NewTableArray + NewSize + 1);
  NewTableArray[NewSize] = (StringMapEntryBase*)2;

  // Rehash all the items into their new buckets.  Luckily :) we already have
  // the hash values available, so we don't have to rehash any strings.
  for (unsigned I = 0, E = NumBuckets; I != E; ++I) {
    StringMapEntryBase *Bucket = TheTable[I];
    if (Bucket && Bucket != getTombstoneVal()) {
      // Fast case, bucket available.
      unsigned FullHash = HashTable[I];
      unsigned NewBucket = FullHash & (NewSize-1);
      if (!NewTableArray[NewBucket]) {
        NewTableArray[FullHash & (NewSize-1)] = Bucket;
        NewHashArray[FullHash & (NewSize-1)] = FullHash;
        if (I == BucketNo)
          NewBucketNo = NewBucket;
        continue;
      }
      
      // Otherwise probe for a spot.
      unsigned ProbeSize = 1;
      do {
        NewBucket = (NewBucket + ProbeSize++) & (NewSize-1);
      } while (NewTableArray[NewBucket]);
      
      // Finally found a slot.  Fill it in.
      NewTableArray[NewBucket] = Bucket;
      NewHashArray[NewBucket] = FullHash;
      if (I == BucketNo)
        NewBucketNo = NewBucket;
    }
  }
  
  ::operator delete(TheTable); // HLSL Change: Use overridable operator delete
  
  TheTable = NewTableArray;
  NumBuckets = NewSize;
  NumTombstones = 0;
  return NewBucketNo;
}
