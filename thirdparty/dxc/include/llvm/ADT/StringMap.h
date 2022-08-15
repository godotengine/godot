//===--- StringMap.h - String Hash table map interface ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the StringMap class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_STRINGMAP_H
#define LLVM_ADT_STRINGMAP_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Allocator.h"
#include <cstring>
#include <utility>

namespace llvm {
  template<typename ValueT>
  class StringMapConstIterator;
  template<typename ValueT>
  class StringMapIterator;
  template<typename ValueTy>
  class StringMapEntry;

/// StringMapEntryBase - Shared base class of StringMapEntry instances.
class StringMapEntryBase {
  unsigned StrLen;
public:
  explicit StringMapEntryBase(unsigned Len) : StrLen(Len) {}

  unsigned getKeyLength() const { return StrLen; }
};

/// StringMapImpl - This is the base class of StringMap that is shared among
/// all of its instantiations.
class StringMapImpl {
protected:
  // Array of NumBuckets pointers to entries, null pointers are holes.
  // TheTable[NumBuckets] contains a sentinel value for easy iteration. Followed
  // by an array of the actual hash values as unsigned integers.
  StringMapEntryBase **TheTable;
  unsigned NumBuckets;
  unsigned NumItems;
  unsigned NumTombstones;
  unsigned ItemSize;
protected:
  explicit StringMapImpl(unsigned itemSize)
      : TheTable(nullptr),
        // Initialize the map with zero buckets to allocation.
        NumBuckets(0), NumItems(0), NumTombstones(0), ItemSize(itemSize) {}
  StringMapImpl(StringMapImpl &&RHS)
      : TheTable(RHS.TheTable), NumBuckets(RHS.NumBuckets),
        NumItems(RHS.NumItems), NumTombstones(RHS.NumTombstones),
        ItemSize(RHS.ItemSize) {
    RHS.TheTable = nullptr;
    RHS.NumBuckets = 0;
    RHS.NumItems = 0;
    RHS.NumTombstones = 0;
  }

  StringMapImpl(unsigned InitSize, unsigned ItemSize);
  unsigned RehashTable(unsigned BucketNo = 0);

  /// LookupBucketFor - Look up the bucket that the specified string should end
  /// up in.  If it already exists as a key in the map, the Item pointer for the
  /// specified bucket will be non-null.  Otherwise, it will be null.  In either
  /// case, the FullHashValue field of the bucket will be set to the hash value
  /// of the string.
  unsigned LookupBucketFor(StringRef Key);

  /// FindKey - Look up the bucket that contains the specified key. If it exists
  /// in the map, return the bucket number of the key.  Otherwise return -1.
  /// This does not modify the map.
  int FindKey(StringRef Key) const;

  /// RemoveKey - Remove the specified StringMapEntry from the table, but do not
  /// delete it.  This aborts if the value isn't in the table.
  void RemoveKey(StringMapEntryBase *V);

  /// RemoveKey - Remove the StringMapEntry for the specified key from the
  /// table, returning it.  If the key is not in the table, this returns null.
  StringMapEntryBase *RemoveKey(StringRef Key);
private:
  void init(unsigned Size);
public:
  static StringMapEntryBase *getTombstoneVal() {
    return (StringMapEntryBase*)-1;
  }

  unsigned getNumBuckets() const { return NumBuckets; }
  unsigned getNumItems() const { return NumItems; }

  bool empty() const { return NumItems == 0; }
  unsigned size() const { return NumItems; }

  void swap(StringMapImpl &Other) {
    std::swap(TheTable, Other.TheTable);
    std::swap(NumBuckets, Other.NumBuckets);
    std::swap(NumItems, Other.NumItems);
    std::swap(NumTombstones, Other.NumTombstones);
  }
};

/// StringMapEntry - This is used to represent one value that is inserted into
/// a StringMap.  It contains the Value itself and the key: the string length
/// and data.
template<typename ValueTy>
class StringMapEntry : public StringMapEntryBase {
  StringMapEntry(StringMapEntry &E) = delete;
public:
  ValueTy second;

  explicit StringMapEntry(unsigned strLen)
    : StringMapEntryBase(strLen), second() {}
  template <class InitTy>
  StringMapEntry(unsigned strLen, InitTy &&V)
      : StringMapEntryBase(strLen), second(std::forward<InitTy>(V)) {}

  StringRef getKey() const {
    return StringRef(getKeyData(), getKeyLength());
  }

  const ValueTy &getValue() const { return second; }
  ValueTy &getValue() { return second; }

  void setValue(const ValueTy &V) { second = V; }

  /// getKeyData - Return the start of the string data that is the key for this
  /// value.  The string data is always stored immediately after the
  /// StringMapEntry object.
  const char *getKeyData() const {return reinterpret_cast<const char*>(this+1);}

  StringRef first() const { return StringRef(getKeyData(), getKeyLength()); }

  /// Create - Create a StringMapEntry for the specified key and default
  /// construct the value.
  template <typename AllocatorTy, typename InitType>
  static StringMapEntry *Create(StringRef Key, AllocatorTy &Allocator,
                                InitType &&InitVal) {
    unsigned KeyLength = Key.size();

    // Allocate a new item with space for the string at the end and a null
    // terminator.
    unsigned AllocSize = static_cast<unsigned>(sizeof(StringMapEntry))+
      KeyLength+1;
    unsigned Alignment = alignOf<StringMapEntry>();

    StringMapEntry *NewItem =
      static_cast<StringMapEntry*>(Allocator.Allocate(AllocSize,Alignment));

    // Default construct the value.
    new (NewItem) StringMapEntry(KeyLength, std::forward<InitType>(InitVal));

    // Copy the string information.
    char *StrBuffer = const_cast<char*>(NewItem->getKeyData());
    if (KeyLength > 0)
      memcpy(StrBuffer, Key.data(), KeyLength);
    StrBuffer[KeyLength] = 0;  // Null terminate for convenience of clients.
    return NewItem;
  }

  template<typename AllocatorTy>
  static StringMapEntry *Create(StringRef Key, AllocatorTy &Allocator) {
    return Create(Key, Allocator, ValueTy());
  }

  /// Create - Create a StringMapEntry with normal malloc/free.
  template<typename InitType>
  static StringMapEntry *Create(StringRef Key, InitType &&InitVal) {
    MallocAllocator A;
    return Create(Key, A, std::forward<InitType>(InitVal));
  }

  static StringMapEntry *Create(StringRef Key) {
    return Create(Key, ValueTy());
  }

  /// GetStringMapEntryFromKeyData - Given key data that is known to be embedded
  /// into a StringMapEntry, return the StringMapEntry itself.
  static StringMapEntry &GetStringMapEntryFromKeyData(const char *KeyData) {
    char *Ptr = const_cast<char*>(KeyData) - sizeof(StringMapEntry<ValueTy>);
    return *reinterpret_cast<StringMapEntry*>(Ptr);
  }

  /// Destroy - Destroy this StringMapEntry, releasing memory back to the
  /// specified allocator.
  template<typename AllocatorTy>
  void Destroy(AllocatorTy &Allocator) {
    // Free memory referenced by the item.
    unsigned AllocSize =
        static_cast<unsigned>(sizeof(StringMapEntry)) + getKeyLength() + 1;
    this->~StringMapEntry();
    Allocator.Deallocate(static_cast<void *>(this), AllocSize);
  }

  /// Destroy this object, releasing memory back to the malloc allocator.
  void Destroy() {
    MallocAllocator A;
    Destroy(A);
  }
};


/// StringMap - This is an unconventional map that is specialized for handling
/// keys that are "strings", which are basically ranges of bytes. This does some
/// funky memory allocation and hashing things to make it extremely efficient,
/// storing the string data *after* the value in the map.
template<typename ValueTy, typename AllocatorTy = MallocAllocator>
class StringMap : public StringMapImpl {
  AllocatorTy Allocator;
public:
  typedef StringMapEntry<ValueTy> MapEntryTy;
  
  StringMap() : StringMapImpl(static_cast<unsigned>(sizeof(MapEntryTy))) {}
  explicit StringMap(unsigned InitialSize)
    : StringMapImpl(InitialSize, static_cast<unsigned>(sizeof(MapEntryTy))) {}

  explicit StringMap(AllocatorTy A)
    : StringMapImpl(static_cast<unsigned>(sizeof(MapEntryTy))), Allocator(A) {}

  StringMap(unsigned InitialSize, AllocatorTy A)
    : StringMapImpl(InitialSize, static_cast<unsigned>(sizeof(MapEntryTy))),
      Allocator(A) {}

  StringMap(StringMap &&RHS)
      : StringMapImpl(std::move(RHS)), Allocator(std::move(RHS.Allocator)) {}

  StringMap &operator=(StringMap RHS) {
    StringMapImpl::swap(RHS);
    std::swap(Allocator, RHS.Allocator);
    return *this;
  }

  // FIXME: Implement copy operations if/when they're needed.

  AllocatorTy &getAllocator() { return Allocator; }
  const AllocatorTy &getAllocator() const { return Allocator; }

  typedef const char* key_type;
  typedef ValueTy mapped_type;
  typedef StringMapEntry<ValueTy> value_type;
  typedef size_t size_type;

  typedef StringMapConstIterator<ValueTy> const_iterator;
  typedef StringMapIterator<ValueTy> iterator;

  iterator begin() {
    return iterator(TheTable, NumBuckets == 0);
  }
  iterator end() {
    return iterator(TheTable+NumBuckets, true);
  }
  const_iterator begin() const {
    return const_iterator(TheTable, NumBuckets == 0);
  }
  const_iterator end() const {
    return const_iterator(TheTable+NumBuckets, true);
  }

  iterator find(StringRef Key) {
    int Bucket = FindKey(Key);
    if (Bucket == -1) return end();
    return iterator(TheTable+Bucket, true);
  }

  const_iterator find(StringRef Key) const {
    int Bucket = FindKey(Key);
    if (Bucket == -1) return end();
    return const_iterator(TheTable+Bucket, true);
  }

  /// lookup - Return the entry for the specified key, or a default
  /// constructed value if no such entry exists.
  ValueTy lookup(StringRef Key) const {
    const_iterator it = find(Key);
    if (it != end())
      return it->second;
    return ValueTy();
  }

  ValueTy &operator[](StringRef Key) {
    return insert(std::make_pair(Key, ValueTy())).first->second;
  }

  /// count - Return 1 if the element is in the map, 0 otherwise.
  size_type count(StringRef Key) const {
    return find(Key) == end() ? 0 : 1;
  }

  /// insert - Insert the specified key/value pair into the map.  If the key
  /// already exists in the map, return false and ignore the request, otherwise
  /// insert it and return true.
  bool insert(MapEntryTy *KeyValue) {
    unsigned BucketNo = LookupBucketFor(KeyValue->getKey());
    StringMapEntryBase *&Bucket = TheTable[BucketNo];
    if (Bucket && Bucket != getTombstoneVal())
      return false;  // Already exists in map.

    if (Bucket == getTombstoneVal())
      --NumTombstones;
    Bucket = KeyValue;
    ++NumItems;
    assert(NumItems + NumTombstones <= NumBuckets);

    RehashTable();
    return true;
  }

  /// insert - Inserts the specified key/value pair into the map if the key
  /// isn't already in the map. The bool component of the returned pair is true
  /// if and only if the insertion takes place, and the iterator component of
  /// the pair points to the element with key equivalent to the key of the pair.
  std::pair<iterator, bool> insert(std::pair<StringRef, ValueTy> KV) {
    unsigned BucketNo = LookupBucketFor(KV.first);
    StringMapEntryBase *&Bucket = TheTable[BucketNo];
    if (Bucket && Bucket != getTombstoneVal())
      return std::make_pair(iterator(TheTable + BucketNo, false),
                            false); // Already exists in map.

    if (Bucket == getTombstoneVal())
      --NumTombstones;
    Bucket =
        MapEntryTy::Create(KV.first, Allocator, std::move(KV.second));
    ++NumItems;
    assert(NumItems + NumTombstones <= NumBuckets);

    BucketNo = RehashTable(BucketNo);
    return std::make_pair(iterator(TheTable + BucketNo, false), true);
  }

  // clear - Empties out the StringMap
  void clear() {
    if (empty()) return;

    // Zap all values, resetting the keys back to non-present (not tombstone),
    // which is safe because we're removing all elements.
    for (unsigned I = 0, E = NumBuckets; I != E; ++I) {
      StringMapEntryBase *&Bucket = TheTable[I];
      if (Bucket && Bucket != getTombstoneVal()) {
        static_cast<MapEntryTy*>(Bucket)->Destroy(Allocator);
      }
      Bucket = nullptr;
    }

    NumItems = 0;
    NumTombstones = 0;
  }

  /// remove - Remove the specified key/value pair from the map, but do not
  /// erase it.  This aborts if the key is not in the map.
  void remove(MapEntryTy *KeyValue) {
    RemoveKey(KeyValue);
  }

  void erase(iterator I) {
    MapEntryTy &V = *I;
    remove(&V);
    V.Destroy(Allocator);
  }

  bool erase(StringRef Key) {
    iterator I = find(Key);
    if (I == end()) return false;
    erase(I);
    return true;
  }

  ~StringMap() {
    // Delete all the elements in the map, but don't reset the elements
    // to default values.  This is a copy of clear(), but avoids unnecessary
    // work not required in the destructor.
    if (!empty()) {
      for (unsigned I = 0, E = NumBuckets; I != E; ++I) {
        StringMapEntryBase *Bucket = TheTable[I];
        if (Bucket && Bucket != getTombstoneVal()) {
          static_cast<MapEntryTy*>(Bucket)->Destroy(Allocator);
        }
      }
    }
    ::operator delete(TheTable); // HLSL Change Begin: Use overridable operator delete
  }
};


template<typename ValueTy>
class StringMapConstIterator {
protected:
  StringMapEntryBase **Ptr;
public:
  typedef StringMapEntry<ValueTy> value_type;

  StringMapConstIterator() : Ptr(nullptr) { }

  explicit StringMapConstIterator(StringMapEntryBase **Bucket,
                                  bool NoAdvance = false)
  : Ptr(Bucket) {
    if (!NoAdvance) AdvancePastEmptyBuckets();
  }

  const value_type &operator*() const {
    return *static_cast<StringMapEntry<ValueTy>*>(*Ptr);
  }
  const value_type *operator->() const {
    return static_cast<StringMapEntry<ValueTy>*>(*Ptr);
  }

  bool operator==(const StringMapConstIterator &RHS) const {
    return Ptr == RHS.Ptr;
  }
  bool operator!=(const StringMapConstIterator &RHS) const {
    return Ptr != RHS.Ptr;
  }

  inline StringMapConstIterator& operator++() {   // Preincrement
    ++Ptr;
    AdvancePastEmptyBuckets();
    return *this;
  }
  StringMapConstIterator operator++(int) {        // Postincrement
    StringMapConstIterator tmp = *this; ++*this; return tmp;
  }

private:
  void AdvancePastEmptyBuckets() {
    while (*Ptr == nullptr || *Ptr == StringMapImpl::getTombstoneVal())
      ++Ptr;
  }
};

template<typename ValueTy>
class StringMapIterator : public StringMapConstIterator<ValueTy> {
public:
  StringMapIterator() {}
  explicit StringMapIterator(StringMapEntryBase **Bucket,
                             bool NoAdvance = false)
    : StringMapConstIterator<ValueTy>(Bucket, NoAdvance) {
  }
  StringMapEntry<ValueTy> &operator*() const {
    return *static_cast<StringMapEntry<ValueTy>*>(*this->Ptr);
  }
  StringMapEntry<ValueTy> *operator->() const {
    return static_cast<StringMapEntry<ValueTy>*>(*this->Ptr);
  }
};

}

#endif
