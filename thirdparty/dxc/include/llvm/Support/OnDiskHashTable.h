//===--- OnDiskHashTable.h - On-Disk Hash Table Implementation --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Defines facilities for reading and writing on-disk hash tables.
///
//===----------------------------------------------------------------------===//
#ifndef LLVM_SUPPORT_ONDISKHASHTABLE_H
#define LLVM_SUPPORT_ONDISKHASHTABLE_H

#include "llvm/Support/AlignOf.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstdlib>

namespace llvm {

/// \brief Generates an on disk hash table.
///
/// This needs an \c Info that handles storing values into the hash table's
/// payload and computes the hash for a given key. This should provide the
/// following interface:
///
/// \code
/// class ExampleInfo {
/// public:
///   typedef ExampleKey key_type;   // Must be copy constructible
///   typedef ExampleKey &key_type_ref;
///   typedef ExampleData data_type; // Must be copy constructible
///   typedef ExampleData &data_type_ref;
///   typedef uint32_t hash_value_type; // The type the hash function returns.
///   typedef uint32_t offset_type; // The type for offsets into the table.
///
///   /// Calculate the hash for Key
///   static hash_value_type ComputeHash(key_type_ref Key);
///   /// Return the lengths, in bytes, of the given Key/Data pair.
///   static std::pair<offset_type, offset_type>
///   EmitKeyDataLength(raw_ostream &Out, key_type_ref Key, data_type_ref Data);
///   /// Write Key to Out.  KeyLen is the length from EmitKeyDataLength.
///   static void EmitKey(raw_ostream &Out, key_type_ref Key,
///                       offset_type KeyLen);
///   /// Write Data to Out.  DataLen is the length from EmitKeyDataLength.
///   static void EmitData(raw_ostream &Out, key_type_ref Key,
///                        data_type_ref Data, offset_type DataLen);
/// };
/// \endcode
template <typename Info> class OnDiskChainedHashTableGenerator {
  /// \brief A single item in the hash table.
  class Item {
  public:
    typename Info::key_type Key;
    typename Info::data_type Data;
    Item *Next;
    const typename Info::hash_value_type Hash;

    Item(typename Info::key_type_ref Key, typename Info::data_type_ref Data,
         Info &InfoObj)
        : Key(Key), Data(Data), Next(nullptr), Hash(InfoObj.ComputeHash(Key)) {}
  };

  typedef typename Info::offset_type offset_type;
  offset_type NumBuckets;
  offset_type NumEntries;
  llvm::SpecificBumpPtrAllocator<Item> BA;

  /// \brief A linked list of values in a particular hash bucket.
  struct Bucket {
    offset_type Off;
    unsigned Length;
    Item *Head;
  };

  Bucket *Buckets;

private:
  /// \brief Insert an item into the appropriate hash bucket.
  void insert(Bucket *Buckets, size_t Size, Item *E) {
    Bucket &B = Buckets[E->Hash & (Size - 1)];
    E->Next = B.Head;
    ++B.Length;
    B.Head = E;
  }

  /// \brief Resize the hash table, moving the old entries into the new buckets.
  void resize(size_t NewSize) {
    // HLSL Change Begin: Use overridable operator new
    Bucket* NewBuckets = new Bucket[NewSize];
    std::memset(NewBuckets, 0, NewSize * sizeof(Bucket));
    // HLSL Change End
    // Populate NewBuckets with the old entries.
    for (size_t I = 0; I < NumBuckets; ++I)
      for (Item *E = Buckets[I].Head; E;) {
        Item *N = E->Next;
        E->Next = nullptr;
        insert(NewBuckets, NewSize, E);
        E = N;
      }

    delete[] Buckets; // HLSL Change: Use overridable operator delete
    NumBuckets = NewSize;
    Buckets = NewBuckets;
  }

public:
  /// \brief Insert an entry into the table.
  void insert(typename Info::key_type_ref Key,
              typename Info::data_type_ref Data) {
    Info InfoObj;
    insert(Key, Data, InfoObj);
  }

  /// \brief Insert an entry into the table.
  ///
  /// Uses the provided Info instead of a stack allocated one.
  void insert(typename Info::key_type_ref Key,
              typename Info::data_type_ref Data, Info &InfoObj) {

    ++NumEntries;
    if (4 * NumEntries >= 3 * NumBuckets)
      resize(NumBuckets * 2);
    insert(Buckets, NumBuckets, new (BA.Allocate()) Item(Key, Data, InfoObj));
  }

  /// \brief Emit the table to Out, which must not be at offset 0.
  offset_type Emit(raw_ostream &Out) {
    Info InfoObj;
    return Emit(Out, InfoObj);
  }

  /// \brief Emit the table to Out, which must not be at offset 0.
  ///
  /// Uses the provided Info instead of a stack allocated one.
  offset_type Emit(raw_ostream &Out, Info &InfoObj) {
    using namespace llvm::support;
    endian::Writer<little> LE(Out);

    // Emit the payload of the table.
    for (offset_type I = 0; I < NumBuckets; ++I) {
      Bucket &B = Buckets[I];
      if (!B.Head)
        continue;

      // Store the offset for the data of this bucket.
      B.Off = Out.tell();
      assert(B.Off && "Cannot write a bucket at offset 0. Please add padding.");

      // Write out the number of items in the bucket.
      LE.write<uint16_t>(B.Length);
      assert(B.Length != 0 && "Bucket has a head but zero length?");

      // Write out the entries in the bucket.
      for (Item *I = B.Head; I; I = I->Next) {
        LE.write<typename Info::hash_value_type>(I->Hash);
        const std::pair<offset_type, offset_type> &Len =
            InfoObj.EmitKeyDataLength(Out, I->Key, I->Data);
        InfoObj.EmitKey(Out, I->Key, Len.first);
        InfoObj.EmitData(Out, I->Key, I->Data, Len.second);
      }
    }

    // Pad with zeros so that we can start the hashtable at an aligned address.
    offset_type TableOff = Out.tell();
    uint64_t N = llvm::OffsetToAlignment(TableOff, alignOf<offset_type>());
    TableOff += N;
    while (N--)
      LE.write<uint8_t>(0);

    // Emit the hashtable itself.
    LE.write<offset_type>(NumBuckets);
    LE.write<offset_type>(NumEntries);
    for (offset_type I = 0; I < NumBuckets; ++I)
      LE.write<offset_type>(Buckets[I].Off);

    return TableOff;
  }

  OnDiskChainedHashTableGenerator() {
    NumEntries = 0;
    NumBuckets = 64;
    // Note that we do not need to run the constructors of the individual
    // Bucket objects since 'calloc' returns bytes that are all 0.
    // HLSL Change Begin: Use overridable operator new
    Buckets = new Bucket[NumBuckets];
    std::memset(Buckets, 0, NumBuckets * sizeof(Bucket));
    // HLSL Change End
  }

  ~OnDiskChainedHashTableGenerator() { delete[] Buckets; } // HLSL Change: Use overridable operator delete
};

/// \brief Provides lookup on an on disk hash table.
///
/// This needs an \c Info that handles reading values from the hash table's
/// payload and computes the hash for a given key. This should provide the
/// following interface:
///
/// \code
/// class ExampleLookupInfo {
/// public:
///   typedef ExampleData data_type;
///   typedef ExampleInternalKey internal_key_type; // The stored key type.
///   typedef ExampleKey external_key_type; // The type to pass to find().
///   typedef uint32_t hash_value_type; // The type the hash function returns.
///   typedef uint32_t offset_type; // The type for offsets into the table.
///
///   /// Compare two keys for equality.
///   static bool EqualKey(internal_key_type &Key1, internal_key_type &Key2);
///   /// Calculate the hash for the given key.
///   static hash_value_type ComputeHash(internal_key_type &IKey);
///   /// Translate from the semantic type of a key in the hash table to the
///   /// type that is actually stored and used for hashing and comparisons.
///   /// The internal and external types are often the same, in which case this
///   /// can simply return the passed in value.
///   static const internal_key_type &GetInternalKey(external_key_type &EKey);
///   /// Read the key and data length from Buffer, leaving it pointing at the
///   /// following byte.
///   static std::pair<offset_type, offset_type>
///   ReadKeyDataLength(const unsigned char *&Buffer);
///   /// Read the key from Buffer, given the KeyLen as reported from
///   /// ReadKeyDataLength.
///   const internal_key_type &ReadKey(const unsigned char *Buffer,
///                                    offset_type KeyLen);
///   /// Read the data for Key from Buffer, given the DataLen as reported from
///   /// ReadKeyDataLength.
///   data_type ReadData(StringRef Key, const unsigned char *Buffer,
///                      offset_type DataLen);
/// };
/// \endcode
template <typename Info> class OnDiskChainedHashTable {
  const typename Info::offset_type NumBuckets;
  const typename Info::offset_type NumEntries;
  const unsigned char *const Buckets;
  const unsigned char *const Base;
  Info InfoObj;

public:
  typedef typename Info::internal_key_type internal_key_type;
  typedef typename Info::external_key_type external_key_type;
  typedef typename Info::data_type         data_type;
  typedef typename Info::hash_value_type   hash_value_type;
  typedef typename Info::offset_type       offset_type;

  OnDiskChainedHashTable(offset_type NumBuckets, offset_type NumEntries,
                         const unsigned char *Buckets,
                         const unsigned char *Base,
                         const Info &InfoObj = Info())
      : NumBuckets(NumBuckets), NumEntries(NumEntries), Buckets(Buckets),
        Base(Base), InfoObj(InfoObj) {
    assert((reinterpret_cast<uintptr_t>(Buckets) & 0x3) == 0 &&
           "'buckets' must have a 4-byte alignment");
  }

  offset_type getNumBuckets() const { return NumBuckets; }
  offset_type getNumEntries() const { return NumEntries; }
  const unsigned char *getBase() const { return Base; }
  const unsigned char *getBuckets() const { return Buckets; }

  bool isEmpty() const { return NumEntries == 0; }

  class iterator {
    internal_key_type Key;
    const unsigned char *const Data;
    const offset_type Len;
    Info *InfoObj;

  public:
    iterator() : Data(nullptr), Len(0) {}
    iterator(const internal_key_type K, const unsigned char *D, offset_type L,
             Info *InfoObj)
        : Key(K), Data(D), Len(L), InfoObj(InfoObj) {}

    data_type operator*() const { return InfoObj->ReadData(Key, Data, Len); }
    bool operator==(const iterator &X) const { return X.Data == Data; }
    bool operator!=(const iterator &X) const { return X.Data != Data; }
  };

  /// \brief Look up the stored data for a particular key.
  iterator find(const external_key_type &EKey, Info *InfoPtr = nullptr) {
    const internal_key_type &IKey = InfoObj.GetInternalKey(EKey);
    hash_value_type KeyHash = InfoObj.ComputeHash(IKey);
    return find_hashed(IKey, KeyHash, InfoPtr);
  }

  /// \brief Look up the stored data for a particular key with a known hash.
  iterator find_hashed(const internal_key_type &IKey, hash_value_type KeyHash,
                       Info *InfoPtr = nullptr) {
    using namespace llvm::support;

    if (!InfoPtr)
      InfoPtr = &InfoObj;

    // Each bucket is just an offset into the hash table file.
    offset_type Idx = KeyHash & (NumBuckets - 1);
    const unsigned char *Bucket = Buckets + sizeof(offset_type) * Idx;

    offset_type Offset = endian::readNext<offset_type, little, aligned>(Bucket);
    if (Offset == 0)
      return iterator(); // Empty bucket.
    const unsigned char *Items = Base + Offset;

    // 'Items' starts with a 16-bit unsigned integer representing the
    // number of items in this bucket.
    unsigned Len = endian::readNext<uint16_t, little, unaligned>(Items);

    for (unsigned i = 0; i < Len; ++i) {
      // Read the hash.
      hash_value_type ItemHash =
          endian::readNext<hash_value_type, little, unaligned>(Items);

      // Determine the length of the key and the data.
      const std::pair<offset_type, offset_type> &L =
          Info::ReadKeyDataLength(Items);
      offset_type ItemLen = L.first + L.second;

      // Compare the hashes.  If they are not the same, skip the entry entirely.
      if (ItemHash != KeyHash) {
        Items += ItemLen;
        continue;
      }

      // Read the key.
      const internal_key_type &X =
          InfoPtr->ReadKey((const unsigned char *const)Items, L.first);

      // If the key doesn't match just skip reading the value.
      if (!InfoPtr->EqualKey(X, IKey)) {
        Items += ItemLen;
        continue;
      }

      // The key matches!
      return iterator(X, Items + L.first, L.second, InfoPtr);
    }

    return iterator();
  }

  iterator end() const { return iterator(); }

  Info &getInfoObj() { return InfoObj; }

  /// \brief Create the hash table.
  ///
  /// \param Buckets is the beginning of the hash table itself, which follows
  /// the payload of entire structure. This is the value returned by
  /// OnDiskHashTableGenerator::Emit.
  ///
  /// \param Base is the point from which all offsets into the structure are
  /// based. This is offset 0 in the stream that was used when Emitting the
  /// table.
  static OnDiskChainedHashTable *Create(const unsigned char *Buckets,
                                        const unsigned char *const Base,
                                        const Info &InfoObj = Info()) {
    using namespace llvm::support;
    assert(Buckets > Base);
    assert((reinterpret_cast<uintptr_t>(Buckets) & 0x3) == 0 &&
           "buckets should be 4-byte aligned.");

    offset_type NumBuckets =
        endian::readNext<offset_type, little, aligned>(Buckets);
    offset_type NumEntries =
        endian::readNext<offset_type, little, aligned>(Buckets);
    return new OnDiskChainedHashTable<Info>(NumBuckets, NumEntries, Buckets,
                                            Base, InfoObj);
  }
};

/// \brief Provides lookup and iteration over an on disk hash table.
///
/// \copydetails llvm::OnDiskChainedHashTable
template <typename Info>
class OnDiskIterableChainedHashTable : public OnDiskChainedHashTable<Info> {
  const unsigned char *Payload;

public:
  typedef OnDiskChainedHashTable<Info>          base_type;
  typedef typename base_type::internal_key_type internal_key_type;
  typedef typename base_type::external_key_type external_key_type;
  typedef typename base_type::data_type         data_type;
  typedef typename base_type::hash_value_type   hash_value_type;
  typedef typename base_type::offset_type       offset_type;

  OnDiskIterableChainedHashTable(offset_type NumBuckets, offset_type NumEntries,
                                 const unsigned char *Buckets,
                                 const unsigned char *Payload,
                                 const unsigned char *Base,
                                 const Info &InfoObj = Info())
      : base_type(NumBuckets, NumEntries, Buckets, Base, InfoObj),
        Payload(Payload) {}

  /// \brief Iterates over all of the keys in the table.
  class key_iterator {
    const unsigned char *Ptr;
    offset_type NumItemsInBucketLeft;
    offset_type NumEntriesLeft;
    Info *InfoObj;

  public:
    typedef external_key_type value_type;

    key_iterator(const unsigned char *const Ptr, offset_type NumEntries,
                 Info *InfoObj)
        : Ptr(Ptr), NumItemsInBucketLeft(0), NumEntriesLeft(NumEntries),
          InfoObj(InfoObj) {}
    key_iterator()
        : Ptr(nullptr), NumItemsInBucketLeft(0), NumEntriesLeft(0),
          InfoObj(0) {}

    friend bool operator==(const key_iterator &X, const key_iterator &Y) {
      return X.NumEntriesLeft == Y.NumEntriesLeft;
    }
    friend bool operator!=(const key_iterator &X, const key_iterator &Y) {
      return X.NumEntriesLeft != Y.NumEntriesLeft;
    }

    key_iterator &operator++() { // Preincrement
      using namespace llvm::support;
      if (!NumItemsInBucketLeft) {
        // 'Items' starts with a 16-bit unsigned integer representing the
        // number of items in this bucket.
        NumItemsInBucketLeft =
            endian::readNext<uint16_t, little, unaligned>(Ptr);
      }
      Ptr += sizeof(hash_value_type); // Skip the hash.
      // Determine the length of the key and the data.
      const std::pair<offset_type, offset_type> &L =
          Info::ReadKeyDataLength(Ptr);
      Ptr += L.first + L.second;
      assert(NumItemsInBucketLeft);
      --NumItemsInBucketLeft;
      assert(NumEntriesLeft);
      --NumEntriesLeft;
      return *this;
    }
    key_iterator operator++(int) { // Postincrement
      key_iterator tmp = *this; ++*this; return tmp;
    }

    value_type operator*() const {
      const unsigned char *LocalPtr = Ptr;
      if (!NumItemsInBucketLeft)
        LocalPtr += 2; // number of items in bucket
      LocalPtr += sizeof(hash_value_type); // Skip the hash.

      // Determine the length of the key and the data.
      const std::pair<offset_type, offset_type> &L =
          Info::ReadKeyDataLength(LocalPtr);

      // Read the key.
      const internal_key_type &Key = InfoObj->ReadKey(LocalPtr, L.first);
      return InfoObj->GetExternalKey(Key);
    }
  };

  key_iterator key_begin() {
    return key_iterator(Payload, this->getNumEntries(), &this->getInfoObj());
  }
  key_iterator key_end() { return key_iterator(); }

  iterator_range<key_iterator> keys() {
    return make_range(key_begin(), key_end());
  }

  /// \brief Iterates over all the entries in the table, returning the data.
  class data_iterator {
    const unsigned char *Ptr;
    offset_type NumItemsInBucketLeft;
    offset_type NumEntriesLeft;
    Info *InfoObj;

  public:
    typedef data_type value_type;

    data_iterator(const unsigned char *const Ptr, offset_type NumEntries,
                  Info *InfoObj)
        : Ptr(Ptr), NumItemsInBucketLeft(0), NumEntriesLeft(NumEntries),
          InfoObj(InfoObj) {}
    data_iterator()
        : Ptr(nullptr), NumItemsInBucketLeft(0), NumEntriesLeft(0),
          InfoObj(nullptr) {}

    bool operator==(const data_iterator &X) const {
      return X.NumEntriesLeft == NumEntriesLeft;
    }
    bool operator!=(const data_iterator &X) const {
      return X.NumEntriesLeft != NumEntriesLeft;
    }

    data_iterator &operator++() { // Preincrement
      using namespace llvm::support;
      if (!NumItemsInBucketLeft) {
        // 'Items' starts with a 16-bit unsigned integer representing the
        // number of items in this bucket.
        NumItemsInBucketLeft =
            endian::readNext<uint16_t, little, unaligned>(Ptr);
      }
      Ptr += sizeof(hash_value_type); // Skip the hash.
      // Determine the length of the key and the data.
      const std::pair<offset_type, offset_type> &L =
          Info::ReadKeyDataLength(Ptr);
      Ptr += L.first + L.second;
      assert(NumItemsInBucketLeft);
      --NumItemsInBucketLeft;
      assert(NumEntriesLeft);
      --NumEntriesLeft;
      return *this;
    }
    data_iterator operator++(int) { // Postincrement
      data_iterator tmp = *this; ++*this; return tmp;
    }

    value_type operator*() const {
      const unsigned char *LocalPtr = Ptr;
      if (!NumItemsInBucketLeft)
        LocalPtr += 2; // number of items in bucket
      LocalPtr += sizeof(hash_value_type); // Skip the hash.

      // Determine the length of the key and the data.
      const std::pair<offset_type, offset_type> &L =
          Info::ReadKeyDataLength(LocalPtr);

      // Read the key.
      const internal_key_type &Key = InfoObj->ReadKey(LocalPtr, L.first);
      return InfoObj->ReadData(Key, LocalPtr + L.first, L.second);
    }
  };

  data_iterator data_begin() {
    return data_iterator(Payload, this->getNumEntries(), &this->getInfoObj());
  }
  data_iterator data_end() { return data_iterator(); }

  iterator_range<data_iterator> data() {
    return make_range(data_begin(), data_end());
  }

  /// \brief Create the hash table.
  ///
  /// \param Buckets is the beginning of the hash table itself, which follows
  /// the payload of entire structure. This is the value returned by
  /// OnDiskHashTableGenerator::Emit.
  ///
  /// \param Payload is the beginning of the data contained in the table.  This
  /// is Base plus any padding or header data that was stored, ie, the offset
  /// that the stream was at when calling Emit.
  ///
  /// \param Base is the point from which all offsets into the structure are
  /// based. This is offset 0 in the stream that was used when Emitting the
  /// table.
  static OnDiskIterableChainedHashTable *
  Create(const unsigned char *Buckets, const unsigned char *const Payload,
         const unsigned char *const Base, const Info &InfoObj = Info()) {
    using namespace llvm::support;
    assert(Buckets > Base);
    assert((reinterpret_cast<uintptr_t>(Buckets) & 0x3) == 0 &&
           "buckets should be 4-byte aligned.");

    offset_type NumBuckets =
        endian::readNext<offset_type, little, aligned>(Buckets);
    offset_type NumEntries =
        endian::readNext<offset_type, little, aligned>(Buckets);
    return new OnDiskIterableChainedHashTable<Info>(
        NumBuckets, NumEntries, Buckets, Payload, Base, InfoObj);
  }
};

} // end namespace llvm

#endif
