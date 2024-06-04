// Copyright (c) 2007, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef COMMON_SIMPLE_STRING_DICTIONARY_H_
#define COMMON_SIMPLE_STRING_DICTIONARY_H_

#include <assert.h>
#include <string.h>

#include "common/basictypes.h"

namespace google_breakpad {

// Opaque type for the serialized representation of a NonAllocatingMap. One is
// created in NonAllocatingMap::Serialize and can be deserialized using one of
// the constructors.
struct SerializedNonAllocatingMap;

// NonAllocatingMap is an implementation of a map/dictionary collection that
// uses a fixed amount of storage, so that it does not perform any dynamic
// allocations for its operations.
//
// The actual map storage (the Entry) is guaranteed to be POD, so that it can
// be transmitted over various IPC mechanisms.
//
// The template parameters control the amount of storage used for the key,
// value, and map. The KeySize and ValueSize are measured in bytes, not glyphs,
// and includes space for a \0 byte. This gives space for KeySize-1 and
// ValueSize-1 characters in an entry. NumEntries is the total number of
// entries that will fit in the map.
template <size_t KeySize, size_t ValueSize, size_t NumEntries>
class NonAllocatingMap {
 public:
  // Constant and publicly accessible versions of the template parameters.
  static const size_t key_size = KeySize;
  static const size_t value_size = ValueSize;
  static const size_t num_entries = NumEntries;

  // An Entry object is a single entry in the map. If the key is a 0-length
  // NUL-terminated string, the entry is empty.
  struct Entry {
    char key[KeySize];
    char value[ValueSize];

    bool is_active() const {
      return key[0] != '\0';
    }
  };

  // An Iterator can be used to iterate over all the active entries in a
  // NonAllocatingMap.
  class Iterator {
   public:
    explicit Iterator(const NonAllocatingMap& map)
        : map_(map),
          current_(0) {
    }

    // Returns the next entry in the map, or NULL if at the end of the
    // collection.
    const Entry* Next() {
      while (current_ < map_.num_entries) {
        const Entry* entry = &map_.entries_[current_++];
        if (entry->is_active()) {
          return entry;
        }
      }
      return NULL;
    }

   private:
    const NonAllocatingMap& map_;
    size_t current_;

    DISALLOW_COPY_AND_ASSIGN(Iterator);
  };

  NonAllocatingMap() : entries_() {
  }

  NonAllocatingMap(const NonAllocatingMap& other) {
    *this = other;
  }

  NonAllocatingMap& operator=(const NonAllocatingMap& other) {
    assert(other.key_size == key_size);
    assert(other.value_size == value_size);
    assert(other.num_entries == num_entries);
    if (other.key_size == key_size && other.value_size == value_size &&
        other.num_entries == num_entries) {
      memcpy(entries_, other.entries_, sizeof(entries_));
    }
    return *this;
  }

  // Constructs a map from its serialized form. |map| should be the out
  // parameter from Serialize() and |size| should be its return value.
  NonAllocatingMap(const SerializedNonAllocatingMap* map, size_t size) {
    assert(size == sizeof(entries_));
    if (size == sizeof(entries_)) {
      memcpy(entries_, map, size);
    }
  }

  // Returns the number of active key/value pairs. The upper limit for this
  // is NumEntries.
  size_t GetCount() const {
    size_t count = 0;
    for (size_t i = 0; i < num_entries; ++i) {
      if (entries_[i].is_active()) {
        ++count;
      }
    }
    return count;
  }

  // Given |key|, returns its corresponding |value|. |key| must not be NULL. If
  // the key is not found, NULL is returned.
  const char* GetValueForKey(const char* key) const {
    assert(key);
    if (!key)
      return NULL;

    size_t index = GetEntryIndexForKey(key);
    if (index == num_entries)
      return NULL;

    return entries_[index].value;
  }

  // Stores |value| into |key|, replacing the existing value if |key| is
  // already present. |key| must not be NULL. If |value| is NULL, the key is
  // removed from the map. If there is no more space in the map, then the
  // operation silently fails. Returns an index into the map that can be used
  // to quickly access the entry, or |num_entries| on failure or when clearing
  // a key with a null value.
  size_t SetKeyValue(const char* key, const char* value) {
    if (!value) {
      RemoveKey(key);
      return num_entries;
    }

    assert(key);
    if (!key)
      return num_entries;

    // Key must not be an empty string.
    assert(key[0] != '\0');
    if (key[0] == '\0')
      return num_entries;

    size_t entry_index = GetEntryIndexForKey(key);

    // If it does not yet exist, attempt to insert it.
    if (entry_index == num_entries) {
      for (size_t i = 0; i < num_entries; ++i) {
        if (!entries_[i].is_active()) {
          entry_index = i;
          Entry* entry = &entries_[i];

          strncpy(entry->key, key, key_size);
          entry->key[key_size - 1] = '\0';

          break;
        }
      }
    }

    // If the map is out of space, entry will be NULL.
    if (entry_index == num_entries)
      return num_entries;

#ifndef NDEBUG
    // Sanity check that the key only appears once.
    int count = 0;
    for (size_t i = 0; i < num_entries; ++i) {
      if (strncmp(entries_[i].key, key, key_size) == 0)
        ++count;
    }
    assert(count == 1);
#endif

    strncpy(entries_[entry_index].value, value, value_size);
    entries_[entry_index].value[value_size - 1] = '\0';

    return entry_index;
  }

  // Sets a value for a key that has already been set with SetKeyValue(), using
  // the index returned from that function.
  void SetValueAtIndex(size_t index, const char* value) {
    assert(index < num_entries);
    if (index >= num_entries)
      return;

    Entry* entry = &entries_[index];
    assert(entry->key[0] != '\0');

    strncpy(entry->value, value, value_size);
    entry->value[value_size - 1] = '\0';
  }

  // Given |key|, removes any associated value. |key| must not be NULL. If
  // the key is not found, this is a noop. This invalidates the index
  // returned by SetKeyValue().
  bool RemoveKey(const char* key) {
    assert(key);
    if (!key)
      return false;

    return RemoveAtIndex(GetEntryIndexForKey(key));
  }

  // Removes a value and key using an index that was returned from
  // SetKeyValue(). After a call to this function, the index is invalidated.
  bool RemoveAtIndex(size_t index) {
    if (index >= num_entries)
      return false;

    entries_[index].key[0] = '\0';
    entries_[index].value[0] = '\0';
    return true;
  }

  // Places a serialized version of the map into |map| and returns the size.
  // Both of these should be passed to the deserializing constructor. Note that
  // the serialized |map| is scoped to the lifetime of the non-serialized
  // instance of this class. The |map| can be copied across IPC boundaries.
  size_t Serialize(const SerializedNonAllocatingMap** map) const {
    *map = reinterpret_cast<const SerializedNonAllocatingMap*>(entries_);
    return sizeof(entries_);
  }

 private:
  size_t GetEntryIndexForKey(const char* key) const {
    for (size_t i = 0; i < num_entries; ++i) {
      if (strncmp(key, entries_[i].key, key_size) == 0) {
        return i;
      }
    }
    return num_entries;
  }

  Entry entries_[NumEntries];
};

// For historical reasons this specialized version is available with the same
// size factors as a previous implementation.
typedef NonAllocatingMap<256, 256, 64> SimpleStringDictionary;

}  // namespace google_breakpad

#endif  // COMMON_SIMPLE_STRING_DICTIONARY_H_
