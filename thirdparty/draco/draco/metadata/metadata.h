// Copyright 2017 The Draco Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
#ifndef DRACO_METADATA_METADATA_H_
#define DRACO_METADATA_METADATA_H_

#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "draco/core/hash_utils.h"

namespace draco {

// Class for storing a value of an entry in Metadata. Internally it is
// represented by a buffer of data. It can be accessed by various data types,
// e.g. int, float, binary data or string.
class EntryValue {
 public:
  template <typename DataTypeT>
  explicit EntryValue(const DataTypeT &data) {
    const size_t data_type_size = sizeof(DataTypeT);
    data_.resize(data_type_size);
    memcpy(&data_[0], &data, data_type_size);
  }

  template <typename DataTypeT>
  explicit EntryValue(const std::vector<DataTypeT> &data) {
    const size_t total_size = sizeof(DataTypeT) * data.size();
    data_.resize(total_size);
    memcpy(&data_[0], &data[0], total_size);
  }

  EntryValue(const EntryValue &value);
  explicit EntryValue(const std::string &value);

  template <typename DataTypeT>
  bool GetValue(DataTypeT *value) const {
    const size_t data_type_size = sizeof(DataTypeT);
    if (data_type_size != data_.size()) {
      return false;
    }
    memcpy(value, &data_[0], data_type_size);
    return true;
  }

  template <typename DataTypeT>
  bool GetValue(std::vector<DataTypeT> *value) const {
    if (data_.empty()) {
      return false;
    }
    const size_t data_type_size = sizeof(DataTypeT);
    if (data_.size() % data_type_size != 0) {
      return false;
    }
    value->resize(data_.size() / data_type_size);
    memcpy(&value->at(0), &data_[0], data_.size());
    return true;
  }

  const std::vector<uint8_t> &data() const { return data_; }

 private:
  std::vector<uint8_t> data_;

  friend struct EntryValueHasher;
};

// Functor for computing a hash from data stored within an EntryValue.
struct EntryValueHasher {
  size_t operator()(const EntryValue &ev) const {
    size_t hash = ev.data_.size();
    for (size_t i = 0; i < ev.data_.size(); ++i) {
      hash = HashCombine(ev.data_[i], hash);
    }
    return hash;
  }
};

// Class for holding generic metadata. It has a list of entries which consist of
// an entry name and an entry value. Each Metadata could also have nested
// metadata.
class Metadata {
 public:
  Metadata() {}
  Metadata(const Metadata &metadata);
  // In theory, we support all types of data as long as it could be serialized
  // to binary data. We provide the following functions for inserting and
  // accessing entries of common data types. For now, developers need to know
  // the type of entries they are requesting.
  void AddEntryInt(const std::string &name, int32_t value);

  // Returns false if Metadata does not contain an entry with a key of |name|.
  // This function does not guarantee that entry's type is int32_t.
  bool GetEntryInt(const std::string &name, int32_t *value) const;

  void AddEntryIntArray(const std::string &name,
                        const std::vector<int32_t> &value);

  // Returns false if Metadata does not contain an entry with a key of |name|.
  // This function does not guarantee that entry's type is a vector of int32_t.
  bool GetEntryIntArray(const std::string &name,
                        std::vector<int32_t> *value) const;

  void AddEntryDouble(const std::string &name, double value);

  // Returns false if Metadata does not contain an entry with a key of |name|.
  // This function does not guarantee that entry's type is double.
  bool GetEntryDouble(const std::string &name, double *value) const;

  void AddEntryDoubleArray(const std::string &name,
                           const std::vector<double> &value);

  // Returns false if Metadata does not contain an entry with a key of |name|.
  // This function does not guarantee that entry's type is a vector of double.
  bool GetEntryDoubleArray(const std::string &name,
                           std::vector<double> *value) const;

  void AddEntryString(const std::string &name, const std::string &value);

  // Returns false if Metadata does not contain an entry with a key of |name|.
  // This function does not guarantee that entry's type is std::string.
  bool GetEntryString(const std::string &name, std::string *value) const;

  // Add a blob of data as an entry.
  void AddEntryBinary(const std::string &name,
                      const std::vector<uint8_t> &value);

  // Returns false if Metadata does not contain an entry with a key of |name|.
  // This function does not guarantee that entry's type is a vector of uint8_t.
  bool GetEntryBinary(const std::string &name,
                      std::vector<uint8_t> *value) const;

  bool AddSubMetadata(const std::string &name,
                      std::unique_ptr<Metadata> sub_metadata);
  const Metadata *GetSubMetadata(const std::string &name) const;
  Metadata *sub_metadata(const std::string &name);

  void RemoveEntry(const std::string &name);

  int num_entries() const { return static_cast<int>(entries_.size()); }
  const std::map<std::string, EntryValue> &entries() const { return entries_; }
  const std::map<std::string, std::unique_ptr<Metadata>> &sub_metadatas()
      const {
    return sub_metadatas_;
  }

 private:
  // Make this function private to avoid adding undefined data types.
  template <typename DataTypeT>
  void AddEntry(const std::string &entry_name, const DataTypeT &entry_value) {
    const auto itr = entries_.find(entry_name);
    if (itr != entries_.end()) {
      entries_.erase(itr);
    }
    entries_.insert(std::make_pair(entry_name, EntryValue(entry_value)));
  }

  // Make this function private to avoid adding undefined data types.
  template <typename DataTypeT>
  bool GetEntry(const std::string &entry_name, DataTypeT *entry_value) const {
    const auto itr = entries_.find(entry_name);
    if (itr == entries_.end()) {
      return false;
    }
    return itr->second.GetValue(entry_value);
  }

  std::map<std::string, EntryValue> entries_;
  std::map<std::string, std::unique_ptr<Metadata>> sub_metadatas_;

  friend struct MetadataHasher;
};

// Functor for computing a hash from data stored within a metadata class.
struct MetadataHasher {
  size_t operator()(const Metadata &metadata) const {
    size_t hash =
        HashCombine(metadata.entries_.size(), metadata.sub_metadatas_.size());
    EntryValueHasher entry_value_hasher;
    for (const auto &entry : metadata.entries_) {
      hash = HashCombine(entry.first, hash);
      hash = HashCombine(entry_value_hasher(entry.second), hash);
    }
    MetadataHasher metadata_hasher;
    for (auto &&sub_metadata : metadata.sub_metadatas_) {
      hash = HashCombine(sub_metadata.first, hash);
      hash = HashCombine(metadata_hasher(*sub_metadata.second), hash);
    }
    return hash;
  }
};

}  // namespace draco

#endif  // DRACO_METADATA_METADATA_H_
