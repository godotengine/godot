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
#include "draco/metadata/metadata.h"

#include <utility>

namespace draco {

EntryValue::EntryValue(const EntryValue &value) {
  data_.resize(value.data_.size());
  memcpy(&data_[0], &value.data_[0], value.data_.size());
}

EntryValue::EntryValue(const std::string &value) {
  data_.resize(value.size());
  memcpy(&data_[0], &value[0], value.size());
}

template <>
bool EntryValue::GetValue(std::string *value) const {
  if (data_.empty()) {
    return false;
  }
  value->resize(data_.size());
  memcpy(&value->at(0), &data_[0], data_.size());
  return true;
}

Metadata::Metadata(const Metadata &metadata) {
  entries_.insert(metadata.entries_.begin(), metadata.entries_.end());
  for (const auto &sub_metadata_entry : metadata.sub_metadatas_) {
    std::unique_ptr<Metadata> sub_metadata =
        std::unique_ptr<Metadata>(new Metadata(*sub_metadata_entry.second));
    sub_metadatas_[sub_metadata_entry.first] = std::move(sub_metadata);
  }
}

void Metadata::AddEntryInt(const std::string &name, int32_t value) {
  AddEntry(name, value);
}

bool Metadata::GetEntryInt(const std::string &name, int32_t *value) const {
  return GetEntry(name, value);
}

void Metadata::AddEntryIntArray(const std::string &name,
                                const std::vector<int32_t> &value) {
  AddEntry(name, value);
}

bool Metadata::GetEntryIntArray(const std::string &name,
                                std::vector<int32_t> *value) const {
  return GetEntry(name, value);
}

void Metadata::AddEntryDouble(const std::string &name, double value) {
  AddEntry(name, value);
}

bool Metadata::GetEntryDouble(const std::string &name, double *value) const {
  return GetEntry(name, value);
}

void Metadata::AddEntryDoubleArray(const std::string &name,
                                   const std::vector<double> &value) {
  AddEntry(name, value);
}

bool Metadata::GetEntryDoubleArray(const std::string &name,
                                   std::vector<double> *value) const {
  return GetEntry(name, value);
}

void Metadata::AddEntryString(const std::string &name,
                              const std::string &value) {
  AddEntry(name, value);
}

bool Metadata::GetEntryString(const std::string &name,
                              std::string *value) const {
  return GetEntry(name, value);
}

void Metadata::AddEntryBinary(const std::string &name,
                              const std::vector<uint8_t> &value) {
  AddEntry(name, value);
}

bool Metadata::GetEntryBinary(const std::string &name,
                              std::vector<uint8_t> *value) const {
  return GetEntry(name, value);
}

bool Metadata::AddSubMetadata(const std::string &name,
                              std::unique_ptr<Metadata> sub_metadata) {
  auto sub_ptr = sub_metadatas_.find(name);
  // Avoid accidentally writing over a sub-metadata with the same name.
  if (sub_ptr != sub_metadatas_.end()) {
    return false;
  }
  sub_metadatas_[name] = std::move(sub_metadata);
  return true;
}

const Metadata *Metadata::GetSubMetadata(const std::string &name) const {
  auto sub_ptr = sub_metadatas_.find(name);
  if (sub_ptr == sub_metadatas_.end()) {
    return nullptr;
  }
  return sub_ptr->second.get();
}

Metadata *Metadata::sub_metadata(const std::string &name) {
  auto sub_ptr = sub_metadatas_.find(name);
  if (sub_ptr == sub_metadatas_.end()) {
    return nullptr;
  }
  return sub_ptr->second.get();
}

void Metadata::RemoveEntry(const std::string &name) {
  // Actually just remove "name", no need to check if it exists.
  auto entry_ptr = entries_.find(name);
  if (entry_ptr != entries_.end()) {
    entries_.erase(entry_ptr);
  }
}
}  // namespace draco
