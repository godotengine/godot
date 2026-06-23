// Copyright 2022 The Draco Authors.
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
#include "draco/metadata/property_table.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifdef DRACO_TRANSCODER_SUPPORTED

namespace draco {

bool PropertyTable::Property::Data::operator==(const Data &other) const {
  return data == other.data && target == other.target;
}

bool PropertyTable::Property::Offsets::operator==(const Offsets &other) const {
  return data == other.data && type == other.type;
}

bool PropertyTable::Property::operator==(const Property &other) const {
  return name_ == other.name_ && data_ == other.data_ &&
         array_offsets_ == other.array_offsets_ &&
         string_offsets_ == other.string_offsets_;
}

void PropertyTable::Property::Copy(const Property &src) {
  name_ = src.name_;
  data_ = src.data_;
  array_offsets_ = src.array_offsets_;
  string_offsets_ = src.string_offsets_;
}

void PropertyTable::Property::SetName(const std::string &name) { name_ = name; }
const std::string &PropertyTable::Property::GetName() const { return name_; }

PropertyTable::Property::Data &PropertyTable::Property::GetData() {
  return data_;
}
const PropertyTable::Property::Data &PropertyTable::Property::GetData() const {
  return data_;
}

const PropertyTable::Property::Offsets &
PropertyTable::Property::GetArrayOffsets() const {
  return array_offsets_;
}
PropertyTable::Property::Offsets &PropertyTable::Property::GetArrayOffsets() {
  return array_offsets_;
}

const PropertyTable::Property::Offsets &
PropertyTable::Property::GetStringOffsets() const {
  return string_offsets_;
}
PropertyTable::Property::Offsets &PropertyTable::Property::GetStringOffsets() {
  return string_offsets_;
}

PropertyTable::PropertyTable() : count_(0) {}

bool PropertyTable::operator==(const PropertyTable &other) const {
  if (name_ != other.name_ || class_ != other.class_ ||
      count_ != other.count_ ||
      properties_.size() != other.properties_.size()) {
    return false;
  }
  for (int i = 0; i < properties_.size(); ++i) {
    if (*properties_[i] != *other.properties_[i]) {
      return false;
    }
  }
  return true;
}

void PropertyTable::Copy(const PropertyTable &src) {
  name_ = src.name_;
  class_ = src.class_;
  count_ = src.count_;
  properties_.clear();
  properties_.reserve(src.properties_.size());
  for (int i = 0; i < src.properties_.size(); ++i) {
    std::unique_ptr<Property> property(new Property());
    property->Copy(src.GetProperty(i));
    properties_.push_back(std::move(property));
  }
}

void PropertyTable::SetName(const std::string &value) { name_ = value; }
const std::string &PropertyTable::GetName() const { return name_; }

void PropertyTable::SetClass(const std::string &value) { class_ = value; }
const std::string &PropertyTable::GetClass() const { return class_; }

void PropertyTable::SetCount(int count) { count_ = count; }
int PropertyTable::GetCount() const { return count_; }

int PropertyTable::AddProperty(std::unique_ptr<Property> property) {
  properties_.push_back(std::move(property));
  return properties_.size() - 1;
}
int PropertyTable::NumProperties() const { return properties_.size(); }
const PropertyTable::Property &PropertyTable::GetProperty(int index) const {
  return *properties_[index];
}
PropertyTable::Property &PropertyTable::GetProperty(int index) {
  return *properties_[index];
}
void PropertyTable::RemoveProperty(int index) {
  properties_.erase(properties_.begin() + index);
}

}  // namespace draco

#endif  // DRACO_TRANSCODER_SUPPORTED
