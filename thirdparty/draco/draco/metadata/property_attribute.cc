// Copyright 2023 The Draco Authors.
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
#include "draco/metadata/property_attribute.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifdef DRACO_TRANSCODER_SUPPORTED

namespace draco {

bool PropertyAttribute::Property::operator==(const Property &other) const {
  return name_ == other.name_ && attribute_name_ == other.attribute_name_;
}

void PropertyAttribute::Property::Copy(const Property &src) {
  name_ = src.name_;
  attribute_name_ = src.attribute_name_;
}

void PropertyAttribute::Property::SetName(const std::string &name) {
  name_ = name;
}

const std::string &PropertyAttribute::Property::GetName() const {
  return name_;
}

void PropertyAttribute::Property::SetAttributeName(const std::string &name) {
  attribute_name_ = name;
}

const std::string &PropertyAttribute::Property::GetAttributeName() const {
  return attribute_name_;
}

bool PropertyAttribute::operator==(const PropertyAttribute &other) const {
  if (name_ != other.name_ || class_ != other.class_ ||
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

void PropertyAttribute::Copy(const PropertyAttribute &src) {
  name_ = src.name_;
  class_ = src.class_;
  properties_.clear();
  properties_.reserve(src.properties_.size());
  for (int i = 0; i < src.properties_.size(); ++i) {
    std::unique_ptr<Property> property(new Property());
    property->Copy(src.GetProperty(i));
    properties_.push_back(std::move(property));
  }
}

void PropertyAttribute::SetName(const std::string &value) { name_ = value; }

const std::string &PropertyAttribute::GetName() const { return name_; }

void PropertyAttribute::SetClass(const std::string &value) { class_ = value; }

const std::string &PropertyAttribute::GetClass() const { return class_; }

int PropertyAttribute::AddProperty(std::unique_ptr<Property> property) {
  properties_.push_back(std::move(property));
  return properties_.size() - 1;
}

int PropertyAttribute::NumProperties() const { return properties_.size(); }

const PropertyAttribute::Property &PropertyAttribute::GetProperty(
    int index) const {
  return *properties_[index];
}

PropertyAttribute::Property &PropertyAttribute::GetProperty(int index) {
  return *properties_[index];
}

void PropertyAttribute::RemoveProperty(int index) {
  properties_.erase(properties_.begin() + index);
}

}  // namespace draco

#endif  // DRACO_TRANSCODER_SUPPORTED
