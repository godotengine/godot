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
#include "draco/metadata/structural_metadata_schema.h"

#include <string>
#include <vector>

#ifdef DRACO_TRANSCODER_SUPPORTED

namespace draco {

StructuralMetadataSchema::Object::Object() : Object("") {}

StructuralMetadataSchema::Object::Object(const std::string &name)
    : name_(name), type_(OBJECT), integer_(0), boolean_(false) {}

StructuralMetadataSchema::Object::Object(const std::string &name,
                                         const std::string &value)
    : Object(name) {
  SetString(value);
}

StructuralMetadataSchema::Object::Object(const std::string &name,
                                         const char *value)
    : Object(name) {
  SetString(value);
}

StructuralMetadataSchema::Object::Object(const std::string &name, int value)
    : Object(name) {
  SetInteger(value);
}

StructuralMetadataSchema::Object::Object(const std::string &name, bool value)
    : Object(name) {
  SetBoolean(value);
}

bool StructuralMetadataSchema::Object::operator==(const Object &other) const {
  if (type_ != other.type_ || name_ != other.name_) {
    return false;
  }
  switch (type_) {
    case OBJECT:
      if (objects_.size() != other.objects_.size()) {
        return false;
      }
      for (int i = 0; i < objects_.size(); ++i) {
        if (objects_[i] != other.objects_[i]) {
          return false;
        }
      }
      break;
    case ARRAY:
      if (array_.size() != other.array_.size()) {
        return false;
      }
      for (int i = 0; i < array_.size(); ++i) {
        if (array_[i] != other.array_[i]) {
          return false;
        }
      }
      break;
    case STRING:
      return string_ == other.string_;
    case INTEGER:
      return integer_ == other.integer_;
    case BOOLEAN:
      return boolean_ == other.boolean_;
  }
  return true;
}

bool StructuralMetadataSchema::Object::operator!=(const Object &other) const {
  return !(*this == other);
}

void StructuralMetadataSchema::Object::Copy(const Object &src) {
  name_ = src.name_;
  type_ = src.type_;
  objects_.reserve(src.objects_.size());
  for (const Object &obj : src.objects_) {
    objects_.emplace_back();
    objects_.back().Copy(obj);
  }
  array_.reserve(src.array_.size());
  for (const Object &obj : src.array_) {
    array_.emplace_back();
    array_.back().Copy(obj);
  }
  string_ = src.string_;
  integer_ = src.integer_;
  boolean_ = src.boolean_;
}

const StructuralMetadataSchema::Object *
StructuralMetadataSchema::Object::GetObjectByName(
    const std::string &name) const {
  for (const Object &obj : objects_) {
    if (obj.GetName() == name) {
      return &obj;
    }
  }
  return nullptr;
}

std::vector<StructuralMetadataSchema::Object> &
StructuralMetadataSchema::Object::SetObjects() {
  type_ = OBJECT;
  return objects_;
}

std::vector<StructuralMetadataSchema::Object> &
StructuralMetadataSchema::Object::SetArray() {
  type_ = ARRAY;
  return array_;
}

void StructuralMetadataSchema::Object::SetString(const std::string &value) {
  type_ = STRING;
  string_ = value;
}

void StructuralMetadataSchema::Object::SetInteger(int value) {
  type_ = INTEGER;
  integer_ = value;
}

void StructuralMetadataSchema::Object::SetBoolean(bool value) {
  type_ = BOOLEAN;
  boolean_ = value;
}

bool StructuralMetadataSchema::operator==(
    const StructuralMetadataSchema &other) const {
  return json == other.json;
}

bool StructuralMetadataSchema::operator!=(
    const StructuralMetadataSchema &other) const {
  return !(*this == other);
}

}  // namespace draco

#endif  // DRACO_TRANSCODER_SUPPORTED
