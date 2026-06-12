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
#include "draco/metadata/structural_metadata.h"

#include <memory>
#include <utility>

#ifdef DRACO_TRANSCODER_SUPPORTED

namespace draco {

// Returns true if vectors |a| and |b| have the same size and their entries
// (unique pointers) point to objects that compare equally.
template <typename T>
bool VectorsAreEqual(const std::vector<std::unique_ptr<T>> &a,
                     const std::vector<std::unique_ptr<T>> &b) {
  if (a.size() != b.size()) {
    return false;
  }
  for (int i = 0; i < a.size(); ++i) {
    if (*a[i] != *b[i]) {
      return false;
    }
  }
  return true;
}

bool StructuralMetadata::operator==(const StructuralMetadata &other) const {
  return schema_ == other.schema_ &&
         VectorsAreEqual(property_tables_, other.property_tables_) &&
         VectorsAreEqual(property_attributes_, other.property_attributes_);
}

void StructuralMetadata::Copy(const StructuralMetadata &src) {
  // Copy schema.
  schema_.json.Copy(src.schema_.json);

  // Copy property tables.
  property_tables_.resize(src.property_tables_.size());
  for (int i = 0; i < property_tables_.size(); ++i) {
    property_tables_[i] = std::unique_ptr<PropertyTable>(new PropertyTable());
    property_tables_[i]->Copy(*src.property_tables_[i]);
  }

  // Copy property attributes.
  property_attributes_.resize(src.property_attributes_.size());
  for (int i = 0; i < property_attributes_.size(); ++i) {
    property_attributes_[i] =
        std::unique_ptr<PropertyAttribute>(new PropertyAttribute());
    property_attributes_[i]->Copy(*src.property_attributes_[i]);
  }
}

void StructuralMetadata::SetSchema(const StructuralMetadataSchema &schema) {
  schema_ = schema;
}

const StructuralMetadataSchema &StructuralMetadata::GetSchema() const {
  return schema_;
}

int StructuralMetadata::AddPropertyTable(
    std::unique_ptr<PropertyTable> property_table) {
  property_tables_.push_back(std::move(property_table));
  return property_tables_.size() - 1;
}

int StructuralMetadata::NumPropertyTables() const {
  return property_tables_.size();
}

const PropertyTable &StructuralMetadata::GetPropertyTable(int index) const {
  return *property_tables_[index];
}

PropertyTable &StructuralMetadata::GetPropertyTable(int index) {
  return *property_tables_[index];
}

void StructuralMetadata::RemovePropertyTable(int index) {
  property_tables_.erase(property_tables_.begin() + index);
}

int StructuralMetadata::AddPropertyAttribute(
    std::unique_ptr<PropertyAttribute> property_attribute) {
  property_attributes_.push_back(std::move(property_attribute));
  return property_attributes_.size() - 1;
}

int StructuralMetadata::NumPropertyAttributes() const {
  return property_attributes_.size();
}

const PropertyAttribute &StructuralMetadata::GetPropertyAttribute(
    int index) const {
  return *property_attributes_[index];
}

PropertyAttribute &StructuralMetadata::GetPropertyAttribute(int index) {
  return *property_attributes_[index];
}

void StructuralMetadata::RemovePropertyAttribute(int index) {
  property_attributes_.erase(property_attributes_.begin() + index);
}

}  // namespace draco

#endif  // DRACO_TRANSCODER_SUPPORTED
