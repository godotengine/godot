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
#ifndef DRACO_METADATA_STRUCTURAL_METADATA_H_
#define DRACO_METADATA_STRUCTURAL_METADATA_H_

#include "draco/draco_features.h"

#ifdef DRACO_TRANSCODER_SUPPORTED

#include <memory>
#include <vector>

#include "draco/metadata/property_attribute.h"
#include "draco/metadata/property_table.h"
#include "draco/metadata/structural_metadata_schema.h"

namespace draco {

// Holds data associated with EXT_structural_metadata glTF extension.
class StructuralMetadata {
 public:
  StructuralMetadata() = default;

  // Methods for comparing two structural metadata objects.
  bool operator==(const StructuralMetadata &other) const;
  bool operator!=(const StructuralMetadata &other) const {
    return !(*this == other);
  }

  // Copies |src| structural metadata into this object.
  void Copy(const StructuralMetadata &src);

  // Schema of the structural metadata.
  void SetSchema(const StructuralMetadataSchema &schema);
  const StructuralMetadataSchema &GetSchema() const;

  // Property tables.
  int AddPropertyTable(std::unique_ptr<PropertyTable> property_table);
  int NumPropertyTables() const;
  const PropertyTable &GetPropertyTable(int index) const;
  PropertyTable &GetPropertyTable(int index);
  void RemovePropertyTable(int index);

  // Property attributes.
  int AddPropertyAttribute(
      std::unique_ptr<PropertyAttribute> property_attribute);
  int NumPropertyAttributes() const;
  const PropertyAttribute &GetPropertyAttribute(int index) const;
  PropertyAttribute &GetPropertyAttribute(int index);
  void RemovePropertyAttribute(int index);

 private:
  // Schema of the structural metadata.
  StructuralMetadataSchema schema_;

  // Property tables.
  std::vector<std::unique_ptr<PropertyTable>> property_tables_;

  // Property attributes.
  std::vector<std::unique_ptr<PropertyAttribute>> property_attributes_;
};

}  // namespace draco

#endif  // DRACO_TRANSCODER_SUPPORTED
#endif  // DRACO_METADATA_STRUCTURAL_METADATA_H_
