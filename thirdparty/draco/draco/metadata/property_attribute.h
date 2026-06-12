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
#ifndef DRACO_METADATA_PROPERTY_ATTRIBUTE_H_
#define DRACO_METADATA_PROPERTY_ATTRIBUTE_H_

#include "draco/draco_features.h"

#ifdef DRACO_TRANSCODER_SUPPORTED

#include <memory>
#include <string>
#include <vector>

#include "draco/core/status_or.h"

namespace draco {

// Describes a property attribute as defined in the EXT_structural_metadata glTF
// extension.
class PropertyAttribute {
 public:
  // Describes where property is stored (as an attribute).
  class Property {
   public:
    // Creates an empty property.
    Property() = default;

    // Methods for comparing two properties.
    bool operator==(const Property &other) const;
    bool operator!=(const Property &other) const { return !(*this == other); }

    // Copies all data from |src| property.
    void Copy(const Property &src);

    // Name of this property.
    void SetName(const std::string &name);
    const std::string &GetName() const;

    // Name of glTF attribute containing property values, like "_DIRECTION".
    void SetAttributeName(const std::string &name);
    const std::string &GetAttributeName() const;

   private:
    // Name of this property as in structural metadata schema class property.
    std::string name_;

    // Name of glTF attribute containing property values, like "_DIRECTION".
    std::string attribute_name_;

    // TODO(vytyaz): Support property value modifiers min, max, offset, scale.
  };

  // Creates an empty property attribute.
  PropertyAttribute() = default;

  // Methods for comparing two property attributes.
  bool operator==(const PropertyAttribute &other) const;
  bool operator!=(const PropertyAttribute &other) const {
    return !(*this == other);
  }

  // Copies all data from |src| property attribute.
  void Copy(const PropertyAttribute &src);

  // Name of this property attribute.
  void SetName(const std::string &value);
  const std::string &GetName() const;

  // Class of this property attribute.
  void SetClass(const std::string &value);
  const std::string &GetClass() const;

  // Properties.
  int AddProperty(std::unique_ptr<Property> property);
  int NumProperties() const;
  const Property &GetProperty(int index) const;
  Property &GetProperty(int index);
  void RemoveProperty(int index);

 private:
  // The name of the property attribute, e.g., for display purposes.
  std::string name_;

  // The class in structural metadata schema that property values conform to.
  std::string class_;

  // Properties corresponding to schema class properties, describing where the
  // property values are stored (as attributes).
  std::vector<std::unique_ptr<Property>> properties_;
};

}  // namespace draco

#endif  // DRACO_TRANSCODER_SUPPORTED
#endif  // DRACO_METADATA_PROPERTY_ATTRIBUTE_H_
