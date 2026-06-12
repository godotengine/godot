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
#ifndef DRACO_METADATA_PROPERTY_TABLE_H_
#define DRACO_METADATA_PROPERTY_TABLE_H_

#include "draco/draco_features.h"

#ifdef DRACO_TRANSCODER_SUPPORTED

#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "draco/core/status_or.h"

namespace draco {

// Describes a property table (properties are table columns) as defined in the
// EXT_structural_metadata glTF extension.
class PropertyTable {
 public:
  // Describes a property (column) of a property table.
  class Property {
   public:
    // Describes glTF buffer view data.
    struct Data {
      // Methods for comparing two data objects.
      bool operator==(const Data &other) const;
      bool operator!=(const Data &other) const { return !(*this == other); }

      // Buffer view data.
      std::vector<uint8_t> data;

      // Data target corresponds to the target property of the glTF bufferView
      // object and classifies the type or nature of the data.
      int target = 0;
    };

    // Describes offsets of the entries in property data when the data
    // represents an array of strings or an array of variable-length number
    // arrays.
    struct Offsets {
      // Methods for comparing two offsets.
      bool operator==(const Offsets &other) const;
      bool operator!=(const Offsets &other) const { return !(*this == other); }

      // Data containing the offset entries.
      Data data;

      // Data type of the offset entries.
      std::string type;

      // Builds a new Offsets object given the offsets in |ints|. The resultant
      // offsets will choose the smallest possible result.type that can contain
      // all of the input |ints|.
      static Offsets MakeFromInts(const std::vector<uint64_t> &ints) {
        uint64_t max_value = 0;
        for (uint64_t i = 0; i < ints.size(); ++i) {
          if (ints[i] > max_value) {
            max_value = ints[i];
          }
        }

        Offsets result;
        int bytes_per_int = 0;
        if (max_value <= std::numeric_limits<uint8_t>::max()) {
          result.type = "UINT8";
          bytes_per_int = 1;
        } else if (max_value <= std::numeric_limits<uint16_t>::max()) {
          result.type = "UINT16";
          bytes_per_int = 2;
        } else if (max_value <= std::numeric_limits<uint32_t>::max()) {
          result.type = "UINT32";
          bytes_per_int = 4;
        } else {
          result.type = "UINT64";
          bytes_per_int = 8;
        }

        result.data.data.resize(ints.size() * bytes_per_int);
        for (uint64_t i = 0; i < ints.size(); ++i) {
          // This assumes execution on a little endian platform.
          memcpy(&result.data.data[i * bytes_per_int], &ints[i], bytes_per_int);
        }

        return result;
      }

      // Decodes the binary data in Offsets::data into offset integers as
      // defined by the EXT_structural_metadata extension. Returns an error if
      // Offsets::type is not one of the types allowed by the spec.
      StatusOr<std::vector<uint64_t>> ParseToInts() const {
        if (data.data.empty()) {
          return std::vector<uint64_t>();
        }

        int bytes_per_int = 0;
        if (type == "UINT8") {
          bytes_per_int = 1;
        } else if (type == "UINT16") {
          bytes_per_int = 2;
        } else if (type == "UINT32") {
          bytes_per_int = 4;
        } else if (type == "UINT64") {
          bytes_per_int = 8;
        } else {
          return Status(Status::DRACO_ERROR, "Offsets data type invalid");
        }

        const int count = data.data.size() / bytes_per_int;
        std::vector<uint64_t> result(count);
        for (int i = 0; i < count; ++i) {
          // This assumes execution on a little endian platform.
          memcpy(&result[i], &data.data[i * bytes_per_int], bytes_per_int);
        }
        return result;
      }
    };

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

    // Property data stores one table column worth of data. For example, when
    // the data of type UINT8 is [11, 22] then the property values are 11 and 22
    // for the first and second table rows. See EXT_structural_metadata glTF
    // extension documentation for more details.
    Data &GetData();
    const Data &GetData() const;

    // Array offsets are used when property data contains a variable-length
    // number arrays. For example, when the data is [0, 1, 2, 3, 4] and the
    // array offsets are [0, 2, 5] for a two-row table, then the property value
    // arrays are [0, 1] and [2, 3, 4] for the first and second table rows,
    // respectively. See EXT_structural_metadata glTF extension documentation
    // for more details.
    const Offsets &GetArrayOffsets() const;
    Offsets &GetArrayOffsets();

    // String offsets are used when property data contains strings. For example,
    // when the data is "SeaLand" and the array offsets are [0, 3, 7] for a
    // two-row table, then the property strings are "Sea"  and "Land" for the
    // first and second table rows, respectively. See EXT_structural_metadata
    // glTF extension documentation for more details.
    const Offsets &GetStringOffsets() const;
    Offsets &GetStringOffsets();

   private:
    std::string name_;
    Data data_;
    Offsets array_offsets_;
    Offsets string_offsets_;
    // TODO(vytyaz): Support property value modifiers min, max, offset, scale.
  };

  // Creates an empty property table.
  PropertyTable();

  // Methods for comparing two property tables.
  bool operator==(const PropertyTable &other) const;
  bool operator!=(const PropertyTable &other) const {
    return !(*this == other);
  }

  // Copies all data from |src| property table.
  void Copy(const PropertyTable &src);

  // Name of this property table.
  void SetName(const std::string &value);
  const std::string &GetName() const;

  // Class of this property table.
  void SetClass(const std::string &value);
  const std::string &GetClass() const;

  // Number of rows in this property table.
  void SetCount(int count);
  int GetCount() const;

  // Table properties (columns).
  int AddProperty(std::unique_ptr<Property> property);
  int NumProperties() const;
  const Property &GetProperty(int index) const;
  Property &GetProperty(int index);
  void RemoveProperty(int index);

 private:
  std::string name_;
  std::string class_;
  int count_;
  std::vector<std::unique_ptr<Property>> properties_;
};

}  // namespace draco

#endif  // DRACO_TRANSCODER_SUPPORTED
#endif  // DRACO_METADATA_PROPERTY_TABLE_H_
