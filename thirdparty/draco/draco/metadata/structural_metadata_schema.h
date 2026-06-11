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
#ifndef DRACO_METADATA_SCHEMA_H_
#define DRACO_METADATA_SCHEMA_H_

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

// Defines schema that describes the structure of the metadata as defined in the
// EXT_structural_metadata glTF extension, in the form of a JSON object.
struct StructuralMetadataSchema {
  // JSON object of the schema.
  // TODO(vytyaz): Consider using a third_party/json library. Currently there
  // is a conflict between Filament's assert_invariant() macro and JSON
  // library's assert_invariant() method that causes compile errors in Draco
  // visualization library.
  class Object {
   public:
    enum Type { OBJECT, ARRAY, STRING, INTEGER, BOOLEAN };

    // Constructors.
    Object();
    explicit Object(const std::string &name);
    Object(const std::string &name, const std::string &value);
    Object(const std::string &name, const char *value);
    Object(const std::string &name, int value);
    Object(const std::string &name, bool value);

    // Methods for comparing two objects.
    bool operator==(const Object &other) const;
    bool operator!=(const Object &other) const;

    // Method for copying the object.
    void Copy(const Object &src);

    // Methods for getting object name and type.
    const std::string &GetName() const { return name_; }
    Type GetType() const { return type_; }

    // Methods for getting object value.
    const std::vector<Object> &GetObjects() const { return objects_; }
    const std::vector<Object> &GetArray() const { return array_; }
    const std::string &GetString() const { return string_; }
    int GetInteger() const { return integer_; }
    bool GetBoolean() const { return boolean_; }

    // Looks for a child object matching the given |name|. If no object is
    // found, returns nullptr.
    //
    // Note that this is not recursive. I.e., for the following object:
    //
    // { "level1": { "level2": "value" } }
    //
    // GetObjectByName("level1") will return '{ "level2": "value" }', but
    // GetObjectByName("level2") will return nullptr. Instead, the user should
    // use GetObjectByName("level1")->GetObjectByName("level2") to get the
    // nested child. Note that this follows the typical JSON semantics.
    const Object *GetObjectByName(const std::string &name) const;

    // Methods for setting object value.
    std::vector<Object> &SetObjects();
    std::vector<Object> &SetArray();
    void SetString(const std::string &value);
    void SetInteger(int value);
    void SetBoolean(bool value);

   private:
    std::string name_;
    Type type_;
    std::vector<Object> objects_;
    std::vector<Object> array_;
    std::string string_;
    int integer_;
    bool boolean_;
  };

  // Valid schema top-level JSON object name is "schema".
  StructuralMetadataSchema() : json("schema") {}

  // Methods for comparing two schemas.
  bool operator==(const StructuralMetadataSchema &other) const;
  bool operator!=(const StructuralMetadataSchema &other) const;

  // Valid schema top-level JSON object is required to have child objects.
  bool Empty() const { return json.GetObjects().empty(); }

  // Top-level JSON object of the schema.
  Object json;
};

}  // namespace draco

#endif  // DRACO_TRANSCODER_SUPPORTED
#endif  // DRACO_METADATA_SCHEMA_H_
