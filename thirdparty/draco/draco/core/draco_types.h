// Copyright 2016 The Draco Authors.
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
#ifndef DRACO_CORE_DRACO_TYPES_H_
#define DRACO_CORE_DRACO_TYPES_H_

#include <stdint.h>

#include <string>

#include "draco/draco_features.h"

namespace draco {

enum DataType {
  // Not a legal value for DataType. Used to indicate a field has not been set.
  DT_INVALID = 0,
  DT_INT8,
  DT_UINT8,
  DT_INT16,
  DT_UINT16,
  DT_INT32,
  DT_UINT32,
  DT_INT64,
  DT_UINT64,
  DT_FLOAT32,
  DT_FLOAT64,
  DT_BOOL,
  DT_TYPES_COUNT
};

int32_t DataTypeLength(DataType dt);

// Equivalent to std::is_integral for draco::DataType. Returns true for all
// signed and unsigned integer types (including DT_BOOL). Returns false
// otherwise.
bool IsDataTypeIntegral(DataType dt);

}  // namespace draco

#endif  // DRACO_CORE_DRACO_TYPES_H_
