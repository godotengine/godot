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
#include "draco/core/draco_types.h"

namespace draco {

int32_t DataTypeLength(DataType dt) {
  switch (dt) {
    case DT_INT8:
    case DT_UINT8:
      return 1;
    case DT_INT16:
    case DT_UINT16:
      return 2;
    case DT_INT32:
    case DT_UINT32:
      return 4;
    case DT_INT64:
    case DT_UINT64:
      return 8;
    case DT_FLOAT32:
      return 4;
    case DT_FLOAT64:
      return 8;
    case DT_BOOL:
      return 1;
    default:
      return -1;
  }
}

bool IsDataTypeIntegral(DataType dt) {
  switch (dt) {
    case DT_INT8:
    case DT_UINT8:
    case DT_INT16:
    case DT_UINT16:
    case DT_INT32:
    case DT_UINT32:
    case DT_INT64:
    case DT_UINT64:
    case DT_BOOL:
      return true;
    default:
      return false;
  }
}

}  // namespace draco
