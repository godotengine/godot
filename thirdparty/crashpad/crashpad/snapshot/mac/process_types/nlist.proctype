// Copyright 2014 The Crashpad Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This file corresponds to the system’s <mach-o/nlist.h>.
//
// This file is intended to be included multiple times in the same translation
// unit, so #include guards are intentionally absent.
//
// This file is included by snapshot/mac/process_types.h and
// snapshot/mac/process_types.cc to produce process type struct definitions and
// accessors.

PROCESS_TYPE_STRUCT_BEGIN(nlist)
  PROCESS_TYPE_STRUCT_MEMBER(uint32_t, n_strx)  // n_un.n_strx
  PROCESS_TYPE_STRUCT_MEMBER(uint8_t, n_type)
  PROCESS_TYPE_STRUCT_MEMBER(uint8_t, n_sect)
  PROCESS_TYPE_STRUCT_MEMBER(uint16_t, n_desc)  // 32-bit: int16_t
  PROCESS_TYPE_STRUCT_MEMBER(ULong, n_value)
PROCESS_TYPE_STRUCT_END(nlist)
