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

// The file corresponds to Crashpad’s client/crashpad_info.h.
//
// This file is intended to be included multiple times in the same translation
// unit, so #include guards are intentionally absent.
//
// This file is included by snapshot/mac/process_types.h and
// snapshot/mac/process_types.cc to produce process type struct definitions and
// accessors.

// Client Mach-O images will contain a __DATA,crashpad_info section formatted
// according to this structure.
//
// CrashpadInfo is variable-length. Its length dictated by its |size| field
// which is always present. A custom implementation of the flavored
// ReadSpecificInto function that understands how to use this field is provided
// in snapshot/mac/process_types/custom.cc. No implementation of ReadArrayInto
// is provided because CrashpadInfo structs are singletons in a module and are
// never present in arrays, so the functionality is unnecessary.

#if !defined(PROCESS_TYPE_STRUCT_IMPLEMENT_INTERNAL_READ_INTO) && \
    !defined(PROCESS_TYPE_STRUCT_IMPLEMENT_ARRAY)

PROCESS_TYPE_STRUCT_BEGIN(CrashpadInfo)
  PROCESS_TYPE_STRUCT_MEMBER(uint32_t, signature)

  PROCESS_TYPE_STRUCT_MEMBER(uint32_t, size)
  PROCESS_TYPE_STRUCT_SIZED(CrashpadInfo, size)

  PROCESS_TYPE_STRUCT_MEMBER(uint32_t, version)

  PROCESS_TYPE_STRUCT_MEMBER(uint32_t, indirectly_referenced_memory_cap)
  PROCESS_TYPE_STRUCT_MEMBER(uint32_t, padding_0)
  PROCESS_TYPE_STRUCT_MEMBER(uint8_t, crashpad_handler_behavior)  // TriState

  // TriState
  PROCESS_TYPE_STRUCT_MEMBER(uint8_t, system_crash_reporter_forwarding)

  // TriState
  PROCESS_TYPE_STRUCT_MEMBER(uint8_t, gather_indirectly_referenced_memory)

  PROCESS_TYPE_STRUCT_MEMBER(uint8_t, padding_1)

  // SimpleAddressRangeBag*
  PROCESS_TYPE_STRUCT_MEMBER(Pointer, extra_memory_ranges)

  // SimpleStringDictionary*
  PROCESS_TYPE_STRUCT_MEMBER(Pointer, simple_annotations)

  // UserDataStreamListEntry*
  PROCESS_TYPE_STRUCT_MEMBER(Pointer, user_data_minidump_stream_head)

  // AnnotationList*
  PROCESS_TYPE_STRUCT_MEMBER(Pointer, annotations_list)
PROCESS_TYPE_STRUCT_END(CrashpadInfo)

#endif  // ! PROCESS_TYPE_STRUCT_IMPLEMENT_INTERNAL_READ_INTO &&
        // ! PROCESS_TYPE_STRUCT_IMPLEMENT_ARRAY
