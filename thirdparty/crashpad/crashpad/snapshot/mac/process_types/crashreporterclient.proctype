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

// The name of this file was chosen based on
// https://llvm.org/svn/llvm-project/llvm/trunk/lib/Support/PrettyStackTrace.cpp.
// The name of the structure it describes was chosen based on that file as well
// as 10.9.2 cups-372.2/cups/backend/usb-darwin.c. That file also provided the
// names and types of the fields in the structure.
//
// This file is intended to be included multiple times in the same translation
// unit, so #include guards are intentionally absent.
//
// This file is included by snapshot/mac/process_types.h and
// snapshot/mac/process_types.cc to produce process type struct definitions and
// accessors.

// Client Mach-O images will contain a __DATA,__crash_info section formatted
// according to this structure.
//
// crashreporter_annotations_t is variable-length. Its length dictated by its
// |version| field which is always present. A custom implementation of the
// flavored ReadSpecificInto function that understands how to map this field to
// the structure’s actual size is provided in
// snapshot/mac/process_types/custom.cc. No implementation of ReadArrayInto is
// provided because crashreporter_annotations_t structs are singletons in a
// module and are never present in arrays, so the functionality is unnecessary.

#if !defined(PROCESS_TYPE_STRUCT_IMPLEMENT_INTERNAL_READ_INTO) && \
    !defined(PROCESS_TYPE_STRUCT_IMPLEMENT_ARRAY)

PROCESS_TYPE_STRUCT_BEGIN(crashreporter_annotations_t)
  PROCESS_TYPE_STRUCT_MEMBER(uint64_t, version)  // unsigned long
  PROCESS_TYPE_STRUCT_VERSIONED(crashreporter_annotations_t, version)

  // Version 4 (OS X 10.7)
  PROCESS_TYPE_STRUCT_MEMBER(uint64_t, message)  // char*
  PROCESS_TYPE_STRUCT_MEMBER(uint64_t, signature_string)  // char*
  PROCESS_TYPE_STRUCT_MEMBER(uint64_t, backtrace)  // char*
  PROCESS_TYPE_STRUCT_MEMBER(uint64_t, message2)  // char*
  PROCESS_TYPE_STRUCT_MEMBER(uint64_t, thread)
  PROCESS_TYPE_STRUCT_MEMBER(uint64_t, dialog_mode)  // unsigned int

  // Version 5 (OS X 10.11)
  PROCESS_TYPE_STRUCT_MEMBER(uint64_t, unknown_0)
PROCESS_TYPE_STRUCT_END(crashreporter_annotations_t)

#endif  // ! PROCESS_TYPE_STRUCT_IMPLEMENT_INTERNAL_READ_INTO &&
        // ! PROCESS_TYPE_STRUCT_IMPLEMENT_ARRAY
