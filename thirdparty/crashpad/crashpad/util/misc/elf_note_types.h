// Copyright 2018 The Crashpad Authors. All rights reserved.
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

#ifndef CRASHPAD_UTIL_MISC_ELF_NOTE_TYPES_H_
#define CRASHPAD_UTIL_MISC_ELF_NOTE_TYPES_H_

// This header defines types of ELF "notes" that are embedded sections. These
// can be read by ElfImageReader in the snapshot library, and are created in
// client modules. All notes used by Crashpad use the name "Crashpad" and one of
// the types defined here. Note that this file is #included into .S files, so
// must be relatively plain (no C++ features).

#define CRASHPAD_ELF_NOTE_NAME "Crashpad"

// Used by ElfImageReader for testing purposes.
#define CRASHPAD_ELF_NOTE_TYPE_SNAPSHOT_TEST 1

// Used by the client library to stash a pointer to the CrashpadInfo structure
// for retrieval by the module snapshot. 'OFNI' == 0x4f464e49 which appears as
// "INFO" in readelf -x.
#define CRASHPAD_ELF_NOTE_TYPE_CRASHPAD_INFO 0x4f464e49

#endif  // CRASHPAD_UTIL_MISC_ELF_NOTE_TYPES_H_
