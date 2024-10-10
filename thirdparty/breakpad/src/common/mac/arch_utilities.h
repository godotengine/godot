// Copyright 2012 Google LLC
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google LLC nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// arch_utilities.h: Utilities for architecture introspection for Mac platform.

#ifndef COMMON_MAC_ARCH_UTILITIES_H__
#define COMMON_MAC_ARCH_UTILITIES_H__

#include <mach/machine.h>

#include <optional>

static constexpr const char* kUnknownArchName = "<Unknown architecture>";

struct ArchInfo {
  cpu_type_t cputype;
  cpu_subtype_t cpusubtype;
};

// Returns architecture info if `arch_name` corresponds to a valid, known
// architecture, and std::nullopt otherwise.
std::optional<ArchInfo> GetArchInfoFromName(const char* arch_name);

// Returns the name of the architecture specified by `cpu_type` and
// `cpu_subtype`, or `kUnknownArchName` if it's unknown or invalid.
const char* GetNameFromCPUType(cpu_type_t cpu_type, cpu_subtype_t cpu_subtype);

// Returns the architecture of the machine this code is running on.
ArchInfo GetLocalArchInfo();

#endif  // COMMON_MAC_ARCH_UTILITIES_H__
