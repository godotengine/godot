// Copyright 2014 Google LLC
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

#ifndef CLIENT_LINUX_MINIDUMP_WRITER_MICRODUMP_WRITER_H_
#define CLIENT_LINUX_MINIDUMP_WRITER_MICRODUMP_WRITER_H_

#include <stdint.h>
#include <sys/types.h>

#include "client/linux/dump_writer_common/mapping_info.h"

namespace google_breakpad {

struct MicrodumpExtraInfo;

// Writes a microdump (a reduced dump containing only the state of the crashing
// thread) on the console (logcat on Android). These functions do not malloc nor
// use libc functions which may. Thus, it can be used in contexts where the
// state of the heap may be corrupt.
// Args:
//   crashing_process: the pid of the crashing process. This must be trusted.
//   blob: a blob of data from the crashing process. See exception_handler.h
//   blob_size: the length of |blob| in bytes.
//   mappings: a list of additional mappings provided by the application.
//   build_fingerprint: a (optional) C string which determines the OS
//     build fingerprint (e.g., aosp/occam/mako:5.1.1/LMY47W/1234:eng/dev-keys).
//   product_info: a (optional) C string which determines the product name and
//     version (e.g., WebView:42.0.2311.136).
//
// Returns true iff successful.
bool WriteMicrodump(pid_t crashing_process,
                    const void* blob,
                    size_t blob_size,
                    const MappingList& mappings,
                    bool skip_dump_if_main_module_not_referenced,
                    uintptr_t address_within_main_module,
                    bool sanitize_stack,
                    const MicrodumpExtraInfo& microdump_extra_info);

}  // namespace google_breakpad

#endif  // CLIENT_LINUX_MINIDUMP_WRITER_MICRODUMP_WRITER_H_
