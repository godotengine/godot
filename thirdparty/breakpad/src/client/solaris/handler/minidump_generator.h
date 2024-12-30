// Copyright 2007 Google LLC
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

// Author: Alfred Peng

#ifndef CLIENT_SOLARIS_HANDLER_MINIDUMP_GENERATOR_H__
#define CLIENT_SOLARIS_HANDLER_MINIDUMP_GENERATOR_H__

#include <ucontext.h>

#include "client/minidump_file_writer.h"
#include "client/solaris/handler/solaris_lwp.h"
#include "google_breakpad/common/breakpad_types.h"
#include "google_breakpad/common/minidump_format.h"

namespace google_breakpad {

//
// MinidumpGenerator
//
// A minidump generator should be created before any exception happen.
//
class MinidumpGenerator {
  // Callback run for writing lwp information in the process.
  friend bool LwpInformationCallback(lwpstatus_t* lsp, void* context);

  // Callback run for writing module information in the process.
  friend bool ModuleInfoCallback(const ModuleInfo& module_info, void* context);

 public:
  MinidumpGenerator();

  ~MinidumpGenerator();

  // Write minidump.
  bool WriteMinidumpToFile(const char* file_pathname,
                           int signo,
                           uintptr_t sighandler_ebp,
                           ucontext_t** sig_ctx) const;
};

}  // namespace google_breakpad

#endif   // CLIENT_SOLARIS_HANDLER_MINIDUMP_GENERATOR_H_
