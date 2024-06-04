// Copyright (c) 2012, Google Inc.
// All rights reserved.
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
//     * Neither the name of Google Inc. nor the names of its
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

// ios_exception_minidump_generator.h:  Create a fake minidump from a
// NSException.

#ifndef CLIENT_IOS_HANDLER_IOS_EXCEPTION_MINIDUMP_GENERATOR_H_
#define CLIENT_IOS_HANDLER_IOS_EXCEPTION_MINIDUMP_GENERATOR_H_

#include <Foundation/Foundation.h>

#include "client/mac/handler/minidump_generator.h"

namespace google_breakpad {

class IosExceptionMinidumpGenerator : public MinidumpGenerator {
 public:
  explicit IosExceptionMinidumpGenerator(NSException* exception);
  virtual ~IosExceptionMinidumpGenerator();

 protected:
  virtual bool WriteExceptionStream(MDRawDirectory* exception_stream);
  virtual bool WriteThreadStream(mach_port_t thread_id, MDRawThread* thread);

 private:

  // Get the crashing program counter from the exception.
  uintptr_t GetPCFromException();

  // Get the crashing link register from the exception.
  uintptr_t GetLRFromException();

  // Write a virtual thread context for the crashing site.
  bool WriteCrashingContext(MDLocationDescriptor* register_location);
  // Per-CPU implementations of the above method.
#ifdef HAS_ARM_SUPPORT
  bool WriteCrashingContextARM(MDLocationDescriptor* register_location);
#endif
#ifdef HAS_ARM64_SUPPORT
  bool WriteCrashingContextARM64(MDLocationDescriptor* register_location);
#endif

  NSArray* return_addresses_;
};

}  // namespace google_breakpad

#endif  // CLIENT_IOS_HANDLER_IOS_EXCEPTION_MINIDUMP_GENERATOR_H_
