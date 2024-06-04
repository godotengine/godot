// Copyright (c) 2012 Google Inc.
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

#include "client/linux/log/log.h"

#if defined(__ANDROID__)
#include <android/log.h>
#include <dlfcn.h>
#else
#include "third_party/lss/linux_syscall_support.h"
#endif

namespace logger {

#if defined(__ANDROID__)
namespace {

// __android_log_buf_write() is not exported in the NDK and is being used by
// dynamic runtime linking. Its declaration is taken from Android's
// system/core/include/log/log.h.
using AndroidLogBufferWriteFunc = int (*)(int bufID, int prio, const char* tag,
                                          const char* text);
const int kAndroidCrashLogId = 4;  // From LOG_ID_CRASH in log.h.
const char kAndroidLogTag[] = "google-breakpad";

bool g_crash_log_initialized = false;
AndroidLogBufferWriteFunc g_android_log_buf_write = nullptr;

}  // namespace

void initializeCrashLogWriter() {
  if (g_crash_log_initialized)
    return;
  g_android_log_buf_write = reinterpret_cast<AndroidLogBufferWriteFunc>(
      dlsym(RTLD_DEFAULT, "__android_log_buf_write"));
  g_crash_log_initialized = true;
}

int writeToCrashLog(const char* buf) {
  // Try writing to the crash log ring buffer. If not available, fall back to
  // the standard log buffer.
  if (g_android_log_buf_write) {
    return g_android_log_buf_write(kAndroidCrashLogId, ANDROID_LOG_FATAL,
                                   kAndroidLogTag, buf);
  }
  return __android_log_write(ANDROID_LOG_FATAL, kAndroidLogTag, buf);
}
#endif

int write(const char* buf, size_t nbytes) {
#if defined(__ANDROID__)
  return __android_log_write(ANDROID_LOG_WARN, kAndroidLogTag, buf);
#else
  return sys_write(2, buf, nbytes);
#endif
}

}  // namespace logger
