// Copyright 2017 The Crashpad Authors. All rights reserved.
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

#include "dlfcn_internal.h"

#include <android/api-level.h>
#include <dlfcn.h>
#include <errno.h>
#include <setjmp.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/syscall.h>
#include <sys/system_properties.h>
#include <unistd.h>

#include <mutex>

namespace crashpad {
namespace internal {

// KitKat supports API levels up to 20.
#if __ANDROID_API__ < 21

namespace {

class ScopedSigactionRestore {
 public:
  ScopedSigactionRestore() : old_action_(), signo_(-1), valid_(false) {}

  ~ScopedSigactionRestore() { Reset(); }

  bool Reset() {
    bool result = true;
    if (valid_) {
      result = sigaction(signo_, &old_action_, nullptr) == 0;
      if (!result) {
        PrintErrmsg(errno);
      }
    }
    valid_ = false;
    signo_ = -1;
    return result;
  }

  bool ResetAndInstallHandler(int signo,
                              void (*handler)(int, siginfo_t*, void*)) {
    Reset();

    struct sigaction act;
    act.sa_sigaction = handler;
    sigemptyset(&act.sa_mask);
    act.sa_flags = SA_SIGINFO;
    if (sigaction(signo, &act, &old_action_) != 0) {
      PrintErrmsg(errno);
      return false;
    }
    signo_ = signo;
    valid_ = true;
    return true;
  }

 private:
  void PrintErrmsg(int err) {
    char errmsg[256];

    if (strerror_r(err, errmsg, sizeof(errmsg)) != 0) {
      snprintf(errmsg,
               sizeof(errmsg),
               "%s:%d: Couldn't set errmsg for %d: %d",
               __FILE__,
               __LINE__,
               err,
               errno);
      return;
    }

    fprintf(stderr, "%s:%d: sigaction: %s", __FILE__, __LINE__, errmsg);
  }

  struct sigaction old_action_;
  int signo_;
  bool valid_;
};

bool IsKitKat() {
  char prop_buf[PROP_VALUE_MAX];
  int length = __system_property_get("ro.build.version.sdk", prop_buf);
  if (length <= 0) {
    fprintf(stderr, "%s:%d: Couldn't get version", __FILE__, __LINE__);
    // It's safer to assume this is KitKat and execute dlsym with a signal
    // handler installed.
    return true;
  }
  if (strcmp(prop_buf, "19") == 0 || strcmp(prop_buf, "20") == 0) {
    return true;
  }
  return false;
}

class ScopedSetTID {
 public:
  explicit ScopedSetTID(pid_t* tid) : tid_(tid) { *tid_ = syscall(SYS_gettid); }

  ~ScopedSetTID() { *tid_ = -1; }

 private:
  pid_t* tid_;
};

sigjmp_buf dlsym_sigjmp_env;

pid_t dlsym_tid = -1;

void HandleSIGFPE(int signo, siginfo_t* siginfo, void* context) {
  if (siginfo->si_code != FPE_INTDIV || syscall(SYS_gettid) != dlsym_tid) {
    return;
  }
  siglongjmp(dlsym_sigjmp_env, 1);
}

}  // namespace

void* Dlsym(void* handle, const char* symbol) {
  if (!IsKitKat()) {
    return dlsym(handle, symbol);
  }

  static std::mutex* signal_handler_mutex = new std::mutex();
  std::lock_guard<std::mutex> lock(*signal_handler_mutex);

  ScopedSetTID set_tid(&dlsym_tid);

  ScopedSigactionRestore sig_restore;
  if (!sig_restore.ResetAndInstallHandler(SIGFPE, HandleSIGFPE)) {
    return nullptr;
  }

  if (sigsetjmp(dlsym_sigjmp_env, 1) != 0) {
    return nullptr;
  }

  return dlsym(handle, symbol);
}

#else

void* Dlsym(void* handle, const char* symbol) {
  return dlsym(handle, symbol);
}

#endif  // __ANDROID_API__ < 21

}  // namespace internal
}  // namespace crashpad
