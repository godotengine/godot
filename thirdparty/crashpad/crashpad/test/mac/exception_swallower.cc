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

#include "test/mac/exception_swallower.h"

#include <errno.h>
#include <stdlib.h>
#include <unistd.h>

#include <string>

#include "base/logging.h"
#include "base/mac/scoped_mach_port.h"
#include "base/strings/stringprintf.h"
#include "handler/mac/exception_handler_server.h"
#include "util/mach/exc_server_variants.h"
#include "util/mach/exception_ports.h"
#include "util/mach/mach_extensions.h"
#include "util/misc/random_string.h"
#include "util/thread/thread.h"

namespace crashpad {
namespace test {

namespace {

constexpr char kServiceEnvironmentVariable[] =
    "CRASHPAD_EXCEPTION_SWALLOWER_SERVICE";

ExceptionSwallower* g_exception_swallower;

// Like getenv(), but fails a CHECK() if the underlying function fails. It’s not
// considered a failure for |name| to be unset in the environment. In that case,
// nullptr is returned.
const char* CheckedGetenv(const char* name) {
  errno = 0;
  const char* value;
  PCHECK((value = getenv(name)) || errno == 0) << "getenv";
  return value;
}

}  // namespace

class ExceptionSwallower::ExceptionSwallowerThread
    : public Thread,
      public UniversalMachExcServer::Interface {
 public:
  explicit ExceptionSwallowerThread(
      base::mac::ScopedMachReceiveRight receive_right)
      : Thread(),
        UniversalMachExcServer::Interface(),
        exception_handler_server_(std::move(receive_right), true),
        pid_(getpid()) {
    Start();
  }

  ~ExceptionSwallowerThread() override {}

  void Stop() { exception_handler_server_.Stop(); }

  // Returns the process ID that the thread is running in. This is used to
  // detect misuses that place the exception swallower server thread and code
  // that wants its exceptions swallowed in the same process.
  pid_t ProcessID() const { return pid_; }

 private:
  // Thread:

  void ThreadMain() override { exception_handler_server_.Run(this); }

  // UniversalMachExcServer::Interface:

  kern_return_t CatchMachException(exception_behavior_t behavior,
                                   exception_handler_t exception_port,
                                   thread_t thread,
                                   task_t task,
                                   exception_type_t exception,
                                   const mach_exception_data_type_t* code,
                                   mach_msg_type_number_t code_count,
                                   thread_state_flavor_t* flavor,
                                   ConstThreadState old_state,
                                   mach_msg_type_number_t old_state_count,
                                   thread_state_t new_state,
                                   mach_msg_type_number_t* new_state_count,
                                   const mach_msg_trailer_t* trailer,
                                   bool* destroy_complex_request) override {
    *destroy_complex_request = true;

    // Swallow.

    ExcServerCopyState(
        behavior, old_state, old_state_count, new_state, new_state_count);
    return ExcServerSuccessfulReturnValue(exception, behavior, false);
  }

  ExceptionHandlerServer exception_handler_server_;
  pid_t pid_;

  DISALLOW_COPY_AND_ASSIGN(ExceptionSwallowerThread);
};

ExceptionSwallower::ExceptionSwallower() : exception_swallower_thread_() {
  CHECK(!g_exception_swallower);
  g_exception_swallower = this;

  if (CheckedGetenv(kServiceEnvironmentVariable)) {
    // The environment variable is already set, so just proceed with the
    // existing service. This normally happens when the gtest “threadsafe” death
    // test style is chosen, because the test child process will re-execute code
    // already run in the test parent process. See
    // https://github.com/google/googletest/blob/master/googletest/docs/AdvancedGuide.md#death-test-styles.
    return;
  }

  std::string service_name =
      base::StringPrintf("org.chromium.crashpad.test.exception_swallower.%d.%s",
                         getpid(),
                         RandomString().c_str());
  base::mac::ScopedMachReceiveRight receive_right(
      BootstrapCheckIn(service_name));
  CHECK(receive_right.is_valid());

  exception_swallower_thread_.reset(
      new ExceptionSwallowerThread(std::move(receive_right)));

  PCHECK(setenv(kServiceEnvironmentVariable, service_name.c_str(), 1) == 0)
      << "setenv";
}

ExceptionSwallower::~ExceptionSwallower() {
  PCHECK(unsetenv(kServiceEnvironmentVariable) == 0) << "unsetenv";

  exception_swallower_thread_->Stop();
  exception_swallower_thread_->Join();

  CHECK_EQ(g_exception_swallower, this);
  g_exception_swallower = nullptr;
}

// static
void ExceptionSwallower::SwallowExceptions() {
  // The exception swallower thread can’t be in this process, because the
  // EXC_CRASH or EXC_CORPSE_NOTIFY exceptions that it needs to swallow will be
  // delivered after a crash has occurred and none of its threads will be
  // scheduled to run.
  CHECK(!g_exception_swallower ||
        !g_exception_swallower->exception_swallower_thread_ ||
        g_exception_swallower->exception_swallower_thread_->ProcessID() !=
            getpid());

  const char* service_name = CheckedGetenv(kServiceEnvironmentVariable);
  CHECK(service_name);

  base::mac::ScopedMachSendRight exception_swallower_port(
      BootstrapLookUp(service_name));
  CHECK(exception_swallower_port.is_valid());

  ExceptionPorts task_exception_ports(ExceptionPorts::kTargetTypeTask,
                                      TASK_NULL);

  // The mask is similar to the one used by CrashpadClient::UseHandler(), but
  // EXC_CORPSE_NOTIFY is added. This is done for the benefit of tests that
  // crash intentionally with their own custom exception port set for EXC_CRASH.
  // In that case, depending on the actions taken by the EXC_CRASH handler, the
  // exception may be transformed by the kernel into an EXC_CORPSE_NOTIFY, which
  // would be sent to an EXC_CORPSE_NOTIFY handler, normally the system’s crash
  // reporter at the task or host level. See 10.13.0
  // xnu-4570.1.46/bsd/kern/kern_exit.c proc_prepareexit(). Swallowing
  // EXC_CORPSE_NOTIFY at the task level prevents such exceptions from reaching
  // the system’s crash reporter.
  CHECK(task_exception_ports.SetExceptionPort(
      (EXC_MASK_CRASH |
       EXC_MASK_RESOURCE |
       EXC_MASK_GUARD |
       EXC_MASK_CORPSE_NOTIFY) & ExcMaskValid(),
      exception_swallower_port.get(),
      EXCEPTION_DEFAULT,
      THREAD_STATE_NONE));
}

}  // namespace test
}  // namespace crashpad
