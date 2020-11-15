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

#include "util/mach/exc_client_variants.h"

#include <mach/mach.h>
#include <pthread.h>
#include <string.h>
#include <sys/types.h>

#include "base/macros.h"
#include "base/strings/stringprintf.h"
#include "gtest/gtest.h"
#include "test/mac/mach_errors.h"
#include "test/mac/mach_multiprocess.h"
#include "util/mach/exc_server_variants.h"
#include "util/mach/exception_behaviors.h"
#include "util/mach/mach_extensions.h"
#include "util/mach/mach_message.h"
#include "util/mach/mach_message_server.h"
#include "util/misc/implicit_cast.h"

namespace crashpad {
namespace test {
namespace {

class TestExcClientVariants : public MachMultiprocess,
                              public UniversalMachExcServer::Interface {
 public:
  TestExcClientVariants(exception_behavior_t behavior, bool all_fields)
      : MachMultiprocess(),
        UniversalMachExcServer::Interface(),
        behavior_(behavior),
        all_fields_(all_fields),
        handled_(false) {
    ++exception_;
    ++exception_code_;
    ++exception_subcode_;
  }

  // UniversalMachExcServer::Interface:

  virtual kern_return_t CatchMachException(
      exception_behavior_t behavior,
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

    EXPECT_FALSE(handled_);
    handled_ = true;

    EXPECT_EQ(behavior, behavior_);
    EXPECT_EQ(exception_port, LocalPort());

    if (HasIdentity()) {
      EXPECT_NE(thread, THREAD_NULL);
      EXPECT_EQ(task, ChildTask());
    } else {
      EXPECT_EQ(thread, THREAD_NULL);
      EXPECT_EQ(task, TASK_NULL);
    }

    mach_exception_code_t expect_code = exception_code_;
    mach_exception_subcode_t expect_subcode = exception_subcode_;
    if ((behavior & MACH_EXCEPTION_CODES) == 0) {
      expect_code = implicit_cast<exception_data_type_t>(expect_code);
      expect_subcode = implicit_cast<exception_data_type_t>(expect_subcode);
    }

    EXPECT_EQ(exception, exception_);
    EXPECT_EQ(code_count, 2u);

    // The code_count check above would ideally use ASSERT_EQ so that the next
    // conditionals would not be necessary, but ASSERT_* requires a function
    // returning type void, and the interface dictates otherwise here.
    if (code_count >= 1) {
      EXPECT_EQ(code[0], expect_code);
    }
    if (code_count >= 2) {
      EXPECT_EQ(code[1], expect_subcode);
    }

    if (HasState()) {
      EXPECT_EQ(*flavor, exception_ + 10);
      EXPECT_EQ(old_state_count, MACHINE_THREAD_STATE_COUNT);
      EXPECT_NE(old_state, nullptr);
      EXPECT_EQ(*new_state_count,
                implicit_cast<mach_msg_type_number_t>(THREAD_STATE_MAX));
      EXPECT_NE(new_state, nullptr);

      for (size_t index = 0; index < old_state_count; ++index) {
        EXPECT_EQ(old_state[index], index);
      }

      // Use a flavor known to be different from the incoming flavor, for a test
      // of the “out” side of the in-out flavor parameter.
      *flavor = exception_ + 20;
      *new_state_count = MACHINE_THREAD_STATE_COUNT;

      // Send a new state back to the client.
      for (size_t index = 0; index < *new_state_count; ++index) {
        new_state[index] = MACHINE_THREAD_STATE_COUNT - index;
      }
    } else {
      EXPECT_EQ(*flavor, THREAD_STATE_NONE);
      EXPECT_EQ(old_state_count, 0u);
      EXPECT_EQ(old_state, nullptr);
      EXPECT_EQ(*new_state_count, 0u);
      EXPECT_EQ(new_state, nullptr);
    }

    return KERN_SUCCESS;
  }

 private:
  // MachMultiprocess:

  void MachMultiprocessParent() override {
    UniversalMachExcServer universal_mach_exc_server(this);

    kern_return_t kr =
        MachMessageServer::Run(&universal_mach_exc_server,
                               LocalPort(),
                               MACH_MSG_OPTION_NONE,
                               MachMessageServer::kOneShot,
                               MachMessageServer::kReceiveLargeError,
                               kMachMessageTimeoutWaitIndefinitely);
    EXPECT_EQ(kr, KERN_SUCCESS)
        << MachErrorMessage(kr, "MachMessageServer::Run");

    EXPECT_TRUE(handled_);
  }

  void MachMultiprocessChild() override {
    const exception_type_t exception = exception_;
    const mach_exception_data_type_t code[] = {
        exception_code_,
        exception_subcode_
    };

    thread_t thread = THREAD_NULL;
    task_t task = TASK_NULL;
    if (all_fields_ || HasIdentity()) {
      thread = MachThreadSelf();
      task = mach_task_self();
    }

    thread_state_flavor_t flavor;
    thread_state_flavor_t* flavor_p = nullptr;
    natural_t old_state[MACHINE_THREAD_STATE_COUNT];
    thread_state_t old_state_p = nullptr;
    mach_msg_type_number_t old_state_count = 0;
    natural_t new_state[THREAD_STATE_MAX];
    thread_state_t new_state_p = nullptr;
    mach_msg_type_number_t new_state_count;
    mach_msg_type_number_t* new_state_count_p = nullptr;
    if (all_fields_ || HasState()) {
      // Pick a different flavor each time based on the value of exception_.
      // These aren’t real flavors, it’s just for testing.
      flavor = exception_ + 10;
      flavor_p = &flavor;
      for (size_t index = 0; index < arraysize(old_state); ++index) {
        old_state[index] = index;
      }
      old_state_p = reinterpret_cast<thread_state_t>(&old_state);
      old_state_count = arraysize(old_state);

      // new_state and new_state_count are out parameters that the server should
      // never see or use, so set them to bogus values. The call to the server
      // should overwrite these.
      memset(new_state, 0xa5, sizeof(new_state));
      new_state_p = reinterpret_cast<thread_state_t>(&new_state);
      new_state_count = 0x5a;
      new_state_count_p = &new_state_count;
    }

    EXPECT_EQ(UniversalExceptionRaise(behavior_,
                                      RemotePort(),
                                      thread,
                                      task,
                                      exception,
                                      code,
                                      arraysize(code),
                                      flavor_p,
                                      old_state_p,
                                      old_state_count,
                                      new_state_p,
                                      new_state_count_p),
              KERN_SUCCESS);

    if (HasState()) {
      // Verify the out parameters.

      EXPECT_EQ(flavor, exception_ + 20);
      EXPECT_EQ(new_state_count, MACHINE_THREAD_STATE_COUNT);

      for (size_t index = 0; index < new_state_count; ++index) {
        EXPECT_EQ(new_state[index], MACHINE_THREAD_STATE_COUNT - index);
      }
    }
  }

  bool HasIdentity() const {
    return ExceptionBehaviorHasIdentity(behavior_);
  }

  bool HasState() const {
    return ExceptionBehaviorHasState(behavior_);
  }

  // The behavior to test.
  exception_behavior_t behavior_;

  // If false, only fields required for the current value of behavior_ are set
  // in a call to UniversalExceptionRaise(). The thread and task fields are only
  // set for identity-carrying behaviors, and the flavor and state fields are
  // only set for state-carrying behaviors. If true, all fields are set
  // regardless of the behavior. Testing in both ways verifies that
  // UniversalExceptionRaise() can tolerate the null arguments documented as
  // usable when the behavior allows it, and that it ignores these arguments
  // even when set when the behavior does not make use of them.
  bool all_fields_;

  // true if an exception message was handled.
  bool handled_;

  // These fields will increment for each instantiation of the test class.
  static exception_type_t exception_;
  static mach_exception_code_t exception_code_;
  static mach_exception_subcode_t exception_subcode_;

  DISALLOW_COPY_AND_ASSIGN(TestExcClientVariants);
};

exception_type_t TestExcClientVariants::exception_ = 0;

// exception_code_ and exception_subcode_ are always large enough to require
// 64 bits, so that when the 32-bit-only variants not using MACH_EXCEPITON_CODES
// are tested, the code and subcode fields can be checked for proper truncation.
mach_exception_code_t TestExcClientVariants::exception_code_ = 0x100000000;
mach_exception_subcode_t TestExcClientVariants::exception_subcode_ =
        0xffffffff00000000;

TEST(ExcClientVariants, UniversalExceptionRaise) {
  static constexpr exception_behavior_t kBehaviors[] = {
      EXCEPTION_DEFAULT,
      EXCEPTION_STATE,
      EXCEPTION_STATE_IDENTITY,
      kMachExceptionCodes | EXCEPTION_DEFAULT,
      kMachExceptionCodes | EXCEPTION_STATE,
      kMachExceptionCodes | EXCEPTION_STATE_IDENTITY,
  };

  for (size_t index = 0; index < arraysize(kBehaviors); ++index) {
    exception_behavior_t behavior = kBehaviors[index];
    SCOPED_TRACE(base::StringPrintf("index %zu, behavior %d", index, behavior));

    {
      SCOPED_TRACE("all_fields = false");

      TestExcClientVariants test_exc_client_variants(behavior, false);
      test_exc_client_variants.Run();
    }

    {
      SCOPED_TRACE("all_fields = true");

      TestExcClientVariants test_exc_client_variants(behavior, true);
      test_exc_client_variants.Run();
    }
  }
}

}  // namespace
}  // namespace test
}  // namespace crashpad
