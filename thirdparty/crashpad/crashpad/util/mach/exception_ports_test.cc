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

#include "util/mach/exception_ports.h"

#include <mach/mach.h>
#include <pthread.h>
#include <signal.h>
#include <unistd.h>

#include "base/logging.h"
#include "base/mac/mach_logging.h"
#include "base/mac/scoped_mach_port.h"
#include "base/macros.h"
#include "base/strings/stringprintf.h"
#include "gtest/gtest.h"
#include "test/mac/mach_errors.h"
#include "test/mac/mach_multiprocess.h"
#include "util/file/file_io.h"
#include "util/mach/exc_server_variants.h"
#include "util/mach/exception_types.h"
#include "util/mach/mach_extensions.h"
#include "util/mach/mach_message.h"
#include "util/mach/mach_message_server.h"
#include "util/misc/scoped_forbid_return.h"
#include "util/synchronization/semaphore.h"

namespace crashpad {
namespace test {
namespace {

// Calls GetExceptionPorts() on its |exception_ports| argument to look up the
// EXC_MASK_CRASH handler. If |expect_port| is not MACH_PORT_NULL, it expects to
// find a handler for this mask whose port matches |expect_port| and whose
// behavior matches |expect_behavior| exactly. In this case, if
// |expect_behavior| is a state-carrying behavior, the looked-up thread state
// flavor is expected to be MACHINE_THREAD_STATE, otherwise, it is expected to
// be THREAD_STATE_NONE. If |expect_port| is MACH_PORT_NULL, no handler for
// EXC_MASK_CRASH is expected to be found.
//
// A second GetExceptionPorts() lookup is also performed on a wider exception
// mask, EXC_MASK_ALL | EXC_MASK_CRASH. The EXC_MASK_CRASH handler’s existence
// and properties from this second lookup are validated in the same way.
//
// This function uses gtest EXPECT_* and ASSERT_* macros to perform its
// validation.
void TestGetExceptionPorts(const ExceptionPorts& exception_ports,
                           mach_port_t expect_port,
                           exception_behavior_t expect_behavior) {
  constexpr exception_mask_t kExceptionMask = EXC_MASK_CRASH;

  thread_state_flavor_t expect_flavor = (expect_behavior == EXCEPTION_DEFAULT)
                                            ? THREAD_STATE_NONE
                                            : MACHINE_THREAD_STATE;

  ExceptionPorts::ExceptionHandlerVector crash_handler;
  ASSERT_TRUE(
      exception_ports.GetExceptionPorts(kExceptionMask, &crash_handler));

  if (expect_port != MACH_PORT_NULL) {
    ASSERT_EQ(crash_handler.size(), 1u);

    EXPECT_EQ(crash_handler[0].mask, kExceptionMask);
    EXPECT_EQ(crash_handler[0].port, expect_port);
    EXPECT_EQ(crash_handler[0].behavior, expect_behavior);
    EXPECT_EQ(crash_handler[0].flavor, expect_flavor);
  } else {
    EXPECT_TRUE(crash_handler.empty());
  }

  ExceptionPorts::ExceptionHandlerVector handlers;
  ASSERT_TRUE(exception_ports.GetExceptionPorts(ExcMaskValid(), &handlers));

  EXPECT_GE(handlers.size(), crash_handler.size());
  bool found = false;
  for (const ExceptionPorts::ExceptionHandler& handler : handlers) {
    if ((handler.mask & kExceptionMask) != 0) {
      EXPECT_FALSE(found);
      found = true;
      EXPECT_EQ(handler.port, expect_port);
      EXPECT_EQ(handler.behavior, expect_behavior);
      EXPECT_EQ(handler.flavor, expect_flavor);
    }
  }

  if (expect_port != MACH_PORT_NULL) {
    EXPECT_TRUE(found);
  } else {
    EXPECT_FALSE(found);
  }
}

class TestExceptionPorts : public MachMultiprocess,
                           public UniversalMachExcServer::Interface {
 public:
  // Which entities to set exception ports for.
  enum SetOn {
    kSetOnTaskOnly = 0,
    kSetOnTaskAndThreads,
  };

  // Where to call ExceptionPorts::SetExceptionPort() from.
  enum SetType {
    // Call it from the child process on itself.
    kSetInProcess = 0,

    // Call it from the parent process on the child.
    kSetOutOfProcess,
  };

  // Which thread in the child process is expected to crash.
  enum WhoCrashes {
    kNobodyCrashes = 0,
    kMainThreadCrashes,
    kOtherThreadCrashes,
  };

  TestExceptionPorts(SetOn set_on, SetType set_type, WhoCrashes who_crashes)
      : MachMultiprocess(),
        UniversalMachExcServer::Interface(),
        set_on_(set_on),
        set_type_(set_type),
        who_crashes_(who_crashes),
        handled_(false) {
    if (who_crashes_ != kNobodyCrashes) {
      // This is how the __builtin_trap() in Child::Crash() appears.
      SetExpectedChildTermination(kTerminationSignal, SIGILL);
    }
  }

  SetOn set_on() const { return set_on_; }
  SetType set_type() const { return set_type_; }
  WhoCrashes who_crashes() const { return who_crashes_; }

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

    // To be able to distinguish between which handler was actually triggered,
    // the different handlers are registered with different behavior values.
    exception_behavior_t expect_behavior;
    if (set_on_ == kSetOnTaskOnly) {
      expect_behavior = EXCEPTION_DEFAULT;
    } else if (who_crashes_ == kMainThreadCrashes) {
      expect_behavior = EXCEPTION_STATE;
    } else if (who_crashes_ == kOtherThreadCrashes) {
      expect_behavior = EXCEPTION_STATE_IDENTITY;
    } else {
      NOTREACHED();
      expect_behavior = 0;
    }

    EXPECT_EQ(behavior, expect_behavior);

    EXPECT_EQ(exception_port, LocalPort());

    EXPECT_EQ(exception, EXC_CRASH);
    EXPECT_EQ(code_count, 2u);

    // The exception and code_count checks above would ideally use ASSERT_EQ so
    // that the next conditional would not be necessary, but ASSERT_* requires a
    // function returning type void, and the interface dictates otherwise here.
    if (exception == EXC_CRASH && code_count >= 1) {
      int signal;
      ExcCrashRecoverOriginalException(code[0], nullptr, &signal);

      // The child crashed with __builtin_trap(), which shows up as SIGILL.
      EXPECT_EQ(signal, SIGILL);
    }

    EXPECT_EQ(AuditPIDFromMachMessageTrailer(trailer), 0);

    ExcServerCopyState(
        behavior, old_state, old_state_count, new_state, new_state_count);
    return ExcServerSuccessfulReturnValue(exception, behavior, false);
  }

 private:
  class Child {
   public:
    explicit Child(TestExceptionPorts* test_exception_ports)
        : test_exception_ports_(test_exception_ports),
          thread_(),
          init_semaphore_(0),
          crash_semaphore_(0) {}

    ~Child() {}

    void Run() {
      ExceptionPorts self_task_ports(ExceptionPorts::kTargetTypeTask,
                                     TASK_NULL);
      ExceptionPorts self_thread_ports(ExceptionPorts::kTargetTypeThread,
                                       THREAD_NULL);

      mach_port_t remote_port = test_exception_ports_->RemotePort();

      // Set the task’s and this thread’s exception ports, if appropriate.
      if (test_exception_ports_->set_type() == kSetInProcess) {
        ASSERT_TRUE(self_task_ports.SetExceptionPort(
            EXC_MASK_CRASH, remote_port, EXCEPTION_DEFAULT, THREAD_STATE_NONE));

        if (test_exception_ports_->set_on() == kSetOnTaskAndThreads) {
          ASSERT_TRUE(self_thread_ports.SetExceptionPort(EXC_MASK_CRASH,
                                                         remote_port,
                                                         EXCEPTION_STATE,
                                                         MACHINE_THREAD_STATE));
        }
      }

      int rv_int = pthread_create(&thread_, nullptr, ThreadMainThunk, this);
      ASSERT_EQ(rv_int, 0);

      // Wait for the new thread to be ready.
      init_semaphore_.Wait();

      // Tell the parent process that everything is set up.
      char c = '\0';
      CheckedWriteFile(test_exception_ports_->WritePipeHandle(), &c, 1);

      // Wait for the parent process to say that its end is set up.
      CheckedReadFileExactly(test_exception_ports_->ReadPipeHandle(), &c, 1);
      EXPECT_EQ(c, '\0');

      // Regardless of where ExceptionPorts::SetExceptionPort() ran,
      // ExceptionPorts::GetExceptionPorts() can always be tested in-process.
      {
        SCOPED_TRACE("task");
        TestGetExceptionPorts(self_task_ports, remote_port, EXCEPTION_DEFAULT);
      }

      {
        SCOPED_TRACE("main_thread");
        mach_port_t thread_handler =
            (test_exception_ports_->set_on() == kSetOnTaskAndThreads)
                ? remote_port
                : MACH_PORT_NULL;
        TestGetExceptionPorts(
            self_thread_ports, thread_handler, EXCEPTION_STATE);
      }

      // Let the other thread know it’s safe to proceed.
      crash_semaphore_.Signal();

      // If this thread is the one that crashes, do it.
      if (test_exception_ports_->who_crashes() == kMainThreadCrashes) {
        Crash();
      }

      // Reap the other thread.
      rv_int = pthread_join(thread_, nullptr);
      ASSERT_EQ(rv_int, 0);
    }

   private:
    // Calls ThreadMain().
    static void* ThreadMainThunk(void* argument) {
      Child* self = reinterpret_cast<Child*>(argument);
      return self->ThreadMain();
    }

    // Runs the “other” thread.
    void* ThreadMain() {
      ExceptionPorts self_thread_ports(ExceptionPorts::kTargetTypeThread,
                                       THREAD_NULL);
      mach_port_t remote_port = test_exception_ports_->RemotePort();

      // Set this thread’s exception handler, if appropriate.
      if (test_exception_ports_->set_type() == kSetInProcess &&
          test_exception_ports_->set_on() == kSetOnTaskAndThreads) {
        CHECK(self_thread_ports.SetExceptionPort(EXC_MASK_CRASH,
                                                 remote_port,
                                                 EXCEPTION_STATE_IDENTITY,
                                                 MACHINE_THREAD_STATE));
      }

      // Let the main thread know that this thread is ready.
      init_semaphore_.Signal();

      // Wait for the main thread to signal that it’s safe to proceed.
      crash_semaphore_.Wait();

      // Regardless of where ExceptionPorts::SetExceptionPort() ran,
      // ExceptionPorts::GetExceptionPorts() can always be tested in-process.
      {
        SCOPED_TRACE("other_thread");
        mach_port_t thread_handler =
            (test_exception_ports_->set_on() == kSetOnTaskAndThreads)
                ? remote_port
                : MACH_PORT_NULL;
        TestGetExceptionPorts(
            self_thread_ports, thread_handler, EXCEPTION_STATE_IDENTITY);
      }

      // If this thread is the one that crashes, do it.
      if (test_exception_ports_->who_crashes() == kOtherThreadCrashes) {
        Crash();
      }

      return nullptr;
    }

    static void Crash() {
      __builtin_trap();
    }

    // The parent object.
    TestExceptionPorts* test_exception_ports_;  // weak

    // The “other” thread.
    pthread_t thread_;

    // The main thread waits on this for the other thread to start up and
    // perform its own initialization.
    Semaphore init_semaphore_;

    // The child thread waits on this for the parent thread to indicate that the
    // child can test its exception ports and possibly crash, as appropriate.
    Semaphore crash_semaphore_;

    DISALLOW_COPY_AND_ASSIGN(Child);
  };

  // MachMultiprocess:

  void MachMultiprocessParent() override {
    // Wait for the child process to be ready. It needs to have all of its
    // threads set up before proceeding if in kSetOutOfProcess mode.
    char c;
    CheckedReadFileExactly(ReadPipeHandle(), &c, 1);
    EXPECT_EQ(c, '\0');

    mach_port_t local_port = LocalPort();

    // Get an ExceptionPorts object for the task and each of its threads.
    ExceptionPorts task_ports(ExceptionPorts::kTargetTypeTask, ChildTask());
    EXPECT_STREQ("task", task_ports.TargetTypeName());

    // Hopefully the threads returned by task_threads() are in order, with the
    // main thread first and the other thread second. This is currently always
    // the case, although nothing guarantees that it will remain so.
    thread_act_array_t threads;
    mach_msg_type_number_t thread_count = 0;
    kern_return_t kr = task_threads(ChildTask(), &threads, &thread_count);
    ASSERT_EQ(kr, KERN_SUCCESS) << MachErrorMessage(kr, "task_threads");

    ScopedForbidReturn threads_need_owners;
    ASSERT_EQ(thread_count, 2u);
    base::mac::ScopedMachSendRight main_thread(threads[0]);
    base::mac::ScopedMachSendRight other_thread(threads[1]);
    threads_need_owners.Disarm();

    ExceptionPorts main_thread_ports(ExceptionPorts::kTargetTypeThread,
                                     main_thread.get());
    ExceptionPorts other_thread_ports(ExceptionPorts::kTargetTypeThread,
                                      other_thread.get());
    EXPECT_STREQ("thread", main_thread_ports.TargetTypeName());
    EXPECT_STREQ("thread", other_thread_ports.TargetTypeName());

    if (set_type_ == kSetOutOfProcess) {
      // Test ExceptionPorts::SetExceptionPorts() being called from
      // out-of-process.
      //
      // local_port is only a receive right, but a send right is needed for
      // ExceptionPorts::SetExceptionPort(). Make a send right, which can be
      // deallocated once the calls to ExceptionPorts::SetExceptionPort() are
      // done.
      kr = mach_port_insert_right(
          mach_task_self(), local_port, local_port, MACH_MSG_TYPE_MAKE_SEND);
      ASSERT_EQ(kr, KERN_SUCCESS)
          << MachErrorMessage(kr, "mach_port_insert_right");
      base::mac::ScopedMachSendRight send_owner(local_port);

      ASSERT_TRUE(task_ports.SetExceptionPort(
          EXC_MASK_CRASH, local_port, EXCEPTION_DEFAULT, THREAD_STATE_NONE));

      if (set_on_ == kSetOnTaskAndThreads) {
        ASSERT_TRUE(main_thread_ports.SetExceptionPort(
            EXC_MASK_CRASH, local_port, EXCEPTION_STATE, MACHINE_THREAD_STATE));

        ASSERT_TRUE(
            other_thread_ports.SetExceptionPort(EXC_MASK_CRASH,
                                                local_port,
                                                EXCEPTION_STATE_IDENTITY,
                                                MACHINE_THREAD_STATE));
      }
    }

    // Regardless of where ExceptionPorts::SetExceptionPort() ran,
    // ExceptionPorts::GetExceptionPorts() can always be tested out-of-process.
    {
      SCOPED_TRACE("task");
      TestGetExceptionPorts(task_ports, local_port, EXCEPTION_DEFAULT);
    }

    mach_port_t thread_handler =
        (set_on_ == kSetOnTaskAndThreads) ? local_port : MACH_PORT_NULL;

    {
      SCOPED_TRACE("main_thread");
      TestGetExceptionPorts(main_thread_ports, thread_handler, EXCEPTION_STATE);
    }

    {
      SCOPED_TRACE("other_thread");
      TestGetExceptionPorts(
          other_thread_ports, thread_handler, EXCEPTION_STATE_IDENTITY);
    }

    // Let the child process know that everything in the parent process is set
    // up.
    c = '\0';
    CheckedWriteFile(WritePipeHandle(), &c, 1);

    if (who_crashes_ != kNobodyCrashes) {
      UniversalMachExcServer universal_mach_exc_server(this);

      constexpr mach_msg_timeout_t kTimeoutMs = 50;
      kern_return_t kr =
          MachMessageServer::Run(&universal_mach_exc_server,
                                 local_port,
                                 kMachMessageReceiveAuditTrailer,
                                 MachMessageServer::kOneShot,
                                 MachMessageServer::kReceiveLargeError,
                                 kTimeoutMs);
      EXPECT_EQ(kr, KERN_SUCCESS)
          << MachErrorMessage(kr, "MachMessageServer::Run");

      EXPECT_TRUE(handled_);
    }

    // Wait for the child process to exit or terminate, as indicated by it
    // closing its pipe. This keeps LocalPort() alive in the child as
    // RemotePort(), for the child’s use in its TestGetExceptionPorts().
    CheckedReadFileAtEOF(ReadPipeHandle());
  }

  void MachMultiprocessChild() override {
    Child child(this);
    child.Run();
  }

  SetOn set_on_;
  SetType set_type_;
  WhoCrashes who_crashes_;

  // true if an exception message was handled.
  bool handled_;

  DISALLOW_COPY_AND_ASSIGN(TestExceptionPorts);
};

TEST(ExceptionPorts, TaskExceptionPorts_SetInProcess_NoCrash) {
  TestExceptionPorts test_exception_ports(
      TestExceptionPorts::kSetOnTaskOnly,
      TestExceptionPorts::kSetInProcess,
      TestExceptionPorts::kNobodyCrashes);
  test_exception_ports.Run();
}

TEST(ExceptionPorts, TaskExceptionPorts_SetInProcess_MainThreadCrash) {
  TestExceptionPorts test_exception_ports(
      TestExceptionPorts::kSetOnTaskOnly,
      TestExceptionPorts::kSetInProcess,
      TestExceptionPorts::kMainThreadCrashes);
  test_exception_ports.Run();
}

TEST(ExceptionPorts, TaskExceptionPorts_SetInProcess_OtherThreadCrash) {
  TestExceptionPorts test_exception_ports(
      TestExceptionPorts::kSetOnTaskOnly,
      TestExceptionPorts::kSetInProcess,
      TestExceptionPorts::kOtherThreadCrashes);
  test_exception_ports.Run();
}

TEST(ExceptionPorts, TaskAndThreadExceptionPorts_SetInProcess_NoCrash) {
  TestExceptionPorts test_exception_ports(
      TestExceptionPorts::kSetOnTaskAndThreads,
      TestExceptionPorts::kSetInProcess,
      TestExceptionPorts::kNobodyCrashes);
  test_exception_ports.Run();
}

TEST(ExceptionPorts, TaskAndThreadExceptionPorts_SetInProcess_MainThreadCrash) {
  TestExceptionPorts test_exception_ports(
      TestExceptionPorts::kSetOnTaskAndThreads,
      TestExceptionPorts::kSetInProcess,
      TestExceptionPorts::kMainThreadCrashes);
  test_exception_ports.Run();
}

TEST(ExceptionPorts,
     TaskAndThreadExceptionPorts_SetInProcess_OtherThreadCrash) {
  TestExceptionPorts test_exception_ports(
      TestExceptionPorts::kSetOnTaskAndThreads,
      TestExceptionPorts::kSetInProcess,
      TestExceptionPorts::kOtherThreadCrashes);
  test_exception_ports.Run();
}

TEST(ExceptionPorts, TaskExceptionPorts_SetOutOfProcess_NoCrash) {
  TestExceptionPorts test_exception_ports(
      TestExceptionPorts::kSetOnTaskOnly,
      TestExceptionPorts::kSetOutOfProcess,
      TestExceptionPorts::kNobodyCrashes);
  test_exception_ports.Run();
}

TEST(ExceptionPorts, TaskExceptionPorts_SetOutOfProcess_MainThreadCrash) {
  TestExceptionPorts test_exception_ports(
      TestExceptionPorts::kSetOnTaskOnly,
      TestExceptionPorts::kSetOutOfProcess,
      TestExceptionPorts::kMainThreadCrashes);
  test_exception_ports.Run();
}

TEST(ExceptionPorts, TaskExceptionPorts_SetOutOfProcess_OtherThreadCrash) {
  TestExceptionPorts test_exception_ports(
      TestExceptionPorts::kSetOnTaskOnly,
      TestExceptionPorts::kSetOutOfProcess,
      TestExceptionPorts::kOtherThreadCrashes);
  test_exception_ports.Run();
}

TEST(ExceptionPorts, TaskAndThreadExceptionPorts_SetOutOfProcess_NoCrash) {
  TestExceptionPorts test_exception_ports(
      TestExceptionPorts::kSetOnTaskAndThreads,
      TestExceptionPorts::kSetOutOfProcess,
      TestExceptionPorts::kNobodyCrashes);
  test_exception_ports.Run();
}

TEST(ExceptionPorts,
     TaskAndThreadExceptionPorts_SetOutOfProcess_MainThreadCrash) {
  TestExceptionPorts test_exception_ports(
      TestExceptionPorts::kSetOnTaskAndThreads,
      TestExceptionPorts::kSetOutOfProcess,
      TestExceptionPorts::kMainThreadCrashes);
  test_exception_ports.Run();
}

TEST(ExceptionPorts,
     TaskAndThreadExceptionPorts_SetOutOfProcess_OtherThreadCrash) {
  TestExceptionPorts test_exception_ports(
      TestExceptionPorts::kSetOnTaskAndThreads,
      TestExceptionPorts::kSetOutOfProcess,
      TestExceptionPorts::kOtherThreadCrashes);
  test_exception_ports.Run();
}

TEST(ExceptionPorts, HostExceptionPorts) {
  // ExceptionPorts isn’t expected to work as non-root. Just do a quick test to
  // make sure that TargetTypeName() returns the right string, and that the
  // underlying host_get_exception_ports() function appears to be called by
  // looking for a KERN_INVALID_ARGUMENT return value. Or, on the off chance
  // that the test is being run as root, just look for KERN_SUCCESS.
  // host_set_exception_ports() is not tested, because if the test were running
  // as root and the call succeeded, it would have global effects.

  const bool expect_success = geteuid() == 0;

  base::mac::ScopedMachSendRight host(mach_host_self());
  ExceptionPorts explicit_host_ports(ExceptionPorts::kTargetTypeHost,
                                     host.get());
  EXPECT_STREQ("host", explicit_host_ports.TargetTypeName());

  ExceptionPorts::ExceptionHandlerVector explicit_handlers;
  bool rv =
      explicit_host_ports.GetExceptionPorts(ExcMaskValid(), &explicit_handlers);
  EXPECT_EQ(rv, expect_success);

  ExceptionPorts implicit_host_ports(ExceptionPorts::kTargetTypeHost,
                                     HOST_NULL);
  EXPECT_STREQ("host", implicit_host_ports.TargetTypeName());

  ExceptionPorts::ExceptionHandlerVector implicit_handlers;
  rv =
      implicit_host_ports.GetExceptionPorts(ExcMaskValid(), &implicit_handlers);
  EXPECT_EQ(rv, expect_success);

  EXPECT_EQ(implicit_handlers.size(), explicit_handlers.size());
}

}  // namespace
}  // namespace test
}  // namespace crashpad
