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

#include "snapshot/linux/exception_snapshot_linux.h"

#include <linux/posix_types.h>
#include <signal.h>
#include <string.h>
#include <time.h>
#include <ucontext.h>
#include <unistd.h>

#include "base/bit_cast.h"
#include "base/macros.h"
#include "base/strings/stringprintf.h"
#include "gtest/gtest.h"
#include "snapshot/cpu_architecture.h"
#include "snapshot/linux/process_reader_linux.h"
#include "snapshot/linux/signal_context.h"
#include "sys/syscall.h"
#include "test/errors.h"
#include "test/linux/fake_ptrace_connection.h"
#include "util/linux/address_types.h"
#include "util/misc/clock.h"
#include "util/misc/from_pointer_cast.h"
#include "util/posix/signals.h"
#include "util/synchronization/semaphore.h"

namespace crashpad {
namespace test {
namespace {

pid_t gettid() {
  return syscall(SYS_gettid);
}

#if defined(ARCH_CPU_X86)
struct FxsaveUContext {
  ucontext_t ucontext;
  CPUContextX86::Fxsave fxsave;
};
using NativeCPUContext = FxsaveUContext;

void InitializeContext(NativeCPUContext* context) {
  context->ucontext.uc_mcontext.gregs[REG_EAX] = 0xabcd1234;
  context->ucontext.uc_mcontext.fpregs = &context->ucontext.__fpregs_mem;
  // glibc and bionic use an unsigned long for status, but the kernel treats
  // status as two uint16_t, with the upper 16 bits called "magic" which, if set
  // to X86_FXSR_MAGIC, indicate that an fxsave follows.
  reinterpret_cast<uint16_t*>(&context->ucontext.__fpregs_mem.status)[1] =
      X86_FXSR_MAGIC;
  memset(&context->fxsave, 43, sizeof(context->fxsave));
}

void ExpectContext(const CPUContext& actual, const NativeCPUContext& expected) {
  EXPECT_EQ(actual.architecture, kCPUArchitectureX86);
  EXPECT_EQ(actual.x86->eax,
            bit_cast<uint32_t>(expected.ucontext.uc_mcontext.gregs[REG_EAX]));
  for (unsigned int byte_offset = 0; byte_offset < sizeof(actual.x86->fxsave);
       ++byte_offset) {
    SCOPED_TRACE(base::StringPrintf("byte offset = %u\n", byte_offset));
    EXPECT_EQ(reinterpret_cast<const char*>(&actual.x86->fxsave)[byte_offset],
              reinterpret_cast<const char*>(&expected.fxsave)[byte_offset]);
  }
}
#elif defined(ARCH_CPU_X86_64)
using NativeCPUContext = ucontext_t;

void InitializeContext(NativeCPUContext* context) {
  context->uc_mcontext.gregs[REG_RAX] = 0xabcd1234abcd1234;
  context->uc_mcontext.fpregs = &context->__fpregs_mem;
  memset(&context->__fpregs_mem, 44, sizeof(context->__fpregs_mem));
}

void ExpectContext(const CPUContext& actual, const NativeCPUContext& expected) {
  EXPECT_EQ(actual.architecture, kCPUArchitectureX86_64);
  EXPECT_EQ(actual.x86_64->rax,
            bit_cast<uint64_t>(expected.uc_mcontext.gregs[REG_RAX]));
  for (unsigned int byte_offset = 0;
       byte_offset < sizeof(actual.x86_64->fxsave);
       ++byte_offset) {
    SCOPED_TRACE(base::StringPrintf("byte offset = %u\n", byte_offset));
    EXPECT_EQ(
        reinterpret_cast<const char*>(&actual.x86_64->fxsave)[byte_offset],
        reinterpret_cast<const char*>(&expected.__fpregs_mem)[byte_offset]);
  }
}
#elif defined(ARCH_CPU_ARMEL)
// A native ucontext_t on ARM doesn't have enough regspace (yet) to hold all of
// the different possible coprocessor contexts at once. However, the ABI allows
// it and the native regspace may be expanded in the future. Append some extra
// space so this is testable now.
struct NativeCPUContext {
  ucontext_t ucontext;
  char extra[1024];
};

struct CrunchContext {
  uint32_t mvdx[16][2];
  uint32_t mvax[4][3];
  uint32_t dspsc[2];
};

struct IWMMXTContext {
  uint32_t save[38];
};

struct TestCoprocessorContext {
  struct {
    internal::CoprocessorContextHead head;
    CrunchContext context;
  } crunch;
  struct {
    internal::CoprocessorContextHead head;
    IWMMXTContext context;
  } iwmmxt;
  struct {
    internal::CoprocessorContextHead head;
    IWMMXTContext context;
  } dummy;
  struct {
    internal::CoprocessorContextHead head;
    internal::SignalVFPContext context;
  } vfp;
  internal::CoprocessorContextHead terminator;
};

void InitializeContext(NativeCPUContext* context) {
  memset(context, 'x', sizeof(*context));

  for (int index = 0; index < (&context->ucontext.uc_mcontext.fault_address -
                               &context->ucontext.uc_mcontext.arm_r0);
       ++index) {
    (&context->ucontext.uc_mcontext.arm_r0)[index] = index;
  }

  static_assert(
      sizeof(TestCoprocessorContext) <=
          sizeof(context->ucontext.uc_regspace) + sizeof(context->extra),
      "Insufficient context space");
  auto test_context =
      reinterpret_cast<TestCoprocessorContext*>(context->ucontext.uc_regspace);

  test_context->crunch.head.magic = CRUNCH_MAGIC;
  test_context->crunch.head.size = sizeof(test_context->crunch);
  memset(
      &test_context->crunch.context, 'c', sizeof(test_context->crunch.context));

  test_context->iwmmxt.head.magic = IWMMXT_MAGIC;
  test_context->iwmmxt.head.size = sizeof(test_context->iwmmxt);
  memset(
      &test_context->iwmmxt.context, 'i', sizeof(test_context->iwmmxt.context));

  test_context->dummy.head.magic = DUMMY_MAGIC;
  test_context->dummy.head.size = sizeof(test_context->dummy);
  memset(
      &test_context->dummy.context, 'd', sizeof(test_context->dummy.context));

  test_context->vfp.head.magic = VFP_MAGIC;
  test_context->vfp.head.size = sizeof(test_context->vfp);
  memset(&test_context->vfp.context, 'v', sizeof(test_context->vfp.context));
  for (size_t reg = 0; reg < arraysize(test_context->vfp.context.vfp.fpregs);
       ++reg) {
    test_context->vfp.context.vfp.fpregs[reg] = reg;
  }
  test_context->vfp.context.vfp.fpscr = 42;

  test_context->terminator.magic = 0;
  test_context->terminator.size = 0;
}

void ExpectContext(const CPUContext& actual, const NativeCPUContext& expected) {
  EXPECT_EQ(actual.architecture, kCPUArchitectureARM);

  EXPECT_EQ(memcmp(actual.arm->regs,
                   &expected.ucontext.uc_mcontext.arm_r0,
                   sizeof(actual.arm->regs)),
            0);
  EXPECT_EQ(actual.arm->fp, expected.ucontext.uc_mcontext.arm_fp);
  EXPECT_EQ(actual.arm->ip, expected.ucontext.uc_mcontext.arm_ip);
  EXPECT_EQ(actual.arm->sp, expected.ucontext.uc_mcontext.arm_sp);
  EXPECT_EQ(actual.arm->lr, expected.ucontext.uc_mcontext.arm_lr);
  EXPECT_EQ(actual.arm->pc, expected.ucontext.uc_mcontext.arm_pc);
  EXPECT_EQ(actual.arm->cpsr, expected.ucontext.uc_mcontext.arm_cpsr);

  EXPECT_FALSE(actual.arm->have_fpa_regs);

  EXPECT_TRUE(actual.arm->have_vfp_regs);

  auto test_context = reinterpret_cast<const TestCoprocessorContext*>(
      expected.ucontext.uc_regspace);

  EXPECT_EQ(memcmp(actual.arm->vfp_regs.vfp,
                   &test_context->vfp.context.vfp,
                   sizeof(actual.arm->vfp_regs.vfp)),
            0);
}
#elif defined(ARCH_CPU_ARM64)
using NativeCPUContext = ucontext_t;

struct TestCoprocessorContext {
  esr_context esr;
  fpsimd_context fpsimd;
  _aarch64_ctx terminator;
};

void InitializeContext(NativeCPUContext* context) {
  memset(context, 'x', sizeof(*context));

  for (size_t index = 0; index < arraysize(context->uc_mcontext.regs);
       ++index) {
    context->uc_mcontext.regs[index] = index;
  }
  context->uc_mcontext.sp = 1;
  context->uc_mcontext.pc = 2;
  context->uc_mcontext.pstate = 3;

  auto test_context = reinterpret_cast<TestCoprocessorContext*>(
      context->uc_mcontext.__reserved);

  test_context->esr.head.magic = ESR_MAGIC;
  test_context->esr.head.size = sizeof(test_context->esr);
  memset(&test_context->esr.esr, 'e', sizeof(test_context->esr.esr));

  test_context->fpsimd.head.magic = FPSIMD_MAGIC;
  test_context->fpsimd.head.size = sizeof(test_context->fpsimd);
  test_context->fpsimd.fpsr = 1;
  test_context->fpsimd.fpcr = 2;
  for (size_t reg = 0; reg < arraysize(test_context->fpsimd.vregs); ++reg) {
    test_context->fpsimd.vregs[reg] = reg;
  }

  test_context->terminator.magic = 0;
  test_context->terminator.size = 0;
}

void ExpectContext(const CPUContext& actual, const NativeCPUContext& expected) {
  EXPECT_EQ(actual.architecture, kCPUArchitectureARM64);

  EXPECT_EQ(memcmp(actual.arm64->regs,
                   expected.uc_mcontext.regs,
                   sizeof(actual.arm64->regs)),
            0);
  EXPECT_EQ(actual.arm64->sp, expected.uc_mcontext.sp);
  EXPECT_EQ(actual.arm64->pc, expected.uc_mcontext.pc);
  EXPECT_EQ(actual.arm64->pstate, expected.uc_mcontext.pstate);

  auto test_context = reinterpret_cast<const TestCoprocessorContext*>(
      expected.uc_mcontext.__reserved);

  EXPECT_EQ(actual.arm64->fpsr, test_context->fpsimd.fpsr);
  EXPECT_EQ(actual.arm64->fpcr, test_context->fpsimd.fpcr);
  EXPECT_EQ(memcmp(actual.arm64->fpsimd,
                   &test_context->fpsimd.vregs,
                   sizeof(actual.arm64->fpsimd)),
            0);
}
#elif defined(ARCH_CPU_MIPS_FAMILY)
using NativeCPUContext = ucontext_t;

void InitializeContext(NativeCPUContext* context) {
  for (size_t reg = 0; reg < arraysize(context->uc_mcontext.gregs); ++reg) {
    context->uc_mcontext.gregs[reg] = reg;
  }
  memset(&context->uc_mcontext.fpregs, 44, sizeof(context->uc_mcontext.fpregs));
}

void ExpectContext(const CPUContext& actual, const NativeCPUContext& expected) {
#if defined(ARCH_CPU_MIPSEL)
  EXPECT_EQ(actual.architecture, kCPUArchitectureMIPSEL);
#define CPU_ARCH_NAME mipsel
#elif defined(ARCH_CPU_MIPS64EL)
  EXPECT_EQ(actual.architecture, kCPUArchitectureMIPS64EL);
#define CPU_ARCH_NAME mips64
#endif

  for (size_t reg = 0; reg < arraysize(expected.uc_mcontext.gregs); ++reg) {
    EXPECT_EQ(actual.CPU_ARCH_NAME->regs[reg], expected.uc_mcontext.gregs[reg]);
  }

  EXPECT_EQ(memcmp(&actual.CPU_ARCH_NAME->fpregs,
                   &expected.uc_mcontext.fpregs,
                   sizeof(actual.CPU_ARCH_NAME->fpregs)),
            0);
#undef CPU_ARCH_NAME
}

#else
#error Port.
#endif

TEST(ExceptionSnapshotLinux, SelfBasic) {
  FakePtraceConnection connection;
  ASSERT_TRUE(connection.Initialize(getpid()));

  ProcessReaderLinux process_reader;
  ASSERT_TRUE(process_reader.Initialize(&connection));

  siginfo_t siginfo;
  siginfo.si_signo = SIGSEGV;
  siginfo.si_errno = 42;
  siginfo.si_code = SEGV_MAPERR;
  siginfo.si_addr = reinterpret_cast<void*>(0xdeadbeef);

  NativeCPUContext context;
  InitializeContext(&context);

  internal::ExceptionSnapshotLinux exception;
  ASSERT_TRUE(exception.Initialize(&process_reader,
                                   FromPointerCast<LinuxVMAddress>(&siginfo),
                                   FromPointerCast<LinuxVMAddress>(&context),
                                   gettid()));
  EXPECT_EQ(exception.Exception(), static_cast<uint32_t>(siginfo.si_signo));
  EXPECT_EQ(exception.ExceptionInfo(), static_cast<uint32_t>(siginfo.si_code));
  EXPECT_EQ(exception.ExceptionAddress(),
            FromPointerCast<uint64_t>(siginfo.si_addr));
  ExpectContext(*exception.Context(), context);
}

class ScopedSigactionRestore {
 public:
  ScopedSigactionRestore() : old_action_(), signo_(-1), valid_(false) {}

  ~ScopedSigactionRestore() { Reset(); }

  bool Reset() {
    if (valid_) {
      int res = sigaction(signo_, &old_action_, nullptr);
      EXPECT_EQ(res, 0) << ErrnoMessage("sigaction");
      if (res != 0) {
        return false;
      }
    }
    valid_ = false;
    signo_ = -1;
    return true;
  }

  bool ResetInstallHandler(int signo, Signals::Handler handler) {
    if (Reset() && Signals::InstallHandler(signo, handler, 0, &old_action_)) {
      signo_ = signo;
      valid_ = true;
      return true;
    }
    return false;
  }

 private:
  struct sigaction old_action_;
  int signo_;
  bool valid_;

  DISALLOW_COPY_AND_ASSIGN(ScopedSigactionRestore);
};

class RaiseTest {
 public:
  static void Run() {
    test_complete_ = false;

    ScopedSigactionRestore sigrestore;
    ASSERT_TRUE(sigrestore.ResetInstallHandler(kSigno, HandleRaisedSignal));

    EXPECT_EQ(raise(kSigno), 0) << ErrnoMessage("raise");
    EXPECT_TRUE(test_complete_);
  }

 private:
  static void HandleRaisedSignal(int signo, siginfo_t* siginfo, void* context) {
    FakePtraceConnection connection;
    ASSERT_TRUE(connection.Initialize(getpid()));

    ProcessReaderLinux process_reader;
    ASSERT_TRUE(process_reader.Initialize(&connection));

    internal::ExceptionSnapshotLinux exception;
    ASSERT_TRUE(exception.Initialize(&process_reader,
                                     FromPointerCast<LinuxVMAddress>(siginfo),
                                     FromPointerCast<LinuxVMAddress>(context),
                                     gettid()));

    EXPECT_EQ(exception.Exception(), static_cast<uint32_t>(kSigno));

    EXPECT_EQ(exception.Codes().size(), 3u);
    EXPECT_EQ(exception.Codes()[0], static_cast<uint64_t>(getpid()));
    EXPECT_EQ(exception.Codes()[1], getuid());
    // Codes()[2] is not set by kill, but we still expect to get it because some
    // interfaces may set it and we don't necessarily know where this signal
    // came
    // from.

    test_complete_ = true;
  }

  static constexpr uint32_t kSigno = SIGUSR1;
  static bool test_complete_;

  DISALLOW_IMPLICIT_CONSTRUCTORS(RaiseTest);
};
bool RaiseTest::test_complete_ = false;

TEST(ExceptionSnapshotLinux, Raise) {
  RaiseTest::Run();
}

class TimerTest {
 public:
  TimerTest() : event_(), timer_(-1), test_complete_(0) { test_ = this; }
  ~TimerTest() { test_ = nullptr; }

  void Run() {
    ScopedSigactionRestore sigrestore;
    ASSERT_TRUE(sigrestore.ResetInstallHandler(kSigno, HandleTimer));

    event_.sigev_notify = SIGEV_SIGNAL;
    event_.sigev_signo = kSigno;
    event_.sigev_value.sival_int = 42;
    ASSERT_EQ(syscall(SYS_timer_create, CLOCK_MONOTONIC, &event_, &timer_), 0);

    itimerspec spec;
    spec.it_interval.tv_sec = 0;
    spec.it_interval.tv_nsec = 0;
    spec.it_value.tv_sec = 0;
    spec.it_value.tv_nsec = 1;
    ASSERT_EQ(syscall(SYS_timer_settime, timer_, TIMER_ABSTIME, &spec, nullptr),
              0);

    ASSERT_TRUE(test_complete_.TimedWait(5));
  }

 private:
  static void HandleTimer(int signo, siginfo_t* siginfo, void* context) {
    FakePtraceConnection connection;
    ASSERT_TRUE(connection.Initialize(getpid()));

    ProcessReaderLinux process_reader;
    ASSERT_TRUE(process_reader.Initialize(&connection));

    internal::ExceptionSnapshotLinux exception;
    ASSERT_TRUE(exception.Initialize(&process_reader,
                                     FromPointerCast<LinuxVMAddress>(siginfo),
                                     FromPointerCast<LinuxVMAddress>(context),
                                     gettid()));

    EXPECT_EQ(exception.Exception(), static_cast<uint32_t>(kSigno));

    EXPECT_EQ(exception.Codes().size(), 3u);
    EXPECT_EQ(exception.Codes()[0], static_cast<uint64_t>(test_->timer_));
    int overruns = syscall(SYS_timer_getoverrun, test_->timer_);
    ASSERT_GE(overruns, 0);
    EXPECT_EQ(exception.Codes()[1], static_cast<uint64_t>(overruns));
    EXPECT_EQ(exception.Codes()[2],
              static_cast<uint64_t>(test_->event_.sigev_value.sival_int));

    test_->test_complete_.Signal();
  }

  sigevent event_;
  __kernel_timer_t timer_;
  Semaphore test_complete_;

  static constexpr uint32_t kSigno = SIGALRM;
  static TimerTest* test_;

  DISALLOW_COPY_AND_ASSIGN(TimerTest);
};
TimerTest* TimerTest::test_;

TEST(ExceptionSnapshotLinux, SelfTimer) {
  TimerTest test;
  test.Run();
}

}  // namespace
}  // namespace test
}  // namespace crashpad
