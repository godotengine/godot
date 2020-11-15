// Copyright 2015 The Crashpad Authors. All rights reserved.
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

#include "util/mach/exception_types.h"

#include <kern/exc_resource.h>
#include <signal.h>
#include <stdint.h>
#include <sys/types.h>
#include <unistd.h>

#include "base/macros.h"
#include "base/strings/stringprintf.h"
#include "gtest/gtest.h"
#include "util/mac/mac_util.h"
#include "util/mach/mach_extensions.h"

namespace crashpad {
namespace test {
namespace {

TEST(ExceptionTypes, ExcCrashRecoverOriginalException) {
  static constexpr struct {
    mach_exception_code_t code_0;
    exception_type_t exception;
    mach_exception_code_t original_code_0;
    int signal;
  } kTestData[] = {
      {0xb100001, EXC_BAD_ACCESS, KERN_INVALID_ADDRESS, SIGSEGV},
      {0xb100002, EXC_BAD_ACCESS, KERN_PROTECTION_FAILURE, SIGSEGV},
      {0xa100002, EXC_BAD_ACCESS, KERN_PROTECTION_FAILURE, SIGBUS},
      {0xa100005, EXC_BAD_ACCESS, VM_PROT_READ | VM_PROT_EXECUTE, SIGBUS},
      {0xa10000d, EXC_BAD_ACCESS, EXC_I386_GPFLT, SIGBUS},
      {0x9100032, EXC_BAD_ACCESS, KERN_CODESIGN_ERROR, SIGKILL},
      {0x4200001, EXC_BAD_INSTRUCTION, EXC_I386_INVOP, SIGILL},
      {0x420000b, EXC_BAD_INSTRUCTION, EXC_I386_SEGNPFLT, SIGILL},
      {0x420000c, EXC_BAD_INSTRUCTION, EXC_I386_STKFLT, SIGILL},
      {0x8300001, EXC_ARITHMETIC, EXC_I386_DIV, SIGFPE},
      {0x8300002, EXC_ARITHMETIC, EXC_I386_INTO, SIGFPE},
      {0x8300005, EXC_ARITHMETIC, EXC_I386_EXTERR, SIGFPE},
      {0x8300008, EXC_ARITHMETIC, EXC_I386_SSEEXTERR, SIGFPE},
      {0x5500007, EXC_SOFTWARE, EXC_I386_BOUND, SIGTRAP},
      {0x5600001, EXC_BREAKPOINT, EXC_I386_SGL, SIGTRAP},
      {0x5600002, EXC_BREAKPOINT, EXC_I386_BPT, SIGTRAP},
      {0x0700080, EXC_SYSCALL, 128, 0},
      {0x0706000, EXC_SYSCALL, 0x6000, 0},
      {0x3000000, 0, 0, SIGQUIT},
      {0x4000000, 0, 0, SIGILL},
      {0x5000000, 0, 0, SIGTRAP},
      {0x6000000, 0, 0, SIGABRT},
      {0x7000000, 0, 0, SIGEMT},
      {0x8000000, 0, 0, SIGFPE},
      {0xa000000, 0, 0, SIGBUS},
      {0xb000000, 0, 0, SIGSEGV},
      {0xc000000, 0, 0, SIGSYS},
      {0, 0, 0, 0},
  };

  for (size_t index = 0; index < arraysize(kTestData); ++index) {
    const auto& test_data = kTestData[index];
    SCOPED_TRACE(base::StringPrintf(
        "index %zu, code_0 0x%llx", index, test_data.code_0));

    mach_exception_code_t original_code_0;
    int signal;
    exception_type_t exception = ExcCrashRecoverOriginalException(
        test_data.code_0, &original_code_0, &signal);

    EXPECT_EQ(exception, test_data.exception);
    EXPECT_EQ(original_code_0, test_data.original_code_0);
    EXPECT_EQ(signal, test_data.signal);
  }

  // Now make sure that ExcCrashRecoverOriginalException() properly ignores
  // optional arguments.
  static_assert(arraysize(kTestData) >= 1, "must have something to test");
  const auto& test_data = kTestData[0];
  EXPECT_EQ(
      ExcCrashRecoverOriginalException(test_data.code_0, nullptr, nullptr),
      test_data.exception);

  mach_exception_code_t original_code_0;
  EXPECT_EQ(ExcCrashRecoverOriginalException(
                test_data.code_0, &original_code_0, nullptr),
            test_data.exception);
  EXPECT_EQ(original_code_0, test_data.original_code_0);

  int signal;
  EXPECT_EQ(
      ExcCrashRecoverOriginalException(test_data.code_0, nullptr, &signal),
      test_data.exception);
  EXPECT_EQ(signal, test_data.signal);
}

TEST(ExceptionTypes, ExcCrashCouldContainException) {
  // This seems wrong, but it happens when EXC_CRASH carries an exception not
  // originally caused by a hardware fault, such as SIGABRT.
  EXPECT_TRUE(ExcCrashCouldContainException(0));

  EXPECT_TRUE(ExcCrashCouldContainException(EXC_BAD_ACCESS));
  EXPECT_TRUE(ExcCrashCouldContainException(EXC_BAD_INSTRUCTION));
  EXPECT_TRUE(ExcCrashCouldContainException(EXC_ARITHMETIC));
  EXPECT_TRUE(ExcCrashCouldContainException(EXC_EMULATION));
  EXPECT_TRUE(ExcCrashCouldContainException(EXC_SOFTWARE));
  EXPECT_TRUE(ExcCrashCouldContainException(EXC_BREAKPOINT));
  EXPECT_TRUE(ExcCrashCouldContainException(EXC_SYSCALL));
  EXPECT_TRUE(ExcCrashCouldContainException(EXC_MACH_SYSCALL));
  EXPECT_TRUE(ExcCrashCouldContainException(EXC_RPC_ALERT));
  EXPECT_FALSE(ExcCrashCouldContainException(EXC_CRASH));
  EXPECT_FALSE(ExcCrashCouldContainException(EXC_RESOURCE));
  EXPECT_FALSE(ExcCrashCouldContainException(EXC_GUARD));
  EXPECT_FALSE(ExcCrashCouldContainException(EXC_CORPSE_NOTIFY));
  EXPECT_FALSE(ExcCrashCouldContainException(kMachExceptionSimulated));
}

// This macro is adapted from those in the #ifdef KERNEL section of
// <kern/exc_resource.h>: 10.12.2 xnu-3789.31.2/osfmk/kern/exc_resource.h.
#define EXC_RESOURCE_ENCODE_TYPE_FLAVOR(type, flavor)   \
  (static_cast<mach_exception_code_t>(                  \
      (((static_cast<uint64_t>(type) & 0x7ull) << 61) | \
       (static_cast<uint64_t>(flavor) & 0x7ull) << 58)))

TEST(ExceptionTypes, ExceptionCodeForMetrics) {
  static constexpr struct {
    exception_type_t exception;
    mach_exception_code_t code_0;
    int32_t metrics_code;
  } kTestData[] = {
#define ENCODE_EXC(type, code_0) \
  { (type), (code_0), ((type) << 16) | (code_0) }
      ENCODE_EXC(EXC_BAD_ACCESS, KERN_INVALID_ADDRESS),
      ENCODE_EXC(EXC_BAD_ACCESS, KERN_PROTECTION_FAILURE),
      ENCODE_EXC(EXC_BAD_ACCESS, VM_PROT_READ | VM_PROT_EXECUTE),
      ENCODE_EXC(EXC_BAD_ACCESS, EXC_I386_GPFLT),
      ENCODE_EXC(EXC_BAD_ACCESS, KERN_CODESIGN_ERROR),
      ENCODE_EXC(EXC_BAD_INSTRUCTION, EXC_I386_INVOP),
      ENCODE_EXC(EXC_BAD_INSTRUCTION, EXC_I386_SEGNPFLT),
      ENCODE_EXC(EXC_BAD_INSTRUCTION, EXC_I386_STKFLT),
      ENCODE_EXC(EXC_ARITHMETIC, EXC_I386_DIV),
      ENCODE_EXC(EXC_ARITHMETIC, EXC_I386_INTO),
      ENCODE_EXC(EXC_ARITHMETIC, EXC_I386_EXTERR),
      ENCODE_EXC(EXC_ARITHMETIC, EXC_I386_SSEEXTERR),
      ENCODE_EXC(EXC_SOFTWARE, EXC_I386_BOUND),
      ENCODE_EXC(EXC_BREAKPOINT, EXC_I386_SGL),
      ENCODE_EXC(EXC_BREAKPOINT, EXC_I386_BPT),
      ENCODE_EXC(EXC_SYSCALL, 128),
      ENCODE_EXC(EXC_SYSCALL, 0x6000),
#undef ENCODE_EXC

#define ENCODE_EXC_CRASH(type, code_0)                        \
  {                                                           \
    EXC_CRASH, (((type) & 0xf) << 20) | ((code_0) & 0xfffff), \
        ((type) << 16) | (code_0)                             \
  }
      ENCODE_EXC_CRASH(EXC_BAD_ACCESS, KERN_INVALID_ADDRESS),
      ENCODE_EXC_CRASH(EXC_BAD_ACCESS, KERN_PROTECTION_FAILURE),
      ENCODE_EXC_CRASH(EXC_BAD_ACCESS, VM_PROT_READ | VM_PROT_EXECUTE),
      ENCODE_EXC_CRASH(EXC_BAD_ACCESS, EXC_I386_GPFLT),
      ENCODE_EXC_CRASH(EXC_BAD_ACCESS, KERN_CODESIGN_ERROR),
      ENCODE_EXC_CRASH(EXC_BAD_INSTRUCTION, EXC_I386_INVOP),
      ENCODE_EXC_CRASH(EXC_BAD_INSTRUCTION, EXC_I386_SEGNPFLT),
      ENCODE_EXC_CRASH(EXC_BAD_INSTRUCTION, EXC_I386_STKFLT),
      ENCODE_EXC_CRASH(EXC_ARITHMETIC, EXC_I386_DIV),
      ENCODE_EXC_CRASH(EXC_ARITHMETIC, EXC_I386_INTO),
      ENCODE_EXC_CRASH(EXC_ARITHMETIC, EXC_I386_EXTERR),
      ENCODE_EXC_CRASH(EXC_ARITHMETIC, EXC_I386_SSEEXTERR),
      ENCODE_EXC_CRASH(EXC_SOFTWARE, EXC_I386_BOUND),
      ENCODE_EXC_CRASH(EXC_BREAKPOINT, EXC_I386_SGL),
      ENCODE_EXC_CRASH(EXC_BREAKPOINT, EXC_I386_BPT),
      ENCODE_EXC_CRASH(EXC_SYSCALL, 128),
      ENCODE_EXC_CRASH(EXC_SYSCALL, 0x6000),
#undef ENCODE_EXC_CRASH

#define ENCODE_EXC_CRASH_SIGNAL(signal) \
  { EXC_CRASH, (((signal) & 0xff) << 24), (EXC_CRASH << 16) | (signal) }
      ENCODE_EXC_CRASH_SIGNAL(SIGQUIT),
      ENCODE_EXC_CRASH_SIGNAL(SIGILL),
      ENCODE_EXC_CRASH_SIGNAL(SIGTRAP),
      ENCODE_EXC_CRASH_SIGNAL(SIGABRT),
      ENCODE_EXC_CRASH_SIGNAL(SIGEMT),
      ENCODE_EXC_CRASH_SIGNAL(SIGFPE),
      ENCODE_EXC_CRASH_SIGNAL(SIGBUS),
      ENCODE_EXC_CRASH_SIGNAL(SIGSEGV),
      ENCODE_EXC_CRASH_SIGNAL(SIGSYS),
#undef ENCODE_EXC_CRASH_SIGNAL

#define ENCODE_EXC_RESOURCE(type, flavor)                            \
  {                                                                  \
    EXC_RESOURCE, EXC_RESOURCE_ENCODE_TYPE_FLAVOR((type), (flavor)), \
        (EXC_RESOURCE << 16) | ((type) << 8) | (flavor)              \
  }
      ENCODE_EXC_RESOURCE(RESOURCE_TYPE_CPU, FLAVOR_CPU_MONITOR),
      ENCODE_EXC_RESOURCE(RESOURCE_TYPE_CPU, FLAVOR_CPU_MONITOR_FATAL),
      ENCODE_EXC_RESOURCE(RESOURCE_TYPE_WAKEUPS, FLAVOR_WAKEUPS_MONITOR),
      ENCODE_EXC_RESOURCE(RESOURCE_TYPE_MEMORY, FLAVOR_HIGH_WATERMARK),
      ENCODE_EXC_RESOURCE(RESOURCE_TYPE_IO, FLAVOR_IO_PHYSICAL_WRITES),
      ENCODE_EXC_RESOURCE(RESOURCE_TYPE_IO, FLAVOR_IO_LOGICAL_WRITES),
#undef ENCODE_EXC_RESOURCE

#define ENCODE_EXC_GUARD(type, flavor)                                         \
  {                                                                            \
    EXC_GUARD,                                                                 \
        static_cast<mach_exception_code_t>(static_cast<uint64_t>((type) & 0x7) \
                                           << 61) |                            \
            (static_cast<uint64_t>((1 << (flavor)) & 0x1ffffffff) << 32),      \
        (EXC_GUARD << 16) | ((type) << 8) | (flavor)                           \
  }
      ENCODE_EXC_GUARD(GUARD_TYPE_MACH_PORT, 0),  // kGUARD_EXC_DESTROY
      ENCODE_EXC_GUARD(GUARD_TYPE_MACH_PORT, 1),  // kGUARD_EXC_MOD_REFS
      ENCODE_EXC_GUARD(GUARD_TYPE_MACH_PORT, 2),  // kGUARD_EXC_SET_CONTEXT
      ENCODE_EXC_GUARD(GUARD_TYPE_MACH_PORT, 3),  // kGUARD_EXC_UNGUARDED
      ENCODE_EXC_GUARD(GUARD_TYPE_MACH_PORT, 4),  // kGUARD_EXC_INCORRECT_GUARD

      // 2 is GUARD_TYPE_FD from 10.12.2 xnu-3789.31.2/bsd/sys/guarded.h.
      ENCODE_EXC_GUARD(2, 0),  // kGUARD_EXC_CLOSE
      ENCODE_EXC_GUARD(2, 1),  // kGUARD_EXC_DUP
      ENCODE_EXC_GUARD(2, 2),  // kGUARD_EXC_NOCLOEXEC
      ENCODE_EXC_GUARD(2, 3),  // kGUARD_EXC_SOCKET_IPC
      ENCODE_EXC_GUARD(2, 4),  // kGUARD_EXC_FILEPORT
      ENCODE_EXC_GUARD(2, 5),  // kGUARD_EXC_MISMATCH
      ENCODE_EXC_GUARD(2, 6),  // kGUARD_EXC_WRITE
#undef ENCODE_EXC_GUARD

      // Test that overflow saturates.
      {0x00010000, 0x00001000, static_cast<int32_t>(0xffff1000)},
      {0x00001000, 0x00010000, 0x1000ffff},
      {0x00010000, 0x00010000, static_cast<int32_t>(0xffffffff)},
  };

  for (size_t index = 0; index < arraysize(kTestData); ++index) {
    const auto& test_data = kTestData[index];
    SCOPED_TRACE(base::StringPrintf("index %zu, exception 0x%x, code_0 0x%llx",
                                    index,
                                    test_data.exception,
                                    test_data.code_0));

    int32_t metrics_code =
        ExceptionCodeForMetrics(test_data.exception, test_data.code_0);

    EXPECT_EQ(metrics_code, test_data.metrics_code);
  }
}

TEST(ExceptionTypes, IsExceptionNonfatalResource) {
  const pid_t pid = getpid();

  mach_exception_code_t code =
      EXC_RESOURCE_ENCODE_TYPE_FLAVOR(RESOURCE_TYPE_CPU, FLAVOR_CPU_MONITOR);
  EXPECT_TRUE(IsExceptionNonfatalResource(EXC_RESOURCE, code, pid));

  if (MacOSXMinorVersion() >= 10) {
    // FLAVOR_CPU_MONITOR_FATAL was introduced in OS X 10.10.
    code = EXC_RESOURCE_ENCODE_TYPE_FLAVOR(RESOURCE_TYPE_CPU,
                                           FLAVOR_CPU_MONITOR_FATAL);
    EXPECT_FALSE(IsExceptionNonfatalResource(EXC_RESOURCE, code, pid));
  }

  // This assumes that WAKEMON_MAKE_FATAL is not set for this process. The
  // default is for WAKEMON_MAKE_FATAL to not be set, there’s no public API to
  // enable it, and nothing in this process should have enabled it.
  code = EXC_RESOURCE_ENCODE_TYPE_FLAVOR(RESOURCE_TYPE_WAKEUPS,
                                         FLAVOR_WAKEUPS_MONITOR);
  EXPECT_TRUE(IsExceptionNonfatalResource(EXC_RESOURCE, code, pid));

  code = EXC_RESOURCE_ENCODE_TYPE_FLAVOR(RESOURCE_TYPE_MEMORY,
                                         FLAVOR_HIGH_WATERMARK);
  EXPECT_TRUE(IsExceptionNonfatalResource(EXC_RESOURCE, code, pid));

  // Non-EXC_RESOURCE exceptions should never be considered nonfatal resource
  // exceptions, because they aren’t resource exceptions at all.
  EXPECT_FALSE(IsExceptionNonfatalResource(EXC_CRASH, 0xb100001, pid));
  EXPECT_FALSE(IsExceptionNonfatalResource(EXC_CRASH, 0x0b00000, pid));
  EXPECT_FALSE(IsExceptionNonfatalResource(EXC_CRASH, 0x6000000, pid));
  EXPECT_FALSE(
      IsExceptionNonfatalResource(EXC_BAD_ACCESS, KERN_INVALID_ADDRESS, pid));
  EXPECT_FALSE(IsExceptionNonfatalResource(EXC_BAD_INSTRUCTION, 1, pid));
  EXPECT_FALSE(IsExceptionNonfatalResource(EXC_ARITHMETIC, 1, pid));
  EXPECT_FALSE(IsExceptionNonfatalResource(EXC_BREAKPOINT, 2, pid));
  EXPECT_FALSE(IsExceptionNonfatalResource(0, 0, pid));
  EXPECT_FALSE(IsExceptionNonfatalResource(kMachExceptionSimulated, 0, pid));
}

}  // namespace
}  // namespace test
}  // namespace crashpad
