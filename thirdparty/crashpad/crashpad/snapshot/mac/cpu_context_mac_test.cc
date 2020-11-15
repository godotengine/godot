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

#include "snapshot/mac/cpu_context_mac.h"

#include <mach/mach.h>

#include "gtest/gtest.h"

namespace crashpad {
namespace test {
namespace {

#if defined(ARCH_CPU_X86_FAMILY)

TEST(CPUContextMac, InitializeContextX86) {
  x86_thread_state32_t x86_thread_state32 = {};
  x86_float_state32_t x86_float_state32 = {};
  x86_debug_state32_t x86_debug_state32 = {};
  x86_thread_state32.__eax = 1;
  x86_float_state32.__fpu_ftw = 2;
  x86_debug_state32.__dr0 = 3;

  // Test the simple case, where everything in the CPUContextX86 argument is set
  // directly from the supplied thread, float, and debug state parameters.
  {
    CPUContextX86 cpu_context_x86 = {};
    internal::InitializeCPUContextX86(&cpu_context_x86,
                                      THREAD_STATE_NONE,
                                      nullptr,
                                      0,
                                      &x86_thread_state32,
                                      &x86_float_state32,
                                      &x86_debug_state32);
    EXPECT_EQ(cpu_context_x86.eax, 1u);
    EXPECT_EQ(cpu_context_x86.fxsave.ftw, 2u);
    EXPECT_EQ(cpu_context_x86.dr0, 3u);
  }

  // Supply context in a CPU-specific “flavor” parameter expected to be used
  // instead of the supplied thread, float, or debug state parameters. Do this
  // once for each of the three valid flavors. This simulates how
  // InitializeCPUContextX86() might be used to initialize the context in an
  // exception handler, where the exception handler may have received the
  // “flavor” parameter and this context should be used to initialize the
  // CPUContextX86.

  {
    x86_thread_state32_t alt_x86_thread_state32 = {};
    alt_x86_thread_state32.__eax = 4;

    CPUContextX86 cpu_context_x86 = {};
    internal::InitializeCPUContextX86(
        &cpu_context_x86,
        x86_THREAD_STATE32,
        reinterpret_cast<natural_t*>(&alt_x86_thread_state32),
        x86_THREAD_STATE32_COUNT,
        &x86_thread_state32,
        &x86_float_state32,
        &x86_debug_state32);
    EXPECT_EQ(cpu_context_x86.eax, 4u);
    EXPECT_EQ(cpu_context_x86.fxsave.ftw, 2u);
    EXPECT_EQ(cpu_context_x86.dr0, 3u);
  }

  {
    x86_float_state32_t alt_x86_float_state32 = {};
    alt_x86_float_state32.__fpu_ftw = 5;

    CPUContextX86 cpu_context_x86 = {};
    internal::InitializeCPUContextX86(
        &cpu_context_x86,
        x86_FLOAT_STATE32,
        reinterpret_cast<natural_t*>(&alt_x86_float_state32),
        x86_FLOAT_STATE32_COUNT,
        &x86_thread_state32,
        &x86_float_state32,
        &x86_debug_state32);
    EXPECT_EQ(cpu_context_x86.eax, 1u);
    EXPECT_EQ(cpu_context_x86.fxsave.ftw, 5u);
    EXPECT_EQ(cpu_context_x86.dr0, 3u);
  }

  {
    x86_debug_state32_t alt_x86_debug_state32 = {};
    alt_x86_debug_state32.__dr0 = 6;

    CPUContextX86 cpu_context_x86 = {};
    internal::InitializeCPUContextX86(
        &cpu_context_x86,
        x86_DEBUG_STATE32,
        reinterpret_cast<natural_t*>(&alt_x86_debug_state32),
        x86_DEBUG_STATE32_COUNT,
        &x86_thread_state32,
        &x86_float_state32,
        &x86_debug_state32);
    EXPECT_EQ(cpu_context_x86.eax, 1u);
    EXPECT_EQ(cpu_context_x86.fxsave.ftw, 2u);
    EXPECT_EQ(cpu_context_x86.dr0, 6u);
  }

  // Supply context in a universal “flavor” parameter expected to be used
  // instead of the supplied thread, float, or debug state parameters. The
  // universal format allows an exception handler to be registered to receive
  // thread, float, or debug state without having to know in advance whether it
  // will be receiving the state from a 32-bit or 64-bit process. For
  // CPUContextX86, only the 32-bit form is supported.

  {
    x86_thread_state x86_thread_state_3264 = {};
    x86_thread_state_3264.tsh.flavor = x86_THREAD_STATE32;
    x86_thread_state_3264.tsh.count = x86_THREAD_STATE32_COUNT;
    x86_thread_state_3264.uts.ts32.__eax = 7;

    CPUContextX86 cpu_context_x86 = {};
    internal::InitializeCPUContextX86(
        &cpu_context_x86,
        x86_THREAD_STATE,
        reinterpret_cast<natural_t*>(&x86_thread_state_3264),
        x86_THREAD_STATE_COUNT,
        &x86_thread_state32,
        &x86_float_state32,
        &x86_debug_state32);
    EXPECT_EQ(cpu_context_x86.eax, 7u);
    EXPECT_EQ(cpu_context_x86.fxsave.ftw, 2u);
    EXPECT_EQ(cpu_context_x86.dr0, 3u);
  }

  {
    x86_float_state x86_float_state_3264 = {};
    x86_float_state_3264.fsh.flavor = x86_FLOAT_STATE32;
    x86_float_state_3264.fsh.count = x86_FLOAT_STATE32_COUNT;
    x86_float_state_3264.ufs.fs32.__fpu_ftw = 8;

    CPUContextX86 cpu_context_x86 = {};
    internal::InitializeCPUContextX86(
        &cpu_context_x86,
        x86_FLOAT_STATE,
        reinterpret_cast<natural_t*>(&x86_float_state_3264),
        x86_FLOAT_STATE_COUNT,
        &x86_thread_state32,
        &x86_float_state32,
        &x86_debug_state32);
    EXPECT_EQ(cpu_context_x86.eax, 1u);
    EXPECT_EQ(cpu_context_x86.fxsave.ftw, 8u);
    EXPECT_EQ(cpu_context_x86.dr0, 3u);
  }

  {
    x86_debug_state x86_debug_state_3264 = {};
    x86_debug_state_3264.dsh.flavor = x86_DEBUG_STATE32;
    x86_debug_state_3264.dsh.count = x86_DEBUG_STATE32_COUNT;
    x86_debug_state_3264.uds.ds32.__dr0 = 9;

    CPUContextX86 cpu_context_x86 = {};
    internal::InitializeCPUContextX86(
        &cpu_context_x86,
        x86_DEBUG_STATE,
        reinterpret_cast<natural_t*>(&x86_debug_state_3264),
        x86_DEBUG_STATE_COUNT,
        &x86_thread_state32,
        &x86_float_state32,
        &x86_debug_state32);
    EXPECT_EQ(cpu_context_x86.eax, 1u);
    EXPECT_EQ(cpu_context_x86.fxsave.ftw, 2u);
    EXPECT_EQ(cpu_context_x86.dr0, 9u);
  }

  // Supply inappropriate “flavor” contexts to test that
  // InitializeCPUContextX86() detects the problem and refuses to use the
  // supplied “flavor” context, falling back to the thread, float, and debug
  // states.

  {
    x86_thread_state64_t x86_thread_state64 = {};

    CPUContextX86 cpu_context_x86 = {};
    internal::InitializeCPUContextX86(
        &cpu_context_x86,
        x86_THREAD_STATE64,
        reinterpret_cast<natural_t*>(&x86_thread_state64),
        x86_THREAD_STATE64_COUNT,
        &x86_thread_state32,
        &x86_float_state32,
        &x86_debug_state32);
    EXPECT_EQ(cpu_context_x86.eax, 1u);
    EXPECT_EQ(cpu_context_x86.fxsave.ftw, 2u);
    EXPECT_EQ(cpu_context_x86.dr0, 3u);
  }

  {
    x86_thread_state x86_thread_state_3264 = {};
    x86_thread_state_3264.tsh.flavor = x86_THREAD_STATE64;
    x86_thread_state_3264.tsh.count = x86_THREAD_STATE64_COUNT;

    CPUContextX86 cpu_context_x86 = {};
    internal::InitializeCPUContextX86(
        &cpu_context_x86,
        x86_THREAD_STATE,
        reinterpret_cast<natural_t*>(&x86_thread_state_3264),
        x86_THREAD_STATE_COUNT,
        &x86_thread_state32,
        &x86_float_state32,
        &x86_debug_state32);
    EXPECT_EQ(cpu_context_x86.eax, 1u);
    EXPECT_EQ(cpu_context_x86.fxsave.ftw, 2u);
    EXPECT_EQ(cpu_context_x86.dr0, 3u);
  }
}

TEST(CPUContextMac, InitializeContextX86_64) {
  x86_thread_state64_t x86_thread_state64 = {};
  x86_float_state64_t x86_float_state64 = {};
  x86_debug_state64_t x86_debug_state64 = {};
  x86_thread_state64.__rax = 10;
  x86_float_state64.__fpu_ftw = 11;
  x86_debug_state64.__dr0 = 12;

  // Test the simple case, where everything in the CPUContextX86_64 argument is
  // set directly from the supplied thread, float, and debug state parameters.
  {
    CPUContextX86_64 cpu_context_x86_64 = {};
    internal::InitializeCPUContextX86_64(&cpu_context_x86_64,
                                         THREAD_STATE_NONE,
                                         nullptr,
                                         0,
                                         &x86_thread_state64,
                                         &x86_float_state64,
                                         &x86_debug_state64);
    EXPECT_EQ(cpu_context_x86_64.rax, 10u);
    EXPECT_EQ(cpu_context_x86_64.fxsave.ftw, 11u);
    EXPECT_EQ(cpu_context_x86_64.dr0, 12u);
  }

  // Supply context in a CPU-specific “flavor” parameter expected to be used
  // instead of the supplied thread, float, or debug state parameters. Do this
  // once for each of the three valid flavors. This simulates how
  // InitializeCPUContextX86_64() might be used to initialize the context in an
  // exception handler, where the exception handler may have received the
  // “flavor” parameter and this context should be used to initialize the
  // CPUContextX86_64.

  {
    x86_thread_state64_t alt_x86_thread_state64 = {};
    alt_x86_thread_state64.__rax = 13;

    CPUContextX86_64 cpu_context_x86_64 = {};
    internal::InitializeCPUContextX86_64(
        &cpu_context_x86_64,
        x86_THREAD_STATE64,
        reinterpret_cast<natural_t*>(&alt_x86_thread_state64),
        x86_THREAD_STATE64_COUNT,
        &x86_thread_state64,
        &x86_float_state64,
        &x86_debug_state64);
    EXPECT_EQ(cpu_context_x86_64.rax, 13u);
    EXPECT_EQ(cpu_context_x86_64.fxsave.ftw, 11u);
    EXPECT_EQ(cpu_context_x86_64.dr0, 12u);
  }

  {
    x86_float_state64_t alt_x86_float_state64 = {};
    alt_x86_float_state64.__fpu_ftw = 14;

    CPUContextX86_64 cpu_context_x86_64 = {};
    internal::InitializeCPUContextX86_64(
        &cpu_context_x86_64,
        x86_FLOAT_STATE64,
        reinterpret_cast<natural_t*>(&alt_x86_float_state64),
        x86_FLOAT_STATE64_COUNT,
        &x86_thread_state64,
        &x86_float_state64,
        &x86_debug_state64);
    EXPECT_EQ(cpu_context_x86_64.rax, 10u);
    EXPECT_EQ(cpu_context_x86_64.fxsave.ftw, 14u);
    EXPECT_EQ(cpu_context_x86_64.dr0, 12u);
  }

  {
    x86_debug_state64_t alt_x86_debug_state64 = {};
    alt_x86_debug_state64.__dr0 = 15;

    CPUContextX86_64 cpu_context_x86_64 = {};
    internal::InitializeCPUContextX86_64(
        &cpu_context_x86_64,
        x86_DEBUG_STATE64,
        reinterpret_cast<natural_t*>(&alt_x86_debug_state64),
        x86_DEBUG_STATE64_COUNT,
        &x86_thread_state64,
        &x86_float_state64,
        &x86_debug_state64);
    EXPECT_EQ(cpu_context_x86_64.rax, 10u);
    EXPECT_EQ(cpu_context_x86_64.fxsave.ftw, 11u);
    EXPECT_EQ(cpu_context_x86_64.dr0, 15u);
  }

  // Supply context in a universal “flavor” parameter expected to be used
  // instead of the supplied thread, float, or debug state parameters. The
  // universal format allows an exception handler to be registered to receive
  // thread, float, or debug state without having to know in advance whether it
  // will be receiving the state from a 32-bit or 64-bit process. For
  // CPUContextX86_64, only the 64-bit form is supported.

  {
    x86_thread_state x86_thread_state_3264 = {};
    x86_thread_state_3264.tsh.flavor = x86_THREAD_STATE64;
    x86_thread_state_3264.tsh.count = x86_THREAD_STATE64_COUNT;
    x86_thread_state_3264.uts.ts64.__rax = 16;

    CPUContextX86_64 cpu_context_x86_64 = {};
    internal::InitializeCPUContextX86_64(
        &cpu_context_x86_64,
        x86_THREAD_STATE,
        reinterpret_cast<natural_t*>(&x86_thread_state_3264),
        x86_THREAD_STATE_COUNT,
        &x86_thread_state64,
        &x86_float_state64,
        &x86_debug_state64);
    EXPECT_EQ(cpu_context_x86_64.rax, 16u);
    EXPECT_EQ(cpu_context_x86_64.fxsave.ftw, 11u);
    EXPECT_EQ(cpu_context_x86_64.dr0, 12u);
  }

  {
    x86_float_state x86_float_state_3264 = {};
    x86_float_state_3264.fsh.flavor = x86_FLOAT_STATE64;
    x86_float_state_3264.fsh.count = x86_FLOAT_STATE64_COUNT;
    x86_float_state_3264.ufs.fs64.__fpu_ftw = 17;

    CPUContextX86_64 cpu_context_x86_64 = {};
    internal::InitializeCPUContextX86_64(
        &cpu_context_x86_64,
        x86_FLOAT_STATE,
        reinterpret_cast<natural_t*>(&x86_float_state_3264),
        x86_FLOAT_STATE_COUNT,
        &x86_thread_state64,
        &x86_float_state64,
        &x86_debug_state64);
    EXPECT_EQ(cpu_context_x86_64.rax, 10u);
    EXPECT_EQ(cpu_context_x86_64.fxsave.ftw, 17u);
    EXPECT_EQ(cpu_context_x86_64.dr0, 12u);
  }

  {
    x86_debug_state x86_debug_state_3264 = {};
    x86_debug_state_3264.dsh.flavor = x86_DEBUG_STATE64;
    x86_debug_state_3264.dsh.count = x86_DEBUG_STATE64_COUNT;
    x86_debug_state_3264.uds.ds64.__dr0 = 18;

    CPUContextX86_64 cpu_context_x86_64 = {};
    internal::InitializeCPUContextX86_64(
        &cpu_context_x86_64,
        x86_DEBUG_STATE,
        reinterpret_cast<natural_t*>(&x86_debug_state_3264),
        x86_DEBUG_STATE_COUNT,
        &x86_thread_state64,
        &x86_float_state64,
        &x86_debug_state64);
    EXPECT_EQ(cpu_context_x86_64.rax, 10u);
    EXPECT_EQ(cpu_context_x86_64.fxsave.ftw, 11u);
    EXPECT_EQ(cpu_context_x86_64.dr0, 18u);
  }

  // Supply inappropriate “flavor” contexts to test that
  // InitializeCPUContextX86() detects the problem and refuses to use the
  // supplied “flavor” context, falling back to the thread, float, and debug
  // states.

  {
    x86_thread_state32_t x86_thread_state32 = {};

    CPUContextX86_64 cpu_context_x86_64 = {};
    internal::InitializeCPUContextX86_64(
        &cpu_context_x86_64,
        x86_THREAD_STATE32,
        reinterpret_cast<natural_t*>(&x86_thread_state32),
        x86_THREAD_STATE32_COUNT,
        &x86_thread_state64,
        &x86_float_state64,
        &x86_debug_state64);
    EXPECT_EQ(cpu_context_x86_64.rax, 10u);
    EXPECT_EQ(cpu_context_x86_64.fxsave.ftw, 11u);
    EXPECT_EQ(cpu_context_x86_64.dr0, 12u);
  }

  {
    x86_thread_state x86_thread_state_3264 = {};
    x86_thread_state_3264.tsh.flavor = x86_THREAD_STATE32;
    x86_thread_state_3264.tsh.count = x86_THREAD_STATE32_COUNT;

    CPUContextX86_64 cpu_context_x86_64 = {};
    internal::InitializeCPUContextX86_64(
        &cpu_context_x86_64,
        x86_THREAD_STATE,
        reinterpret_cast<natural_t*>(&x86_thread_state_3264),
        x86_THREAD_STATE_COUNT,
        &x86_thread_state64,
        &x86_float_state64,
        &x86_debug_state64);
    EXPECT_EQ(cpu_context_x86_64.rax, 10u);
    EXPECT_EQ(cpu_context_x86_64.fxsave.ftw, 11u);
    EXPECT_EQ(cpu_context_x86_64.dr0, 12u);
  }
}

#endif

}  // namespace
}  // namespace test
}  // namespace crashpad
