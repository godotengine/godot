// Copyright 2018 The Crashpad Authors. All rights reserved.
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

#include "util/misc/capture_context_test_util.h"

#include "base/macros.h"
#include "gtest/gtest.h"

namespace crashpad {
namespace test {

void SanityCheckContext(const NativeCPUContext& context) {
#if defined(ARCH_CPU_X86)
  constexpr uint32_t must_have = CONTEXT_i386 | CONTEXT_CONTROL |
                                 CONTEXT_INTEGER | CONTEXT_SEGMENTS |
                                 CONTEXT_FLOATING_POINT;
  ASSERT_EQ(context.ContextFlags & must_have, must_have);
  constexpr uint32_t may_have = CONTEXT_EXTENDED_REGISTERS;
  ASSERT_EQ(context.ContextFlags & ~(must_have | may_have), 0u);
#elif defined(ARCH_CPU_X86_64)
  ASSERT_EQ(
      context.ContextFlags,
      static_cast<DWORD>(CONTEXT_AMD64 | CONTEXT_CONTROL | CONTEXT_INTEGER |
                         CONTEXT_SEGMENTS | CONTEXT_FLOATING_POINT));
#endif

#if defined(ARCH_CPU_X86_FAMILY)
  // Many bit positions in the flags register are reserved and will always read
  // a known value. Most reserved bits are always 0, but bit 1 is always 1.
  // Check that the reserved bits are all set to their expected values. Note
  // that the set of reserved bits may be relaxed over time with newer CPUs, and
  // that this test may need to be changed to reflect these developments. The
  // current set of reserved bits are 1, 3, 5, 15, and 22 and higher. See Intel
  // Software Developer’s Manual, Volume 1: Basic Architecture (253665-055),
  // 3.4.3 “EFLAGS Register”, and AMD Architecture Programmer’s Manual, Volume
  // 2: System Programming (24593-3.25), 3.1.6 “RFLAGS Register”.
  EXPECT_EQ(context.EFlags & 0xffc0802a, 2u);

  // CaptureContext() doesn’t capture debug registers, so make sure they read 0.
  EXPECT_EQ(context.Dr0, 0u);
  EXPECT_EQ(context.Dr1, 0u);
  EXPECT_EQ(context.Dr2, 0u);
  EXPECT_EQ(context.Dr3, 0u);
  EXPECT_EQ(context.Dr6, 0u);
  EXPECT_EQ(context.Dr7, 0u);
#endif

#if defined(ARCH_CPU_X86)
  // fxsave doesn’t write these bytes.
  for (size_t i = 464; i < arraysize(context.ExtendedRegisters); ++i) {
    SCOPED_TRACE(i);
    EXPECT_EQ(context.ExtendedRegisters[i], 0);
  }
#elif defined(ARCH_CPU_X86_64)
  // mxcsr shows up twice in the context structure. Make sure the values are
  // identical.
  EXPECT_EQ(context.FltSave.MxCsr, context.MxCsr);

  // fxsave doesn’t write these bytes.
  for (size_t i = 0; i < arraysize(context.FltSave.Reserved4); ++i) {
    SCOPED_TRACE(i);
    EXPECT_EQ(context.FltSave.Reserved4[i], 0);
  }

  // CaptureContext() doesn’t use these fields.
  EXPECT_EQ(context.P1Home, 0u);
  EXPECT_EQ(context.P2Home, 0u);
  EXPECT_EQ(context.P3Home, 0u);
  EXPECT_EQ(context.P4Home, 0u);
  EXPECT_EQ(context.P5Home, 0u);
  EXPECT_EQ(context.P6Home, 0u);
  for (size_t i = 0; i < arraysize(context.VectorRegister); ++i) {
    SCOPED_TRACE(i);
    EXPECT_EQ(context.VectorRegister[i].Low, 0u);
    EXPECT_EQ(context.VectorRegister[i].High, 0u);
  }
  EXPECT_EQ(context.VectorControl, 0u);
  EXPECT_EQ(context.DebugControl, 0u);
  EXPECT_EQ(context.LastBranchToRip, 0u);
  EXPECT_EQ(context.LastBranchFromRip, 0u);
  EXPECT_EQ(context.LastExceptionToRip, 0u);
  EXPECT_EQ(context.LastExceptionFromRip, 0u);
#endif
}

uintptr_t ProgramCounterFromContext(const NativeCPUContext& context) {
#if defined(ARCH_CPU_X86)
  return context.Eip;
#elif defined(ARCH_CPU_X86_64)
  return context.Rip;
#endif
}

uintptr_t StackPointerFromContext(const NativeCPUContext& context) {
#if defined(ARCH_CPU_X86)
  return context.Esp;
#elif defined(ARCH_CPU_X86_64)
  return context.Rsp;
#endif
}

}  // namespace test
}  // namespace crashpad
