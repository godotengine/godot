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

#include "gtest/gtest.h"
#include "util/misc/implicit_cast.h"

namespace crashpad {
namespace test {

void SanityCheckContext(const NativeCPUContext& context) {
#if defined(ARCH_CPU_X86)
  ASSERT_EQ(implicit_cast<thread_state_flavor_t>(context.tsh.flavor),
            implicit_cast<thread_state_flavor_t>(x86_THREAD_STATE32));
  ASSERT_EQ(implicit_cast<uint32_t>(context.tsh.count),
            implicit_cast<uint32_t>(x86_THREAD_STATE32_COUNT));
#elif defined(ARCH_CPU_X86_64)
  ASSERT_EQ(implicit_cast<thread_state_flavor_t>(context.tsh.flavor),
            implicit_cast<thread_state_flavor_t>(x86_THREAD_STATE64));
  ASSERT_EQ(implicit_cast<uint32_t>(context.tsh.count),
            implicit_cast<uint32_t>(x86_THREAD_STATE64_COUNT));
#endif

#if defined(ARCH_CPU_X86_FAMILY)
// The segment registers are only capable of storing 16-bit quantities, but
// the context structure provides native integer-width fields for them. Ensure
// that the high bits are all clear.
//
// Many bit positions in the flags register are reserved and will always read
// a known value. Most reserved bits are always 0, but bit 1 is always 1.
// Check that the reserved bits are all set to their expected values. Note
// that the set of reserved bits may be relaxed over time with newer CPUs, and
// that this test may need to be changed to reflect these developments. The
// current set of reserved bits are 1, 3, 5, 15, and 22 and higher. See Intel
// Software Developer’s Manual, Volume 1: Basic Architecture (253665-051),
// 3.4.3 “EFLAGS Register”, and AMD Architecture Programmer’s Manual, Volume
// 2: System Programming (24593-3.24), 3.1.6 “RFLAGS Register”.
#if defined(ARCH_CPU_X86)
  EXPECT_EQ(context.uts.ts32.__cs & ~0xffff, 0u);
  EXPECT_EQ(context.uts.ts32.__ds & ~0xffff, 0u);
  EXPECT_EQ(context.uts.ts32.__es & ~0xffff, 0u);
  EXPECT_EQ(context.uts.ts32.__fs & ~0xffff, 0u);
  EXPECT_EQ(context.uts.ts32.__gs & ~0xffff, 0u);
  EXPECT_EQ(context.uts.ts32.__ss & ~0xffff, 0u);
  EXPECT_EQ(context.uts.ts32.__eflags & 0xffc0802a, 2u);
#elif defined(ARCH_CPU_X86_64)
  EXPECT_EQ(context.uts.ts64.__cs & ~UINT64_C(0xffff), 0u);
  EXPECT_EQ(context.uts.ts64.__fs & ~UINT64_C(0xffff), 0u);
  EXPECT_EQ(context.uts.ts64.__gs & ~UINT64_C(0xffff), 0u);
  EXPECT_EQ(context.uts.ts64.__rflags & UINT64_C(0xffffffffffc0802a), 2u);
#endif
#endif
}

uintptr_t ProgramCounterFromContext(const NativeCPUContext& context) {
#if defined(ARCH_CPU_X86)
  return context.uts.ts32.__eip;
#elif defined(ARCH_CPU_X86_64)
  return context.uts.ts64.__rip;
#endif
}

uintptr_t StackPointerFromContext(const NativeCPUContext& context) {
#if defined(ARCH_CPU_X86)
  return context.uts.ts32.__esp;
#elif defined(ARCH_CPU_X86_64)
  return context.uts.ts64.__rsp;
#endif
}

}  // namespace test
}  // namespace crashpad
