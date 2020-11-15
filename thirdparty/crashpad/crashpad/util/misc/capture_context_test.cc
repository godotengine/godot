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

#include "util/misc/capture_context.h"

#include <stdint.h>

#include <algorithm>

#include "gtest/gtest.h"
#include "util/misc/address_sanitizer.h"
#include "util/misc/capture_context_test_util.h"

namespace crashpad {
namespace test {
namespace {

#if defined(OS_FUCHSIA)
// Fuchsia uses -fsanitize=safe-stack by default, which splits local variables
// and the call stack into separate regions (see
// https://clang.llvm.org/docs/SafeStack.html). Because this test would like to
// find an approximately valid stack pointer by comparing locals to the
// captured one, disable safe-stack for this function.
__attribute__((no_sanitize("safe-stack")))
#endif

void TestCaptureContext() {
  NativeCPUContext context_1;
  CaptureContext(&context_1);

  {
    SCOPED_TRACE("context_1");
    ASSERT_NO_FATAL_FAILURE(SanityCheckContext(context_1));
  }

  // The program counter reference value is this function’s address. The
  // captured program counter should be slightly greater than or equal to the
  // reference program counter.
  uintptr_t pc = ProgramCounterFromContext(context_1);

#if !defined(ADDRESS_SANITIZER) && !defined(ARCH_CPU_MIPS_FAMILY)
  // AddressSanitizer can cause enough code bloat that the “nearby” check would
  // likely fail.
  const uintptr_t kReferencePC =
      reinterpret_cast<uintptr_t>(TestCaptureContext);
  EXPECT_PRED2([](uintptr_t actual,
                  uintptr_t reference) { return actual - reference < 64u; },
               pc,
               kReferencePC);
#endif  // !defined(ADDRESS_SANITIZER)

  const uintptr_t sp = StackPointerFromContext(context_1);

  // Declare context_2 here because all local variables need to be declared
  // before computing the stack pointer reference value, so that the reference
  // value can be the lowest value possible.
  NativeCPUContext context_2;

// AddressSanitizer on Linux causes stack variables to be stored separately from
// the call stack.
#if !defined(ADDRESS_SANITIZER) || (!defined(OS_LINUX) && !defined(OS_ANDROID))
  // The stack pointer reference value is the lowest address of a local variable
  // in this function. The captured program counter will be slightly less than
  // or equal to the reference stack pointer.
  const uintptr_t kReferenceSP =
      std::min(std::min(reinterpret_cast<uintptr_t>(&context_1),
                        reinterpret_cast<uintptr_t>(&context_2)),
               std::min(reinterpret_cast<uintptr_t>(&pc),
                        reinterpret_cast<uintptr_t>(&sp)));
  EXPECT_PRED2([](uintptr_t actual,
                  uintptr_t reference) { return reference - actual < 768u; },
               sp,
               kReferenceSP);
#endif  // !ADDRESS_SANITIZER || (!OS_LINUX && !OS_ANDROID)

  // Capture the context again, expecting that the stack pointer stays the same
  // and the program counter increases. Strictly speaking, there’s no guarantee
  // that these conditions will hold, although they do for known compilers even
  // under typical optimization.
  CaptureContext(&context_2);

  {
    SCOPED_TRACE("context_2");
    ASSERT_NO_FATAL_FAILURE(SanityCheckContext(context_2));
  }

  EXPECT_EQ(StackPointerFromContext(context_2), sp);
  EXPECT_GT(ProgramCounterFromContext(context_2), pc);
}

TEST(CaptureContext, CaptureContext) {
  ASSERT_NO_FATAL_FAILURE(TestCaptureContext());
}

}  // namespace
}  // namespace test
}  // namespace crashpad
