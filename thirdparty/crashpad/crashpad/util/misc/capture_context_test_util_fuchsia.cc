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

#include "base/logging.h"
#include "gtest/gtest.h"
#include "util/misc/from_pointer_cast.h"

namespace crashpad {
namespace test {

#if defined(ARCH_CPU_X86_64)
static_assert(offsetof(NativeCPUContext, uc_mcontext) == 0x28,
              "unexpected mcontext offset");
static_assert(offsetof(NativeCPUContext, uc_mcontext.gregs[REG_RSP]) == 0xa0,
              "unexpected rsp offset");
static_assert(offsetof(NativeCPUContext, uc_mcontext.gregs[REG_RIP]) == 0xa8,
              "unexpected rip offset");
#endif  // ARCH_CPU_X86_64

void SanityCheckContext(const NativeCPUContext& context) {
#if defined(ARCH_CPU_X86_64)
  EXPECT_EQ(context.uc_mcontext.gregs[REG_RDI],
            FromPointerCast<intptr_t>(&context));
#elif defined(ARCH_CPU_ARM64)
  EXPECT_EQ(context.uc_mcontext.regs[0], FromPointerCast<uintptr_t>(&context));
#endif
}

uintptr_t ProgramCounterFromContext(const NativeCPUContext& context) {
#if defined(ARCH_CPU_X86_64)
  return context.uc_mcontext.gregs[REG_RIP];
#elif defined(ARCH_CPU_ARM64)
  return context.uc_mcontext.pc;
#endif
}

uintptr_t StackPointerFromContext(const NativeCPUContext& context) {
#if defined(ARCH_CPU_X86_64)
  return context.uc_mcontext.gregs[REG_RSP];
#elif defined(ARCH_CPU_ARM64)
  return context.uc_mcontext.sp;
#endif
}

}  // namespace test
}  // namespace crashpad
