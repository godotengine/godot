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

#include "snapshot/fuchsia/cpu_context_fuchsia.h"

#include <string.h>

namespace crashpad {
namespace internal {

#if defined(ARCH_CPU_X86_64)

void InitializeCPUContextX86_64(
    const zx_thread_state_general_regs_t& thread_context,
    CPUContextX86_64* context) {
  memset(context, 0, sizeof(*context));
  context->rax = thread_context.rax;
  context->rbx = thread_context.rbx;
  context->rcx = thread_context.rcx;
  context->rdx = thread_context.rdx;
  context->rdi = thread_context.rdi;
  context->rsi = thread_context.rsi;
  context->rbp = thread_context.rbp;
  context->rsp = thread_context.rsp;
  context->r8 = thread_context.r8;
  context->r9 = thread_context.r9;
  context->r10 = thread_context.r10;
  context->r11 = thread_context.r11;
  context->r12 = thread_context.r12;
  context->r13 = thread_context.r13;
  context->r14 = thread_context.r14;
  context->r15 = thread_context.r15;
  context->rip = thread_context.rip;
  context->rflags = thread_context.rflags;
}

#endif  // ARCH_CPU_X86_64

}  // namespace internal
}  // namespace crashpad
