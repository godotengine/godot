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

#include "snapshot/win/cpu_context_win.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "base/logging.h"
#include "snapshot/cpu_context.h"

namespace crashpad {

namespace {

// Validation for casts used with CPUContextX86::FsaveToFxsave().
static_assert(sizeof(CPUContextX86::Fsave) ==
                  offsetof(WOW64_FLOATING_SAVE_AREA, Cr0NpxState),
              "WoW64 fsave types must be equivalent");
#if defined(ARCH_CPU_X86)
static_assert(sizeof(CPUContextX86::Fsave) ==
                  offsetof(FLOATING_SAVE_AREA, Spare0),
              "fsave types must be equivalent");
#endif  // ARCH_CPU_X86

template <typename T>
bool HasContextPart(const T& context, uint32_t bits) {
  return (context.ContextFlags & bits) == bits;
}

template <class T>
void CommonInitializeX86Context(const T& context, CPUContextX86* out) {
  // This function assumes that the WOW64_CONTEXT_* and x86 CONTEXT_* values
  // for ContextFlags are identical. This can be tested when targeting 32-bit
  // x86.
#if defined(ARCH_CPU_X86)
  static_assert(sizeof(CONTEXT) == sizeof(WOW64_CONTEXT),
                "type mismatch: CONTEXT");
#define ASSERT_WOW64_EQUIVALENT(x)                        \
  do {                                                    \
    static_assert(x == WOW64_##x, "value mismatch: " #x); \
  } while (false)
  ASSERT_WOW64_EQUIVALENT(CONTEXT_i386);
  ASSERT_WOW64_EQUIVALENT(CONTEXT_i486);
  ASSERT_WOW64_EQUIVALENT(CONTEXT_CONTROL);
  ASSERT_WOW64_EQUIVALENT(CONTEXT_INTEGER);
  ASSERT_WOW64_EQUIVALENT(CONTEXT_SEGMENTS);
  ASSERT_WOW64_EQUIVALENT(CONTEXT_FLOATING_POINT);
  ASSERT_WOW64_EQUIVALENT(CONTEXT_DEBUG_REGISTERS);
  ASSERT_WOW64_EQUIVALENT(CONTEXT_EXTENDED_REGISTERS);
  ASSERT_WOW64_EQUIVALENT(CONTEXT_FULL);
  ASSERT_WOW64_EQUIVALENT(CONTEXT_ALL);
  ASSERT_WOW64_EQUIVALENT(CONTEXT_XSTATE);
#undef ASSERT_WOW64_EQUIVALENT
#endif  // ARCH_CPU_X86

  memset(out, 0, sizeof(*out));

  LOG_IF(ERROR, !HasContextPart(context, WOW64_CONTEXT_i386))
      << "non-x86 context";

  if (HasContextPart(context, WOW64_CONTEXT_CONTROL)) {
    out->ebp = context.Ebp;
    out->eip = context.Eip;
    out->cs = static_cast<uint16_t>(context.SegCs);
    out->eflags = context.EFlags;
    out->esp = context.Esp;
    out->ss = static_cast<uint16_t>(context.SegSs);
  }

  if (HasContextPart(context, WOW64_CONTEXT_INTEGER)) {
    out->eax = context.Eax;
    out->ebx = context.Ebx;
    out->ecx = context.Ecx;
    out->edx = context.Edx;
    out->edi = context.Edi;
    out->esi = context.Esi;
  }

  if (HasContextPart(context, WOW64_CONTEXT_SEGMENTS)) {
    out->ds = static_cast<uint16_t>(context.SegDs);
    out->es = static_cast<uint16_t>(context.SegEs);
    out->fs = static_cast<uint16_t>(context.SegFs);
    out->gs = static_cast<uint16_t>(context.SegGs);
  }

  if (HasContextPart(context, WOW64_CONTEXT_DEBUG_REGISTERS)) {
    out->dr0 = context.Dr0;
    out->dr1 = context.Dr1;
    out->dr2 = context.Dr2;
    out->dr3 = context.Dr3;

    // DR4 and DR5 are obsolete synonyms for DR6 and DR7, see
    // https://en.wikipedia.org/wiki/X86_debug_register.
    out->dr4 = context.Dr6;
    out->dr5 = context.Dr7;

    out->dr6 = context.Dr6;
    out->dr7 = context.Dr7;
  }

  if (HasContextPart(context, WOW64_CONTEXT_EXTENDED_REGISTERS)) {
    static_assert(sizeof(out->fxsave) == sizeof(context.ExtendedRegisters),
                  "fxsave types must be equivalent");
    memcpy(&out->fxsave, &context.ExtendedRegisters, sizeof(out->fxsave));
  } else if (HasContextPart(context, WOW64_CONTEXT_FLOATING_POINT)) {
    // The static_assert that validates this cast can’t be here because it
    // relies on field names that vary based on the template parameter.
    CPUContextX86::FsaveToFxsave(
        *reinterpret_cast<const CPUContextX86::Fsave*>(&context.FloatSave),
        &out->fxsave);
  }
}

}  // namespace

#if defined(ARCH_CPU_64_BITS)

void InitializeX86Context(const WOW64_CONTEXT& context, CPUContextX86* out) {
  CommonInitializeX86Context(context, out);
}

void InitializeX64Context(const CONTEXT& context, CPUContextX86_64* out) {
  memset(out, 0, sizeof(*out));

  LOG_IF(ERROR, !HasContextPart(context, CONTEXT_AMD64)) << "non-x64 context";

  if (HasContextPart(context, CONTEXT_CONTROL)) {
    out->cs = context.SegCs;
    out->rflags = context.EFlags;
    out->rip = context.Rip;
    out->rsp = context.Rsp;
    // SegSs ignored.
  }

  if (HasContextPart(context, CONTEXT_INTEGER)) {
    out->rax = context.Rax;
    out->rbx = context.Rbx;
    out->rcx = context.Rcx;
    out->rdx = context.Rdx;
    out->rdi = context.Rdi;
    out->rsi = context.Rsi;
    out->rbp = context.Rbp;
    out->r8 = context.R8;
    out->r9 = context.R9;
    out->r10 = context.R10;
    out->r11 = context.R11;
    out->r12 = context.R12;
    out->r13 = context.R13;
    out->r14 = context.R14;
    out->r15 = context.R15;
  }

  if (HasContextPart(context, CONTEXT_SEGMENTS)) {
    out->fs = context.SegFs;
    out->gs = context.SegGs;
    // SegDs ignored.
    // SegEs ignored.
  }

  if (HasContextPart(context, CONTEXT_DEBUG_REGISTERS)) {
    out->dr0 = context.Dr0;
    out->dr1 = context.Dr1;
    out->dr2 = context.Dr2;
    out->dr3 = context.Dr3;

    // DR4 and DR5 are obsolete synonyms for DR6 and DR7, see
    // https://en.wikipedia.org/wiki/X86_debug_register.
    out->dr4 = context.Dr6;
    out->dr5 = context.Dr7;

    out->dr6 = context.Dr6;
    out->dr7 = context.Dr7;
  }

  if (HasContextPart(context, CONTEXT_FLOATING_POINT)) {
    static_assert(sizeof(out->fxsave) == sizeof(context.FltSave),
                  "types must be equivalent");
    memcpy(&out->fxsave, &context.FltSave, sizeof(out->fxsave));
  }
}

#else  // ARCH_CPU_64_BITS

void InitializeX86Context(const CONTEXT& context, CPUContextX86* out) {
  CommonInitializeX86Context(context, out);
}

#endif  // ARCH_CPU_64_BITS

}  // namespace crashpad
