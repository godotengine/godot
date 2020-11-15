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

#include <stddef.h>
#include <string.h>

#include "base/logging.h"

namespace crashpad {

#if defined(ARCH_CPU_X86_FAMILY)

namespace {

void InitializeCPUContextX86Thread(
    CPUContextX86* context,
    const x86_thread_state32_t* x86_thread_state32) {
  context->eax = x86_thread_state32->__eax;
  context->ebx = x86_thread_state32->__ebx;
  context->ecx = x86_thread_state32->__ecx;
  context->edx = x86_thread_state32->__edx;
  context->edi = x86_thread_state32->__edi;
  context->esi = x86_thread_state32->__esi;
  context->ebp = x86_thread_state32->__ebp;
  context->esp = x86_thread_state32->__esp;
  context->eip = x86_thread_state32->__eip;
  context->eflags = x86_thread_state32->__eflags;
  context->cs = x86_thread_state32->__cs;
  context->ds = x86_thread_state32->__ds;
  context->es = x86_thread_state32->__es;
  context->fs = x86_thread_state32->__fs;
  context->gs = x86_thread_state32->__gs;
  context->ss = x86_thread_state32->__ss;
}

void InitializeCPUContextX86Float(
    CPUContextX86* context, const x86_float_state32_t* x86_float_state32) {
  // This relies on both x86_float_state32_t and context->fxsave having
  // identical (fxsave) layout.
  static_assert(offsetof(x86_float_state32_t, __fpu_reserved1) -
                         offsetof(x86_float_state32_t, __fpu_fcw) ==
                     sizeof(context->fxsave),
                "types must be equivalent");

  memcpy(
      &context->fxsave, &x86_float_state32->__fpu_fcw, sizeof(context->fxsave));
}

void InitializeCPUContextX86Debug(
    CPUContextX86* context, const x86_debug_state32_t* x86_debug_state32) {
  context->dr0 = x86_debug_state32->__dr0;
  context->dr1 = x86_debug_state32->__dr1;
  context->dr2 = x86_debug_state32->__dr2;
  context->dr3 = x86_debug_state32->__dr3;
  context->dr4 = x86_debug_state32->__dr4;
  context->dr5 = x86_debug_state32->__dr5;
  context->dr6 = x86_debug_state32->__dr6;
  context->dr7 = x86_debug_state32->__dr7;
}

// Initializes |context| from the native thread state structure |state|, which
// is interpreted according to |flavor|. |state_count| must be at least the
// expected size for |flavor|. This handles the architecture-specific
// x86_THREAD_STATE32, x86_FLOAT_STATE32, and x86_DEBUG_STATE32 flavors. It also
// handles the universal x86_THREAD_STATE, x86_FLOAT_STATE, and x86_DEBUG_STATE
// flavors provided that the associated structure carries 32-bit data of the
// corresponding state type. |flavor| may be THREAD_STATE_NONE to avoid setting
// any thread state in |context|. This returns the architecture-specific flavor
// value for the thread state that was actually set, or THREAD_STATE_NONE if no
// thread state was set.
thread_state_flavor_t InitializeCPUContextX86Flavor(
    CPUContextX86* context,
    thread_state_flavor_t flavor,
    ConstThreadState state,
    mach_msg_type_number_t state_count) {
  mach_msg_type_number_t expected_state_count;
  switch (flavor) {
    case x86_THREAD_STATE:
      expected_state_count = x86_THREAD_STATE_COUNT;
      break;
    case x86_FLOAT_STATE:
      expected_state_count = x86_FLOAT_STATE_COUNT;
      break;
    case x86_DEBUG_STATE:
      expected_state_count = x86_DEBUG_STATE_COUNT;
      break;
    case x86_THREAD_STATE32:
      expected_state_count = x86_THREAD_STATE32_COUNT;
      break;
    case x86_FLOAT_STATE32:
      expected_state_count = x86_FLOAT_STATE32_COUNT;
      break;
    case x86_DEBUG_STATE32:
      expected_state_count = x86_DEBUG_STATE32_COUNT;
      break;
    case THREAD_STATE_NONE:
      expected_state_count = 0;
      break;
    default:
      LOG(WARNING) << "unhandled flavor " << flavor;
      return THREAD_STATE_NONE;
  }

  if (state_count < expected_state_count) {
    LOG(WARNING) << "expected state_count " << expected_state_count
                 << " for flavor " << flavor << ", observed " << state_count;
    return THREAD_STATE_NONE;
  }

  switch (flavor) {
    case x86_THREAD_STATE: {
      const x86_thread_state_t* x86_thread_state =
          reinterpret_cast<const x86_thread_state_t*>(state);
      if (x86_thread_state->tsh.flavor != x86_THREAD_STATE32) {
        LOG(WARNING) << "expected flavor x86_THREAD_STATE32, observed "
                     << x86_thread_state->tsh.flavor;
        return THREAD_STATE_NONE;
      }
      return InitializeCPUContextX86Flavor(
          context,
          x86_thread_state->tsh.flavor,
          reinterpret_cast<ConstThreadState>(&x86_thread_state->uts.ts32),
          x86_thread_state->tsh.count);
    }

    case x86_FLOAT_STATE: {
      const x86_float_state_t* x86_float_state =
          reinterpret_cast<const x86_float_state_t*>(state);
      if (x86_float_state->fsh.flavor != x86_FLOAT_STATE32) {
        LOG(WARNING) << "expected flavor x86_FLOAT_STATE32, observed "
                     << x86_float_state->fsh.flavor;
        return THREAD_STATE_NONE;
      }
      return InitializeCPUContextX86Flavor(
          context,
          x86_float_state->fsh.flavor,
          reinterpret_cast<ConstThreadState>(&x86_float_state->ufs.fs32),
          x86_float_state->fsh.count);
    }

    case x86_DEBUG_STATE: {
      const x86_debug_state_t* x86_debug_state =
          reinterpret_cast<const x86_debug_state_t*>(state);
      if (x86_debug_state->dsh.flavor != x86_DEBUG_STATE32) {
        LOG(WARNING) << "expected flavor x86_DEBUG_STATE32, observed "
                     << x86_debug_state->dsh.flavor;
        return THREAD_STATE_NONE;
      }
      return InitializeCPUContextX86Flavor(
          context,
          x86_debug_state->dsh.flavor,
          reinterpret_cast<ConstThreadState>(&x86_debug_state->uds.ds32),
          x86_debug_state->dsh.count);
    }

    case x86_THREAD_STATE32: {
      const x86_thread_state32_t* x86_thread_state32 =
          reinterpret_cast<const x86_thread_state32_t*>(state);
      InitializeCPUContextX86Thread(context, x86_thread_state32);
      return flavor;
    }

    case x86_FLOAT_STATE32: {
      const x86_float_state32_t* x86_float_state32 =
          reinterpret_cast<const x86_float_state32_t*>(state);
      InitializeCPUContextX86Float(context, x86_float_state32);
      return flavor;
    }

    case x86_DEBUG_STATE32: {
      const x86_debug_state32_t* x86_debug_state32 =
          reinterpret_cast<const x86_debug_state32_t*>(state);
      InitializeCPUContextX86Debug(context, x86_debug_state32);
      return flavor;
    }

    case THREAD_STATE_NONE: {
      // This may happen without error when called without exception-style
      // flavor data, or even from an exception handler when the exception
      // behavior is EXCEPTION_DEFAULT.
      return flavor;
    }

    default: {
      NOTREACHED();
      return THREAD_STATE_NONE;
    }
  }
}

void InitializeCPUContextX86_64Thread(
    CPUContextX86_64* context, const x86_thread_state64_t* x86_thread_state64) {
  context->rax = x86_thread_state64->__rax;
  context->rbx = x86_thread_state64->__rbx;
  context->rcx = x86_thread_state64->__rcx;
  context->rdx = x86_thread_state64->__rdx;
  context->rdi = x86_thread_state64->__rdi;
  context->rsi = x86_thread_state64->__rsi;
  context->rbp = x86_thread_state64->__rbp;
  context->rsp = x86_thread_state64->__rsp;
  context->r8 = x86_thread_state64->__r8;
  context->r9 = x86_thread_state64->__r9;
  context->r10 = x86_thread_state64->__r10;
  context->r11 = x86_thread_state64->__r11;
  context->r12 = x86_thread_state64->__r12;
  context->r13 = x86_thread_state64->__r13;
  context->r14 = x86_thread_state64->__r14;
  context->r15 = x86_thread_state64->__r15;
  context->rip = x86_thread_state64->__rip;
  context->rflags = x86_thread_state64->__rflags;
  context->cs = x86_thread_state64->__cs;
  context->fs = x86_thread_state64->__fs;
  context->gs = x86_thread_state64->__gs;
}

void InitializeCPUContextX86_64Float(
    CPUContextX86_64* context, const x86_float_state64_t* x86_float_state64) {
  // This relies on both x86_float_state64_t and context->fxsave having
  // identical (fxsave) layout.
  static_assert(offsetof(x86_float_state64_t, __fpu_reserved1) -
                         offsetof(x86_float_state64_t, __fpu_fcw) ==
                     sizeof(context->fxsave),
                "types must be equivalent");

  memcpy(&context->fxsave,
         &x86_float_state64->__fpu_fcw,
         sizeof(context->fxsave));
}

void InitializeCPUContextX86_64Debug(
    CPUContextX86_64* context, const x86_debug_state64_t* x86_debug_state64) {
  context->dr0 = x86_debug_state64->__dr0;
  context->dr1 = x86_debug_state64->__dr1;
  context->dr2 = x86_debug_state64->__dr2;
  context->dr3 = x86_debug_state64->__dr3;
  context->dr4 = x86_debug_state64->__dr4;
  context->dr5 = x86_debug_state64->__dr5;
  context->dr6 = x86_debug_state64->__dr6;
  context->dr7 = x86_debug_state64->__dr7;
}

// Initializes |context| from the native thread state structure |state|, which
// is interpreted according to |flavor|. |state_count| must be at least the
// expected size for |flavor|. This handles the architecture-specific
// x86_THREAD_STATE64, x86_FLOAT_STATE64, and x86_DEBUG_STATE64 flavors. It also
// handles the universal x86_THREAD_STATE, x86_FLOAT_STATE, and x86_DEBUG_STATE
// flavors provided that the associated structure carries 64-bit data of the
// corresponding state type. |flavor| may be THREAD_STATE_NONE to avoid setting
// any thread state in |context|. This returns the architecture-specific flavor
// value for the thread state that was actually set, or THREAD_STATE_NONE if no
// thread state was set.
thread_state_flavor_t InitializeCPUContextX86_64Flavor(
    CPUContextX86_64* context,
    thread_state_flavor_t flavor,
    ConstThreadState state,
    mach_msg_type_number_t state_count) {
  mach_msg_type_number_t expected_state_count;
  switch (flavor) {
    case x86_THREAD_STATE:
      expected_state_count = x86_THREAD_STATE_COUNT;
      break;
    case x86_FLOAT_STATE:
      expected_state_count = x86_FLOAT_STATE_COUNT;
      break;
    case x86_DEBUG_STATE:
      expected_state_count = x86_DEBUG_STATE_COUNT;
      break;
    case x86_THREAD_STATE64:
      expected_state_count = x86_THREAD_STATE64_COUNT;
      break;
    case x86_FLOAT_STATE64:
      expected_state_count = x86_FLOAT_STATE64_COUNT;
      break;
    case x86_DEBUG_STATE64:
      expected_state_count = x86_DEBUG_STATE64_COUNT;
      break;
    case THREAD_STATE_NONE:
      expected_state_count = 0;
      break;
    default:
      LOG(WARNING) << "unhandled flavor " << flavor;
      return THREAD_STATE_NONE;
  }

  if (state_count < expected_state_count) {
    LOG(WARNING) << "expected state_count " << expected_state_count
                 << " for flavor " << flavor << ", observed " << state_count;
    return THREAD_STATE_NONE;
  }

  switch (flavor) {
    case x86_THREAD_STATE: {
      const x86_thread_state_t* x86_thread_state =
          reinterpret_cast<const x86_thread_state_t*>(state);
      if (x86_thread_state->tsh.flavor != x86_THREAD_STATE64) {
        LOG(WARNING) << "expected flavor x86_THREAD_STATE64, observed "
                     << x86_thread_state->tsh.flavor;
        return THREAD_STATE_NONE;
      }
      return InitializeCPUContextX86_64Flavor(
          context,
          x86_thread_state->tsh.flavor,
          reinterpret_cast<ConstThreadState>(&x86_thread_state->uts.ts64),
          x86_thread_state->tsh.count);
    }

    case x86_FLOAT_STATE: {
      const x86_float_state_t* x86_float_state =
          reinterpret_cast<const x86_float_state_t*>(state);
      if (x86_float_state->fsh.flavor != x86_FLOAT_STATE64) {
        LOG(WARNING) << "expected flavor x86_FLOAT_STATE64, observed "
                     << x86_float_state->fsh.flavor;
        return THREAD_STATE_NONE;
      }
      return InitializeCPUContextX86_64Flavor(
          context,
          x86_float_state->fsh.flavor,
          reinterpret_cast<ConstThreadState>(&x86_float_state->ufs.fs64),
          x86_float_state->fsh.count);
    }

    case x86_DEBUG_STATE: {
      const x86_debug_state_t* x86_debug_state =
          reinterpret_cast<const x86_debug_state_t*>(state);
      if (x86_debug_state->dsh.flavor != x86_DEBUG_STATE64) {
        LOG(WARNING) << "expected flavor x86_DEBUG_STATE64, observed "
                     << x86_debug_state->dsh.flavor;
        return THREAD_STATE_NONE;
      }
      return InitializeCPUContextX86_64Flavor(
          context,
          x86_debug_state->dsh.flavor,
          reinterpret_cast<ConstThreadState>(&x86_debug_state->uds.ds64),
          x86_debug_state->dsh.count);
    }

    case x86_THREAD_STATE64: {
      const x86_thread_state64_t* x86_thread_state64 =
          reinterpret_cast<const x86_thread_state64_t*>(state);
      InitializeCPUContextX86_64Thread(context, x86_thread_state64);
      return flavor;
    }

    case x86_FLOAT_STATE64: {
      const x86_float_state64_t* x86_float_state64 =
          reinterpret_cast<const x86_float_state64_t*>(state);
      InitializeCPUContextX86_64Float(context, x86_float_state64);
      return flavor;
    }

    case x86_DEBUG_STATE64: {
      const x86_debug_state64_t* x86_debug_state64 =
          reinterpret_cast<const x86_debug_state64_t*>(state);
      InitializeCPUContextX86_64Debug(context, x86_debug_state64);
      return flavor;
    }

    case THREAD_STATE_NONE: {
      // This may happen without error when called without exception-style
      // flavor data, or even from an exception handler when the exception
      // behavior is EXCEPTION_DEFAULT.
      return flavor;
    }

    default: {
      NOTREACHED();
      return THREAD_STATE_NONE;
    }
  }
}

}  // namespace

namespace internal {

void InitializeCPUContextX86(CPUContextX86* context,
                             thread_state_flavor_t flavor,
                             ConstThreadState state,
                             mach_msg_type_number_t state_count,
                             const x86_thread_state32_t* x86_thread_state32,
                             const x86_float_state32_t* x86_float_state32,
                             const x86_debug_state32_t* x86_debug_state32) {
  thread_state_flavor_t set_flavor = THREAD_STATE_NONE;
  if (flavor != THREAD_STATE_NONE) {
    set_flavor =
        InitializeCPUContextX86Flavor(context, flavor, state, state_count);
  }

  if (set_flavor != x86_THREAD_STATE32) {
    InitializeCPUContextX86Thread(context, x86_thread_state32);
  }
  if (set_flavor != x86_FLOAT_STATE32) {
    InitializeCPUContextX86Float(context, x86_float_state32);
  }
  if (set_flavor != x86_DEBUG_STATE32) {
    InitializeCPUContextX86Debug(context, x86_debug_state32);
  }
}

void InitializeCPUContextX86_64(CPUContextX86_64* context,
                                thread_state_flavor_t flavor,
                                ConstThreadState state,
                                mach_msg_type_number_t state_count,
                                const x86_thread_state64_t* x86_thread_state64,
                                const x86_float_state64_t* x86_float_state64,
                                const x86_debug_state64_t* x86_debug_state64) {
  thread_state_flavor_t set_flavor = THREAD_STATE_NONE;
  if (flavor != THREAD_STATE_NONE) {
    set_flavor =
        InitializeCPUContextX86_64Flavor(context, flavor, state, state_count);
  }

  if (set_flavor != x86_THREAD_STATE64) {
    InitializeCPUContextX86_64Thread(context, x86_thread_state64);
  }
  if (set_flavor != x86_FLOAT_STATE64) {
    InitializeCPUContextX86_64Float(context, x86_float_state64);
  }
  if (set_flavor != x86_DEBUG_STATE64) {
    InitializeCPUContextX86_64Debug(context, x86_debug_state64);
  }
}

}  // namespace internal

#endif

}  // namespace crashpad
