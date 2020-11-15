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

#ifndef CRASHPAD_SNAPSHOT_LINUX_SNAPSHOT_SIGNAL_CONTEXT_H_
#define CRASHPAD_SNAPSHOT_LINUX_SNAPSHOT_SIGNAL_CONTEXT_H_

#include <signal.h>
#include <stdint.h>
#include <sys/types.h>
#include <sys/ucontext.h>

#include <cstddef>
#include <type_traits>

#include "build/build_config.h"
#include "util/linux/thread_info.h"
#include "util/linux/traits.h"

namespace crashpad {
namespace internal {

#pragma pack(push, 1)

template <class Traits>
union Sigval {
  int32_t sigval;
  typename Traits::Address pointer;
};

template <class Traits>
struct Siginfo {
  int32_t signo;
#ifdef ARCH_CPU_MIPS_FAMILY
  // Attribute order for signo_t defined in kernel is different for MIPS.
  int32_t code;
  int32_t err;
#else
  int32_t err;
  int32_t code;
#endif
  typename Traits::UInteger32_64Only padding;

  union {
    // SIGSEGV, SIGILL, SIGFPE, SIGBUS, SIGTRAP
    struct {
      typename Traits::Address address;
    };

    // SIGPOLL
    struct {
      typename Traits::Long band;
      int32_t fd;
    };

    // SIGSYS
    struct {
      typename Traits::Address call_address;
      int32_t syscall;
      uint32_t arch;
    };

    // Everything else
    struct {
      union {
        struct {
          pid_t pid;
          uid_t uid;
        };
        struct {
          int32_t timerid;
          int32_t overrun;
        };
      };

      union {
        Sigval<Traits> sigval;

        // SIGCHLD
        struct {
          int32_t status;
          typename Traits::Clock utime;
          typename Traits::Clock stime;
        };
      };
    };
  };
};

template <typename Traits>
struct SignalStack {
  typename Traits::Address stack_pointer;
  uint32_t flags;
  typename Traits::UInteger32_64Only padding;
  typename Traits::Size size;
};

template <typename Traits, typename Enable = void>
struct Sigset {};

template <typename Traits>
struct Sigset<
    Traits,
    typename std::enable_if<std::is_base_of<Traits32, Traits>::value>::type> {
  uint64_t val;
};

template <typename Traits>
struct Sigset<
    Traits,
    typename std::enable_if<std::is_base_of<Traits64, Traits>::value>::type> {
#if defined(OS_ANDROID)
  uint64_t val;
#else
  typename Traits::ULong val[16];
#endif  // OS_ANDROID
};

#if defined(ARCH_CPU_X86_FAMILY)

struct SignalThreadContext32 {
  uint32_t xgs;
  uint32_t xfs;
  uint32_t xes;
  uint32_t xds;
  uint32_t edi;
  uint32_t esi;
  uint32_t ebp;
  uint32_t esp;
  uint32_t ebx;
  uint32_t edx;
  uint32_t ecx;
  uint32_t eax;
  uint32_t trapno;
  uint32_t err;
  uint32_t eip;
  uint32_t xcs;
  uint32_t eflags;
  uint32_t uesp;
  uint32_t xss;
};

struct SignalThreadContext64 {
  uint64_t r8;
  uint64_t r9;
  uint64_t r10;
  uint64_t r11;
  uint64_t r12;
  uint64_t r13;
  uint64_t r14;
  uint64_t r15;
  uint64_t rdi;
  uint64_t rsi;
  uint64_t rbp;
  uint64_t rbx;
  uint64_t rdx;
  uint64_t rax;
  uint64_t rcx;
  uint64_t rsp;
  uint64_t rip;
  uint64_t eflags;
  uint16_t cs;
  uint16_t gs;
  uint16_t fs;
  uint16_t padding;
  uint64_t err;
  uint64_t trapno;
  uint64_t oldmask;
  uint64_t cr2;
};

struct SignalFloatContext32 {
  CPUContextX86::Fsave fsave;
  uint16_t status;
  uint16_t magic;
  CPUContextX86::Fxsave fxsave[0];
};

using SignalFloatContext64 = CPUContextX86_64::Fxsave;

struct ContextTraits32 : public Traits32 {
  using ThreadContext = SignalThreadContext32;
  using FloatContext = SignalFloatContext32;
};

struct ContextTraits64 : public Traits64 {
  using ThreadContext = SignalThreadContext64;
  using FloatContext = SignalFloatContext64;
};

template <typename Traits>
struct MContext {
  typename Traits::ThreadContext gprs;
  typename Traits::Address fpptr;
  typename Traits::ULong_32Only oldmask;
  typename Traits::ULong_32Only cr2;
  typename Traits::ULong_64Only reserved[8];
};

template <typename Traits>
struct UContext {
  typename Traits::ULong flags;
  typename Traits::Address link;
  SignalStack<Traits> stack;
  MContext<Traits> mcontext;
  Sigset<Traits> sigmask;
  char fpregs_mem[0];
};

#elif defined(ARCH_CPU_ARM_FAMILY)

struct CoprocessorContextHead {
  uint32_t magic;
  uint32_t size;
};

struct SignalFPSIMDContext {
  uint32_t fpsr;
  uint32_t fpcr;
  uint128_struct vregs[32];
};

struct SignalVFPContext {
  FloatContext::f32_t::vfp_t vfp;
  struct vfp_exc {
    uint32_t fpexc;
    uint32_t fpinst;
    uint32_t fpinst2;
  } vfp_exc;
  uint32_t padding;
};

struct SignalThreadContext32 {
  uint32_t regs[11];
  uint32_t fp;
  uint32_t ip;
  uint32_t sp;
  uint32_t lr;
  uint32_t pc;
  uint32_t cpsr;
};

using SignalThreadContext64 = ThreadContext::t64_t;

struct MContext32Data {
  uint32_t trap_no;
  uint32_t error_code;
  uint32_t oldmask;
  SignalThreadContext32 gprs;
  uint32_t fault_address;
};

struct MContext64Data {
  uint64_t fault_address;
  SignalThreadContext64 gprs;
};

struct ContextTraits32 : public Traits32 {
  using MContext32 = MContext32Data;
  using MContext64 = Nothing;
};

struct ContextTraits64 : public Traits64 {
  using MContext32 = Nothing;
  using MContext64 = MContext64Data;
};

template <typename Traits>
struct UContext {
  typename Traits::ULong flags;
  typename Traits::Address link;
  SignalStack<Traits> stack;
  typename Traits::MContext32 mcontext32;
  Sigset<Traits> sigmask;
  char padding[128 - sizeof(sigmask)];
  typename Traits::Char_64Only padding2[8];
  typename Traits::MContext64 mcontext64;
  typename Traits::Char_64Only padding3[8];
  char reserved[0];
};

#if defined(ARCH_CPU_ARMEL)
static_assert(offsetof(UContext<ContextTraits32>, mcontext32) ==
                  offsetof(ucontext_t, uc_mcontext),
              "context offset mismatch");
static_assert(offsetof(UContext<ContextTraits32>, reserved) ==
                  offsetof(ucontext_t, uc_regspace),
              "regspace offset mismatch");

#elif defined(ARCH_CPU_ARM64)
static_assert(offsetof(UContext<ContextTraits64>, mcontext64) ==
                  offsetof(ucontext_t, uc_mcontext),
              "context offset mismtach");
static_assert(offsetof(UContext<ContextTraits64>, reserved) ==
                  offsetof(ucontext_t, uc_mcontext) +
                      offsetof(mcontext_t, __reserved),
              "reserved space offset mismtach");
#endif

#elif defined(ARCH_CPU_MIPS_FAMILY)

struct MContext32 {
  uint32_t regmask;
  uint32_t status;
  uint64_t pc;
  uint64_t gregs[32];
  struct {
    float _fp_fregs;
    unsigned int _fp_pad;
  } fpregs[32];
  uint32_t fp_owned;
  uint32_t fpc_csr;
  uint32_t fpc_eir;
  uint32_t used_math;
  uint32_t dsp;
  uint64_t mdhi;
  uint64_t mdlo;
  uint32_t hi1;
  uint32_t lo1;
  uint32_t hi2;
  uint32_t lo2;
  uint32_t hi3;
  uint32_t lo3;
};

struct MContext64 {
  uint64_t gregs[32];
  double fpregs[32];
  uint64_t mdhi;
  uint64_t hi1;
  uint64_t hi2;
  uint64_t hi3;
  uint64_t mdlo;
  uint64_t lo1;
  uint64_t lo2;
  uint64_t lo3;
  uint64_t pc;
  uint32_t fpc_csr;
  uint32_t used_math;
  uint32_t dsp;
  uint32_t __glibc_reserved1;
};

struct SignalThreadContext32 {
  uint64_t regs[32];
  uint32_t lo;
  uint32_t hi;
  uint32_t cp0_epc;
  uint32_t cp0_badvaddr;
  uint32_t cp0_status;
  uint32_t cp0_cause;

  SignalThreadContext32() {}
  explicit SignalThreadContext32(
      const struct ThreadContext::t32_t& thread_context) {
    for (size_t reg = 0; reg < 32; ++reg) {
      regs[reg] = thread_context.regs[reg];
    }
    lo = thread_context.lo;
    hi = thread_context.hi;
    cp0_epc = thread_context.cp0_epc;
    cp0_badvaddr = thread_context.cp0_badvaddr;
    cp0_status = thread_context.cp0_status;
    cp0_cause = thread_context.cp0_cause;
  }
};

struct ContextTraits32 : public Traits32 {
  using MContext = MContext32;
  using SignalThreadContext = SignalThreadContext32;
  using SignalFloatContext = FloatContext::f32_t;
  using CPUContext = CPUContextMIPS;
};

struct ContextTraits64 : public Traits64 {
  using MContext = MContext64;
  using SignalThreadContext = ThreadContext::t64_t;
  using SignalFloatContext = FloatContext::f64_t;
  using CPUContext = CPUContextMIPS64;
};

template <typename Traits>
struct UContext {
  typename Traits::ULong flags;
  typename Traits::Address link;
  SignalStack<Traits> stack;
  typename Traits::ULong_32Only alignment_padding_;
  typename Traits::MContext mcontext;
  Sigset<Traits> sigmask;
};

#if defined(ARCH_CPU_MIPSEL)
static_assert(offsetof(UContext<ContextTraits32>, mcontext) ==
                  offsetof(ucontext_t, uc_mcontext),
              "context offset mismatch");
static_assert(offsetof(UContext<ContextTraits32>, mcontext.gregs) ==
                  offsetof(ucontext_t, uc_mcontext.gregs),
              "context offset mismatch");
static_assert(offsetof(UContext<ContextTraits32>, mcontext.fpregs) ==
                  offsetof(ucontext_t, uc_mcontext.fpregs),
              "context offset mismatch");

#elif defined(ARCH_CPU_MIPS64EL)
static_assert(offsetof(UContext<ContextTraits64>, mcontext) ==
                  offsetof(ucontext_t, uc_mcontext),
              "context offset mismtach");
static_assert(offsetof(UContext<ContextTraits64>, mcontext.gregs) ==
                  offsetof(ucontext_t, uc_mcontext.gregs),
              "context offset mismatch");
static_assert(offsetof(UContext<ContextTraits64>, mcontext.fpregs) ==
                  offsetof(ucontext_t, uc_mcontext.fpregs),
              "context offset mismatch");
#endif

#else
#error Port.
#endif  // ARCH_CPU_X86_FAMILY

#pragma pack(pop)

}  // namespace internal
}  // namespace crashpad

#endif  // CRASHPAD_SNAPSHOT_LINUX_SNAPSHOT_SIGNAL_CONTEXT_H_
