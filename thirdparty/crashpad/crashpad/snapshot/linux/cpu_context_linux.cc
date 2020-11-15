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

#include "snapshot/linux/cpu_context_linux.h"

#include <stddef.h>
#include <string.h>

#include "base/logging.h"

namespace crashpad {
namespace internal {

#if defined(ARCH_CPU_X86_FAMILY)

#define SET_GPRS32()                         \
  do {                                       \
    context->eax = thread_context.eax;       \
    context->ebx = thread_context.ebx;       \
    context->ecx = thread_context.ecx;       \
    context->edx = thread_context.edx;       \
    context->edi = thread_context.edi;       \
    context->esi = thread_context.esi;       \
    context->ebp = thread_context.ebp;       \
    context->esp = thread_context.esp;       \
    context->eip = thread_context.eip;       \
    context->eflags = thread_context.eflags; \
    context->cs = thread_context.xcs;        \
    context->ds = thread_context.xds;        \
    context->es = thread_context.xes;        \
    context->fs = thread_context.xfs;        \
    context->gs = thread_context.xgs;        \
    context->ss = thread_context.xss;        \
  } while (false)

void InitializeCPUContextX86(const ThreadContext::t32_t& thread_context,
                             const FloatContext::f32_t& float_context,
                             CPUContextX86* context) {
  SET_GPRS32();

  static_assert(sizeof(context->fxsave) == sizeof(float_context.fxsave),
                "fxsave size mismatch");
  memcpy(&context->fxsave, &float_context.fxsave, sizeof(context->fxsave));

  // TODO(jperaza): debug registers
  context->dr0 = 0;
  context->dr1 = 0;
  context->dr2 = 0;
  context->dr3 = 0;
  context->dr4 = 0;
  context->dr5 = 0;
  context->dr6 = 0;
  context->dr7 = 0;

}

void InitializeCPUContextX86(const SignalThreadContext32& thread_context,
                             const SignalFloatContext32& float_context,
                             CPUContextX86* context) {
  InitializeCPUContextX86_NoFloatingPoint(thread_context, context);
  CPUContextX86::FsaveToFxsave(float_context.fsave, &context->fxsave);
}

void InitializeCPUContextX86_NoFloatingPoint(
    const SignalThreadContext32& thread_context,
    CPUContextX86* context) {
  SET_GPRS32();

  memset(&context->fxsave, 0, sizeof(context->fxsave));

  context->dr0 = 0;
  context->dr1 = 0;
  context->dr2 = 0;
  context->dr3 = 0;
  context->dr4 = 0;
  context->dr5 = 0;
  context->dr6 = 0;
  context->dr7 = 0;
}

#define SET_GPRS64()                         \
  do {                                       \
    context->rax = thread_context.rax;       \
    context->rbx = thread_context.rbx;       \
    context->rcx = thread_context.rcx;       \
    context->rdx = thread_context.rdx;       \
    context->rdi = thread_context.rdi;       \
    context->rsi = thread_context.rsi;       \
    context->rbp = thread_context.rbp;       \
    context->rsp = thread_context.rsp;       \
    context->r8 = thread_context.r8;         \
    context->r9 = thread_context.r9;         \
    context->r10 = thread_context.r10;       \
    context->r11 = thread_context.r11;       \
    context->r12 = thread_context.r12;       \
    context->r13 = thread_context.r13;       \
    context->r14 = thread_context.r14;       \
    context->r15 = thread_context.r15;       \
    context->rip = thread_context.rip;       \
    context->rflags = thread_context.eflags; \
    context->cs = thread_context.cs;         \
    context->fs = thread_context.fs;         \
    context->gs = thread_context.gs;         \
  } while (false)

void InitializeCPUContextX86_64(const ThreadContext::t64_t& thread_context,
                                const FloatContext::f64_t& float_context,
                                CPUContextX86_64* context) {
  SET_GPRS64();

  static_assert(sizeof(context->fxsave) == sizeof(float_context.fxsave),
                "fxsave size mismatch");
  memcpy(&context->fxsave, &float_context.fxsave, sizeof(context->fxsave));

  // TODO(jperaza): debug registers.
  context->dr0 = 0;
  context->dr1 = 0;
  context->dr2 = 0;
  context->dr3 = 0;
  context->dr4 = 0;
  context->dr5 = 0;
  context->dr6 = 0;
  context->dr7 = 0;
}

void InitializeCPUContextX86_64(const SignalThreadContext64& thread_context,
                                const SignalFloatContext64& float_context,
                                CPUContextX86_64* context) {
  SET_GPRS64();

  static_assert(
      std::is_same<SignalFloatContext64, CPUContextX86_64::Fxsave>::value,
      "signal float context has unexpected type");
  memcpy(&context->fxsave, &float_context, sizeof(context->fxsave));

  context->dr0 = 0;
  context->dr1 = 0;
  context->dr2 = 0;
  context->dr3 = 0;
  context->dr4 = 0;
  context->dr5 = 0;
  context->dr6 = 0;
  context->dr7 = 0;
}

void InitializeCPUContextX86_64_NoFloatingPoint(
    const SignalThreadContext64& thread_context,
    CPUContextX86_64* context) {
  SET_GPRS64();

  memset(&context->fxsave, 0, sizeof(context->fxsave));

  context->dr0 = 0;
  context->dr1 = 0;
  context->dr2 = 0;
  context->dr3 = 0;
  context->dr4 = 0;
  context->dr5 = 0;
  context->dr6 = 0;
  context->dr7 = 0;
}

#elif defined(ARCH_CPU_ARM_FAMILY)

void InitializeCPUContextARM(const ThreadContext::t32_t& thread_context,
                             const FloatContext::f32_t& float_context,
                             CPUContextARM* context) {
  static_assert(sizeof(context->regs) == sizeof(thread_context.regs),
                "registers size mismatch");
  memcpy(&context->regs, &thread_context.regs, sizeof(context->regs));
  context->fp = thread_context.fp;
  context->ip = thread_context.ip;
  context->sp = thread_context.sp;
  context->lr = thread_context.lr;
  context->pc = thread_context.pc;
  context->cpsr = thread_context.cpsr;

  static_assert(sizeof(context->vfp_regs) == sizeof(float_context.vfp),
                "vfp size mismatch");
  context->have_vfp_regs = float_context.have_vfp;
  if (float_context.have_vfp) {
    memcpy(&context->vfp_regs, &float_context.vfp, sizeof(context->vfp_regs));
  }

  static_assert(sizeof(context->fpa_regs) == sizeof(float_context.fpregs),
                "fpregs size mismatch");
  context->have_fpa_regs = float_context.have_fpregs;
  if (float_context.have_fpregs) {
    memcpy(
        &context->fpa_regs, &float_context.fpregs, sizeof(context->fpa_regs));
  }
}

void InitializeCPUContextARM_NoFloatingPoint(
    const SignalThreadContext32& thread_context,
    CPUContextARM* context) {
  static_assert(sizeof(context->regs) == sizeof(thread_context.regs),
                "registers size mismatch");
  memcpy(&context->regs, &thread_context.regs, sizeof(context->regs));
  context->fp = thread_context.fp;
  context->ip = thread_context.ip;
  context->sp = thread_context.sp;
  context->lr = thread_context.lr;
  context->pc = thread_context.pc;
  context->cpsr = thread_context.cpsr;

  memset(&context->fpa_regs, 0, sizeof(context->fpa_regs));
  memset(&context->vfp_regs, 0, sizeof(context->vfp_regs));
  context->have_fpa_regs = false;
  context->have_vfp_regs = false;
}

void InitializeCPUContextARM64(const ThreadContext::t64_t& thread_context,
                               const FloatContext::f64_t& float_context,
                               CPUContextARM64* context) {
  InitializeCPUContextARM64_NoFloatingPoint(thread_context, context);

  static_assert(sizeof(context->fpsimd) == sizeof(float_context.vregs),
                "fpsimd context size mismatch");
  memcpy(context->fpsimd, float_context.vregs, sizeof(context->fpsimd));
  context->fpsr = float_context.fpsr;
  context->fpcr = float_context.fpcr;
}

void InitializeCPUContextARM64_NoFloatingPoint(
    const ThreadContext::t64_t& thread_context,
    CPUContextARM64* context) {
  static_assert(sizeof(context->regs) == sizeof(thread_context.regs),
                "gpr context size mismtach");
  memcpy(context->regs, thread_context.regs, sizeof(context->regs));
  context->sp = thread_context.sp;
  context->pc = thread_context.pc;
  context->pstate = thread_context.pstate;

  memset(&context->fpsimd, 0, sizeof(context->fpsimd));
  context->fpsr = 0;
  context->fpcr = 0;
}

void InitializeCPUContextARM64_OnlyFPSIMD(
    const SignalFPSIMDContext& float_context,
    CPUContextARM64* context) {
  static_assert(sizeof(context->fpsimd) == sizeof(float_context.vregs),
                "fpsimd context size mismatch");
  memcpy(context->fpsimd, float_context.vregs, sizeof(context->fpsimd));
  context->fpsr = float_context.fpsr;
  context->fpcr = float_context.fpcr;
}

#endif  // ARCH_CPU_X86_FAMILY

}  // namespace internal
}  // namespace crashpad
