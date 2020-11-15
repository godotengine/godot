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

#include "minidump/test/minidump_context_test_util.h"

#include <string.h>
#include <sys/types.h>

#include "base/format_macros.h"
#include "base/macros.h"
#include "base/strings/stringprintf.h"
#include "gtest/gtest.h"
#include "snapshot/cpu_context.h"
#include "snapshot/test/test_cpu_context.h"
#include "test/hex_string.h"

namespace crashpad {
namespace test {

void InitializeMinidumpContextX86(MinidumpContextX86* context, uint32_t seed) {
  if (seed == 0) {
    memset(context, 0, sizeof(*context));
    context->context_flags = kMinidumpContextX86;
    return;
  }

  context->context_flags = kMinidumpContextX86All;

  uint32_t value = seed;

  context->eax = value++;
  context->ebx = value++;
  context->ecx = value++;
  context->edx = value++;
  context->edi = value++;
  context->esi = value++;
  context->ebp = value++;
  context->esp = value++;
  context->eip = value++;
  context->eflags = value++;
  context->cs = value++ & 0xffff;
  context->ds = value++ & 0xffff;
  context->es = value++ & 0xffff;
  context->fs = value++ & 0xffff;
  context->gs = value++ & 0xffff;
  context->ss = value++ & 0xffff;

  InitializeCPUContextX86Fxsave(&context->fxsave, &value);
  CPUContextX86::FxsaveToFsave(context->fxsave, &context->fsave);

  context->dr0 = value++;
  context->dr1 = value++;
  context->dr2 = value++;
  context->dr3 = value++;
  value += 2;  // Minidumps don’t carry dr4 or dr5.
  context->dr6 = value++;
  context->dr7 = value++;

  // Set this field last, because it has no analogue in CPUContextX86.
  context->float_save.spare_0 = value++;
}

void InitializeMinidumpContextAMD64(MinidumpContextAMD64* context,
                                    uint32_t seed) {
  if (seed == 0) {
    memset(context, 0, sizeof(*context));
    context->context_flags = kMinidumpContextAMD64;
    return;
  }

  context->context_flags = kMinidumpContextAMD64All;

  uint32_t value = seed;

  context->rax = value++;
  context->rbx = value++;
  context->rcx = value++;
  context->rdx = value++;
  context->rdi = value++;
  context->rsi = value++;
  context->rbp = value++;
  context->rsp = value++;
  context->r8 = value++;
  context->r9 = value++;
  context->r10 = value++;
  context->r11 = value++;
  context->r12 = value++;
  context->r13 = value++;
  context->r14 = value++;
  context->r15 = value++;
  context->rip = value++;
  context->eflags = value++;
  context->cs = static_cast<uint16_t>(value++);
  context->fs = static_cast<uint16_t>(value++);
  context->gs = static_cast<uint16_t>(value++);

  InitializeCPUContextX86_64Fxsave(&context->fxsave, &value);

  // mxcsr appears twice, and the two values should be aliased.
  context->mx_csr = context->fxsave.mxcsr;

  context->dr0 = value++;
  context->dr1 = value++;
  context->dr2 = value++;
  context->dr3 = value++;
  value += 2;  // Minidumps don’t carry dr4 or dr5.
  context->dr6 = value++;
  context->dr7 = value++;

  // Set these fields last, because they have no analogues in CPUContextX86_64.
  context->p1_home = value++;
  context->p2_home = value++;
  context->p3_home = value++;
  context->p4_home = value++;
  context->p5_home = value++;
  context->p6_home = value++;
  context->ds = static_cast<uint16_t>(value++);
  context->es = static_cast<uint16_t>(value++);
  context->ss = static_cast<uint16_t>(value++);
  for (size_t index = 0; index < arraysize(context->vector_register); ++index) {
    context->vector_register[index].lo = value++;
    context->vector_register[index].hi = value++;
  }
  context->vector_control = value++;
  context->debug_control = value++;
  context->last_branch_to_rip = value++;
  context->last_branch_from_rip = value++;
  context->last_exception_to_rip = value++;
  context->last_exception_from_rip = value++;
}

void InitializeMinidumpContextARM(MinidumpContextARM* context, uint32_t seed) {
  if (seed == 0) {
    memset(context, 0, sizeof(*context));
    context->context_flags = kMinidumpContextARM;
    return;
  }

  context->context_flags = kMinidumpContextARMAll;

  uint32_t value = seed;

  for (size_t index = 0; index < arraysize(context->regs); ++index) {
    context->regs[index] = value++;
  }
  context->fp = value++;
  context->ip = value++;
  context->ip = value++;
  context->sp = value++;
  context->lr = value++;
  context->pc = value++;
  context->cpsr = value++;

  for (size_t index = 0; index < arraysize(context->vfp); ++index) {
    context->vfp[index] = value++;
  }
  context->fpscr = value++;
}

void InitializeMinidumpContextARM64(MinidumpContextARM64* context,
                                    uint32_t seed) {
  if (seed == 0) {
    memset(context, 0, sizeof(*context));
    context->context_flags = kMinidumpContextARM64;
    return;
  }

  context->context_flags = kMinidumpContextARM64Full;

  uint32_t value = seed;

  for (size_t index = 0; index < arraysize(context->regs); ++index) {
    context->regs[index] = value++;
  }
  context->fp = value++;
  context->lr = value++;
  context->sp = value++;
  context->pc = value++;
  context->cpsr = value++;

  for (size_t index = 0; index < arraysize(context->fpsimd); ++index) {
    context->fpsimd[index].lo = value++;
    context->fpsimd[index].hi = value++;
  }
  context->fpsr = value++;
  context->fpcr = value++;
}

void InitializeMinidumpContextMIPS(MinidumpContextMIPS* context,
                                   uint32_t seed) {
  if (seed == 0) {
    memset(context, 0, sizeof(*context));
    context->context_flags = kMinidumpContextMIPS;
    return;
  }

  context->context_flags = kMinidumpContextMIPSAll;

  uint32_t value = seed;

  for (size_t index = 0; index < arraysize(context->regs); ++index) {
    context->regs[index] = value++;
  }

  context->mdlo = value++;
  context->mdhi = value++;
  context->epc = value++;
  context->badvaddr = value++;
  context->status = value++;
  context->cause = value++;

  for (size_t index = 0; index < arraysize(context->fpregs.fregs); ++index) {
    context->fpregs.fregs[index]._fp_fregs = static_cast<float>(value++);
  }

  context->fpcsr = value++;
  context->fir = value++;

  for (size_t index = 0; index < 3; ++index) {
    context->hi[index] = value++;
    context->lo[index] = value++;
  }

  context->dsp_control = value++;
}

void InitializeMinidumpContextMIPS64(MinidumpContextMIPS64* context,
                                     uint32_t seed) {
  if (seed == 0) {
    memset(context, 0, sizeof(*context));
    context->context_flags = kMinidumpContextMIPS64;
    return;
  }

  context->context_flags = kMinidumpContextMIPS64All;

  uint64_t value = seed;

  for (size_t index = 0; index < arraysize(context->regs); ++index) {
    context->regs[index] = value++;
  }

  context->mdlo = value++;
  context->mdhi = value++;
  context->epc = value++;
  context->badvaddr = value++;
  context->status = value++;
  context->cause = value++;

  for (size_t index = 0; index < arraysize(context->fpregs.dregs); ++index) {
    context->fpregs.dregs[index] = static_cast<double>(value++);
  }
  context->fpcsr = value++;
  context->fir = value++;

  for (size_t index = 0; index < 3; ++index) {
    context->hi[index] = value++;
    context->lo[index] = value++;
  }
  context->dsp_control = value++;
}

namespace {

// Using gtest assertions, compares |expected| to |observed|. This is
// templatized because the CPUContextX86::Fxsave and CPUContextX86_64::Fxsave
// are nearly identical but have different sizes for the members |xmm|,
// |reserved_4|, and |available|.
template <typename FxsaveType>
void ExpectMinidumpContextFxsave(const FxsaveType* expected,
                                 const FxsaveType* observed) {
  EXPECT_EQ(observed->fcw, expected->fcw);
  EXPECT_EQ(observed->fsw, expected->fsw);
  EXPECT_EQ(observed->ftw, expected->ftw);
  EXPECT_EQ(observed->reserved_1, expected->reserved_1);
  EXPECT_EQ(observed->fop, expected->fop);
  EXPECT_EQ(observed->fpu_ip, expected->fpu_ip);
  EXPECT_EQ(observed->fpu_cs, expected->fpu_cs);
  EXPECT_EQ(observed->reserved_2, expected->reserved_2);
  EXPECT_EQ(observed->fpu_dp, expected->fpu_dp);
  EXPECT_EQ(observed->fpu_ds, expected->fpu_ds);
  EXPECT_EQ(observed->reserved_3, expected->reserved_3);
  EXPECT_EQ(observed->mxcsr, expected->mxcsr);
  EXPECT_EQ(observed->mxcsr_mask, expected->mxcsr_mask);
  for (size_t st_mm_index = 0;
       st_mm_index < arraysize(expected->st_mm);
       ++st_mm_index) {
    SCOPED_TRACE(base::StringPrintf("st_mm_index %" PRIuS, st_mm_index));
    EXPECT_EQ(BytesToHexString(observed->st_mm[st_mm_index].st,
                               arraysize(observed->st_mm[st_mm_index].st)),
              BytesToHexString(expected->st_mm[st_mm_index].st,
                               arraysize(expected->st_mm[st_mm_index].st)));
    EXPECT_EQ(
        BytesToHexString(observed->st_mm[st_mm_index].st_reserved,
                         arraysize(observed->st_mm[st_mm_index].st_reserved)),
        BytesToHexString(expected->st_mm[st_mm_index].st_reserved,
                         arraysize(expected->st_mm[st_mm_index].st_reserved)));
  }
  for (size_t xmm_index = 0;
       xmm_index < arraysize(expected->xmm);
       ++xmm_index) {
    EXPECT_EQ(BytesToHexString(observed->xmm[xmm_index],
                               arraysize(observed->xmm[xmm_index])),
              BytesToHexString(expected->xmm[xmm_index],
                               arraysize(expected->xmm[xmm_index])))
        << "xmm_index " << xmm_index;
  }
  EXPECT_EQ(
      BytesToHexString(observed->reserved_4, arraysize(observed->reserved_4)),
      BytesToHexString(expected->reserved_4, arraysize(expected->reserved_4)));
  EXPECT_EQ(
      BytesToHexString(observed->available, arraysize(observed->available)),
      BytesToHexString(expected->available, arraysize(expected->available)));
}

}  // namespace

void ExpectMinidumpContextX86(
    uint32_t expect_seed, const MinidumpContextX86* observed, bool snapshot) {
  MinidumpContextX86 expected;
  InitializeMinidumpContextX86(&expected, expect_seed);

  EXPECT_EQ(observed->context_flags, expected.context_flags);
  EXPECT_EQ(observed->dr0, expected.dr0);
  EXPECT_EQ(observed->dr1, expected.dr1);
  EXPECT_EQ(observed->dr2, expected.dr2);
  EXPECT_EQ(observed->dr3, expected.dr3);
  EXPECT_EQ(observed->dr6, expected.dr6);
  EXPECT_EQ(observed->dr7, expected.dr7);

  EXPECT_EQ(observed->fsave.fcw, expected.fsave.fcw);
  EXPECT_EQ(observed->fsave.fsw, expected.fsave.fsw);
  EXPECT_EQ(observed->fsave.ftw, expected.fsave.ftw);
  EXPECT_EQ(observed->fsave.fpu_ip, expected.fsave.fpu_ip);
  EXPECT_EQ(observed->fsave.fpu_cs, expected.fsave.fpu_cs);
  EXPECT_EQ(observed->fsave.fpu_dp, expected.fsave.fpu_dp);
  EXPECT_EQ(observed->fsave.fpu_ds, expected.fsave.fpu_ds);
  for (size_t index = 0; index < arraysize(expected.fsave.st); ++index) {
    EXPECT_EQ(BytesToHexString(observed->fsave.st[index],
                               arraysize(observed->fsave.st[index])),
              BytesToHexString(expected.fsave.st[index],
                               arraysize(expected.fsave.st[index])))
        << "index " << index;
  }
  if (snapshot) {
    EXPECT_EQ(observed->float_save.spare_0, 0u);
  } else {
    EXPECT_EQ(observed->float_save.spare_0, expected.float_save.spare_0);
  }

  EXPECT_EQ(observed->gs, expected.gs);
  EXPECT_EQ(observed->fs, expected.fs);
  EXPECT_EQ(observed->es, expected.es);
  EXPECT_EQ(observed->ds, expected.ds);
  EXPECT_EQ(observed->edi, expected.edi);
  EXPECT_EQ(observed->esi, expected.esi);
  EXPECT_EQ(observed->ebx, expected.ebx);
  EXPECT_EQ(observed->edx, expected.edx);
  EXPECT_EQ(observed->ecx, expected.ecx);
  EXPECT_EQ(observed->eax, expected.eax);
  EXPECT_EQ(observed->ebp, expected.ebp);
  EXPECT_EQ(observed->eip, expected.eip);
  EXPECT_EQ(observed->cs, expected.cs);
  EXPECT_EQ(observed->eflags, expected.eflags);
  EXPECT_EQ(observed->esp, expected.esp);
  EXPECT_EQ(observed->ss, expected.ss);

  ExpectMinidumpContextFxsave(&expected.fxsave, &observed->fxsave);
}

void ExpectMinidumpContextAMD64(
    uint32_t expect_seed, const MinidumpContextAMD64* observed, bool snapshot) {
  MinidumpContextAMD64 expected;
  InitializeMinidumpContextAMD64(&expected, expect_seed);

  EXPECT_EQ(observed->context_flags, expected.context_flags);

  if (snapshot) {
    EXPECT_EQ(observed->p1_home, 0u);
    EXPECT_EQ(observed->p2_home, 0u);
    EXPECT_EQ(observed->p3_home, 0u);
    EXPECT_EQ(observed->p4_home, 0u);
    EXPECT_EQ(observed->p5_home, 0u);
    EXPECT_EQ(observed->p6_home, 0u);
  } else {
    EXPECT_EQ(observed->p1_home, expected.p1_home);
    EXPECT_EQ(observed->p2_home, expected.p2_home);
    EXPECT_EQ(observed->p3_home, expected.p3_home);
    EXPECT_EQ(observed->p4_home, expected.p4_home);
    EXPECT_EQ(observed->p5_home, expected.p5_home);
    EXPECT_EQ(observed->p6_home, expected.p6_home);
  }

  EXPECT_EQ(observed->mx_csr, expected.mx_csr);

  EXPECT_EQ(observed->cs, expected.cs);
  if (snapshot) {
    EXPECT_EQ(observed->ds, 0u);
    EXPECT_EQ(observed->es, 0u);
  } else {
    EXPECT_EQ(observed->ds, expected.ds);
    EXPECT_EQ(observed->es, expected.es);
  }
  EXPECT_EQ(observed->fs, expected.fs);
  EXPECT_EQ(observed->gs, expected.gs);
  if (snapshot) {
    EXPECT_EQ(observed->ss, 0u);
  } else {
    EXPECT_EQ(observed->ss, expected.ss);
  }

  EXPECT_EQ(observed->eflags, expected.eflags);

  EXPECT_EQ(observed->dr0, expected.dr0);
  EXPECT_EQ(observed->dr1, expected.dr1);
  EXPECT_EQ(observed->dr2, expected.dr2);
  EXPECT_EQ(observed->dr3, expected.dr3);
  EXPECT_EQ(observed->dr6, expected.dr6);
  EXPECT_EQ(observed->dr7, expected.dr7);

  EXPECT_EQ(observed->rax, expected.rax);
  EXPECT_EQ(observed->rcx, expected.rcx);
  EXPECT_EQ(observed->rdx, expected.rdx);
  EXPECT_EQ(observed->rbx, expected.rbx);
  EXPECT_EQ(observed->rsp, expected.rsp);
  EXPECT_EQ(observed->rbp, expected.rbp);
  EXPECT_EQ(observed->rsi, expected.rsi);
  EXPECT_EQ(observed->rdi, expected.rdi);
  EXPECT_EQ(observed->r8, expected.r8);
  EXPECT_EQ(observed->r9, expected.r9);
  EXPECT_EQ(observed->r10, expected.r10);
  EXPECT_EQ(observed->r11, expected.r11);
  EXPECT_EQ(observed->r12, expected.r12);
  EXPECT_EQ(observed->r13, expected.r13);
  EXPECT_EQ(observed->r14, expected.r14);
  EXPECT_EQ(observed->r15, expected.r15);
  EXPECT_EQ(observed->rip, expected.rip);

  ExpectMinidumpContextFxsave(&expected.fxsave, &observed->fxsave);

  for (size_t index = 0; index < arraysize(expected.vector_register); ++index) {
    if (snapshot) {
      EXPECT_EQ(observed->vector_register[index].lo, 0u) << "index " << index;
      EXPECT_EQ(observed->vector_register[index].hi, 0u) << "index " << index;
    } else {
      EXPECT_EQ(observed->vector_register[index].lo,
                expected.vector_register[index].lo)
          << "index " << index;
      EXPECT_EQ(observed->vector_register[index].hi,
                expected.vector_register[index].hi)
          << "index " << index;
    }
  }

  if (snapshot) {
    EXPECT_EQ(observed->vector_control, 0u);
    EXPECT_EQ(observed->debug_control, 0u);
    EXPECT_EQ(observed->last_branch_to_rip, 0u);
    EXPECT_EQ(observed->last_branch_from_rip, 0u);
    EXPECT_EQ(observed->last_exception_to_rip, 0u);
    EXPECT_EQ(observed->last_exception_from_rip, 0u);
  } else {
    EXPECT_EQ(observed->vector_control, expected.vector_control);
    EXPECT_EQ(observed->debug_control, expected.debug_control);
    EXPECT_EQ(observed->last_branch_to_rip, expected.last_branch_to_rip);
    EXPECT_EQ(observed->last_branch_from_rip, expected.last_branch_from_rip);
    EXPECT_EQ(observed->last_exception_to_rip, expected.last_exception_to_rip);
    EXPECT_EQ(observed->last_exception_from_rip,
              expected.last_exception_from_rip);
  }
}

void ExpectMinidumpContextARM(uint32_t expect_seed,
                              const MinidumpContextARM* observed,
                              bool snapshot) {
  MinidumpContextARM expected;
  InitializeMinidumpContextARM(&expected, expect_seed);

  EXPECT_EQ(observed->context_flags, expected.context_flags);

  for (size_t index = 0; index < arraysize(expected.regs); ++index) {
    EXPECT_EQ(observed->regs[index], expected.regs[index]);
  }
  EXPECT_EQ(observed->fp, expected.fp);
  EXPECT_EQ(observed->ip, expected.ip);
  EXPECT_EQ(observed->sp, expected.sp);
  EXPECT_EQ(observed->lr, expected.lr);
  EXPECT_EQ(observed->pc, expected.pc);
  EXPECT_EQ(observed->cpsr, expected.cpsr);

  EXPECT_EQ(observed->fpscr, expected.fpscr);
  for (size_t index = 0; index < arraysize(expected.vfp); ++index) {
    EXPECT_EQ(observed->vfp[index], expected.vfp[index]);
  }
  for (size_t index = 0; index < arraysize(expected.extra); ++index) {
    EXPECT_EQ(observed->extra[index], snapshot ? 0 : expected.extra[index]);
  }
}

void ExpectMinidumpContextARM64(uint32_t expect_seed,
                                const MinidumpContextARM64* observed,
                                bool snapshot) {
  MinidumpContextARM64 expected;
  InitializeMinidumpContextARM64(&expected, expect_seed);

  EXPECT_EQ(observed->context_flags, expected.context_flags);

  for (size_t index = 0; index < arraysize(expected.regs); ++index) {
    EXPECT_EQ(observed->regs[index], expected.regs[index]);
  }
  EXPECT_EQ(observed->cpsr, expected.cpsr);

  EXPECT_EQ(observed->fpsr, expected.fpsr);
  EXPECT_EQ(observed->fpcr, expected.fpcr);
  for (size_t index = 0; index < arraysize(expected.fpsimd); ++index) {
    EXPECT_EQ(observed->fpsimd[index].lo, expected.fpsimd[index].lo);
    EXPECT_EQ(observed->fpsimd[index].hi, expected.fpsimd[index].hi);
  }
}

void ExpectMinidumpContextMIPS(uint32_t expect_seed,
                               const MinidumpContextMIPS* observed,
                               bool snapshot) {
  MinidumpContextMIPS expected;
  InitializeMinidumpContextMIPS(&expected, expect_seed);

  EXPECT_EQ(observed->context_flags, expected.context_flags);

  for (size_t index = 0; index < arraysize(expected.regs); ++index) {
    EXPECT_EQ(observed->regs[index], expected.regs[index]);
  }

  EXPECT_EQ(observed->mdlo, expected.mdlo);
  EXPECT_EQ(observed->mdhi, expected.mdhi);
  EXPECT_EQ(observed->epc, expected.epc);
  EXPECT_EQ(observed->badvaddr, expected.badvaddr);
  EXPECT_EQ(observed->status, expected.status);
  EXPECT_EQ(observed->cause, expected.cause);

  for (size_t index = 0; index < arraysize(expected.fpregs.fregs); ++index) {
    EXPECT_EQ(observed->fpregs.fregs[index]._fp_fregs,
              expected.fpregs.fregs[index]._fp_fregs);
  }
  EXPECT_EQ(observed->fpcsr, expected.fpcsr);
  EXPECT_EQ(observed->fir, expected.fir);

  for (size_t index = 0; index < 3; ++index) {
    EXPECT_EQ(observed->hi[index], expected.hi[index]);
    EXPECT_EQ(observed->lo[index], expected.lo[index]);
  }
  EXPECT_EQ(observed->dsp_control, expected.dsp_control);
}

void ExpectMinidumpContextMIPS64(uint32_t expect_seed,
                                 const MinidumpContextMIPS64* observed,
                                 bool snapshot) {
  MinidumpContextMIPS64 expected;
  InitializeMinidumpContextMIPS64(&expected, expect_seed);

  EXPECT_EQ(observed->context_flags, expected.context_flags);

  for (size_t index = 0; index < arraysize(expected.regs); ++index) {
    EXPECT_EQ(observed->regs[index], expected.regs[index]);
  }

  EXPECT_EQ(observed->mdlo, expected.mdlo);
  EXPECT_EQ(observed->mdhi, expected.mdhi);
  EXPECT_EQ(observed->epc, expected.epc);
  EXPECT_EQ(observed->badvaddr, expected.badvaddr);
  EXPECT_EQ(observed->status, expected.status);
  EXPECT_EQ(observed->cause, expected.cause);

  for (size_t index = 0; index < arraysize(expected.fpregs.dregs); ++index) {
    EXPECT_EQ(observed->fpregs.dregs[index], expected.fpregs.dregs[index]);
  }
  EXPECT_EQ(observed->fpcsr, expected.fpcsr);
  EXPECT_EQ(observed->fir, expected.fir);

  for (size_t index = 0; index < 3; ++index) {
    EXPECT_EQ(observed->hi[index], expected.hi[index]);
    EXPECT_EQ(observed->lo[index], expected.lo[index]);
  }
  EXPECT_EQ(observed->dsp_control, expected.dsp_control);
}

}  // namespace test
}  // namespace crashpad
