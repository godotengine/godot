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

#include "snapshot/test/test_cpu_context.h"

#include <string.h>
#include <sys/types.h>

#include "base/macros.h"

namespace crashpad {
namespace test {

namespace {

// This is templatized because the CPUContextX86::Fxsave and
// CPUContextX86_64::Fxsave are nearly identical but have different sizes for
// the members |xmm|, |reserved_4|, and |available|.
template <typename FxsaveType>
void InitializeCPUContextFxsave(FxsaveType* fxsave, uint32_t* seed) {
  uint32_t value = *seed;

  fxsave->fcw = static_cast<uint16_t>(value++);
  fxsave->fsw = static_cast<uint16_t>(value++);
  fxsave->ftw = static_cast<uint8_t>(value++);
  fxsave->reserved_1 = static_cast<uint8_t>(value++);
  fxsave->fop = static_cast<uint16_t>(value++);
  fxsave->fpu_ip = value++;
  fxsave->fpu_cs = static_cast<uint16_t>(value++);
  fxsave->reserved_2 = static_cast<uint16_t>(value++);
  fxsave->fpu_dp = value++;
  fxsave->fpu_ds = static_cast<uint16_t>(value++);
  fxsave->reserved_3 = static_cast<uint16_t>(value++);
  fxsave->mxcsr = value++;
  fxsave->mxcsr_mask = value++;
  for (size_t st_mm_index = 0; st_mm_index < arraysize(fxsave->st_mm);
       ++st_mm_index) {
    for (size_t byte = 0; byte < arraysize(fxsave->st_mm[st_mm_index].st);
         ++byte) {
      fxsave->st_mm[st_mm_index].st[byte] = static_cast<uint8_t>(value++);
    }
    for (size_t byte = 0;
         byte < arraysize(fxsave->st_mm[st_mm_index].st_reserved);
         ++byte) {
      fxsave->st_mm[st_mm_index].st_reserved[byte] =
          static_cast<uint8_t>(value);
    }
  }
  for (size_t xmm_index = 0; xmm_index < arraysize(fxsave->xmm); ++xmm_index) {
    for (size_t byte = 0; byte < arraysize(fxsave->xmm[xmm_index]); ++byte) {
      fxsave->xmm[xmm_index][byte] = static_cast<uint8_t>(value++);
    }
  }
  for (size_t byte = 0; byte < arraysize(fxsave->reserved_4); ++byte) {
    fxsave->reserved_4[byte] = static_cast<uint8_t>(value++);
  }
  for (size_t byte = 0; byte < arraysize(fxsave->available); ++byte) {
    fxsave->available[byte] = static_cast<uint8_t>(value++);
  }

  *seed = value;
}

}  // namespace

void InitializeCPUContextX86Fxsave(CPUContextX86::Fxsave* fxsave,
                                   uint32_t* seed) {
  return InitializeCPUContextFxsave(fxsave, seed);
}

void InitializeCPUContextX86_64Fxsave(CPUContextX86_64::Fxsave* fxsave,
                                      uint32_t* seed) {
  return InitializeCPUContextFxsave(fxsave, seed);
}

void InitializeCPUContextX86(CPUContext* context, uint32_t seed) {
  context->architecture = kCPUArchitectureX86;

  if (seed == 0) {
    memset(context->x86, 0, sizeof(*context->x86));
    return;
  }

  uint32_t value = seed;

  context->x86->eax = value++;
  context->x86->ebx = value++;
  context->x86->ecx = value++;
  context->x86->edx = value++;
  context->x86->edi = value++;
  context->x86->esi = value++;
  context->x86->ebp = value++;
  context->x86->esp = value++;
  context->x86->eip = value++;
  context->x86->eflags = value++;
  context->x86->cs = static_cast<uint16_t>(value++);
  context->x86->ds = static_cast<uint16_t>(value++);
  context->x86->es = static_cast<uint16_t>(value++);
  context->x86->fs = static_cast<uint16_t>(value++);
  context->x86->gs = static_cast<uint16_t>(value++);
  context->x86->ss = static_cast<uint16_t>(value++);
  InitializeCPUContextX86Fxsave(&context->x86->fxsave, &value);
  context->x86->dr0 = value++;
  context->x86->dr1 = value++;
  context->x86->dr2 = value++;
  context->x86->dr3 = value++;
  context->x86->dr4 = value++;
  context->x86->dr5 = value++;
  context->x86->dr6 = value++;
  context->x86->dr7 = value++;
}

void InitializeCPUContextX86_64(CPUContext* context, uint32_t seed) {
  context->architecture = kCPUArchitectureX86_64;

  if (seed == 0) {
    memset(context->x86_64, 0, sizeof(*context->x86_64));
    return;
  }

  uint32_t value = seed;

  context->x86_64->rax = value++;
  context->x86_64->rbx = value++;
  context->x86_64->rcx = value++;
  context->x86_64->rdx = value++;
  context->x86_64->rdi = value++;
  context->x86_64->rsi = value++;
  context->x86_64->rbp = value++;
  context->x86_64->rsp = value++;
  context->x86_64->r8 = value++;
  context->x86_64->r9 = value++;
  context->x86_64->r10 = value++;
  context->x86_64->r11 = value++;
  context->x86_64->r12 = value++;
  context->x86_64->r13 = value++;
  context->x86_64->r14 = value++;
  context->x86_64->r15 = value++;
  context->x86_64->rip = value++;
  context->x86_64->rflags = value++;
  context->x86_64->cs = static_cast<uint16_t>(value++);
  context->x86_64->fs = static_cast<uint16_t>(value++);
  context->x86_64->gs = static_cast<uint16_t>(value++);
  InitializeCPUContextX86_64Fxsave(&context->x86_64->fxsave, &value);
  context->x86_64->dr0 = value++;
  context->x86_64->dr1 = value++;
  context->x86_64->dr2 = value++;
  context->x86_64->dr3 = value++;
  context->x86_64->dr4 = value++;
  context->x86_64->dr5 = value++;
  context->x86_64->dr6 = value++;
  context->x86_64->dr7 = value++;
}

void InitializeCPUContextARM(CPUContext* context, uint32_t seed) {
  context->architecture = kCPUArchitectureARM;
  CPUContextARM* arm = context->arm;

  if (seed == 0) {
    memset(arm, 0, sizeof(*arm));
    return;
  }

  uint32_t value = seed;

  for (size_t index = 0; index < arraysize(arm->regs); ++index) {
    arm->regs[index] = value++;
  }
  arm->fp = value++;
  arm->ip = value++;
  arm->ip = value++;
  arm->sp = value++;
  arm->lr = value++;
  arm->pc = value++;
  arm->cpsr = value++;

  for (size_t index = 0; index < arraysize(arm->vfp_regs.vfp); ++index) {
    arm->vfp_regs.vfp[index] = value++;
  }
  arm->vfp_regs.fpscr = value++;

  arm->have_fpa_regs = false;
  arm->have_vfp_regs = true;
}

void InitializeCPUContextARM64(CPUContext* context, uint32_t seed) {
  context->architecture = kCPUArchitectureARM64;
  CPUContextARM64* arm64 = context->arm64;

  if (seed == 0) {
    memset(arm64, 0, sizeof(*arm64));
    return;
  }

  uint32_t value = seed;

  for (size_t index = 0; index < arraysize(arm64->regs); ++index) {
    arm64->regs[index] = value++;
  }
  arm64->sp = value++;
  arm64->pc = value++;
  arm64->pstate = value++;

  for (size_t index = 0; index < arraysize(arm64->fpsimd); ++index) {
    arm64->fpsimd[index].lo = value++;
    arm64->fpsimd[index].hi = value++;
  }
  arm64->fpsr = value++;
  arm64->fpcr = value++;
}

void InitializeCPUContextMIPS(CPUContext* context, uint32_t seed) {
  context->architecture = kCPUArchitectureMIPSEL;
  CPUContextMIPS* mipsel = context->mipsel;

  if (seed == 0) {
    memset(mipsel, 0, sizeof(*mipsel));
    return;
  }

  uint32_t value = seed;

  for (size_t index = 0; index < arraysize(mipsel->regs); ++index) {
    mipsel->regs[index] = value++;
  }

  mipsel->mdlo = value++;
  mipsel->mdhi = value++;
  mipsel->cp0_epc = value++;
  mipsel->cp0_badvaddr = value++;
  mipsel->cp0_status = value++;
  mipsel->cp0_cause = value++;

  for (size_t index = 0; index < arraysize(mipsel->fpregs.fregs); ++index) {
    mipsel->fpregs.fregs[index]._fp_fregs = static_cast<float>(value++);
  }

  mipsel->fpcsr = value++;
  mipsel->fir = value++;

  for (size_t index = 0; index < 3; ++index) {
    mipsel->hi[index] = value++;
    mipsel->lo[index] = value++;
  }
  mipsel->dsp_control = value++;
}

void InitializeCPUContextMIPS64(CPUContext* context, uint32_t seed) {
  context->architecture = kCPUArchitectureMIPS64EL;
  CPUContextMIPS64* mips64 = context->mips64;

  if (seed == 0) {
    memset(mips64, 0, sizeof(*mips64));
    return;
  }

  uint64_t value = seed;

  for (size_t index = 0; index < arraysize(mips64->regs); ++index) {
    mips64->regs[index] = value++;
  }

  mips64->mdlo = value++;
  mips64->mdhi = value++;
  mips64->cp0_epc = value++;
  mips64->cp0_badvaddr = value++;
  mips64->cp0_status = value++;
  mips64->cp0_cause = value++;

  for (size_t index = 0; index < arraysize(mips64->fpregs.dregs); ++index) {
    mips64->fpregs.dregs[index] = static_cast<double>(value++);
  }

  mips64->fpcsr = value++;
  mips64->fir = value++;

  for (size_t index = 0; index < 3; ++index) {
    mips64->hi[index] = value++;
    mips64->lo[index] = value++;
  }
  mips64->dsp_control = value++;
}

}  // namespace test
}  // namespace crashpad
