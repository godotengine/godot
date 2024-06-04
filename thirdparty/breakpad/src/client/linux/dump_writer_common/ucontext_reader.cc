// Copyright (c) 2014, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "client/linux/dump_writer_common/ucontext_reader.h"

#include "common/linux/linux_libc_support.h"
#include "google_breakpad/common/minidump_format.h"

namespace google_breakpad {

// Minidump defines register structures which are different from the raw
// structures which we get from the kernel. These are platform specific
// functions to juggle the ucontext_t and user structures into minidump format.

#if defined(__i386__)

uintptr_t UContextReader::GetStackPointer(const ucontext_t* uc) {
  return uc->uc_mcontext.gregs[REG_ESP];
}

uintptr_t UContextReader::GetInstructionPointer(const ucontext_t* uc) {
  return uc->uc_mcontext.gregs[REG_EIP];
}

void UContextReader::FillCPUContext(RawContextCPU* out, const ucontext_t* uc,
                                    const fpstate_t* fp) {
  const greg_t* regs = uc->uc_mcontext.gregs;

  out->context_flags = MD_CONTEXT_X86_FULL |
                       MD_CONTEXT_X86_FLOATING_POINT;

  out->gs = regs[REG_GS];
  out->fs = regs[REG_FS];
  out->es = regs[REG_ES];
  out->ds = regs[REG_DS];

  out->edi = regs[REG_EDI];
  out->esi = regs[REG_ESI];
  out->ebx = regs[REG_EBX];
  out->edx = regs[REG_EDX];
  out->ecx = regs[REG_ECX];
  out->eax = regs[REG_EAX];

  out->ebp = regs[REG_EBP];
  out->eip = regs[REG_EIP];
  out->cs = regs[REG_CS];
  out->eflags = regs[REG_EFL];
  out->esp = regs[REG_UESP];
  out->ss = regs[REG_SS];

  out->float_save.control_word = fp->cw;
  out->float_save.status_word = fp->sw;
  out->float_save.tag_word = fp->tag;
  out->float_save.error_offset = fp->ipoff;
  out->float_save.error_selector = fp->cssel;
  out->float_save.data_offset = fp->dataoff;
  out->float_save.data_selector = fp->datasel;

  // 8 registers * 10 bytes per register.
  my_memcpy(out->float_save.register_area, fp->_st, 10 * 8);
}

#elif defined(__x86_64)

uintptr_t UContextReader::GetStackPointer(const ucontext_t* uc) {
  return uc->uc_mcontext.gregs[REG_RSP];
}

uintptr_t UContextReader::GetInstructionPointer(const ucontext_t* uc) {
  return uc->uc_mcontext.gregs[REG_RIP];
}

void UContextReader::FillCPUContext(RawContextCPU* out, const ucontext_t* uc,
                                    const fpstate_t* fpregs) {
  const greg_t* regs = uc->uc_mcontext.gregs;

  out->context_flags = MD_CONTEXT_AMD64_FULL;

  out->cs = regs[REG_CSGSFS] & 0xffff;

  out->fs = (regs[REG_CSGSFS] >> 32) & 0xffff;
  out->gs = (regs[REG_CSGSFS] >> 16) & 0xffff;

  out->eflags = regs[REG_EFL];

  out->rax = regs[REG_RAX];
  out->rcx = regs[REG_RCX];
  out->rdx = regs[REG_RDX];
  out->rbx = regs[REG_RBX];

  out->rsp = regs[REG_RSP];
  out->rbp = regs[REG_RBP];
  out->rsi = regs[REG_RSI];
  out->rdi = regs[REG_RDI];
  out->r8 = regs[REG_R8];
  out->r9 = regs[REG_R9];
  out->r10 = regs[REG_R10];
  out->r11 = regs[REG_R11];
  out->r12 = regs[REG_R12];
  out->r13 = regs[REG_R13];
  out->r14 = regs[REG_R14];
  out->r15 = regs[REG_R15];

  out->rip = regs[REG_RIP];

  out->flt_save.control_word = fpregs->cwd;
  out->flt_save.status_word = fpregs->swd;
  out->flt_save.tag_word = fpregs->ftw;
  out->flt_save.error_opcode = fpregs->fop;
  out->flt_save.error_offset = fpregs->rip;
  out->flt_save.data_offset = fpregs->rdp;
  out->flt_save.error_selector = 0;  // We don't have this.
  out->flt_save.data_selector = 0;  // We don't have this.
  out->flt_save.mx_csr = fpregs->mxcsr;
  out->flt_save.mx_csr_mask = fpregs->mxcr_mask;
  my_memcpy(&out->flt_save.float_registers, &fpregs->_st, 8 * 16);
  my_memcpy(&out->flt_save.xmm_registers, &fpregs->_xmm, 16 * 16);
}

#elif defined(__ARM_EABI__)

uintptr_t UContextReader::GetStackPointer(const ucontext_t* uc) {
  return uc->uc_mcontext.arm_sp;
}

uintptr_t UContextReader::GetInstructionPointer(const ucontext_t* uc) {
  return uc->uc_mcontext.arm_pc;
}

void UContextReader::FillCPUContext(RawContextCPU* out, const ucontext_t* uc) {
  out->context_flags = MD_CONTEXT_ARM_FULL;

  out->iregs[0] = uc->uc_mcontext.arm_r0;
  out->iregs[1] = uc->uc_mcontext.arm_r1;
  out->iregs[2] = uc->uc_mcontext.arm_r2;
  out->iregs[3] = uc->uc_mcontext.arm_r3;
  out->iregs[4] = uc->uc_mcontext.arm_r4;
  out->iregs[5] = uc->uc_mcontext.arm_r5;
  out->iregs[6] = uc->uc_mcontext.arm_r6;
  out->iregs[7] = uc->uc_mcontext.arm_r7;
  out->iregs[8] = uc->uc_mcontext.arm_r8;
  out->iregs[9] = uc->uc_mcontext.arm_r9;
  out->iregs[10] = uc->uc_mcontext.arm_r10;

  out->iregs[11] = uc->uc_mcontext.arm_fp;
  out->iregs[12] = uc->uc_mcontext.arm_ip;
  out->iregs[13] = uc->uc_mcontext.arm_sp;
  out->iregs[14] = uc->uc_mcontext.arm_lr;
  out->iregs[15] = uc->uc_mcontext.arm_pc;

  out->cpsr = uc->uc_mcontext.arm_cpsr;

  // TODO: fix this after fixing ExceptionHandler
  out->float_save.fpscr = 0;
  my_memset(&out->float_save.regs, 0, sizeof(out->float_save.regs));
  my_memset(&out->float_save.extra, 0, sizeof(out->float_save.extra));
}

#elif defined(__aarch64__)

uintptr_t UContextReader::GetStackPointer(const ucontext_t* uc) {
  return uc->uc_mcontext.sp;
}

uintptr_t UContextReader::GetInstructionPointer(const ucontext_t* uc) {
  return uc->uc_mcontext.pc;
}

void UContextReader::FillCPUContext(RawContextCPU* out, const ucontext_t* uc,
                                    const struct fpsimd_context* fpregs) {
  out->context_flags = MD_CONTEXT_ARM64_FULL_OLD;

  out->cpsr = static_cast<uint32_t>(uc->uc_mcontext.pstate);
  for (int i = 0; i < MD_CONTEXT_ARM64_REG_SP; ++i)
    out->iregs[i] = uc->uc_mcontext.regs[i];
  out->iregs[MD_CONTEXT_ARM64_REG_SP] = uc->uc_mcontext.sp;
  out->iregs[MD_CONTEXT_ARM64_REG_PC] = uc->uc_mcontext.pc;

  out->float_save.fpsr = fpregs->fpsr;
  out->float_save.fpcr = fpregs->fpcr;
  my_memcpy(&out->float_save.regs, &fpregs->vregs,
      MD_FLOATINGSAVEAREA_ARM64_FPR_COUNT * 16);
}

#elif defined(__mips__)

uintptr_t UContextReader::GetStackPointer(const ucontext_t* uc) {
  return uc->uc_mcontext.gregs[MD_CONTEXT_MIPS_REG_SP];
}

uintptr_t UContextReader::GetInstructionPointer(const ucontext_t* uc) {
  return uc->uc_mcontext.pc;
}

void UContextReader::FillCPUContext(RawContextCPU* out, const ucontext_t* uc) {
#if _MIPS_SIM == _ABI64
  out->context_flags = MD_CONTEXT_MIPS64_FULL;
#elif _MIPS_SIM == _ABIO32
  out->context_flags = MD_CONTEXT_MIPS_FULL;
#else
#error "This mips ABI is currently not supported (n32)"
#endif

  for (int i = 0; i < MD_CONTEXT_MIPS_GPR_COUNT; ++i)
    out->iregs[i] = uc->uc_mcontext.gregs[i];

  out->mdhi = uc->uc_mcontext.mdhi;
  out->mdlo = uc->uc_mcontext.mdlo;

  out->hi[0] = uc->uc_mcontext.hi1;
  out->hi[1] = uc->uc_mcontext.hi2;
  out->hi[2] = uc->uc_mcontext.hi3;
  out->lo[0] = uc->uc_mcontext.lo1;
  out->lo[1] = uc->uc_mcontext.lo2;
  out->lo[2] = uc->uc_mcontext.lo3;
  out->dsp_control = uc->uc_mcontext.dsp;

  out->epc = uc->uc_mcontext.pc;
  out->badvaddr = 0;  // Not reported in signal context.
  out->status = 0;  // Not reported in signal context.
  out->cause = 0;  // Not reported in signal context.

  for (int i = 0; i < MD_FLOATINGSAVEAREA_MIPS_FPR_COUNT; ++i)
    out->float_save.regs[i] = uc->uc_mcontext.fpregs.fp_r.fp_dregs[i];

  out->float_save.fpcsr = uc->uc_mcontext.fpc_csr;
#if _MIPS_SIM == _ABIO32
  out->float_save.fir = uc->uc_mcontext.fpc_eir;  // Unused.
#endif
}
#endif

}  // namespace google_breakpad
