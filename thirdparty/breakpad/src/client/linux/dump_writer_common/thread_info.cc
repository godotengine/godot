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

#include "client/linux/dump_writer_common/thread_info.h"

#include <string.h>
#include <assert.h>

#include "common/linux/linux_libc_support.h"
#include "google_breakpad/common/minidump_format.h"

namespace {

#if defined(__i386__)
// Write a uint16_t to memory
//   out: memory location to write to
//   v: value to write.
void U16(void* out, uint16_t v) {
  my_memcpy(out, &v, sizeof(v));
}

// Write a uint32_t to memory
//   out: memory location to write to
//   v: value to write.
void U32(void* out, uint32_t v) {
  my_memcpy(out, &v, sizeof(v));
}
#endif

}

namespace google_breakpad {

#if defined(__i386__)

uintptr_t ThreadInfo::GetInstructionPointer() const {
  return regs.eip;
}

void ThreadInfo::FillCPUContext(RawContextCPU* out) const {
  out->context_flags = MD_CONTEXT_X86_ALL;

  out->dr0 = dregs[0];
  out->dr1 = dregs[1];
  out->dr2 = dregs[2];
  out->dr3 = dregs[3];
  // 4 and 5 deliberatly omitted because they aren't included in the minidump
  // format.
  out->dr6 = dregs[6];
  out->dr7 = dregs[7];

  out->gs = regs.xgs;
  out->fs = regs.xfs;
  out->es = regs.xes;
  out->ds = regs.xds;

  out->edi = regs.edi;
  out->esi = regs.esi;
  out->ebx = regs.ebx;
  out->edx = regs.edx;
  out->ecx = regs.ecx;
  out->eax = regs.eax;

  out->ebp = regs.ebp;
  out->eip = regs.eip;
  out->cs = regs.xcs;
  out->eflags = regs.eflags;
  out->esp = regs.esp;
  out->ss = regs.xss;

  out->float_save.control_word = fpregs.cwd;
  out->float_save.status_word = fpregs.swd;
  out->float_save.tag_word = fpregs.twd;
  out->float_save.error_offset = fpregs.fip;
  out->float_save.error_selector = fpregs.fcs;
  out->float_save.data_offset = fpregs.foo;
  out->float_save.data_selector = fpregs.fos;

  // 8 registers * 10 bytes per register.
  my_memcpy(out->float_save.register_area, fpregs.st_space, 10 * 8);

  // This matches the Intel fpsave format.
  U16(out->extended_registers + 0, fpregs.cwd);
  U16(out->extended_registers + 2, fpregs.swd);
  U16(out->extended_registers + 4, fpregs.twd);
  U16(out->extended_registers + 6, fpxregs.fop);
  U32(out->extended_registers + 8, fpxregs.fip);
  U16(out->extended_registers + 12, fpxregs.fcs);
  U32(out->extended_registers + 16, fpregs.foo);
  U16(out->extended_registers + 20, fpregs.fos);
  U32(out->extended_registers + 24, fpxregs.mxcsr);

  my_memcpy(out->extended_registers + 32, &fpxregs.st_space, 128);
  my_memcpy(out->extended_registers + 160, &fpxregs.xmm_space, 128);
}

#elif defined(__x86_64)

uintptr_t ThreadInfo::GetInstructionPointer() const {
  return regs.rip;
}

void ThreadInfo::FillCPUContext(RawContextCPU* out) const {
  out->context_flags = MD_CONTEXT_AMD64_FULL |
                       MD_CONTEXT_AMD64_SEGMENTS;

  out->cs = regs.cs;

  out->ds = regs.ds;
  out->es = regs.es;
  out->fs = regs.fs;
  out->gs = regs.gs;

  out->ss = regs.ss;
  out->eflags = regs.eflags;

  out->dr0 = dregs[0];
  out->dr1 = dregs[1];
  out->dr2 = dregs[2];
  out->dr3 = dregs[3];
  // 4 and 5 deliberatly omitted because they aren't included in the minidump
  // format.
  out->dr6 = dregs[6];
  out->dr7 = dregs[7];

  out->rax = regs.rax;
  out->rcx = regs.rcx;
  out->rdx = regs.rdx;
  out->rbx = regs.rbx;

  out->rsp = regs.rsp;

  out->rbp = regs.rbp;
  out->rsi = regs.rsi;
  out->rdi = regs.rdi;
  out->r8 = regs.r8;
  out->r9 = regs.r9;
  out->r10 = regs.r10;
  out->r11 = regs.r11;
  out->r12 = regs.r12;
  out->r13 = regs.r13;
  out->r14 = regs.r14;
  out->r15 = regs.r15;

  out->rip = regs.rip;

  out->flt_save.control_word = fpregs.cwd;
  out->flt_save.status_word = fpregs.swd;
  out->flt_save.tag_word = fpregs.ftw;
  out->flt_save.error_opcode = fpregs.fop;
  out->flt_save.error_offset = fpregs.rip;
  out->flt_save.error_selector = 0;  // We don't have this.
  out->flt_save.data_offset = fpregs.rdp;
  out->flt_save.data_selector = 0;   // We don't have this.
  out->flt_save.mx_csr = fpregs.mxcsr;
  out->flt_save.mx_csr_mask = fpregs.mxcr_mask;

  my_memcpy(&out->flt_save.float_registers, &fpregs.st_space, 8 * 16);
  my_memcpy(&out->flt_save.xmm_registers, &fpregs.xmm_space, 16 * 16);
}

#elif defined(__ARM_EABI__)

uintptr_t ThreadInfo::GetInstructionPointer() const {
  return regs.uregs[15];
}

void ThreadInfo::FillCPUContext(RawContextCPU* out) const {
  out->context_flags = MD_CONTEXT_ARM_FULL;

  for (int i = 0; i < MD_CONTEXT_ARM_GPR_COUNT; ++i)
    out->iregs[i] = regs.uregs[i];
  // No CPSR register in ThreadInfo(it's not accessible via ptrace)
  out->cpsr = 0;
#if !defined(__ANDROID__)
  out->float_save.fpscr = fpregs.fpsr |
    (static_cast<uint64_t>(fpregs.fpcr) << 32);
  // TODO: sort this out, actually collect floating point registers
  my_memset(&out->float_save.regs, 0, sizeof(out->float_save.regs));
  my_memset(&out->float_save.extra, 0, sizeof(out->float_save.extra));
#endif
}

#elif defined(__aarch64__)

uintptr_t ThreadInfo::GetInstructionPointer() const {
  return regs.pc;
}

void ThreadInfo::FillCPUContext(RawContextCPU* out) const {
  out->context_flags = MD_CONTEXT_ARM64_FULL_OLD;

  out->cpsr = static_cast<uint32_t>(regs.pstate);
  for (int i = 0; i < MD_CONTEXT_ARM64_REG_SP; ++i)
    out->iregs[i] = regs.regs[i];
  out->iregs[MD_CONTEXT_ARM64_REG_SP] = regs.sp;
  out->iregs[MD_CONTEXT_ARM64_REG_PC] = regs.pc;

  out->float_save.fpsr = fpregs.fpsr;
  out->float_save.fpcr = fpregs.fpcr;
  my_memcpy(&out->float_save.regs, &fpregs.vregs,
      MD_FLOATINGSAVEAREA_ARM64_FPR_COUNT * 16);
}

#elif defined(__mips__)

uintptr_t ThreadInfo::GetInstructionPointer() const {
  return mcontext.pc;
}

void ThreadInfo::FillCPUContext(RawContextCPU* out) const {
#if _MIPS_SIM == _ABI64
  out->context_flags = MD_CONTEXT_MIPS64_FULL;
#elif _MIPS_SIM == _ABIO32
  out->context_flags = MD_CONTEXT_MIPS_FULL;
#else
# error "This mips ABI is currently not supported (n32)"
#endif

  for (int i = 0; i < MD_CONTEXT_MIPS_GPR_COUNT; ++i)
    out->iregs[i] = mcontext.gregs[i];

  out->mdhi = mcontext.mdhi;
  out->mdlo = mcontext.mdlo;
  out->dsp_control = mcontext.dsp;

  out->hi[0] = mcontext.hi1;
  out->lo[0] = mcontext.lo1;
  out->hi[1] = mcontext.hi2;
  out->lo[1] = mcontext.lo2;
  out->hi[2] = mcontext.hi3;
  out->lo[2] = mcontext.lo3;

  out->epc = mcontext.pc;
  out->badvaddr = 0; // Not stored in mcontext
  out->status = 0; // Not stored in mcontext
  out->cause = 0; // Not stored in mcontext

  for (int i = 0; i < MD_FLOATINGSAVEAREA_MIPS_FPR_COUNT; ++i)
    out->float_save.regs[i] = mcontext.fpregs.fp_r.fp_fregs[i]._fp_fregs;

  out->float_save.fpcsr = mcontext.fpc_csr;
#if _MIPS_SIM == _ABIO32
  out->float_save.fir = mcontext.fpc_eir;
#endif
}
#endif  // __mips__

void ThreadInfo::GetGeneralPurposeRegisters(void** gp_regs, size_t* size) {
  assert(gp_regs || size);
#if defined(__mips__)
  if (gp_regs)
    *gp_regs = mcontext.gregs;
  if (size)
    *size = sizeof(mcontext.gregs);
#else
  if (gp_regs)
    *gp_regs = &regs;
  if (size)
    *size = sizeof(regs);
#endif
}

void ThreadInfo::GetFloatingPointRegisters(void** fp_regs, size_t* size) {
  assert(fp_regs || size);
#if defined(__mips__)
  if (fp_regs)
    *fp_regs = &mcontext.fpregs;
  if (size)
    *size = sizeof(mcontext.fpregs);
#else
  if (fp_regs)
    *fp_regs = &fpregs;
  if (size)
    *size = sizeof(fpregs);
#endif
}

}  // namespace google_breakpad
