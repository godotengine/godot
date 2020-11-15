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

#include "minidump/minidump_context_writer.h"

#include <windows.h>
#include <dbghelp.h>
#include <stdint.h>
#include <string.h>

#include "base/compiler_specific.h"
#include "base/logging.h"
#include "snapshot/cpu_context.h"
#include "util/file/file_writer.h"
#include "util/stdlib/aligned_allocator.h"

namespace crashpad {

namespace {

// Sanity-check complex structures to ensure interoperability.
static_assert(sizeof(MinidumpContextX86) == 716, "MinidumpContextX86 size");
static_assert(sizeof(MinidumpContextAMD64) == 1232,
              "MinidumpContextAMD64 size");

// These structures can also be checked against definitions in the Windows SDK.
#if defined(OS_WIN)
#if defined(ARCH_CPU_X86_FAMILY)
static_assert(sizeof(MinidumpContextX86) == sizeof(WOW64_CONTEXT),
              "WOW64_CONTEXT size");
#if defined(ARCH_CPU_X86)
static_assert(sizeof(MinidumpContextX86) == sizeof(CONTEXT), "CONTEXT size");
#elif defined(ARCH_CPU_X86_64)
static_assert(sizeof(MinidumpContextAMD64) == sizeof(CONTEXT), "CONTEXT size");
#endif
#endif  // ARCH_CPU_X86_FAMILY
#endif  // OS_WIN

}  // namespace

MinidumpContextWriter::~MinidumpContextWriter() {
}

// static
std::unique_ptr<MinidumpContextWriter>
MinidumpContextWriter::CreateFromSnapshot(const CPUContext* context_snapshot) {
  std::unique_ptr<MinidumpContextWriter> context;

  switch (context_snapshot->architecture) {
    case kCPUArchitectureX86: {
      MinidumpContextX86Writer* context_x86 = new MinidumpContextX86Writer();
      context.reset(context_x86);
      context_x86->InitializeFromSnapshot(context_snapshot->x86);
      break;
    }

    case kCPUArchitectureX86_64: {
      MinidumpContextAMD64Writer* context_amd64 =
          new MinidumpContextAMD64Writer();
      context.reset(context_amd64);
      context_amd64->InitializeFromSnapshot(context_snapshot->x86_64);
      break;
    }

    case kCPUArchitectureARM: {
      context = std::make_unique<MinidumpContextARMWriter>();
      reinterpret_cast<MinidumpContextARMWriter*>(context.get())
          ->InitializeFromSnapshot(context_snapshot->arm);
      break;
    }

    case kCPUArchitectureARM64: {
      context = std::make_unique<MinidumpContextARM64Writer>();
      reinterpret_cast<MinidumpContextARM64Writer*>(context.get())
          ->InitializeFromSnapshot(context_snapshot->arm64);
      break;
    }

    case kCPUArchitectureMIPSEL: {
      context = std::make_unique<MinidumpContextMIPSWriter>();
      reinterpret_cast<MinidumpContextMIPSWriter*>(context.get())
          ->InitializeFromSnapshot(context_snapshot->mipsel);
      break;
    }

    case kCPUArchitectureMIPS64EL: {
      context = std::make_unique<MinidumpContextMIPS64Writer>();
      reinterpret_cast<MinidumpContextMIPS64Writer*>(context.get())
          ->InitializeFromSnapshot(context_snapshot->mips64);
      break;
    }

    default: {
      LOG(ERROR) << "unknown context architecture "
                 << context_snapshot->architecture;
      break;
    }
  }

  return context;
}

size_t MinidumpContextWriter::SizeOfObject() {
  DCHECK_GE(state(), kStateFrozen);

  return ContextSize();
}

MinidumpContextX86Writer::MinidumpContextX86Writer()
    : MinidumpContextWriter(), context_() {
  context_.context_flags = kMinidumpContextX86;
}

MinidumpContextX86Writer::~MinidumpContextX86Writer() {
}


void MinidumpContextX86Writer::InitializeFromSnapshot(
    const CPUContextX86* context_snapshot) {
  DCHECK_EQ(state(), kStateMutable);
  DCHECK_EQ(context_.context_flags, kMinidumpContextX86);

  context_.context_flags = kMinidumpContextX86All;

  context_.dr0 = context_snapshot->dr0;
  context_.dr1 = context_snapshot->dr1;
  context_.dr2 = context_snapshot->dr2;
  context_.dr3 = context_snapshot->dr3;
  context_.dr6 = context_snapshot->dr6;
  context_.dr7 = context_snapshot->dr7;

  // The contents of context_.fsave effectively alias everything in
  // context_.fxsave that’s related to x87 FPU state. context_.fsave doesn’t
  // carry state specific to SSE (or later), such as mxcsr and the xmm
  // registers.
  CPUContextX86::FxsaveToFsave(context_snapshot->fxsave, &context_.fsave);

  context_.gs = context_snapshot->gs;
  context_.fs = context_snapshot->fs;
  context_.es = context_snapshot->es;
  context_.ds = context_snapshot->ds;
  context_.edi = context_snapshot->edi;
  context_.esi = context_snapshot->esi;
  context_.ebx = context_snapshot->ebx;
  context_.edx = context_snapshot->edx;
  context_.ecx = context_snapshot->ecx;
  context_.eax = context_snapshot->eax;
  context_.ebp = context_snapshot->ebp;
  context_.eip = context_snapshot->eip;
  context_.cs = context_snapshot->cs;
  context_.eflags = context_snapshot->eflags;
  context_.esp = context_snapshot->esp;
  context_.ss = context_snapshot->ss;

  // This is effectively a memcpy() of a big structure.
  context_.fxsave = context_snapshot->fxsave;
}

bool MinidumpContextX86Writer::WriteObject(FileWriterInterface* file_writer) {
  DCHECK_EQ(state(), kStateWritable);

  return file_writer->Write(&context_, sizeof(context_));
}

size_t MinidumpContextX86Writer::ContextSize() const {
  DCHECK_GE(state(), kStateFrozen);

  return sizeof(context_);
}

static_assert(alignof(MinidumpContextAMD64) >= 16,
              "MinidumpContextAMD64 alignment");
static_assert(alignof(MinidumpContextAMD64Writer) >=
                  alignof(MinidumpContextAMD64),
              "MinidumpContextAMD64Writer alignment");

MinidumpContextAMD64Writer::MinidumpContextAMD64Writer()
    : MinidumpContextWriter(), context_() {
  context_.context_flags = kMinidumpContextAMD64;
}

MinidumpContextAMD64Writer::~MinidumpContextAMD64Writer() {
}

// static
void* MinidumpContextAMD64Writer::operator new(size_t size) {
  // MinidumpContextAMD64 requests an alignment of 16, which can be larger than
  // what standard new provides. This may trigger MSVC warning C4316. As a
  // workaround to this language deficiency, provide a custom allocation
  // function to allocate a block meeting the alignment requirement.
  return AlignedAllocate(alignof(MinidumpContextAMD64Writer), size);
}

// static
void MinidumpContextAMD64Writer::operator delete(void* pointer) {
  return AlignedFree(pointer);
}

void MinidumpContextAMD64Writer::InitializeFromSnapshot(
    const CPUContextX86_64* context_snapshot) {
  DCHECK_EQ(state(), kStateMutable);
  DCHECK_EQ(context_.context_flags, kMinidumpContextAMD64);

  context_.context_flags = kMinidumpContextAMD64All;

  context_.mx_csr = context_snapshot->fxsave.mxcsr;
  context_.cs = context_snapshot->cs;
  context_.fs = context_snapshot->fs;
  context_.gs = context_snapshot->gs;
  // The top 32 bits of rflags are reserved/unused.
  context_.eflags = static_cast<uint32_t>(context_snapshot->rflags);
  context_.dr0 = context_snapshot->dr0;
  context_.dr1 = context_snapshot->dr1;
  context_.dr2 = context_snapshot->dr2;
  context_.dr3 = context_snapshot->dr3;
  context_.dr6 = context_snapshot->dr6;
  context_.dr7 = context_snapshot->dr7;
  context_.rax = context_snapshot->rax;
  context_.rcx = context_snapshot->rcx;
  context_.rdx = context_snapshot->rdx;
  context_.rbx = context_snapshot->rbx;
  context_.rsp = context_snapshot->rsp;
  context_.rbp = context_snapshot->rbp;
  context_.rsi = context_snapshot->rsi;
  context_.rdi = context_snapshot->rdi;
  context_.r8 = context_snapshot->r8;
  context_.r9 = context_snapshot->r9;
  context_.r10 = context_snapshot->r10;
  context_.r11 = context_snapshot->r11;
  context_.r12 = context_snapshot->r12;
  context_.r13 = context_snapshot->r13;
  context_.r14 = context_snapshot->r14;
  context_.r15 = context_snapshot->r15;
  context_.rip = context_snapshot->rip;

  // This is effectively a memcpy() of a big structure.
  context_.fxsave = context_snapshot->fxsave;
}

size_t MinidumpContextAMD64Writer::Alignment() {
  DCHECK_GE(state(), kStateFrozen);

  // Match the alignment of MinidumpContextAMD64.
  return 16;
}

bool MinidumpContextAMD64Writer::WriteObject(FileWriterInterface* file_writer) {
  DCHECK_EQ(state(), kStateWritable);

  return file_writer->Write(&context_, sizeof(context_));
}

size_t MinidumpContextAMD64Writer::ContextSize() const {
  DCHECK_GE(state(), kStateFrozen);

  return sizeof(context_);
}

MinidumpContextARMWriter::MinidumpContextARMWriter()
    : MinidumpContextWriter(), context_() {
  context_.context_flags = kMinidumpContextARM;
}

MinidumpContextARMWriter::~MinidumpContextARMWriter() = default;

void MinidumpContextARMWriter::InitializeFromSnapshot(
    const CPUContextARM* context_snapshot) {
  DCHECK_EQ(state(), kStateMutable);
  DCHECK_EQ(context_.context_flags, kMinidumpContextARM);

  context_.context_flags = kMinidumpContextARMAll;

  static_assert(sizeof(context_.regs) == sizeof(context_snapshot->regs),
                "GPRS size mismatch");
  memcpy(context_.regs, context_snapshot->regs, sizeof(context_.regs));
  context_.fp = context_snapshot->fp;
  context_.ip = context_snapshot->ip;
  context_.sp = context_snapshot->sp;
  context_.lr = context_snapshot->lr;
  context_.pc = context_snapshot->pc;
  context_.cpsr = context_snapshot->cpsr;

  context_.fpscr = context_snapshot->vfp_regs.fpscr;
  static_assert(sizeof(context_.vfp) == sizeof(context_snapshot->vfp_regs.vfp),
                "VFP size mismatch");
  memcpy(context_.vfp, context_snapshot->vfp_regs.vfp, sizeof(context_.vfp));

  memset(context_.extra, 0, sizeof(context_.extra));
}

bool MinidumpContextARMWriter::WriteObject(FileWriterInterface* file_writer) {
  DCHECK_EQ(state(), kStateWritable);
  return file_writer->Write(&context_, sizeof(context_));
}

size_t MinidumpContextARMWriter::ContextSize() const {
  DCHECK_GE(state(), kStateFrozen);
  return sizeof(context_);
}

MinidumpContextARM64Writer::MinidumpContextARM64Writer()
    : MinidumpContextWriter(), context_() {
  context_.context_flags = kMinidumpContextARM64;
}

MinidumpContextARM64Writer::~MinidumpContextARM64Writer() = default;

void MinidumpContextARM64Writer::InitializeFromSnapshot(
    const CPUContextARM64* context_snapshot) {
  DCHECK_EQ(state(), kStateMutable);
  DCHECK_EQ(context_.context_flags, kMinidumpContextARM64);

  context_.context_flags = kMinidumpContextARM64Full;

  if (context_snapshot->pstate >
      std::numeric_limits<decltype(context_.cpsr)>::max()) {
    LOG(WARNING) << "pstate truncation";
  }
  context_.cpsr =
      static_cast<decltype(context_.cpsr)>(context_snapshot->pstate);

  static_assert(
      sizeof(context_.regs) == sizeof(context_snapshot->regs) -
                                   2 * sizeof(context_snapshot->regs[0]),
      "GPRs size mismatch");
  memcpy(context_.regs, context_snapshot->regs, sizeof(context_.regs));
  context_.fp = context_snapshot->regs[29];
  context_.lr = context_snapshot->regs[30];
  context_.sp = context_snapshot->sp;
  context_.pc = context_snapshot->pc;

  static_assert(sizeof(context_.fpsimd) == sizeof(context_snapshot->fpsimd),
                "FPSIMD size mismatch");
  memcpy(context_.fpsimd, context_snapshot->fpsimd, sizeof(context_.fpsimd));
  context_.fpcr = context_snapshot->fpcr;
  context_.fpsr = context_snapshot->fpsr;

  memset(context_.bcr, 0, sizeof(context_.bcr));
  memset(context_.bvr, 0, sizeof(context_.bvr));
  memset(context_.wcr, 0, sizeof(context_.wcr));
  memset(context_.wvr, 0, sizeof(context_.wvr));
}

bool MinidumpContextARM64Writer::WriteObject(FileWriterInterface* file_writer) {
  DCHECK_EQ(state(), kStateWritable);
  return file_writer->Write(&context_, sizeof(context_));
}

size_t MinidumpContextARM64Writer::ContextSize() const {
  DCHECK_GE(state(), kStateFrozen);
  return sizeof(context_);
}

MinidumpContextMIPSWriter::MinidumpContextMIPSWriter()
    : MinidumpContextWriter(), context_() {
  context_.context_flags = kMinidumpContextMIPS;
}

MinidumpContextMIPSWriter::~MinidumpContextMIPSWriter() = default;

void MinidumpContextMIPSWriter::InitializeFromSnapshot(
    const CPUContextMIPS* context_snapshot) {
  DCHECK_EQ(state(), kStateMutable);
  DCHECK_EQ(context_.context_flags, kMinidumpContextMIPS);

  context_.context_flags = kMinidumpContextMIPSAll;

  static_assert(sizeof(context_.regs) == sizeof(context_snapshot->regs),
                "GPRs size mismatch");
  memcpy(context_.regs, context_snapshot->regs, sizeof(context_.regs));
  context_.mdhi = context_snapshot->mdhi;
  context_.mdlo = context_snapshot->mdlo;
  context_.epc = context_snapshot->cp0_epc;
  context_.badvaddr = context_snapshot->cp0_badvaddr;
  context_.status = context_snapshot->cp0_status;
  context_.cause = context_snapshot->cp0_cause;

  static_assert(sizeof(context_.fpregs) == sizeof(context_snapshot->fpregs),
                "FPRs size mismatch");
  memcpy(&context_.fpregs, &context_snapshot->fpregs, sizeof(context_.fpregs));
  context_.fpcsr = context_snapshot->fpcsr;
  context_.fir = context_snapshot->fir;

  for (size_t index = 0; index < 3; ++index) {
    context_.hi[index] = context_snapshot->hi[index];
    context_.lo[index] = context_snapshot->lo[index];
  }
  context_.dsp_control = context_snapshot->dsp_control;
}

bool MinidumpContextMIPSWriter::WriteObject(FileWriterInterface* file_writer) {
  DCHECK_EQ(state(), kStateWritable);
  return file_writer->Write(&context_, sizeof(context_));
}

size_t MinidumpContextMIPSWriter::ContextSize() const {
  DCHECK_GE(state(), kStateFrozen);
  return sizeof(context_);
}

MinidumpContextMIPS64Writer::MinidumpContextMIPS64Writer()
    : MinidumpContextWriter(), context_() {
  context_.context_flags = kMinidumpContextMIPS64;
}

MinidumpContextMIPS64Writer::~MinidumpContextMIPS64Writer() = default;

void MinidumpContextMIPS64Writer::InitializeFromSnapshot(
    const CPUContextMIPS64* context_snapshot) {
  DCHECK_EQ(state(), kStateMutable);
  DCHECK_EQ(context_.context_flags, kMinidumpContextMIPS64);

  context_.context_flags = kMinidumpContextMIPS64All;

  static_assert(sizeof(context_.regs) == sizeof(context_snapshot->regs),
                "GPRs size mismatch");
  memcpy(context_.regs, context_snapshot->regs, sizeof(context_.regs));
  context_.mdhi = context_snapshot->mdhi;
  context_.mdlo = context_snapshot->mdlo;
  context_.epc = context_snapshot->cp0_epc;
  context_.badvaddr = context_snapshot->cp0_badvaddr;
  context_.status = context_snapshot->cp0_status;
  context_.cause = context_snapshot->cp0_cause;

  static_assert(sizeof(context_.fpregs) == sizeof(context_snapshot->fpregs),
                "FPRs size mismatch");
  memcpy(context_.fpregs.dregs,
         context_snapshot->fpregs.dregs,
         sizeof(context_.fpregs.dregs));
  context_.fpcsr = context_snapshot->fpcsr;
  context_.fir = context_snapshot->fir;

  for (size_t index = 0; index < 3; ++index) {
    context_.hi[index] = context_snapshot->hi[index];
    context_.lo[index] = context_snapshot->lo[index];
  }
  context_.dsp_control = context_snapshot->dsp_control;
}

bool MinidumpContextMIPS64Writer::WriteObject(
    FileWriterInterface* file_writer) {
  DCHECK_EQ(state(), kStateWritable);
  return file_writer->Write(&context_, sizeof(context_));
}

size_t MinidumpContextMIPS64Writer::ContextSize() const {
  DCHECK_GE(state(), kStateFrozen);
  return sizeof(context_);
}

}  // namespace crashpad
