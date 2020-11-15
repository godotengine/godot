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

#ifndef CRASHPAD_MINIDUMP_MINIDUMP_CONTEXT_H_
#define CRASHPAD_MINIDUMP_MINIDUMP_CONTEXT_H_

#include <stdint.h>

#include "base/compiler_specific.h"
#include "snapshot/cpu_context.h"
#include "util/numeric/int128.h"

namespace crashpad {

//! \brief Architecture-independent flags for `context_flags` fields in Minidump
//!    context structures.
//
// https://zachsaw.blogspot.com/2010/11/wow64-bug-getthreadcontext-may-return.html#c5639760895973344002
enum MinidumpContextFlags : uint32_t {
  //! \brief The thread was executing a trap handler in kernel mode
  //!     (`CONTEXT_EXCEPTION_ACTIVE`).
  //!
  //! If this bit is set, it indicates that the context is from a thread that
  //! was executing a trap handler in the kernel. This bit is only valid when
  //! ::kMinidumpContextExceptionReporting is also set. This bit is only used on
  //! Windows.
  kMinidumpContextExceptionActive = 0x08000000,

  //! \brief The thread was executing a system call in kernel mode
  //!     (`CONTEXT_SERVICE_ACTIVE`).
  //!
  //! If this bit is set, it indicates that the context is from a thread that
  //! was executing a system call in the kernel. This bit is only valid when
  //! ::kMinidumpContextExceptionReporting is also set. This bit is only used on
  //! Windows.
  kMinidumpContextServiceActive = 0x10000000,

  //! \brief Kernel-mode state reporting is desired
  //!     (`CONTEXT_EXCEPTION_REQUEST`).
  //!
  //! This bit is not used in context structures containing snapshots of thread
  //! CPU context. It used when calling `GetThreadContext()` on Windows to
  //! specify that kernel-mode state reporting
  //! (::kMinidumpContextExceptionReporting) is desired in the returned context
  //! structure.
  kMinidumpContextExceptionRequest = 0x40000000,

  //! \brief Kernel-mode state reporting is provided
  //!     (`CONTEXT_EXCEPTION_REPORTING`).
  //!
  //! If this bit is set, it indicates that the bits indicating how the thread
  //! had entered kernel mode (::kMinidumpContextExceptionActive and
  //! ::kMinidumpContextServiceActive) are valid. This bit is only used on
  //! Windows.
  kMinidumpContextExceptionReporting = 0x80000000,
};

//! \brief 32-bit x86-specifc flags for MinidumpContextX86::context_flags.
enum MinidumpContextX86Flags : uint32_t {
  //! \brief Identifies the context structure as 32-bit x86. This is the same as
  //!     `CONTEXT_i386` and `CONTEXT_i486` on Windows for this architecture.
  kMinidumpContextX86 = 0x00010000,

  //! \brief Indicates the validity of control registers (`CONTEXT_CONTROL`).
  //!
  //! The `ebp`, `eip`, `cs`, `eflags`, `esp`, and `ss` fields are valid.
  kMinidumpContextX86Control = kMinidumpContextX86 | 0x00000001,

  //! \brief Indicates the validity of non-control integer registers
  //!     (`CONTEXT_INTEGER`).
  //!
  //! The `edi`, `esi`, `ebx`, `edx`, `ecx, and `eax` fields are valid.
  kMinidumpContextX86Integer = kMinidumpContextX86 | 0x00000002,

  //! \brief Indicates the validity of non-control segment registers
  //!     (`CONTEXT_SEGMENTS`).
  //!
  //! The `gs`, `fs`, `es`, and `ds` fields are valid.
  kMinidumpContextX86Segment = kMinidumpContextX86 | 0x00000004,

  //! \brief Indicates the validity of floating-point state
  //!     (`CONTEXT_FLOATING_POINT`).
  //!
  //! The `fsave` field is valid. The `float_save` field is included in this
  //! definition, but its members have no practical use asdie from `fsave`.
  kMinidumpContextX86FloatingPoint = kMinidumpContextX86 | 0x00000008,

  //! \brief Indicates the validity of debug registers
  //!     (`CONTEXT_DEBUG_REGISTERS`).
  //!
  //! The `dr0` through `dr3`, `dr6`, and `dr7` fields are valid.
  kMinidumpContextX86Debug = kMinidumpContextX86 | 0x00000010,

  //! \brief Indicates the validity of extended registers in `fxsave` format
  //!     (`CONTEXT_EXTENDED_REGISTERS`).
  //!
  //! The `extended_registers` field is valid and contains `fxsave` data.
  kMinidumpContextX86Extended = kMinidumpContextX86 | 0x00000020,

  //! \brief Indicates the validity of `xsave` data (`CONTEXT_XSTATE`).
  //!
  //! The context contains `xsave` data. This is used with an extended context
  //! structure not currently defined here.
  kMinidumpContextX86Xstate = kMinidumpContextX86 | 0x00000040,

  //! \brief Indicates the validity of control, integer, and segment registers.
  //!     (`CONTEXT_FULL`).
  kMinidumpContextX86Full = kMinidumpContextX86Control |
                            kMinidumpContextX86Integer |
                            kMinidumpContextX86Segment,

  //! \brief Indicates the validity of all registers except `xsave` data.
  //!     (`CONTEXT_ALL`).
  kMinidumpContextX86All = kMinidumpContextX86Full |
                           kMinidumpContextX86FloatingPoint |
                           kMinidumpContextX86Debug |
                           kMinidumpContextX86Extended,
};

//! \brief A 32-bit x86 CPU context (register state) carried in a minidump file.
//!
//! This is analogous to the `CONTEXT` structure on Windows when targeting
//! 32-bit x86, and the `WOW64_CONTEXT` structure when targeting an x86-family
//! CPU, either 32- or 64-bit. This structure is used instead of `CONTEXT` or
//! `WOW64_CONTEXT` to make it available when targeting other architectures.
//!
//! \note This structure doesn’t carry `dr4` or `dr5`, which are obsolete and
//!     normally alias `dr6` and `dr7`, respectively. See Intel Software
//!     Developer’s Manual, Volume 3B: System Programming, Part 2 (253669-052),
//!     17.2.2 “Debug Registers DR4 and DR5”.
struct MinidumpContextX86 {
  //! \brief A bitfield composed of values of #MinidumpContextFlags and
  //!     #MinidumpContextX86Flags.
  //!
  //! This field identifies the context structure as a 32-bit x86 CPU context,
  //! and indicates which other fields in the structure are valid.
  uint32_t context_flags;

  uint32_t dr0;
  uint32_t dr1;
  uint32_t dr2;
  uint32_t dr3;
  uint32_t dr6;
  uint32_t dr7;

  // CPUContextX86::Fsave has identical layout to what the x86 CONTEXT structure
  // places here.
  CPUContextX86::Fsave fsave;
  union {
    uint32_t spare_0;  // As in the native x86 CONTEXT structure since Windows 8
    uint32_t cr0_npx_state;  // As in WOW64_CONTEXT and older SDKs’ x86 CONTEXT
  } float_save;

  uint32_t gs;
  uint32_t fs;
  uint32_t es;
  uint32_t ds;

  uint32_t edi;
  uint32_t esi;
  uint32_t ebx;
  uint32_t edx;
  uint32_t ecx;
  uint32_t eax;

  uint32_t ebp;
  uint32_t eip;
  uint32_t cs;
  uint32_t eflags;
  uint32_t esp;
  uint32_t ss;

  // CPUContextX86::Fxsave has identical layout to what the x86 CONTEXT
  // structure places here.
  CPUContextX86::Fxsave fxsave;
};

//! \brief x86_64-specific flags for MinidumpContextAMD64::context_flags.
enum MinidumpContextAMD64Flags : uint32_t {
  //! \brief Identifies the context structure as x86_64. This is the same as
  //!     `CONTEXT_AMD64` on Windows for this architecture.
  kMinidumpContextAMD64 = 0x00100000,

  //! \brief Indicates the validity of control registers (`CONTEXT_CONTROL`).
  //!
  //! The `cs`, `ss`, `eflags`, `rsp`, and `rip` fields are valid.
  kMinidumpContextAMD64Control = kMinidumpContextAMD64 | 0x00000001,

  //! \brief Indicates the validity of non-control integer registers
  //!     (`CONTEXT_INTEGER`).
  //!
  //! The `rax`, `rcx`, `rdx`, `rbx`, `rbp`, `rsi`, `rdi`, and `r8` through
  //! `r15` fields are valid.
  kMinidumpContextAMD64Integer = kMinidumpContextAMD64 | 0x00000002,

  //! \brief Indicates the validity of non-control segment registers
  //!     (`CONTEXT_SEGMENTS`).
  //!
  //! The `ds`, `es`, `fs`, and `gs` fields are valid.
  kMinidumpContextAMD64Segment = kMinidumpContextAMD64 | 0x00000004,

  //! \brief Indicates the validity of floating-point state
  //!     (`CONTEXT_FLOATING_POINT`).
  //!
  //! The `xmm0` through `xmm15` fields are valid.
  kMinidumpContextAMD64FloatingPoint = kMinidumpContextAMD64 | 0x00000008,

  //! \brief Indicates the validity of debug registers
  //!     (`CONTEXT_DEBUG_REGISTERS`).
  //!
  //! The `dr0` through `dr3`, `dr6`, and `dr7` fields are valid.
  kMinidumpContextAMD64Debug = kMinidumpContextAMD64 | 0x00000010,

  //! \brief Indicates the validity of `xsave` data (`CONTEXT_XSTATE`).
  //!
  //! The context contains `xsave` data. This is used with an extended context
  //! structure not currently defined here.
  kMinidumpContextAMD64Xstate = kMinidumpContextAMD64 | 0x00000040,

  //! \brief Indicates the validity of control, integer, and floating-point
  //!     registers (`CONTEXT_FULL`).
  kMinidumpContextAMD64Full = kMinidumpContextAMD64Control |
                              kMinidumpContextAMD64Integer |
                              kMinidumpContextAMD64FloatingPoint,

  //! \brief Indicates the validity of all registers except `xsave` data
  //!     (`CONTEXT_ALL`).
  kMinidumpContextAMD64All = kMinidumpContextAMD64Full |
                             kMinidumpContextAMD64Segment |
                             kMinidumpContextAMD64Debug,
};

//! \brief An x86_64 (AMD64) CPU context (register state) carried in a minidump
//!     file.
//!
//! This is analogous to the `CONTEXT` structure on Windows when targeting
//! x86_64. This structure is used instead of `CONTEXT` to make it available
//! when targeting other architectures.
//!
//! \note This structure doesn’t carry `dr4` or `dr5`, which are obsolete and
//!     normally alias `dr6` and `dr7`, respectively. See Intel Software
//!     Developer’s Manual, Volume 3B: System Programming, Part 2 (253669-052),
//!     17.2.2 “Debug Registers DR4 and DR5”.
struct alignas(16) MinidumpContextAMD64 {
  //! \brief Register parameter home address.
  //!
  //! On Windows, this field may contain the “home” address (on-stack, in the
  //! shadow area) of a parameter passed by register. This field is present for
  //! convenience but is not necessarily populated, even if a corresponding
  //! parameter was passed by register.
  //!
  //! \{
  uint64_t p1_home;
  uint64_t p2_home;
  uint64_t p3_home;
  uint64_t p4_home;
  uint64_t p5_home;
  uint64_t p6_home;
  //! \}

  //! \brief A bitfield composed of values of #MinidumpContextFlags and
  //!     #MinidumpContextAMD64Flags.
  //!
  //! This field identifies the context structure as an x86_64 CPU context, and
  //! indicates which other fields in the structure are valid.
  uint32_t context_flags;

  uint32_t mx_csr;

  uint16_t cs;
  uint16_t ds;
  uint16_t es;
  uint16_t fs;
  uint16_t gs;
  uint16_t ss;

  uint32_t eflags;

  uint64_t dr0;
  uint64_t dr1;
  uint64_t dr2;
  uint64_t dr3;
  uint64_t dr6;
  uint64_t dr7;

  uint64_t rax;
  uint64_t rcx;
  uint64_t rdx;
  uint64_t rbx;
  uint64_t rsp;
  uint64_t rbp;
  uint64_t rsi;
  uint64_t rdi;
  uint64_t r8;
  uint64_t r9;
  uint64_t r10;
  uint64_t r11;
  uint64_t r12;
  uint64_t r13;
  uint64_t r14;
  uint64_t r15;

  uint64_t rip;

  // CPUContextX86_64::Fxsave has identical layout to what the x86_64 CONTEXT
  // structure places here.
  CPUContextX86_64::Fxsave fxsave;

  uint128_struct vector_register[26];
  uint64_t vector_control;

  //! \brief Model-specific debug extension register.
  //!
  //! See Intel Software Developer’s Manual, Volume 3B: System Programming, Part
  //! 2 (253669-051), 17.4 “Last Branch, Interrupt, and Exception Recording
  //! Overview”, and AMD Architecture Programmer’s Manual, Volume 2: System
  //! Programming (24593-3.24), 13.1.6 “Control-Transfer Breakpoint Features”.
  //!
  //! \{
  uint64_t debug_control;
  uint64_t last_branch_to_rip;
  uint64_t last_branch_from_rip;
  uint64_t last_exception_to_rip;
  uint64_t last_exception_from_rip;
  //! \}
};

//! \brief 32-bit ARM-specifc flags for MinidumpContextARM::context_flags.
enum MinidumpContextARMFlags : uint32_t {
  //! \brief Identifies the context structure as 32-bit ARM.
  kMinidumpContextARM = 0x40000000,

  //! \brief Indicates the validity of integer regsiters.
  //!
  //! Regsiters `r0`-`r15` and `cpsr` are valid.
  kMinidumpContextARMInteger = kMinidumpContextARM | 0x00000002,

  //! \brief Inidicates the validity of VFP regsiters.
  //!
  //! Registers `d0`-`d31` and `fpscr` are valid.
  kMinidumpContextARMVFP = kMinidumpContextARM | 0x00000004,

  //! \brief Indicates the validity of all registers.
  kMinidumpContextARMAll = kMinidumpContextARMInteger | kMinidumpContextARMVFP,
};

//! \brief A 32-bit ARM CPU context (register state) carried in a minidump file.
struct MinidumpContextARM {
  //! \brief A bitfield composed of values of #MinidumpContextFlags and
  //!     #MinidumpContextARMFlags.
  //!
  //! This field identifies the context structure as a 32-bit ARM CPU context,
  //! and indicates which other fields in the structure are valid.
  uint32_t context_flags;

  //! \brief General-purpose registers `r0`-`r15`.
  uint32_t regs[11];
  uint32_t fp;  // r11
  uint32_t ip;  // r12
  uint32_t sp;  // r13
  uint32_t lr;  // r14
  uint32_t pc;  // r15

  //! \brief Current program status register.
  uint32_t cpsr;

  //! \brief Floating-point status and control register.
  uint32_t fpscr;

  //! \brief VFP registers `d0`-`d31`.
  uint64_t vfp[32];

  //! \brief This space is unused. It is included for compatibility with
  //!     breakpad (which also doesn't use it).
  uint32_t extra[8];
};

//! \brief 64-bit ARM-specifc flags for MinidumpContextARM64::context_flags.
enum MinidumpContextARM64Flags : uint32_t {
  //! \brief Identifies the context structure as 64-bit ARM.
  kMinidumpContextARM64 = 0x00400000,

  //! \brief Indicates the validity of control registers.
  //!
  //! Registers `fp`, `lr`, `sp`, `pc`, and `cpsr`.
  kMinidumpContextARM64Control = kMinidumpContextARM64 | 0x00000001,

  //! \brief Indicates the validty of integer registers.
  //!
  //! Registers `x0`-`x28`.
  kMinidumpContextARM64Integer = kMinidumpContextARM64 | 0x00000002,

  //! \brief Indicates the validity of fpsimd registers.
  //!
  //! Registers `v0`-`v31`, `fpsr`, and `fpcr` are valid.
  kMinidumpContextARM64Fpsimd = kMinidumpContextARM64 | 0x00000004,

  //! \brief Indicates the validity of debug registers.
  //!
  //! `bcr`, `bvr`, `wcr`, and `wvr` are valid.
  kMinidumpContextARM64Debug = kMinidumpContextARM64 | 0x00000008,

  //! \brief Indicates the validity of control, integer and floating point
  //!     registers.
  kMinidumpContextARM64Full = kMinidumpContextARM64Control |
                              kMinidumpContextARM64Integer |
                              kMinidumpContextARM64Fpsimd,

  //! \brief Indicates the validity of all registers.
  kMinidumpContextARM64All =
      kMinidumpContextARM64Full | kMinidumpContextARM64Debug,
};

//! \brief A 64-bit ARM CPU context (register state) carried in a minidump file.
struct MinidumpContextARM64 {
  uint32_t context_flags;

  //! \brief Current program status register.
  uint32_t cpsr;

  //! \brief General-purpose registers `x0`-`x28`.
  uint64_t regs[29];

  //! \brief Frame pointer or `x29`.
  uint64_t fp;

  //! \brief Link register or `x30`.
  uint64_t lr;

  //! \brief Stack pointer or `x31`.
  uint64_t sp;

  //! \brief Program counter.
  uint64_t pc;

  //! \brief NEON registers `v0`-`v31`.
  uint128_struct fpsimd[32];

  //! \brief Floating-point control register.
  uint32_t fpcr;

  //! \brief Floating-point status register.
  uint32_t fpsr;

  //! \brief Debug registers.
  uint32_t bcr[8];
  uint64_t bvr[8];
  uint32_t wcr[2];
  uint64_t wvr[2];
};

//! \brief 32bit MIPS-specifc flags for MinidumpContextMIPS::context_flags.
//! Based on minidump_cpu_mips.h from breakpad
enum MinidumpContextMIPSFlags : uint32_t {
  //! \brief Identifies the context structure as MIPSEL.
  kMinidumpContextMIPS = 0x00040000,

  //! \brief Indicates the validity of integer registers.
  //!
  //! Registers `0`-`31`, `mdhi`, `mdlo`, `epc`, `badvaddr`, `status` and
  //! `cause` are valid.
  kMinidumpContextMIPSInteger = kMinidumpContextMIPS | 0x00000002,

  //! \brief Indicates the validity of floating point registers.
  //!
  //! Floating point registers `0`-`31`, `fpcsr` and `fir` are valid
  kMinidumpContextMIPSFloatingPoint = kMinidumpContextMIPS | 0x00000004,

  //! \brief Indicates the validity of DSP registers.
  //!
  //! Registers `hi0`-`hi2`, `lo0`-`lo2` and `dsp_control` are valid
  kMinidumpContextMIPSDSP = kMinidumpContextMIPS | 0x00000008,

  //! \brief Indicates the validity of all registers.
  kMinidumpContextMIPSAll = kMinidumpContextMIPSInteger |
                            kMinidumpContextMIPSFloatingPoint |
                            kMinidumpContextMIPSDSP,
};

//! \brief A 32bit MIPS CPU context (register state) carried in a minidump file.
struct MinidumpContextMIPS {
  uint32_t context_flags;

  //! \brief This padding field is included for breakpad compatibility.
  uint32_t _pad0;
  //! \brief General purpose registers `0`-`31`.
  uint64_t regs[32];

  //! \brief Multiply/divide result.
  uint64_t mdhi, mdlo;

  //! \brief DSP registers.
  uint32_t hi[3];
  uint32_t lo[3];
  uint32_t dsp_control;
  //! \brief This padding field is included for breakpad compatibility.
  uint32_t _pad1;

  // \brief cp0 registers.
  uint64_t epc;
  uint64_t badvaddr;
  uint32_t status;
  uint32_t cause;

  //! \brief FPU registers.
  union {
    struct {
      float _fp_fregs;
      uint32_t _fp_pad;
    } fregs[32];
    double dregs[32];
  } fpregs;

  //! \brief FPU status register.
  uint32_t fpcsr;
  //! \brief FPU implementation register.
  uint32_t fir;
};

//! \brief 64bit MIPS-specifc flags for MinidumpContextMIPS64::context_flags.
//! Based on minidump_cpu_mips.h from breakpad
enum MinidumpContextMIPS64Flags : uint32_t {
  //! \brief Identifies the context structure as MIPS64EL.
  kMinidumpContextMIPS64 = 0x00080000,

  //! \brief Indicates the validity of integer registers.
  //!
  //! Registers `0`-`31`, `mdhi`, `mdlo`, `epc`, `badvaddr`, `status` and
  //! `cause` are valid.
  kMinidumpContextMIPS64Integer = kMinidumpContextMIPS64 | 0x00000002,

  //! \brief Indicates the validity of floating point registers.
  //!
  //! Floating point registers `0`-`31`, `fpcsr` and `fir` are valid
  kMinidumpContextMIPS64FloatingPoint = kMinidumpContextMIPS64 | 0x00000004,

  //! \brief Indicates the validity of DSP registers.
  //!
  //! Registers `hi0`-`hi2`, `lo0`-`lo2` and `dsp_control` are valid.
  kMinidumpContextMIPS64DSP = kMinidumpContextMIPS64 | 0x00000008,

  //! \brief Indicates the validity of all registers.
  kMinidumpContextMIPS64All = kMinidumpContextMIPS64Integer |
                              kMinidumpContextMIPS64FloatingPoint |
                              kMinidumpContextMIPS64DSP,
};

//! \brief A 32bit MIPS CPU context (register state) carried in a minidump file.
struct MinidumpContextMIPS64 {
  uint64_t context_flags;

  //! \brief General purpose registers.
  uint64_t regs[32];

  //! \brief Multiply/divide result.
  uint64_t mdhi, mdlo;

  //! \brief DSP registers.
  uint64_t hi[3];
  uint64_t lo[3];
  uint64_t dsp_control;

  //! \brief cp0 registers.
  uint64_t epc;
  uint64_t badvaddr;
  uint64_t status;
  uint64_t cause;

  //! \brief FPU registers.
  union {
    struct {
      float _fp_fregs;
      uint32_t _fp_pad;
    } fregs[32];
    double dregs[32];
  } fpregs;

  //! \brief FPU status register.
  uint64_t fpcsr;
  //! \brief FPU implementation register.
  uint64_t fir;
};

}  // namespace crashpad

#endif  // CRASHPAD_MINIDUMP_MINIDUMP_CONTEXT_H_
