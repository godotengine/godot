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

#ifndef CRASHPAD_SNAPSHOT_SNAPSHOT_CPU_CONTEXT_H_
#define CRASHPAD_SNAPSHOT_SNAPSHOT_CPU_CONTEXT_H_

#include <stdint.h>

#include "snapshot/cpu_architecture.h"
#include "util/numeric/int128.h"

namespace crashpad {

//! \brief A context structure carrying 32-bit x86 CPU state.
struct CPUContextX86 {
  using X87Register = uint8_t[10];

  struct Fsave {
    uint16_t fcw;  // FPU control word
    uint16_t reserved_1;
    uint16_t fsw;  // FPU status word
    uint16_t reserved_2;
    uint16_t ftw;  // full FPU tag word
    uint16_t reserved_3;
    uint32_t fpu_ip;  // FPU instruction pointer offset
    uint16_t fpu_cs;  // FPU instruction pointer segment selector
    uint16_t fop;  // FPU opcode
    uint32_t fpu_dp;  // FPU data pointer offset
    uint16_t fpu_ds;  // FPU data pointer segment selector
    uint16_t reserved_4;
    X87Register st[8];
  };

  union X87OrMMXRegister {
    struct {
      X87Register st;
      uint8_t st_reserved[6];
    };
    struct {
      uint8_t mm_value[8];
      uint8_t mm_reserved[8];
    };
  };

  using XMMRegister = uint8_t[16];

  struct Fxsave {
    uint16_t fcw;  // FPU control word
    uint16_t fsw;  // FPU status word
    uint8_t ftw;  // abridged FPU tag word
    uint8_t reserved_1;
    uint16_t fop;  // FPU opcode
    uint32_t fpu_ip;  // FPU instruction pointer offset
    uint16_t fpu_cs;  // FPU instruction pointer segment selector
    uint16_t reserved_2;
    uint32_t fpu_dp;  // FPU data pointer offset
    uint16_t fpu_ds;  // FPU data pointer segment selector
    uint16_t reserved_3;
    uint32_t mxcsr;  // multimedia extensions status and control register
    uint32_t mxcsr_mask;  // valid bits in mxcsr
    X87OrMMXRegister st_mm[8];
    XMMRegister xmm[8];
    uint8_t reserved_4[176];
    uint8_t available[48];
  };

  //! \brief Converts an `fxsave` area to an `fsave` area.
  //!
  //! `fsave` state is restricted to the x87 FPU, while `fxsave` state includes
  //! state related to the x87 FPU as well as state specific to SSE.
  //!
  //! As the `fxsave` format is a superset of the `fsave` format, this operation
  //! fully populates the `fsave` area. `fsave` uses the full 16-bit form for
  //! the x87 floating-point tag word, so FxsaveToFsaveTagWord() is used to
  //! derive Fsave::ftw from the abridged 8-bit form used by `fxsave`. Reserved
  //! fields in \a fsave are set to `0`.
  //!
  //! \param[in] fxsave The `fxsave` area to convert.
  //! \param[out] fsave The `fsave` area to populate.
  //!
  //! \sa FsaveToFxsave()
  static void FxsaveToFsave(const Fxsave& fxsave, Fsave* fsave);

  //! \brief Converts an `fsave` area to an `fxsave` area.
  //!
  //! `fsave` state is restricted to the x87 FPU, while `fxsave` state includes
  //! state related to the x87 FPU as well as state specific to SSE.
  //!
  //! As the `fsave` format is a subset of the `fxsave` format, this operation
  //! cannot fully populate the `fxsave` area. Fields in \a fxsave that have no
  //! equivalent in \a fsave are set to `0`, including Fxsave::mxcsr,
  //! Fxsave::mxcsr_mask, Fxsave::xmm, and Fxsave::available.
  //! FsaveToFxsaveTagWord() is used to derive Fxsave::ftw from the full 16-bit
  //! form used by `fsave`. Reserved fields in \a fxsave are set to `0`.
  //!
  //! \param[in] fsave The `fsave` area to convert.
  //! \param[out] fxsave The `fxsave` area to populate.
  //!
  //! \sa FxsaveToFsave()
  static void FsaveToFxsave(const Fsave& fsave, Fxsave* fxsave);

  //! \brief Converts x87 floating-point tag words from `fxsave` (abridged,
  //!     8-bit) to `fsave` (full, 16-bit) form.
  //!
  //! `fxsave` stores the x87 floating-point tag word in abridged 8-bit form,
  //! and `fsave` stores it in full 16-bit form. Some users, notably
  //! CPUContextX86::Fsave::ftw, require the full 16-bit form, where most other
  //! contemporary code uses `fxsave` and thus the abridged 8-bit form found in
  //! CPUContextX86::Fxsave::ftw.
  //!
  //! This function converts an abridged tag word to the full version by using
  //! the abridged tag word and the contents of the registers it describes. See
  //! Intel Software Developer’s Manual, Volume 2A: Instruction Set Reference
  //! A-M (253666-052), 3.2 “FXSAVE”, specifically, the notes on the abridged
  //! FTW and recreating the FSAVE format, and AMD Architecture Programmer’s
  //! Manual, Volume 2: System Programming (24593-3.24), “FXSAVE Format for x87
  //! Tag Word”.
  //!
  //! \sa FsaveToFxsaveTagWord()
  //!
  //! \param[in] fsw The FPU status word, used to map logical \a st_mm registers
  //!     to their physical counterparts. This can be taken from
  //!     CPUContextX86::Fxsave::fsw.
  //! \param[in] fxsave_tag The abridged FPU tag word. This can be taken from
  //!     CPUContextX86::Fxsave::ftw.
  //! \param[in] st_mm The floating-point registers in logical order. This can
  //!     be taken from CPUContextX86::Fxsave::st_mm.
  //!
  //! \return The full FPU tag word.
  static uint16_t FxsaveToFsaveTagWord(
      uint16_t fsw, uint8_t fxsave_tag, const X87OrMMXRegister st_mm[8]);

  //! \brief Converts x87 floating-point tag words from `fsave` (full, 16-bit)
  //!     to `fxsave` (abridged, 8-bit) form.
  //!
  //! This function performs the inverse operation of FxsaveToFsaveTagWord().
  //!
  //! \param[in] fsave_tag The full FPU tag word.
  //!
  //! \return The abridged FPU tag word.
  static uint8_t FsaveToFxsaveTagWord(uint16_t fsave_tag);

  // Integer registers.
  uint32_t eax;
  uint32_t ebx;
  uint32_t ecx;
  uint32_t edx;
  uint32_t edi;  // destination index
  uint32_t esi;  // source index
  uint32_t ebp;  // base pointer
  uint32_t esp;  // stack pointer
  uint32_t eip;  // instruction pointer
  uint32_t eflags;
  uint16_t cs;  // code segment selector
  uint16_t ds;  // data segment selector
  uint16_t es;  // extra segment selector
  uint16_t fs;
  uint16_t gs;
  uint16_t ss;  // stack segment selector

  // Floating-point and vector registers.
  Fxsave fxsave;

  // Debug registers.
  uint32_t dr0;
  uint32_t dr1;
  uint32_t dr2;
  uint32_t dr3;
  uint32_t dr4;  // obsolete, normally an alias for dr6
  uint32_t dr5;  // obsolete, normally an alias for dr7
  uint32_t dr6;
  uint32_t dr7;
};

//! \brief A context structure carrying x86_64 CPU state.
struct CPUContextX86_64 {
  using X87Register = CPUContextX86::X87Register;
  using X87OrMMXRegister = CPUContextX86::X87OrMMXRegister;
  using XMMRegister = CPUContextX86::XMMRegister;

  struct Fxsave {
    uint16_t fcw;  // FPU control word
    uint16_t fsw;  // FPU status word
    uint8_t ftw;  // abridged FPU tag word
    uint8_t reserved_1;
    uint16_t fop;  // FPU opcode
    union {
      // The expression of these union members is determined by the use of
      // fxsave/fxrstor or fxsave64/fxrstor64 (fxsaveq/fxrstorq). macOS and
      // Windows systems use the traditional fxsave/fxrstor structure.
      struct {
        // fxsave/fxrstor
        uint32_t fpu_ip;  // FPU instruction pointer offset
        uint16_t fpu_cs;  // FPU instruction pointer segment selector
        uint16_t reserved_2;
        uint32_t fpu_dp;  // FPU data pointer offset
        uint16_t fpu_ds;  // FPU data pointer segment selector
        uint16_t reserved_3;
      };
      struct {
        // fxsave64/fxrstor64 (fxsaveq/fxrstorq)
        uint64_t fpu_ip_64;  // FPU instruction pointer
        uint64_t fpu_dp_64;  // FPU data pointer
      };
    };
    uint32_t mxcsr;  // multimedia extensions status and control register
    uint32_t mxcsr_mask;  // valid bits in mxcsr
    X87OrMMXRegister st_mm[8];
    XMMRegister xmm[16];
    uint8_t reserved_4[48];
    uint8_t available[48];
  };

  // Integer registers.
  uint64_t rax;
  uint64_t rbx;
  uint64_t rcx;
  uint64_t rdx;
  uint64_t rdi;  // destination index
  uint64_t rsi;  // source index
  uint64_t rbp;  // base pointer
  uint64_t rsp;  // stack pointer
  uint64_t r8;
  uint64_t r9;
  uint64_t r10;
  uint64_t r11;
  uint64_t r12;
  uint64_t r13;
  uint64_t r14;
  uint64_t r15;
  uint64_t rip;  // instruction pointer
  uint64_t rflags;
  uint16_t cs;  // code segment selector
  uint16_t fs;
  uint16_t gs;

  // Floating-point and vector registers.
  Fxsave fxsave;

  // Debug registers.
  uint64_t dr0;
  uint64_t dr1;
  uint64_t dr2;
  uint64_t dr3;
  uint64_t dr4;  // obsolete, normally an alias for dr6
  uint64_t dr5;  // obsolete, normally an alias for dr7
  uint64_t dr6;
  uint64_t dr7;
};

//! \brief A context structure carrying ARM CPU state.
struct CPUContextARM {
  uint32_t regs[11];
  uint32_t fp;  // r11
  uint32_t ip;  // r12
  uint32_t sp;  // r13
  uint32_t lr;  // r14
  uint32_t pc;  // r15
  uint32_t cpsr;

  struct {
    struct fp_reg {
      uint32_t sign1 : 1;
      uint32_t unused : 15;
      uint32_t sign2 : 1;
      uint32_t exponent : 14;
      uint32_t j : 1;
      uint32_t mantissa1 : 31;
      uint32_t mantisss0 : 32;
    } fpregs[8];
    uint32_t fpsr : 32;
    uint32_t fpcr : 32;
    uint8_t type[8];
    uint32_t init_flag;
  } fpa_regs;

  struct {
    uint64_t vfp[32];
    uint32_t fpscr;
  } vfp_regs;

  bool have_fpa_regs;
  bool have_vfp_regs;
};

//! \brief A context structure carrying ARM64 CPU state.
struct CPUContextARM64 {
  uint64_t regs[31];
  uint64_t sp;
  uint64_t pc;
  uint64_t pstate;

  uint128_struct fpsimd[32];
  uint32_t fpsr;
  uint32_t fpcr;
};

//! \brief A context structure carrying MIPS CPU state.
struct CPUContextMIPS {
  uint64_t regs[32];
  uint32_t mdlo;
  uint32_t mdhi;
  uint32_t cp0_epc;
  uint32_t cp0_badvaddr;
  uint32_t cp0_status;
  uint32_t cp0_cause;
  uint32_t hi[3];
  uint32_t lo[3];
  uint32_t dsp_control;
  union {
    double dregs[32];
    struct {
      float _fp_fregs;
      uint32_t _fp_pad;
    } fregs[32];
  } fpregs;
  uint32_t fpcsr;
  uint32_t fir;
};

//! \brief A context structure carrying MIPS64 CPU state.
struct CPUContextMIPS64 {
  uint64_t regs[32];
  uint64_t mdlo;
  uint64_t mdhi;
  uint64_t cp0_epc;
  uint64_t cp0_badvaddr;
  uint64_t cp0_status;
  uint64_t cp0_cause;
  uint64_t hi[3];
  uint64_t lo[3];
  uint64_t dsp_control;
  union {
    double dregs[32];
    struct {
      float _fp_fregs;
      uint32_t _fp_pad;
    } fregs[32];
  } fpregs;
  uint64_t fpcsr;
  uint64_t fir;
};

//! \brief A context structure capable of carrying the context of any supported
//!     CPU architecture.
struct CPUContext {
  //! \brief Returns the instruction pointer value from the context structure.
  //!
  //! This is a CPU architecture-independent method that is capable of
  //! recovering the instruction pointer from any supported CPU architecture’s
  //! context structure.
  uint64_t InstructionPointer() const;

  //! \brief Returns the stack pointer value from the context structure.
  //!
  //! This is a CPU architecture-independent method that is capable of
  //! recovering the stack pointer from any supported CPU architecture’s
  //! context structure.
  uint64_t StackPointer() const;

  //! \brief Returns `true` if this context is for a 64-bit architecture.
  bool Is64Bit() const;

  //! \brief The CPU architecture of a context structure. This field controls
  //!     the expression of the union.
  CPUArchitecture architecture;
  union {
    CPUContextX86* x86;
    CPUContextX86_64* x86_64;
    CPUContextARM* arm;
    CPUContextARM64* arm64;
    CPUContextMIPS* mipsel;
    CPUContextMIPS64* mips64;
  };
};

}  // namespace crashpad

#endif  // CRASHPAD_SNAPSHOT_SNAPSHOT_CPU_CONTEXT_H_
