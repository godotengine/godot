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

#include "snapshot/cpu_context.h"

#include <stddef.h>
#include <string.h>

#include "base/logging.h"
#include "base/macros.h"
#include "util/misc/implicit_cast.h"

namespace crashpad {

namespace {

// Sanity-check complex structures to ensure interoperability.
static_assert(sizeof(CPUContextX86::Fsave) == 108, "CPUContextX86::Fsave size");
static_assert(sizeof(CPUContextX86::Fxsave) == 512,
              "CPUContextX86::Fxsave size");
static_assert(sizeof(CPUContextX86_64::Fxsave) == 512,
              "CPUContextX86_64::Fxsave size");

enum {
  kX87TagValid = 0,
  kX87TagZero,
  kX87TagSpecial,
  kX87TagEmpty,
};

}  // namespace

// static
void CPUContextX86::FxsaveToFsave(const Fxsave& fxsave, Fsave* fsave) {
  fsave->fcw = fxsave.fcw;
  fsave->reserved_1 = 0;
  fsave->fsw = fxsave.fsw;
  fsave->reserved_2 = 0;
  fsave->ftw = FxsaveToFsaveTagWord(fxsave.fsw, fxsave.ftw, fxsave.st_mm);
  fsave->reserved_3 = 0;
  fsave->fpu_ip = fxsave.fpu_ip;
  fsave->fpu_cs = fxsave.fpu_cs;
  fsave->fop = fxsave.fop;
  fsave->fpu_dp = fxsave.fpu_dp;
  fsave->fpu_ds = fxsave.fpu_ds;
  fsave->reserved_4 = 0;
  static_assert(arraysize(fsave->st) == arraysize(fxsave.st_mm),
                "FPU stack registers must be equivalent");
  for (size_t index = 0; index < arraysize(fsave->st); ++index) {
    memcpy(fsave->st[index], fxsave.st_mm[index].st, sizeof(fsave->st[index]));
  }
}

// static
void CPUContextX86::FsaveToFxsave(const Fsave& fsave, Fxsave* fxsave) {
  fxsave->fcw = fsave.fcw;
  fxsave->fsw = fsave.fsw;
  fxsave->ftw = FsaveToFxsaveTagWord(fsave.ftw);
  fxsave->reserved_1 = 0;
  fxsave->fop = fsave.fop;
  fxsave->fpu_ip = fsave.fpu_ip;
  fxsave->fpu_cs = fsave.fpu_cs;
  fxsave->reserved_2 = 0;
  fxsave->fpu_dp = fsave.fpu_dp;
  fxsave->fpu_ds = fsave.fpu_ds;
  fxsave->reserved_3 = 0;
  fxsave->mxcsr = 0;
  fxsave->mxcsr_mask = 0;
  static_assert(arraysize(fxsave->st_mm) == arraysize(fsave.st),
                "FPU stack registers must be equivalent");
  for (size_t index = 0; index < arraysize(fsave.st); ++index) {
    memcpy(fxsave->st_mm[index].st, fsave.st[index], sizeof(fsave.st[index]));
    memset(fxsave->st_mm[index].st_reserved,
           0,
           sizeof(fxsave->st_mm[index].st_reserved));
  }
  memset(fxsave->xmm, 0, sizeof(*fxsave) - offsetof(Fxsave, xmm));
}

// static
uint16_t CPUContextX86::FxsaveToFsaveTagWord(
    uint16_t fsw,
    uint8_t fxsave_tag,
    const CPUContextX86::X87OrMMXRegister st_mm[8]) {
  // The x87 tag word (in both abridged and full form) identifies physical
  // registers, but |st_mm| is arranged in logical stack order. In order to map
  // physical tag word bits to the logical stack registers they correspond to,
  // the “stack top” value from the x87 status word is necessary.
  int stack_top = (fsw >> 11) & 0x7;

  uint16_t fsave_tag = 0;
  for (int physical_index = 0; physical_index < 8; ++physical_index) {
    bool fxsave_bit = (fxsave_tag & (1 << physical_index)) != 0;
    uint8_t fsave_bits;

    if (fxsave_bit) {
      int st_index = (physical_index + 8 - stack_top) % 8;
      const CPUContextX86::X87Register& st = st_mm[st_index].st;

      uint32_t exponent = ((st[9] & 0x7f) << 8) | st[8];
      if (exponent == 0x7fff) {
        // Infinity, NaN, pseudo-infinity, or pseudo-NaN. If it was important to
        // distinguish between these, the J bit and the M bit (the most
        // significant bit of |fraction|) could be consulted.
        fsave_bits = kX87TagSpecial;
      } else {
        // The integer bit the “J bit”.
        bool integer_bit = (st[7] & 0x80) != 0;
        if (exponent == 0) {
          uint64_t fraction = ((implicit_cast<uint64_t>(st[7]) & 0x7f) << 56) |
                              (implicit_cast<uint64_t>(st[6]) << 48) |
                              (implicit_cast<uint64_t>(st[5]) << 40) |
                              (implicit_cast<uint64_t>(st[4]) << 32) |
                              (implicit_cast<uint32_t>(st[3]) << 24) |
                              (st[2] << 16) | (st[1] << 8) | st[0];
          if (!integer_bit && fraction == 0) {
            fsave_bits = kX87TagZero;
          } else {
            // Denormal (if the J bit is clear) or pseudo-denormal.
            fsave_bits = kX87TagSpecial;
          }
        } else if (integer_bit) {
          fsave_bits = kX87TagValid;
        } else {
          // Unnormal.
          fsave_bits = kX87TagSpecial;
        }
      }
    } else {
      fsave_bits = kX87TagEmpty;
    }

    fsave_tag |= (fsave_bits << (physical_index * 2));
  }

  return fsave_tag;
}

// static
uint8_t CPUContextX86::FsaveToFxsaveTagWord(uint16_t fsave_tag) {
  uint8_t fxsave_tag = 0;
  for (int physical_index = 0; physical_index < 8; ++physical_index) {
    const uint8_t fsave_bits = (fsave_tag >> (physical_index * 2)) & 0x3;
    const bool fxsave_bit = fsave_bits != kX87TagEmpty;
    fxsave_tag |= fxsave_bit << physical_index;
  }
  return fxsave_tag;
}

uint64_t CPUContext::InstructionPointer() const {
  switch (architecture) {
    case kCPUArchitectureX86:
      return x86->eip;
    case kCPUArchitectureX86_64:
      return x86_64->rip;
    case kCPUArchitectureARM:
      return arm->pc;
    case kCPUArchitectureARM64:
      return arm64->pc;
    default:
      NOTREACHED();
      return ~0ull;
  }
}

uint64_t CPUContext::StackPointer() const {
  switch (architecture) {
    case kCPUArchitectureX86:
      return x86->esp;
    case kCPUArchitectureX86_64:
      return x86_64->rsp;
    case kCPUArchitectureARM:
      return arm->sp;
    case kCPUArchitectureARM64:
      return arm64->sp;
    default:
      NOTREACHED();
      return ~0ull;
  }
}

bool CPUContext::Is64Bit() const {
  switch (architecture) {
    case kCPUArchitectureX86_64:
    case kCPUArchitectureARM64:
    case kCPUArchitectureMIPS64EL:
      return true;
    case kCPUArchitectureX86:
    case kCPUArchitectureARM:
    case kCPUArchitectureMIPSEL:
      return false;
    default:
      NOTREACHED();
      return false;
  }
}

}  // namespace crashpad
