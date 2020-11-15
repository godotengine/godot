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
#include <sys/types.h>

#include "base/macros.h"
#include "gtest/gtest.h"
#include "test/hex_string.h"

namespace crashpad {
namespace test {
namespace {

enum ExponentValue {
  kExponentAllZero = 0,
  kExponentAllOne,
  kExponentNormal,
};

enum FractionValue {
  kFractionAllZero = 0,
  kFractionNormal,
};

//! \brief Initializes an x87 register to a known bit pattern.
//!
//! \param[out] st_mm The x87 register to initialize. The reserved portion of
//!     the register is always zeroed out.
//! \param[in] exponent_value The bit pattern to use for the exponent. If this
//!     is kExponentAllZero, the sign bit will be set to `1`, and if this is
//!     kExponentAllOne, the sign bit will be set to `0`. This tests that the
//!     implementation doesn’t erroneously consider the sign bit to be part of
//!     the exponent. This may also be kExponentNormal, indicating that the
//!     exponent shall neither be all zeroes nor all ones.
//! \param[in] j_bit The value to use for the “J bit” (“integer bit”).
//! \param[in] fraction_value If kFractionAllZero, the fraction will be zeroed
//!     out. If kFractionNormal, the fraction will not be all zeroes.
void SetX87Register(CPUContextX86::X87Register* st,
                    ExponentValue exponent_value,
                    bool j_bit,
                    FractionValue fraction_value) {
  switch (exponent_value) {
    case kExponentAllZero:
      (*st)[9] = 0x80;
      (*st)[8] = 0;
      break;
    case kExponentAllOne:
      (*st)[9] = 0x7f;
      (*st)[8] = 0xff;
      break;
    case kExponentNormal:
      (*st)[9] = 0x55;
      (*st)[8] = 0x55;
      break;
  }

  uint8_t fraction_pattern = fraction_value == kFractionAllZero ? 0 : 0x55;
  memset(st, fraction_pattern, 8);

  if (j_bit) {
    (*st)[7] |= 0x80;
  } else {
    (*st)[7] &= ~0x80;
  }
}

//! \brief Initializes an x87 register to a known bit pattern.
//!
//! This behaves as SetX87Register() but also clears the reserved portion of the
//! field as used in the `fxsave` format.
void SetX87OrMMXRegister(CPUContextX86::X87OrMMXRegister* st_mm,
                         ExponentValue exponent_value,
                         bool j_bit,
                         FractionValue fraction_value) {
  SetX87Register(&st_mm->st, exponent_value, j_bit, fraction_value);
  memset(st_mm->st_reserved, 0, sizeof(st_mm->st_reserved));
}

TEST(CPUContextX86, FxsaveToFsave) {
  // Establish a somewhat plausible fxsave state. Use nonzero values for
  // reserved fields and things that aren’t present in fsave.
  CPUContextX86::Fxsave fxsave;
  fxsave.fcw = 0x027f;  // mask exceptions, 53-bit precision, round to nearest
  fxsave.fsw = 1 << 11;  // top = 1: logical 0-7 maps to physical 1-7, 0
  fxsave.ftw = 0x1f;  // physical 5-7 (logical 4-6) empty
  fxsave.reserved_1 = 0x5a;
  fxsave.fop = 0x1fe;  // fsin
  fxsave.fpu_ip = 0x76543210;
  fxsave.fpu_cs = 0x0007;
  fxsave.reserved_2 = 0x5a5a;
  fxsave.fpu_dp = 0xfedcba98;
  fxsave.fpu_ds = 0x000f;
  fxsave.reserved_3 = 0x5a5a;
  fxsave.mxcsr = 0x1f80;
  fxsave.mxcsr_mask = 0xffff;
  SetX87Register(
      &fxsave.st_mm[0].st, kExponentNormal, true, kFractionAllZero);  // valid
  SetX87Register(
      &fxsave.st_mm[1].st, kExponentAllZero, false, kFractionAllZero);  // zero
  SetX87Register(
      &fxsave.st_mm[2].st, kExponentAllOne, true, kFractionAllZero);  // spec.
  SetX87Register(
      &fxsave.st_mm[3].st, kExponentAllOne, true, kFractionNormal);  // spec.
  SetX87Register(
      &fxsave.st_mm[4].st, kExponentAllZero, false, kFractionAllZero);
  SetX87Register(
      &fxsave.st_mm[5].st, kExponentAllZero, false, kFractionAllZero);
  SetX87Register(
      &fxsave.st_mm[6].st, kExponentAllZero, false, kFractionAllZero);
  SetX87Register(
      &fxsave.st_mm[7].st, kExponentNormal, true, kFractionNormal);  // valid
  for (size_t index = 0; index < arraysize(fxsave.st_mm); ++index) {
    memset(&fxsave.st_mm[index].st_reserved,
           0x5a,
           sizeof(fxsave.st_mm[index].st_reserved));
  }
  memset(&fxsave.xmm, 0x5a, sizeof(fxsave) - offsetof(decltype(fxsave), xmm));

  CPUContextX86::Fsave fsave;
  CPUContextX86::FxsaveToFsave(fxsave, &fsave);

  // Everything should have come over from fxsave. Reserved fields should be
  // zero.
  EXPECT_EQ(fsave.fcw, fxsave.fcw);
  EXPECT_EQ(fsave.reserved_1, 0);
  EXPECT_EQ(fsave.fsw, fxsave.fsw);
  EXPECT_EQ(fsave.reserved_2, 0);
  EXPECT_EQ(fsave.ftw, 0xfe90);  // FxsaveToFsaveTagWord
  EXPECT_EQ(fsave.reserved_3, 0);
  EXPECT_EQ(fsave.fpu_ip, fxsave.fpu_ip);
  EXPECT_EQ(fsave.fpu_cs, fxsave.fpu_cs);
  EXPECT_EQ(fsave.fop, fxsave.fop);
  EXPECT_EQ(fsave.fpu_dp, fxsave.fpu_dp);
  EXPECT_EQ(fsave.fpu_ds, fxsave.fpu_ds);
  EXPECT_EQ(fsave.reserved_4, 0);
  for (size_t index = 0; index < arraysize(fsave.st); ++index) {
    EXPECT_EQ(BytesToHexString(fsave.st[index], arraysize(fsave.st[index])),
              BytesToHexString(fxsave.st_mm[index].st,
                               arraysize(fxsave.st_mm[index].st)))
        << "index " << index;
  }
}

TEST(CPUContextX86, FsaveToFxsave) {
  // Establish a somewhat plausible fsave state. Use nonzero values for
  // reserved fields.
  CPUContextX86::Fsave fsave;
  fsave.fcw = 0x0300;  // unmask exceptions, 64-bit precision, round to nearest
  fsave.reserved_1 = 0xa5a5;
  fsave.fsw = 2 << 11;  // top = 2: logical 0-7 maps to physical 2-7, 0-1
  fsave.reserved_2 = 0xa5a5;
  fsave.ftw = 0xa9ff;  // physical 0-3 (logical 6-7, 0-1) empty; physical 4
                       // (logical 2) zero; physical 5-7 (logical 3-5) special
  fsave.reserved_3 = 0xa5a5;
  fsave.fpu_ip = 0x456789ab;
  fsave.fpu_cs = 0x1013;
  fsave.fop = 0x01ee;  // fldz
  fsave.fpu_dp = 0x0123cdef;
  fsave.fpu_ds = 0x2017;
  fsave.reserved_4 = 0xa5a5;
  SetX87Register(&fsave.st[0], kExponentAllZero, false, kFractionNormal);
  SetX87Register(&fsave.st[1], kExponentAllZero, true, kFractionNormal);
  SetX87Register(
      &fsave.st[2], kExponentAllZero, false, kFractionAllZero);  // zero
  SetX87Register(
      &fsave.st[3], kExponentAllZero, true, kFractionAllZero);  // spec.
  SetX87Register(
      &fsave.st[4], kExponentAllZero, false, kFractionNormal);  // spec.
  SetX87Register(
      &fsave.st[5], kExponentAllZero, true, kFractionNormal);  // spec.
  SetX87Register(&fsave.st[6], kExponentAllZero, false, kFractionAllZero);
  SetX87Register(&fsave.st[7], kExponentAllZero, true, kFractionAllZero);

  CPUContextX86::Fxsave fxsave;
  CPUContextX86::FsaveToFxsave(fsave, &fxsave);

  // Everything in fsave should have come over from there. Fields not present in
  // fsave and reserved fields should be zero.
  EXPECT_EQ(fxsave.fcw, fsave.fcw);
  EXPECT_EQ(fxsave.fsw, fsave.fsw);
  EXPECT_EQ(fxsave.ftw, 0xf0);  // FsaveToFxsaveTagWord
  EXPECT_EQ(fxsave.reserved_1, 0);
  EXPECT_EQ(fxsave.fop, fsave.fop);
  EXPECT_EQ(fxsave.fpu_ip, fsave.fpu_ip);
  EXPECT_EQ(fxsave.fpu_cs, fsave.fpu_cs);
  EXPECT_EQ(fxsave.reserved_2, 0);
  EXPECT_EQ(fxsave.fpu_dp, fsave.fpu_dp);
  EXPECT_EQ(fxsave.fpu_ds, fsave.fpu_ds);
  EXPECT_EQ(fxsave.reserved_3, 0);
  EXPECT_EQ(fxsave.mxcsr, 0u);
  EXPECT_EQ(fxsave.mxcsr_mask, 0u);
  for (size_t index = 0; index < arraysize(fxsave.st_mm); ++index) {
    EXPECT_EQ(BytesToHexString(fxsave.st_mm[index].st,
                               arraysize(fxsave.st_mm[index].st)),
              BytesToHexString(fsave.st[index], arraysize(fsave.st[index])))
        << "index " << index;
    EXPECT_EQ(BytesToHexString(fxsave.st_mm[index].st_reserved,
                               arraysize(fxsave.st_mm[index].st_reserved)),
              std::string(arraysize(fxsave.st_mm[index].st_reserved) * 2, '0'))
        << "index " << index;
  }
  size_t unused_len = sizeof(fxsave) - offsetof(decltype(fxsave), xmm);
  EXPECT_EQ(BytesToHexString(fxsave.xmm, unused_len),
            std::string(unused_len * 2, '0'));

  // Since the fsave format is a subset of the fxsave format, fsave-fxsave-fsave
  // should round-trip cleanly.
  CPUContextX86::Fsave fsave_2;
  CPUContextX86::FxsaveToFsave(fxsave, &fsave_2);

  // Clear the reserved fields in the original fsave structure, since they’re
  // expected to be clear in the copy.
  fsave.reserved_1 = 0;
  fsave.reserved_2 = 0;
  fsave.reserved_3 = 0;
  fsave.reserved_4 = 0;
  EXPECT_EQ(memcmp(&fsave, &fsave_2, sizeof(fsave)), 0);
}

TEST(CPUContextX86, FxsaveToFsaveTagWord) {
  // The fsave tag word uses bit pattern 00 for valid, 01 for zero, 10 for
  // “special”, and 11 for empty. Like the fxsave tag word, it is arranged by
  // physical register. The fxsave tag word determines whether a register is
  // empty, and analysis of the x87 register content distinguishes between
  // valid, zero, and special. In the initializations below, comments show
  // whether a register is expected to be considered valid, zero, or special,
  // except where the tag word is expected to indicate that it is empty. Each
  // combination appears twice: once where the fxsave tag word indicates a
  // nonempty register, and once again where it indicates an empty register.

  uint16_t fsw = 0 << 11;  // top = 0: logical 0-7 maps to physical 0-7
  uint8_t fxsave_tag = 0x0f;  // physical 4-7 (logical 4-7) empty
  CPUContextX86::X87OrMMXRegister st_mm[8];
  SetX87OrMMXRegister(
      &st_mm[0], kExponentNormal, false, kFractionNormal);  // spec.
  SetX87OrMMXRegister(
      &st_mm[1], kExponentNormal, true, kFractionNormal);  // valid
  SetX87OrMMXRegister(
      &st_mm[2], kExponentNormal, false, kFractionAllZero);  // spec.
  SetX87OrMMXRegister(
      &st_mm[3], kExponentNormal, true, kFractionAllZero);  // valid
  SetX87OrMMXRegister(&st_mm[4], kExponentNormal, false, kFractionNormal);
  SetX87OrMMXRegister(&st_mm[5], kExponentNormal, true, kFractionNormal);
  SetX87OrMMXRegister(&st_mm[6], kExponentNormal, false, kFractionAllZero);
  SetX87OrMMXRegister(&st_mm[7], kExponentNormal, true, kFractionAllZero);
  EXPECT_EQ(CPUContextX86::FxsaveToFsaveTagWord(fsw, fxsave_tag, st_mm),
            0xff22);

  fsw = 2 << 11;  // top = 2: logical 0-7 maps to physical 2-7, 0-1
  fxsave_tag = 0xf0;  // physical 0-3 (logical 6-7, 0-1) empty
  SetX87OrMMXRegister(&st_mm[0], kExponentAllZero, false, kFractionNormal);
  SetX87OrMMXRegister(&st_mm[1], kExponentAllZero, true, kFractionNormal);
  SetX87OrMMXRegister(
      &st_mm[2], kExponentAllZero, false, kFractionAllZero);  // zero
  SetX87OrMMXRegister(
      &st_mm[3], kExponentAllZero, true, kFractionAllZero);  // spec.
  SetX87OrMMXRegister(
      &st_mm[4], kExponentAllZero, false, kFractionNormal);  // spec.
  SetX87OrMMXRegister(
      &st_mm[5], kExponentAllZero, true, kFractionNormal);  // spec.
  SetX87OrMMXRegister(&st_mm[6], kExponentAllZero, false, kFractionAllZero);
  SetX87OrMMXRegister(&st_mm[7], kExponentAllZero, true, kFractionAllZero);
  EXPECT_EQ(CPUContextX86::FxsaveToFsaveTagWord(fsw, fxsave_tag, st_mm),
            0xa9ff);

  fsw = 5 << 11;  // top = 5: logical 0-7 maps to physical 5-7, 0-4
  fxsave_tag = 0x5a;  // physical 0, 2, 5, and 7 (logical 5, 0, 2, and 3) empty
  SetX87OrMMXRegister(&st_mm[0], kExponentAllOne, false, kFractionNormal);
  SetX87OrMMXRegister(
      &st_mm[1], kExponentAllOne, true, kFractionNormal);  // spec.
  SetX87OrMMXRegister(&st_mm[2], kExponentAllOne, false, kFractionAllZero);
  SetX87OrMMXRegister(&st_mm[3], kExponentAllOne, true, kFractionAllZero);
  SetX87OrMMXRegister(
      &st_mm[4], kExponentAllOne, false, kFractionNormal);  // spec.
  SetX87OrMMXRegister(&st_mm[5], kExponentAllOne, true, kFractionNormal);
  SetX87OrMMXRegister(
      &st_mm[6], kExponentAllOne, false, kFractionAllZero);  // spec.
  SetX87OrMMXRegister(
      &st_mm[7], kExponentAllOne, true, kFractionAllZero);  // spec.
  EXPECT_EQ(CPUContextX86::FxsaveToFsaveTagWord(fsw, fxsave_tag, st_mm),
            0xeebb);

  // This set set is just a mix of all of the possible tag types in a single
  // register file.
  fsw = 1 << 11;  // top = 1: logical 0-7 maps to physical 1-7, 0
  fxsave_tag = 0x1f;  // physical 5-7 (logical 4-6) empty
  SetX87OrMMXRegister(
      &st_mm[0], kExponentNormal, true, kFractionAllZero);  // valid
  SetX87OrMMXRegister(
      &st_mm[1], kExponentAllZero, false, kFractionAllZero);  // zero
  SetX87OrMMXRegister(
      &st_mm[2], kExponentAllOne, true, kFractionAllZero);  // spec.
  SetX87OrMMXRegister(
      &st_mm[3], kExponentAllOne, true, kFractionNormal);  // spec.
  SetX87OrMMXRegister(&st_mm[4], kExponentAllZero, false, kFractionAllZero);
  SetX87OrMMXRegister(&st_mm[5], kExponentAllZero, false, kFractionAllZero);
  SetX87OrMMXRegister(&st_mm[6], kExponentAllZero, false, kFractionAllZero);
  SetX87OrMMXRegister(
      &st_mm[7], kExponentNormal, true, kFractionNormal);  // valid
  EXPECT_EQ(CPUContextX86::FxsaveToFsaveTagWord(fsw, fxsave_tag, st_mm),
            0xfe90);

  // In this set, everything is valid.
  fsw = 0 << 11;  // top = 0: logical 0-7 maps to physical 0-7
  fxsave_tag = 0xff;  // nothing empty
  for (size_t index = 0; index < arraysize(st_mm); ++index) {
    SetX87OrMMXRegister(&st_mm[index], kExponentNormal, true, kFractionAllZero);
  }
  EXPECT_EQ(CPUContextX86::FxsaveToFsaveTagWord(fsw, fxsave_tag, st_mm), 0);

  // In this set, everything is empty. The registers shouldn’t be consulted at
  // all, so they’re left alone from the previous set.
  fsw = 0 << 11;  // top = 0: logical 0-7 maps to physical 0-7
  fxsave_tag = 0;  // everything empty
  EXPECT_EQ(CPUContextX86::FxsaveToFsaveTagWord(fsw, fxsave_tag, st_mm),
            0xffff);
}

TEST(CPUContextX86, FsaveToFxsaveTagWord) {
  // The register sets that these x87 tag words might apply to are given in the
  // FxsaveToFsaveTagWord test above.
  EXPECT_EQ(CPUContextX86::FsaveToFxsaveTagWord(0xff22), 0x0f);
  EXPECT_EQ(CPUContextX86::FsaveToFxsaveTagWord(0xa9ff), 0xf0);
  EXPECT_EQ(CPUContextX86::FsaveToFxsaveTagWord(0xeebb), 0x5a);
  EXPECT_EQ(CPUContextX86::FsaveToFxsaveTagWord(0xfe90), 0x1f);
  EXPECT_EQ(CPUContextX86::FsaveToFxsaveTagWord(0x0000), 0xff);
  EXPECT_EQ(CPUContextX86::FsaveToFxsaveTagWord(0xffff), 0x00);
}

}  // namespace
}  // namespace test
}  // namespace crashpad
