// Copyright 2014 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// MIPS common macros

#ifndef WEBP_DSP_MIPS_MACRO_H_
#define WEBP_DSP_MIPS_MACRO_H_

#if defined(__GNUC__) && defined(__ANDROID__) && LOCAL_GCC_VERSION == 0x409
#define WORK_AROUND_GCC
#endif

#define STR(s) #s
#define XSTR(s) STR(s)

// O0[31..16 | 15..0] = I0[31..16 | 15..0] + I1[31..16 | 15..0]
// O1[31..16 | 15..0] = I0[31..16 | 15..0] - I1[31..16 | 15..0]
// O - output
// I - input (macro doesn't change it)
#define ADD_SUB_HALVES(O0, O1,                                                 \
                       I0, I1)                                                 \
  "addq.ph          %[" #O0 "],   %[" #I0 "],  %[" #I1 "]           \n\t"      \
  "subq.ph          %[" #O1 "],   %[" #I0 "],  %[" #I1 "]           \n\t"

// O - output
// I - input (macro doesn't change it)
// I[0/1] - offset in bytes
#define LOAD_IN_X2(O0, O1,                                                     \
                   I0, I1)                                                     \
  "lh               %[" #O0 "],   " #I0 "(%[in])                  \n\t"        \
  "lh               %[" #O1 "],   " #I1 "(%[in])                  \n\t"

// I0 - location
// I1..I9 - offsets in bytes
#define LOAD_WITH_OFFSET_X4(O0, O1, O2, O3,                                    \
                            I0, I1, I2, I3, I4, I5, I6, I7, I8, I9)            \
  "ulw    %[" #O0 "],    " #I1 "+" XSTR(I9) "*" #I5 "(%[" #I0 "])       \n\t"  \
  "ulw    %[" #O1 "],    " #I2 "+" XSTR(I9) "*" #I6 "(%[" #I0 "])       \n\t"  \
  "ulw    %[" #O2 "],    " #I3 "+" XSTR(I9) "*" #I7 "(%[" #I0 "])       \n\t"  \
  "ulw    %[" #O3 "],    " #I4 "+" XSTR(I9) "*" #I8 "(%[" #I0 "])       \n\t"


// O - output
// I - input (macro doesn't change it so it should be different from I)
#define MUL_SHIFT_C1(O, I)                                                     \
  "mul              %[" #O "],    %[" #I "],    %[kC1]        \n\t"            \
  "sra              %[" #O "],    %[" #O "],    16            \n\t"            \
  "addu             %[" #O "],    %[" #O "],    %[" #I "]     \n\t"
#define MUL_SHIFT_C2(O, I) \
  "mul              %[" #O "],    %[" #I "],    %[kC2]        \n\t"            \
  "sra              %[" #O "],    %[" #O "],    16            \n\t"

// Same as #define MUL_SHIFT_C1 but I and O are the same. It stores the
// intermediary result in TMP.
#define MUL_SHIFT_C1_IO(IO, TMP)                                               \
  "mul              %[" #TMP "],  %[" #IO  "], %[kC1]     \n\t"                \
  "sra              %[" #TMP "],  %[" #TMP "], 16         \n\t"                \
  "addu             %[" #IO  "],  %[" #TMP "], %[" #IO "] \n\t"

// O - output
// IO - input/output
// I - input (macro doesn't change it)
#define MUL_SHIFT_SUM(O0, O1, O2, O3, O4, O5, O6, O7,                          \
                      IO0, IO1, IO2, IO3,                                      \
                      I0, I1, I2, I3, I4, I5, I6, I7)                          \
  MUL_SHIFT_C2(O0, I0)                                                         \
  MUL_SHIFT_C1(O1, I0)                                                         \
  MUL_SHIFT_C2(O2, I1)                                                         \
  MUL_SHIFT_C1(O3, I1)                                                         \
  MUL_SHIFT_C2(O4, I2)                                                         \
  MUL_SHIFT_C1(O5, I2)                                                         \
  MUL_SHIFT_C2(O6, I3)                                                         \
  MUL_SHIFT_C1(O7, I3)                                                         \
  "addu             %[" #IO0 "],  %[" #IO0 "],  %[" #I4 "]    \n\t"            \
  "addu             %[" #IO1 "],  %[" #IO1 "],  %[" #I5 "]    \n\t"            \
  "subu             %[" #IO2 "],  %[" #IO2 "],  %[" #I6 "]    \n\t"            \
  "subu             %[" #IO3 "],  %[" #IO3 "],  %[" #I7 "]    \n\t"

// O - output
// I - input (macro doesn't change it)
#define INSERT_HALF_X2(O0, O1,                                                 \
                       I0, I1)                                                 \
  "ins              %[" #O0 "],   %[" #I0 "], 16,    16           \n\t"        \
  "ins              %[" #O1 "],   %[" #I1 "], 16,    16           \n\t"

// O - output
// I - input (macro doesn't change it)
#define SRA_16(O0, O1, O2, O3,                                                 \
               I0, I1, I2, I3)                                                 \
  "sra              %[" #O0 "],  %[" #I0 "],  16                  \n\t"        \
  "sra              %[" #O1 "],  %[" #I1 "],  16                  \n\t"        \
  "sra              %[" #O2 "],  %[" #I2 "],  16                  \n\t"        \
  "sra              %[" #O3 "],  %[" #I3 "],  16                  \n\t"

// temp0[31..16 | 15..0] = temp8[31..16 | 15..0] + temp12[31..16 | 15..0]
// temp1[31..16 | 15..0] = temp8[31..16 | 15..0] - temp12[31..16 | 15..0]
// temp0[31..16 | 15..0] = temp0[31..16 >> 3 | 15..0 >> 3]
// temp1[31..16 | 15..0] = temp1[31..16 >> 3 | 15..0 >> 3]
// O - output
// I - input (macro doesn't change it)
#define SHIFT_R_SUM_X2(O0, O1, O2, O3, O4, O5, O6, O7,                         \
                       I0, I1, I2, I3, I4, I5, I6, I7)                         \
  "addq.ph          %[" #O0 "],   %[" #I0 "],   %[" #I4 "]    \n\t"            \
  "subq.ph          %[" #O1 "],   %[" #I0 "],   %[" #I4 "]    \n\t"            \
  "addq.ph          %[" #O2 "],   %[" #I1 "],   %[" #I5 "]    \n\t"            \
  "subq.ph          %[" #O3 "],   %[" #I1 "],   %[" #I5 "]    \n\t"            \
  "addq.ph          %[" #O4 "],   %[" #I2 "],   %[" #I6 "]    \n\t"            \
  "subq.ph          %[" #O5 "],   %[" #I2 "],   %[" #I6 "]    \n\t"            \
  "addq.ph          %[" #O6 "],   %[" #I3 "],   %[" #I7 "]    \n\t"            \
  "subq.ph          %[" #O7 "],   %[" #I3 "],   %[" #I7 "]    \n\t"            \
  "shra.ph          %[" #O0 "],   %[" #O0 "],   3             \n\t"            \
  "shra.ph          %[" #O1 "],   %[" #O1 "],   3             \n\t"            \
  "shra.ph          %[" #O2 "],   %[" #O2 "],   3             \n\t"            \
  "shra.ph          %[" #O3 "],   %[" #O3 "],   3             \n\t"            \
  "shra.ph          %[" #O4 "],   %[" #O4 "],   3             \n\t"            \
  "shra.ph          %[" #O5 "],   %[" #O5 "],   3             \n\t"            \
  "shra.ph          %[" #O6 "],   %[" #O6 "],   3             \n\t"            \
  "shra.ph          %[" #O7 "],   %[" #O7 "],   3             \n\t"

// precrq.ph.w temp0, temp8, temp2
//   temp0 = temp8[31..16] | temp2[31..16]
// ins temp2, temp8, 16, 16
//   temp2 = temp8[31..16] | temp2[15..0]
// O - output
// IO - input/output
// I - input (macro doesn't change it)
#define PACK_2_HALVES_TO_WORD(O0, O1, O2, O3,                                  \
                              IO0, IO1, IO2, IO3,                              \
                              I0, I1, I2, I3)                                  \
  "precrq.ph.w      %[" #O0 "],    %[" #I0 "],  %[" #IO0 "]       \n\t"        \
  "precrq.ph.w      %[" #O1 "],    %[" #I1 "],  %[" #IO1 "]       \n\t"        \
  "ins              %[" #IO0 "],   %[" #I0 "],  16,    16         \n\t"        \
  "ins              %[" #IO1 "],   %[" #I1 "],  16,    16         \n\t"        \
  "precrq.ph.w      %[" #O2 "],    %[" #I2 "],  %[" #IO2 "]       \n\t"        \
  "precrq.ph.w      %[" #O3 "],    %[" #I3 "],  %[" #IO3 "]       \n\t"        \
  "ins              %[" #IO2 "],   %[" #I2 "],  16,    16         \n\t"        \
  "ins              %[" #IO3 "],   %[" #I3 "],  16,    16         \n\t"

// preceu.ph.qbr temp0, temp8
//   temp0 = 0 | 0 | temp8[23..16] | temp8[7..0]
// preceu.ph.qbl temp1, temp8
//   temp1 = temp8[23..16] | temp8[7..0] | 0 | 0
// O - output
// I - input (macro doesn't change it)
#define CONVERT_2_BYTES_TO_HALF(O0, O1, O2, O3, O4, O5, O6, O7,                \
                                I0, I1, I2, I3)                                \
  "preceu.ph.qbr    %[" #O0 "],   %[" #I0 "]                      \n\t"        \
  "preceu.ph.qbl    %[" #O1 "],   %[" #I0 "]                      \n\t"        \
  "preceu.ph.qbr    %[" #O2 "],   %[" #I1 "]                      \n\t"        \
  "preceu.ph.qbl    %[" #O3 "],   %[" #I1 "]                      \n\t"        \
  "preceu.ph.qbr    %[" #O4 "],   %[" #I2 "]                      \n\t"        \
  "preceu.ph.qbl    %[" #O5 "],   %[" #I2 "]                      \n\t"        \
  "preceu.ph.qbr    %[" #O6 "],   %[" #I3 "]                      \n\t"        \
  "preceu.ph.qbl    %[" #O7 "],   %[" #I3 "]                      \n\t"

// temp0[31..16 | 15..0] = temp0[31..16 | 15..0] + temp8[31..16 | 15..0]
// temp0[31..16 | 15..0] = temp0[31..16 <<(s) 7 | 15..0 <<(s) 7]
// temp1..temp7 same as temp0
// precrqu_s.qb.ph temp0, temp1, temp0:
//   temp0 = temp1[31..24] | temp1[15..8] | temp0[31..24] | temp0[15..8]
// store temp0 to dst
// IO - input/output
// I - input (macro doesn't change it)
#define STORE_SAT_SUM_X2(IO0, IO1, IO2, IO3, IO4, IO5, IO6, IO7,               \
                         I0, I1, I2, I3, I4, I5, I6, I7,                       \
                         I8, I9, I10, I11, I12, I13)                           \
  "addq.ph          %[" #IO0 "],  %[" #IO0 "],  %[" #I0 "]          \n\t"      \
  "addq.ph          %[" #IO1 "],  %[" #IO1 "],  %[" #I1 "]          \n\t"      \
  "addq.ph          %[" #IO2 "],  %[" #IO2 "],  %[" #I2 "]          \n\t"      \
  "addq.ph          %[" #IO3 "],  %[" #IO3 "],  %[" #I3 "]          \n\t"      \
  "addq.ph          %[" #IO4 "],  %[" #IO4 "],  %[" #I4 "]          \n\t"      \
  "addq.ph          %[" #IO5 "],  %[" #IO5 "],  %[" #I5 "]          \n\t"      \
  "addq.ph          %[" #IO6 "],  %[" #IO6 "],  %[" #I6 "]          \n\t"      \
  "addq.ph          %[" #IO7 "],  %[" #IO7 "],  %[" #I7 "]          \n\t"      \
  "shll_s.ph        %[" #IO0 "],  %[" #IO0 "],  7                   \n\t"      \
  "shll_s.ph        %[" #IO1 "],  %[" #IO1 "],  7                   \n\t"      \
  "shll_s.ph        %[" #IO2 "],  %[" #IO2 "],  7                   \n\t"      \
  "shll_s.ph        %[" #IO3 "],  %[" #IO3 "],  7                   \n\t"      \
  "shll_s.ph        %[" #IO4 "],  %[" #IO4 "],  7                   \n\t"      \
  "shll_s.ph        %[" #IO5 "],  %[" #IO5 "],  7                   \n\t"      \
  "shll_s.ph        %[" #IO6 "],  %[" #IO6 "],  7                   \n\t"      \
  "shll_s.ph        %[" #IO7 "],  %[" #IO7 "],  7                   \n\t"      \
  "precrqu_s.qb.ph  %[" #IO0 "],  %[" #IO1 "],  %[" #IO0 "]         \n\t"      \
  "precrqu_s.qb.ph  %[" #IO2 "],  %[" #IO3 "],  %[" #IO2 "]         \n\t"      \
  "precrqu_s.qb.ph  %[" #IO4 "],  %[" #IO5 "],  %[" #IO4 "]         \n\t"      \
  "precrqu_s.qb.ph  %[" #IO6 "],  %[" #IO7 "],  %[" #IO6 "]         \n\t"      \
  "usw              %[" #IO0 "],  " XSTR(I13) "*" #I9 "(%[" #I8 "])   \n\t"    \
  "usw              %[" #IO2 "],  " XSTR(I13) "*" #I10 "(%[" #I8 "])  \n\t"    \
  "usw              %[" #IO4 "],  " XSTR(I13) "*" #I11 "(%[" #I8 "])  \n\t"    \
  "usw              %[" #IO6 "],  " XSTR(I13) "*" #I12 "(%[" #I8 "])  \n\t"

#define OUTPUT_EARLY_CLOBBER_REGS_10()                                         \
  : [temp1]"=&r"(temp1), [temp2]"=&r"(temp2), [temp3]"=&r"(temp3),             \
    [temp4]"=&r"(temp4), [temp5]"=&r"(temp5), [temp6]"=&r"(temp6),             \
    [temp7]"=&r"(temp7), [temp8]"=&r"(temp8), [temp9]"=&r"(temp9),             \
    [temp10]"=&r"(temp10)

#define OUTPUT_EARLY_CLOBBER_REGS_18()                                         \
  OUTPUT_EARLY_CLOBBER_REGS_10(),                                              \
  [temp11]"=&r"(temp11), [temp12]"=&r"(temp12), [temp13]"=&r"(temp13),         \
  [temp14]"=&r"(temp14), [temp15]"=&r"(temp15), [temp16]"=&r"(temp16),         \
  [temp17]"=&r"(temp17), [temp18]"=&r"(temp18)

#endif  // WEBP_DSP_MIPS_MACRO_H_
