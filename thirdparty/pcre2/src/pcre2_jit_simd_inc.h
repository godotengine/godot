/*************************************************
*      Perl-Compatible Regular Expressions       *
*************************************************/

/* PCRE is a library of functions to support regular expressions whose syntax
and semantics are as close as possible to those of the Perl 5 language.

                       Written by Philip Hazel
                    This module by Zoltan Herczeg
     Original API code Copyright (c) 1997-2012 University of Cambridge
          New API code Copyright (c) 2016-2019 University of Cambridge

-----------------------------------------------------------------------------
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the name of the University of Cambridge nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
-----------------------------------------------------------------------------
*/

#if (defined SLJIT_CONFIG_X86 && SLJIT_CONFIG_X86) && !(defined SUPPORT_VALGRIND)

#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
static struct sljit_jump *jump_if_utf_char_start(struct sljit_compiler *compiler, sljit_s32 reg)
{
#if PCRE2_CODE_UNIT_WIDTH == 8
OP2(SLJIT_AND, reg, 0, reg, 0, SLJIT_IMM, 0xc0);
return CMP(SLJIT_NOT_EQUAL, reg, 0, SLJIT_IMM, 0x80);
#elif PCRE2_CODE_UNIT_WIDTH == 16
OP2(SLJIT_AND, reg, 0, reg, 0, SLJIT_IMM, 0xfc00);
return CMP(SLJIT_NOT_EQUAL, reg, 0, SLJIT_IMM, 0xdc00);
#else
#error "Unknown code width"
#endif
}
#endif

static sljit_s32 character_to_int32(PCRE2_UCHAR chr)
{
sljit_u32 value = chr;
#if PCRE2_CODE_UNIT_WIDTH == 8
#define SSE2_COMPARE_TYPE_INDEX 0
return (sljit_s32)((value << 24) | (value << 16) | (value << 8) | value);
#elif PCRE2_CODE_UNIT_WIDTH == 16
#define SSE2_COMPARE_TYPE_INDEX 1
return (sljit_s32)((value << 16) | value);
#elif PCRE2_CODE_UNIT_WIDTH == 32
#define SSE2_COMPARE_TYPE_INDEX 2
return (sljit_s32)(value);
#else
#error "Unsupported unit width"
#endif
}

static void load_from_mem_sse2(struct sljit_compiler *compiler, sljit_s32 dst_xmm_reg, sljit_s32 src_general_reg, sljit_s8 offset)
{
sljit_u8 instruction[5];

SLJIT_ASSERT(dst_xmm_reg < 8);
SLJIT_ASSERT(src_general_reg < 8);

/* MOVDQA xmm1, xmm2/m128 */
instruction[0] = ((sljit_u8)offset & 0xf) == 0 ? 0x66 : 0xf3;
instruction[1] = 0x0f;
instruction[2] = 0x6f;

if (offset == 0)
  {
  instruction[3] = (dst_xmm_reg << 3) | src_general_reg;
  sljit_emit_op_custom(compiler, instruction, 4);
  return;
  }

instruction[3] = 0x40 | (dst_xmm_reg << 3) | src_general_reg;
instruction[4] = (sljit_u8)offset;
sljit_emit_op_custom(compiler, instruction, 5);
}

typedef enum {
    sse2_compare_match1,
    sse2_compare_match1i,
    sse2_compare_match2,
} sse2_compare_type;

static void fast_forward_char_pair_sse2_compare(struct sljit_compiler *compiler, sse2_compare_type compare_type,
  int step, sljit_s32 dst_ind, sljit_s32 cmp1_ind, sljit_s32 cmp2_ind, sljit_s32 tmp_ind)
{
sljit_u8 instruction[4];
instruction[0] = 0x66;
instruction[1] = 0x0f;

SLJIT_ASSERT(step >= 0 && step <= 3);

if (compare_type != sse2_compare_match2)
  {
  if (step == 0)
    {
    if (compare_type == sse2_compare_match1i)
      {
      /* POR xmm1, xmm2/m128 */
      /* instruction[0] = 0x66; */
      /* instruction[1] = 0x0f; */
      instruction[2] = 0xeb;
      instruction[3] = 0xc0 | (dst_ind << 3) | cmp2_ind;
      sljit_emit_op_custom(compiler, instruction, 4);
      }
    return;
    }

  if (step != 2)
    return;

  /* PCMPEQB/W/D xmm1, xmm2/m128 */
  /* instruction[0] = 0x66; */
  /* instruction[1] = 0x0f; */
  instruction[2] = 0x74 + SSE2_COMPARE_TYPE_INDEX;
  instruction[3] = 0xc0 | (dst_ind << 3) | cmp1_ind;
  sljit_emit_op_custom(compiler, instruction, 4);
  return;
  }

switch (step)
  {
  case 0:
  /* MOVDQA xmm1, xmm2/m128 */
  /* instruction[0] = 0x66; */
  /* instruction[1] = 0x0f; */
  instruction[2] = 0x6f;
  instruction[3] = 0xc0 | (tmp_ind << 3) | dst_ind;
  sljit_emit_op_custom(compiler, instruction, 4);
  return;

  case 1:
  /* PCMPEQB/W/D xmm1, xmm2/m128 */
  /* instruction[0] = 0x66; */
  /* instruction[1] = 0x0f; */
  instruction[2] = 0x74 + SSE2_COMPARE_TYPE_INDEX;
  instruction[3] = 0xc0 | (dst_ind << 3) | cmp1_ind;
  sljit_emit_op_custom(compiler, instruction, 4);
  return;

  case 2:
  /* PCMPEQB/W/D xmm1, xmm2/m128 */
  /* instruction[0] = 0x66; */
  /* instruction[1] = 0x0f; */
  instruction[2] = 0x74 + SSE2_COMPARE_TYPE_INDEX;
  instruction[3] = 0xc0 | (tmp_ind << 3) | cmp2_ind;
  sljit_emit_op_custom(compiler, instruction, 4);
  return;

  case 3:
  /* POR xmm1, xmm2/m128 */
  /* instruction[0] = 0x66; */
  /* instruction[1] = 0x0f; */
  instruction[2] = 0xeb;
  instruction[3] = 0xc0 | (dst_ind << 3) | tmp_ind;
  sljit_emit_op_custom(compiler, instruction, 4);
  return;
  }
}

#define JIT_HAS_FAST_FORWARD_CHAR_SIMD (sljit_has_cpu_feature(SLJIT_HAS_SSE2))

static void fast_forward_char_simd(compiler_common *common, PCRE2_UCHAR char1, PCRE2_UCHAR char2, sljit_s32 offset)
{
DEFINE_COMPILER;
struct sljit_label *start;
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
struct sljit_label *restart;
#endif
struct sljit_jump *quit;
struct sljit_jump *partial_quit[2];
sse2_compare_type compare_type = sse2_compare_match1;
sljit_u8 instruction[8];
sljit_s32 tmp1_reg_ind = sljit_get_register_index(TMP1);
sljit_s32 str_ptr_reg_ind = sljit_get_register_index(STR_PTR);
sljit_s32 data_ind = 0;
sljit_s32 tmp_ind = 1;
sljit_s32 cmp1_ind = 2;
sljit_s32 cmp2_ind = 3;
sljit_u32 bit = 0;
int i;

SLJIT_UNUSED_ARG(offset);

if (char1 != char2)
  {
  bit = char1 ^ char2;
  compare_type = sse2_compare_match1i;

  if (!is_powerof2(bit))
    {
    bit = 0;
    compare_type = sse2_compare_match2;
    }
  }

partial_quit[0] = CMP(SLJIT_GREATER_EQUAL, STR_PTR, 0, STR_END, 0);
if (common->mode == PCRE2_JIT_COMPLETE)
  add_jump(compiler, &common->failed_match, partial_quit[0]);

/* First part (unaligned start) */

OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, character_to_int32(char1 | bit));

SLJIT_ASSERT(tmp1_reg_ind < 8);

/* MOVD xmm, r/m32 */
instruction[0] = 0x66;
instruction[1] = 0x0f;
instruction[2] = 0x6e;
instruction[3] = 0xc0 | (cmp1_ind << 3) | tmp1_reg_ind;
sljit_emit_op_custom(compiler, instruction, 4);

if (char1 != char2)
  {
  OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, character_to_int32(bit != 0 ? bit : char2));

  /* MOVD xmm, r/m32 */
  instruction[3] = 0xc0 | (cmp2_ind << 3) | tmp1_reg_ind;
  sljit_emit_op_custom(compiler, instruction, 4);
  }

OP1(SLJIT_MOV, TMP2, 0, STR_PTR, 0);

/* PSHUFD xmm1, xmm2/m128, imm8 */
/* instruction[0] = 0x66; */
/* instruction[1] = 0x0f; */
instruction[2] = 0x70;
instruction[3] = 0xc0 | (cmp1_ind << 3) | cmp1_ind;
instruction[4] = 0;
sljit_emit_op_custom(compiler, instruction, 5);

if (char1 != char2)
  {
  /* PSHUFD xmm1, xmm2/m128, imm8 */
  instruction[3] = 0xc0 | (cmp2_ind << 3) | cmp2_ind;
  sljit_emit_op_custom(compiler, instruction, 5);
  }

#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
restart = LABEL();
#endif
OP2(SLJIT_AND, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, ~0xf);
OP2(SLJIT_AND, TMP2, 0, TMP2, 0, SLJIT_IMM, 0xf);

load_from_mem_sse2(compiler, data_ind, str_ptr_reg_ind, 0);
for (i = 0; i < 4; i++)
  fast_forward_char_pair_sse2_compare(compiler, compare_type, i, data_ind, cmp1_ind, cmp2_ind, tmp_ind);

/* PMOVMSKB reg, xmm */
/* instruction[0] = 0x66; */
/* instruction[1] = 0x0f; */
instruction[2] = 0xd7;
instruction[3] = 0xc0 | (tmp1_reg_ind << 3) | data_ind;
sljit_emit_op_custom(compiler, instruction, 4);

OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, TMP2, 0);
OP2(SLJIT_LSHR, TMP1, 0, TMP1, 0, TMP2, 0);

quit = CMP(SLJIT_NOT_ZERO, TMP1, 0, SLJIT_IMM, 0);

OP2(SLJIT_SUB, STR_PTR, 0, STR_PTR, 0, TMP2, 0);

/* Second part (aligned) */
start = LABEL();

OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, 16);

partial_quit[1] = CMP(SLJIT_GREATER_EQUAL, STR_PTR, 0, STR_END, 0);
if (common->mode == PCRE2_JIT_COMPLETE)
  add_jump(compiler, &common->failed_match, partial_quit[1]);

load_from_mem_sse2(compiler, data_ind, str_ptr_reg_ind, 0);
for (i = 0; i < 4; i++)
  fast_forward_char_pair_sse2_compare(compiler, compare_type, i, data_ind, cmp1_ind, cmp2_ind, tmp_ind);

/* PMOVMSKB reg, xmm */
/* instruction[0] = 0x66; */
/* instruction[1] = 0x0f; */
instruction[2] = 0xd7;
instruction[3] = 0xc0 | (tmp1_reg_ind << 3) | data_ind;
sljit_emit_op_custom(compiler, instruction, 4);

CMPTO(SLJIT_ZERO, TMP1, 0, SLJIT_IMM, 0, start);

JUMPHERE(quit);

/* BSF r32, r/m32 */
instruction[0] = 0x0f;
instruction[1] = 0xbc;
instruction[2] = 0xc0 | (tmp1_reg_ind << 3) | tmp1_reg_ind;
sljit_emit_op_custom(compiler, instruction, 3);

OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, TMP1, 0);

if (common->mode != PCRE2_JIT_COMPLETE)
  {
  JUMPHERE(partial_quit[0]);
  JUMPHERE(partial_quit[1]);
  OP2(SLJIT_SUB | SLJIT_SET_GREATER, SLJIT_UNUSED, 0, STR_PTR, 0, STR_END, 0);
  CMOV(SLJIT_GREATER, STR_PTR, STR_END, 0);
  }
else
  add_jump(compiler, &common->failed_match, CMP(SLJIT_GREATER_EQUAL, STR_PTR, 0, STR_END, 0));

#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
if (common->utf && offset > 0)
  {
  SLJIT_ASSERT(common->mode == PCRE2_JIT_COMPLETE);

  OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(-offset));

  quit = jump_if_utf_char_start(compiler, TMP1);

  OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
  add_jump(compiler, &common->failed_match, CMP(SLJIT_GREATER_EQUAL, STR_PTR, 0, STR_END, 0));
  OP1(SLJIT_MOV, TMP2, 0, STR_PTR, 0);
  JUMPTO(SLJIT_JUMP, restart);

  JUMPHERE(quit);
  }
#endif
}

#ifndef _WIN64

static SLJIT_INLINE sljit_u32 max_fast_forward_char_pair_offset(void)
{
#if PCRE2_CODE_UNIT_WIDTH == 8
return 15;
#elif PCRE2_CODE_UNIT_WIDTH == 16
return 7;
#elif PCRE2_CODE_UNIT_WIDTH == 32
return 3;
#else
#error "Unsupported unit width"
#endif
}

#define JIT_HAS_FAST_FORWARD_CHAR_PAIR_SIMD (sljit_has_cpu_feature(SLJIT_HAS_SSE2))

static void fast_forward_char_pair_simd(compiler_common *common, sljit_s32 offs1,
  PCRE2_UCHAR char1a, PCRE2_UCHAR char1b, sljit_s32 offs2, PCRE2_UCHAR char2a, PCRE2_UCHAR char2b)
{
DEFINE_COMPILER;
sse2_compare_type compare1_type = sse2_compare_match1;
sse2_compare_type compare2_type = sse2_compare_match1;
sljit_u32 bit1 = 0;
sljit_u32 bit2 = 0;
sljit_u32 diff = IN_UCHARS(offs1 - offs2);
sljit_s32 tmp1_reg_ind = sljit_get_register_index(TMP1);
sljit_s32 tmp2_reg_ind = sljit_get_register_index(TMP2);
sljit_s32 str_ptr_reg_ind = sljit_get_register_index(STR_PTR);
sljit_s32 data1_ind = 0;
sljit_s32 data2_ind = 1;
sljit_s32 tmp1_ind = 2;
sljit_s32 tmp2_ind = 3;
sljit_s32 cmp1a_ind = 4;
sljit_s32 cmp1b_ind = 5;
sljit_s32 cmp2a_ind = 6;
sljit_s32 cmp2b_ind = 7;
struct sljit_label *start;
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
struct sljit_label *restart;
#endif
struct sljit_jump *jump[2];
sljit_u8 instruction[8];
int i;

SLJIT_ASSERT(common->mode == PCRE2_JIT_COMPLETE && offs1 > offs2);
SLJIT_ASSERT(diff <= IN_UCHARS(max_fast_forward_char_pair_offset()));
SLJIT_ASSERT(tmp1_reg_ind < 8 && tmp2_reg_ind == 1);

/* Initialize. */
if (common->match_end_ptr != 0)
  {
  OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), common->match_end_ptr);
  OP1(SLJIT_MOV, TMP3, 0, STR_END, 0);
  OP2(SLJIT_ADD, TMP1, 0, TMP1, 0, SLJIT_IMM, IN_UCHARS(offs1 + 1));

  OP2(SLJIT_SUB | SLJIT_SET_LESS, SLJIT_UNUSED, 0, TMP1, 0, STR_END, 0);
  CMOV(SLJIT_LESS, STR_END, TMP1, 0);
  }

OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(offs1));
add_jump(compiler, &common->failed_match, CMP(SLJIT_GREATER_EQUAL, STR_PTR, 0, STR_END, 0));

/* MOVD xmm, r/m32 */
instruction[0] = 0x66;
instruction[1] = 0x0f;
instruction[2] = 0x6e;

if (char1a == char1b)
  OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, character_to_int32(char1a));
else
  {
  bit1 = char1a ^ char1b;
  if (is_powerof2(bit1))
    {
    compare1_type = sse2_compare_match1i;
    OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, character_to_int32(char1a | bit1));
    OP1(SLJIT_MOV, TMP2, 0, SLJIT_IMM, character_to_int32(bit1));
    }
  else
    {
    compare1_type = sse2_compare_match2;
    bit1 = 0;
    OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, character_to_int32(char1a));
    OP1(SLJIT_MOV, TMP2, 0, SLJIT_IMM, character_to_int32(char1b));
    }
  }

instruction[3] = 0xc0 | (cmp1a_ind << 3) | tmp1_reg_ind;
sljit_emit_op_custom(compiler, instruction, 4);

if (char1a != char1b)
  {
  instruction[3] = 0xc0 | (cmp1b_ind << 3) | tmp2_reg_ind;
  sljit_emit_op_custom(compiler, instruction, 4);
  }

if (char2a == char2b)
  OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, character_to_int32(char2a));
else
  {
  bit2 = char2a ^ char2b;
  if (is_powerof2(bit2))
    {
    compare2_type = sse2_compare_match1i;
    OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, character_to_int32(char2a | bit2));
    OP1(SLJIT_MOV, TMP2, 0, SLJIT_IMM, character_to_int32(bit2));
    }
  else
    {
    compare2_type = sse2_compare_match2;
    bit2 = 0;
    OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, character_to_int32(char2a));
    OP1(SLJIT_MOV, TMP2, 0, SLJIT_IMM, character_to_int32(char2b));
    }
  }

instruction[3] = 0xc0 | (cmp2a_ind << 3) | tmp1_reg_ind;
sljit_emit_op_custom(compiler, instruction, 4);

if (char2a != char2b)
  {
  instruction[3] = 0xc0 | (cmp2b_ind << 3) | tmp2_reg_ind;
  sljit_emit_op_custom(compiler, instruction, 4);
  }

/* PSHUFD xmm1, xmm2/m128, imm8 */
/* instruction[0] = 0x66; */
/* instruction[1] = 0x0f; */
instruction[2] = 0x70;
instruction[4] = 0;

instruction[3] = 0xc0 | (cmp1a_ind << 3) | cmp1a_ind;
sljit_emit_op_custom(compiler, instruction, 5);

if (char1a != char1b)
  {
  instruction[3] = 0xc0 | (cmp1b_ind << 3) | cmp1b_ind;
  sljit_emit_op_custom(compiler, instruction, 5);
  }

instruction[3] = 0xc0 | (cmp2a_ind << 3) | cmp2a_ind;
sljit_emit_op_custom(compiler, instruction, 5);

if (char2a != char2b)
  {
  instruction[3] = 0xc0 | (cmp2b_ind << 3) | cmp2b_ind;
  sljit_emit_op_custom(compiler, instruction, 5);
  }

#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
restart = LABEL();
#endif

OP2(SLJIT_SUB, TMP1, 0, STR_PTR, 0, SLJIT_IMM, diff);
OP1(SLJIT_MOV, TMP2, 0, STR_PTR, 0);
OP2(SLJIT_AND, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, ~0xf);

load_from_mem_sse2(compiler, data1_ind, str_ptr_reg_ind, 0);

jump[0] = CMP(SLJIT_GREATER_EQUAL, TMP1, 0, STR_PTR, 0);

load_from_mem_sse2(compiler, data2_ind, str_ptr_reg_ind, -(sljit_s8)diff);
jump[1] = JUMP(SLJIT_JUMP);

JUMPHERE(jump[0]);

/* MOVDQA xmm1, xmm2/m128 */
/* instruction[0] = 0x66; */
/* instruction[1] = 0x0f; */
instruction[2] = 0x6f;
instruction[3] = 0xc0 | (data2_ind << 3) | data1_ind;
sljit_emit_op_custom(compiler, instruction, 4);

/* PSLLDQ xmm1, imm8 */
/* instruction[0] = 0x66; */
/* instruction[1] = 0x0f; */
instruction[2] = 0x73;
instruction[3] = 0xc0 | (7 << 3) | data2_ind;
instruction[4] = diff;
sljit_emit_op_custom(compiler, instruction, 5);

JUMPHERE(jump[1]);

OP2(SLJIT_AND, TMP2, 0, TMP2, 0, SLJIT_IMM, 0xf);

for (i = 0; i < 4; i++)
  {
  fast_forward_char_pair_sse2_compare(compiler, compare2_type, i, data2_ind, cmp2a_ind, cmp2b_ind, tmp2_ind);
  fast_forward_char_pair_sse2_compare(compiler, compare1_type, i, data1_ind, cmp1a_ind, cmp1b_ind, tmp1_ind);
  }

/* PAND xmm1, xmm2/m128 */
/* instruction[0] = 0x66; */
/* instruction[1] = 0x0f; */
instruction[2] = 0xdb;
instruction[3] = 0xc0 | (data1_ind << 3) | data2_ind;
sljit_emit_op_custom(compiler, instruction, 4);

/* PMOVMSKB reg, xmm */
/* instruction[0] = 0x66; */
/* instruction[1] = 0x0f; */
instruction[2] = 0xd7;
instruction[3] = 0xc0 | (tmp1_reg_ind << 3) | 0;
sljit_emit_op_custom(compiler, instruction, 4);

/* Ignore matches before the first STR_PTR. */
OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, TMP2, 0);
OP2(SLJIT_LSHR, TMP1, 0, TMP1, 0, TMP2, 0);

jump[0] = CMP(SLJIT_NOT_ZERO, TMP1, 0, SLJIT_IMM, 0);

OP2(SLJIT_SUB, STR_PTR, 0, STR_PTR, 0, TMP2, 0);

/* Main loop. */
start = LABEL();

OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, 16);
add_jump(compiler, &common->failed_match, CMP(SLJIT_GREATER_EQUAL, STR_PTR, 0, STR_END, 0));

load_from_mem_sse2(compiler, data1_ind, str_ptr_reg_ind, 0);
load_from_mem_sse2(compiler, data2_ind, str_ptr_reg_ind, -(sljit_s8)diff);

for (i = 0; i < 4; i++)
  {
  fast_forward_char_pair_sse2_compare(compiler, compare1_type, i, data1_ind, cmp1a_ind, cmp1b_ind, tmp2_ind);
  fast_forward_char_pair_sse2_compare(compiler, compare2_type, i, data2_ind, cmp2a_ind, cmp2b_ind, tmp1_ind);
  }

/* PAND xmm1, xmm2/m128 */
/* instruction[0] = 0x66; */
/* instruction[1] = 0x0f; */
instruction[2] = 0xdb;
instruction[3] = 0xc0 | (data1_ind << 3) | data2_ind;
sljit_emit_op_custom(compiler, instruction, 4);

/* PMOVMSKB reg, xmm */
/* instruction[0] = 0x66; */
/* instruction[1] = 0x0f; */
instruction[2] = 0xd7;
instruction[3] = 0xc0 | (tmp1_reg_ind << 3) | 0;
sljit_emit_op_custom(compiler, instruction, 4);

CMPTO(SLJIT_ZERO, TMP1, 0, SLJIT_IMM, 0, start);

JUMPHERE(jump[0]);

/* BSF r32, r/m32 */
instruction[0] = 0x0f;
instruction[1] = 0xbc;
instruction[2] = 0xc0 | (tmp1_reg_ind << 3) | tmp1_reg_ind;
sljit_emit_op_custom(compiler, instruction, 3);

OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, TMP1, 0);

add_jump(compiler, &common->failed_match, CMP(SLJIT_GREATER_EQUAL, STR_PTR, 0, STR_END, 0));

if (common->match_end_ptr != 0)
  OP1(SLJIT_MOV, STR_END, 0, SLJIT_MEM1(SLJIT_SP), common->match_end_ptr);

#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
if (common->utf)
  {
  OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(-offs1));

  jump[0] = jump_if_utf_char_start(compiler, TMP1);

  OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
  CMPTO(SLJIT_LESS, STR_PTR, 0, STR_END, 0, restart);

  add_jump(compiler, &common->failed_match, JUMP(SLJIT_JUMP));

  JUMPHERE(jump[0]);
  }
#endif

OP2(SLJIT_SUB, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(offs1));

if (common->match_end_ptr != 0)
  OP1(SLJIT_MOV, STR_END, 0, TMP3, 0);
}

#endif /* !_WIN64 */

#undef SSE2_COMPARE_TYPE_INDEX

#endif /* SLJIT_CONFIG_X86 && !SUPPORT_VALGRIND */

#if (defined SLJIT_CONFIG_ARM_64 && SLJIT_CONFIG_ARM_64 && (defined __ARM_NEON || defined __ARM_NEON__))

#include <arm_neon.h>

typedef union {
  unsigned int x;
  struct { unsigned char c1, c2, c3, c4; } c;
} int_char;

#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
static SLJIT_INLINE int utf_continue(sljit_u8 *s)
{
#if PCRE2_CODE_UNIT_WIDTH == 8
return (*s & 0xc0) == 0x80;
#elif PCRE2_CODE_UNIT_WIDTH == 16
return (*s & 0xfc00) == 0xdc00;
#else
#error "Unknown code width"
#endif
}
#endif /* SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32 */

#if PCRE2_CODE_UNIT_WIDTH == 8
# define VECTOR_FACTOR 16
# define vect_t uint8x16_t
# define VLD1Q(X) vld1q_u8((sljit_u8 *)(X))
# define VCEQQ vceqq_u8
# define VORRQ vorrq_u8
# define VST1Q vst1q_u8
# define VDUPQ vdupq_n_u8
# define VEXTQ vextq_u8
# define VANDQ vandq_u8
typedef union {
       uint8_t mem[16];
       uint64_t dw[2];
} quad_word;
#elif PCRE2_CODE_UNIT_WIDTH == 16
# define VECTOR_FACTOR 8
# define vect_t uint16x8_t
# define VLD1Q(X) vld1q_u16((sljit_u16 *)(X))
# define VCEQQ vceqq_u16
# define VORRQ vorrq_u16
# define VST1Q vst1q_u16
# define VDUPQ vdupq_n_u16
# define VEXTQ vextq_u16
# define VANDQ vandq_u16
typedef union {
       uint16_t mem[8];
       uint64_t dw[2];
} quad_word;
#else
# define VECTOR_FACTOR 4
# define vect_t uint32x4_t
# define VLD1Q(X) vld1q_u32((sljit_u32 *)(X))
# define VCEQQ vceqq_u32
# define VORRQ vorrq_u32
# define VST1Q vst1q_u32
# define VDUPQ vdupq_n_u32
# define VEXTQ vextq_u32
# define VANDQ vandq_u32
typedef union {
       uint32_t mem[4];
       uint64_t dw[2];
} quad_word;
#endif

#define FFCS
#include "pcre2_jit_neon_inc.h"
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
# define FF_UTF
# include "pcre2_jit_neon_inc.h"
# undef FF_UTF
#endif
#undef FFCS

#define FFCS_2
#include "pcre2_jit_neon_inc.h"
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
# define FF_UTF
# include "pcre2_jit_neon_inc.h"
# undef FF_UTF
#endif
#undef FFCS_2

#define FFCS_MASK
#include "pcre2_jit_neon_inc.h"
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
# define FF_UTF
# include "pcre2_jit_neon_inc.h"
# undef FF_UTF
#endif
#undef FFCS_MASK

#define JIT_HAS_FAST_FORWARD_CHAR_SIMD 1

static void fast_forward_char_simd(compiler_common *common, PCRE2_UCHAR char1, PCRE2_UCHAR char2, sljit_s32 offset)
{
DEFINE_COMPILER;
int_char ic;
struct sljit_jump *partial_quit;
/* Save temporary registers. */
OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), LOCALS0, STR_PTR, 0);
OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), LOCALS1, TMP3, 0);

/* Prepare function arguments */
OP1(SLJIT_MOV, SLJIT_R0, 0, STR_END, 0);
OP1(SLJIT_MOV, SLJIT_R1, 0, STR_PTR, 0);
OP1(SLJIT_MOV, SLJIT_R2, 0, SLJIT_IMM, offset);

if (char1 == char2)
  {
    ic.c.c1 = char1;
    ic.c.c2 = char2;
    OP1(SLJIT_MOV, SLJIT_R4, 0, SLJIT_IMM, ic.x);

#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
  if (common->utf && offset > 0)
    sljit_emit_icall(compiler, SLJIT_CALL, SLJIT_RET(SW) | SLJIT_ARG1(SW) | SLJIT_ARG2(UW) | SLJIT_ARG3(UW) | SLJIT_ARG4(UW),
                     SLJIT_IMM, SLJIT_FUNC_OFFSET(ffcs_utf));
  else
    sljit_emit_icall(compiler, SLJIT_CALL, SLJIT_RET(SW) | SLJIT_ARG1(SW) | SLJIT_ARG2(UW) | SLJIT_ARG3(UW) | SLJIT_ARG4(UW),
                     SLJIT_IMM, SLJIT_FUNC_OFFSET(ffcs));
#else
  sljit_emit_icall(compiler, SLJIT_CALL, SLJIT_RET(SW) | SLJIT_ARG1(SW) | SLJIT_ARG2(UW) | SLJIT_ARG3(UW) | SLJIT_ARG4(UW),
                   SLJIT_IMM, SLJIT_FUNC_OFFSET(ffcs));
#endif
  }
else
  {
  PCRE2_UCHAR mask = char1 ^ char2;
  if (is_powerof2(mask))
    {
    ic.c.c1 = char1 | mask;
    ic.c.c2 = mask;
    OP1(SLJIT_MOV, SLJIT_R4, 0, SLJIT_IMM, ic.x);

#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
    if (common->utf && offset > 0)
      sljit_emit_icall(compiler, SLJIT_CALL, SLJIT_RET(SW) | SLJIT_ARG1(SW) | SLJIT_ARG2(UW) | SLJIT_ARG3(UW) | SLJIT_ARG4(UW),
                       SLJIT_IMM, SLJIT_FUNC_OFFSET(ffcs_mask_utf));
    else
      sljit_emit_icall(compiler, SLJIT_CALL, SLJIT_RET(SW) | SLJIT_ARG1(SW) | SLJIT_ARG2(UW) | SLJIT_ARG3(UW) | SLJIT_ARG4(UW),
                       SLJIT_IMM, SLJIT_FUNC_OFFSET(ffcs_mask));
#else
    sljit_emit_icall(compiler, SLJIT_CALL, SLJIT_RET(SW) | SLJIT_ARG1(SW) | SLJIT_ARG2(UW) | SLJIT_ARG3(UW) | SLJIT_ARG4(UW),
                     SLJIT_IMM, SLJIT_FUNC_OFFSET(ffcs_mask));
#endif
    }
  else
    {
      ic.c.c1 = char1;
      ic.c.c2 = char2;
      OP1(SLJIT_MOV, SLJIT_R4, 0, SLJIT_IMM, ic.x);

#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
    if (common->utf && offset > 0)
      sljit_emit_icall(compiler, SLJIT_CALL, SLJIT_RET(SW) | SLJIT_ARG1(SW) | SLJIT_ARG2(UW) | SLJIT_ARG3(UW) | SLJIT_ARG4(UW),
                       SLJIT_IMM, SLJIT_FUNC_OFFSET(ffcs_2_utf));
    else
      sljit_emit_icall(compiler, SLJIT_CALL, SLJIT_RET(SW) | SLJIT_ARG1(SW) | SLJIT_ARG2(UW) | SLJIT_ARG3(UW) | SLJIT_ARG4(UW),
                       SLJIT_IMM, SLJIT_FUNC_OFFSET(ffcs_2));
#else
    sljit_emit_icall(compiler, SLJIT_CALL, SLJIT_RET(SW) | SLJIT_ARG1(SW) | SLJIT_ARG2(UW) | SLJIT_ARG3(UW) | SLJIT_ARG4(UW),
                     SLJIT_IMM, SLJIT_FUNC_OFFSET(ffcs_2));
#endif
    }
  }
/* Restore registers. */
OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_MEM1(SLJIT_SP), LOCALS0);
OP1(SLJIT_MOV, TMP3, 0, SLJIT_MEM1(SLJIT_SP), LOCALS1);

/* Check return value. */
partial_quit = CMP(SLJIT_EQUAL, SLJIT_RETURN_REG, 0, SLJIT_IMM, 0);
if (common->mode == PCRE2_JIT_COMPLETE)
  add_jump(compiler, &common->failed_match, partial_quit);

/* Fast forward STR_PTR to the result of memchr. */
OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_RETURN_REG, 0);

if (common->mode != PCRE2_JIT_COMPLETE)
  JUMPHERE(partial_quit);
}

typedef enum {
  compare_match1,
  compare_match1i,
  compare_match2,
} compare_type;

static inline vect_t fast_forward_char_pair_compare(compare_type ctype, vect_t dst, vect_t cmp1, vect_t cmp2)
{
if (ctype == compare_match2)
  {
  vect_t tmp = dst;
  dst = VCEQQ(dst, cmp1);
  tmp = VCEQQ(tmp, cmp2);
  dst = VORRQ(dst, tmp);
  return dst;
  }

if (ctype == compare_match1i)
  dst = VORRQ(dst, cmp2);
dst = VCEQQ(dst, cmp1);
return dst;
}

static SLJIT_INLINE sljit_u32 max_fast_forward_char_pair_offset(void)
{
#if PCRE2_CODE_UNIT_WIDTH == 8
return 15;
#elif PCRE2_CODE_UNIT_WIDTH == 16
return 7;
#elif PCRE2_CODE_UNIT_WIDTH == 32
return 3;
#else
#error "Unsupported unit width"
#endif
}

/* ARM doesn't have a shift left across lanes. */
static SLJIT_INLINE vect_t shift_left_n_lanes(vect_t a, sljit_u8 n)
{
vect_t zero = VDUPQ(0);
SLJIT_ASSERT(0 < n && n < VECTOR_FACTOR);
/* VEXTQ takes an immediate as last argument. */
#define C(X) case X: return VEXTQ(zero, a, VECTOR_FACTOR - X);
switch (n)
  {
  C(1); C(2); C(3);
#if PCRE2_CODE_UNIT_WIDTH != 32
  C(4); C(5); C(6); C(7);
# if PCRE2_CODE_UNIT_WIDTH != 16
  C(8); C(9); C(10); C(11); C(12); C(13); C(14); C(15);
# endif
#endif
  default:
    /* Based on the ASSERT(0 < n && n < VECTOR_FACTOR) above, this won't
       happen. The return is still here for compilers to not warn. */
    return a;
  }
}

#define FFCPS
#define FFCPS_DIFF1
#define FFCPS_CHAR1A2A

#define FFCPS_0
#include "pcre2_jit_neon_inc.h"
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
# define FF_UTF
# include "pcre2_jit_neon_inc.h"
# undef FF_UTF
#endif
#undef FFCPS_0

#undef FFCPS_CHAR1A2A

#define FFCPS_1
#include "pcre2_jit_neon_inc.h"
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
# define FF_UTF
# include "pcre2_jit_neon_inc.h"
# undef FF_UTF
#endif
#undef FFCPS_1

#undef FFCPS_DIFF1

#define FFCPS_DEFAULT
#include "pcre2_jit_neon_inc.h"
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
# define FF_UTF
# include "pcre2_jit_neon_inc.h"
# undef FF_UTF
#endif
#undef FFCPS

#define JIT_HAS_FAST_FORWARD_CHAR_PAIR_SIMD 1

static void fast_forward_char_pair_simd(compiler_common *common, sljit_s32 offs1,
  PCRE2_UCHAR char1a, PCRE2_UCHAR char1b, sljit_s32 offs2, PCRE2_UCHAR char2a, PCRE2_UCHAR char2b)
{
DEFINE_COMPILER;
sljit_u32 diff = IN_UCHARS(offs1 - offs2);
struct sljit_jump *partial_quit;
int_char ic;
SLJIT_ASSERT(common->mode == PCRE2_JIT_COMPLETE && offs1 > offs2);
SLJIT_ASSERT(diff <= IN_UCHARS(max_fast_forward_char_pair_offset()));
SLJIT_ASSERT(compiler->scratches == 5);

/* Save temporary register STR_PTR. */
OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), LOCALS0, STR_PTR, 0);

/* Prepare arguments for the function call. */
if (common->match_end_ptr == 0)
   OP1(SLJIT_MOV, SLJIT_R0, 0, STR_END, 0);
else
  {
  OP1(SLJIT_MOV, SLJIT_R0, 0, SLJIT_MEM1(SLJIT_SP), common->match_end_ptr);
  OP2(SLJIT_ADD, SLJIT_R0, 0, SLJIT_R0, 0, SLJIT_IMM, IN_UCHARS(offs1 + 1));

  OP2(SLJIT_SUB | SLJIT_SET_LESS, SLJIT_UNUSED, 0, STR_END, 0, SLJIT_R0, 0);
  CMOV(SLJIT_LESS, SLJIT_R0, STR_END, 0);
  }

OP1(SLJIT_MOV, SLJIT_R1, 0, STR_PTR, 0); 
OP1(SLJIT_MOV_S32, SLJIT_R2, 0, SLJIT_IMM, offs1);
OP1(SLJIT_MOV_S32, SLJIT_R3, 0, SLJIT_IMM, offs2);
ic.c.c1 = char1a;
ic.c.c2 = char1b;
ic.c.c3 = char2a;
ic.c.c4 = char2b;
OP1(SLJIT_MOV_U32, SLJIT_R4, 0, SLJIT_IMM, ic.x);

if (diff == 1) {
  if (char1a == char1b && char2a == char2b) {
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
    if (common->utf)
      sljit_emit_icall(compiler, SLJIT_CALL, SLJIT_RET(SW) | SLJIT_ARG1(SW) | SLJIT_ARG2(SW) | SLJIT_ARG3(SW) | SLJIT_ARG4(SW),
                       SLJIT_IMM, SLJIT_FUNC_OFFSET(ffcps_0_utf));
    else
#endif
      sljit_emit_icall(compiler, SLJIT_CALL, SLJIT_RET(SW) | SLJIT_ARG1(SW) | SLJIT_ARG2(SW) | SLJIT_ARG3(SW) | SLJIT_ARG4(SW),
                       SLJIT_IMM, SLJIT_FUNC_OFFSET(ffcps_0));
  } else {
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
    if (common->utf)
      sljit_emit_icall(compiler, SLJIT_CALL, SLJIT_RET(SW) | SLJIT_ARG1(SW) | SLJIT_ARG2(SW) | SLJIT_ARG3(SW) | SLJIT_ARG4(SW),
                       SLJIT_IMM, SLJIT_FUNC_OFFSET(ffcps_1_utf));
    else
#endif
      sljit_emit_icall(compiler, SLJIT_CALL, SLJIT_RET(SW) | SLJIT_ARG1(SW) | SLJIT_ARG2(SW) | SLJIT_ARG3(SW) | SLJIT_ARG4(SW),
                       SLJIT_IMM, SLJIT_FUNC_OFFSET(ffcps_1));
  }
} else {
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
  if (common->utf)
    sljit_emit_icall(compiler, SLJIT_CALL, SLJIT_RET(SW) | SLJIT_ARG1(SW) | SLJIT_ARG2(SW) | SLJIT_ARG3(SW) | SLJIT_ARG4(SW),
                     SLJIT_IMM, SLJIT_FUNC_OFFSET(ffcps_default_utf));
  else
#endif
    sljit_emit_icall(compiler, SLJIT_CALL, SLJIT_RET(SW) | SLJIT_ARG1(SW) | SLJIT_ARG2(SW) | SLJIT_ARG3(SW) | SLJIT_ARG4(SW),
                     SLJIT_IMM, SLJIT_FUNC_OFFSET(ffcps_default));
}

/* Restore STR_PTR register. */
OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_MEM1(SLJIT_SP), LOCALS0);

/* Check return value. */
partial_quit = CMP(SLJIT_EQUAL, SLJIT_RETURN_REG, 0, SLJIT_IMM, 0);
add_jump(compiler, &common->failed_match, partial_quit);

/* Fast forward STR_PTR to the result of memchr. */
OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_RETURN_REG, 0);

JUMPHERE(partial_quit);
}

#endif /* SLJIT_CONFIG_ARM_64 && SLJIT_CONFIG_ARM_64 */
