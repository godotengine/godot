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

#if !(defined SUPPORT_VALGRIND)

#if ((defined SLJIT_CONFIG_X86 && SLJIT_CONFIG_X86) \
     || (defined SLJIT_CONFIG_S390X && SLJIT_CONFIG_S390X) \
     || (defined SLJIT_CONFIG_LOONGARCH_64 && SLJIT_CONFIG_LOONGARCH_64))

typedef enum {
  vector_compare_match1,
  vector_compare_match1i,
  vector_compare_match2,
} vector_compare_type;

#if (defined SLJIT_CONFIG_X86 && SLJIT_CONFIG_X86)
static SLJIT_INLINE sljit_s32 max_fast_forward_char_pair_offset(void)
{
#if PCRE2_CODE_UNIT_WIDTH == 8
/* The AVX2 code path is currently disabled. */
/* return sljit_has_cpu_feature(SLJIT_HAS_AVX2) ? 31 : 15; */
return 15;
#elif PCRE2_CODE_UNIT_WIDTH == 16
/* The AVX2 code path is currently disabled. */
/* return sljit_has_cpu_feature(SLJIT_HAS_AVX2) ? 15 : 7; */
return 7;
#elif PCRE2_CODE_UNIT_WIDTH == 32
/* The AVX2 code path is currently disabled. */
/* return sljit_has_cpu_feature(SLJIT_HAS_AVX2) ? 7 : 3; */
return 3;
#else
#error "Unsupported unit width"
#endif
}
#else /* !SLJIT_CONFIG_X86 */
static SLJIT_INLINE sljit_s32 max_fast_forward_char_pair_offset(void)
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
#endif /* SLJIT_CONFIG_X86 */

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

#endif /* SLJIT_CONFIG_X86 || SLJIT_CONFIG_S390X */

#if (defined SLJIT_CONFIG_X86 && SLJIT_CONFIG_X86)

static sljit_s32 character_to_int32(PCRE2_UCHAR chr)
{
sljit_u32 value = chr;
#if PCRE2_CODE_UNIT_WIDTH == 8
#define SIMD_COMPARE_TYPE_INDEX 0
return (sljit_s32)((value << 24) | (value << 16) | (value << 8) | value);
#elif PCRE2_CODE_UNIT_WIDTH == 16
#define SIMD_COMPARE_TYPE_INDEX 1
return (sljit_s32)((value << 16) | value);
#elif PCRE2_CODE_UNIT_WIDTH == 32
#define SIMD_COMPARE_TYPE_INDEX 2
return (sljit_s32)(value);
#else
#error "Unsupported unit width"
#endif
}

static void fast_forward_char_pair_sse2_compare(struct sljit_compiler *compiler, vector_compare_type compare_type,
  sljit_s32 reg_type, int step, sljit_s32 dst_ind, sljit_s32 cmp1_ind, sljit_s32 cmp2_ind, sljit_s32 tmp_ind)
{
sljit_u8 instruction[4];

if (reg_type == SLJIT_SIMD_REG_128)
  {
  instruction[0] = 0x66;
  instruction[1] = 0x0f;
  }
else
  {
  /* Two byte VEX prefix. */
  instruction[0] = 0xc5;
  instruction[1] = 0xfd;
  }

SLJIT_ASSERT(step >= 0 && step <= 3);

if (compare_type != vector_compare_match2)
  {
  if (step == 0)
    {
    if (compare_type == vector_compare_match1i)
      {
      /* POR xmm1, xmm2/m128 */
      if (reg_type == SLJIT_SIMD_REG_256)
        instruction[1] ^= (dst_ind << 3);

      /* Prefix is filled. */
      instruction[2] = 0xeb;
      instruction[3] = 0xc0 | (dst_ind << 3) | cmp2_ind;
      sljit_emit_op_custom(compiler, instruction, 4);
      }
    return;
    }

  if (step != 2)
    return;

  /* PCMPEQB/W/D xmm1, xmm2/m128 */
  if (reg_type == SLJIT_SIMD_REG_256)
    instruction[1] ^= (dst_ind << 3);

  /* Prefix is filled. */
  instruction[2] = 0x74 + SIMD_COMPARE_TYPE_INDEX;
  instruction[3] = 0xc0 | (dst_ind << 3) | cmp1_ind;
  sljit_emit_op_custom(compiler, instruction, 4);
  return;
  }

if (reg_type == SLJIT_SIMD_REG_256)
  {
  if (step == 2)
    return;

  if (step == 0)
    {
    step = 2;
    instruction[1] ^= (dst_ind << 3);
    }
  }

switch (step)
  {
  case 0:
  SLJIT_ASSERT(reg_type == SLJIT_SIMD_REG_128);

  /* MOVDQA xmm1, xmm2/m128 */
  /* Prefix is filled. */
  instruction[2] = 0x6f;
  instruction[3] = 0xc0 | (tmp_ind << 3) | dst_ind;
  sljit_emit_op_custom(compiler, instruction, 4);
  return;

  case 1:
  /* PCMPEQB/W/D xmm1, xmm2/m128 */
  if (reg_type == SLJIT_SIMD_REG_256)
    instruction[1] ^= (dst_ind << 3);

  /* Prefix is filled. */
  instruction[2] = 0x74 + SIMD_COMPARE_TYPE_INDEX;
  instruction[3] = 0xc0 | (dst_ind << 3) | cmp1_ind;
  sljit_emit_op_custom(compiler, instruction, 4);
  return;

  case 2:
  /* PCMPEQB/W/D xmm1, xmm2/m128 */
  /* Prefix is filled. */
  instruction[2] = 0x74 + SIMD_COMPARE_TYPE_INDEX;
  instruction[3] = 0xc0 | (tmp_ind << 3) | cmp2_ind;
  sljit_emit_op_custom(compiler, instruction, 4);
  return;

  case 3:
  /* POR xmm1, xmm2/m128 */
  if (reg_type == SLJIT_SIMD_REG_256)
    instruction[1] ^= (dst_ind << 3);

  /* Prefix is filled. */
  instruction[2] = 0xeb;
  instruction[3] = 0xc0 | (dst_ind << 3) | tmp_ind;
  sljit_emit_op_custom(compiler, instruction, 4);
  return;
  }
}

#define JIT_HAS_FAST_FORWARD_CHAR_SIMD (sljit_has_cpu_feature(SLJIT_HAS_SIMD))

static void fast_forward_char_simd(compiler_common *common, PCRE2_UCHAR char1, PCRE2_UCHAR char2, sljit_s32 offset)
{
DEFINE_COMPILER;
sljit_u8 instruction[8];
/* The AVX2 code path is currently disabled. */
/* sljit_s32 reg_type = sljit_has_cpu_feature(SLJIT_HAS_AVX2) ? SLJIT_SIMD_REG_256 : SLJIT_SIMD_REG_128; */
sljit_s32 reg_type = SLJIT_SIMD_REG_128;
sljit_s32 value;
struct sljit_label *start;
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
struct sljit_label *restart;
#endif
struct sljit_jump *quit;
struct sljit_jump *partial_quit[2];
vector_compare_type compare_type = vector_compare_match1;
sljit_s32 tmp1_reg_ind = sljit_get_register_index(SLJIT_GP_REGISTER, TMP1);
sljit_s32 data_ind = sljit_get_register_index(SLJIT_FLOAT_REGISTER, SLJIT_FR0);
sljit_s32 cmp1_ind = sljit_get_register_index(SLJIT_FLOAT_REGISTER, SLJIT_FR1);
sljit_s32 cmp2_ind = sljit_get_register_index(SLJIT_FLOAT_REGISTER, SLJIT_FR2);
sljit_s32 tmp_ind = sljit_get_register_index(SLJIT_FLOAT_REGISTER, SLJIT_FR3);
sljit_u32 bit = 0;
int i;

SLJIT_UNUSED_ARG(offset);

if (char1 != char2)
  {
  bit = char1 ^ char2;
  compare_type = vector_compare_match1i;

  if (!is_powerof2(bit))
    {
    bit = 0;
    compare_type = vector_compare_match2;
    }
  }

partial_quit[0] = CMP(SLJIT_GREATER_EQUAL, STR_PTR, 0, STR_END, 0);
if (common->mode == PCRE2_JIT_COMPLETE)
  add_jump(compiler, &common->failed_match, partial_quit[0]);

/* First part (unaligned start) */
value = SLJIT_SIMD_REG_128 | SLJIT_SIMD_ELEM_32 | SLJIT_SIMD_LANE_ZERO;
sljit_emit_simd_lane_mov(compiler, value, SLJIT_FR1, 0, SLJIT_IMM, character_to_int32(char1 | bit));

if (char1 != char2)
  sljit_emit_simd_lane_mov(compiler, value, SLJIT_FR2, 0, SLJIT_IMM, character_to_int32(bit != 0 ? bit : char2));

OP1(SLJIT_MOV, TMP2, 0, STR_PTR, 0);

sljit_emit_simd_lane_replicate(compiler, reg_type | SLJIT_SIMD_ELEM_32, SLJIT_FR1, SLJIT_FR1, 0);

if (char1 != char2)
  sljit_emit_simd_lane_replicate(compiler, reg_type | SLJIT_SIMD_ELEM_32, SLJIT_FR2, SLJIT_FR2, 0);

#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
restart = LABEL();
#endif

value = (reg_type == SLJIT_SIMD_REG_256) ? 0x1f : 0xf;
OP2(SLJIT_AND, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, ~value);
OP2(SLJIT_AND, TMP2, 0, TMP2, 0, SLJIT_IMM, value);

value = (reg_type == SLJIT_SIMD_REG_256) ? SLJIT_SIMD_MEM_ALIGNED_256 : SLJIT_SIMD_MEM_ALIGNED_128;
sljit_emit_simd_mov(compiler, reg_type | value, SLJIT_FR0, SLJIT_MEM1(STR_PTR), 0);

for (i = 0; i < 4; i++)
  fast_forward_char_pair_sse2_compare(compiler, compare_type, reg_type, i, data_ind, cmp1_ind, cmp2_ind, tmp_ind);

sljit_emit_simd_sign(compiler, SLJIT_SIMD_STORE | reg_type | SLJIT_SIMD_ELEM_8, SLJIT_FR0, TMP1, 0);
OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, TMP2, 0);
OP2(SLJIT_LSHR, TMP1, 0, TMP1, 0, TMP2, 0);

quit = CMP(SLJIT_NOT_ZERO, TMP1, 0, SLJIT_IMM, 0);

OP2(SLJIT_SUB, STR_PTR, 0, STR_PTR, 0, TMP2, 0);

/* Second part (aligned) */
start = LABEL();

value = (reg_type == SLJIT_SIMD_REG_256) ? 32 : 16;
OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, value);

partial_quit[1] = CMP(SLJIT_GREATER_EQUAL, STR_PTR, 0, STR_END, 0);
if (common->mode == PCRE2_JIT_COMPLETE)
  add_jump(compiler, &common->failed_match, partial_quit[1]);

value = (reg_type == SLJIT_SIMD_REG_256) ? SLJIT_SIMD_MEM_ALIGNED_256 : SLJIT_SIMD_MEM_ALIGNED_128;
sljit_emit_simd_mov(compiler, reg_type | value, SLJIT_FR0, SLJIT_MEM1(STR_PTR), 0);
for (i = 0; i < 4; i++)
  fast_forward_char_pair_sse2_compare(compiler, compare_type, reg_type, i, data_ind, cmp1_ind, cmp2_ind, tmp_ind);

sljit_emit_simd_sign(compiler, SLJIT_SIMD_STORE | reg_type | SLJIT_SIMD_ELEM_8, SLJIT_FR0, TMP1, 0);
CMPTO(SLJIT_ZERO, TMP1, 0, SLJIT_IMM, 0, start);

JUMPHERE(quit);

SLJIT_ASSERT(tmp1_reg_ind < 8);
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
  OP2U(SLJIT_SUB | SLJIT_SET_GREATER, STR_PTR, 0, STR_END, 0);
  SELECT(SLJIT_GREATER, STR_PTR, STR_END, 0, STR_PTR);
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

#define JIT_HAS_FAST_REQUESTED_CHAR_SIMD (sljit_has_cpu_feature(SLJIT_HAS_SIMD))

static jump_list *fast_requested_char_simd(compiler_common *common, PCRE2_UCHAR char1, PCRE2_UCHAR char2)
{
DEFINE_COMPILER;
sljit_u8 instruction[8];
/* The AVX2 code path is currently disabled. */
/* sljit_s32 reg_type = sljit_has_cpu_feature(SLJIT_HAS_AVX2) ? SLJIT_SIMD_REG_256 : SLJIT_SIMD_REG_128; */
sljit_s32 reg_type = SLJIT_SIMD_REG_128;
sljit_s32 value;
struct sljit_label *start;
struct sljit_jump *quit;
jump_list *not_found = NULL;
vector_compare_type compare_type = vector_compare_match1;
sljit_s32 tmp1_reg_ind = sljit_get_register_index(SLJIT_GP_REGISTER, TMP1);
sljit_s32 data_ind = sljit_get_register_index(SLJIT_FLOAT_REGISTER, SLJIT_FR0);
sljit_s32 cmp1_ind = sljit_get_register_index(SLJIT_FLOAT_REGISTER, SLJIT_FR1);
sljit_s32 cmp2_ind = sljit_get_register_index(SLJIT_FLOAT_REGISTER, SLJIT_FR2);
sljit_s32 tmp_ind = sljit_get_register_index(SLJIT_FLOAT_REGISTER, SLJIT_FR3);
sljit_u32 bit = 0;
int i;

if (char1 != char2)
  {
  bit = char1 ^ char2;
  compare_type = vector_compare_match1i;

  if (!is_powerof2(bit))
    {
    bit = 0;
    compare_type = vector_compare_match2;
    }
  }

add_jump(compiler, &not_found, CMP(SLJIT_GREATER_EQUAL, TMP1, 0, STR_END, 0));
OP1(SLJIT_MOV, TMP2, 0, TMP1, 0);
OP1(SLJIT_MOV, TMP3, 0, STR_PTR, 0);

/* First part (unaligned start) */

value = SLJIT_SIMD_REG_128 | SLJIT_SIMD_ELEM_32 | SLJIT_SIMD_LANE_ZERO;
sljit_emit_simd_lane_mov(compiler, value, SLJIT_FR1, 0, SLJIT_IMM, character_to_int32(char1 | bit));

if (char1 != char2)
  sljit_emit_simd_lane_mov(compiler, value, SLJIT_FR2, 0, SLJIT_IMM, character_to_int32(bit != 0 ? bit : char2));

OP1(SLJIT_MOV, STR_PTR, 0, TMP2, 0);

sljit_emit_simd_lane_replicate(compiler, reg_type | SLJIT_SIMD_ELEM_32, SLJIT_FR1, SLJIT_FR1, 0);

if (char1 != char2)
  sljit_emit_simd_lane_replicate(compiler, reg_type | SLJIT_SIMD_ELEM_32, SLJIT_FR2, SLJIT_FR2, 0);

value = (reg_type == SLJIT_SIMD_REG_256) ? 0x1f : 0xf;
OP2(SLJIT_AND, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, ~value);
OP2(SLJIT_AND, TMP2, 0, TMP2, 0, SLJIT_IMM, value);

value = (reg_type == SLJIT_SIMD_REG_256) ? SLJIT_SIMD_MEM_ALIGNED_256 : SLJIT_SIMD_MEM_ALIGNED_128;
sljit_emit_simd_mov(compiler, reg_type | value, SLJIT_FR0, SLJIT_MEM1(STR_PTR), 0);

for (i = 0; i < 4; i++)
  fast_forward_char_pair_sse2_compare(compiler, compare_type, reg_type, i, data_ind, cmp1_ind, cmp2_ind, tmp_ind);

sljit_emit_simd_sign(compiler, SLJIT_SIMD_STORE | reg_type | SLJIT_SIMD_ELEM_8, SLJIT_FR0, TMP1, 0);
OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, TMP2, 0);
OP2(SLJIT_LSHR, TMP1, 0, TMP1, 0, TMP2, 0);

quit = CMP(SLJIT_NOT_ZERO, TMP1, 0, SLJIT_IMM, 0);

OP2(SLJIT_SUB, STR_PTR, 0, STR_PTR, 0, TMP2, 0);

/* Second part (aligned) */
start = LABEL();

value = (reg_type == SLJIT_SIMD_REG_256) ? 32 : 16;
OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, value);

add_jump(compiler, &not_found, CMP(SLJIT_GREATER_EQUAL, STR_PTR, 0, STR_END, 0));

value = (reg_type == SLJIT_SIMD_REG_256) ? SLJIT_SIMD_MEM_ALIGNED_256 : SLJIT_SIMD_MEM_ALIGNED_128;
sljit_emit_simd_mov(compiler, reg_type | value, SLJIT_FR0, SLJIT_MEM1(STR_PTR), 0);

for (i = 0; i < 4; i++)
  fast_forward_char_pair_sse2_compare(compiler, compare_type, reg_type, i, data_ind, cmp1_ind, cmp2_ind, tmp_ind);

sljit_emit_simd_sign(compiler, SLJIT_SIMD_STORE | reg_type | SLJIT_SIMD_ELEM_8, SLJIT_FR0, TMP1, 0);
CMPTO(SLJIT_ZERO, TMP1, 0, SLJIT_IMM, 0, start);

JUMPHERE(quit);

SLJIT_ASSERT(tmp1_reg_ind < 8);
/* BSF r32, r/m32 */
instruction[0] = 0x0f;
instruction[1] = 0xbc;
instruction[2] = 0xc0 | (tmp1_reg_ind << 3) | tmp1_reg_ind;
sljit_emit_op_custom(compiler, instruction, 3);

OP2(SLJIT_ADD, TMP1, 0, TMP1, 0, STR_PTR, 0);
add_jump(compiler, &not_found, CMP(SLJIT_GREATER_EQUAL, TMP1, 0, STR_END, 0));

OP1(SLJIT_MOV, STR_PTR, 0, TMP3, 0);
return not_found;
}

#ifndef _WIN64

#define JIT_HAS_FAST_FORWARD_CHAR_PAIR_SIMD (sljit_has_cpu_feature(SLJIT_HAS_SIMD))

static void fast_forward_char_pair_simd(compiler_common *common, sljit_s32 offs1,
  PCRE2_UCHAR char1a, PCRE2_UCHAR char1b, sljit_s32 offs2, PCRE2_UCHAR char2a, PCRE2_UCHAR char2b)
{
DEFINE_COMPILER;
sljit_u8 instruction[8];
/* The AVX2 code path is currently disabled. */
/* sljit_s32 reg_type = sljit_has_cpu_feature(SLJIT_HAS_AVX2) ? SLJIT_SIMD_REG_256 : SLJIT_SIMD_REG_128; */
sljit_s32 reg_type = SLJIT_SIMD_REG_128;
sljit_s32 value;
vector_compare_type compare1_type = vector_compare_match1;
vector_compare_type compare2_type = vector_compare_match1;
sljit_u32 bit1 = 0;
sljit_u32 bit2 = 0;
sljit_u32 diff = IN_UCHARS(offs1 - offs2);
sljit_s32 tmp1_reg_ind = sljit_get_register_index(SLJIT_GP_REGISTER, TMP1);
sljit_s32 data1_ind = sljit_get_register_index(SLJIT_FLOAT_REGISTER, SLJIT_FR0);
sljit_s32 data2_ind = sljit_get_register_index(SLJIT_FLOAT_REGISTER, SLJIT_FR1);
sljit_s32 cmp1a_ind = sljit_get_register_index(SLJIT_FLOAT_REGISTER, SLJIT_FR2);
sljit_s32 cmp2a_ind = sljit_get_register_index(SLJIT_FLOAT_REGISTER, SLJIT_FR3);
sljit_s32 cmp1b_ind = sljit_get_register_index(SLJIT_FLOAT_REGISTER, SLJIT_FR4);
sljit_s32 cmp2b_ind = sljit_get_register_index(SLJIT_FLOAT_REGISTER, SLJIT_FR5);
sljit_s32 tmp1_ind = sljit_get_register_index(SLJIT_FLOAT_REGISTER, SLJIT_FR6);
sljit_s32 tmp2_ind = sljit_get_register_index(SLJIT_FLOAT_REGISTER, SLJIT_TMP_FR0);
struct sljit_label *start;
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
struct sljit_label *restart;
#endif
struct sljit_jump *jump[2];
int i;

SLJIT_ASSERT(common->mode == PCRE2_JIT_COMPLETE && offs1 > offs2 && offs2 >= 0);
SLJIT_ASSERT(diff <= (unsigned)IN_UCHARS(max_fast_forward_char_pair_offset()));

/* Initialize. */
if (common->match_end_ptr != 0)
  {
  OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), common->match_end_ptr);
  OP1(SLJIT_MOV, TMP3, 0, STR_END, 0);
  OP2(SLJIT_ADD, TMP1, 0, TMP1, 0, SLJIT_IMM, IN_UCHARS(offs1 + 1));

  OP2U(SLJIT_SUB | SLJIT_SET_LESS, TMP1, 0, STR_END, 0);
  SELECT(SLJIT_LESS, STR_END, TMP1, 0, STR_END);
  }

OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(offs1));
add_jump(compiler, &common->failed_match, CMP(SLJIT_GREATER_EQUAL, STR_PTR, 0, STR_END, 0));

if (char1a == char1b)
  OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, character_to_int32(char1a));
else
  {
  bit1 = char1a ^ char1b;
  if (is_powerof2(bit1))
    {
    compare1_type = vector_compare_match1i;
    OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, character_to_int32(char1a | bit1));
    OP1(SLJIT_MOV, TMP2, 0, SLJIT_IMM, character_to_int32(bit1));
    }
  else
    {
    compare1_type = vector_compare_match2;
    bit1 = 0;
    OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, character_to_int32(char1a));
    OP1(SLJIT_MOV, TMP2, 0, SLJIT_IMM, character_to_int32(char1b));
    }
  }

value = SLJIT_SIMD_REG_128 | SLJIT_SIMD_ELEM_32 | SLJIT_SIMD_LANE_ZERO;
sljit_emit_simd_lane_mov(compiler, value, SLJIT_FR2, 0, TMP1, 0);

if (char1a != char1b)
  sljit_emit_simd_lane_mov(compiler, value, SLJIT_FR4, 0, TMP2, 0);

if (char2a == char2b)
  OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, character_to_int32(char2a));
else
  {
  bit2 = char2a ^ char2b;
  if (is_powerof2(bit2))
    {
    compare2_type = vector_compare_match1i;
    OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, character_to_int32(char2a | bit2));
    OP1(SLJIT_MOV, TMP2, 0, SLJIT_IMM, character_to_int32(bit2));
    }
  else
    {
    compare2_type = vector_compare_match2;
    bit2 = 0;
    OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, character_to_int32(char2a));
    OP1(SLJIT_MOV, TMP2, 0, SLJIT_IMM, character_to_int32(char2b));
    }
  }

sljit_emit_simd_lane_mov(compiler, value, SLJIT_FR3, 0, TMP1, 0);

if (char2a != char2b)
  sljit_emit_simd_lane_mov(compiler, value, SLJIT_FR5, 0, TMP2, 0);

sljit_emit_simd_lane_replicate(compiler, reg_type | SLJIT_SIMD_ELEM_32, SLJIT_FR2, SLJIT_FR2, 0);
if (char1a != char1b)
  sljit_emit_simd_lane_replicate(compiler, reg_type | SLJIT_SIMD_ELEM_32, SLJIT_FR4, SLJIT_FR4, 0);

sljit_emit_simd_lane_replicate(compiler, reg_type | SLJIT_SIMD_ELEM_32, SLJIT_FR3, SLJIT_FR3, 0);
if (char2a != char2b)
  sljit_emit_simd_lane_replicate(compiler, reg_type | SLJIT_SIMD_ELEM_32, SLJIT_FR5, SLJIT_FR5, 0);

#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
restart = LABEL();
#endif

OP2(SLJIT_SUB, TMP1, 0, STR_PTR, 0, SLJIT_IMM, diff);
OP1(SLJIT_MOV, TMP2, 0, STR_PTR, 0);
value = (reg_type == SLJIT_SIMD_REG_256) ? ~0x1f : ~0xf;
OP2(SLJIT_AND, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, value);

value = (reg_type == SLJIT_SIMD_REG_256) ? SLJIT_SIMD_MEM_ALIGNED_256 : SLJIT_SIMD_MEM_ALIGNED_128;
sljit_emit_simd_mov(compiler, reg_type | value, SLJIT_FR0, SLJIT_MEM1(STR_PTR), 0);

jump[0] = CMP(SLJIT_GREATER_EQUAL, TMP1, 0, STR_PTR, 0);

sljit_emit_simd_mov(compiler, reg_type, SLJIT_FR1, SLJIT_MEM1(STR_PTR), -(sljit_sw)diff);
jump[1] = JUMP(SLJIT_JUMP);

JUMPHERE(jump[0]);

if (reg_type == SLJIT_SIMD_REG_256)
  {
  if (diff != 16)
    {
    /* PSLLDQ ymm1, ymm2, imm8 */
    instruction[0] = 0xc5;
    instruction[1] = (sljit_u8)(0xf9 ^ (data2_ind << 3));
    instruction[2] = 0x73;
    instruction[3] = 0xc0 | (7 << 3) | data1_ind;
    instruction[4] = diff & 0xf;
    sljit_emit_op_custom(compiler, instruction, 5);
    }

  instruction[0] = 0xc4;
  instruction[1] = 0xe3;
  if (diff < 16)
    {
    /* VINSERTI128 xmm1, xmm2, xmm3/m128 */
    /* instruction[0] = 0xc4; */
    /* instruction[1] = 0xe3; */
    instruction[2] = (sljit_u8)(0x7d ^ (data2_ind << 3));
    instruction[3] = 0x38;
    SLJIT_ASSERT(sljit_get_register_index(SLJIT_GP_REGISTER, STR_PTR) <= 7);
    instruction[4] = 0x40 | (data2_ind << 3) | sljit_get_register_index(SLJIT_GP_REGISTER, STR_PTR);
    instruction[5] = (sljit_u8)(16 - diff);
    instruction[6] = 1;
    sljit_emit_op_custom(compiler, instruction, 7);
    }
  else
    {
    /* VPERM2I128 xmm1, xmm2, xmm3/m128 */
    /* instruction[0] = 0xc4; */
    /* instruction[1] = 0xe3; */
    value = (diff == 16) ? data1_ind : data2_ind;
    instruction[2] = (sljit_u8)(0x7d ^ (value << 3));
    instruction[3] = 0x46;
    instruction[4] = 0xc0 | (data2_ind << 3) | value;
    instruction[5] = 0x08;
    sljit_emit_op_custom(compiler, instruction, 6);
    }
  }
else
  {
  /* MOVDQA xmm1, xmm2/m128 */
  instruction[0] = 0x66;
  instruction[1] = 0x0f;
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
  }

JUMPHERE(jump[1]);

value = (reg_type == SLJIT_SIMD_REG_256) ? 0x1f : 0xf;
OP2(SLJIT_AND, TMP2, 0, TMP2, 0, SLJIT_IMM, value);

for (i = 0; i < 4; i++)
  {
  fast_forward_char_pair_sse2_compare(compiler, compare2_type, reg_type, i, data2_ind, cmp2a_ind, cmp2b_ind, tmp2_ind);
  fast_forward_char_pair_sse2_compare(compiler, compare1_type, reg_type, i, data1_ind, cmp1a_ind, cmp1b_ind, tmp1_ind);
  }

sljit_emit_simd_op2(compiler, SLJIT_SIMD_OP2_AND | reg_type, SLJIT_FR0, SLJIT_FR0, SLJIT_FR1);
sljit_emit_simd_sign(compiler, SLJIT_SIMD_STORE | reg_type | SLJIT_SIMD_ELEM_8, SLJIT_FR0, TMP1, 0);

/* Ignore matches before the first STR_PTR. */
OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, TMP2, 0);
OP2(SLJIT_LSHR, TMP1, 0, TMP1, 0, TMP2, 0);

jump[0] = CMP(SLJIT_NOT_ZERO, TMP1, 0, SLJIT_IMM, 0);

OP2(SLJIT_SUB, STR_PTR, 0, STR_PTR, 0, TMP2, 0);

/* Main loop. */
start = LABEL();

value = (reg_type == SLJIT_SIMD_REG_256) ? 32 : 16;
OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, value);
add_jump(compiler, &common->failed_match, CMP(SLJIT_GREATER_EQUAL, STR_PTR, 0, STR_END, 0));

value = (reg_type == SLJIT_SIMD_REG_256) ? SLJIT_SIMD_MEM_ALIGNED_256 : SLJIT_SIMD_MEM_ALIGNED_128;
sljit_emit_simd_mov(compiler, reg_type | value, SLJIT_FR0, SLJIT_MEM1(STR_PTR), 0);
sljit_emit_simd_mov(compiler, reg_type, SLJIT_FR1, SLJIT_MEM1(STR_PTR), -(sljit_sw)diff);

for (i = 0; i < 4; i++)
  {
  fast_forward_char_pair_sse2_compare(compiler, compare1_type, reg_type, i, data1_ind, cmp1a_ind, cmp1b_ind, tmp2_ind);
  fast_forward_char_pair_sse2_compare(compiler, compare2_type, reg_type, i, data2_ind, cmp2a_ind, cmp2b_ind, tmp1_ind);
  }

sljit_emit_simd_op2(compiler, SLJIT_SIMD_OP2_AND | reg_type, SLJIT_FR0, SLJIT_FR0, SLJIT_FR1);
sljit_emit_simd_sign(compiler, SLJIT_SIMD_STORE | reg_type | SLJIT_SIMD_ELEM_8, SLJIT_FR0, TMP1, 0);

CMPTO(SLJIT_ZERO, TMP1, 0, SLJIT_IMM, 0, start);

JUMPHERE(jump[0]);

SLJIT_ASSERT(tmp1_reg_ind < 8);
/* BSF r32, r/m32 */
instruction[0] = 0x0f;
instruction[1] = 0xbc;
instruction[2] = 0xc0 | (tmp1_reg_ind << 3) | tmp1_reg_ind;
sljit_emit_op_custom(compiler, instruction, 3);

OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, TMP1, 0);

add_jump(compiler, &common->failed_match, CMP(SLJIT_GREATER_EQUAL, STR_PTR, 0, STR_END, 0));

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

#undef SIMD_COMPARE_TYPE_INDEX

#endif /* SLJIT_CONFIG_X86 */

#if (defined SLJIT_CONFIG_ARM_64 && SLJIT_CONFIG_ARM_64 && (defined __ARM_NEON || defined __ARM_NEON__))

#include <arm_neon.h>

typedef union {
  unsigned int x;
  struct { unsigned char c1, c2, c3, c4; } c;
} int_char;

#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
static SLJIT_INLINE int utf_continue(PCRE2_SPTR s)
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
struct sljit_jump *partial_quit, *quit;
/* Save temporary registers. */
OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), LOCALS0, STR_PTR, 0);
OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), LOCALS1, TMP3, 0);

/* Prepare function arguments */
OP1(SLJIT_MOV, SLJIT_R0, 0, STR_END, 0);
GET_LOCAL_BASE(SLJIT_R1, 0, LOCALS0);
OP1(SLJIT_MOV, SLJIT_R2, 0, SLJIT_IMM, offset);

if (char1 == char2)
  {
    ic.c.c1 = char1;
    ic.c.c2 = char2;
    OP1(SLJIT_MOV, SLJIT_R4, 0, SLJIT_IMM, ic.x);

#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
  if (common->utf && offset > 0)
    sljit_emit_icall(compiler, SLJIT_CALL, SLJIT_ARGS4(W, W, W, W, W),
                     SLJIT_IMM, SLJIT_FUNC_ADDR(ffcs_utf));
  else
    sljit_emit_icall(compiler, SLJIT_CALL, SLJIT_ARGS4(W, W, W, W, W),
                     SLJIT_IMM, SLJIT_FUNC_ADDR(ffcs));
#else
  sljit_emit_icall(compiler, SLJIT_CALL, SLJIT_ARGS4(W, W, W, W, W),
                   SLJIT_IMM, SLJIT_FUNC_ADDR(ffcs));
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
      sljit_emit_icall(compiler, SLJIT_CALL, SLJIT_ARGS4(W, W, W, W, W),
                       SLJIT_IMM, SLJIT_FUNC_ADDR(ffcs_mask_utf));
    else
      sljit_emit_icall(compiler, SLJIT_CALL, SLJIT_ARGS4(W, W, W, W, W),
                       SLJIT_IMM, SLJIT_FUNC_ADDR(ffcs_mask));
#else
    sljit_emit_icall(compiler, SLJIT_CALL, SLJIT_ARGS4(W, W, W, W, W),
                     SLJIT_IMM, SLJIT_FUNC_ADDR(ffcs_mask));
#endif
    }
  else
    {
      ic.c.c1 = char1;
      ic.c.c2 = char2;
      OP1(SLJIT_MOV, SLJIT_R4, 0, SLJIT_IMM, ic.x);

#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
    if (common->utf && offset > 0)
      sljit_emit_icall(compiler, SLJIT_CALL, SLJIT_ARGS4(W, W, W, W, W),
                       SLJIT_IMM, SLJIT_FUNC_ADDR(ffcs_2_utf));
    else
      sljit_emit_icall(compiler, SLJIT_CALL, SLJIT_ARGS4(W, W, W, W, W),
                       SLJIT_IMM, SLJIT_FUNC_ADDR(ffcs_2));
#else
    sljit_emit_icall(compiler, SLJIT_CALL, SLJIT_ARGS4(W, W, W, W, W),
                     SLJIT_IMM, SLJIT_FUNC_ADDR(ffcs_2));
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
  {
  quit = CMP(SLJIT_NOT_ZERO, SLJIT_RETURN_REG, 0, SLJIT_IMM, 0);
  JUMPHERE(partial_quit);
  OP2U(SLJIT_SUB | SLJIT_SET_GREATER, STR_PTR, 0, STR_END, 0);
  SELECT(SLJIT_GREATER, STR_PTR, STR_END, 0, STR_PTR);
  JUMPHERE(quit);
  }
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

  OP2U(SLJIT_SUB | SLJIT_SET_LESS, STR_END, 0, SLJIT_R0, 0);
  SELECT(SLJIT_LESS, SLJIT_R0, STR_END, 0, SLJIT_R0);
  }

GET_LOCAL_BASE(SLJIT_R1, 0, LOCALS0);
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
      sljit_emit_icall(compiler, SLJIT_CALL, SLJIT_ARGS4(W, W, W, W, W),
                       SLJIT_IMM, SLJIT_FUNC_ADDR(ffcps_0_utf));
    else
#endif
      sljit_emit_icall(compiler, SLJIT_CALL, SLJIT_ARGS4(W, W, W, W, W),
                       SLJIT_IMM, SLJIT_FUNC_ADDR(ffcps_0));
  } else {
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
    if (common->utf)
      sljit_emit_icall(compiler, SLJIT_CALL, SLJIT_ARGS4(W, W, W, W, W),
                       SLJIT_IMM, SLJIT_FUNC_ADDR(ffcps_1_utf));
    else
#endif
      sljit_emit_icall(compiler, SLJIT_CALL, SLJIT_ARGS4(W, W, W, W, W),
                       SLJIT_IMM, SLJIT_FUNC_ADDR(ffcps_1));
  }
} else {
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
  if (common->utf)
    sljit_emit_icall(compiler, SLJIT_CALL, SLJIT_ARGS4(W, W, W, W, W),
                     SLJIT_IMM, SLJIT_FUNC_ADDR(ffcps_default_utf));
  else
#endif
    sljit_emit_icall(compiler, SLJIT_CALL, SLJIT_ARGS4(W, W, W, W, W),
                     SLJIT_IMM, SLJIT_FUNC_ADDR(ffcps_default));
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

#if (defined SLJIT_CONFIG_S390X && SLJIT_CONFIG_S390X)

#if PCRE2_CODE_UNIT_WIDTH == 8
#define VECTOR_ELEMENT_SIZE 0
#elif PCRE2_CODE_UNIT_WIDTH == 16
#define VECTOR_ELEMENT_SIZE 1
#elif PCRE2_CODE_UNIT_WIDTH == 32
#define VECTOR_ELEMENT_SIZE 2
#else
#error "Unsupported unit width"
#endif

static void load_from_mem_vector(struct sljit_compiler *compiler, BOOL vlbb, sljit_s32 dst_vreg,
  sljit_s32 base_reg, sljit_s32 index_reg)
{
sljit_u16 instruction[3];

instruction[0] = (sljit_u16)(0xe700 | (dst_vreg << 4) | index_reg);
instruction[1] = (sljit_u16)(base_reg << 12);
instruction[2] = (sljit_u16)((0x8 << 8) | (vlbb ? 0x07 : 0x06));

sljit_emit_op_custom(compiler, instruction, 6);
}

#if PCRE2_CODE_UNIT_WIDTH == 32

static void replicate_imm_vector(struct sljit_compiler *compiler, int step, sljit_s32 dst_vreg,
  PCRE2_UCHAR chr, sljit_s32 tmp_general_reg)
{
sljit_u16 instruction[3];

SLJIT_ASSERT(step >= 0 && step <= 1);

if (chr < 0x7fff)
  {
  if (step == 1)
    return;

  /* VREPI */
  instruction[0] = (sljit_u16)(0xe700 | (dst_vreg << 4));
  instruction[1] = (sljit_u16)chr;
  instruction[2] = (sljit_u16)((VECTOR_ELEMENT_SIZE << 12) | (0x8 << 8) | 0x45);
  sljit_emit_op_custom(compiler, instruction, 6);
  return;
  }

if (step == 0)
  {
  OP1(SLJIT_MOV, tmp_general_reg, 0, SLJIT_IMM, chr);

  /* VLVG */
  instruction[0] = (sljit_u16)(0xe700 | (dst_vreg << 4) | sljit_get_register_index(SLJIT_GP_REGISTER, tmp_general_reg));
  instruction[1] = 0;
  instruction[2] = (sljit_u16)((VECTOR_ELEMENT_SIZE << 12) | (0x8 << 8) | 0x22);
  sljit_emit_op_custom(compiler, instruction, 6);
  return;
  }

/* VREP */
instruction[0] = (sljit_u16)(0xe700 | (dst_vreg << 4) | dst_vreg);
instruction[1] = 0;
instruction[2] = (sljit_u16)((VECTOR_ELEMENT_SIZE << 12) | (0xc << 8) | 0x4d);
sljit_emit_op_custom(compiler, instruction, 6);
}

#endif

static void fast_forward_char_pair_sse2_compare(struct sljit_compiler *compiler, vector_compare_type compare_type,
  int step, sljit_s32 dst_ind, sljit_s32 cmp1_ind, sljit_s32 cmp2_ind, sljit_s32 tmp_ind)
{
sljit_u16 instruction[3];

SLJIT_ASSERT(step >= 0 && step <= 2);

if (step == 1)
  {
  /* VCEQ */
  instruction[0] = (sljit_u16)(0xe700 | (dst_ind << 4) | dst_ind);
  instruction[1] = (sljit_u16)(cmp1_ind << 12);
  instruction[2] = (sljit_u16)((VECTOR_ELEMENT_SIZE << 12) | (0xe << 8) | 0xf8);
  sljit_emit_op_custom(compiler, instruction, 6);
  return;
  }

if (compare_type != vector_compare_match2)
  {
  if (step == 0 && compare_type == vector_compare_match1i)
    {
    /* VO */
    instruction[0] = (sljit_u16)(0xe700 | (dst_ind << 4) | dst_ind);
    instruction[1] = (sljit_u16)(cmp2_ind << 12);
    instruction[2] = (sljit_u16)((0xe << 8) | 0x6a);
    sljit_emit_op_custom(compiler, instruction, 6);
    }
  return;
  }

switch (step)
  {
  case 0:
  /* VCEQ */
  instruction[0] = (sljit_u16)(0xe700 | (tmp_ind << 4) | dst_ind);
  instruction[1] = (sljit_u16)(cmp2_ind << 12);
  instruction[2] = (sljit_u16)((VECTOR_ELEMENT_SIZE << 12) | (0xe << 8) | 0xf8);
  sljit_emit_op_custom(compiler, instruction, 6);
  return;

  case 2:
  /* VO */
  instruction[0] = (sljit_u16)(0xe700 | (dst_ind << 4) | dst_ind);
  instruction[1] = (sljit_u16)(tmp_ind << 12);
  instruction[2] = (sljit_u16)((0xe << 8) | 0x6a);
  sljit_emit_op_custom(compiler, instruction, 6);
  return;
  }
}

#define JIT_HAS_FAST_FORWARD_CHAR_SIMD 1

static void fast_forward_char_simd(compiler_common *common, PCRE2_UCHAR char1, PCRE2_UCHAR char2, sljit_s32 offset)
{
DEFINE_COMPILER;
sljit_u16 instruction[3];
struct sljit_label *start;
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
struct sljit_label *restart;
#endif
struct sljit_jump *quit;
struct sljit_jump *partial_quit[2];
vector_compare_type compare_type = vector_compare_match1;
sljit_s32 tmp1_reg_ind = sljit_get_register_index(SLJIT_GP_REGISTER, TMP1);
sljit_s32 str_ptr_reg_ind = sljit_get_register_index(SLJIT_GP_REGISTER, STR_PTR);
sljit_s32 data_ind = 0;
sljit_s32 tmp_ind = 1;
sljit_s32 cmp1_ind = 2;
sljit_s32 cmp2_ind = 3;
sljit_s32 zero_ind = 4;
sljit_u32 bit = 0;
int i;

SLJIT_UNUSED_ARG(offset);

if (char1 != char2)
  {
  bit = char1 ^ char2;
  compare_type = vector_compare_match1i;

  if (!is_powerof2(bit))
    {
    bit = 0;
    compare_type = vector_compare_match2;
    }
  }

partial_quit[0] = CMP(SLJIT_GREATER_EQUAL, STR_PTR, 0, STR_END, 0);
if (common->mode == PCRE2_JIT_COMPLETE)
  add_jump(compiler, &common->failed_match, partial_quit[0]);

/* First part (unaligned start) */

OP2(SLJIT_ADD, TMP2, 0, STR_PTR, 0, SLJIT_IMM, 16);

#if PCRE2_CODE_UNIT_WIDTH != 32

/* VREPI */
instruction[0] = (sljit_u16)(0xe700 | (cmp1_ind << 4));
instruction[1] = (sljit_u16)(char1 | bit);
instruction[2] = (sljit_u16)((VECTOR_ELEMENT_SIZE << 12) | (0x8 << 8) | 0x45);
sljit_emit_op_custom(compiler, instruction, 6);

if (char1 != char2)
  {
  /* VREPI */
  instruction[0] = (sljit_u16)(0xe700 | (cmp2_ind << 4));
  instruction[1] = (sljit_u16)(bit != 0 ? bit : char2);
  /* instruction[2] = (sljit_u16)((VECTOR_ELEMENT_SIZE << 12) | (0x8 << 8) | 0x45); */
  sljit_emit_op_custom(compiler, instruction, 6);
  }

#else /* PCRE2_CODE_UNIT_WIDTH == 32 */

for (int i = 0; i < 2; i++)
  {
  replicate_imm_vector(compiler, i, cmp1_ind, char1 | bit, TMP1);

  if (char1 != char2)
    replicate_imm_vector(compiler, i, cmp2_ind, bit != 0 ? bit : char2, TMP1);
  }

#endif /* PCRE2_CODE_UNIT_WIDTH != 32 */

if (compare_type == vector_compare_match2)
  {
  /* VREPI */
  instruction[0] = (sljit_u16)(0xe700 | (zero_ind << 4));
  instruction[1] = 0;
  instruction[2] = (sljit_u16)((0x8 << 8) | 0x45);
  sljit_emit_op_custom(compiler, instruction, 6);
  }

#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
restart = LABEL();
#endif

load_from_mem_vector(compiler, TRUE, data_ind, str_ptr_reg_ind, 0);
OP2(SLJIT_AND, TMP2, 0, TMP2, 0, SLJIT_IMM, ~15);

if (compare_type != vector_compare_match2)
  {
  if (compare_type == vector_compare_match1i)
    fast_forward_char_pair_sse2_compare(compiler, compare_type, 0, data_ind, cmp1_ind, cmp2_ind, tmp_ind);

  /* VFEE */
  instruction[0] = (sljit_u16)(0xe700 | (data_ind << 4) | data_ind);
  instruction[1] = (sljit_u16)((cmp1_ind << 12) | (1 << 4));
  instruction[2] = (sljit_u16)((VECTOR_ELEMENT_SIZE << 12) | (0xe << 8) | 0x80);
  sljit_emit_op_custom(compiler, instruction, 6);
  }
else
  {
  for (i = 0; i < 3; i++)
    fast_forward_char_pair_sse2_compare(compiler, compare_type, i, data_ind, cmp1_ind, cmp2_ind, tmp_ind);

  /* VFENE */
  instruction[0] = (sljit_u16)(0xe700 | (data_ind << 4) | data_ind);
  instruction[1] = (sljit_u16)((zero_ind << 12) | (1 << 4));
  instruction[2] = (sljit_u16)((0xe << 8) | 0x81);
  sljit_emit_op_custom(compiler, instruction, 6);
  }

/* VLGVB */
instruction[0] = (sljit_u16)(0xe700 | (tmp1_reg_ind << 4) | data_ind);
instruction[1] = 7;
instruction[2] = (sljit_u16)((0x4 << 8) | 0x21);
sljit_emit_op_custom(compiler, instruction, 6);

OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, TMP1, 0);
quit = CMP(SLJIT_LESS, STR_PTR, 0, TMP2, 0);

OP2(SLJIT_SUB, STR_PTR, 0, TMP2, 0, SLJIT_IMM, 16);

/* Second part (aligned) */
start = LABEL();

OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, 16);

partial_quit[1] = CMP(SLJIT_GREATER_EQUAL, STR_PTR, 0, STR_END, 0);
if (common->mode == PCRE2_JIT_COMPLETE)
  add_jump(compiler, &common->failed_match, partial_quit[1]);

load_from_mem_vector(compiler, TRUE, data_ind, str_ptr_reg_ind, 0);

if (compare_type != vector_compare_match2)
  {
  if (compare_type == vector_compare_match1i)
    fast_forward_char_pair_sse2_compare(compiler, compare_type, 0, data_ind, cmp1_ind, cmp2_ind, tmp_ind);

  /* VFEE */
  instruction[0] = (sljit_u16)(0xe700 | (data_ind << 4) | data_ind);
  instruction[1] = (sljit_u16)((cmp1_ind << 12) | (1 << 4));
  instruction[2] = (sljit_u16)((VECTOR_ELEMENT_SIZE << 12) | (0xe << 8) | 0x80);
  sljit_emit_op_custom(compiler, instruction, 6);
  }
else
  {
  for (i = 0; i < 3; i++)
    fast_forward_char_pair_sse2_compare(compiler, compare_type, i, data_ind, cmp1_ind, cmp2_ind, tmp_ind);

  /* VFENE */
  instruction[0] = (sljit_u16)(0xe700 | (data_ind << 4) | data_ind);
  instruction[1] = (sljit_u16)((zero_ind << 12) | (1 << 4));
  instruction[2] = (sljit_u16)((0xe << 8) | 0x81);
  sljit_emit_op_custom(compiler, instruction, 6);
  }

sljit_set_current_flags(compiler, SLJIT_SET_OVERFLOW);
JUMPTO(SLJIT_OVERFLOW, start);

/* VLGVB */
instruction[0] = (sljit_u16)(0xe700 | (tmp1_reg_ind << 4) | data_ind);
instruction[1] = 7;
instruction[2] = (sljit_u16)((0x4 << 8) | 0x21);
sljit_emit_op_custom(compiler, instruction, 6);

OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, TMP1, 0);

JUMPHERE(quit);

if (common->mode != PCRE2_JIT_COMPLETE)
  {
  JUMPHERE(partial_quit[0]);
  JUMPHERE(partial_quit[1]);
  OP2U(SLJIT_SUB | SLJIT_SET_GREATER, STR_PTR, 0, STR_END, 0);
  SELECT(SLJIT_GREATER, STR_PTR, STR_END, 0, STR_PTR);
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

  OP2(SLJIT_ADD, TMP2, 0, STR_PTR, 0, SLJIT_IMM, 16);
  JUMPTO(SLJIT_JUMP, restart);

  JUMPHERE(quit);
  }
#endif
}

#define JIT_HAS_FAST_REQUESTED_CHAR_SIMD 1

static jump_list *fast_requested_char_simd(compiler_common *common, PCRE2_UCHAR char1, PCRE2_UCHAR char2)
{
DEFINE_COMPILER;
sljit_u16 instruction[3];
struct sljit_label *start;
struct sljit_jump *quit;
jump_list *not_found = NULL;
vector_compare_type compare_type = vector_compare_match1;
sljit_s32 tmp1_reg_ind = sljit_get_register_index(SLJIT_GP_REGISTER, TMP1);
sljit_s32 tmp3_reg_ind = sljit_get_register_index(SLJIT_GP_REGISTER, TMP3);
sljit_s32 data_ind = 0;
sljit_s32 tmp_ind = 1;
sljit_s32 cmp1_ind = 2;
sljit_s32 cmp2_ind = 3;
sljit_s32 zero_ind = 4;
sljit_u32 bit = 0;
int i;

if (char1 != char2)
  {
  bit = char1 ^ char2;
  compare_type = vector_compare_match1i;

  if (!is_powerof2(bit))
    {
    bit = 0;
    compare_type = vector_compare_match2;
    }
  }

add_jump(compiler, &not_found, CMP(SLJIT_GREATER_EQUAL, TMP1, 0, STR_END, 0));

/* First part (unaligned start) */

OP2(SLJIT_ADD, TMP2, 0, TMP1, 0, SLJIT_IMM, 16);

#if PCRE2_CODE_UNIT_WIDTH != 32

/* VREPI */
instruction[0] = (sljit_u16)(0xe700 | (cmp1_ind << 4));
instruction[1] = (sljit_u16)(char1 | bit);
instruction[2] = (sljit_u16)((VECTOR_ELEMENT_SIZE << 12) | (0x8 << 8) | 0x45);
sljit_emit_op_custom(compiler, instruction, 6);

if (char1 != char2)
  {
  /* VREPI */
  instruction[0] = (sljit_u16)(0xe700 | (cmp2_ind << 4));
  instruction[1] = (sljit_u16)(bit != 0 ? bit : char2);
  /* instruction[2] = (sljit_u16)((VECTOR_ELEMENT_SIZE << 12) | (0x8 << 8) | 0x45); */
  sljit_emit_op_custom(compiler, instruction, 6);
  }

#else /* PCRE2_CODE_UNIT_WIDTH == 32 */

for (int i = 0; i < 2; i++)
  {
  replicate_imm_vector(compiler, i, cmp1_ind, char1 | bit, TMP3);

  if (char1 != char2)
    replicate_imm_vector(compiler, i, cmp2_ind, bit != 0 ? bit : char2, TMP3);
  }

#endif /* PCRE2_CODE_UNIT_WIDTH != 32 */

if (compare_type == vector_compare_match2)
  {
  /* VREPI */
  instruction[0] = (sljit_u16)(0xe700 | (zero_ind << 4));
  instruction[1] = 0;
  instruction[2] = (sljit_u16)((0x8 << 8) | 0x45);
  sljit_emit_op_custom(compiler, instruction, 6);
  }

load_from_mem_vector(compiler, TRUE, data_ind, tmp1_reg_ind, 0);
OP2(SLJIT_AND, TMP2, 0, TMP2, 0, SLJIT_IMM, ~15);

if (compare_type != vector_compare_match2)
  {
  if (compare_type == vector_compare_match1i)
    fast_forward_char_pair_sse2_compare(compiler, compare_type, 0, data_ind, cmp1_ind, cmp2_ind, tmp_ind);

  /* VFEE */
  instruction[0] = (sljit_u16)(0xe700 | (data_ind << 4) | data_ind);
  instruction[1] = (sljit_u16)((cmp1_ind << 12) | (1 << 4));
  instruction[2] = (sljit_u16)((VECTOR_ELEMENT_SIZE << 12) | (0xe << 8) | 0x80);
  sljit_emit_op_custom(compiler, instruction, 6);
  }
else
  {
  for (i = 0; i < 3; i++)
    fast_forward_char_pair_sse2_compare(compiler, compare_type, i, data_ind, cmp1_ind, cmp2_ind, tmp_ind);

  /* VFENE */
  instruction[0] = (sljit_u16)(0xe700 | (data_ind << 4) | data_ind);
  instruction[1] = (sljit_u16)((zero_ind << 12) | (1 << 4));
  instruction[2] = (sljit_u16)((0xe << 8) | 0x81);
  sljit_emit_op_custom(compiler, instruction, 6);
  }

/* VLGVB */
instruction[0] = (sljit_u16)(0xe700 | (tmp3_reg_ind << 4) | data_ind);
instruction[1] = 7;
instruction[2] = (sljit_u16)((0x4 << 8) | 0x21);
sljit_emit_op_custom(compiler, instruction, 6);

OP2(SLJIT_ADD, TMP1, 0, TMP1, 0, TMP3, 0);
quit = CMP(SLJIT_LESS, TMP1, 0, TMP2, 0);

OP2(SLJIT_SUB, TMP1, 0, TMP2, 0, SLJIT_IMM, 16);

/* Second part (aligned) */
start = LABEL();

OP2(SLJIT_ADD, TMP1, 0, TMP1, 0, SLJIT_IMM, 16);

add_jump(compiler, &not_found, CMP(SLJIT_GREATER_EQUAL, TMP1, 0, STR_END, 0));

load_from_mem_vector(compiler, TRUE, data_ind, tmp1_reg_ind, 0);

if (compare_type != vector_compare_match2)
  {
  if (compare_type == vector_compare_match1i)
    fast_forward_char_pair_sse2_compare(compiler, compare_type, 0, data_ind, cmp1_ind, cmp2_ind, tmp_ind);

  /* VFEE */
  instruction[0] = (sljit_u16)(0xe700 | (data_ind << 4) | data_ind);
  instruction[1] = (sljit_u16)((cmp1_ind << 12) | (1 << 4));
  instruction[2] = (sljit_u16)((VECTOR_ELEMENT_SIZE << 12) | (0xe << 8) | 0x80);
  sljit_emit_op_custom(compiler, instruction, 6);
  }
else
  {
  for (i = 0; i < 3; i++)
    fast_forward_char_pair_sse2_compare(compiler, compare_type, i, data_ind, cmp1_ind, cmp2_ind, tmp_ind);

  /* VFENE */
  instruction[0] = (sljit_u16)(0xe700 | (data_ind << 4) | data_ind);
  instruction[1] = (sljit_u16)((zero_ind << 12) | (1 << 4));
  instruction[2] = (sljit_u16)((0xe << 8) | 0x81);
  sljit_emit_op_custom(compiler, instruction, 6);
  }

sljit_set_current_flags(compiler, SLJIT_SET_OVERFLOW);
JUMPTO(SLJIT_OVERFLOW, start);

/* VLGVB */
instruction[0] = (sljit_u16)(0xe700 | (tmp3_reg_ind << 4) | data_ind);
instruction[1] = 7;
instruction[2] = (sljit_u16)((0x4 << 8) | 0x21);
sljit_emit_op_custom(compiler, instruction, 6);

OP2(SLJIT_ADD, TMP1, 0, TMP1, 0, TMP3, 0);

JUMPHERE(quit);
add_jump(compiler, &not_found, CMP(SLJIT_GREATER_EQUAL, TMP1, 0, STR_END, 0));

return not_found;
}

#define JIT_HAS_FAST_FORWARD_CHAR_PAIR_SIMD 1

static void fast_forward_char_pair_simd(compiler_common *common, sljit_s32 offs1,
  PCRE2_UCHAR char1a, PCRE2_UCHAR char1b, sljit_s32 offs2, PCRE2_UCHAR char2a, PCRE2_UCHAR char2b)
{
DEFINE_COMPILER;
sljit_u16 instruction[3];
struct sljit_label *start;
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
struct sljit_label *restart;
#endif
struct sljit_jump *quit;
struct sljit_jump *jump[2];
vector_compare_type compare1_type = vector_compare_match1;
vector_compare_type compare2_type = vector_compare_match1;
sljit_u32 bit1 = 0;
sljit_u32 bit2 = 0;
sljit_s32 diff = IN_UCHARS(offs2 - offs1);
sljit_s32 tmp1_reg_ind = sljit_get_register_index(SLJIT_GP_REGISTER, TMP1);
sljit_s32 tmp2_reg_ind = sljit_get_register_index(SLJIT_GP_REGISTER, TMP2);
sljit_s32 str_ptr_reg_ind = sljit_get_register_index(SLJIT_GP_REGISTER, STR_PTR);
sljit_s32 data1_ind = 0;
sljit_s32 data2_ind = 1;
sljit_s32 tmp1_ind = 2;
sljit_s32 tmp2_ind = 3;
sljit_s32 cmp1a_ind = 4;
sljit_s32 cmp1b_ind = 5;
sljit_s32 cmp2a_ind = 6;
sljit_s32 cmp2b_ind = 7;
sljit_s32 zero_ind = 8;
int i;

SLJIT_ASSERT(common->mode == PCRE2_JIT_COMPLETE && offs1 > offs2);
SLJIT_ASSERT(-diff <= (sljit_s32)IN_UCHARS(max_fast_forward_char_pair_offset()));
SLJIT_ASSERT(tmp1_reg_ind != 0 && tmp2_reg_ind != 0);

if (char1a != char1b)
  {
  bit1 = char1a ^ char1b;
  compare1_type = vector_compare_match1i;

  if (!is_powerof2(bit1))
    {
    bit1 = 0;
    compare1_type = vector_compare_match2;
    }
  }

if (char2a != char2b)
  {
  bit2 = char2a ^ char2b;
  compare2_type = vector_compare_match1i;

  if (!is_powerof2(bit2))
    {
    bit2 = 0;
    compare2_type = vector_compare_match2;
    }
  }

/* Initialize. */
if (common->match_end_ptr != 0)
  {
  OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), common->match_end_ptr);
  OP1(SLJIT_MOV, TMP3, 0, STR_END, 0);
  OP2(SLJIT_ADD, TMP1, 0, TMP1, 0, SLJIT_IMM, IN_UCHARS(offs1 + 1));

  OP2U(SLJIT_SUB | SLJIT_SET_LESS, TMP1, 0, STR_END, 0);
  SELECT(SLJIT_LESS, STR_END, TMP1, 0, STR_END);
  }

OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(offs1));
add_jump(compiler, &common->failed_match, CMP(SLJIT_GREATER_EQUAL, STR_PTR, 0, STR_END, 0));
OP2(SLJIT_AND, TMP2, 0, STR_PTR, 0, SLJIT_IMM, ~15);

#if PCRE2_CODE_UNIT_WIDTH != 32

OP2(SLJIT_SUB, TMP1, 0, STR_PTR, 0, SLJIT_IMM, -diff);

/* VREPI */
instruction[0] = (sljit_u16)(0xe700 | (cmp1a_ind << 4));
instruction[1] = (sljit_u16)(char1a | bit1);
instruction[2] = (sljit_u16)((VECTOR_ELEMENT_SIZE << 12) | (0x8 << 8) | 0x45);
sljit_emit_op_custom(compiler, instruction, 6);

if (char1a != char1b)
  {
  /* VREPI */
  instruction[0] = (sljit_u16)(0xe700 | (cmp1b_ind << 4));
  instruction[1] = (sljit_u16)(bit1 != 0 ? bit1 : char1b);
  /* instruction[2] = (sljit_u16)((VECTOR_ELEMENT_SIZE << 12) | (0x8 << 8) | 0x45); */
  sljit_emit_op_custom(compiler, instruction, 6);
  }

/* VREPI */
instruction[0] = (sljit_u16)(0xe700 | (cmp2a_ind << 4));
instruction[1] = (sljit_u16)(char2a | bit2);
/* instruction[2] = (sljit_u16)((VECTOR_ELEMENT_SIZE << 12) | (0x8 << 8) | 0x45); */
sljit_emit_op_custom(compiler, instruction, 6);

if (char2a != char2b)
  {
  /* VREPI */
  instruction[0] = (sljit_u16)(0xe700 | (cmp2b_ind << 4));
  instruction[1] = (sljit_u16)(bit2 != 0 ? bit2 : char2b);
  /* instruction[2] = (sljit_u16)((VECTOR_ELEMENT_SIZE << 12) | (0x8 << 8) | 0x45); */
  sljit_emit_op_custom(compiler, instruction, 6);
  }

#else /* PCRE2_CODE_UNIT_WIDTH == 32 */

for (int i = 0; i < 2; i++)
  {
  replicate_imm_vector(compiler, i, cmp1a_ind, char1a | bit1, TMP1);

  if (char1a != char1b)
    replicate_imm_vector(compiler, i, cmp1b_ind, bit1 != 0 ? bit1 : char1b, TMP1);

  replicate_imm_vector(compiler, i, cmp2a_ind, char2a | bit2, TMP1);

  if (char2a != char2b)
    replicate_imm_vector(compiler, i, cmp2b_ind, bit2 != 0 ? bit2 : char2b, TMP1);
  }

OP2(SLJIT_SUB, TMP1, 0, STR_PTR, 0, SLJIT_IMM, -diff);

#endif /* PCRE2_CODE_UNIT_WIDTH != 32 */

/* VREPI */
instruction[0] = (sljit_u16)(0xe700 | (zero_ind << 4));
instruction[1] = 0;
instruction[2] = (sljit_u16)((0x8 << 8) | 0x45);
sljit_emit_op_custom(compiler, instruction, 6);

#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
restart = LABEL();
#endif

jump[0] = CMP(SLJIT_LESS, TMP1, 0, TMP2, 0);
load_from_mem_vector(compiler, TRUE, data2_ind, tmp1_reg_ind, 0);
jump[1] = JUMP(SLJIT_JUMP);
JUMPHERE(jump[0]);
load_from_mem_vector(compiler, FALSE, data2_ind, tmp1_reg_ind, 0);
JUMPHERE(jump[1]);

load_from_mem_vector(compiler, TRUE, data1_ind, str_ptr_reg_ind, 0);
OP2(SLJIT_ADD, TMP2, 0, TMP2, 0, SLJIT_IMM, 16);

for (i = 0; i < 3; i++)
  {
  fast_forward_char_pair_sse2_compare(compiler, compare1_type, i, data1_ind, cmp1a_ind, cmp1b_ind, tmp1_ind);
  fast_forward_char_pair_sse2_compare(compiler, compare2_type, i, data2_ind, cmp2a_ind, cmp2b_ind, tmp2_ind);
  }

/* VN */
instruction[0] = (sljit_u16)(0xe700 | (data1_ind << 4) | data1_ind);
instruction[1] = (sljit_u16)(data2_ind << 12);
instruction[2] = (sljit_u16)((0xe << 8) | 0x68);
sljit_emit_op_custom(compiler, instruction, 6);

/* VFENE */
instruction[0] = (sljit_u16)(0xe700 | (data1_ind << 4) | data1_ind);
instruction[1] = (sljit_u16)((zero_ind << 12) | (1 << 4));
instruction[2] = (sljit_u16)((0xe << 8) | 0x81);
sljit_emit_op_custom(compiler, instruction, 6);

/* VLGVB */
instruction[0] = (sljit_u16)(0xe700 | (tmp1_reg_ind << 4) | data1_ind);
instruction[1] = 7;
instruction[2] = (sljit_u16)((0x4 << 8) | 0x21);
sljit_emit_op_custom(compiler, instruction, 6);

OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, TMP1, 0);
quit = CMP(SLJIT_LESS, STR_PTR, 0, TMP2, 0);

OP2(SLJIT_SUB, STR_PTR, 0, TMP2, 0, SLJIT_IMM, 16);
OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, diff);

/* Main loop. */
start = LABEL();

OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, 16);
add_jump(compiler, &common->failed_match, CMP(SLJIT_GREATER_EQUAL, STR_PTR, 0, STR_END, 0));

load_from_mem_vector(compiler, FALSE, data1_ind, str_ptr_reg_ind, 0);
load_from_mem_vector(compiler, FALSE, data2_ind, str_ptr_reg_ind, tmp1_reg_ind);

for (i = 0; i < 3; i++)
  {
  fast_forward_char_pair_sse2_compare(compiler, compare1_type, i, data1_ind, cmp1a_ind, cmp1b_ind, tmp1_ind);
  fast_forward_char_pair_sse2_compare(compiler, compare2_type, i, data2_ind, cmp2a_ind, cmp2b_ind, tmp2_ind);
  }

/* VN */
instruction[0] = (sljit_u16)(0xe700 | (data1_ind << 4) | data1_ind);
instruction[1] = (sljit_u16)(data2_ind << 12);
instruction[2] = (sljit_u16)((0xe << 8) | 0x68);
sljit_emit_op_custom(compiler, instruction, 6);

/* VFENE */
instruction[0] = (sljit_u16)(0xe700 | (data1_ind << 4) | data1_ind);
instruction[1] = (sljit_u16)((zero_ind << 12) | (1 << 4));
instruction[2] = (sljit_u16)((0xe << 8) | 0x81);
sljit_emit_op_custom(compiler, instruction, 6);

sljit_set_current_flags(compiler, SLJIT_SET_OVERFLOW);
JUMPTO(SLJIT_OVERFLOW, start);

/* VLGVB */
instruction[0] = (sljit_u16)(0xe700 | (tmp2_reg_ind << 4) | data1_ind);
instruction[1] = 7;
instruction[2] = (sljit_u16)((0x4 << 8) | 0x21);
sljit_emit_op_custom(compiler, instruction, 6);

OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, TMP2, 0);

JUMPHERE(quit);

add_jump(compiler, &common->failed_match, CMP(SLJIT_GREATER_EQUAL, STR_PTR, 0, STR_END, 0));

#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
if (common->utf)
  {
  SLJIT_ASSERT(common->mode == PCRE2_JIT_COMPLETE);

  OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), IN_UCHARS(-offs1));

  quit = jump_if_utf_char_start(compiler, TMP1);

  OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
  add_jump(compiler, &common->failed_match, CMP(SLJIT_GREATER_EQUAL, STR_PTR, 0, STR_END, 0));

  /* TMP1 contains diff. */
  OP2(SLJIT_AND, TMP2, 0, STR_PTR, 0, SLJIT_IMM, ~15);
  OP2(SLJIT_SUB, TMP1, 0, STR_PTR, 0, SLJIT_IMM, -diff);
  JUMPTO(SLJIT_JUMP, restart);

  JUMPHERE(quit);
  }
#endif

OP2(SLJIT_SUB, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(offs1));

if (common->match_end_ptr != 0)
  OP1(SLJIT_MOV, STR_END, 0, TMP3, 0);
}

#endif /* SLJIT_CONFIG_S390X */

#if (defined SLJIT_CONFIG_LOONGARCH_64 && SLJIT_CONFIG_LOONGARCH_64)

#ifdef __linux__
/* Using getauxval(AT_HWCAP) under Linux for detecting whether LSX is available */
#include <sys/auxv.h>
#define LOONGARCH_HWCAP_LSX  (1 << 4)
#define HAS_LSX_SUPPORT ((getauxval(AT_HWCAP) & LOONGARCH_HWCAP_LSX) != 0)
#else
#define HAS_LSX_SUPPORT 0
#endif

typedef sljit_ins sljit_u32;

#define SI12_IMM_MASK   0x003ffc00
#define UI5_IMM_MASK    0x00007c00
#define UI2_IMM_MASK    0x00000c00

#define VD(vd)      ((sljit_ins)vd << 0)
#define VJ(vj)      ((sljit_ins)vj << 5)
#define VK(vk)      ((sljit_ins)vk << 10)
#define RD_V(rd)    ((sljit_ins)rd << 0)
#define RJ_V(rj)    ((sljit_ins)rj << 5)

#define IMM_SI12(imm)   (((sljit_ins)(imm) << 10) & SI12_IMM_MASK)
#define IMM_UI5(imm)    (((sljit_ins)(imm) << 10) & UI5_IMM_MASK)
#define IMM_UI2(imm)    (((sljit_ins)(imm) << 10) & UI2_IMM_MASK)

// LSX OPCODES:
#define VLD           0x2c000000
#define VOR_V         0x71268000
#define VAND_V        0x71260000
#define VBSLL_V       0x728e0000
#define VMSKLTZ_B     0x729c4000
#define VPICKVE2GR_WU 0x72f3e000

#if PCRE2_CODE_UNIT_WIDTH == 8
#define VREPLGR2VR  0x729f0000
#define VSEQ        0x70000000
#elif PCRE2_CODE_UNIT_WIDTH == 16
#define VREPLGR2VR  0x729f0400
#define VSEQ        0x70008000
#else
#define VREPLGR2VR  0x729f0800
#define VSEQ        0x70010000
#endif

static void fast_forward_char_pair_lsx_compare(struct sljit_compiler *compiler, vector_compare_type compare_type,
  sljit_s32 dst_ind, sljit_s32 cmp1_ind, sljit_s32 cmp2_ind, sljit_s32 tmp_ind)
{
if (compare_type != vector_compare_match2)
  {
  if (compare_type == vector_compare_match1i)
    {
    /* VOR.V vd, vj, vk */
    push_inst(compiler, VOR_V | VD(dst_ind) | VJ(cmp2_ind) | VK(dst_ind));
    }

  /* VSEQ.B/H/W vd, vj, vk */
  push_inst(compiler, VSEQ | VD(dst_ind) | VJ(dst_ind) | VK(cmp1_ind));
  return;
  }

/* VBSLL.V vd, vj, ui5 */
push_inst(compiler, VBSLL_V | VD(tmp_ind) | VJ(dst_ind) | IMM_UI5(0));

/* VSEQ.B/H/W vd, vj, vk */
push_inst(compiler, VSEQ | VD(dst_ind) | VJ(dst_ind) | VK(cmp1_ind));

/* VSEQ.B/H/W vd, vj, vk */
push_inst(compiler, VSEQ | VD(tmp_ind) | VJ(tmp_ind) | VK(cmp2_ind));

/* VOR vd, vj, vk */
push_inst(compiler, VOR_V | VD(dst_ind) | VJ(tmp_ind) | VK(dst_ind));
return;
}

#define JIT_HAS_FAST_FORWARD_CHAR_SIMD HAS_LSX_SUPPORT

static void fast_forward_char_simd(compiler_common *common, PCRE2_UCHAR char1, PCRE2_UCHAR char2, sljit_s32 offset)
{
DEFINE_COMPILER;
struct sljit_label *start;
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
struct sljit_label *restart;
#endif
struct sljit_jump *quit;
struct sljit_jump *partial_quit[2];
vector_compare_type compare_type = vector_compare_match1;
sljit_s32 tmp1_reg_ind = sljit_get_register_index(SLJIT_GP_REGISTER, TMP1);
sljit_s32 str_ptr_reg_ind = sljit_get_register_index(SLJIT_GP_REGISTER, STR_PTR);
sljit_s32 data_ind = 0;
sljit_s32 tmp_ind = 1;
sljit_s32 cmp1_ind = 2;
sljit_s32 cmp2_ind = 3;
sljit_u32 bit = 0;

SLJIT_UNUSED_ARG(offset);

if (char1 != char2)
  {
  bit = char1 ^ char2;
  compare_type = vector_compare_match1i;

  if (!is_powerof2(bit))
    {
    bit = 0;
    compare_type = vector_compare_match2;
    }
  }

partial_quit[0] = CMP(SLJIT_GREATER_EQUAL, STR_PTR, 0, STR_END, 0);
if (common->mode == PCRE2_JIT_COMPLETE)
  add_jump(compiler, &common->failed_match, partial_quit[0]);

/* First part (unaligned start) */

OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, char1 | bit);

/* VREPLGR2VR.B/H/W vd, rj */
push_inst(compiler, VREPLGR2VR | VD(cmp1_ind) | RJ_V(tmp1_reg_ind));

if (char1 != char2)
  {
  OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, bit != 0 ? bit : char2);

  /* VREPLGR2VR.B/H/W vd, rj */
  push_inst(compiler, VREPLGR2VR | VD(cmp2_ind) | RJ_V(tmp1_reg_ind));
  }

OP1(SLJIT_MOV, TMP2, 0, STR_PTR, 0);

#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
restart = LABEL();
#endif

OP2(SLJIT_AND, TMP2, 0, TMP2, 0, SLJIT_IMM, 0xf);
OP2(SLJIT_SUB, STR_PTR, 0, STR_PTR, 0, TMP2, 0);

/* VLD vd, rj, si12 */
push_inst(compiler, VLD | VD(data_ind) | RJ_V(str_ptr_reg_ind) | IMM_SI12(0));
fast_forward_char_pair_lsx_compare(compiler, compare_type, data_ind, cmp1_ind, cmp2_ind, tmp_ind);

/* VMSKLTZ.B vd, vj */
push_inst(compiler, VMSKLTZ_B | VD(tmp_ind) | VJ(data_ind));

/* VPICKVE2GR.WU rd, vj, ui2 */
push_inst(compiler, VPICKVE2GR_WU | RD_V(tmp1_reg_ind) | VJ(tmp_ind) | IMM_UI2(0));

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

/* VLD vd, rj, si12 */
push_inst(compiler, VLD | VD(data_ind) | RJ_V(str_ptr_reg_ind) | IMM_SI12(0));
fast_forward_char_pair_lsx_compare(compiler, compare_type, data_ind, cmp1_ind, cmp2_ind, tmp_ind);

/* VMSKLTZ.B vd, vj */
push_inst(compiler, VMSKLTZ_B | VD(tmp_ind) | VJ(data_ind));

/* VPICKVE2GR.WU rd, vj, ui2 */
push_inst(compiler, VPICKVE2GR_WU | RD_V(tmp1_reg_ind) | VJ(tmp_ind) | IMM_UI2(0));

CMPTO(SLJIT_ZERO, TMP1, 0, SLJIT_IMM, 0, start);

JUMPHERE(quit);

/* CTZ.W rd, rj */
push_inst(compiler, CTZ_W | RD_V(tmp1_reg_ind) | RJ_V(tmp1_reg_ind));

OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, TMP1, 0);

if (common->mode != PCRE2_JIT_COMPLETE)
  {
  JUMPHERE(partial_quit[0]);
  JUMPHERE(partial_quit[1]);
  OP2U(SLJIT_SUB | SLJIT_SET_GREATER, STR_PTR, 0, STR_END, 0);
  SELECT(SLJIT_GREATER, STR_PTR, STR_END, 0, STR_PTR);
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

#define JIT_HAS_FAST_REQUESTED_CHAR_SIMD HAS_LSX_SUPPORT

static jump_list *fast_requested_char_simd(compiler_common *common, PCRE2_UCHAR char1, PCRE2_UCHAR char2)
{
DEFINE_COMPILER;
struct sljit_label *start;
struct sljit_jump *quit;
jump_list *not_found = NULL;
vector_compare_type compare_type = vector_compare_match1;
sljit_s32 tmp1_reg_ind = sljit_get_register_index(SLJIT_GP_REGISTER, TMP1);
sljit_s32 str_ptr_reg_ind = sljit_get_register_index(SLJIT_GP_REGISTER, STR_PTR);
sljit_s32 data_ind = 0;
sljit_s32 tmp_ind = 1;
sljit_s32 cmp1_ind = 2;
sljit_s32 cmp2_ind = 3;
sljit_u32 bit = 0;

if (char1 != char2)
  {
  bit = char1 ^ char2;
  compare_type = vector_compare_match1i;

  if (!is_powerof2(bit))
    {
    bit = 0;
    compare_type = vector_compare_match2;
    }
  }

add_jump(compiler, &not_found, CMP(SLJIT_GREATER_EQUAL, TMP1, 0, STR_END, 0));
OP1(SLJIT_MOV, TMP2, 0, TMP1, 0);
OP1(SLJIT_MOV, TMP3, 0, STR_PTR, 0);

/* First part (unaligned start) */

OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, char1 | bit);

/* VREPLGR2VR vd, rj */
push_inst(compiler, VREPLGR2VR | VD(cmp1_ind) | RJ_V(tmp1_reg_ind));

if (char1 != char2)
  {
  OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, bit != 0 ? bit : char2);
  /* VREPLGR2VR vd, rj */
  push_inst(compiler, VREPLGR2VR | VD(cmp2_ind) | RJ_V(tmp1_reg_ind));
  }

OP1(SLJIT_MOV, STR_PTR, 0, TMP2, 0);
OP2(SLJIT_AND, TMP2, 0, TMP2, 0, SLJIT_IMM, 0xf);
OP2(SLJIT_SUB, STR_PTR, 0, STR_PTR, 0, TMP2, 0);

/* VLD vd, rj, si12 */
push_inst(compiler, VLD | VD(data_ind) | RJ_V(str_ptr_reg_ind) | IMM_SI12(0));
fast_forward_char_pair_lsx_compare(compiler, compare_type, data_ind, cmp1_ind, cmp2_ind, tmp_ind);

/* VMSKLTZ.B vd, vj */
push_inst(compiler, VMSKLTZ_B | VD(tmp_ind) | VJ(data_ind));

/* VPICKVE2GR.WU rd, vj, ui2 */
push_inst(compiler, VPICKVE2GR_WU | RD_V(tmp1_reg_ind) | VJ(tmp_ind) | IMM_UI2(0));

OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, TMP2, 0);
OP2(SLJIT_LSHR, TMP1, 0, TMP1, 0, TMP2, 0);

quit = CMP(SLJIT_NOT_ZERO, TMP1, 0, SLJIT_IMM, 0);

OP2(SLJIT_SUB, STR_PTR, 0, STR_PTR, 0, TMP2, 0);

/* Second part (aligned) */
start = LABEL();

OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, 16);

add_jump(compiler, &not_found, CMP(SLJIT_GREATER_EQUAL, STR_PTR, 0, STR_END, 0));

/* VLD vd, rj, si12 */
push_inst(compiler, VLD | VD(data_ind) | RJ_V(str_ptr_reg_ind) | IMM_SI12(0));
fast_forward_char_pair_lsx_compare(compiler, compare_type, data_ind, cmp1_ind, cmp2_ind, tmp_ind);

/* VMSKLTZ.B vd, vj */
push_inst(compiler, VMSKLTZ_B | VD(tmp_ind) | VJ(data_ind));

/* VPICKVE2GR.WU rd, vj, ui2 */
push_inst(compiler, VPICKVE2GR_WU | RD_V(tmp1_reg_ind) | VJ(tmp_ind) | IMM_UI2(0));

CMPTO(SLJIT_ZERO, TMP1, 0, SLJIT_IMM, 0, start);

JUMPHERE(quit);

/* CTZ.W rd, rj */
push_inst(compiler, CTZ_W | RD_V(tmp1_reg_ind) | RJ_V(tmp1_reg_ind));

OP2(SLJIT_ADD, TMP1, 0, TMP1, 0, STR_PTR, 0);
add_jump(compiler, &not_found, CMP(SLJIT_GREATER_EQUAL, TMP1, 0, STR_END, 0));

OP1(SLJIT_MOV, STR_PTR, 0, TMP3, 0);
return not_found;
}

#define JIT_HAS_FAST_FORWARD_CHAR_PAIR_SIMD HAS_LSX_SUPPORT

static void fast_forward_char_pair_simd(compiler_common *common, sljit_s32 offs1,
  PCRE2_UCHAR char1a, PCRE2_UCHAR char1b, sljit_s32 offs2, PCRE2_UCHAR char2a, PCRE2_UCHAR char2b)
{
DEFINE_COMPILER;
vector_compare_type compare1_type = vector_compare_match1;
vector_compare_type compare2_type = vector_compare_match1;
sljit_u32 bit1 = 0;
sljit_u32 bit2 = 0;
sljit_u32 diff = IN_UCHARS(offs1 - offs2);
sljit_s32 tmp1_reg_ind = sljit_get_register_index(SLJIT_GP_REGISTER, TMP1);
sljit_s32 tmp2_reg_ind = sljit_get_register_index(SLJIT_GP_REGISTER, TMP2);
sljit_s32 str_ptr_reg_ind = sljit_get_register_index(SLJIT_GP_REGISTER, STR_PTR);
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

SLJIT_ASSERT(common->mode == PCRE2_JIT_COMPLETE && offs1 > offs2);
SLJIT_ASSERT(diff <= IN_UCHARS(max_fast_forward_char_pair_offset()));

/* Initialize. */
if (common->match_end_ptr != 0)
  {
  OP1(SLJIT_MOV, TMP1, 0, SLJIT_MEM1(SLJIT_SP), common->match_end_ptr);
  OP2(SLJIT_ADD, TMP1, 0, TMP1, 0, SLJIT_IMM, IN_UCHARS(offs1 + 1));
  OP1(SLJIT_MOV, TMP3, 0, STR_END, 0);

  OP2U(SLJIT_SUB | SLJIT_SET_LESS, TMP1, 0, STR_END, 0);
  SELECT(SLJIT_LESS, STR_END, TMP1, 0, STR_END);
  }

OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(offs1));
add_jump(compiler, &common->failed_match, CMP(SLJIT_GREATER_EQUAL, STR_PTR, 0, STR_END, 0));

if (char1a == char1b)
  OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, char1a);
else
  {
  bit1 = char1a ^ char1b;
  if (is_powerof2(bit1))
    {
    compare1_type = vector_compare_match1i;
    OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, char1a | bit1);
    OP1(SLJIT_MOV, TMP2, 0, SLJIT_IMM, bit1);
    }
  else
    {
    compare1_type = vector_compare_match2;
    bit1 = 0;
    OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, char1a);
    OP1(SLJIT_MOV, TMP2, 0, SLJIT_IMM, char1b);
    }
  }

/* VREPLGR2VR vd, rj */
push_inst(compiler, VREPLGR2VR | VD(cmp1a_ind) | RJ_V(tmp1_reg_ind));

if (char1a != char1b)
  {
  /* VREPLGR2VR vd, rj */
  push_inst(compiler, VREPLGR2VR | VD(cmp1b_ind) | RJ_V(tmp2_reg_ind));
  }

if (char2a == char2b)
  OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, char2a);
else
  {
  bit2 = char2a ^ char2b;
  if (is_powerof2(bit2))
    {
    compare2_type = vector_compare_match1i;
    OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, char2a | bit2);
    OP1(SLJIT_MOV, TMP2, 0, SLJIT_IMM, bit2);
    }
  else
    {
    compare2_type = vector_compare_match2;
    bit2 = 0;
    OP1(SLJIT_MOV, TMP1, 0, SLJIT_IMM, char2a);
    OP1(SLJIT_MOV, TMP2, 0, SLJIT_IMM, char2b);
    }
  }

/* VREPLGR2VR vd, rj */
push_inst(compiler, VREPLGR2VR | VD(cmp2a_ind) | RJ_V(tmp1_reg_ind));

if (char2a != char2b)
  {
  /* VREPLGR2VR vd, rj */
  push_inst(compiler, VREPLGR2VR | VD(cmp2b_ind) | RJ_V(tmp2_reg_ind));
  }

#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
restart = LABEL();
#endif

OP2(SLJIT_SUB, TMP1, 0, STR_PTR, 0, SLJIT_IMM, diff);
OP1(SLJIT_MOV, TMP2, 0, STR_PTR, 0);
OP2(SLJIT_AND, TMP2, 0, TMP2, 0, SLJIT_IMM, 0xf);
OP2(SLJIT_SUB, STR_PTR, 0, STR_PTR, 0, TMP2, 0);

/* VLD vd, rj, si12 */
push_inst(compiler, VLD | VD(data1_ind) | RJ_V(str_ptr_reg_ind) | IMM_SI12(0));

jump[0] = CMP(SLJIT_GREATER_EQUAL, TMP1, 0, STR_PTR, 0);

/* VLD vd, rj, si12 */
push_inst(compiler, VLD | VD(data2_ind) | RJ_V(str_ptr_reg_ind) | IMM_SI12(-(sljit_s8)diff));
jump[1] = JUMP(SLJIT_JUMP);

JUMPHERE(jump[0]);

/* VBSLL.V vd, vj, ui5 */
push_inst(compiler, VBSLL_V | VD(data2_ind) | VJ(data1_ind) | IMM_UI5(diff));

JUMPHERE(jump[1]);

fast_forward_char_pair_lsx_compare(compiler, compare2_type, data2_ind, cmp2a_ind, cmp2b_ind, tmp2_ind);
fast_forward_char_pair_lsx_compare(compiler, compare1_type, data1_ind, cmp1a_ind, cmp1b_ind, tmp1_ind);

/* VAND vd, vj, vk */
push_inst(compiler, VOR_V | VD(data1_ind) | VJ(data1_ind) | VK(data2_ind));

/* VMSKLTZ.B vd, vj */
push_inst(compiler, VMSKLTZ_B | VD(tmp1_ind) | VJ(data1_ind));

/* VPICKVE2GR.WU rd, vj, ui2 */
push_inst(compiler, VPICKVE2GR_WU | RD_V(tmp1_reg_ind) | VJ(tmp1_ind) | IMM_UI2(0));

/* Ignore matches before the first STR_PTR. */
OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, TMP2, 0);
OP2(SLJIT_LSHR, TMP1, 0, TMP1, 0, TMP2, 0);

jump[0] = CMP(SLJIT_NOT_ZERO, TMP1, 0, SLJIT_IMM, 0);

OP2(SLJIT_SUB, STR_PTR, 0, STR_PTR, 0, TMP2, 0);

/* Main loop. */
start = LABEL();

OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, 16);
add_jump(compiler, &common->failed_match, CMP(SLJIT_GREATER_EQUAL, STR_PTR, 0, STR_END, 0));

/* VLD vd, rj, si12 */
push_inst(compiler, VLD | VD(data1_ind) | RJ_V(str_ptr_reg_ind) | IMM_SI12(0));
push_inst(compiler, VLD | VD(data2_ind) | RJ_V(str_ptr_reg_ind) | IMM_SI12(-(sljit_s8)diff));

fast_forward_char_pair_lsx_compare(compiler, compare1_type, data1_ind, cmp1a_ind, cmp1b_ind, tmp2_ind);
fast_forward_char_pair_lsx_compare(compiler, compare2_type, data2_ind, cmp2a_ind, cmp2b_ind, tmp1_ind);

/* VAND.V vd, vj, vk */
push_inst(compiler, VAND_V | VD(data1_ind) | VJ(data1_ind) | VK(data2_ind));

/* VMSKLTZ.B vd, vj */
push_inst(compiler, VMSKLTZ_B | VD(tmp1_ind) | VJ(data1_ind));

/* VPICKVE2GR.WU rd, vj, ui2 */
push_inst(compiler, VPICKVE2GR_WU | RD_V(tmp1_reg_ind) | VJ(tmp1_ind) | IMM_UI2(0));

CMPTO(SLJIT_ZERO, TMP1, 0, SLJIT_IMM, 0, start);

JUMPHERE(jump[0]);

/* CTZ.W rd, rj */
push_inst(compiler, CTZ_W | RD_V(tmp1_reg_ind) | RJ_V(tmp1_reg_ind));

OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, TMP1, 0);

add_jump(compiler, &common->failed_match, CMP(SLJIT_GREATER_EQUAL, STR_PTR, 0, STR_END, 0));

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

#endif /* SLJIT_CONFIG_LOONGARCH_64 */

#endif /* !SUPPORT_VALGRIND */
