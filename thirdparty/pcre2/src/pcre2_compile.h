/*************************************************
*      Perl-Compatible Regular Expressions       *
*************************************************/

/* PCRE2 is a library of functions to support regular expressions whose syntax
and semantics are as close as possible to those of the Perl 5 language.

                       Written by Philip Hazel
     Original API code Copyright (c) 1997-2012 University of Cambridge
          New API code Copyright (c) 2016-2024 University of Cambridge

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

#ifndef PCRE2_COMPILE_H_IDEMPOTENT_GUARD
#define PCRE2_COMPILE_H_IDEMPOTENT_GUARD

#include "pcre2_internal.h"

/* Compile time error code numbers. They are given names so that they can more
easily be tracked. When a new number is added, the tables called eint1 and
eint2 in pcre2posix.c may need to be updated, and a new error text must be
added to compile_error_texts in pcre2_error.c. Also, the error codes in
pcre2.h.in must be updated - their values are exactly 100 greater than these
values. */

enum { ERR0 = COMPILE_ERROR_BASE,
       ERR1,  ERR2,  ERR3,  ERR4,  ERR5,  ERR6,  ERR7,  ERR8,  ERR9,  ERR10,
       ERR11, ERR12, ERR13, ERR14, ERR15, ERR16, ERR17, ERR18, ERR19, ERR20,
       ERR21, ERR22, ERR23, ERR24, ERR25, ERR26, ERR27, ERR28, ERR29, ERR30,
       ERR31, ERR32, ERR33, ERR34, ERR35, ERR36, ERR37, ERR38, ERR39, ERR40,
       ERR41, ERR42, ERR43, ERR44, ERR45, ERR46, ERR47, ERR48, ERR49, ERR50,
       ERR51, ERR52, ERR53, ERR54, ERR55, ERR56, ERR57, ERR58, ERR59, ERR60,
       ERR61, ERR62, ERR63, ERR64, ERR65, ERR66, ERR67, ERR68, ERR69, ERR70,
       ERR71, ERR72, ERR73, ERR74, ERR75, ERR76, ERR77, ERR78, ERR79, ERR80,
       ERR81, ERR82, ERR83, ERR84, ERR85, ERR86, ERR87, ERR88, ERR89, ERR90,
       ERR91, ERR92, ERR93, ERR94, ERR95, ERR96, ERR97, ERR98, ERR99, ERR100,
       ERR101,ERR102,ERR103,ERR104,ERR105,ERR106,ERR107,ERR108,ERR109,ERR110,
       ERR111,ERR112,ERR113,ERR114,ERR115,ERR116 };

/* Code values for parsed patterns, which are stored in a vector of 32-bit
unsigned ints. Values less than META_END are literal data values. The coding
for identifying the item is in the top 16-bits, leaving 16 bits for the
additional data that some of them need. The META_CODE, META_DATA, and META_DIFF
macros are used to manipulate parsed pattern elements.

NOTE: When these definitions are changed, the table of extra lengths for each
code (meta_extra_lengths) must be updated to remain in step. */

#define META_END              0x80000000u  /* End of pattern */

#define META_ALT              0x80010000u  /* alternation */
#define META_ATOMIC           0x80020000u  /* atomic group */
#define META_BACKREF          0x80030000u  /* Back ref */
#define META_BACKREF_BYNAME   0x80040000u  /* \k'name' */
#define META_BIGVALUE         0x80050000u  /* Next is a literal > META_END */
#define META_CALLOUT_NUMBER   0x80060000u  /* (?C with numerical argument */
#define META_CALLOUT_STRING   0x80070000u  /* (?C with string argument */
#define META_CAPTURE          0x80080000u  /* Capturing parenthesis */
#define META_CIRCUMFLEX       0x80090000u  /* ^ metacharacter */
#define META_CLASS            0x800a0000u  /* start non-empty class */
#define META_CLASS_EMPTY      0x800b0000u  /* empty class */
#define META_CLASS_EMPTY_NOT  0x800c0000u  /* negative empty class */
#define META_CLASS_END        0x800d0000u  /* end of non-empty class */
#define META_CLASS_NOT        0x800e0000u  /* start non-empty negative class */
#define META_COND_ASSERT      0x800f0000u  /* (?(?assertion)... */
#define META_COND_DEFINE      0x80100000u  /* (?(DEFINE)... */
#define META_COND_NAME        0x80110000u  /* (?(<name>)... */
#define META_COND_NUMBER      0x80120000u  /* (?(digits)... */
#define META_COND_RNAME       0x80130000u  /* (?(R&name)... */
#define META_COND_RNUMBER     0x80140000u  /* (?(Rdigits)... */
#define META_COND_VERSION     0x80150000u  /* (?(VERSION<op>x.y)... */
#define META_OFFSET           0x80160000u  /* Setting offset for various
                                              META codes (e.g. META_SCS_NAME) */
#define META_SCS              0x80170000u  /* (*scan_substring:... */
#define META_SCS_NAME         0x80180000u  /* Next <name> of scan_substring */
#define META_SCS_NUMBER       0x80190000u  /* Next digits of scan_substring */
#define META_DOLLAR           0x801a0000u  /* $ metacharacter */
#define META_DOT              0x801b0000u  /* . metacharacter */
#define META_ESCAPE           0x801c0000u  /* \d and friends */
#define META_KET              0x801d0000u  /* closing parenthesis */
#define META_NOCAPTURE        0x801e0000u  /* no capture parens */
#define META_OPTIONS          0x801f0000u  /* (?i) and friends */
#define META_POSIX            0x80200000u  /* POSIX class item */
#define META_POSIX_NEG        0x80210000u  /* negative POSIX class item */
#define META_RANGE_ESCAPED    0x80220000u  /* range with at least one escape */
#define META_RANGE_LITERAL    0x80230000u  /* range defined literally */
#define META_RECURSE          0x80240000u  /* Recursion */
#define META_RECURSE_BYNAME   0x80250000u  /* (?&name) */
#define META_SCRIPT_RUN       0x80260000u  /* (*script_run:...) */

/* These must be kept together to make it easy to check that an assertion
is present where expected in a conditional group. */

#define META_LOOKAHEAD        0x80270000u  /* (?= */
#define META_LOOKAHEADNOT     0x80280000u  /* (?! */
#define META_LOOKBEHIND       0x80290000u  /* (?<= */
#define META_LOOKBEHINDNOT    0x802a0000u  /* (?<! */

/* These cannot be conditions */

#define META_LOOKAHEAD_NA     0x802b0000u  /* (*napla: */
#define META_LOOKBEHIND_NA    0x802c0000u  /* (*naplb: */

/* These must be kept in this order, with consecutive values, and the _ARG
versions of COMMIT, PRUNE, SKIP, and THEN immediately after their non-argument
versions. */

#define META_MARK             0x802d0000u  /* (*MARK) */
#define META_ACCEPT           0x802e0000u  /* (*ACCEPT) */
#define META_FAIL             0x802f0000u  /* (*FAIL) */
#define META_COMMIT           0x80300000u  /* These               */
#define META_COMMIT_ARG       0x80310000u  /*   pairs             */
#define META_PRUNE            0x80320000u  /*     must            */
#define META_PRUNE_ARG        0x80330000u  /*       be            */
#define META_SKIP             0x80340000u  /*         kept        */
#define META_SKIP_ARG         0x80350000u  /*           in        */
#define META_THEN             0x80360000u  /*             this    */
#define META_THEN_ARG         0x80370000u  /*               order */

/* These must be kept in groups of adjacent 3 values, and all together. */

#define META_ASTERISK         0x80380000u  /* *  */
#define META_ASTERISK_PLUS    0x80390000u  /* *+ */
#define META_ASTERISK_QUERY   0x803a0000u  /* *? */
#define META_PLUS             0x803b0000u  /* +  */
#define META_PLUS_PLUS        0x803c0000u  /* ++ */
#define META_PLUS_QUERY       0x803d0000u  /* +? */
#define META_QUERY            0x803e0000u  /* ?  */
#define META_QUERY_PLUS       0x803f0000u  /* ?+ */
#define META_QUERY_QUERY      0x80400000u  /* ?? */
#define META_MINMAX           0x80410000u  /* {n,m}  repeat */
#define META_MINMAX_PLUS      0x80420000u  /* {n,m}+ repeat */
#define META_MINMAX_QUERY     0x80430000u  /* {n,m}? repeat */

/* These meta codes must be kept in a group, with the OR/SUB/XOR in
this order, and AND/NOT at the start/end. */

#define META_ECLASS_AND       0x80440000u  /* && (or &) in a class */
#define META_ECLASS_OR        0x80450000u  /* || (or |, +) in a class */
#define META_ECLASS_SUB       0x80460000u  /* -- (or -) in a class */
#define META_ECLASS_XOR       0x80470000u  /* ~~ (or ^) in a class */
#define META_ECLASS_NOT       0x80480000u  /* ! in a class */

/* Convenience aliases. */

#define META_FIRST_QUANTIFIER META_ASTERISK
#define META_LAST_QUANTIFIER  META_MINMAX_QUERY

/* This is a special "meta code" that is used only to distinguish (*asr: from
(*sr: in the table of alphabetic assertions. It is never stored in the parsed
pattern because (*asr: is turned into (*sr:(*atomic: at that stage. There is
therefore no need for it to have a length entry, so use a high value. */

#define META_ATOMIC_SCRIPT_RUN 0x8fff0000u

/* Macros for manipulating elements of the parsed pattern vector. */

#define META_CODE(x)   (x & 0xffff0000u)
#define META_DATA(x)   (x & 0x0000ffffu)
#define META_DIFF(x,y) ((x-y)>>16)

/* Extended class management flags. */

#define CLASS_IS_ECLASS 0x1

/* Macro for the highest character value. */

#if PCRE2_CODE_UNIT_WIDTH == 8
#define MAX_UCHAR_VALUE 0xffu
#elif PCRE2_CODE_UNIT_WIDTH == 16
#define MAX_UCHAR_VALUE 0xffffu
#else
#define MAX_UCHAR_VALUE 0xffffffffu
#endif

#define GET_MAX_CHAR_VALUE(utf) \
  ((utf) ? MAX_UTF_CODE_POINT : MAX_UCHAR_VALUE)

/* Macro for setting individual bits in class bitmaps. */

#define SETBIT(a,b) a[(b) >> 3] |= (uint8_t)(1u << ((b) & 0x7))

/* Macro for 8 bit specific checks. */
#if PCRE2_CODE_UNIT_WIDTH == 8
#define SELECT_VALUE8(value8, value) (value8)
#else
#define SELECT_VALUE8(value8, value) (value)
#endif

/* Macro for aligning data. */
#define CLIST_ALIGN_TO(base, align) \
  ((base + ((size_t)(align) - 1)) & ~((size_t)(align) - 1))

/* Structure for holding information about an OP_ECLASS internal operand.
An "operand" here could be just a single OP_[X]CLASS, or it could be some
complex expression; but it's some sequence of ECL_* codes which pushes one
value to the stack. */
typedef struct {
  /* The position of the operand - or NULL if (lengthptr != NULL). */
  PCRE2_UCHAR *code_start;
  PCRE2_SIZE length;
  /* The operand's type if it is a single code (ECL_XCLASS, ECL_ANY, ECL_NONE);
  otherwise zero if the operand is not atomic. */
  uint8_t op_single_type;
  /* Regardless of whether it's a single code or not, we fully constant-fold
  the bitmap for code points < 256. */
  class_bits_storage bits;
} eclass_op_info;

/* Macros for the definitions below, to prevent name collisions. */

#define _pcre2_posix_class_maps          PCRE2_SUFFIX(_pcre2_posix_class_maps)
#define _pcre2_update_classbits          PCRE2_SUFFIX(_pcre2_update_classbits_)
#define _pcre2_compile_class_nested      PCRE2_SUFFIX(_pcre2_compile_class_nested_)
#define _pcre2_compile_class_not_nested  PCRE2_SUFFIX(_pcre2_compile_class_not_nested_)


/* Indices of the POSIX classes in posix_names, posix_name_lengths,
posix_class_maps, and posix_substitutes. They must be kept in sync. */

#define PC_DIGIT   7
#define PC_GRAPH   8
#define PC_PRINT   9
#define PC_PUNCT  10
#define PC_XDIGIT 13

extern const int PRIV(posix_class_maps)[];


/* Set bits in classbits according to the property type */

void PRIV(update_classbits)(uint32_t ptype, uint32_t pdata, BOOL negated,
  uint8_t *classbits);

/* Compile the META codes from start_ptr...end_ptr, writing a single OP_CLASS
OP_CLASS, OP_NCLASS, OP_XCLASS, or OP_ALLANY into pcode. */

uint32_t *PRIV(compile_class_not_nested)(uint32_t options, uint32_t xoptions,
  uint32_t *start_ptr, PCRE2_UCHAR **pcode, BOOL negate_class, BOOL* has_bitmap,
  int *errorcodeptr, compile_block *cb, PCRE2_SIZE *lengthptr);

/* Compile the META codes in pptr into opcodes written to pcode. The pptr must
start at a META_CLASS or META_CLASS_NOT.

The pptr will be left pointing at the matching META_CLASS_END. */

BOOL PRIV(compile_class_nested)(uint32_t options, uint32_t xoptions,
  uint32_t **pptr, PCRE2_UCHAR **pcode, int *errorcodeptr,
  compile_block *cb, PCRE2_SIZE *lengthptr);

#endif  /* PCRE2_COMPILE_H_IDEMPOTENT_GUARD */

/* End of pcre2_compile.h */
