/*************************************************
*      Perl-Compatible Regular Expressions       *
*************************************************/

/* PCRE is a library of functions to support regular expressions whose syntax
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


/* This module contains some fixed tables that are used by more than one of the
PCRE2 code modules. The tables are also #included by the pcre2test program,
which uses macros to change their names from _pcre2_xxx to xxxx, thereby
avoiding name clashes with the library. In this case, PCRE2_PCRE2TEST is
defined. */


#if !defined(PCRE2_PCRE2TEST) && !defined(PCRE2_DFTABLES) && \
    !defined(PCRE2_PCRE2POSIX) /* We're compiling the library */
#include "pcre2_internal.h"
#endif


/* Utility macros */
#define ARR_SIZE(x) sizeof(x)/sizeof(x[0])


#if !defined(PCRE2_PCRE2TEST) && !defined(PCRE2_DFTABLES) && \
    !defined(PCRE2_PCRE2POSIX)

/* Table of sizes for the fixed-length opcodes. It's defined in a macro so that
the definition is next to the definition of the opcodes in pcre2_internal.h.
This is mode-dependent, so it is skipped when this file is included by
pcre2test. */

const uint8_t PRIV(OP_lengths)[] = { OP_LENGTHS };

/* Tables of horizontal and vertical whitespace characters, suitable for
adding to classes. */

const uint32_t PRIV(hspace_list)[] = { HSPACE_LIST };
const uint32_t PRIV(vspace_list)[] = { VSPACE_LIST };

#endif /* !PCRE2_PCRE2TEST && !PCRE2_DFTABLES && !PCRE2_PCRE2POSIX */


#if !defined(PCRE2_DFTABLES) && !defined(PCRE2_PCRE2POSIX)

/* These tables are the pairs of delimiters that are valid for callout string
arguments. For each starting delimiter there must be a matching ending
delimiter, which in fact is different only for bracket-like delimiters. */

const uint32_t PRIV(callout_start_delims)[] = {
  CHAR_GRAVE_ACCENT, CHAR_APOSTROPHE, CHAR_QUOTATION_MARK,
  CHAR_CIRCUMFLEX_ACCENT, CHAR_PERCENT_SIGN, CHAR_NUMBER_SIGN,
  CHAR_DOLLAR_SIGN, CHAR_LEFT_CURLY_BRACKET, 0 };

const uint32_t PRIV(callout_end_delims[]) = {
  CHAR_GRAVE_ACCENT, CHAR_APOSTROPHE, CHAR_QUOTATION_MARK,
  CHAR_CIRCUMFLEX_ACCENT, CHAR_PERCENT_SIGN, CHAR_NUMBER_SIGN,
  CHAR_DOLLAR_SIGN, CHAR_RIGHT_CURLY_BRACKET, 0 };

#endif /* !PCRE2_DFTABLES && !PCRE2_PCRE2POSIX */


/*************************************************
*           Tables for UTF-8 support             *
*************************************************/

/* These tables are required by pcre2test in 16- or 32-bit mode, as well
as for the library in 8-bit mode, because pcre2test uses UTF-8 internally for
handling wide characters. */

#if defined PCRE2_PCRE2TEST || \
    (!defined(PCRE2_DFTABLES) && !defined(PCRE2_PCRE2POSIX) && \
     defined SUPPORT_UNICODE && \
     defined PCRE2_CODE_UNIT_WIDTH && \
     PCRE2_CODE_UNIT_WIDTH == 8)

/* These are the breakpoints for different numbers of bytes in a UTF-8
character. */

const int PRIV(utf8_table1)[] =
  { 0x7f, 0x7ff, 0xffff, 0x1fffff, 0x3ffffff, 0x7fffffff };

const unsigned PRIV(utf8_table1_size) = ARR_SIZE(PRIV(utf8_table1));

/* These are the indicator bits and the mask for the data bits to set in the
first byte of a character, indexed by the number of additional bytes. */

const int PRIV(utf8_table2)[] = { 0,    0xc0, 0xe0, 0xf0, 0xf8, 0xfc };
const int PRIV(utf8_table3)[] = { 0xff, 0x1f, 0x0f, 0x07, 0x03, 0x01 };

/* Table of the number of extra bytes, indexed by the first byte masked with
0x3f. The highest number for a valid UTF-8 first byte is in fact 0x3d. */

const uint8_t PRIV(utf8_table4)[] = {
  1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
  1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
  2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
  3,3,3,3,3,3,3,3,4,4,4,4,5,5,5,5 };

#endif /* UTF-8 support needed */

/* Tables concerned with Unicode properties are relevant only when Unicode
support is enabled. See also the pcre2_ucptables_inc.h file, which is generated by
a Python script from Unicode data files. */

#if !defined(PCRE2_DFTABLES) && !defined(PCRE2_PCRE2POSIX) && \
    defined(SUPPORT_UNICODE)

/* Table to translate from particular type value to the general value. */

const uint32_t PRIV(ucp_gentype)[] = {
  ucp_C, ucp_C, ucp_C, ucp_C, ucp_C,  /* Cc, Cf, Cn, Co, Cs */
  ucp_L, ucp_L, ucp_L, ucp_L, ucp_L,  /* Ll, Lu, Lm, Lo, Lt */
  ucp_M, ucp_M, ucp_M,                /* Mc, Me, Mn */
  ucp_N, ucp_N, ucp_N,                /* Nd, Nl, No */
  ucp_P, ucp_P, ucp_P, ucp_P, ucp_P,  /* Pc, Pd, Pe, Pf, Pi */
  ucp_P, ucp_P,                       /* Ps, Po */
  ucp_S, ucp_S, ucp_S, ucp_S,         /* Sc, Sk, Sm, So */
  ucp_Z, ucp_Z, ucp_Z                 /* Zl, Zp, Zs */
};

/* This table encodes the rules for finding the end of an extended grapheme
cluster. Every code point has a grapheme break property which is one of the
ucp_gbXX values defined in pcre2_ucp.h. These changed between Unicode versions
10 and 11. The 2-dimensional table is indexed by the properties of two adjacent
code points. The left property selects a word from the table, and the right
property selects a bit from that word like this:

  PRIV(ucp_gbtable)[left-property] & (1u << right-property)

The value is non-zero if a grapheme break is NOT permitted between the relevant
two code points. The breaking rules are as follows:

1. Break at the start and end of text (pretty obviously).

2. Do not break between a CR and LF; otherwise, break before and after
   controls.

3. Do not break Hangul syllable sequences, the rules for which are:

    L may be followed by L, V, LV or LVT
    LV or V may be followed by V or T
    LVT or T may be followed by T

4. Do not break before extending characters or zero-width-joiner (ZWJ).

The following rules are only for extended grapheme clusters (but that's what we
are implementing).

5. Do not break before SpacingMarks.

6. Do not break after Prepend characters.

7. Do not break within emoji modifier sequences or emoji zwj sequences. That
   is, do not break between characters with the Extended_Pictographic property
   if a ZWJ intervenes. Extend characters are allowed between the characters;
   this cannot be represented in this table, the code has to deal with it.

8. Do not break within emoji flag sequences. That is, do not break between
   regional indicator (RI) symbols if there are an odd number of RI characters
   before the break point. This table encodes "join RI characters"; the code
   has to deal with checking for previous adjoining RIs.

9. Otherwise, break everywhere.
*/

#define ESZ (1<<ucp_gbExtend)|(1<<ucp_gbSpacingMark)|(1<<ucp_gbZWJ)

const uint32_t PRIV(ucp_gbtable)[] = {
   (1u<<ucp_gbLF),                                      /*  0 CR */
   0,                                                   /*  1 LF */
   0,                                                   /*  2 Control */
   ESZ,                                                 /*  3 Extend */
   ESZ|(1u<<ucp_gbPrepend)|                             /*  4 Prepend */
       (1u<<ucp_gbL)|(1u<<ucp_gbV)|(1u<<ucp_gbT)|
       (1u<<ucp_gbLV)|(1u<<ucp_gbLVT)|(1u<<ucp_gbOther)|
       (1u<<ucp_gbRegional_Indicator),
   ESZ,                                                 /*  5 SpacingMark */
   ESZ|(1u<<ucp_gbL)|(1u<<ucp_gbV)|(1u<<ucp_gbLV)|      /*  6 L */
       (1u<<ucp_gbLVT),
   ESZ|(1u<<ucp_gbV)|(1u<<ucp_gbT),                     /*  7 V */
   ESZ|(1u<<ucp_gbT),                                   /*  8 T */
   ESZ|(1u<<ucp_gbV)|(1u<<ucp_gbT),                     /*  9 LV */
   ESZ|(1u<<ucp_gbT),                                   /* 10 LVT */
   (1u<<ucp_gbRegional_Indicator),                      /* 11 Regional Indicator */
   ESZ,                                                 /* 12 Other */
   ESZ|(1u<<ucp_gbExtended_Pictographic),               /* 13 ZWJ */
   ESZ                                                  /* 14 Extended Pictographic */
};

#undef ESZ

#ifdef SUPPORT_JIT
/* This table reverses PRIV(ucp_gentype). We can save the cost
of a memory load. */

const int PRIV(ucp_typerange)[] = {
  ucp_Cc, ucp_Cs,
  ucp_Ll, ucp_Lu,
  ucp_Mc, ucp_Mn,
  ucp_Nd, ucp_No,
  ucp_Pc, ucp_Ps,
  ucp_Sc, ucp_So,
  ucp_Zl, ucp_Zs,
};
#endif /* SUPPORT_JIT */

/* Finally, include the tables that are auto-generated from the Unicode data
files. */

#include "pcre2_ucptables_inc.h"

#endif /* Unicode support needed */


/*************************************************
*          Tables for EBCDIC support             *
*************************************************/

#if defined(EBCDIC) && \
  (defined(PCRE2_PCRE2TEST) || defined(PCRE2_DFTABLES) || 'a' != 0x81)

const uint8_t PRIV(ebcdic_1047_to_ascii)[256] = {
  0x00,0x01,0x02,0x03,0x9c,0x09,0x86,0x7f,0x97,0x8d,0x8e,0x0b,0x0c,0x0d,0x0e,0x0f,
#ifdef EBCDIC_NL25
  0x10,0x11,0x12,0x13,0x9d,0x85,0x08,0x87,0x18,0x19,0x92,0x8f,0x1c,0x1d,0x1e,0x1f,
  0x80,0x81,0x82,0x83,0x84,0x0a,0x17,0x1b,0x88,0x89,0x8a,0x8b,0x8c,0x05,0x06,0x07,
#else
  0x10,0x11,0x12,0x13,0x9d,0x0a,0x08,0x87,0x18,0x19,0x92,0x8f,0x1c,0x1d,0x1e,0x1f,
  0x80,0x81,0x82,0x83,0x84,0x85,0x17,0x1b,0x88,0x89,0x8a,0x8b,0x8c,0x05,0x06,0x07,
#endif
  0x90,0x91,0x16,0x93,0x94,0x95,0x96,0x04,0x98,0x99,0x9a,0x9b,0x14,0x15,0x9e,0x1a,
  0x20,0xa0,0xe2,0xe4,0xe0,0xe1,0xe3,0xe5,0xe7,0xf1,0xa2,0x2e,0x3c,0x28,0x2b,0x7c,
  0x26,0xe9,0xea,0xeb,0xe8,0xed,0xee,0xef,0xec,0xdf,0x21,0x24,0x2a,0x29,0x3b,0x5e,
  0x2d,0x2f,0xc2,0xc4,0xc0,0xc1,0xc3,0xc5,0xc7,0xd1,0xa6,0x2c,0x25,0x5f,0x3e,0x3f,
  0xf8,0xc9,0xca,0xcb,0xc8,0xcd,0xce,0xcf,0xcc,0x60,0x3a,0x23,0x40,0x27,0x3d,0x22,
  0xd8,0x61,0x62,0x63,0x64,0x65,0x66,0x67,0x68,0x69,0xab,0xbb,0xf0,0xfd,0xfe,0xb1,
  0xb0,0x6a,0x6b,0x6c,0x6d,0x6e,0x6f,0x70,0x71,0x72,0xaa,0xba,0xe6,0xb8,0xc6,0xa4,
  0xb5,0x7e,0x73,0x74,0x75,0x76,0x77,0x78,0x79,0x7a,0xa1,0xbf,0xd0,0x5b,0xde,0xae,
  0xac,0xa3,0xa5,0xb7,0xa9,0xa7,0xb6,0xbc,0xbd,0xbe,0xdd,0xa8,0xaf,0x5d,0xb4,0xd7,
  0x7b,0x41,0x42,0x43,0x44,0x45,0x46,0x47,0x48,0x49,0xad,0xf4,0xf6,0xf2,0xf3,0xf5,
  0x7d,0x4a,0x4b,0x4c,0x4d,0x4e,0x4f,0x50,0x51,0x52,0xb9,0xfb,0xfc,0xf9,0xfa,0xff,
  0x5c,0xf7,0x53,0x54,0x55,0x56,0x57,0x58,0x59,0x5a,0xb2,0xd4,0xd6,0xd2,0xd3,0xd5,
  0x30,0x31,0x32,0x33,0x34,0x35,0x36,0x37,0x38,0x39,0xb3,0xdb,0xdc,0xd9,0xda,0x9f,
};

const uint8_t PRIV(ascii_to_ebcdic_1047)[256] = {
#ifdef EBCDIC_NL25
  0x00,0x01,0x02,0x03,0x37,0x2d,0x2e,0x2f,0x16,0x05,0x25,0x0b,0x0c,0x0d,0x0e,0x0f,
#else
  0x00,0x01,0x02,0x03,0x37,0x2d,0x2e,0x2f,0x16,0x05,0x15,0x0b,0x0c,0x0d,0x0e,0x0f,
#endif
  0x10,0x11,0x12,0x13,0x3c,0x3d,0x32,0x26,0x18,0x19,0x3f,0x27,0x1c,0x1d,0x1e,0x1f,
  0x40,0x5a,0x7f,0x7b,0x5b,0x6c,0x50,0x7d,0x4d,0x5d,0x5c,0x4e,0x6b,0x60,0x4b,0x61,
  0xf0,0xf1,0xf2,0xf3,0xf4,0xf5,0xf6,0xf7,0xf8,0xf9,0x7a,0x5e,0x4c,0x7e,0x6e,0x6f,
  0x7c,0xc1,0xc2,0xc3,0xc4,0xc5,0xc6,0xc7,0xc8,0xc9,0xd1,0xd2,0xd3,0xd4,0xd5,0xd6,
  0xd7,0xd8,0xd9,0xe2,0xe3,0xe4,0xe5,0xe6,0xe7,0xe8,0xe9,0xad,0xe0,0xbd,0x5f,0x6d,
  0x79,0x81,0x82,0x83,0x84,0x85,0x86,0x87,0x88,0x89,0x91,0x92,0x93,0x94,0x95,0x96,
  0x97,0x98,0x99,0xa2,0xa3,0xa4,0xa5,0xa6,0xa7,0xa8,0xa9,0xc0,0x4f,0xd0,0xa1,0x07,
#ifdef EBCDIC_NL25
  0x20,0x21,0x22,0x23,0x24,0x15,0x06,0x17,0x28,0x29,0x2a,0x2b,0x2c,0x09,0x0a,0x1b,
#else
  0x20,0x21,0x22,0x23,0x24,0x25,0x06,0x17,0x28,0x29,0x2a,0x2b,0x2c,0x09,0x0a,0x1b,
#endif
  0x30,0x31,0x1a,0x33,0x34,0x35,0x36,0x08,0x38,0x39,0x3a,0x3b,0x04,0x14,0x3e,0xff,
  0x41,0xaa,0x4a,0xb1,0x9f,0xb2,0x6a,0xb5,0xbb,0xb4,0x9a,0x8a,0xb0,0xca,0xaf,0xbc,
  0x90,0x8f,0xea,0xfa,0xbe,0xa0,0xb6,0xb3,0x9d,0xda,0x9b,0x8b,0xb7,0xb8,0xb9,0xab,
  0x64,0x65,0x62,0x66,0x63,0x67,0x9e,0x68,0x74,0x71,0x72,0x73,0x78,0x75,0x76,0x77,
  0xac,0x69,0xed,0xee,0xeb,0xef,0xec,0xbf,0x80,0xfd,0xfe,0xfb,0xfc,0xba,0xae,0x59,
  0x44,0x45,0x42,0x46,0x43,0x47,0x9c,0x48,0x54,0x51,0x52,0x53,0x58,0x55,0x56,0x57,
  0x8c,0x49,0xcd,0xce,0xcb,0xcf,0xcc,0xe1,0x70,0xdd,0xde,0xdb,0xdc,0x8d,0x8e,0xdf,
};

#endif /* EBCDIC support needed */

/* End of pcre2_tables.c */
