/*************************************************
*      Perl-Compatible Regular Expressions       *
*************************************************/

/* PCRE is a library of functions to support regular expressions whose syntax
and semantics are as close as possible to those of the Perl 5 language.

                       Written by Philip Hazel
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

/* This module contains some fixed tables that are used by more than one of the
PCRE2 code modules. The tables are also #included by the pcre2test program,
which uses macros to change their names from _pcre2_xxx to xxxx, thereby
avoiding name clashes with the library. In this case, PCRE2_PCRE2TEST is
defined. */

#ifndef PCRE2_PCRE2TEST           /* We're compiling the library */
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "pcre2_internal.h"
#endif /* PCRE2_PCRE2TEST */


/* Table of sizes for the fixed-length opcodes. It's defined in a macro so that
the definition is next to the definition of the opcodes in pcre2_internal.h.
This is mode-dependent, so is skipped when this file is included by pcre2test. */

#ifndef PCRE2_PCRE2TEST
const uint8_t PRIV(OP_lengths)[] = { OP_LENGTHS };
#endif

/* Tables of horizontal and vertical whitespace characters, suitable for
adding to classes. */

const uint32_t PRIV(hspace_list)[] = { HSPACE_LIST };
const uint32_t PRIV(vspace_list)[] = { VSPACE_LIST };

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


/*************************************************
*           Tables for UTF-8 support             *
*************************************************/

/* These tables are required by pcre2test in 16- or 32-bit mode, as well
as for the library in 8-bit mode, because pcre2test uses UTF-8 internally for
handling wide characters. */

#if defined PCRE2_PCRE2TEST || \
   (defined SUPPORT_UNICODE && \
    defined PCRE2_CODE_UNIT_WIDTH && \
    PCRE2_CODE_UNIT_WIDTH == 8)

/* These are the breakpoints for different numbers of bytes in a UTF-8
character. */

const int PRIV(utf8_table1)[] =
  { 0x7f, 0x7ff, 0xffff, 0x1fffff, 0x3ffffff, 0x7fffffff};

const int PRIV(utf8_table1_size) = sizeof(PRIV(utf8_table1)) / sizeof(int);

/* These are the indicator bits and the mask for the data bits to set in the
first byte of a character, indexed by the number of additional bytes. */

const int PRIV(utf8_table2)[] = { 0,    0xc0, 0xe0, 0xf0, 0xf8, 0xfc};
const int PRIV(utf8_table3)[] = { 0xff, 0x1f, 0x0f, 0x07, 0x03, 0x01};

/* Table of the number of extra bytes, indexed by the first byte masked with
0x3f. The highest number for a valid UTF-8 first byte is in fact 0x3d. */

const uint8_t PRIV(utf8_table4)[] = {
  1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
  1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
  2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
  3,3,3,3,3,3,3,3,4,4,4,4,5,5,5,5 };

#endif /* UTF-8 support needed */


#ifdef SUPPORT_UNICODE

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
   is, do not break between characters with the Extended_Pictographic property.
   Extend and ZWJ characters are allowed between the characters; this cannot be
   represented in this table, the code has to deal with it.

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
       (1u<<ucp_gbRegionalIndicator),
   ESZ,                                                 /*  5 SpacingMark */
   ESZ|(1u<<ucp_gbL)|(1u<<ucp_gbV)|(1u<<ucp_gbLV)|      /*  6 L */
       (1u<<ucp_gbLVT),
   ESZ|(1u<<ucp_gbV)|(1u<<ucp_gbT),                     /*  7 V */
   ESZ|(1u<<ucp_gbT),                                   /*  8 T */
   ESZ|(1u<<ucp_gbV)|(1u<<ucp_gbT),                     /*  9 LV */
   ESZ|(1u<<ucp_gbT),                                   /* 10 LVT */
   (1u<<ucp_gbRegionalIndicator),                       /* 11 RegionalIndicator */
   ESZ,                                                 /* 12 Other */
   ESZ,                                                 /* 13 ZWJ */
   ESZ|(1u<<ucp_gbExtended_Pictographic)                /* 14 Extended Pictographic */
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

/* The PRIV(utt)[] table below translates Unicode property names into type and
code values. It is searched by binary chop, so must be in collating sequence of
name. Originally, the table contained pointers to the name strings in the first
field of each entry. However, that leads to a large number of relocations when
a shared library is dynamically loaded. A significant reduction is made by
putting all the names into a single, large string and then using offsets in the
table itself. Maintenance is more error-prone, but frequent changes to this
data are unlikely.

July 2008: There is now a script called maint/GenerateUtt.py that can be used
to generate this data automatically instead of maintaining it by hand.

The script was updated in March 2009 to generate a new EBCDIC-compliant
version. Like all other character and string literals that are compared against
the regular expression pattern, we must use STR_ macros instead of literal
strings to make sure that UTF-8 support works on EBCDIC platforms. */

#define STRING_Adlam0 STR_A STR_d STR_l STR_a STR_m "\0"
#define STRING_Ahom0 STR_A STR_h STR_o STR_m "\0"
#define STRING_Anatolian_Hieroglyphs0 STR_A STR_n STR_a STR_t STR_o STR_l STR_i STR_a STR_n STR_UNDERSCORE STR_H STR_i STR_e STR_r STR_o STR_g STR_l STR_y STR_p STR_h STR_s "\0"
#define STRING_Any0 STR_A STR_n STR_y "\0"
#define STRING_Arabic0 STR_A STR_r STR_a STR_b STR_i STR_c "\0"
#define STRING_Armenian0 STR_A STR_r STR_m STR_e STR_n STR_i STR_a STR_n "\0"
#define STRING_Avestan0 STR_A STR_v STR_e STR_s STR_t STR_a STR_n "\0"
#define STRING_Balinese0 STR_B STR_a STR_l STR_i STR_n STR_e STR_s STR_e "\0"
#define STRING_Bamum0 STR_B STR_a STR_m STR_u STR_m "\0"
#define STRING_Bassa_Vah0 STR_B STR_a STR_s STR_s STR_a STR_UNDERSCORE STR_V STR_a STR_h "\0"
#define STRING_Batak0 STR_B STR_a STR_t STR_a STR_k "\0"
#define STRING_Bengali0 STR_B STR_e STR_n STR_g STR_a STR_l STR_i "\0"
#define STRING_Bhaiksuki0 STR_B STR_h STR_a STR_i STR_k STR_s STR_u STR_k STR_i "\0"
#define STRING_Bopomofo0 STR_B STR_o STR_p STR_o STR_m STR_o STR_f STR_o "\0"
#define STRING_Brahmi0 STR_B STR_r STR_a STR_h STR_m STR_i "\0"
#define STRING_Braille0 STR_B STR_r STR_a STR_i STR_l STR_l STR_e "\0"
#define STRING_Buginese0 STR_B STR_u STR_g STR_i STR_n STR_e STR_s STR_e "\0"
#define STRING_Buhid0 STR_B STR_u STR_h STR_i STR_d "\0"
#define STRING_C0 STR_C "\0"
#define STRING_Canadian_Aboriginal0 STR_C STR_a STR_n STR_a STR_d STR_i STR_a STR_n STR_UNDERSCORE STR_A STR_b STR_o STR_r STR_i STR_g STR_i STR_n STR_a STR_l "\0"
#define STRING_Carian0 STR_C STR_a STR_r STR_i STR_a STR_n "\0"
#define STRING_Caucasian_Albanian0 STR_C STR_a STR_u STR_c STR_a STR_s STR_i STR_a STR_n STR_UNDERSCORE STR_A STR_l STR_b STR_a STR_n STR_i STR_a STR_n "\0"
#define STRING_Cc0 STR_C STR_c "\0"
#define STRING_Cf0 STR_C STR_f "\0"
#define STRING_Chakma0 STR_C STR_h STR_a STR_k STR_m STR_a "\0"
#define STRING_Cham0 STR_C STR_h STR_a STR_m "\0"
#define STRING_Cherokee0 STR_C STR_h STR_e STR_r STR_o STR_k STR_e STR_e "\0"
#define STRING_Cn0 STR_C STR_n "\0"
#define STRING_Co0 STR_C STR_o "\0"
#define STRING_Common0 STR_C STR_o STR_m STR_m STR_o STR_n "\0"
#define STRING_Coptic0 STR_C STR_o STR_p STR_t STR_i STR_c "\0"
#define STRING_Cs0 STR_C STR_s "\0"
#define STRING_Cuneiform0 STR_C STR_u STR_n STR_e STR_i STR_f STR_o STR_r STR_m "\0"
#define STRING_Cypriot0 STR_C STR_y STR_p STR_r STR_i STR_o STR_t "\0"
#define STRING_Cyrillic0 STR_C STR_y STR_r STR_i STR_l STR_l STR_i STR_c "\0"
#define STRING_Deseret0 STR_D STR_e STR_s STR_e STR_r STR_e STR_t "\0"
#define STRING_Devanagari0 STR_D STR_e STR_v STR_a STR_n STR_a STR_g STR_a STR_r STR_i "\0"
#define STRING_Dogra0 STR_D STR_o STR_g STR_r STR_a "\0"
#define STRING_Duployan0 STR_D STR_u STR_p STR_l STR_o STR_y STR_a STR_n "\0"
#define STRING_Egyptian_Hieroglyphs0 STR_E STR_g STR_y STR_p STR_t STR_i STR_a STR_n STR_UNDERSCORE STR_H STR_i STR_e STR_r STR_o STR_g STR_l STR_y STR_p STR_h STR_s "\0"
#define STRING_Elbasan0 STR_E STR_l STR_b STR_a STR_s STR_a STR_n "\0"
#define STRING_Elymaic0 STR_E STR_l STR_y STR_m STR_a STR_i STR_c "\0"
#define STRING_Ethiopic0 STR_E STR_t STR_h STR_i STR_o STR_p STR_i STR_c "\0"
#define STRING_Georgian0 STR_G STR_e STR_o STR_r STR_g STR_i STR_a STR_n "\0"
#define STRING_Glagolitic0 STR_G STR_l STR_a STR_g STR_o STR_l STR_i STR_t STR_i STR_c "\0"
#define STRING_Gothic0 STR_G STR_o STR_t STR_h STR_i STR_c "\0"
#define STRING_Grantha0 STR_G STR_r STR_a STR_n STR_t STR_h STR_a "\0"
#define STRING_Greek0 STR_G STR_r STR_e STR_e STR_k "\0"
#define STRING_Gujarati0 STR_G STR_u STR_j STR_a STR_r STR_a STR_t STR_i "\0"
#define STRING_Gunjala_Gondi0 STR_G STR_u STR_n STR_j STR_a STR_l STR_a STR_UNDERSCORE STR_G STR_o STR_n STR_d STR_i "\0"
#define STRING_Gurmukhi0 STR_G STR_u STR_r STR_m STR_u STR_k STR_h STR_i "\0"
#define STRING_Han0 STR_H STR_a STR_n "\0"
#define STRING_Hangul0 STR_H STR_a STR_n STR_g STR_u STR_l "\0"
#define STRING_Hanifi_Rohingya0 STR_H STR_a STR_n STR_i STR_f STR_i STR_UNDERSCORE STR_R STR_o STR_h STR_i STR_n STR_g STR_y STR_a "\0"
#define STRING_Hanunoo0 STR_H STR_a STR_n STR_u STR_n STR_o STR_o "\0"
#define STRING_Hatran0 STR_H STR_a STR_t STR_r STR_a STR_n "\0"
#define STRING_Hebrew0 STR_H STR_e STR_b STR_r STR_e STR_w "\0"
#define STRING_Hiragana0 STR_H STR_i STR_r STR_a STR_g STR_a STR_n STR_a "\0"
#define STRING_Imperial_Aramaic0 STR_I STR_m STR_p STR_e STR_r STR_i STR_a STR_l STR_UNDERSCORE STR_A STR_r STR_a STR_m STR_a STR_i STR_c "\0"
#define STRING_Inherited0 STR_I STR_n STR_h STR_e STR_r STR_i STR_t STR_e STR_d "\0"
#define STRING_Inscriptional_Pahlavi0 STR_I STR_n STR_s STR_c STR_r STR_i STR_p STR_t STR_i STR_o STR_n STR_a STR_l STR_UNDERSCORE STR_P STR_a STR_h STR_l STR_a STR_v STR_i "\0"
#define STRING_Inscriptional_Parthian0 STR_I STR_n STR_s STR_c STR_r STR_i STR_p STR_t STR_i STR_o STR_n STR_a STR_l STR_UNDERSCORE STR_P STR_a STR_r STR_t STR_h STR_i STR_a STR_n "\0"
#define STRING_Javanese0 STR_J STR_a STR_v STR_a STR_n STR_e STR_s STR_e "\0"
#define STRING_Kaithi0 STR_K STR_a STR_i STR_t STR_h STR_i "\0"
#define STRING_Kannada0 STR_K STR_a STR_n STR_n STR_a STR_d STR_a "\0"
#define STRING_Katakana0 STR_K STR_a STR_t STR_a STR_k STR_a STR_n STR_a "\0"
#define STRING_Kayah_Li0 STR_K STR_a STR_y STR_a STR_h STR_UNDERSCORE STR_L STR_i "\0"
#define STRING_Kharoshthi0 STR_K STR_h STR_a STR_r STR_o STR_s STR_h STR_t STR_h STR_i "\0"
#define STRING_Khmer0 STR_K STR_h STR_m STR_e STR_r "\0"
#define STRING_Khojki0 STR_K STR_h STR_o STR_j STR_k STR_i "\0"
#define STRING_Khudawadi0 STR_K STR_h STR_u STR_d STR_a STR_w STR_a STR_d STR_i "\0"
#define STRING_L0 STR_L "\0"
#define STRING_L_AMPERSAND0 STR_L STR_AMPERSAND "\0"
#define STRING_Lao0 STR_L STR_a STR_o "\0"
#define STRING_Latin0 STR_L STR_a STR_t STR_i STR_n "\0"
#define STRING_Lepcha0 STR_L STR_e STR_p STR_c STR_h STR_a "\0"
#define STRING_Limbu0 STR_L STR_i STR_m STR_b STR_u "\0"
#define STRING_Linear_A0 STR_L STR_i STR_n STR_e STR_a STR_r STR_UNDERSCORE STR_A "\0"
#define STRING_Linear_B0 STR_L STR_i STR_n STR_e STR_a STR_r STR_UNDERSCORE STR_B "\0"
#define STRING_Lisu0 STR_L STR_i STR_s STR_u "\0"
#define STRING_Ll0 STR_L STR_l "\0"
#define STRING_Lm0 STR_L STR_m "\0"
#define STRING_Lo0 STR_L STR_o "\0"
#define STRING_Lt0 STR_L STR_t "\0"
#define STRING_Lu0 STR_L STR_u "\0"
#define STRING_Lycian0 STR_L STR_y STR_c STR_i STR_a STR_n "\0"
#define STRING_Lydian0 STR_L STR_y STR_d STR_i STR_a STR_n "\0"
#define STRING_M0 STR_M "\0"
#define STRING_Mahajani0 STR_M STR_a STR_h STR_a STR_j STR_a STR_n STR_i "\0"
#define STRING_Makasar0 STR_M STR_a STR_k STR_a STR_s STR_a STR_r "\0"
#define STRING_Malayalam0 STR_M STR_a STR_l STR_a STR_y STR_a STR_l STR_a STR_m "\0"
#define STRING_Mandaic0 STR_M STR_a STR_n STR_d STR_a STR_i STR_c "\0"
#define STRING_Manichaean0 STR_M STR_a STR_n STR_i STR_c STR_h STR_a STR_e STR_a STR_n "\0"
#define STRING_Marchen0 STR_M STR_a STR_r STR_c STR_h STR_e STR_n "\0"
#define STRING_Masaram_Gondi0 STR_M STR_a STR_s STR_a STR_r STR_a STR_m STR_UNDERSCORE STR_G STR_o STR_n STR_d STR_i "\0"
#define STRING_Mc0 STR_M STR_c "\0"
#define STRING_Me0 STR_M STR_e "\0"
#define STRING_Medefaidrin0 STR_M STR_e STR_d STR_e STR_f STR_a STR_i STR_d STR_r STR_i STR_n "\0"
#define STRING_Meetei_Mayek0 STR_M STR_e STR_e STR_t STR_e STR_i STR_UNDERSCORE STR_M STR_a STR_y STR_e STR_k "\0"
#define STRING_Mende_Kikakui0 STR_M STR_e STR_n STR_d STR_e STR_UNDERSCORE STR_K STR_i STR_k STR_a STR_k STR_u STR_i "\0"
#define STRING_Meroitic_Cursive0 STR_M STR_e STR_r STR_o STR_i STR_t STR_i STR_c STR_UNDERSCORE STR_C STR_u STR_r STR_s STR_i STR_v STR_e "\0"
#define STRING_Meroitic_Hieroglyphs0 STR_M STR_e STR_r STR_o STR_i STR_t STR_i STR_c STR_UNDERSCORE STR_H STR_i STR_e STR_r STR_o STR_g STR_l STR_y STR_p STR_h STR_s "\0"
#define STRING_Miao0 STR_M STR_i STR_a STR_o "\0"
#define STRING_Mn0 STR_M STR_n "\0"
#define STRING_Modi0 STR_M STR_o STR_d STR_i "\0"
#define STRING_Mongolian0 STR_M STR_o STR_n STR_g STR_o STR_l STR_i STR_a STR_n "\0"
#define STRING_Mro0 STR_M STR_r STR_o "\0"
#define STRING_Multani0 STR_M STR_u STR_l STR_t STR_a STR_n STR_i "\0"
#define STRING_Myanmar0 STR_M STR_y STR_a STR_n STR_m STR_a STR_r "\0"
#define STRING_N0 STR_N "\0"
#define STRING_Nabataean0 STR_N STR_a STR_b STR_a STR_t STR_a STR_e STR_a STR_n "\0"
#define STRING_Nandinagari0 STR_N STR_a STR_n STR_d STR_i STR_n STR_a STR_g STR_a STR_r STR_i "\0"
#define STRING_Nd0 STR_N STR_d "\0"
#define STRING_New_Tai_Lue0 STR_N STR_e STR_w STR_UNDERSCORE STR_T STR_a STR_i STR_UNDERSCORE STR_L STR_u STR_e "\0"
#define STRING_Newa0 STR_N STR_e STR_w STR_a "\0"
#define STRING_Nko0 STR_N STR_k STR_o "\0"
#define STRING_Nl0 STR_N STR_l "\0"
#define STRING_No0 STR_N STR_o "\0"
#define STRING_Nushu0 STR_N STR_u STR_s STR_h STR_u "\0"
#define STRING_Nyiakeng_Puachue_Hmong0 STR_N STR_y STR_i STR_a STR_k STR_e STR_n STR_g STR_UNDERSCORE STR_P STR_u STR_a STR_c STR_h STR_u STR_e STR_UNDERSCORE STR_H STR_m STR_o STR_n STR_g "\0"
#define STRING_Ogham0 STR_O STR_g STR_h STR_a STR_m "\0"
#define STRING_Ol_Chiki0 STR_O STR_l STR_UNDERSCORE STR_C STR_h STR_i STR_k STR_i "\0"
#define STRING_Old_Hungarian0 STR_O STR_l STR_d STR_UNDERSCORE STR_H STR_u STR_n STR_g STR_a STR_r STR_i STR_a STR_n "\0"
#define STRING_Old_Italic0 STR_O STR_l STR_d STR_UNDERSCORE STR_I STR_t STR_a STR_l STR_i STR_c "\0"
#define STRING_Old_North_Arabian0 STR_O STR_l STR_d STR_UNDERSCORE STR_N STR_o STR_r STR_t STR_h STR_UNDERSCORE STR_A STR_r STR_a STR_b STR_i STR_a STR_n "\0"
#define STRING_Old_Permic0 STR_O STR_l STR_d STR_UNDERSCORE STR_P STR_e STR_r STR_m STR_i STR_c "\0"
#define STRING_Old_Persian0 STR_O STR_l STR_d STR_UNDERSCORE STR_P STR_e STR_r STR_s STR_i STR_a STR_n "\0"
#define STRING_Old_Sogdian0 STR_O STR_l STR_d STR_UNDERSCORE STR_S STR_o STR_g STR_d STR_i STR_a STR_n "\0"
#define STRING_Old_South_Arabian0 STR_O STR_l STR_d STR_UNDERSCORE STR_S STR_o STR_u STR_t STR_h STR_UNDERSCORE STR_A STR_r STR_a STR_b STR_i STR_a STR_n "\0"
#define STRING_Old_Turkic0 STR_O STR_l STR_d STR_UNDERSCORE STR_T STR_u STR_r STR_k STR_i STR_c "\0"
#define STRING_Oriya0 STR_O STR_r STR_i STR_y STR_a "\0"
#define STRING_Osage0 STR_O STR_s STR_a STR_g STR_e "\0"
#define STRING_Osmanya0 STR_O STR_s STR_m STR_a STR_n STR_y STR_a "\0"
#define STRING_P0 STR_P "\0"
#define STRING_Pahawh_Hmong0 STR_P STR_a STR_h STR_a STR_w STR_h STR_UNDERSCORE STR_H STR_m STR_o STR_n STR_g "\0"
#define STRING_Palmyrene0 STR_P STR_a STR_l STR_m STR_y STR_r STR_e STR_n STR_e "\0"
#define STRING_Pau_Cin_Hau0 STR_P STR_a STR_u STR_UNDERSCORE STR_C STR_i STR_n STR_UNDERSCORE STR_H STR_a STR_u "\0"
#define STRING_Pc0 STR_P STR_c "\0"
#define STRING_Pd0 STR_P STR_d "\0"
#define STRING_Pe0 STR_P STR_e "\0"
#define STRING_Pf0 STR_P STR_f "\0"
#define STRING_Phags_Pa0 STR_P STR_h STR_a STR_g STR_s STR_UNDERSCORE STR_P STR_a "\0"
#define STRING_Phoenician0 STR_P STR_h STR_o STR_e STR_n STR_i STR_c STR_i STR_a STR_n "\0"
#define STRING_Pi0 STR_P STR_i "\0"
#define STRING_Po0 STR_P STR_o "\0"
#define STRING_Ps0 STR_P STR_s "\0"
#define STRING_Psalter_Pahlavi0 STR_P STR_s STR_a STR_l STR_t STR_e STR_r STR_UNDERSCORE STR_P STR_a STR_h STR_l STR_a STR_v STR_i "\0"
#define STRING_Rejang0 STR_R STR_e STR_j STR_a STR_n STR_g "\0"
#define STRING_Runic0 STR_R STR_u STR_n STR_i STR_c "\0"
#define STRING_S0 STR_S "\0"
#define STRING_Samaritan0 STR_S STR_a STR_m STR_a STR_r STR_i STR_t STR_a STR_n "\0"
#define STRING_Saurashtra0 STR_S STR_a STR_u STR_r STR_a STR_s STR_h STR_t STR_r STR_a "\0"
#define STRING_Sc0 STR_S STR_c "\0"
#define STRING_Sharada0 STR_S STR_h STR_a STR_r STR_a STR_d STR_a "\0"
#define STRING_Shavian0 STR_S STR_h STR_a STR_v STR_i STR_a STR_n "\0"
#define STRING_Siddham0 STR_S STR_i STR_d STR_d STR_h STR_a STR_m "\0"
#define STRING_SignWriting0 STR_S STR_i STR_g STR_n STR_W STR_r STR_i STR_t STR_i STR_n STR_g "\0"
#define STRING_Sinhala0 STR_S STR_i STR_n STR_h STR_a STR_l STR_a "\0"
#define STRING_Sk0 STR_S STR_k "\0"
#define STRING_Sm0 STR_S STR_m "\0"
#define STRING_So0 STR_S STR_o "\0"
#define STRING_Sogdian0 STR_S STR_o STR_g STR_d STR_i STR_a STR_n "\0"
#define STRING_Sora_Sompeng0 STR_S STR_o STR_r STR_a STR_UNDERSCORE STR_S STR_o STR_m STR_p STR_e STR_n STR_g "\0"
#define STRING_Soyombo0 STR_S STR_o STR_y STR_o STR_m STR_b STR_o "\0"
#define STRING_Sundanese0 STR_S STR_u STR_n STR_d STR_a STR_n STR_e STR_s STR_e "\0"
#define STRING_Syloti_Nagri0 STR_S STR_y STR_l STR_o STR_t STR_i STR_UNDERSCORE STR_N STR_a STR_g STR_r STR_i "\0"
#define STRING_Syriac0 STR_S STR_y STR_r STR_i STR_a STR_c "\0"
#define STRING_Tagalog0 STR_T STR_a STR_g STR_a STR_l STR_o STR_g "\0"
#define STRING_Tagbanwa0 STR_T STR_a STR_g STR_b STR_a STR_n STR_w STR_a "\0"
#define STRING_Tai_Le0 STR_T STR_a STR_i STR_UNDERSCORE STR_L STR_e "\0"
#define STRING_Tai_Tham0 STR_T STR_a STR_i STR_UNDERSCORE STR_T STR_h STR_a STR_m "\0"
#define STRING_Tai_Viet0 STR_T STR_a STR_i STR_UNDERSCORE STR_V STR_i STR_e STR_t "\0"
#define STRING_Takri0 STR_T STR_a STR_k STR_r STR_i "\0"
#define STRING_Tamil0 STR_T STR_a STR_m STR_i STR_l "\0"
#define STRING_Tangut0 STR_T STR_a STR_n STR_g STR_u STR_t "\0"
#define STRING_Telugu0 STR_T STR_e STR_l STR_u STR_g STR_u "\0"
#define STRING_Thaana0 STR_T STR_h STR_a STR_a STR_n STR_a "\0"
#define STRING_Thai0 STR_T STR_h STR_a STR_i "\0"
#define STRING_Tibetan0 STR_T STR_i STR_b STR_e STR_t STR_a STR_n "\0"
#define STRING_Tifinagh0 STR_T STR_i STR_f STR_i STR_n STR_a STR_g STR_h "\0"
#define STRING_Tirhuta0 STR_T STR_i STR_r STR_h STR_u STR_t STR_a "\0"
#define STRING_Ugaritic0 STR_U STR_g STR_a STR_r STR_i STR_t STR_i STR_c "\0"
#define STRING_Unknown0 STR_U STR_n STR_k STR_n STR_o STR_w STR_n "\0"
#define STRING_Vai0 STR_V STR_a STR_i "\0"
#define STRING_Wancho0 STR_W STR_a STR_n STR_c STR_h STR_o "\0"
#define STRING_Warang_Citi0 STR_W STR_a STR_r STR_a STR_n STR_g STR_UNDERSCORE STR_C STR_i STR_t STR_i "\0"
#define STRING_Xan0 STR_X STR_a STR_n "\0"
#define STRING_Xps0 STR_X STR_p STR_s "\0"
#define STRING_Xsp0 STR_X STR_s STR_p "\0"
#define STRING_Xuc0 STR_X STR_u STR_c "\0"
#define STRING_Xwd0 STR_X STR_w STR_d "\0"
#define STRING_Yi0 STR_Y STR_i "\0"
#define STRING_Z0 STR_Z "\0"
#define STRING_Zanabazar_Square0 STR_Z STR_a STR_n STR_a STR_b STR_a STR_z STR_a STR_r STR_UNDERSCORE STR_S STR_q STR_u STR_a STR_r STR_e "\0"
#define STRING_Zl0 STR_Z STR_l "\0"
#define STRING_Zp0 STR_Z STR_p "\0"
#define STRING_Zs0 STR_Z STR_s "\0"

const char PRIV(utt_names)[] =
  STRING_Adlam0
  STRING_Ahom0
  STRING_Anatolian_Hieroglyphs0
  STRING_Any0
  STRING_Arabic0
  STRING_Armenian0
  STRING_Avestan0
  STRING_Balinese0
  STRING_Bamum0
  STRING_Bassa_Vah0
  STRING_Batak0
  STRING_Bengali0
  STRING_Bhaiksuki0
  STRING_Bopomofo0
  STRING_Brahmi0
  STRING_Braille0
  STRING_Buginese0
  STRING_Buhid0
  STRING_C0
  STRING_Canadian_Aboriginal0
  STRING_Carian0
  STRING_Caucasian_Albanian0
  STRING_Cc0
  STRING_Cf0
  STRING_Chakma0
  STRING_Cham0
  STRING_Cherokee0
  STRING_Cn0
  STRING_Co0
  STRING_Common0
  STRING_Coptic0
  STRING_Cs0
  STRING_Cuneiform0
  STRING_Cypriot0
  STRING_Cyrillic0
  STRING_Deseret0
  STRING_Devanagari0
  STRING_Dogra0
  STRING_Duployan0
  STRING_Egyptian_Hieroglyphs0
  STRING_Elbasan0
  STRING_Elymaic0
  STRING_Ethiopic0
  STRING_Georgian0
  STRING_Glagolitic0
  STRING_Gothic0
  STRING_Grantha0
  STRING_Greek0
  STRING_Gujarati0
  STRING_Gunjala_Gondi0
  STRING_Gurmukhi0
  STRING_Han0
  STRING_Hangul0
  STRING_Hanifi_Rohingya0
  STRING_Hanunoo0
  STRING_Hatran0
  STRING_Hebrew0
  STRING_Hiragana0
  STRING_Imperial_Aramaic0
  STRING_Inherited0
  STRING_Inscriptional_Pahlavi0
  STRING_Inscriptional_Parthian0
  STRING_Javanese0
  STRING_Kaithi0
  STRING_Kannada0
  STRING_Katakana0
  STRING_Kayah_Li0
  STRING_Kharoshthi0
  STRING_Khmer0
  STRING_Khojki0
  STRING_Khudawadi0
  STRING_L0
  STRING_L_AMPERSAND0
  STRING_Lao0
  STRING_Latin0
  STRING_Lepcha0
  STRING_Limbu0
  STRING_Linear_A0
  STRING_Linear_B0
  STRING_Lisu0
  STRING_Ll0
  STRING_Lm0
  STRING_Lo0
  STRING_Lt0
  STRING_Lu0
  STRING_Lycian0
  STRING_Lydian0
  STRING_M0
  STRING_Mahajani0
  STRING_Makasar0
  STRING_Malayalam0
  STRING_Mandaic0
  STRING_Manichaean0
  STRING_Marchen0
  STRING_Masaram_Gondi0
  STRING_Mc0
  STRING_Me0
  STRING_Medefaidrin0
  STRING_Meetei_Mayek0
  STRING_Mende_Kikakui0
  STRING_Meroitic_Cursive0
  STRING_Meroitic_Hieroglyphs0
  STRING_Miao0
  STRING_Mn0
  STRING_Modi0
  STRING_Mongolian0
  STRING_Mro0
  STRING_Multani0
  STRING_Myanmar0
  STRING_N0
  STRING_Nabataean0
  STRING_Nandinagari0
  STRING_Nd0
  STRING_New_Tai_Lue0
  STRING_Newa0
  STRING_Nko0
  STRING_Nl0
  STRING_No0
  STRING_Nushu0
  STRING_Nyiakeng_Puachue_Hmong0
  STRING_Ogham0
  STRING_Ol_Chiki0
  STRING_Old_Hungarian0
  STRING_Old_Italic0
  STRING_Old_North_Arabian0
  STRING_Old_Permic0
  STRING_Old_Persian0
  STRING_Old_Sogdian0
  STRING_Old_South_Arabian0
  STRING_Old_Turkic0
  STRING_Oriya0
  STRING_Osage0
  STRING_Osmanya0
  STRING_P0
  STRING_Pahawh_Hmong0
  STRING_Palmyrene0
  STRING_Pau_Cin_Hau0
  STRING_Pc0
  STRING_Pd0
  STRING_Pe0
  STRING_Pf0
  STRING_Phags_Pa0
  STRING_Phoenician0
  STRING_Pi0
  STRING_Po0
  STRING_Ps0
  STRING_Psalter_Pahlavi0
  STRING_Rejang0
  STRING_Runic0
  STRING_S0
  STRING_Samaritan0
  STRING_Saurashtra0
  STRING_Sc0
  STRING_Sharada0
  STRING_Shavian0
  STRING_Siddham0
  STRING_SignWriting0
  STRING_Sinhala0
  STRING_Sk0
  STRING_Sm0
  STRING_So0
  STRING_Sogdian0
  STRING_Sora_Sompeng0
  STRING_Soyombo0
  STRING_Sundanese0
  STRING_Syloti_Nagri0
  STRING_Syriac0
  STRING_Tagalog0
  STRING_Tagbanwa0
  STRING_Tai_Le0
  STRING_Tai_Tham0
  STRING_Tai_Viet0
  STRING_Takri0
  STRING_Tamil0
  STRING_Tangut0
  STRING_Telugu0
  STRING_Thaana0
  STRING_Thai0
  STRING_Tibetan0
  STRING_Tifinagh0
  STRING_Tirhuta0
  STRING_Ugaritic0
  STRING_Unknown0
  STRING_Vai0
  STRING_Wancho0
  STRING_Warang_Citi0
  STRING_Xan0
  STRING_Xps0
  STRING_Xsp0
  STRING_Xuc0
  STRING_Xwd0
  STRING_Yi0
  STRING_Z0
  STRING_Zanabazar_Square0
  STRING_Zl0
  STRING_Zp0
  STRING_Zs0;

const ucp_type_table PRIV(utt)[] = {
  {   0, PT_SC, ucp_Adlam },
  {   6, PT_SC, ucp_Ahom },
  {  11, PT_SC, ucp_Anatolian_Hieroglyphs },
  {  33, PT_ANY, 0 },
  {  37, PT_SC, ucp_Arabic },
  {  44, PT_SC, ucp_Armenian },
  {  53, PT_SC, ucp_Avestan },
  {  61, PT_SC, ucp_Balinese },
  {  70, PT_SC, ucp_Bamum },
  {  76, PT_SC, ucp_Bassa_Vah },
  {  86, PT_SC, ucp_Batak },
  {  92, PT_SC, ucp_Bengali },
  { 100, PT_SC, ucp_Bhaiksuki },
  { 110, PT_SC, ucp_Bopomofo },
  { 119, PT_SC, ucp_Brahmi },
  { 126, PT_SC, ucp_Braille },
  { 134, PT_SC, ucp_Buginese },
  { 143, PT_SC, ucp_Buhid },
  { 149, PT_GC, ucp_C },
  { 151, PT_SC, ucp_Canadian_Aboriginal },
  { 171, PT_SC, ucp_Carian },
  { 178, PT_SC, ucp_Caucasian_Albanian },
  { 197, PT_PC, ucp_Cc },
  { 200, PT_PC, ucp_Cf },
  { 203, PT_SC, ucp_Chakma },
  { 210, PT_SC, ucp_Cham },
  { 215, PT_SC, ucp_Cherokee },
  { 224, PT_PC, ucp_Cn },
  { 227, PT_PC, ucp_Co },
  { 230, PT_SC, ucp_Common },
  { 237, PT_SC, ucp_Coptic },
  { 244, PT_PC, ucp_Cs },
  { 247, PT_SC, ucp_Cuneiform },
  { 257, PT_SC, ucp_Cypriot },
  { 265, PT_SC, ucp_Cyrillic },
  { 274, PT_SC, ucp_Deseret },
  { 282, PT_SC, ucp_Devanagari },
  { 293, PT_SC, ucp_Dogra },
  { 299, PT_SC, ucp_Duployan },
  { 308, PT_SC, ucp_Egyptian_Hieroglyphs },
  { 329, PT_SC, ucp_Elbasan },
  { 337, PT_SC, ucp_Elymaic },
  { 345, PT_SC, ucp_Ethiopic },
  { 354, PT_SC, ucp_Georgian },
  { 363, PT_SC, ucp_Glagolitic },
  { 374, PT_SC, ucp_Gothic },
  { 381, PT_SC, ucp_Grantha },
  { 389, PT_SC, ucp_Greek },
  { 395, PT_SC, ucp_Gujarati },
  { 404, PT_SC, ucp_Gunjala_Gondi },
  { 418, PT_SC, ucp_Gurmukhi },
  { 427, PT_SC, ucp_Han },
  { 431, PT_SC, ucp_Hangul },
  { 438, PT_SC, ucp_Hanifi_Rohingya },
  { 454, PT_SC, ucp_Hanunoo },
  { 462, PT_SC, ucp_Hatran },
  { 469, PT_SC, ucp_Hebrew },
  { 476, PT_SC, ucp_Hiragana },
  { 485, PT_SC, ucp_Imperial_Aramaic },
  { 502, PT_SC, ucp_Inherited },
  { 512, PT_SC, ucp_Inscriptional_Pahlavi },
  { 534, PT_SC, ucp_Inscriptional_Parthian },
  { 557, PT_SC, ucp_Javanese },
  { 566, PT_SC, ucp_Kaithi },
  { 573, PT_SC, ucp_Kannada },
  { 581, PT_SC, ucp_Katakana },
  { 590, PT_SC, ucp_Kayah_Li },
  { 599, PT_SC, ucp_Kharoshthi },
  { 610, PT_SC, ucp_Khmer },
  { 616, PT_SC, ucp_Khojki },
  { 623, PT_SC, ucp_Khudawadi },
  { 633, PT_GC, ucp_L },
  { 635, PT_LAMP, 0 },
  { 638, PT_SC, ucp_Lao },
  { 642, PT_SC, ucp_Latin },
  { 648, PT_SC, ucp_Lepcha },
  { 655, PT_SC, ucp_Limbu },
  { 661, PT_SC, ucp_Linear_A },
  { 670, PT_SC, ucp_Linear_B },
  { 679, PT_SC, ucp_Lisu },
  { 684, PT_PC, ucp_Ll },
  { 687, PT_PC, ucp_Lm },
  { 690, PT_PC, ucp_Lo },
  { 693, PT_PC, ucp_Lt },
  { 696, PT_PC, ucp_Lu },
  { 699, PT_SC, ucp_Lycian },
  { 706, PT_SC, ucp_Lydian },
  { 713, PT_GC, ucp_M },
  { 715, PT_SC, ucp_Mahajani },
  { 724, PT_SC, ucp_Makasar },
  { 732, PT_SC, ucp_Malayalam },
  { 742, PT_SC, ucp_Mandaic },
  { 750, PT_SC, ucp_Manichaean },
  { 761, PT_SC, ucp_Marchen },
  { 769, PT_SC, ucp_Masaram_Gondi },
  { 783, PT_PC, ucp_Mc },
  { 786, PT_PC, ucp_Me },
  { 789, PT_SC, ucp_Medefaidrin },
  { 801, PT_SC, ucp_Meetei_Mayek },
  { 814, PT_SC, ucp_Mende_Kikakui },
  { 828, PT_SC, ucp_Meroitic_Cursive },
  { 845, PT_SC, ucp_Meroitic_Hieroglyphs },
  { 866, PT_SC, ucp_Miao },
  { 871, PT_PC, ucp_Mn },
  { 874, PT_SC, ucp_Modi },
  { 879, PT_SC, ucp_Mongolian },
  { 889, PT_SC, ucp_Mro },
  { 893, PT_SC, ucp_Multani },
  { 901, PT_SC, ucp_Myanmar },
  { 909, PT_GC, ucp_N },
  { 911, PT_SC, ucp_Nabataean },
  { 921, PT_SC, ucp_Nandinagari },
  { 933, PT_PC, ucp_Nd },
  { 936, PT_SC, ucp_New_Tai_Lue },
  { 948, PT_SC, ucp_Newa },
  { 953, PT_SC, ucp_Nko },
  { 957, PT_PC, ucp_Nl },
  { 960, PT_PC, ucp_No },
  { 963, PT_SC, ucp_Nushu },
  { 969, PT_SC, ucp_Nyiakeng_Puachue_Hmong },
  { 992, PT_SC, ucp_Ogham },
  { 998, PT_SC, ucp_Ol_Chiki },
  { 1007, PT_SC, ucp_Old_Hungarian },
  { 1021, PT_SC, ucp_Old_Italic },
  { 1032, PT_SC, ucp_Old_North_Arabian },
  { 1050, PT_SC, ucp_Old_Permic },
  { 1061, PT_SC, ucp_Old_Persian },
  { 1073, PT_SC, ucp_Old_Sogdian },
  { 1085, PT_SC, ucp_Old_South_Arabian },
  { 1103, PT_SC, ucp_Old_Turkic },
  { 1114, PT_SC, ucp_Oriya },
  { 1120, PT_SC, ucp_Osage },
  { 1126, PT_SC, ucp_Osmanya },
  { 1134, PT_GC, ucp_P },
  { 1136, PT_SC, ucp_Pahawh_Hmong },
  { 1149, PT_SC, ucp_Palmyrene },
  { 1159, PT_SC, ucp_Pau_Cin_Hau },
  { 1171, PT_PC, ucp_Pc },
  { 1174, PT_PC, ucp_Pd },
  { 1177, PT_PC, ucp_Pe },
  { 1180, PT_PC, ucp_Pf },
  { 1183, PT_SC, ucp_Phags_Pa },
  { 1192, PT_SC, ucp_Phoenician },
  { 1203, PT_PC, ucp_Pi },
  { 1206, PT_PC, ucp_Po },
  { 1209, PT_PC, ucp_Ps },
  { 1212, PT_SC, ucp_Psalter_Pahlavi },
  { 1228, PT_SC, ucp_Rejang },
  { 1235, PT_SC, ucp_Runic },
  { 1241, PT_GC, ucp_S },
  { 1243, PT_SC, ucp_Samaritan },
  { 1253, PT_SC, ucp_Saurashtra },
  { 1264, PT_PC, ucp_Sc },
  { 1267, PT_SC, ucp_Sharada },
  { 1275, PT_SC, ucp_Shavian },
  { 1283, PT_SC, ucp_Siddham },
  { 1291, PT_SC, ucp_SignWriting },
  { 1303, PT_SC, ucp_Sinhala },
  { 1311, PT_PC, ucp_Sk },
  { 1314, PT_PC, ucp_Sm },
  { 1317, PT_PC, ucp_So },
  { 1320, PT_SC, ucp_Sogdian },
  { 1328, PT_SC, ucp_Sora_Sompeng },
  { 1341, PT_SC, ucp_Soyombo },
  { 1349, PT_SC, ucp_Sundanese },
  { 1359, PT_SC, ucp_Syloti_Nagri },
  { 1372, PT_SC, ucp_Syriac },
  { 1379, PT_SC, ucp_Tagalog },
  { 1387, PT_SC, ucp_Tagbanwa },
  { 1396, PT_SC, ucp_Tai_Le },
  { 1403, PT_SC, ucp_Tai_Tham },
  { 1412, PT_SC, ucp_Tai_Viet },
  { 1421, PT_SC, ucp_Takri },
  { 1427, PT_SC, ucp_Tamil },
  { 1433, PT_SC, ucp_Tangut },
  { 1440, PT_SC, ucp_Telugu },
  { 1447, PT_SC, ucp_Thaana },
  { 1454, PT_SC, ucp_Thai },
  { 1459, PT_SC, ucp_Tibetan },
  { 1467, PT_SC, ucp_Tifinagh },
  { 1476, PT_SC, ucp_Tirhuta },
  { 1484, PT_SC, ucp_Ugaritic },
  { 1493, PT_SC, ucp_Unknown },
  { 1501, PT_SC, ucp_Vai },
  { 1505, PT_SC, ucp_Wancho },
  { 1512, PT_SC, ucp_Warang_Citi },
  { 1524, PT_ALNUM, 0 },
  { 1528, PT_PXSPACE, 0 },
  { 1532, PT_SPACE, 0 },
  { 1536, PT_UCNC, 0 },
  { 1540, PT_WORD, 0 },
  { 1544, PT_SC, ucp_Yi },
  { 1547, PT_GC, ucp_Z },
  { 1549, PT_SC, ucp_Zanabazar_Square },
  { 1566, PT_PC, ucp_Zl },
  { 1569, PT_PC, ucp_Zp },
  { 1572, PT_PC, ucp_Zs }
};

const size_t PRIV(utt_size) = sizeof(PRIV(utt)) / sizeof(ucp_type_table);

#endif /* SUPPORT_UNICODE */

/* End of pcre2_tables.c */
