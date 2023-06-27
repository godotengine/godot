/*************************************************
*       Perl-Compatible Regular Expressions      *
*************************************************/

/* This is the public header file for the PCRE library, second API, to be
#included by applications that call PCRE2 functions.

           Copyright (c) 2016-2021 University of Cambridge

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

#ifndef PCRE2_H_IDEMPOTENT_GUARD
#define PCRE2_H_IDEMPOTENT_GUARD

/* The current PCRE version information. */

#define PCRE2_MAJOR           10
#define PCRE2_MINOR           42
#define PCRE2_PRERELEASE      
#define PCRE2_DATE            2022-12-11

/* When an application links to a PCRE DLL in Windows, the symbols that are
imported have to be identified as such. When building PCRE2, the appropriate
export setting is defined in pcre2_internal.h, which includes this file. So we
don't change existing definitions of PCRE2_EXP_DECL. */

#if defined(_WIN32) && !defined(PCRE2_STATIC)
#  ifndef PCRE2_EXP_DECL
#    define PCRE2_EXP_DECL  extern __declspec(dllimport)
#  endif
#endif

/* By default, we use the standard "extern" declarations. */

#ifndef PCRE2_EXP_DECL
#  ifdef __cplusplus
#    define PCRE2_EXP_DECL  extern "C"
#  else
#    define PCRE2_EXP_DECL  extern
#  endif
#endif

/* When compiling with the MSVC compiler, it is sometimes necessary to include
a "calling convention" before exported function names. (This is secondhand
information; I know nothing about MSVC myself). For example, something like

  void __cdecl function(....)

might be needed. In order so make this easy, all the exported functions have
PCRE2_CALL_CONVENTION just before their names. It is rarely needed; if not
set, we ensure here that it has no effect. */

#ifndef PCRE2_CALL_CONVENTION
#define PCRE2_CALL_CONVENTION
#endif

/* Have to include limits.h, stdlib.h, and inttypes.h to ensure that size_t and
uint8_t, UCHAR_MAX, etc are defined. Some systems that do have inttypes.h do
not have stdint.h, which is why we use inttypes.h, which according to the C
standard is a superset of stdint.h. If inttypes.h is not available the build
will break and the relevant values must be provided by some other means. */

#include <limits.h>
#include <stdlib.h>
#include <inttypes.h>

/* Allow for C++ users compiling this directly. */

#ifdef __cplusplus
extern "C" {
#endif

/* The following option bits can be passed to pcre2_compile(), pcre2_match(),
or pcre2_dfa_match(). PCRE2_NO_UTF_CHECK affects only the function to which it
is passed. Put these bits at the most significant end of the options word so
others can be added next to them */

#define PCRE2_ANCHORED            0x80000000u
#define PCRE2_NO_UTF_CHECK        0x40000000u
#define PCRE2_ENDANCHORED         0x20000000u

/* The following option bits can be passed only to pcre2_compile(). However,
they may affect compilation, JIT compilation, and/or interpretive execution.
The following tags indicate which:

C   alters what is compiled by pcre2_compile()
J   alters what is compiled by pcre2_jit_compile()
M   is inspected during pcre2_match() execution
D   is inspected during pcre2_dfa_match() execution
*/

#define PCRE2_ALLOW_EMPTY_CLASS   0x00000001u  /* C       */
#define PCRE2_ALT_BSUX            0x00000002u  /* C       */
#define PCRE2_AUTO_CALLOUT        0x00000004u  /* C       */
#define PCRE2_CASELESS            0x00000008u  /* C       */
#define PCRE2_DOLLAR_ENDONLY      0x00000010u  /*   J M D */
#define PCRE2_DOTALL              0x00000020u  /* C       */
#define PCRE2_DUPNAMES            0x00000040u  /* C       */
#define PCRE2_EXTENDED            0x00000080u  /* C       */
#define PCRE2_FIRSTLINE           0x00000100u  /*   J M D */
#define PCRE2_MATCH_UNSET_BACKREF 0x00000200u  /* C J M   */
#define PCRE2_MULTILINE           0x00000400u  /* C       */
#define PCRE2_NEVER_UCP           0x00000800u  /* C       */
#define PCRE2_NEVER_UTF           0x00001000u  /* C       */
#define PCRE2_NO_AUTO_CAPTURE     0x00002000u  /* C       */
#define PCRE2_NO_AUTO_POSSESS     0x00004000u  /* C       */
#define PCRE2_NO_DOTSTAR_ANCHOR   0x00008000u  /* C       */
#define PCRE2_NO_START_OPTIMIZE   0x00010000u  /*   J M D */
#define PCRE2_UCP                 0x00020000u  /* C J M D */
#define PCRE2_UNGREEDY            0x00040000u  /* C       */
#define PCRE2_UTF                 0x00080000u  /* C J M D */
#define PCRE2_NEVER_BACKSLASH_C   0x00100000u  /* C       */
#define PCRE2_ALT_CIRCUMFLEX      0x00200000u  /*   J M D */
#define PCRE2_ALT_VERBNAMES       0x00400000u  /* C       */
#define PCRE2_USE_OFFSET_LIMIT    0x00800000u  /*   J M D */
#define PCRE2_EXTENDED_MORE       0x01000000u  /* C       */
#define PCRE2_LITERAL             0x02000000u  /* C       */
#define PCRE2_MATCH_INVALID_UTF   0x04000000u  /*   J M D */

/* An additional compile options word is available in the compile context. */

#define PCRE2_EXTRA_ALLOW_SURROGATE_ESCAPES  0x00000001u  /* C */
#define PCRE2_EXTRA_BAD_ESCAPE_IS_LITERAL    0x00000002u  /* C */
#define PCRE2_EXTRA_MATCH_WORD               0x00000004u  /* C */
#define PCRE2_EXTRA_MATCH_LINE               0x00000008u  /* C */
#define PCRE2_EXTRA_ESCAPED_CR_IS_LF         0x00000010u  /* C */
#define PCRE2_EXTRA_ALT_BSUX                 0x00000020u  /* C */
#define PCRE2_EXTRA_ALLOW_LOOKAROUND_BSK     0x00000040u  /* C */

/* These are for pcre2_jit_compile(). */

#define PCRE2_JIT_COMPLETE        0x00000001u  /* For full matching */
#define PCRE2_JIT_PARTIAL_SOFT    0x00000002u
#define PCRE2_JIT_PARTIAL_HARD    0x00000004u
#define PCRE2_JIT_INVALID_UTF     0x00000100u

/* These are for pcre2_match(), pcre2_dfa_match(), pcre2_jit_match(), and
pcre2_substitute(). Some are allowed only for one of the functions, and in
these cases it is noted below. Note that PCRE2_ANCHORED, PCRE2_ENDANCHORED and
PCRE2_NO_UTF_CHECK can also be passed to these functions (though
pcre2_jit_match() ignores the latter since it bypasses all sanity checks). */

#define PCRE2_NOTBOL                      0x00000001u
#define PCRE2_NOTEOL                      0x00000002u
#define PCRE2_NOTEMPTY                    0x00000004u  /* ) These two must be kept */
#define PCRE2_NOTEMPTY_ATSTART            0x00000008u  /* ) adjacent to each other. */
#define PCRE2_PARTIAL_SOFT                0x00000010u
#define PCRE2_PARTIAL_HARD                0x00000020u
#define PCRE2_DFA_RESTART                 0x00000040u  /* pcre2_dfa_match() only */
#define PCRE2_DFA_SHORTEST                0x00000080u  /* pcre2_dfa_match() only */
#define PCRE2_SUBSTITUTE_GLOBAL           0x00000100u  /* pcre2_substitute() only */
#define PCRE2_SUBSTITUTE_EXTENDED         0x00000200u  /* pcre2_substitute() only */
#define PCRE2_SUBSTITUTE_UNSET_EMPTY      0x00000400u  /* pcre2_substitute() only */
#define PCRE2_SUBSTITUTE_UNKNOWN_UNSET    0x00000800u  /* pcre2_substitute() only */
#define PCRE2_SUBSTITUTE_OVERFLOW_LENGTH  0x00001000u  /* pcre2_substitute() only */
#define PCRE2_NO_JIT                      0x00002000u  /* Not for pcre2_dfa_match() */
#define PCRE2_COPY_MATCHED_SUBJECT        0x00004000u
#define PCRE2_SUBSTITUTE_LITERAL          0x00008000u  /* pcre2_substitute() only */
#define PCRE2_SUBSTITUTE_MATCHED          0x00010000u  /* pcre2_substitute() only */
#define PCRE2_SUBSTITUTE_REPLACEMENT_ONLY 0x00020000u  /* pcre2_substitute() only */

/* Options for pcre2_pattern_convert(). */

#define PCRE2_CONVERT_UTF                    0x00000001u
#define PCRE2_CONVERT_NO_UTF_CHECK           0x00000002u
#define PCRE2_CONVERT_POSIX_BASIC            0x00000004u
#define PCRE2_CONVERT_POSIX_EXTENDED         0x00000008u
#define PCRE2_CONVERT_GLOB                   0x00000010u
#define PCRE2_CONVERT_GLOB_NO_WILD_SEPARATOR 0x00000030u
#define PCRE2_CONVERT_GLOB_NO_STARSTAR       0x00000050u

/* Newline and \R settings, for use in compile contexts. The newline values
must be kept in step with values set in config.h and both sets must all be
greater than zero. */

#define PCRE2_NEWLINE_CR          1
#define PCRE2_NEWLINE_LF          2
#define PCRE2_NEWLINE_CRLF        3
#define PCRE2_NEWLINE_ANY         4
#define PCRE2_NEWLINE_ANYCRLF     5
#define PCRE2_NEWLINE_NUL         6

#define PCRE2_BSR_UNICODE         1
#define PCRE2_BSR_ANYCRLF         2

/* Error codes for pcre2_compile(). Some of these are also used by
pcre2_pattern_convert(). */

#define PCRE2_ERROR_END_BACKSLASH                  101
#define PCRE2_ERROR_END_BACKSLASH_C                102
#define PCRE2_ERROR_UNKNOWN_ESCAPE                 103
#define PCRE2_ERROR_QUANTIFIER_OUT_OF_ORDER        104
#define PCRE2_ERROR_QUANTIFIER_TOO_BIG             105
#define PCRE2_ERROR_MISSING_SQUARE_BRACKET         106
#define PCRE2_ERROR_ESCAPE_INVALID_IN_CLASS        107
#define PCRE2_ERROR_CLASS_RANGE_ORDER              108
#define PCRE2_ERROR_QUANTIFIER_INVALID             109
#define PCRE2_ERROR_INTERNAL_UNEXPECTED_REPEAT     110
#define PCRE2_ERROR_INVALID_AFTER_PARENS_QUERY     111
#define PCRE2_ERROR_POSIX_CLASS_NOT_IN_CLASS       112
#define PCRE2_ERROR_POSIX_NO_SUPPORT_COLLATING     113
#define PCRE2_ERROR_MISSING_CLOSING_PARENTHESIS    114
#define PCRE2_ERROR_BAD_SUBPATTERN_REFERENCE       115
#define PCRE2_ERROR_NULL_PATTERN                   116
#define PCRE2_ERROR_BAD_OPTIONS                    117
#define PCRE2_ERROR_MISSING_COMMENT_CLOSING        118
#define PCRE2_ERROR_PARENTHESES_NEST_TOO_DEEP      119
#define PCRE2_ERROR_PATTERN_TOO_LARGE              120
#define PCRE2_ERROR_HEAP_FAILED                    121
#define PCRE2_ERROR_UNMATCHED_CLOSING_PARENTHESIS  122
#define PCRE2_ERROR_INTERNAL_CODE_OVERFLOW         123
#define PCRE2_ERROR_MISSING_CONDITION_CLOSING      124
#define PCRE2_ERROR_LOOKBEHIND_NOT_FIXED_LENGTH    125
#define PCRE2_ERROR_ZERO_RELATIVE_REFERENCE        126
#define PCRE2_ERROR_TOO_MANY_CONDITION_BRANCHES    127
#define PCRE2_ERROR_CONDITION_ASSERTION_EXPECTED   128
#define PCRE2_ERROR_BAD_RELATIVE_REFERENCE         129
#define PCRE2_ERROR_UNKNOWN_POSIX_CLASS            130
#define PCRE2_ERROR_INTERNAL_STUDY_ERROR           131
#define PCRE2_ERROR_UNICODE_NOT_SUPPORTED          132
#define PCRE2_ERROR_PARENTHESES_STACK_CHECK        133
#define PCRE2_ERROR_CODE_POINT_TOO_BIG             134
#define PCRE2_ERROR_LOOKBEHIND_TOO_COMPLICATED     135
#define PCRE2_ERROR_LOOKBEHIND_INVALID_BACKSLASH_C 136
#define PCRE2_ERROR_UNSUPPORTED_ESCAPE_SEQUENCE    137
#define PCRE2_ERROR_CALLOUT_NUMBER_TOO_BIG         138
#define PCRE2_ERROR_MISSING_CALLOUT_CLOSING        139
#define PCRE2_ERROR_ESCAPE_INVALID_IN_VERB         140
#define PCRE2_ERROR_UNRECOGNIZED_AFTER_QUERY_P     141
#define PCRE2_ERROR_MISSING_NAME_TERMINATOR        142
#define PCRE2_ERROR_DUPLICATE_SUBPATTERN_NAME      143
#define PCRE2_ERROR_INVALID_SUBPATTERN_NAME        144
#define PCRE2_ERROR_UNICODE_PROPERTIES_UNAVAILABLE 145
#define PCRE2_ERROR_MALFORMED_UNICODE_PROPERTY     146
#define PCRE2_ERROR_UNKNOWN_UNICODE_PROPERTY       147
#define PCRE2_ERROR_SUBPATTERN_NAME_TOO_LONG       148
#define PCRE2_ERROR_TOO_MANY_NAMED_SUBPATTERNS     149
#define PCRE2_ERROR_CLASS_INVALID_RANGE            150
#define PCRE2_ERROR_OCTAL_BYTE_TOO_BIG             151
#define PCRE2_ERROR_INTERNAL_OVERRAN_WORKSPACE     152
#define PCRE2_ERROR_INTERNAL_MISSING_SUBPATTERN    153
#define PCRE2_ERROR_DEFINE_TOO_MANY_BRANCHES       154
#define PCRE2_ERROR_BACKSLASH_O_MISSING_BRACE      155
#define PCRE2_ERROR_INTERNAL_UNKNOWN_NEWLINE       156
#define PCRE2_ERROR_BACKSLASH_G_SYNTAX             157
#define PCRE2_ERROR_PARENS_QUERY_R_MISSING_CLOSING 158
/* Error 159 is obsolete and should now never occur */
#define PCRE2_ERROR_VERB_ARGUMENT_NOT_ALLOWED      159
#define PCRE2_ERROR_VERB_UNKNOWN                   160
#define PCRE2_ERROR_SUBPATTERN_NUMBER_TOO_BIG      161
#define PCRE2_ERROR_SUBPATTERN_NAME_EXPECTED       162
#define PCRE2_ERROR_INTERNAL_PARSED_OVERFLOW       163
#define PCRE2_ERROR_INVALID_OCTAL                  164
#define PCRE2_ERROR_SUBPATTERN_NAMES_MISMATCH      165
#define PCRE2_ERROR_MARK_MISSING_ARGUMENT          166
#define PCRE2_ERROR_INVALID_HEXADECIMAL            167
#define PCRE2_ERROR_BACKSLASH_C_SYNTAX             168
#define PCRE2_ERROR_BACKSLASH_K_SYNTAX             169
#define PCRE2_ERROR_INTERNAL_BAD_CODE_LOOKBEHINDS  170
#define PCRE2_ERROR_BACKSLASH_N_IN_CLASS           171
#define PCRE2_ERROR_CALLOUT_STRING_TOO_LONG        172
#define PCRE2_ERROR_UNICODE_DISALLOWED_CODE_POINT  173
#define PCRE2_ERROR_UTF_IS_DISABLED                174
#define PCRE2_ERROR_UCP_IS_DISABLED                175
#define PCRE2_ERROR_VERB_NAME_TOO_LONG             176
#define PCRE2_ERROR_BACKSLASH_U_CODE_POINT_TOO_BIG 177
#define PCRE2_ERROR_MISSING_OCTAL_OR_HEX_DIGITS    178
#define PCRE2_ERROR_VERSION_CONDITION_SYNTAX       179
#define PCRE2_ERROR_INTERNAL_BAD_CODE_AUTO_POSSESS 180
#define PCRE2_ERROR_CALLOUT_NO_STRING_DELIMITER    181
#define PCRE2_ERROR_CALLOUT_BAD_STRING_DELIMITER   182
#define PCRE2_ERROR_BACKSLASH_C_CALLER_DISABLED    183
#define PCRE2_ERROR_QUERY_BARJX_NEST_TOO_DEEP      184
#define PCRE2_ERROR_BACKSLASH_C_LIBRARY_DISABLED   185
#define PCRE2_ERROR_PATTERN_TOO_COMPLICATED        186
#define PCRE2_ERROR_LOOKBEHIND_TOO_LONG            187
#define PCRE2_ERROR_PATTERN_STRING_TOO_LONG        188
#define PCRE2_ERROR_INTERNAL_BAD_CODE              189
#define PCRE2_ERROR_INTERNAL_BAD_CODE_IN_SKIP      190
#define PCRE2_ERROR_NO_SURROGATES_IN_UTF16         191
#define PCRE2_ERROR_BAD_LITERAL_OPTIONS            192
#define PCRE2_ERROR_SUPPORTED_ONLY_IN_UNICODE      193
#define PCRE2_ERROR_INVALID_HYPHEN_IN_OPTIONS      194
#define PCRE2_ERROR_ALPHA_ASSERTION_UNKNOWN        195
#define PCRE2_ERROR_SCRIPT_RUN_NOT_AVAILABLE       196
#define PCRE2_ERROR_TOO_MANY_CAPTURES              197
#define PCRE2_ERROR_CONDITION_ATOMIC_ASSERTION_EXPECTED  198
#define PCRE2_ERROR_BACKSLASH_K_IN_LOOKAROUND      199


/* "Expected" matching error codes: no match and partial match. */

#define PCRE2_ERROR_NOMATCH          (-1)
#define PCRE2_ERROR_PARTIAL          (-2)

/* Error codes for UTF-8 validity checks */

#define PCRE2_ERROR_UTF8_ERR1        (-3)
#define PCRE2_ERROR_UTF8_ERR2        (-4)
#define PCRE2_ERROR_UTF8_ERR3        (-5)
#define PCRE2_ERROR_UTF8_ERR4        (-6)
#define PCRE2_ERROR_UTF8_ERR5        (-7)
#define PCRE2_ERROR_UTF8_ERR6        (-8)
#define PCRE2_ERROR_UTF8_ERR7        (-9)
#define PCRE2_ERROR_UTF8_ERR8       (-10)
#define PCRE2_ERROR_UTF8_ERR9       (-11)
#define PCRE2_ERROR_UTF8_ERR10      (-12)
#define PCRE2_ERROR_UTF8_ERR11      (-13)
#define PCRE2_ERROR_UTF8_ERR12      (-14)
#define PCRE2_ERROR_UTF8_ERR13      (-15)
#define PCRE2_ERROR_UTF8_ERR14      (-16)
#define PCRE2_ERROR_UTF8_ERR15      (-17)
#define PCRE2_ERROR_UTF8_ERR16      (-18)
#define PCRE2_ERROR_UTF8_ERR17      (-19)
#define PCRE2_ERROR_UTF8_ERR18      (-20)
#define PCRE2_ERROR_UTF8_ERR19      (-21)
#define PCRE2_ERROR_UTF8_ERR20      (-22)
#define PCRE2_ERROR_UTF8_ERR21      (-23)

/* Error codes for UTF-16 validity checks */

#define PCRE2_ERROR_UTF16_ERR1      (-24)
#define PCRE2_ERROR_UTF16_ERR2      (-25)
#define PCRE2_ERROR_UTF16_ERR3      (-26)

/* Error codes for UTF-32 validity checks */

#define PCRE2_ERROR_UTF32_ERR1      (-27)
#define PCRE2_ERROR_UTF32_ERR2      (-28)

/* Miscellaneous error codes for pcre2[_dfa]_match(), substring extraction
functions, context functions, and serializing functions. They are in numerical
order. Originally they were in alphabetical order too, but now that PCRE2 is
released, the numbers must not be changed. */

#define PCRE2_ERROR_BADDATA           (-29)
#define PCRE2_ERROR_MIXEDTABLES       (-30)  /* Name was changed */
#define PCRE2_ERROR_BADMAGIC          (-31)
#define PCRE2_ERROR_BADMODE           (-32)
#define PCRE2_ERROR_BADOFFSET         (-33)
#define PCRE2_ERROR_BADOPTION         (-34)
#define PCRE2_ERROR_BADREPLACEMENT    (-35)
#define PCRE2_ERROR_BADUTFOFFSET      (-36)
#define PCRE2_ERROR_CALLOUT           (-37)  /* Never used by PCRE2 itself */
#define PCRE2_ERROR_DFA_BADRESTART    (-38)
#define PCRE2_ERROR_DFA_RECURSE       (-39)
#define PCRE2_ERROR_DFA_UCOND         (-40)
#define PCRE2_ERROR_DFA_UFUNC         (-41)
#define PCRE2_ERROR_DFA_UITEM         (-42)
#define PCRE2_ERROR_DFA_WSSIZE        (-43)
#define PCRE2_ERROR_INTERNAL          (-44)
#define PCRE2_ERROR_JIT_BADOPTION     (-45)
#define PCRE2_ERROR_JIT_STACKLIMIT    (-46)
#define PCRE2_ERROR_MATCHLIMIT        (-47)
#define PCRE2_ERROR_NOMEMORY          (-48)
#define PCRE2_ERROR_NOSUBSTRING       (-49)
#define PCRE2_ERROR_NOUNIQUESUBSTRING (-50)
#define PCRE2_ERROR_NULL              (-51)
#define PCRE2_ERROR_RECURSELOOP       (-52)
#define PCRE2_ERROR_DEPTHLIMIT        (-53)
#define PCRE2_ERROR_RECURSIONLIMIT    (-53)  /* Obsolete synonym */
#define PCRE2_ERROR_UNAVAILABLE       (-54)
#define PCRE2_ERROR_UNSET             (-55)
#define PCRE2_ERROR_BADOFFSETLIMIT    (-56)
#define PCRE2_ERROR_BADREPESCAPE      (-57)
#define PCRE2_ERROR_REPMISSINGBRACE   (-58)
#define PCRE2_ERROR_BADSUBSTITUTION   (-59)
#define PCRE2_ERROR_BADSUBSPATTERN    (-60)
#define PCRE2_ERROR_TOOMANYREPLACE    (-61)
#define PCRE2_ERROR_BADSERIALIZEDDATA (-62)
#define PCRE2_ERROR_HEAPLIMIT         (-63)
#define PCRE2_ERROR_CONVERT_SYNTAX    (-64)
#define PCRE2_ERROR_INTERNAL_DUPMATCH (-65)
#define PCRE2_ERROR_DFA_UINVALID_UTF  (-66)


/* Request types for pcre2_pattern_info() */

#define PCRE2_INFO_ALLOPTIONS            0
#define PCRE2_INFO_ARGOPTIONS            1
#define PCRE2_INFO_BACKREFMAX            2
#define PCRE2_INFO_BSR                   3
#define PCRE2_INFO_CAPTURECOUNT          4
#define PCRE2_INFO_FIRSTCODEUNIT         5
#define PCRE2_INFO_FIRSTCODETYPE         6
#define PCRE2_INFO_FIRSTBITMAP           7
#define PCRE2_INFO_HASCRORLF             8
#define PCRE2_INFO_JCHANGED              9
#define PCRE2_INFO_JITSIZE              10
#define PCRE2_INFO_LASTCODEUNIT         11
#define PCRE2_INFO_LASTCODETYPE         12
#define PCRE2_INFO_MATCHEMPTY           13
#define PCRE2_INFO_MATCHLIMIT           14
#define PCRE2_INFO_MAXLOOKBEHIND        15
#define PCRE2_INFO_MINLENGTH            16
#define PCRE2_INFO_NAMECOUNT            17
#define PCRE2_INFO_NAMEENTRYSIZE        18
#define PCRE2_INFO_NAMETABLE            19
#define PCRE2_INFO_NEWLINE              20
#define PCRE2_INFO_DEPTHLIMIT           21
#define PCRE2_INFO_RECURSIONLIMIT       21  /* Obsolete synonym */
#define PCRE2_INFO_SIZE                 22
#define PCRE2_INFO_HASBACKSLASHC        23
#define PCRE2_INFO_FRAMESIZE            24
#define PCRE2_INFO_HEAPLIMIT            25
#define PCRE2_INFO_EXTRAOPTIONS         26

/* Request types for pcre2_config(). */

#define PCRE2_CONFIG_BSR                     0
#define PCRE2_CONFIG_JIT                     1
#define PCRE2_CONFIG_JITTARGET               2
#define PCRE2_CONFIG_LINKSIZE                3
#define PCRE2_CONFIG_MATCHLIMIT              4
#define PCRE2_CONFIG_NEWLINE                 5
#define PCRE2_CONFIG_PARENSLIMIT             6
#define PCRE2_CONFIG_DEPTHLIMIT              7
#define PCRE2_CONFIG_RECURSIONLIMIT          7  /* Obsolete synonym */
#define PCRE2_CONFIG_STACKRECURSE            8  /* Obsolete */
#define PCRE2_CONFIG_UNICODE                 9
#define PCRE2_CONFIG_UNICODE_VERSION        10
#define PCRE2_CONFIG_VERSION                11
#define PCRE2_CONFIG_HEAPLIMIT              12
#define PCRE2_CONFIG_NEVER_BACKSLASH_C      13
#define PCRE2_CONFIG_COMPILED_WIDTHS        14
#define PCRE2_CONFIG_TABLES_LENGTH          15


/* Types for code units in patterns and subject strings. */

typedef uint8_t  PCRE2_UCHAR8;
typedef uint16_t PCRE2_UCHAR16;
typedef uint32_t PCRE2_UCHAR32;

typedef const PCRE2_UCHAR8  *PCRE2_SPTR8;
typedef const PCRE2_UCHAR16 *PCRE2_SPTR16;
typedef const PCRE2_UCHAR32 *PCRE2_SPTR32;

/* The PCRE2_SIZE type is used for all string lengths and offsets in PCRE2,
including pattern offsets for errors and subject offsets after a match. We
define special values to indicate zero-terminated strings and unset offsets in
the offset vector (ovector). */

#define PCRE2_SIZE            size_t
#define PCRE2_SIZE_MAX        SIZE_MAX
#define PCRE2_ZERO_TERMINATED (~(PCRE2_SIZE)0)
#define PCRE2_UNSET           (~(PCRE2_SIZE)0)

/* Generic types for opaque structures and JIT callback functions. These
declarations are defined in a macro that is expanded for each width later. */

#define PCRE2_TYPES_LIST \
struct pcre2_real_general_context; \
typedef struct pcre2_real_general_context pcre2_general_context; \
\
struct pcre2_real_compile_context; \
typedef struct pcre2_real_compile_context pcre2_compile_context; \
\
struct pcre2_real_match_context; \
typedef struct pcre2_real_match_context pcre2_match_context; \
\
struct pcre2_real_convert_context; \
typedef struct pcre2_real_convert_context pcre2_convert_context; \
\
struct pcre2_real_code; \
typedef struct pcre2_real_code pcre2_code; \
\
struct pcre2_real_match_data; \
typedef struct pcre2_real_match_data pcre2_match_data; \
\
struct pcre2_real_jit_stack; \
typedef struct pcre2_real_jit_stack pcre2_jit_stack; \
\
typedef pcre2_jit_stack *(*pcre2_jit_callback)(void *);


/* The structures for passing out data via callout functions. We use structures
so that new fields can be added on the end in future versions, without changing
the API of the function, thereby allowing old clients to work without
modification. Define the generic versions in a macro; the width-specific
versions are generated from this macro below. */

/* Flags for the callout_flags field. These are cleared after a callout. */

#define PCRE2_CALLOUT_STARTMATCH    0x00000001u  /* Set for each bumpalong */
#define PCRE2_CALLOUT_BACKTRACK     0x00000002u  /* Set after a backtrack */

#define PCRE2_STRUCTURE_LIST \
typedef struct pcre2_callout_block { \
  uint32_t      version;           /* Identifies version of block */ \
  /* ------------------------ Version 0 ------------------------------- */ \
  uint32_t      callout_number;    /* Number compiled into pattern */ \
  uint32_t      capture_top;       /* Max current capture */ \
  uint32_t      capture_last;      /* Most recently closed capture */ \
  PCRE2_SIZE   *offset_vector;     /* The offset vector */ \
  PCRE2_SPTR    mark;              /* Pointer to current mark or NULL */ \
  PCRE2_SPTR    subject;           /* The subject being matched */ \
  PCRE2_SIZE    subject_length;    /* The length of the subject */ \
  PCRE2_SIZE    start_match;       /* Offset to start of this match attempt */ \
  PCRE2_SIZE    current_position;  /* Where we currently are in the subject */ \
  PCRE2_SIZE    pattern_position;  /* Offset to next item in the pattern */ \
  PCRE2_SIZE    next_item_length;  /* Length of next item in the pattern */ \
  /* ------------------- Added for Version 1 -------------------------- */ \
  PCRE2_SIZE    callout_string_offset; /* Offset to string within pattern */ \
  PCRE2_SIZE    callout_string_length; /* Length of string compiled into pattern */ \
  PCRE2_SPTR    callout_string;    /* String compiled into pattern */ \
  /* ------------------- Added for Version 2 -------------------------- */ \
  uint32_t      callout_flags;     /* See above for list */ \
  /* ------------------------------------------------------------------ */ \
} pcre2_callout_block; \
\
typedef struct pcre2_callout_enumerate_block { \
  uint32_t      version;           /* Identifies version of block */ \
  /* ------------------------ Version 0 ------------------------------- */ \
  PCRE2_SIZE    pattern_position;  /* Offset to next item in the pattern */ \
  PCRE2_SIZE    next_item_length;  /* Length of next item in the pattern */ \
  uint32_t      callout_number;    /* Number compiled into pattern */ \
  PCRE2_SIZE    callout_string_offset; /* Offset to string within pattern */ \
  PCRE2_SIZE    callout_string_length; /* Length of string compiled into pattern */ \
  PCRE2_SPTR    callout_string;    /* String compiled into pattern */ \
  /* ------------------------------------------------------------------ */ \
} pcre2_callout_enumerate_block; \
\
typedef struct pcre2_substitute_callout_block { \
  uint32_t      version;           /* Identifies version of block */ \
  /* ------------------------ Version 0 ------------------------------- */ \
  PCRE2_SPTR    input;             /* Pointer to input subject string */ \
  PCRE2_SPTR    output;            /* Pointer to output buffer */ \
  PCRE2_SIZE    output_offsets[2]; /* Changed portion of the output */ \
  PCRE2_SIZE   *ovector;           /* Pointer to current ovector */ \
  uint32_t      oveccount;         /* Count of pairs set in ovector */ \
  uint32_t      subscount;         /* Substitution number */ \
  /* ------------------------------------------------------------------ */ \
} pcre2_substitute_callout_block;


/* List the generic forms of all other functions in macros, which will be
expanded for each width below. Start with functions that give general
information. */

#define PCRE2_GENERAL_INFO_FUNCTIONS \
PCRE2_EXP_DECL int PCRE2_CALL_CONVENTION pcre2_config(uint32_t, void *);


/* Functions for manipulating contexts. */

#define PCRE2_GENERAL_CONTEXT_FUNCTIONS \
PCRE2_EXP_DECL pcre2_general_context *PCRE2_CALL_CONVENTION \
  pcre2_general_context_copy(pcre2_general_context *); \
PCRE2_EXP_DECL pcre2_general_context *PCRE2_CALL_CONVENTION \
  pcre2_general_context_create(void *(*)(PCRE2_SIZE, void *), \
    void (*)(void *, void *), void *); \
PCRE2_EXP_DECL void PCRE2_CALL_CONVENTION \
  pcre2_general_context_free(pcre2_general_context *);

#define PCRE2_COMPILE_CONTEXT_FUNCTIONS \
PCRE2_EXP_DECL pcre2_compile_context *PCRE2_CALL_CONVENTION \
  pcre2_compile_context_copy(pcre2_compile_context *); \
PCRE2_EXP_DECL pcre2_compile_context *PCRE2_CALL_CONVENTION \
  pcre2_compile_context_create(pcre2_general_context *);\
PCRE2_EXP_DECL void PCRE2_CALL_CONVENTION \
  pcre2_compile_context_free(pcre2_compile_context *); \
PCRE2_EXP_DECL int PCRE2_CALL_CONVENTION \
  pcre2_set_bsr(pcre2_compile_context *, uint32_t); \
PCRE2_EXP_DECL int PCRE2_CALL_CONVENTION \
  pcre2_set_character_tables(pcre2_compile_context *, const uint8_t *); \
PCRE2_EXP_DECL int PCRE2_CALL_CONVENTION \
  pcre2_set_compile_extra_options(pcre2_compile_context *, uint32_t); \
PCRE2_EXP_DECL int PCRE2_CALL_CONVENTION \
  pcre2_set_max_pattern_length(pcre2_compile_context *, PCRE2_SIZE); \
PCRE2_EXP_DECL int PCRE2_CALL_CONVENTION \
  pcre2_set_newline(pcre2_compile_context *, uint32_t); \
PCRE2_EXP_DECL int PCRE2_CALL_CONVENTION \
  pcre2_set_parens_nest_limit(pcre2_compile_context *, uint32_t); \
PCRE2_EXP_DECL int PCRE2_CALL_CONVENTION \
  pcre2_set_compile_recursion_guard(pcre2_compile_context *, \
    int (*)(uint32_t, void *), void *);

#define PCRE2_MATCH_CONTEXT_FUNCTIONS \
PCRE2_EXP_DECL pcre2_match_context *PCRE2_CALL_CONVENTION \
  pcre2_match_context_copy(pcre2_match_context *); \
PCRE2_EXP_DECL pcre2_match_context *PCRE2_CALL_CONVENTION \
  pcre2_match_context_create(pcre2_general_context *); \
PCRE2_EXP_DECL void PCRE2_CALL_CONVENTION \
  pcre2_match_context_free(pcre2_match_context *); \
PCRE2_EXP_DECL int PCRE2_CALL_CONVENTION \
  pcre2_set_callout(pcre2_match_context *, \
    int (*)(pcre2_callout_block *, void *), void *); \
PCRE2_EXP_DECL int PCRE2_CALL_CONVENTION \
  pcre2_set_substitute_callout(pcre2_match_context *, \
    int (*)(pcre2_substitute_callout_block *, void *), void *); \
PCRE2_EXP_DECL int PCRE2_CALL_CONVENTION \
  pcre2_set_depth_limit(pcre2_match_context *, uint32_t); \
PCRE2_EXP_DECL int PCRE2_CALL_CONVENTION \
  pcre2_set_heap_limit(pcre2_match_context *, uint32_t); \
PCRE2_EXP_DECL int PCRE2_CALL_CONVENTION \
  pcre2_set_match_limit(pcre2_match_context *, uint32_t); \
PCRE2_EXP_DECL int PCRE2_CALL_CONVENTION \
  pcre2_set_offset_limit(pcre2_match_context *, PCRE2_SIZE); \
PCRE2_EXP_DECL int PCRE2_CALL_CONVENTION \
  pcre2_set_recursion_limit(pcre2_match_context *, uint32_t); \
PCRE2_EXP_DECL int PCRE2_CALL_CONVENTION \
  pcre2_set_recursion_memory_management(pcre2_match_context *, \
    void *(*)(PCRE2_SIZE, void *), void (*)(void *, void *), void *);

#define PCRE2_CONVERT_CONTEXT_FUNCTIONS \
PCRE2_EXP_DECL pcre2_convert_context *PCRE2_CALL_CONVENTION \
  pcre2_convert_context_copy(pcre2_convert_context *); \
PCRE2_EXP_DECL pcre2_convert_context *PCRE2_CALL_CONVENTION \
  pcre2_convert_context_create(pcre2_general_context *); \
PCRE2_EXP_DECL void PCRE2_CALL_CONVENTION \
  pcre2_convert_context_free(pcre2_convert_context *); \
PCRE2_EXP_DECL int PCRE2_CALL_CONVENTION \
  pcre2_set_glob_escape(pcre2_convert_context *, uint32_t); \
PCRE2_EXP_DECL int PCRE2_CALL_CONVENTION \
  pcre2_set_glob_separator(pcre2_convert_context *, uint32_t);


/* Functions concerned with compiling a pattern to PCRE internal code. */

#define PCRE2_COMPILE_FUNCTIONS \
PCRE2_EXP_DECL pcre2_code *PCRE2_CALL_CONVENTION \
  pcre2_compile(PCRE2_SPTR, PCRE2_SIZE, uint32_t, int *, PCRE2_SIZE *, \
    pcre2_compile_context *); \
PCRE2_EXP_DECL void PCRE2_CALL_CONVENTION \
  pcre2_code_free(pcre2_code *); \
PCRE2_EXP_DECL pcre2_code *PCRE2_CALL_CONVENTION \
  pcre2_code_copy(const pcre2_code *); \
PCRE2_EXP_DECL pcre2_code *PCRE2_CALL_CONVENTION \
  pcre2_code_copy_with_tables(const pcre2_code *);


/* Functions that give information about a compiled pattern. */

#define PCRE2_PATTERN_INFO_FUNCTIONS \
PCRE2_EXP_DECL int PCRE2_CALL_CONVENTION \
  pcre2_pattern_info(const pcre2_code *, uint32_t, void *); \
PCRE2_EXP_DECL int PCRE2_CALL_CONVENTION \
  pcre2_callout_enumerate(const pcre2_code *, \
    int (*)(pcre2_callout_enumerate_block *, void *), void *);


/* Functions for running a match and inspecting the result. */

#define PCRE2_MATCH_FUNCTIONS \
PCRE2_EXP_DECL pcre2_match_data *PCRE2_CALL_CONVENTION \
  pcre2_match_data_create(uint32_t, pcre2_general_context *); \
PCRE2_EXP_DECL pcre2_match_data *PCRE2_CALL_CONVENTION \
  pcre2_match_data_create_from_pattern(const pcre2_code *, \
    pcre2_general_context *); \
PCRE2_EXP_DECL int PCRE2_CALL_CONVENTION \
  pcre2_dfa_match(const pcre2_code *, PCRE2_SPTR, PCRE2_SIZE, PCRE2_SIZE, \
    uint32_t, pcre2_match_data *, pcre2_match_context *, int *, PCRE2_SIZE); \
PCRE2_EXP_DECL int PCRE2_CALL_CONVENTION \
  pcre2_match(const pcre2_code *, PCRE2_SPTR, PCRE2_SIZE, PCRE2_SIZE, \
    uint32_t, pcre2_match_data *, pcre2_match_context *); \
PCRE2_EXP_DECL void PCRE2_CALL_CONVENTION \
  pcre2_match_data_free(pcre2_match_data *); \
PCRE2_EXP_DECL PCRE2_SPTR PCRE2_CALL_CONVENTION \
  pcre2_get_mark(pcre2_match_data *); \
PCRE2_EXP_DECL PCRE2_SIZE PCRE2_CALL_CONVENTION \
  pcre2_get_match_data_size(pcre2_match_data *); \
PCRE2_EXP_DECL uint32_t PCRE2_CALL_CONVENTION \
  pcre2_get_ovector_count(pcre2_match_data *); \
PCRE2_EXP_DECL PCRE2_SIZE *PCRE2_CALL_CONVENTION \
  pcre2_get_ovector_pointer(pcre2_match_data *); \
PCRE2_EXP_DECL PCRE2_SIZE PCRE2_CALL_CONVENTION \
  pcre2_get_startchar(pcre2_match_data *);


/* Convenience functions for handling matched substrings. */

#define PCRE2_SUBSTRING_FUNCTIONS \
PCRE2_EXP_DECL int PCRE2_CALL_CONVENTION \
  pcre2_substring_copy_byname(pcre2_match_data *, PCRE2_SPTR, PCRE2_UCHAR *, \
    PCRE2_SIZE *); \
PCRE2_EXP_DECL int PCRE2_CALL_CONVENTION \
  pcre2_substring_copy_bynumber(pcre2_match_data *, uint32_t, PCRE2_UCHAR *, \
    PCRE2_SIZE *); \
PCRE2_EXP_DECL void PCRE2_CALL_CONVENTION \
  pcre2_substring_free(PCRE2_UCHAR *); \
PCRE2_EXP_DECL int PCRE2_CALL_CONVENTION \
  pcre2_substring_get_byname(pcre2_match_data *, PCRE2_SPTR, PCRE2_UCHAR **, \
    PCRE2_SIZE *); \
PCRE2_EXP_DECL int PCRE2_CALL_CONVENTION \
  pcre2_substring_get_bynumber(pcre2_match_data *, uint32_t, PCRE2_UCHAR **, \
    PCRE2_SIZE *); \
PCRE2_EXP_DECL int PCRE2_CALL_CONVENTION \
  pcre2_substring_length_byname(pcre2_match_data *, PCRE2_SPTR, PCRE2_SIZE *); \
PCRE2_EXP_DECL int PCRE2_CALL_CONVENTION \
  pcre2_substring_length_bynumber(pcre2_match_data *, uint32_t, PCRE2_SIZE *); \
PCRE2_EXP_DECL int PCRE2_CALL_CONVENTION \
  pcre2_substring_nametable_scan(const pcre2_code *, PCRE2_SPTR, PCRE2_SPTR *, \
    PCRE2_SPTR *); \
PCRE2_EXP_DECL int PCRE2_CALL_CONVENTION \
  pcre2_substring_number_from_name(const pcre2_code *, PCRE2_SPTR); \
PCRE2_EXP_DECL void PCRE2_CALL_CONVENTION \
  pcre2_substring_list_free(PCRE2_SPTR *); \
PCRE2_EXP_DECL int PCRE2_CALL_CONVENTION \
  pcre2_substring_list_get(pcre2_match_data *, PCRE2_UCHAR ***, PCRE2_SIZE **);

/* Functions for serializing / deserializing compiled patterns. */

#define PCRE2_SERIALIZE_FUNCTIONS \
PCRE2_EXP_DECL int32_t PCRE2_CALL_CONVENTION \
  pcre2_serialize_encode(const pcre2_code **, int32_t, uint8_t **, \
    PCRE2_SIZE *, pcre2_general_context *); \
PCRE2_EXP_DECL int32_t PCRE2_CALL_CONVENTION \
  pcre2_serialize_decode(pcre2_code **, int32_t, const uint8_t *, \
    pcre2_general_context *); \
PCRE2_EXP_DECL int32_t PCRE2_CALL_CONVENTION \
  pcre2_serialize_get_number_of_codes(const uint8_t *); \
PCRE2_EXP_DECL void PCRE2_CALL_CONVENTION \
  pcre2_serialize_free(uint8_t *);


/* Convenience function for match + substitute. */

#define PCRE2_SUBSTITUTE_FUNCTION \
PCRE2_EXP_DECL int PCRE2_CALL_CONVENTION \
  pcre2_substitute(const pcre2_code *, PCRE2_SPTR, PCRE2_SIZE, PCRE2_SIZE, \
    uint32_t, pcre2_match_data *, pcre2_match_context *, PCRE2_SPTR, \
    PCRE2_SIZE, PCRE2_UCHAR *, PCRE2_SIZE *);


/* Functions for converting pattern source strings. */

#define PCRE2_CONVERT_FUNCTIONS \
PCRE2_EXP_DECL int PCRE2_CALL_CONVENTION \
  pcre2_pattern_convert(PCRE2_SPTR, PCRE2_SIZE, uint32_t, PCRE2_UCHAR **, \
    PCRE2_SIZE *, pcre2_convert_context *); \
PCRE2_EXP_DECL void PCRE2_CALL_CONVENTION \
  pcre2_converted_pattern_free(PCRE2_UCHAR *);


/* Functions for JIT processing */

#define PCRE2_JIT_FUNCTIONS \
PCRE2_EXP_DECL int PCRE2_CALL_CONVENTION \
  pcre2_jit_compile(pcre2_code *, uint32_t); \
PCRE2_EXP_DECL int PCRE2_CALL_CONVENTION \
  pcre2_jit_match(const pcre2_code *, PCRE2_SPTR, PCRE2_SIZE, PCRE2_SIZE, \
    uint32_t, pcre2_match_data *, pcre2_match_context *); \
PCRE2_EXP_DECL void PCRE2_CALL_CONVENTION \
  pcre2_jit_free_unused_memory(pcre2_general_context *); \
PCRE2_EXP_DECL pcre2_jit_stack *PCRE2_CALL_CONVENTION \
  pcre2_jit_stack_create(PCRE2_SIZE, PCRE2_SIZE, pcre2_general_context *); \
PCRE2_EXP_DECL void PCRE2_CALL_CONVENTION \
  pcre2_jit_stack_assign(pcre2_match_context *, pcre2_jit_callback, void *); \
PCRE2_EXP_DECL void PCRE2_CALL_CONVENTION \
  pcre2_jit_stack_free(pcre2_jit_stack *);


/* Other miscellaneous functions. */

#define PCRE2_OTHER_FUNCTIONS \
PCRE2_EXP_DECL int PCRE2_CALL_CONVENTION \
  pcre2_get_error_message(int, PCRE2_UCHAR *, PCRE2_SIZE); \
PCRE2_EXP_DECL const uint8_t *PCRE2_CALL_CONVENTION \
  pcre2_maketables(pcre2_general_context *); \
PCRE2_EXP_DECL void PCRE2_CALL_CONVENTION \
  pcre2_maketables_free(pcre2_general_context *, const uint8_t *);

/* Define macros that generate width-specific names from generic versions. The
three-level macro scheme is necessary to get the macros expanded when we want
them to be. First we get the width from PCRE2_LOCAL_WIDTH, which is used for
generating three versions of everything below. After that, PCRE2_SUFFIX will be
re-defined to use PCRE2_CODE_UNIT_WIDTH, for use when macros such as
pcre2_compile are called by application code. */

#define PCRE2_JOIN(a,b) a ## b
#define PCRE2_GLUE(a,b) PCRE2_JOIN(a,b)
#define PCRE2_SUFFIX(a) PCRE2_GLUE(a,PCRE2_LOCAL_WIDTH)


/* Data types */

#define PCRE2_UCHAR                 PCRE2_SUFFIX(PCRE2_UCHAR)
#define PCRE2_SPTR                  PCRE2_SUFFIX(PCRE2_SPTR)

#define pcre2_code                  PCRE2_SUFFIX(pcre2_code_)
#define pcre2_jit_callback          PCRE2_SUFFIX(pcre2_jit_callback_)
#define pcre2_jit_stack             PCRE2_SUFFIX(pcre2_jit_stack_)

#define pcre2_real_code             PCRE2_SUFFIX(pcre2_real_code_)
#define pcre2_real_general_context  PCRE2_SUFFIX(pcre2_real_general_context_)
#define pcre2_real_compile_context  PCRE2_SUFFIX(pcre2_real_compile_context_)
#define pcre2_real_convert_context  PCRE2_SUFFIX(pcre2_real_convert_context_)
#define pcre2_real_match_context    PCRE2_SUFFIX(pcre2_real_match_context_)
#define pcre2_real_jit_stack        PCRE2_SUFFIX(pcre2_real_jit_stack_)
#define pcre2_real_match_data       PCRE2_SUFFIX(pcre2_real_match_data_)


/* Data blocks */

#define pcre2_callout_block            PCRE2_SUFFIX(pcre2_callout_block_)
#define pcre2_callout_enumerate_block  PCRE2_SUFFIX(pcre2_callout_enumerate_block_)
#define pcre2_substitute_callout_block PCRE2_SUFFIX(pcre2_substitute_callout_block_)
#define pcre2_general_context          PCRE2_SUFFIX(pcre2_general_context_)
#define pcre2_compile_context          PCRE2_SUFFIX(pcre2_compile_context_)
#define pcre2_convert_context          PCRE2_SUFFIX(pcre2_convert_context_)
#define pcre2_match_context            PCRE2_SUFFIX(pcre2_match_context_)
#define pcre2_match_data               PCRE2_SUFFIX(pcre2_match_data_)


/* Functions: the complete list in alphabetical order */

#define pcre2_callout_enumerate               PCRE2_SUFFIX(pcre2_callout_enumerate_)
#define pcre2_code_copy                       PCRE2_SUFFIX(pcre2_code_copy_)
#define pcre2_code_copy_with_tables           PCRE2_SUFFIX(pcre2_code_copy_with_tables_)
#define pcre2_code_free                       PCRE2_SUFFIX(pcre2_code_free_)
#define pcre2_compile                         PCRE2_SUFFIX(pcre2_compile_)
#define pcre2_compile_context_copy            PCRE2_SUFFIX(pcre2_compile_context_copy_)
#define pcre2_compile_context_create          PCRE2_SUFFIX(pcre2_compile_context_create_)
#define pcre2_compile_context_free            PCRE2_SUFFIX(pcre2_compile_context_free_)
#define pcre2_config                          PCRE2_SUFFIX(pcre2_config_)
#define pcre2_convert_context_copy            PCRE2_SUFFIX(pcre2_convert_context_copy_)
#define pcre2_convert_context_create          PCRE2_SUFFIX(pcre2_convert_context_create_)
#define pcre2_convert_context_free            PCRE2_SUFFIX(pcre2_convert_context_free_)
#define pcre2_converted_pattern_free          PCRE2_SUFFIX(pcre2_converted_pattern_free_)
#define pcre2_dfa_match                       PCRE2_SUFFIX(pcre2_dfa_match_)
#define pcre2_general_context_copy            PCRE2_SUFFIX(pcre2_general_context_copy_)
#define pcre2_general_context_create          PCRE2_SUFFIX(pcre2_general_context_create_)
#define pcre2_general_context_free            PCRE2_SUFFIX(pcre2_general_context_free_)
#define pcre2_get_error_message               PCRE2_SUFFIX(pcre2_get_error_message_)
#define pcre2_get_mark                        PCRE2_SUFFIX(pcre2_get_mark_)
#define pcre2_get_match_data_size             PCRE2_SUFFIX(pcre2_get_match_data_size_)
#define pcre2_get_ovector_pointer             PCRE2_SUFFIX(pcre2_get_ovector_pointer_)
#define pcre2_get_ovector_count               PCRE2_SUFFIX(pcre2_get_ovector_count_)
#define pcre2_get_startchar                   PCRE2_SUFFIX(pcre2_get_startchar_)
#define pcre2_jit_compile                     PCRE2_SUFFIX(pcre2_jit_compile_)
#define pcre2_jit_match                       PCRE2_SUFFIX(pcre2_jit_match_)
#define pcre2_jit_free_unused_memory          PCRE2_SUFFIX(pcre2_jit_free_unused_memory_)
#define pcre2_jit_stack_assign                PCRE2_SUFFIX(pcre2_jit_stack_assign_)
#define pcre2_jit_stack_create                PCRE2_SUFFIX(pcre2_jit_stack_create_)
#define pcre2_jit_stack_free                  PCRE2_SUFFIX(pcre2_jit_stack_free_)
#define pcre2_maketables                      PCRE2_SUFFIX(pcre2_maketables_)
#define pcre2_maketables_free                 PCRE2_SUFFIX(pcre2_maketables_free_)
#define pcre2_match                           PCRE2_SUFFIX(pcre2_match_)
#define pcre2_match_context_copy              PCRE2_SUFFIX(pcre2_match_context_copy_)
#define pcre2_match_context_create            PCRE2_SUFFIX(pcre2_match_context_create_)
#define pcre2_match_context_free              PCRE2_SUFFIX(pcre2_match_context_free_)
#define pcre2_match_data_create               PCRE2_SUFFIX(pcre2_match_data_create_)
#define pcre2_match_data_create_from_pattern  PCRE2_SUFFIX(pcre2_match_data_create_from_pattern_)
#define pcre2_match_data_free                 PCRE2_SUFFIX(pcre2_match_data_free_)
#define pcre2_pattern_convert                 PCRE2_SUFFIX(pcre2_pattern_convert_)
#define pcre2_pattern_info                    PCRE2_SUFFIX(pcre2_pattern_info_)
#define pcre2_serialize_decode                PCRE2_SUFFIX(pcre2_serialize_decode_)
#define pcre2_serialize_encode                PCRE2_SUFFIX(pcre2_serialize_encode_)
#define pcre2_serialize_free                  PCRE2_SUFFIX(pcre2_serialize_free_)
#define pcre2_serialize_get_number_of_codes   PCRE2_SUFFIX(pcre2_serialize_get_number_of_codes_)
#define pcre2_set_bsr                         PCRE2_SUFFIX(pcre2_set_bsr_)
#define pcre2_set_callout                     PCRE2_SUFFIX(pcre2_set_callout_)
#define pcre2_set_character_tables            PCRE2_SUFFIX(pcre2_set_character_tables_)
#define pcre2_set_compile_extra_options       PCRE2_SUFFIX(pcre2_set_compile_extra_options_)
#define pcre2_set_compile_recursion_guard     PCRE2_SUFFIX(pcre2_set_compile_recursion_guard_)
#define pcre2_set_depth_limit                 PCRE2_SUFFIX(pcre2_set_depth_limit_)
#define pcre2_set_glob_escape                 PCRE2_SUFFIX(pcre2_set_glob_escape_)
#define pcre2_set_glob_separator              PCRE2_SUFFIX(pcre2_set_glob_separator_)
#define pcre2_set_heap_limit                  PCRE2_SUFFIX(pcre2_set_heap_limit_)
#define pcre2_set_match_limit                 PCRE2_SUFFIX(pcre2_set_match_limit_)
#define pcre2_set_max_pattern_length          PCRE2_SUFFIX(pcre2_set_max_pattern_length_)
#define pcre2_set_newline                     PCRE2_SUFFIX(pcre2_set_newline_)
#define pcre2_set_parens_nest_limit           PCRE2_SUFFIX(pcre2_set_parens_nest_limit_)
#define pcre2_set_offset_limit                PCRE2_SUFFIX(pcre2_set_offset_limit_)
#define pcre2_set_substitute_callout          PCRE2_SUFFIX(pcre2_set_substitute_callout_)
#define pcre2_substitute                      PCRE2_SUFFIX(pcre2_substitute_)
#define pcre2_substring_copy_byname           PCRE2_SUFFIX(pcre2_substring_copy_byname_)
#define pcre2_substring_copy_bynumber         PCRE2_SUFFIX(pcre2_substring_copy_bynumber_)
#define pcre2_substring_free                  PCRE2_SUFFIX(pcre2_substring_free_)
#define pcre2_substring_get_byname            PCRE2_SUFFIX(pcre2_substring_get_byname_)
#define pcre2_substring_get_bynumber          PCRE2_SUFFIX(pcre2_substring_get_bynumber_)
#define pcre2_substring_length_byname         PCRE2_SUFFIX(pcre2_substring_length_byname_)
#define pcre2_substring_length_bynumber       PCRE2_SUFFIX(pcre2_substring_length_bynumber_)
#define pcre2_substring_list_get              PCRE2_SUFFIX(pcre2_substring_list_get_)
#define pcre2_substring_list_free             PCRE2_SUFFIX(pcre2_substring_list_free_)
#define pcre2_substring_nametable_scan        PCRE2_SUFFIX(pcre2_substring_nametable_scan_)
#define pcre2_substring_number_from_name      PCRE2_SUFFIX(pcre2_substring_number_from_name_)

/* Keep this old function name for backwards compatibility */
#define pcre2_set_recursion_limit PCRE2_SUFFIX(pcre2_set_recursion_limit_)

/* Keep this obsolete function for backwards compatibility: it is now a noop. */
#define pcre2_set_recursion_memory_management PCRE2_SUFFIX(pcre2_set_recursion_memory_management_)

/* Now generate all three sets of width-specific structures and function
prototypes. */

#define PCRE2_TYPES_STRUCTURES_AND_FUNCTIONS \
PCRE2_TYPES_LIST \
PCRE2_STRUCTURE_LIST \
PCRE2_GENERAL_INFO_FUNCTIONS \
PCRE2_GENERAL_CONTEXT_FUNCTIONS \
PCRE2_COMPILE_CONTEXT_FUNCTIONS \
PCRE2_CONVERT_CONTEXT_FUNCTIONS \
PCRE2_CONVERT_FUNCTIONS \
PCRE2_MATCH_CONTEXT_FUNCTIONS \
PCRE2_COMPILE_FUNCTIONS \
PCRE2_PATTERN_INFO_FUNCTIONS \
PCRE2_MATCH_FUNCTIONS \
PCRE2_SUBSTRING_FUNCTIONS \
PCRE2_SERIALIZE_FUNCTIONS \
PCRE2_SUBSTITUTE_FUNCTION \
PCRE2_JIT_FUNCTIONS \
PCRE2_OTHER_FUNCTIONS

#define PCRE2_LOCAL_WIDTH 8
PCRE2_TYPES_STRUCTURES_AND_FUNCTIONS
#undef PCRE2_LOCAL_WIDTH

#define PCRE2_LOCAL_WIDTH 16
PCRE2_TYPES_STRUCTURES_AND_FUNCTIONS
#undef PCRE2_LOCAL_WIDTH

#define PCRE2_LOCAL_WIDTH 32
PCRE2_TYPES_STRUCTURES_AND_FUNCTIONS
#undef PCRE2_LOCAL_WIDTH

/* Undefine the list macros; they are no longer needed. */

#undef PCRE2_TYPES_LIST
#undef PCRE2_STRUCTURE_LIST
#undef PCRE2_GENERAL_INFO_FUNCTIONS
#undef PCRE2_GENERAL_CONTEXT_FUNCTIONS
#undef PCRE2_COMPILE_CONTEXT_FUNCTIONS
#undef PCRE2_CONVERT_CONTEXT_FUNCTIONS
#undef PCRE2_MATCH_CONTEXT_FUNCTIONS
#undef PCRE2_COMPILE_FUNCTIONS
#undef PCRE2_PATTERN_INFO_FUNCTIONS
#undef PCRE2_MATCH_FUNCTIONS
#undef PCRE2_SUBSTRING_FUNCTIONS
#undef PCRE2_SERIALIZE_FUNCTIONS
#undef PCRE2_SUBSTITUTE_FUNCTION
#undef PCRE2_JIT_FUNCTIONS
#undef PCRE2_OTHER_FUNCTIONS
#undef PCRE2_TYPES_STRUCTURES_AND_FUNCTIONS

/* PCRE2_CODE_UNIT_WIDTH must be defined. If it is 8, 16, or 32, redefine
PCRE2_SUFFIX to use it. If it is 0, undefine the other macros and make
PCRE2_SUFFIX a no-op. Otherwise, generate an error. */

#undef PCRE2_SUFFIX
#ifndef PCRE2_CODE_UNIT_WIDTH
#error PCRE2_CODE_UNIT_WIDTH must be defined before including pcre2.h.
#error Use 8, 16, or 32; or 0 for a multi-width application.
#else  /* PCRE2_CODE_UNIT_WIDTH is defined */
#if PCRE2_CODE_UNIT_WIDTH == 8 || \
    PCRE2_CODE_UNIT_WIDTH == 16 || \
    PCRE2_CODE_UNIT_WIDTH == 32
#define PCRE2_SUFFIX(a) PCRE2_GLUE(a, PCRE2_CODE_UNIT_WIDTH)
#elif PCRE2_CODE_UNIT_WIDTH == 0
#undef PCRE2_JOIN
#undef PCRE2_GLUE
#define PCRE2_SUFFIX(a) a
#else
#error PCRE2_CODE_UNIT_WIDTH must be 0, 8, 16, or 32.
#endif
#endif  /* PCRE2_CODE_UNIT_WIDTH is defined */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* PCRE2_H_IDEMPOTENT_GUARD */

/* End of pcre2.h */
