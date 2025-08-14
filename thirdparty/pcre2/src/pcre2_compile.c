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


#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#define NLBLOCK cb             /* Block containing newline information */
#define PSSTART start_pattern  /* Field containing processed string start */
#define PSEND   end_pattern    /* Field containing processed string end */

#include "pcre2_compile.h"

/* In rare error cases debugging might require calling pcre2_printint(). */

#if 0
#ifdef EBCDIC
#define PRINTABLE(c) ((c) >= 64 && (c) < 255)
#else
#define PRINTABLE(c) ((c) >= 32 && (c) < 127)
#endif
#include "pcre2_printint.c"
#define DEBUG_CALL_PRINTINT
#endif

/* Other debugging code can be enabled by these defines. */

/* #define DEBUG_SHOW_CAPTURES */
/* #define DEBUG_SHOW_PARSED */

/* There are a few things that vary with different code unit sizes. Handle them
by defining macros in order to minimize #if usage. */

#if PCRE2_CODE_UNIT_WIDTH == 8
#define STRING_UTFn_RIGHTPAR     STRING_UTF8_RIGHTPAR, 5
#define XDIGIT(c)                xdigitab[c]

#else  /* Either 16-bit or 32-bit */
#define XDIGIT(c)                (MAX_255(c)? xdigitab[c] : 0xff)

#if PCRE2_CODE_UNIT_WIDTH == 16
#define STRING_UTFn_RIGHTPAR     STRING_UTF16_RIGHTPAR, 6

#else  /* 32-bit */
#define STRING_UTFn_RIGHTPAR     STRING_UTF32_RIGHTPAR, 6
#endif
#endif

/* Macros to store and retrieve a PCRE2_SIZE value in the parsed pattern, which
consists of uint32_t elements. Assume that if uint32_t can't hold it, two of
them will be able to (i.e. assume a 64-bit world). */

#if PCRE2_SIZE_MAX <= UINT32_MAX
#define PUTOFFSET(s,p) *p++ = s
#define GETOFFSET(s,p) s = *p++
#define GETPLUSOFFSET(s,p) s = *(++p)
#define READPLUSOFFSET(s,p) s = p[1]
#define SKIPOFFSET(p) p++
#define SIZEOFFSET 1
#else
#define PUTOFFSET(s,p) \
  { *p++ = (uint32_t)(s >> 32); *p++ = (uint32_t)(s & 0xffffffff); }
#define GETOFFSET(s,p) \
  { s = ((PCRE2_SIZE)p[0] << 32) | (PCRE2_SIZE)p[1]; p += 2; }
#define GETPLUSOFFSET(s,p) \
  { s = ((PCRE2_SIZE)p[1] << 32) | (PCRE2_SIZE)p[2]; p += 2; }
#define READPLUSOFFSET(s,p) \
  { s = ((PCRE2_SIZE)p[1] << 32) | (PCRE2_SIZE)p[2]; }
#define SKIPOFFSET(p) p += 2
#define SIZEOFFSET 2
#endif

/* Function definitions to allow mutual recursion */

static int
  compile_regex(uint32_t, uint32_t, PCRE2_UCHAR **, uint32_t **, int *,
    uint32_t, uint32_t *, uint32_t *, uint32_t *, uint32_t *, branch_chain *,
    open_capitem *, compile_block *, PCRE2_SIZE *);

static int
  get_branchlength(uint32_t **, int *, int *, int *, parsed_recurse_check *,
    compile_block *);

static BOOL
  set_lookbehind_lengths(uint32_t **, int *, int *, parsed_recurse_check *,
    compile_block *);

static int
  check_lookbehinds(uint32_t *, uint32_t **, parsed_recurse_check *,
    compile_block *, int *);


/*************************************************
*      Code parameters and static tables         *
*************************************************/

#define MAX_GROUP_NUMBER   65535u
#define MAX_REPEAT_COUNT   65535u
#define REPEAT_UNLIMITED   (MAX_REPEAT_COUNT+1)

/* COMPILE_WORK_SIZE specifies the size of stack workspace, which is used in
different ways in the different pattern scans. The parsing and group-
identifying pre-scan uses it to handle nesting, and needs it to be 16-bit
aligned for this. Having defined the size in code units, we set up
C16_WORK_SIZE as the number of elements in the 16-bit vector.

During the first compiling phase, when determining how much memory is required,
the regex is partly compiled into this space, but the compiled parts are
discarded as soon as they can be, so that hopefully there will never be an
overrun. The code does, however, check for an overrun, which can occur for
pathological patterns. The size of the workspace depends on LINK_SIZE because
the length of compiled items varies with this.

In the real compile phase, this workspace is not currently used. */

#define COMPILE_WORK_SIZE (3000*LINK_SIZE)   /* Size in code units */

#define C16_WORK_SIZE \
  ((COMPILE_WORK_SIZE * sizeof(PCRE2_UCHAR))/sizeof(uint16_t))

/* A uint32_t vector is used for caching information about the size of
capturing groups, to improve performance. A default is created on the stack of
this size. */

#define GROUPINFO_DEFAULT_SIZE 256

/* The overrun tests check for a slightly smaller size so that they detect the
overrun before it actually does run off the end of the data block. */

#define WORK_SIZE_SAFETY_MARGIN (100)

/* This value determines the size of the initial vector that is used for
remembering named groups during the pre-compile. It is allocated on the stack,
but if it is too small, it is expanded, in a similar way to the workspace. The
value is the number of slots in the list. */

#define NAMED_GROUP_LIST_SIZE  20

/* The pre-compiling pass over the pattern creates a parsed pattern in a vector
of uint32_t. For short patterns this lives on the stack, with this size. Heap
memory is used for longer patterns. */

#define PARSED_PATTERN_DEFAULT_SIZE 1024

/* Maximum length value to check against when making sure that the variable
that holds the compiled pattern length does not overflow. We make it a bit less
than INT_MAX to allow for adding in group terminating code units, so that we
don't have to check them every time. */

#define OFLOW_MAX (INT_MAX - 20)

/* Table of extra lengths for each of the meta codes. Must be kept in step with
the definitions above. For some items these values are a basic length to which
a variable amount has to be added. */

static unsigned char meta_extra_lengths[] = {
  0,             /* META_END */
  0,             /* META_ALT */
  0,             /* META_ATOMIC */
  0,             /* META_BACKREF - more if group is >= 10 */
  1+SIZEOFFSET,  /* META_BACKREF_BYNAME */
  1,             /* META_BIGVALUE */
  3,             /* META_CALLOUT_NUMBER */
  3+SIZEOFFSET,  /* META_CALLOUT_STRING */
  0,             /* META_CAPTURE */
  0,             /* META_CIRCUMFLEX */
  0,             /* META_CLASS */
  0,             /* META_CLASS_EMPTY */
  0,             /* META_CLASS_EMPTY_NOT */
  0,             /* META_CLASS_END */
  0,             /* META_CLASS_NOT */
  0,             /* META_COND_ASSERT */
  SIZEOFFSET,    /* META_COND_DEFINE */
  1+SIZEOFFSET,  /* META_COND_NAME */
  1+SIZEOFFSET,  /* META_COND_NUMBER */
  1+SIZEOFFSET,  /* META_COND_RNAME */
  1+SIZEOFFSET,  /* META_COND_RNUMBER */
  3,             /* META_COND_VERSION */
  SIZEOFFSET,    /* META_OFFSET */
  0,             /* META_SCS */
  1,             /* META_SCS_NAME */
  1,             /* META_SCS_NUMBER */
  0,             /* META_DOLLAR */
  0,             /* META_DOT */
  0,             /* META_ESCAPE - one more for ESC_P and ESC_p */
  0,             /* META_KET */
  0,             /* META_NOCAPTURE */
  2,             /* META_OPTIONS */
  1,             /* META_POSIX */
  1,             /* META_POSIX_NEG */
  0,             /* META_RANGE_ESCAPED */
  0,             /* META_RANGE_LITERAL */
  SIZEOFFSET,    /* META_RECURSE */
  1+SIZEOFFSET,  /* META_RECURSE_BYNAME */
  0,             /* META_SCRIPT_RUN */
  0,             /* META_LOOKAHEAD */
  0,             /* META_LOOKAHEADNOT */
  SIZEOFFSET,    /* META_LOOKBEHIND */
  SIZEOFFSET,    /* META_LOOKBEHINDNOT */
  0,             /* META_LOOKAHEAD_NA */
  SIZEOFFSET,    /* META_LOOKBEHIND_NA */
  1,             /* META_MARK - plus the string length */
  0,             /* META_ACCEPT */
  0,             /* META_FAIL */
  0,             /* META_COMMIT */
  1,             /* META_COMMIT_ARG - plus the string length */
  0,             /* META_PRUNE */
  1,             /* META_PRUNE_ARG - plus the string length */
  0,             /* META_SKIP */
  1,             /* META_SKIP_ARG - plus the string length */
  0,             /* META_THEN */
  1,             /* META_THEN_ARG - plus the string length */
  0,             /* META_ASTERISK */
  0,             /* META_ASTERISK_PLUS */
  0,             /* META_ASTERISK_QUERY */
  0,             /* META_PLUS */
  0,             /* META_PLUS_PLUS */
  0,             /* META_PLUS_QUERY */
  0,             /* META_QUERY */
  0,             /* META_QUERY_PLUS */
  0,             /* META_QUERY_QUERY */
  2,             /* META_MINMAX */
  2,             /* META_MINMAX_PLUS */
  2,             /* META_MINMAX_QUERY */
  0,             /* META_ECLASS_AND */
  0,             /* META_ECLASS_OR */
  0,             /* META_ECLASS_SUB */
  0,             /* META_ECLASS_XOR */
  0              /* META_ECLASS_NOT */
};

/* Types for skipping parts of a parsed pattern. */

enum { PSKIP_ALT, PSKIP_CLASS, PSKIP_KET };

/* Values and flags for the unsigned xxcuflags variables that accompany xxcu
variables, which are concerned with first and required code units. A value
greater than or equal to REQ_NONE means "no code unit set"; otherwise the
matching xxcu variable is set, and the low valued bits are relevant. */

#define REQ_UNSET     0xffffffffu  /* Not yet found anything */
#define REQ_NONE      0xfffffffeu  /* Found not fixed character */
#define REQ_CASELESS  0x00000001u  /* Code unit in xxcu is caseless */
#define REQ_VARY      0x00000002u  /* Code unit is followed by non-literal */

/* These flags are used in the groupinfo vector. */

#define GI_SET_FIXED_LENGTH    0x80000000u
#define GI_NOT_FIXED_LENGTH    0x40000000u
#define GI_FIXED_LENGTH_MASK   0x0000ffffu

/* This simple test for a decimal digit works for both ASCII/Unicode and EBCDIC
and is fast (a good compiler can turn it into a subtraction and unsigned
comparison). */

#define IS_DIGIT(x) ((x) >= CHAR_0 && (x) <= CHAR_9)

/* Table to identify hex digits. The tables in chartables are dependent on the
locale, and may mark arbitrary characters as digits. We want to recognize only
0-9, a-z, and A-Z as hex digits, which is why we have a private table here. It
costs 256 bytes, but it is a lot faster than doing character value tests (at
least in some simple cases I timed), and in some applications one wants PCRE2
to compile efficiently as well as match efficiently. The value in the table is
the binary hex digit value, or 0xff for non-hex digits. */

/* This is the "normal" case, for ASCII systems, and EBCDIC systems running in
UTF-8 mode. */

#ifndef EBCDIC
static const uint8_t xdigitab[] =
  {
  0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff, /*   0-  7 */
  0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff, /*   8- 15 */
  0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff, /*  16- 23 */
  0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff, /*  24- 31 */
  0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff, /*    - '  */
  0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff, /*  ( - /  */
  0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07, /*  0 - 7  */
  0x08,0x09,0xff,0xff,0xff,0xff,0xff,0xff, /*  8 - ?  */
  0xff,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f,0xff, /*  @ - G  */
  0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff, /*  H - O  */
  0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff, /*  P - W  */
  0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff, /*  X - _  */
  0xff,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f,0xff, /*  ` - g  */
  0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff, /*  h - o  */
  0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff, /*  p - w  */
  0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff, /*  x -127 */
  0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff, /* 128-135 */
  0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff, /* 136-143 */
  0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff, /* 144-151 */
  0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff, /* 152-159 */
  0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff, /* 160-167 */
  0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff, /* 168-175 */
  0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff, /* 176-183 */
  0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff, /* 184-191 */
  0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff, /* 192-199 */
  0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff, /* 2ff-207 */
  0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff, /* 208-215 */
  0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff, /* 216-223 */
  0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff, /* 224-231 */
  0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff, /* 232-239 */
  0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff, /* 240-247 */
  0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff};/* 248-255 */

#else

/* This is the "abnormal" case, for EBCDIC systems not running in UTF-8 mode. */

static const uint8_t xdigitab[] =
  {
  0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff, /*   0-  7  0 */
  0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff, /*   8- 15    */
  0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff, /*  16- 23 10 */
  0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff, /*  24- 31    */
  0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff, /*  32- 39 20 */
  0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff, /*  40- 47    */
  0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff, /*  48- 55 30 */
  0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff, /*  56- 63    */
  0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff, /*    - 71 40 */
  0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff, /*  72- |     */
  0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff, /*  & - 87 50 */
  0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff, /*  88- 95    */
  0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff, /*  - -103 60 */
  0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff, /* 104- ?     */
  0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff, /* 112-119 70 */
  0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff, /* 120- "     */
  0xff,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f,0xff, /* 128- g  80 */
  0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff, /*  h -143    */
  0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff, /* 144- p  90 */
  0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff, /*  q -159    */
  0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff, /* 160- x  A0 */
  0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff, /*  y -175    */
  0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff, /*  ^ -183 B0 */
  0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff, /* 184-191    */
  0xff,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f,0xff, /*  { - G  C0 */
  0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff, /*  H -207    */
  0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff, /*  } - P  D0 */
  0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff, /*  Q -223    */
  0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff, /*  \ - X  E0 */
  0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff, /*  Y -239    */
  0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07, /*  0 - 7  F0 */
  0x08,0x09,0xff,0xff,0xff,0xff,0xff,0xff};/*  8 -255    */
#endif  /* EBCDIC */


/* Table for handling alphanumeric escaped characters. Positive returns are
simple data values; negative values are for special things like \d and so on.
Zero means further processing is needed (for things like \x), or the escape is
invalid. */

/* This is the "normal" table for ASCII systems or for EBCDIC systems running
in UTF-8 mode. It runs from '0' to 'z'. */

#ifndef EBCDIC
#define ESCAPES_FIRST       CHAR_0
#define ESCAPES_LAST        CHAR_z
#define UPPER_CASE(c)       (c-32)

static const short int escapes[] = {
    /* 0 */ 0,                       /* 1 */ 0,
    /* 2 */ 0,                       /* 3 */ 0,
    /* 4 */ 0,                       /* 5 */ 0,
    /* 6 */ 0,                       /* 7 */ 0,
    /* 8 */ 0,                       /* 9 */ 0,
    /* : */ CHAR_COLON,              /* ; */ CHAR_SEMICOLON,
    /* < */ CHAR_LESS_THAN_SIGN,     /* = */ CHAR_EQUALS_SIGN,
    /* > */ CHAR_GREATER_THAN_SIGN,  /* ? */ CHAR_QUESTION_MARK,
    /* @ */ CHAR_COMMERCIAL_AT,      /* A */ -ESC_A,
    /* B */ -ESC_B,                  /* C */ -ESC_C,
    /* D */ -ESC_D,                  /* E */ -ESC_E,
    /* F */ 0,                       /* G */ -ESC_G,
    /* H */ -ESC_H,                  /* I */ 0,
    /* J */ 0,                       /* K */ -ESC_K,
    /* L */ 0,                       /* M */ 0,
    /* N */ -ESC_N,                  /* O */ 0,
    /* P */ -ESC_P,                  /* Q */ -ESC_Q,
    /* R */ -ESC_R,                  /* S */ -ESC_S,
    /* T */ 0,                       /* U */ 0,
    /* V */ -ESC_V,                  /* W */ -ESC_W,
    /* X */ -ESC_X,                  /* Y */ 0,
    /* Z */ -ESC_Z,                  /* [ */ CHAR_LEFT_SQUARE_BRACKET,
    /* \ */ CHAR_BACKSLASH,          /* ] */ CHAR_RIGHT_SQUARE_BRACKET,
    /* ^ */ CHAR_CIRCUMFLEX_ACCENT,  /* _ */ CHAR_UNDERSCORE,
    /* ` */ CHAR_GRAVE_ACCENT,       /* a */ CHAR_BEL,
    /* b */ -ESC_b,                  /* c */ 0,
    /* d */ -ESC_d,                  /* e */ CHAR_ESC,
    /* f */ CHAR_FF,                 /* g */ 0,
    /* h */ -ESC_h,                  /* i */ 0,
    /* j */ 0,                       /* k */ -ESC_k,
    /* l */ 0,                       /* m */ 0,
    /* n */ CHAR_LF,                 /* o */ 0,
    /* p */ -ESC_p,                  /* q */ 0,
    /* r */ CHAR_CR,                 /* s */ -ESC_s,
    /* t */ CHAR_HT,                 /* u */ 0,
    /* v */ -ESC_v,                  /* w */ -ESC_w,
    /* x */ 0,                       /* y */ 0,
    /* z */ -ESC_z
};

#else

/* This is the "abnormal" table for EBCDIC systems without UTF-8 support.
It runs from 'a' to '9'. For some minimal testing of EBCDIC features, the code
is sometimes compiled on an ASCII system. In this case, we must not use CHAR_a
because it is defined as 'a', which of course picks up the ASCII value. */

#if 'a' == 0x81                    /* Check for a real EBCDIC environment */
#define ESCAPES_FIRST       CHAR_a
#define ESCAPES_LAST        CHAR_9
#define UPPER_CASE(c)       (c+64)
#else                              /* Testing in an ASCII environment */
#define ESCAPES_FIRST  ((unsigned char)'\x81')   /* EBCDIC 'a' */
#define ESCAPES_LAST   ((unsigned char)'\xf9')   /* EBCDIC '9' */
#define UPPER_CASE(c)  (c-32)
#endif

static const short int escapes[] = {
/*  80 */         CHAR_BEL, -ESC_b,       0, -ESC_d, CHAR_ESC, CHAR_FF,      0,
/*  88 */ -ESC_h,        0,      0,     '{',      0,        0,       0,      0,
/*  90 */      0,        0, -ESC_k,       0,      0,  CHAR_LF,       0, -ESC_p,
/*  98 */      0,  CHAR_CR,      0,     '}',      0,        0,       0,      0,
/*  A0 */      0,      '~', -ESC_s, CHAR_HT,      0,   -ESC_v,  -ESC_w,      0,
/*  A8 */      0,   -ESC_z,      0,       0,      0,      '[',       0,      0,
/*  B0 */      0,        0,      0,       0,      0,        0,       0,      0,
/*  B8 */      0,        0,      0,       0,      0,      ']',     '=',    '-',
/*  C0 */    '{',   -ESC_A, -ESC_B,  -ESC_C, -ESC_D,   -ESC_E,       0, -ESC_G,
/*  C8 */ -ESC_H,        0,      0,       0,      0,        0,       0,      0,
/*  D0 */    '}',        0, -ESC_K,       0,      0,   -ESC_N,       0, -ESC_P,
/*  D8 */ -ESC_Q,   -ESC_R,      0,       0,      0,        0,       0,      0,
/*  E0 */   '\\',        0, -ESC_S,       0,      0,   -ESC_V,  -ESC_W, -ESC_X,
/*  E8 */      0,   -ESC_Z,      0,       0,      0,        0,       0,      0,
/*  F0 */      0,        0,      0,       0,      0,        0,       0,      0,
/*  F8 */      0,        0
};

/* We also need a table of characters that may follow \c in an EBCDIC
environment for characters 0-31. */

static unsigned char ebcdic_escape_c[] = "@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_";

#endif   /* EBCDIC */


/* Table of special "verbs" like (*PRUNE). This is a short table, so it is
searched linearly. Put all the names into a single string, in order to reduce
the number of relocations when a shared library is dynamically linked. The
string is built from string macros so that it works in UTF-8 mode on EBCDIC
platforms. */

typedef struct verbitem {
  unsigned int len;          /* Length of verb name */
  uint32_t meta;             /* Base META_ code */
  int has_arg;               /* Argument requirement */
} verbitem;

static const char verbnames[] =
  "\0"                       /* Empty name is a shorthand for MARK */
  STRING_MARK0
  STRING_ACCEPT0
  STRING_F0
  STRING_FAIL0
  STRING_COMMIT0
  STRING_PRUNE0
  STRING_SKIP0
  STRING_THEN;

static const verbitem verbs[] = {
  { 0, META_MARK,   +1 },  /* > 0 => must have an argument */
  { 4, META_MARK,   +1 },
  { 6, META_ACCEPT, -1 },  /* < 0 => Optional argument, convert to pre-MARK */
  { 1, META_FAIL,   -1 },
  { 4, META_FAIL,   -1 },
  { 6, META_COMMIT,  0 },
  { 5, META_PRUNE,   0 },  /* Optional argument; bump META code if found */
  { 4, META_SKIP,    0 },
  { 4, META_THEN,    0 }
};

static const int verbcount = sizeof(verbs)/sizeof(verbitem);

/* Verb opcodes, indexed by their META code offset from META_MARK. */

static const uint32_t verbops[] = {
  OP_MARK, OP_ACCEPT, OP_FAIL, OP_COMMIT, OP_COMMIT_ARG, OP_PRUNE,
  OP_PRUNE_ARG, OP_SKIP, OP_SKIP_ARG, OP_THEN, OP_THEN_ARG };

/* Table of "alpha assertions" like (*pla:...), similar to the (*VERB) table. */

typedef struct alasitem {
  unsigned int len;          /* Length of name */
  uint32_t meta;             /* Base META_ code */
} alasitem;

static const char alasnames[] =
  STRING_pla0
  STRING_plb0
  STRING_napla0
  STRING_naplb0
  STRING_nla0
  STRING_nlb0
  STRING_positive_lookahead0
  STRING_positive_lookbehind0
  STRING_non_atomic_positive_lookahead0
  STRING_non_atomic_positive_lookbehind0
  STRING_negative_lookahead0
  STRING_negative_lookbehind0
  STRING_scs0
  STRING_scan_substring0
  STRING_atomic0
  STRING_sr0
  STRING_asr0
  STRING_script_run0
  STRING_atomic_script_run;

static const alasitem alasmeta[] = {
  {  3, META_LOOKAHEAD         },
  {  3, META_LOOKBEHIND        },
  {  5, META_LOOKAHEAD_NA      },
  {  5, META_LOOKBEHIND_NA     },
  {  3, META_LOOKAHEADNOT      },
  {  3, META_LOOKBEHINDNOT     },
  { 18, META_LOOKAHEAD         },
  { 19, META_LOOKBEHIND        },
  { 29, META_LOOKAHEAD_NA      },
  { 30, META_LOOKBEHIND_NA     },
  { 18, META_LOOKAHEADNOT      },
  { 19, META_LOOKBEHINDNOT     },
  {  3, META_SCS               },
  { 14, META_SCS               },
  {  6, META_ATOMIC            },
  {  2, META_SCRIPT_RUN        }, /* sr = script run */
  {  3, META_ATOMIC_SCRIPT_RUN }, /* asr = atomic script run */
  { 10, META_SCRIPT_RUN        }, /* script run */
  { 17, META_ATOMIC_SCRIPT_RUN }  /* atomic script run */
};

static const int alascount = sizeof(alasmeta)/sizeof(alasitem);

/* Offsets from OP_STAR for case-independent and negative repeat opcodes. */

static uint32_t chartypeoffset[] = {
  OP_STAR - OP_STAR,    OP_STARI - OP_STAR,
  OP_NOTSTAR - OP_STAR, OP_NOTSTARI - OP_STAR };

/* Tables of names of POSIX character classes and their lengths. The names are
now all in a single string, to reduce the number of relocations when a shared
library is dynamically loaded. The list of lengths is terminated by a zero
length entry. The first three must be alpha, lower, upper, as this is assumed
for handling case independence.

The indices for several classes are stored in pcre2_compile.h - these must
be kept in sync with posix_names, posix_name_lengths, posix_class_maps,
and posix_substitutes. */

static const char posix_names[] =
  STRING_alpha0 STRING_lower0 STRING_upper0 STRING_alnum0
  STRING_ascii0 STRING_blank0 STRING_cntrl0 STRING_digit0
  STRING_graph0 STRING_print0 STRING_punct0 STRING_space0
  STRING_word0  STRING_xdigit;

static const uint8_t posix_name_lengths[] = {
  5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 6, 0 };

/* Table of class bit maps for each POSIX class. Each class is formed from a
base map, with an optional addition or removal of another map. Then, for some
classes, there is some additional tweaking: for [:blank:] the vertical space
characters are removed, and for [:alpha:] and [:alnum:] the underscore
character is removed. The triples in the table consist of the base map offset,
second map offset or -1 if no second map, and a non-negative value for map
addition or a negative value for map subtraction (if there are two maps). The
absolute value of the third field has these meanings: 0 => no tweaking, 1 =>
remove vertical space characters, 2 => remove underscore. */

const int PRIV(posix_class_maps)[] = {
  cbit_word,   cbit_digit, -2,            /* alpha */
  cbit_lower,  -1,          0,            /* lower */
  cbit_upper,  -1,          0,            /* upper */
  cbit_word,   -1,          2,            /* alnum - word without underscore */
  cbit_print,  cbit_cntrl,  0,            /* ascii */
  cbit_space,  -1,          1,            /* blank - a GNU extension */
  cbit_cntrl,  -1,          0,            /* cntrl */
  cbit_digit,  -1,          0,            /* digit */
  cbit_graph,  -1,          0,            /* graph */
  cbit_print,  -1,          0,            /* print */
  cbit_punct,  -1,          0,            /* punct */
  cbit_space,  -1,          0,            /* space */
  cbit_word,   -1,          0,            /* word - a Perl extension */
  cbit_xdigit, -1,          0             /* xdigit */
};

#ifdef SUPPORT_UNICODE

/* The POSIX class Unicode property substitutes that are used in UCP mode must
be in the order of the POSIX class names, defined above. */

static int posix_substitutes[] = {
  PT_GC, ucp_L,     /* alpha */
  PT_PC, ucp_Ll,    /* lower */
  PT_PC, ucp_Lu,    /* upper */
  PT_ALNUM, 0,      /* alnum */
  -1, 0,            /* ascii, treat as non-UCP */
  -1, 1,            /* blank, treat as \h */
  PT_PC, ucp_Cc,    /* cntrl */
  PT_PC, ucp_Nd,    /* digit */
  PT_PXGRAPH, 0,    /* graph */
  PT_PXPRINT, 0,    /* print */
  PT_PXPUNCT, 0,    /* punct */
  PT_PXSPACE, 0,    /* space */   /* Xps is POSIX space, but from 8.34 */
  PT_WORD, 0,       /* word  */   /* Perl and POSIX space are the same */
  PT_PXXDIGIT, 0    /* xdigit */  /* Perl has additional hex digits */
};
#endif  /* SUPPORT_UNICODE */

/* Masks for checking option settings. When PCRE2_LITERAL is set, only a subset
are allowed. */

#define PUBLIC_LITERAL_COMPILE_OPTIONS \
  (PCRE2_ANCHORED|PCRE2_AUTO_CALLOUT|PCRE2_CASELESS|PCRE2_ENDANCHORED| \
   PCRE2_FIRSTLINE|PCRE2_LITERAL|PCRE2_MATCH_INVALID_UTF| \
   PCRE2_NO_START_OPTIMIZE|PCRE2_NO_UTF_CHECK|PCRE2_USE_OFFSET_LIMIT|PCRE2_UTF)

#define PUBLIC_COMPILE_OPTIONS \
  (PUBLIC_LITERAL_COMPILE_OPTIONS| \
   PCRE2_ALLOW_EMPTY_CLASS|PCRE2_ALT_BSUX|PCRE2_ALT_CIRCUMFLEX| \
   PCRE2_ALT_VERBNAMES|PCRE2_DOLLAR_ENDONLY|PCRE2_DOTALL|PCRE2_DUPNAMES| \
   PCRE2_EXTENDED|PCRE2_EXTENDED_MORE|PCRE2_MATCH_UNSET_BACKREF| \
   PCRE2_MULTILINE|PCRE2_NEVER_BACKSLASH_C|PCRE2_NEVER_UCP| \
   PCRE2_NEVER_UTF|PCRE2_NO_AUTO_CAPTURE|PCRE2_NO_AUTO_POSSESS| \
   PCRE2_NO_DOTSTAR_ANCHOR|PCRE2_UCP|PCRE2_UNGREEDY|PCRE2_ALT_EXTENDED_CLASS)

#define PUBLIC_LITERAL_COMPILE_EXTRA_OPTIONS \
   (PCRE2_EXTRA_MATCH_LINE|PCRE2_EXTRA_MATCH_WORD| \
    PCRE2_EXTRA_CASELESS_RESTRICT|PCRE2_EXTRA_TURKISH_CASING)

#define PUBLIC_COMPILE_EXTRA_OPTIONS \
   (PUBLIC_LITERAL_COMPILE_EXTRA_OPTIONS| \
    PCRE2_EXTRA_ALLOW_SURROGATE_ESCAPES|PCRE2_EXTRA_BAD_ESCAPE_IS_LITERAL| \
    PCRE2_EXTRA_ESCAPED_CR_IS_LF|PCRE2_EXTRA_ALT_BSUX| \
    PCRE2_EXTRA_ALLOW_LOOKAROUND_BSK|PCRE2_EXTRA_ASCII_BSD| \
    PCRE2_EXTRA_ASCII_BSS|PCRE2_EXTRA_ASCII_BSW|PCRE2_EXTRA_ASCII_POSIX| \
    PCRE2_EXTRA_ASCII_DIGIT|PCRE2_EXTRA_PYTHON_OCTAL|PCRE2_EXTRA_NO_BS0| \
    PCRE2_EXTRA_NEVER_CALLOUT)

/* This is a table of start-of-pattern options such as (*UTF) and settings such
as (*LIMIT_MATCH=nnnn) and (*CRLF). For completeness and backward
compatibility, (*UTFn) is supported in the relevant libraries, but (*UTF) is
generic and always supported. */

enum { PSO_OPT,     /* Value is an option bit */
       PSO_XOPT,    /* Value is an xoption bit */
       PSO_FLG,     /* Value is a flag bit */
       PSO_NL,      /* Value is a newline type */
       PSO_BSR,     /* Value is a \R type */
       PSO_LIMH,    /* Read integer value for heap limit */
       PSO_LIMM,    /* Read integer value for match limit */
       PSO_LIMD,    /* Read integer value for depth limit */
       PSO_OPTMZ    /* Value is an optimization bit */
     };

typedef struct pso {
  const char *name;
  uint16_t length;
  uint16_t type;
  uint32_t value;
} pso;

/* NB: STRING_UTFn_RIGHTPAR contains the length as well */

static const pso pso_list[] = {
  { STRING_UTFn_RIGHTPAR,                  PSO_OPT, PCRE2_UTF },
  { STRING_UTF_RIGHTPAR,                4, PSO_OPT, PCRE2_UTF },
  { STRING_UCP_RIGHTPAR,                4, PSO_OPT, PCRE2_UCP },
  { STRING_NOTEMPTY_RIGHTPAR,           9, PSO_FLG, PCRE2_NOTEMPTY_SET },
  { STRING_NOTEMPTY_ATSTART_RIGHTPAR,  17, PSO_FLG, PCRE2_NE_ATST_SET },
  { STRING_NO_AUTO_POSSESS_RIGHTPAR,   16, PSO_OPTMZ, PCRE2_OPTIM_AUTO_POSSESS },
  { STRING_NO_DOTSTAR_ANCHOR_RIGHTPAR, 18, PSO_OPTMZ, PCRE2_OPTIM_DOTSTAR_ANCHOR },
  { STRING_NO_JIT_RIGHTPAR,             7, PSO_FLG, PCRE2_NOJIT },
  { STRING_NO_START_OPT_RIGHTPAR,      13, PSO_OPTMZ, PCRE2_OPTIM_START_OPTIMIZE },
  { STRING_CASELESS_RESTRICT_RIGHTPAR, 18, PSO_XOPT, PCRE2_EXTRA_CASELESS_RESTRICT },
  { STRING_TURKISH_CASING_RIGHTPAR,    15, PSO_XOPT, PCRE2_EXTRA_TURKISH_CASING },
  { STRING_LIMIT_HEAP_EQ,              11, PSO_LIMH, 0 },
  { STRING_LIMIT_MATCH_EQ,             12, PSO_LIMM, 0 },
  { STRING_LIMIT_DEPTH_EQ,             12, PSO_LIMD, 0 },
  { STRING_LIMIT_RECURSION_EQ,         16, PSO_LIMD, 0 },
  { STRING_CR_RIGHTPAR,                 3, PSO_NL,  PCRE2_NEWLINE_CR },
  { STRING_LF_RIGHTPAR,                 3, PSO_NL,  PCRE2_NEWLINE_LF },
  { STRING_CRLF_RIGHTPAR,               5, PSO_NL,  PCRE2_NEWLINE_CRLF },
  { STRING_ANY_RIGHTPAR,                4, PSO_NL,  PCRE2_NEWLINE_ANY },
  { STRING_NUL_RIGHTPAR,                4, PSO_NL,  PCRE2_NEWLINE_NUL },
  { STRING_ANYCRLF_RIGHTPAR,            8, PSO_NL,  PCRE2_NEWLINE_ANYCRLF },
  { STRING_BSR_ANYCRLF_RIGHTPAR,       12, PSO_BSR, PCRE2_BSR_ANYCRLF },
  { STRING_BSR_UNICODE_RIGHTPAR,       12, PSO_BSR, PCRE2_BSR_UNICODE }
};

/* This table is used when converting repeating opcodes into possessified
versions as a result of an explicit possessive quantifier such as ++. A zero
value means there is no possessified version - in those cases the item in
question must be wrapped in ONCE brackets. The table is truncated at OP_CALLOUT
because all relevant opcodes are less than that. */

static const uint8_t opcode_possessify[] = {
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   /* 0 - 15  */
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   /* 16 - 31 */

  0,                       /* NOTI */
  OP_POSSTAR, 0,           /* STAR, MINSTAR */
  OP_POSPLUS, 0,           /* PLUS, MINPLUS */
  OP_POSQUERY, 0,          /* QUERY, MINQUERY */
  OP_POSUPTO, 0,           /* UPTO, MINUPTO */
  0,                       /* EXACT */
  0, 0, 0, 0,              /* POS{STAR,PLUS,QUERY,UPTO} */

  OP_POSSTARI, 0,          /* STARI, MINSTARI */
  OP_POSPLUSI, 0,          /* PLUSI, MINPLUSI */
  OP_POSQUERYI, 0,         /* QUERYI, MINQUERYI */
  OP_POSUPTOI, 0,          /* UPTOI, MINUPTOI */
  0,                       /* EXACTI */
  0, 0, 0, 0,              /* POS{STARI,PLUSI,QUERYI,UPTOI} */

  OP_NOTPOSSTAR, 0,        /* NOTSTAR, NOTMINSTAR */
  OP_NOTPOSPLUS, 0,        /* NOTPLUS, NOTMINPLUS */
  OP_NOTPOSQUERY, 0,       /* NOTQUERY, NOTMINQUERY */
  OP_NOTPOSUPTO, 0,        /* NOTUPTO, NOTMINUPTO */
  0,                       /* NOTEXACT */
  0, 0, 0, 0,              /* NOTPOS{STAR,PLUS,QUERY,UPTO} */

  OP_NOTPOSSTARI, 0,       /* NOTSTARI, NOTMINSTARI */
  OP_NOTPOSPLUSI, 0,       /* NOTPLUSI, NOTMINPLUSI */
  OP_NOTPOSQUERYI, 0,      /* NOTQUERYI, NOTMINQUERYI */
  OP_NOTPOSUPTOI, 0,       /* NOTUPTOI, NOTMINUPTOI */
  0,                       /* NOTEXACTI */
  0, 0, 0, 0,              /* NOTPOS{STARI,PLUSI,QUERYI,UPTOI} */

  OP_TYPEPOSSTAR, 0,       /* TYPESTAR, TYPEMINSTAR */
  OP_TYPEPOSPLUS, 0,       /* TYPEPLUS, TYPEMINPLUS */
  OP_TYPEPOSQUERY, 0,      /* TYPEQUERY, TYPEMINQUERY */
  OP_TYPEPOSUPTO, 0,       /* TYPEUPTO, TYPEMINUPTO */
  0,                       /* TYPEEXACT */
  0, 0, 0, 0,              /* TYPEPOS{STAR,PLUS,QUERY,UPTO} */

  OP_CRPOSSTAR, 0,         /* CRSTAR, CRMINSTAR */
  OP_CRPOSPLUS, 0,         /* CRPLUS, CRMINPLUS */
  OP_CRPOSQUERY, 0,        /* CRQUERY, CRMINQUERY */
  OP_CRPOSRANGE, 0,        /* CRRANGE, CRMINRANGE */
  0, 0, 0, 0,              /* CRPOS{STAR,PLUS,QUERY,RANGE} */

  0, 0, 0, 0,              /* CLASS, NCLASS, XCLASS, ECLASS */
  0, 0,                    /* REF, REFI */
  0, 0,                    /* DNREF, DNREFI */
  0, 0,                    /* RECURSE, CALLOUT */
};

/* Compile-time check that the table has the correct size. */
STATIC_ASSERT(sizeof(opcode_possessify) == OP_CALLOUT+1, opcode_possessify);


#ifdef DEBUG_SHOW_PARSED
/*************************************************
*     Show the parsed pattern for debugging      *
*************************************************/

/* For debugging the pre-scan, this code, which outputs the parsed data vector,
can be enabled. */

static void show_parsed(compile_block *cb)
{
uint32_t *pptr = cb->parsed_pattern;

for (;;)
  {
  int max, min;
  PCRE2_SIZE offset;
  uint32_t i;
  uint32_t length;
  uint32_t meta_arg = META_DATA(*pptr);

  fprintf(stderr, "+++ %02d %.8x ", (int)(pptr - cb->parsed_pattern), *pptr);

  if (*pptr < META_END)
    {
    if (*pptr > 32 && *pptr < 128) fprintf(stderr, "%c", *pptr);
    pptr++;
    }

  else switch (META_CODE(*pptr++))
    {
    default:
    fprintf(stderr, "**** OOPS - unknown META value - giving up ****\n");
    return;

    case META_END:
    fprintf(stderr, "META_END\n");
    return;

    case META_CAPTURE:
    fprintf(stderr, "META_CAPTURE %d", meta_arg);
    break;

    case META_RECURSE:
    GETOFFSET(offset, pptr);
    fprintf(stderr, "META_RECURSE %d %zd", meta_arg, offset);
    break;

    case META_BACKREF:
    if (meta_arg < 10)
      offset = cb->small_ref_offset[meta_arg];
    else
      GETOFFSET(offset, pptr);
    fprintf(stderr, "META_BACKREF %d %zd", meta_arg, offset);
    break;

    case META_ESCAPE:
    if (meta_arg == ESC_P || meta_arg == ESC_p)
      {
      uint32_t ptype = *pptr >> 16;
      uint32_t pvalue = *pptr++ & 0xffff;
      fprintf(stderr, "META \\%c %d %d", (meta_arg == ESC_P)? CHAR_P:CHAR_p,
        ptype, pvalue);
      }
    else
      {
      uint32_t cc;
      /* There's just one escape we might have here that isn't negated in the
      escapes table. */
      if (meta_arg == ESC_g) cc = CHAR_g;
      else for (cc = ESCAPES_FIRST; cc <= ESCAPES_LAST; cc++)
        {
        if (meta_arg == (uint32_t)(-escapes[cc - ESCAPES_FIRST])) break;
        }
      if (cc > ESCAPES_LAST) cc = CHAR_QUESTION_MARK;
      fprintf(stderr, "META \\%c", cc);
      }
    break;

    case META_MINMAX:
    min = *pptr++;
    max = *pptr++;
    if (max != REPEAT_UNLIMITED)
      fprintf(stderr, "META {%d,%d}", min, max);
    else
      fprintf(stderr, "META {%d,}", min);
    break;

    case META_MINMAX_QUERY:
    min = *pptr++;
    max = *pptr++;
    if (max != REPEAT_UNLIMITED)
      fprintf(stderr, "META {%d,%d}?", min, max);
    else
      fprintf(stderr, "META {%d,}?", min);
    break;

    case META_MINMAX_PLUS:
    min = *pptr++;
    max = *pptr++;
    if (max != REPEAT_UNLIMITED)
      fprintf(stderr, "META {%d,%d}+", min, max);
    else
      fprintf(stderr, "META {%d,}+", min);
    break;

    case META_BIGVALUE: fprintf(stderr, "META_BIGVALUE %.8x", *pptr++); break;
    case META_CIRCUMFLEX: fprintf(stderr, "META_CIRCUMFLEX"); break;
    case META_COND_ASSERT: fprintf(stderr, "META_COND_ASSERT"); break;
    case META_DOLLAR: fprintf(stderr, "META_DOLLAR"); break;
    case META_DOT: fprintf(stderr, "META_DOT"); break;
    case META_ASTERISK: fprintf(stderr, "META *"); break;
    case META_ASTERISK_QUERY: fprintf(stderr, "META *?"); break;
    case META_ASTERISK_PLUS: fprintf(stderr, "META *+"); break;
    case META_PLUS: fprintf(stderr, "META +"); break;
    case META_PLUS_QUERY: fprintf(stderr, "META +?"); break;
    case META_PLUS_PLUS: fprintf(stderr, "META ++"); break;
    case META_QUERY: fprintf(stderr, "META ?"); break;
    case META_QUERY_QUERY: fprintf(stderr, "META ??"); break;
    case META_QUERY_PLUS: fprintf(stderr, "META ?+"); break;

    case META_ATOMIC: fprintf(stderr, "META (?>"); break;
    case META_NOCAPTURE: fprintf(stderr, "META (?:"); break;
    case META_LOOKAHEAD: fprintf(stderr, "META (?="); break;
    case META_LOOKAHEADNOT: fprintf(stderr, "META (?!"); break;
    case META_LOOKAHEAD_NA: fprintf(stderr, "META (*napla:"); break;
    case META_SCRIPT_RUN: fprintf(stderr, "META (*sr:"); break;
    case META_KET: fprintf(stderr, "META )"); break;
    case META_ALT: fprintf(stderr, "META | %d", meta_arg); break;

    case META_CLASS: fprintf(stderr, "META ["); break;
    case META_CLASS_NOT: fprintf(stderr, "META [^"); break;
    case META_CLASS_END: fprintf(stderr, "META ]"); break;
    case META_CLASS_EMPTY: fprintf(stderr, "META []"); break;
    case META_CLASS_EMPTY_NOT: fprintf(stderr, "META [^]"); break;

    case META_RANGE_LITERAL: fprintf(stderr, "META - (literal)"); break;
    case META_RANGE_ESCAPED: fprintf(stderr, "META - (escaped)"); break;

    case META_POSIX: fprintf(stderr, "META_POSIX %d", *pptr++); break;
    case META_POSIX_NEG: fprintf(stderr, "META_POSIX_NEG %d", *pptr++); break;

    case META_ACCEPT: fprintf(stderr, "META (*ACCEPT)"); break;
    case META_FAIL: fprintf(stderr, "META (*FAIL)"); break;
    case META_COMMIT: fprintf(stderr, "META (*COMMIT)"); break;
    case META_PRUNE: fprintf(stderr, "META (*PRUNE)"); break;
    case META_SKIP: fprintf(stderr, "META (*SKIP)"); break;
    case META_THEN: fprintf(stderr, "META (*THEN)"); break;

    case META_OPTIONS:
    fprintf(stderr, "META_OPTIONS 0x%08x 0x%08x", pptr[0], pptr[1]);
    pptr += 2;
    break;

    case META_LOOKBEHIND:
    fprintf(stderr, "META (?<= %d %d", meta_arg, *pptr);
    pptr += 2;
    break;

    case META_LOOKBEHIND_NA:
    fprintf(stderr, "META (*naplb: %d %d", meta_arg, *pptr);
    pptr += 2;
    break;

    case META_LOOKBEHINDNOT:
    fprintf(stderr, "META (?<! %d %d", meta_arg, *pptr);
    pptr += 2;
    break;

    case META_CALLOUT_NUMBER:
    fprintf(stderr, "META (?C%d) next=%d/%d", pptr[2], pptr[0],
       pptr[1]);
    pptr += 3;
    break;

    case META_CALLOUT_STRING:
      {
      uint32_t patoffset = *pptr++;    /* Offset of next pattern item */
      uint32_t patlength = *pptr++;    /* Length of next pattern item */
      fprintf(stderr, "META (?Cstring) length=%d offset=", *pptr++);
      GETOFFSET(offset, pptr);
      fprintf(stderr, "%zd next=%d/%d", offset, patoffset, patlength);
      }
    break;

    case META_RECURSE_BYNAME:
    fprintf(stderr, "META (?(&name) length=%d offset=", *pptr++);
    GETOFFSET(offset, pptr);
    fprintf(stderr, "%zd", offset);
    break;

    case META_BACKREF_BYNAME:
    fprintf(stderr, "META_BACKREF_BYNAME length=%d offset=", *pptr++);
    GETOFFSET(offset, pptr);
    fprintf(stderr, "%zd", offset);
    break;

    case META_COND_NUMBER:
    fprintf(stderr, "META_COND_NUMBER %d offset=", pptr[SIZEOFFSET]);
    GETOFFSET(offset, pptr);
    fprintf(stderr, "%zd", offset);
    pptr++;
    break;

    case META_COND_DEFINE:
    fprintf(stderr, "META (?(DEFINE) offset=");
    GETOFFSET(offset, pptr);
    fprintf(stderr, "%zd", offset);
    break;

    case META_COND_VERSION:
    fprintf(stderr, "META (?(VERSION%s", (*pptr++ == 0)? "=" : ">=");
    fprintf(stderr, "%d.", *pptr++);
    fprintf(stderr, "%d)", *pptr++);
    break;

    case META_COND_NAME:
    fprintf(stderr, "META (?(<name>) length=%d offset=", *pptr++);
    GETOFFSET(offset, pptr);
    fprintf(stderr, "%zd", offset);
    break;

    case META_COND_RNAME:
    fprintf(stderr, "META (?(R&name) length=%d offset=", *pptr++);
    GETOFFSET(offset, pptr);
    fprintf(stderr, "%zd", offset);
    break;

    /* This is kept as a name, because it might be. */

    case META_COND_RNUMBER:
    fprintf(stderr, "META (?(Rnumber) length=%d offset=", *pptr++);
    GETOFFSET(offset, pptr);
    fprintf(stderr, "%zd", offset);
    break;

    case META_OFFSET:
    fprintf(stderr, "META_OFFSET offset=");
    GETOFFSET(offset, pptr);
    fprintf(stderr, "%zd", offset);
    break;

    case META_SCS:
    fprintf(stderr, "META (*scan_substring:");
    break;

    case META_SCS_NAME:
    fprintf(stderr, "META_SCS_NAME length=%d relative_offset=%d", *pptr++, (int)meta_arg);
    break;

    case META_SCS_NUMBER:
    fprintf(stderr, "META_SCS_NUMBER %d relative_offset=%d", *pptr++, (int)meta_arg);
    break;

    case META_MARK:
    fprintf(stderr, "META (*MARK:");
    goto SHOWARG;

    case META_COMMIT_ARG:
    fprintf(stderr, "META (*COMMIT:");
    goto SHOWARG;

    case META_PRUNE_ARG:
    fprintf(stderr, "META (*PRUNE:");
    goto SHOWARG;

    case META_SKIP_ARG:
    fprintf(stderr, "META (*SKIP:");
    goto SHOWARG;

    case META_THEN_ARG:
    fprintf(stderr, "META (*THEN:");
    SHOWARG:
    length = *pptr++;
    for (i = 0; i < length; i++)
      {
      uint32_t cc = *pptr++;
      if (cc > 32 && cc < 128) fprintf(stderr, "%c", cc);
        else fprintf(stderr, "\\x{%x}", cc);
      }
    fprintf(stderr, ") length=%u", length);
    break;

    case META_ECLASS_AND: fprintf(stderr, "META_ECLASS_AND"); break;
    case META_ECLASS_OR: fprintf(stderr, "META_ECLASS_OR"); break;
    case META_ECLASS_SUB: fprintf(stderr, "META_ECLASS_SUB"); break;
    case META_ECLASS_XOR: fprintf(stderr, "META_ECLASS_XOR"); break;
    case META_ECLASS_NOT: fprintf(stderr, "META_ECLASS_NOT"); break;
    }
  fprintf(stderr, "\n");
  }
return;
}
#endif  /* DEBUG_SHOW_PARSED */



/*************************************************
*               Copy compiled code               *
*************************************************/

/* Compiled JIT code cannot be copied, so the new compiled block has no
associated JIT data. */

PCRE2_EXP_DEFN pcre2_code * PCRE2_CALL_CONVENTION
pcre2_code_copy(const pcre2_code *code)
{
PCRE2_SIZE *ref_count;
pcre2_code *newcode;

if (code == NULL) return NULL;
newcode = code->memctl.malloc(code->blocksize, code->memctl.memory_data);
if (newcode == NULL) return NULL;
memcpy(newcode, code, code->blocksize);
newcode->executable_jit = NULL;

/* If the code is one that has been deserialized, increment the reference count
in the decoded tables. */

if ((code->flags & PCRE2_DEREF_TABLES) != 0)
  {
  ref_count = (PCRE2_SIZE *)(code->tables + TABLES_LENGTH);
  (*ref_count)++;
  }

return newcode;
}



/*************************************************
*     Copy compiled code and character tables    *
*************************************************/

/* Compiled JIT code cannot be copied, so the new compiled block has no
associated JIT data. This version of code_copy also makes a separate copy of
the character tables. */

PCRE2_EXP_DEFN pcre2_code * PCRE2_CALL_CONVENTION
pcre2_code_copy_with_tables(const pcre2_code *code)
{
PCRE2_SIZE* ref_count;
pcre2_code *newcode;
uint8_t *newtables;

if (code == NULL) return NULL;
newcode = code->memctl.malloc(code->blocksize, code->memctl.memory_data);
if (newcode == NULL) return NULL;
memcpy(newcode, code, code->blocksize);
newcode->executable_jit = NULL;

newtables = code->memctl.malloc(TABLES_LENGTH + sizeof(PCRE2_SIZE),
  code->memctl.memory_data);
if (newtables == NULL)
  {
  code->memctl.free((void *)newcode, code->memctl.memory_data);
  return NULL;
  }
memcpy(newtables, code->tables, TABLES_LENGTH);
ref_count = (PCRE2_SIZE *)(newtables + TABLES_LENGTH);
*ref_count = 1;

newcode->tables = newtables;
newcode->flags |= PCRE2_DEREF_TABLES;
return newcode;
}



/*************************************************
*               Free compiled code               *
*************************************************/

PCRE2_EXP_DEFN void PCRE2_CALL_CONVENTION
pcre2_code_free(pcre2_code *code)
{
PCRE2_SIZE* ref_count;

if (code != NULL)
  {
#ifdef SUPPORT_JIT
  if (code->executable_jit != NULL)
    PRIV(jit_free)(code->executable_jit, &code->memctl);
#endif

  if ((code->flags & PCRE2_DEREF_TABLES) != 0)
    {
    /* Decoded tables belong to the codes after deserialization, and they must
    be freed when there are no more references to them. The *ref_count should
    always be > 0. */

    ref_count = (PCRE2_SIZE *)(code->tables + TABLES_LENGTH);
    if (*ref_count > 0)
      {
      (*ref_count)--;
      if (*ref_count == 0)
        code->memctl.free((void *)code->tables, code->memctl.memory_data);
      }
    }

  code->memctl.free(code, code->memctl.memory_data);
  }
}



/*************************************************
*         Read a number, possibly signed         *
*************************************************/

/* This function is used to read numbers in the pattern. The initial pointer
must be at the sign or first digit of the number. When relative values
(introduced by + or -) are allowed, they are relative group numbers, and the
result must be greater than zero.

Arguments:
  ptrptr      points to the character pointer variable
  ptrend      points to the end of the input string
  allow_sign  if < 0, sign not allowed; if >= 0, sign is relative to this
  max_value   the largest number allowed;
              you must not pass a value for max_value larger than
              INT_MAX/10 - 1 because this function relies on max_value to
              avoid integer overflow
  max_error   the error to give for an over-large number
  intptr      where to put the result
  errcodeptr  where to put an error code

Returns:      TRUE  - a number was read
              FALSE - errorcode == 0 => no number was found
                      errorcode != 0 => an error occurred
*/

static BOOL
read_number(PCRE2_SPTR *ptrptr, PCRE2_SPTR ptrend, int32_t allow_sign,
  uint32_t max_value, uint32_t max_error, int *intptr, int *errorcodeptr)
{
int sign = 0;
uint32_t n = 0;
PCRE2_SPTR ptr = *ptrptr;
BOOL yield = FALSE;

PCRE2_ASSERT(max_value <= INT_MAX/10 - 1);

*errorcodeptr = 0;

if (allow_sign >= 0 && ptr < ptrend)
  {
  if (*ptr == CHAR_PLUS)
    {
    sign = +1;
    max_value -= allow_sign;
    ptr++;
    }
  else if (*ptr == CHAR_MINUS)
    {
    sign = -1;
    ptr++;
    }
  }

if (ptr >= ptrend || !IS_DIGIT(*ptr)) return FALSE;
while (ptr < ptrend && IS_DIGIT(*ptr))
  {
  n = n * 10 + (*ptr++ - CHAR_0);
  if (n > max_value)
    {
    *errorcodeptr = max_error;
    while (ptr < ptrend && IS_DIGIT(*ptr)) ptr++;
    goto EXIT;
    }
  }

if (allow_sign >= 0 && sign != 0)
  {
  if (n == 0)
    {
    *errorcodeptr = ERR26;  /* +0 and -0 are not allowed */
    goto EXIT;
    }

  if (sign > 0) n += allow_sign;
  else if (n > (uint32_t)allow_sign)
    {
    *errorcodeptr = ERR15;  /* Non-existent subpattern */
    goto EXIT;
    }
  else n = allow_sign + 1 - n;
  }

yield = TRUE;

EXIT:
*intptr = n;
*ptrptr = ptr;
return yield;
}



/*************************************************
*         Read repeat counts                     *
*************************************************/

/* Read an item of the form {n,m} and return the values when non-NULL pointers
are supplied. Repeat counts must be less than 65536 (MAX_REPEAT_COUNT); a
larger value is used for "unlimited". We have to use signed arguments for
read_number() because it is capable of returning a signed value. As of Perl
5.34.0 either n or m may be absent, but not both. Perl also allows spaces and
tabs after { and before } and between the numbers and the comma, so we do too.

Arguments:
  ptrptr         points to pointer to character after '{'
  ptrend         pointer to end of input
  minp           if not NULL, pointer to int for min
  maxp           if not NULL, pointer to int for max
  errorcodeptr   points to error code variable

Returns:         FALSE if not a repeat quantifier, errorcode set zero
                 FALSE on error, with errorcode set non-zero
                 TRUE on success, with pointer updated to point after '}'
*/

static BOOL
read_repeat_counts(PCRE2_SPTR *ptrptr, PCRE2_SPTR ptrend, uint32_t *minp,
  uint32_t *maxp, int *errorcodeptr)
{
PCRE2_SPTR p = *ptrptr;
PCRE2_SPTR pp;
BOOL yield = FALSE;
BOOL had_minimum = FALSE;
int32_t min = 0;
int32_t max = REPEAT_UNLIMITED; /* This value is larger than MAX_REPEAT_COUNT */

*errorcodeptr = 0;
while (p < ptrend && (*p == CHAR_SPACE || *p == CHAR_HT)) p++;

/* Check the syntax before interpreting. Otherwise, a non-quantifier sequence
such as "X{123456ABC" would incorrectly give a "number too big in quantifier"
error. */

pp = p;
if (pp < ptrend && IS_DIGIT(*pp))
  {
  had_minimum = TRUE;
  while (++pp < ptrend && IS_DIGIT(*pp)) {}
  }

while (pp < ptrend && (*pp == CHAR_SPACE || *pp == CHAR_HT)) pp++;
if (pp >= ptrend) return FALSE;

if (*pp == CHAR_RIGHT_CURLY_BRACKET)
  {
  if (!had_minimum) return FALSE;
  }
else
  {
  if (*pp++ != CHAR_COMMA) return FALSE;
  while (pp < ptrend && (*pp == CHAR_SPACE || *pp == CHAR_HT)) pp++;
  if (pp >= ptrend) return FALSE;
  if (IS_DIGIT(*pp))
    {
    while (++pp < ptrend && IS_DIGIT(*pp)) {}
    }
  else if (!had_minimum) return FALSE;
  while (pp < ptrend && (*pp == CHAR_SPACE || *pp == CHAR_HT)) pp++;
  if (pp >= ptrend || *pp != CHAR_RIGHT_CURLY_BRACKET) return FALSE;
  }

/* Now process the quantifier for real. We know it must be {n} or {n,} or {,m}
or {n,m}. The only error that read_number() can return is for a number that is
too big. If *errorcodeptr is returned as zero it means no number was found. */

/* Deal with {,m} or n too big. If we successfully read m there is no need to
check m >= n because n defaults to zero. */

if (!read_number(&p, ptrend, -1, MAX_REPEAT_COUNT, ERR5, &min, errorcodeptr))
  {
  if (*errorcodeptr != 0) goto EXIT;    /* n too big */
  p++;  /* Skip comma and subsequent spaces */
  while (p < ptrend && (*p == CHAR_SPACE || *p == CHAR_HT)) p++;
  if (!read_number(&p, ptrend, -1, MAX_REPEAT_COUNT, ERR5, &max, errorcodeptr))
    {
    if (*errorcodeptr != 0) goto EXIT;  /* m too big */
    }
  }

/* Have read one number. Deal with {n} or {n,} or {n,m} */

else
  {
  while (p < ptrend && (*p == CHAR_SPACE || *p == CHAR_HT)) p++;
  if (*p == CHAR_RIGHT_CURLY_BRACKET)
    {
    max = min;
    }
  else   /* Handle {n,} or {n,m} */
    {
    p++;    /* Skip comma and subsequent spaces */
    while (p < ptrend && (*p == CHAR_SPACE || *p == CHAR_HT)) p++;
    if (!read_number(&p, ptrend, -1, MAX_REPEAT_COUNT, ERR5, &max, errorcodeptr))
      {
      if (*errorcodeptr != 0) goto EXIT;   /* m too big */
      }

    if (max < min)
      {
      *errorcodeptr = ERR4;
      goto EXIT;
      }
    }
  }

/* Valid quantifier exists */

while (p < ptrend && (*p == CHAR_SPACE || *p == CHAR_HT)) p++;
p++;
yield = TRUE;
if (minp != NULL) *minp = (uint32_t)min;
if (maxp != NULL) *maxp = (uint32_t)max;

/* Update the pattern pointer */

EXIT:
*ptrptr = p;
return yield;
}



/*************************************************
*            Handle escapes                      *
*************************************************/

/* This function is called when a \ has been encountered. It either returns a
positive value for a simple escape such as \d, or 0 for a data character, which
is placed in chptr. A backreference to group n is returned as -(n+1). On
entry, ptr is pointing at the character after \. On exit, it points after the
final code unit of the escape sequence.

This function is also called from pcre2_substitute() to handle escape sequences
in replacement strings. In this case, the cb argument is NULL, and in the case
of escapes that have further processing, only sequences that define a data
character are recognised. The options argument is the final value of the
compiled pattern's options.

Arguments:
  ptrptr         points to the input position pointer
  ptrend         points to the end of the input
  chptr          points to a returned data character
  errorcodeptr   points to the errorcode variable (containing zero)
  options        the current options bits
  xoptions       the current extra options bits
  bracount       the number of capturing parentheses encountered so far
  isclass        TRUE if in a character class
  cb             compile data block or NULL when called from pcre2_substitute()

Returns:         zero => a data character
                 positive => a special escape sequence
                 negative => a numerical back reference
                 on error, errorcodeptr is set non-zero
*/

int
PRIV(check_escape)(PCRE2_SPTR *ptrptr, PCRE2_SPTR ptrend, uint32_t *chptr,
  int *errorcodeptr, uint32_t options, uint32_t xoptions, uint32_t bracount,
  BOOL isclass, compile_block *cb)
{
BOOL utf = (options & PCRE2_UTF) != 0;
BOOL alt_bsux =
  ((options & PCRE2_ALT_BSUX) | (xoptions & PCRE2_EXTRA_ALT_BSUX)) != 0;
PCRE2_SPTR ptr = *ptrptr;
uint32_t c, cc;
int escape = 0;
int i;

/* If backslash is at the end of the string, it's an error. */

if (ptr >= ptrend)
  {
  *errorcodeptr = ERR1;
  return 0;
  }

GETCHARINCTEST(c, ptr);         /* Get character value, increment pointer */
*errorcodeptr = 0;              /* Be optimistic */

/* Non-alphanumerics are literals, so we just leave the value in c. An initial
value test saves a memory lookup for code points outside the alphanumeric
range. */

if (c < ESCAPES_FIRST || c > ESCAPES_LAST) {}  /* Definitely literal */

/* Otherwise, do a table lookup. Non-zero values need little processing here. A
positive value is a literal value for something like \n. A negative value is
the negation of one of the ESC_ macros that is passed back for handling by the
calling function. Some extra checking is needed for \N because only \N{U+dddd}
is supported. If the value is zero, further processing is handled below. */

else if ((i = escapes[c - ESCAPES_FIRST]) != 0)
  {
  if (i > 0)
    {
    c = (uint32_t)i;
    if (c == CHAR_CR && (xoptions & PCRE2_EXTRA_ESCAPED_CR_IS_LF) != 0)
      c = CHAR_LF;
    }
  else  /* Negative table entry */
    {
    escape = -i;                    /* Else return a special escape */
    if (cb != NULL && (escape == ESC_P || escape == ESC_p || escape == ESC_X))
      cb->external_flags |= PCRE2_HASBKPORX;   /* Note \P, \p, or \X */

    /* Perl supports \N{name} for character names and \N{U+dddd} for numerical
    Unicode code points, as well as plain \N for "not newline". PCRE does not
    support \N{name}. However, it does support quantification such as \N{2,3},
    so if \N{ is not followed by U+dddd we check for a quantifier. */

    if (escape == ESC_N && ptr < ptrend && *ptr == CHAR_LEFT_CURLY_BRACKET)
      {
      PCRE2_SPTR p = ptr + 1;

      /* Perl ignores spaces and tabs after { */

      while (p < ptrend && (*p == CHAR_SPACE || *p == CHAR_HT)) p++;

      /* \N{U+ can be handled by the \x{ code. However, this construction is
      not valid in EBCDIC environments because it specifies a Unicode
      character, not a codepoint in the local code. For example \N{U+0041}
      must be "A" in all environments. Also, in Perl, \N{U+ forces Unicode
      casing semantics for the entire pattern, so allow it only in UTF (i.e.
      Unicode) mode. */

      if (ptrend - p > 1 && *p == CHAR_U && p[1] == CHAR_PLUS)
        {
#ifndef EBCDIC
        if (utf)
          {
          ptr = p + 2;
          escape = 0;   /* Not a fancy escape after all */
          goto COME_FROM_NU;
          }
#endif
        *errorcodeptr = ERR93;
        }

      /* Give an error in contexts where quantifiers are not allowed
      (character classes; substitution strings). */

      else if (isclass || cb == NULL)
        {
        *errorcodeptr = ERR37;
        }

      /* Give an error if what follows is not a quantifier, but don't override
      an error set by the quantifier reader (e.g. number overflow). */

      else
        {
        if (!read_repeat_counts(&p, ptrend, NULL, NULL, errorcodeptr) &&
             *errorcodeptr == 0)
          *errorcodeptr = ERR37;
        }
      }
    }
  }

/* Escapes that need further processing, including those that are unknown, have
a zero entry in the lookup table. When called from pcre2_substitute(), only \c,
\o, and \x are recognized (\u and \U can never appear as they are used for case
forcing). */

else
  {
  int s;
  PCRE2_SPTR oldptr;
  BOOL overflow;

  /* Filter calls from pcre2_substitute(). */

  if (cb == NULL)
    {
    if (c < CHAR_0 ||
       (c > CHAR_9 && (c != CHAR_c && c != CHAR_o && c != CHAR_x && c != CHAR_g)))
      {
      *errorcodeptr = ERR3;
      return 0;
      }
    alt_bsux = FALSE;   /* Do not modify \x handling */
    }

  switch (c)
    {
    /* A number of Perl escapes are not handled by PCRE. We give an explicit
    error. */

    case CHAR_F:
    case CHAR_l:
    case CHAR_L:
    *errorcodeptr = ERR37;
    break;

    /* \u is unrecognized when neither PCRE2_ALT_BSUX nor PCRE2_EXTRA_ALT_BSUX
    is set. Otherwise, \u must be followed by exactly four hex digits or, if
    PCRE2_EXTRA_ALT_BSUX is set, by any number of hex digits in braces.
    Otherwise it is a lowercase u letter. This gives some compatibility with
    ECMAScript (aka JavaScript). Unlike other braced items, white space is NOT
    allowed. When \u{ is not followed by hex digits, a special return is given
    because otherwise \u{ 12} (for example) would be treated as u{12}. */

    case CHAR_u:
    if (!alt_bsux) *errorcodeptr = ERR37; else
      {
      uint32_t xc;

      if (ptr >= ptrend) break;
      if (*ptr == CHAR_LEFT_CURLY_BRACKET &&
          (xoptions & PCRE2_EXTRA_ALT_BSUX) != 0)
        {
        PCRE2_SPTR hptr = ptr + 1;

        cc = 0;
        while (hptr < ptrend && (xc = XDIGIT(*hptr)) != 0xff)
          {
          if ((cc & 0xf0000000) != 0)  /* Test for 32-bit overflow */
            {
            *errorcodeptr = ERR77;
            ptr = hptr;   /* Show where */
            break;        /* *hptr != } will cause another break below */
            }
          cc = (cc << 4) | xc;
          hptr++;
          }

        if (hptr == ptr + 1 ||   /* No hex digits */
            hptr >= ptrend ||    /* Hit end of input */
            *hptr != CHAR_RIGHT_CURLY_BRACKET)  /* No } terminator */
          {
          if (isclass) break; /* In a class, just treat as '\u' literal */
          escape = ESC_ub;    /* Special return */
          ptr++;              /* Skip { */
          break;              /* Hex escape not recognized */
          }

        c = cc;          /* Accept the code point */
        ptr = hptr + 1;
        }

      else  /* Must be exactly 4 hex digits */
        {
        if (ptrend - ptr < 4) break;               /* Less than 4 chars */
        if ((cc = XDIGIT(ptr[0])) == 0xff) break;  /* Not a hex digit */
        if ((xc = XDIGIT(ptr[1])) == 0xff) break;  /* Not a hex digit */
        cc = (cc << 4) | xc;
        if ((xc = XDIGIT(ptr[2])) == 0xff) break;  /* Not a hex digit */
        cc = (cc << 4) | xc;
        if ((xc = XDIGIT(ptr[3])) == 0xff) break;  /* Not a hex digit */
        c = (cc << 4) | xc;
        ptr += 4;
        }

      if (utf)
        {
        if (c > 0x10ffffU) *errorcodeptr = ERR77;
        else
          if (c >= 0xd800 && c <= 0xdfff &&
              (xoptions & PCRE2_EXTRA_ALLOW_SURROGATE_ESCAPES) == 0)
                *errorcodeptr = ERR73;
        }
      else if (c > MAX_NON_UTF_CHAR) *errorcodeptr = ERR77;
      }
    break;

    /* \U is unrecognized unless PCRE2_ALT_BSUX or PCRE2_EXTRA_ALT_BSUX is set,
    in which case it is an upper case letter. */

    case CHAR_U:
    if (!alt_bsux) *errorcodeptr = ERR37;
    break;

    /* In a character class, \g is just a literal "g". Outside a character
    class, \g must be followed by one of a number of specific things:

    (1) A number, either plain or braced. If positive, it is an absolute
    backreference. If negative, it is a relative backreference. This is a Perl
    5.10 feature.

    (2) Perl 5.10 also supports \g{name} as a reference to a named group. This
    is part of Perl's movement towards a unified syntax for back references. As
    this is synonymous with \k{name}, we fudge it up by pretending it really
    was \k{name}.

    (3) For Oniguruma compatibility we also support \g followed by a name or a
    number either in angle brackets or in single quotes. However, these are
    (possibly recursive) subroutine calls, _not_ backreferences. We return
    the ESC_g code.

    Summary: Return a negative number for a numerical back reference (offset
    by 1), ESC_k for a named back reference, and ESC_g for a named or
    numbered subroutine call.

    The above describes the \g behaviour inside patterns. Inside replacement
    strings (pcre2_substitute) we support only \g<nameornum> for Python
    compatibility. Return ESG_g for the named case, and -(num+1) for the
    numbered case.
    */

    case CHAR_g:
    if (isclass) break;

    if (ptr >= ptrend)
      {
      *errorcodeptr = ERR57;
      break;
      }

    if (cb == NULL)
      {
      PCRE2_SPTR p;
      /* Substitution strings */
      if (*ptr != CHAR_LESS_THAN_SIGN)
        {
        *errorcodeptr = ERR57;
        break;
        }

      p = ptr + 1;

      if (!read_number(&p, ptrend, -1, MAX_GROUP_NUMBER, ERR61, &s,
          errorcodeptr))
        {
        if (*errorcodeptr == 0) escape = ESC_g;  /* No number found */
        break;
        }

      if (p >= ptrend || *p != CHAR_GREATER_THAN_SIGN)
        {
        /* not advancing ptr; report error at the \g character */
        *errorcodeptr = ERR57;
        break;
        }

      /* This is the reason that back references are returned as -(s+1) rather
      than just -s. In a pattern, \0 is not a back reference, but \g<0> is
      valid in a substitution string, so this must be representable. */
      ptr = p + 1;
      escape = -(s+1);
      break;
      }

    if (*ptr == CHAR_LESS_THAN_SIGN || *ptr == CHAR_APOSTROPHE)
      {
      escape = ESC_g;
      break;
      }

    /* If there is a brace delimiter, try to read a numerical reference. If
    there isn't one, assume we have a name and treat it as \k. */

    if (*ptr == CHAR_LEFT_CURLY_BRACKET)
      {
      PCRE2_SPTR p = ptr + 1;

      while (p < ptrend && (*p == CHAR_SPACE || *p == CHAR_HT)) p++;
      if (!read_number(&p, ptrend, bracount, MAX_GROUP_NUMBER, ERR61, &s,
          errorcodeptr))
        {
        if (*errorcodeptr == 0) escape = ESC_k;  /* No number found */
        break;
        }
      while (p < ptrend && (*p == CHAR_SPACE || *p == CHAR_HT)) p++;

      if (p >= ptrend || *p != CHAR_RIGHT_CURLY_BRACKET)
        {
        /* not advancing ptr; report error at the \g character */
        *errorcodeptr = ERR57;
        break;
        }
      ptr = p + 1;
      }

    /* Read an undelimited number */

    else
      {
      if (!read_number(&ptr, ptrend, bracount, MAX_GROUP_NUMBER, ERR61, &s,
          errorcodeptr))
        {
        if (*errorcodeptr == 0) *errorcodeptr = ERR57;  /* No number found */
        break;
        }
      }

    if (s <= 0)
      {
      *errorcodeptr = ERR15;
      break;
      }

    escape = -(s+1);
    break;

    /* The handling of escape sequences consisting of a string of digits
    starting with one that is not zero is not straightforward. Perl has changed
    over the years. Nowadays \g{} for backreferences and \o{} for octal are
    recommended to avoid the ambiguities in the old syntax.

    Outside a character class, the digits are read as a decimal number. If the
    number is less than 10, or if there are that many previous extracting left
    brackets, it is a back reference. Otherwise, up to three octal digits are
    read to form an escaped character code. Thus \123 is likely to be octal 123
    (cf \0123, which is octal 012 followed by the literal 3). This is the "Perl
    style" of handling ambiguous octal/backrefences such as \12.

    There is an alternative disambiguation strategy, selected by
    PCRE2_EXTRA_PYTHON_OCTAL, which follows Python's behaviour. An octal must
    have either a leading zero, or exactly three octal digits; otherwise it's
    a backreference. The disambiguation is stable, and does not depend on how
    many capture groups are defined (it's simply an invalid backreference if
    there is no corresponding capture group). Additionally, octal values above
    \377 (\xff) are rejected.

    Inside a character class, \ followed by a digit is always either a literal
    8 or 9 or an octal number. */

    case CHAR_1: case CHAR_2: case CHAR_3: case CHAR_4: case CHAR_5:
    case CHAR_6: case CHAR_7: case CHAR_8: case CHAR_9:

    if (isclass)
      {
      /* Fall through to octal handling; never a backreference inside a class. */
      }
    else if ((xoptions & PCRE2_EXTRA_PYTHON_OCTAL) != 0)
      {
      /* Python-style disambiguation. */
      if (ptr[-1] <= CHAR_7 && ptr + 1 < ptrend && ptr[0] >= CHAR_0 &&
          ptr[0] <= CHAR_7 && ptr[1] >= CHAR_0 && ptr[1] <= CHAR_7)
        {
        /* We peeked a three-digit octal, so fall through */
        }
      else
        {
        /* We are at a digit, so the only possible error from read_number() is
        a number that is too large. */
        ptr--;   /* Back to the digit */

        if (!read_number(&ptr, ptrend, -1, MAX_GROUP_NUMBER, 0, &s, errorcodeptr))
          {
          *errorcodeptr = ERR61;
          break;
          }

        escape = -(s+1);
        break;
        }
      }
    else
      {
      /* Perl-style disambiguation. */
      oldptr = ptr;
      ptr--;   /* Back to the digit */

      /* As we know we are at a digit, the only possible error from
      read_number() is a number that is too large to be a group number. Because
      that number might be still valid if read as an octal, errorcodeptr is not
      set on failure and therefore a sentinel value of INT_MAX is used instead
      of the original value, and will be used later to properly set the error,
      if not falling through. */

      if (!read_number(&ptr, ptrend, -1, MAX_GROUP_NUMBER, 0, &s, errorcodeptr))
        s = INT_MAX;

      /* \1 to \9 are always back references. \8x and \9x are too; \1x to \7x
      are octal escapes if there are not that many previous captures. */

      if (s < 10 || c >= CHAR_8 || (unsigned)s <= bracount)
        {
        /* s > MAX_GROUP_NUMBER should not be possible because of read_number(),
        but we keep it just to be safe and because it will also catch the
        sentinel value that was set on failure by that function. */

        if ((unsigned)s > MAX_GROUP_NUMBER)
          {
          PCRE2_ASSERT(s == INT_MAX);
          *errorcodeptr = ERR61;
          }
        else escape = -(s+1);     /* Indicates a back reference */
        break;
        }

      ptr = oldptr;      /* Put the pointer back and fall through */
      }

    /* Handle a digit following \ when the number is not a back reference, or
    we are within a character class. If the first digit is 8 or 9, Perl used to
    generate a binary zero and then treat the digit as a following literal. At
    least by Perl 5.18 this changed so as not to insert the binary zero. */

    if (c >= CHAR_8) break;

    /* Fall through */

    /* \0 always starts an octal number, but we may drop through to here with a
    larger first octal digit. The original code used just to take the least
    significant 8 bits of octal numbers (I think this is what early Perls used
    to do). Nowadays we allow for larger numbers in UTF-8 mode and 16/32-bit mode,
    but no more than 3 octal digits. */

    case CHAR_0:
    c -= CHAR_0;
    while(i++ < 2 && ptr < ptrend && *ptr >= CHAR_0 && *ptr <= CHAR_7)
        c = c * 8 + *ptr++ - CHAR_0;
    if (c > 0xff)
      {
      if ((xoptions & PCRE2_EXTRA_PYTHON_OCTAL) != 0) *errorcodeptr = ERR102;
#if PCRE2_CODE_UNIT_WIDTH == 8
      else if (!utf) *errorcodeptr = ERR51;
#endif
      }

    /* PCRE2_EXTRA_NO_BS0 disables the NUL escape '\0' but doesn't affect
    two- or three-character octal escapes \00 and \000, nor \x00. */

    if ((xoptions & PCRE2_EXTRA_NO_BS0) != 0 && c == 0 && i == 1)
        *errorcodeptr = ERR98;
    break;

    /* \o is a relatively new Perl feature, supporting a more general way of
    specifying character codes in octal. The only supported form is \o{ddd},
    with optional spaces or tabs after { and before }. */

    case CHAR_o:
    if (ptr >= ptrend || *ptr++ != CHAR_LEFT_CURLY_BRACKET)
      {
      ptr--;
      *errorcodeptr = ERR55;
      break;
      }

    while (ptr < ptrend && (*ptr == CHAR_SPACE || *ptr == CHAR_HT)) ptr++;
    if (ptr >= ptrend || *ptr == CHAR_RIGHT_CURLY_BRACKET)
      {
      *errorcodeptr = ERR78;
      break;
      }

    c = 0;
    overflow = FALSE;
    while (ptr < ptrend && *ptr >= CHAR_0 && *ptr <= CHAR_7)
      {
      cc = *ptr++;
      if (c == 0 && cc == CHAR_0) continue;     /* Leading zeroes */
#if PCRE2_CODE_UNIT_WIDTH == 32
      if (c >= 0x20000000u) { overflow = TRUE; break; }
#endif
      c = (c << 3) + (cc - CHAR_0);
#if PCRE2_CODE_UNIT_WIDTH == 8
      if (c > (utf ? 0x10ffffU : 0xffU)) { overflow = TRUE; break; }
#elif PCRE2_CODE_UNIT_WIDTH == 16
      if (c > (utf ? 0x10ffffU : 0xffffU)) { overflow = TRUE; break; }
#elif PCRE2_CODE_UNIT_WIDTH == 32
      if (utf && c > 0x10ffffU) { overflow = TRUE; break; }
#endif
      }

    while (ptr < ptrend && (*ptr == CHAR_SPACE || *ptr == CHAR_HT)) ptr++;

    if (overflow)
      {
      while (ptr < ptrend && *ptr >= CHAR_0 && *ptr <= CHAR_7) ptr++;
      *errorcodeptr = ERR34;
      }
    else if (ptr < ptrend && *ptr++ == CHAR_RIGHT_CURLY_BRACKET)
      {
      if (utf && c >= 0xd800 && c <= 0xdfff &&
          (xoptions & PCRE2_EXTRA_ALLOW_SURROGATE_ESCAPES) == 0)
        {
        ptr--;
        *errorcodeptr = ERR73;
        }
      }
    else
      {
      ptr--;
      *errorcodeptr = ERR64;
      }
    break;

    /* When PCRE2_ALT_BSUX or PCRE2_EXTRA_ALT_BSUX is set, \x must be followed
    by two hexadecimal digits. Otherwise it is a lowercase x letter. */

    case CHAR_x:
    if (alt_bsux)
      {
      uint32_t xc;
      if (ptrend - ptr < 2) break;               /* Less than 2 characters */
      if ((cc = XDIGIT(ptr[0])) == 0xff) break;  /* Not a hex digit */
      if ((xc = XDIGIT(ptr[1])) == 0xff) break;  /* Not a hex digit */
      c = (cc << 4) | xc;
      ptr += 2;
      }

    /* Handle \x in Perl's style. \x{ddd} is a character code which can be
    greater than 0xff in UTF-8 or non-8bit mode, but only if the ddd are hex
    digits. If not, { used to be treated as a data character. However, Perl
    seems to read hex digits up to the first non-such, and ignore the rest, so
    that, for example \x{zz} matches a binary zero. This seems crazy, so PCRE
    now gives an error. */

    else
      {
      if (ptr < ptrend && *ptr == CHAR_LEFT_CURLY_BRACKET)
        {
        ptr++;
        while (ptr < ptrend && (*ptr == CHAR_SPACE || *ptr == CHAR_HT)) ptr++;

#ifndef EBCDIC
        COME_FROM_NU:
#endif
        if (ptr >= ptrend || *ptr == CHAR_RIGHT_CURLY_BRACKET)
          {
          *errorcodeptr = ERR78;
          break;
          }
        c = 0;
        overflow = FALSE;

        while (ptr < ptrend && (cc = XDIGIT(*ptr)) != 0xff)
          {
          ptr++;
          if (c == 0 && cc == 0) continue;   /* Leading zeroes */
#if PCRE2_CODE_UNIT_WIDTH == 32
          if (c >= 0x10000000l) { overflow = TRUE; break; }
#endif
          c = (c << 4) | cc;
          if ((utf && c > 0x10ffffU) || (!utf && c > MAX_NON_UTF_CHAR))
            {
            overflow = TRUE;
            break;
            }
          }

        /* Perl ignores spaces and tabs before } */

        while (ptr < ptrend && (*ptr == CHAR_SPACE || *ptr == CHAR_HT)) ptr++;

        /* On overflow, skip remaining hex digits */

        if (overflow)
          {
          while (ptr < ptrend && XDIGIT(*ptr) != 0xff) ptr++;
          *errorcodeptr = ERR34;
          }
        else if (ptr < ptrend && *ptr++ == CHAR_RIGHT_CURLY_BRACKET)
          {
          if (utf && c >= 0xd800 && c <= 0xdfff &&
              (xoptions & PCRE2_EXTRA_ALLOW_SURROGATE_ESCAPES) == 0)
            {
            ptr--;
            *errorcodeptr = ERR73;
            }
          }

        /* If the sequence of hex digits (followed by optional space) does not
        end with '}', give an error. We used just to recognize this construct
        and fall through to the normal \x handling, but nowadays Perl gives an
        error, which seems much more sensible, so we do too. */

        else
          {
          ptr--;
          *errorcodeptr = ERR67;
          }
        }   /* End of \x{} processing */

      /* Read a up to two hex digits after \x */

      else
        {
        /* Perl has the surprising/broken behaviour that \x without following
        hex digits is treated as an escape for NUL. Their source code laments
        this but keeps it for backwards compatibility. A warning is printed
        when "use warnings" is enabled. Because we don't have warnings, we
        simply forbid it. */
        if (ptr >= ptrend || (cc = XDIGIT(*ptr)) == 0xff)
          {
          /* Not a hex digit */
          *errorcodeptr = ERR78;
          break;
          }
        ptr++;
        c = cc;

        /* With "use re 'strict'" Perl actually requires exactly two digits (error
        for \x, \xA and \xAAA). While \x was already rejected, this seems overly
        strict, and there seems little incentive to align with that, given the
        backwards-compatibility cost.

        For comparison, note that other engines disagree. For example:
          - Java allows 1 or 2 hex digits. Error if 0 digits. No error if >2 digits
          - .NET requires 2 hex digits. Error if 0, 1 digits. No error if >2 digits.
        */
        if (ptr >= ptrend || (cc = XDIGIT(*ptr)) == 0xff) break;  /* Not a hex digit */
        ptr++;
        c = (c << 4) | cc;
        }     /* End of \xdd handling */
      }       /* End of Perl-style \x handling */
    break;

    /* The handling of \c is different in ASCII and EBCDIC environments. In an
    ASCII (or Unicode) environment, an error is given if the character
    following \c is not a printable ASCII character. Otherwise, the following
    character is upper-cased if it is a letter, and after that the 0x40 bit is
    flipped. The result is the value of the escape.

    In an EBCDIC environment the handling of \c is compatible with the
    specification in the perlebcdic document. The following character must be
    a letter or one of small number of special characters. These provide a
    means of defining the character values 0-31.

    For testing the EBCDIC handling of \c in an ASCII environment, recognize
    the EBCDIC value of 'c' explicitly. */

#if defined EBCDIC && 'a' != 0x81
    case 0x83:
#else
    case CHAR_c:
#endif
    if (ptr >= ptrend)
      {
      *errorcodeptr = ERR2;
      break;
      }
    c = *ptr;
    if (c >= CHAR_a && c <= CHAR_z) c = UPPER_CASE(c);

    /* Handle \c in an ASCII/Unicode environment. */

#ifndef EBCDIC    /* ASCII/UTF-8 coding */
    if (c < 32 || c > 126)  /* Excludes all non-printable ASCII */
      {
      *errorcodeptr = ERR68;
      break;
      }
    c ^= 0x40;

    /* Handle \c in an EBCDIC environment. The special case \c? is converted to
    255 (0xff) or 95 (0x5f) if other characters suggest we are using the
    POSIX-BC encoding. (This is the way Perl indicates that it handles \c?.)
    The other valid sequences correspond to a list of specific characters. */

#else
    if (c == CHAR_QUESTION_MARK)
      c = ('\\' == 188 && '`' == 74)? 0x5f : 0xff;
    else
      {
      for (i = 0; i < 32; i++)
        {
        if (c == ebcdic_escape_c[i]) break;
        }
      if (i < 32) c = i; else *errorcodeptr = ERR68;
      }
#endif  /* EBCDIC */

    ptr++;
    break;

    /* Any other alphanumeric following \ is an error. Perl gives an error only
    if in warning mode, but PCRE doesn't have a warning mode. */

    default:
    *errorcodeptr = ERR3;
    *ptrptr = ptr - 1;     /* Point to the character at fault */
    return 0;
    }
  }

/* Set the pointer to the next character before returning. */

*ptrptr = ptr;
*chptr = c;
return escape;
}



#ifdef SUPPORT_UNICODE
/*************************************************
*               Handle \P and \p                 *
*************************************************/

/* This function is called after \P or \p has been encountered, provided that
PCRE2 is compiled with support for UTF and Unicode properties. On entry, the
contents of ptrptr are pointing after the P or p. On exit, it is left pointing
after the final code unit of the escape sequence.

Arguments:
  ptrptr         the pattern position pointer
  negptr         a boolean that is set TRUE for negation else FALSE
  ptypeptr       an unsigned int that is set to the type value
  pdataptr       an unsigned int that is set to the detailed property value
  errorcodeptr   the error code variable
  cb             the compile data

Returns:         TRUE if the type value was found, or FALSE for an invalid type
*/

static BOOL
get_ucp(PCRE2_SPTR *ptrptr, BOOL *negptr, uint16_t *ptypeptr,
  uint16_t *pdataptr, int *errorcodeptr, compile_block *cb)
{
PCRE2_UCHAR c;
PCRE2_SIZE i, bot, top;
PCRE2_SPTR ptr = *ptrptr;
PCRE2_UCHAR name[50];
PCRE2_UCHAR *vptr = NULL;
uint16_t ptscript = PT_NOTSCRIPT;

if (ptr >= cb->end_pattern) goto ERROR_RETURN;
c = *ptr++;
*negptr = FALSE;

/* \P or \p can be followed by a name in {}, optionally preceded by ^ for
negation. We must be handling Unicode encoding here, though we may be compiling
for UTF-8 input in an EBCDIC environment. (PCRE2 does not support both EBCDIC
input and Unicode input in the same build.) In accordance with Unicode's "loose
matching" rules, ASCII white space, hyphens, and underscores are ignored. We
don't use isspace() or tolower() because (a) code points may be greater than
255, and (b) they wouldn't work when compiling for Unicode in an EBCDIC
environment. */

if (c == CHAR_LEFT_CURLY_BRACKET)
  {
  if (ptr >= cb->end_pattern) goto ERROR_RETURN;

  for (i = 0; i < (int)(sizeof(name) / sizeof(PCRE2_UCHAR)) - 1; i++)
    {
    REDO:

    if (ptr >= cb->end_pattern) goto ERROR_RETURN;
    c = *ptr++;

    /* Skip ignorable Unicode characters. */

    while (c == CHAR_UNDERSCORE || c == CHAR_MINUS || c == CHAR_SPACE ||
          (c >= CHAR_HT && c <= CHAR_CR))
      {
      if (ptr >= cb->end_pattern) goto ERROR_RETURN;
      c = *ptr++;
      }

    /* The first significant character being circumflex negates the meaning of
    the item. */

    if (i == 0 && !*negptr && c == CHAR_CIRCUMFLEX_ACCENT)
      {
      *negptr = TRUE;
      goto REDO;
      }

    if (c == CHAR_RIGHT_CURLY_BRACKET) break;

    /* Names consist of ASCII letters and digits, but equals and colon may also
    occur as a name/value separator. We must also allow for \p{L&}. A simple
    check for a value between '&' and 'z' suffices because anything else in a
    name or value will cause an "unknown property" error anyway. */

    if (c < CHAR_AMPERSAND || c > CHAR_z) goto ERROR_RETURN;

    /* Lower case a capital letter or remember where the name/value separator
    is. */

    if (c >= CHAR_A && c <= CHAR_Z) c |= 0x20;
    else if ((c == CHAR_COLON || c == CHAR_EQUALS_SIGN) && vptr == NULL)
      vptr = name + i;

    name[i] = c;
    }

  /* Error if the loop didn't end with '}' - either we hit the end of the
  pattern or the name was longer than any legal property name. */

  if (c != CHAR_RIGHT_CURLY_BRACKET) goto ERROR_RETURN;
  name[i] = 0;
  }

/* If { doesn't follow \p or \P there is just one following character, which
must be an ASCII letter. */

else if (c >= CHAR_A && c <= CHAR_Z)
  {
  name[0] = c | 0x20;  /* Lower case */
  name[1] = 0;
  }
else if (c >= CHAR_a && c <= CHAR_z)
  {
  name[0] = c;
  name[1] = 0;
  }
else goto ERROR_RETURN;

*ptrptr = ptr;   /* Update pattern pointer */

/* If the property contains ':' or '=' we have class name and value separately
specified. The following are supported:

  . Bidi_Class (synonym bc), for which the property names are "bidi<name>".
  . Script (synonym sc) for which the property name is the script name
  . Script_Extensions (synonym scx), ditto

As this is a small number, we currently just check the names directly. If this
grows, a sorted table and a switch will be neater.

For both the script properties, set a PT_xxx value so that (1) they can be
distinguished and (2) invalid script names that happen to be the name of
another property can be diagnosed. */

if (vptr != NULL)
  {
  int offset = 0;
  PCRE2_UCHAR sname[8];

  *vptr = 0;   /* Terminate property name */
  if (PRIV(strcmp_c8)(name, STRING_bidiclass) == 0 ||
      PRIV(strcmp_c8)(name, STRING_bc) == 0)
    {
    offset = 4;
    sname[0] = CHAR_b;
    sname[1] = CHAR_i;  /* There is no strcpy_c8 function */
    sname[2] = CHAR_d;
    sname[3] = CHAR_i;
    }

  else if (PRIV(strcmp_c8)(name, STRING_script) == 0 ||
           PRIV(strcmp_c8)(name, STRING_sc) == 0)
    ptscript = PT_SC;

  else if (PRIV(strcmp_c8)(name, STRING_scriptextensions) == 0 ||
           PRIV(strcmp_c8)(name, STRING_scx) == 0)
    ptscript = PT_SCX;

  else
    {
    *errorcodeptr = ERR47;
    return FALSE;
    }

  /* Adjust the string in name[] as needed */

  memmove(name + offset, vptr + 1, (name + i - vptr)*sizeof(PCRE2_UCHAR));
  if (offset != 0) memmove(name, sname, offset*sizeof(PCRE2_UCHAR));
  }

/* Search for a recognized property using binary chop. */

bot = 0;
top = PRIV(utt_size);

while (bot < top)
  {
  int r;
  i = (bot + top) >> 1;
  r = PRIV(strcmp_c8)(name, PRIV(utt_names) + PRIV(utt)[i].name_offset);

  /* When a matching property is found, some extra checking is needed when the
  \p{xx:yy} syntax is used and xx is either sc or scx. */

  if (r == 0)
    {
    *pdataptr = PRIV(utt)[i].value;
    if (vptr == NULL || ptscript == PT_NOTSCRIPT)
      {
      *ptypeptr = PRIV(utt)[i].type;
      return TRUE;
      }

    switch (PRIV(utt)[i].type)
      {
      case PT_SC:
      *ptypeptr = PT_SC;
      return TRUE;

      case PT_SCX:
      *ptypeptr = ptscript;
      return TRUE;
      }

    break;  /* Non-script found */
    }

  if (r > 0) bot = i + 1; else top = i;
  }

*errorcodeptr = ERR47;   /* Unrecognized property */
return FALSE;

ERROR_RETURN:            /* Malformed \P or \p */
*errorcodeptr = ERR46;
*ptrptr = ptr;
return FALSE;
}
#endif



/*************************************************
*           Check for POSIX class syntax         *
*************************************************/

/* This function is called when the sequence "[:" or "[." or "[=" is
encountered in a character class. It checks whether this is followed by a
sequence of characters terminated by a matching ":]" or ".]" or "=]". If we
reach an unescaped ']' without the special preceding character, return FALSE.

Originally, this function only recognized a sequence of letters between the
terminators, but it seems that Perl recognizes any sequence of characters,
though of course unknown POSIX names are subsequently rejected. Perl gives an
"Unknown POSIX class" error for [:f\oo:] for example, where previously PCRE
didn't consider this to be a POSIX class. Likewise for [:1234:].

The problem in trying to be exactly like Perl is in the handling of escapes. We
have to be sure that [abc[:x\]pqr] is *not* treated as containing a POSIX
class, but [abc[:x\]pqr:]] is (so that an error can be generated). The code
below handles the special cases \\ and \], but does not try to do any other
escape processing. This makes it different from Perl for cases such as
[:l\ower:] where Perl recognizes it as the POSIX class "lower" but PCRE does
not recognize "l\ower". This is a lesser evil than not diagnosing bad classes
when Perl does, I think.

A user pointed out that PCRE was rejecting [:a[:digit:]] whereas Perl was not.
It seems that the appearance of a nested POSIX class supersedes an apparent
external class. For example, [:a[:digit:]b:] matches "a", "b", ":", or
a digit. This is handled by returning FALSE if the start of a new group with
the same terminator is encountered, since the next closing sequence must close
the nested group, not the outer one.

In Perl, unescaped square brackets may also appear as part of class names. For
example, [:a[:abc]b:] gives unknown POSIX class "[:abc]b:]". However, for
[:a[:abc]b][b:] it gives unknown POSIX class "[:abc]b][b:]", which does not
seem right at all. PCRE does not allow closing square brackets in POSIX class
names.

Arguments:
  ptr      pointer to the character after the initial [ (colon, dot, equals)
  ptrend   pointer to the end of the pattern
  endptr   where to return a pointer to the terminating ':', '.', or '='

Returns:   TRUE or FALSE
*/

static BOOL
check_posix_syntax(PCRE2_SPTR ptr, PCRE2_SPTR ptrend, PCRE2_SPTR *endptr)
{
PCRE2_UCHAR terminator;  /* Don't combine these lines; the Solaris cc */
terminator = *ptr++;     /* compiler warns about "non-constant" initializer. */

for (; ptrend - ptr >= 2; ptr++)
  {
  if (*ptr == CHAR_BACKSLASH &&
      (ptr[1] == CHAR_RIGHT_SQUARE_BRACKET || ptr[1] == CHAR_BACKSLASH))
    ptr++;

  else if ((*ptr == CHAR_LEFT_SQUARE_BRACKET && ptr[1] == terminator) ||
            *ptr == CHAR_RIGHT_SQUARE_BRACKET) return FALSE;

  else if (*ptr == terminator && ptr[1] == CHAR_RIGHT_SQUARE_BRACKET)
    {
    *endptr = ptr;
    return TRUE;
    }
  }

return FALSE;
}



/*************************************************
*          Check POSIX class name                *
*************************************************/

/* This function is called to check the name given in a POSIX-style class entry
such as [:alnum:].

Arguments:
  ptr        points to the first letter
  len        the length of the name

Returns:     a value representing the name, or -1 if unknown
*/

static int
check_posix_name(PCRE2_SPTR ptr, int len)
{
const char *pn = posix_names;
int yield = 0;
while (posix_name_lengths[yield] != 0)
  {
  if (len == posix_name_lengths[yield] &&
    PRIV(strncmp_c8)(ptr, pn, (unsigned int)len) == 0) return yield;
  pn += posix_name_lengths[yield] + 1;
  yield++;
  }
return -1;
}



/*************************************************
*       Read a subpattern or VERB name           *
*************************************************/

/* This function is called from parse_regex() below whenever it needs to read
the name of a subpattern or a (*VERB) or an (*alpha_assertion). The initial
pointer must be to the preceding character. If that character is '*' we are
reading a verb or alpha assertion name. The pointer is updated to point after
the name, for a VERB or alpha assertion name, or after tha name's terminator
for a subpattern name. Returning both the offset and the name pointer is
redundant information, but some callers use one and some the other, so it is
simplest just to return both. When the name is in braces, spaces and tabs are
allowed (and ignored) at either end.

Arguments:
  ptrptr      points to the character pointer variable
  ptrend      points to the end of the input string
  utf         true if the input is UTF-encoded
  terminator  the terminator of a subpattern name must be this
  offsetptr   where to put the offset from the start of the pattern
  nameptr     where to put a pointer to the name in the input
  namelenptr  where to put the length of the name
  errcodeptr  where to put an error code
  cb          pointer to the compile data block

Returns:    TRUE if a name was read
            FALSE otherwise, with error code set
*/

static BOOL
read_name(PCRE2_SPTR *ptrptr, PCRE2_SPTR ptrend, BOOL utf, uint32_t terminator,
  PCRE2_SIZE *offsetptr, PCRE2_SPTR *nameptr, uint32_t *namelenptr,
  int *errorcodeptr, compile_block *cb)
{
PCRE2_SPTR ptr = *ptrptr;
BOOL is_group = (*ptr++ != CHAR_ASTERISK);
BOOL is_braced = terminator == CHAR_RIGHT_CURLY_BRACKET;

if (is_braced)
  while (ptr < ptrend && (*ptr == CHAR_SPACE || *ptr == CHAR_HT)) ptr++;

if (ptr >= ptrend)                 /* No characters in name */
  {
  *errorcodeptr = is_group? ERR62: /* Subpattern name expected */
                            ERR60; /* Verb not recognized or malformed */
  goto FAILED;
  }

*nameptr = ptr;
*offsetptr = (PCRE2_SIZE)(ptr - cb->start_pattern);

/* If this logic were ever to change, the matching function in pcre2_substitute.c
ought to be updated to match. */

/* In UTF mode, a group name may contain letters and decimal digits as defined
by Unicode properties, and underscores, but must not start with a digit. */

#ifdef SUPPORT_UNICODE
if (utf && is_group)
  {
  uint32_t c, type;

  GETCHAR(c, ptr);
  type = UCD_CHARTYPE(c);

  if (type == ucp_Nd)
    {
    *errorcodeptr = ERR44;
    goto FAILED;
    }

  for(;;)
    {
    if (type != ucp_Nd && PRIV(ucp_gentype)[type] != ucp_L &&
        c != CHAR_UNDERSCORE) break;
    ptr++;
    FORWARDCHARTEST(ptr, ptrend);
    if (ptr >= ptrend) break;
    GETCHAR(c, ptr);
    type = UCD_CHARTYPE(c);
    }
  }
else
#else
(void)utf;  /* Avoid compiler warning */
#endif      /* SUPPORT_UNICODE */

/* Handle non-group names and group names in non-UTF modes. A group name must
not start with a digit. If either of the others start with a digit it just
won't be recognized. */

  {
  if (is_group && IS_DIGIT(*ptr))
    {
    *errorcodeptr = ERR44;
    goto FAILED;
    }

  while (ptr < ptrend && MAX_255(*ptr) && (cb->ctypes[*ptr] & ctype_word) != 0)
    {
    ptr++;
    }
  }

/* Check name length */

if (ptr > *nameptr + MAX_NAME_SIZE)
  {
  *errorcodeptr = ERR48;
  goto FAILED;
  }
*namelenptr = (uint32_t)(ptr - *nameptr);

/* Subpattern names must not be empty, and their terminator is checked here.
(What follows a verb or alpha assertion name is checked separately.) */

if (is_group)
  {
  if (ptr == *nameptr)
    {
    *errorcodeptr = ERR62;   /* Subpattern name expected */
    goto FAILED;
    }
  if (is_braced)
    while (ptr < ptrend && (*ptr == CHAR_SPACE || *ptr == CHAR_HT)) ptr++;
  if (ptr >= ptrend || *ptr != (PCRE2_UCHAR)terminator)
    {
    *errorcodeptr = ERR42;
    goto FAILED;
    }
  ptr++;
  }

*ptrptr = ptr;
return TRUE;

FAILED:
*ptrptr = ptr;
return FALSE;
}



/*************************************************
*          Manage callouts at start of cycle     *
*************************************************/

/* At the start of a new item in parse_regex() we are able to record the
details of the previous item in a prior callout, and also to set up an
automatic callout if enabled. Avoid having two adjacent automatic callouts,
which would otherwise happen for items such as \Q that contribute nothing to
the parsed pattern.

Arguments:
  ptr              current pattern pointer
  pcalloutptr      points to a pointer to previous callout, or NULL
  auto_callout     TRUE if auto_callouts are enabled
  parsed_pattern   the parsed pattern pointer
  cb               compile block

Returns: possibly updated parsed_pattern pointer.
*/

static uint32_t *
manage_callouts(PCRE2_SPTR ptr, uint32_t **pcalloutptr, BOOL auto_callout,
  uint32_t *parsed_pattern, compile_block *cb)
{
uint32_t *previous_callout = *pcalloutptr;

if (previous_callout != NULL) previous_callout[2] = (uint32_t)(ptr -
  cb->start_pattern - (PCRE2_SIZE)previous_callout[1]);

if (!auto_callout) previous_callout = NULL; else
  {
  if (previous_callout == NULL ||
      previous_callout != parsed_pattern - 4 ||
      previous_callout[3] != 255)
    {
    previous_callout = parsed_pattern;  /* Set up new automatic callout */
    parsed_pattern += 4;
    previous_callout[0] = META_CALLOUT_NUMBER;
    previous_callout[2] = 0;
    previous_callout[3] = 255;
    }
  previous_callout[1] = (uint32_t)(ptr - cb->start_pattern);
  }

*pcalloutptr = previous_callout;
return parsed_pattern;
}



/*************************************************
*          Handle \d, \D, \s, \S, \w, \W         *
*************************************************/

/* This function is called from parse_regex() below, both for freestanding
escapes, and those within classes, to handle those escapes that may change when
Unicode property support is requested. Note that PCRE2_UCP will never be set
without Unicode support because that is checked when pcre2_compile() is called.

Arguments:
  escape          the ESC_... value
  parsed_pattern  where to add the code
  options         options bits
  xoptions        extra options bits

Returns:          updated value of parsed_pattern
*/
static uint32_t *
handle_escdsw(int escape, uint32_t *parsed_pattern, uint32_t options,
  uint32_t xoptions)
{
uint32_t ascii_option = 0;
uint32_t prop = ESC_p;

switch(escape)
  {
  case ESC_D:
  prop = ESC_P;
  /* Fall through */
  case ESC_d:
  ascii_option = PCRE2_EXTRA_ASCII_BSD;
  break;

  case ESC_S:
  prop = ESC_P;
  /* Fall through */
  case ESC_s:
  ascii_option = PCRE2_EXTRA_ASCII_BSS;
  break;

  case ESC_W:
  prop = ESC_P;
  /* Fall through */
  case ESC_w:
  ascii_option = PCRE2_EXTRA_ASCII_BSW;
  break;
  }

if ((options & PCRE2_UCP) == 0 || (xoptions & ascii_option) != 0)
  {
  *parsed_pattern++ = META_ESCAPE + escape;
  }
else
  {
  *parsed_pattern++ = META_ESCAPE + prop;
  switch(escape)
    {
    case ESC_d:
    case ESC_D:
    *parsed_pattern++ = (PT_PC << 16) | ucp_Nd;
    break;

    case ESC_s:
    case ESC_S:
    *parsed_pattern++ = PT_SPACE << 16;
    break;

    case ESC_w:
    case ESC_W:
    *parsed_pattern++ = PT_WORD << 16;
    break;
    }
  }

return parsed_pattern;
}



/*************************************************
* Maximum size of parsed_pattern for given input *
*************************************************/

/* This function is called from parse_regex() below, to determine the amount
of memory to allocate for parsed_pattern. It is also called to check whether
the amount of data written respects the amount of memory allocated.

Arguments:
  ptr             points to the start of the pattern
  ptrend          points to the end of the pattern
  utf             TRUE in UTF mode
  options         the options bits

Returns:          the number of uint32_t units for parsed_pattern
*/
static ptrdiff_t
max_parsed_pattern(PCRE2_SPTR ptr, PCRE2_SPTR ptrend, BOOL utf,
  uint32_t options)
{
PCRE2_SIZE big32count = 0;
ptrdiff_t parsed_size_needed;

/* When PCRE2_AUTO_CALLOUT is not set, in all but one case the number of
unsigned 32-bit ints written out to the parsed pattern is bounded by the length
of the pattern. The exceptional case is when running in 32-bit, non-UTF mode,
when literal characters greater than META_END (0x80000000) have to be coded as
two units. In this case, therefore, we scan the pattern to check for such
values. */

#if PCRE2_CODE_UNIT_WIDTH == 32
if (!utf)
  {
  PCRE2_SPTR p;
  for (p = ptr; p < ptrend; p++) if (*p >= META_END) big32count++;
  }
#else
(void)utf;  /* Avoid compiler warning */
#endif

parsed_size_needed = (ptrend - ptr) + big32count;

/* When PCRE2_AUTO_CALLOUT is set we have to assume a numerical callout (4
elements) for each character. This is overkill, but memory is plentiful these
days. */

if ((options & PCRE2_AUTO_CALLOUT) != 0)
  parsed_size_needed += (ptrend - ptr) * 4;

return parsed_size_needed;
}



/*************************************************
*      Parse regex and identify named groups     *
*************************************************/

/* This function is called first of all. It scans the pattern and does two
things: (1) It identifies capturing groups and makes a table of named capturing
groups so that information about them is fully available to both the compiling
scans. (2) It writes a parsed version of the pattern with comments omitted and
escapes processed into the parsed_pattern vector.

Arguments:
  ptr             points to the start of the pattern
  options         compiling dynamic options (may change during the scan)
  has_lookbehind  points to a boolean, set TRUE if a lookbehind is found
  cb              pointer to the compile data block

Returns:   zero on success or a non-zero error code, with the
             error offset placed in the cb field
*/

/* A structure and some flags for dealing with nested groups. */

typedef struct nest_save {
  uint16_t  nest_depth;
  uint16_t  reset_group;
  uint16_t  max_group;
  uint16_t  flags;
  uint32_t  options;
  uint32_t  xoptions;
} nest_save;

#define NSF_RESET          0x0001u
#define NSF_CONDASSERT     0x0002u
#define NSF_ATOMICSR       0x0004u

/* Options that are changeable within the pattern must be tracked during
parsing. Some (e.g. PCRE2_EXTENDED) are implemented entirely during parsing,
but all must be tracked so that META_OPTIONS items set the correct values for
the main compiling phase. */

#define PARSE_TRACKED_OPTIONS (PCRE2_CASELESS|PCRE2_DOTALL|PCRE2_DUPNAMES| \
  PCRE2_EXTENDED|PCRE2_EXTENDED_MORE|PCRE2_MULTILINE|PCRE2_NO_AUTO_CAPTURE| \
  PCRE2_UNGREEDY)

#define PARSE_TRACKED_EXTRA_OPTIONS (PCRE2_EXTRA_CASELESS_RESTRICT| \
  PCRE2_EXTRA_ASCII_BSD|PCRE2_EXTRA_ASCII_BSS|PCRE2_EXTRA_ASCII_BSW| \
  PCRE2_EXTRA_ASCII_DIGIT|PCRE2_EXTRA_ASCII_POSIX)

/* States used for analyzing ranges in character classes. The two OK values
must be last. */

enum {
  RANGE_NO, /* State after '[' (initial), or '[a-z'; hyphen is literal */
  RANGE_STARTED, /* State after '[1-'; last-emitted code is META_RANGE_XYZ */
  RANGE_FORBID_NO, /* State after '[\d'; '-]' is allowed but not '-1]' */
  RANGE_FORBID_STARTED, /* State after '[\d-'*/
  RANGE_OK_ESCAPED, /* State after '[\1'; hyphen may be a range */
  RANGE_OK_LITERAL /* State after '[1'; hyphen may be a range */
};

/* States used for analyzing operators and operands in extended character
classes. */

enum {
  CLASS_OP_EMPTY, /* At start of an expression; empty previous contents */
  CLASS_OP_OPERAND, /* Have preceding operand; after "z" a "--" can follow */
  CLASS_OP_OPERATOR /* Have preceding operator; after "--" operand must follow */
};

/* States used for determining the parse mode in character classes. The two
PERL_EXT values must be last. */

enum {
  CLASS_MODE_NORMAL, /* Ordinary PCRE2 '[...]' class. */
  CLASS_MODE_ALT_EXT, /* UTS#18-style extended '[...]' class. */
  CLASS_MODE_PERL_EXT, /* Perl extended '(?[...])' class. */
  CLASS_MODE_PERL_EXT_LEAF /* Leaf within extended '(?[ [...] ])' class. */
};

/* Only in 32-bit mode can there be literals > META_END. A macro encapsulates
the storing of literal values in the main parsed pattern, where they can always
be quantified. */

#if PCRE2_CODE_UNIT_WIDTH == 32
#define PARSED_LITERAL(c, p) \
  { \
  if (c >= META_END) *p++ = META_BIGVALUE; \
  *p++ = c; \
  okquantifier = TRUE; \
  }
#else
#define PARSED_LITERAL(c, p) *p++ = c; okquantifier = TRUE;
#endif

/* Here's the actual function. */

static int parse_regex(PCRE2_SPTR ptr, uint32_t options, uint32_t xoptions,
  BOOL *has_lookbehind, compile_block *cb)
{
uint32_t c;
uint32_t delimiter;
uint32_t namelen;
uint32_t class_range_state;
uint32_t class_op_state;
uint32_t class_mode_state;
uint32_t *class_start;
uint32_t *verblengthptr = NULL;     /* Value avoids compiler warning */
uint32_t *verbstartptr = NULL;
uint32_t *previous_callout = NULL;
uint32_t *parsed_pattern = cb->parsed_pattern;
uint32_t *parsed_pattern_end = cb->parsed_pattern_end;
uint32_t *this_parsed_item = NULL;
uint32_t *prev_parsed_item = NULL;
uint32_t meta_quantifier = 0;
uint32_t add_after_mark = 0;
uint16_t nest_depth = 0;
int16_t class_depth_m1 = -1; /* The m1 means minus 1. */
int16_t class_maxdepth_m1 = -1;
int after_manual_callout = 0;
int expect_cond_assert = 0;
int errorcode = 0;
int escape;
int i;
BOOL inescq = FALSE;
BOOL inverbname = FALSE;
BOOL utf = (options & PCRE2_UTF) != 0;
BOOL auto_callout = (options & PCRE2_AUTO_CALLOUT) != 0;
BOOL isdupname;
BOOL negate_class;
BOOL okquantifier = FALSE;
PCRE2_SPTR thisptr;
PCRE2_SPTR name;
PCRE2_SPTR ptrend = cb->end_pattern;
PCRE2_SPTR verbnamestart = NULL;    /* Value avoids compiler warning */
PCRE2_SPTR class_range_forbid_ptr = NULL;
named_group *ng;
nest_save *top_nest, *end_nests;
#ifdef PCRE2_DEBUG
uint32_t *parsed_pattern_check;
ptrdiff_t parsed_pattern_extra = 0;
ptrdiff_t parsed_pattern_extra_check = 0;
PCRE2_SPTR ptr_check;
#endif

PCRE2_ASSERT(parsed_pattern != NULL);

/* Insert leading items for word and line matching (features provided for the
benefit of pcre2grep). */

if ((xoptions & PCRE2_EXTRA_MATCH_LINE) != 0)
  {
  *parsed_pattern++ = META_CIRCUMFLEX;
  *parsed_pattern++ = META_NOCAPTURE;
  }
else if ((xoptions & PCRE2_EXTRA_MATCH_WORD) != 0)
  {
  *parsed_pattern++ = META_ESCAPE + ESC_b;
  *parsed_pattern++ = META_NOCAPTURE;
  }

#ifdef PCRE2_DEBUG
parsed_pattern_check = parsed_pattern;
ptr_check = ptr;
#endif

/* If the pattern is actually a literal string, process it separately to avoid
cluttering up the main loop. */

if ((options & PCRE2_LITERAL) != 0)
  {
  while (ptr < ptrend)
    {
    if (parsed_pattern >= parsed_pattern_end)
      {
      PCRE2_DEBUG_UNREACHABLE();
      errorcode = ERR63;  /* Internal error (parsed pattern overflow) */
      goto FAILED;
      }
    thisptr = ptr;
    GETCHARINCTEST(c, ptr);
    if (auto_callout)
      parsed_pattern = manage_callouts(thisptr, &previous_callout,
        auto_callout, parsed_pattern, cb);
    PARSED_LITERAL(c, parsed_pattern);
    }
  goto PARSED_END;
  }

/* Process a real regex which may contain meta-characters. */

top_nest = NULL;
end_nests = (nest_save *)(cb->start_workspace + cb->workspace_size);

/* The size of the nest_save structure might not be a factor of the size of the
workspace. Therefore we must round down end_nests so as to correctly avoid
creating a nest_save that spans the end of the workspace. */

end_nests = (nest_save *)((char *)end_nests -
  ((cb->workspace_size * sizeof(PCRE2_UCHAR)) % sizeof(nest_save)));

/* PCRE2_EXTENDED_MORE implies PCRE2_EXTENDED */

if ((options & PCRE2_EXTENDED_MORE) != 0) options |= PCRE2_EXTENDED;

/* Now scan the pattern */

while (ptr < ptrend)
  {
  int prev_expect_cond_assert;
  uint32_t min_repeat = 0, max_repeat = 0;
  uint32_t set, unset, *optset;
  uint32_t xset, xunset, *xoptset;
  uint32_t terminator;
  uint32_t prev_meta_quantifier;
  BOOL prev_okquantifier;
  PCRE2_SPTR tempptr;
  PCRE2_SIZE offset;

  if (nest_depth > cb->cx->parens_nest_limit)
    {
    errorcode = ERR19;
    goto FAILED;        /* Parentheses too deeply nested */
    }

  /* Check that we haven't emitted too much into parsed_pattern. We allocate
  a suitably-sized buffer upfront, then do unchecked writes to it. If we only
  write a little bit too much, everything will appear to be OK, because the
  upfront size is an overestimate... but a malicious pattern could end up
  forcing a write past the buffer end. We must catch this during
  development. */

#ifdef PCRE2_DEBUG
  /* Strong post-write check. Won't help in release builds - at this point
  the write has already occurred so it's too late. However, should stop us
  committing unsafe code. */
  PCRE2_ASSERT((parsed_pattern - parsed_pattern_check) +
               (parsed_pattern_extra - parsed_pattern_extra_check) <=
                 max_parsed_pattern(ptr_check, ptr, utf, options));
  parsed_pattern_check = parsed_pattern;
  parsed_pattern_extra_check = parsed_pattern_extra;
  ptr_check = ptr;
#endif

  if (parsed_pattern >= parsed_pattern_end)
    {
    /* Weak pre-write check; only ensures parsed_pattern[0] is writeable
    (but the code below can write many chars). Better than nothing. */
    PCRE2_DEBUG_UNREACHABLE();
    errorcode = ERR63;  /* Internal error (parsed pattern overflow) */
    goto FAILED;
    }

  /* If the last time round this loop something was added, parsed_pattern will
  no longer be equal to this_parsed_item. Remember where the previous item
  started and reset for the next item. Note that sometimes round the loop,
  nothing gets added (e.g. for ignored white space). */

  if (this_parsed_item != parsed_pattern)
    {
    prev_parsed_item = this_parsed_item;
    this_parsed_item = parsed_pattern;
    }

  /* Get next input character, save its position for callout handling. */

  thisptr = ptr;
  GETCHARINCTEST(c, ptr);

  /* Copy quoted literals until \E, allowing for the possibility of automatic
  callouts, except when processing a (*VERB) "name".  */

  if (inescq)
    {
    if (c == CHAR_BACKSLASH && ptr < ptrend && *ptr == CHAR_E)
      {
      inescq = FALSE;
      ptr++;   /* Skip E */
      }
    else
      {
      if (expect_cond_assert > 0)   /* A literal is not allowed if we are */
        {                           /* expecting a conditional assertion, */
        ptr--;                      /* but an empty \Q\E sequence is OK.  */
        errorcode = ERR28;
        goto FAILED;
        }
      if (inverbname)
        {                          /* Don't use PARSED_LITERAL() because it */
#if PCRE2_CODE_UNIT_WIDTH == 32    /* sets okquantifier. */
        if (c >= META_END) *parsed_pattern++ = META_BIGVALUE;
#endif
        *parsed_pattern++ = c;
        }
      else
        {
        if (after_manual_callout-- <= 0)
          parsed_pattern = manage_callouts(thisptr, &previous_callout,
            auto_callout, parsed_pattern, cb);
        PARSED_LITERAL(c, parsed_pattern);
        }
      meta_quantifier = 0;
      }
    continue;  /* Next character */
    }

  /* If we are processing the "name" part of a (*VERB:NAME) item, all
  characters up to the closing parenthesis are literals except when
  PCRE2_ALT_VERBNAMES is set. That causes backslash interpretation, but only \Q
  and \E and escaped characters are allowed (no character types such as \d). If
  PCRE2_EXTENDED is also set, we must ignore white space and # comments. Do
  this by not entering the special (*VERB:NAME) processing - they are then
  picked up below. Note that c is a character, not a code unit, so we must not
  use MAX_255 to test its size because MAX_255 tests code units and is assumed
  TRUE in 8-bit mode. */

  if (inverbname &&
       (
        /* EITHER: not both options set */
        ((options & (PCRE2_EXTENDED | PCRE2_ALT_VERBNAMES)) !=
                    (PCRE2_EXTENDED | PCRE2_ALT_VERBNAMES)) ||
#ifdef SUPPORT_UNICODE
        /* OR: character > 255 AND not Unicode Pattern White Space */
        (c > 255 && (c|1) != 0x200f && (c|1) != 0x2029) ||
#endif
        /* OR: not a # comment or isspace() white space */
        (c < 256 && c != CHAR_NUMBER_SIGN && (cb->ctypes[c] & ctype_space) == 0
#ifdef SUPPORT_UNICODE
        /* and not CHAR_NEL when Unicode is supported */
          && c != CHAR_NEL
#endif
       )))
    {
    PCRE2_SIZE verbnamelength;

    switch(c)
      {
      default:                     /* Don't use PARSED_LITERAL() because it */
#if PCRE2_CODE_UNIT_WIDTH == 32    /* sets okquantifier. */
      if (c >= META_END) *parsed_pattern++ = META_BIGVALUE;
#endif
      *parsed_pattern++ = c;
      break;

      case CHAR_RIGHT_PARENTHESIS:
      inverbname = FALSE;
      /* This is the length in characters */
      verbnamelength = (PCRE2_SIZE)(parsed_pattern - verblengthptr - 1);
      /* But the limit on the length is in code units */
      if (ptr - verbnamestart - 1 > (int)MAX_MARK)
        {
        ptr--;
        errorcode = ERR76;
        goto FAILED;
        }
      *verblengthptr = (uint32_t)verbnamelength;

      /* If this name was on a verb such as (*ACCEPT) which does not continue,
      a (*MARK) was generated for the name. We now add the original verb as the
      next item. */

      if (add_after_mark != 0)
        {
        *parsed_pattern++ = add_after_mark;
        add_after_mark = 0;
        }
      break;

      case CHAR_BACKSLASH:
      if ((options & PCRE2_ALT_VERBNAMES) != 0)
        {
        escape = PRIV(check_escape)(&ptr, ptrend, &c, &errorcode, options,
          xoptions, cb->bracount, FALSE, cb);
        if (errorcode != 0) goto FAILED;
        }
      else escape = 0;   /* Treat all as literal */

      switch(escape)
        {
        case 0:                    /* Don't use PARSED_LITERAL() because it */
#if PCRE2_CODE_UNIT_WIDTH == 32    /* sets okquantifier. */
        if (c >= META_END) *parsed_pattern++ = META_BIGVALUE;
#endif
        *parsed_pattern++ = c;
        break;

        case ESC_ub:
        *parsed_pattern++ = CHAR_u;
        PARSED_LITERAL(CHAR_LEFT_CURLY_BRACKET, parsed_pattern);
        break;

        case ESC_Q:
        inescq = TRUE;
        break;

        case ESC_E:           /* Ignore */
        break;

        default:
        errorcode = ERR40;    /* Invalid in verb name */
        goto FAILED;
        }
      }
    continue;   /* Next character in pattern */
    }

  /* Not a verb name character. At this point we must process everything that
  must not change the quantification state. This is mainly comments, but we
  handle \Q and \E here as well, so that an item such as A\Q\E+ is treated as
  A+, as in Perl. An isolated \E is ignored. */

  if (c == CHAR_BACKSLASH && ptr < ptrend)
    {
    if (*ptr == CHAR_Q || *ptr == CHAR_E)
      {
      inescq = *ptr == CHAR_Q;
      ptr++;
      continue;
      }
    }

  /* Skip over whitespace and # comments in extended mode. Note that c is a
  character, not a code unit, so we must not use MAX_255 to test its size
  because MAX_255 tests code units and is assumed TRUE in 8-bit mode. The
  whitespace characters are those designated as "Pattern White Space" by
  Unicode, which are the isspace() characters plus CHAR_NEL (newline), which is
  U+0085 in Unicode, plus U+200E, U+200F, U+2028, and U+2029. These are a
  subset of space characters that match \h and \v. */

  if ((options & PCRE2_EXTENDED) != 0)
    {
    if (c < 256 && (cb->ctypes[c] & ctype_space) != 0) continue;
#ifdef SUPPORT_UNICODE
    if (c == CHAR_NEL || (c|1) == 0x200f || (c|1) == 0x2029) continue;
#endif
    if (c == CHAR_NUMBER_SIGN)
      {
      while (ptr < ptrend)
        {
        if (IS_NEWLINE(ptr))      /* For non-fixed-length newline cases, */
          {                       /* IS_NEWLINE sets cb->nllen. */
          ptr += cb->nllen;
          break;
          }
        ptr++;
#ifdef SUPPORT_UNICODE
        if (utf) FORWARDCHARTEST(ptr, ptrend);
#endif
        }
      continue;  /* Next character in pattern */
      }
    }

  /* Skip over bracketed comments */

  if (c == CHAR_LEFT_PARENTHESIS && ptrend - ptr >= 2 &&
      ptr[0] == CHAR_QUESTION_MARK && ptr[1] == CHAR_NUMBER_SIGN)
    {
    while (++ptr < ptrend && *ptr != CHAR_RIGHT_PARENTHESIS);
    if (ptr >= ptrend)
      {
      errorcode = ERR18;  /* A special error for missing ) in a comment */
      goto FAILED;        /* to make it easier to debug. */
      }
    ptr++;
    continue;  /* Next character in pattern */
    }

  /* If the next item is not a quantifier, fill in length of any previous
  callout and create an auto callout if required. */

  if (c != CHAR_ASTERISK && c != CHAR_PLUS && c != CHAR_QUESTION_MARK &&
       (c != CHAR_LEFT_CURLY_BRACKET ||
         (tempptr = ptr,
         !read_repeat_counts(&tempptr, ptrend, NULL, NULL, &errorcode))))
    {
    if (after_manual_callout-- <= 0)
      {
      parsed_pattern = manage_callouts(thisptr, &previous_callout, auto_callout,
        parsed_pattern, cb);
      this_parsed_item = parsed_pattern;  /* New start for current item */
      }
    }

  /* If expect_cond_assert is 2, we have just passed (?( and are expecting an
  assertion, possibly preceded by a callout. If the value is 1, we have just
  had the callout and expect an assertion. There must be at least 3 more
  characters in all cases. When expect_cond_assert is 2, we know that the
  current character is an opening parenthesis, as otherwise we wouldn't be
  here. However, when it is 1, we need to check, and it's easiest just to check
  always. Note that expect_cond_assert may be negative, since all callouts just
  decrement it. */

  if (expect_cond_assert > 0)
    {
    BOOL ok = c == CHAR_LEFT_PARENTHESIS && ptrend - ptr >= 3 &&
              (ptr[0] == CHAR_QUESTION_MARK || ptr[0] == CHAR_ASTERISK);
    if (ok)
      {
      if (ptr[0] == CHAR_ASTERISK)  /* New alpha assertion format, possibly */
        {
        ok = MAX_255(ptr[1]) && (cb->ctypes[ptr[1]] & ctype_lcletter) != 0;
        }
      else switch(ptr[1])  /* Traditional symbolic format */
        {
        case CHAR_C:
        ok = expect_cond_assert == 2;
        break;

        case CHAR_EQUALS_SIGN:
        case CHAR_EXCLAMATION_MARK:
        break;

        case CHAR_LESS_THAN_SIGN:
        ok = ptr[2] == CHAR_EQUALS_SIGN || ptr[2] == CHAR_EXCLAMATION_MARK;
        break;

        default:
        ok = FALSE;
        }
      }

    if (!ok)
      {
      ptr--;   /* Adjust error offset */
      errorcode = ERR28;
      goto FAILED;
      }
    }

  /* Remember whether we are expecting a conditional assertion, and set the
  default for this item. */

  prev_expect_cond_assert = expect_cond_assert;
  expect_cond_assert = 0;

  /* Remember quantification status for the previous significant item, then set
  default for this item. */

  prev_okquantifier = okquantifier;
  prev_meta_quantifier = meta_quantifier;
  okquantifier = FALSE;
  meta_quantifier = 0;

  /* If the previous significant item was a quantifier, adjust the parsed code
  if there is a following modifier. The base meta value is always followed by
  the PLUS and QUERY values, in that order. We do this here rather than after
  reading a quantifier so that intervening comments and /x whitespace can be
  ignored without having to replicate code. */

  if (prev_meta_quantifier != 0 && (c == CHAR_QUESTION_MARK || c == CHAR_PLUS))
    {
    parsed_pattern[(prev_meta_quantifier == META_MINMAX)? -3 : -1] =
      prev_meta_quantifier + ((c == CHAR_QUESTION_MARK)?
        0x00020000u : 0x00010000u);
    continue;  /* Next character in pattern */
    }

  /* Process the next item in the main part of a pattern. */

  switch(c)
    {
    default:              /* Non-special character */
    PARSED_LITERAL(c, parsed_pattern);
    break;


    /* ---- Escape sequence ---- */

    case CHAR_BACKSLASH:
    tempptr = ptr;
    escape = PRIV(check_escape)(&ptr, ptrend, &c, &errorcode, options,
      xoptions, cb->bracount, FALSE, cb);
    if (errorcode != 0)
      {
      ESCAPE_FAILED:
      if ((xoptions & PCRE2_EXTRA_BAD_ESCAPE_IS_LITERAL) == 0)
        goto FAILED;
      ptr = tempptr;
      if (ptr >= ptrend) c = CHAR_BACKSLASH; else
        {
        GETCHARINCTEST(c, ptr);   /* Get character value, increment pointer */
        }
      escape = 0;                 /* Treat as literal character */
      }

    /* The escape was a data escape or literal character. */

    if (escape == 0)
      {
      PARSED_LITERAL(c, parsed_pattern);
      }

    /* The escape was a back (or forward) reference. We keep the offset in
    order to give a more useful diagnostic for a bad forward reference. For
    references to groups numbered less than 10 we can't use more than two items
    in parsed_pattern because they may be just two characters in the input (and
    in a 64-bit world an offset may need two elements). So for them, the offset
    of the first occurrent is held in a special vector. */

    else if (escape < 0)
      {
      offset = (PCRE2_SIZE)(ptr - cb->start_pattern - 1);
      escape = -escape - 1;
      *parsed_pattern++ = META_BACKREF | (uint32_t)escape;
      if (escape < 10)
        {
        if (cb->small_ref_offset[escape] == PCRE2_UNSET)
          cb->small_ref_offset[escape] = offset;
        }
      else
        {
        PUTOFFSET(offset, parsed_pattern);
        }
      okquantifier = TRUE;
      }

    /* The escape was a character class such as \d etc. or other special
    escape indicator such as \A or \X. Most of them generate just a single
    parsed item, but \P and \p are followed by a 16-bit type and a 16-bit
    value. They are supported only when Unicode is available. The type and
    value are packed into a single 32-bit value so that the whole sequences
    uses only two elements in the parsed_vector. This is because the same
    coding is used if \d (for example) is turned into \p{Nd} when PCRE2_UCP is
    set.

    There are also some cases where the escape sequence is followed by a name:
    \k{name}, \k<name>, and \k'name' are backreferences by name, and \g<name>
    and \g'name' are subroutine calls by name; \g{name} is a synonym for
    \k{name}. Note that \g<number> and \g'number' are handled by check_escape()
    and returned as a negative value (handled above). A name is coded as an
    offset into the pattern and a length. */

    else switch (escape)
      {
      case ESC_C:
#ifdef NEVER_BACKSLASH_C
      errorcode = ERR85;
      goto ESCAPE_FAILED;
#else
      if ((options & PCRE2_NEVER_BACKSLASH_C) != 0)
        {
        errorcode = ERR83;
        goto ESCAPE_FAILED;
        }
#endif
      okquantifier = TRUE;
      *parsed_pattern++ = META_ESCAPE + escape;
      break;

      /* This is a special return that happens only in EXTRA_ALT_BSUX mode,
      when \u{ is not followed by hex digits and }. It requests two literal
      characters, u and { and we need this, as otherwise \u{ 12} (for example)
      would be treated as u{12} now that spaces are allowed in quantifiers. */

      case ESC_ub:
      *parsed_pattern++ = CHAR_u;
      PARSED_LITERAL(CHAR_LEFT_CURLY_BRACKET, parsed_pattern);
      break;

      case ESC_X:
#ifndef SUPPORT_UNICODE
      errorcode = ERR45;   /* Supported only with Unicode support */
      goto ESCAPE_FAILED;
#endif
      case ESC_H:
      case ESC_h:
      case ESC_N:
      case ESC_R:
      case ESC_V:
      case ESC_v:
      okquantifier = TRUE;
      *parsed_pattern++ = META_ESCAPE + escape;
      break;

      default:  /* \A, \B, \b, \G, \K, \Z, \z cannot be quantified. */
      *parsed_pattern++ = META_ESCAPE + escape;
      break;

      /* Escapes that may change in UCP mode. */

      case ESC_d:
      case ESC_D:
      case ESC_s:
      case ESC_S:
      case ESC_w:
      case ESC_W:
      okquantifier = TRUE;
      parsed_pattern = handle_escdsw(escape, parsed_pattern, options,
        xoptions);
      break;

      /* Unicode property matching */

      case ESC_P:
      case ESC_p:
#ifdef SUPPORT_UNICODE
        {
        BOOL negated;
        uint16_t ptype = 0, pdata = 0;
        if (!get_ucp(&ptr, &negated, &ptype, &pdata, &errorcode, cb))
          goto ESCAPE_FAILED;
        if (negated) escape = (escape == ESC_P)? ESC_p : ESC_P;
        *parsed_pattern++ = META_ESCAPE + escape;
        *parsed_pattern++ = (ptype << 16) | pdata;
        okquantifier = TRUE;
        }
#else
      errorcode = ERR45;
      goto ESCAPE_FAILED;
#endif
      break;  /* End \P and \p */

      /* When \g is used with quotes or angle brackets as delimiters, it is a
      numerical or named subroutine call, and control comes here. When used
      with brace delimiters it is a numerical back reference and does not come
      here because check_escape() returns it directly as a reference. \k is
      always a named back reference. */

      case ESC_g:
      case ESC_k:
      if (ptr >= ptrend || (*ptr != CHAR_LEFT_CURLY_BRACKET &&
          *ptr != CHAR_LESS_THAN_SIGN && *ptr != CHAR_APOSTROPHE))
        {
        errorcode = (escape == ESC_g)? ERR57 : ERR69;
        goto ESCAPE_FAILED;
        }
      terminator = (*ptr == CHAR_LESS_THAN_SIGN)?
        CHAR_GREATER_THAN_SIGN : (*ptr == CHAR_APOSTROPHE)?
        CHAR_APOSTROPHE : CHAR_RIGHT_CURLY_BRACKET;

      /* For a non-braced \g, check for a numerical recursion. */

      if (escape == ESC_g && terminator != CHAR_RIGHT_CURLY_BRACKET)
        {
        PCRE2_SPTR p = ptr + 1;

        if (read_number(&p, ptrend, cb->bracount, MAX_GROUP_NUMBER, ERR61, &i,
            &errorcode))
          {
          if (p >= ptrend || *p != terminator)
            {
            errorcode = ERR57;
            goto ESCAPE_FAILED;
            }
          ptr = p;
          goto SET_RECURSION;
          }
        if (errorcode != 0) goto ESCAPE_FAILED;
        }

      /* Not a numerical recursion. Perl allows spaces and tabs after { and
      before } but not for other delimiters. */

      if (!read_name(&ptr, ptrend, utf, terminator, &offset, &name, &namelen,
          &errorcode, cb)) goto ESCAPE_FAILED;

      /* \k and \g when used with braces are back references, whereas \g used
      with quotes or angle brackets is a recursion */

      *parsed_pattern++ =
        (escape == ESC_k || terminator == CHAR_RIGHT_CURLY_BRACKET)?
          META_BACKREF_BYNAME : META_RECURSE_BYNAME;
      *parsed_pattern++ = namelen;

      PUTOFFSET(offset, parsed_pattern);
      okquantifier = TRUE;
      break;  /* End special escape processing */
      }
    break;    /* End escape sequence processing */


    /* ---- Single-character special items ---- */

    case CHAR_CIRCUMFLEX_ACCENT:
    *parsed_pattern++ = META_CIRCUMFLEX;
    break;

    case CHAR_DOLLAR_SIGN:
    *parsed_pattern++ = META_DOLLAR;
    break;

    case CHAR_DOT:
    *parsed_pattern++ = META_DOT;
    okquantifier = TRUE;
    break;


    /* ---- Single-character quantifiers ---- */

    case CHAR_ASTERISK:
    meta_quantifier = META_ASTERISK;
    goto CHECK_QUANTIFIER;

    case CHAR_PLUS:
    meta_quantifier = META_PLUS;
    goto CHECK_QUANTIFIER;

    case CHAR_QUESTION_MARK:
    meta_quantifier = META_QUERY;
    goto CHECK_QUANTIFIER;


    /* ---- Potential {n,m} quantifier ---- */

    case CHAR_LEFT_CURLY_BRACKET:
    if (!read_repeat_counts(&ptr, ptrend, &min_repeat, &max_repeat,
        &errorcode))
      {
      if (errorcode != 0) goto FAILED;     /* Error in quantifier. */
      PARSED_LITERAL(c, parsed_pattern);   /* Not a quantifier */
      break;                               /* No more quantifier processing */
      }
    meta_quantifier = META_MINMAX;
    /* Fall through */


    /* ---- Quantifier post-processing ---- */

    /* Check that a quantifier is allowed after the previous item. This
    guarantees that there is a previous item. */

    CHECK_QUANTIFIER:
    if (!prev_okquantifier)
      {
      errorcode = ERR9;
      goto FAILED_BACK;  // TODO https://github.com/PCRE2Project/pcre2/issues/549
      }

    /* Most (*VERB)s are not allowed to be quantified, but an ungreedy
    quantifier can be useful for (*ACCEPT) - meaning "succeed on backtrack", a
    sort of negated (*COMMIT). We therefore allow (*ACCEPT) to be quantified by
    wrapping it in non-capturing brackets, but we have to allow for a preceding
    (*MARK) for when (*ACCEPT) has an argument. */

    if (*prev_parsed_item == META_ACCEPT)
      {
      uint32_t *p;
      for (p = parsed_pattern - 1; p >= verbstartptr; p--) p[1] = p[0];
      *verbstartptr = META_NOCAPTURE;
      parsed_pattern[1] = META_KET;
      parsed_pattern += 2;

#ifdef PCRE2_DEBUG
      PCRE2_ASSERT(parsed_pattern_extra >= 2);
      parsed_pattern_extra -= 2;
#endif
      }

    /* Now we can put the quantifier into the parsed pattern vector. At this
    stage, we have only the basic quantifier. The check for a following + or ?
    modifier happens at the top of the loop, after any intervening comments
    have been removed. */

    *parsed_pattern++ = meta_quantifier;
    if (c == CHAR_LEFT_CURLY_BRACKET)
      {
      *parsed_pattern++ = min_repeat;
      *parsed_pattern++ = max_repeat;
      }
    break;


    /* ---- Character class ---- */

    case CHAR_LEFT_SQUARE_BRACKET:

    /* In another (POSIX) regex library, the ugly syntax [[:<:]] and [[:>:]] is
    used for "start of word" and "end of word". As these are otherwise illegal
    sequences, we don't break anything by recognizing them. They are replaced
    by \b(?=\w) and \b(?<=\w) respectively. Sequences like [a[:<:]] are
    erroneous and are handled by the normal code below. */

    if (ptrend - ptr >= 6 &&
         (PRIV(strncmp_c8)(ptr, STRING_WEIRD_STARTWORD, 6) == 0 ||
          PRIV(strncmp_c8)(ptr, STRING_WEIRD_ENDWORD, 6) == 0))
      {
      *parsed_pattern++ = META_ESCAPE + ESC_b;

      if (ptr[2] == CHAR_LESS_THAN_SIGN)
        {
        *parsed_pattern++ = META_LOOKAHEAD;
        }
      else
        {
        *parsed_pattern++ = META_LOOKBEHIND;
        *has_lookbehind = TRUE;

        /* The offset is used only for the "non-fixed length" error; this won't
        occur here, so just store zero. */

        PUTOFFSET((PCRE2_SIZE)0, parsed_pattern);
        }

      if ((options & PCRE2_UCP) == 0)
        *parsed_pattern++ = META_ESCAPE + ESC_w;
      else
        {
        *parsed_pattern++ = META_ESCAPE + ESC_p;
        *parsed_pattern++ = PT_WORD << 16;
        }
      *parsed_pattern++ = META_KET;
      ptr += 6;
      okquantifier = TRUE;
      break;
      }

    /* PCRE supports POSIX class stuff inside a class. Perl gives an error if
    they are encountered at the top level, so we'll do that too. */

    if (ptr < ptrend && (*ptr == CHAR_COLON || *ptr == CHAR_DOT ||
         *ptr == CHAR_EQUALS_SIGN) &&
        check_posix_syntax(ptr, ptrend, &tempptr))
      {
      errorcode = (*ptr-- == CHAR_COLON)? ERR12 : ERR13;
      goto FAILED;
      }

    class_mode_state = ((options & PCRE2_ALT_EXTENDED_CLASS) != 0)?
        CLASS_MODE_ALT_EXT : CLASS_MODE_NORMAL;

    /* Jump here from '(?[...])'. That jump must initialize class_mode_state,
    set c to the '[' character, and ptr to just after the '['. */

    FROM_PERL_EXTENDED_CLASS:
    okquantifier = TRUE;

    /* In an EBCDIC environment, Perl treats alphabetic ranges specially
    because there are holes in the encoding, and simply using the range A-Z
    (for example) would include the characters in the holes. This applies only
    to ranges where both values are literal; [\xC1-\xE9] is different to [A-Z]
    in this respect. In order to accommodate this, we keep track of whether
    character values are literal or not, and a state variable for handling
    ranges. */

    /* Loop for the contents of the class. Classes may be nested, if
    PCRE2_ALT_EXTENDED_CLASS is set, or the class is of the form (?[...]). */

    /* c is still set to '[' so the loop will handle the start of the class. */

    class_depth_m1 = -1;
    class_maxdepth_m1 = -1;
    class_range_state = RANGE_NO;
    class_op_state = CLASS_OP_EMPTY;
    class_start = NULL;

    for (;;)
      {
      BOOL char_is_literal = TRUE;

      /* Inside \Q...\E everything is literal except \E */

      if (inescq)
        {
        if (c == CHAR_BACKSLASH && ptr < ptrend && *ptr == CHAR_E)
          {
          inescq = FALSE;                   /* Reset literal state */
          ptr++;                            /* Skip the 'E' */
          goto CLASS_CONTINUE;
          }

        /* Surprisingly, you cannot use \Q..\E to escape a character inside a
        Perl extended class. However, empty \Q\E sequences are allowed, so here
        were're only giving an error if the \Q..\E is non-empty. */

        if (class_mode_state == CLASS_MODE_PERL_EXT)
          {
          errorcode = ERR116;
          goto FAILED;
          }

        goto CLASS_LITERAL;
        }

      /* Skip over space and tab (only) in extended-more mode, or anywhere
      inside a Perl extended class (which implies /xx). */

      if ((c == CHAR_SPACE || c == CHAR_HT) &&
          ((options & PCRE2_EXTENDED_MORE) != 0 ||
           class_mode_state >= CLASS_MODE_PERL_EXT))
        goto CLASS_CONTINUE;

      /* Handle POSIX class names. Perl allows a negation extension of the
      form [:^name:]. A square bracket that doesn't match the syntax is
      treated as a literal. We also recognize the POSIX constructions
      [.ch.] and [=ch=] ("collating elements") and fault them, as Perl
      5.6 and 5.8 do. */

      if (class_depth_m1 >= 0 &&
          c == CHAR_LEFT_SQUARE_BRACKET &&
          ptrend - ptr >= 3 &&
          (*ptr == CHAR_COLON || *ptr == CHAR_DOT ||
           *ptr == CHAR_EQUALS_SIGN) &&
          check_posix_syntax(ptr, ptrend, &tempptr))
        {
        BOOL posix_negate = FALSE;
        int posix_class;

        /* Perl treats a hyphen before a POSIX class as a literal, not the
        start of a range. However, it gives a warning in its warning mode. PCRE
        does not have a warning mode, so we give an error, because this is
        likely an error on the user's part. */

        if (class_range_state == RANGE_STARTED)
          {
          ptr = tempptr + 2;
          errorcode = ERR50;
          goto FAILED;
          }

        /* Perl treats a hyphen after a POSIX class as a literal, not the
        start of a range. However, it gives a warning in its warning mode
        unless the hyphen is the last character in the class. PCRE does not
        have a warning mode, so we give an error, because this is likely an
        error on the user's part.

        Roll back to the hyphen for the error position. */

        if (class_range_state == RANGE_FORBID_STARTED)
          {
          ptr = class_range_forbid_ptr;
          errorcode = ERR50;
          goto FAILED;
          }

        /* Disallow implicit union in Perl extended classes. */

        if (class_op_state == CLASS_OP_OPERAND &&
            class_mode_state == CLASS_MODE_PERL_EXT)
          {
          ptr = tempptr + 2;
          errorcode = ERR113;
          goto FAILED;
          }

        if (*ptr != CHAR_COLON)
          {
          ptr = tempptr + 2;
          errorcode = ERR13;
          goto FAILED;
          }

        if (*(++ptr) == CHAR_CIRCUMFLEX_ACCENT)
          {
          posix_negate = TRUE;
          ptr++;
          }

        posix_class = check_posix_name(ptr, (int)(tempptr - ptr));
        ptr = tempptr + 2;
        if (posix_class < 0)
          {
          errorcode = ERR30;
          goto FAILED;
          }

        /* Set "a hyphen is forbidden to be the start of a range". For the '-]'
        case, the hyphen is treated as a literal, but for '-1' it is disallowed
        (because it would be interpreted as range). */

        class_range_state = RANGE_FORBID_NO;
        class_op_state = CLASS_OP_OPERAND;

        /* When PCRE2_UCP is set, unless PCRE2_EXTRA_ASCII_POSIX is set, some
        of the POSIX classes are converted to use Unicode properties \p or \P
        or, in one case, \h or \H. The substitutes table has two values per
        class, containing the type and value of a \p or \P item. The special
        cases are specified with a negative type: a non-zero value causes \h or
        \H to be used, and a zero value falls through to behave like a non-UCP
        POSIX class. There are now also some extra options that force ASCII for
        some classes. */

#ifdef SUPPORT_UNICODE
        if ((options & PCRE2_UCP) != 0 &&
            (xoptions & PCRE2_EXTRA_ASCII_POSIX) == 0 &&
            !((xoptions & PCRE2_EXTRA_ASCII_DIGIT) != 0 &&
              (posix_class == PC_DIGIT || posix_class == PC_XDIGIT)))
          {
          int ptype = posix_substitutes[2*posix_class];
          int pvalue = posix_substitutes[2*posix_class + 1];

          if (ptype >= 0)
            {
            *parsed_pattern++ = META_ESCAPE + (posix_negate? ESC_P : ESC_p);
            *parsed_pattern++ = (ptype << 16) | pvalue;
            goto CLASS_CONTINUE;
            }

          if (pvalue != 0)
            {
            *parsed_pattern++ = META_ESCAPE + (posix_negate? ESC_H : ESC_h);
            goto CLASS_CONTINUE;
            }

          /* Fall through */
          }
#endif  /* SUPPORT_UNICODE */

        /* Non-UCP POSIX class */

        *parsed_pattern++ = posix_negate? META_POSIX_NEG : META_POSIX;
        *parsed_pattern++ = posix_class;
        }

      /* Check for the start of the outermost class, or the start of a nested class. */

      else if ((c == CHAR_LEFT_SQUARE_BRACKET &&
                (class_depth_m1 < 0 || class_mode_state == CLASS_MODE_ALT_EXT ||
                 class_mode_state == CLASS_MODE_PERL_EXT)) ||
               (c == CHAR_LEFT_PARENTHESIS &&
                class_mode_state == CLASS_MODE_PERL_EXT))
        {
        uint32_t start_c = c;
        uint32_t new_class_mode_state;

        /* Update the class mode, if moving into a 'leaf' inside a Perl extended
        class. */

        if (start_c == CHAR_LEFT_SQUARE_BRACKET &&
            class_mode_state == CLASS_MODE_PERL_EXT && class_depth_m1 >= 0)
          new_class_mode_state = CLASS_MODE_PERL_EXT_LEAF;
        else
          new_class_mode_state = class_mode_state;

        /* Tidy up the other class before starting the nested class. */
        /* -[ beginning a nested class is a literal '-' */

        if (class_range_state == RANGE_STARTED)
          parsed_pattern[-1] = CHAR_MINUS;

        /* Disallow implicit union in Perl extended classes. */

        if (class_op_state == CLASS_OP_OPERAND &&
            class_mode_state == CLASS_MODE_PERL_EXT)
          {
          errorcode = ERR113;
          goto FAILED;
          }

        /* Validate nesting depth */
        if (class_depth_m1 >= ECLASS_NEST_LIMIT - 1)
          {
          errorcode = ERR107;
          goto FAILED;        /* Classes too deeply nested */
          }

        /* Process the character class start. If the first character is '^', set
        the negation flag. If the first few characters (either before or after ^)
        are \Q\E or \E or space or tab in extended-more mode, we skip them too.
        This makes for compatibility with Perl. */

        negate_class = FALSE;
        for (;;)
          {
          if (ptr >= ptrend)
            {
            if (start_c == CHAR_LEFT_PARENTHESIS)
              errorcode = ERR14;  /* Missing terminating ')' */
            else
              errorcode = ERR6;   /* Missing terminating ']' */
            goto FAILED;
            }

          GETCHARINCTEST(c, ptr);
          if (new_class_mode_state == CLASS_MODE_PERL_EXT) break;
          else if (c == CHAR_BACKSLASH)
            {
            if (ptr < ptrend && *ptr == CHAR_E) ptr++;
            else if (ptrend - ptr >= 3 &&
                PRIV(strncmp_c8)(ptr, STR_Q STR_BACKSLASH STR_E, 3) == 0)
              ptr += 3;
            else
              break;
            }
          else if ((c == CHAR_SPACE || c == CHAR_HT) &&  /* Note: just these two */
                   ((options & PCRE2_EXTENDED_MORE) != 0 ||
                    new_class_mode_state >= CLASS_MODE_PERL_EXT))
            continue;
          else if (!negate_class && c == CHAR_CIRCUMFLEX_ACCENT)
            negate_class = TRUE;
          else break;
          }

        /* Now the real contents of the class; c has the first "real" character.
        Empty classes are permitted only if the option is set, and if it's not
        a Perl-extended class. */

        if (c == CHAR_RIGHT_SQUARE_BRACKET &&
            (cb->external_options & PCRE2_ALLOW_EMPTY_CLASS) != 0 &&
            new_class_mode_state < CLASS_MODE_PERL_EXT)
          {
          PCRE2_ASSERT(start_c == CHAR_LEFT_SQUARE_BRACKET);

          if (class_start != NULL)
            {
            PCRE2_ASSERT(class_depth_m1 >= 0);
            /* Represents that the class is an extended class. */
            *class_start |= CLASS_IS_ECLASS;
            class_start = NULL;
            }

          *parsed_pattern++ = negate_class? META_CLASS_EMPTY_NOT : META_CLASS_EMPTY;

          /* Leave nesting depth unchanged; but check for zero depth to handle the
          very first (top-level) class being empty. */
          if (class_depth_m1 < 0) break;

          class_range_state = RANGE_NO; /* for processing the containing class */
          class_op_state = CLASS_OP_OPERAND;
          goto CLASS_CONTINUE;
          }

        /* Enter a non-empty class. */

        if (class_start != NULL)
          {
          PCRE2_ASSERT(class_depth_m1 >= 0);
          /* Represents that the class is an extended class. */
          *class_start |= CLASS_IS_ECLASS;
          class_start = NULL;
          }

        class_start = parsed_pattern;
        *parsed_pattern++ = negate_class? META_CLASS_NOT : META_CLASS;
        class_range_state = RANGE_NO;
        class_op_state = CLASS_OP_EMPTY;
        class_mode_state = new_class_mode_state;
        ++class_depth_m1;
        if (class_maxdepth_m1 < class_depth_m1)
          class_maxdepth_m1 = class_depth_m1;
        /* Reset; no op seen yet at new depth. */
        cb->class_op_used[class_depth_m1] = 0;

        /* Implement the special start-of-class literal meaning of ']'. */
        if (c == CHAR_RIGHT_SQUARE_BRACKET &&
            new_class_mode_state != CLASS_MODE_PERL_EXT)
          {
          class_range_state = RANGE_OK_LITERAL;
          class_op_state = CLASS_OP_OPERAND;
          PARSED_LITERAL(c, parsed_pattern);
          goto CLASS_CONTINUE;
          }

        continue;  /* We have already loaded c with the next character */
        }

      /* Check for the end of the class. */

      else if (c == CHAR_RIGHT_SQUARE_BRACKET ||
               (c == CHAR_RIGHT_PARENTHESIS && class_mode_state == CLASS_MODE_PERL_EXT))
        {
        /* In Perl extended mode, the ']' can only be used to match the
        opening '[', and ')' must match an opening parenthesis. */
        if (class_mode_state == CLASS_MODE_PERL_EXT)
          {
          if (c == CHAR_RIGHT_SQUARE_BRACKET && class_depth_m1 != 0)
            {
            errorcode = ERR14;
            goto FAILED_BACK;
            }
          if (c == CHAR_RIGHT_PARENTHESIS && class_depth_m1 < 1)
            {
            errorcode = ERR22;
            goto FAILED;
            }
          }

        /* Check no trailing operator. */
        if (class_op_state == CLASS_OP_OPERATOR)
          {
          errorcode = ERR110;
          goto FAILED;
          }

        /* Check no empty expression for Perl extended expressions. */
        if (class_mode_state == CLASS_MODE_PERL_EXT &&
            class_op_state == CLASS_OP_EMPTY)
          {
          errorcode = ERR114;
          goto FAILED;
          }

        /* -] at the end of a class is a literal '-' */
        if (class_range_state == RANGE_STARTED)
          parsed_pattern[-1] = CHAR_MINUS;

        *parsed_pattern++ = META_CLASS_END;

        if (--class_depth_m1 < 0)
          {
          /* Check for and consume ')' after '(?[...]'. */
          PCRE2_ASSERT(class_mode_state != CLASS_MODE_PERL_EXT_LEAF);
          if (class_mode_state == CLASS_MODE_PERL_EXT)
            {
            if (ptr >= ptrend || *ptr != CHAR_RIGHT_PARENTHESIS)
              {
              errorcode = ERR115;
              goto FAILED;
              }

            ptr++;
            }

          break;
          }

        class_range_state = RANGE_NO; /* for processing the containing class */
        class_op_state = CLASS_OP_OPERAND;
        if (class_mode_state == CLASS_MODE_PERL_EXT_LEAF)
          class_mode_state = CLASS_MODE_PERL_EXT;
        /* The extended class flag has already
        been set for the parent class. */
        class_start = NULL;
        }

      /* Handle a Perl set binary operator */

      else if (class_mode_state == CLASS_MODE_PERL_EXT &&
               (c == CHAR_PLUS || c == CHAR_VERTICAL_LINE || c == CHAR_MINUS ||
                c == CHAR_AMPERSAND || c == CHAR_CIRCUMFLEX_ACCENT))
        {
        /* Check that there was a preceding operand. */
        if (class_op_state != CLASS_OP_OPERAND)
          {
          errorcode = ERR109;
          goto FAILED;
          }

        if (class_start != NULL)
          {
          PCRE2_ASSERT(class_depth_m1 >= 0);
          /* Represents that the class is an extended class. */
          *class_start |= CLASS_IS_ECLASS;
          class_start = NULL;
          }

        PCRE2_ASSERT(class_range_state != RANGE_STARTED &&
                     class_range_state != RANGE_FORBID_STARTED);

        *parsed_pattern++ = c == CHAR_PLUS? META_ECLASS_OR :
                            c == CHAR_VERTICAL_LINE? META_ECLASS_OR :
                            c == CHAR_MINUS? META_ECLASS_SUB :
                            c == CHAR_AMPERSAND? META_ECLASS_AND :
                            META_ECLASS_XOR;
        class_range_state = RANGE_NO;
        class_op_state = CLASS_OP_OPERATOR;
        }

      /* Handle a Perl set unary operator */

      else if (class_mode_state == CLASS_MODE_PERL_EXT &&
               c == CHAR_EXCLAMATION_MARK)
        {
        /* Check that the "!" has not got a preceding operand (i.e. it's the
        start of the class, or follows an operator). */
        if (class_op_state == CLASS_OP_OPERAND)
          {
          errorcode = ERR113;
          goto FAILED;
          }

        if (class_start != NULL)
          {
          PCRE2_ASSERT(class_depth_m1 >= 0);
          /* Represents that the class is an extended class. */
          *class_start |= CLASS_IS_ECLASS;
          class_start = NULL;
          }

        PCRE2_ASSERT(class_range_state != RANGE_STARTED &&
                     class_range_state != RANGE_FORBID_STARTED);

        *parsed_pattern++ = META_ECLASS_NOT;
        class_range_state = RANGE_NO;
        class_op_state = CLASS_OP_OPERATOR;
        }

      /* Handle a UTS#18 set operator */

      else if (class_mode_state == CLASS_MODE_ALT_EXT &&
               (c == CHAR_VERTICAL_LINE || c == CHAR_MINUS ||
                c == CHAR_AMPERSAND || c == CHAR_TILDE) &&
               ptr < ptrend && *ptr == c)
        {
        ++ptr;

        /* Check there isn't a triple-repetition. */
        if (ptr < ptrend && *ptr == c)
          {
          while (ptr < ptrend && *ptr == c) ++ptr;  /* Improve error offset. */
          errorcode = ERR108;
          goto FAILED;
          }

        /* Check for a preceding operand. */
        if (class_op_state != CLASS_OP_OPERAND)
          {
          errorcode = ERR109;
          goto FAILED;
          }

        /* Check for mixed precedence. Forbid [A--B&&C]. */
        if (cb->class_op_used[class_depth_m1] != 0 &&
            cb->class_op_used[class_depth_m1] != (uint8_t)c)
          {
          errorcode = ERR111;
          goto FAILED;
          }

        if (class_start != NULL)
          {
          PCRE2_ASSERT(class_depth_m1 >= 0);
          /* Represents that the class is an extended class. */
          *class_start |= CLASS_IS_ECLASS;
          class_start = NULL;
          }

        /* Dangling '-' before an operator is a literal */
        if (class_range_state == RANGE_STARTED)
          parsed_pattern[-1] = CHAR_MINUS;

        *parsed_pattern++ = c == CHAR_VERTICAL_LINE? META_ECLASS_OR :
                            c == CHAR_MINUS? META_ECLASS_SUB :
                            c == CHAR_AMPERSAND? META_ECLASS_AND :
                            META_ECLASS_XOR;
        class_range_state = RANGE_NO;
        class_op_state = CLASS_OP_OPERATOR;
        cb->class_op_used[class_depth_m1] = (uint8_t)c;
        }

      /* Handle escapes in a class */

      else if (c == CHAR_BACKSLASH)
        {
        tempptr = ptr;
        escape = PRIV(check_escape)(&ptr, ptrend, &c, &errorcode, options,
          xoptions, cb->bracount, TRUE, cb);

        if (errorcode != 0)
          {
          if ((xoptions & PCRE2_EXTRA_BAD_ESCAPE_IS_LITERAL) == 0 ||
              class_mode_state >= CLASS_MODE_PERL_EXT)
            goto FAILED;
          ptr = tempptr;
          if (ptr >= ptrend) c = CHAR_BACKSLASH; else
            {
            GETCHARINCTEST(c, ptr);   /* Get character value, increment pointer */
            }
          escape = 0;                 /* Treat as literal character */
          }

        switch(escape)
          {
          case 0:  /* Escaped character code point is in c */
          char_is_literal = FALSE;
          goto CLASS_LITERAL;      /* (a few lines above) */

          case ESC_b:
          c = CHAR_BS;    /* \b is backspace in a class */
          char_is_literal = FALSE;
          goto CLASS_LITERAL;

          case ESC_k:
          c = CHAR_k;     /* \k is not special in a class, just like \g */
          char_is_literal = FALSE;
          goto CLASS_LITERAL;

          case ESC_Q:
          inescq = TRUE;  /* Enter literal mode */
          goto CLASS_CONTINUE;

          case ESC_E:     /* Ignore orphan \E */
          goto CLASS_CONTINUE;

          case ESC_B:     /* Always an error in a class */
          case ESC_R:
          case ESC_X:
          errorcode = ERR7;
          ptr--;  // TODO https://github.com/PCRE2Project/pcre2/issues/549
          goto FAILED;

          case ESC_N:     /* Not permitted by Perl either */
          errorcode = ERR71;
          goto FAILED;

          case ESC_H:
          case ESC_h:
          case ESC_V:
          case ESC_v:
          *parsed_pattern++ = META_ESCAPE + escape;
          break;

          /* These escapes may be converted to Unicode property tests when
          PCRE2_UCP is set. */

          case ESC_d:
          case ESC_D:
          case ESC_s:
          case ESC_S:
          case ESC_w:
          case ESC_W:
          parsed_pattern = handle_escdsw(escape, parsed_pattern, options,
            xoptions);
          break;

          /* Explicit Unicode property matching */

          case ESC_P:
          case ESC_p:
#ifdef SUPPORT_UNICODE
            {
            BOOL negated;
            uint16_t ptype = 0, pdata = 0;
            if (!get_ucp(&ptr, &negated, &ptype, &pdata, &errorcode, cb))
              goto FAILED;

            /* In caseless matching, particular characteristics Lu, Ll, and Lt
            get converted to the general characteristic L&. That is, upper,
            lower, and title case letters are all conflated. */

            if ((options & PCRE2_CASELESS) != 0 && ptype == PT_PC &&
                (pdata == ucp_Lu || pdata == ucp_Ll || pdata == ucp_Lt))
              {
              ptype = PT_LAMP;
              pdata = 0;
              }

            if (negated) escape = (escape == ESC_P)? ESC_p : ESC_P;
            *parsed_pattern++ = META_ESCAPE + escape;
            *parsed_pattern++ = (ptype << 16) | pdata;
            }
#else
          errorcode = ERR45;
          goto FAILED;
#endif
          break;  /* End \P and \p */

          /* All others are not allowed in a class */

          default:
          PCRE2_DEBUG_UNREACHABLE();
          /* Fall through */

          case ESC_A:
          case ESC_Z:
          case ESC_z:
          case ESC_G:
          case ESC_K:
          case ESC_C:
          errorcode = ERR7;
          ptr--;  // TODO https://github.com/PCRE2Project/pcre2/issues/549
          goto FAILED;
          }

        /* All the switch-cases above which end in "break" describe a set
        of characters. None may start a range. */

        /* The second part of a range can be a single-character escape
        sequence (detected above), but not any of the other escapes. Perl
        treats a hyphen as a literal in such circumstances. However, in Perl's
        warning mode, a warning is given, so PCRE now faults it, as it is
        almost certainly a mistake on the user's part. */

        if (class_range_state == RANGE_STARTED)
          {
          errorcode = ERR50;
          goto FAILED;
          }

        /* Perl gives a warning unless the hyphen following a multi-character
        escape is the last character in the class. PCRE throws an error. */

        if (class_range_state == RANGE_FORBID_STARTED)
          {
          ptr = class_range_forbid_ptr;
          errorcode = ERR50;
          goto FAILED;
          }

        /* Disallow implicit union in Perl extended classes. */

        if (class_op_state == CLASS_OP_OPERAND &&
            class_mode_state == CLASS_MODE_PERL_EXT)
          {
          errorcode = ERR113;
          goto FAILED;
          }

        class_range_state = RANGE_FORBID_NO;
        class_op_state = CLASS_OP_OPERAND;
        }

      /* Forbid unescaped literals, and the special meaning of '-', inside a
      Perl extended class. */

      else if (class_mode_state == CLASS_MODE_PERL_EXT)
        {
        errorcode = ERR116;
        goto FAILED;
        }

      /* Handle potential start of range */

      else if (c == CHAR_MINUS && class_range_state >= RANGE_OK_ESCAPED)
        {
        *parsed_pattern++ = (class_range_state == RANGE_OK_LITERAL)?
          META_RANGE_LITERAL : META_RANGE_ESCAPED;
        class_range_state = RANGE_STARTED;
        }

      /* Handle forbidden start of range */

      else if (c == CHAR_MINUS && class_range_state == RANGE_FORBID_NO)
        {
        *parsed_pattern++ = CHAR_MINUS;
        class_range_state = RANGE_FORBID_STARTED;
        class_range_forbid_ptr = ptr;
        }

      /* Handle a literal character */

      else
        {
        CLASS_LITERAL:

        /* Disallow implicit union in Perl extended classes. */

        if (class_op_state == CLASS_OP_OPERAND &&
            class_mode_state == CLASS_MODE_PERL_EXT)
          {
          errorcode = ERR113;
          goto FAILED;
          }

        if (class_range_state == RANGE_STARTED)
          {
          if (c == parsed_pattern[-2])       /* Optimize one-char range */
            parsed_pattern--;
          else if (parsed_pattern[-2] > c)   /* Check range is in order */
            {
            errorcode = ERR8;
            goto FAILED_BACK;  // TODO https://github.com/PCRE2Project/pcre2/issues/549
            }
          else
            {
            if (!char_is_literal && parsed_pattern[-1] == META_RANGE_LITERAL)
              parsed_pattern[-1] = META_RANGE_ESCAPED;
            PARSED_LITERAL(c, parsed_pattern);
            }
          class_range_state = RANGE_NO;
          class_op_state = CLASS_OP_OPERAND;
          }
        else if (class_range_state == RANGE_FORBID_STARTED)
          {
          ptr = class_range_forbid_ptr;
          errorcode = ERR50;
          goto FAILED;
          }
        else  /* Potential start of range */
          {
          class_range_state = char_is_literal?
            RANGE_OK_LITERAL : RANGE_OK_ESCAPED;
          class_op_state = CLASS_OP_OPERAND;
          PARSED_LITERAL(c, parsed_pattern);
          }
        }

      /* Proceed to next thing in the class. */

      CLASS_CONTINUE:
      if (ptr >= ptrend)
        {
        if (class_mode_state == CLASS_MODE_PERL_EXT && class_depth_m1 > 0)
          errorcode = ERR14;   /* Missing terminating ')' */
        if (class_mode_state == CLASS_MODE_ALT_EXT &&
            class_depth_m1 == 0 && class_maxdepth_m1 == 1)
          errorcode = ERR112;  /* Missing terminating ']', but we saw '[ [ ]...' */
        else
          errorcode = ERR6;    /* Missing terminating ']' */
        goto FAILED;
        }
      GETCHARINCTEST(c, ptr);
      }     /* End of class-processing loop */

    break;  /* End of character class */


    /* ---- Opening parenthesis ---- */

    case CHAR_LEFT_PARENTHESIS:
    if (ptr >= ptrend) goto UNCLOSED_PARENTHESIS;

    /* If ( is not followed by ? it is either a capture or a special verb or an
    alpha assertion or a positive non-atomic lookahead. */

    if (*ptr != CHAR_QUESTION_MARK)
      {
      const char *vn;

      /* Handle capturing brackets (or non-capturing if auto-capture is turned
      off). */

      if (*ptr != CHAR_ASTERISK)
        {
        nest_depth++;
        if ((options & PCRE2_NO_AUTO_CAPTURE) == 0)
          {
          if (cb->bracount >= MAX_GROUP_NUMBER)
            {
            errorcode = ERR97;
            goto FAILED;
            }
          cb->bracount++;
          *parsed_pattern++ = META_CAPTURE | cb->bracount;
          }
        else *parsed_pattern++ = META_NOCAPTURE;
        }

      /* Do nothing for (* followed by end of pattern or ) so it gives a "bad
      quantifier" error rather than "(*MARK) must have an argument". */

      else if (ptrend - ptr <= 1 || (c = ptr[1]) == CHAR_RIGHT_PARENTHESIS)
        break;

      /* Handle "alpha assertions" such as (*pla:...). Most of these are
      synonyms for the historical symbolic assertions, but the script run and
      non-atomic lookaround ones are new. They are distinguished by starting
      with a lower case letter. Checking both ends of the alphabet makes this
      work in all character codes. */

      else if (CHMAX_255(c) && (cb->ctypes[c] & ctype_lcletter) != 0)
        {
        uint32_t meta;

        vn = alasnames;
        if (!read_name(&ptr, ptrend, utf, 0, &offset, &name, &namelen,
          &errorcode, cb)) goto FAILED;
        if (ptr >= ptrend || *ptr != CHAR_COLON)
          {
          errorcode = ERR95;  /* Malformed */
          goto FAILED;
          }

        /* Scan the table of alpha assertion names */

        for (i = 0; i < alascount; i++)
          {
          if (namelen == alasmeta[i].len &&
              PRIV(strncmp_c8)(name, vn, namelen) == 0)
            break;
          vn += alasmeta[i].len + 1;
          }

        if (i >= alascount)
          {
          errorcode = ERR95;  /* Alpha assertion not recognized */
          goto FAILED;
          }

        /* Check for expecting an assertion condition. If so, only atomic
        lookaround assertions are valid. */

        meta = alasmeta[i].meta;
        if (prev_expect_cond_assert > 0 &&
            (meta < META_LOOKAHEAD || meta > META_LOOKBEHINDNOT))
          {
          errorcode = ERR28;  /* Atomic assertion expected */
          goto FAILED;
          }

        /* The lookaround alphabetic synonyms can mostly be handled by jumping
        to the code that handles the traditional symbolic forms. */

        switch(meta)
          {
          default:
          PCRE2_DEBUG_UNREACHABLE();
          errorcode = ERR89;  /* Unknown code; should never occur because */
          goto FAILED;        /* the meta values come from a table above. */

          case META_ATOMIC:
          goto ATOMIC_GROUP;

          case META_LOOKAHEAD:
          goto POSITIVE_LOOK_AHEAD;

          case META_LOOKAHEAD_NA:
          goto POSITIVE_NONATOMIC_LOOK_AHEAD;

          case META_LOOKAHEADNOT:
          goto NEGATIVE_LOOK_AHEAD;

          case META_SCS:
          if (++ptr >= ptrend) goto UNCLOSED_PARENTHESIS;

          if (*ptr != CHAR_LEFT_PARENTHESIS)
            {
            errorcode = ERR15;
            goto FAILED;
            }

          ptr++;
          *parsed_pattern++ = META_SCS;
          /* Temporary variable, zero in the first iteration. */
          offset = 0;

          for (;;)
            {
            PCRE2_SIZE next_offset = (PCRE2_SIZE)(ptr - cb->start_pattern);

            /* Handle (scan_substring:([+-]number)... */
            if (read_number(&ptr, ptrend, cb->bracount, MAX_GROUP_NUMBER, ERR61,
                &i, &errorcode))
              {
              PCRE2_ASSERT(i >= 0);
              if (i <= 0)
                {
                errorcode = ERR15;
                goto FAILED;
                }
              meta = META_SCS_NUMBER;
              namelen = (uint32_t)i;
              }
            else if (errorcode != 0) goto FAILED;   /* Number too big */
            else
              {
              if (ptr >= ptrend) goto UNCLOSED_PARENTHESIS;

              /* Handle (*scan_substring:('name') or (*scan_substring:(<name>) */
              if (*ptr == CHAR_LESS_THAN_SIGN)
                terminator = CHAR_GREATER_THAN_SIGN;
              else if (*ptr == CHAR_APOSTROPHE)
                terminator = CHAR_APOSTROPHE;
              else
                {
                errorcode = ERR15;
                goto FAILED;
                }

              if (!read_name(&ptr, ptrend, utf, terminator, &next_offset,
                  &name, &namelen, &errorcode, cb)) goto FAILED;

              meta = META_SCS_NAME;
              }

            PCRE2_ASSERT(next_offset > 0);
            if (offset == 0 || (next_offset - offset) >= 0x10000)
              {
              *parsed_pattern++ = META_OFFSET;
              PUTOFFSET(next_offset, parsed_pattern);
              offset = next_offset;
              }

            /* The offset is encoded as a relative offset, because for some
            inputs such as ",2" in (*scs:(1,2,3)...), we only have space for
            two uint32_t values, and an opcode and absolute offset may require
            three uint32_t values. */
            *parsed_pattern++ = meta | (uint32_t)(next_offset - offset);
            *parsed_pattern++ = namelen;
            offset = next_offset;

            if (ptr >= ptrend) goto UNCLOSED_PARENTHESIS;

            if (*ptr == CHAR_RIGHT_PARENTHESIS) break;

            if (*ptr != CHAR_COMMA)
              {
              errorcode = ERR24;
              goto FAILED;
              }

            ptr++;
            }
          ptr++;
          goto POST_ASSERTION;

          case META_LOOKBEHIND:
          case META_LOOKBEHINDNOT:
          case META_LOOKBEHIND_NA:
          *parsed_pattern++ = meta;
          ptr--;
          goto POST_LOOKBEHIND;

          /* The script run facilities are handled here. Unicode support is
          required (give an error if not, as this is a security issue). Always
          record a META_SCRIPT_RUN item. Then, for the atomic version, insert
          META_ATOMIC and remember that we need two META_KETs at the end. */

          case META_SCRIPT_RUN:
          case META_ATOMIC_SCRIPT_RUN:
#ifdef SUPPORT_UNICODE
          *parsed_pattern++ = META_SCRIPT_RUN;
          nest_depth++;
          ptr++;
          if (meta == META_ATOMIC_SCRIPT_RUN)
            {
            *parsed_pattern++ = META_ATOMIC;
            if (top_nest == NULL) top_nest = (nest_save *)(cb->start_workspace);
            else if (++top_nest >= end_nests)
              {
              errorcode = ERR84;
              goto FAILED;
              }
            top_nest->nest_depth = nest_depth;
            top_nest->flags = NSF_ATOMICSR;
            top_nest->options = options & PARSE_TRACKED_OPTIONS;
            top_nest->xoptions = xoptions & PARSE_TRACKED_EXTRA_OPTIONS;

#ifdef PCRE2_DEBUG
            /* We'll write out two META_KETs for a single ")" in the input
            pattern, so we reserve space for that in our bounds check. */
            parsed_pattern_extra++;
#endif
            }
          break;
#else  /* SUPPORT_UNICODE */
          errorcode = ERR96;
          goto FAILED;
#endif
          }
        }


      /* ---- Handle (*VERB) and (*VERB:NAME) ---- */

      else
        {
        vn = verbnames;
        if (!read_name(&ptr, ptrend, utf, 0, &offset, &name, &namelen,
          &errorcode, cb)) goto FAILED;
        if (ptr >= ptrend || (*ptr != CHAR_COLON &&
                              *ptr != CHAR_RIGHT_PARENTHESIS))
          {
          errorcode = ERR60;  /* Malformed */
          goto FAILED;
          }

        /* Scan the table of verb names */

        for (i = 0; i < verbcount; i++)
          {
          if (namelen == verbs[i].len &&
              PRIV(strncmp_c8)(name, vn, namelen) == 0)
            break;
          vn += verbs[i].len + 1;
          }

        if (i >= verbcount)
          {
          errorcode = ERR60;  /* Verb not recognized */
          goto FAILED;
          }

        /* An empty argument is treated as no argument. */

        if (*ptr == CHAR_COLON && ptr + 1 < ptrend &&
             ptr[1] == CHAR_RIGHT_PARENTHESIS)
          ptr++;    /* Advance to the closing parens */

        /* Check for mandatory non-empty argument; this is (*MARK) */

        if (verbs[i].has_arg > 0 && *ptr != CHAR_COLON)
          {
          errorcode = ERR66;
          goto FAILED;
          }

        /* Remember where this verb, possibly with a preceding (*MARK), starts,
        for handling quantified (*ACCEPT). */

        verbstartptr = parsed_pattern;
        okquantifier = (verbs[i].meta == META_ACCEPT);
#ifdef PCRE2_DEBUG
        /* Reserve space in our bounds check for optionally wrapping the (*ACCEPT)
        with a non-capturing bracket, if there is a following quantifier. */
        if (okquantifier) parsed_pattern_extra += 2;
#endif

        /* It appears that Perl allows any characters whatsoever, other than a
        closing parenthesis, to appear in arguments ("names"), so we no longer
        insist on letters, digits, and underscores. Perl does not, however, do
        any interpretation within arguments, and has no means of including a
        closing parenthesis. PCRE supports escape processing but only when it
        is requested by an option. We set inverbname TRUE here, and let the
        main loop take care of this so that escape and \x processing is done by
        the main code above. */

        if (*ptr++ == CHAR_COLON)   /* Skip past : or ) */
          {
          /* Some optional arguments can be treated as a preceding (*MARK) */

          if (verbs[i].has_arg < 0)
            {
            add_after_mark = verbs[i].meta;
            *parsed_pattern++ = META_MARK;
            }

          /* The remaining verbs with arguments (except *MARK) need a different
          opcode. */

          else
            {
            *parsed_pattern++ = verbs[i].meta +
              ((verbs[i].meta != META_MARK)? 0x00010000u:0);
            }

          /* Set up for reading the name in the main loop. */

          verblengthptr = parsed_pattern++;
          verbnamestart = ptr;
          inverbname = TRUE;
          }
        else  /* No verb "name" argument */
          {
          *parsed_pattern++ = verbs[i].meta;
          }
        }     /* End of (*VERB) handling */
      break;  /* Done with this parenthesis */
      }       /* End of groups that don't start with (? */


    /* ---- Items starting (? ---- */

    /* The type of item is determined by what follows (?. Handle (?| and option
    changes under "default" because both need a new block on the nest stack.
    Comments starting with (?# are handled above. Note that there is some
    ambiguity about the sequence (?- because if a digit follows it's a relative
    recursion or subroutine call whereas otherwise it's an option unsetting. */

    if (++ptr >= ptrend) goto UNCLOSED_PARENTHESIS;

    switch(*ptr)
      {
      default:
      if (*ptr == CHAR_MINUS && ptrend - ptr > 1 && IS_DIGIT(ptr[1]))
        goto RECURSION_BYNUMBER;  /* The + case is handled by CHAR_PLUS */

      /* We now have either (?| or a (possibly empty) option setting,
      optionally followed by a non-capturing group. */

      nest_depth++;
      if (top_nest == NULL) top_nest = (nest_save *)(cb->start_workspace);
      else if (++top_nest >= end_nests)
        {
        errorcode = ERR84;
        goto FAILED;
        }
      top_nest->nest_depth = nest_depth;
      top_nest->flags = 0;
      top_nest->options = options & PARSE_TRACKED_OPTIONS;
      top_nest->xoptions = xoptions & PARSE_TRACKED_EXTRA_OPTIONS;

      /* Start of non-capturing group that resets the capture count for each
      branch. */

      if (*ptr == CHAR_VERTICAL_LINE)
        {
        top_nest->reset_group = (uint16_t)cb->bracount;
        top_nest->max_group = (uint16_t)cb->bracount;
        top_nest->flags |= NSF_RESET;
        cb->external_flags |= PCRE2_DUPCAPUSED;
        *parsed_pattern++ = META_NOCAPTURE;
        ptr++;
        }

      /* Scan for options imnrsxJU to be set or unset. */

      else
        {
        BOOL hyphenok = TRUE;
        uint32_t oldoptions = options;
        uint32_t oldxoptions = xoptions;

        top_nest->reset_group = 0;
        top_nest->max_group = 0;
        set = unset = 0;
        optset = &set;
        xset = xunset = 0;
        xoptset = &xset;

        /* ^ at the start unsets irmnsx and disables the subsequent use of - */

        if (ptr < ptrend && *ptr == CHAR_CIRCUMFLEX_ACCENT)
          {
          options &= ~(PCRE2_CASELESS|PCRE2_MULTILINE|PCRE2_NO_AUTO_CAPTURE|
                       PCRE2_DOTALL|PCRE2_EXTENDED|PCRE2_EXTENDED_MORE);
          xoptions &= ~(PCRE2_EXTRA_CASELESS_RESTRICT);
          hyphenok = FALSE;
          ptr++;
          }

        while (ptr < ptrend && *ptr != CHAR_RIGHT_PARENTHESIS &&
                               *ptr != CHAR_COLON)
          {
          switch (*ptr++)
            {
            case CHAR_MINUS:
            if (!hyphenok)
              {
              errorcode = ERR94;
              ptr--;  /* Correct the offset */
              goto FAILED;
              }
            optset = &unset;
            xoptset = &xunset;
            hyphenok = FALSE;
            break;

            /* There are some two-character sequences that start with 'a'. */

            case CHAR_a:
            if (ptr < ptrend)
              {
              if (*ptr == CHAR_D)
                {
                *xoptset |= PCRE2_EXTRA_ASCII_BSD;
                ptr++;
                break;
                }
              if (*ptr == CHAR_P)
                {
                *xoptset |= (PCRE2_EXTRA_ASCII_POSIX|PCRE2_EXTRA_ASCII_DIGIT);
                ptr++;
                break;
                }
              if (*ptr == CHAR_S)
                {
                *xoptset |= PCRE2_EXTRA_ASCII_BSS;
                ptr++;
                break;
                }
              if (*ptr == CHAR_T)
                {
                *xoptset |= PCRE2_EXTRA_ASCII_DIGIT;
                ptr++;
                break;
                }
              if (*ptr == CHAR_W)
                {
                *xoptset |= PCRE2_EXTRA_ASCII_BSW;
                ptr++;
                break;
                }
              }
            *xoptset |= PCRE2_EXTRA_ASCII_BSD|PCRE2_EXTRA_ASCII_BSS|
                        PCRE2_EXTRA_ASCII_BSW|
                        PCRE2_EXTRA_ASCII_DIGIT|PCRE2_EXTRA_ASCII_POSIX;
            break;

            case CHAR_J:  /* Record that it changed in the external options */
            *optset |= PCRE2_DUPNAMES;
            cb->external_flags |= PCRE2_JCHANGED;
            break;

            case CHAR_i: *optset |= PCRE2_CASELESS; break;
            case CHAR_m: *optset |= PCRE2_MULTILINE; break;
            case CHAR_n: *optset |= PCRE2_NO_AUTO_CAPTURE; break;
            case CHAR_r: *xoptset|= PCRE2_EXTRA_CASELESS_RESTRICT; break;
            case CHAR_s: *optset |= PCRE2_DOTALL; break;
            case CHAR_U: *optset |= PCRE2_UNGREEDY; break;

            /* If x appears twice it sets the extended extended option. */

            case CHAR_x:
            *optset |= PCRE2_EXTENDED;
            if (ptr < ptrend && *ptr == CHAR_x)
              {
              *optset |= PCRE2_EXTENDED_MORE;
              ptr++;
              }
            break;

            default:
            errorcode = ERR11;
            ptr--;    /* Correct the offset */
            goto FAILED;
            }
          }

        /* If we are setting extended without extended-more, ensure that any
        existing extended-more gets unset. Also, unsetting extended must also
        unset extended-more. */

        if ((set & (PCRE2_EXTENDED|PCRE2_EXTENDED_MORE)) == PCRE2_EXTENDED ||
            (unset & PCRE2_EXTENDED) != 0)
          unset |= PCRE2_EXTENDED_MORE;

        options = (options | set) & (~unset);
        xoptions = (xoptions | xset) & (~xunset);

        /* If the options ended with ')' this is not the start of a nested
        group with option changes, so the options change at this level.
        In this case, if the previous level set up a nest block, discard the
        one we have just created. Otherwise adjust it for the previous level.
        If the options ended with ':' we are starting a non-capturing group,
        possibly with an options setting. */

        if (ptr >= ptrend) goto UNCLOSED_PARENTHESIS;
        if (*ptr++ == CHAR_RIGHT_PARENTHESIS)
          {
          nest_depth--;  /* This is not a nested group after all. */
          if (top_nest > (nest_save *)(cb->start_workspace) &&
              (top_nest-1)->nest_depth == nest_depth) top_nest--;
          else top_nest->nest_depth = nest_depth;
          }
        else *parsed_pattern++ = META_NOCAPTURE;

        /* If nothing changed, no need to record. */

        if (options != oldoptions || xoptions != oldxoptions)
          {
          *parsed_pattern++ = META_OPTIONS;
          *parsed_pattern++ = options;
          *parsed_pattern++ = xoptions;
          }
        }     /* End options processing */
      break;  /* End default case after (? */


      /* ---- Python syntax support ---- */

      case CHAR_P:
      if (++ptr >= ptrend) goto UNCLOSED_PARENTHESIS;

      /* (?P<name> is the same as (?<name>, which defines a named group. */

      if (*ptr == CHAR_LESS_THAN_SIGN)
        {
        terminator = CHAR_GREATER_THAN_SIGN;
        goto DEFINE_NAME;
        }

      /* (?P>name) is the same as (?&name), which is a recursion or subroutine
      call. */

      if (*ptr == CHAR_GREATER_THAN_SIGN) goto RECURSE_BY_NAME;

      /* (?P=name) is the same as \k<name>, a back reference by name. Anything
      else after (?P is an error. */

      if (*ptr != CHAR_EQUALS_SIGN)
        {
        errorcode = ERR41;
        goto FAILED;
        }
      if (!read_name(&ptr, ptrend, utf, CHAR_RIGHT_PARENTHESIS, &offset, &name,
          &namelen, &errorcode, cb)) goto FAILED;
      *parsed_pattern++ = META_BACKREF_BYNAME;
      *parsed_pattern++ = namelen;
      PUTOFFSET(offset, parsed_pattern);
      okquantifier = TRUE;
      break;   /* End of (?P processing */


      /* ---- Recursion/subroutine calls by number ---- */

      case CHAR_R:
      i = 0;         /* (?R) == (?R0) */
      ptr++;
      if (ptr >= ptrend || *ptr != CHAR_RIGHT_PARENTHESIS)
        {
        errorcode = ERR58;
        goto FAILED;
        }
      goto SET_RECURSION;

      /* An item starting (?- followed by a digit comes here via the "default"
      case because (?- followed by a non-digit is an options setting. */

      case CHAR_PLUS:
      if (ptrend - ptr < 2 || !IS_DIGIT(ptr[1]))
        {
        errorcode = ERR29;   /* Missing number */
        goto FAILED;
        }
      /* Fall through */

      case CHAR_0: case CHAR_1: case CHAR_2: case CHAR_3: case CHAR_4:
      case CHAR_5: case CHAR_6: case CHAR_7: case CHAR_8: case CHAR_9:
      RECURSION_BYNUMBER:
      if (!read_number(&ptr, ptrend,
          (IS_DIGIT(*ptr))? -1:(int)(cb->bracount), /* + and - are relative */
          MAX_GROUP_NUMBER, ERR61,
          &i, &errorcode)) goto FAILED;
      PCRE2_ASSERT(i >= 0);  /* NB (?0) is permitted, represented by i=0 */
      if (ptr >= ptrend || *ptr != CHAR_RIGHT_PARENTHESIS)
        goto UNCLOSED_PARENTHESIS;

      SET_RECURSION:
      *parsed_pattern++ = META_RECURSE | (uint32_t)i;
      offset = (PCRE2_SIZE)(ptr - cb->start_pattern);
      ptr++;
      PUTOFFSET(offset, parsed_pattern);
      okquantifier = TRUE;
      break;  /* End of recursive call by number handling */


      /* ---- Recursion/subroutine calls by name ---- */

      case CHAR_AMPERSAND:
      RECURSE_BY_NAME:
      if (!read_name(&ptr, ptrend, utf, CHAR_RIGHT_PARENTHESIS, &offset, &name,
          &namelen, &errorcode, cb)) goto FAILED;
      *parsed_pattern++ = META_RECURSE_BYNAME;
      *parsed_pattern++ = namelen;
      PUTOFFSET(offset, parsed_pattern);
      okquantifier = TRUE;
      break;

      /* ---- Callout with numerical or string argument ---- */

      case CHAR_C:
      if ((xoptions & PCRE2_EXTRA_NEVER_CALLOUT) != 0)
        {
        errorcode = ERR103;
        goto FAILED;
        }

      if (++ptr >= ptrend) goto UNCLOSED_PARENTHESIS;

      /* If the previous item was a condition starting (?(? an assertion,
      optionally preceded by a callout, is expected. This is checked later on,
      during actual compilation. However we need to identify this kind of
      assertion in this pass because it must not be qualified. The value of
      expect_cond_assert is set to 2 after (?(? is processed. We decrement it
      for a callout - still leaving a positive value that identifies the
      assertion. Multiple callouts or any other items will make it zero or
      less, which doesn't matter because they will cause an error later. */

      expect_cond_assert = prev_expect_cond_assert - 1;

      /* If previous_callout is not NULL, it means this follows a previous
      callout. If it was a manual callout, do nothing; this means its "length
      of next pattern item" field will remain zero. If it was an automatic
      callout, abolish it. */

      if (previous_callout != NULL && (options & PCRE2_AUTO_CALLOUT) != 0 &&
          previous_callout == parsed_pattern - 4 &&
          parsed_pattern[-1] == 255)
        parsed_pattern = previous_callout;

      /* Save for updating next pattern item length, and skip one item before
      completing. */

      previous_callout = parsed_pattern;
      after_manual_callout = 1;

      /* Handle a string argument; specific delimiter is required. */

      if (*ptr != CHAR_RIGHT_PARENTHESIS && !IS_DIGIT(*ptr))
        {
        PCRE2_SIZE calloutlength;
        PCRE2_SPTR startptr = ptr;

        delimiter = 0;
        for (i = 0; PRIV(callout_start_delims)[i] != 0; i++)
          {
          if (*ptr == PRIV(callout_start_delims)[i])
            {
            delimiter = PRIV(callout_end_delims)[i];
            break;
            }
          }
        if (delimiter == 0)
          {
          errorcode = ERR82;
          goto FAILED;
          }

        *parsed_pattern = META_CALLOUT_STRING;
        parsed_pattern += 3;   /* Skip pattern info */

        for (;;)
          {
          if (++ptr >= ptrend)
            {
            errorcode = ERR81;
            ptr = startptr;   /* To give a more useful message */
            goto FAILED;
            }
          if (*ptr == delimiter && (++ptr >= ptrend || *ptr != delimiter))
            break;
          }

        calloutlength = (PCRE2_SIZE)(ptr - startptr);
        if (calloutlength > UINT32_MAX)
          {
          errorcode = ERR72;
          goto FAILED;
          }
        *parsed_pattern++ = (uint32_t)calloutlength;
        offset = (PCRE2_SIZE)(startptr - cb->start_pattern);
        PUTOFFSET(offset, parsed_pattern);
        }

      /* Handle a callout with an optional numerical argument, which must be
      less than or equal to 255. A missing argument gives 0. */

      else
        {
        int n = 0;
        *parsed_pattern = META_CALLOUT_NUMBER;     /* Numerical callout */
        parsed_pattern += 3;                       /* Skip pattern info */
        while (ptr < ptrend && IS_DIGIT(*ptr))
          {
          n = n * 10 + (*ptr++ - CHAR_0);
          if (n > 255)
            {
            errorcode = ERR38;
            goto FAILED;
            }
          }
        *parsed_pattern++ = n;
        }

      /* Both formats must have a closing parenthesis */

      if (ptr >= ptrend || *ptr != CHAR_RIGHT_PARENTHESIS)
        {
        errorcode = ERR39;
        goto FAILED;
        }
      ptr++;

      /* Remember the offset to the next item in the pattern, and set a default
      length. This should get updated after the next item is read. */

      previous_callout[1] = (uint32_t)(ptr - cb->start_pattern);
      previous_callout[2] = 0;
      break;                  /* End callout */


      /* ---- Conditional group ---- */

      /* A condition can be an assertion, a number (referring to a numbered
      group's having been set), a name (referring to a named group), or 'R',
      referring to overall recursion. R<digits> and R&name are also permitted
      for recursion state tests. Numbers may be preceded by + or - to specify a
      relative group number.

      There are several syntaxes for testing a named group: (?(name)) is used
      by Python; Perl 5.10 onwards uses (?(<name>) or (?('name')).

      There are two unfortunate ambiguities. 'R' can be the recursive thing or
      the name 'R' (and similarly for 'R' followed by digits). 'DEFINE' can be
      the Perl DEFINE feature or the Python named test. We look for a name
      first; if not found, we try the other case.

      For compatibility with auto-callouts, we allow a callout to be specified
      before a condition that is an assertion. */

      case CHAR_LEFT_PARENTHESIS:
      if (++ptr >= ptrend) goto UNCLOSED_PARENTHESIS;
      nest_depth++;

      /* If the next character is ? or * there must be an assertion next
      (optionally preceded by a callout). We do not check this here, but
      instead we set expect_cond_assert to 2. If this is still greater than
      zero (callouts decrement it) when the next assertion is read, it will be
      marked as a condition that must not be repeated. A value greater than
      zero also causes checking that an assertion (possibly with callout)
      follows. */

      if (*ptr == CHAR_QUESTION_MARK || *ptr == CHAR_ASTERISK)
        {
        *parsed_pattern++ = META_COND_ASSERT;
        ptr--;   /* Pull pointer back to the opening parenthesis. */
        expect_cond_assert = 2;
        break;  /* End of conditional */
        }

      /* Handle (?([+-]number)... */

      if (read_number(&ptr, ptrend, cb->bracount, MAX_GROUP_NUMBER, ERR61, &i,
          &errorcode))
        {
        PCRE2_ASSERT(i >= 0);
        if (i <= 0)
          {
          errorcode = ERR15;
          goto FAILED;
          }
        *parsed_pattern++ = META_COND_NUMBER;
        offset = (PCRE2_SIZE)(ptr - cb->start_pattern - 2);
        PUTOFFSET(offset, parsed_pattern);
        *parsed_pattern++ = i;
        }
      else if (errorcode != 0) goto FAILED;   /* Number too big */

      /* No number found. Handle the special case (?(VERSION[>]=n.m)... */

      else if (ptrend - ptr >= 10 &&
               PRIV(strncmp_c8)(ptr, STRING_VERSION, 7) == 0 &&
               ptr[7] != CHAR_RIGHT_PARENTHESIS)
        {
        uint32_t ge = 0;
        int major = 0;
        int minor = 0;

        ptr += 7;
        if (*ptr == CHAR_GREATER_THAN_SIGN)
          {
          ge = 1;
          ptr++;
          }

        /* NOTE: cannot write IS_DIGIT(*(++ptr)) here because IS_DIGIT
        references its argument twice. */

        if (*ptr != CHAR_EQUALS_SIGN || (ptr++, !IS_DIGIT(*ptr)))
          goto BAD_VERSION_CONDITION;

        if (!read_number(&ptr, ptrend, -1, 1000, ERR79, &major, &errorcode))
          goto FAILED;

        if (ptr >= ptrend) goto BAD_VERSION_CONDITION;
        if (*ptr == CHAR_DOT)
          {
          if (++ptr >= ptrend || !IS_DIGIT(*ptr)) goto BAD_VERSION_CONDITION;
          minor = (*ptr++ - CHAR_0) * 10;
          if (ptr >= ptrend) goto BAD_VERSION_CONDITION;
          if (IS_DIGIT(*ptr)) minor += *ptr++ - CHAR_0;
          if (ptr >= ptrend || *ptr != CHAR_RIGHT_PARENTHESIS)
            goto BAD_VERSION_CONDITION;
          }

        *parsed_pattern++ = META_COND_VERSION;
        *parsed_pattern++ = ge;
        *parsed_pattern++ = major;
        *parsed_pattern++ = minor;
        }

      /* All the remaining cases now require us to read a name. We cannot at
      this stage distinguish ambiguous cases such as (?(R12) which might be a
      recursion test by number or a name, because the named groups have not yet
      all been identified. Those cases are treated as names, but given a
      different META code. */

      else
        {
        BOOL was_r_ampersand = FALSE;

        if (*ptr == CHAR_R && ptrend - ptr > 1 && ptr[1] == CHAR_AMPERSAND)
          {
          terminator = CHAR_RIGHT_PARENTHESIS;
          was_r_ampersand = TRUE;
          ptr++;
          }
        else if (*ptr == CHAR_LESS_THAN_SIGN)
          terminator = CHAR_GREATER_THAN_SIGN;
        else if (*ptr == CHAR_APOSTROPHE)
          terminator = CHAR_APOSTROPHE;
        else
          {
          terminator = CHAR_RIGHT_PARENTHESIS;
          ptr--;   /* Point to char before name */
          }
        if (!read_name(&ptr, ptrend, utf, terminator, &offset, &name, &namelen,
            &errorcode, cb)) goto FAILED;

        /* Handle (?(R&name) */

        if (was_r_ampersand)
          {
          *parsed_pattern = META_COND_RNAME;
          ptr--;   /* Back to closing parens */
          }

        /* Handle (?(name). If the name is "DEFINE" we identify it with a
        special code. Likewise if the name consists of R followed only by
        digits. Otherwise, handle it like a quoted name. */

        else if (terminator == CHAR_RIGHT_PARENTHESIS)
          {
          if (namelen == 6 && PRIV(strncmp_c8)(name, STRING_DEFINE, 6) == 0)
            *parsed_pattern = META_COND_DEFINE;
          else
            {
            for (i = 1; i < (int)namelen; i++)
              if (!IS_DIGIT(name[i])) break;
            *parsed_pattern = (*name == CHAR_R && i >= (int)namelen)?
              META_COND_RNUMBER : META_COND_NAME;
            }
          ptr--;   /* Back to closing parens */
          }

        /* Handle (?('name') or (?(<name>) */

        else *parsed_pattern = META_COND_NAME;

        /* All these cases except DEFINE end with the name length and offset;
        DEFINE just has an offset (for the "too many branches" error). */

        if (*parsed_pattern++ != META_COND_DEFINE) *parsed_pattern++ = namelen;
        PUTOFFSET(offset, parsed_pattern);
        }  /* End cases that read a name */

      /* Check the closing parenthesis of the condition */

      if (ptr >= ptrend || *ptr != CHAR_RIGHT_PARENTHESIS)
        {
        errorcode = ERR24;
        goto FAILED;
        }
      ptr++;
      break;  /* End of condition processing */


      /* ---- Atomic group ---- */

      case CHAR_GREATER_THAN_SIGN:
      ATOMIC_GROUP:                          /* Come from (*atomic: */
      *parsed_pattern++ = META_ATOMIC;
      nest_depth++;
      ptr++;
      break;


      /* ---- Lookahead assertions ---- */

      case CHAR_EQUALS_SIGN:
      POSITIVE_LOOK_AHEAD:                   /* Come from (*pla: */
      *parsed_pattern++ = META_LOOKAHEAD;
      ptr++;
      goto POST_ASSERTION;

      case CHAR_ASTERISK:
      POSITIVE_NONATOMIC_LOOK_AHEAD:         /* Come from (*napla: */
      *parsed_pattern++ = META_LOOKAHEAD_NA;
      ptr++;
      goto POST_ASSERTION;

      case CHAR_EXCLAMATION_MARK:
      NEGATIVE_LOOK_AHEAD:                   /* Come from (*nla: */
      *parsed_pattern++ = META_LOOKAHEADNOT;
      ptr++;
      goto POST_ASSERTION;


      /* ---- Lookbehind assertions ---- */

      /* (?< followed by = or ! or * is a lookbehind assertion. Otherwise (?<
      is the start of the name of a capturing group. */

      case CHAR_LESS_THAN_SIGN:
      if (ptrend - ptr <= 1 ||
         (ptr[1] != CHAR_EQUALS_SIGN &&
          ptr[1] != CHAR_EXCLAMATION_MARK &&
          ptr[1] != CHAR_ASTERISK))
        {
        terminator = CHAR_GREATER_THAN_SIGN;
        goto DEFINE_NAME;
        }
      *parsed_pattern++ = (ptr[1] == CHAR_EQUALS_SIGN)?
        META_LOOKBEHIND : (ptr[1] == CHAR_EXCLAMATION_MARK)?
        META_LOOKBEHINDNOT : META_LOOKBEHIND_NA;

      POST_LOOKBEHIND:           /* Come from (*plb: (*naplb: and (*nlb: */
      *has_lookbehind = TRUE;
      offset = (PCRE2_SIZE)(ptr - cb->start_pattern - 2);
      PUTOFFSET(offset, parsed_pattern);
      ptr += 2;
      /* Fall through */

      /* If the previous item was a condition starting (?(? an assertion,
      optionally preceded by a callout, is expected. This is checked later on,
      during actual compilation. However we need to identify this kind of
      assertion in this pass because it must not be qualified. The value of
      expect_cond_assert is set to 2 after (?(? is processed. We decrement it
      for a callout - still leaving a positive value that identifies the
      assertion. Multiple callouts or any other items will make it zero or
      less, which doesn't matter because they will cause an error later. */

      POST_ASSERTION:
      nest_depth++;
      if (prev_expect_cond_assert > 0)
        {
        if (top_nest == NULL) top_nest = (nest_save *)(cb->start_workspace);
        else if (++top_nest >= end_nests)
          {
          errorcode = ERR84;
          goto FAILED;
          }
        top_nest->nest_depth = nest_depth;
        top_nest->flags = NSF_CONDASSERT;
        top_nest->options = options & PARSE_TRACKED_OPTIONS;
        top_nest->xoptions = xoptions & PARSE_TRACKED_EXTRA_OPTIONS;
        }
      break;


      /* ---- Define a named group ---- */

      /* A named group may be defined as (?'name') or (?<name>). In the latter
      case we jump to DEFINE_NAME from the disambiguation of (?< above with the
      terminator set to '>'. */

      case CHAR_APOSTROPHE:
      terminator = CHAR_APOSTROPHE;    /* Terminator */

      DEFINE_NAME:
      if (!read_name(&ptr, ptrend, utf, terminator, &offset, &name, &namelen,
          &errorcode, cb)) goto FAILED;

      /* We have a name for this capturing group. It is also assigned a number,
      which is its primary means of identification. */

      if (cb->bracount >= MAX_GROUP_NUMBER)
        {
        errorcode = ERR97;
        goto FAILED;
        }
      cb->bracount++;
      *parsed_pattern++ = META_CAPTURE | cb->bracount;
      nest_depth++;

      /* Check not too many names */

      if (cb->names_found >= MAX_NAME_COUNT)
        {
        errorcode = ERR49;
        goto FAILED;
        }

      /* Adjust the entry size to accommodate the longest name found. */

      if (namelen + IMM2_SIZE + 1 > cb->name_entry_size)
        cb->name_entry_size = (uint16_t)(namelen + IMM2_SIZE + 1);

      /* Scan the list to check for duplicates. For duplicate names, if the
      number is the same, break the loop, which causes the name to be
      discarded; otherwise, if DUPNAMES is not set, give an error.
      If it is set, allow the name with a different number, but continue
      scanning in case this is a duplicate with the same number. For
      non-duplicate names, give an error if the number is duplicated. */

      isdupname = FALSE;
      ng = cb->named_groups;
      for (i = 0; i < cb->names_found; i++, ng++)
        {
        if (namelen == ng->length &&
            PRIV(strncmp)(name, ng->name, (PCRE2_SIZE)namelen) == 0)
          {
          if (ng->number == cb->bracount) break;
          if ((options & PCRE2_DUPNAMES) == 0)
            {
            errorcode = ERR43;
            goto FAILED;
            }
          isdupname = ng->isdup = TRUE;     /* Mark as a duplicate */
          cb->dupnames = TRUE;              /* Duplicate names exist */
          }
        else if (ng->number == cb->bracount)
          {
          errorcode = ERR65;
          goto FAILED;
          }
        }

      if (i < cb->names_found) break;   /* Ignore duplicate with same number */

      /* Increase the list size if necessary */

      if (cb->names_found >= cb->named_group_list_size)
        {
        uint32_t newsize = cb->named_group_list_size * 2;
        named_group *newspace =
          cb->cx->memctl.malloc(newsize * sizeof(named_group),
          cb->cx->memctl.memory_data);
        if (newspace == NULL)
          {
          errorcode = ERR21;
          goto FAILED;
          }

        memcpy(newspace, cb->named_groups,
          cb->named_group_list_size * sizeof(named_group));
        if (cb->named_group_list_size > NAMED_GROUP_LIST_SIZE)
          cb->cx->memctl.free((void *)cb->named_groups,
          cb->cx->memctl.memory_data);
        cb->named_groups = newspace;
        cb->named_group_list_size = newsize;
        }

      /* Add this name to the list */

      cb->named_groups[cb->names_found].name = name;
      cb->named_groups[cb->names_found].length = (uint16_t)namelen;
      cb->named_groups[cb->names_found].number = cb->bracount;
      cb->named_groups[cb->names_found].isdup = (uint16_t)isdupname;
      cb->names_found++;
      break;


      /* ---- Perl extended character class ---- */

      /* These are of the form '(?[...])'. We handle these via the same parser
      that consumes ordinary '[...]' classes, but with a flag set to activate
      the extended behaviour. */

      case CHAR_LEFT_SQUARE_BRACKET:
      class_mode_state = CLASS_MODE_PERL_EXT;
      c = *ptr++;
      goto FROM_PERL_EXTENDED_CLASS;
      }        /* End of (? switch */
    break;     /* End of ( handling */


    /* ---- Branch terminators ---- */

    /* Alternation: reset the capture count if we are in a (?| group. */

    case CHAR_VERTICAL_LINE:
    if (top_nest != NULL && top_nest->nest_depth == nest_depth &&
        (top_nest->flags & NSF_RESET) != 0)
      {
      if (cb->bracount > top_nest->max_group)
        top_nest->max_group = (uint16_t)cb->bracount;
      cb->bracount = top_nest->reset_group;
      }
    *parsed_pattern++ = META_ALT;
    break;

    /* End of group; reset the capture count to the maximum if we are in a (?|
    group and/or reset the options that are tracked during parsing. Disallow
    quantifier for a condition that is an assertion. */

    case CHAR_RIGHT_PARENTHESIS:
    okquantifier = TRUE;
    if (top_nest != NULL && top_nest->nest_depth == nest_depth)
      {
      options = (options & ~PARSE_TRACKED_OPTIONS) | top_nest->options;
      xoptions = (xoptions & ~PARSE_TRACKED_EXTRA_OPTIONS) | top_nest->xoptions;
      if ((top_nest->flags & NSF_RESET) != 0 &&
          top_nest->max_group > cb->bracount)
        cb->bracount = top_nest->max_group;
      if ((top_nest->flags & NSF_CONDASSERT) != 0)
        okquantifier = FALSE;

      if ((top_nest->flags & NSF_ATOMICSR) != 0)
        {
        *parsed_pattern++ = META_KET;

#ifdef PCRE2_DEBUG
        PCRE2_ASSERT(parsed_pattern_extra > 0);
        parsed_pattern_extra--;
#endif
        }

      if (top_nest == (nest_save *)(cb->start_workspace)) top_nest = NULL;
        else top_nest--;
      }
    if (nest_depth == 0)    /* Unmatched closing parenthesis */
      {
      errorcode = ERR22;
      goto FAILED_BACK;  // TODO https://github.com/PCRE2Project/pcre2/issues/549
      }
    nest_depth--;
    *parsed_pattern++ = META_KET;
    break;
    }  /* End of switch on pattern character */
  }    /* End of main character scan loop */

/* End of pattern reached. Check for missing ) at the end of a verb name. */

if (inverbname && ptr >= ptrend)
  {
  errorcode = ERR60;
  goto FAILED;
  }


PARSED_END:

PCRE2_ASSERT((parsed_pattern - parsed_pattern_check) +
             (parsed_pattern_extra - parsed_pattern_extra_check) <=
               max_parsed_pattern(ptr_check, ptr, utf, options));

/* Manage callout for the final item */

parsed_pattern = manage_callouts(ptr, &previous_callout, auto_callout,
  parsed_pattern, cb);

/* Insert trailing items for word and line matching (features provided for the
benefit of pcre2grep). */

if ((xoptions & PCRE2_EXTRA_MATCH_LINE) != 0)
  {
  *parsed_pattern++ = META_KET;
  *parsed_pattern++ = META_DOLLAR;
  }
else if ((xoptions & PCRE2_EXTRA_MATCH_WORD) != 0)
  {
  *parsed_pattern++ = META_KET;
  *parsed_pattern++ = META_ESCAPE + ESC_b;
  }

/* Terminate the parsed pattern, then return success if all groups are closed.
Otherwise we have unclosed parentheses. */

if (parsed_pattern >= parsed_pattern_end)
  {
  PCRE2_DEBUG_UNREACHABLE();
  errorcode = ERR63;  /* Internal error (parsed pattern overflow) */
  goto FAILED;
  }

*parsed_pattern = META_END;
if (nest_depth == 0) return 0;

UNCLOSED_PARENTHESIS:
errorcode = ERR14;

/* Come here for all failures. */

FAILED:
cb->erroroffset = (PCRE2_SIZE)(ptr - cb->start_pattern);
return errorcode;

/* Some errors need to indicate the previous character. */

FAILED_BACK:
ptr--;
goto FAILED;

/* This failure happens several times. */

BAD_VERSION_CONDITION:
errorcode = ERR79;
goto FAILED;
}



/*************************************************
*       Find first significant opcode            *
*************************************************/

/* This is called by several functions that scan a compiled expression looking
for a fixed first character, or an anchoring opcode etc. It skips over things
that do not influence this. For some calls, it makes sense to skip negative
forward and all backward assertions, and also the \b assertion; for others it
does not.

Arguments:
  code         pointer to the start of the group
  skipassert   TRUE if certain assertions are to be skipped

Returns:       pointer to the first significant opcode
*/

static const PCRE2_UCHAR*
first_significant_code(PCRE2_SPTR code, BOOL skipassert)
{
for (;;)
  {
  switch ((int)*code)
    {
    case OP_ASSERT_NOT:
    case OP_ASSERTBACK:
    case OP_ASSERTBACK_NOT:
    case OP_ASSERTBACK_NA:
    if (!skipassert) return code;
    do code += GET(code, 1); while (*code == OP_ALT);
    code += PRIV(OP_lengths)[*code];
    break;

    case OP_WORD_BOUNDARY:
    case OP_NOT_WORD_BOUNDARY:
    case OP_UCP_WORD_BOUNDARY:
    case OP_NOT_UCP_WORD_BOUNDARY:
    if (!skipassert) return code;
    /* Fall through */

    case OP_CALLOUT:
    case OP_CREF:
    case OP_DNCREF:
    case OP_RREF:
    case OP_DNRREF:
    case OP_FALSE:
    case OP_TRUE:
    code += PRIV(OP_lengths)[*code];
    break;

    case OP_CALLOUT_STR:
    code += GET(code, 1 + 2*LINK_SIZE);
    break;

    case OP_SKIPZERO:
    code += 2 + GET(code, 2) + LINK_SIZE;
    break;

    case OP_COND:
    case OP_SCOND:
    if (code[1+LINK_SIZE] != OP_FALSE ||   /* Not DEFINE */
        code[GET(code, 1)] != OP_KET)      /* More than one branch */
      return code;
    code += GET(code, 1) + 1 + LINK_SIZE;
    break;

    case OP_MARK:
    case OP_COMMIT_ARG:
    case OP_PRUNE_ARG:
    case OP_SKIP_ARG:
    case OP_THEN_ARG:
    code += code[1] + PRIV(OP_lengths)[*code];
    break;

    default:
    return code;
    }
  }

PCRE2_DEBUG_UNREACHABLE(); /* Control should never reach here */
}



/*************************************************
*    Find details of duplicate group names       *
*************************************************/

/* This is called from compile_branch() when it needs to know the index and
count of duplicates in the names table when processing named backreferences,
either directly, or as conditions.

Arguments:
  name          points to the name
  length        the length of the name
  indexptr      where to put the index
  countptr      where to put the count of duplicates
  errorcodeptr  where to put an error code
  cb            the compile block

Returns:        TRUE if OK, FALSE if not, error code set
*/

static BOOL
find_dupname_details(PCRE2_SPTR name, uint32_t length, int *indexptr,
  int *countptr, int *errorcodeptr, compile_block *cb)
{
uint32_t i, groupnumber;
int count;
PCRE2_UCHAR *slot = cb->name_table;

/* Find the first entry in the table */

for (i = 0; i < cb->names_found; i++)
  {
  if (PRIV(strncmp)(name, slot+IMM2_SIZE, length) == 0 &&
      slot[IMM2_SIZE+length] == 0) break;
  slot += cb->name_entry_size;
  }

/* This should not occur, because this function is called only when we know we
have duplicate names. Give an internal error. */

if (i >= cb->names_found)
  {
  PCRE2_DEBUG_UNREACHABLE();
  *errorcodeptr = ERR53;
  cb->erroroffset = name - cb->start_pattern;
  return FALSE;
  }

/* Record the index and then see how many duplicates there are, updating the
backref map and maximum back reference as we do. */

*indexptr = i;
count = 0;

for (;;)
  {
  count++;
  groupnumber = GET2(slot,0);
  cb->backref_map |= (groupnumber < 32)? (1u << groupnumber) : 1;
  if (groupnumber > cb->top_backref) cb->top_backref = groupnumber;
  if (++i >= cb->names_found) break;
  slot += cb->name_entry_size;
  if (PRIV(strncmp)(name, slot+IMM2_SIZE, length) != 0 ||
    (slot+IMM2_SIZE)[length] != 0) break;
  }

*countptr = count;
return TRUE;
}



/*************************************************
*           Compile one branch                   *
*************************************************/

/* Scan the parsed pattern, compiling it into the a vector of PCRE2_UCHAR. If
the options are changed during the branch, the pointer is used to change the
external options bits. This function is used during the pre-compile phase when
we are trying to find out the amount of memory needed, as well as during the
real compile phase. The value of lengthptr distinguishes the two phases.

Arguments:
  optionsptr        pointer to the option bits
  xoptionsptr       pointer to the extra option bits
  codeptr           points to the pointer to the current code point
  pptrptr           points to the current parsed pattern pointer
  errorcodeptr      points to error code variable
  firstcuptr        place to put the first required code unit
  firstcuflagsptr   place to put the first code unit flags
  reqcuptr          place to put the last required code unit
  reqcuflagsptr     place to put the last required code unit flags
  bcptr             points to current branch chain
  open_caps         points to current capitem
  cb                contains pointers to tables etc.
  lengthptr         NULL during the real compile phase
                    points to length accumulator during pre-compile phase

Returns:            0 There's been an error, *errorcodeptr is non-zero
                   +1 Success, this branch must match at least one character
                   -1 Success, this branch may match an empty string
*/

static int
compile_branch(uint32_t *optionsptr, uint32_t *xoptionsptr,
  PCRE2_UCHAR **codeptr, uint32_t **pptrptr, int *errorcodeptr,
  uint32_t *firstcuptr, uint32_t *firstcuflagsptr, uint32_t *reqcuptr,
  uint32_t *reqcuflagsptr, branch_chain *bcptr, open_capitem *open_caps,
  compile_block *cb, PCRE2_SIZE *lengthptr)
{
int bravalue = 0;
int okreturn = -1;
int group_return = 0;
uint32_t repeat_min = 0, repeat_max = 0;      /* To please picky compilers */
uint32_t greedy_default, greedy_non_default;
uint32_t repeat_type, op_type;
uint32_t options = *optionsptr;               /* May change dynamically */
uint32_t xoptions = *xoptionsptr;             /* May change dynamically */
uint32_t firstcu, reqcu;
uint32_t zeroreqcu, zerofirstcu;
uint32_t *pptr = *pptrptr;
uint32_t meta, meta_arg;
uint32_t firstcuflags, reqcuflags;
uint32_t zeroreqcuflags, zerofirstcuflags;
uint32_t req_caseopt, reqvary, tempreqvary;
/* Some opcodes, such as META_SCS_NUMBER or META_SCS_NAME,
depends on the previous value of offset. */
PCRE2_SIZE offset = 0;
PCRE2_SIZE length_prevgroup = 0;
PCRE2_UCHAR *code = *codeptr;
PCRE2_UCHAR *last_code = code;
PCRE2_UCHAR *orig_code = code;
PCRE2_UCHAR *tempcode;
PCRE2_UCHAR *previous = NULL;
PCRE2_UCHAR op_previous;
BOOL groupsetfirstcu = FALSE;
BOOL had_accept = FALSE;
BOOL matched_char = FALSE;
BOOL previous_matched_char = FALSE;
BOOL reset_caseful = FALSE;

/* We can fish out the UTF setting once and for all into a BOOL, but we must
not do this for other options (e.g. PCRE2_EXTENDED) that may change dynamically
as we process the pattern. */

#ifdef SUPPORT_UNICODE
BOOL utf = (options & PCRE2_UTF) != 0;
BOOL ucp = (options & PCRE2_UCP) != 0;
#else  /* No Unicode support */
BOOL utf = FALSE;
#endif

/* Set up the default and non-default settings for greediness */

greedy_default = ((options & PCRE2_UNGREEDY) != 0);
greedy_non_default = greedy_default ^ 1;

/* Initialize no first unit, no required unit. REQ_UNSET means "no char
matching encountered yet". It gets changed to REQ_NONE if we hit something that
matches a non-fixed first unit; reqcu just remains unset if we never find one.

When we hit a repeat whose minimum is zero, we may have to adjust these values
to take the zero repeat into account. This is implemented by setting them to
zerofirstcu and zeroreqcu when such a repeat is encountered. The individual
item types that can be repeated set these backoff variables appropriately. */

firstcu = reqcu = zerofirstcu = zeroreqcu = 0;
firstcuflags = reqcuflags = zerofirstcuflags = zeroreqcuflags = REQ_UNSET;

/* The variable req_caseopt contains either the REQ_CASELESS bit or zero,
according to the current setting of the caseless flag. The REQ_CASELESS value
leaves the lower 28 bit empty. It is added into the firstcu or reqcu variables
to record the case status of the value. This is used only for ASCII characters.
*/

req_caseopt = ((options & PCRE2_CASELESS) != 0)? REQ_CASELESS : 0;

/* Switch on next META item until the end of the branch */

for (;; pptr++)
  {
  BOOL possessive_quantifier;
  BOOL note_group_empty;
  uint32_t mclength;
  uint32_t skipunits;
  uint32_t subreqcu, subfirstcu;
  uint32_t groupnumber;
  uint32_t verbarglen, verbculen;
  uint32_t subreqcuflags, subfirstcuflags;
  open_capitem *oc;
  PCRE2_UCHAR mcbuffer[8];

  /* Get next META item in the pattern and its potential argument. */

  meta = META_CODE(*pptr);
  meta_arg = META_DATA(*pptr);

  /* If we are in the pre-compile phase, accumulate the length used for the
  previous cycle of this loop, unless the next item is a quantifier. */

  if (lengthptr != NULL)
    {
    if (code > cb->start_workspace + cb->workspace_size -
        WORK_SIZE_SAFETY_MARGIN)                       /* Check for overrun */
      {
      if (code >= cb->start_workspace + cb->workspace_size)
        {
        PCRE2_DEBUG_UNREACHABLE();
        *errorcodeptr = ERR52;  /* Over-ran workspace - internal error */
        }
      else
        *errorcodeptr = ERR86;
      return 0;
      }

    /* There is at least one situation where code goes backwards: this is the
    case of a zero quantifier after a class (e.g. [ab]{0}). When the quantifier
    is processed, the whole class is eliminated. However, it is created first,
    so we have to allow memory for it. Therefore, don't ever reduce the length
    at this point. */

    if (code < last_code) code = last_code;

    /* If the next thing is not a quantifier, we add the length of the previous
    item into the total, and reset the code pointer to the start of the
    workspace. Otherwise leave the previous item available to be quantified. */

    if (meta < META_ASTERISK || meta > META_MINMAX_QUERY)
      {
      if (OFLOW_MAX - *lengthptr < (PCRE2_SIZE)(code - orig_code))
        {
        *errorcodeptr = ERR20;   /* Integer overflow */
        return 0;
        }
      *lengthptr += (PCRE2_SIZE)(code - orig_code);
      if (*lengthptr > MAX_PATTERN_SIZE)
        {
        *errorcodeptr = ERR20;   /* Pattern is too large */
        return 0;
        }
      code = orig_code;
      }

    /* Remember where this code item starts so we can catch the "backwards"
    case above next time round. */

    last_code = code;
    }

  /* Process the next parsed pattern item. If it is not a quantifier, remember
  where it starts so that it can be quantified when a quantifier follows.
  Checking for the legality of quantifiers happens in parse_regex(), except for
  a quantifier after an assertion that is a condition. */

  if (meta < META_ASTERISK || meta > META_MINMAX_QUERY)
    {
    previous = code;
    if (matched_char && !had_accept) okreturn = 1;
    }

  previous_matched_char = matched_char;
  matched_char = FALSE;
  note_group_empty = FALSE;
  skipunits = 0;         /* Default value for most subgroups */

  switch(meta)
    {
    /* ===================================================================*/
    /* The branch terminates at pattern end or | or ) */

    case META_END:
    case META_ALT:
    case META_KET:
    *firstcuptr = firstcu;
    *firstcuflagsptr = firstcuflags;
    *reqcuptr = reqcu;
    *reqcuflagsptr = reqcuflags;
    *codeptr = code;
    *pptrptr = pptr;
    return okreturn;


    /* ===================================================================*/
    /* Handle single-character metacharacters. In multiline mode, ^ disables
    the setting of any following char as a first character. */

    case META_CIRCUMFLEX:
    if ((options & PCRE2_MULTILINE) != 0)
      {
      if (firstcuflags == REQ_UNSET)
        zerofirstcuflags = firstcuflags = REQ_NONE;
      *code++ = OP_CIRCM;
      }
    else *code++ = OP_CIRC;
    break;

    case META_DOLLAR:
    *code++ = ((options & PCRE2_MULTILINE) != 0)? OP_DOLLM : OP_DOLL;
    break;

    /* There can never be a first char if '.' is first, whatever happens about
    repeats. The value of reqcu doesn't change either. */

    case META_DOT:
    matched_char = TRUE;
    if (firstcuflags == REQ_UNSET) firstcuflags = REQ_NONE;
    zerofirstcu = firstcu;
    zerofirstcuflags = firstcuflags;
    zeroreqcu = reqcu;
    zeroreqcuflags = reqcuflags;
    *code++ = ((options & PCRE2_DOTALL) != 0)? OP_ALLANY: OP_ANY;
    break;


    /* ===================================================================*/
    /* Empty character classes are allowed if PCRE2_ALLOW_EMPTY_CLASS is set.
    Otherwise, an initial ']' is taken as a data character. When empty classes
    are allowed, [] must generate an empty class - we have no dedicated opcode
    to optimise the representation, but it's a rare case (the '(*FAIL)'
    construct would be a clearer way for a pattern author to represent a
    non-matching branch, but it does have different semantics to '[]' if both
    are followed by a quantifier). The empty-negated [^] matches any character,
    so is useful: generate OP_ALLANY for this. */

    case META_CLASS_EMPTY:
    case META_CLASS_EMPTY_NOT:
    matched_char = TRUE;
    if (meta == META_CLASS_EMPTY_NOT) *code++ = OP_ALLANY;
    else
      {
      *code++ = OP_CLASS;
      memset(code, 0, 32);
      code += 32 / sizeof(PCRE2_UCHAR);
      }

    if (firstcuflags == REQ_UNSET) firstcuflags = REQ_NONE;
    zerofirstcu = firstcu;
    zerofirstcuflags = firstcuflags;
    break;


    /* ===================================================================*/
    /* Non-empty character class. If the included characters are all < 256, we
    build a 32-byte bitmap of the permitted characters, except in the special
    case where there is only one such character. For negated classes, we build
    the map as usual, then invert it at the end. However, we use a different
    opcode so that data characters > 255 can be handled correctly.

    If the class contains characters outside the 0-255 range, a different
    opcode is compiled. It may optionally have a bit map for characters < 256,
    but those above are explicitly listed afterwards. A flag code unit tells
    whether the bitmap is present, and whether this is a negated class or
    not. */

    case META_CLASS_NOT:
    case META_CLASS:
    matched_char = TRUE;

    /* Check for complex extended classes and handle them separately. */

    if ((*pptr & CLASS_IS_ECLASS) != 0)
      {
      if (!PRIV(compile_class_nested)(options, xoptions, &pptr, &code,
                                      errorcodeptr, cb, lengthptr))
        return 0;
      goto CLASS_END_PROCESSING;
      }

    /* We can optimize the case of a single character in a class by generating
    OP_CHAR or OP_CHARI if it's positive, or OP_NOT or OP_NOTI if it's
    negative. In the negative case there can be no first char if this item is
    first, whatever repeat count may follow. In the case of reqcu, save the
    previous value for reinstating. */

    /* NOTE: at present this optimization is not effective if the only
    character in a class in 32-bit, non-UCP mode has its top bit set. */

    if (pptr[1] < META_END && pptr[2] == META_CLASS_END)
      {
      uint32_t c = pptr[1];

      pptr += 2;                 /* Move on to class end */
      if (meta == META_CLASS)    /* A positive one-char class can be */
        {                        /* handled as a normal literal character. */
        meta = c;                /* Set up the character */
        goto NORMAL_CHAR_SET;
        }

      /* Handle a negative one-character class */

      zeroreqcu = reqcu;
      zeroreqcuflags = reqcuflags;
      if (firstcuflags == REQ_UNSET) firstcuflags = REQ_NONE;
      zerofirstcu = firstcu;
      zerofirstcuflags = firstcuflags;

      /* For caseless UTF or UCP mode, check whether this character has more
      than one other case. If so, generate a special OP_NOTPROP item instead of
      OP_NOTI. When restricted by PCRE2_EXTRA_CASELESS_RESTRICT, ignore any
      caseless set that starts with an ASCII character. If the character is
      affected by the special Turkish rules, hardcode the not-matching
      characters using a caseset. */

#ifdef SUPPORT_UNICODE
      if ((utf||ucp) && (options & PCRE2_CASELESS) != 0)
        {
        uint32_t caseset;

        if ((xoptions & (PCRE2_EXTRA_TURKISH_CASING|PCRE2_EXTRA_CASELESS_RESTRICT)) ==
              PCRE2_EXTRA_TURKISH_CASING &&
            UCD_ANY_I(c))
          {
          caseset = PRIV(ucd_turkish_dotted_i_caseset) + (UCD_DOTTED_I(c)? 0 : 3);
          }
        else if ((caseset = UCD_CASESET(c)) != 0 &&
                 (xoptions & PCRE2_EXTRA_CASELESS_RESTRICT) != 0 &&
                 PRIV(ucd_caseless_sets)[caseset] < 128)
          {
          caseset = 0;  /* Ignore the caseless set if it's restricted. */
          }

        if (caseset != 0)
          {
          *code++ = OP_NOTPROP;
          *code++ = PT_CLIST;
          *code++ = caseset;
          break;   /* We are finished with this class */
          }
        }
#endif
      /* Char has only one other (usable) case, or UCP not available */

      *code++ = ((options & PCRE2_CASELESS) != 0)? OP_NOTI: OP_NOT;
      code += PUTCHAR(c, code);
      break;   /* We are finished with this class */
      }        /* End of 1-char optimization */

    /* Handle character classes that contain more than just one literal
    character. If there are exactly two characters in a positive class, see if
    they are case partners. This can be optimized to generate a caseless single
    character match (which also sets first/required code units if relevant).
    When casing restrictions apply, ignore a caseless set if both characters
    are ASCII. When Turkish casing applies, an 'i' does not match its normal
    Unicode "othercase". */

    if (meta == META_CLASS && pptr[1] < META_END && pptr[2] < META_END &&
        pptr[3] == META_CLASS_END)
      {
      uint32_t c = pptr[1];

#ifdef SUPPORT_UNICODE
      if ((UCD_CASESET(c) == 0 ||
           ((xoptions & PCRE2_EXTRA_CASELESS_RESTRICT) != 0 &&
            c < 128 && pptr[2] < 128)) &&
          !((xoptions & (PCRE2_EXTRA_TURKISH_CASING|PCRE2_EXTRA_CASELESS_RESTRICT)) ==
              PCRE2_EXTRA_TURKISH_CASING &&
            UCD_ANY_I(c)))
#endif
        {
        uint32_t d;

#ifdef SUPPORT_UNICODE
        if ((utf || ucp) && c > 127) d = UCD_OTHERCASE(c); else
#endif
          {
#if PCRE2_CODE_UNIT_WIDTH != 8
          if (c > 255) d = c; else
#endif
          d = TABLE_GET(c, cb->fcc, c);
          }

        if (c != d && pptr[2] == d)
          {
          pptr += 3;                 /* Move on to class end */
          meta = c;
          if ((options & PCRE2_CASELESS) == 0)
            {
            reset_caseful = TRUE;
            options |= PCRE2_CASELESS;
            req_caseopt = REQ_CASELESS;
            }
          goto CLASS_CASELESS_CHAR;
          }
        }
      }

    /* Now emit the OP_CLASS/OP_NCLASS/OP_XCLASS/OP_ALLANY opcode. */

    pptr = PRIV(compile_class_not_nested)(options, xoptions, pptr + 1,
                                          &code, meta == META_CLASS_NOT, NULL,
                                          errorcodeptr, cb, lengthptr);
    if (pptr == NULL) return 0;
    PCRE2_ASSERT(*pptr == META_CLASS_END);

    CLASS_END_PROCESSING:

    /* If this class is the first thing in the branch, there can be no first
    char setting, whatever the repeat count. Any reqcu setting must remain
    unchanged after any kind of repeat. */

    if (firstcuflags == REQ_UNSET) firstcuflags = REQ_NONE;
    zerofirstcu = firstcu;
    zerofirstcuflags = firstcuflags;
    zeroreqcu = reqcu;
    zeroreqcuflags = reqcuflags;
    break;  /* End of class processing */


    /* ===================================================================*/
    /* Deal with (*VERB)s. */

    /* Check for open captures before ACCEPT and close those that are within
    the same assertion level, also converting ACCEPT to ASSERT_ACCEPT in an
    assertion. In the first pass, just accumulate the length required;
    otherwise hitting (*ACCEPT) inside many nested parentheses can cause
    workspace overflow. Do not set firstcu after *ACCEPT. */

    case META_ACCEPT:
    cb->had_accept = had_accept = TRUE;
    for (oc = open_caps;
         oc != NULL && oc->assert_depth >= cb->assert_depth;
         oc = oc->next)
      {
      if (lengthptr != NULL)
        {
        *lengthptr += CU2BYTES(1) + IMM2_SIZE;
        }
      else
        {
        *code++ = OP_CLOSE;
        PUT2INC(code, 0, oc->number);
        }
      }
    *code++ = (cb->assert_depth > 0)? OP_ASSERT_ACCEPT : OP_ACCEPT;
    if (firstcuflags == REQ_UNSET) firstcuflags = REQ_NONE;
    break;

    case META_PRUNE:
    case META_SKIP:
    cb->had_pruneorskip = TRUE;
    /* Fall through */
    case META_COMMIT:
    case META_FAIL:
    *code++ = verbops[(meta - META_MARK) >> 16];
    break;

    case META_THEN:
    cb->external_flags |= PCRE2_HASTHEN;
    *code++ = OP_THEN;
    break;

    /* Handle verbs with arguments. Arguments can be very long, especially in
    16- and 32-bit modes, and can overflow the workspace in the first pass.
    However, the argument length is constrained to be small enough to fit in
    one code unit. This check happens in parse_regex(). In the first pass,
    instead of putting the argument into memory, we just update the length
    counter and set up an empty argument. */

    case META_THEN_ARG:
    cb->external_flags |= PCRE2_HASTHEN;
    goto VERB_ARG;

    case META_PRUNE_ARG:
    case META_SKIP_ARG:
    cb->had_pruneorskip = TRUE;
    /* Fall through */
    case META_MARK:
    case META_COMMIT_ARG:
    VERB_ARG:
    *code++ = verbops[(meta - META_MARK) >> 16];
    /* The length is in characters. */
    verbarglen = *(++pptr);
    verbculen = 0;
    tempcode = code++;
    for (int i = 0; i < (int)verbarglen; i++)
      {
      meta = *(++pptr);
#ifdef SUPPORT_UNICODE
      if (utf) mclength = PRIV(ord2utf)(meta, mcbuffer); else
#endif
        {
        mclength = 1;
        mcbuffer[0] = meta;
        }
      if (lengthptr != NULL) *lengthptr += mclength; else
        {
        memcpy(code, mcbuffer, CU2BYTES(mclength));
        code += mclength;
        verbculen += mclength;
        }
      }

    *tempcode = verbculen;   /* Fill in the code unit length */
    *code++ = 0;             /* Terminating zero */
    break;


    /* ===================================================================*/
    /* Handle options change. The new setting must be passed back for use in
    subsequent branches. Reset the greedy defaults and the case value for
    firstcu and reqcu. */

    case META_OPTIONS:
    *optionsptr = options = *(++pptr);
    *xoptionsptr = xoptions = *(++pptr);
    greedy_default = ((options & PCRE2_UNGREEDY) != 0);
    greedy_non_default = greedy_default ^ 1;
    req_caseopt = ((options & PCRE2_CASELESS) != 0)? REQ_CASELESS : 0;
    break;

    case META_OFFSET:
    GETPLUSOFFSET(offset, pptr);
    break;

    case META_SCS:
    bravalue = OP_ASSERT_SCS;
    cb->assert_depth += 1;
    goto GROUP_PROCESS;


    /* ===================================================================*/
    /* Handle conditional subpatterns. The case of (?(Rdigits) is ambiguous
    because it could be a numerical check on recursion, or a name check on a
    group's being set. The pre-pass sets up META_COND_RNUMBER as a name so that
    we can handle it either way. We first try for a name; if not found, process
    the number. */

    case META_COND_RNUMBER:   /* (?(Rdigits) */
    case META_COND_NAME:      /* (?(name) or (?'name') or ?(<name>) */
    case META_COND_RNAME:     /* (?(R&name) - test for recursion */
    case META_SCS_NAME:       /* Name of scan substring */
    bravalue = OP_COND;
      {
      int count, index;
      unsigned int i;
      PCRE2_SPTR name;
      named_group *ng = cb->named_groups;
      uint32_t length = *(++pptr);

      if (meta == META_SCS_NAME)
        offset += meta_arg;
      else
        GETPLUSOFFSET(offset, pptr);
      name = cb->start_pattern + offset;

      /* In the first pass, the names generated in the pre-pass are available,
      but the main name table has not yet been created. Scan the list of names
      generated in the pre-pass in order to get a number and whether or not
      this name is duplicated. If it is not duplicated, we can handle it as a
      numerical group. */

      for (i = 0; i < cb->names_found; i++, ng++)
        if (length == ng->length &&
            PRIV(strncmp)(name, ng->name, length) == 0) break;

      if (i >= cb->names_found)
        {
        /* If the name was not found we have a bad reference, unless we are
        dealing with R<digits>, which is treated as a recursion test by
        number. */

        groupnumber = 0;
        if (meta == META_COND_RNUMBER)
          {
          for (i = 1; i < length; i++)
            {
            groupnumber = groupnumber * 10 + (name[i] - CHAR_0);
            if (groupnumber > MAX_GROUP_NUMBER)
              {
              *errorcodeptr = ERR61;
              cb->erroroffset = offset + i;
              return 0;
              }
            }
          }

        if (meta != META_COND_RNUMBER || groupnumber > cb->bracount)
          {
          *errorcodeptr = ERR15;
          cb->erroroffset = offset;
          return 0;
          }

        /* (?Rdigits) treated as a recursion reference by number. A value of
        zero (which is the result of both (?R) and (?R0)) means "any", and is
        translated into RREF_ANY (which is 0xffff). */

        if (groupnumber == 0) groupnumber = RREF_ANY;
        code[1+LINK_SIZE] = OP_RREF;
        PUT2(code, 2+LINK_SIZE, groupnumber);
        skipunits = 1+IMM2_SIZE;
        goto GROUP_PROCESS_NOTE_EMPTY;
        }
      else if (!ng->isdup)
        {
        /* Otherwise found a duplicated name */
        if (ng->number > cb->top_backref) cb->top_backref = ng->number;

        if (meta == META_SCS_NAME)
          {
          code[0] = OP_CREF;
          PUT2(code, 1, ng->number);
          code += 1+IMM2_SIZE;
          break;
          }

        code[1+LINK_SIZE] = (meta == META_COND_RNAME)? OP_RREF : OP_CREF;
        PUT2(code, 2+LINK_SIZE, ng->number);
        skipunits = 1+IMM2_SIZE;
        if (meta != META_SCS_NAME) goto GROUP_PROCESS_NOTE_EMPTY;
        cb->assert_depth += 1;
        goto GROUP_PROCESS;
        }

      /* We have a duplicated name. In the compile pass we have to search the
      main table in order to get the index and count values. */

      count = 0;  /* Values for first pass (avoids compiler warning) */
      index = 0;
      if (lengthptr == NULL && !find_dupname_details(name, length, &index,
            &count, errorcodeptr, cb)) return 0;

      if (meta == META_SCS_NAME)
        {
        code[0] = OP_DNCREF;
        PUT2(code, 1, index);
        PUT2(code, 1+IMM2_SIZE, count);
        code += 1+2*IMM2_SIZE;
        break;
        }

      /* A duplicated name was found. Note that if an R<digits> name is found
      (META_COND_RNUMBER), it is a reference test, not a recursion test. */

      code[1+LINK_SIZE] = (meta == META_COND_RNAME)? OP_DNRREF : OP_DNCREF;

      /* Insert appropriate data values. */
      skipunits = 1+2*IMM2_SIZE;
      PUT2(code, 2+LINK_SIZE, index);
      PUT2(code, 2+LINK_SIZE+IMM2_SIZE, count);
      }

    PCRE2_ASSERT(meta != META_SCS_NAME);
    goto GROUP_PROCESS_NOTE_EMPTY;

    /* The DEFINE condition is always false. Its internal groups may never
    be called, so matched_char must remain false, hence the jump to
    GROUP_PROCESS rather than GROUP_PROCESS_NOTE_EMPTY. */

    case META_COND_DEFINE:
    bravalue = OP_COND;
    GETPLUSOFFSET(offset, pptr);
    code[1+LINK_SIZE] = OP_DEFINE;
    skipunits = 1;
    goto GROUP_PROCESS;

    /* Conditional test of a group's being set. */

    case META_COND_NUMBER:
    case META_SCS_NUMBER:
    bravalue = OP_COND;
    if (meta == META_SCS_NUMBER)
      offset += meta_arg;
    else
      GETPLUSOFFSET(offset, pptr);

    groupnumber = *(++pptr);
    if (groupnumber > cb->bracount)
      {
      *errorcodeptr = ERR15;
      cb->erroroffset = offset;
      return 0;
      }
    if (groupnumber > cb->top_backref) cb->top_backref = groupnumber;

    if (meta == META_SCS_NUMBER)
      {
      code[0] = OP_CREF;
      PUT2(code, 1, groupnumber);
      code += 1+IMM2_SIZE;
      break;
      }

    /* Point at initial ( for too many branches error */
    offset -= 2;
    code[1+LINK_SIZE] = OP_CREF;
    skipunits = 1+IMM2_SIZE;
    PUT2(code, 2+LINK_SIZE, groupnumber);
    goto GROUP_PROCESS_NOTE_EMPTY;

    /* Test for the PCRE2 version. */

    case META_COND_VERSION:
    bravalue = OP_COND;
    if (pptr[1] > 0)
      code[1+LINK_SIZE] = ((PCRE2_MAJOR > pptr[2]) ||
        (PCRE2_MAJOR == pptr[2] && PCRE2_MINOR >= pptr[3]))?
          OP_TRUE : OP_FALSE;
    else
      code[1+LINK_SIZE] = (PCRE2_MAJOR == pptr[2] && PCRE2_MINOR == pptr[3])?
        OP_TRUE : OP_FALSE;
    skipunits = 1;
    pptr += 3;
    goto GROUP_PROCESS_NOTE_EMPTY;

    /* The condition is an assertion, possibly preceded by a callout. */

    case META_COND_ASSERT:
    bravalue = OP_COND;
    goto GROUP_PROCESS_NOTE_EMPTY;


    /* ===================================================================*/
    /* Handle all kinds of nested bracketed groups. The non-capturing,
    non-conditional cases are here; others come to GROUP_PROCESS via goto. */

    case META_LOOKAHEAD:
    bravalue = OP_ASSERT;
    cb->assert_depth += 1;
    goto GROUP_PROCESS;

    case META_LOOKAHEAD_NA:
    bravalue = OP_ASSERT_NA;
    cb->assert_depth += 1;
    goto GROUP_PROCESS;

    /* Optimize (?!) to (*FAIL) unless it is quantified - which is a weird
    thing to do, but Perl allows all assertions to be quantified, and when
    they contain capturing parentheses there may be a potential use for
    this feature. Not that that applies to a quantified (?!) but we allow
    it for uniformity. */

    case META_LOOKAHEADNOT:
    if (pptr[1] == META_KET &&
         (pptr[2] < META_ASTERISK || pptr[2] > META_MINMAX_QUERY))
      {
      *code++ = OP_FAIL;
      pptr++;
      }
    else
      {
      bravalue = OP_ASSERT_NOT;
      cb->assert_depth += 1;
      goto GROUP_PROCESS;
      }
    break;

    case META_LOOKBEHIND:
    bravalue = OP_ASSERTBACK;
    cb->assert_depth += 1;
    goto GROUP_PROCESS;

    case META_LOOKBEHINDNOT:
    bravalue = OP_ASSERTBACK_NOT;
    cb->assert_depth += 1;
    goto GROUP_PROCESS;

    case META_LOOKBEHIND_NA:
    bravalue = OP_ASSERTBACK_NA;
    cb->assert_depth += 1;
    goto GROUP_PROCESS;

    case META_ATOMIC:
    bravalue = OP_ONCE;
    goto GROUP_PROCESS_NOTE_EMPTY;

    case META_SCRIPT_RUN:
    bravalue = OP_SCRIPT_RUN;
    goto GROUP_PROCESS_NOTE_EMPTY;

    case META_NOCAPTURE:
    bravalue = OP_BRA;
    /* Fall through */

    /* Process nested bracketed regex. The nesting depth is maintained for the
    benefit of the stackguard function. The test for too deep nesting is now
    done in parse_regex(). Assertion and DEFINE groups come to GROUP_PROCESS;
    others come to GROUP_PROCESS_NOTE_EMPTY, to indicate that we need to take
    note of whether or not they may match an empty string. */

    GROUP_PROCESS_NOTE_EMPTY:
    note_group_empty = TRUE;

    GROUP_PROCESS:
    cb->parens_depth += 1;
    *code = bravalue;
    pptr++;
    tempcode = code;
    tempreqvary = cb->req_varyopt;        /* Save value before group */
    length_prevgroup = 0;                 /* Initialize for pre-compile phase */

    if ((group_return =
         compile_regex(
         options,                         /* The options state */
         xoptions,                        /* The extra options state */
         &tempcode,                       /* Where to put code (updated) */
         &pptr,                           /* Input pointer (updated) */
         errorcodeptr,                    /* Where to put an error message */
         skipunits,                       /* Skip over bracket number */
         &subfirstcu,                     /* For possible first char */
         &subfirstcuflags,
         &subreqcu,                       /* For possible last char */
         &subreqcuflags,
         bcptr,                           /* Current branch chain */
         open_caps,                       /* Pointer to capture stack */
         cb,                              /* Compile data block */
         (lengthptr == NULL)? NULL :      /* Actual compile phase */
           &length_prevgroup              /* Pre-compile phase */
         )) == 0)
      return 0;  /* Error */

    cb->parens_depth -= 1;

    /* If that was a non-conditional significant group (not an assertion, not a
    DEFINE) that matches at least one character, then the current item matches
    a character. Conditionals are handled below. */

    if (note_group_empty && bravalue != OP_COND && group_return > 0)
      matched_char = TRUE;

    /* If we've just compiled an assertion, pop the assert depth. */

    if (bravalue >= OP_ASSERT && bravalue <= OP_ASSERT_SCS)
      cb->assert_depth -= 1;

    /* At the end of compiling, code is still pointing to the start of the
    group, while tempcode has been updated to point past the end of the group.
    The parsed pattern pointer (pptr) is on the closing META_KET.

    If this is a conditional bracket, check that there are no more than
    two branches in the group, or just one if it's a DEFINE group. We do this
    in the real compile phase, not in the pre-pass, where the whole group may
    not be available. */

    if (bravalue == OP_COND && lengthptr == NULL)
      {
      PCRE2_UCHAR *tc = code;
      int condcount = 0;

      do {
         condcount++;
         tc += GET(tc,1);
         }
      while (*tc != OP_KET);

      /* A DEFINE group is never obeyed inline (the "condition" is always
      false). It must have only one branch. Having checked this, change the
      opcode to OP_FALSE. */

      if (code[LINK_SIZE+1] == OP_DEFINE)
        {
        if (condcount > 1)
          {
          cb->erroroffset = offset;
          *errorcodeptr = ERR54;
          return 0;
          }
        code[LINK_SIZE+1] = OP_FALSE;
        bravalue = OP_DEFINE;   /* A flag to suppress char handling below */
        }

      /* A "normal" conditional group. If there is just one branch, we must not
      make use of its firstcu or reqcu, because this is equivalent to an
      empty second branch. Also, it may match an empty string. If there are two
      branches, this item must match a character if the group must. */

      else
        {
        if (condcount > 2)
          {
          cb->erroroffset = offset;
          *errorcodeptr = ERR27;
          return 0;
          }
        if (condcount == 1) subfirstcuflags = subreqcuflags = REQ_NONE;
          else if (group_return > 0) matched_char = TRUE;
        }
      }

    /* In the pre-compile phase, update the length by the length of the group,
    less the brackets at either end. Then reduce the compiled code to just a
    set of non-capturing brackets so that it doesn't use much memory if it is
    duplicated by a quantifier.*/

    if (lengthptr != NULL)
      {
      if (OFLOW_MAX - *lengthptr < length_prevgroup - 2 - 2*LINK_SIZE)
        {
        *errorcodeptr = ERR20;
        return 0;
        }
      *lengthptr += length_prevgroup - 2 - 2*LINK_SIZE;
      code++;   /* This already contains bravalue */
      PUTINC(code, 0, 1 + LINK_SIZE);
      *code++ = OP_KET;
      PUTINC(code, 0, 1 + LINK_SIZE);
      break;    /* No need to waste time with special character handling */
      }

    /* Otherwise update the main code pointer to the end of the group. */

    code = tempcode;

    /* For a DEFINE group, required and first character settings are not
    relevant. */

    if (bravalue == OP_DEFINE) break;

    /* Handle updating of the required and first code units for other types of
    group. Update for normal brackets of all kinds, and conditions with two
    branches (see code above). If the bracket is followed by a quantifier with
    zero repeat, we have to back off. Hence the definition of zeroreqcu and
    zerofirstcu outside the main loop so that they can be accessed for the back
    off. */

    zeroreqcu = reqcu;
    zeroreqcuflags = reqcuflags;
    zerofirstcu = firstcu;
    zerofirstcuflags = firstcuflags;
    groupsetfirstcu = FALSE;

    if (bravalue >= OP_ONCE)  /* Not an assertion */
      {
      /* If we have not yet set a firstcu in this branch, take it from the
      subpattern, remembering that it was set here so that a repeat of more
      than one can replicate it as reqcu if necessary. If the subpattern has
      no firstcu, set "none" for the whole branch. In both cases, a zero
      repeat forces firstcu to "none". */

      if (firstcuflags == REQ_UNSET && subfirstcuflags != REQ_UNSET)
        {
        if (subfirstcuflags < REQ_NONE)
          {
          firstcu = subfirstcu;
          firstcuflags = subfirstcuflags;
          groupsetfirstcu = TRUE;
          }
        else firstcuflags = REQ_NONE;
        zerofirstcuflags = REQ_NONE;
        }

      /* If firstcu was previously set, convert the subpattern's firstcu
      into reqcu if there wasn't one, using the vary flag that was in
      existence beforehand. */

      else if (subfirstcuflags < REQ_NONE && subreqcuflags >= REQ_NONE)
        {
        subreqcu = subfirstcu;
        subreqcuflags = subfirstcuflags | tempreqvary;
        }

      /* If the subpattern set a required code unit (or set a first code unit
      that isn't really the first code unit - see above), set it. */

      if (subreqcuflags < REQ_NONE)
        {
        reqcu = subreqcu;
        reqcuflags = subreqcuflags;
        }
      }

    /* For a forward assertion, we take the reqcu, if set, provided that the
    group has also set a firstcu. This can be helpful if the pattern that
    follows the assertion doesn't set a different char. For example, it's
    useful for /(?=abcde).+/. We can't set firstcu for an assertion, however
    because it leads to incorrect effect for patterns such as /(?=a)a.+/ when
    the "real" "a" would then become a reqcu instead of a firstcu. This is
    overcome by a scan at the end if there's no firstcu, looking for an
    asserted first char. A similar effect for patterns like /(?=.*X)X$/ means
    we must only take the reqcu when the group also set a firstcu. Otherwise,
    in that example, 'X' ends up set for both. */

    else if ((bravalue == OP_ASSERT || bravalue == OP_ASSERT_NA) &&
             subreqcuflags < REQ_NONE && subfirstcuflags < REQ_NONE)
      {
      reqcu = subreqcu;
      reqcuflags = subreqcuflags;
      }

    break;  /* End of nested group handling */


    /* ===================================================================*/
    /* Handle named backreferences and recursions. */

    case META_BACKREF_BYNAME:
    case META_RECURSE_BYNAME:
      {
      int count, index;
      PCRE2_SPTR name;
      BOOL is_dupname = FALSE;
      named_group *ng = cb->named_groups;
      uint32_t length = *(++pptr);

      GETPLUSOFFSET(offset, pptr);
      name = cb->start_pattern + offset;

      /* In the first pass, the names generated in the pre-pass are available,
      but the main name table has not yet been created. Scan the list of names
      generated in the pre-pass in order to get a number and whether or not
      this name is duplicated. */

      groupnumber = 0;
      for (unsigned int i = 0; i < cb->names_found; i++, ng++)
        {
        if (length == ng->length &&
            PRIV(strncmp)(name, ng->name, length) == 0)
          {
          is_dupname = ng->isdup;
          groupnumber = ng->number;

          /* For a recursion, that's all that is needed. We can now go to
          the code that handles numerical recursion, applying it to the first
          group with the given name. */

          if (meta == META_RECURSE_BYNAME)
            {
            meta_arg = groupnumber;
            goto HANDLE_NUMERICAL_RECURSION;
            }

          /* For a back reference, update the back reference map and the
          maximum back reference. */

          cb->backref_map |= (groupnumber < 32)? (1u << groupnumber) : 1;
          if (groupnumber > cb->top_backref)
            cb->top_backref = groupnumber;
          }
        }

      /* If the name was not found we have a bad reference. */

      if (groupnumber == 0)
        {
        *errorcodeptr = ERR15;
        cb->erroroffset = offset;
        return 0;
        }

      /* If a back reference name is not duplicated, we can handle it as
      a numerical reference. */

      if (!is_dupname)
        {
        meta_arg = groupnumber;
        goto HANDLE_SINGLE_REFERENCE;
        }

      /* If a back reference name is duplicated, we generate a different
      opcode to a numerical back reference. In the second pass we must
      search for the index and count in the final name table. */

      count = 0;  /* Values for first pass (avoids compiler warning) */
      index = 0;
      if (lengthptr == NULL && !find_dupname_details(name, length, &index,
            &count, errorcodeptr, cb)) return 0;

      if (firstcuflags == REQ_UNSET) firstcuflags = REQ_NONE;
      *code++ = ((options & PCRE2_CASELESS) != 0)? OP_DNREFI : OP_DNREF;
      PUT2INC(code, 0, index);
      PUT2INC(code, 0, count);
      if ((options & PCRE2_CASELESS) != 0)
        *code++ = (((xoptions & PCRE2_EXTRA_CASELESS_RESTRICT) != 0)?
                   REFI_FLAG_CASELESS_RESTRICT : 0) |
                  (((xoptions & PCRE2_EXTRA_TURKISH_CASING) != 0)?
                   REFI_FLAG_TURKISH_CASING : 0);
      }
    break;


    /* ===================================================================*/
    /* Handle a numerical callout. */

    case META_CALLOUT_NUMBER:
    code[0] = OP_CALLOUT;
    PUT(code, 1, pptr[1]);               /* Offset to next pattern item */
    PUT(code, 1 + LINK_SIZE, pptr[2]);   /* Length of next pattern item */
    code[1 + 2*LINK_SIZE] = pptr[3];
    pptr += 3;
    code += PRIV(OP_lengths)[OP_CALLOUT];
    break;


    /* ===================================================================*/
    /* Handle a callout with a string argument. In the pre-pass we just compute
    the length without generating anything. The length in pptr[3] includes both
    delimiters; in the actual compile only the first one is copied, but a
    terminating zero is added. Any doubled delimiters within the string make
    this an overestimate, but it is not worth bothering about. */

    case META_CALLOUT_STRING:
    if (lengthptr != NULL)
      {
      *lengthptr += pptr[3] + (1 + 4*LINK_SIZE);
      pptr += 3;
      SKIPOFFSET(pptr);
      }

    /* In the real compile we can copy the string. The starting delimiter is
     included so that the client can discover it if they want. We also pass the
     start offset to help a script language give better error messages. */

    else
      {
      PCRE2_SPTR pp;
      uint32_t delimiter;
      uint32_t length = pptr[3];
      PCRE2_UCHAR *callout_string = code + (1 + 4*LINK_SIZE);

      code[0] = OP_CALLOUT_STR;
      PUT(code, 1, pptr[1]);               /* Offset to next pattern item */
      PUT(code, 1 + LINK_SIZE, pptr[2]);   /* Length of next pattern item */

      pptr += 3;
      GETPLUSOFFSET(offset, pptr);         /* Offset to string in pattern */
      pp = cb->start_pattern + offset;
      delimiter = *callout_string++ = *pp++;
      if (delimiter == CHAR_LEFT_CURLY_BRACKET)
        delimiter = CHAR_RIGHT_CURLY_BRACKET;
      PUT(code, 1 + 3*LINK_SIZE, (int)(offset + 1));  /* One after delimiter */

      /* The syntax of the pattern was checked in the parsing scan. The length
      includes both delimiters, but we have passed the opening one just above,
      so we reduce length before testing it. The test is for > 1 because we do
      not want to copy the final delimiter. This also ensures that pp[1] is
      accessible. */

      while (--length > 1)
        {
        if (*pp == delimiter && pp[1] == delimiter)
          {
          *callout_string++ = delimiter;
          pp += 2;
          length--;
          }
        else *callout_string++ = *pp++;
        }
      *callout_string++ = CHAR_NUL;

      /* Set the length of the entire item, the advance to its end. */

      PUT(code, 1 + 2*LINK_SIZE, (int)(callout_string - code));
      code = callout_string;
      }
    break;


    /* ===================================================================*/
    /* Handle repetition. The different types are all sorted out in the parsing
    pass. */

    case META_MINMAX_PLUS:
    case META_MINMAX_QUERY:
    case META_MINMAX:
    repeat_min = *(++pptr);
    repeat_max = *(++pptr);
    goto REPEAT;

    case META_ASTERISK:
    case META_ASTERISK_PLUS:
    case META_ASTERISK_QUERY:
    repeat_min = 0;
    repeat_max = REPEAT_UNLIMITED;
    goto REPEAT;

    case META_PLUS:
    case META_PLUS_PLUS:
    case META_PLUS_QUERY:
    repeat_min = 1;
    repeat_max = REPEAT_UNLIMITED;
    goto REPEAT;

    case META_QUERY:
    case META_QUERY_PLUS:
    case META_QUERY_QUERY:
    repeat_min = 0;
    repeat_max = 1;

    REPEAT:
    if (previous_matched_char && repeat_min > 0) matched_char = TRUE;

    /* Remember whether this is a variable length repeat, and default to
    single-char opcodes. */

    reqvary = (repeat_min == repeat_max)? 0 : REQ_VARY;

    /* Adjust first and required code units for a zero repeat. */

    if (repeat_min == 0)
      {
      firstcu = zerofirstcu;
      firstcuflags = zerofirstcuflags;
      reqcu = zeroreqcu;
      reqcuflags = zeroreqcuflags;
      }

    /* Note the greediness and possessiveness. */

    switch (meta)
      {
      case META_MINMAX_PLUS:
      case META_ASTERISK_PLUS:
      case META_PLUS_PLUS:
      case META_QUERY_PLUS:
      repeat_type = 0;                  /* Force greedy */
      possessive_quantifier = TRUE;
      break;

      case META_MINMAX_QUERY:
      case META_ASTERISK_QUERY:
      case META_PLUS_QUERY:
      case META_QUERY_QUERY:
      repeat_type = greedy_non_default;
      possessive_quantifier = FALSE;
      break;

      default:
      repeat_type = greedy_default;
      possessive_quantifier = FALSE;
      break;
      }

    /* Save start of previous item, in case we have to move it up in order to
    insert something before it, and remember what it was. */

    PCRE2_ASSERT(previous != NULL);
    tempcode = previous;
    op_previous = *previous;

    /* Now handle repetition for the different types of item. If the repeat
    minimum and the repeat maximum are both 1, we can ignore the quantifier for
    non-parenthesized items, as they have only one alternative. For anything in
    parentheses, we must not ignore if {1} is possessive. */

    switch (op_previous)
      {
      /* If previous was a character or negated character match, abolish the
      item and generate a repeat item instead. If a char item has a minimum of
      more than one, ensure that it is set in reqcu - it might not be if a
      sequence such as x{3} is the first thing in a branch because the x will
      have gone into firstcu instead.  */

      case OP_CHAR:
      case OP_CHARI:
      case OP_NOT:
      case OP_NOTI:
      if (repeat_max == 1 && repeat_min == 1) goto END_REPEAT;
      op_type = chartypeoffset[op_previous - OP_CHAR];

      /* Deal with UTF characters that take up more than one code unit. */

#ifdef MAYBE_UTF_MULTI
      if (utf && NOT_FIRSTCU(code[-1]))
        {
        PCRE2_UCHAR *lastchar = code - 1;
        BACKCHAR(lastchar);
        mclength = (uint32_t)(code - lastchar);   /* Length of UTF character */
        memcpy(mcbuffer, lastchar, CU2BYTES(mclength));  /* Save the char */
        }
      else
#endif  /* MAYBE_UTF_MULTI */

      /* Handle the case of a single code unit - either with no UTF support, or
      with UTF disabled, or for a single-code-unit UTF character. In the latter
      case, for a repeated positive match, get the caseless flag for the
      required code unit from the previous character, because a class like [Aa]
      sets a caseless A but by now the req_caseopt flag has been reset. */

        {
        mcbuffer[0] = code[-1];
        mclength = 1;
        if (op_previous <= OP_CHARI && repeat_min > 1)
          {
          reqcu = mcbuffer[0];
          reqcuflags = cb->req_varyopt;
          if (op_previous == OP_CHARI) reqcuflags |= REQ_CASELESS;
          }
        }
      goto OUTPUT_SINGLE_REPEAT;  /* Code shared with single character types */

      /* If previous was a character class or a back reference, we put the
      repeat stuff after it, but just skip the item if the repeat was {0,0}. */

#ifdef SUPPORT_WIDE_CHARS
      case OP_XCLASS:
      case OP_ECLASS:
#endif
      case OP_CLASS:
      case OP_NCLASS:
      case OP_REF:
      case OP_REFI:
      case OP_DNREF:
      case OP_DNREFI:

      if (repeat_max == 0)
        {
        code = previous;
        goto END_REPEAT;
        }
      if (repeat_max == 1 && repeat_min == 1) goto END_REPEAT;

      if (repeat_min == 0 && repeat_max == REPEAT_UNLIMITED)
        *code++ = OP_CRSTAR + repeat_type;
      else if (repeat_min == 1 && repeat_max == REPEAT_UNLIMITED)
        *code++ = OP_CRPLUS + repeat_type;
      else if (repeat_min == 0 && repeat_max == 1)
        *code++ = OP_CRQUERY + repeat_type;
      else
        {
        *code++ = OP_CRRANGE + repeat_type;
        PUT2INC(code, 0, repeat_min);
        if (repeat_max == REPEAT_UNLIMITED) repeat_max = 0;  /* 2-byte encoding for max */
        PUT2INC(code, 0, repeat_max);
        }
      break;

      /* Prior to 10.30, repeated recursions were wrapped in OP_ONCE brackets
      because pcre2_match() could not handle backtracking into recursively
      called groups. Now that this backtracking is available, we no longer need
      to do this. However, we still need to replicate recursions as we do for
      groups so as to have independent backtracking points. We can replicate
      for the minimum number of repeats directly. For optional repeats we now
      wrap the recursion in OP_BRA brackets and make use of the bracket
      repetition. */

      case OP_RECURSE:
      if (repeat_max == 1 && repeat_min == 1 && !possessive_quantifier)
        goto END_REPEAT;

      /* Generate unwrapped repeats for a non-zero minimum, except when the
      minimum is 1 and the maximum unlimited, because that can be handled with
      OP_BRA terminated by OP_KETRMAX/MIN. When the maximum is equal to the
      minimum, we just need to generate the appropriate additional copies.
      Otherwise we need to generate one more, to simulate the situation when
      the minimum is zero. */

      if (repeat_min > 0 && (repeat_min != 1 || repeat_max != REPEAT_UNLIMITED))
        {
        int replicate = repeat_min;
        if (repeat_min == repeat_max) replicate--;

        /* In the pre-compile phase, we don't actually do the replication. We
        just adjust the length as if we had. Do some paranoid checks for
        potential integer overflow. */

        if (lengthptr != NULL)
          {
          PCRE2_SIZE delta;
          if (PRIV(ckd_smul)(&delta, replicate, 1 + LINK_SIZE) ||
              OFLOW_MAX - *lengthptr < delta)
            {
            *errorcodeptr = ERR20;
            return 0;
            }
          *lengthptr += delta;
          }

        else for (int i = 0; i < replicate; i++)
          {
          memcpy(code, previous, CU2BYTES(1 + LINK_SIZE));
          previous = code;
          code += 1 + LINK_SIZE;
          }

        /* If the number of repeats is fixed, we are done. Otherwise, adjust
        the counts and fall through. */

        if (repeat_min == repeat_max) break;
        if (repeat_max != REPEAT_UNLIMITED) repeat_max -= repeat_min;
        repeat_min = 0;
        }

      /* Wrap the recursion call in OP_BRA brackets. */

      (void)memmove(previous + 1 + LINK_SIZE, previous, CU2BYTES(1 + LINK_SIZE));
      op_previous = *previous = OP_BRA;
      PUT(previous, 1, 2 + 2*LINK_SIZE);
      previous[2 + 2*LINK_SIZE] = OP_KET;
      PUT(previous, 3 + 2*LINK_SIZE, 2 + 2*LINK_SIZE);
      code += 2 + 2 * LINK_SIZE;
      length_prevgroup = 3 + 3*LINK_SIZE;
      group_return = -1;  /* Set "may match empty string" */

      /* Now treat as a repeated OP_BRA. */
      /* Fall through */

      /* If previous was a bracket group, we may have to replicate it in
      certain cases. Note that at this point we can encounter only the "basic"
      bracket opcodes such as BRA and CBRA, as this is the place where they get
      converted into the more special varieties such as BRAPOS and SBRA.
      Originally, PCRE did not allow repetition of assertions, but now it does,
      for Perl compatibility. */

      case OP_ASSERT:
      case OP_ASSERT_NOT:
      case OP_ASSERT_NA:
      case OP_ASSERTBACK:
      case OP_ASSERTBACK_NOT:
      case OP_ASSERTBACK_NA:
      case OP_ASSERT_SCS:
      case OP_ONCE:
      case OP_SCRIPT_RUN:
      case OP_BRA:
      case OP_CBRA:
      case OP_COND:
        {
        int len = (int)(code - previous);
        PCRE2_UCHAR *bralink = NULL;
        PCRE2_UCHAR *brazeroptr = NULL;

        if (repeat_max == 1 && repeat_min == 1 && !possessive_quantifier)
          goto END_REPEAT;

        /* Repeating a DEFINE group (or any group where the condition is always
        FALSE and there is only one branch) is pointless, but Perl allows the
        syntax, so we just ignore the repeat. */

        if (op_previous == OP_COND && previous[LINK_SIZE+1] == OP_FALSE &&
            previous[GET(previous, 1)] != OP_ALT)
          goto END_REPEAT;

        /* Perl allows all assertions to be quantified, and when they contain
        capturing parentheses and/or are optional there are potential uses for
        this feature. PCRE2 used to force the maximum quantifier to 1 on the
        invalid grounds that further repetition was never useful. This was
        always a bit pointless, since an assertion could be wrapped with a
        repeated group to achieve the effect. General repetition is now
        permitted, but if the maximum is unlimited it is set to one more than
        the minimum. */

        if (op_previous < OP_ONCE)    /* Assertion */
          {
          if (repeat_max == REPEAT_UNLIMITED) repeat_max = repeat_min + 1;
          }

        /* The case of a zero minimum is special because of the need to stick
        OP_BRAZERO in front of it, and because the group appears once in the
        data, whereas in other cases it appears the minimum number of times. For
        this reason, it is simplest to treat this case separately, as otherwise
        the code gets far too messy. There are several special subcases when the
        minimum is zero. */

        if (repeat_min == 0)
          {
          /* If the maximum is also zero, we used to just omit the group from
          the output altogether, like this:

          ** if (repeat_max == 0)
          **   {
          **   code = previous;
          **   goto END_REPEAT;
          **   }

          However, that fails when a group or a subgroup within it is
          referenced as a subroutine from elsewhere in the pattern, so now we
          stick in OP_SKIPZERO in front of it so that it is skipped on
          execution. As we don't have a list of which groups are referenced, we
          cannot do this selectively.

          If the maximum is 1 or unlimited, we just have to stick in the
          BRAZERO and do no more at this point. */

          if (repeat_max <= 1 || repeat_max == REPEAT_UNLIMITED)
            {
            (void)memmove(previous + 1, previous, CU2BYTES(len));
            code++;
            if (repeat_max == 0)
              {
              *previous++ = OP_SKIPZERO;
              goto END_REPEAT;
              }
            brazeroptr = previous;    /* Save for possessive optimizing */
            *previous++ = OP_BRAZERO + repeat_type;
            }

          /* If the maximum is greater than 1 and limited, we have to replicate
          in a nested fashion, sticking OP_BRAZERO before each set of brackets.
          The first one has to be handled carefully because it's the original
          copy, which has to be moved up. The remainder can be handled by code
          that is common with the non-zero minimum case below. We have to
          adjust the value or repeat_max, since one less copy is required. */

          else
            {
            int linkoffset;
            (void)memmove(previous + 2 + LINK_SIZE, previous, CU2BYTES(len));
            code += 2 + LINK_SIZE;
            *previous++ = OP_BRAZERO + repeat_type;
            *previous++ = OP_BRA;

            /* We chain together the bracket link offset fields that have to be
            filled in later when the ends of the brackets are reached. */

            linkoffset = (bralink == NULL)? 0 : (int)(previous - bralink);
            bralink = previous;
            PUTINC(previous, 0, linkoffset);
            }

          if (repeat_max != REPEAT_UNLIMITED) repeat_max--;
          }

        /* If the minimum is greater than zero, replicate the group as many
        times as necessary, and adjust the maximum to the number of subsequent
        copies that we need. */

        else
          {
          if (repeat_min > 1)
            {
            /* In the pre-compile phase, we don't actually do the replication.
            We just adjust the length as if we had. Do some paranoid checks for
            potential integer overflow. */

            if (lengthptr != NULL)
              {
              PCRE2_SIZE delta;
              if (PRIV(ckd_smul)(&delta, repeat_min - 1,
                                 (int)length_prevgroup) ||
                  OFLOW_MAX - *lengthptr < delta)
                {
                *errorcodeptr = ERR20;
                return 0;
                }
              *lengthptr += delta;
              }

            /* This is compiling for real. If there is a set first code unit
            for the group, and we have not yet set a "required code unit", set
            it. */

            else
              {
              if (groupsetfirstcu && reqcuflags >= REQ_NONE)
                {
                reqcu = firstcu;
                reqcuflags = firstcuflags;
                }
              for (uint32_t i = 1; i < repeat_min; i++)
                {
                memcpy(code, previous, CU2BYTES(len));
                code += len;
                }
              }
            }

          if (repeat_max != REPEAT_UNLIMITED) repeat_max -= repeat_min;
          }

        /* This code is common to both the zero and non-zero minimum cases. If
        the maximum is limited, it replicates the group in a nested fashion,
        remembering the bracket starts on a stack. In the case of a zero
        minimum, the first one was set up above. In all cases the repeat_max
        now specifies the number of additional copies needed. Again, we must
        remember to replicate entries on the forward reference list. */

        if (repeat_max != REPEAT_UNLIMITED)
          {
          /* In the pre-compile phase, we don't actually do the replication. We
          just adjust the length as if we had. For each repetition we must add
          1 to the length for BRAZERO and for all but the last repetition we
          must add 2 + 2*LINKSIZE to allow for the nesting that occurs. Do some
          paranoid checks to avoid integer overflow. */

          if (lengthptr != NULL && repeat_max > 0)
            {
            PCRE2_SIZE delta;
            if (PRIV(ckd_smul)(&delta, repeat_max,
                               (int)length_prevgroup + 1 + 2 + 2*LINK_SIZE) ||
                OFLOW_MAX + (2 + 2*LINK_SIZE) - *lengthptr < delta)
              {
              *errorcodeptr = ERR20;
              return 0;
              }
            delta -= (2 + 2*LINK_SIZE);   /* Last one doesn't nest */
            *lengthptr += delta;
            }

          /* This is compiling for real */

          else for (uint32_t i = repeat_max; i >= 1; i--)
            {
            *code++ = OP_BRAZERO + repeat_type;

            /* All but the final copy start a new nesting, maintaining the
            chain of brackets outstanding. */

            if (i != 1)
              {
              int linkoffset;
              *code++ = OP_BRA;
              linkoffset = (bralink == NULL)? 0 : (int)(code - bralink);
              bralink = code;
              PUTINC(code, 0, linkoffset);
              }

            memcpy(code, previous, CU2BYTES(len));
            code += len;
            }

          /* Now chain through the pending brackets, and fill in their length
          fields (which are holding the chain links pro tem). */

          while (bralink != NULL)
            {
            int oldlinkoffset;
            int linkoffset = (int)(code - bralink + 1);
            PCRE2_UCHAR *bra = code - linkoffset;
            oldlinkoffset = GET(bra, 1);
            bralink = (oldlinkoffset == 0)? NULL : bralink - oldlinkoffset;
            *code++ = OP_KET;
            PUTINC(code, 0, linkoffset);
            PUT(bra, 1, linkoffset);
            }
          }

        /* If the maximum is unlimited, set a repeater in the final copy. For
        SCRIPT_RUN and ONCE brackets, that's all we need to do. However,
        possessively repeated ONCE brackets can be converted into non-capturing
        brackets, as the behaviour of (?:xx)++ is the same as (?>xx)++ and this
        saves having to deal with possessive ONCEs specially.

        Otherwise, when we are doing the actual compile phase, check to see
        whether this group is one that could match an empty string. If so,
        convert the initial operator to the S form (e.g. OP_BRA -> OP_SBRA) so
        that runtime checking can be done. [This check is also applied to ONCE
        and SCRIPT_RUN groups at runtime, but in a different way.]

        Then, if the quantifier was possessive and the bracket is not a
        conditional, we convert the BRA code to the POS form, and the KET code
        to KETRPOS. (It turns out to be convenient at runtime to detect this
        kind of subpattern at both the start and at the end.) The use of
        special opcodes makes it possible to reduce greatly the stack usage in
        pcre2_match(). If the group is preceded by OP_BRAZERO, convert this to
        OP_BRAPOSZERO.

        Then, if the minimum number of matches is 1 or 0, cancel the possessive
        flag so that the default action below, of wrapping everything inside
        atomic brackets, does not happen. When the minimum is greater than 1,
        there will be earlier copies of the group, and so we still have to wrap
        the whole thing. */

        else
          {
          PCRE2_UCHAR *ketcode = code - 1 - LINK_SIZE;
          PCRE2_UCHAR *bracode = ketcode - GET(ketcode, 1);

          /* Convert possessive ONCE brackets to non-capturing */

          if (*bracode == OP_ONCE && possessive_quantifier) *bracode = OP_BRA;

          /* For non-possessive ONCE and for SCRIPT_RUN brackets, all we need
          to do is to set the KET. */

          if (*bracode == OP_ONCE || *bracode == OP_SCRIPT_RUN)
            *ketcode = OP_KETRMAX + repeat_type;

          /* Handle non-SCRIPT_RUN and non-ONCE brackets and possessive ONCEs
          (which have been converted to non-capturing above). */

          else
            {
            /* In the compile phase, adjust the opcode if the group can match
            an empty string. For a conditional group with only one branch, the
            value of group_return will not show "could be empty", so we must
            check that separately. */

            if (lengthptr == NULL)
              {
              if (group_return < 0) *bracode += OP_SBRA - OP_BRA;
              if (*bracode == OP_COND && bracode[GET(bracode,1)] != OP_ALT)
                *bracode = OP_SCOND;
              }

            /* Handle possessive quantifiers. */

            if (possessive_quantifier)
              {
              /* For COND brackets, we wrap the whole thing in a possessively
              repeated non-capturing bracket, because we have not invented POS
              versions of the COND opcodes. */

              if (*bracode == OP_COND || *bracode == OP_SCOND)
                {
                int nlen = (int)(code - bracode);
                (void)memmove(bracode + 1 + LINK_SIZE, bracode, CU2BYTES(nlen));
                code += 1 + LINK_SIZE;
                nlen += 1 + LINK_SIZE;
                *bracode = (*bracode == OP_COND)? OP_BRAPOS : OP_SBRAPOS;
                *code++ = OP_KETRPOS;
                PUTINC(code, 0, nlen);
                PUT(bracode, 1, nlen);
                }

              /* For non-COND brackets, we modify the BRA code and use KETRPOS. */

              else
                {
                *bracode += 1;              /* Switch to xxxPOS opcodes */
                *ketcode = OP_KETRPOS;
                }

              /* If the minimum is zero, mark it as possessive, then unset the
              possessive flag when the minimum is 0 or 1. */

              if (brazeroptr != NULL) *brazeroptr = OP_BRAPOSZERO;
              if (repeat_min < 2) possessive_quantifier = FALSE;
              }

            /* Non-possessive quantifier */

            else *ketcode = OP_KETRMAX + repeat_type;
            }
          }
        }
      break;

      /* If previous was a character type match (\d or similar), abolish it and
      create a suitable repeat item. The code is shared with single-character
      repeats by setting op_type to add a suitable offset into repeat_type.
      Note the the Unicode property types will be present only when
      SUPPORT_UNICODE is defined, but we don't wrap the little bits of code
      here because it just makes it horribly messy. */

      default:
      if (op_previous >= OP_EODN || op_previous <= OP_WORD_BOUNDARY)
        {
        PCRE2_DEBUG_UNREACHABLE();
        *errorcodeptr = ERR10;  /* Not a character type - internal error */
        return 0;
        }
      else
        {
        int prop_type, prop_value;
        PCRE2_UCHAR *oldcode;

        if (repeat_max == 1 && repeat_min == 1) goto END_REPEAT;

        op_type = OP_TYPESTAR - OP_STAR;      /* Use type opcodes */
        mclength = 0;                         /* Not a character */

        if (op_previous == OP_PROP || op_previous == OP_NOTPROP)
          {
          prop_type = previous[1];
          prop_value = previous[2];
          }
        else
          {
          /* Come here from just above with a character in mcbuffer/mclength.
          You must also set op_type before the jump. */
          OUTPUT_SINGLE_REPEAT:
          prop_type = prop_value = -1;
          }

        /* At this point, if prop_type == prop_value == -1 we either have a
        character in mcbuffer when mclength is greater than zero, or we have
        mclength zero, in which case there is a non-property character type in
        op_previous. If prop_type/value are not negative, we have a property
        character type in op_previous. */

        oldcode = code;                   /* Save where we were */
        code = previous;                  /* Usually overwrite previous item */

        /* If the maximum is zero then the minimum must also be zero; Perl allows
        this case, so we do too - by simply omitting the item altogether. */

        if (repeat_max == 0) goto END_REPEAT;

        /* Combine the op_type with the repeat_type */

        repeat_type += op_type;

        /* A minimum of zero is handled either as the special case * or ?, or as
        an UPTO, with the maximum given. */

        if (repeat_min == 0)
          {
          if (repeat_max == REPEAT_UNLIMITED) *code++ = OP_STAR + repeat_type;
            else if (repeat_max == 1) *code++ = OP_QUERY + repeat_type;
          else
            {
            *code++ = OP_UPTO + repeat_type;
            PUT2INC(code, 0, repeat_max);
            }
          }

        /* A repeat minimum of 1 is optimized into some special cases. If the
        maximum is unlimited, we use OP_PLUS. Otherwise, the original item is
        left in place and, if the maximum is greater than 1, we use OP_UPTO with
        one less than the maximum. */

        else if (repeat_min == 1)
          {
          if (repeat_max == REPEAT_UNLIMITED)
            *code++ = OP_PLUS + repeat_type;
          else
            {
            code = oldcode;  /* Leave previous item in place */
            if (repeat_max == 1) goto END_REPEAT;
            *code++ = OP_UPTO + repeat_type;
            PUT2INC(code, 0, repeat_max - 1);
            }
          }

        /* The case {n,n} is just an EXACT, while the general case {n,m} is
        handled as an EXACT followed by an UPTO or STAR or QUERY. */

        else
          {
          *code++ = OP_EXACT + op_type;  /* NB EXACT doesn't have repeat_type */
          PUT2INC(code, 0, repeat_min);

          /* Unless repeat_max equals repeat_min, fill in the data for EXACT,
          and then generate the second opcode. For a repeated Unicode property
          match, there are two extra values that define the required property,
          and mclength is set zero to indicate this. */

          if (repeat_max != repeat_min)
            {
            if (mclength > 0)
              {
              memcpy(code, mcbuffer, CU2BYTES(mclength));
              code += mclength;
              }
            else
              {
              *code++ = op_previous;
              if (prop_type >= 0)
                {
                *code++ = prop_type;
                *code++ = prop_value;
                }
              }

            /* Now set up the following opcode */

            if (repeat_max == REPEAT_UNLIMITED)
              *code++ = OP_STAR + repeat_type;
            else
              {
              repeat_max -= repeat_min;
              if (repeat_max == 1)
                {
                *code++ = OP_QUERY + repeat_type;
                }
              else
                {
                *code++ = OP_UPTO + repeat_type;
                PUT2INC(code, 0, repeat_max);
                }
              }
            }
          }

        /* Fill in the character or character type for the final opcode. */

        if (mclength > 0)
          {
          memcpy(code, mcbuffer, CU2BYTES(mclength));
          code += mclength;
          }
        else
          {
          *code++ = op_previous;
          if (prop_type >= 0)
            {
            *code++ = prop_type;
            *code++ = prop_value;
            }
          }
        }
      break;
      }  /* End of switch on different op_previous values */


    /* If the character following a repeat is '+', possessive_quantifier is
    TRUE. For some opcodes, there are special alternative opcodes for this
    case. For anything else, we wrap the entire repeated item inside OP_ONCE
    brackets. Logically, the '+' notation is just syntactic sugar, taken from
    Sun's Java package, but the special opcodes can optimize it.

    Some (but not all) possessively repeated subpatterns have already been
    completely handled in the code just above. For them, possessive_quantifier
    is always FALSE at this stage. Note that the repeated item starts at
    tempcode, not at previous, which might be the first part of a string whose
    (former) last char we repeated. */

    if (possessive_quantifier)
      {
      int len;

      /* Possessifying an EXACT quantifier has no effect, so we can ignore it.
      However, QUERY, STAR, or UPTO may follow (for quantifiers such as {5,6},
      {5,}, or {5,10}). We skip over an EXACT item; if the length of what
      remains is greater than zero, there's a further opcode that can be
      handled. If not, do nothing, leaving the EXACT alone. */

      switch(*tempcode)
        {
        case OP_TYPEEXACT:
        tempcode += PRIV(OP_lengths)[*tempcode] +
          ((tempcode[1 + IMM2_SIZE] == OP_PROP
          || tempcode[1 + IMM2_SIZE] == OP_NOTPROP)? 2 : 0);
        break;

        /* CHAR opcodes are used for exacts whose count is 1. */

        case OP_CHAR:
        case OP_CHARI:
        case OP_NOT:
        case OP_NOTI:
        case OP_EXACT:
        case OP_EXACTI:
        case OP_NOTEXACT:
        case OP_NOTEXACTI:
        tempcode += PRIV(OP_lengths)[*tempcode];
#ifdef SUPPORT_UNICODE
        if (utf && HAS_EXTRALEN(tempcode[-1]))
          tempcode += GET_EXTRALEN(tempcode[-1]);
#endif
        break;

        /* For the class opcodes, the repeat operator appears at the end;
        adjust tempcode to point to it. */

        case OP_CLASS:
        case OP_NCLASS:
        tempcode += 1 + 32/sizeof(PCRE2_UCHAR);
        break;

#ifdef SUPPORT_WIDE_CHARS
        case OP_XCLASS:
        case OP_ECLASS:
        tempcode += GET(tempcode, 1);
        break;
#endif
        }

      /* If tempcode is equal to code (which points to the end of the repeated
      item), it means we have skipped an EXACT item but there is no following
      QUERY, STAR, or UPTO; the value of len will be 0, and we do nothing. In
      all other cases, tempcode will be pointing to the repeat opcode, and will
      be less than code, so the value of len will be greater than 0. */

      len = (int)(code - tempcode);
      if (len > 0)
        {
        unsigned int repcode = *tempcode;

        /* There is a table for possessifying opcodes, all of which are less
        than OP_CALLOUT. A zero entry means there is no possessified version.
        */

        if (repcode < OP_CALLOUT && opcode_possessify[repcode] > 0)
          *tempcode = opcode_possessify[repcode];

        /* For opcode without a special possessified version, wrap the item in
        ONCE brackets. */

        else
          {
          (void)memmove(tempcode + 1 + LINK_SIZE, tempcode, CU2BYTES(len));
          code += 1 + LINK_SIZE;
          len += 1 + LINK_SIZE;
          tempcode[0] = OP_ONCE;
          *code++ = OP_KET;
          PUTINC(code, 0, len);
          PUT(tempcode, 1, len);
          }
        }
      }

    /* We set the "follows varying string" flag for subsequently encountered
    reqcus if it isn't already set and we have just passed a varying length
    item. */

    END_REPEAT:
    cb->req_varyopt |= reqvary;
    break;


    /* ===================================================================*/
    /* Handle a 32-bit data character with a value greater than META_END. */

    case META_BIGVALUE:
    pptr++;
    goto NORMAL_CHAR;


    /* ===============================================================*/
    /* Handle a back reference by number, which is the meta argument. The
    pattern offsets for back references to group numbers less than 10 are held
    in a special vector, to avoid using more than two parsed pattern elements
    in 64-bit environments. We only need the offset to the first occurrence,
    because if that doesn't fail, subsequent ones will also be OK. */

    case META_BACKREF:
    if (meta_arg < 10) offset = cb->small_ref_offset[meta_arg];
      else GETPLUSOFFSET(offset, pptr);

    if (meta_arg > cb->bracount)
      {
      cb->erroroffset = offset;
      *errorcodeptr = ERR15;  /* Non-existent subpattern */
      return 0;
      }

    /* Come here from named backref handling when the reference is to a
    single group (that is, not to a duplicated name). The back reference
    data will have already been updated. We must disable firstcu if not
    set, to cope with cases like (?=(\w+))\1: which would otherwise set ':'
    later. */

    HANDLE_SINGLE_REFERENCE:
    if (firstcuflags == REQ_UNSET) zerofirstcuflags = firstcuflags = REQ_NONE;
    *code++ = ((options & PCRE2_CASELESS) != 0)? OP_REFI : OP_REF;
    PUT2INC(code, 0, meta_arg);
    if ((options & PCRE2_CASELESS) != 0)
      *code++ = (((xoptions & PCRE2_EXTRA_CASELESS_RESTRICT) != 0)?
                 REFI_FLAG_CASELESS_RESTRICT : 0) |
                (((xoptions & PCRE2_EXTRA_TURKISH_CASING) != 0)?
                 REFI_FLAG_TURKISH_CASING : 0);

    /* Update the map of back references, and keep the highest one. We
    could do this in parse_regex() for numerical back references, but not
    for named back references, because we don't know the numbers to which
    named back references refer. So we do it all in this function. */

    cb->backref_map |= (meta_arg < 32)? (1u << meta_arg) : 1;
    if (meta_arg > cb->top_backref) cb->top_backref = meta_arg;
    break;


    /* ===============================================================*/
    /* Handle recursion by inserting the number of the called group (which is
    the meta argument) after OP_RECURSE. At the end of compiling the pattern is
    scanned and these numbers are replaced by offsets within the pattern. It is
    done like this to avoid problems with forward references and adjusting
    offsets when groups are duplicated and moved (as discovered in previous
    implementations). Note that a recursion does not have a set first
    character. */

    case META_RECURSE:
    GETPLUSOFFSET(offset, pptr);
    if (meta_arg > cb->bracount)
      {
      cb->erroroffset = offset;
      *errorcodeptr = ERR15;  /* Non-existent subpattern */
      return 0;
      }
    HANDLE_NUMERICAL_RECURSION:
    *code = OP_RECURSE;
    PUT(code, 1, meta_arg);
    code += 1 + LINK_SIZE;
    groupsetfirstcu = FALSE;
    cb->had_recurse = TRUE;
    if (firstcuflags == REQ_UNSET) firstcuflags = REQ_NONE;
    zerofirstcu = firstcu;
    zerofirstcuflags = firstcuflags;
    break;


    /* ===============================================================*/
    /* Handle capturing parentheses; the number is the meta argument. */

    case META_CAPTURE:
    bravalue = OP_CBRA;
    skipunits = IMM2_SIZE;
    PUT2(code, 1+LINK_SIZE, meta_arg);
    cb->lastcapture = meta_arg;
    goto GROUP_PROCESS_NOTE_EMPTY;


    /* ===============================================================*/
    /* Handle escape sequence items. For ones like \d, the ESC_values are
    arranged to be the same as the corresponding OP_values in the default case
    when PCRE2_UCP is not set (which is the only case in which they will appear
    here).

    Note: \Q and \E are never seen here, as they were dealt with in
    parse_pattern(). Neither are numerical back references or recursions, which
    were turned into META_BACKREF or META_RECURSE items, respectively. \k and
    \g, when followed by names, are turned into META_BACKREF_BYNAME or
    META_RECURSE_BYNAME. */

    case META_ESCAPE:

    /* We can test for escape sequences that consume a character because their
    values lie between ESC_b and ESC_Z; this may have to change if any new ones
    are ever created. For these sequences, we disable the setting of a first
    character if it hasn't already been set. */

    if (meta_arg > ESC_b && meta_arg < ESC_Z)
      {
      matched_char = TRUE;
      if (firstcuflags == REQ_UNSET) firstcuflags = REQ_NONE;
      }

    /* Set values to reset to if this is followed by a zero repeat. */

    zerofirstcu = firstcu;
    zerofirstcuflags = firstcuflags;
    zeroreqcu = reqcu;
    zeroreqcuflags = reqcuflags;

    /* If Unicode is not supported, \P and \p are not allowed and are
    faulted at parse time, so will never appear here. */

#ifdef SUPPORT_UNICODE
    if (meta_arg == ESC_P || meta_arg == ESC_p)
      {
      uint32_t ptype = *(++pptr) >> 16;
      uint32_t pdata = *pptr & 0xffff;

      /* In caseless matching, particular characteristics Lu, Ll, and Lt get
      converted to the general characteristic L&. That is, upper, lower, and
      title case letters are all conflated. */

      if ((options & PCRE2_CASELESS) != 0 && ptype == PT_PC &&
          (pdata == ucp_Lu || pdata == ucp_Ll || pdata == ucp_Lt))
        {
        ptype = PT_LAMP;
        pdata = 0;
        }

      /* The special case of \p{Any} is compiled to OP_ALLANY and \P{Any}
      is compiled to [] so as to benefit from the auto-anchoring code. */

      if (ptype == PT_ANY)
        {
        if (meta_arg == ESC_P)
          {
          *code++ = OP_CLASS;
          memset(code, 0, 32);
          code += 32 / sizeof(PCRE2_UCHAR);
          }
        else
          *code++ = OP_ALLANY;
        }
      else
        {
        *code++ = (meta_arg == ESC_p)? OP_PROP : OP_NOTPROP;
        *code++ = ptype;
        *code++ = pdata;
        }
      break;  /* End META_ESCAPE */
      }
#endif

    /* \K is forbidden in lookarounds since 10.38 because that's what Perl has
    done. However, there's an option, in case anyone was relying on it. */

    if (cb->assert_depth > 0 && meta_arg == ESC_K &&
        (xoptions & PCRE2_EXTRA_ALLOW_LOOKAROUND_BSK) == 0)
      {
      *errorcodeptr = ERR99;
      return 0;
      }

    /* For the rest (including \X when Unicode is supported - if not it's
    faulted at parse time), the OP value is the escape value when PCRE2_UCP is
    not set; if it is set, most of them do not show up here because they are
    converted into Unicode property tests in parse_regex().

    In non-UTF mode, and for both 32-bit modes, we turn \C into OP_ALLANY
    instead of OP_ANYBYTE so that it works in DFA mode and in lookbehinds.
    There are special UCP codes for \B and \b which are used in UCP mode unless
    "word" matching is being forced to ASCII.

    Note that \b and \B do a one-character lookbehind, and \A also behaves as
    if it does. */

    switch(meta_arg)
      {
      case ESC_C:
      cb->external_flags |= PCRE2_HASBKC;  /* Record */
#if PCRE2_CODE_UNIT_WIDTH == 32
      meta_arg = OP_ALLANY;
#else
      if (!utf) meta_arg = OP_ALLANY;
#endif
      break;

      case ESC_B:
      case ESC_b:
      if ((options & PCRE2_UCP) != 0 && (xoptions & PCRE2_EXTRA_ASCII_BSW) == 0)
        meta_arg = (meta_arg == ESC_B)? OP_NOT_UCP_WORD_BOUNDARY :
          OP_UCP_WORD_BOUNDARY;
      /* Fall through */

      case ESC_A:
      if (cb->max_lookbehind == 0) cb->max_lookbehind = 1;
      break;
      }

    *code++ = meta_arg;
    break;  /* End META_ESCAPE */


    /* ===================================================================*/
    /* Handle an unrecognized meta value. A parsed pattern value less than
    META_END is a literal. Otherwise we have a problem. */

    default:
    if (meta >= META_END)
      {
      PCRE2_DEBUG_UNREACHABLE();
      *errorcodeptr = ERR89;  /* Internal error - unrecognized. */
      return 0;
      }

    /* Handle a literal character. We come here by goto in the case of a
    32-bit, non-UTF character whose value is greater than META_END. */

    NORMAL_CHAR:
    meta = *pptr;     /* Get the full 32 bits */
    NORMAL_CHAR_SET:  /* Character is already in meta */
    matched_char = TRUE;

    /* For caseless UTF or UCP mode, check whether this character has more than
    one other case. If so, generate a special OP_PROP item instead of OP_CHARI.
    When casing restrictions apply, ignore caseless sets that start with an
    ASCII character. If the character is affected by the special Turkish rules,
    hardcode the matching characters using a caseset. */

#ifdef SUPPORT_UNICODE
    if ((utf||ucp) && (options & PCRE2_CASELESS) != 0)
      {
      uint32_t caseset;

      if ((xoptions & (PCRE2_EXTRA_TURKISH_CASING|PCRE2_EXTRA_CASELESS_RESTRICT)) ==
            PCRE2_EXTRA_TURKISH_CASING &&
          UCD_ANY_I(meta))
        {
        caseset = PRIV(ucd_turkish_dotted_i_caseset) + (UCD_DOTTED_I(meta)? 0 : 3);
        }
      else if ((caseset = UCD_CASESET(meta)) != 0 &&
               (xoptions & PCRE2_EXTRA_CASELESS_RESTRICT) != 0 &&
               PRIV(ucd_caseless_sets)[caseset] < 128)
        {
        caseset = 0;  /* Ignore the caseless set if it's restricted. */
        }

      if (caseset != 0)
        {
        *code++ = OP_PROP;
        *code++ = PT_CLIST;
        *code++ = caseset;
        if (firstcuflags == REQ_UNSET)
          firstcuflags = zerofirstcuflags = REQ_NONE;
        break;  /* End handling this meta item */
        }
      }
#endif

    /* Caseful matches, or caseless and not one of the multicase characters. We
    come here by goto in the case of a positive class that contains only
    case-partners of a character with just two cases; matched_char has already
    been set TRUE and options fudged if necessary. */

    CLASS_CASELESS_CHAR:

    /* Get the character's code units into mcbuffer, with the length in
    mclength. When not in UTF mode, the length is always 1. */

#ifdef SUPPORT_UNICODE
    if (utf) mclength = PRIV(ord2utf)(meta, mcbuffer); else
#endif
      {
      mclength = 1;
      mcbuffer[0] = meta;
      }

    /* Generate the appropriate code */

    *code++ = ((options & PCRE2_CASELESS) != 0)? OP_CHARI : OP_CHAR;
    memcpy(code, mcbuffer, CU2BYTES(mclength));
    code += mclength;

    /* Remember if \r or \n were seen */

    if (mcbuffer[0] == CHAR_CR || mcbuffer[0] == CHAR_NL)
      cb->external_flags |= PCRE2_HASCRORLF;

    /* Set the first and required code units appropriately. If no previous
    first code unit, set it from this character, but revert to none on a zero
    repeat. Otherwise, leave the firstcu value alone, and don't change it on
    a zero repeat. */

    if (firstcuflags == REQ_UNSET)
      {
      zerofirstcuflags = REQ_NONE;
      zeroreqcu = reqcu;
      zeroreqcuflags = reqcuflags;

      /* If the character is more than one code unit long, we can set a single
      firstcu only if it is not to be matched caselessly. Multiple possible
      starting code units may be picked up later in the studying code. */

      if (mclength == 1 || req_caseopt == 0)
        {
        firstcu = mcbuffer[0];
        firstcuflags = req_caseopt;
        if (mclength != 1)
          {
          reqcu = code[-1];
          reqcuflags = cb->req_varyopt;
          }
        }
      else firstcuflags = reqcuflags = REQ_NONE;
      }

    /* firstcu was previously set; we can set reqcu only if the length is
    1 or the matching is caseful. */

    else
      {
      zerofirstcu = firstcu;
      zerofirstcuflags = firstcuflags;
      zeroreqcu = reqcu;
      zeroreqcuflags = reqcuflags;
      if (mclength == 1 || req_caseopt == 0)
        {
        reqcu = code[-1];
        reqcuflags = req_caseopt | cb->req_varyopt;
        }
      }

    /* If caselessness was temporarily instated, reset it. */

    if (reset_caseful)
      {
      options &= ~PCRE2_CASELESS;
      req_caseopt = 0;
      reset_caseful = FALSE;
      }

    break;    /* End literal character handling */
    }         /* End of big switch */
  }           /* End of big loop */

PCRE2_DEBUG_UNREACHABLE(); /* Control should never reach here */
return 0;                  /* Avoid compiler warnings */
}



/*************************************************
*   Compile regex: a sequence of alternatives    *
*************************************************/

/* On entry, pptr is pointing past the bracket meta, but on return it points to
the closing bracket or META_END. The code variable is pointing at the code unit
into which the BRA operator has been stored. This function is used during the
pre-compile phase when we are trying to find out the amount of memory needed,
as well as during the real compile phase. The value of lengthptr distinguishes
the two phases.

Arguments:
  options           option bits, including any changes for this subpattern
  xoptions          extra option bits, ditto
  codeptr           -> the address of the current code pointer
  pptrptr           -> the address of the current parsed pattern pointer
  errorcodeptr      -> pointer to error code variable
  skipunits         skip this many code units at start (for brackets and OP_COND)
  firstcuptr        place to put the first required code unit
  firstcuflagsptr   place to put the first code unit flags
  reqcuptr          place to put the last required code unit
  reqcuflagsptr     place to put the last required code unit flags
  bcptr             pointer to the chain of currently open branches
  cb                points to the data block with tables pointers etc.
  lengthptr         NULL during the real compile phase
                    points to length accumulator during pre-compile phase

Returns:            0 There has been an error
                   +1 Success, this group must match at least one character
                   -1 Success, this group may match an empty string
*/

static int
compile_regex(uint32_t options, uint32_t xoptions, PCRE2_UCHAR **codeptr,
  uint32_t **pptrptr, int *errorcodeptr, uint32_t skipunits,
  uint32_t *firstcuptr, uint32_t *firstcuflagsptr, uint32_t *reqcuptr,
  uint32_t *reqcuflagsptr, branch_chain *bcptr, open_capitem *open_caps,
  compile_block *cb, PCRE2_SIZE *lengthptr)
{
PCRE2_UCHAR *code = *codeptr;
PCRE2_UCHAR *last_branch = code;
PCRE2_UCHAR *start_bracket = code;
BOOL lookbehind;
open_capitem capitem;
int capnumber = 0;
int okreturn = 1;
uint32_t *pptr = *pptrptr;
uint32_t firstcu, reqcu;
uint32_t lookbehindlength;
uint32_t lookbehindminlength;
uint32_t firstcuflags, reqcuflags;
PCRE2_SIZE length;
branch_chain bc;

/* If set, call the external function that checks for stack availability. */

if (cb->cx->stack_guard != NULL &&
    cb->cx->stack_guard(cb->parens_depth, cb->cx->stack_guard_data))
  {
  *errorcodeptr= ERR33;
  return 0;
  }

/* Miscellaneous initialization */

bc.outer = bcptr;
bc.current_branch = code;

firstcu = reqcu = 0;
firstcuflags = reqcuflags = REQ_UNSET;

/* Accumulate the length for use in the pre-compile phase. Start with the
length of the BRA and KET and any extra code units that are required at the
beginning. We accumulate in a local variable to save frequent testing of
lengthptr for NULL. We cannot do this by looking at the value of 'code' at the
start and end of each alternative, because compiled items are discarded during
the pre-compile phase so that the workspace is not exceeded. */

length = 2 + 2*LINK_SIZE + skipunits;

/* Remember if this is a lookbehind assertion, and if it is, save its length
and skip over the pattern offset. */

lookbehind = *code == OP_ASSERTBACK ||
             *code == OP_ASSERTBACK_NOT ||
             *code == OP_ASSERTBACK_NA;

if (lookbehind)
  {
  lookbehindlength = META_DATA(pptr[-1]);
  lookbehindminlength = *pptr;
  pptr += SIZEOFFSET;
  }
else lookbehindlength = lookbehindminlength = 0;

/* If this is a capturing subpattern, add to the chain of open capturing items
so that we can detect them if (*ACCEPT) is encountered. Note that only OP_CBRA
need be tested here; changing this opcode to one of its variants, e.g.
OP_SCBRAPOS, happens later, after the group has been compiled. */

if (*code == OP_CBRA)
  {
  capnumber = GET2(code, 1 + LINK_SIZE);
  capitem.number = capnumber;
  capitem.next = open_caps;
  capitem.assert_depth = cb->assert_depth;
  open_caps = &capitem;
  }

/* Offset is set zero to mark that this bracket is still open */

PUT(code, 1, 0);
code += 1 + LINK_SIZE + skipunits;

/* Loop for each alternative branch */

for (;;)
  {
  int branch_return;
  uint32_t branchfirstcu = 0, branchreqcu = 0;
  uint32_t branchfirstcuflags = REQ_UNSET, branchreqcuflags = REQ_UNSET;

  /* Insert OP_REVERSE or OP_VREVERSE if this is a lookbehind assertion. There
  is only a single minimum length for the whole assertion. When the minimum
  length is LOOKBEHIND_MAX it means that all branches are of fixed length,
  though not necessarily the same length. In this case, the original OP_REVERSE
  can be used. It can also be used if a branch in a variable length lookbehind
  has the same maximum and minimum. Otherwise, use OP_VREVERSE, which has both
  maximum and minimum values. */

  if (lookbehind && lookbehindlength > 0)
    {
    if (lookbehindminlength == LOOKBEHIND_MAX ||
        lookbehindminlength == lookbehindlength)
      {
      *code++ = OP_REVERSE;
      PUT2INC(code, 0, lookbehindlength);
      length += 1 + IMM2_SIZE;
      }
    else
      {
      *code++ = OP_VREVERSE;
      PUT2INC(code, 0, lookbehindminlength);
      PUT2INC(code, 0, lookbehindlength);
      length += 1 + 2*IMM2_SIZE;
      }
    }

  /* Now compile the branch; in the pre-compile phase its length gets added
  into the length. */

  if ((branch_return =
        compile_branch(&options, &xoptions, &code, &pptr, errorcodeptr,
          &branchfirstcu, &branchfirstcuflags, &branchreqcu, &branchreqcuflags,
          &bc, open_caps, cb, (lengthptr == NULL)? NULL : &length)) == 0)
    return 0;

  /* If a branch can match an empty string, so can the whole group. */

  if (branch_return < 0) okreturn = -1;

  /* In the real compile phase, there is some post-processing to be done. */

  if (lengthptr == NULL)
    {
    /* If this is the first branch, the firstcu and reqcu values for the
    branch become the values for the regex. */

    if (*last_branch != OP_ALT)
      {
      firstcu = branchfirstcu;
      firstcuflags = branchfirstcuflags;
      reqcu = branchreqcu;
      reqcuflags = branchreqcuflags;
      }

    /* If this is not the first branch, the first char and reqcu have to
    match the values from all the previous branches, except that if the
    previous value for reqcu didn't have REQ_VARY set, it can still match,
    and we set REQ_VARY for the group from this branch's value. */

    else
      {
      /* If we previously had a firstcu, but it doesn't match the new branch,
      we have to abandon the firstcu for the regex, but if there was
      previously no reqcu, it takes on the value of the old firstcu. */

      if (firstcuflags != branchfirstcuflags || firstcu != branchfirstcu)
        {
        if (firstcuflags < REQ_NONE)
          {
          if (reqcuflags >= REQ_NONE)
            {
            reqcu = firstcu;
            reqcuflags = firstcuflags;
            }
          }
        firstcuflags = REQ_NONE;
        }

      /* If we (now or from before) have no firstcu, a firstcu from the
      branch becomes a reqcu if there isn't a branch reqcu. */

      if (firstcuflags >= REQ_NONE && branchfirstcuflags < REQ_NONE &&
          branchreqcuflags >= REQ_NONE)
        {
        branchreqcu = branchfirstcu;
        branchreqcuflags = branchfirstcuflags;
        }

      /* Now ensure that the reqcus match */

      if (((reqcuflags & ~REQ_VARY) != (branchreqcuflags & ~REQ_VARY)) ||
          reqcu != branchreqcu)
        reqcuflags = REQ_NONE;
      else
        {
        reqcu = branchreqcu;
        reqcuflags |= branchreqcuflags; /* To "or" REQ_VARY if present */
        }
      }
    }

  /* Handle reaching the end of the expression, either ')' or end of pattern.
  In the real compile phase, go back through the alternative branches and
  reverse the chain of offsets, with the field in the BRA item now becoming an
  offset to the first alternative. If there are no alternatives, it points to
  the end of the group. The length in the terminating ket is always the length
  of the whole bracketed item. Return leaving the pointer at the terminating
  char. */

  if (META_CODE(*pptr) != META_ALT)
    {
    if (lengthptr == NULL)
      {
      uint32_t branch_length = (uint32_t)(code - last_branch);
      do
        {
        uint32_t prev_length = GET(last_branch, 1);
        PUT(last_branch, 1, branch_length);
        branch_length = prev_length;
        last_branch -= branch_length;
        }
      while (branch_length > 0);
      }

    /* Fill in the ket */

    *code = OP_KET;
    PUT(code, 1, (uint32_t)(code - start_bracket));
    code += 1 + LINK_SIZE;

    /* Set values to pass back */

    *codeptr = code;
    *pptrptr = pptr;
    *firstcuptr = firstcu;
    *firstcuflagsptr = firstcuflags;
    *reqcuptr = reqcu;
    *reqcuflagsptr = reqcuflags;
    if (lengthptr != NULL)
      {
      if (OFLOW_MAX - *lengthptr < length)
        {
        *errorcodeptr = ERR20;
        return 0;
        }
      *lengthptr += length;
      }
    return okreturn;
    }

  /* Another branch follows. In the pre-compile phase, we can move the code
  pointer back to where it was for the start of the first branch. (That is,
  pretend that each branch is the only one.)

  In the real compile phase, insert an ALT node. Its length field points back
  to the previous branch while the bracket remains open. At the end the chain
  is reversed. It's done like this so that the start of the bracket has a
  zero offset until it is closed, making it possible to detect recursion. */

  if (lengthptr != NULL)
    {
    code = *codeptr + 1 + LINK_SIZE + skipunits;
    length += 1 + LINK_SIZE;
    }
  else
    {
    *code = OP_ALT;
    PUT(code, 1, (int)(code - last_branch));
    bc.current_branch = last_branch = code;
    code += 1 + LINK_SIZE;
    }

  /* Set the maximum lookbehind length for the next branch (if not in a
  lookbehind the value will be zero) and then advance past the vertical bar. */

  lookbehindlength = META_DATA(*pptr);
  pptr++;
  }

PCRE2_DEBUG_UNREACHABLE(); /* Control should never reach here */
return 0;                  /* Avoid compiler warnings */
}



/*************************************************
*          Check for anchored pattern            *
*************************************************/

/* Try to find out if this is an anchored regular expression. Consider each
alternative branch. If they all start with OP_SOD or OP_CIRC, or with a bracket
all of whose alternatives start with OP_SOD or OP_CIRC (recurse ad lib), then
it's anchored. However, if this is a multiline pattern, then only OP_SOD will
be found, because ^ generates OP_CIRCM in that mode.

We can also consider a regex to be anchored if OP_SOM starts all its branches.
This is the code for \G, which means "match at start of match position, taking
into account the match offset".

A branch is also implicitly anchored if it starts with .* and DOTALL is set,
because that will try the rest of the pattern at all possible matching points,
so there is no point trying again.... er ....

.... except when the .* appears inside capturing parentheses, and there is a
subsequent back reference to those parentheses. We haven't enough information
to catch that case precisely.

At first, the best we could do was to detect when .* was in capturing brackets
and the highest back reference was greater than or equal to that level.
However, by keeping a bitmap of the first 31 back references, we can catch some
of the more common cases more precisely.

... A second exception is when the .* appears inside an atomic group, because
this prevents the number of characters it matches from being adjusted.

Arguments:
  code           points to start of the compiled pattern
  bracket_map    a bitmap of which brackets we are inside while testing; this
                   handles up to substring 31; after that we just have to take
                   the less precise approach
  cb             points to the compile data block
  atomcount      atomic group level
  inassert       TRUE if in an assertion
  dotstar_anchor TRUE if automatic anchoring optimization is enabled

Returns:     TRUE or FALSE
*/

static BOOL
is_anchored(PCRE2_SPTR code, uint32_t bracket_map, compile_block *cb,
  int atomcount, BOOL inassert, BOOL dotstar_anchor)
{
do {
   PCRE2_SPTR scode = first_significant_code(
     code + PRIV(OP_lengths)[*code], FALSE);
   int op = *scode;

   /* Non-capturing brackets */

   if (op == OP_BRA  || op == OP_BRAPOS ||
       op == OP_SBRA || op == OP_SBRAPOS)
     {
     if (!is_anchored(scode, bracket_map, cb, atomcount, inassert, dotstar_anchor))
       return FALSE;
     }

   /* Capturing brackets */

   else if (op == OP_CBRA  || op == OP_CBRAPOS ||
            op == OP_SCBRA || op == OP_SCBRAPOS)
     {
     int n = GET2(scode, 1+LINK_SIZE);
     uint32_t new_map = bracket_map | ((n < 32)? (1u << n) : 1);
     if (!is_anchored(scode, new_map, cb, atomcount, inassert, dotstar_anchor)) return FALSE;
     }

   /* Positive forward assertion */

   else if (op == OP_ASSERT || op == OP_ASSERT_NA)
     {
     if (!is_anchored(scode, bracket_map, cb, atomcount, TRUE, dotstar_anchor)) return FALSE;
     }

   /* Condition. If there is no second branch, it can't be anchored. */

   else if (op == OP_COND || op == OP_SCOND)
     {
     if (scode[GET(scode,1)] != OP_ALT) return FALSE;
     if (!is_anchored(scode, bracket_map, cb, atomcount, inassert, dotstar_anchor))
       return FALSE;
     }

   /* Atomic groups */

   else if (op == OP_ONCE)
     {
     if (!is_anchored(scode, bracket_map, cb, atomcount + 1, inassert, dotstar_anchor))
       return FALSE;
     }

   /* .* is not anchored unless DOTALL is set (which generates OP_ALLANY) and
   it isn't in brackets that are or may be referenced or inside an atomic
   group or an assertion. Also the pattern must not contain *PRUNE or *SKIP,
   because these break the feature. Consider, for example, /(?s).*?(*PRUNE)b/
   with the subject "aab", which matches "b", i.e. not at the start of a line.
   There is also an option that disables auto-anchoring. */

   else if ((op == OP_TYPESTAR || op == OP_TYPEMINSTAR ||
             op == OP_TYPEPOSSTAR))
     {
     if (scode[1] != OP_ALLANY || (bracket_map & cb->backref_map) != 0 ||
         atomcount > 0 || cb->had_pruneorskip || inassert || !dotstar_anchor)
       return FALSE;
     }

   /* Check for explicit anchoring */

   else if (op != OP_SOD && op != OP_SOM && op != OP_CIRC) return FALSE;

   code += GET(code, 1);
   }
while (*code == OP_ALT);   /* Loop for each alternative */
return TRUE;
}



/*************************************************
*         Check for starting with ^ or .*        *
*************************************************/

/* This is called to find out if every branch starts with ^ or .* so that
"first char" processing can be done to speed things up in multiline
matching and for non-DOTALL patterns that start with .* (which must start at
the beginning or after \n). As in the case of is_anchored() (see above), we
have to take account of back references to capturing brackets that contain .*
because in that case we can't make the assumption. Also, the appearance of .*
inside atomic brackets or in an assertion, or in a pattern that contains *PRUNE
or *SKIP does not count, because once again the assumption no longer holds.

Arguments:
  code           points to start of the compiled pattern or a group
  bracket_map    a bitmap of which brackets we are inside while testing; this
                   handles up to substring 31; after that we just have to take
                   the less precise approach
  cb             points to the compile data
  atomcount      atomic group level
  inassert       TRUE if in an assertion
  dotstar_anchor TRUE if automatic anchoring optimization is enabled

Returns:         TRUE or FALSE
*/

static BOOL
is_startline(PCRE2_SPTR code, unsigned int bracket_map, compile_block *cb,
  int atomcount, BOOL inassert, BOOL dotstar_anchor)
{
do {
   PCRE2_SPTR scode = first_significant_code(
     code + PRIV(OP_lengths)[*code], FALSE);
   int op = *scode;

   /* If we are at the start of a conditional assertion group, *both* the
   conditional assertion *and* what follows the condition must satisfy the test
   for start of line. Other kinds of condition fail. Note that there may be an
   auto-callout at the start of a condition. */

   if (op == OP_COND)
     {
     scode += 1 + LINK_SIZE;

     if (*scode == OP_CALLOUT) scode += PRIV(OP_lengths)[OP_CALLOUT];
       else if (*scode == OP_CALLOUT_STR) scode += GET(scode, 1 + 2*LINK_SIZE);

     switch (*scode)
       {
       case OP_CREF:
       case OP_DNCREF:
       case OP_RREF:
       case OP_DNRREF:
       case OP_FAIL:
       case OP_FALSE:
       case OP_TRUE:
       return FALSE;

       default:     /* Assertion */
       if (!is_startline(scode, bracket_map, cb, atomcount, TRUE, dotstar_anchor))
         return FALSE;
       do scode += GET(scode, 1); while (*scode == OP_ALT);
       scode += 1 + LINK_SIZE;
       break;
       }
     scode = first_significant_code(scode, FALSE);
     op = *scode;
     }

   /* Non-capturing brackets */

   if (op == OP_BRA  || op == OP_BRAPOS ||
       op == OP_SBRA || op == OP_SBRAPOS)
     {
     if (!is_startline(scode, bracket_map, cb, atomcount, inassert, dotstar_anchor))
       return FALSE;
     }

   /* Capturing brackets */

   else if (op == OP_CBRA  || op == OP_CBRAPOS ||
            op == OP_SCBRA || op == OP_SCBRAPOS)
     {
     int n = GET2(scode, 1+LINK_SIZE);
     unsigned int new_map = bracket_map | ((n < 32)? (1u << n) : 1);
     if (!is_startline(scode, new_map, cb, atomcount, inassert, dotstar_anchor))
       return FALSE;
     }

   /* Positive forward assertions */

   else if (op == OP_ASSERT || op == OP_ASSERT_NA)
     {
     if (!is_startline(scode, bracket_map, cb, atomcount, TRUE, dotstar_anchor))
       return FALSE;
     }

   /* Atomic brackets */

   else if (op == OP_ONCE)
     {
     if (!is_startline(scode, bracket_map, cb, atomcount + 1, inassert, dotstar_anchor))
       return FALSE;
     }

   /* .* means "start at start or after \n" if it isn't in atomic brackets or
   brackets that may be referenced or an assertion, and as long as the pattern
   does not contain *PRUNE or *SKIP, because these break the feature. Consider,
   for example, /.*?a(*PRUNE)b/ with the subject "aab", which matches "ab",
   i.e. not at the start of a line. There is also an option that disables this
   optimization. */

   else if (op == OP_TYPESTAR || op == OP_TYPEMINSTAR || op == OP_TYPEPOSSTAR)
     {
     if (scode[1] != OP_ANY || (bracket_map & cb->backref_map) != 0 ||
         atomcount > 0 || cb->had_pruneorskip || inassert || !dotstar_anchor)
       return FALSE;
     }

   /* Check for explicit circumflex; anything else gives a FALSE result. Note
   in particular that this includes atomic brackets OP_ONCE because the number
   of characters matched by .* cannot be adjusted inside them. */

   else if (op != OP_CIRC && op != OP_CIRCM) return FALSE;

   /* Move on to the next alternative */

   code += GET(code, 1);
   }
while (*code == OP_ALT);  /* Loop for each alternative */
return TRUE;
}



/*************************************************
*   Scan compiled regex for recursion reference  *
*************************************************/

/* This function scans through a compiled pattern until it finds an instance of
OP_RECURSE.

Arguments:
  code        points to start of expression
  utf         TRUE in UTF mode

Returns:      pointer to the opcode for OP_RECURSE, or NULL if not found
*/

static PCRE2_UCHAR *
find_recurse(PCRE2_UCHAR *code, BOOL utf)
{
for (;;)
  {
  PCRE2_UCHAR c = *code;
  if (c == OP_END) return NULL;
  if (c == OP_RECURSE) return code;

  /* XCLASS is used for classes that cannot be represented just by a bit map.
  This includes negated single high-valued characters. ECLASS is used for
  classes that use set operations internally. CALLOUT_STR is used for
  callouts with string arguments. In each case the length in the table is
  zero; the actual length is stored in the compiled code. */

  if (c == OP_XCLASS || c == OP_ECLASS) code += GET(code, 1);
  else if (c == OP_CALLOUT_STR) code += GET(code, 1 + 2*LINK_SIZE);

  /* Otherwise, we can get the item's length from the table, except that for
  repeated character types, we have to test for \p and \P, which have an extra
  two code units of parameters, and for MARK/PRUNE/SKIP/THEN with an argument,
  we must add in its length. */

  else
    {
    switch(c)
      {
      case OP_TYPESTAR:
      case OP_TYPEMINSTAR:
      case OP_TYPEPLUS:
      case OP_TYPEMINPLUS:
      case OP_TYPEQUERY:
      case OP_TYPEMINQUERY:
      case OP_TYPEPOSSTAR:
      case OP_TYPEPOSPLUS:
      case OP_TYPEPOSQUERY:
      if (code[1] == OP_PROP || code[1] == OP_NOTPROP) code += 2;
      break;

      case OP_TYPEPOSUPTO:
      case OP_TYPEUPTO:
      case OP_TYPEMINUPTO:
      case OP_TYPEEXACT:
      if (code[1 + IMM2_SIZE] == OP_PROP || code[1 + IMM2_SIZE] == OP_NOTPROP)
        code += 2;
      break;

      case OP_MARK:
      case OP_COMMIT_ARG:
      case OP_PRUNE_ARG:
      case OP_SKIP_ARG:
      case OP_THEN_ARG:
      code += code[1];
      break;
      }

    /* Add in the fixed length from the table */

    code += PRIV(OP_lengths)[c];

    /* In UTF-8 and UTF-16 modes, opcodes that are followed by a character may
    be followed by a multi-unit character. The length in the table is a
    minimum, so we have to arrange to skip the extra units. */

#ifdef MAYBE_UTF_MULTI
    if (utf) switch(c)
      {
      case OP_CHAR:
      case OP_CHARI:
      case OP_NOT:
      case OP_NOTI:
      case OP_EXACT:
      case OP_EXACTI:
      case OP_NOTEXACT:
      case OP_NOTEXACTI:
      case OP_UPTO:
      case OP_UPTOI:
      case OP_NOTUPTO:
      case OP_NOTUPTOI:
      case OP_MINUPTO:
      case OP_MINUPTOI:
      case OP_NOTMINUPTO:
      case OP_NOTMINUPTOI:
      case OP_POSUPTO:
      case OP_POSUPTOI:
      case OP_NOTPOSUPTO:
      case OP_NOTPOSUPTOI:
      case OP_STAR:
      case OP_STARI:
      case OP_NOTSTAR:
      case OP_NOTSTARI:
      case OP_MINSTAR:
      case OP_MINSTARI:
      case OP_NOTMINSTAR:
      case OP_NOTMINSTARI:
      case OP_POSSTAR:
      case OP_POSSTARI:
      case OP_NOTPOSSTAR:
      case OP_NOTPOSSTARI:
      case OP_PLUS:
      case OP_PLUSI:
      case OP_NOTPLUS:
      case OP_NOTPLUSI:
      case OP_MINPLUS:
      case OP_MINPLUSI:
      case OP_NOTMINPLUS:
      case OP_NOTMINPLUSI:
      case OP_POSPLUS:
      case OP_POSPLUSI:
      case OP_NOTPOSPLUS:
      case OP_NOTPOSPLUSI:
      case OP_QUERY:
      case OP_QUERYI:
      case OP_NOTQUERY:
      case OP_NOTQUERYI:
      case OP_MINQUERY:
      case OP_MINQUERYI:
      case OP_NOTMINQUERY:
      case OP_NOTMINQUERYI:
      case OP_POSQUERY:
      case OP_POSQUERYI:
      case OP_NOTPOSQUERY:
      case OP_NOTPOSQUERYI:
      if (HAS_EXTRALEN(code[-1])) code += GET_EXTRALEN(code[-1]);
      break;
      }
#else
    (void)(utf);  /* Keep compiler happy by referencing function argument */
#endif  /* MAYBE_UTF_MULTI */
    }
  }
}



/*************************************************
*    Check for asserted fixed first code unit    *
*************************************************/

/* During compilation, the "first code unit" settings from forward assertions
are discarded, because they can cause conflicts with actual literals that
follow. However, if we end up without a first code unit setting for an
unanchored pattern, it is worth scanning the regex to see if there is an
initial asserted first code unit. If all branches start with the same asserted
code unit, or with a non-conditional bracket all of whose alternatives start
with the same asserted code unit (recurse ad lib), then we return that code
unit, with the flags set to zero or REQ_CASELESS; otherwise return zero with
REQ_NONE in the flags.

Arguments:
  code       points to start of compiled pattern
  flags      points to the first code unit flags
  inassert   non-zero if in an assertion

Returns:     the fixed first code unit, or 0 with REQ_NONE in flags
*/

static uint32_t
find_firstassertedcu(PCRE2_SPTR code, uint32_t *flags, uint32_t inassert)
{
uint32_t c = 0;
uint32_t cflags = REQ_NONE;

*flags = REQ_NONE;
do {
   uint32_t d;
   uint32_t dflags;
   int xl = (*code == OP_CBRA || *code == OP_SCBRA ||
             *code == OP_CBRAPOS || *code == OP_SCBRAPOS)? IMM2_SIZE:0;
   PCRE2_SPTR scode = first_significant_code(code + 1+LINK_SIZE + xl, TRUE);
   PCRE2_UCHAR op = *scode;

   switch(op)
     {
     default:
     return 0;

     case OP_BRA:
     case OP_BRAPOS:
     case OP_CBRA:
     case OP_SCBRA:
     case OP_CBRAPOS:
     case OP_SCBRAPOS:
     case OP_ASSERT:
     case OP_ASSERT_NA:
     case OP_ONCE:
     case OP_SCRIPT_RUN:
     d = find_firstassertedcu(scode, &dflags, inassert +
       ((op == OP_ASSERT || op == OP_ASSERT_NA)?1:0));
     if (dflags >= REQ_NONE) return 0;
     if (cflags >= REQ_NONE) { c = d; cflags = dflags; }
       else if (c != d || cflags != dflags) return 0;
     break;

     case OP_EXACT:
     scode += IMM2_SIZE;
     /* Fall through */

     case OP_CHAR:
     case OP_PLUS:
     case OP_MINPLUS:
     case OP_POSPLUS:
     if (inassert == 0) return 0;
     if (cflags >= REQ_NONE) { c = scode[1]; cflags = 0; }
       else if (c != scode[1]) return 0;
     break;

     case OP_EXACTI:
     scode += IMM2_SIZE;
     /* Fall through */

     case OP_CHARI:
     case OP_PLUSI:
     case OP_MINPLUSI:
     case OP_POSPLUSI:
     if (inassert == 0) return 0;

     /* If the character is more than one code unit long, we cannot set its
     first code unit when matching caselessly. Later scanning may pick up
     multiple code units. */

#ifdef SUPPORT_UNICODE
#if PCRE2_CODE_UNIT_WIDTH == 8
     if (scode[1] >= 0x80) return 0;
#elif PCRE2_CODE_UNIT_WIDTH == 16
     if (scode[1] >= 0xd800 && scode[1] <= 0xdfff) return 0;
#endif
#endif

     if (cflags >= REQ_NONE) { c = scode[1]; cflags = REQ_CASELESS; }
       else if (c != scode[1]) return 0;
     break;
     }

   code += GET(code, 1);
   }
while (*code == OP_ALT);

*flags = cflags;
return c;
}



/*************************************************
*     Add an entry to the name/number table      *
*************************************************/

/* This function is called between compiling passes to add an entry to the
name/number table, maintaining alphabetical order. Checking for permitted
and forbidden duplicates has already been done.

Arguments:
  cb           the compile data block
  name         the name to add
  length       the length of the name
  groupno      the group number
  tablecount   the count of names in the table so far

Returns:       nothing
*/

static void
add_name_to_table(compile_block *cb, PCRE2_SPTR name, int length,
  unsigned int groupno, uint32_t tablecount)
{
uint32_t i;
PCRE2_UCHAR *slot = cb->name_table;

for (i = 0; i < tablecount; i++)
  {
  int crc = memcmp(name, slot+IMM2_SIZE, CU2BYTES(length));
  if (crc == 0 && slot[IMM2_SIZE+length] != 0)
    crc = -1; /* Current name is a substring */

  /* Make space in the table and break the loop for an earlier name. For a
  duplicate or later name, carry on. We do this for duplicates so that in the
  simple case (when ?(| is not used) they are in order of their numbers. In all
  cases they are in the order in which they appear in the pattern. */

  if (crc < 0)
    {
    (void)memmove(slot + cb->name_entry_size, slot,
      CU2BYTES((tablecount - i) * cb->name_entry_size));
    break;
    }

  /* Continue the loop for a later or duplicate name */

  slot += cb->name_entry_size;
  }

PUT2(slot, 0, groupno);
memcpy(slot + IMM2_SIZE, name, CU2BYTES(length));

/* Add a terminating zero and fill the rest of the slot with zeroes so that
the memory is all initialized. Otherwise valgrind moans about uninitialized
memory when saving serialized compiled patterns. */

memset(slot + IMM2_SIZE + length, 0,
  CU2BYTES(cb->name_entry_size - length - IMM2_SIZE));
}



/*************************************************
*             Skip in parsed pattern             *
*************************************************/

/* This function is called to skip parts of the parsed pattern when finding the
length of a lookbehind branch. It is called after (*ACCEPT) and (*FAIL) to find
the end of the branch, it is called to skip over an internal lookaround or
(DEFINE) group, and it is also called to skip to the end of a class, during
which it will never encounter nested groups (but there's no need to have
special code for that).

When called to find the end of a branch or group, pptr must point to the first
meta code inside the branch, not the branch-starting code. In other cases it
can point to the item that causes the function to be called.

Arguments:
  pptr       current pointer to skip from
  skiptype   PSKIP_CLASS when skipping to end of class
             PSKIP_ALT when META_ALT ends the skip
             PSKIP_KET when only META_KET ends the skip

Returns:     new value of pptr
             NULL if META_END is reached - should never occur
               or for an unknown meta value - likewise
*/

static uint32_t *
parsed_skip(uint32_t *pptr, uint32_t skiptype)
{
uint32_t nestlevel = 0;

for (;; pptr++)
  {
  uint32_t meta = META_CODE(*pptr);

  switch(meta)
    {
    default:  /* Just skip over most items */
    if (meta < META_END) continue;  /* Literal */
    break;

    case META_END:

    /* The parsed regex is malformed; we have reached the end and did
    not find the end of the construct which we are skipping over. */

    PCRE2_DEBUG_UNREACHABLE();
    return NULL;

    /* The data for these items is variable in length. */

    case META_BACKREF:  /* Offset is present only if group >= 10 */
    if (META_DATA(*pptr) >= 10) pptr += SIZEOFFSET;
    break;

    case META_ESCAPE:
    if (*pptr - META_ESCAPE == ESC_P || *pptr - META_ESCAPE == ESC_p)
      pptr += 1;     /* Skip prop data */
    break;

    case META_MARK:     /* Add the length of the name. */
    case META_COMMIT_ARG:
    case META_PRUNE_ARG:
    case META_SKIP_ARG:
    case META_THEN_ARG:
    pptr += pptr[1];
    break;

    /* These are the "active" items in this loop. */

    case META_CLASS_END:
    if (skiptype == PSKIP_CLASS) return pptr;
    break;

    case META_ATOMIC:
    case META_CAPTURE:
    case META_COND_ASSERT:
    case META_COND_DEFINE:
    case META_COND_NAME:
    case META_COND_NUMBER:
    case META_COND_RNAME:
    case META_COND_RNUMBER:
    case META_COND_VERSION:
    case META_SCS:
    case META_LOOKAHEAD:
    case META_LOOKAHEADNOT:
    case META_LOOKAHEAD_NA:
    case META_LOOKBEHIND:
    case META_LOOKBEHINDNOT:
    case META_LOOKBEHIND_NA:
    case META_NOCAPTURE:
    case META_SCRIPT_RUN:
    nestlevel++;
    break;

    case META_ALT:
    if (nestlevel == 0 && skiptype == PSKIP_ALT) return pptr;
    break;

    case META_KET:
    if (nestlevel == 0) return pptr;
    nestlevel--;
    break;
    }

  /* The extra data item length for each meta is in a table. */

  meta = (meta >> 16) & 0x7fff;
  if (meta >= sizeof(meta_extra_lengths)) return NULL;
  pptr += meta_extra_lengths[meta];
  }

PCRE2_UNREACHABLE(); /* Control never reaches here */
}



/*************************************************
*       Find length of a parsed group            *
*************************************************/

/* This is called for nested groups within a branch of a lookbehind whose
length is being computed. On entry, the pointer must be at the first element
after the group initializing code. On exit it points to OP_KET. Caching is used
to improve processing speed when the same capturing group occurs many times.

Arguments:
  pptrptr     pointer to pointer in the parsed pattern
  minptr      where to return the minimum length
  isinline    FALSE if a reference or recursion; TRUE for inline group
  errcodeptr  pointer to the errorcode
  lcptr       pointer to the loop counter
  group       number of captured group or -1 for a non-capturing group
  recurses    chain of recurse_check to catch mutual recursion
  cb          pointer to the compile data

Returns:      the maximum group length or a negative number
*/

static int
get_grouplength(uint32_t **pptrptr, int *minptr, BOOL isinline, int *errcodeptr,
  int *lcptr, int group, parsed_recurse_check *recurses, compile_block *cb)
{
uint32_t *gi = cb->groupinfo + 2 * group;
int branchlength, branchminlength;
int grouplength = -1;
int groupminlength = INT_MAX;

/* The cache can be used only if there is no possibility of there being two
groups with the same number. We do not need to set the end pointer for a group
that is being processed as a back reference or recursion, but we must do so for
an inline group. */

if (group > 0 && (cb->external_flags & PCRE2_DUPCAPUSED) == 0)
  {
  uint32_t groupinfo = gi[0];
  if ((groupinfo & GI_NOT_FIXED_LENGTH) != 0) return -1;
  if ((groupinfo & GI_SET_FIXED_LENGTH) != 0)
    {
    if (isinline) *pptrptr = parsed_skip(*pptrptr, PSKIP_KET);
    *minptr = gi[1];
    return groupinfo & GI_FIXED_LENGTH_MASK;
    }
  }

/* Scan the group. In this case we find the end pointer of necessity. */

for(;;)
  {
  branchlength = get_branchlength(pptrptr, &branchminlength, errcodeptr, lcptr,
    recurses, cb);
  if (branchlength < 0) goto ISNOTFIXED;
  if (branchlength > grouplength) grouplength = branchlength;
  if (branchminlength < groupminlength) groupminlength = branchminlength;
  if (**pptrptr == META_KET) break;
  *pptrptr += 1;   /* Skip META_ALT */
  }

if (group > 0)
  {
  gi[0] |= (uint32_t)(GI_SET_FIXED_LENGTH | grouplength);
  gi[1] = groupminlength;
  }

*minptr = groupminlength;
return grouplength;

ISNOTFIXED:
if (group > 0) gi[0] |= GI_NOT_FIXED_LENGTH;
return -1;
}



/*************************************************
*        Find length of a parsed branch          *
*************************************************/

/* Return fixed maximum and minimum lengths for a branch in a lookbehind,
giving an error if the length is not limited. On entry, *pptrptr points to the
first element inside the branch. On exit it is set to point to the ALT or KET.

Arguments:
  pptrptr     pointer to pointer in the parsed pattern
  minptr      where to return the minimum length
  errcodeptr  pointer to error code
  lcptr       pointer to loop counter
  recurses    chain of recurse_check to catch mutual recursion
  cb          pointer to compile block

Returns:      the maximum length, or a negative value on error
*/

static int
get_branchlength(uint32_t **pptrptr, int *minptr, int *errcodeptr, int *lcptr,
  parsed_recurse_check *recurses, compile_block *cb)
{
int branchlength = 0;
int branchminlength = 0;
int grouplength, groupminlength;
uint32_t lastitemlength = 0;
uint32_t lastitemminlength = 0;
uint32_t *pptr = *pptrptr;
PCRE2_SIZE offset;
parsed_recurse_check this_recurse;

/* A large and/or complex regex can take too long to process. This can happen
more often when (?| groups are present in the pattern because their length
cannot be cached. */

if ((*lcptr)++ > 2000)
  {
  *errcodeptr = ERR35;  /* Lookbehind is too complicated */
  return -1;
  }

/* Scan the branch, accumulating the length. */

for (;; pptr++)
  {
  parsed_recurse_check *r;
  uint32_t *gptr, *gptrend;
  uint32_t escape;
  uint32_t min, max;
  uint32_t group = 0;
  uint32_t itemlength = 0;
  uint32_t itemminlength = 0;

  if (*pptr < META_END)
    {
    itemlength = itemminlength = 1;
    }

  else switch (META_CODE(*pptr))
    {
    case META_KET:
    case META_ALT:
    goto EXIT;

    /* (*ACCEPT) and (*FAIL) terminate the branch, but we must skip to the
    actual termination. */

    case META_ACCEPT:
    case META_FAIL:
    pptr = parsed_skip(pptr, PSKIP_ALT);
    if (pptr == NULL) goto PARSED_SKIP_FAILED;
    goto EXIT;

    case META_MARK:
    case META_COMMIT_ARG:
    case META_PRUNE_ARG:
    case META_SKIP_ARG:
    case META_THEN_ARG:
    pptr += pptr[1] + 1;
    break;

    case META_CIRCUMFLEX:
    case META_COMMIT:
    case META_DOLLAR:
    case META_PRUNE:
    case META_SKIP:
    case META_THEN:
    break;

    case META_OPTIONS:
    pptr += 2;
    break;

    case META_BIGVALUE:
    itemlength = itemminlength = 1;
    pptr += 1;
    break;

    case META_CLASS:
    case META_CLASS_NOT:
    itemlength = itemminlength = 1;
    pptr = parsed_skip(pptr, PSKIP_CLASS);
    if (pptr == NULL) goto PARSED_SKIP_FAILED;
    break;

    case META_CLASS_EMPTY_NOT:
    case META_DOT:
    itemlength = itemminlength = 1;
    break;

    case META_CALLOUT_NUMBER:
    pptr += 3;
    break;

    case META_CALLOUT_STRING:
    pptr += 3 + SIZEOFFSET;
    break;

    /* Only some escapes consume a character. Of those, \R can match one or two
    characters, but \X is never allowed because it matches an unknown number of
    characters. \C is allowed only in 32-bit and non-UTF 8/16-bit modes. */

    case META_ESCAPE:
    escape = META_DATA(*pptr);
    if (escape == ESC_X) return -1;
    if (escape == ESC_R)
      {
      itemminlength = 1;
      itemlength = 2;
      }
    else if (escape > ESC_b && escape < ESC_Z)
      {
#if PCRE2_CODE_UNIT_WIDTH != 32
      if ((cb->external_options & PCRE2_UTF) != 0 && escape == ESC_C)
        {
        *errcodeptr = ERR36;
        return -1;
        }
#endif
      itemlength = itemminlength = 1;
      if (escape == ESC_p || escape == ESC_P) pptr++;  /* Skip prop data */
      }
    break;

    /* Lookaheads do not contribute to the length of this branch, but they may
    contain lookbehinds within them whose lengths need to be set. */

    case META_LOOKAHEAD:
    case META_LOOKAHEADNOT:
    case META_LOOKAHEAD_NA:
    case META_SCS:
    *errcodeptr = check_lookbehinds(pptr + 1, &pptr, recurses, cb, lcptr);
    if (*errcodeptr != 0) return -1;

    /* Ignore any qualifiers that follow a lookahead assertion. */

    switch (pptr[1])
      {
      case META_ASTERISK:
      case META_ASTERISK_PLUS:
      case META_ASTERISK_QUERY:
      case META_PLUS:
      case META_PLUS_PLUS:
      case META_PLUS_QUERY:
      case META_QUERY:
      case META_QUERY_PLUS:
      case META_QUERY_QUERY:
      pptr++;
      break;

      case META_MINMAX:
      case META_MINMAX_PLUS:
      case META_MINMAX_QUERY:
      pptr += 3;
      break;

      default:
      break;
      }
    break;

    /* A nested lookbehind does not contribute any length to this lookbehind,
    but must itself be checked and have its lengths set. Note that
    set_lookbehind_lengths() updates pptr, leaving it pointing to the final ket
    of the group, so no need to update it here. */

    case META_LOOKBEHIND:
    case META_LOOKBEHINDNOT:
    case META_LOOKBEHIND_NA:
    if (!set_lookbehind_lengths(&pptr, errcodeptr, lcptr, recurses, cb))
      return -1;
    break;

    /* Back references and recursions are handled by very similar code. At this
    stage, the names generated in the parsing pass are available, but the main
    name table has not yet been created. So for the named varieties, scan the
    list of names in order to get the number of the first one in the pattern,
    and whether or not this name is duplicated. */

    case META_BACKREF_BYNAME:
    if ((cb->external_options & PCRE2_MATCH_UNSET_BACKREF) != 0)
      goto ISNOTFIXED;
    /* Fall through */

    case META_RECURSE_BYNAME:
      {
      int i;
      PCRE2_SPTR name;
      BOOL is_dupname = FALSE;
      named_group *ng = cb->named_groups;
      uint32_t meta_code = META_CODE(*pptr);
      uint32_t length = *(++pptr);

      GETPLUSOFFSET(offset, pptr);
      name = cb->start_pattern + offset;
      for (i = 0; i < cb->names_found; i++, ng++)
        {
        if (length == ng->length && PRIV(strncmp)(name, ng->name, length) == 0)
          {
          group = ng->number;
          is_dupname = ng->isdup;
          break;
          }
        }

      if (group == 0)
        {
        *errcodeptr = ERR15;  /* Non-existent subpattern */
        cb->erroroffset = offset;
        return -1;
        }

      /* A numerical back reference can be fixed length if duplicate capturing
      groups are not being used. A non-duplicate named back reference can also
      be handled. */

      if (meta_code == META_RECURSE_BYNAME ||
          (!is_dupname && (cb->external_flags & PCRE2_DUPCAPUSED) == 0))
        goto RECURSE_OR_BACKREF_LENGTH;  /* Handle as a numbered version. */
      }
    goto ISNOTFIXED;                     /* Duplicate name or number */

    /* The offset values for back references < 10 are in a separate vector
    because otherwise they would use more than two parsed pattern elements on
    64-bit systems. */

    case META_BACKREF:
    if ((cb->external_options & PCRE2_MATCH_UNSET_BACKREF) != 0 ||
        (cb->external_flags & PCRE2_DUPCAPUSED) != 0)
      goto ISNOTFIXED;
    group = META_DATA(*pptr);
    if (group < 10)
      {
      offset = cb->small_ref_offset[group];
      goto RECURSE_OR_BACKREF_LENGTH;
      }

    /* Fall through */
    /* For groups >= 10 - picking up group twice does no harm. */

    /* A true recursion implies not fixed length, but a subroutine call may
    be OK. Back reference "recursions" are also failed. */

    case META_RECURSE:
    group = META_DATA(*pptr);
    GETPLUSOFFSET(offset, pptr);

    RECURSE_OR_BACKREF_LENGTH:
    if (group > cb->bracount)
      {
      cb->erroroffset = offset;
      *errcodeptr = ERR15;  /* Non-existent subpattern */
      return -1;
      }
    if (group == 0) goto ISNOTFIXED;  /* Local recursion */
    for (gptr = cb->parsed_pattern; *gptr != META_END; gptr++)
      {
      if (META_CODE(*gptr) == META_BIGVALUE) gptr++;
        else if (*gptr == (META_CAPTURE | group)) break;
      }

    /* We must start the search for the end of the group at the first meta code
    inside the group. Otherwise it will be treated as an enclosed group. */

    gptrend = parsed_skip(gptr + 1, PSKIP_KET);
    if (gptrend == NULL) goto PARSED_SKIP_FAILED;
    if (pptr > gptr && pptr < gptrend) goto ISNOTFIXED;  /* Local recursion */
    for (r = recurses; r != NULL; r = r->prev) if (r->groupptr == gptr) break;
    if (r != NULL) goto ISNOTFIXED;   /* Mutual recursion */
    this_recurse.prev = recurses;
    this_recurse.groupptr = gptr;

    /* We do not need to know the position of the end of the group, that is,
    gptr is not used after the call to get_grouplength(). Setting the second
    argument FALSE stops it scanning for the end when the length can be found
    in the cache. */

    gptr++;
    grouplength = get_grouplength(&gptr, &groupminlength, FALSE, errcodeptr,
      lcptr, group, &this_recurse, cb);
    if (grouplength < 0)
      {
      if (*errcodeptr == 0) goto ISNOTFIXED;
      return -1;  /* Error already set */
      }
    itemlength = grouplength;
    itemminlength = groupminlength;
    break;

    /* A (DEFINE) group is never obeyed inline and so it does not contribute to
    the length of this branch. Skip from the following item to the next
    unpaired ket. */

    case META_COND_DEFINE:
    pptr = parsed_skip(pptr + 1, PSKIP_KET);
    break;

    /* Check other nested groups - advance past the initial data for each type
    and then seek a fixed length with get_grouplength(). */

    case META_COND_NAME:
    case META_COND_NUMBER:
    case META_COND_RNAME:
    case META_COND_RNUMBER:
    pptr += 2 + SIZEOFFSET;
    goto CHECK_GROUP;

    case META_COND_ASSERT:
    pptr += 1;
    goto CHECK_GROUP;

    case META_COND_VERSION:
    pptr += 4;
    goto CHECK_GROUP;

    case META_CAPTURE:
    group = META_DATA(*pptr);
    /* Fall through */

    case META_ATOMIC:
    case META_NOCAPTURE:
    case META_SCRIPT_RUN:
    pptr++;
    CHECK_GROUP:
    grouplength = get_grouplength(&pptr, &groupminlength, TRUE, errcodeptr,
      lcptr, group, recurses, cb);
    if (grouplength < 0) return -1;
    itemlength = grouplength;
    itemminlength = groupminlength;
    break;

    case META_QUERY:
    case META_QUERY_PLUS:
    case META_QUERY_QUERY:
    min = 0;
    max = 1;
    goto REPETITION;

    /* Exact repetition is OK; variable repetition is not. A repetition of zero
    must subtract the length that has already been added. */

    case META_MINMAX:
    case META_MINMAX_PLUS:
    case META_MINMAX_QUERY:
    min = pptr[1];
    max = pptr[2];
    pptr += 2;

    REPETITION:
    if (max != REPEAT_UNLIMITED)
      {
      if (lastitemlength != 0 &&  /* Should not occur, but just in case */
          max != 0 &&
          (INT_MAX - branchlength)/lastitemlength < max - 1)
        {
        *errcodeptr = ERR87;  /* Integer overflow; lookbehind too big */
        return -1;
        }
      if (min == 0) branchminlength -= lastitemminlength;
        else itemminlength = (min - 1) * lastitemminlength;
      if (max == 0) branchlength -= lastitemlength;
        else itemlength = (max - 1) * lastitemlength;
      break;
      }
    /* Fall through */

    /* Any other item means this branch does not have a fixed length. */

    default:
    ISNOTFIXED:
    *errcodeptr = ERR25;   /* Not fixed length */
    return -1;
    }

  /* Add the item length to the branchlength, checking for integer overflow and
  for the branch length exceeding the overall limit. Later, if there is at
  least one variable-length branch in the group, there is a test for the
  (smaller) variable-length branch length limit. */

  if (INT_MAX - branchlength < (int)itemlength ||
      (branchlength += itemlength) > LOOKBEHIND_MAX)
    {
    *errcodeptr = ERR87;
    return -1;
    }

  branchminlength += itemminlength;

  /* Save this item length for use if the next item is a quantifier. */

  lastitemlength = itemlength;
  lastitemminlength = itemminlength;
  }

EXIT:
*pptrptr = pptr;
*minptr = branchminlength;
return branchlength;

PARSED_SKIP_FAILED:
PCRE2_DEBUG_UNREACHABLE();
*errcodeptr = ERR90;  /* Unhandled META code - internal error */
return -1;
}



/*************************************************
*        Set lengths in a lookbehind             *
*************************************************/

/* This function is called for each lookbehind, to set the lengths in its
branches. An error occurs if any branch does not have a limited maximum length
that is less than the limit (65535). On exit, the pointer must be left on the
final ket.

The function also maintains the max_lookbehind value. Any lookbehind branch
that contains a nested lookbehind may actually look further back than the
length of the branch. The additional amount is passed back from
get_branchlength() as an "extra" value.

Arguments:
  pptrptr     pointer to pointer in the parsed pattern
  errcodeptr  pointer to error code
  lcptr       pointer to loop counter
  recurses    chain of recurse_check to catch mutual recursion
  cb          pointer to compile block

Returns:      TRUE if all is well
              FALSE otherwise, with error code and offset set
*/

static BOOL
set_lookbehind_lengths(uint32_t **pptrptr, int *errcodeptr, int *lcptr,
  parsed_recurse_check *recurses, compile_block *cb)
{
PCRE2_SIZE offset;
uint32_t *bptr = *pptrptr;
uint32_t *gbptr = bptr;
int maxlength = 0;
int minlength = INT_MAX;
BOOL variable = FALSE;

READPLUSOFFSET(offset, bptr);  /* Offset for error messages */
*pptrptr += SIZEOFFSET;

/* Each branch can have a different maximum length, but we can keep only a
single minimum for the whole group, because there's nowhere to save individual
values in the META_ALT item. */

do
  {
  int branchlength, branchminlength;

  *pptrptr += 1;
  branchlength = get_branchlength(pptrptr, &branchminlength, errcodeptr, lcptr,
    recurses, cb);

  if (branchlength < 0)
    {
    /* The errorcode and offset may already be set from a nested lookbehind. */
    if (*errcodeptr == 0) *errcodeptr = ERR25;
    if (cb->erroroffset == PCRE2_UNSET) cb->erroroffset = offset;
    return FALSE;
    }

  if (branchlength != branchminlength) variable = TRUE;
  if (branchminlength < minlength) minlength = branchminlength;
  if (branchlength > maxlength) maxlength = branchlength;
  if (branchlength > cb->max_lookbehind) cb->max_lookbehind = branchlength;
  *bptr |= branchlength;  /* branchlength never more than 65535 */
  bptr = *pptrptr;
  }
while (META_CODE(*bptr) == META_ALT);

/* If any branch is of variable length, the whole lookbehind is of variable
length. If the maximum length of any branch exceeds the maximum for variable
lookbehinds, give an error. Otherwise, the minimum length is set in the word
that follows the original group META value. For a fixed-length lookbehind, this
is set to LOOKBEHIND_MAX, to indicate that each branch is of a fixed (but
possibly different) length. */

if (variable)
  {
  gbptr[1] = minlength;
  if ((PCRE2_SIZE)maxlength > cb->max_varlookbehind)
    {
    *errcodeptr = ERR100;
    cb->erroroffset = offset;
    return FALSE;
    }
  }
else gbptr[1] = LOOKBEHIND_MAX;

return TRUE;
}



/*************************************************
*         Check parsed pattern lookbehinds       *
*************************************************/

/* This function is called at the end of parsing a pattern if any lookbehinds
were encountered. It scans the parsed pattern for them, calling
set_lookbehind_lengths() for each one. At the start, the errorcode is zero and
the error offset is marked unset. The enables the functions above not to
override settings from deeper nestings.

This function is called recursively from get_branchlength() for lookaheads in
order to process any lookbehinds that they may contain. It stops when it hits a
non-nested closing parenthesis in this case, returning a pointer to it.

Arguments
  pptr      points to where to start (start of pattern or start of lookahead)
  retptr    if not NULL, return the ket pointer here
  recurses  chain of recurse_check to catch mutual recursion
  cb        points to the compile block
  lcptr     points to loop counter

Returns:    0 on success, or an errorcode (cb->erroroffset will be set)
*/

static int
check_lookbehinds(uint32_t *pptr, uint32_t **retptr,
  parsed_recurse_check *recurses, compile_block *cb, int *lcptr)
{
int errorcode = 0;
int nestlevel = 0;

cb->erroroffset = PCRE2_UNSET;

for (; *pptr != META_END; pptr++)
  {
  if (*pptr < META_END) continue;  /* Literal */

  switch (META_CODE(*pptr))
    {
    default:

    /* The following erroroffset is a bogus but safe value. This branch should
    be avoided by providing a proper implementation for all supported cases
    below. */

    PCRE2_DEBUG_UNREACHABLE();
    cb->erroroffset = 0;
    return ERR70;  /* Unrecognized meta code */

    case META_ESCAPE:
    if (*pptr - META_ESCAPE == ESC_P || *pptr - META_ESCAPE == ESC_p)
      pptr += 1;    /* Skip prop data */
    break;

    case META_KET:
    if (--nestlevel < 0)
      {
      if (retptr != NULL) *retptr = pptr;
      return 0;
      }
    break;

    case META_ATOMIC:
    case META_CAPTURE:
    case META_COND_ASSERT:
    case META_SCS:
    case META_LOOKAHEAD:
    case META_LOOKAHEADNOT:
    case META_LOOKAHEAD_NA:
    case META_NOCAPTURE:
    case META_SCRIPT_RUN:
    nestlevel++;
    break;

    case META_ACCEPT:
    case META_ALT:
    case META_ASTERISK:
    case META_ASTERISK_PLUS:
    case META_ASTERISK_QUERY:
    case META_BACKREF:
    case META_CIRCUMFLEX:
    case META_CLASS:
    case META_CLASS_EMPTY:
    case META_CLASS_EMPTY_NOT:
    case META_CLASS_END:
    case META_CLASS_NOT:
    case META_COMMIT:
    case META_DOLLAR:
    case META_DOT:
    case META_FAIL:
    case META_PLUS:
    case META_PLUS_PLUS:
    case META_PLUS_QUERY:
    case META_PRUNE:
    case META_QUERY:
    case META_QUERY_PLUS:
    case META_QUERY_QUERY:
    case META_RANGE_ESCAPED:
    case META_RANGE_LITERAL:
    case META_SKIP:
    case META_THEN:
    break;

    case META_OFFSET:
    case META_RECURSE:
    pptr += SIZEOFFSET;
    break;

    case META_BACKREF_BYNAME:
    case META_RECURSE_BYNAME:
    pptr += 1 + SIZEOFFSET;
    break;

    case META_COND_DEFINE:
    pptr += SIZEOFFSET;
    nestlevel++;
    break;

    case META_COND_NAME:
    case META_COND_NUMBER:
    case META_COND_RNAME:
    case META_COND_RNUMBER:
    pptr += 1 + SIZEOFFSET;
    nestlevel++;
    break;

    case META_COND_VERSION:
    pptr += 3;
    nestlevel++;
    break;

    case META_CALLOUT_STRING:
    pptr += 3 + SIZEOFFSET;
    break;

    case META_BIGVALUE:
    case META_POSIX:
    case META_POSIX_NEG:
    case META_SCS_NAME:
    case META_SCS_NUMBER:
    pptr += 1;
    break;

    case META_MINMAX:
    case META_MINMAX_QUERY:
    case META_MINMAX_PLUS:
    case META_OPTIONS:
    pptr += 2;
    break;

    case META_CALLOUT_NUMBER:
    pptr += 3;
    break;

    case META_MARK:
    case META_COMMIT_ARG:
    case META_PRUNE_ARG:
    case META_SKIP_ARG:
    case META_THEN_ARG:
    pptr += 1 + pptr[1];
    break;

    /* Note that set_lookbehind_lengths() updates pptr, leaving it pointing to
    the final ket of the group, so no need to update it here. */

    case META_LOOKBEHIND:
    case META_LOOKBEHINDNOT:
    case META_LOOKBEHIND_NA:
    if (!set_lookbehind_lengths(&pptr, &errorcode, lcptr, recurses, cb))
      return errorcode;
    break;
    }
  }

return 0;
}



/*************************************************
*     External function to compile a pattern     *
*************************************************/

/* This function reads a regular expression in the form of a string and returns
a pointer to a block of store holding a compiled version of the expression.

Arguments:
  pattern       the regular expression
  patlen        the length of the pattern, or PCRE2_ZERO_TERMINATED
  options       option bits
  errorptr      pointer to errorcode
  erroroffset   pointer to error offset
  ccontext      points to a compile context or is NULL

Returns:        pointer to compiled data block, or NULL on error,
                with errorcode and erroroffset set
*/

PCRE2_EXP_DEFN pcre2_code * PCRE2_CALL_CONVENTION
pcre2_compile(PCRE2_SPTR pattern, PCRE2_SIZE patlen, uint32_t options,
   int *errorptr, PCRE2_SIZE *erroroffset, pcre2_compile_context *ccontext)
{
BOOL utf;                             /* Set TRUE for UTF mode */
BOOL ucp;                             /* Set TRUE for UCP mode */
BOOL has_lookbehind = FALSE;          /* Set TRUE if a lookbehind is found */
BOOL zero_terminated;                 /* Set TRUE for zero-terminated pattern */
pcre2_real_code *re = NULL;           /* What we will return */
compile_block cb;                     /* "Static" compile-time data */
const uint8_t *tables;                /* Char tables base pointer */

PCRE2_UCHAR *code;                    /* Current pointer in compiled code */
PCRE2_UCHAR * codestart;              /* Start of compiled code */
PCRE2_SPTR ptr;                       /* Current pointer in pattern */
uint32_t *pptr;                       /* Current pointer in parsed pattern */

PCRE2_SIZE length = 1;                /* Allow for final END opcode */
PCRE2_SIZE usedlength;                /* Actual length used */
PCRE2_SIZE re_blocksize;              /* Size of memory block */
PCRE2_SIZE parsed_size_needed;        /* Needed for parsed pattern */

uint32_t firstcuflags, reqcuflags;    /* Type of first/req code unit */
uint32_t firstcu, reqcu;              /* Value of first/req code unit */
uint32_t setflags = 0;                /* NL and BSR set flags */
uint32_t xoptions;                    /* Flags from context, modified */

uint32_t skipatstart;                 /* When checking (*UTF) etc */
uint32_t limit_heap  = UINT32_MAX;
uint32_t limit_match = UINT32_MAX;    /* Unset match limits */
uint32_t limit_depth = UINT32_MAX;

int newline = 0;                      /* Unset; can be set by the pattern */
int bsr = 0;                          /* Unset; can be set by the pattern */
int errorcode = 0;                    /* Initialize to avoid compiler warn */
int regexrc;                          /* Return from compile */

uint32_t i;                           /* Local loop counter */

/* Enable all optimizations by default. */
uint32_t optim_flags = ccontext != NULL ? ccontext->optimization_flags :
                                          PCRE2_OPTIMIZATION_ALL;

/* Comments at the head of this file explain about these variables. */

uint32_t stack_groupinfo[GROUPINFO_DEFAULT_SIZE];
uint32_t stack_parsed_pattern[PARSED_PATTERN_DEFAULT_SIZE];
named_group named_groups[NAMED_GROUP_LIST_SIZE];

/* The workspace is used in different ways in the different compiling phases.
It needs to be 16-bit aligned for the preliminary parsing scan. */

uint32_t c16workspace[C16_WORK_SIZE];
PCRE2_UCHAR *cworkspace = (PCRE2_UCHAR *)c16workspace;


/* -------------- Check arguments and set up the pattern ----------------- */

/* There must be error code and offset pointers. */

if (errorptr == NULL || erroroffset == NULL) return NULL;
*errorptr = ERR0;
*erroroffset = 0;

/* There must be a pattern, but NULL is allowed with zero length. */

if (pattern == NULL)
  {
  if (patlen == 0) pattern = (PCRE2_SPTR)""; else
    {
    *errorptr = ERR16;
    return NULL;
    }
  }

/* A NULL compile context means "use a default context" */

if (ccontext == NULL)
  ccontext = (pcre2_compile_context *)(&PRIV(default_compile_context));

/* PCRE2_MATCH_INVALID_UTF implies UTF */

if ((options & PCRE2_MATCH_INVALID_UTF) != 0) options |= PCRE2_UTF;

/* Check that all undefined public option bits are zero. */

if ((options & ~PUBLIC_COMPILE_OPTIONS) != 0 ||
    (ccontext->extra_options & ~PUBLIC_COMPILE_EXTRA_OPTIONS) != 0)
  {
  *errorptr = ERR17;
  return NULL;
  }

if ((options & PCRE2_LITERAL) != 0 &&
    ((options & ~PUBLIC_LITERAL_COMPILE_OPTIONS) != 0 ||
     (ccontext->extra_options & ~PUBLIC_LITERAL_COMPILE_EXTRA_OPTIONS) != 0))
  {
  *errorptr = ERR92;
  return NULL;
  }

/* A zero-terminated pattern is indicated by the special length value
PCRE2_ZERO_TERMINATED. Check for an overlong pattern. */

if ((zero_terminated = (patlen == PCRE2_ZERO_TERMINATED)))
  patlen = PRIV(strlen)(pattern);
(void)zero_terminated; /* Silence compiler; only used if Valgrind enabled */

if (patlen > ccontext->max_pattern_length)
  {
  *errorptr = ERR88;
  return NULL;
  }

/* Optimization flags in 'options' can override those in the compile context.
This is because some options to disable optimizations were added before the
optimization flags word existed, and we need to continue supporting them
for backwards compatibility. */

if ((options & PCRE2_NO_AUTO_POSSESS) != 0)
  optim_flags &= ~PCRE2_OPTIM_AUTO_POSSESS;
if ((options & PCRE2_NO_DOTSTAR_ANCHOR) != 0)
  optim_flags &= ~PCRE2_OPTIM_DOTSTAR_ANCHOR;
if ((options & PCRE2_NO_START_OPTIMIZE) != 0)
  optim_flags &= ~PCRE2_OPTIM_START_OPTIMIZE;

/* From here on, all returns from this function should end up going via the
EXIT label. */


/* ------------ Initialize the "static" compile data -------------- */

tables = (ccontext->tables != NULL)? ccontext->tables : PRIV(default_tables);

cb.lcc = tables + lcc_offset;          /* Individual */
cb.fcc = tables + fcc_offset;          /*   character */
cb.cbits = tables + cbits_offset;      /*      tables */
cb.ctypes = tables + ctypes_offset;

cb.assert_depth = 0;
cb.bracount = 0;
cb.cx = ccontext;
cb.dupnames = FALSE;
cb.end_pattern = pattern + patlen;
cb.erroroffset = 0;
cb.external_flags = 0;
cb.external_options = options;
cb.groupinfo = stack_groupinfo;
cb.had_recurse = FALSE;
cb.lastcapture = 0;
cb.max_lookbehind = 0;                               /* Max encountered */
cb.max_varlookbehind = ccontext->max_varlookbehind;  /* Limit */
cb.name_entry_size = 0;
cb.name_table = NULL;
cb.named_groups = named_groups;
cb.named_group_list_size = NAMED_GROUP_LIST_SIZE;
cb.names_found = 0;
cb.parens_depth = 0;
cb.parsed_pattern = stack_parsed_pattern;
cb.req_varyopt = 0;
cb.start_code = cworkspace;
cb.start_pattern = pattern;
cb.start_workspace = cworkspace;
cb.workspace_size = COMPILE_WORK_SIZE;
#ifdef SUPPORT_WIDE_CHARS
cb.cranges = NULL;
cb.next_cranges = NULL;
cb.char_lists_size = 0;
#endif

/* Maximum back reference and backref bitmap. The bitmap records up to 31 back
references to help in deciding whether (.*) can be treated as anchored or not.
*/

cb.top_backref = 0;
cb.backref_map = 0;

/* Escape sequences \1 to \9 are always back references, but as they are only
two characters long, only two elements can be used in the parsed_pattern
vector. The first contains the reference, and we'd like to use the second to
record the offset in the pattern, so that forward references to non-existent
groups can be diagnosed later with an offset. However, on 64-bit systems,
PCRE2_SIZE won't fit. Instead, we have a vector of offsets for the first
occurrence of \1 to \9, indexed by the second parsed_pattern value. All other
references have enough space for the offset to be put into the parsed pattern.
*/

for (i = 0; i < 10; i++) cb.small_ref_offset[i] = PCRE2_UNSET;


/* --------------- Start looking at the pattern --------------- */

/* Unless PCRE2_LITERAL is set, check for global one-time option settings at
the start of the pattern, and remember the offset to the actual regex. With
valgrind support, make the terminator of a zero-terminated pattern
inaccessible. This catches bugs that would otherwise only show up for
non-zero-terminated patterns. */

#ifdef SUPPORT_VALGRIND
if (zero_terminated) VALGRIND_MAKE_MEM_NOACCESS(pattern + patlen, CU2BYTES(1));
#endif

xoptions = ccontext->extra_options;
ptr = pattern;
skipatstart = 0;

if ((options & PCRE2_LITERAL) == 0)
  {
  while (patlen - skipatstart >= 2 &&
         ptr[skipatstart] == CHAR_LEFT_PARENTHESIS &&
         ptr[skipatstart+1] == CHAR_ASTERISK)
    {
    for (i = 0; i < sizeof(pso_list)/sizeof(pso); i++)
      {
      const pso *p = pso_list + i;

      if (patlen - skipatstart - 2 >= p->length &&
          PRIV(strncmp_c8)(ptr + skipatstart + 2, p->name, p->length) == 0)
        {
        uint32_t c, pp;

        skipatstart += p->length + 2;
        switch(p->type)
          {
          case PSO_OPT:
          cb.external_options |= p->value;
          break;

          case PSO_XOPT:
          xoptions |= p->value;
          break;

          case PSO_FLG:
          setflags |= p->value;
          break;

          case PSO_NL:
          newline = p->value;
          setflags |= PCRE2_NL_SET;
          break;

          case PSO_BSR:
          bsr = p->value;
          setflags |= PCRE2_BSR_SET;
          break;

          case PSO_LIMM:
          case PSO_LIMD:
          case PSO_LIMH:
          c = 0;
          pp = skipatstart;
          while (pp < patlen && IS_DIGIT(ptr[pp]))
            {
            if (c > UINT32_MAX / 10 - 1) break;   /* Integer overflow */
            c = c*10 + (ptr[pp++] - CHAR_0);
            }
          if (pp >= patlen || pp == skipatstart || ptr[pp] != CHAR_RIGHT_PARENTHESIS)
            {
            errorcode = ERR60;
            ptr += pp;
            goto HAD_EARLY_ERROR;
            }
          if (p->type == PSO_LIMH) limit_heap = c;
            else if (p->type == PSO_LIMM) limit_match = c;
            else limit_depth = c;
          skipatstart = ++pp;
          break;

          case PSO_OPTMZ:
          optim_flags &= ~(p->value);

          /* For backward compatibility the three original VERBs to disable
          optimizations need to also update the corresponding bit in the
          external options. */

          switch(p->value)
            {
            case PCRE2_OPTIM_AUTO_POSSESS:
            cb.external_options |= PCRE2_NO_AUTO_POSSESS;
            break;

            case PCRE2_OPTIM_DOTSTAR_ANCHOR:
            cb.external_options |= PCRE2_NO_DOTSTAR_ANCHOR;
            break;

            case PCRE2_OPTIM_START_OPTIMIZE:
            cb.external_options |= PCRE2_NO_START_OPTIMIZE;
            break;
            }

          break;

          default:
          /* All values in the enum need an explicit entry for this switch
          but until a better way to prevent coding mistakes is invented keep
          a catch all that triggers a debug build assert as a failsafe */
          PCRE2_DEBUG_UNREACHABLE();
          }
        break;   /* Out of the table scan loop */
        }
      }
    if (i >= sizeof(pso_list)/sizeof(pso)) break;   /* Out of pso loop */
    }
    PCRE2_ASSERT(skipatstart <= patlen);
  }

/* End of pattern-start options; advance to start of real regex. */

ptr += skipatstart;

/* Can't support UTF or UCP if PCRE2 was built without Unicode support. */

#ifndef SUPPORT_UNICODE
if ((cb.external_options & (PCRE2_UTF|PCRE2_UCP)) != 0)
  {
  errorcode = ERR32;
  goto HAD_EARLY_ERROR;
  }
#endif

/* Check UTF. We have the original options in 'options', with that value as
modified by (*UTF) etc in cb->external_options. The extra option
PCRE2_EXTRA_ALLOW_SURROGATE_ESCAPES is not permitted in UTF-16 mode because the
surrogate code points cannot be represented in UTF-16. */

utf = (cb.external_options & PCRE2_UTF) != 0;
if (utf)
  {
  if ((options & PCRE2_NEVER_UTF) != 0)
    {
    errorcode = ERR74;
    goto HAD_EARLY_ERROR;
    }
  if ((options & PCRE2_NO_UTF_CHECK) == 0 &&
       (errorcode = PRIV(valid_utf)(pattern, patlen, erroroffset)) != 0)
    goto HAD_ERROR;  /* Offset was set by valid_utf() */

#if PCRE2_CODE_UNIT_WIDTH == 16
  if ((ccontext->extra_options & PCRE2_EXTRA_ALLOW_SURROGATE_ESCAPES) != 0)
    {
    errorcode = ERR91;
    goto HAD_EARLY_ERROR;
    }
#endif
  }

/* Check UCP lockout. */

ucp = (cb.external_options & PCRE2_UCP) != 0;
if (ucp && (cb.external_options & PCRE2_NEVER_UCP) != 0)
  {
  errorcode = ERR75;
  goto HAD_EARLY_ERROR;
  }

/* PCRE2_EXTRA_TURKISH_CASING checks */

if ((xoptions & PCRE2_EXTRA_TURKISH_CASING) != 0)
  {
  if (!utf && !ucp)
    {
    errorcode = ERR104;
    goto HAD_EARLY_ERROR;
    }

#if PCRE2_CODE_UNIT_WIDTH == 8
  if (!utf)
    {
    errorcode = ERR105;
    goto HAD_EARLY_ERROR;
    }
#endif

  if ((xoptions & PCRE2_EXTRA_CASELESS_RESTRICT) != 0)
    {
    errorcode = ERR106;
    goto HAD_EARLY_ERROR;
    }
  }

/* Process the BSR setting. */

if (bsr == 0) bsr = ccontext->bsr_convention;

/* Process the newline setting. */

if (newline == 0) newline = ccontext->newline_convention;
cb.nltype = NLTYPE_FIXED;
switch(newline)
  {
  case PCRE2_NEWLINE_CR:
  cb.nllen = 1;
  cb.nl[0] = CHAR_CR;
  break;

  case PCRE2_NEWLINE_LF:
  cb.nllen = 1;
  cb.nl[0] = CHAR_NL;
  break;

  case PCRE2_NEWLINE_NUL:
  cb.nllen = 1;
  cb.nl[0] = CHAR_NUL;
  break;

  case PCRE2_NEWLINE_CRLF:
  cb.nllen = 2;
  cb.nl[0] = CHAR_CR;
  cb.nl[1] = CHAR_NL;
  break;

  case PCRE2_NEWLINE_ANY:
  cb.nltype = NLTYPE_ANY;
  break;

  case PCRE2_NEWLINE_ANYCRLF:
  cb.nltype = NLTYPE_ANYCRLF;
  break;

  default:
  PCRE2_DEBUG_UNREACHABLE();
  errorcode = ERR56;
  goto HAD_EARLY_ERROR;
  }

/* Pre-scan the pattern to do two things: (1) Discover the named groups and
their numerical equivalents, so that this information is always available for
the remaining processing. (2) At the same time, parse the pattern and put a
processed version into the parsed_pattern vector. This has escapes interpreted
and comments removed (amongst other things). */

/* Ensure that the parsed pattern buffer is big enough. For many smaller
patterns the vector on the stack (which was set up above) can be used. */

parsed_size_needed = max_parsed_pattern(ptr, cb.end_pattern, utf, options);

/* Allow for 2x uint32_t at the start and 2 at the end, for
PCRE2_EXTRA_MATCH_WORD or PCRE2_EXTRA_MATCH_LINE (which are exclusive). */

if ((ccontext->extra_options &
     (PCRE2_EXTRA_MATCH_WORD|PCRE2_EXTRA_MATCH_LINE)) != 0)
  parsed_size_needed += 4;

/* When PCRE2_AUTO_CALLOUT is set we allow for one callout at the end. */

if ((options & PCRE2_AUTO_CALLOUT) != 0)
  parsed_size_needed += 4;

parsed_size_needed += 1;  /* For the final META_END */

if (parsed_size_needed > PARSED_PATTERN_DEFAULT_SIZE)
  {
  uint32_t *heap_parsed_pattern = ccontext->memctl.malloc(
    parsed_size_needed * sizeof(uint32_t), ccontext->memctl.memory_data);
  if (heap_parsed_pattern == NULL)
    {
    *errorptr = ERR21;
    goto EXIT;
    }
  cb.parsed_pattern = heap_parsed_pattern;
  }
cb.parsed_pattern_end = cb.parsed_pattern + parsed_size_needed;

/* Do the parsing scan. */

errorcode = parse_regex(ptr, cb.external_options, xoptions, &has_lookbehind, &cb);
if (errorcode != 0) goto HAD_CB_ERROR;

/* If there are any lookbehinds, scan the parsed pattern to figure out their
lengths. Workspace is needed to remember whether numbered groups are or are not
of limited length, and if limited, what the minimum and maximum lengths are.
This caching saves re-computing the length of any group that is referenced more
than once, which is particularly relevant when recursion is involved.
Unnumbered groups do not have this exposure because they cannot be referenced.
If there are sufficiently few groups, the default index vector on the stack, as
set up above, can be used. Otherwise we have to get/free some heap memory. The
vector must be initialized to zero. */

if (has_lookbehind)
  {
  int loopcount = 0;
  if (cb.bracount >= GROUPINFO_DEFAULT_SIZE/2)
    {
    cb.groupinfo = ccontext->memctl.malloc(
      (2 * (cb.bracount + 1))*sizeof(uint32_t), ccontext->memctl.memory_data);
    if (cb.groupinfo == NULL)
      {
      errorcode = ERR21;
      cb.erroroffset = 0;
      goto HAD_CB_ERROR;
      }
    }
  memset(cb.groupinfo, 0, (2 * cb.bracount + 1) * sizeof(uint32_t));
  errorcode = check_lookbehinds(cb.parsed_pattern, NULL, NULL, &cb, &loopcount);
  if (errorcode != 0) goto HAD_CB_ERROR;
  }

/* For debugging, there is a function that shows the parsed pattern vector. */

#ifdef DEBUG_SHOW_PARSED
fprintf(stderr, "+++ Pre-scan complete:\n");
show_parsed(&cb);
#endif

/* For debugging capturing information this code can be enabled. */

#ifdef DEBUG_SHOW_CAPTURES
  {
  named_group *ng = cb.named_groups;
  fprintf(stderr, "+++Captures: %d\n", cb.bracount);
  for (i = 0; i < cb.names_found; i++, ng++)
    {
    fprintf(stderr, "+++%3d %.*s\n", ng->number, ng->length, ng->name);
    }
  }
#endif

/* Pretend to compile the pattern while actually just accumulating the amount
of memory required in the 'length' variable. This behaviour is triggered by
passing a non-NULL final argument to compile_regex(). We pass a block of
workspace (cworkspace) for it to compile parts of the pattern into; the
compiled code is discarded when it is no longer needed, so hopefully this
workspace will never overflow, though there is a test for its doing so.

On error, errorcode will be set non-zero, so we don't need to look at the
result of the function. The initial options have been put into the cb block,
but we still have to pass a separate options variable (the first argument)
because the options may change as the pattern is processed. */

cb.erroroffset = patlen;   /* For any subsequent errors that do not set it */
pptr = cb.parsed_pattern;
code = cworkspace;
*code = OP_BRA;

(void)compile_regex(cb.external_options, xoptions, &code, &pptr,
   &errorcode, 0, &firstcu, &firstcuflags, &reqcu, &reqcuflags, NULL, NULL,
   &cb, &length);

if (errorcode != 0) goto HAD_CB_ERROR;  /* Offset is in cb.erroroffset */

/* This should be caught in compile_regex(), but just in case... */

#if defined SUPPORT_WIDE_CHARS
PCRE2_ASSERT((cb.char_lists_size & 0x3) == 0);
if (length > MAX_PATTERN_SIZE ||
    MAX_PATTERN_SIZE - length < (cb.char_lists_size / sizeof(PCRE2_UCHAR)))
#else
if (length > MAX_PATTERN_SIZE)
#endif
  {
  errorcode = ERR20;
  goto HAD_CB_ERROR;
  }

/* Compute the size of, then, if not too large, get and initialize the data
block for storing the compiled pattern and names table. Integer overflow should
no longer be possible because nowadays we limit the maximum value of
cb.names_found and cb.name_entry_size. */

re_blocksize =
  CU2BYTES((PCRE2_SIZE)cb.names_found * (PCRE2_SIZE)cb.name_entry_size);

#if defined SUPPORT_WIDE_CHARS
if (cb.char_lists_size != 0)
  {
#if PCRE2_CODE_UNIT_WIDTH != 32
  /* Align to 32 bit first. This ensures the
  allocated area will also be 32 bit aligned. */
  re_blocksize = (PCRE2_SIZE)CLIST_ALIGN_TO(re_blocksize, sizeof(uint32_t));
#endif
  re_blocksize += cb.char_lists_size;
  }
#endif

re_blocksize += CU2BYTES(length);

if (re_blocksize > ccontext->max_pattern_compiled_length)
  {
  errorcode = ERR101;
  goto HAD_CB_ERROR;
  }

re_blocksize += sizeof(pcre2_real_code);
re = (pcre2_real_code *)
  ccontext->memctl.malloc(re_blocksize, ccontext->memctl.memory_data);
if (re == NULL)
  {
  errorcode = ERR21;
  goto HAD_CB_ERROR;
  }

/* The compiler may put padding at the end of the pcre2_real_code structure in
order to round it up to a multiple of 4 or 8 bytes. This means that when a
compiled pattern is copied (for example, when serialized) undefined bytes are
read, and this annoys debuggers such as valgrind. To avoid this, we explicitly
write to the last 8 bytes of the structure before setting the fields. */

memset((char *)re + sizeof(pcre2_real_code) - 8, 0, 8);
re->memctl = ccontext->memctl;
re->tables = tables;
re->executable_jit = NULL;
memset(re->start_bitmap, 0, 32 * sizeof(uint8_t));
re->blocksize = re_blocksize;
re->code_start = re_blocksize - CU2BYTES(length);
re->magic_number = MAGIC_NUMBER;
re->compile_options = options;
re->overall_options = cb.external_options;
re->extra_options = xoptions;
re->flags = PCRE2_CODE_UNIT_WIDTH/8 | cb.external_flags | setflags;
re->limit_heap = limit_heap;
re->limit_match = limit_match;
re->limit_depth = limit_depth;
re->first_codeunit = 0;
re->last_codeunit = 0;
re->bsr_convention = bsr;
re->newline_convention = newline;
re->max_lookbehind = 0;
re->minlength = 0;
re->top_bracket = 0;
re->top_backref = 0;
re->name_entry_size = cb.name_entry_size;
re->name_count = cb.names_found;
re->optimization_flags = optim_flags;

/* The basic block is immediately followed by the name table, and the compiled
code follows after that. */

codestart = (PCRE2_UCHAR *)((uint8_t *)re + re->code_start);

/* Update the compile data block for the actual compile. The starting points of
the name/number translation table and of the code are passed around in the
compile data block. The start/end pattern and initial options are already set
from the pre-compile phase, as is the name_entry_size field. */

cb.parens_depth = 0;
cb.assert_depth = 0;
cb.lastcapture = 0;
cb.name_table = (PCRE2_UCHAR *)((uint8_t *)re + sizeof(pcre2_real_code));
cb.start_code = codestart;
cb.req_varyopt = 0;
cb.had_accept = FALSE;
cb.had_pruneorskip = FALSE;
#ifdef SUPPORT_WIDE_CHARS
cb.char_lists_size = 0;
#endif


/* If any named groups were found, create the name/number table from the list
created in the pre-pass. */

if (cb.names_found > 0)
  {
  named_group *ng = cb.named_groups;
  for (i = 0; i < cb.names_found; i++, ng++)
    add_name_to_table(&cb, ng->name, ng->length, ng->number, i);
  }

/* Set up a starting, non-extracting bracket, then compile the expression. On
error, errorcode will be set non-zero, so we don't need to look at the result
of the function here. */

pptr = cb.parsed_pattern;
code = (PCRE2_UCHAR *)codestart;
*code = OP_BRA;
regexrc = compile_regex(re->overall_options, re->extra_options, &code,
  &pptr, &errorcode, 0, &firstcu, &firstcuflags, &reqcu, &reqcuflags, NULL,
  NULL, &cb, NULL);
if (regexrc < 0) re->flags |= PCRE2_MATCH_EMPTY;
re->top_bracket = cb.bracount;
re->top_backref = cb.top_backref;
re->max_lookbehind = cb.max_lookbehind;

if (cb.had_accept)
  {
  reqcu = 0;                     /* Must disable after (*ACCEPT) */
  reqcuflags = REQ_NONE;
  re->flags |= PCRE2_HASACCEPT;  /* Disables minimum length */
  }

/* Fill in the final opcode and check for disastrous overflow. If no overflow,
but the estimated length exceeds the really used length, adjust the value of
re->blocksize, and if valgrind support is configured, mark the extra allocated
memory as unaddressable, so that any out-of-bound reads can be detected. */

*code++ = OP_END;
usedlength = code - codestart;
if (usedlength > length)
  {
  PCRE2_DEBUG_UNREACHABLE();
  errorcode = ERR23;  /* Overflow of code block - internal error */
  }
else
  {
  re->blocksize -= CU2BYTES(length - usedlength);
#ifdef SUPPORT_VALGRIND
  VALGRIND_MAKE_MEM_NOACCESS(code, CU2BYTES(length - usedlength));
#endif
  }

/* Scan the pattern for recursion/subroutine calls and convert the group
numbers into offsets. Maintain a small cache so that repeated groups containing
recursions are efficiently handled. */

#define RSCAN_CACHE_SIZE 8

if (errorcode == 0 && cb.had_recurse)
  {
  PCRE2_UCHAR *rcode;
  PCRE2_SPTR rgroup;
  unsigned int ccount = 0;
  int start = RSCAN_CACHE_SIZE;
  recurse_cache rc[RSCAN_CACHE_SIZE];

  for (rcode = find_recurse(codestart, utf);
       rcode != NULL;
       rcode = find_recurse(rcode + 1 + LINK_SIZE, utf))
    {
    int p, groupnumber;

    groupnumber = (int)GET(rcode, 1);
    if (groupnumber == 0) rgroup = codestart; else
      {
      PCRE2_SPTR search_from = codestart;
      rgroup = NULL;
      for (i = 0, p = start; i < ccount; i++, p = (p + 1) & 7)
        {
        if (groupnumber == rc[p].groupnumber)
          {
          rgroup = rc[p].group;
          break;
          }

        /* Group n+1 must always start to the right of group n, so we can save
        search time below when the new group number is greater than any of the
        previously found groups. */

        if (groupnumber > rc[p].groupnumber) search_from = rc[p].group;
        }

      if (rgroup == NULL)
        {
        rgroup = PRIV(find_bracket)(search_from, utf, groupnumber);
        if (rgroup == NULL)
          {
          PCRE2_DEBUG_UNREACHABLE();
          errorcode = ERR53;
          break;
          }
        if (--start < 0) start = RSCAN_CACHE_SIZE - 1;
        rc[start].groupnumber = groupnumber;
        rc[start].group = rgroup;
        if (ccount < RSCAN_CACHE_SIZE) ccount++;
        }
      }

    PUT(rcode, 1, (uint32_t)(rgroup - codestart));
    }
  }

/* In rare debugging situations we sometimes need to look at the compiled code
at this stage. */

#ifdef DEBUG_CALL_PRINTINT
pcre2_printint(re, stderr, TRUE);
fprintf(stderr, "Length=%lu Used=%lu\n", length, usedlength);
#endif

/* Unless disabled, check whether any single character iterators can be
auto-possessified. The function overwrites the appropriate opcode values, so
the type of the pointer must be cast. NOTE: the intermediate variable "temp" is
used in this code because at least one compiler gives a warning about loss of
"const" attribute if the cast (PCRE2_UCHAR *)codestart is used directly in the
function call. */

if (errorcode == 0 && (optim_flags & PCRE2_OPTIM_AUTO_POSSESS) != 0)
  {
  PCRE2_UCHAR *temp = (PCRE2_UCHAR *)codestart;
  if (PRIV(auto_possessify)(temp, &cb) != 0)
    {
    PCRE2_DEBUG_UNREACHABLE();
    errorcode = ERR80;
    }
  }

/* Failed to compile, or error while post-processing. */

if (errorcode != 0) goto HAD_CB_ERROR;

/* Successful compile. If the anchored option was not passed, set it if
we can determine that the pattern is anchored by virtue of ^ characters or \A
or anything else, such as starting with non-atomic .* when DOTALL is set and
there are no occurrences of *PRUNE or *SKIP (though there is an option to
disable this case). */

if ((re->overall_options & PCRE2_ANCHORED) == 0)
  {
  BOOL dotstar_anchor = ((optim_flags & PCRE2_OPTIM_DOTSTAR_ANCHOR) != 0);
  if (is_anchored(codestart, 0, &cb, 0, FALSE, dotstar_anchor))
    re->overall_options |= PCRE2_ANCHORED;
  }

/* Set up the first code unit or startline flag, the required code unit, and
then study the pattern. This code need not be obeyed if PCRE2_OPTIM_START_OPTIMIZE
is disabled, as the data it would create will not be used. Note that a first code
unit (but not the startline flag) is useful for anchored patterns because it
can still give a quick "no match" and also avoid searching for a last code
unit. */

if ((optim_flags & PCRE2_OPTIM_START_OPTIMIZE) != 0)
  {
  int minminlength = 0;  /* For minimal minlength from first/required CU */

  /* If we do not have a first code unit, see if there is one that is asserted
  (these are not saved during the compile because they can cause conflicts with
  actual literals that follow). */

  if (firstcuflags >= REQ_NONE) {
    uint32_t assertedcuflags = 0;
    uint32_t assertedcu = find_firstassertedcu(codestart, &assertedcuflags, 0);
    /* It would be wrong to use the asserted first code unit as `firstcu` for
     * regexes which are able to match a 1-character string (e.g. /(?=a)b?a/)
     * For that example, if we set both firstcu and reqcu to 'a', it would mean
     * the subject string needs to be at least 2 characters long, which is wrong.
     * With more analysis, we would be able to set firstcu in more cases. */
    if (assertedcuflags < REQ_NONE && assertedcu != reqcu) {
      firstcu = assertedcu;
      firstcuflags = assertedcuflags;
    }
  }

  /* Save the data for a first code unit. The existence of one means the
  minimum length must be at least 1. */

  if (firstcuflags < REQ_NONE)
    {
    re->first_codeunit = firstcu;
    re->flags |= PCRE2_FIRSTSET;
    minminlength++;

    /* Handle caseless first code units. */

    if ((firstcuflags & REQ_CASELESS) != 0)
      {
      if (firstcu < 128 || (!utf && !ucp && firstcu < 255))
        {
        if (cb.fcc[firstcu] != firstcu) re->flags |= PCRE2_FIRSTCASELESS;
        }

      /* The first code unit is > 128 in UTF or UCP mode, or > 255 otherwise.
      In 8-bit UTF mode, code units in the range 128-255 are introductory code
      units and cannot have another case, but if UCP is set they may do. */

#ifdef SUPPORT_UNICODE
#if PCRE2_CODE_UNIT_WIDTH == 8
      else if (ucp && !utf && UCD_OTHERCASE(firstcu) != firstcu)
        re->flags |= PCRE2_FIRSTCASELESS;
#else
      else if ((utf || ucp) && firstcu <= MAX_UTF_CODE_POINT &&
               UCD_OTHERCASE(firstcu) != firstcu)
        re->flags |= PCRE2_FIRSTCASELESS;
#endif
#endif  /* SUPPORT_UNICODE */
      }
    }

  /* When there is no first code unit, for non-anchored patterns, see if we can
  set the PCRE2_STARTLINE flag. This is helpful for multiline matches when all
  branches start with ^ and also when all branches start with non-atomic .* for
  non-DOTALL matches when *PRUNE and SKIP are not present. (There is an option
  that disables this case.) */

  else if ((re->overall_options & PCRE2_ANCHORED) == 0)
    {
    BOOL dotstar_anchor = ((optim_flags & PCRE2_OPTIM_DOTSTAR_ANCHOR) != 0);
    if (is_startline(codestart, 0, &cb, 0, FALSE, dotstar_anchor))
      re->flags |= PCRE2_STARTLINE;
    }

  /* Handle the "required code unit", if one is set. In the UTF case we can
  increment the minimum minimum length only if we are sure this really is a
  different character and not a non-starting code unit of the first character,
  because the minimum length count is in characters, not code units. */

  if (reqcuflags < REQ_NONE)
    {
#if PCRE2_CODE_UNIT_WIDTH == 16
    if ((re->overall_options & PCRE2_UTF) == 0 ||   /* Not UTF */
        firstcuflags >= REQ_NONE ||                 /* First not set */
        (firstcu & 0xf800) != 0xd800 ||             /* First not surrogate */
        (reqcu & 0xfc00) != 0xdc00)                 /* Req not low surrogate */
#elif PCRE2_CODE_UNIT_WIDTH == 8
    if ((re->overall_options & PCRE2_UTF) == 0 ||   /* Not UTF */
        firstcuflags >= REQ_NONE ||                 /* First not set */
        (firstcu & 0x80) == 0 ||                    /* First is ASCII */
        (reqcu & 0x80) == 0)                        /* Req is ASCII */
#endif
      {
      minminlength++;
      }

    /* In the case of an anchored pattern, set up the value only if it follows
    a variable length item in the pattern. */

    if ((re->overall_options & PCRE2_ANCHORED) == 0 ||
        (reqcuflags & REQ_VARY) != 0)
      {
      re->last_codeunit = reqcu;
      re->flags |= PCRE2_LASTSET;

      /* Handle caseless required code units as for first code units (above). */

      if ((reqcuflags & REQ_CASELESS) != 0)
        {
        if (reqcu < 128 || (!utf && !ucp && reqcu < 255))
          {
          if (cb.fcc[reqcu] != reqcu) re->flags |= PCRE2_LASTCASELESS;
          }
#ifdef SUPPORT_UNICODE
#if PCRE2_CODE_UNIT_WIDTH == 8
      else if (ucp && !utf && UCD_OTHERCASE(reqcu) != reqcu)
        re->flags |= PCRE2_LASTCASELESS;
#else
      else if ((utf || ucp) && reqcu <= MAX_UTF_CODE_POINT &&
               UCD_OTHERCASE(reqcu) != reqcu)
        re->flags |= PCRE2_LASTCASELESS;
#endif
#endif  /* SUPPORT_UNICODE */
        }
      }
    }

  /* Study the compiled pattern to set up information such as a bitmap of
  starting code units and a minimum matching length. */

  if (PRIV(study)(re) != 0)
    {
    PCRE2_DEBUG_UNREACHABLE();
    errorcode = ERR31;
    goto HAD_CB_ERROR;
    }

  /* If study() set a bitmap of starting code units, it implies a minimum
  length of at least one. */

  if ((re->flags & PCRE2_FIRSTMAPSET) != 0 && minminlength == 0)
    minminlength = 1;

  /* If the minimum length set (or not set) by study() is less than the minimum
  implied by required code units, override it. */

  if (re->minlength < minminlength) re->minlength = minminlength;
  }   /* End of start-of-match optimizations. */

/* Control ends up here in all cases. When running under valgrind, make a
pattern's terminating zero defined again. If memory was obtained for the parsed
version of the pattern, free it before returning. Also free the list of named
groups if a larger one had to be obtained, and likewise the group information
vector. */

#ifdef SUPPORT_UNICODE
PCRE2_ASSERT(cb.cranges == NULL);
#endif

EXIT:
#ifdef SUPPORT_VALGRIND
if (zero_terminated) VALGRIND_MAKE_MEM_DEFINED(pattern + patlen, CU2BYTES(1));
#endif
if (cb.parsed_pattern != stack_parsed_pattern)
  ccontext->memctl.free(cb.parsed_pattern, ccontext->memctl.memory_data);
if (cb.named_group_list_size > NAMED_GROUP_LIST_SIZE)
  ccontext->memctl.free((void *)cb.named_groups, ccontext->memctl.memory_data);
if (cb.groupinfo != stack_groupinfo)
  ccontext->memctl.free((void *)cb.groupinfo, ccontext->memctl.memory_data);

return re;    /* Will be NULL after an error */

/* Errors discovered in parse_regex() set the offset value in the compile
block. Errors discovered before it is called must compute it from the ptr
value. After parse_regex() is called, the offset in the compile block is set to
the end of the pattern, but certain errors in compile_regex() may reset it if
an offset is available in the parsed pattern. */

HAD_CB_ERROR:
ptr = pattern + cb.erroroffset;

HAD_EARLY_ERROR:
PCRE2_ASSERT(ptr >= pattern); /* Ensure we don't return invalid erroroffset */
PCRE2_ASSERT(ptr <= (pattern + patlen));
*erroroffset = ptr - pattern;

HAD_ERROR:
*errorptr = errorcode;
pcre2_code_free(re);
re = NULL;

#ifdef SUPPORT_WIDE_CHARS
if (cb.cranges != NULL)
  {
  class_ranges* cranges = cb.cranges;
  do
    {
    class_ranges* next_cranges = cranges->next;
    cb.cx->memctl.free(cranges, cb.cx->memctl.memory_data);
    cranges = next_cranges;
    }
  while (cranges != NULL);
  }
#endif
goto EXIT;
}

/* These #undefs are here to enable unity builds with CMake. */

#undef NLBLOCK /* Block containing newline information */
#undef PSSTART /* Field containing processed string start */
#undef PSEND   /* Field containing processed string end */

/* End of pcre2_compile.c */
