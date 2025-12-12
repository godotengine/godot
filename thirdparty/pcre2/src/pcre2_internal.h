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

#ifndef PCRE2_INTERNAL_H_IDEMPOTENT_GUARD
#define PCRE2_INTERNAL_H_IDEMPOTENT_GUARD

/* We do not assume that the config.h file has an idempotent include guard,
since it may well be written by clients. The standard Autoheader config.h does
not have an include guard (although we could customise that). */

#if defined HAVE_CONFIG_H && !defined PCRE2_CONFIG_H_IDEMPOTENT_GUARD
#define PCRE2_CONFIG_H_IDEMPOTENT_GUARD
#include "config.h"
#endif

/* We do not support both EBCDIC and Unicode at the same time. The "configure"
script prevents both being selected, but not everybody uses "configure". EBCDIC
is only supported for the 8-bit library, but the check for this has to be later
in this file, because the first part is not width-dependent, and is included by
pcre2test.c with CODE_UNIT_WIDTH == 0. */

#if defined EBCDIC && defined SUPPORT_UNICODE
#error The use of both EBCDIC and SUPPORT_UNICODE is not supported.
#endif

/* When compiling one of the libraries, the value of PCRE2_CODE_UNIT_WIDTH must
be 8, 16, or 32. AutoTools and CMake ensure that this is always the case, but
other other building methods may not, so here is a check. It is cut out when
building pcre2test, bcause that sets the value to zero. No other source should
be including this file. There is no explicit way of forcing a compile to be
abandoned, but trying to include a non-existent file seems cleanest. Otherwise
there will be many irrelevant consequential errors. */

#if (!defined PCRE2_PCRE2TEST && !defined PCRE2_DFTABLES) && \
  (!defined PCRE2_CODE_UNIT_WIDTH ||     \
    (PCRE2_CODE_UNIT_WIDTH != 8 &&       \
     PCRE2_CODE_UNIT_WIDTH != 16 &&      \
     PCRE2_CODE_UNIT_WIDTH != 32))
#error PCRE2_CODE_UNIT_WIDTH must be defined as 8, 16, or 32.
#endif


/* Standard C headers */

#include <ctype.h>
#include <limits.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Macros to make boolean values more obvious. The #ifndef is to pacify
compiler warnings in environments where these macros are defined elsewhere.
Unfortunately, there is no way to do the same for the typedef. */

typedef int BOOL;
#ifndef FALSE
#define FALSE   0
#define TRUE    1
#endif

/* Helper macro for static (compile-time) assertions. Can be used inside
functions, or at the top-level of a file. */
#define STATIC_ASSERT_JOIN(a,b) a ## b
#define STATIC_ASSERT(cond, msg) \
  typedef int STATIC_ASSERT_JOIN(static_assertion_,msg)[(cond)?1:-1]

/* Valgrind (memcheck) support */

#ifdef SUPPORT_VALGRIND
#include <valgrind/memcheck.h>
#endif

/* -ftrivial-auto-var-init support supports initializing all local variables
to avoid some classes of bug, but this can cause an unacceptable slowdown
for large on-stack arrays in hot functions. This macro lets us annotate
such arrays. */

#ifdef HAVE_ATTRIBUTE_UNINITIALIZED
#define PCRE2_KEEP_UNINITIALIZED __attribute__((uninitialized))
#else
#define PCRE2_KEEP_UNINITIALIZED
#endif

/* Older versions of MSVC lack snprintf(). This define allows for
warning/error-free compilation and testing with MSVC compilers back to at least
MSVC 10/2010. Except for VC6 (which is missing some fundamentals and fails). */

#if defined(_MSC_VER) && (_MSC_VER < 1900)
#define snprintf _snprintf
#endif

/* When compiling a DLL for Windows, the exported symbols have to be declared
using some MS magic, as documented here:
https://learn.microsoft.com/en-us/cpp/build/exporting-from-a-dll-using-declspec-dllexport

In pcre2.h (which is included below), we define only PCRE2_EXP_DECL,
which is all that is needed for applications (they just import the symbols). To
compile the library, we use:

  PCRE2_EXP_DECL    for declarations
  PCRE2_EXP_DEFN    for definitions

The reason for wrapping this in #ifndef PCRE2_EXP_DECL is so that pcre2test,
which is an application, but needs to import this file in order to "peek" at
internals, can #include pcre2.h first to get an application's-eye view.

In principle, people compiling for non-Windows, non-Unix-like (i.e. uncommon,
special-purpose environments) might want to stick other stuff in front of
exported symbols. That's why, in the non-Windows case, we set PCRE2_EXP_DEFN
only if it is not already set. */

#if defined __cplusplus
#error This project uses C99. C++ is not supported.
#endif

#ifndef PCRE2_EXP_DECL
#  if defined(_WIN32) && !defined(PCRE2_STATIC)
#    define PCRE2_EXP_DECL  extern __declspec(dllexport)
#  else
#    define PCRE2_EXP_DECL  extern PCRE2_EXPORT
#  endif
#endif

#ifndef PCRE2_EXP_DEFN
#  if defined(_WIN32) && !defined(PCRE2_STATIC)
#    define PCRE2_EXP_DEFN  extern __declspec(dllexport)
#  else
#    define PCRE2_EXP_DEFN  extern PCRE2_EXPORT
#  endif
#endif

/* Include the public PCRE2 header and the definitions of UCP character
property values. This must follow the setting of PCRE2_EXP_DECL above. */

#include "pcre2.h"
#include "pcre2_ucp.h"

/* When checking for integer overflow, we need to handle large integers.
If a 64-bit integer type is available, we can use that.
Otherwise we have to cast to double, which of course requires floating point
arithmetic. Handle this by defining a macro for the appropriate type. */

#if defined INT64_MAX || defined int64_t
#define INT64_OR_DOUBLE int64_t
#else
#define INT64_OR_DOUBLE double
#endif

/* External (in the C sense) functions and tables that are private to the
libraries are always referenced using the PRIV macro. This makes it possible
for pcre2test.c to include some of the source files from the libraries using a
different PRIV definition to avoid name clashes. It also makes it clear in the
code that a non-static object is being referenced. */

#ifndef PRIV
#define PRIV(name) _pcre2_##name
#endif

/* This is an unsigned int value that no UTF character can ever have, as
Unicode doesn't go beyond 0x0010ffff. */

#define NOTACHAR 0xffffffff

/* This is the largest valid UTF/Unicode code point. */

#define MAX_UTF_CODE_POINT 0x10ffff

/* Compile-time positive error numbers (all except UTF errors, which are
negative) start at this value. It should probably never be changed, in case
some application is checking for specific numbers. There is a copy of this
#define in pcre2posix.c (which now no longer includes this file). Ideally, a
way of having a single definition should be found, but as the number is
unlikely to change, this is not a pressing issue. The original reason for
having a base other than 0 was to keep the absolute values of compile-time and
run-time error numbers numerically different, but in the event the code does
not rely on this. */

#define COMPILE_ERROR_BASE 100

/* The initial frames vector for remembering pcre2_match() backtracking points
is allocated on the heap, of this size (bytes) or ten times the frame size if
larger, unless the heap limit is smaller. Typical frame sizes are a few hundred
bytes (it depends on the number of capturing parentheses) so 20KiB handles
quite a few frames. A larger vector on the heap is obtained for matches that
need more frames, subject to the heap limit. */

#define START_FRAMES_SIZE 20480

/* For DFA matching, an initial internal workspace vector is allocated on the
stack. The heap is used only if this turns out to be too small. */

#define DFA_START_RWS_SIZE 30720

/* Define the default BSR convention. */

#ifdef BSR_ANYCRLF
#define BSR_DEFAULT PCRE2_BSR_ANYCRLF
#else
#define BSR_DEFAULT PCRE2_BSR_UNICODE
#endif


/* ---------------- Basic UTF-8 macros ---------------- */

/* These UTF-8 macros are always defined because they are used in pcre2test for
handling wide characters in 16-bit and 32-bit modes, even if an 8-bit library
is not supported. */

/* Tests whether a UTF-8 code point needs extra bytes to decode. */

#define HASUTF8EXTRALEN(c) ((c) >= 0xc0)

/* The following macros were originally written in the form of loops that used
data from the tables whose names start with PRIV(utf8_table). They were
rewritten by a user so as not to use loops, because in some environments this
gives a significant performance advantage, and it seems never to do any harm.
*/

/* Base macro to pick up the remaining bytes of a UTF-8 character, not
advancing the pointer. */

#define GETUTF8(c, eptr) \
    { \
    if ((c & 0x20u) == 0) \
      c = ((c & 0x1fu) << 6) | (eptr[1] & 0x3fu); \
    else if ((c & 0x10u) == 0) \
      c = ((c & 0x0fu) << 12) | ((eptr[1] & 0x3fu) << 6) | (eptr[2] & 0x3fu); \
    else if ((c & 0x08u) == 0) \
      c = ((c & 0x07u) << 18) | ((eptr[1] & 0x3fu) << 12) | \
      ((eptr[2] & 0x3fu) << 6) | (eptr[3] & 0x3fu); \
    else if ((c & 0x04u) == 0) \
      c = ((c & 0x03u) << 24) | ((eptr[1] & 0x3fu) << 18) | \
          ((eptr[2] & 0x3fu) << 12) | ((eptr[3] & 0x3fu) << 6) | \
          (eptr[4] & 0x3fu); \
    else \
      c = ((c & 0x01u) << 30) | ((eptr[1] & 0x3fu) << 24) | \
          ((eptr[2] & 0x3fu) << 18) | ((eptr[3] & 0x3fu) << 12) | \
          ((eptr[4] & 0x3fu) << 6) | (eptr[5] & 0x3fu); \
    }

/* Base macro to pick up the remaining bytes of a UTF-8 character, advancing
the pointer. */

#define GETUTF8INC(c, eptr) \
    { \
    if ((c & 0x20u) == 0) \
      c = ((c & 0x1fu) << 6) | (*eptr++ & 0x3fu); \
    else if ((c & 0x10u) == 0) \
      { \
      c = ((c & 0x0fu) << 12) | ((*eptr & 0x3fu) << 6) | (eptr[1] & 0x3fu); \
      eptr += 2; \
      } \
    else if ((c & 0x08u) == 0) \
      { \
      c = ((c & 0x07u) << 18) | ((*eptr & 0x3fu) << 12) | \
          ((eptr[1] & 0x3fu) << 6) | (eptr[2] & 0x3fu); \
      eptr += 3; \
      } \
    else if ((c & 0x04u) == 0) \
      { \
      c = ((c & 0x03u) << 24) | ((*eptr & 0x3fu) << 18) | \
          ((eptr[1] & 0x3fu) << 12) | ((eptr[2] & 0x3fu) << 6) | \
          (eptr[3] & 0x3fu); \
      eptr += 4; \
      } \
    else \
      { \
      c = ((c & 0x01u) << 30) | ((*eptr & 0x3fu) << 24) | \
          ((eptr[1] & 0x3fu) << 18) | ((eptr[2] & 0x3fu) << 12) | \
          ((eptr[3] & 0x3fu) << 6) | (eptr[4] & 0x3fu); \
      eptr += 5; \
      } \
    }

/* Base macro to pick up the remaining bytes of a UTF-8 character, not
advancing the pointer, incrementing the length. */

#define GETUTF8LEN(c, eptr, len) \
    { \
    if ((c & 0x20u) == 0) \
      { \
      c = ((c & 0x1fu) << 6) | (eptr[1] & 0x3fu); \
      len++; \
      } \
    else if ((c & 0x10u)  == 0) \
      { \
      c = ((c & 0x0fu) << 12) | ((eptr[1] & 0x3fu) << 6) | (eptr[2] & 0x3fu); \
      len += 2; \
      } \
    else if ((c & 0x08u)  == 0) \
      {\
      c = ((c & 0x07u) << 18) | ((eptr[1] & 0x3fu) << 12) | \
          ((eptr[2] & 0x3fu) << 6) | (eptr[3] & 0x3fu); \
      len += 3; \
      } \
    else if ((c & 0x04u)  == 0) \
      { \
      c = ((c & 0x03u) << 24) | ((eptr[1] & 0x3fu) << 18) | \
          ((eptr[2] & 0x3fu) << 12) | ((eptr[3] & 0x3fu) << 6) | \
          (eptr[4] & 0x3fu); \
      len += 4; \
      } \
    else \
      {\
      c = ((c & 0x01u) << 30) | ((eptr[1] & 0x3fu) << 24) | \
          ((eptr[2] & 0x3fu) << 18) | ((eptr[3] & 0x3fu) << 12) | \
          ((eptr[4] & 0x3fu) << 6) | (eptr[5] & 0x3fu); \
      len += 5; \
      } \
    }

/* --------------- Whitespace macros ---------------- */

/* Tests for Unicode horizontal and vertical whitespace characters must check a
number of different values. Using a switch statement for this generates the
fastest code (no loop, no memory access), and there are several places in the
interpreter code where this happens. In order to ensure that all the case lists
remain in step, we use macros so that there is only one place where the lists
are defined.

These values are also required as lists in pcre2_compile.c when processing \h,
\H, \v and \V in a character class. The lists are defined in pcre2_tables.c,
but macros that define the values are here so that all the definitions are
together. The lists must be in ascending character order, terminated by
NOTACHAR (which is 0xffffffff).

Any changes should ensure that the various macros are kept in step with each
other. NOTE: The values also appear in pcre2_jit_compile.c. */

/* -------------- ASCII/Unicode environments -------------- */

#ifndef EBCDIC

/* Character U+180E (Mongolian Vowel Separator) is not included in the list of
spaces in the Unicode file PropList.txt, and Perl does not recognize it as a
space. However, in many other sources it is listed as a space and has been in
PCRE (both APIs) for a long time. */

#define HSPACE_LIST \
  CHAR_HT, CHAR_SPACE, CHAR_NBSP, \
  0x1680, 0x180e, 0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005, \
  0x2006, 0x2007, 0x2008, 0x2009, 0x200a, 0x202f, 0x205f, 0x3000, \
  NOTACHAR

#define HSPACE_MULTIBYTE_CASES \
  case 0x1680:  /* OGHAM SPACE MARK */ \
  case 0x180e:  /* MONGOLIAN VOWEL SEPARATOR */ \
  case 0x2000:  /* EN QUAD */ \
  case 0x2001:  /* EM QUAD */ \
  case 0x2002:  /* EN SPACE */ \
  case 0x2003:  /* EM SPACE */ \
  case 0x2004:  /* THREE-PER-EM SPACE */ \
  case 0x2005:  /* FOUR-PER-EM SPACE */ \
  case 0x2006:  /* SIX-PER-EM SPACE */ \
  case 0x2007:  /* FIGURE SPACE */ \
  case 0x2008:  /* PUNCTUATION SPACE */ \
  case 0x2009:  /* THIN SPACE */ \
  case 0x200a:  /* HAIR SPACE */ \
  case 0x202f:  /* NARROW NO-BREAK SPACE */ \
  case 0x205f:  /* MEDIUM MATHEMATICAL SPACE */ \
  case 0x3000   /* IDEOGRAPHIC SPACE */

#define HSPACE_BYTE_CASES \
  case CHAR_HT: \
  case CHAR_SPACE: \
  case CHAR_NBSP

#define HSPACE_CASES \
  HSPACE_BYTE_CASES: \
  HSPACE_MULTIBYTE_CASES

#define VSPACE_LIST \
  CHAR_LF, CHAR_VT, CHAR_FF, CHAR_CR, CHAR_NEL, 0x2028, 0x2029, NOTACHAR

#define VSPACE_MULTIBYTE_CASES \
  case 0x2028:    /* LINE SEPARATOR */ \
  case 0x2029     /* PARAGRAPH SEPARATOR */

#define VSPACE_BYTE_CASES \
  case CHAR_LF: \
  case CHAR_VT: \
  case CHAR_FF: \
  case CHAR_CR: \
  case CHAR_NEL

#define VSPACE_CASES \
  VSPACE_BYTE_CASES: \
  VSPACE_MULTIBYTE_CASES

/* -------------- EBCDIC environments -------------- */

#else
#define HSPACE_LIST CHAR_HT, CHAR_SPACE, CHAR_NBSP, NOTACHAR

#define HSPACE_BYTE_CASES \
  case CHAR_HT: \
  case CHAR_SPACE: \
  case CHAR_NBSP

#define HSPACE_CASES HSPACE_BYTE_CASES

#ifdef EBCDIC_NL25
#define VSPACE_LIST \
  CHAR_VT, CHAR_FF, CHAR_CR, CHAR_NEL, CHAR_LF, NOTACHAR
#else
#define VSPACE_LIST \
  CHAR_VT, CHAR_FF, CHAR_CR, CHAR_LF, CHAR_NEL, NOTACHAR
#endif

#define VSPACE_BYTE_CASES \
  case CHAR_LF: \
  case CHAR_VT: \
  case CHAR_FF: \
  case CHAR_CR: \
  case CHAR_NEL

#define VSPACE_CASES VSPACE_BYTE_CASES
#endif  /* EBCDIC */

/* -------------- End of whitespace macros -------------- */


/* PCRE2 is able to support several different kinds of newline (CR, LF, CRLF,
"any" and "anycrlf" at present). The following macros are used to package up
testing for newlines. NLBLOCK, PSSTART, and PSEND are defined in the various
modules to indicate in which datablock the parameters exist, and what the
start/end of string field names are. */

#define NLTYPE_FIXED    0     /* Newline is a fixed length string */
#define NLTYPE_ANY      1     /* Newline is any Unicode line ending */
#define NLTYPE_ANYCRLF  2     /* Newline is CR, LF, or CRLF */

/* This macro checks for a newline at the given position */

#define IS_NEWLINE(p) \
  ((NLBLOCK->nltype != NLTYPE_FIXED)? \
    ((p) < NLBLOCK->PSEND && \
     PRIV(is_newline)((p), NLBLOCK->nltype, NLBLOCK->PSEND, \
       &(NLBLOCK->nllen), utf)) \
    : \
    ((p) <= NLBLOCK->PSEND - NLBLOCK->nllen && \
     UCHAR21TEST(p) == NLBLOCK->nl[0] && \
     (NLBLOCK->nllen == 1 || UCHAR21TEST(p+1) == NLBLOCK->nl[1])       \
    ) \
  )

/* This macro checks for a newline immediately preceding the given position */

#define WAS_NEWLINE(p) \
  ((NLBLOCK->nltype != NLTYPE_FIXED)? \
    ((p) > NLBLOCK->PSSTART && \
     PRIV(was_newline)((p), NLBLOCK->nltype, NLBLOCK->PSSTART, \
       &(NLBLOCK->nllen), utf)) \
    : \
    ((p) >= NLBLOCK->PSSTART + NLBLOCK->nllen && \
     UCHAR21TEST(p - NLBLOCK->nllen) == NLBLOCK->nl[0] &&              \
     (NLBLOCK->nllen == 1 || UCHAR21TEST(p - NLBLOCK->nllen + 1) == NLBLOCK->nl[1]) \
    ) \
  )

/* Private flags containing information about the compiled pattern. The first
three must not be changed, because whichever is set is actually the number of
bytes in a code unit in that mode. */

#define PCRE2_MODE8         0x00000001u /* compiled in 8 bit mode */
#define PCRE2_MODE16        0x00000002u /* compiled in 16 bit mode */
#define PCRE2_MODE32        0x00000004u /* compiled in 32 bit mode */
#define PCRE2_FIRSTSET      0x00000010u /* first_code unit is set */
#define PCRE2_FIRSTCASELESS 0x00000020u /* caseless first code unit */
#define PCRE2_FIRSTMAPSET   0x00000040u /* bitmap of first code units is set */
#define PCRE2_LASTSET       0x00000080u /* last code unit is set */
#define PCRE2_LASTCASELESS  0x00000100u /* caseless last code unit */
#define PCRE2_STARTLINE     0x00000200u /* start after \n for multiline */
#define PCRE2_JCHANGED      0x00000400u /* j option used in pattern */
#define PCRE2_HASCRORLF     0x00000800u /* explicit \r or \n in pattern */
#define PCRE2_HASTHEN       0x00001000u /* pattern contains (*THEN) */
#define PCRE2_MATCH_EMPTY   0x00002000u /* pattern can match empty string */
#define PCRE2_BSR_SET       0x00004000u /* BSR was set in the pattern */
#define PCRE2_NL_SET        0x00008000u /* newline was set in the pattern */
#define PCRE2_NOTEMPTY_SET  0x00010000u /* (*NOTEMPTY) used        ) keep */
#define PCRE2_NE_ATST_SET   0x00020000u /* (*NOTEMPTY_ATSTART) used) together */
#define PCRE2_DEREF_TABLES  0x00040000u /* release character tables */
#define PCRE2_NOJIT         0x00080000u /* (*NOJIT) used */
#define PCRE2_HASBKPORX     0x00100000u /* contains \P, \p, or \X */
#define PCRE2_DUPCAPUSED    0x00200000u /* contains (?| */
#define PCRE2_HASBKC        0x00400000u /* contains \C */
#define PCRE2_HASACCEPT     0x00800000u /* contains (*ACCEPT) */
#define PCRE2_HASBSK        0x01000000u /* contains \K */

#define PCRE2_MODE_MASK     (PCRE2_MODE8 | PCRE2_MODE16 | PCRE2_MODE32)

/* Values for the matchedby field in a match data block. */

enum { PCRE2_MATCHEDBY_INTERPRETER,     /* pcre2_match() */
       PCRE2_MATCHEDBY_DFA_INTERPRETER, /* pcre2_dfa_match() */
       PCRE2_MATCHEDBY_JIT };           /* pcre2_jit_match() */

/* Values for the flags field in a match data block. */

#define PCRE2_MD_COPIED_SUBJECT  0x01u

/* Magic number to provide a small check against being handed junk. */

#define MAGIC_NUMBER  0x50435245UL   /* 'PCRE' */

/* The maximum remaining length of subject we are prepared to search for a
req_unit match from an anchored pattern. In 8-bit mode, memchr() is used and is
much faster than the search loop that has to be used in 16-bit and 32-bit
modes. */

#if PCRE2_CODE_UNIT_WIDTH == 8
#define REQ_CU_MAX       5000
#else
#define REQ_CU_MAX       2000
#endif

/* The maximum nesting depth for Unicode character class sets.
Currently fixed. Warning: the interpreter relies on this so it can encode
the operand stack in a uint32_t. A nesting limit of 15 implies (15*2+1)=31
stack operands required, due to the fact that we have two (and only two)
levels of operator precedence. In the UTS#18 syntax, you can write 'x&&y[z]'
and in Perl syntax you can write '(?[ x - y & (z) ])', both of which imply
pushing the match results for x & y to the stack. */

#define ECLASS_NEST_LIMIT  15

/* Offsets for the bitmap tables in the cbits set of tables. Each table
contains a set of bits for a class map. Some classes are built by combining
these tables. */

#define cbit_space     0      /* [:space:] or \s */
#define cbit_xdigit   32      /* [:xdigit:] */
#define cbit_digit    64      /* [:digit:] or \d */
#define cbit_upper    96      /* [:upper:] */
#define cbit_lower   128      /* [:lower:] */
#define cbit_word    160      /* [:word:] or \w */
#define cbit_graph   192      /* [:graph:] */
#define cbit_print   224      /* [:print:] */
#define cbit_punct   256      /* [:punct:] */
#define cbit_cntrl   288      /* [:cntrl:] */
#define cbit_length  320      /* Length of the cbits table */

/* Bit definitions for entries in the ctypes table. Do not change these values
without checking pcre2_jit_compile.c, which has an assertion to ensure that
ctype_word has the value 16. */

#define ctype_space    0x01
#define ctype_letter   0x02
#define ctype_lcletter 0x04
#define ctype_digit    0x08
#define ctype_word     0x10    /* alphanumeric or '_' */

/* Offsets of the various tables from the base tables pointer, and
total length of the tables. */

#define lcc_offset      0                           /* Lower case */
#define fcc_offset    256                           /* Flip case */
#define cbits_offset  512                           /* Character classes */
#define ctypes_offset (cbits_offset + cbit_length)  /* Character types */
#define TABLES_LENGTH (ctypes_offset + 256)

/* Private flags used in compile_context.optimization_flags */

#define PCRE2_OPTIM_AUTO_POSSESS    0x00000001u
#define PCRE2_OPTIM_DOTSTAR_ANCHOR  0x00000002u
#define PCRE2_OPTIM_START_OPTIMIZE  0x00000004u

#define PCRE2_OPTIMIZATION_ALL      0x00000007u

/* -------------------- Character and string names ------------------------ */

/* If PCRE2 is to support UTF-8 on EBCDIC platforms, we cannot use normal
character constants like '*' because the compiler would emit their EBCDIC code,
which is different from their ASCII/UTF-8 code. Instead we define macros for
the characters so that they always use the ASCII/UTF-8 code when UTF-8 support
is enabled. When UTF-8 support is not enabled, the definitions use character
literals. Both character and string versions of each character are needed, and
there are some longer strings as well.

This means that, on EBCDIC platforms, the PCRE2 library can handle either
EBCDIC, or UTF-8, but not both. To support both in the same compiled library
would need different lookups depending on whether PCRE2_UTF was set or not.
This would make it impossible to use characters in switch/case statements,
which would reduce performance. For a theoretical use (which nobody has asked
for) in a minority area (EBCDIC platforms), this is not sensible. Any
application that did need both could compile two versions of the library, using
macros to give the functions distinct names. */

#ifndef SUPPORT_UNICODE

/* UTF-8 support is not enabled; use the platform-dependent character literals
so that PCRE2 works in both ASCII and EBCDIC environments, but only in non-UTF
mode. Newline characters are problematic in EBCDIC. Though it has CR and LF
characters, a common practice has been to use its NL (0x15) character as the
line terminator in C-like processing environments. However, sometimes the LF
(0x25) character is used instead, according to this Unicode document:

http://unicode.org/standard/reports/tr13/tr13-5.html

PCRE2 defaults EBCDIC NL to 0x15, but has a build-time option to select 0x25
instead. Whichever is *not* chosen is defined as NEL.

In both ASCII and EBCDIC environments, CHAR_NL and CHAR_LF are synonyms for the
same code point. */

#ifdef EBCDIC

#ifndef EBCDIC_NL25
#define CHAR_NL                     '\x15'
#define CHAR_NEL                    '\x25'
#define STR_NL                      "\x15"
#define STR_NEL                     "\x25"
#else
#define CHAR_NL                     '\x25'
#define CHAR_NEL                    '\x15'
#define STR_NL                      "\x25"
#define STR_NEL                     "\x15"
#endif

#define CHAR_LF                     CHAR_NL
#define STR_LF                      STR_NL

#define CHAR_ESC                    '\047'
#define CHAR_DEL                    '\007'
#define CHAR_NBSP                   ((unsigned char)'\x41')
#define STR_ESC                     "\047"
#define STR_DEL                     "\007"

#else  /* Not EBCDIC */

/* In ASCII/Unicode, linefeed is '\n' and we equate this to NL for
compatibility. NEL is the Unicode newline character; make sure it is
a positive value. */

#if '\n' != 0x0a
#error "ASCII character '\n' is not 0x0a"
#endif

#define CHAR_LF                     '\n'
#define CHAR_NL                     CHAR_LF
#define CHAR_NEL                    ((unsigned char)'\x85')
#define CHAR_ESC                    '\033'
#define CHAR_DEL                    '\177'
#define CHAR_NBSP                   ((unsigned char)'\xa0')

#define STR_LF                      "\n"
#define STR_NL                      STR_LF
#define STR_NEL                     "\x85"
#define STR_ESC                     "\033"
#define STR_DEL                     "\177"

#endif  /* EBCDIC */

/* When we want to use EBCDIC with an ASCII compiler, for testing EBCDIC on
ASCII platforms, then we can hardcode an EBCDIC codepage (IBM-1047). */

#ifdef EBCDIC_IGNORING_COMPILER

#define CHAR_NUL                    '\000'
#define CHAR_HT                     '\005'
#define CHAR_VT                     '\013'
#define CHAR_FF                     '\014'
#define CHAR_CR                     '\015'
#define CHAR_BS                     '\026'
#define CHAR_BEL                    '\057'

#define CHAR_SPACE                  '\100'
#define CHAR_EXCLAMATION_MARK       '\132'
#define CHAR_QUOTATION_MARK         '\177'
#define CHAR_NUMBER_SIGN            '\173'
#define CHAR_DOLLAR_SIGN            '\133'
#define CHAR_PERCENT_SIGN           '\154'
#define CHAR_AMPERSAND              '\120'
#define CHAR_APOSTROPHE             '\175'
#define CHAR_LEFT_PARENTHESIS       '\115'
#define CHAR_RIGHT_PARENTHESIS      '\135'
#define CHAR_ASTERISK               '\134'
#define CHAR_PLUS                   '\116'
#define CHAR_COMMA                  '\153'
#define CHAR_MINUS                  '\140'
#define CHAR_DOT                    '\113'
#define CHAR_SLASH                  '\141'
#define CHAR_0                      ((unsigned char)'\xf0')
#define CHAR_1                      ((unsigned char)'\xf1')
#define CHAR_2                      ((unsigned char)'\xf2')
#define CHAR_3                      ((unsigned char)'\xf3')
#define CHAR_4                      ((unsigned char)'\xf4')
#define CHAR_5                      ((unsigned char)'\xf5')
#define CHAR_6                      ((unsigned char)'\xf6')
#define CHAR_7                      ((unsigned char)'\xf7')
#define CHAR_8                      ((unsigned char)'\xf8')
#define CHAR_9                      ((unsigned char)'\xf9')
#define CHAR_COLON                  '\172'
#define CHAR_SEMICOLON              '\136'
#define CHAR_LESS_THAN_SIGN         '\114'
#define CHAR_EQUALS_SIGN            '\176'
#define CHAR_GREATER_THAN_SIGN      '\156'
#define CHAR_QUESTION_MARK          '\157'
#define CHAR_COMMERCIAL_AT          '\174'
#define CHAR_A                      ((unsigned char)'\xc1')
#define CHAR_B                      ((unsigned char)'\xc2')
#define CHAR_C                      ((unsigned char)'\xc3')
#define CHAR_D                      ((unsigned char)'\xc4')
#define CHAR_E                      ((unsigned char)'\xc5')
#define CHAR_F                      ((unsigned char)'\xc6')
#define CHAR_G                      ((unsigned char)'\xc7')
#define CHAR_H                      ((unsigned char)'\xc8')
#define CHAR_I                      ((unsigned char)'\xc9')
#define CHAR_J                      ((unsigned char)'\xd1')
#define CHAR_K                      ((unsigned char)'\xd2')
#define CHAR_L                      ((unsigned char)'\xd3')
#define CHAR_M                      ((unsigned char)'\xd4')
#define CHAR_N                      ((unsigned char)'\xd5')
#define CHAR_O                      ((unsigned char)'\xd6')
#define CHAR_P                      ((unsigned char)'\xd7')
#define CHAR_Q                      ((unsigned char)'\xd8')
#define CHAR_R                      ((unsigned char)'\xd9')
#define CHAR_S                      ((unsigned char)'\xe2')
#define CHAR_T                      ((unsigned char)'\xe3')
#define CHAR_U                      ((unsigned char)'\xe4')
#define CHAR_V                      ((unsigned char)'\xe5')
#define CHAR_W                      ((unsigned char)'\xe6')
#define CHAR_X                      ((unsigned char)'\xe7')
#define CHAR_Y                      ((unsigned char)'\xe8')
#define CHAR_Z                      ((unsigned char)'\xe9')
#define CHAR_LEFT_SQUARE_BRACKET    ((unsigned char)'\xad')
#define CHAR_BACKSLASH              ((unsigned char)'\xe0')
#define CHAR_RIGHT_SQUARE_BRACKET   ((unsigned char)'\xbd')
#define CHAR_CIRCUMFLEX_ACCENT      '\137'
#define CHAR_UNDERSCORE             '\155'
#define CHAR_GRAVE_ACCENT           '\171'
#define CHAR_a                      ((unsigned char)'\x81')
#define CHAR_b                      ((unsigned char)'\x82')
#define CHAR_c                      ((unsigned char)'\x83')
#define CHAR_d                      ((unsigned char)'\x84')
#define CHAR_e                      ((unsigned char)'\x85')
#define CHAR_f                      ((unsigned char)'\x86')
#define CHAR_g                      ((unsigned char)'\x87')
#define CHAR_h                      ((unsigned char)'\x88')
#define CHAR_i                      ((unsigned char)'\x89')
#define CHAR_j                      ((unsigned char)'\x91')
#define CHAR_k                      ((unsigned char)'\x92')
#define CHAR_l                      ((unsigned char)'\x93')
#define CHAR_m                      ((unsigned char)'\x94')
#define CHAR_n                      ((unsigned char)'\x95')
#define CHAR_o                      ((unsigned char)'\x96')
#define CHAR_p                      ((unsigned char)'\x97')
#define CHAR_q                      ((unsigned char)'\x98')
#define CHAR_r                      ((unsigned char)'\x99')
#define CHAR_s                      ((unsigned char)'\xa2')
#define CHAR_t                      ((unsigned char)'\xa3')
#define CHAR_u                      ((unsigned char)'\xa4')
#define CHAR_v                      ((unsigned char)'\xa5')
#define CHAR_w                      ((unsigned char)'\xa6')
#define CHAR_x                      ((unsigned char)'\xa7')
#define CHAR_y                      ((unsigned char)'\xa8')
#define CHAR_z                      ((unsigned char)'\xa9')
#define CHAR_LEFT_CURLY_BRACKET     ((unsigned char)'\xc0')
#define CHAR_VERTICAL_LINE          '\117'
#define CHAR_RIGHT_CURLY_BRACKET    ((unsigned char)'\xd0')
#define CHAR_TILDE                  ((unsigned char)'\xa1')

#define STR_HT                      "\005"
#define STR_VT                      "\013"
#define STR_FF                      "\014"
#define STR_CR                      "\015"
#define STR_BS                      "\026"
#define STR_BEL                     "\057"

#define STR_SPACE                   "\100"
#define STR_EXCLAMATION_MARK        "\132"
#define STR_QUOTATION_MARK          "\177"
#define STR_NUMBER_SIGN             "\173"
#define STR_DOLLAR_SIGN             "\133"
#define STR_PERCENT_SIGN            "\154"
#define STR_AMPERSAND               "\120"
#define STR_APOSTROPHE              "\175"
#define STR_LEFT_PARENTHESIS        "\115"
#define STR_RIGHT_PARENTHESIS       "\135"
#define STR_ASTERISK                "\134"
#define STR_PLUS                    "\116"
#define STR_COMMA                   "\153"
#define STR_MINUS                   "\140"
#define STR_DOT                     "\113"
#define STR_SLASH                   "\141"
#define STR_0                       "\360"
#define STR_1                       "\361"
#define STR_2                       "\362"
#define STR_3                       "\363"
#define STR_4                       "\364"
#define STR_5                       "\365"
#define STR_6                       "\366"
#define STR_7                       "\367"
#define STR_8                       "\370"
#define STR_9                       "\371"
#define STR_COLON                   "\172"
#define STR_SEMICOLON               "\136"
#define STR_LESS_THAN_SIGN          "\114"
#define STR_EQUALS_SIGN             "\176"
#define STR_GREATER_THAN_SIGN       "\156"
#define STR_QUESTION_MARK           "\157"
#define STR_COMMERCIAL_AT           "\174"
#define STR_A                       "\301"
#define STR_B                       "\302"
#define STR_C                       "\303"
#define STR_D                       "\304"
#define STR_E                       "\305"
#define STR_F                       "\306"
#define STR_G                       "\307"
#define STR_H                       "\310"
#define STR_I                       "\311"
#define STR_J                       "\321"
#define STR_K                       "\322"
#define STR_L                       "\323"
#define STR_M                       "\324"
#define STR_N                       "\325"
#define STR_O                       "\326"
#define STR_P                       "\327"
#define STR_Q                       "\330"
#define STR_R                       "\331"
#define STR_S                       "\342"
#define STR_T                       "\343"
#define STR_U                       "\344"
#define STR_V                       "\345"
#define STR_W                       "\346"
#define STR_X                       "\347"
#define STR_Y                       "\350"
#define STR_Z                       "\351"
#define STR_LEFT_SQUARE_BRACKET     "\255"
#define STR_BACKSLASH               "\340"
#define STR_RIGHT_SQUARE_BRACKET    "\275"
#define STR_CIRCUMFLEX_ACCENT       "\137"
#define STR_UNDERSCORE              "\155"
#define STR_GRAVE_ACCENT            "\171"
#define STR_a                       "\201"
#define STR_b                       "\202"
#define STR_c                       "\203"
#define STR_d                       "\204"
#define STR_e                       "\205"
#define STR_f                       "\206"
#define STR_g                       "\207"
#define STR_h                       "\210"
#define STR_i                       "\211"
#define STR_j                       "\221"
#define STR_k                       "\222"
#define STR_l                       "\223"
#define STR_m                       "\224"
#define STR_n                       "\225"
#define STR_o                       "\226"
#define STR_p                       "\227"
#define STR_q                       "\230"
#define STR_r                       "\231"
#define STR_s                       "\242"
#define STR_t                       "\243"
#define STR_u                       "\244"
#define STR_v                       "\245"
#define STR_w                       "\246"
#define STR_x                       "\247"
#define STR_y                       "\250"
#define STR_z                       "\251"
#define STR_LEFT_CURLY_BRACKET      "\300"
#define STR_VERTICAL_LINE           "\117"
#define STR_RIGHT_CURLY_BRACKET     "\320"
#define STR_TILDE                   "\241"

#else  /* EBCDIC_IGNORING_COMPILER */

/* Otherwise, on a real EBCDIC compiler or an ASCII compiler, we can use simple
string and character literals. */

#ifdef EBCDIC
#if 'a' != 0x81
#error "EBCDIC character 'a' is not 0x81"
#endif
#else
#if 'a' != 0x61
#error "ASCII character 'a' is not 0x61"
#endif
#endif

#define CHAR_NUL                    '\0'
#define CHAR_HT                     '\t'
#define CHAR_VT                     '\v'
#define CHAR_FF                     '\f'
#define CHAR_CR                     '\r'
#define CHAR_BS                     '\b'
#define CHAR_BEL                    '\a'

#define CHAR_SPACE                  ' '
#define CHAR_EXCLAMATION_MARK       '!'
#define CHAR_QUOTATION_MARK         '"'
#define CHAR_NUMBER_SIGN            '#'
#define CHAR_DOLLAR_SIGN            '$'
#define CHAR_PERCENT_SIGN           '%'
#define CHAR_AMPERSAND              '&'
#define CHAR_APOSTROPHE             '\''
#define CHAR_LEFT_PARENTHESIS       '('
#define CHAR_RIGHT_PARENTHESIS      ')'
#define CHAR_ASTERISK               '*'
#define CHAR_PLUS                   '+'
#define CHAR_COMMA                  ','
#define CHAR_MINUS                  '-'
#define CHAR_DOT                    '.'
#define CHAR_SLASH                  '/'
#define CHAR_0                      '0'
#define CHAR_1                      '1'
#define CHAR_2                      '2'
#define CHAR_3                      '3'
#define CHAR_4                      '4'
#define CHAR_5                      '5'
#define CHAR_6                      '6'
#define CHAR_7                      '7'
#define CHAR_8                      '8'
#define CHAR_9                      '9'
#define CHAR_COLON                  ':'
#define CHAR_SEMICOLON              ';'
#define CHAR_LESS_THAN_SIGN         '<'
#define CHAR_EQUALS_SIGN            '='
#define CHAR_GREATER_THAN_SIGN      '>'
#define CHAR_QUESTION_MARK          '?'
#define CHAR_COMMERCIAL_AT          '@'
#define CHAR_A                      'A'
#define CHAR_B                      'B'
#define CHAR_C                      'C'
#define CHAR_D                      'D'
#define CHAR_E                      'E'
#define CHAR_F                      'F'
#define CHAR_G                      'G'
#define CHAR_H                      'H'
#define CHAR_I                      'I'
#define CHAR_J                      'J'
#define CHAR_K                      'K'
#define CHAR_L                      'L'
#define CHAR_M                      'M'
#define CHAR_N                      'N'
#define CHAR_O                      'O'
#define CHAR_P                      'P'
#define CHAR_Q                      'Q'
#define CHAR_R                      'R'
#define CHAR_S                      'S'
#define CHAR_T                      'T'
#define CHAR_U                      'U'
#define CHAR_V                      'V'
#define CHAR_W                      'W'
#define CHAR_X                      'X'
#define CHAR_Y                      'Y'
#define CHAR_Z                      'Z'
#define CHAR_LEFT_SQUARE_BRACKET    '['
#define CHAR_BACKSLASH              '\\'
#define CHAR_RIGHT_SQUARE_BRACKET   ']'
#define CHAR_CIRCUMFLEX_ACCENT      '^'
#define CHAR_UNDERSCORE             '_'
#define CHAR_GRAVE_ACCENT           '`'
#define CHAR_a                      'a'
#define CHAR_b                      'b'
#define CHAR_c                      'c'
#define CHAR_d                      'd'
#define CHAR_e                      'e'
#define CHAR_f                      'f'
#define CHAR_g                      'g'
#define CHAR_h                      'h'
#define CHAR_i                      'i'
#define CHAR_j                      'j'
#define CHAR_k                      'k'
#define CHAR_l                      'l'
#define CHAR_m                      'm'
#define CHAR_n                      'n'
#define CHAR_o                      'o'
#define CHAR_p                      'p'
#define CHAR_q                      'q'
#define CHAR_r                      'r'
#define CHAR_s                      's'
#define CHAR_t                      't'
#define CHAR_u                      'u'
#define CHAR_v                      'v'
#define CHAR_w                      'w'
#define CHAR_x                      'x'
#define CHAR_y                      'y'
#define CHAR_z                      'z'
#define CHAR_LEFT_CURLY_BRACKET     '{'
#define CHAR_VERTICAL_LINE          '|'
#define CHAR_RIGHT_CURLY_BRACKET    '}'
#define CHAR_TILDE                  '~'

#define STR_HT                      "\t"
#define STR_VT                      "\v"
#define STR_FF                      "\f"
#define STR_CR                      "\r"
#define STR_BS                      "\b"
#define STR_BEL                     "\a"

#define STR_SPACE                   " "
#define STR_EXCLAMATION_MARK        "!"
#define STR_QUOTATION_MARK          "\""
#define STR_NUMBER_SIGN             "#"
#define STR_DOLLAR_SIGN             "$"
#define STR_PERCENT_SIGN            "%"
#define STR_AMPERSAND               "&"
#define STR_APOSTROPHE              "'"
#define STR_LEFT_PARENTHESIS        "("
#define STR_RIGHT_PARENTHESIS       ")"
#define STR_ASTERISK                "*"
#define STR_PLUS                    "+"
#define STR_COMMA                   ","
#define STR_MINUS                   "-"
#define STR_DOT                     "."
#define STR_SLASH                   "/"
#define STR_0                       "0"
#define STR_1                       "1"
#define STR_2                       "2"
#define STR_3                       "3"
#define STR_4                       "4"
#define STR_5                       "5"
#define STR_6                       "6"
#define STR_7                       "7"
#define STR_8                       "8"
#define STR_9                       "9"
#define STR_COLON                   ":"
#define STR_SEMICOLON               ";"
#define STR_LESS_THAN_SIGN          "<"
#define STR_EQUALS_SIGN             "="
#define STR_GREATER_THAN_SIGN       ">"
#define STR_QUESTION_MARK           "?"
#define STR_COMMERCIAL_AT           "@"
#define STR_A                       "A"
#define STR_B                       "B"
#define STR_C                       "C"
#define STR_D                       "D"
#define STR_E                       "E"
#define STR_F                       "F"
#define STR_G                       "G"
#define STR_H                       "H"
#define STR_I                       "I"
#define STR_J                       "J"
#define STR_K                       "K"
#define STR_L                       "L"
#define STR_M                       "M"
#define STR_N                       "N"
#define STR_O                       "O"
#define STR_P                       "P"
#define STR_Q                       "Q"
#define STR_R                       "R"
#define STR_S                       "S"
#define STR_T                       "T"
#define STR_U                       "U"
#define STR_V                       "V"
#define STR_W                       "W"
#define STR_X                       "X"
#define STR_Y                       "Y"
#define STR_Z                       "Z"
#define STR_LEFT_SQUARE_BRACKET     "["
#define STR_BACKSLASH               "\\"
#define STR_RIGHT_SQUARE_BRACKET    "]"
#define STR_CIRCUMFLEX_ACCENT       "^"
#define STR_UNDERSCORE              "_"
#define STR_GRAVE_ACCENT            "`"
#define STR_a                       "a"
#define STR_b                       "b"
#define STR_c                       "c"
#define STR_d                       "d"
#define STR_e                       "e"
#define STR_f                       "f"
#define STR_g                       "g"
#define STR_h                       "h"
#define STR_i                       "i"
#define STR_j                       "j"
#define STR_k                       "k"
#define STR_l                       "l"
#define STR_m                       "m"
#define STR_n                       "n"
#define STR_o                       "o"
#define STR_p                       "p"
#define STR_q                       "q"
#define STR_r                       "r"
#define STR_s                       "s"
#define STR_t                       "t"
#define STR_u                       "u"
#define STR_v                       "v"
#define STR_w                       "w"
#define STR_x                       "x"
#define STR_y                       "y"
#define STR_z                       "z"
#define STR_LEFT_CURLY_BRACKET      "{"
#define STR_VERTICAL_LINE           "|"
#define STR_RIGHT_CURLY_BRACKET     "}"
#define STR_TILDE                   "~"

#endif  /* EBCDIC_WITH_ASCII_COMPILER */

#else  /* SUPPORT_UNICODE */

/* UTF-8 support is enabled; always use UTF-8 (=ASCII) character codes. This
works in both modes non-EBCDIC platforms, and on EBCDIC platforms in UTF-8 mode
only. */

#define CHAR_HT                     '\011'
#define CHAR_VT                     '\013'
#define CHAR_FF                     '\014'
#define CHAR_CR                     '\015'
#define CHAR_LF                     '\012'
#define CHAR_NL                     CHAR_LF
#define CHAR_NEL                    ((unsigned char)'\x85')
#define CHAR_BS                     '\010'
#define CHAR_BEL                    '\007'
#define CHAR_ESC                    '\033'
#define CHAR_DEL                    '\177'

#define CHAR_NUL                    '\0'
#define CHAR_SPACE                  '\040'
#define CHAR_EXCLAMATION_MARK       '\041'
#define CHAR_QUOTATION_MARK         '\042'
#define CHAR_NUMBER_SIGN            '\043'
#define CHAR_DOLLAR_SIGN            '\044'
#define CHAR_PERCENT_SIGN           '\045'
#define CHAR_AMPERSAND              '\046'
#define CHAR_APOSTROPHE             '\047'
#define CHAR_LEFT_PARENTHESIS       '\050'
#define CHAR_RIGHT_PARENTHESIS      '\051'
#define CHAR_ASTERISK               '\052'
#define CHAR_PLUS                   '\053'
#define CHAR_COMMA                  '\054'
#define CHAR_MINUS                  '\055'
#define CHAR_DOT                    '\056'
#define CHAR_SLASH                  '\057'
#define CHAR_0                      '\060'
#define CHAR_1                      '\061'
#define CHAR_2                      '\062'
#define CHAR_3                      '\063'
#define CHAR_4                      '\064'
#define CHAR_5                      '\065'
#define CHAR_6                      '\066'
#define CHAR_7                      '\067'
#define CHAR_8                      '\070'
#define CHAR_9                      '\071'
#define CHAR_COLON                  '\072'
#define CHAR_SEMICOLON              '\073'
#define CHAR_LESS_THAN_SIGN         '\074'
#define CHAR_EQUALS_SIGN            '\075'
#define CHAR_GREATER_THAN_SIGN      '\076'
#define CHAR_QUESTION_MARK          '\077'
#define CHAR_COMMERCIAL_AT          '\100'
#define CHAR_A                      '\101'
#define CHAR_B                      '\102'
#define CHAR_C                      '\103'
#define CHAR_D                      '\104'
#define CHAR_E                      '\105'
#define CHAR_F                      '\106'
#define CHAR_G                      '\107'
#define CHAR_H                      '\110'
#define CHAR_I                      '\111'
#define CHAR_J                      '\112'
#define CHAR_K                      '\113'
#define CHAR_L                      '\114'
#define CHAR_M                      '\115'
#define CHAR_N                      '\116'
#define CHAR_O                      '\117'
#define CHAR_P                      '\120'
#define CHAR_Q                      '\121'
#define CHAR_R                      '\122'
#define CHAR_S                      '\123'
#define CHAR_T                      '\124'
#define CHAR_U                      '\125'
#define CHAR_V                      '\126'
#define CHAR_W                      '\127'
#define CHAR_X                      '\130'
#define CHAR_Y                      '\131'
#define CHAR_Z                      '\132'
#define CHAR_LEFT_SQUARE_BRACKET    '\133'
#define CHAR_BACKSLASH              '\134'
#define CHAR_RIGHT_SQUARE_BRACKET   '\135'
#define CHAR_CIRCUMFLEX_ACCENT      '\136'
#define CHAR_UNDERSCORE             '\137'
#define CHAR_GRAVE_ACCENT           '\140'
#define CHAR_a                      '\141'
#define CHAR_b                      '\142'
#define CHAR_c                      '\143'
#define CHAR_d                      '\144'
#define CHAR_e                      '\145'
#define CHAR_f                      '\146'
#define CHAR_g                      '\147'
#define CHAR_h                      '\150'
#define CHAR_i                      '\151'
#define CHAR_j                      '\152'
#define CHAR_k                      '\153'
#define CHAR_l                      '\154'
#define CHAR_m                      '\155'
#define CHAR_n                      '\156'
#define CHAR_o                      '\157'
#define CHAR_p                      '\160'
#define CHAR_q                      '\161'
#define CHAR_r                      '\162'
#define CHAR_s                      '\163'
#define CHAR_t                      '\164'
#define CHAR_u                      '\165'
#define CHAR_v                      '\166'
#define CHAR_w                      '\167'
#define CHAR_x                      '\170'
#define CHAR_y                      '\171'
#define CHAR_z                      '\172'
#define CHAR_LEFT_CURLY_BRACKET     '\173'
#define CHAR_VERTICAL_LINE          '\174'
#define CHAR_RIGHT_CURLY_BRACKET    '\175'
#define CHAR_TILDE                  '\176'
#define CHAR_NBSP                   ((unsigned char)'\xa0')

#define STR_HT                      "\011"
#define STR_VT                      "\013"
#define STR_FF                      "\014"
#define STR_CR                      "\015"
#define STR_NL                      "\012"
#define STR_BS                      "\010"
#define STR_BEL                     "\007"
#define STR_ESC                     "\033"
#define STR_DEL                     "\177"

#define STR_SPACE                   "\040"
#define STR_EXCLAMATION_MARK        "\041"
#define STR_QUOTATION_MARK          "\042"
#define STR_NUMBER_SIGN             "\043"
#define STR_DOLLAR_SIGN             "\044"
#define STR_PERCENT_SIGN            "\045"
#define STR_AMPERSAND               "\046"
#define STR_APOSTROPHE              "\047"
#define STR_LEFT_PARENTHESIS        "\050"
#define STR_RIGHT_PARENTHESIS       "\051"
#define STR_ASTERISK                "\052"
#define STR_PLUS                    "\053"
#define STR_COMMA                   "\054"
#define STR_MINUS                   "\055"
#define STR_DOT                     "\056"
#define STR_SLASH                   "\057"
#define STR_0                       "\060"
#define STR_1                       "\061"
#define STR_2                       "\062"
#define STR_3                       "\063"
#define STR_4                       "\064"
#define STR_5                       "\065"
#define STR_6                       "\066"
#define STR_7                       "\067"
#define STR_8                       "\070"
#define STR_9                       "\071"
#define STR_COLON                   "\072"
#define STR_SEMICOLON               "\073"
#define STR_LESS_THAN_SIGN          "\074"
#define STR_EQUALS_SIGN             "\075"
#define STR_GREATER_THAN_SIGN       "\076"
#define STR_QUESTION_MARK           "\077"
#define STR_COMMERCIAL_AT           "\100"
#define STR_A                       "\101"
#define STR_B                       "\102"
#define STR_C                       "\103"
#define STR_D                       "\104"
#define STR_E                       "\105"
#define STR_F                       "\106"
#define STR_G                       "\107"
#define STR_H                       "\110"
#define STR_I                       "\111"
#define STR_J                       "\112"
#define STR_K                       "\113"
#define STR_L                       "\114"
#define STR_M                       "\115"
#define STR_N                       "\116"
#define STR_O                       "\117"
#define STR_P                       "\120"
#define STR_Q                       "\121"
#define STR_R                       "\122"
#define STR_S                       "\123"
#define STR_T                       "\124"
#define STR_U                       "\125"
#define STR_V                       "\126"
#define STR_W                       "\127"
#define STR_X                       "\130"
#define STR_Y                       "\131"
#define STR_Z                       "\132"
#define STR_LEFT_SQUARE_BRACKET     "\133"
#define STR_BACKSLASH               "\134"
#define STR_RIGHT_SQUARE_BRACKET    "\135"
#define STR_CIRCUMFLEX_ACCENT       "\136"
#define STR_UNDERSCORE              "\137"
#define STR_GRAVE_ACCENT            "\140"
#define STR_a                       "\141"
#define STR_b                       "\142"
#define STR_c                       "\143"
#define STR_d                       "\144"
#define STR_e                       "\145"
#define STR_f                       "\146"
#define STR_g                       "\147"
#define STR_h                       "\150"
#define STR_i                       "\151"
#define STR_j                       "\152"
#define STR_k                       "\153"
#define STR_l                       "\154"
#define STR_m                       "\155"
#define STR_n                       "\156"
#define STR_o                       "\157"
#define STR_p                       "\160"
#define STR_q                       "\161"
#define STR_r                       "\162"
#define STR_s                       "\163"
#define STR_t                       "\164"
#define STR_u                       "\165"
#define STR_v                       "\166"
#define STR_w                       "\167"
#define STR_x                       "\170"
#define STR_y                       "\171"
#define STR_z                       "\172"
#define STR_LEFT_CURLY_BRACKET      "\173"
#define STR_VERTICAL_LINE           "\174"
#define STR_RIGHT_CURLY_BRACKET     "\175"
#define STR_TILDE                   "\176"

#endif  /* SUPPORT_UNICODE */


#define STRING_ACCEPT0               STR_A STR_C STR_C STR_E STR_P STR_T "\0"
#define STRING_COMMIT0               STR_C STR_O STR_M STR_M STR_I STR_T "\0"
#define STRING_F0                    STR_F "\0"
#define STRING_FAIL0                 STR_F STR_A STR_I STR_L "\0"
#define STRING_MARK0                 STR_M STR_A STR_R STR_K "\0"
#define STRING_PRUNE0                STR_P STR_R STR_U STR_N STR_E "\0"
#define STRING_SKIP0                 STR_S STR_K STR_I STR_P "\0"
#define STRING_THEN                  STR_T STR_H STR_E STR_N

#define STRING_atomic0               STR_a STR_t STR_o STR_m STR_i STR_c "\0"
#define STRING_pla0                  STR_p STR_l STR_a "\0"
#define STRING_plb0                  STR_p STR_l STR_b "\0"
#define STRING_napla0                STR_n STR_a STR_p STR_l STR_a "\0"
#define STRING_naplb0                STR_n STR_a STR_p STR_l STR_b "\0"
#define STRING_nla0                  STR_n STR_l STR_a "\0"
#define STRING_nlb0                  STR_n STR_l STR_b "\0"
#define STRING_scs0                  STR_s STR_c STR_s "\0"
#define STRING_sr0                   STR_s STR_r "\0"
#define STRING_asr0                  STR_a STR_s STR_r "\0"
#define STRING_positive_lookahead0   STR_p STR_o STR_s STR_i STR_t STR_i STR_v STR_e STR_UNDERSCORE STR_l STR_o STR_o STR_k STR_a STR_h STR_e STR_a STR_d "\0"
#define STRING_positive_lookbehind0  STR_p STR_o STR_s STR_i STR_t STR_i STR_v STR_e STR_UNDERSCORE STR_l STR_o STR_o STR_k STR_b STR_e STR_h STR_i STR_n STR_d "\0"
#define STRING_non_atomic_positive_lookahead0   STR_n STR_o STR_n STR_UNDERSCORE STR_a STR_t STR_o STR_m STR_i STR_c STR_UNDERSCORE STR_p STR_o STR_s STR_i STR_t STR_i STR_v STR_e STR_UNDERSCORE STR_l STR_o STR_o STR_k STR_a STR_h STR_e STR_a STR_d "\0"
#define STRING_non_atomic_positive_lookbehind0  STR_n STR_o STR_n STR_UNDERSCORE STR_a STR_t STR_o STR_m STR_i STR_c STR_UNDERSCORE STR_p STR_o STR_s STR_i STR_t STR_i STR_v STR_e STR_UNDERSCORE STR_l STR_o STR_o STR_k STR_b STR_e STR_h STR_i STR_n STR_d "\0"
#define STRING_negative_lookahead0   STR_n STR_e STR_g STR_a STR_t STR_i STR_v STR_e STR_UNDERSCORE STR_l STR_o STR_o STR_k STR_a STR_h STR_e STR_a STR_d "\0"
#define STRING_negative_lookbehind0  STR_n STR_e STR_g STR_a STR_t STR_i STR_v STR_e STR_UNDERSCORE STR_l STR_o STR_o STR_k STR_b STR_e STR_h STR_i STR_n STR_d "\0"
#define STRING_script_run0           STR_s STR_c STR_r STR_i STR_p STR_t STR_UNDERSCORE STR_r STR_u STR_n "\0"
#define STRING_atomic_script_run     STR_a STR_t STR_o STR_m STR_i STR_c STR_UNDERSCORE STR_s STR_c STR_r STR_i STR_p STR_t STR_UNDERSCORE STR_r STR_u STR_n
#define STRING_scan_substring0       STR_s STR_c STR_a STR_n STR_UNDERSCORE STR_s STR_u STR_b STR_s STR_t STR_r STR_i STR_n STR_g "\0"

#define STRING_alpha0                STR_a STR_l STR_p STR_h STR_a "\0"
#define STRING_lower0                STR_l STR_o STR_w STR_e STR_r "\0"
#define STRING_upper0                STR_u STR_p STR_p STR_e STR_r "\0"
#define STRING_alnum0                STR_a STR_l STR_n STR_u STR_m "\0"
#define STRING_ascii0                STR_a STR_s STR_c STR_i STR_i "\0"
#define STRING_blank0                STR_b STR_l STR_a STR_n STR_k "\0"
#define STRING_cntrl0                STR_c STR_n STR_t STR_r STR_l "\0"
#define STRING_digit0                STR_d STR_i STR_g STR_i STR_t "\0"
#define STRING_graph0                STR_g STR_r STR_a STR_p STR_h "\0"
#define STRING_print0                STR_p STR_r STR_i STR_n STR_t "\0"
#define STRING_punct0                STR_p STR_u STR_n STR_c STR_t "\0"
#define STRING_space0                STR_s STR_p STR_a STR_c STR_e "\0"
#define STRING_word0                 STR_w STR_o STR_r STR_d       "\0"
#define STRING_xdigit                STR_x STR_d STR_i STR_g STR_i STR_t

#define STRING_DEFINE                STR_D STR_E STR_F STR_I STR_N STR_E
#define STRING_VERSION               STR_V STR_E STR_R STR_S STR_I STR_O STR_N
#define STRING_WEIRD_STARTWORD       STR_LEFT_SQUARE_BRACKET STR_COLON STR_LESS_THAN_SIGN STR_COLON STR_RIGHT_SQUARE_BRACKET STR_RIGHT_SQUARE_BRACKET
#define STRING_WEIRD_ENDWORD         STR_LEFT_SQUARE_BRACKET STR_COLON STR_GREATER_THAN_SIGN STR_COLON STR_RIGHT_SQUARE_BRACKET STR_RIGHT_SQUARE_BRACKET

#define STRING_CR_RIGHTPAR                STR_C STR_R STR_RIGHT_PARENTHESIS
#define STRING_LF_RIGHTPAR                STR_L STR_F STR_RIGHT_PARENTHESIS
#define STRING_CRLF_RIGHTPAR              STR_C STR_R STR_L STR_F STR_RIGHT_PARENTHESIS
#define STRING_ANY_RIGHTPAR               STR_A STR_N STR_Y STR_RIGHT_PARENTHESIS
#define STRING_ANYCRLF_RIGHTPAR           STR_A STR_N STR_Y STR_C STR_R STR_L STR_F STR_RIGHT_PARENTHESIS
#define STRING_NUL_RIGHTPAR               STR_N STR_U STR_L STR_RIGHT_PARENTHESIS
#define STRING_BSR_ANYCRLF_RIGHTPAR       STR_B STR_S STR_R STR_UNDERSCORE STR_A STR_N STR_Y STR_C STR_R STR_L STR_F STR_RIGHT_PARENTHESIS
#define STRING_BSR_UNICODE_RIGHTPAR       STR_B STR_S STR_R STR_UNDERSCORE STR_U STR_N STR_I STR_C STR_O STR_D STR_E STR_RIGHT_PARENTHESIS
#define STRING_UTF8_RIGHTPAR              STR_U STR_T STR_F STR_8 STR_RIGHT_PARENTHESIS
#define STRING_UTF16_RIGHTPAR             STR_U STR_T STR_F STR_1 STR_6 STR_RIGHT_PARENTHESIS
#define STRING_UTF32_RIGHTPAR             STR_U STR_T STR_F STR_3 STR_2 STR_RIGHT_PARENTHESIS
#define STRING_UTF_RIGHTPAR               STR_U STR_T STR_F STR_RIGHT_PARENTHESIS
#define STRING_UCP_RIGHTPAR               STR_U STR_C STR_P STR_RIGHT_PARENTHESIS
#define STRING_NO_AUTO_POSSESS_RIGHTPAR   STR_N STR_O STR_UNDERSCORE STR_A STR_U STR_T STR_O STR_UNDERSCORE STR_P STR_O STR_S STR_S STR_E STR_S STR_S STR_RIGHT_PARENTHESIS
#define STRING_NO_DOTSTAR_ANCHOR_RIGHTPAR STR_N STR_O STR_UNDERSCORE STR_D STR_O STR_T STR_S STR_T STR_A STR_R STR_UNDERSCORE STR_A STR_N STR_C STR_H STR_O STR_R STR_RIGHT_PARENTHESIS
#define STRING_NO_JIT_RIGHTPAR            STR_N STR_O STR_UNDERSCORE STR_J STR_I STR_T STR_RIGHT_PARENTHESIS
#define STRING_NO_START_OPT_RIGHTPAR      STR_N STR_O STR_UNDERSCORE STR_S STR_T STR_A STR_R STR_T STR_UNDERSCORE STR_O STR_P STR_T STR_RIGHT_PARENTHESIS
#define STRING_NOTEMPTY_RIGHTPAR          STR_N STR_O STR_T STR_E STR_M STR_P STR_T STR_Y STR_RIGHT_PARENTHESIS
#define STRING_NOTEMPTY_ATSTART_RIGHTPAR  STR_N STR_O STR_T STR_E STR_M STR_P STR_T STR_Y STR_UNDERSCORE STR_A STR_T STR_S STR_T STR_A STR_R STR_T STR_RIGHT_PARENTHESIS
#define STRING_CASELESS_RESTRICT_RIGHTPAR STR_C STR_A STR_S STR_E STR_L STR_E STR_S STR_S STR_UNDERSCORE STR_R STR_E STR_S STR_T STR_R STR_I STR_C STR_T STR_RIGHT_PARENTHESIS
#define STRING_TURKISH_CASING_RIGHTPAR    STR_T STR_U STR_R STR_K STR_I STR_S STR_H STR_UNDERSCORE STR_C STR_A STR_S STR_I STR_N STR_G STR_RIGHT_PARENTHESIS
#define STRING_LIMIT_HEAP_EQ              STR_L STR_I STR_M STR_I STR_T STR_UNDERSCORE STR_H STR_E STR_A STR_P STR_EQUALS_SIGN
#define STRING_LIMIT_MATCH_EQ             STR_L STR_I STR_M STR_I STR_T STR_UNDERSCORE STR_M STR_A STR_T STR_C STR_H STR_EQUALS_SIGN
#define STRING_LIMIT_DEPTH_EQ             STR_L STR_I STR_M STR_I STR_T STR_UNDERSCORE STR_D STR_E STR_P STR_T STR_H STR_EQUALS_SIGN
#define STRING_LIMIT_RECURSION_EQ         STR_L STR_I STR_M STR_I STR_T STR_UNDERSCORE STR_R STR_E STR_C STR_U STR_R STR_S STR_I STR_O STR_N STR_EQUALS_SIGN
#define STRING_MARK                       STR_M STR_A STR_R STR_K

#define STRING_bc                         STR_b STR_c
#define STRING_bidiclass                  STR_b STR_i STR_d STR_i STR_c STR_l STR_a STR_s STR_s
#define STRING_sc                         STR_s STR_c
#define STRING_script                     STR_s STR_c STR_r STR_i STR_p STR_t
#define STRING_scriptextensions           STR_s STR_c STR_r STR_i STR_p STR_t STR_e STR_x STR_t STR_e STR_n STR_s STR_i STR_o STR_n STR_s
#define STRING_scx                        STR_s STR_c STR_x


/* -------------------- End of character and string names -------------------*/

/* -------------------- Definitions for compiled patterns -------------------*/

/* Codes for different types of Unicode property. If these definitions are
changed, the autopossessifying table in pcre2_auto_possess.c must be updated to
match. */

#define PT_LAMP       0    /* L& - the union of Lu, Ll, Lt */
#define PT_GC         1    /* Specified general characteristic (e.g. L) */
#define PT_PC         2    /* Specified particular characteristic (e.g. Lu) */
#define PT_SC         3    /* Script only (e.g. Han) */
#define PT_SCX        4    /* Script extensions (includes SC) */
#define PT_ALNUM      5    /* Alphanumeric - the union of L and N */
#define PT_SPACE      6    /* Perl space - general category Z plus 9,10,12,13 */
#define PT_PXSPACE    7    /* POSIX space - Z plus 9,10,11,12,13 */
#define PT_WORD       8    /* Word - L, N, Mn, or Pc */
#define PT_CLIST      9    /* Pseudo-property: match character list */
#define PT_UCNC      10    /* Universal Character nameable character */
#define PT_BIDICL    11    /* Specified bidi class */
#define PT_BOOL      12    /* Boolean property */
#define PT_ANY       13    /* Must be the last entry!
                              Any property - matches all chars */
#define PT_TABSIZE PT_ANY  /* Size of square table for autopossessify tests */

/* The following special properties are used only in XCLASS items, when POSIX
classes are specified and PCRE2_UCP is set - in other words, for Unicode
handling of these classes. They are not available via the \p or \P escapes like
those in the above list, and so they do not take part in the autopossessifying
table. */

#define PT_PXGRAPH   14    /* [:graph:] - characters that mark the paper */
#define PT_PXPRINT   15    /* [:print:] - [:graph:] plus non-control spaces */
#define PT_PXPUNCT   16    /* [:punct:] - punctuation characters */
#define PT_PXXDIGIT  17    /* [:xdigit:] - hex digits */

/* This value is used when parsing \p and \P escapes to indicate that neither
\p{script:...} nor \p{scx:...} has been encountered. */

#define PT_NOTSCRIPT 255

/* Flag bits and data types for the extended class (OP_XCLASS) for classes that
contain characters with values greater than 255. */

#define XCL_NOT      0x01  /* Flag: this is a negative class */
#define XCL_MAP      0x02  /* Flag: a 32-byte map is present */
#define XCL_HASPROP  0x04  /* Flag: property checks are present. */

#define XCL_END      0     /* Marks end of individual items */
#define XCL_SINGLE   1     /* Single item (one multibyte char) follows */
#define XCL_RANGE    2     /* A range (two multibyte chars) follows */
#define XCL_PROP     3     /* Unicode property (2-byte property code follows) */
#define XCL_NOTPROP  4     /* Unicode inverted property (ditto) */
/* This value represents the beginning of character lists. The value
is 16 bit long, and stored as a high and low byte pair in 8 bit mode.
The lower 12 bit contains information about character lists (see later). */
#define XCL_LIST     (sizeof(PCRE2_UCHAR) == 1 ? 0x10 : 0x1000)

/* When a character class contains many characters/ranges,
they are stored in character lists. There are four character
lists which contain characters/ranges within a given range.

The name, character range and item size for each list:
Low16    [0x100 - 0x7fff]            16 bit items
High16   [0x8000 - 0xffff]           16 bit items
Low32    [0x10000 - 0x7fffffff]      32 bit items
High32   [0x80000000 - 0xffffffff]   32 bit items

The Low32 character list is used only when utf encoding or 32 bit
character width is enabled, and the High32 character is used only
when 32 bit character width is enabled.

Each character list contain items. The lowest bit represents that
an item is the beginning of a range (bit is cleared), or not (bit
is set). The other bits represent the character shifted left by
one, so its highest bit is discarded. Due to the layout of character
lists, the highest bit of a character is always known:

Low16 and Low32: the highest bit is always zero
High16 and High32: the highest bit is always one

The items are ordered in increasing order, so binary search can be
used to find the lower bound of an input character. The lower bound
is the highest item, which value is less or equal than the input
character. If the lower bit of the item is cleard, or the character
stored in the item equals to the input character, the input
character is in the character list. */

/* Character list constants. */
#define XCL_CHAR_LIST_LOW_16_START 0x100
#define XCL_CHAR_LIST_LOW_16_END 0x7fff
#define XCL_CHAR_LIST_LOW_16_ADD 0x0

#define XCL_CHAR_LIST_HIGH_16_START 0x8000
#define XCL_CHAR_LIST_HIGH_16_END 0xffff
#define XCL_CHAR_LIST_HIGH_16_ADD 0x8000

#define XCL_CHAR_LIST_LOW_32_START 0x10000
#define XCL_CHAR_LIST_LOW_32_END 0x7fffffff
#define XCL_CHAR_LIST_LOW_32_ADD 0x0

#define XCL_CHAR_LIST_HIGH_32_START 0x80000000
#define XCL_CHAR_LIST_HIGH_32_END 0xffffffff
#define XCL_CHAR_LIST_HIGH_32_ADD 0x80000000

/* Mask for getting the descriptors of character list ranges.
Each descriptor has XCL_TYPE_BIT_LEN bits, and can be processed
by XCL_BEGIN_WITH_RANGE and XCL_ITEM_COUNT_MASK macros. */
#define XCL_TYPE_MASK 0xfff
#define XCL_TYPE_BIT_LEN 3
/* If this bit is set, the first item of the character list is the
end of a range, which started before the starting character of the
character list. */
#define XCL_BEGIN_WITH_RANGE 0x4
/* Number of items in the character list: 0, 1, or 2. The value 3
represents that the item count is stored at the begining of the
character list. The item count has the same width as the items
in the character list (e.g. 16 bit for Low16 and High16 lists). */
#define XCL_ITEM_COUNT_MASK 0x3
/* Shift and flag for constructing character list items. The XCL_CHAR_END
is set, when the item is not the beginning of a range. The XCL_CHAR_SHIFT
can be used to encode / decode the character value stored in an item. */
#define XCL_CHAR_END 0x1
#define XCL_CHAR_SHIFT 1

/* Flag bits for an extended class (OP_ECLASS), which is used for complex
character matches such as [\p{Greek} && \p{Ll}]. */

#define ECL_MAP     0x01  /* Flag: a 32-byte map is present */

/* Type tags for the items stored in an extended class (OP_ECLASS). These items
follow the OP_ECLASS's flag char and bitmap, and represent a Reverse Polish
Notation list of operands and operators manipulating a stack of bits. */

#define ECL_AND     1 /* Pop two from the stack, AND, and push result. */
#define ECL_OR      2 /* Pop two from the stack, OR, and push result. */
#define ECL_XOR     3 /* Pop two from the stack, XOR, and push result. */
#define ECL_NOT     4 /* Pop one from the stack, NOT, and push result. */
#define ECL_XCLASS  5 /* XCLASS nested within ECLASS; match and push result. */
#define ECL_ANY     6 /* Temporary, only used during compilation. */
#define ECL_NONE    7 /* Temporary, only used during compilation. */

/* These are escaped items that aren't just an encoding of a particular data
value such as \n. They must have non-zero values, as check_escape() returns 0
for a data character. In the escapes[] table in pcre2_compile.c their values
are negated in order to distinguish them from data values.

They must appear here in the same order as in the opcode definitions below, up
to ESC_z. There's a dummy for OP_ALLANY because it corresponds to "." in DOTALL
mode rather than an escape sequence. It is also used for [^] in JavaScript
compatibility mode, and for \C in non-utf mode. In non-DOTALL mode, "." behaves
like \N.

ESC_ub is a special return from check_escape() when, in BSUX mode, \u{ is not
followed by hex digits and }, in which case it should mean a literal "u"
followed by a literal "{". This hack is necessary for cases like \u{ 12}
because without it, this is interpreted as u{12} now that spaces are allowed in
quantifiers.

Negative numbers are used to encode a backreference (\1, \2, \3, etc.) in
check_escape(). There are tests in the code for an escape greater than ESC_b
and less than ESC_Z to detect the types that may be repeated. These are the
types that consume characters. If any new escapes are put in between that don't
consume a character, that code will have to change. */

enum { ESC_A = 1, ESC_G, ESC_K, ESC_B, ESC_b, ESC_D, ESC_d, ESC_S, ESC_s,
       ESC_W, ESC_w, ESC_N, ESC_dum, ESC_C, ESC_P, ESC_p, ESC_R, ESC_H,
       ESC_h, ESC_V, ESC_v, ESC_X, ESC_Z, ESC_z,
       ESC_E, ESC_Q, ESC_g, ESC_k, ESC_ub };


/********************** Opcode definitions ******************/

/****** NOTE NOTE NOTE ******

Starting from 1 (i.e. after OP_END), the values up to OP_EOD must correspond in
order to the list of escapes immediately above. Furthermore, values up to
OP_DOLLM must not be changed without adjusting the table called autoposstab in
pcre2_auto_possess.c.

Whenever this list is updated, the two macro definitions that follow must be
updated to match. The possessification table called "opcode_possessify" in
pcre2_compile.c must also be updated, and also the tables called "coptable"
and "poptable" in pcre2_dfa_match.c.

****** NOTE NOTE NOTE ******/


/* The values between FIRST_AUTOTAB_OP and LAST_AUTOTAB_RIGHT_OP, inclusive,
are used in a table for deciding whether a repeated character type can be
auto-possessified. */

#define FIRST_AUTOTAB_OP       OP_NOT_DIGIT
#define LAST_AUTOTAB_LEFT_OP   OP_EXTUNI
#define LAST_AUTOTAB_RIGHT_OP  OP_DOLLM

enum {
  OP_END,            /* 0 End of pattern */

  /* Values corresponding to backslashed metacharacters */

  OP_SOD,            /* 1 Start of data: \A */
  OP_SOM,            /* 2 Start of match (subject + offset): \G */
  OP_SET_SOM,        /* 3 Set start of match (\K) */
  OP_NOT_WORD_BOUNDARY,  /*  4 \B -- see also OP_NOT_UCP_WORD_BOUNDARY */
  OP_WORD_BOUNDARY,      /*  5 \b -- see also OP_UCP_WORD_BOUNDARY */
  OP_NOT_DIGIT,          /*  6 \D */
  OP_DIGIT,              /*  7 \d */
  OP_NOT_WHITESPACE,     /*  8 \S */
  OP_WHITESPACE,         /*  9 \s */
  OP_NOT_WORDCHAR,       /* 10 \W */
  OP_WORDCHAR,           /* 11 \w */

  OP_ANY,            /* 12 Match any character except newline (\N) */
  OP_ALLANY,         /* 13 Match any character */
  OP_ANYBYTE,        /* 14 Match any byte (\C); different to OP_ANY for UTF-8 */
  OP_NOTPROP,        /* 15 \P (not Unicode property) */
  OP_PROP,           /* 16 \p (Unicode property) */
  OP_ANYNL,          /* 17 \R (any newline sequence) */
  OP_NOT_HSPACE,     /* 18 \H (not horizontal whitespace) */
  OP_HSPACE,         /* 19 \h (horizontal whitespace) */
  OP_NOT_VSPACE,     /* 20 \V (not vertical whitespace) */
  OP_VSPACE,         /* 21 \v (vertical whitespace) */
  OP_EXTUNI,         /* 22 \X (extended Unicode sequence */
  OP_EODN,           /* 23 End of data or \n at end of data (\Z) */
  OP_EOD,            /* 24 End of data (\z) */

  /* Line end assertions */

  OP_DOLL,           /* 25 End of line - not multiline */
  OP_DOLLM,          /* 26 End of line - multiline */
  OP_CIRC,           /* 27 Start of line - not multiline */
  OP_CIRCM,          /* 28 Start of line - multiline */

  /* Single characters; caseful must precede the caseless ones, and these
  must remain in this order, and adjacent. */

  OP_CHAR,           /* 29 Match one character, casefully */
  OP_CHARI,          /* 30 Match one character, caselessly */
  OP_NOT,            /* 31 Match one character, not the given one, casefully */
  OP_NOTI,           /* 32 Match one character, not the given one, caselessly */

  /* The following sets of 13 opcodes must always be kept in step because
  the offset from the first one is used to generate the others. */

  /* Repeated characters; caseful must precede the caseless ones */

  OP_STAR,           /* 33 The maximizing and minimizing versions of */
  OP_MINSTAR,        /* 34 these six opcodes must come in pairs, with */
  OP_PLUS,           /* 35 the minimizing one second. */
  OP_MINPLUS,        /* 36 */
  OP_QUERY,          /* 37 */
  OP_MINQUERY,       /* 38 */

  OP_UPTO,           /* 39 From 0 to n matches of one character, caseful*/
  OP_MINUPTO,        /* 40 */
  OP_EXACT,          /* 41 Exactly n matches */

  OP_POSSTAR,        /* 42 Possessified star, caseful */
  OP_POSPLUS,        /* 43 Possessified plus, caseful */
  OP_POSQUERY,       /* 44 Posesssified query, caseful */
  OP_POSUPTO,        /* 45 Possessified upto, caseful */

  /* Repeated characters; caseless must follow the caseful ones */

  OP_STARI,          /* 46 */
  OP_MINSTARI,       /* 47 */
  OP_PLUSI,          /* 48 */
  OP_MINPLUSI,       /* 49 */
  OP_QUERYI,         /* 50 */
  OP_MINQUERYI,      /* 51 */

  OP_UPTOI,          /* 52 From 0 to n matches of one character, caseless */
  OP_MINUPTOI,       /* 53 */
  OP_EXACTI,         /* 54 */

  OP_POSSTARI,       /* 55 Possessified star, caseless */
  OP_POSPLUSI,       /* 56 Possessified plus, caseless */
  OP_POSQUERYI,      /* 57 Posesssified query, caseless */
  OP_POSUPTOI,       /* 58 Possessified upto, caseless */

  /* The negated ones must follow the non-negated ones, and match them */
  /* Negated repeated character, caseful; must precede the caseless ones */

  OP_NOTSTAR,        /* 59 The maximizing and minimizing versions of */
  OP_NOTMINSTAR,     /* 60 these six opcodes must come in pairs, with */
  OP_NOTPLUS,        /* 61 the minimizing one second. They must be in */
  OP_NOTMINPLUS,     /* 62 exactly the same order as those above. */
  OP_NOTQUERY,       /* 63 */
  OP_NOTMINQUERY,    /* 64 */

  OP_NOTUPTO,        /* 65 From 0 to n matches, caseful */
  OP_NOTMINUPTO,     /* 66 */
  OP_NOTEXACT,       /* 67 Exactly n matches */

  OP_NOTPOSSTAR,     /* 68 Possessified versions, caseful */
  OP_NOTPOSPLUS,     /* 69 */
  OP_NOTPOSQUERY,    /* 70 */
  OP_NOTPOSUPTO,     /* 71 */

  /* Negated repeated character, caseless; must follow the caseful ones */

  OP_NOTSTARI,       /* 72 */
  OP_NOTMINSTARI,    /* 73 */
  OP_NOTPLUSI,       /* 74 */
  OP_NOTMINPLUSI,    /* 75 */
  OP_NOTQUERYI,      /* 76 */
  OP_NOTMINQUERYI,   /* 77 */

  OP_NOTUPTOI,       /* 78 From 0 to n matches, caseless */
  OP_NOTMINUPTOI,    /* 79 */
  OP_NOTEXACTI,      /* 80 Exactly n matches */

  OP_NOTPOSSTARI,    /* 81 Possessified versions, caseless */
  OP_NOTPOSPLUSI,    /* 82 */
  OP_NOTPOSQUERYI,   /* 83 */
  OP_NOTPOSUPTOI,    /* 84 */

  /* Character types */

  OP_TYPESTAR,       /* 85 The maximizing and minimizing versions of */
  OP_TYPEMINSTAR,    /* 86 these six opcodes must come in pairs, with */
  OP_TYPEPLUS,       /* 87 the minimizing one second. These codes must */
  OP_TYPEMINPLUS,    /* 88 be in exactly the same order as those above. */
  OP_TYPEQUERY,      /* 89 */
  OP_TYPEMINQUERY,   /* 90 */

  OP_TYPEUPTO,       /* 91 From 0 to n matches */
  OP_TYPEMINUPTO,    /* 92 */
  OP_TYPEEXACT,      /* 93 Exactly n matches */

  OP_TYPEPOSSTAR,    /* 94 Possessified versions */
  OP_TYPEPOSPLUS,    /* 95 */
  OP_TYPEPOSQUERY,   /* 96 */
  OP_TYPEPOSUPTO,    /* 97 */

  /* These are used for character classes and back references; only the
  first six are the same as the sets above. */

  OP_CRSTAR,         /* 98 The maximizing and minimizing versions of */
  OP_CRMINSTAR,      /* 99 all these opcodes must come in pairs, with */
  OP_CRPLUS,         /* 100 the minimizing one second. These codes must */
  OP_CRMINPLUS,      /* 101 be in exactly the same order as those above. */
  OP_CRQUERY,        /* 102 */
  OP_CRMINQUERY,     /* 103 */

  OP_CRRANGE,        /* 104 These are different to the three sets above. */
  OP_CRMINRANGE,     /* 105 */

  OP_CRPOSSTAR,      /* 106 Possessified versions */
  OP_CRPOSPLUS,      /* 107 */
  OP_CRPOSQUERY,     /* 108 */
  OP_CRPOSRANGE,     /* 109 */

  /* End of quantifier opcodes */

  OP_CLASS,          /* 110 Match a character class, chars < 256 only */
  OP_NCLASS,         /* 111 Same, but the bitmap was created from a negative
                              class - the difference is relevant only when a
                              character > 255 is encountered. */
  OP_XCLASS,         /* 112 Extended class for handling > 255 chars within the
                              class. This does both positive and negative. */
  OP_ECLASS,         /* 113 Really-extended class, for handling logical
                              expressions computed over characters. */
  OP_REF,            /* 114 Match a back reference, casefully */
  OP_REFI,           /* 115 Match a back reference, caselessly */
  OP_DNREF,          /* 116 Match a duplicate name backref, casefully */
  OP_DNREFI,         /* 117 Match a duplicate name backref, caselessly */
  OP_RECURSE,        /* 118 Match a numbered subpattern (possibly recursive) */
  OP_CALLOUT,        /* 119 Call out to external function if provided */
  OP_CALLOUT_STR,    /* 120 Call out with string argument */

  OP_ALT,            /* 121 Start of alternation */
  OP_KET,            /* 122 End of group that doesn't have an unbounded repeat */
  OP_KETRMAX,        /* 123 These two must remain together and in this */
  OP_KETRMIN,        /* 124 order. They are for groups the repeat for ever. */
  OP_KETRPOS,        /* 125 Possessive unlimited repeat. */

  /* The assertions must come before BRA, CBRA, ONCE, and COND. */

  OP_REVERSE,        /* 126 Move pointer back - used in lookbehind assertions */
  OP_VREVERSE,       /* 127 Move pointer back - variable */
  OP_ASSERT,         /* 128 Positive lookahead */
  OP_ASSERT_NOT,     /* 129 Negative lookahead */
  OP_ASSERTBACK,     /* 130 Positive lookbehind */
  OP_ASSERTBACK_NOT, /* 131 Negative lookbehind */
  OP_ASSERT_NA,      /* 132 Positive non-atomic lookahead */
  OP_ASSERTBACK_NA,  /* 133 Positive non-atomic lookbehind */
  OP_ASSERT_SCS,     /* 134 Scan substring */

  /* ONCE, SCRIPT_RUN, BRA, BRAPOS, CBRA, CBRAPOS, and COND must come
  immediately after the assertions, with ONCE first, as there's a test for >=
  ONCE for a subpattern that isn't an assertion. The POS versions must
  immediately follow the non-POS versions in each case. */

  OP_ONCE,           /* 135 Atomic group, contains captures */
  OP_SCRIPT_RUN,     /* 136 Non-capture, but check characters' scripts */
  OP_BRA,            /* 137 Start of non-capturing bracket */
  OP_BRAPOS,         /* 138 Ditto, with unlimited, possessive repeat */
  OP_CBRA,           /* 139 Start of capturing bracket */
  OP_CBRAPOS,        /* 140 Ditto, with unlimited, possessive repeat */
  OP_COND,           /* 141 Conditional group */

  /* These five must follow the previous five, in the same order. There's a
  check for >= SBRA to distinguish the two sets. */

  OP_SBRA,           /* 142 Start of non-capturing bracket, check empty  */
  OP_SBRAPOS,        /* 143 Ditto, with unlimited, possessive repeat */
  OP_SCBRA,          /* 144 Start of capturing bracket, check empty */
  OP_SCBRAPOS,       /* 145 Ditto, with unlimited, possessive repeat */
  OP_SCOND,          /* 146 Conditional group, check empty */

  /* The next two pairs must (respectively) be kept together. */

  OP_CREF,           /* 147 Used to hold a capture number as condition */
  OP_DNCREF,         /* 148 Used to point to duplicate names as a condition */
  OP_RREF,           /* 149 Used to hold a recursion number as condition */
  OP_DNRREF,         /* 150 Used to point to duplicate names as a condition */
  OP_FALSE,          /* 151 Always false (used by DEFINE and VERSION) */
  OP_TRUE,           /* 152 Always true (used by VERSION) */

  OP_BRAZERO,        /* 153 These two must remain together and in this */
  OP_BRAMINZERO,     /* 154 order. */
  OP_BRAPOSZERO,     /* 155 */

  /* These are backtracking control verbs */

  OP_MARK,           /* 156 always has an argument */
  OP_PRUNE,          /* 157 */
  OP_PRUNE_ARG,      /* 158 same, but with argument */
  OP_SKIP,           /* 159 */
  OP_SKIP_ARG,       /* 160 same, but with argument */
  OP_THEN,           /* 161 */
  OP_THEN_ARG,       /* 162 same, but with argument */
  OP_COMMIT,         /* 163 */
  OP_COMMIT_ARG,     /* 164 same, but with argument */

  /* These are forced failure and success verbs. FAIL and ACCEPT do accept an
  argument, but these cases can be compiled as, for example, (*MARK:X)(*FAIL)
  without the need for a special opcode. */

  OP_FAIL,           /* 165 */
  OP_ACCEPT,         /* 166 */
  OP_ASSERT_ACCEPT,  /* 167 Used inside assertions */
  OP_CLOSE,          /* 168 Used before OP_ACCEPT to close open captures */

  /* This is used to skip a subpattern with a {0} quantifier */

  OP_SKIPZERO,       /* 169 */

  /* This is used to identify a DEFINE group during compilation so that it can
  be checked for having only one branch. It is changed to OP_FALSE before
  compilation finishes. */

  OP_DEFINE,         /* 170 */

  /* These opcodes replace their normal counterparts in UCP mode when
  PCRE2_EXTRA_ASCII_BSW is not set. */

  OP_NOT_UCP_WORD_BOUNDARY, /* 171 */
  OP_UCP_WORD_BOUNDARY,     /* 172 */

  /* This is not an opcode, but is used to check that tables indexed by opcode
  are the correct length, in order to catch updating errors - there have been
  some in the past. */

  OP_TABLE_LENGTH

};

/* *** NOTE NOTE NOTE *** Whenever the list above is updated, the two macro
definitions that follow must also be updated to match. There are also tables
called "opcode_possessify" in pcre2_compile.c and "coptable" and "poptable" in
pcre2_dfa_match.c that must be updated. */


/* This macro defines textual names for all the opcodes. These are used only
for debugging, and some of them are only partial names. The macro is referenced
only in pcre2_printint_inc.h, which fills out the full names in many cases (and in
some cases doesn't actually use these names at all). */

#define OP_NAME_LIST \
  "End", "\\A", "\\G", "\\K", "\\B", "\\b", "\\D", "\\d",         \
  "\\S", "\\s", "\\W", "\\w", "Any", "AllAny", "Anybyte",         \
  "notprop", "prop", "\\R", "\\H", "\\h", "\\V", "\\v",           \
  "extuni",  "\\Z", "\\z",                                        \
  "$", "$", "^", "^", "char", "chari", "not", "noti",             \
  "*", "*?", "+", "+?", "?", "??",                                \
  "{", "{", "{",                                                  \
  "*+","++", "?+", "{",                                           \
  "*", "*?", "+", "+?", "?", "??",                                \
  "{", "{", "{",                                                  \
  "*+","++", "?+", "{",                                           \
  "*", "*?", "+", "+?", "?", "??",                                \
  "{", "{", "{",                                                  \
  "*+","++", "?+", "{",                                           \
  "*", "*?", "+", "+?", "?", "??",                                \
  "{", "{", "{",                                                  \
  "*+","++", "?+", "{",                                           \
  "*", "*?", "+", "+?", "?", "??", "{", "{", "{",                 \
  "*+","++", "?+", "{",                                           \
  "*", "*?", "+", "+?", "?", "??", "{", "{",                      \
  "*+","++", "?+", "{",                                           \
  "class", "nclass", "xclass", "eclass",                          \
  "Ref", "Refi", "DnRef", "DnRefi",                               \
  "Recurse", "Callout", "CalloutStr",                             \
  "Alt", "Ket", "KetRmax", "KetRmin", "KetRpos",                  \
  "Reverse", "VReverse", "Assert", "Assert not",                  \
  "Assert back", "Assert back not",                               \
  "Non-atomic assert", "Non-atomic assert back",                  \
  "Scan substring",                                               \
  "Once",                                                         \
  "Script run",                                                   \
  "Bra", "BraPos", "CBra", "CBraPos",                             \
  "Cond",                                                         \
  "SBra", "SBraPos", "SCBra", "SCBraPos",                         \
  "SCond",                                                        \
  "Capture ref", "Capture dnref", "Cond rec", "Cond dnrec",       \
  "Cond false", "Cond true",                                      \
  "Brazero", "Braminzero", "Braposzero",                          \
  "*MARK", "*PRUNE", "*PRUNE", "*SKIP", "*SKIP",                  \
  "*THEN", "*THEN", "*COMMIT", "*COMMIT", "*FAIL",                \
  "*ACCEPT", "*ASSERT_ACCEPT",                                    \
  "Close", "Skip zero", "Define", "\\B (ucp)", "\\b (ucp)"


/* This macro defines the length of fixed length operations in the compiled
regex. The lengths are used when searching for specific things, and also in the
debugging printing of a compiled regex. We use a macro so that it can be
defined close to the definitions of the opcodes themselves.

As things have been extended, some of these are no longer fixed lenths, but are
minima instead. For example, the length of a single-character repeat may vary
in UTF-8 mode. The code that uses this table must know about such things. */

#define OP_LENGTHS \
  1,                             /* End                                    */ \
  1, 1, 1, 1, 1,                 /* \A, \G, \K, \B, \b                     */ \
  1, 1, 1, 1, 1, 1,              /* \D, \d, \S, \s, \W, \w                 */ \
  1, 1, 1,                       /* Any, AllAny, Anybyte                   */ \
  3, 3,                          /* \P, \p                                 */ \
  1, 1, 1, 1, 1,                 /* \R, \H, \h, \V, \v                     */ \
  1,                             /* \X                                     */ \
  1, 1, 1, 1, 1, 1,              /* \Z, \z, $, $M ^, ^M                    */ \
  2,                             /* Char  - the minimum length             */ \
  2,                             /* Chari  - the minimum length            */ \
  2,                             /* not                                    */ \
  2,                             /* noti                                   */ \
  /* Positive single-char repeats                             ** These are */ \
  2, 2, 2, 2, 2, 2,              /* *, *?, +, +?, ?, ??       ** minima in */ \
  2+IMM2_SIZE, 2+IMM2_SIZE,      /* upto, minupto             ** mode      */ \
  2+IMM2_SIZE,                   /* exact                                  */ \
  2, 2, 2, 2+IMM2_SIZE,          /* *+, ++, ?+, upto+                      */ \
  2, 2, 2, 2, 2, 2,              /* *I, *?I, +I, +?I, ?I, ??I ** UTF-8     */ \
  2+IMM2_SIZE, 2+IMM2_SIZE,      /* upto I, minupto I                      */ \
  2+IMM2_SIZE,                   /* exact I                                */ \
  2, 2, 2, 2+IMM2_SIZE,          /* *+I, ++I, ?+I, upto+I                  */ \
  /* Negative single-char repeats - only for chars < 256                   */ \
  2, 2, 2, 2, 2, 2,              /* NOT *, *?, +, +?, ?, ??                */ \
  2+IMM2_SIZE, 2+IMM2_SIZE,      /* NOT upto, minupto                      */ \
  2+IMM2_SIZE,                   /* NOT exact                              */ \
  2, 2, 2, 2+IMM2_SIZE,          /* Possessive NOT *, +, ?, upto           */ \
  2, 2, 2, 2, 2, 2,              /* NOT *I, *?I, +I, +?I, ?I, ??I          */ \
  2+IMM2_SIZE, 2+IMM2_SIZE,      /* NOT upto I, minupto I                  */ \
  2+IMM2_SIZE,                   /* NOT exact I                            */ \
  2, 2, 2, 2+IMM2_SIZE,          /* Possessive NOT *I, +I, ?I, upto I      */ \
  /* Positive type repeats                                                 */ \
  2, 2, 2, 2, 2, 2,              /* Type *, *?, +, +?, ?, ??               */ \
  2+IMM2_SIZE, 2+IMM2_SIZE,      /* Type upto, minupto                     */ \
  2+IMM2_SIZE,                   /* Type exact                             */ \
  2, 2, 2, 2+IMM2_SIZE,          /* Possessive *+, ++, ?+, upto+           */ \
  /* Character class & ref repeats                                         */ \
  1, 1, 1, 1, 1, 1,              /* *, *?, +, +?, ?, ??                    */ \
  1+2*IMM2_SIZE, 1+2*IMM2_SIZE,  /* CRRANGE, CRMINRANGE                    */ \
  1, 1, 1, 1+2*IMM2_SIZE,        /* Possessive *+, ++, ?+, CRPOSRANGE      */ \
  1+(32/sizeof(PCRE2_UCHAR)),    /* CLASS                                  */ \
  1+(32/sizeof(PCRE2_UCHAR)),    /* NCLASS                                 */ \
  0,                             /* XCLASS - variable length               */ \
  0,                             /* ECLASS - variable length               */ \
  1+IMM2_SIZE,                   /* REF                                    */ \
  1+IMM2_SIZE+1,                 /* REFI                                   */ \
  1+2*IMM2_SIZE,                 /* DNREF                                  */ \
  1+2*IMM2_SIZE+1,               /* DNREFI                                 */ \
  1+LINK_SIZE,                   /* RECURSE                                */ \
  1+2*LINK_SIZE+1,               /* CALLOUT                                */ \
  0,                             /* CALLOUT_STR - variable length          */ \
  1+LINK_SIZE,                   /* Alt                                    */ \
  1+LINK_SIZE,                   /* Ket                                    */ \
  1+LINK_SIZE,                   /* KetRmax                                */ \
  1+LINK_SIZE,                   /* KetRmin                                */ \
  1+LINK_SIZE,                   /* KetRpos                                */ \
  1+IMM2_SIZE,                   /* Reverse                                */ \
  1+2*IMM2_SIZE,                 /* VReverse                               */ \
  1+LINK_SIZE,                   /* Assert                                 */ \
  1+LINK_SIZE,                   /* Assert not                             */ \
  1+LINK_SIZE,                   /* Assert behind                          */ \
  1+LINK_SIZE,                   /* Assert behind not                      */ \
  1+LINK_SIZE,                   /* NA Assert                              */ \
  1+LINK_SIZE,                   /* NA Assert behind                       */ \
  1+LINK_SIZE,                   /* Scan substring                         */ \
  1+LINK_SIZE,                   /* ONCE                                   */ \
  1+LINK_SIZE,                   /* SCRIPT_RUN                             */ \
  1+LINK_SIZE,                   /* BRA                                    */ \
  1+LINK_SIZE,                   /* BRAPOS                                 */ \
  1+LINK_SIZE+IMM2_SIZE,         /* CBRA                                   */ \
  1+LINK_SIZE+IMM2_SIZE,         /* CBRAPOS                                */ \
  1+LINK_SIZE,                   /* COND                                   */ \
  1+LINK_SIZE,                   /* SBRA                                   */ \
  1+LINK_SIZE,                   /* SBRAPOS                                */ \
  1+LINK_SIZE+IMM2_SIZE,         /* SCBRA                                  */ \
  1+LINK_SIZE+IMM2_SIZE,         /* SCBRAPOS                               */ \
  1+LINK_SIZE,                   /* SCOND                                  */ \
  1+IMM2_SIZE, 1+2*IMM2_SIZE,    /* CREF, DNCREF                           */ \
  1+IMM2_SIZE, 1+2*IMM2_SIZE,    /* RREF, DNRREF                           */ \
  1, 1,                          /* FALSE, TRUE                            */ \
  1, 1, 1,                       /* BRAZERO, BRAMINZERO, BRAPOSZERO        */ \
  3, 1, 3,                       /* MARK, PRUNE, PRUNE_ARG                 */ \
  1, 3,                          /* SKIP, SKIP_ARG                         */ \
  1, 3,                          /* THEN, THEN_ARG                         */ \
  1, 3,                          /* COMMIT, COMMIT_ARG                     */ \
  1, 1, 1,                       /* FAIL, ACCEPT, ASSERT_ACCEPT            */ \
  1+IMM2_SIZE, 1,                /* CLOSE, SKIPZERO                        */ \
  1,                             /* DEFINE                                 */ \
  1, 1                           /* \B and \b in UCP mode                  */

/* A magic value for OP_RREF to indicate the "any recursion" condition. */

#define RREF_ANY  0xffff

/* Constants used by OP_REFI and OP_DNREFI to control matching behaviour. */

#define REFI_FLAG_CASELESS_RESTRICT  0x1
#define REFI_FLAG_TURKISH_CASING     0x2


/* ---------- Private structures that are mode-independent. ---------- */

/* Structure to hold data for custom memory management. */

typedef struct pcre2_memctl {
  void *    (*malloc)(size_t, void *);
  void      (*free)(void *, void *);
  void      *memory_data;
} pcre2_memctl;

/* Structure for building a chain of open capturing subpatterns during
compiling, so that instructions to close them can be compiled when (*ACCEPT) is
encountered. */

typedef struct open_capitem {
  struct open_capitem *next;    /* Chain link */
  uint16_t number;              /* Capture number */
  uint16_t assert_depth;        /* Assertion depth when opened */
} open_capitem;

/* Layout of the UCP type table that translates property names into types and
codes. Each entry used to point directly to a name, but to reduce the number of
relocations in shared libraries, it now has an offset into a single string
instead. */

typedef struct {
  uint16_t name_offset;
  uint16_t type;
  uint16_t value;
} ucp_type_table;

/* Unicode character database (UCD) record format */

typedef struct {
  uint8_t script;     /* ucp_Arabic, etc. */
  uint8_t chartype;   /* ucp_Cc, etc. (general categories) */
  uint8_t gbprop;     /* ucp_gbControl, etc. (grapheme break property) */
  uint8_t caseset;    /* offset to multichar other cases or zero */
  int32_t other_case; /* offset to other case, or zero if none */
  uint16_t scriptx_bidiclass; /* script extension (11 bit) and bidi class (5 bit) values */
  uint16_t bprops;    /* binary properties offset */
} ucd_record;

/* UCD access macros */

#define UCD_BLOCK_SIZE 128
#define REAL_GET_UCD(ch) (PRIV(ucd_records) + \
        PRIV(ucd_stage2)[PRIV(ucd_stage1)[(int)(ch) / UCD_BLOCK_SIZE] * \
        UCD_BLOCK_SIZE + (int)(ch) % UCD_BLOCK_SIZE])

#if PCRE2_CODE_UNIT_WIDTH == 32
#define GET_UCD(ch) ((ch > MAX_UTF_CODE_POINT)? \
  PRIV(dummy_ucd_record) : REAL_GET_UCD(ch))
#else
#define GET_UCD(ch) REAL_GET_UCD(ch)
#endif

#define UCD_SCRIPTX_MASK 0x3ff
#define UCD_BIDICLASS_SHIFT 11
#define UCD_BPROPS_MASK 0xfff

#define UCD_SCRIPTX_PROP(prop) ((prop)->scriptx_bidiclass & UCD_SCRIPTX_MASK)
#define UCD_BIDICLASS_PROP(prop) ((prop)->scriptx_bidiclass >> UCD_BIDICLASS_SHIFT)
#define UCD_BPROPS_PROP(prop) ((prop)->bprops & UCD_BPROPS_MASK)

#define UCD_CHARTYPE(ch)    GET_UCD(ch)->chartype
#define UCD_SCRIPT(ch)      GET_UCD(ch)->script
#define UCD_CATEGORY(ch)    PRIV(ucp_gentype)[UCD_CHARTYPE(ch)]
#define UCD_GRAPHBREAK(ch)  GET_UCD(ch)->gbprop
#define UCD_CASESET(ch)     GET_UCD(ch)->caseset
#define UCD_OTHERCASE(ch)   ((uint32_t)((int)ch + (int)(GET_UCD(ch)->other_case)))
#define UCD_SCRIPTX(ch)     UCD_SCRIPTX_PROP(GET_UCD(ch))
#define UCD_BPROPS(ch)      UCD_BPROPS_PROP(GET_UCD(ch))
#define UCD_BIDICLASS(ch)   UCD_BIDICLASS_PROP(GET_UCD(ch))
#define UCD_ANY_I(ch) \
  /* match any of the four characters 'i', 'I', U+0130, U+0131 */ \
  (((uint32_t)(ch) | 0x20u) == 0x69u || ((uint32_t)(ch) | 1u) == 0x0131u)
#define UCD_DOTTED_I(ch) \
  ((uint32_t)(ch) == 0x69u || (uint32_t)(ch) == 0x0130u)
#define UCD_FOLD_I_TURKISH(ch) \
  ((uint32_t)(ch) == 0x0130u ?   0x69u : \
   (uint32_t)(ch) ==   0x49u ? 0x0131u : (uint32_t)(ch))

/* The "scriptx" and bprops fields contain offsets into vectors of 32-bit words
that form a bitmap representing a list of scripts or boolean properties. These
macros test or set a bit in the map by number. */

#define MAPBIT(map,n) ((map)[(n)/32]&(1u<<((n)%32)))
#define MAPSET(map,n) ((map)[(n)/32]|=(1u<<((n)%32)))

/* Header for serialized pcre2 codes. */

typedef struct pcre2_serialized_data {
  uint32_t magic;
  uint32_t version;
  uint32_t config;
  int32_t  number_of_codes;
} pcre2_serialized_data;



/* ----------------- Items that need PCRE2_CODE_UNIT_WIDTH ----------------- */

/* When this file is included by pcre2test, PCRE2_CODE_UNIT_WIDTH is defined as
0, so the following items are omitted. */

#if defined PCRE2_CODE_UNIT_WIDTH && PCRE2_CODE_UNIT_WIDTH != 0

/* EBCDIC is supported only for the 8-bit library. */

#if defined EBCDIC && PCRE2_CODE_UNIT_WIDTH != 8
#error EBCDIC is not supported for the 16-bit or 32-bit libraries
#endif

/* This is the largest non-UTF code point. */

#define MAX_NON_UTF_CHAR (0xffffffffU >> (32 - PCRE2_CODE_UNIT_WIDTH))

/* Internal shared data tables and variables. These are used by more than one
of the exported public functions. They have to be "external" in the C sense,
but are not part of the PCRE2 public API. Although the data for some of them is
identical in all libraries, they must have different names so that multiple
libraries can be simultaneously linked to a single application. However, UTF-8
tables are needed only when compiling the 8-bit library. */

#if PCRE2_CODE_UNIT_WIDTH == 8
extern const int              PRIV(utf8_table1)[];
extern const unsigned         PRIV(utf8_table1_size);
extern const int              PRIV(utf8_table2)[];
extern const int              PRIV(utf8_table3)[];
extern const uint8_t          PRIV(utf8_table4)[];
#endif

#define _pcre2_OP_lengths              PCRE2_SUFFIX(_pcre2_OP_lengths_)
#define _pcre2_callout_end_delims      PCRE2_SUFFIX(_pcre2_callout_end_delims_)
#define _pcre2_callout_start_delims    PCRE2_SUFFIX(_pcre2_callout_start_delims_)
#define _pcre2_default_compile_context PCRE2_SUFFIX(_pcre2_default_compile_context_)
#define _pcre2_default_convert_context PCRE2_SUFFIX(_pcre2_default_convert_context_)
#define _pcre2_default_match_context   PCRE2_SUFFIX(_pcre2_default_match_context_)
#define _pcre2_default_tables          PCRE2_SUFFIX(_pcre2_default_tables_)
#if PCRE2_CODE_UNIT_WIDTH == 32
#define _pcre2_dummy_ucd_record        PCRE2_SUFFIX(_pcre2_dummy_ucd_record_)
#endif
#define _pcre2_hspace_list             PCRE2_SUFFIX(_pcre2_hspace_list_)
#define _pcre2_vspace_list             PCRE2_SUFFIX(_pcre2_vspace_list_)
#define _pcre2_ucd_boolprop_sets       PCRE2_SUFFIX(_pcre2_ucd_boolprop_sets_)
#define _pcre2_ucd_caseless_sets       PCRE2_SUFFIX(_pcre2_ucd_caseless_sets_)
#define _pcre2_ucd_turkish_dotted_i_caseset  PCRE2_SUFFIX(_pcre2_ucd_turkish_dotted_i_caseset_)
#define _pcre2_ucd_nocase_ranges       PCRE2_SUFFIX(_pcre2_ucd_nocase_ranges_)
#define _pcre2_ucd_nocase_ranges_size  PCRE2_SUFFIX(_pcre2_ucd_nocase_ranges_size_)
#define _pcre2_ucd_digit_sets          PCRE2_SUFFIX(_pcre2_ucd_digit_sets_)
#define _pcre2_ucd_script_sets         PCRE2_SUFFIX(_pcre2_ucd_script_sets_)
#define _pcre2_ucd_records             PCRE2_SUFFIX(_pcre2_ucd_records_)
#define _pcre2_ucd_stage1              PCRE2_SUFFIX(_pcre2_ucd_stage1_)
#define _pcre2_ucd_stage2              PCRE2_SUFFIX(_pcre2_ucd_stage2_)
#define _pcre2_ucp_gbtable             PCRE2_SUFFIX(_pcre2_ucp_gbtable_)
#define _pcre2_ucp_gentype             PCRE2_SUFFIX(_pcre2_ucp_gentype_)
#define _pcre2_ucp_typerange           PCRE2_SUFFIX(_pcre2_ucp_typerange_)
#define _pcre2_unicode_version         PCRE2_SUFFIX(_pcre2_unicode_version_)
#define _pcre2_utt                     PCRE2_SUFFIX(_pcre2_utt_)
#define _pcre2_utt_names               PCRE2_SUFFIX(_pcre2_utt_names_)
#define _pcre2_utt_size                PCRE2_SUFFIX(_pcre2_utt_size_)
#define _pcre2_ebcdic_1047_to_ascii    PCRE2_SUFFIX(_pcre2_ebcdic_1047_to_ascii_)
#define _pcre2_ascii_to_ebcdic_1047    PCRE2_SUFFIX(_pcre2_ascii_to_ebcdic_1047_)

extern const uint8_t                   PRIV(OP_lengths)[];
extern const uint32_t                  PRIV(callout_end_delims)[];
extern const uint32_t                  PRIV(callout_start_delims)[];
extern pcre2_compile_context           PRIV(default_compile_context);
extern pcre2_convert_context           PRIV(default_convert_context);
extern pcre2_match_context             PRIV(default_match_context);
extern const uint8_t                   PRIV(default_tables)[];
extern const uint32_t                  PRIV(hspace_list)[];
extern const uint32_t                  PRIV(vspace_list)[];
extern const uint32_t                  PRIV(ucd_boolprop_sets)[];
extern const uint32_t                  PRIV(ucd_caseless_sets)[];
extern const uint32_t                  PRIV(ucd_turkish_dotted_i_caseset);
extern const uint32_t                  PRIV(ucd_nocase_ranges)[];
extern const uint32_t                  PRIV(ucd_nocase_ranges_size);
extern const uint32_t                  PRIV(ucd_digit_sets)[];
extern const uint32_t                  PRIV(ucd_script_sets)[];
extern const ucd_record                PRIV(ucd_records)[];
#if PCRE2_CODE_UNIT_WIDTH == 32
extern const ucd_record                PRIV(dummy_ucd_record)[];
#endif
extern const uint16_t                  PRIV(ucd_stage1)[];
extern const uint16_t                  PRIV(ucd_stage2)[];
extern const uint32_t                  PRIV(ucp_gbtable)[];
extern const uint32_t                  PRIV(ucp_gentype)[];
#ifdef SUPPORT_JIT
extern const int                       PRIV(ucp_typerange)[];
#endif
extern const char                     *PRIV(unicode_version);
extern const ucp_type_table            PRIV(utt)[];
extern const char                      PRIV(utt_names)[];
extern const size_t                    PRIV(utt_size);
extern const uint8_t                   PRIV(ebcdic_1047_to_ascii)[];
extern const uint8_t                   PRIV(ascii_to_ebcdic_1047)[];

/* Mode-dependent macros and hidden and private structures are defined in a
separate file so that pcre2test can include them at all supported widths. When
compiling the library, PCRE2_CODE_UNIT_WIDTH will be defined, and we can
include them at the appropriate width, after setting up suffix macros for the
private structures. */

#define branch_chain                 PCRE2_SUFFIX(branch_chain_)
#define compile_block                PCRE2_SUFFIX(compile_block_)
#define dfa_match_block              PCRE2_SUFFIX(dfa_match_block_)
#define match_block                  PCRE2_SUFFIX(match_block_)
#define named_group                  PCRE2_SUFFIX(named_group_)

#include "pcre2_intmodedep.h"

/* Private "external" functions. These are internal functions that are called
from modules other than the one in which they are defined. They have to be
"external" in the C sense, but are not part of the PCRE2 public API. They are
not referenced from pcre2test, and must not be defined when no code unit width
is available. */

#define _pcre2_auto_possessify       PCRE2_SUFFIX(_pcre2_auto_possessify_)
#define _pcre2_check_escape          PCRE2_SUFFIX(_pcre2_check_escape_)
#define _pcre2_ckd_smul              PCRE2_SUFFIX(_pcre2_ckd_smul_)
#define _pcre2_extuni                PCRE2_SUFFIX(_pcre2_extuni_)
#define _pcre2_find_bracket          PCRE2_SUFFIX(_pcre2_find_bracket_)
#define _pcre2_is_newline            PCRE2_SUFFIX(_pcre2_is_newline_)
#define _pcre2_jit_free_rodata       PCRE2_SUFFIX(_pcre2_jit_free_rodata_)
#define _pcre2_jit_free              PCRE2_SUFFIX(_pcre2_jit_free_)
#define _pcre2_jit_get_size          PCRE2_SUFFIX(_pcre2_jit_get_size_)
#define _pcre2_jit_get_target        PCRE2_SUFFIX(_pcre2_jit_get_target_)
#define _pcre2_memctl_malloc         PCRE2_SUFFIX(_pcre2_memctl_malloc_)
#define _pcre2_ord2utf               PCRE2_SUFFIX(_pcre2_ord2utf_)
#define _pcre2_script_run            PCRE2_SUFFIX(_pcre2_script_run_)
#define _pcre2_strcmp                PCRE2_SUFFIX(_pcre2_strcmp_)
#define _pcre2_strcmp_c8             PCRE2_SUFFIX(_pcre2_strcmp_c8_)
#define _pcre2_strcpy_c8             PCRE2_SUFFIX(_pcre2_strcpy_c8_)
#define _pcre2_strlen                PCRE2_SUFFIX(_pcre2_strlen_)
#define _pcre2_strncmp               PCRE2_SUFFIX(_pcre2_strncmp_)
#define _pcre2_strncmp_c8            PCRE2_SUFFIX(_pcre2_strncmp_c8_)
#define _pcre2_study                 PCRE2_SUFFIX(_pcre2_study_)
#define _pcre2_valid_utf             PCRE2_SUFFIX(_pcre2_valid_utf_)
#define _pcre2_was_newline           PCRE2_SUFFIX(_pcre2_was_newline_)
#define _pcre2_xclass                PCRE2_SUFFIX(_pcre2_xclass_)
#define _pcre2_eclass                PCRE2_SUFFIX(_pcre2_eclass_)

extern int          _pcre2_auto_possessify(PCRE2_UCHAR *,
                      const compile_block *);
extern int          _pcre2_check_escape(PCRE2_SPTR *, PCRE2_SPTR, uint32_t *,
                      int *, uint32_t, uint32_t, uint32_t, BOOL, compile_block *);
extern BOOL         _pcre2_ckd_smul(PCRE2_SIZE *, int, int);
extern PCRE2_SPTR   _pcre2_extuni(uint32_t, PCRE2_SPTR, PCRE2_SPTR, PCRE2_SPTR,
                      BOOL, int *);
extern PCRE2_SPTR   _pcre2_find_bracket(PCRE2_SPTR, BOOL, int);
extern BOOL         _pcre2_is_newline(PCRE2_SPTR, uint32_t, PCRE2_SPTR,
                      uint32_t *, BOOL);
extern void         _pcre2_jit_free_rodata(void *, void *);
extern void         _pcre2_jit_free(void *, pcre2_memctl *);
extern size_t       _pcre2_jit_get_size(void *);
const char *        _pcre2_jit_get_target(void);
extern void *       _pcre2_memctl_malloc(size_t, pcre2_memctl *);
extern unsigned int _pcre2_ord2utf(uint32_t, PCRE2_UCHAR *);
extern BOOL         _pcre2_script_run(PCRE2_SPTR, PCRE2_SPTR, BOOL);
extern int          _pcre2_strcmp(PCRE2_SPTR, PCRE2_SPTR);
extern int          _pcre2_strcmp_c8(PCRE2_SPTR, const char *);
extern PCRE2_SIZE   _pcre2_strcpy_c8(PCRE2_UCHAR *, const char *);
extern PCRE2_SIZE   _pcre2_strlen(PCRE2_SPTR);
extern int          _pcre2_strncmp(PCRE2_SPTR, PCRE2_SPTR, size_t);
extern int          _pcre2_strncmp_c8(PCRE2_SPTR, const char *, size_t);
extern int          _pcre2_study(pcre2_real_code *);
extern int          _pcre2_valid_utf(PCRE2_SPTR, PCRE2_SIZE, PCRE2_SIZE *);
extern BOOL         _pcre2_was_newline(PCRE2_SPTR, uint32_t, PCRE2_SPTR,
                      uint32_t *, BOOL);
extern BOOL         _pcre2_xclass(uint32_t, PCRE2_SPTR, const uint8_t *, BOOL);
extern BOOL         _pcre2_eclass(uint32_t, PCRE2_SPTR, PCRE2_SPTR,
                      const uint8_t *, BOOL);

#endif  /* PCRE2_CODE_UNIT_WIDTH */

#include "pcre2_util.h"

#endif  /* PCRE2_INTERNAL_H_IDEMPOTENT_GUARD */

/* End of pcre2_internal.h */
