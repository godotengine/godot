/*************************************************
*      Perl-Compatible Regular Expressions       *
*************************************************/

/* PCRE2 is a library of functions to support regular expressions whose syntax
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

#ifndef PCRE2_INTERNAL_H_IDEMPOTENT_GUARD
#define PCRE2_INTERNAL_H_IDEMPOTENT_GUARD

/* We do not support both EBCDIC and Unicode at the same time. The "configure"
script prevents both being selected, but not everybody uses "configure". EBCDIC
is only supported for the 8-bit library, but the check for this has to be later
in this file, because the first part is not width-dependent, and is included by
pcre2test.c with CODE_UNIT_WIDTH == 0. */

#if defined EBCDIC && defined SUPPORT_UNICODE
#error The use of both EBCDIC and SUPPORT_UNICODE is not supported.
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

/* Valgrind (memcheck) support */

#ifdef SUPPORT_VALGRIND
#include <valgrind/memcheck.h>
#endif

/* Older versions of MSVC lack snprintf(). This define allows for
warning/error-free compilation and testing with MSVC compilers back to at least
MSVC 10/2010. Except for VC6 (which is missing some fundamentals and fails). */

#if defined(_MSC_VER) && (_MSC_VER < 1900)
#define snprintf _snprintf
#endif

/* When compiling a DLL for Windows, the exported symbols have to be declared
using some MS magic. I found some useful information on this web page:
http://msdn2.microsoft.com/en-us/library/y4h7bcy6(VS.80).aspx. According to the
information there, using __declspec(dllexport) without "extern" we have a
definition; with "extern" we have a declaration. The settings here override the
setting in pcre2.h (which is included below); it defines only PCRE2_EXP_DECL,
which is all that is needed for applications (they just import the symbols). We
use:

  PCRE2_EXP_DECL    for declarations
  PCRE2_EXP_DEFN    for definitions

The reason for wrapping this in #ifndef PCRE2_EXP_DECL is so that pcre2test,
which is an application, but needs to import this file in order to "peek" at
internals, can #include pcre2.h first to get an application's-eye view.

In principle, people compiling for non-Windows, non-Unix-like (i.e. uncommon,
special-purpose environments) might want to stick other stuff in front of
exported symbols. That's why, in the non-Windows case, we set PCRE2_EXP_DEFN
only if it is not already set. */

#ifndef PCRE2_EXP_DECL
#  ifdef _WIN32
#    ifndef PCRE2_STATIC
#      define PCRE2_EXP_DECL       extern __declspec(dllexport)
#      define PCRE2_EXP_DEFN       __declspec(dllexport)
#    else
#      define PCRE2_EXP_DECL       extern
#      define PCRE2_EXP_DEFN
#    endif
#  else
#    ifdef __cplusplus
#      define PCRE2_EXP_DECL       extern "C"
#    else
#      define PCRE2_EXP_DECL       extern
#    endif
#    ifndef PCRE2_EXP_DEFN
#      define PCRE2_EXP_DEFN       PCRE2_EXP_DECL
#    endif
#  endif
#endif

/* Include the public PCRE2 header and the definitions of UCP character
property values. This must follow the setting of PCRE2_EXP_DECL above. */

#include "pcre2.h"
#include "pcre2_ucp.h"

/* When PCRE2 is compiled as a C++ library, the subject pointer can be replaced
with a custom type. This makes it possible, for example, to allow pcre2_match()
to process subject strings that are discontinuous by using a smart pointer
class. It must always be possible to inspect all of the subject string in
pcre2_match() because of the way it backtracks. */

/* WARNING: This is as yet untested for PCRE2. */

#ifdef CUSTOM_SUBJECT_PTR
#undef PCRE2_SPTR
#define PCRE2_SPTR CUSTOM_SUBJECT_PTR
#endif

/* When checking for integer overflow in pcre2_compile(), we need to handle
large integers. If a 64-bit integer type is available, we can use that.
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

/* When compiling for use with the Virtual Pascal compiler, these functions
need to have their names changed. PCRE2 must be compiled with the -DVPCOMPAT
option on the command line. */

#ifdef VPCOMPAT
#define strlen(s)        _strlen(s)
#define strncmp(s1,s2,m) _strncmp(s1,s2,m)
#define memcmp(s,c,n)    _memcmp(s,c,n)
#define memcpy(d,s,n)    _memcpy(d,s,n)
#define memmove(d,s,n)   _memmove(d,s,n)
#define memset(s,c,n)    _memset(s,c,n)
#else  /* VPCOMPAT */

/* Otherwise, to cope with SunOS4 and other systems that lack memmove(), define
a macro that calls an emulating function. */

#ifndef HAVE_MEMMOVE
#undef  memmove          /* Some systems may have a macro */
#define memmove(a, b, c) PRIV(memmove)(a, b, c)
#endif   /* not HAVE_MEMMOVE */
#endif   /* not VPCOMPAT */

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

/* The initial frames vector for remembering backtracking points in
pcre2_match() is allocated on the system stack, of this size (bytes). The size
must be a multiple of sizeof(PCRE2_SPTR) in all environments, so making it a
multiple of 8 is best. Typical frame sizes are a few hundred bytes (it depends
on the number of capturing parentheses) so 20KiB handles quite a few frames. A
larger vector on the heap is obtained for patterns that need more frames. The
maximum size of this can be limited. */

#define START_FRAMES_SIZE 20480

/* Similarly, for DFA matching, an initial internal workspace vector is
allocated on the stack. */

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
  0x2006, 0x2007, 0x2008, 0x2009, 0x200A, 0x202f, 0x205f, 0x3000, \
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
  case 0x200A:  /* HAIR SPACE */ \
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

#define PCRE2_MODE8         0x00000001  /* compiled in 8 bit mode */
#define PCRE2_MODE16        0x00000002  /* compiled in 16 bit mode */
#define PCRE2_MODE32        0x00000004  /* compiled in 32 bit mode */
#define PCRE2_FIRSTSET      0x00000010  /* first_code unit is set */
#define PCRE2_FIRSTCASELESS 0x00000020  /* caseless first code unit */
#define PCRE2_FIRSTMAPSET   0x00000040  /* bitmap of first code units is set */
#define PCRE2_LASTSET       0x00000080  /* last code unit is set */
#define PCRE2_LASTCASELESS  0x00000100  /* caseless last code unit */
#define PCRE2_STARTLINE     0x00000200  /* start after \n for multiline */
#define PCRE2_JCHANGED      0x00000400  /* j option used in pattern */
#define PCRE2_HASCRORLF     0x00000800  /* explicit \r or \n in pattern */
#define PCRE2_HASTHEN       0x00001000  /* pattern contains (*THEN) */
#define PCRE2_MATCH_EMPTY   0x00002000  /* pattern can match empty string */
#define PCRE2_BSR_SET       0x00004000  /* BSR was set in the pattern */
#define PCRE2_NL_SET        0x00008000  /* newline was set in the pattern */
#define PCRE2_NOTEMPTY_SET  0x00010000  /* (*NOTEMPTY) used        ) keep */
#define PCRE2_NE_ATST_SET   0x00020000  /* (*NOTEMPTY_ATSTART) used) together */
#define PCRE2_DEREF_TABLES  0x00040000  /* release character tables */
#define PCRE2_NOJIT         0x00080000  /* (*NOJIT) used */
#define PCRE2_HASBKPORX     0x00100000  /* contains \P, \p, or \X */
#define PCRE2_DUPCAPUSED    0x00200000  /* contains (?| */
#define PCRE2_HASBKC        0x00400000  /* contains \C */
#define PCRE2_HASACCEPT     0x00800000  /* contains (*ACCEPT) */

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
#define tables_length (ctypes_offset + 256)


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

/* The remaining definitions work in both environments. */

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

#define STRING_ACCEPT0               "ACCEPT\0"
#define STRING_COMMIT0               "COMMIT\0"
#define STRING_F0                    "F\0"
#define STRING_FAIL0                 "FAIL\0"
#define STRING_MARK0                 "MARK\0"
#define STRING_PRUNE0                "PRUNE\0"
#define STRING_SKIP0                 "SKIP\0"
#define STRING_THEN                  "THEN"

#define STRING_atomic0               "atomic\0"
#define STRING_pla0                  "pla\0"
#define STRING_plb0                  "plb\0"
#define STRING_napla0                "napla\0"
#define STRING_naplb0                "naplb\0"
#define STRING_nla0                  "nla\0"
#define STRING_nlb0                  "nlb\0"
#define STRING_sr0                   "sr\0"
#define STRING_asr0                  "asr\0"
#define STRING_positive_lookahead0   "positive_lookahead\0"
#define STRING_positive_lookbehind0  "positive_lookbehind\0"
#define STRING_non_atomic_positive_lookahead0   "non_atomic_positive_lookahead\0"
#define STRING_non_atomic_positive_lookbehind0  "non_atomic_positive_lookbehind\0"
#define STRING_negative_lookahead0   "negative_lookahead\0"
#define STRING_negative_lookbehind0  "negative_lookbehind\0"
#define STRING_script_run0           "script_run\0"
#define STRING_atomic_script_run     "atomic_script_run"

#define STRING_alpha0                "alpha\0"
#define STRING_lower0                "lower\0"
#define STRING_upper0                "upper\0"
#define STRING_alnum0                "alnum\0"
#define STRING_ascii0                "ascii\0"
#define STRING_blank0                "blank\0"
#define STRING_cntrl0                "cntrl\0"
#define STRING_digit0                "digit\0"
#define STRING_graph0                "graph\0"
#define STRING_print0                "print\0"
#define STRING_punct0                "punct\0"
#define STRING_space0                "space\0"
#define STRING_word0                 "word\0"
#define STRING_xdigit                "xdigit"

#define STRING_DEFINE                "DEFINE"
#define STRING_VERSION               "VERSION"
#define STRING_WEIRD_STARTWORD       "[:<:]]"
#define STRING_WEIRD_ENDWORD         "[:>:]]"

#define STRING_CR_RIGHTPAR                "CR)"
#define STRING_LF_RIGHTPAR                "LF)"
#define STRING_CRLF_RIGHTPAR              "CRLF)"
#define STRING_ANY_RIGHTPAR               "ANY)"
#define STRING_ANYCRLF_RIGHTPAR           "ANYCRLF)"
#define STRING_NUL_RIGHTPAR               "NUL)"
#define STRING_BSR_ANYCRLF_RIGHTPAR       "BSR_ANYCRLF)"
#define STRING_BSR_UNICODE_RIGHTPAR       "BSR_UNICODE)"
#define STRING_UTF8_RIGHTPAR              "UTF8)"
#define STRING_UTF16_RIGHTPAR             "UTF16)"
#define STRING_UTF32_RIGHTPAR             "UTF32)"
#define STRING_UTF_RIGHTPAR               "UTF)"
#define STRING_UCP_RIGHTPAR               "UCP)"
#define STRING_NO_AUTO_POSSESS_RIGHTPAR   "NO_AUTO_POSSESS)"
#define STRING_NO_DOTSTAR_ANCHOR_RIGHTPAR "NO_DOTSTAR_ANCHOR)"
#define STRING_NO_JIT_RIGHTPAR            "NO_JIT)"
#define STRING_NO_START_OPT_RIGHTPAR      "NO_START_OPT)"
#define STRING_NOTEMPTY_RIGHTPAR          "NOTEMPTY)"
#define STRING_NOTEMPTY_ATSTART_RIGHTPAR  "NOTEMPTY_ATSTART)"
#define STRING_LIMIT_HEAP_EQ              "LIMIT_HEAP="
#define STRING_LIMIT_MATCH_EQ             "LIMIT_MATCH="
#define STRING_LIMIT_DEPTH_EQ             "LIMIT_DEPTH="
#define STRING_LIMIT_RECURSION_EQ         "LIMIT_RECURSION="
#define STRING_MARK                       "MARK"

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
#define STRING_LIMIT_HEAP_EQ              STR_L STR_I STR_M STR_I STR_T STR_UNDERSCORE STR_H STR_E STR_A STR_P STR_EQUALS_SIGN
#define STRING_LIMIT_MATCH_EQ             STR_L STR_I STR_M STR_I STR_T STR_UNDERSCORE STR_M STR_A STR_T STR_C STR_H STR_EQUALS_SIGN
#define STRING_LIMIT_DEPTH_EQ             STR_L STR_I STR_M STR_I STR_T STR_UNDERSCORE STR_D STR_E STR_P STR_T STR_H STR_EQUALS_SIGN
#define STRING_LIMIT_RECURSION_EQ         STR_L STR_I STR_M STR_I STR_T STR_UNDERSCORE STR_R STR_E STR_C STR_U STR_R STR_S STR_I STR_O STR_N STR_EQUALS_SIGN
#define STRING_MARK                       STR_M STR_A STR_R STR_K

#endif  /* SUPPORT_UNICODE */

/* -------------------- End of character and string names -------------------*/

/* -------------------- Definitions for compiled patterns -------------------*/

/* Codes for different types of Unicode property */

#define PT_ANY        0    /* Any property - matches all chars */
#define PT_LAMP       1    /* L& - the union of Lu, Ll, Lt */
#define PT_GC         2    /* Specified general characteristic (e.g. L) */
#define PT_PC         3    /* Specified particular characteristic (e.g. Lu) */
#define PT_SC         4    /* Script (e.g. Han) */
#define PT_ALNUM      5    /* Alphanumeric - the union of L and N */
#define PT_SPACE      6    /* Perl space - Z plus 9,10,12,13 */
#define PT_PXSPACE    7    /* POSIX space - Z plus 9,10,11,12,13 */
#define PT_WORD       8    /* Word - L plus N plus underscore */
#define PT_CLIST      9    /* Pseudo-property: match character list */
#define PT_UCNC      10    /* Universal Character nameable character */
#define PT_TABSIZE   11    /* Size of square table for autopossessify tests */

/* The following special properties are used only in XCLASS items, when POSIX
classes are specified and PCRE2_UCP is set - in other words, for Unicode
handling of these classes. They are not available via the \p or \P escapes like
those in the above list, and so they do not take part in the autopossessifying
table. */

#define PT_PXGRAPH   11    /* [:graph:] - characters that mark the paper */
#define PT_PXPRINT   12    /* [:print:] - [:graph:] plus non-control spaces */
#define PT_PXPUNCT   13    /* [:punct:] - punctuation characters */

/* Flag bits and data types for the extended class (OP_XCLASS) for classes that
contain characters with values greater than 255. */

#define XCL_NOT       0x01    /* Flag: this is a negative class */
#define XCL_MAP       0x02    /* Flag: a 32-byte map is present */
#define XCL_HASPROP   0x04    /* Flag: property checks are present. */

#define XCL_END       0    /* Marks end of individual items */
#define XCL_SINGLE    1    /* Single item (one multibyte char) follows */
#define XCL_RANGE     2    /* A range (two multibyte chars) follows */
#define XCL_PROP      3    /* Unicode property (2-byte property code follows) */
#define XCL_NOTPROP   4    /* Unicode inverted property (ditto) */

/* These are escaped items that aren't just an encoding of a particular data
value such as \n. They must have non-zero values, as check_escape() returns 0
for a data character. In the escapes[] table in pcre2_compile.c their values
are negated in order to distinguish them from data values.

They must appear here in the same order as in the opcode definitions below, up
to ESC_z. There's a dummy for OP_ALLANY because it corresponds to "." in DOTALL
mode rather than an escape sequence. It is also used for [^] in JavaScript
compatibility mode, and for \C in non-utf mode. In non-DOTALL mode, "." behaves
like \N.

Negative numbers are used to encode a backreference (\1, \2, \3, etc.) in
check_escape(). There are tests in the code for an escape greater than ESC_b
and less than ESC_Z to detect the types that may be repeated. These are the
types that consume characters. If any new escapes are put in between that don't
consume a character, that code will have to change. */

enum { ESC_A = 1, ESC_G, ESC_K, ESC_B, ESC_b, ESC_D, ESC_d, ESC_S, ESC_s,
       ESC_W, ESC_w, ESC_N, ESC_dum, ESC_C, ESC_P, ESC_p, ESC_R, ESC_H,
       ESC_h, ESC_V, ESC_v, ESC_X, ESC_Z, ESC_z,
       ESC_E, ESC_Q, ESC_g, ESC_k };


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
  OP_NOT_WORD_BOUNDARY,  /*  4 \B */
  OP_WORD_BOUNDARY,      /*  5 \b */
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
  OP_REF,            /* 113 Match a back reference, casefully */
  OP_REFI,           /* 114 Match a back reference, caselessly */
  OP_DNREF,          /* 115 Match a duplicate name backref, casefully */
  OP_DNREFI,         /* 116 Match a duplicate name backref, caselessly */
  OP_RECURSE,        /* 117 Match a numbered subpattern (possibly recursive) */
  OP_CALLOUT,        /* 118 Call out to external function if provided */
  OP_CALLOUT_STR,    /* 119 Call out with string argument */

  OP_ALT,            /* 120 Start of alternation */
  OP_KET,            /* 121 End of group that doesn't have an unbounded repeat */
  OP_KETRMAX,        /* 122 These two must remain together and in this */
  OP_KETRMIN,        /* 123 order. They are for groups the repeat for ever. */
  OP_KETRPOS,        /* 124 Possessive unlimited repeat. */

  /* The assertions must come before BRA, CBRA, ONCE, and COND. */

  OP_REVERSE,        /* 125 Move pointer back - used in lookbehind assertions */
  OP_ASSERT,         /* 126 Positive lookahead */
  OP_ASSERT_NOT,     /* 127 Negative lookahead */
  OP_ASSERTBACK,     /* 128 Positive lookbehind */
  OP_ASSERTBACK_NOT, /* 129 Negative lookbehind */
  OP_ASSERT_NA,      /* 130 Positive non-atomic lookahead */
  OP_ASSERTBACK_NA,  /* 131 Positive non-atomic lookbehind */

  /* ONCE, SCRIPT_RUN, BRA, BRAPOS, CBRA, CBRAPOS, and COND must come
  immediately after the assertions, with ONCE first, as there's a test for >=
  ONCE for a subpattern that isn't an assertion. The POS versions must
  immediately follow the non-POS versions in each case. */

  OP_ONCE,           /* 132 Atomic group, contains captures */
  OP_SCRIPT_RUN,     /* 133 Non-capture, but check characters' scripts */
  OP_BRA,            /* 134 Start of non-capturing bracket */
  OP_BRAPOS,         /* 135 Ditto, with unlimited, possessive repeat */
  OP_CBRA,           /* 136 Start of capturing bracket */
  OP_CBRAPOS,        /* 137 Ditto, with unlimited, possessive repeat */
  OP_COND,           /* 138 Conditional group */

  /* These five must follow the previous five, in the same order. There's a
  check for >= SBRA to distinguish the two sets. */

  OP_SBRA,           /* 139 Start of non-capturing bracket, check empty  */
  OP_SBRAPOS,        /* 149 Ditto, with unlimited, possessive repeat */
  OP_SCBRA,          /* 141 Start of capturing bracket, check empty */
  OP_SCBRAPOS,       /* 142 Ditto, with unlimited, possessive repeat */
  OP_SCOND,          /* 143 Conditional group, check empty */

  /* The next two pairs must (respectively) be kept together. */

  OP_CREF,           /* 144 Used to hold a capture number as condition */
  OP_DNCREF,         /* 145 Used to point to duplicate names as a condition */
  OP_RREF,           /* 146 Used to hold a recursion number as condition */
  OP_DNRREF,         /* 147 Used to point to duplicate names as a condition */
  OP_FALSE,          /* 148 Always false (used by DEFINE and VERSION) */
  OP_TRUE,           /* 149 Always true (used by VERSION) */

  OP_BRAZERO,        /* 150 These two must remain together and in this */
  OP_BRAMINZERO,     /* 151 order. */
  OP_BRAPOSZERO,     /* 152 */

  /* These are backtracking control verbs */

  OP_MARK,           /* 153 always has an argument */
  OP_PRUNE,          /* 154 */
  OP_PRUNE_ARG,      /* 155 same, but with argument */
  OP_SKIP,           /* 156 */
  OP_SKIP_ARG,       /* 157 same, but with argument */
  OP_THEN,           /* 158 */
  OP_THEN_ARG,       /* 159 same, but with argument */
  OP_COMMIT,         /* 160 */
  OP_COMMIT_ARG,     /* 161 same, but with argument */

  /* These are forced failure and success verbs. FAIL and ACCEPT do accept an
  argument, but these cases can be compiled as, for example, (*MARK:X)(*FAIL)
  without the need for a special opcode. */

  OP_FAIL,           /* 162 */
  OP_ACCEPT,         /* 163 */
  OP_ASSERT_ACCEPT,  /* 164 Used inside assertions */
  OP_CLOSE,          /* 165 Used before OP_ACCEPT to close open captures */

  /* This is used to skip a subpattern with a {0} quantifier */

  OP_SKIPZERO,       /* 166 */

  /* This is used to identify a DEFINE group during compilation so that it can
  be checked for having only one branch. It is changed to OP_FALSE before
  compilation finishes. */

  OP_DEFINE,         /* 167 */

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
only in pcre2_printint.c, which fills out the full names in many cases (and in
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
  "class", "nclass", "xclass", "Ref", "Refi", "DnRef", "DnRefi",  \
  "Recurse", "Callout", "CalloutStr",                             \
  "Alt", "Ket", "KetRmax", "KetRmin", "KetRpos",                  \
  "Reverse", "Assert", "Assert not",                              \
  "Assert back", "Assert back not",                               \
  "Non-atomic assert", "Non-atomic assert back",                  \
  "Once",                                                         \
  "Script run",                                                   \
  "Bra", "BraPos", "CBra", "CBraPos",                             \
  "Cond",                                                         \
  "SBra", "SBraPos", "SCBra", "SCBraPos",                         \
  "SCond",                                                        \
  "Cond ref", "Cond dnref", "Cond rec", "Cond dnrec",             \
  "Cond false", "Cond true",                                      \
  "Brazero", "Braminzero", "Braposzero",                          \
  "*MARK", "*PRUNE", "*PRUNE", "*SKIP", "*SKIP",                  \
  "*THEN", "*THEN", "*COMMIT", "*COMMIT", "*FAIL",                \
  "*ACCEPT", "*ASSERT_ACCEPT",                                    \
  "Close", "Skip zero", "Define"


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
  1+IMM2_SIZE,                   /* REF                                    */ \
  1+IMM2_SIZE,                   /* REFI                                   */ \
  1+2*IMM2_SIZE,                 /* DNREF                                  */ \
  1+2*IMM2_SIZE,                 /* DNREFI                                 */ \
  1+LINK_SIZE,                   /* RECURSE                                */ \
  1+2*LINK_SIZE+1,               /* CALLOUT                                */ \
  0,                             /* CALLOUT_STR - variable length          */ \
  1+LINK_SIZE,                   /* Alt                                    */ \
  1+LINK_SIZE,                   /* Ket                                    */ \
  1+LINK_SIZE,                   /* KetRmax                                */ \
  1+LINK_SIZE,                   /* KetRmin                                */ \
  1+LINK_SIZE,                   /* KetRpos                                */ \
  1+LINK_SIZE,                   /* Reverse                                */ \
  1+LINK_SIZE,                   /* Assert                                 */ \
  1+LINK_SIZE,                   /* Assert not                             */ \
  1+LINK_SIZE,                   /* Assert behind                          */ \
  1+LINK_SIZE,                   /* Assert behind not                      */ \
  1+LINK_SIZE,                   /* NA Assert                              */ \
  1+LINK_SIZE,                   /* NA Assert behind                       */ \
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
  1                              /* DEFINE                                 */

/* A magic value for OP_RREF to indicate the "any recursion" condition. */

#define RREF_ANY  0xffff


/* ---------- Private structures that are mode-independent. ---------- */

/* Structure to hold data for custom memory management. */

typedef struct pcre2_memctl {
  void *    (*malloc)(size_t, void *);
  void      (*free)(void *, void *);
  void      *memory_data;
} pcre2_memctl;

/* Structure for building a chain of open capturing subpatterns during
compiling, so that instructions to close them can be compiled when (*ACCEPT) is
encountered. This is also used to identify subpatterns that contain recursive
back references to themselves, so that they can be made atomic. */

typedef struct open_capitem {
  struct open_capitem *next;    /* Chain link */
  uint16_t number;              /* Capture number */
  uint16_t flag;                /* Set TRUE if recursive back ref */
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
  int16_t scriptx;    /* script extension value */
  int16_t dummy;      /* spare - to round to multiple of 4 bytes */
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

#define UCD_CHARTYPE(ch)    GET_UCD(ch)->chartype
#define UCD_SCRIPT(ch)      GET_UCD(ch)->script
#define UCD_CATEGORY(ch)    PRIV(ucp_gentype)[UCD_CHARTYPE(ch)]
#define UCD_GRAPHBREAK(ch)  GET_UCD(ch)->gbprop
#define UCD_CASESET(ch)     GET_UCD(ch)->caseset
#define UCD_OTHERCASE(ch)   ((uint32_t)((int)ch + (int)(GET_UCD(ch)->other_case)))
#define UCD_SCRIPTX(ch)     GET_UCD(ch)->scriptx

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
extern const int              PRIV(utf8_table1_size);
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
#define _pcre2_ucd_caseless_sets       PCRE2_SUFFIX(_pcre2_ucd_caseless_sets_)
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

extern const uint8_t                   PRIV(OP_lengths)[];
extern const uint32_t                  PRIV(callout_end_delims)[];
extern const uint32_t                  PRIV(callout_start_delims)[];
extern const pcre2_compile_context     PRIV(default_compile_context);
extern const pcre2_convert_context     PRIV(default_convert_context);
extern const pcre2_match_context       PRIV(default_match_context);
extern const uint8_t                   PRIV(default_tables)[];
extern const uint32_t                  PRIV(hspace_list)[];
extern const uint32_t                  PRIV(vspace_list)[];
extern const uint32_t                  PRIV(ucd_caseless_sets)[];
extern const uint32_t                  PRIV(ucd_digit_sets)[];
extern const uint8_t                   PRIV(ucd_script_sets)[];
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

extern int          _pcre2_auto_possessify(PCRE2_UCHAR *, BOOL,
                      const compile_block *);
extern int          _pcre2_check_escape(PCRE2_SPTR *, PCRE2_SPTR, uint32_t *,
                      int *, uint32_t, uint32_t, BOOL, compile_block *);
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
extern BOOL         _pcre2_xclass(uint32_t, PCRE2_SPTR, BOOL);

/* This function is needed only when memmove() is not available. */

#if !defined(VPCOMPAT) && !defined(HAVE_MEMMOVE)
#define _pcre2_memmove               PCRE2_SUFFIX(_pcre2_memmove)
extern void *       _pcre2_memmove(void *, const void *, size_t);
#endif

#endif  /* PCRE2_CODE_UNIT_WIDTH */
#endif  /* PCRE2_INTERNAL_H_IDEMPOTENT_GUARD */

/* End of pcre2_internal.h */
