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


#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "pcre2_internal.h"

#define STRING(a)  # a
#define XSTRING(s) STRING(s)

/* The texts of compile-time error messages. Compile-time error numbers start
at COMPILE_ERROR_BASE (100).

This used to be a table of strings, but in order to reduce the number of
relocations needed when a shared library is loaded dynamically, it is now one
long string. We cannot use a table of offsets, because the lengths of inserts
such as XSTRING(MAX_NAME_SIZE) are not known. Instead,
pcre2_get_error_message() counts through to the one it wants - this isn't a
performance issue because these strings are used only when there is an error.

Each substring ends with \0 to insert a null character. This includes the final
substring, so that the whole string ends with \0\0, which can be detected when
counting through. */

static const unsigned char compile_error_texts[] =
  "no error\0"
  "\\ at end of pattern\0"
  "\\c at end of pattern\0"
  "unrecognized character follows \\\0"
  "numbers out of order in {} quantifier\0"
  /* 5 */
  "number too big in {} quantifier\0"
  "missing terminating ] for character class\0"
  "escape sequence is invalid in character class\0"
  "range out of order in character class\0"
  "quantifier does not follow a repeatable item\0"
  /* 10 */
  "internal error: unexpected repeat\0"
  "unrecognized character after (? or (?-\0"
  "POSIX named classes are supported only within a class\0"
  "POSIX collating elements are not supported\0"
  "missing closing parenthesis\0"
  /* 15 */
  "reference to non-existent subpattern\0"
  "pattern passed as NULL\0"
  "unrecognised compile-time option bit(s)\0"
  "missing ) after (?# comment\0"
  "parentheses are too deeply nested\0"
  /* 20 */
  "regular expression is too large\0"
  "failed to allocate heap memory\0"
  "unmatched closing parenthesis\0"
  "internal error: code overflow\0"
  "missing closing parenthesis for condition\0"
  /* 25 */
  "lookbehind assertion is not fixed length\0"
  "a relative value of zero is not allowed\0"
  "conditional subpattern contains more than two branches\0"
  "assertion expected after (?( or (?(?C)\0"
  "digit expected after (?+ or (?-\0"
  /* 30 */
  "unknown POSIX class name\0"
  "internal error in pcre2_study(): should not occur\0"
  "this version of PCRE2 does not have Unicode support\0"
  "parentheses are too deeply nested (stack check)\0"
  "character code point value in \\x{} or \\o{} is too large\0"
  /* 35 */
  "lookbehind is too complicated\0"
  "\\C is not allowed in a lookbehind assertion in UTF-" XSTRING(PCRE2_CODE_UNIT_WIDTH) " mode\0"
  "PCRE2 does not support \\F, \\L, \\l, \\N{name}, \\U, or \\u\0"
  "number after (?C is greater than 255\0"
  "closing parenthesis for (?C expected\0"
  /* 40 */
  "invalid escape sequence in (*VERB) name\0"
  "unrecognized character after (?P\0"
  "syntax error in subpattern name (missing terminator?)\0"
  "two named subpatterns have the same name (PCRE2_DUPNAMES not set)\0"
  "subpattern name must start with a non-digit\0"
  /* 45 */
  "this version of PCRE2 does not have support for \\P, \\p, or \\X\0"
  "malformed \\P or \\p sequence\0"
  "unknown property name after \\P or \\p\0"
  "subpattern name is too long (maximum " XSTRING(MAX_NAME_SIZE) " code units)\0"
  "too many named subpatterns (maximum " XSTRING(MAX_NAME_COUNT) ")\0"
  /* 50 */
  "invalid range in character class\0"
  "octal value is greater than \\377 in 8-bit non-UTF-8 mode\0"
  "internal error: overran compiling workspace\0"
  "internal error: previously-checked referenced subpattern not found\0"
  "DEFINE subpattern contains more than one branch\0"
  /* 55 */
  "missing opening brace after \\o\0"
  "internal error: unknown newline setting\0"
  "\\g is not followed by a braced, angle-bracketed, or quoted name/number or by a plain number\0"
  "(?R (recursive pattern call) must be followed by a closing parenthesis\0"
  /* "an argument is not allowed for (*ACCEPT), (*FAIL), or (*COMMIT)\0" */
  "obsolete error (should not occur)\0"  /* Was the above */
  /* 60 */
  "(*VERB) not recognized or malformed\0"
  "subpattern number is too big\0"
  "subpattern name expected\0"
  "internal error: parsed pattern overflow\0"
  "non-octal character in \\o{} (closing brace missing?)\0"
  /* 65 */
  "different names for subpatterns of the same number are not allowed\0"
  "(*MARK) must have an argument\0"
  "non-hex character in \\x{} (closing brace missing?)\0"
#ifndef EBCDIC
  "\\c must be followed by a printable ASCII character\0"
#else
  "\\c must be followed by a letter or one of [\\]^_?\0"
#endif
  "\\k is not followed by a braced, angle-bracketed, or quoted name\0"
  /* 70 */
  "internal error: unknown meta code in check_lookbehinds()\0"
  "\\N is not supported in a class\0"
  "callout string is too long\0"
  "disallowed Unicode code point (>= 0xd800 && <= 0xdfff)\0"
  "using UTF is disabled by the application\0"
  /* 75 */
  "using UCP is disabled by the application\0"
  "name is too long in (*MARK), (*PRUNE), (*SKIP), or (*THEN)\0"
  "character code point value in \\u.... sequence is too large\0"
  "digits missing in \\x{} or \\o{} or \\N{U+}\0"
  "syntax error or number too big in (?(VERSION condition\0"
  /* 80 */
  "internal error: unknown opcode in auto_possessify()\0"
  "missing terminating delimiter for callout with string argument\0"
  "unrecognized string delimiter follows (?C\0"
  "using \\C is disabled by the application\0"
  "(?| and/or (?J: or (?x: parentheses are too deeply nested\0"
  /* 85 */
  "using \\C is disabled in this PCRE2 library\0"
  "regular expression is too complicated\0"
  "lookbehind assertion is too long\0"
  "pattern string is longer than the limit set by the application\0"
  "internal error: unknown code in parsed pattern\0"
  /* 90 */
  "internal error: bad code value in parsed_skip()\0"
  "PCRE2_EXTRA_ALLOW_SURROGATE_ESCAPES is not allowed in UTF-16 mode\0"
  "invalid option bits with PCRE2_LITERAL\0"
  "\\N{U+dddd} is supported only in Unicode (UTF) mode\0"
  "invalid hyphen in option setting\0"
  /* 95 */
  "(*alpha_assertion) not recognized\0"
  "script runs require Unicode support, which this version of PCRE2 does not have\0"
  "too many capturing groups (maximum 65535)\0"
  "atomic assertion expected after (?( or (?(?C)\0"
  ;

/* Match-time and UTF error texts are in the same format. */

static const unsigned char match_error_texts[] =
  "no error\0"
  "no match\0"
  "partial match\0"
  "UTF-8 error: 1 byte missing at end\0"
  "UTF-8 error: 2 bytes missing at end\0"
  /* 5 */
  "UTF-8 error: 3 bytes missing at end\0"
  "UTF-8 error: 4 bytes missing at end\0"
  "UTF-8 error: 5 bytes missing at end\0"
  "UTF-8 error: byte 2 top bits not 0x80\0"
  "UTF-8 error: byte 3 top bits not 0x80\0"
  /* 10 */
  "UTF-8 error: byte 4 top bits not 0x80\0"
  "UTF-8 error: byte 5 top bits not 0x80\0"
  "UTF-8 error: byte 6 top bits not 0x80\0"
  "UTF-8 error: 5-byte character is not allowed (RFC 3629)\0"
  "UTF-8 error: 6-byte character is not allowed (RFC 3629)\0"
  /* 15 */
  "UTF-8 error: code points greater than 0x10ffff are not defined\0"
  "UTF-8 error: code points 0xd800-0xdfff are not defined\0"
  "UTF-8 error: overlong 2-byte sequence\0"
  "UTF-8 error: overlong 3-byte sequence\0"
  "UTF-8 error: overlong 4-byte sequence\0"
  /* 20 */
  "UTF-8 error: overlong 5-byte sequence\0"
  "UTF-8 error: overlong 6-byte sequence\0"
  "UTF-8 error: isolated byte with 0x80 bit set\0"
  "UTF-8 error: illegal byte (0xfe or 0xff)\0"
  "UTF-16 error: missing low surrogate at end\0"
  /* 25 */
  "UTF-16 error: invalid low surrogate\0"
  "UTF-16 error: isolated low surrogate\0"
  "UTF-32 error: code points 0xd800-0xdfff are not defined\0"
  "UTF-32 error: code points greater than 0x10ffff are not defined\0"
  "bad data value\0"
  /* 30 */
  "patterns do not all use the same character tables\0"
  "magic number missing\0"
  "pattern compiled in wrong mode: 8/16/32-bit error\0"
  "bad offset value\0"
  "bad option value\0"
  /* 35 */
  "invalid replacement string\0"
  "bad offset into UTF string\0"
  "callout error code\0"              /* Never returned by PCRE2 itself */
  "invalid data in workspace for DFA restart\0"
  "too much recursion for DFA matching\0"
  /* 40 */
  "backreference condition or recursion test is not supported for DFA matching\0"
  "function is not supported for DFA matching\0"
  "pattern contains an item that is not supported for DFA matching\0"
  "workspace size exceeded in DFA matching\0"
  "internal error - pattern overwritten?\0"
  /* 45 */
  "bad JIT option\0"
  "JIT stack limit reached\0"
  "match limit exceeded\0"
  "no more memory\0"
  "unknown substring\0"
  /* 50 */
  "non-unique substring name\0"
  "NULL argument passed\0"
  "nested recursion at the same subject position\0"
  "matching depth limit exceeded\0"
  "requested value is not available\0"
  /* 55 */
  "requested value is not set\0"
  "offset limit set without PCRE2_USE_OFFSET_LIMIT\0"
  "bad escape sequence in replacement string\0"
  "expected closing curly bracket in replacement string\0"
  "bad substitution in replacement string\0"
  /* 60 */
  "match with end before start or start moved backwards is not supported\0"
  "too many replacements (more than INT_MAX)\0"
  "bad serialized data\0"
  "heap limit exceeded\0"
  "invalid syntax\0"
  /* 65 */
  "internal error - duplicate substitution match\0"
  "PCRE2_MATCH_INVALID_UTF is not supported for DFA matching\0"
  ;


/*************************************************
*            Return error message                *
*************************************************/

/* This function copies an error message into a buffer whose units are of an
appropriate width. Error numbers are positive for compile-time errors, and
negative for match-time errors (except for UTF errors), but the numbers are all
distinct.

Arguments:
  enumber       error number
  buffer        where to put the message (zero terminated)
  size          size of the buffer in code units

Returns:        length of message if all is well
                negative on error
*/

PCRE2_EXP_DEFN int PCRE2_CALL_CONVENTION
pcre2_get_error_message(int enumber, PCRE2_UCHAR *buffer, PCRE2_SIZE size)
{
const unsigned char *message;
PCRE2_SIZE i;
int n;

if (size == 0) return PCRE2_ERROR_NOMEMORY;

if (enumber >= COMPILE_ERROR_BASE)  /* Compile error */
  {
  message = compile_error_texts;
  n = enumber - COMPILE_ERROR_BASE;
  }
else if (enumber < 0)               /* Match or UTF error */
  {
  message = match_error_texts;
  n = -enumber;
  }
else                                /* Invalid error number */
  {
  message = (unsigned char *)"\0";  /* Empty message list */
  n = 1;
  }

for (; n > 0; n--)
  {
  while (*message++ != CHAR_NUL) {};
  if (*message == CHAR_NUL) return PCRE2_ERROR_BADDATA;
  }

for (i = 0; *message != 0; i++)
  {
  if (i >= size - 1)
    {
    buffer[i] = 0;     /* Terminate partial message */
    return PCRE2_ERROR_NOMEMORY;
    }
  buffer[i] = *message++;
  }

buffer[i] = 0;
return (int)i;
}

/* End of pcre2_error.c */
