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

#include "pcre2_internal.h"

#define TYPE_OPTIONS (PCRE2_CONVERT_GLOB| \
  PCRE2_CONVERT_POSIX_BASIC|PCRE2_CONVERT_POSIX_EXTENDED)

#define ALL_OPTIONS (PCRE2_CONVERT_UTF|PCRE2_CONVERT_NO_UTF_CHECK| \
  PCRE2_CONVERT_GLOB_NO_WILD_SEPARATOR| \
  PCRE2_CONVERT_GLOB_NO_STARSTAR| \
  TYPE_OPTIONS)

#define DUMMY_BUFFER_SIZE 100

/* Generated pattern fragments */

#define STR_BACKSLASH_A STR_BACKSLASH STR_A
#define STR_BACKSLASH_z STR_BACKSLASH STR_z
#define STR_COLON_RIGHT_SQUARE_BRACKET STR_COLON STR_RIGHT_SQUARE_BRACKET
#define STR_DOT_STAR_LOOKBEHIND STR_DOT STR_ASTERISK STR_LEFT_PARENTHESIS STR_QUESTION_MARK STR_LESS_THAN_SIGN STR_EQUALS_SIGN
#define STR_LOOKAHEAD_NOT_DOT STR_LEFT_PARENTHESIS STR_QUESTION_MARK STR_EXCLAMATION_MARK STR_BACKSLASH STR_DOT STR_RIGHT_PARENTHESIS
#define STR_QUERY_s STR_LEFT_PARENTHESIS STR_QUESTION_MARK STR_s STR_RIGHT_PARENTHESIS
#define STR_STAR_NUL STR_LEFT_PARENTHESIS STR_ASTERISK STR_N STR_U STR_L STR_RIGHT_PARENTHESIS

/* States for POSIX processing */

enum { POSIX_START_REGEX, POSIX_ANCHORED, POSIX_NOT_BRACKET,
       POSIX_CLASS_NOT_STARTED, POSIX_CLASS_STARTING, POSIX_CLASS_STARTED };

/* Macro to add a character string to the output buffer, checking for overflow. */

#define PUTCHARS(string) \
  { \
  for (const char *s = string; *s != 0; s++) \
    { \
    if (p >= endp) return PCRE2_ERROR_NOMEMORY; \
    *p++ = *s; \
    } \
  }

/* Literals that must be escaped: \ ? * + | . ^ $ { } [ ] ( ) */

static const char *pcre2_escaped_literals =
  STR_BACKSLASH STR_QUESTION_MARK STR_ASTERISK STR_PLUS
  STR_VERTICAL_LINE STR_DOT STR_CIRCUMFLEX_ACCENT STR_DOLLAR_SIGN
  STR_LEFT_CURLY_BRACKET STR_RIGHT_CURLY_BRACKET
  STR_LEFT_SQUARE_BRACKET STR_RIGHT_SQUARE_BRACKET
  STR_LEFT_PARENTHESIS STR_RIGHT_PARENTHESIS;

/* Recognized escaped metacharacters in POSIX basic patterns. */

static const char *posix_meta_escapes =
  STR_LEFT_PARENTHESIS STR_RIGHT_PARENTHESIS
  STR_LEFT_CURLY_BRACKET STR_RIGHT_CURLY_BRACKET
  STR_1 STR_2 STR_3 STR_4 STR_5 STR_6 STR_7 STR_8 STR_9;



/*************************************************
*           Convert a POSIX pattern              *
*************************************************/

/* This function handles both basic and extended POSIX patterns.

Arguments:
  pattype        the pattern type
  pattern        the pattern
  plength        length in code units
  utf            TRUE if UTF
  use_buffer     where to put the output
  use_length     length of use_buffer
  bufflenptr     where to put the used length
  dummyrun       TRUE if a dummy run
  ccontext       the convert context

Returns:         0 => success
                !0 => error code
*/

static int
convert_posix(uint32_t pattype, PCRE2_SPTR pattern, PCRE2_SIZE plength,
  BOOL utf, PCRE2_UCHAR *use_buffer, PCRE2_SIZE use_length,
  PCRE2_SIZE *bufflenptr, BOOL dummyrun, pcre2_convert_context *ccontext)
{
PCRE2_SPTR posix = pattern;
PCRE2_UCHAR *p = use_buffer;
PCRE2_UCHAR *pp = p;
PCRE2_UCHAR *endp = p + use_length - 1;  /* Allow for trailing zero */
PCRE2_SIZE convlength = 0;

uint32_t bracount = 0;
uint32_t posix_state = POSIX_START_REGEX;
uint32_t lastspecial = 0;
BOOL extended = (pattype & PCRE2_CONVERT_POSIX_EXTENDED) != 0;
BOOL nextisliteral = FALSE;

(void)utf;       /* Not used when Unicode not supported */
(void)ccontext;  /* Not currently used */

/* Initialize default for error offset as end of input. */

*bufflenptr = plength;
PUTCHARS(STR_STAR_NUL);

/* Now scan the input. */

while (plength > 0)
  {
  uint32_t c, sc;
  int clength = 1;

  /* Add in the length of the last item, then, if in the dummy run, pull the
  pointer back to the start of the (temporary) buffer and then remember the
  start of the next item. */

  convlength += p - pp;
  if (dummyrun) p = use_buffer;
  pp = p;

  /* Pick up the next character */

#ifndef SUPPORT_UNICODE
  c = *posix;
#else
  GETCHARLENTEST(c, posix, clength);
#endif
  posix += clength;
  plength -= clength;

  sc = nextisliteral? 0 : c;
  nextisliteral = FALSE;

  /* Handle a character within a class. */

  if (posix_state >= POSIX_CLASS_NOT_STARTED)
    {
    if (c == CHAR_RIGHT_SQUARE_BRACKET)
      {
      PUTCHARS(STR_RIGHT_SQUARE_BRACKET);
      posix_state = POSIX_NOT_BRACKET;
      }

    /* Not the end of the class */

    else
      {
      switch (posix_state)
        {
        case POSIX_CLASS_STARTED:
        if (c <= 127 && islower(c)) break;  /* Remain in started state */
        posix_state = POSIX_CLASS_NOT_STARTED;
        if (c == CHAR_COLON  && plength > 0 &&
            *posix == CHAR_RIGHT_SQUARE_BRACKET)
          {
          PUTCHARS(STR_COLON_RIGHT_SQUARE_BRACKET);
          plength--;
          posix++;
          continue;    /* With next character after :] */
          }
        /* Fall through */

        case POSIX_CLASS_NOT_STARTED:
        if (c == CHAR_LEFT_SQUARE_BRACKET)
          posix_state = POSIX_CLASS_STARTING;
        break;

        case POSIX_CLASS_STARTING:
        if (c == CHAR_COLON) posix_state = POSIX_CLASS_STARTED;
        break;
        }

      if (c == CHAR_BACKSLASH) PUTCHARS(STR_BACKSLASH);
      if (p + clength > endp) return PCRE2_ERROR_NOMEMORY;
      memcpy(p, posix - clength, CU2BYTES(clength));
      p += clength;
      }
    }

  /* Handle a character not within a class. */

  else switch(sc)
    {
    case CHAR_LEFT_SQUARE_BRACKET:
    PUTCHARS(STR_LEFT_SQUARE_BRACKET);

#ifdef NEVER
    /* We could handle special cases [[:<:]] and [[:>:]] (which PCRE does
    support) but they are not part of POSIX 1003.1. */

    if (plength >= 6)
      {
      if (posix[0] == CHAR_LEFT_SQUARE_BRACKET &&
          posix[1] == CHAR_COLON &&
          (posix[2] == CHAR_LESS_THAN_SIGN ||
           posix[2] == CHAR_GREATER_THAN_SIGN) &&
          posix[3] == CHAR_COLON &&
          posix[4] == CHAR_RIGHT_SQUARE_BRACKET &&
          posix[5] == CHAR_RIGHT_SQUARE_BRACKET)
        {
        if (p + 6 > endp) return PCRE2_ERROR_NOMEMORY;
        memcpy(p, posix, CU2BYTES(6));
        p += 6;
        posix += 6;
        plength -= 6;
        continue;  /* With next character */
        }
      }
#endif

    /* Handle start of "normal" character classes */

    posix_state = POSIX_CLASS_NOT_STARTED;

    /* Handle ^ and ] as first characters */

    if (plength > 0)
      {
      if (*posix == CHAR_CIRCUMFLEX_ACCENT)
        {
        posix++;
        plength--;
        PUTCHARS(STR_CIRCUMFLEX_ACCENT);
        }
      if (plength > 0 && *posix == CHAR_RIGHT_SQUARE_BRACKET)
        {
        posix++;
        plength--;
        PUTCHARS(STR_RIGHT_SQUARE_BRACKET);
        }
      }
    break;

    case CHAR_BACKSLASH:
    if (plength == 0) return PCRE2_ERROR_END_BACKSLASH;
    if (extended) nextisliteral = TRUE; else
      {
      if (*posix < 127 && strchr(posix_meta_escapes, *posix) != NULL)
        {
        if (isdigit(*posix)) PUTCHARS(STR_BACKSLASH);
        if (p + 1 > endp) return PCRE2_ERROR_NOMEMORY;
        lastspecial = *p++ = *posix++;
        plength--;
        }
      else nextisliteral = TRUE;
      }
    break;

    case CHAR_RIGHT_PARENTHESIS:
    if (!extended || bracount == 0) goto ESCAPE_LITERAL;
    bracount--;
    goto COPY_SPECIAL;

    case CHAR_LEFT_PARENTHESIS:
    bracount++;
    /* Fall through */

    case CHAR_QUESTION_MARK:
    case CHAR_PLUS:
    case CHAR_LEFT_CURLY_BRACKET:
    case CHAR_RIGHT_CURLY_BRACKET:
    case CHAR_VERTICAL_LINE:
    if (!extended) goto ESCAPE_LITERAL;
    /* Fall through */

    case CHAR_DOT:
    case CHAR_DOLLAR_SIGN:
    posix_state = POSIX_NOT_BRACKET;
    COPY_SPECIAL:
    lastspecial = c;
    if (p + 1 > endp) return PCRE2_ERROR_NOMEMORY;
    *p++ = c;
    break;

    case CHAR_ASTERISK:
    if (lastspecial != CHAR_ASTERISK)
      {
      if (!extended && (posix_state < POSIX_NOT_BRACKET ||
          lastspecial == CHAR_LEFT_PARENTHESIS))
        goto ESCAPE_LITERAL;
      goto COPY_SPECIAL;
      }
    break;   /* Ignore second and subsequent asterisks */

    case CHAR_CIRCUMFLEX_ACCENT:
    if (extended) goto COPY_SPECIAL;
    if (posix_state == POSIX_START_REGEX ||
        lastspecial == CHAR_LEFT_PARENTHESIS)
      {
      posix_state = POSIX_ANCHORED;
      goto COPY_SPECIAL;
      }
    /* Fall through */

    default:
    if (c < 128 && strchr(pcre2_escaped_literals, c) != NULL)
      {
      ESCAPE_LITERAL:
      PUTCHARS(STR_BACKSLASH);
      }
    lastspecial = 0xff;  /* Indicates nothing special */
    if (p + clength > endp) return PCRE2_ERROR_NOMEMORY;
    memcpy(p, posix - clength, CU2BYTES(clength));
    p += clength;
    posix_state = POSIX_NOT_BRACKET;
    break;
    }
  }

if (posix_state >= POSIX_CLASS_NOT_STARTED)
  return PCRE2_ERROR_MISSING_SQUARE_BRACKET;
convlength += p - pp;        /* Final segment */
*bufflenptr = convlength;
*p++ = 0;
return 0;
}


/*************************************************
*           Convert a glob pattern               *
*************************************************/

/* Context for writing the output into a buffer. */

typedef struct pcre2_output_context {
  PCRE2_UCHAR *output;                  /* current output position */
  PCRE2_SPTR output_end;                /* output end */
  PCRE2_SIZE output_size;               /* size of the output */
  uint8_t out_str[8];                   /* string copied to the output */
} pcre2_output_context;


/* Write a character into the output.

Arguments:
  out            output context
  chr            the next character
*/

static void
convert_glob_write(pcre2_output_context *out, PCRE2_UCHAR chr)
{
out->output_size++;

if (out->output < out->output_end)
  *out->output++ = chr;
}


/* Write a string into the output.

Arguments:
  out            output context
  length         length of out->out_str
*/

static void
convert_glob_write_str(pcre2_output_context *out, PCRE2_SIZE length)
{
uint8_t *out_str = out->out_str;
PCRE2_UCHAR *output = out->output;
PCRE2_SPTR output_end = out->output_end;
PCRE2_SIZE output_size = out->output_size;

do
  {
  output_size++;

  if (output < output_end)
    *output++ = *out_str++;
  }
while (--length != 0);

out->output = output;
out->output_size = output_size;
}


/* Prints the separator into the output.

Arguments:
  out            output context
  separator      glob separator
  with_escape    backslash is needed before separator
*/

static void
convert_glob_print_separator(pcre2_output_context *out,
  PCRE2_UCHAR separator, BOOL with_escape)
{
if (with_escape)
  convert_glob_write(out, CHAR_BACKSLASH);

convert_glob_write(out, separator);
}


/* Prints a wildcard into the output.

Arguments:
  out            output context
  separator      glob separator
  with_escape    backslash is needed before separator
*/

static void
convert_glob_print_wildcard(pcre2_output_context *out,
  PCRE2_UCHAR separator, BOOL with_escape)
{
out->out_str[0] = CHAR_LEFT_SQUARE_BRACKET;
out->out_str[1] = CHAR_CIRCUMFLEX_ACCENT;
convert_glob_write_str(out, 2);

convert_glob_print_separator(out, separator, with_escape);

convert_glob_write(out, CHAR_RIGHT_SQUARE_BRACKET);
}


/* Parse a posix class.

Arguments:
  from           starting point of scanning the range
  pattern_end    end of pattern
  out            output context

Returns:  >0 => class index
          0  => malformed class
*/

static int
convert_glob_parse_class(PCRE2_SPTR *from, PCRE2_SPTR pattern_end,
  pcre2_output_context *out)
{
static const char *posix_classes = "alnum:alpha:ascii:blank:cntrl:digit:"
  "graph:lower:print:punct:space:upper:word:xdigit:";
PCRE2_SPTR start = *from + 1;
PCRE2_SPTR pattern = start;
const char *class_ptr;
PCRE2_UCHAR c;
int class_index;

while (TRUE)
  {
  if (pattern >= pattern_end) return 0;

  c = *pattern++;

  if (c < CHAR_a || c > CHAR_z) break;
  }

if (c != CHAR_COLON || pattern >= pattern_end ||
    *pattern != CHAR_RIGHT_SQUARE_BRACKET)
  return 0;

class_ptr = posix_classes;
class_index = 1;

while (TRUE)
  {
  if (*class_ptr == CHAR_NUL) return 0;

  pattern = start;

  while (*pattern == (PCRE2_UCHAR) *class_ptr)
    {
    if (*pattern == CHAR_COLON)
      {
      pattern += 2;
      start -= 2;

      do convert_glob_write(out, *start++); while (start < pattern);

      *from = pattern;
      return class_index;
      }
    pattern++;
    class_ptr++;
    }

  while (*class_ptr != CHAR_COLON) class_ptr++;
  class_ptr++;
  class_index++;
  }
}

/* Checks whether the character is in the class.

Arguments:
  class_index    class index
  c              character

Returns:   !0 => character is found in the class
            0 => otherwise
*/

static BOOL
convert_glob_char_in_class(int class_index, PCRE2_UCHAR c)
{
#if PCRE2_CODE_UNIT_WIDTH != 8
if (c > 0xff)
  {
  /* ctype functions are not sane for c > 0xff */
  return 0;
  }
#endif

switch (class_index)
  {
  case 1: return isalnum(c);
  case 2: return isalpha(c);
  case 3: return 1;
  case 4: return c == CHAR_HT || c == CHAR_SPACE;
  case 5: return iscntrl(c);
  case 6: return isdigit(c);
  case 7: return isgraph(c);
  case 8: return islower(c);
  case 9: return isprint(c);
  case 10: return ispunct(c);
  case 11: return isspace(c);
  case 12: return isupper(c);
  case 13: return isalnum(c) || c == CHAR_UNDERSCORE;
  default: return isxdigit(c);
  }
}

/* Parse a range of characters.

Arguments:
  from           starting point of scanning the range
  pattern_end    end of pattern
  out            output context
  separator      glob separator
  with_escape    backslash is needed before separator

Returns:         0 => success
                !0 => error code
*/

static int
convert_glob_parse_range(PCRE2_SPTR *from, PCRE2_SPTR pattern_end,
  pcre2_output_context *out, BOOL utf, PCRE2_UCHAR separator,
  BOOL with_escape, PCRE2_UCHAR escape, BOOL no_wildsep)
{
BOOL is_negative = FALSE;
BOOL separator_seen = FALSE;
BOOL has_prev_c;
PCRE2_SPTR pattern = *from;
PCRE2_SPTR char_start = NULL;
uint32_t c, prev_c;
int len, class_index;

(void)utf; /* Avoid compiler warning. */

if (pattern >= pattern_end)
  {
  *from = pattern;
  return PCRE2_ERROR_MISSING_SQUARE_BRACKET;
  }

if (*pattern == CHAR_EXCLAMATION_MARK
    || *pattern == CHAR_CIRCUMFLEX_ACCENT)
  {
  pattern++;

  if (pattern >= pattern_end)
    {
    *from = pattern;
    return PCRE2_ERROR_MISSING_SQUARE_BRACKET;
    }

  is_negative = TRUE;

  out->out_str[0] = CHAR_LEFT_SQUARE_BRACKET;
  out->out_str[1] = CHAR_CIRCUMFLEX_ACCENT;
  len = 2;

  if (!no_wildsep)
    {
    if (with_escape)
      {
      out->out_str[len] = CHAR_BACKSLASH;
      len++;
      }
    out->out_str[len] = (uint8_t) separator;
    }

  convert_glob_write_str(out, len + 1);
  }
else
  convert_glob_write(out, CHAR_LEFT_SQUARE_BRACKET);

has_prev_c = FALSE;
prev_c = 0;

if (*pattern == CHAR_RIGHT_SQUARE_BRACKET)
  {
  out->out_str[0] = CHAR_BACKSLASH;
  out->out_str[1] = CHAR_RIGHT_SQUARE_BRACKET;
  convert_glob_write_str(out, 2);
  has_prev_c = TRUE;
  prev_c = CHAR_RIGHT_SQUARE_BRACKET;
  pattern++;
  }

while (pattern < pattern_end)
  {
  char_start = pattern;
  GETCHARINCTEST(c, pattern);

  if (c == CHAR_RIGHT_SQUARE_BRACKET)
    {
    convert_glob_write(out, c);

    if (!is_negative && !no_wildsep && separator_seen)
      {
      out->out_str[0] = CHAR_LEFT_PARENTHESIS;
      out->out_str[1] = CHAR_QUESTION_MARK;
      out->out_str[2] = CHAR_LESS_THAN_SIGN;
      out->out_str[3] = CHAR_EXCLAMATION_MARK;
      convert_glob_write_str(out, 4);

      convert_glob_print_separator(out, separator, with_escape);
      convert_glob_write(out, CHAR_RIGHT_PARENTHESIS);
      }

    *from = pattern;
    return 0;
    }

  if (pattern >= pattern_end) break;

  if (c == CHAR_LEFT_SQUARE_BRACKET && *pattern == CHAR_COLON)
    {
    *from = pattern;
    class_index = convert_glob_parse_class(from, pattern_end, out);

    if (class_index != 0)
      {
      pattern = *from;

      has_prev_c = FALSE;
      prev_c = 0;

      if (!is_negative &&
          convert_glob_char_in_class (class_index, separator))
        separator_seen = TRUE;
      continue;
      }
    }
  else if (c == CHAR_MINUS && has_prev_c &&
           *pattern != CHAR_RIGHT_SQUARE_BRACKET)
    {
    convert_glob_write(out, CHAR_MINUS);

    char_start = pattern;
    GETCHARINCTEST(c, pattern);

    if (pattern >= pattern_end) break;

    if (escape != 0 && c == escape)
      {
      char_start = pattern;
      GETCHARINCTEST(c, pattern);
      }
    else if (c == CHAR_LEFT_SQUARE_BRACKET && *pattern == CHAR_COLON)
      {
      *from = pattern;
      return PCRE2_ERROR_CONVERT_SYNTAX;
      }

    if (prev_c > c)
      {
      *from = pattern;
      return PCRE2_ERROR_CONVERT_SYNTAX;
      }

    if (prev_c < separator && separator < c) separator_seen = TRUE;

    has_prev_c = FALSE;
    prev_c = 0;
    }
  else
    {
    if (escape != 0 && c == escape)
      {
      char_start = pattern;
      GETCHARINCTEST(c, pattern);

      if (pattern >= pattern_end) break;
      }

    has_prev_c = TRUE;
    prev_c = c;
    }

  if (c == CHAR_LEFT_SQUARE_BRACKET || c == CHAR_RIGHT_SQUARE_BRACKET ||
      c == CHAR_BACKSLASH || c == CHAR_MINUS)
    convert_glob_write(out, CHAR_BACKSLASH);

  if (c == separator) separator_seen = TRUE;

  do convert_glob_write(out, *char_start++); while (char_start < pattern);
  }

*from = pattern;
return PCRE2_ERROR_MISSING_SQUARE_BRACKET;
}


/* Prints a (*COMMIT) into the output.

Arguments:
  out            output context
*/

static void
convert_glob_print_commit(pcre2_output_context *out)
{
out->out_str[0] = CHAR_LEFT_PARENTHESIS;
out->out_str[1] = CHAR_ASTERISK;
out->out_str[2] = CHAR_C;
out->out_str[3] = CHAR_O;
out->out_str[4] = CHAR_M;
out->out_str[5] = CHAR_M;
out->out_str[6] = CHAR_I;
out->out_str[7] = CHAR_T;
convert_glob_write_str(out, 8);
convert_glob_write(out, CHAR_RIGHT_PARENTHESIS);
}


/* Bash glob converter.

Arguments:
  pattype        the pattern type
  pattern        the pattern
  plength        length in code units
  utf            TRUE if UTF
  use_buffer     where to put the output
  use_length     length of use_buffer
  bufflenptr     where to put the used length
  dummyrun       TRUE if a dummy run
  ccontext       the convert context

Returns:         0 => success
                !0 => error code
*/

static int
convert_glob(uint32_t options, PCRE2_SPTR pattern, PCRE2_SIZE plength,
  BOOL utf, PCRE2_UCHAR *use_buffer, PCRE2_SIZE use_length,
  PCRE2_SIZE *bufflenptr, BOOL dummyrun, pcre2_convert_context *ccontext)
{
pcre2_output_context out;
PCRE2_SPTR pattern_start = pattern;
PCRE2_SPTR pattern_end = pattern + plength;
PCRE2_UCHAR separator = ccontext->glob_separator;
PCRE2_UCHAR escape = ccontext->glob_escape;
PCRE2_UCHAR c;
BOOL no_wildsep = (options & PCRE2_CONVERT_GLOB_NO_WILD_SEPARATOR) != 0;
BOOL no_starstar = (options & PCRE2_CONVERT_GLOB_NO_STARSTAR) != 0;
BOOL in_atomic = FALSE;
BOOL after_starstar = FALSE;
BOOL no_slash_z = FALSE;
BOOL with_escape, is_start, after_separator;
int result = 0;

(void)utf; /* Avoid compiler warning. */

#ifdef SUPPORT_UNICODE
if (utf && (separator >= 128 || escape >= 128))
  {
  /* Currently only ASCII characters are supported. */
  *bufflenptr = 0;
  return PCRE2_ERROR_CONVERT_SYNTAX;
  }
#endif

with_escape = strchr(pcre2_escaped_literals, separator) != NULL;

/* Initialize default for error offset as end of input. */
out.output = use_buffer;
out.output_end = use_buffer + use_length;
out.output_size = 0;

out.out_str[0] = CHAR_LEFT_PARENTHESIS;
out.out_str[1] = CHAR_QUESTION_MARK;
out.out_str[2] = CHAR_s;
out.out_str[3] = CHAR_RIGHT_PARENTHESIS;
convert_glob_write_str(&out, 4);

is_start = TRUE;

if (pattern < pattern_end && pattern[0] == CHAR_ASTERISK)
  {
  if (no_wildsep)
    is_start = FALSE;
  else if (!no_starstar && pattern + 1 < pattern_end &&
           pattern[1] == CHAR_ASTERISK)
    is_start = FALSE;
  }

if (is_start)
  {
  out.out_str[0] = CHAR_BACKSLASH;
  out.out_str[1] = CHAR_A;
  convert_glob_write_str(&out, 2);
  }

while (pattern < pattern_end)
  {
  c = *pattern++;

  if (c == CHAR_ASTERISK)
    {
    is_start = pattern == pattern_start + 1;

    if (in_atomic)
      {
      convert_glob_write(&out, CHAR_RIGHT_PARENTHESIS);
      in_atomic = FALSE;
      }

    if (!no_starstar && pattern < pattern_end && *pattern == CHAR_ASTERISK)
      {
      after_separator = is_start || (pattern[-2] == separator);

      do pattern++; while (pattern < pattern_end &&
                           *pattern == CHAR_ASTERISK);

      if (pattern >= pattern_end)
        {
        no_slash_z = TRUE;
        break;
        }

      after_starstar = TRUE;

      if (after_separator && escape != 0 && *pattern == escape &&
          pattern + 1 < pattern_end && pattern[1] == separator)
        pattern++;

      if (is_start)
        {
        if (*pattern != separator) continue;

        out.out_str[0] = CHAR_LEFT_PARENTHESIS;
        out.out_str[1] = CHAR_QUESTION_MARK;
        out.out_str[2] = CHAR_COLON;
        out.out_str[3] = CHAR_BACKSLASH;
        out.out_str[4] = CHAR_A;
        out.out_str[5] = CHAR_VERTICAL_LINE;
        convert_glob_write_str(&out, 6);

        convert_glob_print_separator(&out, separator, with_escape);
        convert_glob_write(&out, CHAR_RIGHT_PARENTHESIS);

        pattern++;
        continue;
        }

      convert_glob_print_commit(&out);

      if (!after_separator || *pattern != separator)
        {
        out.out_str[0] = CHAR_DOT;
        out.out_str[1] = CHAR_ASTERISK;
        out.out_str[2] = CHAR_QUESTION_MARK;
        convert_glob_write_str(&out, 3);
        continue;
        }

      out.out_str[0] = CHAR_LEFT_PARENTHESIS;
      out.out_str[1] = CHAR_QUESTION_MARK;
      out.out_str[2] = CHAR_COLON;
      out.out_str[3] = CHAR_DOT;
      out.out_str[4] = CHAR_ASTERISK;
      out.out_str[5] = CHAR_QUESTION_MARK;

      convert_glob_write_str(&out, 6);

      convert_glob_print_separator(&out, separator, with_escape);

      out.out_str[0] = CHAR_RIGHT_PARENTHESIS;
      out.out_str[1] = CHAR_QUESTION_MARK;
      out.out_str[2] = CHAR_QUESTION_MARK;
      convert_glob_write_str(&out, 3);

      pattern++;
      continue;
      }

    if (pattern < pattern_end && *pattern == CHAR_ASTERISK)
      {
      do pattern++; while (pattern < pattern_end &&
                           *pattern == CHAR_ASTERISK);
      }

    if (no_wildsep)
      {
      if (pattern >= pattern_end)
        {
        no_slash_z = TRUE;
        break;
        }

      /* Start check must be after the end check. */
      if (is_start) continue;
      }

    if (!is_start)
      {
      if (after_starstar)
        {
        out.out_str[0] = CHAR_LEFT_PARENTHESIS;
        out.out_str[1] = CHAR_QUESTION_MARK;
        out.out_str[2] = CHAR_GREATER_THAN_SIGN;
        convert_glob_write_str(&out, 3);
        in_atomic = TRUE;
        }
      else
        convert_glob_print_commit(&out);
      }

    if (no_wildsep)
      convert_glob_write(&out, CHAR_DOT);
    else
      convert_glob_print_wildcard(&out, separator, with_escape);

    out.out_str[0] = CHAR_ASTERISK;
    out.out_str[1] = CHAR_QUESTION_MARK;
    if (pattern >= pattern_end)
      out.out_str[1] = CHAR_PLUS;
    convert_glob_write_str(&out, 2);
    continue;
    }

  if (c == CHAR_QUESTION_MARK)
    {
    if (no_wildsep)
      convert_glob_write(&out, CHAR_DOT);
    else
      convert_glob_print_wildcard(&out, separator, with_escape);
    continue;
    }

  if (c == CHAR_LEFT_SQUARE_BRACKET)
    {
    result = convert_glob_parse_range(&pattern, pattern_end,
      &out, utf, separator, with_escape, escape, no_wildsep);
    if (result != 0) break;
    continue;
    }

  if (escape != 0 && c == escape)
    {
    if (pattern >= pattern_end)
      {
      result = PCRE2_ERROR_CONVERT_SYNTAX;
      break;
      }
    c = *pattern++;
    }

  if (c < 128 && strchr(pcre2_escaped_literals, c) != NULL)
    convert_glob_write(&out, CHAR_BACKSLASH);

  convert_glob_write(&out, c);
  }

if (result == 0)
  {
  if (!no_slash_z)
    {
    out.out_str[0] = CHAR_BACKSLASH;
    out.out_str[1] = CHAR_z;
    convert_glob_write_str(&out, 2);
    }

  if (in_atomic)
    convert_glob_write(&out, CHAR_RIGHT_PARENTHESIS);

  convert_glob_write(&out, CHAR_NUL);

  if (!dummyrun && out.output_size != (PCRE2_SIZE) (out.output - use_buffer))
    result = PCRE2_ERROR_NOMEMORY;
  }

if (result != 0)
  {
  *bufflenptr = pattern - pattern_start;
  return result;
  }

*bufflenptr = out.output_size - 1;
return 0;
}


/*************************************************
*                Convert pattern                 *
*************************************************/

/* This is the external-facing function for converting other forms of pattern
into PCRE2 regular expression patterns. On error, the bufflenptr argument is
used to return an offset in the original pattern.

Arguments:
  pattern     the input pattern
  plength     length of input, or PCRE2_ZERO_TERMINATED
  options     options bits
  buffptr     pointer to pointer to output buffer
  bufflenptr  pointer to length of output buffer
  ccontext    convert context or NULL

Returns:      0 for success, else an error code (+ve or -ve)
*/

PCRE2_EXP_DEFN int PCRE2_CALL_CONVENTION
pcre2_pattern_convert(PCRE2_SPTR pattern, PCRE2_SIZE plength, uint32_t options,
  PCRE2_UCHAR **buffptr, PCRE2_SIZE *bufflenptr,
  pcre2_convert_context *ccontext)
{
int rc;
PCRE2_UCHAR dummy_buffer[DUMMY_BUFFER_SIZE];
PCRE2_UCHAR *use_buffer = dummy_buffer;
PCRE2_SIZE use_length = DUMMY_BUFFER_SIZE;
BOOL utf = (options & PCRE2_CONVERT_UTF) != 0;
uint32_t pattype = options & TYPE_OPTIONS;

if (pattern == NULL || bufflenptr == NULL) return PCRE2_ERROR_NULL;

if ((options & ~ALL_OPTIONS) != 0 ||        /* Undefined bit set */
    (pattype & (~pattype+1)) != pattype ||  /* More than one type set */
    pattype == 0)                           /* No type set */
  {
  *bufflenptr = 0;                          /* Error offset */
  return PCRE2_ERROR_BADOPTION;
  }

if (plength == PCRE2_ZERO_TERMINATED) plength = PRIV(strlen)(pattern);
if (ccontext == NULL) ccontext =
  (pcre2_convert_context *)(&PRIV(default_convert_context));

/* Check UTF if required. */

#ifndef SUPPORT_UNICODE
if (utf)
  {
  *bufflenptr = 0;  /* Error offset */
  return PCRE2_ERROR_UNICODE_NOT_SUPPORTED;
  }
#else
if (utf && (options & PCRE2_CONVERT_NO_UTF_CHECK) == 0)
  {
  PCRE2_SIZE erroroffset;
  rc = PRIV(valid_utf)(pattern, plength, &erroroffset);
  if (rc != 0)
    {
    *bufflenptr = erroroffset;
    return rc;
    }
  }
#endif

/* If buffptr is not NULL, and what it points to is not NULL, we are being
provided with a buffer and a length, so set them as the buffer to use. */

if (buffptr != NULL && *buffptr != NULL)
  {
  use_buffer = *buffptr;
  use_length = *bufflenptr;
  }

/* Call an individual converter, either just once (if a buffer was provided or
just the length is needed), or twice (if a memory allocation is required). */

for (int i = 0; i < 2; i++)
  {
  PCRE2_UCHAR *allocated;
  BOOL dummyrun = buffptr == NULL || *buffptr == NULL;

  switch(pattype)
    {
    case PCRE2_CONVERT_GLOB:
    rc = convert_glob(options & ~PCRE2_CONVERT_GLOB, pattern, plength, utf,
      use_buffer, use_length, bufflenptr, dummyrun, ccontext);
    break;

    case PCRE2_CONVERT_POSIX_BASIC:
    case PCRE2_CONVERT_POSIX_EXTENDED:
    rc = convert_posix(pattype, pattern, plength, utf, use_buffer, use_length,
      bufflenptr, dummyrun, ccontext);
    break;

    default:
    goto EXIT;
    }

  if (rc != 0 ||           /* Error */
      buffptr == NULL ||   /* Just the length is required */
      *buffptr != NULL)    /* Buffer was provided or allocated */
    return rc;

  /* Allocate memory for the buffer, with hidden space for an allocator at
  the start. The next time round the loop runs the conversion for real. */

  allocated = PRIV(memctl_malloc)(sizeof(pcre2_memctl) +
    (*bufflenptr + 1)*PCRE2_CODE_UNIT_WIDTH, (pcre2_memctl *)ccontext);
  if (allocated == NULL) return PCRE2_ERROR_NOMEMORY;
  *buffptr = (PCRE2_UCHAR *)(((char *)allocated) + sizeof(pcre2_memctl));

  use_buffer = *buffptr;
  use_length = *bufflenptr + 1;
  }

/* Something went terribly wrong. Trigger an assert and return an error */
PCRE2_DEBUG_UNREACHABLE();

EXIT:

*bufflenptr = 0;  /* Error offset */
return PCRE2_ERROR_INTERNAL;
}


/*************************************************
*            Free converted pattern              *
*************************************************/

/* This frees a converted pattern that was put in newly-allocated memory.

Argument:   the converted pattern
Returns:    nothing
*/

PCRE2_EXP_DEFN void PCRE2_CALL_CONVENTION
pcre2_converted_pattern_free(PCRE2_UCHAR *converted)
{
if (converted != NULL)
  {
  pcre2_memctl *memctl =
    (pcre2_memctl *)((char *)converted - sizeof(pcre2_memctl));
  memctl->free(memctl, memctl->memory_data);
  }
}

/* End of pcre2_convert.c */
