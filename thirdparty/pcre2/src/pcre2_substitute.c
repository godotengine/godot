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

#define PTR_STACK_SIZE 20

#define SUBSTITUTE_OPTIONS \
  (PCRE2_SUBSTITUTE_EXTENDED|PCRE2_SUBSTITUTE_GLOBAL| \
   PCRE2_SUBSTITUTE_LITERAL|PCRE2_SUBSTITUTE_MATCHED| \
   PCRE2_SUBSTITUTE_OVERFLOW_LENGTH|PCRE2_SUBSTITUTE_REPLACEMENT_ONLY| \
   PCRE2_SUBSTITUTE_UNKNOWN_UNSET|PCRE2_SUBSTITUTE_UNSET_EMPTY)



/*************************************************
*           Find end of substitute text          *
*************************************************/

/* In extended mode, we recognize ${name:+set text:unset text} and similar
constructions. This requires the identification of unescaped : and }
characters. This function scans for such. It must deal with nested ${
constructions. The pointer to the text is updated, either to the required end
character, or to where an error was detected.

Arguments:
  code      points to the compiled expression (for options)
  ptrptr    points to the pointer to the start of the text (updated)
  ptrend    end of the whole string
  last      TRUE if the last expected string (only } recognized)

Returns:    0 on success
            negative error code on failure
*/

static int
find_text_end(const pcre2_code *code, PCRE2_SPTR *ptrptr, PCRE2_SPTR ptrend,
  BOOL last)
{
int rc = 0;
uint32_t nestlevel = 0;
BOOL literal = FALSE;
PCRE2_SPTR ptr = *ptrptr;

for (; ptr < ptrend; ptr++)
  {
  if (literal)
    {
    if (ptr[0] == CHAR_BACKSLASH && ptr < ptrend - 1 && ptr[1] == CHAR_E)
      {
      literal = FALSE;
      ptr += 1;
      }
    }

  else if (*ptr == CHAR_RIGHT_CURLY_BRACKET)
    {
    if (nestlevel == 0) goto EXIT;
    nestlevel--;
    }

  else if (*ptr == CHAR_COLON && !last && nestlevel == 0) goto EXIT;

  else if (*ptr == CHAR_DOLLAR_SIGN)
    {
    if (ptr < ptrend - 1 && ptr[1] == CHAR_LEFT_CURLY_BRACKET)
      {
      nestlevel++;
      ptr += 1;
      }
    }

  else if (*ptr == CHAR_BACKSLASH)
    {
    int erc;
    int errorcode;
    uint32_t ch;

    if (ptr < ptrend - 1) switch (ptr[1])
      {
      case CHAR_L:
      case CHAR_l:
      case CHAR_U:
      case CHAR_u:
      ptr += 1;
      continue;
      }

    ptr += 1;  /* Must point after \ */
    erc = PRIV(check_escape)(&ptr, ptrend, &ch, &errorcode,
      code->overall_options, code->extra_options, code->top_bracket, FALSE, NULL);
    ptr -= 1;  /* Back to last code unit of escape */
    if (errorcode != 0)
      {
      /* errorcode from check_escape is positive, so must not be returned by
      pcre2_substitute(). */
      rc = PCRE2_ERROR_BADREPESCAPE;
      goto EXIT;
      }

    switch(erc)
      {
      case 0:      /* Data character */
      case ESC_b:  /* Data character */
      case ESC_v:  /* Data character */
      case ESC_E:  /* Isolated \E is ignored */
      break;

      case ESC_Q:
      literal = TRUE;
      break;

      case ESC_g:
      /* The \g<name> form (\g<number> already handled by check_escape)

      Don't worry about finding the matching ">". We are super, super lenient
      about validating ${} replacements inside find_text_end(), so we certainly
      don't need to worry about other syntax. Importantly, a \g<..> or $<...>
      sequence can't contain a '}' character. */
      break;

      default:
      if (erc < 0)
          break;  /* capture group reference */
      rc = PCRE2_ERROR_BADREPESCAPE;
      goto EXIT;
      }
    }
  }

rc = PCRE2_ERROR_REPMISSINGBRACE;   /* Terminator not found */

EXIT:
*ptrptr = ptr;
return rc;
}


/*************************************************
*           Validate group name                  *
*************************************************/

/* This function scans for a capture group name, validating it
consists of legal characters, is not empty, and does not exceed
MAX_NAME_SIZE.

Arguments:
  ptrptr    points to the pointer to the start of the text (updated)
  ptrend    end of the whole string
  utf       true if the input is UTF-encoded
  ctypes    pointer to the character types table

Returns:    TRUE if a name was read
            FALSE otherwise
*/

static BOOL
read_name_subst(PCRE2_SPTR *ptrptr, PCRE2_SPTR ptrend, BOOL utf,
    const uint8_t* ctypes)
{
PCRE2_SPTR ptr = *ptrptr;
PCRE2_SPTR nameptr = ptr;

if (ptr >= ptrend)                 /* No characters in name */
  goto FAILED;

/* We do not need to check whether the name starts with a non-digit.
We are simply referencing names here, not defining them. */

/* See read_name in the pcre2_compile.c for the corresponding logic
restricting group names inside the pattern itself. */

#ifdef SUPPORT_UNICODE
if (utf)
  {
  uint32_t c, type;

  while (ptr < ptrend)
    {
    GETCHAR(c, ptr);
    type = UCD_CHARTYPE(c);
    if (type != ucp_Nd && PRIV(ucp_gentype)[type] != ucp_L &&
        c != CHAR_UNDERSCORE) break;
    ptr++;
    FORWARDCHARTEST(ptr, ptrend);
    }
  }
else
#else
(void)utf;  /* Avoid compiler warning */
#endif      /* SUPPORT_UNICODE */

/* Handle group names in non-UTF modes. */

  {
  while (ptr < ptrend && MAX_255(*ptr) && (ctypes[*ptr] & ctype_word) != 0)
    {
    ptr++;
    }
  }

/* Check name length */

if (ptr - nameptr > MAX_NAME_SIZE)
  goto FAILED;

/* Subpattern names must not be empty */
if (ptr == nameptr)
  goto FAILED;

*ptrptr = ptr;
return TRUE;

FAILED:
*ptrptr = ptr;
return FALSE;
}


/*************************************************
*              Case transformations              *
*************************************************/

#define PCRE2_SUBSTITUTE_CASE_NONE                 0
// 1, 2, 3 are PCRE2_SUBSTITUTE_CASE_LOWER, UPPER, TITLE_FIRST.
#define PCRE2_SUBSTITUTE_CASE_REVERSE_TITLE_FIRST  4

typedef struct {
  int to_case; /* One of PCRE2_SUBSTITUTE_CASE_xyz */
  BOOL single_char;
} case_state;

/* Helper to guess how much a string is likely to increase in size when
case-transformed. Usually, strings don't change size at all, but some rare
characters do grow. Estimate +10%, plus another few characters.

Performing this estimation is unfortunate, but inevitable, since we can't call
the callout if we ran out of buffer space to prepare its input.

Because this estimate is inexact (and in pathological cases, underestimates the
required buffer size) we must document that when you have a
substitute_case_callout, and you are using PCRE2_SUBSTITUTE_OVERFLOW_LENGTH, you
may need more than two calls to determine the final buffer size. */

static PCRE2_SIZE
pessimistic_case_inflation(PCRE2_SIZE len)
{
return (len >> 3u) + 10;
}

/* Case transformation behaviour if no callout is passed. */

static PCRE2_SIZE
default_substitute_case_callout(
  PCRE2_SPTR input, PCRE2_SIZE input_len,
  PCRE2_UCHAR *output, PCRE2_SIZE output_cap,
  case_state *state, const pcre2_code *code)
{
PCRE2_SPTR input_end = input + input_len;
#ifdef SUPPORT_UNICODE
BOOL utf;
BOOL ucp;
#endif
PCRE2_UCHAR temp[6];
BOOL next_to_upper;
BOOL rest_to_upper;
BOOL single_char;
BOOL overflow = FALSE;
PCRE2_SIZE written = 0;

/* Helpful simplifying invariant: input and output are disjoint buffers.
I believe that this code is technically undefined behaviour, because the two
pointers input/output are "unrelated" pointers and hence not comparable. Casting
via char* bypasses some but not all of those technical rules. It is not included
in release builds, in any case. */
PCRE2_ASSERT((char *)(input + input_len) <= (char *)output ||
             (char *)(output + output_cap) <= (char *)input);

#ifdef SUPPORT_UNICODE
utf = (code->overall_options & PCRE2_UTF) != 0;
ucp = (code->overall_options & PCRE2_UCP) != 0;
#endif

if (input_len == 0) return 0;

switch (state->to_case)
  {
  default:
  PCRE2_DEBUG_UNREACHABLE();
  return 0;

  case PCRE2_SUBSTITUTE_CASE_LOWER: // Can be single_char TRUE or FALSE
  case PCRE2_SUBSTITUTE_CASE_UPPER: // Can only be single_char FALSE
  next_to_upper = rest_to_upper = (state->to_case == PCRE2_SUBSTITUTE_CASE_UPPER);
  break;

  case PCRE2_SUBSTITUTE_CASE_TITLE_FIRST: // Can be single_char TRUE or FALSE
  next_to_upper = TRUE;
  rest_to_upper = FALSE;
  state->to_case = PCRE2_SUBSTITUTE_CASE_LOWER;
  break;

  case PCRE2_SUBSTITUTE_CASE_REVERSE_TITLE_FIRST: // Can only be single_char FALSE
  next_to_upper = FALSE;
  rest_to_upper = TRUE;
  state->to_case = PCRE2_SUBSTITUTE_CASE_UPPER;
  break;
  }

single_char = state->single_char;
if (single_char)
  state->to_case = PCRE2_SUBSTITUTE_CASE_NONE;

while (input < input_end)
  {
  uint32_t ch;
  unsigned int chlen;

  GETCHARINCTEST(ch, input);

#ifdef SUPPORT_UNICODE
  if ((utf || ucp) && ch >= 128)
    {
    uint32_t type = UCD_CHARTYPE(ch);
    if (PRIV(ucp_gentype)[type] == ucp_L &&
        type != (next_to_upper? ucp_Lu : ucp_Ll))
      ch = UCD_OTHERCASE(ch);

    /* TODO This is far from correct... it doesn't support the SpecialCasing.txt
    mappings, but worse, it's not even correct for all the ordinary case
    mappings. We should add support for those (at least), and then add the
    SpecialCasing.txt mappings for Esszet and ligatures, and finally use the
    Turkish casing flag on the match context. */
    }
  else
#endif
  if (MAX_255(ch))
    {
    if (((code->tables + cbits_offset +
        (next_to_upper? cbit_upper:cbit_lower)
        )[ch/8] & (1u << (ch%8))) == 0)
      ch = (code->tables + fcc_offset)[ch];
    }

#ifdef SUPPORT_UNICODE
  if (utf) chlen = PRIV(ord2utf)(ch, temp); else
#endif
    {
    temp[0] = ch;
    chlen = 1;
    }

  if (!overflow && chlen <= output_cap)
    {
    memcpy(output, temp, CU2BYTES(chlen));
    output += chlen;
    output_cap -= chlen;
    }
  else
    {
    overflow = TRUE;
    }

  if (chlen > ~(PCRE2_SIZE)0 - written)  /* Integer overflow */
    return ~(PCRE2_SIZE)0;
  written += chlen;

  next_to_upper = rest_to_upper;

  /* memcpy the remainder, if only transforming a single character. */

  if (single_char)
    {
    PCRE2_SIZE rest_len = input_end - input;

    if (!overflow && rest_len <= output_cap)
      memcpy(output, input, CU2BYTES(rest_len));

    if (rest_len > ~(PCRE2_SIZE)0 - written)  /* Integer overflow */
      return ~(PCRE2_SIZE)0;
    written += rest_len;

    return written;
    }
  }

return written;
}

/* Helper to perform the call to the substitute_case_callout. We wrap the
user-provided callout because our internal arguments are slightly extended. We
don't want the user callout to handle the case of "\l" (first character only to
lowercase) or "\l\U" (first character to lowercase, rest to uppercase) because
those are not operations defined by Unicode. Instead the user callout simply
needs to provide the three Unicode primitives: lower, upper, titlecase. */

static PCRE2_SIZE
do_case_copy(
  PCRE2_UCHAR *input_output, PCRE2_SIZE input_len, PCRE2_SIZE output_cap,
  case_state *state, BOOL utf,
  PCRE2_SIZE (*substitute_case_callout)(PCRE2_SPTR, PCRE2_SIZE, PCRE2_UCHAR *,
                                        PCRE2_SIZE, int, void *),
  void *substitute_case_callout_data)
{
PCRE2_SPTR input = input_output;
PCRE2_UCHAR *output = input_output;
PCRE2_SIZE rc;
PCRE2_SIZE rc2;
int ch1_to_case;
int rest_to_case;
PCRE2_UCHAR ch1[6];
PCRE2_SIZE ch1_len;
PCRE2_SPTR rest;
PCRE2_SIZE rest_len;
BOOL ch1_overflow = FALSE;
BOOL rest_overflow = FALSE;

#if PCRE2_CODE_UNIT_WIDTH == 32 || !defined(SUPPORT_UNICODE)
(void)utf; /* Avoid compiler warning. */
#endif

PCRE2_ASSERT(input_len != 0);

switch (state->to_case)
  {
  default:
  PCRE2_DEBUG_UNREACHABLE();
  return 0;

  case PCRE2_SUBSTITUTE_CASE_LOWER: // Can be single_char TRUE or FALSE
  case PCRE2_SUBSTITUTE_CASE_UPPER: // Can only be single_char FALSE
  case PCRE2_SUBSTITUTE_CASE_TITLE_FIRST: // Can be single_char TRUE or FALSE

  /* The easy case, where our internal casing operations align with those of
  the callout. */

  if (state->single_char == FALSE)
    {
    rc = substitute_case_callout(input, input_len, output, output_cap,
                                 state->to_case, substitute_case_callout_data);

    if (state->to_case == PCRE2_SUBSTITUTE_CASE_TITLE_FIRST)
      state->to_case = PCRE2_SUBSTITUTE_CASE_LOWER;

    return rc;
    }

  ch1_to_case = state->to_case;
  rest_to_case = PCRE2_SUBSTITUTE_CASE_NONE;
  break;

  case PCRE2_SUBSTITUTE_CASE_REVERSE_TITLE_FIRST: // Can only be single_char FALSE
  ch1_to_case = PCRE2_SUBSTITUTE_CASE_LOWER;
  rest_to_case = PCRE2_SUBSTITUTE_CASE_UPPER;
  break;
  }

/* Identify the leading character. Take copy, because its storage overlaps with
`output`, and hence may be scrambled by the callout. */

  {
  PCRE2_SPTR ch_end = input;
  uint32_t ch;

  GETCHARINCTEST(ch, ch_end);
  (void) ch;
  PCRE2_ASSERT(ch_end <= input + input_len && ch_end - input <= 6);
  ch1_len = ch_end - input;
  memcpy(ch1, input, CU2BYTES(ch1_len));
  }

rest = input + ch1_len;
rest_len = input_len - ch1_len;

/* Transform just ch1. The buffers are always in-place (input == output). With a
custom callout, we need a loop to discover its required buffer size. The loop
wouldn't be required if the callout were well-behaved, but it might be naughty
and return "5" the first time, then "10" the next time we call it using the
exact same input! */

  {
  PCRE2_SIZE ch1_cap;
  PCRE2_SIZE max_ch1_cap;

  ch1_cap = ch1_len;  /* First attempt uses the space vacated by ch1. */
  PCRE2_ASSERT(output_cap >= input_len && input_len >= rest_len);
  max_ch1_cap = output_cap - rest_len;

  while (TRUE)
    {
    rc = substitute_case_callout(ch1, ch1_len, output, ch1_cap, ch1_to_case,
                                 substitute_case_callout_data);
    if (rc == ~(PCRE2_SIZE)0) return rc;

    if (rc <= ch1_cap) break;

    if (rc > max_ch1_cap)
      {
      ch1_overflow = TRUE;
      break;
      }

    /* Move the rest to the right, to make room for expanding ch1. */

    memmove(input_output + rc, rest, CU2BYTES(rest_len));
    rest = input + rc;

    ch1_cap = rc;

    /* Proof of loop termination: `ch1_cap` is growing on each iteration, but
    the loop ends if `rc` reaches the (unchanging) upper bound of output_cap. */
    }
  }

if (rest_to_case == PCRE2_SUBSTITUTE_CASE_NONE)
  {
  if (!ch1_overflow)
    {
    PCRE2_ASSERT(rest_len <= output_cap - rc);
    memmove(output + rc, rest, CU2BYTES(rest_len));
    }
  rc2 = rest_len;

  state->to_case = PCRE2_SUBSTITUTE_CASE_NONE;
  }
else
  {
  PCRE2_UCHAR dummy[1];

  rc2 = substitute_case_callout(rest, rest_len,
                                ch1_overflow? dummy : output + rc,
                                ch1_overflow? 0u : output_cap - rc,
                                rest_to_case, substitute_case_callout_data);
  if (rc2 == ~(PCRE2_SIZE)0) return rc2;

  if (!ch1_overflow && rc2 > output_cap - rc) rest_overflow = TRUE;

  /* If ch1 grows so that `xform(ch1)+rest` can't fit in the buffer, but then
  `rest` shrinks, it's actually possible for the total calculated length of
  `xform(ch1)+xform(rest)` to come out at less than output_cap. But we can't
  report that, because it would make it seem that the operation succeeded.
  If either of xform(ch1) or xform(rest) won't fit in the buffer, our final
  result must be > output_cap. */
  if (ch1_overflow && rc2 < rest_len)
    rc2 = rest_len;

  state->to_case = PCRE2_SUBSTITUTE_CASE_UPPER;
  }

if (rc2 > ~(PCRE2_SIZE)0 - rc)  /* Integer overflow */
  return ~(PCRE2_SIZE)0;

PCRE2_ASSERT(!(ch1_overflow || rest_overflow) || rc + rc2 > output_cap);
(void)rest_overflow;

return rc + rc2;
}


/*************************************************
*              Match and substitute              *
*************************************************/

/* This function applies a compiled re to a subject string and creates a new
string with substitutions. The first 7 arguments are the same as for
pcre2_match(). Either string length may be PCRE2_ZERO_TERMINATED.

Arguments:
  code            points to the compiled expression
  subject         points to the subject string
  length          length of subject string (may contain binary zeros)
  start_offset    where to start in the subject string
  options         option bits
  match_data      points to a match_data block, or is NULL
  context         points a PCRE2 context
  replacement     points to the replacement string
  rlength         length of replacement string
  buffer          where to put the substituted string
  blength         points to length of buffer; updated to length of string

Returns:          >= 0 number of substitutions made
                  < 0 an error code
                  PCRE2_ERROR_BADREPLACEMENT means invalid use of $
*/

/* This macro checks for space in the buffer before copying into it. On
overflow, either give an error immediately, or keep on, accumulating the
length. */

#define CHECKMEMCPY(from, length_) \
  do {    \
     PCRE2_SIZE chkmc_length = length_; \
     if (overflowed) \
       {  \
       if (chkmc_length > ~(PCRE2_SIZE)0 - extra_needed)  /* Integer overflow */ \
         goto TOOLARGEREPLACE; \
       extra_needed += chkmc_length; \
       }  \
     else if (lengthleft < chkmc_length) \
       {  \
       if ((suboptions & PCRE2_SUBSTITUTE_OVERFLOW_LENGTH) == 0) goto NOROOM; \
       overflowed = TRUE; \
       extra_needed = chkmc_length - lengthleft; \
       }  \
     else \
       {  \
       memcpy(buffer + buff_offset, from, CU2BYTES(chkmc_length)); \
       buff_offset += chkmc_length; \
       lengthleft -= chkmc_length; \
       }  \
     }    \
  while (0)

/* This macro checks for space and copies characters with casing modifications.
On overflow, it behaves as for CHECKMEMCPY().

When substitute_case_callout is NULL, the source and destination buffers must
not overlap, because our default handler does not support this. */

#define CHECKCASECPY_BASE(length_, do_call) \
  do {    \
     PCRE2_SIZE chkcc_length = (PCRE2_SIZE)(length_); \
     PCRE2_SIZE chkcc_rc; \
     do_call \
     if (lengthleft < chkcc_rc) \
       {  \
       if ((suboptions & PCRE2_SUBSTITUTE_OVERFLOW_LENGTH) == 0) goto NOROOM; \
       overflowed = TRUE; \
       extra_needed = chkcc_rc - lengthleft; \
       }  \
     else \
       {  \
       buff_offset += chkcc_rc; \
       lengthleft -= chkcc_rc; \
       }  \
     }    \
  while (0)

#define CHECKCASECPY_DEFAULT(from, length_) \
  CHECKCASECPY_BASE(length_, { \
    chkcc_rc = default_substitute_case_callout(from, chkcc_length,         \
                                               buffer + buff_offset,       \
                                               overflowed? 0 : lengthleft, \
                                               &forcecase, code);          \
    if (overflowed) \
      { \
      if (chkcc_rc > ~(PCRE2_SIZE)0 - extra_needed)  /* Integer overflow */ \
        goto TOOLARGEREPLACE; \
      extra_needed += chkcc_rc; \
      break; \
      } \
  })

#define CHECKCASECPY_CALLOUT(length_) \
  CHECKCASECPY_BASE(length_, { \
    chkcc_rc = do_case_copy(buffer + buff_offset, chkcc_length, \
                            lengthleft, &forcecase, utf,        \
                            substitute_case_callout,            \
                            substitute_case_callout_data);      \
    if (chkcc_rc == ~(PCRE2_SIZE)0) goto CASEERROR; \
  })

/* This macro does a delayed case transformation, for the situation when we have
a case-forcing callout. */

#define DELAYEDFORCECASE() \
  do {      \
     PCRE2_SIZE chars_outstanding = (buff_offset - casestart_offset) + \
            (extra_needed - casestart_extra_needed); \
     if (chars_outstanding > 0) \
       {    \
       if (overflowed) \
         {  \
         PCRE2_SIZE guess = pessimistic_case_inflation(chars_outstanding); \
         if (guess > ~(PCRE2_SIZE)0 - extra_needed)  /* Integer overflow */ \
           goto TOOLARGEREPLACE; \
         extra_needed += guess; \
         }  \
       else \
         {  \
         /* Rewind the buffer */ \
         lengthleft += (buff_offset - casestart_offset); \
         buff_offset = casestart_offset; \
         /* Care! In-place case transformation */ \
         CHECKCASECPY_CALLOUT(chars_outstanding); \
         }  \
       }    \
     }      \
  while (0)


/* Here's the function */

PCRE2_EXP_DEFN int PCRE2_CALL_CONVENTION
pcre2_substitute(const pcre2_code *code, PCRE2_SPTR subject, PCRE2_SIZE length,
  PCRE2_SIZE start_offset, uint32_t options, pcre2_match_data *match_data,
  pcre2_match_context *mcontext, PCRE2_SPTR replacement, PCRE2_SIZE rlength,
  PCRE2_UCHAR *buffer, PCRE2_SIZE *blength)
{
int rc;
int subs;
uint32_t ovector_count;
uint32_t goptions = 0;
uint32_t suboptions;
pcre2_match_data *internal_match_data = NULL;
BOOL escaped_literal = FALSE;
BOOL overflowed = FALSE;
BOOL use_existing_match;
BOOL replacement_only;
BOOL utf = (code->overall_options & PCRE2_UTF) != 0;
PCRE2_UCHAR temp[6];
PCRE2_SPTR ptr;
PCRE2_SPTR repend = NULL;
PCRE2_SIZE extra_needed = 0;
PCRE2_SIZE buff_offset, buff_length, lengthleft, fraglength;
PCRE2_SIZE *ovector;
PCRE2_SIZE ovecsave[3];
pcre2_substitute_callout_block scb;
PCRE2_SIZE sub_start_extra_needed;
PCRE2_SIZE (*substitute_case_callout)(PCRE2_SPTR, PCRE2_SIZE, PCRE2_UCHAR *,
                                      PCRE2_SIZE, int, void *) = NULL;
void *substitute_case_callout_data = NULL;

/* General initialization */

buff_offset = 0;
lengthleft = buff_length = *blength;
*blength = PCRE2_UNSET;
ovecsave[0] = ovecsave[1] = ovecsave[2] = PCRE2_UNSET;

if (mcontext != NULL)
  {
  substitute_case_callout = mcontext->substitute_case_callout;
  substitute_case_callout_data = mcontext->substitute_case_callout_data;
  }

/* Partial matching is not valid. This must come after setting *blength to
PCRE2_UNSET, so as not to imply an offset in the replacement. */

if ((options & (PCRE2_PARTIAL_HARD|PCRE2_PARTIAL_SOFT)) != 0)
  return PCRE2_ERROR_BADOPTION;

/* Validate length and find the end of the replacement. A NULL replacement of
zero length is interpreted as an empty string. */

if (replacement == NULL)
  {
  if (rlength != 0) return PCRE2_ERROR_NULL;
  replacement = (PCRE2_SPTR)"";
  }

if (rlength == PCRE2_ZERO_TERMINATED) rlength = PRIV(strlen)(replacement);
repend = replacement + rlength;

/* Check for using a match that has already happened. Note that the subject
pointer in the match data may be NULL after a no-match. */

use_existing_match = ((options & PCRE2_SUBSTITUTE_MATCHED) != 0);
replacement_only = ((options & PCRE2_SUBSTITUTE_REPLACEMENT_ONLY) != 0);

/* If starting from an existing match, there must be an externally provided
match data block. We create an internal match_data block in two cases: (a) an
external one is not supplied (and we are not starting from an existing match);
(b) an existing match is to be used for the first substitution. In the latter
case, we copy the existing match into the internal block, except for any cached
heap frame size and pointer. This ensures that no changes are made to the
external match data block. */

/* WARNING: In both cases below a general context is constructed "by hand"
because calling pcre2_general_context_create() involves a memory allocation. If
the contents of a general context control block are ever changed there will
have to be changes below. */

if (match_data == NULL)
  {
  pcre2_general_context gcontext;
  if (use_existing_match) return PCRE2_ERROR_NULL;
  gcontext.memctl = (mcontext == NULL)?
    ((const pcre2_real_code *)code)->memctl :
    ((pcre2_real_match_context *)mcontext)->memctl;
  match_data = internal_match_data =
    pcre2_match_data_create_from_pattern(code, &gcontext);
  if (internal_match_data == NULL) return PCRE2_ERROR_NOMEMORY;
  }

else if (use_existing_match)
  {
  int pairs;
  pcre2_general_context gcontext;
  gcontext.memctl = (mcontext == NULL)?
    ((const pcre2_real_code *)code)->memctl :
    ((pcre2_real_match_context *)mcontext)->memctl;
  pairs = (code->top_bracket + 1 < match_data->oveccount)?
    code->top_bracket + 1 : match_data->oveccount;
  internal_match_data = pcre2_match_data_create(match_data->oveccount,
    &gcontext);
  if (internal_match_data == NULL) return PCRE2_ERROR_NOMEMORY;
  memcpy(internal_match_data, match_data, offsetof(pcre2_match_data, ovector)
    + 2*pairs*sizeof(PCRE2_SIZE));
  internal_match_data->heapframes = NULL;
  internal_match_data->heapframes_size = 0;
  match_data = internal_match_data;
  }

/* Remember ovector details */

ovector = pcre2_get_ovector_pointer(match_data);
ovector_count = pcre2_get_ovector_count(match_data);

/* Fixed things in the callout block */

scb.version = 0;
scb.input = subject;
scb.output = (PCRE2_SPTR)buffer;
scb.ovector = ovector;

/* A NULL subject of zero length is treated as an empty string. */

if (subject == NULL)
  {
  if (length != 0) return PCRE2_ERROR_NULL;
  subject = (PCRE2_SPTR)"";
  }

/* Find length of zero-terminated subject */

if (length == PCRE2_ZERO_TERMINATED)
  length = subject? PRIV(strlen)(subject) : 0;

/* Check UTF replacement string if necessary. */

#ifdef SUPPORT_UNICODE
if (utf && (options & PCRE2_NO_UTF_CHECK) == 0)
  {
  rc = PRIV(valid_utf)(replacement, rlength, &(match_data->startchar));
  if (rc != 0)
    {
    match_data->leftchar = 0;
    goto EXIT;
    }
  }
#endif  /* SUPPORT_UNICODE */

/* Save the substitute options and remove them from the match options. */

suboptions = options & SUBSTITUTE_OPTIONS;
options &= ~SUBSTITUTE_OPTIONS;

/* Error if the start match offset is greater than the length of the subject. */

if (start_offset > length)
  {
  match_data->leftchar = 0;
  rc = PCRE2_ERROR_BADOFFSET;
  goto EXIT;
  }

/* Copy up to the start offset, unless only the replacement is required. */

if (!replacement_only) CHECKMEMCPY(subject, start_offset);

/* Loop for global substituting. If PCRE2_SUBSTITUTE_MATCHED is set, the first
match is taken from the match_data that was passed in. */

subs = 0;
do
  {
  PCRE2_SPTR ptrstack[PTR_STACK_SIZE];
  uint32_t ptrstackptr = 0;
  case_state forcecase = { PCRE2_SUBSTITUTE_CASE_NONE, FALSE };
  PCRE2_SIZE casestart_offset = 0;
  PCRE2_SIZE casestart_extra_needed = 0;

  if (use_existing_match)
    {
    rc = match_data->rc;
    use_existing_match = FALSE;
    }
  else rc = pcre2_match(code, subject, length, start_offset, options|goptions,
    match_data, mcontext);

#ifdef SUPPORT_UNICODE
  if (utf) options |= PCRE2_NO_UTF_CHECK;  /* Only need to check once */
#endif

  /* Any error other than no match returns the error code. No match when not
  doing the special after-empty-match global rematch, or when at the end of the
  subject, breaks the global loop. Otherwise, advance the starting point by one
  character, copying it to the output, and try again. */

  if (rc < 0)
    {
    PCRE2_SIZE save_start;

    if (rc != PCRE2_ERROR_NOMATCH) goto EXIT;
    if (goptions == 0 || start_offset >= length) break;

    /* Advance by one code point. Then, if CRLF is a valid newline sequence and
    we have advanced into the middle of it, advance one more code point. In
    other words, do not start in the middle of CRLF, even if CR and LF on their
    own are valid newlines. */

    save_start = start_offset++;
    if (subject[start_offset-1] == CHAR_CR &&
        (code->newline_convention == PCRE2_NEWLINE_CRLF ||
         code->newline_convention == PCRE2_NEWLINE_ANY ||
         code->newline_convention == PCRE2_NEWLINE_ANYCRLF) &&
        start_offset < length &&
        subject[start_offset] == CHAR_LF)
      start_offset++;

    /* Otherwise, in UTF mode, advance past any secondary code points. */

    else if ((code->overall_options & PCRE2_UTF) != 0)
      {
#if PCRE2_CODE_UNIT_WIDTH == 8
      while (start_offset < length && (subject[start_offset] & 0xc0) == 0x80)
        start_offset++;
#elif PCRE2_CODE_UNIT_WIDTH == 16
      while (start_offset < length &&
            (subject[start_offset] & 0xfc00) == 0xdc00)
        start_offset++;
#endif
      }

    /* Copy what we have advanced past (unless not required), reset the special
    global options, and continue to the next match. */

    fraglength = start_offset - save_start;
    if (!replacement_only) CHECKMEMCPY(subject + save_start, fraglength);
    goptions = 0;
    continue;
    }

  /* Handle a successful match. Matches that use \K to end before they start
  or start before the current point in the subject are not supported. */

  if (ovector[1] < ovector[0] || ovector[0] < start_offset)
    {
    rc = PCRE2_ERROR_BADSUBSPATTERN;
    goto EXIT;
    }

  /* Check for the same match as previous. This is legitimate after matching an
  empty string that starts after the initial match offset. We have tried again
  at the match point in case the pattern is one like /(?<=\G.)/ which can never
  match at its starting point, so running the match achieves the bumpalong. If
  we do get the same (null) match at the original match point, it isn't such a
  pattern, so we now do the empty string magic. In all other cases, a repeat
  match should never occur. */

  if (ovecsave[0] == ovector[0] && ovecsave[1] == ovector[1])
    {
    if (ovector[0] == ovector[1] && ovecsave[2] != start_offset)
      {
      goptions = PCRE2_NOTEMPTY_ATSTART | PCRE2_ANCHORED;
      ovecsave[2] = start_offset;
      continue;    /* Back to the top of the loop */
      }
    rc = PCRE2_ERROR_INTERNAL_DUPMATCH;
    goto EXIT;
    }

  /* Count substitutions with a paranoid check for integer overflow; surely no
  real call to this function would ever hit this! */

  if (subs == INT_MAX)
    {
    rc = PCRE2_ERROR_TOOMANYREPLACE;
    goto EXIT;
    }
  subs++;

  /* Copy the text leading up to the match (unless not required); remember
  where the insert begins and how many ovector pairs are set; and remember how
  much space we have requested in extra_needed. */

  if (rc == 0) rc = ovector_count;
  fraglength = ovector[0] - start_offset;
  if (!replacement_only) CHECKMEMCPY(subject + start_offset, fraglength);
  scb.output_offsets[0] = buff_offset;
  scb.oveccount = rc;
  sub_start_extra_needed = extra_needed;

  /* Process the replacement string. If the entire replacement is literal, just
  copy it with length check. */

  ptr = replacement;
  if ((suboptions & PCRE2_SUBSTITUTE_LITERAL) != 0)
    {
    CHECKMEMCPY(ptr, rlength);
    }

  /* Within a non-literal replacement, which must be scanned character by
  character, local literal mode can be set by \Q, but only in extended mode
  when backslashes are being interpreted. In extended mode we must handle
  nested substrings that are to be reprocessed. */

  else for (;;)
    {
    uint32_t ch;
    unsigned int chlen;
    int group;
    uint32_t special;
    PCRE2_SPTR text1_start = NULL;
    PCRE2_SPTR text1_end = NULL;
    PCRE2_SPTR text2_start = NULL;
    PCRE2_SPTR text2_end = NULL;
    PCRE2_UCHAR name[MAX_NAME_SIZE + 1];

    /* If at the end of a nested substring, pop the stack. */

    if (ptr >= repend)
      {
      if (ptrstackptr == 0) break;       /* End of replacement string */
      repend = ptrstack[--ptrstackptr];
      ptr = ptrstack[--ptrstackptr];
      continue;
      }

    /* Handle the next character */

    if (escaped_literal)
      {
      if (ptr[0] == CHAR_BACKSLASH && ptr < repend - 1 && ptr[1] == CHAR_E)
        {
        escaped_literal = FALSE;
        ptr += 2;
        continue;
        }
      goto LOADLITERAL;
      }

    /* Not in literal mode. */

    if (*ptr == CHAR_DOLLAR_SIGN)
      {
      BOOL inparens;
      BOOL inangle;
      BOOL star;
      PCRE2_SIZE sublength;
      PCRE2_UCHAR next;
      PCRE2_SPTR subptr, subptrend;

      if (++ptr >= repend) goto BAD;
      if ((next = *ptr) == CHAR_DOLLAR_SIGN) goto LOADLITERAL;

      special = 0;
      text1_start = NULL;
      text1_end = NULL;
      text2_start = NULL;
      text2_end = NULL;
      group = -1;
      inparens = FALSE;
      inangle = FALSE;
      star = FALSE;
      subptr = NULL;
      subptrend = NULL;

      /* Special $ sequences, as supported by Perl, JavaScript, .NET and others. */
      if (next == CHAR_AMPERSAND)
        {
        ++ptr;
        group = 0;
        goto GROUP_SUBSTITUTE;
        }
      if (next == CHAR_GRAVE_ACCENT || next == CHAR_APOSTROPHE)
        {
        ++ptr;
        rc = pcre2_substring_length_bynumber(match_data, 0, &sublength);
        if (rc < 0) goto PTREXIT; /* (Sanity-check ovector before reading from it.) */

        if (next == CHAR_GRAVE_ACCENT)
          {
          subptr = subject;
          subptrend = subject + ovector[0];
          }
        else
          {
          subptr = subject + ovector[1];
          subptrend = subject + length;
          }

        goto SUBPTR_SUBSTITUTE;
        }
      if (next == CHAR_UNDERSCORE)
        {
        /* Java, .NET support $_ for "entire input string". */
        ++ptr;
        subptr = subject;
        subptrend = subject + length;
        goto SUBPTR_SUBSTITUTE;
        }

      if (next == CHAR_LEFT_CURLY_BRACKET)
        {
        if (++ptr >= repend) goto BAD;
        next = *ptr;
        inparens = TRUE;
        }
      else if (next == CHAR_LESS_THAN_SIGN)
        {
        /* JavaScript compatibility syntax, $<name>. Processes only named
        groups (not numbered) and does not support extensions such as star
        (you can do ${name} and ${*name}, but not $<*name>). */
        if (++ptr >= repend) goto BAD;
        next = *ptr;
        inangle = TRUE;
        }

      if (!inangle && next == CHAR_ASTERISK)
        {
        if (++ptr >= repend) goto BAD;
        next = *ptr;
        star = TRUE;
        }

      if (!star && !inangle && next >= CHAR_0 && next <= CHAR_9)
        {
        group = next - CHAR_0;
        while (++ptr < repend)
          {
          next = *ptr;
          if (next < CHAR_0 || next > CHAR_9) break;
          group = group * 10 + (next - CHAR_0);

          /* A check for a number greater than the hightest captured group
          is sufficient here; no need for a separate overflow check. If unknown
          groups are to be treated as unset, just skip over any remaining
          digits and carry on. */

          if (group > code->top_bracket)
            {
            if ((suboptions & PCRE2_SUBSTITUTE_UNKNOWN_UNSET) != 0)
              {
              while (++ptr < repend && *ptr >= CHAR_0 && *ptr <= CHAR_9);
              break;
              }
            else
              {
              rc = PCRE2_ERROR_NOSUBSTRING;
              goto PTREXIT;
              }
            }
          }
        }
      else
        {
        PCRE2_SIZE name_len;
        PCRE2_SPTR name_start = ptr;
        if (!read_name_subst(&ptr, repend, utf, code->tables + ctypes_offset))
          goto BAD;
        name_len = ptr - name_start;
        memcpy(name, name_start, CU2BYTES(name_len));
        name[name_len] = 0;
        }

      next = 0; /* not used or updated after this point */
      (void)next;

      /* In extended mode we recognize ${name:+set text:unset text} and
      ${name:-default text}. */

      if (inparens)
        {
        if ((suboptions & PCRE2_SUBSTITUTE_EXTENDED) != 0 &&
             !star && ptr < repend - 2 && *ptr == CHAR_COLON)
          {
          special = *(++ptr);
          if (special != CHAR_PLUS && special != CHAR_MINUS)
            {
            rc = PCRE2_ERROR_BADSUBSTITUTION;
            goto PTREXIT;
            }

          text1_start = ++ptr;
          rc = find_text_end(code, &ptr, repend, special == CHAR_MINUS);
          if (rc != 0) goto PTREXIT;
          text1_end = ptr;

          if (special == CHAR_PLUS && *ptr == CHAR_COLON)
            {
            text2_start = ++ptr;
            rc = find_text_end(code, &ptr, repend, TRUE);
            if (rc != 0) goto PTREXIT;
            text2_end = ptr;
            }
          }

        else
          {
          if (ptr >= repend || *ptr != CHAR_RIGHT_CURLY_BRACKET)
            {
            rc = PCRE2_ERROR_REPMISSINGBRACE;
            goto PTREXIT;
            }
          }

        ptr++;
        }

      if (inangle)
        {
        if (ptr >= repend || *ptr != CHAR_GREATER_THAN_SIGN)
          goto BAD;
        ptr++;
        }

      /* Have found a syntactically correct group number or name, or *name.
      Only *MARK is currently recognized. */

      if (star)
        {
        if (PRIV(strcmp_c8)(name, STRING_MARK) == 0)
          {
          PCRE2_SPTR mark = pcre2_get_mark(match_data);
          if (mark != NULL)
            {
            /* Peek backwards one code unit to obtain the length of the mark.
            It can (theoretically) contain an embedded NUL. */
            fraglength = mark[-1];
            if (forcecase.to_case != PCRE2_SUBSTITUTE_CASE_NONE &&
                substitute_case_callout == NULL)
              CHECKCASECPY_DEFAULT(mark, fraglength);
            else
              CHECKMEMCPY(mark, fraglength);
            }
          }
        else goto BAD;
        }

      /* Substitute the contents of a group. We don't use substring_copy
      functions any more, in order to support case forcing. */

      else
        {
        GROUP_SUBSTITUTE:
        /* Find a number for a named group. In case there are duplicate names,
        search for the first one that is set. If the name is not found when
        PCRE2_SUBSTITUTE_UNKNOWN_EMPTY is set, set the group number to a
        non-existent group. */

        if (group < 0)
          {
          PCRE2_SPTR first, last, entry;
          rc = pcre2_substring_nametable_scan(code, name, &first, &last);
          if (rc == PCRE2_ERROR_NOSUBSTRING &&
              (suboptions & PCRE2_SUBSTITUTE_UNKNOWN_UNSET) != 0)
            {
            group = code->top_bracket + 1;
            }
          else
            {
            if (rc < 0) goto PTREXIT;
            for (entry = first; entry <= last; entry += rc)
              {
              uint32_t ng = GET2(entry, 0);
              if (ng < ovector_count)
                {
                if (group < 0) group = ng;          /* First in ovector */
                if (ovector[ng*2] != PCRE2_UNSET)
                  {
                  group = ng;                       /* First that is set */
                  break;
                  }
                }
              }

            /* If group is still negative, it means we did not find a group
            that is in the ovector. Just set the first group. */

            if (group < 0) group = GET2(first, 0);
            }
          }

        /* We now have a group that is identified by number. Find the length of
        the captured string. If a group in a non-special substitution is unset
        when PCRE2_SUBSTITUTE_UNSET_EMPTY is set, substitute nothing. */

        rc = pcre2_substring_length_bynumber(match_data, group, &sublength);
        if (rc < 0)
          {
          if (rc == PCRE2_ERROR_NOSUBSTRING &&
              (suboptions & PCRE2_SUBSTITUTE_UNKNOWN_UNSET) != 0)
            {
            rc = PCRE2_ERROR_UNSET;
            }
          if (rc != PCRE2_ERROR_UNSET) goto PTREXIT;  /* Non-unset errors */
          if (special == 0)                           /* Plain substitution */
            {
            if ((suboptions & PCRE2_SUBSTITUTE_UNSET_EMPTY) != 0) continue;
            goto PTREXIT;                             /* Else error */
            }
          }

        /* If special is '+' we have a 'set' and possibly an 'unset' text,
        both of which are reprocessed when used. If special is '-' we have a
        default text for when the group is unset; it must be reprocessed. */

        if (special != 0)
          {
          if (special == CHAR_MINUS)
            {
            if (rc == 0) goto LITERAL_SUBSTITUTE;
            text2_start = text1_start;
            text2_end = text1_end;
            }

          if (ptrstackptr >= PTR_STACK_SIZE) goto BAD;
          ptrstack[ptrstackptr++] = ptr;
          ptrstack[ptrstackptr++] = repend;

          if (rc == 0)
            {
            ptr = text1_start;
            repend = text1_end;
            }
          else
            {
            ptr = text2_start;
            repend = text2_end;
            }
          continue;
          }

        /* Otherwise we have a literal substitution of a group's contents. */

        LITERAL_SUBSTITUTE:
        subptr = subject + ovector[group*2];
        subptrend = subject + ovector[group*2 + 1];

        /* Substitute a literal string, possibly forcing alphabetic case. */

        SUBPTR_SUBSTITUTE:
        if (forcecase.to_case != PCRE2_SUBSTITUTE_CASE_NONE &&
            substitute_case_callout == NULL)
          CHECKCASECPY_DEFAULT(subptr, subptrend - subptr);
        else
          CHECKMEMCPY(subptr, subptrend - subptr);
        }
      }   /* End of $ processing */

    /* Handle an escape sequence in extended mode. We can use check_escape()
    to process \Q, \E, \c, \o, \x and \ followed by non-alphanumerics, but
    the case-forcing escapes are not supported in pcre2_compile() so must be
    recognized here. */

    else if ((suboptions & PCRE2_SUBSTITUTE_EXTENDED) != 0 &&
              *ptr == CHAR_BACKSLASH)
      {
      int errorcode;
      case_state new_forcecase = { PCRE2_SUBSTITUTE_CASE_NONE, FALSE };

      if (ptr < repend - 1) switch (ptr[1])
        {
        case CHAR_L:
        new_forcecase.to_case = PCRE2_SUBSTITUTE_CASE_LOWER;
        new_forcecase.single_char = FALSE;
        ptr += 2;
        break;

        case CHAR_l:
        new_forcecase.to_case = PCRE2_SUBSTITUTE_CASE_LOWER;
        new_forcecase.single_char = TRUE;
        ptr += 2;
        if (ptr + 2 < repend && ptr[0] == CHAR_BACKSLASH && ptr[1] == CHAR_U)
          {
          /* Perl reverse-title-casing feature for \l\U */
          new_forcecase.to_case = PCRE2_SUBSTITUTE_CASE_REVERSE_TITLE_FIRST;
          new_forcecase.single_char = FALSE;
          ptr += 2;
          }
        break;

        case CHAR_U:
        new_forcecase.to_case = PCRE2_SUBSTITUTE_CASE_UPPER;
        new_forcecase.single_char = FALSE;
        ptr += 2;
        break;

        case CHAR_u:
        new_forcecase.to_case = PCRE2_SUBSTITUTE_CASE_TITLE_FIRST;
        new_forcecase.single_char = TRUE;
        ptr += 2;
        if (ptr + 2 < repend && ptr[0] == CHAR_BACKSLASH && ptr[1] == CHAR_L)
          {
          /* Perl title-casing feature for \u\L */
          new_forcecase.to_case = PCRE2_SUBSTITUTE_CASE_TITLE_FIRST;
          new_forcecase.single_char = FALSE;
          ptr += 2;
          }
        break;

        default:
        break;
        }

      if (new_forcecase.to_case != PCRE2_SUBSTITUTE_CASE_NONE)
        {
        SETFORCECASE:

        /* If the substitute_case_callout is unset, our case-forcing is done
        immediately. If there is a callout however, then its action is delayed
        until all the characters have been collected.

        Apply the callout now, before we set the new casing mode. */

        if (substitute_case_callout != NULL &&
            forcecase.to_case != PCRE2_SUBSTITUTE_CASE_NONE)
          DELAYEDFORCECASE();

        forcecase = new_forcecase;
        casestart_offset = buff_offset;
        casestart_extra_needed = extra_needed;
        continue;
        }

      ptr++;  /* Point after \ */
      rc = PRIV(check_escape)(&ptr, repend, &ch, &errorcode,
        code->overall_options, code->extra_options, code->top_bracket, FALSE, NULL);
      if (errorcode != 0) goto BADESCAPE;

      switch(rc)
        {
        case ESC_E:
        goto SETFORCECASE;

        case ESC_Q:
        escaped_literal = TRUE;
        continue;

        case 0:      /* Data character */
        case ESC_b:  /* \b is backspace in a substitution */
        case ESC_v:  /* \v is vertical tab in a substitution */

        if (rc == ESC_b) ch = CHAR_BS;
        if (rc == ESC_v) ch = CHAR_VT;

#ifdef SUPPORT_UNICODE
        if (utf) chlen = PRIV(ord2utf)(ch, temp); else
#endif
          {
          temp[0] = ch;
          chlen = 1;
          }

        if (forcecase.to_case != PCRE2_SUBSTITUTE_CASE_NONE &&
            substitute_case_callout == NULL)
          CHECKCASECPY_DEFAULT(temp, chlen);
        else
          CHECKMEMCPY(temp, chlen);
        continue;

        case ESC_g:
          {
          PCRE2_SIZE name_len;
          PCRE2_SPTR name_start;

          /* Parse the \g<name> form (\g<number> already handled by check_escape) */
          if (ptr >= repend || *ptr != CHAR_LESS_THAN_SIGN)
            goto BADESCAPE;
          ++ptr;

          name_start = ptr;
          if (!read_name_subst(&ptr, repend, utf, code->tables + ctypes_offset))
            goto BADESCAPE;
          name_len = ptr - name_start;

          if (ptr >= repend || *ptr != CHAR_GREATER_THAN_SIGN)
            goto BADESCAPE;
          ++ptr;

          special = 0;
          group = -1;
          memcpy(name, name_start, CU2BYTES(name_len));
          name[name_len] = 0;
          goto GROUP_SUBSTITUTE;
          }

        default:
        if (rc < 0)
          {
          special = 0;
          group = -rc - 1;
          goto GROUP_SUBSTITUTE;
          }
        goto BADESCAPE;
        }
      }   /* End of backslash processing */

    /* Handle a literal code unit */

    else
      {
      PCRE2_SPTR ch_start;

      LOADLITERAL:
      ch_start = ptr;
      GETCHARINCTEST(ch, ptr);    /* Get character value, increment pointer */
      (void) ch;

      if (forcecase.to_case != PCRE2_SUBSTITUTE_CASE_NONE &&
          substitute_case_callout == NULL)
        CHECKCASECPY_DEFAULT(ch_start, ptr - ch_start);
      else
        CHECKMEMCPY(ch_start, ptr - ch_start);
      } /* End handling a literal code unit */
    }   /* End of loop for scanning the replacement. */

  /* If the substitute_case_callout is unset, our case-forcing is done
  immediately. If there is a callout however, then its action is delayed
  until all the characters have been collected.

  We now clean up any trailing section of the replacement for which we deferred
  the case-forcing. */

  if (substitute_case_callout != NULL &&
      forcecase.to_case != PCRE2_SUBSTITUTE_CASE_NONE)
    DELAYEDFORCECASE();

  /* The replacement has been copied to the output, or its size has been
  remembered. Handle the callout if there is one. */

  if (mcontext != NULL && mcontext->substitute_callout != NULL)
    {
    /* If we an actual (non-simulated) replacement, do the callout. */

    if (!overflowed)
      {
      scb.subscount = subs;
      scb.output_offsets[1] = buff_offset;
      rc = mcontext->substitute_callout(&scb,
                                        mcontext->substitute_callout_data);

      /* A non-zero return means cancel this substitution. Instead, copy the
      matched string fragment. */

      if (rc != 0)
        {
        PCRE2_SIZE newlength = scb.output_offsets[1] - scb.output_offsets[0];
        PCRE2_SIZE oldlength = ovector[1] - ovector[0];

        buff_offset -= newlength;
        lengthleft += newlength;
        if (!replacement_only) CHECKMEMCPY(subject + ovector[0], oldlength);

        /* A negative return means do not do any more. */

        if (rc < 0) suboptions &= (~PCRE2_SUBSTITUTE_GLOBAL);
        }
      }

    /* In this interesting case, we cannot do the callout, so it's hard to
    estimate the required buffer size. What callers want is to be able to make
    two calls to pcre2_substitute(), once with PCRE2_SUBSTITUTE_OVERFLOW_LENGTH
    to discover the buffer size, and then a second and final call. Older
    versions of PCRE2 violated this assumption, by proceding as if the callout
    had returned zero - but on the second call to pcre2_substitute() it could
    return non-zero and then overflow the buffer again. Callers probably don't
    want to keep on looping to incrementally discover the buffer size. */

    else
      {
      PCRE2_SIZE newlength_buf = buff_offset - scb.output_offsets[0];
      PCRE2_SIZE newlength_extra = extra_needed - sub_start_extra_needed;
      PCRE2_SIZE newlength =
        (newlength_extra > ~(PCRE2_SIZE)0 - newlength_buf)?  /* Integer overflow */
        ~(PCRE2_SIZE)0 : newlength_buf + newlength_extra;    /* Cap the addition */
      PCRE2_SIZE oldlength = ovector[1] - ovector[0];

      /* Be pessimistic: request whichever buffer size is larger out of
      accepting or rejecting the substitution. */

      if (oldlength > newlength)
        {
        PCRE2_SIZE additional = oldlength - newlength;
        if (additional > ~(PCRE2_SIZE)0 - extra_needed)  /* Integer overflow */
          goto TOOLARGEREPLACE;
        extra_needed += additional;
        }

      /* Proceed as if the callout did not return a negative. A negative
      effectively rejects all future substitutions, but we want to examine them
      pessimistically. */
      }
    }

  /* Save the details of this match. See above for how this data is used. If we
  matched an empty string, do the magic for global matches. Update the start
  offset to point to the rest of the subject string. If we re-used an existing
  match for the first match, switch to the internal match data block. */

  ovecsave[0] = ovector[0];
  ovecsave[1] = ovector[1];
  ovecsave[2] = start_offset;

  goptions = (ovector[0] != ovector[1] || ovector[0] > start_offset)? 0 :
    PCRE2_ANCHORED|PCRE2_NOTEMPTY_ATSTART;
  start_offset = ovector[1];
  } while ((suboptions & PCRE2_SUBSTITUTE_GLOBAL) != 0);  /* Repeat "do" loop */

/* Copy the rest of the subject unless not required, and terminate the output
with a binary zero. */

if (!replacement_only)
  {
  fraglength = length - start_offset;
  CHECKMEMCPY(subject + start_offset, fraglength);
  }

temp[0] = 0;
CHECKMEMCPY(temp, 1);

/* If overflowed is set it means the PCRE2_SUBSTITUTE_OVERFLOW_LENGTH is set,
and matching has carried on after a full buffer, in order to compute the length
needed. Otherwise, an overflow generates an immediate error return. */

if (overflowed)
  {
  rc = PCRE2_ERROR_NOMEMORY;

  if (extra_needed > ~(PCRE2_SIZE)0 - buff_length)  /* Integer overflow */
    goto TOOLARGEREPLACE;
  *blength = buff_length + extra_needed;
  }

/* After a successful execution, return the number of substitutions and set the
length of buffer used, excluding the trailing zero. */

else
  {
  rc = subs;
  *blength = buff_offset - 1;
  }

EXIT:
if (internal_match_data != NULL) pcre2_match_data_free(internal_match_data);
  else match_data->rc = rc;
return rc;

NOROOM:
rc = PCRE2_ERROR_NOMEMORY;
goto EXIT;

CASEERROR:
rc = PCRE2_ERROR_REPLACECASE;
goto EXIT;

TOOLARGEREPLACE:
rc = PCRE2_ERROR_TOOLARGEREPLACE;
goto EXIT;

BAD:
rc = PCRE2_ERROR_BADREPLACEMENT;
goto PTREXIT;

BADESCAPE:
rc = PCRE2_ERROR_BADREPESCAPE;

PTREXIT:
*blength = (PCRE2_SIZE)(ptr - replacement);
goto EXIT;
}

/* End of pcre2_substitute.c */
