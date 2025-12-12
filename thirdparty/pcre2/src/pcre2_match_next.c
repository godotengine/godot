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


#include "pcre2_internal.h"



/* Advance the offset by one code unit, and return the new value.
It is only called when the offset is not at the end of the subject. */

static PCRE2_SIZE do_bumpalong(pcre2_match_data *match_data,
  PCRE2_SIZE offset)
{
PCRE2_SPTR subject = match_data->subject;
PCRE2_SIZE subject_length = match_data->subject_length;
#ifdef SUPPORT_UNICODE
BOOL utf = (match_data->code->overall_options & PCRE2_UTF) != 0;
#endif

/* Skip over CRLF as an atomic sequence, if CRLF is configured as a newline
sequence. */

if (subject[offset] == CHAR_CR && offset + 1 < subject_length &&
    subject[offset + 1] == CHAR_LF)
  {
  switch(match_data->code->newline_convention)
    {
    case PCRE2_NEWLINE_CRLF:
    case PCRE2_NEWLINE_ANY:
    case PCRE2_NEWLINE_ANYCRLF:
    return offset + 2;
    }
  }

/* Advance by one full character if in UTF mode. */

#ifdef SUPPORT_UNICODE
if (utf)
  {
  PCRE2_SPTR next = subject + offset + 1;
  PCRE2_SPTR subject_end = subject + subject_length;

  (void)subject_end; /* Suppress warning; 32-bit FORWARDCHARTEST ignores this */
  FORWARDCHARTEST(next, subject_end);
  return next - subject;
  }
#endif

return offset + 1;
}



/*************************************************
*                Advance the match               *
*************************************************/

PCRE2_EXP_DEFN int PCRE2_CALL_CONVENTION
pcre2_next_match(pcre2_match_data *match_data, PCRE2_SIZE *pstart_offset,
  uint32_t *poptions)
{
int rc = match_data->rc;
PCRE2_SIZE start_offset = match_data->start_offset;
PCRE2_SIZE *ovector = match_data->ovector;

/* Match error, or no match: no further iteration possible. In previous versions
of PCRE2, we recommended that clients use a strategy which involved retrying in
certain cases after PCRE2_ERROR_NOMATCH, but this is no longer required. */

if (rc < 0)
  return FALSE;

/* Match succeeded: get the start offset for the next match */

/* Although \K can affect the position of ovector[0], there are no ways to do
anything surprising with ovector[1], which must always be >= start_offset. */

PCRE2_ASSERT(ovector[1] >= start_offset);

/* Special handling for patterns which contain \K in a lookaround, which enables
the match start to be pushed back to before the starting search offset
(ovector[0] < start_offset) or after the match ends (ovector[0] > ovector[1]).
This is not a problem if ovector[1] > start_offset, because in this case, we can
just attempt the next match at ovector[1]: we are making progress, which is all
that we require.

However, if we have ovector[1] == start_offset, then we have a very rare case
which must be handled specially, because it's a non-empty match which
nonetheless fails to make progress through the subject. */

if (ovector[0] != start_offset && ovector[1] == start_offset)
  {
  /* If the match end is at the end of the subject, we are done. */

  if (start_offset >= match_data->subject_length)
    return FALSE;

  /* Otherwise, bump along by one code unit, and do a normal search. */

  *pstart_offset = do_bumpalong(match_data, ovector[1]);
  *poptions = 0;
  return TRUE;
  }

/* If the previous match was for an empty string, we are finished if we are at
the end of the subject. Otherwise, arrange to run another match at the same
point to see if a non-empty match can be found. */

if (ovector[0] == ovector[1])
  {
  /* If the match is at the end of the subject, we are done. */

  if (ovector[0] >= match_data->subject_length)
    return FALSE;

  /* Otherwise, continue at this exact same point, but we must set the flag
  which ensures that we don't return the exact same empty match again. */

  *pstart_offset = ovector[1];
  *poptions = PCRE2_NOTEMPTY_ATSTART;
  return TRUE;
  }

/* Finally, we must be in the happy state of a non-empty match, where the end of
the match is further on in the subject than start_offset, so we are easily able
to continue and make progress. */

*pstart_offset = ovector[1];
*poptions = 0;
return TRUE;
}

/* End of pcre2_match_next.c */
