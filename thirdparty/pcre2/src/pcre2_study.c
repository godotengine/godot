/*************************************************
*      Perl-Compatible Regular Expressions       *
*************************************************/

/* PCRE is a library of functions to support regular expressions whose syntax
and semantics are as close as possible to those of the Perl 5 language.

                       Written by Philip Hazel
     Original API code Copyright (c) 1997-2012 University of Cambridge
          New API code Copyright (c) 2016-2023 University of Cambridge

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

/* This module contains functions for scanning a compiled pattern and
collecting data (e.g. minimum matching length). */


#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "pcre2_internal.h"

/* The maximum remembered capturing brackets minimum. */

#define MAX_CACHE_BACKREF 128

/* Set a bit in the starting code unit bit map. */

#define SET_BIT(c) re->start_bitmap[(c)/8] |= (1u << ((c)&7))

/* Returns from set_start_bits() */

enum { SSB_FAIL, SSB_DONE, SSB_CONTINUE, SSB_UNKNOWN, SSB_TOODEEP };


/*************************************************
*   Find the minimum subject length for a group  *
*************************************************/

/* Scan a parenthesized group and compute the minimum length of subject that
is needed to match it. This is a lower bound; it does not mean there is a
string of that length that matches. In UTF mode, the result is in characters
rather than code units. The field in a compiled pattern for storing the minimum
length is 16-bits long (on the grounds that anything longer than that is
pathological), so we give up when we reach that amount. This also means that
integer overflow for really crazy patterns cannot happen.

Backreference minimum lengths are cached to speed up multiple references. This
function is called only when the highest back reference in the pattern is less
than or equal to MAX_CACHE_BACKREF, which is one less than the size of the
caching vector. The zeroth element contains the number of the highest set
value.

Arguments:
  re              compiled pattern block
  code            pointer to start of group (the bracket)
  startcode       pointer to start of the whole pattern's code
  utf             UTF flag
  recurses        chain of recurse_check to catch mutual recursion
  countptr        pointer to call count (to catch over complexity)
  backref_cache   vector for caching back references.

This function is no longer called when the pattern contains (*ACCEPT); however,
the old code for returning -1 is retained, just in case.

Returns:   the minimum length
           -1 \C in UTF-8 mode
              or (*ACCEPT)
              or pattern too complicated
           -2 internal error (missing capturing bracket)
           -3 internal error (opcode not listed)
*/

static int
find_minlength(const pcre2_real_code *re, PCRE2_SPTR code,
  PCRE2_SPTR startcode, BOOL utf, recurse_check *recurses, int *countptr,
  int *backref_cache)
{
int length = -1;
int branchlength = 0;
int prev_cap_recno = -1;
int prev_cap_d = 0;
int prev_recurse_recno = -1;
int prev_recurse_d = 0;
uint32_t once_fudge = 0;
BOOL had_recurse = FALSE;
BOOL dupcapused = (re->flags & PCRE2_DUPCAPUSED) != 0;
PCRE2_SPTR nextbranch = code + GET(code, 1);
PCRE2_UCHAR *cc = (PCRE2_UCHAR *)code + 1 + LINK_SIZE;
recurse_check this_recurse;

/* If this is a "could be empty" group, its minimum length is 0. */

if (*code >= OP_SBRA && *code <= OP_SCOND) return 0;

/* Skip over capturing bracket number */

if (*code == OP_CBRA || *code == OP_CBRAPOS) cc += IMM2_SIZE;

/* A large and/or complex regex can take too long to process. */

if ((*countptr)++ > 1000) return -1;

/* Scan along the opcodes for this branch. If we get to the end of the branch,
check the length against that of the other branches. If the accumulated length
passes 16-bits, reset to that value and skip the rest of the branch. */

for (;;)
  {
  int d, min, recno;
  PCRE2_UCHAR op, *cs, *ce;

  if (branchlength >= UINT16_MAX)
    {
    branchlength = UINT16_MAX;
    cc = (PCRE2_UCHAR *)nextbranch;
    }

  op = *cc;
  switch (op)
    {
    case OP_COND:
    case OP_SCOND:

    /* If there is only one branch in a condition, the implied branch has zero
    length, so we don't add anything. This covers the DEFINE "condition"
    automatically. If there are two branches we can treat it the same as any
    other non-capturing subpattern. */

    cs = cc + GET(cc, 1);
    if (*cs != OP_ALT)
      {
      cc = cs + 1 + LINK_SIZE;
      break;
      }
    goto PROCESS_NON_CAPTURE;

    case OP_BRA:
    /* There's a special case of OP_BRA, when it is wrapped round a repeated
    OP_RECURSE. We'd like to process the latter at this level so that
    remembering the value works for repeated cases. So we do nothing, but
    set a fudge value to skip over the OP_KET after the recurse. */

    if (cc[1+LINK_SIZE] == OP_RECURSE && cc[2*(1+LINK_SIZE)] == OP_KET)
      {
      once_fudge = 1 + LINK_SIZE;
      cc += 1 + LINK_SIZE;
      break;
      }
    /* Fall through */

    case OP_ONCE:
    case OP_SCRIPT_RUN:
    case OP_SBRA:
    case OP_BRAPOS:
    case OP_SBRAPOS:
    PROCESS_NON_CAPTURE:
    d = find_minlength(re, cc, startcode, utf, recurses, countptr,
      backref_cache);
    if (d < 0) return d;
    branchlength += d;
    do cc += GET(cc, 1); while (*cc == OP_ALT);
    cc += 1 + LINK_SIZE;
    break;

    /* To save time for repeated capturing subpatterns, we remember the
    length of the previous one. Unfortunately we can't do the same for
    the unnumbered ones above. Nor can we do this if (?| is present in the
    pattern because captures with the same number are not then identical. */

    case OP_CBRA:
    case OP_SCBRA:
    case OP_CBRAPOS:
    case OP_SCBRAPOS:
    recno = (int)GET2(cc, 1+LINK_SIZE);
    if (dupcapused || recno != prev_cap_recno)
      {
      prev_cap_recno = recno;
      prev_cap_d = find_minlength(re, cc, startcode, utf, recurses, countptr,
        backref_cache);
      if (prev_cap_d < 0) return prev_cap_d;
      }
    branchlength += prev_cap_d;
    do cc += GET(cc, 1); while (*cc == OP_ALT);
    cc += 1 + LINK_SIZE;
    break;

    /* ACCEPT makes things far too complicated; we have to give up. In fact,
    from 10.34 onwards, if a pattern contains (*ACCEPT), this function is not
    used. However, leave the code in place, just in case. */

    case OP_ACCEPT:
    case OP_ASSERT_ACCEPT:
    return -1;

    /* Reached end of a branch; if it's a ket it is the end of a nested
    call. If it's ALT it is an alternation in a nested call. If it is END it's
    the end of the outer call. All can be handled by the same code. If the
    length of any branch is zero, there is no need to scan any subsequent
    branches. */

    case OP_ALT:
    case OP_KET:
    case OP_KETRMAX:
    case OP_KETRMIN:
    case OP_KETRPOS:
    case OP_END:
    if (length < 0 || (!had_recurse && branchlength < length))
      length = branchlength;
    if (op != OP_ALT || length == 0) return length;
    nextbranch = cc + GET(cc, 1);
    cc += 1 + LINK_SIZE;
    branchlength = 0;
    had_recurse = FALSE;
    break;

    /* Skip over assertive subpatterns */

    case OP_ASSERT:
    case OP_ASSERT_NOT:
    case OP_ASSERTBACK:
    case OP_ASSERTBACK_NOT:
    case OP_ASSERT_NA:
    case OP_ASSERTBACK_NA:
    do cc += GET(cc, 1); while (*cc == OP_ALT);
    /* Fall through */

    /* Skip over things that don't match chars */

    case OP_REVERSE:
    case OP_VREVERSE:
    case OP_CREF:
    case OP_DNCREF:
    case OP_RREF:
    case OP_DNRREF:
    case OP_FALSE:
    case OP_TRUE:
    case OP_CALLOUT:
    case OP_SOD:
    case OP_SOM:
    case OP_EOD:
    case OP_EODN:
    case OP_CIRC:
    case OP_CIRCM:
    case OP_DOLL:
    case OP_DOLLM:
    case OP_NOT_WORD_BOUNDARY:
    case OP_WORD_BOUNDARY:
    case OP_NOT_UCP_WORD_BOUNDARY:
    case OP_UCP_WORD_BOUNDARY:
    cc += PRIV(OP_lengths)[*cc];
    break;

    case OP_CALLOUT_STR:
    cc += GET(cc, 1 + 2*LINK_SIZE);
    break;

    /* Skip over a subpattern that has a {0} or {0,x} quantifier */

    case OP_BRAZERO:
    case OP_BRAMINZERO:
    case OP_BRAPOSZERO:
    case OP_SKIPZERO:
    cc += PRIV(OP_lengths)[*cc];
    do cc += GET(cc, 1); while (*cc == OP_ALT);
    cc += 1 + LINK_SIZE;
    break;

    /* Handle literal characters and + repetitions */

    case OP_CHAR:
    case OP_CHARI:
    case OP_NOT:
    case OP_NOTI:
    case OP_PLUS:
    case OP_PLUSI:
    case OP_MINPLUS:
    case OP_MINPLUSI:
    case OP_POSPLUS:
    case OP_POSPLUSI:
    case OP_NOTPLUS:
    case OP_NOTPLUSI:
    case OP_NOTMINPLUS:
    case OP_NOTMINPLUSI:
    case OP_NOTPOSPLUS:
    case OP_NOTPOSPLUSI:
    branchlength++;
    cc += 2;
#ifdef SUPPORT_UNICODE
    if (utf && HAS_EXTRALEN(cc[-1])) cc += GET_EXTRALEN(cc[-1]);
#endif
    break;

    case OP_TYPEPLUS:
    case OP_TYPEMINPLUS:
    case OP_TYPEPOSPLUS:
    branchlength++;
    cc += (cc[1] == OP_PROP || cc[1] == OP_NOTPROP)? 4 : 2;
    break;

    /* Handle exact repetitions. The count is already in characters, but we
    may need to skip over a multibyte character in UTF mode.  */

    case OP_EXACT:
    case OP_EXACTI:
    case OP_NOTEXACT:
    case OP_NOTEXACTI:
    branchlength += GET2(cc,1);
    cc += 2 + IMM2_SIZE;
#ifdef SUPPORT_UNICODE
    if (utf && HAS_EXTRALEN(cc[-1])) cc += GET_EXTRALEN(cc[-1]);
#endif
    break;

    case OP_TYPEEXACT:
    branchlength += GET2(cc,1);
    cc += 2 + IMM2_SIZE + ((cc[1 + IMM2_SIZE] == OP_PROP
      || cc[1 + IMM2_SIZE] == OP_NOTPROP)? 2 : 0);
    break;

    /* Handle single-char non-literal matchers */

    case OP_PROP:
    case OP_NOTPROP:
    cc += 2;
    /* Fall through */

    case OP_NOT_DIGIT:
    case OP_DIGIT:
    case OP_NOT_WHITESPACE:
    case OP_WHITESPACE:
    case OP_NOT_WORDCHAR:
    case OP_WORDCHAR:
    case OP_ANY:
    case OP_ALLANY:
    case OP_EXTUNI:
    case OP_HSPACE:
    case OP_NOT_HSPACE:
    case OP_VSPACE:
    case OP_NOT_VSPACE:
    branchlength++;
    cc++;
    break;

    /* "Any newline" might match two characters, but it also might match just
    one. */

    case OP_ANYNL:
    branchlength += 1;
    cc++;
    break;

    /* The single-byte matcher means we can't proceed in UTF mode. (In
    non-UTF mode \C will actually be turned into OP_ALLANY, so won't ever
    appear, but leave the code, just in case.) */

    case OP_ANYBYTE:
#ifdef SUPPORT_UNICODE
    if (utf) return -1;
#endif
    branchlength++;
    cc++;
    break;

    /* For repeated character types, we have to test for \p and \P, which have
    an extra two bytes of parameters. */

    case OP_TYPESTAR:
    case OP_TYPEMINSTAR:
    case OP_TYPEQUERY:
    case OP_TYPEMINQUERY:
    case OP_TYPEPOSSTAR:
    case OP_TYPEPOSQUERY:
    if (cc[1] == OP_PROP || cc[1] == OP_NOTPROP) cc += 2;
    cc += PRIV(OP_lengths)[op];
    break;

    case OP_TYPEUPTO:
    case OP_TYPEMINUPTO:
    case OP_TYPEPOSUPTO:
    if (cc[1 + IMM2_SIZE] == OP_PROP
      || cc[1 + IMM2_SIZE] == OP_NOTPROP) cc += 2;
    cc += PRIV(OP_lengths)[op];
    break;

    /* Check a class for variable quantification */

    case OP_CLASS:
    case OP_NCLASS:
#ifdef SUPPORT_WIDE_CHARS
    case OP_XCLASS:
    /* The original code caused an unsigned overflow in 64 bit systems,
    so now we use a conditional statement. */
    if (op == OP_XCLASS)
      cc += GET(cc, 1);
    else
      cc += PRIV(OP_lengths)[OP_CLASS];
#else
    cc += PRIV(OP_lengths)[OP_CLASS];
#endif

    switch (*cc)
      {
      case OP_CRPLUS:
      case OP_CRMINPLUS:
      case OP_CRPOSPLUS:
      branchlength++;
      /* Fall through */

      case OP_CRSTAR:
      case OP_CRMINSTAR:
      case OP_CRQUERY:
      case OP_CRMINQUERY:
      case OP_CRPOSSTAR:
      case OP_CRPOSQUERY:
      cc++;
      break;

      case OP_CRRANGE:
      case OP_CRMINRANGE:
      case OP_CRPOSRANGE:
      branchlength += GET2(cc,1);
      cc += 1 + 2 * IMM2_SIZE;
      break;

      default:
      branchlength++;
      break;
      }
    break;

    /* Backreferences and subroutine calls (OP_RECURSE) are treated in the same
    way: we find the minimum length for the subpattern. A recursion
    (backreference or subroutine) causes an a flag to be set that causes the
    length of this branch to be ignored. The logic is that a recursion can only
    make sense if there is another alternative that stops the recursing. That
    will provide the minimum length (when no recursion happens).

    If PCRE2_MATCH_UNSET_BACKREF is set, a backreference to an unset bracket
    matches an empty string (by default it causes a matching failure), so in
    that case we must set the minimum length to zero.

    For backreferenes, if duplicate numbers are present in the pattern we check
    for a reference to a duplicate. If it is, we don't know which version will
    be referenced, so we have to set the minimum length to zero. */

    /* Duplicate named pattern back reference. */

    case OP_DNREF:
    case OP_DNREFI:
    if (!dupcapused && (re->overall_options & PCRE2_MATCH_UNSET_BACKREF) == 0)
      {
      int count = GET2(cc, 1+IMM2_SIZE);
      PCRE2_UCHAR *slot =
        (PCRE2_UCHAR *)((uint8_t *)re + sizeof(pcre2_real_code)) +
          GET2(cc, 1) * re->name_entry_size;

      d = INT_MAX;

      /* Scan all groups with the same name; find the shortest. */

      while (count-- > 0)
        {
        int dd, i;
        recno = GET2(slot, 0);

        if (recno <= backref_cache[0] && backref_cache[recno] >= 0)
          dd = backref_cache[recno];
        else
          {
          ce = cs = (PCRE2_UCHAR *)PRIV(find_bracket)(startcode, utf, recno);
          if (cs == NULL) return -2;
          do ce += GET(ce, 1); while (*ce == OP_ALT);

          dd = 0;
          if (!dupcapused ||
              (PCRE2_UCHAR *)PRIV(find_bracket)(ce, utf, recno) == NULL)
            {
            if (cc > cs && cc < ce)    /* Simple recursion */
              {
              had_recurse = TRUE;
              }
            else
              {
              recurse_check *r = recurses;
              for (r = recurses; r != NULL; r = r->prev)
                if (r->group == cs) break;
              if (r != NULL)           /* Mutual recursion */
                {
                had_recurse = TRUE;
                }
              else
                {
                this_recurse.prev = recurses;  /* No recursion */
                this_recurse.group = cs;
                dd = find_minlength(re, cs, startcode, utf, &this_recurse,
                  countptr, backref_cache);
                if (dd < 0) return dd;
                }
              }
            }

          backref_cache[recno] = dd;
          for (i = backref_cache[0] + 1; i < recno; i++) backref_cache[i] = -1;
          backref_cache[0] = recno;
          }

        if (dd < d) d = dd;
        if (d <= 0) break;    /* No point looking at any more */
        slot += re->name_entry_size;
        }
      }
    else d = 0;
    cc += 1 + 2*IMM2_SIZE;
    goto REPEAT_BACK_REFERENCE;

    /* Single back reference by number. References by name are converted to by
    number when there is no duplication. */

    case OP_REF:
    case OP_REFI:
    recno = GET2(cc, 1);
    if (recno <= backref_cache[0] && backref_cache[recno] >= 0)
      d = backref_cache[recno];
    else
      {
      int i;
      d = 0;

      if ((re->overall_options & PCRE2_MATCH_UNSET_BACKREF) == 0)
        {
        ce = cs = (PCRE2_UCHAR *)PRIV(find_bracket)(startcode, utf, recno);
        if (cs == NULL) return -2;
        do ce += GET(ce, 1); while (*ce == OP_ALT);

        if (!dupcapused ||
            (PCRE2_UCHAR *)PRIV(find_bracket)(ce, utf, recno) == NULL)
          {
          if (cc > cs && cc < ce)    /* Simple recursion */
            {
            had_recurse = TRUE;
            }
          else
            {
            recurse_check *r = recurses;
            for (r = recurses; r != NULL; r = r->prev) if (r->group == cs) break;
            if (r != NULL)           /* Mutual recursion */
              {
              had_recurse = TRUE;
              }
            else                     /* No recursion */
              {
              this_recurse.prev = recurses;
              this_recurse.group = cs;
              d = find_minlength(re, cs, startcode, utf, &this_recurse, countptr,
                backref_cache);
              if (d < 0) return d;
              }
            }
          }
        }

      backref_cache[recno] = d;
      for (i = backref_cache[0] + 1; i < recno; i++) backref_cache[i] = -1;
      backref_cache[0] = recno;
      }

    cc += 1 + IMM2_SIZE;

    /* Handle repeated back references */

    REPEAT_BACK_REFERENCE:
    switch (*cc)
      {
      case OP_CRSTAR:
      case OP_CRMINSTAR:
      case OP_CRQUERY:
      case OP_CRMINQUERY:
      case OP_CRPOSSTAR:
      case OP_CRPOSQUERY:
      min = 0;
      cc++;
      break;

      case OP_CRPLUS:
      case OP_CRMINPLUS:
      case OP_CRPOSPLUS:
      min = 1;
      cc++;
      break;

      case OP_CRRANGE:
      case OP_CRMINRANGE:
      case OP_CRPOSRANGE:
      min = GET2(cc, 1);
      cc += 1 + 2 * IMM2_SIZE;
      break;

      default:
      min = 1;
      break;
      }

     /* Take care not to overflow: (1) min and d are ints, so check that their
     product is not greater than INT_MAX. (2) branchlength is limited to
     UINT16_MAX (checked at the top of the loop). */

    if ((d > 0 && (INT_MAX/d) < min) || UINT16_MAX - branchlength < min*d)
      branchlength = UINT16_MAX;
    else branchlength += min * d;
    break;

    /* Recursion always refers to the first occurrence of a subpattern with a
    given number. Therefore, we can always make use of caching, even when the
    pattern contains multiple subpatterns with the same number. */

    case OP_RECURSE:
    cs = ce = (PCRE2_UCHAR *)startcode + GET(cc, 1);
    recno = GET2(cs, 1+LINK_SIZE);
    if (recno == prev_recurse_recno)
      {
      branchlength += prev_recurse_d;
      }
    else
      {
      do ce += GET(ce, 1); while (*ce == OP_ALT);
      if (cc > cs && cc < ce)    /* Simple recursion */
        had_recurse = TRUE;
      else
        {
        recurse_check *r = recurses;
        for (r = recurses; r != NULL; r = r->prev) if (r->group == cs) break;
        if (r != NULL)          /* Mutual recursion */
          had_recurse = TRUE;
        else
          {
          this_recurse.prev = recurses;
          this_recurse.group = cs;
          prev_recurse_d = find_minlength(re, cs, startcode, utf, &this_recurse,
            countptr, backref_cache);
          if (prev_recurse_d < 0) return prev_recurse_d;
          prev_recurse_recno = recno;
          branchlength += prev_recurse_d;
          }
        }
      }
    cc += 1 + LINK_SIZE + once_fudge;
    once_fudge = 0;
    break;

    /* Anything else does not or need not match a character. We can get the
    item's length from the table, but for those that can match zero occurrences
    of a character, we must take special action for UTF-8 characters. As it
    happens, the "NOT" versions of these opcodes are used at present only for
    ASCII characters, so they could be omitted from this list. However, in
    future that may change, so we include them here so as not to leave a
    gotcha for a future maintainer. */

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

    cc += PRIV(OP_lengths)[op];
#ifdef SUPPORT_UNICODE
    if (utf && HAS_EXTRALEN(cc[-1])) cc += GET_EXTRALEN(cc[-1]);
#endif
    break;

    /* Skip these, but we need to add in the name length. */

    case OP_MARK:
    case OP_COMMIT_ARG:
    case OP_PRUNE_ARG:
    case OP_SKIP_ARG:
    case OP_THEN_ARG:
    cc += PRIV(OP_lengths)[op] + cc[1];
    break;

    /* The remaining opcodes are just skipped over. */

    case OP_CLOSE:
    case OP_COMMIT:
    case OP_FAIL:
    case OP_PRUNE:
    case OP_SET_SOM:
    case OP_SKIP:
    case OP_THEN:
    cc += PRIV(OP_lengths)[op];
    break;

    /* This should not occur: we list all opcodes explicitly so that when
    new ones get added they are properly considered. */

    default:
    return -3;
    }
  }
/* Control never gets here */
}



/*************************************************
*      Set a bit and maybe its alternate case    *
*************************************************/

/* Given a character, set its first code unit's bit in the table, and also the
corresponding bit for the other version of a letter if we are caseless.

Arguments:
  re            points to the regex block
  p             points to the first code unit of the character
  caseless      TRUE if caseless
  utf           TRUE for UTF mode
  ucp           TRUE for UCP mode

Returns:        pointer after the character
*/

static PCRE2_SPTR
set_table_bit(pcre2_real_code *re, PCRE2_SPTR p, BOOL caseless, BOOL utf,
  BOOL ucp)
{
uint32_t c = *p++;   /* First code unit */

(void)utf;           /* Stop compiler warnings when UTF not supported */
(void)ucp;

/* In 16-bit and 32-bit modes, code units greater than 0xff set the bit for
0xff. */

#if PCRE2_CODE_UNIT_WIDTH != 8
if (c > 0xff) SET_BIT(0xff); else
#endif

SET_BIT(c);

/* In UTF-8 or UTF-16 mode, pick up the remaining code units in order to find
the end of the character, even when caseless. */

#ifdef SUPPORT_UNICODE
if (utf)
  {
#if PCRE2_CODE_UNIT_WIDTH == 8
  if (c >= 0xc0) GETUTF8INC(c, p);
#elif PCRE2_CODE_UNIT_WIDTH == 16
  if ((c & 0xfc00) == 0xd800) GETUTF16INC(c, p);
#endif
  }
#endif  /* SUPPORT_UNICODE */

/* If caseless, handle the other case of the character. */

if (caseless)
  {
#ifdef SUPPORT_UNICODE
  if (utf || ucp)
    {
    c = UCD_OTHERCASE(c);
#if PCRE2_CODE_UNIT_WIDTH == 8
    if (utf)
      {
      PCRE2_UCHAR buff[6];
      (void)PRIV(ord2utf)(c, buff);
      SET_BIT(buff[0]);
      }
    else if (c < 256) SET_BIT(c);
#else  /* 16-bit or 32-bit mode */
    if (c > 0xff) SET_BIT(0xff); else SET_BIT(c);
#endif
    }

  else
#endif  /* SUPPORT_UNICODE */

  /* Not UTF or UCP */

  if (MAX_255(c)) SET_BIT(re->tables[fcc_offset + c]);
  }

return p;
}



/*************************************************
*     Set bits for a positive character type     *
*************************************************/

/* This function sets starting bits for a character type. In UTF-8 mode, we can
only do a direct setting for bytes less than 128, as otherwise there can be
confusion with bytes in the middle of UTF-8 characters. In a "traditional"
environment, the tables will only recognize ASCII characters anyway, but in at
least one Windows environment, some higher bytes bits were set in the tables.
So we deal with that case by considering the UTF-8 encoding.

Arguments:
  re             the regex block
  cbit type      the type of character wanted
  table_limit    32 for non-UTF-8; 16 for UTF-8

Returns:         nothing
*/

static void
set_type_bits(pcre2_real_code *re, int cbit_type, unsigned int table_limit)
{
uint32_t c;
for (c = 0; c < table_limit; c++)
  re->start_bitmap[c] |= re->tables[c+cbits_offset+cbit_type];
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH == 8
if (table_limit == 32) return;
for (c = 128; c < 256; c++)
  {
  if ((re->tables[cbits_offset + c/8] & (1u << (c&7))) != 0)
    {
    PCRE2_UCHAR buff[6];
    (void)PRIV(ord2utf)(c, buff);
    SET_BIT(buff[0]);
    }
  }
#endif  /* UTF-8 */
}


/*************************************************
*     Set bits for a negative character type     *
*************************************************/

/* This function sets starting bits for a negative character type such as \D.
In UTF-8 mode, we can only do a direct setting for bytes less than 128, as
otherwise there can be confusion with bytes in the middle of UTF-8 characters.
Unlike in the positive case, where we can set appropriate starting bits for
specific high-valued UTF-8 characters, in this case we have to set the bits for
all high-valued characters. The lowest is 0xc2, but we overkill by starting at
0xc0 (192) for simplicity.

Arguments:
  re             the regex block
  cbit type      the type of character wanted
  table_limit    32 for non-UTF-8; 16 for UTF-8

Returns:         nothing
*/

static void
set_nottype_bits(pcre2_real_code *re, int cbit_type, unsigned int table_limit)
{
uint32_t c;
for (c = 0; c < table_limit; c++)
  re->start_bitmap[c] |= (uint8_t)(~(re->tables[c+cbits_offset+cbit_type]));
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH == 8
if (table_limit != 32) for (c = 24; c < 32; c++) re->start_bitmap[c] = 0xff;
#endif
}



/*************************************************
*      Create bitmap of starting code units      *
*************************************************/

/* This function scans a compiled unanchored expression recursively and
attempts to build a bitmap of the set of possible starting code units whose
values are less than 256. In 16-bit and 32-bit mode, values above 255 all cause
the 255 bit to be set. When calling set[_not]_type_bits() in UTF-8 (sic) mode
we pass a value of 16 rather than 32 as the final argument. (See comments in
those functions for the reason.)

The SSB_CONTINUE return is useful for parenthesized groups in patterns such as
(a*)b where the group provides some optional starting code units but scanning
must continue at the outer level to find at least one mandatory code unit. At
the outermost level, this function fails unless the result is SSB_DONE.

We restrict recursion (for nested groups) to 1000 to avoid stack overflow
issues.

Arguments:
  re           points to the compiled regex block
  code         points to an expression
  utf          TRUE if in UTF mode
  ucp          TRUE if in UCP mode
  depthptr     pointer to recurse depth

Returns:       SSB_FAIL     => Failed to find any starting code units
               SSB_DONE     => Found mandatory starting code units
               SSB_CONTINUE => Found optional starting code units
               SSB_UNKNOWN  => Hit an unrecognized opcode
               SSB_TOODEEP  => Recursion is too deep
*/

static int
set_start_bits(pcre2_real_code *re, PCRE2_SPTR code, BOOL utf, BOOL ucp,
  int *depthptr)
{
uint32_t c;
int yield = SSB_DONE;

#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH == 8
int table_limit = utf? 16:32;
#else
int table_limit = 32;
#endif

*depthptr += 1;
if (*depthptr > 1000) return SSB_TOODEEP;

do
  {
  BOOL try_next = TRUE;
  PCRE2_SPTR tcode = code + 1 + LINK_SIZE;

  if (*code == OP_CBRA || *code == OP_SCBRA ||
      *code == OP_CBRAPOS || *code == OP_SCBRAPOS) tcode += IMM2_SIZE;

  while (try_next)    /* Loop for items in this branch */
    {
    int rc;
    PCRE2_SPTR ncode;
    uint8_t *classmap = NULL;
#ifdef SUPPORT_WIDE_CHARS
    PCRE2_UCHAR xclassflags;
#endif

    switch(*tcode)
      {
      /* If we reach something we don't understand, it means a new opcode has
      been created that hasn't been added to this function. Hopefully this
      problem will be discovered during testing. */

      default:
      return SSB_UNKNOWN;

      /* Fail for a valid opcode that implies no starting bits. */

      case OP_ACCEPT:
      case OP_ASSERT_ACCEPT:
      case OP_ALLANY:
      case OP_ANY:
      case OP_ANYBYTE:
      case OP_CIRCM:
      case OP_CLOSE:
      case OP_COMMIT:
      case OP_COMMIT_ARG:
      case OP_COND:
      case OP_CREF:
      case OP_FALSE:
      case OP_TRUE:
      case OP_DNCREF:
      case OP_DNREF:
      case OP_DNREFI:
      case OP_DNRREF:
      case OP_DOLL:
      case OP_DOLLM:
      case OP_END:
      case OP_EOD:
      case OP_EODN:
      case OP_EXTUNI:
      case OP_FAIL:
      case OP_MARK:
      case OP_NOT:
      case OP_NOTEXACT:
      case OP_NOTEXACTI:
      case OP_NOTI:
      case OP_NOTMINPLUS:
      case OP_NOTMINPLUSI:
      case OP_NOTMINQUERY:
      case OP_NOTMINQUERYI:
      case OP_NOTMINSTAR:
      case OP_NOTMINSTARI:
      case OP_NOTMINUPTO:
      case OP_NOTMINUPTOI:
      case OP_NOTPLUS:
      case OP_NOTPLUSI:
      case OP_NOTPOSPLUS:
      case OP_NOTPOSPLUSI:
      case OP_NOTPOSQUERY:
      case OP_NOTPOSQUERYI:
      case OP_NOTPOSSTAR:
      case OP_NOTPOSSTARI:
      case OP_NOTPOSUPTO:
      case OP_NOTPOSUPTOI:
      case OP_NOTPROP:
      case OP_NOTQUERY:
      case OP_NOTQUERYI:
      case OP_NOTSTAR:
      case OP_NOTSTARI:
      case OP_NOTUPTO:
      case OP_NOTUPTOI:
      case OP_NOT_HSPACE:
      case OP_NOT_VSPACE:
      case OP_PRUNE:
      case OP_PRUNE_ARG:
      case OP_RECURSE:
      case OP_REF:
      case OP_REFI:
      case OP_REVERSE:
      case OP_VREVERSE:
      case OP_RREF:
      case OP_SCOND:
      case OP_SET_SOM:
      case OP_SKIP:
      case OP_SKIP_ARG:
      case OP_SOD:
      case OP_SOM:
      case OP_THEN:
      case OP_THEN_ARG:
      return SSB_FAIL;

      /* OP_CIRC happens only at the start of an anchored branch (multiline ^
      uses OP_CIRCM). Skip over it. */

      case OP_CIRC:
      tcode += PRIV(OP_lengths)[OP_CIRC];
      break;

      /* A "real" property test implies no starting bits, but the fake property
      PT_CLIST identifies a list of characters. These lists are short, as they
      are used for characters with more than one "other case", so there is no
      point in recognizing them for OP_NOTPROP. */

      case OP_PROP:
      if (tcode[1] != PT_CLIST) return SSB_FAIL;
        {
        const uint32_t *p = PRIV(ucd_caseless_sets) + tcode[2];
        while ((c = *p++) < NOTACHAR)
          {
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH == 8
          if (utf)
            {
            PCRE2_UCHAR buff[6];
            (void)PRIV(ord2utf)(c, buff);
            c = buff[0];
            }
#endif
          if (c > 0xff) SET_BIT(0xff); else SET_BIT(c);
          }
        }
      try_next = FALSE;
      break;

      /* We can ignore word boundary tests. */

      case OP_WORD_BOUNDARY:
      case OP_NOT_WORD_BOUNDARY:
      case OP_UCP_WORD_BOUNDARY:
      case OP_NOT_UCP_WORD_BOUNDARY:
      tcode++;
      break;

      /* For a positive lookahead assertion, inspect what immediately follows,
      ignoring intermediate assertions and callouts. If the next item is one
      that sets a mandatory character, skip this assertion. Otherwise, treat it
      the same as other bracket groups. */

      case OP_ASSERT:
      case OP_ASSERT_NA:
      ncode = tcode + GET(tcode, 1);
      while (*ncode == OP_ALT) ncode += GET(ncode, 1);
      ncode += 1 + LINK_SIZE;

      /* Skip irrelevant items */

      for (BOOL done = FALSE; !done;)
        {
        switch (*ncode)
          {
          case OP_ASSERT:
          case OP_ASSERT_NOT:
          case OP_ASSERTBACK:
          case OP_ASSERTBACK_NOT:
          case OP_ASSERT_NA:
          case OP_ASSERTBACK_NA:
          ncode += GET(ncode, 1);
          while (*ncode == OP_ALT) ncode += GET(ncode, 1);
          ncode += 1 + LINK_SIZE;
          break;

          case OP_WORD_BOUNDARY:
          case OP_NOT_WORD_BOUNDARY:
          case OP_UCP_WORD_BOUNDARY:
          case OP_NOT_UCP_WORD_BOUNDARY:
          ncode++;
          break;

          case OP_CALLOUT:
          ncode += PRIV(OP_lengths)[OP_CALLOUT];
          break;

          case OP_CALLOUT_STR:
          ncode += GET(ncode, 1 + 2*LINK_SIZE);
          break;

          default:
          done = TRUE;
          break;
          }
        }

      /* Now check the next significant item. */

      switch(*ncode)
        {
        default:
        break;

        case OP_PROP:
        if (ncode[1] != PT_CLIST) break;
        /* Fall through */
        case OP_ANYNL:
        case OP_CHAR:
        case OP_CHARI:
        case OP_EXACT:
        case OP_EXACTI:
        case OP_HSPACE:
        case OP_MINPLUS:
        case OP_MINPLUSI:
        case OP_PLUS:
        case OP_PLUSI:
        case OP_POSPLUS:
        case OP_POSPLUSI:
        case OP_VSPACE:
        /* Note that these types will only be present in non-UCP mode. */
        case OP_DIGIT:
        case OP_NOT_DIGIT:
        case OP_WORDCHAR:
        case OP_NOT_WORDCHAR:
        case OP_WHITESPACE:
        case OP_NOT_WHITESPACE:
        tcode = ncode;
        continue;   /* With the following significant opcode */
        }
      /* Fall through */

      /* For a group bracket or a positive assertion without an immediately
      following mandatory setting, recurse to set bits from within the
      subpattern. If it can't find anything, we have to give up. If it finds
      some mandatory character(s), we are done for this branch. Otherwise,
      carry on scanning after the subpattern. */

      case OP_BRA:
      case OP_SBRA:
      case OP_CBRA:
      case OP_SCBRA:
      case OP_BRAPOS:
      case OP_SBRAPOS:
      case OP_CBRAPOS:
      case OP_SCBRAPOS:
      case OP_ONCE:
      case OP_SCRIPT_RUN:
      rc = set_start_bits(re, tcode, utf, ucp, depthptr);
      if (rc == SSB_DONE)
        {
        try_next = FALSE;
        }
      else if (rc == SSB_CONTINUE)
        {
        do tcode += GET(tcode, 1); while (*tcode == OP_ALT);
        tcode += 1 + LINK_SIZE;
        }
      else return rc;   /* FAIL, UNKNOWN, or TOODEEP */
      break;

      /* If we hit ALT or KET, it means we haven't found anything mandatory in
      this branch, though we might have found something optional. For ALT, we
      continue with the next alternative, but we have to arrange that the final
      result from subpattern is SSB_CONTINUE rather than SSB_DONE. For KET,
      return SSB_CONTINUE: if this is the top level, that indicates failure,
      but after a nested subpattern, it causes scanning to continue. */

      case OP_ALT:
      yield = SSB_CONTINUE;
      try_next = FALSE;
      break;

      case OP_KET:
      case OP_KETRMAX:
      case OP_KETRMIN:
      case OP_KETRPOS:
      return SSB_CONTINUE;

      /* Skip over callout */

      case OP_CALLOUT:
      tcode += PRIV(OP_lengths)[OP_CALLOUT];
      break;

      case OP_CALLOUT_STR:
      tcode += GET(tcode, 1 + 2*LINK_SIZE);
      break;

      /* Skip over lookbehind and negative lookahead assertions */

      case OP_ASSERT_NOT:
      case OP_ASSERTBACK:
      case OP_ASSERTBACK_NOT:
      case OP_ASSERTBACK_NA:
      do tcode += GET(tcode, 1); while (*tcode == OP_ALT);
      tcode += 1 + LINK_SIZE;
      break;

      /* BRAZERO does the bracket, but carries on. */

      case OP_BRAZERO:
      case OP_BRAMINZERO:
      case OP_BRAPOSZERO:
      rc = set_start_bits(re, ++tcode, utf, ucp, depthptr);
      if (rc == SSB_FAIL || rc == SSB_UNKNOWN || rc == SSB_TOODEEP) return rc;
      do tcode += GET(tcode,1); while (*tcode == OP_ALT);
      tcode += 1 + LINK_SIZE;
      break;

      /* SKIPZERO skips the bracket. */

      case OP_SKIPZERO:
      tcode++;
      do tcode += GET(tcode,1); while (*tcode == OP_ALT);
      tcode += 1 + LINK_SIZE;
      break;

      /* Single-char * or ? sets the bit and tries the next item */

      case OP_STAR:
      case OP_MINSTAR:
      case OP_POSSTAR:
      case OP_QUERY:
      case OP_MINQUERY:
      case OP_POSQUERY:
      tcode = set_table_bit(re, tcode + 1, FALSE, utf, ucp);
      break;

      case OP_STARI:
      case OP_MINSTARI:
      case OP_POSSTARI:
      case OP_QUERYI:
      case OP_MINQUERYI:
      case OP_POSQUERYI:
      tcode = set_table_bit(re, tcode + 1, TRUE, utf, ucp);
      break;

      /* Single-char upto sets the bit and tries the next */

      case OP_UPTO:
      case OP_MINUPTO:
      case OP_POSUPTO:
      tcode = set_table_bit(re, tcode + 1 + IMM2_SIZE, FALSE, utf, ucp);
      break;

      case OP_UPTOI:
      case OP_MINUPTOI:
      case OP_POSUPTOI:
      tcode = set_table_bit(re, tcode + 1 + IMM2_SIZE, TRUE, utf, ucp);
      break;

      /* At least one single char sets the bit and stops */

      case OP_EXACT:
      tcode += IMM2_SIZE;
      /* Fall through */
      case OP_CHAR:
      case OP_PLUS:
      case OP_MINPLUS:
      case OP_POSPLUS:
      (void)set_table_bit(re, tcode + 1, FALSE, utf, ucp);
      try_next = FALSE;
      break;

      case OP_EXACTI:
      tcode += IMM2_SIZE;
      /* Fall through */
      case OP_CHARI:
      case OP_PLUSI:
      case OP_MINPLUSI:
      case OP_POSPLUSI:
      (void)set_table_bit(re, tcode + 1, TRUE, utf, ucp);
      try_next = FALSE;
      break;

      /* Special spacing and line-terminating items. These recognize specific
      lists of characters. The difference between VSPACE and ANYNL is that the
      latter can match the two-character CRLF sequence, but that is not
      relevant for finding the first character, so their code here is
      identical. */

      case OP_HSPACE:
      SET_BIT(CHAR_HT);
      SET_BIT(CHAR_SPACE);

      /* For the 16-bit and 32-bit libraries (which can never be EBCDIC), set
      the bits for 0xA0 and for code units >= 255, independently of UTF. */

#if PCRE2_CODE_UNIT_WIDTH != 8
      SET_BIT(0xA0);
      SET_BIT(0xFF);
#else
      /* For the 8-bit library in UTF-8 mode, set the bits for the first code
      units of horizontal space characters. */

#ifdef SUPPORT_UNICODE
      if (utf)
        {
        SET_BIT(0xC2);  /* For U+00A0 */
        SET_BIT(0xE1);  /* For U+1680, U+180E */
        SET_BIT(0xE2);  /* For U+2000 - U+200A, U+202F, U+205F */
        SET_BIT(0xE3);  /* For U+3000 */
        }
      else
#endif
      /* For the 8-bit library not in UTF-8 mode, set the bit for 0xA0, unless
      the code is EBCDIC. */
        {
#ifndef EBCDIC
        SET_BIT(0xA0);
#endif  /* Not EBCDIC */
        }
#endif  /* 8-bit support */

      try_next = FALSE;
      break;

      case OP_ANYNL:
      case OP_VSPACE:
      SET_BIT(CHAR_LF);
      SET_BIT(CHAR_VT);
      SET_BIT(CHAR_FF);
      SET_BIT(CHAR_CR);

      /* For the 16-bit and 32-bit libraries (which can never be EBCDIC), set
      the bits for NEL and for code units >= 255, independently of UTF. */

#if PCRE2_CODE_UNIT_WIDTH != 8
      SET_BIT(CHAR_NEL);
      SET_BIT(0xFF);
#else
      /* For the 8-bit library in UTF-8 mode, set the bits for the first code
      units of vertical space characters. */

#ifdef SUPPORT_UNICODE
      if (utf)
        {
        SET_BIT(0xC2);  /* For U+0085 (NEL) */
        SET_BIT(0xE2);  /* For U+2028, U+2029 */
        }
      else
#endif
      /* For the 8-bit library not in UTF-8 mode, set the bit for NEL. */
        {
        SET_BIT(CHAR_NEL);
        }
#endif  /* 8-bit support */

      try_next = FALSE;
      break;

      /* Single character types set the bits and stop. Note that if PCRE2_UCP
      is set, we do not see these opcodes because \d etc are converted to
      properties. Therefore, these apply in the case when only characters less
      than 256 are recognized to match the types. */

      case OP_NOT_DIGIT:
      set_nottype_bits(re, cbit_digit, table_limit);
      try_next = FALSE;
      break;

      case OP_DIGIT:
      set_type_bits(re, cbit_digit, table_limit);
      try_next = FALSE;
      break;

      case OP_NOT_WHITESPACE:
      set_nottype_bits(re, cbit_space, table_limit);
      try_next = FALSE;
      break;

      case OP_WHITESPACE:
      set_type_bits(re, cbit_space, table_limit);
      try_next = FALSE;
      break;

      case OP_NOT_WORDCHAR:
      set_nottype_bits(re, cbit_word, table_limit);
      try_next = FALSE;
      break;

      case OP_WORDCHAR:
      set_type_bits(re, cbit_word, table_limit);
      try_next = FALSE;
      break;

      /* One or more character type fudges the pointer and restarts, knowing
      it will hit a single character type and stop there. */

      case OP_TYPEPLUS:
      case OP_TYPEMINPLUS:
      case OP_TYPEPOSPLUS:
      tcode++;
      break;

      case OP_TYPEEXACT:
      tcode += 1 + IMM2_SIZE;
      break;

      /* Zero or more repeats of character types set the bits and then
      try again. */

      case OP_TYPEUPTO:
      case OP_TYPEMINUPTO:
      case OP_TYPEPOSUPTO:
      tcode += IMM2_SIZE;  /* Fall through */

      case OP_TYPESTAR:
      case OP_TYPEMINSTAR:
      case OP_TYPEPOSSTAR:
      case OP_TYPEQUERY:
      case OP_TYPEMINQUERY:
      case OP_TYPEPOSQUERY:
      switch(tcode[1])
        {
        default:
        case OP_ANY:
        case OP_ALLANY:
        return SSB_FAIL;

        case OP_HSPACE:
        SET_BIT(CHAR_HT);
        SET_BIT(CHAR_SPACE);

        /* For the 16-bit and 32-bit libraries (which can never be EBCDIC), set
        the bits for 0xA0 and for code units >= 255, independently of UTF. */

#if PCRE2_CODE_UNIT_WIDTH != 8
        SET_BIT(0xA0);
        SET_BIT(0xFF);
#else
        /* For the 8-bit library in UTF-8 mode, set the bits for the first code
        units of horizontal space characters. */

#ifdef SUPPORT_UNICODE
        if (utf)
          {
          SET_BIT(0xC2);  /* For U+00A0 */
          SET_BIT(0xE1);  /* For U+1680, U+180E */
          SET_BIT(0xE2);  /* For U+2000 - U+200A, U+202F, U+205F */
          SET_BIT(0xE3);  /* For U+3000 */
          }
        else
#endif
        /* For the 8-bit library not in UTF-8 mode, set the bit for 0xA0, unless
        the code is EBCDIC. */
          {
#ifndef EBCDIC
          SET_BIT(0xA0);
#endif  /* Not EBCDIC */
          }
#endif  /* 8-bit support */
        break;

        case OP_ANYNL:
        case OP_VSPACE:
        SET_BIT(CHAR_LF);
        SET_BIT(CHAR_VT);
        SET_BIT(CHAR_FF);
        SET_BIT(CHAR_CR);

        /* For the 16-bit and 32-bit libraries (which can never be EBCDIC), set
        the bits for NEL and for code units >= 255, independently of UTF. */

#if PCRE2_CODE_UNIT_WIDTH != 8
        SET_BIT(CHAR_NEL);
        SET_BIT(0xFF);
#else
        /* For the 8-bit library in UTF-8 mode, set the bits for the first code
        units of vertical space characters. */

#ifdef SUPPORT_UNICODE
        if (utf)
          {
          SET_BIT(0xC2);  /* For U+0085 (NEL) */
          SET_BIT(0xE2);  /* For U+2028, U+2029 */
          }
        else
#endif
        /* For the 8-bit library not in UTF-8 mode, set the bit for NEL. */
          {
          SET_BIT(CHAR_NEL);
          }
#endif  /* 8-bit support */
        break;

        case OP_NOT_DIGIT:
        set_nottype_bits(re, cbit_digit, table_limit);
        break;

        case OP_DIGIT:
        set_type_bits(re, cbit_digit, table_limit);
        break;

        case OP_NOT_WHITESPACE:
        set_nottype_bits(re, cbit_space, table_limit);
        break;

        case OP_WHITESPACE:
        set_type_bits(re, cbit_space, table_limit);
        break;

        case OP_NOT_WORDCHAR:
        set_nottype_bits(re, cbit_word, table_limit);
        break;

        case OP_WORDCHAR:
        set_type_bits(re, cbit_word, table_limit);
        break;
        }

      tcode += 2;
      break;

      /* Extended class: if there are any property checks, or if this is a
      negative XCLASS without a map, give up. If there are no property checks,
      there must be wide characters on the XCLASS list, because otherwise an
      XCLASS would not have been created. This means that code points >= 255
      are potential starters. In the UTF-8 case we can scan them and set bits
      for the relevant leading bytes. */

#ifdef SUPPORT_WIDE_CHARS
      case OP_XCLASS:
      xclassflags = tcode[1 + LINK_SIZE];
      if ((xclassflags & XCL_HASPROP) != 0 ||
          (xclassflags & (XCL_MAP|XCL_NOT)) == XCL_NOT)
        return SSB_FAIL;

      /* We have a positive XCLASS or a negative one without a map. Set up the
      map pointer if there is one, and fall through. */

      classmap = ((xclassflags & XCL_MAP) == 0)? NULL :
        (uint8_t *)(tcode + 1 + LINK_SIZE + 1);

      /* In UTF-8 mode, scan the character list and set bits for leading bytes,
      then jump to handle the map. */

#if PCRE2_CODE_UNIT_WIDTH == 8
      if (utf && (xclassflags & XCL_NOT) == 0)
        {
        PCRE2_UCHAR b, e;
        PCRE2_SPTR p = tcode + 1 + LINK_SIZE + 1 + ((classmap == NULL)? 0:32);
        tcode += GET(tcode, 1);

        for (;;) switch (*p++)
          {
          case XCL_SINGLE:
          b = *p++;
          while ((*p & 0xc0) == 0x80) p++;
          re->start_bitmap[b/8] |= (1u << (b&7));
          break;

          case XCL_RANGE:
          b = *p++;
          while ((*p & 0xc0) == 0x80) p++;
          e = *p++;
          while ((*p & 0xc0) == 0x80) p++;
          for (; b <= e; b++)
            re->start_bitmap[b/8] |= (1u << (b&7));
          break;

          case XCL_END:
          goto HANDLE_CLASSMAP;

          default:
          return SSB_UNKNOWN;   /* Internal error, should not occur */
          }
        }
#endif  /* SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH == 8 */
#endif  /* SUPPORT_WIDE_CHARS */

      /* It seems that the fall through comment must be outside the #ifdef if
      it is to avoid the gcc compiler warning. */

      /* Fall through */

      /* Enter here for a negative non-XCLASS. In the 8-bit library, if we are
      in UTF mode, any byte with a value >= 0xc4 is a potentially valid starter
      because it starts a character with a value > 255. In 8-bit non-UTF mode,
      there is no difference between CLASS and NCLASS. In all other wide
      character modes, set the 0xFF bit to indicate code units >= 255. */

      case OP_NCLASS:
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH == 8
      if (utf)
        {
        re->start_bitmap[24] |= 0xf0;            /* Bits for 0xc4 - 0xc8 */
        memset(re->start_bitmap+25, 0xff, 7);    /* Bits for 0xc9 - 0xff */
        }
#elif PCRE2_CODE_UNIT_WIDTH != 8
      SET_BIT(0xFF);                             /* For characters >= 255 */
#endif
      /* Fall through */

      /* Enter here for a positive non-XCLASS. If we have fallen through from
      an XCLASS, classmap will already be set; just advance the code pointer.
      Otherwise, set up classmap for a a non-XCLASS and advance past it. */

      case OP_CLASS:
      if (*tcode == OP_XCLASS) tcode += GET(tcode, 1); else
        {
        classmap = (uint8_t *)(++tcode);
        tcode += 32 / sizeof(PCRE2_UCHAR);
        }

      /* When wide characters are supported, classmap may be NULL. In UTF-8
      (sic) mode, the bits in a class bit map correspond to character values,
      not to byte values. However, the bit map we are constructing is for byte
      values. So we have to do a conversion for characters whose code point is
      greater than 127. In fact, there are only two possible starting bytes for
      characters in the range 128 - 255. */

#if defined SUPPORT_WIDE_CHARS && PCRE2_CODE_UNIT_WIDTH == 8
      HANDLE_CLASSMAP:
#endif
      if (classmap != NULL)
        {
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH == 8
        if (utf)
          {
          for (c = 0; c < 16; c++) re->start_bitmap[c] |= classmap[c];
          for (c = 128; c < 256; c++)
            {
            if ((classmap[c/8] & (1u << (c&7))) != 0)
              {
              int d = (c >> 6) | 0xc0;                 /* Set bit for this starter */
              re->start_bitmap[d/8] |= (1u << (d&7));  /* and then skip on to the */
              c = (c & 0xc0) + 0x40 - 1;               /* next relevant character. */
              }
            }
          }
        else
#endif
        /* In all modes except UTF-8, the two bit maps are compatible. */

          {
          for (c = 0; c < 32; c++) re->start_bitmap[c] |= classmap[c];
          }
        }

      /* Act on what follows the class. For a zero minimum repeat, continue;
      otherwise stop processing. */

      switch (*tcode)
        {
        case OP_CRSTAR:
        case OP_CRMINSTAR:
        case OP_CRQUERY:
        case OP_CRMINQUERY:
        case OP_CRPOSSTAR:
        case OP_CRPOSQUERY:
        tcode++;
        break;

        case OP_CRRANGE:
        case OP_CRMINRANGE:
        case OP_CRPOSRANGE:
        if (GET2(tcode, 1) == 0) tcode += 1 + 2 * IMM2_SIZE;
          else try_next = FALSE;
        break;

        default:
        try_next = FALSE;
        break;
        }
      break; /* End of class handling case */
      }      /* End of switch for opcodes */
    }        /* End of try_next loop */

  code += GET(code, 1);   /* Advance to next branch */
  }
while (*code == OP_ALT);

return yield;
}



/*************************************************
*          Study a compiled expression           *
*************************************************/

/* This function is handed a compiled expression that it must study to produce
information that will speed up the matching.

Argument:
  re       points to the compiled expression

Returns:   0 normally; non-zero should never normally occur
           1 unknown opcode in set_start_bits
           2 missing capturing bracket
           3 unknown opcode in find_minlength
*/

int
PRIV(study)(pcre2_real_code *re)
{
int count = 0;
PCRE2_UCHAR *code;
BOOL utf = (re->overall_options & PCRE2_UTF) != 0;
BOOL ucp = (re->overall_options & PCRE2_UCP) != 0;

/* Find start of compiled code */

code = (PCRE2_UCHAR *)((uint8_t *)re + sizeof(pcre2_real_code)) +
  re->name_entry_size * re->name_count;

/* For a pattern that has a first code unit, or a multiline pattern that
matches only at "line start", there is no point in seeking a list of starting
code units. */

if ((re->flags & (PCRE2_FIRSTSET|PCRE2_STARTLINE)) == 0)
  {
  int depth = 0;
  int rc = set_start_bits(re, code, utf, ucp, &depth);
  if (rc == SSB_UNKNOWN) return 1;

  /* If a list of starting code units was set up, scan the list to see if only
  one or two were listed. Having only one listed is rare because usually a
  single starting code unit will have been recognized and PCRE2_FIRSTSET set.
  If two are listed, see if they are caseless versions of the same character;
  if so we can replace the list with a caseless first code unit. This gives
  better performance and is plausibly worth doing for patterns such as [Ww]ord
  or (word|WORD). */

  if (rc == SSB_DONE)
    {
    int i;
    int a = -1;
    int b = -1;
    uint8_t *p = re->start_bitmap;
    uint32_t flags = PCRE2_FIRSTMAPSET;

    for (i = 0; i < 256; p++, i += 8)
      {
      uint8_t x = *p;
      if (x != 0)
        {
        int c;
        uint8_t y = x & (~x + 1);   /* Least significant bit */
        if (y != x) goto DONE;      /* More than one bit set */

        /* In the 16-bit and 32-bit libraries, the bit for 0xff means "0xff and
        all wide characters", so we cannot use it here. */

#if PCRE2_CODE_UNIT_WIDTH != 8
        if (i == 248 && x == 0x80) goto DONE;
#endif

        /* Compute the character value */

        c = i;
        switch (x)
          {
          case 1:   break;
          case 2:   c += 1; break;  case 4:  c += 2; break;
          case 8:   c += 3; break;  case 16: c += 4; break;
          case 32:  c += 5; break;  case 64: c += 6; break;
          case 128: c += 7; break;
          }

        /* c contains the code unit value, in the range 0-255. In 8-bit UTF
        mode, only values < 128 can be used. In all the other cases, c is a
        character value. */

#if PCRE2_CODE_UNIT_WIDTH == 8
        if (utf && c > 127) goto DONE;
#endif
        if (a < 0) a = c;   /* First one found, save in a */
        else if (b < 0)     /* Second one found */
          {
          int d = TABLE_GET((unsigned int)c, re->tables + fcc_offset, c);

#ifdef SUPPORT_UNICODE
          if (utf || ucp)
            {
            if (UCD_CASESET(c) != 0) goto DONE;     /* Multiple case set */
            if (c > 127) d = UCD_OTHERCASE(c);
            }
#endif  /* SUPPORT_UNICODE */

          if (d != a) goto DONE;   /* Not the other case of a */
          b = c;                   /* Save second in b */
          }
        else goto DONE;   /* More than two characters found */
        }
      }

    /* Replace the start code unit bits with a first code unit, but only if it
    is not the same as a required later code unit. This is because a search for
    a required code unit starts after an explicit first code unit, but at a
    code unit found from the bitmap. Patterns such as /a*a/ don't work
    if both the start unit and required unit are the same. */

    if (a >= 0 &&
        (
        (re->flags & PCRE2_LASTSET) == 0 ||
          (
          re->last_codeunit != (uint32_t)a &&
          (b < 0 || re->last_codeunit != (uint32_t)b)
          )
        ))
      {
      re->first_codeunit = a;
      flags = PCRE2_FIRSTSET;
      if (b >= 0) flags |= PCRE2_FIRSTCASELESS;
      }

    DONE:
    re->flags |= flags;
    }
  }

/* Find the minimum length of subject string. If the pattern can match an empty
string, the minimum length is already known. If the pattern contains (*ACCEPT)
all bets are off, and we don't even try to find a minimum length. If there are
more back references than the size of the vector we are going to cache them in,
do nothing. A pattern that complicated will probably take a long time to
analyze and may in any case turn out to be too complicated. Note that back
reference minima are held as 16-bit numbers. */

if ((re->flags & (PCRE2_MATCH_EMPTY|PCRE2_HASACCEPT)) == 0 &&
     re->top_backref <= MAX_CACHE_BACKREF)
  {
  int min;
  int backref_cache[MAX_CACHE_BACKREF+1];
  backref_cache[0] = 0;    /* Highest one that is set */
  min = find_minlength(re, code, code, utf, NULL, &count, backref_cache);
  switch(min)
    {
    case -1:  /* \C in UTF mode or over-complex regex */
    break;    /* Leave minlength unchanged (will be zero) */

    case -2:
    return 2; /* missing capturing bracket */

    case -3:
    return 3; /* unrecognized opcode */

    default:
    re->minlength = (min > UINT16_MAX)? UINT16_MAX : min;
    break;
    }
  }

return 0;
}

/* End of pcre2_study.c */
