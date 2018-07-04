/*************************************************
*      Perl-Compatible Regular Expressions       *
*************************************************/

/* PCRE is a library of functions to support regular expressions whose syntax
and semantics are as close as possible to those of the Perl 5 language.

                       Written by Philip Hazel
     Original API code Copyright (c) 1997-2012 University of Cambridge
          New API code Copyright (c) 2016-2018 University of Cambridge

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


/* This module contains the external function pcre2_dfa_match(), which is an
alternative matching function that uses a sort of DFA algorithm (not a true
FSM). This is NOT Perl-compatible, but it has advantages in certain
applications. */


/* NOTE ABOUT PERFORMANCE: A user of this function sent some code that improved
the performance of his patterns greatly. I could not use it as it stood, as it
was not thread safe, and made assumptions about pattern sizes. Also, it caused
test 7 to loop, and test 9 to crash with a segfault.

The issue is the check for duplicate states, which is done by a simple linear
search up the state list. (Grep for "duplicate" below to find the code.) For
many patterns, there will never be many states active at one time, so a simple
linear search is fine. In patterns that have many active states, it might be a
bottleneck. The suggested code used an indexing scheme to remember which states
had previously been used for each character, and avoided the linear search when
it knew there was no chance of a duplicate. This was implemented when adding
states to the state lists.

I wrote some thread-safe, not-limited code to try something similar at the time
of checking for duplicates (instead of when adding states), using index vectors
on the stack. It did give a 13% improvement with one specially constructed
pattern for certain subject strings, but on other strings and on many of the
simpler patterns in the test suite it did worse. The major problem, I think,
was the extra time to initialize the index. This had to be done for each call
of internal_dfa_match(). (The supplied patch used a static vector, initialized
only once - I suspect this was the cause of the problems with the tests.)

Overall, I concluded that the gains in some cases did not outweigh the losses
in others, so I abandoned this code. */


#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#define NLBLOCK mb             /* Block containing newline information */
#define PSSTART start_subject  /* Field containing processed string start */
#define PSEND   end_subject    /* Field containing processed string end */

#include "pcre2_internal.h"

#define PUBLIC_DFA_MATCH_OPTIONS \
  (PCRE2_ANCHORED|PCRE2_ENDANCHORED|PCRE2_NOTBOL|PCRE2_NOTEOL|PCRE2_NOTEMPTY| \
   PCRE2_NOTEMPTY_ATSTART|PCRE2_NO_UTF_CHECK|PCRE2_PARTIAL_HARD| \
   PCRE2_PARTIAL_SOFT|PCRE2_DFA_SHORTEST|PCRE2_DFA_RESTART)


/*************************************************
*      Code parameters and static tables         *
*************************************************/

/* These are offsets that are used to turn the OP_TYPESTAR and friends opcodes
into others, under special conditions. A gap of 20 between the blocks should be
enough. The resulting opcodes don't have to be less than 256 because they are
never stored, so we push them well clear of the normal opcodes. */

#define OP_PROP_EXTRA       300
#define OP_EXTUNI_EXTRA     320
#define OP_ANYNL_EXTRA      340
#define OP_HSPACE_EXTRA     360
#define OP_VSPACE_EXTRA     380


/* This table identifies those opcodes that are followed immediately by a
character that is to be tested in some way. This makes it possible to
centralize the loading of these characters. In the case of Type * etc, the
"character" is the opcode for \D, \d, \S, \s, \W, or \w, which will always be a
small value. Non-zero values in the table are the offsets from the opcode where
the character is to be found. ***NOTE*** If the start of this table is
modified, the three tables that follow must also be modified. */

static const uint8_t coptable[] = {
  0,                             /* End                                    */
  0, 0, 0, 0, 0,                 /* \A, \G, \K, \B, \b                     */
  0, 0, 0, 0, 0, 0,              /* \D, \d, \S, \s, \W, \w                 */
  0, 0, 0,                       /* Any, AllAny, Anybyte                   */
  0, 0,                          /* \P, \p                                 */
  0, 0, 0, 0, 0,                 /* \R, \H, \h, \V, \v                     */
  0,                             /* \X                                     */
  0, 0, 0, 0, 0, 0,              /* \Z, \z, $, $M, ^, ^M                   */
  1,                             /* Char                                   */
  1,                             /* Chari                                  */
  1,                             /* not                                    */
  1,                             /* noti                                   */
  /* Positive single-char repeats                                          */
  1, 1, 1, 1, 1, 1,              /* *, *?, +, +?, ?, ??                    */
  1+IMM2_SIZE, 1+IMM2_SIZE,      /* upto, minupto                          */
  1+IMM2_SIZE,                   /* exact                                  */
  1, 1, 1, 1+IMM2_SIZE,          /* *+, ++, ?+, upto+                      */
  1, 1, 1, 1, 1, 1,              /* *I, *?I, +I, +?I, ?I, ??I              */
  1+IMM2_SIZE, 1+IMM2_SIZE,      /* upto I, minupto I                      */
  1+IMM2_SIZE,                   /* exact I                                */
  1, 1, 1, 1+IMM2_SIZE,          /* *+I, ++I, ?+I, upto+I                  */
  /* Negative single-char repeats - only for chars < 256                   */
  1, 1, 1, 1, 1, 1,              /* NOT *, *?, +, +?, ?, ??                */
  1+IMM2_SIZE, 1+IMM2_SIZE,      /* NOT upto, minupto                      */
  1+IMM2_SIZE,                   /* NOT exact                              */
  1, 1, 1, 1+IMM2_SIZE,          /* NOT *+, ++, ?+, upto+                  */
  1, 1, 1, 1, 1, 1,              /* NOT *I, *?I, +I, +?I, ?I, ??I          */
  1+IMM2_SIZE, 1+IMM2_SIZE,      /* NOT upto I, minupto I                  */
  1+IMM2_SIZE,                   /* NOT exact I                            */
  1, 1, 1, 1+IMM2_SIZE,          /* NOT *+I, ++I, ?+I, upto+I              */
  /* Positive type repeats                                                 */
  1, 1, 1, 1, 1, 1,              /* Type *, *?, +, +?, ?, ??               */
  1+IMM2_SIZE, 1+IMM2_SIZE,      /* Type upto, minupto                     */
  1+IMM2_SIZE,                   /* Type exact                             */
  1, 1, 1, 1+IMM2_SIZE,          /* Type *+, ++, ?+, upto+                 */
  /* Character class & ref repeats                                         */
  0, 0, 0, 0, 0, 0,              /* *, *?, +, +?, ?, ??                    */
  0, 0,                          /* CRRANGE, CRMINRANGE                    */
  0, 0, 0, 0,                    /* Possessive *+, ++, ?+, CRPOSRANGE      */
  0,                             /* CLASS                                  */
  0,                             /* NCLASS                                 */
  0,                             /* XCLASS - variable length               */
  0,                             /* REF                                    */
  0,                             /* REFI                                   */
  0,                             /* DNREF                                  */
  0,                             /* DNREFI                                 */
  0,                             /* RECURSE                                */
  0,                             /* CALLOUT                                */
  0,                             /* CALLOUT_STR                            */
  0,                             /* Alt                                    */
  0,                             /* Ket                                    */
  0,                             /* KetRmax                                */
  0,                             /* KetRmin                                */
  0,                             /* KetRpos                                */
  0,                             /* Reverse                                */
  0,                             /* Assert                                 */
  0,                             /* Assert not                             */
  0,                             /* Assert behind                          */
  0,                             /* Assert behind not                      */
  0,                             /* ONCE                                   */
  0, 0, 0, 0, 0,                 /* BRA, BRAPOS, CBRA, CBRAPOS, COND       */
  0, 0, 0, 0, 0,                 /* SBRA, SBRAPOS, SCBRA, SCBRAPOS, SCOND  */
  0, 0,                          /* CREF, DNCREF                           */
  0, 0,                          /* RREF, DNRREF                           */
  0, 0,                          /* FALSE, TRUE                            */
  0, 0, 0,                       /* BRAZERO, BRAMINZERO, BRAPOSZERO        */
  0, 0, 0,                       /* MARK, PRUNE, PRUNE_ARG                 */
  0, 0, 0, 0,                    /* SKIP, SKIP_ARG, THEN, THEN_ARG         */
  0, 0, 0, 0,                    /* COMMIT, FAIL, ACCEPT, ASSERT_ACCEPT    */
  0, 0, 0                        /* CLOSE, SKIPZERO, DEFINE                */
};

/* This table identifies those opcodes that inspect a character. It is used to
remember the fact that a character could have been inspected when the end of
the subject is reached. ***NOTE*** If the start of this table is modified, the
two tables that follow must also be modified. */

static const uint8_t poptable[] = {
  0,                             /* End                                    */
  0, 0, 0, 1, 1,                 /* \A, \G, \K, \B, \b                     */
  1, 1, 1, 1, 1, 1,              /* \D, \d, \S, \s, \W, \w                 */
  1, 1, 1,                       /* Any, AllAny, Anybyte                   */
  1, 1,                          /* \P, \p                                 */
  1, 1, 1, 1, 1,                 /* \R, \H, \h, \V, \v                     */
  1,                             /* \X                                     */
  0, 0, 0, 0, 0, 0,              /* \Z, \z, $, $M, ^, ^M                   */
  1,                             /* Char                                   */
  1,                             /* Chari                                  */
  1,                             /* not                                    */
  1,                             /* noti                                   */
  /* Positive single-char repeats                                          */
  1, 1, 1, 1, 1, 1,              /* *, *?, +, +?, ?, ??                    */
  1, 1, 1,                       /* upto, minupto, exact                   */
  1, 1, 1, 1,                    /* *+, ++, ?+, upto+                      */
  1, 1, 1, 1, 1, 1,              /* *I, *?I, +I, +?I, ?I, ??I              */
  1, 1, 1,                       /* upto I, minupto I, exact I             */
  1, 1, 1, 1,                    /* *+I, ++I, ?+I, upto+I                  */
  /* Negative single-char repeats - only for chars < 256                   */
  1, 1, 1, 1, 1, 1,              /* NOT *, *?, +, +?, ?, ??                */
  1, 1, 1,                       /* NOT upto, minupto, exact               */
  1, 1, 1, 1,                    /* NOT *+, ++, ?+, upto+                  */
  1, 1, 1, 1, 1, 1,              /* NOT *I, *?I, +I, +?I, ?I, ??I          */
  1, 1, 1,                       /* NOT upto I, minupto I, exact I         */
  1, 1, 1, 1,                    /* NOT *+I, ++I, ?+I, upto+I              */
  /* Positive type repeats                                                 */
  1, 1, 1, 1, 1, 1,              /* Type *, *?, +, +?, ?, ??               */
  1, 1, 1,                       /* Type upto, minupto, exact              */
  1, 1, 1, 1,                    /* Type *+, ++, ?+, upto+                 */
  /* Character class & ref repeats                                         */
  1, 1, 1, 1, 1, 1,              /* *, *?, +, +?, ?, ??                    */
  1, 1,                          /* CRRANGE, CRMINRANGE                    */
  1, 1, 1, 1,                    /* Possessive *+, ++, ?+, CRPOSRANGE      */
  1,                             /* CLASS                                  */
  1,                             /* NCLASS                                 */
  1,                             /* XCLASS - variable length               */
  0,                             /* REF                                    */
  0,                             /* REFI                                   */
  0,                             /* DNREF                                  */
  0,                             /* DNREFI                                 */
  0,                             /* RECURSE                                */
  0,                             /* CALLOUT                                */
  0,                             /* CALLOUT_STR                            */
  0,                             /* Alt                                    */
  0,                             /* Ket                                    */
  0,                             /* KetRmax                                */
  0,                             /* KetRmin                                */
  0,                             /* KetRpos                                */
  0,                             /* Reverse                                */
  0,                             /* Assert                                 */
  0,                             /* Assert not                             */
  0,                             /* Assert behind                          */
  0,                             /* Assert behind not                      */
  0,                             /* ONCE                                   */
  0, 0, 0, 0, 0,                 /* BRA, BRAPOS, CBRA, CBRAPOS, COND       */
  0, 0, 0, 0, 0,                 /* SBRA, SBRAPOS, SCBRA, SCBRAPOS, SCOND  */
  0, 0,                          /* CREF, DNCREF                           */
  0, 0,                          /* RREF, DNRREF                           */
  0, 0,                          /* FALSE, TRUE                            */
  0, 0, 0,                       /* BRAZERO, BRAMINZERO, BRAPOSZERO        */
  0, 0, 0,                       /* MARK, PRUNE, PRUNE_ARG                 */
  0, 0, 0, 0,                    /* SKIP, SKIP_ARG, THEN, THEN_ARG         */
  0, 0, 0, 0,                    /* COMMIT, FAIL, ACCEPT, ASSERT_ACCEPT    */
  0, 0, 0                        /* CLOSE, SKIPZERO, DEFINE                */
};

/* These 2 tables allow for compact code for testing for \D, \d, \S, \s, \W,
and \w */

static const uint8_t toptable1[] = {
  0, 0, 0, 0, 0, 0,
  ctype_digit, ctype_digit,
  ctype_space, ctype_space,
  ctype_word,  ctype_word,
  0, 0                            /* OP_ANY, OP_ALLANY */
};

static const uint8_t toptable2[] = {
  0, 0, 0, 0, 0, 0,
  ctype_digit, 0,
  ctype_space, 0,
  ctype_word,  0,
  1, 1                            /* OP_ANY, OP_ALLANY */
};


/* Structure for holding data about a particular state, which is in effect the
current data for an active path through the match tree. It must consist
entirely of ints because the working vector we are passed, and which we put
these structures in, is a vector of ints. */

typedef struct stateblock {
  int offset;                     /* Offset to opcode (-ve has meaning) */
  int count;                      /* Count for repeats */
  int data;                       /* Some use extra data */
} stateblock;

#define INTS_PER_STATEBLOCK  (int)(sizeof(stateblock)/sizeof(int))



/*************************************************
*               Process a callout                *
*************************************************/

/* This function is called to perform a callout.

Arguments:
  code              current code pointer
  offsets           points to current capture offsets
  current_subject   start of current subject match
  ptr               current position in subject
  mb                the match block
  extracode         extra code offset when called from condition
  lengthptr         where to return the callout length

Returns:            the return from the callout
*/

static int
do_callout(PCRE2_SPTR code, PCRE2_SIZE *offsets, PCRE2_SPTR current_subject,
  PCRE2_SPTR ptr, dfa_match_block *mb, PCRE2_SIZE extracode,
  PCRE2_SIZE *lengthptr)
{
pcre2_callout_block *cb = mb->cb;

*lengthptr = (code[extracode] == OP_CALLOUT)?
  (PCRE2_SIZE)PRIV(OP_lengths)[OP_CALLOUT] :
  (PCRE2_SIZE)GET(code, 1 + 2*LINK_SIZE + extracode);

if (mb->callout == NULL) return 0;    /* No callout provided */

/* Fixed fields in the callout block are set once and for all at the start of
matching. */

cb->offset_vector    = offsets;
cb->start_match      = (PCRE2_SIZE)(current_subject - mb->start_subject);
cb->current_position = (PCRE2_SIZE)(ptr - mb->start_subject);
cb->pattern_position = GET(code, 1 + extracode);
cb->next_item_length = GET(code, 1 + LINK_SIZE + extracode);

if (code[extracode] == OP_CALLOUT)
  {
  cb->callout_number = code[1 + 2*LINK_SIZE + extracode];
  cb->callout_string_offset = 0;
  cb->callout_string = NULL;
  cb->callout_string_length = 0;
  }
else
  {
  cb->callout_number = 0;
  cb->callout_string_offset = GET(code, 1 + 3*LINK_SIZE + extracode);
  cb->callout_string = code + (1 + 4*LINK_SIZE + extracode) + 1;
  cb->callout_string_length = *lengthptr - (1 + 4*LINK_SIZE) - 2;
  }

return (mb->callout)(cb, mb->callout_data);
}



/*************************************************
*     Match a Regular Expression - DFA engine    *
*************************************************/

/* This internal function applies a compiled pattern to a subject string,
starting at a given point, using a DFA engine. This function is called from the
external one, possibly multiple times if the pattern is not anchored. The
function calls itself recursively for some kinds of subpattern.

Arguments:
  mb                the match_data block with fixed information
  this_start_code   the opening bracket of this subexpression's code
  current_subject   where we currently are in the subject string
  start_offset      start offset in the subject string
  offsets           vector to contain the matching string offsets
  offsetcount       size of same
  workspace         vector of workspace
  wscount           size of same
  rlevel            function call recursion level

Returns:            > 0 => number of match offset pairs placed in offsets
                    = 0 => offsets overflowed; longest matches are present
                     -1 => failed to match
                   < -1 => some kind of unexpected problem

The following macros are used for adding states to the two state vectors (one
for the current character, one for the following character). */

#define ADD_ACTIVE(x,y) \
  if (active_count++ < wscount) \
    { \
    next_active_state->offset = (x); \
    next_active_state->count  = (y); \
    next_active_state++; \
    } \
  else return PCRE2_ERROR_DFA_WSSIZE

#define ADD_ACTIVE_DATA(x,y,z) \
  if (active_count++ < wscount) \
    { \
    next_active_state->offset = (x); \
    next_active_state->count  = (y); \
    next_active_state->data   = (z); \
    next_active_state++; \
    } \
  else return PCRE2_ERROR_DFA_WSSIZE

#define ADD_NEW(x,y) \
  if (new_count++ < wscount) \
    { \
    next_new_state->offset = (x); \
    next_new_state->count  = (y); \
    next_new_state++; \
    } \
  else return PCRE2_ERROR_DFA_WSSIZE

#define ADD_NEW_DATA(x,y,z) \
  if (new_count++ < wscount) \
    { \
    next_new_state->offset = (x); \
    next_new_state->count  = (y); \
    next_new_state->data   = (z); \
    next_new_state++; \
    } \
  else return PCRE2_ERROR_DFA_WSSIZE

/* And now, here is the code */

static int
internal_dfa_match(
  dfa_match_block *mb,
  PCRE2_SPTR this_start_code,
  PCRE2_SPTR current_subject,
  PCRE2_SIZE start_offset,
  PCRE2_SIZE *offsets,
  uint32_t offsetcount,
  int *workspace,
  int wscount,
  uint32_t  rlevel)
{
stateblock *active_states, *new_states, *temp_states;
stateblock *next_active_state, *next_new_state;
const uint8_t *ctypes, *lcc, *fcc;
PCRE2_SPTR ptr;
PCRE2_SPTR end_code;
dfa_recursion_info new_recursive;
int active_count, new_count, match_count;

/* Some fields in the mb block are frequently referenced, so we load them into
independent variables in the hope that this will perform better. */

PCRE2_SPTR start_subject = mb->start_subject;
PCRE2_SPTR end_subject = mb->end_subject;
PCRE2_SPTR start_code = mb->start_code;

#ifdef SUPPORT_UNICODE
BOOL utf = (mb->poptions & PCRE2_UTF) != 0;
#else
BOOL utf = FALSE;
#endif

BOOL reset_could_continue = FALSE;

if (mb->match_call_count++ >= mb->match_limit) return PCRE2_ERROR_MATCHLIMIT;
if (rlevel++ > mb->match_limit_depth) return PCRE2_ERROR_DEPTHLIMIT;
offsetcount &= (uint32_t)(-2);  /* Round down */

wscount -= 2;
wscount = (wscount - (wscount % (INTS_PER_STATEBLOCK * 2))) /
          (2 * INTS_PER_STATEBLOCK);

ctypes = mb->tables + ctypes_offset;
lcc = mb->tables + lcc_offset;
fcc = mb->tables + fcc_offset;

match_count = PCRE2_ERROR_NOMATCH;   /* A negative number */

active_states = (stateblock *)(workspace + 2);
next_new_state = new_states = active_states + wscount;
new_count = 0;

/* The first thing in any (sub) pattern is a bracket of some sort. Push all
the alternative states onto the list, and find out where the end is. This
makes is possible to use this function recursively, when we want to stop at a
matching internal ket rather than at the end.

If we are dealing with a backward assertion we have to find out the maximum
amount to move back, and set up each alternative appropriately. */

if (*this_start_code == OP_ASSERTBACK || *this_start_code == OP_ASSERTBACK_NOT)
  {
  size_t max_back = 0;
  size_t gone_back;

  end_code = this_start_code;
  do
    {
    size_t back = (size_t)GET(end_code, 2+LINK_SIZE);
    if (back > max_back) max_back = back;
    end_code += GET(end_code, 1);
    }
  while (*end_code == OP_ALT);

  /* If we can't go back the amount required for the longest lookbehind
  pattern, go back as far as we can; some alternatives may still be viable. */

#ifdef SUPPORT_UNICODE
  /* In character mode we have to step back character by character */

  if (utf)
    {
    for (gone_back = 0; gone_back < max_back; gone_back++)
      {
      if (current_subject <= start_subject) break;
      current_subject--;
      ACROSSCHAR(current_subject > start_subject, current_subject,
        current_subject--);
      }
    }
  else
#endif

  /* In byte-mode we can do this quickly. */

    {
    size_t current_offset = (size_t)(current_subject - start_subject);
    gone_back = (current_offset < max_back)? current_offset : max_back;
    current_subject -= gone_back;
    }

  /* Save the earliest consulted character */

  if (current_subject < mb->start_used_ptr)
    mb->start_used_ptr = current_subject;

  /* Now we can process the individual branches. There will be an OP_REVERSE at
  the start of each branch, except when the length of the branch is zero. */

  end_code = this_start_code;
  do
    {
    uint32_t revlen = (end_code[1+LINK_SIZE] == OP_REVERSE)? 1 + LINK_SIZE : 0;
    size_t back = (revlen == 0)? 0 : (size_t)GET(end_code, 2+LINK_SIZE);
    if (back <= gone_back)
      {
      int bstate = (int)(end_code - start_code + 1 + LINK_SIZE + revlen);
      ADD_NEW_DATA(-bstate, 0, (int)(gone_back - back));
      }
    end_code += GET(end_code, 1);
    }
  while (*end_code == OP_ALT);
 }

/* This is the code for a "normal" subpattern (not a backward assertion). The
start of a whole pattern is always one of these. If we are at the top level,
we may be asked to restart matching from the same point that we reached for a
previous partial match. We still have to scan through the top-level branches to
find the end state. */

else
  {
  end_code = this_start_code;

  /* Restarting */

  if (rlevel == 1 && (mb->moptions & PCRE2_DFA_RESTART) != 0)
    {
    do { end_code += GET(end_code, 1); } while (*end_code == OP_ALT);
    new_count = workspace[1];
    if (!workspace[0])
      memcpy(new_states, active_states, (size_t)new_count * sizeof(stateblock));
    }

  /* Not restarting */

  else
    {
    int length = 1 + LINK_SIZE +
      ((*this_start_code == OP_CBRA || *this_start_code == OP_SCBRA ||
        *this_start_code == OP_CBRAPOS || *this_start_code == OP_SCBRAPOS)
        ? IMM2_SIZE:0);
    do
      {
      ADD_NEW((int)(end_code - start_code + length), 0);
      end_code += GET(end_code, 1);
      length = 1 + LINK_SIZE;
      }
    while (*end_code == OP_ALT);
    }
  }

workspace[0] = 0;    /* Bit indicating which vector is current */

/* Loop for scanning the subject */

ptr = current_subject;
for (;;)
  {
  int i, j;
  int clen, dlen;
  uint32_t c, d;
  int forced_fail = 0;
  BOOL partial_newline = FALSE;
  BOOL could_continue = reset_could_continue;
  reset_could_continue = FALSE;

  if (ptr > mb->last_used_ptr) mb->last_used_ptr = ptr;

  /* Make the new state list into the active state list and empty the
  new state list. */

  temp_states = active_states;
  active_states = new_states;
  new_states = temp_states;
  active_count = new_count;
  new_count = 0;

  workspace[0] ^= 1;              /* Remember for the restarting feature */
  workspace[1] = active_count;

  /* Set the pointers for adding new states */

  next_active_state = active_states + active_count;
  next_new_state = new_states;

  /* Load the current character from the subject outside the loop, as many
  different states may want to look at it, and we assume that at least one
  will. */

  if (ptr < end_subject)
    {
    clen = 1;        /* Number of data items in the character */
#ifdef SUPPORT_UNICODE
    GETCHARLENTEST(c, ptr, clen);
#else
    c = *ptr;
#endif  /* SUPPORT_UNICODE */
    }
  else
    {
    clen = 0;        /* This indicates the end of the subject */
    c = NOTACHAR;    /* This value should never actually be used */
    }

  /* Scan up the active states and act on each one. The result of an action
  may be to add more states to the currently active list (e.g. on hitting a
  parenthesis) or it may be to put states on the new list, for considering
  when we move the character pointer on. */

  for (i = 0; i < active_count; i++)
    {
    stateblock *current_state = active_states + i;
    BOOL caseless = FALSE;
    PCRE2_SPTR code;
    uint32_t codevalue;
    int state_offset = current_state->offset;
    int rrc;
    int count;

    /* A negative offset is a special case meaning "hold off going to this
    (negated) state until the number of characters in the data field have
    been skipped". If the could_continue flag was passed over from a previous
    state, arrange for it to passed on. */

    if (state_offset < 0)
      {
      if (current_state->data > 0)
        {
        ADD_NEW_DATA(state_offset, current_state->count,
          current_state->data - 1);
        if (could_continue) reset_could_continue = TRUE;
        continue;
        }
      else
        {
        current_state->offset = state_offset = -state_offset;
        }
      }

    /* Check for a duplicate state with the same count, and skip if found.
    See the note at the head of this module about the possibility of improving
    performance here. */

    for (j = 0; j < i; j++)
      {
      if (active_states[j].offset == state_offset &&
          active_states[j].count == current_state->count)
        goto NEXT_ACTIVE_STATE;
      }

    /* The state offset is the offset to the opcode */

    code = start_code + state_offset;
    codevalue = *code;

    /* If this opcode inspects a character, but we are at the end of the
    subject, remember the fact for use when testing for a partial match. */

    if (clen == 0 && poptable[codevalue] != 0)
      could_continue = TRUE;

    /* If this opcode is followed by an inline character, load it. It is
    tempting to test for the presence of a subject character here, but that
    is wrong, because sometimes zero repetitions of the subject are
    permitted.

    We also use this mechanism for opcodes such as OP_TYPEPLUS that take an
    argument that is not a data character - but is always one byte long because
    the values are small. We have to take special action to deal with  \P, \p,
    \H, \h, \V, \v and \X in this case. To keep the other cases fast, convert
    these ones to new opcodes. */

    if (coptable[codevalue] > 0)
      {
      dlen = 1;
#ifdef SUPPORT_UNICODE
      if (utf) { GETCHARLEN(d, (code + coptable[codevalue]), dlen); } else
#endif  /* SUPPORT_UNICODE */
      d = code[coptable[codevalue]];
      if (codevalue >= OP_TYPESTAR)
        {
        switch(d)
          {
          case OP_ANYBYTE: return PCRE2_ERROR_DFA_UITEM;
          case OP_NOTPROP:
          case OP_PROP: codevalue += OP_PROP_EXTRA; break;
          case OP_ANYNL: codevalue += OP_ANYNL_EXTRA; break;
          case OP_EXTUNI: codevalue += OP_EXTUNI_EXTRA; break;
          case OP_NOT_HSPACE:
          case OP_HSPACE: codevalue += OP_HSPACE_EXTRA; break;
          case OP_NOT_VSPACE:
          case OP_VSPACE: codevalue += OP_VSPACE_EXTRA; break;
          default: break;
          }
        }
      }
    else
      {
      dlen = 0;         /* Not strictly necessary, but compilers moan */
      d = NOTACHAR;     /* if these variables are not set. */
      }


    /* Now process the individual opcodes */

    switch (codevalue)
      {
/* ========================================================================== */
      /* These cases are never obeyed. This is a fudge that causes a compile-
      time error if the vectors coptable or poptable, which are indexed by
      opcode, are not the correct length. It seems to be the only way to do
      such a check at compile time, as the sizeof() operator does not work
      in the C preprocessor. */

      case OP_TABLE_LENGTH:
      case OP_TABLE_LENGTH +
        ((sizeof(coptable) == OP_TABLE_LENGTH) &&
         (sizeof(poptable) == OP_TABLE_LENGTH)):
      return 0;

/* ========================================================================== */
      /* Reached a closing bracket. If not at the end of the pattern, carry
      on with the next opcode. For repeating opcodes, also add the repeat
      state. Note that KETRPOS will always be encountered at the end of the
      subpattern, because the possessive subpattern repeats are always handled
      using recursive calls. Thus, it never adds any new states.

      At the end of the (sub)pattern, unless we have an empty string and
      PCRE2_NOTEMPTY is set, or PCRE2_NOTEMPTY_ATSTART is set and we are at the
      start of the subject, save the match data, shifting up all previous
      matches so we always have the longest first. */

      case OP_KET:
      case OP_KETRMIN:
      case OP_KETRMAX:
      case OP_KETRPOS:
      if (code != end_code)
        {
        ADD_ACTIVE(state_offset + 1 + LINK_SIZE, 0);
        if (codevalue != OP_KET)
          {
          ADD_ACTIVE(state_offset - (int)GET(code, 1), 0);
          }
        }
      else
        {
        if (ptr > current_subject ||
            ((mb->moptions & PCRE2_NOTEMPTY) == 0 &&
              ((mb->moptions & PCRE2_NOTEMPTY_ATSTART) == 0 ||
                current_subject > start_subject + mb->start_offset)))
          {
          if (match_count < 0) match_count = (offsetcount >= 2)? 1 : 0;
            else if (match_count > 0 && ++match_count * 2 > (int)offsetcount)
              match_count = 0;
          count = ((match_count == 0)? (int)offsetcount : match_count * 2) - 2;
          if (count > 0) memmove(offsets + 2, offsets,
            (size_t)count * sizeof(PCRE2_SIZE));
          if (offsetcount >= 2)
            {
            offsets[0] = (PCRE2_SIZE)(current_subject - start_subject);
            offsets[1] = (PCRE2_SIZE)(ptr - start_subject);
            }
          if ((mb->moptions & PCRE2_DFA_SHORTEST) != 0) return match_count;
          }
        }
      break;

/* ========================================================================== */
      /* These opcodes add to the current list of states without looking
      at the current character. */

      /*-----------------------------------------------------------------*/
      case OP_ALT:
      do { code += GET(code, 1); } while (*code == OP_ALT);
      ADD_ACTIVE((int)(code - start_code), 0);
      break;

      /*-----------------------------------------------------------------*/
      case OP_BRA:
      case OP_SBRA:
      do
        {
        ADD_ACTIVE((int)(code - start_code + 1 + LINK_SIZE), 0);
        code += GET(code, 1);
        }
      while (*code == OP_ALT);
      break;

      /*-----------------------------------------------------------------*/
      case OP_CBRA:
      case OP_SCBRA:
      ADD_ACTIVE((int)(code - start_code + 1 + LINK_SIZE + IMM2_SIZE),  0);
      code += GET(code, 1);
      while (*code == OP_ALT)
        {
        ADD_ACTIVE((int)(code - start_code + 1 + LINK_SIZE),  0);
        code += GET(code, 1);
        }
      break;

      /*-----------------------------------------------------------------*/
      case OP_BRAZERO:
      case OP_BRAMINZERO:
      ADD_ACTIVE(state_offset + 1, 0);
      code += 1 + GET(code, 2);
      while (*code == OP_ALT) code += GET(code, 1);
      ADD_ACTIVE((int)(code - start_code + 1 + LINK_SIZE), 0);
      break;

      /*-----------------------------------------------------------------*/
      case OP_SKIPZERO:
      code += 1 + GET(code, 2);
      while (*code == OP_ALT) code += GET(code, 1);
      ADD_ACTIVE((int)(code - start_code + 1 + LINK_SIZE), 0);
      break;

      /*-----------------------------------------------------------------*/
      case OP_CIRC:
      if (ptr == start_subject && (mb->moptions & PCRE2_NOTBOL) == 0)
        { ADD_ACTIVE(state_offset + 1, 0); }
      break;

      /*-----------------------------------------------------------------*/
      case OP_CIRCM:
      if ((ptr == start_subject && (mb->moptions & PCRE2_NOTBOL) == 0) ||
          ((ptr != end_subject || (mb->poptions & PCRE2_ALT_CIRCUMFLEX) != 0 )
            && WAS_NEWLINE(ptr)))
        { ADD_ACTIVE(state_offset + 1, 0); }
      break;

      /*-----------------------------------------------------------------*/
      case OP_EOD:
      if (ptr >= end_subject)
        {
        if ((mb->moptions & PCRE2_PARTIAL_HARD) != 0)
          could_continue = TRUE;
        else { ADD_ACTIVE(state_offset + 1, 0); }
        }
      break;

      /*-----------------------------------------------------------------*/
      case OP_SOD:
      if (ptr == start_subject) { ADD_ACTIVE(state_offset + 1, 0); }
      break;

      /*-----------------------------------------------------------------*/
      case OP_SOM:
      if (ptr == start_subject + start_offset) { ADD_ACTIVE(state_offset + 1, 0); }
      break;


/* ========================================================================== */
      /* These opcodes inspect the next subject character, and sometimes
      the previous one as well, but do not have an argument. The variable
      clen contains the length of the current character and is zero if we are
      at the end of the subject. */

      /*-----------------------------------------------------------------*/
      case OP_ANY:
      if (clen > 0 && !IS_NEWLINE(ptr))
        {
        if (ptr + 1 >= mb->end_subject &&
            (mb->moptions & (PCRE2_PARTIAL_HARD)) != 0 &&
            NLBLOCK->nltype == NLTYPE_FIXED &&
            NLBLOCK->nllen == 2 &&
            c == NLBLOCK->nl[0])
          {
          could_continue = partial_newline = TRUE;
          }
        else
          {
          ADD_NEW(state_offset + 1, 0);
          }
        }
      break;

      /*-----------------------------------------------------------------*/
      case OP_ALLANY:
      if (clen > 0)
        { ADD_NEW(state_offset + 1, 0); }
      break;

      /*-----------------------------------------------------------------*/
      case OP_EODN:
      if (clen == 0 && (mb->moptions & PCRE2_PARTIAL_HARD) != 0)
        could_continue = TRUE;
      else if (clen == 0 || (IS_NEWLINE(ptr) && ptr == end_subject - mb->nllen))
        { ADD_ACTIVE(state_offset + 1, 0); }
      break;

      /*-----------------------------------------------------------------*/
      case OP_DOLL:
      if ((mb->moptions & PCRE2_NOTEOL) == 0)
        {
        if (clen == 0 && (mb->moptions & PCRE2_PARTIAL_HARD) != 0)
          could_continue = TRUE;
        else if (clen == 0 ||
            ((mb->poptions & PCRE2_DOLLAR_ENDONLY) == 0 && IS_NEWLINE(ptr) &&
               (ptr == end_subject - mb->nllen)
            ))
          { ADD_ACTIVE(state_offset + 1, 0); }
        else if (ptr + 1 >= mb->end_subject &&
                 (mb->moptions & (PCRE2_PARTIAL_HARD|PCRE2_PARTIAL_SOFT)) != 0 &&
                 NLBLOCK->nltype == NLTYPE_FIXED &&
                 NLBLOCK->nllen == 2 &&
                 c == NLBLOCK->nl[0])
          {
          if ((mb->moptions & PCRE2_PARTIAL_HARD) != 0)
            {
            reset_could_continue = TRUE;
            ADD_NEW_DATA(-(state_offset + 1), 0, 1);
            }
          else could_continue = partial_newline = TRUE;
          }
        }
      break;

      /*-----------------------------------------------------------------*/
      case OP_DOLLM:
      if ((mb->moptions & PCRE2_NOTEOL) == 0)
        {
        if (clen == 0 && (mb->moptions & PCRE2_PARTIAL_HARD) != 0)
          could_continue = TRUE;
        else if (clen == 0 ||
            ((mb->poptions & PCRE2_DOLLAR_ENDONLY) == 0 && IS_NEWLINE(ptr)))
          { ADD_ACTIVE(state_offset + 1, 0); }
        else if (ptr + 1 >= mb->end_subject &&
                 (mb->moptions & (PCRE2_PARTIAL_HARD|PCRE2_PARTIAL_SOFT)) != 0 &&
                 NLBLOCK->nltype == NLTYPE_FIXED &&
                 NLBLOCK->nllen == 2 &&
                 c == NLBLOCK->nl[0])
          {
          if ((mb->moptions & PCRE2_PARTIAL_HARD) != 0)
            {
            reset_could_continue = TRUE;
            ADD_NEW_DATA(-(state_offset + 1), 0, 1);
            }
          else could_continue = partial_newline = TRUE;
          }
        }
      else if (IS_NEWLINE(ptr))
        { ADD_ACTIVE(state_offset + 1, 0); }
      break;

      /*-----------------------------------------------------------------*/

      case OP_DIGIT:
      case OP_WHITESPACE:
      case OP_WORDCHAR:
      if (clen > 0 && c < 256 &&
            ((ctypes[c] & toptable1[codevalue]) ^ toptable2[codevalue]) != 0)
        { ADD_NEW(state_offset + 1, 0); }
      break;

      /*-----------------------------------------------------------------*/
      case OP_NOT_DIGIT:
      case OP_NOT_WHITESPACE:
      case OP_NOT_WORDCHAR:
      if (clen > 0 && (c >= 256 ||
            ((ctypes[c] & toptable1[codevalue]) ^ toptable2[codevalue]) != 0))
        { ADD_NEW(state_offset + 1, 0); }
      break;

      /*-----------------------------------------------------------------*/
      case OP_WORD_BOUNDARY:
      case OP_NOT_WORD_BOUNDARY:
        {
        int left_word, right_word;

        if (ptr > start_subject)
          {
          PCRE2_SPTR temp = ptr - 1;
          if (temp < mb->start_used_ptr) mb->start_used_ptr = temp;
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
          if (utf) { BACKCHAR(temp); }
#endif
          GETCHARTEST(d, temp);
#ifdef SUPPORT_UNICODE
          if ((mb->poptions & PCRE2_UCP) != 0)
            {
            if (d == '_') left_word = TRUE; else
              {
              uint32_t cat = UCD_CATEGORY(d);
              left_word = (cat == ucp_L || cat == ucp_N);
              }
            }
          else
#endif
          left_word = d < 256 && (ctypes[d] & ctype_word) != 0;
          }
        else left_word = FALSE;

        if (clen > 0)
          {
          if (ptr >= mb->last_used_ptr)
            {
            PCRE2_SPTR temp = ptr + 1;
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
            if (utf) { FORWARDCHARTEST(temp, mb->end_subject); }
#endif
            mb->last_used_ptr = temp;
            }
#ifdef SUPPORT_UNICODE
          if ((mb->poptions & PCRE2_UCP) != 0)
            {
            if (c == '_') right_word = TRUE; else
              {
              uint32_t cat = UCD_CATEGORY(c);
              right_word = (cat == ucp_L || cat == ucp_N);
              }
            }
          else
#endif
          right_word = c < 256 && (ctypes[c] & ctype_word) != 0;
          }
        else right_word = FALSE;

        if ((left_word == right_word) == (codevalue == OP_NOT_WORD_BOUNDARY))
          { ADD_ACTIVE(state_offset + 1, 0); }
        }
      break;


      /*-----------------------------------------------------------------*/
      /* Check the next character by Unicode property. We will get here only
      if the support is in the binary; otherwise a compile-time error occurs.
      */

#ifdef SUPPORT_UNICODE
      case OP_PROP:
      case OP_NOTPROP:
      if (clen > 0)
        {
        BOOL OK;
        const uint32_t *cp;
        const ucd_record * prop = GET_UCD(c);
        switch(code[1])
          {
          case PT_ANY:
          OK = TRUE;
          break;

          case PT_LAMP:
          OK = prop->chartype == ucp_Lu || prop->chartype == ucp_Ll ||
               prop->chartype == ucp_Lt;
          break;

          case PT_GC:
          OK = PRIV(ucp_gentype)[prop->chartype] == code[2];
          break;

          case PT_PC:
          OK = prop->chartype == code[2];
          break;

          case PT_SC:
          OK = prop->script == code[2];
          break;

          /* These are specials for combination cases. */

          case PT_ALNUM:
          OK = PRIV(ucp_gentype)[prop->chartype] == ucp_L ||
               PRIV(ucp_gentype)[prop->chartype] == ucp_N;
          break;

          /* Perl space used to exclude VT, but from Perl 5.18 it is included,
          which means that Perl space and POSIX space are now identical. PCRE
          was changed at release 8.34. */

          case PT_SPACE:    /* Perl space */
          case PT_PXSPACE:  /* POSIX space */
          switch(c)
            {
            HSPACE_CASES:
            VSPACE_CASES:
            OK = TRUE;
            break;

            default:
            OK = PRIV(ucp_gentype)[prop->chartype] == ucp_Z;
            break;
            }
          break;

          case PT_WORD:
          OK = PRIV(ucp_gentype)[prop->chartype] == ucp_L ||
               PRIV(ucp_gentype)[prop->chartype] == ucp_N ||
               c == CHAR_UNDERSCORE;
          break;

          case PT_CLIST:
          cp = PRIV(ucd_caseless_sets) + code[2];
          for (;;)
            {
            if (c < *cp) { OK = FALSE; break; }
            if (c == *cp++) { OK = TRUE; break; }
            }
          break;

          case PT_UCNC:
          OK = c == CHAR_DOLLAR_SIGN || c == CHAR_COMMERCIAL_AT ||
               c == CHAR_GRAVE_ACCENT || (c >= 0xa0 && c <= 0xd7ff) ||
               c >= 0xe000;
          break;

          /* Should never occur, but keep compilers from grumbling. */

          default:
          OK = codevalue != OP_PROP;
          break;
          }

        if (OK == (codevalue == OP_PROP)) { ADD_NEW(state_offset + 3, 0); }
        }
      break;
#endif



/* ========================================================================== */
      /* These opcodes likewise inspect the subject character, but have an
      argument that is not a data character. It is one of these opcodes:
      OP_ANY, OP_ALLANY, OP_DIGIT, OP_NOT_DIGIT, OP_WHITESPACE, OP_NOT_SPACE,
      OP_WORDCHAR, OP_NOT_WORDCHAR. The value is loaded into d. */

      case OP_TYPEPLUS:
      case OP_TYPEMINPLUS:
      case OP_TYPEPOSPLUS:
      count = current_state->count;  /* Already matched */
      if (count > 0) { ADD_ACTIVE(state_offset + 2, 0); }
      if (clen > 0)
        {
        if (d == OP_ANY && ptr + 1 >= mb->end_subject &&
            (mb->moptions & (PCRE2_PARTIAL_HARD)) != 0 &&
            NLBLOCK->nltype == NLTYPE_FIXED &&
            NLBLOCK->nllen == 2 &&
            c == NLBLOCK->nl[0])
          {
          could_continue = partial_newline = TRUE;
          }
        else if ((c >= 256 && d != OP_DIGIT && d != OP_WHITESPACE && d != OP_WORDCHAR) ||
            (c < 256 &&
              (d != OP_ANY || !IS_NEWLINE(ptr)) &&
              ((ctypes[c] & toptable1[d]) ^ toptable2[d]) != 0))
          {
          if (count > 0 && codevalue == OP_TYPEPOSPLUS)
            {
            active_count--;            /* Remove non-match possibility */
            next_active_state--;
            }
          count++;
          ADD_NEW(state_offset, count);
          }
        }
      break;

      /*-----------------------------------------------------------------*/
      case OP_TYPEQUERY:
      case OP_TYPEMINQUERY:
      case OP_TYPEPOSQUERY:
      ADD_ACTIVE(state_offset + 2, 0);
      if (clen > 0)
        {
        if (d == OP_ANY && ptr + 1 >= mb->end_subject &&
            (mb->moptions & (PCRE2_PARTIAL_HARD)) != 0 &&
            NLBLOCK->nltype == NLTYPE_FIXED &&
            NLBLOCK->nllen == 2 &&
            c == NLBLOCK->nl[0])
          {
          could_continue = partial_newline = TRUE;
          }
        else if ((c >= 256 && d != OP_DIGIT && d != OP_WHITESPACE && d != OP_WORDCHAR) ||
            (c < 256 &&
              (d != OP_ANY || !IS_NEWLINE(ptr)) &&
              ((ctypes[c] & toptable1[d]) ^ toptable2[d]) != 0))
          {
          if (codevalue == OP_TYPEPOSQUERY)
            {
            active_count--;            /* Remove non-match possibility */
            next_active_state--;
            }
          ADD_NEW(state_offset + 2, 0);
          }
        }
      break;

      /*-----------------------------------------------------------------*/
      case OP_TYPESTAR:
      case OP_TYPEMINSTAR:
      case OP_TYPEPOSSTAR:
      ADD_ACTIVE(state_offset + 2, 0);
      if (clen > 0)
        {
        if (d == OP_ANY && ptr + 1 >= mb->end_subject &&
            (mb->moptions & (PCRE2_PARTIAL_HARD)) != 0 &&
            NLBLOCK->nltype == NLTYPE_FIXED &&
            NLBLOCK->nllen == 2 &&
            c == NLBLOCK->nl[0])
          {
          could_continue = partial_newline = TRUE;
          }
        else if ((c >= 256 && d != OP_DIGIT && d != OP_WHITESPACE && d != OP_WORDCHAR) ||
            (c < 256 &&
              (d != OP_ANY || !IS_NEWLINE(ptr)) &&
              ((ctypes[c] & toptable1[d]) ^ toptable2[d]) != 0))
          {
          if (codevalue == OP_TYPEPOSSTAR)
            {
            active_count--;            /* Remove non-match possibility */
            next_active_state--;
            }
          ADD_NEW(state_offset, 0);
          }
        }
      break;

      /*-----------------------------------------------------------------*/
      case OP_TYPEEXACT:
      count = current_state->count;  /* Number already matched */
      if (clen > 0)
        {
        if (d == OP_ANY && ptr + 1 >= mb->end_subject &&
            (mb->moptions & (PCRE2_PARTIAL_HARD)) != 0 &&
            NLBLOCK->nltype == NLTYPE_FIXED &&
            NLBLOCK->nllen == 2 &&
            c == NLBLOCK->nl[0])
          {
          could_continue = partial_newline = TRUE;
          }
        else if ((c >= 256 && d != OP_DIGIT && d != OP_WHITESPACE && d != OP_WORDCHAR) ||
            (c < 256 &&
              (d != OP_ANY || !IS_NEWLINE(ptr)) &&
              ((ctypes[c] & toptable1[d]) ^ toptable2[d]) != 0))
          {
          if (++count >= (int)GET2(code, 1))
            { ADD_NEW(state_offset + 1 + IMM2_SIZE + 1, 0); }
          else
            { ADD_NEW(state_offset, count); }
          }
        }
      break;

      /*-----------------------------------------------------------------*/
      case OP_TYPEUPTO:
      case OP_TYPEMINUPTO:
      case OP_TYPEPOSUPTO:
      ADD_ACTIVE(state_offset + 2 + IMM2_SIZE, 0);
      count = current_state->count;  /* Number already matched */
      if (clen > 0)
        {
        if (d == OP_ANY && ptr + 1 >= mb->end_subject &&
            (mb->moptions & (PCRE2_PARTIAL_HARD)) != 0 &&
            NLBLOCK->nltype == NLTYPE_FIXED &&
            NLBLOCK->nllen == 2 &&
            c == NLBLOCK->nl[0])
          {
          could_continue = partial_newline = TRUE;
          }
        else if ((c >= 256 && d != OP_DIGIT && d != OP_WHITESPACE && d != OP_WORDCHAR) ||
            (c < 256 &&
              (d != OP_ANY || !IS_NEWLINE(ptr)) &&
              ((ctypes[c] & toptable1[d]) ^ toptable2[d]) != 0))
          {
          if (codevalue == OP_TYPEPOSUPTO)
            {
            active_count--;           /* Remove non-match possibility */
            next_active_state--;
            }
          if (++count >= (int)GET2(code, 1))
            { ADD_NEW(state_offset + 2 + IMM2_SIZE, 0); }
          else
            { ADD_NEW(state_offset, count); }
          }
        }
      break;

/* ========================================================================== */
      /* These are virtual opcodes that are used when something like
      OP_TYPEPLUS has OP_PROP, OP_NOTPROP, OP_ANYNL, or OP_EXTUNI as its
      argument. It keeps the code above fast for the other cases. The argument
      is in the d variable. */

#ifdef SUPPORT_UNICODE
      case OP_PROP_EXTRA + OP_TYPEPLUS:
      case OP_PROP_EXTRA + OP_TYPEMINPLUS:
      case OP_PROP_EXTRA + OP_TYPEPOSPLUS:
      count = current_state->count;           /* Already matched */
      if (count > 0) { ADD_ACTIVE(state_offset + 4, 0); }
      if (clen > 0)
        {
        BOOL OK;
        const uint32_t *cp;
        const ucd_record * prop = GET_UCD(c);
        switch(code[2])
          {
          case PT_ANY:
          OK = TRUE;
          break;

          case PT_LAMP:
          OK = prop->chartype == ucp_Lu || prop->chartype == ucp_Ll ||
            prop->chartype == ucp_Lt;
          break;

          case PT_GC:
          OK = PRIV(ucp_gentype)[prop->chartype] == code[3];
          break;

          case PT_PC:
          OK = prop->chartype == code[3];
          break;

          case PT_SC:
          OK = prop->script == code[3];
          break;

          /* These are specials for combination cases. */

          case PT_ALNUM:
          OK = PRIV(ucp_gentype)[prop->chartype] == ucp_L ||
               PRIV(ucp_gentype)[prop->chartype] == ucp_N;
          break;

          /* Perl space used to exclude VT, but from Perl 5.18 it is included,
          which means that Perl space and POSIX space are now identical. PCRE
          was changed at release 8.34. */

          case PT_SPACE:    /* Perl space */
          case PT_PXSPACE:  /* POSIX space */
          switch(c)
            {
            HSPACE_CASES:
            VSPACE_CASES:
            OK = TRUE;
            break;

            default:
            OK = PRIV(ucp_gentype)[prop->chartype] == ucp_Z;
            break;
            }
          break;

          case PT_WORD:
          OK = PRIV(ucp_gentype)[prop->chartype] == ucp_L ||
               PRIV(ucp_gentype)[prop->chartype] == ucp_N ||
               c == CHAR_UNDERSCORE;
          break;

          case PT_CLIST:
          cp = PRIV(ucd_caseless_sets) + code[3];
          for (;;)
            {
            if (c < *cp) { OK = FALSE; break; }
            if (c == *cp++) { OK = TRUE; break; }
            }
          break;

          case PT_UCNC:
          OK = c == CHAR_DOLLAR_SIGN || c == CHAR_COMMERCIAL_AT ||
               c == CHAR_GRAVE_ACCENT || (c >= 0xa0 && c <= 0xd7ff) ||
               c >= 0xe000;
          break;

          /* Should never occur, but keep compilers from grumbling. */

          default:
          OK = codevalue != OP_PROP;
          break;
          }

        if (OK == (d == OP_PROP))
          {
          if (count > 0 && codevalue == OP_PROP_EXTRA + OP_TYPEPOSPLUS)
            {
            active_count--;           /* Remove non-match possibility */
            next_active_state--;
            }
          count++;
          ADD_NEW(state_offset, count);
          }
        }
      break;

      /*-----------------------------------------------------------------*/
      case OP_EXTUNI_EXTRA + OP_TYPEPLUS:
      case OP_EXTUNI_EXTRA + OP_TYPEMINPLUS:
      case OP_EXTUNI_EXTRA + OP_TYPEPOSPLUS:
      count = current_state->count;  /* Already matched */
      if (count > 0) { ADD_ACTIVE(state_offset + 2, 0); }
      if (clen > 0)
        {
        int ncount = 0;
        if (count > 0 && codevalue == OP_EXTUNI_EXTRA + OP_TYPEPOSPLUS)
          {
          active_count--;           /* Remove non-match possibility */
          next_active_state--;
          }
        (void)PRIV(extuni)(c, ptr + clen, mb->start_subject, end_subject, utf,
          &ncount);
        count++;
        ADD_NEW_DATA(-state_offset, count, ncount);
        }
      break;
#endif

      /*-----------------------------------------------------------------*/
      case OP_ANYNL_EXTRA + OP_TYPEPLUS:
      case OP_ANYNL_EXTRA + OP_TYPEMINPLUS:
      case OP_ANYNL_EXTRA + OP_TYPEPOSPLUS:
      count = current_state->count;  /* Already matched */
      if (count > 0) { ADD_ACTIVE(state_offset + 2, 0); }
      if (clen > 0)
        {
        int ncount = 0;
        switch (c)
          {
          case CHAR_VT:
          case CHAR_FF:
          case CHAR_NEL:
#ifndef EBCDIC
          case 0x2028:
          case 0x2029:
#endif  /* Not EBCDIC */
          if (mb->bsr_convention == PCRE2_BSR_ANYCRLF) break;
          goto ANYNL01;

          case CHAR_CR:
          if (ptr + 1 < end_subject && UCHAR21TEST(ptr + 1) == CHAR_LF) ncount = 1;
          /* Fall through */

          ANYNL01:
          case CHAR_LF:
          if (count > 0 && codevalue == OP_ANYNL_EXTRA + OP_TYPEPOSPLUS)
            {
            active_count--;           /* Remove non-match possibility */
            next_active_state--;
            }
          count++;
          ADD_NEW_DATA(-state_offset, count, ncount);
          break;

          default:
          break;
          }
        }
      break;

      /*-----------------------------------------------------------------*/
      case OP_VSPACE_EXTRA + OP_TYPEPLUS:
      case OP_VSPACE_EXTRA + OP_TYPEMINPLUS:
      case OP_VSPACE_EXTRA + OP_TYPEPOSPLUS:
      count = current_state->count;  /* Already matched */
      if (count > 0) { ADD_ACTIVE(state_offset + 2, 0); }
      if (clen > 0)
        {
        BOOL OK;
        switch (c)
          {
          VSPACE_CASES:
          OK = TRUE;
          break;

          default:
          OK = FALSE;
          break;
          }

        if (OK == (d == OP_VSPACE))
          {
          if (count > 0 && codevalue == OP_VSPACE_EXTRA + OP_TYPEPOSPLUS)
            {
            active_count--;           /* Remove non-match possibility */
            next_active_state--;
            }
          count++;
          ADD_NEW_DATA(-state_offset, count, 0);
          }
        }
      break;

      /*-----------------------------------------------------------------*/
      case OP_HSPACE_EXTRA + OP_TYPEPLUS:
      case OP_HSPACE_EXTRA + OP_TYPEMINPLUS:
      case OP_HSPACE_EXTRA + OP_TYPEPOSPLUS:
      count = current_state->count;  /* Already matched */
      if (count > 0) { ADD_ACTIVE(state_offset + 2, 0); }
      if (clen > 0)
        {
        BOOL OK;
        switch (c)
          {
          HSPACE_CASES:
          OK = TRUE;
          break;

          default:
          OK = FALSE;
          break;
          }

        if (OK == (d == OP_HSPACE))
          {
          if (count > 0 && codevalue == OP_HSPACE_EXTRA + OP_TYPEPOSPLUS)
            {
            active_count--;           /* Remove non-match possibility */
            next_active_state--;
            }
          count++;
          ADD_NEW_DATA(-state_offset, count, 0);
          }
        }
      break;

      /*-----------------------------------------------------------------*/
#ifdef SUPPORT_UNICODE
      case OP_PROP_EXTRA + OP_TYPEQUERY:
      case OP_PROP_EXTRA + OP_TYPEMINQUERY:
      case OP_PROP_EXTRA + OP_TYPEPOSQUERY:
      count = 4;
      goto QS1;

      case OP_PROP_EXTRA + OP_TYPESTAR:
      case OP_PROP_EXTRA + OP_TYPEMINSTAR:
      case OP_PROP_EXTRA + OP_TYPEPOSSTAR:
      count = 0;

      QS1:

      ADD_ACTIVE(state_offset + 4, 0);
      if (clen > 0)
        {
        BOOL OK;
        const uint32_t *cp;
        const ucd_record * prop = GET_UCD(c);
        switch(code[2])
          {
          case PT_ANY:
          OK = TRUE;
          break;

          case PT_LAMP:
          OK = prop->chartype == ucp_Lu || prop->chartype == ucp_Ll ||
            prop->chartype == ucp_Lt;
          break;

          case PT_GC:
          OK = PRIV(ucp_gentype)[prop->chartype] == code[3];
          break;

          case PT_PC:
          OK = prop->chartype == code[3];
          break;

          case PT_SC:
          OK = prop->script == code[3];
          break;

          /* These are specials for combination cases. */

          case PT_ALNUM:
          OK = PRIV(ucp_gentype)[prop->chartype] == ucp_L ||
               PRIV(ucp_gentype)[prop->chartype] == ucp_N;
          break;

          /* Perl space used to exclude VT, but from Perl 5.18 it is included,
          which means that Perl space and POSIX space are now identical. PCRE
          was changed at release 8.34. */

          case PT_SPACE:    /* Perl space */
          case PT_PXSPACE:  /* POSIX space */
          switch(c)
            {
            HSPACE_CASES:
            VSPACE_CASES:
            OK = TRUE;
            break;

            default:
            OK = PRIV(ucp_gentype)[prop->chartype] == ucp_Z;
            break;
            }
          break;

          case PT_WORD:
          OK = PRIV(ucp_gentype)[prop->chartype] == ucp_L ||
               PRIV(ucp_gentype)[prop->chartype] == ucp_N ||
               c == CHAR_UNDERSCORE;
          break;

          case PT_CLIST:
          cp = PRIV(ucd_caseless_sets) + code[3];
          for (;;)
            {
            if (c < *cp) { OK = FALSE; break; }
            if (c == *cp++) { OK = TRUE; break; }
            }
          break;

          case PT_UCNC:
          OK = c == CHAR_DOLLAR_SIGN || c == CHAR_COMMERCIAL_AT ||
               c == CHAR_GRAVE_ACCENT || (c >= 0xa0 && c <= 0xd7ff) ||
               c >= 0xe000;
          break;

          /* Should never occur, but keep compilers from grumbling. */

          default:
          OK = codevalue != OP_PROP;
          break;
          }

        if (OK == (d == OP_PROP))
          {
          if (codevalue == OP_PROP_EXTRA + OP_TYPEPOSSTAR ||
              codevalue == OP_PROP_EXTRA + OP_TYPEPOSQUERY)
            {
            active_count--;           /* Remove non-match possibility */
            next_active_state--;
            }
          ADD_NEW(state_offset + count, 0);
          }
        }
      break;

      /*-----------------------------------------------------------------*/
      case OP_EXTUNI_EXTRA + OP_TYPEQUERY:
      case OP_EXTUNI_EXTRA + OP_TYPEMINQUERY:
      case OP_EXTUNI_EXTRA + OP_TYPEPOSQUERY:
      count = 2;
      goto QS2;

      case OP_EXTUNI_EXTRA + OP_TYPESTAR:
      case OP_EXTUNI_EXTRA + OP_TYPEMINSTAR:
      case OP_EXTUNI_EXTRA + OP_TYPEPOSSTAR:
      count = 0;

      QS2:

      ADD_ACTIVE(state_offset + 2, 0);
      if (clen > 0)
        {
        int ncount = 0;
        if (codevalue == OP_EXTUNI_EXTRA + OP_TYPEPOSSTAR ||
            codevalue == OP_EXTUNI_EXTRA + OP_TYPEPOSQUERY)
          {
          active_count--;           /* Remove non-match possibility */
          next_active_state--;
          }
        (void)PRIV(extuni)(c, ptr + clen, mb->start_subject, end_subject, utf,
          &ncount);
        ADD_NEW_DATA(-(state_offset + count), 0, ncount);
        }
      break;
#endif

      /*-----------------------------------------------------------------*/
      case OP_ANYNL_EXTRA + OP_TYPEQUERY:
      case OP_ANYNL_EXTRA + OP_TYPEMINQUERY:
      case OP_ANYNL_EXTRA + OP_TYPEPOSQUERY:
      count = 2;
      goto QS3;

      case OP_ANYNL_EXTRA + OP_TYPESTAR:
      case OP_ANYNL_EXTRA + OP_TYPEMINSTAR:
      case OP_ANYNL_EXTRA + OP_TYPEPOSSTAR:
      count = 0;

      QS3:
      ADD_ACTIVE(state_offset + 2, 0);
      if (clen > 0)
        {
        int ncount = 0;
        switch (c)
          {
          case CHAR_VT:
          case CHAR_FF:
          case CHAR_NEL:
#ifndef EBCDIC
          case 0x2028:
          case 0x2029:
#endif  /* Not EBCDIC */
          if (mb->bsr_convention == PCRE2_BSR_ANYCRLF) break;
          goto ANYNL02;

          case CHAR_CR:
          if (ptr + 1 < end_subject && UCHAR21TEST(ptr + 1) == CHAR_LF) ncount = 1;
          /* Fall through */

          ANYNL02:
          case CHAR_LF:
          if (codevalue == OP_ANYNL_EXTRA + OP_TYPEPOSSTAR ||
              codevalue == OP_ANYNL_EXTRA + OP_TYPEPOSQUERY)
            {
            active_count--;           /* Remove non-match possibility */
            next_active_state--;
            }
          ADD_NEW_DATA(-(state_offset + (int)count), 0, ncount);
          break;

          default:
          break;
          }
        }
      break;

      /*-----------------------------------------------------------------*/
      case OP_VSPACE_EXTRA + OP_TYPEQUERY:
      case OP_VSPACE_EXTRA + OP_TYPEMINQUERY:
      case OP_VSPACE_EXTRA + OP_TYPEPOSQUERY:
      count = 2;
      goto QS4;

      case OP_VSPACE_EXTRA + OP_TYPESTAR:
      case OP_VSPACE_EXTRA + OP_TYPEMINSTAR:
      case OP_VSPACE_EXTRA + OP_TYPEPOSSTAR:
      count = 0;

      QS4:
      ADD_ACTIVE(state_offset + 2, 0);
      if (clen > 0)
        {
        BOOL OK;
        switch (c)
          {
          VSPACE_CASES:
          OK = TRUE;
          break;

          default:
          OK = FALSE;
          break;
          }
        if (OK == (d == OP_VSPACE))
          {
          if (codevalue == OP_VSPACE_EXTRA + OP_TYPEPOSSTAR ||
              codevalue == OP_VSPACE_EXTRA + OP_TYPEPOSQUERY)
            {
            active_count--;           /* Remove non-match possibility */
            next_active_state--;
            }
          ADD_NEW_DATA(-(state_offset + (int)count), 0, 0);
          }
        }
      break;

      /*-----------------------------------------------------------------*/
      case OP_HSPACE_EXTRA + OP_TYPEQUERY:
      case OP_HSPACE_EXTRA + OP_TYPEMINQUERY:
      case OP_HSPACE_EXTRA + OP_TYPEPOSQUERY:
      count = 2;
      goto QS5;

      case OP_HSPACE_EXTRA + OP_TYPESTAR:
      case OP_HSPACE_EXTRA + OP_TYPEMINSTAR:
      case OP_HSPACE_EXTRA + OP_TYPEPOSSTAR:
      count = 0;

      QS5:
      ADD_ACTIVE(state_offset + 2, 0);
      if (clen > 0)
        {
        BOOL OK;
        switch (c)
          {
          HSPACE_CASES:
          OK = TRUE;
          break;

          default:
          OK = FALSE;
          break;
          }

        if (OK == (d == OP_HSPACE))
          {
          if (codevalue == OP_HSPACE_EXTRA + OP_TYPEPOSSTAR ||
              codevalue == OP_HSPACE_EXTRA + OP_TYPEPOSQUERY)
            {
            active_count--;           /* Remove non-match possibility */
            next_active_state--;
            }
          ADD_NEW_DATA(-(state_offset + (int)count), 0, 0);
          }
        }
      break;

      /*-----------------------------------------------------------------*/
#ifdef SUPPORT_UNICODE
      case OP_PROP_EXTRA + OP_TYPEEXACT:
      case OP_PROP_EXTRA + OP_TYPEUPTO:
      case OP_PROP_EXTRA + OP_TYPEMINUPTO:
      case OP_PROP_EXTRA + OP_TYPEPOSUPTO:
      if (codevalue != OP_PROP_EXTRA + OP_TYPEEXACT)
        { ADD_ACTIVE(state_offset + 1 + IMM2_SIZE + 3, 0); }
      count = current_state->count;  /* Number already matched */
      if (clen > 0)
        {
        BOOL OK;
        const uint32_t *cp;
        const ucd_record * prop = GET_UCD(c);
        switch(code[1 + IMM2_SIZE + 1])
          {
          case PT_ANY:
          OK = TRUE;
          break;

          case PT_LAMP:
          OK = prop->chartype == ucp_Lu || prop->chartype == ucp_Ll ||
            prop->chartype == ucp_Lt;
          break;

          case PT_GC:
          OK = PRIV(ucp_gentype)[prop->chartype] == code[1 + IMM2_SIZE + 2];
          break;

          case PT_PC:
          OK = prop->chartype == code[1 + IMM2_SIZE + 2];
          break;

          case PT_SC:
          OK = prop->script == code[1 + IMM2_SIZE + 2];
          break;

          /* These are specials for combination cases. */

          case PT_ALNUM:
          OK = PRIV(ucp_gentype)[prop->chartype] == ucp_L ||
               PRIV(ucp_gentype)[prop->chartype] == ucp_N;
          break;

          /* Perl space used to exclude VT, but from Perl 5.18 it is included,
          which means that Perl space and POSIX space are now identical. PCRE
          was changed at release 8.34. */

          case PT_SPACE:    /* Perl space */
          case PT_PXSPACE:  /* POSIX space */
          switch(c)
            {
            HSPACE_CASES:
            VSPACE_CASES:
            OK = TRUE;
            break;

            default:
            OK = PRIV(ucp_gentype)[prop->chartype] == ucp_Z;
            break;
            }
          break;

          case PT_WORD:
          OK = PRIV(ucp_gentype)[prop->chartype] == ucp_L ||
               PRIV(ucp_gentype)[prop->chartype] == ucp_N ||
               c == CHAR_UNDERSCORE;
          break;

          case PT_CLIST:
          cp = PRIV(ucd_caseless_sets) + code[1 + IMM2_SIZE + 2];
          for (;;)
            {
            if (c < *cp) { OK = FALSE; break; }
            if (c == *cp++) { OK = TRUE; break; }
            }
          break;

          case PT_UCNC:
          OK = c == CHAR_DOLLAR_SIGN || c == CHAR_COMMERCIAL_AT ||
               c == CHAR_GRAVE_ACCENT || (c >= 0xa0 && c <= 0xd7ff) ||
               c >= 0xe000;
          break;

          /* Should never occur, but keep compilers from grumbling. */

          default:
          OK = codevalue != OP_PROP;
          break;
          }

        if (OK == (d == OP_PROP))
          {
          if (codevalue == OP_PROP_EXTRA + OP_TYPEPOSUPTO)
            {
            active_count--;           /* Remove non-match possibility */
            next_active_state--;
            }
          if (++count >= (int)GET2(code, 1))
            { ADD_NEW(state_offset + 1 + IMM2_SIZE + 3, 0); }
          else
            { ADD_NEW(state_offset, count); }
          }
        }
      break;

      /*-----------------------------------------------------------------*/
      case OP_EXTUNI_EXTRA + OP_TYPEEXACT:
      case OP_EXTUNI_EXTRA + OP_TYPEUPTO:
      case OP_EXTUNI_EXTRA + OP_TYPEMINUPTO:
      case OP_EXTUNI_EXTRA + OP_TYPEPOSUPTO:
      if (codevalue != OP_EXTUNI_EXTRA + OP_TYPEEXACT)
        { ADD_ACTIVE(state_offset + 2 + IMM2_SIZE, 0); }
      count = current_state->count;  /* Number already matched */
      if (clen > 0)
        {
        PCRE2_SPTR nptr;
        int ncount = 0;
        if (codevalue == OP_EXTUNI_EXTRA + OP_TYPEPOSUPTO)
          {
          active_count--;           /* Remove non-match possibility */
          next_active_state--;
          }
        nptr = PRIV(extuni)(c, ptr + clen, mb->start_subject, end_subject, utf,
          &ncount);
        if (nptr >= end_subject && (mb->moptions & PCRE2_PARTIAL_HARD) != 0)
            reset_could_continue = TRUE;
        if (++count >= (int)GET2(code, 1))
          { ADD_NEW_DATA(-(state_offset + 2 + IMM2_SIZE), 0, ncount); }
        else
          { ADD_NEW_DATA(-state_offset, count, ncount); }
        }
      break;
#endif

      /*-----------------------------------------------------------------*/
      case OP_ANYNL_EXTRA + OP_TYPEEXACT:
      case OP_ANYNL_EXTRA + OP_TYPEUPTO:
      case OP_ANYNL_EXTRA + OP_TYPEMINUPTO:
      case OP_ANYNL_EXTRA + OP_TYPEPOSUPTO:
      if (codevalue != OP_ANYNL_EXTRA + OP_TYPEEXACT)
        { ADD_ACTIVE(state_offset + 2 + IMM2_SIZE, 0); }
      count = current_state->count;  /* Number already matched */
      if (clen > 0)
        {
        int ncount = 0;
        switch (c)
          {
          case CHAR_VT:
          case CHAR_FF:
          case CHAR_NEL:
#ifndef EBCDIC
          case 0x2028:
          case 0x2029:
#endif  /* Not EBCDIC */
          if (mb->bsr_convention == PCRE2_BSR_ANYCRLF) break;
          goto ANYNL03;

          case CHAR_CR:
          if (ptr + 1 < end_subject && UCHAR21TEST(ptr + 1) == CHAR_LF) ncount = 1;
          /* Fall through */

          ANYNL03:
          case CHAR_LF:
          if (codevalue == OP_ANYNL_EXTRA + OP_TYPEPOSUPTO)
            {
            active_count--;           /* Remove non-match possibility */
            next_active_state--;
            }
          if (++count >= (int)GET2(code, 1))
            { ADD_NEW_DATA(-(state_offset + 2 + IMM2_SIZE), 0, ncount); }
          else
            { ADD_NEW_DATA(-state_offset, count, ncount); }
          break;

          default:
          break;
          }
        }
      break;

      /*-----------------------------------------------------------------*/
      case OP_VSPACE_EXTRA + OP_TYPEEXACT:
      case OP_VSPACE_EXTRA + OP_TYPEUPTO:
      case OP_VSPACE_EXTRA + OP_TYPEMINUPTO:
      case OP_VSPACE_EXTRA + OP_TYPEPOSUPTO:
      if (codevalue != OP_VSPACE_EXTRA + OP_TYPEEXACT)
        { ADD_ACTIVE(state_offset + 2 + IMM2_SIZE, 0); }
      count = current_state->count;  /* Number already matched */
      if (clen > 0)
        {
        BOOL OK;
        switch (c)
          {
          VSPACE_CASES:
          OK = TRUE;
          break;

          default:
          OK = FALSE;
          }

        if (OK == (d == OP_VSPACE))
          {
          if (codevalue == OP_VSPACE_EXTRA + OP_TYPEPOSUPTO)
            {
            active_count--;           /* Remove non-match possibility */
            next_active_state--;
            }
          if (++count >= (int)GET2(code, 1))
            { ADD_NEW_DATA(-(state_offset + 2 + IMM2_SIZE), 0, 0); }
          else
            { ADD_NEW_DATA(-state_offset, count, 0); }
          }
        }
      break;

      /*-----------------------------------------------------------------*/
      case OP_HSPACE_EXTRA + OP_TYPEEXACT:
      case OP_HSPACE_EXTRA + OP_TYPEUPTO:
      case OP_HSPACE_EXTRA + OP_TYPEMINUPTO:
      case OP_HSPACE_EXTRA + OP_TYPEPOSUPTO:
      if (codevalue != OP_HSPACE_EXTRA + OP_TYPEEXACT)
        { ADD_ACTIVE(state_offset + 2 + IMM2_SIZE, 0); }
      count = current_state->count;  /* Number already matched */
      if (clen > 0)
        {
        BOOL OK;
        switch (c)
          {
          HSPACE_CASES:
          OK = TRUE;
          break;

          default:
          OK = FALSE;
          break;
          }

        if (OK == (d == OP_HSPACE))
          {
          if (codevalue == OP_HSPACE_EXTRA + OP_TYPEPOSUPTO)
            {
            active_count--;           /* Remove non-match possibility */
            next_active_state--;
            }
          if (++count >= (int)GET2(code, 1))
            { ADD_NEW_DATA(-(state_offset + 2 + IMM2_SIZE), 0, 0); }
          else
            { ADD_NEW_DATA(-state_offset, count, 0); }
          }
        }
      break;

/* ========================================================================== */
      /* These opcodes are followed by a character that is usually compared
      to the current subject character; it is loaded into d. We still get
      here even if there is no subject character, because in some cases zero
      repetitions are permitted. */

      /*-----------------------------------------------------------------*/
      case OP_CHAR:
      if (clen > 0 && c == d) { ADD_NEW(state_offset + dlen + 1, 0); }
      break;

      /*-----------------------------------------------------------------*/
      case OP_CHARI:
      if (clen == 0) break;

#ifdef SUPPORT_UNICODE
      if (utf)
        {
        if (c == d) { ADD_NEW(state_offset + dlen + 1, 0); } else
          {
          unsigned int othercase;
          if (c < 128)
            othercase = fcc[c];
          else
            othercase = UCD_OTHERCASE(c);
          if (d == othercase) { ADD_NEW(state_offset + dlen + 1, 0); }
          }
        }
      else
#endif  /* SUPPORT_UNICODE */
      /* Not UTF mode */
        {
        if (TABLE_GET(c, lcc, c) == TABLE_GET(d, lcc, d))
          { ADD_NEW(state_offset + 2, 0); }
        }
      break;


#ifdef SUPPORT_UNICODE
      /*-----------------------------------------------------------------*/
      /* This is a tricky one because it can match more than one character.
      Find out how many characters to skip, and then set up a negative state
      to wait for them to pass before continuing. */

      case OP_EXTUNI:
      if (clen > 0)
        {
        int ncount = 0;
        PCRE2_SPTR nptr = PRIV(extuni)(c, ptr + clen, mb->start_subject,
          end_subject, utf, &ncount);
        if (nptr >= end_subject && (mb->moptions & PCRE2_PARTIAL_HARD) != 0)
            reset_could_continue = TRUE;
        ADD_NEW_DATA(-(state_offset + 1), 0, ncount);
        }
      break;
#endif

      /*-----------------------------------------------------------------*/
      /* This is a tricky like EXTUNI because it too can match more than one
      character (when CR is followed by LF). In this case, set up a negative
      state to wait for one character to pass before continuing. */

      case OP_ANYNL:
      if (clen > 0) switch(c)
        {
        case CHAR_VT:
        case CHAR_FF:
        case CHAR_NEL:
#ifndef EBCDIC
        case 0x2028:
        case 0x2029:
#endif  /* Not EBCDIC */
        if (mb->bsr_convention == PCRE2_BSR_ANYCRLF) break;
        /* Fall through */

        case CHAR_LF:
        ADD_NEW(state_offset + 1, 0);
        break;

        case CHAR_CR:
        if (ptr + 1 >= end_subject)
          {
          ADD_NEW(state_offset + 1, 0);
          if ((mb->moptions & PCRE2_PARTIAL_HARD) != 0)
            reset_could_continue = TRUE;
          }
        else if (UCHAR21TEST(ptr + 1) == CHAR_LF)
          {
          ADD_NEW_DATA(-(state_offset + 1), 0, 1);
          }
        else
          {
          ADD_NEW(state_offset + 1, 0);
          }
        break;
        }
      break;

      /*-----------------------------------------------------------------*/
      case OP_NOT_VSPACE:
      if (clen > 0) switch(c)
        {
        VSPACE_CASES:
        break;

        default:
        ADD_NEW(state_offset + 1, 0);
        break;
        }
      break;

      /*-----------------------------------------------------------------*/
      case OP_VSPACE:
      if (clen > 0) switch(c)
        {
        VSPACE_CASES:
        ADD_NEW(state_offset + 1, 0);
        break;

        default:
        break;
        }
      break;

      /*-----------------------------------------------------------------*/
      case OP_NOT_HSPACE:
      if (clen > 0) switch(c)
        {
        HSPACE_CASES:
        break;

        default:
        ADD_NEW(state_offset + 1, 0);
        break;
        }
      break;

      /*-----------------------------------------------------------------*/
      case OP_HSPACE:
      if (clen > 0) switch(c)
        {
        HSPACE_CASES:
        ADD_NEW(state_offset + 1, 0);
        break;

        default:
        break;
        }
      break;

      /*-----------------------------------------------------------------*/
      /* Match a negated single character casefully. */

      case OP_NOT:
      if (clen > 0 && c != d) { ADD_NEW(state_offset + dlen + 1, 0); }
      break;

      /*-----------------------------------------------------------------*/
      /* Match a negated single character caselessly. */

      case OP_NOTI:
      if (clen > 0)
        {
        uint32_t otherd;
#ifdef SUPPORT_UNICODE
        if (utf && d >= 128)
          otherd = UCD_OTHERCASE(d);
        else
#endif  /* SUPPORT_UNICODE */
        otherd = TABLE_GET(d, fcc, d);
        if (c != d && c != otherd)
          { ADD_NEW(state_offset + dlen + 1, 0); }
        }
      break;

      /*-----------------------------------------------------------------*/
      case OP_PLUSI:
      case OP_MINPLUSI:
      case OP_POSPLUSI:
      case OP_NOTPLUSI:
      case OP_NOTMINPLUSI:
      case OP_NOTPOSPLUSI:
      caseless = TRUE;
      codevalue -= OP_STARI - OP_STAR;

      /* Fall through */
      case OP_PLUS:
      case OP_MINPLUS:
      case OP_POSPLUS:
      case OP_NOTPLUS:
      case OP_NOTMINPLUS:
      case OP_NOTPOSPLUS:
      count = current_state->count;  /* Already matched */
      if (count > 0) { ADD_ACTIVE(state_offset + dlen + 1, 0); }
      if (clen > 0)
        {
        uint32_t otherd = NOTACHAR;
        if (caseless)
          {
#ifdef SUPPORT_UNICODE
          if (utf && d >= 128)
            otherd = UCD_OTHERCASE(d);
          else
#endif  /* SUPPORT_UNICODE */
          otherd = TABLE_GET(d, fcc, d);
          }
        if ((c == d || c == otherd) == (codevalue < OP_NOTSTAR))
          {
          if (count > 0 &&
              (codevalue == OP_POSPLUS || codevalue == OP_NOTPOSPLUS))
            {
            active_count--;             /* Remove non-match possibility */
            next_active_state--;
            }
          count++;
          ADD_NEW(state_offset, count);
          }
        }
      break;

      /*-----------------------------------------------------------------*/
      case OP_QUERYI:
      case OP_MINQUERYI:
      case OP_POSQUERYI:
      case OP_NOTQUERYI:
      case OP_NOTMINQUERYI:
      case OP_NOTPOSQUERYI:
      caseless = TRUE;
      codevalue -= OP_STARI - OP_STAR;
      /* Fall through */
      case OP_QUERY:
      case OP_MINQUERY:
      case OP_POSQUERY:
      case OP_NOTQUERY:
      case OP_NOTMINQUERY:
      case OP_NOTPOSQUERY:
      ADD_ACTIVE(state_offset + dlen + 1, 0);
      if (clen > 0)
        {
        uint32_t otherd = NOTACHAR;
        if (caseless)
          {
#ifdef SUPPORT_UNICODE
          if (utf && d >= 128)
            otherd = UCD_OTHERCASE(d);
          else
#endif  /* SUPPORT_UNICODE */
          otherd = TABLE_GET(d, fcc, d);
          }
        if ((c == d || c == otherd) == (codevalue < OP_NOTSTAR))
          {
          if (codevalue == OP_POSQUERY || codevalue == OP_NOTPOSQUERY)
            {
            active_count--;            /* Remove non-match possibility */
            next_active_state--;
            }
          ADD_NEW(state_offset + dlen + 1, 0);
          }
        }
      break;

      /*-----------------------------------------------------------------*/
      case OP_STARI:
      case OP_MINSTARI:
      case OP_POSSTARI:
      case OP_NOTSTARI:
      case OP_NOTMINSTARI:
      case OP_NOTPOSSTARI:
      caseless = TRUE;
      codevalue -= OP_STARI - OP_STAR;
      /* Fall through */
      case OP_STAR:
      case OP_MINSTAR:
      case OP_POSSTAR:
      case OP_NOTSTAR:
      case OP_NOTMINSTAR:
      case OP_NOTPOSSTAR:
      ADD_ACTIVE(state_offset + dlen + 1, 0);
      if (clen > 0)
        {
        uint32_t otherd = NOTACHAR;
        if (caseless)
          {
#ifdef SUPPORT_UNICODE
          if (utf && d >= 128)
            otherd = UCD_OTHERCASE(d);
          else
#endif  /* SUPPORT_UNICODE */
          otherd = TABLE_GET(d, fcc, d);
          }
        if ((c == d || c == otherd) == (codevalue < OP_NOTSTAR))
          {
          if (codevalue == OP_POSSTAR || codevalue == OP_NOTPOSSTAR)
            {
            active_count--;            /* Remove non-match possibility */
            next_active_state--;
            }
          ADD_NEW(state_offset, 0);
          }
        }
      break;

      /*-----------------------------------------------------------------*/
      case OP_EXACTI:
      case OP_NOTEXACTI:
      caseless = TRUE;
      codevalue -= OP_STARI - OP_STAR;
      /* Fall through */
      case OP_EXACT:
      case OP_NOTEXACT:
      count = current_state->count;  /* Number already matched */
      if (clen > 0)
        {
        uint32_t otherd = NOTACHAR;
        if (caseless)
          {
#ifdef SUPPORT_UNICODE
          if (utf && d >= 128)
            otherd = UCD_OTHERCASE(d);
          else
#endif  /* SUPPORT_UNICODE */
          otherd = TABLE_GET(d, fcc, d);
          }
        if ((c == d || c == otherd) == (codevalue < OP_NOTSTAR))
          {
          if (++count >= (int)GET2(code, 1))
            { ADD_NEW(state_offset + dlen + 1 + IMM2_SIZE, 0); }
          else
            { ADD_NEW(state_offset, count); }
          }
        }
      break;

      /*-----------------------------------------------------------------*/
      case OP_UPTOI:
      case OP_MINUPTOI:
      case OP_POSUPTOI:
      case OP_NOTUPTOI:
      case OP_NOTMINUPTOI:
      case OP_NOTPOSUPTOI:
      caseless = TRUE;
      codevalue -= OP_STARI - OP_STAR;
      /* Fall through */
      case OP_UPTO:
      case OP_MINUPTO:
      case OP_POSUPTO:
      case OP_NOTUPTO:
      case OP_NOTMINUPTO:
      case OP_NOTPOSUPTO:
      ADD_ACTIVE(state_offset + dlen + 1 + IMM2_SIZE, 0);
      count = current_state->count;  /* Number already matched */
      if (clen > 0)
        {
        uint32_t otherd = NOTACHAR;
        if (caseless)
          {
#ifdef SUPPORT_UNICODE
          if (utf && d >= 128)
            otherd = UCD_OTHERCASE(d);
          else
#endif  /* SUPPORT_UNICODE */
          otherd = TABLE_GET(d, fcc, d);
          }
        if ((c == d || c == otherd) == (codevalue < OP_NOTSTAR))
          {
          if (codevalue == OP_POSUPTO || codevalue == OP_NOTPOSUPTO)
            {
            active_count--;             /* Remove non-match possibility */
            next_active_state--;
            }
          if (++count >= (int)GET2(code, 1))
            { ADD_NEW(state_offset + dlen + 1 + IMM2_SIZE, 0); }
          else
            { ADD_NEW(state_offset, count); }
          }
        }
      break;


/* ========================================================================== */
      /* These are the class-handling opcodes */

      case OP_CLASS:
      case OP_NCLASS:
      case OP_XCLASS:
        {
        BOOL isinclass = FALSE;
        int next_state_offset;
        PCRE2_SPTR ecode;

        /* For a simple class, there is always just a 32-byte table, and we
        can set isinclass from it. */

        if (codevalue != OP_XCLASS)
          {
          ecode = code + 1 + (32 / sizeof(PCRE2_UCHAR));
          if (clen > 0)
            {
            isinclass = (c > 255)? (codevalue == OP_NCLASS) :
              ((((uint8_t *)(code + 1))[c/8] & (1 << (c&7))) != 0);
            }
          }

        /* An extended class may have a table or a list of single characters,
        ranges, or both, and it may be positive or negative. There's a
        function that sorts all this out. */

        else
         {
         ecode = code + GET(code, 1);
         if (clen > 0) isinclass = PRIV(xclass)(c, code + 1 + LINK_SIZE, utf);
         }

        /* At this point, isinclass is set for all kinds of class, and ecode
        points to the byte after the end of the class. If there is a
        quantifier, this is where it will be. */

        next_state_offset = (int)(ecode - start_code);

        switch (*ecode)
          {
          case OP_CRSTAR:
          case OP_CRMINSTAR:
          case OP_CRPOSSTAR:
          ADD_ACTIVE(next_state_offset + 1, 0);
          if (isinclass)
            {
            if (*ecode == OP_CRPOSSTAR)
              {
              active_count--;           /* Remove non-match possibility */
              next_active_state--;
              }
            ADD_NEW(state_offset, 0);
            }
          break;

          case OP_CRPLUS:
          case OP_CRMINPLUS:
          case OP_CRPOSPLUS:
          count = current_state->count;  /* Already matched */
          if (count > 0) { ADD_ACTIVE(next_state_offset + 1, 0); }
          if (isinclass)
            {
            if (count > 0 && *ecode == OP_CRPOSPLUS)
              {
              active_count--;           /* Remove non-match possibility */
              next_active_state--;
              }
            count++;
            ADD_NEW(state_offset, count);
            }
          break;

          case OP_CRQUERY:
          case OP_CRMINQUERY:
          case OP_CRPOSQUERY:
          ADD_ACTIVE(next_state_offset + 1, 0);
          if (isinclass)
            {
            if (*ecode == OP_CRPOSQUERY)
              {
              active_count--;           /* Remove non-match possibility */
              next_active_state--;
              }
            ADD_NEW(next_state_offset + 1, 0);
            }
          break;

          case OP_CRRANGE:
          case OP_CRMINRANGE:
          case OP_CRPOSRANGE:
          count = current_state->count;  /* Already matched */
          if (count >= (int)GET2(ecode, 1))
            { ADD_ACTIVE(next_state_offset + 1 + 2 * IMM2_SIZE, 0); }
          if (isinclass)
            {
            int max = (int)GET2(ecode, 1 + IMM2_SIZE);

            if (*ecode == OP_CRPOSRANGE && count >= (int)GET2(ecode, 1))
              {
              active_count--;           /* Remove non-match possibility */
              next_active_state--;
              }

            if (++count >= max && max != 0)   /* Max 0 => no limit */
              { ADD_NEW(next_state_offset + 1 + 2 * IMM2_SIZE, 0); }
            else
              { ADD_NEW(state_offset, count); }
            }
          break;

          default:
          if (isinclass) { ADD_NEW(next_state_offset, 0); }
          break;
          }
        }
      break;

/* ========================================================================== */
      /* These are the opcodes for fancy brackets of various kinds. We have
      to use recursion in order to handle them. The "always failing" assertion
      (?!) is optimised to OP_FAIL when compiling, so we have to support that,
      though the other "backtracking verbs" are not supported. */

      case OP_FAIL:
      forced_fail++;    /* Count FAILs for multiple states */
      break;

      case OP_ASSERT:
      case OP_ASSERT_NOT:
      case OP_ASSERTBACK:
      case OP_ASSERTBACK_NOT:
        {
        PCRE2_SPTR endasscode = code + GET(code, 1);
        PCRE2_SIZE local_offsets[2];
        int rc;
        int local_workspace[1000];

        while (*endasscode == OP_ALT) endasscode += GET(endasscode, 1);

        rc = internal_dfa_match(
          mb,                                   /* static match data */
          code,                                 /* this subexpression's code */
          ptr,                                  /* where we currently are */
          (PCRE2_SIZE)(ptr - start_subject),    /* start offset */
          local_offsets,                        /* offset vector */
          sizeof(local_offsets)/sizeof(PCRE2_SIZE), /* size of same */
          local_workspace,                      /* workspace vector */
          sizeof(local_workspace)/sizeof(int),  /* size of same */
          rlevel);                              /* function recursion level */

        if (rc < 0 && rc != PCRE2_ERROR_NOMATCH) return rc;
        if ((rc >= 0) == (codevalue == OP_ASSERT || codevalue == OP_ASSERTBACK))
            { ADD_ACTIVE((int)(endasscode + LINK_SIZE + 1 - start_code), 0); }
        }
      break;

      /*-----------------------------------------------------------------*/
      case OP_COND:
      case OP_SCOND:
        {
        PCRE2_SIZE local_offsets[1000];
        int local_workspace[1000];
        int codelink = (int)GET(code, 1);
        PCRE2_UCHAR condcode;

        /* Because of the way auto-callout works during compile, a callout item
        is inserted between OP_COND and an assertion condition. This does not
        happen for the other conditions. */

        if (code[LINK_SIZE + 1] == OP_CALLOUT
            || code[LINK_SIZE + 1] == OP_CALLOUT_STR)
          {
          PCRE2_SIZE callout_length;
          rrc = do_callout(code, offsets, current_subject, ptr, mb,
            1 + LINK_SIZE, &callout_length);
          if (rrc < 0) return rrc;                 /* Abandon */
          if (rrc > 0) break;                      /* Fail this thread */
          code += callout_length;                  /* Skip callout data */
          }

        condcode = code[LINK_SIZE+1];

        /* Back reference conditions and duplicate named recursion conditions
        are not supported */

        if (condcode == OP_CREF || condcode == OP_DNCREF ||
            condcode == OP_DNRREF)
          return PCRE2_ERROR_DFA_UCOND;

        /* The DEFINE condition is always false, and the assertion (?!) is
        converted to OP_FAIL. */

        if (condcode == OP_FALSE || condcode == OP_FAIL)
          { ADD_ACTIVE(state_offset + codelink + LINK_SIZE + 1, 0); }

        /* There is also an always-true condition */

        else if (condcode == OP_TRUE)
          { ADD_ACTIVE(state_offset + LINK_SIZE + 2 + IMM2_SIZE, 0); }

        /* The only supported version of OP_RREF is for the value RREF_ANY,
        which means "test if in any recursion". We can't test for specifically
        recursed groups. */

        else if (condcode == OP_RREF)
          {
          unsigned int value = GET2(code, LINK_SIZE + 2);
          if (value != RREF_ANY) return PCRE2_ERROR_DFA_UCOND;
          if (mb->recursive != NULL)
            { ADD_ACTIVE(state_offset + LINK_SIZE + 2 + IMM2_SIZE, 0); }
          else { ADD_ACTIVE(state_offset + codelink + LINK_SIZE + 1, 0); }
          }

        /* Otherwise, the condition is an assertion */

        else
          {
          int rc;
          PCRE2_SPTR asscode = code + LINK_SIZE + 1;
          PCRE2_SPTR endasscode = asscode + GET(asscode, 1);

          while (*endasscode == OP_ALT) endasscode += GET(endasscode, 1);

          rc = internal_dfa_match(
            mb,                                   /* fixed match data */
            asscode,                              /* this subexpression's code */
            ptr,                                  /* where we currently are */
            (PCRE2_SIZE)(ptr - start_subject),    /* start offset */
            local_offsets,                        /* offset vector */
            sizeof(local_offsets)/sizeof(PCRE2_SIZE), /* size of same */
            local_workspace,                      /* workspace vector */
            sizeof(local_workspace)/sizeof(int),  /* size of same */
            rlevel);                              /* function recursion level */

          if (rc < 0 && rc != PCRE2_ERROR_NOMATCH) return rc;
          if ((rc >= 0) ==
                (condcode == OP_ASSERT || condcode == OP_ASSERTBACK))
            { ADD_ACTIVE((int)(endasscode + LINK_SIZE + 1 - start_code), 0); }
          else
            { ADD_ACTIVE(state_offset + codelink + LINK_SIZE + 1, 0); }
          }
        }
      break;

      /*-----------------------------------------------------------------*/
      case OP_RECURSE:
        {
        dfa_recursion_info *ri;
        PCRE2_SIZE local_offsets[1000];
        int local_workspace[1000];
        PCRE2_SPTR callpat = start_code + GET(code, 1);
        uint32_t recno = (callpat == mb->start_code)? 0 :
          GET2(callpat, 1 + LINK_SIZE);
        int rc;

        /* Check for repeating a recursion without advancing the subject
        pointer. This should catch convoluted mutual recursions. (Some simple
        cases are caught at compile time.) */

        for (ri = mb->recursive; ri != NULL; ri = ri->prevrec)
          if (recno == ri->group_num && ptr == ri->subject_position)
            return PCRE2_ERROR_RECURSELOOP;

        /* Remember this recursion and where we started it so as to
        catch infinite loops. */

        new_recursive.group_num = recno;
        new_recursive.subject_position = ptr;
        new_recursive.prevrec = mb->recursive;
        mb->recursive = &new_recursive;

        rc = internal_dfa_match(
          mb,                                   /* fixed match data */
          callpat,                              /* this subexpression's code */
          ptr,                                  /* where we currently are */
          (PCRE2_SIZE)(ptr - start_subject),    /* start offset */
          local_offsets,                        /* offset vector */
          sizeof(local_offsets)/sizeof(PCRE2_SIZE), /* size of same */
          local_workspace,                      /* workspace vector */
          sizeof(local_workspace)/sizeof(int),  /* size of same */
          rlevel);                              /* function recursion level */

        mb->recursive = new_recursive.prevrec;  /* Done this recursion */

        /* Ran out of internal offsets */

        if (rc == 0) return PCRE2_ERROR_DFA_RECURSE;

        /* For each successful matched substring, set up the next state with a
        count of characters to skip before trying it. Note that the count is in
        characters, not bytes. */

        if (rc > 0)
          {
          for (rc = rc*2 - 2; rc >= 0; rc -= 2)
            {
            PCRE2_SIZE charcount = local_offsets[rc+1] - local_offsets[rc];
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
            if (utf)
              {
              PCRE2_SPTR p = start_subject + local_offsets[rc];
              PCRE2_SPTR pp = start_subject + local_offsets[rc+1];
              while (p < pp) if (NOT_FIRSTCU(*p++)) charcount--;
              }
#endif
            if (charcount > 0)
              {
              ADD_NEW_DATA(-(state_offset + LINK_SIZE + 1), 0,
                (int)(charcount - 1));
              }
            else
              {
              ADD_ACTIVE(state_offset + LINK_SIZE + 1, 0);
              }
            }
          }
        else if (rc != PCRE2_ERROR_NOMATCH) return rc;
        }
      break;

      /*-----------------------------------------------------------------*/
      case OP_BRAPOS:
      case OP_SBRAPOS:
      case OP_CBRAPOS:
      case OP_SCBRAPOS:
      case OP_BRAPOSZERO:
        {
        PCRE2_SIZE charcount, matched_count;
        PCRE2_SPTR local_ptr = ptr;
        BOOL allow_zero;

        if (codevalue == OP_BRAPOSZERO)
          {
          allow_zero = TRUE;
          codevalue = *(++code);  /* Codevalue will be one of above BRAs */
          }
        else allow_zero = FALSE;

        /* Loop to match the subpattern as many times as possible as if it were
        a complete pattern. */

        for (matched_count = 0;; matched_count++)
          {
          PCRE2_SIZE local_offsets[2];
          int local_workspace[1000];

          int rc = internal_dfa_match(
            mb,                                   /* fixed match data */
            code,                                 /* this subexpression's code */
            local_ptr,                            /* where we currently are */
            (PCRE2_SIZE)(ptr - start_subject),    /* start offset */
            local_offsets,                        /* offset vector */
            sizeof(local_offsets)/sizeof(PCRE2_SIZE), /* size of same */
            local_workspace,                      /* workspace vector */
            sizeof(local_workspace)/sizeof(int),  /* size of same */
            rlevel);                              /* function recursion level */

          /* Failed to match */

          if (rc < 0)
            {
            if (rc != PCRE2_ERROR_NOMATCH) return rc;
            break;
            }

          /* Matched: break the loop if zero characters matched. */

          charcount = local_offsets[1] - local_offsets[0];
          if (charcount == 0) break;
          local_ptr += charcount;    /* Advance temporary position ptr */
          }

        /* At this point we have matched the subpattern matched_count
        times, and local_ptr is pointing to the character after the end of the
        last match. */

        if (matched_count > 0 || allow_zero)
          {
          PCRE2_SPTR end_subpattern = code;
          int next_state_offset;

          do { end_subpattern += GET(end_subpattern, 1); }
            while (*end_subpattern == OP_ALT);
          next_state_offset =
            (int)(end_subpattern - start_code + LINK_SIZE + 1);

          /* Optimization: if there are no more active states, and there
          are no new states yet set up, then skip over the subject string
          right here, to save looping. Otherwise, set up the new state to swing
          into action when the end of the matched substring is reached. */

          if (i + 1 >= active_count && new_count == 0)
            {
            ptr = local_ptr;
            clen = 0;
            ADD_NEW(next_state_offset, 0);
            }
          else
            {
            PCRE2_SPTR p = ptr;
            PCRE2_SPTR pp = local_ptr;
            charcount = (PCRE2_SIZE)(pp - p);
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
            if (utf) while (p < pp) if (NOT_FIRSTCU(*p++)) charcount--;
#endif
            ADD_NEW_DATA(-next_state_offset, 0, (int)(charcount - 1));
            }
          }
        }
      break;

      /*-----------------------------------------------------------------*/
      case OP_ONCE:
        {
        PCRE2_SIZE local_offsets[2];
        int local_workspace[1000];

        int rc = internal_dfa_match(
          mb,                                   /* fixed match data */
          code,                                 /* this subexpression's code */
          ptr,                                  /* where we currently are */
          (PCRE2_SIZE)(ptr - start_subject),    /* start offset */
          local_offsets,                        /* offset vector */
          sizeof(local_offsets)/sizeof(PCRE2_SIZE), /* size of same */
          local_workspace,                      /* workspace vector */
          sizeof(local_workspace)/sizeof(int),  /* size of same */
          rlevel);                              /* function recursion level */

        if (rc >= 0)
          {
          PCRE2_SPTR end_subpattern = code;
          PCRE2_SIZE charcount = local_offsets[1] - local_offsets[0];
          int next_state_offset, repeat_state_offset;

          do { end_subpattern += GET(end_subpattern, 1); }
            while (*end_subpattern == OP_ALT);
          next_state_offset =
            (int)(end_subpattern - start_code + LINK_SIZE + 1);

          /* If the end of this subpattern is KETRMAX or KETRMIN, we must
          arrange for the repeat state also to be added to the relevant list.
          Calculate the offset, or set -1 for no repeat. */

          repeat_state_offset = (*end_subpattern == OP_KETRMAX ||
                                 *end_subpattern == OP_KETRMIN)?
            (int)(end_subpattern - start_code - GET(end_subpattern, 1)) : -1;

          /* If we have matched an empty string, add the next state at the
          current character pointer. This is important so that the duplicate
          checking kicks in, which is what breaks infinite loops that match an
          empty string. */

          if (charcount == 0)
            {
            ADD_ACTIVE(next_state_offset, 0);
            }

          /* Optimization: if there are no more active states, and there
          are no new states yet set up, then skip over the subject string
          right here, to save looping. Otherwise, set up the new state to swing
          into action when the end of the matched substring is reached. */

          else if (i + 1 >= active_count && new_count == 0)
            {
            ptr += charcount;
            clen = 0;
            ADD_NEW(next_state_offset, 0);

            /* If we are adding a repeat state at the new character position,
            we must fudge things so that it is the only current state.
            Otherwise, it might be a duplicate of one we processed before, and
            that would cause it to be skipped. */

            if (repeat_state_offset >= 0)
              {
              next_active_state = active_states;
              active_count = 0;
              i = -1;
              ADD_ACTIVE(repeat_state_offset, 0);
              }
            }
          else
            {
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 32
            if (utf)
              {
              PCRE2_SPTR p = start_subject + local_offsets[0];
              PCRE2_SPTR pp = start_subject + local_offsets[1];
              while (p < pp) if (NOT_FIRSTCU(*p++)) charcount--;
              }
#endif
            ADD_NEW_DATA(-next_state_offset, 0, (int)(charcount - 1));
            if (repeat_state_offset >= 0)
              { ADD_NEW_DATA(-repeat_state_offset, 0, (int)(charcount - 1)); }
            }
          }
        else if (rc != PCRE2_ERROR_NOMATCH) return rc;
        }
      break;


/* ========================================================================== */
      /* Handle callouts */

      case OP_CALLOUT:
      case OP_CALLOUT_STR:
        {
        PCRE2_SIZE callout_length;
        rrc = do_callout(code, offsets, current_subject, ptr, mb, 0,
          &callout_length);
        if (rrc < 0) return rrc;   /* Abandon */
        if (rrc == 0)
          { ADD_ACTIVE(state_offset + (int)callout_length, 0); }
        }
      break;


/* ========================================================================== */
      default:        /* Unsupported opcode */
      return PCRE2_ERROR_DFA_UITEM;
      }

    NEXT_ACTIVE_STATE: continue;

    }      /* End of loop scanning active states */

  /* We have finished the processing at the current subject character. If no
  new states have been set for the next character, we have found all the
  matches that we are going to find. If we are at the top level and partial
  matching has been requested, check for appropriate conditions.

  The "forced_ fail" variable counts the number of (*F) encountered for the
  character. If it is equal to the original active_count (saved in
  workspace[1]) it means that (*F) was found on every active state. In this
  case we don't want to give a partial match.

  The "could_continue" variable is true if a state could have continued but
  for the fact that the end of the subject was reached. */

  if (new_count <= 0)
    {
    if (rlevel == 1 &&                               /* Top level, and */
        could_continue &&                            /* Some could go on, and */
        forced_fail != workspace[1] &&               /* Not all forced fail & */
        (                                            /* either... */
        (mb->moptions & PCRE2_PARTIAL_HARD) != 0      /* Hard partial */
        ||                                           /* or... */
        ((mb->moptions & PCRE2_PARTIAL_SOFT) != 0 &&  /* Soft partial and */
         match_count < 0)                            /* no matches */
        ) &&                                         /* And... */
        (
        partial_newline ||                           /* Either partial NL */
          (                                          /* or ... */
          ptr >= end_subject &&                /* End of subject and */
          ptr > mb->start_used_ptr)            /* Inspected non-empty string */
          )
        )
      match_count = PCRE2_ERROR_PARTIAL;
    break;  /* Exit from loop along the subject string */
    }

  /* One or more states are active for the next character. */

  ptr += clen;    /* Advance to next subject character */
  }               /* Loop to move along the subject string */

/* Control gets here from "break" a few lines above. If we have a match and
PCRE2_ENDANCHORED is set, the match fails. */

if (match_count >= 0 &&
    ((mb->moptions | mb->poptions) & PCRE2_ENDANCHORED) != 0 &&
    ptr < end_subject)
  match_count = PCRE2_ERROR_NOMATCH;

return match_count;
}



/*************************************************
*     Match a pattern using the DFA algorithm    *
*************************************************/

/* This function matches a compiled pattern to a subject string, using the
alternate matching algorithm that finds all matches at once.

Arguments:
  code          points to the compiled pattern
  subject       subject string
  length        length of subject string
  startoffset   where to start matching in the subject
  options       option bits
  match_data    points to a match data structure
  gcontext      points to a match context
  workspace     pointer to workspace
  wscount       size of workspace

Returns:        > 0 => number of match offset pairs placed in offsets
                = 0 => offsets overflowed; longest matches are present
                 -1 => failed to match
               < -1 => some kind of unexpected problem
*/

PCRE2_EXP_DEFN int PCRE2_CALL_CONVENTION
pcre2_dfa_match(const pcre2_code *code, PCRE2_SPTR subject, PCRE2_SIZE length,
  PCRE2_SIZE start_offset, uint32_t options, pcre2_match_data *match_data,
  pcre2_match_context *mcontext, int *workspace, PCRE2_SIZE wscount)
{
const pcre2_real_code *re = (const pcre2_real_code *)code;

PCRE2_SPTR start_match;
PCRE2_SPTR end_subject;
PCRE2_SPTR bumpalong_limit;
PCRE2_SPTR req_cu_ptr;

BOOL utf, anchored, startline, firstline;

BOOL has_first_cu = FALSE;
BOOL has_req_cu = FALSE;
PCRE2_UCHAR first_cu = 0;
PCRE2_UCHAR first_cu2 = 0;
PCRE2_UCHAR req_cu = 0;
PCRE2_UCHAR req_cu2 = 0;

const uint8_t *start_bits = NULL;

/* We need to have mb pointing to a match block, because the IS_NEWLINE macro
is used below, and it expects NLBLOCK to be defined as a pointer. */

pcre2_callout_block cb;
dfa_match_block actual_match_block;
dfa_match_block *mb = &actual_match_block;

/* A length equal to PCRE2_ZERO_TERMINATED implies a zero-terminated
subject string. */

if (length == PCRE2_ZERO_TERMINATED) length = PRIV(strlen)(subject);

/* Plausibility checks */

if ((options & ~PUBLIC_DFA_MATCH_OPTIONS) != 0) return PCRE2_ERROR_BADOPTION;
if (re == NULL || subject == NULL || workspace == NULL || match_data == NULL)
  return PCRE2_ERROR_NULL;
if (wscount < 20) return PCRE2_ERROR_DFA_WSSIZE;
if (start_offset > length) return PCRE2_ERROR_BADOFFSET;

/* Partial matching and PCRE2_ENDANCHORED are currently not allowed at the same
time. */

if ((options & (PCRE2_PARTIAL_HARD|PCRE2_PARTIAL_SOFT)) != 0 &&
   ((re->overall_options | options) & PCRE2_ENDANCHORED) != 0)
  return PCRE2_ERROR_BADOPTION;

/* Check that the first field in the block is the magic number. If it is not,
return with PCRE2_ERROR_BADMAGIC. */

if (re->magic_number != MAGIC_NUMBER) return PCRE2_ERROR_BADMAGIC;

/* Check the code unit width. */

if ((re->flags & PCRE2_MODE_MASK) != PCRE2_CODE_UNIT_WIDTH/8)
  return PCRE2_ERROR_BADMODE;

/* PCRE2_NOTEMPTY and PCRE2_NOTEMPTY_ATSTART are match-time flags in the
options variable for this function. Users of PCRE2 who are not calling the
function directly would like to have a way of setting these flags, in the same
way that they can set pcre2_compile() flags like PCRE2_NO_AUTOPOSSESS with
constructions like (*NO_AUTOPOSSESS). To enable this, (*NOTEMPTY) and
(*NOTEMPTY_ATSTART) set bits in the pattern's "flag" function which can now be
transferred to the options for this function. The bits are guaranteed to be
adjacent, but do not have the same values. This bit of Boolean trickery assumes
that the match-time bits are not more significant than the flag bits. If by
accident this is not the case, a compile-time division by zero error will
occur. */

#define FF (PCRE2_NOTEMPTY_SET|PCRE2_NE_ATST_SET)
#define OO (PCRE2_NOTEMPTY|PCRE2_NOTEMPTY_ATSTART)
options |= (re->flags & FF) / ((FF & (~FF+1)) / (OO & (~OO+1)));
#undef FF
#undef OO

/* If restarting after a partial match, do some sanity checks on the contents
of the workspace. */

if ((options & PCRE2_DFA_RESTART) != 0)
  {
  if ((workspace[0] & (-2)) != 0 || workspace[1] < 1 ||
    workspace[1] > (int)((wscount - 2)/INTS_PER_STATEBLOCK))
      return PCRE2_ERROR_DFA_BADRESTART;
  }

/* Set some local values */

utf = (re->overall_options & PCRE2_UTF) != 0;
start_match = subject + start_offset;
end_subject = subject + length;
req_cu_ptr = start_match - 1;
anchored = (options & (PCRE2_ANCHORED|PCRE2_DFA_RESTART)) != 0 ||
  (re->overall_options & PCRE2_ANCHORED) != 0;

/* The "must be at the start of a line" flags are used in a loop when finding
where to start. */

startline = (re->flags & PCRE2_STARTLINE) != 0;
firstline = (re->overall_options & PCRE2_FIRSTLINE) != 0;
bumpalong_limit = end_subject;

/* Initialize and set up the fixed fields in the callout block, with a pointer
in the match block. */

mb->cb = &cb;
cb.version = 2;
cb.subject = subject;
cb.subject_length = (PCRE2_SIZE)(end_subject - subject);
cb.callout_flags = 0;
cb.capture_top      = 1;      /* No capture support */
cb.capture_last     = 0;
cb.mark             = NULL;   /* No (*MARK) support */

/* Get data from the match context, if present, and fill in the remaining
fields in the match block. It is an error to set an offset limit without
setting the flag at compile time. */

if (mcontext == NULL)
  {
  mb->callout = NULL;
  mb->memctl = re->memctl;
  mb->match_limit = PRIV(default_match_context).match_limit;
  mb->match_limit_depth = PRIV(default_match_context).depth_limit;
  }
else
  {
  if (mcontext->offset_limit != PCRE2_UNSET)
    {
    if ((re->overall_options & PCRE2_USE_OFFSET_LIMIT) == 0)
      return PCRE2_ERROR_BADOFFSETLIMIT;
    bumpalong_limit = subject + mcontext->offset_limit;
    }
  mb->callout = mcontext->callout;
  mb->callout_data = mcontext->callout_data;
  mb->memctl = mcontext->memctl;
  mb->match_limit = mcontext->match_limit;
  mb->match_limit_depth = mcontext->depth_limit;
  }

if (mb->match_limit > re->limit_match)
  mb->match_limit = re->limit_match;

if (mb->match_limit_depth > re->limit_depth)
  mb->match_limit_depth = re->limit_depth;

mb->start_code = (PCRE2_UCHAR *)((uint8_t *)re + sizeof(pcre2_real_code)) +
  re->name_count * re->name_entry_size;
mb->tables = re->tables;
mb->start_subject = subject;
mb->end_subject = end_subject;
mb->start_offset = start_offset;
mb->moptions = options;
mb->poptions = re->overall_options;
mb->match_call_count = 0;

/* Process the \R and newline settings. */

mb->bsr_convention = re->bsr_convention;
mb->nltype = NLTYPE_FIXED;
switch(re->newline_convention)
  {
  case PCRE2_NEWLINE_CR:
  mb->nllen = 1;
  mb->nl[0] = CHAR_CR;
  break;

  case PCRE2_NEWLINE_LF:
  mb->nllen = 1;
  mb->nl[0] = CHAR_NL;
  break;

  case PCRE2_NEWLINE_NUL:
  mb->nllen = 1;
  mb->nl[0] = CHAR_NUL;
  break;

  case PCRE2_NEWLINE_CRLF:
  mb->nllen = 2;
  mb->nl[0] = CHAR_CR;
  mb->nl[1] = CHAR_NL;
  break;

  case PCRE2_NEWLINE_ANY:
  mb->nltype = NLTYPE_ANY;
  break;

  case PCRE2_NEWLINE_ANYCRLF:
  mb->nltype = NLTYPE_ANYCRLF;
  break;

  default: return PCRE2_ERROR_INTERNAL;
  }

/* Check a UTF string for validity if required. For 8-bit and 16-bit strings,
we must also check that a starting offset does not point into the middle of a
multiunit character. We check only the portion of the subject that is going to
be inspected during matching - from the offset minus the maximum back reference
to the given length. This saves time when a small part of a large subject is
being matched by the use of a starting offset. Note that the maximum lookbehind
is a number of characters, not code units. */

#ifdef SUPPORT_UNICODE
if (utf && (options & PCRE2_NO_UTF_CHECK) == 0)
  {
  PCRE2_SPTR check_subject = start_match;  /* start_match includes offset */

  if (start_offset > 0)
    {
#if PCRE2_CODE_UNIT_WIDTH != 32
    unsigned int i;
    if (start_match < end_subject && NOT_FIRSTCU(*start_match))
      return PCRE2_ERROR_BADUTFOFFSET;
    for (i = re->max_lookbehind; i > 0 && check_subject > subject; i--)
      {
      check_subject--;
      while (check_subject > subject &&
#if PCRE2_CODE_UNIT_WIDTH == 8
      (*check_subject & 0xc0) == 0x80)
#else  /* 16-bit */
      (*check_subject & 0xfc00) == 0xdc00)
#endif /* PCRE2_CODE_UNIT_WIDTH == 8 */
        check_subject--;
      }
#else   /* In the 32-bit library, one code unit equals one character. */
    check_subject -= re->max_lookbehind;
    if (check_subject < subject) check_subject = subject;
#endif  /* PCRE2_CODE_UNIT_WIDTH != 32 */
    }

  /* Validate the relevant portion of the subject. After an error, adjust the
  offset to be an absolute offset in the whole string. */

  match_data->rc = PRIV(valid_utf)(check_subject,
    length - (PCRE2_SIZE)(check_subject - subject), &(match_data->startchar));
  if (match_data->rc != 0)
    {
    match_data->startchar += (PCRE2_SIZE)(check_subject - subject);
    return match_data->rc;
    }
  }
#endif  /* SUPPORT_UNICODE */

/* Set up the first code unit to match, if available. If there's no first code
unit there may be a bitmap of possible first characters. */

if ((re->flags & PCRE2_FIRSTSET) != 0)
  {
  has_first_cu = TRUE;
  first_cu = first_cu2 = (PCRE2_UCHAR)(re->first_codeunit);
  if ((re->flags & PCRE2_FIRSTCASELESS) != 0)
    {
    first_cu2 = TABLE_GET(first_cu, mb->tables + fcc_offset, first_cu);
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 8
    if (utf && first_cu > 127)
      first_cu2 = (PCRE2_UCHAR)UCD_OTHERCASE(first_cu);
#endif
    }
  }
else
  if (!startline && (re->flags & PCRE2_FIRSTMAPSET) != 0)
    start_bits = re->start_bitmap;

/* There may be a "last known required code unit" set. */

if ((re->flags & PCRE2_LASTSET) != 0)
  {
  has_req_cu = TRUE;
  req_cu = req_cu2 = (PCRE2_UCHAR)(re->last_codeunit);
  if ((re->flags & PCRE2_LASTCASELESS) != 0)
    {
    req_cu2 = TABLE_GET(req_cu, mb->tables + fcc_offset, req_cu);
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 8
    if (utf && req_cu > 127) req_cu2 = (PCRE2_UCHAR)UCD_OTHERCASE(req_cu);
#endif
    }
  }

/* Fill in fields that are always returned in the match data. */

match_data->code = re;
match_data->subject = subject;
match_data->mark = NULL;
match_data->matchedby = PCRE2_MATCHEDBY_DFA_INTERPRETER;

/* Call the main matching function, looping for a non-anchored regex after a
failed match. If not restarting, perform certain optimizations at the start of
a match. */

for (;;)
  {
  int rc;

  /* ----------------- Start of match optimizations ---------------- */

  /* There are some optimizations that avoid running the match if a known
  starting point is not found, or if a known later code unit is not present.
  However, there is an option (settable at compile time) that disables
  these, for testing and for ensuring that all callouts do actually occur.
  The optimizations must also be avoided when restarting a DFA match. */

  if ((re->overall_options & PCRE2_NO_START_OPTIMIZE) == 0 &&
      (options & PCRE2_DFA_RESTART) == 0)
    {
    /* If firstline is TRUE, the start of the match is constrained to the first
    line of a multiline string. That is, the match must be before or at the
    first newline following the start of matching. Temporarily adjust
    end_subject so that we stop the optimization scans for a first code unit
    immediately after the first character of a newline (the first code unit can
    legitimately be a newline). If the match fails at the newline, later code
    breaks this loop. */

    if (firstline)
      {
      PCRE2_SPTR t = start_match;
#ifdef SUPPORT_UNICODE
      if (utf)
        {
        while (t < end_subject && !IS_NEWLINE(t))
          {
          t++;
          ACROSSCHAR(t < end_subject, t, t++);
          }
        }
      else
#endif
      while (t < end_subject && !IS_NEWLINE(t)) t++;
      end_subject = t;
      }

    /* Anchored: check the first code unit if one is recorded. This may seem
    pointless but it can help in detecting a no match case without scanning for
    the required code unit. */

    if (anchored)
      {
      if (has_first_cu || start_bits != NULL)
        {
        BOOL ok = start_match < end_subject;
        if (ok)
          {
          PCRE2_UCHAR c = UCHAR21TEST(start_match);
          ok = has_first_cu && (c == first_cu || c == first_cu2);
          if (!ok && start_bits != NULL)
            {
#if PCRE2_CODE_UNIT_WIDTH != 8
            if (c > 255) c = 255;
#endif
            ok = (start_bits[c/8] & (1 << (c&7))) != 0;
            }
          }
        if (!ok) break;
        }
      }

    /* Not anchored. Advance to a unique first code unit if there is one. In
    8-bit mode, the use of memchr() gives a big speed up, even though we have
    to call it twice in caseless mode, in order to find the earliest occurrence
    of the character in either of its cases. */

    else
      {
      if (has_first_cu)
        {
        if (first_cu != first_cu2)  /* Caseless */
          {
#if PCRE2_CODE_UNIT_WIDTH != 8
          PCRE2_UCHAR smc;
          while (start_match < end_subject &&
                (smc = UCHAR21TEST(start_match)) != first_cu &&
                  smc != first_cu2)
            start_match++;
#else  /* 8-bit code units */
          PCRE2_SPTR pp1 =
            memchr(start_match, first_cu, end_subject-start_match);
          PCRE2_SPTR pp2 =
            memchr(start_match, first_cu2, end_subject-start_match);
          if (pp1 == NULL)
            start_match = (pp2 == NULL)? end_subject : pp2;
          else
            start_match = (pp2 == NULL || pp1 < pp2)? pp1 : pp2;
#endif
          }

        /* The caseful case */

        else
          {
#if PCRE2_CODE_UNIT_WIDTH != 8
          while (start_match < end_subject && UCHAR21TEST(start_match) !=
                 first_cu)
            start_match++;
#else
          start_match = memchr(start_match, first_cu, end_subject - start_match);
          if (start_match == NULL) start_match = end_subject;
#endif
          }

        /* If we can't find the required code unit, having reached the true end
        of the subject, break the bumpalong loop, to force a match failure,
        except when doing partial matching, when we let the next cycle run at
        the end of the subject. To see why, consider the pattern /(?<=abc)def/,
        which partially matches "abc", even though the string does not contain
        the starting character "d". If we have not reached the true end of the
        subject (PCRE2_FIRSTLINE caused end_subject to be temporarily modified)
        we also let the cycle run, because the matching string is legitimately
        allowed to start with the first code unit of a newline. */

        if ((mb->moptions & (PCRE2_PARTIAL_HARD|PCRE2_PARTIAL_SOFT)) == 0 &&
            start_match >= mb->end_subject)
          break;
        }

      /* If there's no first code unit, advance to just after a linebreak for a
      multiline match if required. */

      else if (startline)
        {
        if (start_match > mb->start_subject + start_offset)
          {
#ifdef SUPPORT_UNICODE
          if (utf)
            {
            while (start_match < end_subject && !WAS_NEWLINE(start_match))
              {
              start_match++;
              ACROSSCHAR(start_match < end_subject, start_match, start_match++);
              }
            }
          else
#endif
          while (start_match < end_subject && !WAS_NEWLINE(start_match))
            start_match++;

          /* If we have just passed a CR and the newline option is ANY or
          ANYCRLF, and we are now at a LF, advance the match position by one
          more code unit. */

          if (start_match[-1] == CHAR_CR &&
               (mb->nltype == NLTYPE_ANY || mb->nltype == NLTYPE_ANYCRLF) &&
               start_match < end_subject &&
               UCHAR21TEST(start_match) == CHAR_NL)
            start_match++;
          }
        }

      /* If there's no first code unit or a requirement for a multiline line
      start, advance to a non-unique first code unit if any have been
      identified. The bitmap contains only 256 bits. When code units are 16 or
      32 bits wide, all code units greater than 254 set the 255 bit. */

      else if (start_bits != NULL)
        {
        while (start_match < end_subject)
          {
          uint32_t c = UCHAR21TEST(start_match);
#if PCRE2_CODE_UNIT_WIDTH != 8
          if (c > 255) c = 255;
#endif
          if ((start_bits[c/8] & (1 << (c&7))) != 0) break;
          start_match++;
          }

        /* See comment above in first_cu checking about the next line. */

        if ((mb->moptions & (PCRE2_PARTIAL_HARD|PCRE2_PARTIAL_SOFT)) == 0 &&
            start_match >= mb->end_subject)
          break;
        }
      }  /* End of first code unit handling */

    /* Restore fudged end_subject */

    end_subject = mb->end_subject;

    /* The following two optimizations are disabled for partial matching. */

    if ((mb->moptions & (PCRE2_PARTIAL_HARD|PCRE2_PARTIAL_SOFT)) == 0)
      {
      /* The minimum matching length is a lower bound; no actual string of that
      length may actually match the pattern. Although the value is, strictly,
      in characters, we treat it as code units to avoid spending too much time
      in this optimization. */

      if (end_subject - start_match < re->minlength) return PCRE2_ERROR_NOMATCH;

      /* If req_cu is set, we know that that code unit must appear in the
      subject for the match to succeed. If the first code unit is set, req_cu
      must be later in the subject; otherwise the test starts at the match
      point. This optimization can save a huge amount of backtracking in
      patterns with nested unlimited repeats that aren't going to match.
      Writing separate code for cased/caseless versions makes it go faster, as
      does using an autoincrement and backing off on a match.

      HOWEVER: when the subject string is very, very long, searching to its end
      can take a long time, and give bad performance on quite ordinary
      patterns. This showed up when somebody was matching something like
      /^\d+C/ on a 32-megabyte string... so we don't do this when the string is
      sufficiently long. */

      if (has_req_cu && end_subject - start_match < REQ_CU_MAX)
        {
        PCRE2_SPTR p = start_match + (has_first_cu? 1:0);

        /* We don't need to repeat the search if we haven't yet reached the
        place we found it at last time. */

        if (p > req_cu_ptr)
          {
          if (req_cu != req_cu2)
            {
            while (p < end_subject)
              {
              uint32_t pp = UCHAR21INCTEST(p);
              if (pp == req_cu || pp == req_cu2) { p--; break; }
              }
            }
          else
            {
            while (p < end_subject)
              {
              if (UCHAR21INCTEST(p) == req_cu) { p--; break; }
              }
            }

          /* If we can't find the required code unit, break the matching loop,
          forcing a match failure. */

          if (p >= end_subject) break;

          /* If we have found the required code unit, save the point where we
          found it, so that we don't search again next time round the loop if
          the start hasn't passed this code unit yet. */

          req_cu_ptr = p;
          }
        }
      }
    }

  /* ------------ End of start of match optimizations ------------ */

  /* Give no match if we have passed the bumpalong limit. */

  if (start_match > bumpalong_limit) break;

  /* OK, now we can do the business */

  mb->start_used_ptr = start_match;
  mb->last_used_ptr = start_match;
  mb->recursive = NULL;

  rc = internal_dfa_match(
    mb,                           /* fixed match data */
    mb->start_code,               /* this subexpression's code */
    start_match,                  /* where we currently are */
    start_offset,                 /* start offset in subject */
    match_data->ovector,          /* offset vector */
    (uint32_t)match_data->oveccount * 2,  /* actual size of same */
    workspace,                    /* workspace vector */
    (int)wscount,                 /* size of same */
    0);                           /* function recurse level */

  /* Anything other than "no match" means we are done, always; otherwise, carry
  on only if not anchored. */

  if (rc != PCRE2_ERROR_NOMATCH || anchored)
    {
    if (rc == PCRE2_ERROR_PARTIAL && match_data->oveccount > 0)
      {
      match_data->ovector[0] = (PCRE2_SIZE)(start_match - subject);
      match_data->ovector[1] = (PCRE2_SIZE)(end_subject - subject);
      }
    match_data->leftchar = (PCRE2_SIZE)(mb->start_used_ptr - subject);
    match_data->rightchar = (PCRE2_SIZE)( mb->last_used_ptr - subject);
    match_data->startchar = (PCRE2_SIZE)(start_match - subject);
    match_data->rc = rc;
    return rc;
    }

  /* Advance to the next subject character unless we are at the end of a line
  and firstline is set. */

  if (firstline && IS_NEWLINE(start_match)) break;
  start_match++;
#ifdef SUPPORT_UNICODE
  if (utf)
    {
    ACROSSCHAR(start_match < end_subject, start_match, start_match++);
    }
#endif
  if (start_match > end_subject) break;

  /* If we have just passed a CR and we are now at a LF, and the pattern does
  not contain any explicit matches for \r or \n, and the newline option is CRLF
  or ANY or ANYCRLF, advance the match position by one more character. */

  if (UCHAR21TEST(start_match - 1) == CHAR_CR &&
      start_match < end_subject &&
      UCHAR21TEST(start_match) == CHAR_NL &&
      (re->flags & PCRE2_HASCRORLF) == 0 &&
        (mb->nltype == NLTYPE_ANY ||
         mb->nltype == NLTYPE_ANYCRLF ||
         mb->nllen == 2))
    start_match++;

  }   /* "Bumpalong" loop */


return PCRE2_ERROR_NOMATCH;
}

/* End of pcre2_dfa_match.c */
