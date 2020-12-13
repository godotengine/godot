/*************************************************
*      Perl-Compatible Regular Expressions       *
*************************************************/

/* PCRE is a library of functions to support regular expressions whose syntax
and semantics are as close as possible to those of the Perl 5 language.

                       Written by Philip Hazel
     Original API code Copyright (c) 1997-2012 University of Cambridge
          New API code Copyright (c) 2015-2019 University of Cambridge

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

/* These defines enable debugging code */

/* #define DEBUG_FRAMES_DISPLAY */
/* #define DEBUG_SHOW_OPS */
/* #define DEBUG_SHOW_RMATCH */

#ifdef DEBUG_FRAME_DISPLAY
#include <stdarg.h>
#endif

/* These defines identify the name of the block containing "static"
information, and fields within it. */

#define NLBLOCK mb              /* Block containing newline information */
#define PSSTART start_subject   /* Field containing processed string start */
#define PSEND   end_subject     /* Field containing processed string end */

#include "pcre2_internal.h"

#define RECURSE_UNSET 0xffffffffu  /* Bigger than max group number */

/* Masks for identifying the public options that are permitted at match time. */

#define PUBLIC_MATCH_OPTIONS \
  (PCRE2_ANCHORED|PCRE2_ENDANCHORED|PCRE2_NOTBOL|PCRE2_NOTEOL|PCRE2_NOTEMPTY| \
   PCRE2_NOTEMPTY_ATSTART|PCRE2_NO_UTF_CHECK|PCRE2_PARTIAL_HARD| \
   PCRE2_PARTIAL_SOFT|PCRE2_NO_JIT|PCRE2_COPY_MATCHED_SUBJECT)

#define PUBLIC_JIT_MATCH_OPTIONS \
   (PCRE2_NO_UTF_CHECK|PCRE2_NOTBOL|PCRE2_NOTEOL|PCRE2_NOTEMPTY|\
    PCRE2_NOTEMPTY_ATSTART|PCRE2_PARTIAL_SOFT|PCRE2_PARTIAL_HARD|\
    PCRE2_COPY_MATCHED_SUBJECT)

/* Non-error returns from and within the match() function. Error returns are
externally defined PCRE2_ERROR_xxx codes, which are all negative. */

#define MATCH_MATCH        1
#define MATCH_NOMATCH      0

/* Special internal returns used in the match() function. Make them
sufficiently negative to avoid the external error codes. */

#define MATCH_ACCEPT       (-999)
#define MATCH_KETRPOS      (-998)
/* The next 5 must be kept together and in sequence so that a test that checks
for any one of them can use a range. */
#define MATCH_COMMIT       (-997)
#define MATCH_PRUNE        (-996)
#define MATCH_SKIP         (-995)
#define MATCH_SKIP_ARG     (-994)
#define MATCH_THEN         (-993)
#define MATCH_BACKTRACK_MAX MATCH_THEN
#define MATCH_BACKTRACK_MIN MATCH_COMMIT

/* Group frame type values. Zero means the frame is not a group frame. The
lower 16 bits are used for data (e.g. the capture number). Group frames are
used for most groups so that information about the start is easily available at
the end without having to scan back through intermediate frames (backtrack
points). */

#define GF_CAPTURE     0x00010000u
#define GF_NOCAPTURE   0x00020000u
#define GF_CONDASSERT  0x00030000u
#define GF_RECURSE     0x00040000u

/* Masks for the identity and data parts of the group frame type. */

#define GF_IDMASK(a)   ((a) & 0xffff0000u)
#define GF_DATAMASK(a) ((a) & 0x0000ffffu)

/* Repetition types */

enum { REPTYPE_MIN, REPTYPE_MAX, REPTYPE_POS };

/* Min and max values for the common repeats; a maximum of UINT32_MAX =>
infinity. */

static const uint32_t rep_min[] = {
  0, 0,       /* * and *? */
  1, 1,       /* + and +? */
  0, 0,       /* ? and ?? */
  0, 0,       /* dummy placefillers for OP_CR[MIN]RANGE */
  0, 1, 0 };  /* OP_CRPOS{STAR, PLUS, QUERY} */

static const uint32_t rep_max[] = {
  UINT32_MAX, UINT32_MAX,      /* * and *? */
  UINT32_MAX, UINT32_MAX,      /* + and +? */
  1, 1,                        /* ? and ?? */
  0, 0,                        /* dummy placefillers for OP_CR[MIN]RANGE */
  UINT32_MAX, UINT32_MAX, 1 }; /* OP_CRPOS{STAR, PLUS, QUERY} */

/* Repetition types - must include OP_CRPOSRANGE (not needed above) */

static const uint32_t rep_typ[] = {
  REPTYPE_MAX, REPTYPE_MIN,    /* * and *? */
  REPTYPE_MAX, REPTYPE_MIN,    /* + and +? */
  REPTYPE_MAX, REPTYPE_MIN,    /* ? and ?? */
  REPTYPE_MAX, REPTYPE_MIN,    /* OP_CRRANGE and OP_CRMINRANGE */
  REPTYPE_POS, REPTYPE_POS,    /* OP_CRPOSSTAR, OP_CRPOSPLUS */
  REPTYPE_POS, REPTYPE_POS };  /* OP_CRPOSQUERY, OP_CRPOSRANGE */

/* Numbers for RMATCH calls at backtracking points. When these lists are
changed, the code at RETURN_SWITCH below must be updated in sync.  */

enum { RM1=1, RM2,  RM3,  RM4,  RM5,  RM6,  RM7,  RM8,  RM9,  RM10,
       RM11,  RM12, RM13, RM14, RM15, RM16, RM17, RM18, RM19, RM20,
       RM21,  RM22, RM23, RM24, RM25, RM26, RM27, RM28, RM29, RM30,
       RM31,  RM32, RM33, RM34, RM35, RM36 };

#ifdef SUPPORT_WIDE_CHARS
enum { RM100=100, RM101 };
#endif

#ifdef SUPPORT_UNICODE
enum { RM200=200, RM201, RM202, RM203, RM204, RM205, RM206, RM207,
       RM208,     RM209, RM210, RM211, RM212, RM213, RM214, RM215,
       RM216,     RM217, RM218, RM219, RM220, RM221, RM222 };
#endif

/* Define short names for general fields in the current backtrack frame, which
is always pointed to by the F variable. Occasional references to fields in
other frames are written out explicitly. There are also some fields in the
current frame whose names start with "temp" that are used for short-term,
localised backtracking memory. These are #defined with Lxxx names at the point
of use and undefined afterwards. */

#define Fback_frame        F->back_frame
#define Fcapture_last      F->capture_last
#define Fcurrent_recurse   F->current_recurse
#define Fecode             F->ecode
#define Feptr              F->eptr
#define Fgroup_frame_type  F->group_frame_type
#define Flast_group_offset F->last_group_offset
#define Flength            F->length
#define Fmark              F->mark
#define Frdepth            F->rdepth
#define Fstart_match       F->start_match
#define Foffset_top        F->offset_top
#define Foccu              F->occu
#define Fop                F->op
#define Fovector           F->ovector
#define Freturn_id         F->return_id


#ifdef DEBUG_FRAMES_DISPLAY
/*************************************************
*      Display current frames and contents       *
*************************************************/

/* This debugging function displays the current set of frames and their
contents. It is not called automatically from anywhere, the intention being
that calls can be inserted where necessary when debugging frame-related
problems.

Arguments:
  f           the file to write to
  F           the current top frame
  P           a previous frame of interest
  frame_size  the frame size
  mb          points to the match block
  s           identification text

Returns:    nothing
*/

static void
display_frames(FILE *f, heapframe *F, heapframe *P, PCRE2_SIZE frame_size,
  match_block *mb, const char *s, ...)
{
uint32_t i;
heapframe *Q;
va_list ap;
va_start(ap, s);

fprintf(f, "FRAMES ");
vfprintf(f, s, ap);
va_end(ap);

if (P != NULL) fprintf(f, " P=%lu",
  ((char *)P - (char *)(mb->match_frames))/frame_size);
fprintf(f, "\n");

for (i = 0, Q = mb->match_frames;
     Q <= F;
     i++, Q = (heapframe *)((char *)Q + frame_size))
  {
  fprintf(f, "Frame %d type=%x subj=%lu code=%d back=%lu id=%d",
    i, Q->group_frame_type, Q->eptr - mb->start_subject, *(Q->ecode),
    Q->back_frame, Q->return_id);

  if (Q->last_group_offset == PCRE2_UNSET)
    fprintf(f, " lgoffset=unset\n");
  else
    fprintf(f, " lgoffset=%lu\n",  Q->last_group_offset/frame_size);
  }
}

#endif



/*************************************************
*                Process a callout               *
*************************************************/

/* This function is called for all callouts, whether "standalone" or at the
start of a conditional group. Feptr will be pointing to either OP_CALLOUT or
OP_CALLOUT_STR. A callout block is allocated in pcre2_match() and initialized
with fixed values.

Arguments:
  F          points to the current backtracking frame
  mb         points to the match block
  lengthptr  where to return the length of the callout item

Returns:     the return from the callout
             or 0 if no callout function exists
*/

static int
do_callout(heapframe *F, match_block *mb, PCRE2_SIZE *lengthptr)
{
int rc;
PCRE2_SIZE save0, save1;
PCRE2_SIZE *callout_ovector;
pcre2_callout_block *cb;

*lengthptr = (*Fecode == OP_CALLOUT)?
  PRIV(OP_lengths)[OP_CALLOUT] : GET(Fecode, 1 + 2*LINK_SIZE);

if (mb->callout == NULL) return 0;   /* No callout function provided */

/* The original matching code (pre 10.30) worked directly with the ovector
passed by the user, and this was passed to callouts. Now that the working
ovector is in the backtracking frame, it no longer needs to reserve space for
the overall match offsets (which would waste space in the frame). For backward
compatibility, however, we pass capture_top and offset_vector to the callout as
if for the extended ovector, and we ensure that the first two slots are unset
by preserving and restoring their current contents. Picky compilers complain if
references such as Fovector[-2] are use directly, so we set up a separate
pointer. */

callout_ovector = (PCRE2_SIZE *)(Fovector) - 2;

/* The cb->version, cb->subject, cb->subject_length, and cb->start_match fields
are set externally. The first 3 never change; the last is updated for each
bumpalong. */

cb = mb->cb;
cb->capture_top      = (uint32_t)Foffset_top/2 + 1;
cb->capture_last     = Fcapture_last;
cb->offset_vector    = callout_ovector;
cb->mark             = mb->nomatch_mark;
cb->current_position = (PCRE2_SIZE)(Feptr - mb->start_subject);
cb->pattern_position = GET(Fecode, 1);
cb->next_item_length = GET(Fecode, 1 + LINK_SIZE);

if (*Fecode == OP_CALLOUT)  /* Numerical callout */
  {
  cb->callout_number = Fecode[1 + 2*LINK_SIZE];
  cb->callout_string_offset = 0;
  cb->callout_string = NULL;
  cb->callout_string_length = 0;
  }
else  /* String callout */
  {
  cb->callout_number = 0;
  cb->callout_string_offset = GET(Fecode, 1 + 3*LINK_SIZE);
  cb->callout_string = Fecode + (1 + 4*LINK_SIZE) + 1;
  cb->callout_string_length =
    *lengthptr - (1 + 4*LINK_SIZE) - 2;
  }

save0 = callout_ovector[0];
save1 = callout_ovector[1];
callout_ovector[0] = callout_ovector[1] = PCRE2_UNSET;
rc = mb->callout(cb, mb->callout_data);
callout_ovector[0] = save0;
callout_ovector[1] = save1;
cb->callout_flags = 0;
return rc;
}



/*************************************************
*          Match a back-reference                *
*************************************************/

/* This function is called only when it is known that the offset lies within
the offsets that have so far been used in the match. Note that in caseless
UTF-8 mode, the number of subject bytes matched may be different to the number
of reference bytes. (In theory this could also happen in UTF-16 mode, but it
seems unlikely.)

Arguments:
  offset      index into the offset vector
  caseless    TRUE if caseless
  F           the current backtracking frame pointer
  mb          points to match block
  lengthptr   pointer for returning the length matched

Returns:      = 0 sucessful match; number of code units matched is set
              < 0 no match
              > 0 partial match
*/

static int
match_ref(PCRE2_SIZE offset, BOOL caseless, heapframe *F, match_block *mb,
  PCRE2_SIZE *lengthptr)
{
PCRE2_SPTR p;
PCRE2_SIZE length;
PCRE2_SPTR eptr;
PCRE2_SPTR eptr_start;

/* Deal with an unset group. The default is no match, but there is an option to
match an empty string. */

if (offset >= Foffset_top || Fovector[offset] == PCRE2_UNSET)
  {
  if ((mb->poptions & PCRE2_MATCH_UNSET_BACKREF) != 0)
    {
    *lengthptr = 0;
    return 0;      /* Match */
    }
  else return -1;  /* No match */
  }

/* Separate the caseless and UTF cases for speed. */

eptr = eptr_start = Feptr;
p = mb->start_subject + Fovector[offset];
length = Fovector[offset+1] - Fovector[offset];

if (caseless)
  {
#if defined SUPPORT_UNICODE
  if ((mb->poptions & PCRE2_UTF) != 0)
    {
    /* Match characters up to the end of the reference. NOTE: the number of
    code units matched may differ, because in UTF-8 there are some characters
    whose upper and lower case codes have different numbers of bytes. For
    example, U+023A (2 bytes in UTF-8) is the upper case version of U+2C65 (3
    bytes in UTF-8); a sequence of 3 of the former uses 6 bytes, as does a
    sequence of two of the latter. It is important, therefore, to check the
    length along the reference, not along the subject (earlier code did this
    wrong). */

    PCRE2_SPTR endptr = p + length;
    while (p < endptr)
      {
      uint32_t c, d;
      const ucd_record *ur;
      if (eptr >= mb->end_subject) return 1;   /* Partial match */
      GETCHARINC(c, eptr);
      GETCHARINC(d, p);
      ur = GET_UCD(d);
      if (c != d && c != (uint32_t)((int)d + ur->other_case))
        {
        const uint32_t *pp = PRIV(ucd_caseless_sets) + ur->caseset;
        for (;;)
          {
          if (c < *pp) return -1;  /* No match */
          if (c == *pp++) break;
          }
        }
      }
    }
  else
#endif

  /* Not in UTF mode */
    {
    for (; length > 0; length--)
      {
      uint32_t cc, cp;
      if (eptr >= mb->end_subject) return 1;   /* Partial match */
      cc = UCHAR21TEST(eptr);
      cp = UCHAR21TEST(p);
      if (TABLE_GET(cp, mb->lcc, cp) != TABLE_GET(cc, mb->lcc, cc))
        return -1;  /* No match */
      p++;
      eptr++;
      }
    }
  }

/* In the caseful case, we can just compare the code units, whether or not we
are in UTF mode. When partial matching, we have to do this unit-by-unit. */

else
  {
  if (mb->partial != 0)
    {
    for (; length > 0; length--)
      {
      if (eptr >= mb->end_subject) return 1;   /* Partial match */
      if (UCHAR21INCTEST(p) != UCHAR21INCTEST(eptr)) return -1;  /* No match */
      }
    }

  /* Not partial matching */

  else
    {
    if ((PCRE2_SIZE)(mb->end_subject - eptr) < length) return 1; /* Partial */
    if (memcmp(p, eptr, CU2BYTES(length)) != 0) return -1;  /* No match */
    eptr += length;
    }
  }

*lengthptr = eptr - eptr_start;
return 0;  /* Match */
}



/******************************************************************************
*******************************************************************************
                   "Recursion" in the match() function

The original match() function was highly recursive, but this proved to be the
source of a number of problems over the years, mostly because of the relatively
small system stacks that are commonly found. As new features were added to
patterns, various kludges were invented to reduce the amount of stack used,
making the code hard to understand in places.

A version did exist that used individual frames on the heap instead of calling
match() recursively, but this ran substantially slower. The current version is
a refactoring that uses a vector of frames to remember backtracking points.
This runs no slower, and possibly even a bit faster than the original recursive
implementation. An initial vector of size START_FRAMES_SIZE (enough for maybe
50 frames) is allocated on the system stack. If this is not big enough, the
heap is used for a larger vector.

*******************************************************************************
******************************************************************************/




/*************************************************
*       Macros for the match() function          *
*************************************************/

/* These macros pack up tests that are used for partial matching several times
in the code. The second one is used when we already know we are past the end of
the subject. We set the "hit end" flag if the pointer is at the end of the
subject and either (a) the pointer is past the earliest inspected character
(i.e. something has been matched, even if not part of the actual matched
string), or (b) the pattern contains a lookbehind. These are the conditions for
which adding more characters may allow the current match to continue.

For hard partial matching, we immediately return a partial match. Otherwise,
carrying on means that a complete match on the current subject will be sought.
A partial match is returned only if no complete match can be found. */

#define CHECK_PARTIAL()\
  if (Feptr >= mb->end_subject) \
    { \
    SCHECK_PARTIAL(); \
    }

#define SCHECK_PARTIAL()\
  if (mb->partial != 0 && \
      (Feptr > mb->start_used_ptr || mb->allowemptypartial)) \
    { \
    mb->hitend = TRUE; \
    if (mb->partial > 1) return PCRE2_ERROR_PARTIAL; \
    }


/* These macros are used to implement backtracking. They simulate a recursive
call to the match() function by means of a local vector of frames which
remember the backtracking points. */

#define RMATCH(ra,rb)\
  {\
  start_ecode = ra;\
  Freturn_id = rb;\
  goto MATCH_RECURSE;\
  L_##rb:;\
  }

#define RRETURN(ra)\
  {\
  rrc = ra;\
  goto RETURN_SWITCH;\
  }



/*************************************************
*         Match from current position            *
*************************************************/

/* This function is called to run one match attempt at a single starting point
in the subject.

Performance note: It might be tempting to extract commonly used fields from the
mb structure (e.g. end_subject) into individual variables to improve
performance. Tests using gcc on a SPARC disproved this; in the first case, it
made performance worse.

Arguments:
   start_eptr   starting character in subject
   start_ecode  starting position in compiled code
   ovector      pointer to the final output vector
   oveccount    number of pairs in ovector
   top_bracket  number of capturing parentheses in the pattern
   frame_size   size of each backtracking frame
   mb           pointer to "static" variables block

Returns:        MATCH_MATCH if matched            )  these values are >= 0
                MATCH_NOMATCH if failed to match  )
                negative MATCH_xxx value for PRUNE, SKIP, etc
                negative PCRE2_ERROR_xxx value if aborted by an error condition
                (e.g. stopped by repeated call or depth limit)
*/

static int
match(PCRE2_SPTR start_eptr, PCRE2_SPTR start_ecode, PCRE2_SIZE *ovector,
  uint16_t oveccount, uint16_t top_bracket, PCRE2_SIZE frame_size,
  match_block *mb)
{
/* Frame-handling variables */

heapframe *F;           /* Current frame pointer */
heapframe *N = NULL;    /* Temporary frame pointers */
heapframe *P = NULL;
heapframe *assert_accept_frame;  /* For passing back the frame with captures */
PCRE2_SIZE frame_copy_size;      /* Amount to copy when creating a new frame */

/* Local variables that do not need to be preserved over calls to RRMATCH(). */

PCRE2_SPTR bracode;     /* Temp pointer to start of group */
PCRE2_SIZE offset;      /* Used for group offsets */
PCRE2_SIZE length;      /* Used for various length calculations */

int rrc;                /* Return from functions & backtracking "recursions" */
#ifdef SUPPORT_UNICODE
int proptype;           /* Type of character property */
#endif

uint32_t i;             /* Used for local loops */
uint32_t fc;            /* Character values */
uint32_t number;        /* Used for group and other numbers */
uint32_t reptype = 0;   /* Type of repetition (0 to avoid compiler warning) */
uint32_t group_frame_type;  /* Specifies type for new group frames */

BOOL condition;         /* Used in conditional groups */
BOOL cur_is_word;       /* Used in "word" tests */
BOOL prev_is_word;      /* Used in "word" tests */

/* UTF flag */

#ifdef SUPPORT_UNICODE
BOOL utf = (mb->poptions & PCRE2_UTF) != 0;
#else
BOOL utf = FALSE;
#endif

/* This is the length of the last part of a backtracking frame that must be
copied when a new frame is created. */

frame_copy_size = frame_size - offsetof(heapframe, eptr);

/* Set up the first current frame at the start of the vector, and initialize
fields that are not reset for new frames. */

F = mb->match_frames;
Frdepth = 0;                        /* "Recursion" depth */
Fcapture_last = 0;                  /* Number of most recent capture */
Fcurrent_recurse = RECURSE_UNSET;   /* Not pattern recursing. */
Fstart_match = Feptr = start_eptr;  /* Current data pointer and start match */
Fmark = NULL;                       /* Most recent mark */
Foffset_top = 0;                    /* End of captures within the frame */
Flast_group_offset = PCRE2_UNSET;   /* Saved frame of most recent group */
group_frame_type = 0;               /* Not a start of group frame */
goto NEW_FRAME;                     /* Start processing with this frame */

/* Come back here when we want to create a new frame for remembering a
backtracking point. */

MATCH_RECURSE:

/* Set up a new backtracking frame. If the vector is full, get a new one
on the heap, doubling the size, but constrained by the heap limit. */

N = (heapframe *)((char *)F + frame_size);
if (N >= mb->match_frames_top)
  {
  PCRE2_SIZE newsize = mb->frame_vector_size * 2;
  heapframe *new;

  if ((newsize / 1024) > mb->heap_limit)
    {
    PCRE2_SIZE maxsize = ((mb->heap_limit * 1024)/frame_size) * frame_size;
    if (mb->frame_vector_size >= maxsize) return PCRE2_ERROR_HEAPLIMIT;
    newsize = maxsize;
    }

  new = mb->memctl.malloc(newsize, mb->memctl.memory_data);
  if (new == NULL) return PCRE2_ERROR_NOMEMORY;
  memcpy(new, mb->match_frames, mb->frame_vector_size);

  F = (heapframe *)((char *)new + ((char *)F - (char *)mb->match_frames));
  N = (heapframe *)((char *)F + frame_size);

  if (mb->match_frames != mb->stack_frames)
    mb->memctl.free(mb->match_frames, mb->memctl.memory_data);
  mb->match_frames = new;
  mb->match_frames_top = (heapframe *)((char *)mb->match_frames + newsize);
  mb->frame_vector_size = newsize;
  }

#ifdef DEBUG_SHOW_RMATCH
fprintf(stderr, "++ RMATCH %2d frame=%d", Freturn_id, Frdepth + 1);
if (group_frame_type != 0)
  {
  fprintf(stderr, " type=%x ", group_frame_type);
  switch (GF_IDMASK(group_frame_type))
    {
    case GF_CAPTURE:
    fprintf(stderr, "capture=%d", GF_DATAMASK(group_frame_type));
    break;

    case GF_NOCAPTURE:
    fprintf(stderr, "nocapture op=%d", GF_DATAMASK(group_frame_type));
    break;

    case GF_CONDASSERT:
    fprintf(stderr, "condassert op=%d", GF_DATAMASK(group_frame_type));
    break;

    case GF_RECURSE:
    fprintf(stderr, "recurse=%d", GF_DATAMASK(group_frame_type));
    break;

    default:
    fprintf(stderr, "*** unknown ***");
    break;
    }
  }
fprintf(stderr, "\n");
#endif

/* Copy those fields that must be copied into the new frame, increase the
"recursion" depth (i.e. the new frame's index) and then make the new frame
current. */

memcpy((char *)N + offsetof(heapframe, eptr),
       (char *)F + offsetof(heapframe, eptr),
       frame_copy_size);

N->rdepth = Frdepth + 1;
F = N;

/* Carry on processing with a new frame. */

NEW_FRAME:
Fgroup_frame_type = group_frame_type;
Fecode = start_ecode;      /* Starting code pointer */
Fback_frame = frame_size;  /* Default is go back one frame */

/* If this is a special type of group frame, remember its offset for quick
access at the end of the group. If this is a recursion, set a new current
recursion value. */

if (group_frame_type != 0)
  {
  Flast_group_offset = (char *)F - (char *)mb->match_frames;
  if (GF_IDMASK(group_frame_type) == GF_RECURSE)
    Fcurrent_recurse = GF_DATAMASK(group_frame_type);
  group_frame_type = 0;
  }


/* ========================================================================= */
/* This is the main processing loop. First check that we haven't recorded too
many backtracks (search tree is too large), or that we haven't exceeded the
recursive depth limit (used too many backtracking frames). If not, process the
opcodes. */

if (mb->match_call_count++ >= mb->match_limit) return PCRE2_ERROR_MATCHLIMIT;
if (Frdepth >= mb->match_limit_depth) return PCRE2_ERROR_DEPTHLIMIT;

for (;;)
  {
#ifdef DEBUG_SHOW_OPS
fprintf(stderr, "++ op=%d\n", *Fecode);
#endif

  Fop = (uint8_t)(*Fecode);  /* Cast needed for 16-bit and 32-bit modes */
  switch(Fop)
    {
    /* ===================================================================== */
    /* Before OP_ACCEPT there may be any number of OP_CLOSE opcodes, to close
    any currently open capturing brackets. Unlike reaching the end of a group,
    where we know the starting frame is at the top of the chained frames, in
    this case we have to search back for the relevant frame in case other types
    of group that use chained frames have intervened. Multiple OP_CLOSEs always
    come innermost first, which matches the chain order. We can ignore this in
    a recursion, because captures are not passed out of recursions. */

    case OP_CLOSE:
    if (Fcurrent_recurse == RECURSE_UNSET)
      {
      number = GET2(Fecode, 1);
      offset = Flast_group_offset;
      for(;;)
        {
        if (offset == PCRE2_UNSET) return PCRE2_ERROR_INTERNAL;
        N = (heapframe *)((char *)mb->match_frames + offset);
        P = (heapframe *)((char *)N - frame_size);
        if (N->group_frame_type == (GF_CAPTURE | number)) break;
        offset = P->last_group_offset;
        }
      offset = (number << 1) - 2;
      Fcapture_last = number;
      Fovector[offset] = P->eptr - mb->start_subject;
      Fovector[offset+1] = Feptr - mb->start_subject;
      if (offset >= Foffset_top) Foffset_top = offset + 2;
      }
    Fecode += PRIV(OP_lengths)[*Fecode];
    break;


    /* ===================================================================== */
    /* Real or forced end of the pattern, assertion, or recursion. In an
    assertion ACCEPT, update the last used pointer and remember the current
    frame so that the captures and mark can be fished out of it. */

    case OP_ASSERT_ACCEPT:
    if (Feptr > mb->last_used_ptr) mb->last_used_ptr = Feptr;
    assert_accept_frame = F;
    RRETURN(MATCH_ACCEPT);

    /* If recursing, we have to find the most recent recursion. */

    case OP_ACCEPT:
    case OP_END:

    /* Handle end of a recursion. */

    if (Fcurrent_recurse != RECURSE_UNSET)
      {
      offset = Flast_group_offset;
      for(;;)
        {
        if (offset == PCRE2_UNSET) return PCRE2_ERROR_INTERNAL;
        N = (heapframe *)((char *)mb->match_frames + offset);
        P = (heapframe *)((char *)N - frame_size);
        if (GF_IDMASK(N->group_frame_type) == GF_RECURSE) break;
        offset = P->last_group_offset;
        }

      /* N is now the frame of the recursion; the previous frame is at the
      OP_RECURSE position. Go back there, copying the current subject position
      and mark, and move on past the OP_RECURSE. */

      P->eptr = Feptr;
      P->mark = Fmark;
      F = P;
      Fecode += 1 + LINK_SIZE;
      continue;
      }

    /* Not a recursion. Fail for an empty string match if either PCRE2_NOTEMPTY
    is set, or if PCRE2_NOTEMPTY_ATSTART is set and we have matched at the
    start of the subject. In both cases, backtracking will then try other
    alternatives, if any. */

    if (Feptr == Fstart_match &&
         ((mb->moptions & PCRE2_NOTEMPTY) != 0 ||
           ((mb->moptions & PCRE2_NOTEMPTY_ATSTART) != 0 &&
             Fstart_match == mb->start_subject + mb->start_offset)))
      RRETURN(MATCH_NOMATCH);

    /* Also fail if PCRE2_ENDANCHORED is set and the end of the match is not
    the end of the subject. After (*ACCEPT) we fail the entire match (at this
    position) but backtrack on reaching the end of the pattern. */

    if (Feptr < mb->end_subject &&
        ((mb->moptions | mb->poptions) & PCRE2_ENDANCHORED) != 0)
      {
      if (Fop == OP_END) RRETURN(MATCH_NOMATCH);
      return MATCH_NOMATCH;
      }

    /* We have a successful match of the whole pattern. Record the result and
    then do a direct return from the function. If there is space in the offset
    vector, set any pairs that follow the highest-numbered captured string but
    are less than the number of capturing groups in the pattern to PCRE2_UNSET.
    It is documented that this happens. "Gaps" are set to PCRE2_UNSET
    dynamically. It is only those at the end that need setting here. */

    mb->end_match_ptr = Feptr;           /* Record where we ended */
    mb->end_offset_top = Foffset_top;    /* and how many extracts were taken */
    mb->mark = Fmark;                    /* and the last success mark */
    if (Feptr > mb->last_used_ptr) mb->last_used_ptr = Feptr;

    ovector[0] = Fstart_match - mb->start_subject;
    ovector[1] = Feptr - mb->start_subject;

    /* Set i to the smaller of the sizes of the external and frame ovectors. */

    i = 2 * ((top_bracket + 1 > oveccount)? oveccount : top_bracket + 1);
    memcpy(ovector + 2, Fovector, (i - 2) * sizeof(PCRE2_SIZE));
    while (--i >= Foffset_top + 2) ovector[i] = PCRE2_UNSET;
    return MATCH_MATCH;  /* Note: NOT RRETURN */


    /*===================================================================== */
    /* Match any single character type except newline; have to take care with
    CRLF newlines and partial matching. */

    case OP_ANY:
    if (IS_NEWLINE(Feptr)) RRETURN(MATCH_NOMATCH);
    if (mb->partial != 0 &&
        Feptr == mb->end_subject - 1 &&
        NLBLOCK->nltype == NLTYPE_FIXED &&
        NLBLOCK->nllen == 2 &&
        UCHAR21TEST(Feptr) == NLBLOCK->nl[0])
      {
      mb->hitend = TRUE;
      if (mb->partial > 1) return PCRE2_ERROR_PARTIAL;
      }
    /* Fall through */

    /* Match any single character whatsoever. */

    case OP_ALLANY:
    if (Feptr >= mb->end_subject)  /* DO NOT merge the Feptr++ here; it must */
      {                            /* not be updated before SCHECK_PARTIAL. */
      SCHECK_PARTIAL();
      RRETURN(MATCH_NOMATCH);
      }
    Feptr++;
#ifdef SUPPORT_UNICODE
    if (utf) ACROSSCHAR(Feptr < mb->end_subject, Feptr, Feptr++);
#endif
    Fecode++;
    break;


    /* ===================================================================== */
    /* Match a single code unit, even in UTF mode. This opcode really does
    match any code unit, even newline. (It really should be called ANYCODEUNIT,
    of course - the byte name is from pre-16 bit days.) */

    case OP_ANYBYTE:
    if (Feptr >= mb->end_subject)   /* DO NOT merge the Feptr++ here; it must */
      {                             /* not be updated before SCHECK_PARTIAL. */
      SCHECK_PARTIAL();
      RRETURN(MATCH_NOMATCH);
      }
    Feptr++;
    Fecode++;
    break;


    /* ===================================================================== */
    /* Match a single character, casefully */

    case OP_CHAR:
#ifdef SUPPORT_UNICODE
    if (utf)
      {
      Flength = 1;
      Fecode++;
      GETCHARLEN(fc, Fecode, Flength);
      if (Flength > (PCRE2_SIZE)(mb->end_subject - Feptr))
        {
        CHECK_PARTIAL();             /* Not SCHECK_PARTIAL() */
        RRETURN(MATCH_NOMATCH);
        }
      for (; Flength > 0; Flength--)
        {
        if (*Fecode++ != UCHAR21INC(Feptr)) RRETURN(MATCH_NOMATCH);
        }
      }
    else
#endif
    /* Not UTF mode */
      {
      if (mb->end_subject - Feptr < 1)
        {
        SCHECK_PARTIAL();            /* This one can use SCHECK_PARTIAL() */
        RRETURN(MATCH_NOMATCH);
        }
      if (Fecode[1] != *Feptr++) RRETURN(MATCH_NOMATCH);
      Fecode += 2;
      }
    break;


    /* ===================================================================== */
    /* Match a single character, caselessly. If we are at the end of the
    subject, give up immediately. We get here only when the pattern character
    has at most one other case. Characters with more than two cases are coded
    as OP_PROP with the pseudo-property PT_CLIST. */

    case OP_CHARI:
    if (Feptr >= mb->end_subject)
      {
      SCHECK_PARTIAL();
      RRETURN(MATCH_NOMATCH);
      }

#ifdef SUPPORT_UNICODE
    if (utf)
      {
      Flength = 1;
      Fecode++;
      GETCHARLEN(fc, Fecode, Flength);

      /* If the pattern character's value is < 128, we know that its other case
      (if any) is also < 128 (and therefore only one code unit long in all
      code-unit widths), so we can use the fast lookup table. We checked above
      that there is at least one character left in the subject. */

      if (fc < 128)
        {
        uint32_t cc = UCHAR21(Feptr);
        if (mb->lcc[fc] != TABLE_GET(cc, mb->lcc, cc)) RRETURN(MATCH_NOMATCH);
        Fecode++;
        Feptr++;
        }

      /* Otherwise we must pick up the subject character and use Unicode
      property support to test its other case. Note that we cannot use the
      value of "Flength" to check for sufficient bytes left, because the other
      case of the character may have more or fewer code units. */

      else
        {
        uint32_t dc;
        GETCHARINC(dc, Feptr);
        Fecode += Flength;
        if (dc != fc && dc != UCD_OTHERCASE(fc)) RRETURN(MATCH_NOMATCH);
        }
      }
    else
#endif   /* SUPPORT_UNICODE */

    /* Not UTF mode; use the table for characters < 256. */
      {
      if (TABLE_GET(Fecode[1], mb->lcc, Fecode[1])
          != TABLE_GET(*Feptr, mb->lcc, *Feptr)) RRETURN(MATCH_NOMATCH);
      Feptr++;
      Fecode += 2;
      }
    break;


    /* ===================================================================== */
    /* Match not a single character. */

    case OP_NOT:
    case OP_NOTI:
    if (Feptr >= mb->end_subject)
      {
      SCHECK_PARTIAL();
      RRETURN(MATCH_NOMATCH);
      }
#ifdef SUPPORT_UNICODE
    if (utf)
      {
      uint32_t ch;
      Fecode++;
      GETCHARINC(ch, Fecode);
      GETCHARINC(fc, Feptr);
      if (ch == fc)
        {
        RRETURN(MATCH_NOMATCH);  /* Caseful match */
        }
      else if (Fop == OP_NOTI)   /* If caseless */
        {
        if (ch > 127)
          ch = UCD_OTHERCASE(ch);
        else
          ch = TABLE_GET(ch, mb->fcc, ch);
        if (ch == fc) RRETURN(MATCH_NOMATCH);
        }
      }
    else
#endif  /* SUPPORT_UNICODE */
      {
      uint32_t ch = Fecode[1];
      fc = *Feptr++;
      if (ch == fc || (Fop == OP_NOTI && TABLE_GET(ch, mb->fcc, ch) == fc))
        RRETURN(MATCH_NOMATCH);
      Fecode += 2;
      }
    break;


    /* ===================================================================== */
    /* Match a single character repeatedly. */

#define Loclength    F->temp_size
#define Lstart_eptr  F->temp_sptr[0]
#define Lcharptr     F->temp_sptr[1]
#define Lmin         F->temp_32[0]
#define Lmax         F->temp_32[1]
#define Lc           F->temp_32[2]
#define Loc          F->temp_32[3]

    case OP_EXACT:
    case OP_EXACTI:
    Lmin = Lmax = GET2(Fecode, 1);
    Fecode += 1 + IMM2_SIZE;
    goto REPEATCHAR;

    case OP_POSUPTO:
    case OP_POSUPTOI:
    reptype = REPTYPE_POS;
    Lmin = 0;
    Lmax = GET2(Fecode, 1);
    Fecode += 1 + IMM2_SIZE;
    goto REPEATCHAR;

    case OP_UPTO:
    case OP_UPTOI:
    reptype = REPTYPE_MAX;
    Lmin = 0;
    Lmax = GET2(Fecode, 1);
    Fecode += 1 + IMM2_SIZE;
    goto REPEATCHAR;

    case OP_MINUPTO:
    case OP_MINUPTOI:
    reptype = REPTYPE_MIN;
    Lmin = 0;
    Lmax = GET2(Fecode, 1);
    Fecode += 1 + IMM2_SIZE;
    goto REPEATCHAR;

    case OP_POSSTAR:
    case OP_POSSTARI:
    reptype = REPTYPE_POS;
    Lmin = 0;
    Lmax = UINT32_MAX;
    Fecode++;
    goto REPEATCHAR;

    case OP_POSPLUS:
    case OP_POSPLUSI:
    reptype = REPTYPE_POS;
    Lmin = 1;
    Lmax = UINT32_MAX;
    Fecode++;
    goto REPEATCHAR;

    case OP_POSQUERY:
    case OP_POSQUERYI:
    reptype = REPTYPE_POS;
    Lmin = 0;
    Lmax = 1;
    Fecode++;
    goto REPEATCHAR;

    case OP_STAR:
    case OP_STARI:
    case OP_MINSTAR:
    case OP_MINSTARI:
    case OP_PLUS:
    case OP_PLUSI:
    case OP_MINPLUS:
    case OP_MINPLUSI:
    case OP_QUERY:
    case OP_QUERYI:
    case OP_MINQUERY:
    case OP_MINQUERYI:
    fc = *Fecode++ - ((Fop < OP_STARI)? OP_STAR : OP_STARI);
    Lmin = rep_min[fc];
    Lmax = rep_max[fc];
    reptype = rep_typ[fc];

    /* Common code for all repeated single-character matches. We first check
    for the minimum number of characters. If the minimum equals the maximum, we
    are done. Otherwise, if minimizing, check the rest of the pattern for a
    match; if there isn't one, advance up to the maximum, one character at a
    time.

    If maximizing, advance up to the maximum number of matching characters,
    until Feptr is past the end of the maximum run. If possessive, we are
    then done (no backing up). Otherwise, match at this position; anything
    other than no match is immediately returned. For nomatch, back up one
    character, unless we are matching \R and the last thing matched was
    \r\n, in which case, back up two code units until we reach the first
    optional character position.

    The various UTF/non-UTF and caseful/caseless cases are handled separately,
    for speed. */

    REPEATCHAR:
#ifdef SUPPORT_UNICODE
    if (utf)
      {
      Flength = 1;
      Lcharptr = Fecode;
      GETCHARLEN(fc, Fecode, Flength);
      Fecode += Flength;

      /* Handle multi-code-unit character matching, caseful and caseless. */

      if (Flength > 1)
        {
        uint32_t othercase;

        if (Fop >= OP_STARI &&     /* Caseless */
            (othercase = UCD_OTHERCASE(fc)) != fc)
          Loclength = PRIV(ord2utf)(othercase, Foccu);
        else Loclength = 0;

        for (i = 1; i <= Lmin; i++)
          {
          if (Feptr <= mb->end_subject - Flength &&
            memcmp(Feptr, Lcharptr, CU2BYTES(Flength)) == 0) Feptr += Flength;
          else if (Loclength > 0 &&
                   Feptr <= mb->end_subject - Loclength &&
                   memcmp(Feptr, Foccu, CU2BYTES(Loclength)) == 0)
            Feptr += Loclength;
          else
            {
            CHECK_PARTIAL();
            RRETURN(MATCH_NOMATCH);
            }
          }

        if (Lmin == Lmax) continue;

        if (reptype == REPTYPE_MIN)
          {
          for (;;)
            {
            RMATCH(Fecode, RM202);
            if (rrc != MATCH_NOMATCH) RRETURN(rrc);
            if (Lmin++ >= Lmax) RRETURN(MATCH_NOMATCH);
            if (Feptr <= mb->end_subject - Flength &&
              memcmp(Feptr, Lcharptr, CU2BYTES(Flength)) == 0) Feptr += Flength;
            else if (Loclength > 0 &&
                     Feptr <= mb->end_subject - Loclength &&
                     memcmp(Feptr, Foccu, CU2BYTES(Loclength)) == 0)
              Feptr += Loclength;
            else
              {
              CHECK_PARTIAL();
              RRETURN(MATCH_NOMATCH);
              }
            }
          /* Control never gets here */
          }

        else  /* Maximize */
          {
          Lstart_eptr = Feptr;
          for (i = Lmin; i < Lmax; i++)
            {
            if (Feptr <= mb->end_subject - Flength &&
                memcmp(Feptr, Lcharptr, CU2BYTES(Flength)) == 0)
              Feptr += Flength;
            else if (Loclength > 0 &&
                     Feptr <= mb->end_subject - Loclength &&
                     memcmp(Feptr, Foccu, CU2BYTES(Loclength)) == 0)
              Feptr += Loclength;
            else
              {
              CHECK_PARTIAL();
              break;
              }
            }

          /* After \C in UTF mode, Lstart_eptr might be in the middle of a
          Unicode character. Use <= Lstart_eptr to ensure backtracking doesn't
          go too far. */

          if (reptype != REPTYPE_POS) for(;;)
            {
            if (Feptr <= Lstart_eptr) break;
            RMATCH(Fecode, RM203);
            if (rrc != MATCH_NOMATCH) RRETURN(rrc);
            Feptr--;
            BACKCHAR(Feptr);
            }
          }
        break;   /* End of repeated wide character handling */
        }

      /* Length of UTF character is 1. Put it into the preserved variable and
      fall through to the non-UTF code. */

      Lc = fc;
      }
    else
#endif  /* SUPPORT_UNICODE */

    /* When not in UTF mode, load a single-code-unit character. Then proceed as
    above. */

    Lc = *Fecode++;

    /* Caseless comparison */

    if (Fop >= OP_STARI)
      {
#if PCRE2_CODE_UNIT_WIDTH == 8
      /* Lc must be < 128 in UTF-8 mode. */
      Loc = mb->fcc[Lc];
#else /* 16-bit & 32-bit */
#ifdef SUPPORT_UNICODE
      if (utf && Lc > 127) Loc = UCD_OTHERCASE(Lc);
      else
#endif  /* SUPPORT_UNICODE */
      Loc = TABLE_GET(Lc, mb->fcc, Lc);
#endif  /* PCRE2_CODE_UNIT_WIDTH == 8 */

      for (i = 1; i <= Lmin; i++)
        {
        uint32_t cc;                 /* Faster than PCRE2_UCHAR */
        if (Feptr >= mb->end_subject)
          {
          SCHECK_PARTIAL();
          RRETURN(MATCH_NOMATCH);
          }
        cc = UCHAR21TEST(Feptr);
        if (Lc != cc && Loc != cc) RRETURN(MATCH_NOMATCH);
        Feptr++;
        }
      if (Lmin == Lmax) continue;

      if (reptype == REPTYPE_MIN)
        {
        for (;;)
          {
          uint32_t cc;               /* Faster than PCRE2_UCHAR */
          RMATCH(Fecode, RM25);
          if (rrc != MATCH_NOMATCH) RRETURN(rrc);
          if (Lmin++ >= Lmax) RRETURN(MATCH_NOMATCH);
          if (Feptr >= mb->end_subject)
            {
            SCHECK_PARTIAL();
            RRETURN(MATCH_NOMATCH);
            }
          cc = UCHAR21TEST(Feptr);
          if (Lc != cc && Loc != cc) RRETURN(MATCH_NOMATCH);
          Feptr++;
          }
        /* Control never gets here */
        }

      else  /* Maximize */
        {
        Lstart_eptr = Feptr;
        for (i = Lmin; i < Lmax; i++)
          {
          uint32_t cc;               /* Faster than PCRE2_UCHAR */
          if (Feptr >= mb->end_subject)
            {
            SCHECK_PARTIAL();
            break;
            }
          cc = UCHAR21TEST(Feptr);
          if (Lc != cc && Loc != cc) break;
          Feptr++;
          }
        if (reptype != REPTYPE_POS) for (;;)
          {
          if (Feptr == Lstart_eptr) break;
          RMATCH(Fecode, RM26);
          Feptr--;
          if (rrc != MATCH_NOMATCH) RRETURN(rrc);
          }
        }
      }

    /* Caseful comparisons (includes all multi-byte characters) */

    else
      {
      for (i = 1; i <= Lmin; i++)
        {
        if (Feptr >= mb->end_subject)
          {
          SCHECK_PARTIAL();
          RRETURN(MATCH_NOMATCH);
          }
        if (Lc != UCHAR21INCTEST(Feptr)) RRETURN(MATCH_NOMATCH);
        }

      if (Lmin == Lmax) continue;

      if (reptype == REPTYPE_MIN)
        {
        for (;;)
          {
          RMATCH(Fecode, RM27);
          if (rrc != MATCH_NOMATCH) RRETURN(rrc);
          if (Lmin++ >= Lmax) RRETURN(MATCH_NOMATCH);
          if (Feptr >= mb->end_subject)
            {
            SCHECK_PARTIAL();
            RRETURN(MATCH_NOMATCH);
            }
          if (Lc != UCHAR21INCTEST(Feptr)) RRETURN(MATCH_NOMATCH);
          }
        /* Control never gets here */
        }
      else  /* Maximize */
        {
        Lstart_eptr = Feptr;
        for (i = Lmin; i < Lmax; i++)
          {
          if (Feptr >= mb->end_subject)
            {
            SCHECK_PARTIAL();
            break;
            }

          if (Lc != UCHAR21TEST(Feptr)) break;
          Feptr++;
          }

        if (reptype != REPTYPE_POS) for (;;)
          {
          if (Feptr <= Lstart_eptr) break;
          RMATCH(Fecode, RM28);
          Feptr--;
          if (rrc != MATCH_NOMATCH) RRETURN(rrc);
          }
        }
      }
    break;

#undef Loclength
#undef Lstart_eptr
#undef Lcharptr
#undef Lmin
#undef Lmax
#undef Lc
#undef Loc


    /* ===================================================================== */
    /* Match a negated single one-byte character repeatedly. This is almost a
    repeat of the code for a repeated single character, but I haven't found a
    nice way of commoning these up that doesn't require a test of the
    positive/negative option for each character match. Maybe that wouldn't add
    very much to the time taken, but character matching *is* what this is all
    about... */

#define Lstart_eptr  F->temp_sptr[0]
#define Lmin         F->temp_32[0]
#define Lmax         F->temp_32[1]
#define Lc           F->temp_32[2]
#define Loc          F->temp_32[3]

    case OP_NOTEXACT:
    case OP_NOTEXACTI:
    Lmin = Lmax = GET2(Fecode, 1);
    Fecode += 1 + IMM2_SIZE;
    goto REPEATNOTCHAR;

    case OP_NOTUPTO:
    case OP_NOTUPTOI:
    Lmin = 0;
    Lmax = GET2(Fecode, 1);
    reptype = REPTYPE_MAX;
    Fecode += 1 + IMM2_SIZE;
    goto REPEATNOTCHAR;

    case OP_NOTMINUPTO:
    case OP_NOTMINUPTOI:
    Lmin = 0;
    Lmax = GET2(Fecode, 1);
    reptype = REPTYPE_MIN;
    Fecode += 1 + IMM2_SIZE;
    goto REPEATNOTCHAR;

    case OP_NOTPOSSTAR:
    case OP_NOTPOSSTARI:
    reptype = REPTYPE_POS;
    Lmin = 0;
    Lmax = UINT32_MAX;
    Fecode++;
    goto REPEATNOTCHAR;

    case OP_NOTPOSPLUS:
    case OP_NOTPOSPLUSI:
    reptype = REPTYPE_POS;
    Lmin = 1;
    Lmax = UINT32_MAX;
    Fecode++;
    goto REPEATNOTCHAR;

    case OP_NOTPOSQUERY:
    case OP_NOTPOSQUERYI:
    reptype = REPTYPE_POS;
    Lmin = 0;
    Lmax = 1;
    Fecode++;
    goto REPEATNOTCHAR;

    case OP_NOTPOSUPTO:
    case OP_NOTPOSUPTOI:
    reptype = REPTYPE_POS;
    Lmin = 0;
    Lmax = GET2(Fecode, 1);
    Fecode += 1 + IMM2_SIZE;
    goto REPEATNOTCHAR;

    case OP_NOTSTAR:
    case OP_NOTSTARI:
    case OP_NOTMINSTAR:
    case OP_NOTMINSTARI:
    case OP_NOTPLUS:
    case OP_NOTPLUSI:
    case OP_NOTMINPLUS:
    case OP_NOTMINPLUSI:
    case OP_NOTQUERY:
    case OP_NOTQUERYI:
    case OP_NOTMINQUERY:
    case OP_NOTMINQUERYI:
    fc = *Fecode++ - ((Fop >= OP_NOTSTARI)? OP_NOTSTARI: OP_NOTSTAR);
    Lmin = rep_min[fc];
    Lmax = rep_max[fc];
    reptype = rep_typ[fc];

    /* Common code for all repeated single-character non-matches. */

    REPEATNOTCHAR:
    GETCHARINCTEST(Lc, Fecode);

    /* The code is duplicated for the caseless and caseful cases, for speed,
    since matching characters is likely to be quite common. First, ensure the
    minimum number of matches are present. If Lmin = Lmax, we are done.
    Otherwise, if minimizing, keep trying the rest of the expression and
    advancing one matching character if failing, up to the maximum.
    Alternatively, if maximizing, find the maximum number of characters and
    work backwards. */

    if (Fop >= OP_NOTSTARI)     /* Caseless */
      {
#ifdef SUPPORT_UNICODE
      if (utf && Lc > 127)
        Loc = UCD_OTHERCASE(Lc);
      else
#endif /* SUPPORT_UNICODE */

      Loc = TABLE_GET(Lc, mb->fcc, Lc);  /* Other case from table */

#ifdef SUPPORT_UNICODE
      if (utf)
        {
        uint32_t d;
        for (i = 1; i <= Lmin; i++)
          {
          if (Feptr >= mb->end_subject)
            {
            SCHECK_PARTIAL();
            RRETURN(MATCH_NOMATCH);
            }
          GETCHARINC(d, Feptr);
          if (Lc == d || Loc == d) RRETURN(MATCH_NOMATCH);
          }
        }
      else
#endif  /* SUPPORT_UNICODE */

      /* Not UTF mode */
        {
        for (i = 1; i <= Lmin; i++)
          {
          if (Feptr >= mb->end_subject)
            {
            SCHECK_PARTIAL();
            RRETURN(MATCH_NOMATCH);
            }
          if (Lc == *Feptr || Loc == *Feptr) RRETURN(MATCH_NOMATCH);
          Feptr++;
          }
        }

      if (Lmin == Lmax) continue;  /* Finished for exact count */

      if (reptype == REPTYPE_MIN)
        {
#ifdef SUPPORT_UNICODE
        if (utf)
          {
          uint32_t d;
          for (;;)
            {
            RMATCH(Fecode, RM204);
            if (rrc != MATCH_NOMATCH) RRETURN(rrc);
            if (Lmin++ >= Lmax) RRETURN(MATCH_NOMATCH);
            if (Feptr >= mb->end_subject)
              {
              SCHECK_PARTIAL();
              RRETURN(MATCH_NOMATCH);
              }
            GETCHARINC(d, Feptr);
            if (Lc == d || Loc == d) RRETURN(MATCH_NOMATCH);
            }
          }
        else
#endif  /*SUPPORT_UNICODE */

        /* Not UTF mode */
          {
          for (;;)
            {
            RMATCH(Fecode, RM29);
            if (rrc != MATCH_NOMATCH) RRETURN(rrc);
            if (Lmin++ >= Lmax) RRETURN(MATCH_NOMATCH);
            if (Feptr >= mb->end_subject)
              {
              SCHECK_PARTIAL();
              RRETURN(MATCH_NOMATCH);
              }
            if (Lc == *Feptr || Loc == *Feptr) RRETURN(MATCH_NOMATCH);
            Feptr++;
            }
          }
        /* Control never gets here */
        }

      /* Maximize case */

      else
        {
        Lstart_eptr = Feptr;

#ifdef SUPPORT_UNICODE
        if (utf)
          {
          uint32_t d;
          for (i = Lmin; i < Lmax; i++)
            {
            int len = 1;
            if (Feptr >= mb->end_subject)
              {
              SCHECK_PARTIAL();
              break;
              }
            GETCHARLEN(d, Feptr, len);
            if (Lc == d || Loc == d) break;
            Feptr += len;
            }

          /* After \C in UTF mode, Lstart_eptr might be in the middle of a
          Unicode character. Use <= Lstart_eptr to ensure backtracking doesn't
          go too far. */

          if (reptype != REPTYPE_POS) for(;;)
            {
            if (Feptr <= Lstart_eptr) break;
            RMATCH(Fecode, RM205);
            if (rrc != MATCH_NOMATCH) RRETURN(rrc);
            Feptr--;
            BACKCHAR(Feptr);
            }
          }
        else
#endif  /* SUPPORT_UNICODE */

        /* Not UTF mode */
          {
          for (i = Lmin; i < Lmax; i++)
            {
            if (Feptr >= mb->end_subject)
              {
              SCHECK_PARTIAL();
              break;
              }
            if (Lc == *Feptr || Loc == *Feptr) break;
            Feptr++;
            }
          if (reptype != REPTYPE_POS) for (;;)
            {
            if (Feptr == Lstart_eptr) break;
            RMATCH(Fecode, RM30);
            if (rrc != MATCH_NOMATCH) RRETURN(rrc);
            Feptr--;
            }
          }
        }
      }

    /* Caseful comparisons */

    else
      {
#ifdef SUPPORT_UNICODE
      if (utf)
        {
        uint32_t d;
        for (i = 1; i <= Lmin; i++)
          {
          if (Feptr >= mb->end_subject)
            {
            SCHECK_PARTIAL();
            RRETURN(MATCH_NOMATCH);
            }
          GETCHARINC(d, Feptr);
          if (Lc == d) RRETURN(MATCH_NOMATCH);
          }
        }
      else
#endif
      /* Not UTF mode */
        {
        for (i = 1; i <= Lmin; i++)
          {
          if (Feptr >= mb->end_subject)
            {
            SCHECK_PARTIAL();
            RRETURN(MATCH_NOMATCH);
            }
          if (Lc == *Feptr++) RRETURN(MATCH_NOMATCH);
          }
        }

      if (Lmin == Lmax) continue;

      if (reptype == REPTYPE_MIN)
        {
#ifdef SUPPORT_UNICODE
        if (utf)
          {
          uint32_t d;
          for (;;)
            {
            RMATCH(Fecode, RM206);
            if (rrc != MATCH_NOMATCH) RRETURN(rrc);
            if (Lmin++ >= Lmax) RRETURN(MATCH_NOMATCH);
            if (Feptr >= mb->end_subject)
              {
              SCHECK_PARTIAL();
              RRETURN(MATCH_NOMATCH);
              }
            GETCHARINC(d, Feptr);
            if (Lc == d) RRETURN(MATCH_NOMATCH);
            }
          }
        else
#endif
        /* Not UTF mode */
          {
          for (;;)
            {
            RMATCH(Fecode, RM31);
            if (rrc != MATCH_NOMATCH) RRETURN(rrc);
            if (Lmin++ >= Lmax) RRETURN(MATCH_NOMATCH);
            if (Feptr >= mb->end_subject)
              {
              SCHECK_PARTIAL();
              RRETURN(MATCH_NOMATCH);
              }
            if (Lc == *Feptr++) RRETURN(MATCH_NOMATCH);
            }
          }
        /* Control never gets here */
        }

      /* Maximize case */

      else
        {
        Lstart_eptr = Feptr;

#ifdef SUPPORT_UNICODE
        if (utf)
          {
          uint32_t d;
          for (i = Lmin; i < Lmax; i++)
            {
            int len = 1;
            if (Feptr >= mb->end_subject)
              {
              SCHECK_PARTIAL();
              break;
              }
            GETCHARLEN(d, Feptr, len);
            if (Lc == d) break;
            Feptr += len;
            }

          /* After \C in UTF mode, Lstart_eptr might be in the middle of a
          Unicode character. Use <= Lstart_eptr to ensure backtracking doesn't
          go too far. */

          if (reptype != REPTYPE_POS) for(;;)
            {
            if (Feptr <= Lstart_eptr) break;
            RMATCH(Fecode, RM207);
            if (rrc != MATCH_NOMATCH) RRETURN(rrc);
            Feptr--;
            BACKCHAR(Feptr);
            }
          }
        else
#endif
        /* Not UTF mode */
          {
          for (i = Lmin; i < Lmax; i++)
            {
            if (Feptr >= mb->end_subject)
              {
              SCHECK_PARTIAL();
              break;
              }
            if (Lc == *Feptr) break;
            Feptr++;
            }
          if (reptype != REPTYPE_POS) for (;;)
            {
            if (Feptr == Lstart_eptr) break;
            RMATCH(Fecode, RM32);
            if (rrc != MATCH_NOMATCH) RRETURN(rrc);
            Feptr--;
            }
          }
        }
      }
    break;

#undef Lstart_eptr
#undef Lmin
#undef Lmax
#undef Lc
#undef Loc


    /* ===================================================================== */
    /* Match a bit-mapped character class, possibly repeatedly. These opcodes
    are used when all the characters in the class have values in the range
    0-255, and either the matching is caseful, or the characters are in the
    range 0-127 when UTF processing is enabled. The only difference between
    OP_CLASS and OP_NCLASS occurs when a data character outside the range is
    encountered. */

#define Lmin               F->temp_32[0]
#define Lmax               F->temp_32[1]
#define Lstart_eptr        F->temp_sptr[0]
#define Lbyte_map_address  F->temp_sptr[1]
#define Lbyte_map          ((unsigned char *)Lbyte_map_address)

    case OP_NCLASS:
    case OP_CLASS:
      {
      Lbyte_map_address = Fecode + 1;           /* Save for matching */
      Fecode += 1 + (32 / sizeof(PCRE2_UCHAR)); /* Advance past the item */

      /* Look past the end of the item to see if there is repeat information
      following. Then obey similar code to character type repeats. */

      switch (*Fecode)
        {
        case OP_CRSTAR:
        case OP_CRMINSTAR:
        case OP_CRPLUS:
        case OP_CRMINPLUS:
        case OP_CRQUERY:
        case OP_CRMINQUERY:
        case OP_CRPOSSTAR:
        case OP_CRPOSPLUS:
        case OP_CRPOSQUERY:
        fc = *Fecode++ - OP_CRSTAR;
        Lmin = rep_min[fc];
        Lmax = rep_max[fc];
        reptype = rep_typ[fc];
        break;

        case OP_CRRANGE:
        case OP_CRMINRANGE:
        case OP_CRPOSRANGE:
        Lmin = GET2(Fecode, 1);
        Lmax = GET2(Fecode, 1 + IMM2_SIZE);
        if (Lmax == 0) Lmax = UINT32_MAX;       /* Max 0 => infinity */
        reptype = rep_typ[*Fecode - OP_CRSTAR];
        Fecode += 1 + 2 * IMM2_SIZE;
        break;

        default:               /* No repeat follows */
        Lmin = Lmax = 1;
        break;
        }

      /* First, ensure the minimum number of matches are present. */

#ifdef SUPPORT_UNICODE
      if (utf)
        {
        for (i = 1; i <= Lmin; i++)
          {
          if (Feptr >= mb->end_subject)
            {
            SCHECK_PARTIAL();
            RRETURN(MATCH_NOMATCH);
            }
          GETCHARINC(fc, Feptr);
          if (fc > 255)
            {
            if (Fop == OP_CLASS) RRETURN(MATCH_NOMATCH);
            }
          else
            if ((Lbyte_map[fc/8] & (1u << (fc&7))) == 0) RRETURN(MATCH_NOMATCH);
          }
        }
      else
#endif
      /* Not UTF mode */
        {
        for (i = 1; i <= Lmin; i++)
          {
          if (Feptr >= mb->end_subject)
            {
            SCHECK_PARTIAL();
            RRETURN(MATCH_NOMATCH);
            }
          fc = *Feptr++;
#if PCRE2_CODE_UNIT_WIDTH != 8
          if (fc > 255)
            {
            if (Fop == OP_CLASS) RRETURN(MATCH_NOMATCH);
            }
          else
#endif
          if ((Lbyte_map[fc/8] & (1u << (fc&7))) == 0) RRETURN(MATCH_NOMATCH);
          }
        }

      /* If Lmax == Lmin we are done. Continue with main loop. */

      if (Lmin == Lmax) continue;

      /* If minimizing, keep testing the rest of the expression and advancing
      the pointer while it matches the class. */

      if (reptype == REPTYPE_MIN)
        {
#ifdef SUPPORT_UNICODE
        if (utf)
          {
          for (;;)
            {
            RMATCH(Fecode, RM200);
            if (rrc != MATCH_NOMATCH) RRETURN(rrc);
            if (Lmin++ >= Lmax) RRETURN(MATCH_NOMATCH);
            if (Feptr >= mb->end_subject)
              {
              SCHECK_PARTIAL();
              RRETURN(MATCH_NOMATCH);
              }
            GETCHARINC(fc, Feptr);
            if (fc > 255)
              {
              if (Fop == OP_CLASS) RRETURN(MATCH_NOMATCH);
              }
            else
              if ((Lbyte_map[fc/8] & (1u << (fc&7))) == 0) RRETURN(MATCH_NOMATCH);
            }
          }
        else
#endif
        /* Not UTF mode */
          {
          for (;;)
            {
            RMATCH(Fecode, RM23);
            if (rrc != MATCH_NOMATCH) RRETURN(rrc);
            if (Lmin++ >= Lmax) RRETURN(MATCH_NOMATCH);
            if (Feptr >= mb->end_subject)
              {
              SCHECK_PARTIAL();
              RRETURN(MATCH_NOMATCH);
              }
            fc = *Feptr++;
#if PCRE2_CODE_UNIT_WIDTH != 8
            if (fc > 255)
              {
              if (Fop == OP_CLASS) RRETURN(MATCH_NOMATCH);
              }
            else
#endif
            if ((Lbyte_map[fc/8] & (1u << (fc&7))) == 0) RRETURN(MATCH_NOMATCH);
            }
          }
        /* Control never gets here */
        }

      /* If maximizing, find the longest possible run, then work backwards. */

      else
        {
        Lstart_eptr = Feptr;

#ifdef SUPPORT_UNICODE
        if (utf)
          {
          for (i = Lmin; i < Lmax; i++)
            {
            int len = 1;
            if (Feptr >= mb->end_subject)
              {
              SCHECK_PARTIAL();
              break;
              }
            GETCHARLEN(fc, Feptr, len);
            if (fc > 255)
              {
              if (Fop == OP_CLASS) break;
              }
            else
              if ((Lbyte_map[fc/8] & (1u << (fc&7))) == 0) break;
            Feptr += len;
            }

          if (reptype == REPTYPE_POS) continue;    /* No backtracking */

          /* After \C in UTF mode, Lstart_eptr might be in the middle of a
          Unicode character. Use <= Lstart_eptr to ensure backtracking doesn't
          go too far. */

          for (;;)
            {
            RMATCH(Fecode, RM201);
            if (rrc != MATCH_NOMATCH) RRETURN(rrc);
            if (Feptr-- <= Lstart_eptr) break;  /* Tried at original position */
            BACKCHAR(Feptr);
            }
          }
        else
#endif
          /* Not UTF mode */
          {
          for (i = Lmin; i < Lmax; i++)
            {
            if (Feptr >= mb->end_subject)
              {
              SCHECK_PARTIAL();
              break;
              }
            fc = *Feptr;
#if PCRE2_CODE_UNIT_WIDTH != 8
            if (fc > 255)
              {
              if (Fop == OP_CLASS) break;
              }
            else
#endif
            if ((Lbyte_map[fc/8] & (1u << (fc&7))) == 0) break;
            Feptr++;
            }

          if (reptype == REPTYPE_POS) continue;    /* No backtracking */

          while (Feptr >= Lstart_eptr)
            {
            RMATCH(Fecode, RM24);
            if (rrc != MATCH_NOMATCH) RRETURN(rrc);
            Feptr--;
            }
          }

        RRETURN(MATCH_NOMATCH);
        }
      }
    /* Control never gets here */

#undef Lbyte_map_address
#undef Lbyte_map
#undef Lstart_eptr
#undef Lmin
#undef Lmax


    /* ===================================================================== */
    /* Match an extended character class. In the 8-bit library, this opcode is
    encountered only when UTF-8 mode mode is supported. In the 16-bit and
    32-bit libraries, codepoints greater than 255 may be encountered even when
    UTF is not supported. */

#define Lstart_eptr  F->temp_sptr[0]
#define Lxclass_data F->temp_sptr[1]
#define Lmin         F->temp_32[0]
#define Lmax         F->temp_32[1]

#ifdef SUPPORT_WIDE_CHARS
    case OP_XCLASS:
      {
      Lxclass_data = Fecode + 1 + LINK_SIZE;  /* Save for matching */
      Fecode += GET(Fecode, 1);               /* Advance past the item */

      switch (*Fecode)
        {
        case OP_CRSTAR:
        case OP_CRMINSTAR:
        case OP_CRPLUS:
        case OP_CRMINPLUS:
        case OP_CRQUERY:
        case OP_CRMINQUERY:
        case OP_CRPOSSTAR:
        case OP_CRPOSPLUS:
        case OP_CRPOSQUERY:
        fc = *Fecode++ - OP_CRSTAR;
        Lmin = rep_min[fc];
        Lmax = rep_max[fc];
        reptype = rep_typ[fc];
        break;

        case OP_CRRANGE:
        case OP_CRMINRANGE:
        case OP_CRPOSRANGE:
        Lmin = GET2(Fecode, 1);
        Lmax = GET2(Fecode, 1 + IMM2_SIZE);
        if (Lmax == 0) Lmax = UINT32_MAX;  /* Max 0 => infinity */
        reptype = rep_typ[*Fecode - OP_CRSTAR];
        Fecode += 1 + 2 * IMM2_SIZE;
        break;

        default:               /* No repeat follows */
        Lmin = Lmax = 1;
        break;
        }

      /* First, ensure the minimum number of matches are present. */

      for (i = 1; i <= Lmin; i++)
        {
        if (Feptr >= mb->end_subject)
          {
          SCHECK_PARTIAL();
          RRETURN(MATCH_NOMATCH);
          }
        GETCHARINCTEST(fc, Feptr);
        if (!PRIV(xclass)(fc, Lxclass_data, utf)) RRETURN(MATCH_NOMATCH);
        }

      /* If Lmax == Lmin we can just continue with the main loop. */

      if (Lmin == Lmax) continue;

      /* If minimizing, keep testing the rest of the expression and advancing
      the pointer while it matches the class. */

      if (reptype == REPTYPE_MIN)
        {
        for (;;)
          {
          RMATCH(Fecode, RM100);
          if (rrc != MATCH_NOMATCH) RRETURN(rrc);
          if (Lmin++ >= Lmax) RRETURN(MATCH_NOMATCH);
          if (Feptr >= mb->end_subject)
            {
            SCHECK_PARTIAL();
            RRETURN(MATCH_NOMATCH);
            }
          GETCHARINCTEST(fc, Feptr);
          if (!PRIV(xclass)(fc, Lxclass_data, utf)) RRETURN(MATCH_NOMATCH);
          }
        /* Control never gets here */
        }

      /* If maximizing, find the longest possible run, then work backwards. */

      else
        {
        Lstart_eptr = Feptr;
        for (i = Lmin; i < Lmax; i++)
          {
          int len = 1;
          if (Feptr >= mb->end_subject)
            {
            SCHECK_PARTIAL();
            break;
            }
#ifdef SUPPORT_UNICODE
          GETCHARLENTEST(fc, Feptr, len);
#else
          fc = *Feptr;
#endif
          if (!PRIV(xclass)(fc, Lxclass_data, utf)) break;
          Feptr += len;
          }

        if (reptype == REPTYPE_POS) continue;    /* No backtracking */

        /* After \C in UTF mode, Lstart_eptr might be in the middle of a
        Unicode character. Use <= Lstart_eptr to ensure backtracking doesn't
        go too far. */

        for(;;)
          {
          RMATCH(Fecode, RM101);
          if (rrc != MATCH_NOMATCH) RRETURN(rrc);
          if (Feptr-- <= Lstart_eptr) break;  /* Tried at original position */
#ifdef SUPPORT_UNICODE
          if (utf) BACKCHAR(Feptr);
#endif
          }
        RRETURN(MATCH_NOMATCH);
        }

      /* Control never gets here */
      }
#endif  /* SUPPORT_WIDE_CHARS: end of XCLASS */

#undef Lstart_eptr
#undef Lxclass_data
#undef Lmin
#undef Lmax


    /* ===================================================================== */
    /* Match various character types when PCRE2_UCP is not set. These opcodes
    are not generated when PCRE2_UCP is set - instead appropriate property
    tests are compiled. */

    case OP_NOT_DIGIT:
    if (Feptr >= mb->end_subject)
      {
      SCHECK_PARTIAL();
      RRETURN(MATCH_NOMATCH);
      }
    GETCHARINCTEST(fc, Feptr);
    if (CHMAX_255(fc) && (mb->ctypes[fc] & ctype_digit) != 0)
      RRETURN(MATCH_NOMATCH);
    Fecode++;
    break;

    case OP_DIGIT:
    if (Feptr >= mb->end_subject)
      {
      SCHECK_PARTIAL();
      RRETURN(MATCH_NOMATCH);
      }
    GETCHARINCTEST(fc, Feptr);
    if (!CHMAX_255(fc) || (mb->ctypes[fc] & ctype_digit) == 0)
      RRETURN(MATCH_NOMATCH);
    Fecode++;
    break;

    case OP_NOT_WHITESPACE:
    if (Feptr >= mb->end_subject)
      {
      SCHECK_PARTIAL();
      RRETURN(MATCH_NOMATCH);
      }
    GETCHARINCTEST(fc, Feptr);
    if (CHMAX_255(fc) && (mb->ctypes[fc] & ctype_space) != 0)
      RRETURN(MATCH_NOMATCH);
    Fecode++;
    break;

    case OP_WHITESPACE:
    if (Feptr >= mb->end_subject)
      {
      SCHECK_PARTIAL();
      RRETURN(MATCH_NOMATCH);
      }
    GETCHARINCTEST(fc, Feptr);
    if (!CHMAX_255(fc) || (mb->ctypes[fc] & ctype_space) == 0)
      RRETURN(MATCH_NOMATCH);
    Fecode++;
    break;

    case OP_NOT_WORDCHAR:
    if (Feptr >= mb->end_subject)
      {
      SCHECK_PARTIAL();
      RRETURN(MATCH_NOMATCH);
      }
    GETCHARINCTEST(fc, Feptr);
    if (CHMAX_255(fc) && (mb->ctypes[fc] & ctype_word) != 0)
      RRETURN(MATCH_NOMATCH);
    Fecode++;
    break;

    case OP_WORDCHAR:
    if (Feptr >= mb->end_subject)
      {
      SCHECK_PARTIAL();
      RRETURN(MATCH_NOMATCH);
      }
    GETCHARINCTEST(fc, Feptr);
    if (!CHMAX_255(fc) || (mb->ctypes[fc] & ctype_word) == 0)
      RRETURN(MATCH_NOMATCH);
    Fecode++;
    break;

    case OP_ANYNL:
    if (Feptr >= mb->end_subject)
      {
      SCHECK_PARTIAL();
      RRETURN(MATCH_NOMATCH);
      }
    GETCHARINCTEST(fc, Feptr);
    switch(fc)
      {
      default: RRETURN(MATCH_NOMATCH);

      case CHAR_CR:
      if (Feptr >= mb->end_subject)
        {
        SCHECK_PARTIAL();
        }
      else if (UCHAR21TEST(Feptr) == CHAR_LF) Feptr++;
      break;

      case CHAR_LF:
      break;

      case CHAR_VT:
      case CHAR_FF:
      case CHAR_NEL:
#ifndef EBCDIC
      case 0x2028:
      case 0x2029:
#endif  /* Not EBCDIC */
      if (mb->bsr_convention == PCRE2_BSR_ANYCRLF) RRETURN(MATCH_NOMATCH);
      break;
      }
    Fecode++;
    break;

    case OP_NOT_HSPACE:
    if (Feptr >= mb->end_subject)
      {
      SCHECK_PARTIAL();
      RRETURN(MATCH_NOMATCH);
      }
    GETCHARINCTEST(fc, Feptr);
    switch(fc)
      {
      HSPACE_CASES: RRETURN(MATCH_NOMATCH);  /* Byte and multibyte cases */
      default: break;
      }
    Fecode++;
    break;

    case OP_HSPACE:
    if (Feptr >= mb->end_subject)
      {
      SCHECK_PARTIAL();
      RRETURN(MATCH_NOMATCH);
      }
    GETCHARINCTEST(fc, Feptr);
    switch(fc)
      {
      HSPACE_CASES: break;  /* Byte and multibyte cases */
      default: RRETURN(MATCH_NOMATCH);
      }
    Fecode++;
    break;

    case OP_NOT_VSPACE:
    if (Feptr >= mb->end_subject)
      {
      SCHECK_PARTIAL();
      RRETURN(MATCH_NOMATCH);
      }
    GETCHARINCTEST(fc, Feptr);
    switch(fc)
      {
      VSPACE_CASES: RRETURN(MATCH_NOMATCH);
      default: break;
      }
    Fecode++;
    break;

    case OP_VSPACE:
    if (Feptr >= mb->end_subject)
      {
      SCHECK_PARTIAL();
      RRETURN(MATCH_NOMATCH);
      }
    GETCHARINCTEST(fc, Feptr);
    switch(fc)
      {
      VSPACE_CASES: break;
      default: RRETURN(MATCH_NOMATCH);
      }
    Fecode++;
    break;


#ifdef SUPPORT_UNICODE

    /* ===================================================================== */
    /* Check the next character by Unicode property. We will get here only
    if the support is in the binary; otherwise a compile-time error occurs. */

    case OP_PROP:
    case OP_NOTPROP:
    if (Feptr >= mb->end_subject)
      {
      SCHECK_PARTIAL();
      RRETURN(MATCH_NOMATCH);
      }
    GETCHARINCTEST(fc, Feptr);
      {
      const uint32_t *cp;
      const ucd_record *prop = GET_UCD(fc);

      switch(Fecode[1])
        {
        case PT_ANY:
        if (Fop == OP_NOTPROP) RRETURN(MATCH_NOMATCH);
        break;

        case PT_LAMP:
        if ((prop->chartype == ucp_Lu ||
             prop->chartype == ucp_Ll ||
             prop->chartype == ucp_Lt) == (Fop == OP_NOTPROP))
          RRETURN(MATCH_NOMATCH);
        break;

        case PT_GC:
        if ((Fecode[2] != PRIV(ucp_gentype)[prop->chartype]) == (Fop == OP_PROP))
          RRETURN(MATCH_NOMATCH);
        break;

        case PT_PC:
        if ((Fecode[2] != prop->chartype) == (Fop == OP_PROP))
          RRETURN(MATCH_NOMATCH);
        break;

        case PT_SC:
        if ((Fecode[2] != prop->script) == (Fop == OP_PROP))
          RRETURN(MATCH_NOMATCH);
        break;

        /* These are specials */

        case PT_ALNUM:
        if ((PRIV(ucp_gentype)[prop->chartype] == ucp_L ||
             PRIV(ucp_gentype)[prop->chartype] == ucp_N) == (Fop == OP_NOTPROP))
          RRETURN(MATCH_NOMATCH);
        break;

        /* Perl space used to exclude VT, but from Perl 5.18 it is included,
        which means that Perl space and POSIX space are now identical. PCRE
        was changed at release 8.34. */

        case PT_SPACE:    /* Perl space */
        case PT_PXSPACE:  /* POSIX space */
        switch(fc)
          {
          HSPACE_CASES:
          VSPACE_CASES:
          if (Fop == OP_NOTPROP) RRETURN(MATCH_NOMATCH);
          break;

          default:
          if ((PRIV(ucp_gentype)[prop->chartype] == ucp_Z) ==
            (Fop == OP_NOTPROP)) RRETURN(MATCH_NOMATCH);
          break;
          }
        break;

        case PT_WORD:
        if ((PRIV(ucp_gentype)[prop->chartype] == ucp_L ||
             PRIV(ucp_gentype)[prop->chartype] == ucp_N ||
             fc == CHAR_UNDERSCORE) == (Fop == OP_NOTPROP))
          RRETURN(MATCH_NOMATCH);
        break;

        case PT_CLIST:
        cp = PRIV(ucd_caseless_sets) + Fecode[2];
        for (;;)
          {
          if (fc < *cp)
            { if (Fop == OP_PROP) { RRETURN(MATCH_NOMATCH); } else break; }
          if (fc == *cp++)
            { if (Fop == OP_PROP) break; else { RRETURN(MATCH_NOMATCH); } }
          }
        break;

        case PT_UCNC:
        if ((fc == CHAR_DOLLAR_SIGN || fc == CHAR_COMMERCIAL_AT ||
             fc == CHAR_GRAVE_ACCENT || (fc >= 0xa0 && fc <= 0xd7ff) ||
             fc >= 0xe000) == (Fop == OP_NOTPROP))
          RRETURN(MATCH_NOMATCH);
        break;

        /* This should never occur */

        default:
        return PCRE2_ERROR_INTERNAL;
        }

      Fecode += 3;
      }
    break;


    /* ===================================================================== */
    /* Match an extended Unicode sequence. We will get here only if the support
    is in the binary; otherwise a compile-time error occurs. */

    case OP_EXTUNI:
    if (Feptr >= mb->end_subject)
      {
      SCHECK_PARTIAL();
      RRETURN(MATCH_NOMATCH);
      }
    else
      {
      GETCHARINCTEST(fc, Feptr);
      Feptr = PRIV(extuni)(fc, Feptr, mb->start_subject, mb->end_subject, utf,
        NULL);
      }
    CHECK_PARTIAL();
    Fecode++;
    break;

#endif  /* SUPPORT_UNICODE */


    /* ===================================================================== */
    /* Match a single character type repeatedly. Note that the property type
    does not need to be in a stack frame as it is not used within an RMATCH()
    loop. */

#define Lstart_eptr  F->temp_sptr[0]
#define Lmin         F->temp_32[0]
#define Lmax         F->temp_32[1]
#define Lctype       F->temp_32[2]
#define Lpropvalue   F->temp_32[3]

    case OP_TYPEEXACT:
    Lmin = Lmax = GET2(Fecode, 1);
    Fecode += 1 + IMM2_SIZE;
    goto REPEATTYPE;

    case OP_TYPEUPTO:
    case OP_TYPEMINUPTO:
    Lmin = 0;
    Lmax = GET2(Fecode, 1);
    reptype = (*Fecode == OP_TYPEMINUPTO)? REPTYPE_MIN : REPTYPE_MAX;
    Fecode += 1 + IMM2_SIZE;
    goto REPEATTYPE;

    case OP_TYPEPOSSTAR:
    reptype = REPTYPE_POS;
    Lmin = 0;
    Lmax = UINT32_MAX;
    Fecode++;
    goto REPEATTYPE;

    case OP_TYPEPOSPLUS:
    reptype = REPTYPE_POS;
    Lmin = 1;
    Lmax = UINT32_MAX;
    Fecode++;
    goto REPEATTYPE;

    case OP_TYPEPOSQUERY:
    reptype = REPTYPE_POS;
    Lmin = 0;
    Lmax = 1;
    Fecode++;
    goto REPEATTYPE;

    case OP_TYPEPOSUPTO:
    reptype = REPTYPE_POS;
    Lmin = 0;
    Lmax = GET2(Fecode, 1);
    Fecode += 1 + IMM2_SIZE;
    goto REPEATTYPE;

    case OP_TYPESTAR:
    case OP_TYPEMINSTAR:
    case OP_TYPEPLUS:
    case OP_TYPEMINPLUS:
    case OP_TYPEQUERY:
    case OP_TYPEMINQUERY:
    fc = *Fecode++ - OP_TYPESTAR;
    Lmin = rep_min[fc];
    Lmax = rep_max[fc];
    reptype = rep_typ[fc];

    /* Common code for all repeated character type matches. */

    REPEATTYPE:
    Lctype = *Fecode++;      /* Code for the character type */

#ifdef SUPPORT_UNICODE
    if (Lctype == OP_PROP || Lctype == OP_NOTPROP)
      {
      proptype = *Fecode++;
      Lpropvalue = *Fecode++;
      }
    else proptype = -1;
#endif

    /* First, ensure the minimum number of matches are present. Use inline
    code for maximizing the speed, and do the type test once at the start
    (i.e. keep it out of the loop). The code for UTF mode is separated out for
    tidiness, except for Unicode property tests. */

    if (Lmin > 0)
      {
#ifdef SUPPORT_UNICODE
      if (proptype >= 0)  /* Property tests in all modes */
        {
        switch(proptype)
          {
          case PT_ANY:
          if (Lctype == OP_NOTPROP) RRETURN(MATCH_NOMATCH);
          for (i = 1; i <= Lmin; i++)
            {
            if (Feptr >= mb->end_subject)
              {
              SCHECK_PARTIAL();
              RRETURN(MATCH_NOMATCH);
              }
            GETCHARINCTEST(fc, Feptr);
            }
          break;

          case PT_LAMP:
          for (i = 1; i <= Lmin; i++)
            {
            int chartype;
            if (Feptr >= mb->end_subject)
              {
              SCHECK_PARTIAL();
              RRETURN(MATCH_NOMATCH);
              }
            GETCHARINCTEST(fc, Feptr);
            chartype = UCD_CHARTYPE(fc);
            if ((chartype == ucp_Lu ||
                 chartype == ucp_Ll ||
                 chartype == ucp_Lt) == (Lctype == OP_NOTPROP))
              RRETURN(MATCH_NOMATCH);
            }
          break;

          case PT_GC:
          for (i = 1; i <= Lmin; i++)
            {
            if (Feptr >= mb->end_subject)
              {
              SCHECK_PARTIAL();
              RRETURN(MATCH_NOMATCH);
              }
            GETCHARINCTEST(fc, Feptr);
            if ((UCD_CATEGORY(fc) == Lpropvalue) == (Lctype == OP_NOTPROP))
              RRETURN(MATCH_NOMATCH);
            }
          break;

          case PT_PC:
          for (i = 1; i <= Lmin; i++)
            {
            if (Feptr >= mb->end_subject)
              {
              SCHECK_PARTIAL();
              RRETURN(MATCH_NOMATCH);
              }
            GETCHARINCTEST(fc, Feptr);
            if ((UCD_CHARTYPE(fc) == Lpropvalue) == (Lctype == OP_NOTPROP))
              RRETURN(MATCH_NOMATCH);
            }
          break;

          case PT_SC:
          for (i = 1; i <= Lmin; i++)
            {
            if (Feptr >= mb->end_subject)
              {
              SCHECK_PARTIAL();
              RRETURN(MATCH_NOMATCH);
              }
            GETCHARINCTEST(fc, Feptr);
            if ((UCD_SCRIPT(fc) == Lpropvalue) == (Lctype == OP_NOTPROP))
              RRETURN(MATCH_NOMATCH);
            }
          break;

          case PT_ALNUM:
          for (i = 1; i <= Lmin; i++)
            {
            int category;
            if (Feptr >= mb->end_subject)
              {
              SCHECK_PARTIAL();
              RRETURN(MATCH_NOMATCH);
              }
            GETCHARINCTEST(fc, Feptr);
            category = UCD_CATEGORY(fc);
            if ((category == ucp_L || category == ucp_N) == (Lctype == OP_NOTPROP))
              RRETURN(MATCH_NOMATCH);
            }
          break;

          /* Perl space used to exclude VT, but from Perl 5.18 it is included,
          which means that Perl space and POSIX space are now identical. PCRE
          was changed at release 8.34. */

          case PT_SPACE:    /* Perl space */
          case PT_PXSPACE:  /* POSIX space */
          for (i = 1; i <= Lmin; i++)
            {
            if (Feptr >= mb->end_subject)
              {
              SCHECK_PARTIAL();
              RRETURN(MATCH_NOMATCH);
              }
            GETCHARINCTEST(fc, Feptr);
            switch(fc)
              {
              HSPACE_CASES:
              VSPACE_CASES:
              if (Lctype == OP_NOTPROP) RRETURN(MATCH_NOMATCH);
              break;

              default:
              if ((UCD_CATEGORY(fc) == ucp_Z) == (Lctype == OP_NOTPROP))
                RRETURN(MATCH_NOMATCH);
              break;
              }
            }
          break;

          case PT_WORD:
          for (i = 1; i <= Lmin; i++)
            {
            int category;
            if (Feptr >= mb->end_subject)
              {
              SCHECK_PARTIAL();
              RRETURN(MATCH_NOMATCH);
              }
            GETCHARINCTEST(fc, Feptr);
            category = UCD_CATEGORY(fc);
            if ((category == ucp_L || category == ucp_N ||
                fc == CHAR_UNDERSCORE) == (Lctype == OP_NOTPROP))
              RRETURN(MATCH_NOMATCH);
            }
          break;

          case PT_CLIST:
          for (i = 1; i <= Lmin; i++)
            {
            const uint32_t *cp;
            if (Feptr >= mb->end_subject)
              {
              SCHECK_PARTIAL();
              RRETURN(MATCH_NOMATCH);
              }
            GETCHARINCTEST(fc, Feptr);
            cp = PRIV(ucd_caseless_sets) + Lpropvalue;
            for (;;)
              {
              if (fc < *cp)
                {
                if (Lctype == OP_NOTPROP) break;
                RRETURN(MATCH_NOMATCH);
                }
              if (fc == *cp++)
                {
                if (Lctype == OP_NOTPROP) RRETURN(MATCH_NOMATCH);
                break;
                }
              }
            }
          break;

          case PT_UCNC:
          for (i = 1; i <= Lmin; i++)
            {
            if (Feptr >= mb->end_subject)
              {
              SCHECK_PARTIAL();
              RRETURN(MATCH_NOMATCH);
              }
            GETCHARINCTEST(fc, Feptr);
            if ((fc == CHAR_DOLLAR_SIGN || fc == CHAR_COMMERCIAL_AT ||
                 fc == CHAR_GRAVE_ACCENT || (fc >= 0xa0 && fc <= 0xd7ff) ||
                 fc >= 0xe000) == (Lctype == OP_NOTPROP))
              RRETURN(MATCH_NOMATCH);
            }
          break;

          /* This should not occur */

          default:
          return PCRE2_ERROR_INTERNAL;
          }
        }

      /* Match extended Unicode sequences. We will get here only if the
      support is in the binary; otherwise a compile-time error occurs. */

      else if (Lctype == OP_EXTUNI)
        {
        for (i = 1; i <= Lmin; i++)
          {
          if (Feptr >= mb->end_subject)
            {
            SCHECK_PARTIAL();
            RRETURN(MATCH_NOMATCH);
            }
          else
            {
            GETCHARINCTEST(fc, Feptr);
            Feptr = PRIV(extuni)(fc, Feptr, mb->start_subject,
              mb->end_subject, utf, NULL);
            }
          CHECK_PARTIAL();
          }
        }
      else
#endif     /* SUPPORT_UNICODE */

/* Handle all other cases in UTF mode */

#ifdef SUPPORT_UNICODE
      if (utf) switch(Lctype)
        {
        case OP_ANY:
        for (i = 1; i <= Lmin; i++)
          {
          if (Feptr >= mb->end_subject)
            {
            SCHECK_PARTIAL();
            RRETURN(MATCH_NOMATCH);
            }
          if (IS_NEWLINE(Feptr)) RRETURN(MATCH_NOMATCH);
          if (mb->partial != 0 &&
              Feptr + 1 >= mb->end_subject &&
              NLBLOCK->nltype == NLTYPE_FIXED &&
              NLBLOCK->nllen == 2 &&
              UCHAR21(Feptr) == NLBLOCK->nl[0])
            {
            mb->hitend = TRUE;
            if (mb->partial > 1) return PCRE2_ERROR_PARTIAL;
            }
          Feptr++;
          ACROSSCHAR(Feptr < mb->end_subject, Feptr, Feptr++);
          }
        break;

        case OP_ALLANY:
        for (i = 1; i <= Lmin; i++)
          {
          if (Feptr >= mb->end_subject)
            {
            SCHECK_PARTIAL();
            RRETURN(MATCH_NOMATCH);
            }
          Feptr++;
          ACROSSCHAR(Feptr < mb->end_subject, Feptr, Feptr++);
          }
        break;

        case OP_ANYBYTE:
        if (Feptr > mb->end_subject - Lmin) RRETURN(MATCH_NOMATCH);
        Feptr += Lmin;
        break;

        case OP_ANYNL:
        for (i = 1; i <= Lmin; i++)
          {
          if (Feptr >= mb->end_subject)
            {
            SCHECK_PARTIAL();
            RRETURN(MATCH_NOMATCH);
            }
          GETCHARINC(fc, Feptr);
          switch(fc)
            {
            default: RRETURN(MATCH_NOMATCH);

            case CHAR_CR:
            if (Feptr < mb->end_subject && UCHAR21(Feptr) == CHAR_LF) Feptr++;
            break;

            case CHAR_LF:
            break;

            case CHAR_VT:
            case CHAR_FF:
            case CHAR_NEL:
#ifndef EBCDIC
            case 0x2028:
            case 0x2029:
#endif  /* Not EBCDIC */
            if (mb->bsr_convention == PCRE2_BSR_ANYCRLF) RRETURN(MATCH_NOMATCH);
            break;
            }
          }
        break;

        case OP_NOT_HSPACE:
        for (i = 1; i <= Lmin; i++)
          {
          if (Feptr >= mb->end_subject)
            {
            SCHECK_PARTIAL();
            RRETURN(MATCH_NOMATCH);
            }
          GETCHARINC(fc, Feptr);
          switch(fc)
            {
            HSPACE_CASES: RRETURN(MATCH_NOMATCH);
            default: break;
            }
          }
        break;

        case OP_HSPACE:
        for (i = 1; i <= Lmin; i++)
          {
          if (Feptr >= mb->end_subject)
            {
            SCHECK_PARTIAL();
            RRETURN(MATCH_NOMATCH);
            }
          GETCHARINC(fc, Feptr);
          switch(fc)
            {
            HSPACE_CASES: break;
            default: RRETURN(MATCH_NOMATCH);
            }
          }
        break;

        case OP_NOT_VSPACE:
        for (i = 1; i <= Lmin; i++)
          {
          if (Feptr >= mb->end_subject)
            {
            SCHECK_PARTIAL();
            RRETURN(MATCH_NOMATCH);
            }
          GETCHARINC(fc, Feptr);
          switch(fc)
            {
            VSPACE_CASES: RRETURN(MATCH_NOMATCH);
            default: break;
            }
          }
        break;

        case OP_VSPACE:
        for (i = 1; i <= Lmin; i++)
          {
          if (Feptr >= mb->end_subject)
            {
            SCHECK_PARTIAL();
            RRETURN(MATCH_NOMATCH);
            }
          GETCHARINC(fc, Feptr);
          switch(fc)
            {
            VSPACE_CASES: break;
            default: RRETURN(MATCH_NOMATCH);
            }
          }
        break;

        case OP_NOT_DIGIT:
        for (i = 1; i <= Lmin; i++)
          {
          if (Feptr >= mb->end_subject)
            {
            SCHECK_PARTIAL();
            RRETURN(MATCH_NOMATCH);
            }
          GETCHARINC(fc, Feptr);
          if (fc < 128 && (mb->ctypes[fc] & ctype_digit) != 0)
            RRETURN(MATCH_NOMATCH);
          }
        break;

        case OP_DIGIT:
        for (i = 1; i <= Lmin; i++)
          {
          uint32_t cc;
          if (Feptr >= mb->end_subject)
            {
            SCHECK_PARTIAL();
            RRETURN(MATCH_NOMATCH);
            }
          cc = UCHAR21(Feptr);
          if (cc >= 128 || (mb->ctypes[cc] & ctype_digit) == 0)
            RRETURN(MATCH_NOMATCH);
          Feptr++;
          /* No need to skip more code units - we know it has only one. */
          }
        break;

        case OP_NOT_WHITESPACE:
        for (i = 1; i <= Lmin; i++)
          {
          uint32_t cc;
          if (Feptr >= mb->end_subject)
            {
            SCHECK_PARTIAL();
            RRETURN(MATCH_NOMATCH);
            }
          cc = UCHAR21(Feptr);
          if (cc < 128 && (mb->ctypes[cc] & ctype_space) != 0)
            RRETURN(MATCH_NOMATCH);
          Feptr++;
          ACROSSCHAR(Feptr < mb->end_subject, Feptr, Feptr++);
          }
        break;

        case OP_WHITESPACE:
        for (i = 1; i <= Lmin; i++)
          {
          uint32_t cc;
          if (Feptr >= mb->end_subject)
            {
            SCHECK_PARTIAL();
            RRETURN(MATCH_NOMATCH);
            }
          cc = UCHAR21(Feptr);
          if (cc >= 128 || (mb->ctypes[cc] & ctype_space) == 0)
            RRETURN(MATCH_NOMATCH);
          Feptr++;
          /* No need to skip more code units - we know it has only one. */
          }
        break;

        case OP_NOT_WORDCHAR:
        for (i = 1; i <= Lmin; i++)
          {
          uint32_t cc;
          if (Feptr >= mb->end_subject)
            {
            SCHECK_PARTIAL();
            RRETURN(MATCH_NOMATCH);
            }
          cc = UCHAR21(Feptr);
          if (cc < 128 && (mb->ctypes[cc] & ctype_word) != 0)
            RRETURN(MATCH_NOMATCH);
          Feptr++;
          ACROSSCHAR(Feptr < mb->end_subject, Feptr, Feptr++);
          }
        break;

        case OP_WORDCHAR:
        for (i = 1; i <= Lmin; i++)
          {
          uint32_t cc;
          if (Feptr >= mb->end_subject)
            {
            SCHECK_PARTIAL();
            RRETURN(MATCH_NOMATCH);
            }
          cc = UCHAR21(Feptr);
          if (cc >= 128 || (mb->ctypes[cc] & ctype_word) == 0)
            RRETURN(MATCH_NOMATCH);
          Feptr++;
          /* No need to skip more code units - we know it has only one. */
          }
        break;

        default:
        return PCRE2_ERROR_INTERNAL;
        }  /* End switch(Lctype) */

      else
#endif     /* SUPPORT_UNICODE */

      /* Code for the non-UTF case for minimum matching of operators other
      than OP_PROP and OP_NOTPROP. */

      switch(Lctype)
        {
        case OP_ANY:
        for (i = 1; i <= Lmin; i++)
          {
          if (Feptr >= mb->end_subject)
            {
            SCHECK_PARTIAL();
            RRETURN(MATCH_NOMATCH);
            }
          if (IS_NEWLINE(Feptr)) RRETURN(MATCH_NOMATCH);
          if (mb->partial != 0 &&
              Feptr + 1 >= mb->end_subject &&
              NLBLOCK->nltype == NLTYPE_FIXED &&
              NLBLOCK->nllen == 2 &&
              *Feptr == NLBLOCK->nl[0])
            {
            mb->hitend = TRUE;
            if (mb->partial > 1) return PCRE2_ERROR_PARTIAL;
            }
          Feptr++;
          }
        break;

        case OP_ALLANY:
        if (Feptr > mb->end_subject - Lmin)
          {
          SCHECK_PARTIAL();
          RRETURN(MATCH_NOMATCH);
          }
        Feptr += Lmin;
        break;

        /* This OP_ANYBYTE case will never be reached because \C gets turned
        into OP_ALLANY in non-UTF mode. Cut out the code so that coverage
        reports don't complain about it's never being used. */

/*        case OP_ANYBYTE:
*        if (Feptr > mb->end_subject - Lmin)
*          {
*          SCHECK_PARTIAL();
*          RRETURN(MATCH_NOMATCH);
*          }
*        Feptr += Lmin;
*        break;
*/
        case OP_ANYNL:
        for (i = 1; i <= Lmin; i++)
          {
          if (Feptr >= mb->end_subject)
            {
            SCHECK_PARTIAL();
            RRETURN(MATCH_NOMATCH);
            }
          switch(*Feptr++)
            {
            default: RRETURN(MATCH_NOMATCH);

            case CHAR_CR:
            if (Feptr < mb->end_subject && *Feptr == CHAR_LF) Feptr++;
            break;

            case CHAR_LF:
            break;

            case CHAR_VT:
            case CHAR_FF:
            case CHAR_NEL:
#if PCRE2_CODE_UNIT_WIDTH != 8
            case 0x2028:
            case 0x2029:
#endif
            if (mb->bsr_convention == PCRE2_BSR_ANYCRLF) RRETURN(MATCH_NOMATCH);
            break;
            }
          }
        break;

        case OP_NOT_HSPACE:
        for (i = 1; i <= Lmin; i++)
          {
          if (Feptr >= mb->end_subject)
            {
            SCHECK_PARTIAL();
            RRETURN(MATCH_NOMATCH);
            }
          switch(*Feptr++)
            {
            default: break;
            HSPACE_BYTE_CASES:
#if PCRE2_CODE_UNIT_WIDTH != 8
            HSPACE_MULTIBYTE_CASES:
#endif
            RRETURN(MATCH_NOMATCH);
            }
          }
        break;

        case OP_HSPACE:
        for (i = 1; i <= Lmin; i++)
          {
          if (Feptr >= mb->end_subject)
            {
            SCHECK_PARTIAL();
            RRETURN(MATCH_NOMATCH);
            }
          switch(*Feptr++)
            {
            default: RRETURN(MATCH_NOMATCH);
            HSPACE_BYTE_CASES:
#if PCRE2_CODE_UNIT_WIDTH != 8
            HSPACE_MULTIBYTE_CASES:
#endif
            break;
            }
          }
        break;

        case OP_NOT_VSPACE:
        for (i = 1; i <= Lmin; i++)
          {
          if (Feptr >= mb->end_subject)
            {
            SCHECK_PARTIAL();
            RRETURN(MATCH_NOMATCH);
            }
          switch(*Feptr++)
            {
            VSPACE_BYTE_CASES:
#if PCRE2_CODE_UNIT_WIDTH != 8
            VSPACE_MULTIBYTE_CASES:
#endif
            RRETURN(MATCH_NOMATCH);
            default: break;
            }
          }
        break;

        case OP_VSPACE:
        for (i = 1; i <= Lmin; i++)
          {
          if (Feptr >= mb->end_subject)
            {
            SCHECK_PARTIAL();
            RRETURN(MATCH_NOMATCH);
            }
          switch(*Feptr++)
            {
            default: RRETURN(MATCH_NOMATCH);
            VSPACE_BYTE_CASES:
#if PCRE2_CODE_UNIT_WIDTH != 8
            VSPACE_MULTIBYTE_CASES:
#endif
            break;
            }
          }
        break;

        case OP_NOT_DIGIT:
        for (i = 1; i <= Lmin; i++)
          {
          if (Feptr >= mb->end_subject)
            {
            SCHECK_PARTIAL();
            RRETURN(MATCH_NOMATCH);
            }
          if (MAX_255(*Feptr) && (mb->ctypes[*Feptr] & ctype_digit) != 0)
            RRETURN(MATCH_NOMATCH);
          Feptr++;
          }
        break;

        case OP_DIGIT:
        for (i = 1; i <= Lmin; i++)
          {
          if (Feptr >= mb->end_subject)
            {
            SCHECK_PARTIAL();
            RRETURN(MATCH_NOMATCH);
            }
          if (!MAX_255(*Feptr) || (mb->ctypes[*Feptr] & ctype_digit) == 0)
            RRETURN(MATCH_NOMATCH);
          Feptr++;
          }
        break;

        case OP_NOT_WHITESPACE:
        for (i = 1; i <= Lmin; i++)
          {
          if (Feptr >= mb->end_subject)
            {
            SCHECK_PARTIAL();
            RRETURN(MATCH_NOMATCH);
            }
          if (MAX_255(*Feptr) && (mb->ctypes[*Feptr] & ctype_space) != 0)
            RRETURN(MATCH_NOMATCH);
          Feptr++;
          }
        break;

        case OP_WHITESPACE:
        for (i = 1; i <= Lmin; i++)
          {
          if (Feptr >= mb->end_subject)
            {
            SCHECK_PARTIAL();
            RRETURN(MATCH_NOMATCH);
            }
          if (!MAX_255(*Feptr) || (mb->ctypes[*Feptr] & ctype_space) == 0)
            RRETURN(MATCH_NOMATCH);
          Feptr++;
          }
        break;

        case OP_NOT_WORDCHAR:
        for (i = 1; i <= Lmin; i++)
          {
          if (Feptr >= mb->end_subject)
            {
            SCHECK_PARTIAL();
            RRETURN(MATCH_NOMATCH);
            }
          if (MAX_255(*Feptr) && (mb->ctypes[*Feptr] & ctype_word) != 0)
            RRETURN(MATCH_NOMATCH);
          Feptr++;
          }
        break;

        case OP_WORDCHAR:
        for (i = 1; i <= Lmin; i++)
          {
          if (Feptr >= mb->end_subject)
            {
            SCHECK_PARTIAL();
            RRETURN(MATCH_NOMATCH);
            }
          if (!MAX_255(*Feptr) || (mb->ctypes[*Feptr] & ctype_word) == 0)
            RRETURN(MATCH_NOMATCH);
          Feptr++;
          }
        break;

        default:
        return PCRE2_ERROR_INTERNAL;
        }
      }

    /* If Lmin = Lmax we are done. Continue with the main loop. */

    if (Lmin == Lmax) continue;

    /* If minimizing, we have to test the rest of the pattern before each
    subsequent match. */

    if (reptype == REPTYPE_MIN)
      {
#ifdef SUPPORT_UNICODE
      if (proptype >= 0)
        {
        switch(proptype)
          {
          case PT_ANY:
          for (;;)
            {
            RMATCH(Fecode, RM208);
            if (rrc != MATCH_NOMATCH) RRETURN(rrc);
            if (Lmin++ >= Lmax) RRETURN(MATCH_NOMATCH);
            if (Feptr >= mb->end_subject)
              {
              SCHECK_PARTIAL();
              RRETURN(MATCH_NOMATCH);
              }
            GETCHARINCTEST(fc, Feptr);
            if (Lctype == OP_NOTPROP) RRETURN(MATCH_NOMATCH);
            }
          /* Control never gets here */

          case PT_LAMP:
          for (;;)
            {
            int chartype;
            RMATCH(Fecode, RM209);
            if (rrc != MATCH_NOMATCH) RRETURN(rrc);
            if (Lmin++ >= Lmax) RRETURN(MATCH_NOMATCH);
            if (Feptr >= mb->end_subject)
              {
              SCHECK_PARTIAL();
              RRETURN(MATCH_NOMATCH);
              }
            GETCHARINCTEST(fc, Feptr);
            chartype = UCD_CHARTYPE(fc);
            if ((chartype == ucp_Lu ||
                 chartype == ucp_Ll ||
                 chartype == ucp_Lt) == (Lctype == OP_NOTPROP))
              RRETURN(MATCH_NOMATCH);
            }
          /* Control never gets here */

          case PT_GC:
          for (;;)
            {
            RMATCH(Fecode, RM210);
            if (rrc != MATCH_NOMATCH) RRETURN(rrc);
            if (Lmin++ >= Lmax) RRETURN(MATCH_NOMATCH);
            if (Feptr >= mb->end_subject)
              {
              SCHECK_PARTIAL();
              RRETURN(MATCH_NOMATCH);
              }
            GETCHARINCTEST(fc, Feptr);
            if ((UCD_CATEGORY(fc) == Lpropvalue) == (Lctype == OP_NOTPROP))
              RRETURN(MATCH_NOMATCH);
            }
          /* Control never gets here */

          case PT_PC:
          for (;;)
            {
            RMATCH(Fecode, RM211);
            if (rrc != MATCH_NOMATCH) RRETURN(rrc);
            if (Lmin++ >= Lmax) RRETURN(MATCH_NOMATCH);
            if (Feptr >= mb->end_subject)
              {
              SCHECK_PARTIAL();
              RRETURN(MATCH_NOMATCH);
              }
            GETCHARINCTEST(fc, Feptr);
            if ((UCD_CHARTYPE(fc) == Lpropvalue) == (Lctype == OP_NOTPROP))
              RRETURN(MATCH_NOMATCH);
            }
          /* Control never gets here */

          case PT_SC:
          for (;;)
            {
            RMATCH(Fecode, RM212);
            if (rrc != MATCH_NOMATCH) RRETURN(rrc);
            if (Lmin++ >= Lmax) RRETURN(MATCH_NOMATCH);
            if (Feptr >= mb->end_subject)
              {
              SCHECK_PARTIAL();
              RRETURN(MATCH_NOMATCH);
              }
            GETCHARINCTEST(fc, Feptr);
            if ((UCD_SCRIPT(fc) == Lpropvalue) == (Lctype == OP_NOTPROP))
              RRETURN(MATCH_NOMATCH);
            }
          /* Control never gets here */

          case PT_ALNUM:
          for (;;)
            {
            int category;
            RMATCH(Fecode, RM213);
            if (rrc != MATCH_NOMATCH) RRETURN(rrc);
            if (Lmin++ >= Lmax) RRETURN(MATCH_NOMATCH);
            if (Feptr >= mb->end_subject)
              {
              SCHECK_PARTIAL();
              RRETURN(MATCH_NOMATCH);
              }
            GETCHARINCTEST(fc, Feptr);
            category = UCD_CATEGORY(fc);
            if ((category == ucp_L || category == ucp_N) ==
                (Lctype == OP_NOTPROP))
              RRETURN(MATCH_NOMATCH);
            }
          /* Control never gets here */

          /* Perl space used to exclude VT, but from Perl 5.18 it is included,
          which means that Perl space and POSIX space are now identical. PCRE
          was changed at release 8.34. */

          case PT_SPACE:    /* Perl space */
          case PT_PXSPACE:  /* POSIX space */
          for (;;)
            {
            RMATCH(Fecode, RM214);
            if (rrc != MATCH_NOMATCH) RRETURN(rrc);
            if (Lmin++ >= Lmax) RRETURN(MATCH_NOMATCH);
            if (Feptr >= mb->end_subject)
              {
              SCHECK_PARTIAL();
              RRETURN(MATCH_NOMATCH);
              }
            GETCHARINCTEST(fc, Feptr);
            switch(fc)
              {
              HSPACE_CASES:
              VSPACE_CASES:
              if (Lctype == OP_NOTPROP) RRETURN(MATCH_NOMATCH);
              break;

              default:
              if ((UCD_CATEGORY(fc) == ucp_Z) == (Lctype == OP_NOTPROP))
                RRETURN(MATCH_NOMATCH);
              break;
              }
            }
          /* Control never gets here */

          case PT_WORD:
          for (;;)
            {
            int category;
            RMATCH(Fecode, RM215);
            if (rrc != MATCH_NOMATCH) RRETURN(rrc);
            if (Lmin++ >= Lmax) RRETURN(MATCH_NOMATCH);
            if (Feptr >= mb->end_subject)
              {
              SCHECK_PARTIAL();
              RRETURN(MATCH_NOMATCH);
              }
            GETCHARINCTEST(fc, Feptr);
            category = UCD_CATEGORY(fc);
            if ((category == ucp_L ||
                 category == ucp_N ||
                 fc == CHAR_UNDERSCORE) == (Lctype == OP_NOTPROP))
              RRETURN(MATCH_NOMATCH);
            }
          /* Control never gets here */

          case PT_CLIST:
          for (;;)
            {
            const uint32_t *cp;
            RMATCH(Fecode, RM216);
            if (rrc != MATCH_NOMATCH) RRETURN(rrc);
            if (Lmin++ >= Lmax) RRETURN(MATCH_NOMATCH);
            if (Feptr >= mb->end_subject)
              {
              SCHECK_PARTIAL();
              RRETURN(MATCH_NOMATCH);
              }
            GETCHARINCTEST(fc, Feptr);
            cp = PRIV(ucd_caseless_sets) + Lpropvalue;
            for (;;)
              {
              if (fc < *cp)
                {
                if (Lctype == OP_NOTPROP) break;
                RRETURN(MATCH_NOMATCH);
                }
              if (fc == *cp++)
                {
                if (Lctype == OP_NOTPROP) RRETURN(MATCH_NOMATCH);
                break;
                }
              }
            }
          /* Control never gets here */

          case PT_UCNC:
          for (;;)
            {
            RMATCH(Fecode, RM217);
            if (rrc != MATCH_NOMATCH) RRETURN(rrc);
            if (Lmin++ >= Lmax) RRETURN(MATCH_NOMATCH);
            if (Feptr >= mb->end_subject)
              {
              SCHECK_PARTIAL();
              RRETURN(MATCH_NOMATCH);
              }
            GETCHARINCTEST(fc, Feptr);
            if ((fc == CHAR_DOLLAR_SIGN || fc == CHAR_COMMERCIAL_AT ||
                 fc == CHAR_GRAVE_ACCENT || (fc >= 0xa0 && fc <= 0xd7ff) ||
                 fc >= 0xe000) == (Lctype == OP_NOTPROP))
              RRETURN(MATCH_NOMATCH);
            }
          /* Control never gets here */

          /* This should never occur */
          default:
          return PCRE2_ERROR_INTERNAL;
          }
        }

      /* Match extended Unicode sequences. We will get here only if the
      support is in the binary; otherwise a compile-time error occurs. */

      else if (Lctype == OP_EXTUNI)
        {
        for (;;)
          {
          RMATCH(Fecode, RM218);
          if (rrc != MATCH_NOMATCH) RRETURN(rrc);
          if (Lmin++ >= Lmax) RRETURN(MATCH_NOMATCH);
          if (Feptr >= mb->end_subject)
            {
            SCHECK_PARTIAL();
            RRETURN(MATCH_NOMATCH);
            }
          else
            {
            GETCHARINCTEST(fc, Feptr);
            Feptr = PRIV(extuni)(fc, Feptr, mb->start_subject, mb->end_subject,
              utf, NULL);
            }
          CHECK_PARTIAL();
          }
        }
      else
#endif     /* SUPPORT_UNICODE */

      /* UTF mode for non-property testing character types. */

#ifdef SUPPORT_UNICODE
      if (utf)
        {
        for (;;)
          {
          RMATCH(Fecode, RM219);
          if (rrc != MATCH_NOMATCH) RRETURN(rrc);
          if (Lmin++ >= Lmax) RRETURN(MATCH_NOMATCH);
          if (Feptr >= mb->end_subject)
            {
            SCHECK_PARTIAL();
            RRETURN(MATCH_NOMATCH);
            }
          if (Lctype == OP_ANY && IS_NEWLINE(Feptr)) RRETURN(MATCH_NOMATCH);
          GETCHARINC(fc, Feptr);
          switch(Lctype)
            {
            case OP_ANY:               /* This is the non-NL case */
            if (mb->partial != 0 &&    /* Take care with CRLF partial */
                Feptr >= mb->end_subject &&
                NLBLOCK->nltype == NLTYPE_FIXED &&
                NLBLOCK->nllen == 2 &&
                fc == NLBLOCK->nl[0])
              {
              mb->hitend = TRUE;
              if (mb->partial > 1) return PCRE2_ERROR_PARTIAL;
              }
            break;

            case OP_ALLANY:
            case OP_ANYBYTE:
            break;

            case OP_ANYNL:
            switch(fc)
              {
              default: RRETURN(MATCH_NOMATCH);

              case CHAR_CR:
              if (Feptr < mb->end_subject && UCHAR21(Feptr) == CHAR_LF) Feptr++;
              break;

              case CHAR_LF:
              break;

              case CHAR_VT:
              case CHAR_FF:
              case CHAR_NEL:
#ifndef EBCDIC
              case 0x2028:
              case 0x2029:
#endif  /* Not EBCDIC */
              if (mb->bsr_convention == PCRE2_BSR_ANYCRLF)
                RRETURN(MATCH_NOMATCH);
              break;
              }
            break;

            case OP_NOT_HSPACE:
            switch(fc)
              {
              HSPACE_CASES: RRETURN(MATCH_NOMATCH);
              default: break;
              }
            break;

            case OP_HSPACE:
            switch(fc)
              {
              HSPACE_CASES: break;
              default: RRETURN(MATCH_NOMATCH);
              }
            break;

            case OP_NOT_VSPACE:
            switch(fc)
              {
              VSPACE_CASES: RRETURN(MATCH_NOMATCH);
              default: break;
              }
            break;

            case OP_VSPACE:
            switch(fc)
              {
              VSPACE_CASES: break;
              default: RRETURN(MATCH_NOMATCH);
              }
            break;

            case OP_NOT_DIGIT:
            if (fc < 256 && (mb->ctypes[fc] & ctype_digit) != 0)
              RRETURN(MATCH_NOMATCH);
            break;

            case OP_DIGIT:
            if (fc >= 256 || (mb->ctypes[fc] & ctype_digit) == 0)
              RRETURN(MATCH_NOMATCH);
            break;

            case OP_NOT_WHITESPACE:
            if (fc < 256 && (mb->ctypes[fc] & ctype_space) != 0)
              RRETURN(MATCH_NOMATCH);
            break;

            case OP_WHITESPACE:
            if (fc >= 256 || (mb->ctypes[fc] & ctype_space) == 0)
              RRETURN(MATCH_NOMATCH);
            break;

            case OP_NOT_WORDCHAR:
            if (fc < 256 && (mb->ctypes[fc] & ctype_word) != 0)
              RRETURN(MATCH_NOMATCH);
            break;

            case OP_WORDCHAR:
            if (fc >= 256 || (mb->ctypes[fc] & ctype_word) == 0)
              RRETURN(MATCH_NOMATCH);
            break;

            default:
            return PCRE2_ERROR_INTERNAL;
            }
          }
        }
      else
#endif  /* SUPPORT_UNICODE */

      /* Not UTF mode */
        {
        for (;;)
          {
          RMATCH(Fecode, RM33);
          if (rrc != MATCH_NOMATCH) RRETURN(rrc);
          if (Lmin++ >= Lmax) RRETURN(MATCH_NOMATCH);
          if (Feptr >= mb->end_subject)
            {
            SCHECK_PARTIAL();
            RRETURN(MATCH_NOMATCH);
            }
          if (Lctype == OP_ANY && IS_NEWLINE(Feptr))
            RRETURN(MATCH_NOMATCH);
          fc = *Feptr++;
          switch(Lctype)
            {
            case OP_ANY:               /* This is the non-NL case */
            if (mb->partial != 0 &&    /* Take care with CRLF partial */
                Feptr >= mb->end_subject &&
                NLBLOCK->nltype == NLTYPE_FIXED &&
                NLBLOCK->nllen == 2 &&
                fc == NLBLOCK->nl[0])
              {
              mb->hitend = TRUE;
              if (mb->partial > 1) return PCRE2_ERROR_PARTIAL;
              }
            break;

            case OP_ALLANY:
            case OP_ANYBYTE:
            break;

            case OP_ANYNL:
            switch(fc)
              {
              default: RRETURN(MATCH_NOMATCH);

              case CHAR_CR:
              if (Feptr < mb->end_subject && *Feptr == CHAR_LF) Feptr++;
              break;

              case CHAR_LF:
              break;

              case CHAR_VT:
              case CHAR_FF:
              case CHAR_NEL:
#if PCRE2_CODE_UNIT_WIDTH != 8
              case 0x2028:
              case 0x2029:
#endif
              if (mb->bsr_convention == PCRE2_BSR_ANYCRLF)
                RRETURN(MATCH_NOMATCH);
              break;
              }
            break;

            case OP_NOT_HSPACE:
            switch(fc)
              {
              default: break;
              HSPACE_BYTE_CASES:
#if PCRE2_CODE_UNIT_WIDTH != 8
              HSPACE_MULTIBYTE_CASES:
#endif
              RRETURN(MATCH_NOMATCH);
              }
            break;

            case OP_HSPACE:
            switch(fc)
              {
              default: RRETURN(MATCH_NOMATCH);
              HSPACE_BYTE_CASES:
#if PCRE2_CODE_UNIT_WIDTH != 8
              HSPACE_MULTIBYTE_CASES:
#endif
              break;
              }
            break;

            case OP_NOT_VSPACE:
            switch(fc)
              {
              default: break;
              VSPACE_BYTE_CASES:
#if PCRE2_CODE_UNIT_WIDTH != 8
              VSPACE_MULTIBYTE_CASES:
#endif
              RRETURN(MATCH_NOMATCH);
              }
            break;

            case OP_VSPACE:
            switch(fc)
              {
              default: RRETURN(MATCH_NOMATCH);
              VSPACE_BYTE_CASES:
#if PCRE2_CODE_UNIT_WIDTH != 8
              VSPACE_MULTIBYTE_CASES:
#endif
              break;
              }
            break;

            case OP_NOT_DIGIT:
            if (MAX_255(fc) && (mb->ctypes[fc] & ctype_digit) != 0)
              RRETURN(MATCH_NOMATCH);
            break;

            case OP_DIGIT:
            if (!MAX_255(fc) || (mb->ctypes[fc] & ctype_digit) == 0)
              RRETURN(MATCH_NOMATCH);
            break;

            case OP_NOT_WHITESPACE:
            if (MAX_255(fc) && (mb->ctypes[fc] & ctype_space) != 0)
              RRETURN(MATCH_NOMATCH);
            break;

            case OP_WHITESPACE:
            if (!MAX_255(fc) || (mb->ctypes[fc] & ctype_space) == 0)
              RRETURN(MATCH_NOMATCH);
            break;

            case OP_NOT_WORDCHAR:
            if (MAX_255(fc) && (mb->ctypes[fc] & ctype_word) != 0)
              RRETURN(MATCH_NOMATCH);
            break;

            case OP_WORDCHAR:
            if (!MAX_255(fc) || (mb->ctypes[fc] & ctype_word) == 0)
              RRETURN(MATCH_NOMATCH);
            break;

            default:
            return PCRE2_ERROR_INTERNAL;
            }
          }
        }
      /* Control never gets here */
      }

    /* If maximizing, it is worth using inline code for speed, doing the type
    test once at the start (i.e. keep it out of the loop). */

    else
      {
      Lstart_eptr = Feptr;  /* Remember where we started */

#ifdef SUPPORT_UNICODE
      if (proptype >= 0)
        {
        switch(proptype)
          {
          case PT_ANY:
          for (i = Lmin; i < Lmax; i++)
            {
            int len = 1;
            if (Feptr >= mb->end_subject)
              {
              SCHECK_PARTIAL();
              break;
              }
            GETCHARLENTEST(fc, Feptr, len);
            if (Lctype == OP_NOTPROP) break;
            Feptr+= len;
            }
          break;

          case PT_LAMP:
          for (i = Lmin; i < Lmax; i++)
            {
            int chartype;
            int len = 1;
            if (Feptr >= mb->end_subject)
              {
              SCHECK_PARTIAL();
              break;
              }
            GETCHARLENTEST(fc, Feptr, len);
            chartype = UCD_CHARTYPE(fc);
            if ((chartype == ucp_Lu ||
                 chartype == ucp_Ll ||
                 chartype == ucp_Lt) == (Lctype == OP_NOTPROP))
              break;
            Feptr+= len;
            }
          break;

          case PT_GC:
          for (i = Lmin; i < Lmax; i++)
            {
            int len = 1;
            if (Feptr >= mb->end_subject)
              {
              SCHECK_PARTIAL();
              break;
              }
            GETCHARLENTEST(fc, Feptr, len);
            if ((UCD_CATEGORY(fc) == Lpropvalue) == (Lctype == OP_NOTPROP))
              break;
            Feptr+= len;
            }
          break;

          case PT_PC:
          for (i = Lmin; i < Lmax; i++)
            {
            int len = 1;
            if (Feptr >= mb->end_subject)
              {
              SCHECK_PARTIAL();
              break;
              }
            GETCHARLENTEST(fc, Feptr, len);
            if ((UCD_CHARTYPE(fc) == Lpropvalue) == (Lctype == OP_NOTPROP))
              break;
            Feptr+= len;
            }
          break;

          case PT_SC:
          for (i = Lmin; i < Lmax; i++)
            {
            int len = 1;
            if (Feptr >= mb->end_subject)
              {
              SCHECK_PARTIAL();
              break;
              }
            GETCHARLENTEST(fc, Feptr, len);
            if ((UCD_SCRIPT(fc) == Lpropvalue) == (Lctype == OP_NOTPROP))
              break;
            Feptr+= len;
            }
          break;

          case PT_ALNUM:
          for (i = Lmin; i < Lmax; i++)
            {
            int category;
            int len = 1;
            if (Feptr >= mb->end_subject)
              {
              SCHECK_PARTIAL();
              break;
              }
            GETCHARLENTEST(fc, Feptr, len);
            category = UCD_CATEGORY(fc);
            if ((category == ucp_L || category == ucp_N) ==
                (Lctype == OP_NOTPROP))
              break;
            Feptr+= len;
            }
          break;

          /* Perl space used to exclude VT, but from Perl 5.18 it is included,
          which means that Perl space and POSIX space are now identical. PCRE
          was changed at release 8.34. */

          case PT_SPACE:    /* Perl space */
          case PT_PXSPACE:  /* POSIX space */
          for (i = Lmin; i < Lmax; i++)
            {
            int len = 1;
            if (Feptr >= mb->end_subject)
              {
              SCHECK_PARTIAL();
              break;
              }
            GETCHARLENTEST(fc, Feptr, len);
            switch(fc)
              {
              HSPACE_CASES:
              VSPACE_CASES:
              if (Lctype == OP_NOTPROP) goto ENDLOOP99;  /* Break the loop */
              break;

              default:
              if ((UCD_CATEGORY(fc) == ucp_Z) == (Lctype == OP_NOTPROP))
                goto ENDLOOP99;   /* Break the loop */
              break;
              }
            Feptr+= len;
            }
          ENDLOOP99:
          break;

          case PT_WORD:
          for (i = Lmin; i < Lmax; i++)
            {
            int category;
            int len = 1;
            if (Feptr >= mb->end_subject)
              {
              SCHECK_PARTIAL();
              break;
              }
            GETCHARLENTEST(fc, Feptr, len);
            category = UCD_CATEGORY(fc);
            if ((category == ucp_L || category == ucp_N ||
                 fc == CHAR_UNDERSCORE) == (Lctype == OP_NOTPROP))
              break;
            Feptr+= len;
            }
          break;

          case PT_CLIST:
          for (i = Lmin; i < Lmax; i++)
            {
            const uint32_t *cp;
            int len = 1;
            if (Feptr >= mb->end_subject)
              {
              SCHECK_PARTIAL();
              break;
              }
            GETCHARLENTEST(fc, Feptr, len);
            cp = PRIV(ucd_caseless_sets) + Lpropvalue;
            for (;;)
              {
              if (fc < *cp)
                { if (Lctype == OP_NOTPROP) break; else goto GOT_MAX; }
              if (fc == *cp++)
                { if (Lctype == OP_NOTPROP) goto GOT_MAX; else break; }
              }
            Feptr += len;
            }
          GOT_MAX:
          break;

          case PT_UCNC:
          for (i = Lmin; i < Lmax; i++)
            {
            int len = 1;
            if (Feptr >= mb->end_subject)
              {
              SCHECK_PARTIAL();
              break;
              }
            GETCHARLENTEST(fc, Feptr, len);
            if ((fc == CHAR_DOLLAR_SIGN || fc == CHAR_COMMERCIAL_AT ||
                 fc == CHAR_GRAVE_ACCENT || (fc >= 0xa0 && fc <= 0xd7ff) ||
                 fc >= 0xe000) == (Lctype == OP_NOTPROP))
              break;
            Feptr += len;
            }
          break;

          default:
          return PCRE2_ERROR_INTERNAL;
          }

        /* Feptr is now past the end of the maximum run */

        if (reptype == REPTYPE_POS) continue;    /* No backtracking */

        /* After \C in UTF mode, Lstart_eptr might be in the middle of a
        Unicode character. Use <= Lstart_eptr to ensure backtracking doesn't
        go too far. */

        for(;;)
          {
          if (Feptr <= Lstart_eptr) break;
          RMATCH(Fecode, RM222);
          if (rrc != MATCH_NOMATCH) RRETURN(rrc);
          Feptr--;
          if (utf) BACKCHAR(Feptr);
          }
        }

      /* Match extended Unicode grapheme clusters. We will get here only if the
      support is in the binary; otherwise a compile-time error occurs. */

      else if (Lctype == OP_EXTUNI)
        {
        for (i = Lmin; i < Lmax; i++)
          {
          if (Feptr >= mb->end_subject)
            {
            SCHECK_PARTIAL();
            break;
            }
          else
            {
            GETCHARINCTEST(fc, Feptr);
            Feptr = PRIV(extuni)(fc, Feptr, mb->start_subject, mb->end_subject,
              utf, NULL);
            }
          CHECK_PARTIAL();
          }

        /* Feptr is now past the end of the maximum run */

        if (reptype == REPTYPE_POS) continue;    /* No backtracking */

        /* We use <= Lstart_eptr rather than == Lstart_eptr to detect the start
        of the run while backtracking because the use of \C in UTF mode can
        cause BACKCHAR to move back past Lstart_eptr. This is just palliative;
        the use of \C in UTF mode is fraught with danger. */

        for(;;)
          {
          int lgb, rgb;
          PCRE2_SPTR fptr;

          if (Feptr <= Lstart_eptr) break;   /* At start of char run */
          RMATCH(Fecode, RM220);
          if (rrc != MATCH_NOMATCH) RRETURN(rrc);

          /* Backtracking over an extended grapheme cluster involves inspecting
          the previous two characters (if present) to see if a break is
          permitted between them. */

          Feptr--;
          if (!utf) fc = *Feptr; else
            {
            BACKCHAR(Feptr);
            GETCHAR(fc, Feptr);
            }
          rgb = UCD_GRAPHBREAK(fc);

          for (;;)
            {
            if (Feptr <= Lstart_eptr) break;   /* At start of char run */
            fptr = Feptr - 1;
            if (!utf) fc = *fptr; else
              {
              BACKCHAR(fptr);
              GETCHAR(fc, fptr);
              }
            lgb = UCD_GRAPHBREAK(fc);
            if ((PRIV(ucp_gbtable)[lgb] & (1u << rgb)) == 0) break;
            Feptr = fptr;
            rgb = lgb;
            }
          }
        }

      else
#endif   /* SUPPORT_UNICODE */

#ifdef SUPPORT_UNICODE
      if (utf)
        {
        switch(Lctype)
          {
          case OP_ANY:
          for (i = Lmin; i < Lmax; i++)
            {
            if (Feptr >= mb->end_subject)
              {
              SCHECK_PARTIAL();
              break;
              }
            if (IS_NEWLINE(Feptr)) break;
            if (mb->partial != 0 &&    /* Take care with CRLF partial */
                Feptr + 1 >= mb->end_subject &&
                NLBLOCK->nltype == NLTYPE_FIXED &&
                NLBLOCK->nllen == 2 &&
                UCHAR21(Feptr) == NLBLOCK->nl[0])
              {
              mb->hitend = TRUE;
              if (mb->partial > 1) return PCRE2_ERROR_PARTIAL;
              }
            Feptr++;
            ACROSSCHAR(Feptr < mb->end_subject, Feptr, Feptr++);
            }
          break;

          case OP_ALLANY:
          if (Lmax < UINT32_MAX)
            {
            for (i = Lmin; i < Lmax; i++)
              {
              if (Feptr >= mb->end_subject)
                {
                SCHECK_PARTIAL();
                break;
                }
              Feptr++;
              ACROSSCHAR(Feptr < mb->end_subject, Feptr, Feptr++);
              }
            }
          else
            {
            Feptr = mb->end_subject;   /* Unlimited UTF-8 repeat */
            SCHECK_PARTIAL();
            }
          break;

          /* The "byte" (i.e. "code unit") case is the same as non-UTF */

          case OP_ANYBYTE:
          fc = Lmax - Lmin;
          if (fc > (uint32_t)(mb->end_subject - Feptr))
            {
            Feptr = mb->end_subject;
            SCHECK_PARTIAL();
            }
          else Feptr += fc;
          break;

          case OP_ANYNL:
          for (i = Lmin; i < Lmax; i++)
            {
            int len = 1;
            if (Feptr >= mb->end_subject)
              {
              SCHECK_PARTIAL();
              break;
              }
            GETCHARLEN(fc, Feptr, len);
            if (fc == CHAR_CR)
              {
              if (++Feptr >= mb->end_subject) break;
              if (UCHAR21(Feptr) == CHAR_LF) Feptr++;
              }
            else
              {
              if (fc != CHAR_LF &&
                  (mb->bsr_convention == PCRE2_BSR_ANYCRLF ||
                   (fc != CHAR_VT && fc != CHAR_FF && fc != CHAR_NEL
#ifndef EBCDIC
                    && fc != 0x2028 && fc != 0x2029
#endif  /* Not EBCDIC */
                    )))
                break;
              Feptr += len;
              }
            }
          break;

          case OP_NOT_HSPACE:
          case OP_HSPACE:
          for (i = Lmin; i < Lmax; i++)
            {
            BOOL gotspace;
            int len = 1;
            if (Feptr >= mb->end_subject)
              {
              SCHECK_PARTIAL();
              break;
              }
            GETCHARLEN(fc, Feptr, len);
            switch(fc)
              {
              HSPACE_CASES: gotspace = TRUE; break;
              default: gotspace = FALSE; break;
              }
            if (gotspace == (Lctype == OP_NOT_HSPACE)) break;
            Feptr += len;
            }
          break;

          case OP_NOT_VSPACE:
          case OP_VSPACE:
          for (i = Lmin; i < Lmax; i++)
            {
            BOOL gotspace;
            int len = 1;
            if (Feptr >= mb->end_subject)
              {
              SCHECK_PARTIAL();
              break;
              }
            GETCHARLEN(fc, Feptr, len);
            switch(fc)
              {
              VSPACE_CASES: gotspace = TRUE; break;
              default: gotspace = FALSE; break;
              }
            if (gotspace == (Lctype == OP_NOT_VSPACE)) break;
            Feptr += len;
            }
          break;

          case OP_NOT_DIGIT:
          for (i = Lmin; i < Lmax; i++)
            {
            int len = 1;
            if (Feptr >= mb->end_subject)
              {
              SCHECK_PARTIAL();
              break;
              }
            GETCHARLEN(fc, Feptr, len);
            if (fc < 256 && (mb->ctypes[fc] & ctype_digit) != 0) break;
            Feptr+= len;
            }
          break;

          case OP_DIGIT:
          for (i = Lmin; i < Lmax; i++)
            {
            int len = 1;
            if (Feptr >= mb->end_subject)
              {
              SCHECK_PARTIAL();
              break;
              }
            GETCHARLEN(fc, Feptr, len);
            if (fc >= 256 ||(mb->ctypes[fc] & ctype_digit) == 0) break;
            Feptr+= len;
            }
          break;

          case OP_NOT_WHITESPACE:
          for (i = Lmin; i < Lmax; i++)
            {
            int len = 1;
            if (Feptr >= mb->end_subject)
              {
              SCHECK_PARTIAL();
              break;
              }
            GETCHARLEN(fc, Feptr, len);
            if (fc < 256 && (mb->ctypes[fc] & ctype_space) != 0) break;
            Feptr+= len;
            }
          break;

          case OP_WHITESPACE:
          for (i = Lmin; i < Lmax; i++)
            {
            int len = 1;
            if (Feptr >= mb->end_subject)
              {
              SCHECK_PARTIAL();
              break;
              }
            GETCHARLEN(fc, Feptr, len);
            if (fc >= 256 ||(mb->ctypes[fc] & ctype_space) == 0) break;
            Feptr+= len;
            }
          break;

          case OP_NOT_WORDCHAR:
          for (i = Lmin; i < Lmax; i++)
            {
            int len = 1;
            if (Feptr >= mb->end_subject)
              {
              SCHECK_PARTIAL();
              break;
              }
            GETCHARLEN(fc, Feptr, len);
            if (fc < 256 && (mb->ctypes[fc] & ctype_word) != 0) break;
            Feptr+= len;
            }
          break;

          case OP_WORDCHAR:
          for (i = Lmin; i < Lmax; i++)
            {
            int len = 1;
            if (Feptr >= mb->end_subject)
              {
              SCHECK_PARTIAL();
              break;
              }
            GETCHARLEN(fc, Feptr, len);
            if (fc >= 256 || (mb->ctypes[fc] & ctype_word) == 0) break;
            Feptr+= len;
            }
          break;

          default:
          return PCRE2_ERROR_INTERNAL;
          }

        if (reptype == REPTYPE_POS) continue;    /* No backtracking */

        /* After \C in UTF mode, Lstart_eptr might be in the middle of a
        Unicode character. Use <= Lstart_eptr to ensure backtracking doesn't go
        too far. */

        for(;;)
          {
          if (Feptr <= Lstart_eptr) break;
          RMATCH(Fecode, RM221);
          if (rrc != MATCH_NOMATCH) RRETURN(rrc);
          Feptr--;
          BACKCHAR(Feptr);
          if (Lctype == OP_ANYNL && Feptr > Lstart_eptr &&
              UCHAR21(Feptr) == CHAR_NL && UCHAR21(Feptr - 1) == CHAR_CR)
            Feptr--;
          }
        }
      else
#endif  /* SUPPORT_UNICODE */

      /* Not UTF mode */
        {
        switch(Lctype)
          {
          case OP_ANY:
          for (i = Lmin; i < Lmax; i++)
            {
            if (Feptr >= mb->end_subject)
              {
              SCHECK_PARTIAL();
              break;
              }
            if (IS_NEWLINE(Feptr)) break;
            if (mb->partial != 0 &&    /* Take care with CRLF partial */
                Feptr + 1 >= mb->end_subject &&
                NLBLOCK->nltype == NLTYPE_FIXED &&
                NLBLOCK->nllen == 2 &&
                *Feptr == NLBLOCK->nl[0])
              {
              mb->hitend = TRUE;
              if (mb->partial > 1) return PCRE2_ERROR_PARTIAL;
              }
            Feptr++;
            }
          break;

          case OP_ALLANY:
          case OP_ANYBYTE:
          fc = Lmax - Lmin;
          if (fc > (uint32_t)(mb->end_subject - Feptr))
            {
            Feptr = mb->end_subject;
            SCHECK_PARTIAL();
            }
          else Feptr += fc;
          break;

          case OP_ANYNL:
          for (i = Lmin; i < Lmax; i++)
            {
            if (Feptr >= mb->end_subject)
              {
              SCHECK_PARTIAL();
              break;
              }
            fc = *Feptr;
            if (fc == CHAR_CR)
              {
              if (++Feptr >= mb->end_subject) break;
              if (*Feptr == CHAR_LF) Feptr++;
              }
            else
              {
              if (fc != CHAR_LF && (mb->bsr_convention == PCRE2_BSR_ANYCRLF ||
                 (fc != CHAR_VT && fc != CHAR_FF && fc != CHAR_NEL
#if PCRE2_CODE_UNIT_WIDTH != 8
                 && fc != 0x2028 && fc != 0x2029
#endif
                 ))) break;
              Feptr++;
              }
            }
          break;

          case OP_NOT_HSPACE:
          for (i = Lmin; i < Lmax; i++)
            {
            if (Feptr >= mb->end_subject)
              {
              SCHECK_PARTIAL();
              break;
              }
            switch(*Feptr)
              {
              default: Feptr++; break;
              HSPACE_BYTE_CASES:
#if PCRE2_CODE_UNIT_WIDTH != 8
              HSPACE_MULTIBYTE_CASES:
#endif
              goto ENDLOOP00;
              }
            }
          ENDLOOP00:
          break;

          case OP_HSPACE:
          for (i = Lmin; i < Lmax; i++)
            {
            if (Feptr >= mb->end_subject)
              {
              SCHECK_PARTIAL();
              break;
              }
            switch(*Feptr)
              {
              default: goto ENDLOOP01;
              HSPACE_BYTE_CASES:
#if PCRE2_CODE_UNIT_WIDTH != 8
              HSPACE_MULTIBYTE_CASES:
#endif
              Feptr++; break;
              }
            }
          ENDLOOP01:
          break;

          case OP_NOT_VSPACE:
          for (i = Lmin; i < Lmax; i++)
            {
            if (Feptr >= mb->end_subject)
              {
              SCHECK_PARTIAL();
              break;
              }
            switch(*Feptr)
              {
              default: Feptr++; break;
              VSPACE_BYTE_CASES:
#if PCRE2_CODE_UNIT_WIDTH != 8
              VSPACE_MULTIBYTE_CASES:
#endif
              goto ENDLOOP02;
              }
            }
          ENDLOOP02:
          break;

          case OP_VSPACE:
          for (i = Lmin; i < Lmax; i++)
            {
            if (Feptr >= mb->end_subject)
              {
              SCHECK_PARTIAL();
              break;
              }
            switch(*Feptr)
              {
              default: goto ENDLOOP03;
              VSPACE_BYTE_CASES:
#if PCRE2_CODE_UNIT_WIDTH != 8
              VSPACE_MULTIBYTE_CASES:
#endif
              Feptr++; break;
              }
            }
          ENDLOOP03:
          break;

          case OP_NOT_DIGIT:
          for (i = Lmin; i < Lmax; i++)
            {
            if (Feptr >= mb->end_subject)
              {
              SCHECK_PARTIAL();
              break;
              }
            if (MAX_255(*Feptr) && (mb->ctypes[*Feptr] & ctype_digit) != 0)
              break;
            Feptr++;
            }
          break;

          case OP_DIGIT:
          for (i = Lmin; i < Lmax; i++)
            {
            if (Feptr >= mb->end_subject)
              {
              SCHECK_PARTIAL();
              break;
              }
            if (!MAX_255(*Feptr) || (mb->ctypes[*Feptr] & ctype_digit) == 0)
              break;
            Feptr++;
            }
          break;

          case OP_NOT_WHITESPACE:
          for (i = Lmin; i < Lmax; i++)
            {
            if (Feptr >= mb->end_subject)
              {
              SCHECK_PARTIAL();
              break;
              }
            if (MAX_255(*Feptr) && (mb->ctypes[*Feptr] & ctype_space) != 0)
              break;
            Feptr++;
            }
          break;

          case OP_WHITESPACE:
          for (i = Lmin; i < Lmax; i++)
            {
            if (Feptr >= mb->end_subject)
              {
              SCHECK_PARTIAL();
              break;
              }
            if (!MAX_255(*Feptr) || (mb->ctypes[*Feptr] & ctype_space) == 0)
              break;
            Feptr++;
            }
          break;

          case OP_NOT_WORDCHAR:
          for (i = Lmin; i < Lmax; i++)
            {
            if (Feptr >= mb->end_subject)
              {
              SCHECK_PARTIAL();
              break;
              }
            if (MAX_255(*Feptr) && (mb->ctypes[*Feptr] & ctype_word) != 0)
              break;
            Feptr++;
            }
          break;

          case OP_WORDCHAR:
          for (i = Lmin; i < Lmax; i++)
            {
            if (Feptr >= mb->end_subject)
              {
              SCHECK_PARTIAL();
              break;
              }
            if (!MAX_255(*Feptr) || (mb->ctypes[*Feptr] & ctype_word) == 0)
              break;
            Feptr++;
            }
          break;

          default:
          return PCRE2_ERROR_INTERNAL;
          }

        if (reptype == REPTYPE_POS) continue;    /* No backtracking */

        for (;;)
          {
          if (Feptr == Lstart_eptr) break;
          RMATCH(Fecode, RM34);
          if (rrc != MATCH_NOMATCH) RRETURN(rrc);
          Feptr--;
          if (Lctype == OP_ANYNL && Feptr > Lstart_eptr && *Feptr == CHAR_LF &&
              Feptr[-1] == CHAR_CR) Feptr--;
          }
        }
      }
    break;  /* End of repeat character type processing */

#undef Lstart_eptr
#undef Lmin
#undef Lmax
#undef Lctype
#undef Lpropvalue


    /* ===================================================================== */
    /* Match a back reference, possibly repeatedly. Look past the end of the
    item to see if there is repeat information following. The OP_REF and
    OP_REFI opcodes are used for a reference to a numbered group or to a
    non-duplicated named group. For a duplicated named group, OP_DNREF and
    OP_DNREFI are used. In this case we must scan the list of groups to which
    the name refers, and use the first one that is set. */

#define Lmin      F->temp_32[0]
#define Lmax      F->temp_32[1]
#define Lcaseless F->temp_32[2]
#define Lstart    F->temp_sptr[0]
#define Loffset   F->temp_size

    case OP_DNREF:
    case OP_DNREFI:
    Lcaseless = (Fop == OP_DNREFI);
      {
      int count = GET2(Fecode, 1+IMM2_SIZE);
      PCRE2_SPTR slot = mb->name_table + GET2(Fecode, 1) * mb->name_entry_size;
      Fecode += 1 + 2*IMM2_SIZE;

      while (count-- > 0)
        {
        Loffset = (GET2(slot, 0) << 1) - 2;
        if (Loffset < Foffset_top && Fovector[Loffset] != PCRE2_UNSET) break;
        slot += mb->name_entry_size;
        }
      }
    goto REF_REPEAT;

    case OP_REF:
    case OP_REFI:
    Lcaseless = (Fop == OP_REFI);
    Loffset = (GET2(Fecode, 1) << 1) - 2;
    Fecode += 1 + IMM2_SIZE;

    /* Set up for repetition, or handle the non-repeated case. The maximum and
    minimum must be in the heap frame, but as they are short-term values, we
    use temporary fields. */

    REF_REPEAT:
    switch (*Fecode)
      {
      case OP_CRSTAR:
      case OP_CRMINSTAR:
      case OP_CRPLUS:
      case OP_CRMINPLUS:
      case OP_CRQUERY:
      case OP_CRMINQUERY:
      fc = *Fecode++ - OP_CRSTAR;
      Lmin = rep_min[fc];
      Lmax = rep_max[fc];
      reptype = rep_typ[fc];
      break;

      case OP_CRRANGE:
      case OP_CRMINRANGE:
      Lmin = GET2(Fecode, 1);
      Lmax = GET2(Fecode, 1 + IMM2_SIZE);
      reptype = rep_typ[*Fecode - OP_CRSTAR];
      if (Lmax == 0) Lmax = UINT32_MAX;  /* Max 0 => infinity */
      Fecode += 1 + 2 * IMM2_SIZE;
      break;

      default:                  /* No repeat follows */
        {
        rrc = match_ref(Loffset, Lcaseless, F, mb, &length);
        if (rrc != 0)
          {
          if (rrc > 0) Feptr = mb->end_subject;   /* Partial match */
          CHECK_PARTIAL();
          RRETURN(MATCH_NOMATCH);
          }
        }
      Feptr += length;
      continue;              /* With the main loop */
      }

    /* Handle repeated back references. If a set group has length zero, just
    continue with the main loop, because it matches however many times. For an
    unset reference, if the minimum is zero, we can also just continue. We can
    also continue if PCRE2_MATCH_UNSET_BACKREF is set, because this makes unset
    group behave as a zero-length group. For any other unset cases, carrying
    on will result in NOMATCH. */

    if (Loffset < Foffset_top && Fovector[Loffset] != PCRE2_UNSET)
      {
      if (Fovector[Loffset] == Fovector[Loffset + 1]) continue;
      }
    else  /* Group is not set */
      {
      if (Lmin == 0 || (mb->poptions & PCRE2_MATCH_UNSET_BACKREF) != 0)
        continue;
      }

    /* First, ensure the minimum number of matches are present. */

    for (i = 1; i <= Lmin; i++)
      {
      PCRE2_SIZE slength;
      rrc = match_ref(Loffset, Lcaseless, F, mb, &slength);
      if (rrc != 0)
        {
        if (rrc > 0) Feptr = mb->end_subject;   /* Partial match */
        CHECK_PARTIAL();
        RRETURN(MATCH_NOMATCH);
        }
      Feptr += slength;
      }

    /* If min = max, we are done. They are not both allowed to be zero. */

    if (Lmin == Lmax) continue;

    /* If minimizing, keep trying and advancing the pointer. */

    if (reptype == REPTYPE_MIN)
      {
      for (;;)
        {
        PCRE2_SIZE slength;
        RMATCH(Fecode, RM20);
        if (rrc != MATCH_NOMATCH) RRETURN(rrc);
        if (Lmin++ >= Lmax) RRETURN(MATCH_NOMATCH);
        rrc = match_ref(Loffset, Lcaseless, F, mb, &slength);
        if (rrc != 0)
          {
          if (rrc > 0) Feptr = mb->end_subject;   /* Partial match */
          CHECK_PARTIAL();
          RRETURN(MATCH_NOMATCH);
          }
        Feptr += slength;
        }
      /* Control never gets here */
      }

    /* If maximizing, find the longest string and work backwards, as long as
    the matched lengths for each iteration are the same. */

    else
      {
      BOOL samelengths = TRUE;
      Lstart = Feptr;     /* Starting position */
      Flength = Fovector[Loffset+1] - Fovector[Loffset];

      for (i = Lmin; i < Lmax; i++)
        {
        PCRE2_SIZE slength;
        rrc = match_ref(Loffset, Lcaseless, F, mb, &slength);
        if (rrc != 0)
          {
          /* Can't use CHECK_PARTIAL because we don't want to update Feptr in
          the soft partial matching case. */

          if (rrc > 0 && mb->partial != 0 &&
              mb->end_subject > mb->start_used_ptr)
            {
            mb->hitend = TRUE;
            if (mb->partial > 1) return PCRE2_ERROR_PARTIAL;
            }
          break;
          }

        if (slength != Flength) samelengths = FALSE;
        Feptr += slength;
        }

      /* If the length matched for each repetition is the same as the length of
      the captured group, we can easily work backwards. This is the normal
      case. However, in caseless UTF-8 mode there are pairs of case-equivalent
      characters whose lengths (in terms of code units) differ. However, this
      is very rare, so we handle it by re-matching fewer and fewer times. */

      if (samelengths)
        {
        while (Feptr >= Lstart)
          {
          RMATCH(Fecode, RM21);
          if (rrc != MATCH_NOMATCH) RRETURN(rrc);
          Feptr -= Flength;
          }
        }

      /* The rare case of non-matching lengths. Re-scan the repetition for each
      iteration. We know that match_ref() will succeed every time. */

      else
        {
        Lmax = i;
        for (;;)
          {
          RMATCH(Fecode, RM22);
          if (rrc != MATCH_NOMATCH) RRETURN(rrc);
          if (Feptr == Lstart) break; /* Failed after minimal repetition */
          Feptr = Lstart;
          Lmax--;
          for (i = Lmin; i < Lmax; i++)
            {
            PCRE2_SIZE slength;
            (void)match_ref(Loffset, Lcaseless, F, mb, &slength);
            Feptr += slength;
            }
          }
        }

      RRETURN(MATCH_NOMATCH);
      }
    /* Control never gets here */

#undef Lcaseless
#undef Lmin
#undef Lmax
#undef Lstart
#undef Loffset



/* ========================================================================= */
/*           Opcodes for the start of various parenthesized items            */
/* ========================================================================= */

    /* In all cases, if the result of RMATCH() is MATCH_THEN, check whether the
    (*THEN) is within the current branch by comparing the address of OP_THEN
    that is passed back with the end of the branch. If (*THEN) is within the
    current branch, and the branch is one of two or more alternatives (it
    either starts or ends with OP_ALT), we have reached the limit of THEN's
    action, so convert the return code to NOMATCH, which will cause normal
    backtracking to happen from now on. Otherwise, THEN is passed back to an
    outer alternative. This implements Perl's treatment of parenthesized
    groups, where a group not containing | does not affect the current
    alternative, that is, (X) is NOT the same as (X|(*F)). */


    /* ===================================================================== */
    /* BRAZERO, BRAMINZERO and SKIPZERO occur just before a non-possessive
    bracket group, indicating that it may occur zero times. It may repeat
    infinitely, or not at all - i.e. it could be ()* or ()? or even (){0} in
    the pattern. Brackets with fixed upper repeat limits are compiled as a
    number of copies, with the optional ones preceded by BRAZERO or BRAMINZERO.
    Possessive groups with possible zero repeats are preceded by BRAPOSZERO. */

#define Lnext_ecode F->temp_sptr[0]

    case OP_BRAZERO:
    Lnext_ecode = Fecode + 1;
    RMATCH(Lnext_ecode, RM9);
    if (rrc != MATCH_NOMATCH) RRETURN(rrc);
    do Lnext_ecode += GET(Lnext_ecode, 1); while (*Lnext_ecode == OP_ALT);
    Fecode = Lnext_ecode + 1 + LINK_SIZE;
    break;

    case OP_BRAMINZERO:
    Lnext_ecode = Fecode + 1;
    do Lnext_ecode += GET(Lnext_ecode, 1); while (*Lnext_ecode == OP_ALT);
    RMATCH(Lnext_ecode + 1 + LINK_SIZE, RM10);
    if (rrc != MATCH_NOMATCH) RRETURN(rrc);
    Fecode++;
    break;

#undef Lnext_ecode

    case OP_SKIPZERO:
    Fecode++;
    do Fecode += GET(Fecode,1); while (*Fecode == OP_ALT);
    Fecode += 1 + LINK_SIZE;
    break;


    /* ===================================================================== */
    /* Handle possessive brackets with an unlimited repeat. The end of these
    brackets will always be OP_KETRPOS, which returns MATCH_KETRPOS without
    going further in the pattern. */

#define Lframe_type    F->temp_32[0]
#define Lmatched_once  F->temp_32[1]
#define Lzero_allowed  F->temp_32[2]
#define Lstart_eptr    F->temp_sptr[0]
#define Lstart_group   F->temp_sptr[1]

    case OP_BRAPOSZERO:
    Lzero_allowed = TRUE;                /* Zero repeat is allowed */
    Fecode += 1;
    if (*Fecode == OP_CBRAPOS || *Fecode == OP_SCBRAPOS)
      goto POSSESSIVE_CAPTURE;
    goto POSSESSIVE_NON_CAPTURE;

    case OP_BRAPOS:
    case OP_SBRAPOS:
    Lzero_allowed = FALSE;               /* Zero repeat not allowed */

    POSSESSIVE_NON_CAPTURE:
    Lframe_type = GF_NOCAPTURE;          /* Remembered frame type */
    goto POSSESSIVE_GROUP;

    case OP_CBRAPOS:
    case OP_SCBRAPOS:
    Lzero_allowed = FALSE;               /* Zero repeat not allowed */

    POSSESSIVE_CAPTURE:
    number = GET2(Fecode, 1+LINK_SIZE);
    Lframe_type = GF_CAPTURE | number;   /* Remembered frame type */

    POSSESSIVE_GROUP:
    Lmatched_once = FALSE;               /* Never matched */
    Lstart_group = Fecode;               /* Start of this group */

    for (;;)
      {
      Lstart_eptr = Feptr;               /* Position at group start */
      group_frame_type = Lframe_type;
      RMATCH(Fecode + PRIV(OP_lengths)[*Fecode], RM8);
      if (rrc == MATCH_KETRPOS)
        {
        Lmatched_once = TRUE;            /* Matched at least once */
        if (Feptr == Lstart_eptr)        /* Empty match; skip to end */
          {
          do Fecode += GET(Fecode, 1); while (*Fecode == OP_ALT);
          break;
          }

        Fecode = Lstart_group;
        continue;
        }

      /* See comment above about handling THEN. */

      if (rrc == MATCH_THEN)
        {
        PCRE2_SPTR next_ecode = Fecode + GET(Fecode,1);
        if (mb->verb_ecode_ptr < next_ecode &&
            (*Fecode == OP_ALT || *next_ecode == OP_ALT))
          rrc = MATCH_NOMATCH;
        }

      if (rrc != MATCH_NOMATCH) RRETURN(rrc);
      Fecode += GET(Fecode, 1);
      if (*Fecode != OP_ALT) break;
      }

    /* Success if matched something or zero repeat allowed */

    if (Lmatched_once || Lzero_allowed)
      {
      Fecode += 1 + LINK_SIZE;
      break;
      }

    RRETURN(MATCH_NOMATCH);

#undef Lmatched_once
#undef Lzero_allowed
#undef Lframe_type
#undef Lstart_eptr
#undef Lstart_group


    /* ===================================================================== */
    /* Handle non-capturing brackets that cannot match an empty string. When we
    get to the final alternative within the brackets, as long as there are no
    THEN's in the pattern, we can optimize by not recording a new backtracking
    point. (Ideally we should test for a THEN within this group, but we don't
    have that information.) Don't do this if we are at the very top level,
    however, because that would make handling assertions and once-only brackets
    messier when there is nothing to go back to. */

#define Lframe_type F->temp_32[0]     /* Set for all that use GROUPLOOP */
#define Lnext_branch F->temp_sptr[0]  /* Used only in OP_BRA handling */

    case OP_BRA:
    if (mb->hasthen || Frdepth == 0)
      {
      Lframe_type = 0;
      goto GROUPLOOP;
      }

    for (;;)
      {
      Lnext_branch = Fecode + GET(Fecode, 1);
      if (*Lnext_branch != OP_ALT) break;

      /* This is never the final branch. We do not need to test for MATCH_THEN
      here because this code is not used when there is a THEN in the pattern. */

      RMATCH(Fecode + PRIV(OP_lengths)[*Fecode], RM1);
      if (rrc != MATCH_NOMATCH) RRETURN(rrc);
      Fecode = Lnext_branch;
      }

    /* Hit the start of the final branch. Continue at this level. */

    Fecode += PRIV(OP_lengths)[*Fecode];
    break;

#undef Lnext_branch


    /* ===================================================================== */
    /* Handle a capturing bracket, other than those that are possessive with an
    unlimited repeat. */

    case OP_CBRA:
    case OP_SCBRA:
    Lframe_type = GF_CAPTURE | GET2(Fecode, 1+LINK_SIZE);
    goto GROUPLOOP;


    /* ===================================================================== */
    /* Atomic groups and non-capturing brackets that can match an empty string
    must record a backtracking point and also set up a chained frame. */

    case OP_ONCE:
    case OP_SCRIPT_RUN:
    case OP_SBRA:
    Lframe_type = GF_NOCAPTURE | Fop;

    GROUPLOOP:
    for (;;)
      {
      group_frame_type = Lframe_type;
      RMATCH(Fecode + PRIV(OP_lengths)[*Fecode], RM2);
      if (rrc == MATCH_THEN)
        {
        PCRE2_SPTR next_ecode = Fecode + GET(Fecode,1);
        if (mb->verb_ecode_ptr < next_ecode &&
            (*Fecode == OP_ALT || *next_ecode == OP_ALT))
          rrc = MATCH_NOMATCH;
        }
      if (rrc != MATCH_NOMATCH) RRETURN(rrc);
      Fecode += GET(Fecode, 1);
      if (*Fecode != OP_ALT) RRETURN(MATCH_NOMATCH);
      }
    /* Control never reaches here. */

#undef Lframe_type


    /* ===================================================================== */
    /* Recursion either matches the current regex, or some subexpression. The
    offset data is the offset to the starting bracket from the start of the
    whole pattern. (This is so that it works from duplicated subpatterns.) */

#define Lframe_type F->temp_32[0]
#define Lstart_branch F->temp_sptr[0]

    case OP_RECURSE:
    bracode = mb->start_code + GET(Fecode, 1);
    number = (bracode == mb->start_code)? 0 : GET2(bracode, 1 + LINK_SIZE);

    /* If we are already in a recursion, check for repeating the same one
    without advancing the subject pointer. This should catch convoluted mutual
    recursions. (Some simple cases are caught at compile time.) */

    if (Fcurrent_recurse != RECURSE_UNSET)
      {
      offset = Flast_group_offset;
      while (offset != PCRE2_UNSET)
        {
        N = (heapframe *)((char *)mb->match_frames + offset);
        P = (heapframe *)((char *)N - frame_size);
        if (N->group_frame_type == (GF_RECURSE | number))
          {
          if (Feptr == P->eptr) return PCRE2_ERROR_RECURSELOOP;
          break;
          }
        offset = P->last_group_offset;
        }
      }

    /* Now run the recursion, branch by branch. */

    Lstart_branch = bracode;
    Lframe_type = GF_RECURSE | number;

    for (;;)
      {
      PCRE2_SPTR next_ecode;

      group_frame_type = Lframe_type;
      RMATCH(Lstart_branch + PRIV(OP_lengths)[*Lstart_branch], RM11);
      next_ecode = Lstart_branch + GET(Lstart_branch,1);

      /* Handle backtracking verbs, which are defined in a range that can
      easily be tested for. PCRE does not allow THEN, SKIP, PRUNE or COMMIT to
      escape beyond a recursion; they cause a NOMATCH for the entire recursion.

      When one of these verbs triggers, the current recursion group number is
      recorded. If it matches the recursion we are processing, the verb
      happened within the recursion and we must deal with it. Otherwise it must
      have happened after the recursion completed, and so has to be passed
      back. See comment above about handling THEN. */

      if (rrc >= MATCH_BACKTRACK_MIN && rrc <= MATCH_BACKTRACK_MAX &&
          mb->verb_current_recurse == (Lframe_type ^ GF_RECURSE))
        {
        if (rrc == MATCH_THEN && mb->verb_ecode_ptr < next_ecode &&
            (*Lstart_branch == OP_ALT || *next_ecode == OP_ALT))
          rrc = MATCH_NOMATCH;
        else RRETURN(MATCH_NOMATCH);
        }

      /* Note that carrying on after (*ACCEPT) in a recursion is handled in the
      OP_ACCEPT code. Nothing needs to be done here. */

      if (rrc != MATCH_NOMATCH) RRETURN(rrc);
      Lstart_branch = next_ecode;
      if (*Lstart_branch != OP_ALT) RRETURN(MATCH_NOMATCH);
      }
    /* Control never reaches here. */

#undef Lframe_type
#undef Lstart_branch


    /* ===================================================================== */
    /* Positive assertions are like other groups except that PCRE doesn't allow
    the effect of (*THEN) to escape beyond an assertion; it is therefore
    treated as NOMATCH. (*ACCEPT) is treated as successful assertion, with its
    captures and mark retained. Any other return is an error. */

#define Lframe_type  F->temp_32[0]

    case OP_ASSERT:
    case OP_ASSERTBACK:
    case OP_ASSERT_NA:
    case OP_ASSERTBACK_NA:
    Lframe_type = GF_NOCAPTURE | Fop;
    for (;;)
      {
      group_frame_type = Lframe_type;
      RMATCH(Fecode + PRIV(OP_lengths)[*Fecode], RM3);
      if (rrc == MATCH_ACCEPT)
        {
        memcpy(Fovector,
              (char *)assert_accept_frame + offsetof(heapframe, ovector),
              assert_accept_frame->offset_top * sizeof(PCRE2_SIZE));
        Foffset_top = assert_accept_frame->offset_top;
        Fmark = assert_accept_frame->mark;
        break;
        }
      if (rrc != MATCH_NOMATCH && rrc != MATCH_THEN) RRETURN(rrc);
      Fecode += GET(Fecode, 1);
      if (*Fecode != OP_ALT) RRETURN(MATCH_NOMATCH);
      }

    do Fecode += GET(Fecode, 1); while (*Fecode == OP_ALT);
    Fecode += 1 + LINK_SIZE;
    break;

#undef Lframe_type


    /* ===================================================================== */
    /* Handle negative assertions. Loop for each non-matching branch as for
    positive assertions. */

#define Lframe_type  F->temp_32[0]

    case OP_ASSERT_NOT:
    case OP_ASSERTBACK_NOT:
    Lframe_type  = GF_NOCAPTURE | Fop;

    for (;;)
      {
      group_frame_type = Lframe_type;
      RMATCH(Fecode + PRIV(OP_lengths)[*Fecode], RM4);
      switch(rrc)
        {
        case MATCH_ACCEPT:   /* Assertion matched, therefore it fails. */
        case MATCH_MATCH:
        RRETURN (MATCH_NOMATCH);

        case MATCH_NOMATCH:  /* Branch failed, try next if present. */
        case MATCH_THEN:
        Fecode += GET(Fecode, 1);
        if (*Fecode != OP_ALT) goto ASSERT_NOT_FAILED;
        break;

        case MATCH_COMMIT:   /* Assertion forced to fail, therefore continue. */
        case MATCH_SKIP:
        case MATCH_PRUNE:
        do Fecode += GET(Fecode, 1); while (*Fecode == OP_ALT);
        goto ASSERT_NOT_FAILED;

        default:             /* Pass back any other return */
        RRETURN(rrc);
        }
      }

    /* None of the branches have matched or there was a backtrack to (*COMMIT),
    (*SKIP), (*PRUNE), or (*THEN) in the last branch. This is success for a
    negative assertion, so carry on. */

    ASSERT_NOT_FAILED:
    Fecode += 1 + LINK_SIZE;
    break;

#undef Lframe_type


    /* ===================================================================== */
    /* The callout item calls an external function, if one is provided, passing
    details of the match so far. This is mainly for debugging, though the
    function is able to force a failure. */

    case OP_CALLOUT:
    case OP_CALLOUT_STR:
    rrc = do_callout(F, mb, &length);
    if (rrc > 0) RRETURN(MATCH_NOMATCH);
    if (rrc < 0) RRETURN(rrc);
    Fecode += length;
    break;


    /* ===================================================================== */
    /* Conditional group: compilation checked that there are no more than two
    branches. If the condition is false, skipping the first branch takes us
    past the end of the item if there is only one branch, but that's exactly
    what we want. */

    case OP_COND:
    case OP_SCOND:

    /* The variable Flength will be added to Fecode when the condition is
    false, to get to the second branch. Setting it to the offset to the ALT or
    KET, then incrementing Fecode achieves this effect. However, if the second
    branch is non-existent, we must point to the KET so that the end of the
    group is correctly processed. We now have Fecode pointing to the condition
    or callout. */

    Flength = GET(Fecode, 1);    /* Offset to the second branch */
    if (Fecode[Flength] != OP_ALT) Flength -= 1 + LINK_SIZE;
    Fecode += 1 + LINK_SIZE;     /* From this opcode */

    /* Because of the way auto-callout works during compile, a callout item is
    inserted between OP_COND and an assertion condition. Such a callout can
    also be inserted manually. */

    if (*Fecode == OP_CALLOUT || *Fecode == OP_CALLOUT_STR)
      {
      rrc = do_callout(F, mb, &length);
      if (rrc > 0) RRETURN(MATCH_NOMATCH);
      if (rrc < 0) RRETURN(rrc);

      /* Advance Fecode past the callout, so it now points to the condition. We
      must adjust Flength so that the value of Fecode+Flength is unchanged. */

      Fecode += length;
      Flength -= length;
      }

    /* Test the various possible conditions */

    condition = FALSE;
    switch(*Fecode)
      {
      case OP_RREF:                  /* Group recursion test */
      if (Fcurrent_recurse != RECURSE_UNSET)
        {
        number = GET2(Fecode, 1);
        condition = (number == RREF_ANY || number == Fcurrent_recurse);
        }
      break;

      case OP_DNRREF:       /* Duplicate named group recursion test */
      if (Fcurrent_recurse != RECURSE_UNSET)
        {
        int count = GET2(Fecode, 1 + IMM2_SIZE);
        PCRE2_SPTR slot = mb->name_table + GET2(Fecode, 1) * mb->name_entry_size;
        while (count-- > 0)
          {
          number = GET2(slot, 0);
          condition = number == Fcurrent_recurse;
          if (condition) break;
          slot += mb->name_entry_size;
          }
        }
      break;

      case OP_CREF:                         /* Numbered group used test */
      offset = (GET2(Fecode, 1) << 1) - 2;  /* Doubled ref number */
      condition = offset < Foffset_top && Fovector[offset] != PCRE2_UNSET;
      break;

      case OP_DNCREF:      /* Duplicate named group used test */
        {
        int count = GET2(Fecode, 1 + IMM2_SIZE);
        PCRE2_SPTR slot = mb->name_table + GET2(Fecode, 1) * mb->name_entry_size;
        while (count-- > 0)
          {
          offset = (GET2(slot, 0) << 1) - 2;
          condition = offset < Foffset_top && Fovector[offset] != PCRE2_UNSET;
          if (condition) break;
          slot += mb->name_entry_size;
          }
        }
      break;

      case OP_FALSE:
      case OP_FAIL:   /* The assertion (?!) becomes OP_FAIL */
      break;

      case OP_TRUE:
      condition = TRUE;
      break;

      /* The condition is an assertion. Run code similar to the assertion code
      above. */

#define Lpositive      F->temp_32[0]
#define Lstart_branch  F->temp_sptr[0]

      default:
      Lpositive = (*Fecode == OP_ASSERT || *Fecode == OP_ASSERTBACK);
      Lstart_branch = Fecode;

      for (;;)
        {
        group_frame_type = GF_CONDASSERT | *Fecode;
        RMATCH(Lstart_branch + PRIV(OP_lengths)[*Lstart_branch], RM5);

        switch(rrc)
          {
          case MATCH_ACCEPT:  /* Save captures */
          memcpy(Fovector,
                (char *)assert_accept_frame + offsetof(heapframe, ovector),
                assert_accept_frame->offset_top * sizeof(PCRE2_SIZE));
          Foffset_top = assert_accept_frame->offset_top;

          /* Fall through */
          /* In the case of a match, the captures have already been put into
          the current frame. */

          case MATCH_MATCH:
          condition = Lpositive;   /* TRUE for positive assertion */
          break;

          /* PCRE doesn't allow the effect of (*THEN) to escape beyond an
          assertion; it is therefore always treated as NOMATCH. */

          case MATCH_NOMATCH:
          case MATCH_THEN:
          Lstart_branch += GET(Lstart_branch, 1);
          if (*Lstart_branch == OP_ALT) continue;  /* Try next branch */
          condition = !Lpositive;  /* TRUE for negative assertion */
          break;

          /* These force no match without checking other branches. */

          case MATCH_COMMIT:
          case MATCH_SKIP:
          case MATCH_PRUNE:
          condition = !Lpositive;
          break;

          default:
          RRETURN(rrc);
          }
        break;  /* Out of the branch loop */
        }

      /* If the condition is true, find the end of the assertion so that
      advancing past it gets us to the start of the first branch. */

      if (condition)
        {
        do Fecode += GET(Fecode, 1); while (*Fecode == OP_ALT);
        }
      break;  /* End of assertion condition */
      }

#undef Lpositive
#undef Lstart_branch

    /* Choose branch according to the condition. */

    Fecode += condition? PRIV(OP_lengths)[*Fecode] : Flength;

    /* If the opcode is OP_SCOND it means we are at a repeated conditional
    group that might match an empty string. We must therefore descend a level
    so that the start is remembered for checking. For OP_COND we can just
    continue at this level. */

    if (Fop == OP_SCOND)
      {
      group_frame_type  = GF_NOCAPTURE | Fop;
      RMATCH(Fecode, RM35);
      RRETURN(rrc);
      }
    break;



/* ========================================================================= */
/*                  End of start of parenthesis opcodes                      */
/* ========================================================================= */


    /* ===================================================================== */
    /* Move the subject pointer back. This occurs only at the start of each
    branch of a lookbehind assertion. If we are too close to the start to move
    back, fail. When working with UTF-8 we move back a number of characters,
    not bytes. */

    case OP_REVERSE:
    number = GET(Fecode, 1);
#ifdef SUPPORT_UNICODE
    if (utf)
      {
      while (number-- > 0)
        {
        if (Feptr <= mb->check_subject) RRETURN(MATCH_NOMATCH);
        Feptr--;
        BACKCHAR(Feptr);
        }
      }
    else
#endif

    /* No UTF-8 support, or not in UTF-8 mode: count is code unit count */

      {
      if ((ptrdiff_t)number > Feptr - mb->start_subject) RRETURN(MATCH_NOMATCH);
      Feptr -= number;
      }

    /* Save the earliest consulted character, then skip to next opcode */

    if (Feptr < mb->start_used_ptr) mb->start_used_ptr = Feptr;
    Fecode += 1 + LINK_SIZE;
    break;


    /* ===================================================================== */
    /* An alternation is the end of a branch; scan along to find the end of the
    bracketed group. */

    case OP_ALT:
    do Fecode += GET(Fecode,1); while (*Fecode == OP_ALT);
    break;


    /* ===================================================================== */
    /* The end of a parenthesized group. For all but OP_BRA and OP_COND, the
    starting frame was added to the chained frames in order to remember the
    starting subject position for the group. */

    case OP_KET:
    case OP_KETRMIN:
    case OP_KETRMAX:
    case OP_KETRPOS:

    bracode = Fecode - GET(Fecode, 1);

    /* Point N to the frame at the start of the most recent group.
    Remember the subject pointer at the start of the group. */

    if (*bracode != OP_BRA && *bracode != OP_COND)
      {
      N = (heapframe *)((char *)mb->match_frames + Flast_group_offset);
      P = (heapframe *)((char *)N - frame_size);
      Flast_group_offset = P->last_group_offset;

#ifdef DEBUG_SHOW_RMATCH
      fprintf(stderr, "++ KET for frame=%d type=%x prev char offset=%lu\n",
        N->rdepth, N->group_frame_type,
        (char *)P->eptr - (char *)mb->start_subject);
#endif

      /* If we are at the end of an assertion that is a condition, return a
      match, discarding any intermediate backtracking points. Copy back the
      mark setting and the captures into the frame before N so that they are
      set on return. Doing this for all assertions, both positive and negative,
      seems to match what Perl does. */

      if (GF_IDMASK(N->group_frame_type) == GF_CONDASSERT)
        {
        memcpy((char *)P + offsetof(heapframe, ovector), Fovector,
          Foffset_top * sizeof(PCRE2_SIZE));
        P->offset_top = Foffset_top;
        P->mark = Fmark;
        Fback_frame = (char *)F - (char *)P;
        RRETURN(MATCH_MATCH);
        }
      }
    else P = NULL;   /* Indicates starting frame not recorded */

    /* The group was not a conditional assertion. */

    switch (*bracode)
      {
      case OP_BRA:    /* No need to do anything for these */
      case OP_COND:
      case OP_SCOND:
      break;

      /* Non-atomic positive assertions are like OP_BRA, except that the
      subject pointer must be put back to where it was at the start of the
      assertion. */

      case OP_ASSERT_NA:
      case OP_ASSERTBACK_NA:
      if (Feptr > mb->last_used_ptr) mb->last_used_ptr = Feptr;
      Feptr = P->eptr;
      break;

      /* Atomic positive assertions are like OP_ONCE, except that in addition
      the subject pointer must be put back to where it was at the start of the
      assertion. */

      case OP_ASSERT:
      case OP_ASSERTBACK:
      if (Feptr > mb->last_used_ptr) mb->last_used_ptr = Feptr;
      Feptr = P->eptr;
      /* Fall through */

      /* For an atomic group, discard internal backtracking points. We must
      also ensure that any remaining branches within the top-level of the group
      are not tried. Do this by adjusting the code pointer within the backtrack
      frame so that it points to the final branch. */

      case OP_ONCE:
      Fback_frame = ((char *)F - (char *)P);
      for (;;)
        {
        uint32_t y = GET(P->ecode,1);
        if ((P->ecode)[y] != OP_ALT) break;
        P->ecode += y;
        }
      break;

      /* A matching negative assertion returns MATCH, which is turned into
      NOMATCH at the assertion level. */

      case OP_ASSERT_NOT:
      case OP_ASSERTBACK_NOT:
      RRETURN(MATCH_MATCH);

      /* At the end of a script run, apply the script-checking rules. This code
      will never by exercised if Unicode support it not compiled, because in
      that environment script runs cause an error at compile time. */

      case OP_SCRIPT_RUN:
      if (!PRIV(script_run)(P->eptr, Feptr, utf)) RRETURN(MATCH_NOMATCH);
      break;

      /* Whole-pattern recursion is coded as a recurse into group 0, so it
      won't be picked up here. Instead, we catch it when the OP_END is reached.
      Other recursion is handled here. */

      case OP_CBRA:
      case OP_CBRAPOS:
      case OP_SCBRA:
      case OP_SCBRAPOS:
      number = GET2(bracode, 1+LINK_SIZE);

      /* Handle a recursively called group. We reinstate the previous set of
      captures and then carry on after the recursion call. */

      if (Fcurrent_recurse == number)
        {
        P = (heapframe *)((char *)N - frame_size);
        memcpy((char *)F + offsetof(heapframe, ovector), P->ovector,
          P->offset_top * sizeof(PCRE2_SIZE));
        Foffset_top = P->offset_top;
        Fcapture_last = P->capture_last;
        Fcurrent_recurse = P->current_recurse;
        Fecode = P->ecode + 1 + LINK_SIZE;
        continue;  /* With next opcode */
        }

      /* Deal with actual capturing. */

      offset = (number << 1) - 2;
      Fcapture_last = number;
      Fovector[offset] = P->eptr - mb->start_subject;
      Fovector[offset+1] = Feptr - mb->start_subject;
      if (offset >= Foffset_top) Foffset_top = offset + 2;
      break;
      }  /* End actions relating to the starting opcode */

    /* OP_KETRPOS is a possessive repeating ket. Remember the current position,
    and return the MATCH_KETRPOS. This makes it possible to do the repeats one
    at a time from the outer level. This must precede the empty string test -
    in this case that test is done at the outer level. */

    if (*Fecode == OP_KETRPOS)
      {
      memcpy((char *)P + offsetof(heapframe, eptr),
             (char *)F + offsetof(heapframe, eptr),
             frame_copy_size);
      RRETURN(MATCH_KETRPOS);
      }

    /* Handle the different kinds of closing brackets. A non-repeating ket
    needs no special action, just continuing at this level. This also happens
    for the repeating kets if the group matched no characters, in order to
    forcibly break infinite loops. Otherwise, the repeating kets try the rest
    of the pattern or restart from the preceding bracket, in the appropriate
    order. */

    if (Fop != OP_KET && (P == NULL || Feptr != P->eptr))
      {
      if (Fop == OP_KETRMIN)
        {
        RMATCH(Fecode + 1 + LINK_SIZE, RM6);
        if (rrc != MATCH_NOMATCH) RRETURN(rrc);
        Fecode -= GET(Fecode, 1);
        break;   /* End of ket processing */
        }

      /* Repeat the maximum number of times (KETRMAX) */

      RMATCH(bracode, RM7);
      if (rrc != MATCH_NOMATCH) RRETURN(rrc);
      }

    /* Carry on at this level for a non-repeating ket, or after matching an
    empty string, or after repeating for a maximum number of times. */

    Fecode += 1 + LINK_SIZE;
    break;


    /* ===================================================================== */
    /* Start and end of line assertions, not multiline mode. */

    case OP_CIRC:   /* Start of line, unless PCRE2_NOTBOL is set. */
    if (Feptr != mb->start_subject || (mb->moptions & PCRE2_NOTBOL) != 0)
      RRETURN(MATCH_NOMATCH);
    Fecode++;
    break;

    case OP_SOD:    /* Unconditional start of subject */
    if (Feptr != mb->start_subject) RRETURN(MATCH_NOMATCH);
    Fecode++;
    break;

    /* When PCRE2_NOTEOL is unset, assert before the subject end, or a
    terminating newline unless PCRE2_DOLLAR_ENDONLY is set. */

    case OP_DOLL:
    if ((mb->moptions & PCRE2_NOTEOL) != 0) RRETURN(MATCH_NOMATCH);
    if ((mb->poptions & PCRE2_DOLLAR_ENDONLY) == 0) goto ASSERT_NL_OR_EOS;

    /* Fall through */
    /* Unconditional end of subject assertion (\z) */

    case OP_EOD:
    if (Feptr < mb->end_subject) RRETURN(MATCH_NOMATCH);
    if (mb->partial != 0)
      {
      mb->hitend = TRUE;
      if (mb->partial > 1) return PCRE2_ERROR_PARTIAL;
      }
    Fecode++;
    break;

    /* End of subject or ending \n assertion (\Z) */

    case OP_EODN:
    ASSERT_NL_OR_EOS:
    if (Feptr < mb->end_subject &&
        (!IS_NEWLINE(Feptr) || Feptr != mb->end_subject - mb->nllen))
      {
      if (mb->partial != 0 &&
          Feptr + 1 >= mb->end_subject &&
          NLBLOCK->nltype == NLTYPE_FIXED &&
          NLBLOCK->nllen == 2 &&
          UCHAR21TEST(Feptr) == NLBLOCK->nl[0])
        {
        mb->hitend = TRUE;
        if (mb->partial > 1) return PCRE2_ERROR_PARTIAL;
        }
      RRETURN(MATCH_NOMATCH);
      }

    /* Either at end of string or \n before end. */

    if (mb->partial != 0)
      {
      mb->hitend = TRUE;
      if (mb->partial > 1) return PCRE2_ERROR_PARTIAL;
      }
    Fecode++;
    break;


    /* ===================================================================== */
    /* Start and end of line assertions, multiline mode. */

    /* Start of subject unless notbol, or after any newline except for one at
    the very end, unless PCRE2_ALT_CIRCUMFLEX is set. */

    case OP_CIRCM:
    if ((mb->moptions & PCRE2_NOTBOL) != 0 && Feptr == mb->start_subject)
      RRETURN(MATCH_NOMATCH);
    if (Feptr != mb->start_subject &&
        ((Feptr == mb->end_subject &&
           (mb->poptions & PCRE2_ALT_CIRCUMFLEX) == 0) ||
         !WAS_NEWLINE(Feptr)))
      RRETURN(MATCH_NOMATCH);
    Fecode++;
    break;

    /* Assert before any newline, or before end of subject unless noteol is
    set. */

    case OP_DOLLM:
    if (Feptr < mb->end_subject)
      {
      if (!IS_NEWLINE(Feptr))
        {
        if (mb->partial != 0 &&
            Feptr + 1 >= mb->end_subject &&
            NLBLOCK->nltype == NLTYPE_FIXED &&
            NLBLOCK->nllen == 2 &&
            UCHAR21TEST(Feptr) == NLBLOCK->nl[0])
          {
          mb->hitend = TRUE;
          if (mb->partial > 1) return PCRE2_ERROR_PARTIAL;
          }
        RRETURN(MATCH_NOMATCH);
        }
      }
    else
      {
      if ((mb->moptions & PCRE2_NOTEOL) != 0) RRETURN(MATCH_NOMATCH);
      SCHECK_PARTIAL();
      }
    Fecode++;
    break;


    /* ===================================================================== */
    /* Start of match assertion */

    case OP_SOM:
    if (Feptr != mb->start_subject + mb->start_offset) RRETURN(MATCH_NOMATCH);
    Fecode++;
    break;


    /* ===================================================================== */
    /* Reset the start of match point */

    case OP_SET_SOM:
    Fstart_match = Feptr;
    Fecode++;
    break;


    /* ===================================================================== */
    /* Word boundary assertions. Find out if the previous and current
    characters are "word" characters. It takes a bit more work in UTF mode.
    Characters > 255 are assumed to be "non-word" characters when PCRE2_UCP is
    not set. When it is set, use Unicode properties if available, even when not
    in UTF mode. Remember the earliest and latest consulted characters. */

    case OP_NOT_WORD_BOUNDARY:
    case OP_WORD_BOUNDARY:
    if (Feptr == mb->check_subject) prev_is_word = FALSE; else
      {
      PCRE2_SPTR lastptr = Feptr - 1;
#ifdef SUPPORT_UNICODE
      if (utf)
        {
        BACKCHAR(lastptr);
        GETCHAR(fc, lastptr);
        }
      else
#endif  /* SUPPORT_UNICODE */
      fc = *lastptr;
      if (lastptr < mb->start_used_ptr) mb->start_used_ptr = lastptr;
#ifdef SUPPORT_UNICODE
      if ((mb->poptions & PCRE2_UCP) != 0)
        {
        if (fc == '_') prev_is_word = TRUE; else
          {
          int cat = UCD_CATEGORY(fc);
          prev_is_word = (cat == ucp_L || cat == ucp_N);
          }
        }
      else
#endif  /* SUPPORT_UNICODE */
      prev_is_word = CHMAX_255(fc) && (mb->ctypes[fc] & ctype_word) != 0;
      }

    /* Get status of next character */

    if (Feptr >= mb->end_subject)
      {
      SCHECK_PARTIAL();
      cur_is_word = FALSE;
      }
    else
      {
      PCRE2_SPTR nextptr = Feptr + 1;
#ifdef SUPPORT_UNICODE
      if (utf)
        {
        FORWARDCHARTEST(nextptr, mb->end_subject);
        GETCHAR(fc, Feptr);
        }
      else
#endif  /* SUPPORT_UNICODE */
      fc = *Feptr;
      if (nextptr > mb->last_used_ptr) mb->last_used_ptr = nextptr;
#ifdef SUPPORT_UNICODE
      if ((mb->poptions & PCRE2_UCP) != 0)
        {
        if (fc == '_') cur_is_word = TRUE; else
          {
          int cat = UCD_CATEGORY(fc);
          cur_is_word = (cat == ucp_L || cat == ucp_N);
          }
        }
      else
#endif  /* SUPPORT_UNICODE */
      cur_is_word = CHMAX_255(fc) && (mb->ctypes[fc] & ctype_word) != 0;
      }

    /* Now see if the situation is what we want */

    if ((*Fecode++ == OP_WORD_BOUNDARY)?
         cur_is_word == prev_is_word : cur_is_word != prev_is_word)
      RRETURN(MATCH_NOMATCH);
    break;


    /* ===================================================================== */
    /* Backtracking (*VERB)s, with and without arguments. Note that if the
    pattern is successfully matched, we do not come back from RMATCH. */

    case OP_MARK:
    Fmark = mb->nomatch_mark = Fecode + 2;
    RMATCH(Fecode + PRIV(OP_lengths)[*Fecode] + Fecode[1], RM12);

    /* A return of MATCH_SKIP_ARG means that matching failed at SKIP with an
    argument, and we must check whether that argument matches this MARK's
    argument. It is passed back in mb->verb_skip_ptr. If it does match, we
    return MATCH_SKIP with mb->verb_skip_ptr now pointing to the subject
    position that corresponds to this mark. Otherwise, pass back the return
    code unaltered. */

    if (rrc == MATCH_SKIP_ARG &&
             PRIV(strcmp)(Fecode + 2, mb->verb_skip_ptr) == 0)
      {
      mb->verb_skip_ptr = Feptr;   /* Pass back current position */
      RRETURN(MATCH_SKIP);
      }
    RRETURN(rrc);

    case OP_FAIL:
    RRETURN(MATCH_NOMATCH);

    /* Record the current recursing group number in mb->verb_current_recurse
    when a backtracking return such as MATCH_COMMIT is given. This enables the
    recurse processing to catch verbs from within the recursion. */

    case OP_COMMIT:
    RMATCH(Fecode + PRIV(OP_lengths)[*Fecode], RM13);
    if (rrc != MATCH_NOMATCH) RRETURN(rrc);
    mb->verb_current_recurse = Fcurrent_recurse;
    RRETURN(MATCH_COMMIT);

    case OP_COMMIT_ARG:
    Fmark = mb->nomatch_mark = Fecode + 2;
    RMATCH(Fecode + PRIV(OP_lengths)[*Fecode] + Fecode[1], RM36);
    if (rrc != MATCH_NOMATCH) RRETURN(rrc);
    mb->verb_current_recurse = Fcurrent_recurse;
    RRETURN(MATCH_COMMIT);

    case OP_PRUNE:
    RMATCH(Fecode + PRIV(OP_lengths)[*Fecode], RM14);
    if (rrc != MATCH_NOMATCH) RRETURN(rrc);
    mb->verb_current_recurse = Fcurrent_recurse;
    RRETURN(MATCH_PRUNE);

    case OP_PRUNE_ARG:
    Fmark = mb->nomatch_mark = Fecode + 2;
    RMATCH(Fecode + PRIV(OP_lengths)[*Fecode] + Fecode[1], RM15);
    if (rrc != MATCH_NOMATCH) RRETURN(rrc);
    mb->verb_current_recurse = Fcurrent_recurse;
    RRETURN(MATCH_PRUNE);

    case OP_SKIP:
    RMATCH(Fecode + PRIV(OP_lengths)[*Fecode], RM16);
    if (rrc != MATCH_NOMATCH) RRETURN(rrc);
    mb->verb_skip_ptr = Feptr;   /* Pass back current position */
    mb->verb_current_recurse = Fcurrent_recurse;
    RRETURN(MATCH_SKIP);

    /* Note that, for Perl compatibility, SKIP with an argument does NOT set
    nomatch_mark. When a pattern match ends with a SKIP_ARG for which there was
    not a matching mark, we have to re-run the match, ignoring the SKIP_ARG
    that failed and any that precede it (either they also failed, or were not
    triggered). To do this, we maintain a count of executed SKIP_ARGs. If a
    SKIP_ARG gets to top level, the match is re-run with mb->ignore_skip_arg
    set to the count of the one that failed. */

    case OP_SKIP_ARG:
    mb->skip_arg_count++;
    if (mb->skip_arg_count <= mb->ignore_skip_arg)
      {
      Fecode += PRIV(OP_lengths)[*Fecode] + Fecode[1];
      break;
      }
    RMATCH(Fecode + PRIV(OP_lengths)[*Fecode] + Fecode[1], RM17);
    if (rrc != MATCH_NOMATCH) RRETURN(rrc);

    /* Pass back the current skip name and return the special MATCH_SKIP_ARG
    return code. This will either be caught by a matching MARK, or get to the
    top, where it causes a rematch with mb->ignore_skip_arg set to the value of
    mb->skip_arg_count. */

    mb->verb_skip_ptr = Fecode + 2;
    mb->verb_current_recurse = Fcurrent_recurse;
    RRETURN(MATCH_SKIP_ARG);

    /* For THEN (and THEN_ARG) we pass back the address of the opcode, so that
    the branch in which it occurs can be determined. */

    case OP_THEN:
    RMATCH(Fecode + PRIV(OP_lengths)[*Fecode], RM18);
    if (rrc != MATCH_NOMATCH) RRETURN(rrc);
    mb->verb_ecode_ptr = Fecode;
    mb->verb_current_recurse = Fcurrent_recurse;
    RRETURN(MATCH_THEN);

    case OP_THEN_ARG:
    Fmark = mb->nomatch_mark = Fecode + 2;
    RMATCH(Fecode + PRIV(OP_lengths)[*Fecode] + Fecode[1], RM19);
    if (rrc != MATCH_NOMATCH) RRETURN(rrc);
    mb->verb_ecode_ptr = Fecode;
    mb->verb_current_recurse = Fcurrent_recurse;
    RRETURN(MATCH_THEN);


    /* ===================================================================== */
    /* There's been some horrible disaster. Arrival here can only mean there is
    something seriously wrong in the code above or the OP_xxx definitions. */

    default:
    return PCRE2_ERROR_INTERNAL;
    }

  /* Do not insert any code in here without much thought; it is assumed
  that "continue" in the code above comes out to here to repeat the main
  loop. */

  }  /* End of main loop */
/* Control never reaches here */


/* ========================================================================= */
/* The RRETURN() macro jumps here. The number that is saved in Freturn_id
indicates which label we actually want to return to. The value in Frdepth is
the index number of the frame in the vector. The return value has been placed
in rrc. */

#define LBL(val) case val: goto L_RM##val;

RETURN_SWITCH:
if (Feptr > mb->last_used_ptr) mb->last_used_ptr = Feptr;
if (Frdepth == 0) return rrc;                     /* Exit from the top level */
F = (heapframe *)((char *)F - Fback_frame);       /* Backtrack */
mb->cb->callout_flags |= PCRE2_CALLOUT_BACKTRACK; /* Note for callouts */

#ifdef DEBUG_SHOW_RMATCH
fprintf(stderr, "++ RETURN %d to %d\n", rrc, Freturn_id);
#endif

switch (Freturn_id)
  {
  LBL( 1) LBL( 2) LBL( 3) LBL( 4) LBL( 5) LBL( 6) LBL( 7) LBL( 8)
  LBL( 9) LBL(10) LBL(11) LBL(12) LBL(13) LBL(14) LBL(15) LBL(16)
  LBL(17) LBL(18) LBL(19) LBL(20) LBL(21) LBL(22) LBL(23) LBL(24)
  LBL(25) LBL(26) LBL(27) LBL(28) LBL(29) LBL(30) LBL(31) LBL(32)
  LBL(33) LBL(34) LBL(35) LBL(36)

#ifdef SUPPORT_WIDE_CHARS
  LBL(100) LBL(101)
#endif

#ifdef SUPPORT_UNICODE
  LBL(200) LBL(201) LBL(202) LBL(203) LBL(204) LBL(205) LBL(206)
  LBL(207) LBL(208) LBL(209) LBL(210) LBL(211) LBL(212) LBL(213)
  LBL(214) LBL(215) LBL(216) LBL(217) LBL(218) LBL(219) LBL(220)
  LBL(221) LBL(222)
#endif

  default:
  return PCRE2_ERROR_INTERNAL;
  }
#undef LBL
}


/*************************************************
*           Match a Regular Expression           *
*************************************************/

/* This function applies a compiled pattern to a subject string and picks out
portions of the string if it matches. Two elements in the vector are set for
each substring: the offsets to the start and end of the substring.

Arguments:
  code            points to the compiled expression
  subject         points to the subject string
  length          length of subject string (may contain binary zeros)
  start_offset    where to start in the subject string
  options         option bits
  match_data      points to a match_data block
  mcontext        points a PCRE2 context

Returns:          > 0 => success; value is the number of ovector pairs filled
                  = 0 => success, but ovector is not big enough
                  = -1 => failed to match (PCRE2_ERROR_NOMATCH)
                  = -2 => partial match (PCRE2_ERROR_PARTIAL)
                  < -2 => some kind of unexpected problem
*/

PCRE2_EXP_DEFN int PCRE2_CALL_CONVENTION
pcre2_match(const pcre2_code *code, PCRE2_SPTR subject, PCRE2_SIZE length,
  PCRE2_SIZE start_offset, uint32_t options, pcre2_match_data *match_data,
  pcre2_match_context *mcontext)
{
int rc;
int was_zero_terminated = 0;
const uint8_t *start_bits = NULL;
const pcre2_real_code *re = (const pcre2_real_code *)code;

BOOL anchored;
BOOL firstline;
BOOL has_first_cu = FALSE;
BOOL has_req_cu = FALSE;
BOOL startline;
BOOL utf;

#if PCRE2_CODE_UNIT_WIDTH == 8
BOOL memchr_not_found_first_cu = FALSE;
BOOL memchr_not_found_first_cu2 = FALSE;
#endif

PCRE2_UCHAR first_cu = 0;
PCRE2_UCHAR first_cu2 = 0;
PCRE2_UCHAR req_cu = 0;
PCRE2_UCHAR req_cu2 = 0;

PCRE2_SPTR bumpalong_limit;
PCRE2_SPTR end_subject;
PCRE2_SPTR true_end_subject;
PCRE2_SPTR start_match = subject + start_offset;
PCRE2_SPTR req_cu_ptr = start_match - 1;
PCRE2_SPTR start_partial;
PCRE2_SPTR match_partial;

#ifdef SUPPORT_JIT
BOOL use_jit;
#endif

#ifdef SUPPORT_UNICODE
BOOL allow_invalid;
uint32_t fragment_options = 0;
#ifdef SUPPORT_JIT
BOOL jit_checked_utf = FALSE;
#endif
#endif

PCRE2_SIZE frame_size;

/* We need to have mb as a pointer to a match block, because the IS_NEWLINE
macro is used below, and it expects NLBLOCK to be defined as a pointer. */

pcre2_callout_block cb;
match_block actual_match_block;
match_block *mb = &actual_match_block;

/* Allocate an initial vector of backtracking frames on the stack. If this
proves to be too small, it is replaced by a larger one on the heap. To get a
vector of the size required that is aligned for pointers, allocate it as a
vector of pointers. */

PCRE2_SPTR stack_frames_vector[START_FRAMES_SIZE/sizeof(PCRE2_SPTR)];
mb->stack_frames = (heapframe *)stack_frames_vector;

/* A length equal to PCRE2_ZERO_TERMINATED implies a zero-terminated
subject string. */

if (length == PCRE2_ZERO_TERMINATED)
  {
  length = PRIV(strlen)(subject);
  was_zero_terminated = 1;
  }
true_end_subject = end_subject = subject + length;

/* Plausibility checks */

if ((options & ~PUBLIC_MATCH_OPTIONS) != 0) return PCRE2_ERROR_BADOPTION;
if (code == NULL || subject == NULL || match_data == NULL)
  return PCRE2_ERROR_NULL;
if (start_offset > length) return PCRE2_ERROR_BADOFFSET;

/* Check that the first field in the block is the magic number. */

if (re->magic_number != MAGIC_NUMBER) return PCRE2_ERROR_BADMAGIC;

/* Check the code unit width. */

if ((re->flags & PCRE2_MODE_MASK) != PCRE2_CODE_UNIT_WIDTH/8)
  return PCRE2_ERROR_BADMODE;

/* PCRE2_NOTEMPTY and PCRE2_NOTEMPTY_ATSTART are match-time flags in the
options variable for this function. Users of PCRE2 who are not calling the
function directly would like to have a way of setting these flags, in the same
way that they can set pcre2_compile() flags like PCRE2_NO_AUTOPOSSESS with
constructions like (*NO_AUTOPOSSESS). To enable this, (*NOTEMPTY) and
(*NOTEMPTY_ATSTART) set bits in the pattern's "flag" function which we now
transfer to the options for this function. The bits are guaranteed to be
adjacent, but do not have the same values. This bit of Boolean trickery assumes
that the match-time bits are not more significant than the flag bits. If by
accident this is not the case, a compile-time division by zero error will
occur. */

#define FF (PCRE2_NOTEMPTY_SET|PCRE2_NE_ATST_SET)
#define OO (PCRE2_NOTEMPTY|PCRE2_NOTEMPTY_ATSTART)
options |= (re->flags & FF) / ((FF & (~FF+1)) / (OO & (~OO+1)));
#undef FF
#undef OO

/* If the pattern was successfully studied with JIT support, we will run the
JIT executable instead of the rest of this function. Most options must be set
at compile time for the JIT code to be usable. */

#ifdef SUPPORT_JIT
use_jit = (re->executable_jit != NULL &&
          (options & ~PUBLIC_JIT_MATCH_OPTIONS) == 0);
#endif

/* Initialize UTF parameters. */

utf = (re->overall_options & PCRE2_UTF) != 0;
#ifdef SUPPORT_UNICODE
allow_invalid = (re->overall_options & PCRE2_MATCH_INVALID_UTF) != 0;
#endif

/* Convert the partial matching flags into an integer. */

mb->partial = ((options & PCRE2_PARTIAL_HARD) != 0)? 2 :
              ((options & PCRE2_PARTIAL_SOFT) != 0)? 1 : 0;

/* Partial matching and PCRE2_ENDANCHORED are currently not allowed at the same
time. */

if (mb->partial != 0 &&
   ((re->overall_options | options) & PCRE2_ENDANCHORED) != 0)
  return PCRE2_ERROR_BADOPTION;

/* It is an error to set an offset limit without setting the flag at compile
time. */

if (mcontext != NULL && mcontext->offset_limit != PCRE2_UNSET &&
     (re->overall_options & PCRE2_USE_OFFSET_LIMIT) == 0)
  return PCRE2_ERROR_BADOFFSETLIMIT;

/* If the match data block was previously used with PCRE2_COPY_MATCHED_SUBJECT,
free the memory that was obtained. Set the field to NULL for no match cases. */

if ((match_data->flags & PCRE2_MD_COPIED_SUBJECT) != 0)
  {
  match_data->memctl.free((void *)match_data->subject,
    match_data->memctl.memory_data);
  match_data->flags &= ~PCRE2_MD_COPIED_SUBJECT;
  }
match_data->subject = NULL;

/* Zero the error offset in case the first code unit is invalid UTF. */

match_data->startchar = 0;


/* ============================= JIT matching ============================== */

/* Prepare for JIT matching. Check a UTF string for validity unless no check is
requested or invalid UTF can be handled. We check only the portion of the
subject that might be be inspected during matching - from the offset minus the
maximum lookbehind to the given length. This saves time when a small part of a
large subject is being matched by the use of a starting offset. Note that the
maximum lookbehind is a number of characters, not code units. */

#ifdef SUPPORT_JIT
if (use_jit)
  {
#ifdef SUPPORT_UNICODE
  if (utf && (options & PCRE2_NO_UTF_CHECK) == 0 && !allow_invalid)
    {
#if PCRE2_CODE_UNIT_WIDTH != 32
    unsigned int i;
#endif

    /* For 8-bit and 16-bit UTF, check that the first code unit is a valid
    character start. */

#if PCRE2_CODE_UNIT_WIDTH != 32
    if (start_match < end_subject && NOT_FIRSTCU(*start_match))
      {
      if (start_offset > 0) return PCRE2_ERROR_BADUTFOFFSET;
#if PCRE2_CODE_UNIT_WIDTH == 8
      return PCRE2_ERROR_UTF8_ERR20;  /* Isolated 0x80 byte */
#else
      return PCRE2_ERROR_UTF16_ERR3;  /* Isolated low surrogate */
#endif
      }
#endif  /* WIDTH != 32 */

    /* Move back by the maximum lookbehind, just in case it happens at the very
    start of matching. */

#if PCRE2_CODE_UNIT_WIDTH != 32
    for (i = re->max_lookbehind; i > 0 && start_match > subject; i--)
      {
      start_match--;
      while (start_match > subject &&
#if PCRE2_CODE_UNIT_WIDTH == 8
      (*start_match & 0xc0) == 0x80)
#else  /* 16-bit */
      (*start_match & 0xfc00) == 0xdc00)
#endif
        start_match--;
      }
#else  /* PCRE2_CODE_UNIT_WIDTH != 32 */

    /* In the 32-bit library, one code unit equals one character. However,
    we cannot just subtract the lookbehind and then compare pointers, because
    a very large lookbehind could create an invalid pointer. */

    if (start_offset >= re->max_lookbehind)
      start_match -= re->max_lookbehind;
    else
      start_match = subject;
#endif  /* PCRE2_CODE_UNIT_WIDTH != 32 */

    /* Validate the relevant portion of the subject. Adjust the offset of an
    invalid code point to be an absolute offset in the whole string. */

    match_data->rc = PRIV(valid_utf)(start_match,
      length - (start_match - subject), &(match_data->startchar));
    if (match_data->rc != 0)
      {
      match_data->startchar += start_match - subject;
      return match_data->rc;
      }
    jit_checked_utf = TRUE;
    }
#endif  /* SUPPORT_UNICODE */

  /* If JIT returns BADOPTION, which means that the selected complete or
  partial matching mode was not compiled, fall through to the interpreter. */

  rc = pcre2_jit_match(code, subject, length, start_offset, options,
    match_data, mcontext);
  if (rc != PCRE2_ERROR_JIT_BADOPTION)
    {
    if (rc >= 0 && (options & PCRE2_COPY_MATCHED_SUBJECT) != 0)
      {
      length = CU2BYTES(length + was_zero_terminated);
      match_data->subject = match_data->memctl.malloc(length,
        match_data->memctl.memory_data);
      if (match_data->subject == NULL) return PCRE2_ERROR_NOMEMORY;
      memcpy((void *)match_data->subject, subject, length);
      match_data->flags |= PCRE2_MD_COPIED_SUBJECT;
      }
    return rc;
    }
  }
#endif  /* SUPPORT_JIT */

/* ========================= End of JIT matching ========================== */


/* Proceed with non-JIT matching. The default is to allow lookbehinds to the
start of the subject. A UTF check when there is a non-zero offset may change
this. */

mb->check_subject = subject;

/* If a UTF subject string was not checked for validity in the JIT code above,
check it here, and handle support for invalid UTF strings. The check above
happens only when invalid UTF is not supported and PCRE2_NO_CHECK_UTF is unset.
If we get here in those circumstances, it means the subject string is valid,
but for some reason JIT matching was not successful. There is no need to check
the subject again.

We check only the portion of the subject that might be be inspected during
matching - from the offset minus the maximum lookbehind to the given length.
This saves time when a small part of a large subject is being matched by the
use of a starting offset. Note that the maximum lookbehind is a number of
characters, not code units.

Note also that support for invalid UTF forces a check, overriding the setting
of PCRE2_NO_CHECK_UTF. */

#ifdef SUPPORT_UNICODE
if (utf &&
#ifdef SUPPORT_JIT
    !jit_checked_utf &&
#endif
    ((options & PCRE2_NO_UTF_CHECK) == 0 || allow_invalid))
  {
#if PCRE2_CODE_UNIT_WIDTH != 32
  BOOL skipped_bad_start = FALSE;
#endif

  /* For 8-bit and 16-bit UTF, check that the first code unit is a valid
  character start. If we are handling invalid UTF, just skip over such code
  units. Otherwise, give an appropriate error. */

#if PCRE2_CODE_UNIT_WIDTH != 32
  if (allow_invalid)
    {
    while (start_match < end_subject && NOT_FIRSTCU(*start_match))
      {
      start_match++;
      skipped_bad_start = TRUE;
      }
    }
  else if (start_match < end_subject && NOT_FIRSTCU(*start_match))
    {
    if (start_offset > 0) return PCRE2_ERROR_BADUTFOFFSET;
#if PCRE2_CODE_UNIT_WIDTH == 8
    return PCRE2_ERROR_UTF8_ERR20;  /* Isolated 0x80 byte */
#else
    return PCRE2_ERROR_UTF16_ERR3;  /* Isolated low surrogate */
#endif
    }
#endif  /* WIDTH != 32 */

  /* The mb->check_subject field points to the start of UTF checking;
  lookbehinds can go back no further than this. */

  mb->check_subject = start_match;

  /* Move back by the maximum lookbehind, just in case it happens at the very
  start of matching, but don't do this if we skipped bad 8-bit or 16-bit code
  units above. */

#if PCRE2_CODE_UNIT_WIDTH != 32
  if (!skipped_bad_start)
    {
    unsigned int i;
    for (i = re->max_lookbehind; i > 0 && mb->check_subject > subject; i--)
      {
      mb->check_subject--;
      while (mb->check_subject > subject &&
#if PCRE2_CODE_UNIT_WIDTH == 8
      (*mb->check_subject & 0xc0) == 0x80)
#else  /* 16-bit */
      (*mb->check_subject & 0xfc00) == 0xdc00)
#endif
        mb->check_subject--;
      }
    }
#else  /* PCRE2_CODE_UNIT_WIDTH != 32 */

  /* In the 32-bit library, one code unit equals one character. However,
  we cannot just subtract the lookbehind and then compare pointers, because
  a very large lookbehind could create an invalid pointer. */

  if (start_offset >= re->max_lookbehind)
    mb->check_subject -= re->max_lookbehind;
  else
    mb->check_subject = subject;
#endif  /* PCRE2_CODE_UNIT_WIDTH != 32 */

  /* Validate the relevant portion of the subject. There's a loop in case we
  encounter bad UTF in the characters preceding start_match which we are
  scanning because of a lookbehind. */

  for (;;)
    {
    match_data->rc = PRIV(valid_utf)(mb->check_subject,
      length - (mb->check_subject - subject), &(match_data->startchar));

    if (match_data->rc == 0) break;   /* Valid UTF string */

    /* Invalid UTF string. Adjust the offset to be an absolute offset in the
    whole string. If we are handling invalid UTF strings, set end_subject to
    stop before the bad code unit, and set the options to "not end of line".
    Otherwise return the error. */

    match_data->startchar += mb->check_subject - subject;
    if (!allow_invalid || match_data->rc > 0) return match_data->rc;
    end_subject = subject + match_data->startchar;

    /* If the end precedes start_match, it means there is invalid UTF in the
    extra code units we reversed over because of a lookbehind. Advance past the
    first bad code unit, and then skip invalid character starting code units in
    8-bit and 16-bit modes, and try again. */

    if (end_subject < start_match)
      {
      mb->check_subject = end_subject + 1;
#if PCRE2_CODE_UNIT_WIDTH != 32
      while (mb->check_subject < start_match && NOT_FIRSTCU(*mb->check_subject))
        mb->check_subject++;
#endif
      }

    /* Otherwise, set the not end of line option, and do the match. */

    else
      {
      fragment_options = PCRE2_NOTEOL;
      break;
      }
    }
  }
#endif  /* SUPPORT_UNICODE */

/* A NULL match context means "use a default context", but we take the memory
control functions from the pattern. */

if (mcontext == NULL)
  {
  mcontext = (pcre2_match_context *)(&PRIV(default_match_context));
  mb->memctl = re->memctl;
  }
else mb->memctl = mcontext->memctl;

anchored = ((re->overall_options | options) & PCRE2_ANCHORED) != 0;
firstline = (re->overall_options & PCRE2_FIRSTLINE) != 0;
startline = (re->flags & PCRE2_STARTLINE) != 0;
bumpalong_limit = (mcontext->offset_limit == PCRE2_UNSET)?
  true_end_subject : subject + mcontext->offset_limit;

/* Initialize and set up the fixed fields in the callout block, with a pointer
in the match block. */

mb->cb = &cb;
cb.version = 2;
cb.subject = subject;
cb.subject_length = (PCRE2_SIZE)(end_subject - subject);
cb.callout_flags = 0;

/* Fill in the remaining fields in the match block, except for moptions, which
gets set later. */

mb->callout = mcontext->callout;
mb->callout_data = mcontext->callout_data;

mb->start_subject = subject;
mb->start_offset = start_offset;
mb->end_subject = end_subject;
mb->hasthen = (re->flags & PCRE2_HASTHEN) != 0;
mb->allowemptypartial = (re->max_lookbehind > 0) ||
    (re->flags & PCRE2_MATCH_EMPTY) != 0;
mb->poptions = re->overall_options;          /* Pattern options */
mb->ignore_skip_arg = 0;
mb->mark = mb->nomatch_mark = NULL;          /* In case never set */

/* The name table is needed for finding all the numbers associated with a
given name, for condition testing. The code follows the name table. */

mb->name_table = (PCRE2_UCHAR *)((uint8_t *)re + sizeof(pcre2_real_code));
mb->name_count = re->name_count;
mb->name_entry_size = re->name_entry_size;
mb->start_code = mb->name_table + re->name_count * re->name_entry_size;

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

/* The backtracking frames have fixed data at the front, and a PCRE2_SIZE
vector at the end, whose size depends on the number of capturing parentheses in
the pattern. It is not used at all if there are no capturing parentheses.

  frame_size             is the total size of each frame
  mb->frame_vector_size  is the total usable size of the vector (rounded down
                           to a whole number of frames)

The last of these is changed within the match() function if the frame vector
has to be expanded. We therefore put it into the match block so that it is
correct when calling match() more than once for non-anchored patterns. */

frame_size = offsetof(heapframe, ovector) +
  re->top_bracket * 2 * sizeof(PCRE2_SIZE);

/* Limits set in the pattern override the match context only if they are
smaller. */

mb->heap_limit = (mcontext->heap_limit < re->limit_heap)?
  mcontext->heap_limit : re->limit_heap;

mb->match_limit = (mcontext->match_limit < re->limit_match)?
  mcontext->match_limit : re->limit_match;

mb->match_limit_depth = (mcontext->depth_limit < re->limit_depth)?
  mcontext->depth_limit : re->limit_depth;

/* If a pattern has very many capturing parentheses, the frame size may be very
large. Ensure that there are at least 10 available frames by getting an initial
vector on the heap if necessary, except when the heap limit prevents this. Get
fewer if possible. (The heap limit is in kibibytes.) */

if (frame_size <= START_FRAMES_SIZE/10)
  {
  mb->match_frames = mb->stack_frames;   /* Initial frame vector on the stack */
  mb->frame_vector_size = ((START_FRAMES_SIZE/frame_size) * frame_size);
  }
else
  {
  mb->frame_vector_size = frame_size * 10;
  if ((mb->frame_vector_size / 1024) > mb->heap_limit)
    {
    if (frame_size > mb->heap_limit * 1024) return PCRE2_ERROR_HEAPLIMIT;
    mb->frame_vector_size = ((mb->heap_limit * 1024)/frame_size) * frame_size;
    }
  mb->match_frames = mb->memctl.malloc(mb->frame_vector_size,
    mb->memctl.memory_data);
  if (mb->match_frames == NULL) return PCRE2_ERROR_NOMEMORY;
  }

mb->match_frames_top =
  (heapframe *)((char *)mb->match_frames + mb->frame_vector_size);

/* Write to the ovector within the first frame to mark every capture unset and
to avoid uninitialized memory read errors when it is copied to a new frame. */

memset((char *)(mb->match_frames) + offsetof(heapframe, ovector), 0xff,
  re->top_bracket * 2 * sizeof(PCRE2_SIZE));

/* Pointers to the individual character tables */

mb->lcc = re->tables + lcc_offset;
mb->fcc = re->tables + fcc_offset;
mb->ctypes = re->tables + ctypes_offset;

/* Set up the first code unit to match, if available. If there's no first code
unit there may be a bitmap of possible first characters. */

if ((re->flags & PCRE2_FIRSTSET) != 0)
  {
  has_first_cu = TRUE;
  first_cu = first_cu2 = (PCRE2_UCHAR)(re->first_codeunit);
  if ((re->flags & PCRE2_FIRSTCASELESS) != 0)
    {
    first_cu2 = TABLE_GET(first_cu, mb->fcc, first_cu);
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 8
    if (utf && first_cu > 127) first_cu2 = UCD_OTHERCASE(first_cu);
#endif
    }
  }
else
  if (!startline && (re->flags & PCRE2_FIRSTMAPSET) != 0)
    start_bits = re->start_bitmap;

/* There may also be a "last known required character" set. */

if ((re->flags & PCRE2_LASTSET) != 0)
  {
  has_req_cu = TRUE;
  req_cu = req_cu2 = (PCRE2_UCHAR)(re->last_codeunit);
  if ((re->flags & PCRE2_LASTCASELESS) != 0)
    {
    req_cu2 = TABLE_GET(req_cu, mb->fcc, req_cu);
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH != 8
    if (utf && req_cu > 127) req_cu2 = UCD_OTHERCASE(req_cu);
#endif
    }
  }


/* ==========================================================================*/

/* Loop for handling unanchored repeated matching attempts; for anchored regexs
the loop runs just once. */

#ifdef SUPPORT_UNICODE
FRAGMENT_RESTART:
#endif

start_partial = match_partial = NULL;
mb->hitend = FALSE;

for(;;)
  {
  PCRE2_SPTR new_start_match;

  /* ----------------- Start of match optimizations ---------------- */

  /* There are some optimizations that avoid running the match if a known
  starting point is not found, or if a known later code unit is not present.
  However, there is an option (settable at compile time) that disables these,
  for testing and for ensuring that all callouts do actually occur. */

  if ((re->overall_options & PCRE2_NO_START_OPTIMIZE) == 0)
    {
    /* If firstline is TRUE, the start of the match is constrained to the first
    line of a multiline string. That is, the match must be before or at the
    first newline following the start of matching. Temporarily adjust
    end_subject so that we stop the scans for a first code unit at a newline.
    If the match fails at the newline, later code breaks the loop. */

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
            ok = (start_bits[c/8] & (1u << (c&7))) != 0;
            }
          }
        if (!ok)
          {
          rc = MATCH_NOMATCH;
          break;
          }
        }
      }

    /* Not anchored. Advance to a unique first code unit if there is one. In
    8-bit mode, the use of memchr() gives a big speed up, even though we have
    to call it twice in caseless mode, in order to find the earliest occurrence
    of the character in either of its cases. If a call to memchr() that
    searches the rest of the subject fails to find one case, remember that in
    order not to keep on repeating the search. This can make a huge difference
    when the strings are very long and only one case is present. */

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
          PCRE2_SPTR pp1 = NULL;
          PCRE2_SPTR pp2 = NULL;
          PCRE2_SIZE cu2size = end_subject - start_match;

          if (!memchr_not_found_first_cu)
            {
            pp1 = memchr(start_match, first_cu, end_subject - start_match);
            if (pp1 == NULL) memchr_not_found_first_cu = TRUE;
              else cu2size = pp1 - start_match;
            }

          /* If pp1 is not NULL, we have arranged to search only as far as pp1,
          to see if the other case is earlier, so we can set "not found" only
          when both searches have returned NULL. */

          if (!memchr_not_found_first_cu2)
            {
            pp2 = memchr(start_match, first_cu2, cu2size);
            memchr_not_found_first_cu2 = (pp2 == NULL && pp1 == NULL);
            }

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

        if (mb->partial == 0 && start_match >= mb->end_subject)
          {
          rc = MATCH_NOMATCH;
          break;
          }
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
          if ((start_bits[c/8] & (1u << (c&7))) != 0) break;
          start_match++;
          }

        /* See comment above in first_cu checking about the next few lines. */

        if (mb->partial == 0 && start_match >= mb->end_subject)
          {
          rc = MATCH_NOMATCH;
          break;
          }
        }
      }   /* End first code unit handling */

    /* Restore fudged end_subject */

    end_subject = mb->end_subject;

    /* The following two optimizations must be disabled for partial matching. */

    if (mb->partial == 0)
      {
      PCRE2_SPTR p;

      /* The minimum matching length is a lower bound; no string of that length
      may actually match the pattern. Although the value is, strictly, in
      characters, we treat it as code units to avoid spending too much time in
      this optimization. */

      if (end_subject - start_match < re->minlength)
        {
        rc = MATCH_NOMATCH;
        break;
        }

      /* If req_cu is set, we know that that code unit must appear in the
      subject for the (non-partial) match to succeed. If the first code unit is
      set, req_cu must be later in the subject; otherwise the test starts at
      the match point. This optimization can save a huge amount of backtracking
      in patterns with nested unlimited repeats that aren't going to match.
      Writing separate code for caseful/caseless versions makes it go faster,
      as does using an autoincrement and backing off on a match. As in the case
      of the first code unit, using memchr() in the 8-bit library gives a big
      speed up. Unlike the first_cu check above, we do not need to call
      memchr() twice in the caseless case because we only need to check for the
      presence of the character in either case, not find the first occurrence.

      The search can be skipped if the code unit was found later than the
      current starting point in a previous iteration of the bumpalong loop.

      HOWEVER: when the subject string is very, very long, searching to its end
      can take a long time, and give bad performance on quite ordinary
      anchored patterns. This showed up when somebody was matching something
      like /^\d+C/ on a 32-megabyte string... so we don't do this when the
      string is sufficiently long, but it's worth searching a lot more for
      unanchored patterns. */

      p = start_match + (has_first_cu? 1:0);
      if (has_req_cu && p > req_cu_ptr)
        {
        PCRE2_SIZE check_length = end_subject - start_match;

        if (check_length < REQ_CU_MAX ||
              (!anchored && check_length < REQ_CU_MAX * 1000))
          {
          if (req_cu != req_cu2)  /* Caseless */
            {
#if PCRE2_CODE_UNIT_WIDTH != 8
            while (p < end_subject)
              {
              uint32_t pp = UCHAR21INCTEST(p);
              if (pp == req_cu || pp == req_cu2) { p--; break; }
              }
#else  /* 8-bit code units */
            PCRE2_SPTR pp = p;
            p = memchr(pp, req_cu, end_subject - pp);
            if (p == NULL)
              {
              p = memchr(pp, req_cu2, end_subject - pp);
              if (p == NULL) p = end_subject;
              }
#endif /* PCRE2_CODE_UNIT_WIDTH != 8 */
            }

          /* The caseful case */

          else
            {
#if PCRE2_CODE_UNIT_WIDTH != 8
            while (p < end_subject)
              {
              if (UCHAR21INCTEST(p) == req_cu) { p--; break; }
              }

#else  /* 8-bit code units */
            p = memchr(p, req_cu, end_subject - p);
            if (p == NULL) p = end_subject;
#endif
            }

          /* If we can't find the required code unit, break the bumpalong loop,
          forcing a match failure. */

          if (p >= end_subject)
            {
            rc = MATCH_NOMATCH;
            break;
            }

          /* If we have found the required code unit, save the point where we
          found it, so that we don't search again next time round the bumpalong
          loop if the start hasn't yet passed this code unit. */

          req_cu_ptr = p;
          }
        }
      }
    }

  /* ------------ End of start of match optimizations ------------ */

  /* Give no match if we have passed the bumpalong limit. */

  if (start_match > bumpalong_limit)
    {
    rc = MATCH_NOMATCH;
    break;
    }

  /* OK, we can now run the match. If "hitend" is set afterwards, remember the
  first starting point for which a partial match was found. */

  cb.start_match = (PCRE2_SIZE)(start_match - subject);
  cb.callout_flags |= PCRE2_CALLOUT_STARTMATCH;

  mb->start_used_ptr = start_match;
  mb->last_used_ptr = start_match;
#ifdef SUPPORT_UNICODE
  mb->moptions = options | fragment_options;
#else
  mb->moptions = options;
#endif
  mb->match_call_count = 0;
  mb->end_offset_top = 0;
  mb->skip_arg_count = 0;

  rc = match(start_match, mb->start_code, match_data->ovector,
    match_data->oveccount, re->top_bracket, frame_size, mb);

  if (mb->hitend && start_partial == NULL)
    {
    start_partial = mb->start_used_ptr;
    match_partial = start_match;
    }

  switch(rc)
    {
    /* If MATCH_SKIP_ARG reaches this level it means that a MARK that matched
    the SKIP's arg was not found. In this circumstance, Perl ignores the SKIP
    entirely. The only way we can do that is to re-do the match at the same
    point, with a flag to force SKIP with an argument to be ignored. Just
    treating this case as NOMATCH does not work because it does not check other
    alternatives in patterns such as A(*SKIP:A)B|AC when the subject is AC. */

    case MATCH_SKIP_ARG:
    new_start_match = start_match;
    mb->ignore_skip_arg = mb->skip_arg_count;
    break;

    /* SKIP passes back the next starting point explicitly, but if it is no
    greater than the match we have just done, treat it as NOMATCH. */

    case MATCH_SKIP:
    if (mb->verb_skip_ptr > start_match)
      {
      new_start_match = mb->verb_skip_ptr;
      break;
      }
    /* Fall through */

    /* NOMATCH and PRUNE advance by one character. THEN at this level acts
    exactly like PRUNE. Unset ignore SKIP-with-argument. */

    case MATCH_NOMATCH:
    case MATCH_PRUNE:
    case MATCH_THEN:
    mb->ignore_skip_arg = 0;
    new_start_match = start_match + 1;
#ifdef SUPPORT_UNICODE
    if (utf)
      ACROSSCHAR(new_start_match < end_subject, new_start_match,
        new_start_match++);
#endif
    break;

    /* COMMIT disables the bumpalong, but otherwise behaves as NOMATCH. */

    case MATCH_COMMIT:
    rc = MATCH_NOMATCH;
    goto ENDLOOP;

    /* Any other return is either a match, or some kind of error. */

    default:
    goto ENDLOOP;
    }

  /* Control reaches here for the various types of "no match at this point"
  result. Reset the code to MATCH_NOMATCH for subsequent checking. */

  rc = MATCH_NOMATCH;

  /* If PCRE2_FIRSTLINE is set, the match must happen before or at the first
  newline in the subject (though it may continue over the newline). Therefore,
  if we have just failed to match, starting at a newline, do not continue. */

  if (firstline && IS_NEWLINE(start_match)) break;

  /* Advance to new matching position */

  start_match = new_start_match;

  /* Break the loop if the pattern is anchored or if we have passed the end of
  the subject. */

  if (anchored || start_match > end_subject) break;

  /* If we have just passed a CR and we are now at a LF, and the pattern does
  not contain any explicit matches for \r or \n, and the newline option is CRLF
  or ANY or ANYCRLF, advance the match position by one more code unit. In
  normal matching start_match will aways be greater than the first position at
  this stage, but a failed *SKIP can cause a return at the same point, which is
  why the first test exists. */

  if (start_match > subject + start_offset &&
      start_match[-1] == CHAR_CR &&
      start_match < end_subject &&
      *start_match == CHAR_NL &&
      (re->flags & PCRE2_HASCRORLF) == 0 &&
        (mb->nltype == NLTYPE_ANY ||
         mb->nltype == NLTYPE_ANYCRLF ||
         mb->nllen == 2))
    start_match++;

  mb->mark = NULL;   /* Reset for start of next match attempt */
  }                  /* End of for(;;) "bumpalong" loop */

/* ==========================================================================*/

/* When we reach here, one of the following stopping conditions is true:

(1) The match succeeded, either completely, or partially;

(2) The pattern is anchored or the match was failed after (*COMMIT);

(3) We are past the end of the subject or the bumpalong limit;

(4) PCRE2_FIRSTLINE is set and we have failed to match at a newline, because
    this option requests that a match occur at or before the first newline in
    the subject.

(5) Some kind of error occurred.

*/

ENDLOOP:

/* If end_subject != true_end_subject, it means we are handling invalid UTF,
and have just processed a non-terminal fragment. If this resulted in no match
or a partial match we must carry on to the next fragment (a partial match is
returned to the caller only at the very end of the subject). A loop is used to
avoid trying to match against empty fragments; if the pattern can match an
empty string it would have done so already. */

#ifdef SUPPORT_UNICODE
if (utf && end_subject != true_end_subject &&
    (rc == MATCH_NOMATCH || rc == PCRE2_ERROR_PARTIAL))
  {
  for (;;)
    {
    /* Advance past the first bad code unit, and then skip invalid character
    starting code units in 8-bit and 16-bit modes. */

    start_match = end_subject + 1;
#if PCRE2_CODE_UNIT_WIDTH != 32
    while (start_match < true_end_subject && NOT_FIRSTCU(*start_match))
      start_match++;
#endif

    /* If we have hit the end of the subject, there isn't another non-empty
    fragment, so give up. */

    if (start_match >= true_end_subject)
      {
      rc = MATCH_NOMATCH;  /* In case it was partial */
      break;
      }

    /* Check the rest of the subject */

    mb->check_subject = start_match;
    rc = PRIV(valid_utf)(start_match, length - (start_match - subject),
      &(match_data->startchar));

    /* The rest of the subject is valid UTF. */

    if (rc == 0)
      {
      mb->end_subject = end_subject = true_end_subject;
      fragment_options = PCRE2_NOTBOL;
      goto FRAGMENT_RESTART;
      }

    /* A subsequent UTF error has been found; if the next fragment is
    non-empty, set up to process it. Otherwise, let the loop advance. */

    else if (rc < 0)
      {
      mb->end_subject = end_subject = start_match + match_data->startchar;
      if (end_subject > start_match)
        {
        fragment_options = PCRE2_NOTBOL|PCRE2_NOTEOL;
        goto FRAGMENT_RESTART;
        }
      }
    }
  }
#endif  /* SUPPORT_UNICODE */

/* Release an enlarged frame vector that is on the heap. */

if (mb->match_frames != mb->stack_frames)
  mb->memctl.free(mb->match_frames, mb->memctl.memory_data);

/* Fill in fields that are always returned in the match data. */

match_data->code = re;
match_data->mark = mb->mark;
match_data->matchedby = PCRE2_MATCHEDBY_INTERPRETER;

/* Handle a fully successful match. Set the return code to the number of
captured strings, or 0 if there were too many to fit into the ovector, and then
set the remaining returned values before returning. Make a copy of the subject
string if requested. */

if (rc == MATCH_MATCH)
  {
  match_data->rc = ((int)mb->end_offset_top >= 2 * match_data->oveccount)?
    0 : (int)mb->end_offset_top/2 + 1;
  match_data->startchar = start_match - subject;
  match_data->leftchar = mb->start_used_ptr - subject;
  match_data->rightchar = ((mb->last_used_ptr > mb->end_match_ptr)?
    mb->last_used_ptr : mb->end_match_ptr) - subject;
  if ((options & PCRE2_COPY_MATCHED_SUBJECT) != 0)
    {
    length = CU2BYTES(length + was_zero_terminated);
    match_data->subject = match_data->memctl.malloc(length,
      match_data->memctl.memory_data);
    if (match_data->subject == NULL) return PCRE2_ERROR_NOMEMORY;
    memcpy((void *)match_data->subject, subject, length);
    match_data->flags |= PCRE2_MD_COPIED_SUBJECT;
    }
  else match_data->subject = subject;
  return match_data->rc;
  }

/* Control gets here if there has been a partial match, an error, or if the
overall match attempt has failed at all permitted starting positions. Any mark
data is in the nomatch_mark field. */

match_data->mark = mb->nomatch_mark;

/* For anything other than nomatch or partial match, just return the code. */

if (rc != MATCH_NOMATCH && rc != PCRE2_ERROR_PARTIAL) match_data->rc = rc;

/* Handle a partial match. If a "soft" partial match was requested, searching
for a complete match will have continued, and the value of rc at this point
will be MATCH_NOMATCH. For a "hard" partial match, it will already be
PCRE2_ERROR_PARTIAL. */

else if (match_partial != NULL)
  {
  match_data->subject = subject;
  match_data->ovector[0] = match_partial - subject;
  match_data->ovector[1] = end_subject - subject;
  match_data->startchar = match_partial - subject;
  match_data->leftchar = start_partial - subject;
  match_data->rightchar = end_subject - subject;
  match_data->rc = PCRE2_ERROR_PARTIAL;
  }

/* Else this is the classic nomatch case. */

else match_data->rc = PCRE2_ERROR_NOMATCH;

return match_data->rc;
}

/* End of pcre2_match.c */
