/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2025 Sam Lantinga <slouken@libsdl.org>

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/
#include "SDL_internal.h"

// SDL3 always uses its own internal qsort implementation, below, so
// it can guarantee stable sorts across platforms and not have to
// tapdance to support the various qsort_r interfaces, or bridge from
// the C runtime's non-SDLCALL compare functions.

#ifdef assert
#undef assert
#endif
#define assert SDL_assert
#ifdef malloc
#undef malloc
#endif
#define malloc SDL_malloc
#ifdef free
#undef free
#endif
#define free SDL_free
#ifdef memcpy
#undef memcpy
#endif
#define memcpy SDL_memcpy
#ifdef memmove
#undef memmove
#endif
#define memmove SDL_memmove

/*
This code came from Gareth McCaughan, under the zlib license.
Specifically this: https://www.mccaughan.org.uk/software/qsort.c-1.16

Everything below this comment until the HAVE_QSORT #endif was from Gareth
(any minor changes will be noted inline).

Thank you to Gareth for relicensing this code under the zlib license for our
benefit!

Update for SDL3: we have modified this from a qsort function to qsort_r.

--ryan.
*/

/* This is a drop-in replacement for the C library's |qsort()| routine.
 *
 * It is intended for use where you know or suspect that your
 * platform's qsort is bad. If that isn't the case, then you
 * should probably use the qsort your system gives you in preference
 * to mine -- it will likely have been tested and tuned better.
 *
 * Features:
 *   - Median-of-three pivoting (and more)
 *   - Truncation and final polishing by a single insertion sort
 *   - Early truncation when no swaps needed in pivoting step
 *   - Explicit recursion, guaranteed not to overflow
 *   - A few little wrinkles stolen from the GNU |qsort()|.
 *     (For the avoidance of doubt, no code was stolen, only
 *     broad ideas.)
 *   - separate code for non-aligned / aligned / word-size objects
 *
 * Earlier releases of this code used an idiosyncratic licence
 * I wrote myself, because I'm an idiot. The code is now released
 * under the "zlib/libpng licence"; you will find the actual
 * terms in the next comment. I request (but do not require)
 * that if you make any changes beyond the name of the exported
 * routine and reasonable tweaks to the TRUNC_* and
 * PIVOT_THRESHOLD values, you modify the _ID string so as
 * to make it clear that you have changed the code.
 *
 * If you find problems with this code, or find ways of
 * making it significantly faster, please let me know!
 * My e-mail address, valid as of early 2016 and for the
 * foreseeable future, is
 *    gareth.mccaughan@pobox.com
 * Thanks!
 *
 * Gareth McCaughan
 */

/* Copyright (c) 1998-2021 Gareth McCaughan
 *
 * This software is provided 'as-is', without any express or implied
 * warranty. In no event will the authors be held liable for any
 * damages arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented;
 *    you must not claim that you wrote the original software.
 *    If you use this software in a product, an acknowledgment
 *    in the product documentation would be appreciated but
 *    is not required.
 *
 * 2. Altered source versions must be plainly marked as such,
 *    and must not be misrepresented as being the original software.
 *
 * 3. This notice may not be removed or altered from any source
 *    distribution.
 */

/* Revision history since release:
 *   1998-03-19 v1.12 First release I have any records of.
 *   2007-09-02 v1.13 Fix bug kindly reported by Dan Bodoh
 *                    (premature termination of recursion).
 *                    Add a few clarifying comments.
 *                    Minor improvements to debug output.
 *   2016-02-21 v1.14 Replace licence with 2-clause BSD,
 *                    and clarify a couple of things in
 *                    comments. No code changes.
 *   2016-03-10 v1.15 Fix bug kindly reported by Ryan Gordon
 *                    (pre-insertion-sort messed up).
 *                    Disable DEBUG_QSORT by default.
 *                    Tweak comments very slightly.
 *   2021-02-20 v1.16 Fix bug kindly reported by Ray Gardner
 *                    (error in recursion leading to possible
 *                    stack overflow).
 *                    When checking alignment, avoid casting
 *                    pointer to possibly-smaller integer.
 */

/* BEGIN SDL CHANGE ... commented this out with an #if 0 block. --ryan. */
#if 0
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#undef DEBUG_QSORT

static char _ID[]="<qsort.c gjm WITH CHANGES FOR SDL3 1.16 2021-02-20>";
#endif
/* END SDL CHANGE ... commented this out with an #if 0 block. --ryan. */

/* How many bytes are there per word? (Must be a power of 2,
 * and must in fact equal sizeof(int).)
 */
#define WORD_BYTES sizeof(int)

/* How big does our stack need to be? Answer: one entry per
 * bit in a |size_t|. (Actually, a bit less because we don't
 * recurse all the way down to size-1 subarrays.)
 */
#define STACK_SIZE (8*sizeof(size_t))

/* Different situations have slightly different requirements,
 * and we make life epsilon easier by using different truncation
 * points for the three different cases.
 * So far, I have tuned TRUNC_words and guessed that the same
 * value might work well for the other two cases. Of course
 * what works well on my machine might work badly on yours.
 */
#define TRUNC_nonaligned	12
#define TRUNC_aligned		12
#define TRUNC_words		12*WORD_BYTES	/* nb different meaning */

/* We use a simple pivoting algorithm for shortish sub-arrays
 * and a more complicated one for larger ones. The threshold
 * is PIVOT_THRESHOLD.
 */
#define PIVOT_THRESHOLD 40

typedef struct { char * first; char * last; } stack_entry;
#define pushLeft {stack[stacktop].first=ffirst;stack[stacktop++].last=last;}
#define pushRight {stack[stacktop].first=first;stack[stacktop++].last=llast;}
#define doLeft {first=ffirst;llast=last;continue;}
#define doRight {ffirst=first;last=llast;continue;}
#define pop {if (--stacktop<0) break;\
  first=ffirst=stack[stacktop].first;\
  last=llast=stack[stacktop].last;\
  continue;}

/* Some comments on the implementation.
 * 1. When we finish partitioning the array into "low"
 *    and "high", we forget entirely about short subarrays,
 *    because they'll be done later by insertion sort.
 *    Doing lots of little insertion sorts might be a win
 *    on large datasets for locality-of-reference reasons,
 *    but it makes the code much nastier and increases
 *    bookkeeping overhead.
 * 2. We always save the longer and get to work on the
 *    shorter. This guarantees that whenever we push
 *    a k'th entry onto the stack we are about to get
 *    working on something of size <= N/2^k where N is
 *    the original array size; so the stack can't need
 *    more than log_2(max-array-size) entries.
 * 3. We choose a pivot by looking at the first, last
 *    and middle elements. We arrange them into order
 *    because it's easy to do that in conjunction with
 *    choosing the pivot, and it makes things a little
 *    easier in the partitioning step. Anyway, the pivot
 *    is the middle of these three. It's still possible
 *    to construct datasets where the algorithm takes
 *    time of order n^2, but it simply never happens in
 *    practice.
 * 3' Newsflash: On further investigation I find that
 *    it's easy to construct datasets where median-of-3
 *    simply isn't good enough. So on large-ish subarrays
 *    we do a more sophisticated pivoting: we take three
 *    sets of 3 elements, find their medians, and then
 *    take the median of those.
 * 4. We copy the pivot element to a separate place
 *    because that way we can always do our comparisons
 *    directly against a pointer to that separate place,
 *    and don't have to wonder "did we move the pivot
 *    element?". This makes the inner loop better.
 * 5. It's possible to make the pivoting even more
 *    reliable by looking at more candidates when n
 *    is larger. (Taking this to its logical conclusion
 *    results in a variant of quicksort that doesn't
 *    have that n^2 worst case.) However, the overhead
 *    from the extra bookkeeping means that it's just
 *    not worth while.
 * 6. This is pretty clean and portable code. Here are
 *    all the potential portability pitfalls and problems
 *    I know of:
 *      - In one place (the insertion sort) I construct
 *        a pointer that points just past the end of the
 *        supplied array, and assume that (a) it won't
 *        compare equal to any pointer within the array,
 *        and (b) it will compare equal to a pointer
 *        obtained by stepping off the end of the array.
 *        These might fail on some segmented architectures.
 *      - I assume that there are 8 bits in a |char| when
 *        computing the size of stack needed. This would
 *        fail on machines with 9-bit or 16-bit bytes.
 *      - I assume that if |((int)base&(sizeof(int)-1))==0|
 *        and |(size&(sizeof(int)-1))==0| then it's safe to
 *        get at array elements via |int*|s, and that if
 *        actually |size==sizeof(int)| as well then it's
 *        safe to treat the elements as |int|s. This might
 *        fail on systems that convert pointers to integers
 *        in non-standard ways.
 *      - I assume that |8*sizeof(size_t)<=INT_MAX|. This
 *        would be false on a machine with 8-bit |char|s,
 *        16-bit |int|s and 4096-bit |size_t|s. :-)
 */

/* The recursion logic is the same in each case.
 * We keep chopping up until we reach subarrays of size
 * strictly less than Trunc; we leave these unsorted. */
#define Recurse(Trunc)				\
      { size_t l=last-ffirst,r=llast-first;	\
        if (l<Trunc) {				\
          if (r>=Trunc) doRight			\
          else pop				\
        }					\
        else if (l<=r) { pushRight; doLeft }	\
        else if (r>=Trunc) { pushLeft; doRight }\
        else doLeft				\
      }

/* and so is the pivoting logic (note: last is inclusive): */
#define Pivot(swapper,sz)			\
  if ((size_t)(last-first)>PIVOT_THRESHOLD*sz) mid=pivot_big(first,mid,last,sz,compare,userdata);\
  else {	\
    if (compare(userdata,first,mid)<0) {			\
      if (compare(userdata,mid,last)>0) {		\
        swapper(mid,last);			\
        if (compare(userdata,first,mid)>0) swapper(first,mid);\
      }						\
    }						\
    else {					\
      if (compare(userdata,mid,last)>0) swapper(first,last)\
      else {					\
        swapper(first,mid);			\
        if (compare(userdata,mid,last)>0) swapper(mid,last);\
      }						\
    }						\
    first+=sz; last-=sz;			\
  }

#ifdef DEBUG_QSORT
#include <stdio.h>
#endif

/* and so is the partitioning logic: */
#define Partition(swapper,sz) {			\
  do {						\
    while (compare(userdata,first,pivot)<0) first+=sz;	\
    while (compare(userdata,pivot,last)<0) last-=sz;	\
    if (first<last) {				\
      swapper(first,last);			\
      first+=sz; last-=sz; }			\
    else if (first==last) { first+=sz; last-=sz; break; }\
  } while (first<=last);			\
}

/* and so is the pre-insertion-sort operation of putting
 * the smallest element into place as a sentinel.
 * Doing this makes the inner loop nicer. I got this
 * idea from the GNU implementation of qsort().
 * We find the smallest element from the first |nmemb|,
 * or the first |limit|, whichever is smaller;
 * therefore we must have ensured that the globally smallest
 * element is in the first |limit| (because our
 * quicksort recursion bottoms out only once we
 * reach subarrays smaller than |limit|).
 */
#define PreInsertion(swapper,limit,sz)		\
  first=base;					\
  last=first + ((nmemb>limit ? limit : nmemb)-1)*sz;\
  while (last!=base) {				\
    if (compare(userdata,first,last)>0) first=last;	\
    last-=sz; }					\
  if (first!=base) swapper(first,(char*)base);

/* and so is the insertion sort, in the first two cases: */
#define Insertion(swapper)			\
  last=((char*)base)+nmemb*size;		\
  for (first=((char*)base)+size;first!=last;first+=size) {	\
    char *test;					\
    /* Find the right place for |first|.	\
     * My apologies for var reuse. */		\
    for (test=first-size;compare(userdata,test,first)>0;test-=size) ;	\
    test+=size;					\
    if (test!=first) {				\
      /* Shift everything in [test,first)	\
       * up by one, and place |first|		\
       * where |test| is. */			\
      memcpy(pivot,first,size);			\
      memmove(test+size,test,first-test);	\
      memcpy(test,pivot,size);			\
    }						\
  }

#define SWAP_nonaligned(a,b) { \
  register char *aa=(a),*bb=(b); \
  register size_t sz=size; \
  do { register char t=*aa; *aa++=*bb; *bb++=t; } while (--sz); }

#define SWAP_aligned(a,b) { \
  register int *aa=(int*)(a),*bb=(int*)(b); \
  register size_t sz=size; \
  do { register int t=*aa;*aa++=*bb; *bb++=t; } while (sz-=WORD_BYTES); }

#define SWAP_words(a,b) { \
  register int t=*((int*)a); *((int*)a)=*((int*)b); *((int*)b)=t; }

/* ---------------------------------------------------------------------- */

static char * pivot_big(char *first, char *mid, char *last, size_t size,
                        int (SDLCALL *compare)(void *, const void *, const void *), void *userdata) {
  size_t d=(((last-first)/size)>>3)*size;
#ifdef DEBUG_QSORT
fprintf(stderr, "pivot_big: first=%p last=%p size=%lu n=%lu\n", first, (unsigned long)last, size, (unsigned long)((last-first+1)/size));
#endif
  char *m1,*m2,*m3;
  { char *a=first, *b=first+d, *c=first+2*d;
#ifdef DEBUG_QSORT
fprintf(stderr,"< %d %d %d @ %p %p %p\n",*(int*)a,*(int*)b,*(int*)c, a,b,c);
#endif
    m1 = compare(userdata,a,b)<0 ?
           (compare(userdata,b,c)<0 ? b : (compare(userdata,a,c)<0 ? c : a))
         : (compare(userdata,a,c)<0 ? a : (compare(userdata,b,c)<0 ? c : b));
  }
  { char *a=mid-d, *b=mid, *c=mid+d;
#ifdef DEBUG_QSORT
fprintf(stderr,". %d %d %d @ %p %p %p\n",*(int*)a,*(int*)b,*(int*)c, a,b,c);
#endif
    m2 = compare(userdata,a,b)<0 ?
           (compare(userdata,b,c)<0 ? b : (compare(userdata,a,c)<0 ? c : a))
         : (compare(userdata,a,c)<0 ? a : (compare(userdata,b,c)<0 ? c : b));
  }
  { char *a=last-2*d, *b=last-d, *c=last;
#ifdef DEBUG_QSORT
fprintf(stderr,"> %d %d %d @ %p %p %p\n",*(int*)a,*(int*)b,*(int*)c, a,b,c);
#endif
    m3 = compare(userdata,a,b)<0 ?
           (compare(userdata,b,c)<0 ? b : (compare(userdata,a,c)<0 ? c : a))
         : (compare(userdata,a,c)<0 ? a : (compare(userdata,b,c)<0 ? c : b));
  }
#ifdef DEBUG_QSORT
fprintf(stderr,"-> %d %d %d @ %p %p %p\n",*(int*)m1,*(int*)m2,*(int*)m3, m1,m2,m3);
#endif
  return compare(userdata,m1,m2)<0 ?
           (compare(userdata,m2,m3)<0 ? m2 : (compare(userdata,m1,m3)<0 ? m3 : m1))
         : (compare(userdata,m1,m3)<0 ? m1 : (compare(userdata,m2,m3)<0 ? m3 : m2));
}

/* ---------------------------------------------------------------------- */

static void qsort_r_nonaligned(void *base, size_t nmemb, size_t size,
           int (SDLCALL *compare)(void *, const void *, const void *), void *userdata) {

  stack_entry stack[STACK_SIZE];
  int stacktop=0;
  char *first,*last;
  char *pivot=malloc(size);
  size_t trunc=TRUNC_nonaligned*size;
  assert(pivot != NULL);

  first=(char*)base; last=first+(nmemb-1)*size;

  if ((size_t)(last-first)>=trunc) {
    char *ffirst=first, *llast=last;
    while (1) {
      /* Select pivot */
      { char * mid=first+size*((last-first)/size >> 1);
        Pivot(SWAP_nonaligned,size);
        memcpy(pivot,mid,size);
      }
      /* Partition. */
      Partition(SWAP_nonaligned,size);
      /* Prepare to recurse/iterate. */
      Recurse(trunc)
    }
  }
  PreInsertion(SWAP_nonaligned,TRUNC_nonaligned,size);
  Insertion(SWAP_nonaligned);
  free(pivot);
}

static void qsort_r_aligned(void *base, size_t nmemb, size_t size,
           int (SDLCALL *compare)(void *,const void *, const void *), void *userdata) {

  stack_entry stack[STACK_SIZE];
  int stacktop=0;
  char *first,*last;
  char *pivot=malloc(size);
  size_t trunc=TRUNC_aligned*size;
  assert(pivot != NULL);

  first=(char*)base; last=first+(nmemb-1)*size;

  if ((size_t)(last-first)>=trunc) {
    char *ffirst=first,*llast=last;
    while (1) {
      /* Select pivot */
      { char * mid=first+size*((last-first)/size >> 1);
        Pivot(SWAP_aligned,size);
        memcpy(pivot,mid,size);
      }
      /* Partition. */
      Partition(SWAP_aligned,size);
      /* Prepare to recurse/iterate. */
      Recurse(trunc)
    }
  }
  PreInsertion(SWAP_aligned,TRUNC_aligned,size);
  Insertion(SWAP_aligned);
  free(pivot);
}

static void qsort_r_words(void *base, size_t nmemb,
           int (SDLCALL *compare)(void *,const void *, const void *), void *userdata) {

  stack_entry stack[STACK_SIZE];
  int stacktop=0;
  char *first,*last;
  char *pivot=malloc(WORD_BYTES);
  assert(pivot != NULL);

  first=(char*)base; last=first+(nmemb-1)*WORD_BYTES;

  if (last-first>=TRUNC_words) {
    char *ffirst=first, *llast=last;
    while (1) {
#ifdef DEBUG_QSORT
fprintf(stderr,"Doing %d:%d: ",
        (first-(char*)base)/WORD_BYTES,
        (last-(char*)base)/WORD_BYTES);
#endif
      /* Select pivot */
      { char * mid=first+WORD_BYTES*((last-first) / (2*WORD_BYTES));
        Pivot(SWAP_words,WORD_BYTES);
        *(int*)pivot=*(int*)mid;
#ifdef DEBUG_QSORT
fprintf(stderr,"pivot = %p = #%lu = %d\n", mid, (unsigned long)(((int*)mid)-((int*)base)), *(int*)mid);
#endif
      }
      /* Partition. */
      Partition(SWAP_words,WORD_BYTES);
#ifdef DEBUG_QSORT
fprintf(stderr, "after partitioning first=#%lu last=#%lu\n", (first-(char*)base)/4lu, (last-(char*)base)/4lu);
#endif
      /* Prepare to recurse/iterate. */
      Recurse(TRUNC_words)
    }
  }
  PreInsertion(SWAP_words,TRUNC_words/WORD_BYTES,WORD_BYTES);
  /* Now do insertion sort. */
  last=((char*)base)+nmemb*WORD_BYTES;
  for (first=((char*)base)+WORD_BYTES;first!=last;first+=WORD_BYTES) {
    /* Find the right place for |first|. My apologies for var reuse */
    int *pl=(int*)(first-WORD_BYTES),*pr=(int*)first;
    *(int*)pivot=*(int*)first;
    for (;compare(userdata,pl,pivot)>0;pr=pl,--pl) {
      *pr=*pl; }
    if (pr!=(int*)first) *pr=*(int*)pivot;
  }
  free(pivot);
}

/* ---------------------------------------------------------------------- */

void SDL_qsort_r(void *base, size_t nmemb, size_t size,
           SDL_CompareCallback_r compare, void *userdata) {

  if (nmemb<=1) return;
  if (((uintptr_t)base|size)&(WORD_BYTES-1))
    qsort_r_nonaligned(base,nmemb,size,compare,userdata);
  else if (size!=WORD_BYTES)
    qsort_r_aligned(base,nmemb,size,compare,userdata);
  else
    qsort_r_words(base,nmemb,compare,userdata);
}

static int SDLCALL qsort_non_r_bridge(void *userdata, const void *a, const void *b)
{
    int (SDLCALL *compare)(const void *, const void *) = (int (SDLCALL *)(const void *, const void *)) userdata;
    return compare(a, b);
}

void SDL_qsort(void *base, size_t nmemb, size_t size, SDL_CompareCallback compare)
{
    SDL_qsort_r(base, nmemb, size, qsort_non_r_bridge, compare);
}

// Don't use the C runtime for such a simple function, since we want to allow SDLCALL callbacks and userdata.
// SDL's replacement: Taken from the Public Domain C Library (PDCLib):
// Permission is granted to use, modify, and / or redistribute at will.
void *SDL_bsearch_r(const void *key, const void *base, size_t nmemb, size_t size, SDL_CompareCallback_r compare, void *userdata)
{
    const void *pivot;
    size_t corr;
    int rc;

    while (nmemb) {
        /* algorithm needs -1 correction if remaining elements are an even number. */
        corr = nmemb % 2;
        nmemb /= 2;
        pivot = (const char *)base + (nmemb * size);
        rc = compare(userdata, key, pivot);

        if (rc > 0) {
            base = (const char *)pivot + size;
            /* applying correction */
            nmemb -= (1 - corr);
        } else if (rc == 0) {
            return (void *)pivot;
        }
    }

    return NULL;
}

void *SDL_bsearch(const void *key, const void *base, size_t nmemb, size_t size, SDL_CompareCallback compare)
{
    // qsort_non_r_bridge just happens to match calling conventions, so reuse it.
    return SDL_bsearch_r(key, base, nmemb, size, qsort_non_r_bridge, compare);
}

