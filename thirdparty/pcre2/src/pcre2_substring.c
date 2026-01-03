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



/*************************************************
*   Copy named captured string to given buffer   *
*************************************************/

/* This function copies a single captured substring into a given buffer,
identifying it by name. If the regex permits duplicate names, the first
substring that is set is chosen.

Arguments:
  match_data     points to the match data
  stringname     the name of the required substring
  buffer         where to put the substring
  sizeptr        the size of the buffer, updated to the size of the substring

Returns:         if successful: zero
                 if not successful, a negative error code:
                   (1) an error from nametable_scan()
                   (2) an error from copy_bynumber()
                   (3) PCRE2_ERROR_UNAVAILABLE: no group is in ovector
                   (4) PCRE2_ERROR_UNSET: all named groups in ovector are unset
*/

PCRE2_EXP_DEFN int PCRE2_CALL_CONVENTION
pcre2_substring_copy_byname(pcre2_match_data *match_data, PCRE2_SPTR stringname,
  PCRE2_UCHAR *buffer, PCRE2_SIZE *sizeptr)
{
PCRE2_SPTR first, last, entry;
int failrc, entrysize;
if (match_data->matchedby == PCRE2_MATCHEDBY_DFA_INTERPRETER)
  return PCRE2_ERROR_DFA_UFUNC;
entrysize = pcre2_substring_nametable_scan(match_data->code, stringname,
  &first, &last);
if (entrysize < 0) return entrysize;
failrc = PCRE2_ERROR_UNAVAILABLE;
for (entry = first; entry <= last; entry += entrysize)
  {
  uint32_t n = GET2(entry, 0);
  if (n < match_data->oveccount)
    {
    if (match_data->ovector[n*2] != PCRE2_UNSET)
      return pcre2_substring_copy_bynumber(match_data, n, buffer, sizeptr);
    failrc = PCRE2_ERROR_UNSET;
    }
  }
return failrc;
}



/*************************************************
*  Copy numbered captured string to given buffer *
*************************************************/

/* This function copies a single captured substring into a given buffer,
identifying it by number.

Arguments:
  match_data     points to the match data
  stringnumber   the number of the required substring
  buffer         where to put the substring
  sizeptr        the size of the buffer, updated to the size of the substring

Returns:         if successful: 0
                 if not successful, a negative error code:
                   PCRE2_ERROR_NOMEMORY: buffer too small
                   PCRE2_ERROR_NOSUBSTRING: no such substring
                   PCRE2_ERROR_UNAVAILABLE: ovector too small
                   PCRE2_ERROR_UNSET: substring is not set
*/

PCRE2_EXP_DEFN int PCRE2_CALL_CONVENTION
pcre2_substring_copy_bynumber(pcre2_match_data *match_data,
  uint32_t stringnumber, PCRE2_UCHAR *buffer, PCRE2_SIZE *sizeptr)
{
int rc;
PCRE2_SIZE size;
rc = pcre2_substring_length_bynumber(match_data, stringnumber, &size);
if (rc < 0) return rc;
if (size + 1 > *sizeptr) return PCRE2_ERROR_NOMEMORY;
if (size != 0) memcpy(buffer, match_data->subject + match_data->ovector[stringnumber*2],
  CU2BYTES(size));
buffer[size] = 0;
*sizeptr = size;
return 0;
}



/*************************************************
*          Extract named captured string         *
*************************************************/

/* This function copies a single captured substring, identified by name, into
new memory. If the regex permits duplicate names, the first substring that is
set is chosen.

Arguments:
  match_data     pointer to match_data
  stringname     the name of the required substring
  stringptr      where to put the pointer to the new memory
  sizeptr        where to put the length of the substring

Returns:         if successful: zero
                 if not successful, a negative value:
                   (1) an error from nametable_scan()
                   (2) an error from get_bynumber()
                   (3) PCRE2_ERROR_UNAVAILABLE: no group is in ovector
                   (4) PCRE2_ERROR_UNSET: all named groups in ovector are unset
*/

PCRE2_EXP_DEFN int PCRE2_CALL_CONVENTION
pcre2_substring_get_byname(pcre2_match_data *match_data,
  PCRE2_SPTR stringname, PCRE2_UCHAR **stringptr, PCRE2_SIZE *sizeptr)
{
PCRE2_SPTR first, last, entry;
int failrc, entrysize;
if (match_data->matchedby == PCRE2_MATCHEDBY_DFA_INTERPRETER)
  return PCRE2_ERROR_DFA_UFUNC;
entrysize = pcre2_substring_nametable_scan(match_data->code, stringname,
  &first, &last);
if (entrysize < 0) return entrysize;
failrc = PCRE2_ERROR_UNAVAILABLE;
for (entry = first; entry <= last; entry += entrysize)
  {
  uint32_t n = GET2(entry, 0);
  if (n < match_data->oveccount)
    {
    if (match_data->ovector[n*2] != PCRE2_UNSET)
      return pcre2_substring_get_bynumber(match_data, n, stringptr, sizeptr);
    failrc = PCRE2_ERROR_UNSET;
    }
  }
return failrc;
}



/*************************************************
*      Extract captured string to new memory     *
*************************************************/

/* This function copies a single captured substring into a piece of new
memory.

Arguments:
  match_data     points to match data
  stringnumber   the number of the required substring
  stringptr      where to put a pointer to the new memory
  sizeptr        where to put the size of the substring

Returns:         if successful: 0
                 if not successful, a negative error code:
                   PCRE2_ERROR_NOMEMORY: failed to get memory
                   PCRE2_ERROR_NOSUBSTRING: no such substring
                   PCRE2_ERROR_UNAVAILABLE: ovector too small
                   PCRE2_ERROR_UNSET: substring is not set
*/

PCRE2_EXP_DEFN int PCRE2_CALL_CONVENTION
pcre2_substring_get_bynumber(pcre2_match_data *match_data,
  uint32_t stringnumber, PCRE2_UCHAR **stringptr, PCRE2_SIZE *sizeptr)
{
int rc;
PCRE2_SIZE size;
PCRE2_UCHAR *yield;
rc = pcre2_substring_length_bynumber(match_data, stringnumber, &size);
if (rc < 0) return rc;
yield = PRIV(memctl_malloc)(sizeof(pcre2_memctl) +
  (size + 1)*PCRE2_CODE_UNIT_WIDTH, (pcre2_memctl *)match_data);
if (yield == NULL) return PCRE2_ERROR_NOMEMORY;
yield = (PCRE2_UCHAR *)(((char *)yield) + sizeof(pcre2_memctl));
if (size != 0) memcpy(yield, match_data->subject + match_data->ovector[stringnumber*2],
  CU2BYTES(size));
yield[size] = 0;
*stringptr = yield;
*sizeptr = size;
return 0;
}



/*************************************************
*       Free memory obtained by get_substring    *
*************************************************/

/*
Argument:     the result of a previous pcre2_substring_get_byxxx()
Returns:      nothing
*/

PCRE2_EXP_DEFN void PCRE2_CALL_CONVENTION
pcre2_substring_free(PCRE2_UCHAR *string)
{
if (string != NULL)
  {
  pcre2_memctl *memctl = (pcre2_memctl *)((char *)string - sizeof(pcre2_memctl));
  memctl->free(memctl, memctl->memory_data);
  }
}



/*************************************************
*         Get length of a named substring        *
*************************************************/

/* This function returns the length of a named captured substring. If the regex
permits duplicate names, the first substring that is set is chosen.

Arguments:
  match_data      pointer to match data
  stringname      the name of the required substring
  sizeptr         where to put the length, if not NULL

Returns:          0 if successful, else a negative error number
*/

PCRE2_EXP_DEFN int PCRE2_CALL_CONVENTION
pcre2_substring_length_byname(pcre2_match_data *match_data,
  PCRE2_SPTR stringname, PCRE2_SIZE *sizeptr)
{
PCRE2_SPTR first, last, entry;
int failrc, entrysize;
if (match_data->matchedby == PCRE2_MATCHEDBY_DFA_INTERPRETER)
  return PCRE2_ERROR_DFA_UFUNC;
entrysize = pcre2_substring_nametable_scan(match_data->code, stringname,
  &first, &last);
if (entrysize < 0) return entrysize;
failrc = PCRE2_ERROR_UNAVAILABLE;
for (entry = first; entry <= last; entry += entrysize)
  {
  uint32_t n = GET2(entry, 0);
  if (n < match_data->oveccount)
    {
    if (match_data->ovector[n*2] != PCRE2_UNSET)
      return pcre2_substring_length_bynumber(match_data, n, sizeptr);
    failrc = PCRE2_ERROR_UNSET;
    }
  }
return failrc;
}



/*************************************************
*        Get length of a numbered substring      *
*************************************************/

/* This function returns the length of a captured substring. If the start is
beyond the end (which can happen when \K is used in an assertion), it sets the
length to zero.

Arguments:
  match_data      pointer to match data
  stringnumber    the number of the required substring
  sizeptr         where to put the length, if not NULL

Returns:         if successful: 0
                 if not successful, a negative error code:
                   PCRE2_ERROR_NOSUBSTRING: no such substring
                   PCRE2_ERROR_UNAVAILABLE: ovector is too small
                   PCRE2_ERROR_UNSET: substring is not set
                   PCRE2_ERROR_INVALIDOFFSET: internal error, should not occur
*/

PCRE2_EXP_DEFN int PCRE2_CALL_CONVENTION
pcre2_substring_length_bynumber(pcre2_match_data *match_data,
  uint32_t stringnumber, PCRE2_SIZE *sizeptr)
{
PCRE2_SIZE left, right;
int count = match_data->rc;
if (count == PCRE2_ERROR_PARTIAL)
  {
  if (stringnumber > 0) return PCRE2_ERROR_PARTIAL;
  count = 0;
  }
else if (count < 0) return count;            /* Match failed */

if (match_data->matchedby != PCRE2_MATCHEDBY_DFA_INTERPRETER)
  {
  if (stringnumber > match_data->code->top_bracket)
    return PCRE2_ERROR_NOSUBSTRING;
  if (stringnumber >= match_data->oveccount)
    return PCRE2_ERROR_UNAVAILABLE;
  if (match_data->ovector[stringnumber*2] == PCRE2_UNSET)
    return PCRE2_ERROR_UNSET;
  }
else  /* Matched using pcre2_dfa_match() */
  {
  if (stringnumber >= match_data->oveccount) return PCRE2_ERROR_UNAVAILABLE;
  if (count != 0 && stringnumber >= (uint32_t)count) return PCRE2_ERROR_UNSET;
  }

left = match_data->ovector[stringnumber*2];
right = match_data->ovector[stringnumber*2+1];
/* LCOV_EXCL_START - this appears to be unreachable, as the ovector and
subject_length should always be set consistently, no matter what misbehaviour
the caller has committed. */
if (left > match_data->subject_length || right > match_data->subject_length)
  {
  PCRE2_DEBUG_UNREACHABLE();
  return PCRE2_ERROR_INVALIDOFFSET;
  }
/* LCOV_EXCL_STOP */
if (sizeptr != NULL) *sizeptr = (left > right)? 0 : right - left;
return 0;
}



/*************************************************
*    Extract all captured strings to new memory  *
*************************************************/

/* This function gets one chunk of memory and builds a list of pointers and all
the captured substrings in it. A NULL pointer is put on the end of the list.
The substrings are zero-terminated, but also, if the final argument is
non-NULL, a list of lengths is also returned. This allows binary data to be
handled.

Arguments:
  match_data     points to the match data
  listptr        set to point to the list of pointers
  lengthsptr     set to point to the list of lengths (may be NULL)

Returns:         if successful: 0
                 if not successful, a negative error code:
                   PCRE2_ERROR_NOMEMORY: failed to get memory,
                   or a match failure code
*/

PCRE2_EXP_DEFN int PCRE2_CALL_CONVENTION
pcre2_substring_list_get(pcre2_match_data *match_data, PCRE2_UCHAR ***listptr,
  PCRE2_SIZE **lengthsptr)
{
int i, count, count2;
PCRE2_SIZE size;
PCRE2_SIZE *lensp;
pcre2_memctl *memp;
PCRE2_UCHAR **listp;
PCRE2_UCHAR *sp;
PCRE2_SIZE *ovector;

if ((count = match_data->rc) < 0) return count;   /* Match failed */
if (count == 0) count = match_data->oveccount;    /* Ovector too small */

count2 = 2*count;
ovector = match_data->ovector;
size = sizeof(pcre2_memctl) + sizeof(PCRE2_UCHAR *);      /* For final NULL */
if (lengthsptr != NULL) size += sizeof(PCRE2_SIZE) * count;  /* For lengths */

for (i = 0; i < count2; i += 2)
  {
  size += sizeof(PCRE2_UCHAR *) + CU2BYTES(1);
  if (ovector[i+1] > ovector[i]) size += CU2BYTES(ovector[i+1] - ovector[i]);
  }

memp = PRIV(memctl_malloc)(size, (pcre2_memctl *)match_data);
if (memp == NULL) return PCRE2_ERROR_NOMEMORY;

*listptr = listp = (PCRE2_UCHAR **)((char *)memp + sizeof(pcre2_memctl));
lensp = (PCRE2_SIZE *)((char *)listp + sizeof(PCRE2_UCHAR *) * (count + 1));

if (lengthsptr == NULL)
  {
  sp = (PCRE2_UCHAR *)lensp;
  lensp = NULL;
  }
else
  {
  *lengthsptr = lensp;
  sp = (PCRE2_UCHAR *)((char *)lensp + sizeof(PCRE2_SIZE) * count);
  }

for (i = 0; i < count2; i += 2)
  {
  size = (ovector[i+1] > ovector[i])? (ovector[i+1] - ovector[i]) : 0;

  /* Size == 0 includes the case when the capture is unset. Avoid adding
  PCRE2_UNSET to match_data->subject because it overflows, even though with
  zero size calling memcpy() is harmless. */

  if (size != 0) memcpy(sp, match_data->subject + ovector[i], CU2BYTES(size));
  *listp++ = sp;
  if (lensp != NULL) *lensp++ = size;
  sp += size;
  *sp++ = 0;
  }

*listp = NULL;
return 0;
}



/*************************************************
*   Free memory obtained by substring_list_get   *
*************************************************/

/*
Argument:     the result of a previous pcre2_substring_list_get()
Returns:      nothing
*/

PCRE2_EXP_DEFN void PCRE2_CALL_CONVENTION
pcre2_substring_list_free(PCRE2_UCHAR **list)
{
if (list != NULL)
  {
  pcre2_memctl *memctl = (pcre2_memctl *)((char *)list - sizeof(pcre2_memctl));
  memctl->free(memctl, memctl->memory_data);
  }
}



/*************************************************
*     Find (multiple) entries for named string   *
*************************************************/

/* This function scans the nametable for a given name, using binary chop. It
returns either two pointers to the entries in the table, or, if no pointers are
given, the number of a unique group with the given name. If duplicate names are
permitted, and the name is not unique, an error is generated.

Arguments:
  code        the compiled regex
  stringname  the name whose entries required
  firstptr    where to put the pointer to the first entry
  lastptr     where to put the pointer to the last entry

Returns:      PCRE2_ERROR_NOSUBSTRING if the name is not found
              otherwise, if firstptr and lastptr are NULL:
                a group number for a unique substring
                else PCRE2_ERROR_NOUNIQUESUBSTRING
              otherwise:
                the length of each entry, having set firstptr and lastptr
*/

PCRE2_EXP_DEFN int PCRE2_CALL_CONVENTION
pcre2_substring_nametable_scan(const pcre2_code *code, PCRE2_SPTR stringname,
  PCRE2_SPTR *firstptr, PCRE2_SPTR *lastptr)
{
uint16_t bot = 0;
uint16_t top = code->name_count;
uint16_t entrysize = code->name_entry_size;
PCRE2_SPTR nametable = (PCRE2_SPTR)((const char *)code + sizeof(pcre2_real_code));

while (top > bot)
  {
  uint16_t mid = (top + bot) / 2;
  PCRE2_SPTR entry = nametable + entrysize*mid;
  int c = PRIV(strcmp)(stringname, entry + IMM2_SIZE);
  if (c == 0)
    {
    PCRE2_SPTR first;
    PCRE2_SPTR last;
    PCRE2_SPTR lastentry;
    lastentry = nametable + entrysize * (code->name_count - 1);
    first = last = entry;
    while (first > nametable)
      {
      if (PRIV(strcmp)(stringname, (first - entrysize + IMM2_SIZE)) != 0) break;
      first -= entrysize;
      }
    while (last < lastentry)
      {
      if (PRIV(strcmp)(stringname, (last + entrysize + IMM2_SIZE)) != 0) break;
      last += entrysize;
      }
    if (firstptr == NULL) return (first == last)?
      (int)GET2(entry, 0) : PCRE2_ERROR_NOUNIQUESUBSTRING;
    *firstptr = first;
    *lastptr = last;
    return entrysize;
    }
  if (c > 0) bot = mid + 1; else top = mid;
  }

return PCRE2_ERROR_NOSUBSTRING;
}


/*************************************************
*           Find number for named string         *
*************************************************/

/* This function is a convenience wrapper for pcre2_substring_nametable_scan()
when it is known that names are unique. If there are duplicate names, it is not
defined which number is returned.

Arguments:
  code        the compiled regex
  stringname  the name whose number is required

Returns:      the number of the named parenthesis, or a negative number
                PCRE2_ERROR_NOSUBSTRING if not found
                PCRE2_ERROR_NOUNIQUESUBSTRING if not unique
*/

PCRE2_EXP_DEFN int PCRE2_CALL_CONVENTION
pcre2_substring_number_from_name(const pcre2_code *code,
  PCRE2_SPTR stringname)
{
return pcre2_substring_nametable_scan(code, stringname, NULL, NULL);
}

/* End of pcre2_substring.c */
