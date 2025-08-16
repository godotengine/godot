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


/*************************************************
*        Return info about compiled pattern      *
*************************************************/

/*
Arguments:
  code          points to compiled code
  what          what information is required
  where         where to put the information; if NULL, return length

Returns:        0 when data returned
                > 0 when length requested
                < 0 on error or unset value
*/

PCRE2_EXP_DEFN int PCRE2_CALL_CONVENTION
pcre2_pattern_info(const pcre2_code *code, uint32_t what, void *where)
{
const pcre2_real_code *re = (const pcre2_real_code *)code;

if (where == NULL)   /* Requests field length */
  {
  switch(what)
    {
    case PCRE2_INFO_ALLOPTIONS:
    case PCRE2_INFO_ARGOPTIONS:
    case PCRE2_INFO_BACKREFMAX:
    case PCRE2_INFO_BSR:
    case PCRE2_INFO_CAPTURECOUNT:
    case PCRE2_INFO_DEPTHLIMIT:
    case PCRE2_INFO_EXTRAOPTIONS:
    case PCRE2_INFO_FIRSTCODETYPE:
    case PCRE2_INFO_FIRSTCODEUNIT:
    case PCRE2_INFO_HASBACKSLASHC:
    case PCRE2_INFO_HASCRORLF:
    case PCRE2_INFO_HEAPLIMIT:
    case PCRE2_INFO_JCHANGED:
    case PCRE2_INFO_LASTCODETYPE:
    case PCRE2_INFO_LASTCODEUNIT:
    case PCRE2_INFO_MATCHEMPTY:
    case PCRE2_INFO_MATCHLIMIT:
    case PCRE2_INFO_MAXLOOKBEHIND:
    case PCRE2_INFO_MINLENGTH:
    case PCRE2_INFO_NAMEENTRYSIZE:
    case PCRE2_INFO_NAMECOUNT:
    case PCRE2_INFO_NEWLINE:
    return sizeof(uint32_t);

    case PCRE2_INFO_FIRSTBITMAP:
    return sizeof(const uint8_t *);

    case PCRE2_INFO_JITSIZE:
    case PCRE2_INFO_SIZE:
    case PCRE2_INFO_FRAMESIZE:
    return sizeof(size_t);

    case PCRE2_INFO_NAMETABLE:
    return sizeof(PCRE2_SPTR);
    }
  }

if (re == NULL) return PCRE2_ERROR_NULL;

/* Check that the first field in the block is the magic number. If it is not,
return with PCRE2_ERROR_BADMAGIC. */

if (re->magic_number != MAGIC_NUMBER) return PCRE2_ERROR_BADMAGIC;

/* Check that this pattern was compiled in the correct bit mode */

if ((re->flags & (PCRE2_CODE_UNIT_WIDTH/8)) == 0) return PCRE2_ERROR_BADMODE;

switch(what)
  {
  case PCRE2_INFO_ALLOPTIONS:
  *((uint32_t *)where) = re->overall_options;
  break;

  case PCRE2_INFO_ARGOPTIONS:
  *((uint32_t *)where) = re->compile_options;
  break;

  case PCRE2_INFO_BACKREFMAX:
  *((uint32_t *)where) = re->top_backref;
  break;

  case PCRE2_INFO_BSR:
  *((uint32_t *)where) = re->bsr_convention;
  break;

  case PCRE2_INFO_CAPTURECOUNT:
  *((uint32_t *)where) = re->top_bracket;
  break;

  case PCRE2_INFO_DEPTHLIMIT:
  *((uint32_t *)where) = re->limit_depth;
  if (re->limit_depth == UINT32_MAX) return PCRE2_ERROR_UNSET;
  break;

  case PCRE2_INFO_EXTRAOPTIONS:
  *((uint32_t *)where) = re->extra_options;
  break;

  case PCRE2_INFO_FIRSTCODETYPE:
  *((uint32_t *)where) = ((re->flags & PCRE2_FIRSTSET) != 0)? 1 :
                         ((re->flags & PCRE2_STARTLINE) != 0)? 2 : 0;
  break;

  case PCRE2_INFO_FIRSTCODEUNIT:
  *((uint32_t *)where) = ((re->flags & PCRE2_FIRSTSET) != 0)?
    re->first_codeunit : 0;
  break;

  case PCRE2_INFO_FIRSTBITMAP:
  *((const uint8_t **)where) = ((re->flags & PCRE2_FIRSTMAPSET) != 0)?
    &(re->start_bitmap[0]) : NULL;
  break;

  case PCRE2_INFO_FRAMESIZE:
  *((size_t *)where) = offsetof(heapframe, ovector) +
    re->top_bracket * 2 * sizeof(PCRE2_SIZE);
  break;

  case PCRE2_INFO_HASBACKSLASHC:
  *((uint32_t *)where) = (re->flags & PCRE2_HASBKC) != 0;
  break;

  case PCRE2_INFO_HASCRORLF:
  *((uint32_t *)where) = (re->flags & PCRE2_HASCRORLF) != 0;
  break;

  case PCRE2_INFO_HEAPLIMIT:
  *((uint32_t *)where) = re->limit_heap;
  if (re->limit_heap == UINT32_MAX) return PCRE2_ERROR_UNSET;
  break;

  case PCRE2_INFO_JCHANGED:
  *((uint32_t *)where) = (re->flags & PCRE2_JCHANGED) != 0;
  break;

  case PCRE2_INFO_JITSIZE:
#ifdef SUPPORT_JIT
  *((size_t *)where) = (re->executable_jit != NULL)?
    PRIV(jit_get_size)(re->executable_jit) : 0;
#else
  *((size_t *)where) = 0;
#endif
  break;

  case PCRE2_INFO_LASTCODETYPE:
  *((uint32_t *)where) = ((re->flags & PCRE2_LASTSET) != 0)? 1 : 0;
  break;

  case PCRE2_INFO_LASTCODEUNIT:
  *((uint32_t *)where) = ((re->flags & PCRE2_LASTSET) != 0)?
    re->last_codeunit : 0;
  break;

  case PCRE2_INFO_MATCHEMPTY:
  *((uint32_t *)where) = (re->flags & PCRE2_MATCH_EMPTY) != 0;
  break;

  case PCRE2_INFO_MATCHLIMIT:
  *((uint32_t *)where) = re->limit_match;
  if (re->limit_match == UINT32_MAX) return PCRE2_ERROR_UNSET;
  break;

  case PCRE2_INFO_MAXLOOKBEHIND:
  *((uint32_t *)where) = re->max_lookbehind;
  break;

  case PCRE2_INFO_MINLENGTH:
  *((uint32_t *)where) = re->minlength;
  break;

  case PCRE2_INFO_NAMEENTRYSIZE:
  *((uint32_t *)where) = re->name_entry_size;
  break;

  case PCRE2_INFO_NAMECOUNT:
  *((uint32_t *)where) = re->name_count;
  break;

  case PCRE2_INFO_NAMETABLE:
  *((PCRE2_SPTR *)where) = (PCRE2_SPTR)((const char *)re +
    sizeof(pcre2_real_code));
  break;

  case PCRE2_INFO_NEWLINE:
  *((uint32_t *)where) = re->newline_convention;
  break;

  case PCRE2_INFO_SIZE:
  *((size_t *)where) = re->blocksize;
  break;

  default: return PCRE2_ERROR_BADOPTION;
  }

return 0;
}



/*************************************************
*              Callout enumerator                *
*************************************************/

/*
Arguments:
  code          points to compiled code
  callback      function called for each callout block
  callout_data  user data passed to the callback

Returns:        0 when successfully completed
                < 0 on local error
               != 0 for callback error
*/

PCRE2_EXP_DEFN int PCRE2_CALL_CONVENTION
pcre2_callout_enumerate(const pcre2_code *code,
  int (*callback)(pcre2_callout_enumerate_block *, void *), void *callout_data)
{
const pcre2_real_code *re = (const pcre2_real_code *)code;
pcre2_callout_enumerate_block cb;
PCRE2_SPTR cc;
#ifdef SUPPORT_UNICODE
BOOL utf;
#endif

if (re == NULL) return PCRE2_ERROR_NULL;

#ifdef SUPPORT_UNICODE
utf = (re->overall_options & PCRE2_UTF) != 0;
#endif

/* Check that the first field in the block is the magic number. If it is not,
return with PCRE2_ERROR_BADMAGIC. */

if (re->magic_number != MAGIC_NUMBER) return PCRE2_ERROR_BADMAGIC;

/* Check that this pattern was compiled in the correct bit mode */

if ((re->flags & (PCRE2_CODE_UNIT_WIDTH/8)) == 0) return PCRE2_ERROR_BADMODE;

cb.version = 0;
cc = (PCRE2_SPTR)((const uint8_t *)re + sizeof(pcre2_real_code))
     + re->name_count * re->name_entry_size;

while (TRUE)
  {
  int rc;
  switch (*cc)
    {
    case OP_END:
    return 0;

    case OP_CHAR:
    case OP_CHARI:
    case OP_NOT:
    case OP_NOTI:
    case OP_STAR:
    case OP_MINSTAR:
    case OP_PLUS:
    case OP_MINPLUS:
    case OP_QUERY:
    case OP_MINQUERY:
    case OP_UPTO:
    case OP_MINUPTO:
    case OP_EXACT:
    case OP_POSSTAR:
    case OP_POSPLUS:
    case OP_POSQUERY:
    case OP_POSUPTO:
    case OP_STARI:
    case OP_MINSTARI:
    case OP_PLUSI:
    case OP_MINPLUSI:
    case OP_QUERYI:
    case OP_MINQUERYI:
    case OP_UPTOI:
    case OP_MINUPTOI:
    case OP_EXACTI:
    case OP_POSSTARI:
    case OP_POSPLUSI:
    case OP_POSQUERYI:
    case OP_POSUPTOI:
    case OP_NOTSTAR:
    case OP_NOTMINSTAR:
    case OP_NOTPLUS:
    case OP_NOTMINPLUS:
    case OP_NOTQUERY:
    case OP_NOTMINQUERY:
    case OP_NOTUPTO:
    case OP_NOTMINUPTO:
    case OP_NOTEXACT:
    case OP_NOTPOSSTAR:
    case OP_NOTPOSPLUS:
    case OP_NOTPOSQUERY:
    case OP_NOTPOSUPTO:
    case OP_NOTSTARI:
    case OP_NOTMINSTARI:
    case OP_NOTPLUSI:
    case OP_NOTMINPLUSI:
    case OP_NOTQUERYI:
    case OP_NOTMINQUERYI:
    case OP_NOTUPTOI:
    case OP_NOTMINUPTOI:
    case OP_NOTEXACTI:
    case OP_NOTPOSSTARI:
    case OP_NOTPOSPLUSI:
    case OP_NOTPOSQUERYI:
    case OP_NOTPOSUPTOI:
    cc += PRIV(OP_lengths)[*cc];
#ifdef SUPPORT_UNICODE
    if (utf && HAS_EXTRALEN(cc[-1])) cc += GET_EXTRALEN(cc[-1]);
#endif
    break;

    case OP_TYPESTAR:
    case OP_TYPEMINSTAR:
    case OP_TYPEPLUS:
    case OP_TYPEMINPLUS:
    case OP_TYPEQUERY:
    case OP_TYPEMINQUERY:
    case OP_TYPEUPTO:
    case OP_TYPEMINUPTO:
    case OP_TYPEEXACT:
    case OP_TYPEPOSSTAR:
    case OP_TYPEPOSPLUS:
    case OP_TYPEPOSQUERY:
    case OP_TYPEPOSUPTO:
    cc += PRIV(OP_lengths)[*cc];
#ifdef SUPPORT_UNICODE
    if (cc[-1] == OP_PROP || cc[-1] == OP_NOTPROP) cc += 2;
#endif
    break;

#ifdef SUPPORT_WIDE_CHARS
    case OP_XCLASS:
    case OP_ECLASS:
    cc += GET(cc, 1);
    break;
#endif

    case OP_MARK:
    case OP_COMMIT_ARG:
    case OP_PRUNE_ARG:
    case OP_SKIP_ARG:
    case OP_THEN_ARG:
    cc += PRIV(OP_lengths)[*cc] + cc[1];
    break;

    case OP_CALLOUT:
    cb.pattern_position = GET(cc, 1);
    cb.next_item_length = GET(cc, 1 + LINK_SIZE);
    cb.callout_number = cc[1 + 2*LINK_SIZE];
    cb.callout_string_offset = 0;
    cb.callout_string_length = 0;
    cb.callout_string = NULL;
    rc = callback(&cb, callout_data);
    if (rc != 0) return rc;
    cc += PRIV(OP_lengths)[*cc];
    break;

    case OP_CALLOUT_STR:
    cb.pattern_position = GET(cc, 1);
    cb.next_item_length = GET(cc, 1 + LINK_SIZE);
    cb.callout_number = 0;
    cb.callout_string_offset = GET(cc, 1 + 3*LINK_SIZE);
    cb.callout_string_length =
      GET(cc, 1 + 2*LINK_SIZE) - (1 + 4*LINK_SIZE) - 2;
    cb.callout_string = cc + (1 + 4*LINK_SIZE) + 1;
    rc = callback(&cb, callout_data);
    if (rc != 0) return rc;
    cc += GET(cc, 1 + 2*LINK_SIZE);
    break;

    default:
    cc += PRIV(OP_lengths)[*cc];
    break;
    }
  }
}

/* End of pcre2_pattern_info.c */
