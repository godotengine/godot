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


#include "pcre2_compile.h"



typedef struct {
  /* Option bits for eclass. */
  uint32_t options;
  uint32_t xoptions;
  /* Rarely used members. */
  int *errorcodeptr;
  compile_block *cb;
  /* Bitmap is needed. */
  BOOL needs_bitmap;
} eclass_context;

/* Checks the allowed tokens at the end of a class structure in debug mode.
When a new token is not processed by all loops, and the token is equals to
a) one of the cases here:
   the compiler will complain about a duplicated case value.
b) none of the cases here:
   the loop without the handler will stop with an assertion failure. */

#ifdef PCRE2_DEBUG
#define CLASS_END_CASES(meta) \
  default: \
  PCRE2_ASSERT((meta) <= META_END); \
  PCRE2_FALLTHROUGH /* Fall through */ \
  case META_CLASS: \
  case META_CLASS_NOT: \
  case META_CLASS_EMPTY: \
  case META_CLASS_EMPTY_NOT: \
  case META_CLASS_END: \
  case META_ECLASS_AND: \
  case META_ECLASS_OR: \
  case META_ECLASS_SUB: \
  case META_ECLASS_XOR: \
  case META_ECLASS_NOT:
#else
#define CLASS_END_CASES(meta) \
  default:
#endif

#ifdef SUPPORT_WIDE_CHARS

/* Heapsort algorithm. */

static void do_heapify(uint32_t *buffer, size_t size, size_t i)
{
size_t max;
size_t left;
size_t right;
uint32_t tmp1, tmp2;

while (TRUE)
  {
  max = i;
  left = (i << 1) + 2;
  right = left + 2;

  if (left < size && buffer[left] > buffer[max]) max = left;
  if (right < size && buffer[right] > buffer[max]) max = right;
  if (i == max) return;

  /* Swap items. */
  tmp1 = buffer[i];
  tmp2 = buffer[i + 1];
  buffer[i] = buffer[max];
  buffer[i + 1] = buffer[max + 1];
  buffer[max] = tmp1;
  buffer[max + 1] = tmp2;
  i = max;
  }
}

#ifdef SUPPORT_UNICODE

#define PARSE_CLASS_UTF               0x1
#define PARSE_CLASS_CASELESS_UTF      0x2
#define PARSE_CLASS_RESTRICTED_UTF    0x4
#define PARSE_CLASS_TURKISH_UTF       0x8

/* Get the range of nocase characters which includes the
'c' character passed as argument, or directly follows 'c'. */

static const uint32_t*
get_nocase_range(uint32_t c)
{
uint32_t left = 0;
uint32_t right = PRIV(ucd_nocase_ranges_size);
uint32_t middle;

if (c > MAX_UTF_CODE_POINT) return PRIV(ucd_nocase_ranges) + right;

while (TRUE)
  {
  /* Range end of the middle element. */
  middle = ((left + right) >> 1) | 0x1;

  if (PRIV(ucd_nocase_ranges)[middle] <= c)
    left = middle + 1;
  else if (middle > 1 && PRIV(ucd_nocase_ranges)[middle - 2] > c)
    right = middle - 1;
  else
    return PRIV(ucd_nocase_ranges) + (middle - 1);
  }
}

/* Get the list of othercase characters, which belongs to the passed range.
Create ranges from these characters, and append them to the buffer argument. */

static size_t
utf_caseless_extend(uint32_t start, uint32_t end, uint32_t options,
  uint32_t *buffer)
{
uint32_t new_start = start;
uint32_t new_end = end;
uint32_t c = start;
const uint32_t *list;
uint32_t tmp[3];
size_t result = 2;
const uint32_t *skip_range = get_nocase_range(c);
uint32_t skip_start = skip_range[0];

#if PCRE2_CODE_UNIT_WIDTH == 8
PCRE2_ASSERT(options & PARSE_CLASS_UTF);
#endif

#if PCRE2_CODE_UNIT_WIDTH == 32
if (end > MAX_UTF_CODE_POINT) end = MAX_UTF_CODE_POINT;
#endif

while (c <= end)
  {
  uint32_t co;

  if (c > skip_start)
    {
    c = skip_range[1];
    skip_range += 2;
    skip_start = skip_range[0];
    continue;
    }

  /* Compute caseless set. */

  if ((options & (PARSE_CLASS_TURKISH_UTF|PARSE_CLASS_RESTRICTED_UTF)) ==
        PARSE_CLASS_TURKISH_UTF &&
      UCD_ANY_I(c))
    {
    co = PRIV(ucd_turkish_dotted_i_caseset) + (UCD_DOTTED_I(c)? 0 : 3);
    }
  else if ((co = UCD_CASESET(c)) != 0 &&
           (options & PARSE_CLASS_RESTRICTED_UTF) != 0 &&
           PRIV(ucd_caseless_sets)[co] < 128)
    {
    co = 0;  /* Ignore the caseless set if it's restricted. */
    }

  if (co != 0)
    list = PRIV(ucd_caseless_sets) + co;
  else
    {
    co = UCD_OTHERCASE(c);
    list = tmp;
    tmp[0] = c;
    tmp[1] = NOTACHAR;

    if (co != c)
      {
      tmp[1] = co;
      tmp[2] = NOTACHAR;
      }
    }
  c++;

  /* Add characters. */
  do
    {
#if PCRE2_CODE_UNIT_WIDTH == 16
    if (!(options & PARSE_CLASS_UTF) && *list > 0xffff) continue;
#endif

    if (*list < new_start)
      {
      if (*list + 1 == new_start)
        {
        new_start--;
        continue;
        }
      }
    else if (*list > new_end)
      {
      if (*list - 1 == new_end)
        {
        new_end++;
        continue;
        }
      }
    else continue;

    result += 2;
    if (buffer != NULL)
      {
      buffer[0] = *list;
      buffer[1] = *list;
      buffer += 2;
      }
    }
  while (*(++list) != NOTACHAR);
  }

  if (buffer != NULL)
    {
    buffer[0] = new_start;
    buffer[1] = new_end;
    buffer += 2;
    (void)buffer;
    }
  return result;
}

#endif

/* Add a character list to a buffer. */

static size_t
append_char_list(const uint32_t *p, uint32_t *buffer)
{
const uint32_t *n;
size_t result = 0;

while (*p != NOTACHAR)
  {
  n = p;
  while (n[0] == n[1] - 1) n++;

  PCRE2_ASSERT(*p < 0xffff);

  if (buffer != NULL)
    {
    buffer[0] = *p;
    buffer[1] = *n;
    buffer += 2;
    }

  result += 2;
  p = n + 1;
  }

  return result;
}

static uint32_t
get_highest_char(uint32_t options)
{
(void)options; /* Avoid compiler warning. */

#if PCRE2_CODE_UNIT_WIDTH == 8
return MAX_UTF_CODE_POINT;
#else
#ifdef SUPPORT_UNICODE
return GET_MAX_CHAR_VALUE((options & PARSE_CLASS_UTF) != 0);
#else
return MAX_UCHAR_VALUE;
#endif
#endif
}

/* Add a negated character list to a buffer. */
static size_t
append_negated_char_list(const uint32_t *p, uint32_t options, uint32_t *buffer)
{
const uint32_t *n;
uint32_t start = 0;
size_t result = 2;

PCRE2_ASSERT(*p > 0);

while (*p != NOTACHAR)
  {
  n = p;
  while (n[0] == n[1] - 1) n++;

  PCRE2_ASSERT(*p < 0xffff);

  if (buffer != NULL)
    {
    buffer[0] = start;
    buffer[1] = *p - 1;
    buffer += 2;
    }

  result += 2;
  start = *n + 1;
  p = n + 1;
  }

  if (buffer != NULL)
    {
    buffer[0] = start;
    buffer[1] = get_highest_char(options);
    buffer += 2;
    (void)buffer;
    }

  return result;
}

static uint32_t *
append_non_ascii_range(uint32_t options, uint32_t *buffer)
{
  if (buffer == NULL) return NULL;

  buffer[0] = 0x100;
  buffer[1] = get_highest_char(options);
  return buffer + 2;
}

static size_t
parse_class(uint32_t *ptr, uint32_t options, uint32_t *buffer)
{
size_t total_size = 0;
size_t size;
uint32_t meta_arg;
uint32_t start_char;

while (TRUE)
  {
  switch (META_CODE(*ptr))
    {
    case META_ESCAPE:
      meta_arg = META_DATA(*ptr);
      switch (meta_arg)
        {
        case ESC_D:
        case ESC_W:
        case ESC_S:
        buffer = append_non_ascii_range(options, buffer);
        total_size += 2;
        break;

        case ESC_h:
        size = append_char_list(PRIV(hspace_list), buffer);
        total_size += size;
        if (buffer != NULL) buffer += size;
        break;

        case ESC_H:
        size = append_negated_char_list(PRIV(hspace_list), options, buffer);
        total_size += size;
        if (buffer != NULL) buffer += size;
        break;

        case ESC_v:
        size = append_char_list(PRIV(vspace_list), buffer);
        total_size += size;
        if (buffer != NULL) buffer += size;
        break;

        case ESC_V:
        size = append_negated_char_list(PRIV(vspace_list), options, buffer);
        total_size += size;
        if (buffer != NULL) buffer += size;
        break;

        case ESC_p:
        case ESC_P:
        ptr++;
        if (meta_arg == ESC_p && (*ptr >> 16) == PT_ANY)
          {
          if (buffer != NULL)
            {
            buffer[0] = 0;
            buffer[1] = get_highest_char(options);
            buffer += 2;
            }
          total_size += 2;
          }
        break;
        }
      ptr++;
      continue;
    case META_POSIX_NEG:
      buffer = append_non_ascii_range(options, buffer);
      total_size += 2;
      ptr += 2;
      continue;
    case META_POSIX:
      ptr += 2;
      continue;
    case META_BIGVALUE:
      /* Character literal */
      ptr++;
      break;
    CLASS_END_CASES(*ptr)
      if (*ptr >= META_END) return total_size;
      break;
    }

    start_char = *ptr;

    if (ptr[1] == META_RANGE_LITERAL || ptr[1] == META_RANGE_ESCAPED)
      {
      ptr += 2;
      PCRE2_ASSERT(*ptr < META_END || *ptr == META_BIGVALUE);

      if (*ptr == META_BIGVALUE) ptr++;

#ifdef EBCDIC
#error "Missing EBCDIC support"
#endif
      }

#ifdef SUPPORT_UNICODE
    if (options & PARSE_CLASS_CASELESS_UTF)
      {
      size = utf_caseless_extend(start_char, *ptr++, options, buffer);
      if (buffer != NULL) buffer += size;
      total_size += size;
      continue;
      }
#endif

    if (buffer != NULL)
      {
      buffer[0] = start_char;
      buffer[1] = *ptr;
      buffer += 2;
      }

    ptr++;
    total_size += 2;
  }

  return total_size;
}

/* Extra uint32_t values for storing the lengths of range lists in
the worst case. Two uint32_t lengths and a range end for a range
starting before 255 */
#define CHAR_LIST_EXTRA_SIZE 3

/* Starting character values for each character list. */

static const uint32_t char_list_starts[] = {
#if PCRE2_CODE_UNIT_WIDTH == 32
  XCL_CHAR_LIST_HIGH_32_START,
#endif
#if PCRE2_CODE_UNIT_WIDTH == 32 || defined SUPPORT_UNICODE
  XCL_CHAR_LIST_LOW_32_START,
#endif
  XCL_CHAR_LIST_HIGH_16_START,
  /* Must be terminated by XCL_CHAR_LIST_LOW_16_START,
  which also represents the end of the bitset. */
  XCL_CHAR_LIST_LOW_16_START,
};

static class_ranges *
compile_optimize_class(uint32_t *start_ptr, uint32_t options,
  uint32_t xoptions, compile_block *cb)
{
class_ranges* cranges;
uint32_t *ptr;
uint32_t *buffer;
uint32_t *dst;
uint32_t class_options = 0;
size_t range_list_size = 0, total_size, i;
uint32_t tmp1, tmp2;
const uint32_t *char_list_next;
uint16_t *next_char;
uint32_t char_list_start, char_list_end;
uint32_t range_start, range_end;

#ifdef SUPPORT_UNICODE
if (options & PCRE2_UTF)
  class_options |= PARSE_CLASS_UTF;

if ((options & PCRE2_CASELESS) && (options & (PCRE2_UTF|PCRE2_UCP)))
  class_options |= PARSE_CLASS_CASELESS_UTF;

if (xoptions & PCRE2_EXTRA_CASELESS_RESTRICT)
  class_options |= PARSE_CLASS_RESTRICTED_UTF;

if (xoptions & PCRE2_EXTRA_TURKISH_CASING)
  class_options |= PARSE_CLASS_TURKISH_UTF;
#else
(void)options;   /* Avoid compiler warning. */
(void)xoptions;  /* Avoid compiler warning. */
#endif

/* Compute required space for the range. */

range_list_size = parse_class(start_ptr, class_options, NULL);
PCRE2_ASSERT((range_list_size & 0x1) == 0);

/* Allocate buffer. The total_size also represents the end of the buffer. */

total_size = range_list_size +
   ((range_list_size >= 2) ? CHAR_LIST_EXTRA_SIZE : 0);

cranges = cb->cx->memctl.malloc(
  sizeof(class_ranges) + total_size * sizeof(uint32_t),
  cb->cx->memctl.memory_data);

if (cranges == NULL) return NULL;

cranges->header.next = NULL;
#ifdef PCRE2_DEBUG
cranges->header.type = CDATA_CRANGE;
#endif
cranges->range_list_size = (uint16_t)range_list_size;
cranges->char_lists_types = 0;
cranges->char_lists_size = 0;
cranges->char_lists_start = 0;

if (range_list_size == 0) return cranges;

buffer = (uint32_t*)(cranges + 1);
parse_class(start_ptr, class_options, buffer);

/* Using <= instead of == to help static analysis. */
if (range_list_size <= 2) return cranges;

/* In-place sorting of ranges. */

i = (((range_list_size >> 2) - 1) << 1);
while (TRUE)
  {
  do_heapify(buffer, range_list_size, i);
  if (i == 0) break;
  i -= 2;
  }

i = range_list_size - 2;
while (TRUE)
  {
  tmp1 = buffer[i];
  tmp2 = buffer[i + 1];
  buffer[i] = buffer[0];
  buffer[i + 1] = buffer[1];
  buffer[0] = tmp1;
  buffer[1] = tmp2;

  do_heapify(buffer, i, 0);
  if (i == 0) break;
  i -= 2;
  }

/* Merge ranges whenever possible. */
dst = buffer;
ptr = buffer + 2;
range_list_size -= 2;

/* The second condition is a very rare corner case, where the end of the last
range is the maximum character. This range cannot be extended further. */

while (range_list_size > 0 && dst[1] != ~(uint32_t)0)
  {
  if (dst[1] + 1 < ptr[0])
    {
    dst += 2;
    dst[0] = ptr[0];
    dst[1] = ptr[1];
    }
  else if (dst[1] < ptr[1]) dst[1] = ptr[1];

  ptr += 2;
  range_list_size -= 2;
  }

PCRE2_ASSERT(dst[1] <= get_highest_char(class_options));

/* When the number of ranges are less than six,
they are not converted to range lists. */

ptr = buffer;
while (ptr < dst && ptr[1] < 0x100) ptr += 2;
if (dst - ptr < (2 * (6 - 1)))
  {
  cranges->range_list_size = (uint16_t)(dst + 2 - buffer);
  return cranges;
  }

/* Compute character lists structures. */

char_list_next = char_list_starts;
char_list_start = *char_list_next++;
#if PCRE2_CODE_UNIT_WIDTH == 32
char_list_end = XCL_CHAR_LIST_HIGH_32_END;
#elif defined SUPPORT_UNICODE
char_list_end = XCL_CHAR_LIST_LOW_32_END;
#else
char_list_end = XCL_CHAR_LIST_HIGH_16_END;
#endif
next_char = (uint16_t*)(buffer + total_size);

tmp1 = 0;
tmp2 = ((sizeof(char_list_starts) / sizeof(uint32_t)) - 1) * XCL_TYPE_BIT_LEN;
PCRE2_ASSERT(tmp2 <= 3 * XCL_TYPE_BIT_LEN && tmp2 >= XCL_TYPE_BIT_LEN);
range_start = dst[0];
range_end = dst[1];

while (TRUE)
  {
  if (range_start >= char_list_start)
    {
    if (range_start == range_end || range_end < char_list_end)
      {
      tmp1++;
      next_char--;

      if (char_list_start < XCL_CHAR_LIST_LOW_32_START)
        *next_char = (uint16_t)((range_end << XCL_CHAR_SHIFT) | XCL_CHAR_END);
      else
        *(uint32_t*)(--next_char) =
          (range_end << XCL_CHAR_SHIFT) | XCL_CHAR_END;
      }

    if (range_start < range_end)
      {
      if (range_start > char_list_start)
        {
        tmp1++;
        next_char--;

        if (char_list_start < XCL_CHAR_LIST_LOW_32_START)
          *next_char = (uint16_t)(range_start << XCL_CHAR_SHIFT);
        else
          *(uint32_t*)(--next_char) = (range_start << XCL_CHAR_SHIFT);
        }
      else
        cranges->char_lists_types |= XCL_BEGIN_WITH_RANGE << tmp2;
      }

    PCRE2_ASSERT((uint32_t*)next_char >= dst + 2);

    if (dst > buffer)
      {
      dst -= 2;
      range_start = dst[0];
      range_end = dst[1];
      continue;
      }

    range_start = 0;
    range_end = 0;
    }

  if (range_end >= char_list_start)
    {
    PCRE2_ASSERT(range_start < char_list_start);

    if (range_end < char_list_end)
      {
      tmp1++;
      next_char--;

      if (char_list_start < XCL_CHAR_LIST_LOW_32_START)
        *next_char = (uint16_t)((range_end << XCL_CHAR_SHIFT) | XCL_CHAR_END);
      else
        *(uint32_t*)(--next_char) =
          (range_end << XCL_CHAR_SHIFT) | XCL_CHAR_END;

      PCRE2_ASSERT((uint32_t*)next_char >= dst + 2);
      }

    cranges->char_lists_types |= XCL_BEGIN_WITH_RANGE << tmp2;
    }

  if (tmp1 >= XCL_ITEM_COUNT_MASK)
    {
    cranges->char_lists_types |= XCL_ITEM_COUNT_MASK << tmp2;
    next_char--;

    if (char_list_start < XCL_CHAR_LIST_LOW_32_START)
      *next_char = (uint16_t)tmp1;
    else
      *(uint32_t*)(--next_char) = tmp1;
    }
  else
    cranges->char_lists_types |= tmp1 << tmp2;

  if (range_start < XCL_CHAR_LIST_LOW_16_START) break;

  PCRE2_ASSERT(tmp2 >= XCL_TYPE_BIT_LEN);
  char_list_end = char_list_start - 1;
  char_list_start = *char_list_next++;
  tmp1 = 0;
  tmp2 -= XCL_TYPE_BIT_LEN;
  }

if (dst[0] < XCL_CHAR_LIST_LOW_16_START) dst += 2;
PCRE2_ASSERT((uint16_t*)dst <= next_char);

cranges->char_lists_size =
  (size_t)((uint8_t*)(buffer + total_size) - (uint8_t*)next_char);
cranges->char_lists_start = (size_t)((uint8_t*)next_char - (uint8_t*)buffer);
cranges->range_list_size = (uint16_t)(dst - buffer);
return cranges;
}

#endif /* SUPPORT_WIDE_CHARS */

#ifdef SUPPORT_UNICODE

void PRIV(update_classbits)(uint32_t ptype, uint32_t pdata, BOOL negated,
  uint8_t *classbits)
{
/* Update PRIV(xclass) when this function is changed. */
int c, chartype;
const ucd_record *prop;
uint32_t gentype;
BOOL set_bit;

if (ptype == PT_ANY)
  {
  if (!negated) memset(classbits, 0xff, 32);
  return;
  }

for (c = 0; c < 256; c++)
  {
  prop = GET_UCD(c);
  set_bit = FALSE;
  (void)set_bit;

  switch (ptype)
    {
    case PT_LAMP:
    chartype = prop->chartype;
    set_bit = (chartype == ucp_Lu || chartype == ucp_Ll || chartype == ucp_Lt);
    break;

    case PT_GC:
    set_bit = (PRIV(ucp_gentype)[prop->chartype] == pdata);
    break;

    case PT_PC:
    set_bit = (prop->chartype == pdata);
    break;

    case PT_SC:
    set_bit = (prop->script == pdata);
    break;

    case PT_SCX:
    set_bit = (prop->script == pdata ||
      MAPBIT(PRIV(ucd_script_sets) + UCD_SCRIPTX_PROP(prop), pdata) != 0);
    break;

    case PT_ALNUM:
    gentype = PRIV(ucp_gentype)[prop->chartype];
    set_bit = (gentype == ucp_L || gentype == ucp_N);
    break;

    case PT_SPACE:    /* Perl space */
    case PT_PXSPACE:  /* POSIX space */
    switch(c)
      {
      HSPACE_BYTE_CASES:
      VSPACE_BYTE_CASES:
      set_bit = TRUE;
      break;

      default:
      set_bit = (PRIV(ucp_gentype)[prop->chartype] == ucp_Z);
      break;
      }
    break;

    case PT_WORD:
    chartype = prop->chartype;
    gentype = PRIV(ucp_gentype)[chartype];
    set_bit = (gentype == ucp_L || gentype == ucp_N ||
               chartype == ucp_Mn || chartype == ucp_Pc);
    break;

    case PT_UCNC:
    set_bit = (c == CHAR_DOLLAR_SIGN || c == CHAR_COMMERCIAL_AT ||
               c == CHAR_GRAVE_ACCENT || c >= 0xa0);
    break;

    case PT_BIDICL:
    set_bit = (UCD_BIDICLASS_PROP(prop) == pdata);
    break;

    case PT_BOOL:
    set_bit = MAPBIT(PRIV(ucd_boolprop_sets) +
                     UCD_BPROPS_PROP(prop), pdata) != 0;
    break;

    case PT_PXGRAPH:
    chartype = prop->chartype;
    gentype = PRIV(ucp_gentype)[chartype];
    set_bit = (gentype != ucp_Z && (gentype != ucp_C || chartype == ucp_Cf));
    break;

    case PT_PXPRINT:
    chartype = prop->chartype;
    set_bit = (chartype != ucp_Zl && chartype != ucp_Zp &&
       (PRIV(ucp_gentype)[chartype] != ucp_C || chartype == ucp_Cf));
    break;

    case PT_PXPUNCT:
    gentype = PRIV(ucp_gentype)[prop->chartype];
    set_bit = (gentype == ucp_P || (c < 128 && gentype == ucp_S));
    break;

    default:
    PCRE2_ASSERT(ptype == PT_PXXDIGIT);
    set_bit = (c >= CHAR_0 && c <= CHAR_9) ||
              (c >= CHAR_A && c <= CHAR_F) ||
              (c >= CHAR_a && c <= CHAR_f);
    break;
    }

  if (negated) set_bit = !set_bit;
  if (set_bit) *classbits |= (uint8_t)(1 << (c & 0x7));
  if ((c & 0x7) == 0x7) classbits++;
  }
}

#endif /* SUPPORT_UNICODE */



#ifdef SUPPORT_WIDE_CHARS

/*************************************************
*           XClass related properties            *
*************************************************/

/* XClass needs to be generated. */
#define XCLASS_REQUIRED 0x1
/* XClass has 8 bit character. */
#define XCLASS_HAS_8BIT_CHARS 0x2
/* XClass has properties. */
#define XCLASS_HAS_PROPS 0x4
/* XClass has character lists. */
#define XCLASS_HAS_CHAR_LISTS 0x8
/* XClass matches to all >= 256 characters. */
#define XCLASS_HIGH_ANY 0x10

#endif


/*************************************************
*   Internal entry point for add range to class  *
*************************************************/

/* This function sets the overall range for characters < 256.
It also handles non-utf case folding.

Arguments:
  options       the options bits
  xoptions      the extra options bits
  cb            compile data
  start         start of range character
  end           end of range character

Returns:        cb->classbits is updated
*/

static void
add_to_class(uint32_t options, uint32_t xoptions, compile_block *cb,
  uint32_t start, uint32_t end)
{
uint8_t *classbits = cb->classbits.classbits;
uint32_t c, byte_start, byte_end;
uint32_t classbits_end = (end <= 0xff ? end : 0xff);

#ifndef SUPPORT_UNICODE
(void)xoptions; /* Avoid compiler warning. */
#endif

/* If caseless matching is required, scan the range and process alternate
cases. In Unicode, there are 8-bit characters that have alternate cases that
are greater than 255 and vice-versa (though these may be ignored if caseless
restriction is in force). Sometimes we can just extend the original range. */

if ((options & PCRE2_CASELESS) != 0)
  {
#ifdef SUPPORT_UNICODE
  /* UTF mode. This branch is taken if we don't support wide characters (e.g.
  8-bit library, without UTF), but we do treat those characters as Unicode
  (if UCP flag is set). In this case, we only need to expand the character class
  set to include the case pairs which are in the 0-255 codepoint range. */
  if ((options & (PCRE2_UTF|PCRE2_UCP)) != 0)
    {
      BOOL turkish_i = (xoptions & (PCRE2_EXTRA_TURKISH_CASING|PCRE2_EXTRA_CASELESS_RESTRICT)) ==
        PCRE2_EXTRA_TURKISH_CASING;
      if (start < 128)
        {
        uint32_t lo_end = (classbits_end < 127 ? classbits_end : 127);
        for (c = start; c <= lo_end; c++)
          {
          if (turkish_i && UCD_ANY_I(c)) continue;
          SETBIT(classbits, cb->fcc[c]);
          }
        }
      if (classbits_end >= 128)
        {
        uint32_t hi_start = (start > 128 ? start : 128);
        for (c = hi_start; c <= classbits_end; c++)
          {
          uint32_t co = UCD_OTHERCASE(c);
          if (co <= 0xff) SETBIT(classbits, co);
          }
        }
    }

  else
#endif  /* SUPPORT_UNICODE */

  /* Not UTF mode */
    {
    for (c = start; c <= classbits_end; c++)
      SETBIT(classbits, cb->fcc[c]);
    }
  }

/* Use the bitmap for characters < 256. Otherwise use extra data. */

byte_start = (start + 7) >> 3;
byte_end = (classbits_end + 1) >> 3;

if (byte_start >= byte_end)
  {
  for (c = start; c <= classbits_end; c++)
    /* Regardless of start, c will always be <= 255. */
    SETBIT(classbits, c);
  return;
  }

for (c = byte_start; c < byte_end; c++)
  classbits[c] = 0xff;

byte_start <<= 3;
byte_end <<= 3;

for (c = start; c < byte_start; c++)
  SETBIT(classbits, c);

for (c = byte_end; c <= classbits_end; c++)
  SETBIT(classbits, c);
}


#if PCRE2_CODE_UNIT_WIDTH == 8
/*************************************************
*   Internal entry point for add list to class   *
*************************************************/

/* This function is used for adding a list of horizontal or vertical whitespace
characters to a class. The list must be in order so that ranges of characters
can be detected and handled appropriately. This function sets the overall range
so that the internal functions can try to avoid duplication when handling
case-independence.

Arguments:
  options       the options bits
  xoptions      the extra options bits
  cb            contains pointers to tables etc.
  p             points to row of 32-bit values, terminated by NOTACHAR

Returns:        cb->classbits is updated
*/

static void
add_list_to_class(uint32_t options, uint32_t xoptions, compile_block *cb,
  const uint32_t *p)
{
while (p[0] < 256)
  {
  unsigned int n = 0;

  while(p[n+1] == p[0] + n + 1) n++;
  add_to_class(options, xoptions, cb, p[0], p[n]);

  p += n + 1;
  }
}



/*************************************************
*    Add characters not in a list to a class     *
*************************************************/

/* This function is used for adding the complement of a list of horizontal or
vertical whitespace to a class. The list must be in order.

Arguments:
  options       the options bits
  xoptions      the extra options bits
  cb            contains pointers to tables etc.
  p             points to row of 32-bit values, terminated by NOTACHAR

Returns:        cb->classbits is updated
*/

static void
add_not_list_to_class(uint32_t options, uint32_t xoptions, compile_block *cb,
  const uint32_t *p)
{
if (p[0] > 0)
  add_to_class(options, xoptions, cb, 0, p[0] - 1);
while (p[0] < 256)
  {
  while (p[1] == p[0] + 1) p++;
  add_to_class(options, xoptions, cb, p[0] + 1, (p[1] > 255) ? 255 : p[1] - 1);
  p++;
  }
}
#endif /* PCRE2_CODE_UNIT_WIDTH == 8 */



/*************************************************
*  Main entry-point to compile a character class *
*************************************************/

/* This function consumes a "leaf", which is a set of characters that will
become a single OP_CLASS OP_NCLASS, OP_XCLASS, or OP_ALLANY. */

uint32_t *
PRIV(compile_class_not_nested)(uint32_t options, uint32_t xoptions,
  uint32_t *start_ptr, PCRE2_UCHAR **pcode, BOOL negate_class, BOOL* has_bitmap,
  int *errorcodeptr, compile_block *cb, PCRE2_SIZE *lengthptr)
{
uint32_t *pptr = start_ptr;
PCRE2_UCHAR *code = *pcode;
BOOL should_flip_negation;
const uint8_t *cbits = cb->cbits;
/* Some functions such as add_to_class() or eclass processing
expects that the bitset is stored in cb->classbits.classbits. */
uint8_t *const classbits = cb->classbits.classbits;

#ifdef SUPPORT_UNICODE
BOOL utf = (options & PCRE2_UTF) != 0;
#else  /* No Unicode support */
BOOL utf = FALSE;
#endif

/* Helper variables for OP_XCLASS opcode (for characters > 255). */

#ifdef SUPPORT_WIDE_CHARS
uint32_t xclass_props;
PCRE2_UCHAR *class_uchardata;
class_ranges* cranges;
#else
(void)has_bitmap;    /* Avoid compiler warning. */
(void)errorcodeptr;  /* Avoid compiler warning. */
(void)lengthptr;     /* Avoid compiler warning. */
#endif

/* If an XClass contains a negative special such as \S, we need to flip the
negation flag at the end, so that support for characters > 255 works correctly
(they are all included in the class). An XClass may need to insert specific
matching or non-matching code for wide characters.
*/

should_flip_negation = FALSE;

/* XClass will be used when characters > 255 might match. */

#ifdef SUPPORT_WIDE_CHARS
xclass_props = 0;

#if PCRE2_CODE_UNIT_WIDTH == 8
cranges = NULL;

if (utf)
#endif
  {
  if (lengthptr != NULL)
    {
    cranges = compile_optimize_class(pptr, options, xoptions, cb);

    if (cranges == NULL)
      {
      *errorcodeptr = ERR21;
      return NULL;
      }

    /* Caching the pre-processed character ranges. */
    if (cb->last_data != NULL)
      cb->last_data->next = &cranges->header;
    else
      cb->first_data = &cranges->header;

    cb->last_data = &cranges->header;
    }
  else
    {
    /* Reuse the pre-processed character ranges. */
    cranges = (class_ranges*)cb->first_data;
    PCRE2_ASSERT(cranges != NULL && cranges->header.type == CDATA_CRANGE);
    cb->first_data = cranges->header.next;
    }

  if (cranges->range_list_size > 0)
    {
    const uint32_t *ranges = (const uint32_t*)(cranges + 1);

    if (ranges[0] <= 255)
      xclass_props |= XCLASS_HAS_8BIT_CHARS;

    if (ranges[cranges->range_list_size - 1] == GET_MAX_CHAR_VALUE(utf) &&
        ranges[cranges->range_list_size - 2] <= 256)
      xclass_props |= XCLASS_HIGH_ANY;
    }
  }

class_uchardata = code + LINK_SIZE + 2;   /* For XCLASS items */
#endif /* SUPPORT_WIDE_CHARS */

/* Initialize the 256-bit (32-byte) bit map to all zeros. We build the map
in a temporary bit of memory, in case the class contains fewer than two
8-bit characters because in that case the compiled code doesn't use the bit
map. */

memset(classbits, 0, 32);

/* Process items until end_ptr is reached. */

while (TRUE)
  {
  uint32_t meta = *(pptr++);
  BOOL local_negate;
  int posix_class;
  int taboffset, tabopt;
  class_bits_storage pbits;
  uint32_t escape, c;

  /* Handle POSIX classes such as [:alpha:] etc. */
  switch (META_CODE(meta))
    {
    case META_POSIX:
    case META_POSIX_NEG:

    local_negate = (meta == META_POSIX_NEG);
    posix_class = *(pptr++);

    if (local_negate) should_flip_negation = TRUE;  /* Note negative special */

    /* If matching is caseless, upper and lower are converted to alpha.
    This relies on the fact that the class table starts with alpha,
    lower, upper as the first 3 entries. */

    if ((options & PCRE2_CASELESS) != 0 && posix_class <= 2)
      posix_class = 0;

    /* When PCRE2_UCP is set, some of the POSIX classes are converted to
    different escape sequences that use Unicode properties \p or \P.
    Others that are not available via \p or \P have to generate
    XCL_PROP/XCL_NOTPROP directly, which is done here. */

#ifdef SUPPORT_UNICODE
    /* TODO This entire block of code here appears to be unreachable!? I simply
    can't see how it can be hit, given that the frontend parser doesn't emit
    META_POSIX for GRAPH/PRINT/PUNCT when UCP is set. */
    if ((options & PCRE2_UCP) != 0 &&
        (xoptions & PCRE2_EXTRA_ASCII_POSIX) == 0)
      {
      uint32_t ptype;

      switch(posix_class)
        {
        case PC_GRAPH:
        case PC_PRINT:
        case PC_PUNCT:
        ptype = (posix_class == PC_GRAPH)? PT_PXGRAPH :
                (posix_class == PC_PRINT)? PT_PXPRINT : PT_PXPUNCT;

        PRIV(update_classbits)(ptype, 0, local_negate, classbits);

        if ((xclass_props & XCLASS_HIGH_ANY) == 0)
          {
          if (lengthptr != NULL)
            *lengthptr += 3;
          else
            {
            *class_uchardata++ = local_negate? XCL_NOTPROP : XCL_PROP;
            *class_uchardata++ = (PCRE2_UCHAR)ptype;
            *class_uchardata++ = 0;
            }
          xclass_props |= XCLASS_REQUIRED | XCLASS_HAS_PROPS;
          }
        continue;

        /* For the other POSIX classes (ex: ascii) we are going to
        fall through to the non-UCP case and build a bit map for
        characters with code points less than 256. However, if we are in
        a negated POSIX class, characters with code points greater than
        255 must either all match or all not match, depending on whether
        the whole class is not or is negated. For example, for
        [[:^ascii:]... they must all match, whereas for [^[:^ascii:]...
        they must not.

        In the special case where there are no xclass items, this is
        automatically handled by the use of OP_CLASS or OP_NCLASS, but an
        explicit range is needed for OP_XCLASS. Setting a flag here
        causes the range to be generated later when it is known that
        OP_XCLASS is required. In the 8-bit library this is relevant only in
        utf mode, since no wide characters can exist otherwise. */

        default:
        break;
        }
      }
#endif  /* SUPPORT_UNICODE */

    /* In the non-UCP case, or when UCP makes no difference, we build the
    bit map for the POSIX class in a chunk of local store because we may
    be adding and subtracting from it, and we don't want to subtract bits
    that may be in the main map already. At the end we or the result into
    the bit map that is being built. */

    posix_class *= 3;

    /* Copy in the first table (always present) */

    memcpy(pbits.classbits, cbits + PRIV(posix_class_maps)[posix_class], 32);

    /* If there is a second table, add or remove it as required. */

    taboffset = PRIV(posix_class_maps)[posix_class + 1];
    tabopt = PRIV(posix_class_maps)[posix_class + 2];

    if (taboffset >= 0)
      {
      if (tabopt >= 0)
        for (int i = 0; i < 32; i++)
          pbits.classbits[i] |= cbits[i + taboffset];
      else
        for (int i = 0; i < 32; i++)
          pbits.classbits[i] &= (uint8_t)(~cbits[i + taboffset]);
      }

    /* Now see if we need to remove any special characters. An option
    value of 1 removes vertical space and 2 removes underscore. */

    if (tabopt < 0) tabopt = -tabopt;
#ifdef EBCDIC
      {
      uint8_t posix_vertical[4] = { CHAR_LF, CHAR_VT, CHAR_FF, CHAR_CR };
      uint8_t posix_underscore = CHAR_UNDERSCORE;
      uint8_t *chars = NULL;
      int n = 0;

      if (tabopt == 1) { chars = posix_vertical; n = 4; }
      else if (tabopt == 2) { chars = &posix_underscore; n = 1; }

      for (; n > 0; ++chars, --n)
        pbits.classbits[*chars/8] &= ~(1u << (*chars&7));
      }
#else
    if (tabopt == 1) pbits.classbits[1] &= ~0x3c;
    else if (tabopt == 2) pbits.classbits[11] &= 0x7f;
#endif

    /* Add the POSIX table or its complement into the main table that is
    being built and we are done. */

      {
      uint32_t *classwords = cb->classbits.classwords;

      if (local_negate)
        for (int i = 0; i < 8; i++)
          classwords[i] |= (uint32_t)(~pbits.classwords[i]);
      else
        for (int i = 0; i < 8; i++)
          classwords[i] |= pbits.classwords[i];
      }

#ifdef SUPPORT_WIDE_CHARS
    /* Every class contains at least one < 256 character. */
    xclass_props |= XCLASS_HAS_8BIT_CHARS;
#endif
    continue;               /* End of POSIX handling */

    /* Other than POSIX classes, the only items we should encounter are
    \d-type escapes and literal characters (possibly as ranges). */
    case META_BIGVALUE:
    meta = *(pptr++);
    break;

    case META_ESCAPE:
    escape = META_DATA(meta);

    switch(escape)
      {
      case ESC_d:
      for (int i = 0; i < 32; i++) classbits[i] |= cbits[i+cbit_digit];
      break;

      case ESC_D:
      should_flip_negation = TRUE;
      for (int i = 0; i < 32; i++)
        classbits[i] |= (uint8_t)(~cbits[i+cbit_digit]);
      break;

      case ESC_w:
      for (int i = 0; i < 32; i++) classbits[i] |= cbits[i+cbit_word];
      break;

      case ESC_W:
      should_flip_negation = TRUE;
      for (int i = 0; i < 32; i++)
        classbits[i] |= (uint8_t)(~cbits[i+cbit_word]);
      break;

      /* Perl 5.004 onwards omitted VT from \s, but restored it at Perl
      5.18. Before PCRE 8.34, we had to preserve the VT bit if it was
      previously set by something earlier in the character class.
      Luckily, the value of CHAR_VT is 0x0b in both ASCII and EBCDIC, so
      we could just adjust the appropriate bit. From PCRE 8.34 we no
      longer treat \s and \S specially. */

      case ESC_s:
      for (int i = 0; i < 32; i++) classbits[i] |= cbits[i+cbit_space];
      break;

      case ESC_S:
      should_flip_negation = TRUE;
      for (int i = 0; i < 32; i++)
        classbits[i] |= (uint8_t)(~cbits[i+cbit_space]);
      break;

      /* When adding the horizontal or vertical space lists to a class, or
      their complements, disable PCRE2_CASELESS, because it justs wastes
      time, and in the "not-x" UTF cases can create unwanted duplicates in
      the XCLASS list (provoked by characters that have more than one other
      case and by both cases being in the same "not-x" sublist). */

      case ESC_h:
#if PCRE2_CODE_UNIT_WIDTH == 8
#ifdef SUPPORT_UNICODE
      if (cranges != NULL) break;
#endif
      add_list_to_class(options & ~PCRE2_CASELESS, xoptions,
        cb, PRIV(hspace_list));
#else
      PCRE2_ASSERT(cranges != NULL);
#endif
      break;

      case ESC_H:
#if PCRE2_CODE_UNIT_WIDTH == 8
#ifdef SUPPORT_UNICODE
      if (cranges != NULL) break;
#endif
      add_not_list_to_class(options & ~PCRE2_CASELESS, xoptions,
        cb, PRIV(hspace_list));
#else
      PCRE2_ASSERT(cranges != NULL);
#endif
      break;

      case ESC_v:
#if PCRE2_CODE_UNIT_WIDTH == 8
#ifdef SUPPORT_UNICODE
      if (cranges != NULL) break;
#endif
      add_list_to_class(options & ~PCRE2_CASELESS, xoptions,
        cb, PRIV(vspace_list));
#else
      PCRE2_ASSERT(cranges != NULL);
#endif
      break;

      case ESC_V:
#if PCRE2_CODE_UNIT_WIDTH == 8
#ifdef SUPPORT_UNICODE
      if (cranges != NULL) break;
#endif
      add_not_list_to_class(options & ~PCRE2_CASELESS, xoptions,
        cb, PRIV(vspace_list));
#else
      PCRE2_ASSERT(cranges != NULL);
#endif
      break;

      /* If Unicode is not supported, \P and \p are not allowed and are
      faulted at parse time, so will never appear here. */

#ifdef SUPPORT_UNICODE
      case ESC_p:
      case ESC_P:
        {
        uint32_t ptype = *pptr >> 16;
        uint32_t pdata = *(pptr++) & 0xffff;

        /* The "Any" is processed by PRIV(update_classbits)(). */
        if (ptype == PT_ANY)
          {
#if PCRE2_CODE_UNIT_WIDTH == 8
          if (!utf && escape == ESC_p) memset(classbits, 0xff, 32);
#endif
          continue;
          }

        PRIV(update_classbits)(ptype, pdata, (escape == ESC_P), classbits);

        if ((xclass_props & XCLASS_HIGH_ANY) == 0)
          {
          if (lengthptr != NULL)
            *lengthptr += 3;
          else
            {
            *class_uchardata++ = (escape == ESC_p)? XCL_PROP : XCL_NOTPROP;
            *class_uchardata++ = ptype;
            *class_uchardata++ = pdata;
            }
          xclass_props |= XCLASS_REQUIRED | XCLASS_HAS_PROPS;
          }
        }
      continue;
#endif
      }

#ifdef SUPPORT_WIDE_CHARS
    /* Every non-property class contains at least one < 256 character. */
    xclass_props |= XCLASS_HAS_8BIT_CHARS;
#endif
    /* End handling \d-type escapes */
    continue;

    CLASS_END_CASES(meta)
    /* Literals. */
    if (meta < META_END) break;
    /* Non-literals: end of class contents. */
    goto END_PROCESSING;
    }

  /* A literal character may be followed by a range meta. At parse time
  there are checks for out-of-order characters, for ranges where the two
  characters are equal, and for hyphens that cannot indicate a range. At
  this point, therefore, no checking is needed. */

  c = meta;

  /* Remember if \r or \n were explicitly used */

  if (c == CHAR_CR || c == CHAR_NL) cb->external_flags |= PCRE2_HASCRORLF;

  /* Process a character range */

  if (*pptr == META_RANGE_LITERAL || *pptr == META_RANGE_ESCAPED)
    {
    uint32_t d;

#ifdef EBCDIC
    BOOL range_is_literal = (*pptr == META_RANGE_LITERAL);
#endif
    ++pptr;
    d = *(pptr++);
    if (d == META_BIGVALUE) d = *(pptr++);

    /* Remember an explicit \r or \n, and add the range to the class. */

    if (d == CHAR_CR || d == CHAR_NL) cb->external_flags |= PCRE2_HASCRORLF;

#if PCRE2_CODE_UNIT_WIDTH == 8
#ifdef SUPPORT_UNICODE
    if (cranges != NULL) continue;
    xclass_props |= XCLASS_HAS_8BIT_CHARS;
#endif

    /* In an EBCDIC environment, Perl treats alphabetic ranges specially
    because there are holes in the encoding, and simply using the range
    A-Z (for example) would include the characters in the holes. This
    applies only to literal ranges; [\xC1-\xE9] is different to [A-Z]. */

#ifdef EBCDIC
    if (range_is_literal &&
         (cb->ctypes[c] & ctype_letter) != 0 &&
         (cb->ctypes[d] & ctype_letter) != 0 &&
         (c <= CHAR_z) == (d <= CHAR_z))
      {
      uint32_t uc = (d <= CHAR_z)? 0 : 64;
      uint32_t C = c - uc;
      uint32_t D = d - uc;

      if (C <= CHAR_i)
        {
        add_to_class(options, xoptions, cb, C + uc,
          ((D < CHAR_i)? D : CHAR_i) + uc);
        C = CHAR_j;
        }

      if (C <= D && C <= CHAR_r)
        {
        add_to_class(options, xoptions, cb, C + uc,
          ((D < CHAR_r)? D : CHAR_r) + uc);
        C = CHAR_s;
        }

      if (C <= D)
        add_to_class(options, xoptions, cb, C + uc, D + uc);
      }
    else
#endif
    /* Not an EBCDIC special range */

    add_to_class(options, xoptions, cb, c, d);
#else
    PCRE2_ASSERT(cranges != NULL);
#endif
    continue;
    }  /* End of range handling */

  /* Character ranges are ignored when class_ranges is present. */
#if PCRE2_CODE_UNIT_WIDTH == 8
#ifdef SUPPORT_UNICODE
  if (cranges != NULL) continue;
  xclass_props |= XCLASS_HAS_8BIT_CHARS;
#endif
  /* Handle a single character. */

  add_to_class(options, xoptions, cb, meta, meta);
#else
  PCRE2_ASSERT(cranges != NULL);
#endif
  }   /* End of main class-processing loop */

END_PROCESSING:

#ifdef SUPPORT_WIDE_CHARS
PCRE2_ASSERT((xclass_props & XCLASS_HAS_PROPS) == 0 ||
             (xclass_props & XCLASS_HIGH_ANY) == 0);

if (cranges != NULL)
  {
  uint32_t *range = (uint32_t*)(cranges + 1);
  uint32_t *end = range + cranges->range_list_size;

  while (range < end && range[0] < 256)
    {
    PCRE2_ASSERT((xclass_props & XCLASS_HAS_8BIT_CHARS) != 0);
    /* Add range to bitset. If we are in UTF or UCP mode, then clear the
    caseless bit, because the cranges handle caselessness (only) in this
    condition; see the condition for PARSE_CLASS_CASELESS_UTF in
    compile_optimize_class(). */
    add_to_class(((options & (PCRE2_UTF|PCRE2_UCP)) != 0)?
        (options & ~PCRE2_CASELESS) : options, xoptions, cb, range[0], range[1]);

    if (range[1] > 255) break;
    range += 2;
    }

  if (cranges->char_lists_size > 0)
    {
    /* The cranges structure is still used and freed later. */
    PCRE2_ASSERT((xclass_props & XCLASS_HIGH_ANY) == 0);
    xclass_props |= XCLASS_REQUIRED | XCLASS_HAS_CHAR_LISTS;
    }
  else
    {
    if ((xclass_props & XCLASS_HIGH_ANY) != 0)
      {
      PCRE2_ASSERT(range + 2 == end && range[0] <= 256 &&
        range[1] >= GET_MAX_CHAR_VALUE(utf));
      should_flip_negation = TRUE;
      range = end;
      }

    while (range < end)
      {
      uint32_t range_start = range[0];
      uint32_t range_end = range[1];

      range += 2;
      xclass_props |= XCLASS_REQUIRED;

      if (range_start < 256) range_start = 256;

      if (lengthptr != NULL)
        {
#ifdef SUPPORT_UNICODE
        if (utf)
          {
          *lengthptr += 1;

          if (range_start < range_end)
            *lengthptr += PRIV(ord2utf)(range_start, class_uchardata);

          *lengthptr += PRIV(ord2utf)(range_end, class_uchardata);
          continue;
          }
#endif  /* SUPPORT_UNICODE */

        *lengthptr += range_start < range_end ? 3 : 2;
        continue;
        }

#ifdef SUPPORT_UNICODE
      if (utf)
        {
        if (range_start < range_end)
          {
          *class_uchardata++ = XCL_RANGE;
          class_uchardata += PRIV(ord2utf)(range_start, class_uchardata);
          }
        else
          *class_uchardata++ = XCL_SINGLE;

        class_uchardata += PRIV(ord2utf)(range_end, class_uchardata);
        continue;
        }
#endif  /* SUPPORT_UNICODE */

      /* Without UTF support, character values are constrained
      by the bit length, and can only be > 256 for 16-bit and
      32-bit libraries. */
#if PCRE2_CODE_UNIT_WIDTH != 8
      if (range_start < range_end)
        {
        *class_uchardata++ = XCL_RANGE;
        *class_uchardata++ = range_start;
        }
      else
        *class_uchardata++ = XCL_SINGLE;

      *class_uchardata++ = range_end;
#endif  /* PCRE2_CODE_UNIT_WIDTH == 8 */
      }

    if (lengthptr == NULL)
      cb->cx->memctl.free(cranges, cb->cx->memctl.memory_data);
    }
  }
#endif /* SUPPORT_WIDE_CHARS */

/* If there are characters with values > 255, or Unicode property settings
(\p or \P), we have to compile an extended class, with its own opcode,
unless there were no property settings and there was a negated special such
as \S in the class, and PCRE2_UCP is not set, because in that case all
characters > 255 are in or not in the class, so any that were explicitly
given as well can be ignored.

In the UCP case, if certain negated POSIX classes (ex: [:^ascii:]) were
were present in a class, we either have to match or not match all wide
characters (depending on whether the whole class is or is not negated).
This requirement is indicated by match_all_or_no_wide_chars being true.
We do this by including an explicit range, which works in both cases.
This applies only in UTF and 16-bit and 32-bit non-UTF modes, since there
cannot be any wide characters in 8-bit non-UTF mode.

When there *are* properties in a positive UTF-8 or any 16-bit or 32_bit
class where \S etc is present without PCRE2_UCP, causing an extended class
to be compiled, we make sure that all characters > 255 are included by
forcing match_all_or_no_wide_chars to be true.

If, when generating an xclass, there are no characters < 256, we can omit
the bitmap in the actual compiled code. */

#ifdef SUPPORT_WIDE_CHARS  /* Defined for 16/32 bits, or 8-bit with Unicode */
if ((xclass_props & XCLASS_REQUIRED) != 0)
  {
  PCRE2_UCHAR *previous = code;

  if ((xclass_props & XCLASS_HAS_CHAR_LISTS) == 0)
    *class_uchardata++ = XCL_END;    /* Marks the end of extra data */
  *code++ = OP_XCLASS;
  code += LINK_SIZE;
  *code = negate_class? XCL_NOT:0;
  if ((xclass_props & XCLASS_HAS_PROPS) != 0) *code |= XCL_HASPROP;

  /* If the map is required, move up the extra data to make room for it;
  otherwise just move the code pointer to the end of the extra data. */

  if ((xclass_props & XCLASS_HAS_8BIT_CHARS) != 0 || has_bitmap != NULL)
    {
    if (negate_class)
      {
      uint32_t *classwords = cb->classbits.classwords;
      for (int i = 0; i < 8; i++) classwords[i] = ~classwords[i];
      }

    if (has_bitmap == NULL)
      {
      *code++ |= XCL_MAP;
      (void)memmove(code + (32 / sizeof(PCRE2_UCHAR)), code,
        CU2BYTES(class_uchardata - code));
      memcpy(code, classbits, 32);
      code = class_uchardata + (32 / sizeof(PCRE2_UCHAR));
      }
    else
      {
      code = class_uchardata;
      if ((xclass_props & XCLASS_HAS_8BIT_CHARS) != 0)
        *has_bitmap = TRUE;
      }
    }
  else code = class_uchardata;

  if ((xclass_props & XCLASS_HAS_CHAR_LISTS) != 0)
    {
    /* Char lists size is an even number, because all items are 16 or 32
    bit values. The character list data is always aligned to 32 bits. */
    size_t char_lists_size = cranges->char_lists_size;
    PCRE2_ASSERT((char_lists_size & 0x1) == 0 &&
                 (cb->char_lists_size & 0x3) == 0);

    if (lengthptr != NULL)
      {
      char_lists_size = CLIST_ALIGN_TO(char_lists_size, sizeof(uint32_t));

#if PCRE2_CODE_UNIT_WIDTH == 8
      *lengthptr += 2 + LINK_SIZE;
#else
      *lengthptr += 1 + LINK_SIZE;
#endif

      cb->char_lists_size += char_lists_size;

      char_lists_size /= sizeof(PCRE2_UCHAR);

      /* Storage space for character lists is included
      in the maximum pattern size. */
      if (*lengthptr > MAX_PATTERN_SIZE ||
          MAX_PATTERN_SIZE - *lengthptr < char_lists_size)
        {
        *errorcodeptr = ERR20;   /* Pattern is too large */
        return NULL;
        }
      }
    else
      {
      uint8_t *data;

      PCRE2_ASSERT(cranges->char_lists_types <= XCL_TYPE_MASK);
#if PCRE2_CODE_UNIT_WIDTH == 8
      /* Encode as high / low bytes. */
      code[0] = (uint8_t)(XCL_LIST |
        (cranges->char_lists_types >> 8));
      code[1] = (uint8_t)cranges->char_lists_types;
      code += 2;
#else
      *code++ = (PCRE2_UCHAR)(XCL_LIST | cranges->char_lists_types);
#endif

      /* Character lists are stored in backwards direction from
      byte code start. The non-dfa/dfa matchers can access these
      lists using the byte code start stored in match blocks.
      Each list is aligned to 32 bit with an optional unused
      16 bit value at the beginning of the character list. */

      cb->char_lists_size += char_lists_size;
      data = (uint8_t*)cb->start_code - cb->char_lists_size;

      memcpy(data, (uint8_t*)(cranges + 1) + cranges->char_lists_start,
        char_lists_size);

      /* Since character lists total size is less than MAX_PATTERN_SIZE,
      their starting offset fits into a value which size is LINK_SIZE. */

      char_lists_size = cb->char_lists_size;
      PUT(code, 0, (uint32_t)(char_lists_size >> 1));
      code += LINK_SIZE;

#if defined PCRE2_DEBUG || defined SUPPORT_VALGRIND
      if ((char_lists_size & 0x2) != 0)
        {
        /* In debug the unused 16 bit value is set
        to a fixed value and marked unused. */
        ((uint16_t*)data)[-1] = 0x5555;
#ifdef SUPPORT_VALGRIND
        VALGRIND_MAKE_MEM_NOACCESS(data - 2, 2);
#endif
        }
#endif

      cb->char_lists_size =
        CLIST_ALIGN_TO(char_lists_size, sizeof(uint32_t));

      cb->cx->memctl.free(cranges, cb->cx->memctl.memory_data);
      }
    }

  /* Now fill in the complete length of the item */

  PUT(previous, 1, (int)(code - previous));
  goto DONE;   /* End of class handling */
  }
#endif  /* SUPPORT_WIDE_CHARS */

/* If there are no characters > 255, or they are all to be included or
excluded, set the opcode to OP_CLASS or OP_NCLASS, depending on whether the
whole class was negated and whether there were negative specials such as \S
(non-UCP) in the class. Then copy the 32-byte map into the code vector,
negating it if necessary. */

if (negate_class)
  {
  uint32_t *classwords = cb->classbits.classwords;

  for (int i = 0; i < 8; i++) classwords[i] = ~classwords[i];
  }

if ((SELECT_VALUE8(!utf, 0) || negate_class != should_flip_negation) &&
    cb->classbits.classwords[0] == ~(uint32_t)0)
  {
  const uint32_t *classwords = cb->classbits.classwords;
  int i;

  for (i = 0; i < 8; i++)
    if (classwords[i] != ~(uint32_t)0) break;

  if (i == 8)
    {
    *code++ = OP_ALLANY;
    goto DONE;   /* End of class handling */
    }
  }

*code++ = (negate_class == should_flip_negation) ? OP_CLASS : OP_NCLASS;
memcpy(code, classbits, 32);
code += 32 / sizeof(PCRE2_UCHAR);

DONE:
*pcode = code;
return pptr - 1;
}



/* ===================================================================*/
/* Here follows a block of ECLASS-compiling functions. You may well want to
read them from top to bottom; they are ordered from leafmost (at the top) to
outermost parser (at the bottom of the file). */

/* This function folds one operand using the negation operator.
The new, combined chunk of stack code is written out to *pop_info. */

static void
fold_negation(eclass_op_info *pop_info, PCRE2_SIZE *lengthptr,
  BOOL preserve_classbits)
{
/* If the chunk of stack code is already composed of multiple ops, we won't
descend in and try and propagate the negation down the tree. (That would lead
to O(n^2) compile-time, which could be exploitable with a malicious regex -
although maybe that's not really too much of a worry in a library that offers
an exponential-time matching function!) */

if (pop_info->op_single_type == 0)
  {
  if (lengthptr != NULL)
    *lengthptr += 1;
  else
    pop_info->code_start[pop_info->length] = ECL_NOT;
  pop_info->length += 1;
  }

/* Otherwise, it's a nice single-op item, so we can easily fold in the negation
without needing to produce an ECL_NOT. */

else if (pop_info->op_single_type == ECL_ANY ||
         pop_info->op_single_type == ECL_NONE)
  {
  pop_info->op_single_type = (pop_info->op_single_type == ECL_NONE)?
      ECL_ANY : ECL_NONE;
  if (lengthptr == NULL)
    *(pop_info->code_start) = pop_info->op_single_type;
  }
else
  {
  PCRE2_ASSERT(pop_info->op_single_type == ECL_XCLASS &&
               pop_info->length >= 1 + LINK_SIZE + 1);
  if (lengthptr == NULL)
    pop_info->code_start[1 + LINK_SIZE] ^= XCL_NOT;
  }

if (!preserve_classbits)
  {
  for (int i = 0; i < 8; i++)
    pop_info->bits.classwords[i] = ~pop_info->bits.classwords[i];
  }
}



/* This function folds together two operands using a binary operator.
The new, combined chunk of stack code is written out to *lhs_op_info. */

static void
fold_binary(int op, eclass_op_info *lhs_op_info, eclass_op_info *rhs_op_info,
  PCRE2_SIZE *lengthptr)
{
switch (op)
  {
  /* ECL_AND truth table:

     LHS  RHS  RESULT
     ----------------
     ANY  *    RHS
     *    ANY  LHS
     NONE *    NONE
     *    NONE NONE
     X    Y    X & Y
  */

  case ECL_AND:
  if (rhs_op_info->op_single_type == ECL_ANY)
    {
    /* no-op: drop the RHS */
    }
  else if (lhs_op_info->op_single_type == ECL_ANY)
    {
    /* no-op: drop the LHS, and memmove the RHS into its place */
    if (lengthptr == NULL)
      memmove(lhs_op_info->code_start, rhs_op_info->code_start,
              CU2BYTES(rhs_op_info->length));
    lhs_op_info->length = rhs_op_info->length;
    lhs_op_info->op_single_type = rhs_op_info->op_single_type;
    }
  else if (rhs_op_info->op_single_type == ECL_NONE)
    {
    /* the result is ECL_NONE: write into the LHS */
    if (lengthptr == NULL)
      lhs_op_info->code_start[0] = ECL_NONE;
    lhs_op_info->length = 1;
    lhs_op_info->op_single_type = ECL_NONE;
    }
  else if (lhs_op_info->op_single_type == ECL_NONE)
    {
    /* the result is ECL_NONE: drop the RHS */
    }
  else
    {
    /* Both of LHS & RHS are either ECL_XCLASS, or compound operations. */
    if (lengthptr != NULL)
      *lengthptr += 1;
    else
      {
      PCRE2_ASSERT(rhs_op_info->code_start ==
          lhs_op_info->code_start + lhs_op_info->length);
      rhs_op_info->code_start[rhs_op_info->length] = ECL_AND;
      }
    lhs_op_info->length += rhs_op_info->length + 1;
    lhs_op_info->op_single_type = 0;
    }

  for (int i = 0; i < 8; i++)
    lhs_op_info->bits.classwords[i] &= rhs_op_info->bits.classwords[i];
  break;

  /* ECL_OR truth table:

     LHS  RHS  RESULT
     ----------------
     ANY  *    ANY
     *    ANY  ANY
     NONE *    RHS
     *    NONE LHS
     X    Y    X | Y
  */

  case ECL_OR:
  if (rhs_op_info->op_single_type == ECL_NONE)
    {
    /* no-op: drop the RHS */
    }
  else if (lhs_op_info->op_single_type == ECL_NONE)
    {
    /* no-op: drop the LHS, and memmove the RHS into its place */
    if (lengthptr == NULL)
      memmove(lhs_op_info->code_start, rhs_op_info->code_start,
              CU2BYTES(rhs_op_info->length));
    lhs_op_info->length = rhs_op_info->length;
    lhs_op_info->op_single_type = rhs_op_info->op_single_type;
    }
  else if (rhs_op_info->op_single_type == ECL_ANY)
    {
    /* the result is ECL_ANY: write into the LHS */
    if (lengthptr == NULL)
      lhs_op_info->code_start[0] = ECL_ANY;
    lhs_op_info->length = 1;
    lhs_op_info->op_single_type = ECL_ANY;
    }
  else if (lhs_op_info->op_single_type == ECL_ANY)
    {
    /* the result is ECL_ANY: drop the RHS */
    }
  else
    {
    /* Both of LHS & RHS are either ECL_XCLASS, or compound operations. */
    if (lengthptr != NULL)
      *lengthptr += 1;
    else
      {
      PCRE2_ASSERT(rhs_op_info->code_start ==
          lhs_op_info->code_start + lhs_op_info->length);
      rhs_op_info->code_start[rhs_op_info->length] = ECL_OR;
      }
    lhs_op_info->length += rhs_op_info->length + 1;
    lhs_op_info->op_single_type = 0;
    }

  for (int i = 0; i < 8; i++)
    lhs_op_info->bits.classwords[i] |= rhs_op_info->bits.classwords[i];
  break;

  /* ECL_XOR truth table:

     LHS  RHS  RESULT
     ----------------
     ANY  *    !RHS
     *    ANY  !LHS
     NONE *    RHS
     *    NONE LHS
     X    Y    X ^ Y
  */

  case ECL_XOR:
  if (rhs_op_info->op_single_type == ECL_NONE)
    {
    /* no-op: drop the RHS */
    }
  else if (lhs_op_info->op_single_type == ECL_NONE)
    {
    /* no-op: drop the LHS, and memmove the RHS into its place */
    if (lengthptr == NULL)
      memmove(lhs_op_info->code_start, rhs_op_info->code_start,
              CU2BYTES(rhs_op_info->length));
    lhs_op_info->length = rhs_op_info->length;
    lhs_op_info->op_single_type = rhs_op_info->op_single_type;
    }
  else if (rhs_op_info->op_single_type == ECL_ANY)
    {
    /* the result is !LHS: fold in the negation, and drop the RHS */
    /* Preserve the classbits, because we promise to deal with them later. */
    fold_negation(lhs_op_info, lengthptr, TRUE);
    }
  else if (lhs_op_info->op_single_type == ECL_ANY)
    {
    /* the result is !RHS: drop the LHS, memmove the RHS into its place, and
    fold in the negation */
    if (lengthptr == NULL)
      memmove(lhs_op_info->code_start, rhs_op_info->code_start,
              CU2BYTES(rhs_op_info->length));
    lhs_op_info->length = rhs_op_info->length;
    lhs_op_info->op_single_type = rhs_op_info->op_single_type;

    /* Preserve the classbits, because we promise to deal with them later. */
    fold_negation(lhs_op_info, lengthptr, TRUE);
    }
  else
    {
    /* Both of LHS & RHS are either ECL_XCLASS, or compound operations. */
    if (lengthptr != NULL)
      *lengthptr += 1;
    else
      {
      PCRE2_ASSERT(rhs_op_info->code_start ==
          lhs_op_info->code_start + lhs_op_info->length);
      rhs_op_info->code_start[rhs_op_info->length] = ECL_XOR;
      }
    lhs_op_info->length += rhs_op_info->length + 1;
    lhs_op_info->op_single_type = 0;
    }

  for (int i = 0; i < 8; i++)
    lhs_op_info->bits.classwords[i] ^= rhs_op_info->bits.classwords[i];
  break;

  /* LCOV_EXCL_START */
  default:
  PCRE2_DEBUG_UNREACHABLE();
  break;
  /* LCOV_EXCL_STOP */
  }
}



static BOOL
compile_eclass_nested(eclass_context *context, BOOL negated,
  uint32_t **pptr, PCRE2_UCHAR **pcode,
  eclass_op_info *pop_info, PCRE2_SIZE *lengthptr);

/* This function consumes a group of implicitly-unioned class elements.
These can be characters, ranges, properties, or nested classes, as long
as they are all joined by being placed adjacently. */

static BOOL
compile_class_operand(eclass_context *context, BOOL negated,
  uint32_t **pptr, PCRE2_UCHAR **pcode, eclass_op_info *pop_info,
  PCRE2_SIZE *lengthptr)
{
uint32_t *ptr = *pptr;
uint32_t *prev_ptr;
PCRE2_UCHAR *code = *pcode;
PCRE2_UCHAR *code_start = code;
PCRE2_SIZE prev_length = (lengthptr != NULL)? *lengthptr : 0;
PCRE2_SIZE extra_length;
uint32_t meta = META_CODE(*ptr);

switch (meta)
  {
  case META_CLASS_EMPTY_NOT:
  case META_CLASS_EMPTY:
  ++ptr;
  pop_info->length = 1;
  if ((meta == META_CLASS_EMPTY) == negated)
    {
    *code++ = pop_info->op_single_type = ECL_ANY;
    memset(pop_info->bits.classbits, 0xff, 32);
    }
  else
    {
    *code++ = pop_info->op_single_type = ECL_NONE;
    memset(pop_info->bits.classbits, 0, 32);
    }
  break;

  case META_CLASS:
  case META_CLASS_NOT:
  if ((*ptr & CLASS_IS_ECLASS) != 0)
    {
    if (!compile_eclass_nested(context, negated, &ptr, &code,
                               pop_info, lengthptr))
      return FALSE;

    PCRE2_ASSERT(*ptr == META_CLASS_END);
    ptr++;
    goto DONE;
    }

  ptr++;
  PCRE2_FALLTHROUGH /* Fall through */

  default:
  /* Scan forward characters, ranges, and properties.
  For example: inside [a-z_ -- m] we don't have brackets around "a-z_" but
  we still need to collect that fragment up into a "leaf" OP_CLASS. */

  prev_ptr = ptr;
  ptr = PRIV(compile_class_not_nested)(
    context->options, context->xoptions, ptr, &code,
    (meta != META_CLASS_NOT) == negated, &context->needs_bitmap,
    context->errorcodeptr, context->cb, lengthptr);
  if (ptr == NULL) return FALSE;

  /* We must have a 100% guarantee that ptr increases when
  compile_class_operand() returns, even on Release builds, so that we can
  statically prove our loops terminate. */
  /* LCOV_EXCL_START */
  if (ptr <= prev_ptr)
    {
    PCRE2_DEBUG_UNREACHABLE();
    return FALSE;
    }
  /* LCOV_EXCL_STOP */

  /* If we fell through above, consume the closing ']'. */
  if (meta == META_CLASS || meta == META_CLASS_NOT)
    {
    PCRE2_ASSERT(*ptr == META_CLASS_END);
    ptr++;
    }

  /* Regardless of whether (lengthptr == NULL), some data will still be written
  out to *pcode, which we need: we have to peek at it, to transform the opcode
  into the ECLASS version (since we need to hoist up the bitmaps). */
  PCRE2_ASSERT(code > code_start);
  extra_length = (lengthptr != NULL)? *lengthptr - prev_length : 0;

  /* Easiest case: convert OP_ALLANY to ECL_ANY */

  if (*code_start == OP_ALLANY)
    {
    PCRE2_ASSERT(code - code_start == 1 && extra_length == 0);
    pop_info->length = 1;
    *code_start = pop_info->op_single_type = ECL_ANY;
    memset(pop_info->bits.classbits, 0xff, 32);
    }

  /* For OP_CLASS and OP_NCLASS, we hoist out the bitmap and convert to
  ECL_NONE / ECL_ANY respectively. */

  else if (*code_start == OP_CLASS || *code_start == OP_NCLASS)
    {
    PCRE2_ASSERT(code - code_start == 1 + 32 / sizeof(PCRE2_UCHAR) &&
                 extra_length == 0);
    pop_info->length = 1;
    *code_start = pop_info->op_single_type =
        (*code_start == OP_CLASS)? ECL_NONE : ECL_ANY;
    memcpy(pop_info->bits.classbits, code_start + 1, 32);
    /* Rewind the code pointer, but make sure we adjust *lengthptr, because we
    do need to reserve that space (even though we only use it temporarily). */
    if (lengthptr != NULL)
      *lengthptr += code - (code_start + 1);
    code = code_start + 1;

    if (!context->needs_bitmap && *code_start == ECL_NONE)
      {
      uint32_t *classwords = pop_info->bits.classwords;

      for (int i = 0; i < 8; i++)
        if (classwords[i] != 0)
          {
          context->needs_bitmap = TRUE;
          break;
          }
      }
    else
      context->needs_bitmap = TRUE;
    }

  /* Finally, for OP_XCLASS we hoist out the bitmap (if any), and convert to
  ECL_XCLASS. */

  else
    {
    PCRE2_ASSERT(*code_start == OP_XCLASS);
    *code_start = pop_info->op_single_type = ECL_XCLASS;

    PCRE2_ASSERT(code - code_start >= 1 + LINK_SIZE + 1);

    memcpy(pop_info->bits.classbits, context->cb->classbits.classbits, 32);
    pop_info->length = (code - code_start) + extra_length;
    }

  break;
  }  /* End of switch(meta) */

pop_info->code_start = (lengthptr == NULL)? code_start : NULL;

if (lengthptr != NULL)
  {
  *lengthptr += code - code_start;
  code = code_start;
  }

DONE:
PCRE2_ASSERT(lengthptr == NULL || (code == code_start));

*pptr = ptr;
*pcode = code;
return TRUE;
}



/* This function consumes a group of implicitly-unioned class elements.
These can be characters, ranges, properties, or nested classes, as long
as they are all joined by being placed adjacently. */

static BOOL
compile_class_juxtaposition(eclass_context *context, BOOL negated,
  uint32_t **pptr, PCRE2_UCHAR **pcode, eclass_op_info *pop_info,
  PCRE2_SIZE *lengthptr)
{
uint32_t *ptr = *pptr;
PCRE2_UCHAR *code = *pcode;
#ifdef PCRE2_DEBUG
PCRE2_UCHAR *start_code = *pcode;
#endif

/* See compile_class_binary_loose() for comments on compile-time folding of
the "negated" flag. */

/* Because it's a non-empty class, there must be an operand at the start. */
if (!compile_class_operand(context, negated, &ptr, &code, pop_info, lengthptr))
  return FALSE;

while (*ptr != META_CLASS_END &&
       !(*ptr >= META_ECLASS_AND && *ptr <= META_ECLASS_NOT))
  {
  uint32_t op;
  BOOL rhs_negated;
  eclass_op_info rhs_op_info;

  if (negated)
    {
    /* !(A juxtapose B)  ->  !A && !B */
    op = ECL_AND;
    rhs_negated = TRUE;
    }
  else
    {
    /* A juxtapose B  ->  A || B */
    op = ECL_OR;
    rhs_negated = FALSE;
    }

  /* An operand must follow the operator. */
  if (!compile_class_operand(context, rhs_negated, &ptr, &code,
                             &rhs_op_info, lengthptr))
    return FALSE;

  /* Convert infix to postfix (RPN). */
  fold_binary(op, pop_info, &rhs_op_info, lengthptr);
  if (lengthptr == NULL)
    code = pop_info->code_start + pop_info->length;
  }

PCRE2_ASSERT(lengthptr == NULL || code == start_code);

*pptr = ptr;
*pcode = code;
return TRUE;
}



/* This function consumes unary prefix operators. */

static BOOL
compile_class_unary(eclass_context *context, BOOL negated,
  uint32_t **pptr, PCRE2_UCHAR **pcode, eclass_op_info *pop_info,
  PCRE2_SIZE *lengthptr)
{
uint32_t *ptr = *pptr;
#ifdef PCRE2_DEBUG
PCRE2_UCHAR *start_code = *pcode;
#endif

while (*ptr == META_ECLASS_NOT)
  {
  ++ptr;
  negated = !negated;
  }

*pptr = ptr;
/* Because it's a non-empty class, there must be an operand. */
if (!compile_class_juxtaposition(context, negated, pptr, pcode,
                                 pop_info, lengthptr))
  return FALSE;

PCRE2_ASSERT(lengthptr == NULL || *pcode == start_code);
return TRUE;
}



/* This function consumes tightly-binding binary operators. */

static BOOL
compile_class_binary_tight(eclass_context *context, BOOL negated,
  uint32_t **pptr, PCRE2_UCHAR **pcode, eclass_op_info *pop_info,
  PCRE2_SIZE *lengthptr)
{
uint32_t *ptr = *pptr;
PCRE2_UCHAR *code = *pcode;
#ifdef PCRE2_DEBUG
PCRE2_UCHAR *start_code = *pcode;
#endif

/* See compile_class_binary_loose() for comments on compile-time folding of
the "negated" flag. */

/* Because it's a non-empty class, there must be an operand at the start. */
if (!compile_class_unary(context, negated, &ptr, &code, pop_info, lengthptr))
  return FALSE;

while (*ptr == META_ECLASS_AND)
  {
  uint32_t op;
  BOOL rhs_negated;
  eclass_op_info rhs_op_info;

  if (negated)
    {
    /* !(A && B)  ->  !A || !B */
    op = ECL_OR;
    rhs_negated = TRUE;
    }
  else
    {
    /* A && B  ->  A && B */
    op = ECL_AND;
    rhs_negated = FALSE;
    }

  ++ptr;

  /* An operand must follow the operator. */
  if (!compile_class_unary(context, rhs_negated, &ptr, &code,
                           &rhs_op_info, lengthptr))
    return FALSE;

  /* Convert infix to postfix (RPN). */
  fold_binary(op, pop_info, &rhs_op_info, lengthptr);
  if (lengthptr == NULL)
    code = pop_info->code_start + pop_info->length;
  }

PCRE2_ASSERT(lengthptr == NULL || code == start_code);

*pptr = ptr;
*pcode = code;
return TRUE;
}



/* This function consumes loosely-binding binary operators. */

static BOOL
compile_class_binary_loose(eclass_context *context, BOOL negated,
  uint32_t **pptr, PCRE2_UCHAR **pcode, eclass_op_info *pop_info,
  PCRE2_SIZE *lengthptr)
{
uint32_t *ptr = *pptr;
PCRE2_UCHAR *code = *pcode;
#ifdef PCRE2_DEBUG
PCRE2_UCHAR *start_code = *pcode;
#endif

/* We really want to fold the negation operator, if at all possible, so that
simple cases can be reduced down. In particular, in 8-bit no-UTF mode, we want
to produce a fully-folded expression, so that we can guarantee not to emit any
OP_ECLASS codes (in the same way that we never emit OP_XCLASS in this mode).

This has the consequence that with a little ingenuity, we can in fact avoid
emitting (nearly...) all cases of the "NOT" operator. Imagine that we have:
    !(A ...
We have parsed the preceding "!", and we are about to parse the "A" operand. We
don't know yet whether there will even be a following binary operand! Both of
these are possibilities for what follows:
    !(A && B)
    !(A)
However, we can still fold the "!" into the "A" operand, because no matter what
the following binary operator will be, we can produce an expression which is
equivalent. */

/* Because it's a non-empty class, there must be an operand at the start. */
if (!compile_class_binary_tight(context, negated, &ptr, &code,
                                pop_info, lengthptr))
  return FALSE;

while (*ptr >= META_ECLASS_OR && *ptr <= META_ECLASS_XOR)
  {
  uint32_t op;
  BOOL op_neg;
  BOOL rhs_negated;
  eclass_op_info rhs_op_info;

  if (negated)
    {
    /* The whole expression is being negated; we respond by unconditionally
    negating the LHS A, before seeing what follows. And hooray! We can recover,
    no matter what follows. */
    /* !(A || B)   ->  !A && !B                     */
    /* !(A -- B)   ->  !(A && !B)    ->  !A || B    */
    /* !(A XOR B)  ->  !(!A XOR !B)  ->  !A XNOR !B */
    op = (*ptr == META_ECLASS_OR )? ECL_AND :
         (*ptr == META_ECLASS_SUB)? ECL_OR  :
         /*ptr == META_ECLASS_XOR*/ ECL_XOR;
    op_neg = (*ptr == META_ECLASS_XOR);
    rhs_negated = *ptr != META_ECLASS_SUB;
    }
  else
    {
    /* A || B   ->  A || B  */
    /* A -- B   ->  A && !B */
    /* A XOR B  ->  A XOR B */
    op = (*ptr == META_ECLASS_OR )? ECL_OR  :
         (*ptr == META_ECLASS_SUB)? ECL_AND :
         /*ptr == META_ECLASS_XOR*/ ECL_XOR;
    op_neg = FALSE;
    rhs_negated = *ptr == META_ECLASS_SUB;
    }

  ++ptr;

  /* An operand must follow the operator. */
  if (!compile_class_binary_tight(context, rhs_negated, &ptr, &code,
                                  &rhs_op_info, lengthptr))
    return FALSE;

  /* Convert infix to postfix (RPN). */
  fold_binary(op, pop_info, &rhs_op_info, lengthptr);
  if (op_neg) fold_negation(pop_info, lengthptr, FALSE);
  if (lengthptr == NULL)
    code = pop_info->code_start + pop_info->length;
  }

PCRE2_ASSERT(lengthptr == NULL || code == start_code);

*pptr = ptr;
*pcode = code;
return TRUE;
}



/* This function converts the META codes in pptr into opcodes written to
pcode. The pptr must start at a META_CLASS or META_CLASS_NOT.

The class is compiled as a left-associative sequence of operator
applications.

The pptr will be left pointing at the matching META_CLASS_END. */

static BOOL
compile_eclass_nested(eclass_context *context, BOOL negated,
  uint32_t **pptr, PCRE2_UCHAR **pcode,
  eclass_op_info *pop_info, PCRE2_SIZE *lengthptr)
{
uint32_t *ptr = *pptr;
#ifdef PCRE2_DEBUG
PCRE2_UCHAR *start_code = *pcode;
#endif

/* The CLASS_IS_ECLASS bit must be set since it is a nested class. */
PCRE2_ASSERT(*ptr == (META_CLASS | CLASS_IS_ECLASS) ||
             *ptr == (META_CLASS_NOT | CLASS_IS_ECLASS));

if (*ptr++ == (META_CLASS_NOT | CLASS_IS_ECLASS))
  negated = !negated;

(*pptr)++;

/* Because it's a non-empty class, there must be an operand at the start. */
if (!compile_class_binary_loose(context, negated, pptr, pcode,
                                pop_info, lengthptr))
  return FALSE;

PCRE2_ASSERT(**pptr == META_CLASS_END);
PCRE2_ASSERT(lengthptr == NULL || *pcode == start_code);
return TRUE;
}

BOOL
PRIV(compile_class_nested)(uint32_t options, uint32_t xoptions,
  uint32_t **pptr, PCRE2_UCHAR **pcode, int *errorcodeptr,
  compile_block *cb, PCRE2_SIZE *lengthptr)
{
eclass_context context;
eclass_op_info op_info;
PCRE2_SIZE previous_length = (lengthptr != NULL)? *lengthptr : 0;
PCRE2_UCHAR *code = *pcode;
PCRE2_UCHAR *previous;
BOOL allbitsone = TRUE;

context.needs_bitmap = FALSE;
context.options = options;
context.xoptions = xoptions;
context.errorcodeptr = errorcodeptr;
context.cb = cb;

previous = code;
*code++ = OP_ECLASS;
code += LINK_SIZE;
*code++ = 0;  /* Flags, currently zero. */
if (!compile_eclass_nested(&context, FALSE, pptr, &code, &op_info, lengthptr))
  return FALSE;

if (lengthptr != NULL)
  {
  *lengthptr += code - previous;
  code = previous;
  /* (*lengthptr - previous_length) now holds the amount of buffer that
  we require to make the call to compile_class_nested() with
  lengthptr = NULL, and including the (1+LINK_SIZE+1) that we write out
  before that call. */
  }

/* Do some useful counting of what's in the bitmap. */
for (int i = 0; i < 8; i++)
  if (op_info.bits.classwords[i] != 0xffffffff)
    {
    allbitsone = FALSE;
    break;
    }

/* After constant-folding the extended class syntax, it may turn out to be
a simple class after all. In that case, we can unwrap it from the
OP_ECLASS container - and in fact, we must do so, because in 8-bit
no-Unicode mode the matcher is compiled without support for OP_ECLASS. */

#ifndef SUPPORT_WIDE_CHARS
PCRE2_ASSERT(op_info.op_single_type != 0);
#else
if (op_info.op_single_type != 0)
#endif
  {
  /* Rewind back over the OP_ECLASS. */
  code = previous;

  /* If the bits are all ones, and the "high characters" are all matched
  too, we use a special-cased encoding of OP_ALLANY. */

  if (op_info.op_single_type == ECL_ANY && allbitsone)
    {
    /* Advancing code means rewinding lengthptr, at this point. */
    if (lengthptr != NULL) *lengthptr -= 1;
    *code++ = OP_ALLANY;
    }

  /* If the high bits are all matched / all not-matched, then we emit an
  OP_NCLASS/OP_CLASS respectively. */

  else if (op_info.op_single_type == ECL_ANY ||
           op_info.op_single_type == ECL_NONE)
    {
    PCRE2_SIZE required_len = 1 + (32 / sizeof(PCRE2_UCHAR));

    if (lengthptr != NULL)
      {
      if (required_len > (*lengthptr - previous_length))
      *lengthptr = previous_length + required_len;
      }

    /* Advancing code means rewinding lengthptr, at this point. */
    if (lengthptr != NULL) *lengthptr -= required_len;
    *code++ = (op_info.op_single_type == ECL_ANY)? OP_NCLASS : OP_CLASS;
    memcpy(code, op_info.bits.classbits, 32);
    code += 32 / sizeof(PCRE2_UCHAR);
    }

  /* Otherwise, we have an ECL_XCLASS, so we have the OP_XCLASS data
  there, but, we pulled out its bitmap into op_info, so now we have to
  put that back into the OP_XCLASS. */

  else
    {
#ifndef SUPPORT_WIDE_CHARS
    PCRE2_DEBUG_UNREACHABLE();
#else
    BOOL need_map = context.needs_bitmap;
    PCRE2_SIZE required_len;

    PCRE2_ASSERT(op_info.op_single_type == ECL_XCLASS);
    required_len = op_info.length + (need_map? 32/sizeof(PCRE2_UCHAR) : 0);

    if (lengthptr != NULL)
      {
      /* Don't unconditionally request all the space we need - we may
      already have asked for more during processing of the ECLASS. */
      if (required_len > (*lengthptr - previous_length))
        *lengthptr = previous_length + required_len;

      /* The code we write out here won't be ignored, even during the
      (lengthptr != NULL) phase, because if there's a following quantifier
      it will peek backwards. So we do have to write out a (truncated)
      OP_XCLASS, even on this branch. */
      *lengthptr -= 1 + LINK_SIZE + 1;
      *code++ = OP_XCLASS;
      PUT(code, 0, 1 + LINK_SIZE + 1);
      code += LINK_SIZE;
      *code++ = 0;
      }
    else
      {
      PCRE2_UCHAR *rest;
      PCRE2_SIZE rest_len;
      PCRE2_UCHAR flags;

      /* 1 unit: OP_XCLASS | LINK_SIZE units | 1 unit: flags | ...rest */
      PCRE2_ASSERT(op_info.length >= 1 + LINK_SIZE + 1);
      rest = op_info.code_start + 1 + LINK_SIZE + 1;
      rest_len = (op_info.code_start + op_info.length) - rest;

      /* First read any data we use, before memmove splats it. */
      flags = op_info.code_start[1 + LINK_SIZE];
      PCRE2_ASSERT((flags & XCL_MAP) == 0);

      /* Next do the memmove before any writes. */
      memmove(code + 1 + LINK_SIZE + 1 + (need_map? 32/sizeof(PCRE2_UCHAR) : 0),
              rest, CU2BYTES(rest_len));

      /* Finally write the header data. */
      *code++ = OP_XCLASS;
      PUT(code, 0, (int)required_len);
      code += LINK_SIZE;
      *code++ = flags | (need_map? XCL_MAP : 0);
      if (need_map)
        {
        memcpy(code, op_info.bits.classbits, 32);
        code += 32 / sizeof(PCRE2_UCHAR);
        }
      code += rest_len;
      }
#endif /* SUPPORT_WIDE_CHARS */
    }
  }

/* Otherwise, we're going to keep the OP_ECLASS. However, again we need
to do some adjustment to insert the bitmap if we have one. */

#ifdef SUPPORT_WIDE_CHARS
else
  {
  BOOL need_map = context.needs_bitmap;
  PCRE2_SIZE required_len =
    1 + LINK_SIZE + 1 + (need_map? 32/sizeof(PCRE2_UCHAR) : 0) + op_info.length;

  if (lengthptr != NULL)
    {
    if (required_len > (*lengthptr - previous_length))
      *lengthptr = previous_length + required_len;

    /* As for the XCLASS branch above, we do have to write out a dummy
    OP_ECLASS, because of the backwards peek by the quantifier code. Write
    out a (truncated) OP_ECLASS, even on this branch. */
    *lengthptr -= 1 + LINK_SIZE + 1;
    *code++ = OP_ECLASS;
    PUT(code, 0, 1 + LINK_SIZE + 1);
    code += LINK_SIZE;
    *code++ = 0;
    }
  else
    {
    if (need_map)
      {
      PCRE2_UCHAR *map_start = previous + 1 + LINK_SIZE + 1;
      previous[1 + LINK_SIZE] |= ECL_MAP;
      memmove(map_start + 32/sizeof(PCRE2_UCHAR), map_start,
              CU2BYTES(code - map_start));
      memcpy(map_start, op_info.bits.classbits, 32);
      code += 32 / sizeof(PCRE2_UCHAR);
      }
    PUT(previous, 1, (int)(code - previous));
    }
  }
#endif /* SUPPORT_WIDE_CHARS */

*pcode = code;
return TRUE;
}

/* End of pcre2_compile_class.c */
