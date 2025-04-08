/*************************************************
*      Perl-Compatible Regular Expressions       *
*************************************************/

/* PCRE is a library of functions to support regular expressions whose syntax
and semantics are as close as possible to those of the Perl 5 language.

                       Written by Philip Hazel
                    This module by Zoltan Herczeg
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

/* XClass matching code. */

#ifdef SUPPORT_WIDE_CHARS

#define ECLASS_CHAR_DATA STACK_TOP
#define ECLASS_STACK_DATA STACK_LIMIT

#define SET_CHAR_OFFSET(value) \
  if ((value) != charoffset) \
    { \
    if ((value) < charoffset) \
      OP2(SLJIT_ADD, TMP1, 0, TMP1, 0, SLJIT_IMM, (sljit_sw)(charoffset - (value))); \
    else \
      OP2(SLJIT_SUB, TMP1, 0, TMP1, 0, SLJIT_IMM, (sljit_sw)((value) - charoffset)); \
    } \
  charoffset = (value);

#define READ_FROM_CHAR_LIST(destination) \
  if (list_ind <= 1) \
    { \
    destination = *(const uint16_t*)next_char; \
    next_char += 2; \
    } \
  else \
    { \
    destination = *(const uint32_t*)next_char; \
    next_char += 4; \
    }

#define XCLASS_LOCAL_RANGES_SIZE 32
#define XCLASS_LOCAL_RANGES_LOG2_SIZE 5

typedef struct xclass_stack_item {
  sljit_u32 first_item;
  sljit_u32 last_item;
  struct sljit_jump *jump;
} xclass_stack_item;

typedef struct xclass_ranges {
  size_t range_count;
  /* Pointer to ranges. A stack area is provided when a small buffer is enough. */
  uint32_t *ranges;
  uint32_t local_ranges[XCLASS_LOCAL_RANGES_SIZE * 2];
  /* Stack size must be log2(ranges / 2). */
  xclass_stack_item *stack;
  xclass_stack_item local_stack[XCLASS_LOCAL_RANGES_LOG2_SIZE];
} xclass_ranges;

static void xclass_compute_ranges(compiler_common *common, PCRE2_SPTR cc, xclass_ranges *ranges)
{
DEFINE_COMPILER;
size_t range_count = 0, est_range_count;
size_t est_stack_size, tmp;
uint32_t type, list_ind;
uint32_t est_type;
uint32_t char_list_add, range_start, range_end;
const uint8_t *next_char;
const uint8_t *est_next_char;
#if defined SUPPORT_UNICODE && (PCRE2_CODE_UNIT_WIDTH == 8 || PCRE2_CODE_UNIT_WIDTH == 16)
BOOL utf = common->utf;
#endif /* SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH == [8|16] */

if (*cc == XCL_SINGLE || *cc == XCL_RANGE)
  {
  /* Only a few ranges are present. */
  do
    {
    type = *cc++;
    SLJIT_ASSERT(type == XCL_SINGLE || type == XCL_RANGE);
    GETCHARINCTEST(range_end, cc);
    ranges->ranges[range_count] = range_end;

    if (type == XCL_RANGE)
      {
      GETCHARINCTEST(range_end, cc);
      }

    ranges->ranges[range_count + 1] = range_end;
    range_count += 2;
    }
  while (*cc != XCL_END);

  SLJIT_ASSERT(range_count <= XCLASS_LOCAL_RANGES_SIZE);
  ranges->range_count = range_count;
  return;
  }

SLJIT_ASSERT(cc[0] >= XCL_LIST);
#if PCRE2_CODE_UNIT_WIDTH == 8
type = (uint32_t)(cc[0] << 8) | cc[1];
cc += 2;
#else
type = cc[0];
cc++;
#endif  /* CODE_UNIT_WIDTH */

/* Align characters. */
next_char = (const uint8_t*)common->start - (GET(cc, 0) << 1);
type &= XCL_TYPE_MASK;

/* Estimate size. */
est_next_char = next_char;
est_type = type;
est_range_count = 0;
list_ind = 0;

while (est_type > 0)
  {
  uint32_t item_count = est_type & XCL_ITEM_COUNT_MASK;

  if (item_count == XCL_ITEM_COUNT_MASK)
    {
    if (list_ind <= 1)
      {
      item_count = *(const uint16_t*)est_next_char;
      est_next_char += 2;
      }
    else
      {
      item_count = *(const uint32_t*)est_next_char;
      est_next_char += 4;
      }
    }

  est_type >>= XCL_TYPE_BIT_LEN;
  est_next_char += (size_t)item_count << (list_ind <= 1 ? 1 : 2);
  list_ind++;
  est_range_count += item_count + 1;
  }

if (est_range_count > XCLASS_LOCAL_RANGES_SIZE)
  {
  est_stack_size = 0;
  tmp = est_range_count - 1;

  /* Compute log2(est_range_count) */
  while (tmp > 0)
    {
    est_stack_size++;
    tmp >>= 1;
    }

  ranges->stack = (xclass_stack_item*)SLJIT_MALLOC((sizeof(xclass_stack_item) * est_stack_size)
    + ((sizeof(uint32_t) << 1) * (size_t)est_range_count), compiler->allocator_data);

  if (ranges->stack == NULL)
    {
    sljit_set_compiler_memory_error(compiler);
    ranges->ranges = NULL;
    return;
    }

  ranges->ranges = (uint32_t*)(ranges->stack + est_stack_size);
  }

char_list_add = XCL_CHAR_LIST_LOW_16_ADD;
range_start = ~(uint32_t)0;
list_ind = 0;

if ((type & XCL_BEGIN_WITH_RANGE) != 0)
  range_start = XCL_CHAR_LIST_LOW_16_START;

while (type > 0)
  {
  uint32_t item_count = type & XCL_ITEM_COUNT_MASK;

  if (item_count == XCL_ITEM_COUNT_MASK)
    {
    READ_FROM_CHAR_LIST(item_count);
    SLJIT_ASSERT(item_count >= XCL_ITEM_COUNT_MASK);
    }

  while (item_count > 0)
    {
    READ_FROM_CHAR_LIST(range_end);

    if ((range_end & XCL_CHAR_END) != 0)
      {
      range_end = char_list_add + (range_end >> XCL_CHAR_SHIFT);

      if (range_start == ~(uint32_t)0)
        range_start = range_end;

      ranges->ranges[range_count] = range_start;
      ranges->ranges[range_count + 1] = range_end;
      range_count += 2;
      range_start = ~(uint32_t)0;
      }
    else
      range_start = char_list_add + (range_end >> XCL_CHAR_SHIFT);

    item_count--;
    }

  list_ind++;
  type >>= XCL_TYPE_BIT_LEN;

  if (range_start == ~(uint32_t)0)
    {
    if ((type & XCL_BEGIN_WITH_RANGE) != 0)
      {
      if (list_ind == 1) range_start = XCL_CHAR_LIST_HIGH_16_START;
#if PCRE2_CODE_UNIT_WIDTH == 32
      else if (list_ind == 2) range_start = XCL_CHAR_LIST_LOW_32_START;
      else range_start = XCL_CHAR_LIST_HIGH_32_START;
#else
      else range_start = XCL_CHAR_LIST_LOW_32_START;
#endif
      }
    }
  else if ((type & XCL_BEGIN_WITH_RANGE) == 0)
    {
    if (list_ind == 1) range_end = XCL_CHAR_LIST_LOW_16_END;
    else if (list_ind == 2) range_end = XCL_CHAR_LIST_HIGH_16_END;
#if PCRE2_CODE_UNIT_WIDTH == 32
    else if (list_ind == 3) range_end = XCL_CHAR_LIST_LOW_32_END;
    else range_end = XCL_CHAR_LIST_HIGH_32_END;
#else
    else range_end = XCL_CHAR_LIST_LOW_32_END;
#endif

    ranges->ranges[range_count] = range_start;
    ranges->ranges[range_count + 1] = range_end;
    range_count += 2;
    range_start = ~(uint32_t)0;
    }

  if (list_ind == 1) char_list_add = XCL_CHAR_LIST_HIGH_16_ADD;
#if PCRE2_CODE_UNIT_WIDTH == 32
  else if (list_ind == 2) char_list_add = XCL_CHAR_LIST_LOW_32_ADD;
  else char_list_add = XCL_CHAR_LIST_HIGH_32_ADD;
#else
  else char_list_add = XCL_CHAR_LIST_LOW_32_ADD;
#endif
  }

SLJIT_ASSERT(range_count > 0 && range_count <= (est_range_count << 1));
SLJIT_ASSERT(next_char <= (const uint8_t*)common->start);
ranges->range_count = range_count;
}

static void xclass_check_bitset(compiler_common *common, const sljit_u8 *bitset, jump_list **found, jump_list **backtracks)
{
DEFINE_COMPILER;
struct sljit_jump *jump;

jump = CMP(SLJIT_GREATER, TMP1, 0, SLJIT_IMM, 255);
if (!optimize_class(common, bitset, (bitset[31] & 0x80) != 0, TRUE, found))
  {
  OP2(SLJIT_AND, TMP2, 0, TMP1, 0, SLJIT_IMM, 0x7);
  OP2(SLJIT_LSHR, TMP1, 0, TMP1, 0, SLJIT_IMM, 3);
  OP1(SLJIT_MOV_U8, TMP1, 0, SLJIT_MEM1(TMP1), (sljit_sw)bitset);
  OP2(SLJIT_SHL, TMP2, 0, SLJIT_IMM, 1, TMP2, 0);
  OP2U(SLJIT_AND | SLJIT_SET_Z, TMP1, 0, TMP2, 0);
  add_jump(compiler, found, JUMP(SLJIT_NOT_ZERO));
  }

add_jump(compiler, backtracks, JUMP(SLJIT_JUMP));
JUMPHERE(jump);
}

#if defined SUPPORT_UNICODE && (PCRE2_CODE_UNIT_WIDTH == 8 || PCRE2_CODE_UNIT_WIDTH == 16)

static void xclass_update_min_max(compiler_common *common, PCRE2_SPTR cc, sljit_u32 *min_ptr, sljit_u32 *max_ptr)
{
uint32_t type, list_ind, c;
sljit_u32 min = *min_ptr;
sljit_u32 max = *max_ptr;
uint32_t char_list_add;
const uint8_t *next_char;
BOOL utf = TRUE;

/* This function is pointless without utf 8/16. */
SLJIT_ASSERT(common->utf);
if (*cc == XCL_SINGLE || *cc == XCL_RANGE)
  {
  /* Only a few ranges are present. */
  do
    {
    type = *cc++;
    SLJIT_ASSERT(type == XCL_SINGLE || type == XCL_RANGE);
    GETCHARINCTEST(c, cc);

    if (c < min)
      min = c;

    if (type == XCL_RANGE)
      {
      GETCHARINCTEST(c, cc);
      }

    if (c > max)
      max = c;
    }
  while (*cc != XCL_END);

  SLJIT_ASSERT(min <= MAX_UTF_CODE_POINT && max <= MAX_UTF_CODE_POINT && min <= max);
  *min_ptr = min;
  *max_ptr = max;
  return;
  }

SLJIT_ASSERT(cc[0] >= XCL_LIST);
#if PCRE2_CODE_UNIT_WIDTH == 8
type = (uint32_t)(cc[0] << 8) | cc[1];
cc += 2;
#else
type = cc[0];
cc++;
#endif  /* CODE_UNIT_WIDTH */

/* Align characters. */
next_char = (const uint8_t*)common->start - (GET(cc, 0) << 1);
type &= XCL_TYPE_MASK;

SLJIT_ASSERT(type != 0);

/* Detect minimum. */

/* Skip unused ranges. */
list_ind = 0;
while ((type & (XCL_BEGIN_WITH_RANGE | XCL_ITEM_COUNT_MASK)) == 0)
  {
  type >>= XCL_TYPE_BIT_LEN;
  list_ind++;
  }

SLJIT_ASSERT(list_ind <= 2);
switch (list_ind)
  {
  case 0:
  char_list_add = XCL_CHAR_LIST_LOW_16_ADD;
  c = XCL_CHAR_LIST_LOW_16_START;
  break;

  case 1:
  char_list_add = XCL_CHAR_LIST_HIGH_16_ADD;
  c = XCL_CHAR_LIST_HIGH_16_START;
  break;

  default:
  char_list_add = XCL_CHAR_LIST_LOW_32_ADD;
  c = XCL_CHAR_LIST_LOW_32_START;
  break;
  }

if ((type & XCL_BEGIN_WITH_RANGE) != 0)
  {
  if (c < min)
    min = c;
  }
else
  {
  if ((type & XCL_ITEM_COUNT_MASK) == XCL_ITEM_COUNT_MASK)
    {
    if (list_ind <= 1)
      c = *(const uint16_t*)(next_char + 2);
    else
      c = *(const uint32_t*)(next_char + 4);
    }
  else
    {
    if (list_ind <= 1)
      c = *(const uint16_t*)next_char;
    else
      c = *(const uint32_t*)next_char;
    }

  c = char_list_add + (c >> XCL_CHAR_SHIFT);
  if (c < min)
    min = c;
  }

/* Detect maximum. */

/* Skip intermediate ranges. */
while (TRUE)
  {
  if ((type & XCL_ITEM_COUNT_MASK) == XCL_ITEM_COUNT_MASK)
    {
    if (list_ind <= 1)
      {
      c = *(const uint16_t*)next_char;
      next_char += (c + 1) << 1;
      }
    else
      {
      c = *(const uint32_t*)next_char;
      next_char += (c + 1) << 2;
      }
    }
  else
    next_char += (type & XCL_ITEM_COUNT_MASK) << (list_ind <= 1 ? 1 : 2);

  if ((type >> XCL_TYPE_BIT_LEN) == 0)
    break;

  list_ind++;
  type >>= XCL_TYPE_BIT_LEN;
  }

SLJIT_ASSERT(list_ind <= 2 && type != 0);
switch (list_ind)
  {
  case 0:
  char_list_add = XCL_CHAR_LIST_LOW_16_ADD;
  c = XCL_CHAR_LIST_LOW_16_END;
  break;

  case 1:
  char_list_add = XCL_CHAR_LIST_HIGH_16_ADD;
  c = XCL_CHAR_LIST_HIGH_16_END;
  break;

  default:
  char_list_add = XCL_CHAR_LIST_LOW_32_ADD;
  c = XCL_CHAR_LIST_LOW_32_END;
  break;
  }

if ((type & XCL_ITEM_COUNT_MASK) != 0)
  {
  /* Type is reused as temporary. */
  if (list_ind <= 1)
    type = *(const uint16_t*)(next_char - 2);
  else
    type = *(const uint32_t*)(next_char - 4);

  if (type & XCL_CHAR_END)
    c = char_list_add + (type >> XCL_CHAR_SHIFT);
  }

if (c > max)
  max = c;

SLJIT_ASSERT(min <= MAX_UTF_CODE_POINT && max <= MAX_UTF_CODE_POINT && min <= max);
*min_ptr = min;
*max_ptr = max;
}

#endif /* SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH == [8|16] */

#define XCLASS_IS_ECLASS 0x001
#ifdef SUPPORT_UNICODE
#define XCLASS_SAVE_CHAR 0x002
#define XCLASS_HAS_TYPE 0x004
#define XCLASS_HAS_SCRIPT 0x008
#define XCLASS_HAS_SCRIPT_EXTENSION 0x010
#define XCLASS_HAS_BOOL 0x020
#define XCLASS_HAS_BIDICL 0x040
#define XCLASS_NEEDS_UCD (XCLASS_HAS_TYPE | XCLASS_HAS_SCRIPT | XCLASS_HAS_SCRIPT_EXTENSION | XCLASS_HAS_BOOL | XCLASS_HAS_BIDICL)
#define XCLASS_SCRIPT_EXTENSION_NOTPROP 0x080
#define XCLASS_SCRIPT_EXTENSION_RESTORE_RETURN_ADDR 0x100
#define XCLASS_SCRIPT_EXTENSION_RESTORE_LOCAL0 0x200
#endif /* SUPPORT_UNICODE */

static PCRE2_SPTR compile_char1_matchingpath(compiler_common *common, PCRE2_UCHAR type, PCRE2_SPTR cc, jump_list **backtracks, BOOL check_str_ptr);

/* TMP3 must be preserved because it is used by compile_iterator_matchingpath. */
static void compile_xclass_matchingpath(compiler_common *common, PCRE2_SPTR cc, jump_list **backtracks, sljit_u32 status)
{
DEFINE_COMPILER;
jump_list *found = NULL;
jump_list *check_result = NULL;
jump_list **list = (cc[0] & XCL_NOT) == 0 ? &found : backtracks;
sljit_uw c, charoffset;
sljit_u32 max = READ_CHAR_MAX, min = 0;
struct sljit_jump *jump = NULL;
PCRE2_UCHAR flags;
PCRE2_SPTR ccbegin;
sljit_u32 compares, invertcmp, depth;
sljit_u32 first_item, last_item, mid_item;
sljit_u32 range_start, range_end;
xclass_ranges ranges;
BOOL has_cmov, last_range_set;

#ifdef SUPPORT_UNICODE
sljit_u32 category_list = 0;
sljit_u32 items;
int typereg = TMP1;
#endif /* SUPPORT_UNICODE */

SLJIT_ASSERT(common->locals_size >= SSIZE_OF(sw));
/* Scanning the necessary info. */
flags = *cc++;
ccbegin = cc;
compares = 0;

if (flags & XCL_MAP)
  cc += 32 / sizeof(PCRE2_UCHAR);

#ifdef SUPPORT_UNICODE
while (*cc == XCL_PROP || *cc == XCL_NOTPROP)
  {
  compares++;
  cc++;

  items = 0;

  switch(*cc)
    {
    case PT_LAMP:
    items = UCPCAT3(ucp_Lu, ucp_Ll, ucp_Lt);
    break;

    case PT_GC:
    items = UCPCAT_RANGE(PRIV(ucp_typerange)[(int)cc[1] * 2], PRIV(ucp_typerange)[(int)cc[1] * 2 + 1]);
    break;

    case PT_PC:
    items = UCPCAT(cc[1]);
    break;

    case PT_WORD:
    items = UCPCAT2(ucp_Mn, ucp_Pc) | UCPCAT_L | UCPCAT_N;
    break;

    case PT_ALNUM:
    items = UCPCAT_L | UCPCAT_N;
    break;

    case PT_SCX:
    status |= XCLASS_HAS_SCRIPT_EXTENSION;
    if (cc[-1] == XCL_NOTPROP)
      {
      status |= XCLASS_SCRIPT_EXTENSION_NOTPROP;
      break;
      }
    compares++;
    /* Fall through */

    case PT_SC:
    status |= XCLASS_HAS_SCRIPT;
    break;

    case PT_SPACE:
    case PT_PXSPACE:
    case PT_PXGRAPH:
    case PT_PXPRINT:
    case PT_PXPUNCT:
    status |= XCLASS_SAVE_CHAR | XCLASS_HAS_TYPE;
    break;

    case PT_UCNC:
    case PT_PXXDIGIT:
    status |= XCLASS_SAVE_CHAR;
    break;

    case PT_BOOL:
    status |= XCLASS_HAS_BOOL;
    break;

    case PT_BIDICL:
    status |= XCLASS_HAS_BIDICL;
    break;

    default:
    SLJIT_UNREACHABLE();
    break;
    }

  if (items > 0)
    {
    if (cc[-1] == XCL_NOTPROP)
      items ^= UCPCAT_ALL;
    category_list |= items;
    status |= XCLASS_HAS_TYPE;
    compares--;
    }

  cc += 2;
  }

if (category_list == UCPCAT_ALL)
  {
  /* All or no characters are accepted, same as dotall. */
  if (status & XCLASS_IS_ECLASS)
    {
    if (list != backtracks)
      OP2(SLJIT_OR, ECLASS_STACK_DATA, 0, ECLASS_STACK_DATA, 0, SLJIT_IMM, 1);
    return;
    }

  compile_char1_matchingpath(common, OP_ALLANY, cc, backtracks, FALSE);
  if (list == backtracks)
    add_jump(compiler, backtracks, JUMP(SLJIT_JUMP));
  return;
  }

if (category_list != 0)
  compares++;
#endif

if (*cc != XCL_END)
  {
#if defined SUPPORT_UNICODE && (PCRE2_CODE_UNIT_WIDTH == 8 || PCRE2_CODE_UNIT_WIDTH == 16)
  if (common->utf && compares == 0 && !(status & XCLASS_IS_ECLASS))
    {
    SLJIT_ASSERT(category_list == 0);
    max = 0;
    min = (flags & XCL_MAP) != 0 ? 0 : READ_CHAR_MAX;
    xclass_update_min_max(common, cc, &min, &max);
    }
#endif
  compares++;
#ifdef SUPPORT_UNICODE
  status |= XCLASS_SAVE_CHAR;
#endif /* SUPPORT_UNICODE */
  }

#ifdef SUPPORT_UNICODE
SLJIT_ASSERT(compares > 0 || category_list != 0);
#else /* !SUPPORT_UNICODE */
SLJIT_ASSERT(compares > 0);
#endif /* SUPPORT_UNICODE */

/* We are not necessary in utf mode even in 8 bit mode. */
cc = ccbegin;
if (!(status & XCLASS_IS_ECLASS))
  {
  if ((flags & XCL_NOT) != 0)
    read_char(common, min, max, backtracks, READ_CHAR_UPDATE_STR_PTR);
  else
    {
#ifdef SUPPORT_UNICODE
    read_char(common, min, max, (status & XCLASS_NEEDS_UCD) ? backtracks : NULL, 0);
#else /* !SUPPORT_UNICODE */
    read_char(common, min, max, NULL, 0);
#endif /* SUPPORT_UNICODE */
    }
  }

if ((flags & XCL_MAP) != 0)
  {
  SLJIT_ASSERT(!(status & XCLASS_IS_ECLASS));
  xclass_check_bitset(common, (const sljit_u8 *)cc, &found, backtracks);
  cc += 32 / sizeof(PCRE2_UCHAR);
  }

#ifdef SUPPORT_UNICODE
if (status & XCLASS_NEEDS_UCD)
  {
  if ((status & (XCLASS_SAVE_CHAR | XCLASS_IS_ECLASS)) == XCLASS_SAVE_CHAR)
    OP1(SLJIT_MOV, RETURN_ADDR, 0, TMP1, 0);

#if PCRE2_CODE_UNIT_WIDTH == 32
  if (!common->utf)
    {
    OP2U(SLJIT_SUB | SLJIT_SET_GREATER_EQUAL, TMP1, 0, SLJIT_IMM, MAX_UTF_CODE_POINT + 1);
    SELECT(SLJIT_GREATER_EQUAL, TMP1, SLJIT_IMM, UNASSIGNED_UTF_CHAR, TMP1);
    }
#endif /* PCRE2_CODE_UNIT_WIDTH == 32 */

  OP2(SLJIT_LSHR, TMP2, 0, TMP1, 0, SLJIT_IMM, UCD_BLOCK_SHIFT);
  OP2(SLJIT_SHL, TMP2, 0, TMP2, 0, SLJIT_IMM, 1);
  OP1(SLJIT_MOV_U16, TMP2, 0, SLJIT_MEM1(TMP2), (sljit_sw)PRIV(ucd_stage1));
  OP2(SLJIT_AND, TMP1, 0, TMP1, 0, SLJIT_IMM, UCD_BLOCK_MASK);
  OP2(SLJIT_SHL, TMP2, 0, TMP2, 0, SLJIT_IMM, UCD_BLOCK_SHIFT);
  OP2(SLJIT_ADD, TMP1, 0, TMP1, 0, TMP2, 0);
  OP1(SLJIT_MOV, TMP2, 0, SLJIT_IMM, (sljit_sw)PRIV(ucd_stage2));
  OP1(SLJIT_MOV_U16, TMP2, 0, SLJIT_MEM2(TMP2, TMP1), 1);
  OP2(SLJIT_SHL, TMP1, 0, TMP2, 0, SLJIT_IMM, 3);
  OP2(SLJIT_SHL, TMP2, 0, TMP2, 0, SLJIT_IMM, 2);
  OP2(SLJIT_ADD, TMP2, 0, TMP2, 0, TMP1, 0);

  ccbegin = cc;

  if (status & XCLASS_HAS_BIDICL)
    {
    OP1(SLJIT_MOV_U16, TMP1, 0, SLJIT_MEM1(TMP2), (sljit_sw)PRIV(ucd_records) + SLJIT_OFFSETOF(ucd_record, scriptx_bidiclass));
    OP2(SLJIT_LSHR, TMP1, 0, TMP1, 0, SLJIT_IMM, UCD_BIDICLASS_SHIFT);

    while (*cc == XCL_PROP || *cc == XCL_NOTPROP)
      {
      cc++;

      if (*cc == PT_BIDICL)
        {
        compares--;
        invertcmp = (compares == 0 && list != backtracks);
        if (cc[-1] == XCL_NOTPROP)
          invertcmp ^= 0x1;
        jump = CMP(SLJIT_EQUAL ^ invertcmp, TMP1, 0, SLJIT_IMM, (int)cc[1]);
        add_jump(compiler, compares > 0 ? list : backtracks, jump);
        }
      cc += 2;
      }

    cc = ccbegin;
    }

  if (status & XCLASS_HAS_BOOL)
    {
    OP1(SLJIT_MOV_U16, TMP1, 0, SLJIT_MEM1(TMP2), (sljit_sw)PRIV(ucd_records) + SLJIT_OFFSETOF(ucd_record, bprops));
    OP2(SLJIT_AND, TMP1, 0, TMP1, 0, SLJIT_IMM, UCD_BPROPS_MASK);
    OP2(SLJIT_SHL, TMP1, 0, TMP1, 0, SLJIT_IMM, 2);

    while (*cc == XCL_PROP || *cc == XCL_NOTPROP)
      {
      cc++;
      if (*cc == PT_BOOL)
        {
        compares--;
        invertcmp = (compares == 0 && list != backtracks);
        if (cc[-1] == XCL_NOTPROP)
          invertcmp ^= 0x1;

        OP2U(SLJIT_AND32 | SLJIT_SET_Z, SLJIT_MEM1(TMP1), (sljit_sw)(PRIV(ucd_boolprop_sets) + (cc[1] >> 5)), SLJIT_IMM, (sljit_sw)(1u << (cc[1] & 0x1f)));
        add_jump(compiler, compares > 0 ? list : backtracks, JUMP(SLJIT_NOT_ZERO ^ invertcmp));
        }
      cc += 2;
      }

    cc = ccbegin;
    }

  if (status & XCLASS_HAS_SCRIPT)
    {
    OP1(SLJIT_MOV_U8, TMP1, 0, SLJIT_MEM1(TMP2), (sljit_sw)PRIV(ucd_records) + SLJIT_OFFSETOF(ucd_record, script));

    while (*cc == XCL_PROP || *cc == XCL_NOTPROP)
      {
      cc++;

      switch (*cc)
        {
        case PT_SCX:
        if (cc[-1] == XCL_NOTPROP)
          break;
        /* Fall through */

        case PT_SC:
        compares--;
        invertcmp = (compares == 0 && list != backtracks);
        if (cc[-1] == XCL_NOTPROP)
          invertcmp ^= 0x1;

        add_jump(compiler, compares > 0 ? list : backtracks, CMP(SLJIT_EQUAL ^ invertcmp, TMP1, 0, SLJIT_IMM, (int)cc[1]));
        }
      cc += 2;
      }

    cc = ccbegin;
    }

  if (status & XCLASS_HAS_SCRIPT_EXTENSION)
    {
    OP1(SLJIT_MOV_U16, TMP1, 0, SLJIT_MEM1(TMP2), (sljit_sw)PRIV(ucd_records) + SLJIT_OFFSETOF(ucd_record, scriptx_bidiclass));
    OP2(SLJIT_AND, TMP1, 0, TMP1, 0, SLJIT_IMM, UCD_SCRIPTX_MASK);
    OP2(SLJIT_SHL, TMP1, 0, TMP1, 0, SLJIT_IMM, 2);

    if (status & XCLASS_SCRIPT_EXTENSION_NOTPROP)
      {
      if (status & XCLASS_HAS_TYPE)
        {
        if ((status & (XCLASS_SAVE_CHAR | XCLASS_IS_ECLASS)) == XCLASS_SAVE_CHAR)
          {
          OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), LOCAL0, TMP2, 0);
          status |= XCLASS_SCRIPT_EXTENSION_RESTORE_LOCAL0;
          }
        else
          {
          OP1(SLJIT_MOV, RETURN_ADDR, 0, TMP2, 0);
          status |= XCLASS_SCRIPT_EXTENSION_RESTORE_RETURN_ADDR;
          }
        }
      OP1(SLJIT_MOV_U8, TMP2, 0, SLJIT_MEM1(TMP2), (sljit_sw)PRIV(ucd_records) + SLJIT_OFFSETOF(ucd_record, script));
      }

    while (*cc == XCL_PROP || *cc == XCL_NOTPROP)
      {
      cc++;

      if (*cc == PT_SCX)
        {
        compares--;
        invertcmp = (compares == 0 && list != backtracks);

        jump = NULL;
        if (cc[-1] == XCL_NOTPROP)
          {
          jump = CMP(SLJIT_EQUAL, TMP2, 0, SLJIT_IMM, (int)cc[1]);
          if (invertcmp)
            {
            add_jump(compiler, backtracks, jump);
            jump = NULL;
            }
          invertcmp ^= 0x1;
          }

        OP2U(SLJIT_AND32 | SLJIT_SET_Z, SLJIT_MEM1(TMP1), (sljit_sw)(PRIV(ucd_script_sets) + (cc[1] >> 5)), SLJIT_IMM, (sljit_sw)(1u << (cc[1] & 0x1f)));
        add_jump(compiler, compares > 0 ? list : backtracks, JUMP(SLJIT_NOT_ZERO ^ invertcmp));

        if (jump != NULL)
          JUMPHERE(jump);
        }
      cc += 2;
      }

    if (status & XCLASS_SCRIPT_EXTENSION_RESTORE_LOCAL0)
      OP1(SLJIT_MOV, TMP2, 0, SLJIT_MEM1(SLJIT_SP), LOCAL0);
    else if (status & XCLASS_SCRIPT_EXTENSION_RESTORE_RETURN_ADDR)
      OP1(SLJIT_MOV, TMP2, 0, RETURN_ADDR, 0);
    cc = ccbegin;
    }

  if (status & XCLASS_SAVE_CHAR)
    OP1(SLJIT_MOV, TMP1, 0, (status & XCLASS_IS_ECLASS) ? ECLASS_CHAR_DATA : RETURN_ADDR, 0);

  if (status & XCLASS_HAS_TYPE)
    {
    if (status & XCLASS_SAVE_CHAR)
      typereg = RETURN_ADDR;

    OP1(SLJIT_MOV_U8, TMP2, 0, SLJIT_MEM1(TMP2), (sljit_sw)PRIV(ucd_records) + SLJIT_OFFSETOF(ucd_record, chartype));
    OP2(SLJIT_SHL, typereg, 0, SLJIT_IMM, 1, TMP2, 0);

    if (category_list > 0)
      {
      compares--;
      invertcmp = (compares == 0 && list != backtracks);
      OP2U(SLJIT_AND | SLJIT_SET_Z, typereg, 0, SLJIT_IMM, category_list);
      add_jump(compiler, compares > 0 ? list : backtracks, JUMP(SLJIT_NOT_ZERO ^ invertcmp));
      }
    }
  }
#endif /* SUPPORT_UNICODE */

/* Generating code. */
charoffset = 0;

#ifdef SUPPORT_UNICODE
while (*cc == XCL_PROP || *cc == XCL_NOTPROP)
  {
  compares--;
  invertcmp = (compares == 0 && list != backtracks);
  jump = NULL;

  if (*cc == XCL_NOTPROP)
    invertcmp ^= 0x1;
  cc++;
  switch(*cc)
    {
    case PT_LAMP:
    case PT_GC:
    case PT_PC:
    case PT_SC:
    case PT_SCX:
    case PT_BOOL:
    case PT_BIDICL:
    case PT_WORD:
    case PT_ALNUM:
    compares++;
    /* Already handled. */
    break;

    case PT_SPACE:
    case PT_PXSPACE:
    SET_CHAR_OFFSET(9);
    OP2U(SLJIT_SUB | SLJIT_SET_LESS_EQUAL, TMP1, 0, SLJIT_IMM, 0xd - 0x9);
    OP_FLAGS(SLJIT_MOV, TMP2, 0, SLJIT_LESS_EQUAL);

    OP2U(SLJIT_SUB | SLJIT_SET_Z, TMP1, 0, SLJIT_IMM, 0x85 - 0x9);
    OP_FLAGS(SLJIT_OR, TMP2, 0, SLJIT_EQUAL);

    OP2U(SLJIT_SUB | SLJIT_SET_Z, TMP1, 0, SLJIT_IMM, 0x180e - 0x9);
    OP_FLAGS(SLJIT_OR, TMP2, 0, SLJIT_EQUAL);

    OP2U(SLJIT_AND | SLJIT_SET_Z, typereg, 0, SLJIT_IMM, UCPCAT_RANGE(ucp_Zl, ucp_Zs));
    OP_FLAGS(SLJIT_OR | SLJIT_SET_Z, TMP2, 0, SLJIT_NOT_ZERO);
    jump = JUMP(SLJIT_NOT_ZERO ^ invertcmp);
    break;

    case PT_UCNC:
    OP2U(SLJIT_SUB | SLJIT_SET_Z, TMP1, 0, SLJIT_IMM, (sljit_sw)(CHAR_DOLLAR_SIGN - charoffset));
    OP_FLAGS(SLJIT_MOV, TMP2, 0, SLJIT_EQUAL);
    OP2U(SLJIT_SUB | SLJIT_SET_Z, TMP1, 0, SLJIT_IMM, (sljit_sw)(CHAR_COMMERCIAL_AT - charoffset));
    OP_FLAGS(SLJIT_OR, TMP2, 0, SLJIT_EQUAL);
    OP2U(SLJIT_SUB | SLJIT_SET_Z, TMP1, 0, SLJIT_IMM, (sljit_sw)(CHAR_GRAVE_ACCENT - charoffset));
    OP_FLAGS(SLJIT_OR, TMP2, 0, SLJIT_EQUAL);

    SET_CHAR_OFFSET(0xa0);
    OP2U(SLJIT_SUB | SLJIT_SET_LESS_EQUAL, TMP1, 0, SLJIT_IMM, (sljit_sw)(0xd7ff - charoffset));
    OP_FLAGS(SLJIT_OR, TMP2, 0, SLJIT_LESS_EQUAL);
    SET_CHAR_OFFSET(0);
    OP2U(SLJIT_SUB | SLJIT_SET_GREATER_EQUAL, TMP1, 0, SLJIT_IMM, 0xe000 - 0);
    OP_FLAGS(SLJIT_OR | SLJIT_SET_Z, TMP2, 0, SLJIT_GREATER_EQUAL);
    jump = JUMP(SLJIT_NOT_ZERO ^ invertcmp);
    break;

    case PT_PXGRAPH:
    OP2U(SLJIT_AND | SLJIT_SET_Z, typereg, 0, SLJIT_IMM, UCPCAT_RANGE(ucp_Cc, ucp_Cs) | UCPCAT_RANGE(ucp_Zl, ucp_Zs));
    OP_FLAGS(SLJIT_MOV, TMP2, 0, SLJIT_NOT_ZERO);

    OP2U(SLJIT_AND | SLJIT_SET_Z, typereg, 0, SLJIT_IMM, UCPCAT(ucp_Cf));
    jump = JUMP(SLJIT_ZERO);

    c = charoffset;
    /* In case of ucp_Cf, we overwrite the result. */
    SET_CHAR_OFFSET(0x2066);
    OP2U(SLJIT_SUB | SLJIT_SET_LESS_EQUAL, TMP1, 0, SLJIT_IMM, 0x2069 - 0x2066);
    OP_FLAGS(SLJIT_MOV, TMP2, 0, SLJIT_LESS_EQUAL);

    OP2U(SLJIT_SUB | SLJIT_SET_Z, TMP1, 0, SLJIT_IMM, 0x061c - 0x2066);
    OP_FLAGS(SLJIT_OR, TMP2, 0, SLJIT_EQUAL);

    OP2U(SLJIT_SUB | SLJIT_SET_Z, TMP1, 0, SLJIT_IMM, 0x180e - 0x2066);
    OP_FLAGS(SLJIT_OR, TMP2, 0, SLJIT_EQUAL);

    /* Restore charoffset. */
    SET_CHAR_OFFSET(c);

    JUMPHERE(jump);
    jump = CMP(SLJIT_ZERO ^ invertcmp, TMP2, 0, SLJIT_IMM, 0);
    break;

    case PT_PXPRINT:
    OP2U(SLJIT_AND | SLJIT_SET_Z, typereg, 0, SLJIT_IMM, UCPCAT_RANGE(ucp_Cc, ucp_Cs) | UCPCAT2(ucp_Zl, ucp_Zp));
    OP_FLAGS(SLJIT_MOV, TMP2, 0, SLJIT_NOT_ZERO);

    OP2U(SLJIT_AND | SLJIT_SET_Z, typereg, 0, SLJIT_IMM, UCPCAT(ucp_Cf));
    jump = JUMP(SLJIT_ZERO);

    c = charoffset;
    /* In case of ucp_Cf, we overwrite the result. */
    SET_CHAR_OFFSET(0x2066);
    OP2U(SLJIT_SUB | SLJIT_SET_LESS_EQUAL, TMP1, 0, SLJIT_IMM, 0x2069 - 0x2066);
    OP_FLAGS(SLJIT_MOV, TMP2, 0, SLJIT_LESS_EQUAL);

    OP2U(SLJIT_SUB | SLJIT_SET_Z, TMP1, 0, SLJIT_IMM, 0x061c - 0x2066);
    OP_FLAGS(SLJIT_OR, TMP2, 0, SLJIT_EQUAL);

    /* Restore charoffset. */
    SET_CHAR_OFFSET(c);

    JUMPHERE(jump);
    jump = CMP(SLJIT_ZERO ^ invertcmp, TMP2, 0, SLJIT_IMM, 0);
    break;

    case PT_PXPUNCT:
    OP2U(SLJIT_AND | SLJIT_SET_Z, typereg, 0, SLJIT_IMM, UCPCAT_RANGE(ucp_Sc, ucp_So));
    OP_FLAGS(SLJIT_MOV, TMP2, 0, SLJIT_NOT_ZERO);

    SET_CHAR_OFFSET(0);
    OP2U(SLJIT_SUB | SLJIT_SET_LESS_EQUAL, TMP1, 0, SLJIT_IMM, 0x7f);
    OP_FLAGS(SLJIT_AND, TMP2, 0, SLJIT_LESS_EQUAL);

    OP2U(SLJIT_AND | SLJIT_SET_Z, typereg, 0, SLJIT_IMM, UCPCAT_RANGE(ucp_Pc, ucp_Ps));
    OP_FLAGS(SLJIT_OR | SLJIT_SET_Z, TMP2, 0, SLJIT_NOT_ZERO);
    jump = JUMP(SLJIT_NOT_ZERO ^ invertcmp);
    break;

    case PT_PXXDIGIT:
    SET_CHAR_OFFSET(CHAR_A);
    OP2(SLJIT_AND, TMP2, 0, TMP1, 0, SLJIT_IMM, ~0x20);
    OP2U(SLJIT_SUB | SLJIT_SET_LESS_EQUAL, TMP2, 0, SLJIT_IMM, CHAR_F - CHAR_A);
    OP_FLAGS(SLJIT_MOV, TMP2, 0, SLJIT_LESS_EQUAL);

    SET_CHAR_OFFSET(CHAR_0);
    OP2U(SLJIT_SUB | SLJIT_SET_LESS_EQUAL, TMP1, 0, SLJIT_IMM, CHAR_9 - CHAR_0);
    OP_FLAGS(SLJIT_OR, TMP2, 0, SLJIT_LESS_EQUAL);

    SET_CHAR_OFFSET(0xff10);
    jump = CMP(SLJIT_GREATER, TMP1, 0, SLJIT_IMM, 0xff46 - 0xff10);

    OP2U(SLJIT_SUB | SLJIT_SET_LESS_EQUAL, TMP1, 0, SLJIT_IMM, 0xff19 - 0xff10);
    OP_FLAGS(SLJIT_OR, TMP2, 0, SLJIT_LESS_EQUAL);

    SET_CHAR_OFFSET(0xff21);
    OP2U(SLJIT_SUB | SLJIT_SET_LESS_EQUAL, TMP1, 0, SLJIT_IMM, 0xff26 - 0xff21);
    OP_FLAGS(SLJIT_OR, TMP2, 0, SLJIT_LESS_EQUAL);

    SET_CHAR_OFFSET(0xff41);
    OP2U(SLJIT_SUB | SLJIT_SET_LESS_EQUAL, TMP1, 0, SLJIT_IMM, 0xff46 - 0xff41);
    OP_FLAGS(SLJIT_OR, TMP2, 0, SLJIT_LESS_EQUAL);

    SET_CHAR_OFFSET(0xff10);

    JUMPHERE(jump);
    OP2U(SLJIT_SUB | SLJIT_SET_Z, TMP2, 0, SLJIT_IMM, 0);
    jump = JUMP(SLJIT_NOT_ZERO ^ invertcmp);
    break;

    default:
    SLJIT_UNREACHABLE();
    break;
    }

  cc += 2;

  if (jump != NULL)
    add_jump(compiler, compares > 0 ? list : backtracks, jump);
  }

if (compares == 0)
  {
  if (found != NULL)
    set_jumps(found, LABEL());

  if (status & XCLASS_IS_ECLASS)
    OP2(SLJIT_OR, ECLASS_STACK_DATA, 0, ECLASS_STACK_DATA, 0, SLJIT_IMM, 1);
  return;
  }
#endif /* SUPPORT_UNICODE */

SLJIT_ASSERT(compares == 1);
ranges.range_count = 0;
ranges.ranges = ranges.local_ranges;
ranges.stack = ranges.local_stack;

xclass_compute_ranges(common, cc, &ranges);

/* Memory error is set for the compiler. */
if (ranges.stack == NULL)
  return;

#if (defined SLJIT_DEBUG && SLJIT_DEBUG) && \
  defined SUPPORT_UNICODE && (PCRE2_CODE_UNIT_WIDTH == 8 || PCRE2_CODE_UNIT_WIDTH == 16)
if (common->utf)
  {
  min = READ_CHAR_MAX;
  max = 0;
  xclass_update_min_max(common, cc, &min, &max);
  SLJIT_ASSERT(ranges.ranges[0] == min && ranges.ranges[ranges.range_count - 1] == max);
  }
#endif /* SLJIT_DEBUG && SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH == [8|16] */

invertcmp = (list != backtracks);

if (ranges.range_count == 2)
  {
  range_start = ranges.ranges[0];
  range_end = ranges.ranges[1];

  if (range_start < range_end)
    {
    SET_CHAR_OFFSET(range_start);
    jump = CMP(SLJIT_LESS_EQUAL ^ invertcmp, TMP1, 0, SLJIT_IMM, (sljit_sw)(range_end - range_start));
    }
  else
    jump = CMP(SLJIT_EQUAL ^ invertcmp, TMP1, 0, SLJIT_IMM, (sljit_sw)(range_start - charoffset));

  add_jump(compiler, backtracks, jump);

  SLJIT_ASSERT(ranges.stack == ranges.local_stack);
  if (found != NULL)
    set_jumps(found, LABEL());

  if (status & XCLASS_IS_ECLASS)
    OP2(SLJIT_OR, ECLASS_STACK_DATA, 0, ECLASS_STACK_DATA, 0, SLJIT_IMM, 1);
  return;
  }

range_start = ranges.ranges[0];
SET_CHAR_OFFSET(range_start);
if (ranges.range_count >= 6)
  {
  /* Early fail. */
  range_end = ranges.ranges[ranges.range_count - 1];
  add_jump(compiler, (flags & XCL_NOT) == 0 ? backtracks : &found,
    CMP(SLJIT_GREATER, TMP1, 0, SLJIT_IMM, (sljit_sw)(range_end - range_start)));
  }

depth = 0;
first_item = 0;
last_item = ranges.range_count - 2;
has_cmov = sljit_has_cpu_feature(SLJIT_HAS_CMOV) != 0;

while (TRUE)
  {
  /* At least two items are present. */
  SLJIT_ASSERT(first_item < last_item && charoffset == ranges.ranges[0]);
  last_range_set = FALSE;

  if (first_item + 6 <= last_item)
    {
    mid_item = ((first_item + last_item) >> 1) & ~(sljit_u32)1;
    SLJIT_ASSERT(last_item >= mid_item + 4);

    range_end = ranges.ranges[mid_item + 1];
    if (first_item + 6 > mid_item && ranges.ranges[mid_item] == range_end)
      {
      OP2U(SLJIT_SUB | SLJIT_SET_GREATER | SLJIT_SET_Z, TMP1, 0, SLJIT_IMM, (sljit_sw)(range_end - charoffset));
      ranges.stack[depth].jump = JUMP(SLJIT_GREATER);
      OP_FLAGS(SLJIT_MOV, TMP2, 0, SLJIT_EQUAL);
      last_range_set = TRUE;
      }
    else
      ranges.stack[depth].jump = CMP(SLJIT_GREATER, TMP1, 0, SLJIT_IMM, (sljit_sw)(range_end - charoffset));

    ranges.stack[depth].first_item = (sljit_u32)(mid_item + 2);
    ranges.stack[depth].last_item = (sljit_u32)last_item;

    depth++;
    SLJIT_ASSERT(ranges.stack == ranges.local_stack ?
      depth <= XCLASS_LOCAL_RANGES_LOG2_SIZE : (ranges.stack + depth) <= (xclass_stack_item*)ranges.ranges);

    last_item = mid_item;
    if (!last_range_set)
      continue;

    last_item -= 2;
    }

  if (!last_range_set)
    {
    range_start = ranges.ranges[first_item];
    range_end = ranges.ranges[first_item + 1];

    if (range_start < range_end)
      {
      SET_CHAR_OFFSET(range_start);
      OP2U(SLJIT_SUB | SLJIT_SET_LESS_EQUAL, TMP1, 0, SLJIT_IMM, (sljit_sw)(range_end - range_start));
      OP_FLAGS(SLJIT_MOV, TMP2, 0, SLJIT_LESS_EQUAL);
      }
    else
      {
      OP2U(SLJIT_SUB | SLJIT_SET_Z, TMP1, 0, SLJIT_IMM, (sljit_sw)(range_start - charoffset));
      OP_FLAGS(SLJIT_MOV, TMP2, 0, SLJIT_EQUAL);
      }
    first_item += 2;
    }

  SLJIT_ASSERT(first_item <= last_item);

  do
    {
    range_start = ranges.ranges[first_item];
    range_end = ranges.ranges[first_item + 1];

    if (range_start < range_end)
      {
      SET_CHAR_OFFSET(range_start);
      OP2U(SLJIT_SUB | SLJIT_SET_LESS_EQUAL, TMP1, 0, SLJIT_IMM, (sljit_sw)(range_end - range_start));

      if (has_cmov)
        SELECT(SLJIT_LESS_EQUAL, TMP2, STR_END, 0, TMP2);
      else
        OP_FLAGS(SLJIT_OR | ((first_item == last_item) ? SLJIT_SET_Z : 0), TMP2, 0, SLJIT_LESS_EQUAL);
      }
    else
      {
      OP2U(SLJIT_SUB | SLJIT_SET_Z, TMP1, 0, SLJIT_IMM, (sljit_sw)(range_start - charoffset));

      if (has_cmov)
        SELECT(SLJIT_EQUAL, TMP2, STR_END, 0, TMP2);
      else
        OP_FLAGS(SLJIT_OR | ((first_item == last_item) ? SLJIT_SET_Z : 0), TMP2, 0, SLJIT_EQUAL);
      }

    first_item += 2;
    }
  while (first_item <= last_item);

  if (depth == 0) break;

  add_jump(compiler, &check_result, JUMP(SLJIT_JUMP));

  /* The charoffset resets after the end of a branch is reached. */
  charoffset = ranges.ranges[0];
  depth--;
  first_item = ranges.stack[depth].first_item;
  last_item = ranges.stack[depth].last_item;
  JUMPHERE(ranges.stack[depth].jump);
  }

if (check_result != NULL)
  set_jumps(check_result, LABEL());

if (has_cmov)
  jump = CMP(SLJIT_NOT_EQUAL ^ invertcmp, TMP2, 0, SLJIT_IMM, 0);
else
  {
  sljit_set_current_flags(compiler, SLJIT_SET_Z);
  jump = JUMP(SLJIT_NOT_EQUAL ^ invertcmp);
  }

add_jump(compiler, backtracks, jump);

if (found != NULL)
  set_jumps(found, LABEL());

if (status & XCLASS_IS_ECLASS)
  OP2(SLJIT_OR, ECLASS_STACK_DATA, 0, ECLASS_STACK_DATA, 0, SLJIT_IMM, 1);

if (ranges.stack != ranges.local_stack)
  SLJIT_FREE(ranges.stack, compiler->allocator_data);
}

static PCRE2_SPTR compile_eclass_matchingpath(compiler_common *common, PCRE2_SPTR cc, jump_list **backtracks)
{
DEFINE_COMPILER;
PCRE2_SPTR end = cc + GET(cc, 0) - 1;
PCRE2_SPTR begin;
jump_list *not_found;
jump_list *found = NULL;

cc += LINK_SIZE;

/* Should be optimized later. */
read_char(common, 0, READ_CHAR_MAX, backtracks, 0);

if (((*cc++) & ECL_MAP) != 0)
  {
  xclass_check_bitset(common, (const sljit_u8 *)cc, &found, backtracks);
  cc += 32 / sizeof(PCRE2_UCHAR);
  }

begin = cc;

OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), LOCAL0, ECLASS_CHAR_DATA, 0);
OP1(SLJIT_MOV, SLJIT_MEM1(SLJIT_SP), LOCAL1, ECLASS_STACK_DATA, 0);
OP1(SLJIT_MOV, ECLASS_STACK_DATA, 0, SLJIT_IMM, 0);
OP1(SLJIT_MOV, ECLASS_CHAR_DATA, 0, TMP1, 0);

/* All eclass must start with an xclass. */
SLJIT_ASSERT(*cc == ECL_XCLASS);

while (cc < end)
  {
  switch (*cc)
    {
    case ECL_AND:
    ++cc;
    OP2(SLJIT_OR, TMP2, 0, ECLASS_STACK_DATA, 0, SLJIT_IMM, ~(sljit_sw)1);
    OP2(SLJIT_LSHR, ECLASS_STACK_DATA, 0, ECLASS_STACK_DATA, 0, SLJIT_IMM, 1);
    OP2(SLJIT_AND, ECLASS_STACK_DATA, 0, ECLASS_STACK_DATA, 0, TMP2, 0);
    break;

    case ECL_OR:
    ++cc;
    OP2(SLJIT_AND, TMP2, 0, ECLASS_STACK_DATA, 0, SLJIT_IMM, 1);
    OP2(SLJIT_LSHR, ECLASS_STACK_DATA, 0, ECLASS_STACK_DATA, 0, SLJIT_IMM, 1);
    OP2(SLJIT_OR, ECLASS_STACK_DATA, 0, ECLASS_STACK_DATA, 0, TMP2, 0);
    break;

    case ECL_XOR:
    ++cc;
    OP2(SLJIT_AND, TMP2, 0, ECLASS_STACK_DATA, 0, SLJIT_IMM, 1);
    OP2(SLJIT_LSHR, ECLASS_STACK_DATA, 0, ECLASS_STACK_DATA, 0, SLJIT_IMM, 1);
    OP2(SLJIT_XOR, ECLASS_STACK_DATA, 0, ECLASS_STACK_DATA, 0, TMP2, 0);
    break;

    case ECL_NOT:
    ++cc;
    OP2(SLJIT_XOR, ECLASS_STACK_DATA, 0, ECLASS_STACK_DATA, 0, SLJIT_IMM, 1);
    break;

    default:
    SLJIT_ASSERT(*cc == ECL_XCLASS);
    if (cc != begin)
      {
      OP1(SLJIT_MOV, TMP1, 0, ECLASS_CHAR_DATA, 0);
      OP2(SLJIT_SHL, ECLASS_STACK_DATA, 0, ECLASS_STACK_DATA, 0, SLJIT_IMM, 1);
      }

    not_found = NULL;
    compile_xclass_matchingpath(common, cc + 1 + LINK_SIZE, &not_found, XCLASS_IS_ECLASS);
    set_jumps(not_found, LABEL());

    cc += GET(cc, 1);
    break;
    }
  }

OP2U(SLJIT_SUB | SLJIT_SET_Z, ECLASS_STACK_DATA, 0, SLJIT_IMM, 0);
OP1(SLJIT_MOV, ECLASS_CHAR_DATA, 0, SLJIT_MEM1(SLJIT_SP), LOCAL0);
OP1(SLJIT_MOV, ECLASS_STACK_DATA, 0, SLJIT_MEM1(SLJIT_SP), LOCAL1);
add_jump(compiler, backtracks, JUMP(SLJIT_EQUAL));
set_jumps(found, LABEL());
return end;
}

/* Generic character matching code. */

#undef SET_CHAR_OFFSET
#undef READ_FROM_CHAR_LIST
#undef XCLASS_LOCAL_RANGES_SIZE
#undef XCLASS_LOCAL_RANGES_LOG2_SIZE

#endif /* SUPPORT_WIDE_CHARS */

static PCRE2_SPTR byte_sequence_compare(compiler_common *common, BOOL caseless, PCRE2_SPTR cc,
    compare_context *context, jump_list **backtracks)
{
DEFINE_COMPILER;
unsigned int othercasebit = 0;
PCRE2_SPTR othercasechar = NULL;
#ifdef SUPPORT_UNICODE
int utflength;
#endif

if (caseless && char_has_othercase(common, cc))
  {
  othercasebit = char_get_othercase_bit(common, cc);
  SLJIT_ASSERT(othercasebit);
  /* Extracting bit difference info. */
#if PCRE2_CODE_UNIT_WIDTH == 8
  othercasechar = cc + (othercasebit >> 8);
  othercasebit &= 0xff;
#elif PCRE2_CODE_UNIT_WIDTH == 16 || PCRE2_CODE_UNIT_WIDTH == 32
  /* Note that this code only handles characters in the BMP. If there
  ever are characters outside the BMP whose othercase differs in only one
  bit from itself (there currently are none), this code will need to be
  revised for PCRE2_CODE_UNIT_WIDTH == 32. */
  othercasechar = cc + (othercasebit >> 9);
  if ((othercasebit & 0x100) != 0)
    othercasebit = (othercasebit & 0xff) << 8;
  else
    othercasebit &= 0xff;
#endif /* PCRE2_CODE_UNIT_WIDTH == [8|16|32] */
  }

if (context->sourcereg == -1)
  {
#if PCRE2_CODE_UNIT_WIDTH == 8
#if defined SLJIT_UNALIGNED && SLJIT_UNALIGNED
  if (context->length >= 4)
    OP1(SLJIT_MOV_S32, TMP1, 0, SLJIT_MEM1(STR_PTR), -context->length);
  else if (context->length >= 2)
    OP1(SLJIT_MOV_U16, TMP1, 0, SLJIT_MEM1(STR_PTR), -context->length);
  else
#endif
    OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), -context->length);
#elif PCRE2_CODE_UNIT_WIDTH == 16
#if defined SLJIT_UNALIGNED && SLJIT_UNALIGNED
  if (context->length >= 4)
    OP1(SLJIT_MOV_S32, TMP1, 0, SLJIT_MEM1(STR_PTR), -context->length);
  else
#endif
    OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), -context->length);
#elif PCRE2_CODE_UNIT_WIDTH == 32
  OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), -context->length);
#endif /* PCRE2_CODE_UNIT_WIDTH == [8|16|32] */
  context->sourcereg = TMP2;
  }

#ifdef SUPPORT_UNICODE
utflength = 1;
if (common->utf && HAS_EXTRALEN(*cc))
  utflength += GET_EXTRALEN(*cc);

do
  {
#endif

  context->length -= IN_UCHARS(1);
#if (defined SLJIT_UNALIGNED && SLJIT_UNALIGNED) && (PCRE2_CODE_UNIT_WIDTH == 8 || PCRE2_CODE_UNIT_WIDTH == 16)

  /* Unaligned read is supported. */
  if (othercasebit != 0 && othercasechar == cc)
    {
    context->c.asuchars[context->ucharptr] = *cc | othercasebit;
    context->oc.asuchars[context->ucharptr] = othercasebit;
    }
  else
    {
    context->c.asuchars[context->ucharptr] = *cc;
    context->oc.asuchars[context->ucharptr] = 0;
    }
  context->ucharptr++;

#if PCRE2_CODE_UNIT_WIDTH == 8
  if (context->ucharptr >= 4 || context->length == 0 || (context->ucharptr == 2 && context->length == 1))
#else
  if (context->ucharptr >= 2 || context->length == 0)
#endif
    {
    if (context->length >= 4)
      OP1(SLJIT_MOV_S32, context->sourcereg, 0, SLJIT_MEM1(STR_PTR), -context->length);
    else if (context->length >= 2)
      OP1(SLJIT_MOV_U16, context->sourcereg, 0, SLJIT_MEM1(STR_PTR), -context->length);
#if PCRE2_CODE_UNIT_WIDTH == 8
    else if (context->length >= 1)
      OP1(SLJIT_MOV_U8, context->sourcereg, 0, SLJIT_MEM1(STR_PTR), -context->length);
#endif /* PCRE2_CODE_UNIT_WIDTH == 8 */
    context->sourcereg = context->sourcereg == TMP1 ? TMP2 : TMP1;

    switch(context->ucharptr)
      {
      case 4 / sizeof(PCRE2_UCHAR):
      if (context->oc.asint != 0)
        OP2(SLJIT_OR, context->sourcereg, 0, context->sourcereg, 0, SLJIT_IMM, context->oc.asint);
      add_jump(compiler, backtracks, CMP(SLJIT_NOT_EQUAL, context->sourcereg, 0, SLJIT_IMM, context->c.asint | context->oc.asint));
      break;

      case 2 / sizeof(PCRE2_UCHAR):
      if (context->oc.asushort != 0)
        OP2(SLJIT_OR, context->sourcereg, 0, context->sourcereg, 0, SLJIT_IMM, context->oc.asushort);
      add_jump(compiler, backtracks, CMP(SLJIT_NOT_EQUAL, context->sourcereg, 0, SLJIT_IMM, context->c.asushort | context->oc.asushort));
      break;

#if PCRE2_CODE_UNIT_WIDTH == 8
      case 1:
      if (context->oc.asbyte != 0)
        OP2(SLJIT_OR, context->sourcereg, 0, context->sourcereg, 0, SLJIT_IMM, context->oc.asbyte);
      add_jump(compiler, backtracks, CMP(SLJIT_NOT_EQUAL, context->sourcereg, 0, SLJIT_IMM, context->c.asbyte | context->oc.asbyte));
      break;
#endif

      default:
      SLJIT_UNREACHABLE();
      break;
      }
    context->ucharptr = 0;
    }

#else

  /* Unaligned read is unsupported or in 32 bit mode. */
  if (context->length >= 1)
    OP1(MOV_UCHAR, context->sourcereg, 0, SLJIT_MEM1(STR_PTR), -context->length);

  context->sourcereg = context->sourcereg == TMP1 ? TMP2 : TMP1;

  if (othercasebit != 0 && othercasechar == cc)
    {
    OP2(SLJIT_OR, context->sourcereg, 0, context->sourcereg, 0, SLJIT_IMM, othercasebit);
    add_jump(compiler, backtracks, CMP(SLJIT_NOT_EQUAL, context->sourcereg, 0, SLJIT_IMM, *cc | othercasebit));
    }
  else
    add_jump(compiler, backtracks, CMP(SLJIT_NOT_EQUAL, context->sourcereg, 0, SLJIT_IMM, *cc));

#endif

  cc++;
#ifdef SUPPORT_UNICODE
  utflength--;
  }
while (utflength > 0);
#endif

return cc;
}

#ifdef SUPPORT_UNICODE

#if PCRE2_CODE_UNIT_WIDTH != 32

/* The code in this function copies the logic of the interpreter function that
is defined in the pcre2_extuni.c source. If that code is updated, this
function, and those below it, must be kept in step (note by PH, June 2024). */

static PCRE2_SPTR SLJIT_FUNC do_extuni_utf(jit_arguments *args, PCRE2_SPTR cc)
{
PCRE2_SPTR start_subject = args->begin;
PCRE2_SPTR end_subject = args->end;
int lgb, rgb, ricount;
PCRE2_SPTR prevcc, endcc, bptr;
BOOL first = TRUE;
BOOL was_ep_ZWJ = FALSE;
uint32_t c;

prevcc = cc;
endcc = NULL;
do
  {
  GETCHARINC(c, cc);
  rgb = UCD_GRAPHBREAK(c);

  if (first)
    {
    lgb = rgb;
    endcc = cc;
    first = FALSE;
    continue;
    }

  if ((PRIV(ucp_gbtable)[lgb] & (1 << rgb)) == 0)
    break;

  /* ZWJ followed by Extended Pictographic is allowed only if the ZWJ was
  preceded by Extended Pictographic. */

  if (lgb == ucp_gbZWJ && rgb == ucp_gbExtended_Pictographic && !was_ep_ZWJ)
    break;

  /* Not breaking between Regional Indicators is allowed only if there
  are an even number of preceding RIs. */

  if (lgb == ucp_gbRegional_Indicator && rgb == ucp_gbRegional_Indicator)
    {
    ricount = 0;
    bptr = prevcc;

    /* bptr is pointing to the left-hand character */
    while (bptr > start_subject)
      {
      bptr--;
      BACKCHAR(bptr);
      GETCHAR(c, bptr);

      if (UCD_GRAPHBREAK(c) != ucp_gbRegional_Indicator)
        break;

      ricount++;
      }

    if ((ricount & 1) != 0) break;  /* Grapheme break required */
    }

  /* Set a flag when ZWJ follows Extended Pictographic (with optional Extend in
  between; see next statement). */

  was_ep_ZWJ = (lgb == ucp_gbExtended_Pictographic && rgb == ucp_gbZWJ);

  /* If Extend follows Extended_Pictographic, do not update lgb; this allows
  any number of them before a following ZWJ. */

  if (rgb != ucp_gbExtend || lgb != ucp_gbExtended_Pictographic)
    lgb = rgb;

  prevcc = endcc;
  endcc = cc;
  }
while (cc < end_subject);

return endcc;
}

#endif /* PCRE2_CODE_UNIT_WIDTH != 32 */

/* The code in this function copies the logic of the interpreter function that
is defined in the pcre2_extuni.c source. If that code is updated, this
function, and the one below it, must be kept in step (note by PH, June 2024). */

static PCRE2_SPTR SLJIT_FUNC do_extuni_utf_invalid(jit_arguments *args, PCRE2_SPTR cc)
{
PCRE2_SPTR start_subject = args->begin;
PCRE2_SPTR end_subject = args->end;
int lgb, rgb, ricount;
PCRE2_SPTR prevcc, endcc, bptr;
BOOL first = TRUE;
BOOL was_ep_ZWJ = FALSE;
uint32_t c;

prevcc = cc;
endcc = NULL;
do
  {
  GETCHARINC_INVALID(c, cc, end_subject, break);
  rgb = UCD_GRAPHBREAK(c);

  if (first)
    {
    lgb = rgb;
    endcc = cc;
    first = FALSE;
    continue;
    }

  if ((PRIV(ucp_gbtable)[lgb] & (1 << rgb)) == 0)
    break;

  /* ZWJ followed by Extended Pictographic is allowed only if the ZWJ was
  preceded by Extended Pictographic. */

  if (lgb == ucp_gbZWJ && rgb == ucp_gbExtended_Pictographic && !was_ep_ZWJ)
    break;

  /* Not breaking between Regional Indicators is allowed only if there
  are an even number of preceding RIs. */

  if (lgb == ucp_gbRegional_Indicator && rgb == ucp_gbRegional_Indicator)
    {
    ricount = 0;
    bptr = prevcc;

    /* bptr is pointing to the left-hand character */
    while (bptr > start_subject)
      {
      GETCHARBACK_INVALID(c, bptr, start_subject, break);

      if (UCD_GRAPHBREAK(c) != ucp_gbRegional_Indicator)
        break;

      ricount++;
      }

    if ((ricount & 1) != 0)
      break;  /* Grapheme break required */
    }

  /* Set a flag when ZWJ follows Extended Pictographic (with optional Extend in
  between; see next statement). */

  was_ep_ZWJ = (lgb == ucp_gbExtended_Pictographic && rgb == ucp_gbZWJ);

  /* If Extend follows Extended_Pictographic, do not update lgb; this allows
  any number of them before a following ZWJ. */

  if (rgb != ucp_gbExtend || lgb != ucp_gbExtended_Pictographic)
    lgb = rgb;

  prevcc = endcc;
  endcc = cc;
  }
while (cc < end_subject);

return endcc;
}

/* The code in this function copies the logic of the interpreter function that
is defined in the pcre2_extuni.c source. If that code is updated, this
function must be kept in step (note by PH, June 2024). */

static PCRE2_SPTR SLJIT_FUNC do_extuni_no_utf(jit_arguments *args, PCRE2_SPTR cc)
{
PCRE2_SPTR start_subject = args->begin;
PCRE2_SPTR end_subject = args->end;
int lgb, rgb, ricount;
PCRE2_SPTR bptr;
uint32_t c;
BOOL was_ep_ZWJ = FALSE;

/* Patch by PH */
/* GETCHARINC(c, cc); */
c = *cc++;

#if PCRE2_CODE_UNIT_WIDTH == 32
if (c >= 0x110000)
  return cc;
#endif /* PCRE2_CODE_UNIT_WIDTH == 32 */
lgb = UCD_GRAPHBREAK(c);

while (cc < end_subject)
  {
  c = *cc;
#if PCRE2_CODE_UNIT_WIDTH == 32
  if (c >= 0x110000)
    break;
#endif /* PCRE2_CODE_UNIT_WIDTH == 32 */
  rgb = UCD_GRAPHBREAK(c);

  if ((PRIV(ucp_gbtable)[lgb] & (1 << rgb)) == 0)
    break;

  /* ZWJ followed by Extended Pictographic is allowed only if the ZWJ was
  preceded by Extended Pictographic. */

  if (lgb == ucp_gbZWJ && rgb == ucp_gbExtended_Pictographic && !was_ep_ZWJ)
    break;

  /* Not breaking between Regional Indicators is allowed only if there
  are an even number of preceding RIs. */

  if (lgb == ucp_gbRegional_Indicator && rgb == ucp_gbRegional_Indicator)
    {
    ricount = 0;
    bptr = cc - 1;

    /* bptr is pointing to the left-hand character */
    while (bptr > start_subject)
      {
      bptr--;
      c = *bptr;
#if PCRE2_CODE_UNIT_WIDTH == 32
      if (c >= 0x110000)
        break;
#endif /* PCRE2_CODE_UNIT_WIDTH == 32 */

      if (UCD_GRAPHBREAK(c) != ucp_gbRegional_Indicator) break;

      ricount++;
      }

    if ((ricount & 1) != 0)
      break;  /* Grapheme break required */
    }

  /* Set a flag when ZWJ follows Extended Pictographic (with optional Extend in
  between; see next statement). */

  was_ep_ZWJ = (lgb == ucp_gbExtended_Pictographic && rgb == ucp_gbZWJ);

  /* If Extend follows Extended_Pictographic, do not update lgb; this allows
  any number of them before a following ZWJ. */

  if (rgb != ucp_gbExtend || lgb != ucp_gbExtended_Pictographic)
    lgb = rgb;

  cc++;
  }

return cc;
}

static void compile_clist(compiler_common *common, PCRE2_SPTR cc, jump_list **backtracks)
{
DEFINE_COMPILER;
const sljit_u32 *other_cases;
struct sljit_jump *jump;
sljit_u32 min = 0, max = READ_CHAR_MAX;
BOOL has_cmov = sljit_has_cpu_feature(SLJIT_HAS_CMOV) != 0;

SLJIT_ASSERT(cc[1] == PT_CLIST);

if (cc[0] == OP_PROP)
  {
  other_cases = PRIV(ucd_caseless_sets) + cc[2];

  min = *other_cases++;
  max = min;

  while (*other_cases != NOTACHAR)
    {
    if (*other_cases > max) max = *other_cases;
    if (*other_cases < min) min = *other_cases;
    other_cases++;
    }
  }

other_cases = PRIV(ucd_caseless_sets) + cc[2];
SLJIT_ASSERT(other_cases[0] != NOTACHAR && other_cases[1] != NOTACHAR);
/* The NOTACHAR is higher than any character. */
SLJIT_ASSERT(other_cases[0] < other_cases[1] && other_cases[1] < other_cases[2]);

read_char(common, min, max, backtracks, READ_CHAR_UPDATE_STR_PTR);

/* At least two characters are required.
   Otherwise this case would be handled by the normal code path. */
/* NOTACHAR is the unsigned maximum. */

/* Optimizing character pairs, if their difference is power of 2. */
if (is_powerof2(other_cases[1] ^ other_cases[0]))
  {
  OP2(SLJIT_OR, TMP2, 0, TMP1, 0, SLJIT_IMM, (sljit_sw)(other_cases[1] ^ other_cases[0]));
  OP2U(SLJIT_SUB | SLJIT_SET_Z, TMP2, 0, SLJIT_IMM, other_cases[1]);
  OP_FLAGS(SLJIT_MOV, TMP2, 0, SLJIT_EQUAL);
  other_cases += 2;
  }
else if (is_powerof2(other_cases[2] ^ other_cases[1]))
  {
  SLJIT_ASSERT(other_cases[2] != NOTACHAR);

  OP2(SLJIT_OR, TMP2, 0, TMP1, 0, SLJIT_IMM, (sljit_sw)(other_cases[2] ^ other_cases[1]));
  OP2U(SLJIT_SUB | SLJIT_SET_Z, TMP2, 0, SLJIT_IMM, other_cases[2]);
  OP_FLAGS(SLJIT_MOV, TMP2, 0, SLJIT_EQUAL);

  OP2U(SLJIT_SUB | SLJIT_SET_Z, TMP1, 0, SLJIT_IMM, (sljit_sw)other_cases[0]);

  if (has_cmov)
    SELECT(SLJIT_EQUAL, TMP2, STR_END, 0, TMP2);
  else
    OP_FLAGS(SLJIT_OR | ((other_cases[3] == NOTACHAR) ? SLJIT_SET_Z : 0), TMP2, 0, SLJIT_EQUAL);

  other_cases += 3;
  }
else
  {
  OP2U(SLJIT_SUB | SLJIT_SET_Z, TMP1, 0, SLJIT_IMM, (sljit_sw)(*other_cases++));
  OP_FLAGS(SLJIT_MOV, TMP2, 0, SLJIT_EQUAL);
  }

while (*other_cases != NOTACHAR)
  {
  OP2U(SLJIT_SUB | SLJIT_SET_Z, TMP1, 0, SLJIT_IMM, (sljit_sw)(*other_cases++));

  if (has_cmov)
    SELECT(SLJIT_EQUAL, TMP2, STR_END, 0, TMP2);
  else
    OP_FLAGS(SLJIT_OR | ((*other_cases == NOTACHAR) ? SLJIT_SET_Z : 0), TMP2, 0, SLJIT_EQUAL);
  }

if (has_cmov)
  jump = CMP(cc[0] == OP_PROP ? SLJIT_EQUAL : SLJIT_NOT_EQUAL, TMP2, 0, SLJIT_IMM, 0);
else
  jump = JUMP(cc[0] == OP_PROP ? SLJIT_ZERO : SLJIT_NOT_ZERO);

add_jump(compiler, backtracks, jump);
}

#endif /* SUPPORT_UNICODE */

static PCRE2_SPTR compile_char1_matchingpath(compiler_common *common, PCRE2_UCHAR type, PCRE2_SPTR cc, jump_list **backtracks, BOOL check_str_ptr)
{
DEFINE_COMPILER;
int length;
unsigned int c, oc, bit;
compare_context context;
struct sljit_jump *jump[3];
jump_list *end_list;
#ifdef SUPPORT_UNICODE
PCRE2_UCHAR propdata[5];
#endif /* SUPPORT_UNICODE */

switch(type)
  {
  case OP_NOT_DIGIT:
  case OP_DIGIT:
  /* Digits are usually 0-9, so it is worth to optimize them. */
  if (check_str_ptr)
    detect_partial_match(common, backtracks);
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH == 8
  if (common->utf && is_char7_bitset((const sljit_u8*)common->ctypes - cbit_length + cbit_digit, FALSE))
    read_char7_type(common, backtracks, type == OP_NOT_DIGIT);
  else
#endif
    read_char8_type(common, backtracks, type == OP_NOT_DIGIT);
    /* Flip the starting bit in the negative case. */
  OP2U(SLJIT_AND | SLJIT_SET_Z, TMP1, 0, SLJIT_IMM, ctype_digit);
  add_jump(compiler, backtracks, JUMP(type == OP_DIGIT ? SLJIT_ZERO : SLJIT_NOT_ZERO));
  return cc;

  case OP_NOT_WHITESPACE:
  case OP_WHITESPACE:
  if (check_str_ptr)
    detect_partial_match(common, backtracks);
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH == 8
  if (common->utf && is_char7_bitset((const sljit_u8*)common->ctypes - cbit_length + cbit_space, FALSE))
    read_char7_type(common, backtracks, type == OP_NOT_WHITESPACE);
  else
#endif
    read_char8_type(common, backtracks, type == OP_NOT_WHITESPACE);
  OP2U(SLJIT_AND | SLJIT_SET_Z, TMP1, 0, SLJIT_IMM, ctype_space);
  add_jump(compiler, backtracks, JUMP(type == OP_WHITESPACE ? SLJIT_ZERO : SLJIT_NOT_ZERO));
  return cc;

  case OP_NOT_WORDCHAR:
  case OP_WORDCHAR:
  if (check_str_ptr)
    detect_partial_match(common, backtracks);
#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH == 8
  if (common->utf && is_char7_bitset((const sljit_u8*)common->ctypes - cbit_length + cbit_word, FALSE))
    read_char7_type(common, backtracks, type == OP_NOT_WORDCHAR);
  else
#endif
    read_char8_type(common, backtracks, type == OP_NOT_WORDCHAR);
  OP2U(SLJIT_AND | SLJIT_SET_Z, TMP1, 0, SLJIT_IMM, ctype_word);
  add_jump(compiler, backtracks, JUMP(type == OP_WORDCHAR ? SLJIT_ZERO : SLJIT_NOT_ZERO));
  return cc;

  case OP_ANY:
  if (check_str_ptr)
    detect_partial_match(common, backtracks);
  read_char(common, common->nlmin, common->nlmax, backtracks, READ_CHAR_UPDATE_STR_PTR);
  if (common->nltype == NLTYPE_FIXED && common->newline > 255)
    {
    jump[0] = CMP(SLJIT_NOT_EQUAL, TMP1, 0, SLJIT_IMM, (common->newline >> 8) & 0xff);
    end_list = NULL;
    if (common->mode != PCRE2_JIT_PARTIAL_HARD)
      add_jump(compiler, &end_list, CMP(SLJIT_GREATER_EQUAL, STR_PTR, 0, STR_END, 0));
    else
      check_str_end(common, &end_list);

    OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), 0);
    add_jump(compiler, backtracks, CMP(SLJIT_EQUAL, TMP1, 0, SLJIT_IMM, common->newline & 0xff));
    set_jumps(end_list, LABEL());
    JUMPHERE(jump[0]);
    }
  else
    check_newlinechar(common, common->nltype, backtracks, TRUE);
  return cc;

  case OP_ALLANY:
  if (check_str_ptr)
    detect_partial_match(common, backtracks);
#ifdef SUPPORT_UNICODE
  if (common->utf && common->invalid_utf)
    {
    read_char(common, 0, READ_CHAR_MAX, backtracks, READ_CHAR_UPDATE_STR_PTR);
    return cc;
    }
#endif /* SUPPORT_UNICODE */

  skip_valid_char(common);
  return cc;

  case OP_ANYBYTE:
  if (check_str_ptr)
    detect_partial_match(common, backtracks);
  OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
  return cc;

#ifdef SUPPORT_UNICODE
  case OP_NOTPROP:
  case OP_PROP:
  if (check_str_ptr)
    detect_partial_match(common, backtracks);
  if (cc[0] == PT_CLIST)
    {
    compile_clist(common, cc - 1, backtracks);
    return cc + 2;
    }

  propdata[0] = 0;
  propdata[1] = type == OP_NOTPROP ? XCL_NOTPROP : XCL_PROP;
  propdata[2] = cc[0];
  propdata[3] = cc[1];
  propdata[4] = XCL_END;
  compile_xclass_matchingpath(common, propdata, backtracks, 0);
  return cc + 2;
#endif

  case OP_ANYNL:
  if (check_str_ptr)
    detect_partial_match(common, backtracks);
  read_char(common, common->bsr_nlmin, common->bsr_nlmax, NULL, 0);
  jump[0] = CMP(SLJIT_NOT_EQUAL, TMP1, 0, SLJIT_IMM, CHAR_CR);
  /* We don't need to handle soft partial matching case. */
  end_list = NULL;
  if (common->mode != PCRE2_JIT_PARTIAL_HARD)
    add_jump(compiler, &end_list, CMP(SLJIT_GREATER_EQUAL, STR_PTR, 0, STR_END, 0));
  else
    check_str_end(common, &end_list);
  OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(STR_PTR), 0);
  OP2U(SLJIT_SUB | SLJIT_SET_Z, TMP1, 0, SLJIT_IMM, CHAR_NL);
  OP_FLAGS(SLJIT_MOV, TMP1, 0, SLJIT_EQUAL);
#if PCRE2_CODE_UNIT_WIDTH == 16 || PCRE2_CODE_UNIT_WIDTH == 32
  OP2(SLJIT_SHL, TMP1, 0, TMP1, 0, SLJIT_IMM, UCHAR_SHIFT);
#endif
  OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, TMP1, 0);
  jump[1] = JUMP(SLJIT_JUMP);
  JUMPHERE(jump[0]);
  check_newlinechar(common, common->bsr_nltype, backtracks, FALSE);
  set_jumps(end_list, LABEL());
  JUMPHERE(jump[1]);
  return cc;

  case OP_NOT_HSPACE:
  case OP_HSPACE:
  if (check_str_ptr)
    detect_partial_match(common, backtracks);

  if (type == OP_NOT_HSPACE)
    read_char(common, 0x9, 0x3000, backtracks, READ_CHAR_UPDATE_STR_PTR);
  else
    read_char(common, 0x9, 0x3000, NULL, 0);

  add_jump(compiler, &common->hspace, JUMP(SLJIT_FAST_CALL));
  sljit_set_current_flags(compiler, SLJIT_SET_Z);
  add_jump(compiler, backtracks, JUMP(type == OP_NOT_HSPACE ? SLJIT_NOT_ZERO : SLJIT_ZERO));
  return cc;

  case OP_NOT_VSPACE:
  case OP_VSPACE:
  if (check_str_ptr)
    detect_partial_match(common, backtracks);

  if (type == OP_NOT_VSPACE)
    read_char(common, 0xa, 0x2029, backtracks, READ_CHAR_UPDATE_STR_PTR);
  else
    read_char(common, 0xa, 0x2029, NULL, 0);

  add_jump(compiler, &common->vspace, JUMP(SLJIT_FAST_CALL));
  sljit_set_current_flags(compiler, SLJIT_SET_Z);
  add_jump(compiler, backtracks, JUMP(type == OP_NOT_VSPACE ? SLJIT_NOT_ZERO : SLJIT_ZERO));
  return cc;

#ifdef SUPPORT_UNICODE
  case OP_EXTUNI:
  if (check_str_ptr)
    detect_partial_match(common, backtracks);

  SLJIT_ASSERT(TMP1 == SLJIT_R0 && STR_PTR == SLJIT_R1);
  OP1(SLJIT_MOV, SLJIT_R0, 0, ARGUMENTS, 0);

#if PCRE2_CODE_UNIT_WIDTH != 32
  sljit_emit_icall(compiler, SLJIT_CALL, SLJIT_ARGS2(W, W, W), SLJIT_IMM,
    common->utf ? (common->invalid_utf ? SLJIT_FUNC_ADDR(do_extuni_utf_invalid) : SLJIT_FUNC_ADDR(do_extuni_utf)) : SLJIT_FUNC_ADDR(do_extuni_no_utf));
  if (common->invalid_utf)
    add_jump(compiler, backtracks, CMP(SLJIT_EQUAL, SLJIT_RETURN_REG, 0, SLJIT_IMM, 0));
#else
  sljit_emit_icall(compiler, SLJIT_CALL, SLJIT_ARGS2(W, W, W), SLJIT_IMM,
    common->invalid_utf ? SLJIT_FUNC_ADDR(do_extuni_utf_invalid) : SLJIT_FUNC_ADDR(do_extuni_no_utf));
  if (common->invalid_utf)
    add_jump(compiler, backtracks, CMP(SLJIT_EQUAL, SLJIT_RETURN_REG, 0, SLJIT_IMM, 0));
#endif

  OP1(SLJIT_MOV, STR_PTR, 0, SLJIT_RETURN_REG, 0);

  if (common->mode == PCRE2_JIT_PARTIAL_HARD)
    {
    jump[0] = CMP(SLJIT_LESS, SLJIT_RETURN_REG, 0, STR_END, 0);
    /* Since we successfully read a char above, partial matching must occur. */
    check_partial(common, TRUE);
    JUMPHERE(jump[0]);
    }
  return cc;
#endif

  case OP_CHAR:
  case OP_CHARI:
  length = 1;
#ifdef SUPPORT_UNICODE
  if (common->utf && HAS_EXTRALEN(*cc)) length += GET_EXTRALEN(*cc);
#endif

  if (check_str_ptr && common->mode != PCRE2_JIT_COMPLETE)
    detect_partial_match(common, backtracks);

  if (type == OP_CHAR || !char_has_othercase(common, cc) || char_get_othercase_bit(common, cc) != 0)
    {
    OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(length));
    if (length > 1 || (check_str_ptr && common->mode == PCRE2_JIT_COMPLETE))
      add_jump(compiler, backtracks, CMP(SLJIT_GREATER, STR_PTR, 0, STR_END, 0));

    context.length = IN_UCHARS(length);
    context.sourcereg = -1;
#if defined SLJIT_UNALIGNED && SLJIT_UNALIGNED
    context.ucharptr = 0;
#endif
    return byte_sequence_compare(common, type == OP_CHARI, cc, &context, backtracks);
    }

#ifdef SUPPORT_UNICODE
  if (common->utf)
    {
    GETCHAR(c, cc);
    }
  else
#endif
    c = *cc;

  SLJIT_ASSERT(type == OP_CHARI && char_has_othercase(common, cc));

  if (check_str_ptr && common->mode == PCRE2_JIT_COMPLETE)
    add_jump(compiler, backtracks, CMP(SLJIT_GREATER_EQUAL, STR_PTR, 0, STR_END, 0));

  oc = char_othercase(common, c);
  read_char(common, c < oc ? c : oc, c > oc ? c : oc, NULL, 0);

  SLJIT_ASSERT(!is_powerof2(c ^ oc));

  if (sljit_has_cpu_feature(SLJIT_HAS_CMOV))
    {
    OP2U(SLJIT_SUB | SLJIT_SET_Z, TMP1, 0, SLJIT_IMM, oc);
    SELECT(SLJIT_EQUAL, TMP1, SLJIT_IMM, c, TMP1);
    add_jump(compiler, backtracks, CMP(SLJIT_NOT_EQUAL, TMP1, 0, SLJIT_IMM, c));
    }
  else
    {
    jump[0] = CMP(SLJIT_EQUAL, TMP1, 0, SLJIT_IMM, c);
    add_jump(compiler, backtracks, CMP(SLJIT_NOT_EQUAL, TMP1, 0, SLJIT_IMM, oc));
    JUMPHERE(jump[0]);
    }
  return cc + length;

  case OP_NOT:
  case OP_NOTI:
  if (check_str_ptr)
    detect_partial_match(common, backtracks);

  length = 1;
#ifdef SUPPORT_UNICODE
  if (common->utf)
    {
#if PCRE2_CODE_UNIT_WIDTH == 8
    c = *cc;
    if (c < 128 && !common->invalid_utf)
      {
      OP1(SLJIT_MOV_U8, TMP1, 0, SLJIT_MEM1(STR_PTR), 0);
      if (type == OP_NOT || !char_has_othercase(common, cc))
        add_jump(compiler, backtracks, CMP(SLJIT_EQUAL, TMP1, 0, SLJIT_IMM, c));
      else
        {
        /* Since UTF8 code page is fixed, we know that c is in [a-z] or [A-Z] range. */
        OP2(SLJIT_OR, TMP2, 0, TMP1, 0, SLJIT_IMM, 0x20);
        add_jump(compiler, backtracks, CMP(SLJIT_EQUAL, TMP2, 0, SLJIT_IMM, c | 0x20));
        }
      /* Skip the variable-length character. */
      OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, IN_UCHARS(1));
      jump[0] = CMP(SLJIT_LESS, TMP1, 0, SLJIT_IMM, 0xc0);
      OP1(MOV_UCHAR, TMP1, 0, SLJIT_MEM1(TMP1), (sljit_sw)PRIV(utf8_table4) - 0xc0);
      OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, TMP1, 0);
      JUMPHERE(jump[0]);
      return cc + 1;
      }
    else
#endif /* PCRE2_CODE_UNIT_WIDTH == 8 */
      {
      GETCHARLEN(c, cc, length);
      }
    }
  else
#endif /* SUPPORT_UNICODE */
    c = *cc;

  if (type == OP_NOT || !char_has_othercase(common, cc))
    {
    read_char(common, c, c, backtracks, READ_CHAR_UPDATE_STR_PTR);
    add_jump(compiler, backtracks, CMP(SLJIT_EQUAL, TMP1, 0, SLJIT_IMM, c));
    }
  else
    {
    oc = char_othercase(common, c);
    read_char(common, c < oc ? c : oc, c > oc ? c : oc, backtracks, READ_CHAR_UPDATE_STR_PTR);
    bit = c ^ oc;
    if (is_powerof2(bit))
      {
      OP2(SLJIT_OR, TMP1, 0, TMP1, 0, SLJIT_IMM, bit);
      add_jump(compiler, backtracks, CMP(SLJIT_EQUAL, TMP1, 0, SLJIT_IMM, c | bit));
      }
    else
      {
      add_jump(compiler, backtracks, CMP(SLJIT_EQUAL, TMP1, 0, SLJIT_IMM, c));
      add_jump(compiler, backtracks, CMP(SLJIT_EQUAL, TMP1, 0, SLJIT_IMM, oc));
      }
    }
  return cc + length;

  case OP_CLASS:
  case OP_NCLASS:
  if (check_str_ptr)
    detect_partial_match(common, backtracks);

#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH == 8
  bit = (common->utf && is_char7_bitset((const sljit_u8 *)cc, type == OP_NCLASS)) ? 127 : 255;
  if (type == OP_NCLASS)
    read_char(common, 0, bit, backtracks, READ_CHAR_UPDATE_STR_PTR);
  else
    read_char(common, 0, bit, NULL, 0);
#else
  if (type == OP_NCLASS)
    read_char(common, 0, 255, backtracks, READ_CHAR_UPDATE_STR_PTR);
  else
    read_char(common, 0, 255, NULL, 0);
#endif

  if (optimize_class(common, (const sljit_u8 *)cc, type == OP_NCLASS, FALSE, backtracks))
    return cc + 32 / sizeof(PCRE2_UCHAR);

#if defined SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH == 8
  jump[0] = NULL;
  if (common->utf)
    {
    jump[0] = CMP(SLJIT_GREATER, TMP1, 0, SLJIT_IMM, bit);
    if (type == OP_CLASS)
      {
      add_jump(compiler, backtracks, jump[0]);
      jump[0] = NULL;
      }
    }
#elif PCRE2_CODE_UNIT_WIDTH != 8
  jump[0] = CMP(SLJIT_GREATER, TMP1, 0, SLJIT_IMM, 255);
  if (type == OP_CLASS)
    {
    add_jump(compiler, backtracks, jump[0]);
    jump[0] = NULL;
    }
#endif /* SUPPORT_UNICODE && PCRE2_CODE_UNIT_WIDTH == 8 */

  OP2(SLJIT_AND, TMP2, 0, TMP1, 0, SLJIT_IMM, 0x7);
  OP2(SLJIT_LSHR, TMP1, 0, TMP1, 0, SLJIT_IMM, 3);
  OP1(SLJIT_MOV_U8, TMP1, 0, SLJIT_MEM1(TMP1), (sljit_sw)cc);
  OP2(SLJIT_SHL, TMP2, 0, SLJIT_IMM, 1, TMP2, 0);
  OP2U(SLJIT_AND | SLJIT_SET_Z, TMP1, 0, TMP2, 0);
  add_jump(compiler, backtracks, JUMP(SLJIT_ZERO));

#if defined SUPPORT_UNICODE || PCRE2_CODE_UNIT_WIDTH != 8
  if (jump[0] != NULL)
    JUMPHERE(jump[0]);
#endif
  return cc + 32 / sizeof(PCRE2_UCHAR);

#if defined SUPPORT_UNICODE || PCRE2_CODE_UNIT_WIDTH == 16 || PCRE2_CODE_UNIT_WIDTH == 32
  case OP_XCLASS:
  if (check_str_ptr)
    detect_partial_match(common, backtracks);
  compile_xclass_matchingpath(common, cc + LINK_SIZE, backtracks, 0);
  return cc + GET(cc, 0) - 1;

  case OP_ECLASS:
  if (check_str_ptr)
    detect_partial_match(common, backtracks);
  return compile_eclass_matchingpath(common, cc, backtracks);
#endif
  }
SLJIT_UNREACHABLE();
return cc;
}

static SLJIT_INLINE PCRE2_SPTR compile_charn_matchingpath(compiler_common *common, PCRE2_SPTR cc, PCRE2_SPTR ccend, jump_list **backtracks)
{
/* This function consumes at least one input character. */
/* To decrease the number of length checks, we try to concatenate the fixed length character sequences. */
DEFINE_COMPILER;
PCRE2_SPTR ccbegin = cc;
compare_context context;
int size;

context.length = 0;
do
  {
  if (cc >= ccend)
    break;

  if (*cc == OP_CHAR)
    {
    size = 1;
#ifdef SUPPORT_UNICODE
    if (common->utf && HAS_EXTRALEN(cc[1]))
      size += GET_EXTRALEN(cc[1]);
#endif
    }
  else if (*cc == OP_CHARI)
    {
    size = 1;
#ifdef SUPPORT_UNICODE
    if (common->utf)
      {
      if (char_has_othercase(common, cc + 1) && char_get_othercase_bit(common, cc + 1) == 0)
        size = 0;
      else if (HAS_EXTRALEN(cc[1]))
        size += GET_EXTRALEN(cc[1]);
      }
    else
#endif
    if (char_has_othercase(common, cc + 1) && char_get_othercase_bit(common, cc + 1) == 0)
      size = 0;
    }
  else
    size = 0;

  cc += 1 + size;
  context.length += IN_UCHARS(size);
  }
while (size > 0 && context.length <= 128);

cc = ccbegin;
if (context.length > 0)
  {
  /* We have a fixed-length byte sequence. */
  OP2(SLJIT_ADD, STR_PTR, 0, STR_PTR, 0, SLJIT_IMM, context.length);
  add_jump(compiler, backtracks, CMP(SLJIT_GREATER, STR_PTR, 0, STR_END, 0));

  context.sourcereg = -1;
#if defined SLJIT_UNALIGNED && SLJIT_UNALIGNED
  context.ucharptr = 0;
#endif
  do cc = byte_sequence_compare(common, *cc == OP_CHARI, cc + 1, &context, backtracks); while (context.length > 0);
  return cc;
  }

/* A non-fixed length character will be checked if length == 0. */
return compile_char1_matchingpath(common, *cc, cc + 1, backtracks, TRUE);
}


