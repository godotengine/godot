/*************************************************
*      Perl-Compatible Regular Expressions       *
*************************************************/

/* PCRE is a library of functions to support regular expressions whose syntax
and semantics are as close as possible to those of the Perl 5 language.

                       Written by Philip Hazel
     Original API code Copyright (c) 1997-2012 University of Cambridge
          New API code Copyright (c) 2016-2019 University of Cambridge

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

/* This module contains functions that scan a compiled pattern and change
repeats into possessive repeats where possible. */


#ifdef HAVE_CONFIG_H
#include "config.h"
#endif


#include "pcre2_internal.h"


/*************************************************
*        Tables for auto-possessification        *
*************************************************/

/* This table is used to check whether auto-possessification is possible
between adjacent character-type opcodes. The left-hand (repeated) opcode is
used to select the row, and the right-hand opcode is use to select the column.
A value of 1 means that auto-possessification is OK. For example, the second
value in the first row means that \D+\d can be turned into \D++\d.

The Unicode property types (\P and \p) have to be present to fill out the table
because of what their opcode values are, but the table values should always be
zero because property types are handled separately in the code. The last four
columns apply to items that cannot be repeated, so there is no need to have
rows for them. Note that OP_DIGIT etc. are generated only when PCRE_UCP is
*not* set. When it is set, \d etc. are converted into OP_(NOT_)PROP codes. */

#define APTROWS (LAST_AUTOTAB_LEFT_OP - FIRST_AUTOTAB_OP + 1)
#define APTCOLS (LAST_AUTOTAB_RIGHT_OP - FIRST_AUTOTAB_OP + 1)

static const uint8_t autoposstab[APTROWS][APTCOLS] = {
/* \D \d \S \s \W \w  . .+ \C \P \p \R \H \h \V \v \X \Z \z  $ $M */
  { 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 },  /* \D */
  { 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1 },  /* \d */
  { 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1 },  /* \S */
  { 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 },  /* \s */
  { 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 },  /* \W */
  { 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1 },  /* \w */
  { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0 },  /* .  */
  { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 },  /* .+ */
  { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 },  /* \C */
  { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },  /* \P */
  { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },  /* \p */
  { 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0 },  /* \R */
  { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0 },  /* \H */
  { 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0 },  /* \h */
  { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0 },  /* \V */
  { 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0 },  /* \v */
  { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 }   /* \X */
};

#ifdef SUPPORT_UNICODE
/* This table is used to check whether auto-possessification is possible
between adjacent Unicode property opcodes (OP_PROP and OP_NOTPROP). The
left-hand (repeated) opcode is used to select the row, and the right-hand
opcode is used to select the column. The values are as follows:

  0   Always return FALSE (never auto-possessify)
  1   Character groups are distinct (possessify if both are OP_PROP)
  2   Check character categories in the same group (general or particular)
  3   TRUE if the two opcodes are not the same (PROP vs NOTPROP)

  4   Check left general category vs right particular category
  5   Check right general category vs left particular category

  6   Left alphanum vs right general category
  7   Left space vs right general category
  8   Left word vs right general category

  9   Right alphanum vs left general category
 10   Right space vs left general category
 11   Right word vs left general category

 12   Left alphanum vs right particular category
 13   Left space vs right particular category
 14   Left word vs right particular category

 15   Right alphanum vs left particular category
 16   Right space vs left particular category
 17   Right word vs left particular category
*/

static const uint8_t propposstab[PT_TABSIZE][PT_TABSIZE] = {
/* ANY LAMP GC  PC  SC ALNUM SPACE PXSPACE WORD CLIST UCNC */
  { 0,  0,  0,  0,  0,    0,    0,      0,   0,    0,   0 },  /* PT_ANY */
  { 0,  3,  0,  0,  0,    3,    1,      1,   0,    0,   0 },  /* PT_LAMP */
  { 0,  0,  2,  4,  0,    9,   10,     10,  11,    0,   0 },  /* PT_GC */
  { 0,  0,  5,  2,  0,   15,   16,     16,  17,    0,   0 },  /* PT_PC */
  { 0,  0,  0,  0,  2,    0,    0,      0,   0,    0,   0 },  /* PT_SC */
  { 0,  3,  6, 12,  0,    3,    1,      1,   0,    0,   0 },  /* PT_ALNUM */
  { 0,  1,  7, 13,  0,    1,    3,      3,   1,    0,   0 },  /* PT_SPACE */
  { 0,  1,  7, 13,  0,    1,    3,      3,   1,    0,   0 },  /* PT_PXSPACE */
  { 0,  0,  8, 14,  0,    0,    1,      1,   3,    0,   0 },  /* PT_WORD */
  { 0,  0,  0,  0,  0,    0,    0,      0,   0,    0,   0 },  /* PT_CLIST */
  { 0,  0,  0,  0,  0,    0,    0,      0,   0,    0,   3 }   /* PT_UCNC */
};

/* This table is used to check whether auto-possessification is possible
between adjacent Unicode property opcodes (OP_PROP and OP_NOTPROP) when one
specifies a general category and the other specifies a particular category. The
row is selected by the general category and the column by the particular
category. The value is 1 if the particular category is not part of the general
category. */

static const uint8_t catposstab[7][30] = {
/* Cc Cf Cn Co Cs Ll Lm Lo Lt Lu Mc Me Mn Nd Nl No Pc Pd Pe Pf Pi Po Ps Sc Sk Sm So Zl Zp Zs */
  { 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },  /* C */
  { 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },  /* L */
  { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },  /* M */
  { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },  /* N */
  { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1 },  /* P */
  { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1 },  /* S */
  { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0 }   /* Z */
};

/* This table is used when checking ALNUM, (PX)SPACE, SPACE, and WORD against
a general or particular category. The properties in each row are those
that apply to the character set in question. Duplication means that a little
unnecessary work is done when checking, but this keeps things much simpler
because they can all use the same code. For more details see the comment where
this table is used.

Note: SPACE and PXSPACE used to be different because Perl excluded VT from
"space", but from Perl 5.18 it's included, so both categories are treated the
same here. */

static const uint8_t posspropstab[3][4] = {
  { ucp_L, ucp_N, ucp_N, ucp_Nl },  /* ALNUM, 3rd and 4th values redundant */
  { ucp_Z, ucp_Z, ucp_C, ucp_Cc },  /* SPACE and PXSPACE, 2nd value redundant */
  { ucp_L, ucp_N, ucp_P, ucp_Po }   /* WORD */
};
#endif  /* SUPPORT_UNICODE */



#ifdef SUPPORT_UNICODE
/*************************************************
*        Check a character and a property        *
*************************************************/

/* This function is called by compare_opcodes() when a property item is
adjacent to a fixed character.

Arguments:
  c            the character
  ptype        the property type
  pdata        the data for the type
  negated      TRUE if it's a negated property (\P or \p{^)

Returns:       TRUE if auto-possessifying is OK
*/

static BOOL
check_char_prop(uint32_t c, unsigned int ptype, unsigned int pdata,
  BOOL negated)
{
const uint32_t *p;
const ucd_record *prop = GET_UCD(c);

switch(ptype)
  {
  case PT_LAMP:
  return (prop->chartype == ucp_Lu ||
          prop->chartype == ucp_Ll ||
          prop->chartype == ucp_Lt) == negated;

  case PT_GC:
  return (pdata == PRIV(ucp_gentype)[prop->chartype]) == negated;

  case PT_PC:
  return (pdata == prop->chartype) == negated;

  case PT_SC:
  return (pdata == prop->script) == negated;

  /* These are specials */

  case PT_ALNUM:
  return (PRIV(ucp_gentype)[prop->chartype] == ucp_L ||
          PRIV(ucp_gentype)[prop->chartype] == ucp_N) == negated;

  /* Perl space used to exclude VT, but from Perl 5.18 it is included, which
  means that Perl space and POSIX space are now identical. PCRE was changed
  at release 8.34. */

  case PT_SPACE:    /* Perl space */
  case PT_PXSPACE:  /* POSIX space */
  switch(c)
    {
    HSPACE_CASES:
    VSPACE_CASES:
    return negated;

    default:
    return (PRIV(ucp_gentype)[prop->chartype] == ucp_Z) == negated;
    }
  break;  /* Control never reaches here */

  case PT_WORD:
  return (PRIV(ucp_gentype)[prop->chartype] == ucp_L ||
          PRIV(ucp_gentype)[prop->chartype] == ucp_N ||
          c == CHAR_UNDERSCORE) == negated;

  case PT_CLIST:
  p = PRIV(ucd_caseless_sets) + prop->caseset;
  for (;;)
    {
    if (c < *p) return !negated;
    if (c == *p++) return negated;
    }
  break;  /* Control never reaches here */
  }

return FALSE;
}
#endif  /* SUPPORT_UNICODE */



/*************************************************
*        Base opcode of repeated opcodes         *
*************************************************/

/* Returns the base opcode for repeated single character type opcodes. If the
opcode is not a repeated character type, it returns with the original value.

Arguments:  c opcode
Returns:    base opcode for the type
*/

static PCRE2_UCHAR
get_repeat_base(PCRE2_UCHAR c)
{
return (c > OP_TYPEPOSUPTO)? c :
       (c >= OP_TYPESTAR)?   OP_TYPESTAR :
       (c >= OP_NOTSTARI)?   OP_NOTSTARI :
       (c >= OP_NOTSTAR)?    OP_NOTSTAR :
       (c >= OP_STARI)?      OP_STARI :
                             OP_STAR;
}


/*************************************************
*        Fill the character property list        *
*************************************************/

/* Checks whether the code points to an opcode that can take part in auto-
possessification, and if so, fills a list with its properties.

Arguments:
  code        points to start of expression
  utf         TRUE if in UTF mode
  fcc         points to the case-flipping table
  list        points to output list
              list[0] will be filled with the opcode
              list[1] will be non-zero if this opcode
                can match an empty character string
              list[2..7] depends on the opcode

Returns:      points to the start of the next opcode if *code is accepted
              NULL if *code is not accepted
*/

static PCRE2_SPTR
get_chr_property_list(PCRE2_SPTR code, BOOL utf, const uint8_t *fcc,
  uint32_t *list)
{
PCRE2_UCHAR c = *code;
PCRE2_UCHAR base;
PCRE2_SPTR end;
uint32_t chr;

#ifdef SUPPORT_UNICODE
uint32_t *clist_dest;
const uint32_t *clist_src;
#else
(void)utf;    /* Suppress "unused parameter" compiler warning */
#endif

list[0] = c;
list[1] = FALSE;
code++;

if (c >= OP_STAR && c <= OP_TYPEPOSUPTO)
  {
  base = get_repeat_base(c);
  c -= (base - OP_STAR);

  if (c == OP_UPTO || c == OP_MINUPTO || c == OP_EXACT || c == OP_POSUPTO)
    code += IMM2_SIZE;

  list[1] = (c != OP_PLUS && c != OP_MINPLUS && c != OP_EXACT &&
             c != OP_POSPLUS);

  switch(base)
    {
    case OP_STAR:
    list[0] = OP_CHAR;
    break;

    case OP_STARI:
    list[0] = OP_CHARI;
    break;

    case OP_NOTSTAR:
    list[0] = OP_NOT;
    break;

    case OP_NOTSTARI:
    list[0] = OP_NOTI;
    break;

    case OP_TYPESTAR:
    list[0] = *code;
    code++;
    break;
    }
  c = list[0];
  }

switch(c)
  {
  case OP_NOT_DIGIT:
  case OP_DIGIT:
  case OP_NOT_WHITESPACE:
  case OP_WHITESPACE:
  case OP_NOT_WORDCHAR:
  case OP_WORDCHAR:
  case OP_ANY:
  case OP_ALLANY:
  case OP_ANYNL:
  case OP_NOT_HSPACE:
  case OP_HSPACE:
  case OP_NOT_VSPACE:
  case OP_VSPACE:
  case OP_EXTUNI:
  case OP_EODN:
  case OP_EOD:
  case OP_DOLL:
  case OP_DOLLM:
  return code;

  case OP_CHAR:
  case OP_NOT:
  GETCHARINCTEST(chr, code);
  list[2] = chr;
  list[3] = NOTACHAR;
  return code;

  case OP_CHARI:
  case OP_NOTI:
  list[0] = (c == OP_CHARI) ? OP_CHAR : OP_NOT;
  GETCHARINCTEST(chr, code);
  list[2] = chr;

#ifdef SUPPORT_UNICODE
  if (chr < 128 || (chr < 256 && !utf))
    list[3] = fcc[chr];
  else
    list[3] = UCD_OTHERCASE(chr);
#elif defined SUPPORT_WIDE_CHARS
  list[3] = (chr < 256) ? fcc[chr] : chr;
#else
  list[3] = fcc[chr];
#endif

  /* The othercase might be the same value. */

  if (chr == list[3])
    list[3] = NOTACHAR;
  else
    list[4] = NOTACHAR;
  return code;

#ifdef SUPPORT_UNICODE
  case OP_PROP:
  case OP_NOTPROP:
  if (code[0] != PT_CLIST)
    {
    list[2] = code[0];
    list[3] = code[1];
    return code + 2;
    }

  /* Convert only if we have enough space. */

  clist_src = PRIV(ucd_caseless_sets) + code[1];
  clist_dest = list + 2;
  code += 2;

  do {
     if (clist_dest >= list + 8)
       {
       /* Early return if there is not enough space. This should never
       happen, since all clists are shorter than 5 character now. */
       list[2] = code[0];
       list[3] = code[1];
       return code;
       }
     *clist_dest++ = *clist_src;
     }
  while(*clist_src++ != NOTACHAR);

  /* All characters are stored. The terminating NOTACHAR is copied from the
  clist itself. */

  list[0] = (c == OP_PROP) ? OP_CHAR : OP_NOT;
  return code;
#endif

  case OP_NCLASS:
  case OP_CLASS:
#ifdef SUPPORT_WIDE_CHARS
  case OP_XCLASS:
  if (c == OP_XCLASS)
    end = code + GET(code, 0) - 1;
  else
#endif
    end = code + 32 / sizeof(PCRE2_UCHAR);

  switch(*end)
    {
    case OP_CRSTAR:
    case OP_CRMINSTAR:
    case OP_CRQUERY:
    case OP_CRMINQUERY:
    case OP_CRPOSSTAR:
    case OP_CRPOSQUERY:
    list[1] = TRUE;
    end++;
    break;

    case OP_CRPLUS:
    case OP_CRMINPLUS:
    case OP_CRPOSPLUS:
    end++;
    break;

    case OP_CRRANGE:
    case OP_CRMINRANGE:
    case OP_CRPOSRANGE:
    list[1] = (GET2(end, 1) == 0);
    end += 1 + 2 * IMM2_SIZE;
    break;
    }
  list[2] = (uint32_t)(end - code);
  return end;
  }
return NULL;    /* Opcode not accepted */
}



/*************************************************
*    Scan further character sets for match       *
*************************************************/

/* Checks whether the base and the current opcode have a common character, in
which case the base cannot be possessified.

Arguments:
  code        points to the byte code
  utf         TRUE in UTF mode
  cb          compile data block
  base_list   the data list of the base opcode
  base_end    the end of the base opcode
  rec_limit   points to recursion depth counter

Returns:      TRUE if the auto-possessification is possible
*/

static BOOL
compare_opcodes(PCRE2_SPTR code, BOOL utf, const compile_block *cb,
  const uint32_t *base_list, PCRE2_SPTR base_end, int *rec_limit)
{
PCRE2_UCHAR c;
uint32_t list[8];
const uint32_t *chr_ptr;
const uint32_t *ochr_ptr;
const uint32_t *list_ptr;
PCRE2_SPTR next_code;
#ifdef SUPPORT_WIDE_CHARS
PCRE2_SPTR xclass_flags;
#endif
const uint8_t *class_bitset;
const uint8_t *set1, *set2, *set_end;
uint32_t chr;
BOOL accepted, invert_bits;
BOOL entered_a_group = FALSE;

if (--(*rec_limit) <= 0) return FALSE;  /* Recursion has gone too deep */

/* Note: the base_list[1] contains whether the current opcode has a greedy
(represented by a non-zero value) quantifier. This is a different from
other character type lists, which store here that the character iterator
matches to an empty string (also represented by a non-zero value). */

for(;;)
  {
  /* All operations move the code pointer forward.
  Therefore infinite recursions are not possible. */

  c = *code;

  /* Skip over callouts */

  if (c == OP_CALLOUT)
    {
    code += PRIV(OP_lengths)[c];
    continue;
    }

  if (c == OP_CALLOUT_STR)
    {
    code += GET(code, 1 + 2*LINK_SIZE);
    continue;
    }

  /* At the end of a branch, skip to the end of the group. */

  if (c == OP_ALT)
    {
    do code += GET(code, 1); while (*code == OP_ALT);
    c = *code;
    }

  /* Inspect the next opcode. */

  switch(c)
    {
    /* We can always possessify a greedy iterator at the end of the pattern,
    which is reached after skipping over the final OP_KET. A non-greedy
    iterator must never be possessified. */

    case OP_END:
    return base_list[1] != 0;

    /* When an iterator is at the end of certain kinds of group we can inspect
    what follows the group by skipping over the closing ket. Note that this
    does not apply to OP_KETRMAX or OP_KETRMIN because what follows any given
    iteration is variable (could be another iteration or could be the next
    item). As these two opcodes are not listed in the next switch, they will
    end up as the next code to inspect, and return FALSE by virtue of being
    unsupported. */

    case OP_KET:
    case OP_KETRPOS:
    /* The non-greedy case cannot be converted to a possessive form. */

    if (base_list[1] == 0) return FALSE;

    /* If the bracket is capturing it might be referenced by an OP_RECURSE
    so its last iterator can never be possessified if the pattern contains
    recursions. (This could be improved by keeping a list of group numbers that
    are called by recursion.) */

    switch(*(code - GET(code, 1)))
      {
      case OP_CBRA:
      case OP_SCBRA:
      case OP_CBRAPOS:
      case OP_SCBRAPOS:
      if (cb->had_recurse) return FALSE;
      break;

      /* A script run might have to backtrack if the iterated item can match
      characters from more than one script. So give up unless repeating an
      explicit character. */

      case OP_SCRIPT_RUN:
      if (base_list[0] != OP_CHAR && base_list[0] != OP_CHARI)
        return FALSE;
      break;

      /* Atomic sub-patterns and assertions can always auto-possessify their
      last iterator. However, if the group was entered as a result of checking
      a previous iterator, this is not possible. */

      case OP_ASSERT:
      case OP_ASSERT_NOT:
      case OP_ASSERTBACK:
      case OP_ASSERTBACK_NOT:
      case OP_ONCE:
      return !entered_a_group;

      /* Non-atomic assertions - don't possessify last iterator. This needs
      more thought. */

      case OP_ASSERT_NA:
      case OP_ASSERTBACK_NA:
      return FALSE;
      }

    /* Skip over the bracket and inspect what comes next. */

    code += PRIV(OP_lengths)[c];
    continue;

    /* Handle cases where the next item is a group. */

    case OP_ONCE:
    case OP_BRA:
    case OP_CBRA:
    next_code = code + GET(code, 1);
    code += PRIV(OP_lengths)[c];

    /* Check each branch. We have to recurse a level for all but the last
    branch. */

    while (*next_code == OP_ALT)
      {
      if (!compare_opcodes(code, utf, cb, base_list, base_end, rec_limit))
        return FALSE;
      code = next_code + 1 + LINK_SIZE;
      next_code += GET(next_code, 1);
      }

    entered_a_group = TRUE;
    continue;

    case OP_BRAZERO:
    case OP_BRAMINZERO:

    next_code = code + 1;
    if (*next_code != OP_BRA && *next_code != OP_CBRA &&
        *next_code != OP_ONCE) return FALSE;

    do next_code += GET(next_code, 1); while (*next_code == OP_ALT);

    /* The bracket content will be checked by the OP_BRA/OP_CBRA case above. */

    next_code += 1 + LINK_SIZE;
    if (!compare_opcodes(next_code, utf, cb, base_list, base_end, rec_limit))
      return FALSE;

    code += PRIV(OP_lengths)[c];
    continue;

    /* The next opcode does not need special handling; fall through and use it
    to see if the base can be possessified. */

    default:
    break;
    }

  /* We now have the next appropriate opcode to compare with the base. Check
  for a supported opcode, and load its properties. */

  code = get_chr_property_list(code, utf, cb->fcc, list);
  if (code == NULL) return FALSE;    /* Unsupported */

  /* If either opcode is a small character list, set pointers for comparing
  characters from that list with another list, or with a property. */

  if (base_list[0] == OP_CHAR)
    {
    chr_ptr = base_list + 2;
    list_ptr = list;
    }
  else if (list[0] == OP_CHAR)
    {
    chr_ptr = list + 2;
    list_ptr = base_list;
    }

  /* Character bitsets can also be compared to certain opcodes. */

  else if (base_list[0] == OP_CLASS || list[0] == OP_CLASS
#if PCRE2_CODE_UNIT_WIDTH == 8
      /* In 8 bit, non-UTF mode, OP_CLASS and OP_NCLASS are the same. */
      || (!utf && (base_list[0] == OP_NCLASS || list[0] == OP_NCLASS))
#endif
      )
    {
#if PCRE2_CODE_UNIT_WIDTH == 8
    if (base_list[0] == OP_CLASS || (!utf && base_list[0] == OP_NCLASS))
#else
    if (base_list[0] == OP_CLASS)
#endif
      {
      set1 = (uint8_t *)(base_end - base_list[2]);
      list_ptr = list;
      }
    else
      {
      set1 = (uint8_t *)(code - list[2]);
      list_ptr = base_list;
      }

    invert_bits = FALSE;
    switch(list_ptr[0])
      {
      case OP_CLASS:
      case OP_NCLASS:
      set2 = (uint8_t *)
        ((list_ptr == list ? code : base_end) - list_ptr[2]);
      break;

#ifdef SUPPORT_WIDE_CHARS
      case OP_XCLASS:
      xclass_flags = (list_ptr == list ? code : base_end) - list_ptr[2] + LINK_SIZE;
      if ((*xclass_flags & XCL_HASPROP) != 0) return FALSE;
      if ((*xclass_flags & XCL_MAP) == 0)
        {
        /* No bits are set for characters < 256. */
        if (list[1] == 0) return (*xclass_flags & XCL_NOT) == 0;
        /* Might be an empty repeat. */
        continue;
        }
      set2 = (uint8_t *)(xclass_flags + 1);
      break;
#endif

      case OP_NOT_DIGIT:
      invert_bits = TRUE;
      /* Fall through */
      case OP_DIGIT:
      set2 = (uint8_t *)(cb->cbits + cbit_digit);
      break;

      case OP_NOT_WHITESPACE:
      invert_bits = TRUE;
      /* Fall through */
      case OP_WHITESPACE:
      set2 = (uint8_t *)(cb->cbits + cbit_space);
      break;

      case OP_NOT_WORDCHAR:
      invert_bits = TRUE;
      /* Fall through */
      case OP_WORDCHAR:
      set2 = (uint8_t *)(cb->cbits + cbit_word);
      break;

      default:
      return FALSE;
      }

    /* Because the bit sets are unaligned bytes, we need to perform byte
    comparison here. */

    set_end = set1 + 32;
    if (invert_bits)
      {
      do
        {
        if ((*set1++ & ~(*set2++)) != 0) return FALSE;
        }
      while (set1 < set_end);
      }
    else
      {
      do
        {
        if ((*set1++ & *set2++) != 0) return FALSE;
        }
      while (set1 < set_end);
      }

    if (list[1] == 0) return TRUE;
    /* Might be an empty repeat. */
    continue;
    }

  /* Some property combinations also acceptable. Unicode property opcodes are
  processed specially; the rest can be handled with a lookup table. */

  else
    {
    uint32_t leftop, rightop;

    leftop = base_list[0];
    rightop = list[0];

#ifdef SUPPORT_UNICODE
    accepted = FALSE; /* Always set in non-unicode case. */
    if (leftop == OP_PROP || leftop == OP_NOTPROP)
      {
      if (rightop == OP_EOD)
        accepted = TRUE;
      else if (rightop == OP_PROP || rightop == OP_NOTPROP)
        {
        int n;
        const uint8_t *p;
        BOOL same = leftop == rightop;
        BOOL lisprop = leftop == OP_PROP;
        BOOL risprop = rightop == OP_PROP;
        BOOL bothprop = lisprop && risprop;

        /* There's a table that specifies how each combination is to be
        processed:
          0   Always return FALSE (never auto-possessify)
          1   Character groups are distinct (possessify if both are OP_PROP)
          2   Check character categories in the same group (general or particular)
          3   Return TRUE if the two opcodes are not the same
          ... see comments below
        */

        n = propposstab[base_list[2]][list[2]];
        switch(n)
          {
          case 0: break;
          case 1: accepted = bothprop; break;
          case 2: accepted = (base_list[3] == list[3]) != same; break;
          case 3: accepted = !same; break;

          case 4:  /* Left general category, right particular category */
          accepted = risprop && catposstab[base_list[3]][list[3]] == same;
          break;

          case 5:  /* Right general category, left particular category */
          accepted = lisprop && catposstab[list[3]][base_list[3]] == same;
          break;

          /* This code is logically tricky. Think hard before fiddling with it.
          The posspropstab table has four entries per row. Each row relates to
          one of PCRE's special properties such as ALNUM or SPACE or WORD.
          Only WORD actually needs all four entries, but using repeats for the
          others means they can all use the same code below.

          The first two entries in each row are Unicode general categories, and
          apply always, because all the characters they include are part of the
          PCRE character set. The third and fourth entries are a general and a
          particular category, respectively, that include one or more relevant
          characters. One or the other is used, depending on whether the check
          is for a general or a particular category. However, in both cases the
          category contains more characters than the specials that are defined
          for the property being tested against. Therefore, it cannot be used
          in a NOTPROP case.

          Example: the row for WORD contains ucp_L, ucp_N, ucp_P, ucp_Po.
          Underscore is covered by ucp_P or ucp_Po. */

          case 6:  /* Left alphanum vs right general category */
          case 7:  /* Left space vs right general category */
          case 8:  /* Left word vs right general category */
          p = posspropstab[n-6];
          accepted = risprop && lisprop ==
            (list[3] != p[0] &&
             list[3] != p[1] &&
            (list[3] != p[2] || !lisprop));
          break;

          case 9:   /* Right alphanum vs left general category */
          case 10:  /* Right space vs left general category */
          case 11:  /* Right word vs left general category */
          p = posspropstab[n-9];
          accepted = lisprop && risprop ==
            (base_list[3] != p[0] &&
             base_list[3] != p[1] &&
            (base_list[3] != p[2] || !risprop));
          break;

          case 12:  /* Left alphanum vs right particular category */
          case 13:  /* Left space vs right particular category */
          case 14:  /* Left word vs right particular category */
          p = posspropstab[n-12];
          accepted = risprop && lisprop ==
            (catposstab[p[0]][list[3]] &&
             catposstab[p[1]][list[3]] &&
            (list[3] != p[3] || !lisprop));
          break;

          case 15:  /* Right alphanum vs left particular category */
          case 16:  /* Right space vs left particular category */
          case 17:  /* Right word vs left particular category */
          p = posspropstab[n-15];
          accepted = lisprop && risprop ==
            (catposstab[p[0]][base_list[3]] &&
             catposstab[p[1]][base_list[3]] &&
            (base_list[3] != p[3] || !risprop));
          break;
          }
        }
      }

    else
#endif  /* SUPPORT_UNICODE */

    accepted = leftop >= FIRST_AUTOTAB_OP && leftop <= LAST_AUTOTAB_LEFT_OP &&
           rightop >= FIRST_AUTOTAB_OP && rightop <= LAST_AUTOTAB_RIGHT_OP &&
           autoposstab[leftop - FIRST_AUTOTAB_OP][rightop - FIRST_AUTOTAB_OP];

    if (!accepted) return FALSE;

    if (list[1] == 0) return TRUE;
    /* Might be an empty repeat. */
    continue;
    }

  /* Control reaches here only if one of the items is a small character list.
  All characters are checked against the other side. */

  do
    {
    chr = *chr_ptr;

    switch(list_ptr[0])
      {
      case OP_CHAR:
      ochr_ptr = list_ptr + 2;
      do
        {
        if (chr == *ochr_ptr) return FALSE;
        ochr_ptr++;
        }
      while(*ochr_ptr != NOTACHAR);
      break;

      case OP_NOT:
      ochr_ptr = list_ptr + 2;
      do
        {
        if (chr == *ochr_ptr)
          break;
        ochr_ptr++;
        }
      while(*ochr_ptr != NOTACHAR);
      if (*ochr_ptr == NOTACHAR) return FALSE;   /* Not found */
      break;

      /* Note that OP_DIGIT etc. are generated only when PCRE2_UCP is *not*
      set. When it is set, \d etc. are converted into OP_(NOT_)PROP codes. */

      case OP_DIGIT:
      if (chr < 256 && (cb->ctypes[chr] & ctype_digit) != 0) return FALSE;
      break;

      case OP_NOT_DIGIT:
      if (chr > 255 || (cb->ctypes[chr] & ctype_digit) == 0) return FALSE;
      break;

      case OP_WHITESPACE:
      if (chr < 256 && (cb->ctypes[chr] & ctype_space) != 0) return FALSE;
      break;

      case OP_NOT_WHITESPACE:
      if (chr > 255 || (cb->ctypes[chr] & ctype_space) == 0) return FALSE;
      break;

      case OP_WORDCHAR:
      if (chr < 255 && (cb->ctypes[chr] & ctype_word) != 0) return FALSE;
      break;

      case OP_NOT_WORDCHAR:
      if (chr > 255 || (cb->ctypes[chr] & ctype_word) == 0) return FALSE;
      break;

      case OP_HSPACE:
      switch(chr)
        {
        HSPACE_CASES: return FALSE;
        default: break;
        }
      break;

      case OP_NOT_HSPACE:
      switch(chr)
        {
        HSPACE_CASES: break;
        default: return FALSE;
        }
      break;

      case OP_ANYNL:
      case OP_VSPACE:
      switch(chr)
        {
        VSPACE_CASES: return FALSE;
        default: break;
        }
      break;

      case OP_NOT_VSPACE:
      switch(chr)
        {
        VSPACE_CASES: break;
        default: return FALSE;
        }
      break;

      case OP_DOLL:
      case OP_EODN:
      switch (chr)
        {
        case CHAR_CR:
        case CHAR_LF:
        case CHAR_VT:
        case CHAR_FF:
        case CHAR_NEL:
#ifndef EBCDIC
        case 0x2028:
        case 0x2029:
#endif  /* Not EBCDIC */
        return FALSE;
        }
      break;

      case OP_EOD:    /* Can always possessify before \z */
      break;

#ifdef SUPPORT_UNICODE
      case OP_PROP:
      case OP_NOTPROP:
      if (!check_char_prop(chr, list_ptr[2], list_ptr[3],
            list_ptr[0] == OP_NOTPROP))
        return FALSE;
      break;
#endif

      case OP_NCLASS:
      if (chr > 255) return FALSE;
      /* Fall through */

      case OP_CLASS:
      if (chr > 255) break;
      class_bitset = (uint8_t *)
        ((list_ptr == list ? code : base_end) - list_ptr[2]);
      if ((class_bitset[chr >> 3] & (1u << (chr & 7))) != 0) return FALSE;
      break;

#ifdef SUPPORT_WIDE_CHARS
      case OP_XCLASS:
      if (PRIV(xclass)(chr, (list_ptr == list ? code : base_end) -
          list_ptr[2] + LINK_SIZE, utf)) return FALSE;
      break;
#endif

      default:
      return FALSE;
      }

    chr_ptr++;
    }
  while(*chr_ptr != NOTACHAR);

  /* At least one character must be matched from this opcode. */

  if (list[1] == 0) return TRUE;
  }

/* Control never reaches here. There used to be a fail-save return FALSE; here,
but some compilers complain about an unreachable statement. */
}



/*************************************************
*    Scan compiled regex for auto-possession     *
*************************************************/

/* Replaces single character iterations with their possessive alternatives
if appropriate. This function modifies the compiled opcode! Hitting a
non-existent opcode may indicate a bug in PCRE2, but it can also be caused if a
bad UTF string was compiled with PCRE2_NO_UTF_CHECK. The rec_limit catches
overly complicated or large patterns. In these cases, the check just stops,
leaving the remainder of the pattern unpossessified.

Arguments:
  code        points to start of the byte code
  utf         TRUE in UTF mode
  cb          compile data block

Returns:      0 for success
              -1 if a non-existant opcode is encountered
*/

int
PRIV(auto_possessify)(PCRE2_UCHAR *code, BOOL utf, const compile_block *cb)
{
PCRE2_UCHAR c;
PCRE2_SPTR end;
PCRE2_UCHAR *repeat_opcode;
uint32_t list[8];
int rec_limit = 1000;  /* Was 10,000 but clang+ASAN uses a lot of stack. */

for (;;)
  {
  c = *code;

  if (c >= OP_TABLE_LENGTH) return -1;   /* Something gone wrong */

  if (c >= OP_STAR && c <= OP_TYPEPOSUPTO)
    {
    c -= get_repeat_base(c) - OP_STAR;
    end = (c <= OP_MINUPTO) ?
      get_chr_property_list(code, utf, cb->fcc, list) : NULL;
    list[1] = c == OP_STAR || c == OP_PLUS || c == OP_QUERY || c == OP_UPTO;

    if (end != NULL && compare_opcodes(end, utf, cb, list, end, &rec_limit))
      {
      switch(c)
        {
        case OP_STAR:
        *code += OP_POSSTAR - OP_STAR;
        break;

        case OP_MINSTAR:
        *code += OP_POSSTAR - OP_MINSTAR;
        break;

        case OP_PLUS:
        *code += OP_POSPLUS - OP_PLUS;
        break;

        case OP_MINPLUS:
        *code += OP_POSPLUS - OP_MINPLUS;
        break;

        case OP_QUERY:
        *code += OP_POSQUERY - OP_QUERY;
        break;

        case OP_MINQUERY:
        *code += OP_POSQUERY - OP_MINQUERY;
        break;

        case OP_UPTO:
        *code += OP_POSUPTO - OP_UPTO;
        break;

        case OP_MINUPTO:
        *code += OP_POSUPTO - OP_MINUPTO;
        break;
        }
      }
    c = *code;
    }
  else if (c == OP_CLASS || c == OP_NCLASS || c == OP_XCLASS)
    {
#ifdef SUPPORT_WIDE_CHARS
    if (c == OP_XCLASS)
      repeat_opcode = code + GET(code, 1);
    else
#endif
      repeat_opcode = code + 1 + (32 / sizeof(PCRE2_UCHAR));

    c = *repeat_opcode;
    if (c >= OP_CRSTAR && c <= OP_CRMINRANGE)
      {
      /* end must not be NULL. */
      end = get_chr_property_list(code, utf, cb->fcc, list);

      list[1] = (c & 1) == 0;

      if (compare_opcodes(end, utf, cb, list, end, &rec_limit))
        {
        switch (c)
          {
          case OP_CRSTAR:
          case OP_CRMINSTAR:
          *repeat_opcode = OP_CRPOSSTAR;
          break;

          case OP_CRPLUS:
          case OP_CRMINPLUS:
          *repeat_opcode = OP_CRPOSPLUS;
          break;

          case OP_CRQUERY:
          case OP_CRMINQUERY:
          *repeat_opcode = OP_CRPOSQUERY;
          break;

          case OP_CRRANGE:
          case OP_CRMINRANGE:
          *repeat_opcode = OP_CRPOSRANGE;
          break;
          }
        }
      }
    c = *code;
    }

  switch(c)
    {
    case OP_END:
    return 0;

    case OP_TYPESTAR:
    case OP_TYPEMINSTAR:
    case OP_TYPEPLUS:
    case OP_TYPEMINPLUS:
    case OP_TYPEQUERY:
    case OP_TYPEMINQUERY:
    case OP_TYPEPOSSTAR:
    case OP_TYPEPOSPLUS:
    case OP_TYPEPOSQUERY:
    if (code[1] == OP_PROP || code[1] == OP_NOTPROP) code += 2;
    break;

    case OP_TYPEUPTO:
    case OP_TYPEMINUPTO:
    case OP_TYPEEXACT:
    case OP_TYPEPOSUPTO:
    if (code[1 + IMM2_SIZE] == OP_PROP || code[1 + IMM2_SIZE] == OP_NOTPROP)
      code += 2;
    break;

    case OP_CALLOUT_STR:
    code += GET(code, 1 + 2*LINK_SIZE);
    break;

#ifdef SUPPORT_WIDE_CHARS
    case OP_XCLASS:
    code += GET(code, 1);
    break;
#endif

    case OP_MARK:
    case OP_COMMIT_ARG:
    case OP_PRUNE_ARG:
    case OP_SKIP_ARG:
    case OP_THEN_ARG:
    code += code[1];
    break;
    }

  /* Add in the fixed length from the table */

  code += PRIV(OP_lengths)[c];

  /* In UTF-8 and UTF-16 modes, opcodes that are followed by a character may be
  followed by a multi-byte character. The length in the table is a minimum, so
  we have to arrange to skip the extra code units. */

#ifdef MAYBE_UTF_MULTI
  if (utf) switch(c)
    {
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
    if (HAS_EXTRALEN(code[-1])) code += GET_EXTRALEN(code[-1]);
    break;
    }
#else
  (void)(utf);  /* Keep compiler happy by referencing function argument */
#endif  /* SUPPORT_WIDE_CHARS */
  }
}

/* End of pcre2_auto_possess.c */
