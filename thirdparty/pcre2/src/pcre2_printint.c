/*************************************************
*      Perl-Compatible Regular Expressions       *
*************************************************/

/* PCRE is a library of functions to support regular expressions whose syntax
and semantics are as close as possible to those of the Perl 5 language.

                       Written by Philip Hazel
     Original API code Copyright (c) 1997-2012 University of Cambridge
         New API code Copyright (c) 2016 University of Cambridge

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


/* This module contains a PCRE private debugging function for printing out the
internal form of a compiled regular expression, along with some supporting
local functions. This source file is #included in pcre2test.c at each supported
code unit width, with PCRE2_SUFFIX set appropriately, just like the functions
that comprise the library. It can also optionally be included in
pcre2_compile.c for detailed debugging in error situations. */


/* Tables of operator names. The same 8-bit table is used for all code unit
widths, so it must be defined only once. The list itself is defined in
pcre2_internal.h, which is #included by pcre2test before this file. */

#ifndef OP_LISTS_DEFINED
static const char *OP_names[] = { OP_NAME_LIST };
#define OP_LISTS_DEFINED
#endif

/* The functions and tables herein must all have mode-dependent names. */

#define OP_lengths            PCRE2_SUFFIX(OP_lengths_)
#define get_ucpname           PCRE2_SUFFIX(get_ucpname_)
#define pcre2_printint        PCRE2_SUFFIX(pcre2_printint_)
#define print_char            PCRE2_SUFFIX(print_char_)
#define print_custring        PCRE2_SUFFIX(print_custring_)
#define print_custring_bylen  PCRE2_SUFFIX(print_custring_bylen_)
#define print_prop            PCRE2_SUFFIX(print_prop_)

/* Table of sizes for the fixed-length opcodes. It's defined in a macro so that
the definition is next to the definition of the opcodes in pcre2_internal.h.
The contents of the table are, however, mode-dependent. */

static const uint8_t OP_lengths[] = { OP_LENGTHS };



/*************************************************
*       Print one character from a string        *
*************************************************/

/* In UTF mode the character may occupy more than one code unit.

Arguments:
  f           file to write to
  ptr         pointer to first code unit of the character
  utf         TRUE if string is UTF (will be FALSE if UTF is not supported)

Returns:      number of additional code units used
*/

static unsigned int
print_char(FILE *f, PCRE2_SPTR ptr, BOOL utf)
{
uint32_t c = *ptr;
BOOL one_code_unit = !utf;

/* If UTF is supported and requested, check for a valid single code unit. */

#ifdef SUPPORT_UNICODE
if (utf)
  {
#if PCRE2_CODE_UNIT_WIDTH == 8
  one_code_unit = c < 0x80;
#elif PCRE2_CODE_UNIT_WIDTH == 16
  one_code_unit = (c & 0xfc00) != 0xd800;
#else
  one_code_unit = (c & 0xfffff800u) != 0xd800u;
#endif  /* CODE_UNIT_WIDTH */
  }
#endif  /* SUPPORT_UNICODE */

/* Handle a valid one-code-unit character at any width. */

if (one_code_unit)
  {
  if (PRINTABLE(c)) fprintf(f, "%c", (char)c);
  else if (c < 0x80) fprintf(f, "\\x%02x", c);
  else fprintf(f, "\\x{%02x}", c);
  return 0;
  }

/* Code for invalid UTF code units and multi-unit UTF characters is different
for each width. If UTF is not supported, control should never get here, but we
need a return statement to keep the compiler happy. */

#ifndef SUPPORT_UNICODE
return 0;
#else

/* Malformed UTF-8 should occur only if the sanity check has been turned off.
Rather than swallow random bytes, just stop if we hit a bad one. Print it with
\X instead of \x as an indication. */

#if PCRE2_CODE_UNIT_WIDTH == 8
if ((c & 0xc0) != 0xc0)
  {
  fprintf(f, "\\X{%x}", c);       /* Invalid starting byte */
  return 0;
  }
else
  {
  int i;
  int a = PRIV(utf8_table4)[c & 0x3f];  /* Number of additional bytes */
  int s = 6*a;
  c = (c & PRIV(utf8_table3)[a]) << s;
  for (i = 1; i <= a; i++)
    {
    if ((ptr[i] & 0xc0) != 0x80)
      {
      fprintf(f, "\\X{%x}", c);   /* Invalid secondary byte */
      return i - 1;
      }
    s -= 6;
    c |= (ptr[i] & 0x3f) << s;
    }
  fprintf(f, "\\x{%x}", c);
  return a;
}
#endif  /* PCRE2_CODE_UNIT_WIDTH == 8 */

/* UTF-16: rather than swallow a low surrogate, just stop if we hit a bad one.
Print it with \X instead of \x as an indication. */

#if PCRE2_CODE_UNIT_WIDTH == 16
if ((ptr[1] & 0xfc00) != 0xdc00)
  {
  fprintf(f, "\\X{%x}", c);
  return 0;
  }
c = (((c & 0x3ff) << 10) | (ptr[1] & 0x3ff)) + 0x10000;
fprintf(f, "\\x{%x}", c);
return 1;
#endif  /* PCRE2_CODE_UNIT_WIDTH == 16 */

/* For UTF-32 we get here only for a malformed code unit, which should only
occur if the sanity check has been turned off. Print it with \X instead of \x
as an indication. */

#if PCRE2_CODE_UNIT_WIDTH == 32
fprintf(f, "\\X{%x}", c);
return 0;
#endif  /* PCRE2_CODE_UNIT_WIDTH == 32 */
#endif  /* SUPPORT_UNICODE */
}



/*************************************************
*     Print string as a list of code units       *
*************************************************/

/* These take no account of UTF as they always print each individual code unit.
The string is zero-terminated for print_custring(); the length is given for
print_custring_bylen().

Arguments:
  f          file to write to
  ptr        point to the string
  len        length for print_custring_bylen()

Returns:     nothing
*/

static void
print_custring(FILE *f, PCRE2_SPTR ptr)
{
while (*ptr != '\0')
  {
  uint32_t c = *ptr++;
  if (PRINTABLE(c)) fprintf(f, "%c", c); else fprintf(f, "\\x{%x}", c);
  }
}

static void
print_custring_bylen(FILE *f, PCRE2_SPTR ptr, PCRE2_UCHAR len)
{
for (; len > 0; len--)
  {
  uint32_t c = *ptr++;
  if (PRINTABLE(c)) fprintf(f, "%c", c); else fprintf(f, "\\x{%x}", c);
  }
}



/*************************************************
*          Find Unicode property name            *
*************************************************/

/* When there is no UTF/UCP support, the table of names does not exist. This
function should not be called in such configurations, because a pattern that
tries to use Unicode properties won't compile. Rather than put lots of #ifdefs
into the main code, however, we just put one into this function. */

static const char *
get_ucpname(unsigned int ptype, unsigned int pvalue)
{
#ifdef SUPPORT_UNICODE
int i;
for (i = PRIV(utt_size) - 1; i >= 0; i--)
  {
  if (ptype == PRIV(utt)[i].type && pvalue == PRIV(utt)[i].value) break;
  }
return (i >= 0)? PRIV(utt_names) + PRIV(utt)[i].name_offset : "??";
#else   /* No UTF support */
(void)ptype;
(void)pvalue;
return "??";
#endif  /* SUPPORT_UNICODE */
}



/*************************************************
*       Print Unicode property value             *
*************************************************/

/* "Normal" properties can be printed from tables. The PT_CLIST property is a
pseudo-property that contains a pointer to a list of case-equivalent
characters.

Arguments:
  f            file to write to
  code         pointer in the compiled code
  before       text to print before
  after        text to print after

Returns:       nothing
*/

static void
print_prop(FILE *f, PCRE2_SPTR code, const char *before, const char *after)
{
if (code[1] != PT_CLIST)
  {
  fprintf(f, "%s%s %s%s", before, OP_names[*code], get_ucpname(code[1],
    code[2]), after);
  }
else
  {
  const char *not = (*code == OP_PROP)? "" : "not ";
  const uint32_t *p = PRIV(ucd_caseless_sets) + code[2];
  fprintf (f, "%s%sclist", before, not);
  while (*p < NOTACHAR) fprintf(f, " %04x", *p++);
  fprintf(f, "%s", after);
  }
}



/*************************************************
*            Print compiled pattern              *
*************************************************/

/* The print_lengths flag controls whether offsets and lengths of items are
printed. Lenths can be turned off from pcre2test so that automatic tests on
bytecode can be written that do not depend on the value of LINK_SIZE.

Arguments:
  re              a compiled pattern
  f               the file to write to
  print_lengths   show various lengths

Returns:          nothing
*/

static void
pcre2_printint(pcre2_code *re, FILE *f, BOOL print_lengths)
{
PCRE2_SPTR codestart, nametable, code;
uint32_t nesize = re->name_entry_size;
BOOL utf = (re->overall_options & PCRE2_UTF) != 0;

nametable = (PCRE2_SPTR)((uint8_t *)re + sizeof(pcre2_real_code));
code = codestart = nametable + re->name_count * re->name_entry_size;

for(;;)
  {
  PCRE2_SPTR ccode;
  uint32_t c;
  int i;
  const char *flag = "  ";
  unsigned int extra = 0;

  if (print_lengths)
    fprintf(f, "%3d ", (int)(code - codestart));
  else
    fprintf(f, "    ");

  switch(*code)
    {
/* ========================================================================== */
      /* These cases are never obeyed. This is a fudge that causes a compile-
      time error if the vectors OP_names or OP_lengths, which are indexed
      by opcode, are not the correct length. It seems to be the only way to do
      such a check at compile time, as the sizeof() operator does not work in
      the C preprocessor. */

      case OP_TABLE_LENGTH:
      case OP_TABLE_LENGTH +
        ((sizeof(OP_names)/sizeof(const char *) == OP_TABLE_LENGTH) &&
        (sizeof(OP_lengths) == OP_TABLE_LENGTH)):
      break;
/* ========================================================================== */

    case OP_END:
    fprintf(f, "    %s\n", OP_names[*code]);
    fprintf(f, "------------------------------------------------------------------\n");
    return;

    case OP_CHAR:
    fprintf(f, "    ");
    do
      {
      code++;
      code += 1 + print_char(f, code, utf);
      }
    while (*code == OP_CHAR);
    fprintf(f, "\n");
    continue;

    case OP_CHARI:
    fprintf(f, " /i ");
    do
      {
      code++;
      code += 1 + print_char(f, code, utf);
      }
    while (*code == OP_CHARI);
    fprintf(f, "\n");
    continue;

    case OP_CBRA:
    case OP_CBRAPOS:
    case OP_SCBRA:
    case OP_SCBRAPOS:
    if (print_lengths) fprintf(f, "%3d ", GET(code, 1));
      else fprintf(f, "    ");
    fprintf(f, "%s %d", OP_names[*code], GET2(code, 1+LINK_SIZE));
    break;

    case OP_BRA:
    case OP_BRAPOS:
    case OP_SBRA:
    case OP_SBRAPOS:
    case OP_KETRMAX:
    case OP_KETRMIN:
    case OP_KETRPOS:
    case OP_ALT:
    case OP_KET:
    case OP_ASSERT:
    case OP_ASSERT_NOT:
    case OP_ASSERTBACK:
    case OP_ASSERTBACK_NOT:
    case OP_ONCE:
    case OP_ONCE_NC:
    case OP_COND:
    case OP_SCOND:
    case OP_REVERSE:
    if (print_lengths) fprintf(f, "%3d ", GET(code, 1));
      else fprintf(f, "    ");
    fprintf(f, "%s", OP_names[*code]);
    break;

    case OP_CLOSE:
    fprintf(f, "    %s %d", OP_names[*code], GET2(code, 1));
    break;

    case OP_CREF:
    fprintf(f, "%3d %s", GET2(code,1), OP_names[*code]);
    break;

    case OP_DNCREF:
      {
      PCRE2_SPTR entry = nametable + (GET2(code, 1) * nesize) + IMM2_SIZE;
      fprintf(f, " %s Cond ref <", flag);
      print_custring(f, entry);
      fprintf(f, ">%d", GET2(code, 1 + IMM2_SIZE));
      }
    break;

    case OP_RREF:
    c = GET2(code, 1);
    if (c == RREF_ANY)
      fprintf(f, "    Cond recurse any");
    else
      fprintf(f, "    Cond recurse %d", c);
    break;

    case OP_DNRREF:
      {
      PCRE2_SPTR entry = nametable + (GET2(code, 1) * nesize) + IMM2_SIZE;
      fprintf(f, " %s Cond recurse <", flag);
      print_custring(f, entry);
      fprintf(f, ">%d", GET2(code, 1 + IMM2_SIZE));
      }
    break;

    case OP_FALSE:
    fprintf(f, "    Cond false");
    break;

    case OP_TRUE:
    fprintf(f, "    Cond true");
    break;

    case OP_STARI:
    case OP_MINSTARI:
    case OP_POSSTARI:
    case OP_PLUSI:
    case OP_MINPLUSI:
    case OP_POSPLUSI:
    case OP_QUERYI:
    case OP_MINQUERYI:
    case OP_POSQUERYI:
    flag = "/i";
    /* Fall through */
    case OP_STAR:
    case OP_MINSTAR:
    case OP_POSSTAR:
    case OP_PLUS:
    case OP_MINPLUS:
    case OP_POSPLUS:
    case OP_QUERY:
    case OP_MINQUERY:
    case OP_POSQUERY:
    case OP_TYPESTAR:
    case OP_TYPEMINSTAR:
    case OP_TYPEPOSSTAR:
    case OP_TYPEPLUS:
    case OP_TYPEMINPLUS:
    case OP_TYPEPOSPLUS:
    case OP_TYPEQUERY:
    case OP_TYPEMINQUERY:
    case OP_TYPEPOSQUERY:
    fprintf(f, " %s ", flag);

    if (*code >= OP_TYPESTAR)
      {
      if (code[1] == OP_PROP || code[1] == OP_NOTPROP)
        {
        print_prop(f, code + 1, "", " ");
        extra = 2;
        }
      else fprintf(f, "%s", OP_names[code[1]]);
      }
    else extra = print_char(f, code+1, utf);
    fprintf(f, "%s", OP_names[*code]);
    break;

    case OP_EXACTI:
    case OP_UPTOI:
    case OP_MINUPTOI:
    case OP_POSUPTOI:
    flag = "/i";
    /* Fall through */
    case OP_EXACT:
    case OP_UPTO:
    case OP_MINUPTO:
    case OP_POSUPTO:
    fprintf(f, " %s ", flag);
    extra = print_char(f, code + 1 + IMM2_SIZE, utf);
    fprintf(f, "{");
    if (*code != OP_EXACT && *code != OP_EXACTI) fprintf(f, "0,");
    fprintf(f, "%d}", GET2(code,1));
    if (*code == OP_MINUPTO || *code == OP_MINUPTOI) fprintf(f, "?");
      else if (*code == OP_POSUPTO || *code == OP_POSUPTOI) fprintf(f, "+");
    break;

    case OP_TYPEEXACT:
    case OP_TYPEUPTO:
    case OP_TYPEMINUPTO:
    case OP_TYPEPOSUPTO:
    if (code[1 + IMM2_SIZE] == OP_PROP || code[1 + IMM2_SIZE] == OP_NOTPROP)
      {
      print_prop(f, code + IMM2_SIZE + 1, "    ", " ");
      extra = 2;
      }
    else fprintf(f, "    %s", OP_names[code[1 + IMM2_SIZE]]);
    fprintf(f, "{");
    if (*code != OP_TYPEEXACT) fprintf(f, "0,");
    fprintf(f, "%d}", GET2(code,1));
    if (*code == OP_TYPEMINUPTO) fprintf(f, "?");
      else if (*code == OP_TYPEPOSUPTO) fprintf(f, "+");
    break;

    case OP_NOTI:
    flag = "/i";
    /* Fall through */
    case OP_NOT:
    fprintf(f, " %s [^", flag);
    extra = print_char(f, code + 1, utf);
    fprintf(f, "]");
    break;

    case OP_NOTSTARI:
    case OP_NOTMINSTARI:
    case OP_NOTPOSSTARI:
    case OP_NOTPLUSI:
    case OP_NOTMINPLUSI:
    case OP_NOTPOSPLUSI:
    case OP_NOTQUERYI:
    case OP_NOTMINQUERYI:
    case OP_NOTPOSQUERYI:
    flag = "/i";
    /* Fall through */

    case OP_NOTSTAR:
    case OP_NOTMINSTAR:
    case OP_NOTPOSSTAR:
    case OP_NOTPLUS:
    case OP_NOTMINPLUS:
    case OP_NOTPOSPLUS:
    case OP_NOTQUERY:
    case OP_NOTMINQUERY:
    case OP_NOTPOSQUERY:
    fprintf(f, " %s [^", flag);
    extra = print_char(f, code + 1, utf);
    fprintf(f, "]%s", OP_names[*code]);
    break;

    case OP_NOTEXACTI:
    case OP_NOTUPTOI:
    case OP_NOTMINUPTOI:
    case OP_NOTPOSUPTOI:
    flag = "/i";
    /* Fall through */

    case OP_NOTEXACT:
    case OP_NOTUPTO:
    case OP_NOTMINUPTO:
    case OP_NOTPOSUPTO:
    fprintf(f, " %s [^", flag);
    extra = print_char(f, code + 1 + IMM2_SIZE, utf);
    fprintf(f, "]{");
    if (*code != OP_NOTEXACT && *code != OP_NOTEXACTI) fprintf(f, "0,");
    fprintf(f, "%d}", GET2(code,1));
    if (*code == OP_NOTMINUPTO || *code == OP_NOTMINUPTOI) fprintf(f, "?");
      else
    if (*code == OP_NOTPOSUPTO || *code == OP_NOTPOSUPTOI) fprintf(f, "+");
    break;

    case OP_RECURSE:
    if (print_lengths) fprintf(f, "%3d ", GET(code, 1));
      else fprintf(f, "    ");
    fprintf(f, "%s", OP_names[*code]);
    break;

    case OP_REFI:
    flag = "/i";
    /* Fall through */
    case OP_REF:
    fprintf(f, " %s \\%d", flag, GET2(code,1));
    ccode = code + OP_lengths[*code];
    goto CLASS_REF_REPEAT;

    case OP_DNREFI:
    flag = "/i";
    /* Fall through */
    case OP_DNREF:
      {
      PCRE2_SPTR entry = nametable + (GET2(code, 1) * nesize) + IMM2_SIZE;
      fprintf(f, " %s \\k<", flag);
      print_custring(f, entry);
      fprintf(f, ">%d", GET2(code, 1 + IMM2_SIZE));
      }
    ccode = code + OP_lengths[*code];
    goto CLASS_REF_REPEAT;

    case OP_CALLOUT:
    fprintf(f, "    %s %d %d %d", OP_names[*code], code[1 + 2*LINK_SIZE],
      GET(code, 1), GET(code, 1 + LINK_SIZE));
    break;

    case OP_CALLOUT_STR:
    c = code[1 + 4*LINK_SIZE];
    fprintf(f, "    %s %c", OP_names[*code], c);
    extra = GET(code, 1 + 2*LINK_SIZE);
    print_custring_bylen(f, code + 2 + 4*LINK_SIZE, extra - 3 - 4*LINK_SIZE);
    for (i = 0; PRIV(callout_start_delims)[i] != 0; i++)
      if (c == PRIV(callout_start_delims)[i])
        {
        c = PRIV(callout_end_delims)[i];
        break;
        }
    fprintf(f, "%c %d %d %d", c, GET(code, 1 + 3*LINK_SIZE), GET(code, 1),
      GET(code, 1 + LINK_SIZE));
    break;

    case OP_PROP:
    case OP_NOTPROP:
    print_prop(f, code, "    ", "");
    break;

    /* OP_XCLASS cannot occur in 8-bit, non-UTF mode. However, there's no harm
    in having this code always here, and it makes it less messy without all
    those #ifdefs. */

    case OP_CLASS:
    case OP_NCLASS:
    case OP_XCLASS:
      {
      unsigned int min, max;
      BOOL printmap;
      BOOL invertmap = FALSE;
      uint8_t *map;
      uint8_t inverted_map[32];

      fprintf(f, "    [");

      if (*code == OP_XCLASS)
        {
        extra = GET(code, 1);
        ccode = code + LINK_SIZE + 1;
        printmap = (*ccode & XCL_MAP) != 0;
        if ((*ccode & XCL_NOT) != 0)
          {
          invertmap = (*ccode & XCL_HASPROP) == 0;
          fprintf(f, "^");
          }
        ccode++;
        }
      else
        {
        printmap = TRUE;
        ccode = code + 1;
        }

      /* Print a bit map */

      if (printmap)
        {
        map = (uint8_t *)ccode;
        if (invertmap)
          {
          for (i = 0; i < 32; i++) inverted_map[i] = ~map[i];
          map = inverted_map;
          }

        for (i = 0; i < 256; i++)
          {
          if ((map[i/8] & (1 << (i&7))) != 0)
            {
            int j;
            for (j = i+1; j < 256; j++)
              if ((map[j/8] & (1 << (j&7))) == 0) break;
            if (i == '-' || i == ']') fprintf(f, "\\");
            if (PRINTABLE(i)) fprintf(f, "%c", i);
              else fprintf(f, "\\x%02x", i);
            if (--j > i)
              {
              if (j != i + 1) fprintf(f, "-");
              if (j == '-' || j == ']') fprintf(f, "\\");
              if (PRINTABLE(j)) fprintf(f, "%c", j);
                else fprintf(f, "\\x%02x", j);
              }
            i = j;
            }
          }
        ccode += 32 / sizeof(PCRE2_UCHAR);
        }

      /* For an XCLASS there is always some additional data */

      if (*code == OP_XCLASS)
        {
        PCRE2_UCHAR ch;
        while ((ch = *ccode++) != XCL_END)
          {
          BOOL not = FALSE;
          const char *notch = "";

          switch(ch)
            {
            case XCL_NOTPROP:
            not = TRUE;
            notch = "^";
            /* Fall through */

            case XCL_PROP:
              {
              unsigned int ptype = *ccode++;
              unsigned int pvalue = *ccode++;

              switch(ptype)
                {
                case PT_PXGRAPH:
                fprintf(f, "[:%sgraph:]", notch);
                break;

                case PT_PXPRINT:
                fprintf(f, "[:%sprint:]", notch);
                break;

                case PT_PXPUNCT:
                fprintf(f, "[:%spunct:]", notch);
                break;

                default:
                fprintf(f, "\\%c{%s}", (not? 'P':'p'),
                  get_ucpname(ptype, pvalue));
                break;
                }
              }
            break;

            default:
            ccode += 1 + print_char(f, ccode, utf);
            if (ch == XCL_RANGE)
              {
              fprintf(f, "-");
              ccode += 1 + print_char(f, ccode, utf);
              }
            break;
            }
          }
        }

      /* Indicate a non-UTF class which was created by negation */

      fprintf(f, "]%s", (*code == OP_NCLASS)? " (neg)" : "");

      /* Handle repeats after a class or a back reference */

      CLASS_REF_REPEAT:
      switch(*ccode)
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
        fprintf(f, "%s", OP_names[*ccode]);
        extra += OP_lengths[*ccode];
        break;

        case OP_CRRANGE:
        case OP_CRMINRANGE:
        case OP_CRPOSRANGE:
        min = GET2(ccode,1);
        max = GET2(ccode,1 + IMM2_SIZE);
        if (max == 0) fprintf(f, "{%u,}", min);
        else fprintf(f, "{%u,%u}", min, max);
        if (*ccode == OP_CRMINRANGE) fprintf(f, "?");
        else if (*ccode == OP_CRPOSRANGE) fprintf(f, "+");
        extra += OP_lengths[*ccode];
        break;

        /* Do nothing if it's not a repeat; this code stops picky compilers
        warning about the lack of a default code path. */

        default:
        break;
        }
      }
    break;

    case OP_MARK:
    case OP_PRUNE_ARG:
    case OP_SKIP_ARG:
    case OP_THEN_ARG:
    fprintf(f, "    %s ", OP_names[*code]);
    print_custring_bylen(f, code + 2, code[1]);
    extra += code[1];
    break;

    case OP_THEN:
    fprintf(f, "    %s", OP_names[*code]);
    break;

    case OP_CIRCM:
    case OP_DOLLM:
    flag = "/m";
    /* Fall through */

    /* Anything else is just an item with no data, but possibly a flag. */

    default:
    fprintf(f, " %s %s", flag, OP_names[*code]);
    break;
    }

  code += OP_lengths[*code] + extra;
  fprintf(f, "\n");
  }
}

/* End of pcre2_printint.c */
