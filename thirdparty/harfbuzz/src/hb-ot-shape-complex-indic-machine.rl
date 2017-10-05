/*
 * Copyright © 2011,2012  Google, Inc.
 *
 *  This is part of HarfBuzz, a text shaping library.
 *
 * Permission is hereby granted, without written agreement and without
 * license or royalty fees, to use, copy, modify, and distribute this
 * software and its documentation for any purpose, provided that the
 * above copyright notice and the following two paragraphs appear in
 * all copies of this software.
 *
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE TO ANY PARTY FOR
 * DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES
 * ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN
 * IF THE COPYRIGHT HOLDER HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH
 * DAMAGE.
 *
 * THE COPYRIGHT HOLDER SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING,
 * BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE.  THE SOFTWARE PROVIDED HEREUNDER IS
 * ON AN "AS IS" BASIS, AND THE COPYRIGHT HOLDER HAS NO OBLIGATION TO
 * PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
 *
 * Google Author(s): Behdad Esfahbod
 */

#ifndef HB_OT_SHAPE_COMPLEX_INDIC_MACHINE_HH
#define HB_OT_SHAPE_COMPLEX_INDIC_MACHINE_HH

#include "hb-private.hh"

%%{
  machine indic_syllable_machine;
  alphtype unsigned char;
  write data;
}%%

%%{

# Same order as enum indic_category_t.  Not sure how to avoid duplication.
C    = 1;
V    = 2;
N    = 3;
H    = 4;
ZWNJ = 5;
ZWJ  = 6;
M    = 7;
SM   = 8;
A    = 10;
PLACEHOLDER = 11;
DOTTEDCIRCLE = 12;
RS    = 13;
Repha = 15;
Ra    = 16;
CM    = 17;
Symbol= 18;
CS    = 19;

c = (C | Ra);			# is_consonant
n = ((ZWNJ?.RS)? (N.N?)?);	# is_consonant_modifier
z = ZWJ|ZWNJ;			# is_joiner
reph = (Ra H | Repha);		# possible reph

cn = c.ZWJ?.n?;
forced_rakar = ZWJ H ZWJ Ra;
symbol = Symbol.N?;
matra_group = z{0,3}.M.N?.(H | forced_rakar)?;
syllable_tail = (z?.SM.SM?.ZWNJ?)? A{0,3}?;
halant_group = (z?.H.(ZWJ.N?)?);
final_halant_group = halant_group | H.ZWNJ;
medial_group = CM?;
halant_or_matra_group = (final_halant_group | (H.ZWJ)? matra_group{0,4});


consonant_syllable =	(Repha|CS)? (cn.halant_group){0,4} cn medial_group halant_or_matra_group syllable_tail;
vowel_syllable =	reph? V.n? (ZWJ | (halant_group.cn){0,4} medial_group halant_or_matra_group syllable_tail);
standalone_cluster =	((Repha|CS)? PLACEHOLDER | reph? DOTTEDCIRCLE).n? (halant_group.cn){0,4} medial_group halant_or_matra_group syllable_tail;
symbol_cluster = 	symbol syllable_tail;
broken_cluster =	reph? n? (halant_group.cn){0,4} medial_group halant_or_matra_group syllable_tail;
other =			any;

main := |*
	consonant_syllable	=> { found_syllable (consonant_syllable); };
	vowel_syllable		=> { found_syllable (vowel_syllable); };
	standalone_cluster	=> { found_syllable (standalone_cluster); };
	symbol_cluster		=> { found_syllable (symbol_cluster); };
	broken_cluster		=> { found_syllable (broken_cluster); };
	other			=> { found_syllable (non_indic_cluster); };
*|;


}%%

#define found_syllable(syllable_type) \
  HB_STMT_START { \
    if (0) fprintf (stderr, "syllable %d..%d %s\n", last, p+1, #syllable_type); \
    for (unsigned int i = last; i < p+1; i++) \
      info[i].syllable() = (syllable_serial << 4) | syllable_type; \
    last = p+1; \
    syllable_serial++; \
    if (unlikely (syllable_serial == 16)) syllable_serial = 1; \
  } HB_STMT_END

static void
find_syllables (hb_buffer_t *buffer)
{
  unsigned int p, pe, eof, ts HB_UNUSED, te, act;
  int cs;
  hb_glyph_info_t *info = buffer->info;
  %%{
    write init;
    getkey info[p].indic_category();
  }%%

  p = 0;
  pe = eof = buffer->len;

  unsigned int last = 0;
  unsigned int syllable_serial = 1;
  %%{
    write exec;
  }%%
}

#endif /* HB_OT_SHAPE_COMPLEX_INDIC_MACHINE_HH */
