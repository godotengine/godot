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

#ifndef HB_OT_SHAPE_COMPLEX_KHMER_MACHINE_HH
#define HB_OT_SHAPE_COMPLEX_KHMER_MACHINE_HH

#include "hb-private.hh"

%%{
  machine khmer_syllable_machine;
  alphtype unsigned char;
  write data;
}%%

%%{

# Same order as enum khmer_category_t.  Not sure how to avoid duplication.
C    = 1;
V    = 2;
N    = 3;
ZWNJ = 5;
ZWJ  = 6;
M    = 7;
SM   = 8;
PLACEHOLDER = 11;
DOTTEDCIRCLE = 12;
RS    = 13;
Coeng = 14;
Ra    = 16;

c = (C | Ra | V);		# is_consonant
n = ((ZWNJ?.RS)? (N.N?)?);	# is_consonant_modifier
z = ZWJ|ZWNJ;			# is_joiner

cn = c.n?;
matra_group = z?.M.N?;
syllable_tail = (SM.SM?)?;


broken_cluster =	n? (Coeng.cn)* matra_group* (Coeng.cn)? syllable_tail;
consonant_syllable =	(c|PLACEHOLDER|DOTTEDCIRCLE) broken_cluster;
other =			any;

main := |*
	consonant_syllable	=> { found_syllable (consonant_syllable); };
	broken_cluster		=> { found_syllable (broken_cluster); };
	other			=> { found_syllable (non_khmer_cluster); };
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
  unsigned int p, pe, eof, ts HB_UNUSED, te, act HB_UNUSED;
  int cs;
  hb_glyph_info_t *info = buffer->info;
  %%{
    write init;
    getkey info[p].khmer_category();
  }%%

  p = 0;
  pe = eof = buffer->len;

  unsigned int last = 0;
  unsigned int syllable_serial = 1;
  %%{
    write exec;
  }%%
}

#endif /* HB_OT_SHAPE_COMPLEX_KHMER_MACHINE_HH */
