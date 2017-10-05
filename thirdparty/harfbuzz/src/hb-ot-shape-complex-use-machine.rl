/*
 * Copyright © 2015  Mozilla Foundation.
 * Copyright © 2015  Google, Inc.
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
 * Mozilla Author(s): Jonathan Kew
 * Google Author(s): Behdad Esfahbod
 */

#ifndef HB_OT_SHAPE_COMPLEX_USE_MACHINE_HH
#define HB_OT_SHAPE_COMPLEX_USE_MACHINE_HH

#include "hb-private.hh"

%%{
  machine use_syllable_machine;
  alphtype unsigned char;
  write data;
}%%

%%{

# Same order as enum use_category_t.  Not sure how to avoid duplication.

O	= 0; # OTHER

B	= 1; # BASE
IND	= 3; # BASE_IND
N	= 4; # BASE_NUM
GB	= 5; # BASE_OTHER
CGJ	= 6; # CGJ
#F	= 7; # CONS_FINAL
FM	= 8; # CONS_FINAL_MOD
#M	= 9; # CONS_MED
#CM	= 10; # CONS_MOD
SUB	= 11; # CONS_SUB
H	= 12; # HALANT

HN	= 13; # HALANT_NUM
ZWNJ	= 14; # Zero width non-joiner
ZWJ	= 15; # Zero width joiner
WJ	= 16; # Word joiner
Rsv	= 17; # Reserved characters
R	= 18; # REPHA
S	= 19; # SYM
#SM	= 20; # SYM_MOD
VS	= 21; # VARIATION_SELECTOR
#V	= 36; # VOWEL
#VM	= 40; # VOWEL_MOD

FAbv	= 24; # CONS_FINAL_ABOVE
FBlw	= 25; # CONS_FINAL_BELOW
FPst	= 26; # CONS_FINAL_POST
MAbv	= 27; # CONS_MED_ABOVE
MBlw	= 28; # CONS_MED_BELOW
MPst	= 29; # CONS_MED_POST
MPre	= 30; # CONS_MED_PRE
CMAbv	= 31; # CONS_MOD_ABOVE
CMBlw	= 32; # CONS_MOD_BELOW
VAbv	= 33; # VOWEL_ABOVE / VOWEL_ABOVE_BELOW / VOWEL_ABOVE_BELOW_POST / VOWEL_ABOVE_POST
VBlw	= 34; # VOWEL_BELOW / VOWEL_BELOW_POST
VPst	= 35; # VOWEL_POST	UIPC = Right
VPre	= 22; # VOWEL_PRE / VOWEL_PRE_ABOVE / VOWEL_PRE_ABOVE_POST / VOWEL_PRE_POST
VMAbv	= 37; # VOWEL_MOD_ABOVE
VMBlw	= 38; # VOWEL_MOD_BELOW
VMPst	= 39; # VOWEL_MOD_POST
VMPre	= 23; # VOWEL_MOD_PRE
SMAbv	= 41; # SYM_MOD_ABOVE
SMBlw	= 42; # SYM_MOD_BELOW
CS	= 43; # CONS_WITH_STACKER


# Override: Adhoc ZWJ placement. https://github.com/harfbuzz/harfbuzz/issues/542#issuecomment-353169729
consonant_modifiers = CMAbv* CMBlw* ((ZWJ?.H.ZWJ? B | SUB) VS? CMAbv? CMBlw*)*;
# Override: Allow two MBlw. https://github.com/harfbuzz/harfbuzz/issues/376
medial_consonants = MPre? MAbv? MBlw?.MBlw? MPst?;
dependent_vowels = VPre* VAbv* VBlw* VPst*;
vowel_modifiers = VMPre* VMAbv* VMBlw* VMPst*;
final_consonants = FAbv* FBlw* FPst* FM?;

virama_terminated_cluster =
	(R|CS)? (B | GB) VS?
	consonant_modifiers
	ZWJ?.H.ZWJ?
;
standard_cluster =
	(R|CS)? (B | GB) VS?
	consonant_modifiers
	medial_consonants
	dependent_vowels
	vowel_modifiers
	final_consonants
;

broken_cluster =
	R?
	consonant_modifiers
	medial_consonants
	dependent_vowels
	vowel_modifiers
	final_consonants
;

number_joiner_terminated_cluster = N VS? (HN N VS?)* HN;
numeral_cluster = N VS? (HN N VS?)*;
symbol_cluster = S VS? SMAbv* SMBlw*;
independent_cluster = (IND | O | Rsv | WJ) VS?;
other = any;

main := |*
	independent_cluster			=> { found_syllable (independent_cluster); };
	virama_terminated_cluster		=> { found_syllable (virama_terminated_cluster); };
	standard_cluster			=> { found_syllable (standard_cluster); };
	number_joiner_terminated_cluster	=> { found_syllable (number_joiner_terminated_cluster); };
	numeral_cluster				=> { found_syllable (numeral_cluster); };
	symbol_cluster				=> { found_syllable (symbol_cluster); };
	broken_cluster				=> { found_syllable (broken_cluster); };
	other					=> { found_syllable (non_cluster); };
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
    getkey info[p].use_category();
  }%%

  p = 0;
  pe = eof = buffer->len;

  unsigned int last = 0;
  unsigned int syllable_serial = 1;
  %%{
    write exec;
  }%%
}

#undef found_syllable

#endif /* HB_OT_SHAPE_COMPLEX_USE_MACHINE_HH */
