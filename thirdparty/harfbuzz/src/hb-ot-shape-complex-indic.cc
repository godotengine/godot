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

#include "hb.hh"

#ifndef HB_NO_OT_SHAPE

#include "hb-ot-shape-complex-indic.hh"
#include "hb-ot-shape-complex-indic-machine.hh"
#include "hb-ot-shape-complex-vowel-constraints.hh"
#include "hb-ot-layout.hh"


/*
 * Indic shaper.
 */


/*
 * Indic configurations.  Note that we do not want to keep every single script-specific
 * behavior in these tables necessarily.  This should mainly be used for per-script
 * properties that are cheaper keeping here, than in the code.  Ie. if, say, one and
 * only one script has an exception, that one script can be if'ed directly in the code,
 * instead of adding a new flag in these structs.
 */

enum base_position_t {
  BASE_POS_LAST_SINHALA,
  BASE_POS_LAST
};
enum reph_position_t {
  REPH_POS_AFTER_MAIN  = POS_AFTER_MAIN,
  REPH_POS_BEFORE_SUB  = POS_BEFORE_SUB,
  REPH_POS_AFTER_SUB   = POS_AFTER_SUB,
  REPH_POS_BEFORE_POST = POS_BEFORE_POST,
  REPH_POS_AFTER_POST  = POS_AFTER_POST
};
enum reph_mode_t {
  REPH_MODE_IMPLICIT,  /* Reph formed out of initial Ra,H sequence. */
  REPH_MODE_EXPLICIT,  /* Reph formed out of initial Ra,H,ZWJ sequence. */
  REPH_MODE_LOG_REPHA  /* Encoded Repha character, needs reordering. */
};
enum blwf_mode_t {
  BLWF_MODE_PRE_AND_POST, /* Below-forms feature applied to pre-base and post-base. */
  BLWF_MODE_POST_ONLY     /* Below-forms feature applied to post-base only. */
};
struct indic_config_t
{
  hb_script_t     script;
  bool            has_old_spec;
  hb_codepoint_t  virama;
  base_position_t base_pos;
  reph_position_t reph_pos;
  reph_mode_t     reph_mode;
  blwf_mode_t     blwf_mode;
};

static const indic_config_t indic_configs[] =
{
  /* Default.  Should be first. */
  {HB_SCRIPT_INVALID,	false,      0,BASE_POS_LAST, REPH_POS_BEFORE_POST,REPH_MODE_IMPLICIT, BLWF_MODE_PRE_AND_POST},
  {HB_SCRIPT_DEVANAGARI,true, 0x094Du,BASE_POS_LAST, REPH_POS_BEFORE_POST,REPH_MODE_IMPLICIT, BLWF_MODE_PRE_AND_POST},
  {HB_SCRIPT_BENGALI,	true, 0x09CDu,BASE_POS_LAST, REPH_POS_AFTER_SUB,  REPH_MODE_IMPLICIT, BLWF_MODE_PRE_AND_POST},
  {HB_SCRIPT_GURMUKHI,	true, 0x0A4Du,BASE_POS_LAST, REPH_POS_BEFORE_SUB, REPH_MODE_IMPLICIT, BLWF_MODE_PRE_AND_POST},
  {HB_SCRIPT_GUJARATI,	true, 0x0ACDu,BASE_POS_LAST, REPH_POS_BEFORE_POST,REPH_MODE_IMPLICIT, BLWF_MODE_PRE_AND_POST},
  {HB_SCRIPT_ORIYA,	true, 0x0B4Du,BASE_POS_LAST, REPH_POS_AFTER_MAIN, REPH_MODE_IMPLICIT, BLWF_MODE_PRE_AND_POST},
  {HB_SCRIPT_TAMIL,	true, 0x0BCDu,BASE_POS_LAST, REPH_POS_AFTER_POST, REPH_MODE_IMPLICIT, BLWF_MODE_PRE_AND_POST},
  {HB_SCRIPT_TELUGU,	true, 0x0C4Du,BASE_POS_LAST, REPH_POS_AFTER_POST, REPH_MODE_EXPLICIT, BLWF_MODE_POST_ONLY},
  {HB_SCRIPT_KANNADA,	true, 0x0CCDu,BASE_POS_LAST, REPH_POS_AFTER_POST, REPH_MODE_IMPLICIT, BLWF_MODE_POST_ONLY},
  {HB_SCRIPT_MALAYALAM,	true, 0x0D4Du,BASE_POS_LAST, REPH_POS_AFTER_MAIN, REPH_MODE_LOG_REPHA,BLWF_MODE_PRE_AND_POST},
  {HB_SCRIPT_SINHALA,	false,0x0DCAu,BASE_POS_LAST_SINHALA,
						     REPH_POS_AFTER_POST, REPH_MODE_EXPLICIT, BLWF_MODE_PRE_AND_POST},
};



/*
 * Indic shaper.
 */

static const hb_ot_map_feature_t
indic_features[] =
{
  /*
   * Basic features.
   * These features are applied in order, one at a time, after initial_reordering.
   */
  {HB_TAG('n','u','k','t'), F_GLOBAL_MANUAL_JOINERS},
  {HB_TAG('a','k','h','n'), F_GLOBAL_MANUAL_JOINERS},
  {HB_TAG('r','p','h','f'),        F_MANUAL_JOINERS},
  {HB_TAG('r','k','r','f'), F_GLOBAL_MANUAL_JOINERS},
  {HB_TAG('p','r','e','f'),        F_MANUAL_JOINERS},
  {HB_TAG('b','l','w','f'),        F_MANUAL_JOINERS},
  {HB_TAG('a','b','v','f'),        F_MANUAL_JOINERS},
  {HB_TAG('h','a','l','f'),        F_MANUAL_JOINERS},
  {HB_TAG('p','s','t','f'),        F_MANUAL_JOINERS},
  {HB_TAG('v','a','t','u'), F_GLOBAL_MANUAL_JOINERS},
  {HB_TAG('c','j','c','t'), F_GLOBAL_MANUAL_JOINERS},
  /*
   * Other features.
   * These features are applied all at once, after final_reordering
   * but before clearing syllables.
   * Default Bengali font in Windows for example has intermixed
   * lookups for init,pres,abvs,blws features.
   */
  {HB_TAG('i','n','i','t'),        F_MANUAL_JOINERS},
  {HB_TAG('p','r','e','s'), F_GLOBAL_MANUAL_JOINERS},
  {HB_TAG('a','b','v','s'), F_GLOBAL_MANUAL_JOINERS},
  {HB_TAG('b','l','w','s'), F_GLOBAL_MANUAL_JOINERS},
  {HB_TAG('p','s','t','s'), F_GLOBAL_MANUAL_JOINERS},
  {HB_TAG('h','a','l','n'), F_GLOBAL_MANUAL_JOINERS},
};

/*
 * Must be in the same order as the indic_features array.
 */
enum {
  _INDIC_NUKT,
  _INDIC_AKHN,
  INDIC_RPHF,
  _INDIC_RKRF,
  INDIC_PREF,
  INDIC_BLWF,
  INDIC_ABVF,
  INDIC_HALF,
  INDIC_PSTF,
  _INDIC_VATU,
  _INDIC_CJCT,

  INDIC_INIT,
  _INDIC_PRES,
  _INDIC_ABVS,
  _INDIC_BLWS,
  _INDIC_PSTS,
  _INDIC_HALN,

  INDIC_NUM_FEATURES,
  INDIC_BASIC_FEATURES = INDIC_INIT, /* Don't forget to update this! */
};

static void
setup_syllables_indic (const hb_ot_shape_plan_t *plan,
		       hb_font_t *font,
		       hb_buffer_t *buffer);
static void
initial_reordering_indic (const hb_ot_shape_plan_t *plan,
			  hb_font_t *font,
			  hb_buffer_t *buffer);
static void
final_reordering_indic (const hb_ot_shape_plan_t *plan,
			hb_font_t *font,
			hb_buffer_t *buffer);

static void
collect_features_indic (hb_ot_shape_planner_t *plan)
{
  hb_ot_map_builder_t *map = &plan->map;

  /* Do this before any lookups have been applied. */
  map->add_gsub_pause (setup_syllables_indic);

  map->enable_feature (HB_TAG('l','o','c','l'));
  /* The Indic specs do not require ccmp, but we apply it here since if
   * there is a use of it, it's typically at the beginning. */
  map->enable_feature (HB_TAG('c','c','m','p'));


  unsigned int i = 0;
  map->add_gsub_pause (initial_reordering_indic);

  for (; i < INDIC_BASIC_FEATURES; i++) {
    map->add_feature (indic_features[i]);
    map->add_gsub_pause (nullptr);
  }

  map->add_gsub_pause (final_reordering_indic);

  for (; i < INDIC_NUM_FEATURES; i++)
    map->add_feature (indic_features[i]);

  map->enable_feature (HB_TAG('c','a','l','t'));
  map->enable_feature (HB_TAG('c','l','i','g'));

  map->add_gsub_pause (_hb_clear_syllables);
}

static void
override_features_indic (hb_ot_shape_planner_t *plan)
{
  plan->map.disable_feature (HB_TAG('l','i','g','a'));
}


struct indic_shape_plan_t
{
  bool load_virama_glyph (hb_font_t *font, hb_codepoint_t *pglyph) const
  {
    hb_codepoint_t glyph = virama_glyph.get_relaxed ();
    if (unlikely (glyph == (hb_codepoint_t) -1))
    {
      if (!config->virama || !font->get_nominal_glyph (config->virama, &glyph))
	glyph = 0;
      /* Technically speaking, the spec says we should apply 'locl' to virama too.
       * Maybe one day... */

      /* Our get_nominal_glyph() function needs a font, so we can't get the virama glyph
       * during shape planning...  Instead, overwrite it here. */
      virama_glyph.set_relaxed ((int) glyph);
    }

    *pglyph = glyph;
    return glyph != 0;
  }

  const indic_config_t *config;

  bool is_old_spec;
#ifndef HB_NO_UNISCRIBE_BUG_COMPATIBLE
  bool uniscribe_bug_compatible;
#else
  static constexpr bool uniscribe_bug_compatible = false;
#endif
  mutable hb_atomic_int_t virama_glyph;

  hb_indic_would_substitute_feature_t rphf;
  hb_indic_would_substitute_feature_t pref;
  hb_indic_would_substitute_feature_t blwf;
  hb_indic_would_substitute_feature_t pstf;
  hb_indic_would_substitute_feature_t vatu;

  hb_mask_t mask_array[INDIC_NUM_FEATURES];
};

static void *
data_create_indic (const hb_ot_shape_plan_t *plan)
{
  indic_shape_plan_t *indic_plan = (indic_shape_plan_t *) calloc (1, sizeof (indic_shape_plan_t));
  if (unlikely (!indic_plan))
    return nullptr;

  indic_plan->config = &indic_configs[0];
  for (unsigned int i = 1; i < ARRAY_LENGTH (indic_configs); i++)
    if (plan->props.script == indic_configs[i].script) {
      indic_plan->config = &indic_configs[i];
      break;
    }

  indic_plan->is_old_spec = indic_plan->config->has_old_spec && ((plan->map.chosen_script[0] & 0x000000FFu) != '2');
#ifndef HB_NO_UNISCRIBE_BUG_COMPATIBLE
  indic_plan->uniscribe_bug_compatible = hb_options ().uniscribe_bug_compatible;
#endif
  indic_plan->virama_glyph.set_relaxed (-1);

  /* Use zero-context would_substitute() matching for new-spec of the main
   * Indic scripts, and scripts with one spec only, but not for old-specs.
   * The new-spec for all dual-spec scripts says zero-context matching happens.
   *
   * However, testing with Malayalam shows that old and new spec both allow
   * context.  Testing with Bengali new-spec however shows that it doesn't.
   * So, the heuristic here is the way it is.  It should *only* be changed,
   * as we discover more cases of what Windows does.  DON'T TOUCH OTHERWISE.
   */
  bool zero_context = !indic_plan->is_old_spec && plan->props.script != HB_SCRIPT_MALAYALAM;
  indic_plan->rphf.init (&plan->map, HB_TAG('r','p','h','f'), zero_context);
  indic_plan->pref.init (&plan->map, HB_TAG('p','r','e','f'), zero_context);
  indic_plan->blwf.init (&plan->map, HB_TAG('b','l','w','f'), zero_context);
  indic_plan->pstf.init (&plan->map, HB_TAG('p','s','t','f'), zero_context);
  indic_plan->vatu.init (&plan->map, HB_TAG('v','a','t','u'), zero_context);

  for (unsigned int i = 0; i < ARRAY_LENGTH (indic_plan->mask_array); i++)
    indic_plan->mask_array[i] = (indic_features[i].flags & F_GLOBAL) ?
				 0 : plan->map.get_1_mask (indic_features[i].tag);

  return indic_plan;
}

static void
data_destroy_indic (void *data)
{
  free (data);
}

static indic_position_t
consonant_position_from_face (const indic_shape_plan_t *indic_plan,
			      const hb_codepoint_t consonant,
			      const hb_codepoint_t virama,
			      hb_face_t *face)
{
  /* For old-spec, the order of glyphs is Consonant,Virama,
   * whereas for new-spec, it's Virama,Consonant.  However,
   * some broken fonts (like Free Sans) simply copied lookups
   * from old-spec to new-spec without modification.
   * And oddly enough, Uniscribe seems to respect those lookups.
   * Eg. in the sequence U+0924,U+094D,U+0930, Uniscribe finds
   * base at 0.  The font however, only has lookups matching
   * 930,94D in 'blwf', not the expected 94D,930 (with new-spec
   * table).  As such, we simply match both sequences.  Seems
   * to work.
   *
   * Vatu is done as well, for:
   * https://github.com/harfbuzz/harfbuzz/issues/1587
   */
  hb_codepoint_t glyphs[3] = {virama, consonant, virama};
  if (indic_plan->blwf.would_substitute (glyphs  , 2, face) ||
      indic_plan->blwf.would_substitute (glyphs+1, 2, face) ||
      indic_plan->vatu.would_substitute (glyphs  , 2, face) ||
      indic_plan->vatu.would_substitute (glyphs+1, 2, face))
    return POS_BELOW_C;
  if (indic_plan->pstf.would_substitute (glyphs  , 2, face) ||
      indic_plan->pstf.would_substitute (glyphs+1, 2, face))
    return POS_POST_C;
  if (indic_plan->pref.would_substitute (glyphs  , 2, face) ||
      indic_plan->pref.would_substitute (glyphs+1, 2, face))
    return POS_POST_C;
  return POS_BASE_C;
}

static void
setup_masks_indic (const hb_ot_shape_plan_t *plan HB_UNUSED,
		   hb_buffer_t              *buffer,
		   hb_font_t                *font HB_UNUSED)
{
  HB_BUFFER_ALLOCATE_VAR (buffer, indic_category);
  HB_BUFFER_ALLOCATE_VAR (buffer, indic_position);

  /* We cannot setup masks here.  We save information about characters
   * and setup masks later on in a pause-callback. */

  unsigned int count = buffer->len;
  hb_glyph_info_t *info = buffer->info;
  for (unsigned int i = 0; i < count; i++)
    set_indic_properties (info[i]);
}

static void
setup_syllables_indic (const hb_ot_shape_plan_t *plan HB_UNUSED,
		       hb_font_t *font HB_UNUSED,
		       hb_buffer_t *buffer)
{
  find_syllables_indic (buffer);
  foreach_syllable (buffer, start, end)
    buffer->unsafe_to_break (start, end);
}

static int
compare_indic_order (const hb_glyph_info_t *pa, const hb_glyph_info_t *pb)
{
  int a = pa->indic_position();
  int b = pb->indic_position();

  return a < b ? -1 : a == b ? 0 : +1;
}



static void
update_consonant_positions_indic (const hb_ot_shape_plan_t *plan,
				  hb_font_t         *font,
				  hb_buffer_t       *buffer)
{
  const indic_shape_plan_t *indic_plan = (const indic_shape_plan_t *) plan->data;

  if (indic_plan->config->base_pos != BASE_POS_LAST)
    return;

  hb_codepoint_t virama;
  if (indic_plan->load_virama_glyph (font, &virama))
  {
    hb_face_t *face = font->face;
    unsigned int count = buffer->len;
    hb_glyph_info_t *info = buffer->info;
    for (unsigned int i = 0; i < count; i++)
      if (info[i].indic_position() == POS_BASE_C)
      {
	hb_codepoint_t consonant = info[i].codepoint;
	info[i].indic_position() = consonant_position_from_face (indic_plan, consonant, virama, face);
      }
  }
}


/* Rules from:
 * https://docs.microsqoft.com/en-us/typography/script-development/devanagari */

static void
initial_reordering_consonant_syllable (const hb_ot_shape_plan_t *plan,
				       hb_face_t *face,
				       hb_buffer_t *buffer,
				       unsigned int start, unsigned int end)
{
  const indic_shape_plan_t *indic_plan = (const indic_shape_plan_t *) plan->data;
  hb_glyph_info_t *info = buffer->info;

  /* https://github.com/harfbuzz/harfbuzz/issues/435#issuecomment-335560167
   * // For compatibility with legacy usage in Kannada,
   * // Ra+h+ZWJ must behave like Ra+ZWJ+h...
   */
  if (buffer->props.script == HB_SCRIPT_KANNADA &&
      start + 3 <= end &&
      is_one_of (info[start  ], FLAG (OT_Ra)) &&
      is_one_of (info[start+1], FLAG (OT_H)) &&
      is_one_of (info[start+2], FLAG (OT_ZWJ)))
  {
    buffer->merge_clusters (start+1, start+3);
    hb_glyph_info_t tmp = info[start+1];
    info[start+1] = info[start+2];
    info[start+2] = tmp;
  }

  /* 1. Find base consonant:
   *
   * The shaping engine finds the base consonant of the syllable, using the
   * following algorithm: starting from the end of the syllable, move backwards
   * until a consonant is found that does not have a below-base or post-base
   * form (post-base forms have to follow below-base forms), or that is not a
   * pre-base-reordering Ra, or arrive at the first consonant. The consonant
   * stopped at will be the base.
   *
   *   o If the syllable starts with Ra + Halant (in a script that has Reph)
   *     and has more than one consonant, Ra is excluded from candidates for
   *     base consonants.
   */

  unsigned int base = end;
  bool has_reph = false;

  {
    /* -> If the syllable starts with Ra + Halant (in a script that has Reph)
     *    and has more than one consonant, Ra is excluded from candidates for
     *    base consonants. */
    unsigned int limit = start;
    if (indic_plan->mask_array[INDIC_RPHF] &&
	start + 3 <= end &&
	(
	 (indic_plan->config->reph_mode == REPH_MODE_IMPLICIT && !is_joiner (info[start + 2])) ||
	 (indic_plan->config->reph_mode == REPH_MODE_EXPLICIT && info[start + 2].indic_category() == OT_ZWJ)
	))
    {
      /* See if it matches the 'rphf' feature. */
      hb_codepoint_t glyphs[3] = {info[start].codepoint,
				  info[start + 1].codepoint,
				  indic_plan->config->reph_mode == REPH_MODE_EXPLICIT ?
				    info[start + 2].codepoint : 0};
      if (indic_plan->rphf.would_substitute (glyphs, 2, face) ||
	  (indic_plan->config->reph_mode == REPH_MODE_EXPLICIT &&
	   indic_plan->rphf.would_substitute (glyphs, 3, face)))
      {
	limit += 2;
	while (limit < end && is_joiner (info[limit]))
	  limit++;
	base = start;
	has_reph = true;
      }
    } else if (indic_plan->config->reph_mode == REPH_MODE_LOG_REPHA && info[start].indic_category() == OT_Repha)
    {
	limit += 1;
	while (limit < end && is_joiner (info[limit]))
	  limit++;
	base = start;
	has_reph = true;
    }

    switch (indic_plan->config->base_pos)
    {
      case BASE_POS_LAST:
      {
	/* -> starting from the end of the syllable, move backwards */
	unsigned int i = end;
	bool seen_below = false;
	do {
	  i--;
	  /* -> until a consonant is found */
	  if (is_consonant (info[i]))
	  {
	    /* -> that does not have a below-base or post-base form
	     * (post-base forms have to follow below-base forms), */
	    if (info[i].indic_position() != POS_BELOW_C &&
		(info[i].indic_position() != POS_POST_C || seen_below))
	    {
	      base = i;
	      break;
	    }
	    if (info[i].indic_position() == POS_BELOW_C)
	      seen_below = true;

	    /* -> or that is not a pre-base-reordering Ra,
	     *
	     * IMPLEMENTATION NOTES:
	     *
	     * Our pre-base-reordering Ra's are marked POS_POST_C, so will be skipped
	     * by the logic above already.
	     */

	    /* -> or arrive at the first consonant. The consonant stopped at will
	     * be the base. */
	    base = i;
	  }
	  else
	  {
	    /* A ZWJ after a Halant stops the base search, and requests an explicit
	     * half form.
	     * A ZWJ before a Halant, requests a subjoined form instead, and hence
	     * search continues.  This is particularly important for Bengali
	     * sequence Ra,H,Ya that should form Ya-Phalaa by subjoining Ya. */
	    if (start < i &&
		info[i].indic_category() == OT_ZWJ &&
		info[i - 1].indic_category() == OT_H)
	      break;
	  }
	} while (i > limit);
      }
      break;

      case BASE_POS_LAST_SINHALA:
      {
	/* Sinhala base positioning is slightly different from main Indic, in that:
	 * 1. Its ZWJ behavior is different,
	 * 2. We don't need to look into the font for consonant positions.
	 */

	if (!has_reph)
	  base = limit;

	/* Find the last base consonant that is not blocked by ZWJ.  If there is
	 * a ZWJ right before a base consonant, that would request a subjoined form. */
	for (unsigned int i = limit; i < end; i++)
	  if (is_consonant (info[i]))
	  {
	    if (limit < i && info[i - 1].indic_category() == OT_ZWJ)
	      break;
	    else
	      base = i;
	  }

	/* Mark all subsequent consonants as below. */
	for (unsigned int i = base + 1; i < end; i++)
	  if (is_consonant (info[i]))
	    info[i].indic_position() = POS_BELOW_C;
      }
      break;
    }

    /* -> If the syllable starts with Ra + Halant (in a script that has Reph)
     *    and has more than one consonant, Ra is excluded from candidates for
     *    base consonants.
     *
     *  Only do this for unforced Reph. (ie. not for Ra,H,ZWJ. */
    if (has_reph && base == start && limit - base <= 2) {
      /* Have no other consonant, so Reph is not formed and Ra becomes base. */
      has_reph = false;
    }
  }


  /* 2. Decompose and reorder Matras:
   *
   * Each matra and any syllable modifier sign in the syllable are moved to the
   * appropriate position relative to the consonant(s) in the syllable. The
   * shaping engine decomposes two- or three-part matras into their constituent
   * parts before any repositioning. Matra characters are classified by which
   * consonant in a conjunct they have affinity for and are reordered to the
   * following positions:
   *
   *   o Before first half form in the syllable
   *   o After subjoined consonants
   *   o After post-form consonant
   *   o After main consonant (for above marks)
   *
   * IMPLEMENTATION NOTES:
   *
   * The normalize() routine has already decomposed matras for us, so we don't
   * need to worry about that.
   */


  /* 3.  Reorder marks to canonical order:
   *
   * Adjacent nukta and halant or nukta and vedic sign are always repositioned
   * if necessary, so that the nukta is first.
   *
   * IMPLEMENTATION NOTES:
   *
   * We don't need to do this: the normalize() routine already did this for us.
   */


  /* Reorder characters */

  for (unsigned int i = start; i < base; i++)
    info[i].indic_position() = hb_min (POS_PRE_C, (indic_position_t) info[i].indic_position());

  if (base < end)
    info[base].indic_position() = POS_BASE_C;

  /* Mark final consonants.  A final consonant is one appearing after a matra.
   * Happens in Sinhala. */
  for (unsigned int i = base + 1; i < end; i++)
    if (info[i].indic_category() == OT_M) {
      for (unsigned int j = i + 1; j < end; j++)
	if (is_consonant (info[j])) {
	 info[j].indic_position() = POS_FINAL_C;
	 break;
       }
      break;
    }

  /* Handle beginning Ra */
  if (has_reph)
    info[start].indic_position() = POS_RA_TO_BECOME_REPH;

  /* For old-style Indic script tags, move the first post-base Halant after
   * last consonant.
   *
   * Reports suggest that in some scripts Uniscribe does this only if there
   * is *not* a Halant after last consonant already.  We know that is the
   * case for Kannada, while it reorders unconditionally in other scripts,
   * eg. Malayalam, Bengali, and Devanagari.  We don't currently know about
   * other scripts, so we block Kannada.
   *
   * Kannada test case:
   * U+0C9A,U+0CCD,U+0C9A,U+0CCD
   * With some versions of Lohit Kannada.
   * https://bugs.freedesktop.org/show_bug.cgi?id=59118
   *
   * Malayalam test case:
   * U+0D38,U+0D4D,U+0D31,U+0D4D,U+0D31,U+0D4D
   * With lohit-ttf-20121122/Lohit-Malayalam.ttf
   *
   * Bengali test case:
   * U+0998,U+09CD,U+09AF,U+09CD
   * With Windows XP vrinda.ttf
   * https://github.com/harfbuzz/harfbuzz/issues/1073
   *
   * Devanagari test case:
   * U+091F,U+094D,U+0930,U+094D
   * With chandas.ttf
   * https://github.com/harfbuzz/harfbuzz/issues/1071
   */
  if (indic_plan->is_old_spec)
  {
    bool disallow_double_halants = buffer->props.script == HB_SCRIPT_KANNADA;
    for (unsigned int i = base + 1; i < end; i++)
      if (info[i].indic_category() == OT_H)
      {
	unsigned int j;
	for (j = end - 1; j > i; j--)
	  if (is_consonant (info[j]) ||
	      (disallow_double_halants && info[j].indic_category() == OT_H))
	    break;
	if (info[j].indic_category() != OT_H && j > i) {
	  /* Move Halant to after last consonant. */
	  hb_glyph_info_t t = info[i];
	  memmove (&info[i], &info[i + 1], (j - i) * sizeof (info[0]));
	  info[j] = t;
	}
	break;
      }
  }

  /* Attach misc marks to previous char to move with them. */
  {
    indic_position_t last_pos = POS_START;
    for (unsigned int i = start; i < end; i++)
    {
      if ((FLAG_UNSAFE (info[i].indic_category()) & (JOINER_FLAGS | FLAG (OT_N) | FLAG (OT_RS) | MEDIAL_FLAGS | FLAG (OT_H))))
      {
	info[i].indic_position() = last_pos;
	if (unlikely (info[i].indic_category() == OT_H &&
		      info[i].indic_position() == POS_PRE_M))
	{
	  /*
	   * Uniscribe doesn't move the Halant with Left Matra.
	   * TEST: U+092B,U+093F,U+094DE
	   * We follow.  This is important for the Sinhala
	   * U+0DDA split matra since it decomposes to U+0DD9,U+0DCA
	   * where U+0DD9 is a left matra and U+0DCA is the virama.
	   * We don't want to move the virama with the left matra.
	   * TEST: U+0D9A,U+0DDA
	   */
	  for (unsigned int j = i; j > start; j--)
	    if (info[j - 1].indic_position() != POS_PRE_M) {
	      info[i].indic_position() = info[j - 1].indic_position();
	      break;
	    }
	}
      } else if (info[i].indic_position() != POS_SMVD) {
	last_pos = (indic_position_t) info[i].indic_position();
      }
    }
  }
  /* For post-base consonants let them own anything before them
   * since the last consonant or matra. */
  {
    unsigned int last = base;
    for (unsigned int i = base + 1; i < end; i++)
      if (is_consonant (info[i]))
      {
	for (unsigned int j = last + 1; j < i; j++)
	  if (info[j].indic_position() < POS_SMVD)
	    info[j].indic_position() = info[i].indic_position();
	last = i;
      } else if (info[i].indic_category() == OT_M)
	last = i;
  }


  {
    /* Use syllable() for sort accounting temporarily. */
    unsigned int syllable = info[start].syllable();
    for (unsigned int i = start; i < end; i++)
      info[i].syllable() = i - start;

    /* Sit tight, rock 'n roll! */
    hb_stable_sort (info + start, end - start, compare_indic_order);
    /* Find base again */
    base = end;
    for (unsigned int i = start; i < end; i++)
      if (info[i].indic_position() == POS_BASE_C)
      {
	base = i;
	break;
      }
    /* Things are out-of-control for post base positions, they may shuffle
     * around like crazy.  In old-spec mode, we move halants around, so in
     * that case merge all clusters after base.  Otherwise, check the sort
     * order and merge as needed.
     * For pre-base stuff, we handle cluster issues in final reordering.
     *
     * We could use buffer->sort() for this, if there was no special
     * reordering of pre-base stuff happening later...
     * We don't want to merge_clusters all of that, which buffer->sort()
     * would.  Here's a concrete example:
     *
     * Assume there's a pre-base consonant and explicit Halant before base,
     * followed by a prebase-reordering (left) Matra:
     *
     *   C,H,ZWNJ,B,M
     *
     * At this point in reordering we would have:
     *
     *   M,C,H,ZWNJ,B
     *
     * whereas in final reordering we will bring the Matra closer to Base:
     *
     *   C,H,ZWNJ,M,B
     *
     * That's why we don't want to merge-clusters anything before the Base
     * at this point.  But if something moved from after Base to before it,
     * we should merge clusters from base to them.  In final-reordering, we
     * only move things around before base, and merge-clusters up to base.
     * These two merge-clusters from the two sides of base will interlock
     * to merge things correctly.  See:
     * https://github.com/harfbuzz/harfbuzz/issues/2272
     */
    if (indic_plan->is_old_spec || end - start > 127)
      buffer->merge_clusters (base, end);
    else
    {
      /* Note!  syllable() is a one-byte field. */
      for (unsigned int i = base; i < end; i++)
	if (info[i].syllable() != 255)
	{
	  unsigned int min = i;
	  unsigned int max = i;
	  unsigned int j = start + info[i].syllable();
	  while (j != i)
	  {
	    min = hb_min (min, j);
	    max = hb_max (max, j);
	    unsigned int next = start + info[j].syllable();
	    info[j].syllable() = 255; /* So we don't process j later again. */
	    j = next;
	  }
	  buffer->merge_clusters (hb_max (base, min), max + 1);
	}
    }

    /* Put syllable back in. */
    for (unsigned int i = start; i < end; i++)
      info[i].syllable() = syllable;
  }

  /* Setup masks now */

  {
    hb_mask_t mask;

    /* Reph */
    for (unsigned int i = start; i < end && info[i].indic_position() == POS_RA_TO_BECOME_REPH; i++)
      info[i].mask |= indic_plan->mask_array[INDIC_RPHF];

    /* Pre-base */
    mask = indic_plan->mask_array[INDIC_HALF];
    if (!indic_plan->is_old_spec &&
	indic_plan->config->blwf_mode == BLWF_MODE_PRE_AND_POST)
      mask |= indic_plan->mask_array[INDIC_BLWF];
    for (unsigned int i = start; i < base; i++)
      info[i].mask  |= mask;
    /* Base */
    mask = 0;
    if (base < end)
      info[base].mask |= mask;
    /* Post-base */
    mask = indic_plan->mask_array[INDIC_BLWF] |
	   indic_plan->mask_array[INDIC_ABVF] |
	   indic_plan->mask_array[INDIC_PSTF];
    for (unsigned int i = base + 1; i < end; i++)
      info[i].mask  |= mask;
  }

  if (indic_plan->is_old_spec &&
      buffer->props.script == HB_SCRIPT_DEVANAGARI)
  {
    /* Old-spec eye-lash Ra needs special handling.  From the
     * spec:
     *
     * "The feature 'below-base form' is applied to consonants
     * having below-base forms and following the base consonant.
     * The exception is vattu, which may appear below half forms
     * as well as below the base glyph. The feature 'below-base
     * form' will be applied to all such occurrences of Ra as well."
     *
     * Test case: U+0924,U+094D,U+0930,U+094d,U+0915
     * with Sanskrit 2003 font.
     *
     * However, note that Ra,Halant,ZWJ is the correct way to
     * request eyelash form of Ra, so we wouldbn't inhibit it
     * in that sequence.
     *
     * Test case: U+0924,U+094D,U+0930,U+094d,U+200D,U+0915
     */
    for (unsigned int i = start; i + 1 < base; i++)
      if (info[i  ].indic_category() == OT_Ra &&
	  info[i+1].indic_category() == OT_H  &&
	  (i + 2 == base ||
	   info[i+2].indic_category() != OT_ZWJ))
      {
	info[i  ].mask |= indic_plan->mask_array[INDIC_BLWF];
	info[i+1].mask |= indic_plan->mask_array[INDIC_BLWF];
      }
  }

  unsigned int pref_len = 2;
  if (indic_plan->mask_array[INDIC_PREF] && base + pref_len < end)
  {
    /* Find a Halant,Ra sequence and mark it for pre-base-reordering processing. */
    for (unsigned int i = base + 1; i + pref_len - 1 < end; i++) {
      hb_codepoint_t glyphs[2];
      for (unsigned int j = 0; j < pref_len; j++)
	glyphs[j] = info[i + j].codepoint;
      if (indic_plan->pref.would_substitute (glyphs, pref_len, face))
      {
	for (unsigned int j = 0; j < pref_len; j++)
	  info[i++].mask |= indic_plan->mask_array[INDIC_PREF];
	break;
      }
    }
  }

  /* Apply ZWJ/ZWNJ effects */
  for (unsigned int i = start + 1; i < end; i++)
    if (is_joiner (info[i])) {
      bool non_joiner = info[i].indic_category() == OT_ZWNJ;
      unsigned int j = i;

      do {
	j--;

	/* ZWJ/ZWNJ should disable CJCT.  They do that by simply
	 * being there, since we don't skip them for the CJCT
	 * feature (ie. F_MANUAL_ZWJ) */

	/* A ZWNJ disables HALF. */
	if (non_joiner)
	  info[j].mask &= ~indic_plan->mask_array[INDIC_HALF];

      } while (j > start && !is_consonant (info[j]));
    }
}

static void
initial_reordering_standalone_cluster (const hb_ot_shape_plan_t *plan,
				       hb_face_t *face,
				       hb_buffer_t *buffer,
				       unsigned int start, unsigned int end)
{
  /* We treat placeholder/dotted-circle as if they are consonants, so we
   * should just chain.  Only if not in compatibility mode that is... */

  const indic_shape_plan_t *indic_plan = (const indic_shape_plan_t *) plan->data;
  if (indic_plan->uniscribe_bug_compatible)
  {
    /* For dotted-circle, this is what Uniscribe does:
     * If dotted-circle is the last glyph, it just does nothing.
     * Ie. It doesn't form Reph. */
    if (buffer->info[end - 1].indic_category() == OT_DOTTEDCIRCLE)
      return;
  }

  initial_reordering_consonant_syllable (plan, face, buffer, start, end);
}

static void
initial_reordering_syllable_indic (const hb_ot_shape_plan_t *plan,
				   hb_face_t *face,
				   hb_buffer_t *buffer,
				   unsigned int start, unsigned int end)
{
  indic_syllable_type_t syllable_type = (indic_syllable_type_t) (buffer->info[start].syllable() & 0x0F);
  switch (syllable_type)
  {
    case indic_vowel_syllable: /* We made the vowels look like consonants.  So let's call the consonant logic! */
    case indic_consonant_syllable:
     initial_reordering_consonant_syllable (plan, face, buffer, start, end);
     break;

    case indic_broken_cluster: /* We already inserted dotted-circles, so just call the standalone_cluster. */
    case indic_standalone_cluster:
     initial_reordering_standalone_cluster (plan, face, buffer, start, end);
     break;

    case indic_symbol_cluster:
    case indic_non_indic_cluster:
      break;
  }
}

static void
initial_reordering_indic (const hb_ot_shape_plan_t *plan,
			  hb_font_t *font,
			  hb_buffer_t *buffer)
{
  if (!buffer->message (font, "start reordering indic initial"))
    return;

  update_consonant_positions_indic (plan, font, buffer);
  hb_syllabic_insert_dotted_circles (font, buffer,
				     indic_broken_cluster,
				     OT_DOTTEDCIRCLE,
				     OT_Repha);

  foreach_syllable (buffer, start, end)
    initial_reordering_syllable_indic (plan, font->face, buffer, start, end);

  (void) buffer->message (font, "end reordering indic initial");
}

static void
final_reordering_syllable_indic (const hb_ot_shape_plan_t *plan,
				 hb_buffer_t *buffer,
				 unsigned int start, unsigned int end)
{
  const indic_shape_plan_t *indic_plan = (const indic_shape_plan_t *) plan->data;
  hb_glyph_info_t *info = buffer->info;


  /* This function relies heavily on halant glyphs.  Lots of ligation
   * and possibly multiple substitutions happened prior to this
   * phase, and that might have messed up our properties.  Recover
   * from a particular case of that where we're fairly sure that a
   * class of OT_H is desired but has been lost. */
  /* We don't call load_virama_glyph(), since we know it's already
   * loaded. */
  hb_codepoint_t virama_glyph = indic_plan->virama_glyph.get_relaxed ();
  if (virama_glyph)
  {
    for (unsigned int i = start; i < end; i++)
      if (info[i].codepoint == virama_glyph &&
	  _hb_glyph_info_ligated (&info[i]) &&
	  _hb_glyph_info_multiplied (&info[i]))
      {
	/* This will make sure that this glyph passes is_halant() test. */
	info[i].indic_category() = OT_H;
	_hb_glyph_info_clear_ligated_and_multiplied (&info[i]);
      }
  }


  /* 4. Final reordering:
   *
   * After the localized forms and basic shaping forms GSUB features have been
   * applied (see below), the shaping engine performs some final glyph
   * reordering before applying all the remaining font features to the entire
   * syllable.
   */

  bool try_pref = !!indic_plan->mask_array[INDIC_PREF];

  /* Find base again */
  unsigned int base;
  for (base = start; base < end; base++)
    if (info[base].indic_position() >= POS_BASE_C)
    {
      if (try_pref && base + 1 < end)
      {
	for (unsigned int i = base + 1; i < end; i++)
	  if ((info[i].mask & indic_plan->mask_array[INDIC_PREF]) != 0)
	  {
	    if (!(_hb_glyph_info_substituted (&info[i]) &&
		  _hb_glyph_info_ligated_and_didnt_multiply (&info[i])))
	    {
	      /* Ok, this was a 'pref' candidate but didn't form any.
	       * Base is around here... */
	      base = i;
	      while (base < end && is_halant (info[base]))
		base++;
	      info[base].indic_position() = POS_BASE_C;

	      try_pref = false;
	    }
	    break;
	  }
      }
      /* For Malayalam, skip over unformed below- (but NOT post-) forms. */
      if (buffer->props.script == HB_SCRIPT_MALAYALAM)
      {
	for (unsigned int i = base + 1; i < end; i++)
	{
	  while (i < end && is_joiner (info[i]))
	    i++;
	  if (i == end || !is_halant (info[i]))
	    break;
	  i++; /* Skip halant. */
	  while (i < end && is_joiner (info[i]))
	    i++;
	  if (i < end && is_consonant (info[i]) && info[i].indic_position() == POS_BELOW_C)
	  {
	    base = i;
	    info[base].indic_position() = POS_BASE_C;
	  }
	}
      }

      if (start < base && info[base].indic_position() > POS_BASE_C)
	base--;
      break;
    }
  if (base == end && start < base &&
      is_one_of (info[base - 1], FLAG (OT_ZWJ)))
    base--;
  if (base < end)
    while (start < base &&
	   is_one_of (info[base], (FLAG (OT_N) | FLAG (OT_H))))
      base--;


  /*   o Reorder matras:
   *
   *     If a pre-base matra character had been reordered before applying basic
   *     features, the glyph can be moved closer to the main consonant based on
   *     whether half-forms had been formed. Actual position for the matra is
   *     defined as “after last standalone halant glyph, after initial matra
   *     position and before the main consonant”. If ZWJ or ZWNJ follow this
   *     halant, position is moved after it.
   *
   * IMPLEMENTATION NOTES:
   *
   * It looks like the last sentence is wrong.  Testing, with Windows 7 Uniscribe
   * and Devanagari shows that the behavior is best described as:
   *
   * "If ZWJ follows this halant, matra is NOT repositioned after this halant.
   *  If ZWNJ follows this halant, position is moved after it."
   *
   * Test case, with Adobe Devanagari or Nirmala UI:
   *
   *   U+091F,U+094D,U+200C,U+092F,U+093F
   *   (Matra moves to the middle, after ZWNJ.)
   *
   *   U+091F,U+094D,U+200D,U+092F,U+093F
   *   (Matra does NOT move, stays to the left.)
   *
   * https://github.com/harfbuzz/harfbuzz/issues/1070
   */

  if (start + 1 < end && start < base) /* Otherwise there can't be any pre-base matra characters. */
  {
    /* If we lost track of base, alas, position before last thingy. */
    unsigned int new_pos = base == end ? base - 2 : base - 1;

    /* Malayalam / Tamil do not have "half" forms or explicit virama forms.
     * The glyphs formed by 'half' are Chillus or ligated explicit viramas.
     * We want to position matra after them.
     */
    if (buffer->props.script != HB_SCRIPT_MALAYALAM && buffer->props.script != HB_SCRIPT_TAMIL)
    {
    search:
      while (new_pos > start &&
	     !(is_one_of (info[new_pos], (FLAG (OT_M) | FLAG (OT_H)))))
	new_pos--;

      /* If we found no Halant we are done.
       * Otherwise only proceed if the Halant does
       * not belong to the Matra itself! */
      if (is_halant (info[new_pos]) &&
	  info[new_pos].indic_position() != POS_PRE_M)
      {
#if 0 // See comment above
	/* -> If ZWJ or ZWNJ follow this halant, position is moved after it. */
	if (new_pos + 1 < end && is_joiner (info[new_pos + 1]))
	  new_pos++;
#endif
	if (new_pos + 1 < end)
	{
	  /* -> If ZWJ follows this halant, matra is NOT repositioned after this halant. */
	  if (info[new_pos + 1].indic_category() == OT_ZWJ)
	  {
	    /* Keep searching. */
	    if (new_pos > start)
	    {
	      new_pos--;
	      goto search;
	    }
	  }
	  /* -> If ZWNJ follows this halant, position is moved after it.
	   *
	   * IMPLEMENTATION NOTES:
	   *
	   * This is taken care of by the state-machine. A Halant,ZWNJ is a terminating
	   * sequence for a consonant syllable; any pre-base matras occurring after it
	   * will belong to the subsequent syllable.
	   */
	}
      }
      else
	new_pos = start; /* No move. */
    }

    if (start < new_pos && info[new_pos].indic_position () != POS_PRE_M)
    {
      /* Now go see if there's actually any matras... */
      for (unsigned int i = new_pos; i > start; i--)
	if (info[i - 1].indic_position () == POS_PRE_M)
	{
	  unsigned int old_pos = i - 1;
	  if (old_pos < base && base <= new_pos) /* Shouldn't actually happen. */
	    base--;

	  hb_glyph_info_t tmp = info[old_pos];
	  memmove (&info[old_pos], &info[old_pos + 1], (new_pos - old_pos) * sizeof (info[0]));
	  info[new_pos] = tmp;

	  /* Note: this merge_clusters() is intentionally *after* the reordering.
	   * Indic matra reordering is special and tricky... */
	  buffer->merge_clusters (new_pos, hb_min (end, base + 1));

	  new_pos--;
	}
    } else {
      for (unsigned int i = start; i < base; i++)
	if (info[i].indic_position () == POS_PRE_M) {
	  buffer->merge_clusters (i, hb_min (end, base + 1));
	  break;
	}
    }
  }


  /*   o Reorder reph:
   *
   *     Reph’s original position is always at the beginning of the syllable,
   *     (i.e. it is not reordered at the character reordering stage). However,
   *     it will be reordered according to the basic-forms shaping results.
   *     Possible positions for reph, depending on the script, are; after main,
   *     before post-base consonant forms, and after post-base consonant forms.
   */

  /* Two cases:
   *
   * - If repha is encoded as a sequence of characters (Ra,H or Ra,H,ZWJ), then
   *   we should only move it if the sequence ligated to the repha form.
   *
   * - If repha is encoded separately and in the logical position, we should only
   *   move it if it did NOT ligate.  If it ligated, it's probably the font trying
   *   to make it work without the reordering.
   */
  if (start + 1 < end &&
      info[start].indic_position() == POS_RA_TO_BECOME_REPH &&
      ((info[start].indic_category() == OT_Repha) ^
       _hb_glyph_info_ligated_and_didnt_multiply (&info[start])))
  {
    unsigned int new_reph_pos;
    reph_position_t reph_pos = indic_plan->config->reph_pos;

    /*       1. If reph should be positioned after post-base consonant forms,
     *          proceed to step 5.
     */
    if (reph_pos == REPH_POS_AFTER_POST)
    {
      goto reph_step_5;
    }

    /*       2. If the reph repositioning class is not after post-base: target
     *          position is after the first explicit halant glyph between the
     *          first post-reph consonant and last main consonant. If ZWJ or ZWNJ
     *          are following this halant, position is moved after it. If such
     *          position is found, this is the target position. Otherwise,
     *          proceed to the next step.
     *
     *          Note: in old-implementation fonts, where classifications were
     *          fixed in shaping engine, there was no case where reph position
     *          will be found on this step.
     */
    {
      new_reph_pos = start + 1;
      while (new_reph_pos < base && !is_halant (info[new_reph_pos]))
	new_reph_pos++;

      if (new_reph_pos < base && is_halant (info[new_reph_pos]))
      {
	/* ->If ZWJ or ZWNJ are following this halant, position is moved after it. */
	if (new_reph_pos + 1 < base && is_joiner (info[new_reph_pos + 1]))
	  new_reph_pos++;
	goto reph_move;
      }
    }

    /*       3. If reph should be repositioned after the main consonant: find the
     *          first consonant not ligated with main, or find the first
     *          consonant that is not a potential pre-base-reordering Ra.
     */
    if (reph_pos == REPH_POS_AFTER_MAIN)
    {
      new_reph_pos = base;
      while (new_reph_pos + 1 < end && info[new_reph_pos + 1].indic_position() <= POS_AFTER_MAIN)
	new_reph_pos++;
      if (new_reph_pos < end)
	goto reph_move;
    }

    /*       4. If reph should be positioned before post-base consonant, find
     *          first post-base classified consonant not ligated with main. If no
     *          consonant is found, the target position should be before the
     *          first matra, syllable modifier sign or vedic sign.
     */
    /* This is our take on what step 4 is trying to say (and failing, BADLY). */
    if (reph_pos == REPH_POS_AFTER_SUB)
    {
      new_reph_pos = base;
      while (new_reph_pos + 1 < end &&
	     !( FLAG_UNSAFE (info[new_reph_pos + 1].indic_position()) & (FLAG (POS_POST_C) | FLAG (POS_AFTER_POST) | FLAG (POS_SMVD))))
	new_reph_pos++;
      if (new_reph_pos < end)
	goto reph_move;
    }

    /*       5. If no consonant is found in steps 3 or 4, move reph to a position
     *          immediately before the first post-base matra, syllable modifier
     *          sign or vedic sign that has a reordering class after the intended
     *          reph position. For example, if the reordering position for reph
     *          is post-main, it will skip above-base matras that also have a
     *          post-main position.
     */
    reph_step_5:
    {
      /* Copied from step 2. */
      new_reph_pos = start + 1;
      while (new_reph_pos < base && !is_halant (info[new_reph_pos]))
	new_reph_pos++;

      if (new_reph_pos < base && is_halant (info[new_reph_pos]))
      {
	/* ->If ZWJ or ZWNJ are following this halant, position is moved after it. */
	if (new_reph_pos + 1 < base && is_joiner (info[new_reph_pos + 1]))
	  new_reph_pos++;
	goto reph_move;
      }
    }
    /* See https://github.com/harfbuzz/harfbuzz/issues/2298#issuecomment-615318654 */

    /*       6. Otherwise, reorder reph to the end of the syllable.
     */
    {
      new_reph_pos = end - 1;
      while (new_reph_pos > start && info[new_reph_pos].indic_position() == POS_SMVD)
	new_reph_pos--;

      /*
       * If the Reph is to be ending up after a Matra,Halant sequence,
       * position it before that Halant so it can interact with the Matra.
       * However, if it's a plain Consonant,Halant we shouldn't do that.
       * Uniscribe doesn't do this.
       * TEST: U+0930,U+094D,U+0915,U+094B,U+094D
       */
      if (!indic_plan->uniscribe_bug_compatible &&
	  unlikely (is_halant (info[new_reph_pos])))
      {
	for (unsigned int i = base + 1; i < new_reph_pos; i++)
	  if (info[i].indic_category() == OT_M) {
	    /* Ok, got it. */
	    new_reph_pos--;
	  }
      }

      goto reph_move;
    }

    reph_move:
    {
      /* Move */
      buffer->merge_clusters (start, new_reph_pos + 1);
      hb_glyph_info_t reph = info[start];
      memmove (&info[start], &info[start + 1], (new_reph_pos - start) * sizeof (info[0]));
      info[new_reph_pos] = reph;

      if (start < base && base <= new_reph_pos)
	base--;
    }
  }


  /*   o Reorder pre-base-reordering consonants:
   *
   *     If a pre-base-reordering consonant is found, reorder it according to
   *     the following rules:
   */

  if (try_pref && base + 1 < end) /* Otherwise there can't be any pre-base-reordering Ra. */
  {
    for (unsigned int i = base + 1; i < end; i++)
      if ((info[i].mask & indic_plan->mask_array[INDIC_PREF]) != 0)
      {
	/*       1. Only reorder a glyph produced by substitution during application
	 *          of the <pref> feature. (Note that a font may shape a Ra consonant with
	 *          the feature generally but block it in certain contexts.)
	 */
	/* Note: We just check that something got substituted.  We don't check that
	 * the <pref> feature actually did it...
	 *
	 * Reorder pref only if it ligated. */
	if (_hb_glyph_info_ligated_and_didnt_multiply (&info[i]))
	{
	  /*
	   *       2. Try to find a target position the same way as for pre-base matra.
	   *          If it is found, reorder pre-base consonant glyph.
	   *
	   *       3. If position is not found, reorder immediately before main
	   *          consonant.
	   */

	  unsigned int new_pos = base;
	  /* Malayalam / Tamil do not have "half" forms or explicit virama forms.
	   * The glyphs formed by 'half' are Chillus or ligated explicit viramas.
	   * We want to position matra after them.
	   */
	  if (buffer->props.script != HB_SCRIPT_MALAYALAM && buffer->props.script != HB_SCRIPT_TAMIL)
	  {
	    while (new_pos > start &&
		   !(is_one_of (info[new_pos - 1], FLAG(OT_M) | FLAG (OT_H))))
	      new_pos--;
	  }

	  if (new_pos > start && is_halant (info[new_pos - 1]))
	  {
	    /* -> If ZWJ or ZWNJ follow this halant, position is moved after it. */
	    if (new_pos < end && is_joiner (info[new_pos]))
	      new_pos++;
	  }

	  {
	    unsigned int old_pos = i;

	    buffer->merge_clusters (new_pos, old_pos + 1);
	    hb_glyph_info_t tmp = info[old_pos];
	    memmove (&info[new_pos + 1], &info[new_pos], (old_pos - new_pos) * sizeof (info[0]));
	    info[new_pos] = tmp;

	    if (new_pos <= base && base < old_pos)
	      base++;
	  }
	}

	break;
      }
  }


  /* Apply 'init' to the Left Matra if it's a word start. */
  if (info[start].indic_position () == POS_PRE_M)
  {
    if (!start ||
	!(FLAG_UNSAFE (_hb_glyph_info_get_general_category (&info[start - 1])) &
	 FLAG_RANGE (HB_UNICODE_GENERAL_CATEGORY_FORMAT, HB_UNICODE_GENERAL_CATEGORY_NON_SPACING_MARK)))
      info[start].mask |= indic_plan->mask_array[INDIC_INIT];
    else
      buffer->unsafe_to_break (start - 1, start + 1);
  }


  /*
   * Finish off the clusters and go home!
   */
  if (indic_plan->uniscribe_bug_compatible)
  {
    switch ((hb_tag_t) plan->props.script)
    {
      case HB_SCRIPT_TAMIL:
      case HB_SCRIPT_SINHALA:
	break;

      default:
	/* Uniscribe merges the entire syllable into a single cluster... Except for Tamil & Sinhala.
	 * This means, half forms are submerged into the main consonant's cluster.
	 * This is unnecessary, and makes cursor positioning harder, but that's what
	 * Uniscribe does. */
	buffer->merge_clusters (start, end);
	break;
    }
  }
}


static void
final_reordering_indic (const hb_ot_shape_plan_t *plan,
			hb_font_t *font HB_UNUSED,
			hb_buffer_t *buffer)
{
  unsigned int count = buffer->len;
  if (unlikely (!count)) return;

  if (buffer->message (font, "start reordering indic final")) {
    foreach_syllable (buffer, start, end)
      final_reordering_syllable_indic (plan, buffer, start, end);
    (void) buffer->message (font, "end reordering indic final");
  }

  HB_BUFFER_DEALLOCATE_VAR (buffer, indic_category);
  HB_BUFFER_DEALLOCATE_VAR (buffer, indic_position);
}


static void
preprocess_text_indic (const hb_ot_shape_plan_t *plan,
		       hb_buffer_t              *buffer,
		       hb_font_t                *font)
{
  _hb_preprocess_text_vowel_constraints (plan, buffer, font);
}

static bool
decompose_indic (const hb_ot_shape_normalize_context_t *c,
		 hb_codepoint_t  ab,
		 hb_codepoint_t *a,
		 hb_codepoint_t *b)
{
  switch (ab)
  {
    /* Don't decompose these. */
    case 0x0931u  : return false; /* DEVANAGARI LETTER RRA */
    // https://github.com/harfbuzz/harfbuzz/issues/779
    case 0x09DCu  : return false; /* BENGALI LETTER RRA */
    case 0x09DDu  : return false; /* BENGALI LETTER RHA */
    case 0x0B94u  : return false; /* TAMIL LETTER AU */


    /*
     * Decompose split matras that don't have Unicode decompositions.
     */

#if 0
    /* Gujarati */
    /* This one has no decomposition in Unicode, but needs no decomposition either. */
    /* case 0x0AC9u  : return false; */

    /* Oriya */
    case 0x0B57u  : *a = no decomp, -> RIGHT; return true;
#endif
  }

  if ((ab == 0x0DDAu || hb_in_range<hb_codepoint_t> (ab, 0x0DDCu, 0x0DDEu)))
  {
    /*
     * Sinhala split matras...  Let the fun begin.
     *
     * These four characters have Unicode decompositions.  However, Uniscribe
     * decomposes them "Khmer-style", that is, it uses the character itself to
     * get the second half.  The first half of all four decompositions is always
     * U+0DD9.
     *
     * Now, there are buggy fonts, namely, the widely used lklug.ttf, that are
     * broken with Uniscribe.  But we need to support them.  As such, we only
     * do the Uniscribe-style decomposition if the character is transformed into
     * its "sec.half" form by the 'pstf' feature.  Otherwise, we fall back to
     * Unicode decomposition.
     *
     * Note that we can't unconditionally use Unicode decomposition.  That would
     * break some other fonts, that are designed to work with Uniscribe, and
     * don't have positioning features for the Unicode-style decomposition.
     *
     * Argh...
     *
     * The Uniscribe behavior is now documented in the newly published Sinhala
     * spec in 2012:
     *
     *   https://docs.microsoft.com/en-us/typography/script-development/sinhala#shaping
     */


    const indic_shape_plan_t *indic_plan = (const indic_shape_plan_t *) c->plan->data;
    hb_codepoint_t glyph;
    if (indic_plan->uniscribe_bug_compatible ||
	(c->font->get_nominal_glyph (ab, &glyph) &&
	 indic_plan->pstf.would_substitute (&glyph, 1, c->font->face)))
    {
      /* Ok, safe to use Uniscribe-style decomposition. */
      *a = 0x0DD9u;
      *b = ab;
      return true;
    }
  }

  return (bool) c->unicode->decompose (ab, a, b);
}

static bool
compose_indic (const hb_ot_shape_normalize_context_t *c,
	       hb_codepoint_t  a,
	       hb_codepoint_t  b,
	       hb_codepoint_t *ab)
{
  /* Avoid recomposing split matras. */
  if (HB_UNICODE_GENERAL_CATEGORY_IS_MARK (c->unicode->general_category (a)))
    return false;

  /* Composition-exclusion exceptions that we want to recompose. */
  if (a == 0x09AFu && b == 0x09BCu) { *ab = 0x09DFu; return true; }

  return (bool) c->unicode->compose (a, b, ab);
}


const hb_ot_complex_shaper_t _hb_ot_complex_shaper_indic =
{
  collect_features_indic,
  override_features_indic,
  data_create_indic,
  data_destroy_indic,
  preprocess_text_indic,
  nullptr, /* postprocess_glyphs */
  HB_OT_SHAPE_NORMALIZATION_MODE_COMPOSED_DIACRITICS_NO_SHORT_CIRCUIT,
  decompose_indic,
  compose_indic,
  setup_masks_indic,
  HB_TAG_NONE, /* gpos_tag */
  nullptr, /* reorder_marks */
  HB_OT_SHAPE_ZERO_WIDTH_MARKS_NONE,
  false, /* fallback_position */
};


#endif
