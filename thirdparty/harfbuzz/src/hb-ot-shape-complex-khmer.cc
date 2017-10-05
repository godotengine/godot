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

#include "hb-ot-shape-complex-indic-private.hh"
#include "hb-ot-layout-private.hh"

/* buffer var allocations */
#define khmer_category() complex_var_u8_0() /* khmer_category_t */
#define khmer_position() complex_var_u8_1() /* khmer_position_t */


/*
 * Khmer shaper.
 */

typedef indic_category_t khmer_category_t;
typedef indic_position_t khmer_position_t;


static inline khmer_position_t
matra_position (khmer_position_t side)
{
  switch ((int) side)
  {
    case POS_PRE_C:
      return POS_PRE_M;

    case POS_POST_C:
    case POS_ABOVE_C:
    case POS_BELOW_C:
      return POS_AFTER_POST;

    default:
      return side;
  };
}

static inline bool
is_one_of (const hb_glyph_info_t &info, unsigned int flags)
{
  /* If it ligated, all bets are off. */
  if (_hb_glyph_info_ligated (&info)) return false;
  return !!(FLAG_UNSAFE (info.khmer_category()) & flags);
}

static inline bool
is_joiner (const hb_glyph_info_t &info)
{
  return is_one_of (info, JOINER_FLAGS);
}

static inline bool
is_consonant (const hb_glyph_info_t &info)
{
  return is_one_of (info, CONSONANT_FLAGS | FLAG (OT_V));
}

static inline bool
is_coeng (const hb_glyph_info_t &info)
{
  return is_one_of (info, FLAG (OT_Coeng));
}

static inline void
set_khmer_properties (hb_glyph_info_t &info)
{
  hb_codepoint_t u = info.codepoint;
  unsigned int type = hb_indic_get_categories (u);
  khmer_category_t cat = (khmer_category_t) (type & 0x7Fu);
  khmer_position_t pos = (khmer_position_t) (type >> 8);


  /*
   * Re-assign category
   */

  if (unlikely (u == 0x17C6u)) cat = OT_N; /* Khmer Bindu doesn't like to be repositioned. */
  else if (unlikely (hb_in_range<hb_codepoint_t> (u, 0x17CDu, 0x17D1u) ||
		     u == 0x17CBu || u == 0x17D3u || u == 0x17DDu)) /* Khmer Various signs */
  {
    /* These can occur mid-syllable (eg. before matras), even though Unicode marks them as Syllable_Modifier.
     * https://github.com/roozbehp/unicode-data/issues/5 */
    cat = OT_M;
    pos = POS_ABOVE_C;
  }
  else if (unlikely (hb_in_range<hb_codepoint_t> (u, 0x2010u, 0x2011u))) cat = OT_PLACEHOLDER;
  else if (unlikely (u == 0x25CCu)) cat = OT_DOTTEDCIRCLE;


  /*
   * Re-assign position.
   */

  if ((FLAG_UNSAFE (cat) & CONSONANT_FLAGS))
  {
    pos = POS_BASE_C;
    if (u == 0x179Au)
      cat = OT_Ra;
  }
  else if (cat == OT_M)
  {
    pos = matra_position (pos);
  }
  else if ((FLAG_UNSAFE (cat) & (FLAG (OT_SM) | FLAG (OT_A) | FLAG (OT_Symbol))))
  {
    pos = POS_SMVD;
  }

  info.khmer_category() = cat;
  info.khmer_position() = pos;
}

/*
 * Things above this line should ideally be moved to the Indic table itself.
 */


/*
 * Khmer shaper.
 */

struct feature_list_t {
  hb_tag_t tag;
  hb_ot_map_feature_flags_t flags;
};

static const feature_list_t
khmer_features[] =
{
  /*
   * Basic features.
   * These features are applied in order, one at a time, after initial_reordering.
   */
  {HB_TAG('p','r','e','f'), F_NONE},
  {HB_TAG('b','l','w','f'), F_NONE},
  {HB_TAG('a','b','v','f'), F_NONE},
  {HB_TAG('p','s','t','f'), F_NONE},
  {HB_TAG('c','f','a','r'), F_NONE},
  /*
   * Other features.
   * These features are applied all at once, after final_reordering.
   * Default Bengali font in Windows for example has intermixed
   * lookups for init,pres,abvs,blws features.
   */
  {HB_TAG('p','r','e','s'), F_GLOBAL},
  {HB_TAG('a','b','v','s'), F_GLOBAL},
  {HB_TAG('b','l','w','s'), F_GLOBAL},
  {HB_TAG('p','s','t','s'), F_GLOBAL},
  /* Positioning features, though we don't care about the types. */
  {HB_TAG('d','i','s','t'), F_GLOBAL},
  {HB_TAG('a','b','v','m'), F_GLOBAL},
  {HB_TAG('b','l','w','m'), F_GLOBAL},
};

/*
 * Must be in the same order as the khmer_features array.
 */
enum {
  PREF,
  BLWF,
  ABVF,
  PSTF,
  CFAR,

  _PRES,
  _ABVS,
  _BLWS,
  _PSTS,
  _DIST,
  _ABVM,
  _BLWM,

  KHMER_NUM_FEATURES,
  KHMER_BASIC_FEATURES = _PRES /* Don't forget to update this! */
};

static void
setup_syllables (const hb_ot_shape_plan_t *plan,
		 hb_font_t *font,
		 hb_buffer_t *buffer);
static void
initial_reordering (const hb_ot_shape_plan_t *plan,
		    hb_font_t *font,
		    hb_buffer_t *buffer);
static void
final_reordering (const hb_ot_shape_plan_t *plan,
		  hb_font_t *font,
		  hb_buffer_t *buffer);
static void
clear_syllables (const hb_ot_shape_plan_t *plan,
		 hb_font_t *font,
		 hb_buffer_t *buffer);

static void
collect_features_khmer (hb_ot_shape_planner_t *plan)
{
  hb_ot_map_builder_t *map = &plan->map;

  /* Do this before any lookups have been applied. */
  map->add_gsub_pause (setup_syllables);

  map->add_global_bool_feature (HB_TAG('l','o','c','l'));
  /* The Indic specs do not require ccmp, but we apply it here since if
   * there is a use of it, it's typically at the beginning. */
  map->add_global_bool_feature (HB_TAG('c','c','m','p'));


  unsigned int i = 0;
  map->add_gsub_pause (initial_reordering);
  for (; i < KHMER_BASIC_FEATURES; i++) {
    map->add_feature (khmer_features[i].tag, 1, khmer_features[i].flags | F_MANUAL_ZWJ | F_MANUAL_ZWNJ);
    map->add_gsub_pause (nullptr);
  }
  map->add_gsub_pause (final_reordering);
  for (; i < KHMER_NUM_FEATURES; i++) {
    map->add_feature (khmer_features[i].tag, 1, khmer_features[i].flags | F_MANUAL_ZWJ | F_MANUAL_ZWNJ);
  }

  map->add_global_bool_feature (HB_TAG('c','a','l','t'));
  map->add_global_bool_feature (HB_TAG('c','l','i','g'));

  map->add_gsub_pause (clear_syllables);
}

static void
override_features_khmer (hb_ot_shape_planner_t *plan)
{
  /* Uniscribe does not apply 'kern' in Khmer. */
  if (hb_options ().uniscribe_bug_compatible)
  {
    plan->map.add_feature (HB_TAG('k','e','r','n'), 0, F_GLOBAL);
  }

  plan->map.add_feature (HB_TAG('l','i','g','a'), 0, F_GLOBAL);
}


struct would_substitute_feature_t
{
  inline void init (const hb_ot_map_t *map, hb_tag_t feature_tag, bool zero_context_)
  {
    zero_context = zero_context_;
    map->get_stage_lookups (0/*GSUB*/,
			    map->get_feature_stage (0/*GSUB*/, feature_tag),
			    &lookups, &count);
  }

  inline bool would_substitute (const hb_codepoint_t *glyphs,
				unsigned int          glyphs_count,
				hb_face_t            *face) const
  {
    for (unsigned int i = 0; i < count; i++)
      if (hb_ot_layout_lookup_would_substitute_fast (face, lookups[i].index, glyphs, glyphs_count, zero_context))
	return true;
    return false;
  }

  private:
  const hb_ot_map_t::lookup_map_t *lookups;
  unsigned int count;
  bool zero_context;
};

struct khmer_shape_plan_t
{
  ASSERT_POD ();

  inline bool get_virama_glyph (hb_font_t *font, hb_codepoint_t *pglyph) const
  {
    hb_codepoint_t glyph = virama_glyph;
    if (unlikely (virama_glyph == (hb_codepoint_t) -1))
    {
      if (!font->get_nominal_glyph (0x17D2u, &glyph))
	glyph = 0;
      /* Technically speaking, the spec says we should apply 'locl' to virama too.
       * Maybe one day... */

      /* Our get_nominal_glyph() function needs a font, so we can't get the virama glyph
       * during shape planning...  Instead, overwrite it here.  It's safe.  Don't worry! */
      virama_glyph = glyph;
    }

    *pglyph = glyph;
    return glyph != 0;
  }

  mutable hb_codepoint_t virama_glyph;

  would_substitute_feature_t pref;

  hb_mask_t mask_array[KHMER_NUM_FEATURES];
};

static void *
data_create_khmer (const hb_ot_shape_plan_t *plan)
{
  khmer_shape_plan_t *khmer_plan = (khmer_shape_plan_t *) calloc (1, sizeof (khmer_shape_plan_t));
  if (unlikely (!khmer_plan))
    return nullptr;

  khmer_plan->virama_glyph = (hb_codepoint_t) -1;

  khmer_plan->pref.init (&plan->map, HB_TAG('p','r','e','f'), true);

  for (unsigned int i = 0; i < ARRAY_LENGTH (khmer_plan->mask_array); i++)
    khmer_plan->mask_array[i] = (khmer_features[i].flags & F_GLOBAL) ?
				 0 : plan->map.get_1_mask (khmer_features[i].tag);

  return khmer_plan;
}

static void
data_destroy_khmer (void *data)
{
  free (data);
}


enum syllable_type_t {
  consonant_syllable,
  broken_cluster,
  non_khmer_cluster,
};

#include "hb-ot-shape-complex-khmer-machine.hh"

static void
setup_masks_khmer (const hb_ot_shape_plan_t *plan HB_UNUSED,
		   hb_buffer_t              *buffer,
		   hb_font_t                *font HB_UNUSED)
{
  HB_BUFFER_ALLOCATE_VAR (buffer, khmer_category);
  HB_BUFFER_ALLOCATE_VAR (buffer, khmer_position);

  /* We cannot setup masks here.  We save information about characters
   * and setup masks later on in a pause-callback. */

  unsigned int count = buffer->len;
  hb_glyph_info_t *info = buffer->info;
  for (unsigned int i = 0; i < count; i++)
    set_khmer_properties (info[i]);
}

static void
setup_syllables (const hb_ot_shape_plan_t *plan HB_UNUSED,
		 hb_font_t *font HB_UNUSED,
		 hb_buffer_t *buffer)
{
  find_syllables (buffer);
  foreach_syllable (buffer, start, end)
    buffer->unsafe_to_break (start, end);
}

static int
compare_khmer_order (const hb_glyph_info_t *pa, const hb_glyph_info_t *pb)
{
  int a = pa->khmer_position();
  int b = pb->khmer_position();

  return a < b ? -1 : a == b ? 0 : +1;
}


/* Rules from:
 * https://www.microsoft.com/typography/otfntdev/devanot/shaping.aspx */

static void
initial_reordering_consonant_syllable (const hb_ot_shape_plan_t *plan,
				       hb_face_t *face,
				       hb_buffer_t *buffer,
				       unsigned int start, unsigned int end)
{
  const khmer_shape_plan_t *khmer_plan = (const khmer_shape_plan_t *) plan->data;
  hb_glyph_info_t *info = buffer->info;

  /* 1. Khmer shaping assumes that a syllable will begin with a Cons, IndV, or Number. */

  /* The first consonant is always the base. */
  unsigned int base = start;
  info[base].khmer_position() = POS_BASE_C;

  /* Mark all subsequent consonants as below. */
  for (unsigned int i = base + 1; i < end; i++)
    if (is_consonant (info[i]))
      info[i].khmer_position() = POS_BELOW_C;

  /* Mark final consonants.  A final consonant is one appearing after a matra,
   * like in Khmer. */
  for (unsigned int i = base + 1; i < end; i++)
    if (info[i].khmer_category() == OT_M) {
      for (unsigned int j = i + 1; j < end; j++)
        if (is_consonant (info[j])) {
	  info[j].khmer_position() = POS_FINAL_C;
	  break;
	}
      break;
    }

  /* Attach misc marks to previous char to move with them. */
  {
    khmer_position_t last_pos = POS_START;
    for (unsigned int i = start; i < end; i++)
    {
      if ((FLAG_UNSAFE (info[i].khmer_category()) & (JOINER_FLAGS | FLAG (OT_N) | FLAG (OT_RS) | MEDIAL_FLAGS | FLAG (OT_Coeng))))
      {
	info[i].khmer_position() = last_pos;
	if (unlikely (info[i].khmer_category() == OT_H &&
		      info[i].khmer_position() == POS_PRE_M))
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
	    if (info[j - 1].khmer_position() != POS_PRE_M) {
	      info[i].khmer_position() = info[j - 1].khmer_position();
	      break;
	    }
	}
      } else if (info[i].khmer_position() != POS_SMVD) {
        last_pos = (khmer_position_t) info[i].khmer_position();
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
	  if (info[j].khmer_position() < POS_SMVD)
	    info[j].khmer_position() = info[i].khmer_position();
	last = i;
      } else if (info[i].khmer_category() == OT_M)
        last = i;
  }

  {
    /* Use syllable() for sort accounting temporarily. */
    unsigned int syllable = info[start].syllable();
    for (unsigned int i = start; i < end; i++)
      info[i].syllable() = i - start;

    /* Sit tight, rock 'n roll! */
    hb_stable_sort (info + start, end - start, compare_khmer_order);
    /* Find base again */
    base = end;
    for (unsigned int i = start; i < end; i++)
      if (info[i].khmer_position() == POS_BASE_C)
      {
	base = i;
	break;
      }

    /* Note!  syllable() is a one-byte field. */
    for (unsigned int i = base; i < end; i++)
      if (info[i].syllable() != 255)
      {
	unsigned int max = i;
	unsigned int j = start + info[i].syllable();
	while (j != i)
	{
	  max = MAX (max, j);
	  unsigned int next = start + info[j].syllable();
	  info[j].syllable() = 255; /* So we don't process j later again. */
	  j = next;
	}
	if (i != max)
	  buffer->merge_clusters (i, max + 1);
      }

    /* Put syllable back in. */
    for (unsigned int i = start; i < end; i++)
      info[i].syllable() = syllable;
  }

  /* Setup masks now */

  {
    hb_mask_t mask;

    /* Post-base */
    mask = khmer_plan->mask_array[BLWF] | khmer_plan->mask_array[ABVF] | khmer_plan->mask_array[PSTF];
    for (unsigned int i = base + 1; i < end; i++)
      info[i].mask  |= mask;
  }

  unsigned int pref_len = 2;
  if (khmer_plan->mask_array[PREF] && base + pref_len < end)
  {
    /* Find a Halant,Ra sequence and mark it for pre-base-reordering processing. */
    for (unsigned int i = base + 1; i + pref_len - 1 < end; i++) {
      hb_codepoint_t glyphs[2];
      for (unsigned int j = 0; j < pref_len; j++)
        glyphs[j] = info[i + j].codepoint;
      if (khmer_plan->pref.would_substitute (glyphs, pref_len, face))
      {
	for (unsigned int j = 0; j < pref_len; j++)
	  info[i++].mask |= khmer_plan->mask_array[PREF];

	/* Mark the subsequent stuff with 'cfar'.  Used in Khmer.
	 * Read the feature spec.
	 * This allows distinguishing the following cases with MS Khmer fonts:
	 * U+1784,U+17D2,U+179A,U+17D2,U+1782
	 * U+1784,U+17D2,U+1782,U+17D2,U+179A
	 */
	if (khmer_plan->mask_array[CFAR])
	  for (; i < end; i++)
	    info[i].mask |= khmer_plan->mask_array[CFAR];

	break;
      }
    }
  }
}

static void
initial_reordering_syllable (const hb_ot_shape_plan_t *plan,
			     hb_face_t *face,
			     hb_buffer_t *buffer,
			     unsigned int start, unsigned int end)
{
  syllable_type_t syllable_type = (syllable_type_t) (buffer->info[start].syllable() & 0x0F);
  switch (syllable_type)
  {
    case broken_cluster: /* We already inserted dotted-circles, so just call the consonant_syllable. */
    case consonant_syllable:
     initial_reordering_consonant_syllable (plan, face, buffer, start, end);
     break;

    case non_khmer_cluster:
      break;
  }
}

static inline void
insert_dotted_circles (const hb_ot_shape_plan_t *plan HB_UNUSED,
		       hb_font_t *font,
		       hb_buffer_t *buffer)
{
  /* Note: This loop is extra overhead, but should not be measurable. */
  bool has_broken_syllables = false;
  unsigned int count = buffer->len;
  hb_glyph_info_t *info = buffer->info;
  for (unsigned int i = 0; i < count; i++)
    if ((info[i].syllable() & 0x0F) == broken_cluster)
    {
      has_broken_syllables = true;
      break;
    }
  if (likely (!has_broken_syllables))
    return;


  hb_codepoint_t dottedcircle_glyph;
  if (!font->get_nominal_glyph (0x25CCu, &dottedcircle_glyph))
    return;

  hb_glyph_info_t dottedcircle = {0};
  dottedcircle.codepoint = 0x25CCu;
  set_khmer_properties (dottedcircle);
  dottedcircle.codepoint = dottedcircle_glyph;

  buffer->clear_output ();

  buffer->idx = 0;
  unsigned int last_syllable = 0;
  while (buffer->idx < buffer->len && !buffer->in_error)
  {
    unsigned int syllable = buffer->cur().syllable();
    syllable_type_t syllable_type = (syllable_type_t) (syllable & 0x0F);
    if (unlikely (last_syllable != syllable && syllable_type == broken_cluster))
    {
      last_syllable = syllable;

      hb_glyph_info_t ginfo = dottedcircle;
      ginfo.cluster = buffer->cur().cluster;
      ginfo.mask = buffer->cur().mask;
      ginfo.syllable() = buffer->cur().syllable();
      /* TODO Set glyph_props? */

      /* Insert dottedcircle after possible Repha. */
      while (buffer->idx < buffer->len && !buffer->in_error &&
	     last_syllable == buffer->cur().syllable() &&
	     buffer->cur().khmer_category() == OT_Repha)
        buffer->next_glyph ();

      buffer->output_info (ginfo);
    }
    else
      buffer->next_glyph ();
  }

  buffer->swap_buffers ();
}

static void
initial_reordering (const hb_ot_shape_plan_t *plan,
		    hb_font_t *font,
		    hb_buffer_t *buffer)
{
  insert_dotted_circles (plan, font, buffer);

  foreach_syllable (buffer, start, end)
    initial_reordering_syllable (plan, font->face, buffer, start, end);
}

static void
final_reordering_syllable (const hb_ot_shape_plan_t *plan,
			   hb_buffer_t *buffer,
			   unsigned int start, unsigned int end)
{
  const khmer_shape_plan_t *khmer_plan = (const khmer_shape_plan_t *) plan->data;
  hb_glyph_info_t *info = buffer->info;


  /* This function relies heavily on halant glyphs.  Lots of ligation
   * and possibly multiple substitutions happened prior to this
   * phase, and that might have messed up our properties.  Recover
   * from a particular case of that where we're fairly sure that a
   * class of OT_H is desired but has been lost. */
  if (khmer_plan->virama_glyph)
  {
    unsigned int virama_glyph = khmer_plan->virama_glyph;
    for (unsigned int i = start; i < end; i++)
      if (info[i].codepoint == virama_glyph &&
	  _hb_glyph_info_ligated (&info[i]) &&
	  _hb_glyph_info_multiplied (&info[i]))
      {
        /* This will make sure that this glyph passes is_coeng() test. */
	info[i].khmer_category() = OT_H;
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

  bool try_pref = !!khmer_plan->mask_array[PREF];

  /* Find base again */
  unsigned int base;
  for (base = start; base < end; base++)
    if (info[base].khmer_position() >= POS_BASE_C)
    {
      if (try_pref && base + 1 < end)
      {
	for (unsigned int i = base + 1; i < end; i++)
	  if ((info[i].mask & khmer_plan->mask_array[PREF]) != 0)
	  {
	    if (!(_hb_glyph_info_substituted (&info[i]) &&
		  _hb_glyph_info_ligated_and_didnt_multiply (&info[i])))
	    {
	      /* Ok, this was a 'pref' candidate but didn't form any.
	       * Base is around here... */
	      base = i;
	      while (base < end && is_coeng (info[base]))
		base++;
	      info[base].khmer_position() = POS_BASE_C;

	      try_pref = false;
	    }
	    break;
	  }
      }

      if (start < base && info[base].khmer_position() > POS_BASE_C)
        base--;
      break;
    }
  if (base == end && start < base &&
      is_one_of (info[base - 1], FLAG (OT_ZWJ)))
    base--;
  if (base < end)
    while (start < base &&
	   is_one_of (info[base], (FLAG (OT_N) | FLAG (OT_Coeng))))
      base--;


  /*   o Reorder matras:
   *
   *     If a pre-base matra character had been reordered before applying basic
   *     features, the glyph can be moved closer to the main consonant based on
   *     whether half-forms had been formed. Actual position for the matra is
   *     defined as “after last standalone halant glyph, after initial matra
   *     position and before the main consonant”. If ZWJ or ZWNJ follow this
   *     halant, position is moved after it.
   */

  if (start + 1 < end && start < base) /* Otherwise there can't be any pre-base matra characters. */
  {
    /* If we lost track of base, alas, position before last thingy. */
    unsigned int new_pos = base == end ? base - 2 : base - 1;

    while (new_pos > start &&
	   !(is_one_of (info[new_pos], (FLAG (OT_M) | FLAG (OT_Coeng)))))
      new_pos--;

    /* If we found no Halant we are done.
     * Otherwise only proceed if the Halant does
     * not belong to the Matra itself! */
    if (is_coeng (info[new_pos]) &&
	info[new_pos].khmer_position() != POS_PRE_M)
    {
      /* -> If ZWJ or ZWNJ follow this halant, position is moved after it. */
      if (new_pos + 1 < end && is_joiner (info[new_pos + 1]))
	new_pos++;
    }
    else
      new_pos = start; /* No move. */

    if (start < new_pos && info[new_pos].khmer_position () != POS_PRE_M)
    {
      /* Now go see if there's actually any matras... */
      for (unsigned int i = new_pos; i > start; i--)
	if (info[i - 1].khmer_position () == POS_PRE_M)
	{
	  unsigned int old_pos = i - 1;
	  if (old_pos < base && base <= new_pos) /* Shouldn't actually happen. */
	    base--;

	  hb_glyph_info_t tmp = info[old_pos];
	  memmove (&info[old_pos], &info[old_pos + 1], (new_pos - old_pos) * sizeof (info[0]));
	  info[new_pos] = tmp;

	  /* Note: this merge_clusters() is intentionally *after* the reordering.
	   * Indic matra reordering is special and tricky... */
	  buffer->merge_clusters (new_pos, MIN (end, base + 1));

	  new_pos--;
	}
    } else {
      for (unsigned int i = start; i < base; i++)
	if (info[i].khmer_position () == POS_PRE_M) {
	  buffer->merge_clusters (i, MIN (end, base + 1));
	  break;
	}
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
      if ((info[i].mask & khmer_plan->mask_array[PREF]) != 0)
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
	  while (new_pos > start &&
		 !(is_one_of (info[new_pos - 1], FLAG(OT_M) | FLAG (OT_Coeng))))
	    new_pos--;

	  /* In Khmer coeng model, a H,Ra can go *after* matras.  If it goes after a
	   * split matra, it should be reordered to *before* the left part of such matra. */
	  if (new_pos > start && info[new_pos - 1].khmer_category() == OT_M)
	  {
	    unsigned int old_pos = i;
	    for (unsigned int j = base + 1; j < old_pos; j++)
	      if (info[j].khmer_category() == OT_M)
	      {
		new_pos--;
		break;
	      }
	  }

	  if (new_pos > start && is_coeng (info[new_pos - 1]))
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


  /*
   * Finish off the clusters and go home!
   */
  if (hb_options ().uniscribe_bug_compatible)
  {
    /* Uniscribe merges the entire syllable into a single cluster... Except for Tamil & Sinhala.
     * This means, half forms are submerged into the main consonant's cluster.
     * This is unnecessary, and makes cursor positioning harder, but that's what
     * Uniscribe does. */
    buffer->merge_clusters (start, end);
  }
}


static void
final_reordering (const hb_ot_shape_plan_t *plan,
		  hb_font_t *font HB_UNUSED,
		  hb_buffer_t *buffer)
{
  unsigned int count = buffer->len;
  if (unlikely (!count)) return;

  foreach_syllable (buffer, start, end)
    final_reordering_syllable (plan, buffer, start, end);

  HB_BUFFER_DEALLOCATE_VAR (buffer, khmer_category);
  HB_BUFFER_DEALLOCATE_VAR (buffer, khmer_position);
}


static void
clear_syllables (const hb_ot_shape_plan_t *plan HB_UNUSED,
		 hb_font_t *font HB_UNUSED,
		 hb_buffer_t *buffer)
{
  hb_glyph_info_t *info = buffer->info;
  unsigned int count = buffer->len;
  for (unsigned int i = 0; i < count; i++)
    info[i].syllable() = 0;
}


static bool
decompose_khmer (const hb_ot_shape_normalize_context_t *c,
		 hb_codepoint_t  ab,
		 hb_codepoint_t *a,
		 hb_codepoint_t *b)
{
  switch (ab)
  {
    /*
     * Decompose split matras that don't have Unicode decompositions.
     */

    /* Khmer */
    case 0x17BEu  : *a = 0x17C1u; *b= 0x17BEu; return true;
    case 0x17BFu  : *a = 0x17C1u; *b= 0x17BFu; return true;
    case 0x17C0u  : *a = 0x17C1u; *b= 0x17C0u; return true;
    case 0x17C4u  : *a = 0x17C1u; *b= 0x17C4u; return true;
    case 0x17C5u  : *a = 0x17C1u; *b= 0x17C5u; return true;
  }

  return (bool) c->unicode->decompose (ab, a, b);
}

static bool
compose_khmer (const hb_ot_shape_normalize_context_t *c,
	       hb_codepoint_t  a,
	       hb_codepoint_t  b,
	       hb_codepoint_t *ab)
{
  /* Avoid recomposing split matras. */
  if (HB_UNICODE_GENERAL_CATEGORY_IS_MARK (c->unicode->general_category (a)))
    return false;

  return (bool) c->unicode->compose (a, b, ab);
}


const hb_ot_complex_shaper_t _hb_ot_complex_shaper_khmer =
{
  collect_features_khmer,
  override_features_khmer,
  data_create_khmer,
  data_destroy_khmer,
  nullptr, /* preprocess_text */
  nullptr, /* postprocess_glyphs */
  HB_OT_SHAPE_NORMALIZATION_MODE_COMPOSED_DIACRITICS_NO_SHORT_CIRCUIT,
  decompose_khmer,
  compose_khmer,
  setup_masks_khmer,
  nullptr, /* disable_otl */
  nullptr, /* reorder_marks */
  HB_OT_SHAPE_ZERO_WIDTH_MARKS_NONE,
  false, /* fallback_position */
};
