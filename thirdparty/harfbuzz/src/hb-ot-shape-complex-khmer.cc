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

#include "hb-ot-shape-complex-khmer-private.hh"
#include "hb-ot-layout-private.hh"


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
   * These features are applied in order, one at a time, after reordering.
   */
  {HB_TAG('p','r','e','f'), F_NONE},
  {HB_TAG('b','l','w','f'), F_NONE},
  {HB_TAG('a','b','v','f'), F_NONE},
  {HB_TAG('p','s','t','f'), F_NONE},
  {HB_TAG('c','f','a','r'), F_NONE},
  /*
   * Other features.
   * These features are applied all at once.
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
reorder (const hb_ot_shape_plan_t *plan,
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
  map->add_gsub_pause (reorder);

  /* Testing suggests that Uniscribe does NOT pause between basic
   * features.  Test with KhmerUI.ttf and the following three
   * sequences:
   *
   *   U+1789,U+17BC
   *   U+1789,U+17D2,U+1789
   *   U+1789,U+17D2,U+1789,U+17BC
   *
   * https://github.com/harfbuzz/harfbuzz/issues/974
   */
  map->add_global_bool_feature (HB_TAG('l','o','c','l'));
  map->add_global_bool_feature (HB_TAG('c','c','m','p'));

  unsigned int i = 0;
  for (; i < KHMER_BASIC_FEATURES; i++) {
    map->add_feature (khmer_features[i].tag, 1, khmer_features[i].flags | F_MANUAL_ZWJ | F_MANUAL_ZWNJ);
  }

  map->add_gsub_pause (clear_syllables);

  for (; i < KHMER_NUM_FEATURES; i++) {
    map->add_feature (khmer_features[i].tag, 1, khmer_features[i].flags | F_MANUAL_ZWJ | F_MANUAL_ZWNJ);
  }

  map->add_global_bool_feature (HB_TAG('c','a','l','t'));
  map->add_global_bool_feature (HB_TAG('c','l','i','g'));

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


/* Rules from:
 * https://docs.microsoft.com/en-us/typography/script-development/devanagari */

static void
reorder_consonant_syllable (const hb_ot_shape_plan_t *plan,
			    hb_face_t *face,
			    hb_buffer_t *buffer,
			    unsigned int start, unsigned int end)
{
  const khmer_shape_plan_t *khmer_plan = (const khmer_shape_plan_t *) plan->data;
  hb_glyph_info_t *info = buffer->info;

  /* Setup masks. */
  {
    /* Post-base */
    hb_mask_t mask = khmer_plan->mask_array[BLWF] | khmer_plan->mask_array[ABVF] | khmer_plan->mask_array[PSTF];
    for (unsigned int i = start + 1; i < end; i++)
      info[i].mask  |= mask;
  }

  unsigned int num_coengs = 0;
  for (unsigned int i = start + 1; i < end; i++)
  {
    /* """
     * When a COENG + (Cons | IndV) combination are found (and subscript count
     * is less than two) the character combination is handled according to the
     * subscript type of the character following the COENG.
     *
     * ...
     *
     * Subscript Type 2 - The COENG + RO characters are reordered to immediately
     * before the base glyph. Then the COENG + RO characters are assigned to have
     * the 'pref' OpenType feature applied to them.
     * """
     */
    if (info[i].khmer_category() == OT_Coeng && num_coengs <= 2 && i + 1 < end)
    {
      num_coengs++;

      if (info[i + 1].khmer_category() == OT_Ra)
      {
	for (unsigned int j = 0; j < 2; j++)
	  info[i + j].mask |= khmer_plan->mask_array[PREF];

	/* Move the Coeng,Ro sequence to the start. */
	buffer->merge_clusters (start, i + 2);
	hb_glyph_info_t t0 = info[i];
	hb_glyph_info_t t1 = info[i + 1];
	memmove (&info[start + 2], &info[start], (i - start) * sizeof (info[0]));
	info[start] = t0;
	info[start + 1] = t1;

	/* Mark the subsequent stuff with 'cfar'.  Used in Khmer.
	 * Read the feature spec.
	 * This allows distinguishing the following cases with MS Khmer fonts:
	 * U+1784,U+17D2,U+179A,U+17D2,U+1782
	 * U+1784,U+17D2,U+1782,U+17D2,U+179A
	 */
	if (khmer_plan->mask_array[CFAR])
	  for (unsigned int j = i + 2; j < end; j++)
	    info[j].mask |= khmer_plan->mask_array[CFAR];

	num_coengs = 2; /* Done. */
      }
    }

    /* Reorder left matra piece. */
    else if (info[i].khmer_position() == POS_PRE_M)
    {
      /* Move to the start. */
      buffer->merge_clusters (start, i + 1);
      hb_glyph_info_t t = info[i];
      memmove (&info[start + 1], &info[start], (i - start) * sizeof (info[0]));
      info[start] = t;
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
     reorder_consonant_syllable (plan, face, buffer, start, end);
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
  while (buffer->idx < buffer->len && buffer->successful)
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
      while (buffer->idx < buffer->len && buffer->successful &&
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
reorder (const hb_ot_shape_plan_t *plan,
	 hb_font_t *font,
	 hb_buffer_t *buffer)
{
  insert_dotted_circles (plan, font, buffer);

  foreach_syllable (buffer, start, end)
    initial_reordering_syllable (plan, font->face, buffer, start, end);

  HB_BUFFER_DEALLOCATE_VAR (buffer, khmer_category);
  HB_BUFFER_DEALLOCATE_VAR (buffer, khmer_position);
}

static void
clear_syllables (const hb_ot_shape_plan_t *plan HB_UNUSED,
		 hb_font_t *font HB_UNUSED,
		 hb_buffer_t *buffer)
{
  /* TODO: In USE, we clear syllables right after reorder.  Figure out
   * what Uniscribe does. */
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
