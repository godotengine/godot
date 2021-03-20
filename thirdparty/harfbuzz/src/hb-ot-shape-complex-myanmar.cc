/*
 * Copyright © 2011,2012,2013  Google, Inc.
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

#include "hb-ot-shape-complex-myanmar.hh"
#include "hb-ot-shape-complex-myanmar-machine.hh"


/*
 * Myanmar shaper.
 */

static const hb_tag_t
myanmar_basic_features[] =
{
  /*
   * Basic features.
   * These features are applied in order, one at a time, after reordering.
   */
  HB_TAG('r','p','h','f'),
  HB_TAG('p','r','e','f'),
  HB_TAG('b','l','w','f'),
  HB_TAG('p','s','t','f'),
};
static const hb_tag_t
myanmar_other_features[] =
{
  /*
   * Other features.
   * These features are applied all at once, after clearing syllables.
   */
  HB_TAG('p','r','e','s'),
  HB_TAG('a','b','v','s'),
  HB_TAG('b','l','w','s'),
  HB_TAG('p','s','t','s'),
};

static void
setup_syllables_myanmar (const hb_ot_shape_plan_t *plan,
			 hb_font_t *font,
			 hb_buffer_t *buffer);
static void
reorder_myanmar (const hb_ot_shape_plan_t *plan,
		 hb_font_t *font,
		 hb_buffer_t *buffer);

static void
collect_features_myanmar (hb_ot_shape_planner_t *plan)
{
  hb_ot_map_builder_t *map = &plan->map;

  /* Do this before any lookups have been applied. */
  map->add_gsub_pause (setup_syllables_myanmar);

  map->enable_feature (HB_TAG('l','o','c','l'));
  /* The Indic specs do not require ccmp, but we apply it here since if
   * there is a use of it, it's typically at the beginning. */
  map->enable_feature (HB_TAG('c','c','m','p'));


  map->add_gsub_pause (reorder_myanmar);

  for (unsigned int i = 0; i < ARRAY_LENGTH (myanmar_basic_features); i++)
  {
    map->enable_feature (myanmar_basic_features[i], F_MANUAL_ZWJ);
    map->add_gsub_pause (nullptr);
  }

  map->add_gsub_pause (_hb_clear_syllables);

  for (unsigned int i = 0; i < ARRAY_LENGTH (myanmar_other_features); i++)
    map->enable_feature (myanmar_other_features[i], F_MANUAL_ZWJ);
}

static void
setup_masks_myanmar (const hb_ot_shape_plan_t *plan HB_UNUSED,
		     hb_buffer_t              *buffer,
		     hb_font_t                *font HB_UNUSED)
{
  HB_BUFFER_ALLOCATE_VAR (buffer, myanmar_category);
  HB_BUFFER_ALLOCATE_VAR (buffer, myanmar_position);

  /* We cannot setup masks here.  We save information about characters
   * and setup masks later on in a pause-callback. */

  unsigned int count = buffer->len;
  hb_glyph_info_t *info = buffer->info;
  for (unsigned int i = 0; i < count; i++)
    set_myanmar_properties (info[i]);
}

static void
setup_syllables_myanmar (const hb_ot_shape_plan_t *plan HB_UNUSED,
			 hb_font_t *font HB_UNUSED,
			 hb_buffer_t *buffer)
{
  find_syllables_myanmar (buffer);
  foreach_syllable (buffer, start, end)
    buffer->unsafe_to_break (start, end);
}

static int
compare_myanmar_order (const hb_glyph_info_t *pa, const hb_glyph_info_t *pb)
{
  int a = pa->myanmar_position();
  int b = pb->myanmar_position();

  return a < b ? -1 : a == b ? 0 : +1;
}


/* Rules from:
 * https://docs.microsoft.com/en-us/typography/script-development/myanmar */

static void
initial_reordering_consonant_syllable (hb_buffer_t *buffer,
				       unsigned int start, unsigned int end)
{
  hb_glyph_info_t *info = buffer->info;

  unsigned int base = end;
  bool has_reph = false;

  {
    unsigned int limit = start;
    if (start + 3 <= end &&
	info[start  ].myanmar_category() == OT_Ra &&
	info[start+1].myanmar_category() == OT_As &&
	info[start+2].myanmar_category() == OT_H)
    {
      limit += 3;
      base = start;
      has_reph = true;
    }

    {
      if (!has_reph)
	base = limit;

      for (unsigned int i = limit; i < end; i++)
	if (is_consonant (info[i]))
	{
	  base = i;
	  break;
	}
    }
  }

  /* Reorder! */
  {
    unsigned int i = start;
    for (; i < start + (has_reph ? 3 : 0); i++)
      info[i].myanmar_position() = POS_AFTER_MAIN;
    for (; i < base; i++)
      info[i].myanmar_position() = POS_PRE_C;
    if (i < end)
    {
      info[i].myanmar_position() = POS_BASE_C;
      i++;
    }
    indic_position_t pos = POS_AFTER_MAIN;
    /* The following loop may be ugly, but it implements all of
     * Myanmar reordering! */
    for (; i < end; i++)
    {
      if (info[i].myanmar_category() == OT_MR) /* Pre-base reordering */
      {
	info[i].myanmar_position() = POS_PRE_C;
	continue;
      }
      if (info[i].myanmar_position() < POS_BASE_C) /* Left matra */
      {
	continue;
      }
      if (info[i].myanmar_category() == OT_VS)
      {
	info[i].myanmar_position() = info[i - 1].myanmar_position();
	continue;
      }

      if (pos == POS_AFTER_MAIN && info[i].myanmar_category() == OT_VBlw)
      {
	pos = POS_BELOW_C;
	info[i].myanmar_position() = pos;
	continue;
      }

      if (pos == POS_BELOW_C && info[i].myanmar_category() == OT_A)
      {
	info[i].myanmar_position() = POS_BEFORE_SUB;
	continue;
      }
      if (pos == POS_BELOW_C && info[i].myanmar_category() == OT_VBlw)
      {
	info[i].myanmar_position() = pos;
	continue;
      }
      if (pos == POS_BELOW_C && info[i].myanmar_category() != OT_A)
      {
	pos = POS_AFTER_SUB;
	info[i].myanmar_position() = pos;
	continue;
      }
      info[i].myanmar_position() = pos;
    }
  }

  /* Sit tight, rock 'n roll! */
  buffer->sort (start, end, compare_myanmar_order);
}

static void
reorder_syllable_myanmar (const hb_ot_shape_plan_t *plan HB_UNUSED,
			  hb_face_t *face HB_UNUSED,
			  hb_buffer_t *buffer,
			  unsigned int start, unsigned int end)
{
  myanmar_syllable_type_t syllable_type = (myanmar_syllable_type_t) (buffer->info[start].syllable() & 0x0F);
  switch (syllable_type) {

    case myanmar_broken_cluster: /* We already inserted dotted-circles, so just call the consonant_syllable. */
    case myanmar_consonant_syllable:
      initial_reordering_consonant_syllable  (buffer, start, end);
      break;

    case myanmar_punctuation_cluster:
    case myanmar_non_myanmar_cluster:
      break;
  }
}

static void
reorder_myanmar (const hb_ot_shape_plan_t *plan,
		 hb_font_t *font,
		 hb_buffer_t *buffer)
{
  if (buffer->message (font, "start reordering myanmar"))
  {
    hb_syllabic_insert_dotted_circles (font, buffer,
				       myanmar_broken_cluster,
				       OT_GB);

    foreach_syllable (buffer, start, end)
      reorder_syllable_myanmar (plan, font->face, buffer, start, end);
    (void) buffer->message (font, "end reordering myanmar");
  }

  HB_BUFFER_DEALLOCATE_VAR (buffer, myanmar_category);
  HB_BUFFER_DEALLOCATE_VAR (buffer, myanmar_position);
}


const hb_ot_complex_shaper_t _hb_ot_complex_shaper_myanmar =
{
  collect_features_myanmar,
  nullptr, /* override_features */
  nullptr, /* data_create */
  nullptr, /* data_destroy */
  nullptr, /* preprocess_text */
  nullptr, /* postprocess_glyphs */
  HB_OT_SHAPE_NORMALIZATION_MODE_COMPOSED_DIACRITICS_NO_SHORT_CIRCUIT,
  nullptr, /* decompose */
  nullptr, /* compose */
  setup_masks_myanmar,
  HB_TAG_NONE, /* gpos_tag */
  nullptr, /* reorder_marks */
  HB_OT_SHAPE_ZERO_WIDTH_MARKS_BY_GDEF_EARLY,
  false, /* fallback_position */
};


/* Ugly Zawgyi encoding.
 * Disable all auto processing.
 * https://github.com/harfbuzz/harfbuzz/issues/1162 */
const hb_ot_complex_shaper_t _hb_ot_complex_shaper_myanmar_zawgyi =
{
  nullptr, /* collect_features */
  nullptr, /* override_features */
  nullptr, /* data_create */
  nullptr, /* data_destroy */
  nullptr, /* preprocess_text */
  nullptr, /* postprocess_glyphs */
  HB_OT_SHAPE_NORMALIZATION_MODE_NONE,
  nullptr, /* decompose */
  nullptr, /* compose */
  nullptr, /* setup_masks */
  HB_TAG_NONE, /* gpos_tag */
  nullptr, /* reorder_marks */
  HB_OT_SHAPE_ZERO_WIDTH_MARKS_NONE,
  false, /* fallback_position */
};


#endif
