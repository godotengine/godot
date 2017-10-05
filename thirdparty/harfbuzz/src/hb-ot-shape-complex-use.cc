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

#include "hb-ot-shape-complex-use-private.hh"
#include "hb-ot-shape-complex-arabic-private.hh"

/* buffer var allocations */
#define use_category() complex_var_u8_0()


/*
 * Universal Shaping Engine.
 * https://www.microsoft.com/typography/OpenTypeDev/USE/intro.htm
 */

static const hb_tag_t
basic_features[] =
{
  /*
   * Basic features.
   * These features are applied all at once, before reordering.
   */
  HB_TAG('r','k','r','f'),
  HB_TAG('a','b','v','f'),
  HB_TAG('b','l','w','f'),
  HB_TAG('h','a','l','f'),
  HB_TAG('p','s','t','f'),
  HB_TAG('v','a','t','u'),
  HB_TAG('c','j','c','t'),
};
static const hb_tag_t
arabic_features[] =
{
  HB_TAG('i','s','o','l'),
  HB_TAG('i','n','i','t'),
  HB_TAG('m','e','d','i'),
  HB_TAG('f','i','n','a'),
  /* The spec doesn't specify these but we apply anyway, since our Arabic shaper
   * does.  These are only used in Syriac spec. */
  HB_TAG('m','e','d','2'),
  HB_TAG('f','i','n','2'),
  HB_TAG('f','i','n','3'),
};
/* Same order as arabic_features.  Don't need Syriac stuff.*/
enum joining_form_t {
  ISOL,
  INIT,
  MEDI,
  FINA,
  _NONE
};
static const hb_tag_t
other_features[] =
{
  /*
   * Other features.
   * These features are applied all at once, after reordering.
   */
  HB_TAG('a','b','v','s'),
  HB_TAG('b','l','w','s'),
  HB_TAG('h','a','l','n'),
  HB_TAG('p','r','e','s'),
  HB_TAG('p','s','t','s'),
  /* Positioning features, though we don't care about the types. */
  HB_TAG('d','i','s','t'),
  HB_TAG('a','b','v','m'),
  HB_TAG('b','l','w','m'),
};

static void
setup_syllables (const hb_ot_shape_plan_t *plan,
		 hb_font_t *font,
		 hb_buffer_t *buffer);
static void
clear_substitution_flags (const hb_ot_shape_plan_t *plan,
			  hb_font_t *font,
			  hb_buffer_t *buffer);
static void
record_rphf (const hb_ot_shape_plan_t *plan,
	     hb_font_t *font,
	     hb_buffer_t *buffer);
static void
record_pref (const hb_ot_shape_plan_t *plan,
	     hb_font_t *font,
	     hb_buffer_t *buffer);
static void
reorder (const hb_ot_shape_plan_t *plan,
	 hb_font_t *font,
	 hb_buffer_t *buffer);

static void
collect_features_use (hb_ot_shape_planner_t *plan)
{
  hb_ot_map_builder_t *map = &plan->map;

  /* Do this before any lookups have been applied. */
  map->add_gsub_pause (setup_syllables);

  /* "Default glyph pre-processing group" */
  map->add_global_bool_feature (HB_TAG('l','o','c','l'));
  map->add_global_bool_feature (HB_TAG('c','c','m','p'));
  map->add_global_bool_feature (HB_TAG('n','u','k','t'));
  map->add_global_bool_feature (HB_TAG('a','k','h','n'));

  /* "Reordering group" */
  map->add_gsub_pause (clear_substitution_flags);
  map->add_feature (HB_TAG('r','p','h','f'), 1, F_MANUAL_ZWJ);
  map->add_gsub_pause (record_rphf);
  map->add_gsub_pause (clear_substitution_flags);
  map->add_feature (HB_TAG('p','r','e','f'), 1, F_GLOBAL | F_MANUAL_ZWJ);
  map->add_gsub_pause (record_pref);

  /* "Orthographic unit shaping group" */
  for (unsigned int i = 0; i < ARRAY_LENGTH (basic_features); i++)
    map->add_feature (basic_features[i], 1, F_GLOBAL | F_MANUAL_ZWJ);

  map->add_gsub_pause (reorder);

  /* "Topographical features" */
  for (unsigned int i = 0; i < ARRAY_LENGTH (arabic_features); i++)
    map->add_feature (arabic_features[i], 1, F_NONE);
  map->add_gsub_pause (nullptr);

  /* "Standard typographic presentation" and "Positional feature application" */
  for (unsigned int i = 0; i < ARRAY_LENGTH (other_features); i++)
    map->add_feature (other_features[i], 1, F_GLOBAL | F_MANUAL_ZWJ);
}

struct use_shape_plan_t
{
  ASSERT_POD ();

  hb_mask_t rphf_mask;

  arabic_shape_plan_t *arabic_plan;
};

static bool
has_arabic_joining (hb_script_t script)
{
  /* List of scripts that have data in arabic-table. */
  switch ((int) script)
  {
    /* Unicode-1.1 additions */
    case HB_SCRIPT_ARABIC:

    /* Unicode-3.0 additions */
    case HB_SCRIPT_MONGOLIAN:
    case HB_SCRIPT_SYRIAC:

    /* Unicode-5.0 additions */
    case HB_SCRIPT_NKO:
    case HB_SCRIPT_PHAGS_PA:

    /* Unicode-6.0 additions */
    case HB_SCRIPT_MANDAIC:

    /* Unicode-7.0 additions */
    case HB_SCRIPT_MANICHAEAN:
    case HB_SCRIPT_PSALTER_PAHLAVI:

    /* Unicode-9.0 additions */
    case HB_SCRIPT_ADLAM:

      return true;

    default:
      return false;
  }
}

static void *
data_create_use (const hb_ot_shape_plan_t *plan)
{
  use_shape_plan_t *use_plan = (use_shape_plan_t *) calloc (1, sizeof (use_shape_plan_t));
  if (unlikely (!use_plan))
    return nullptr;

  use_plan->rphf_mask = plan->map.get_1_mask (HB_TAG('r','p','h','f'));

  if (has_arabic_joining (plan->props.script))
  {
    use_plan->arabic_plan = (arabic_shape_plan_t *) data_create_arabic (plan);
    if (unlikely (!use_plan->arabic_plan))
    {
      free (use_plan);
      return nullptr;
    }
  }

  return use_plan;
}

static void
data_destroy_use (void *data)
{
  use_shape_plan_t *use_plan = (use_shape_plan_t *) data;

  if (use_plan->arabic_plan)
    data_destroy_arabic (use_plan->arabic_plan);

  free (data);
}

enum syllable_type_t {
  independent_cluster,
  virama_terminated_cluster,
  standard_cluster,
  number_joiner_terminated_cluster,
  numeral_cluster,
  symbol_cluster,
  broken_cluster,
  non_cluster,
};

#include "hb-ot-shape-complex-use-machine.hh"


static void
setup_masks_use (const hb_ot_shape_plan_t *plan,
		 hb_buffer_t              *buffer,
		 hb_font_t                *font HB_UNUSED)
{
  const use_shape_plan_t *use_plan = (const use_shape_plan_t *) plan->data;

  /* Do this before allocating use_category(). */
  if (use_plan->arabic_plan)
  {
    setup_masks_arabic_plan (use_plan->arabic_plan, buffer, plan->props.script);
  }

  HB_BUFFER_ALLOCATE_VAR (buffer, use_category);

  /* We cannot setup masks here.  We save information about characters
   * and setup masks later on in a pause-callback. */

  unsigned int count = buffer->len;
  hb_glyph_info_t *info = buffer->info;
  for (unsigned int i = 0; i < count; i++)
    info[i].use_category() = hb_use_get_categories (info[i].codepoint);
}

static void
setup_rphf_mask (const hb_ot_shape_plan_t *plan,
		 hb_buffer_t *buffer)
{
  const use_shape_plan_t *use_plan = (const use_shape_plan_t *) plan->data;

  hb_mask_t mask = use_plan->rphf_mask;
  if (!mask) return;

  hb_glyph_info_t *info = buffer->info;

  foreach_syllable (buffer, start, end)
  {
    unsigned int limit = info[start].use_category() == USE_R ? 1 : MIN (3u, end - start);
    for (unsigned int i = start; i < start + limit; i++)
      info[i].mask |= mask;
  }
}

static void
setup_topographical_masks (const hb_ot_shape_plan_t *plan,
			   hb_buffer_t *buffer)
{
  const use_shape_plan_t *use_plan = (const use_shape_plan_t *) plan->data;
  if (use_plan->arabic_plan)
    return;

  static_assert ((INIT < 4 && ISOL < 4 && MEDI < 4 && FINA < 4), "");
  hb_mask_t masks[4], all_masks = 0;
  for (unsigned int i = 0; i < 4; i++)
  {
    masks[i] = plan->map.get_1_mask (arabic_features[i]);
    if (masks[i] == plan->map.get_global_mask ())
      masks[i] = 0;
    all_masks |= masks[i];
  }
  if (!all_masks)
    return;
  hb_mask_t other_masks = ~all_masks;

  unsigned int last_start = 0;
  joining_form_t last_form = _NONE;
  hb_glyph_info_t *info = buffer->info;
  foreach_syllable (buffer, start, end)
  {
    syllable_type_t syllable_type = (syllable_type_t) (info[start].syllable() & 0x0F);
    switch (syllable_type)
    {
      case independent_cluster:
      case symbol_cluster:
      case non_cluster:
	/* These don't join.  Nothing to do. */
	last_form = _NONE;
	break;

      case virama_terminated_cluster:
      case standard_cluster:
      case number_joiner_terminated_cluster:
      case numeral_cluster:
      case broken_cluster:

	bool join = last_form == FINA || last_form == ISOL;

	if (join)
	{
	  /* Fixup previous syllable's form. */
	  last_form = last_form == FINA ? MEDI : INIT;
	  for (unsigned int i = last_start; i < start; i++)
	    info[i].mask = (info[i].mask & other_masks) | masks[last_form];
	}

	/* Form for this syllable. */
	last_form = join ? FINA : ISOL;
	for (unsigned int i = start; i < end; i++)
	  info[i].mask = (info[i].mask & other_masks) | masks[last_form];

	break;
    }

    last_start = start;
  }
}

static void
setup_syllables (const hb_ot_shape_plan_t *plan,
		 hb_font_t *font HB_UNUSED,
		 hb_buffer_t *buffer)
{
  find_syllables (buffer);
  foreach_syllable (buffer, start, end)
    buffer->unsafe_to_break (start, end);
  setup_rphf_mask (plan, buffer);
  setup_topographical_masks (plan, buffer);
}

static void
clear_substitution_flags (const hb_ot_shape_plan_t *plan,
			  hb_font_t *font HB_UNUSED,
			  hb_buffer_t *buffer)
{
  hb_glyph_info_t *info = buffer->info;
  unsigned int count = buffer->len;
  for (unsigned int i = 0; i < count; i++)
    _hb_glyph_info_clear_substituted (&info[i]);
}

static void
record_rphf (const hb_ot_shape_plan_t *plan,
	     hb_font_t *font,
	     hb_buffer_t *buffer)
{
  const use_shape_plan_t *use_plan = (const use_shape_plan_t *) plan->data;

  hb_mask_t mask = use_plan->rphf_mask;
  if (!mask) return;
  hb_glyph_info_t *info = buffer->info;

  foreach_syllable (buffer, start, end)
  {
    /* Mark a substituted repha as USE_R. */
    for (unsigned int i = start; i < end && (info[i].mask & mask); i++)
      if (_hb_glyph_info_substituted (&info[i]))
      {
	info[i].use_category() = USE_R;
	break;
      }
  }
}

static void
record_pref (const hb_ot_shape_plan_t *plan,
	     hb_font_t *font,
	     hb_buffer_t *buffer)
{
  hb_glyph_info_t *info = buffer->info;

  foreach_syllable (buffer, start, end)
  {
    /* Mark a substituted pref as VPre, as they behave the same way. */
    for (unsigned int i = start; i < end; i++)
      if (_hb_glyph_info_substituted (&info[i]))
      {
	info[i].use_category() = USE_VPre;
	break;
      }
  }
}

static inline bool
is_halant (const hb_glyph_info_t &info)
{
  return info.use_category() == USE_H && !_hb_glyph_info_ligated (&info);
}

static void
reorder_syllable (hb_buffer_t *buffer, unsigned int start, unsigned int end)
{
  syllable_type_t syllable_type = (syllable_type_t) (buffer->info[start].syllable() & 0x0F);
  /* Only a few syllable types need reordering. */
  if (unlikely (!(FLAG_UNSAFE (syllable_type) &
		  (FLAG (virama_terminated_cluster) |
		   FLAG (standard_cluster) |
		   FLAG (broken_cluster) |
		   0))))
    return;

  hb_glyph_info_t *info = buffer->info;

#define BASE_FLAGS (FLAG (USE_B) | FLAG (USE_GB))

  /* Move things forward. */
  if (info[start].use_category() == USE_R && end - start > 1)
  {
    /* Got a repha.  Reorder it to after first base, before first halant. */
    for (unsigned int i = start + 1; i < end; i++)
      if ((FLAG_UNSAFE (info[i].use_category()) & (BASE_FLAGS)) || is_halant (info[i]))
      {
	/* If we hit a halant, move before it; otherwise it's a base: move to it's
	 * place, and shift things in between backward. */

	if (is_halant (info[i]))
	  i--;

	buffer->merge_clusters (start, i + 1);
	hb_glyph_info_t t = info[start];
	memmove (&info[start], &info[start + 1], (i - start) * sizeof (info[0]));
	info[i] = t;

	break;
      }
  }

  /* Move things back. */
  unsigned int j = end;
  for (unsigned int i = start; i < end; i++)
  {
    uint32_t flag = FLAG_UNSAFE (info[i].use_category());
    if ((flag & (BASE_FLAGS)) || is_halant (info[i]))
    {
      /* If we hit a halant, move after it; otherwise it's a base: move to it's
       * place, and shift things in between backward. */
      if (is_halant (info[i]))
	j = i + 1;
      else
	j = i;
    }
    else if (((flag) & (FLAG (USE_VPre) | FLAG (USE_VMPre))) &&
	     /* Only move the first component of a MultipleSubst. */
	     0 == _hb_glyph_info_get_lig_comp (&info[i]) &&
	     j < i)
    {
      buffer->merge_clusters (j, i + 1);
      hb_glyph_info_t t = info[i];
      memmove (&info[j + 1], &info[j], (i - j) * sizeof (info[0]));
      info[j] = t;
    }
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

  hb_glyph_info_t dottedcircle = {0};
  if (!font->get_nominal_glyph (0x25CCu, &dottedcircle.codepoint))
    return;
  dottedcircle.use_category() = hb_use_get_categories (0x25CC);

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
	     buffer->cur().use_category() == USE_R)
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

  hb_glyph_info_t *info = buffer->info;

  foreach_syllable (buffer, start, end)
    reorder_syllable (buffer, start, end);

  /* Zero syllables now... */
  unsigned int count = buffer->len;
  for (unsigned int i = 0; i < count; i++)
    info[i].syllable() = 0;

  HB_BUFFER_DEALLOCATE_VAR (buffer, use_category);
}

static bool
decompose_use (const hb_ot_shape_normalize_context_t *c,
                hb_codepoint_t  ab,
                hb_codepoint_t *a,
                hb_codepoint_t *b)
{
  switch (ab)
  {
    /* Chakma:
     * Special case where the Unicode decomp gives matras in the wrong order
     * for cluster validation.
     */
    case 0x1112Eu : *a = 0x11127u; *b= 0x11131u; return true;
    case 0x1112Fu : *a = 0x11127u; *b= 0x11132u; return true;
  }

  return (bool) c->unicode->decompose (ab, a, b);
}

static bool
compose_use (const hb_ot_shape_normalize_context_t *c,
	     hb_codepoint_t  a,
	     hb_codepoint_t  b,
	     hb_codepoint_t *ab)
{
  /* Avoid recomposing split matras. */
  if (HB_UNICODE_GENERAL_CATEGORY_IS_MARK (c->unicode->general_category (a)))
    return false;

  return (bool)c->unicode->compose (a, b, ab);
}


const hb_ot_complex_shaper_t _hb_ot_complex_shaper_use =
{
  collect_features_use,
  nullptr, /* override_features */
  data_create_use,
  data_destroy_use,
  nullptr, /* preprocess_text */
  nullptr, /* postprocess_glyphs */
  HB_OT_SHAPE_NORMALIZATION_MODE_COMPOSED_DIACRITICS_NO_SHORT_CIRCUIT,
  decompose_use,
  compose_use,
  setup_masks_use,
  nullptr, /* disable_otl */
  nullptr, /* reorder_marks */
  HB_OT_SHAPE_ZERO_WIDTH_MARKS_BY_GDEF_EARLY,
  false, /* fallback_position */
};
