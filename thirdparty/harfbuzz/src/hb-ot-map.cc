/*
 * Copyright © 2009,2010  Red Hat, Inc.
 * Copyright © 2010,2011,2013  Google, Inc.
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
 * Red Hat Author(s): Behdad Esfahbod
 * Google Author(s): Behdad Esfahbod
 */

#include "hb.hh"

#ifndef HB_NO_OT_SHAPE

#include "hb-ot-map.hh"
#include "hb-ot-shape.hh"
#include "hb-ot-layout.hh"


void hb_ot_map_t::collect_lookups (unsigned int table_index, hb_set_t *lookups_out) const
{
  for (unsigned int i = 0; i < lookups[table_index].length; i++)
    lookups_out->add (lookups[table_index][i].index);
}


hb_ot_map_builder_t::hb_ot_map_builder_t (hb_face_t *face_,
					  const hb_segment_properties_t &props_)
{
  hb_memset (this, 0, sizeof (*this));

  feature_infos.init ();
  for (unsigned int table_index = 0; table_index < 2; table_index++)
    stages[table_index].init ();

  face = face_;
  props = props_;

  /* Fetch script/language indices for GSUB/GPOS.  We need these later to skip
   * features not available in either table and not waste precious bits for them. */

  unsigned int script_count = HB_OT_MAX_TAGS_PER_SCRIPT;
  unsigned int language_count = HB_OT_MAX_TAGS_PER_LANGUAGE;
  hb_tag_t script_tags[HB_OT_MAX_TAGS_PER_SCRIPT];
  hb_tag_t language_tags[HB_OT_MAX_TAGS_PER_LANGUAGE];

  hb_ot_tags_from_script_and_language (props.script,
				       props.language,
				       &script_count,
				       script_tags,
				       &language_count,
				       language_tags);

  for (unsigned int table_index = 0; table_index < 2; table_index++)
  {
    hb_tag_t table_tag = table_tags[table_index];
    found_script[table_index] = (bool) hb_ot_layout_table_select_script (face,
									 table_tag,
									 script_count,
									 script_tags,
									 &script_index[table_index],
									 &chosen_script[table_index]);
    hb_ot_layout_script_select_language (face,
					 table_tag,
					 script_index[table_index],
					 language_count,
					 language_tags,
					 &language_index[table_index]);
  }
}

hb_ot_map_builder_t::~hb_ot_map_builder_t ()
{
  feature_infos.fini ();
  for (unsigned int table_index = 0; table_index < 2; table_index++)
    stages[table_index].fini ();
}

void hb_ot_map_builder_t::add_feature (hb_tag_t tag,
				       hb_ot_map_feature_flags_t flags,
				       unsigned int value)
{
  if (unlikely (!tag)) return;
  feature_info_t *info = feature_infos.push();
  info->tag = tag;
  info->seq = feature_infos.length;
  info->max_value = value;
  info->flags = flags;
  info->default_value = (flags & F_GLOBAL) ? value : 0;
  info->stage[0] = current_stage[0];
  info->stage[1] = current_stage[1];
}

bool hb_ot_map_builder_t::has_feature (hb_tag_t tag)
{
  for (unsigned int table_index = 0; table_index < 2; table_index++)
  {
    if (hb_ot_layout_language_find_feature (face,
					    table_tags[table_index],
					    script_index[table_index],
					    language_index[table_index],
					    tag,
					    nullptr))
      return true;
  }
  return false;
}

void
hb_ot_map_builder_t::add_lookups (hb_ot_map_t  &m,
				  unsigned int  table_index,
				  unsigned int  feature_index,
				  unsigned int  variations_index,
				  hb_mask_t     mask,
				  bool          auto_zwnj,
				  bool          auto_zwj,
				  bool          random,
				  bool          per_syllable,
				  hb_tag_t      feature_tag)
{
  unsigned int lookup_indices[32];
  unsigned int offset, len;
  unsigned int table_lookup_count;

  table_lookup_count = hb_ot_layout_table_get_lookup_count (face, table_tags[table_index]);

  offset = 0;
  do {
    len = ARRAY_LENGTH (lookup_indices);
    hb_ot_layout_feature_with_variations_get_lookups (face,
						      table_tags[table_index],
						      feature_index,
						      variations_index,
						      offset, &len,
						      lookup_indices);

    for (unsigned int i = 0; i < len; i++)
    {
      if (lookup_indices[i] >= table_lookup_count)
	continue;
      hb_ot_map_t::lookup_map_t *lookup = m.lookups[table_index].push ();
      lookup->mask = mask;
      lookup->index = lookup_indices[i];
      lookup->auto_zwnj = auto_zwnj;
      lookup->auto_zwj = auto_zwj;
      lookup->random = random;
      lookup->per_syllable = per_syllable;
      lookup->feature_tag = feature_tag;
    }

    offset += len;
  } while (len == ARRAY_LENGTH (lookup_indices));
}


void hb_ot_map_builder_t::add_pause (unsigned int table_index, hb_ot_map_t::pause_func_t pause_func)
{
  stage_info_t *s = stages[table_index].push ();
  s->index = current_stage[table_index];
  s->pause_func = pause_func;

  current_stage[table_index]++;
}

void
hb_ot_map_builder_t::compile (hb_ot_map_t                  &m,
			      const hb_ot_shape_plan_key_t &key)
{
  unsigned int global_bit_shift = 8 * sizeof (hb_mask_t) - 1;
  unsigned int global_bit_mask = 1u << global_bit_shift;

  m.global_mask = global_bit_mask;

  unsigned int required_feature_index[2];
  hb_tag_t required_feature_tag[2];
  /* We default to applying required feature in stage 0.  If the required
   * feature has a tag that is known to the shaper, we apply required feature
   * in the stage for that tag.
   */
  unsigned int required_feature_stage[2] = {0, 0};

  for (unsigned int table_index = 0; table_index < 2; table_index++)
  {
    m.chosen_script[table_index] = chosen_script[table_index];
    m.found_script[table_index] = found_script[table_index];

    hb_ot_layout_language_get_required_feature (face,
						table_tags[table_index],
						script_index[table_index],
						language_index[table_index],
						&required_feature_index[table_index],
						&required_feature_tag[table_index]);
  }

  /* Sort features and merge duplicates */
  if (feature_infos.length)
  {
    if (!is_simple)
      feature_infos.qsort ();
    auto *f = feature_infos.arrayZ;
    unsigned int j = 0;
    unsigned count = feature_infos.length;
    for (unsigned int i = 1; i < count; i++)
      if (f[i].tag != f[j].tag)
	f[++j] = f[i];
      else {
	if (f[i].flags & F_GLOBAL) {
	  f[j].flags |= F_GLOBAL;
	  f[j].max_value = f[i].max_value;
	  f[j].default_value = f[i].default_value;
	} else {
	  if (f[j].flags & F_GLOBAL)
	    f[j].flags ^= F_GLOBAL;
	  f[j].max_value = hb_max (f[j].max_value, f[i].max_value);
	  /* Inherit default_value from j */
	}
	f[j].flags |= (f[i].flags & F_HAS_FALLBACK);
	f[j].stage[0] = hb_min (f[j].stage[0], f[i].stage[0]);
	f[j].stage[1] = hb_min (f[j].stage[1], f[i].stage[1]);
      }
    feature_infos.shrink (j + 1);
  }

  hb_map_t feature_indices[2];
  for (unsigned int table_index = 0; table_index < 2; table_index++)
    hb_ot_layout_collect_features_map (face,
				       table_tags[table_index],
				       script_index[table_index],
				       language_index[table_index],
				       &feature_indices[table_index]);

  /* Allocate bits now */
  static_assert ((!(HB_GLYPH_FLAG_DEFINED & (HB_GLYPH_FLAG_DEFINED + 1))), "");
  unsigned int next_bit = hb_popcount (HB_GLYPH_FLAG_DEFINED) + 1;

  unsigned count = feature_infos.length;
  for (unsigned int i = 0; i < count; i++)
  {
    const feature_info_t *info = &feature_infos[i];

    unsigned int bits_needed;

    if ((info->flags & F_GLOBAL) && info->max_value == 1)
      /* Uses the global bit */
      bits_needed = 0;
    else
      /* Limit bits per feature. */
      bits_needed = hb_min (HB_OT_MAP_MAX_BITS, hb_bit_storage (info->max_value));

    if (!info->max_value || next_bit + bits_needed >= global_bit_shift)
      continue; /* Feature disabled, or not enough bits. */

    bool found = false;
    unsigned int feature_index[2];
    for (unsigned int table_index = 0; table_index < 2; table_index++)
    {
      if (required_feature_tag[table_index] == info->tag)
	required_feature_stage[table_index] = info->stage[table_index];

      hb_codepoint_t *index;
      if (feature_indices[table_index].has (info->tag, &index))
      {
        feature_index[table_index] = *index;
        found = true;
      }
      else
        feature_index[table_index] = HB_OT_LAYOUT_NO_FEATURE_INDEX;
    }
    if (!found && (info->flags & F_GLOBAL_SEARCH))
    {
      for (unsigned int table_index = 0; table_index < 2; table_index++)
      {
	found |= (bool) hb_ot_layout_table_find_feature (face,
							 table_tags[table_index],
							 info->tag,
							 &feature_index[table_index]);
      }
    }
    if (!found && !(info->flags & F_HAS_FALLBACK))
      continue;


    hb_ot_map_t::feature_map_t *map = m.features.push ();

    map->tag = info->tag;
    map->index[0] = feature_index[0];
    map->index[1] = feature_index[1];
    map->stage[0] = info->stage[0];
    map->stage[1] = info->stage[1];
    map->auto_zwnj = !(info->flags & F_MANUAL_ZWNJ);
    map->auto_zwj = !(info->flags & F_MANUAL_ZWJ);
    map->random = !!(info->flags & F_RANDOM);
    map->per_syllable = !!(info->flags & F_PER_SYLLABLE);
    if ((info->flags & F_GLOBAL) && info->max_value == 1) {
      /* Uses the global bit */
      map->shift = global_bit_shift;
      map->mask = global_bit_mask;
    } else {
      map->shift = next_bit;
      map->mask = (1u << (next_bit + bits_needed)) - (1u << next_bit);
      next_bit += bits_needed;
      m.global_mask |= (info->default_value << map->shift) & map->mask;
    }
    map->_1_mask = (1u << map->shift) & map->mask;
    map->needs_fallback = !found;
  }
  //feature_infos.shrink (0); /* Done with these */
  if (is_simple)
    m.features.qsort ();

  add_gsub_pause (nullptr);
  add_gpos_pause (nullptr);

  for (unsigned int table_index = 0; table_index < 2; table_index++)
  {
    /* Collect lookup indices for features */
    auto &lookups = m.lookups[table_index];

    unsigned int stage_index = 0;
    unsigned int last_num_lookups = 0;
    for (unsigned stage = 0; stage < current_stage[table_index]; stage++)
    {
      if (required_feature_index[table_index] != HB_OT_LAYOUT_NO_FEATURE_INDEX &&
	  required_feature_stage[table_index] == stage)
	add_lookups (m, table_index,
		     required_feature_index[table_index],
		     key.variations_index[table_index],
		     global_bit_mask);

      for (auto &feature : m.features)
      {
	if (feature.stage[table_index] == stage)
	  add_lookups (m, table_index,
		       feature.index[table_index],
		       key.variations_index[table_index],
		       feature.mask,
		       feature.auto_zwnj,
		       feature.auto_zwj,
		       feature.random,
		       feature.per_syllable,
		       feature.tag);
      }

      /* Sort lookups and merge duplicates */
      if (last_num_lookups + 1 < lookups.length)
      {
	lookups.as_array ().sub_array (last_num_lookups, lookups.length - last_num_lookups).qsort ();

	unsigned int j = last_num_lookups;
	for (unsigned int i = j + 1; i < lookups.length; i++)
	  if (lookups.arrayZ[i].index != lookups.arrayZ[j].index)
	    lookups.arrayZ[++j] = lookups.arrayZ[i];
	  else
	  {
	    lookups.arrayZ[j].mask |= lookups.arrayZ[i].mask;
	    lookups.arrayZ[j].auto_zwnj &= lookups.arrayZ[i].auto_zwnj;
	    lookups.arrayZ[j].auto_zwj &= lookups.arrayZ[i].auto_zwj;
	  }
	lookups.shrink (j + 1);
      }

      last_num_lookups = lookups.length;

      if (stage_index < stages[table_index].length && stages[table_index][stage_index].index == stage) {
	hb_ot_map_t::stage_map_t *stage_map = m.stages[table_index].push ();
	stage_map->last_lookup = last_num_lookups;
	stage_map->pause_func = stages[table_index][stage_index].pause_func;

	stage_index++;
      }
    }
  }
}

unsigned int hb_ot_map_t::get_feature_tags (unsigned int start_offset, unsigned int *tag_count, hb_tag_t *tags) const
{
  if (tag_count)
  {
    auto sub_features = features.as_array ().sub_array (start_offset, tag_count);
    if (tags)
    {
      for (unsigned int i = 0; i < sub_features.length; i++)
        tags[i] = sub_features[i].tag;
    }
  }

  return features.length;
}

#endif
