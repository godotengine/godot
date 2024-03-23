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

#ifndef HB_NO_AAT_SHAPE

#include "hb-aat-map.hh"

#include "hb-aat-layout.hh"
#include "hb-aat-layout-feat-table.hh"


void hb_aat_map_builder_t::add_feature (const hb_feature_t &feature)
{
  if (!face->table.feat->has_data ()) return;

  if (feature.tag == HB_TAG ('a','a','l','t'))
  {
    if (!face->table.feat->exposes_feature (HB_AAT_LAYOUT_FEATURE_TYPE_CHARACTER_ALTERNATIVES))
      return;
    feature_range_t *range = features.push();
    range->start = feature.start;
    range->end = feature.end;
    range->info.type = HB_AAT_LAYOUT_FEATURE_TYPE_CHARACTER_ALTERNATIVES;
    range->info.setting = (hb_aat_layout_feature_selector_t) feature.value;
    range->info.seq = features.length;
    range->info.is_exclusive = true;
    return;
  }

  const hb_aat_feature_mapping_t *mapping = hb_aat_layout_find_feature_mapping (feature.tag);
  if (!mapping) return;

  const AAT::FeatureName* feature_name = &face->table.feat->get_feature (mapping->aatFeatureType);
  if (!feature_name->has_data ())
  {
    /* Special case: Chain::compile_flags will fall back to the deprecated version of
     * small-caps if necessary, so we need to check for that possibility.
     * https://github.com/harfbuzz/harfbuzz/issues/2307 */
    if (mapping->aatFeatureType == HB_AAT_LAYOUT_FEATURE_TYPE_LOWER_CASE &&
	mapping->selectorToEnable == HB_AAT_LAYOUT_FEATURE_SELECTOR_LOWER_CASE_SMALL_CAPS)
    {
      feature_name = &face->table.feat->get_feature (HB_AAT_LAYOUT_FEATURE_TYPE_LETTER_CASE);
      if (!feature_name->has_data ()) return;
    }
    else return;
  }

  feature_range_t *range = features.push();
  range->start = feature.start;
  range->end = feature.end;
  range->info.type = mapping->aatFeatureType;
  range->info.setting = feature.value ? mapping->selectorToEnable : mapping->selectorToDisable;
  range->info.seq = features.length;
  range->info.is_exclusive = feature_name->is_exclusive ();
}

void
hb_aat_map_builder_t::compile (hb_aat_map_t  &m)
{
  /* Compute active features per range, and compile each. */

  /* Sort features by start/end events. */
  hb_vector_t<feature_event_t> feature_events;
  for (unsigned int i = 0; i < features.length; i++)
  {
    auto &feature = features[i];

    if (features[i].start == features[i].end)
      continue;

    feature_event_t *event;

    event = feature_events.push ();
    event->index = features[i].start;
    event->start = true;
    event->feature = feature.info;

    event = feature_events.push ();
    event->index = features[i].end;
    event->start = false;
    event->feature = feature.info;
  }
  feature_events.qsort ();
  /* Add a strategic final event. */
  {
    feature_info_t feature;
    feature.seq = features.length + 1;

    feature_event_t *event = feature_events.push ();
    event->index = -1; /* This value does magic. */
    event->start = false;
    event->feature = feature;
  }

  /* Scan events and save features for each range. */
  hb_sorted_vector_t<feature_info_t> active_features;
  unsigned int last_index = 0;
  for (unsigned int i = 0; i < feature_events.length; i++)
  {
    feature_event_t *event = &feature_events[i];

    if (event->index != last_index)
    {
      /* Save a snapshot of active features and the range. */

      /* Sort features and merge duplicates */
      current_features = active_features;
      range_first = last_index;
      range_last = event->index - 1;
      if (current_features.length)
      {
	current_features.qsort ();
	unsigned int j = 0;
	for (unsigned int i = 1; i < current_features.length; i++)
	  if (current_features[i].type != current_features[j].type ||
	      /* Nonexclusive feature selectors come in even/odd pairs to turn a setting on/off
	       * respectively, so we mask out the low-order bit when checking for "duplicates"
	       * (selectors referring to the same feature setting) here. */
	      (!current_features[i].is_exclusive && ((current_features[i].setting & ~1) != (current_features[j].setting & ~1))))
	    current_features[++j] = current_features[i];
	current_features.shrink (j + 1);
      }

      hb_aat_layout_compile_map (this, &m);

      last_index = event->index;
    }

    if (event->start)
    {
      active_features.push (event->feature);
    } else {
      feature_info_t *feature = active_features.lsearch (event->feature);
      if (feature)
	active_features.remove_ordered (feature - active_features.arrayZ);
    }
  }

  for (auto &chain_flags : m.chain_flags)
    // With our above setup this value is one less than desired; adjust it.
    chain_flags.tail().cluster_last = HB_FEATURE_GLOBAL_END;
}


#endif
