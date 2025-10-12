/*
 * Copyright © 2023  Google, Inc.
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
 * Google Author(s): Garret Rieger, Qunxin Liu, Roderick Sheeter
 */

#include "hb-subset-plan.hh"

#include "hb-ot-layout-gdef-table.hh"
#include "hb-ot-layout-gpos-table.hh"
#include "hb-ot-layout-gsub-table.hh"

using OT::Layout::GSUB;
using OT::Layout::GPOS;

#ifndef HB_NO_SUBSET_LAYOUT

void
remap_used_mark_sets (hb_subset_plan_t *plan,
                      hb_map_t& used_mark_sets_map)
{
  hb_blob_ptr_t<OT::GDEF> gdef = plan->source_table<OT::GDEF> ();

  if (!gdef->has_data () || !gdef->has_mark_glyph_sets ())
  {
    gdef.destroy ();
    return;
  }

  hb_set_t used_mark_sets;
  gdef->get_mark_glyph_sets ().collect_used_mark_sets (plan->_glyphset_gsub, used_mark_sets);
  gdef.destroy ();

  remap_indexes (&used_mark_sets, &used_mark_sets_map);
}

/*
 * Removes all tags from 'tags' that are not in filter. Additionally eliminates any duplicates.
 * Returns true if anything was removed (not including duplicates).
 */
static bool _filter_tag_list(hb_vector_t<hb_tag_t>* tags, /* IN/OUT */
                             const hb_set_t* filter)
{
  hb_vector_t<hb_tag_t> out;
  out.alloc (tags->get_size() + 1); // +1 is to allocate room for the null terminator.

  bool removed = false;
  hb_set_t visited;

  for (hb_tag_t tag : *tags)
  {
    if (!tag) continue;
    if (visited.has (tag)) continue;

    if (!filter->has (tag))
    {
      removed = true;
      continue;
    }

    visited.add (tag);
    out.push (tag);
  }

  // The collect function needs a null element to signal end of the array.
  out.push (HB_TAG_NONE);

  hb_swap (out, *tags);
  return removed;
}

template <typename T>
static void _collect_layout_indices (hb_subset_plan_t     *plan,
                                     const T&              table,
                                     hb_set_t		  *lookup_indices, /* OUT */
                                     hb_set_t		  *feature_indices, /* OUT */
                                     hb_hashmap_t<unsigned, hb::shared_ptr<hb_set_t>> *feature_record_cond_idx_map, /* OUT */
                                     hb_hashmap_t<unsigned, const OT::Feature*> *feature_substitutes_map, /* OUT */
                                     hb_set_t& catch_all_record_feature_idxes, /* OUT */
                                     hb_hashmap_t<unsigned, hb_pair_t<const void*, const void*>>& catch_all_record_idx_feature_map /* OUT */)
{
  unsigned num_features = table.get_feature_count ();
  hb_vector_t<hb_tag_t> features;
  if (!plan->check_success (features.resize (num_features))) return;
  table.get_feature_tags (0, &num_features, features.arrayZ);
  bool retain_all_features = !_filter_tag_list (&features, &plan->layout_features);

  unsigned num_scripts = table.get_script_count ();
  hb_vector_t<hb_tag_t> scripts;
  if (!plan->check_success (scripts.resize (num_scripts))) return;
  table.get_script_tags (0, &num_scripts, scripts.arrayZ);
  bool retain_all_scripts = !_filter_tag_list (&scripts, &plan->layout_scripts);

  if (!plan->check_success (!features.in_error ()) || !features
      || !plan->check_success (!scripts.in_error ()) || !scripts)
    return;

  hb_ot_layout_collect_features (plan->source,
                                 T::tableTag,
                                 retain_all_scripts ? nullptr : scripts.arrayZ,
                                 nullptr,
                                 retain_all_features ? nullptr : features.arrayZ,
                                 feature_indices);

#ifndef HB_NO_VAR
  // collect feature substitutes with variations
  if (!plan->user_axes_location.is_empty ())
  {
    hb_hashmap_t<hb::shared_ptr<hb_map_t>, unsigned> conditionset_map;
    OT::hb_collect_feature_substitutes_with_var_context_t c =
    {
      &plan->axes_old_index_tag_map,
      &plan->axes_location,
      feature_record_cond_idx_map,
      feature_substitutes_map,
      catch_all_record_feature_idxes,
      feature_indices,
      false,
      false,
      false,
      0,
      &conditionset_map
    };
    table.collect_feature_substitutes_with_variations (&c);
  }
#endif

  for (unsigned feature_index : *feature_indices)
  {
    const OT::Feature* f = &(table.get_feature (feature_index));
    const OT::Feature **p = nullptr;
    if (feature_substitutes_map->has (feature_index, &p))
      f = *p;

    f->add_lookup_indexes_to (lookup_indices);
  }

#ifndef HB_NO_VAR
  if (catch_all_record_feature_idxes)
  {
    for (unsigned feature_index : catch_all_record_feature_idxes)
    {
      const OT::Feature& f = table.get_feature (feature_index);
      f.add_lookup_indexes_to (lookup_indices);
      const void *tag = reinterpret_cast<const void*> (&(table.get_feature_list ().get_tag (feature_index)));
      catch_all_record_idx_feature_map.set (feature_index, hb_pair (&f, tag));
    }
  }

  // If all axes are pinned then all feature variations will be dropped so there's no need
  // to collect lookups from them.
  if (!plan->all_axes_pinned)
    table.feature_variation_collect_lookups (feature_indices,
                                             plan->user_axes_location.is_empty () ? nullptr: feature_record_cond_idx_map,
                                             lookup_indices);
#endif
}


static inline void
_GSUBGPOS_find_duplicate_features (const OT::GSUBGPOS &g,
				   const hb_map_t *lookup_indices,
				   const hb_set_t *feature_indices,
				   const hb_hashmap_t<unsigned, const OT::Feature*> *feature_substitutes_map,
				   hb_map_t *duplicate_feature_map /* OUT */)
{
  if (feature_indices->is_empty ()) return;
  hb_hashmap_t<hb_tag_t, hb::unique_ptr<hb_set_t>> unique_features;
  //find out duplicate features after subset
  for (unsigned i : feature_indices->iter ())
  {
    hb_tag_t t = g.get_feature_tag (i);
    if (t == HB_MAP_VALUE_INVALID) continue;
    if (!unique_features.has (t))
    {
      if (unlikely (!unique_features.set (t, hb::unique_ptr<hb_set_t> {hb_set_create ()})))
	return;
      if (unique_features.has (t))
	unique_features.get (t)->add (i);
      duplicate_feature_map->set (i, i);
      continue;
    }

    bool found = false;

    hb_set_t* same_tag_features = unique_features.get (t);
    for (unsigned other_f_index : same_tag_features->iter ())
    {
      const OT::Feature* f = &(g.get_feature (i));
      const OT::Feature **p = nullptr;
      if (feature_substitutes_map->has (i, &p))
        f = *p;

      const OT::Feature* other_f = &(g.get_feature (other_f_index));
      if (feature_substitutes_map->has (other_f_index, &p))
        other_f = *p;

      auto f_iter =
      + hb_iter (f->lookupIndex)
      | hb_filter (lookup_indices)
      ;

      auto other_f_iter =
      + hb_iter (other_f->lookupIndex)
      | hb_filter (lookup_indices)
      ;

      bool is_equal = true;
      for (; f_iter && other_f_iter; f_iter++, other_f_iter++)
      {
	unsigned a = *f_iter;
	unsigned b = *other_f_iter;
	if (a != b) { is_equal = false; break; }
      }

      if (is_equal == false || f_iter || other_f_iter) continue;

      found = true;
      duplicate_feature_map->set (i, other_f_index);
      break;
    }

    if (found == false)
    {
      same_tag_features->add (i);
      duplicate_feature_map->set (i, i);
    }
  }
}

static void
remap_feature_indices (const hb_set_t &feature_indices,
                       const hb_map_t &duplicate_feature_map,
                       const hb_hashmap_t<unsigned, hb_pair_t<const void*, const void*>>& catch_all_record_idx_feature_map,
                       hb_map_t       *mapping, /* OUT */
                       hb_map_t       *mapping_w_duplicates /* OUT */)
{
  unsigned i = 0;
  for (const auto _ : feature_indices)
  {
    // retain those features in case we need to insert a catch-all record to reinstate the old features
    if (catch_all_record_idx_feature_map.has (_))
    {
      mapping->set (_, i);
      mapping_w_duplicates->set (_, i);
      i++;
    }
    else
    {
      uint32_t f_idx = duplicate_feature_map.get (_);
      uint32_t *new_idx;
      if (mapping-> has (f_idx, &new_idx))
      {
        mapping_w_duplicates->set (_, *new_idx);
      }
      else
      {
        mapping->set (_, i);
        mapping_w_duplicates->set (_, i);
        i++;
      }
    }
  }
}

template <typename T>
static void
_closure_glyphs_lookups_features (hb_subset_plan_t   *plan,
				 hb_set_t	     *gids_to_retain,
				 hb_map_t	     *lookups,
				 hb_map_t	     *features,
				 hb_map_t	     *features_w_duplicates,
				 script_langsys_map *langsys_map,
				 hb_hashmap_t<unsigned, hb::shared_ptr<hb_set_t>> *feature_record_cond_idx_map,
				 hb_hashmap_t<unsigned, const OT::Feature*> *feature_substitutes_map,
                                 hb_set_t &catch_all_record_feature_idxes,
                                 hb_hashmap_t<unsigned, hb_pair_t<const void*, const void*>>& catch_all_record_idx_feature_map)
{
  hb_blob_ptr_t<T> table = plan->source_table<T> ();
  hb_tag_t table_tag = table->tableTag;
  hb_set_t lookup_indices, feature_indices;
  _collect_layout_indices<T> (plan,
                              *table,
                              &lookup_indices,
                              &feature_indices,
                              feature_record_cond_idx_map,
                              feature_substitutes_map,
                              catch_all_record_feature_idxes,
                              catch_all_record_idx_feature_map);

  if (table_tag == HB_OT_TAG_GSUB && !(plan->flags & HB_SUBSET_FLAGS_NO_LAYOUT_CLOSURE))
    hb_ot_layout_lookups_substitute_closure (plan->source,
                                             &lookup_indices,
					     gids_to_retain);
  table->closure_lookups (plan->source,
			  gids_to_retain,
                          &lookup_indices);
  remap_indexes (&lookup_indices, lookups);

  // prune features
  table->prune_features (lookups,
                         plan->user_axes_location.is_empty () ? nullptr : feature_record_cond_idx_map,
                         feature_substitutes_map,
                         &feature_indices);
  hb_map_t duplicate_feature_map;
  _GSUBGPOS_find_duplicate_features (*table, lookups, &feature_indices, feature_substitutes_map, &duplicate_feature_map);

  feature_indices.clear ();
  table->prune_langsys (&duplicate_feature_map, &plan->layout_scripts, langsys_map, &feature_indices);
  remap_feature_indices (feature_indices, duplicate_feature_map, catch_all_record_idx_feature_map, features, features_w_duplicates);

  table.destroy ();
}

void layout_nameid_closure (hb_subset_plan_t* plan,
                             hb_set_t* drop_tables)
{
  if (!drop_tables->has (HB_OT_TAG_GPOS))
  {
    hb_blob_ptr_t<GPOS> gpos = plan->source_table<GPOS> ();
    gpos->collect_name_ids (&plan->gpos_features, &plan->name_ids);
    gpos.destroy ();
  }
  if (!drop_tables->has (HB_OT_TAG_GSUB))
  {
    hb_blob_ptr_t<GSUB> gsub = plan->source_table<GSUB> ();
    gsub->collect_name_ids (&plan->gsub_features, &plan->name_ids);
    gsub.destroy ();
  }
}

void
layout_populate_gids_to_retain (hb_subset_plan_t* plan,
		                hb_set_t* drop_tables) {
  if (!drop_tables->has (HB_OT_TAG_GSUB))
    // closure all glyphs/lookups/features needed for GSUB substitutions.
    _closure_glyphs_lookups_features<GSUB> (
        plan,
        &plan->_glyphset_gsub,
        &plan->gsub_lookups,
        &plan->gsub_features,
        &plan->gsub_features_w_duplicates,
        &plan->gsub_langsys,
        &plan->gsub_feature_record_cond_idx_map,
        &plan->gsub_feature_substitutes_map,
        plan->gsub_old_features,
        plan->gsub_old_feature_idx_tag_map);

  if (!drop_tables->has (HB_OT_TAG_GPOS))
    _closure_glyphs_lookups_features<GPOS> (
        plan,
        &plan->_glyphset_gsub,
        &plan->gpos_lookups,
        &plan->gpos_features,
        &plan->gpos_features_w_duplicates,
        &plan->gpos_langsys,
        &plan->gpos_feature_record_cond_idx_map,
        &plan->gpos_feature_substitutes_map,
        plan->gpos_old_features,
        plan->gpos_old_feature_idx_tag_map);
}

#ifndef HB_NO_VAR
void
collect_layout_variation_indices (hb_subset_plan_t* plan)
{
  hb_blob_ptr_t<OT::GDEF> gdef = plan->source_table<OT::GDEF> ();
  hb_blob_ptr_t<GPOS> gpos = plan->source_table<GPOS> ();

  if (!gdef->has_data () || !gdef->has_var_store ())
  {
    gdef.destroy ();
    gpos.destroy ();
    return;
  }

  hb_set_t varidx_set;
  OT::hb_collect_variation_indices_context_t c (&varidx_set,
                                                &plan->_glyphset_gsub,
                                                &plan->gpos_lookups);
  gdef->collect_variation_indices (&c);

  if (hb_ot_layout_has_positioning (plan->source))
    gpos->collect_variation_indices (&c);

  remap_variation_indices (gdef->get_var_store (),
                           varidx_set, plan->normalized_coords,
                           !plan->pinned_at_default,
                           plan->all_axes_pinned,
                           plan->layout_variation_idx_delta_map);

  unsigned subtable_count = gdef->get_var_store ().get_sub_table_count ();
  generate_varstore_inner_maps (varidx_set, subtable_count, plan->gdef_varstore_inner_maps);

  gdef.destroy ();
  gpos.destroy ();
}
#endif

#endif
