/*
 * Copyright © 2018  Google, Inc.
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
 * Google Author(s): Garret Rieger, Roderick Sheeter
 */

#include "hb-subset-plan.hh"
#include "hb-subset-accelerator.hh"
#include "hb-map.hh"
#include "hb-set.hh"

#include "hb-ot-cmap-table.hh"
#include "hb-ot-glyf-table.hh"
#include "hb-ot-layout-gdef-table.hh"
#include "hb-ot-layout-gpos-table.hh"
#include "hb-ot-layout-gsub-table.hh"
#include "hb-ot-cff1-table.hh"
#include "hb-ot-color-colr-table.hh"
#include "hb-ot-color-colrv1-closure.hh"
#include "hb-ot-var-fvar-table.hh"
#include "hb-ot-var-avar-table.hh"
#include "hb-ot-stat-table.hh"
#include "hb-ot-math-table.hh"

using OT::Layout::GSUB;
using OT::Layout::GPOS;

typedef hb_hashmap_t<unsigned, hb::unique_ptr<hb_set_t>> script_langsys_map;
#ifndef HB_NO_SUBSET_CFF
static inline void
_add_cff_seac_components (const OT::cff1::accelerator_t &cff,
			  hb_codepoint_t gid,
			  hb_set_t *gids_to_retain)
{
  hb_codepoint_t base_gid, accent_gid;
  if (cff.get_seac_components (gid, &base_gid, &accent_gid))
  {
    gids_to_retain->add (base_gid);
    gids_to_retain->add (accent_gid);
  }
}
#endif

static void
_remap_palette_indexes (const hb_set_t *palette_indexes,
			hb_map_t       *mapping /* OUT */)
{
  unsigned new_idx = 0;
  for (unsigned palette_index : palette_indexes->iter ())
  {
    if (palette_index == 0xFFFF)
    {
      mapping->set (palette_index, palette_index);
      continue;
    }
    mapping->set (palette_index, new_idx);
    new_idx++;
  }
}

static void
_remap_indexes (const hb_set_t *indexes,
		hb_map_t       *mapping /* OUT */)
{
  unsigned count = indexes->get_population ();

  for (auto _ : + hb_zip (indexes->iter (), hb_range (count)))
    mapping->set (_.first, _.second);

}

#ifndef HB_NO_SUBSET_LAYOUT

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
                                     hb_hashmap_t<unsigned, const OT::Feature*> *feature_substitutes_map /* OUT */)
{
  unsigned num_features = table.get_feature_count ();
  hb_vector_t<hb_tag_t> features;
  if (!plan->check_success (features.resize (num_features))) return;
  table.get_feature_tags (0, &num_features, features.arrayZ);
  bool retain_all_features = !_filter_tag_list (&features, plan->layout_features);

  unsigned num_scripts = table.get_script_count ();
  hb_vector_t<hb_tag_t> scripts;
  if (!plan->check_success (scripts.resize (num_scripts))) return;
  table.get_script_tags (0, &num_scripts, scripts.arrayZ);
  bool retain_all_scripts = !_filter_tag_list (&scripts, plan->layout_scripts);

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
  if (!plan->user_axes_location->is_empty ())
  {
    hb_hashmap_t<hb::shared_ptr<hb_map_t>, unsigned> conditionset_map;
    OT::hb_collect_feature_substitutes_with_var_context_t c =
    {
      plan->axes_old_index_tag_map,
      plan->axes_location,
      feature_record_cond_idx_map,
      feature_substitutes_map,
      feature_indices,
      true,
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

  table.feature_variation_collect_lookups (feature_indices, feature_substitutes_map, lookup_indices);
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
        f = *p;

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

template <typename T>
static inline void
_closure_glyphs_lookups_features (hb_subset_plan_t   *plan,
				  hb_set_t	     *gids_to_retain,
				  hb_map_t	     *lookups,
				  hb_map_t	     *features,
				  script_langsys_map *langsys_map,
				  hb_hashmap_t<unsigned, hb::shared_ptr<hb_set_t>> *feature_record_cond_idx_map,
				  hb_hashmap_t<unsigned, const OT::Feature*> *feature_substitutes_map)
{
  hb_blob_ptr_t<T> table = plan->source_table<T> ();
  hb_tag_t table_tag = table->tableTag;
  hb_set_t lookup_indices, feature_indices;
  _collect_layout_indices<T> (plan,
                              *table,
                              &lookup_indices,
                              &feature_indices,
                              feature_record_cond_idx_map,
                              feature_substitutes_map);

  if (table_tag == HB_OT_TAG_GSUB)
    hb_ot_layout_lookups_substitute_closure (plan->source,
                                             &lookup_indices,
					     gids_to_retain);
  table->closure_lookups (plan->source,
			  gids_to_retain,
                          &lookup_indices);
  _remap_indexes (&lookup_indices, lookups);

  // prune features
  table->prune_features (lookups,
                         plan->user_axes_location->is_empty () ? nullptr : feature_record_cond_idx_map,
                         feature_substitutes_map,
                         &feature_indices);
  hb_map_t duplicate_feature_map;
  _GSUBGPOS_find_duplicate_features (*table, lookups, &feature_indices, feature_substitutes_map, &duplicate_feature_map);

  feature_indices.clear ();
  table->prune_langsys (&duplicate_feature_map, plan->layout_scripts, langsys_map, &feature_indices);
  _remap_indexes (&feature_indices, features);

  table.destroy ();
}

#endif

#ifndef HB_NO_VAR
static inline void
_generate_varstore_inner_maps (const hb_set_t& varidx_set,
                               unsigned subtable_count,
                               hb_vector_t<hb_inc_bimap_t> &inner_maps /* OUT */)
{
  if (varidx_set.is_empty () || subtable_count == 0) return;

  inner_maps.resize (subtable_count);
  for (unsigned idx : varidx_set)
  {
    uint16_t major = idx >> 16;
    uint16_t minor = idx & 0xFFFF;

    if (major >= subtable_count)
      continue;
    inner_maps[major].add (minor);
  }
}

static inline hb_font_t*
_get_hb_font_with_variations (const hb_subset_plan_t *plan)
{
  hb_font_t *font = hb_font_create (plan->source);

  hb_vector_t<hb_variation_t> vars;
  vars.alloc (plan->user_axes_location->get_population ());

  for (auto _ : *plan->user_axes_location)
  {
    hb_variation_t var;
    var.tag = _.first;
    var.value = _.second;
    vars.push (var);
  }

  hb_font_set_variations (font, vars.arrayZ, plan->user_axes_location->get_population ());
  return font;
}

static inline void
_collect_layout_variation_indices (hb_subset_plan_t* plan)
{
  hb_blob_ptr_t<OT::GDEF> gdef = plan->source_table<OT::GDEF> ();
  hb_blob_ptr_t<GPOS> gpos = plan->source_table<GPOS> ();

  if (!gdef->has_data ())
  {
    gdef.destroy ();
    gpos.destroy ();
    return;
  }

  const OT::VariationStore *var_store = nullptr;
  hb_set_t varidx_set;
  hb_font_t *font = nullptr;
  float *store_cache = nullptr;
  bool collect_delta = plan->pinned_at_default ? false : true;
  if (collect_delta)
  {
    font = _get_hb_font_with_variations (plan);
    if (gdef->has_var_store ())
    {
      var_store = &(gdef->get_var_store ());
      store_cache = var_store->create_cache ();
    }
  }

  OT::hb_collect_variation_indices_context_t c (&varidx_set,
                                                plan->layout_variation_idx_delta_map,
                                                font, var_store,
                                                plan->_glyphset_gsub,
                                                plan->gpos_lookups,
                                                store_cache);
  gdef->collect_variation_indices (&c);

  if (hb_ot_layout_has_positioning (plan->source))
    gpos->collect_variation_indices (&c);

  hb_font_destroy (font);
  var_store->destroy_cache (store_cache);

  gdef->remap_layout_variation_indices (&varidx_set, plan->layout_variation_idx_delta_map);

  unsigned subtable_count = gdef->has_var_store () ? gdef->get_var_store ().get_sub_table_count () : 0;
  _generate_varstore_inner_maps (varidx_set, subtable_count, plan->gdef_varstore_inner_maps);

  gdef.destroy ();
  gpos.destroy ();
}
#endif

static inline void
_cmap_closure (hb_face_t	   *face,
	       const hb_set_t	   *unicodes,
	       hb_set_t		   *glyphset)
{
  OT::cmap::accelerator_t cmap (face);
  cmap.table->closure_glyphs (unicodes, glyphset);
}

static void _colr_closure (hb_face_t *face,
                           hb_map_t *layers_map,
                           hb_map_t *palettes_map,
                           hb_set_t *glyphs_colred)
{
  OT::COLR::accelerator_t colr (face);
  if (!colr.is_valid ()) return;

  hb_set_t palette_indices, layer_indices;
  // Collect all glyphs referenced by COLRv0
  hb_set_t glyphset_colrv0;
  for (hb_codepoint_t gid : *glyphs_colred)
    colr.closure_glyphs (gid, &glyphset_colrv0);

  glyphs_colred->union_ (glyphset_colrv0);

  //closure for COLRv1
  colr.closure_forV1 (glyphs_colred, &layer_indices, &palette_indices);

  colr.closure_V0palette_indices (glyphs_colred, &palette_indices);
  _remap_indexes (&layer_indices, layers_map);
  _remap_palette_indexes (&palette_indices, palettes_map);
}

static inline void
_math_closure (hb_subset_plan_t *plan,
               hb_set_t         *glyphset)
{
  hb_blob_ptr_t<OT::MATH> math = plan->source_table<OT::MATH> ();
  if (math->has_data ())
    math->closure_glyphs (glyphset);
  math.destroy ();
}


static inline void
_remove_invalid_gids (hb_set_t *glyphs,
		      unsigned int num_glyphs)
{
  glyphs->del_range (num_glyphs, HB_SET_VALUE_INVALID);
}

static void
_populate_unicodes_to_retain (const hb_set_t *unicodes,
                              const hb_set_t *glyphs,
                              hb_subset_plan_t *plan)
{
  OT::cmap::accelerator_t cmap (plan->source);
  unsigned size_threshold = plan->source->get_num_glyphs ();
  if (glyphs->is_empty () && unicodes->get_population () < size_threshold)
  {

    const hb_map_t* unicode_to_gid = nullptr;
    if (plan->accelerator)
      unicode_to_gid = &plan->accelerator->unicode_to_gid;

    // This is approach to collection is faster, but can only be used  if glyphs
    // are not being explicitly added to the subset and the input unicodes set is
    // not excessively large (eg. an inverted set).
    plan->unicode_to_new_gid_list.alloc (unicodes->get_population ());
    if (!unicode_to_gid) {
      for (hb_codepoint_t cp : *unicodes)
      {
        hb_codepoint_t gid;
        if (!cmap.get_nominal_glyph (cp, &gid))
        {
          DEBUG_MSG(SUBSET, nullptr, "Drop U+%04X; no gid", cp);
          continue;
        }

        plan->codepoint_to_glyph->set (cp, gid);
        plan->unicode_to_new_gid_list.push (hb_pair (cp, gid));
      }
    } else {
      // Use in memory unicode to gid map it's faster then looking up from
      // the map. This code is mostly duplicated from above to avoid doing
      // conditionals on the presence of the unicode_to_gid map each
      // iteration.
      for (hb_codepoint_t cp : *unicodes)
      {
        hb_codepoint_t gid = unicode_to_gid->get (cp);
        if (gid == HB_MAP_VALUE_INVALID)
        {
          DEBUG_MSG(SUBSET, nullptr, "Drop U+%04X; no gid", cp);
          continue;
        }

        plan->codepoint_to_glyph->set (cp, gid);
        plan->unicode_to_new_gid_list.push (hb_pair (cp, gid));
      }
    }
  }
  else
  {
    // This approach is slower, but can handle adding in glyphs to the subset and will match
    // them with cmap entries.

    hb_map_t unicode_glyphid_map_storage;
    hb_set_t cmap_unicodes_storage;
    const hb_map_t* unicode_glyphid_map = &unicode_glyphid_map_storage;
    const hb_set_t* cmap_unicodes = &cmap_unicodes_storage;

    if (!plan->accelerator) {
      cmap.collect_mapping (&cmap_unicodes_storage, &unicode_glyphid_map_storage);
      plan->unicode_to_new_gid_list.alloc (hb_min(unicodes->get_population ()
                                                  + glyphs->get_population (),
                                                  cmap_unicodes->get_population ()));
    } else {
      unicode_glyphid_map = &plan->accelerator->unicode_to_gid;
      cmap_unicodes = &plan->accelerator->unicodes;
    }

    for (hb_codepoint_t cp : *cmap_unicodes)
    {
      hb_codepoint_t gid = (*unicode_glyphid_map)[cp];
      if (!unicodes->has (cp) && !glyphs->has (gid))
        continue;

      plan->codepoint_to_glyph->set (cp, gid);
      plan->unicode_to_new_gid_list.push (hb_pair (cp, gid));
    }

    /* Add gids which where requested, but not mapped in cmap */
    for (hb_codepoint_t gid : *glyphs)
    {
      if (gid >= plan->source->get_num_glyphs ())
	break;
      plan->_glyphset_gsub->add (gid);
    }
  }

  auto &arr = plan->unicode_to_new_gid_list;
  if (arr.length)
  {
    plan->unicodes->add_sorted_array (&arr.arrayZ->first, arr.length, sizeof (*arr.arrayZ));
    plan->_glyphset_gsub->add_array (&arr.arrayZ->second, arr.length, sizeof (*arr.arrayZ));
  }
}

#ifndef HB_COMPOSITE_OPERATIONS_PER_GLYPH
#define HB_COMPOSITE_OPERATIONS_PER_GLYPH 64
#endif

static unsigned
_glyf_add_gid_and_children (const OT::glyf_accelerator_t &glyf,
			    hb_codepoint_t gid,
			    hb_set_t *gids_to_retain,
			    int operation_count,
			    unsigned depth = 0)
{
  if (unlikely (depth++ > HB_MAX_NESTING_LEVEL)) return operation_count;
  if (unlikely (--operation_count < 0)) return operation_count;
  /* Check if is already visited */
  if (gids_to_retain->has (gid)) return operation_count;

  gids_to_retain->add (gid);

  for (auto item : glyf.glyph_for_gid (gid).get_composite_iterator ())
    operation_count =
      _glyf_add_gid_and_children (glyf,
				  item.get_gid (),
				  gids_to_retain,
				  operation_count,
				  depth);

  return operation_count;
}

static void
_populate_gids_to_retain (hb_subset_plan_t* plan,
		          hb_set_t* drop_tables)
{
  OT::glyf_accelerator_t glyf (plan->source);
#ifndef HB_NO_SUBSET_CFF
  OT::cff1::accelerator_t cff (plan->source);
#endif

  plan->_glyphset_gsub->add (0); // Not-def

  _cmap_closure (plan->source, plan->unicodes, plan->_glyphset_gsub);

#ifndef HB_NO_SUBSET_LAYOUT
  if (!drop_tables->has (HB_OT_TAG_GSUB))
    // closure all glyphs/lookups/features needed for GSUB substitutions.
    _closure_glyphs_lookups_features<GSUB> (
        plan,
        plan->_glyphset_gsub,
        plan->gsub_lookups,
        plan->gsub_features,
        plan->gsub_langsys,
        plan->gsub_feature_record_cond_idx_map,
        plan->gsub_feature_substitutes_map);

  if (!drop_tables->has (HB_OT_TAG_GPOS))
    _closure_glyphs_lookups_features<GPOS> (
        plan,
        plan->_glyphset_gsub,
        plan->gpos_lookups,
        plan->gpos_features,
        plan->gpos_langsys,
        plan->gpos_feature_record_cond_idx_map,
        plan->gpos_feature_substitutes_map);
#endif
  _remove_invalid_gids (plan->_glyphset_gsub, plan->source->get_num_glyphs ());

  hb_set_set (plan->_glyphset_mathed, plan->_glyphset_gsub);
  if (!drop_tables->has (HB_OT_TAG_MATH))
  {
    _math_closure (plan, plan->_glyphset_mathed);
    _remove_invalid_gids (plan->_glyphset_mathed, plan->source->get_num_glyphs ());
  }

  hb_set_t cur_glyphset = *plan->_glyphset_mathed;
  if (!drop_tables->has (HB_OT_TAG_COLR))
  {
    _colr_closure (plan->source, plan->colrv1_layers, plan->colr_palettes, &cur_glyphset);
    _remove_invalid_gids (&cur_glyphset, plan->source->get_num_glyphs ());
  }

  hb_set_set (plan->_glyphset_colred, &cur_glyphset);

  /* Populate a full set of glyphs to retain by adding all referenced
   * composite glyphs. */
  if (glyf.has_data ())
    for (hb_codepoint_t gid : cur_glyphset)
      _glyf_add_gid_and_children (glyf, gid, plan->_glyphset,
				  cur_glyphset.get_population () * HB_COMPOSITE_OPERATIONS_PER_GLYPH);
  else
    plan->_glyphset->union_ (cur_glyphset);
#ifndef HB_NO_SUBSET_CFF
  if (cff.is_valid ())
    for (hb_codepoint_t gid : cur_glyphset)
      _add_cff_seac_components (cff, gid, plan->_glyphset);
#endif

  _remove_invalid_gids (plan->_glyphset, plan->source->get_num_glyphs ());


#ifndef HB_NO_VAR
  if (!drop_tables->has (HB_OT_TAG_GDEF))
    _collect_layout_variation_indices (plan);
#endif
}

static void
_create_glyph_map_gsub (const hb_set_t* glyph_set_gsub,
                        const hb_map_t* glyph_map,
                        hb_map_t* out)
{
  + hb_iter (glyph_set_gsub)
  | hb_map ([&] (hb_codepoint_t gid) {
    return hb_pair_t<hb_codepoint_t, hb_codepoint_t> (gid,
                                                      glyph_map->get (gid));
  })
  | hb_sink (out)
  ;
}

static void
_create_old_gid_to_new_gid_map (const hb_face_t *face,
				bool		 retain_gids,
				const hb_set_t	*all_gids_to_retain,
				hb_map_t	*glyph_map, /* OUT */
				hb_map_t	*reverse_glyph_map, /* OUT */
				unsigned int	*num_glyphs /* OUT */)
{
  unsigned pop = all_gids_to_retain->get_population ();
  reverse_glyph_map->resize (pop);
  glyph_map->resize (pop);

  if (!retain_gids)
  {
    + hb_enumerate (hb_iter (all_gids_to_retain), (hb_codepoint_t) 0)
    | hb_sink (reverse_glyph_map)
    ;
    *num_glyphs = reverse_glyph_map->get_population ();
  }
  else
  {
    + hb_iter (all_gids_to_retain)
    | hb_map ([] (hb_codepoint_t _) {
		return hb_pair_t<hb_codepoint_t, hb_codepoint_t> (_, _);
	      })
    | hb_sink (reverse_glyph_map)
    ;

    hb_codepoint_t max_glyph = HB_SET_VALUE_INVALID;
    hb_set_previous (all_gids_to_retain, &max_glyph);

    *num_glyphs = max_glyph + 1;
  }

  + reverse_glyph_map->iter ()
  | hb_map (&hb_pair_t<hb_codepoint_t, hb_codepoint_t>::reverse)
  | hb_sink (glyph_map)
  ;
}

static void
_nameid_closure (hb_face_t *face,
		 hb_set_t  *nameids,
		 bool all_axes_pinned,
		 hb_hashmap_t<hb_tag_t, float> *user_axes_location)
{
#ifndef HB_NO_STYLE
  face->table.STAT->collect_name_ids (user_axes_location, nameids);
#endif
#ifndef HB_NO_VAR
  if (!all_axes_pinned)
    face->table.fvar->collect_name_ids (user_axes_location, nameids);
#endif
}

#ifndef HB_NO_VAR
static void
_normalize_axes_location (hb_face_t *face, hb_subset_plan_t *plan)
{
  if (plan->user_axes_location->is_empty ())
    return;

  hb_array_t<const OT::AxisRecord> axes = face->table.fvar->get_axes ();

  bool has_avar = face->table.avar->has_data ();
  const OT::SegmentMaps *seg_maps = nullptr;
  if (has_avar)
    seg_maps = face->table.avar->get_segment_maps ();

  bool axis_not_pinned = false;
  unsigned old_axis_idx = 0, new_axis_idx = 0;
  for (const auto& axis : axes)
  {
    hb_tag_t axis_tag = axis.get_axis_tag ();
    plan->axes_old_index_tag_map->set (old_axis_idx, axis_tag);

    if (!plan->user_axes_location->has (axis_tag))
    {
      axis_not_pinned = true;
      plan->axes_index_map->set (old_axis_idx, new_axis_idx);
      new_axis_idx++;
    }
    else
    {
      int normalized_v = axis.normalize_axis_value (plan->user_axes_location->get (axis_tag));
      if (has_avar && old_axis_idx < face->table.avar->get_axis_count ())
      {
        normalized_v = seg_maps->map (normalized_v);
      }
      plan->axes_location->set (axis_tag, normalized_v);
      if (normalized_v != 0)
        plan->pinned_at_default = false;
    }
    if (has_avar)
      seg_maps = &StructAfter<OT::SegmentMaps> (*seg_maps);

    old_axis_idx++;
  }
  plan->all_axes_pinned = !axis_not_pinned;
}
#endif
/**
 * hb_subset_plan_create_or_fail:
 * @face: font face to create the plan for.
 * @input: a #hb_subset_input_t input.
 *
 * Computes a plan for subsetting the supplied face according
 * to a provided input. The plan describes
 * which tables and glyphs should be retained.
 *
 * Return value: (transfer full): New subset plan. Destroy with
 * hb_subset_plan_destroy(). If there is a failure creating the plan
 * nullptr will be returned.
 *
 * Since: 4.0.0
 **/
hb_subset_plan_t *
hb_subset_plan_create_or_fail (hb_face_t	 *face,
                               const hb_subset_input_t *input)
{
  hb_subset_plan_t *plan;
  if (unlikely (!(plan = hb_object_create<hb_subset_plan_t> ())))
    return nullptr;

  plan->successful = true;
  plan->flags = input->flags;
  plan->unicodes = hb_set_create ();

  plan->unicode_to_new_gid_list.init ();

  plan->name_ids = hb_set_copy (input->sets.name_ids);
  plan->name_languages = hb_set_copy (input->sets.name_languages);
  plan->layout_features = hb_set_copy (input->sets.layout_features);
  plan->layout_scripts = hb_set_copy (input->sets.layout_scripts);
  plan->glyphs_requested = hb_set_copy (input->sets.glyphs);
  plan->drop_tables = hb_set_copy (input->sets.drop_tables);
  plan->no_subset_tables = hb_set_copy (input->sets.no_subset_tables);
  plan->source = hb_face_reference (face);
  plan->dest = hb_face_builder_create ();

  plan->_glyphset = hb_set_create ();
  plan->_glyphset_gsub = hb_set_create ();
  plan->_glyphset_mathed = hb_set_create ();
  plan->_glyphset_colred = hb_set_create ();
  plan->codepoint_to_glyph = hb_map_create ();
  plan->glyph_map = hb_map_create ();
  plan->reverse_glyph_map = hb_map_create ();
  plan->glyph_map_gsub = hb_map_create ();
  plan->gsub_lookups = hb_map_create ();
  plan->gpos_lookups = hb_map_create ();

  plan->check_success (plan->gsub_langsys = hb_hashmap_create<unsigned, hb::unique_ptr<hb_set_t>> ());
  plan->check_success (plan->gpos_langsys = hb_hashmap_create<unsigned, hb::unique_ptr<hb_set_t>> ());

  plan->gsub_features = hb_map_create ();
  plan->gpos_features = hb_map_create ();

  plan->check_success (plan->gsub_feature_record_cond_idx_map = hb_hashmap_create<unsigned, hb::shared_ptr<hb_set_t>> ());
  plan->check_success (plan->gpos_feature_record_cond_idx_map = hb_hashmap_create<unsigned, hb::shared_ptr<hb_set_t>> ());

  plan->check_success (plan->gsub_feature_substitutes_map = hb_hashmap_create<unsigned, const OT::Feature*> ());
  plan->check_success (plan->gpos_feature_substitutes_map = hb_hashmap_create<unsigned, const OT::Feature*> ());

  plan->colrv1_layers = hb_map_create ();
  plan->colr_palettes = hb_map_create ();
  plan->check_success (plan->layout_variation_idx_delta_map = hb_hashmap_create<unsigned, hb_pair_t<unsigned, int>> ());
  plan->gdef_varstore_inner_maps.init ();

  plan->check_success (plan->sanitized_table_cache = hb_hashmap_create<hb_tag_t, hb::unique_ptr<hb_blob_t>> ());
  plan->check_success (plan->axes_location = hb_hashmap_create<hb_tag_t, int> ());
  plan->check_success (plan->user_axes_location = hb_hashmap_create<hb_tag_t, float> ());
  if (plan->user_axes_location && input->axes_location)
      *plan->user_axes_location = *input->axes_location;
  plan->check_success (plan->axes_index_map = hb_map_create ());
  plan->check_success (plan->axes_old_index_tag_map = hb_map_create ());
  plan->all_axes_pinned = false;
  plan->pinned_at_default = true;

  plan->check_success (plan->vmtx_map = hb_hashmap_create<unsigned, hb_pair_t<unsigned, int>> ());
  plan->check_success (plan->hmtx_map = hb_hashmap_create<unsigned, hb_pair_t<unsigned, int>> ());

  void* accel = hb_face_get_user_data(face, hb_subset_accelerator_t::user_data_key());

  plan->attach_accelerator_data = input->attach_accelerator_data;
  if (accel)
    plan->accelerator = (hb_subset_accelerator_t*) accel;


  if (unlikely (plan->in_error ())) {
    hb_subset_plan_destroy (plan);
    return nullptr;
  }

#ifndef HB_NO_VAR
  _normalize_axes_location (face, plan);
#endif

  _populate_unicodes_to_retain (input->sets.unicodes, input->sets.glyphs, plan);

  _populate_gids_to_retain (plan, input->sets.drop_tables);

  _create_old_gid_to_new_gid_map (face,
                                  input->flags & HB_SUBSET_FLAGS_RETAIN_GIDS,
				  plan->_glyphset,
				  plan->glyph_map,
				  plan->reverse_glyph_map,
				  &plan->_num_output_glyphs);

  _create_glyph_map_gsub (
      plan->_glyphset_gsub,
      plan->glyph_map,
      plan->glyph_map_gsub);

  // Now that we have old to new gid map update the unicode to new gid list.
  for (unsigned i = 0; i < plan->unicode_to_new_gid_list.length; i++)
  {
    // Use raw array access for performance.
    plan->unicode_to_new_gid_list.arrayZ[i].second =
        plan->glyph_map->get(plan->unicode_to_new_gid_list.arrayZ[i].second);
  }

  _nameid_closure (face, plan->name_ids, plan->all_axes_pinned, plan->user_axes_location);
  if (unlikely (plan->in_error ())) {
    hb_subset_plan_destroy (plan);
    return nullptr;
  }
  return plan;
}

/**
 * hb_subset_plan_destroy:
 * @plan: a #hb_subset_plan_t
 *
 * Decreases the reference count on @plan, and if it reaches zero, destroys
 * @plan, freeing all memory.
 *
 * Since: 4.0.0
 **/
void
hb_subset_plan_destroy (hb_subset_plan_t *plan)
{
  if (!hb_object_destroy (plan)) return;

  hb_free (plan);
}

/**
 * hb_subset_plan_old_to_new_glyph_mapping:
 * @plan: a subsetting plan.
 *
 * Returns the mapping between glyphs in the original font to glyphs in the
 * subset that will be produced by @plan
 *
 * Return value: (transfer none):
 * A pointer to the #hb_map_t of the mapping.
 *
 * Since: 4.0.0
 **/
const hb_map_t*
hb_subset_plan_old_to_new_glyph_mapping (const hb_subset_plan_t *plan)
{
  return plan->glyph_map;
}

/**
 * hb_subset_plan_new_to_old_glyph_mapping:
 * @plan: a subsetting plan.
 *
 * Returns the mapping between glyphs in the subset that will be produced by
 * @plan and the glyph in the original font.
 *
 * Return value: (transfer none):
 * A pointer to the #hb_map_t of the mapping.
 *
 * Since: 4.0.0
 **/
const hb_map_t*
hb_subset_plan_new_to_old_glyph_mapping (const hb_subset_plan_t *plan)
{
  return plan->reverse_glyph_map;
}

/**
 * hb_subset_plan_unicode_to_old_glyph_mapping:
 * @plan: a subsetting plan.
 *
 * Returns the mapping between codepoints in the original font and the
 * associated glyph id in the original font.
 *
 * Return value: (transfer none):
 * A pointer to the #hb_map_t of the mapping.
 *
 * Since: 4.0.0
 **/
const hb_map_t*
hb_subset_plan_unicode_to_old_glyph_mapping (const hb_subset_plan_t *plan)
{
  return plan->codepoint_to_glyph;
}

/**
 * hb_subset_plan_reference: (skip)
 * @plan: a #hb_subset_plan_t object.
 *
 * Increases the reference count on @plan.
 *
 * Return value: @plan.
 *
 * Since: 4.0.0
 **/
hb_subset_plan_t *
hb_subset_plan_reference (hb_subset_plan_t *plan)
{
  return hb_object_reference (plan);
}

/**
 * hb_subset_plan_set_user_data: (skip)
 * @plan: a #hb_subset_plan_t object.
 * @key: The user-data key to set
 * @data: A pointer to the user data
 * @destroy: (nullable): A callback to call when @data is not needed anymore
 * @replace: Whether to replace an existing data with the same key
 *
 * Attaches a user-data key/data pair to the given subset plan object.
 *
 * Return value: `true` if success, `false` otherwise
 *
 * Since: 4.0.0
 **/
hb_bool_t
hb_subset_plan_set_user_data (hb_subset_plan_t   *plan,
                              hb_user_data_key_t *key,
                              void               *data,
                              hb_destroy_func_t   destroy,
                              hb_bool_t	          replace)
{
  return hb_object_set_user_data (plan, key, data, destroy, replace);
}

/**
 * hb_subset_plan_get_user_data: (skip)
 * @plan: a #hb_subset_plan_t object.
 * @key: The user-data key to query
 *
 * Fetches the user data associated with the specified key,
 * attached to the specified subset plan object.
 *
 * Return value: (transfer none): A pointer to the user data
 *
 * Since: 4.0.0
 **/
void *
hb_subset_plan_get_user_data (const hb_subset_plan_t *plan,
                              hb_user_data_key_t     *key)
{
  return hb_object_get_user_data (plan, key);
}
