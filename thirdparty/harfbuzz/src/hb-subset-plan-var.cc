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

 #include "hb-ot-layout-common.hh"
#include "hb-subset-plan.hh"

 #include "hb-ot-var-common.hh"
 #include "hb-ot-layout-base-table.hh"
 #include "hb-ot-glyf-table.hh"
 #include "hb-ot-var-fvar-table.hh"
 #include "hb-ot-var-avar-table.hh"
 #include "hb-ot-cff2-table.hh"

 #ifndef HB_NO_VAR

 void
 generate_varstore_inner_maps (const hb_set_t& varidx_set,
                               unsigned subtable_count,
                               hb_vector_t<hb_inc_bimap_t> &inner_maps /* OUT */)
 {
   if (varidx_set.is_empty () || subtable_count == 0) return;

   if (unlikely (!inner_maps.resize (subtable_count))) return;
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
   if (!vars.alloc (plan->user_axes_location.get_population ())) {
     hb_font_destroy (font);
     return nullptr;
   }

   for (auto _ : plan->user_axes_location)
   {
     hb_variation_t var;
     var.tag = _.first;
     var.value = _.second.middle;
     vars.push (var);
   }

   hb_font_set_variations (font, vars.arrayZ, plan->user_axes_location.get_population ());
   return font;
 }

 template<typename ItemVarStore>
 void
 remap_variation_indices (const ItemVarStore &var_store,
                          const hb_set_t &variation_indices,
                          const hb_vector_t<int>& normalized_coords,
                          bool calculate_delta, /* not pinned at default */
                          bool no_variations, /* all axes pinned */
                          hb_hashmap_t<unsigned, hb_pair_t<unsigned, int>> &variation_idx_delta_map /* OUT */)
 {
   if (&var_store == &Null (OT::ItemVariationStore)) return;
   unsigned subtable_count = var_store.get_sub_table_count ();
   auto *store_cache = var_store.create_cache ();

   unsigned new_major = 0, new_minor = 0;
   unsigned last_major = (variation_indices.get_min ()) >> 16;
   for (unsigned idx : variation_indices)
   {
     int delta = 0;
     if (calculate_delta)
       delta = roundf (var_store.get_delta (idx, normalized_coords.arrayZ,
                                            normalized_coords.length, store_cache));

     if (no_variations)
     {
       variation_idx_delta_map.set (idx, hb_pair_t<unsigned, int> (HB_OT_LAYOUT_NO_VARIATIONS_INDEX, delta));
       continue;
     }

     uint16_t major = idx >> 16;
     if (major >= subtable_count) break;
     if (major != last_major)
     {
       new_minor = 0;
       ++new_major;
     }

     unsigned new_idx = (new_major << 16) + new_minor;
     variation_idx_delta_map.set (idx, hb_pair_t<unsigned, int> (new_idx, delta));
     ++new_minor;
     last_major = major;
   }
   var_store.destroy_cache (store_cache);
 }

 template
 void
 remap_variation_indices<OT::ItemVariationStore> (const OT::ItemVariationStore &var_store,
                          const hb_set_t &variation_indices,
                          const hb_vector_t<int>& normalized_coords,
                          bool calculate_delta, /* not pinned at default */
                          bool no_variations, /* all axes pinned */
                          hb_hashmap_t<unsigned, hb_pair_t<unsigned, int>> &variation_idx_delta_map /* OUT */);

 #ifndef HB_NO_BASE
 void
 collect_base_variation_indices (hb_subset_plan_t* plan)
 {
   hb_blob_ptr_t<OT::BASE> base = plan->source_table<OT::BASE> ();
   if (!base->has_var_store ())
   {
     base.destroy ();
     return;
   }

   hb_set_t varidx_set;
   base->collect_variation_indices (plan, varidx_set);
   const OT::ItemVariationStore &var_store = base->get_var_store ();
   unsigned subtable_count = var_store.get_sub_table_count ();


   remap_variation_indices (var_store, varidx_set,
                             plan->normalized_coords,
                             !plan->pinned_at_default,
                             plan->all_axes_pinned,
                             plan->base_variation_idx_map);
   generate_varstore_inner_maps (varidx_set, subtable_count, plan->base_varstore_inner_maps);

   base.destroy ();
 }

 #endif

void
normalize_axes_location (hb_face_t *face, hb_subset_plan_t *plan)
{
  if (plan->user_axes_location.is_empty ())
    return;

  hb_array_t<const OT::AxisRecord> axes = face->table.fvar->get_axes ();
  plan->normalized_coords.resize (axes.length);

  bool has_avar = face->table.avar->has_data ();
  const OT::SegmentMaps *seg_maps = nullptr;
  unsigned avar_axis_count = 0;
  if (has_avar)
  {
    seg_maps = face->table.avar->get_segment_maps ();
    avar_axis_count = face->table.avar->get_axis_count();
  }

  bool axis_not_pinned = false;
  unsigned old_axis_idx = 0, new_axis_idx = 0;
  for (const auto& axis : axes)
  {
    hb_tag_t axis_tag = axis.get_axis_tag ();
    plan->axes_old_index_tag_map.set (old_axis_idx, axis_tag);

    if (!plan->user_axes_location.has (axis_tag) ||
        !plan->user_axes_location.get (axis_tag).is_point ())
    {
      axis_not_pinned = true;
      plan->axes_index_map.set (old_axis_idx, new_axis_idx);
      plan->axis_tags.push (axis_tag);
      new_axis_idx++;
    }

    Triple *axis_range;
    if (plan->user_axes_location.has (axis_tag, &axis_range))
    {
      plan->axes_triple_distances.set (axis_tag, axis.get_triple_distances ());

      float normalized_min = axis.normalize_axis_value (axis_range->minimum);
      float normalized_default = axis.normalize_axis_value (axis_range->middle);
      float normalized_max = axis.normalize_axis_value (axis_range->maximum);

      // TODO(behdad): Spec says axis normalization should be done in 16.16;
      // We used to do it in 2.14, but that's not correct.  I fixed this in
      // the fvar/avar code, but keeping 2.14 here for now to keep tests
      // happy. We might need to adjust fonttools as well.
      // I'm only fairly confident in the above statement. Anyway,
      // we should look deeper into this, and also update fonttools if
      // needed.

      // Round to 2.14
      normalized_min = roundf (normalized_min * 16384.f) / 16384.f;
      normalized_default = roundf (normalized_default * 16384.f) / 16384.f;
      normalized_max = roundf (normalized_max * 16384.f) / 16384.f;

      if (has_avar && old_axis_idx < avar_axis_count)
      {
	normalized_min = seg_maps->map_float (normalized_min);
	normalized_default = seg_maps->map_float (normalized_default);
	normalized_max = seg_maps->map_float (normalized_max);

	// Round to 2.14
	normalized_min = roundf (normalized_min * 16384.f) / 16384.f;
	normalized_default = roundf (normalized_default * 16384.f) / 16384.f;
	normalized_max = roundf (normalized_max * 16384.f) / 16384.f;
      }
      plan->axes_location.set (axis_tag, Triple ((double) normalized_min,
                                                 (double) normalized_default,
                                                 (double) normalized_max));

      if (normalized_default == -0.f)
        normalized_default = 0.f; // Normalize -0 to 0
      if (normalized_default != 0.f)
        plan->pinned_at_default = false;

      plan->normalized_coords[old_axis_idx] = roundf (normalized_default * 16384.f);
    }

    old_axis_idx++;

    if (has_avar && old_axis_idx < avar_axis_count)
      seg_maps = &StructAfter<OT::SegmentMaps> (*seg_maps);
  }
  plan->all_axes_pinned = !axis_not_pinned;
}

void
update_instance_metrics_map_from_cff2 (hb_subset_plan_t *plan)
{
  if (!plan->normalized_coords) return;
  OT::cff2::accelerator_t cff2 (plan->source);
  if (!cff2.is_valid ()) return;

  hb_font_t *font = _get_hb_font_with_variations (plan);
  if (unlikely (!plan->check_success (font != nullptr)))
  {
    hb_font_destroy (font);
    return;
  }

  hb_glyph_extents_t extents = {0x7FFF, -0x7FFF};
  OT::hmtx_accelerator_t _hmtx (plan->source);
  OT::hb_scalar_cache_t *hvar_store_cache = nullptr;
  if (_hmtx.has_data () && _hmtx.var_table.get_length ())
    hvar_store_cache = _hmtx.var_table->get_var_store ().create_cache ();

  OT::vmtx_accelerator_t _vmtx (plan->source);
  OT::hb_scalar_cache_t *vvar_store_cache = nullptr;
  if (_vmtx.has_data () && _vmtx.var_table.get_length ())
    vvar_store_cache = _vmtx.var_table->get_var_store ().create_cache ();

  for (auto p : *plan->glyph_map)
  {
    hb_codepoint_t old_gid = p.first;
    hb_codepoint_t new_gid = p.second;
    if (!cff2.get_extents (font, old_gid, &extents)) continue;
    bool has_bounds_info = true;
    if (extents.x_bearing == 0 && extents.width == 0 &&
        extents.height == 0 && extents.y_bearing == 0)
      has_bounds_info = false;

    if (has_bounds_info)
    {
      plan->head_maxp_info.xMin = hb_min (plan->head_maxp_info.xMin, extents.x_bearing);
      plan->head_maxp_info.xMax = hb_max (plan->head_maxp_info.xMax, extents.x_bearing + extents.width);
      plan->head_maxp_info.yMax = hb_max (plan->head_maxp_info.yMax, extents.y_bearing);
      plan->head_maxp_info.yMin = hb_min (plan->head_maxp_info.yMin, extents.y_bearing + extents.height);
    }

    if (_hmtx.has_data ())
    {
      int hori_aw = _hmtx.get_advance_without_var_unscaled (old_gid);
      if (_hmtx.var_table.get_length ())
        hori_aw += (int) roundf (_hmtx.var_table->get_advance_delta_unscaled (old_gid, font->coords, font->num_coords,
                                                                              hvar_store_cache));
      int lsb = extents.x_bearing;
      if (!has_bounds_info)
      {
        _hmtx.get_leading_bearing_without_var_unscaled (old_gid, &lsb);
      }
      plan->hmtx_map.set (new_gid, hb_pair ((unsigned) hori_aw, lsb));
      plan->bounds_width_vec[new_gid] = extents.width;
    }

    if (_vmtx.has_data ())
    {
      int vert_aw = _vmtx.get_advance_without_var_unscaled (old_gid);
      if (_vmtx.var_table.get_length ())
        vert_aw += (int) roundf (_vmtx.var_table->get_advance_delta_unscaled (old_gid, font->coords, font->num_coords,
                                                                              vvar_store_cache));
      hb_position_t vorg_x = 0;
      hb_position_t vorg_y = 0;
      int tsb = 0;
      if (has_bounds_info &&
           hb_font_get_glyph_v_origin (font, old_gid, &vorg_x, &vorg_y))
      {
        tsb = vorg_y - extents.y_bearing;
      } else {
        _vmtx.get_leading_bearing_without_var_unscaled (old_gid, &tsb);
      }

      plan->vmtx_map.set (new_gid, hb_pair ((unsigned) vert_aw, tsb));
      plan->bounds_height_vec[new_gid] = extents.height;
    }
  }
  hb_font_destroy (font);
  if (hvar_store_cache)
    _hmtx.var_table->get_var_store ().destroy_cache (hvar_store_cache);
  if (vvar_store_cache)
    _vmtx.var_table->get_var_store ().destroy_cache (vvar_store_cache);
}

bool
get_instance_glyphs_contour_points (hb_subset_plan_t *plan)
{
  /* contour_points vector only needed for updating gvar table (infer delta and
   * iup delta optimization) during partial instancing */
  if (plan->user_axes_location.is_empty () || plan->all_axes_pinned)
    return true;

  OT::glyf_accelerator_t glyf (plan->source);

  for (auto &_ : plan->new_to_old_gid_list)
  {
    hb_codepoint_t new_gid = _.first;
    contour_point_vector_t all_points;
    if (new_gid == 0 && !(plan->flags & HB_SUBSET_FLAGS_NOTDEF_OUTLINE))
    {
      if (unlikely (!plan->new_gid_contour_points_map.set (new_gid, all_points)))
        return false;
      continue;
    }

    hb_codepoint_t old_gid = _.second;
    auto glyph = glyf.glyph_for_gid (old_gid);
    if (unlikely (!glyph.get_all_points_without_var (plan->source, all_points)))
      return false;
    if (unlikely (!plan->new_gid_contour_points_map.set (new_gid, all_points)))
      return false;

    /* composite new gids are only needed by iup delta optimization */
    if ((plan->flags & HB_SUBSET_FLAGS_OPTIMIZE_IUP_DELTAS) && glyph.is_composite ())
      plan->composite_new_gids.add (new_gid);
  }
  return true;
}

template<typename DeltaSetIndexMap>
void
remap_colrv1_delta_set_index_indices (const DeltaSetIndexMap &index_map,
                                      const hb_set_t &delta_set_idxes,
                                      hb_hashmap_t<unsigned, hb_pair_t<unsigned, int>> &variation_idx_delta_map, /* IN/OUT */
                                      hb_map_t &new_deltaset_idx_varidx_map /* OUT */)
{
  if (!index_map.get_map_count ())
    return;

  hb_hashmap_t<unsigned, hb_pair_t<unsigned, int>> delta_set_idx_delta_map;
  unsigned new_delta_set_idx = 0;
  for (unsigned delta_set_idx : delta_set_idxes)
  {
    unsigned var_idx = index_map.map (delta_set_idx);
    unsigned new_varidx = HB_OT_LAYOUT_NO_VARIATIONS_INDEX;
    int delta = 0;

    if (var_idx != HB_OT_LAYOUT_NO_VARIATIONS_INDEX)
    {
      hb_pair_t<unsigned, int> *new_varidx_delta;
      if (!variation_idx_delta_map.has (var_idx, &new_varidx_delta)) continue;

      new_varidx = hb_first (*new_varidx_delta);
      delta = hb_second (*new_varidx_delta);
    }

    new_deltaset_idx_varidx_map.set (new_delta_set_idx, new_varidx);
    delta_set_idx_delta_map.set (delta_set_idx, hb_pair_t<unsigned, int> (new_delta_set_idx, delta));
    new_delta_set_idx++;
  }
  variation_idx_delta_map = std::move (delta_set_idx_delta_map);
}

template void
remap_colrv1_delta_set_index_indices<OT::DeltaSetIndexMap> (const OT::DeltaSetIndexMap &index_map,
                                      const hb_set_t &delta_set_idxes,
                                      hb_hashmap_t<unsigned, hb_pair_t<unsigned, int>> &variation_idx_delta_map, /* IN/OUT */
                                      hb_map_t &new_deltaset_idx_varidx_map /* OUT */);

 #endif
