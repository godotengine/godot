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

 /* Evaluate a variation-region tent at v, mirroring VarRegionAxis::evaluate
  * (invalid regions evaluate as constant 1). */
 static double
 _tent_eval (const Triple &tent, double v)
 {
   double start = tent.minimum, peak = tent.middle, end = tent.maximum;
   if (unlikely (start > peak || peak > end)) return 1.0;
   if (unlikely (start < 0.0 && end > 0.0 && peak != 0.0)) return 1.0;
   if (peak == 0.0 || v == peak) return 1.0;
   if (v <= start || end <= v) return 0.0;
   if (v < peak) return (v - start) / (peak - start);
   return (end - v) / (end - peak);
 }

 /* Range of a tent's scalar over an input interval [lo, hi]. The tent is
  * unimodal, so the extremes are at the interval endpoints, plus the peak
  * when it falls inside. */
 static void
 _tent_value_range (const Triple &tent, double lo, double hi,
                    double *tmin, double *tmax)
 {
   double a = _tent_eval (tent, lo);
   double b = _tent_eval (tent, hi);
   *tmin = hb_min (a, b);
   *tmax = hb_max (a, b);
   if (lo <= tent.middle && tent.middle <= hi)
     *tmax = hb_max (*tmax, 1.0);
 }

 /* Compute conservative reachable old-space final-coord ranges per axis for
  * avar2 partial instancing. With offset compensation, the instanced font's
  * final coordinates equal the original font's over the retained user box,
  * so the reachable range of axis i's final coordinate is the range of
  *   final_i = intermediate_i + delta_i(intermediates)
  * over the box of retained old-intermediate ranges: restricted axes span
  * their retained [a, b], free public axes [-1, +1], pinned axes sit at
  * their d_i, and hidden (private) axes at 0. Interval arithmetic over the
  * avar2 VarStore regions bounds delta_i. Mirrors fontTools'
  * _computeReachableRangesForAvar2, but computed from the original store at
  * plan time (the same quantity fontTools bounds via getExtremes on the
  * instanced store). Only constraining ranges (narrower than [-1, +1]) are
  * recorded. */
 static bool
 _compute_avar2_reachable_ranges (hb_subset_plan_t *plan,
                                  hb_array_t<const OT::AxisRecord> axes,
                                  const OT::avar *avar_table,
                                  bool detect_self_contained)
 {
   const OT::ItemVariationStore *var_store;
   const OT::DeltaSetIndexMap *varidx_map;
   if (!avar_table->get_v2_store_and_map (&var_store, &varidx_map))
     return false;

   /* Old-intermediate input box per axis. */
   hb_hashmap_t<hb_tag_t, hb_pair_t<double, double>> box;
   for (const auto &axis : axes)
   {
     hb_tag_t tag = axis.get_axis_tag ();
     double lo = -1.0, hi = +1.0;
     Triple *old_int;
     if (plan->user_axes_location.has (tag))
     {
       if (!plan->old_intermediates.has (tag, &old_int)) return false;
       lo = old_int->minimum;
       hi = old_int->maximum;
     }
     else if (axis.is_hidden ())
       lo = hi = 0.0; /* private axis: intermediate is always 0 */
     if (!box.set (tag, hb_pair (lo, hi))) return false;
   }

   hb_vector_t<hb_hashmap_t<hb_tag_t, Triple>> regions;
   if (!var_store->get_region_list ().get_var_regions (plan->axes_old_index_tag_map, regions))
     return false;

   hb_hashmap_t<hb_tag_t, unsigned> grid_pos;
   hb_vector_t<hb_vector_t<double>> grid_points;
   for (unsigned i = 0; i < axes.length; i++)
   {
     grid_pos.reset ();
     grid_points.reset ();

     hb_tag_t tag = axes[i].get_axis_tag ();
     hb_pair_t<double, double> *identity;
     if (!box.has (tag, &identity)) return false;

     /* Collect the row's active regions (non-zero delta). A varidx (or the
      * implicit identity mapping) that doesn't resolve to a real store row
      * behaves as "no variation" at runtime, so it contributes no regions;
      * this also covers a NULL VarStore. */
     hb_vector_t<hb_pair_t<int, unsigned>> active; /* (delta, region index) */
     uint32_t varidx = varidx_map->map (i);
     /* Runtime treats out-of-range delta-set indices as zero. */
     if (varidx != HB_OT_LAYOUT_NO_VARIATIONS_INDEX &&
	 !var_store->has_delta_set (varidx))
       varidx = HB_OT_LAYOUT_NO_VARIATIONS_INDEX;
     if (varidx != HB_OT_LAYOUT_NO_VARIATIONS_INDEX)
     {
       unsigned outer = varidx >> 16;
       unsigned inner = varidx & 0xFFFF;
       const OT::VarData &vd = var_store->get_sub_table (outer);
       const OT::HBUINT8 *delta_bytes = vd.get_delta_bytes ();
       unsigned row_size = vd.get_row_size ();
       unsigned region_count = vd.get_region_index_count ();
       for (unsigned r = 0; r < region_count; r++)
       {
	 int delta = vd.get_item_delta_fast (inner, r, delta_bytes, row_size);
	 if (!delta) continue;
	 unsigned region_idx = vd.get_region_index (r);
	 if (region_idx >= regions.length) return false;
	 active.push (hb_pair (delta, region_idx));
       }
       if (active.in_error ()) return false;
     }

     /* The delta is piecewise multilinear in the axis coordinates (each
      * region scalar is a product of independent per-axis tents), so over
      * the box its exact extremes are attained at vertices of the grid of
      * per-axis tent breakpoints. Enumerate that grid when it is small
      * (real fonts); otherwise fall back to per-region interval
      * arithmetic, which ignores correlations between regions and is
      * looser but still conservative. Mirrors fontTools'
      * VarStore.getExtremes. */
     constexpr unsigned MAX_GRID = 1u << 14;
     bool exact = true;
     double dmin = 0.0, dmax = 0.0; /* delta extremes, F2Dot14 units */
     double vmin = 0.0, vmax = 0.0; /* identity + delta extremes, coord units */

     /* Grid axes: the target axis (identity term), plus every axis a valid
      * tent of an active region references. */
     auto add_grid_axis = [&] (hb_tag_t t) -> int
     {
       unsigned *pos;
       if (grid_pos.has (t, &pos)) return (int) *pos;
       unsigned new_pos = grid_pos.get_population ();
       hb_pair_t<double, double> *input;
       double blo = -1.0, bhi = +1.0;
       if (box.has (t, &input)) { blo = input->first; bhi = input->second; }
       grid_points.push (hb_vector_t<double> ());
       if (grid_points.in_error ()) return -1;
       auto &points = grid_points[new_pos];
       points.push (blo);
       points.push (bhi);
       if (points.in_error () || !grid_pos.set (t, new_pos)) return -1;
       return (int) new_pos;
     };

     if (add_grid_axis (tag) < 0) return false;
     /* Per active region: (grid axis position, tent) for its valid tents. */
     hb_vector_t<hb_vector_t<hb_pair_t<unsigned, Triple>>> region_tents;
     for (const auto &ar : active)
     {
       region_tents.push (hb_vector_t<hb_pair_t<unsigned, Triple>> ());
       if (region_tents.in_error ()) return false;
       auto &tents = region_tents[region_tents.length - 1];
       for (const auto &_ : regions[ar.second])
       {
	 const Triple &tent = _.second;
	 /* Tents the runtime ignores (constant scalar 1) add no breakpoints. */
	 if (tent.middle == 0.0 ||
	     tent.minimum > tent.middle || tent.middle > tent.maximum ||
	     (tent.minimum < 0.0 && tent.maximum > 0.0))
	   continue;
	 int pos = add_grid_axis (_.first);
	 if (pos < 0) return false;
	 auto &points = grid_points[pos];
	 double blo = points[0], bhi = points[1];
	 points.push (hb_clamp (tent.minimum, blo, bhi));
	 points.push (hb_clamp (tent.middle, blo, bhi));
	 points.push (hb_clamp (tent.maximum, blo, bhi));
	 tents.push (hb_pair ((unsigned) pos, tent));
	 if (points.in_error () || tents.in_error ()) return false;
       }
     }

     unsigned grid_size = 1;
     for (auto &points : grid_points)
     {
       points.qsort ([] (const double &a, const double &b) -> int
		     { return a < b ? -1 : a > b ? +1 : 0; });
       unsigned n = 0;
       for (unsigned j = 0; j < points.length; j++)
	 if (!n || points.arrayZ[j] != points.arrayZ[n - 1])
	   points.arrayZ[n++] = points.arrayZ[j];
       points.shrink (n);
       if (grid_size > MAX_GRID / points.length) { exact = false; break; }
       grid_size *= points.length;
     }

     if (exact)
     {
       unsigned target_pos = 0; /* the target axis was added first */
       hb_vector_t<unsigned> odometer;
       if (!odometer.resize (grid_pos.get_population ())) return false;
       bool first = true;
       while (true)
       {
	 double delta_sum = 0.0;
	 for (unsigned r = 0; r < active.length; r++)
	 {
	   double scalar = 1.0;
	   for (const auto &pt : region_tents[r])
	   {
	     double v = grid_points[pt.first][odometer[pt.first]];
	     scalar *= _tent_eval (pt.second, v);
	     if (scalar == 0.0) break;
	   }
	   delta_sum += scalar * active[r].first;
	 }
	 double coord = grid_points[target_pos][odometer[target_pos]];
	 double value = coord + delta_sum / 16384.0;
	 if (first)
	 {
	   dmin = dmax = delta_sum;
	   vmin = vmax = value;
	   first = false;
	 }
	 else
	 {
	   dmin = hb_min (dmin, delta_sum); dmax = hb_max (dmax, delta_sum);
	   vmin = hb_min (vmin, value);     vmax = hb_max (vmax, value);
	 }
	 unsigned k = 0;
	 for (; k < odometer.length; k++)
	 {
	   if (++odometer[k] < grid_points[k].length) break;
	   odometer[k] = 0;
	 }
	 if (k == odometer.length) break;
       }
     }
     else
     {
       for (const auto &ar : active)
       {
	 int delta = ar.first;
	 double smin = 1.0, smax = 1.0;
	 for (const auto &_ : regions[ar.second])
	 {
	   hb_pair_t<double, double> *input;
	   double blo = -1.0, bhi = +1.0;
	   if (box.has (_.first, &input)) { blo = input->first; bhi = input->second; }
	   double tmin, tmax;
	   _tent_value_range (_.second, blo, bhi, &tmin, &tmax);
	   smin *= tmin;
	   smax *= tmax;
	   if (smax == 0.0) break;
	 }
	 if (delta > 0) { dmin += smin * delta; dmax += smax * delta; }
	 else           { dmin += smax * delta; dmax += smin * delta; }
       }
       vmin = identity->first + dmin / 16384.0;
       vmax = identity->second + dmax / 16384.0;
     }
     /* A pinned axis whose delta is constant over the box is self-contained:
      * its final coordinate is a constant. It can be removed from fvar/avar
      * and its contribution baked into the variation tables like an ordinary
      * pin at that coordinate (in old final space). Round the delta alone
      * and add it to the exact quantized intermediate, matching the runtime
      * (map_coords_2_14) and fontTools. */
     Triple *user;
     if (detect_self_contained && dmin == dmax &&
	 plan->user_axes_location.has (tag, &user) && user->is_point ())
     {
       int v_int = (int) roundf ((float) identity->first * 16384.f) +
		   (int) roundf ((float) dmin);
       v_int = hb_clamp (v_int, -(1 << 14), +(1 << 14));
       if (!plan->avar2_self_contained.set (tag, v_int / 16384.0)) return false;
       continue; /* axis is dropped; no region can reference it afterwards */
     }

     /* Pad by one F2Dot14 unit against rounding differences with the
      * runtime's fixed-point evaluation. */
     double lo = hb_clamp (vmin - 1.0 / 16384.0, -1.0, +1.0);
     double hi = hb_clamp (vmax + 1.0 / 16384.0, -1.0, +1.0);
     lo = floor (lo * 16384.0) / 16384.0;
     hi = ceil (hi * 16384.0) / 16384.0;
     if (lo <= -1.0 && hi >= +1.0)
       continue; /* not constraining */
     if (!plan->avar2_reachable_ranges.set (tag, Triple (lo, hb_clamp (0.0, lo, hi), hi)))
       return false;
   }
   return true;
 }

#ifndef HB_NO_OT_FONT_CFF
 static inline hb_font_t*
 _get_hb_font_with_variations (const hb_subset_plan_t *plan)
 {
   hb_font_t *font = hb_font_create (plan->source);

   if (plan->has_avar2)
   {
     /* Under avar2, instancing applies only to the self-contained pins,
      * whose constant final coordinates the plan holds in normalized_coords.
      * Setting user-space variations would run the full avar2 mapping and
      * bake contributions that remain live in the instance's variations. */
     hb_font_set_var_coords_normalized (font, plan->normalized_coords.arrayZ,
                                        plan->normalized_coords.length);
     return font;
   }

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
#endif

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

bool
normalize_axes_location (hb_face_t *face, hb_subset_plan_t *plan)
{
  if (plan->user_axes_location.is_empty ())
    return true;

  hb_array_t<const OT::AxisRecord> axes = face->table.fvar->get_axes ();
  if (!plan->check_success (plan->normalized_coords.resize (axes.length)))
    return false;

  bool has_avar = face->table.avar->has_data ();
  hb_vector_t<float> normalized_mins;
  hb_vector_t<float> normalized_defaults;
  hb_vector_t<float> normalized_maxs;
  if (has_avar)
  {
    if (!plan->check_success (normalized_mins.resize (axes.length)) ||
        !plan->check_success (normalized_defaults.resize (axes.length)) ||
        !plan->check_success (normalized_maxs.resize (axes.length)))
      return false;
  }

  bool axis_not_pinned = false;
  unsigned new_axis_idx = 0;
  unsigned last_idx = 0;
  for (const auto& _ : + hb_enumerate (axes))
  {
    unsigned i = _.first;
    const OT::AxisRecord &axis = _.second;
    hb_tag_t axis_tag = axis.get_axis_tag ();
    plan->axes_old_index_tag_map.set (i, axis_tag);

    if (!plan->user_axes_location.has (axis_tag) ||
        !plan->user_axes_location.get (axis_tag).is_point ())
    {
      axis_not_pinned = true;
      plan->axes_index_map.set (i, new_axis_idx);
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

      if (has_avar)
      {
        normalized_mins[i] = normalized_min;
        normalized_defaults[i] = normalized_default;
        normalized_maxs[i] = normalized_max;
        last_idx = i;
      }
      else
      {
        plan->axes_location.set (axis_tag, Triple ((double) normalized_min,
                                                   (double) normalized_default,
                                                   (double) normalized_max));
        if (normalized_default == -0.f)
          normalized_default = 0.f; // Normalize -0 to 0
        if (normalized_default != 0.f)
          plan->pinned_at_default = false;

        plan->normalized_coords[i] = roundf (normalized_default * 16384.f);
      }
    }
  }
  plan->all_axes_pinned = !axis_not_pinned;

  // TODO: use avar map_coords_16_16() when normalization is changed to 16.16
  // in fonttools
  if (has_avar)
  {
    const OT::avar* avar_table = face->table.avar;
    unsigned coords_len = last_idx + 1;

    if (avar_table->has_v2_data () && !plan->all_axes_pinned)
    {
      /* avar2 partial instancing: use v1-only mapping to get intermediate-space
       * coords. These are used for IVS rebasing and offset compensation. */
      plan->has_avar2 = true;

      if (!plan->check_success (avar_table->map_coords_2_14 (normalized_mins.arrayZ, coords_len, true)) ||
          !plan->check_success (avar_table->map_coords_2_14 (normalized_defaults.arrayZ, coords_len, true)) ||
          !plan->check_success (avar_table->map_coords_2_14 (normalized_maxs.arrayZ, coords_len, true)))
        return false;

      for (const auto& _ : + hb_enumerate (axes))
      {
        unsigned i = _.first;
        hb_tag_t axis_tag = _.second.get_axis_tag ();
        if (plan->user_axes_location.has (axis_tag))
        {
          /* Store intermediate-space coords for offset compensation,
           * segment-map renormalization, and IVS rebasing. */
          plan->old_intermediates.set (axis_tag, Triple ((double) normalized_mins[i],
                                                         (double) normalized_defaults[i],
                                                         (double) normalized_maxs[i]));
        }
      }

      /* Reachable-range computation also detects self-contained pinned
       * axes: pinned axes whose final coordinate is constant. Those are
       * removed from fvar, so suppress the detection when the face has
       * tables that cannot follow: CFF2 has no partial-pin instancing path
       * (its 'pinned' path flattens ALL blends; lift this once
       * https://github.com/harfbuzz/harfbuzz/pull/4710 lands, porting
       * fontTools' instantiateCFF2 from
       * https://github.com/fonttools/fonttools/pull/3506), and VARC passes
       * through verbatim with explicit fvar axis indices that renumbering
       * would desynchronize. Such axes stay in fvar as ordinary hidden
       * pins. */
      bool detect_self_contained = true;
      for (hb_tag_t table_tag : { HB_TAG ('C','F','F','2'), HB_TAG ('V','A','R','C') })
      {
        hb_blob_t *blob = hb_face_reference_table (face, table_tag);
        if (hb_blob_get_length (blob))
          detect_self_contained = false;
        hb_blob_destroy (blob);
      }
      if (!_compute_avar2_reachable_ranges (plan, axes, avar_table,
                                            detect_self_contained))
        return false;

      /* Keep all axes in fvar (pinned ones as hidden), EXCEPT self-contained
       * pinned axes, which are removed entirely. */
      plan->axes_index_map.reset ();
      plan->axis_tags.reset ();
      unsigned retained_axis_idx = 0;
      for (const auto& _ : + hb_enumerate (axes))
      {
        unsigned i = _.first;
        hb_tag_t axis_tag = _.second.get_axis_tag ();
        if (plan->avar2_self_contained.has (axis_tag))
          continue;
        plan->axes_index_map.set (i, retained_axis_idx++);
        plan->axis_tags.push (axis_tag);
      }
      /* The other tables see ONLY the self-contained pins, as an ordinary
       * partial instancing in old final-coordinate space; the standard
       * instancing machinery bakes those axes' contributions in. The
       * remaining restriction is carried entirely by avar. */
      plan->axes_location.reset ();
      if (plan->avar2_self_contained.get_population ())
      {
        bool sc_pinned_at_default = true;
        for (auto _ : plan->avar2_self_contained)
        {
          plan->axes_location.set (_.first, Triple (_.second, _.second, _.second));
          if (_.second != 0.0)
            sc_pinned_at_default = false;
        }
        for (const auto& _ : + hb_enumerate (axes))
        {
          double *v;
          plan->normalized_coords[_.first] =
              plan->avar2_self_contained.has (_.second.get_axis_tag (), &v)
              ? (int) roundf ((float) (*v * 16384.0)) : 0;
        }
        plan->pinned_at_default = sc_pinned_at_default;
      }
      else
        plan->normalized_coords.shrink (0);
    }
    else
    {
      /* Standard avar v1 (or v1+v2 with all axes pinned) */
      if (!plan->check_success (avar_table->map_coords_2_14 (normalized_mins.arrayZ, coords_len)) ||
          !plan->check_success (avar_table->map_coords_2_14 (normalized_defaults.arrayZ, coords_len)) ||
          !plan->check_success (avar_table->map_coords_2_14 (normalized_maxs.arrayZ, coords_len)))
        return false;

      for (const auto& _ : + hb_enumerate (axes))
      {
        unsigned i = _.first;
        hb_tag_t axis_tag = _.second.get_axis_tag ();
        if (plan->user_axes_location.has (axis_tag))
        {
          plan->axes_location.set (axis_tag, Triple ((double) normalized_mins[i],
                                                     (double) normalized_defaults[i],
                                                     (double) normalized_maxs[i]));
          float normalized_default = normalized_defaults[i];
          if (normalized_default == -0.f)
            normalized_default = 0.f;
          if (normalized_default != 0.f)
            plan->pinned_at_default = false;

          plan->normalized_coords[i] = roundf (normalized_default * 16384.f);
        }
      }
    }
  }
  return true;
}

#ifndef HB_NO_OT_FONT_CFF
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
      plan->head_maxp_info.xMax = hb_max (plan->head_maxp_info.xMax,
						 hb_saturate_add (extents.x_bearing, extents.width));
      plan->head_maxp_info.yMax = hb_max (plan->head_maxp_info.yMax, extents.y_bearing);
      plan->head_maxp_info.yMin = hb_min (plan->head_maxp_info.yMin,
						 hb_saturate_add (extents.y_bearing, extents.height));
    }

    if (_hmtx.has_data ())
    {
      double hori_aw = _hmtx.get_advance_without_var_unscaled (old_gid);
      if (_hmtx.var_table.get_length ())
        hori_aw += (double) roundf (_hmtx.var_table->get_advance_delta_unscaled (old_gid, font->coords, font->num_coords,
									 hvar_store_cache));
      /* Malicious deltas can take the advance out of the UFWORD range that
       * hmtx serialization stores.  Clamp, to keep downstream consumers
       * (e.g. CFF width optimizer) bounded and consistent with hmtx. */
      hori_aw = hb_clamp (hori_aw, 0., 65535.);
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
      double vert_aw = _vmtx.get_advance_without_var_unscaled (old_gid);
      if (_vmtx.var_table.get_length ())
        vert_aw += (double) roundf (_vmtx.var_table->get_advance_delta_unscaled (old_gid, font->coords, font->num_coords,
									 vvar_store_cache));
      vert_aw = hb_clamp (vert_aw, 0., 65535.);
      hb_position_t vorg_x = 0;
      hb_position_t vorg_y = 0;
      int tsb = 0;
      if (has_bounds_info &&
           hb_font_get_glyph_v_origin (font, old_gid, &vorg_x, &vorg_y))
      {
        tsb = hb_saturate_sub (vorg_y, extents.y_bearing);
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
#endif

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
