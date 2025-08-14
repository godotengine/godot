/*
 * Copyright © 2018  Google, Inc.
 * Copyright © 2023  Behdad Esfahbod
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

#ifndef HB_SUBSET_PLAN_MEMBER_LIST_HH
#define HB_SUBSET_PLAN_MEMBER_LIST_HH
#endif /* HB_SUBSET_PLAN_MEMBER_LIST_HH */ /* Dummy header guards */

#define E(x, y) x, y

// For each cp that we'd like to retain maps to the corresponding gid.
HB_SUBSET_PLAN_MEMBER (hb_set_t, unicodes)
HB_SUBSET_PLAN_MEMBER (hb_sorted_vector_t<hb_codepoint_pair_t>, unicode_to_new_gid_list)

HB_SUBSET_PLAN_MEMBER (hb_sorted_vector_t<hb_codepoint_pair_t>, new_to_old_gid_list)

// name_ids we would like to retain
HB_SUBSET_PLAN_MEMBER (hb_set_t, name_ids)

// name_languages we would like to retain
HB_SUBSET_PLAN_MEMBER (hb_set_t, name_languages)

//layout features which will be preserved
HB_SUBSET_PLAN_MEMBER (hb_set_t, layout_features)

// layout scripts which will be preserved.
HB_SUBSET_PLAN_MEMBER (hb_set_t, layout_scripts)

//glyph ids requested to retain
HB_SUBSET_PLAN_MEMBER (hb_set_t, glyphs_requested)

// Tables which should not be processed, just pass them through.
HB_SUBSET_PLAN_MEMBER (hb_set_t, no_subset_tables)

// Tables which should be dropped.
HB_SUBSET_PLAN_MEMBER (hb_set_t, drop_tables)

// Old -> New glyph id mapping
HB_SUBSET_PLAN_MEMBER (hb_map_t, glyph_map_gsub)

HB_SUBSET_PLAN_MEMBER (hb_set_t, _glyphset)
HB_SUBSET_PLAN_MEMBER (hb_set_t, _glyphset_gsub)
HB_SUBSET_PLAN_MEMBER (hb_set_t, _glyphset_mathed)
HB_SUBSET_PLAN_MEMBER (hb_set_t, _glyphset_colred)

//active lookups we'd like to retain
HB_SUBSET_PLAN_MEMBER (hb_map_t, gsub_lookups)
HB_SUBSET_PLAN_MEMBER (hb_map_t, gpos_lookups)

//use_mark_sets mapping: old->new
HB_SUBSET_PLAN_MEMBER (hb_map_t, used_mark_sets_map)

//active langsys we'd like to retain
HB_SUBSET_PLAN_MEMBER (hb_hashmap_t E(<unsigned, hb::unique_ptr<hb_set_t>>), gsub_langsys)
HB_SUBSET_PLAN_MEMBER (hb_hashmap_t E(<unsigned, hb::unique_ptr<hb_set_t>>), gpos_langsys)

//active features after removing redundant langsys and prune_features
HB_SUBSET_PLAN_MEMBER (hb_map_t, gsub_features)
HB_SUBSET_PLAN_MEMBER (hb_map_t, gpos_features)

//active feature variation records/condition index with variations
HB_SUBSET_PLAN_MEMBER (hb_hashmap_t E(<unsigned, hb::shared_ptr<hb_set_t>>), gsub_feature_record_cond_idx_map)
HB_SUBSET_PLAN_MEMBER (hb_hashmap_t E(<unsigned, hb::shared_ptr<hb_set_t>>), gpos_feature_record_cond_idx_map)

//feature index-> address of substituation feature table mapping with
//variations
HB_SUBSET_PLAN_MEMBER (hb_hashmap_t E(<unsigned, const OT::Feature*>), gsub_feature_substitutes_map)
HB_SUBSET_PLAN_MEMBER (hb_hashmap_t E(<unsigned, const OT::Feature*>), gpos_feature_substitutes_map)

// old feature_indexes set, used to reinstate the old features
HB_SUBSET_PLAN_MEMBER (hb_set_t, gsub_old_features)
HB_SUBSET_PLAN_MEMBER (hb_set_t, gpos_old_features)

//feature_index->pair of (address of old feature, feature tag), used for inserting a catch all record
//if necessary
HB_SUBSET_PLAN_MEMBER (hb_hashmap_t E(<unsigned, hb_pair_t E(<const void*, const void*>)>), gsub_old_feature_idx_tag_map)
HB_SUBSET_PLAN_MEMBER (hb_hashmap_t E(<unsigned, hb_pair_t E(<const void*, const void*>)>), gpos_old_feature_idx_tag_map)

//active layers/palettes we'd like to retain
HB_SUBSET_PLAN_MEMBER (hb_map_t, colrv1_layers)
HB_SUBSET_PLAN_MEMBER (hb_map_t, colr_palettes)
//colrv1 varstore retained varidx mapping
HB_SUBSET_PLAN_MEMBER (hb_vector_t<hb_inc_bimap_t>, colrv1_varstore_inner_maps)
//colrv1 retained varidx -> (new varidx, delta) mapping
HB_SUBSET_PLAN_MEMBER (mutable hb_hashmap_t E(<unsigned, hb_pair_t E(<unsigned, int>)>), colrv1_variation_idx_delta_map)
//colrv1 retained new delta set index -> new varidx mapping
HB_SUBSET_PLAN_MEMBER (hb_map_t, colrv1_new_deltaset_idx_varidx_map)

//Old layout item variation index -> (New varidx, delta) mapping
HB_SUBSET_PLAN_MEMBER (mutable hb_hashmap_t E(<unsigned, hb_pair_t E(<unsigned, int>)>), layout_variation_idx_delta_map)

//gdef varstore retained varidx mapping
HB_SUBSET_PLAN_MEMBER (hb_vector_t<hb_inc_bimap_t>, gdef_varstore_inner_maps)

HB_SUBSET_PLAN_MEMBER (hb_hashmap_t E(<hb_tag_t, hb::unique_ptr<hb_blob_t>>), sanitized_table_cache)

//normalized axes range map
HB_SUBSET_PLAN_MEMBER (hb_hashmap_t E(<hb_tag_t, Triple>), axes_location)
HB_SUBSET_PLAN_MEMBER (hb_vector_t<int>, normalized_coords)

//user specified axes range map
HB_SUBSET_PLAN_MEMBER (hb_hashmap_t E(<hb_tag_t, Triple>), user_axes_location)
//axis->TripleDistances map (distances in the pre-normalized space)
HB_SUBSET_PLAN_MEMBER (hb_hashmap_t E(<hb_tag_t, TripleDistances>), axes_triple_distances)

//retained old axis index -> new axis index mapping in fvar axis array
HB_SUBSET_PLAN_MEMBER (hb_map_t, axes_index_map)

//axis_index->axis_tag mapping in fvar axis array
HB_SUBSET_PLAN_MEMBER (hb_map_t, axes_old_index_tag_map)
//vector of retained axis tags in the order of axes given in the 'fvar' table
HB_SUBSET_PLAN_MEMBER (hb_vector_t<hb_tag_t>, axis_tags)

//hmtx metrics map: new gid->(advance, lsb)
HB_SUBSET_PLAN_MEMBER (mutable hb_hashmap_t E(<hb_codepoint_t, hb_pair_t E(<unsigned, int>)>), hmtx_map)
//vmtx metrics map: new gid->(advance, lsb)
HB_SUBSET_PLAN_MEMBER (mutable hb_hashmap_t E(<hb_codepoint_t, hb_pair_t E(<unsigned, int>)>), vmtx_map)
//boundsWidth map: new gid->boundsWidth, boundWidth=xMax - xMin
HB_SUBSET_PLAN_MEMBER (mutable hb_vector_t<unsigned>, bounds_width_vec)
//boundsHeight map: new gid->boundsHeight, boundsHeight=yMax - yMin
HB_SUBSET_PLAN_MEMBER (mutable hb_vector_t<unsigned>, bounds_height_vec)

//map: new_gid -> contour points vector
HB_SUBSET_PLAN_MEMBER (mutable hb_hashmap_t E(<hb_codepoint_t, contour_point_vector_t>), new_gid_contour_points_map)

//new gids set for composite glyphs
HB_SUBSET_PLAN_MEMBER (hb_set_t, composite_new_gids)

//Old BASE item variation index -> (New varidx, 0) mapping
HB_SUBSET_PLAN_MEMBER (mutable hb_hashmap_t E(<unsigned, hb_pair_t E(<unsigned, int>)>), base_variation_idx_map)

//BASE table varstore retained varidx mapping
HB_SUBSET_PLAN_MEMBER (hb_vector_t<hb_inc_bimap_t>, base_varstore_inner_maps)

#ifdef HB_EXPERIMENTAL_API
// name table overrides map: hb_ot_name_record_ids_t-> name string new value or
// None to indicate should remove
HB_SUBSET_PLAN_MEMBER (hb_hashmap_t E(<hb_ot_name_record_ids_t, hb_bytes_t>), name_table_overrides)
#endif

#undef E
