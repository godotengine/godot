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

#ifndef HB_SUBSET_PLAN_HH
#define HB_SUBSET_PLAN_HH

#include "hb.hh"

#include "hb-subset.h"
#include "hb-subset-input.hh"
#include "hb-subset-accelerator.hh"

#include "hb-map.hh"
#include "hb-bimap.hh"
#include "hb-set.hh"

namespace OT {
struct Feature;
}

struct os2_info_t {
  hb_codepoint_t min_cmap_codepoint;
  hb_codepoint_t max_cmap_codepoint;
};

typedef struct os2_info_t os2_info_t;

struct head_maxp_info_t
{
  head_maxp_info_t ()
      :xMin (0x7FFF), xMax (-0x7FFF), yMin (0x7FFF), yMax (-0x7FFF),
      maxPoints (0), maxContours (0),
      maxCompositePoints (0),
      maxCompositeContours (0),
      maxComponentElements (0),
      maxComponentDepth (0),
      allXMinIsLsb (true) {}

  int xMin;
  int xMax;
  int yMin;
  int yMax;
  unsigned maxPoints;
  unsigned maxContours;
  unsigned maxCompositePoints;
  unsigned maxCompositeContours;
  unsigned maxComponentElements;
  unsigned maxComponentDepth;
  bool allXMinIsLsb;
};

typedef struct head_maxp_info_t head_maxp_info_t;

struct contour_point_t
{
  void init (float x_ = 0.f, float y_ = 0.f, bool is_end_point_ = false)
  { flag = 0; x = x_; y = y_; is_end_point = is_end_point_; }

  void transform (const float (&matrix)[4])
  {
    float x_ = x * matrix[0] + y * matrix[2];
	  y  = x * matrix[1] + y * matrix[3];
    x  = x_;
  }

  void add_delta (float delta_x, float delta_y)
  {
    x += delta_x;
    y += delta_y;
  }

  HB_ALWAYS_INLINE
  void translate (const contour_point_t &p) { x += p.x; y += p.y; }

  float x;
  float y;
  uint8_t flag;
  bool is_end_point;
};

struct contour_point_vector_t : hb_vector_t<contour_point_t>
{
  bool add_deltas (hb_array_t<const float> deltas_x,
                   hb_array_t<const float> deltas_y,
                   hb_array_t<const bool> indices)
  {
    if (indices.length != deltas_x.length ||
        indices.length != deltas_y.length)
      return false;

    for (unsigned i = 0; i < indices.length; i++)
    {
      if (!indices.arrayZ[i]) continue;
      arrayZ[i].add_delta (deltas_x.arrayZ[i], deltas_y.arrayZ[i]);
    }
    return true;
  }
};

namespace OT {
  struct cff1_subset_accelerator_t;
  struct cff2_subset_accelerator_t;
}

struct hb_subset_plan_t
{
  HB_INTERNAL hb_subset_plan_t (hb_face_t *,
				const hb_subset_input_t *input);

  HB_INTERNAL ~hb_subset_plan_t();

  hb_object_header_t header;

  bool successful;
  unsigned flags;
  bool attach_accelerator_data = false;
  bool force_long_loca = false;

  // The glyph subset
  hb_map_t *codepoint_to_glyph; // Needs to be heap-allocated

  // Old -> New glyph id mapping
  hb_map_t *glyph_map; // Needs to be heap-allocated
  hb_map_t *reverse_glyph_map; // Needs to be heap-allocated

  // Plan is only good for a specific source/dest so keep them with it
  hb_face_t *source;
#ifndef HB_NO_SUBSET_CFF
  // These have to be immediately after source:
  hb_face_lazy_loader_t<OT::cff1_subset_accelerator_t, 1> cff1_accel;
  hb_face_lazy_loader_t<OT::cff2_subset_accelerator_t, 2> cff2_accel;
#endif

  hb_face_t *dest;

  unsigned int _num_output_glyphs;

  bool all_axes_pinned;
  bool pinned_at_default;
  bool has_seac;

  // whether to insert a catch-all FeatureVariationRecord
  bool gsub_insert_catch_all_feature_variation_rec;
  bool gpos_insert_catch_all_feature_variation_rec;

  // whether GDEF ItemVariationStore is retained
  mutable bool has_gdef_varstore;

#define HB_SUBSET_PLAN_MEMBER(Type, Name) Type Name;
#include "hb-subset-plan-member-list.hh"
#undef HB_SUBSET_PLAN_MEMBER

  //recalculated head/maxp table info after instancing
  mutable head_maxp_info_t head_maxp_info;

  os2_info_t os2_info;

  const hb_subset_accelerator_t* accelerator;
  hb_subset_accelerator_t* inprogress_accelerator;

 public:

  template<typename T>
  struct source_table_loader
  {
    hb_blob_ptr_t<T> operator () (hb_subset_plan_t *plan)
    {
      hb_lock_t lock (plan->accelerator ? &plan->accelerator->sanitized_table_cache_lock : nullptr);

      auto *cache = plan->accelerator ? &plan->accelerator->sanitized_table_cache : &plan->sanitized_table_cache;
      if (cache
	  && !cache->in_error ()
	  && cache->has (+T::tableTag)) {
	return hb_blob_reference (cache->get (+T::tableTag).get ());
      }

      hb::unique_ptr<hb_blob_t> table_blob {hb_sanitize_context_t ().reference_table<T> (plan->source)};
      hb_blob_t* ret = hb_blob_reference (table_blob.get ());

      if (likely (cache))
	cache->set (+T::tableTag, std::move (table_blob));

      return ret;
    }
  };

  template<typename T>
  auto source_table() HB_AUTO_RETURN (source_table_loader<T> {} (this))

  bool in_error () const { return !successful; }

  bool check_success(bool success)
  {
    successful = (successful && success);
    return successful;
  }

  /*
   * The set of input glyph ids which will be retained in the subset.
   * Does NOT include ids kept due to retain_gids. You probably want to use
   * glyph_map/reverse_glyph_map.
   */
  inline const hb_set_t *
  glyphset () const
  {
    return &_glyphset;
  }

  /*
   * The set of input glyph ids which will be retained in the subset.
   */
  inline const hb_set_t *
  glyphset_gsub () const
  {
    return &_glyphset_gsub;
  }

  /*
   * The total number of output glyphs in the final subset.
   */
  inline unsigned int
  num_output_glyphs () const
  {
    return _num_output_glyphs;
  }

  inline bool new_gid_for_codepoint (hb_codepoint_t codepoint,
				     hb_codepoint_t *new_gid) const
  {
    hb_codepoint_t old_gid = codepoint_to_glyph->get (codepoint);
    if (old_gid == HB_MAP_VALUE_INVALID)
      return false;

    return new_gid_for_old_gid (old_gid, new_gid);
  }

  inline bool new_gid_for_old_gid (hb_codepoint_t old_gid,
				   hb_codepoint_t *new_gid) const
  {
    hb_codepoint_t gid = glyph_map->get (old_gid);
    if (gid == HB_MAP_VALUE_INVALID)
      return false;

    *new_gid = gid;
    return true;
  }

  inline bool old_gid_for_new_gid (hb_codepoint_t  new_gid,
				   hb_codepoint_t *old_gid) const
  {
    hb_codepoint_t gid = reverse_glyph_map->get (new_gid);
    if (gid == HB_MAP_VALUE_INVALID)
      return false;

    *old_gid = gid;
    return true;
  }

  inline bool
  add_table (hb_tag_t tag,
	     hb_blob_t *contents)
  {
    if (HB_DEBUG_SUBSET)
    {
      hb_blob_t *source_blob = source->reference_table (tag);
      DEBUG_MSG(SUBSET, nullptr, "add table %c%c%c%c, dest %u bytes, source %u bytes",
		HB_UNTAG(tag),
		hb_blob_get_length (contents),
		hb_blob_get_length (source_blob));
      hb_blob_destroy (source_blob);
    }
    return hb_face_builder_add_table (dest, tag, contents);
  }
};

// hb-subset-plan implementation is split into multiple files to keep
// compile times more reasonable:
// - hb-subset-plan.cc
// - hb-subset-plan-layout.cc
// - hb-subset-plan-var.cc
//
// The functions below are those needed to connect the split files
// above together.
HB_INTERNAL void
remap_indexes (const hb_set_t *indexes,
               hb_map_t       *mapping /* OUT */);


#ifndef HB_NO_VAR
template<typename ItemVarStore>
HB_INTERNAL void
remap_variation_indices (const ItemVarStore &var_store,
                         const hb_set_t &variation_indices,
                         const hb_vector_t<int>& normalized_coords,
                         bool calculate_delta, /* not pinned at default */
                         bool no_variations, /* all axes pinned */
                         hb_hashmap_t<unsigned, hb_pair_t<unsigned, int>> &variation_idx_delta_map /* OUT */);


template<typename DeltaSetIndexMap>
HB_INTERNAL void
remap_colrv1_delta_set_index_indices (const DeltaSetIndexMap &index_map,
                                      const hb_set_t &delta_set_idxes,
                                      hb_hashmap_t<unsigned, hb_pair_t<unsigned, int>> &variation_idx_delta_map, /* IN/OUT */
                                      hb_map_t &new_deltaset_idx_varidx_map /* OUT */);


HB_INTERNAL void
generate_varstore_inner_maps (const hb_set_t& varidx_set,
                              unsigned subtable_count,
                              hb_vector_t<hb_inc_bimap_t> &inner_maps /* OUT */);

HB_INTERNAL void
normalize_axes_location (hb_face_t *face, hb_subset_plan_t *plan);

HB_INTERNAL void
update_instance_metrics_map_from_cff2 (hb_subset_plan_t *plan);

HB_INTERNAL bool
get_instance_glyphs_contour_points (hb_subset_plan_t *plan);

#ifndef HB_NO_BASE
HB_INTERNAL void
collect_base_variation_indices (hb_subset_plan_t* plan);
#endif
#endif

#ifndef HB_NO_SUBSET_LAYOUT
typedef hb_hashmap_t<unsigned, hb::unique_ptr<hb_set_t>> script_langsys_map;

HB_INTERNAL void
remap_used_mark_sets (hb_subset_plan_t *plan,
                      hb_map_t& used_mark_sets_map);

HB_INTERNAL void
layout_nameid_closure (hb_subset_plan_t* plan,
                       hb_set_t* drop_tables);

HB_INTERNAL void
layout_populate_gids_to_retain (hb_subset_plan_t* plan,
                                hb_set_t* drop_tables);

HB_INTERNAL void
collect_layout_variation_indices (hb_subset_plan_t* plan);
#endif


#endif /* HB_SUBSET_PLAN_HH */
