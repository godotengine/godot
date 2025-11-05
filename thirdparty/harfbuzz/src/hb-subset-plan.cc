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
#include "hb-multimap.hh"
#include "hb-set.hh"
#include "hb-subset.h"
#include "hb-unicode.h"

#include "hb-ot-cmap-table.hh"
#include "hb-ot-glyf-table.hh"
#include "hb-ot-layout-base-table.hh"
#include "hb-ot-cff1-table.hh"
#include "hb-ot-cff2-table.hh"
#include "OT/Color/COLR/COLR.hh"
#include "OT/Color/COLR/colrv1-closure.hh"
#include "OT/Color/CPAL/CPAL.hh"
#include "hb-ot-var-fvar-table.hh"
#include "hb-ot-stat-table.hh"
#include "hb-ot-math-table.hh"

hb_subset_accelerator_t::~hb_subset_accelerator_t ()
{
  if (cmap_cache && destroy_cmap_cache)
    destroy_cmap_cache ((void*) cmap_cache);

#ifndef HB_NO_SUBSET_CFF
  cff1_accel.fini ();
  cff2_accel.fini ();
#endif
  hb_face_destroy (source);
}


#ifndef HB_NO_SUBSET_CFF
static inline bool
_add_cff_seac_components (const OT::cff1::accelerator_subset_t &cff,
			  hb_codepoint_t gid,
			  hb_set_t *gids_to_retain)
{
  hb_codepoint_t base_gid, accent_gid;
  if (cff.get_seac_components (gid, &base_gid, &accent_gid))
  {
    gids_to_retain->add (base_gid);
    gids_to_retain->add (accent_gid);
    return true;
  }
  return false;
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

void
remap_indexes (const hb_set_t *indexes,
               hb_map_t       *mapping /* OUT */)
{
  for (auto _ : + hb_enumerate (indexes->iter ()))
    mapping->set (_.second, _.first);
}

static inline void
_cmap_closure (hb_face_t	   *face,
	       const hb_set_t	   *unicodes,
	       hb_set_t		   *glyphset)
{
  OT::cmap::accelerator_t cmap (face);
  cmap.table->closure_glyphs (unicodes, glyphset);
}

static void _colr_closure (hb_subset_plan_t* plan,
                           hb_set_t *glyphs_colred)
{
  OT::COLR::accelerator_t colr (plan->source);
  if (!colr.is_valid ()) return;

  hb_set_t palette_indices, layer_indices;
  // Collect all glyphs referenced by COLRv0
  hb_set_t glyphset_colrv0;
  for (hb_codepoint_t gid : *glyphs_colred)
    colr.closure_glyphs (gid, &glyphset_colrv0);

  glyphs_colred->union_ (glyphset_colrv0);

  //closure for COLRv1
  hb_set_t variation_indices, delta_set_indices;
  colr.closure_forV1 (glyphs_colred, &layer_indices, &palette_indices, &variation_indices, &delta_set_indices);

  colr.closure_V0palette_indices (glyphs_colred, &palette_indices);
  remap_indexes (&layer_indices, &plan->colrv1_layers);
  _remap_palette_indexes (&palette_indices, &plan->colr_palettes);

#ifndef HB_NO_VAR
  if (!colr.has_var_store () || !variation_indices) return;

  const OT::ItemVariationStore &var_store = colr.get_var_store ();
  // generated inner_maps is used by ItemVariationStore serialize(), which is subset only
  unsigned subtable_count = var_store.get_sub_table_count ();
  generate_varstore_inner_maps (variation_indices, subtable_count, plan->colrv1_varstore_inner_maps);

  /* colr variation indices mapping during planning phase:
   * generate colrv1_variation_idx_delta_map. When delta set index map is not
   * included, it's a mapping from varIdx-> (new varIdx,delta). Otherwise, it's
   * a mapping from old delta set idx-> (new delta set idx, delta). Mapping
   * delta set indices is the same as gid mapping.
   * Besides, we need to generate a delta set idx-> new var_idx map for updating
   * delta set index map if exists. This map will be updated again after
   * instancing. */
  if (!plan->all_axes_pinned)
  {
    remap_variation_indices (var_store,
                              variation_indices,
                              plan->normalized_coords,
                              false, /* no need to calculate delta for COLR during planning */
                              plan->all_axes_pinned,
                              plan->colrv1_variation_idx_delta_map);

    if (colr.has_delta_set_index_map ())
      remap_colrv1_delta_set_index_indices (colr.get_delta_set_index_map (),
                                             delta_set_indices,
                                             plan->colrv1_variation_idx_delta_map,
                                             plan->colrv1_new_deltaset_idx_varidx_map);
  }
#endif
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

template<bool GID_ALWAYS_EXISTS = false, typename I, typename F, typename G, hb_requires (hb_is_iterator (I))>
static void
_fill_unicode_and_glyph_map(hb_subset_plan_t *plan,
                            I unicode_iterator,
                            F unicode_to_gid_for_iterator,
                            G unicode_to_gid_general)
{
  for (hb_codepoint_t cp : unicode_iterator)
  {
    hb_codepoint_t gid = unicode_to_gid_for_iterator(cp);
    if (!GID_ALWAYS_EXISTS && gid == HB_MAP_VALUE_INVALID)
    {
      DEBUG_MSG(SUBSET, nullptr, "Drop U+%04X; no gid", cp);
      continue;
    }

    plan->codepoint_to_glyph->set (cp, gid);
    plan->unicode_to_new_gid_list.push (hb_pair (cp, gid));
  }
}

template<bool GID_ALWAYS_EXISTS = false, typename I, typename F, hb_requires (hb_is_iterator (I))>
static void
_fill_unicode_and_glyph_map(hb_subset_plan_t *plan,
                            I unicode_iterator,
                            F unicode_to_gid_for_iterator)
{
  _fill_unicode_and_glyph_map(plan, unicode_iterator, unicode_to_gid_for_iterator, unicode_to_gid_for_iterator);
}

/*
 * Finds additional unicode codepoints which are reachable from the input unicode set.
 * Currently this adds in mirrored variants (needed for bidi) of any input unicodes.
 */
static hb_set_t
_unicode_closure (const hb_set_t* unicodes, bool bidi_closure) {
  // TODO: we may want to also consider pulling in reachable unicode composition and decompositions.
  //       see: https://github.com/harfbuzz/harfbuzz/issues/2283
  hb_set_t out = *unicodes;
  if (!bidi_closure) return out;

  if (out.is_inverted()) {
    // don't closure inverted sets, they are asking to specifically exclude certain codepoints.
    // otherwise everything is already included.
    return out;
  }

  auto unicode_funcs = hb_unicode_funcs_get_default ();
  for (hb_codepoint_t cp : *unicodes) {
   hb_codepoint_t mirror = hb_unicode_mirroring(unicode_funcs, cp);
   if (unlikely (mirror != cp)) {
     out.add(mirror);
   }
  }

  return out;
}

static void
_populate_unicodes_to_retain (const hb_set_t *unicodes_in,
                              const hb_set_t *glyphs,
                              hb_subset_plan_t *plan)
{
  hb_set_t unicodes = _unicode_closure(unicodes_in,
    !(plan->flags & HB_SUBSET_FLAGS_NO_BIDI_CLOSURE));

  OT::cmap::accelerator_t cmap (plan->source);
  unsigned size_threshold = plan->source->get_num_glyphs ();

  if (glyphs->is_empty () && unicodes.get_population () < size_threshold)
  {

    const hb_map_t* unicode_to_gid = nullptr;
    if (plan->accelerator)
      unicode_to_gid = &plan->accelerator->unicode_to_gid;

    // This is approach to collection is faster, but can only be used  if glyphs
    // are not being explicitly added to the subset and the input unicodes set is
    // not excessively large (eg. an inverted set).
    plan->unicode_to_new_gid_list.alloc (unicodes.get_population ());
    if (!unicode_to_gid) {
      _fill_unicode_and_glyph_map(plan, unicodes.iter(), [&] (hb_codepoint_t cp) {
        hb_codepoint_t gid;
        if (!cmap.get_nominal_glyph (cp, &gid)) {
          return HB_MAP_VALUE_INVALID;
        }
        return gid;
      });
    } else {
      // Use in memory unicode to gid map it's faster then looking up from
      // the map. This code is mostly duplicated from above to avoid doing
      // conditionals on the presence of the unicode_to_gid map each
      // iteration.
      _fill_unicode_and_glyph_map(plan, unicodes.iter(), [&] (hb_codepoint_t cp) {
        return unicode_to_gid->get (cp);
      });
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
      plan->unicode_to_new_gid_list.alloc (hb_min(unicodes.get_population ()
                                                  + glyphs->get_population (),
                                                  cmap_unicodes->get_population ()));
    } else {
      unicode_glyphid_map = &plan->accelerator->unicode_to_gid;
      cmap_unicodes = &plan->accelerator->unicodes;
    }

    if (plan->accelerator &&
	unicodes.get_population () < cmap_unicodes->get_population () &&
	glyphs->get_population () < cmap_unicodes->get_population ())
    {
      plan->codepoint_to_glyph->alloc (unicodes.get_population () + glyphs->get_population ());

      auto &gid_to_unicodes = plan->accelerator->gid_to_unicodes;

      for (hb_codepoint_t gid : *glyphs)
      {
        auto unicodes = gid_to_unicodes.get (gid);
        _fill_unicode_and_glyph_map<true>(plan, unicodes, [&] (hb_codepoint_t cp) {
          return gid;
        },
        [&] (hb_codepoint_t cp) {
          return unicode_glyphid_map->get(cp);
        });
      }

      _fill_unicode_and_glyph_map(plan, unicodes.iter(), [&] (hb_codepoint_t cp) {
          /* Don't double-add entry. */
	if (plan->codepoint_to_glyph->has (cp))
          return HB_MAP_VALUE_INVALID;

        return unicode_glyphid_map->get(cp);
      },
      [&] (hb_codepoint_t cp) {
          return unicode_glyphid_map->get(cp);
      });

      plan->unicode_to_new_gid_list.qsort ();
    }
    else
    {
      plan->codepoint_to_glyph->alloc (cmap_unicodes->get_population ());
      hb_codepoint_t first = HB_SET_VALUE_INVALID, last = HB_SET_VALUE_INVALID;
      for (; cmap_unicodes->next_range (&first, &last); )
      {
        _fill_unicode_and_glyph_map(plan, hb_range(first, last + 1), [&] (hb_codepoint_t cp) {
          hb_codepoint_t gid = (*unicode_glyphid_map)[cp];
	  if (!unicodes.has (cp) && !glyphs->has (gid))
	    return HB_MAP_VALUE_INVALID;
          return gid;
        },
        [&] (hb_codepoint_t cp) {
          return unicode_glyphid_map->get(cp);
        });
      }
    }

    /* Add gids which where requested, but not mapped in cmap */
    unsigned num_glyphs = plan->source->get_num_glyphs ();
    hb_codepoint_t first = HB_SET_VALUE_INVALID, last = HB_SET_VALUE_INVALID;
    for (; glyphs->next_range (&first, &last); )
    {
      if (first >= num_glyphs)
	break;
      if (last >= num_glyphs)
        last = num_glyphs - 1;
      plan->_glyphset_gsub.add_range (first, last);
    }
  }

  auto &arr = plan->unicode_to_new_gid_list;
  if (arr.length)
  {
    plan->unicodes.add_sorted_array (&arr.arrayZ->first, arr.length, sizeof (*arr.arrayZ));
    plan->_glyphset_gsub.add_array (&arr.arrayZ->second, arr.length, sizeof (*arr.arrayZ));
  }

  // Variation selectors don't have glyphs associated with them in the cmap so they will have been filtered out above
  // but should still be retained. Add them back here.

  // However, the min and max codepoints for OS/2 should be calculated without considering variation selectors,
  // so record those first.
  plan->os2_info.min_cmap_codepoint = plan->unicodes.get_min();
  plan->os2_info.max_cmap_codepoint = plan->unicodes.get_max();

  hb_set_t variation_selectors_to_retain;
  cmap.collect_variation_selectors(&variation_selectors_to_retain);
  + variation_selectors_to_retain.iter()
  | hb_filter(unicodes)
  | hb_sink(&plan->unicodes)
  ;
}

static unsigned
_glyf_add_gid_and_children (const OT::glyf_accelerator_t &glyf,
			    hb_codepoint_t gid,
			    hb_set_t *gids_to_retain,
			    int operation_count,
			    unsigned depth = 0)
{
  /* Check if is already visited */
  if (gids_to_retain->has (gid)) return operation_count;

  gids_to_retain->add (gid);

  if (unlikely (depth++ > HB_MAX_NESTING_LEVEL)) return operation_count;
  if (unlikely (--operation_count < 0)) return operation_count;

  auto glyph = glyf.glyph_for_gid (gid);

  for (auto &item : glyph.get_composite_iterator ())
    operation_count =
      _glyf_add_gid_and_children (glyf,
				  item.get_gid (),
				  gids_to_retain,
				  operation_count,
				  depth);

  return operation_count;
}

static void
_nameid_closure (hb_subset_plan_t* plan,
		 hb_set_t* drop_tables)
{
#ifndef HB_NO_STYLE
  if (!drop_tables->has (HB_OT_TAG_STAT))
    plan->source->table.STAT->collect_name_ids (&plan->user_axes_location, &plan->name_ids);
#endif
#ifndef HB_NO_VAR
  if (!plan->all_axes_pinned)
    plan->source->table.fvar->collect_name_ids (&plan->user_axes_location, &plan->axes_old_index_tag_map, &plan->name_ids);
#endif
#ifndef HB_NO_COLOR
  if (!drop_tables->has (HB_OT_TAG_CPAL))
    plan->source->table.CPAL->collect_name_ids (&plan->colr_palettes, &plan->name_ids);
#endif

#ifndef HB_NO_SUBSET_LAYOUT
  layout_nameid_closure(plan, drop_tables);
#endif
}

static void
_populate_gids_to_retain (hb_subset_plan_t* plan,
		          hb_set_t* drop_tables)
{
  OT::glyf_accelerator_t glyf (plan->source);
#ifndef HB_NO_SUBSET_CFF
  // Note: we cannot use inprogress_accelerator here, since it has not been
  // created yet. So in case of preprocessed-face (and otherwise), we do an
  // extra sanitize pass here, which is not ideal.
  OT::cff1::accelerator_subset_t stack_cff (plan->accelerator ? nullptr : plan->source);
  const OT::cff1::accelerator_subset_t *cff (plan->accelerator ? plan->accelerator->cff1_accel.get () : &stack_cff);
#endif

  plan->_glyphset_gsub.add (0); // Not-def

  _cmap_closure (plan->source, &plan->unicodes, &plan->_glyphset_gsub);

#ifndef HB_NO_SUBSET_LAYOUT
  layout_populate_gids_to_retain(plan, drop_tables);
#endif

  _remove_invalid_gids (&plan->_glyphset_gsub, plan->source->get_num_glyphs ());

  plan->_glyphset_mathed = plan->_glyphset_gsub;
  if (!drop_tables->has (HB_OT_TAG_MATH))
  {
    _math_closure (plan, &plan->_glyphset_mathed);
    _remove_invalid_gids (&plan->_glyphset_mathed, plan->source->get_num_glyphs ());
  }

  hb_set_t cur_glyphset = plan->_glyphset_mathed;
  if (!drop_tables->has (HB_OT_TAG_COLR))
  {
    _colr_closure (plan, &cur_glyphset);
    _remove_invalid_gids (&cur_glyphset, plan->source->get_num_glyphs ());
  }

  plan->_glyphset_colred = cur_glyphset;

  // XXX TODO VARC closure / subset

  _nameid_closure (plan, drop_tables);
  /* Populate a full set of glyphs to retain by adding all referenced
   * composite glyphs. */
  if (glyf.has_data ())
    for (hb_codepoint_t gid : cur_glyphset)
      _glyf_add_gid_and_children (glyf, gid, &plan->_glyphset,
				  cur_glyphset.get_population () * HB_MAX_COMPOSITE_OPERATIONS_PER_GLYPH);
  else
    plan->_glyphset.union_ (cur_glyphset);
#ifndef HB_NO_SUBSET_CFF
  if (!plan->accelerator || plan->accelerator->has_seac)
  {
    bool has_seac = false;
    if (cff->is_valid ())
      for (hb_codepoint_t gid : cur_glyphset)
	if (_add_cff_seac_components (*cff, gid, &plan->_glyphset))
	  has_seac = true;
    plan->has_seac = has_seac;
  }
#endif

  _remove_invalid_gids (&plan->_glyphset, plan->source->get_num_glyphs ());

#ifndef HB_NO_VAR
#ifndef HB_NO_SUBSET_LAYOUT
  if (!drop_tables->has (HB_OT_TAG_GDEF))
    collect_layout_variation_indices (plan);
#endif
#endif
}

static void
_create_glyph_map_gsub (const hb_set_t* glyph_set_gsub,
                        const hb_map_t* glyph_map,
                        hb_map_t* out)
{
  out->alloc (glyph_set_gsub->get_population ());
  + hb_iter (glyph_set_gsub)
  | hb_map ([&] (hb_codepoint_t gid) {
    return hb_codepoint_pair_t (gid, glyph_map->get (gid));
  })
  | hb_sink (out)
  ;
}

static bool
_create_old_gid_to_new_gid_map (const hb_face_t *face,
				bool		 retain_gids,
				const hb_set_t	*all_gids_to_retain,
                                const hb_map_t  *requested_glyph_map,
				hb_map_t	*glyph_map, /* OUT */
				hb_map_t	*reverse_glyph_map, /* OUT */
				hb_sorted_vector_t<hb_codepoint_pair_t> *new_to_old_gid_list /* OUT */,
				unsigned int	*num_glyphs /* OUT */)
{
  unsigned pop = all_gids_to_retain->get_population ();
  reverse_glyph_map->alloc (pop);
  glyph_map->alloc (pop);
  new_to_old_gid_list->alloc (pop);

  if (*requested_glyph_map)
  {
    hb_set_t new_gids(requested_glyph_map->values());
    if (new_gids.get_population() != requested_glyph_map->get_population())
    {
      DEBUG_MSG (SUBSET, nullptr, "The provided custom glyph mapping is not unique.");
      return false;
    }

    if (retain_gids)
    {
      DEBUG_MSG (SUBSET, nullptr,
        "HB_SUBSET_FLAGS_RETAIN_GIDS cannot be set if "
        "a custom glyph mapping has been provided.");
      return false;
    }

    hb_codepoint_t max_glyph = 0;
    hb_set_t remaining;
    for (auto old_gid : all_gids_to_retain->iter ())
    {
      if (old_gid == 0) {
	new_to_old_gid_list->push (hb_pair<hb_codepoint_t, hb_codepoint_t> (0u, 0u));
        continue;
      }

      hb_codepoint_t* new_gid;
      if (!requested_glyph_map->has (old_gid, &new_gid))
      {
        remaining.add(old_gid);
        continue;
      }

      if (*new_gid > max_glyph)
        max_glyph = *new_gid;
      new_to_old_gid_list->push (hb_pair (*new_gid, old_gid));
    }
    new_to_old_gid_list->qsort ();

    // Anything that wasn't mapped by the requested mapping should
    // be placed after the requested mapping.
    for (auto old_gid : remaining)
      new_to_old_gid_list->push (hb_pair (++max_glyph, old_gid));

    *num_glyphs = max_glyph + 1;
  }
  else if (!retain_gids)
  {
    + hb_enumerate (hb_iter (all_gids_to_retain), (hb_codepoint_t) 0)
    | hb_sink (new_to_old_gid_list)
    ;
    *num_glyphs = new_to_old_gid_list->length;
  }
  else
  {
    + hb_iter (all_gids_to_retain)
    | hb_map ([] (hb_codepoint_t _) {
		return hb_codepoint_pair_t (_, _);
	      })
    | hb_sink (new_to_old_gid_list)
    ;

    hb_codepoint_t max_glyph = HB_SET_VALUE_INVALID;
    hb_set_previous (all_gids_to_retain, &max_glyph);

    *num_glyphs = max_glyph + 1;
  }

  reverse_glyph_map->alloc (reverse_glyph_map->get_population () + new_to_old_gid_list->length);
  + hb_iter (new_to_old_gid_list)
  | hb_sink (reverse_glyph_map)
  ;
  glyph_map->alloc (glyph_map->get_population () + new_to_old_gid_list->length);
  + hb_iter (new_to_old_gid_list)
  | hb_map (&hb_codepoint_pair_t::reverse)
  | hb_sink (glyph_map)
  ;

  return true;
}

hb_subset_plan_t::hb_subset_plan_t (hb_face_t *face,
				    const hb_subset_input_t *input)
{
  successful = true;
  flags = input->flags;

  unicode_to_new_gid_list.init ();

  name_ids = *input->sets.name_ids;
  name_languages = *input->sets.name_languages;
  layout_features = *input->sets.layout_features;
  layout_scripts = *input->sets.layout_scripts;
  glyphs_requested = *input->sets.glyphs;
  drop_tables = *input->sets.drop_tables;
  no_subset_tables = *input->sets.no_subset_tables;
  source = hb_face_reference (face);
  dest = hb_face_builder_create ();

  codepoint_to_glyph = hb_map_create ();
  glyph_map = hb_map_create ();
  reverse_glyph_map = hb_map_create ();

  gsub_insert_catch_all_feature_variation_rec = false;
  gpos_insert_catch_all_feature_variation_rec = false;
  gdef_varstore_inner_maps.init ();

  user_axes_location = input->axes_location;
  all_axes_pinned = false;
  pinned_at_default = true;
  has_gdef_varstore = false;

#ifdef HB_EXPERIMENTAL_API
  for (auto _ : input->name_table_overrides)
  {
    hb_bytes_t name_bytes = _.second;
    unsigned len = name_bytes.length;
    char *name_str = (char *) hb_malloc (len);
    if (unlikely (!check_success (name_str)))
      break;

    hb_memcpy (name_str, name_bytes.arrayZ, len);
    name_table_overrides.set (_.first, hb_bytes_t (name_str, len));
  }
#endif

  void* accel = hb_face_get_user_data(face, hb_subset_accelerator_t::user_data_key());

  attach_accelerator_data = input->attach_accelerator_data;
  force_long_loca = input->force_long_loca;
#ifdef HB_EXPERIMENTAL_API
  force_long_loca = force_long_loca || (flags & HB_SUBSET_FLAGS_IFTB_REQUIREMENTS);
#endif

  if (accel)
    accelerator = (hb_subset_accelerator_t*) accel;

  if (unlikely (in_error ()))
    return;

#ifndef HB_NO_VAR
  normalize_axes_location (face, this);
#endif

  _populate_unicodes_to_retain (input->sets.unicodes, input->sets.glyphs, this);

  _populate_gids_to_retain (this, input->sets.drop_tables);
  if (unlikely (in_error ()))
    return;

  if (!check_success(_create_old_gid_to_new_gid_map(
          face,
          input->flags & HB_SUBSET_FLAGS_RETAIN_GIDS,
          &_glyphset,
          &input->glyph_map,
          glyph_map,
          reverse_glyph_map,
	  &new_to_old_gid_list,
          &_num_output_glyphs))) {
    return;
  }

#ifdef HB_EXPERIMENTAL_API  
  if ((input->flags & HB_SUBSET_FLAGS_RETAIN_GIDS) &&
      (input->flags & HB_SUBSET_FLAGS_RETAIN_NUM_GLYPHS)) {
    // We've been requested to maintain the num glyphs count from the
    // input face.
    _num_output_glyphs = source->get_num_glyphs ();
  }
#endif

  _create_glyph_map_gsub (
      &_glyphset_gsub,
      glyph_map,
      &glyph_map_gsub);

  // Now that we have old to new gid map update the unicode to new gid list.
  for (unsigned i = 0; i < unicode_to_new_gid_list.length; i++)
  {
    // Use raw array access for performance.
    unicode_to_new_gid_list.arrayZ[i].second =
        glyph_map->get(unicode_to_new_gid_list.arrayZ[i].second);
  }

  bounds_width_vec.resize_dirty  (_num_output_glyphs);
  for (auto &v : bounds_width_vec)
    v = 0xFFFFFFFF;
  bounds_height_vec.resize_dirty  (_num_output_glyphs);
  for (auto &v : bounds_height_vec)
    v = 0xFFFFFFFF;

#ifndef HB_NO_SUBSET_LAYOUT    
  if (!drop_tables.has (HB_OT_TAG_GDEF))
    remap_used_mark_sets (this, used_mark_sets_map);
#endif

#ifndef HB_NO_VAR
#ifndef HB_NO_BASE
  if (!drop_tables.has (HB_OT_TAG_BASE))
    collect_base_variation_indices (this);
#endif
#endif

  if (unlikely (in_error ()))
    return;

#ifndef HB_NO_VAR
  update_instance_metrics_map_from_cff2 (this);
  if (!check_success (get_instance_glyphs_contour_points (this)))
      return;
#endif

  if (attach_accelerator_data)
  {
    inprogress_accelerator =
      hb_subset_accelerator_t::create (source,
				       *codepoint_to_glyph,
                                       unicodes,
				       has_seac);

    check_success (inprogress_accelerator);
  }

#define HB_SUBSET_PLAN_MEMBER(Type, Name) check_success (!Name.in_error ());
#include "hb-subset-plan-member-list.hh"
#undef HB_SUBSET_PLAN_MEMBER
}

hb_subset_plan_t::~hb_subset_plan_t()
{
  hb_face_destroy (dest);

  hb_map_destroy (codepoint_to_glyph);
  hb_map_destroy (glyph_map);
  hb_map_destroy (reverse_glyph_map);
#ifndef HB_NO_SUBSET_CFF
  cff1_accel.fini ();
  cff2_accel.fini ();
#endif
  hb_face_destroy (source);

#ifdef HB_EXPERIMENTAL_API
  for (auto _ : name_table_overrides.iter_ref ())
    _.second.fini ();
#endif

  if (inprogress_accelerator)
    hb_subset_accelerator_t::destroy ((void*) inprogress_accelerator);
}


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
  if (unlikely (!(plan = hb_object_create<hb_subset_plan_t> (face, input))))
    return nullptr;

  if (unlikely (plan->in_error ()))
  {
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
hb_map_t *
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
hb_map_t *
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
hb_map_t *
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
