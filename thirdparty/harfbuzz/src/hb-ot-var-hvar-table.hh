/*
 * Copyright © 2017  Google, Inc.
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
 * Google Author(s): Behdad Esfahbod
 */

#ifndef HB_OT_VAR_HVAR_TABLE_HH
#define HB_OT_VAR_HVAR_TABLE_HH

#include "hb-ot-layout-common.hh"
#include "hb-ot-var-common.hh"

namespace OT {


struct index_map_subset_plan_t
{
  enum index_map_index_t {
    ADV_INDEX,
    LSB_INDEX,	/* dual as TSB */
    RSB_INDEX,	/* dual as BSB */
    VORG_INDEX
  };

  void init (const DeltaSetIndexMap  &index_map,
	     hb_inc_bimap_t	     &outer_map,
	     hb_vector_t<hb_set_t *> &inner_sets,
	     const hb_subset_plan_t  *plan,
	     bool bypass_empty = true)
  {
    map_count = 0;
    outer_bit_count = 0;
    inner_bit_count = 1;
    max_inners.init ();
    output_map.init ();

    if (bypass_empty && !index_map.get_map_count ()) return;

    unsigned int	last_val = (unsigned int)-1;
    hb_codepoint_t	last_gid = HB_CODEPOINT_INVALID;

    outer_bit_count = (index_map.get_width () * 8) - index_map.get_inner_bit_count ();
    max_inners.resize (inner_sets.length);
    for (unsigned i = 0; i < inner_sets.length; i++) max_inners[i] = 0;

    /* Search backwards for a map value different from the last map value */
    auto &new_to_old_gid_list = plan->new_to_old_gid_list;
    unsigned count = new_to_old_gid_list.length;
    for (unsigned j = count; j; j--)
    {
      hb_codepoint_t gid = new_to_old_gid_list.arrayZ[j - 1].first;
      hb_codepoint_t old_gid = new_to_old_gid_list.arrayZ[j - 1].second;

      unsigned int v = index_map.map (old_gid);
      if (last_gid == HB_CODEPOINT_INVALID)
      {
	last_val = v;
	last_gid = gid;
	continue;
      }
      if (v != last_val)
	break;

      last_gid = gid;
    }

    if (unlikely (last_gid == (hb_codepoint_t)-1)) return;
    map_count = last_gid + 1;
    for (auto _ : plan->new_to_old_gid_list)
    {
      hb_codepoint_t gid = _.first;
      if (gid >= map_count) break;

      hb_codepoint_t old_gid = _.second;
      unsigned int v = index_map.map (old_gid);
      unsigned int outer = v >> 16;
      unsigned int inner = v & 0xFFFF;
      outer_map.add (outer);
      if (inner > max_inners[outer]) max_inners[outer] = inner;
      if (outer >= inner_sets.length) return;
      inner_sets[outer]->add (inner);
    }
  }

  void fini ()
  {
    max_inners.fini ();
    output_map.fini ();
  }

  void remap (const DeltaSetIndexMap *input_map,
	      const hb_inc_bimap_t &outer_map,
	      const hb_vector_t<hb_inc_bimap_t> &inner_maps,
	      const hb_subset_plan_t *plan)
  {
    for (unsigned int i = 0; i < max_inners.length; i++)
    {
      if (inner_maps[i].get_population () == 0) continue;
      unsigned int bit_count = (max_inners[i]==0)? 1: hb_bit_storage (inner_maps[i][max_inners[i]]);
      if (bit_count > inner_bit_count) inner_bit_count = bit_count;
    }

    if (unlikely (!output_map.resize (map_count))) return;
    for (const auto &_ : plan->new_to_old_gid_list)
    {
      hb_codepoint_t new_gid = _.first;
      hb_codepoint_t old_gid = _.second;

      if (unlikely (new_gid >= map_count)) break;

      uint32_t v = input_map->map (old_gid);
      unsigned int outer = v >> 16;
      output_map.arrayZ[new_gid] = (outer_map[outer] << 16) | (inner_maps[outer][v & 0xFFFF]);
    }
  }

  bool remap_after_instantiation (const hb_subset_plan_t *plan,
                                  const hb_map_t& varidx_map)
  {
    /* recalculate bit_count after remapping */
    outer_bit_count = 1;
    inner_bit_count = 1;

    for (const auto &_ : plan->new_to_old_gid_list)
    {
      hb_codepoint_t new_gid = _.first;
      if (unlikely (new_gid >= map_count)) break;

      uint32_t v = output_map.arrayZ[new_gid];
      uint32_t *new_varidx;
      if (!varidx_map.has (v, &new_varidx))
        return false;

      output_map.arrayZ[new_gid] = *new_varidx;

      unsigned outer = (*new_varidx) >> 16;
      unsigned bit_count = (outer == 0) ? 1 : hb_bit_storage (outer);
      outer_bit_count = hb_max (bit_count, outer_bit_count);
      
      unsigned inner = (*new_varidx) & 0xFFFF;
      bit_count = (inner == 0) ? 1 : hb_bit_storage (inner);
      inner_bit_count = hb_max (bit_count, inner_bit_count);
    }
    return true;
  }

  unsigned int get_inner_bit_count () const { return inner_bit_count; }
  unsigned int get_width ()           const { return ((outer_bit_count + inner_bit_count + 7) / 8); }
  unsigned int get_map_count ()       const { return map_count; }

  unsigned int get_size () const
  { return (map_count? (DeltaSetIndexMap::min_size + get_width () * map_count): 0); }

  bool is_identity () const { return get_output_map ().length == 0; }
  hb_array_t<const uint32_t> get_output_map () const { return output_map.as_array (); }

  protected:
  unsigned int map_count;
  hb_vector_t<unsigned int> max_inners;
  unsigned int outer_bit_count;
  unsigned int inner_bit_count;
  hb_vector_t<uint32_t> output_map;
};

struct hvarvvar_subset_plan_t
{
  hvarvvar_subset_plan_t() : inner_maps (), index_map_plans () {}
  ~hvarvvar_subset_plan_t() { fini (); }

  void init (const hb_array_t<const DeltaSetIndexMap *> &index_maps,
	     const VariationStore &_var_store,
	     const hb_subset_plan_t *plan)
  {
    index_map_plans.resize (index_maps.length);

    var_store = &_var_store;
    inner_sets.resize (var_store->get_sub_table_count ());
    for (unsigned int i = 0; i < inner_sets.length; i++)
      inner_sets[i] = hb_set_create ();
    adv_set = hb_set_create ();

    inner_maps.resize (var_store->get_sub_table_count ());

    if (unlikely (!index_map_plans.length || !inner_sets.length || !inner_maps.length)) return;

    bool retain_adv_map = false;
    index_map_plans[0].init (*index_maps[0], outer_map, inner_sets, plan, false);
    if (index_maps[0] == &Null (DeltaSetIndexMap))
    {
      retain_adv_map = plan->flags & HB_SUBSET_FLAGS_RETAIN_GIDS;
      outer_map.add (0);
      for (hb_codepoint_t old_gid : plan->glyphset()->iter())
        inner_sets[0]->add (old_gid);
      hb_set_union (adv_set, inner_sets[0]);
    }

    for (unsigned int i = 1; i < index_maps.length; i++)
      index_map_plans[i].init (*index_maps[i], outer_map, inner_sets, plan);

    outer_map.sort ();

    if (retain_adv_map)
    {
      for (const auto &_ : plan->new_to_old_gid_list)
      {
        hb_codepoint_t old_gid = _.second;
	inner_maps[0].add (old_gid);
      }
    }
    else
    {
      inner_maps[0].add_set (adv_set);
      hb_set_subtract (inner_sets[0], adv_set);
      inner_maps[0].add_set (inner_sets[0]);
    }

    for (unsigned int i = 1; i < inner_maps.length; i++)
      inner_maps[i].add_set (inner_sets[i]);

    for (unsigned int i = 0; i < index_maps.length; i++)
      index_map_plans[i].remap (index_maps[i], outer_map, inner_maps, plan);
  }

  /* remap */
  bool remap_index_map_plans (const hb_subset_plan_t *plan,
                              const hb_map_t& varidx_map)
  {
    for (unsigned i = 0; i < index_map_plans.length; i++)
      if (!index_map_plans[i].remap_after_instantiation (plan, varidx_map))
        return false;
    return true;
  }

  void fini ()
  {
    for (unsigned int i = 0; i < inner_sets.length; i++)
      hb_set_destroy (inner_sets[i]);
    hb_set_destroy (adv_set);
    inner_maps.fini ();
    index_map_plans.fini ();
  }

  hb_inc_bimap_t outer_map;
  hb_vector_t<hb_inc_bimap_t> inner_maps;
  hb_vector_t<index_map_subset_plan_t> index_map_plans;
  const VariationStore *var_store;

  protected:
  hb_vector_t<hb_set_t *> inner_sets;
  hb_set_t *adv_set;
};

/*
 * HVAR -- Horizontal Metrics Variations
 * https://docs.microsoft.com/en-us/typography/opentype/spec/hvar
 * VVAR -- Vertical Metrics Variations
 * https://docs.microsoft.com/en-us/typography/opentype/spec/vvar
 */
#define HB_OT_TAG_HVAR HB_TAG('H','V','A','R')
#define HB_OT_TAG_VVAR HB_TAG('V','V','A','R')

struct HVARVVAR
{
  static constexpr hb_tag_t HVARTag = HB_OT_TAG_HVAR;
  static constexpr hb_tag_t VVARTag = HB_OT_TAG_VVAR;

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (version.sanitize (c) &&
		  likely (version.major == 1) &&
		  varStore.sanitize (c, this) &&
		  advMap.sanitize (c, this) &&
		  lsbMap.sanitize (c, this) &&
		  rsbMap.sanitize (c, this));
  }

  const VariationStore& get_var_store () const
  { return this+varStore; }

  void listup_index_maps (hb_vector_t<const DeltaSetIndexMap *> &index_maps) const
  {
    index_maps.push (&(this+advMap));
    index_maps.push (&(this+lsbMap));
    index_maps.push (&(this+rsbMap));
  }

  bool serialize_index_maps (hb_serialize_context_t *c,
			     const hb_array_t<index_map_subset_plan_t> &im_plans)
  {
    TRACE_SERIALIZE (this);
    if (im_plans[index_map_subset_plan_t::ADV_INDEX].is_identity ())
      advMap = 0;
    else if (unlikely (!advMap.serialize_serialize (c, im_plans[index_map_subset_plan_t::ADV_INDEX])))
      return_trace (false);
    if (im_plans[index_map_subset_plan_t::LSB_INDEX].is_identity ())
      lsbMap = 0;
    else if (unlikely (!lsbMap.serialize_serialize (c, im_plans[index_map_subset_plan_t::LSB_INDEX])))
      return_trace (false);
    if (im_plans[index_map_subset_plan_t::RSB_INDEX].is_identity ())
      rsbMap = 0;
    else if (unlikely (!rsbMap.serialize_serialize (c, im_plans[index_map_subset_plan_t::RSB_INDEX])))
      return_trace (false);

    return_trace (true);
  }

  template <typename T>
  bool _subset (hb_subset_context_t *c) const
  {
    TRACE_SUBSET (this);
    if (c->plan->all_axes_pinned)
      return_trace (false);

    hvarvvar_subset_plan_t	hvar_plan;
    hb_vector_t<const DeltaSetIndexMap *>
				index_maps;

    ((T*)this)->listup_index_maps (index_maps);
    hvar_plan.init (index_maps.as_array (), this+varStore, c->plan);

    T *out = c->serializer->allocate_min<T> ();
    if (unlikely (!out)) return_trace (false);

    out->version.major = 1;
    out->version.minor = 0;

    if (c->plan->normalized_coords)
    {
      item_variations_t item_vars;
      if (!item_vars.instantiate (this+varStore, c->plan,
                                  advMap == 0 ? false : true,
                                  false, /* use_no_variation_idx = false */
                                  hvar_plan.inner_maps.as_array ()))
        return_trace (false);

      if (!out->varStore.serialize_serialize (c->serializer,
                                              item_vars.has_long_word (),
                                              c->plan->axis_tags,
                                              item_vars.get_region_list (),
                                              item_vars.get_vardata_encodings ()))
        return_trace (false);

      /* if varstore is optimized, remap output_map */
      if (advMap)
      {
        if (!hvar_plan.remap_index_map_plans (c->plan, item_vars.get_varidx_map ()))
          return_trace (false);
      }
    }
    else
    {
      if (unlikely (!out->varStore
		    .serialize_serialize (c->serializer,
					  hvar_plan.var_store,
					  hvar_plan.inner_maps.as_array ())))
      return_trace (false);
    }

    return_trace (out->T::serialize_index_maps (c->serializer,
						hvar_plan.index_map_plans.as_array ()));
  }

  float get_advance_delta_unscaled (hb_codepoint_t  glyph,
				    const int *coords, unsigned int coord_count,
				    VariationStore::cache_t *store_cache = nullptr) const
  {
    uint32_t varidx = (this+advMap).map (glyph);
    return (this+varStore).get_delta (varidx,
				      coords, coord_count,
				      store_cache);
  }

  bool get_lsb_delta_unscaled (hb_codepoint_t glyph,
			       const int *coords, unsigned int coord_count,
			       float *lsb) const
  {
    if (!lsbMap) return false;
    uint32_t varidx = (this+lsbMap).map (glyph);
    *lsb = (this+varStore).get_delta (varidx, coords, coord_count);
    return true;
  }

  public:
  FixedVersion<>version;	/* Version of the metrics variation table
				 * initially set to 0x00010000u */
  Offset32To<VariationStore>
		varStore;	/* Offset to item variation store table. */
  Offset32To<DeltaSetIndexMap>
		advMap;		/* Offset to advance var-idx mapping. */
  Offset32To<DeltaSetIndexMap>
		lsbMap;		/* Offset to lsb/tsb var-idx mapping. */
  Offset32To<DeltaSetIndexMap>
		rsbMap;		/* Offset to rsb/bsb var-idx mapping. */

  public:
  DEFINE_SIZE_STATIC (20);
};

struct HVAR : HVARVVAR {
  static constexpr hb_tag_t tableTag = HB_OT_TAG_HVAR;
  bool subset (hb_subset_context_t *c) const { return HVARVVAR::_subset<HVAR> (c); }
};
struct VVAR : HVARVVAR {
  static constexpr hb_tag_t tableTag = HB_OT_TAG_VVAR;

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (static_cast<const HVARVVAR *> (this)->sanitize (c) &&
		  vorgMap.sanitize (c, this));
  }

  void listup_index_maps (hb_vector_t<const DeltaSetIndexMap *> &index_maps) const
  {
    HVARVVAR::listup_index_maps (index_maps);
    index_maps.push (&(this+vorgMap));
  }

  bool serialize_index_maps (hb_serialize_context_t *c,
			     const hb_array_t<index_map_subset_plan_t> &im_plans)
  {
    TRACE_SERIALIZE (this);
    if (unlikely (!HVARVVAR::serialize_index_maps (c, im_plans)))
      return_trace (false);
    if (!im_plans[index_map_subset_plan_t::VORG_INDEX].get_map_count ())
      vorgMap = 0;
    else if (unlikely (!vorgMap.serialize_serialize (c, im_plans[index_map_subset_plan_t::VORG_INDEX])))
      return_trace (false);

    return_trace (true);
  }

  bool subset (hb_subset_context_t *c) const { return HVARVVAR::_subset<VVAR> (c); }

  bool get_vorg_delta_unscaled (hb_codepoint_t glyph,
				const int *coords, unsigned int coord_count,
				float *delta) const
  {
    if (!vorgMap) return false;
    uint32_t varidx = (this+vorgMap).map (glyph);
    *delta = (this+varStore).get_delta (varidx, coords, coord_count);
    return true;
  }

  protected:
  Offset32To<DeltaSetIndexMap>
		vorgMap;	/* Offset to vertical-origin var-idx mapping. */

  public:
  DEFINE_SIZE_STATIC (24);
};

} /* namespace OT */


#endif /* HB_OT_VAR_HVAR_TABLE_HH */
