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

#ifndef HB_OT_VAR_AVAR_TABLE_HH
#define HB_OT_VAR_AVAR_TABLE_HH

#include "hb-open-type.hh"
#include "hb-ot-var-common.hh"


/*
 * avar -- Axis Variations
 * https://docs.microsoft.com/en-us/typography/opentype/spec/avar
 */

#define HB_OT_TAG_avar HB_TAG('a','v','a','r')


namespace OT {


/* "Spec": https://github.com/be-fonts/boring-expansion-spec/issues/14 */
struct avarV2Tail
{
  friend struct avar;

  bool sanitize (hb_sanitize_context_t *c,
		 const void *base) const
  {
    TRACE_SANITIZE (this);
    return_trace (varIdxMap.sanitize (c, base) &&
		  varStore.sanitize (c, base));
  }

  protected:
  Offset32To<DeltaSetIndexMap>	varIdxMap;	/* Offset from the beginning of 'avar' table. */
  Offset32To<ItemVariationStore>	varStore;	/* Offset from the beginning of 'avar' table. */

  public:
  DEFINE_SIZE_STATIC (8);
};


struct AxisValueMap
{
  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this));
  }

  void set_mapping (float from_coord, float to_coord)
  {
    coords[0].set_float (from_coord);
    coords[1].set_float (to_coord);
  }

  bool is_outside_axis_range (const Triple& axis_range) const
  {
    double from_coord = (double) coords[0].to_float ();
    return !axis_range.contains (from_coord);
  }

  bool must_include () const
  {
    float from_coord = coords[0].to_float ();
    float to_coord = coords[1].to_float ();
    return (from_coord == -1.f && to_coord == -1.f) ||
           (from_coord == 0.f && to_coord == 0.f) ||
           (from_coord == 1.f && to_coord == 1.f);
  }

  void instantiate (const Triple& axis_range,
                    const Triple& unmapped_range,
                    const TripleDistances& triple_distances)
  {
    float from_coord = coords[0].to_float ();
    float to_coord = coords[1].to_float ();

    from_coord = renormalizeValue ((double) from_coord, unmapped_range, triple_distances);
    to_coord = renormalizeValue ((double) to_coord, axis_range, triple_distances);

    coords[0].set_float (from_coord);
    coords[1].set_float (to_coord);
  }

  HB_INTERNAL static int cmp (const void *pa, const void *pb)
  {
    const AxisValueMap *a = (const AxisValueMap *) pa;
    const AxisValueMap *b = (const AxisValueMap *) pb;

    int a_from = a->coords[0].to_int ();
    int b_from = b->coords[0].to_int ();
    if (a_from != b_from)
      return a_from - b_from;

    /* this should never be reached. according to the spec, all of the axis
     * value map records for a given axis must have different fromCoord values
     * */
    int a_to = a->coords[1].to_int ();
    int b_to = b->coords[1].to_int ();
    return a_to - b_to;
  }

  bool serialize (hb_serialize_context_t *c) const
  {
    TRACE_SERIALIZE (this);
    return_trace (c->embed (this));
  }

  public:
  F2DOT14	coords[2];
//   F2DOT14	fromCoord;	/* A normalized coordinate value obtained using
//				 * default normalization. */
//   F2DOT14	toCoord;	/* The modified, normalized coordinate value. */

  public:
  DEFINE_SIZE_STATIC (4);
};

struct SegmentMaps : Array16Of<AxisValueMap>
{
  int map (int value, unsigned int from_offset = 0, unsigned int to_offset = 1) const
  {
#define fromCoord coords[from_offset].to_int ()
#define toCoord coords[to_offset].to_int ()
    /* The following special-cases are not part of OpenType, which requires
     * that at least -1, 0, and +1 must be mapped. But we include these as
     * part of a better error recovery scheme. */
    if (len < 2)
    {
      if (!len)
	return value;
      else /* len == 1*/
	return value - arrayZ[0].fromCoord + arrayZ[0].toCoord;
    }

    if (value <= arrayZ[0].fromCoord)
      return value - arrayZ[0].fromCoord + arrayZ[0].toCoord;

    unsigned int i;
    unsigned int count = len - 1;
    for (i = 1; i < count && value > arrayZ[i].fromCoord; i++)
      ;

    if (value >= arrayZ[i].fromCoord)
      return value - arrayZ[i].fromCoord + arrayZ[i].toCoord;

    if (unlikely (arrayZ[i-1].fromCoord == arrayZ[i].fromCoord))
      return arrayZ[i-1].toCoord;

    int denom = arrayZ[i].fromCoord - arrayZ[i-1].fromCoord;
    return roundf (arrayZ[i-1].toCoord + ((float) (arrayZ[i].toCoord - arrayZ[i-1].toCoord) *
					  (value - arrayZ[i-1].fromCoord)) / denom);
#undef toCoord
#undef fromCoord
  }

  int unmap (int value) const { return map (value, 1, 0); }

  Triple unmap_axis_range (const Triple& axis_range) const
  {
    F2DOT14 val, unmapped_val;

    val.set_float (axis_range.minimum);
    unmapped_val.set_int (unmap (val.to_int ()));
    float unmapped_min = unmapped_val.to_float ();

    val.set_float (axis_range.middle);
    unmapped_val.set_int (unmap (val.to_int ()));
    float unmapped_middle = unmapped_val.to_float ();

    val.set_float (axis_range.maximum);
    unmapped_val.set_int (unmap (val.to_int ()));
    float unmapped_max = unmapped_val.to_float ();

    return Triple{(double) unmapped_min, (double) unmapped_middle, (double) unmapped_max};
  }

  bool subset (hb_subset_context_t *c, hb_tag_t axis_tag) const
  {
    TRACE_SUBSET (this);
    /* avar mapped normalized axis range*/
    Triple *axis_range;
    if (!c->plan->axes_location.has (axis_tag, &axis_range))
      return c->serializer->embed (*this);

    TripleDistances *axis_triple_distances;
    if (!c->plan->axes_triple_distances.has (axis_tag, &axis_triple_distances))
      return_trace (false);

    auto *out = c->serializer->start_embed (this);
    if (unlikely (!c->serializer->extend_min (out))) return_trace (false);

    Triple unmapped_range = unmap_axis_range (*axis_range);

    /* create a vector of retained mappings and sort */
    hb_vector_t<AxisValueMap> value_mappings;
    for (const auto& _ : as_array ())
    {
      if (_.is_outside_axis_range (unmapped_range))
        continue;
      AxisValueMap mapping;
      mapping = _;
      mapping.instantiate (*axis_range, unmapped_range, *axis_triple_distances);
      /* (-1, -1), (0, 0), (1, 1) mappings will be added later, so avoid
       * duplicates here */
      if (mapping.must_include ())
        continue;
      value_mappings.push (mapping);
    }

    AxisValueMap m;
    m.set_mapping (-1.f, -1.f);
    value_mappings.push (m);

    m.set_mapping (0.f, 0.f);
    value_mappings.push (m);

    m.set_mapping (1.f, 1.f);
    value_mappings.push (m);

    value_mappings.qsort ();

    for (const auto& _ : value_mappings)
    {
      if (!_.serialize (c->serializer))
        return_trace (false);
    }
    return_trace (c->serializer->check_assign (out->len, value_mappings.length, HB_SERIALIZE_ERROR_INT_OVERFLOW));
  }

  public:
  DEFINE_SIZE_ARRAY (2, *this);
};

struct avar
{
  static constexpr hb_tag_t tableTag = HB_OT_TAG_avar;

  bool has_data () const { return version.to_int (); }

  const SegmentMaps* get_segment_maps () const
  { return &firstAxisSegmentMaps; }

  unsigned get_axis_count () const
  { return axisCount; }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    if (!(version.sanitize (c) &&
	  hb_barrier () &&
	  (version.major == 1
#ifndef HB_NO_AVAR2
	   || version.major == 2
#endif
	   ) &&
	  c->check_struct (this)))
      return_trace (false);

    const SegmentMaps *map = &firstAxisSegmentMaps;
    unsigned int count = axisCount;
    for (unsigned int i = 0; i < count; i++)
    {
      if (unlikely (!map->sanitize (c)))
	return_trace (false);
      map = &StructAfter<SegmentMaps> (*map);
    }

#ifndef HB_NO_AVAR2
    if (version.major < 2)
      return_trace (true);
    hb_barrier ();

    const auto &v2 = * (const avarV2Tail *) map;
    if (unlikely (!v2.sanitize (c, this)))
      return_trace (false);
#endif

    return_trace (true);
  }

  void map_coords (int *coords, unsigned int coords_length) const
  {
    unsigned int count = hb_min (coords_length, axisCount);

    const SegmentMaps *map = &firstAxisSegmentMaps;
    for (unsigned int i = 0; i < count; i++)
    {
      coords[i] = map->map (coords[i]);
      map = &StructAfter<SegmentMaps> (*map);
    }

#ifndef HB_NO_AVAR2
    if (version.major < 2)
      return;
    hb_barrier ();

    for (; count < axisCount; count++)
      map = &StructAfter<SegmentMaps> (*map);

    const auto &v2 = * (const avarV2Tail *) map;

    const auto &varidx_map = this+v2.varIdxMap;
    const auto &var_store = this+v2.varStore;
    auto *var_store_cache = var_store.create_cache ();

    hb_vector_t<int> out;
    out.alloc (coords_length);
    for (unsigned i = 0; i < coords_length; i++)
    {
      int v = coords[i];
      uint32_t varidx = varidx_map.map (i);
      float delta = var_store.get_delta (varidx, coords, coords_length, var_store_cache);
      v += roundf (delta);
      v = hb_clamp (v, -(1<<14), +(1<<14));
      out.push (v);
    }
    for (unsigned i = 0; i < coords_length; i++)
      coords[i] = out[i];

    OT::ItemVariationStore::destroy_cache (var_store_cache);
#endif
  }

  void unmap_coords (int *coords, unsigned int coords_length) const
  {
    unsigned int count = hb_min (coords_length, axisCount);

    const SegmentMaps *map = &firstAxisSegmentMaps;
    for (unsigned int i = 0; i < count; i++)
    {
      coords[i] = map->unmap (coords[i]);
      map = &StructAfter<SegmentMaps> (*map);
    }
  }

  bool subset (hb_subset_context_t *c) const
  {
    TRACE_SUBSET (this);
    unsigned retained_axis_count = c->plan->axes_index_map.get_population ();
    if (!retained_axis_count) //all axes are pinned/dropped
      return_trace (false);

    avar *out = c->serializer->allocate_min<avar> ();
    if (unlikely (!out)) return_trace (false);

    out->version.major = 1;
    out->version.minor = 0;
    if (!c->serializer->check_assign (out->axisCount, retained_axis_count, HB_SERIALIZE_ERROR_INT_OVERFLOW))
      return_trace (false);

    const hb_map_t& axes_index_map = c->plan->axes_index_map;
    const SegmentMaps *map = &firstAxisSegmentMaps;
    unsigned count = axisCount;
    for (unsigned int i = 0; i < count; i++)
    {
      if (axes_index_map.has (i))
      {
        hb_tag_t *axis_tag;
        if (!c->plan->axes_old_index_tag_map.has (i, &axis_tag))
          return_trace (false);
        if (!map->subset (c, *axis_tag))
          return_trace (false);
      }
      map = &StructAfter<SegmentMaps> (*map);
    }
    return_trace (true);
  }

  protected:
  FixedVersion<>version;	/* Version of the avar table
				 * initially set to 0x00010000u */
  HBUINT16	reserved;	/* This field is permanently reserved. Set to 0. */
  HBUINT16	axisCount;	/* The number of variation axes in the font. This
				 * must be the same number as axisCount in the
				 * 'fvar' table. */
  SegmentMaps	firstAxisSegmentMaps;

  public:
  DEFINE_SIZE_MIN (8);
};

} /* namespace OT */


#endif /* HB_OT_VAR_AVAR_TABLE_HH */
