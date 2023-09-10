#ifndef OT_LAYOUT_GPOS_PAIRVALUERECORD_HH
#define OT_LAYOUT_GPOS_PAIRVALUERECORD_HH

#include "ValueFormat.hh"

namespace OT {
namespace Layout {
namespace GPOS_impl {


template <typename Types>
struct PairValueRecord
{
  template <typename Types2>
  friend struct PairSet;

  protected:
  typename Types::HBGlyphID
	        secondGlyph;            /* GlyphID of second glyph in the
                                         * pair--first glyph is listed in the
                                         * Coverage table */
  ValueRecord   values;                 /* Positioning data for the first glyph
                                         * followed by for second glyph */
  public:
  DEFINE_SIZE_ARRAY (Types::HBGlyphID::static_size, values);

  int cmp (hb_codepoint_t k) const
  { return secondGlyph.cmp (k); }

  struct context_t
  {
    const void          *base;
    const ValueFormat   *valueFormats;
    const ValueFormat   *newFormats;
    unsigned            len1; /* valueFormats[0].get_len() */
    const hb_map_t      *glyph_map;
    const hb_hashmap_t<unsigned, hb_pair_t<unsigned, int>> *layout_variation_idx_delta_map;
  };

  bool subset (hb_subset_context_t *c,
               context_t *closure) const
  {
    TRACE_SERIALIZE (this);
    auto *s = c->serializer;
    auto *out = s->start_embed (*this);
    if (unlikely (!s->extend_min (out))) return_trace (false);

    out->secondGlyph = (*closure->glyph_map)[secondGlyph];

    closure->valueFormats[0].copy_values (s,
                                          closure->newFormats[0],
                                          closure->base, &values[0],
                                          closure->layout_variation_idx_delta_map);
    closure->valueFormats[1].copy_values (s,
                                          closure->newFormats[1],
                                          closure->base,
                                          &values[closure->len1],
                                          closure->layout_variation_idx_delta_map);

    return_trace (true);
  }

  void collect_variation_indices (hb_collect_variation_indices_context_t *c,
                                  const ValueFormat *valueFormats,
                                  const void *base) const
  {
    unsigned record1_len = valueFormats[0].get_len ();
    unsigned record2_len = valueFormats[1].get_len ();
    const hb_array_t<const Value> values_array = values.as_array (record1_len + record2_len);

    if (valueFormats[0].has_device ())
      valueFormats[0].collect_variation_indices (c, base, values_array.sub_array (0, record1_len));

    if (valueFormats[1].has_device ())
      valueFormats[1].collect_variation_indices (c, base, values_array.sub_array (record1_len, record2_len));
  }

  bool intersects (const hb_set_t& glyphset) const
  {
    return glyphset.has(secondGlyph);
  }

  const Value* get_values_1 () const
  {
    return &values[0];
  }

  const Value* get_values_2 (ValueFormat format1) const
  {
    return &values[format1.get_len ()];
  }
};


}
}
}

#endif  // OT_LAYOUT_GPOS_PAIRVALUERECORD_HH
