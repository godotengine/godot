#ifndef OT_LAYOUT_GPOS_PAIRPOSFORMAT1_HH
#define OT_LAYOUT_GPOS_PAIRPOSFORMAT1_HH

#include "PairSet.hh"

namespace OT {
namespace Layout {
namespace GPOS_impl {


template <typename Types>
struct PairPosFormat1_3
{
  using PairSet = GPOS_impl::PairSet<Types>;
  using PairValueRecord = GPOS_impl::PairValueRecord<Types>;

  protected:
  HBUINT16      format;                 /* Format identifier--format = 1 */
  typename Types::template OffsetTo<Coverage>
                coverage;               /* Offset to Coverage table--from
                                         * beginning of subtable */
  ValueFormat   valueFormat[2];         /* [0] Defines the types of data in
                                         * ValueRecord1--for the first glyph
                                         * in the pair--may be zero (0) */
                                        /* [1] Defines the types of data in
                                         * ValueRecord2--for the second glyph
                                         * in the pair--may be zero (0) */
  Array16Of<typename Types::template OffsetTo<PairSet>>
                pairSet;                /* Array of PairSet tables
                                         * ordered by Coverage Index */
  public:
  DEFINE_SIZE_ARRAY (8 + Types::size, pairSet);

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);

    if (!c->check_struct (this)) return_trace (false);

    unsigned int len1 = valueFormat[0].get_len ();
    unsigned int len2 = valueFormat[1].get_len ();
    typename PairSet::sanitize_closure_t closure =
    {
      valueFormat,
      len1,
      1 + len1 + len2
    };

    return_trace (coverage.sanitize (c, this) && pairSet.sanitize (c, this, &closure));
  }

  bool intersects (const hb_set_t *glyphs) const
  {
    return
    + hb_zip (this+coverage, pairSet)
    | hb_filter (*glyphs, hb_first)
    | hb_map (hb_second)
    | hb_map ([glyphs, this] (const typename Types::template OffsetTo<PairSet> &_)
              { return (this+_).intersects (glyphs, valueFormat); })
    | hb_any
    ;
  }

  void closure_lookups (hb_closure_lookups_context_t *c) const {}
  void collect_variation_indices (hb_collect_variation_indices_context_t *c) const
  {
    if ((!valueFormat[0].has_device ()) && (!valueFormat[1].has_device ())) return;

    auto it =
    + hb_zip (this+coverage, pairSet)
    | hb_filter (c->glyph_set, hb_first)
    | hb_map (hb_second)
    ;

    if (!it) return;
    + it
    | hb_map (hb_add (this))
    | hb_apply ([&] (const PairSet& _) { _.collect_variation_indices (c, valueFormat); })
    ;
  }

  void collect_glyphs (hb_collect_glyphs_context_t *c) const
  {
    if (unlikely (!(this+coverage).collect_coverage (c->input))) return;
    unsigned int count = pairSet.len;
    for (unsigned int i = 0; i < count; i++)
      (this+pairSet[i]).collect_glyphs (c, valueFormat);
  }

  const Coverage &get_coverage () const { return this+coverage; }

  bool apply (hb_ot_apply_context_t *c) const
  {
    TRACE_APPLY (this);
    hb_buffer_t *buffer = c->buffer;
    unsigned int index = (this+coverage).get_coverage  (buffer->cur().codepoint);
    if (likely (index == NOT_COVERED)) return_trace (false);

    hb_ot_apply_context_t::skipping_iterator_t &skippy_iter = c->iter_input;
    skippy_iter.reset (buffer->idx, 1);
    unsigned unsafe_to;
    if (!skippy_iter.next (&unsafe_to))
    {
      buffer->unsafe_to_concat (buffer->idx, unsafe_to);
      return_trace (false);
    }

    return_trace ((this+pairSet[index]).apply (c, valueFormat, skippy_iter.idx));
  }

  bool subset (hb_subset_context_t *c) const
  {
    TRACE_SUBSET (this);

    const hb_set_t &glyphset = *c->plan->glyphset_gsub ();
    const hb_map_t &glyph_map = *c->plan->glyph_map;

    auto *out = c->serializer->start_embed (*this);
    if (unlikely (!c->serializer->extend_min (out))) return_trace (false);
    out->format = format;
    out->valueFormat[0] = valueFormat[0];
    out->valueFormat[1] = valueFormat[1];
    if (c->plan->flags & HB_SUBSET_FLAGS_NO_HINTING)
    {
      hb_pair_t<unsigned, unsigned> newFormats = compute_effective_value_formats (glyphset);
      out->valueFormat[0] = newFormats.first;
      out->valueFormat[1] = newFormats.second;
    }

    hb_sorted_vector_t<hb_codepoint_t> new_coverage;

    + hb_zip (this+coverage, pairSet)
    | hb_filter (glyphset, hb_first)
    | hb_filter ([this, c, out] (const typename Types::template OffsetTo<PairSet>& _)
                 {
                   auto snap = c->serializer->snapshot ();
                   auto *o = out->pairSet.serialize_append (c->serializer);
                   if (unlikely (!o)) return false;
                   bool ret = o->serialize_subset (c, _, this, valueFormat, out->valueFormat);
                   if (!ret)
                   {
                     out->pairSet.pop ();
                     c->serializer->revert (snap);
                   }
                   return ret;
                 },
                 hb_second)
    | hb_map (hb_first)
    | hb_map (glyph_map)
    | hb_sink (new_coverage)
    ;

    out->coverage.serialize_serialize (c->serializer, new_coverage.iter ());

    return_trace (bool (new_coverage));
  }


  hb_pair_t<unsigned, unsigned> compute_effective_value_formats (const hb_set_t& glyphset) const
  {
    unsigned len1 = valueFormat[0].get_len ();
    unsigned len2 = valueFormat[1].get_len ();
    unsigned record_size = HBUINT16::static_size + Value::static_size * (len1 + len2);

    unsigned format1 = 0;
    unsigned format2 = 0;
    for (const auto & _ :
             + hb_zip (this+coverage, pairSet) | hb_filter (glyphset, hb_first) | hb_map (hb_second))
    {
      const PairSet& set = (this + _);
      const PairValueRecord *record = &set.firstPairValueRecord;

      for (unsigned i = 0; i < set.len; i++)
      {
        if (record->intersects (glyphset))
        {
          format1 = format1 | valueFormat[0].get_effective_format (record->get_values_1 ());
          format2 = format2 | valueFormat[1].get_effective_format (record->get_values_2 (valueFormat[0]));
        }
        record = &StructAtOffset<const PairValueRecord> (record, record_size);
      }
    }

    return hb_pair (format1, format2);
  }
};


}
}
}

#endif  // OT_LAYOUT_GPOS_PAIRPOSFORMAT1_HH
