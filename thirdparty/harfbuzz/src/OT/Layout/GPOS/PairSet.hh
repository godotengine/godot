#ifndef OT_LAYOUT_GPOS_PAIRSET_HH
#define OT_LAYOUT_GPOS_PAIRSET_HH

#include "PairValueRecord.hh"

namespace OT {
namespace Layout {
namespace GPOS_impl {


template <typename Types>
struct PairSet
{
  template <typename Types2>
  friend struct PairPosFormat1_3;

  using PairValueRecord = GPOS_impl::PairValueRecord<Types>;

  protected:
  HBUINT16              len;    /* Number of PairValueRecords */
  PairValueRecord       firstPairValueRecord;
                                /* Array of PairValueRecords--ordered
                                 * by GlyphID of the second glyph */
  public:
  DEFINE_SIZE_MIN (2);

  static unsigned get_size (unsigned len1, unsigned len2)
  {
    return Types::HBGlyphID::static_size + Value::static_size * (len1 + len2);
  }
  static unsigned get_size (const ValueFormat valueFormats[2])
  {
    unsigned len1 = valueFormats[0].get_len ();
    unsigned len2 = valueFormats[1].get_len ();
    return get_size (len1, len2);
  }

  struct sanitize_closure_t
  {
    const ValueFormat *valueFormats;
    unsigned int len1; /* valueFormats[0].get_len() */
    unsigned int stride; /* bytes */
  };

  bool sanitize (hb_sanitize_context_t *c, const sanitize_closure_t *closure) const
  {
    TRACE_SANITIZE (this);
    if (!(c->check_struct (this)
       && c->check_range (&firstPairValueRecord,
                          len,
                          closure->stride))) return_trace (false);

    unsigned int count = len;
    const PairValueRecord *record = &firstPairValueRecord;
    return_trace (c->lazy_some_gpos ||
		  (closure->valueFormats[0].sanitize_values_stride_unsafe (c, this, &record->values[0], count, closure->stride) &&
                   closure->valueFormats[1].sanitize_values_stride_unsafe (c, this, &record->values[closure->len1], count, closure->stride)));
  }

  bool intersects (const hb_set_t *glyphs,
                   const ValueFormat *valueFormats) const
  {
    unsigned record_size = get_size (valueFormats);

    const PairValueRecord *record = &firstPairValueRecord;
    unsigned int count = len;
    for (unsigned int i = 0; i < count; i++)
    {
      if (glyphs->has (record->secondGlyph))
        return true;
      record = &StructAtOffset<const PairValueRecord> (record, record_size);
    }
    return false;
  }

  void collect_glyphs (hb_collect_glyphs_context_t *c,
                       const ValueFormat *valueFormats) const
  {
    unsigned record_size = get_size (valueFormats);

    const PairValueRecord *record = &firstPairValueRecord;
    c->input->add_array (&record->secondGlyph, len, record_size);
  }

  void collect_variation_indices (hb_collect_variation_indices_context_t *c,
                                  const ValueFormat *valueFormats) const
  {
    unsigned record_size = get_size (valueFormats);

    const PairValueRecord *record = &firstPairValueRecord;
    unsigned count = len;
    for (unsigned i = 0; i < count; i++)
    {
      if (c->glyph_set->has (record->secondGlyph))
      { record->collect_variation_indices (c, valueFormats, this); }

      record = &StructAtOffset<const PairValueRecord> (record, record_size);
    }
  }

  bool apply (hb_ot_apply_context_t *c,
              const ValueFormat *valueFormats,
              unsigned int pos) const
  {
    TRACE_APPLY (this);
    hb_buffer_t *buffer = c->buffer;
    unsigned int len1 = valueFormats[0].get_len ();
    unsigned int len2 = valueFormats[1].get_len ();
    unsigned record_size = get_size (len1, len2);

    const PairValueRecord *record = hb_bsearch (buffer->info[pos].codepoint,
                                                &firstPairValueRecord,
                                                len,
                                                record_size);
    if (record)
    {
      if (HB_BUFFER_MESSAGE_MORE && c->buffer->messaging ())
      {
	c->buffer->message (c->font,
			    "try kerning glyphs at %u,%u",
			    c->buffer->idx, pos);
      }

      bool applied_first = len1 && valueFormats[0].apply_value (c, this, &record->values[0], buffer->cur_pos());
      bool applied_second = len2 && valueFormats[1].apply_value (c, this, &record->values[len1], buffer->pos[pos]);

      if (applied_first || applied_second)
	if (HB_BUFFER_MESSAGE_MORE && c->buffer->messaging ())
	{
	  c->buffer->message (c->font,
			      "kerned glyphs at %u,%u",
			      c->buffer->idx, pos);
	}

      if (HB_BUFFER_MESSAGE_MORE && c->buffer->messaging ())
      {
	c->buffer->message (c->font,
			    "tried kerning glyphs at %u,%u",
			    c->buffer->idx, pos);
      }

      if (applied_first || applied_second)
        buffer->unsafe_to_break (buffer->idx, pos + 1);

      if (len2)
      {
	pos++;
      // https://github.com/harfbuzz/harfbuzz/issues/3824
      // https://github.com/harfbuzz/harfbuzz/issues/3888#issuecomment-1326781116
      buffer->unsafe_to_break (buffer->idx, pos + 1);
      }

      buffer->idx = pos;
      return_trace (true);
    }
    buffer->unsafe_to_concat (buffer->idx, pos + 1);
    return_trace (false);
  }

  bool subset (hb_subset_context_t *c,
               const ValueFormat valueFormats[2],
               const ValueFormat newFormats[2]) const
  {
    TRACE_SUBSET (this);
    auto snap = c->serializer->snapshot ();

    auto *out = c->serializer->start_embed (*this);
    if (unlikely (!c->serializer->extend_min (out))) return_trace (false);
    out->len = 0;

    const hb_set_t &glyphset = *c->plan->glyphset_gsub ();
    const hb_map_t &glyph_map = *c->plan->glyph_map;

    unsigned len1 = valueFormats[0].get_len ();
    unsigned len2 = valueFormats[1].get_len ();
    unsigned record_size = get_size (len1, len2);

    typename PairValueRecord::context_t context =
    {
      this,
      valueFormats,
      newFormats,
      len1,
      &glyph_map,
      &c->plan->layout_variation_idx_delta_map
    };

    const PairValueRecord *record = &firstPairValueRecord;
    unsigned count = len, num = 0;
    for (unsigned i = 0; i < count; i++)
    {
      if (glyphset.has (record->secondGlyph)
         && record->subset (c, &context)) num++;
      record = &StructAtOffset<const PairValueRecord> (record, record_size);
    }

    out->len = num;
    if (!num) c->serializer->revert (snap);
    return_trace (num);
  }
};


}
}
}

#endif  // OT_LAYOUT_GPOS_PAIRSET_HH
