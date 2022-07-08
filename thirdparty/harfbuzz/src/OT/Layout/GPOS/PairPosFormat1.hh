#ifndef OT_LAYOUT_GPOS_PAIRPOSFORMAT1_HH
#define OT_LAYOUT_GPOS_PAIRPOSFORMAT1_HH

namespace OT {
namespace Layout {
namespace GPOS_impl {

struct PairValueRecord
{
  friend struct PairSet;

  int cmp (hb_codepoint_t k) const
  { return secondGlyph.cmp (k); }

  struct context_t
  {
    const void          *base;
    const ValueFormat   *valueFormats;
    const ValueFormat   *newFormats;
    unsigned            len1; /* valueFormats[0].get_len() */
    const hb_map_t      *glyph_map;
    const hb_map_t      *layout_variation_idx_map;
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
                                          closure->layout_variation_idx_map);
    closure->valueFormats[1].copy_values (s,
                                          closure->newFormats[1],
                                          closure->base,
                                          &values[closure->len1],
                                          closure->layout_variation_idx_map);

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

  protected:
  HBGlyphID16   secondGlyph;            /* GlyphID of second glyph in the
                                         * pair--first glyph is listed in the
                                         * Coverage table */
  ValueRecord   values;                 /* Positioning data for the first glyph
                                         * followed by for second glyph */
  public:
  DEFINE_SIZE_ARRAY (2, values);
};

struct PairSet
{
  friend struct PairPosFormat1;

  bool intersects (const hb_set_t *glyphs,
                   const ValueFormat *valueFormats) const
  {
    unsigned int len1 = valueFormats[0].get_len ();
    unsigned int len2 = valueFormats[1].get_len ();
    unsigned int record_size = HBUINT16::static_size * (1 + len1 + len2);

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
    unsigned int len1 = valueFormats[0].get_len ();
    unsigned int len2 = valueFormats[1].get_len ();
    unsigned int record_size = HBUINT16::static_size * (1 + len1 + len2);

    const PairValueRecord *record = &firstPairValueRecord;
    c->input->add_array (&record->secondGlyph, len, record_size);
  }

  void collect_variation_indices (hb_collect_variation_indices_context_t *c,
                                  const ValueFormat *valueFormats) const
  {
    unsigned len1 = valueFormats[0].get_len ();
    unsigned len2 = valueFormats[1].get_len ();
    unsigned record_size = HBUINT16::static_size * (1 + len1 + len2);

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
    unsigned int record_size = HBUINT16::static_size * (1 + len1 + len2);

    const PairValueRecord *record = hb_bsearch (buffer->info[pos].codepoint,
                                                &firstPairValueRecord,
                                                len,
                                                record_size);
    if (record)
    {
      bool applied_first = valueFormats[0].apply_value (c, this, &record->values[0], buffer->cur_pos());
      bool applied_second = valueFormats[1].apply_value (c, this, &record->values[len1], buffer->pos[pos]);
      if (applied_first || applied_second)
        buffer->unsafe_to_break (buffer->idx, pos + 1);
      if (len2)
        pos++;
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
    unsigned record_size = HBUINT16::static_size + Value::static_size * (len1 + len2);

    PairValueRecord::context_t context =
    {
      this,
      valueFormats,
      newFormats,
      len1,
      &glyph_map,
      c->plan->layout_variation_idx_map
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

  struct sanitize_closure_t
  {
    const ValueFormat *valueFormats;
    unsigned int len1; /* valueFormats[0].get_len() */
    unsigned int stride; /* 1 + len1 + len2 */
  };

  bool sanitize (hb_sanitize_context_t *c, const sanitize_closure_t *closure) const
  {
    TRACE_SANITIZE (this);
    if (!(c->check_struct (this)
       && c->check_range (&firstPairValueRecord,
                          len,
                          HBUINT16::static_size,
                          closure->stride))) return_trace (false);

    unsigned int count = len;
    const PairValueRecord *record = &firstPairValueRecord;
    return_trace (closure->valueFormats[0].sanitize_values_stride_unsafe (c, this, &record->values[0], count, closure->stride) &&
                  closure->valueFormats[1].sanitize_values_stride_unsafe (c, this, &record->values[closure->len1], count, closure->stride));
  }

  protected:
  HBUINT16              len;    /* Number of PairValueRecords */
  PairValueRecord       firstPairValueRecord;
                                /* Array of PairValueRecords--ordered
                                 * by GlyphID of the second glyph */
  public:
  DEFINE_SIZE_MIN (2);
};

struct PairPosFormat1
{
  protected:
  HBUINT16      format;                 /* Format identifier--format = 1 */
  Offset16To<Coverage>
                coverage;               /* Offset to Coverage table--from
                                         * beginning of subtable */
  ValueFormat   valueFormat[2];         /* [0] Defines the types of data in
                                         * ValueRecord1--for the first glyph
                                         * in the pair--may be zero (0) */
                                        /* [1] Defines the types of data in
                                         * ValueRecord2--for the second glyph
                                         * in the pair--may be zero (0) */
  Array16OfOffset16To<PairSet>
                pairSet;                /* Array of PairSet tables
                                         * ordered by Coverage Index */
  public:
  DEFINE_SIZE_ARRAY (10, pairSet);

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);

    if (!c->check_struct (this)) return_trace (false);

    unsigned int len1 = valueFormat[0].get_len ();
    unsigned int len2 = valueFormat[1].get_len ();
    PairSet::sanitize_closure_t closure =
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
    | hb_map ([glyphs, this] (const Offset16To<PairSet> &_)
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
    | hb_filter ([this, c, out] (const Offset16To<PairSet>& _)
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
    for (const Offset16To<PairSet>& _ :
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
