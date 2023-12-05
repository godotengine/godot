#ifndef OT_LAYOUT_GPOS_MARKMARKPOSFORMAT1_HH
#define OT_LAYOUT_GPOS_MARKMARKPOSFORMAT1_HH

#include "MarkMarkPosFormat1.hh"

namespace OT {
namespace Layout {
namespace GPOS_impl {

typedef AnchorMatrix Mark2Array;        /* mark2-major--
                                         * in order of Mark2Coverage Index--,
                                         * mark1-minor--
                                         * ordered by class--zero-based. */

template <typename Types>
struct MarkMarkPosFormat1_2
{
  protected:
  HBUINT16      format;                 /* Format identifier--format = 1 */
  typename Types::template OffsetTo<Coverage>
                mark1Coverage;          /* Offset to Combining Mark1 Coverage
                                         * table--from beginning of MarkMarkPos
                                         * subtable */
  typename Types::template OffsetTo<Coverage>
                mark2Coverage;          /* Offset to Combining Mark2 Coverage
                                         * table--from beginning of MarkMarkPos
                                         * subtable */
  HBUINT16      classCount;             /* Number of defined mark classes */
  typename Types::template OffsetTo<MarkArray>
                mark1Array;             /* Offset to Mark1Array table--from
                                         * beginning of MarkMarkPos subtable */
  typename Types::template OffsetTo<Mark2Array>
                mark2Array;             /* Offset to Mark2Array table--from
                                         * beginning of MarkMarkPos subtable */
  public:
  DEFINE_SIZE_STATIC (4 + 4 * Types::size);

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) &&
                  mark1Coverage.sanitize (c, this) &&
                  mark2Coverage.sanitize (c, this) &&
                  mark1Array.sanitize (c, this) &&
                  mark2Array.sanitize (c, this, (unsigned int) classCount));
  }

  bool intersects (const hb_set_t *glyphs) const
  {
    return (this+mark1Coverage).intersects (glyphs) &&
           (this+mark2Coverage).intersects (glyphs);
  }

  void closure_lookups (hb_closure_lookups_context_t *c) const {}

  void collect_variation_indices (hb_collect_variation_indices_context_t *c) const
  {
    + hb_zip (this+mark1Coverage, this+mark1Array)
    | hb_filter (c->glyph_set, hb_first)
    | hb_map (hb_second)
    | hb_apply ([&] (const MarkRecord& record) { record.collect_variation_indices (c, &(this+mark1Array)); })
    ;

    hb_map_t klass_mapping;
    Markclass_closure_and_remap_indexes (this+mark1Coverage, this+mark1Array, *c->glyph_set, &klass_mapping);

    unsigned mark2_count = (this+mark2Array).rows;
    auto mark2_iter =
    + hb_zip (this+mark2Coverage, hb_range (mark2_count))
    | hb_filter (c->glyph_set, hb_first)
    | hb_map (hb_second)
    ;

    hb_sorted_vector_t<unsigned> mark2_indexes;
    for (const unsigned row : mark2_iter)
    {
      + hb_range ((unsigned) classCount)
      | hb_filter (klass_mapping)
      | hb_map ([&] (const unsigned col) { return row * (unsigned) classCount + col; })
      | hb_sink (mark2_indexes)
      ;
    }
    (this+mark2Array).collect_variation_indices (c, mark2_indexes.iter ());
  }

  void collect_glyphs (hb_collect_glyphs_context_t *c) const
  {
    if (unlikely (!(this+mark1Coverage).collect_coverage (c->input))) return;
    if (unlikely (!(this+mark2Coverage).collect_coverage (c->input))) return;
  }

  const Coverage &get_coverage () const { return this+mark1Coverage; }

  bool apply (hb_ot_apply_context_t *c) const
  {
    TRACE_APPLY (this);
    hb_buffer_t *buffer = c->buffer;
    unsigned int mark1_index = (this+mark1Coverage).get_coverage  (buffer->cur().codepoint);
    if (likely (mark1_index == NOT_COVERED)) return_trace (false);

    /* now we search backwards for a suitable mark glyph until a non-mark glyph */
    hb_ot_apply_context_t::skipping_iterator_t &skippy_iter = c->iter_input;
    skippy_iter.reset_fast (buffer->idx);
    skippy_iter.set_lookup_props (c->lookup_props & ~(uint32_t)LookupFlag::IgnoreFlags);
    unsigned unsafe_from;
    if (unlikely (!skippy_iter.prev (&unsafe_from)))
    {
      buffer->unsafe_to_concat_from_outbuffer (unsafe_from, buffer->idx + 1);
      return_trace (false);
    }

    if (likely (!_hb_glyph_info_is_mark (&buffer->info[skippy_iter.idx])))
    {
      buffer->unsafe_to_concat_from_outbuffer (skippy_iter.idx, buffer->idx + 1);
      return_trace (false);
    }

    unsigned int j = skippy_iter.idx;

    unsigned int id1 = _hb_glyph_info_get_lig_id (&buffer->cur());
    unsigned int id2 = _hb_glyph_info_get_lig_id (&buffer->info[j]);
    unsigned int comp1 = _hb_glyph_info_get_lig_comp (&buffer->cur());
    unsigned int comp2 = _hb_glyph_info_get_lig_comp (&buffer->info[j]);

    if (likely (id1 == id2))
    {
      if (id1 == 0) /* Marks belonging to the same base. */
        goto good;
      else if (comp1 == comp2) /* Marks belonging to the same ligature component. */
        goto good;
    }
    else
    {
      /* If ligature ids don't match, it may be the case that one of the marks
       * itself is a ligature.  In which case match. */
      if ((id1 > 0 && !comp1) || (id2 > 0 && !comp2))
        goto good;
    }

    /* Didn't match. */
    buffer->unsafe_to_concat_from_outbuffer (skippy_iter.idx, buffer->idx + 1);
    return_trace (false);

    good:
    unsigned int mark2_index = (this+mark2Coverage).get_coverage  (buffer->info[j].codepoint);
    if (mark2_index == NOT_COVERED)
    {
      buffer->unsafe_to_concat_from_outbuffer (skippy_iter.idx, buffer->idx + 1);
      return_trace (false);
    }

    return_trace ((this+mark1Array).apply (c, mark1_index, mark2_index, this+mark2Array, classCount, j));
  }

  bool subset (hb_subset_context_t *c) const
  {
    TRACE_SUBSET (this);
    const hb_set_t &glyphset = *c->plan->glyphset_gsub ();
    const hb_map_t &glyph_map = *c->plan->glyph_map;

    auto *out = c->serializer->start_embed (*this);
    if (unlikely (!c->serializer->extend_min (out))) return_trace (false);
    out->format = format;

    hb_map_t klass_mapping;
    Markclass_closure_and_remap_indexes (this+mark1Coverage, this+mark1Array, glyphset, &klass_mapping);

    if (!klass_mapping.get_population ()) return_trace (false);
    out->classCount = klass_mapping.get_population ();

    auto mark1_iter =
    + hb_zip (this+mark1Coverage, this+mark1Array)
    | hb_filter (glyphset, hb_first)
    ;

    hb_sorted_vector_t<hb_codepoint_t> new_coverage;
    + mark1_iter
    | hb_map (hb_first)
    | hb_map (glyph_map)
    | hb_sink (new_coverage)
    ;

    if (!out->mark1Coverage.serialize_serialize (c->serializer, new_coverage.iter ()))
      return_trace (false);

    if (unlikely (!out->mark1Array.serialize_subset (c, mark1Array, this,
						     (this+mark1Coverage).iter (),
						     &klass_mapping)))
      return_trace (false);

    unsigned mark2count = (this+mark2Array).rows;
    auto mark2_iter =
    + hb_zip (this+mark2Coverage, hb_range (mark2count))
    | hb_filter (glyphset, hb_first)
    ;

    new_coverage.reset ();
    + mark2_iter
    | hb_map (hb_first)
    | hb_map (glyph_map)
    | hb_sink (new_coverage)
    ;

    if (!out->mark2Coverage.serialize_serialize (c->serializer, new_coverage.iter ()))
      return_trace (false);

    hb_sorted_vector_t<unsigned> mark2_indexes;
    for (const unsigned row : + mark2_iter
                              | hb_map (hb_second))
    {
      + hb_range ((unsigned) classCount)
      | hb_filter (klass_mapping)
      | hb_map ([&] (const unsigned col) { return row * (unsigned) classCount + col; })
      | hb_sink (mark2_indexes)
      ;
    }

    return_trace (out->mark2Array.serialize_subset (c, mark2Array, this,
						    mark2_iter.len (),
						    mark2_indexes.iter ()));

  }
};


}
}
}

#endif /* OT_LAYOUT_GPOS_MARKMARKPOSFORMAT1_HH */
