#ifndef OT_LAYOUT_GPOS_MARKBASEPOSFORMAT1_HH
#define OT_LAYOUT_GPOS_MARKBASEPOSFORMAT1_HH

#include "MarkArray.hh"

namespace OT {
namespace Layout {
namespace GPOS_impl {

typedef AnchorMatrix BaseArray;         /* base-major--
                                         * in order of BaseCoverage Index--,
                                         * mark-minor--
                                         * ordered by class--zero-based. */

template <typename Types>
struct MarkBasePosFormat1_2
{
  protected:
  HBUINT16      format;                 /* Format identifier--format = 1 */
  typename Types::template OffsetTo<Coverage>
                markCoverage;           /* Offset to MarkCoverage table--from
                                         * beginning of MarkBasePos subtable */
  typename Types::template OffsetTo<Coverage>
                baseCoverage;           /* Offset to BaseCoverage table--from
                                         * beginning of MarkBasePos subtable */
  HBUINT16      classCount;             /* Number of classes defined for marks */
  typename Types::template OffsetTo<MarkArray>
                markArray;              /* Offset to MarkArray table--from
                                         * beginning of MarkBasePos subtable */
  typename Types::template OffsetTo<BaseArray>
                baseArray;              /* Offset to BaseArray table--from
                                         * beginning of MarkBasePos subtable */

  public:
  DEFINE_SIZE_STATIC (4 + 4 * Types::size);

    bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) &&
                  markCoverage.sanitize (c, this) &&
                  baseCoverage.sanitize (c, this) &&
                  markArray.sanitize (c, this) &&
                  baseArray.sanitize (c, this, (unsigned int) classCount));
  }

  bool intersects (const hb_set_t *glyphs) const
  {
    return (this+markCoverage).intersects (glyphs) &&
           (this+baseCoverage).intersects (glyphs);
  }

  void closure_lookups (hb_closure_lookups_context_t *c) const {}

  void collect_variation_indices (hb_collect_variation_indices_context_t *c) const
  {
    + hb_zip (this+markCoverage, this+markArray)
    | hb_filter (c->glyph_set, hb_first)
    | hb_map (hb_second)
    | hb_apply ([&] (const MarkRecord& record) { record.collect_variation_indices (c, &(this+markArray)); })
    ;

    hb_map_t klass_mapping;
    Markclass_closure_and_remap_indexes (this+markCoverage, this+markArray, *c->glyph_set, &klass_mapping);

    unsigned basecount = (this+baseArray).rows;
    auto base_iter =
    + hb_zip (this+baseCoverage, hb_range (basecount))
    | hb_filter (c->glyph_set, hb_first)
    | hb_map (hb_second)
    ;

    hb_sorted_vector_t<unsigned> base_indexes;
    for (const unsigned row : base_iter)
    {
      + hb_range ((unsigned) classCount)
      | hb_filter (klass_mapping)
      | hb_map ([&] (const unsigned col) { return row * (unsigned) classCount + col; })
      | hb_sink (base_indexes)
      ;
    }
    (this+baseArray).collect_variation_indices (c, base_indexes.iter ());
  }

  void collect_glyphs (hb_collect_glyphs_context_t *c) const
  {
    if (unlikely (!(this+markCoverage).collect_coverage (c->input))) return;
    if (unlikely (!(this+baseCoverage).collect_coverage (c->input))) return;
  }

  const Coverage &get_coverage () const { return this+markCoverage; }

  bool apply (hb_ot_apply_context_t *c) const
  {
    TRACE_APPLY (this);
    hb_buffer_t *buffer = c->buffer;
    unsigned int mark_index = (this+markCoverage).get_coverage  (buffer->cur().codepoint);
    if (likely (mark_index == NOT_COVERED)) return_trace (false);

    /* Now we search backwards for a non-mark glyph */
    hb_ot_apply_context_t::skipping_iterator_t &skippy_iter = c->iter_input;
    skippy_iter.reset (buffer->idx, 1);
    skippy_iter.set_lookup_props (LookupFlag::IgnoreMarks);
    do {
      unsigned unsafe_from;
      if (!skippy_iter.prev (&unsafe_from))
      {
        buffer->unsafe_to_concat_from_outbuffer (unsafe_from, buffer->idx + 1);
        return_trace (false);
      }

      /* We only want to attach to the first of a MultipleSubst sequence.
       * https://github.com/harfbuzz/harfbuzz/issues/740
       * Reject others...
       * ...but stop if we find a mark in the MultipleSubst sequence:
       * https://github.com/harfbuzz/harfbuzz/issues/1020 */
      if (!_hb_glyph_info_multiplied (&buffer->info[skippy_iter.idx]) ||
          0 == _hb_glyph_info_get_lig_comp (&buffer->info[skippy_iter.idx]) ||
          (skippy_iter.idx == 0 ||
           _hb_glyph_info_is_mark (&buffer->info[skippy_iter.idx - 1]) ||
           !_hb_glyph_info_multiplied (&buffer->info[skippy_iter.idx - 1]) ||
           _hb_glyph_info_get_lig_id (&buffer->info[skippy_iter.idx]) !=
           _hb_glyph_info_get_lig_id (&buffer->info[skippy_iter.idx - 1]) ||
           _hb_glyph_info_get_lig_comp (&buffer->info[skippy_iter.idx]) !=
           _hb_glyph_info_get_lig_comp (&buffer->info[skippy_iter.idx - 1]) + 1
           ))
        break;
      skippy_iter.reject ();
    } while (true);

    /* Checking that matched glyph is actually a base glyph by GDEF is too strong; disabled */
    //if (!_hb_glyph_info_is_base_glyph (&buffer->info[skippy_iter.idx])) { return_trace (false); }

    unsigned int base_index = (this+baseCoverage).get_coverage  (buffer->info[skippy_iter.idx].codepoint);
    if (base_index == NOT_COVERED)
    {
      buffer->unsafe_to_concat_from_outbuffer (skippy_iter.idx, buffer->idx + 1);
      return_trace (false);
    }

    return_trace ((this+markArray).apply (c, mark_index, base_index, this+baseArray, classCount, skippy_iter.idx));
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
    Markclass_closure_and_remap_indexes (this+markCoverage, this+markArray, glyphset, &klass_mapping);

    if (!klass_mapping.get_population ()) return_trace (false);
    out->classCount = klass_mapping.get_population ();

    auto mark_iter =
    + hb_zip (this+markCoverage, this+markArray)
    | hb_filter (glyphset, hb_first)
    ;

    hb_sorted_vector_t<hb_codepoint_t> new_coverage;
    + mark_iter
    | hb_map (hb_first)
    | hb_map (glyph_map)
    | hb_sink (new_coverage)
    ;

    if (!out->markCoverage.serialize_serialize (c->serializer, new_coverage.iter ()))
      return_trace (false);

    out->markArray.serialize_subset (c, markArray, this,
                                     (this+markCoverage).iter (),
                                     &klass_mapping);

    unsigned basecount = (this+baseArray).rows;
    auto base_iter =
    + hb_zip (this+baseCoverage, hb_range (basecount))
    | hb_filter (glyphset, hb_first)
    ;

    new_coverage.reset ();
    + base_iter
    | hb_map (hb_first)
    | hb_map (glyph_map)
    | hb_sink (new_coverage)
    ;

    if (!out->baseCoverage.serialize_serialize (c->serializer, new_coverage.iter ()))
      return_trace (false);

    hb_sorted_vector_t<unsigned> base_indexes;
    for (const unsigned row : + base_iter
                              | hb_map (hb_second))
    {
      + hb_range ((unsigned) classCount)
      | hb_filter (klass_mapping)
      | hb_map ([&] (const unsigned col) { return row * (unsigned) classCount + col; })
      | hb_sink (base_indexes)
      ;
    }

    out->baseArray.serialize_subset (c, baseArray, this,
                                     base_iter.len (),
                                     base_indexes.iter ());

    return_trace (true);
  }
};


}
}
}

#endif /* OT_LAYOUT_GPOS_MARKBASEPOSFORMAT1_HH */
