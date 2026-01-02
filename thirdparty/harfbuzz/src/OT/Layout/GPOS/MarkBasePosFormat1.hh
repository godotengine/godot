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

  static inline bool accept (hb_buffer_t *buffer, unsigned idx)
  {
    /* We only want to attach to the first of a MultipleSubst sequence.
     * https://github.com/harfbuzz/harfbuzz/issues/740
     * Reject others...
     * ...but stop if we find a mark in the MultipleSubst sequence:
     * https://github.com/harfbuzz/harfbuzz/issues/1020 */
    return !_hb_glyph_info_multiplied (&buffer->info[idx]) ||
	   0 == _hb_glyph_info_get_lig_comp (&buffer->info[idx]) ||
	   (idx == 0 ||
	    _hb_glyph_info_is_mark (&buffer->info[idx - 1]) ||
	    !_hb_glyph_info_multiplied (&buffer->info[idx - 1]) ||
	    _hb_glyph_info_get_lig_id (&buffer->info[idx]) !=
	    _hb_glyph_info_get_lig_id (&buffer->info[idx - 1]) ||
	    _hb_glyph_info_get_lig_comp (&buffer->info[idx]) !=
	    _hb_glyph_info_get_lig_comp (&buffer->info[idx - 1]) + 1
	    );
  }

  bool apply (hb_ot_apply_context_t *c) const
  {
    TRACE_APPLY (this);
    hb_buffer_t *buffer = c->buffer;
    unsigned int mark_index = (this+markCoverage).get_coverage  (buffer->cur().codepoint);
    if (likely (mark_index == NOT_COVERED)) return_trace (false);

    /* Now we search backwards for a non-mark glyph.
     * We don't use skippy_iter.prev() to avoid O(n^2) behavior. */

    auto &skippy_iter = c->iter_input;
    skippy_iter.set_lookup_props (LookupFlag::IgnoreMarks);

    if (c->last_base_until > buffer->idx)
    {
      c->last_base_until = 0;
      c->last_base = -1;
    }
    unsigned j;
    for (j = buffer->idx; j > c->last_base_until; j--)
    {
      auto match = skippy_iter.match (buffer->info[j - 1]);
      if (match == skippy_iter.MATCH)
      {
        // https://github.com/harfbuzz/harfbuzz/issues/4124
	if (!accept (buffer, j - 1) &&
	    NOT_COVERED == (this+baseCoverage).get_coverage  (buffer->info[j - 1].codepoint))
	  match = skippy_iter.SKIP;
      }
      if (match == skippy_iter.MATCH)
      {
	c->last_base = (signed) j - 1;
	break;
      }
    }
    c->last_base_until = buffer->idx;
    if (c->last_base == -1)
    {
      buffer->unsafe_to_concat_from_outbuffer (0, buffer->idx + 1);
      return_trace (false);
    }

    unsigned idx = (unsigned) c->last_base;

    /* Checking that matched glyph is actually a base glyph by GDEF is too strong; disabled */
    //if (!_hb_glyph_info_is_base_glyph (&buffer->info[idx])) { return_trace (false); }

    unsigned int base_index = (this+baseCoverage).get_coverage  (buffer->info[idx].codepoint);
    if (base_index == NOT_COVERED)
    {
      buffer->unsafe_to_concat_from_outbuffer (idx, buffer->idx + 1);
      return_trace (false);
    }

    return_trace ((this+markArray).apply (c, mark_index, base_index, this+baseArray, classCount, idx));
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

    if (unlikely (!out->markArray.serialize_subset (c, markArray, this,
						    (this+markCoverage).iter (),
						    &klass_mapping)))
      return_trace (false);

    unsigned basecount = (this+baseArray).rows;
    auto base_iter =
    + hb_zip (this+baseCoverage, hb_range (basecount))
    | hb_filter (glyphset, hb_first)
    ;

    new_coverage.reset ();
    hb_sorted_vector_t<unsigned> base_indexes;
    auto &base_array = (this+baseArray);
    for (const auto _ : + base_iter)
    {
      unsigned row = _.second;
      bool non_empty = + hb_range ((unsigned) classCount)
                       | hb_filter (klass_mapping)
                       | hb_map ([&] (const unsigned col) { return !base_array.offset_is_null (row, col, (unsigned) classCount); })
                       | hb_any
                       ;

      if (!non_empty) continue;
      
      hb_codepoint_t new_g = glyph_map.get ( _.first);
      new_coverage.push (new_g);

      + hb_range ((unsigned) classCount)
      | hb_filter (klass_mapping)
      | hb_map ([&] (const unsigned col) { return row * (unsigned) classCount + col; })
      | hb_sink (base_indexes)
      ;
    }

    if (!new_coverage) return_trace (false);
    if (!out->baseCoverage.serialize_serialize (c->serializer, new_coverage.iter ()))
      return_trace (false);

    return_trace (out->baseArray.serialize_subset (c, baseArray, this,
						   new_coverage.length,
						   base_indexes.iter ()));
  }
};


}
}
}

#endif /* OT_LAYOUT_GPOS_MARKBASEPOSFORMAT1_HH */
