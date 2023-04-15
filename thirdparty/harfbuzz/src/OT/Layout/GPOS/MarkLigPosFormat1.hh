#ifndef OT_LAYOUT_GPOS_MARKLIGPOSFORMAT1_HH
#define OT_LAYOUT_GPOS_MARKLIGPOSFORMAT1_HH

#include "LigatureArray.hh"

namespace OT {
namespace Layout {
namespace GPOS_impl {


template <typename Types>
struct MarkLigPosFormat1_2
{
  protected:
  HBUINT16      format;                 /* Format identifier--format = 1 */
  typename Types::template OffsetTo<Coverage>
                markCoverage;           /* Offset to Mark Coverage table--from
                                         * beginning of MarkLigPos subtable */
  typename Types::template OffsetTo<Coverage>
                ligatureCoverage;       /* Offset to Ligature Coverage
                                         * table--from beginning of MarkLigPos
                                         * subtable */
  HBUINT16      classCount;             /* Number of defined mark classes */
  typename Types::template OffsetTo<MarkArray>
                markArray;              /* Offset to MarkArray table--from
                                         * beginning of MarkLigPos subtable */
  typename Types::template OffsetTo<LigatureArray>
                ligatureArray;          /* Offset to LigatureArray table--from
                                         * beginning of MarkLigPos subtable */
  public:
  DEFINE_SIZE_STATIC (4 + 4 * Types::size);

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) &&
                  markCoverage.sanitize (c, this) &&
                  ligatureCoverage.sanitize (c, this) &&
                  markArray.sanitize (c, this) &&
                  ligatureArray.sanitize (c, this, (unsigned int) classCount));
  }

  bool intersects (const hb_set_t *glyphs) const
  {
    return (this+markCoverage).intersects (glyphs) &&
           (this+ligatureCoverage).intersects (glyphs);
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

    unsigned ligcount = (this+ligatureArray).len;
    auto lig_iter =
    + hb_zip (this+ligatureCoverage, hb_range (ligcount))
    | hb_filter (c->glyph_set, hb_first)
    | hb_map (hb_second)
    ;

    const LigatureArray& lig_array = this+ligatureArray;
    for (const unsigned i : lig_iter)
    {
      hb_sorted_vector_t<unsigned> lig_indexes;
      unsigned row_count = lig_array[i].rows;
      for (unsigned row : + hb_range (row_count))
      {
        + hb_range ((unsigned) classCount)
        | hb_filter (klass_mapping)
        | hb_map ([&] (const unsigned col) { return row * (unsigned) classCount + col; })
        | hb_sink (lig_indexes)
        ;
      }

      lig_array[i].collect_variation_indices (c, lig_indexes.iter ());
    }
  }

  void collect_glyphs (hb_collect_glyphs_context_t *c) const
  {
    if (unlikely (!(this+markCoverage).collect_coverage (c->input))) return;
    if (unlikely (!(this+ligatureCoverage).collect_coverage (c->input))) return;
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

    /* Checking that matched glyph is actually a ligature by GDEF is too strong; disabled */
    //if (!_hb_glyph_info_is_ligature (&buffer->info[idx])) { return_trace (false); }

    unsigned int lig_index = (this+ligatureCoverage).get_coverage  (buffer->info[idx].codepoint);
    if (lig_index == NOT_COVERED)
    {
      buffer->unsafe_to_concat_from_outbuffer (idx, buffer->idx + 1);
      return_trace (false);
    }

    const LigatureArray& lig_array = this+ligatureArray;
    const LigatureAttach& lig_attach = lig_array[lig_index];

    /* Find component to attach to */
    unsigned int comp_count = lig_attach.rows;
    if (unlikely (!comp_count))
    {
      buffer->unsafe_to_concat_from_outbuffer (idx, buffer->idx + 1);
      return_trace (false);
    }

    /* We must now check whether the ligature ID of the current mark glyph
     * is identical to the ligature ID of the found ligature.  If yes, we
     * can directly use the component index.  If not, we attach the mark
     * glyph to the last component of the ligature. */
    unsigned int comp_index;
    unsigned int lig_id = _hb_glyph_info_get_lig_id (&buffer->info[idx]);
    unsigned int mark_id = _hb_glyph_info_get_lig_id (&buffer->cur());
    unsigned int mark_comp = _hb_glyph_info_get_lig_comp (&buffer->cur());
    if (lig_id && lig_id == mark_id && mark_comp > 0)
      comp_index = hb_min (comp_count, _hb_glyph_info_get_lig_comp (&buffer->cur())) - 1;
    else
      comp_index = comp_count - 1;

    return_trace ((this+markArray).apply (c, mark_index, comp_index, lig_attach, classCount, idx));
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

    auto new_mark_coverage =
    + mark_iter
    | hb_map_retains_sorting (hb_first)
    | hb_map_retains_sorting (glyph_map)
    ;

    if (!out->markCoverage.serialize_serialize (c->serializer, new_mark_coverage))
      return_trace (false);

    out->markArray.serialize_subset (c, markArray, this,
                                     (this+markCoverage).iter (),
                                     &klass_mapping);

    auto new_ligature_coverage =
    + hb_iter (this + ligatureCoverage)
    | hb_filter (glyphset)
    | hb_map_retains_sorting (glyph_map)
    ;

    if (!out->ligatureCoverage.serialize_serialize (c->serializer, new_ligature_coverage))
      return_trace (false);

    out->ligatureArray.serialize_subset (c, ligatureArray, this,
                                         hb_iter (this+ligatureCoverage), classCount, &klass_mapping);

    return_trace (true);
  }

};

}
}
}

#endif /* OT_LAYOUT_GPOS_MARKLIGPOSFORMAT1_HH */
