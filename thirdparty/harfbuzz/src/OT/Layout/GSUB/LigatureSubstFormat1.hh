#ifndef OT_LAYOUT_GSUB_LIGATURESUBSTFORMAT1_HH
#define OT_LAYOUT_GSUB_LIGATURESUBSTFORMAT1_HH

#include "Common.hh"
#include "LigatureSet.hh"

namespace OT {
namespace Layout {
namespace GSUB_impl {

template <typename Types>
struct LigatureSubstFormat1_2
{
  protected:
  HBUINT16      format;                 /* Format identifier--format = 1 */
  typename Types::template OffsetTo<Coverage>
                coverage;               /* Offset to Coverage table--from
                                         * beginning of Substitution table */
  Array16Of<typename Types::template OffsetTo<LigatureSet<Types>>>
                ligatureSet;            /* Array LigatureSet tables
                                         * ordered by Coverage Index */
  public:
  DEFINE_SIZE_ARRAY (4 + Types::size, ligatureSet);

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (coverage.sanitize (c, this) && ligatureSet.sanitize (c, this));
  }

  bool intersects (const hb_set_t *glyphs) const
  {
    return
    + hb_zip (this+coverage, ligatureSet)
    | hb_filter (*glyphs, hb_first)
    | hb_map (hb_second)
    | hb_map ([this, glyphs] (const typename Types::template OffsetTo<LigatureSet<Types>> &_)
              { return (this+_).intersects (glyphs); })
    | hb_any
    ;
  }

  bool may_have_non_1to1 () const
  { return true; }

  void closure (hb_closure_context_t *c) const
  {
    + hb_zip (this+coverage, ligatureSet)
    | hb_filter (c->parent_active_glyphs (), hb_first)
    | hb_map (hb_second)
    | hb_map (hb_add (this))
    | hb_apply ([c] (const LigatureSet<Types> &_) { _.closure (c); })
    ;

  }

  void closure_lookups (hb_closure_lookups_context_t *c) const {}

  void collect_glyphs (hb_collect_glyphs_context_t *c) const
  {
    if (unlikely (!(this+coverage).collect_coverage (c->input))) return;

    + hb_zip (this+coverage, ligatureSet)
    | hb_map (hb_second)
    | hb_map (hb_add (this))
    | hb_apply ([c] (const LigatureSet<Types> &_) { _.collect_glyphs (c); })
    ;
  }

  const Coverage &get_coverage () const { return this+coverage; }

  bool would_apply (hb_would_apply_context_t *c) const
  {
    unsigned int index = (this+coverage).get_coverage (c->glyphs[0]);
    if (likely (index == NOT_COVERED)) return false;

    const auto &lig_set = this+ligatureSet[index];
    return lig_set.would_apply (c);
  }

  bool apply (hb_ot_apply_context_t *c) const
  {
    TRACE_APPLY (this);

    unsigned int index = (this+coverage).get_coverage (c->buffer->cur ().codepoint);
    if (likely (index == NOT_COVERED)) return_trace (false);

    const auto &lig_set = this+ligatureSet[index];
    return_trace (lig_set.apply (c));
  }

  bool serialize (hb_serialize_context_t *c,
                  hb_sorted_array_t<const HBGlyphID16> first_glyphs,
                  hb_array_t<const unsigned int> ligature_per_first_glyph_count_list,
                  hb_array_t<const HBGlyphID16> ligatures_list,
                  hb_array_t<const unsigned int> component_count_list,
                  hb_array_t<const HBGlyphID16> component_list /* Starting from second for each ligature */)
  {
    TRACE_SERIALIZE (this);
    if (unlikely (!c->extend_min (this))) return_trace (false);
    if (unlikely (!ligatureSet.serialize (c, first_glyphs.length))) return_trace (false);
    for (unsigned int i = 0; i < first_glyphs.length; i++)
    {
      unsigned int ligature_count = ligature_per_first_glyph_count_list[i];
      if (unlikely (!ligatureSet[i]
                        .serialize_serialize (c,
                                              ligatures_list.sub_array (0, ligature_count),
                                              component_count_list.sub_array (0, ligature_count),
                                              component_list))) return_trace (false);
      ligatures_list += ligature_count;
      component_count_list += ligature_count;
    }
    return_trace (coverage.serialize_serialize (c, first_glyphs));
  }

  bool subset (hb_subset_context_t *c) const
  {
    TRACE_SUBSET (this);
    const hb_set_t &glyphset = *c->plan->glyphset_gsub ();
    const hb_map_t &glyph_map = *c->plan->glyph_map;

    auto *out = c->serializer->start_embed (*this);
    if (unlikely (!c->serializer->extend_min (out))) return_trace (false);
    out->format = format;

    // Due to a bug in some older versions of windows 7 the Coverage table must be
    // packed after the LigatureSet and Ligature tables, so serialize Coverage first
    // which places it last in the packed order.
    hb_set_t new_coverage;
    + hb_zip (this+coverage, hb_iter (ligatureSet) | hb_map (hb_add (this)))
    | hb_filter (glyphset, hb_first)
    | hb_filter ([&] (const LigatureSet<Types>& _) {
      return _.intersects_lig_glyph (&glyphset);
    }, hb_second)
    | hb_map (hb_first)
    | hb_sink (new_coverage);

    if (!c->serializer->push<Coverage> ()
        ->serialize (c->serializer,
                     + new_coverage.iter () | hb_map_retains_sorting (glyph_map)))
    {
      c->serializer->pop_discard ();
      return_trace (false);
    }

    unsigned coverage_idx = c->serializer->pop_pack ();
     c->serializer->add_link (out->coverage, coverage_idx);

    + hb_zip (this+coverage, ligatureSet)
    | hb_filter (new_coverage, hb_first)
    | hb_map (hb_second)
    // to ensure that the repacker always orders the coverage table after the LigatureSet
    // and LigatureSubtable's they will be linked to the Coverage table via a virtual link
    // the coverage table object idx is passed down to facilitate this.
    | hb_apply (subset_offset_array (c, out->ligatureSet, this, coverage_idx))
    ;

    return_trace (bool (new_coverage));
  }
};

}
}
}

#endif  /* OT_LAYOUT_GSUB_LIGATURESUBSTFORMAT1_HH */
