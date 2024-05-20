#ifndef OT_LAYOUT_GSUB_MULTIPLESUBSTFORMAT1_HH
#define OT_LAYOUT_GSUB_MULTIPLESUBSTFORMAT1_HH

#include "Common.hh"
#include "Sequence.hh"

namespace OT {
namespace Layout {
namespace GSUB_impl {

template <typename Types>
struct MultipleSubstFormat1_2
{
  protected:
  HBUINT16      format;                 /* Format identifier--format = 1 */
  typename Types::template OffsetTo<Coverage>
                coverage;               /* Offset to Coverage table--from
                                         * beginning of Substitution table */
  Array16Of<typename Types::template OffsetTo<Sequence<Types>>>
                sequence;               /* Array of Sequence tables
                                         * ordered by Coverage Index */
  public:
  DEFINE_SIZE_ARRAY (4 + Types::size, sequence);

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (coverage.sanitize (c, this) && sequence.sanitize (c, this));
  }

  bool intersects (const hb_set_t *glyphs) const
  { return (this+coverage).intersects (glyphs); }

  bool may_have_non_1to1 () const
  { return true; }

  void closure (hb_closure_context_t *c) const
  {
    + hb_zip (this+coverage, sequence)
    | hb_filter (c->parent_active_glyphs (), hb_first)
    | hb_map (hb_second)
    | hb_map (hb_add (this))
    | hb_apply ([c] (const Sequence<Types> &_) { _.closure (c); })
    ;
  }

  void closure_lookups (hb_closure_lookups_context_t *c) const {}

  void collect_glyphs (hb_collect_glyphs_context_t *c) const
  {
    if (unlikely (!(this+coverage).collect_coverage (c->input))) return;
    + hb_zip (this+coverage, sequence)
    | hb_map (hb_second)
    | hb_map (hb_add (this))
    | hb_apply ([c] (const Sequence<Types> &_) { _.collect_glyphs (c); })
    ;
  }

  const Coverage &get_coverage () const { return this+coverage; }

  bool would_apply (hb_would_apply_context_t *c) const
  { return c->len == 1 && (this+coverage).get_coverage (c->glyphs[0]) != NOT_COVERED; }

  bool apply (hb_ot_apply_context_t *c) const
  {
    TRACE_APPLY (this);

    unsigned int index = (this+coverage).get_coverage (c->buffer->cur().codepoint);
    if (likely (index == NOT_COVERED)) return_trace (false);

    return_trace ((this+sequence[index]).apply (c));
  }

  template<typename Iterator,
           hb_requires (hb_is_sorted_iterator (Iterator))>
  bool serialize (hb_serialize_context_t *c,
		  Iterator it)
  {
    TRACE_SERIALIZE (this);
    auto sequences =
      + it
      | hb_map (hb_second)
      ;
    auto glyphs =
      + it
      | hb_map_retains_sorting (hb_first)
      ;
    if (unlikely (!c->extend_min (this))) return_trace (false);

    if (unlikely (!sequence.serialize (c, sequences.length))) return_trace (false);

    for (auto& pair : hb_zip (sequences, sequence))
    {
      if (unlikely (!pair.second
		    .serialize_serialize (c, pair.first)))
        return_trace (false);
    }

    return_trace (coverage.serialize_serialize (c, glyphs));
  }

  bool subset (hb_subset_context_t *c) const
  {
    TRACE_SUBSET (this);
    const hb_set_t &glyphset = *c->plan->glyphset_gsub ();
    const hb_map_t &glyph_map = *c->plan->glyph_map;

    auto *out = c->serializer->start_embed (*this);
    if (unlikely (!c->serializer->extend_min (out))) return_trace (false);
    out->format = format;

    hb_sorted_vector_t<hb_codepoint_t> new_coverage;
    + hb_zip (this+coverage, sequence)
    | hb_filter (glyphset, hb_first)
    | hb_filter (subset_offset_array (c, out->sequence, this), hb_second)
    | hb_map (hb_first)
    | hb_map (glyph_map)
    | hb_sink (new_coverage)
    ;
    out->coverage.serialize_serialize (c->serializer, new_coverage.iter ());
    return_trace (bool (new_coverage));
  }
};

}
}
}


#endif /* OT_LAYOUT_GSUB_MULTIPLESUBSTFORMAT1_HH */
