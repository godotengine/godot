#ifndef OT_LAYOUT_GSUB_REVERSECHAINSINGLESUBSTFORMAT1_HH
#define OT_LAYOUT_GSUB_REVERSECHAINSINGLESUBSTFORMAT1_HH

#include "Common.hh"

namespace OT {
namespace Layout {
namespace GSUB_impl {

struct ReverseChainSingleSubstFormat1
{
  protected:
  HBUINT16      format;                 /* Format identifier--format = 1 */
  Offset16To<Coverage>
                coverage;               /* Offset to Coverage table--from
                                         * beginning of table */
  Array16OfOffset16To<Coverage>
                backtrack;              /* Array of coverage tables
                                         * in backtracking sequence, in glyph
                                         * sequence order */
  Array16OfOffset16To<Coverage>
                lookaheadX;             /* Array of coverage tables
                                         * in lookahead sequence, in glyph
                                         * sequence order */
  Array16Of<HBGlyphID16>
                substituteX;            /* Array of substitute
                                         * GlyphIDs--ordered by Coverage Index */
  public:
  DEFINE_SIZE_MIN (10);

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    if (!(coverage.sanitize (c, this) && backtrack.sanitize (c, this)))
      return_trace (false);
    const auto &lookahead = StructAfter<decltype (lookaheadX)> (backtrack);
    if (!lookahead.sanitize (c, this))
      return_trace (false);
    const auto &substitute = StructAfter<decltype (substituteX)> (lookahead);
    return_trace (substitute.sanitize (c));
  }

  bool intersects (const hb_set_t *glyphs) const
  {
    if (!(this+coverage).intersects (glyphs))
      return false;

    const auto &lookahead = StructAfter<decltype (lookaheadX)> (backtrack);

    unsigned int count;

    count = backtrack.len;
    for (unsigned int i = 0; i < count; i++)
      if (!(this+backtrack[i]).intersects (glyphs))
        return false;

    count = lookahead.len;
    for (unsigned int i = 0; i < count; i++)
      if (!(this+lookahead[i]).intersects (glyphs))
        return false;

    return true;
  }

  bool may_have_non_1to1 () const
  { return false; }

  void closure (hb_closure_context_t *c) const
  {
    if (!intersects (c->glyphs)) return;

    const auto &lookahead = StructAfter<decltype (lookaheadX)> (backtrack);
    const auto &substitute = StructAfter<decltype (substituteX)> (lookahead);

    + hb_zip (this+coverage, substitute)
    | hb_filter (c->parent_active_glyphs (), hb_first)
    | hb_map (hb_second)
    | hb_sink (c->output)
    ;
  }

  void closure_lookups (hb_closure_lookups_context_t *c) const {}

  void collect_glyphs (hb_collect_glyphs_context_t *c) const
  {
    if (unlikely (!(this+coverage).collect_coverage (c->input))) return;

    unsigned int count;

    count = backtrack.len;
    for (unsigned int i = 0; i < count; i++)
      if (unlikely (!(this+backtrack[i]).collect_coverage (c->before))) return;

    const auto &lookahead = StructAfter<decltype (lookaheadX)> (backtrack);
    count = lookahead.len;
    for (unsigned int i = 0; i < count; i++)
      if (unlikely (!(this+lookahead[i]).collect_coverage (c->after))) return;

    const auto &substitute = StructAfter<decltype (substituteX)> (lookahead);
    count = substitute.len;
    c->output->add_array (substitute.arrayZ, substitute.len);
  }

  const Coverage &get_coverage () const { return this+coverage; }

  bool would_apply (hb_would_apply_context_t *c) const
  { return c->len == 1 && (this+coverage).get_coverage (c->glyphs[0]) != NOT_COVERED; }

  bool apply (hb_ot_apply_context_t *c) const
  {
    TRACE_APPLY (this);
    if (unlikely (c->nesting_level_left != HB_MAX_NESTING_LEVEL))
      return_trace (false); /* No chaining to this type */

    unsigned int index = (this+coverage).get_coverage (c->buffer->cur ().codepoint);
    if (likely (index == NOT_COVERED)) return_trace (false);

    const auto &lookahead = StructAfter<decltype (lookaheadX)> (backtrack);
    const auto &substitute = StructAfter<decltype (substituteX)> (lookahead);

    if (unlikely (index >= substitute.len)) return_trace (false);

    unsigned int start_index = 0, end_index = 0;
    if (match_backtrack (c,
                         backtrack.len, (HBUINT16 *) backtrack.arrayZ,
                         match_coverage, this,
                         &start_index) &&
        match_lookahead (c,
                         lookahead.len, (HBUINT16 *) lookahead.arrayZ,
                         match_coverage, this,
                         c->buffer->idx + 1, &end_index))
    {
      c->buffer->unsafe_to_break_from_outbuffer (start_index, end_index);

      if (HB_BUFFER_MESSAGE_MORE && c->buffer->messaging ())
      {
	c->buffer->message (c->font,
			    "replacing glyph at %d (reverse chaining substitution)",
			    c->buffer->idx);
      }

      c->replace_glyph_inplace (substitute[index]);

      if (HB_BUFFER_MESSAGE_MORE && c->buffer->messaging ())
      {
	c->buffer->message (c->font,
			    "replaced glyph at %d (reverse chaining substitution)",
			    c->buffer->idx);
      }

      /* Note: We DON'T decrease buffer->idx.  The main loop does it
       * for us.  This is useful for preventing surprises if someone
       * calls us through a Context lookup. */
      return_trace (true);
    }
    else
    {
      c->buffer->unsafe_to_concat_from_outbuffer (start_index, end_index);
      return_trace (false);
    }
  }

  template<typename Iterator,
           hb_requires (hb_is_iterator (Iterator))>
  bool serialize_coverage_offset_array (hb_subset_context_t *c, Iterator it) const
  {
    TRACE_SERIALIZE (this);
    auto *out = c->serializer->start_embed<Array16OfOffset16To<Coverage>> ();

    if (unlikely (!c->serializer->allocate_size<HBUINT16> (HBUINT16::static_size)))
      return_trace (false);

    for (auto& offset : it) {
      auto *o = out->serialize_append (c->serializer);
      if (unlikely (!o) || !o->serialize_subset (c, offset, this))
        return_trace (false);
    }

    return_trace (true);
  }

  template<typename Iterator, typename BacktrackIterator, typename LookaheadIterator,
           hb_requires (hb_is_sorted_source_of (Iterator, hb_codepoint_pair_t)),
           hb_requires (hb_is_iterator (BacktrackIterator)),
           hb_requires (hb_is_iterator (LookaheadIterator))>
  bool serialize (hb_subset_context_t *c,
                  Iterator coverage_subst_iter,
                  BacktrackIterator backtrack_iter,
                  LookaheadIterator lookahead_iter) const
  {
    TRACE_SERIALIZE (this);

    auto *out = c->serializer->start_embed (this);
    if (unlikely (!c->serializer->check_success (out))) return_trace (false);
    if (unlikely (!c->serializer->embed (this->format))) return_trace (false);
    if (unlikely (!c->serializer->embed (this->coverage))) return_trace (false);

    if (!serialize_coverage_offset_array (c, backtrack_iter)) return_trace (false);
    if (!serialize_coverage_offset_array (c, lookahead_iter)) return_trace (false);

    auto *substitute_out = c->serializer->start_embed<Array16Of<HBGlyphID16>> ();
    auto substitutes =
    + coverage_subst_iter
    | hb_map (hb_second)
    ;

    auto glyphs =
    + coverage_subst_iter
    | hb_map_retains_sorting (hb_first)
    ;
    if (unlikely (! c->serializer->check_success (substitute_out->serialize (c->serializer, substitutes))))
        return_trace (false);

    if (unlikely (!out->coverage.serialize_serialize (c->serializer, glyphs)))
      return_trace (false);
    return_trace (true);
  }

  bool subset (hb_subset_context_t *c) const
  {
    TRACE_SUBSET (this);
    const hb_set_t &glyphset = *c->plan->glyphset_gsub ();
    const hb_map_t &glyph_map = *c->plan->glyph_map;

    const auto &lookahead = StructAfter<decltype (lookaheadX)> (backtrack);
    const auto &substitute = StructAfter<decltype (substituteX)> (lookahead);

    auto it =
    + hb_zip (this+coverage, substitute)
    | hb_filter (glyphset, hb_first)
    | hb_filter (glyphset, hb_second)
    | hb_map_retains_sorting ([&] (hb_pair_t<hb_codepoint_t, const HBGlyphID16 &> p) -> hb_codepoint_pair_t
                              { return hb_pair (glyph_map[p.first], glyph_map[p.second]); })
    ;

    return_trace (bool (it) && serialize (c, it, backtrack.iter (), lookahead.iter ()));
  }
};

}
}
}

#endif  /* HB_OT_LAYOUT_GSUB_REVERSECHAINSINGLESUBSTFORMAT1_HH */
