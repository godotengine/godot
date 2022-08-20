#ifndef OT_LAYOUT_GSUB_SINGLESUBSTFORMAT1_HH
#define OT_LAYOUT_GSUB_SINGLESUBSTFORMAT1_HH

#include "Common.hh"

namespace OT {
namespace Layout {
namespace GSUB_impl {

template <typename Types>
struct SingleSubstFormat1_3
{
  protected:
  HBUINT16      format;                 /* Format identifier--format = 1 */
  typename Types::template OffsetTo<Coverage>
                coverage;               /* Offset to Coverage table--from
                                         * beginning of Substitution table */
  typename Types::HBUINT
                deltaGlyphID;           /* Add to original GlyphID to get
                                         * substitute GlyphID, modulo 0x10000 */

  public:
  DEFINE_SIZE_STATIC (2 + 2 * Types::size);

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (coverage.sanitize (c, this) && deltaGlyphID.sanitize (c));
  }

  hb_codepoint_t get_mask () const
  { return (1 << (8 * Types::size)) - 1; }

  bool intersects (const hb_set_t *glyphs) const
  { return (this+coverage).intersects (glyphs); }

  bool may_have_non_1to1 () const
  { return false; }

  void closure (hb_closure_context_t *c) const
  {
    hb_codepoint_t d = deltaGlyphID;
    hb_codepoint_t mask = get_mask ();

    hb_set_t intersection;
    (this+coverage).intersect_set (c->parent_active_glyphs (), intersection);

    /* In degenerate fuzzer-found fonts, but not real fonts,
     * this table can keep adding new glyphs in each round of closure.
     * Refuse to close-over, if it maps glyph range to overlapping range. */
    hb_codepoint_t min_before = intersection.get_min ();
    hb_codepoint_t max_before = intersection.get_max ();
    hb_codepoint_t min_after = (min_before + d) & mask;
    hb_codepoint_t max_after = (max_before + d) & mask;
    if ((this+coverage).get_population () >= max_before - min_before &&
	((min_before <= min_after && min_after <= max_before) ||
	 (min_before <= max_after && max_after <= max_before)))
      return;

    + hb_iter (intersection)
    | hb_map ([d, mask] (hb_codepoint_t g) { return (g + d) & mask; })
    | hb_sink (c->output)
    ;
  }

  void closure_lookups (hb_closure_lookups_context_t *c) const {}

  void collect_glyphs (hb_collect_glyphs_context_t *c) const
  {
    if (unlikely (!(this+coverage).collect_coverage (c->input))) return;
    hb_codepoint_t d = deltaGlyphID;
    hb_codepoint_t mask = get_mask ();

    + hb_iter (this+coverage)
    | hb_map ([d, mask] (hb_codepoint_t g) { return (g + d) & mask; })
    | hb_sink (c->output)
    ;
  }

  const Coverage &get_coverage () const { return this+coverage; }

  bool would_apply (hb_would_apply_context_t *c) const
  { return c->len == 1 && (this+coverage).get_coverage (c->glyphs[0]) != NOT_COVERED; }

  bool apply (hb_ot_apply_context_t *c) const
  {
    TRACE_APPLY (this);
    hb_codepoint_t glyph_id = c->buffer->cur().codepoint;
    unsigned int index = (this+coverage).get_coverage (glyph_id);
    if (likely (index == NOT_COVERED)) return_trace (false);

    hb_codepoint_t d = deltaGlyphID;
    hb_codepoint_t mask = get_mask ();

    glyph_id = (glyph_id + d) & mask;

    if (HB_BUFFER_MESSAGE_MORE && c->buffer->messaging ())
    {
      c->buffer->sync_so_far ();
      c->buffer->message (c->font,
			  "replacing glyph at %d (single substitution)",
			  c->buffer->idx);
    }

    c->replace_glyph (glyph_id);

    if (HB_BUFFER_MESSAGE_MORE && c->buffer->messaging ())
    {
      c->buffer->message (c->font,
			  "replaced glyph at %d (single substitution)",
			  c->buffer->idx - 1);
    }

    return_trace (true);
  }

  template<typename Iterator,
           hb_requires (hb_is_sorted_source_of (Iterator, hb_codepoint_t))>
  bool serialize (hb_serialize_context_t *c,
                  Iterator glyphs,
                  unsigned delta)
  {
    TRACE_SERIALIZE (this);
    if (unlikely (!c->extend_min (this))) return_trace (false);
    if (unlikely (!coverage.serialize_serialize (c, glyphs))) return_trace (false);
    c->check_assign (deltaGlyphID, delta, HB_SERIALIZE_ERROR_INT_OVERFLOW);
    return_trace (true);
  }

  bool subset (hb_subset_context_t *c) const
  {
    TRACE_SUBSET (this);
    const hb_set_t &glyphset = *c->plan->glyphset_gsub ();
    const hb_map_t &glyph_map = *c->plan->glyph_map;

    hb_codepoint_t d = deltaGlyphID;
    hb_codepoint_t mask = get_mask ();

    hb_set_t intersection;
    (this+coverage).intersect_set (glyphset, intersection);

    auto it =
    + hb_iter (intersection)
    | hb_map_retains_sorting ([d, mask] (hb_codepoint_t g) {
                                return hb_codepoint_pair_t (g,
                                                            (g + d) & mask); })
    | hb_filter (glyphset, hb_second)
    | hb_map_retains_sorting ([&] (hb_codepoint_pair_t p) -> hb_codepoint_pair_t
                              { return hb_pair (glyph_map[p.first], glyph_map[p.second]); })
    ;

    bool ret = bool (it);
    SingleSubst_serialize (c->serializer, it);
    return_trace (ret);
  }
};

}
}
}


#endif /* OT_LAYOUT_GSUB_SINGLESUBSTFORMAT1_HH */
