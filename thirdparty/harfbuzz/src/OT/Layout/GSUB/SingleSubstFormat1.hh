#ifndef OT_LAYOUT_GSUB_SINGLESUBSTFORMAT1_HH
#define OT_LAYOUT_GSUB_SINGLESUBSTFORMAT1_HH

#include "Common.hh"

namespace OT {
namespace Layout {
namespace GSUB {

struct SingleSubstFormat1
{
  protected:
  HBUINT16      format;                 /* Format identifier--format = 1 */
  Offset16To<Coverage>
                coverage;               /* Offset to Coverage table--from
                                         * beginning of Substitution table */
  HBUINT16      deltaGlyphID;           /* Add to original GlyphID to get
                                         * substitute GlyphID, modulo 0x10000 */

  public:
  DEFINE_SIZE_STATIC (6);

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (coverage.sanitize (c, this) && deltaGlyphID.sanitize (c));
  }

  bool intersects (const hb_set_t *glyphs) const
  { return (this+coverage).intersects (glyphs); }

  bool may_have_non_1to1 () const
  { return false; }

  void closure (hb_closure_context_t *c) const
  {
    unsigned d = deltaGlyphID;

    + hb_iter (this+coverage)
    | hb_filter (c->parent_active_glyphs ())
    | hb_map ([d] (hb_codepoint_t g) { return (g + d) & 0xFFFFu; })
    | hb_sink (c->output)
    ;

  }

  void closure_lookups (hb_closure_lookups_context_t *c) const {}

  void collect_glyphs (hb_collect_glyphs_context_t *c) const
  {
    if (unlikely (!(this+coverage).collect_coverage (c->input))) return;
    unsigned d = deltaGlyphID;
    + hb_iter (this+coverage)
    | hb_map ([d] (hb_codepoint_t g) { return (g + d) & 0xFFFFu; })
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

    /* According to the Adobe Annotated OpenType Suite, result is always
     * limited to 16bit. */
    glyph_id = (glyph_id + deltaGlyphID) & 0xFFFFu;
    c->replace_glyph (glyph_id);

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

    hb_codepoint_t delta = deltaGlyphID;

    auto it =
    + hb_iter (this+coverage)
    | hb_filter (glyphset)
    | hb_map_retains_sorting ([&] (hb_codepoint_t g) {
                                return hb_codepoint_pair_t (g,
                                                            (g + delta) & 0xFFFF); })
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
