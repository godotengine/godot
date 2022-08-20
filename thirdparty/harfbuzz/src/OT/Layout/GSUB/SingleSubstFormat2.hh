#ifndef OT_LAYOUT_GSUB_SINGLESUBSTFORMAT2_HH
#define OT_LAYOUT_GSUB_SINGLESUBSTFORMAT2_HH

#include "Common.hh"

namespace OT {
namespace Layout {
namespace GSUB_impl {

template <typename Types>
struct SingleSubstFormat2_4
{
  protected:
  HBUINT16      format;                 /* Format identifier--format = 2 */
  typename Types::template OffsetTo<Coverage>
                coverage;               /* Offset to Coverage table--from
                                         * beginning of Substitution table */
  Array16Of<typename Types::HBGlyphID>
                substitute;             /* Array of substitute
                                         * GlyphIDs--ordered by Coverage Index */

  public:
  DEFINE_SIZE_ARRAY (4 + Types::size, substitute);

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (coverage.sanitize (c, this) && substitute.sanitize (c));
  }

  bool intersects (const hb_set_t *glyphs) const
  { return (this+coverage).intersects (glyphs); }

  bool may_have_non_1to1 () const
  { return false; }

  void closure (hb_closure_context_t *c) const
  {
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
    + hb_zip (this+coverage, substitute)
    | hb_map (hb_second)
    | hb_sink (c->output)
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

    if (unlikely (index >= substitute.len)) return_trace (false);

    if (HB_BUFFER_MESSAGE_MORE && c->buffer->messaging ())
    {
      c->buffer->sync_so_far ();
      c->buffer->message (c->font,
			  "replacing glyph at %d (single substitution)",
			  c->buffer->idx);
    }

    c->replace_glyph (substitute[index]);

    if (HB_BUFFER_MESSAGE_MORE && c->buffer->messaging ())
    {
      c->buffer->message (c->font,
			  "replaced glyph at %d (single substitution)",
			  c->buffer->idx - 1);
    }

    return_trace (true);
  }

  template<typename Iterator,
           hb_requires (hb_is_sorted_source_of (Iterator,
                                                hb_codepoint_pair_t))>
  bool serialize (hb_serialize_context_t *c,
                  Iterator it)
  {
    TRACE_SERIALIZE (this);
    auto substitutes =
      + it
      | hb_map (hb_second)
      ;
    auto glyphs =
      + it
      | hb_map_retains_sorting (hb_first)
      ;
    if (unlikely (!c->extend_min (this))) return_trace (false);
    if (unlikely (!substitute.serialize (c, substitutes))) return_trace (false);
    if (unlikely (!coverage.serialize_serialize (c, glyphs))) return_trace (false);
    return_trace (true);
  }

  bool subset (hb_subset_context_t *c) const
  {
    TRACE_SUBSET (this);
    const hb_set_t &glyphset = *c->plan->glyphset_gsub ();
    const hb_map_t &glyph_map = *c->plan->glyph_map;

    auto it =
    + hb_zip (this+coverage, substitute)
    | hb_filter (glyphset, hb_first)
    | hb_filter (glyphset, hb_second)
    | hb_map_retains_sorting ([&] (hb_pair_t<hb_codepoint_t, const typename Types::HBGlyphID &> p) -> hb_codepoint_pair_t
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

#endif /* OT_LAYOUT_GSUB_SINGLESUBSTFORMAT2_HH */
