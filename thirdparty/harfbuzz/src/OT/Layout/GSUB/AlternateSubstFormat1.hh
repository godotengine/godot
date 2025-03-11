#ifndef OT_LAYOUT_GSUB_ALTERNATESUBSTFORMAT1_HH
#define OT_LAYOUT_GSUB_ALTERNATESUBSTFORMAT1_HH

#include "AlternateSet.hh"
#include "Common.hh"

namespace OT {
namespace Layout {
namespace GSUB_impl {

template <typename Types>
struct AlternateSubstFormat1_2
{
  protected:
  HBUINT16      format;                 /* Format identifier--format = 1 */
  typename Types::template OffsetTo<Coverage>
                coverage;               /* Offset to Coverage table--from
                                         * beginning of Substitution table */
  Array16Of<typename Types::template OffsetTo<AlternateSet<Types>>>
                alternateSet;           /* Array of AlternateSet tables
                                         * ordered by Coverage Index */
  public:
  DEFINE_SIZE_ARRAY (2 + 2 * Types::size, alternateSet);

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (coverage.sanitize (c, this) && alternateSet.sanitize (c, this));
  }

  bool intersects (const hb_set_t *glyphs) const
  { return (this+coverage).intersects (glyphs); }

  bool may_have_non_1to1 () const
  { return false; }

  void closure (hb_closure_context_t *c) const
  {
    + hb_zip (this+coverage, alternateSet)
    | hb_filter (c->parent_active_glyphs (), hb_first)
    | hb_map (hb_second)
    | hb_map (hb_add (this))
    | hb_apply ([c] (const AlternateSet<Types> &_) { _.closure (c); })
    ;
  }

  void closure_lookups (hb_closure_lookups_context_t *c) const {}

  void collect_glyphs (hb_collect_glyphs_context_t *c) const
  {
    if (unlikely (!(this+coverage).collect_coverage (c->input))) return;
    + hb_zip (this+coverage, alternateSet)
    | hb_map (hb_second)
    | hb_map (hb_add (this))
    | hb_apply ([c] (const AlternateSet<Types> &_) { _.collect_glyphs (c); })
    ;
  }

  const Coverage &get_coverage () const { return this+coverage; }

  bool would_apply (hb_would_apply_context_t *c) const
  { return c->len == 1 && (this+coverage).get_coverage (c->glyphs[0]) != NOT_COVERED; }

  unsigned
  get_glyph_alternates (hb_codepoint_t  gid,
                        unsigned        start_offset,
                        unsigned       *alternate_count  /* IN/OUT.  May be NULL. */,
                        hb_codepoint_t *alternate_glyphs /* OUT.     May be NULL. */) const
  { return (this+alternateSet[(this+coverage).get_coverage (gid)])
           .get_alternates (start_offset, alternate_count, alternate_glyphs); }

  bool apply (hb_ot_apply_context_t *c) const
  {
    TRACE_APPLY (this);

    unsigned int index = (this+coverage).get_coverage (c->buffer->cur().codepoint);
    if (index == NOT_COVERED) return_trace (false);

    return_trace ((this+alternateSet[index]).apply (c));
  }

  bool serialize (hb_serialize_context_t *c,
                  hb_sorted_array_t<const HBGlyphID16> glyphs,
                  hb_array_t<const unsigned int> alternate_len_list,
                  hb_array_t<const HBGlyphID16> alternate_glyphs_list)
  {
    TRACE_SERIALIZE (this);
    if (unlikely (!c->extend_min (this))) return_trace (false);
    if (unlikely (!alternateSet.serialize (c, glyphs.length))) return_trace (false);
    for (unsigned int i = 0; i < glyphs.length; i++)
    {
      unsigned int alternate_len = alternate_len_list[i];
      if (unlikely (!alternateSet[i]
                        .serialize_serialize (c, alternate_glyphs_list.sub_array (0, alternate_len))))
        return_trace (false);
      alternate_glyphs_list += alternate_len;
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
    + hb_zip (this+coverage, alternateSet)
    | hb_filter (glyphset, hb_first)
    | hb_filter (subset_offset_array (c, out->alternateSet, this), hb_second)
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

#endif  /* OT_LAYOUT_GSUB_ALTERNATESUBSTFORMAT1_HH */
