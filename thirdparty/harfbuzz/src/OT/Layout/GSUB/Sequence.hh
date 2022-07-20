#ifndef OT_LAYOUT_GSUB_SEQUENCE_HH
#define OT_LAYOUT_GSUB_SEQUENCE_HH

#include "Common.hh"

namespace OT {
namespace Layout {
namespace GSUB {

struct Sequence
{
  protected:
  Array16Of<HBGlyphID16>
                substitute;             /* String of GlyphIDs to substitute */
  public:
  DEFINE_SIZE_ARRAY (2, substitute);

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (substitute.sanitize (c));
  }

  bool intersects (const hb_set_t *glyphs) const
  { return hb_all (substitute, glyphs); }

  void closure (hb_closure_context_t *c) const
  { c->output->add_array (substitute.arrayZ, substitute.len); }

  void collect_glyphs (hb_collect_glyphs_context_t *c) const
  { c->output->add_array (substitute.arrayZ, substitute.len); }

  bool apply (hb_ot_apply_context_t *c) const
  {
    TRACE_APPLY (this);
    unsigned int count = substitute.len;

    /* Special-case to make it in-place and not consider this
     * as a "multiplied" substitution. */
    if (unlikely (count == 1))
    {
      c->replace_glyph (substitute.arrayZ[0]);
      return_trace (true);
    }
    /* Spec disallows this, but Uniscribe allows it.
     * https://github.com/harfbuzz/harfbuzz/issues/253 */
    else if (unlikely (count == 0))
    {
      c->buffer->delete_glyph ();
      return_trace (true);
    }

    unsigned int klass = _hb_glyph_info_is_ligature (&c->buffer->cur()) ?
                         HB_OT_LAYOUT_GLYPH_PROPS_BASE_GLYPH : 0;
    unsigned lig_id = _hb_glyph_info_get_lig_id (&c->buffer->cur());

    for (unsigned int i = 0; i < count; i++)
    {
      /* If is attached to a ligature, don't disturb that.
       * https://github.com/harfbuzz/harfbuzz/issues/3069 */
      if (!lig_id)
        _hb_glyph_info_set_lig_props_for_component (&c->buffer->cur(), i);
      c->output_glyph_for_component (substitute.arrayZ[i], klass);
    }
    c->buffer->skip_glyph ();

    return_trace (true);
  }

  template <typename Iterator,
            hb_requires (hb_is_source_of (Iterator, hb_codepoint_t))>
  bool serialize (hb_serialize_context_t *c,
                  Iterator subst)
  {
    TRACE_SERIALIZE (this);
    return_trace (substitute.serialize (c, subst));
  }

  bool subset (hb_subset_context_t *c) const
  {
    TRACE_SUBSET (this);
    const hb_set_t &glyphset = *c->plan->glyphset_gsub ();
    const hb_map_t &glyph_map = *c->plan->glyph_map;

    if (!intersects (&glyphset)) return_trace (false);

    auto it =
    + hb_iter (substitute)
    | hb_map (glyph_map)
    ;

    auto *out = c->serializer->start_embed (*this);
    return_trace (out->serialize (c->serializer, it));
  }
};


}
}
}


#endif /* OT_LAYOUT_GSUB_SEQUENCE_HH */
