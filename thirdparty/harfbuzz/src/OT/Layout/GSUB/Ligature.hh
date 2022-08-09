#ifndef OT_LAYOUT_GSUB_LIGATURE_HH
#define OT_LAYOUT_GSUB_LIGATURE_HH

#include "Common.hh"

namespace OT {
namespace Layout {
namespace GSUB_impl {

template <typename Types>
struct Ligature
{
  protected:
  typename Types::HBGlyphID
		ligGlyph;               /* GlyphID of ligature to substitute */
  HeadlessArrayOf<typename Types::HBGlyphID>
		component;              /* Array of component GlyphIDs--start
                                         * with the second  component--ordered
                                         * in writing direction */
  public:
  DEFINE_SIZE_ARRAY (Types::size + 2, component);

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (ligGlyph.sanitize (c) && component.sanitize (c));
  }

  bool intersects (const hb_set_t *glyphs) const
  { return hb_all (component, glyphs); }

  void closure (hb_closure_context_t *c) const
  {
    if (!intersects (c->glyphs)) return;
    c->output->add (ligGlyph);
  }

  void collect_glyphs (hb_collect_glyphs_context_t *c) const
  {
    c->input->add_array (component.arrayZ, component.get_length ());
    c->output->add (ligGlyph);
  }

  bool would_apply (hb_would_apply_context_t *c) const
  {
    if (c->len != component.lenP1)
      return false;

    for (unsigned int i = 1; i < c->len; i++)
      if (likely (c->glyphs[i] != component[i]))
        return false;

    return true;
  }

  bool apply (hb_ot_apply_context_t *c) const
  {
    TRACE_APPLY (this);
    unsigned int count = component.lenP1;

    if (unlikely (!count)) return_trace (false);

    /* Special-case to make it in-place and not consider this
     * as a "ligated" substitution. */
    if (unlikely (count == 1))
    {

      if (HB_BUFFER_MESSAGE_MORE && c->buffer->messaging ())
      {
	c->buffer->sync_so_far ();
	c->buffer->message (c->font,
			    "replacing glyph at %d (ligature substitution)",
			    c->buffer->idx);
      }

      c->replace_glyph (ligGlyph);

      if (HB_BUFFER_MESSAGE_MORE && c->buffer->messaging ())
      {
	c->buffer->message (c->font,
			    "replaced glyph at %d (ligature substitution)",
			    c->buffer->idx - 1);
      }

      return_trace (true);
    }

    unsigned int total_component_count = 0;

    unsigned int match_end = 0;
    unsigned int match_positions[HB_MAX_CONTEXT_LENGTH];

    if (likely (!match_input (c, count,
                              &component[1],
                              match_glyph,
                              nullptr,
                              &match_end,
                              match_positions,
                              &total_component_count)))
    {
      c->buffer->unsafe_to_concat (c->buffer->idx, match_end);
      return_trace (false);
    }

    unsigned pos = 0;
    if (HB_BUFFER_MESSAGE_MORE && c->buffer->messaging ())
    {
      unsigned delta = c->buffer->sync_so_far ();

      pos = c->buffer->idx;

      char buf[HB_MAX_CONTEXT_LENGTH * 16] = {0};
      char *p = buf;

      match_end += delta;
      for (unsigned i = 0; i < count; i++)
      {
	match_positions[i] += delta;
	if (i)
	  *p++ = ',';
	sprintf (p, "%u", match_positions[i]);
	p += strlen(p);
      }

      c->buffer->message (c->font,
			  "ligating glyphs at %s",
			  buf);
    }

    ligate_input (c,
                  count,
                  match_positions,
                  match_end,
                  ligGlyph,
                  total_component_count);

    if (HB_BUFFER_MESSAGE_MORE && c->buffer->messaging ())
    {
      c->buffer->sync_so_far ();
      c->buffer->message (c->font,
			  "ligated glyph at %d",
			  pos);
    }

    return_trace (true);
  }

  template <typename Iterator,
            hb_requires (hb_is_source_of (Iterator, hb_codepoint_t))>
  bool serialize (hb_serialize_context_t *c,
                  hb_codepoint_t ligature,
                  Iterator components /* Starting from second */)
  {
    TRACE_SERIALIZE (this);
    if (unlikely (!c->extend_min (this))) return_trace (false);
    ligGlyph = ligature;
    if (unlikely (!component.serialize (c, components))) return_trace (false);
    return_trace (true);
  }

  bool subset (hb_subset_context_t *c, unsigned coverage_idx) const
  {
    TRACE_SUBSET (this);
    const hb_set_t &glyphset = *c->plan->glyphset_gsub ();
    const hb_map_t &glyph_map = *c->plan->glyph_map;

    if (!intersects (&glyphset) || !glyphset.has (ligGlyph)) return_trace (false);
    // Ensure Coverage table is always packed after this.
    c->serializer->add_virtual_link (coverage_idx);

    auto it =
      + hb_iter (component)
      | hb_map (glyph_map)
      ;

    auto *out = c->serializer->start_embed (*this);
    return_trace (out->serialize (c->serializer,
                                  glyph_map[ligGlyph],
                                  it));  }
};


}
}
}

#endif  /* OT_LAYOUT_GSUB_LIGATURE_HH */
