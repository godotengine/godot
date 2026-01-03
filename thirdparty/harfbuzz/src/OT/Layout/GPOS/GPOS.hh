#ifndef OT_LAYOUT_GPOS_GPOS_HH
#define OT_LAYOUT_GPOS_GPOS_HH

#include "../../../hb-ot-layout-common.hh"
#include "../../../hb-ot-layout-gsubgpos.hh"
#include "Common.hh"
#include "PosLookup.hh"

namespace OT {

using Layout::GPOS_impl::PosLookup;

namespace Layout {

static void
propagate_attachment_offsets (hb_glyph_position_t *pos,
                              unsigned int len,
                              unsigned int i,
                              hb_direction_t direction,
                              unsigned nesting_level = HB_MAX_NESTING_LEVEL);

/*
 * GPOS -- Glyph Positioning
 * https://docs.microsoft.com/en-us/typography/opentype/spec/gpos
 */

struct GPOS : GSUBGPOS
{
  static constexpr hb_tag_t tableTag = HB_OT_TAG_GPOS;

  using Lookup = PosLookup;

  const PosLookup& get_lookup (unsigned int i) const
  { return static_cast<const PosLookup &> (GSUBGPOS::get_lookup (i)); }

  static inline void position_start (hb_font_t *font, hb_buffer_t *buffer);
  static inline void position_finish_advances (hb_font_t *font, hb_buffer_t *buffer);
  static inline void position_finish_offsets (hb_font_t *font, hb_buffer_t *buffer);

  bool subset (hb_subset_context_t *c) const
  {
    hb_subset_layout_context_t l (c, tableTag);
    return GSUBGPOS::subset<PosLookup> (&l);
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (GSUBGPOS::sanitize<PosLookup> (c));
  }

  HB_INTERNAL bool is_blocklisted (hb_blob_t *blob,
                                   hb_face_t *face) const;

  void collect_variation_indices (hb_collect_variation_indices_context_t *c) const
  {
    for (unsigned i = 0; i < GSUBGPOS::get_lookup_count (); i++)
    {
      if (!c->gpos_lookups->has (i)) continue;
      const PosLookup &l = get_lookup (i);
      l.dispatch (c);
    }
  }

  void closure_lookups (hb_face_t      *face,
                        const hb_set_t *glyphs,
                        hb_set_t       *lookup_indexes /* IN/OUT */) const
  { GSUBGPOS::closure_lookups<PosLookup> (face, glyphs, lookup_indexes); }

  typedef GSUBGPOS::accelerator_t<GPOS> accelerator_t;
};


static void
propagate_attachment_offsets (hb_glyph_position_t *pos,
                              unsigned int len,
                              unsigned int i,
                              hb_direction_t direction,
                              unsigned nesting_level)
{
  /* Adjusts offsets of attached glyphs (both cursive and mark) to accumulate
   * offset of glyph they are attached to. */
  int chain = pos[i].attach_chain();
  int type = pos[i].attach_type();

  pos[i].attach_chain() = 0;

  unsigned int j = (int) i + chain;

  if (unlikely (j >= len))
    return;

  if (unlikely (!nesting_level))
    return;

  if (pos[j].attach_chain())
    propagate_attachment_offsets (pos, len, j, direction, nesting_level - 1);

  assert (!!(type & GPOS_impl::ATTACH_TYPE_MARK) ^ !!(type & GPOS_impl::ATTACH_TYPE_CURSIVE));

  if (type & GPOS_impl::ATTACH_TYPE_CURSIVE)
  {
    if (HB_DIRECTION_IS_HORIZONTAL (direction))
      pos[i].y_offset += pos[j].y_offset;
    else
      pos[i].x_offset += pos[j].x_offset;
  }
  else /*if (type & GPOS_impl::ATTACH_TYPE_MARK)*/
  {
    pos[i].x_offset += pos[j].x_offset;
    pos[i].y_offset += pos[j].y_offset;

    // i is the position of the mark; j is the base.
    if (j < i)
    {
      /* This is the common case: mark follows base.
       * And currently the only way in OpenType. */
      if (HB_DIRECTION_IS_FORWARD (direction))
	for (unsigned int k = j; k < i; k++) {
	  pos[i].x_offset -= pos[k].x_advance;
	  pos[i].y_offset -= pos[k].y_advance;
	}
      else
	for (unsigned int k = j + 1; k < i + 1; k++) {
	  pos[i].x_offset += pos[k].x_advance;
	  pos[i].y_offset += pos[k].y_advance;
	}
    }
    else // j > i
    {
      /* This can happen with `kerx`: a mark attaching
       * to a base after it in the logical order. */
      if (HB_DIRECTION_IS_FORWARD (direction))
	for (unsigned int k = i; k < j; k++) {
	  pos[i].x_offset += pos[k].x_advance;
	  pos[i].y_offset += pos[k].y_advance;
	}
      else
	for (unsigned int k = i + 1; k < j + 1; k++) {
	  pos[i].x_offset -= pos[k].x_advance;
	  pos[i].y_offset -= pos[k].y_advance;
	}
    }
  }
}

void
GPOS::position_start (hb_font_t *font HB_UNUSED, hb_buffer_t *buffer)
{
  unsigned int count = buffer->len;
  for (unsigned int i = 0; i < count; i++)
    buffer->pos[i].attach_chain() = buffer->pos[i].attach_type() = 0;
}

void
GPOS::position_finish_advances (hb_font_t *font HB_UNUSED, hb_buffer_t *buffer HB_UNUSED)
{
  //_hb_buffer_assert_gsubgpos_vars (buffer);
}

void
GPOS::position_finish_offsets (hb_font_t *font, hb_buffer_t *buffer)
{
  _hb_buffer_assert_gsubgpos_vars (buffer);

  unsigned int len;
  hb_glyph_position_t *pos = hb_buffer_get_glyph_positions (buffer, &len);
  hb_direction_t direction = buffer->props.direction;

  /* Handle attachments */
  if (buffer->scratch_flags & HB_BUFFER_SCRATCH_FLAG_HAS_GPOS_ATTACHMENT)
  {
    auto *pos = buffer->pos;
    // https://github.com/harfbuzz/harfbuzz/issues/5514
    if (HB_DIRECTION_IS_FORWARD (direction))
    {
      for (unsigned i = 0; i < len; i++)
	if (pos[i].attach_chain())
	  propagate_attachment_offsets (pos, len, i, direction);
    } else {
      for (unsigned i = len; i-- > 0; )
	if (pos[i].attach_chain())
	  propagate_attachment_offsets (pos, len, i, direction);
    }
  }

  if (unlikely (font->slant_xy) &&
      HB_DIRECTION_IS_HORIZONTAL (direction))
  {
    /* Slanting shaping results is only supported for horizontal text,
     * as it gets weird otherwise. */
    for (unsigned i = 0; i < len; i++)
      if (unlikely (pos[i].y_offset))
        pos[i].x_offset += roundf (font->slant_xy * pos[i].y_offset);
  }
}

}

struct GPOS_accelerator_t : Layout::GPOS::accelerator_t {
  GPOS_accelerator_t (hb_face_t *face) : Layout::GPOS::accelerator_t (face) {}
};

}

#endif  /* OT_LAYOUT_GPOS_GPOS_HH */
