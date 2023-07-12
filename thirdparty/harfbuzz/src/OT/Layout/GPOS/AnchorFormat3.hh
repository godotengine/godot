#ifndef OT_LAYOUT_GPOS_ANCHORFORMAT3_HH
#define OT_LAYOUT_GPOS_ANCHORFORMAT3_HH

namespace OT {
namespace Layout {
namespace GPOS_impl {

struct AnchorFormat3
{
  protected:
  HBUINT16      format;                 /* Format identifier--format = 3 */
  FWORD         xCoordinate;            /* Horizontal value--in design units */
  FWORD         yCoordinate;            /* Vertical value--in design units */
  Offset16To<Device>
                xDeviceTable;           /* Offset to Device table for X
                                         * coordinate-- from beginning of
                                         * Anchor table (may be NULL) */
  Offset16To<Device>
                yDeviceTable;           /* Offset to Device table for Y
                                         * coordinate-- from beginning of
                                         * Anchor table (may be NULL) */
  public:
  DEFINE_SIZE_STATIC (10);

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    if (unlikely (!c->check_struct (this))) return_trace (false);

    return_trace (xDeviceTable.sanitize (c, this) && yDeviceTable.sanitize (c, this));
  }

  void get_anchor (hb_ot_apply_context_t *c, hb_codepoint_t glyph_id HB_UNUSED,
                   float *x, float *y) const
  {
    hb_font_t *font = c->font;
    *x = font->em_fscale_x (xCoordinate);
    *y = font->em_fscale_y (yCoordinate);

    if ((font->x_ppem || font->num_coords) && xDeviceTable.sanitize (&c->sanitizer, this))
      *x += (this+xDeviceTable).get_x_delta (font, c->var_store, c->var_store_cache);
    if ((font->y_ppem || font->num_coords) && yDeviceTable.sanitize (&c->sanitizer, this))
      *y += (this+yDeviceTable).get_y_delta (font, c->var_store, c->var_store_cache);
  }

  bool subset (hb_subset_context_t *c) const
  {
    TRACE_SUBSET (this);
    auto *out = c->serializer->start_embed (*this);
    if (unlikely (!c->serializer->embed (format))) return_trace (false);
    if (unlikely (!c->serializer->embed (xCoordinate))) return_trace (false);
    if (unlikely (!c->serializer->embed (yCoordinate))) return_trace (false);

    unsigned x_varidx = xDeviceTable ? (this+xDeviceTable).get_variation_index () : HB_OT_LAYOUT_NO_VARIATIONS_INDEX;
    if (c->plan->layout_variation_idx_delta_map.has (x_varidx))
    {
      int delta = hb_second (c->plan->layout_variation_idx_delta_map.get (x_varidx));
      if (delta != 0)
      {
        if (!c->serializer->check_assign (out->xCoordinate, xCoordinate + delta,
                                          HB_SERIALIZE_ERROR_INT_OVERFLOW))
          return_trace (false);
      }
    }

    unsigned y_varidx = yDeviceTable ? (this+yDeviceTable).get_variation_index () : HB_OT_LAYOUT_NO_VARIATIONS_INDEX;
    if (c->plan->layout_variation_idx_delta_map.has (y_varidx))
    {
      int delta = hb_second (c->plan->layout_variation_idx_delta_map.get (y_varidx));
      if (delta != 0)
      {
        if (!c->serializer->check_assign (out->yCoordinate, yCoordinate + delta,
                                          HB_SERIALIZE_ERROR_INT_OVERFLOW))
          return_trace (false);
      }
    }

    if (c->plan->all_axes_pinned)
      return_trace (c->serializer->check_assign (out->format, 1, HB_SERIALIZE_ERROR_INT_OVERFLOW));

    if (!c->serializer->embed (xDeviceTable)) return_trace (false);
    if (!c->serializer->embed (yDeviceTable)) return_trace (false);

    out->xDeviceTable.serialize_copy (c->serializer, xDeviceTable, this, 0, hb_serialize_context_t::Head, &c->plan->layout_variation_idx_delta_map);
    out->yDeviceTable.serialize_copy (c->serializer, yDeviceTable, this, 0, hb_serialize_context_t::Head, &c->plan->layout_variation_idx_delta_map);
    return_trace (out);
  }

  void collect_variation_indices (hb_collect_variation_indices_context_t *c) const
  {
    (this+xDeviceTable).collect_variation_indices (c);
    (this+yDeviceTable).collect_variation_indices (c);
  }
};


}
}
}

#endif  // OT_LAYOUT_GPOS_ANCHORFORMAT3_HH
