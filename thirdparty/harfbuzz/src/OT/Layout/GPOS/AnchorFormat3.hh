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
    return_trace (c->check_struct (this) && xDeviceTable.sanitize (c, this) && yDeviceTable.sanitize (c, this));
  }

  void get_anchor (hb_ot_apply_context_t *c, hb_codepoint_t glyph_id HB_UNUSED,
                   float *x, float *y) const
  {
    hb_font_t *font = c->font;
    *x = font->em_fscale_x (xCoordinate);
    *y = font->em_fscale_y (yCoordinate);

    if (font->x_ppem || font->num_coords)
      *x += (this+xDeviceTable).get_x_delta (font, c->var_store, c->var_store_cache);
    if (font->y_ppem || font->num_coords)
      *y += (this+yDeviceTable).get_y_delta (font, c->var_store, c->var_store_cache);
  }

  AnchorFormat3* copy (hb_serialize_context_t *c,
                       const hb_map_t *layout_variation_idx_map) const
  {
    TRACE_SERIALIZE (this);
    if (!layout_variation_idx_map) return_trace (nullptr);

    auto *out = c->embed<AnchorFormat3> (this);
    if (unlikely (!out)) return_trace (nullptr);

    out->xDeviceTable.serialize_copy (c, xDeviceTable, this, 0, hb_serialize_context_t::Head, layout_variation_idx_map);
    out->yDeviceTable.serialize_copy (c, yDeviceTable, this, 0, hb_serialize_context_t::Head, layout_variation_idx_map);
    return_trace (out);
  }

  void collect_variation_indices (hb_collect_variation_indices_context_t *c) const
  {
    (this+xDeviceTable).collect_variation_indices (c->layout_variation_indices);
    (this+yDeviceTable).collect_variation_indices (c->layout_variation_indices);
  }
};


}
}
}

#endif  // OT_LAYOUT_GPOS_ANCHORFORMAT3_HH
