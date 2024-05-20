#ifndef OT_LAYOUT_GPOS_ANCHORFORMAT2_HH
#define OT_LAYOUT_GPOS_ANCHORFORMAT2_HH

namespace OT {
namespace Layout {
namespace GPOS_impl {

struct AnchorFormat2
{

  protected:
  HBUINT16      format;                 /* Format identifier--format = 2 */
  FWORD         xCoordinate;            /* Horizontal value--in design units */
  FWORD         yCoordinate;            /* Vertical value--in design units */
  HBUINT16      anchorPoint;            /* Index to glyph contour point */
  public:
  DEFINE_SIZE_STATIC (8);

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this));
  }

  void get_anchor (hb_ot_apply_context_t *c, hb_codepoint_t glyph_id,
                   float *x, float *y) const
  {
    hb_font_t *font = c->font;

#ifdef HB_NO_HINTING
    *x = font->em_fscale_x (xCoordinate);
    *y = font->em_fscale_y (yCoordinate);
    return;
#endif

    unsigned int x_ppem = font->x_ppem;
    unsigned int y_ppem = font->y_ppem;
    hb_position_t cx = 0, cy = 0;
    bool ret;

    ret = (x_ppem || y_ppem) &&
          font->get_glyph_contour_point_for_origin (glyph_id, anchorPoint, HB_DIRECTION_LTR, &cx, &cy);
    *x = ret && x_ppem ? cx : font->em_fscale_x (xCoordinate);
    *y = ret && y_ppem ? cy : font->em_fscale_y (yCoordinate);
  }

  AnchorFormat2* copy (hb_serialize_context_t *c) const
  {
    TRACE_SERIALIZE (this);
    return_trace (c->embed<AnchorFormat2> (this));
  }
};

}
}
}

#endif  // OT_LAYOUT_GPOS_ANCHORFORMAT2_HH
