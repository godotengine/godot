#ifndef OT_LAYOUT_GPOS_ANCHORFORMAT1_HH
#define OT_LAYOUT_GPOS_ANCHORFORMAT1_HH

namespace OT {
namespace Layout {
namespace GPOS_impl {

struct AnchorFormat1
{
  protected:
  HBUINT16      format;                 /* Format identifier--format = 1 */
  FWORD         xCoordinate;            /* Horizontal value--in design units */
  FWORD         yCoordinate;            /* Vertical value--in design units */
  public:
  DEFINE_SIZE_STATIC (6);

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this));
  }

  void get_anchor (hb_ot_apply_context_t *c, hb_codepoint_t glyph_id HB_UNUSED,
                   float *x, float *y) const
  {
    hb_font_t *font = c->font;
    *x = font->em_fscale_x (xCoordinate);
    *y = font->em_fscale_y (yCoordinate);
  }

  AnchorFormat1* copy (hb_serialize_context_t *c) const
  {
    TRACE_SERIALIZE (this);
    AnchorFormat1* out = c->embed<AnchorFormat1> (this);
    if (!out) return_trace (out);
    out->format = 1;
    return_trace (out);
  }
};


}
}
}

#endif  // OT_LAYOUT_GPOS_ANCHORFORMAT1_HH
