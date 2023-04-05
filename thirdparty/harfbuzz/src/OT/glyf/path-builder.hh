#ifndef OT_GLYF_PATH_BUILDER_HH
#define OT_GLYF_PATH_BUILDER_HH


#include "../../hb.hh"


namespace OT {
namespace glyf_impl {


struct path_builder_t
{
  hb_font_t *font;
  hb_draw_session_t *draw_session;

  struct optional_point_t
  {
    optional_point_t () {}
    optional_point_t (float x_, float y_) : has_data (true), x (x_), y (y_) {}
    operator bool () const { return has_data; }

    bool has_data = false;
    float x = 0.;
    float y = 0.;

    optional_point_t lerp (optional_point_t p, float t)
    { return optional_point_t (x + t * (p.x - x), y + t * (p.y - y)); }
  } first_oncurve, first_offcurve, last_offcurve;

  path_builder_t (hb_font_t *font_, hb_draw_session_t &draw_session_)
  {
    font = font_;
    draw_session = &draw_session_;
    first_oncurve = first_offcurve = last_offcurve = optional_point_t ();
  }

  /* based on https://github.com/RazrFalcon/ttf-parser/blob/4f32821/src/glyf.rs#L287
     See also:
     * https://developer.apple.com/fonts/TrueType-Reference-Manual/RM01/Chap1.html
     * https://stackoverflow.com/a/20772557 */
  void consume_point (const contour_point_t &point)
  {
    bool is_on_curve = point.flag & glyf_impl::SimpleGlyph::FLAG_ON_CURVE;
    optional_point_t p (font->em_fscalef_x (point.x), font->em_fscalef_y (point.y));
    if (!first_oncurve)
    {
      if (is_on_curve)
      {
	first_oncurve = p;
	draw_session->move_to (p.x, p.y);
      }
      else
      {
	if (first_offcurve)
	{
	  optional_point_t mid = first_offcurve.lerp (p, .5f);
	  first_oncurve = mid;
	  last_offcurve = p;
	  draw_session->move_to (mid.x, mid.y);
	}
	else
	  first_offcurve = p;
      }
    }
    else
    {
      if (last_offcurve)
      {
	if (is_on_curve)
	{
	  draw_session->quadratic_to (last_offcurve.x, last_offcurve.y,
				     p.x, p.y);
	  last_offcurve = optional_point_t ();
	}
	else
	{
	  optional_point_t mid = last_offcurve.lerp (p, .5f);
	  draw_session->quadratic_to (last_offcurve.x, last_offcurve.y,
				     mid.x, mid.y);
	  last_offcurve = p;
	}
      }
      else
      {
	if (is_on_curve)
	  draw_session->line_to (p.x, p.y);
	else
	  last_offcurve = p;
      }
    }

    if (point.is_end_point)
    {
      if (first_offcurve && last_offcurve)
      {
	optional_point_t mid = last_offcurve.lerp (first_offcurve, .5f);
	draw_session->quadratic_to (last_offcurve.x, last_offcurve.y,
				   mid.x, mid.y);
	last_offcurve = optional_point_t ();
	/* now check the rest */
      }

      if (first_offcurve && first_oncurve)
	draw_session->quadratic_to (first_offcurve.x, first_offcurve.y,
				   first_oncurve.x, first_oncurve.y);
      else if (last_offcurve && first_oncurve)
	draw_session->quadratic_to (last_offcurve.x, last_offcurve.y,
				   first_oncurve.x, first_oncurve.y);
      else if (first_oncurve)
	draw_session->line_to (first_oncurve.x, first_oncurve.y);
      else if (first_offcurve)
      {
	float x = first_offcurve.x, y = first_offcurve.y;
	draw_session->move_to (x, y);
	draw_session->quadratic_to (x, y, x, y);
      }

      /* Getting ready for the next contour */
      first_oncurve = first_offcurve = last_offcurve = optional_point_t ();
      draw_session->close_path ();
    }
  }
  void points_end () {}

  bool is_consuming_contour_points () { return true; }
  contour_point_t *get_phantoms_sink () { return nullptr; }
};


} /* namespace glyf_impl */
} /* namespace OT */


#endif /* OT_GLYF_PATH_BUILDER_HH */
