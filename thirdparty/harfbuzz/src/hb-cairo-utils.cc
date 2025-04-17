/*
 * Copyright © 2022  Red Hat, Inc
 * Copyright © 2021, 2022  Black Foundry
 *
 *  This is part of HarfBuzz, a text shaping library.
 *
 * Permission is hereby granted, without written agreement and without
 * license or royalty fees, to use, copy, modify, and distribute this
 * software and its documentation for any purpose, provided that the
 * above copyright notice and the following two paragraphs appear in
 * all copies of this software.
 *
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE TO ANY PARTY FOR
 * DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES
 * ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN
 * IF THE COPYRIGHT HOLDER HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH
 * DAMAGE.
 *
 * THE COPYRIGHT HOLDER SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING,
 * BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE.  THE SOFTWARE PROVIDED HEREUNDER IS
 * ON AN "AS IS" BASIS, AND THE COPYRIGHT HOLDER HAS NO OBLIGATION TO
 * PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
 *
 * Google Author(s): Matthias Clasen
 */

#include "hb.hh"

#ifdef HAVE_CAIRO

#include "hb-cairo-utils.hh"

#include <cairo.h>

/* Some routines in this file were ported from BlackRenderer by Black Foundry.
 * Used by permission to relicense to HarfBuzz license.
 *
 * https://github.com/BlackFoundryCom/black-renderer
 */

#define PREALLOCATED_COLOR_STOPS 16

typedef struct {
  float r, g, b, a;
} hb_cairo_color_t;

static inline cairo_extend_t
hb_cairo_extend (hb_paint_extend_t extend)
{
  switch (extend)
    {
    case HB_PAINT_EXTEND_PAD: return CAIRO_EXTEND_PAD;
    case HB_PAINT_EXTEND_REPEAT: return CAIRO_EXTEND_REPEAT;
    case HB_PAINT_EXTEND_REFLECT: return CAIRO_EXTEND_REFLECT;
    default: break;
    }

  return CAIRO_EXTEND_PAD;
}

#ifdef CAIRO_HAS_PNG_FUNCTIONS
typedef struct
{
  hb_blob_t *blob;
  unsigned int offset;
} hb_cairo_read_blob_data_t;

static cairo_status_t
hb_cairo_read_blob (void *closure,
		    unsigned char *data,
		    unsigned int length)
{
  hb_cairo_read_blob_data_t *r = (hb_cairo_read_blob_data_t *) closure;
  const char *d;
  unsigned int size;

  d = hb_blob_get_data (r->blob, &size);

  if (r->offset + length > size)
    return CAIRO_STATUS_READ_ERROR;

  hb_memcpy (data, d + r->offset, length);
  r->offset += length;

  return CAIRO_STATUS_SUCCESS;
}
#endif

static const cairo_user_data_key_t *_hb_cairo_surface_blob_user_data_key = {0};

static void
_hb_cairo_destroy_blob (void *p)
{
  hb_blob_destroy ((hb_blob_t *) p);
}

hb_bool_t
_hb_cairo_paint_glyph_image (hb_cairo_context_t *c,
			     hb_blob_t *blob,
			     unsigned width,
			     unsigned height,
			     hb_tag_t format,
			     float slant,
			     hb_glyph_extents_t *extents)
{
  cairo_t *cr = c->cr;

  if (!extents) /* SVG currently. */
    return false;

  cairo_surface_t *surface = nullptr;

#ifdef CAIRO_HAS_PNG_FUNCTIONS
  if (format == HB_PAINT_IMAGE_FORMAT_PNG)
  {
    hb_cairo_read_blob_data_t r;
    r.blob = blob;
    r.offset = 0;
    surface = cairo_image_surface_create_from_png_stream (hb_cairo_read_blob, &r);

    /* For PNG, width,height can be unreliable, as is the case for NotoColorEmoji :(.
     * Just pull them out of the surface. */
    width = cairo_image_surface_get_width (surface);
    height = cairo_image_surface_get_width (surface);
  }
  else
#endif
  if (format == HB_PAINT_IMAGE_FORMAT_BGRA)
  {
    /* Byte-endian conversion. */
    unsigned data_size = hb_blob_get_length (blob);
    if (data_size < width * height * 4)
      return false;

    unsigned char *data;
#ifdef __BYTE_ORDER
    if (__BYTE_ORDER == __BIG_ENDIAN)
    {
      data = (unsigned char *) hb_blob_get_data_writable (blob, nullptr);
      if (!data)
        return false;

      unsigned count = width * height * 4;
      for (unsigned i = 0; i < count; i += 4)
      {
        unsigned char b;
	b = data[i];
	data[i] = data[i+3];
	data[i+3] = b;
	b = data[i+1];
	data[i+1] = data[i+2];
	data[i+2] = b;
      }
    }
    else
#endif
      data = (unsigned char *) hb_blob_get_data (blob, nullptr);

    surface = cairo_image_surface_create_for_data (data,
						   CAIRO_FORMAT_ARGB32,
						   width, height,
						   width * 4);

    cairo_surface_set_user_data (surface,
				 _hb_cairo_surface_blob_user_data_key,
				 hb_blob_reference (blob),
				 _hb_cairo_destroy_blob);
  }

  if (!surface)
    return false;

  cairo_save (cr);
  /* this clip is here to work around recording surface limitations */
  cairo_rectangle (cr,
                   extents->x_bearing,
                   extents->y_bearing,
                   extents->width,
                   extents->height);
  cairo_clip (cr);

  cairo_pattern_t *pattern = cairo_pattern_create_for_surface (surface);
  cairo_pattern_set_extend (pattern, CAIRO_EXTEND_PAD);

  cairo_matrix_t matrix = {(double) width, 0, 0, (double) height, 0, 0};
  cairo_pattern_set_matrix (pattern, &matrix);

  /* Undo slant in the extents and apply it in the context. */
  extents->width -= extents->height * slant;
  extents->x_bearing -= extents->y_bearing * slant;
  cairo_matrix_t cairo_matrix = {1., 0., (double) slant, 1., 0., 0.};
  cairo_transform (cr, &cairo_matrix);

  cairo_translate (cr, extents->x_bearing, extents->y_bearing);
  cairo_scale (cr, extents->width, extents->height);
  cairo_set_source (cr, pattern);

  cairo_paint (cr);

  cairo_pattern_destroy (pattern);
  cairo_surface_destroy (surface);

  cairo_restore (cr);

  return true;
}

static void
_hb_cairo_reduce_anchors (float x0, float y0,
			  float x1, float y1,
			  float x2, float y2,
			  float *xx0, float *yy0,
			  float *xx1, float *yy1)
{
  float q1x, q1y, q2x, q2y;
  float s;
  float k;

  q2x = x2 - x0;
  q2y = y2 - y0;
  q1x = x1 - x0;
  q1y = y1 - y0;

  s = q2x * q2x + q2y * q2y;
  if (s < 0.000001f)
    {
      *xx0 = x0; *yy0 = y0;
      *xx1 = x1; *yy1 = y1;
      return;
    }

  k = (q2x * q1x + q2y * q1y) / s;
  *xx0 = x0;
  *yy0 = y0;
  *xx1 = x1 - k * q2x;
  *yy1 = y1 - k * q2y;
}

static int
_hb_cairo_cmp_color_stop (const void *p1,
			  const void *p2)
{
  const hb_color_stop_t *c1 = (const hb_color_stop_t *) p1;
  const hb_color_stop_t *c2 = (const hb_color_stop_t *) p2;

  if (c1->offset < c2->offset)
    return -1;
  else if (c1->offset > c2->offset)
    return 1;
  else
    return 0;
}

static void
_hb_cairo_normalize_color_line (hb_color_stop_t *stops,
				unsigned int len,
				float *omin,
				float *omax)
{
  float min, max;

  hb_qsort (stops, len, sizeof (hb_color_stop_t), _hb_cairo_cmp_color_stop);

  min = max = stops[0].offset;
  for (unsigned int i = 0; i < len; i++)
    {
      min = hb_min (min, stops[i].offset);
      max = hb_max (max, stops[i].offset);
    }

  if (min != max)
    {
      for (unsigned int i = 0; i < len; i++)
        stops[i].offset = (stops[i].offset - min) / (max - min);
    }

  *omin = min;
  *omax = max;
}

static bool
_hb_cairo_get_color_stops (hb_cairo_context_t *c,
			   hb_color_line_t *color_line,
			   unsigned *count,
			   hb_color_stop_t **stops)
{
  unsigned len = hb_color_line_get_color_stops (color_line, 0, nullptr, nullptr);
  if (len > *count)
  {
    *stops = (hb_color_stop_t *) hb_malloc (len * sizeof (hb_color_stop_t));
    if (unlikely (!stops))
      return false;
  }
  hb_color_line_get_color_stops (color_line, 0, &len, *stops);
  for (unsigned i = 0; i < len; i++)
    if ((*stops)[i].is_foreground)
    {
#ifdef HAVE_CAIRO_USER_SCALED_FONT_GET_FOREGROUND_SOURCE
      double r, g, b, a;
      cairo_pattern_t *foreground = cairo_user_scaled_font_get_foreground_source (c->scaled_font);
      if (cairo_pattern_get_rgba (foreground, &r, &g, &b, &a) == CAIRO_STATUS_SUCCESS)
        (*stops)[i].color = HB_COLOR (round (b * 255.), round (g * 255.), round (r * 255.),
                                      round (a * hb_color_get_alpha ((*stops)[i].color)));
      else
#endif
        (*stops)[i].color = HB_COLOR (0, 0, 0, hb_color_get_alpha ((*stops)[i].color));
    }

  *count = len;
  return true;
}

void
_hb_cairo_paint_linear_gradient (hb_cairo_context_t *c,
				 hb_color_line_t *color_line,
				 float x0, float y0,
				 float x1, float y1,
				 float x2, float y2)
{
  cairo_t *cr = c->cr;

  unsigned int len = PREALLOCATED_COLOR_STOPS;
  hb_color_stop_t stops_[PREALLOCATED_COLOR_STOPS];
  hb_color_stop_t *stops = stops_;
  float xx0, yy0, xx1, yy1;
  float xxx0, yyy0, xxx1, yyy1;
  float min, max;
  cairo_pattern_t *pattern;

  if (unlikely (!_hb_cairo_get_color_stops (c, color_line, &len, &stops)))
    return;
  _hb_cairo_normalize_color_line (stops, len, &min, &max);

  _hb_cairo_reduce_anchors (x0, y0, x1, y1, x2, y2, &xx0, &yy0, &xx1, &yy1);

  xxx0 = xx0 + min * (xx1 - xx0);
  yyy0 = yy0 + min * (yy1 - yy0);
  xxx1 = xx0 + max * (xx1 - xx0);
  yyy1 = yy0 + max * (yy1 - yy0);

  pattern = cairo_pattern_create_linear ((double) xxx0, (double) yyy0, (double) xxx1, (double) yyy1);
  cairo_pattern_set_extend (pattern, hb_cairo_extend (hb_color_line_get_extend (color_line)));
  for (unsigned int i = 0; i < len; i++)
    {
      double r, g, b, a;
      r = hb_color_get_red (stops[i].color) / 255.;
      g = hb_color_get_green (stops[i].color) / 255.;
      b = hb_color_get_blue (stops[i].color) / 255.;
      a = hb_color_get_alpha (stops[i].color) / 255.;
      cairo_pattern_add_color_stop_rgba (pattern, (double) stops[i].offset, r, g, b, a);
    }

  cairo_set_source (cr, pattern);
  cairo_paint (cr);

  cairo_pattern_destroy (pattern);

  if (stops != stops_)
    hb_free (stops);
}

void
_hb_cairo_paint_radial_gradient (hb_cairo_context_t *c,
				 hb_color_line_t *color_line,
				 float x0, float y0, float r0,
				 float x1, float y1, float r1)
{
  cairo_t *cr = c->cr;

  unsigned int len = PREALLOCATED_COLOR_STOPS;
  hb_color_stop_t stops_[PREALLOCATED_COLOR_STOPS];
  hb_color_stop_t *stops = stops_;
  float min, max;
  float xx0, yy0, xx1, yy1;
  float rr0, rr1;
  cairo_pattern_t *pattern;

  if (unlikely (!_hb_cairo_get_color_stops (c, color_line, &len, &stops)))
    return;
  _hb_cairo_normalize_color_line (stops, len, &min, &max);

  xx0 = x0 + min * (x1 - x0);
  yy0 = y0 + min * (y1 - y0);
  xx1 = x0 + max * (x1 - x0);
  yy1 = y0 + max * (y1 - y0);
  rr0 = r0 + min * (r1 - r0);
  rr1 = r0 + max * (r1 - r0);

  pattern = cairo_pattern_create_radial ((double) xx0, (double) yy0, (double) rr0, (double) xx1, (double) yy1, (double) rr1);
  cairo_pattern_set_extend (pattern, hb_cairo_extend (hb_color_line_get_extend (color_line)));

  for (unsigned int i = 0; i < len; i++)
    {
      double r, g, b, a;
      r = hb_color_get_red (stops[i].color) / 255.;
      g = hb_color_get_green (stops[i].color) / 255.;
      b = hb_color_get_blue (stops[i].color) / 255.;
      a = hb_color_get_alpha (stops[i].color) / 255.;
      cairo_pattern_add_color_stop_rgba (pattern, (double) stops[i].offset, r, g, b, a);
    }

  cairo_set_source (cr, pattern);
  cairo_paint (cr);

  cairo_pattern_destroy (pattern);

  if (stops != stops_)
    hb_free (stops);
}

typedef struct {
  float x, y;
} hb_cairo_point_t;

static inline float
_hb_cairo_interpolate (float f0, float f1, float f)
{
  return f0 + f * (f1 - f0);
}

static inline void
_hb_cairo_premultiply (hb_cairo_color_t *c)
{
  c->r *= c->a;
  c->g *= c->a;
  c->b *= c->a;
}

static inline void
_hb_cairo_unpremultiply (hb_cairo_color_t *c)
{
  if (c->a != 0.f)
  {
     c->r /= c->a;
     c->g /= c->a;
     c->b /= c->a;
  }
}

static void
_hb_cairo_interpolate_colors (hb_cairo_color_t *c0, hb_cairo_color_t *c1, float k, hb_cairo_color_t *c)
{
  // According to the COLR specification, gradients
  // should be interpolated in premultiplied form
  _hb_cairo_premultiply (c0);
  _hb_cairo_premultiply (c1);
  c->r = c0->r + k * (c1->r - c0->r);
  c->g = c0->g + k * (c1->g - c0->g);
  c->b = c0->b + k * (c1->b - c0->b);
  c->a = c0->a + k * (c1->a - c0->a);
  _hb_cairo_unpremultiply (c);
}

static inline float
_hb_cairo_dot (hb_cairo_point_t p, hb_cairo_point_t q)
{
  return p.x * q.x + p.y * q.y;
}

static inline hb_cairo_point_t
_hb_cairo_normalize (hb_cairo_point_t p)
{
  float len = sqrtf (_hb_cairo_dot (p, p));

  return hb_cairo_point_t { p.x / len, p.y / len };
}

static inline hb_cairo_point_t
_hb_cairo_sum (hb_cairo_point_t p, hb_cairo_point_t q)
{
  return hb_cairo_point_t { p.x + q.x, p.y + q.y };
}

static inline hb_cairo_point_t
_hb_cairo_difference (hb_cairo_point_t p, hb_cairo_point_t q)
{
  return hb_cairo_point_t { p.x - q.x, p.y - q.y };
}

static inline hb_cairo_point_t
_hb_cairo_scale (hb_cairo_point_t p, float f)
{
  return hb_cairo_point_t { p.x * f, p.y * f };
}

typedef struct {
  hb_cairo_point_t center, p0, c0, c1, p1;
  hb_cairo_color_t color0, color1;
} hb_cairo_patch_t;

static void
_hb_cairo_add_patch (cairo_pattern_t *pattern, hb_cairo_point_t *center, hb_cairo_patch_t *p)
{
  cairo_mesh_pattern_begin_patch (pattern);
  cairo_mesh_pattern_move_to (pattern, (double) center->x, (double) center->y);
  cairo_mesh_pattern_line_to (pattern, (double) p->p0.x, (double) p->p0.y);
  cairo_mesh_pattern_curve_to (pattern,
                               (double) p->c0.x, (double) p->c0.y,
                               (double) p->c1.x, (double) p->c1.y,
                               (double) p->p1.x, (double) p->p1.y);
  cairo_mesh_pattern_line_to (pattern, (double) center->x, (double) center->y);
  cairo_mesh_pattern_set_corner_color_rgba (pattern, 0,
                                            (double) p->color0.r,
                                            (double) p->color0.g,
                                            (double) p->color0.b,
                                            (double) p->color0.a);
  cairo_mesh_pattern_set_corner_color_rgba (pattern, 1,
                                            (double) p->color0.r,
                                            (double) p->color0.g,
                                            (double) p->color0.b,
                                            (double) p->color0.a);
  cairo_mesh_pattern_set_corner_color_rgba (pattern, 2,
                                            (double) p->color1.r,
                                            (double) p->color1.g,
                                            (double) p->color1.b,
                                            (double) p->color1.a);
  cairo_mesh_pattern_set_corner_color_rgba (pattern, 3,
                                            (double) p->color1.r,
                                            (double) p->color1.g,
                                            (double) p->color1.b,
                                            (double) p->color1.a);
  cairo_mesh_pattern_end_patch (pattern);
}

#define MAX_ANGLE (HB_PI / 8.f)

static void
_hb_cairo_add_sweep_gradient_patches1 (float cx, float cy, float radius,
				       float a0, hb_cairo_color_t *c0,
				       float a1, hb_cairo_color_t *c1,
				       cairo_pattern_t *pattern)
{
  hb_cairo_point_t center = hb_cairo_point_t { cx, cy };
  int num_splits;
  hb_cairo_point_t p0;
  hb_cairo_color_t color0, color1;

  num_splits = ceilf (fabsf (a1 - a0) / MAX_ANGLE);
  p0 = hb_cairo_point_t { cosf (a0), sinf (a0) };
  color0 = *c0;

  for (int a = 0; a < num_splits; a++)
    {
      float k = (a + 1.) / num_splits;
      float angle1;
      hb_cairo_point_t p1;
      hb_cairo_point_t A, U;
      hb_cairo_point_t C0, C1;
      hb_cairo_patch_t patch;

      angle1 = _hb_cairo_interpolate (a0, a1, k);
      _hb_cairo_interpolate_colors (c0, c1, k, &color1);

      patch.color0 = color0;
      patch.color1 = color1;

      p1 = hb_cairo_point_t { cosf (angle1), sinf (angle1) };
      patch.p0 = _hb_cairo_sum (center, _hb_cairo_scale (p0, radius));
      patch.p1 = _hb_cairo_sum (center, _hb_cairo_scale (p1, radius));

      A = _hb_cairo_normalize (_hb_cairo_sum (p0, p1));
      U = hb_cairo_point_t { -A.y, A.x };
      C0 = _hb_cairo_sum (A, _hb_cairo_scale (U, _hb_cairo_dot (_hb_cairo_difference (p0, A), p0) / _hb_cairo_dot (U, p0)));
      C1 = _hb_cairo_sum (A, _hb_cairo_scale (U, _hb_cairo_dot (_hb_cairo_difference (p1, A), p1) / _hb_cairo_dot (U, p1)));

      patch.c0 = _hb_cairo_sum (center, _hb_cairo_scale (_hb_cairo_sum (C0, _hb_cairo_scale (_hb_cairo_difference (C0, p0), 0.33333f)), radius));
      patch.c1 = _hb_cairo_sum (center, _hb_cairo_scale (_hb_cairo_sum (C1, _hb_cairo_scale (_hb_cairo_difference (C1, p1), 0.33333f)), radius));

      _hb_cairo_add_patch (pattern, &center, &patch);

      p0 = p1;
      color0 = color1;
    }
}

static void
_hb_cairo_add_sweep_gradient_patches (hb_color_stop_t *stops,
				      unsigned int n_stops,
				      cairo_extend_t extend,
				      float cx, float cy,
				      float radius,
				      float start_angle,
				      float end_angle,
				      cairo_pattern_t *pattern)
{
  float angles_[PREALLOCATED_COLOR_STOPS];
  float *angles = angles_;
  hb_cairo_color_t colors_[PREALLOCATED_COLOR_STOPS];
  hb_cairo_color_t *colors = colors_;
  hb_cairo_color_t color0, color1;

  if (start_angle == end_angle)
  {
    if (extend == CAIRO_EXTEND_PAD)
    {
      hb_cairo_color_t c;
      if (start_angle > 0)
      {
	c.r = hb_color_get_red (stops[0].color) / 255.;
	c.g = hb_color_get_green (stops[0].color) / 255.;
	c.b = hb_color_get_blue (stops[0].color) / 255.;
	c.a = hb_color_get_alpha (stops[0].color) / 255.;
	_hb_cairo_add_sweep_gradient_patches1 (cx, cy, radius,
					       0.,          &c,
					       start_angle, &c,
					       pattern);
      }
      if (end_angle < HB_2_PI)
      {
	c.r = hb_color_get_red (stops[n_stops - 1].color) / 255.;
	c.g = hb_color_get_green (stops[n_stops - 1].color) / 255.;
	c.b = hb_color_get_blue (stops[n_stops - 1].color) / 255.;
	c.a = hb_color_get_alpha (stops[n_stops - 1].color) / 255.;
	_hb_cairo_add_sweep_gradient_patches1 (cx, cy, radius,
					       end_angle, &c,
					       HB_2_PI,  &c,
					       pattern);
      }
    }
    return;
  }

  assert (start_angle != end_angle);

  /* handle directions */
  if (end_angle < start_angle)
  {
    hb_swap (start_angle, end_angle);

    for (unsigned i = 0; i < n_stops - 1 - i; i++)
      hb_swap (stops[i], stops[n_stops - 1 - i]);
    for (unsigned i = 0; i < n_stops; i++)
      stops[i].offset = 1 - stops[i].offset;
  }

  if (n_stops > PREALLOCATED_COLOR_STOPS)
  {
    angles = (float *) hb_malloc (sizeof (float) * n_stops);
    colors = (hb_cairo_color_t *) hb_malloc (sizeof (hb_cairo_color_t) * n_stops);
    if (unlikely (!angles || !colors))
    {
      hb_free (angles);
      hb_free (colors);
      return;
    }
  }

  for (unsigned i = 0; i < n_stops; i++)
  {
    angles[i] = start_angle + stops[i].offset * (end_angle - start_angle);
    colors[i].r = hb_color_get_red (stops[i].color) / 255.;
    colors[i].g = hb_color_get_green (stops[i].color) / 255.;
    colors[i].b = hb_color_get_blue (stops[i].color) / 255.;
    colors[i].a = hb_color_get_alpha (stops[i].color) / 255.;
  }

  if (extend == CAIRO_EXTEND_PAD)
  {
    unsigned pos;

    color0 = colors[0];
    for (pos = 0; pos < n_stops; pos++)
    {
      if (angles[pos] >= 0)
      {
	if (pos > 0)
	{
	  float k = (0 - angles[pos - 1]) / (angles[pos] - angles[pos - 1]);
	  _hb_cairo_interpolate_colors (&colors[pos-1], &colors[pos], k, &color0);
	}
	break;
      }
    }
    if (pos == n_stops)
    {
      /* everything is below 0 */
      color0 = colors[n_stops-1];
      _hb_cairo_add_sweep_gradient_patches1 (cx, cy, radius,
					     0.,       &color0,
					     HB_2_PI, &color0,
					     pattern);
      goto done;
    }

    _hb_cairo_add_sweep_gradient_patches1 (cx, cy, radius,
					   0.,          &color0,
					   angles[pos], &colors[pos],
					   pattern);

    for (pos++; pos < n_stops; pos++)
    {
      if (angles[pos] <= HB_2_PI)
      {
	_hb_cairo_add_sweep_gradient_patches1 (cx, cy, radius,
					       angles[pos - 1], &colors[pos-1],
					       angles[pos],     &colors[pos],
					       pattern);
      }
      else
      {
	float k = (HB_2_PI - angles[pos - 1]) / (angles[pos] - angles[pos - 1]);
	_hb_cairo_interpolate_colors (&colors[pos - 1], &colors[pos], k, &color1);
	_hb_cairo_add_sweep_gradient_patches1 (cx, cy, radius,
					       angles[pos - 1], &colors[pos - 1],
					       HB_2_PI,        &color1,
					       pattern);
	break;
      }
    }

    if (pos == n_stops)
    {
      /* everything is below 2*M_PI */
      color0 = colors[n_stops - 1];
      _hb_cairo_add_sweep_gradient_patches1 (cx, cy, radius,
					     angles[n_stops - 1], &color0,
					     HB_2_PI,            &color0,
					     pattern);
      goto done;
    }
  }
  else
  {
    int k;
    float span;

    span = angles[n_stops - 1] - angles[0];
    if (!span)
      goto done;

    k = 0;
    if (angles[0] >= 0)
    {
      float ss = angles[0];
      while (ss > 0)
      {
	if (span > 0)
	{
	  ss -= span;
	  k--;
	}
	else
	{
	  ss += span;
	  k++;
	}
      }
    }
    else if (angles[0] < 0)
    {
      float ee = angles[n_stops - 1];
      while (ee < 0)
      {
	if (span > 0)
	{
	  ee += span;
	  k++;
	}
	else
	{
	  ee -= span;
	  k--;
	}
      }
    }

    //assert (angles[0] + k * span <= 0 && 0 < angles[n_stops - 1] + k * span);
    span = fabsf (span);

    for (signed l = k; l < 1000; l++)
    {
      for (unsigned i = 1; i < n_stops; i++)
      {
        float a0, a1;
	hb_cairo_color_t *c0, *c1;

	if ((l % 2 != 0) && (extend == CAIRO_EXTEND_REFLECT))
	{
	  a0 = angles[0] + angles[n_stops - 1] - angles[n_stops - 1 - (i-1)] + l * span;
	  a1 = angles[0] + angles[n_stops - 1] - angles[n_stops - 1 - i] + l * span;
	  c0 = &colors[n_stops - 1 - (i - 1)];
	  c1 = &colors[n_stops - 1 - i];
	}
	else
	{
	  a0 = angles[i-1] + l * span;
	  a1 = angles[i] + l * span;
	  c0 = &colors[i-1];
	  c1 = &colors[i];
	}

	if (a1 < 0)
	  continue;
	if (a0 < 0)
	{
	  hb_cairo_color_t color;
	  float f = (0 - a0)/(a1 - a0);
	  _hb_cairo_interpolate_colors (c0, c1, f, &color);
	  _hb_cairo_add_sweep_gradient_patches1 (cx, cy, radius,
						 0,  &color,
						 a1, c1,
						 pattern);
	}
	else if (a1 >= HB_2_PI)
	{
	  hb_cairo_color_t color;
	  float f = (HB_2_PI - a0)/(a1 - a0);
	  _hb_cairo_interpolate_colors (c0, c1, f, &color);
	  _hb_cairo_add_sweep_gradient_patches1 (cx, cy, radius,
						 a0,       c0,
						 HB_2_PI, &color,
						 pattern);
	  goto done;
	}
	else
	{
	  _hb_cairo_add_sweep_gradient_patches1 (cx, cy, radius,
						 a0, c0,
						 a1, c1,
						 pattern);
	}
      }
    }
  }

done:

  if (angles != angles_)
    hb_free (angles);
  if (colors != colors_)
    hb_free (colors);
}

void
_hb_cairo_paint_sweep_gradient (hb_cairo_context_t *c,
				hb_color_line_t *color_line,
				float cx, float cy,
				float start_angle,
				float end_angle)
{
  cairo_t *cr = c->cr;

  unsigned int len = PREALLOCATED_COLOR_STOPS;
  hb_color_stop_t stops_[PREALLOCATED_COLOR_STOPS];
  hb_color_stop_t *stops = stops_;
  cairo_extend_t extend;
  double x1, y1, x2, y2;
  float max_x, max_y, radius;
  cairo_pattern_t *pattern;

  if (unlikely (!_hb_cairo_get_color_stops (c, color_line, &len, &stops)))
    return;

  hb_qsort (stops, len, sizeof (hb_color_stop_t), _hb_cairo_cmp_color_stop);

  cairo_clip_extents (cr, &x1, &y1, &x2, &y2);
  max_x = (float) hb_max ((x1 - (double) cx) * (x1 - (double) cx), (x2 - (double) cx) * (x2 - (double) cx));
  max_y = (float) hb_max ((y1 - (double) cy) * (y1 - (double) cy), (y2 - (double) cy) * (y2 - (double) cy));
  radius = sqrtf (max_x + max_y);

  extend = hb_cairo_extend (hb_color_line_get_extend (color_line));
  pattern = cairo_pattern_create_mesh ();

  _hb_cairo_add_sweep_gradient_patches (stops, len, extend, cx, cy,
					radius, start_angle, end_angle, pattern);

  cairo_set_source (cr, pattern);
  cairo_paint (cr);

  cairo_pattern_destroy (pattern);

  if (stops != stops_)
    hb_free (stops);
}

#endif
