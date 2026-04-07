/*
 * Copyright © 2026  Behdad Esfahbod
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
 * Author(s): Behdad Esfahbod
 */

#include "hb.hh"

#include "hb-raster-paint.hh"
#include "hb-raster-svg.hh"
#include "hb-machinery.hh"

#include <math.h>


/*
 * Pixel helpers (paint-specific)
 */

/* Convert unpremultiplied hb_color_t (BGRA order) to premultiplied BGRA32 pixel. */
static inline uint32_t
color_to_premul_pixel (hb_color_t color)
{
  uint8_t a = hb_color_get_alpha (color);
  uint8_t r = hb_raster_div255 (hb_color_get_red (color) * a);
  uint8_t g = hb_raster_div255 (hb_color_get_green (color) * a);
  uint8_t b = hb_raster_div255 (hb_color_get_blue (color) * a);
  return (uint32_t) b | ((uint32_t) g << 8) | ((uint32_t) r << 16) | ((uint32_t) a << 24);
}

/* Bilinear sample from a premultiplied BGRA32 image. */
static inline uint32_t
hb_raster_sample_bilinear_premul (const hb_packed_t<uint32_t> *src,
				  unsigned width,
				  unsigned height,
				  float x,
				  float y)
{
  int x0 = (int) floorf (x);
  int y0 = (int) floorf (y);
  int x1 = x0 + 1;
  int y1 = y0 + 1;
  if (x1 >= (int) width) x1 = (int) width - 1;
  if (y1 >= (int) height) y1 = (int) height - 1;

  float tx = x - x0;
  float ty = y - y0;
  float omtx = 1.f - tx;
  float omty = 1.f - ty;

  uint32_t p00 = (uint32_t) src[(size_t) y0 * width + (size_t) x0];
  uint32_t p10 = (uint32_t) src[(size_t) y0 * width + (size_t) x1];
  uint32_t p01 = (uint32_t) src[(size_t) y1 * width + (size_t) x0];
  uint32_t p11 = (uint32_t) src[(size_t) y1 * width + (size_t) x1];

  float w00 = omtx * omty;
  float w10 = tx * omty;
  float w01 = omtx * ty;
  float w11 = tx * ty;

  float b = ((p00 >>  0) & 0xff) * w00 + ((p10 >>  0) & 0xff) * w10 +
	    ((p01 >>  0) & 0xff) * w01 + ((p11 >>  0) & 0xff) * w11;
  float g = ((p00 >>  8) & 0xff) * w00 + ((p10 >>  8) & 0xff) * w10 +
	    ((p01 >>  8) & 0xff) * w01 + ((p11 >>  8) & 0xff) * w11;
  float r = ((p00 >> 16) & 0xff) * w00 + ((p10 >> 16) & 0xff) * w10 +
	    ((p01 >> 16) & 0xff) * w01 + ((p11 >> 16) & 0xff) * w11;
  float a = ((p00 >> 24) & 0xff) * w00 + ((p10 >> 24) & 0xff) * w10 +
	    ((p01 >> 24) & 0xff) * w01 + ((p11 >> 24) & 0xff) * w11;

  return (uint32_t) (b + 0.5f)
       | ((uint32_t) (g + 0.5f) << 8)
       | ((uint32_t) (r + 0.5f) << 16)
       | ((uint32_t) (a + 0.5f) << 24);
}


/*
 * Paint callbacks
 */

/* Lazy initialization: set up root surface, initial clip and transform.
 * Called from every paint callback that needs state.
 * hb_font_paint_glyph() does NOT wrap with push/pop_transform,
 * so the first callback could be push_clip_glyph or paint_color. */
static void
ensure_initialized (hb_raster_paint_t *c)
{
  if (c->surface_stack.length) return;

  /* Root surface */
  hb_raster_image_t *root = c->acquire_surface ();
  if (unlikely (!root)) return;
  if (unlikely (!c->surface_stack.push_or_fail (root)))
  {
    c->release_surface (root);
    return;
  }

  /* Initial transform */
  if (unlikely (!c->transform_stack.push_or_fail (c->base_transform)))
  {
    c->release_surface (c->surface_stack.pop ());
    return;
  }

  /* Initial clip: full coverage rectangle */
  hb_raster_clip_t clip;
  clip.init_full (c->fixed_extents.width, c->fixed_extents.height);
  if (unlikely (!c->clip_stack.push_or_fail (std::move (clip))))
  {
    c->transform_stack.pop ();
    c->release_surface (c->surface_stack.pop ());
    return;
  }
}

static void
hb_raster_paint_push_transform (hb_paint_funcs_t *pfuncs HB_UNUSED,
				void *paint_data,
				float xx, float yx,
				float xy, float yy,
				float dx, float dy,
				void *user_data HB_UNUSED)
{
  hb_raster_paint_t *c = (hb_raster_paint_t *) paint_data;

  ensure_initialized (c);
  if (unlikely (!c->transform_stack.length)) return;

  hb_transform_t<> t = c->current_transform ();
  t.multiply ({xx, yx, xy, yy, dx, dy});
  (void) c->transform_stack.push (t);
}

static void
hb_raster_paint_pop_transform (hb_paint_funcs_t *pfuncs HB_UNUSED,
			       void *paint_data,
			       void *user_data HB_UNUSED)
{
  hb_raster_paint_t *c = (hb_raster_paint_t *) paint_data;
  c->transform_stack.pop ();
}

static hb_bool_t
hb_raster_paint_color_glyph (hb_paint_funcs_t *pfuncs HB_UNUSED,
			     void *paint_data HB_UNUSED,
			     hb_codepoint_t glyph HB_UNUSED,
			     hb_font_t *font HB_UNUSED,
			     void *user_data HB_UNUSED)
{
  return false;
}

typedef void (*hb_raster_paint_clip_mask_emit_t) (hb_raster_draw_t *rdr, void *user_data);

static void
hb_raster_paint_push_empty_clip (hb_raster_paint_t *c, unsigned w, unsigned h)
{
  hb_raster_clip_t new_clip = c->acquire_clip (w, h);
  new_clip.init_full (w, h);
  new_clip.is_rect = true;
  new_clip.rect_x0 = new_clip.rect_y0 = 0;
  new_clip.rect_x1 = new_clip.rect_y1 = 0;
  new_clip.min_x = new_clip.min_y = new_clip.max_x = new_clip.max_y = 0;
  (void) c->clip_stack.push (std::move (new_clip));
}

static void
hb_raster_paint_push_clip_from_emitter (hb_raster_paint_t *c,
					hb_raster_paint_clip_mask_emit_t emit,
					void *emit_data)
{
  ensure_initialized (c);

  hb_raster_image_t *surf = c->current_surface ();
  if (unlikely (!surf)) return;

  unsigned w = surf->extents.width;
  unsigned h = surf->extents.height;

  hb_raster_clip_t new_clip = c->acquire_clip (w, h);

  hb_raster_draw_t *rdr = c->clip_rdr;
  hb_transform_t<> t = c->current_effective_transform ();
  hb_raster_draw_set_transform (rdr, t.xx, t.yx, t.xy, t.yy, t.x0, t.y0);
  emit (rdr, emit_data);
  hb_raster_image_t *mask_img = hb_raster_draw_render (rdr);

  if (unlikely (!mask_img))
  {
    hb_raster_paint_push_empty_clip (c, w, h);
    return;
  }

  /* Allocate alpha buffer and intersect with previous clip */
  size_t clip_size = (size_t) new_clip.stride * h;
  if (unlikely (clip_size > HB_RASTER_MAX_BUFFER_SIZE ||
                !new_clip.alpha.resize ((unsigned) clip_size)))
  {
    hb_raster_draw_recycle_image (rdr, mask_img);
    hb_raster_paint_push_empty_clip (c, w, h);
    return;
  }

  const uint8_t *mask_buf = hb_raster_image_get_buffer (mask_img);
  hb_raster_extents_t mask_ext;
  hb_raster_image_get_extents (mask_img, &mask_ext);
  const hb_raster_clip_t &old_clip = c->current_clip ();

  /* Convert mask extents from surface coordinates to clip-buffer coordinates. */
  int mask_x0 = mask_ext.x_origin - surf->extents.x_origin;
  int mask_y0 = mask_ext.y_origin - surf->extents.y_origin;
  int mask_x1 = mask_x0 + (int) mask_ext.width;
  int mask_y1 = mask_y0 + (int) mask_ext.height;

  int ix0_i = hb_max ((int) old_clip.min_x, hb_max (mask_x0, 0));
  int iy0_i = hb_max ((int) old_clip.min_y, hb_max (mask_y0, 0));
  int ix1_i = hb_min ((int) old_clip.max_x, hb_min (mask_x1, (int) w));
  int iy1_i = hb_min ((int) old_clip.max_y, hb_min (mask_y1, (int) h));

  if (ix0_i >= ix1_i || iy0_i >= iy1_i)
  {
    hb_raster_draw_recycle_image (rdr, mask_img);
    hb_raster_paint_push_empty_clip (c, w, h);
    return;
  }

  unsigned ix0 = (unsigned) ix0_i;
  unsigned iy0 = (unsigned) iy0_i;
  unsigned ix1 = (unsigned) ix1_i;
  unsigned iy1 = (unsigned) iy1_i;

  new_clip.min_x = w; new_clip.min_y = h;
  new_clip.max_x = 0; new_clip.max_y = 0;

  if (old_clip.is_rect)
  {
    for (unsigned y = iy0; y < iy1; y++)
    {
      const uint8_t *mask_row = mask_buf + (unsigned) ((int) y - mask_y0) * mask_ext.stride;
      uint8_t *out_row = new_clip.alpha.arrayZ + y * new_clip.stride;
      unsigned row_min = ix1;
      unsigned row_max = ix0;
      unsigned mx = (unsigned) ((int) ix0 - mask_x0);
      for (unsigned x = ix0; x < ix1; x++)
      {
	uint8_t a = mask_row[mx++];
	out_row[x] = a;
	if (a && row_min == ix1)
	{
	  row_min = x;
	  row_max = x + 1;
	}
	else if (a)
	  row_max = x + 1;
      }
      if (row_min < row_max)
      {
	new_clip.min_x = hb_min (new_clip.min_x, row_min);
	new_clip.min_y = hb_min (new_clip.min_y, y);
	new_clip.max_x = hb_max (new_clip.max_x, row_max);
	new_clip.max_y = hb_max (new_clip.max_y, y + 1);
      }
    }
  }
  else
  {
    for (unsigned y = iy0; y < iy1; y++)
    {
      const uint8_t *old_row = old_clip.alpha.arrayZ + y * old_clip.stride;
      const uint8_t *mask_row = mask_buf + (unsigned) ((int) y - mask_y0) * mask_ext.stride;
      uint8_t *out_row = new_clip.alpha.arrayZ + y * new_clip.stride;
      unsigned row_min = ix1;
      unsigned row_max = ix0;
      for (unsigned x = ix0; x < ix1; x++)
      {
	unsigned mx = (unsigned) ((int) x - mask_x0);
	uint8_t a = hb_raster_div255 (mask_row[mx] * old_row[x]);
	out_row[x] = a;
	if (a)
	{
	  row_min = hb_min (row_min, x);
	  row_max = x + 1;
	}
      }
      if (row_min < row_max)
      {
	new_clip.min_x = hb_min (new_clip.min_x, row_min);
	new_clip.min_y = hb_min (new_clip.min_y, y);
	new_clip.max_x = hb_max (new_clip.max_x, row_max);
	new_clip.max_y = hb_max (new_clip.max_y, y + 1);
      }
    }
  }

  hb_raster_draw_recycle_image (rdr, mask_img);
  if (unlikely (!c->clip_stack.push_or_fail (std::move (new_clip))))
    hb_raster_paint_push_empty_clip (c, w, h);
}

struct hb_raster_paint_glyph_clip_data_t
{
  hb_codepoint_t glyph;
  hb_font_t *font;
};

static void
hb_raster_paint_emit_clip_glyph_mask (hb_raster_draw_t *rdr, void *user_data)
{
  hb_raster_paint_glyph_clip_data_t *data = (hb_raster_paint_glyph_clip_data_t *) user_data;
  /* Let draw-render choose tight glyph extents; we map by mask origin below. */
  hb_font_draw_glyph (data->font, data->glyph, hb_raster_draw_get_funcs (), rdr);
}

static void
hb_raster_paint_push_clip_glyph (hb_paint_funcs_t *pfuncs HB_UNUSED,
				 void *paint_data,
				 hb_codepoint_t glyph,
				 hb_font_t *font,
				 void *user_data HB_UNUSED)
{
  hb_raster_paint_t *c = (hb_raster_paint_t *) paint_data;
  hb_raster_paint_glyph_clip_data_t data = {glyph, font};
  hb_raster_paint_push_clip_from_emitter (c, hb_raster_paint_emit_clip_glyph_mask, &data);
}

/* Push clip from arbitrary path emitter (used by SVG rasterizer).
 * Identical to push_clip_glyph but calls user func instead of hb_font_draw_glyph. */
struct hb_raster_paint_path_clip_data_t
{
  hb_raster_svg_path_func_t func;
  void *user_data;
};

static void
hb_raster_paint_emit_clip_path_mask (hb_raster_draw_t *rdr, void *user_data)
{
  hb_raster_paint_path_clip_data_t *data = (hb_raster_paint_path_clip_data_t *) user_data;
  data->func (hb_raster_draw_get_funcs (), rdr, data->user_data);
}

void
hb_raster_paint_push_clip_path (hb_raster_paint_t *c,
				hb_raster_svg_path_func_t func,
				void *user_data)
{
  hb_raster_paint_path_clip_data_t data = {func, user_data};
  hb_raster_paint_push_clip_from_emitter (c, hb_raster_paint_emit_clip_path_mask, &data);
}

static void
hb_raster_paint_push_clip_rectangle (hb_paint_funcs_t *pfuncs HB_UNUSED,
				     void *paint_data,
				     float xmin, float ymin,
				     float xmax, float ymax,
				     void *user_data HB_UNUSED)
{
  hb_raster_paint_t *c = (hb_raster_paint_t *) paint_data;

  ensure_initialized (c);

  if (!c->surface_stack.length) return;

  hb_transform_t<> t = c->current_effective_transform ();

  hb_raster_image_t *surf = c->current_surface ();
  if (unlikely (!surf)) return;

  unsigned w = surf->extents.width;
  unsigned h = surf->extents.height;
  bool is_axis_aligned = (t.xy == 0.f && t.yx == 0.f);

  /* Transform the four corners to pixel space */
  float cx[4], cy[4];
  cx[0] = xmin; cy[0] = ymin;
  cx[1] = xmax; cy[1] = ymin;
  cx[2] = xmax; cy[2] = ymax;
  cx[3] = xmin; cy[3] = ymax;
  for (unsigned i = 0; i < 4; i++)
    t.transform_point (cx[i], cy[i]);

  /* Compute bounding box in pixel coords */
  float fmin_x = cx[0], fmin_y = cy[0], fmax_x = cx[0], fmax_y = cy[0];
  for (unsigned i = 1; i < 4; i++)
  {
    fmin_x = hb_min (fmin_x, cx[i]); fmin_y = hb_min (fmin_y, cy[i]);
    fmax_x = hb_max (fmax_x, cx[i]); fmax_y = hb_max (fmax_y, cy[i]);
  }

  int px0 = (int) floorf (fmin_x) - surf->extents.x_origin;
  int py0 = (int) floorf (fmin_y) - surf->extents.y_origin;
  int px1 = (int) ceilf (fmax_x) - surf->extents.x_origin;
  int py1 = (int) ceilf (fmax_y) - surf->extents.y_origin;

  /* Clamp to surface bounds */
  px0 = hb_max (px0, 0);
  py0 = hb_max (py0, 0);
  px1 = hb_min (px1, (int) w);
  py1 = hb_min (py1, (int) h);

  const hb_raster_clip_t &old_clip = c->current_clip ();

  hb_raster_clip_t new_clip = c->acquire_clip (w, h);

  if (is_axis_aligned && old_clip.is_rect)
  {
    /* Fast path: axis-aligned rect-on-rect intersection */
    new_clip.is_rect = true;
    new_clip.rect_x0 = hb_max (px0, old_clip.rect_x0);
    new_clip.rect_y0 = hb_max (py0, old_clip.rect_y0);
    new_clip.rect_x1 = hb_min (px1, old_clip.rect_x1);
    new_clip.rect_y1 = hb_min (py1, old_clip.rect_y1);
    new_clip.update_bounds_from_rect ();
  }
  else
  {
    /* General case: rasterize transformed quad as alpha mask */
    new_clip.is_rect = false;
    size_t clip_size = (size_t) new_clip.stride * h;
    if (unlikely (clip_size > HB_RASTER_MAX_BUFFER_SIZE ||
                  !new_clip.alpha.resize ((unsigned) clip_size)))
    {
      hb_raster_paint_push_empty_clip (c, w, h);
      return;
    }
    hb_memset (new_clip.alpha.arrayZ, 0, (unsigned) clip_size);

    /* Convert quad corners to pixel-relative coords */
    float qx[4], qy[4];
    int ox = surf->extents.x_origin;
    int oy = surf->extents.y_origin;
    for (unsigned i = 0; i < 4; i++)
    {
      qx[i] = cx[i] - ox;
      qy[i] = cy[i] - oy;
    }

    /* For each pixel in the bounding box, test if inside the quad
     * using cross-product edge tests (winding order). */
    unsigned iy0 = (unsigned) hb_max (py0, (int) old_clip.min_y);
    unsigned iy1 = (unsigned) hb_min (py1, (int) old_clip.max_y);
    unsigned ix0 = (unsigned) hb_max (px0, (int) old_clip.min_x);
    unsigned ix1 = (unsigned) hb_min (px1, (int) old_clip.max_x);
    new_clip.min_x = w; new_clip.min_y = h;
    new_clip.max_x = 0; new_clip.max_y = 0;

    /* Precompute edge normals for point-in-quad test.
     * Edge i goes from corner i to corner (i+1)%4.
     * Normal = (dy, -dx); inside test: dot(normal, p-corner) >= 0 */
	    float enx[4], eny[4], ed[4];
	    for (unsigned i = 0; i < 4; i++)
	    {
	      unsigned j = (i + 1) & 3;
	      float edx = qx[j] - qx[i], edy = qy[j] - qy[i];
	      enx[i] = edy;       /* normal x */
	      eny[i] = -edx;      /* normal y */
	      ed[i] = enx[i] * qx[i] + eny[i] * qy[i]; /* distance threshold */
	    }
	    float area2 = 0.f;
	    for (unsigned i = 0; i < 4; i++)
	    {
	      unsigned j = (i + 1) & 3;
	      area2 += qx[i] * qy[j] - qx[j] * qy[i];
	    }
	    bool ccw = area2 >= 0.f;

    if (old_clip.is_rect)
    {
      for (unsigned y = iy0; y < iy1; y++)
	for (unsigned x = ix0; x < ix1; x++)
	{
	  float px_f = x + 0.5f, py_f = y + 0.5f;
	  /* Test if pixel center is inside the quad */
		  bool inside = true;
		  for (unsigned i = 0; i < 4; i++)
		  {
		    float d = enx[i] * px_f + eny[i] * py_f;
		    if (ccw ? d < ed[i] : d > ed[i])
		    {
		      inside = false;
		      break;
		    }
		  }
	  uint8_t a = inside ? 255 : 0;
	  new_clip.alpha[y * new_clip.stride + x] = a;
	  if (a)
	  {
	    new_clip.min_x = hb_min (new_clip.min_x, x);
	    new_clip.min_y = hb_min (new_clip.min_y, y);
	    new_clip.max_x = hb_max (new_clip.max_x, x + 1);
	    new_clip.max_y = hb_max (new_clip.max_y, y + 1);
	  }
	}
    }
    else
    {
      for (unsigned y = iy0; y < iy1; y++)
      {
	const uint8_t *old_row = old_clip.alpha.arrayZ + y * old_clip.stride;
	for (unsigned x = ix0; x < ix1; x++)
	{
	  float px_f = x + 0.5f, py_f = y + 0.5f;
	  /* Test if pixel center is inside the quad */
		  bool inside = true;
		  for (unsigned i = 0; i < 4; i++)
		  {
		    float d = enx[i] * px_f + eny[i] * py_f;
		    if (ccw ? d < ed[i] : d > ed[i])
		    {
		      inside = false;
		      break;
		    }
		  }
	  uint8_t a = inside ? old_row[x] : 0;
	  new_clip.alpha[y * new_clip.stride + x] = a;
	  if (a)
	  {
	    new_clip.min_x = hb_min (new_clip.min_x, x);
	    new_clip.min_y = hb_min (new_clip.min_y, y);
	    new_clip.max_x = hb_max (new_clip.max_x, x + 1);
	    new_clip.max_y = hb_max (new_clip.max_y, y + 1);
	  }
	}
      }
    }
  }

  if (unlikely (!c->clip_stack.push_or_fail (std::move (new_clip))))
    hb_raster_paint_push_empty_clip (c, surf->extents.width, surf->extents.height);
}

static void
hb_raster_paint_pop_clip (hb_paint_funcs_t *pfuncs HB_UNUSED,
			  void *paint_data,
			  void *user_data HB_UNUSED)
{
  hb_raster_paint_t *c = (hb_raster_paint_t *) paint_data;
  if (!c->clip_stack.length) return;
  c->release_clip (c->clip_stack.pop ());
}

static void
hb_raster_paint_push_group (hb_paint_funcs_t *pfuncs HB_UNUSED,
			    void *paint_data,
			    void *user_data HB_UNUSED)
{
  hb_raster_paint_t *c = (hb_raster_paint_t *) paint_data;

  ensure_initialized (c);

  hb_raster_image_t *new_surf = c->acquire_surface ();
  if (unlikely (!new_surf)) return;
  if (unlikely (!c->surface_stack.push_or_fail (new_surf)))
    c->release_surface (new_surf);
}

static void
hb_raster_paint_pop_group (hb_paint_funcs_t *pfuncs HB_UNUSED,
			   void *paint_data,
			   hb_paint_composite_mode_t mode,
			   void *user_data HB_UNUSED)
{
  hb_raster_paint_t *c = (hb_raster_paint_t *) paint_data;

  if (c->surface_stack.length < 2) return;

  hb_raster_image_t *src = c->surface_stack.pop ();
  hb_raster_image_t *dst = c->current_surface ();

  if (dst && src)
    hb_raster_image_composite (dst, src, mode);

  c->release_surface (src);
}

static void
hb_raster_paint_color (hb_paint_funcs_t *pfuncs HB_UNUSED,
		       void *paint_data,
		       hb_bool_t is_foreground,
		       hb_color_t color,
		       void *user_data HB_UNUSED)
{
  hb_raster_paint_t *c = (hb_raster_paint_t *) paint_data;

  ensure_initialized (c);

  hb_raster_image_t *surf = c->current_surface ();
  if (unlikely (!surf)) return;

  if (is_foreground)
  {
    /* Use foreground color, modulating alpha */
    color = HB_COLOR (hb_color_get_blue (c->foreground),
		      hb_color_get_green (c->foreground),
		      hb_color_get_red (c->foreground),
		      hb_raster_div255 (hb_color_get_alpha (c->foreground) *
			      hb_color_get_alpha (color)));
  }

  uint32_t premul = color_to_premul_pixel (color);
  uint8_t premul_a = (uint8_t) (premul >> 24);
  const hb_raster_clip_t &clip = c->current_clip ();

  unsigned stride = surf->extents.stride;
  if (clip.min_x >= clip.max_x || clip.min_y >= clip.max_y) return;
  if (premul_a == 0) return;

  if (likely (!clip.is_rect))
  {
    for (unsigned y = clip.min_y; y < clip.max_y; y++)
    {
      hb_packed_t<uint32_t> *__restrict row = (hb_packed_t<uint32_t> *) (surf->buffer.arrayZ + y * stride);
      const uint8_t *__restrict clip_row = clip.alpha.arrayZ + y * clip.stride;
      for (unsigned x = clip.min_x; x < clip.max_x; x++)
      {
	uint8_t clip_alpha = clip_row[x];
	if (clip_alpha == 0) continue;
	if (clip_alpha == 255)
	{
	  if (premul_a == 255)
	    row[x] = hb_packed_t<uint32_t> (premul);
	  else
	    row[x] = hb_packed_t<uint32_t> (hb_raster_src_over (premul, (uint32_t) row[x]));
	}
	else
	{
	  uint32_t src = hb_raster_alpha_mul (premul, clip_alpha);
	  row[x] = hb_packed_t<uint32_t> (hb_raster_src_over (src, (uint32_t) row[x]));
	}
      }
    }
  }
  else
  {
    for (unsigned y = clip.min_y; y < clip.max_y; y++)
    {
      hb_packed_t<uint32_t> *row = (hb_packed_t<uint32_t> *) (surf->buffer.arrayZ + y * stride);
      for (unsigned x = clip.min_x; x < clip.max_x; x++)
	row[x] = hb_packed_t<uint32_t> (hb_raster_src_over (premul, (uint32_t) row[x]));
    }
  }
}

static hb_bool_t
hb_raster_paint_image (hb_paint_funcs_t *pfuncs HB_UNUSED,
		       void *paint_data,
		       hb_blob_t *blob,
		       unsigned width,
		       unsigned height,
		       hb_tag_t format,
		       float slant HB_UNUSED,
		       hb_glyph_extents_t *extents,
		       void *user_data HB_UNUSED)
{
  hb_raster_paint_t *c = (hb_raster_paint_t *) paint_data;

  ensure_initialized (c);

  /* Handle SVG format */
  if (format == HB_PAINT_IMAGE_FORMAT_SVG)
    return hb_raster_svg_render (c, blob, c->svg_glyph, c->svg_font,
				 c->svg_palette, c->foreground);

  unsigned src_width = width;
  unsigned src_height = height;
  const hb_packed_t<uint32_t> *src_data = nullptr;
  hb_raster_image_t decoded_png;

  if (format == HB_PAINT_IMAGE_FORMAT_BGRA)
  {
    if (src_width == 0 || src_height == 0)
      return false;
    if (src_width > (unsigned) INT_MAX || src_height > (unsigned) INT_MAX)
      return false;

    unsigned data_len;
    const uint8_t *data = (const uint8_t *) hb_blob_get_data (blob, &data_len);
    size_t pixel_count = (size_t) src_width * (size_t) src_height;
    if (src_width && pixel_count / src_width != src_height)
      return false;
    if (pixel_count > (size_t) -1 / 4u)
      return false;
    size_t required_size = pixel_count * 4u;
    if (!data || (size_t) data_len < required_size)
      return false;

    src_data = (const hb_packed_t<uint32_t> *) data;
  }
  else if (format == HB_PAINT_IMAGE_FORMAT_PNG)
  {
#ifdef HAVE_PNG
    if (!decoded_png.deserialize_from_png (blob))
      return false;
    src_width = decoded_png.extents.width;
    src_height = decoded_png.extents.height;
    src_data = (const hb_packed_t<uint32_t> *) decoded_png.buffer.arrayZ;
#else
    return false;
#endif
  }
  else
    return false;

  hb_raster_image_t *surf = c->current_surface ();
  if (unlikely (!surf)) return false;
  if (!extents) return false;

  const hb_raster_clip_t &clip = c->current_clip ();
  hb_transform_t<> t = c->current_effective_transform ();

  /* Compute inverse transform for sampling */
  float det = t.xx * t.yy - t.xy * t.yx;
  if (fabsf (det) < 1e-10f) return false;
  float inv_det = 1.f / det;
  float inv_xx =  t.yy * inv_det;
  float inv_xy = -t.xy * inv_det;
  float inv_yx = -t.yx * inv_det;
  float inv_yy =  t.xx * inv_det;
  float inv_x0 = (t.xy * t.y0 - t.yy * t.x0) * inv_det;
  float inv_y0 = (t.yx * t.x0 - t.xx * t.y0) * inv_det;

  unsigned surf_stride = surf->extents.stride;
  int ox = surf->extents.x_origin;
  int oy = surf->extents.y_origin;

  /* Image source rectangle in glyph space */
  float img_x = extents->x_bearing;
  float img_y = extents->y_bearing;
  float img_sx = (float) extents->width / src_width;
  float img_sy = (float) extents->height / src_height;
  if (fabsf (img_sx) < 1e-10f || fabsf (img_sy) < 1e-10f)
    return false;

  if (clip.is_rect)
  {
    for (unsigned py = clip.min_y; py < clip.max_y; py++)
    {
      hb_packed_t<uint32_t> *row = (hb_packed_t<uint32_t> *) (surf->buffer.arrayZ + py * surf_stride);
      float gx = inv_xx * (float) ((int) clip.min_x + ox) + inv_xy * (float) ((int) py + oy) + inv_x0;
      float gy = inv_yx * (float) ((int) clip.min_x + ox) + inv_yy * (float) ((int) py + oy) + inv_y0;
      for (unsigned px = clip.min_x; px < clip.max_x; px++)
      {
	/* Map glyph space to image texel; bilinear reconstruction. */
	float ix = (gx - img_x) / img_sx;
	float iy = (float) (src_height - 1) - (gy - img_y) / img_sy;

	if (ix < 0.f || iy < 0.f ||
	    ix > (float) (src_width - 1) || iy > (float) (src_height - 1))
	{
	  gx += inv_xx;
	  gy += inv_yx;
	  continue;
	}

	uint32_t src_px = hb_raster_sample_bilinear_premul (src_data, src_width, src_height,
							     ix, iy);
	row[px] = hb_packed_t<uint32_t> (hb_raster_src_over (src_px, (uint32_t) row[px]));
	gx += inv_xx;
	gy += inv_yx;
      }
    }
  }
  else
  {
    for (unsigned py = clip.min_y; py < clip.max_y; py++)
    {
      hb_packed_t<uint32_t> *row = (hb_packed_t<uint32_t> *) (surf->buffer.arrayZ + py * surf_stride);
      const uint8_t *clip_row = clip.alpha.arrayZ + py * clip.stride;
      float gx = inv_xx * (float) ((int) clip.min_x + ox) + inv_xy * (float) ((int) py + oy) + inv_x0;
      float gy = inv_yx * (float) ((int) clip.min_x + ox) + inv_yy * (float) ((int) py + oy) + inv_y0;
      for (unsigned px = clip.min_x; px < clip.max_x; px++)
      {
	uint8_t clip_alpha = clip_row[px];
	if (clip_alpha == 0)
	{
	  gx += inv_xx;
	  gy += inv_yx;
	  continue;
	}

	/* Map glyph space to image texel; bilinear reconstruction. */
	float ix = (gx - img_x) / img_sx;
	float iy = (float) (src_height - 1) - (gy - img_y) / img_sy;

	if (ix < 0.f || iy < 0.f ||
	    ix > (float) (src_width - 1) || iy > (float) (src_height - 1))
	{
	  gx += inv_xx;
	  gy += inv_yx;
	  continue;
	}

	uint32_t src_px = hb_raster_sample_bilinear_premul (src_data, src_width, src_height,
							     ix, iy);
	src_px = hb_raster_alpha_mul (src_px, clip_alpha);
	row[px] = hb_packed_t<uint32_t> (hb_raster_src_over (src_px, (uint32_t) row[px]));
	gx += inv_xx;
	gy += inv_yx;
      }
    }
  }

  return true;
}


/*
 * Gradient helpers
 */

#define PREALLOCATED_COLOR_STOPS 16
#define GRADIENT_LUT_SIZE 256
#define GRADIENT_LUT_MIN_PIXELS (64u * 64u)

static int
cmp_color_stop (const void *p1, const void *p2)
{
  const hb_color_stop_t *c1 = (const hb_color_stop_t *) p1;
  const hb_color_stop_t *c2 = (const hb_color_stop_t *) p2;
  if (c1->offset < c2->offset) return -1;
  if (c1->offset > c2->offset) return 1;
  return 0;
}

static bool
get_color_stops (hb_raster_paint_t *c,
		 hb_color_line_t *color_line,
		 unsigned *count,
		 hb_color_stop_t **stops)
{
  unsigned len = hb_color_line_get_color_stops (color_line, 0, nullptr, nullptr);
  if (len > *count)
  {
    if (unlikely (!c->scratch_color_stops.resize (len)))
      return false;
    *stops = c->scratch_color_stops.arrayZ;
  }
  hb_color_line_get_color_stops (color_line, 0, &len, *stops);
  for (unsigned i = 0; i < len; i++)
    if ((*stops)[i].is_foreground)
      (*stops)[i].color = HB_COLOR (hb_color_get_blue (c->foreground),
				    hb_color_get_green (c->foreground),
				    hb_color_get_red (c->foreground),
				    hb_raster_div255 (hb_color_get_alpha (c->foreground) *
					    hb_color_get_alpha ((*stops)[i].color)));

  *count = len;
  return true;
}

static void
normalize_color_line (hb_color_stop_t *stops,
		      unsigned len,
		      float *omin, float *omax)
{
  hb_qsort (stops, len, sizeof (hb_color_stop_t), cmp_color_stop);

  float mn = stops[0].offset, mx = stops[0].offset;
  for (unsigned i = 1; i < len; i++)
  {
    mn = hb_min (mn, stops[i].offset);
    mx = hb_max (mx, stops[i].offset);
  }
  if (mn != mx)
    for (unsigned i = 0; i < len; i++)
      stops[i].offset = (stops[i].offset - mn) / (mx - mn);

  *omin = mn;
  *omax = mx;
}

/* Evaluate color at normalized position t, interpolating in premultiplied space. */
static uint32_t
evaluate_color_line (const hb_color_stop_t *stops, unsigned len, float t,
		     hb_paint_extend_t extend)
{
  /* Apply extend mode */
  if (extend == HB_PAINT_EXTEND_PAD)
  {
    t = hb_clamp (t, 0.f, 1.f);
  }
  else if (extend == HB_PAINT_EXTEND_REPEAT)
  {
    t = t - floorf (t);
  }
  else /* REFLECT */
  {
    if (t < 0) t = -t;
    int period = (int) floorf (t);
    float frac = t - (float) period;
    t = (period & 1) ? 1.f - frac : frac;
  }

  /* Find bounding stops */
  if (t <= stops[0].offset)
    return color_to_premul_pixel (stops[0].color);
  if (t >= stops[len - 1].offset)
    return color_to_premul_pixel (stops[len - 1].color);

  unsigned i;
  for (i = 0; i < len - 1; i++)
    if (t < stops[i + 1].offset)
      break;

  float range = stops[i + 1].offset - stops[i].offset;
  float k = range > 0.f ? (t - stops[i].offset) / range : 0.f;

  /* Interpolate in premultiplied [0,255] space */
  hb_color_t c0 = stops[i].color;
  hb_color_t c1 = stops[i + 1].color;

  float a0 = hb_color_get_alpha (c0) / 255.f;
  float r0 = hb_color_get_red   (c0) / 255.f * a0;
  float g0 = hb_color_get_green (c0) / 255.f * a0;
  float b0 = hb_color_get_blue  (c0) / 255.f * a0;

  float a1 = hb_color_get_alpha (c1) / 255.f;
  float r1 = hb_color_get_red   (c1) / 255.f * a1;
  float g1 = hb_color_get_green (c1) / 255.f * a1;
  float b1 = hb_color_get_blue  (c1) / 255.f * a1;

  float a = a0 + k * (a1 - a0);
  float r = r0 + k * (r1 - r0);
  float g = g0 + k * (g1 - g0);
  float b = b0 + k * (b1 - b0);

  uint8_t pa = (uint8_t) (a * 255.f + 0.5f);
  uint8_t pr = (uint8_t) (r * 255.f + 0.5f);
  uint8_t pg = (uint8_t) (g * 255.f + 0.5f);
  uint8_t pb = (uint8_t) (b * 255.f + 0.5f);

  return (uint32_t) pb | ((uint32_t) pg << 8) | ((uint32_t) pr << 16) | ((uint32_t) pa << 24);
}

static HB_ALWAYS_INLINE float
normalize_gradient_t (float t, hb_paint_extend_t extend)
{
  if (extend == HB_PAINT_EXTEND_PAD)
    return hb_clamp (t, 0.f, 1.f);
  if (extend == HB_PAINT_EXTEND_REPEAT)
  {
    t = t - floorf (t);
    return t < 0.f ? t + 1.f : t;
  }

  /* REFLECT */
  if (t < 0.f) t = -t;
  int period = (int) floorf (t);
  float frac = t - (float) period;
  return (period & 1) ? 1.f - frac : frac;
}

static void
build_gradient_lut (const hb_color_stop_t *stops,
		    unsigned len,
		    uint32_t *lut)
{
  for (unsigned i = 0; i < GRADIENT_LUT_SIZE; i++)
  {
    float t = (float) i / (GRADIENT_LUT_SIZE - 1);
    lut[i] = evaluate_color_line (stops, len, t, HB_PAINT_EXTEND_PAD);
  }
}

static HB_ALWAYS_INLINE uint32_t
lookup_gradient_lut (const uint32_t *lut,
		     float t,
		     hb_paint_extend_t extend)
{
  float u = normalize_gradient_t (t, extend);
  unsigned idx = (unsigned) (u * (GRADIENT_LUT_SIZE - 1) + 0.5f);
  return lut[idx];
}

static void
reduce_anchors (float x0, float y0,
		float x1, float y1,
		float x2, float y2,
		float *xx0, float *yy0,
		float *xx1, float *yy1)
{
  float q2x = x2 - x0, q2y = y2 - y0;
  float q1x = x1 - x0, q1y = y1 - y0;
  float s = q2x * q2x + q2y * q2y;
  if (s < 0.000001f)
  {
    *xx0 = x0; *yy0 = y0;
    *xx1 = x1; *yy1 = y1;
    return;
  }
  float k = (q2x * q1x + q2y * q1y) / s;
  *xx0 = x0;
  *yy0 = y0;
  *xx1 = x1 - k * q2x;
  *yy1 = y1 - k * q2y;
}


/*
 * Gradient paint callbacks
 */

static void
hb_raster_paint_linear_gradient (hb_paint_funcs_t *pfuncs HB_UNUSED,
				 void *paint_data,
				 hb_color_line_t *color_line,
				 float x0, float y0,
				 float x1, float y1,
				 float x2, float y2,
				 void *user_data HB_UNUSED)
{
  hb_raster_paint_t *c = (hb_raster_paint_t *) paint_data;

  ensure_initialized (c);

  hb_raster_image_t *surf = c->current_surface ();
  if (unlikely (!surf)) return;

  unsigned len = PREALLOCATED_COLOR_STOPS;
  hb_color_stop_t stops_[PREALLOCATED_COLOR_STOPS];
  hb_color_stop_t *stops = stops_;

  if (unlikely (!get_color_stops (c, color_line, &len, &stops)))
    return;
  float mn, mx;
  normalize_color_line (stops, len, &mn, &mx);

  hb_paint_extend_t extend = hb_color_line_get_extend (color_line);
  const hb_raster_clip_t &clip = c->current_clip ();
  unsigned clip_w = clip.max_x > clip.min_x ? clip.max_x - clip.min_x : 0;
  unsigned clip_h = clip.max_y > clip.min_y ? clip.max_y - clip.min_y : 0;
  bool use_lut = (uint64_t) clip_w * clip_h >= GRADIENT_LUT_MIN_PIXELS;
  uint32_t lut[GRADIENT_LUT_SIZE];
  if (use_lut)
    build_gradient_lut (stops, len, lut);

  /* Reduce 3-point anchor to 2-point gradient axis */
  float lx0, ly0, lx1, ly1;
  reduce_anchors (x0, y0, x1, y1, x2, y2, &lx0, &ly0, &lx1, &ly1);

  /* Apply normalization to endpoints */
  float gx0 = lx0 + mn * (lx1 - lx0);
  float gy0 = ly0 + mn * (ly1 - ly0);
  float gx1 = lx0 + mx * (lx1 - lx0);
  float gy1 = ly0 + mx * (ly1 - ly0);

  /* Inverse transform: pixel → glyph space */
  hb_transform_t<> t = c->current_effective_transform ();
  float det = t.xx * t.yy - t.xy * t.yx;
  if (fabsf (det) < 1e-10f) goto done;

  {
    float inv_det = 1.f / det;
    float inv_xx =  t.yy * inv_det;
    float inv_xy = -t.xy * inv_det;
    float inv_yx = -t.yx * inv_det;
    float inv_yy =  t.xx * inv_det;
    float inv_x0 = (t.xy * t.y0 - t.yy * t.x0) * inv_det;
    float inv_y0 = (t.yx * t.x0 - t.xx * t.y0) * inv_det;

    /* Gradient direction vector and denominator for projection */
    float dx = gx1 - gx0, dy = gy1 - gy0;
    float denom = dx * dx + dy * dy;
    if (denom < 1e-10f) goto done;
    float inv_denom = 1.f / denom;

    unsigned stride = surf->extents.stride;
    int ox = surf->extents.x_origin;
    int oy = surf->extents.y_origin;

    if (clip.is_rect)
    {
      for (unsigned py = clip.min_y; py < clip.max_y; py++)
      {
	hb_packed_t<uint32_t> *row = (hb_packed_t<uint32_t> *) (surf->buffer.arrayZ + py * stride);
	float gx = inv_xx * ((float) ((int) clip.min_x + ox) + 0.5f) + inv_xy * ((float) ((int) py + oy) + 0.5f) + inv_x0;
	float gy = inv_yx * ((float) ((int) clip.min_x + ox) + 0.5f) + inv_yy * ((float) ((int) py + oy) + 0.5f) + inv_y0;
	if (use_lut)
	{
	  for (unsigned px = clip.min_x; px < clip.max_x; px++)
	  {
	    float proj_t = ((gx - gx0) * dx + (gy - gy0) * dy) * inv_denom;
	    uint32_t src = lookup_gradient_lut (lut, proj_t, extend);
	    row[px] = hb_packed_t<uint32_t> (hb_raster_src_over (src, (uint32_t) row[px]));
	    gx += inv_xx;
	    gy += inv_yx;
	  }
	}
	else
	{
	  for (unsigned px = clip.min_x; px < clip.max_x; px++)
	  {
	    float proj_t = ((gx - gx0) * dx + (gy - gy0) * dy) * inv_denom;
	    uint32_t src = evaluate_color_line (stops, len, proj_t, extend);
	    row[px] = hb_packed_t<uint32_t> (hb_raster_src_over (src, (uint32_t) row[px]));
	    gx += inv_xx;
	    gy += inv_yx;
	  }
	}
      }
    }
    else
    {
      for (unsigned py = clip.min_y; py < clip.max_y; py++)
      {
	hb_packed_t<uint32_t> *row = (hb_packed_t<uint32_t> *) (surf->buffer.arrayZ + py * stride);
	const uint8_t *clip_row = clip.alpha.arrayZ + py * clip.stride;
	float gx = inv_xx * ((float) ((int) clip.min_x + ox) + 0.5f) + inv_xy * ((float) ((int) py + oy) + 0.5f) + inv_x0;
	float gy = inv_yx * ((float) ((int) clip.min_x + ox) + 0.5f) + inv_yy * ((float) ((int) py + oy) + 0.5f) + inv_y0;
	if (use_lut)
	{
	  for (unsigned px = clip.min_x; px < clip.max_x; px++)
	  {
	    uint8_t clip_alpha = clip_row[px];
	    if (clip_alpha == 0)
	    {
	      gx += inv_xx;
	      gy += inv_yx;
	      continue;
	    }
	    float proj_t = ((gx - gx0) * dx + (gy - gy0) * dy) * inv_denom;
	    uint32_t src = lookup_gradient_lut (lut, proj_t, extend);
	    src = hb_raster_alpha_mul (src, clip_alpha);
	    row[px] = hb_packed_t<uint32_t> (hb_raster_src_over (src, (uint32_t) row[px]));
	    gx += inv_xx;
	    gy += inv_yx;
	  }
	}
	else
	{
	  for (unsigned px = clip.min_x; px < clip.max_x; px++)
	  {
	    uint8_t clip_alpha = clip_row[px];
	    if (clip_alpha == 0)
	    {
	      gx += inv_xx;
	      gy += inv_yx;
	      continue;
	    }
	    float proj_t = ((gx - gx0) * dx + (gy - gy0) * dy) * inv_denom;
	    uint32_t src = evaluate_color_line (stops, len, proj_t, extend);
	    src = hb_raster_alpha_mul (src, clip_alpha);
	    row[px] = hb_packed_t<uint32_t> (hb_raster_src_over (src, (uint32_t) row[px]));
	    gx += inv_xx;
	    gy += inv_yx;
	  }
	}
      }
    }
  }

done:
  (void) stops_;
}

static void
hb_raster_paint_radial_gradient (hb_paint_funcs_t *pfuncs HB_UNUSED,
				 void *paint_data,
				 hb_color_line_t *color_line,
				 float x0, float y0, float r0,
				 float x1, float y1, float r1,
				 void *user_data HB_UNUSED)
{
  hb_raster_paint_t *c = (hb_raster_paint_t *) paint_data;

  ensure_initialized (c);

  hb_raster_image_t *surf = c->current_surface ();
  if (unlikely (!surf)) return;

  unsigned len = PREALLOCATED_COLOR_STOPS;
  hb_color_stop_t stops_[PREALLOCATED_COLOR_STOPS];
  hb_color_stop_t *stops = stops_;

  if (unlikely (!get_color_stops (c, color_line, &len, &stops)))
    return;
  float mn, mx;
  normalize_color_line (stops, len, &mn, &mx);

  hb_paint_extend_t extend = hb_color_line_get_extend (color_line);
  const hb_raster_clip_t &clip = c->current_clip ();
  unsigned clip_w = clip.max_x > clip.min_x ? clip.max_x - clip.min_x : 0;
  unsigned clip_h = clip.max_y > clip.min_y ? clip.max_y - clip.min_y : 0;
  bool use_lut = (uint64_t) clip_w * clip_h >= GRADIENT_LUT_MIN_PIXELS;
  uint32_t lut[GRADIENT_LUT_SIZE];
  if (use_lut)
    build_gradient_lut (stops, len, lut);

  /* Apply normalization to circle parameters */
  float cx0 = x0 + mn * (x1 - x0);
  float cy0 = y0 + mn * (y1 - y0);
  float cr0 = r0 + mn * (r1 - r0);
  float cx1 = x0 + mx * (x1 - x0);
  float cy1 = y0 + mx * (y1 - y0);
  float cr1 = r0 + mx * (r1 - r0);

  /* Inverse transform */
  hb_transform_t<> t = c->current_effective_transform ();
  float det = t.xx * t.yy - t.xy * t.yx;
  if (fabsf (det) < 1e-10f) goto done;

  {
    float inv_det = 1.f / det;
    float inv_xx =  t.yy * inv_det;
    float inv_xy = -t.xy * inv_det;
    float inv_yx = -t.yx * inv_det;
    float inv_yy =  t.xx * inv_det;
    float inv_x0 = (t.xy * t.y0 - t.yy * t.x0) * inv_det;
    float inv_y0 = (t.yx * t.x0 - t.xx * t.y0) * inv_det;

    /* Precompute quadratic coefficients for radial gradient:
     * |p - c0 - t*(c1-c0)|^2 = (r0 + t*(r1-r0))^2
     *
     * Expanding gives At^2 + Bt + C = 0 where:
     *   cdx = c1.x - c0.x, cdy = c1.y - c0.y, dr = r1 - r0
     *   A = cdx^2 + cdy^2 - dr^2
     *   B = -2*(px-c0.x)*cdx - 2*(py-c0.y)*cdy - 2*r0*dr
     *   C = (px-c0.x)^2 + (py-c0.y)^2 - r0^2
     */
    float cdx = cx1 - cx0, cdy = cy1 - cy0;
    float dr = cr1 - cr0;
    float A = cdx * cdx + cdy * cdy - dr * dr;

    unsigned stride = surf->extents.stride;
    int ox = surf->extents.x_origin;
    int oy = surf->extents.y_origin;

    if (clip.is_rect)
    {
      for (unsigned py = clip.min_y; py < clip.max_y; py++)
      {
	hb_packed_t<uint32_t> *row = (hb_packed_t<uint32_t> *) (surf->buffer.arrayZ + py * stride);
	float gx = inv_xx * ((float) ((int) clip.min_x + ox) + 0.5f) + inv_xy * ((float) ((int) py + oy) + 0.5f) + inv_x0;
	float gy = inv_yx * ((float) ((int) clip.min_x + ox) + 0.5f) + inv_yy * ((float) ((int) py + oy) + 0.5f) + inv_y0;
	if (use_lut)
	{
	  for (unsigned px = clip.min_x; px < clip.max_x; px++)
	  {
	    float dpx = gx - cx0, dpy = gy - cy0;
	    float B = -2.f * (dpx * cdx + dpy * cdy + cr0 * dr);
	    float C = dpx * dpx + dpy * dpy - cr0 * cr0;

	    float grad_t;
	    if (fabsf (A) > 1e-10f)
	    {
	      float disc = B * B - 4.f * A * C;
	      if (disc < 0.f)
	      {
		gx += inv_xx;
		gy += inv_yx;
		continue;
	      }
	      float sq = sqrtf (disc);
	      /* Pick the larger root (t closer to 1 = outer circle) */
	      float t1 = (-B + sq) / (2.f * A);
	      float t2 = (-B - sq) / (2.f * A);
	      /* Choose the root that gives a positive radius */
	      if (cr0 + t1 * dr >= 0.f)
		grad_t = t1;
	      else
		grad_t = t2;
	    }
	    else
	    {
	      /* Linear case: Bt + C = 0 */
	      if (fabsf (B) < 1e-10f)
	      {
		gx += inv_xx;
		gy += inv_yx;
		continue;
	      }
	      grad_t = -C / B;
	    }

	    uint32_t src = lookup_gradient_lut (lut, grad_t, extend);
	    row[px] = hb_packed_t<uint32_t> (hb_raster_src_over (src, (uint32_t) row[px]));
	    gx += inv_xx;
	    gy += inv_yx;
	  }
	}
	else
	{
	  for (unsigned px = clip.min_x; px < clip.max_x; px++)
	  {
	    float dpx = gx - cx0, dpy = gy - cy0;
	    float B = -2.f * (dpx * cdx + dpy * cdy + cr0 * dr);
	    float C = dpx * dpx + dpy * dpy - cr0 * cr0;

	    float grad_t;
	    if (fabsf (A) > 1e-10f)
	    {
	      float disc = B * B - 4.f * A * C;
	      if (disc < 0.f)
	      {
		gx += inv_xx;
		gy += inv_yx;
		continue;
	      }
	      float sq = sqrtf (disc);
	      float t1 = (-B + sq) / (2.f * A);
	      float t2 = (-B - sq) / (2.f * A);
	      grad_t = (cr0 + t1 * dr >= 0.f) ? t1 : t2;
	    }
	    else
	    {
	      if (fabsf (B) < 1e-10f)
	      {
		gx += inv_xx;
		gy += inv_yx;
		continue;
	      }
	      grad_t = -C / B;
	    }

	    uint32_t src = evaluate_color_line (stops, len, grad_t, extend);
	    row[px] = hb_packed_t<uint32_t> (hb_raster_src_over (src, (uint32_t) row[px]));
	    gx += inv_xx;
	    gy += inv_yx;
	  }
	}
      }
    }
    else
    {
      for (unsigned py = clip.min_y; py < clip.max_y; py++)
      {
	hb_packed_t<uint32_t> *row = (hb_packed_t<uint32_t> *) (surf->buffer.arrayZ + py * stride);
	const uint8_t *clip_row = clip.alpha.arrayZ + py * clip.stride;
	float gx = inv_xx * ((float) ((int) clip.min_x + ox) + 0.5f) + inv_xy * ((float) ((int) py + oy) + 0.5f) + inv_x0;
	float gy = inv_yx * ((float) ((int) clip.min_x + ox) + 0.5f) + inv_yy * ((float) ((int) py + oy) + 0.5f) + inv_y0;
	if (use_lut)
	{
	  for (unsigned px = clip.min_x; px < clip.max_x; px++)
	  {
	    uint8_t clip_alpha = clip_row[px];
	    if (clip_alpha == 0)
	    {
	      gx += inv_xx;
	      gy += inv_yx;
	      continue;
	    }
	    float dpx = gx - cx0, dpy = gy - cy0;
	    float B = -2.f * (dpx * cdx + dpy * cdy + cr0 * dr);
	    float C = dpx * dpx + dpy * dpy - cr0 * cr0;

	    float grad_t;
	    if (fabsf (A) > 1e-10f)
	    {
	      float disc = B * B - 4.f * A * C;
	      if (disc < 0.f)
	      {
		gx += inv_xx;
		gy += inv_yx;
		continue;
	      }
	      float sq = sqrtf (disc);
	      float t1 = (-B + sq) / (2.f * A);
	      float t2 = (-B - sq) / (2.f * A);
	      grad_t = (cr0 + t1 * dr >= 0.f) ? t1 : t2;
	    }
	    else
	    {
	      if (fabsf (B) < 1e-10f)
	      {
		gx += inv_xx;
		gy += inv_yx;
		continue;
	      }
	      grad_t = -C / B;
	    }

	    uint32_t src = lookup_gradient_lut (lut, grad_t, extend);
	    src = hb_raster_alpha_mul (src, clip_alpha);
	    row[px] = hb_packed_t<uint32_t> (hb_raster_src_over (src, (uint32_t) row[px]));
	    gx += inv_xx;
	    gy += inv_yx;
	  }
	}
	else
	{
	  for (unsigned px = clip.min_x; px < clip.max_x; px++)
	  {
	    uint8_t clip_alpha = clip_row[px];
	    if (clip_alpha == 0)
	    {
	      gx += inv_xx;
	      gy += inv_yx;
	      continue;
	    }
	    float dpx = gx - cx0, dpy = gy - cy0;
	    float B = -2.f * (dpx * cdx + dpy * cdy + cr0 * dr);
	    float C = dpx * dpx + dpy * dpy - cr0 * cr0;

	    float grad_t;
	    if (fabsf (A) > 1e-10f)
	    {
	      float disc = B * B - 4.f * A * C;
	      if (disc < 0.f)
	      {
		gx += inv_xx;
		gy += inv_yx;
		continue;
	      }
	      float sq = sqrtf (disc);
	      float t1 = (-B + sq) / (2.f * A);
	      float t2 = (-B - sq) / (2.f * A);
	      grad_t = (cr0 + t1 * dr >= 0.f) ? t1 : t2;
	    }
	    else
	    {
	      if (fabsf (B) < 1e-10f)
	      {
		gx += inv_xx;
		gy += inv_yx;
		continue;
	      }
	      grad_t = -C / B;
	    }

	    uint32_t src = evaluate_color_line (stops, len, grad_t, extend);
	    src = hb_raster_alpha_mul (src, clip_alpha);
	    row[px] = hb_packed_t<uint32_t> (hb_raster_src_over (src, (uint32_t) row[px]));
	    gx += inv_xx;
	    gy += inv_yx;
	  }
	}
      }
    }
  }

done:
  (void) stops_;
}

static void
hb_raster_paint_sweep_gradient (hb_paint_funcs_t *pfuncs HB_UNUSED,
				void *paint_data,
				hb_color_line_t *color_line,
				float cx, float cy,
				float start_angle,
				float end_angle,
				void *user_data HB_UNUSED)
{
  hb_raster_paint_t *c = (hb_raster_paint_t *) paint_data;

  ensure_initialized (c);

  hb_raster_image_t *surf = c->current_surface ();
  if (unlikely (!surf)) return;

  unsigned len = PREALLOCATED_COLOR_STOPS;
  hb_color_stop_t stops_[PREALLOCATED_COLOR_STOPS];
  hb_color_stop_t *stops = stops_;

  if (unlikely (!get_color_stops (c, color_line, &len, &stops)))
    return;
  float mn, mx;
  normalize_color_line (stops, len, &mn, &mx);

  hb_paint_extend_t extend = hb_color_line_get_extend (color_line);
  const hb_raster_clip_t &clip = c->current_clip ();
  unsigned clip_w = clip.max_x > clip.min_x ? clip.max_x - clip.min_x : 0;
  unsigned clip_h = clip.max_y > clip.min_y ? clip.max_y - clip.min_y : 0;
  bool use_lut = (uint64_t) clip_w * clip_h >= GRADIENT_LUT_MIN_PIXELS;
  uint32_t lut[GRADIENT_LUT_SIZE];
  if (use_lut)
    build_gradient_lut (stops, len, lut);

  /* Apply normalization to angle range */
  float a0 = start_angle + mn * (end_angle - start_angle);
  float a1 = start_angle + mx * (end_angle - start_angle);
  float angle_range = a1 - a0;

  /* Inverse transform */
  hb_transform_t<> t = c->current_effective_transform ();
  float det = t.xx * t.yy - t.xy * t.yx;
  if (fabsf (det) < 1e-10f || fabsf (angle_range) < 1e-10f) goto done;

  {
    float inv_det = 1.f / det;
    float inv_xx =  t.yy * inv_det;
    float inv_xy = -t.xy * inv_det;
    float inv_yx = -t.yx * inv_det;
    float inv_yy =  t.xx * inv_det;
    float inv_x0 = (t.xy * t.y0 - t.yy * t.x0) * inv_det;
    float inv_y0 = (t.yx * t.x0 - t.xx * t.y0) * inv_det;

    float inv_angle_range = 1.f / angle_range;

    unsigned stride = surf->extents.stride;
    int ox = surf->extents.x_origin;
    int oy = surf->extents.y_origin;

    if (clip.is_rect)
    {
      for (unsigned py = clip.min_y; py < clip.max_y; py++)
      {
	hb_packed_t<uint32_t> *row = (hb_packed_t<uint32_t> *) (surf->buffer.arrayZ + py * stride);
	float gx = inv_xx * ((float) ((int) clip.min_x + ox) + 0.5f) + inv_xy * ((float) ((int) py + oy) + 0.5f) + inv_x0;
	float gy = inv_yx * ((float) ((int) clip.min_x + ox) + 0.5f) + inv_yy * ((float) ((int) py + oy) + 0.5f) + inv_y0;
	if (use_lut)
	{
	  for (unsigned px = clip.min_x; px < clip.max_x; px++)
	  {
	    float angle = atan2f (gy - cy, gx - cx);
	    if (angle < 0) angle += (float) HB_2_PI;
	    float grad_t = (angle - a0) * inv_angle_range;
	    uint32_t src = lookup_gradient_lut (lut, grad_t, extend);
	    row[px] = hb_packed_t<uint32_t> (hb_raster_src_over (src, (uint32_t) row[px]));
	    gx += inv_xx;
	    gy += inv_yx;
	  }
	}
	else
	{
	  for (unsigned px = clip.min_x; px < clip.max_x; px++)
	  {
	    float angle = atan2f (gy - cy, gx - cx);
	    if (angle < 0) angle += (float) HB_2_PI;
	    float grad_t = (angle - a0) * inv_angle_range;
	    uint32_t src = evaluate_color_line (stops, len, grad_t, extend);
	    row[px] = hb_packed_t<uint32_t> (hb_raster_src_over (src, (uint32_t) row[px]));
	    gx += inv_xx;
	    gy += inv_yx;
	  }
	}
      }
    }
    else
    {
      for (unsigned py = clip.min_y; py < clip.max_y; py++)
      {
	hb_packed_t<uint32_t> *row = (hb_packed_t<uint32_t> *) (surf->buffer.arrayZ + py * stride);
	const uint8_t *clip_row = clip.alpha.arrayZ + py * clip.stride;
	float gx = inv_xx * ((float) ((int) clip.min_x + ox) + 0.5f) + inv_xy * ((float) ((int) py + oy) + 0.5f) + inv_x0;
	float gy = inv_yx * ((float) ((int) clip.min_x + ox) + 0.5f) + inv_yy * ((float) ((int) py + oy) + 0.5f) + inv_y0;
	if (use_lut)
	{
	  for (unsigned px = clip.min_x; px < clip.max_x; px++)
	  {
	    uint8_t clip_alpha = clip_row[px];
	    if (clip_alpha == 0)
	    {
	      gx += inv_xx;
	      gy += inv_yx;
	      continue;
	    }
	    float angle = atan2f (gy - cy, gx - cx);
	    if (angle < 0) angle += (float) HB_2_PI;
	    float grad_t = (angle - a0) * inv_angle_range;
	    uint32_t src = lookup_gradient_lut (lut, grad_t, extend);
	    src = hb_raster_alpha_mul (src, clip_alpha);
	    row[px] = hb_packed_t<uint32_t> (hb_raster_src_over (src, (uint32_t) row[px]));
	    gx += inv_xx;
	    gy += inv_yx;
	  }
	}
	else
	{
	  for (unsigned px = clip.min_x; px < clip.max_x; px++)
	  {
	    uint8_t clip_alpha = clip_row[px];
	    if (clip_alpha == 0)
	    {
	      gx += inv_xx;
	      gy += inv_yx;
	      continue;
	    }
	    float angle = atan2f (gy - cy, gx - cx);
	    if (angle < 0) angle += (float) HB_2_PI;
	    float grad_t = (angle - a0) * inv_angle_range;
	    uint32_t src = evaluate_color_line (stops, len, grad_t, extend);
	    src = hb_raster_alpha_mul (src, clip_alpha);
	    row[px] = hb_packed_t<uint32_t> (hb_raster_src_over (src, (uint32_t) row[px]));
	    gx += inv_xx;
	    gy += inv_yx;
	  }
	}
      }
    }
  }

done:
  (void) stops_;
}

static hb_bool_t
hb_raster_paint_custom_palette_color (hb_paint_funcs_t *funcs HB_UNUSED,
				      void *paint_data,
				      unsigned int color_index,
				      hb_color_t *color,
				      void *user_data HB_UNUSED)
{
  hb_raster_paint_t *c = (hb_raster_paint_t *) paint_data;
  if (likely (c->custom_palette && hb_map_has (c->custom_palette, color_index)))
  {
    *color = hb_map_get (c->custom_palette, color_index);
    return true;
  }
  return false;
}


/*
 * Lazy-loader singleton for paint funcs
 */

static inline void free_static_raster_paint_funcs ();

static struct hb_raster_paint_funcs_lazy_loader_t : hb_paint_funcs_lazy_loader_t<hb_raster_paint_funcs_lazy_loader_t>
{
  static hb_paint_funcs_t *create ()
  {
    hb_paint_funcs_t *funcs = hb_paint_funcs_create ();

    hb_paint_funcs_set_push_transform_func (funcs, hb_raster_paint_push_transform, nullptr, nullptr);
    hb_paint_funcs_set_pop_transform_func (funcs, hb_raster_paint_pop_transform, nullptr, nullptr);
    hb_paint_funcs_set_color_glyph_func (funcs, hb_raster_paint_color_glyph, nullptr, nullptr);
    hb_paint_funcs_set_push_clip_glyph_func (funcs, hb_raster_paint_push_clip_glyph, nullptr, nullptr);
    hb_paint_funcs_set_push_clip_rectangle_func (funcs, hb_raster_paint_push_clip_rectangle, nullptr, nullptr);
    hb_paint_funcs_set_pop_clip_func (funcs, hb_raster_paint_pop_clip, nullptr, nullptr);
    hb_paint_funcs_set_push_group_func (funcs, hb_raster_paint_push_group, nullptr, nullptr);
    hb_paint_funcs_set_pop_group_func (funcs, hb_raster_paint_pop_group, nullptr, nullptr);
    hb_paint_funcs_set_color_func (funcs, hb_raster_paint_color, nullptr, nullptr);
    hb_paint_funcs_set_image_func (funcs, hb_raster_paint_image, nullptr, nullptr);
    hb_paint_funcs_set_linear_gradient_func (funcs, hb_raster_paint_linear_gradient, nullptr, nullptr);
    hb_paint_funcs_set_radial_gradient_func (funcs, hb_raster_paint_radial_gradient, nullptr, nullptr);
    hb_paint_funcs_set_sweep_gradient_func (funcs, hb_raster_paint_sweep_gradient, nullptr, nullptr);
    hb_paint_funcs_set_custom_palette_color_func (funcs, hb_raster_paint_custom_palette_color, nullptr, nullptr);

    hb_paint_funcs_make_immutable (funcs);

    hb_atexit (free_static_raster_paint_funcs);

    return funcs;
  }
} static_raster_paint_funcs;

static inline void
free_static_raster_paint_funcs ()
{
  static_raster_paint_funcs.free_instance ();
}


/*
 * Public API
 */

/**
 * hb_raster_paint_create_or_fail:
 *
 * Creates a new color-glyph paint context.
 *
 * Return value: (transfer full):
 * A newly allocated #hb_raster_paint_t, or `NULL` on allocation failure.
 *
 * Since: 13.0.0
 **/
hb_raster_paint_t *
hb_raster_paint_create_or_fail (void)
{
  hb_raster_paint_t *paint = hb_object_create<hb_raster_paint_t> ();
  if (unlikely (!paint))
    return nullptr;

  paint->clip_rdr = hb_raster_draw_create_or_fail ();
  if (unlikely (!paint->clip_rdr))
  {
    hb_free (paint);
    return nullptr;
  }

  return paint;
}

/**
 * hb_raster_paint_reference: (skip)
 * @paint: a paint context
 *
 * Increases the reference count on @paint by one.
 *
 * Return value: (transfer full):
 * The referenced #hb_raster_paint_t.
 *
 * Since: 13.0.0
 **/
hb_raster_paint_t *
hb_raster_paint_reference (hb_raster_paint_t *paint)
{
  return hb_object_reference (paint);
}

/**
 * hb_raster_paint_destroy: (skip)
 * @paint: a paint context
 *
 * Decreases the reference count on @paint by one. When the
 * reference count reaches zero, the paint context is freed.
 *
 * Since: 13.0.0
 **/
void
hb_raster_paint_destroy (hb_raster_paint_t *paint)
{
  if (!hb_object_should_destroy (paint))
    return;

  hb_map_destroy (paint->custom_palette);
  hb_raster_draw_destroy (paint->clip_rdr);
  for (auto *s : paint->surface_stack)
    hb_raster_image_destroy (s);
  for (auto *s : paint->surface_cache)
    hb_raster_image_destroy (s);
  hb_object_actually_destroy (paint);
  hb_free (paint);
}

/**
 * hb_raster_paint_set_user_data: (skip)
 * @paint: a paint context
 * @key: the user-data key
 * @data: a pointer to the user data
 * @destroy: (nullable): a callback to call when @data is not needed anymore
 * @replace: whether to replace an existing data with the same key
 *
 * Attaches a user-data key/data pair to the specified paint context.
 *
 * Return value: `true` if success, `false` otherwise
 *
 * Since: 13.0.0
 **/
hb_bool_t
hb_raster_paint_set_user_data (hb_raster_paint_t  *paint,
			       hb_user_data_key_t *key,
			       void               *data,
			       hb_destroy_func_t   destroy,
			       hb_bool_t           replace)
{
  return hb_object_set_user_data (paint, key, data, destroy, replace);
}

/**
 * hb_raster_paint_get_user_data: (skip)
 * @paint: a paint context
 * @key: the user-data key
 *
 * Fetches the user-data associated with the specified key,
 * attached to the specified paint context.
 *
 * Return value: (transfer none):
 * A pointer to the user data
 *
 * Since: 13.0.0
 **/
void *
hb_raster_paint_get_user_data (hb_raster_paint_t  *paint,
			       hb_user_data_key_t *key)
{
  return hb_object_get_user_data (paint, key);
}

/**
 * hb_raster_paint_set_transform:
 * @paint: a paint context
 * @xx: xx component of the transform matrix
 * @yx: yx component of the transform matrix
 * @xy: xy component of the transform matrix
 * @yy: yy component of the transform matrix
 * @dx: x translation
 * @dy: y translation
 *
 * Sets the base 2×3 affine transform that maps from glyph-space
 * coordinates to pixel-space coordinates.
 *
 * Since: 13.0.0
 **/
void
hb_raster_paint_set_transform (hb_raster_paint_t *paint,
			       float xx, float yx,
			       float xy, float yy,
			       float dx, float dy)
{
  paint->base_transform = {xx, yx, xy, yy, dx, dy};
}

/**
 * hb_raster_paint_get_transform:
 * @paint: a paint context
 * @xx: (out) (nullable): xx component of the transform matrix
 * @yx: (out) (nullable): yx component of the transform matrix
 * @xy: (out) (nullable): xy component of the transform matrix
 * @yy: (out) (nullable): yy component of the transform matrix
 * @dx: (out) (nullable): x translation
 * @dy: (out) (nullable): y translation
 *
 * Gets the current base 2x3 affine transform.
 *
 * Since: 13.0.0
 **/
void
hb_raster_paint_get_transform (hb_raster_paint_t *paint,
			       float *xx, float *yx,
			       float *xy, float *yy,
			       float *dx, float *dy)
{
  if (xx) *xx = paint->base_transform.xx;
  if (yx) *yx = paint->base_transform.yx;
  if (xy) *xy = paint->base_transform.xy;
  if (yy) *yy = paint->base_transform.yy;
  if (dx) *dx = paint->base_transform.x0;
  if (dy) *dy = paint->base_transform.y0;
}

/**
 * hb_raster_paint_set_scale_factor:
 * @paint: a paint context
 * @x_scale_factor: x-axis minification factor
 * @y_scale_factor: y-axis minification factor
 *
 * Sets post-transform minification factors applied during painting.
 * Factors larger than 1 shrink the output in pixels. The default is 1.
 *
 * Since: 13.0.0
 **/
void
hb_raster_paint_set_scale_factor (hb_raster_paint_t *paint,
				  float x_scale_factor,
				  float y_scale_factor)
{
  paint->x_scale_factor = x_scale_factor > 0.f ? x_scale_factor : 1.f;
  paint->y_scale_factor = y_scale_factor > 0.f ? y_scale_factor : 1.f;
}

/**
 * hb_raster_paint_get_scale_factor:
 * @paint: a paint context
 * @x_scale_factor: (out) (nullable): x-axis minification factor
 * @y_scale_factor: (out) (nullable): y-axis minification factor
 *
 * Fetches the current post-transform minification factors.
 *
 * Since: 13.0.0
 **/
void
hb_raster_paint_get_scale_factor (hb_raster_paint_t *paint,
				  float *x_scale_factor,
				  float *y_scale_factor)
{
  if (x_scale_factor) *x_scale_factor = paint->x_scale_factor;
  if (y_scale_factor) *y_scale_factor = paint->y_scale_factor;
}

/**
 * hb_raster_paint_set_extents:
 * @paint: a paint context
 * @extents: the desired output extents
 *
 * Sets the output image extents (pixel rectangle).
 *
 * Call this before hb_font_paint_glyph() for each render.
 * A common pattern is:
 * |[<!-- language="plain" -->
 * hb_glyph_extents_t gext;
 * if (hb_font_get_glyph_extents (font, gid, &gext))
 *   hb_raster_paint_set_glyph_extents (paint, &gext);
 * ]|
 *
 * Since: 13.0.0
 **/
void
hb_raster_paint_set_extents (hb_raster_paint_t         *paint,
			     const hb_raster_extents_t *extents)
{
  paint->fixed_extents = *extents;
  paint->has_extents = true;
  if (paint->fixed_extents.stride == 0)
    paint->fixed_extents.stride = paint->fixed_extents.width * 4;
}

/**
 * hb_raster_paint_get_extents:
 * @paint: a paint context
 * @extents: (out) (nullable): where to write current extents
 *
 * Gets currently configured output extents.
 *
 * Return value: `true` if extents are set, `false` otherwise.
 *
 * Since: 13.0.0
 **/
hb_bool_t
hb_raster_paint_get_extents (hb_raster_paint_t   *paint,
			     hb_raster_extents_t *extents)
{
  if (!paint->has_extents)
    return false;

  if (extents)
    *extents = paint->fixed_extents;
  return true;
}

/**
 * hb_raster_paint_set_glyph_extents:
 * @paint: a paint context
 * @glyph_extents: glyph extents from hb_font_get_glyph_extents()
 *
 * Transforms @glyph_extents with the paint context's base transform and
 * sets the resulting output image extents.
 *
 * This is equivalent to computing a transformed bounding box in pixel
 * space and calling hb_raster_paint_set_extents().
 *
 * Return value: `true` if transformed extents are non-empty and set;
 * `false` otherwise.
 *
 * Since: 13.0.0
 **/
hb_bool_t
hb_raster_paint_set_glyph_extents (hb_raster_paint_t        *paint,
				   const hb_glyph_extents_t *glyph_extents)
{
  float x0 = (float) glyph_extents->x_bearing;
  float y0 = (float) glyph_extents->y_bearing;
  float x1 = (float) glyph_extents->x_bearing + glyph_extents->width;
  float y1 = (float) glyph_extents->y_bearing + glyph_extents->height;

  float xmin = hb_min (x0, x1);
  float xmax = hb_max (x0, x1);
  float ymin = hb_min (y0, y1);
  float ymax = hb_max (y0, y1);

  float px[4] = {xmin, xmin, xmax, xmax};
  float py[4] = {ymin, ymax, ymin, ymax};

  hb_transform_t<> t = paint->base_transform;
  paint->apply_scale_factor (t);

  float tx, ty;
  t.transform_point (px[0], py[0]);
  tx = px[0]; ty = py[0];
  float tx_min = tx, tx_max = tx;
  float ty_min = ty, ty_max = ty;

  for (unsigned i = 1; i < 4; i++)
  {
    t.transform_point (px[i], py[i]);
    tx_min = hb_min (tx_min, px[i]);
    tx_max = hb_max (tx_max, px[i]);
    ty_min = hb_min (ty_min, py[i]);
    ty_max = hb_max (ty_max, py[i]);
  }

  int ex0 = (int) floorf (tx_min);
  int ey0 = (int) floorf (ty_min);
  int ex1 = (int) ceilf  (tx_max);
  int ey1 = (int) ceilf  (ty_max);

  if (ex1 <= ex0 || ey1 <= ey0)
  {
    paint->fixed_extents = {};
    paint->has_extents = false;
    return false;
  }

  paint->fixed_extents = {
    ex0, ey0,
    (unsigned) (ex1 - ex0),
    (unsigned) (ey1 - ey0),
    0
  };
  paint->has_extents = true;
  if (paint->fixed_extents.stride == 0)
    paint->fixed_extents.stride = paint->fixed_extents.width * 4;
  return true;
}

/**
 * hb_raster_paint_set_foreground:
 * @paint: a paint context
 * @foreground: the foreground color
 *
 * Sets the foreground color used when paint callbacks request it
 * (e.g. `is_foreground` in color stops or solid fills).
 *
 * Since: 13.0.0
 **/
void
hb_raster_paint_set_foreground (hb_raster_paint_t *paint,
				hb_color_t         foreground)
{
  paint->foreground = foreground;
}

/**
 * hb_raster_paint_clear_custom_palette_colors:
 * @paint: a paint context.
 *
 * Clears all custom palette color overrides previously set on @paint.
 *
 * After this call, palette lookups use the selected font palette without
 * custom override entries.
 *
 * Since: 13.0.0
 **/
void
hb_raster_paint_clear_custom_palette_colors (hb_raster_paint_t *paint)
{
  if (paint->custom_palette)
    hb_map_clear (paint->custom_palette);
}

/**
 * hb_raster_paint_set_custom_palette_color:
 * @paint: a paint context.
 * @color_index: color index to override.
 * @color: replacement color.
 *
 * Overrides one font palette color entry for subsequent paint operations.
 * Overrides are keyed by @color_index and persist on @paint until cleared
 * (or replaced for the same index).
 *
 * These overrides are consulted by paint operations that resolve CPAL
 * entries.
 *
 * Return value: `true` if the override was set; `false` on allocation failure.
 *
 * Since: 13.0.0
 **/
hb_bool_t
hb_raster_paint_set_custom_palette_color (hb_raster_paint_t *paint,
					  unsigned int       color_index,
					  hb_color_t         color)
{
  if (unlikely (!paint->custom_palette))
  {
    paint->custom_palette = hb_map_create ();
    if (unlikely (!paint->custom_palette))
      return false;
  }
  hb_map_set (paint->custom_palette, color_index, color);
  return hb_map_allocation_successful (paint->custom_palette);
}

/**
 * hb_raster_paint_get_funcs:
 *
 * Fetches the singleton #hb_paint_funcs_t that renders color glyphs
 * into an #hb_raster_paint_t.  Pass the #hb_raster_paint_t as the
 * @paint_data argument when calling hb_font_paint_glyph().
 *
 * Return value: (transfer none):
 * The rasterizer paint functions
 *
 * Since: 13.0.0
 **/
hb_paint_funcs_t *
hb_raster_paint_get_funcs (void)
{
  return static_raster_paint_funcs.get_unconst ();
}

/**
 * hb_raster_paint_glyph:
 * @paint: a paint context
 * @font: font to paint from
 * @glyph: glyph ID to paint
 * @pen_x: glyph origin x in font coordinates (pre-transform)
 * @pen_y: glyph origin y in font coordinates (pre-transform)
 * @palette: palette index
 * @foreground: foreground color
 *
 * Convenience wrapper to paint one color glyph at (@pen_x, @pen_y) using
 * the paint context's current transform. The pen coordinates are applied
 * before minification and transformed by the current affine transform.
 *
 * Return value: `true` if painting succeeded, `false` otherwise.
 *
 * Since: 13.0.0
 **/
hb_bool_t
hb_raster_paint_glyph (hb_raster_paint_t *paint,
		       hb_font_t        *font,
		       hb_codepoint_t    glyph,
		       float             pen_x,
		       float             pen_y,
		       unsigned           palette,
		       hb_color_t         foreground)
{
  float xx = paint->base_transform.xx;
  float yx = paint->base_transform.yx;
  float xy = paint->base_transform.xy;
  float yy = paint->base_transform.yy;
  float dx = paint->base_transform.x0;
  float dy = paint->base_transform.y0;

  float tx = dx + xx * pen_x + xy * pen_y;
  float ty = dy + yx * pen_x + yy * pen_y;

  if (!paint->has_extents)
  {
    hb_glyph_extents_t ge;
    if (hb_font_get_glyph_extents (font, glyph, &ge))
    {
      hb_raster_paint_set_transform (paint, xx, yx, xy, yy, tx, ty);
      hb_raster_paint_set_glyph_extents (paint, &ge);
    }
  }

  hb_raster_paint_set_transform (paint, xx, yx, xy, yy, tx, ty);
  paint->svg_glyph = glyph;
  paint->svg_font = font;
  paint->svg_palette = palette;
  hb_bool_t ret = hb_font_paint_glyph_or_fail (font, glyph,
						hb_raster_paint_get_funcs (), paint,
						palette, foreground);
  paint->svg_glyph = 0;
  paint->svg_font = nullptr;
  paint->svg_palette = 0;
  hb_raster_paint_set_transform (paint, xx, yx, xy, yy, dx, dy);
  return ret;
}

/**
 * hb_raster_paint_render:
 * @paint: a paint context
 *
 * Extracts the rendered image after hb_font_paint_glyph() has
 * completed.  The paint context's surface stack is consumed and
 * the result returned as a new #hb_raster_image_t. Output format is
 * always @HB_RASTER_FORMAT_BGRA32.
 *
 * Call hb_font_paint_glyph() before calling this function.
 * hb_raster_paint_set_extents() or hb_raster_paint_set_glyph_extents()
 * must be called before painting; otherwise this function returns `NULL`.
 * Internal drawing state is cleared here so the same object can
 * be reused without client-side clearing.
 *
 * Return value: (transfer full):
 * A rendered #hb_raster_image_t. Returns `NULL` if extents were not set
 * or if allocation/configuration fails. If extents were set but nothing
 * was painted, returns an empty image.
 *
 * Since: 13.0.0
 **/
hb_raster_image_t *
hb_raster_paint_render (hb_raster_paint_t *paint)
{
  hb_raster_image_t *result = nullptr;

  if (unlikely (!paint->has_extents))
    goto fail;

  if (paint->surface_stack.length)
  {
    result = paint->surface_stack[0];
    /* Release any remaining group surfaces (shouldn't happen with
     * well-formed paint calls, but be safe). */
    for (unsigned i = 1; i < paint->surface_stack.length; i++)
      paint->release_surface (paint->surface_stack[i]);
    paint->surface_stack.clear ();
  }
  else
  {
    result = paint->acquire_surface ();
    if (unlikely (!result))
      goto fail;
  }

  /* Clean up stacks and reset auto-extents for next glyph. */
  paint->transform_stack.clear ();
  paint->release_all_clips ();
  hb_raster_draw_reset (paint->clip_rdr);
  paint->has_extents = false;
  paint->fixed_extents = {};

  return result;

fail:
  paint->transform_stack.clear ();
  paint->release_all_clips ();
  for (auto *s : paint->surface_stack)
    paint->release_surface (s);
  paint->surface_stack.clear ();
  hb_raster_draw_reset (paint->clip_rdr);
  paint->has_extents = false;
  paint->fixed_extents = {};
  return nullptr;
}

/**
 * hb_raster_paint_reset:
 * @paint: a paint context
 *
 * Resets the paint context to its initial state, clearing all
 * configuration while preserving internal image caches.
 *
 * Since: 13.0.0
 **/
void
hb_raster_paint_reset (hb_raster_paint_t *paint)
{
  paint->base_transform = {1, 0, 0, 1, 0, 0};
  paint->x_scale_factor = 1.f;
  paint->y_scale_factor = 1.f;
  paint->fixed_extents = {};
  paint->has_extents = false;
  paint->foreground = HB_COLOR (0, 0, 0, 255);
  hb_raster_paint_clear_custom_palette_colors (paint);
  paint->transform_stack.clear ();
  paint->release_all_clips ();
  for (auto *s : paint->surface_stack)
    paint->release_surface (s);
  paint->surface_stack.clear ();
}

/**
 * hb_raster_paint_recycle_image:
 * @paint: a paint context
 * @image: a raster image to recycle
 *
 * Recycles @image for reuse by subsequent render calls.
 * The caller transfers ownership of @image to @paint.
 *
 * Since: 13.0.0
 **/
void
hb_raster_paint_recycle_image (hb_raster_paint_t  *paint,
			       hb_raster_image_t  *image)
{
  paint->release_surface (image);
}
