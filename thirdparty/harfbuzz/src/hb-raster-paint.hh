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

#ifndef HB_RASTER_PAINT_HH
#define HB_RASTER_PAINT_HH

#include "hb.hh"

#include "hb-raster-image.hh"
#include "hb-geometry.hh"

/* hb_raster_clip_t — alpha mask for clipping */
struct hb_raster_clip_t
{
  hb_vector_t<uint8_t> alpha;	/* A8 mask, same extents as root surface */
  unsigned width  = 0;
  unsigned height = 0;
  unsigned stride = 0;

  /* Fast path: simple rectangle (no alpha buffer needed) */
  bool is_rect = true;
  int rect_x0 = 0, rect_y0 = 0;
  int rect_x1 = 0, rect_y1 = 0;

  /* Bounding box of non-zero alpha region (valid for both rect and mask) */
  unsigned min_x = 0, min_y = 0;
  unsigned max_x = 0, max_y = 0;

  void init_full (unsigned w, unsigned h)
  {
    width = w;
    height = h;
    stride = (w + 3u) & ~3u;
    is_rect = true;
    rect_x0 = 0;
    rect_y0 = 0;
    rect_x1 = (int) w;
    rect_y1 = (int) h;
    min_x = 0;
    min_y = 0;
    max_x = w;
    max_y = h;
  }

  void update_bounds_from_rect ()
  {
    min_x = (unsigned) hb_max (rect_x0, 0);
    min_y = (unsigned) hb_max (rect_y0, 0);
    max_x = (unsigned) hb_max (hb_min (rect_x1, (int) width), 0);
    max_y = (unsigned) hb_max (hb_min (rect_y1, (int) height), 0);
  }

  uint8_t get_alpha (unsigned x, unsigned y) const
  {
    if (is_rect)
      return ((int) x >= rect_x0 && (int) x < rect_x1 &&
	      (int) y >= rect_y0 && (int) y < rect_y1) ? 255 : 0;
    if (x >= width || y >= height) return 0;
    return alpha[y * stride + x];
  }
};


/* hb_raster_paint_t — color glyph paint context */
struct hb_raster_paint_t
{
  hb_object_header_t header;

  /* Configuration */
  hb_transform_t<>    base_transform     = {1, 0, 0, 1, 0, 0};
  float               x_scale_factor     = 1.f;
  float               y_scale_factor     = 1.f;
  hb_raster_extents_t fixed_extents      = {};
  bool                has_extents  = false;
  hb_color_t          foreground         = HB_COLOR (0, 0, 0, 255);
  hb_map_t           *custom_palette     = nullptr;

  /* SVG rendering state */
  hb_codepoint_t      svg_glyph          = 0;
  hb_font_t          *svg_font           = nullptr;
  unsigned            svg_palette        = 0;

  /* Stacks */
  hb_vector_t<hb_transform_t<>>     transform_stack;
  hb_vector_t<hb_raster_clip_t>     clip_stack;
  hb_vector_t<hb_raster_clip_t>     clip_cache;
  hb_vector_t<hb_raster_image_t *>  surface_stack;

  /* Cached surface pool (freelist for reuse across push/pop group) */
  hb_vector_t<hb_raster_image_t *>  surface_cache;
  hb_vector_t<hb_color_stop_t>      scratch_color_stops;

  /* Internal rasterizer for clip-to-glyph */
  hb_raster_draw_t *clip_rdr = nullptr;

  /* Helpers */

  hb_raster_image_t *acquire_surface ()
  {
    hb_raster_image_t *img;
    if (surface_cache.length)
      img = surface_cache.pop ();
    else
    {
      img = hb_raster_image_create_or_fail ();
      if (unlikely (!img)) return nullptr;
    }

    if (unlikely (!img->configure (HB_RASTER_FORMAT_BGRA32, fixed_extents)))
    {
      hb_raster_image_destroy (img);
      return nullptr;
    }
    img->clear ();
    return img;
  }

  void release_surface (hb_raster_image_t *img)
  {
    surface_cache.push (img);
  }

  hb_raster_clip_t acquire_clip (unsigned w, unsigned h)
  {
    hb_raster_clip_t clip;
    if (clip_cache.length)
      clip = clip_cache.pop ();
    clip.width = w;
    clip.height = h;
    clip.stride = (w + 3u) & ~3u;
    clip.is_rect = false;
    return clip;
  }

  void release_clip (hb_raster_clip_t &&clip)
  {
    if (clip.alpha.arrayZ)
      clip_cache.push (std::move (clip));
  }

  void release_all_clips ()
  {
    while (clip_stack.length)
      release_clip (clip_stack.pop ());
  }

  hb_raster_image_t *current_surface ()
  {
    return surface_stack.length ? surface_stack.tail () : nullptr;
  }

  hb_raster_clip_t &current_clip ()
  {
    return clip_stack.tail ();
  }

  hb_transform_t<> &current_transform ()
  {
    return transform_stack.tail ();
  }

  void apply_scale_factor (hb_transform_t<> &t) const
  {
    t.xx /= x_scale_factor;
    t.xy /= x_scale_factor;
    t.x0 /= x_scale_factor;
    t.yx /= y_scale_factor;
    t.yy /= y_scale_factor;
    t.y0 /= y_scale_factor;
  }

  hb_transform_t<> current_effective_transform ()
  {
    hb_transform_t<> t = current_transform ();
    apply_scale_factor (t);
    return t;
  }
};


#endif /* HB_RASTER_PAINT_HH */
