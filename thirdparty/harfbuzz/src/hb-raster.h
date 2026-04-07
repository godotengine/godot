/*
 * Copyright (C) 2026  Behdad Esfahbod
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

#ifndef HB_RASTER_H
#define HB_RASTER_H

#include "hb.h"

HB_BEGIN_DECLS

/* Shared types */

/**
 * hb_raster_format_t:
 * @HB_RASTER_FORMAT_A8: 8-bit alpha-only coverage
 * @HB_RASTER_FORMAT_BGRA32: 32-bit BGRA color
 *
 * Pixel format for raster images.
 *
 * Since: 13.0.0
 */
typedef enum {
  HB_RASTER_FORMAT_A8     = 0,
  HB_RASTER_FORMAT_BGRA32 = 1,
} hb_raster_format_t;

/**
 * hb_raster_extents_t:
 * @x_origin: X coordinate of the left edge of the image in glyph space
 * @y_origin: Y coordinate of the bottom edge of the image in glyph space
 * @width: Width in pixels
 * @height: Height in pixels
 * @stride: Bytes per row; 0 means auto-calculate on input, filled on output
 *
 * Pixel-buffer extents for raster operations.
 *
 * Since: 13.0.0
 */
typedef struct hb_raster_extents_t {
  int      x_origin, y_origin;
  unsigned int width, height;
  unsigned int stride;
} hb_raster_extents_t;


/* hb_raster_image_t */

/**
 * hb_raster_image_t:
 *
 * An opaque raster image object holding a pixel buffer produced by
 * hb_raster_draw_render().  Use hb_raster_image_get_buffer() and
 * hb_raster_image_get_extents() to access the pixels.
 *
 * Since: 13.0.0
 **/
typedef struct hb_raster_image_t hb_raster_image_t;

HB_EXTERN hb_raster_image_t *
hb_raster_image_create_or_fail (void);

HB_EXTERN hb_raster_image_t *
hb_raster_image_reference (hb_raster_image_t *image);

HB_EXTERN void
hb_raster_image_destroy (hb_raster_image_t *image);

HB_EXTERN hb_bool_t
hb_raster_image_set_user_data (hb_raster_image_t  *image,
			       hb_user_data_key_t *key,
			       void               *data,
			       hb_destroy_func_t   destroy,
			       hb_bool_t           replace);

HB_EXTERN void *
hb_raster_image_get_user_data (hb_raster_image_t  *image,
			       hb_user_data_key_t *key);

HB_EXTERN hb_bool_t
hb_raster_image_configure (hb_raster_image_t         *image,
			   hb_raster_format_t        format,
			   const hb_raster_extents_t *extents);

HB_EXTERN void
hb_raster_image_clear (hb_raster_image_t *image);

HB_EXTERN const uint8_t *
hb_raster_image_get_buffer (hb_raster_image_t *image);

HB_EXTERN void
hb_raster_image_get_extents (hb_raster_image_t   *image,
			     hb_raster_extents_t *extents);

HB_EXTERN hb_raster_format_t
hb_raster_image_get_format (hb_raster_image_t *image);

HB_EXTERN hb_bool_t
hb_raster_image_deserialize_from_png_or_fail (hb_raster_image_t *image,
					      hb_blob_t         *png);

HB_EXTERN hb_blob_t *
hb_raster_image_serialize_to_png_or_fail (hb_raster_image_t *image);


/* hb_raster_draw_t */

/**
 * hb_raster_draw_t:
 *
 * An opaque outline rasterizer object.  Accumulates glyph outlines
 * via #hb_draw_funcs_t callbacks obtained from hb_raster_draw_get_funcs(),
 * then produces an #hb_raster_image_t with hb_raster_draw_render().
 *
 * Since: 13.0.0
 **/
typedef struct hb_raster_draw_t hb_raster_draw_t;

HB_EXTERN hb_raster_draw_t *
hb_raster_draw_create_or_fail (void);

HB_EXTERN hb_raster_draw_t *
hb_raster_draw_reference (hb_raster_draw_t *draw);

HB_EXTERN void
hb_raster_draw_destroy (hb_raster_draw_t *draw);

HB_EXTERN hb_bool_t
hb_raster_draw_set_user_data (hb_raster_draw_t   *draw,
			      hb_user_data_key_t *key,
			      void               *data,
			      hb_destroy_func_t   destroy,
			      hb_bool_t           replace);

HB_EXTERN void *
hb_raster_draw_get_user_data (hb_raster_draw_t   *draw,
			      hb_user_data_key_t *key);

HB_EXTERN void
hb_raster_draw_set_transform (hb_raster_draw_t *draw,
			      float xx, float yx,
			      float xy, float yy,
			      float dx, float dy);

HB_EXTERN void
hb_raster_draw_set_scale_factor (hb_raster_draw_t *draw,
				 float x_scale_factor,
				 float y_scale_factor);

HB_EXTERN void
hb_raster_draw_get_scale_factor (hb_raster_draw_t *draw,
				 float *x_scale_factor,
				 float *y_scale_factor);

HB_EXTERN void
hb_raster_draw_get_transform (hb_raster_draw_t *draw,
			      float *xx, float *yx,
			      float *xy, float *yy,
			      float *dx, float *dy);

HB_EXTERN void
hb_raster_draw_set_extents (hb_raster_draw_t          *draw,
			    const hb_raster_extents_t *extents);

HB_EXTERN hb_bool_t
hb_raster_draw_get_extents (hb_raster_draw_t    *draw,
			    hb_raster_extents_t *extents);

HB_EXTERN hb_bool_t
hb_raster_draw_set_glyph_extents (hb_raster_draw_t          *draw,
				  const hb_glyph_extents_t  *glyph_extents);

HB_EXTERN hb_draw_funcs_t *
hb_raster_draw_get_funcs (void);

HB_EXTERN void
hb_raster_draw_glyph (hb_raster_draw_t *draw,
		      hb_font_t       *font,
		      hb_codepoint_t   glyph,
		      float            pen_x,
		      float            pen_y);

HB_EXTERN hb_raster_image_t *
hb_raster_draw_render (hb_raster_draw_t *draw);

HB_EXTERN void
hb_raster_draw_reset (hb_raster_draw_t *draw);

HB_EXTERN void
hb_raster_draw_recycle_image (hb_raster_draw_t  *draw,
			      hb_raster_image_t *image);



/* hb_raster_paint_t */

/**
 * hb_raster_paint_t:
 *
 * An opaque color-glyph paint context.  Implements #hb_paint_funcs_t
 * callbacks that render COLRv0/v1 color glyphs into a BGRA32
 * #hb_raster_image_t.
 *
 * Since: 13.0.0
 **/
typedef struct hb_raster_paint_t hb_raster_paint_t;

HB_EXTERN hb_raster_paint_t *
hb_raster_paint_create_or_fail (void);

HB_EXTERN hb_raster_paint_t *
hb_raster_paint_reference (hb_raster_paint_t *paint);

HB_EXTERN void
hb_raster_paint_destroy (hb_raster_paint_t *paint);

HB_EXTERN hb_bool_t
hb_raster_paint_set_user_data (hb_raster_paint_t  *paint,
			       hb_user_data_key_t *key,
			       void               *data,
			       hb_destroy_func_t   destroy,
			       hb_bool_t           replace);

HB_EXTERN void *
hb_raster_paint_get_user_data (hb_raster_paint_t  *paint,
			       hb_user_data_key_t *key);

HB_EXTERN void
hb_raster_paint_set_transform (hb_raster_paint_t *paint,
			       float xx, float yx,
			       float xy, float yy,
			       float dx, float dy);

HB_EXTERN void
hb_raster_paint_get_transform (hb_raster_paint_t *paint,
			       float *xx, float *yx,
			       float *xy, float *yy,
			       float *dx, float *dy);

HB_EXTERN void
hb_raster_paint_set_scale_factor (hb_raster_paint_t *paint,
				  float x_scale_factor,
				  float y_scale_factor);

HB_EXTERN void
hb_raster_paint_get_scale_factor (hb_raster_paint_t *paint,
				  float *x_scale_factor,
				  float *y_scale_factor);

HB_EXTERN void
hb_raster_paint_set_extents (hb_raster_paint_t         *paint,
			     const hb_raster_extents_t *extents);

HB_EXTERN hb_bool_t
hb_raster_paint_get_extents (hb_raster_paint_t   *paint,
			     hb_raster_extents_t *extents);

HB_EXTERN hb_bool_t
hb_raster_paint_set_glyph_extents (hb_raster_paint_t         *paint,
				   const hb_glyph_extents_t  *glyph_extents);

HB_EXTERN void
hb_raster_paint_set_foreground (hb_raster_paint_t *paint,
				hb_color_t         foreground);

HB_EXTERN void
hb_raster_paint_clear_custom_palette_colors (hb_raster_paint_t *paint);

HB_EXTERN hb_bool_t
hb_raster_paint_set_custom_palette_color (hb_raster_paint_t *paint,
					  unsigned int       color_index,
					  hb_color_t         color);

HB_EXTERN hb_paint_funcs_t *
hb_raster_paint_get_funcs (void);

HB_EXTERN hb_bool_t
hb_raster_paint_glyph (hb_raster_paint_t *paint,
		       hb_font_t        *font,
		       hb_codepoint_t    glyph,
		       float             pen_x,
		       float             pen_y,
		       unsigned           palette,
		       hb_color_t         foreground);

HB_EXTERN hb_raster_image_t *
hb_raster_paint_render (hb_raster_paint_t *paint);

HB_EXTERN void
hb_raster_paint_reset (hb_raster_paint_t *paint);

HB_EXTERN void
hb_raster_paint_recycle_image (hb_raster_paint_t  *paint,
			       hb_raster_image_t  *image);


HB_END_DECLS

#endif /* HB_RASTER_H */
