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

#ifndef HB_GPU_H
#define HB_GPU_H

#include "hb.h"

HB_BEGIN_DECLS


/**
 * hb_gpu_shader_lang_t:
 * @HB_GPU_SHADER_LANG_INVALID: Sentinel for an invalid or unspecified language.
 * @HB_GPU_SHADER_LANG_GLSL: GLSL (OpenGL 3.3 / OpenGL ES 3.0 / WebGL 2.0)
 * @HB_GPU_SHADER_LANG_WGSL: WGSL (WebGPU)
 * @HB_GPU_SHADER_LANG_MSL: MSL (Metal)
 * @HB_GPU_SHADER_LANG_HLSL: HLSL (Direct3D)
 *
 * Shader language variant.
 *
 * Since: 14.0.0
 */
typedef enum {
  HB_GPU_SHADER_LANG_INVALID,
  HB_GPU_SHADER_LANG_GLSL,
  HB_GPU_SHADER_LANG_WGSL,
  HB_GPU_SHADER_LANG_MSL,
  HB_GPU_SHADER_LANG_HLSL,
} hb_gpu_shader_lang_t;

/**
 * hb_gpu_shader_stage_t:
 * @HB_GPU_SHADER_STAGE_VERTEX: Vertex shader stage.
 * @HB_GPU_SHADER_STAGE_FRAGMENT: Fragment shader stage.
 *
 * Shader pipeline stage.
 *
 * Since: 14.2.0
 */
typedef enum {
  HB_GPU_SHADER_STAGE_VERTEX,
  HB_GPU_SHADER_STAGE_FRAGMENT,
} hb_gpu_shader_stage_t;

HB_EXTERN const char *
hb_gpu_shader_source (hb_gpu_shader_stage_t stage,
		      hb_gpu_shader_lang_t  lang);

HB_EXTERN const char *
hb_gpu_draw_shader_source (hb_gpu_shader_stage_t stage,
			   hb_gpu_shader_lang_t  lang);


/**
 * hb_gpu_draw_t:
 *
 * An opaque GPU shape encoder.  Accumulates outlines via draw
 * callbacks, then encodes them into a compact blob for GPU
 * rendering.
 *
 * Since: 14.0.0
 */
typedef struct hb_gpu_draw_t hb_gpu_draw_t;

HB_EXTERN hb_gpu_draw_t *
hb_gpu_draw_create_or_fail (void);

HB_EXTERN hb_gpu_draw_t *
hb_gpu_draw_reference (hb_gpu_draw_t *draw);

HB_EXTERN void
hb_gpu_draw_destroy (hb_gpu_draw_t *draw);

HB_EXTERN hb_bool_t
hb_gpu_draw_set_user_data (hb_gpu_draw_t     *draw,
			     hb_user_data_key_t *key,
			     void               *data,
			     hb_destroy_func_t   destroy,
			     hb_bool_t           replace);

HB_EXTERN void *
hb_gpu_draw_get_user_data (const hb_gpu_draw_t     *draw,
			     hb_user_data_key_t *key);


/* Scale */

HB_EXTERN void
hb_gpu_draw_set_scale (hb_gpu_draw_t *draw,
		       int            x_scale,
		       int            y_scale);

HB_EXTERN void
hb_gpu_draw_get_scale (const hb_gpu_draw_t *draw,
		       int                 *x_scale,
		       int                 *y_scale);

/* Draw */

HB_EXTERN hb_draw_funcs_t *
hb_gpu_draw_get_funcs (const hb_gpu_draw_t *draw);

HB_EXTERN void
hb_gpu_draw_glyph (hb_gpu_draw_t  *draw,
		   hb_font_t      *font,
		   hb_codepoint_t  glyph);

HB_EXTERN hb_bool_t
hb_gpu_draw_glyph_or_fail (hb_gpu_draw_t  *draw,
			   hb_font_t      *font,
			   hb_codepoint_t  glyph);

/* For arbitrary shapes beyond a single glyph, callers feed
 * outlines straight into the draw encoder via hb_draw_move_to()
 * / hb_draw_line_to() / hb_draw_quadratic_to() /
 * hb_draw_cubic_to() / hb_draw_close_path() with the funcs from
 * hb_gpu_draw_get_funcs() and the hb_gpu_draw_t as data.  The
 * helpers hb_draw_line() / hb_draw_rectangle() / hb_draw_circle()
 * cover common primitives (tapered lines, rectangles, circles).
 *
 * Coordinate system: the blob format quantizes coordinates to
 * 16 bits with a fixed scale of 4 units per coordinate step,
 * so the usable range is roughly +/-8000 units and the
 * effective precision is ~0.25 units.  Choose a coordinate
 * scale where:
 *
 *   (a) the overall bounding box stays within +/-8000, and
 *   (b) your smallest feature (stroke width, detail) is at
 *       least 1-2 units.
 *
 * Font-unit-style coordinates (e.g. a 1000-unit em) satisfy
 * both comfortably.  Tightly normalized coordinates (e.g. a
 * 0..1 unit square) do not: a 1-pixel stroke in that space
 * quantizes to zero and vanishes.  Scale the rendered quad /
 * vertex transform externally to reach larger on-screen sizes.
 */


/* Encode */

HB_EXTERN hb_blob_t *
hb_gpu_draw_encode (hb_gpu_draw_t      *draw,
                    hb_glyph_extents_t *extents);

HB_EXTERN void
hb_gpu_draw_clear (hb_gpu_draw_t *draw);

HB_EXTERN void
hb_gpu_draw_reset (hb_gpu_draw_t *draw);

HB_EXTERN void
hb_gpu_draw_recycle_blob (hb_gpu_draw_t *draw,
			    hb_blob_t      *blob);


/**
 * hb_gpu_paint_t:
 *
 * An opaque GPU color-glyph encoder.  Accumulates color-glyph
 * paint state via paint callbacks, then encodes it into a compact
 * blob for GPU rendering.
 *
 * Since: 14.2.0
 */
typedef struct hb_gpu_paint_t hb_gpu_paint_t;

HB_EXTERN hb_gpu_paint_t *
hb_gpu_paint_create_or_fail (void);

HB_EXTERN hb_gpu_paint_t *
hb_gpu_paint_reference (hb_gpu_paint_t *paint);

HB_EXTERN void
hb_gpu_paint_destroy (hb_gpu_paint_t *paint);

HB_EXTERN hb_bool_t
hb_gpu_paint_set_user_data (hb_gpu_paint_t     *paint,
			    hb_user_data_key_t *key,
			    void               *data,
			    hb_destroy_func_t   destroy,
			    hb_bool_t           replace);

HB_EXTERN void *
hb_gpu_paint_get_user_data (const hb_gpu_paint_t *paint,
			    hb_user_data_key_t   *key);

HB_EXTERN hb_paint_funcs_t *
hb_gpu_paint_get_funcs (const hb_gpu_paint_t *paint);

HB_EXTERN void
hb_gpu_paint_set_palette (hb_gpu_paint_t *paint,
			  unsigned        palette);

HB_EXTERN unsigned
hb_gpu_paint_get_palette (const hb_gpu_paint_t *paint);

HB_EXTERN void
hb_gpu_paint_clear_custom_palette_colors (hb_gpu_paint_t *paint);

HB_EXTERN hb_bool_t
hb_gpu_paint_set_custom_palette_color (hb_gpu_paint_t *paint,
				       unsigned int    color_index,
				       hb_color_t      color);

HB_EXTERN void
hb_gpu_paint_set_scale (hb_gpu_paint_t *paint,
			int             x_scale,
			int             y_scale);

HB_EXTERN void
hb_gpu_paint_get_scale (const hb_gpu_paint_t *paint,
			int                  *x_scale,
			int                  *y_scale);

HB_EXTERN void
hb_gpu_paint_glyph (hb_gpu_paint_t *paint,
		    hb_font_t      *font,
		    hb_codepoint_t  glyph);

HB_EXTERN hb_bool_t
hb_gpu_paint_glyph_or_fail (hb_gpu_paint_t *paint,
			    hb_font_t      *font,
			    hb_codepoint_t  glyph);

HB_EXTERN hb_blob_t *
hb_gpu_paint_encode (hb_gpu_paint_t     *paint,
		     hb_glyph_extents_t *extents);

HB_EXTERN void
hb_gpu_paint_clear (hb_gpu_paint_t *paint);

HB_EXTERN void
hb_gpu_paint_reset (hb_gpu_paint_t *paint);

HB_EXTERN void
hb_gpu_paint_recycle_blob (hb_gpu_paint_t *paint,
			   hb_blob_t      *blob);

HB_EXTERN const char *
hb_gpu_paint_shader_source (hb_gpu_shader_stage_t stage,
			    hb_gpu_shader_lang_t  lang);


HB_END_DECLS


#if defined(__cplusplus) && defined(HB_CPLUSPLUS_HH)
namespace hb {
HB_DEFINE_VTABLE (gpu_draw,  nullptr);
HB_DEFINE_VTABLE (gpu_paint, nullptr);
} // namespace hb
#endif

#endif /* HB_GPU_H */
