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
#include "hb-vector-paint.hh"
#include "hb-vector-path.hh"
#include "hb-font.hh"

/**
 * hb_vector_paint_create_or_fail:
 * @format: output format.
 *
 * Creates a new paint context for vector output.
 *
 * Return value: (nullable): a newly allocated #hb_vector_paint_t, or `NULL` on failure.
 *
 * Since: 13.0.0
 */
hb_vector_paint_t *
hb_vector_paint_create_or_fail (hb_vector_format_t format)
{
  switch (format)
  {
    case HB_VECTOR_FORMAT_SVG:
    case HB_VECTOR_FORMAT_PDF:
      break;
    case HB_VECTOR_FORMAT_INVALID: default:
      return nullptr;
  }

  hb_vector_paint_t *paint = hb_object_create<hb_vector_paint_t> ();
  if (unlikely (!paint))
    return nullptr;
  paint->format = format;

  paint->active_color_glyphs = hb_set_create ();
  paint->defs.alloc (4096);
  paint->path.alloc (2048);
  paint->captured_scratch.alloc (4096);
  paint->color_stops_scratch.alloc (16);

  return paint;
}

/**
 * hb_vector_paint_reference:
 * @paint: a paint context.
 *
 * Increases the reference count of @paint.
 *
 * Return value: (transfer full): referenced @paint.
 *
 * Since: 13.0.0
 */
hb_vector_paint_t *
hb_vector_paint_reference (hb_vector_paint_t *paint)
{
  return hb_object_reference (paint);
}

/**
 * hb_vector_paint_destroy:
 * @paint: a paint context.
 *
 * Decreases the reference count of @paint and destroys it when it reaches zero.
 *
 * Since: 13.0.0
 */
void
hb_vector_paint_destroy (hb_vector_paint_t *paint)
{
  if (!hb_object_should_destroy (paint))
    return;

  if (paint->format == HB_VECTOR_FORMAT_PDF)
    hb_vector_paint_pdf_free_resources (paint);
  hb_blob_destroy (paint->recycled_blob);
  hb_set_destroy (paint->active_color_glyphs);
  hb_free (paint->id_prefix);
  hb_object_actually_destroy (paint);
  hb_free (paint);
}

/**
 * hb_vector_paint_set_user_data:
 * @paint: a paint context.
 * @key: user-data key.
 * @data: user-data value.
 * @destroy: (nullable): destroy callback for @data.
 * @replace: whether to replace an existing value for @key.
 *
 * Attaches user data to @paint.
 *
 * Return value: `true` on success, `false` otherwise.
 *
 * Since: 13.0.0
 */
hb_bool_t
hb_vector_paint_set_user_data (hb_vector_paint_t  *paint,
                               hb_user_data_key_t *key,
                               void               *data,
                               hb_destroy_func_t   destroy,
                               hb_bool_t           replace)
{
  return hb_object_set_user_data (paint, key, data, destroy, replace);
}

/**
 * hb_vector_paint_get_user_data:
 * @paint: a paint context.
 * @key: user-data key.
 *
 * Gets previously attached user data from @paint.
 *
 * Return value: (nullable): user-data value associated with @key.
 *
 * Since: 13.0.0
 */
void *
hb_vector_paint_get_user_data (const hb_vector_paint_t  *paint,
                               hb_user_data_key_t *key)
{
  return hb_object_get_user_data (paint, key);
}

/**
 * hb_vector_paint_set_transform:
 * @paint: a paint context.
 * @xx: transform xx component.
 * @yx: transform yx component.
 * @xy: transform xy component.
 * @yy: transform yy component.
 * @dx: transform x translation.
 * @dy: transform y translation.
 *
 * Sets the affine transform used when painting glyphs.
 *
 * Since: 13.0.0
 */
void
hb_vector_paint_set_transform (hb_vector_paint_t *paint,
                               float xx, float yx,
                               float xy, float yy,
                               float dx, float dy)
{
  paint->transform = {xx, yx, xy, yy, dx, dy};
}

/**
 * hb_vector_paint_get_transform:
 * @paint: a paint context.
 * @xx: (out) (nullable): transform xx component.
 * @yx: (out) (nullable): transform yx component.
 * @xy: (out) (nullable): transform xy component.
 * @yy: (out) (nullable): transform yy component.
 * @dx: (out) (nullable): transform x translation.
 * @dy: (out) (nullable): transform y translation.
 *
 * Gets the affine transform used when painting glyphs.
 *
 * Since: 13.0.0
 */
void
hb_vector_paint_get_transform (const hb_vector_paint_t *paint,
                               float *xx, float *yx,
                               float *xy, float *yy,
                               float *dx, float *dy)
{
  if (xx) *xx = paint->transform.xx;
  if (yx) *yx = paint->transform.yx;
  if (xy) *xy = paint->transform.xy;
  if (yy) *yy = paint->transform.yy;
  if (dx) *dx = paint->transform.x0;
  if (dy) *dy = paint->transform.y0;
}

/**
 * hb_vector_paint_set_scale_factor:
 * @paint: a paint context.
 * @x_scale_factor: x scale factor.
 * @y_scale_factor: y scale factor.
 *
 * Sets additional output scaling factors.
 *
 * Since: 13.0.0
 */
void
hb_vector_paint_set_scale_factor (hb_vector_paint_t *paint,
                                  float x_scale_factor,
                                  float y_scale_factor)
{
  paint->x_scale_factor = x_scale_factor > 0.f ? x_scale_factor : 1.f;
  paint->y_scale_factor = y_scale_factor > 0.f ? y_scale_factor : 1.f;
}

/**
 * hb_vector_paint_get_scale_factor:
 * @paint: a paint context.
 * @x_scale_factor: (out) (nullable): x scale factor.
 * @y_scale_factor: (out) (nullable): y scale factor.
 *
 * Gets additional output scaling factors.
 *
 * Since: 13.0.0
 */
void
hb_vector_paint_get_scale_factor (const hb_vector_paint_t *paint,
                                  float *x_scale_factor,
                                  float *y_scale_factor)
{
  if (x_scale_factor) *x_scale_factor = paint->x_scale_factor;
  if (y_scale_factor) *y_scale_factor = paint->y_scale_factor;
}

/**
 * hb_vector_paint_set_extents:
 * @paint: a paint context.
 * @extents: (nullable): output extents to set or expand.
 *
 * Sets or expands output extents on @paint. Passing `NULL` clears extents.
 *
 * Since: 13.0.0
 */
void
hb_vector_paint_set_extents (hb_vector_paint_t *paint,
                             const hb_vector_extents_t *extents)
{
  if (!extents)
  {
    paint->extents = {0, 0, 0, 0};
    paint->has_extents = false;
    return;
  }

  if (extents->width == 0.f || extents->height == 0.f)
    return;

  /* Caller-supplied extents are in input-space; divide by
   * scale_factor so they end up in output-space, matching
   * the per-glyph extents accumulated via
   * hb_vector_set_glyph_extents_common (which applies the
   * same divide).  Normalize so origin is the min corner and
   * width/height are positive — callers may pass a
   * glyph-extents-style box with negative height. */
  float x0 = extents->x / paint->x_scale_factor;
  float y0 = extents->y / paint->y_scale_factor;
  float x1 = x0 + extents->width  / paint->x_scale_factor;
  float y1 = y0 + extents->height / paint->y_scale_factor;
  hb_vector_extents_t e = {
    hb_min (x0, x1), hb_min (y0, y1),
    fabsf (x1 - x0), fabsf (y1 - y0),
  };

  if (paint->has_extents)
  {
    float x0 = hb_min (paint->extents.x, e.x);
    float y0 = hb_min (paint->extents.y, e.y);
    float x1 = hb_max (paint->extents.x + paint->extents.width,
		       e.x + e.width);
    float y1 = hb_max (paint->extents.y + paint->extents.height,
		       e.y + e.height);
    paint->extents = {x0, y0, x1 - x0, y1 - y0};
  }
  else
  {
    paint->extents = e;
    paint->has_extents = true;
  }
}

/**
 * hb_vector_paint_get_extents:
 * @paint: a paint context.
 * @extents: (out) (nullable): where to store current output extents.
 *
 * Gets current output extents from @paint.
 *
 * Return value: `true` if extents are set, `false` otherwise.
 *
 * Since: 13.0.0
 */
hb_bool_t
hb_vector_paint_get_extents (const hb_vector_paint_t *paint,
                             hb_vector_extents_t *extents)
{
  if (!paint->has_extents)
    return false;

  if (extents)
    *extents = paint->extents;
  return true;
}

/**
 * hb_vector_paint_set_glyph_extents:
 * @paint: a paint context.
 * @glyph_extents: glyph extents in font units.
 *
 * Expands @paint extents using @glyph_extents under the current transform.
 *
 * Return value: `true` on success, `false` otherwise.
 *
 * Since: 13.0.0
 */
hb_bool_t
hb_vector_paint_set_glyph_extents (hb_vector_paint_t *paint,
                                   const hb_glyph_extents_t *glyph_extents)
{
  hb_bool_t has_extents = paint->has_extents;
  hb_bool_t ret = hb_vector_set_glyph_extents_common (paint->transform,
						   paint->x_scale_factor,
						   paint->y_scale_factor,
						   glyph_extents,
						   &paint->extents,
						   &has_extents);
  paint->has_extents = has_extents;
  return ret;
}

/**
 * hb_vector_paint_set_foreground:
 * @paint: a paint context.
 * @foreground: foreground color used for COLR foreground paints.
 *
 * Sets fallback foreground color used by paint operations.
 *
 * Since: 13.0.0
 */
void
hb_vector_paint_set_foreground (hb_vector_paint_t *paint,
                                hb_color_t foreground)
{
  paint->foreground = foreground;
}

/**
 * hb_vector_paint_get_foreground:
 * @paint: a paint context.
 *
 * Returns the foreground color previously set on @paint, or the
 * default opaque black if none was set.
 *
 * Return value: the foreground color.
 *
 * Since: 14.2.0
 */
hb_color_t
hb_vector_paint_get_foreground (const hb_vector_paint_t *paint)
{
  return paint->foreground;
}

/**
 * hb_vector_paint_set_background:
 * @paint: a paint context.
 * @background: background color.
 *
 * Sets the background color for @paint.  If set to a non-transparent
 * value, the renderer emits a filled rectangle covering the extents
 * behind all glyph content.  Default is transparent (no background).
 *
 * Since: 14.2.0
 */
void
hb_vector_paint_set_background (hb_vector_paint_t *paint,
                                hb_color_t background)
{
  paint->background = background;
}

/**
 * hb_vector_paint_get_background:
 * @paint: a paint context.
 *
 * Returns the background color previously set on @paint, or
 * transparent if none was set.
 *
 * Return value: the background color.
 *
 * Since: 14.2.0
 */
hb_color_t
hb_vector_paint_get_background (const hb_vector_paint_t *paint)
{
  return paint->background;
}

/**
 * hb_vector_paint_set_palette:
 * @paint: a paint context.
 * @palette: palette index for color glyph painting.
 *
 * Sets the color palette index used by paint operations.
 *
 * Since: 13.0.0
 */
void
hb_vector_paint_set_palette (hb_vector_paint_t *paint,
                             int palette)
{
  paint->palette = palette;
}

/**
 * hb_vector_paint_get_palette:
 * @paint: a paint context.
 *
 * Returns the palette index previously set on @paint, or 0 if none
 * was set.
 *
 * Return value: the palette index.
 *
 * Since: 14.2.0
 */
int
hb_vector_paint_get_palette (const hb_vector_paint_t *paint)
{
  return paint->palette;
}

/**
 * hb_vector_paint_set_custom_palette_color:
 * @paint: a paint context.
 * @color_index: color index to override.
 * @color: replacement color.
 *
 * Overrides one font palette color entry for subsequent paint operations.
 * Overrides are keyed by @color_index and persist on @paint until cleared
 * (or replaced for the same index).
 *
 * These overrides are consulted by paint operations that resolve CPAL
 * entries, including SVG glyph content using `var(--colorN)`.
 *
 * Since: 13.0.0
 */
void
hb_vector_paint_set_custom_palette_color (hb_vector_paint_t *paint,
                                          unsigned color_index,
                                          hb_color_t color)
{
  paint->custom_palette_colors.set (color_index, color);
}

/**
 * hb_vector_paint_clear_custom_palette_colors:
 * @paint: a paint context.
 *
 * Clears all custom palette color overrides previously set on @paint.
 *
 * After this call, palette lookups use the selected font palette without
 * custom override entries.
 *
 * Since: 13.0.0
 */
void
hb_vector_paint_clear_custom_palette_colors (hb_vector_paint_t *paint)
{
  paint->custom_palette_colors.clear ();
}

/**
 * hb_vector_paint_get_format:
 * @paint: a vector paint context.
 *
 * Gets the output format @paint was created with.
 *
 * Return value: the output format.
 *
 * Since: 14.2.0
 */
hb_vector_format_t
hb_vector_paint_get_format (const hb_vector_paint_t *paint)
{
  return paint->format;
}

/**
 * hb_vector_paint_get_funcs:
 * @paint: a vector paint context.
 *
 * Gets paint callbacks for emitting paint operations into @paint.
 * Pass @paint as the @paint_data argument when calling them.
 *
 * Return value: (transfer none): immutable #hb_paint_funcs_t.
 *
 * Since: 14.2.0
 */
hb_paint_funcs_t *
hb_vector_paint_get_funcs (const hb_vector_paint_t *paint)
{
  switch (paint ? paint->format : HB_VECTOR_FORMAT_INVALID)
  {
    case HB_VECTOR_FORMAT_PDF:
      return hb_vector_paint_pdf_funcs_get ();
    case HB_VECTOR_FORMAT_SVG:
      return hb_vector_paint_svg_funcs_get ();
    case HB_VECTOR_FORMAT_INVALID:
    default:
      return nullptr;
  }
}


static hb_bool_t
hb_vector_paint_glyph_impl (hb_vector_paint_t *paint,
			    hb_font_t         *font,
			    hb_codepoint_t     glyph,
			    hb_vector_extents_mode_t extents_mode,
			    hb_bool_t          fallible)
{
  if (extents_mode == HB_VECTOR_EXTENTS_MODE_EXPAND)
  {
    hb_glyph_extents_t ge;
    if (hb_font_get_glyph_extents (font, glyph, &ge))
    {
      hb_bool_t has_extents = paint->has_extents;
      hb_vector_set_glyph_extents_common (paint->transform,
					  paint->x_scale_factor,
					  paint->y_scale_factor,
					  &ge,
					  &paint->extents,
					  &has_extents);
      paint->has_extents = has_extents;
    }
  }

  hb_paint_funcs_t *funcs = hb_vector_paint_get_funcs (paint);
  hb_paint_push_transform (funcs, paint,
			   paint->transform.xx, paint->transform.yx,
			   paint->transform.xy, paint->transform.yy,
			   paint->transform.x0, paint->transform.y0);

  hb_bool_t ret = true;
  if (fallible)
    ret = hb_font_paint_glyph_or_fail (font, glyph,
				       funcs, paint,
				       (unsigned) paint->palette,
				       paint->foreground);
  else
    hb_font_paint_glyph (font, glyph,
			 funcs, paint,
			 (unsigned) paint->palette,
			 paint->foreground);

  hb_paint_pop_transform (funcs, paint);
  return ret;
}

/**
 * hb_vector_paint_glyph_or_fail:
 * @paint: a paint context.
 * @font: font object.
 * @glyph: glyph ID.
 * @extents_mode: extents update mode.
 *
 * Convenience to paint one color glyph into @paint.
 * Equivalent to:
 *
 * |[<!-- language="plain" -->
 * // extend extents if requested
 * hb_paint_funcs_t *funcs = hb_vector_paint_get_funcs (paint);
 * hb_paint_push_transform (funcs, paint, ...transform...);
 * hb_font_paint_glyph_or_fail (font, glyph, funcs, paint,
 *   palette, foreground);
 * hb_paint_pop_transform (funcs, paint);
 * ]|
 *
 * Return value: `true` if glyph paint data was emitted, `false` otherwise.
 *
 * Since: 14.2.0
 */
hb_bool_t
hb_vector_paint_glyph_or_fail (hb_vector_paint_t *paint,
			       hb_font_t         *font,
			       hb_codepoint_t     glyph,
			       hb_vector_extents_mode_t extents_mode)
{
  return hb_vector_paint_glyph_impl (paint, font, glyph,
				     extents_mode, true);
}

/**
 * hb_vector_paint_glyph:
 * @paint: a paint context.
 * @font: font object.
 * @glyph: glyph ID.
 * @extents_mode: extents update mode.
 *
 * Paints one glyph into @paint.  Unlike
 * hb_vector_paint_glyph_or_fail(), glyphs with no color paint
 * data fall back to a synthesized foreground-colored outline,
 * so any glyph with an outline or bitmap image produces output.
 *
 * Since: 14.2.0
 */
void
hb_vector_paint_glyph (hb_vector_paint_t *paint,
		       hb_font_t         *font,
		       hb_codepoint_t     glyph,
		       hb_vector_extents_mode_t extents_mode)
{
  hb_vector_paint_glyph_impl (paint, font, glyph,
			      extents_mode, false);
}

/**
 * hb_vector_paint_set_svg_prefix:
 * @paint: a paint context.
 * @prefix: a null-terminated ASCII string to prepend to every emitted
 *          SVG `id` and `url(#...)` reference, or `NULL` for none.
 *
 * Namespaces the paint's SVG output.  Callers that inject multiple
 * hb-vector SVGs into the same document (e.g. several glyph previews
 * on one page) must set a distinct prefix per context so that the
 * short IDs hb-vector uses for clipPaths, gradients, and use-refs
 * don't collide in the DOM.
 *
 * No effect on PDF output.
 *
 * Since: 14.2.0
 */
void
hb_vector_paint_set_svg_prefix (hb_vector_paint_t *paint,
                                const char *prefix)
{
  char *old = paint->id_prefix;
  paint->id_prefix = nullptr;
  paint->id_prefix_length = 0;
  if (prefix && *prefix)
  {
    unsigned len = strlen (prefix);
    char *copy = (char *) hb_malloc (len + 1);
    if (likely (copy))
    {
      hb_memcpy (copy, prefix, len);
      copy[len] = '\0';
      paint->id_prefix = copy;
      paint->id_prefix_length = len;
    }
  }
  hb_free (old);
}

/**
 * hb_vector_paint_get_svg_prefix:
 * @paint: a paint context.
 *
 * Returns the SVG id prefix previously set on @paint, or `""` if
 * none was set.  The pointer remains valid until the next call to
 * hb_vector_paint_set_svg_prefix() or hb_vector_paint_reset() on the
 * same context.
 *
 * Return value: the SVG id prefix.
 *
 * Since: 14.2.0
 */
const char *
hb_vector_paint_get_svg_prefix (const hb_vector_paint_t *paint)
{
  return paint->id_prefix ? paint->id_prefix : "";
}

/**
 * hb_vector_paint_set_precision:
 * @paint: a paint context.
 * @precision: decimal precision.
 *
 * Sets numeric output precision for paint output.
 *
 * Since: 14.2.0
 */
void
hb_vector_paint_set_precision (hb_vector_paint_t *paint,
                                   unsigned precision)
{
  paint->set_precision (precision);
}

/**
 * hb_vector_paint_get_precision:
 * @paint: a paint context.
 *
 * Returns the numeric output precision previously set on @paint,
 * or the default if none was set.
 *
 * Return value: the precision.
 *
 * Since: 14.2.0
 */
unsigned
hb_vector_paint_get_precision (const hb_vector_paint_t *paint)
{
  return paint->get_precision ();
}

/**
 * hb_vector_paint_clear:
 * @paint: a paint context.
 *
 * Discards accumulated paint output so @paint can be reused for
 * another render.  User configuration (transform, scale factors,
 * precision, foreground, palette, custom palette colors)
 * is preserved.  Call hb_vector_paint_reset() to also reset
 * user configuration to defaults.
 *
 * Since: 14.2.0
 */
void
hb_vector_paint_clear (hb_vector_paint_t *paint)
{
  paint->extents = {0, 0, 0, 0};
  paint->has_extents = false;

  if (paint->format == HB_VECTOR_FORMAT_PDF)
    hb_vector_paint_pdf_free_resources (paint);
  paint->defs.clear ();
  paint->path.clear ();
  paint->group_stack.clear ();
  paint->transform_group_open_mask = 0;
  paint->transform_group_depth = 0;
  paint->transform_group_overflow_depth = 0;
  paint->clip_rect_counter = 0;
  paint->clip_path_counter = 0;
  paint->gradient_counter = 0;
  paint->color_glyph_depth = 0;
  paint->path_def_count = 0;
  hb_set_clear (paint->active_color_glyphs);
  paint->color_stops_scratch.clear ();
  paint->captured_scratch.clear ();
}


/**
 * hb_vector_paint_render:
 * @paint: a paint context.
 *
 * Renders accumulated paint content to an output blob.
 *
 * Return value: (transfer full) (nullable): output blob, or `NULL` if rendering cannot proceed.
 *
 * Since: 13.0.0
 */
hb_blob_t *
hb_vector_paint_render (hb_vector_paint_t *paint)
{
  switch (paint->format)
  {
    case HB_VECTOR_FORMAT_SVG:
      return hb_vector_paint_render_svg (paint);

    case HB_VECTOR_FORMAT_PDF:
      return hb_vector_paint_render_pdf (paint);

    case HB_VECTOR_FORMAT_INVALID: default:
      return nullptr;
  }
}

/**
 * hb_vector_paint_reset:
 * @paint: a paint context.
 *
 * Resets @paint state and clears accumulated content.
 *
 * Since: 13.0.0
 */
void
hb_vector_paint_reset (hb_vector_paint_t *paint)
{
  paint->transform = {1, 0, 0, 1, 0, 0};
  paint->x_scale_factor = 1.f;
  paint->y_scale_factor = 1.f;
  paint->foreground = HB_COLOR (0, 0, 0, 255);
  paint->palette = 0;
  paint->set_precision (2);
  hb_vector_paint_clear (paint);
}

/**
 * hb_vector_paint_recycle_blob:
 * @paint: a paint context.
 * @blob: (nullable): previously rendered blob to recycle.
 *
 * Provides a blob for internal buffer reuse by later render calls.
 *
 * Since: 13.0.0
 */
void
hb_vector_paint_recycle_blob (hb_vector_paint_t *paint,
                              hb_blob_t *blob)
{
  hb_blob_destroy (paint->recycled_blob);
  paint->recycled_blob = nullptr;
  if (!blob || blob == hb_blob_get_empty ())
    return;
  paint->recycled_blob = blob;
}
