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

#include "hb-vector-draw.hh"
#include "hb-vector-path.hh"

#include <math.h>
#include <string.h>

HB_UNUSED static inline bool
hb_vector_svg_buffer_contains (const hb_vector_buf_t &buf, const char *needle)
{
  unsigned nlen = (unsigned) strlen (needle);
  if (!nlen || buf.length < nlen)
    return false;

  for (unsigned i = 0; i + nlen <= buf.length; i++)
    if (buf.arrayZ[i] == needle[0] &&
        !memcmp (buf.arrayZ + i, needle, nlen))
      return true;
  return false;
}

/* ---- SVG draw callbacks ---- */

static void hb_vector_draw_svg_move_to (hb_draw_funcs_t *, void *dd, hb_draw_state_t *, float x, float y, void *)
{ auto *d = (hb_vector_draw_t *) dd; d->path.append_c ('M'); d->append_xy_svg (x, y); }

static void hb_vector_draw_svg_line_to (hb_draw_funcs_t *, void *dd, hb_draw_state_t *, float x, float y, void *)
{ auto *d = (hb_vector_draw_t *) dd; d->path.append_c ('L'); d->append_xy_svg (x, y); }

static void hb_vector_draw_svg_quadratic_to (hb_draw_funcs_t *, void *dd, hb_draw_state_t *, float cx, float cy, float x, float y, void *)
{ auto *d = (hb_vector_draw_t *) dd; d->path.append_c ('Q'); d->append_xy_svg (cx, cy); d->path.append_c (' '); d->append_xy_svg (x, y); }

static void hb_vector_draw_svg_cubic_to (hb_draw_funcs_t *, void *dd, hb_draw_state_t *, float c1x, float c1y, float c2x, float c2y, float x, float y, void *)
{ auto *d = (hb_vector_draw_t *) dd; d->path.append_c ('C'); d->append_xy_svg (c1x, c1y); d->path.append_c (' '); d->append_xy_svg (c2x, c2y); d->path.append_c (' '); d->append_xy_svg (x, y); }

static void hb_vector_draw_svg_close_path (hb_draw_funcs_t *, void *dd, hb_draw_state_t *, void *)
{ ((hb_vector_draw_t *) dd)->path.append_c ('Z'); }


/* ---- PDF draw callbacks ---- */

static void hb_vector_draw_pdf_move_to (hb_draw_funcs_t *, void *dd, hb_draw_state_t *, float x, float y, void *)
{ auto *d = (hb_vector_draw_t *) dd; d->append_xy_pdf (x, y); d->path.append_str (" m\n"); }

static void hb_vector_draw_pdf_line_to (hb_draw_funcs_t *, void *dd, hb_draw_state_t *, float x, float y, void *)
{ auto *d = (hb_vector_draw_t *) dd; d->append_xy_pdf (x, y); d->path.append_str (" l\n"); }

/* No quadratic_to — null fallback auto-promotes to cubic. */

static void hb_vector_draw_pdf_cubic_to (hb_draw_funcs_t *, void *dd, hb_draw_state_t *, float c1x, float c1y, float c2x, float c2y, float x, float y, void *)
{ auto *d = (hb_vector_draw_t *) dd; d->append_xy_pdf (c1x, c1y); d->path.append_c (' '); d->append_xy_pdf (c2x, c2y); d->path.append_c (' '); d->append_xy_pdf (x, y); d->path.append_str (" c\n"); }

static void hb_vector_draw_pdf_close_path (hb_draw_funcs_t *, void *dd, hb_draw_state_t *, void *)
{ ((hb_vector_draw_t *) dd)->path.append_str ("h\n"); }


/* ---- Lazy loaders ---- */

static inline void free_static_vector_draw_svg_funcs ();
static struct hb_vector_draw_svg_funcs_lazy_loader_t
  : hb_draw_funcs_lazy_loader_t<hb_vector_draw_svg_funcs_lazy_loader_t>
{
  static hb_draw_funcs_t *create ()
  {
    hb_draw_funcs_t *funcs = hb_draw_funcs_create ();
    hb_draw_funcs_set_move_to_func (funcs, (hb_draw_move_to_func_t) hb_vector_draw_svg_move_to, nullptr, nullptr);
    hb_draw_funcs_set_line_to_func (funcs, (hb_draw_line_to_func_t) hb_vector_draw_svg_line_to, nullptr, nullptr);
    hb_draw_funcs_set_quadratic_to_func (funcs, (hb_draw_quadratic_to_func_t) hb_vector_draw_svg_quadratic_to, nullptr, nullptr);
    hb_draw_funcs_set_cubic_to_func (funcs, (hb_draw_cubic_to_func_t) hb_vector_draw_svg_cubic_to, nullptr, nullptr);
    hb_draw_funcs_set_close_path_func (funcs, (hb_draw_close_path_func_t) hb_vector_draw_svg_close_path, nullptr, nullptr);
    hb_draw_funcs_make_immutable (funcs);
    hb_atexit (free_static_vector_draw_svg_funcs);
    return funcs;
  }
} static_vector_draw_svg_funcs;
static inline void free_static_vector_draw_svg_funcs () { static_vector_draw_svg_funcs.free_instance (); }

static inline void free_static_vector_draw_pdf_funcs ();
static struct hb_vector_draw_pdf_funcs_lazy_loader_t
  : hb_draw_funcs_lazy_loader_t<hb_vector_draw_pdf_funcs_lazy_loader_t>
{
  static hb_draw_funcs_t *create ()
  {
    hb_draw_funcs_t *funcs = hb_draw_funcs_create ();
    hb_draw_funcs_set_move_to_func (funcs, (hb_draw_move_to_func_t) hb_vector_draw_pdf_move_to, nullptr, nullptr);
    hb_draw_funcs_set_line_to_func (funcs, (hb_draw_line_to_func_t) hb_vector_draw_pdf_line_to, nullptr, nullptr);
    /* No quadratic_to: null fallback auto-promotes to cubic. */
    hb_draw_funcs_set_cubic_to_func (funcs, (hb_draw_cubic_to_func_t) hb_vector_draw_pdf_cubic_to, nullptr, nullptr);
    hb_draw_funcs_set_close_path_func (funcs, (hb_draw_close_path_func_t) hb_vector_draw_pdf_close_path, nullptr, nullptr);
    hb_draw_funcs_make_immutable (funcs);
    hb_atexit (free_static_vector_draw_pdf_funcs);
    return funcs;
  }
} static_vector_draw_pdf_funcs;
static inline void free_static_vector_draw_pdf_funcs () { static_vector_draw_pdf_funcs.free_instance (); }



/**
 * hb_vector_draw_create_or_fail:
 * @format: output format.
 *
 * Creates a new draw context for vector output.
 *
 * Return value: (nullable): a newly allocated #hb_vector_draw_t, or `NULL` on failure.
 *
 * Since: 13.0.0
 */
hb_vector_draw_t *
hb_vector_draw_create_or_fail (hb_vector_format_t format)
{
  switch (format)
  {
    case HB_VECTOR_FORMAT_SVG:
    case HB_VECTOR_FORMAT_PDF:
      break;
    case HB_VECTOR_FORMAT_INVALID: default:
      return nullptr;
  }

  hb_vector_draw_t *draw = hb_object_create<hb_vector_draw_t> ();
  if (unlikely (!draw))
    return nullptr;
  draw->format = format;
  draw->defs.alloc (2048);
  draw->body.alloc (8192);
  draw->path.alloc (2048);
  return draw;
}

/**
 * hb_vector_draw_reference:
 * @draw: a draw context.
 *
 * Increases the reference count of @draw.
 *
 * Return value: (transfer full): referenced @draw.
 *
 * Since: 13.0.0
 */
hb_vector_draw_t *
hb_vector_draw_reference (hb_vector_draw_t *draw)
{
  return hb_object_reference (draw);
}

/**
 * hb_vector_draw_destroy:
 * @draw: a draw context.
 *
 * Decreases the reference count of @draw and destroys it when it reaches zero.
 *
 * Since: 13.0.0
 */
void
hb_vector_draw_destroy (hb_vector_draw_t *draw)
{
  if (!hb_object_should_destroy (draw))
    return;

  hb_blob_destroy (draw->recycled_blob);
  hb_object_actually_destroy (draw);
  hb_free (draw);
}

/**
 * hb_vector_draw_set_user_data:
 * @draw: a draw context.
 * @key: user-data key.
 * @data: user-data value.
 * @destroy: (nullable): destroy callback for @data.
 * @replace: whether to replace an existing value for @key.
 *
 * Attaches user data to @draw.
 *
 * Return value: `true` on success, `false` otherwise.
 *
 * Since: 13.0.0
 */
hb_bool_t
hb_vector_draw_set_user_data (hb_vector_draw_t   *draw,
                              hb_user_data_key_t *key,
                              void               *data,
                              hb_destroy_func_t   destroy,
                              hb_bool_t           replace)
{
  return hb_object_set_user_data (draw, key, data, destroy, replace);
}

/**
 * hb_vector_draw_get_user_data:
 * @draw: a draw context.
 * @key: user-data key.
 *
 * Gets previously attached user data from @draw.
 *
 * Return value: (nullable): user-data value associated with @key.
 *
 * Since: 13.0.0
 */
void *
hb_vector_draw_get_user_data (const hb_vector_draw_t   *draw,
                              hb_user_data_key_t *key)
{
  return hb_object_get_user_data (draw, key);
}

/**
 * hb_vector_draw_set_transform:
 * @draw: a draw context.
 * @xx: transform xx component.
 * @yx: transform yx component.
 * @xy: transform xy component.
 * @yy: transform yy component.
 * @dx: transform x translation.
 * @dy: transform y translation.
 *
 * Sets the affine transform used when drawing glyphs.
 *
 * Since: 13.0.0
 */
void
hb_vector_draw_set_transform (hb_vector_draw_t *draw,
                              float xx, float yx,
                              float xy, float yy,
                              float dx, float dy)
{
  draw->transform = {xx, yx, xy, yy, dx, dy};
}

/**
 * hb_vector_draw_get_transform:
 * @draw: a draw context.
 * @xx: (out) (nullable): transform xx component.
 * @yx: (out) (nullable): transform yx component.
 * @xy: (out) (nullable): transform xy component.
 * @yy: (out) (nullable): transform yy component.
 * @dx: (out) (nullable): transform x translation.
 * @dy: (out) (nullable): transform y translation.
 *
 * Gets the affine transform used when drawing glyphs.
 *
 * Since: 13.0.0
 */
void
hb_vector_draw_get_transform (const hb_vector_draw_t *draw,
                              float *xx, float *yx,
                              float *xy, float *yy,
                              float *dx, float *dy)
{
  if (xx) *xx = draw->transform.xx;
  if (yx) *yx = draw->transform.yx;
  if (xy) *xy = draw->transform.xy;
  if (yy) *yy = draw->transform.yy;
  if (dx) *dx = draw->transform.x0;
  if (dy) *dy = draw->transform.y0;
}

/**
 * hb_vector_draw_set_scale_factor:
 * @draw: a draw context.
 * @x_scale_factor: x scale factor.
 * @y_scale_factor: y scale factor.
 *
 * Sets additional output scaling factors.
 *
 * Since: 13.0.0
 */
void
hb_vector_draw_set_scale_factor (hb_vector_draw_t *draw,
                                 float x_scale_factor,
                                 float y_scale_factor)
{
  draw->x_scale_factor = x_scale_factor > 0.f ? x_scale_factor : 1.f;
  draw->y_scale_factor = y_scale_factor > 0.f ? y_scale_factor : 1.f;
}

/**
 * hb_vector_draw_get_scale_factor:
 * @draw: a draw context.
 * @x_scale_factor: (out) (nullable): x scale factor.
 * @y_scale_factor: (out) (nullable): y scale factor.
 *
 * Gets additional output scaling factors.
 *
 * Since: 13.0.0
 */
void
hb_vector_draw_get_scale_factor (const hb_vector_draw_t *draw,
                                 float *x_scale_factor,
                                 float *y_scale_factor)
{
  if (x_scale_factor) *x_scale_factor = draw->x_scale_factor;
  if (y_scale_factor) *y_scale_factor = draw->y_scale_factor;
}

/**
 * hb_vector_draw_set_extents:
 * @draw: a draw context.
 * @extents: (nullable): output extents to set or expand.
 *
 * Sets or expands output extents on @draw. Passing `NULL` clears extents.
 *
 * Since: 13.0.0
 */
void
hb_vector_draw_set_extents (hb_vector_draw_t *draw,
                            const hb_vector_extents_t *extents)
{
  if (!extents)
  {
    draw->extents = {0, 0, 0, 0};
    draw->has_extents = false;
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
  float x0 = extents->x / draw->x_scale_factor;
  float y0 = extents->y / draw->y_scale_factor;
  float x1 = x0 + extents->width  / draw->x_scale_factor;
  float y1 = y0 + extents->height / draw->y_scale_factor;
  hb_vector_extents_t e = {
    hb_min (x0, x1), hb_min (y0, y1),
    fabsf (x1 - x0), fabsf (y1 - y0),
  };

  if (draw->has_extents)
  {
    float x0 = hb_min (draw->extents.x, e.x);
    float y0 = hb_min (draw->extents.y, e.y);
    float x1 = hb_max (draw->extents.x + draw->extents.width,
                       e.x + e.width);
    float y1 = hb_max (draw->extents.y + draw->extents.height,
                       e.y + e.height);
    draw->extents = {x0, y0, x1 - x0, y1 - y0};
  }
  else
  {
    draw->extents = e;
    draw->has_extents = true;
  }
}

/**
 * hb_vector_draw_get_extents:
 * @draw: a draw context.
 * @extents: (out) (nullable): where to store current output extents.
 *
 * Gets current output extents from @draw.
 *
 * Return value: `true` if extents are set, `false` otherwise.
 *
 * Since: 13.0.0
 */
hb_bool_t
hb_vector_draw_get_extents (const hb_vector_draw_t *draw,
                            hb_vector_extents_t *extents)
{
  if (!draw->has_extents)
    return false;

  if (extents)
    *extents = draw->extents;
  return true;
}

/**
 * hb_vector_draw_set_glyph_extents:
 * @draw: a draw context.
 * @glyph_extents: glyph extents in font units.
 *
 * Expands @draw extents using @glyph_extents under the current transform.
 *
 * Return value: `true` on success, `false` otherwise.
 *
 * Since: 13.0.0
 */
hb_bool_t
hb_vector_draw_set_glyph_extents (hb_vector_draw_t *draw,
                                  const hb_glyph_extents_t *glyph_extents)
{
  hb_bool_t has_extents = draw->has_extents;
  hb_bool_t ret = hb_vector_set_glyph_extents_common (draw->transform,
						   draw->x_scale_factor,
						   draw->y_scale_factor,
						   glyph_extents,
						   &draw->extents,
						   &has_extents);
  draw->has_extents = has_extents;
  return ret;
}

/**
 * hb_vector_draw_get_format:
 * @draw: a vector draw context.
 *
 * Gets the output format @draw was created with.
 *
 * Return value: the output format.
 *
 * Since: 14.2.0
 */
hb_vector_format_t
hb_vector_draw_get_format (const hb_vector_draw_t *draw)
{
  return draw->format;
}

/**
 * hb_vector_draw_get_funcs:
 * @draw: a vector draw context.
 *
 * Gets draw callbacks for feeding outline data into @draw.
 * Pass @draw as the @draw_data argument when calling them.
 *
 * Return value: (transfer none): immutable #hb_draw_funcs_t.
 *
 * Since: 14.2.0
 */
hb_draw_funcs_t *
hb_vector_draw_get_funcs (const hb_vector_draw_t *draw)
{
  switch (draw ? draw->format : HB_VECTOR_FORMAT_INVALID)
  {
    case HB_VECTOR_FORMAT_SVG: return static_vector_draw_svg_funcs.get_unconst ();
    case HB_VECTOR_FORMAT_PDF: return static_vector_draw_pdf_funcs.get_unconst ();
    case HB_VECTOR_FORMAT_INVALID: default: return nullptr;
  }
}

/**
 * hb_vector_draw_new_path:
 * @draw: a draw context.
 *
 * Flushes any pending path and starts a new one.  Call this
 * between glyphs to separate their outlines so fill rules
 * don't interact across glyphs.
 *
 * Since: 14.2.0
 */
void
hb_vector_draw_new_path (hb_vector_draw_t *draw)
{
  draw->new_path ();
}

/**
 * hb_vector_draw_glyph_or_fail:
 * @draw: a draw context.
 * @font: font object.
 * @glyph: glyph ID.
 * @extents_mode: extents update mode.
 *
 * Convenience to draw one glyph into @draw.  Equivalent to:
 *
 * |[<!-- language="plain" -->
 * // extend extents if requested
 * hb_vector_draw_new_path (draw);
 * hb_font_draw_glyph_or_fail (font, glyph,
 *   hb_vector_draw_get_funcs (draw), draw);
 * ]|
 *
 * Return value: `true` if glyph data was emitted, `false` otherwise.
 *
 * Since: 14.2.0
 */
hb_bool_t
hb_vector_draw_glyph_or_fail (hb_vector_draw_t *draw,
                      hb_font_t *font,
                      hb_codepoint_t glyph,
                      hb_vector_extents_mode_t extents_mode)
{

  if (extents_mode == HB_VECTOR_EXTENTS_MODE_EXPAND)
  {
    hb_glyph_extents_t ge;
    if (hb_font_get_glyph_extents (font, glyph, &ge))
    {
      hb_bool_t has_extents = draw->has_extents;
      hb_vector_set_glyph_extents_common (draw->transform,
					  draw->x_scale_factor,
					  draw->y_scale_factor,
					  &ge,
					  &draw->extents,
					  &has_extents);
      draw->has_extents = has_extents;
    }
  }

  draw->new_path ();
  return hb_font_draw_glyph_or_fail (font, glyph, hb_vector_draw_get_funcs (draw), draw);
}

/**
 * hb_vector_draw_glyph:
 * @draw: a draw context.
 * @font: font object.
 * @glyph: glyph ID.
 * @extents_mode: extents update mode.
 *
 * Draws one glyph into @draw.  Equivalent to
 * hb_vector_draw_glyph_or_fail() with the return value ignored.
 *
 * Since: 14.2.0
 */
void
hb_vector_draw_glyph (hb_vector_draw_t *draw,
                      hb_font_t *font,
                      hb_codepoint_t glyph,
                      hb_vector_extents_mode_t extents_mode)
{
  hb_vector_draw_glyph_or_fail (draw, font, glyph, extents_mode);
}


/**
 * hb_vector_draw_set_precision:
 * @draw: a draw context.
 * @precision: decimal precision.
 *
 * Sets numeric output precision for draw output.
 *
 * Since: 14.2.0
 */
void
hb_vector_draw_set_precision (hb_vector_draw_t *draw,
                             unsigned precision)
{
  draw->set_precision (precision);
}

/**
 * hb_vector_draw_get_precision:
 * @draw: a draw context.
 *
 * Returns the numeric output precision previously set on @draw,
 * or the default if none was set.
 *
 * Return value: the precision.
 *
 * Since: 14.2.0
 */
unsigned
hb_vector_draw_get_precision (const hb_vector_draw_t *draw)
{
  return draw->get_precision ();
}

/**
 * hb_vector_draw_set_foreground:
 * @draw: a draw context.
 * @foreground: foreground fill color.
 *
 * Sets the fill color for drawn glyph outlines.
 * Default is opaque black.
 *
 * Since: 14.2.0
 */
void
hb_vector_draw_set_foreground (hb_vector_draw_t *draw,
                               hb_color_t foreground)
{
  draw->flush_path ();
  draw->foreground = foreground;
}

/**
 * hb_vector_draw_get_foreground:
 * @draw: a draw context.
 *
 * Returns the foreground fill color.
 *
 * Return value: the foreground color.
 *
 * Since: 14.2.0
 */
hb_color_t
hb_vector_draw_get_foreground (const hb_vector_draw_t *draw)
{
  return draw->foreground;
}

/**
 * hb_vector_draw_set_background:
 * @draw: a draw context.
 * @background: background color.
 *
 * Sets the background color.  If non-transparent, a filled
 * rectangle covering the extents is emitted behind all content.
 * Default is transparent (no background).
 *
 * Since: 14.2.0
 */
void
hb_vector_draw_set_background (hb_vector_draw_t *draw,
                               hb_color_t background)
{
  draw->background = background;
}

/**
 * hb_vector_draw_get_background:
 * @draw: a draw context.
 *
 * Returns the background color.
 *
 * Return value: the background color.
 *
 * Since: 14.2.0
 */
hb_color_t
hb_vector_draw_get_background (const hb_vector_draw_t *draw)
{
  return draw->background;
}

static hb_blob_t *
hb_vector_draw_render_pdf (hb_vector_draw_t *draw)
{
  draw->flush_path ();
  if (!draw->has_extents)
    return nullptr;

  /* Collect the content stream.  The path coordinates are in
   * SVG space (Y-down).  Prepend a CTM that flips Y so the
   * PDF page (Y-up) renders correctly. */
  float ex = draw->extents.x;
  float ey = draw->extents.y;
  float ew = draw->extents.width;
  float eh = draw->extents.height;

  hb_vector_buf_t stream;
  stream.alloc (draw->body.length + draw->path.length + 256);

  /* Background rect. */
  if (hb_color_get_alpha (draw->background))
  {
    float a = hb_color_get_alpha (draw->background) / 255.f;
    if (a < 1.f - 1.f / 512.f)
    {
      stream.append_num (a, 4);
      stream.append_str (" ca gs\n");
    }
    stream.append_num (hb_color_get_red (draw->background) / 255.f, 4);
    stream.append_c (' ');
    stream.append_num (hb_color_get_green (draw->background) / 255.f, 4);
    stream.append_c (' ');
    stream.append_num (hb_color_get_blue (draw->background) / 255.f, 4);
    stream.append_str (" rg\n");
    stream.append_num (ex);
    stream.append_c (' ');
    stream.append_num (-(ey + eh));
    stream.append_c (' ');
    stream.append_num (ew);
    stream.append_c (' ');
    stream.append_num (eh);
    stream.append_str (" re f\n");
  }

  draw->flush_path ();
  if (draw->body.length)
    stream.append_len (draw->body.arrayZ, draw->body.length);

  /* Build PDF objects, tracking byte offsets for xref. */
  hb_vector_buf_t out;
  hb_buf_recover_recycled (draw->recycled_blob, &out);
  out.alloc (stream.length + 512);

  unsigned offsets[5]; /* objects 1-4, plus end */

  out.append_str ("%PDF-1.4\n%\xC0\xC1\xC2\xC3\n");

  /* Object 1: Catalog */
  offsets[0] = out.length;
  out.append_str ("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");

  /* Object 2: Pages */
  offsets[1] = out.length;
  out.append_str ("2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");

  /* Object 3: Page.  Extents are in SVG space (y = -font_y).
   * Convert back: font Y range = [-(ey+eh) .. -ey]. */
  offsets[2] = out.length;
  out.append_str ("3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [");
  out.append_num (ex);
  out.append_c (' ');
  out.append_num (-(ey + eh));
  out.append_c (' ');
  out.append_num (ex + ew);
  out.append_c (' ');
  out.append_num (-ey);
  out.append_str ("] /Contents 4 0 R");
  if (draw->pdf_extgstate_dict.length)
  {
    out.append_str (" /Resources << /ExtGState << ");
    out.append_len (draw->pdf_extgstate_dict.arrayZ, draw->pdf_extgstate_dict.length);
    out.append_str (">> >>");
  }
  out.append_str (" >>\nendobj\n");

  /* Object 4: Content stream */
  offsets[3] = out.length;
  out.append_str ("4 0 obj\n<< /Length ");
  out.append_unsigned (stream.length);
  out.append_str (" >>\nstream\n");
  out.append_len (stream.arrayZ, stream.length);
  out.append_str ("endstream\nendobj\n");

  /* Cross-reference table */
  unsigned xref_offset = out.length;
  out.append_str ("xref\n0 5\n");
  out.append_str ("0000000000 65535 f \n");
  for (unsigned i = 0; i < 4; i++)
  {
    char tmp[21];
    snprintf (tmp, sizeof (tmp), "%010u 00000 n \n", offsets[i]);
    out.append_len (tmp, 20);
  }

  /* Trailer */
  out.append_str ("trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n");
  out.append_unsigned (xref_offset);
  out.append_str ("\n%%EOF\n");

  hb_blob_t *blob = hb_buf_blob_from (&draw->recycled_blob, &out);

  hb_vector_draw_clear (draw);

  return blob;
}

static hb_blob_t *
hb_vector_draw_render_svg (hb_vector_draw_t *draw)
{
  draw->flush_path ();
  if (!draw->has_extents)
    return nullptr;

  hb_vector_buf_t out;
  hb_buf_recover_recycled (draw->recycled_blob, &out);
  unsigned estimated = draw->defs.length +
		       (draw->body.length ? draw->body.length : draw->path.length) +
		       256;
  out.alloc (estimated);
  /* Extents are in Y-up (font) space.  SVG is Y-down.
   * The global scale(1,-1) wrapper flips content; the
   * viewBox maps the flipped range. */
  float vb_x = draw->extents.x;
  float vb_y = -(draw->extents.y + draw->extents.height);
  float vb_w = draw->extents.width;
  float vb_h = draw->extents.height;
  out.append_str ("<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"");
  out.append_num (vb_x);
  out.append_c (' ');
  out.append_num (vb_y);
  out.append_c (' ');
  out.append_num (vb_w);
  out.append_c (' ');
  out.append_num (vb_h);
  out.append_str ("\" width=\"");
  out.append_num (vb_w);
  out.append_str ("\" height=\"");
  out.append_num (vb_h);
  out.append_str ("\">\n");

  if (draw->defs.length)
  {
    out.append_str ("<defs>\n");
    out.append_len (draw->defs.arrayZ, draw->defs.length);
    out.append_str ("</defs>\n");
  }

  /* Background rect. */
  if (hb_color_get_alpha (draw->background))
  {
    out.append_str ("<rect x=\"");
    out.append_num (vb_x);
    out.append_str ("\" y=\"");
    out.append_num (vb_y);
    out.append_str ("\" width=\"");
    out.append_num (vb_w);
    out.append_str ("\" height=\"");
    out.append_num (vb_h);
    out.append_str ("\" fill=\"rgb(");
    out.append_unsigned (hb_color_get_red (draw->background));
    out.append_c (',');
    out.append_unsigned (hb_color_get_green (draw->background));
    out.append_c (',');
    out.append_unsigned (hb_color_get_blue (draw->background));
    out.append_str (")\"");
    if (hb_color_get_alpha (draw->background) < 255)
    {
      out.append_str (" fill-opacity=\"");
      out.append_num (hb_color_get_alpha (draw->background) / 255.f, 4);
      out.append_c ('"');
    }
    out.append_str ("/>\n");
  }

  draw->flush_path ();
  if (draw->body.length)
  {
    out.append_str ("<g transform=\"scale(1,-1)\">\n");
    out.append_len (draw->body.arrayZ, draw->body.length);
    out.append_str ("</g>\n");
  }

  out.append_str ("</svg>\n");

  hb_blob_t *blob = hb_buf_blob_from (&draw->recycled_blob, &out);

  hb_vector_draw_clear (draw);

  return blob;
}

/**
 * hb_vector_draw_render:
 * @draw: a draw context.
 *
 * Renders accumulated draw content to an output blob.
 *
 * Return value: (transfer full) (nullable): output blob, or `NULL` if rendering cannot proceed.
 *
 * Since: 13.0.0
 */
hb_blob_t *
hb_vector_draw_render (hb_vector_draw_t *draw)
{
  switch (draw->format)
  {
    case HB_VECTOR_FORMAT_SVG:
      return hb_vector_draw_render_svg (draw);

    case HB_VECTOR_FORMAT_PDF:
      return hb_vector_draw_render_pdf (draw);

    case HB_VECTOR_FORMAT_INVALID: default:
      return nullptr;
  }
}

/**
 * hb_vector_draw_clear:
 * @draw: a draw context.
 *
 * Discards accumulated draw output so @draw can be reused for
 * another render.  User configuration (transform, scale factors,
 * precision) is preserved.  Call hb_vector_draw_reset() to
 * also reset user configuration to defaults.
 *
 * Since: 14.2.0
 */
void
hb_vector_draw_clear (hb_vector_draw_t *draw)
{
  draw->extents = {0, 0, 0, 0};
  draw->has_extents = false;
  draw->defs.clear ();
  draw->body.clear ();
  draw->path.clear ();
}

/**
 * hb_vector_draw_reset:
 * @draw: a draw context.
 *
 * Resets @draw state and clears accumulated content.
 *
 * Since: 13.0.0
 */
void
hb_vector_draw_reset (hb_vector_draw_t *draw)
{
  draw->transform = {1, 0, 0, 1, 0, 0};
  draw->x_scale_factor = 1.f;
  draw->y_scale_factor = 1.f;
  draw->set_precision (2);
  hb_vector_draw_clear (draw);
}

/**
 * hb_vector_draw_recycle_blob:
 * @draw: a draw context.
 * @blob: (nullable): previously rendered blob to recycle.
 *
 * Provides a blob for internal buffer reuse by later render calls.
 *
 * Since: 13.0.0
 */
void
hb_vector_draw_recycle_blob (hb_vector_draw_t *draw,
                             hb_blob_t *blob)
{
  hb_blob_destroy (draw->recycled_blob);
  draw->recycled_blob = nullptr;
  if (!blob || blob == hb_blob_get_empty ())
    return;
  draw->recycled_blob = blob;
}
