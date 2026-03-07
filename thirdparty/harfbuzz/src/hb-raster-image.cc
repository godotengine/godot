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

#include "hb-raster-image.hh"

#include <math.h>


/*
 * Image compositing
 */

/* Unpack premultiplied pixel to float RGBA [0,1]. */
static inline void
unpack_to_float (uint32_t px, float &r, float &g, float &b, float &a)
{
  b = (px & 0xFF) / 255.f;
  g = ((px >> 8) & 0xFF) / 255.f;
  r = ((px >> 16) & 0xFF) / 255.f;
  a = (px >> 24) / 255.f;
}

/* Pack float RGBA [0,1] premultiplied back to uint32_t. */
static inline uint32_t
pack_from_float (float r, float g, float b, float a)
{
  return hb_raster_pack_pixel ((uint8_t) (hb_clamp (b, 0.f, 1.f) * 255.f + 0.5f),
			       (uint8_t) (hb_clamp (g, 0.f, 1.f) * 255.f + 0.5f),
			       (uint8_t) (hb_clamp (r, 0.f, 1.f) * 255.f + 0.5f),
			       (uint8_t) (hb_clamp (a, 0.f, 1.f) * 255.f + 0.5f));
}

/* Separable blend mode functions: operate on unpremultiplied [0,1] channels. */
static inline float
blend_multiply (float sc, float dc) { return sc * dc; }
static inline float
blend_screen (float sc, float dc) { return sc + dc - sc * dc; }
static inline float
blend_overlay (float sc, float dc)
{ return dc <= 0.5f ? 2.f * sc * dc : 1.f - 2.f * (1.f - sc) * (1.f - dc); }
static inline float
blend_darken (float sc, float dc) { return hb_min (sc, dc); }
static inline float
blend_lighten (float sc, float dc) { return hb_max (sc, dc); }
static inline float
blend_color_dodge (float sc, float dc)
{
  if (dc <= 0.f) return 0.f;
  if (sc >= 1.f) return 1.f;
  return hb_min (1.f, dc / (1.f - sc));
}
static inline float
blend_color_burn (float sc, float dc)
{
  if (dc >= 1.f) return 1.f;
  if (sc <= 0.f) return 0.f;
  return 1.f - hb_min (1.f, (1.f - dc) / sc);
}
static inline float
blend_hard_light (float sc, float dc)
{ return sc <= 0.5f ? 2.f * sc * dc : 1.f - 2.f * (1.f - sc) * (1.f - dc); }
static inline float
blend_soft_light (float sc, float dc)
{
  if (sc <= 0.5f)
    return dc - (1.f - 2.f * sc) * dc * (1.f - dc);
  float d = (dc <= 0.25f) ? ((16.f * dc - 12.f) * dc + 4.f) * dc
			   : sqrtf (dc);
  return dc + (2.f * sc - 1.f) * (d - dc);
}
static inline float
blend_difference (float sc, float dc) { return fabsf (sc - dc); }
static inline float
blend_exclusion (float sc, float dc) { return sc + dc - 2.f * sc * dc; }

/* Apply a separable blend mode per-pixel.
 * Both src and dst are premultiplied BGRA32. */
static inline uint32_t
apply_separable_blend (uint32_t src, uint32_t dst,
		       float (*blend_fn)(float, float))
{
  float sr, sg, sb, sa;
  float dr, dg, db, da;
  unpack_to_float (src, sr, sg, sb, sa);
  unpack_to_float (dst, dr, dg, db, da);

  float usr = sa > 0.f ? sr / sa : 0.f;
  float usg = sa > 0.f ? sg / sa : 0.f;
  float usb = sa > 0.f ? sb / sa : 0.f;
  float udr = da > 0.f ? dr / da : 0.f;
  float udg = da > 0.f ? dg / da : 0.f;
  float udb = da > 0.f ? db / da : 0.f;

  float br = blend_fn (usr, udr);
  float bg = blend_fn (usg, udg);
  float bb = blend_fn (usb, udb);

  float ra = sa + da - sa * da;
  float rr = sa * da * br + sa * (1.f - da) * usr + (1.f - sa) * da * udr;
  float rg = sa * da * bg + sa * (1.f - da) * usg + (1.f - sa) * da * udg;
  float rb = sa * da * bb + sa * (1.f - da) * usb + (1.f - sa) * da * udb;

  return pack_from_float (rr, rg, rb, ra);
}

/* HSL helpers */
static inline float
hsl_luminosity (float r, float g, float b)
{ return 0.299f * r + 0.587f * g + 0.114f * b; }

static inline float
hsl_saturation (float r, float g, float b)
{ return hb_max (hb_max (r, g), b) - hb_min (hb_min (r, g), b); }

static inline void
hsl_clip_color (float &r, float &g, float &b)
{
  float l = hsl_luminosity (r, g, b);
  float mn = hb_min (hb_min (r, g), b);
  float mx = hb_max (hb_max (r, g), b);
  if (mn < 0.f)
  {
    float d = l - mn;
    if (d > 0.f) { r = l + (r - l) * l / d; g = l + (g - l) * l / d; b = l + (b - l) * l / d; }
  }
  if (mx > 1.f)
  {
    float d = mx - l;
    if (d > 0.f) { r = l + (r - l) * (1.f - l) / d; g = l + (g - l) * (1.f - l) / d; b = l + (b - l) * (1.f - l) / d; }
  }
}

static inline void
hsl_set_luminosity (float &r, float &g, float &b, float l)
{
  float d = l - hsl_luminosity (r, g, b);
  r += d; g += d; b += d;
  hsl_clip_color (r, g, b);
}

static inline void
hsl_set_saturation_inner (float &mn, float &mid, float &mx, float s)
{
  if (mx > mn)
  {
    mid = (mid - mn) * s / (mx - mn);
    mx = s;
  }
  else
    mid = mx = 0.f;
  mn = 0.f;
}

static inline void
hsl_set_saturation (float &r, float &g, float &b, float s)
{
  if (r <= g)
  {
    if (g <= b)      hsl_set_saturation_inner (r, g, b, s);
    else if (r <= b) hsl_set_saturation_inner (r, b, g, s);
    else             hsl_set_saturation_inner (b, r, g, s);
  }
  else
  {
    if (r <= b)      hsl_set_saturation_inner (g, r, b, s);
    else if (g <= b) hsl_set_saturation_inner (g, b, r, s);
    else             hsl_set_saturation_inner (b, g, r, s);
  }
}

static inline uint32_t
apply_hsl_blend (uint32_t src, uint32_t dst,
		 hb_paint_composite_mode_t mode)
{
  float sr, sg, sb, sa;
  float dr, dg, db, da;
  unpack_to_float (src, sr, sg, sb, sa);
  unpack_to_float (dst, dr, dg, db, da);

  float usr = sa > 0.f ? sr / sa : 0.f;
  float usg = sa > 0.f ? sg / sa : 0.f;
  float usb = sa > 0.f ? sb / sa : 0.f;
  float udr = da > 0.f ? dr / da : 0.f;
  float udg = da > 0.f ? dg / da : 0.f;
  float udb = da > 0.f ? db / da : 0.f;

  float br = udr, bg = udg, bb = udb;

  if (mode == HB_PAINT_COMPOSITE_MODE_HSL_HUE)
  {
    br = usr; bg = usg; bb = usb;
    hsl_set_saturation (br, bg, bb, hsl_saturation (udr, udg, udb));
    hsl_set_luminosity (br, bg, bb, hsl_luminosity (udr, udg, udb));
  }
  else if (mode == HB_PAINT_COMPOSITE_MODE_HSL_SATURATION)
  {
    br = udr; bg = udg; bb = udb;
    hsl_set_saturation (br, bg, bb, hsl_saturation (usr, usg, usb));
    hsl_set_luminosity (br, bg, bb, hsl_luminosity (udr, udg, udb));
  }
  else if (mode == HB_PAINT_COMPOSITE_MODE_HSL_COLOR)
  {
    br = usr; bg = usg; bb = usb;
    hsl_set_luminosity (br, bg, bb, hsl_luminosity (udr, udg, udb));
  }
  else /* HSL_LUMINOSITY */
  {
    br = udr; bg = udg; bb = udb;
    hsl_set_luminosity (br, bg, bb, hsl_luminosity (usr, usg, usb));
  }

  float ra = sa + da - sa * da;
  float rr = sa * da * br + sa * (1.f - da) * usr + (1.f - sa) * da * udr;
  float rg = sa * da * bg + sa * (1.f - da) * usg + (1.f - sa) * da * udg;
  float rb = sa * da * bb + sa * (1.f - da) * usb + (1.f - sa) * da * udb;

  return pack_from_float (rr, rg, rb, ra);
}

/* Composite per-pixel with full blend mode support. */
static inline uint32_t
composite_pixel (uint32_t src, uint32_t dst,
		 hb_paint_composite_mode_t mode)
{
  uint8_t sa = (uint8_t) (src >> 24);
  uint8_t da = (uint8_t) (dst >> 24);

  switch (mode)
  {
  case HB_PAINT_COMPOSITE_MODE_CLEAR:
    return 0;
  case HB_PAINT_COMPOSITE_MODE_SRC:
    return src;
  case HB_PAINT_COMPOSITE_MODE_DEST:
    return dst;
  case HB_PAINT_COMPOSITE_MODE_SRC_OVER:
    return hb_raster_src_over (src, dst);
  case HB_PAINT_COMPOSITE_MODE_DEST_OVER:
    return hb_raster_src_over (dst, src);
  case HB_PAINT_COMPOSITE_MODE_SRC_IN:
    return hb_raster_alpha_mul (src, da);
  case HB_PAINT_COMPOSITE_MODE_DEST_IN:
    return hb_raster_alpha_mul (dst, sa);
  case HB_PAINT_COMPOSITE_MODE_SRC_OUT:
    return hb_raster_alpha_mul (src, 255 - da);
  case HB_PAINT_COMPOSITE_MODE_DEST_OUT:
    return hb_raster_alpha_mul (dst, 255 - sa);
  case HB_PAINT_COMPOSITE_MODE_SRC_ATOP:
  {
    /* Fa=Da, Fb=1-Sa */
    uint32_t a = hb_raster_alpha_mul (src, da);
    uint32_t b = hb_raster_alpha_mul (dst, 255 - sa);
    uint8_t rb = (uint8_t) hb_min (255u, (unsigned) (a & 0xFF) + (b & 0xFF));
    uint8_t rg = (uint8_t) hb_min (255u, (unsigned) ((a >> 8) & 0xFF) + ((b >> 8) & 0xFF));
    uint8_t rr = (uint8_t) hb_min (255u, (unsigned) ((a >> 16) & 0xFF) + ((b >> 16) & 0xFF));
    uint8_t ra = (uint8_t) hb_min (255u, (unsigned) (a >> 24) + (b >> 24));
    return hb_raster_pack_pixel (rb, rg, rr, ra);
  }
  case HB_PAINT_COMPOSITE_MODE_DEST_ATOP:
  {
    uint32_t a = hb_raster_alpha_mul (dst, sa);
    uint32_t b = hb_raster_alpha_mul (src, 255 - da);
    uint8_t rb = (uint8_t) hb_min (255u, (unsigned) (a & 0xFF) + (b & 0xFF));
    uint8_t rg = (uint8_t) hb_min (255u, (unsigned) ((a >> 8) & 0xFF) + ((b >> 8) & 0xFF));
    uint8_t rr = (uint8_t) hb_min (255u, (unsigned) ((a >> 16) & 0xFF) + ((b >> 16) & 0xFF));
    uint8_t ra = (uint8_t) hb_min (255u, (unsigned) (a >> 24) + (b >> 24));
    return hb_raster_pack_pixel (rb, rg, rr, ra);
  }
  case HB_PAINT_COMPOSITE_MODE_XOR:
  {
    uint32_t a = hb_raster_alpha_mul (src, 255 - da);
    uint32_t b = hb_raster_alpha_mul (dst, 255 - sa);
    uint8_t rb = (uint8_t) hb_min (255u, (unsigned) (a & 0xFF) + (b & 0xFF));
    uint8_t rg = (uint8_t) hb_min (255u, (unsigned) ((a >> 8) & 0xFF) + ((b >> 8) & 0xFF));
    uint8_t rr = (uint8_t) hb_min (255u, (unsigned) ((a >> 16) & 0xFF) + ((b >> 16) & 0xFF));
    uint8_t ra = (uint8_t) hb_min (255u, (unsigned) (a >> 24) + (b >> 24));
    return hb_raster_pack_pixel (rb, rg, rr, ra);
  }
  case HB_PAINT_COMPOSITE_MODE_PLUS:
  {
    uint8_t rb = (uint8_t) hb_min (255u, (unsigned) (src & 0xFF) + (dst & 0xFF));
    uint8_t rg = (uint8_t) hb_min (255u, (unsigned) ((src >> 8) & 0xFF) + ((dst >> 8) & 0xFF));
    uint8_t rr = (uint8_t) hb_min (255u, (unsigned) ((src >> 16) & 0xFF) + ((dst >> 16) & 0xFF));
    uint8_t ra = (uint8_t) hb_min (255u, (unsigned) (src >> 24) + (dst >> 24));
    return hb_raster_pack_pixel (rb, rg, rr, ra);
  }

  case HB_PAINT_COMPOSITE_MODE_MULTIPLY:  return apply_separable_blend (src, dst, blend_multiply);
  case HB_PAINT_COMPOSITE_MODE_SCREEN:    return apply_separable_blend (src, dst, blend_screen);
  case HB_PAINT_COMPOSITE_MODE_OVERLAY:   return apply_separable_blend (src, dst, blend_overlay);
  case HB_PAINT_COMPOSITE_MODE_DARKEN:    return apply_separable_blend (src, dst, blend_darken);
  case HB_PAINT_COMPOSITE_MODE_LIGHTEN:   return apply_separable_blend (src, dst, blend_lighten);
  case HB_PAINT_COMPOSITE_MODE_COLOR_DODGE: return apply_separable_blend (src, dst, blend_color_dodge);
  case HB_PAINT_COMPOSITE_MODE_COLOR_BURN:  return apply_separable_blend (src, dst, blend_color_burn);
  case HB_PAINT_COMPOSITE_MODE_HARD_LIGHT:  return apply_separable_blend (src, dst, blend_hard_light);
  case HB_PAINT_COMPOSITE_MODE_SOFT_LIGHT:  return apply_separable_blend (src, dst, blend_soft_light);
  case HB_PAINT_COMPOSITE_MODE_DIFFERENCE:  return apply_separable_blend (src, dst, blend_difference);
  case HB_PAINT_COMPOSITE_MODE_EXCLUSION:   return apply_separable_blend (src, dst, blend_exclusion);

  case HB_PAINT_COMPOSITE_MODE_HSL_HUE:
  case HB_PAINT_COMPOSITE_MODE_HSL_SATURATION:
  case HB_PAINT_COMPOSITE_MODE_HSL_COLOR:
  case HB_PAINT_COMPOSITE_MODE_HSL_LUMINOSITY:
    return apply_hsl_blend (src, dst, mode);

  default:
    return hb_raster_src_over (src, dst);
  }
}

/* hb_raster_image_t */

#define HB_RASTER_IMAGE_MAX_BUFFER_SIZE ((size_t) 1 << 30)

unsigned
hb_raster_image_t::bytes_per_pixel (hb_raster_format_t format)
{
  return format == HB_RASTER_FORMAT_BGRA32 ? 4u : 1u;
}

bool
hb_raster_image_t::configure (hb_raster_format_t format,
			      hb_raster_extents_t extents)
{
  if (format != HB_RASTER_FORMAT_A8 &&
      format != HB_RASTER_FORMAT_BGRA32)
    format = HB_RASTER_FORMAT_A8;

  unsigned bpp = bytes_per_pixel (format);
  if (extents.width > UINT_MAX / bpp)
    return false;

  unsigned min_stride = extents.width * bpp;
  if (extents.stride == 0 || extents.stride < min_stride)
    extents.stride = min_stride;

  if (extents.height && extents.stride > (size_t) -1 / extents.height)
    return false;

  size_t buf_size = (size_t) extents.stride * extents.height;
  if (buf_size > HB_RASTER_IMAGE_MAX_BUFFER_SIZE)
    return false;
  if (unlikely (!buffer.resize_dirty (buf_size)))
    return false;

  this->format = format;
  this->extents = extents;
  return true;
}

void
hb_raster_image_t::clear ()
{
  size_t buf_size = (size_t) extents.stride * extents.height;
  hb_memset (buffer.arrayZ, 0, buf_size);
}

const uint8_t *
hb_raster_image_t::get_buffer () const
{
  return buffer.arrayZ;
}

void
hb_raster_image_t::composite_from (const hb_raster_image_t *src,
				   hb_paint_composite_mode_t mode)
{
  unsigned w = extents.width;
  unsigned h = extents.height;
  unsigned stride = extents.stride;

  for (unsigned y = 0; y < h; y++)
  {
    uint32_t *dp = reinterpret_cast<uint32_t *> (buffer.arrayZ + y * stride);
    const uint32_t *sp = reinterpret_cast<const uint32_t *> (src->buffer.arrayZ + y * stride);
    for (unsigned x = 0; x < w; x++)
      dp[x] = composite_pixel (sp[x], dp[x], mode);
  }
}

/* Composite src image onto dst image.
 * Both images must have the same extents and BGRA32 format. */
void
hb_raster_image_composite (hb_raster_image_t *dst,
			   const hb_raster_image_t *src,
			   hb_paint_composite_mode_t mode)
{
  dst->composite_from (src, mode);
}

/**
 * hb_raster_image_create_or_fail:
 *
 * Creates a new raster image object.
 *
 * Return value: (transfer full):
 * A newly allocated #hb_raster_image_t with a reference count of 1,
 * or `NULL` on allocation failure.
 *
 * The returned image can be released with hb_raster_image_destroy(), or
 * transferred for reuse with hb_raster_draw_recycle_image() or
 * hb_raster_paint_recycle_image().
 *
 * Since: 13.0.0
 **/
hb_raster_image_t *
hb_raster_image_create_or_fail (void)
{
  return hb_object_create<hb_raster_image_t> ();
}

/**
 * hb_raster_image_reference: (skip)
 * @image: a raster image
 *
 * Increases the reference count on @image by one.
 *
 * This prevents @image from being destroyed until a matching
 * call to hb_raster_image_destroy() is made.
 *
 * Return value: (transfer full):
 * The referenced #hb_raster_image_t.
 *
 * Since: 13.0.0
 **/
hb_raster_image_t *
hb_raster_image_reference (hb_raster_image_t *image)
{
  return hb_object_reference (image);
}

/**
 * hb_raster_image_destroy: (skip)
 * @image: a raster image
 *
 * Decreases the reference count on @image by one. When the
 * reference count reaches zero, the image and its pixel buffer
 * are freed.
 *
 * Since: 13.0.0
 **/
void
hb_raster_image_destroy (hb_raster_image_t *image)
{
  if (!hb_object_destroy (image)) return;
  hb_free (image);
}

/**
 * hb_raster_image_set_user_data: (skip)
 * @image: a raster image
 * @key: the user-data key
 * @data: a pointer to the user data
 * @destroy: (nullable): a callback to call when @data is not needed anymore
 * @replace: whether to replace an existing data with the same key
 *
 * Attaches a user-data key/data pair to the specified raster image.
 *
 * Return value: `true` if success, `false` otherwise
 *
 * Since: 13.0.0
 **/
hb_bool_t
hb_raster_image_set_user_data (hb_raster_image_t  *image,
			       hb_user_data_key_t *key,
			       void               *data,
			       hb_destroy_func_t   destroy,
			       hb_bool_t           replace)
{
  return hb_object_set_user_data (image, key, data, destroy, replace);
}

/**
 * hb_raster_image_get_user_data: (skip)
 * @image: a raster image
 * @key: the user-data key
 *
 * Fetches the user-data associated with the specified key,
 * attached to the specified raster image.
 *
 * Return value: (transfer none):
 * A pointer to the user data
 *
 * Since: 13.0.0
 **/
void *
hb_raster_image_get_user_data (hb_raster_image_t  *image,
			       hb_user_data_key_t *key)
{
  return hb_object_get_user_data (image, key);
}

/**
 * hb_raster_image_configure:
 * @image: a raster image
 * @format: the pixel format
 * @extents: (nullable): desired image extents
 *
 * Configures @image format and extents together, resizing backing storage
 * at most once. This function does not clear pixel contents.
 *
 * Passing `NULL` for @extents clears extents and releases the backing
 * allocation.
 *
 * Return value: `true` if configuration succeeds, `false` on allocation
 * failure
 *
 * Since: 13.0.0
 **/
hb_bool_t
hb_raster_image_configure (hb_raster_image_t         *image,
			   hb_raster_format_t        format,
			   const hb_raster_extents_t *extents)
{
  if (unlikely (!extents))
  {
    image->extents = {};
    image->buffer.resize_exact (0);
    return true;
  }
  return image->configure (format, *extents);
}

/**
 * hb_raster_image_clear:
 * @image: a raster image
 *
 * Clears @image pixels to zero while keeping current extents and format.
 *
 * Since: 13.0.0
 **/
void
hb_raster_image_clear (hb_raster_image_t *image)
{
  image->clear ();
}

/**
 * hb_raster_image_get_buffer:
 * @image: a raster image
 *
 * Fetches the raw pixel buffer of @image.  The buffer layout is
 * described by the extents obtained from hb_raster_image_get_extents()
 * and the format from hb_raster_image_get_format().
 *
 * Return value: (transfer none) (array):
 * The pixel buffer, or `NULL`
 *
 * Since: 13.0.0
 **/
const uint8_t *
hb_raster_image_get_buffer (hb_raster_image_t *image)
{
  return image->get_buffer ();
}

/**
 * hb_raster_image_get_extents:
 * @image: a raster image
 * @extents: (out) (nullable): the image extents
 *
 * Fetches the pixel-buffer extents of @image.
 *
 * Since: 13.0.0
 **/
void
hb_raster_image_get_extents (hb_raster_image_t   *image,
			     hb_raster_extents_t *extents)
{
  if (extents)
    *extents = image->extents;
}

/**
 * hb_raster_image_get_format:
 * @image: a raster image
 *
 * Fetches the pixel format of @image.
 *
 * Return value:
 * The #hb_raster_format_t of the image
 *
 * Since: 13.0.0
 **/
hb_raster_format_t
hb_raster_image_get_format (hb_raster_image_t *image)
{
  return image->format;
}
