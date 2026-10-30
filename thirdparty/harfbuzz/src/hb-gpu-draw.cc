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

#include "hb.hh"

#include "hb-gpu.h"
#include "hb-gpu-draw.hh"
#include "hb-gpu-cu2qu.hh"
#include "hb-machinery.hh"

#include <cmath>

/* ---- Accumulator ---- */

static void
acc_update_extents (hb_gpu_draw_t *g, double x, double y)
{
  g->ext_min_x = hb_min (g->ext_min_x, x);
  g->ext_min_y = hb_min (g->ext_min_y, y);
  g->ext_max_x = hb_max (g->ext_max_x, x);
  g->ext_max_y = hb_max (g->ext_max_y, y);
}

static void
acc_emit (hb_gpu_draw_t *g,
	  bool contour_start,
	  double p1x, double p1y,
	  double p2x, double p2y,
	  double p3x, double p3y)
{
  if (unlikely (g->num_curves >= HB_GPU_DRAW_MAX_CURVES))
  {
    g->success = false;
    return;
  }

  hb_gpu_curve_t c = {p1x, p1y, p2x, p2y, p3x, p3y, contour_start};
  if (unlikely (!g->curves.push_or_fail (c)))
  {
    g->success = false;
    return;
  }

  g->num_curves++;
  g->current_x = p3x;
  g->current_y = p3y;

  acc_update_extents (g, p1x, p1y);
  acc_update_extents (g, p2x, p2y);
  acc_update_extents (g, p3x, p3y);
}

static void
acc_emit_conic (hb_gpu_draw_t *g,
		double cx, double cy,
		double x, double y)
{
  if (g->current_x == x && g->current_y == y)
    return;

  bool contour_start = g->need_moveto;
  if (g->need_moveto) {
    g->start_x = g->current_x;
    g->start_y = g->current_y;
    g->need_moveto = false;
  }

  acc_emit (g, contour_start,
	    g->current_x, g->current_y, cx, cy, x, y);
}

void
hb_gpu_draw_t::acc_move_to (double x, double y)
{
  if (unlikely (!success))
    return;

  need_moveto = true;
  current_x = x;
  current_y = y;
}

void
hb_gpu_draw_t::acc_line_to (double x, double y)
{
  if (unlikely (!success))
    return;

  acc_emit_conic (this, current_x, current_y, x, y);
}

void
hb_gpu_draw_t::acc_conic_to (double cx, double cy, double x, double y)
{
  if (unlikely (!success))
    return;

  acc_emit_conic (this, cx, cy, x, y);
}

void
hb_gpu_draw_t::acc_close_path ()
{
  if (unlikely (!success))
    return;

  if (!need_moveto &&
      (current_x != start_x || current_y != start_y))
    acc_line_to (start_x, start_y);
}


void
hb_gpu_draw_t::acc_cubic_to (double c1x, double c1y,
			       double c2x, double c2y,
			       double x, double y)
{
  if (unlikely (!success))
    return;

  double c0x = current_x, c0y = current_y;

  if (c0x == x && c0y == y &&
      c1x == x && c1y == y &&
      c2x == x && c2y == y)
    return;

  /* Degenerate cubic: all control points are collinear.
   * Emit as a line instead of running cu2qu subdivision. */
  double dx = x - c0x, dy = y - c0y;
  double len_sq = dx * dx + dy * dy;
  if (len_sq > 0.)
  {
    double inv = 1.0 / len_sq;
    double cross1 = (c1x - c0x) * dy - (c1y - c0y) * dx;
    double cross2 = (c2x - c0x) * dy - (c2y - c0y) * dx;
    if (cross1 * cross1 * inv <= HB_GPU_CU2QU_TOLERANCE * HB_GPU_CU2QU_TOLERANCE &&
	cross2 * cross2 * inv <= HB_GPU_CU2QU_TOLERANCE * HB_GPU_CU2QU_TOLERANCE)
    {
      acc_line_to (x, y);
      return;
    }
  }

  hb_gpu_cubic_to_quadratics (this,
			      {c0x, c0y}, {c1x, c1y},
			      {c2x, c2y}, {x, y},
			      HB_GPU_CU2QU_TOLERANCE, 0);
}


/* ---- Draw funcs ---- */

static void
hb_gpu_draw_move_to (hb_draw_funcs_t *dfuncs HB_UNUSED,
		     void *data,
		     hb_draw_state_t *st HB_UNUSED,
		     float to_x, float to_y,
		     void *user_data HB_UNUSED)
{
  hb_gpu_draw_t *g = (hb_gpu_draw_t *) data;
  g->acc_close_path ();
  g->acc_move_to ((double) to_x, (double) to_y);
}

static void
hb_gpu_draw_line_to (hb_draw_funcs_t *dfuncs HB_UNUSED,
		     void *data,
		     hb_draw_state_t *st HB_UNUSED,
		     float to_x, float to_y,
		     void *user_data HB_UNUSED)
{
  hb_gpu_draw_t *g = (hb_gpu_draw_t *) data;
  g->acc_line_to ((double) to_x, (double) to_y);
}

static void
hb_gpu_draw_quadratic_to (hb_draw_funcs_t *dfuncs HB_UNUSED,
			  void *data,
			  hb_draw_state_t *st HB_UNUSED,
			  float control_x, float control_y,
			  float to_x, float to_y,
			  void *user_data HB_UNUSED)
{
  hb_gpu_draw_t *g = (hb_gpu_draw_t *) data;
  g->acc_conic_to ((double) control_x, (double) control_y,
		   (double) to_x, (double) to_y);
}

static void
hb_gpu_draw_cubic_to (hb_draw_funcs_t *dfuncs HB_UNUSED,
		      void *data,
		      hb_draw_state_t *st HB_UNUSED,
		      float control1_x, float control1_y,
		      float control2_x, float control2_y,
		      float to_x, float to_y,
		      void *user_data HB_UNUSED)
{
  hb_gpu_draw_t *g = (hb_gpu_draw_t *) data;
  g->acc_cubic_to ((double) control1_x, (double) control1_y,
		   (double) control2_x, (double) control2_y,
		   (double) to_x, (double) to_y);
}

static void
hb_gpu_draw_close_path (hb_draw_funcs_t *dfuncs HB_UNUSED,
			void *data,
			hb_draw_state_t *st HB_UNUSED,
			void *user_data HB_UNUSED)
{
  hb_gpu_draw_t *g = (hb_gpu_draw_t *) data;
  g->acc_close_path ();
}

static inline void free_static_gpu_draw_funcs ();

static struct hb_gpu_draw_funcs_lazy_loader_t : hb_draw_funcs_lazy_loader_t<hb_gpu_draw_funcs_lazy_loader_t>
{
  static hb_draw_funcs_t *create ()
  {
    hb_draw_funcs_t *funcs = hb_draw_funcs_create ();

    hb_draw_funcs_set_move_to_func      (funcs, hb_gpu_draw_move_to,      nullptr, nullptr);
    hb_draw_funcs_set_line_to_func      (funcs, hb_gpu_draw_line_to,      nullptr, nullptr);
    hb_draw_funcs_set_quadratic_to_func (funcs, hb_gpu_draw_quadratic_to, nullptr, nullptr);
    hb_draw_funcs_set_cubic_to_func     (funcs, hb_gpu_draw_cubic_to,     nullptr, nullptr);
    hb_draw_funcs_set_close_path_func   (funcs, hb_gpu_draw_close_path,   nullptr, nullptr);

    hb_draw_funcs_make_immutable (funcs);

    hb_atexit (free_static_gpu_draw_funcs);

    return funcs;
  }
} static_gpu_draw_funcs;

static inline void
free_static_gpu_draw_funcs ()
{
  static_gpu_draw_funcs.free_instance ();
}


/* ---- Encode ---- */

struct hb_gpu_texel_t
{
  int16_t r, g, b, a;
};

static hb_position_t
clamp_to_hb_position (double v)
{
  return (hb_position_t) hb_clamp (v,
				   (double) hb_int_min (hb_position_t),
				   (double) hb_int_max (hb_position_t));
}

static inline int16_t
quantize (double v)
{
  return (int16_t) round (v * HB_GPU_UNITS_PER_EM);
}

static inline int16_t
quantize_down (double v)
{
  return (int16_t) floor (v * HB_GPU_UNITS_PER_EM);
}

static inline int16_t
quantize_up (double v)
{
  return (int16_t) ceil (v * HB_GPU_UNITS_PER_EM);
}

static inline double
dequantize (int v)
{
  return (double) v / HB_GPU_UNITS_PER_EM;
}

static inline bool
quantize_down_fits_i16 (double v)
{
  double q = floor (v * HB_GPU_UNITS_PER_EM);
  return q >= INT16_MIN &&
	 q <= INT16_MAX;
}

static inline bool
quantize_up_fits_i16 (double v)
{
  double q = ceil (v * HB_GPU_UNITS_PER_EM);
  return q >= INT16_MIN &&
	 q <= INT16_MAX;
}

static inline int16_t
encode_offset (unsigned offset)
{
  return (int16_t) (offset - 32768u);
}

/* Note: the bounds are computed from the *quantized* control points,
 * because that is what the shader sees.  Band membership derived from
 * the unquantized bounds can leave a curve out of the band that its
 * quantized extent reaches into; fragments in that sliver then never
 * see the curve, get the wrong winding number, and render as a thin
 * black line through the glyph. */
static hb_gpu_encode_curve_info_t
encode_curve_info (const hb_gpu_curve_t *c)
{
  hb_gpu_encode_curve_info_t info;

  int p1x = quantize (c->p1x), p2x = quantize (c->p2x), p3x = quantize (c->p3x);
  int p1y = quantize (c->p1y), p2y = quantize (c->p2y), p3y = quantize (c->p3y);

  info.min_x = dequantize (hb_min (hb_min (p1x, p2x), p3x));
  info.max_x = dequantize (hb_max (hb_max (p1x, p2x), p3x));
  info.min_y = dequantize (hb_min (hb_min (p1y, p2y), p3y));
  info.max_y = dequantize (hb_max (hb_max (p1y, p2y), p3y));
  info.is_horizontal = p1y == p2y && p2y == p3y;
  info.is_vertical   = p1x == p2x && p2x == p3x;
  info.hband_lo = 0;
  info.hband_hi = -1;
  info.vband_lo = 0;
  info.vband_hi = -1;

  return info;
}

static void
_hb_gpu_draw_get_extents (hb_gpu_draw_t      *draw,
			  hb_glyph_extents_t *extents)
{
  if (unlikely (!draw->success) ||
      draw->num_curves == 0 ||
      draw->ext_min_x == HUGE_VAL)
  {
    extents->x_bearing = 0;
    extents->y_bearing = 0;
    extents->width = 0;
    extents->height = 0;
    return;
  }

  double min_x = floor (draw->ext_min_x);
  double min_y = floor (draw->ext_min_y);
  double max_x = ceil  (draw->ext_max_x);
  double max_y = ceil  (draw->ext_max_y);

  if (unlikely (!std::isfinite (min_x) ||
		!std::isfinite (min_y) ||
		!std::isfinite (max_x) ||
		!std::isfinite (max_y)))
  {
    extents->x_bearing = 0;
    extents->y_bearing = 0;
    extents->width = 0;
    extents->height = 0;
    return;
  }

  extents->x_bearing = clamp_to_hb_position (min_x);
  extents->y_bearing = clamp_to_hb_position (max_y);
  extents->width     = clamp_to_hb_position (max_x - min_x);
  extents->height    = clamp_to_hb_position (min_y - max_y);
}


/**
 * hb_gpu_draw_encode:
 * @draw: a GPU shape encoder
 * @extents: (out) (nullable): where to store the computed glyph
 *           extents (in font units, Y-up).  Pass `NULL` if not
 *           needed.
 *
 * Encodes the accumulated outlines into a compact blob
 * suitable for GPU rendering.  The blob data is an array of
 * RGBA16I texels (8 bytes each) to be uploaded to a texture
 * buffer object.
 *
 * The returned blob owns its own copy of the data.  On success
 * @draw is auto-cleared so it can be reused for the next glyph;
 * user configuration (font scale) is preserved.
 *
 * Return value: (transfer full):
 * An #hb_blob_t containing the encoded data, or `NULL` if encoding
 * failed (allocation failure or accumulation error).  When the
 * encoder accumulated no outline (e.g. the glyph has no ink, like a
 * space), returns the empty-blob singleton instead of `NULL`, so
 * callers can distinguish "nothing to render" (length 0) from a
 * real failure (`NULL`).
 *
 * Since: 14.0.0
 **/
hb_blob_t *
hb_gpu_draw_encode (hb_gpu_draw_t      *draw,
                    hb_glyph_extents_t *extents)
{
  /* Capture computed extents before auto-clear wipes them. */
  if (extents)
    _hb_gpu_draw_get_extents (draw, extents);
  HB_SCOPE_GUARD (hb_gpu_draw_clear (draw));

  if (unlikely (!draw->success))
    return nullptr;

  const hb_gpu_curve_t *curves = draw->curves.arrayZ;
  unsigned num_curves = draw->curves.length;

  if (num_curves == 0)
    return hb_blob_get_empty ();

  hb_gpu_encode_scratch_t &s = draw->scratch;

  s.curve_infos.reset_if_error ();
  s.hband_curve_counts.reset_if_error ();
  s.vband_curve_counts.reset_if_error ();
  s.hband_offsets.reset_if_error ();
  s.vband_offsets.reset_if_error ();
  s.hband_curves.reset_if_error ();
  s.hband_curves_asc.reset_if_error ();
  s.vband_curves.reset_if_error ();
  s.vband_curves_asc.reset_if_error ();
  s.hband_cursors.reset_if_error ();
  s.vband_cursors.reset_if_error ();
  s.curve_texel_offset.reset_if_error ();

  /* Compute per-curve info and extents */
  if (unlikely (!s.curve_infos.resize (num_curves)))
    return nullptr;

  /* Validate the extents fit in i16 before quantizing: the casts in
   * quantize_down()/quantize_up() are undefined for out-of-range
   * values, and every later i16 conversion of a curve point is bounded
   * by these extents. */
  if (!quantize_down_fits_i16 (draw->ext_min_x) ||
      !quantize_down_fits_i16 (draw->ext_min_y) ||
      !quantize_up_fits_i16 (draw->ext_max_x) ||
      !quantize_up_fits_i16 (draw->ext_max_y))
    return nullptr;

  int16_t min_x_q = quantize_down (draw->ext_min_x);
  int16_t min_y_q = quantize_down (draw->ext_min_y);
  int16_t max_x_q = quantize_up (draw->ext_max_x);
  int16_t max_y_q = quantize_up (draw->ext_max_y);

  double min_x = dequantize (min_x_q);
  double min_y = dequantize (min_y_q);
  double max_x = dequantize (max_x_q);
  double max_y = dequantize (max_y_q);

  for (unsigned i = 0; i < num_curves; i++)
    s.curve_infos.arrayZ[i] = encode_curve_info (&curves[i]);

  /* Choose number of bands (capped at 16 per Slug paper) */
  unsigned num_hbands = hb_min (num_curves, 16u);
  unsigned num_vbands = hb_min (num_curves, 16u);
  num_hbands = hb_max (num_hbands, 1u);
  num_vbands = hb_max (num_vbands, 1u);

  double height = max_y - min_y;
  double width  = max_x - min_x;

  if (height <= 0) num_hbands = 1;
  if (width  <= 0) num_vbands = 1;

  double hband_size = height / num_hbands;
  double vband_size = width  / num_vbands;

  /* The shader recomputes the band index from the fragment position in
   * float32, this code does it in double.  Widen the band range of each
   * curve by a hair so the two cannot disagree at a band boundary. */
  static const double BAND_EPSILON = 1.0 / 1024;

  if (unlikely (!s.hband_curve_counts.resize (num_hbands) ||
		!s.vband_curve_counts.resize (num_vbands)))
    return nullptr;

  for (unsigned b = 0; b < num_hbands; b++) s.hband_curve_counts.arrayZ[b] = 0;
  for (unsigned b = 0; b < num_vbands; b++) s.vband_curve_counts.arrayZ[b] = 0;

  for (unsigned i = 0; i < num_curves; i++)
  {
    hb_gpu_encode_curve_info_t &info = s.curve_infos.arrayZ[i];

    if (!info.is_horizontal)
    {
      if (height > 0) {
	info.hband_lo = (int) floor ((info.min_y - min_y) / hband_size - BAND_EPSILON);
	info.hband_hi = (int) floor ((info.max_y - min_y) / hband_size + BAND_EPSILON);
	info.hband_lo = hb_max (info.hband_lo, 0);
	info.hband_hi = hb_min (info.hband_hi, (int) num_hbands - 1);
	for (int b = info.hband_lo; b <= info.hband_hi; b++)
	  s.hband_curve_counts.arrayZ[b]++;
      } else {
	info.hband_lo = 0;
	info.hband_hi = 0;
	s.hband_curve_counts.arrayZ[0]++;
      }
    }

    if (!info.is_vertical)
    {
      if (width > 0) {
	info.vband_lo = (int) floor ((info.min_x - min_x) / vband_size - BAND_EPSILON);
	info.vband_hi = (int) floor ((info.max_x - min_x) / vband_size + BAND_EPSILON);
	info.vband_lo = hb_max (info.vband_lo, 0);
	info.vband_hi = hb_min (info.vband_hi, (int) num_vbands - 1);
	for (int b = info.vband_lo; b <= info.vband_hi; b++)
	  s.vband_curve_counts.arrayZ[b]++;
      } else {
	info.vband_lo = 0;
	info.vband_hi = 0;
	s.vband_curve_counts.arrayZ[0]++;
      }
    }
  }

  if (unlikely (!s.hband_offsets.resize (num_hbands) ||
		!s.vband_offsets.resize (num_vbands)))
    return nullptr;

  unsigned total_hband_indices = 0;
  unsigned total_vband_indices = 0;

  for (unsigned b = 0; b < num_hbands; b++) {
    s.hband_offsets.arrayZ[b] = total_hband_indices;
    if (unlikely (hb_unsigned_add_overflows (total_hband_indices,
					     s.hband_curve_counts.arrayZ[b],
					     &total_hband_indices)))
      return nullptr;
  }

  for (unsigned b = 0; b < num_vbands; b++) {
    s.vband_offsets.arrayZ[b] = total_vband_indices;
    if (unlikely (hb_unsigned_add_overflows (total_vband_indices,
					     s.vband_curve_counts.arrayZ[b],
					     &total_vband_indices)))
      return nullptr;
  }

  /* Assign curves to bands */
  if (unlikely (!s.hband_curves.resize (total_hband_indices) ||
		!s.hband_curves_asc.resize (total_hband_indices) ||
		!s.vband_curves.resize (total_vband_indices) ||
		!s.vband_curves_asc.resize (total_vband_indices) ||
		!s.hband_cursors.resize (num_hbands) ||
		!s.vband_cursors.resize (num_vbands)))
    return nullptr;

  for (unsigned b = 0; b < num_hbands; b++) s.hband_cursors.arrayZ[b] = s.hband_offsets.arrayZ[b];
  for (unsigned b = 0; b < num_vbands; b++) s.vband_cursors.arrayZ[b] = s.vband_offsets.arrayZ[b];

  for (unsigned i = 0; i < num_curves; i++)
  {
    const hb_gpu_encode_curve_info_t &info = s.curve_infos.arrayZ[i];

    for (int b = info.hband_lo; b <= info.hband_hi; b++) {
      unsigned idx = s.hband_cursors.arrayZ[b]++;
      s.hband_curves.arrayZ[idx] = i;
      s.hband_curves_asc.arrayZ[idx] = i;
    }

    for (int b = info.vband_lo; b <= info.vband_hi; b++) {
      unsigned idx = s.vband_cursors.arrayZ[b]++;
      s.vband_curves.arrayZ[idx] = i;
      s.vband_curves_asc.arrayZ[idx] = i;
    }
  }

  /* Sort: descending by max, ascending by min */
  const hb_gpu_encode_curve_info_t *infos = s.curve_infos.arrayZ;

  for (unsigned b = 0; b < num_hbands; b++)
  {
    unsigned off = s.hband_offsets.arrayZ[b];
    unsigned count = s.hband_curve_counts.arrayZ[b];
    s.hband_curves.as_array ().sub_array (off, count)
      .qsort ([infos] (const unsigned &a, const unsigned &b) {
	auto av = infos[a].max_x, bv = infos[b].max_x;
	return (av < bv) - (av > bv);
      });
    s.hband_curves_asc.as_array ().sub_array (off, count)
      .qsort ([infos] (const unsigned &a, const unsigned &b) {
	auto av = infos[a].min_x, bv = infos[b].min_x;
	return (av > bv) - (av < bv);
      });
  }

  for (unsigned b = 0; b < num_vbands; b++)
  {
    unsigned off = s.vband_offsets.arrayZ[b];
    unsigned count = s.vband_curve_counts.arrayZ[b];
    s.vband_curves.as_array ().sub_array (off, count)
      .qsort ([infos] (const unsigned &a, const unsigned &b) {
	auto av = infos[a].max_y, bv = infos[b].max_y;
	return (av < bv) - (av > bv);
      });
    s.vband_curves_asc.as_array ().sub_array (off, count)
      .qsort ([infos] (const unsigned &a, const unsigned &b) {
	auto av = infos[a].min_y, bv = infos[b].min_y;
	return (av > bv) - (av < bv);
      });
  }

  /* Compute sizes */
  unsigned total_curve_indices;
  if (unlikely (hb_unsigned_add_overflows (total_hband_indices,
					   total_vband_indices,
					   &total_curve_indices) ||
		hb_unsigned_mul_overflows (total_curve_indices,
					   2,
					   &total_curve_indices)))
    return nullptr;

  unsigned header_len = 2;

  unsigned num_contour_breaks = 0;
  for (unsigned i = 0; i + 1 < num_curves; i++)
    if (curves[i + 1].contour_start)
      num_contour_breaks++;

  unsigned curve_data_len;
  if (unlikely (hb_unsigned_add_overflows (num_curves,
					   num_contour_breaks,
					   &curve_data_len) ||
		hb_unsigned_add_overflows (curve_data_len,
					   1,
					   &curve_data_len)))
    return nullptr;

  unsigned band_headers_len;
  if (unlikely (hb_unsigned_add_overflows (num_hbands,
					   num_vbands,
					   &band_headers_len)))
    return nullptr;

  unsigned total_len = header_len;
  if (unlikely (hb_unsigned_add_overflows (total_len,
					   band_headers_len,
					   &total_len) ||
		hb_unsigned_add_overflows (total_len,
					   total_curve_indices,
					   &total_len) ||
		hb_unsigned_add_overflows (total_len,
					   curve_data_len,
					   &total_len)))
    return nullptr;

  /* Validate fits in uint16 offsets (stored as int16 with bias) */
  if (total_len > (unsigned) UINT16_MAX + 1u)
    return nullptr;

  /* Allocate or reuse encode buffer */
  unsigned needed_bytes;
  if (unlikely (hb_unsigned_mul_overflows (total_len,
					   sizeof (hb_gpu_texel_t),
					   &needed_bytes)))
    return nullptr;
  unsigned buf_capacity = 0;
  char *replaced_recycled_buf = nullptr;
  char *buf_raw = hb_blob_t::recycle_acquire (draw->recycled_blob, needed_bytes,
					&buf_capacity, &replaced_recycled_buf);
  if (unlikely (!buf_raw))
    return nullptr;
  hb_gpu_texel_t *buf = (hb_gpu_texel_t *) (void *) buf_raw;

  unsigned curve_data_offset = header_len;
  if (unlikely (hb_unsigned_add_overflows (curve_data_offset,
					   band_headers_len,
					   &curve_data_offset) ||
		hb_unsigned_add_overflows (curve_data_offset,
					   total_curve_indices,
					   &curve_data_offset)))
  {
    hb_blob_t::recycle_abort ((char *) buf, draw->recycled_blob);
    return nullptr;
  }

  /* Pack header */
  buf[0].r = min_x_q;
  buf[0].g = min_y_q;
  buf[0].b = max_x_q;
  buf[0].a = max_y_q;
  buf[1].r = (int16_t) num_hbands;
  buf[1].g = (int16_t) num_vbands;
  buf[1].b = (int16_t) hb_clamp (draw->x_scale, -32768, 32767);
  buf[1].a = (int16_t) hb_clamp (draw->y_scale, -32768, 32767);

  /* Pack curve data with shared endpoints */
  if (unlikely (!s.curve_texel_offset.resize (num_curves)))
  {
    hb_blob_t::recycle_abort ((char *) buf, draw->recycled_blob);
    return nullptr;
  }

  unsigned texel = curve_data_offset;

  for (unsigned i = 0; i < num_curves; i++)
  {
    bool contour_start = curves[i].contour_start;

    if (contour_start) {
      s.curve_texel_offset.arrayZ[i] = texel;
      buf[texel].r = quantize (curves[i].p1x);
      buf[texel].g = quantize (curves[i].p1y);
      buf[texel].b = quantize (curves[i].p2x);
      buf[texel].a = quantize (curves[i].p2y);
      texel++;
    } else {
      s.curve_texel_offset.arrayZ[i] = texel - 1;
    }

    bool has_next = i + 1 < num_curves &&
		    !curves[i + 1].contour_start;

    buf[texel].r = quantize (curves[i].p3x);
    buf[texel].g = quantize (curves[i].p3y);
    if (has_next) {
      buf[texel].b = quantize (curves[i + 1].p2x);
      buf[texel].a = quantize (curves[i + 1].p2y);
    } else {
      buf[texel].b = 0;
      buf[texel].a = 0;
    }
    texel++;
  }

  /* Pack band headers and curve indices */
  unsigned index_offset = header_len + band_headers_len;

  for (unsigned b = 0; b < num_hbands; b++)
  {
    int16_t hband_split;
    {
      unsigned off = s.hband_offsets.arrayZ[b];
      unsigned n = s.hband_curve_counts.arrayZ[b];
      unsigned best_worst = n;
      double best_split = (min_x + max_x) * 0.5;
      unsigned left_count = n;
      for (unsigned ci = 0; ci < n; ci++) {
	double split = s.curve_infos.arrayZ[s.hband_curves.arrayZ[off + ci]].max_x;
	unsigned right_count = ci + 1;
	while (left_count &&
	       s.curve_infos.arrayZ[s.hband_curves_asc.arrayZ[off + left_count - 1]].min_x > split)
	  left_count--;
	unsigned worst = hb_max (right_count, left_count);
	if (worst < best_worst) {
	  best_worst = worst;
	  best_split = split;
	}
      }
      hband_split = quantize (best_split);
    }
    unsigned hdr = header_len + b;
    unsigned desc_off = index_offset;

    for (unsigned ci = 0; ci < s.hband_curve_counts.arrayZ[b]; ci++) {
      buf[index_offset].r = encode_offset (s.curve_texel_offset.arrayZ[s.hband_curves.arrayZ[s.hband_offsets.arrayZ[b] + ci]]);
      buf[index_offset].g = 0;
      buf[index_offset].b = 0;
      buf[index_offset].a = 0;
      index_offset++;
    }

    unsigned asc_off = index_offset;

    for (unsigned ci = 0; ci < s.hband_curve_counts.arrayZ[b]; ci++) {
      buf[index_offset].r = encode_offset (s.curve_texel_offset.arrayZ[s.hband_curves_asc.arrayZ[s.hband_offsets.arrayZ[b] + ci]]);
      buf[index_offset].g = 0;
      buf[index_offset].b = 0;
      buf[index_offset].a = 0;
      index_offset++;
    }

    buf[hdr].r = (int16_t) s.hband_curve_counts.arrayZ[b];
    buf[hdr].g = encode_offset (desc_off);
    buf[hdr].b = encode_offset (asc_off);
    buf[hdr].a = hband_split;
  }

  for (unsigned b = 0; b < num_vbands; b++)
  {
    int16_t vband_split;
    {
      unsigned off = s.vband_offsets.arrayZ[b];
      unsigned n = s.vband_curve_counts.arrayZ[b];
      unsigned best_worst = n;
      double best_split = (min_y + max_y) * 0.5;
      unsigned left_count = n;
      for (unsigned ci = 0; ci < n; ci++) {
	double split = s.curve_infos.arrayZ[s.vband_curves.arrayZ[off + ci]].max_y;
	unsigned right_count = ci + 1;
	while (left_count &&
	       s.curve_infos.arrayZ[s.vband_curves_asc.arrayZ[off + left_count - 1]].min_y > split)
	  left_count--;
	unsigned worst = hb_max (right_count, left_count);
	if (worst < best_worst) {
	  best_worst = worst;
	  best_split = split;
	}
      }
      vband_split = quantize (best_split);
    }

    unsigned hdr = header_len + num_hbands + b;
    unsigned desc_off = index_offset;

    for (unsigned ci = 0; ci < s.vband_curve_counts.arrayZ[b]; ci++) {
      buf[index_offset].r = encode_offset (s.curve_texel_offset.arrayZ[s.vband_curves.arrayZ[s.vband_offsets.arrayZ[b] + ci]]);
      buf[index_offset].g = 0;
      buf[index_offset].b = 0;
      buf[index_offset].a = 0;
      index_offset++;
    }

    unsigned asc_off = index_offset;

    for (unsigned ci = 0; ci < s.vband_curve_counts.arrayZ[b]; ci++) {
      buf[index_offset].r = encode_offset (s.curve_texel_offset.arrayZ[s.vband_curves_asc.arrayZ[s.vband_offsets.arrayZ[b] + ci]]);
      buf[index_offset].g = 0;
      buf[index_offset].b = 0;
      buf[index_offset].a = 0;
      index_offset++;
    }

    buf[hdr].r = (int16_t) s.vband_curve_counts.arrayZ[b];
    buf[hdr].g = encode_offset (desc_off);
    buf[hdr].b = encode_offset (asc_off);
    buf[hdr].a = vband_split;
  }

  hb_blob_t *recycled = draw->recycled_blob;
  draw->recycled_blob = nullptr;
  return hb_blob_t::recycle_finalize ((char *) buf, buf_capacity, needed_bytes,
				recycled, replaced_recycled_buf);
}


/* ---- Public API ---- */

/**
 * hb_gpu_draw_create_or_fail:
 *
 * Creates a new GPU shape encoder.
 *
 * Return value: (transfer full):
 * A newly allocated #hb_gpu_draw_t, or `NULL` on allocation failure.
 *
 * Since: 14.0.0
 **/
hb_gpu_draw_t *
hb_gpu_draw_create_or_fail (void)
{
  return hb_object_create<hb_gpu_draw_t> ();
}

/**
 * hb_gpu_draw_reference: (skip)
 * @draw: a GPU shape encoder
 *
 * Increases the reference count on @draw by one.
 *
 * Return value: (transfer full):
 * The referenced #hb_gpu_draw_t.
 *
 * Since: 14.0.0
 **/
hb_gpu_draw_t *
hb_gpu_draw_reference (hb_gpu_draw_t *draw)
{
  return hb_object_reference (draw);
}

/**
 * hb_gpu_draw_destroy: (skip)
 * @draw: a GPU shape encoder
 *
 * Decreases the reference count on @draw by one. When the
 * reference count reaches zero, the encoder is freed.
 *
 * Since: 14.0.0
 **/
void
hb_gpu_draw_destroy (hb_gpu_draw_t *draw)
{
  if (!hb_object_should_destroy (draw))
    return;

  hb_blob_destroy (draw->recycled_blob);

  hb_object_actually_destroy (draw);
  hb_free (draw);
}

/**
 * hb_gpu_draw_set_user_data: (skip)
 * @draw: a GPU shape encoder
 * @key: the user-data key
 * @data: a pointer to the user data
 * @destroy: (nullable): a callback to call when @data is not needed anymore
 * @replace: whether to replace an existing data with the same key
 *
 * Attaches user data to @draw.
 *
 * Return value: `true` if success, `false` otherwise
 *
 * Since: 14.0.0
 **/
hb_bool_t
hb_gpu_draw_set_user_data (hb_gpu_draw_t     *draw,
			     hb_user_data_key_t *key,
			     void               *data,
			     hb_destroy_func_t   destroy,
			     hb_bool_t           replace)
{
  return hb_object_set_user_data (draw, key, data, destroy, replace);
}

/**
 * hb_gpu_draw_get_user_data: (skip)
 * @draw: a GPU shape encoder
 * @key: the user-data key
 *
 * Fetches the user-data associated with the specified key.
 *
 * Return value: (transfer none):
 * A pointer to the user data
 *
 * Since: 14.0.0
 **/
void *
hb_gpu_draw_get_user_data (const hb_gpu_draw_t     *draw,
			     hb_user_data_key_t *key)
{
  return hb_object_get_user_data (draw, key);
}

/**
 * hb_gpu_draw_get_funcs:
 * @draw: a GPU draw context.
 *
 * Fetches the #hb_draw_funcs_t that feeds outline data into
 * @draw.  Pass @draw as the @draw_data argument when calling
 * the draw functions.
 *
 * Return value: (transfer none):
 * The GPU draw functions
 *
 * Since: 14.2.0
 **/
hb_draw_funcs_t *
hb_gpu_draw_get_funcs (const hb_gpu_draw_t *draw HB_UNUSED)
{
  return static_gpu_draw_funcs.get_unconst ();
}

/**
 * hb_gpu_draw_set_scale:
 * @draw: a GPU shape encoder
 * @x_scale: horizontal scale (typically from hb_font_get_scale())
 * @y_scale: vertical scale
 *
 * Sets the font scale so the encoded blob can embed it for
 * shader use (e.g. computing pixels-per-em).  Called
 * automatically by hb_gpu_draw_glyph().
 *
 * Since: 14.1.0
 **/
void
hb_gpu_draw_set_scale (hb_gpu_draw_t *draw,
		       int            x_scale,
		       int            y_scale)
{
  draw->x_scale = x_scale;
  draw->y_scale = y_scale;
}

/**
 * hb_gpu_draw_get_scale:
 * @draw: a GPU shape encoder
 * @x_scale: (out): horizontal scale
 * @y_scale: (out): vertical scale
 *
 * Gets the font scale previously set via hb_gpu_draw_set_scale() or
 * hb_gpu_draw_glyph().
 *
 * Since: 14.2.0
 **/
void
hb_gpu_draw_get_scale (const hb_gpu_draw_t *draw,
		       int                 *x_scale,
		       int                 *y_scale)
{
  if (x_scale) *x_scale = draw->x_scale;
  if (y_scale) *y_scale = draw->y_scale;
}

/**
 * hb_gpu_draw_glyph_or_fail:
 * @draw: a GPU shape encoder
 * @font: font to draw from
 * @glyph: glyph ID to draw
 *
 * Convenience to draw one glyph outline.  Equivalent to:
 *
 * |[<!-- language="plain" -->
 * hb_gpu_draw_set_scale (draw, x_scale, y_scale);
 * hb_font_draw_glyph_or_fail (font, glyph,
 *   hb_gpu_draw_get_funcs (draw), draw);
 * ]|
 *
 * Return value: `true` if the glyph was drawn, `false` if the font
 * has no outlines for @glyph.
 *
 * Since: 14.2.0
 **/
hb_bool_t
hb_gpu_draw_glyph_or_fail (hb_gpu_draw_t  *draw,
			   hb_font_t      *font,
			   hb_codepoint_t  glyph)
{
  int x_scale, y_scale;
  hb_font_get_scale (font, &x_scale, &y_scale);
  hb_gpu_draw_set_scale (draw, x_scale, y_scale);

  return hb_font_draw_glyph_or_fail (font, glyph,
				     hb_gpu_draw_get_funcs (draw),
				     draw);
}

/**
 * hb_gpu_draw_glyph:
 * @draw: a GPU shape encoder
 * @font: font to draw from
 * @glyph: glyph ID to draw
 *
 * Draws a single glyph outline into the encoder.  Equivalent to
 * hb_gpu_draw_glyph_or_fail() with the return value ignored.
 *
 * Since: 14.0.0
 **/
void
hb_gpu_draw_glyph (hb_gpu_draw_t  *draw,
		   hb_font_t      *font,
		   hb_codepoint_t  glyph)
{
  hb_gpu_draw_glyph_or_fail (draw, font, glyph);
}

/**
 * hb_gpu_draw_clear:
 * @draw: a GPU shape encoder
 *
 * Discards accumulated outlines so @draw can be reused for another
 * encode.  User configuration (font scale) is preserved.  Call
 * hb_gpu_draw_reset() to also reset user configuration to defaults.
 *
 * Since: 14.2.0
 **/
void
hb_gpu_draw_clear (hb_gpu_draw_t *draw)
{
  draw->start_x = draw->start_y = 0;
  draw->current_x = draw->current_y = 0;
  draw->need_moveto = true;
  draw->num_curves = 0;
  draw->success = true;
  draw->curves.reset ();

  draw->ext_min_x =  HUGE_VAL;
  draw->ext_min_y =  HUGE_VAL;
  draw->ext_max_x = -HUGE_VAL;
  draw->ext_max_y = -HUGE_VAL;
}

/**
 * hb_gpu_draw_reset:
 * @draw: a GPU shape encoder
 *
 * Resets the encoder, discarding all accumulated outlines and
 * resetting user configuration to defaults.
 *
 * Since: 14.0.0
 **/
void
hb_gpu_draw_reset (hb_gpu_draw_t *draw)
{
  draw->x_scale = 0;
  draw->y_scale = 0;
  hb_gpu_draw_clear (draw);
}

/**
 * hb_gpu_draw_recycle_blob:
 * @draw: a GPU shape encoder
 * @blob: (transfer full): a blob previously returned by hb_gpu_draw_encode()
 *
 * Returns a blob to the encoder for potential reuse.
 * The caller transfers ownership of @blob.
 *
 * Currently this simply destroys the blob.  A future version
 * may reclaim the underlying buffer to avoid allocation on the
 * next hb_gpu_draw_encode() call.
 *
 * Since: 14.0.0
 **/
void
hb_gpu_draw_recycle_blob (hb_gpu_draw_t *draw,
			    hb_blob_t      *blob)
{
  hb_blob_t::recycle_stash (&draw->recycled_blob, blob);
}


#include "hb-gpu-draw-fragment-glsl.hh"
#include "hb-gpu-draw-fragment-msl.hh"
#include "hb-gpu-draw-fragment-wgsl.hh"
#include "hb-gpu-draw-fragment-hlsl.hh"

/**
 * hb_gpu_draw_shader_source:
 * @stage: pipeline stage (vertex or fragment)
 * @lang: shader language variant
 *
 * Returns the draw-renderer-specific shader source for the
 * specified stage and language.  The returned string is static
 * and must not be freed.
 *
 * This source assumes the shared helpers returned by
 * hb_gpu_shader_source() are concatenated ahead of it.  The
 * caller should assemble the full shader as
 * `#version`-directive + hb_gpu_shader_source() +
 * hb_gpu_draw_shader_source() + caller's `main()`.
 *
 * The vertex stage currently has no draw-specific helpers; this
 * function returns an empty string for that stage so the caller
 * can concatenate unconditionally.
 *
 * Return value: (transfer none):
 * A shader source string, or `NULL` if @stage or @lang is
 * unsupported.
 *
 * Since: 14.2.0
 **/
const char *
hb_gpu_draw_shader_source (hb_gpu_shader_stage_t stage,
			   hb_gpu_shader_lang_t  lang)
{
  switch (stage) {
  case HB_GPU_SHADER_STAGE_FRAGMENT:
    switch (lang) {
    case HB_GPU_SHADER_LANG_GLSL: return hb_gpu_draw_fragment_glsl;
    case HB_GPU_SHADER_LANG_MSL:  return hb_gpu_draw_fragment_msl;
    case HB_GPU_SHADER_LANG_WGSL: return hb_gpu_draw_fragment_wgsl;
    case HB_GPU_SHADER_LANG_HLSL: return hb_gpu_draw_fragment_hlsl;
    case HB_GPU_SHADER_LANG_INVALID:
    default: return nullptr;
    }
  case HB_GPU_SHADER_STAGE_VERTEX:
    switch (lang) {
    case HB_GPU_SHADER_LANG_GLSL:
    case HB_GPU_SHADER_LANG_MSL:
    case HB_GPU_SHADER_LANG_WGSL:
    case HB_GPU_SHADER_LANG_HLSL: return "";
    case HB_GPU_SHADER_LANG_INVALID:
    default: return nullptr;
    }
  default:
    return nullptr;
  }
}
