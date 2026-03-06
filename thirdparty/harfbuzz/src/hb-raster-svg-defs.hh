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

#ifndef HB_RASTER_SVG_DEFS_HH
#define HB_RASTER_SVG_DEFS_HH

#include "hb.hh"

#include "hb-map.hh"
#include "hb-raster-svg-parse.hh"

enum hb_svg_gradient_type_t
{
  SVG_GRADIENT_LINEAR,
  SVG_GRADIENT_RADIAL
};

struct hb_svg_gradient_stop_t
{
  float offset;
  hb_color_t color;
  bool is_current_color = false;
};

struct hb_svg_gradient_t
{
  hb_svg_gradient_type_t type;

  hb_paint_extend_t spread = HB_PAINT_EXTEND_PAD;
  bool has_spread = false;
  hb_svg_transform_t gradient_transform;
  bool has_gradient_transform = false;
  bool units_user_space = false;
  bool has_units_user_space = false;

  float x1 = 0, y1 = 0, x2 = 1, y2 = 0;
  bool has_x1 = false, has_y1 = false, has_x2 = false, has_y2 = false;

  float cx = 0.5f, cy = 0.5f, r = 0.5f;
  float fx = -1.f, fy = -1.f, fr = 0.f;
  bool has_cx = false, has_cy = false, has_r = false;
  bool has_fx = false, has_fy = false, has_fr = false;

  hb_vector_t<hb_svg_gradient_stop_t> stops;

  hb_bytes_t href_id = {};
};

struct hb_svg_clip_path_def_t
{
  unsigned first_shape = 0;
  unsigned shape_count = 0;
  hb_svg_transform_t clip_transform;
  bool has_clip_transform = false;
  bool units_user_space = true;
};

struct hb_svg_clip_shape_t
{
  hb_svg_shape_emit_data_t shape;
  hb_svg_transform_t transform;
  bool has_transform = false;
};

struct hb_svg_defs_t
{
  hb_vector_t<hb_svg_gradient_t> gradients;
  hb_vector_t<hb_svg_clip_shape_t> clip_shapes;
  hb_vector_t<hb_svg_clip_path_def_t> clip_paths;
  hb_hashmap_t<hb_bytes_t, unsigned> gradient_by_id;
  hb_hashmap_t<hb_bytes_t, unsigned> clip_path_by_id;
  hb_vector_t<char *> owned_id_strings;

  ~hb_svg_defs_t ();

  bool add_id_mapping (hb_hashmap_t<hb_bytes_t, unsigned> *map,
                       hb_bytes_t id,
                       unsigned idx);
  bool add_gradient (hb_bytes_t id, const hb_svg_gradient_t &grad);
  const hb_svg_gradient_t *find_gradient (hb_bytes_t id) const;
  bool add_clip_path (hb_bytes_t id, const hb_svg_clip_path_def_t &clip);
  const hb_svg_clip_path_def_t *find_clip_path (hb_bytes_t id) const;
};

#endif /* HB_RASTER_SVG_DEFS_HH */
