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

#ifndef HB_NO_RASTER_SVG

#include "hb.hh"

#include "hb-raster-svg-bbox.hh"

static void
svg_bbox_move_to (hb_draw_funcs_t *dfuncs HB_UNUSED,
                  void *draw_data,
                  hb_draw_state_t *st HB_UNUSED,
                  float to_x, float to_y,
                  void *user_data HB_UNUSED)
{
  ((hb_extents_t<> *) draw_data)->add_point (to_x, to_y);
}

static void
svg_bbox_line_to (hb_draw_funcs_t *dfuncs HB_UNUSED,
                  void *draw_data,
                  hb_draw_state_t *st HB_UNUSED,
                  float to_x, float to_y,
                  void *user_data HB_UNUSED)
{
  ((hb_extents_t<> *) draw_data)->add_point (to_x, to_y);
}

static void
svg_bbox_quadratic_to (hb_draw_funcs_t *dfuncs HB_UNUSED,
                       void *draw_data,
                       hb_draw_state_t *st HB_UNUSED,
                       float control_x, float control_y,
                       float to_x, float to_y,
                       void *user_data HB_UNUSED)
{
  hb_extents_t<> *ext = (hb_extents_t<> *) draw_data;
  ext->add_point (control_x, control_y);
  ext->add_point (to_x, to_y);
}

static void
svg_bbox_cubic_to (hb_draw_funcs_t *dfuncs HB_UNUSED,
                   void *draw_data,
                   hb_draw_state_t *st HB_UNUSED,
                   float control1_x, float control1_y,
                   float control2_x, float control2_y,
                   float to_x, float to_y,
                   void *user_data HB_UNUSED)
{
  hb_extents_t<> *ext = (hb_extents_t<> *) draw_data;
  ext->add_point (control1_x, control1_y);
  ext->add_point (control2_x, control2_y);
  ext->add_point (to_x, to_y);
}

static hb_draw_funcs_t *
svg_bbox_draw_funcs ()
{
  static hb_draw_funcs_t *funcs = nullptr;
  if (unlikely (!funcs))
  {
    funcs = hb_draw_funcs_create ();
    hb_draw_funcs_set_move_to_func (funcs, svg_bbox_move_to, nullptr, nullptr);
    hb_draw_funcs_set_line_to_func (funcs, svg_bbox_line_to, nullptr, nullptr);
    hb_draw_funcs_set_quadratic_to_func (funcs, svg_bbox_quadratic_to, nullptr, nullptr);
    hb_draw_funcs_set_cubic_to_func (funcs, svg_bbox_cubic_to, nullptr, nullptr);
    hb_draw_funcs_make_immutable (funcs);
  }
  return funcs;
}

bool
hb_raster_svg_compute_shape_bbox (const hb_svg_shape_emit_data_t &shape,
                        hb_extents_t<> *bbox)
{
  hb_extents_t<> ext;
  hb_svg_shape_emit_data_t tmp = shape;
  hb_raster_svg_shape_path_emit (svg_bbox_draw_funcs (), &ext, &tmp);
  if (ext.is_empty ()) return false;
  *bbox = ext;
  return true;
}

#endif /* !HB_NO_RASTER_SVG */
