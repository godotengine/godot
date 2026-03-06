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

#include "hb-raster-svg-gradient.hh"

#include "hb-raster-svg-base.hh"

static bool
svg_parse_gradient_stop (hb_svg_xml_parser_t &parser,
                         hb_svg_gradient_t &grad,
                         hb_paint_funcs_t *pfuncs,
                         void *paint_data,
                         hb_color_t foreground,
                         hb_face_t *face,
                         unsigned palette)
{
  const unsigned SVG_MAX_GRADIENT_STOPS = 1024;
  if (grad.stops.length >= SVG_MAX_GRADIENT_STOPS)
    return true;

  hb_svg_attr_view_t attrs (parser);
  hb_svg_str_t style = attrs.get ("style");
  hb_svg_style_props_t style_props;
  svg_parse_style_props (style, &style_props);
  hb_svg_str_t offset_str = svg_pick_attr_or_style (parser, style_props.offset, "offset");
  hb_svg_str_t color_str = svg_pick_attr_or_style (parser, style_props.stop_color, "stop-color");
  hb_svg_str_t opacity_str = svg_pick_attr_or_style (parser, style_props.stop_opacity, "stop-opacity");
  hb_svg_str_t display_str = svg_pick_attr_or_style (parser, style_props.display, "display");
  hb_svg_str_t visibility_str = svg_pick_attr_or_style (parser, style_props.visibility, "visibility");

  if (display_str.trim ().eq_ascii_ci ("none"))
    return true;
  hb_svg_str_t visibility_trim = visibility_str.trim ();
  if (visibility_trim.eq_ascii_ci ("hidden") ||
      visibility_trim.eq_ascii_ci ("collapse"))
    return true;

  float offset = 0;
  if (offset_str.len)
    offset = hb_clamp (svg_parse_number_or_percent (offset_str, nullptr), 0.f, 1.f);
  if (grad.stops.length)
    offset = hb_max (offset, grad.stops.arrayZ[grad.stops.length - 1].offset);

  bool is_none = false;
  hb_color_t color = HB_COLOR (0, 0, 0, 255);
  bool is_current_color = false;
  if (color_str.len && !svg_str_is_inherit (color_str))
  {
    is_current_color = color_str.trim ().eq_ascii_ci ("currentColor");
    color = hb_raster_svg_parse_color (color_str, pfuncs, paint_data, foreground, face, palette, &is_none);
  }

  if (opacity_str.len && !svg_str_is_inherit (opacity_str))
  {
    float opacity = svg_parse_float_clamped01 (opacity_str);
    color = HB_COLOR (hb_color_get_blue (color),
                      hb_color_get_green (color),
                      hb_color_get_red (color),
                      (uint8_t) (hb_color_get_alpha (color) * opacity + 0.5f));
  }

  hb_svg_gradient_stop_t stop;
  stop.offset = offset;
  stop.color = color;
  stop.is_current_color = is_current_color;
  grad.stops.push (stop);
  return !grad.stops.in_error ();
}

static void
svg_parse_gradient_attrs (hb_svg_xml_parser_t &parser,
                          hb_svg_gradient_t &grad)
{
  hb_svg_style_props_t style_props;
  svg_parse_style_props (parser.find_attr ("style"), &style_props);

  hb_svg_str_t spread_str = svg_pick_attr_or_style (parser, style_props.spread_method, "spreadMethod").trim ();
  if (spread_str.eq_ascii_ci ("reflect"))
  {
    grad.spread = HB_PAINT_EXTEND_REFLECT;
    grad.has_spread = true;
  }
  else if (spread_str.eq_ascii_ci ("repeat"))
  {
    grad.spread = HB_PAINT_EXTEND_REPEAT;
    grad.has_spread = true;
  }
  else if (spread_str.eq_ascii_ci ("pad"))
  {
    grad.spread = HB_PAINT_EXTEND_PAD;
    grad.has_spread = true;
  }

  hb_svg_str_t units_str = svg_pick_attr_or_style (parser, style_props.gradient_units, "gradientUnits").trim ();
  if (units_str.eq_ascii_ci ("userSpaceOnUse"))
  {
    grad.units_user_space = true;
    grad.has_units_user_space = true;
  }
  else if (units_str.eq_ascii_ci ("objectBoundingBox"))
  {
    grad.units_user_space = false;
    grad.has_units_user_space = true;
  }

  hb_svg_str_t transform_str = svg_pick_attr_or_style (parser, style_props.gradient_transform, "gradientTransform");
  if (transform_str.len)
  {
    grad.has_gradient_transform = true;
    hb_raster_svg_parse_transform (transform_str, &grad.gradient_transform);
  }

  hb_svg_str_t href = hb_raster_svg_find_href_attr (parser);
  if (href.len)
  {
    hb_svg_str_t href_id;
    if (hb_raster_svg_parse_local_id_ref (href, &href_id, nullptr))
      grad.href_id = hb_bytes_t (href_id.data, href_id.len);
  }
}

static void
svg_parse_gradient_geometry_attrs (hb_svg_xml_parser_t &parser,
                                   hb_svg_gradient_t &grad)
{
  hb_svg_style_props_t style_props;
  svg_parse_style_props (parser.find_attr ("style"), &style_props);
  if (grad.type == SVG_GRADIENT_LINEAR)
  {
    hb_svg_str_t x1_str = svg_pick_attr_or_style (parser, style_props.x1, "x1");
    hb_svg_str_t y1_str = svg_pick_attr_or_style (parser, style_props.y1, "y1");
    hb_svg_str_t x2_str = svg_pick_attr_or_style (parser, style_props.x2, "x2");
    hb_svg_str_t y2_str = svg_pick_attr_or_style (parser, style_props.y2, "y2");
    if (x1_str.len) { grad.x1 = svg_parse_number_or_percent (x1_str, nullptr); grad.has_x1 = true; }
    if (y1_str.len) { grad.y1 = svg_parse_number_or_percent (y1_str, nullptr); grad.has_y1 = true; }
    if (x2_str.len) { grad.x2 = svg_parse_number_or_percent (x2_str, nullptr); grad.has_x2 = true; }
    if (y2_str.len) { grad.y2 = svg_parse_number_or_percent (y2_str, nullptr); grad.has_y2 = true; }

    if (!grad.has_x2)
      grad.x2 = 1.f;
  }
  else
  {
    hb_svg_str_t cx_str = svg_pick_attr_or_style (parser, style_props.cx, "cx");
    hb_svg_str_t cy_str = svg_pick_attr_or_style (parser, style_props.cy, "cy");
    hb_svg_str_t r_str = svg_pick_attr_or_style (parser, style_props.r, "r");
    hb_svg_str_t fx_str = svg_pick_attr_or_style (parser, style_props.fx, "fx");
    hb_svg_str_t fy_str = svg_pick_attr_or_style (parser, style_props.fy, "fy");
    hb_svg_str_t fr_str = svg_pick_attr_or_style (parser, style_props.fr, "fr");

    if (cx_str.len) { grad.cx = svg_parse_number_or_percent (cx_str, nullptr); grad.has_cx = true; }
    if (cy_str.len) { grad.cy = svg_parse_number_or_percent (cy_str, nullptr); grad.has_cy = true; }
    if (r_str.len) { grad.r = svg_parse_number_or_percent (r_str, nullptr); grad.has_r = true; }
    if (fx_str.len) { grad.fx = svg_parse_number_or_percent (fx_str, nullptr); grad.has_fx = true; }
    if (fy_str.len) { grad.fy = svg_parse_number_or_percent (fy_str, nullptr); grad.has_fy = true; }
    if (fr_str.len) { grad.fr = svg_parse_number_or_percent (fr_str, nullptr); grad.has_fr = true; }
  }
}

static void
svg_parse_gradient_children (hb_svg_defs_t *defs,
                             hb_svg_xml_parser_t &parser,
                             hb_svg_gradient_t &grad,
                             hb_svg_str_t *id,
                             hb_paint_funcs_t *pfuncs,
                             void *paint_data,
                             hb_color_t foreground,
                             hb_face_t *face,
                             unsigned palette)
{
  int gdepth = 1;
  bool had_alloc_failure = false;
  while (gdepth > 0)
  {
    hb_svg_token_type_t gt = parser.next ();
    if (gt == SVG_TOKEN_EOF) break;
    if (gt == SVG_TOKEN_CLOSE_TAG) { gdepth--; continue; }
    if ((gt == SVG_TOKEN_OPEN_TAG || gt == SVG_TOKEN_SELF_CLOSE_TAG) &&
        parser.tag_name.eq ("stop"))
      if (unlikely (!svg_parse_gradient_stop (parser, grad,
                                              pfuncs, paint_data,
                                              foreground, face,
                                              palette)))
        had_alloc_failure = true;
    if (gt == SVG_TOKEN_OPEN_TAG && !parser.tag_name.eq ("stop"))
      gdepth++;
  }
  if (had_alloc_failure || defs->gradients.in_error ())
    *id = {};
}

void
hb_raster_svg_process_gradient_def (hb_svg_defs_t *defs,
                          hb_svg_xml_parser_t &parser,
                          hb_svg_token_type_t tok,
                          hb_svg_gradient_type_t type,
                          hb_paint_funcs_t *pfuncs,
                          void *paint_data,
                          hb_color_t foreground,
                          hb_face_t *face,
                          unsigned palette)
{
  hb_svg_gradient_t grad;
  grad.type = type;
  svg_parse_gradient_geometry_attrs (parser, grad);
  svg_parse_gradient_attrs (parser, grad);

  hb_svg_str_t id = parser.find_attr ("id");
  if (tok == SVG_TOKEN_OPEN_TAG)
    svg_parse_gradient_children (defs, parser, grad, &id,
                                 pfuncs, paint_data,
                                 foreground, face,
                                 palette);

  if (id.len)
    (void) defs->add_gradient (hb_bytes_t (id.data, id.len), grad);
}

#endif /* !HB_NO_RASTER_SVG */
