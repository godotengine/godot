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

#include "hb-raster-svg-parse.hh"

#include <math.h>
bool
hb_raster_svg_parse_transform (hb_svg_str_t s, hb_svg_transform_t *out)
{
  hb_svg_float_parser_t fp (s);

  while (fp.p < fp.end)
  {
    fp.skip_ws_comma ();
    if (fp.p >= fp.end) break;

    const char *start = fp.p;
    while (fp.p < fp.end && *fp.p != '(') fp.p++;
    hb_svg_str_t func_name = {start, (unsigned) (fp.p - start)};

    while (func_name.len && (func_name.data[func_name.len - 1] == ' ' ||
                             func_name.data[func_name.len - 1] == '\t'))
      func_name.len--;

    if (fp.p >= fp.end) break;
    fp.p++;

    hb_svg_transform_t t;

    if (func_name.eq_ascii_ci ("matrix"))
    {
      t.xx = fp.next_float ();
      t.yx = fp.next_float ();
      t.xy = fp.next_float ();
      t.yy = fp.next_float ();
      t.dx = fp.next_float ();
      t.dy = fp.next_float ();
    }
    else if (func_name.eq_ascii_ci ("translate"))
    {
      t.dx = fp.next_float ();
      fp.skip_ws_comma ();
      if (fp.p < fp.end && *fp.p != ')')
        t.dy = fp.next_float ();
    }
    else if (func_name.eq_ascii_ci ("scale"))
    {
      t.xx = fp.next_float ();
      fp.skip_ws_comma ();
      if (fp.p < fp.end && *fp.p != ')')
        t.yy = fp.next_float ();
      else
        t.yy = t.xx;
    }
    else if (func_name.eq_ascii_ci ("rotate"))
    {
      float angle = fp.next_float () * (float) M_PI / 180.f;
      float cs = cosf (angle), sn = sinf (angle);
      fp.skip_ws_comma ();
      if (fp.p < fp.end && *fp.p != ')')
      {
        float cx = fp.next_float ();
        float cy = fp.next_float ();
        t.xx = cs;  t.yx = sn;
        t.xy = -sn; t.yy = cs;
        t.dx = cx - cs * cx + sn * cy;
        t.dy = cy - sn * cx - cs * cy;
      }
      else
      {
        t.xx = cs;  t.yx = sn;
        t.xy = -sn; t.yy = cs;
      }
    }
    else if (func_name.eq_ascii_ci ("skewX"))
    {
      float angle = fp.next_float () * (float) M_PI / 180.f;
      t.xy = tanf (angle);
    }
    else if (func_name.eq_ascii_ci ("skewY"))
    {
      float angle = fp.next_float () * (float) M_PI / 180.f;
      t.yx = tanf (angle);
    }

    while (fp.p < fp.end && *fp.p != ')') fp.p++;
    if (fp.p < fp.end) fp.p++;

    out->multiply (t);
  }
  return true;
}

static void
svg_arc_to_cubics (hb_draw_funcs_t *dfuncs, void *draw_data, hb_draw_state_t *st,
                   float cx, float cy,
                   float rx, float ry,
                   float phi, float theta1, float dtheta)
{
  int n_segs = (int) ceilf (fabsf (dtheta) / ((float) M_PI / 2.f));
  if (n_segs < 1) n_segs = 1;
  float seg_angle = dtheta / n_segs;

  float cos_phi = cosf (phi);
  float sin_phi = sinf (phi);

  for (int i = 0; i < n_segs; i++)
  {
    float t1 = theta1 + i * seg_angle;
    float t2 = t1 + seg_angle;
    float alpha = sinf (seg_angle) *
                  (sqrtf (4.f + 3.f * tanf (seg_angle / 2.f) * tanf (seg_angle / 2.f)) - 1.f) / 3.f;

    float cos_t1 = cosf (t1), sin_t1 = sinf (t1);
    float cos_t2 = cosf (t2), sin_t2 = sinf (t2);

    float e1x = rx * cos_t1, e1y = ry * sin_t1;
    float e2x = rx * cos_t2, e2y = ry * sin_t2;
    float d1x = -rx * sin_t1, d1y = ry * cos_t1;
    float d2x = -rx * sin_t2, d2y = ry * cos_t2;

    float cp1x = e1x + alpha * d1x;
    float cp1y = e1y + alpha * d1y;
    float cp2x = e2x - alpha * d2x;
    float cp2y = e2y - alpha * d2y;

    float r_cp1x = cos_phi * cp1x - sin_phi * cp1y + cx;
    float r_cp1y = sin_phi * cp1x + cos_phi * cp1y + cy;
    float r_cp2x = cos_phi * cp2x - sin_phi * cp2y + cx;
    float r_cp2y = sin_phi * cp2x + cos_phi * cp2y + cy;
    float r_e2x  = cos_phi * e2x  - sin_phi * e2y  + cx;
    float r_e2y  = sin_phi * e2x  + cos_phi * e2y  + cy;

    hb_draw_cubic_to (dfuncs, draw_data, st,
                      r_cp1x, r_cp1y,
                      r_cp2x, r_cp2y,
                      r_e2x, r_e2y);
  }
}

static void
svg_arc_endpoint_to_center (float x1, float y1, float x2, float y2,
                            float rx, float ry, float phi_deg,
                            bool large_arc, bool sweep,
                            float *cx_out, float *cy_out,
                            float *theta1_out, float *dtheta_out,
                            float *rx_out, float *ry_out)
{
  float phi = phi_deg * (float) M_PI / 180.f;
  float cos_phi = cosf (phi), sin_phi = sinf (phi);

  float mx = (x1 - x2) / 2.f;
  float my = (y1 - y2) / 2.f;
  float x1p =  cos_phi * mx + sin_phi * my;
  float y1p = -sin_phi * mx + cos_phi * my;

  rx = fabsf (rx); ry = fabsf (ry);
  if (rx < 1e-10f || ry < 1e-10f)
  {
    *cx_out = (x1 + x2) / 2.f;
    *cy_out = (y1 + y2) / 2.f;
    *theta1_out = 0;
    *dtheta_out = 0;
    *rx_out = rx; *ry_out = ry;
    return;
  }

  float x1p2 = x1p * x1p, y1p2 = y1p * y1p;
  float rx2 = rx * rx, ry2 = ry * ry;
  float lambda = x1p2 / rx2 + y1p2 / ry2;
  if (lambda > 1.f)
  {
    float sl = sqrtf (lambda);
    rx *= sl; ry *= sl;
    rx2 = rx * rx; ry2 = ry * ry;
  }

  float num = rx2 * ry2 - rx2 * y1p2 - ry2 * x1p2;
  float den = rx2 * y1p2 + ry2 * x1p2;
  float sq = (den > 0.f) ? sqrtf (hb_max (num / den, 0.f)) : 0.f;
  if (large_arc == sweep) sq = -sq;

  float cxp =  sq * rx * y1p / ry;
  float cyp = -sq * ry * x1p / rx;

  float cx = cos_phi * cxp - sin_phi * cyp + (x1 + x2) / 2.f;
  float cy = sin_phi * cxp + cos_phi * cyp + (y1 + y2) / 2.f;

  auto angle = [] (float ux, float uy, float vx, float vy) -> float {
    float dot = ux * vx + uy * vy;
    float len = sqrtf ((ux * ux + uy * uy) * (vx * vx + vy * vy));
    if (!(len > 0.f) || !isfinite (len))
      return 0.f;
    float a = acosf (hb_clamp (dot / len, -1.f, 1.f));
    if (ux * vy - uy * vx < 0.f) a = -a;
    return a;
  };

  float theta1 = angle (1.f, 0.f, (x1p - cxp) / rx, (y1p - cyp) / ry);
  float dtheta = angle ((x1p - cxp) / rx, (y1p - cyp) / ry,
                        (-x1p - cxp) / rx, (-y1p - cyp) / ry);

  if (!sweep && dtheta > 0.f) dtheta -= 2.f * (float) M_PI;
  if (sweep && dtheta < 0.f)  dtheta += 2.f * (float) M_PI;

  *cx_out = cx; *cy_out = cy;
  *theta1_out = theta1; *dtheta_out = dtheta;
  *rx_out = rx; *ry_out = ry;
}

void
hb_raster_svg_parse_path_data (hb_svg_str_t d, hb_draw_funcs_t *dfuncs, void *draw_data)
{
  hb_draw_state_t st = HB_DRAW_STATE_DEFAULT;
  hb_svg_float_parser_t fp (d);

  float cur_x = 0, cur_y = 0;
  float start_x = 0, start_y = 0;
  float last_cx = 0, last_cy = 0;
  char last_cmd = 0;
  char cmd = 0;

  while (fp.p < fp.end)
  {
    fp.skip_ws_comma ();
    if (fp.p >= fp.end) break;

    char c = *fp.p;
    if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'))
    {
      cmd = c;
      fp.p++;
    }

    switch (cmd)
    {
    case 'M': case 'm':
    {
      float x = fp.next_float ();
      float y = fp.next_float ();
      if (cmd == 'm') { x += cur_x; y += cur_y; }
      hb_draw_move_to (dfuncs, draw_data, &st, x, y);
      cur_x = start_x = x;
      cur_y = start_y = y;
      cmd = (cmd == 'M') ? 'L' : 'l';
      last_cmd = 'M';
      continue;
    }
    case 'L': case 'l':
    {
      float x = fp.next_float ();
      float y = fp.next_float ();
      if (cmd == 'l') { x += cur_x; y += cur_y; }
      hb_draw_line_to (dfuncs, draw_data, &st, x, y);
      cur_x = x; cur_y = y;
      break;
    }
    case 'H': case 'h':
    {
      float x = fp.next_float ();
      if (cmd == 'h') x += cur_x;
      hb_draw_line_to (dfuncs, draw_data, &st, x, cur_y);
      cur_x = x;
      break;
    }
    case 'V': case 'v':
    {
      float y = fp.next_float ();
      if (cmd == 'v') y += cur_y;
      hb_draw_line_to (dfuncs, draw_data, &st, cur_x, y);
      cur_y = y;
      break;
    }
    case 'C': case 'c':
    {
      float x1 = fp.next_float ();
      float y1 = fp.next_float ();
      float x2 = fp.next_float ();
      float y2 = fp.next_float ();
      float x  = fp.next_float ();
      float y  = fp.next_float ();
      if (cmd == 'c')
      { x1 += cur_x; y1 += cur_y; x2 += cur_x; y2 += cur_y; x += cur_x; y += cur_y; }
      hb_draw_cubic_to (dfuncs, draw_data, &st, x1, y1, x2, y2, x, y);
      last_cx = x2; last_cy = y2;
      cur_x = x; cur_y = y;
      break;
    }
    case 'S': case 's':
    {
      float cx1, cy1;
      if (last_cmd == 'C' || last_cmd == 'c' || last_cmd == 'S' || last_cmd == 's')
      { cx1 = 2 * cur_x - last_cx; cy1 = 2 * cur_y - last_cy; }
      else
      { cx1 = cur_x; cy1 = cur_y; }
      float x2 = fp.next_float ();
      float y2 = fp.next_float ();
      float x  = fp.next_float ();
      float y  = fp.next_float ();
      if (cmd == 's')
      { x2 += cur_x; y2 += cur_y; x += cur_x; y += cur_y; }
      hb_draw_cubic_to (dfuncs, draw_data, &st, cx1, cy1, x2, y2, x, y);
      last_cx = x2; last_cy = y2;
      cur_x = x; cur_y = y;
      break;
    }
    case 'Q': case 'q':
    {
      float x1 = fp.next_float ();
      float y1 = fp.next_float ();
      float x  = fp.next_float ();
      float y  = fp.next_float ();
      if (cmd == 'q')
      { x1 += cur_x; y1 += cur_y; x += cur_x; y += cur_y; }
      hb_draw_quadratic_to (dfuncs, draw_data, &st, x1, y1, x, y);
      last_cx = x1; last_cy = y1;
      cur_x = x; cur_y = y;
      break;
    }
    case 'T': case 't':
    {
      float cx1, cy1;
      if (last_cmd == 'Q' || last_cmd == 'q' || last_cmd == 'T' || last_cmd == 't')
      { cx1 = 2 * cur_x - last_cx; cy1 = 2 * cur_y - last_cy; }
      else
      { cx1 = cur_x; cy1 = cur_y; }
      float x = fp.next_float ();
      float y = fp.next_float ();
      if (cmd == 't') { x += cur_x; y += cur_y; }
      hb_draw_quadratic_to (dfuncs, draw_data, &st, cx1, cy1, x, y);
      last_cx = cx1; last_cy = cy1;
      cur_x = x; cur_y = y;
      break;
    }
    case 'A': case 'a':
    {
      float rx = fp.next_float ();
      float ry = fp.next_float ();
      float x_rot = fp.next_float ();
      bool large_arc = fp.next_flag ();
      bool sweep = fp.next_flag ();
      float x = fp.next_float ();
      float y = fp.next_float ();
      if (cmd == 'a') { x += cur_x; y += cur_y; }

      if (fabsf (x - cur_x) < 1e-6f && fabsf (y - cur_y) < 1e-6f)
      {
        cur_x = x; cur_y = y;
        break;
      }

      float cx, cy, theta1, dtheta, adj_rx, adj_ry;
      svg_arc_endpoint_to_center (cur_x, cur_y, x, y,
                                  rx, ry, x_rot,
                                  large_arc, sweep,
                                  &cx, &cy, &theta1, &dtheta,
                                  &adj_rx, &adj_ry);

      float phi = x_rot * (float) M_PI / 180.f;
      svg_arc_to_cubics (dfuncs, draw_data, &st,
                         cx, cy, adj_rx, adj_ry, phi, theta1, dtheta);
      cur_x = x; cur_y = y;
      break;
    }
    case 'Z': case 'z':
      hb_draw_close_path (dfuncs, draw_data, &st);
      cur_x = start_x;
      cur_y = start_y;
      break;

    default:
      fp.p++;
      continue;
    }

    last_cmd = cmd;
  }
}

static void
svg_rect_to_path (float x, float y, float w, float h, float rx, float ry,
                  hb_draw_funcs_t *dfuncs, void *draw_data)
{
  hb_draw_state_t st = HB_DRAW_STATE_DEFAULT;
  if (rx < 0.f || ry < 0.f)
    return;

  if (rx <= 0 && ry <= 0)
  {
    hb_draw_move_to (dfuncs, draw_data, &st, x, y);
    hb_draw_line_to (dfuncs, draw_data, &st, x + w, y);
    hb_draw_line_to (dfuncs, draw_data, &st, x + w, y + h);
    hb_draw_line_to (dfuncs, draw_data, &st, x, y + h);
    hb_draw_close_path (dfuncs, draw_data, &st);
    return;
  }

  if (rx <= 0) rx = ry;
  if (ry <= 0) ry = rx;
  rx = hb_min (rx, w / 2);
  ry = hb_min (ry, h / 2);

  float kx = rx * 0.5522847498f;
  float ky = ry * 0.5522847498f;

  hb_draw_move_to (dfuncs, draw_data, &st, x + rx, y);
  hb_draw_line_to (dfuncs, draw_data, &st, x + w - rx, y);
  hb_draw_cubic_to (dfuncs, draw_data, &st,
                    x + w - rx + kx, y,
                    x + w, y + ry - ky,
                    x + w, y + ry);
  hb_draw_line_to (dfuncs, draw_data, &st, x + w, y + h - ry);
  hb_draw_cubic_to (dfuncs, draw_data, &st,
                    x + w, y + h - ry + ky,
                    x + w - rx + kx, y + h,
                    x + w - rx, y + h);
  hb_draw_line_to (dfuncs, draw_data, &st, x + rx, y + h);
  hb_draw_cubic_to (dfuncs, draw_data, &st,
                    x + rx - kx, y + h,
                    x, y + h - ry + ky,
                    x, y + h - ry);
  hb_draw_line_to (dfuncs, draw_data, &st, x, y + ry);
  hb_draw_cubic_to (dfuncs, draw_data, &st,
                    x, y + ry - ky,
                    x + rx - kx, y,
                    x + rx, y);
  hb_draw_close_path (dfuncs, draw_data, &st);
}

static void
svg_circle_to_path (float cx, float cy, float r,
                    hb_draw_funcs_t *dfuncs, void *draw_data)
{
  hb_draw_state_t st = HB_DRAW_STATE_DEFAULT;
  float k = r * 0.5522847498f;

  hb_draw_move_to (dfuncs, draw_data, &st, cx + r, cy);
  hb_draw_cubic_to (dfuncs, draw_data, &st,
                    cx + r, cy + k,
                    cx + k, cy + r,
                    cx, cy + r);
  hb_draw_cubic_to (dfuncs, draw_data, &st,
                    cx - k, cy + r,
                    cx - r, cy + k,
                    cx - r, cy);
  hb_draw_cubic_to (dfuncs, draw_data, &st,
                    cx - r, cy - k,
                    cx - k, cy - r,
                    cx, cy - r);
  hb_draw_cubic_to (dfuncs, draw_data, &st,
                    cx + k, cy - r,
                    cx + r, cy - k,
                    cx + r, cy);
  hb_draw_close_path (dfuncs, draw_data, &st);
}

static void
svg_ellipse_to_path (float cx, float cy, float rx, float ry,
                     hb_draw_funcs_t *dfuncs, void *draw_data)
{
  hb_draw_state_t st = HB_DRAW_STATE_DEFAULT;
  float kx = rx * 0.5522847498f;
  float ky = ry * 0.5522847498f;

  hb_draw_move_to (dfuncs, draw_data, &st, cx + rx, cy);
  hb_draw_cubic_to (dfuncs, draw_data, &st,
                    cx + rx, cy + ky,
                    cx + kx, cy + ry,
                    cx, cy + ry);
  hb_draw_cubic_to (dfuncs, draw_data, &st,
                    cx - kx, cy + ry,
                    cx - rx, cy + ky,
                    cx - rx, cy);
  hb_draw_cubic_to (dfuncs, draw_data, &st,
                    cx - rx, cy - ky,
                    cx - kx, cy - ry,
                    cx, cy - ry);
  hb_draw_cubic_to (dfuncs, draw_data, &st,
                    cx + kx, cy - ry,
                    cx + rx, cy - ky,
                    cx + rx, cy);
  hb_draw_close_path (dfuncs, draw_data, &st);
}

static void
svg_line_to_path (float x1, float y1, float x2, float y2,
                  hb_draw_funcs_t *dfuncs, void *draw_data)
{
  hb_draw_state_t st = HB_DRAW_STATE_DEFAULT;
  hb_draw_move_to (dfuncs, draw_data, &st, x1, y1);
  hb_draw_line_to (dfuncs, draw_data, &st, x2, y2);
}

static void
svg_polygon_to_path (hb_svg_str_t points, bool close,
                     hb_draw_funcs_t *dfuncs, void *draw_data)
{
  hb_draw_state_t st = HB_DRAW_STATE_DEFAULT;
  hb_svg_float_parser_t fp (points);
  bool first = true;
  while (fp.has_more ())
  {
    float x = fp.next_float ();
    float y = fp.next_float ();
    if (first)
    {
      hb_draw_move_to (dfuncs, draw_data, &st, x, y);
      first = false;
    }
    else
      hb_draw_line_to (dfuncs, draw_data, &st, x, y);
  }
  if (close && !first)
    hb_draw_close_path (dfuncs, draw_data, &st);
}

void
hb_raster_svg_shape_path_emit (hb_draw_funcs_t *dfuncs, void *draw_data, void *user_data)
{
  hb_svg_shape_emit_data_t *shape = (hb_svg_shape_emit_data_t *) user_data;
  switch (shape->type)
  {
  case hb_svg_shape_emit_data_t::SHAPE_PATH:
    hb_raster_svg_parse_path_data (shape->str_data, dfuncs, draw_data);
    break;
  case hb_svg_shape_emit_data_t::SHAPE_RECT:
    svg_rect_to_path (shape->params[0], shape->params[1],
                      shape->params[2], shape->params[3],
                      shape->params[4], shape->params[5],
                      dfuncs, draw_data);
    break;
  case hb_svg_shape_emit_data_t::SHAPE_CIRCLE:
    svg_circle_to_path (shape->params[0], shape->params[1], shape->params[2],
                        dfuncs, draw_data);
    break;
  case hb_svg_shape_emit_data_t::SHAPE_ELLIPSE:
    svg_ellipse_to_path (shape->params[0], shape->params[1],
                         shape->params[2], shape->params[3],
                         dfuncs, draw_data);
    break;
  case hb_svg_shape_emit_data_t::SHAPE_LINE:
    svg_line_to_path (shape->params[0], shape->params[1],
                      shape->params[2], shape->params[3],
                      dfuncs, draw_data);
    break;
  case hb_svg_shape_emit_data_t::SHAPE_POLYLINE:
    svg_polygon_to_path (shape->str_data, false, dfuncs, draw_data);
    break;
  case hb_svg_shape_emit_data_t::SHAPE_POLYGON:
    svg_polygon_to_path (shape->str_data, true, dfuncs, draw_data);
    break;
  }
}

bool
hb_raster_svg_parse_shape_tag (hb_svg_xml_parser_t &parser,
                     hb_svg_shape_emit_data_t *shape)
{
  hb_svg_attr_view_t attrs (parser);
  hb_svg_style_props_t style_props;
  svg_parse_style_props (attrs.get ("style"), &style_props);
  hb_svg_str_t tag = parser.tag_name;
  if (tag.eq ("path"))
  {
    hb_svg_str_t d = svg_pick_attr_or_style (parser, style_props.d, "d");
    if (!d.len) return false;
    shape->type = hb_svg_shape_emit_data_t::SHAPE_PATH;
    shape->str_data = d;
    return true;
  }
  if (tag.eq ("rect"))
  {
    float w = svg_parse_float (svg_pick_attr_or_style (parser, style_props.width, "width"));
    float h = svg_parse_float (svg_pick_attr_or_style (parser, style_props.height, "height"));
    if (w <= 0 || h <= 0) return false;
    float rx = svg_parse_float (svg_pick_attr_or_style (parser, style_props.rx, "rx"));
    float ry = svg_parse_float (svg_pick_attr_or_style (parser, style_props.ry, "ry"));
    if (rx < 0.f || ry < 0.f) return false;
    shape->type = hb_svg_shape_emit_data_t::SHAPE_RECT;
    shape->params[0] = svg_parse_float (svg_pick_attr_or_style (parser, style_props.x, "x"));
    shape->params[1] = svg_parse_float (svg_pick_attr_or_style (parser, style_props.y, "y"));
    shape->params[2] = w;
    shape->params[3] = h;
    shape->params[4] = rx;
    shape->params[5] = ry;
    return true;
  }
  if (tag.eq ("circle"))
  {
    float r = svg_parse_float (svg_pick_attr_or_style (parser, style_props.r, "r"));
    if (r <= 0) return false;
    shape->type = hb_svg_shape_emit_data_t::SHAPE_CIRCLE;
    shape->params[0] = svg_parse_float (svg_pick_attr_or_style (parser, style_props.cx, "cx"));
    shape->params[1] = svg_parse_float (svg_pick_attr_or_style (parser, style_props.cy, "cy"));
    shape->params[2] = r;
    return true;
  }
  if (tag.eq ("ellipse"))
  {
    float rx = svg_parse_float (svg_pick_attr_or_style (parser, style_props.rx, "rx"));
    float ry = svg_parse_float (svg_pick_attr_or_style (parser, style_props.ry, "ry"));
    if (rx <= 0 || ry <= 0) return false;
    shape->type = hb_svg_shape_emit_data_t::SHAPE_ELLIPSE;
    shape->params[0] = svg_parse_float (svg_pick_attr_or_style (parser, style_props.cx, "cx"));
    shape->params[1] = svg_parse_float (svg_pick_attr_or_style (parser, style_props.cy, "cy"));
    shape->params[2] = rx;
    shape->params[3] = ry;
    return true;
  }
  if (tag.eq ("line"))
  {
    shape->type = hb_svg_shape_emit_data_t::SHAPE_LINE;
    shape->params[0] = svg_parse_float (svg_pick_attr_or_style (parser, style_props.x1, "x1"));
    shape->params[1] = svg_parse_float (svg_pick_attr_or_style (parser, style_props.y1, "y1"));
    shape->params[2] = svg_parse_float (svg_pick_attr_or_style (parser, style_props.x2, "x2"));
    shape->params[3] = svg_parse_float (svg_pick_attr_or_style (parser, style_props.y2, "y2"));
    return true;
  }
  if (tag.eq ("polyline"))
  {
    hb_svg_str_t points = svg_pick_attr_or_style (parser, style_props.points, "points");
    if (!points.len) return false;
    shape->type = hb_svg_shape_emit_data_t::SHAPE_POLYLINE;
    shape->str_data = points;
    return true;
  }
  if (tag.eq ("polygon"))
  {
    hb_svg_str_t points = svg_pick_attr_or_style (parser, style_props.points, "points");
    if (!points.len) return false;
    shape->type = hb_svg_shape_emit_data_t::SHAPE_POLYGON;
    shape->str_data = points;
    return true;
  }
  return false;
}

#endif /* !HB_NO_RASTER_SVG */
