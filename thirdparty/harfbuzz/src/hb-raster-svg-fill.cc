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

#include "hb-raster-svg-fill.hh"

#include "hb-raster-svg-base.hh"
#include "hb-raster-svg-color.hh"
#include "hb-decycler.hh"

#include <math.h>

struct hb_svg_color_line_data_t
{
  const hb_svg_gradient_t *grad;
  hb_paint_extend_t extend;
  float alpha_scale;
  hb_color_t current_color;
};

static unsigned
svg_color_line_get_stops (hb_color_line_t *color_line HB_UNUSED,
                          void *color_line_data,
                          unsigned int start,
                          unsigned int *count,
                          hb_color_stop_t *color_stops,
                          void *user_data HB_UNUSED)
{
  hb_svg_color_line_data_t *cl = (hb_svg_color_line_data_t *) color_line_data;
  const hb_svg_gradient_t *grad = cl->grad;
  unsigned total = grad->stops.length;

  if (count)
  {
    unsigned n = hb_min (*count, total > start ? total - start : 0u);
    for (unsigned i = 0; i < n; i++)
    {
      const hb_svg_gradient_stop_t &stop = grad->stops[start + i];
      color_stops[i].offset = stop.offset;
      color_stops[i].is_foreground = false;
      hb_color_t c = stop.is_current_color ? cl->current_color : stop.color;
      if (stop.is_current_color)
      {
        uint8_t a = (uint8_t) hb_raster_div255 (hb_color_get_alpha (c) * hb_color_get_alpha (stop.color));
        c = HB_COLOR (hb_color_get_blue (c),
                      hb_color_get_green (c),
                      hb_color_get_red (c),
                      a);
      }
      if (cl->alpha_scale < 1.f)
      {
        uint8_t a = (uint8_t) hb_clamp ((int) (hb_color_get_alpha (c) * cl->alpha_scale + 0.5f), 0, 255);
        c = HB_COLOR (hb_color_get_blue (c),
                      hb_color_get_green (c),
                      hb_color_get_red (c),
                      a);
      }
      color_stops[i].color = c;
    }
    *count = n;
  }
  return total;
}

static hb_paint_extend_t
svg_color_line_get_extend (hb_color_line_t *color_line HB_UNUSED,
                           void *color_line_data,
                           void *user_data HB_UNUSED)
{
  hb_svg_color_line_data_t *cl = (hb_svg_color_line_data_t *) color_line_data;
  return cl->extend;
}

static bool
svg_parse_paint_url_with_fallback (hb_svg_str_t s,
                                   hb_svg_str_t *out_id,
                                   hb_svg_str_t *fallback)
{
  if (out_id) *out_id = {};
  hb_svg_str_t id;
  if (!hb_raster_svg_parse_local_id_ref (s, &id, fallback))
    return false;
  if (!s.trim ().starts_with_ascii_ci ("url("))
    return false;
  if (out_id) *out_id = id;
  return id.len > 0;
}

void
hb_raster_svg_emit_fill (const hb_svg_fill_context_t *ctx,
               hb_svg_str_t fill_str,
               float fill_opacity,
               const hb_extents_t<> *object_bbox,
               hb_color_t current_color)
{
  bool is_none = false;

  hb_svg_str_t url_id;
  hb_svg_str_t fallback_paint;
  bool has_url_paint = svg_parse_paint_url_with_fallback (fill_str, &url_id, &fallback_paint);

  const hb_svg_gradient_t *grad = has_url_paint ? ctx->defs->find_gradient (hb_bytes_t (url_id.data, url_id.len)) : nullptr;
  if (has_url_paint && !grad)
  {
    if (fallback_paint.len)
      hb_raster_svg_emit_fill (ctx, fallback_paint, fill_opacity, object_bbox, current_color);
    return;
  }
  if (grad)
  {
    const unsigned SVG_MAX_GRADIENT_REF_DEPTH = 1024;
    hb_vector_t<const hb_svg_gradient_t *> chain;
    hb_decycler_t decycler;
    const hb_svg_gradient_t *cur = grad;
    while (cur)
    {
      if (chain.length >= SVG_MAX_GRADIENT_REF_DEPTH)
        break;

      hb_decycler_node_t node (decycler);
      if (unlikely (!node.visit ((uintptr_t) cur)))
        break;

      chain.push (cur);
      if (unlikely (chain.in_error ()))
      {
        if (fallback_paint.len)
          hb_raster_svg_emit_fill (ctx, fallback_paint, fill_opacity, object_bbox, current_color);
        return;
      }

      if (!cur->href_id.length) break;
      const hb_svg_gradient_t *next = ctx->defs->find_gradient (cur->href_id);
      if (!next) break;
      cur = next;
    }
    if (!chain.length)
    {
      if (fallback_paint.len)
        hb_raster_svg_emit_fill (ctx, fallback_paint, fill_opacity, object_bbox, current_color);
      return;
    }

    if (!chain.length) {
      if (fallback_paint.len)
        hb_raster_svg_emit_fill (ctx, fallback_paint, fill_opacity, object_bbox, current_color);
      return;
    }
    hb_svg_gradient_t effective = *chain.arrayZ[chain.length - 1];
    for (int i = (int) chain.length - 2; i >= 0; i--)
    {
      const hb_svg_gradient_t *g = chain.arrayZ[i];
      effective.type = g->type;
      if (g->stops.length) effective.stops = g->stops;
      if (g->has_spread) { effective.spread = g->spread; effective.has_spread = true; }
      if (g->has_units_user_space)
      {
        effective.units_user_space = g->units_user_space;
        effective.has_units_user_space = true;
      }
      if (g->has_gradient_transform)
      {
        effective.gradient_transform = g->gradient_transform;
        effective.has_gradient_transform = true;
      }
      if (g->has_x1) { effective.x1 = g->x1; effective.has_x1 = true; }
      if (g->has_y1) { effective.y1 = g->y1; effective.has_y1 = true; }
      if (g->has_x2) { effective.x2 = g->x2; effective.has_x2 = true; }
      if (g->has_y2) { effective.y2 = g->y2; effective.has_y2 = true; }
      if (g->has_cx) { effective.cx = g->cx; effective.has_cx = true; }
      if (g->has_cy) { effective.cy = g->cy; effective.has_cy = true; }
      if (g->has_r)  { effective.r  = g->r;  effective.has_r  = true; }
      if (g->has_fx) { effective.fx = g->fx; effective.has_fx = true; }
      if (g->has_fy) { effective.fy = g->fy; effective.has_fy = true; }
      if (g->has_fr) { effective.fr = g->fr; effective.has_fr = true; }
    }
    if (!effective.stops.length)
    {
      if (fallback_paint.len)
        hb_raster_svg_emit_fill (ctx, fallback_paint, fill_opacity, object_bbox, current_color);
      return;
    }

    hb_svg_color_line_data_t cl_data;
    cl_data.grad = &effective;
    cl_data.extend = effective.spread;
    cl_data.alpha_scale = hb_clamp (fill_opacity, 0.f, 1.f);
    cl_data.current_color = current_color;

    hb_color_line_t cl = {
      &cl_data,
      svg_color_line_get_stops, nullptr,
      svg_color_line_get_extend, nullptr
    };

    bool has_bbox_transform = !effective.units_user_space && object_bbox && !object_bbox->is_empty ();
    if (has_bbox_transform)
    {
      float w = object_bbox->xmax - object_bbox->xmin;
      float h = object_bbox->ymax - object_bbox->ymin;
      if (isfinite (w) && isfinite (h) && w > 0.f && h > 0.f)
        hb_paint_push_transform (ctx->pfuncs, ctx->paint, w, 0, 0, h, object_bbox->xmin, object_bbox->ymin);
      else
        has_bbox_transform = false;
    }

    if (effective.has_gradient_transform)
      hb_paint_push_transform (ctx->pfuncs, ctx->paint,
                               effective.gradient_transform.xx,
                               effective.gradient_transform.yx,
                               effective.gradient_transform.xy,
                               effective.gradient_transform.yy,
                               effective.gradient_transform.dx,
                               effective.gradient_transform.dy);

    if (effective.type == SVG_GRADIENT_LINEAR)
    {
      hb_paint_linear_gradient (ctx->pfuncs, ctx->paint, &cl,
                                effective.x1, effective.y1,
                                effective.x2, effective.y2,
                                effective.x2 - (effective.y2 - effective.y1),
                                effective.y2 + (effective.x2 - effective.x1));
    }
    else
    {
      if (!isfinite (effective.r) || effective.r < 0.f)
      {
        if (effective.has_gradient_transform)
          hb_paint_pop_transform (ctx->pfuncs, ctx->paint);
        if (has_bbox_transform)
          hb_paint_pop_transform (ctx->pfuncs, ctx->paint);
        if (fallback_paint.len)
          hb_raster_svg_emit_fill (ctx, fallback_paint, fill_opacity, object_bbox, current_color);
        return;
      }

      float fx = effective.has_fx ? effective.fx : effective.cx;
      float fy = effective.has_fy ? effective.fy : effective.cy;
      float fr = effective.has_fr ? effective.fr : 0.f;
      if (!isfinite (fr) || fr < 0.f)
        fr = 0.f;
      fr = hb_min (fr, effective.r);

      hb_paint_radial_gradient (ctx->pfuncs, ctx->paint, &cl,
                                fx, fy, fr,
                                effective.cx, effective.cy, effective.r);
    }

    if (effective.has_gradient_transform)
      hb_paint_pop_transform (ctx->pfuncs, ctx->paint);
    if (has_bbox_transform)
      hb_paint_pop_transform (ctx->pfuncs, ctx->paint);

    return;
  }

  hb_color_t color = hb_raster_svg_parse_color (fill_str,
                                      ctx->pfuncs,
                                      ctx->paint,
                                      current_color,
                                      hb_font_get_face (ctx->font),
                                      ctx->palette,
                                      &is_none);
  if (is_none) return;

  if (fill_opacity < 1.f)
    color = HB_COLOR (hb_color_get_blue (color),
                      hb_color_get_green (color),
                      hb_color_get_red (color),
                      (uint8_t) (hb_color_get_alpha (color) * fill_opacity + 0.5f));

  hb_paint_color (ctx->pfuncs, ctx->paint, false, color);
}

#endif /* !HB_NO_RASTER_SVG */
