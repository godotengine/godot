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

#include "hb-raster-svg-clip.hh"

#include "hb-raster.h"
#include "hb-raster-paint.hh"
#include "hb-raster-svg.hh"
#include "hb-raster-svg-base.hh"
#include "hb-decycler.hh"

#include <math.h>

static inline bool
svg_transform_is_identity (const hb_svg_transform_t &t)
{
  return t.xx == 1.f && t.yx == 0.f &&
         t.xy == 0.f && t.yy == 1.f &&
         t.dx == 0.f && t.dy == 0.f;
}

static bool
svg_parse_element_transform (hb_svg_xml_parser_t &parser,
                             hb_svg_transform_t *out)
{
  hb_svg_style_props_t style_props;
  svg_parse_style_props (parser.find_attr ("style"), &style_props);
  hb_svg_str_t transform = svg_pick_attr_or_style (parser, style_props.transform, "transform");
  if (!transform.len)
    return false;
  hb_raster_svg_parse_transform (transform, out);
  return true;
}

struct hb_svg_clip_collect_context_t
{
  hb_svg_defs_t *defs;
  hb_svg_clip_path_def_t *clip;
  const char *doc_start;
  unsigned doc_len;
  const OT::SVG::accelerator_t *svg_accel;
  const OT::SVG::svg_doc_cache_t *doc_cache;
  hb_decycler_t *use_decycler;
  bool *had_alloc_failure;
};

static inline void
svg_clip_append_shape (hb_svg_clip_collect_context_t *ctx,
                       const hb_svg_shape_emit_data_t &shape,
                       const hb_svg_transform_t &transform)
{
  hb_svg_clip_shape_t clip_shape;
  clip_shape.shape = shape;
  if (!svg_transform_is_identity (transform))
  {
    clip_shape.has_transform = true;
    clip_shape.transform = transform;
  }
  ctx->defs->clip_shapes.push (clip_shape);
  if (likely (!ctx->defs->clip_shapes.in_error ()))
    ctx->clip->shape_count++;
  else if (ctx->had_alloc_failure)
    *ctx->had_alloc_failure = true;
}

static void
svg_skip_subtree (hb_svg_xml_parser_t &parser)
{
  int depth = 1;
  while (depth > 0)
  {
    hb_svg_token_type_t tok = parser.next ();
    if (tok == SVG_TOKEN_EOF) break;
    if (tok == SVG_TOKEN_CLOSE_TAG) depth--;
    else if (tok == SVG_TOKEN_OPEN_TAG) depth++;
  }
}

static inline bool
svg_resolve_element_visibility (hb_svg_xml_parser_t &parser,
                                bool parent_visible)
{
  hb_svg_style_props_t style_props;
  svg_parse_style_props (parser.find_attr ("style"), &style_props);
  hb_svg_str_t display_str = svg_pick_attr_or_style (parser, style_props.display, "display");
  hb_svg_str_t visibility_str = svg_pick_attr_or_style (parser, style_props.visibility, "visibility");
  if (display_str.trim ().eq_ascii_ci ("none"))
    return false;
  hb_svg_str_t vis_trim = visibility_str.trim ();
  if (!vis_trim.len || vis_trim.eq_ascii_ci ("inherit"))
    return parent_visible;
  if (vis_trim.eq_ascii_ci ("hidden") ||
      vis_trim.eq_ascii_ci ("collapse"))
    return false;
  if (vis_trim.eq_ascii_ci ("visible"))
    return true;
  return parent_visible;
}

static void
svg_clip_collect_ref_element (hb_svg_clip_collect_context_t *ctx,
                              hb_svg_xml_parser_t &parser,
                              const hb_svg_transform_t &base_transform,
                              unsigned depth,
                              bool suppress_viewbox_once = false,
                              bool parent_visible = true,
                              bool allow_symbol_once = false);

static void
svg_clip_collect_use_target (hb_svg_clip_collect_context_t *ctx,
                             hb_svg_xml_parser_t &use_parser,
                             const hb_svg_transform_t &base_transform,
                             unsigned depth)
{
  const unsigned SVG_MAX_CLIP_USE_DEPTH = 64;
  if (depth >= SVG_MAX_CLIP_USE_DEPTH)
    return;

  hb_svg_str_t href = hb_raster_svg_find_href_attr (use_parser);
  hb_svg_str_t ref_id;
  if (!hb_raster_svg_parse_local_id_ref (href, &ref_id, nullptr))
    return;

  const char *found = nullptr;
  if (!hb_raster_svg_find_element_by_id (ctx->doc_start, ctx->doc_len,
                                         ctx->svg_accel, ctx->doc_cache,
                                         ref_id, &found))
    return;

  hb_decycler_node_t node (*ctx->use_decycler);
  if (unlikely (!node.visit ((uintptr_t) found)))
    return;

  hb_svg_transform_t effective = base_transform;
  float use_x = 0.f, use_y = 0.f, use_w = 0.f, use_h = 0.f;
  hb_raster_svg_parse_use_geometry (use_parser, &use_x, &use_y, &use_w, &use_h);
  if (use_x != 0.f || use_y != 0.f)
  {
    hb_svg_transform_t tr;
    tr.dx = use_x;
    tr.dy = use_y;
    effective.multiply (tr);
  }

  unsigned remaining = ctx->doc_len - (unsigned) (found - ctx->doc_start);
  hb_svg_xml_parser_t ref_parser (found, remaining);
  hb_svg_token_type_t rt = ref_parser.next ();
  if (rt != SVG_TOKEN_OPEN_TAG && rt != SVG_TOKEN_SELF_CLOSE_TAG)
    return;

  bool viewport_mapped = false;
  hb_svg_transform_t vb_t;
  if (hb_raster_svg_compute_use_target_viewbox_transform (ref_parser, use_w, use_h, &vb_t))
  {
    effective.multiply (vb_t);
    viewport_mapped = true;
  }

  bool allow_symbol = ref_parser.tag_name.eq ("symbol");
  svg_clip_collect_ref_element (ctx, ref_parser, effective, depth + 1,
                                viewport_mapped, true, allow_symbol);
}

static void
svg_clip_collect_ref_element (hb_svg_clip_collect_context_t *ctx,
                              hb_svg_xml_parser_t &parser,
                              const hb_svg_transform_t &base_transform,
                              unsigned depth,
                              bool suppress_viewbox_once,
                              bool parent_visible,
                              bool allow_symbol_once)
{
  const unsigned SVG_MAX_CLIP_REF_DEPTH = 64;
  if (depth >= SVG_MAX_CLIP_REF_DEPTH)
  {
    if (!parser.self_closing)
      svg_skip_subtree (parser);
    return;
  }

  bool is_visible = svg_resolve_element_visibility (parser, parent_visible);
  if (!is_visible)
  {
    if (!parser.self_closing)
      svg_skip_subtree (parser);
    return;
  }

  /* Definitions are not directly renderable clip geometry. */
  if (parser.tag_name.eq ("defs"))
  {
    if (!parser.self_closing)
      svg_skip_subtree (parser);
    return;
  }
  if (parser.tag_name.eq ("symbol") && !allow_symbol_once)
  {
    if (!parser.self_closing)
      svg_skip_subtree (parser);
    return;
  }
  if (parser.tag_name.eq ("symbol"))
    allow_symbol_once = false;

  hb_svg_transform_t effective = base_transform;
  hb_svg_style_props_t geom_style_props;
  svg_parse_style_props (parser.find_attr ("style"), &geom_style_props);
  hb_svg_transform_t local_t;
  if (svg_parse_element_transform (parser, &local_t))
    effective.multiply (local_t);
  if (parser.tag_name.eq ("svg"))
  {
    float svg_x = hb_raster_svg_parse_non_percent_length (svg_pick_attr_or_style (parser, geom_style_props.x, "x"));
    float svg_y = hb_raster_svg_parse_non_percent_length (svg_pick_attr_or_style (parser, geom_style_props.y, "y"));
    if (svg_x != 0.f || svg_y != 0.f)
    {
      hb_svg_transform_t tr;
      tr.dx = svg_x;
      tr.dy = svg_y;
      effective.multiply (tr);
    }

    if (!suppress_viewbox_once)
    {
      float vb_x = 0.f, vb_y = 0.f, vb_w = 0.f, vb_h = 0.f;
    if (hb_raster_svg_parse_viewbox (parser.find_attr ("viewBox"),
                                     &vb_x, &vb_y, &vb_w, &vb_h))
    {
      float viewport_w = hb_raster_svg_parse_non_percent_length (svg_pick_attr_or_style (parser, geom_style_props.width, "width"));
      float viewport_h = hb_raster_svg_parse_non_percent_length (svg_pick_attr_or_style (parser, geom_style_props.height, "height"));
      if (!(viewport_w > 0.f && viewport_h > 0.f))
      {
        viewport_w = vb_w;
          viewport_h = vb_h;
        }
        hb_svg_transform_t vb_t;
        if (hb_raster_svg_compute_viewbox_transform (viewport_w, viewport_h,
                                                     vb_x, vb_y, vb_w, vb_h,
                                                     parser.find_attr ("preserveAspectRatio"),
                                                     &vb_t))
          effective.multiply (vb_t);
      }
      suppress_viewbox_once = false;
    }
  }

  hb_svg_shape_emit_data_t shape;
  if (hb_raster_svg_parse_shape_tag (parser, &shape))
  {
    svg_clip_append_shape (ctx, shape, effective);
    if (!parser.self_closing)
      svg_skip_subtree (parser);
    return;
  }

  if (parser.tag_name.eq ("use"))
  {
    svg_clip_collect_use_target (ctx, parser, effective, depth + 1);
    if (!parser.self_closing)
      svg_skip_subtree (parser);
    return;
  }

  bool is_container = hb_raster_svg_tag_is_container (parser.tag_name);
  if (!is_container || parser.self_closing)
  {
    if (!parser.self_closing)
      svg_skip_subtree (parser);
    return;
  }

  int inner_depth = 1;
  while (inner_depth > 0)
  {
    hb_svg_token_type_t tok = parser.next ();
    if (tok == SVG_TOKEN_EOF) break;
    if (tok == SVG_TOKEN_CLOSE_TAG)
    {
      inner_depth--;
      continue;
    }
    if (tok == SVG_TOKEN_OPEN_TAG || tok == SVG_TOKEN_SELF_CLOSE_TAG)
      svg_clip_collect_ref_element (ctx, parser, effective, depth + 1, false, is_visible, false);
  }
}

void
hb_raster_svg_process_clip_path_def (hb_svg_defs_t *defs,
                           hb_svg_xml_parser_t &parser,
                           hb_svg_token_type_t tok,
                           const char *doc_start,
                           unsigned doc_len,
                           const OT::SVG::accelerator_t *svg_accel,
                           const OT::SVG::svg_doc_cache_t *doc_cache)
{
  hb_svg_clip_path_def_t clip;
  hb_svg_str_t id = parser.find_attr ("id");
  hb_svg_str_t units = parser.find_attr ("clipPathUnits").trim ();
  if (units.eq_ascii_ci ("objectBoundingBox"))
    clip.units_user_space = false;
  else if (units.eq_ascii_ci ("userSpaceOnUse"))
    clip.units_user_space = true;
  else
    clip.units_user_space = true;

  hb_svg_transform_t cp_t;
  if (svg_parse_element_transform (parser, &cp_t))
  {
    clip.has_clip_transform = true;
    clip.clip_transform = cp_t;
  }

  clip.first_shape = defs->clip_shapes.length;
  clip.shape_count = 0;

  if (tok == SVG_TOKEN_OPEN_TAG)
  {
    const unsigned SVG_MAX_CLIP_DEPTH = 64;
    hb_svg_transform_t inherited[SVG_MAX_CLIP_DEPTH];
    bool inherited_visibility[SVG_MAX_CLIP_DEPTH];
    inherited[0] = hb_svg_transform_t ();
    inherited[1] = hb_svg_transform_t ();
    inherited_visibility[0] = true;
    inherited_visibility[1] = true;
    hb_decycler_t use_decycler;

    int cdepth = 1;
    bool had_alloc_failure = false;
    hb_svg_clip_collect_context_t collect_ctx = {
      defs, &clip,
      doc_start, doc_len, svg_accel, doc_cache,
      &use_decycler, &had_alloc_failure
    };
    while (cdepth > 0)
    {
      hb_svg_token_type_t ct = parser.next ();
      if (ct == SVG_TOKEN_EOF) break;
      if (ct == SVG_TOKEN_CLOSE_TAG) { cdepth--; continue; }
      if (ct == SVG_TOKEN_OPEN_TAG || ct == SVG_TOKEN_SELF_CLOSE_TAG)
      {
        if (parser.tag_name.eq ("symbol"))
        {
          if (ct == SVG_TOKEN_OPEN_TAG)
          {
            int skip_depth = 1;
            while (skip_depth > 0)
            {
              hb_svg_token_type_t st = parser.next ();
              if (st == SVG_TOKEN_EOF) break;
              if (st == SVG_TOKEN_CLOSE_TAG) skip_depth--;
              else if (st == SVG_TOKEN_OPEN_TAG) skip_depth++;
            }
          }
          continue;
        }

        if (parser.tag_name.eq ("defs"))
        {
          if (ct == SVG_TOKEN_OPEN_TAG)
          {
            int skip_depth = 1;
            while (skip_depth > 0)
            {
              hb_svg_token_type_t st = parser.next ();
              if (st == SVG_TOKEN_EOF) break;
              if (st == SVG_TOKEN_CLOSE_TAG) skip_depth--;
              else if (st == SVG_TOKEN_OPEN_TAG) skip_depth++;
            }
          }
          continue;
        }

        bool parent_visible = (unsigned) cdepth < SVG_MAX_CLIP_DEPTH
                            ? inherited_visibility[cdepth]
                            : true;
        bool is_visible = svg_resolve_element_visibility (parser, parent_visible);
        bool is_hidden = !is_visible;
        if (is_hidden)
        {
          if (ct == SVG_TOKEN_OPEN_TAG)
          {
            int skip_depth = 1;
            while (skip_depth > 0)
            {
              hb_svg_token_type_t st = parser.next ();
              if (st == SVG_TOKEN_EOF) break;
              if (st == SVG_TOKEN_CLOSE_TAG) skip_depth--;
              else if (st == SVG_TOKEN_OPEN_TAG) skip_depth++;
            }
          }
          continue;
        }

        hb_svg_transform_t effective = (unsigned) cdepth < SVG_MAX_CLIP_DEPTH
                                     ? inherited[cdepth]
                                     : hb_svg_transform_t ();

        hb_svg_transform_t local;
        bool has_local_transform = svg_parse_element_transform (parser, &local);
        if (has_local_transform)
        {
          effective.multiply (local);
        }

        if (parser.tag_name.eq ("use"))
        {
          svg_clip_collect_use_target (&collect_ctx, parser, effective, 0);
        }
        else
        {
          hb_svg_shape_emit_data_t shape;
          if (hb_raster_svg_parse_shape_tag (parser, &shape))
            svg_clip_append_shape (&collect_ctx, shape, effective);
        }

        if (ct == SVG_TOKEN_OPEN_TAG)
        {
          if ((unsigned) (cdepth + 1) < SVG_MAX_CLIP_DEPTH)
          {
            inherited[cdepth + 1] = effective;
            inherited_visibility[cdepth + 1] = is_visible;
          }
          cdepth++;
        }
      }
    }
    if (had_alloc_failure)
      id = {};
  }

  if (id.len)
    (void) defs->add_clip_path (hb_bytes_t (id.data, id.len), clip);
}

struct hb_svg_clip_emit_data_t
{
  const hb_svg_defs_t *defs;
  const hb_svg_clip_path_def_t *clip;
  hb_transform_t<> base_transform;
  hb_transform_t<> bbox_transform;
  bool has_bbox_transform = false;
};

static inline hb_transform_t<>
svg_to_hb_transform (const hb_svg_transform_t &t)
{
  return hb_transform_t<> (t.xx, t.yx, t.xy, t.yy, t.dx, t.dy);
}

static void
svg_clip_path_emit (hb_draw_funcs_t *dfuncs,
                    void *draw_data,
                    void *user_data)
{
  hb_raster_draw_t *rdr = (hb_raster_draw_t *) draw_data;
  hb_svg_clip_emit_data_t *ed = (hb_svg_clip_emit_data_t *) user_data;
  const hb_svg_clip_path_def_t *clip = ed->clip;

  for (unsigned i = 0; i < clip->shape_count; i++)
  {
    const hb_svg_clip_shape_t &s = ed->defs->clip_shapes[clip->first_shape + i];
    hb_transform_t<> t = ed->base_transform;
    if (ed->has_bbox_transform)
      t.multiply (ed->bbox_transform);
    if (clip->has_clip_transform)
      t.multiply (svg_to_hb_transform (clip->clip_transform));
    if (s.has_transform)
      t.multiply (svg_to_hb_transform (s.transform));

    hb_raster_draw_set_transform (rdr, t.xx, t.yx, t.xy, t.yy, t.x0, t.y0);

    hb_svg_shape_emit_data_t shape = s.shape;
    hb_raster_svg_shape_path_emit (dfuncs, draw_data, &shape);
  }
}

bool
hb_raster_svg_push_clip_path_ref (hb_raster_paint_t *paint,
                        hb_svg_defs_t *defs,
                        hb_svg_str_t clip_path_str,
                        const hb_extents_t<> *object_bbox)
{
  if (clip_path_str.is_null ()) return false;
  hb_svg_str_t trimmed = clip_path_str.trim ();
  if (!trimmed.len || trimmed.eq_ascii_ci ("none")) return false;

  hb_svg_str_t clip_id;
  if (!hb_raster_svg_parse_local_id_ref (trimmed, &clip_id, nullptr))
    return false;

  const hb_svg_clip_path_def_t *clip = defs->find_clip_path (hb_bytes_t (clip_id.data, clip_id.len));
  if (!clip) return false;

  hb_svg_clip_emit_data_t ed;
  ed.defs = defs;
  ed.clip = clip;
  ed.base_transform = paint->current_effective_transform ();

  if (!clip->units_user_space)
  {
    if (!object_bbox || object_bbox->is_empty ())
      return false;
    float w = object_bbox->xmax - object_bbox->xmin;
    float h = object_bbox->ymax - object_bbox->ymin;
    if (!(isfinite (w) && isfinite (h)) || w <= 0.f || h <= 0.f)
      return false;
    ed.has_bbox_transform = true;
    ed.bbox_transform = hb_transform_t<> (w, 0, 0, h, object_bbox->xmin, object_bbox->ymin);
  }

  hb_raster_paint_push_clip_path (paint, svg_clip_path_emit, &ed);
  return true;
}

#endif /* !HB_NO_RASTER_SVG */
