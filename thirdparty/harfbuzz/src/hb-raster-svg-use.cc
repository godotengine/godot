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

#include "hb-raster-svg-use.hh"

#include "hb-raster-svg-base.hh"

void
hb_raster_svg_render_use_element (const hb_svg_use_context_t *ctx,
                        hb_svg_xml_parser_t &parser,
                        const void *state,
                        hb_svg_str_t transform_str,
                        hb_svg_use_render_cb_t render_cb,
                        void *render_user)
{
  hb_svg_str_t href = hb_raster_svg_find_href_attr (parser);

  hb_svg_str_t ref_id;
  if (!hb_raster_svg_parse_local_id_ref (href, &ref_id, nullptr))
    return;

  float use_x = 0.f, use_y = 0.f, use_w = 0.f, use_h = 0.f;
  hb_raster_svg_parse_use_geometry (parser, &use_x, &use_y, &use_w, &use_h);

  bool has_translate = (use_x != 0.f || use_y != 0.f);
  bool has_use_transform = transform_str.len > 0;

  if (has_use_transform)
  {
    hb_svg_transform_t t;
    hb_raster_svg_parse_transform (transform_str, &t);
    hb_paint_push_transform (ctx->pfuncs, ctx->paint, t.xx, t.yx, t.xy, t.yy, t.dx, t.dy);
  }

  if (has_translate)
    hb_paint_push_transform (ctx->pfuncs, ctx->paint, 1, 0, 0, 1, use_x, use_y);

  const char *found = nullptr;
  hb_raster_svg_find_element_by_id (ctx->doc_start, ctx->doc_len, ctx->svg_accel, ctx->doc_cache,
                                    ref_id, &found);

  if (found)
  {
    bool can_render = true;
    hb_decycler_node_t node (*ctx->use_decycler);
    if (unlikely (!node.visit ((uintptr_t) found)))
      can_render = false;

    if (can_render)
    {
      unsigned remaining = ctx->doc_start + ctx->doc_len - found;
      hb_svg_xml_parser_t ref_parser (found, remaining);
      hb_svg_token_type_t tok = ref_parser.next ();
      if (tok == SVG_TOKEN_OPEN_TAG || tok == SVG_TOKEN_SELF_CLOSE_TAG)
      {
        bool has_viewport_scale = false;
        hb_svg_transform_t t;
        if (hb_raster_svg_compute_use_target_viewbox_transform (ref_parser, use_w, use_h, &t))
        {
          hb_paint_push_transform (ctx->pfuncs, ctx->paint, t.xx, t.yx, t.xy, t.yy, t.dx, t.dy);
          has_viewport_scale = true;
        }
        render_cb (render_user, ref_parser, state, has_viewport_scale);
        if (has_viewport_scale)
          hb_paint_pop_transform (ctx->pfuncs, ctx->paint);
      }
    }
  }

  if (has_translate)
    hb_paint_pop_transform (ctx->pfuncs, ctx->paint);

  if (has_use_transform)
    hb_paint_pop_transform (ctx->pfuncs, ctx->paint);
}

#endif /* !HB_NO_RASTER_SVG */
