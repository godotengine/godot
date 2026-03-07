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

#include "hb-face.hh"
#include "hb-raster-svg.hh"
#include "hb-raster-svg-base.hh"
#include "hb-raster-svg-parse.hh"
#include "hb-raster-svg-context.hh"
#include "hb-raster-svg-defs-scan.hh"
#include "hb-raster-svg-gradient.hh"
#include "hb-raster-svg-clip.hh"
#include "hb-raster-svg-bbox.hh"
#include "hb-raster-svg-fill.hh"
#include "hb-raster-svg-use.hh"

#include <assert.h>
#include <stdio.h>

#ifndef HB_NO_RASTER_SVG

#define SVG_MAX_DEPTH 32

/*
 * 11. Element renderer — recursive SVG rendering
 */

static void svg_render_element (hb_svg_render_context_t *ctx,
				hb_svg_xml_parser_t &parser,
				const struct hb_svg_cascade_t &inherited);

/* Gradient def parsing lives in hb-raster-svg-gradient.* */

/* Clip-path defs and push helpers live in hb-raster-svg-clip.* */

/* Render a single shape element */
static void
svg_render_shape (hb_svg_render_context_t *ctx,
		  hb_svg_shape_emit_data_t &shape,
		  const hb_svg_cascade_t &state,
		  hb_svg_str_t transform_str)
{
  bool has_transform = transform_str.len > 0;
  bool has_opacity = state.opacity < 1.f;
  bool has_clip_path = false;

  if (has_opacity)
    ctx->push_group ();

  if (has_transform)
  {
    hb_svg_transform_t t;
    hb_raster_svg_parse_transform (transform_str, &t);
    ctx->push_transform (t.xx, t.yx, t.xy, t.yy, t.dx, t.dy);
  }

  hb_extents_t<> bbox;
  bool has_bbox = hb_raster_svg_compute_shape_bbox (shape, &bbox);
  has_clip_path = hb_raster_svg_push_clip_path_ref (ctx->paint, &ctx->defs, state.clip_path,
					  has_bbox ? &bbox : nullptr);

  /* Clip with shape path, then fill */
  hb_raster_paint_push_clip_path (ctx->paint, hb_raster_svg_shape_path_emit, &shape);

  /* Default fill is black */
  if (state.fill.is_null ())
  {
    hb_color_t black = HB_COLOR (0, 0, 0, 255);
    if (state.fill_opacity < 1.f)
      black = HB_COLOR (0, 0, 0, (uint8_t) (255 * state.fill_opacity + 0.5f));
    ctx->paint_color (black);
  }
  else
  {
    hb_svg_fill_context_t fill_ctx = {ctx->paint, ctx->pfuncs, ctx->font, ctx->palette, &ctx->defs};
    hb_raster_svg_emit_fill (&fill_ctx, state.fill, state.fill_opacity, has_bbox ? &bbox : nullptr, state.color);
  }

  ctx->pop_clip ();
  if (has_clip_path)
    ctx->pop_clip ();

  if (has_transform)
    ctx->pop_transform ();

  if (has_opacity)
    ctx->pop_group (HB_PAINT_COMPOSITE_MODE_SRC_OVER);
}

static void
svg_render_container_element (hb_svg_render_context_t *ctx,
			      hb_svg_xml_parser_t &parser,
			      hb_svg_str_t tag,
			      bool self_closing,
			      const hb_svg_cascade_t &state,
			      hb_svg_str_t transform_str,
			      hb_svg_str_t clip_path_str)
{
  bool has_transform = transform_str.len > 0;
  bool has_opacity = state.opacity < 1.f;
  bool has_clip = false;
  bool has_viewbox = false;
  bool has_viewbox_transform = false;
  bool has_svg_translate = false;
  float svg_x = 0.f, svg_y = 0.f;
  float viewport_w = 0.f, viewport_h = 0.f;
  hb_svg_transform_t viewbox_t;
  float vb_x = 0, vb_y = 0, vb_w = 0, vb_h = 0;

  if (tag.eq ("svg") || tag.eq ("symbol"))
  {
    hb_svg_style_props_t geom_style_props;
    svg_parse_style_props (parser.find_attr ("style"), &geom_style_props);

    if (tag.eq ("svg"))
    {
      svg_x = hb_raster_svg_parse_non_percent_length (svg_pick_attr_or_style (parser, geom_style_props.x, "x"));
      svg_y = hb_raster_svg_parse_non_percent_length (svg_pick_attr_or_style (parser, geom_style_props.y, "y"));
      has_svg_translate = (svg_x != 0.f || svg_y != 0.f);
    }

    hb_svg_str_t viewbox_str = parser.find_attr ("viewBox");
    if (hb_raster_svg_parse_viewbox (viewbox_str, &vb_x, &vb_y, &vb_w, &vb_h))
    {
      has_viewbox = true;

      if (tag.eq ("svg"))
      {
        viewport_w = hb_raster_svg_parse_non_percent_length (svg_pick_attr_or_style (parser, geom_style_props.width, "width"));
        viewport_h = hb_raster_svg_parse_non_percent_length (svg_pick_attr_or_style (parser, geom_style_props.height, "height"));
        if (!(viewport_w > 0.f && viewport_h > 0.f))
        {
          viewport_w = vb_w;
          viewport_h = vb_h;
        }
        has_viewbox_transform =
          hb_raster_svg_compute_viewbox_transform (viewport_w, viewport_h,
                                                   vb_x, vb_y, vb_w, vb_h,
                                                   parser.find_attr ("preserveAspectRatio"),
                                                   &viewbox_t);
      }
    }
  }

  if (ctx->suppress_viewbox_once)
  {
    has_viewbox = false;
    has_viewbox_transform = false;
    ctx->suppress_viewbox_once = false;
  }

  if (has_opacity)
    ctx->push_group ();

  if (has_transform)
  {
    hb_svg_transform_t t;
    hb_raster_svg_parse_transform (transform_str, &t);
    ctx->push_transform (t.xx, t.yx, t.xy, t.yy, t.dx, t.dy);
  }

  if (has_svg_translate)
    ctx->push_transform (1, 0, 0, 1, svg_x, svg_y);

  if (has_viewbox_transform)
    ctx->push_transform (viewbox_t.xx, viewbox_t.yx, viewbox_t.xy, viewbox_t.yy,
                         viewbox_t.dx, viewbox_t.dy);
  else if (has_viewbox && vb_w > 0 && vb_h > 0)
    ctx->push_transform (1, 0, 0, 1, -vb_x, -vb_y);

  has_clip = hb_raster_svg_push_clip_path_ref (ctx->paint, &ctx->defs, clip_path_str, nullptr);

  if (!self_closing)
  {
    int depth = 1;
    while (depth > 0)
    {
      hb_svg_token_type_t tok = parser.next ();
      if (tok == SVG_TOKEN_EOF) break;

      if (tok == SVG_TOKEN_CLOSE_TAG)
      {
	depth--;
	continue;
      }

      if (tok == SVG_TOKEN_OPEN_TAG || tok == SVG_TOKEN_SELF_CLOSE_TAG)
      {
	hb_svg_str_t child_tag = parser.tag_name;
	if (parser.tag_name.eq ("defs"))
	{
	  if (tok != SVG_TOKEN_SELF_CLOSE_TAG)
	  {
	    hb_svg_defs_scan_context_t scan_ctx = {
	      &ctx->defs, ctx->pfuncs, ctx->paint,
	      ctx->foreground, hb_font_get_face (ctx->font),
	      ctx->palette,
	      ctx->doc_start, ctx->doc_len,
	      ctx->svg_accel, ctx->doc_cache
	    };
	    hb_raster_svg_process_defs_element (&scan_ctx, parser);
	  }
	  continue;
	}
	svg_render_element (ctx, parser, state);
	if (tok == SVG_TOKEN_OPEN_TAG &&
	    !hb_raster_svg_tag_is_container_or_use (child_tag))
	{
	  /* Skip children of non-container elements we don't handle. */
	  int skip_depth = 1;
	  while (skip_depth > 0)
	  {
	    hb_svg_token_type_t st = parser.next ();
	    if (st == SVG_TOKEN_EOF) break;
	    if (st == SVG_TOKEN_CLOSE_TAG) skip_depth--;
	    else if (st == SVG_TOKEN_OPEN_TAG) skip_depth++;
	  }
	}
      }
    }
  }

  if (has_viewbox_transform || (has_viewbox && vb_w > 0 && vb_h > 0))
    ctx->pop_transform ();

  if (has_svg_translate)
    ctx->pop_transform ();

  if (has_clip)
    ctx->pop_clip ();

  if (has_transform)
    ctx->pop_transform ();

  if (has_opacity)
    ctx->pop_group (HB_PAINT_COMPOSITE_MODE_SRC_OVER);
}

static bool
svg_render_primitive_shape_element (hb_svg_render_context_t *ctx,
				    hb_svg_xml_parser_t &parser,
				    const hb_svg_cascade_t &state,
				    hb_svg_str_t transform_str)
{
  hb_svg_str_t tag = parser.tag_name;
  if (!(tag.eq ("path") || tag.eq ("rect") || tag.eq ("circle") ||
        tag.eq ("ellipse") || tag.eq ("line") || tag.eq ("polyline") ||
        tag.eq ("polygon")))
    return false;

  hb_svg_shape_emit_data_t shape;
  if (hb_raster_svg_parse_shape_tag (parser, &shape))
    svg_render_shape (ctx, shape, state, transform_str);
  return true;
}

static void
svg_render_use_callback (void *render_user,
			 hb_svg_xml_parser_t &parser,
			 const void *state,
			 bool viewport_mapped)
{
  hb_svg_render_context_t *ctx = (hb_svg_render_context_t *) render_user;
  bool old_suppress = ctx->suppress_viewbox_once;
  bool old_allow_symbol = ctx->allow_symbol_render_once;
  if (viewport_mapped)
    ctx->suppress_viewbox_once = true;
  if (parser.tag_name.eq ("symbol"))
    ctx->allow_symbol_render_once = true;
  svg_render_element (ctx, parser, *(const hb_svg_cascade_t *) state);
  ctx->suppress_viewbox_once = old_suppress;
  ctx->allow_symbol_render_once = old_allow_symbol;
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

/* Render one element (may be a container or shape) */
static void
svg_render_element (hb_svg_render_context_t *ctx,
		    hb_svg_xml_parser_t &parser,
		    const hb_svg_cascade_t &inherited)
{
  if (ctx->depth >= SVG_MAX_DEPTH) return;

  const HB_UNUSED unsigned transform_depth = ctx->paint->transform_stack.length;
  const HB_UNUSED unsigned clip_depth = ctx->paint->clip_stack.length;
  const HB_UNUSED unsigned surface_depth = ctx->paint->surface_stack.length;

  ctx->depth++;

  hb_svg_str_t tag = parser.tag_name;
  bool self_closing = parser.self_closing;

  /* Extract common attributes */
  hb_svg_str_t style = parser.find_attr ("style");
  hb_svg_style_props_t style_props;
  svg_parse_style_props (style, &style_props);
  hb_svg_str_t fill_attr = svg_pick_attr_or_style (parser, style_props.fill, "fill");
  hb_svg_str_t fill_opacity_str = svg_pick_attr_or_style (parser, style_props.fill_opacity, "fill-opacity");
  hb_svg_str_t opacity_str = svg_pick_attr_or_style (parser, style_props.opacity, "opacity");
  hb_svg_str_t transform_str = svg_pick_attr_or_style (parser, style_props.transform, "transform");
  hb_svg_str_t clip_path_attr = svg_pick_attr_or_style (parser, style_props.clip_path, "clip-path");
  hb_svg_str_t display_str = svg_pick_attr_or_style (parser, style_props.display, "display");
  hb_svg_str_t color_str = svg_pick_attr_or_style (parser, style_props.color, "color");
  hb_svg_str_t visibility_str = svg_pick_attr_or_style (parser, style_props.visibility, "visibility");

  hb_svg_cascade_t state = inherited;
  state.fill = (fill_attr.is_null () || svg_str_is_inherit (fill_attr)) ? inherited.fill : fill_attr;
  state.fill_opacity = (fill_opacity_str.len && !svg_str_is_inherit (fill_opacity_str))
		       ? svg_parse_float_clamped01 (fill_opacity_str)
		       : inherited.fill_opacity;
  state.opacity = (opacity_str.len && !svg_str_is_inherit (opacity_str) && !svg_str_is_none (opacity_str))
		  ? svg_parse_float_clamped01 (opacity_str)
		  : (svg_str_is_inherit (opacity_str) ? inherited.opacity : 1.f);
  state.clip_path = (clip_path_attr.is_null () || svg_str_is_inherit (clip_path_attr))
		    ? inherited.clip_path
		    : clip_path_attr;
  if (svg_str_is_inherit (transform_str) || svg_str_is_none (transform_str))
    transform_str = {};
  state.color = inherited.color;
  bool is_none = false;
  if (color_str.len && !color_str.trim ().eq_ascii_ci ("inherit"))
    state.color = hb_raster_svg_parse_color (color_str, ctx->pfuncs, ctx->paint,
				   inherited.color, hb_font_get_face (ctx->font),
				   ctx->palette, &is_none);
  state.visibility = inherited.visibility;
  hb_svg_str_t visibility_trim = visibility_str.trim ();
  if (visibility_trim.len && !visibility_trim.eq_ascii_ci ("inherit"))
    state.visibility = !(visibility_trim.eq_ascii_ci ("hidden") ||
			 visibility_trim.eq_ascii_ci ("collapse"));

  if (display_str.trim ().eq_ascii_ci ("none"))
  {
    if (!self_closing)
      svg_skip_subtree (parser);
    ctx->depth--;
    assert (ctx->paint->transform_stack.length == transform_depth);
    assert (ctx->paint->clip_stack.length == clip_depth);
    assert (ctx->paint->surface_stack.length == surface_depth);
    return;
  }
  if (!state.visibility)
  {
    if (!self_closing)
      svg_skip_subtree (parser);
    ctx->depth--;
    assert (ctx->paint->transform_stack.length == transform_depth);
    assert (ctx->paint->clip_stack.length == clip_depth);
    assert (ctx->paint->surface_stack.length == surface_depth);
    return;
  }

  if (tag.eq ("symbol") && !ctx->allow_symbol_render_once)
  {
    if (!self_closing)
      svg_skip_subtree (parser);
    ctx->depth--;
    assert (ctx->paint->transform_stack.length == transform_depth);
    assert (ctx->paint->clip_stack.length == clip_depth);
    assert (ctx->paint->surface_stack.length == surface_depth);
    return;
  }
  if (tag.eq ("symbol"))
    ctx->allow_symbol_render_once = false;

  if (hb_raster_svg_tag_is_container (tag))
    svg_render_container_element (ctx, parser, tag, self_closing,
				  state, transform_str, state.clip_path);
  else if (svg_render_primitive_shape_element (ctx, parser, state, transform_str))
    ;
  else if (tag.eq ("use"))
  {
    hb_svg_use_context_t use_ctx = {ctx->paint, ctx->pfuncs,
				    ctx->doc_start, ctx->doc_len,
				    ctx->svg_accel, ctx->doc_cache,
				    &ctx->use_decycler};
    hb_raster_svg_render_use_element (&use_ctx, parser, &state, transform_str,
			    svg_render_use_callback, ctx);
  }

  ctx->depth--;

  assert (ctx->paint->transform_stack.length == transform_depth);
  assert (ctx->paint->clip_stack.length == clip_depth);
  assert (ctx->paint->surface_stack.length == surface_depth);
}


/*
 * 12. Entry point
 */

hb_bool_t
hb_raster_svg_render (hb_raster_paint_t *paint,
		      hb_blob_t *blob,
		      hb_codepoint_t glyph,
		      hb_font_t *font,
		      unsigned palette,
		      hb_color_t foreground)
{
  unsigned data_len;
  const char *data = hb_blob_get_data (blob, &data_len);
  if (!data || !data_len) return false;

  hb_face_t *face = hb_font_get_face (font);
  const OT::SVG::svg_doc_cache_t *doc_cache = nullptr;
  unsigned doc_index = 0;
  hb_codepoint_t start_glyph = HB_CODEPOINT_INVALID;
  hb_codepoint_t end_glyph = HB_CODEPOINT_INVALID;

  if (face &&
      hb_ot_color_glyph_get_svg_document_index (face, glyph, &doc_index) &&
      hb_ot_color_get_svg_document_glyph_range (face, doc_index, &start_glyph, &end_glyph))
    doc_cache = face->table.SVG->get_or_create_doc_cache (blob, data, data_len,
                                                          doc_index, start_glyph, end_glyph);

  if (doc_cache)
    data = face->table.SVG->doc_cache_get_svg (doc_cache, &data_len);

  hb_paint_funcs_t *pfuncs = hb_raster_paint_get_funcs ();

  hb_svg_render_context_t ctx;
  ctx.paint = paint;
  ctx.pfuncs = pfuncs;
  ctx.font = font;
  ctx.palette = palette;
  ctx.foreground = foreground;
  ctx.doc_start = data;
  ctx.doc_len = data_len;
  ctx.svg_accel = face ? face->table.SVG.get () : nullptr;
  ctx.doc_cache = doc_cache;

  hb_svg_cascade_t initial_state;
  initial_state.color = foreground;

  hb_svg_defs_scan_context_t scan_ctx = {
    &ctx.defs, ctx.pfuncs, ctx.paint,
    ctx.foreground, hb_font_get_face (ctx.font),
    ctx.palette,
    ctx.doc_start, ctx.doc_len,
    ctx.svg_accel, ctx.doc_cache
  };
  hb_raster_svg_collect_defs (&scan_ctx, data, data_len);

  bool found_glyph = false;
  unsigned glyph_start = 0, glyph_end = 0;
  if (doc_cache && face->table.SVG->doc_cache_get_glyph_span (doc_cache, glyph, &glyph_start, &glyph_end))
  {
    hb_svg_xml_parser_t parser (data + glyph_start, glyph_end - glyph_start);
    hb_svg_token_type_t tok = parser.next ();
    if (tok == SVG_TOKEN_OPEN_TAG || tok == SVG_TOKEN_SELF_CLOSE_TAG)
    {
      hb_paint_push_font_transform (ctx.pfuncs, ctx.paint, font);
      ctx.push_transform (1, 0, 0, -1, 0, 0);
	      svg_render_element (&ctx, parser, initial_state);
      ctx.pop_transform ();
      hb_paint_pop_transform (ctx.pfuncs, ctx.paint);
      found_glyph = true;
    }
  }
  else
  {
    /* Fallback for malformed/uncached docs: linear scan by glyph id. */
    char glyph_id_str[32];
    int glyph_id_len = snprintf (glyph_id_str, sizeof (glyph_id_str), "glyph%u", glyph);
    if (glyph_id_len <= 0 || (unsigned) glyph_id_len >= sizeof (glyph_id_str))
      return false;
    hb_svg_xml_parser_t parser (data, data_len);
    while (true)
    {
      hb_svg_token_type_t tok = parser.next ();
      if (tok == SVG_TOKEN_EOF) break;

      if (tok == SVG_TOKEN_OPEN_TAG || tok == SVG_TOKEN_SELF_CLOSE_TAG)
      {
        hb_svg_str_t id = parser.find_attr ("id");
        if (id.len)
        {
          if (id.len == (unsigned) glyph_id_len &&
              0 == memcmp (id.data, glyph_id_str, (unsigned) glyph_id_len))
          {
            hb_paint_push_font_transform (ctx.pfuncs, ctx.paint, font);
            ctx.push_transform (1, 0, 0, -1, 0, 0);
	            svg_render_element (&ctx, parser, initial_state);
            ctx.pop_transform ();
            hb_paint_pop_transform (ctx.pfuncs, ctx.paint);
            found_glyph = true;
            break;
          }
        }
      }
    }
  }

  return found_glyph;
}

#endif /* !HB_NO_RASTER_SVG */
