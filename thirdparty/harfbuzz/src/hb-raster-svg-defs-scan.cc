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

#include "hb-raster-svg-defs-scan.hh"

#include "hb-raster-svg-clip.hh"
#include "hb-raster-svg-gradient.hh"

static inline bool
hb_raster_svg_process_defs_child_tag (const hb_svg_defs_scan_context_t *ctx,
                                      hb_svg_xml_parser_t &parser,
                                      hb_svg_token_type_t tok)
{
  if (parser.tag_name.eq ("linearGradient"))
  {
    hb_raster_svg_process_gradient_def (ctx->defs, parser, tok, SVG_GRADIENT_LINEAR,
                                        ctx->pfuncs, ctx->paint_data,
                                        ctx->foreground, ctx->face,
                                        ctx->palette);
    return true;
  }
  if (parser.tag_name.eq ("radialGradient"))
  {
    hb_raster_svg_process_gradient_def (ctx->defs, parser, tok, SVG_GRADIENT_RADIAL,
                                        ctx->pfuncs, ctx->paint_data,
                                        ctx->foreground, ctx->face,
                                        ctx->palette);
    return true;
  }
  if (parser.tag_name.eq ("clipPath"))
  {
    hb_raster_svg_process_clip_path_def (ctx->defs, parser, tok,
                                         ctx->doc_start, ctx->doc_len,
                                         ctx->svg_accel, ctx->doc_cache);
    return true;
  }
  return false;
}

void
hb_raster_svg_process_defs_element (const hb_svg_defs_scan_context_t *ctx,
                                    hb_svg_xml_parser_t &parser)
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
      if (!hb_raster_svg_process_defs_child_tag (ctx, parser, tok) &&
          tok == SVG_TOKEN_OPEN_TAG)
        depth++;
    }
  }
}

void
hb_raster_svg_collect_defs (const hb_svg_defs_scan_context_t *ctx,
                            const char *data,
                            unsigned data_len)
{
  hb_svg_xml_parser_t parser (data, data_len);
  while (true)
  {
    hb_svg_token_type_t tok = parser.next ();
    if (tok == SVG_TOKEN_EOF) break;
    if (tok == SVG_TOKEN_OPEN_TAG || tok == SVG_TOKEN_SELF_CLOSE_TAG)
      hb_raster_svg_process_defs_child_tag (ctx, parser, tok);
  }
}

#endif /* !HB_NO_RASTER_SVG */
