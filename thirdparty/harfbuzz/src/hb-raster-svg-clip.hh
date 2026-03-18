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

#ifndef HB_RASTER_SVG_CLIP_HH
#define HB_RASTER_SVG_CLIP_HH

#include "hb.hh"

#include "OT/Color/svg/svg.hh"
#include "hb-paint.hh"
#include "hb-raster-svg-defs.hh"

struct hb_raster_paint_t;

HB_INTERNAL void
hb_raster_svg_process_clip_path_def (hb_svg_defs_t *defs,
                           hb_svg_xml_parser_t &parser,
                           hb_svg_token_type_t tok,
                           const char *doc_start,
                           unsigned doc_len,
                           const OT::SVG::accelerator_t *svg_accel,
                           const OT::SVG::svg_doc_cache_t *doc_cache);

HB_INTERNAL bool
hb_raster_svg_push_clip_path_ref (hb_raster_paint_t *paint,
                        hb_svg_defs_t *defs,
                        hb_svg_str_t clip_path_str,
                        const hb_extents_t<> *object_bbox);

#endif /* HB_RASTER_SVG_CLIP_HH */
