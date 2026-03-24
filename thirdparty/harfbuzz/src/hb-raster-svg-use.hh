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

#ifndef HB_RASTER_SVG_USE_HH
#define HB_RASTER_SVG_USE_HH

#include "hb.hh"

#include "hb-raster-svg-context.hh"
#include "hb-raster-svg-parse.hh"

typedef void (*hb_svg_use_render_cb_t) (void *render_user,
                                        hb_svg_xml_parser_t &parser,
                                        const void *state,
                                        bool viewport_mapped);

HB_INTERNAL void
hb_raster_svg_render_use_element (const hb_svg_use_context_t *ctx,
                        hb_svg_xml_parser_t &parser,
                        const void *state,
                        hb_svg_str_t transform_str,
                        hb_svg_use_render_cb_t render_cb,
                        void *render_user);

#endif /* HB_RASTER_SVG_USE_HH */
