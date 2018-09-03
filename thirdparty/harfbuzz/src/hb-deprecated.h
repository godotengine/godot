/*
 * Copyright © 2013  Google, Inc.
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
 * Google Author(s): Behdad Esfahbod
 */

#ifndef HB_H_IN
#error "Include <hb.h> instead."
#endif

#ifndef HB_DEPRECATED_H
#define HB_DEPRECATED_H

#include "hb-common.h"
#include "hb-unicode.h"
#include "hb-font.h"
#include "hb-set.h"

HB_BEGIN_DECLS

#ifndef HB_DISABLE_DEPRECATED

#define HB_SCRIPT_CANADIAN_ABORIGINAL		HB_SCRIPT_CANADIAN_SYLLABICS

#define HB_BUFFER_FLAGS_DEFAULT			HB_BUFFER_FLAG_DEFAULT
#define HB_BUFFER_SERIALIZE_FLAGS_DEFAULT	HB_BUFFER_SERIALIZE_FLAG_DEFAULT

typedef hb_bool_t (*hb_font_get_glyph_func_t) (hb_font_t *font, void *font_data,
					       hb_codepoint_t unicode, hb_codepoint_t variation_selector,
					       hb_codepoint_t *glyph,
					       void *user_data);

HB_EXTERN void
hb_font_funcs_set_glyph_func (hb_font_funcs_t *ffuncs,
			      hb_font_get_glyph_func_t func,
			      void *user_data, hb_destroy_func_t destroy);

HB_EXTERN void
hb_set_invert (hb_set_t *set);

#endif

HB_END_DECLS

#endif /* HB_DEPRECATED_H */
