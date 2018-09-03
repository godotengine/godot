/*
 * Copyright © 2017  Google, Inc.
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
 * Red Hat Author(s): Behdad Esfahbod
 */

#ifndef HB_OT_H_IN
#error "Include <hb-ot.h> instead."
#endif

#ifndef HB_OT_VAR_H
#define HB_OT_VAR_H

#include "hb.h"

HB_BEGIN_DECLS


#define HB_OT_TAG_VAR_AXIS_ITALIC	HB_TAG('i','t','a','l')
#define HB_OT_TAG_VAR_AXIS_OPTICAL_SIZE	HB_TAG('o','p','s','z')
#define HB_OT_TAG_VAR_AXIS_SLANT	HB_TAG('s','l','n','t')
#define HB_OT_TAG_VAR_AXIS_WIDTH	HB_TAG('w','d','t','h')
#define HB_OT_TAG_VAR_AXIS_WEIGHT	HB_TAG('w','g','h','t')


/*
 * fvar / avar
 */

/**
 * hb_ot_var_axis_t:
 *
 * Since: 1.4.2
 */
typedef struct hb_ot_var_axis_t {
  hb_tag_t tag;
  unsigned int name_id;
  float min_value;
  float default_value;
  float max_value;
} hb_ot_var_axis_t;

HB_EXTERN hb_bool_t
hb_ot_var_has_data (hb_face_t *face);

/**
 * HB_OT_VAR_NO_AXIS_INDEX:
 *
 * Since: 1.4.2
 */
#define HB_OT_VAR_NO_AXIS_INDEX		0xFFFFFFFFu

HB_EXTERN unsigned int
hb_ot_var_get_axis_count (hb_face_t *face);

HB_EXTERN unsigned int
hb_ot_var_get_axes (hb_face_t        *face,
		    unsigned int      start_offset,
		    unsigned int     *axes_count /* IN/OUT */,
		    hb_ot_var_axis_t *axes_array /* OUT */);

HB_EXTERN hb_bool_t
hb_ot_var_find_axis (hb_face_t        *face,
		     hb_tag_t          axis_tag,
		     unsigned int     *axis_index,
		     hb_ot_var_axis_t *axis_info);


HB_EXTERN void
hb_ot_var_normalize_variations (hb_face_t            *face,
				const hb_variation_t *variations, /* IN */
				unsigned int          variations_length,
				int                  *coords, /* OUT */
				unsigned int          coords_length);

HB_EXTERN void
hb_ot_var_normalize_coords (hb_face_t    *face,
			    unsigned int coords_length,
			    const float *design_coords, /* IN */
			    int *normalized_coords /* OUT */);


HB_END_DECLS

#endif /* HB_OT_VAR_H */
