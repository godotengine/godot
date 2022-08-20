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

#if !defined(HB_OT_H_IN) && !defined(HB_NO_SINGLE_HEADER_ERROR)
#error "Include <hb-ot.h> instead."
#endif

#ifndef HB_OT_VAR_H
#define HB_OT_VAR_H

#include "hb.h"

HB_BEGIN_DECLS

/**
 * HB_OT_TAG_VAR_AXIS_ITALIC:
 *
 * Registered tag for the roman/italic axis.
 */
#define HB_OT_TAG_VAR_AXIS_ITALIC	HB_TAG('i','t','a','l')

/**
 * HB_OT_TAG_VAR_AXIS_OPTICAL_SIZE:
 *
 * Registered tag for the optical-size axis.
 * <note>Note: The optical-size axis supersedes the OpenType `size` feature.</note>
 */
#define HB_OT_TAG_VAR_AXIS_OPTICAL_SIZE	HB_TAG('o','p','s','z')

/**
 * HB_OT_TAG_VAR_AXIS_SLANT:
 *
 * Registered tag for the slant axis
 */
#define HB_OT_TAG_VAR_AXIS_SLANT	HB_TAG('s','l','n','t')

/**
 * HB_OT_TAG_VAR_AXIS_WIDTH:
 *
 * Registered tag for the width axis.
 */
#define HB_OT_TAG_VAR_AXIS_WIDTH	HB_TAG('w','d','t','h')

/**
 * HB_OT_TAG_VAR_AXIS_WEIGHT:
 *
 * Registered tag for the weight axis.
 */
#define HB_OT_TAG_VAR_AXIS_WEIGHT	HB_TAG('w','g','h','t')


/*
 * fvar / avar
 */

HB_EXTERN hb_bool_t
hb_ot_var_has_data (hb_face_t *face);


/*
 * Variation axes.
 */


HB_EXTERN unsigned int
hb_ot_var_get_axis_count (hb_face_t *face);

/**
 * hb_ot_var_axis_flags_t:
 * @HB_OT_VAR_AXIS_FLAG_HIDDEN: The axis should not be exposed directly in user interfaces.
 *
 * Flags for #hb_ot_var_axis_info_t.
 *
 * Since: 2.2.0
 */
typedef enum { /*< flags >*/
  HB_OT_VAR_AXIS_FLAG_HIDDEN	= 0x00000001u,

  /*< private >*/
  _HB_OT_VAR_AXIS_FLAG_MAX_VALUE= HB_TAG_MAX_SIGNED /*< skip >*/
} hb_ot_var_axis_flags_t;

/**
 * hb_ot_var_axis_info_t:
 * @axis_index: Index of the axis in the variation-axis array
 * @tag: The #hb_tag_t tag identifying the design variation of the axis
 * @name_id: The `name` table Name ID that provides display names for the axis
 * @flags: The #hb_ot_var_axis_flags_t flags for the axis
 * @min_value: The minimum value on the variation axis that the font covers
 * @default_value: The position on the variation axis corresponding to the font's defaults
 * @max_value: The maximum value on the variation axis that the font covers
 * 
 * Data type for holding variation-axis values.
 *
 * The minimum, default, and maximum values are in un-normalized, user scales.
 *
 * <note>Note: at present, the only flag defined for @flags is
 * #HB_OT_VAR_AXIS_FLAG_HIDDEN.</note>
 *
 * Since: 2.2.0
 */
typedef struct hb_ot_var_axis_info_t {
  unsigned int			axis_index;
  hb_tag_t			tag;
  hb_ot_name_id_t		name_id;
  hb_ot_var_axis_flags_t	flags;
  float				min_value;
  float				default_value;
  float				max_value;
  /*< private >*/
  unsigned int			reserved;
} hb_ot_var_axis_info_t;

HB_EXTERN unsigned int
hb_ot_var_get_axis_infos (hb_face_t             *face,
			  unsigned int           start_offset,
			  unsigned int          *axes_count /* IN/OUT */,
			  hb_ot_var_axis_info_t *axes_array /* OUT */);

HB_EXTERN hb_bool_t
hb_ot_var_find_axis_info (hb_face_t             *face,
			  hb_tag_t               axis_tag,
			  hb_ot_var_axis_info_t *axis_info);


/*
 * Named instances.
 */

HB_EXTERN unsigned int
hb_ot_var_get_named_instance_count (hb_face_t *face);

HB_EXTERN hb_ot_name_id_t
hb_ot_var_named_instance_get_subfamily_name_id (hb_face_t   *face,
						unsigned int instance_index);

HB_EXTERN hb_ot_name_id_t
hb_ot_var_named_instance_get_postscript_name_id (hb_face_t  *face,
						unsigned int instance_index);

HB_EXTERN unsigned int
hb_ot_var_named_instance_get_design_coords (hb_face_t    *face,
					    unsigned int  instance_index,
					    unsigned int *coords_length, /* IN/OUT */
					    float        *coords         /* OUT */);


/*
 * Conversions.
 */

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
