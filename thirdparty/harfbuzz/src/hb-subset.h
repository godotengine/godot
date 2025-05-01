/*
 * Copyright © 2018  Google, Inc.
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
 * Google Author(s): Rod Sheeter
 */

#ifndef HB_SUBSET_H
#define HB_SUBSET_H

#include "hb.h"
#include "hb-ot.h"

HB_BEGIN_DECLS

/**
 * hb_subset_input_t:
 *
 * Things that change based on the input. Characters to keep, etc.
 */

typedef struct hb_subset_input_t hb_subset_input_t;

/**
 * hb_subset_plan_t:
 *
 * Contains information about how the subset operation will be executed.
 * Such as mappings from the old glyph ids to the new ones in the subset.
 */

typedef struct hb_subset_plan_t hb_subset_plan_t;

/**
 * hb_subset_flags_t:
 * @HB_SUBSET_FLAGS_DEFAULT: all flags at their default value of false.
 * @HB_SUBSET_FLAGS_NO_HINTING: If set hinting instructions will be dropped in
 * the produced subset. Otherwise hinting instructions will be retained.
 * @HB_SUBSET_FLAGS_RETAIN_GIDS: If set glyph indices will not be modified in
 * the produced subset. If glyphs are dropped their indices will be retained
 * as an empty glyph.
 * @HB_SUBSET_FLAGS_DESUBROUTINIZE: If set and subsetting a CFF font the
 * subsetter will attempt to remove subroutines from the CFF glyphs.
 * @HB_SUBSET_FLAGS_NAME_LEGACY: If set non-unicode name records will be
 * retained in the subset.
 * @HB_SUBSET_FLAGS_SET_OVERLAPS_FLAG:	If set the subsetter will set the
 * OVERLAP_SIMPLE flag on each simple glyph.
 * @HB_SUBSET_FLAGS_PASSTHROUGH_UNRECOGNIZED: If set the subsetter will not
 * drop unrecognized tables and instead pass them through untouched.
 * @HB_SUBSET_FLAGS_NOTDEF_OUTLINE: If set the notdef glyph outline will be
 * retained in the final subset.
 * @HB_SUBSET_FLAGS_GLYPH_NAMES: If set the PS glyph names will be retained
 * in the final subset.
 * @HB_SUBSET_FLAGS_NO_PRUNE_UNICODE_RANGES: If set then the unicode ranges in
 * OS/2 will not be recalculated.
 * @HB_SUBSET_FLAGS_NO_LAYOUT_CLOSURE: If set don't perform glyph closure on layout
 * substitution rules (GSUB). Since: 7.2.0.
 * @HB_SUBSET_FLAGS_OPTIMIZE_IUP_DELTAS: If set perform IUP delta optimization on the
 * remaining gvar table's deltas. Since: 8.5.0
 * @HB_SUBSET_FLAGS_IFTB_REQUIREMENTS: If set enforce requirements on the output subset
 * to allow it to be used with incremental font transfer IFTB patches. Primarily,
 * this forces all outline data to use long (32 bit) offsets. Since: EXPERIMENTAL
 *
 * List of boolean properties that can be configured on the subset input.
 *
 * Since: 2.9.0
 **/
typedef enum { /*< flags >*/
  HB_SUBSET_FLAGS_DEFAULT =		     0x00000000u,
  HB_SUBSET_FLAGS_NO_HINTING =		     0x00000001u,
  HB_SUBSET_FLAGS_RETAIN_GIDS =		     0x00000002u,
  HB_SUBSET_FLAGS_DESUBROUTINIZE =	     0x00000004u,
  HB_SUBSET_FLAGS_NAME_LEGACY =		     0x00000008u,
  HB_SUBSET_FLAGS_SET_OVERLAPS_FLAG =	     0x00000010u,
  HB_SUBSET_FLAGS_PASSTHROUGH_UNRECOGNIZED = 0x00000020u,
  HB_SUBSET_FLAGS_NOTDEF_OUTLINE =	     0x00000040u,
  HB_SUBSET_FLAGS_GLYPH_NAMES =		     0x00000080u,
  HB_SUBSET_FLAGS_NO_PRUNE_UNICODE_RANGES =  0x00000100u,
  HB_SUBSET_FLAGS_NO_LAYOUT_CLOSURE =        0x00000200u,
  HB_SUBSET_FLAGS_OPTIMIZE_IUP_DELTAS	  =  0x00000400u,
#ifdef HB_EXPERIMENTAL_API
  HB_SUBSET_FLAGS_IFTB_REQUIREMENTS       =  0x00000800u,
#endif
} hb_subset_flags_t;

/**
 * hb_subset_sets_t:
 * @HB_SUBSET_SETS_GLYPH_INDEX: the set of glyph indexes to retain in the subset.
 * @HB_SUBSET_SETS_UNICODE: the set of unicode codepoints to retain in the subset.
 * @HB_SUBSET_SETS_NO_SUBSET_TABLE_TAG: the set of table tags which specifies tables that should not be
 * subsetted.
 * @HB_SUBSET_SETS_DROP_TABLE_TAG: the set of table tags which specifies tables which will be dropped
 * in the subset.
 * @HB_SUBSET_SETS_NAME_ID: the set of name ids that will be retained.
 * @HB_SUBSET_SETS_NAME_LANG_ID: the set of name lang ids that will be retained.
 * @HB_SUBSET_SETS_LAYOUT_FEATURE_TAG: the set of layout feature tags that will be retained
 * in the subset.
 * @HB_SUBSET_SETS_LAYOUT_SCRIPT_TAG: the set of layout script tags that will be retained
 * in the subset. Defaults to all tags. Since: 5.0.0
 *
 * List of sets that can be configured on the subset input.
 *
 * Since: 2.9.1
 **/
typedef enum {
  HB_SUBSET_SETS_GLYPH_INDEX = 0,
  HB_SUBSET_SETS_UNICODE,
  HB_SUBSET_SETS_NO_SUBSET_TABLE_TAG,
  HB_SUBSET_SETS_DROP_TABLE_TAG,
  HB_SUBSET_SETS_NAME_ID,
  HB_SUBSET_SETS_NAME_LANG_ID,
  HB_SUBSET_SETS_LAYOUT_FEATURE_TAG,
  HB_SUBSET_SETS_LAYOUT_SCRIPT_TAG,
} hb_subset_sets_t;

HB_EXTERN hb_subset_input_t *
hb_subset_input_create_or_fail (void);

HB_EXTERN hb_subset_input_t *
hb_subset_input_reference (hb_subset_input_t *input);

HB_EXTERN void
hb_subset_input_destroy (hb_subset_input_t *input);

HB_EXTERN hb_bool_t
hb_subset_input_set_user_data (hb_subset_input_t  *input,
			       hb_user_data_key_t *key,
			       void *		   data,
			       hb_destroy_func_t   destroy,
			       hb_bool_t	   replace);

HB_EXTERN void *
hb_subset_input_get_user_data (const hb_subset_input_t *input,
			       hb_user_data_key_t      *key);

HB_EXTERN void
hb_subset_input_keep_everything (hb_subset_input_t *input);

HB_EXTERN hb_set_t *
hb_subset_input_unicode_set (hb_subset_input_t *input);

HB_EXTERN hb_set_t *
hb_subset_input_glyph_set (hb_subset_input_t *input);

HB_EXTERN hb_set_t *
hb_subset_input_set (hb_subset_input_t *input, hb_subset_sets_t set_type);

HB_EXTERN hb_map_t*
hb_subset_input_old_to_new_glyph_mapping (hb_subset_input_t *input);

HB_EXTERN hb_subset_flags_t
hb_subset_input_get_flags (hb_subset_input_t *input);

HB_EXTERN void
hb_subset_input_set_flags (hb_subset_input_t *input,
			   unsigned value);

HB_EXTERN hb_bool_t
hb_subset_input_pin_all_axes_to_default (hb_subset_input_t  *input,
					 hb_face_t          *face);

HB_EXTERN hb_bool_t
hb_subset_input_pin_axis_to_default (hb_subset_input_t  *input,
				     hb_face_t          *face,
				     hb_tag_t            axis_tag);

HB_EXTERN hb_bool_t
hb_subset_input_pin_axis_location (hb_subset_input_t  *input,
				   hb_face_t          *face,
				   hb_tag_t            axis_tag,
				   float               axis_value);

HB_EXTERN hb_bool_t
hb_subset_input_get_axis_range (hb_subset_input_t  *input,
				hb_tag_t            axis_tag,
				float              *axis_min_value,
				float              *axis_max_value,
				float              *axis_def_value);

HB_EXTERN hb_bool_t
hb_subset_input_set_axis_range (hb_subset_input_t  *input,
				hb_face_t          *face,
				hb_tag_t            axis_tag,
				float               axis_min_value,
				float               axis_max_value,
				float               axis_def_value);

HB_EXTERN hb_bool_t
hb_subset_axis_range_from_string (const char *str, int len,
				  float *axis_min_value,
				  float *axis_max_value,
				  float *axis_def_value);

HB_EXTERN void
hb_subset_axis_range_to_string (hb_subset_input_t *input,
				hb_tag_t axis_tag,
				char *buf,
				unsigned size);

#ifdef HB_EXPERIMENTAL_API
HB_EXTERN hb_bool_t
hb_subset_input_override_name_table (hb_subset_input_t  *input,
				     hb_ot_name_id_t     name_id,
				     unsigned            platform_id,
				     unsigned            encoding_id,
				     unsigned            language_id,
				     const char         *name_str,
				     int                 str_len);
#endif

HB_EXTERN hb_face_t *
hb_subset_preprocess (hb_face_t *source);

HB_EXTERN hb_face_t *
hb_subset_or_fail (hb_face_t *source, const hb_subset_input_t *input);

HB_EXTERN hb_face_t *
hb_subset_plan_execute_or_fail (hb_subset_plan_t *plan);

HB_EXTERN hb_subset_plan_t *
hb_subset_plan_create_or_fail (hb_face_t                 *face,
                               const hb_subset_input_t   *input);

HB_EXTERN void
hb_subset_plan_destroy (hb_subset_plan_t *plan);

HB_EXTERN hb_map_t *
hb_subset_plan_old_to_new_glyph_mapping (const hb_subset_plan_t *plan);

HB_EXTERN hb_map_t *
hb_subset_plan_new_to_old_glyph_mapping (const hb_subset_plan_t *plan);

HB_EXTERN hb_map_t *
hb_subset_plan_unicode_to_old_glyph_mapping (const hb_subset_plan_t *plan);


HB_EXTERN hb_subset_plan_t *
hb_subset_plan_reference (hb_subset_plan_t *plan);

HB_EXTERN hb_bool_t
hb_subset_plan_set_user_data (hb_subset_plan_t   *plan,
                              hb_user_data_key_t *key,
                              void               *data,
                              hb_destroy_func_t   destroy,
                              hb_bool_t	          replace);

HB_EXTERN void *
hb_subset_plan_get_user_data (const hb_subset_plan_t *plan,
                              hb_user_data_key_t     *key);


HB_END_DECLS

#endif /* HB_SUBSET_H */
