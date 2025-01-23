/*
 * Copyright © 2007,2008,2009  Red Hat, Inc.
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

#ifndef HB_OT_LAYOUT_H
#define HB_OT_LAYOUT_H

#include "hb.h"

#include "hb-ot-name.h"

HB_BEGIN_DECLS


/**
 * HB_OT_TAG_BASE:
 *
 * OpenType [Baseline Table](https://docs.microsoft.com/en-us/typography/opentype/spec/base).
 */
#define HB_OT_TAG_BASE HB_TAG('B','A','S','E')
/**
 * HB_OT_TAG_GDEF:
 *
 * OpenType [Glyph Definition Table](https://docs.microsoft.com/en-us/typography/opentype/spec/gdef).
 */
#define HB_OT_TAG_GDEF HB_TAG('G','D','E','F')
/**
 * HB_OT_TAG_GSUB:
 *
 * OpenType [Glyph Substitution Table](https://docs.microsoft.com/en-us/typography/opentype/spec/gsub).
 */
#define HB_OT_TAG_GSUB HB_TAG('G','S','U','B')
/**
 * HB_OT_TAG_GPOS:
 *
 * OpenType [Glyph Positioning Table](https://docs.microsoft.com/en-us/typography/opentype/spec/gpos).
 */
#define HB_OT_TAG_GPOS HB_TAG('G','P','O','S')
/**
 * HB_OT_TAG_JSTF:
 *
 * OpenType [Justification Table](https://docs.microsoft.com/en-us/typography/opentype/spec/jstf).
 */
#define HB_OT_TAG_JSTF HB_TAG('J','S','T','F')


/*
 * Script & Language tags.
 */

/**
 * HB_OT_TAG_DEFAULT_SCRIPT:
 *
 * OpenType script tag, `DFLT`, for features that are not script-specific.
 *
 */
#define HB_OT_TAG_DEFAULT_SCRIPT	HB_TAG ('D', 'F', 'L', 'T')
/**
 * HB_OT_TAG_DEFAULT_LANGUAGE:
 *
 * OpenType language tag, `dflt`. Not a valid language tag, but some fonts
 * mistakenly use it.
 */
#define HB_OT_TAG_DEFAULT_LANGUAGE	HB_TAG ('d', 'f', 'l', 't')

/**
 * HB_OT_MAX_TAGS_PER_SCRIPT:
 *
 * Maximum number of OpenType tags that can correspond to a give #hb_script_t.
 *
 * Since: 2.0.0
 **/
#define HB_OT_MAX_TAGS_PER_SCRIPT	3u
/**
 * HB_OT_MAX_TAGS_PER_LANGUAGE:
 *
 * Maximum number of OpenType tags that can correspond to a give #hb_language_t.
 *
 * Since: 2.0.0
 **/
#define HB_OT_MAX_TAGS_PER_LANGUAGE	3u

HB_EXTERN void
hb_ot_tags_from_script_and_language (hb_script_t   script,
				     hb_language_t language,
				     unsigned int *script_count /* IN/OUT */,
				     hb_tag_t     *script_tags /* OUT */,
				     unsigned int *language_count /* IN/OUT */,
				     hb_tag_t     *language_tags /* OUT */);

HB_EXTERN hb_script_t
hb_ot_tag_to_script (hb_tag_t tag);

HB_EXTERN hb_language_t
hb_ot_tag_to_language (hb_tag_t tag);

HB_EXTERN void
hb_ot_tags_to_script_and_language (hb_tag_t       script_tag,
				   hb_tag_t       language_tag,
				   hb_script_t   *script /* OUT */,
				   hb_language_t *language /* OUT */);


/*
 * GDEF
 */

HB_EXTERN hb_bool_t
hb_ot_layout_has_glyph_classes (hb_face_t *face);

/**
 * hb_ot_layout_glyph_class_t:
 * @HB_OT_LAYOUT_GLYPH_CLASS_UNCLASSIFIED: Glyphs not matching the other classifications
 * @HB_OT_LAYOUT_GLYPH_CLASS_BASE_GLYPH: Spacing, single characters, capable of accepting marks
 * @HB_OT_LAYOUT_GLYPH_CLASS_LIGATURE: Glyphs that represent ligation of multiple characters
 * @HB_OT_LAYOUT_GLYPH_CLASS_MARK: Non-spacing, combining glyphs that represent marks
 * @HB_OT_LAYOUT_GLYPH_CLASS_COMPONENT: Spacing glyphs that represent part of a single character
 *
 * The GDEF classes defined for glyphs.
 *
 **/
typedef enum {
  HB_OT_LAYOUT_GLYPH_CLASS_UNCLASSIFIED	= 0,
  HB_OT_LAYOUT_GLYPH_CLASS_BASE_GLYPH	= 1,
  HB_OT_LAYOUT_GLYPH_CLASS_LIGATURE	= 2,
  HB_OT_LAYOUT_GLYPH_CLASS_MARK		= 3,
  HB_OT_LAYOUT_GLYPH_CLASS_COMPONENT	= 4
} hb_ot_layout_glyph_class_t;

HB_EXTERN hb_ot_layout_glyph_class_t
hb_ot_layout_get_glyph_class (hb_face_t      *face,
			      hb_codepoint_t  glyph);

HB_EXTERN void
hb_ot_layout_get_glyphs_in_class (hb_face_t                  *face,
				  hb_ot_layout_glyph_class_t  klass,
				  hb_set_t                   *glyphs /* OUT */);

/* Not that useful.  Provides list of attach points for a glyph that a
 * client may want to cache */
HB_EXTERN unsigned int
hb_ot_layout_get_attach_points (hb_face_t      *face,
				hb_codepoint_t  glyph,
				unsigned int    start_offset,
				unsigned int   *point_count /* IN/OUT */,
				unsigned int   *point_array /* OUT */);

/* Ligature caret positions */
HB_EXTERN unsigned int
hb_ot_layout_get_ligature_carets (hb_font_t      *font,
				  hb_direction_t  direction,
				  hb_codepoint_t  glyph,
				  unsigned int    start_offset,
				  unsigned int   *caret_count /* IN/OUT */,
				  hb_position_t  *caret_array /* OUT */);


/*
 * GSUB/GPOS feature query and enumeration interface
 */

/**
 * HB_OT_LAYOUT_NO_SCRIPT_INDEX:
 *
 * Special value for script index indicating unsupported script.
 */
#define HB_OT_LAYOUT_NO_SCRIPT_INDEX		0xFFFFu
/**
 * HB_OT_LAYOUT_NO_FEATURE_INDEX:
 *
 * Special value for feature index indicating unsupported feature.
 */
#define HB_OT_LAYOUT_NO_FEATURE_INDEX		0xFFFFu
/**
 * HB_OT_LAYOUT_DEFAULT_LANGUAGE_INDEX:
 *
 * Special value for language index indicating default or unsupported language.
 */
#define HB_OT_LAYOUT_DEFAULT_LANGUAGE_INDEX	0xFFFFu
/**
 * HB_OT_LAYOUT_NO_VARIATIONS_INDEX:
 *
 * Special value for variations index indicating unsupported variation.
 */
#define HB_OT_LAYOUT_NO_VARIATIONS_INDEX	0xFFFFFFFFu

HB_EXTERN unsigned int
hb_ot_layout_table_get_script_tags (hb_face_t    *face,
				    hb_tag_t      table_tag,
				    unsigned int  start_offset,
				    unsigned int *script_count /* IN/OUT */,
				    hb_tag_t     *script_tags /* OUT */);

HB_EXTERN hb_bool_t
hb_ot_layout_table_find_script (hb_face_t    *face,
				hb_tag_t      table_tag,
				hb_tag_t      script_tag,
				unsigned int *script_index /* OUT */);

HB_EXTERN hb_bool_t
hb_ot_layout_table_select_script (hb_face_t      *face,
				  hb_tag_t        table_tag,
				  unsigned int    script_count,
				  const hb_tag_t *script_tags,
				  unsigned int   *script_index /* OUT */,
				  hb_tag_t       *chosen_script /* OUT */);

HB_EXTERN unsigned int
hb_ot_layout_table_get_feature_tags (hb_face_t    *face,
				     hb_tag_t      table_tag,
				     unsigned int  start_offset,
				     unsigned int *feature_count /* IN/OUT */,
				     hb_tag_t     *feature_tags /* OUT */);

HB_EXTERN unsigned int
hb_ot_layout_script_get_language_tags (hb_face_t    *face,
				       hb_tag_t      table_tag,
				       unsigned int  script_index,
				       unsigned int  start_offset,
				       unsigned int *language_count /* IN/OUT */,
				       hb_tag_t     *language_tags /* OUT */);

HB_EXTERN hb_bool_t
hb_ot_layout_script_select_language (hb_face_t      *face,
				     hb_tag_t        table_tag,
				     unsigned int    script_index,
				     unsigned int    language_count,
				     const hb_tag_t *language_tags,
				     unsigned int   *language_index /* OUT */);

HB_EXTERN hb_bool_t
hb_ot_layout_script_select_language2 (hb_face_t      *face,
				     hb_tag_t        table_tag,
				     unsigned int    script_index,
				     unsigned int    language_count,
				     const hb_tag_t *language_tags,
				     unsigned int   *language_index /* OUT */,
				     hb_tag_t       *chosen_language /* OUT */);

HB_EXTERN hb_bool_t
hb_ot_layout_language_get_required_feature_index (hb_face_t    *face,
						  hb_tag_t      table_tag,
						  unsigned int  script_index,
						  unsigned int  language_index,
						  unsigned int *feature_index /* OUT */);

HB_EXTERN hb_bool_t
hb_ot_layout_language_get_required_feature (hb_face_t    *face,
					    hb_tag_t      table_tag,
					    unsigned int  script_index,
					    unsigned int  language_index,
					    unsigned int *feature_index /* OUT */,
					    hb_tag_t     *feature_tag /* OUT */);

HB_EXTERN unsigned int
hb_ot_layout_language_get_feature_indexes (hb_face_t    *face,
					   hb_tag_t      table_tag,
					   unsigned int  script_index,
					   unsigned int  language_index,
					   unsigned int  start_offset,
					   unsigned int *feature_count /* IN/OUT */,
					   unsigned int *feature_indexes /* OUT */);

HB_EXTERN unsigned int
hb_ot_layout_language_get_feature_tags (hb_face_t    *face,
					hb_tag_t      table_tag,
					unsigned int  script_index,
					unsigned int  language_index,
					unsigned int  start_offset,
					unsigned int *feature_count /* IN/OUT */,
					hb_tag_t     *feature_tags /* OUT */);

HB_EXTERN hb_bool_t
hb_ot_layout_language_find_feature (hb_face_t    *face,
				    hb_tag_t      table_tag,
				    unsigned int  script_index,
				    unsigned int  language_index,
				    hb_tag_t      feature_tag,
				    unsigned int *feature_index /* OUT */);

HB_EXTERN unsigned int
hb_ot_layout_feature_get_lookups (hb_face_t    *face,
				  hb_tag_t      table_tag,
				  unsigned int  feature_index,
				  unsigned int  start_offset,
				  unsigned int *lookup_count /* IN/OUT */,
				  unsigned int *lookup_indexes /* OUT */);

HB_EXTERN unsigned int
hb_ot_layout_table_get_lookup_count (hb_face_t    *face,
				     hb_tag_t      table_tag);

HB_EXTERN void
hb_ot_layout_collect_features (hb_face_t      *face,
			       hb_tag_t        table_tag,
			       const hb_tag_t *scripts,
			       const hb_tag_t *languages,
			       const hb_tag_t *features,
			       hb_set_t       *feature_indexes /* OUT */);

HB_EXTERN void
hb_ot_layout_collect_features_map (hb_face_t      *face,
				   hb_tag_t        table_tag,
				   unsigned        script_index,
				   unsigned        language_index,
				   hb_map_t       *feature_map /* OUT */);

HB_EXTERN void
hb_ot_layout_collect_lookups (hb_face_t      *face,
			      hb_tag_t        table_tag,
			      const hb_tag_t *scripts,
			      const hb_tag_t *languages,
			      const hb_tag_t *features,
			      hb_set_t       *lookup_indexes /* OUT */);

HB_EXTERN void
hb_ot_layout_lookup_collect_glyphs (hb_face_t    *face,
				    hb_tag_t      table_tag,
				    unsigned int  lookup_index,
				    hb_set_t     *glyphs_before, /* OUT.  May be NULL */
				    hb_set_t     *glyphs_input,  /* OUT.  May be NULL */
				    hb_set_t     *glyphs_after,  /* OUT.  May be NULL */
				    hb_set_t     *glyphs_output  /* OUT.  May be NULL */);


/* Variations support */

HB_EXTERN hb_bool_t
hb_ot_layout_table_find_feature_variations (hb_face_t    *face,
					    hb_tag_t      table_tag,
					    const int    *coords,
					    unsigned int  num_coords,
					    unsigned int *variations_index /* out */);

HB_EXTERN unsigned int
hb_ot_layout_feature_with_variations_get_lookups (hb_face_t    *face,
						  hb_tag_t      table_tag,
						  unsigned int  feature_index,
						  unsigned int  variations_index,
						  unsigned int  start_offset,
						  unsigned int *lookup_count /* IN/OUT */,
						  unsigned int *lookup_indexes /* OUT */);


/*
 * GSUB
 */

HB_EXTERN hb_bool_t
hb_ot_layout_has_substitution (hb_face_t *face);

HB_EXTERN unsigned
hb_ot_layout_lookup_get_glyph_alternates (hb_face_t      *face,
					  unsigned        lookup_index,
					  hb_codepoint_t  glyph,
					  unsigned        start_offset,
					  unsigned       *alternate_count /* IN/OUT */,
					  hb_codepoint_t *alternate_glyphs /* OUT */);

HB_EXTERN hb_bool_t
hb_ot_layout_lookup_would_substitute (hb_face_t            *face,
				      unsigned int          lookup_index,
				      const hb_codepoint_t *glyphs,
				      unsigned int          glyphs_length,
				      hb_bool_t             zero_context);

HB_EXTERN void
hb_ot_layout_lookup_substitute_closure (hb_face_t    *face,
					unsigned int  lookup_index,
					hb_set_t     *glyphs
					/*TODO , hb_bool_t  inclusive */);

HB_EXTERN void
hb_ot_layout_lookups_substitute_closure (hb_face_t      *face,
					 const hb_set_t *lookups,
					 hb_set_t       *glyphs);


/*
 * GPOS
 */

HB_EXTERN hb_bool_t
hb_ot_layout_has_positioning (hb_face_t *face);

/* Optical 'size' feature info.  Returns true if found.
 * https://docs.microsoft.com/en-us/typography/opentype/spec/features_pt#size */
HB_EXTERN hb_bool_t
hb_ot_layout_get_size_params (hb_face_t       *face,
			      unsigned int    *design_size,       /* OUT.  May be NULL */
			      unsigned int    *subfamily_id,      /* OUT.  May be NULL */
			      hb_ot_name_id_t *subfamily_name_id, /* OUT.  May be NULL */
			      unsigned int    *range_start,       /* OUT.  May be NULL */
			      unsigned int    *range_end          /* OUT.  May be NULL */);

HB_EXTERN hb_position_t
hb_ot_layout_lookup_get_optical_bound (hb_font_t      *font,
				       unsigned        lookup_index,
				       hb_direction_t  direction,
				       hb_codepoint_t  glyph);


/*
 * GSUB/GPOS
 */

HB_EXTERN hb_bool_t
hb_ot_layout_feature_get_name_ids (hb_face_t       *face,
				   hb_tag_t         table_tag,
				   unsigned int     feature_index,
				   hb_ot_name_id_t *label_id             /* OUT.  May be NULL */,
				   hb_ot_name_id_t *tooltip_id           /* OUT.  May be NULL */,
				   hb_ot_name_id_t *sample_id            /* OUT.  May be NULL */,
				   unsigned int    *num_named_parameters /* OUT.  May be NULL */,
				   hb_ot_name_id_t *first_param_id       /* OUT.  May be NULL */);


HB_EXTERN unsigned int
hb_ot_layout_feature_get_characters (hb_face_t      *face,
				     hb_tag_t        table_tag,
				     unsigned int    feature_index,
				     unsigned int    start_offset,
				     unsigned int   *char_count    /* IN/OUT.  May be NULL */,
				     hb_codepoint_t *characters    /* OUT.     May be NULL */);


/*
 * BASE
 */

HB_EXTERN hb_bool_t
hb_ot_layout_get_font_extents (hb_font_t         *font,
			       hb_direction_t     direction,
			       hb_tag_t           script_tag,
			       hb_tag_t           language_tag,
			       hb_font_extents_t *extents);

HB_EXTERN hb_bool_t
hb_ot_layout_get_font_extents2 (hb_font_t         *font,
				hb_direction_t     direction,
				hb_script_t        script,
				hb_language_t      language,
				hb_font_extents_t *extents);

/**
 * hb_ot_layout_baseline_tag_t:
 * @HB_OT_LAYOUT_BASELINE_TAG_ROMAN: The baseline used by alphabetic scripts such as Latin, Cyrillic and Greek.
 * In vertical writing mode, the alphabetic baseline for characters rotated 90 degrees clockwise.
 * (This would not apply to alphabetic characters that remain upright in vertical writing mode, since these
 * characters are not rotated.)
 * @HB_OT_LAYOUT_BASELINE_TAG_HANGING: The hanging baseline. In horizontal direction, this is the horizontal
 * line from which syllables seem, to hang in Tibetan and other similar scripts. In vertical writing mode,
 * for Tibetan (or some other similar script) characters rotated 90 degrees clockwise.
 * @HB_OT_LAYOUT_BASELINE_TAG_IDEO_FACE_BOTTOM_OR_LEFT: Ideographic character face bottom or left edge,
 * if the direction is horizontal or vertical, respectively.
 * @HB_OT_LAYOUT_BASELINE_TAG_IDEO_FACE_TOP_OR_RIGHT: Ideographic character face top or right edge,
 * if the direction is horizontal or vertical, respectively.
 * @HB_OT_LAYOUT_BASELINE_TAG_IDEO_FACE_CENTRAL: The center of the ideographic character face. Since: 4.0.0
 * @HB_OT_LAYOUT_BASELINE_TAG_IDEO_EMBOX_BOTTOM_OR_LEFT: Ideographic em-box bottom or left edge,
 * if the direction is horizontal or vertical, respectively.
 * @HB_OT_LAYOUT_BASELINE_TAG_IDEO_EMBOX_TOP_OR_RIGHT: Ideographic em-box top or right edge baseline,
 * @HB_OT_LAYOUT_BASELINE_TAG_IDEO_EMBOX_CENTRAL: The center of the ideographic em-box. Since: 4.0.0
 * if the direction is horizontal or vertical, respectively.
 * @HB_OT_LAYOUT_BASELINE_TAG_MATH: The baseline about which mathematical characters are centered.
 * In vertical writing mode when mathematical characters rotated 90 degrees clockwise, are centered.
 *
 * Baseline tags from [Baseline Tags](https://docs.microsoft.com/en-us/typography/opentype/spec/baselinetags) registry.
 *
 * Since: 2.6.0
 */
typedef enum {
  HB_OT_LAYOUT_BASELINE_TAG_ROMAN			= HB_TAG ('r','o','m','n'),
  HB_OT_LAYOUT_BASELINE_TAG_HANGING			= HB_TAG ('h','a','n','g'),
  HB_OT_LAYOUT_BASELINE_TAG_IDEO_FACE_BOTTOM_OR_LEFT	= HB_TAG ('i','c','f','b'),
  HB_OT_LAYOUT_BASELINE_TAG_IDEO_FACE_TOP_OR_RIGHT	= HB_TAG ('i','c','f','t'),
  HB_OT_LAYOUT_BASELINE_TAG_IDEO_FACE_CENTRAL		= HB_TAG ('I','c','f','c'),
  HB_OT_LAYOUT_BASELINE_TAG_IDEO_EMBOX_BOTTOM_OR_LEFT	= HB_TAG ('i','d','e','o'),
  HB_OT_LAYOUT_BASELINE_TAG_IDEO_EMBOX_TOP_OR_RIGHT	= HB_TAG ('i','d','t','p'),
  HB_OT_LAYOUT_BASELINE_TAG_IDEO_EMBOX_CENTRAL		= HB_TAG ('I','d','c','e'),
  HB_OT_LAYOUT_BASELINE_TAG_MATH			= HB_TAG ('m','a','t','h'),

  /*< private >*/
  _HB_OT_LAYOUT_BASELINE_TAG_MAX_VALUE = HB_TAG_MAX_SIGNED /*< skip >*/
} hb_ot_layout_baseline_tag_t;

HB_EXTERN hb_ot_layout_baseline_tag_t
hb_ot_layout_get_horizontal_baseline_tag_for_script (hb_script_t script);

HB_EXTERN hb_bool_t
hb_ot_layout_get_baseline (hb_font_t                   *font,
			   hb_ot_layout_baseline_tag_t  baseline_tag,
			   hb_direction_t               direction,
			   hb_tag_t                     script_tag,
			   hb_tag_t                     language_tag,
			   hb_position_t               *coord        /* OUT.  May be NULL. */);

HB_EXTERN hb_bool_t
hb_ot_layout_get_baseline2 (hb_font_t                   *font,
			    hb_ot_layout_baseline_tag_t  baseline_tag,
			    hb_direction_t               direction,
			    hb_script_t                  script,
			    hb_language_t                language,
			    hb_position_t               *coord        /* OUT.  May be NULL. */);

HB_EXTERN void
hb_ot_layout_get_baseline_with_fallback (hb_font_t                   *font,
					 hb_ot_layout_baseline_tag_t  baseline_tag,
					 hb_direction_t               direction,
					 hb_tag_t                     script_tag,
					 hb_tag_t                     language_tag,
					 hb_position_t               *coord        /* OUT */);

HB_EXTERN void
hb_ot_layout_get_baseline_with_fallback2 (hb_font_t                   *font,
					  hb_ot_layout_baseline_tag_t  baseline_tag,
					  hb_direction_t               direction,
					  hb_script_t                  script,
					  hb_language_t                language,
					  hb_position_t               *coord        /* OUT */);

HB_END_DECLS

#endif /* HB_OT_LAYOUT_H */
