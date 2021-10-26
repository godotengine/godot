/*
 * Copyright © 2017  Google, Inc.
 * Copyright © 2018  Ebrahim Byagowi
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

#include "hb.hh"

#include "hb-aat-layout.hh"
#include "hb-aat-layout-ankr-table.hh"
#include "hb-aat-layout-bsln-table.hh" // Just so we compile it; unused otherwise.
#include "hb-aat-layout-feat-table.hh"
#include "hb-aat-layout-just-table.hh" // Just so we compile it; unused otherwise.
#include "hb-aat-layout-kerx-table.hh"
#include "hb-aat-layout-morx-table.hh"
#include "hb-aat-layout-trak-table.hh"
#include "hb-aat-ltag-table.hh"


/*
 * hb_aat_apply_context_t
 */

/* Note: This context is used for kerning, even without AAT, hence the condition. */
#if !defined(HB_NO_AAT) || !defined(HB_NO_OT_KERN)

AAT::hb_aat_apply_context_t::hb_aat_apply_context_t (const hb_ot_shape_plan_t *plan_,
						     hb_font_t *font_,
						     hb_buffer_t *buffer_,
						     hb_blob_t *blob) :
						       plan (plan_),
						       font (font_),
						       face (font->face),
						       buffer (buffer_),
						       sanitizer (),
						       ankr_table (&Null (AAT::ankr)),
						       gdef_table (face->table.GDEF->table),
						       lookup_index (0)
{
  sanitizer.init (blob);
  sanitizer.set_num_glyphs (face->get_num_glyphs ());
  sanitizer.start_processing ();
  sanitizer.set_max_ops (HB_SANITIZE_MAX_OPS_MAX);
}

AAT::hb_aat_apply_context_t::~hb_aat_apply_context_t ()
{ sanitizer.end_processing (); }

void
AAT::hb_aat_apply_context_t::set_ankr_table (const AAT::ankr *ankr_table_)
{ ankr_table = ankr_table_; }

#endif


/**
 * SECTION:hb-aat-layout
 * @title: hb-aat-layout
 * @short_description: Apple Advanced Typography Layout
 * @include: hb-aat.h
 *
 * Functions for querying AAT Layout features in the font face.
 *
 * HarfBuzz supports all of the AAT tables used to implement shaping. Other
 * AAT tables and their associated features are not supported.
 **/


#if !defined(HB_NO_AAT) || defined(HAVE_CORETEXT)

/* Mapping from OpenType feature tags to AAT feature names and selectors.
 *
 * Table data courtesy of Apple.  Converted from mnemonics to integers
 * when moving to this file. */
static const hb_aat_feature_mapping_t feature_mappings[] =
{
  {HB_TAG ('a','f','r','c'), HB_AAT_LAYOUT_FEATURE_TYPE_FRACTIONS,               HB_AAT_LAYOUT_FEATURE_SELECTOR_VERTICAL_FRACTIONS,             HB_AAT_LAYOUT_FEATURE_SELECTOR_NO_FRACTIONS},
  {HB_TAG ('c','2','p','c'), HB_AAT_LAYOUT_FEATURE_TYPE_UPPER_CASE,              HB_AAT_LAYOUT_FEATURE_SELECTOR_UPPER_CASE_PETITE_CAPS,         HB_AAT_LAYOUT_FEATURE_SELECTOR_DEFAULT_UPPER_CASE},
  {HB_TAG ('c','2','s','c'), HB_AAT_LAYOUT_FEATURE_TYPE_UPPER_CASE,              HB_AAT_LAYOUT_FEATURE_SELECTOR_UPPER_CASE_SMALL_CAPS,          HB_AAT_LAYOUT_FEATURE_SELECTOR_DEFAULT_UPPER_CASE},
  {HB_TAG ('c','a','l','t'), HB_AAT_LAYOUT_FEATURE_TYPE_CONTEXTUAL_ALTERNATIVES, HB_AAT_LAYOUT_FEATURE_SELECTOR_CONTEXTUAL_ALTERNATES_ON,       HB_AAT_LAYOUT_FEATURE_SELECTOR_CONTEXTUAL_ALTERNATES_OFF},
  {HB_TAG ('c','a','s','e'), HB_AAT_LAYOUT_FEATURE_TYPE_CASE_SENSITIVE_LAYOUT,   HB_AAT_LAYOUT_FEATURE_SELECTOR_CASE_SENSITIVE_LAYOUT_ON,       HB_AAT_LAYOUT_FEATURE_SELECTOR_CASE_SENSITIVE_LAYOUT_OFF},
  {HB_TAG ('c','l','i','g'), HB_AAT_LAYOUT_FEATURE_TYPE_LIGATURES,               HB_AAT_LAYOUT_FEATURE_SELECTOR_CONTEXTUAL_LIGATURES_ON,        HB_AAT_LAYOUT_FEATURE_SELECTOR_CONTEXTUAL_LIGATURES_OFF},
  {HB_TAG ('c','p','s','p'), HB_AAT_LAYOUT_FEATURE_TYPE_CASE_SENSITIVE_LAYOUT,   HB_AAT_LAYOUT_FEATURE_SELECTOR_CASE_SENSITIVE_SPACING_ON,      HB_AAT_LAYOUT_FEATURE_SELECTOR_CASE_SENSITIVE_SPACING_OFF},
  {HB_TAG ('c','s','w','h'), HB_AAT_LAYOUT_FEATURE_TYPE_CONTEXTUAL_ALTERNATIVES, HB_AAT_LAYOUT_FEATURE_SELECTOR_CONTEXTUAL_SWASH_ALTERNATES_ON, HB_AAT_LAYOUT_FEATURE_SELECTOR_CONTEXTUAL_SWASH_ALTERNATES_OFF},
  {HB_TAG ('d','l','i','g'), HB_AAT_LAYOUT_FEATURE_TYPE_LIGATURES,               HB_AAT_LAYOUT_FEATURE_SELECTOR_RARE_LIGATURES_ON,              HB_AAT_LAYOUT_FEATURE_SELECTOR_RARE_LIGATURES_OFF},
  {HB_TAG ('e','x','p','t'), HB_AAT_LAYOUT_FEATURE_TYPE_CHARACTER_SHAPE,         HB_AAT_LAYOUT_FEATURE_SELECTOR_EXPERT_CHARACTERS,              (hb_aat_layout_feature_selector_t) 16},
  {HB_TAG ('f','r','a','c'), HB_AAT_LAYOUT_FEATURE_TYPE_FRACTIONS,               HB_AAT_LAYOUT_FEATURE_SELECTOR_DIAGONAL_FRACTIONS,             HB_AAT_LAYOUT_FEATURE_SELECTOR_NO_FRACTIONS},
  {HB_TAG ('f','w','i','d'), HB_AAT_LAYOUT_FEATURE_TYPE_TEXT_SPACING,            HB_AAT_LAYOUT_FEATURE_SELECTOR_MONOSPACED_TEXT,                (hb_aat_layout_feature_selector_t) 7},
  {HB_TAG ('h','a','l','t'), HB_AAT_LAYOUT_FEATURE_TYPE_TEXT_SPACING,            HB_AAT_LAYOUT_FEATURE_SELECTOR_ALT_HALF_WIDTH_TEXT,            (hb_aat_layout_feature_selector_t) 7},
  {HB_TAG ('h','i','s','t'), HB_AAT_LAYOUT_FEATURE_TYPE_LIGATURES,               HB_AAT_LAYOUT_FEATURE_SELECTOR_HISTORICAL_LIGATURES_ON,        HB_AAT_LAYOUT_FEATURE_SELECTOR_HISTORICAL_LIGATURES_OFF},
  {HB_TAG ('h','k','n','a'), HB_AAT_LAYOUT_FEATURE_TYPE_ALTERNATE_KANA,          HB_AAT_LAYOUT_FEATURE_SELECTOR_ALTERNATE_HORIZ_KANA_ON,        HB_AAT_LAYOUT_FEATURE_SELECTOR_ALTERNATE_HORIZ_KANA_OFF},
  {HB_TAG ('h','l','i','g'), HB_AAT_LAYOUT_FEATURE_TYPE_LIGATURES,               HB_AAT_LAYOUT_FEATURE_SELECTOR_HISTORICAL_LIGATURES_ON,        HB_AAT_LAYOUT_FEATURE_SELECTOR_HISTORICAL_LIGATURES_OFF},
  {HB_TAG ('h','n','g','l'), HB_AAT_LAYOUT_FEATURE_TYPE_TRANSLITERATION,         HB_AAT_LAYOUT_FEATURE_SELECTOR_HANJA_TO_HANGUL,                HB_AAT_LAYOUT_FEATURE_SELECTOR_NO_TRANSLITERATION},
  {HB_TAG ('h','o','j','o'), HB_AAT_LAYOUT_FEATURE_TYPE_CHARACTER_SHAPE,         HB_AAT_LAYOUT_FEATURE_SELECTOR_HOJO_CHARACTERS,                (hb_aat_layout_feature_selector_t) 16},
  {HB_TAG ('h','w','i','d'), HB_AAT_LAYOUT_FEATURE_TYPE_TEXT_SPACING,            HB_AAT_LAYOUT_FEATURE_SELECTOR_HALF_WIDTH_TEXT,                (hb_aat_layout_feature_selector_t) 7},
  {HB_TAG ('i','t','a','l'), HB_AAT_LAYOUT_FEATURE_TYPE_ITALIC_CJK_ROMAN,        HB_AAT_LAYOUT_FEATURE_SELECTOR_CJK_ITALIC_ROMAN_ON,            HB_AAT_LAYOUT_FEATURE_SELECTOR_CJK_ITALIC_ROMAN_OFF},
  {HB_TAG ('j','p','0','4'), HB_AAT_LAYOUT_FEATURE_TYPE_CHARACTER_SHAPE,         HB_AAT_LAYOUT_FEATURE_SELECTOR_JIS2004_CHARACTERS,             (hb_aat_layout_feature_selector_t) 16},
  {HB_TAG ('j','p','7','8'), HB_AAT_LAYOUT_FEATURE_TYPE_CHARACTER_SHAPE,         HB_AAT_LAYOUT_FEATURE_SELECTOR_JIS1978_CHARACTERS,             (hb_aat_layout_feature_selector_t) 16},
  {HB_TAG ('j','p','8','3'), HB_AAT_LAYOUT_FEATURE_TYPE_CHARACTER_SHAPE,         HB_AAT_LAYOUT_FEATURE_SELECTOR_JIS1983_CHARACTERS,             (hb_aat_layout_feature_selector_t) 16},
  {HB_TAG ('j','p','9','0'), HB_AAT_LAYOUT_FEATURE_TYPE_CHARACTER_SHAPE,         HB_AAT_LAYOUT_FEATURE_SELECTOR_JIS1990_CHARACTERS,             (hb_aat_layout_feature_selector_t) 16},
  {HB_TAG ('l','i','g','a'), HB_AAT_LAYOUT_FEATURE_TYPE_LIGATURES,               HB_AAT_LAYOUT_FEATURE_SELECTOR_COMMON_LIGATURES_ON,            HB_AAT_LAYOUT_FEATURE_SELECTOR_COMMON_LIGATURES_OFF},
  {HB_TAG ('l','n','u','m'), HB_AAT_LAYOUT_FEATURE_TYPE_NUMBER_CASE,             HB_AAT_LAYOUT_FEATURE_SELECTOR_UPPER_CASE_NUMBERS,             (hb_aat_layout_feature_selector_t) 2},
  {HB_TAG ('m','g','r','k'), HB_AAT_LAYOUT_FEATURE_TYPE_MATHEMATICAL_EXTRAS,     HB_AAT_LAYOUT_FEATURE_SELECTOR_MATHEMATICAL_GREEK_ON,          HB_AAT_LAYOUT_FEATURE_SELECTOR_MATHEMATICAL_GREEK_OFF},
  {HB_TAG ('n','l','c','k'), HB_AAT_LAYOUT_FEATURE_TYPE_CHARACTER_SHAPE,         HB_AAT_LAYOUT_FEATURE_SELECTOR_NLCCHARACTERS,                  (hb_aat_layout_feature_selector_t) 16},
  {HB_TAG ('o','n','u','m'), HB_AAT_LAYOUT_FEATURE_TYPE_NUMBER_CASE,             HB_AAT_LAYOUT_FEATURE_SELECTOR_LOWER_CASE_NUMBERS,             (hb_aat_layout_feature_selector_t) 2},
  {HB_TAG ('o','r','d','n'), HB_AAT_LAYOUT_FEATURE_TYPE_VERTICAL_POSITION,       HB_AAT_LAYOUT_FEATURE_SELECTOR_ORDINALS,                       HB_AAT_LAYOUT_FEATURE_SELECTOR_NORMAL_POSITION},
  {HB_TAG ('p','a','l','t'), HB_AAT_LAYOUT_FEATURE_TYPE_TEXT_SPACING,            HB_AAT_LAYOUT_FEATURE_SELECTOR_ALT_PROPORTIONAL_TEXT,          (hb_aat_layout_feature_selector_t) 7},
  {HB_TAG ('p','c','a','p'), HB_AAT_LAYOUT_FEATURE_TYPE_LOWER_CASE,              HB_AAT_LAYOUT_FEATURE_SELECTOR_LOWER_CASE_PETITE_CAPS,         HB_AAT_LAYOUT_FEATURE_SELECTOR_DEFAULT_LOWER_CASE},
  {HB_TAG ('p','k','n','a'), HB_AAT_LAYOUT_FEATURE_TYPE_TEXT_SPACING,            HB_AAT_LAYOUT_FEATURE_SELECTOR_PROPORTIONAL_TEXT,              (hb_aat_layout_feature_selector_t) 7},
  {HB_TAG ('p','n','u','m'), HB_AAT_LAYOUT_FEATURE_TYPE_NUMBER_SPACING,          HB_AAT_LAYOUT_FEATURE_SELECTOR_PROPORTIONAL_NUMBERS,           (hb_aat_layout_feature_selector_t) 4},
  {HB_TAG ('p','w','i','d'), HB_AAT_LAYOUT_FEATURE_TYPE_TEXT_SPACING,            HB_AAT_LAYOUT_FEATURE_SELECTOR_PROPORTIONAL_TEXT,              (hb_aat_layout_feature_selector_t) 7},
  {HB_TAG ('q','w','i','d'), HB_AAT_LAYOUT_FEATURE_TYPE_TEXT_SPACING,            HB_AAT_LAYOUT_FEATURE_SELECTOR_QUARTER_WIDTH_TEXT,             (hb_aat_layout_feature_selector_t) 7},
  {HB_TAG ('r','u','b','y'), HB_AAT_LAYOUT_FEATURE_TYPE_RUBY_KANA,               HB_AAT_LAYOUT_FEATURE_SELECTOR_RUBY_KANA_ON,                   HB_AAT_LAYOUT_FEATURE_SELECTOR_RUBY_KANA_OFF},
  {HB_TAG ('s','i','n','f'), HB_AAT_LAYOUT_FEATURE_TYPE_VERTICAL_POSITION,       HB_AAT_LAYOUT_FEATURE_SELECTOR_SCIENTIFIC_INFERIORS,           HB_AAT_LAYOUT_FEATURE_SELECTOR_NORMAL_POSITION},
  {HB_TAG ('s','m','c','p'), HB_AAT_LAYOUT_FEATURE_TYPE_LOWER_CASE,              HB_AAT_LAYOUT_FEATURE_SELECTOR_LOWER_CASE_SMALL_CAPS,          HB_AAT_LAYOUT_FEATURE_SELECTOR_DEFAULT_LOWER_CASE},
  {HB_TAG ('s','m','p','l'), HB_AAT_LAYOUT_FEATURE_TYPE_CHARACTER_SHAPE,         HB_AAT_LAYOUT_FEATURE_SELECTOR_SIMPLIFIED_CHARACTERS,          (hb_aat_layout_feature_selector_t) 16},
  {HB_TAG ('s','s','0','1'), HB_AAT_LAYOUT_FEATURE_TYPE_STYLISTIC_ALTERNATIVES,  HB_AAT_LAYOUT_FEATURE_SELECTOR_STYLISTIC_ALT_ONE_ON,           HB_AAT_LAYOUT_FEATURE_SELECTOR_STYLISTIC_ALT_ONE_OFF},
  {HB_TAG ('s','s','0','2'), HB_AAT_LAYOUT_FEATURE_TYPE_STYLISTIC_ALTERNATIVES,  HB_AAT_LAYOUT_FEATURE_SELECTOR_STYLISTIC_ALT_TWO_ON,           HB_AAT_LAYOUT_FEATURE_SELECTOR_STYLISTIC_ALT_TWO_OFF},
  {HB_TAG ('s','s','0','3'), HB_AAT_LAYOUT_FEATURE_TYPE_STYLISTIC_ALTERNATIVES,  HB_AAT_LAYOUT_FEATURE_SELECTOR_STYLISTIC_ALT_THREE_ON,         HB_AAT_LAYOUT_FEATURE_SELECTOR_STYLISTIC_ALT_THREE_OFF},
  {HB_TAG ('s','s','0','4'), HB_AAT_LAYOUT_FEATURE_TYPE_STYLISTIC_ALTERNATIVES,  HB_AAT_LAYOUT_FEATURE_SELECTOR_STYLISTIC_ALT_FOUR_ON,          HB_AAT_LAYOUT_FEATURE_SELECTOR_STYLISTIC_ALT_FOUR_OFF},
  {HB_TAG ('s','s','0','5'), HB_AAT_LAYOUT_FEATURE_TYPE_STYLISTIC_ALTERNATIVES,  HB_AAT_LAYOUT_FEATURE_SELECTOR_STYLISTIC_ALT_FIVE_ON,          HB_AAT_LAYOUT_FEATURE_SELECTOR_STYLISTIC_ALT_FIVE_OFF},
  {HB_TAG ('s','s','0','6'), HB_AAT_LAYOUT_FEATURE_TYPE_STYLISTIC_ALTERNATIVES,  HB_AAT_LAYOUT_FEATURE_SELECTOR_STYLISTIC_ALT_SIX_ON,           HB_AAT_LAYOUT_FEATURE_SELECTOR_STYLISTIC_ALT_SIX_OFF},
  {HB_TAG ('s','s','0','7'), HB_AAT_LAYOUT_FEATURE_TYPE_STYLISTIC_ALTERNATIVES,  HB_AAT_LAYOUT_FEATURE_SELECTOR_STYLISTIC_ALT_SEVEN_ON,         HB_AAT_LAYOUT_FEATURE_SELECTOR_STYLISTIC_ALT_SEVEN_OFF},
  {HB_TAG ('s','s','0','8'), HB_AAT_LAYOUT_FEATURE_TYPE_STYLISTIC_ALTERNATIVES,  HB_AAT_LAYOUT_FEATURE_SELECTOR_STYLISTIC_ALT_EIGHT_ON,         HB_AAT_LAYOUT_FEATURE_SELECTOR_STYLISTIC_ALT_EIGHT_OFF},
  {HB_TAG ('s','s','0','9'), HB_AAT_LAYOUT_FEATURE_TYPE_STYLISTIC_ALTERNATIVES,  HB_AAT_LAYOUT_FEATURE_SELECTOR_STYLISTIC_ALT_NINE_ON,          HB_AAT_LAYOUT_FEATURE_SELECTOR_STYLISTIC_ALT_NINE_OFF},
  {HB_TAG ('s','s','1','0'), HB_AAT_LAYOUT_FEATURE_TYPE_STYLISTIC_ALTERNATIVES,  HB_AAT_LAYOUT_FEATURE_SELECTOR_STYLISTIC_ALT_TEN_ON,           HB_AAT_LAYOUT_FEATURE_SELECTOR_STYLISTIC_ALT_TEN_OFF},
  {HB_TAG ('s','s','1','1'), HB_AAT_LAYOUT_FEATURE_TYPE_STYLISTIC_ALTERNATIVES,  HB_AAT_LAYOUT_FEATURE_SELECTOR_STYLISTIC_ALT_ELEVEN_ON,        HB_AAT_LAYOUT_FEATURE_SELECTOR_STYLISTIC_ALT_ELEVEN_OFF},
  {HB_TAG ('s','s','1','2'), HB_AAT_LAYOUT_FEATURE_TYPE_STYLISTIC_ALTERNATIVES,  HB_AAT_LAYOUT_FEATURE_SELECTOR_STYLISTIC_ALT_TWELVE_ON,        HB_AAT_LAYOUT_FEATURE_SELECTOR_STYLISTIC_ALT_TWELVE_OFF},
  {HB_TAG ('s','s','1','3'), HB_AAT_LAYOUT_FEATURE_TYPE_STYLISTIC_ALTERNATIVES,  HB_AAT_LAYOUT_FEATURE_SELECTOR_STYLISTIC_ALT_THIRTEEN_ON,      HB_AAT_LAYOUT_FEATURE_SELECTOR_STYLISTIC_ALT_THIRTEEN_OFF},
  {HB_TAG ('s','s','1','4'), HB_AAT_LAYOUT_FEATURE_TYPE_STYLISTIC_ALTERNATIVES,  HB_AAT_LAYOUT_FEATURE_SELECTOR_STYLISTIC_ALT_FOURTEEN_ON,      HB_AAT_LAYOUT_FEATURE_SELECTOR_STYLISTIC_ALT_FOURTEEN_OFF},
  {HB_TAG ('s','s','1','5'), HB_AAT_LAYOUT_FEATURE_TYPE_STYLISTIC_ALTERNATIVES,  HB_AAT_LAYOUT_FEATURE_SELECTOR_STYLISTIC_ALT_FIFTEEN_ON,       HB_AAT_LAYOUT_FEATURE_SELECTOR_STYLISTIC_ALT_FIFTEEN_OFF},
  {HB_TAG ('s','s','1','6'), HB_AAT_LAYOUT_FEATURE_TYPE_STYLISTIC_ALTERNATIVES,  HB_AAT_LAYOUT_FEATURE_SELECTOR_STYLISTIC_ALT_SIXTEEN_ON,       HB_AAT_LAYOUT_FEATURE_SELECTOR_STYLISTIC_ALT_SIXTEEN_OFF},
  {HB_TAG ('s','s','1','7'), HB_AAT_LAYOUT_FEATURE_TYPE_STYLISTIC_ALTERNATIVES,  HB_AAT_LAYOUT_FEATURE_SELECTOR_STYLISTIC_ALT_SEVENTEEN_ON,     HB_AAT_LAYOUT_FEATURE_SELECTOR_STYLISTIC_ALT_SEVENTEEN_OFF},
  {HB_TAG ('s','s','1','8'), HB_AAT_LAYOUT_FEATURE_TYPE_STYLISTIC_ALTERNATIVES,  HB_AAT_LAYOUT_FEATURE_SELECTOR_STYLISTIC_ALT_EIGHTEEN_ON,      HB_AAT_LAYOUT_FEATURE_SELECTOR_STYLISTIC_ALT_EIGHTEEN_OFF},
  {HB_TAG ('s','s','1','9'), HB_AAT_LAYOUT_FEATURE_TYPE_STYLISTIC_ALTERNATIVES,  HB_AAT_LAYOUT_FEATURE_SELECTOR_STYLISTIC_ALT_NINETEEN_ON,      HB_AAT_LAYOUT_FEATURE_SELECTOR_STYLISTIC_ALT_NINETEEN_OFF},
  {HB_TAG ('s','s','2','0'), HB_AAT_LAYOUT_FEATURE_TYPE_STYLISTIC_ALTERNATIVES,  HB_AAT_LAYOUT_FEATURE_SELECTOR_STYLISTIC_ALT_TWENTY_ON,        HB_AAT_LAYOUT_FEATURE_SELECTOR_STYLISTIC_ALT_TWENTY_OFF},
  {HB_TAG ('s','u','b','s'), HB_AAT_LAYOUT_FEATURE_TYPE_VERTICAL_POSITION,       HB_AAT_LAYOUT_FEATURE_SELECTOR_INFERIORS,                      HB_AAT_LAYOUT_FEATURE_SELECTOR_NORMAL_POSITION},
  {HB_TAG ('s','u','p','s'), HB_AAT_LAYOUT_FEATURE_TYPE_VERTICAL_POSITION,       HB_AAT_LAYOUT_FEATURE_SELECTOR_SUPERIORS,                      HB_AAT_LAYOUT_FEATURE_SELECTOR_NORMAL_POSITION},
  {HB_TAG ('s','w','s','h'), HB_AAT_LAYOUT_FEATURE_TYPE_CONTEXTUAL_ALTERNATIVES, HB_AAT_LAYOUT_FEATURE_SELECTOR_SWASH_ALTERNATES_ON,            HB_AAT_LAYOUT_FEATURE_SELECTOR_SWASH_ALTERNATES_OFF},
  {HB_TAG ('t','i','t','l'), HB_AAT_LAYOUT_FEATURE_TYPE_STYLE_OPTIONS,           HB_AAT_LAYOUT_FEATURE_SELECTOR_TITLING_CAPS,                   HB_AAT_LAYOUT_FEATURE_SELECTOR_NO_STYLE_OPTIONS},
  {HB_TAG ('t','n','a','m'), HB_AAT_LAYOUT_FEATURE_TYPE_CHARACTER_SHAPE,         HB_AAT_LAYOUT_FEATURE_SELECTOR_TRADITIONAL_NAMES_CHARACTERS,   (hb_aat_layout_feature_selector_t) 16},
  {HB_TAG ('t','n','u','m'), HB_AAT_LAYOUT_FEATURE_TYPE_NUMBER_SPACING,          HB_AAT_LAYOUT_FEATURE_SELECTOR_MONOSPACED_NUMBERS,             (hb_aat_layout_feature_selector_t) 4},
  {HB_TAG ('t','r','a','d'), HB_AAT_LAYOUT_FEATURE_TYPE_CHARACTER_SHAPE,         HB_AAT_LAYOUT_FEATURE_SELECTOR_TRADITIONAL_CHARACTERS,         (hb_aat_layout_feature_selector_t) 16},
  {HB_TAG ('t','w','i','d'), HB_AAT_LAYOUT_FEATURE_TYPE_TEXT_SPACING,            HB_AAT_LAYOUT_FEATURE_SELECTOR_THIRD_WIDTH_TEXT,               (hb_aat_layout_feature_selector_t) 7},
  {HB_TAG ('u','n','i','c'), HB_AAT_LAYOUT_FEATURE_TYPE_LETTER_CASE,             (hb_aat_layout_feature_selector_t) 14,                 (hb_aat_layout_feature_selector_t) 15},
  {HB_TAG ('v','a','l','t'), HB_AAT_LAYOUT_FEATURE_TYPE_TEXT_SPACING,            HB_AAT_LAYOUT_FEATURE_SELECTOR_ALT_PROPORTIONAL_TEXT,          (hb_aat_layout_feature_selector_t) 7},
  {HB_TAG ('v','e','r','t'), HB_AAT_LAYOUT_FEATURE_TYPE_VERTICAL_SUBSTITUTION,   HB_AAT_LAYOUT_FEATURE_SELECTOR_SUBSTITUTE_VERTICAL_FORMS_ON,   HB_AAT_LAYOUT_FEATURE_SELECTOR_SUBSTITUTE_VERTICAL_FORMS_OFF},
  {HB_TAG ('v','h','a','l'), HB_AAT_LAYOUT_FEATURE_TYPE_TEXT_SPACING,            HB_AAT_LAYOUT_FEATURE_SELECTOR_ALT_HALF_WIDTH_TEXT,            (hb_aat_layout_feature_selector_t) 7},
  {HB_TAG ('v','k','n','a'), HB_AAT_LAYOUT_FEATURE_TYPE_ALTERNATE_KANA,          HB_AAT_LAYOUT_FEATURE_SELECTOR_ALTERNATE_VERT_KANA_ON,         HB_AAT_LAYOUT_FEATURE_SELECTOR_ALTERNATE_VERT_KANA_OFF},
  {HB_TAG ('v','p','a','l'), HB_AAT_LAYOUT_FEATURE_TYPE_TEXT_SPACING,            HB_AAT_LAYOUT_FEATURE_SELECTOR_ALT_PROPORTIONAL_TEXT,          (hb_aat_layout_feature_selector_t) 7},
  {HB_TAG ('v','r','t','2'), HB_AAT_LAYOUT_FEATURE_TYPE_VERTICAL_SUBSTITUTION,   HB_AAT_LAYOUT_FEATURE_SELECTOR_SUBSTITUTE_VERTICAL_FORMS_ON,   HB_AAT_LAYOUT_FEATURE_SELECTOR_SUBSTITUTE_VERTICAL_FORMS_OFF},
  {HB_TAG ('z','e','r','o'), HB_AAT_LAYOUT_FEATURE_TYPE_TYPOGRAPHIC_EXTRAS,      HB_AAT_LAYOUT_FEATURE_SELECTOR_SLASHED_ZERO_ON,                HB_AAT_LAYOUT_FEATURE_SELECTOR_SLASHED_ZERO_OFF},
};

/**
 * hb_aat_layout_find_feature_mapping:
 * @tag: The requested #hb_tag_t feature tag
 *
 * Fetches the AAT feature-and-selector combination that corresponds
 * to a given OpenType feature tag.
 *
 * Return value: the AAT features and selectors corresponding to the
 * OpenType feature tag queried
 *
 **/
const hb_aat_feature_mapping_t *
hb_aat_layout_find_feature_mapping (hb_tag_t tag)
{
  return hb_sorted_array (feature_mappings).bsearch (tag);
}
#endif


#ifndef HB_NO_AAT

/*
 * mort/morx/kerx/trak
 */


void
hb_aat_layout_compile_map (const hb_aat_map_builder_t *mapper,
			   hb_aat_map_t *map)
{
  const AAT::morx& morx = *mapper->face->table.morx;
  if (morx.has_data ())
  {
    morx.compile_flags (mapper, map);
    return;
  }

  const AAT::mort& mort = *mapper->face->table.mort;
  if (mort.has_data ())
  {
    mort.compile_flags (mapper, map);
    return;
  }
}


/**
 * hb_aat_layout_has_substitution:
 * @face: #hb_face_t to work upon
 *
 * Tests whether the specified face includes any substitutions in the
 * `morx` or `mort` tables.
 *
 * <note>Note: does not examine the `GSUB` table.</note>
 *
 * Return value: %true if data found, %false otherwise
 *
 * Since: 2.3.0
 */
hb_bool_t
hb_aat_layout_has_substitution (hb_face_t *face)
{
  return face->table.morx->has_data () ||
	 face->table.mort->has_data ();
}

void
hb_aat_layout_substitute (const hb_ot_shape_plan_t *plan,
			  hb_font_t *font,
			  hb_buffer_t *buffer)
{
  hb_blob_t *morx_blob = font->face->table.morx.get_blob ();
  const AAT::morx& morx = *morx_blob->as<AAT::morx> ();
  if (morx.has_data ())
  {
    AAT::hb_aat_apply_context_t c (plan, font, buffer, morx_blob);
    if (!buffer->message (font, "start table morx")) return;
    morx.apply (&c);
    (void) buffer->message (font, "end table morx");
    return;
  }

  hb_blob_t *mort_blob = font->face->table.mort.get_blob ();
  const AAT::mort& mort = *mort_blob->as<AAT::mort> ();
  if (mort.has_data ())
  {
    AAT::hb_aat_apply_context_t c (plan, font, buffer, mort_blob);
    if (!buffer->message (font, "start table mort")) return;
    mort.apply (&c);
    (void) buffer->message (font, "end table mort");
    return;
  }
}

void
hb_aat_layout_zero_width_deleted_glyphs (hb_buffer_t *buffer)
{
  unsigned int count = buffer->len;
  hb_glyph_info_t *info = buffer->info;
  hb_glyph_position_t *pos = buffer->pos;
  for (unsigned int i = 0; i < count; i++)
    if (unlikely (info[i].codepoint == AAT::DELETED_GLYPH))
      pos[i].x_advance = pos[i].y_advance = pos[i].x_offset = pos[i].y_offset = 0;
}

static bool
is_deleted_glyph (const hb_glyph_info_t *info)
{
  return info->codepoint == AAT::DELETED_GLYPH;
}

void
hb_aat_layout_remove_deleted_glyphs (hb_buffer_t *buffer)
{
  hb_ot_layout_delete_glyphs_inplace (buffer, is_deleted_glyph);
}

/**
 * hb_aat_layout_has_positioning:
 * @face: #hb_face_t to work upon
 *
 * Tests whether the specified face includes any positioning information
 * in the `kerx` table.
 *
 * <note>Note: does not examine the `GPOS` table.</note>
 *
 * Return value: %true if data found, %false otherwise
 *
 * Since: 2.3.0
 */
hb_bool_t
hb_aat_layout_has_positioning (hb_face_t *face)
{
  return face->table.kerx->has_data ();
}

void
hb_aat_layout_position (const hb_ot_shape_plan_t *plan,
			hb_font_t *font,
			hb_buffer_t *buffer)
{
  hb_blob_t *kerx_blob = font->face->table.kerx.get_blob ();
  const AAT::kerx& kerx = *kerx_blob->as<AAT::kerx> ();

  AAT::hb_aat_apply_context_t c (plan, font, buffer, kerx_blob);
  if (!buffer->message (font, "start table kerx")) return;
  c.set_ankr_table (font->face->table.ankr.get ());
  kerx.apply (&c);
  (void) buffer->message (font, "end table kerx");
}


/**
 * hb_aat_layout_has_tracking:
 * @face:: #hb_face_t to work upon
 *
 * Tests whether the specified face includes any tracking information
 * in the `trak` table.
 *
 * Return value: %true if data found, %false otherwise
 *
 * Since: 2.3.0
 */
hb_bool_t
hb_aat_layout_has_tracking (hb_face_t *face)
{
  return face->table.trak->has_data ();
}

void
hb_aat_layout_track (const hb_ot_shape_plan_t *plan,
		     hb_font_t *font,
		     hb_buffer_t *buffer)
{
  const AAT::trak& trak = *font->face->table.trak;

  AAT::hb_aat_apply_context_t c (plan, font, buffer);
  trak.apply (&c);
}

/**
 * hb_aat_layout_get_feature_types:
 * @face: #hb_face_t to work upon
 * @start_offset: offset of the first feature type to retrieve
 * @feature_count: (inout) (optional): Input = the maximum number of feature types to return;
 *                 Output = the actual number of feature types returned (may be zero)
 * @features: (out caller-allocates) (array length=feature_count): Array of feature types found
 *
 * Fetches a list of the AAT feature types included in the specified face.
 *
 * Return value: Number of all available feature types.
 *
 * Since: 2.2.0
 */
unsigned int
hb_aat_layout_get_feature_types (hb_face_t                    *face,
				 unsigned int                  start_offset,
				 unsigned int                 *feature_count, /* IN/OUT.  May be NULL. */
				 hb_aat_layout_feature_type_t *features       /* OUT.     May be NULL. */)
{
  return face->table.feat->get_feature_types (start_offset, feature_count, features);
}

/**
 * hb_aat_layout_feature_type_get_name_id:
 * @face: #hb_face_t to work upon
 * @feature_type: The #hb_aat_layout_feature_type_t of the requested feature type
 *
 * Fetches the name identifier of the specified feature type in the face's `name` table.
 *
 * Return value: Name identifier of the requested feature type
 *
 * Since: 2.2.0
 */
hb_ot_name_id_t
hb_aat_layout_feature_type_get_name_id (hb_face_t                    *face,
					hb_aat_layout_feature_type_t  feature_type)
{
  return face->table.feat->get_feature_name_id (feature_type);
}

/**
 * hb_aat_layout_feature_type_get_selector_infos:
 * @face: #hb_face_t to work upon
 * @feature_type: The #hb_aat_layout_feature_type_t of the requested feature type
 * @start_offset: offset of the first feature type to retrieve
 * @selector_count: (inout) (optional): Input = the maximum number of selectors to return;
 *                  Output = the actual number of selectors returned (may be zero)
 * @selectors: (out caller-allocates) (array length=selector_count) (optional):
 *             A buffer pointer. The selectors available for the feature type queries.
 * @default_index: (out) (optional): The index of the feature's default selector, if any
 *
 * Fetches a list of the selectors available for the specified feature in the given face.
 *
 * If upon return, @default_index is set to #HB_AAT_LAYOUT_NO_SELECTOR_INDEX, then
 * the feature type is non-exclusive.  Otherwise, @default_index is the index of
 * the selector that is selected by default.
 *
 * Return value: Number of all available feature selectors
 *
 * Since: 2.2.0
 */
unsigned int
hb_aat_layout_feature_type_get_selector_infos (hb_face_t                             *face,
					       hb_aat_layout_feature_type_t           feature_type,
					       unsigned int                           start_offset,
					       unsigned int                          *selector_count, /* IN/OUT.  May be NULL. */
					       hb_aat_layout_feature_selector_info_t *selectors,      /* OUT.     May be NULL. */
					       unsigned int                          *default_index   /* OUT.     May be NULL. */)
{
  return face->table.feat->get_selector_infos (feature_type, start_offset, selector_count, selectors, default_index);
}


#endif
