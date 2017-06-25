/***************************************************************************/
/*                                                                         */
/*  ftautoh.h                                                              */
/*                                                                         */
/*    FreeType API for controlling the auto-hinter (specification only).   */
/*                                                                         */
/*  Copyright 2012-2017 by                                                 */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


#ifndef FTAUTOH_H_
#define FTAUTOH_H_

#include <ft2build.h>
#include FT_FREETYPE_H

#ifdef FREETYPE_H
#error "freetype.h of FreeType 1 has been loaded!"
#error "Please fix the directory search order for header files"
#error "so that freetype.h of FreeType 2 is found first."
#endif


FT_BEGIN_HEADER


  /**************************************************************************
   *
   * @section:
   *   auto_hinter
   *
   * @title:
   *   The auto-hinter
   *
   * @abstract:
   *   Controlling the auto-hinting module.
   *
   * @description:
   *   While FreeType's auto-hinter doesn't expose API functions by itself,
   *   it is possible to control its behaviour with @FT_Property_Set and
   *   @FT_Property_Get.  The following lists the available properties
   *   together with the necessary macros and structures.
   *
   *   Note that the auto-hinter's module name is `autofitter' for
   *   historical reasons.
   *
   */


  /**************************************************************************
   *
   * @property:
   *   glyph-to-script-map
   *
   * @description:
   *   *Experimental* *only*
   *
   *   The auto-hinter provides various script modules to hint glyphs.
   *   Examples of supported scripts are Latin or CJK.  Before a glyph is
   *   auto-hinted, the Unicode character map of the font gets examined, and
   *   the script is then determined based on Unicode character ranges, see
   *   below.
   *
   *   OpenType fonts, however, often provide much more glyphs than
   *   character codes (small caps, superscripts, ligatures, swashes, etc.),
   *   to be controlled by so-called `features'.  Handling OpenType features
   *   can be quite complicated and thus needs a separate library on top of
   *   FreeType.
   *
   *   The mapping between glyph indices and scripts (in the auto-hinter
   *   sense, see the @FT_AUTOHINTER_SCRIPT_XXX values) is stored as an
   *   array with `num_glyphs' elements, as found in the font's @FT_Face
   *   structure.  The `glyph-to-script-map' property returns a pointer to
   *   this array, which can be modified as needed.  Note that the
   *   modification should happen before the first glyph gets processed by
   *   the auto-hinter so that the global analysis of the font shapes
   *   actually uses the modified mapping.
   *
   *   The following example code demonstrates how to access it (omitting
   *   the error handling).
   *
   *   {
   *     FT_Library                library;
   *     FT_Face                   face;
   *     FT_Prop_GlyphToScriptMap  prop;
   *
   *
   *     FT_Init_FreeType( &library );
   *     FT_New_Face( library, "foo.ttf", 0, &face );
   *
   *     prop.face = face;
   *
   *     FT_Property_Get( library, "autofitter",
   *                               "glyph-to-script-map", &prop );
   *
   *     // adjust `prop.map' as needed right here
   *
   *     FT_Load_Glyph( face, ..., FT_LOAD_FORCE_AUTOHINT );
   *   }
   *
   */


  /**************************************************************************
   *
   * @enum:
   *   FT_AUTOHINTER_SCRIPT_XXX
   *
   * @description:
   *   *Experimental* *only*
   *
   *   A list of constants used for the @glyph-to-script-map property to
   *   specify the script submodule the auto-hinter should use for hinting a
   *   particular glyph.
   *
   * @values:
   *   FT_AUTOHINTER_SCRIPT_NONE ::
   *     Don't auto-hint this glyph.
   *
   *   FT_AUTOHINTER_SCRIPT_LATIN ::
   *     Apply the latin auto-hinter.  For the auto-hinter, `latin' is a
   *     very broad term, including Cyrillic and Greek also since characters
   *     from those scripts share the same design constraints.
   *
   *     By default, characters from the following Unicode ranges are
   *     assigned to this submodule.
   *
   *     {
   *       U+0020 - U+007F  // Basic Latin (no control characters)
   *       U+00A0 - U+00FF  // Latin-1 Supplement (no control characters)
   *       U+0100 - U+017F  // Latin Extended-A
   *       U+0180 - U+024F  // Latin Extended-B
   *       U+0250 - U+02AF  // IPA Extensions
   *       U+02B0 - U+02FF  // Spacing Modifier Letters
   *       U+0300 - U+036F  // Combining Diacritical Marks
   *       U+0370 - U+03FF  // Greek and Coptic
   *       U+0400 - U+04FF  // Cyrillic
   *       U+0500 - U+052F  // Cyrillic Supplement
   *       U+1D00 - U+1D7F  // Phonetic Extensions
   *       U+1D80 - U+1DBF  // Phonetic Extensions Supplement
   *       U+1DC0 - U+1DFF  // Combining Diacritical Marks Supplement
   *       U+1E00 - U+1EFF  // Latin Extended Additional
   *       U+1F00 - U+1FFF  // Greek Extended
   *       U+2000 - U+206F  // General Punctuation
   *       U+2070 - U+209F  // Superscripts and Subscripts
   *       U+20A0 - U+20CF  // Currency Symbols
   *       U+2150 - U+218F  // Number Forms
   *       U+2460 - U+24FF  // Enclosed Alphanumerics
   *       U+2C60 - U+2C7F  // Latin Extended-C
   *       U+2DE0 - U+2DFF  // Cyrillic Extended-A
   *       U+2E00 - U+2E7F  // Supplemental Punctuation
   *       U+A640 - U+A69F  // Cyrillic Extended-B
   *       U+A720 - U+A7FF  // Latin Extended-D
   *       U+FB00 - U+FB06  // Alphab. Present. Forms (Latin Ligatures)
   *      U+1D400 - U+1D7FF // Mathematical Alphanumeric Symbols
   *      U+1F100 - U+1F1FF // Enclosed Alphanumeric Supplement
   *     }
   *
   *   FT_AUTOHINTER_SCRIPT_CJK ::
   *     Apply the CJK auto-hinter, covering Chinese, Japanese, Korean, old
   *     Vietnamese, and some other scripts.
   *
   *     By default, characters from the following Unicode ranges are
   *     assigned to this submodule.
   *
   *     {
   *       U+1100 - U+11FF  // Hangul Jamo
   *       U+2E80 - U+2EFF  // CJK Radicals Supplement
   *       U+2F00 - U+2FDF  // Kangxi Radicals
   *       U+2FF0 - U+2FFF  // Ideographic Description Characters
   *       U+3000 - U+303F  // CJK Symbols and Punctuation
   *       U+3040 - U+309F  // Hiragana
   *       U+30A0 - U+30FF  // Katakana
   *       U+3100 - U+312F  // Bopomofo
   *       U+3130 - U+318F  // Hangul Compatibility Jamo
   *       U+3190 - U+319F  // Kanbun
   *       U+31A0 - U+31BF  // Bopomofo Extended
   *       U+31C0 - U+31EF  // CJK Strokes
   *       U+31F0 - U+31FF  // Katakana Phonetic Extensions
   *       U+3200 - U+32FF  // Enclosed CJK Letters and Months
   *       U+3300 - U+33FF  // CJK Compatibility
   *       U+3400 - U+4DBF  // CJK Unified Ideographs Extension A
   *       U+4DC0 - U+4DFF  // Yijing Hexagram Symbols
   *       U+4E00 - U+9FFF  // CJK Unified Ideographs
   *       U+A960 - U+A97F  // Hangul Jamo Extended-A
   *       U+AC00 - U+D7AF  // Hangul Syllables
   *       U+D7B0 - U+D7FF  // Hangul Jamo Extended-B
   *       U+F900 - U+FAFF  // CJK Compatibility Ideographs
   *       U+FE10 - U+FE1F  // Vertical forms
   *       U+FE30 - U+FE4F  // CJK Compatibility Forms
   *       U+FF00 - U+FFEF  // Halfwidth and Fullwidth Forms
   *      U+1B000 - U+1B0FF // Kana Supplement
   *      U+1D300 - U+1D35F // Tai Xuan Hing Symbols
   *      U+1F200 - U+1F2FF // Enclosed Ideographic Supplement
   *      U+20000 - U+2A6DF // CJK Unified Ideographs Extension B
   *      U+2A700 - U+2B73F // CJK Unified Ideographs Extension C
   *      U+2B740 - U+2B81F // CJK Unified Ideographs Extension D
   *      U+2F800 - U+2FA1F // CJK Compatibility Ideographs Supplement
   *     }
   *
   *   FT_AUTOHINTER_SCRIPT_INDIC ::
   *     Apply the indic auto-hinter, covering all major scripts from the
   *     Indian sub-continent and some other related scripts like Thai, Lao,
   *     or Tibetan.
   *
   *     By default, characters from the following Unicode ranges are
   *     assigned to this submodule.
   *
   *     {
   *       U+0900 - U+0DFF  // Indic Range
   *       U+0F00 - U+0FFF  // Tibetan
   *       U+1900 - U+194F  // Limbu
   *       U+1B80 - U+1BBF  // Sundanese
   *       U+A800 - U+A82F  // Syloti Nagri
   *       U+ABC0 - U+ABFF  // Meetei Mayek
   *      U+11800 - U+118DF // Sharada
   *     }
   *
   *     Note that currently Indic support is rudimentary only, missing blue
   *     zone support.
   *
   */
#define FT_AUTOHINTER_SCRIPT_NONE   0
#define FT_AUTOHINTER_SCRIPT_LATIN  1
#define FT_AUTOHINTER_SCRIPT_CJK    2
#define FT_AUTOHINTER_SCRIPT_INDIC  3


  /**************************************************************************
   *
   * @struct:
   *   FT_Prop_GlyphToScriptMap
   *
   * @description:
   *   *Experimental* *only*
   *
   *   The data exchange structure for the @glyph-to-script-map property.
   *
   */
  typedef struct  FT_Prop_GlyphToScriptMap_
  {
    FT_Face     face;
    FT_UShort*  map;

  } FT_Prop_GlyphToScriptMap;


  /**************************************************************************
   *
   * @property:
   *   fallback-script
   *
   * @description:
   *   *Experimental* *only*
   *
   *   If no auto-hinter script module can be assigned to a glyph, a
   *   fallback script gets assigned to it (see also the
   *   @glyph-to-script-map property).  By default, this is
   *   @FT_AUTOHINTER_SCRIPT_CJK.  Using the `fallback-script' property,
   *   this fallback value can be changed.
   *
   *   {
   *     FT_Library  library;
   *     FT_UInt     fallback_script = FT_AUTOHINTER_SCRIPT_NONE;
   *
   *
   *     FT_Init_FreeType( &library );
   *
   *     FT_Property_Set( library, "autofitter",
   *                               "fallback-script", &fallback_script );
   *   }
   *
   * @note:
   *   This property can be used with @FT_Property_Get also.
   *
   *   It's important to use the right timing for changing this value: The
   *   creation of the glyph-to-script map that eventually uses the
   *   fallback script value gets triggered either by setting or reading a
   *   face-specific property like @glyph-to-script-map, or by auto-hinting
   *   any glyph from that face.  In particular, if you have already created
   *   an @FT_Face structure but not loaded any glyph (using the
   *   auto-hinter), a change of the fallback script will affect this face.
   *
   */


  /**************************************************************************
   *
   * @property:
   *   default-script
   *
   * @description:
   *   *Experimental* *only*
   *
   *   If FreeType gets compiled with FT_CONFIG_OPTION_USE_HARFBUZZ to make
   *   the HarfBuzz library access OpenType features for getting better
   *   glyph coverages, this property sets the (auto-fitter) script to be
   *   used for the default (OpenType) script data of a font's GSUB table.
   *   Features for the default script are intended for all scripts not
   *   explicitly handled in GSUB; an example is a `dlig' feature,
   *   containing the combination of the characters `T', `E', and `L' to
   *   form a `TEL' ligature.
   *
   *   By default, this is @FT_AUTOHINTER_SCRIPT_LATIN.  Using the
   *   `default-script' property, this default value can be changed.
   *
   *   {
   *     FT_Library  library;
   *     FT_UInt     default_script = FT_AUTOHINTER_SCRIPT_NONE;
   *
   *
   *     FT_Init_FreeType( &library );
   *
   *     FT_Property_Set( library, "autofitter",
   *                               "default-script", &default_script );
   *   }
   *
   * @note:
   *   This property can be used with @FT_Property_Get also.
   *
   *   It's important to use the right timing for changing this value: The
   *   creation of the glyph-to-script map that eventually uses the
   *   default script value gets triggered either by setting or reading a
   *   face-specific property like @glyph-to-script-map, or by auto-hinting
   *   any glyph from that face.  In particular, if you have already created
   *   an @FT_Face structure but not loaded any glyph (using the
   *   auto-hinter), a change of the default script will affect this face.
   *
   */


  /**************************************************************************
   *
   * @property:
   *   increase-x-height
   *
   * @description:
   *   For ppem values in the range 6~<= ppem <= `increase-x-height', round
   *   up the font's x~height much more often than normally.  If the value
   *   is set to~0, which is the default, this feature is switched off.  Use
   *   this property to improve the legibility of small font sizes if
   *   necessary.
   *
   *   {
   *     FT_Library               library;
   *     FT_Face                  face;
   *     FT_Prop_IncreaseXHeight  prop;
   *
   *
   *     FT_Init_FreeType( &library );
   *     FT_New_Face( library, "foo.ttf", 0, &face );
   *     FT_Set_Char_Size( face, 10 * 64, 0, 72, 0 );
   *
   *     prop.face  = face;
   *     prop.limit = 14;
   *
   *     FT_Property_Set( library, "autofitter",
   *                               "increase-x-height", &prop );
   *   }
   *
   * @note:
   *   This property can be used with @FT_Property_Get also.
   *
   *   Set this value right after calling @FT_Set_Char_Size, but before
   *   loading any glyph (using the auto-hinter).
   *
   */


  /**************************************************************************
   *
   * @struct:
   *   FT_Prop_IncreaseXHeight
   *
   * @description:
   *   The data exchange structure for the @increase-x-height property.
   *
   */
  typedef struct  FT_Prop_IncreaseXHeight_
  {
    FT_Face  face;
    FT_UInt  limit;

  } FT_Prop_IncreaseXHeight;


  /**************************************************************************
   *
   * @property:
   *   warping
   *
   * @description:
   *   *Experimental* *only*
   *
   *   If FreeType gets compiled with option AF_CONFIG_OPTION_USE_WARPER to
   *   activate the warp hinting code in the auto-hinter, this property
   *   switches warping on and off.
   *
   *   Warping only works in `light' auto-hinting mode.  The idea of the
   *   code is to slightly scale and shift a glyph along the non-hinted
   *   dimension (which is usually the horizontal axis) so that as much of
   *   its segments are aligned (more or less) to the grid.  To find out a
   *   glyph's optimal scaling and shifting value, various parameter
   *   combinations are tried and scored.
   *
   *   By default, warping is off.  The example below shows how to switch on
   *   warping (omitting the error handling).
   *
   *   {
   *     FT_Library  library;
   *     FT_Bool     warping = 1;
   *
   *
   *     FT_Init_FreeType( &library );
   *
   *     FT_Property_Set( library, "autofitter",
   *                               "warping", &warping );
   *   }
   *
   * @note:
   *   This property can be used with @FT_Property_Get also.
   *
   *   This property can be set via the `FREETYPE_PROPERTIES' environment
   *   variable (using values 1 and 0 for `on' and `off', respectively).
   *
   *   The warping code can also change advance widths.  Have a look at the
   *   `lsb_delta' and `rsb_delta' fields in the @FT_GlyphSlotRec structure
   *   for details on improving inter-glyph distances while rendering.
   *
   *   Since warping is a global property of the auto-hinter it is best to
   *   change its value before rendering any face.  Otherwise, you should
   *   reload all faces that get auto-hinted in `light' hinting mode.
   *
   */


  /**************************************************************************
   *
   * @property:
   *   no-stem-darkening[autofit]
   *
   * @description:
   *   *Experimental* *only*, *requires* *linear* *alpha* *blending* *and*
   *   *gamma* *correction*
   *
   *   Stem darkening emboldens glyphs at smaller sizes to make them more
   *   readable on common low-DPI screens when using linear alpha blending
   *   and gamma correction, see @FT_Render_Glyph.  When not using linear
   *   alpha blending and gamma correction, glyphs will appear heavy and
   *   fuzzy!
   *
   *   Gamma correction essentially lightens fonts since shades of grey are
   *   shifted to higher pixel values (=~higher brightness) to match the
   *   original intention to the reality of our screens.  The side-effect is
   *   that glyphs `thin out'.  Mac OS~X and Adobe's proprietary font
   *   rendering library implement a counter-measure: stem darkening at
   *   smaller sizes where shades of gray dominate.  By emboldening a glyph
   *   slightly in relation to its pixel size, individual pixels get higher
   *   coverage of filled-in outlines and are therefore `blacker'.  This
   *   counteracts the `thinning out' of glyphs, making text remain readable
   *   at smaller sizes.  All glyphs that pass through the auto-hinter will
   *   be emboldened unless this property is set to TRUE.
   *
   *   See the description of the CFF driver for algorithmic details.  Total
   *   consistency with the CFF driver is currently not achieved because the
   *   emboldening method differs and glyphs must be scaled down on the
   *   Y-axis to keep outline points inside their precomputed blue zones.
   *   The smaller the size (especially 9ppem and down), the higher the loss
   *   of emboldening versus the CFF driver.
   *
   *   This property can be set via the `FREETYPE_PROPERTIES' environment
   *   variable similar to the CFF driver.  It can also be set per face
   *   using @FT_Face_Properties with @FT_PARAM_TAG_STEM_DARKENING.
   *
   */


  /**************************************************************************
   *
   * @constant:
   *   FT_PARAM_TAG_STEM_DARKENING
   *
   * @description:
   *   An @FT_Parameter tag to be used with @FT_Face_Properties.  The
   *   corresponding Boolean argument specifies whether to apply stem
   *   darkening, overriding the global default values or the values set up
   *   with @FT_Property_Set (see @no-stem-darkening[autofit] and
   *   @no-stem-darkening[cff]).
   *
   *   This is a passive setting that only takes effect if the font driver
   *   or autohinter honors it, which the CFF driver always does, but the
   *   autohinter only in `light' hinting mode (as of version 2.7.0).
   *
   */
#define FT_PARAM_TAG_STEM_DARKENING \
          FT_MAKE_TAG( 'd', 'a', 'r', 'k' )


  /**************************************************************************
   *
   * @property:
   *   darkening-parameters[autofit]
   *
   * @description:
   *   *Experimental* *only*
   *
   *   See the description of the CFF driver for details.  This
   *   implementation appropriates the
   *   CFF_CONFIG_OPTION_DARKENING_PARAMETER_* #defines for consistency.
   *   Note the differences described in @no-stem-darkening[autofit].
   *
   *   This property can be set via the `FREETYPE_PROPERTIES' environment
   *   variable similar to the CFF driver.
   */


  /* */


FT_END_HEADER

#endif /* FTAUTOH_H_ */


/* END */
