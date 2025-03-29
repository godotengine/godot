/****************************************************************************
 *
 * afranges.c
 *
 *   Auto-fitter Unicode script ranges (body).
 *
 * Copyright (C) 2013-2024 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#include "afranges.h"

  /*
   * The algorithm for assigning properties and styles to the `glyph_styles'
   * array is as follows (cf. the implementation in
   * `af_face_globals_compute_style_coverage').
   *
   *   Walk over all scripts (as listed in `afscript.h').
   *
   *   For a given script, walk over all styles (as listed in `afstyles.h').
   *   The order of styles is important and should be as follows.
   *
   *   - First come styles based on OpenType features (small caps, for
   *     example).  Since features rely on glyph indices, thus completely
   *     bypassing character codes, no properties are assigned.
   *
   *   - Next comes the default style, using the character ranges as defined
   *     below.  This also assigns properties.
   *
   *   Note that there also exist fallback scripts, mainly covering
   *   superscript and subscript glyphs of a script that are not present as
   *   OpenType features.  Fallback scripts are defined below, also
   *   assigning properties; they are applied after the corresponding
   *   script.
   *
   */


  /* XXX Check base character ranges again:                        */
  /*     Right now, they are quickly derived by visual inspection. */
  /*     I can imagine that fine-tuning is necessary.              */

  /* for the auto-hinter, a `non-base character' is something that should */
  /* not be affected by blue zones, regardless of whether this is a       */
  /* spacing or no-spacing glyph                                          */

  /* the `af_xxxx_nonbase_uniranges' ranges must be strict subsets */
  /* of the corresponding `af_xxxx_uniranges' ranges               */


  const AF_Script_UniRangeRec  af_adlm_uniranges[] =
  {
    AF_UNIRANGE_REC( 0x1E900, 0x1E95F ),   /* Adlam */
    AF_UNIRANGE_REC(       0,       0 )
  };

  const AF_Script_UniRangeRec  af_adlm_nonbase_uniranges[] =
  {
    AF_UNIRANGE_REC( 0x1D944, 0x1E94A ),
    AF_UNIRANGE_REC(       0,       0 )
  };


  const AF_Script_UniRangeRec  af_arab_uniranges[] =
  {
    AF_UNIRANGE_REC(  0x0600,  0x06FF ),  /* Arabic                                 */
    AF_UNIRANGE_REC(  0x0750,  0x07FF ),  /* Arabic Supplement                      */
    AF_UNIRANGE_REC(  0x08A0,  0x08FF ),  /* Arabic Extended-A                      */
    AF_UNIRANGE_REC(  0xFB50,  0xFDFF ),  /* Arabic Presentation Forms-A            */
    AF_UNIRANGE_REC(  0xFE70,  0xFEFF ),  /* Arabic Presentation Forms-B            */
    AF_UNIRANGE_REC( 0x1EE00, 0x1EEFF ),  /* Arabic Mathematical Alphabetic Symbols */
    AF_UNIRANGE_REC(       0,       0 )
  };

  const AF_Script_UniRangeRec  af_arab_nonbase_uniranges[] =
  {
    AF_UNIRANGE_REC(  0x0600,  0x0605 ),
    AF_UNIRANGE_REC(  0x0610,  0x061A ),
    AF_UNIRANGE_REC(  0x064B,  0x065F ),
    AF_UNIRANGE_REC(  0x0670,  0x0670 ),
    AF_UNIRANGE_REC(  0x06D6,  0x06DC ),
    AF_UNIRANGE_REC(  0x06DF,  0x06E4 ),
    AF_UNIRANGE_REC(  0x06E7,  0x06E8 ),
    AF_UNIRANGE_REC(  0x06EA,  0x06ED ),
    AF_UNIRANGE_REC(  0x08D4,  0x08E1 ),
    AF_UNIRANGE_REC(  0x08D3,  0x08FF ),
    AF_UNIRANGE_REC(  0xFBB2,  0xFBC1 ),
    AF_UNIRANGE_REC(  0xFE70,  0xFE70 ),
    AF_UNIRANGE_REC(  0xFE72,  0xFE72 ),
    AF_UNIRANGE_REC(  0xFE74,  0xFE74 ),
    AF_UNIRANGE_REC(  0xFE76,  0xFE76 ),
    AF_UNIRANGE_REC(  0xFE78,  0xFE78 ),
    AF_UNIRANGE_REC(  0xFE7A,  0xFE7A ),
    AF_UNIRANGE_REC(  0xFE7C,  0xFE7C ),
    AF_UNIRANGE_REC(  0xFE7E,  0xFE7E ),
    AF_UNIRANGE_REC(       0,       0 )
  };


  const AF_Script_UniRangeRec  af_armn_uniranges[] =
  {
    AF_UNIRANGE_REC(  0x0530,  0x058F ),  /* Armenian                          */
    AF_UNIRANGE_REC(  0xFB13,  0xFB17 ),  /* Alphab. Present. Forms (Armenian) */
    AF_UNIRANGE_REC(       0,       0 )
  };

  const AF_Script_UniRangeRec  af_armn_nonbase_uniranges[] =
  {
    AF_UNIRANGE_REC(  0x0559,  0x055F ),
    AF_UNIRANGE_REC(       0,       0 )
  };


  const AF_Script_UniRangeRec  af_avst_uniranges[] =
  {
    AF_UNIRANGE_REC( 0x10B00,  0x10B3F ),  /* Avestan */
    AF_UNIRANGE_REC(       0,        0 )
  };

  const AF_Script_UniRangeRec  af_avst_nonbase_uniranges[] =
  {
    AF_UNIRANGE_REC( 0x10B39,  0x10B3F ),
    AF_UNIRANGE_REC(       0,        0 )
  };


  const AF_Script_UniRangeRec  af_bamu_uniranges[] =
  {
    AF_UNIRANGE_REC( 0xA6A0,   0xA6FF ),   /* Bamum */
#if 0
    /* The characters in the Bamum supplement are pictograms, */
    /* not (directly) related to the syllabic Bamum script    */
    AF_UNIRANGE_REC( 0x16800, 0x16A3F ),   /* Bamum Supplement */
#endif
    AF_UNIRANGE_REC(       0,       0 )
  };

  const AF_Script_UniRangeRec  af_bamu_nonbase_uniranges[] =
  {
    AF_UNIRANGE_REC(  0xA6F0,  0xA6F1 ),
    AF_UNIRANGE_REC(       0,       0 )
  };


  const AF_Script_UniRangeRec  af_beng_uniranges[] =
  {
    AF_UNIRANGE_REC(  0x0980,  0x09FF ),  /* Bengali */
    AF_UNIRANGE_REC(       0,       0 )
  };

  const AF_Script_UniRangeRec  af_beng_nonbase_uniranges[] =
  {
    AF_UNIRANGE_REC(  0x0981,  0x0981 ),
    AF_UNIRANGE_REC(  0x09BC,  0x09BC ),
    AF_UNIRANGE_REC(  0x09C1,  0x09C4 ),
    AF_UNIRANGE_REC(  0x09CD,  0x09CD ),
    AF_UNIRANGE_REC(  0x09E2,  0x09E3 ),
    AF_UNIRANGE_REC(  0x09FE,  0x09FE ),
    AF_UNIRANGE_REC(       0,       0 )
  };


  const AF_Script_UniRangeRec  af_buhd_uniranges[] =
  {
    AF_UNIRANGE_REC(  0x1740,  0x175F ),   /* Buhid */
    AF_UNIRANGE_REC(       0,       0 )
  };

  const AF_Script_UniRangeRec  af_buhd_nonbase_uniranges[] =
  {
    AF_UNIRANGE_REC(  0x1752,  0x1753 ),
    AF_UNIRANGE_REC(       0,       0 )
  };


  const AF_Script_UniRangeRec  af_cakm_uniranges[] =
  {
    AF_UNIRANGE_REC( 0x11100, 0x1114F ),   /* Chakma */
    AF_UNIRANGE_REC(       0,       0 )
  };

  const AF_Script_UniRangeRec  af_cakm_nonbase_uniranges[] =
  {
    AF_UNIRANGE_REC( 0x11100, 0x11102 ),
    AF_UNIRANGE_REC( 0x11127, 0x11134 ),
    AF_UNIRANGE_REC( 0x11146, 0x11146 ),
    AF_UNIRANGE_REC(       0,       0 )
  };


  const AF_Script_UniRangeRec  af_cans_uniranges[] =
  {
    AF_UNIRANGE_REC(  0x1400,  0x167F ), /* Unified Canadian Aboriginal Syllabics          */
    AF_UNIRANGE_REC(  0x18B0,  0x18FF ), /* Unified Canadian Aboriginal Syllabics Extended */
    AF_UNIRANGE_REC(       0,       0 )
  };

  const AF_Script_UniRangeRec  af_cans_nonbase_uniranges[] =
  {
    AF_UNIRANGE_REC( 0, 0 )
  };


  const AF_Script_UniRangeRec  af_cari_uniranges[] =
  {
    AF_UNIRANGE_REC( 0x102A0, 0x102DF ),   /* Carian */
    AF_UNIRANGE_REC(       0,       0 )
  };

  const AF_Script_UniRangeRec  af_cari_nonbase_uniranges[] =
  {
    AF_UNIRANGE_REC( 0, 0 )
  };


  const AF_Script_UniRangeRec  af_cher_uniranges[] =
  {
    AF_UNIRANGE_REC(  0x13A0,  0x13FF ),  /* Cherokee            */
    AF_UNIRANGE_REC(  0xAB70,  0xABBF ),  /* Cherokee Supplement */
    AF_UNIRANGE_REC(       0,       0 )
  };

  const AF_Script_UniRangeRec  af_cher_nonbase_uniranges[] =
  {
    AF_UNIRANGE_REC( 0, 0 )
  };


  const AF_Script_UniRangeRec  af_copt_uniranges[] =
  {
    AF_UNIRANGE_REC(  0x2C80,  0x2CFF ),   /* Coptic */
    AF_UNIRANGE_REC(       0,       0 )
  };

  const AF_Script_UniRangeRec  af_copt_nonbase_uniranges[] =
  {
    AF_UNIRANGE_REC(  0x2CEF,  0x2CF1 ),
    AF_UNIRANGE_REC(       0,       0 )
  };


  const AF_Script_UniRangeRec  af_cprt_uniranges[] =
  {
    AF_UNIRANGE_REC( 0x10800, 0x1083F ),   /* Cypriot */
    AF_UNIRANGE_REC(       0,       0 )
  };

  const AF_Script_UniRangeRec  af_cprt_nonbase_uniranges[] =
  {
    AF_UNIRANGE_REC( 0, 0 )
  };


  const AF_Script_UniRangeRec  af_cyrl_uniranges[] =
  {
    AF_UNIRANGE_REC(  0x0400,  0x04FF ),  /* Cyrillic            */
    AF_UNIRANGE_REC(  0x0500,  0x052F ),  /* Cyrillic Supplement */
    AF_UNIRANGE_REC(  0x2DE0,  0x2DFF ),  /* Cyrillic Extended-A */
    AF_UNIRANGE_REC(  0xA640,  0xA69F ),  /* Cyrillic Extended-B */
    AF_UNIRANGE_REC(  0x1C80,  0x1C8F ),  /* Cyrillic Extended-C */
    AF_UNIRANGE_REC(       0,       0 )
  };

  const AF_Script_UniRangeRec  af_cyrl_nonbase_uniranges[] =
  {
    AF_UNIRANGE_REC(  0x0483,  0x0489 ),
    AF_UNIRANGE_REC(  0x2DE0,  0x2DFF ),
    AF_UNIRANGE_REC(  0xA66F,  0xA67F ),
    AF_UNIRANGE_REC(  0xA69E,  0xA69F ),
    AF_UNIRANGE_REC(       0,       0 )
  };


  /* There are some characters in the Devanagari Unicode block that are    */
  /* generic to Indic scripts; we omit them so that their presence doesn't */
  /* trigger Devanagari.                                                   */

  const AF_Script_UniRangeRec  af_deva_uniranges[] =
  {
    AF_UNIRANGE_REC(  0x0900,  0x093B ),  /* Devanagari          */
    /* omitting U+093C nukta */
    AF_UNIRANGE_REC(  0x093D,  0x0950 ),  /* ... continued       */
    /* omitting U+0951 udatta, U+0952 anudatta */
    AF_UNIRANGE_REC(  0x0953,  0x0963 ),  /* ... continued       */
    /* omitting U+0964 danda, U+0965 double danda */
    AF_UNIRANGE_REC(  0x0966,  0x097F ),  /* ... continued       */
    AF_UNIRANGE_REC(  0x20B9,  0x20B9 ),  /* (new) Rupee sign    */
    AF_UNIRANGE_REC(  0xA8E0,  0xA8FF ),  /* Devanagari Extended */
    AF_UNIRANGE_REC(       0,       0 )
  };

  const AF_Script_UniRangeRec  af_deva_nonbase_uniranges[] =
  {
    AF_UNIRANGE_REC(  0x0900,  0x0902 ),
    AF_UNIRANGE_REC(  0x093A,  0x093A ),
    AF_UNIRANGE_REC(  0x0941,  0x0948 ),
    AF_UNIRANGE_REC(  0x094D,  0x094D ),
    AF_UNIRANGE_REC(  0x0953,  0x0957 ),
    AF_UNIRANGE_REC(  0x0962,  0x0963 ),
    AF_UNIRANGE_REC(  0xA8E0,  0xA8F1 ),
    AF_UNIRANGE_REC(  0xA8FF,  0xA8FF ),
    AF_UNIRANGE_REC(       0,       0 )
  };


  const AF_Script_UniRangeRec  af_dsrt_uniranges[] =
  {
    AF_UNIRANGE_REC( 0x10400, 0x1044F ),  /* Deseret */
    AF_UNIRANGE_REC(       0,       0 )
  };

  const AF_Script_UniRangeRec  af_dsrt_nonbase_uniranges[] =
  {
    AF_UNIRANGE_REC( 0, 0 )
  };


  const AF_Script_UniRangeRec  af_ethi_uniranges[] =
  {
    AF_UNIRANGE_REC(  0x1200,  0x137F ),  /* Ethiopic            */
    AF_UNIRANGE_REC(  0x1380,  0x139F ),  /* Ethiopic Supplement */
    AF_UNIRANGE_REC(  0x2D80,  0x2DDF ),  /* Ethiopic Extended   */
    AF_UNIRANGE_REC(  0xAB00,  0xAB2F ),  /* Ethiopic Extended-A */
    AF_UNIRANGE_REC(       0,       0 )
  };

  const AF_Script_UniRangeRec  af_ethi_nonbase_uniranges[] =
  {
    AF_UNIRANGE_REC(  0x135D,  0x135F ),
    AF_UNIRANGE_REC(       0,       0 )
  };


  const AF_Script_UniRangeRec  af_geor_uniranges[] =
  {
    AF_UNIRANGE_REC(  0x10D0,  0x10FF ),  /* Georgian (Mkhedruli)          */
    AF_UNIRANGE_REC(  0x1C90,  0x1CBF ),  /* Georgian Extended (Mtavruli)  */
    AF_UNIRANGE_REC(       0,       0 )
  };

  const AF_Script_UniRangeRec  af_geor_nonbase_uniranges[] =
  {
    AF_UNIRANGE_REC( 0, 0 )
  };


  const AF_Script_UniRangeRec  af_geok_uniranges[] =
  {
    /* Khutsuri */
    AF_UNIRANGE_REC(  0x10A0,  0x10CD ),  /* Georgian (Asomtavruli)         */
    AF_UNIRANGE_REC(  0x2D00,  0x2D2D ),  /* Georgian Supplement (Nuskhuri) */
    AF_UNIRANGE_REC(       0,       0 )
  };

  const AF_Script_UniRangeRec  af_geok_nonbase_uniranges[] =
  {
    AF_UNIRANGE_REC( 0, 0 )
  };


  const AF_Script_UniRangeRec  af_glag_uniranges[] =
  {
    AF_UNIRANGE_REC(  0x2C00,  0x2C5F ),  /* Glagolitic */
    AF_UNIRANGE_REC( 0x1E000, 0x1E02F ),  /* Glagolitic Supplement */
    AF_UNIRANGE_REC(       0,       0 )
  };

  const AF_Script_UniRangeRec  af_glag_nonbase_uniranges[] =
  {
    AF_UNIRANGE_REC( 0x1E000, 0x1E02F ),
    AF_UNIRANGE_REC(       0,       0 )
  };


  const AF_Script_UniRangeRec  af_goth_uniranges[] =
  {
    AF_UNIRANGE_REC( 0x10330, 0x1034F ),   /* Gothic */
    AF_UNIRANGE_REC(       0,       0 )
  };

  const AF_Script_UniRangeRec  af_goth_nonbase_uniranges[] =
  {
    AF_UNIRANGE_REC( 0, 0 )
  };


  const AF_Script_UniRangeRec  af_grek_uniranges[] =
  {
    AF_UNIRANGE_REC(  0x0370,  0x03FF ),  /* Greek and Coptic */
    AF_UNIRANGE_REC(  0x1F00,  0x1FFF ),  /* Greek Extended   */
    AF_UNIRANGE_REC(       0,       0 )
  };

  const AF_Script_UniRangeRec  af_grek_nonbase_uniranges[] =
  {
    AF_UNIRANGE_REC(  0x037A,  0x037A ),
    AF_UNIRANGE_REC(  0x0384,  0x0385 ),
    AF_UNIRANGE_REC(  0x1FBD,  0x1FC1 ),
    AF_UNIRANGE_REC(  0x1FCD,  0x1FCF ),
    AF_UNIRANGE_REC(  0x1FDD,  0x1FDF ),
    AF_UNIRANGE_REC(  0x1FED,  0x1FEF ),
    AF_UNIRANGE_REC(  0x1FFD,  0x1FFE ),
    AF_UNIRANGE_REC(       0,       0 )
  };


  const AF_Script_UniRangeRec  af_gujr_uniranges[] =
  {
    AF_UNIRANGE_REC(  0x0A80,  0x0AFF ),  /* Gujarati */
    AF_UNIRANGE_REC(       0,       0 )
  };

  const AF_Script_UniRangeRec  af_gujr_nonbase_uniranges[] =
  {
    AF_UNIRANGE_REC(  0x0A81,  0x0A82 ),
    AF_UNIRANGE_REC(  0x0ABC,  0x0ABC ),
    AF_UNIRANGE_REC(  0x0AC1,  0x0AC8 ),
    AF_UNIRANGE_REC(  0x0ACD,  0x0ACD ),
    AF_UNIRANGE_REC(  0x0AE2,  0x0AE3 ),
    AF_UNIRANGE_REC(  0x0AFA,  0x0AFF ),
    AF_UNIRANGE_REC(       0,       0 )
  };


  const AF_Script_UniRangeRec  af_guru_uniranges[] =
  {
    AF_UNIRANGE_REC(  0x0A00,  0x0A7F ),  /* Gurmukhi */
    AF_UNIRANGE_REC(       0,       0 )
  };

  const AF_Script_UniRangeRec  af_guru_nonbase_uniranges[] =
  {
    AF_UNIRANGE_REC(  0x0A01,  0x0A02 ),
    AF_UNIRANGE_REC(  0x0A3C,  0x0A3C ),
    AF_UNIRANGE_REC(  0x0A41,  0x0A51 ),
    AF_UNIRANGE_REC(  0x0A70,  0x0A71 ),
    AF_UNIRANGE_REC(  0x0A75,  0x0A75 ),
    AF_UNIRANGE_REC(       0,       0 )
  };


  const AF_Script_UniRangeRec  af_hebr_uniranges[] =
  {
    AF_UNIRANGE_REC(  0x0590,  0x05FF ),  /* Hebrew                          */
    AF_UNIRANGE_REC(  0xFB1D,  0xFB4F ),  /* Alphab. Present. Forms (Hebrew) */
    AF_UNIRANGE_REC(       0,       0 )
  };

  const AF_Script_UniRangeRec  af_hebr_nonbase_uniranges[] =
  {
    AF_UNIRANGE_REC(  0x0591,  0x05BF ),
    AF_UNIRANGE_REC(  0x05C1,  0x05C2 ),
    AF_UNIRANGE_REC(  0x05C4,  0x05C5 ),
    AF_UNIRANGE_REC(  0x05C7,  0x05C7 ),
    AF_UNIRANGE_REC(  0xFB1E,  0xFB1E ),
    AF_UNIRANGE_REC(       0,       0 )
  };


  const AF_Script_UniRangeRec  af_kali_uniranges[] =
  {
    AF_UNIRANGE_REC(  0xA900,  0xA92F ),   /* Kayah Li */
    AF_UNIRANGE_REC(       0,       0 )
  };

  const AF_Script_UniRangeRec  af_kali_nonbase_uniranges[] =
  {
    AF_UNIRANGE_REC(  0xA926,  0xA92D ),
    AF_UNIRANGE_REC(       0,       0 )
  };


  const AF_Script_UniRangeRec  af_knda_uniranges[] =
  {
    AF_UNIRANGE_REC(  0x0C80,  0x0CFF ),  /* Kannada */
    AF_UNIRANGE_REC(       0,       0 )
  };

  const AF_Script_UniRangeRec  af_knda_nonbase_uniranges[] =
  {
    AF_UNIRANGE_REC(  0x0C81,  0x0C81 ),
    AF_UNIRANGE_REC(  0x0CBC,  0x0CBC ),
    AF_UNIRANGE_REC(  0x0CBF,  0x0CBF ),
    AF_UNIRANGE_REC(  0x0CC6,  0x0CC6 ),
    AF_UNIRANGE_REC(  0x0CCC,  0x0CCD ),
    AF_UNIRANGE_REC(  0x0CE2,  0x0CE3 ),
    AF_UNIRANGE_REC(       0,       0 )
  };


  const AF_Script_UniRangeRec  af_khmr_uniranges[] =
  {
    AF_UNIRANGE_REC(  0x1780,  0x17FF ),  /* Khmer */
    AF_UNIRANGE_REC(       0,       0 )
  };

  const AF_Script_UniRangeRec  af_khmr_nonbase_uniranges[] =
  {
    AF_UNIRANGE_REC(  0x17B7,  0x17BD ),
    AF_UNIRANGE_REC(  0x17C6,  0x17C6 ),
    AF_UNIRANGE_REC(  0x17C9,  0x17D3 ),
    AF_UNIRANGE_REC(  0x17DD,  0x17DD ),
    AF_UNIRANGE_REC(       0,       0 )
  };


  const AF_Script_UniRangeRec  af_khms_uniranges[] =
  {
    AF_UNIRANGE_REC(  0x19E0,  0x19FF ),  /* Khmer Symbols */
    AF_UNIRANGE_REC(       0,       0 )
  };

  const AF_Script_UniRangeRec  af_khms_nonbase_uniranges[] =
  {
    AF_UNIRANGE_REC( 0, 0 )
  };


  const AF_Script_UniRangeRec  af_lao_uniranges[] =
  {
    AF_UNIRANGE_REC(  0x0E80,  0x0EFF ),  /* Lao */
    AF_UNIRANGE_REC(       0,       0 )
  };

  const AF_Script_UniRangeRec  af_lao_nonbase_uniranges[] =
  {
    AF_UNIRANGE_REC(  0x0EB1,  0x0EB1 ),
    AF_UNIRANGE_REC(  0x0EB4,  0x0EBC ),
    AF_UNIRANGE_REC(  0x0EC8,  0x0ECD ),
    AF_UNIRANGE_REC(       0,       0 )
  };


  const AF_Script_UniRangeRec  af_latn_uniranges[] =
  {
    AF_UNIRANGE_REC(  0x0020,  0x007F ),  /* Basic Latin (no control chars)         */
    AF_UNIRANGE_REC(  0x00A0,  0x00A9 ),  /* Latin-1 Supplement (no control chars)  */
    AF_UNIRANGE_REC(  0x00AB,  0x00B1 ),  /* ... continued                          */
    AF_UNIRANGE_REC(  0x00B4,  0x00B8 ),  /* ... continued                          */
    AF_UNIRANGE_REC(  0x00BB,  0x00FF ),  /* ... continued                          */
    AF_UNIRANGE_REC(  0x0100,  0x017F ),  /* Latin Extended-A                       */
    AF_UNIRANGE_REC(  0x0180,  0x024F ),  /* Latin Extended-B                       */
    AF_UNIRANGE_REC(  0x0250,  0x02AF ),  /* IPA Extensions                         */
    AF_UNIRANGE_REC(  0x02B9,  0x02DF ),  /* Spacing Modifier Letters               */
    AF_UNIRANGE_REC(  0x02E5,  0x02FF ),  /* ... continued                          */
    AF_UNIRANGE_REC(  0x0300,  0x036F ),  /* Combining Diacritical Marks            */
    AF_UNIRANGE_REC(  0x1AB0,  0x1ABE ),  /* Combining Diacritical Marks Extended   */
    AF_UNIRANGE_REC(  0x1D00,  0x1D2B ),  /* Phonetic Extensions                    */
    AF_UNIRANGE_REC(  0x1D6B,  0x1D77 ),  /* ... continued                          */
    AF_UNIRANGE_REC(  0x1D79,  0x1D7F ),  /* ... continued                          */
    AF_UNIRANGE_REC(  0x1D80,  0x1D9A ),  /* Phonetic Extensions Supplement         */
    AF_UNIRANGE_REC(  0x1DC0,  0x1DFF ),  /* Combining Diacritical Marks Supplement */
    AF_UNIRANGE_REC(  0x1E00,  0x1EFF ),  /* Latin Extended Additional              */
    AF_UNIRANGE_REC(  0x2000,  0x206F ),  /* General Punctuation                    */
    AF_UNIRANGE_REC(  0x20A0,  0x20B8 ),  /* Currency Symbols ...                   */
    AF_UNIRANGE_REC(  0x20BA,  0x20CF ),  /* ... except new Rupee sign              */
    AF_UNIRANGE_REC(  0x2150,  0x218F ),  /* Number Forms                           */
    AF_UNIRANGE_REC(  0x2C60,  0x2C7B ),  /* Latin Extended-C                       */
    AF_UNIRANGE_REC(  0x2C7E,  0x2C7F ),  /* ... continued                          */
    AF_UNIRANGE_REC(  0x2E00,  0x2E7F ),  /* Supplemental Punctuation               */
    AF_UNIRANGE_REC(  0xA720,  0xA76F ),  /* Latin Extended-D                       */
    AF_UNIRANGE_REC(  0xA771,  0xA7F7 ),  /* ... continued                          */
    AF_UNIRANGE_REC(  0xA7FA,  0xA7FF ),  /* ... continued                          */
    AF_UNIRANGE_REC(  0xAB30,  0xAB5B ),  /* Latin Extended-E                       */
    AF_UNIRANGE_REC(  0xAB60,  0xAB6F ),  /* ... continued                          */
    AF_UNIRANGE_REC(  0xFB00,  0xFB06 ),  /* Alphab. Present. Forms (Latin Ligs)    */
    AF_UNIRANGE_REC( 0x1D400, 0x1D7FF ),  /* Mathematical Alphanumeric Symbols      */
    AF_UNIRANGE_REC(       0,       0 )
  };

  const AF_Script_UniRangeRec  af_latn_nonbase_uniranges[] =
  {
    AF_UNIRANGE_REC(  0x005E,  0x0060 ),
    AF_UNIRANGE_REC(  0x007E,  0x007E ),
    AF_UNIRANGE_REC(  0x00A8,  0x00A9 ),
    AF_UNIRANGE_REC(  0x00AE,  0x00B0 ),
    AF_UNIRANGE_REC(  0x00B4,  0x00B4 ),
    AF_UNIRANGE_REC(  0x00B8,  0x00B8 ),
    AF_UNIRANGE_REC(  0x00BC,  0x00BE ),
    AF_UNIRANGE_REC(  0x02B9,  0x02DF ),
    AF_UNIRANGE_REC(  0x02E5,  0x02FF ),
    AF_UNIRANGE_REC(  0x0300,  0x036F ),
    AF_UNIRANGE_REC(  0x1AB0,  0x1ABE ),
    AF_UNIRANGE_REC(  0x1DC0,  0x1DFF ),
    AF_UNIRANGE_REC(  0x2017,  0x2017 ),
    AF_UNIRANGE_REC(  0x203E,  0x203E ),
    AF_UNIRANGE_REC(  0xA788,  0xA788 ),
    AF_UNIRANGE_REC(  0xA7F8,  0xA7FA ),
    AF_UNIRANGE_REC(       0,       0 )
  };


  const AF_Script_UniRangeRec  af_latb_uniranges[] =
  {
    AF_UNIRANGE_REC(  0x1D62,  0x1D6A ),  /* some small subscript letters   */
    AF_UNIRANGE_REC(  0x2080,  0x209C ),  /* subscript digits and letters   */
    AF_UNIRANGE_REC(  0x2C7C,  0x2C7C ),  /* latin subscript small letter j */
    AF_UNIRANGE_REC(       0,       0 )
  };

  const AF_Script_UniRangeRec  af_latb_nonbase_uniranges[] =
  {
    AF_UNIRANGE_REC( 0, 0 )
  };


  const AF_Script_UniRangeRec  af_latp_uniranges[] =
  {
    AF_UNIRANGE_REC(  0x00AA,  0x00AA ),  /* feminine ordinal indicator          */
    AF_UNIRANGE_REC(  0x00B2,  0x00B3 ),  /* superscript two and three           */
    AF_UNIRANGE_REC(  0x00B9,  0x00BA ),  /* superscript one, masc. ord. indic.  */
    AF_UNIRANGE_REC(  0x02B0,  0x02B8 ),  /* some latin superscript mod. letters */
    AF_UNIRANGE_REC(  0x02E0,  0x02E4 ),  /* some IPA modifier letters           */
    AF_UNIRANGE_REC(  0x1D2C,  0x1D61 ),  /* latin superscript modifier letters  */
    AF_UNIRANGE_REC(  0x1D78,  0x1D78 ),  /* modifier letter cyrillic en         */
    AF_UNIRANGE_REC(  0x1D9B,  0x1DBF ),  /* more modifier letters               */
    AF_UNIRANGE_REC(  0x2070,  0x207F ),  /* superscript digits and letters      */
    AF_UNIRANGE_REC(  0x2C7D,  0x2C7D ),  /* modifier letter capital v           */
    AF_UNIRANGE_REC(  0xA770,  0xA770 ),  /* modifier letter us                  */
    AF_UNIRANGE_REC(  0xA7F8,  0xA7F9 ),  /* more modifier letters               */
    AF_UNIRANGE_REC(  0xAB5C,  0xAB5F ),  /* more modifier letters               */
    AF_UNIRANGE_REC(       0,       0 )
  };

  const AF_Script_UniRangeRec  af_latp_nonbase_uniranges[] =
  {
    AF_UNIRANGE_REC( 0, 0 )
  };


  const AF_Script_UniRangeRec  af_lisu_uniranges[] =
  {
    AF_UNIRANGE_REC(  0xA4D0,  0xA4FF ),    /* Lisu */
    AF_UNIRANGE_REC(       0,       0 )
  };

  const AF_Script_UniRangeRec  af_lisu_nonbase_uniranges[] =
  {
    AF_UNIRANGE_REC( 0, 0 )
  };


  const AF_Script_UniRangeRec  af_mlym_uniranges[] =
  {
    AF_UNIRANGE_REC(  0x0D00,  0x0D7F ),  /* Malayalam */
    AF_UNIRANGE_REC(       0,       0 )
  };

  const AF_Script_UniRangeRec  af_mlym_nonbase_uniranges[] =
  {
    AF_UNIRANGE_REC(  0x0D00,  0x0D01 ),
    AF_UNIRANGE_REC(  0x0D3B,  0x0D3C ),
    AF_UNIRANGE_REC(  0x0D4D,  0x0D4E ),
    AF_UNIRANGE_REC(  0x0D62,  0x0D63 ),
    AF_UNIRANGE_REC(       0,       0 )
  };


  const AF_Script_UniRangeRec  af_medf_uniranges[] =
  {
    AF_UNIRANGE_REC( 0x16E40, 0x16E9F ),  /* Medefaidrin */
    AF_UNIRANGE_REC(       0,       0 )
  };

  const AF_Script_UniRangeRec  af_medf_nonbase_uniranges[] =
  {
    AF_UNIRANGE_REC(       0,       0 )
  };


  const AF_Script_UniRangeRec  af_mong_uniranges[] =
  {
    AF_UNIRANGE_REC(  0x1800,  0x18AF ),  /* Mongolian            */
    AF_UNIRANGE_REC( 0x11660, 0x1167F ),  /* Mongolian Supplement */
    AF_UNIRANGE_REC(       0,       0 )
  };

  const AF_Script_UniRangeRec  af_mong_nonbase_uniranges[] =
  {
    AF_UNIRANGE_REC(  0x1885,  0x1886 ),
    AF_UNIRANGE_REC(  0x18A9,  0x18A9 ),
    AF_UNIRANGE_REC(       0,       0 )
  };


  const AF_Script_UniRangeRec  af_mymr_uniranges[] =
  {
    AF_UNIRANGE_REC(  0x1000,  0x109F ),    /* Myanmar            */
    AF_UNIRANGE_REC(  0xA9E0,  0xA9FF ),    /* Myanmar Extended-B */
    AF_UNIRANGE_REC(  0xAA60,  0xAA7F ),    /* Myanmar Extended-A */
    AF_UNIRANGE_REC(       0,       0 )
  };

  const AF_Script_UniRangeRec  af_mymr_nonbase_uniranges[] =
  {
    AF_UNIRANGE_REC(  0x102D,  0x1030 ),
    AF_UNIRANGE_REC(  0x1032,  0x1037 ),
    AF_UNIRANGE_REC(  0x103A,  0x103A ),
    AF_UNIRANGE_REC(  0x103D,  0x103E ),
    AF_UNIRANGE_REC(  0x1058,  0x1059 ),
    AF_UNIRANGE_REC(  0x105E,  0x1060 ),
    AF_UNIRANGE_REC(  0x1071,  0x1074 ),
    AF_UNIRANGE_REC(  0x1082,  0x1082 ),
    AF_UNIRANGE_REC(  0x1085,  0x1086 ),
    AF_UNIRANGE_REC(  0x108D,  0x108D ),
    AF_UNIRANGE_REC(  0xA9E5,  0xA9E5 ),
    AF_UNIRANGE_REC(  0xAA7C,  0xAA7C ),
    AF_UNIRANGE_REC(       0,       0 )
  };


  const AF_Script_UniRangeRec  af_nkoo_uniranges[] =
  {
    AF_UNIRANGE_REC(  0x07C0,  0x07FF ),    /* N'Ko */
    AF_UNIRANGE_REC(       0,       0 )
  };

  const AF_Script_UniRangeRec  af_nkoo_nonbase_uniranges[] =
  {
    AF_UNIRANGE_REC(  0x07EB,  0x07F5 ),
    AF_UNIRANGE_REC(  0x07FD,  0x07FD ),
    AF_UNIRANGE_REC(       0,       0 )
  };


  const AF_Script_UniRangeRec  af_none_uniranges[] =
  {
    AF_UNIRANGE_REC( 0, 0 )
  };

  const AF_Script_UniRangeRec  af_none_nonbase_uniranges[] =
  {
    AF_UNIRANGE_REC( 0, 0 )
  };


  const AF_Script_UniRangeRec  af_olck_uniranges[] =
  {
    AF_UNIRANGE_REC(  0x1C50,  0x1C7F ),    /* Ol Chiki */
    AF_UNIRANGE_REC(       0,       0 )
  };

  const AF_Script_UniRangeRec  af_olck_nonbase_uniranges[] =
  {
    AF_UNIRANGE_REC( 0, 0 )
  };


  const AF_Script_UniRangeRec  af_orkh_uniranges[] =
  {
    AF_UNIRANGE_REC( 0x10C00, 0x10C4F ),    /* Old Turkic */
    AF_UNIRANGE_REC(       0,       0 )
  };

  const AF_Script_UniRangeRec  af_orkh_nonbase_uniranges[] =
  {
    AF_UNIRANGE_REC( 0, 0 )
  };


  const AF_Script_UniRangeRec  af_osge_uniranges[] =
  {
    AF_UNIRANGE_REC( 0x104B0, 0x104FF ),    /* Osage */
    AF_UNIRANGE_REC(       0,       0 )
  };

  const AF_Script_UniRangeRec  af_osge_nonbase_uniranges[] =
  {
    AF_UNIRANGE_REC( 0, 0 )
  };


  const AF_Script_UniRangeRec  af_osma_uniranges[] =
  {
    AF_UNIRANGE_REC( 0x10480, 0x104AF ),   /* Osmanya */
    AF_UNIRANGE_REC(       0,       0 )
  };

  const AF_Script_UniRangeRec  af_osma_nonbase_uniranges[] =
  {
    AF_UNIRANGE_REC( 0, 0 )
  };


  const AF_Script_UniRangeRec  af_rohg_uniranges[] =
  {
    AF_UNIRANGE_REC( 0x10D00, 0x10D3F ),   /* Hanifi Rohingya */
    AF_UNIRANGE_REC(       0,       0 )
  };

  const AF_Script_UniRangeRec  af_rohg_nonbase_uniranges[] =
  {
    AF_UNIRANGE_REC( 0, 0 )
  };


  const AF_Script_UniRangeRec  af_saur_uniranges[] =
  {
    AF_UNIRANGE_REC(  0xA880,  0xA8DF ),   /* Saurashtra */
    AF_UNIRANGE_REC(       0,       0 )
  };

  const AF_Script_UniRangeRec  af_saur_nonbase_uniranges[] =
  {
    AF_UNIRANGE_REC(  0xA880,  0xA881 ),
    AF_UNIRANGE_REC(  0xA8B4,  0xA8C5 ),
    AF_UNIRANGE_REC(       0,       0 )
  };


  const AF_Script_UniRangeRec  af_shaw_uniranges[] =
  {
    AF_UNIRANGE_REC( 0x10450, 0x1047F ),   /* Shavian */
    AF_UNIRANGE_REC(       0,       0 )
  };

  const AF_Script_UniRangeRec  af_shaw_nonbase_uniranges[] =
  {
    AF_UNIRANGE_REC( 0, 0 )
  };


  const AF_Script_UniRangeRec  af_sinh_uniranges[] =
  {
    AF_UNIRANGE_REC(  0x0D80,  0x0DFF ),  /* Sinhala */
    AF_UNIRANGE_REC(       0,       0 )
  };

  const AF_Script_UniRangeRec  af_sinh_nonbase_uniranges[] =
  {
    AF_UNIRANGE_REC(  0x0DCA,  0x0DCA ),
    AF_UNIRANGE_REC(  0x0DD2,  0x0DD6 ),
    AF_UNIRANGE_REC(       0,       0 )
  };


  const AF_Script_UniRangeRec  af_sund_uniranges[] =
  {
    AF_UNIRANGE_REC(  0x1B80,  0x1BBF ), /* Sundanese            */
    AF_UNIRANGE_REC(  0x1CC0,  0x1CCF ), /* Sundanese Supplement */
    AF_UNIRANGE_REC(       0,       0 )
  };

  const AF_Script_UniRangeRec  af_sund_nonbase_uniranges[] =
  {
    AF_UNIRANGE_REC(  0x1B80,  0x1B82 ),
    AF_UNIRANGE_REC(  0x1BA1,  0x1BAD ),
    AF_UNIRANGE_REC(       0,       0 )
  };


  const AF_Script_UniRangeRec  af_taml_uniranges[] =
  {
    AF_UNIRANGE_REC(  0x0B80,  0x0BFF ),  /* Tamil */
    AF_UNIRANGE_REC(       0,       0 )
  };

  const AF_Script_UniRangeRec  af_taml_nonbase_uniranges[] =
  {
    AF_UNIRANGE_REC(  0x0B82,  0x0B82 ),
    AF_UNIRANGE_REC(  0x0BC0,  0x0BC2 ),
    AF_UNIRANGE_REC(  0x0BCD,  0x0BCD ),
    AF_UNIRANGE_REC(       0,       0 )
  };


  const AF_Script_UniRangeRec  af_tavt_uniranges[] =
  {
    AF_UNIRANGE_REC(  0xAA80,  0xAADF ),   /* Tai Viet */
    AF_UNIRANGE_REC(       0,       0 )
  };

  const AF_Script_UniRangeRec  af_tavt_nonbase_uniranges[] =
  {
    AF_UNIRANGE_REC(  0xAAB0,  0xAAB0 ),
    AF_UNIRANGE_REC(  0xAAB2,  0xAAB4 ),
    AF_UNIRANGE_REC(  0xAAB7,  0xAAB8 ),
    AF_UNIRANGE_REC(  0xAABE,  0xAABF ),
    AF_UNIRANGE_REC(  0xAAC1,  0xAAC1 ),
    AF_UNIRANGE_REC(       0,       0 )
  };


  const AF_Script_UniRangeRec  af_telu_uniranges[] =
  {
    AF_UNIRANGE_REC(  0x0C00,  0x0C7F ),  /* Telugu */
    AF_UNIRANGE_REC(       0,       0 )
  };

  const AF_Script_UniRangeRec  af_telu_nonbase_uniranges[] =
  {
    AF_UNIRANGE_REC(  0x0C00,  0x0C00 ),
    AF_UNIRANGE_REC(  0x0C04,  0x0C04 ),
    AF_UNIRANGE_REC(  0x0C3E,  0x0C40 ),
    AF_UNIRANGE_REC(  0x0C46,  0x0C56 ),
    AF_UNIRANGE_REC(  0x0C62,  0x0C63 ),
    AF_UNIRANGE_REC(       0,       0 )
  };


  const AF_Script_UniRangeRec  af_thai_uniranges[] =
  {
    AF_UNIRANGE_REC(  0x0E00,  0x0E7F ),  /* Thai */
    AF_UNIRANGE_REC(       0,       0 )
  };

  const AF_Script_UniRangeRec  af_thai_nonbase_uniranges[] =
  {
    AF_UNIRANGE_REC(  0x0E31,  0x0E31 ),
    AF_UNIRANGE_REC(  0x0E34,  0x0E3A ),
    AF_UNIRANGE_REC(  0x0E47,  0x0E4E ),
    AF_UNIRANGE_REC(       0,       0 )
  };


  const AF_Script_UniRangeRec  af_tfng_uniranges[] =
  {
    AF_UNIRANGE_REC(  0x2D30,  0x2D7F ),   /* Tifinagh */
    AF_UNIRANGE_REC(       0,       0 )
  };

  const AF_Script_UniRangeRec  af_tfng_nonbase_uniranges[] =
  {
    AF_UNIRANGE_REC( 0, 0 )
  };


  const AF_Script_UniRangeRec  af_vaii_uniranges[] =
  {
    AF_UNIRANGE_REC(  0xA500,  0xA63F ),   /* Vai */
    AF_UNIRANGE_REC(       0,       0 )
  };

  const AF_Script_UniRangeRec  af_vaii_nonbase_uniranges[] =
  {
    AF_UNIRANGE_REC( 0, 0 )
  };


#ifdef AF_CONFIG_OPTION_INDIC

  const AF_Script_UniRangeRec  af_limb_uniranges[] =
  {
    AF_UNIRANGE_REC(  0x1900,  0x194F ),  /* Limbu */
    AF_UNIRANGE_REC(       0,       0 )
  };

  const AF_Script_UniRangeRec  af_limb_nonbase_uniranges[] =
  {
    AF_UNIRANGE_REC(  0x1920,  0x1922 ),
    AF_UNIRANGE_REC(  0x1927,  0x1934 ),
    AF_UNIRANGE_REC(  0x1937,  0x193B ),
    AF_UNIRANGE_REC(       0,       0 )
  };


  const AF_Script_UniRangeRec  af_orya_uniranges[] =
  {
    AF_UNIRANGE_REC(  0x0B00,  0x0B7F ),  /* Oriya */
    AF_UNIRANGE_REC(       0,       0 )
  };

  const AF_Script_UniRangeRec  af_orya_nonbase_uniranges[] =
  {
    AF_UNIRANGE_REC(  0x0B01,  0x0B02 ),
    AF_UNIRANGE_REC(  0x0B3C,  0x0B3C ),
    AF_UNIRANGE_REC(  0x0B3F,  0x0B3F ),
    AF_UNIRANGE_REC(  0x0B41,  0x0B44 ),
    AF_UNIRANGE_REC(  0x0B4D,  0x0B56 ),
    AF_UNIRANGE_REC(  0x0B62,  0x0B63 ),
    AF_UNIRANGE_REC(       0,       0 )
  };


  const AF_Script_UniRangeRec  af_sylo_uniranges[] =
  {
    AF_UNIRANGE_REC(  0xA800,  0xA82F ),  /* Syloti Nagri */
    AF_UNIRANGE_REC(       0,       0 )
  };

  const AF_Script_UniRangeRec  af_sylo_nonbase_uniranges[] =
  {
    AF_UNIRANGE_REC(  0xA802,  0xA802 ),
    AF_UNIRANGE_REC(  0xA806,  0xA806 ),
    AF_UNIRANGE_REC(  0xA80B,  0xA80B ),
    AF_UNIRANGE_REC(  0xA825,  0xA826 ),
    AF_UNIRANGE_REC(       0,       0 )
  };


  const AF_Script_UniRangeRec  af_tibt_uniranges[] =
  {
    AF_UNIRANGE_REC(  0x0F00,  0x0FFF ),  /* Tibetan */
    AF_UNIRANGE_REC(       0,       0 )
  };

  const AF_Script_UniRangeRec  af_tibt_nonbase_uniranges[] =
  {
    AF_UNIRANGE_REC(  0x0F18,  0x0F19 ),
    AF_UNIRANGE_REC(  0x0F35,  0x0F35 ),
    AF_UNIRANGE_REC(  0x0F37,  0x0F37 ),
    AF_UNIRANGE_REC(  0x0F39,  0x0F39 ),
    AF_UNIRANGE_REC(  0x0F3E,  0x0F3F ),
    AF_UNIRANGE_REC(  0x0F71,  0x0F7E ),
    AF_UNIRANGE_REC(  0x0F80,  0x0F84 ),
    AF_UNIRANGE_REC(  0x0F86,  0x0F87 ),
    AF_UNIRANGE_REC(  0x0F8D,  0x0FBC ),
    AF_UNIRANGE_REC(       0,       0 )
  };

#endif /* !AF_CONFIG_OPTION_INDIC */

#ifdef AF_CONFIG_OPTION_CJK

  /* this corresponds to Unicode 6.0 */

  const AF_Script_UniRangeRec  af_hani_uniranges[] =
  {
    AF_UNIRANGE_REC(  0x1100,  0x11FF ),  /* Hangul Jamo                             */
    AF_UNIRANGE_REC(  0x2E80,  0x2EFF ),  /* CJK Radicals Supplement                 */
    AF_UNIRANGE_REC(  0x2F00,  0x2FDF ),  /* Kangxi Radicals                         */
    AF_UNIRANGE_REC(  0x2FF0,  0x2FFF ),  /* Ideographic Description Characters      */
    AF_UNIRANGE_REC(  0x3000,  0x303F ),  /* CJK Symbols and Punctuation             */
    AF_UNIRANGE_REC(  0x3040,  0x309F ),  /* Hiragana                                */
    AF_UNIRANGE_REC(  0x30A0,  0x30FF ),  /* Katakana                                */
    AF_UNIRANGE_REC(  0x3100,  0x312F ),  /* Bopomofo                                */
    AF_UNIRANGE_REC(  0x3130,  0x318F ),  /* Hangul Compatibility Jamo               */
    AF_UNIRANGE_REC(  0x3190,  0x319F ),  /* Kanbun                                  */
    AF_UNIRANGE_REC(  0x31A0,  0x31BF ),  /* Bopomofo Extended                       */
    AF_UNIRANGE_REC(  0x31C0,  0x31EF ),  /* CJK Strokes                             */
    AF_UNIRANGE_REC(  0x31F0,  0x31FF ),  /* Katakana Phonetic Extensions            */
    AF_UNIRANGE_REC(  0x3300,  0x33FF ),  /* CJK Compatibility                       */
    AF_UNIRANGE_REC(  0x3400,  0x4DBF ),  /* CJK Unified Ideographs Extension A      */
    AF_UNIRANGE_REC(  0x4DC0,  0x4DFF ),  /* Yijing Hexagram Symbols                 */
    AF_UNIRANGE_REC(  0x4E00,  0x9FFF ),  /* CJK Unified Ideographs                  */
    AF_UNIRANGE_REC(  0xA960,  0xA97F ),  /* Hangul Jamo Extended-A                  */
    AF_UNIRANGE_REC(  0xAC00,  0xD7AF ),  /* Hangul Syllables                        */
    AF_UNIRANGE_REC(  0xD7B0,  0xD7FF ),  /* Hangul Jamo Extended-B                  */
    AF_UNIRANGE_REC(  0xF900,  0xFAFF ),  /* CJK Compatibility Ideographs            */
    AF_UNIRANGE_REC(  0xFE10,  0xFE1F ),  /* Vertical forms                          */
    AF_UNIRANGE_REC(  0xFE30,  0xFE4F ),  /* CJK Compatibility Forms                 */
    AF_UNIRANGE_REC(  0xFF00,  0xFFEF ),  /* Halfwidth and Fullwidth Forms           */
    AF_UNIRANGE_REC( 0x1B000, 0x1B0FF ),  /* Kana Supplement                         */
    AF_UNIRANGE_REC( 0x1B100, 0x1B12F ),  /* Kana Extended-A                         */
    AF_UNIRANGE_REC( 0x1D300, 0x1D35F ),  /* Tai Xuan Hing Symbols                   */
    AF_UNIRANGE_REC( 0x20000, 0x2A6DF ),  /* CJK Unified Ideographs Extension B      */
    AF_UNIRANGE_REC( 0x2A700, 0x2B73F ),  /* CJK Unified Ideographs Extension C      */
    AF_UNIRANGE_REC( 0x2B740, 0x2B81F ),  /* CJK Unified Ideographs Extension D      */
    AF_UNIRANGE_REC( 0x2B820, 0x2CEAF ),  /* CJK Unified Ideographs Extension E      */
    AF_UNIRANGE_REC( 0x2CEB0, 0x2EBEF ),  /* CJK Unified Ideographs Extension F      */
    AF_UNIRANGE_REC( 0x2F800, 0x2FA1F ),  /* CJK Compatibility Ideographs Supplement */
    AF_UNIRANGE_REC(       0,       0 )
  };

  const AF_Script_UniRangeRec  af_hani_nonbase_uniranges[] =
  {
    AF_UNIRANGE_REC(  0x302A,  0x302F ),
    AF_UNIRANGE_REC(  0x3190,  0x319F ),
    AF_UNIRANGE_REC(       0,       0 )
  };

#endif /* !AF_CONFIG_OPTION_CJK */

/* END */
