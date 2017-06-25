/***************************************************************************/
/*                                                                         */
/*  t1tokens.h                                                             */
/*                                                                         */
/*    Type 1 tokenizer (specification).                                    */
/*                                                                         */
/*  Copyright 1996-2017 by                                                 */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


#undef  FT_STRUCTURE
#define FT_STRUCTURE  PS_FontInfoRec
#undef  T1CODE
#define T1CODE        T1_FIELD_LOCATION_FONT_INFO

  T1_FIELD_STRING( "version",            version,
                   T1_FIELD_DICT_FONTDICT )
  T1_FIELD_STRING( "Notice",             notice,
                   T1_FIELD_DICT_FONTDICT )
  T1_FIELD_STRING( "FullName",           full_name,
                   T1_FIELD_DICT_FONTDICT )
  T1_FIELD_STRING( "FamilyName",         family_name,
                   T1_FIELD_DICT_FONTDICT )
  T1_FIELD_STRING( "Weight",             weight,
                   T1_FIELD_DICT_FONTDICT )

  /* we use pointers to detect modifications made by synthetic fonts */
  T1_FIELD_NUM   ( "ItalicAngle",        italic_angle,
                   T1_FIELD_DICT_FONTDICT )
  T1_FIELD_BOOL  ( "isFixedPitch",       is_fixed_pitch,
                   T1_FIELD_DICT_FONTDICT )
  T1_FIELD_NUM   ( "UnderlinePosition",  underline_position,
                   T1_FIELD_DICT_FONTDICT )
  T1_FIELD_NUM   ( "UnderlineThickness", underline_thickness,
                   T1_FIELD_DICT_FONTDICT )

#undef  FT_STRUCTURE
#define FT_STRUCTURE  PS_FontExtraRec
#undef  T1CODE
#define T1CODE        T1_FIELD_LOCATION_FONT_EXTRA

  T1_FIELD_NUM   ( "FSType", fs_type,
                   T1_FIELD_DICT_FONTDICT )

#undef  FT_STRUCTURE
#define FT_STRUCTURE  PS_PrivateRec
#undef  T1CODE
#define T1CODE        T1_FIELD_LOCATION_PRIVATE

  T1_FIELD_NUM       ( "UniqueID",         unique_id,
                       T1_FIELD_DICT_FONTDICT | T1_FIELD_DICT_PRIVATE )
  T1_FIELD_NUM       ( "lenIV",            lenIV,
                       T1_FIELD_DICT_PRIVATE )
  T1_FIELD_NUM       ( "LanguageGroup",    language_group,
                       T1_FIELD_DICT_PRIVATE )
  T1_FIELD_NUM       ( "password",         password,
                       T1_FIELD_DICT_PRIVATE )

  T1_FIELD_FIXED_1000( "BlueScale",        blue_scale,
                       T1_FIELD_DICT_PRIVATE )
  T1_FIELD_NUM       ( "BlueShift",        blue_shift,
                       T1_FIELD_DICT_PRIVATE )
  T1_FIELD_NUM       ( "BlueFuzz",         blue_fuzz,
                       T1_FIELD_DICT_PRIVATE )

  T1_FIELD_NUM_TABLE ( "BlueValues",       blue_values,        14,
                       T1_FIELD_DICT_PRIVATE )
  T1_FIELD_NUM_TABLE ( "OtherBlues",       other_blues,        10,
                       T1_FIELD_DICT_PRIVATE )
  T1_FIELD_NUM_TABLE ( "FamilyBlues",      family_blues,       14,
                       T1_FIELD_DICT_PRIVATE )
  T1_FIELD_NUM_TABLE ( "FamilyOtherBlues", family_other_blues, 10,
                       T1_FIELD_DICT_PRIVATE )

  T1_FIELD_NUM_TABLE2( "StdHW",            standard_width,      1,
                       T1_FIELD_DICT_PRIVATE )
  T1_FIELD_NUM_TABLE2( "StdVW",            standard_height,     1,
                       T1_FIELD_DICT_PRIVATE )
  T1_FIELD_NUM_TABLE2( "MinFeature",       min_feature,         2,
                       T1_FIELD_DICT_PRIVATE )

  T1_FIELD_NUM_TABLE ( "StemSnapH",        snap_widths,        12,
                       T1_FIELD_DICT_PRIVATE )
  T1_FIELD_NUM_TABLE ( "StemSnapV",        snap_heights,       12,
                       T1_FIELD_DICT_PRIVATE )

  T1_FIELD_FIXED     ( "ExpansionFactor",  expansion_factor,
                       T1_FIELD_DICT_PRIVATE )
  T1_FIELD_BOOL      ( "ForceBold",        force_bold,
                       T1_FIELD_DICT_PRIVATE )


#undef  FT_STRUCTURE
#define FT_STRUCTURE  T1_FontRec
#undef  T1CODE
#define T1CODE        T1_FIELD_LOCATION_FONT_DICT

  T1_FIELD_KEY  ( "FontName",    font_name,    T1_FIELD_DICT_FONTDICT )
  T1_FIELD_NUM  ( "PaintType",   paint_type,   T1_FIELD_DICT_FONTDICT )
  T1_FIELD_NUM  ( "FontType",    font_type,    T1_FIELD_DICT_FONTDICT )
  T1_FIELD_FIXED( "StrokeWidth", stroke_width, T1_FIELD_DICT_FONTDICT )


#undef  FT_STRUCTURE
#define FT_STRUCTURE  FT_BBox
#undef  T1CODE
#define T1CODE        T1_FIELD_LOCATION_BBOX

  T1_FIELD_BBOX( "FontBBox", xMin, T1_FIELD_DICT_FONTDICT )


#ifndef T1_CONFIG_OPTION_NO_MM_SUPPORT

#undef  FT_STRUCTURE
#define FT_STRUCTURE  T1_FaceRec
#undef  T1CODE
#define T1CODE        T1_FIELD_LOCATION_FACE

  T1_FIELD_NUM( "NDV", ndv_idx, T1_FIELD_DICT_PRIVATE )
  T1_FIELD_NUM( "CDV", cdv_idx, T1_FIELD_DICT_PRIVATE )


#undef  FT_STRUCTURE
#define FT_STRUCTURE  PS_BlendRec
#undef  T1CODE
#define T1CODE        T1_FIELD_LOCATION_BLEND

  T1_FIELD_NUM_TABLE( "DesignVector", default_design_vector,
                      T1_MAX_MM_DESIGNS, T1_FIELD_DICT_FONTDICT )


#endif /* T1_CONFIG_OPTION_NO_MM_SUPPORT */


/* END */
