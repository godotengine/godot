/***************************************************************************/
/*                                                                         */
/*  cidtoken.h                                                             */
/*                                                                         */
/*    CID token definitions (specification only).                          */
/*                                                                         */
/*  Copyright 1996-2018 by                                                 */
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
#define FT_STRUCTURE  CID_FaceInfoRec
#undef  T1CODE
#define T1CODE        T1_FIELD_LOCATION_CID_INFO

  T1_FIELD_KEY   ( "CIDFontName",    cid_font_name, 0 )
  T1_FIELD_FIXED ( "CIDFontVersion", cid_version,   0 )
  T1_FIELD_NUM   ( "CIDFontType",    cid_font_type, 0 )
  T1_FIELD_STRING( "Registry",       registry,      0 )
  T1_FIELD_STRING( "Ordering",       ordering,      0 )
  T1_FIELD_NUM   ( "Supplement",     supplement,    0 )
  T1_FIELD_NUM   ( "UIDBase",        uid_base,      0 )
  T1_FIELD_NUM   ( "CIDMapOffset",   cidmap_offset, 0 )
  T1_FIELD_NUM   ( "FDBytes",        fd_bytes,      0 )
  T1_FIELD_NUM   ( "GDBytes",        gd_bytes,      0 )
  T1_FIELD_NUM   ( "CIDCount",       cid_count,     0 )


#undef  FT_STRUCTURE
#define FT_STRUCTURE  PS_FontInfoRec
#undef  T1CODE
#define T1CODE        T1_FIELD_LOCATION_FONT_INFO

  T1_FIELD_STRING( "version",            version,             0 )
  T1_FIELD_STRING( "Notice",             notice,              0 )
  T1_FIELD_STRING( "FullName",           full_name,           0 )
  T1_FIELD_STRING( "FamilyName",         family_name,         0 )
  T1_FIELD_STRING( "Weight",             weight,              0 )
  T1_FIELD_NUM   ( "ItalicAngle",        italic_angle,        0 )
  T1_FIELD_BOOL  ( "isFixedPitch",       is_fixed_pitch,      0 )
  T1_FIELD_NUM   ( "UnderlinePosition",  underline_position,  0 )
  T1_FIELD_NUM   ( "UnderlineThickness", underline_thickness, 0 )

#undef  FT_STRUCTURE
#define FT_STRUCTURE  PS_FontExtraRec
#undef  T1CODE
#define T1CODE        T1_FIELD_LOCATION_FONT_EXTRA

  T1_FIELD_NUM   ( "FSType",             fs_type,             0 )


#undef  FT_STRUCTURE
#define FT_STRUCTURE  CID_FaceDictRec
#undef  T1CODE
#define T1CODE        T1_FIELD_LOCATION_FONT_DICT

  T1_FIELD_NUM  ( "PaintType",          paint_type,          0 )
  T1_FIELD_NUM  ( "FontType",           font_type,           0 )
  T1_FIELD_NUM  ( "SubrMapOffset",      subrmap_offset,      0 )
  T1_FIELD_NUM  ( "SDBytes",            sd_bytes,            0 )
  T1_FIELD_NUM  ( "SubrCount",          num_subrs,           0 )
  T1_FIELD_NUM  ( "lenBuildCharArray",  len_buildchar,       0 )
  T1_FIELD_FIXED( "ForceBoldThreshold", forcebold_threshold, 0 )
  T1_FIELD_FIXED( "StrokeWidth",        stroke_width,        0 )


#undef  FT_STRUCTURE
#define FT_STRUCTURE  PS_PrivateRec
#undef  T1CODE
#define T1CODE        T1_FIELD_LOCATION_PRIVATE

  T1_FIELD_NUM       ( "UniqueID",         unique_id,      0 )
  T1_FIELD_NUM       ( "lenIV",            lenIV,          0 )
  T1_FIELD_NUM       ( "LanguageGroup",    language_group, 0 )
  T1_FIELD_NUM       ( "password",         password,       0 )

  T1_FIELD_FIXED_1000( "BlueScale",        blue_scale,     0 )
  T1_FIELD_NUM       ( "BlueShift",        blue_shift,     0 )
  T1_FIELD_NUM       ( "BlueFuzz",         blue_fuzz,      0 )

  T1_FIELD_NUM_TABLE ( "BlueValues",       blue_values,        14, 0 )
  T1_FIELD_NUM_TABLE ( "OtherBlues",       other_blues,        10, 0 )
  T1_FIELD_NUM_TABLE ( "FamilyBlues",      family_blues,       14, 0 )
  T1_FIELD_NUM_TABLE ( "FamilyOtherBlues", family_other_blues, 10, 0 )

  T1_FIELD_NUM_TABLE2( "StdHW",            standard_width,      1, 0 )
  T1_FIELD_NUM_TABLE2( "StdVW",            standard_height,     1, 0 )
  T1_FIELD_NUM_TABLE2( "MinFeature",       min_feature,         2, 0 )

  T1_FIELD_NUM_TABLE ( "StemSnapH",        snap_widths,        12, 0 )
  T1_FIELD_NUM_TABLE ( "StemSnapV",        snap_heights,       12, 0 )

  T1_FIELD_BOOL      ( "ForceBold",        force_bold,          0 )


#undef  FT_STRUCTURE
#define FT_STRUCTURE  FT_BBox
#undef  T1CODE
#define T1CODE        T1_FIELD_LOCATION_BBOX

  T1_FIELD_BBOX( "FontBBox", xMin, 0 )


/* END */
