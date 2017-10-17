/***************************************************************************/
/*                                                                         */
/*  cfftoken.h                                                             */
/*                                                                         */
/*    CFF token definitions (specification only).                          */
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
#define FT_STRUCTURE  CFF_FontRecDictRec

#undef  CFFCODE
#define CFFCODE       CFF_CODE_TOPDICT

  CFF_FIELD_STRING  ( 0,     version,             "Version" )
  CFF_FIELD_STRING  ( 1,     notice,              "Notice" )
  CFF_FIELD_STRING  ( 0x100, copyright,           "Copyright" )
  CFF_FIELD_STRING  ( 2,     full_name,           "FullName" )
  CFF_FIELD_STRING  ( 3,     family_name,         "FamilyName" )
  CFF_FIELD_STRING  ( 4,     weight,              "Weight" )
  CFF_FIELD_BOOL    ( 0x101, is_fixed_pitch,      "isFixedPitch" )
  CFF_FIELD_FIXED   ( 0x102, italic_angle,        "ItalicAngle" )
  CFF_FIELD_FIXED   ( 0x103, underline_position,  "UnderlinePosition" )
  CFF_FIELD_FIXED   ( 0x104, underline_thickness, "UnderlineThickness" )
  CFF_FIELD_NUM     ( 0x105, paint_type,          "PaintType" )
  CFF_FIELD_NUM     ( 0x106, charstring_type,     "CharstringType" )
  CFF_FIELD_CALLBACK( 0x107, font_matrix,         "FontMatrix" )
  CFF_FIELD_NUM     ( 13,    unique_id,           "UniqueID" )
  CFF_FIELD_CALLBACK( 5,     font_bbox,           "FontBBox" )
  CFF_FIELD_NUM     ( 0x108, stroke_width,        "StrokeWidth" )
#if 0
  CFF_FIELD_DELTA   ( 14,    xuid, 16,            "XUID" )
#endif
  CFF_FIELD_NUM     ( 15,    charset_offset,      "charset" )
  CFF_FIELD_NUM     ( 16,    encoding_offset,     "Encoding" )
  CFF_FIELD_NUM     ( 17,    charstrings_offset,  "CharStrings" )
  CFF_FIELD_CALLBACK( 18,    private_dict,        "Private" )
  CFF_FIELD_NUM     ( 0x114, synthetic_base,      "SyntheticBase" )
  CFF_FIELD_STRING  ( 0x115, embedded_postscript, "PostScript" )

#if 0
  CFF_FIELD_STRING  ( 0x116, base_font_name,      "BaseFontName" )
  CFF_FIELD_DELTA   ( 0x117, base_font_blend, 16, "BaseFontBlend" )
#endif

  /* the next two operators were removed from the Type2 specification */
  /* in version 16-March-2000                                         */
  CFF_FIELD_CALLBACK( 0x118, multiple_master,     "MultipleMaster" )
#if 0
  CFF_FIELD_CALLBACK( 0x11A, blend_axis_types,    "BlendAxisTypes" )
#endif

  CFF_FIELD_CALLBACK( 0x11E, cid_ros,              "ROS" )
  CFF_FIELD_NUM     ( 0x11F, cid_font_version,     "CIDFontVersion" )
  CFF_FIELD_NUM     ( 0x120, cid_font_revision,    "CIDFontRevision" )
  CFF_FIELD_NUM     ( 0x121, cid_font_type,        "CIDFontType" )
  CFF_FIELD_NUM     ( 0x122, cid_count,            "CIDCount" )
  CFF_FIELD_NUM     ( 0x123, cid_uid_base,         "UIDBase" )
  CFF_FIELD_NUM     ( 0x124, cid_fd_array_offset,  "FDArray" )
  CFF_FIELD_NUM     ( 0x125, cid_fd_select_offset, "FDSelect" )
  CFF_FIELD_STRING  ( 0x126, cid_font_name,        "FontName" )

#if 0
  CFF_FIELD_NUM     ( 0x127, chameleon, "Chameleon" )
#endif


#undef  FT_STRUCTURE
#define FT_STRUCTURE  CFF_PrivateRec
#undef  CFFCODE
#define CFFCODE       CFF_CODE_PRIVATE

  CFF_FIELD_DELTA     ( 6,     blue_values, 14,        "BlueValues" )
  CFF_FIELD_DELTA     ( 7,     other_blues, 10,        "OtherBlues" )
  CFF_FIELD_DELTA     ( 8,     family_blues, 14,       "FamilyBlues" )
  CFF_FIELD_DELTA     ( 9,     family_other_blues, 10, "FamilyOtherBlues" )
  CFF_FIELD_FIXED_1000( 0x109, blue_scale,             "BlueScale" )
  CFF_FIELD_NUM       ( 0x10A, blue_shift,             "BlueShift" )
  CFF_FIELD_NUM       ( 0x10B, blue_fuzz,              "BlueFuzz" )
  CFF_FIELD_NUM       ( 10,    standard_width,         "StdHW" )
  CFF_FIELD_NUM       ( 11,    standard_height,        "StdVW" )
  CFF_FIELD_DELTA     ( 0x10C, snap_widths, 13,        "StemSnapH" )
  CFF_FIELD_DELTA     ( 0x10D, snap_heights, 13,       "StemSnapV" )
  CFF_FIELD_BOOL      ( 0x10E, force_bold,             "ForceBold" )
  CFF_FIELD_FIXED     ( 0x10F, force_bold_threshold,   "ForceBoldThreshold" )
  CFF_FIELD_NUM       ( 0x110, lenIV,                  "lenIV" )
  CFF_FIELD_NUM       ( 0x111, language_group,         "LanguageGroup" )
  CFF_FIELD_FIXED     ( 0x112, expansion_factor,       "ExpansionFactor" )
  CFF_FIELD_NUM       ( 0x113, initial_random_seed,    "initialRandomSeed" )
  CFF_FIELD_NUM       ( 19,    local_subrs_offset,     "Subrs" )
  CFF_FIELD_NUM       ( 20,    default_width,          "defaultWidthX" )
  CFF_FIELD_NUM       ( 21,    nominal_width,          "nominalWidthX" )


#undef  FT_STRUCTURE
#define FT_STRUCTURE  CFF_FontRecDictRec
#undef  CFFCODE
#define CFFCODE       CFF2_CODE_TOPDICT

  CFF_FIELD_CALLBACK( 0x107, font_matrix,          "FontMatrix" )
  CFF_FIELD_NUM     ( 17,    charstrings_offset,   "CharStrings" )
  CFF_FIELD_NUM     ( 0x124, cid_fd_array_offset,  "FDArray" )
  CFF_FIELD_NUM     ( 0x125, cid_fd_select_offset, "FDSelect" )
  CFF_FIELD_NUM     ( 24,    vstore_offset,        "vstore" )
  CFF_FIELD_CALLBACK( 25,    maxstack,             "maxstack" )


#undef  FT_STRUCTURE
#define FT_STRUCTURE  CFF_FontRecDictRec
#undef  CFFCODE
#define CFFCODE       CFF2_CODE_FONTDICT

  CFF_FIELD_CALLBACK( 18,    private_dict, "Private" )
  CFF_FIELD_CALLBACK( 0x107, font_matrix,  "FontMatrix" )


#undef  FT_STRUCTURE
#define FT_STRUCTURE  CFF_PrivateRec
#undef  CFFCODE
#define CFFCODE       CFF2_CODE_PRIVATE

  CFF_FIELD_DELTA     ( 6,     blue_values, 14,        "BlueValues" )
  CFF_FIELD_DELTA     ( 7,     other_blues, 10,        "OtherBlues" )
  CFF_FIELD_DELTA     ( 8,     family_blues, 14,       "FamilyBlues" )
  CFF_FIELD_DELTA     ( 9,     family_other_blues, 10, "FamilyOtherBlues" )
  CFF_FIELD_FIXED_1000( 0x109, blue_scale,             "BlueScale" )
  CFF_FIELD_NUM       ( 0x10A, blue_shift,             "BlueShift" )
  CFF_FIELD_NUM       ( 0x10B, blue_fuzz,              "BlueFuzz" )
  CFF_FIELD_NUM       ( 10,    standard_width,         "StdHW" )
  CFF_FIELD_NUM       ( 11,    standard_height,        "StdVW" )
  CFF_FIELD_DELTA     ( 0x10C, snap_widths, 13,        "StemSnapH" )
  CFF_FIELD_DELTA     ( 0x10D, snap_heights, 13,       "StemSnapV" )
  CFF_FIELD_NUM       ( 0x111, language_group,         "LanguageGroup" )
  CFF_FIELD_FIXED     ( 0x112, expansion_factor,       "ExpansionFactor" )
  CFF_FIELD_CALLBACK  ( 22,    vsindex,                "vsindex" )
  CFF_FIELD_BLEND     ( 23,                            "blend" )
  CFF_FIELD_NUM       ( 19,    local_subrs_offset,     "Subrs" )


/* END */
