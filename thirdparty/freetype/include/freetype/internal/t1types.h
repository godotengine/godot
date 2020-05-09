/****************************************************************************
 *
 * t1types.h
 *
 *   Basic Type1/Type2 type definitions and interface (specification
 *   only).
 *
 * Copyright (C) 1996-2020 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#ifndef T1TYPES_H_
#define T1TYPES_H_


#include <ft2build.h>
#include FT_TYPE1_TABLES_H
#include FT_INTERNAL_POSTSCRIPT_HINTS_H
#include FT_INTERNAL_SERVICE_H
#include FT_INTERNAL_HASH_H
#include FT_SERVICE_POSTSCRIPT_CMAPS_H


FT_BEGIN_HEADER


  /*************************************************************************/
  /*************************************************************************/
  /*************************************************************************/
  /***                                                                   ***/
  /***                                                                   ***/
  /***              REQUIRED TYPE1/TYPE2 TABLES DEFINITIONS              ***/
  /***                                                                   ***/
  /***                                                                   ***/
  /*************************************************************************/
  /*************************************************************************/
  /*************************************************************************/


  /**************************************************************************
   *
   * @struct:
   *   T1_EncodingRec
   *
   * @description:
   *   A structure modeling a custom encoding.
   *
   * @fields:
   *   num_chars ::
   *     The number of character codes in the encoding.  Usually 256.
   *
   *   code_first ::
   *     The lowest valid character code in the encoding.
   *
   *   code_last ::
   *     The highest valid character code in the encoding + 1. When equal to
   *     code_first there are no valid character codes.
   *
   *   char_index ::
   *     An array of corresponding glyph indices.
   *
   *   char_name ::
   *     An array of corresponding glyph names.
   */
  typedef struct  T1_EncodingRecRec_
  {
    FT_Int       num_chars;
    FT_Int       code_first;
    FT_Int       code_last;

    FT_UShort*         char_index;
    const FT_String**  char_name;

  } T1_EncodingRec, *T1_Encoding;


  /* used to hold extra data of PS_FontInfoRec that
   * cannot be stored in the publicly defined structure.
   *
   * Note these can't be blended with multiple-masters.
   */
  typedef struct  PS_FontExtraRec_
  {
    FT_UShort  fs_type;

  } PS_FontExtraRec;


  typedef struct  T1_FontRec_
  {
    PS_FontInfoRec   font_info;         /* font info dictionary   */
    PS_FontExtraRec  font_extra;        /* font info extra fields */
    PS_PrivateRec    private_dict;      /* private dictionary     */
    FT_String*       font_name;         /* top-level dictionary   */

    T1_EncodingType  encoding_type;
    T1_EncodingRec   encoding;

    FT_Byte*         subrs_block;
    FT_Byte*         charstrings_block;
    FT_Byte*         glyph_names_block;

    FT_Int           num_subrs;
    FT_Byte**        subrs;
    FT_UInt*         subrs_len;
    FT_Hash          subrs_hash;

    FT_Int           num_glyphs;
    FT_String**      glyph_names;       /* array of glyph names       */
    FT_Byte**        charstrings;       /* array of glyph charstrings */
    FT_UInt*         charstrings_len;

    FT_Byte          paint_type;
    FT_Byte          font_type;
    FT_Matrix        font_matrix;
    FT_Vector        font_offset;
    FT_BBox          font_bbox;
    FT_Long          font_id;

    FT_Fixed         stroke_width;

  } T1_FontRec, *T1_Font;


  typedef struct  CID_SubrsRec_
  {
    FT_Int     num_subrs;
    FT_Byte**  code;

  } CID_SubrsRec, *CID_Subrs;


  /*************************************************************************/
  /*************************************************************************/
  /*************************************************************************/
  /***                                                                   ***/
  /***                                                                   ***/
  /***                AFM FONT INFORMATION STRUCTURES                    ***/
  /***                                                                   ***/
  /***                                                                   ***/
  /*************************************************************************/
  /*************************************************************************/
  /*************************************************************************/

  typedef struct  AFM_TrackKernRec_
  {
    FT_Int    degree;
    FT_Fixed  min_ptsize;
    FT_Fixed  min_kern;
    FT_Fixed  max_ptsize;
    FT_Fixed  max_kern;

  } AFM_TrackKernRec, *AFM_TrackKern;

  typedef struct  AFM_KernPairRec_
  {
    FT_UInt  index1;
    FT_UInt  index2;
    FT_Int   x;
    FT_Int   y;

  } AFM_KernPairRec, *AFM_KernPair;

  typedef struct  AFM_FontInfoRec_
  {
    FT_Bool        IsCIDFont;
    FT_BBox        FontBBox;
    FT_Fixed       Ascender;
    FT_Fixed       Descender;
    AFM_TrackKern  TrackKerns;   /* free if non-NULL */
    FT_UInt        NumTrackKern;
    AFM_KernPair   KernPairs;    /* free if non-NULL */
    FT_UInt        NumKernPair;

  } AFM_FontInfoRec, *AFM_FontInfo;


  /*************************************************************************/
  /*************************************************************************/
  /*************************************************************************/
  /***                                                                   ***/
  /***                                                                   ***/
  /***                ORIGINAL T1_FACE CLASS DEFINITION                  ***/
  /***                                                                   ***/
  /***                                                                   ***/
  /*************************************************************************/
  /*************************************************************************/
  /*************************************************************************/


  typedef struct T1_FaceRec_*   T1_Face;
  typedef struct CID_FaceRec_*  CID_Face;


  typedef struct  T1_FaceRec_
  {
    FT_FaceRec      root;
    T1_FontRec      type1;
    const void*     psnames;
    const void*     psaux;
    const void*     afm_data;
    FT_CharMapRec   charmaprecs[2];
    FT_CharMap      charmaps[2];

    /* support for Multiple Masters fonts */
    PS_Blend        blend;

    /* undocumented, optional: indices of subroutines that express      */
    /* the NormalizeDesignVector and the ConvertDesignVector procedure, */
    /* respectively, as Type 2 charstrings; -1 if keywords not present  */
    FT_Int           ndv_idx;
    FT_Int           cdv_idx;

    /* undocumented, optional: has the same meaning as len_buildchar */
    /* for Type 2 fonts; manipulated by othersubrs 19, 24, and 25    */
    FT_UInt          len_buildchar;
    FT_Long*         buildchar;

    /* since version 2.1 - interface to PostScript hinter */
    const void*     pshinter;

  } T1_FaceRec;


  typedef struct  CID_FaceRec_
  {
    FT_FaceRec       root;
    void*            psnames;
    void*            psaux;
    CID_FaceInfoRec  cid;
    PS_FontExtraRec  font_extra;
#if 0
    void*            afm_data;
#endif
    CID_Subrs        subrs;

    /* since version 2.1 - interface to PostScript hinter */
    void*            pshinter;

    /* since version 2.1.8, but was originally positioned after `afm_data' */
    FT_Byte*         binary_data; /* used if hex data has been converted */
    FT_Stream        cid_stream;

  } CID_FaceRec;


FT_END_HEADER

#endif /* T1TYPES_H_ */


/* END */
