/****************************************************************************
 *
 * t1types.h
 *
 *   Basic Type1/Type2 type definitions and interface (specification
 *   only).
 *
 * Copyright (C) 1996-2025 by
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


#include <freetype/ftmm.h>
#include <freetype/internal/pshints.h>
#include <freetype/internal/ftserv.h>
#include <freetype/internal/fthash.h>
#include <freetype/internal/services/svpscmap.h>


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


  /* this structure is used to store the BlendDesignMap entry for an axis */
  typedef struct  PS_DesignMap_
  {
    FT_Byte    num_points;
    FT_Long*   design_points;
    FT_Fixed*  blend_points;

  } PS_DesignMapRec, *PS_DesignMap;

  /* backward compatible definition */
  typedef PS_DesignMapRec  T1_DesignMap;


  typedef struct  PS_BlendRec_
  {
    FT_UInt          num_designs;
    FT_UInt          num_axis;

    FT_String*       axis_names[T1_MAX_MM_AXIS];
    FT_Fixed*        design_pos[T1_MAX_MM_DESIGNS];
    PS_DesignMapRec  design_map[T1_MAX_MM_AXIS];

    FT_Fixed*        weight_vector;
    FT_Fixed*        default_weight_vector;

    PS_FontInfo      font_infos[T1_MAX_MM_DESIGNS + 1];
    PS_Private       privates  [T1_MAX_MM_DESIGNS + 1];

    FT_ULong         blend_bitflags;

    FT_BBox*         bboxes    [T1_MAX_MM_DESIGNS + 1];

    /* since 2.3.0 */

    /* undocumented, optional: the default design instance;   */
    /* corresponds to default_weight_vector --                */
    /* num_default_design_vector == 0 means it is not present */
    /* in the font and associated metrics files               */
    FT_UInt          default_design_vector[T1_MAX_MM_DESIGNS];
    FT_UInt          num_default_design_vector;

  } PS_BlendRec, *PS_Blend;


  /* backward compatible definition */
  typedef PS_BlendRec  T1_Blend;


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
    FT_Fixed       Ascender;     /* optional, mind the zero */
    FT_Fixed       Descender;    /* optional, mind the zero */
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
    FT_FaceRec     root;
    T1_FontRec     type1;
    const void*    psnames;
    const void*    psaux;
    const void*    afm_data;
    FT_CharMapRec  charmaprecs[2];
    FT_CharMap     charmaps[2];

    /* support for Multiple Masters fonts */
    PS_Blend       blend;

    /* undocumented, optional: indices of subroutines that express      */
    /* the NormalizeDesignVector and the ConvertDesignVector procedure, */
    /* respectively, as Type 2 charstrings; -1 if keywords not present  */
    FT_Int         ndv_idx;
    FT_Int         cdv_idx;

    /* undocumented, optional: has the same meaning as len_buildchar */
    /* for Type 2 fonts; manipulated by othersubrs 19, 24, and 25    */
    FT_UInt        len_buildchar;
    FT_Long*       buildchar;

    /* since version 2.1 - interface to PostScript hinter */
    const void*    pshinter;

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
