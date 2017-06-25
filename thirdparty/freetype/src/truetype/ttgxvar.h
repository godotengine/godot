/***************************************************************************/
/*                                                                         */
/*  ttgxvar.h                                                              */
/*                                                                         */
/*    TrueType GX Font Variation loader (specification)                    */
/*                                                                         */
/*  Copyright 2004-2017 by                                                 */
/*  David Turner, Robert Wilhelm, Werner Lemberg and George Williams.      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


#ifndef TTGXVAR_H_
#define TTGXVAR_H_


#include <ft2build.h>
#include "ttobjs.h"


FT_BEGIN_HEADER


  /*************************************************************************/
  /*                                                                       */
  /* <Struct>                                                              */
  /*    GX_AVarCorrespondenceRec                                           */
  /*                                                                       */
  /* <Description>                                                         */
  /*    A data structure representing `shortFracCorrespondence' in `avar'  */
  /*    table according to the specifications from Apple.                  */
  /*                                                                       */
  typedef struct  GX_AVarCorrespondenceRec_
  {
    FT_Fixed  fromCoord;
    FT_Fixed  toCoord;

  } GX_AVarCorrespondenceRec_, *GX_AVarCorrespondence;


  /*************************************************************************/
  /*                                                                       */
  /* <Struct>                                                              */
  /*    GX_AVarRec                                                         */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Data from the segment field of `avar' table.                       */
  /*    There is one of these for each axis.                               */
  /*                                                                       */
  typedef struct  GX_AVarSegmentRec_
  {
    FT_UShort              pairCount;
    GX_AVarCorrespondence  correspondence; /* array with pairCount entries */

  } GX_AVarSegmentRec, *GX_AVarSegment;


  typedef struct  GX_ItemVarDataRec_
  {
    FT_UInt    itemCount;      /* number of delta sets per item         */
    FT_UInt    regionIdxCount; /* number of region indices in this data */
    FT_UInt*   regionIndices;  /* array of `regionCount' indices;       */
                               /* these index `varRegionList'           */
    FT_Short*  deltaSet;       /* array of `itemCount' deltas           */
                               /* use `innerIndex' for this array       */

  } GX_ItemVarDataRec, *GX_ItemVarData;


  /* contribution of one axis to a region */
  typedef struct  GX_AxisCoordsRec_
  {
    FT_Fixed  startCoord;
    FT_Fixed  peakCoord;      /* zero means no effect (factor = 1) */
    FT_Fixed  endCoord;

  } GX_AxisCoordsRec, *GX_AxisCoords;


  typedef struct  GX_VarRegionRec_
  {
    GX_AxisCoords  axisList;               /* array of axisCount records */

  } GX_VarRegionRec, *GX_VarRegion;


  /* item variation store */
  typedef struct  GX_ItemVarStoreRec_
  {
    FT_UInt         dataCount;
    GX_ItemVarData  varData;            /* array of dataCount records;     */
                                        /* use `outerIndex' for this array */
    FT_UShort     axisCount;
    FT_UInt       regionCount;          /* total number of regions defined */
    GX_VarRegion  varRegionList;

  } GX_ItemVarStoreRec, *GX_ItemVarStore;


  typedef struct  GX_DeltaSetIdxMapRec_
  {
    FT_UInt   mapCount;
    FT_UInt*  outerIndex;             /* indices to item var data */
    FT_UInt*  innerIndex;             /* indices to delta set     */

  } GX_DeltaSetIdxMapRec, *GX_DeltaSetIdxMap;


  /*************************************************************************/
  /*                                                                       */
  /* <Struct>                                                              */
  /*    GX_HVVarTableRec                                                   */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Data from either the `HVAR' or `VVAR' table.                       */
  /*                                                                       */
  typedef struct  GX_HVVarTableRec_
  {
    GX_ItemVarStoreRec    itemStore;        /* Item Variation Store  */
    GX_DeltaSetIdxMapRec  widthMap;         /* Advance Width Mapping */

#if 0
    GX_DeltaSetIdxMapRec  lsbMap;           /* not implemented */
    GX_DeltaSetIdxMapRec  rsbMap;           /* not implemented */

    GX_DeltaSetIdxMapRec  tsbMap;           /* not implemented */
    GX_DeltaSetIdxMapRec  bsbMap;           /* not implemented */
    GX_DeltaSetIdxMapRec  vorgMap;          /* not implemented */
#endif

  } GX_HVVarTableRec, *GX_HVVarTable;


#define MVAR_TAG_GASP_0  FT_MAKE_TAG( 'g', 's', 'p', '0' )
#define MVAR_TAG_GASP_1  FT_MAKE_TAG( 'g', 's', 'p', '1' )
#define MVAR_TAG_GASP_2  FT_MAKE_TAG( 'g', 's', 'p', '2' )
#define MVAR_TAG_GASP_3  FT_MAKE_TAG( 'g', 's', 'p', '3' )
#define MVAR_TAG_GASP_4  FT_MAKE_TAG( 'g', 's', 'p', '4' )
#define MVAR_TAG_GASP_5  FT_MAKE_TAG( 'g', 's', 'p', '5' )
#define MVAR_TAG_GASP_6  FT_MAKE_TAG( 'g', 's', 'p', '6' )
#define MVAR_TAG_GASP_7  FT_MAKE_TAG( 'g', 's', 'p', '7' )
#define MVAR_TAG_GASP_8  FT_MAKE_TAG( 'g', 's', 'p', '8' )
#define MVAR_TAG_GASP_9  FT_MAKE_TAG( 'g', 's', 'p', '9' )

#define MVAR_TAG_CPHT  FT_MAKE_TAG( 'c', 'p', 'h', 't' )
#define MVAR_TAG_HASC  FT_MAKE_TAG( 'h', 'a', 's', 'c' )
#define MVAR_TAG_HCLA  FT_MAKE_TAG( 'h', 'c', 'l', 'a' )
#define MVAR_TAG_HCLD  FT_MAKE_TAG( 'h', 'c', 'l', 'd' )
#define MVAR_TAG_HCOF  FT_MAKE_TAG( 'h', 'c', 'o', 'f' )
#define MVAR_TAG_HCRN  FT_MAKE_TAG( 'h', 'c', 'r', 'n' )
#define MVAR_TAG_HCRS  FT_MAKE_TAG( 'h', 'c', 'r', 's' )
#define MVAR_TAG_HDSC  FT_MAKE_TAG( 'h', 'd', 's', 'c' )
#define MVAR_TAG_HLGP  FT_MAKE_TAG( 'h', 'l', 'g', 'p' )
#define MVAR_TAG_SBXO  FT_MAKE_TAG( 's', 'b', 'x', 'o' )
#define MVAR_TAG_SBXS  FT_MAKE_TAG( 's', 'b', 'x', 's' )
#define MVAR_TAG_SBYO  FT_MAKE_TAG( 's', 'b', 'y', 'o' )
#define MVAR_TAG_SBYS  FT_MAKE_TAG( 's', 'b', 'y', 's' )
#define MVAR_TAG_SPXO  FT_MAKE_TAG( 's', 'p', 'x', 'o' )
#define MVAR_TAG_SPXS  FT_MAKE_TAG( 's', 'p', 'x', 's' )
#define MVAR_TAG_SPYO  FT_MAKE_TAG( 's', 'p', 'y', 'o' )
#define MVAR_TAG_SPYS  FT_MAKE_TAG( 's', 'p', 'y', 's' )
#define MVAR_TAG_STRO  FT_MAKE_TAG( 's', 't', 'r', 'o' )
#define MVAR_TAG_STRS  FT_MAKE_TAG( 's', 't', 'r', 's' )
#define MVAR_TAG_UNDO  FT_MAKE_TAG( 'u', 'n', 'd', 'o' )
#define MVAR_TAG_UNDS  FT_MAKE_TAG( 'u', 'n', 'd', 's' )
#define MVAR_TAG_VASC  FT_MAKE_TAG( 'v', 'a', 's', 'c' )
#define MVAR_TAG_VCOF  FT_MAKE_TAG( 'v', 'c', 'o', 'f' )
#define MVAR_TAG_VCRN  FT_MAKE_TAG( 'v', 'c', 'r', 'n' )
#define MVAR_TAG_VCRS  FT_MAKE_TAG( 'v', 'c', 'r', 's' )
#define MVAR_TAG_VDSC  FT_MAKE_TAG( 'v', 'd', 's', 'c' )
#define MVAR_TAG_VLGP  FT_MAKE_TAG( 'v', 'l', 'g', 'p' )
#define MVAR_TAG_XHGT  FT_MAKE_TAG( 'x', 'h', 'g', 't' )


  typedef struct  GX_ValueRec_
  {
    FT_ULong   tag;
    FT_UShort  outerIndex;
    FT_UShort  innerIndex;

    FT_Short  unmodified;  /* values are either FT_Short or FT_UShort */

  } GX_ValueRec, *GX_Value;


  /*************************************************************************/
  /*                                                                       */
  /* <Struct>                                                              */
  /*    GX_MVarTableRec                                                    */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Data from the `MVAR' table.                                        */
  /*                                                                       */
  typedef struct  GX_MVarTableRec_
  {
    FT_UShort  valueCount;

    GX_ItemVarStoreRec  itemStore;        /* Item Variation Store  */
    GX_Value            values;           /* Value Records         */

  } GX_MVarTableRec, *GX_MVarTable;


  /*************************************************************************/
  /*                                                                       */
  /* <Struct>                                                              */
  /*    GX_BlendRec                                                        */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Data for interpolating a font from a distortable font specified    */
  /*    by the GX *var tables ([fgcahvm]var).                              */
  /*                                                                       */
  /* <Fields>                                                              */
  /*    num_axis ::                                                        */
  /*      The number of axes along which interpolation may happen.         */
  /*                                                                       */
  /*    coords ::                                                          */
  /*      An array of design coordinates (in user space) indicating the    */
  /*      contribution along each axis to the final interpolated font.     */
  /*      `normalizedcoords' holds the same values.                        */
  /*                                                                       */
  /*    normalizedcoords ::                                                */
  /*      An array of normalized values (between [-1,1]) indicating the    */
  /*      contribution along each axis to the final interpolated font.     */
  /*      `coords' holds the same values.                                  */
  /*                                                                       */
  /*    mmvar ::                                                           */
  /*      Data from the `fvar' table.                                      */
  /*                                                                       */
  /*    mmvar_len ::                                                       */
  /*      The length of the `mmvar' structure.                             */
  /*                                                                       */
  /*    normalized_stylecoords ::                                          */
  /*      A two-dimensional array that holds the named instance data from  */
  /*      `mmvar' as normalized values.                                    */
  /*                                                                       */
  /*    avar_loaded ::                                                     */
  /*      A Boolean; if set, FreeType tried to load (and parse) the `avar' */
  /*      table.                                                           */
  /*                                                                       */
  /*    avar_segment ::                                                    */
  /*      Data from the `avar' table.                                      */
  /*                                                                       */
  /*    hvar_loaded ::                                                     */
  /*      A Boolean; if set, FreeType tried to load (and parse) the `hvar' */
  /*      table.                                                           */
  /*                                                                       */
  /*    hvar_checked ::                                                    */
  /*      A Boolean; if set, FreeType successfully loaded and parsed the   */
  /*      `hvar' table.                                                    */
  /*                                                                       */
  /*    hvar_error ::                                                      */
  /*      If loading and parsing of the `hvar' table failed, this field    */
  /*      holds the corresponding error code.                              */
  /*                                                                       */
  /*    hvar_table ::                                                      */
  /*      Data from the `hvar' table.                                      */
  /*                                                                       */
  /*    vvar_loaded ::                                                     */
  /*      A Boolean; if set, FreeType tried to load (and parse) the `vvar' */
  /*      table.                                                           */
  /*                                                                       */
  /*    vvar_checked ::                                                    */
  /*      A Boolean; if set, FreeType successfully loaded and parsed the   */
  /*      `vvar' table.                                                    */
  /*                                                                       */
  /*    vvar_error ::                                                      */
  /*      If loading and parsing of the `vvar' table failed, this field    */
  /*      holds the corresponding error code.                              */
  /*                                                                       */
  /*    vvar_table ::                                                      */
  /*      Data from the `vvar' table.                                      */
  /*                                                                       */
  /*    mvar_table ::                                                      */
  /*      Data from the `mvar' table.                                      */
  /*                                                                       */
  /*    tuplecount ::                                                      */
  /*      The number of shared tuples in the `gvar' table.                 */
  /*                                                                       */
  /*    tuplecoords ::                                                     */
  /*      A two-dimensional array that holds the shared tuple coordinates  */
  /*      in the `gvar' table.                                             */
  /*                                                                       */
  /*    gv_glyphcnt ::                                                     */
  /*      The number of glyphs handled in the `gvar' table.                */
  /*                                                                       */
  /*    glyphoffsets ::                                                    */
  /*      Offsets into the glyph variation data array.                     */
  /*                                                                       */
  /*    gvar_size ::                                                       */
  /*      The size of the `gvar' table.                                    */
  /*                                                                       */
  typedef struct  GX_BlendRec_
  {
    FT_UInt         num_axis;
    FT_Fixed*       coords;
    FT_Fixed*       normalizedcoords;

    FT_MM_Var*      mmvar;
    FT_Offset       mmvar_len;

    FT_Fixed*       normalized_stylecoords;
                      /* normalized_stylecoords[num_namedstyles][num_axis] */

    FT_Bool         avar_loaded;
    GX_AVarSegment  avar_segment;                /* avar_segment[num_axis] */

    FT_Bool         hvar_loaded;
    FT_Bool         hvar_checked;
    FT_Error        hvar_error;
    GX_HVVarTable   hvar_table;

    FT_Bool         vvar_loaded;
    FT_Bool         vvar_checked;
    FT_Error        vvar_error;
    GX_HVVarTable   vvar_table;

    GX_MVarTable    mvar_table;

    FT_UInt         tuplecount;
    FT_Fixed*       tuplecoords;      /* tuplecoords[tuplecount][num_axis] */

    FT_UInt         gv_glyphcnt;
    FT_ULong*       glyphoffsets;         /* glyphoffsets[gv_glyphcnt + 1] */

    FT_ULong        gvar_size;

  } GX_BlendRec;


  /*************************************************************************/
  /*                                                                       */
  /* <enum>                                                                */
  /*    GX_TupleCountFlags                                                 */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Flags used within the `TupleCount' field of the `gvar' table.      */
  /*                                                                       */
  typedef enum  GX_TupleCountFlags_
  {
    GX_TC_TUPLES_SHARE_POINT_NUMBERS = 0x8000,
    GX_TC_RESERVED_TUPLE_FLAGS       = 0x7000,
    GX_TC_TUPLE_COUNT_MASK           = 0x0FFF

  } GX_TupleCountFlags;


  /*************************************************************************/
  /*                                                                       */
  /* <enum>                                                                */
  /*    GX_TupleIndexFlags                                                 */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Flags used within the `TupleIndex' field of the `gvar' and `cvar'  */
  /*    tables.                                                            */
  /*                                                                       */
  typedef enum  GX_TupleIndexFlags_
  {
    GX_TI_EMBEDDED_TUPLE_COORD  = 0x8000,
    GX_TI_INTERMEDIATE_TUPLE    = 0x4000,
    GX_TI_PRIVATE_POINT_NUMBERS = 0x2000,
    GX_TI_RESERVED_TUPLE_FLAG   = 0x1000,
    GX_TI_TUPLE_INDEX_MASK      = 0x0FFF

  } GX_TupleIndexFlags;


#define TTAG_wght  FT_MAKE_TAG( 'w', 'g', 'h', 't' )
#define TTAG_wdth  FT_MAKE_TAG( 'w', 'd', 't', 'h' )
#define TTAG_opsz  FT_MAKE_TAG( 'o', 'p', 's', 'z' )
#define TTAG_slnt  FT_MAKE_TAG( 's', 'l', 'n', 't' )


  FT_LOCAL( FT_Error )
  TT_Set_MM_Blend( TT_Face    face,
                   FT_UInt    num_coords,
                   FT_Fixed*  coords );

  FT_LOCAL( FT_Error )
  TT_Get_MM_Blend( TT_Face    face,
                   FT_UInt    num_coords,
                   FT_Fixed*  coords );

  FT_LOCAL( FT_Error )
  TT_Set_Var_Design( TT_Face    face,
                     FT_UInt    num_coords,
                     FT_Fixed*  coords );

  FT_LOCAL( FT_Error )
  TT_Get_MM_Var( TT_Face      face,
                 FT_MM_Var*  *master );

  FT_LOCAL( FT_Error )
  TT_Get_Var_Design( TT_Face    face,
                     FT_UInt    num_coords,
                     FT_Fixed*  coords );

  FT_LOCAL( FT_Error )
  tt_face_vary_cvt( TT_Face    face,
                    FT_Stream  stream );


  FT_LOCAL( FT_Error )
  TT_Vary_Apply_Glyph_Deltas( TT_Face      face,
                              FT_UInt      glyph_index,
                              FT_Outline*  outline,
                              FT_UInt      n_points );

  FT_LOCAL( FT_Error )
  tt_hadvance_adjust( TT_Face  face,
                      FT_UInt  gindex,
                      FT_Int  *adelta );

  FT_LOCAL( FT_Error )
  tt_vadvance_adjust( TT_Face  face,
                      FT_UInt  gindex,
                      FT_Int  *adelta );

  FT_LOCAL( void )
  tt_apply_mvar( TT_Face  face );

  FT_LOCAL( FT_Error )
  tt_get_var_blend( TT_Face      face,
                    FT_UInt     *num_coords,
                    FT_Fixed*   *coords,
                    FT_Fixed*   *normalizedcoords,
                    FT_MM_Var*  *mm_var );

  FT_LOCAL( void )
  tt_done_blend( TT_Face  face );


FT_END_HEADER


#endif /* TTGXVAR_H_ */


/* END */
