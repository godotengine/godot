/***************************************************************************/
/*                                                                         */
/*  ttgxvar.h                                                              */
/*                                                                         */
/*    TrueType GX Font Variation loader (specification)                    */
/*                                                                         */
/*  Copyright 2004-2016 by                                                 */
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


  /*************************************************************************/
  /*                                                                       */
  /* <Struct>                                                              */
  /*    GX_BlendRec                                                        */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Data for interpolating a font from a distortable font specified    */
  /*    by the GX *var tables ([fgca]var).                                 */
  /*                                                                       */
  /* <Fields>                                                              */
  /*    num_axis         :: The number of axes along which interpolation   */
  /*                         may happen                                    */
  /*                                                                       */
  /*    normalizedcoords :: A normalized value (between [-1,1]) indicating */
  /*                        the contribution along each axis to the final  */
  /*                        interpolated font.                             */
  /*                                                                       */
  typedef struct  GX_BlendRec_
  {
    FT_UInt         num_axis;
    FT_Fixed*       normalizedcoords;

    FT_MM_Var*      mmvar;
    FT_Offset       mmvar_len;

    FT_Bool         avar_checked;
    GX_AVarSegment  avar_segment;

    FT_UInt         tuplecount;      /* shared tuples in `gvar'           */
    FT_Fixed*       tuplecoords;     /* tuplecoords[tuplecount][num_axis] */

    FT_UInt         gv_glyphcnt;
    FT_ULong*       glyphoffsets;

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
  TT_Set_Var_Design( TT_Face    face,
                     FT_UInt    num_coords,
                     FT_Fixed*  coords );

  FT_LOCAL( FT_Error )
  TT_Get_MM_Var( TT_Face      face,
                 FT_MM_Var*  *master );


  FT_LOCAL( FT_Error )
  tt_face_vary_cvt( TT_Face    face,
                    FT_Stream  stream );


  FT_LOCAL( FT_Error )
  TT_Vary_Apply_Glyph_Deltas( TT_Face      face,
                              FT_UInt      glyph_index,
                              FT_Outline*  outline,
                              FT_UInt      n_points );


  FT_LOCAL( void )
  tt_done_blend( FT_Memory  memory,
                 GX_Blend   blend );


FT_END_HEADER


#endif /* TTGXVAR_H_ */


/* END */
