/***************************************************************************/
/*                                                                         */
/*  ttgxvar.c                                                              */
/*                                                                         */
/*    TrueType GX Font Variation loader                                    */
/*                                                                         */
/*  Copyright 2004-2013 by                                                 */
/*  David Turner, Robert Wilhelm, Werner Lemberg, and George Williams.     */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


  /*************************************************************************/
  /*                                                                       */
  /* Apple documents the `fvar', `gvar', `cvar', and `avar' tables at      */
  /*                                                                       */
  /*   http://developer.apple.com/fonts/TTRefMan/RM06/Chap6[fgca]var.html  */
  /*                                                                       */
  /* The documentation for `fvar' is inconsistent.  At one point it says   */
  /* that `countSizePairs' should be 3, at another point 2.  It should     */
  /* be 2.                                                                 */
  /*                                                                       */
  /* The documentation for `gvar' is not intelligible; `cvar' refers you   */
  /* to `gvar' and is thus also incomprehensible.                          */
  /*                                                                       */
  /* The documentation for `avar' appears correct, but Apple has no fonts  */
  /* with an `avar' table, so it is hard to test.                          */
  /*                                                                       */
  /* Many thanks to John Jenkins (at Apple) in figuring this out.          */
  /*                                                                       */
  /*                                                                       */
  /* Apple's `kern' table has some references to tuple indices, but as     */
  /* there is no indication where these indices are defined, nor how to    */
  /* interpolate the kerning values (different tuples have different       */
  /* classes) this issue is ignored.                                       */
  /*                                                                       */
  /*************************************************************************/


#include <ft2build.h>
#include FT_INTERNAL_DEBUG_H
#include FT_CONFIG_CONFIG_H
#include FT_INTERNAL_STREAM_H
#include FT_INTERNAL_SFNT_H
#include FT_TRUETYPE_TAGS_H
#include FT_MULTIPLE_MASTERS_H

#include "ttpload.h"
#include "ttgxvar.h"

#include "tterrors.h"


#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT


#define FT_Stream_FTell( stream )  \
          (FT_ULong)( (stream)->cursor - (stream)->base )
#define FT_Stream_SeekSet( stream, off ) \
          ( (stream)->cursor = (stream)->base + (off) )


  /*************************************************************************/
  /*                                                                       */
  /* The macro FT_COMPONENT is used in trace mode.  It is an implicit      */
  /* parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log  */
  /* messages during execution.                                            */
  /*                                                                       */
#undef  FT_COMPONENT
#define FT_COMPONENT  trace_ttgxvar


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                       Internal Routines                       *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/


  /*************************************************************************/
  /*                                                                       */
  /* The macro ALL_POINTS is used in `ft_var_readpackedpoints'.  It        */
  /* indicates that there is a delta for every point without needing to    */
  /* enumerate all of them.                                                */
  /*                                                                       */

  /* ensure that value `0' has the same width as a pointer */
#define ALL_POINTS  (FT_UShort*)~(FT_PtrDist)0


#define GX_PT_POINTS_ARE_WORDS      0x80
#define GX_PT_POINT_RUN_COUNT_MASK  0x7F


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    ft_var_readpackedpoints                                            */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Read a set of points to which the following deltas will apply.     */
  /*    Points are packed with a run length encoding.                      */
  /*                                                                       */
  /* <Input>                                                               */
  /*    stream    :: The data stream.                                      */
  /*                                                                       */
  /* <Output>                                                              */
  /*    point_cnt :: The number of points read.  A zero value means that   */
  /*                 all points in the glyph will be affected, without     */
  /*                 enumerating them individually.                        */
  /*                                                                       */
  /* <Return>                                                              */
  /*    An array of FT_UShort containing the affected points or the        */
  /*    special value ALL_POINTS.                                          */
  /*                                                                       */
  static FT_UShort*
  ft_var_readpackedpoints( FT_Stream  stream,
                           FT_UInt   *point_cnt )
  {
    FT_UShort *points = NULL;
    FT_Int     n;
    FT_Int     runcnt;
    FT_Int     i;
    FT_Int     j;
    FT_Int     first;
    FT_Memory  memory = stream->memory;
    FT_Error   error  = FT_Err_Ok;

    FT_UNUSED( error );


    *point_cnt = n = FT_GET_BYTE();
    if ( n == 0 )
      return ALL_POINTS;

    if ( n & GX_PT_POINTS_ARE_WORDS )
      n = FT_GET_BYTE() | ( ( n & GX_PT_POINT_RUN_COUNT_MASK ) << 8 );

    if ( FT_NEW_ARRAY( points, n ) )
      return NULL;

    i = 0;
    while ( i < n )
    {
      runcnt = FT_GET_BYTE();
      if ( runcnt & GX_PT_POINTS_ARE_WORDS )
      {
        runcnt = runcnt & GX_PT_POINT_RUN_COUNT_MASK;
        first  = points[i++] = FT_GET_USHORT();

        if ( runcnt < 1 || i + runcnt >= n )
          goto Exit;

        /* first point not included in runcount */
        for ( j = 0; j < runcnt; ++j )
          points[i++] = (FT_UShort)( first += FT_GET_USHORT() );
      }
      else
      {
        first = points[i++] = FT_GET_BYTE();

        if ( runcnt < 1 || i + runcnt >= n )
          goto Exit;

        for ( j = 0; j < runcnt; ++j )
          points[i++] = (FT_UShort)( first += FT_GET_BYTE() );
      }
    }

  Exit:
    return points;
  }


  enum
  {
    GX_DT_DELTAS_ARE_ZERO      = 0x80,
    GX_DT_DELTAS_ARE_WORDS     = 0x40,
    GX_DT_DELTA_RUN_COUNT_MASK = 0x3F
  };


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    ft_var_readpackeddeltas                                            */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Read a set of deltas.  These are packed slightly differently than  */
  /*    points.  In particular there is no overall count.                  */
  /*                                                                       */
  /* <Input>                                                               */
  /*    stream    :: The data stream.                                      */
  /*                                                                       */
  /*    delta_cnt :: The number of to be read.                             */
  /*                                                                       */
  /* <Return>                                                              */
  /*    An array of FT_Short containing the deltas for the affected        */
  /*    points.  (This only gets the deltas for one dimension.  It will    */
  /*    generally be called twice, once for x, once for y.  When used in   */
  /*    cvt table, it will only be called once.)                           */
  /*                                                                       */
  static FT_Short*
  ft_var_readpackeddeltas( FT_Stream  stream,
                           FT_Offset  delta_cnt )
  {
    FT_Short  *deltas = NULL;
    FT_UInt    runcnt;
    FT_Offset  i;
    FT_UInt    j;
    FT_Memory  memory = stream->memory;
    FT_Error   error  = FT_Err_Ok;

    FT_UNUSED( error );


    if ( FT_NEW_ARRAY( deltas, delta_cnt ) )
      return NULL;

    i = 0;
    while ( i < delta_cnt )
    {
      runcnt = FT_GET_BYTE();
      if ( runcnt & GX_DT_DELTAS_ARE_ZERO )
      {
        /* runcnt zeroes get added */
        for ( j = 0;
              j <= ( runcnt & GX_DT_DELTA_RUN_COUNT_MASK ) && i < delta_cnt;
              ++j )
          deltas[i++] = 0;
      }
      else if ( runcnt & GX_DT_DELTAS_ARE_WORDS )
      {
        /* runcnt shorts from the stack */
        for ( j = 0;
              j <= ( runcnt & GX_DT_DELTA_RUN_COUNT_MASK ) && i < delta_cnt;
              ++j )
          deltas[i++] = FT_GET_SHORT();
      }
      else
      {
        /* runcnt signed bytes from the stack */
        for ( j = 0;
              j <= ( runcnt & GX_DT_DELTA_RUN_COUNT_MASK ) && i < delta_cnt;
              ++j )
          deltas[i++] = FT_GET_CHAR();
      }

      if ( j <= ( runcnt & GX_DT_DELTA_RUN_COUNT_MASK ) )
      {
        /* Bad format */
        FT_FREE( deltas );
        return NULL;
      }
    }

    return deltas;
  }


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    ft_var_load_avar                                                   */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Parse the `avar' table if present.  It need not be, so we return   */
  /*    nothing.                                                           */
  /*                                                                       */
  /* <InOut>                                                               */
  /*    face :: The font face.                                             */
  /*                                                                       */
  static void
  ft_var_load_avar( TT_Face  face )
  {
    FT_Stream       stream = FT_FACE_STREAM(face);
    FT_Memory       memory = stream->memory;
    GX_Blend        blend  = face->blend;
    GX_AVarSegment  segment;
    FT_Error        error = FT_Err_Ok;
    FT_ULong        version;
    FT_Long         axisCount;
    FT_Int          i, j;
    FT_ULong        table_len;

    FT_UNUSED( error );


    blend->avar_checked = TRUE;
    if ( (error = face->goto_table( face, TTAG_avar, stream, &table_len )) != 0 )
      return;

    if ( FT_FRAME_ENTER( table_len ) )
      return;

    version   = FT_GET_LONG();
    axisCount = FT_GET_LONG();

    if ( version != 0x00010000L                       ||
         axisCount != (FT_Long)blend->mmvar->num_axis )
      goto Exit;

    if ( FT_NEW_ARRAY( blend->avar_segment, axisCount ) )
      goto Exit;

    segment = &blend->avar_segment[0];
    for ( i = 0; i < axisCount; ++i, ++segment )
    {
      segment->pairCount = FT_GET_USHORT();
      if ( FT_NEW_ARRAY( segment->correspondence, segment->pairCount ) )
      {
        /* Failure.  Free everything we have done so far.  We must do */
        /* it right now since loading the `avar' table is optional.   */

        for ( j = i - 1; j >= 0; --j )
          FT_FREE( blend->avar_segment[j].correspondence );

        FT_FREE( blend->avar_segment );
        blend->avar_segment = NULL;
        goto Exit;
      }

      for ( j = 0; j < segment->pairCount; ++j )
      {
        segment->correspondence[j].fromCoord =
          FT_GET_SHORT() << 2;    /* convert to Fixed */
        segment->correspondence[j].toCoord =
          FT_GET_SHORT()<<2;    /* convert to Fixed */
      }
    }

  Exit:
    FT_FRAME_EXIT();
  }


  typedef struct  GX_GVar_Head_
  {
    FT_Long    version;
    FT_UShort  axisCount;
    FT_UShort  globalCoordCount;
    FT_ULong   offsetToCoord;
    FT_UShort  glyphCount;
    FT_UShort  flags;
    FT_ULong   offsetToData;

  } GX_GVar_Head;


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    ft_var_load_gvar                                                   */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Parses the `gvar' table if present.  If `fvar' is there, `gvar'    */
  /*    had better be there too.                                           */
  /*                                                                       */
  /* <InOut>                                                               */
  /*    face :: The font face.                                             */
  /*                                                                       */
  /* <Return>                                                              */
  /*    FreeType error code.  0 means success.                             */
  /*                                                                       */
  static FT_Error
  ft_var_load_gvar( TT_Face  face )
  {
    FT_Stream     stream = FT_FACE_STREAM(face);
    FT_Memory     memory = stream->memory;
    GX_Blend      blend  = face->blend;
    FT_Error      error;
    FT_UInt       i, j;
    FT_ULong      table_len;
    FT_ULong      gvar_start;
    FT_ULong      offsetToData;
    GX_GVar_Head  gvar_head;

    static const FT_Frame_Field  gvar_fields[] =
    {

#undef  FT_STRUCTURE
#define FT_STRUCTURE  GX_GVar_Head

      FT_FRAME_START( 20 ),
        FT_FRAME_LONG  ( version ),
        FT_FRAME_USHORT( axisCount ),
        FT_FRAME_USHORT( globalCoordCount ),
        FT_FRAME_ULONG ( offsetToCoord ),
        FT_FRAME_USHORT( glyphCount ),
        FT_FRAME_USHORT( flags ),
        FT_FRAME_ULONG ( offsetToData ),
      FT_FRAME_END
    };

    if ( (error = face->goto_table( face, TTAG_gvar, stream, &table_len )) != 0 )
      goto Exit;

    gvar_start = FT_STREAM_POS( );
    if ( FT_STREAM_READ_FIELDS( gvar_fields, &gvar_head ) )
      goto Exit;

    blend->tuplecount  = gvar_head.globalCoordCount;
    blend->gv_glyphcnt = gvar_head.glyphCount;
    offsetToData       = gvar_start + gvar_head.offsetToData;

    if ( gvar_head.version   != (FT_Long)0x00010000L              ||
         gvar_head.axisCount != (FT_UShort)blend->mmvar->num_axis )
    {
      error = FT_THROW( Invalid_Table );
      goto Exit;
    }

    if ( FT_NEW_ARRAY( blend->glyphoffsets, blend->gv_glyphcnt + 1 ) )
      goto Exit;

    if ( gvar_head.flags & 1 )
    {
      /* long offsets (one more offset than glyphs, to mark size of last) */
      if ( FT_FRAME_ENTER( ( blend->gv_glyphcnt + 1 ) * 4L ) )
        goto Exit;

      for ( i = 0; i <= blend->gv_glyphcnt; ++i )
        blend->glyphoffsets[i] = offsetToData + FT_GET_LONG();

      FT_FRAME_EXIT();
    }
    else
    {
      /* short offsets (one more offset than glyphs, to mark size of last) */
      if ( FT_FRAME_ENTER( ( blend->gv_glyphcnt + 1 ) * 2L ) )
        goto Exit;

      for ( i = 0; i <= blend->gv_glyphcnt; ++i )
        blend->glyphoffsets[i] = offsetToData + FT_GET_USHORT() * 2;
                                              /* XXX: Undocumented: `*2'! */

      FT_FRAME_EXIT();
    }

    if ( blend->tuplecount != 0 )
    {
      if ( FT_NEW_ARRAY( blend->tuplecoords,
                         gvar_head.axisCount * blend->tuplecount ) )
        goto Exit;

      if ( FT_STREAM_SEEK( gvar_start + gvar_head.offsetToCoord )       ||
           FT_FRAME_ENTER( blend->tuplecount * gvar_head.axisCount * 2L )                   )
        goto Exit;

      for ( i = 0; i < blend->tuplecount; ++i )
        for ( j = 0 ; j < (FT_UInt)gvar_head.axisCount; ++j )
          blend->tuplecoords[i * gvar_head.axisCount + j] =
            FT_GET_SHORT() << 2;                /* convert to FT_Fixed */

      FT_FRAME_EXIT();
    }

  Exit:
    return error;
  }


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    ft_var_apply_tuple                                                 */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Figure out whether a given tuple (design) applies to the current   */
  /*    blend, and if so, what is the scaling factor.                      */
  /*                                                                       */
  /* <Input>                                                               */
  /*    blend           :: The current blend of the font.                  */
  /*                                                                       */
  /*    tupleIndex      :: A flag saying whether this is an intermediate   */
  /*                       tuple or not.                                   */
  /*                                                                       */
  /*    tuple_coords    :: The coordinates of the tuple in normalized axis */
  /*                       units.                                          */
  /*                                                                       */
  /*    im_start_coords :: The initial coordinates where this tuple starts */
  /*                       to apply (for intermediate coordinates).        */
  /*                                                                       */
  /*    im_end_coords   :: The final coordinates after which this tuple no */
  /*                       longer applies (for intermediate coordinates).  */
  /*                                                                       */
  /* <Return>                                                              */
  /*    An FT_Fixed value containing the scaling factor.                   */
  /*                                                                       */
  static FT_Fixed
  ft_var_apply_tuple( GX_Blend   blend,
                      FT_UShort  tupleIndex,
                      FT_Fixed*  tuple_coords,
                      FT_Fixed*  im_start_coords,
                      FT_Fixed*  im_end_coords )
  {
    FT_UInt   i;
    FT_Fixed  apply = 0x10000L;


    for ( i = 0; i < blend->num_axis; ++i )
    {
      if ( tuple_coords[i] == 0 )
        /* It's not clear why (for intermediate tuples) we don't need     */
        /* to check against start/end -- the documentation says we don't. */
        /* Similarly, it's unclear why we don't need to scale along the   */
        /* axis.                                                          */
        continue;

      else if ( blend->normalizedcoords[i] == 0                           ||
                ( blend->normalizedcoords[i] < 0 && tuple_coords[i] > 0 ) ||
                ( blend->normalizedcoords[i] > 0 && tuple_coords[i] < 0 ) )
      {
        apply = 0;
        break;
      }

      else if ( !( tupleIndex & GX_TI_INTERMEDIATE_TUPLE ) )
        /* not an intermediate tuple */
        apply = FT_MulFix( apply,
                           blend->normalizedcoords[i] > 0
                             ? blend->normalizedcoords[i]
                             : -blend->normalizedcoords[i] );

      else if ( blend->normalizedcoords[i] <= im_start_coords[i] ||
                blend->normalizedcoords[i] >= im_end_coords[i]   )
      {
        apply = 0;
        break;
      }

      else if ( blend->normalizedcoords[i] < tuple_coords[i] )
        apply = FT_MulDiv( apply,
                           blend->normalizedcoords[i] - im_start_coords[i],
                           tuple_coords[i] - im_start_coords[i] );

      else
        apply = FT_MulDiv( apply,
                           im_end_coords[i] - blend->normalizedcoords[i],
                           im_end_coords[i] - tuple_coords[i] );
    }

    return apply;
  }


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****               MULTIPLE MASTERS SERVICE FUNCTIONS              *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/


  typedef struct  GX_FVar_Head_
  {
    FT_Long    version;
    FT_UShort  offsetToData;
    FT_UShort  countSizePairs;
    FT_UShort  axisCount;
    FT_UShort  axisSize;
    FT_UShort  instanceCount;
    FT_UShort  instanceSize;

  } GX_FVar_Head;


  typedef struct  fvar_axis_
  {
    FT_ULong   axisTag;
    FT_ULong   minValue;
    FT_ULong   defaultValue;
    FT_ULong   maxValue;
    FT_UShort  flags;
    FT_UShort  nameID;

  } GX_FVar_Axis;


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    TT_Get_MM_Var                                                      */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Check that the font's `fvar' table is valid, parse it, and return  */
  /*    those data.                                                        */
  /*                                                                       */
  /* <InOut>                                                               */
  /*    face   :: The font face.                                           */
  /*              TT_Get_MM_Var initializes the blend structure.           */
  /*                                                                       */
  /* <Output>                                                              */
  /*    master :: The `fvar' data (must be freed by caller).               */
  /*                                                                       */
  /* <Return>                                                              */
  /*    FreeType error code.  0 means success.                             */
  /*                                                                       */
  FT_LOCAL_DEF( FT_Error )
  TT_Get_MM_Var( TT_Face      face,
                 FT_MM_Var*  *master )
  {
    FT_Stream            stream = face->root.stream;
    FT_Memory            memory = face->root.memory;
    FT_ULong             table_len;
    FT_Error             error  = FT_Err_Ok;
    FT_ULong             fvar_start;
    FT_Int               i, j;
    FT_MM_Var*           mmvar = NULL;
    FT_Fixed*            next_coords;
    FT_String*           next_name;
    FT_Var_Axis*         a;
    FT_Var_Named_Style*  ns;
    GX_FVar_Head         fvar_head;

    static const FT_Frame_Field  fvar_fields[] =
    {

#undef  FT_STRUCTURE
#define FT_STRUCTURE  GX_FVar_Head

      FT_FRAME_START( 16 ),
        FT_FRAME_LONG  ( version ),
        FT_FRAME_USHORT( offsetToData ),
        FT_FRAME_USHORT( countSizePairs ),
        FT_FRAME_USHORT( axisCount ),
        FT_FRAME_USHORT( axisSize ),
        FT_FRAME_USHORT( instanceCount ),
        FT_FRAME_USHORT( instanceSize ),
      FT_FRAME_END
    };

    static const FT_Frame_Field  fvaraxis_fields[] =
    {

#undef  FT_STRUCTURE
#define FT_STRUCTURE  GX_FVar_Axis

      FT_FRAME_START( 20 ),
        FT_FRAME_ULONG ( axisTag ),
        FT_FRAME_ULONG ( minValue ),
        FT_FRAME_ULONG ( defaultValue ),
        FT_FRAME_ULONG ( maxValue ),
        FT_FRAME_USHORT( flags ),
        FT_FRAME_USHORT( nameID ),
      FT_FRAME_END
    };


    if ( face->blend == NULL )
    {
      /* both `fvar' and `gvar' must be present */
      if ( (error = face->goto_table( face, TTAG_gvar,
                                      stream, &table_len )) != 0 )
        goto Exit;

      if ( (error = face->goto_table( face, TTAG_fvar,
                                      stream, &table_len )) != 0 )
        goto Exit;

      fvar_start = FT_STREAM_POS( );

      if ( FT_STREAM_READ_FIELDS( fvar_fields, &fvar_head ) )
        goto Exit;

      if ( fvar_head.version != (FT_Long)0x00010000L                      ||
           fvar_head.countSizePairs != 2                                  ||
           fvar_head.axisSize != 20                                       ||
           /* axisCount limit implied by 16-bit instanceSize */
           fvar_head.axisCount > 0x3FFE                                   ||
           fvar_head.instanceSize != 4 + 4 * fvar_head.axisCount          ||
           /* instanceCount limit implied by limited range of name IDs */
           fvar_head.instanceCount > 0x7EFF                               ||
           fvar_head.offsetToData + fvar_head.axisCount * 20U +
             fvar_head.instanceCount * fvar_head.instanceSize > table_len )
      {
        error = FT_THROW( Invalid_Table );
        goto Exit;
      }

      if ( FT_NEW( face->blend ) )
        goto Exit;

      /* cannot overflow 32-bit arithmetic because of limits above */
      face->blend->mmvar_len =
        sizeof ( FT_MM_Var ) +
        fvar_head.axisCount * sizeof ( FT_Var_Axis ) +
        fvar_head.instanceCount * sizeof ( FT_Var_Named_Style ) +
        fvar_head.instanceCount * fvar_head.axisCount * sizeof ( FT_Fixed ) +
        5 * fvar_head.axisCount;

      if ( FT_ALLOC( mmvar, face->blend->mmvar_len ) )
        goto Exit;
      face->blend->mmvar = mmvar;

      mmvar->num_axis =
        fvar_head.axisCount;
      mmvar->num_designs =
        ~0U;                   /* meaningless in this context; each glyph */
                               /* may have a different number of designs  */
                               /* (or tuples, as called by Apple)         */
      mmvar->num_namedstyles =
        fvar_head.instanceCount;
      mmvar->axis =
        (FT_Var_Axis*)&(mmvar[1]);
      mmvar->namedstyle =
        (FT_Var_Named_Style*)&(mmvar->axis[fvar_head.axisCount]);

      next_coords =
        (FT_Fixed*)&(mmvar->namedstyle[fvar_head.instanceCount]);
      for ( i = 0; i < fvar_head.instanceCount; ++i )
      {
        mmvar->namedstyle[i].coords  = next_coords;
        next_coords                 += fvar_head.axisCount;
      }

      next_name = (FT_String*)next_coords;
      for ( i = 0; i < fvar_head.axisCount; ++i )
      {
        mmvar->axis[i].name  = next_name;
        next_name           += 5;
      }

      if ( FT_STREAM_SEEK( fvar_start + fvar_head.offsetToData ) )
        goto Exit;

      a = mmvar->axis;
      for ( i = 0; i < fvar_head.axisCount; ++i )
      {
        GX_FVar_Axis  axis_rec;


        if ( FT_STREAM_READ_FIELDS( fvaraxis_fields, &axis_rec ) )
          goto Exit;
        a->tag     = axis_rec.axisTag;
        a->minimum = axis_rec.minValue;     /* A Fixed */
        a->def     = axis_rec.defaultValue; /* A Fixed */
        a->maximum = axis_rec.maxValue;     /* A Fixed */
        a->strid   = axis_rec.nameID;

        a->name[0] = (FT_String)(   a->tag >> 24 );
        a->name[1] = (FT_String)( ( a->tag >> 16 ) & 0xFF );
        a->name[2] = (FT_String)( ( a->tag >>  8 ) & 0xFF );
        a->name[3] = (FT_String)( ( a->tag       ) & 0xFF );
        a->name[4] = 0;

        ++a;
      }

      ns = mmvar->namedstyle;
      for ( i = 0; i < fvar_head.instanceCount; ++i, ++ns )
      {
        if ( FT_FRAME_ENTER( 4L + 4L * fvar_head.axisCount ) )
          goto Exit;

        ns->strid       =    FT_GET_USHORT();
        (void) /* flags = */ FT_GET_USHORT();

        for ( j = 0; j < fvar_head.axisCount; ++j )
          ns->coords[j] = FT_GET_ULONG();     /* A Fixed */

        FT_FRAME_EXIT();
      }
    }

    if ( master != NULL )
    {
      FT_UInt  n;


      if ( FT_ALLOC( mmvar, face->blend->mmvar_len ) )
        goto Exit;
      FT_MEM_COPY( mmvar, face->blend->mmvar, face->blend->mmvar_len );

      mmvar->axis =
        (FT_Var_Axis*)&(mmvar[1]);
      mmvar->namedstyle =
        (FT_Var_Named_Style*)&(mmvar->axis[mmvar->num_axis]);
      next_coords =
        (FT_Fixed*)&(mmvar->namedstyle[mmvar->num_namedstyles]);

      for ( n = 0; n < mmvar->num_namedstyles; ++n )
      {
        mmvar->namedstyle[n].coords  = next_coords;
        next_coords                 += mmvar->num_axis;
      }

      a = mmvar->axis;
      next_name = (FT_String*)next_coords;
      for ( n = 0; n < mmvar->num_axis; ++n )
      {
        a->name = next_name;

        /* standard PostScript names for some standard apple tags */
        if ( a->tag == TTAG_wght )
          a->name = (char *)"Weight";
        else if ( a->tag == TTAG_wdth )
          a->name = (char *)"Width";
        else if ( a->tag == TTAG_opsz )
          a->name = (char *)"OpticalSize";
        else if ( a->tag == TTAG_slnt )
          a->name = (char *)"Slant";

        next_name += 5;
        ++a;
      }

      *master = mmvar;
    }

  Exit:
    return error;
  }


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    TT_Set_MM_Blend                                                    */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Set the blend (normalized) coordinates for this instance of the    */
  /*    font.  Check that the `gvar' table is reasonable and does some     */
  /*    initial preparation.                                               */
  /*                                                                       */
  /* <InOut>                                                               */
  /*    face       :: The font.                                            */
  /*                  Initialize the blend structure with `gvar' data.     */
  /*                                                                       */
  /* <Input>                                                               */
  /*    num_coords :: Must be the axis count of the font.                  */
  /*                                                                       */
  /*    coords     :: An array of num_coords, each between [-1,1].         */
  /*                                                                       */
  /* <Return>                                                              */
  /*    FreeType error code.  0 means success.                             */
  /*                                                                       */
  FT_LOCAL_DEF( FT_Error )
  TT_Set_MM_Blend( TT_Face    face,
                   FT_UInt    num_coords,
                   FT_Fixed*  coords )
  {
    FT_Error    error = FT_Err_Ok;
    GX_Blend    blend;
    FT_MM_Var*  mmvar;
    FT_UInt     i;
    FT_Memory   memory = face->root.memory;

    enum
    {
      mcvt_retain,
      mcvt_modify,
      mcvt_load

    } manageCvt;


    face->doblend = FALSE;

    if ( face->blend == NULL )
    {
      if ( (error = TT_Get_MM_Var( face, NULL)) != 0 )
        goto Exit;
    }

    blend = face->blend;
    mmvar = blend->mmvar;

    if ( num_coords != mmvar->num_axis )
    {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    for ( i = 0; i < num_coords; ++i )
      if ( coords[i] < -0x00010000L || coords[i] > 0x00010000L )
      {
        error = FT_THROW( Invalid_Argument );
        goto Exit;
      }

    if ( blend->glyphoffsets == NULL )
      if ( (error = ft_var_load_gvar( face )) != 0 )
        goto Exit;

    if ( blend->normalizedcoords == NULL )
    {
      if ( FT_NEW_ARRAY( blend->normalizedcoords, num_coords ) )
        goto Exit;

      manageCvt = mcvt_modify;

      /* If we have not set the blend coordinates before this, then the  */
      /* cvt table will still be what we read from the `cvt ' table and  */
      /* we don't need to reload it.  We may need to change it though... */
    }
    else
    {
      manageCvt = mcvt_retain;
      for ( i = 0; i < num_coords; ++i )
      {
        if ( blend->normalizedcoords[i] != coords[i] )
        {
          manageCvt = mcvt_load;
          break;
        }
      }

      /* If we don't change the blend coords then we don't need to do  */
      /* anything to the cvt table.  It will be correct.  Otherwise we */
      /* no longer have the original cvt (it was modified when we set  */
      /* the blend last time), so we must reload and then modify it.   */
    }

    blend->num_axis = num_coords;
    FT_MEM_COPY( blend->normalizedcoords,
                 coords,
                 num_coords * sizeof ( FT_Fixed ) );

    face->doblend = TRUE;

    if ( face->cvt != NULL )
    {
      switch ( manageCvt )
      {
      case mcvt_load:
        /* The cvt table has been loaded already; every time we change the */
        /* blend we may need to reload and remodify the cvt table.         */
        FT_FREE( face->cvt );
        face->cvt = NULL;

        tt_face_load_cvt( face, face->root.stream );
        break;

      case mcvt_modify:
        /* The original cvt table is in memory.  All we need to do is */
        /* apply the `cvar' table (if any).                           */
        tt_face_vary_cvt( face, face->root.stream );
        break;

      case mcvt_retain:
        /* The cvt table is correct for this set of coordinates. */
        break;
      }
    }

  Exit:
    return error;
  }


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    TT_Set_Var_Design                                                  */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Set the coordinates for the instance, measured in the user         */
  /*    coordinate system.  Parse the `avar' table (if present) to convert */
  /*    from user to normalized coordinates.                               */
  /*                                                                       */
  /* <InOut>                                                               */
  /*    face       :: The font face.                                       */
  /*                  Initialize the blend struct with `gvar' data.        */
  /*                                                                       */
  /* <Input>                                                               */
  /*    num_coords :: This must be the axis count of the font.             */
  /*                                                                       */
  /*    coords     :: A coordinate array with `num_coords' elements.       */
  /*                                                                       */
  /* <Return>                                                              */
  /*    FreeType error code.  0 means success.                             */
  /*                                                                       */
  FT_LOCAL_DEF( FT_Error )
  TT_Set_Var_Design( TT_Face    face,
                     FT_UInt    num_coords,
                     FT_Fixed*  coords )
  {
    FT_Error        error      = FT_Err_Ok;
    FT_Fixed*       normalized = NULL;
    GX_Blend        blend;
    FT_MM_Var*      mmvar;
    FT_UInt         i, j;
    FT_Var_Axis*    a;
    GX_AVarSegment  av;
    FT_Memory       memory = face->root.memory;


    if ( face->blend == NULL )
    {
      if ( (error = TT_Get_MM_Var( face, NULL )) != 0 )
        goto Exit;
    }

    blend = face->blend;
    mmvar = blend->mmvar;

    if ( num_coords != mmvar->num_axis )
    {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    /* Axis normalization is a two stage process.  First we normalize */
    /* based on the [min,def,max] values for the axis to be [-1,0,1]. */
    /* Then, if there's an `avar' table, we renormalize this range.   */

    if ( FT_NEW_ARRAY( normalized, mmvar->num_axis ) )
      goto Exit;

    a = mmvar->axis;
    for ( i = 0; i < mmvar->num_axis; ++i, ++a )
    {
      if ( coords[i] > a->maximum || coords[i] < a->minimum )
      {
        error = FT_THROW( Invalid_Argument );
        goto Exit;
      }

      if ( coords[i] < a->def )
        normalized[i] = -FT_DivFix( coords[i] - a->def, a->minimum - a->def );
      else if ( a->maximum == a->def )
        normalized[i] = 0;
      else
        normalized[i] = FT_DivFix( coords[i] - a->def, a->maximum - a->def );
    }

    if ( !blend->avar_checked )
      ft_var_load_avar( face );

    if ( blend->avar_segment != NULL )
    {
      av = blend->avar_segment;
      for ( i = 0; i < mmvar->num_axis; ++i, ++av )
      {
        for ( j = 1; j < (FT_UInt)av->pairCount; ++j )
          if ( normalized[i] < av->correspondence[j].fromCoord )
          {
            normalized[i] =
              FT_MulDiv( normalized[i] - av->correspondence[j - 1].fromCoord,
                         av->correspondence[j].toCoord -
                           av->correspondence[j - 1].toCoord,
                         av->correspondence[j].fromCoord -
                           av->correspondence[j - 1].fromCoord ) +
              av->correspondence[j - 1].toCoord;
            break;
          }
      }
    }

    error = TT_Set_MM_Blend( face, num_coords, normalized );

  Exit:
    FT_FREE( normalized );
    return error;
  }


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                     GX VAR PARSING ROUTINES                   *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    tt_face_vary_cvt                                                   */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Modify the loaded cvt table according to the `cvar' table and the  */
  /*    font's blend.                                                      */
  /*                                                                       */
  /* <InOut>                                                               */
  /*    face   :: A handle to the target face object.                      */
  /*                                                                       */
  /* <Input>                                                               */
  /*    stream :: A handle to the input stream.                            */
  /*                                                                       */
  /* <Return>                                                              */
  /*    FreeType error code.  0 means success.                             */
  /*                                                                       */
  /*    Most errors are ignored.  It is perfectly valid not to have a      */
  /*    `cvar' table even if there is a `gvar' and `fvar' table.           */
  /*                                                                       */
  FT_LOCAL_DEF( FT_Error )
  tt_face_vary_cvt( TT_Face    face,
                    FT_Stream  stream )
  {
    FT_Error    error;
    FT_Memory   memory = stream->memory;
    FT_ULong    table_start;
    FT_ULong    table_len;
    FT_UInt     tupleCount;
    FT_ULong    offsetToData;
    FT_ULong    here;
    FT_UInt     i, j;
    FT_Fixed*   tuple_coords    = NULL;
    FT_Fixed*   im_start_coords = NULL;
    FT_Fixed*   im_end_coords   = NULL;
    GX_Blend    blend           = face->blend;
    FT_UInt     point_count;
    FT_UShort*  localpoints;
    FT_Short*   deltas;


    FT_TRACE2(( "CVAR " ));

    if ( blend == NULL )
    {
      FT_TRACE2(( "tt_face_vary_cvt: no blend specified\n" ));

      error = FT_Err_Ok;
      goto Exit;
    }

    if ( face->cvt == NULL )
    {
      FT_TRACE2(( "tt_face_vary_cvt: no `cvt ' table\n" ));

      error = FT_Err_Ok;
      goto Exit;
    }

    error = face->goto_table( face, TTAG_cvar, stream, &table_len );
    if ( error )
    {
      FT_TRACE2(( "is missing\n" ));

      error = FT_Err_Ok;
      goto Exit;
    }

    if ( FT_FRAME_ENTER( table_len ) )
    {
      error = FT_Err_Ok;
      goto Exit;
    }

    table_start = FT_Stream_FTell( stream );
    if ( FT_GET_LONG() != 0x00010000L )
    {
      FT_TRACE2(( "bad table version\n" ));

      error = FT_Err_Ok;
      goto FExit;
    }

    if ( FT_NEW_ARRAY( tuple_coords, blend->num_axis )    ||
         FT_NEW_ARRAY( im_start_coords, blend->num_axis ) ||
         FT_NEW_ARRAY( im_end_coords, blend->num_axis )   )
      goto FExit;

    tupleCount   = FT_GET_USHORT();
    offsetToData = table_start + FT_GET_USHORT();

    /* The documentation implies there are flags packed into the        */
    /* tuplecount, but John Jenkins says that shared points don't apply */
    /* to `cvar', and no other flags are defined.                       */

    for ( i = 0; i < ( tupleCount & 0xFFF ); ++i )
    {
      FT_UInt   tupleDataSize;
      FT_UInt   tupleIndex;
      FT_Fixed  apply;


      tupleDataSize = FT_GET_USHORT();
      tupleIndex    = FT_GET_USHORT();

      /* There is no provision here for a global tuple coordinate section, */
      /* so John says.  There are no tuple indices, just embedded tuples.  */

      if ( tupleIndex & GX_TI_EMBEDDED_TUPLE_COORD )
      {
        for ( j = 0; j < blend->num_axis; ++j )
          tuple_coords[j] = FT_GET_SHORT() << 2; /* convert from        */
                                                 /* short frac to fixed */
      }
      else
      {
        /* skip this tuple; it makes no sense */

        if ( tupleIndex & GX_TI_INTERMEDIATE_TUPLE )
          for ( j = 0; j < 2 * blend->num_axis; ++j )
            (void)FT_GET_SHORT();

        offsetToData += tupleDataSize;
        continue;
      }

      if ( tupleIndex & GX_TI_INTERMEDIATE_TUPLE )
      {
        for ( j = 0; j < blend->num_axis; ++j )
          im_start_coords[j] = FT_GET_SHORT() << 2;
        for ( j = 0; j < blend->num_axis; ++j )
          im_end_coords[j] = FT_GET_SHORT() << 2;
      }

      apply = ft_var_apply_tuple( blend,
                                  (FT_UShort)tupleIndex,
                                  tuple_coords,
                                  im_start_coords,
                                  im_end_coords );
      if ( /* tuple isn't active for our blend */
           apply == 0                                    ||
           /* global points not allowed,           */
           /* if they aren't local, makes no sense */
           !( tupleIndex & GX_TI_PRIVATE_POINT_NUMBERS ) )
      {
        offsetToData += tupleDataSize;
        continue;
      }

      here = FT_Stream_FTell( stream );

      FT_Stream_SeekSet( stream, offsetToData );

      localpoints = ft_var_readpackedpoints( stream, &point_count );
      deltas      = ft_var_readpackeddeltas( stream,
                                             point_count == 0 ? face->cvt_size
                                                              : point_count );
      if ( localpoints == NULL || deltas == NULL )
        /* failure, ignore it */;

      else if ( localpoints == ALL_POINTS )
      {
        /* this means that there are deltas for every entry in cvt */
        for ( j = 0; j < face->cvt_size; ++j )
          face->cvt[j] = (FT_Short)( face->cvt[j] +
                                     FT_MulFix( deltas[j], apply ) );
      }

      else
      {
        for ( j = 0; j < point_count; ++j )
        {
          int  pindex = localpoints[j];

          face->cvt[pindex] = (FT_Short)( face->cvt[pindex] +
                                          FT_MulFix( deltas[j], apply ) );
        }
      }

      if ( localpoints != ALL_POINTS )
        FT_FREE( localpoints );
      FT_FREE( deltas );

      offsetToData += tupleDataSize;

      FT_Stream_SeekSet( stream, here );
    }

  FExit:
    FT_FRAME_EXIT();

  Exit:
    FT_FREE( tuple_coords );
    FT_FREE( im_start_coords );
    FT_FREE( im_end_coords );

    return error;
  }


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    TT_Vary_Get_Glyph_Deltas                                           */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Load the appropriate deltas for the current glyph.                 */
  /*                                                                       */
  /* <Input>                                                               */
  /*    face        :: A handle to the target face object.                 */
  /*                                                                       */
  /*    glyph_index :: The index of the glyph being modified.              */
  /*                                                                       */
  /*    n_points    :: The number of the points in the glyph, including    */
  /*                   phantom points.                                     */
  /*                                                                       */
  /* <Output>                                                              */
  /*    deltas      :: The array of points to change.                      */
  /*                                                                       */
  /* <Return>                                                              */
  /*    FreeType error code.  0 means success.                             */
  /*                                                                       */
  FT_LOCAL_DEF( FT_Error )
  TT_Vary_Get_Glyph_Deltas( TT_Face      face,
                            FT_UInt      glyph_index,
                            FT_Vector*  *deltas,
                            FT_UInt      n_points )
  {
    FT_Stream   stream = face->root.stream;
    FT_Memory   memory = stream->memory;
    GX_Blend    blend  = face->blend;
    FT_Vector*  delta_xy = NULL;

    FT_Error    error;
    FT_ULong    glyph_start;
    FT_UInt     tupleCount;
    FT_ULong    offsetToData;
    FT_ULong    here;
    FT_UInt     i, j;
    FT_Fixed*   tuple_coords    = NULL;
    FT_Fixed*   im_start_coords = NULL;
    FT_Fixed*   im_end_coords   = NULL;
    FT_UInt     point_count, spoint_count = 0;
    FT_UShort*  sharedpoints = NULL;
    FT_UShort*  localpoints  = NULL;
    FT_UShort*  points;
    FT_Short    *deltas_x, *deltas_y;


    if ( !face->doblend || blend == NULL )
      return FT_THROW( Invalid_Argument );

    /* to be freed by the caller */
    if ( FT_NEW_ARRAY( delta_xy, n_points ) )
      goto Exit;
    *deltas = delta_xy;

    if ( glyph_index >= blend->gv_glyphcnt      ||
         blend->glyphoffsets[glyph_index] ==
           blend->glyphoffsets[glyph_index + 1] )
      return FT_Err_Ok;               /* no variation data for this glyph */

    if ( FT_STREAM_SEEK( blend->glyphoffsets[glyph_index] )   ||
         FT_FRAME_ENTER( blend->glyphoffsets[glyph_index + 1] -
                           blend->glyphoffsets[glyph_index] ) )
      goto Fail1;

    glyph_start = FT_Stream_FTell( stream );

    /* each set of glyph variation data is formatted similarly to `cvar' */
    /* (except we get shared points and global tuples)                   */

    if ( FT_NEW_ARRAY( tuple_coords, blend->num_axis )    ||
         FT_NEW_ARRAY( im_start_coords, blend->num_axis ) ||
         FT_NEW_ARRAY( im_end_coords, blend->num_axis )   )
      goto Fail2;

    tupleCount   = FT_GET_USHORT();
    offsetToData = glyph_start + FT_GET_USHORT();

    if ( tupleCount & GX_TC_TUPLES_SHARE_POINT_NUMBERS )
    {
      here = FT_Stream_FTell( stream );

      FT_Stream_SeekSet( stream, offsetToData );

      sharedpoints = ft_var_readpackedpoints( stream, &spoint_count );
      offsetToData = FT_Stream_FTell( stream );

      FT_Stream_SeekSet( stream, here );
    }

    for ( i = 0; i < ( tupleCount & GX_TC_TUPLE_COUNT_MASK ); ++i )
    {
      FT_UInt   tupleDataSize;
      FT_UInt   tupleIndex;
      FT_Fixed  apply;


      tupleDataSize = FT_GET_USHORT();
      tupleIndex    = FT_GET_USHORT();

      if ( tupleIndex & GX_TI_EMBEDDED_TUPLE_COORD )
      {
        for ( j = 0; j < blend->num_axis; ++j )
          tuple_coords[j] = FT_GET_SHORT() << 2;  /* convert from        */
                                                  /* short frac to fixed */
      }
      else if ( ( tupleIndex & GX_TI_TUPLE_INDEX_MASK ) >= blend->tuplecount )
      {
        error = FT_THROW( Invalid_Table );
        goto Fail3;
      }
      else
      {
        FT_MEM_COPY(
          tuple_coords,
          &blend->tuplecoords[(tupleIndex & 0xFFF) * blend->num_axis],
          blend->num_axis * sizeof ( FT_Fixed ) );
      }

      if ( tupleIndex & GX_TI_INTERMEDIATE_TUPLE )
      {
        for ( j = 0; j < blend->num_axis; ++j )
          im_start_coords[j] = FT_GET_SHORT() << 2;
        for ( j = 0; j < blend->num_axis; ++j )
          im_end_coords[j] = FT_GET_SHORT() << 2;
      }

      apply = ft_var_apply_tuple( blend,
                                  (FT_UShort)tupleIndex,
                                  tuple_coords,
                                  im_start_coords,
                                  im_end_coords );

      if ( apply == 0 )              /* tuple isn't active for our blend */
      {
        offsetToData += tupleDataSize;
        continue;
      }

      here = FT_Stream_FTell( stream );

      if ( tupleIndex & GX_TI_PRIVATE_POINT_NUMBERS )
      {
        FT_Stream_SeekSet( stream, offsetToData );

        localpoints = ft_var_readpackedpoints( stream, &point_count );
        points      = localpoints;
      }
      else
      {
        points      = sharedpoints;
        point_count = spoint_count;
      }

      deltas_x = ft_var_readpackeddeltas( stream,
                                          point_count == 0 ? n_points
                                                           : point_count );
      deltas_y = ft_var_readpackeddeltas( stream,
                                          point_count == 0 ? n_points
                                                           : point_count );

      if ( points == NULL || deltas_y == NULL || deltas_x == NULL )
        ; /* failure, ignore it */

      else if ( points == ALL_POINTS )
      {
        /* this means that there are deltas for every point in the glyph */
        for ( j = 0; j < n_points; ++j )
        {
          delta_xy[j].x += FT_MulFix( deltas_x[j], apply );
          delta_xy[j].y += FT_MulFix( deltas_y[j], apply );
        }
      }

      else
      {
        for ( j = 0; j < point_count; ++j )
        {
          if ( localpoints[j] >= n_points )
            continue;

          delta_xy[localpoints[j]].x += FT_MulFix( deltas_x[j], apply );
          delta_xy[localpoints[j]].y += FT_MulFix( deltas_y[j], apply );
        }
      }

      if ( localpoints != ALL_POINTS )
        FT_FREE( localpoints );
      FT_FREE( deltas_x );
      FT_FREE( deltas_y );

      offsetToData += tupleDataSize;

      FT_Stream_SeekSet( stream, here );
    }

  Fail3:
    FT_FREE( tuple_coords );
    FT_FREE( im_start_coords );
    FT_FREE( im_end_coords );

  Fail2:
    FT_FRAME_EXIT();

  Fail1:
    if ( error )
    {
      FT_FREE( delta_xy );
      *deltas = NULL;
    }

  Exit:
    return error;
  }


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    tt_done_blend                                                      */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Frees the blend internal data structure.                           */
  /*                                                                       */
  FT_LOCAL_DEF( void )
  tt_done_blend( FT_Memory  memory,
                 GX_Blend   blend )
  {
    if ( blend != NULL )
    {
      FT_UInt  i;


      FT_FREE( blend->normalizedcoords );
      FT_FREE( blend->mmvar );

      if ( blend->avar_segment != NULL )
      {
        for ( i = 0; i < blend->num_axis; ++i )
          FT_FREE( blend->avar_segment[i].correspondence );
        FT_FREE( blend->avar_segment );
      }

      FT_FREE( blend->tuplecoords );
      FT_FREE( blend->glyphoffsets );
      FT_FREE( blend );
    }
  }

#endif /* TT_CONFIG_OPTION_GX_VAR_SUPPORT */


/* END */
