/***************************************************************************/
/*                                                                         */
/*  ttgxvar.c                                                              */
/*                                                                         */
/*    TrueType GX Font Variation loader                                    */
/*                                                                         */
/*  Copyright 2004-2017 by                                                 */
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
  /*   https://developer.apple.com/fonts/TrueType-Reference-Manual/RM06/Chap6[fgca]var.html */
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
#include FT_TRUETYPE_IDS_H
#include FT_MULTIPLE_MASTERS_H
#include FT_LIST_H

#include "ttpload.h"
#include "ttgxvar.h"

#include "tterrors.h"


#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT


#define FT_Stream_FTell( stream )                         \
          (FT_ULong)( (stream)->cursor - (stream)->base )
#define FT_Stream_SeekSet( stream, off )                  \
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


#define GX_PT_POINTS_ARE_WORDS      0x80U
#define GX_PT_POINT_RUN_COUNT_MASK  0x7FU


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
  /*    size      :: The size of the table holding the data.               */
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
                           FT_ULong   size,
                           FT_UInt   *point_cnt )
  {
    FT_UShort *points = NULL;
    FT_UInt    n;
    FT_UInt    runcnt;
    FT_UInt    i, j;
    FT_UShort  first;
    FT_Memory  memory = stream->memory;
    FT_Error   error  = FT_Err_Ok;

    FT_UNUSED( error );


    *point_cnt = 0;

    n = FT_GET_BYTE();
    if ( n == 0 )
      return ALL_POINTS;

    if ( n & GX_PT_POINTS_ARE_WORDS )
    {
      n  &= GX_PT_POINT_RUN_COUNT_MASK;
      n <<= 8;
      n  |= FT_GET_BYTE();
    }

    if ( n > size )
    {
      FT_TRACE1(( "ft_var_readpackedpoints: number of points too large\n" ));
      return NULL;
    }

    /* in the nested loops below we increase `i' twice; */
    /* it is faster to simply allocate one more slot    */
    /* than to add another test within the loop         */
    if ( FT_NEW_ARRAY( points, n + 1 ) )
      return NULL;

    *point_cnt = n;

    first = 0;
    i     = 0;
    while ( i < n )
    {
      runcnt = FT_GET_BYTE();
      if ( runcnt & GX_PT_POINTS_ARE_WORDS )
      {
        runcnt     &= GX_PT_POINT_RUN_COUNT_MASK;
        first      += FT_GET_USHORT();
        points[i++] = first;

        /* first point not included in run count */
        for ( j = 0; j < runcnt; j++ )
        {
          first      += FT_GET_USHORT();
          points[i++] = first;
          if ( i >= n )
            break;
        }
      }
      else
      {
        first      += FT_GET_BYTE();
        points[i++] = first;

        for ( j = 0; j < runcnt; j++ )
        {
          first      += FT_GET_BYTE();
          points[i++] = first;
          if ( i >= n )
            break;
        }
      }
    }

    return points;
  }


#define GX_DT_DELTAS_ARE_ZERO       0x80U
#define GX_DT_DELTAS_ARE_WORDS      0x40U
#define GX_DT_DELTA_RUN_COUNT_MASK  0x3FU


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
  /*    size      :: The size of the table holding the data.               */
  /*                                                                       */
  /*    delta_cnt :: The number of deltas to be read.                      */
  /*                                                                       */
  /* <Return>                                                              */
  /*    An array of FT_Short containing the deltas for the affected        */
  /*    points.  (This only gets the deltas for one dimension.  It will    */
  /*    generally be called twice, once for x, once for y.  When used in   */
  /*    cvt table, it will only be called once.)                           */
  /*                                                                       */
  static FT_Short*
  ft_var_readpackeddeltas( FT_Stream  stream,
                           FT_ULong   size,
                           FT_UInt    delta_cnt )
  {
    FT_Short  *deltas = NULL;
    FT_UInt    runcnt, cnt;
    FT_UInt    i, j;
    FT_Memory  memory = stream->memory;
    FT_Error   error  = FT_Err_Ok;

    FT_UNUSED( error );


    if ( delta_cnt > size )
    {
      FT_TRACE1(( "ft_var_readpackeddeltas: number of points too large\n" ));
      return NULL;
    }

    if ( FT_NEW_ARRAY( deltas, delta_cnt ) )
      return NULL;

    i = 0;
    while ( i < delta_cnt )
    {
      runcnt = FT_GET_BYTE();
      cnt    = runcnt & GX_DT_DELTA_RUN_COUNT_MASK;

      if ( runcnt & GX_DT_DELTAS_ARE_ZERO )
      {
        /* `runcnt' zeroes get added */
        for ( j = 0; j <= cnt && i < delta_cnt; j++ )
          deltas[i++] = 0;
      }
      else if ( runcnt & GX_DT_DELTAS_ARE_WORDS )
      {
        /* `runcnt' shorts from the stack */
        for ( j = 0; j <= cnt && i < delta_cnt; j++ )
          deltas[i++] = FT_GET_SHORT();
      }
      else
      {
        /* `runcnt' signed bytes from the stack */
        for ( j = 0; j <= cnt && i < delta_cnt; j++ )
          deltas[i++] = FT_GET_CHAR();
      }

      if ( j <= cnt )
      {
        /* bad format */
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
    FT_Stream       stream = FT_FACE_STREAM( face );
    FT_Memory       memory = stream->memory;
    GX_Blend        blend  = face->blend;
    GX_AVarSegment  segment;
    FT_Error        error = FT_Err_Ok;
    FT_Long         version;
    FT_Long         axisCount;
    FT_Int          i, j;
    FT_ULong        table_len;

    FT_UNUSED( error );


    FT_TRACE2(( "AVAR " ));

    blend->avar_loaded = TRUE;
    error = face->goto_table( face, TTAG_avar, stream, &table_len );
    if ( error )
    {
      FT_TRACE2(( "is missing\n" ));
      return;
    }

    if ( FT_FRAME_ENTER( table_len ) )
      return;

    version   = FT_GET_LONG();
    axisCount = FT_GET_LONG();

    if ( version != 0x00010000L )
    {
      FT_TRACE2(( "bad table version\n" ));
      goto Exit;
    }

    FT_TRACE2(( "loaded\n" ));

    if ( axisCount != (FT_Long)blend->mmvar->num_axis )
    {
      FT_TRACE2(( "ft_var_load_avar: number of axes in `avar' and `fvar'\n"
                  "                  table are different\n" ));
      goto Exit;
    }

    if ( FT_NEW_ARRAY( blend->avar_segment, axisCount ) )
      goto Exit;

    segment = &blend->avar_segment[0];
    for ( i = 0; i < axisCount; i++, segment++ )
    {
      FT_TRACE5(( "  axis %d:\n", i ));

      segment->pairCount = FT_GET_USHORT();
      if ( (FT_ULong)segment->pairCount * 4 > table_len                ||
           FT_NEW_ARRAY( segment->correspondence, segment->pairCount ) )
      {
        /* Failure.  Free everything we have done so far.  We must do */
        /* it right now since loading the `avar' table is optional.   */

        for ( j = i - 1; j >= 0; j-- )
          FT_FREE( blend->avar_segment[j].correspondence );

        FT_FREE( blend->avar_segment );
        blend->avar_segment = NULL;
        goto Exit;
      }

      for ( j = 0; j < segment->pairCount; j++ )
      {
        /* convert to Fixed */
        segment->correspondence[j].fromCoord = FT_GET_SHORT() * 4;
        segment->correspondence[j].toCoord   = FT_GET_SHORT() * 4;

        FT_TRACE5(( "    mapping %.5f to %.5f\n",
                    segment->correspondence[j].fromCoord / 65536.0,
                    segment->correspondence[j].toCoord / 65536.0 ));
      }

      FT_TRACE5(( "\n" ));
    }

  Exit:
    FT_FRAME_EXIT();
  }


  /* some macros we need */
  #define FT_FIXED_ONE  ( (FT_Fixed)0x10000 )

  #define FT_fdot14ToFixed( x )                \
          ( (FT_Fixed)( (FT_ULong)(x) << 2 ) )
  #define FT_intToFixed( i )                    \
          ( (FT_Fixed)( (FT_ULong)(i) << 16 ) )
  #define FT_fixedToInt( x )                                   \
          ( (FT_Short)( ( (FT_UInt32)(x) + 0x8000U ) >> 16 ) )


  static FT_Error
  ft_var_load_item_variation_store( TT_Face          face,
                                    FT_ULong         offset,
                                    GX_ItemVarStore  itemStore )
  {
    FT_Stream  stream = FT_FACE_STREAM( face );
    FT_Memory  memory = stream->memory;

    FT_Error   error;
    FT_UShort  format;
    FT_ULong   region_offset;
    FT_UInt    i, j, k;
    FT_UInt    shortDeltaCount;

    GX_Blend        blend = face->blend;
    GX_ItemVarData  varData;

    FT_ULong*  dataOffsetArray = NULL;


    if ( FT_STREAM_SEEK( offset ) ||
         FT_READ_USHORT( format ) )
      goto Exit;

    if ( format != 1 )
    {
      FT_TRACE2(( "ft_var_load_item_variation_store: bad store format %d\n",
                  format ));
      error = FT_THROW( Invalid_Table );
      goto Exit;
    }

    /* read top level fields */
    if ( FT_READ_ULONG( region_offset )         ||
         FT_READ_USHORT( itemStore->dataCount ) )
      goto Exit;

    /* we need at least one entry in `itemStore->varData' */
    if ( !itemStore->dataCount )
    {
      FT_TRACE2(( "ft_var_load_item_variation_store: missing varData\n" ));
      error = FT_THROW( Invalid_Table );
      goto Exit;
    }

    /* make temporary copy of item variation data offsets; */
    /* we will parse region list first, then come back     */
    if ( FT_NEW_ARRAY( dataOffsetArray, itemStore->dataCount ) )
      goto Exit;

    for ( i = 0; i < itemStore->dataCount; i++ )
    {
      if ( FT_READ_ULONG( dataOffsetArray[i] ) )
        goto Exit;
    }

    /* parse array of region records (region list) */
    if ( FT_STREAM_SEEK( offset + region_offset ) )
      goto Exit;

    if ( FT_READ_USHORT( itemStore->axisCount )   ||
         FT_READ_USHORT( itemStore->regionCount ) )
      goto Exit;

    if ( itemStore->axisCount != (FT_Long)blend->mmvar->num_axis )
    {
      FT_TRACE2(( "ft_var_load_item_variation_store:"
                  " number of axes in item variation store\n"
                  "                                 "
                  " and `fvar' table are different\n" ));
      error = FT_THROW( Invalid_Table );
      goto Exit;
    }

    if ( FT_NEW_ARRAY( itemStore->varRegionList, itemStore->regionCount ) )
      goto Exit;

    for ( i = 0; i < itemStore->regionCount; i++ )
    {
      GX_AxisCoords  axisCoords;


      if ( FT_NEW_ARRAY( itemStore->varRegionList[i].axisList,
                         itemStore->axisCount ) )
        goto Exit;

      axisCoords = itemStore->varRegionList[i].axisList;

      for ( j = 0; j < itemStore->axisCount; j++ )
      {
        FT_Short  start, peak, end;


        if ( FT_READ_SHORT( start ) ||
             FT_READ_SHORT( peak )  ||
             FT_READ_SHORT( end )   )
          goto Exit;

        axisCoords[j].startCoord = FT_fdot14ToFixed( start );
        axisCoords[j].peakCoord  = FT_fdot14ToFixed( peak );
        axisCoords[j].endCoord   = FT_fdot14ToFixed( end );
      }
    }

    /* end of region list parse */

    /* use dataOffsetArray now to parse varData items */
    if ( FT_NEW_ARRAY( itemStore->varData, itemStore->dataCount ) )
      goto Exit;

    for ( i = 0; i < itemStore->dataCount; i++ )
    {
      varData = &itemStore->varData[i];

      if ( FT_STREAM_SEEK( offset + dataOffsetArray[i] ) )
        goto Exit;

      if ( FT_READ_USHORT( varData->itemCount )      ||
           FT_READ_USHORT( shortDeltaCount )         ||
           FT_READ_USHORT( varData->regionIdxCount ) )
        goto Exit;

      /* check some data consistency */
      if ( shortDeltaCount > varData->regionIdxCount )
      {
        FT_TRACE2(( "bad short count %d or region count %d\n",
                    shortDeltaCount,
                    varData->regionIdxCount ));
        error = FT_THROW( Invalid_Table );
        goto Exit;
      }

      if ( varData->regionIdxCount > itemStore->regionCount )
      {
        FT_TRACE2(( "inconsistent regionCount %d in varData[%d]\n",
                    varData->regionIdxCount,
                    i ));
        error = FT_THROW( Invalid_Table );
        goto Exit;
      }

      /* parse region indices */
      if ( FT_NEW_ARRAY( varData->regionIndices,
                         varData->regionIdxCount ) )
        goto Exit;

      for ( j = 0; j < varData->regionIdxCount; j++ )
      {
        if ( FT_READ_USHORT( varData->regionIndices[j] ) )
          goto Exit;

        if ( varData->regionIndices[j] >= itemStore->regionCount )
        {
          FT_TRACE2(( "bad region index %d\n",
                      varData->regionIndices[j] ));
          error = FT_THROW( Invalid_Table );
          goto Exit;
        }
      }

      /* Parse delta set.                                                */
      /*                                                                 */
      /* On input, deltas are (shortDeltaCount + regionIdxCount) bytes   */
      /* each; on output, deltas are expanded to `regionIdxCount' shorts */
      /* each.                                                           */
      if ( FT_NEW_ARRAY( varData->deltaSet,
                         varData->regionIdxCount * varData->itemCount ) )
        goto Exit;

      /* the delta set is stored as a 2-dimensional array of shorts; */
      /* sign-extend signed bytes to signed shorts                   */
      for ( j = 0; j < varData->itemCount * varData->regionIdxCount; )
      {
        for ( k = 0; k < shortDeltaCount; k++, j++ )
        {
          /* read the short deltas */
          FT_Short  delta;


          if ( FT_READ_SHORT( delta ) )
            goto Exit;

          varData->deltaSet[j] = delta;
        }

        for ( ; k < varData->regionIdxCount; k++, j++ )
        {
          /* read the (signed) byte deltas */
          FT_Char  delta;


          if ( FT_READ_CHAR( delta ) )
            goto Exit;

          varData->deltaSet[j] = delta;
        }
      }
    }

  Exit:
    FT_FREE( dataOffsetArray );

    return error;
  }


  static FT_Error
  ft_var_load_delta_set_index_mapping( TT_Face            face,
                                       FT_ULong           offset,
                                       GX_DeltaSetIdxMap  map,
                                       GX_ItemVarStore    itemStore )
  {
    FT_Stream  stream = FT_FACE_STREAM( face );
    FT_Memory  memory = stream->memory;

    FT_Error   error;

    FT_UShort  format;
    FT_UInt    entrySize;
    FT_UInt    innerBitCount;
    FT_UInt    innerIndexMask;
    FT_UInt    i, j;


    if ( FT_STREAM_SEEK( offset )        ||
         FT_READ_USHORT( format )        ||
         FT_READ_USHORT( map->mapCount ) )
      goto Exit;

    if ( format & 0xFFC0 )
    {
      FT_TRACE2(( "bad map format %d\n", format ));
      error = FT_THROW( Invalid_Table );
      goto Exit;
    }

    /* bytes per entry: 1, 2, 3, or 4 */
    entrySize      = ( ( format & 0x0030 ) >> 4 ) + 1;
    innerBitCount  = ( format & 0x000F ) + 1;
    innerIndexMask = ( 1 << innerBitCount ) - 1;

    if ( FT_NEW_ARRAY( map->innerIndex, map->mapCount ) )
      goto Exit;

    if ( FT_NEW_ARRAY( map->outerIndex, map->mapCount ) )
      goto Exit;

    for ( i = 0; i < map->mapCount; i++ )
    {
      FT_UInt  mapData = 0;
      FT_UInt  outerIndex, innerIndex;


      /* read map data one unsigned byte at a time, big endian */
      for ( j = 0; j < entrySize; j++ )
      {
        FT_Byte  data;


        if ( FT_READ_BYTE( data ) )
          goto Exit;

        mapData = ( mapData << 8 ) | data;
      }

      outerIndex = mapData >> innerBitCount;

      if ( outerIndex >= itemStore->dataCount )
      {
        FT_TRACE2(( "outerIndex[%d] == %d out of range\n",
                    i,
                    outerIndex ));
        error = FT_THROW( Invalid_Table );
        goto Exit;
      }

      map->outerIndex[i] = outerIndex;

      innerIndex = mapData & innerIndexMask;

      if ( innerIndex >= itemStore->varData[outerIndex].itemCount )
      {
        FT_TRACE2(( "innerIndex[%d] == %d out of range\n",
                    i,
                    innerIndex ));
        error = FT_THROW( Invalid_Table );
          goto Exit;
      }

      map->innerIndex[i] = innerIndex;
    }

  Exit:
    return error;
  }


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    ft_var_load_hvvar                                                  */
  /*                                                                       */
  /* <Description>                                                         */
  /*    If `vertical' is zero, parse the `HVAR' table and set              */
  /*    `blend->hvar_loaded' to TRUE.  On success, `blend->hvar_checked'   */
  /*    is set to TRUE.                                                    */
  /*                                                                       */
  /*    If `vertical' is not zero, parse the `VVAR' table and set          */
  /*    `blend->vvar_loaded' to TRUE.  On success, `blend->vvar_checked'   */
  /*    is set to TRUE.                                                    */
  /*                                                                       */
  /*    Some memory may remain allocated on error; it is always freed in   */
  /*    `tt_done_blend', however.                                          */
  /*                                                                       */
  /* <InOut>                                                               */
  /*    face :: The font face.                                             */
  /*                                                                       */
  /* <Return>                                                              */
  /*    FreeType error code.  0 means success.                             */
  /*                                                                       */
  static FT_Error
  ft_var_load_hvvar( TT_Face  face,
                     FT_Bool  vertical )
  {
    FT_Stream  stream = FT_FACE_STREAM( face );
    FT_Memory  memory = stream->memory;

    GX_Blend  blend = face->blend;

    GX_HVVarTable  table;

    FT_Error   error;
    FT_UShort  majorVersion;
    FT_ULong   table_len;
    FT_ULong   table_offset;
    FT_ULong   store_offset;
    FT_ULong   widthMap_offset;


    if ( vertical )
    {
      blend->vvar_loaded = TRUE;

      FT_TRACE2(( "VVAR " ));

      error = face->goto_table( face, TTAG_VVAR, stream, &table_len );
    }
    else
    {
      blend->hvar_loaded = TRUE;

      FT_TRACE2(( "HVAR " ));

      error = face->goto_table( face, TTAG_HVAR, stream, &table_len );
    }

    if ( error )
    {
      FT_TRACE2(( "is missing\n" ));
      goto Exit;
    }

    table_offset = FT_STREAM_POS();

    /* skip minor version */
    if ( FT_READ_USHORT( majorVersion ) ||
         FT_STREAM_SKIP( 2 )            )
      goto Exit;

    if ( majorVersion != 1 )
    {
      FT_TRACE2(( "bad table version %d\n", majorVersion ));
      error = FT_THROW( Invalid_Table );
      goto Exit;
    }

    if ( FT_READ_ULONG( store_offset )    ||
         FT_READ_ULONG( widthMap_offset ) )
      goto Exit;

    if ( vertical )
    {
      if ( FT_NEW( blend->vvar_table ) )
        goto Exit;
      table = blend->vvar_table;
    }
    else
    {
      if ( FT_NEW( blend->hvar_table ) )
        goto Exit;
      table = blend->hvar_table;
    }

    error = ft_var_load_item_variation_store(
              face,
              table_offset + store_offset,
              &table->itemStore );
    if ( error )
      goto Exit;

    if ( widthMap_offset )
    {
      error = ft_var_load_delta_set_index_mapping(
                face,
                table_offset + widthMap_offset,
                &table->widthMap,
                &table->itemStore );
      if ( error )
        goto Exit;
    }

    FT_TRACE2(( "loaded\n" ));
    error = FT_Err_Ok;

  Exit:
    if ( !error )
    {
      if ( vertical )
      {
        blend->vvar_checked = TRUE;

        /* FreeType doesn't provide functions to quickly retrieve    */
        /* TSB, BSB, or VORG values; we thus don't have to implement */
        /* support for those three item variation stores.            */

        face->variation_support |= TT_FACE_FLAG_VAR_VADVANCE;
      }
      else
      {
        blend->hvar_checked = TRUE;

        /* FreeType doesn't provide functions to quickly retrieve */
        /* LSB or RSB values; we thus don't have to implement     */
        /* support for those two item variation stores.           */

        face->variation_support |= TT_FACE_FLAG_VAR_HADVANCE;
      }
    }

    return error;
  }


  static FT_Int
  ft_var_get_item_delta( TT_Face          face,
                         GX_ItemVarStore  itemStore,
                         FT_UInt          outerIndex,
                         FT_UInt          innerIndex )
  {
    GX_ItemVarData  varData;
    FT_Short*       deltaSet;

    FT_UInt   master, j;
    FT_Fixed  netAdjustment = 0;     /* accumulated adjustment */
    FT_Fixed  scaledDelta;
    FT_Fixed  delta;


    /* See pseudo code from `Font Variations Overview' */
    /* in the OpenType specification.                  */

    varData  = &itemStore->varData[outerIndex];
    deltaSet = &varData->deltaSet[varData->regionIdxCount * innerIndex];

    /* outer loop steps through master designs to be blended */
    for ( master = 0; master < varData->regionIdxCount; master++ )
    {
      FT_Fixed  scalar      = FT_FIXED_ONE;
      FT_UInt   regionIndex = varData->regionIndices[master];

      GX_AxisCoords  axis = itemStore->varRegionList[regionIndex].axisList;


      /* inner loop steps through axes in this region */
      for ( j = 0; j < itemStore->axisCount; j++, axis++ )
      {
        FT_Fixed  axisScalar;


        /* compute the scalar contribution of this axis; */
        /* ignore invalid ranges                         */
        if ( axis->startCoord > axis->peakCoord ||
             axis->peakCoord > axis->endCoord   )
          axisScalar = FT_FIXED_ONE;

        else if ( axis->startCoord < 0 &&
                  axis->endCoord > 0   &&
                  axis->peakCoord != 0 )
          axisScalar = FT_FIXED_ONE;

        /* peak of 0 means ignore this axis */
        else if ( axis->peakCoord == 0 )
          axisScalar = FT_FIXED_ONE;

        /* ignore this region if coords are out of range */
        else if ( face->blend->normalizedcoords[j] < axis->startCoord ||
                  face->blend->normalizedcoords[j] > axis->endCoord   )
          axisScalar = 0;

        /* calculate a proportional factor */
        else
        {
          if ( face->blend->normalizedcoords[j] == axis->peakCoord )
            axisScalar = FT_FIXED_ONE;
          else if ( face->blend->normalizedcoords[j] < axis->peakCoord )
            axisScalar =
              FT_DivFix( face->blend->normalizedcoords[j] - axis->startCoord,
                         axis->peakCoord - axis->startCoord );
          else
            axisScalar =
              FT_DivFix( axis->endCoord - face->blend->normalizedcoords[j],
                         axis->endCoord - axis->peakCoord );
        }

        /* take product of all the axis scalars */
        scalar = FT_MulFix( scalar, axisScalar );

      } /* per-axis loop */

      /* get the scaled delta for this region */
      delta       = FT_intToFixed( deltaSet[master] );
      scaledDelta = FT_MulFix( scalar, delta );

      /* accumulate the adjustments from each region */
      netAdjustment = netAdjustment + scaledDelta;

    } /* per-region loop */

    return FT_fixedToInt( netAdjustment );
  }


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    tt_hvadvance_adjust                                                */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Apply `HVAR' advance width or `VVAR' advance height adjustment of  */
  /*    a given glyph.                                                     */
  /*                                                                       */
  /* <Input>                                                               */
  /*    gindex   :: The glyph index.                                       */
  /*                                                                       */
  /*    vertical :: If set, handle `VVAR' table.                           */
  /*                                                                       */
  /* <InOut>                                                               */
  /*    face     :: The font face.                                         */
  /*                                                                       */
  /*    adelta   :: Points to width or height value that gets modified.    */
  /*                                                                       */
  static FT_Error
  tt_hvadvance_adjust( TT_Face  face,
                       FT_UInt  gindex,
                       FT_Int  *avalue,
                       FT_Bool  vertical )
  {
    FT_Error  error = FT_Err_Ok;
    FT_UInt   innerIndex, outerIndex;
    FT_Int    delta;

    GX_HVVarTable  table;


    if ( !face->doblend || !face->blend )
      goto Exit;

    if ( vertical )
    {
      if ( !face->blend->vvar_loaded )
      {
        /* initialize vvar table */
        face->blend->vvar_error = ft_var_load_hvvar( face, 1 );
      }

      if ( !face->blend->vvar_checked )
      {
        error = face->blend->vvar_error;
        goto Exit;
      }

      table = face->blend->vvar_table;
    }
    else
    {
      if ( !face->blend->hvar_loaded )
      {
        /* initialize hvar table */
        face->blend->hvar_error = ft_var_load_hvvar( face, 0 );
      }

      if ( !face->blend->hvar_checked )
      {
        error = face->blend->hvar_error;
        goto Exit;
      }

      table = face->blend->hvar_table;
    }

    /* advance width or height adjustments are always present in an */
    /* `HVAR' or `VVAR' table; no need to test for this capability  */

    if ( table->widthMap.innerIndex )
    {
      FT_UInt  idx = gindex;


      if ( idx >= table->widthMap.mapCount )
        idx = table->widthMap.mapCount - 1;

      /* trust that HVAR parser has checked indices */
      outerIndex = table->widthMap.outerIndex[idx];
      innerIndex = table->widthMap.innerIndex[idx];
    }
    else
    {
      GX_ItemVarData  varData;


      /* no widthMap data */
      outerIndex = 0;
      innerIndex = gindex;

      varData = &table->itemStore.varData[outerIndex];
      if ( gindex >= varData->itemCount )
      {
        FT_TRACE2(( "gindex %d out of range\n", gindex ));
        error = FT_THROW( Invalid_Argument );
        goto Exit;
      }
    }

    delta = ft_var_get_item_delta( face,
                                   &table->itemStore,
                                   outerIndex,
                                   innerIndex );

    FT_TRACE5(( "%s value %d adjusted by %d units (%s)\n",
                vertical ? "vertical height" : "horizontal width",
                *avalue,
                delta,
                vertical ? "VVAR" : "HVAR" ));

    *avalue += delta;

  Exit:
    return error;
  }


  FT_LOCAL_DEF( FT_Error )
  tt_hadvance_adjust( TT_Face  face,
                      FT_UInt  gindex,
                      FT_Int  *avalue )
  {
    return tt_hvadvance_adjust( face, gindex, avalue, 0 );
  }


  FT_LOCAL_DEF( FT_Error )
  tt_vadvance_adjust( TT_Face  face,
                      FT_UInt  gindex,
                      FT_Int  *avalue )
  {
    return tt_hvadvance_adjust( face, gindex, avalue, 1 );
  }


#define GX_VALUE_SIZE  8

  /* all values are FT_Short or FT_UShort entities; */
  /* we treat them consistently as FT_Short         */
#define GX_VALUE_CASE( tag, dflt )      \
          case MVAR_TAG_ ## tag :       \
            p = (FT_Short*)&face->dflt; \
            break

#define GX_GASP_CASE( idx )                                       \
          case MVAR_TAG_GASP_ ## idx :                            \
            if ( idx < face->gasp.numRanges - 1 )                 \
              p = (FT_Short*)&face->gasp.gaspRanges[idx].maxPPEM; \
            else                                                  \
              p = NULL;                                           \
            break


  static FT_Short*
  ft_var_get_value_pointer( TT_Face   face,
                            FT_ULong  mvar_tag )
  {
    FT_Short*  p;


    switch ( mvar_tag )
    {
      GX_GASP_CASE( 0 );
      GX_GASP_CASE( 1 );
      GX_GASP_CASE( 2 );
      GX_GASP_CASE( 3 );
      GX_GASP_CASE( 4 );
      GX_GASP_CASE( 5 );
      GX_GASP_CASE( 6 );
      GX_GASP_CASE( 7 );
      GX_GASP_CASE( 8 );
      GX_GASP_CASE( 9 );

      GX_VALUE_CASE( CPHT, os2.sCapHeight );
      GX_VALUE_CASE( HASC, os2.sTypoAscender );
      GX_VALUE_CASE( HCLA, os2.usWinAscent );
      GX_VALUE_CASE( HCLD, os2.usWinDescent );
      GX_VALUE_CASE( HCOF, horizontal.caret_Offset );
      GX_VALUE_CASE( HCRN, horizontal.caret_Slope_Run );
      GX_VALUE_CASE( HCRS, horizontal.caret_Slope_Rise );
      GX_VALUE_CASE( HDSC, os2.sTypoDescender );
      GX_VALUE_CASE( HLGP, os2.sTypoLineGap );
      GX_VALUE_CASE( SBXO, os2.ySubscriptXOffset);
      GX_VALUE_CASE( SBXS, os2.ySubscriptXSize );
      GX_VALUE_CASE( SBYO, os2.ySubscriptYOffset );
      GX_VALUE_CASE( SBYS, os2.ySubscriptYSize );
      GX_VALUE_CASE( SPXO, os2.ySuperscriptXOffset );
      GX_VALUE_CASE( SPXS, os2.ySuperscriptXSize );
      GX_VALUE_CASE( SPYO, os2.ySuperscriptYOffset );
      GX_VALUE_CASE( SPYS, os2.ySuperscriptYSize );
      GX_VALUE_CASE( STRO, os2.yStrikeoutPosition );
      GX_VALUE_CASE( STRS, os2.yStrikeoutSize );
      GX_VALUE_CASE( UNDO, postscript.underlinePosition );
      GX_VALUE_CASE( UNDS, postscript.underlineThickness );
      GX_VALUE_CASE( VASC, vertical.Ascender );
      GX_VALUE_CASE( VCOF, vertical.caret_Offset );
      GX_VALUE_CASE( VCRN, vertical.caret_Slope_Run );
      GX_VALUE_CASE( VCRS, vertical.caret_Slope_Rise );
      GX_VALUE_CASE( VDSC, vertical.Descender );
      GX_VALUE_CASE( VLGP, vertical.Line_Gap );
      GX_VALUE_CASE( XHGT, os2.sxHeight );

    default:
      /* ignore unknown tag */
      p = NULL;
    }

    return p;
  }


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    ft_var_load_mvar                                                   */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Parse the `MVAR' table.                                            */
  /*                                                                       */
  /*    Some memory may remain allocated on error; it is always freed in   */
  /*    `tt_done_blend', however.                                          */
  /*                                                                       */
  /* <InOut>                                                               */
  /*    face :: The font face.                                             */
  /*                                                                       */
  static void
  ft_var_load_mvar( TT_Face  face )
  {
    FT_Stream  stream = FT_FACE_STREAM( face );
    FT_Memory  memory = stream->memory;

    GX_Blend         blend = face->blend;
    GX_ItemVarStore  itemStore;
    GX_Value         value, limit;

    FT_Error   error;
    FT_UShort  majorVersion;
    FT_ULong   table_len;
    FT_ULong   table_offset;
    FT_UShort  store_offset;
    FT_ULong   records_offset;


    FT_TRACE2(( "MVAR " ));

    error = face->goto_table( face, TTAG_MVAR, stream, &table_len );
    if ( error )
    {
      FT_TRACE2(( "is missing\n" ));
      return;
    }

    table_offset = FT_STREAM_POS();

    /* skip minor version */
    if ( FT_READ_USHORT( majorVersion ) ||
         FT_STREAM_SKIP( 2 )            )
      return;

    if ( majorVersion != 1 )
    {
      FT_TRACE2(( "bad table version %d\n", majorVersion ));
      return;
    }

    if ( FT_NEW( blend->mvar_table ) )
      return;

    /* skip reserved entry and value record size */
    if ( FT_STREAM_SKIP( 4 )                             ||
         FT_READ_USHORT( blend->mvar_table->valueCount ) ||
         FT_READ_USHORT( store_offset )                  )
      return;

    records_offset = FT_STREAM_POS();

    error = ft_var_load_item_variation_store(
              face,
              table_offset + store_offset,
              &blend->mvar_table->itemStore );
    if ( error )
      return;

    if ( FT_NEW_ARRAY( blend->mvar_table->values,
                       blend->mvar_table->valueCount ) )
      return;

    if ( FT_STREAM_SEEK( records_offset )                                ||
         FT_FRAME_ENTER( blend->mvar_table->valueCount * GX_VALUE_SIZE ) )
      return;

    value     = blend->mvar_table->values;
    limit     = value + blend->mvar_table->valueCount;
    itemStore = &blend->mvar_table->itemStore;

    for ( ; value < limit; value++ )
    {
      value->tag        = FT_GET_ULONG();
      value->outerIndex = FT_GET_USHORT();
      value->innerIndex = FT_GET_USHORT();

      if ( value->outerIndex >= itemStore->dataCount                  ||
           value->innerIndex >= itemStore->varData[value->outerIndex]
                                                  .itemCount          )
      {
        error = FT_THROW( Invalid_Table );
        break;
      }
    }

    FT_FRAME_EXIT();

    if ( error )
      return;

    FT_TRACE2(( "loaded\n" ));

    value = blend->mvar_table->values;
    limit = value + blend->mvar_table->valueCount;

    /* save original values of the data MVAR is going to modify */
    for ( ; value < limit; value++ )
    {
      FT_Short*  p = ft_var_get_value_pointer( face, value->tag );


      if ( p )
        value->unmodified = *p;
#ifdef FT_DEBUG_LEVEL_TRACE
      else
        FT_TRACE1(( "ft_var_load_mvar: Ignoring unknown tag `%c%c%c%c'\n",
                    (FT_Char)( value->tag >> 24 ),
                    (FT_Char)( value->tag >> 16 ),
                    (FT_Char)( value->tag >> 8 ),
                    (FT_Char)( value->tag ) ));
#endif
    }

    face->variation_support |= TT_FACE_FLAG_VAR_MVAR;
  }


  static FT_Error
  tt_size_reset_iterator( FT_ListNode  node,
                          void*        user )
  {
    TT_Size  size = (TT_Size)node->data;

    FT_UNUSED( user );


    tt_size_reset( size, 1 );

    return FT_Err_Ok;
  }


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    tt_apply_mvar                                                      */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Apply `MVAR' table adjustments.                                    */
  /*                                                                       */
  /* <InOut>                                                               */
  /*    face :: The font face.                                             */
  /*                                                                       */
  FT_LOCAL_DEF( void )
  tt_apply_mvar( TT_Face  face )
  {
    GX_Blend  blend = face->blend;
    GX_Value  value, limit;


    if ( !( face->variation_support & TT_FACE_FLAG_VAR_MVAR ) )
      return;

    value = blend->mvar_table->values;
    limit = value + blend->mvar_table->valueCount;

    for ( ; value < limit; value++ )
    {
      FT_Short*  p = ft_var_get_value_pointer( face, value->tag );
      FT_Int     delta;


      delta = ft_var_get_item_delta( face,
                                     &blend->mvar_table->itemStore,
                                     value->outerIndex,
                                     value->innerIndex );

      if ( p )
      {
        FT_TRACE5(( "value %c%c%c%c (%d units) adjusted by %d units (MVAR)\n",
                    (FT_Char)( value->tag >> 24 ),
                    (FT_Char)( value->tag >> 16 ),
                    (FT_Char)( value->tag >> 8 ),
                    (FT_Char)( value->tag ),
                    value->unmodified,
                    delta ));

        /* since we handle both signed and unsigned values as FT_Short, */
        /* ensure proper overflow arithmetic                            */
        *p = (FT_Short)( value->unmodified + (FT_Short)delta );
      }
    }

    /* adjust all derived values */
    {
      FT_Face  root = &face->root;


      if ( face->os2.version != 0xFFFFU )
      {
        if ( face->os2.sTypoAscender || face->os2.sTypoDescender )
        {
          root->ascender  = face->os2.sTypoAscender;
          root->descender = face->os2.sTypoDescender;

          root->height = root->ascender - root->descender +
                         face->os2.sTypoLineGap;
        }
        else
        {
          root->ascender  =  (FT_Short)face->os2.usWinAscent;
          root->descender = -(FT_Short)face->os2.usWinDescent;

          root->height = root->ascender - root->descender;
        }
      }

      root->underline_position  = face->postscript.underlinePosition -
                                  face->postscript.underlineThickness / 2;
      root->underline_thickness = face->postscript.underlineThickness;

      /* iterate over all FT_Size objects and call `tt_size_reset' */
      /* to propagate the metrics changes                          */
      FT_List_Iterate( &root->sizes_list,
                       tt_size_reset_iterator,
                       NULL );
    }
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
  /*    Parse the `gvar' table if present.  If `fvar' is there, `gvar' had */
  /*    better be there too.                                               */
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
    FT_Stream     stream = FT_FACE_STREAM( face );
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


    FT_TRACE2(( "GVAR " ));

    if ( FT_SET_ERROR( face->goto_table( face,
                                         TTAG_gvar,
                                         stream,
                                         &table_len ) ) )
    {
      FT_TRACE2(( "is missing\n" ));
      goto Exit;
    }

    gvar_start = FT_STREAM_POS( );
    if ( FT_STREAM_READ_FIELDS( gvar_fields, &gvar_head ) )
      goto Exit;

    if ( gvar_head.version != 0x00010000L )
    {
      FT_TRACE1(( "bad table version\n" ));
      error = FT_THROW( Invalid_Table );
      goto Exit;
    }

    if ( gvar_head.axisCount != (FT_UShort)blend->mmvar->num_axis )
    {
      FT_TRACE1(( "ft_var_load_gvar: number of axes in `gvar' and `cvar'\n"
                  "                  table are different\n" ));
      error = FT_THROW( Invalid_Table );
      goto Exit;
    }

    /* rough sanity check, ignoring offsets */
    if ( (FT_ULong)gvar_head.globalCoordCount * gvar_head.axisCount >
           table_len / 2 )
    {
      FT_TRACE1(( "ft_var_load_gvar:"
                  " invalid number of global coordinates\n" ));
      error = FT_THROW( Invalid_Table );
      goto Exit;
    }

    /* rough sanity check: offsets can be either 2 or 4 bytes */
    if ( (FT_ULong)gvar_head.glyphCount *
           ( ( gvar_head.flags & 1 ) ? 4 : 2 ) > table_len )
    {
      FT_TRACE1(( "ft_var_load_gvar: invalid number of glyphs\n" ));
      error = FT_THROW( Invalid_Table );
      goto Exit;
    }

    FT_TRACE2(( "loaded\n" ));

    blend->gvar_size   = table_len;
    blend->tuplecount  = gvar_head.globalCoordCount;
    blend->gv_glyphcnt = gvar_head.glyphCount;
    offsetToData       = gvar_start + gvar_head.offsetToData;

    FT_TRACE5(( "gvar: there are %d shared coordinates:\n",
                blend->tuplecount ));

    if ( FT_NEW_ARRAY( blend->glyphoffsets, blend->gv_glyphcnt + 1 ) )
      goto Exit;

    if ( gvar_head.flags & 1 )
    {
      /* long offsets (one more offset than glyphs, to mark size of last) */
      if ( FT_FRAME_ENTER( ( blend->gv_glyphcnt + 1 ) * 4L ) )
        goto Exit;

      for ( i = 0; i <= blend->gv_glyphcnt; i++ )
        blend->glyphoffsets[i] = offsetToData + FT_GET_ULONG();

      FT_FRAME_EXIT();
    }
    else
    {
      /* short offsets (one more offset than glyphs, to mark size of last) */
      if ( FT_FRAME_ENTER( ( blend->gv_glyphcnt + 1 ) * 2L ) )
        goto Exit;

      for ( i = 0; i <= blend->gv_glyphcnt; i++ )
        blend->glyphoffsets[i] = offsetToData + FT_GET_USHORT() * 2;
                                               /* XXX: Undocumented: `*2'! */

      FT_FRAME_EXIT();
    }

    if ( blend->tuplecount != 0 )
    {
      if ( FT_NEW_ARRAY( blend->tuplecoords,
                         gvar_head.axisCount * blend->tuplecount ) )
        goto Exit;

      if ( FT_STREAM_SEEK( gvar_start + gvar_head.offsetToCoord )         ||
           FT_FRAME_ENTER( blend->tuplecount * gvar_head.axisCount * 2L ) )
        goto Exit;

      for ( i = 0; i < blend->tuplecount; i++ )
      {
        FT_TRACE5(( "  [ " ));
        for ( j = 0; j < (FT_UInt)gvar_head.axisCount; j++ )
        {
          blend->tuplecoords[i * gvar_head.axisCount + j] =
            FT_GET_SHORT() * 4;                 /* convert to FT_Fixed */
          FT_TRACE5(( "%.5f ",
            blend->tuplecoords[i * gvar_head.axisCount + j] / 65536.0 ));
        }
        FT_TRACE5(( "]\n" ));
      }

      FT_TRACE5(( "\n" ));

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


    for ( i = 0; i < blend->num_axis; i++ )
    {
      FT_TRACE6(( "    axis coordinate %d (%.5f):\n",
                  i, blend->normalizedcoords[i] / 65536.0 ));
      if ( !( tupleIndex & GX_TI_INTERMEDIATE_TUPLE ) )
        FT_TRACE6(( "      intermediate coordinates %d (%.5f, %.5f):\n",
                    i,
                    im_start_coords[i] / 65536.0,
                    im_end_coords[i] / 65536.0 ));

      /* It's not clear why (for intermediate tuples) we don't need     */
      /* to check against start/end -- the documentation says we don't. */
      /* Similarly, it's unclear why we don't need to scale along the   */
      /* axis.                                                          */

      if ( tuple_coords[i] == 0 )
      {
        FT_TRACE6(( "      tuple coordinate is zero, ignored\n", i ));
        continue;
      }

      if ( blend->normalizedcoords[i] == 0 )
      {
        FT_TRACE6(( "      axis coordinate is zero, stop\n" ));
        apply = 0;
        break;
      }

      if ( blend->normalizedcoords[i] == tuple_coords[i] )
      {
        FT_TRACE6(( "      tuple coordinate value %.5f fits perfectly\n",
                    tuple_coords[i] / 65536.0 ));
        /* `apply' does not change */
        continue;
      }

      if ( !( tupleIndex & GX_TI_INTERMEDIATE_TUPLE ) )
      {
        /* not an intermediate tuple */

        if ( blend->normalizedcoords[i] < FT_MIN( 0, tuple_coords[i] ) ||
             blend->normalizedcoords[i] > FT_MAX( 0, tuple_coords[i] ) )
        {
          FT_TRACE6(( "      tuple coordinate value %.5f is exceeded, stop\n",
                      tuple_coords[i] / 65536.0 ));
          apply = 0;
          break;
        }

        FT_TRACE6(( "      tuple coordinate value %.5f fits\n",
                    tuple_coords[i] / 65536.0 ));
        apply = FT_MulDiv( apply,
                           blend->normalizedcoords[i],
                           tuple_coords[i] );
      }
      else
      {
        /* intermediate tuple */

        if ( blend->normalizedcoords[i] < im_start_coords[i] ||
             blend->normalizedcoords[i] > im_end_coords[i]   )
        {
          FT_TRACE6(( "      intermediate tuple range [%.5f;%.5f] is exceeded,"
                      " stop\n",
                      im_start_coords[i] / 65536.0,
                      im_end_coords[i] / 65536.0 ));
          apply = 0;
          break;
        }

        else if ( blend->normalizedcoords[i] < tuple_coords[i] )
        {
          FT_TRACE6(( "      intermediate tuple range [%.5f;%.5f] fits\n",
                      im_start_coords[i] / 65536.0,
                      im_end_coords[i] / 65536.0 ));
          apply = FT_MulDiv( apply,
                             blend->normalizedcoords[i] - im_start_coords[i],
                             tuple_coords[i] - im_start_coords[i] );
        }

        else
        {
          FT_TRACE6(( "      intermediate tuple range [%.5f;%.5f] fits\n",
                      im_start_coords[i] / 65536.0,
                      im_end_coords[i] / 65536.0 ));
          apply = FT_MulDiv( apply,
                             im_end_coords[i] - blend->normalizedcoords[i],
                             im_end_coords[i] - tuple_coords[i] );
        }
      }
    }

    FT_TRACE6(( "    apply factor is %.5f\n", apply / 65536.0 ));

    return apply;
  }


  /* convert from design coordinates to normalized coordinates */

  static void
  ft_var_to_normalized( TT_Face    face,
                        FT_UInt    num_coords,
                        FT_Fixed*  coords,
                        FT_Fixed*  normalized )
  {
    GX_Blend        blend;
    FT_MM_Var*      mmvar;
    FT_UInt         i, j;
    FT_Var_Axis*    a;
    GX_AVarSegment  av;


    blend = face->blend;
    mmvar = blend->mmvar;

    if ( num_coords > mmvar->num_axis )
    {
      FT_TRACE2(( "ft_var_to_normalized:"
                  " only using first %d of %d coordinates\n",
                  mmvar->num_axis, num_coords ));
      num_coords = mmvar->num_axis;
    }

    /* Axis normalization is a two-stage process.  First we normalize */
    /* based on the [min,def,max] values for the axis to be [-1,0,1]. */
    /* Then, if there's an `avar' table, we renormalize this range.   */

    FT_TRACE5(( "design coordinates:\n" ));

    a = mmvar->axis;
    for ( i = 0; i < num_coords; i++, a++ )
    {
      FT_Fixed  coord = coords[i];


      FT_TRACE5(( "  %.5f\n", coord / 65536.0 ));
      if ( coord > a->maximum || coord < a->minimum )
      {
        FT_TRACE1((
          "ft_var_to_normalized: design coordinate %.5f\n"
          "                      is out of range [%.5f;%.5f]; clamping\n",
          coord / 65536.0,
          a->minimum / 65536.0,
          a->maximum / 65536.0 ));

        if ( coord > a->maximum)
          coord = a->maximum;
        else
          coord = a->minimum;
      }

      if ( coord < a->def )
        normalized[i] = -FT_DivFix( coords[i] - a->def,
                                    a->minimum - a->def );
      else if ( coord > a->def )
        normalized[i] = FT_DivFix( coords[i] - a->def,
                                   a->maximum - a->def );
      else
        normalized[i] = 0;
    }

    FT_TRACE5(( "\n" ));

    for ( ; i < mmvar->num_axis; i++ )
      normalized[i] = 0;

    if ( blend->avar_segment )
    {
      FT_TRACE5(( "normalized design coordinates"
                  " before applying `avar' data:\n" ));

      av = blend->avar_segment;
      for ( i = 0; i < mmvar->num_axis; i++, av++ )
      {
        for ( j = 1; j < (FT_UInt)av->pairCount; j++ )
        {
          if ( normalized[i] < av->correspondence[j].fromCoord )
          {
            FT_TRACE5(( "  %.5f\n", normalized[i] / 65536.0 ));

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
    }
  }


  /* convert from normalized coordinates to design coordinates */

  static void
  ft_var_to_design( TT_Face    face,
                    FT_UInt    num_coords,
                    FT_Fixed*  coords,
                    FT_Fixed*  design )
  {
    GX_Blend      blend;
    FT_MM_Var*    mmvar;
    FT_Var_Axis*  a;

    FT_UInt  i, j, nc;


    blend = face->blend;

    nc = num_coords;
    if ( num_coords > blend->num_axis )
    {
      FT_TRACE2(( "ft_var_to_design:"
                  " only using first %d of %d coordinates\n",
                  blend->num_axis, num_coords ));
      nc = blend->num_axis;
    }

    if ( face->doblend )
    {
      for ( i = 0; i < nc; i++ )
        design[i] = coords[i];
    }
    else
    {
      for ( i = 0; i < nc; i++ )
        design[i] = 0;
    }

    for ( ; i < num_coords; i++ )
      design[i] = 0;

    if ( blend->avar_segment )
    {
      GX_AVarSegment  av = blend->avar_segment;


      FT_TRACE5(( "design coordinates"
                  " after removing `avar' distortion:\n" ));

      for ( i = 0; i < nc; i++, av++ )
      {
        for ( j = 1; j < (FT_UInt)av->pairCount; j++ )
        {
          if ( design[i] < av->correspondence[j].toCoord )
          {
            design[i] =
              FT_MulDiv( design[i] - av->correspondence[j - 1].toCoord,
                         av->correspondence[j].fromCoord -
                           av->correspondence[j - 1].fromCoord,
                         av->correspondence[j].toCoord -
                           av->correspondence[j - 1].toCoord ) +
              av->correspondence[j - 1].fromCoord;

            FT_TRACE5(( "  %.5f\n", design[i] / 65536.0 ));
            break;
          }
        }
      }
    }

    mmvar = blend->mmvar;
    a     = mmvar->axis;

    for ( i = 0; i < nc; i++, a++ )
    {
      if ( design[i] < 0 )
        design[i] = a->def + FT_MulFix( design[i],
                                        a->def - a->minimum );
      else if ( design[i] > 0 )
        design[i] = a->def + FT_MulFix( design[i],
                                        a->maximum - a->def );
      else
        design[i] = a->def;
    }
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
    FT_UShort  axisCount;
    FT_UShort  axisSize;
    FT_UShort  instanceCount;
    FT_UShort  instanceSize;

  } GX_FVar_Head;


  typedef struct  fvar_axis_
  {
    FT_ULong   axisTag;
    FT_Fixed   minValue;
    FT_Fixed   defaultValue;
    FT_Fixed   maxValue;
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
  /*    those data.  It also loads (and parses) the `MVAR' table, if       */
  /*    possible.                                                          */
  /*                                                                       */
  /* <InOut>                                                               */
  /*    face   :: The font face.                                           */
  /*              TT_Get_MM_Var initializes the blend structure.           */
  /*                                                                       */
  /* <Output>                                                              */
  /*    master :: The `fvar' data (must be freed by caller).  Can be NULL, */
  /*              which makes this function simply load MM support.        */
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
    FT_UInt              i, j;
    FT_MM_Var*           mmvar = NULL;
    FT_Fixed*            next_coords;
    FT_Fixed*            nsc;
    FT_String*           next_name;
    FT_Var_Axis*         a;
    FT_Fixed*            c;
    FT_Var_Named_Style*  ns;
    GX_FVar_Head         fvar_head;
    FT_Bool              usePsName;
    FT_UInt              num_instances;

    static const FT_Frame_Field  fvar_fields[] =
    {

#undef  FT_STRUCTURE
#define FT_STRUCTURE  GX_FVar_Head

      FT_FRAME_START( 16 ),
        FT_FRAME_LONG      ( version ),
        FT_FRAME_USHORT    ( offsetToData ),
        FT_FRAME_SKIP_SHORT,
        FT_FRAME_USHORT    ( axisCount ),
        FT_FRAME_USHORT    ( axisSize ),
        FT_FRAME_USHORT    ( instanceCount ),
        FT_FRAME_USHORT    ( instanceSize ),
      FT_FRAME_END
    };

    static const FT_Frame_Field  fvaraxis_fields[] =
    {

#undef  FT_STRUCTURE
#define FT_STRUCTURE  GX_FVar_Axis

      FT_FRAME_START( 20 ),
        FT_FRAME_ULONG ( axisTag ),
        FT_FRAME_LONG  ( minValue ),
        FT_FRAME_LONG  ( defaultValue ),
        FT_FRAME_LONG  ( maxValue ),
        FT_FRAME_USHORT( flags ),
        FT_FRAME_USHORT( nameID ),
      FT_FRAME_END
    };


    /* read the font data and set up the internal representation */
    /* if not already done                                       */

    if ( !face->blend )
    {
      FT_TRACE2(( "FVAR " ));

      /* both `fvar' and `gvar' must be present */
      if ( FT_SET_ERROR( face->goto_table( face, TTAG_gvar,
                                           stream, &table_len ) ) )
      {
        /* CFF2 is an alternate to gvar here */
        if ( FT_SET_ERROR( face->goto_table( face, TTAG_CFF2,
                                             stream, &table_len ) ) )
        {
          FT_TRACE1(( "\n"
                      "TT_Get_MM_Var: `gvar' or `CFF2' table is missing\n" ));
          goto Exit;
        }
      }

      if ( FT_SET_ERROR( face->goto_table( face, TTAG_fvar,
                                           stream, &table_len ) ) )
      {
        FT_TRACE1(( "is missing\n" ));
        goto Exit;
      }

      fvar_start = FT_STREAM_POS( );

      /* the validity of the `fvar' header data was already checked */
      /* in function `sfnt_init_face'                               */
      if ( FT_STREAM_READ_FIELDS( fvar_fields, &fvar_head ) )
        goto Exit;

      usePsName = FT_BOOL( fvar_head.instanceSize ==
                           6 + 4 * fvar_head.axisCount );

      FT_TRACE2(( "loaded\n" ));

      FT_TRACE5(( "number of GX style axes: %d\n", fvar_head.axisCount ));

      if ( FT_NEW( face->blend ) )
        goto Exit;

      /* `num_instances' holds the number of all named instances, */
      /* including the default instance which might be missing    */
      /* in fvar's table of named instances                       */
      num_instances = face->root.style_flags >> 16;

      /* cannot overflow 32-bit arithmetic because of the size limits */
      /* used in the `fvar' table validity check in `sfnt_init_face'  */
      face->blend->mmvar_len =
        sizeof ( FT_MM_Var ) +
        fvar_head.axisCount * sizeof ( FT_Var_Axis ) +
        num_instances * sizeof ( FT_Var_Named_Style ) +
        num_instances * fvar_head.axisCount * sizeof ( FT_Fixed ) +
        5 * fvar_head.axisCount;

      if ( FT_ALLOC( mmvar, face->blend->mmvar_len ) )
        goto Exit;
      face->blend->mmvar = mmvar;

      /* set up pointers and offsets into the `mmvar' array; */
      /* the data gets filled in later on                    */

      mmvar->num_axis =
        fvar_head.axisCount;
      mmvar->num_designs =
        ~0U;                   /* meaningless in this context; each glyph */
                               /* may have a different number of designs  */
                               /* (or tuples, as called by Apple)         */
      mmvar->num_namedstyles =
        num_instances;
      mmvar->axis =
        (FT_Var_Axis*)&( mmvar[1] );
      mmvar->namedstyle =
        (FT_Var_Named_Style*)&( mmvar->axis[fvar_head.axisCount] );

      next_coords =
        (FT_Fixed*)&( mmvar->namedstyle[num_instances] );
      for ( i = 0; i < num_instances; i++ )
      {
        mmvar->namedstyle[i].coords  = next_coords;
        next_coords                 += fvar_head.axisCount;
      }

      next_name = (FT_String*)next_coords;
      for ( i = 0; i < fvar_head.axisCount; i++ )
      {
        mmvar->axis[i].name  = next_name;
        next_name           += 5;
      }

      /* now fill in the data */

      if ( FT_STREAM_SEEK( fvar_start + fvar_head.offsetToData ) )
        goto Exit;

      a = mmvar->axis;
      for ( i = 0; i < fvar_head.axisCount; i++ )
      {
        GX_FVar_Axis  axis_rec;


        if ( FT_STREAM_READ_FIELDS( fvaraxis_fields, &axis_rec ) )
          goto Exit;
        a->tag     = axis_rec.axisTag;
        a->minimum = axis_rec.minValue;
        a->def     = axis_rec.defaultValue;
        a->maximum = axis_rec.maxValue;
        a->strid   = axis_rec.nameID;

        a->name[0] = (FT_String)(   a->tag >> 24 );
        a->name[1] = (FT_String)( ( a->tag >> 16 ) & 0xFF );
        a->name[2] = (FT_String)( ( a->tag >>  8 ) & 0xFF );
        a->name[3] = (FT_String)( ( a->tag       ) & 0xFF );
        a->name[4] = '\0';

        if ( a->minimum > a->def ||
             a->def > a->maximum )
        {
          FT_TRACE2(( "TT_Get_MM_Var:"
                      " invalid \"%s\" axis record; disabling\n",
                      a->name ));

          a->minimum = a->def;
          a->maximum = a->def;
        }

        FT_TRACE5(( "  \"%s\": minimum=%.5f, default=%.5f, maximum=%.5f\n",
                    a->name,
                    a->minimum / 65536.0,
                    a->def / 65536.0,
                    a->maximum / 65536.0 ));

        a++;
      }

      FT_TRACE5(( "\n" ));

      /* named instance coordinates are stored as design coordinates; */
      /* we have to convert them to normalized coordinates also       */
      if ( FT_NEW_ARRAY( face->blend->normalized_stylecoords,
                         fvar_head.axisCount * num_instances ) )
        goto Exit;

      if ( fvar_head.instanceCount && !face->blend->avar_loaded )
        ft_var_load_avar( face );

      ns  = mmvar->namedstyle;
      nsc = face->blend->normalized_stylecoords;
      for ( i = 0; i < fvar_head.instanceCount; i++, ns++ )
      {
        /* PostScript names add 2 bytes to the instance record size */
        if ( FT_FRAME_ENTER( ( usePsName ? 6L : 4L ) +
                             4L * fvar_head.axisCount ) )
          goto Exit;

        ns->strid       =    FT_GET_USHORT();
        (void) /* flags = */ FT_GET_USHORT();

        c = ns->coords;
        for ( j = 0; j < fvar_head.axisCount; j++, c++ )
          *c = FT_GET_LONG();

        if ( usePsName )
          ns->psid = FT_GET_USHORT();

        ft_var_to_normalized( face,
                              fvar_head.axisCount,
                              ns->coords,
                              nsc );
        nsc += fvar_head.axisCount;

        FT_FRAME_EXIT();
      }

      if ( num_instances != fvar_head.instanceCount )
      {
        SFNT_Service  sfnt = (SFNT_Service)face->sfnt;

        FT_Int   found, dummy1, dummy2;
        FT_UInt  strid = 0xFFFFFFFFUL;


        /* the default instance is missing in array the   */
        /* of named instances; try to synthesize an entry */
        found = sfnt->get_name_id( face,
                                   TT_NAME_ID_TYPOGRAPHIC_SUBFAMILY,
                                   &dummy1,
                                   &dummy2 );
        if ( found )
          strid = TT_NAME_ID_TYPOGRAPHIC_SUBFAMILY;
        else
        {
          found = sfnt->get_name_id( face,
                                     TT_NAME_ID_FONT_SUBFAMILY,
                                     &dummy1,
                                     &dummy2 );
          if ( found )
            strid = TT_NAME_ID_FONT_SUBFAMILY;
        }

        if ( found )
        {
          found = sfnt->get_name_id( face,
                                     TT_NAME_ID_PS_NAME,
                                     &dummy1,
                                     &dummy2 );
          if ( found )
          {
            FT_TRACE5(( "TT_Get_MM_Var:"
                        " Adding default instance to named instances\n" ));

            ns = &mmvar->namedstyle[fvar_head.instanceCount];

            ns->strid = strid;
            ns->psid  = TT_NAME_ID_PS_NAME;

            a = mmvar->axis;
            c = ns->coords;
            for ( j = 0; j < fvar_head.axisCount; j++, a++, c++ )
              *c = a->def;
          }
        }
      }

      ft_var_load_mvar( face );
    }

    /* fill the output array if requested */

    if ( master )
    {
      FT_UInt  n;


      if ( FT_ALLOC( mmvar, face->blend->mmvar_len ) )
        goto Exit;
      FT_MEM_COPY( mmvar, face->blend->mmvar, face->blend->mmvar_len );

      mmvar->axis =
        (FT_Var_Axis*)&( mmvar[1] );
      mmvar->namedstyle =
        (FT_Var_Named_Style*)&( mmvar->axis[mmvar->num_axis] );
      next_coords =
        (FT_Fixed*)&( mmvar->namedstyle[mmvar->num_namedstyles] );

      for ( n = 0; n < mmvar->num_namedstyles; n++ )
      {
        mmvar->namedstyle[n].coords  = next_coords;
        next_coords                 += mmvar->num_axis;
      }

      a         = mmvar->axis;
      next_name = (FT_String*)next_coords;
      for ( n = 0; n < mmvar->num_axis; n++ )
      {
        a->name = next_name;

        /* standard PostScript names for some standard apple tags */
        if ( a->tag == TTAG_wght )
          a->name = (char*)"Weight";
        else if ( a->tag == TTAG_wdth )
          a->name = (char*)"Width";
        else if ( a->tag == TTAG_opsz )
          a->name = (char*)"OpticalSize";
        else if ( a->tag == TTAG_slnt )
          a->name = (char*)"Slant";

        next_name += 5;
        a++;
      }

      *master = mmvar;
    }

  Exit:
    return error;
  }


  static FT_Error
  tt_set_mm_blend( TT_Face    face,
                   FT_UInt    num_coords,
                   FT_Fixed*  coords,
                   FT_Bool    set_design_coords )
  {
    FT_Error    error = FT_Err_Ok;
    GX_Blend    blend;
    FT_MM_Var*  mmvar;
    FT_UInt     i, j;
    FT_Bool     is_default_instance = 1;
    FT_Memory   memory = face->root.memory;

    enum
    {
      mcvt_retain,
      mcvt_modify,
      mcvt_load

    } manageCvt;


    face->doblend = FALSE;

    if ( !face->blend )
    {
      if ( FT_SET_ERROR( TT_Get_MM_Var( face, NULL ) ) )
        goto Exit;
    }

    blend = face->blend;
    mmvar = blend->mmvar;

    if ( num_coords > mmvar->num_axis )
    {
      FT_TRACE2(( "TT_Set_MM_Blend:"
                  " only using first %d of %d coordinates\n",
                  mmvar->num_axis, num_coords ));
      num_coords = mmvar->num_axis;
    }

    FT_TRACE5(( "normalized design coordinates:\n" ));

    for ( i = 0; i < num_coords; i++ )
    {
      FT_TRACE5(( "  %.5f\n", coords[i] / 65536.0 ));
      if ( coords[i] < -0x00010000L || coords[i] > 0x00010000L )
      {
        FT_TRACE1(( "TT_Set_MM_Blend: normalized design coordinate %.5f\n"
                    "                 is out of range [-1;1]\n",
                    coords[i] / 65536.0 ));
        error = FT_THROW( Invalid_Argument );
        goto Exit;
      }

      if ( coords[i] != 0 )
        is_default_instance = 0;
    }

    FT_TRACE5(( "\n" ));

    if ( !face->is_cff2 && !blend->glyphoffsets )
      if ( FT_SET_ERROR( ft_var_load_gvar( face ) ) )
        goto Exit;

    if ( !blend->coords )
    {
      if ( FT_NEW_ARRAY( blend->coords, mmvar->num_axis ) )
        goto Exit;
    }

    if ( !blend->normalizedcoords )
    {
      if ( FT_NEW_ARRAY( blend->normalizedcoords, mmvar->num_axis ) )
        goto Exit;

      manageCvt = mcvt_modify;

      /* If we have not set the blend coordinates before this, then the  */
      /* cvt table will still be what we read from the `cvt ' table and  */
      /* we don't need to reload it.  We may need to change it though... */
    }
    else
    {
      manageCvt = mcvt_retain;

      for ( i = 0; i < num_coords; i++ )
      {
        if ( blend->normalizedcoords[i] != coords[i] )
        {
          manageCvt = mcvt_load;
          break;
        }
      }

      for ( ; i < mmvar->num_axis; i++ )
      {
        if ( blend->normalizedcoords[i] != 0 )
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

    blend->num_axis = mmvar->num_axis;
    FT_MEM_COPY( blend->normalizedcoords,
                 coords,
                 num_coords * sizeof ( FT_Fixed ) );

    if ( set_design_coords )
      ft_var_to_design( face,
                        num_coords,
                        blend->normalizedcoords,
                        blend->coords );

    face->doblend = TRUE;

    if ( face->cvt )
    {
      switch ( manageCvt )
      {
      case mcvt_load:
        /* The cvt table has been loaded already; every time we change the */
        /* blend we may need to reload and remodify the cvt table.         */
        FT_FREE( face->cvt );
        face->cvt = NULL;

        error = tt_face_load_cvt( face, face->root.stream );
        break;

      case mcvt_modify:
        /* The original cvt table is in memory.  All we need to do is */
        /* apply the `cvar' table (if any).                           */
        error = tt_face_vary_cvt( face, face->root.stream );
        break;

      case mcvt_retain:
        /* The cvt table is correct for this set of coordinates. */
        break;
      }
    }

    /* check whether the current variation tuple coincides */
    /* with a named instance                               */

    for ( i = 0; i < blend->mmvar->num_namedstyles; i++ )
    {
      FT_Fixed*  nsc = blend->normalized_stylecoords + i * blend->num_axis;
      FT_Fixed*  ns  = blend->normalizedcoords;


      for ( j = 0; j < blend->num_axis; j++, nsc++, ns++ )
      {
        if ( *nsc != *ns )
          break;
      }

      if ( j == blend->num_axis )
        break;
    }

    /* adjust named instance index */
    face->root.face_index &= 0xFFFF;
    if ( i < blend->mmvar->num_namedstyles )
      face->root.face_index |= ( i + 1 ) << 16;

    face->is_default_instance = is_default_instance;

    /* enforce recomputation of the PostScript name; */
    FT_FREE( face->postscript_name );
    face->postscript_name = NULL;

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
  /*    num_coords :: The number of available coordinates.  If it is       */
  /*                  larger than the number of axes, ignore the excess    */
  /*                  values.  If it is smaller than the number of axes,   */
  /*                  use the default value (0) for the remaining axes.    */
  /*                                                                       */
  /*    coords     :: An array of `num_coords', each between [-1,1].       */
  /*                                                                       */
  /* <Return>                                                              */
  /*    FreeType error code.  0 means success.                             */
  /*                                                                       */
  FT_LOCAL_DEF( FT_Error )
  TT_Set_MM_Blend( TT_Face    face,
                   FT_UInt    num_coords,
                   FT_Fixed*  coords )
  {
    return tt_set_mm_blend( face, num_coords, coords, 1 );
  }


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    TT_Get_MM_Blend                                                    */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Get the blend (normalized) coordinates for this instance of the    */
  /*    font.                                                              */
  /*                                                                       */
  /* <InOut>                                                               */
  /*    face       :: The font.                                            */
  /*                  Initialize the blend structure with `gvar' data.     */
  /*                                                                       */
  /* <Input>                                                               */
  /*    num_coords :: The number of available coordinates.  If it is       */
  /*                  larger than the number of axes, set the excess       */
  /*                  values to 0.                                         */
  /*                                                                       */
  /*    coords     :: An array of `num_coords', each between [-1,1].       */
  /*                                                                       */
  /* <Return>                                                              */
  /*    FreeType error code.  0 means success.                             */
  /*                                                                       */
  FT_LOCAL_DEF( FT_Error )
  TT_Get_MM_Blend( TT_Face    face,
                   FT_UInt    num_coords,
                   FT_Fixed*  coords )
  {
    FT_Error  error = FT_Err_Ok;
    GX_Blend  blend;
    FT_UInt   i, nc;


    if ( !face->blend )
    {
      if ( FT_SET_ERROR( TT_Get_MM_Var( face, NULL ) ) )
        return error;
    }

    blend = face->blend;

    nc = num_coords;
    if ( num_coords > blend->num_axis )
    {
      FT_TRACE2(( "TT_Get_MM_Blend:"
                  " only using first %d of %d coordinates\n",
                  blend->num_axis, num_coords ));
      nc = blend->num_axis;
    }

    if ( face->doblend )
    {
      for ( i = 0; i < nc; i++ )
        coords[i] = blend->normalizedcoords[i];
    }
    else
    {
      for ( i = 0; i < nc; i++ )
        coords[i] = 0;
    }

    for ( ; i < num_coords; i++ )
      coords[i] = 0;

    return FT_Err_Ok;
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
  /*    num_coords :: The number of available coordinates.  If it is       */
  /*                  larger than the number of axes, ignore the excess    */
  /*                  values.  If it is smaller than the number of axes,   */
  /*                  use the default values for the remaining axes.       */
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
    FT_Error    error  = FT_Err_Ok;
    GX_Blend    blend;
    FT_MM_Var*  mmvar;
    FT_UInt     i;
    FT_Memory   memory = face->root.memory;

    FT_Var_Axis*  a;
    FT_Fixed*     c;

    FT_Fixed*  normalized = NULL;


    if ( !face->blend )
    {
      if ( FT_SET_ERROR( TT_Get_MM_Var( face, NULL ) ) )
        goto Exit;
    }

    blend = face->blend;
    mmvar = blend->mmvar;

    if ( num_coords > mmvar->num_axis )
    {
      FT_TRACE2(( "TT_Set_Var_Design:"
                  " only using first %d of %d coordinates\n",
                  mmvar->num_axis, num_coords ));
      num_coords = mmvar->num_axis;
    }

    if ( !blend->coords )
    {
      if ( FT_NEW_ARRAY( blend->coords, mmvar->num_axis ) )
        goto Exit;
    }

    FT_MEM_COPY( blend->coords,
                 coords,
                 num_coords * sizeof ( FT_Fixed ) );

    a = mmvar->axis + num_coords;
    c = coords + num_coords;
    for ( i = num_coords; i < mmvar->num_axis; i++, a++, c++ )
      *c = a->def;

    if ( FT_NEW_ARRAY( normalized, mmvar->num_axis ) )
      goto Exit;

    if ( !face->blend->avar_loaded )
      ft_var_load_avar( face );

    ft_var_to_normalized( face, num_coords, coords, normalized );

    error = tt_set_mm_blend( face, mmvar->num_axis, normalized, 0 );

  Exit:
    FT_FREE( normalized );
    return error;
  }


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    TT_Get_Var_Design                                                  */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Get the design coordinates of the currently selected interpolated  */
  /*    font.                                                              */
  /*                                                                       */
  /* <Input>                                                               */
  /*    face       :: A handle to the source face.                         */
  /*                                                                       */
  /*    num_coords :: The number of design coordinates to retrieve.  If it */
  /*                  is larger than the number of axes, set the excess    */
  /*                  values to~0.                                         */
  /*                                                                       */
  /* <Output>                                                              */
  /*    coords     :: The design coordinates array.                        */
  /*                                                                       */
  /* <Return>                                                              */
  /*    FreeType error code.  0~means success.                             */
  /*                                                                       */
  FT_LOCAL_DEF( FT_Error )
  TT_Get_Var_Design( TT_Face    face,
                     FT_UInt    num_coords,
                     FT_Fixed*  coords )
  {
    FT_Error  error = FT_Err_Ok;
    GX_Blend  blend;
    FT_UInt   i, nc;


    if ( !face->blend )
    {
      if ( FT_SET_ERROR( TT_Get_MM_Var( face, NULL ) ) )
        return error;
    }

    blend = face->blend;

    nc = num_coords;
    if ( num_coords > blend->num_axis )
    {
      FT_TRACE2(( "TT_Get_Var_Design:"
                  " only using first %d of %d coordinates\n",
                  blend->num_axis, num_coords ));
      nc = blend->num_axis;
    }

    if ( face->doblend )
    {
      for ( i = 0; i < nc; i++ )
        coords[i] = blend->coords[i];
    }
    else
    {
      for ( i = 0; i < nc; i++ )
        coords[i] = 0;
    }

    for ( ; i < num_coords; i++ )
      coords[i] = 0;

    return FT_Err_Ok;
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

    if ( !blend )
    {
      FT_TRACE2(( "\n"
                  "tt_face_vary_cvt: no blend specified\n" ));
      error = FT_Err_Ok;
      goto Exit;
    }

    if ( !face->cvt )
    {
      FT_TRACE2(( "\n"
                  "tt_face_vary_cvt: no `cvt ' table\n" ));
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

    FT_TRACE2(( "loaded\n" ));

    if ( FT_NEW_ARRAY( tuple_coords, blend->num_axis )    ||
         FT_NEW_ARRAY( im_start_coords, blend->num_axis ) ||
         FT_NEW_ARRAY( im_end_coords, blend->num_axis )   )
      goto FExit;

    tupleCount   = FT_GET_USHORT();
    offsetToData = FT_GET_USHORT();

    /* rough sanity test */
    if ( offsetToData + ( tupleCount & GX_TC_TUPLE_COUNT_MASK ) * 4 >
           table_len )
    {
      FT_TRACE2(( "tt_face_vary_cvt:"
                  " invalid CVT variation array header\n" ));

      error = FT_THROW( Invalid_Table );
      goto FExit;
    }

    offsetToData += table_start;

    /* The documentation implies there are flags packed into              */
    /* `tupleCount', but John Jenkins says that shared points don't apply */
    /* to `cvar', and no other flags are defined.                         */

    FT_TRACE5(( "cvar: there are %d tuples:\n", tupleCount & 0xFFF ));

    for ( i = 0; i < ( tupleCount & 0xFFF ); i++ )
    {
      FT_UInt   tupleDataSize;
      FT_UInt   tupleIndex;
      FT_Fixed  apply;


      FT_TRACE6(( "  tuple %d:\n", i ));

      tupleDataSize = FT_GET_USHORT();
      tupleIndex    = FT_GET_USHORT();

      /* There is no provision here for a global tuple coordinate section, */
      /* so John says.  There are no tuple indices, just embedded tuples.  */

      if ( tupleIndex & GX_TI_EMBEDDED_TUPLE_COORD )
      {
        for ( j = 0; j < blend->num_axis; j++ )
          tuple_coords[j] = FT_GET_SHORT() * 4;  /* convert from        */
                                                 /* short frac to fixed */
      }
      else
      {
        /* skip this tuple; it makes no sense */

        if ( tupleIndex & GX_TI_INTERMEDIATE_TUPLE )
          for ( j = 0; j < 2 * blend->num_axis; j++ )
            (void)FT_GET_SHORT();

        offsetToData += tupleDataSize;
        continue;
      }

      if ( tupleIndex & GX_TI_INTERMEDIATE_TUPLE )
      {
        for ( j = 0; j < blend->num_axis; j++ )
          im_start_coords[j] = FT_GET_SHORT() * 4;
        for ( j = 0; j < blend->num_axis; j++ )
          im_end_coords[j] = FT_GET_SHORT() * 4;
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

      localpoints = ft_var_readpackedpoints( stream,
                                             table_len,
                                             &point_count );
      deltas      = ft_var_readpackeddeltas( stream,
                                             table_len,
                                             point_count == 0 ? face->cvt_size
                                                              : point_count );
      if ( !localpoints || !deltas )
        ; /* failure, ignore it */

      else if ( localpoints == ALL_POINTS )
      {
#ifdef FT_DEBUG_LEVEL_TRACE
        int  count = 0;
#endif


        FT_TRACE7(( "    CVT deltas:\n" ));

        /* this means that there are deltas for every entry in cvt */
        for ( j = 0; j < face->cvt_size; j++ )
        {
          FT_Long  orig_cvt = face->cvt[j];


          face->cvt[j] = (FT_Short)( orig_cvt +
                                     FT_MulFix( deltas[j], apply ) );

#ifdef FT_DEBUG_LEVEL_TRACE
          if ( orig_cvt != face->cvt[j] )
          {
            FT_TRACE7(( "      %d: %d -> %d\n",
                        j, orig_cvt, face->cvt[j] ));
            count++;
          }
#endif
        }

#ifdef FT_DEBUG_LEVEL_TRACE
        if ( !count )
          FT_TRACE7(( "      none\n" ));
#endif
      }

      else
      {
#ifdef FT_DEBUG_LEVEL_TRACE
        int  count = 0;
#endif


        FT_TRACE7(( "    CVT deltas:\n" ));

        for ( j = 0; j < point_count; j++ )
        {
          int      pindex;
          FT_Long  orig_cvt;


          pindex = localpoints[j];
          if ( (FT_ULong)pindex >= face->cvt_size )
            continue;

          orig_cvt          = face->cvt[pindex];
          face->cvt[pindex] = (FT_Short)( orig_cvt +
                                          FT_MulFix( deltas[j], apply ) );

#ifdef FT_DEBUG_LEVEL_TRACE
          if ( orig_cvt != face->cvt[pindex] )
          {
            FT_TRACE7(( "      %d: %d -> %d\n",
                        pindex, orig_cvt, face->cvt[pindex] ));
            count++;
          }
#endif
        }

#ifdef FT_DEBUG_LEVEL_TRACE
        if ( !count )
          FT_TRACE7(( "      none\n" ));
#endif
      }

      if ( localpoints != ALL_POINTS )
        FT_FREE( localpoints );
      FT_FREE( deltas );

      offsetToData += tupleDataSize;

      FT_Stream_SeekSet( stream, here );
    }

    FT_TRACE5(( "\n" ));

  FExit:
    FT_FRAME_EXIT();

  Exit:
    FT_FREE( tuple_coords );
    FT_FREE( im_start_coords );
    FT_FREE( im_end_coords );

    return error;
  }


  /* Shift the original coordinates of all points between indices `p1' */
  /* and `p2', using the same difference as given by index `ref'.      */

  /* modeled after `af_iup_shift' */

  static void
  tt_delta_shift( int         p1,
                  int         p2,
                  int         ref,
                  FT_Vector*  in_points,
                  FT_Vector*  out_points )
  {
    int        p;
    FT_Vector  delta;


    delta.x = out_points[ref].x - in_points[ref].x;
    delta.y = out_points[ref].y - in_points[ref].y;

    if ( delta.x == 0 && delta.y == 0 )
      return;

    for ( p = p1; p < ref; p++ )
    {
      out_points[p].x += delta.x;
      out_points[p].y += delta.y;
    }

    for ( p = ref + 1; p <= p2; p++ )
    {
      out_points[p].x += delta.x;
      out_points[p].y += delta.y;
    }
  }


  /* Interpolate the original coordinates of all points with indices */
  /* between `p1' and `p2', using `ref1' and `ref2' as the reference */
  /* point indices.                                                  */

  /* modeled after `af_iup_interp', `_iup_worker_interpolate', and */
  /* `Ins_IUP'                                                     */

  static void
  tt_delta_interpolate( int         p1,
                        int         p2,
                        int         ref1,
                        int         ref2,
                        FT_Vector*  in_points,
                        FT_Vector*  out_points )
  {
    int  p, i;

    FT_Pos  out, in1, in2, out1, out2, d1, d2;


    if ( p1 > p2 )
      return;

    /* handle both horizontal and vertical coordinates */
    for ( i = 0; i <= 1; i++ )
    {
      /* shift array pointers so that we can access `foo.y' as `foo.x' */
      in_points  = (FT_Vector*)( (FT_Pos*)in_points + i );
      out_points = (FT_Vector*)( (FT_Pos*)out_points + i );

      if ( in_points[ref1].x > in_points[ref2].x )
      {
        p    = ref1;
        ref1 = ref2;
        ref2 = p;
      }

      in1  = in_points[ref1].x;
      in2  = in_points[ref2].x;
      out1 = out_points[ref1].x;
      out2 = out_points[ref2].x;
      d1   = out1 - in1;
      d2   = out2 - in2;

      /* If the reference points have the same coordinate but different */
      /* delta, inferred delta is zero.  Otherwise interpolate.         */
      if ( in1 != in2 || out1 == out2 )
      {
        FT_Fixed  scale = in1 != in2 ? FT_DivFix( out2 - out1, in2 - in1 )
                                     : 0;


        for ( p = p1; p <= p2; p++ )
        {
          out = in_points[p].x;

          if ( out <= in1 )
            out += d1;
          else if ( out >= in2 )
            out += d2;
          else
            out = out1 + FT_MulFix( out - in1, scale );

          out_points[p].x = out;
        }
      }
    }
  }


  /* Interpolate points without delta values, similar to */
  /* the `IUP' hinting instruction.                      */

  /* modeled after `Ins_IUP */

  static void
  tt_interpolate_deltas( FT_Outline*  outline,
                         FT_Vector*   out_points,
                         FT_Vector*   in_points,
                         FT_Bool*     has_delta )
  {
    FT_Int  first_point;
    FT_Int  end_point;

    FT_Int  first_delta;
    FT_Int  cur_delta;

    FT_Int    point;
    FT_Short  contour;


    /* ignore empty outlines */
    if ( !outline->n_contours )
      return;

    contour = 0;
    point   = 0;

    do
    {
      end_point   = outline->contours[contour];
      first_point = point;

      /* search first point that has a delta */
      while ( point <= end_point && !has_delta[point] )
        point++;

      if ( point <= end_point )
      {
        first_delta = point;
        cur_delta   = point;

        point++;

        while ( point <= end_point )
        {
          /* search next point that has a delta  */
          /* and interpolate intermediate points */
          if ( has_delta[point] )
          {
            tt_delta_interpolate( cur_delta + 1,
                                  point - 1,
                                  cur_delta,
                                  point,
                                  in_points,
                                  out_points );
            cur_delta = point;
          }

          point++;
        }

        /* shift contour if we only have a single delta */
        if ( cur_delta == first_delta )
          tt_delta_shift( first_point,
                          end_point,
                          cur_delta,
                          in_points,
                          out_points );
        else
        {
          /* otherwise handle remaining points       */
          /* at the end and beginning of the contour */
          tt_delta_interpolate( cur_delta + 1,
                                end_point,
                                cur_delta,
                                first_delta,
                                in_points,
                                out_points );

          if ( first_delta > 0 )
            tt_delta_interpolate( first_point,
                                  first_delta - 1,
                                  cur_delta,
                                  first_delta,
                                  in_points,
                                  out_points );
        }
      }
      contour++;

    } while ( contour < outline->n_contours );
  }


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    TT_Vary_Apply_Glyph_Deltas                                         */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Apply the appropriate deltas to the current glyph.                 */
  /*                                                                       */
  /* <Input>                                                               */
  /*    face        :: A handle to the target face object.                 */
  /*                                                                       */
  /*    glyph_index :: The index of the glyph being modified.              */
  /*                                                                       */
  /*    n_points    :: The number of the points in the glyph, including    */
  /*                   phantom points.                                     */
  /*                                                                       */
  /* <InOut>                                                               */
  /*    outline     :: The outline to change.                              */
  /*                                                                       */
  /* <Return>                                                              */
  /*    FreeType error code.  0 means success.                             */
  /*                                                                       */
  FT_LOCAL_DEF( FT_Error )
  TT_Vary_Apply_Glyph_Deltas( TT_Face      face,
                              FT_UInt      glyph_index,
                              FT_Outline*  outline,
                              FT_UInt      n_points )
  {
    FT_Stream   stream = face->root.stream;
    FT_Memory   memory = stream->memory;
    GX_Blend    blend  = face->blend;

    FT_Vector*  points_org = NULL;
    FT_Vector*  points_out = NULL;
    FT_Bool*    has_delta  = NULL;

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


    if ( !face->doblend || !blend )
      return FT_THROW( Invalid_Argument );

    if ( glyph_index >= blend->gv_glyphcnt      ||
         blend->glyphoffsets[glyph_index] ==
           blend->glyphoffsets[glyph_index + 1] )
    {
      FT_TRACE2(( "TT_Vary_Apply_Glyph_Deltas:"
                  " no variation data for this glyph\n" ));
      return FT_Err_Ok;
    }

    if ( FT_NEW_ARRAY( points_org, n_points ) ||
         FT_NEW_ARRAY( points_out, n_points ) ||
         FT_NEW_ARRAY( has_delta, n_points )  )
      goto Fail1;

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
    offsetToData = FT_GET_USHORT();

    /* rough sanity test */
    if ( offsetToData + ( tupleCount & GX_TC_TUPLE_COUNT_MASK ) * 4 >
           blend->gvar_size )
    {
      FT_TRACE2(( "TT_Vary_Apply_Glyph_Deltas:"
                  " invalid glyph variation array header\n" ));

      error = FT_THROW( Invalid_Table );
      goto Fail2;
    }

    offsetToData += glyph_start;

    if ( tupleCount & GX_TC_TUPLES_SHARE_POINT_NUMBERS )
    {
      here = FT_Stream_FTell( stream );

      FT_Stream_SeekSet( stream, offsetToData );

      sharedpoints = ft_var_readpackedpoints( stream,
                                              blend->gvar_size,
                                              &spoint_count );
      offsetToData = FT_Stream_FTell( stream );

      FT_Stream_SeekSet( stream, here );
    }

    FT_TRACE5(( "gvar: there are %d tuples:\n",
                tupleCount & GX_TC_TUPLE_COUNT_MASK ));

    for ( j = 0; j < n_points; j++ )
      points_org[j] = outline->points[j];

    for ( i = 0; i < ( tupleCount & GX_TC_TUPLE_COUNT_MASK ); i++ )
    {
      FT_UInt   tupleDataSize;
      FT_UInt   tupleIndex;
      FT_Fixed  apply;


      FT_TRACE6(( "  tuple %d:\n", i ));

      tupleDataSize = FT_GET_USHORT();
      tupleIndex    = FT_GET_USHORT();

      if ( tupleIndex & GX_TI_EMBEDDED_TUPLE_COORD )
      {
        for ( j = 0; j < blend->num_axis; j++ )
          tuple_coords[j] = FT_GET_SHORT() * 4;   /* convert from        */
                                                  /* short frac to fixed */
      }
      else if ( ( tupleIndex & GX_TI_TUPLE_INDEX_MASK ) >= blend->tuplecount )
      {
        FT_TRACE2(( "TT_Vary_Apply_Glyph_Deltas:"
                    " invalid tuple index\n" ));

        error = FT_THROW( Invalid_Table );
        goto Fail2;
      }
      else
        FT_MEM_COPY(
          tuple_coords,
          &blend->tuplecoords[( tupleIndex & 0xFFF ) * blend->num_axis],
          blend->num_axis * sizeof ( FT_Fixed ) );

      if ( tupleIndex & GX_TI_INTERMEDIATE_TUPLE )
      {
        for ( j = 0; j < blend->num_axis; j++ )
          im_start_coords[j] = FT_GET_SHORT() * 4;
        for ( j = 0; j < blend->num_axis; j++ )
          im_end_coords[j] = FT_GET_SHORT() * 4;
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

      FT_Stream_SeekSet( stream, offsetToData );

      if ( tupleIndex & GX_TI_PRIVATE_POINT_NUMBERS )
      {
        localpoints = ft_var_readpackedpoints( stream,
                                               blend->gvar_size,
                                               &point_count );
        points      = localpoints;
      }
      else
      {
        points      = sharedpoints;
        point_count = spoint_count;
      }

      deltas_x = ft_var_readpackeddeltas( stream,
                                          blend->gvar_size,
                                          point_count == 0 ? n_points
                                                           : point_count );
      deltas_y = ft_var_readpackeddeltas( stream,
                                          blend->gvar_size,
                                          point_count == 0 ? n_points
                                                           : point_count );

      if ( !points || !deltas_y || !deltas_x )
        ; /* failure, ignore it */

      else if ( points == ALL_POINTS )
      {
#ifdef FT_DEBUG_LEVEL_TRACE
        int  count = 0;
#endif


        FT_TRACE7(( "    point deltas:\n" ));

        /* this means that there are deltas for every point in the glyph */
        for ( j = 0; j < n_points; j++ )
        {
          FT_Pos  delta_x = FT_MulFix( deltas_x[j], apply );
          FT_Pos  delta_y = FT_MulFix( deltas_y[j], apply );


          if ( j < n_points - 3 )
          {
            outline->points[j].x += delta_x;
            outline->points[j].y += delta_y;
          }
          else
          {
            /* To avoid double adjustment of advance width or height, */
            /* adjust phantom points only if there is no HVAR or VVAR */
            /* support, respectively.                                 */
            if ( j == ( n_points - 3 )          &&
                 !( face->variation_support   &
                    TT_FACE_FLAG_VAR_HADVANCE ) )
              outline->points[j].x += delta_x;

            else if ( j == ( n_points - 2 )        &&
                      !( face->variation_support &
                         TT_FACE_FLAG_VAR_LSB    ) )
              outline->points[j].x += delta_x;

            else if ( j == ( n_points - 1 )          &&
                      !( face->variation_support   &
                         TT_FACE_FLAG_VAR_VADVANCE ) )
              outline->points[j].y += delta_y;

            else if ( j == ( n_points - 0 )        &&
                      !( face->variation_support &
                         TT_FACE_FLAG_VAR_TSB    ) )
              outline->points[j].y += delta_y;
          }

#ifdef FT_DEBUG_LEVEL_TRACE
          if ( delta_x || delta_y )
          {
            FT_TRACE7(( "      %d: (%d, %d) -> (%d, %d)\n",
                        j,
                        outline->points[j].x - delta_x,
                        outline->points[j].y - delta_y,
                        outline->points[j].x,
                        outline->points[j].y ));
            count++;
          }
#endif
        }

#ifdef FT_DEBUG_LEVEL_TRACE
        if ( !count )
          FT_TRACE7(( "      none\n" ));
#endif
      }

      else
      {
#ifdef FT_DEBUG_LEVEL_TRACE
        int  count = 0;
#endif


        /* we have to interpolate the missing deltas similar to the */
        /* IUP bytecode instruction                                 */
        for ( j = 0; j < n_points; j++ )
        {
          has_delta[j]  = FALSE;
          points_out[j] = points_org[j];
        }

        for ( j = 0; j < point_count; j++ )
        {
          FT_UShort  idx = points[j];


          if ( idx >= n_points )
            continue;

          has_delta[idx] = TRUE;

          points_out[idx].x += FT_MulFix( deltas_x[j], apply );
          points_out[idx].y += FT_MulFix( deltas_y[j], apply );
        }

        /* no need to handle phantom points here,      */
        /* since solitary points can't be interpolated */
        tt_interpolate_deltas( outline,
                               points_out,
                               points_org,
                               has_delta );

        FT_TRACE7(( "    point deltas:\n" ));

        for ( j = 0; j < n_points; j++ )
        {
          FT_Pos  delta_x = points_out[j].x - points_org[j].x;
          FT_Pos  delta_y = points_out[j].y - points_org[j].y;


          outline->points[j].x += delta_x;
          outline->points[j].y += delta_y;

#ifdef FT_DEBUG_LEVEL_TRACE
          if ( delta_x || delta_y )
          {
            FT_TRACE7(( "      %d: (%d, %d) -> (%d, %d)\n",
                        j,
                        outline->points[j].x - delta_x,
                        outline->points[j].y - delta_y,
                        outline->points[j].x,
                        outline->points[j].y ));
            count++;
          }
#endif
        }

#ifdef FT_DEBUG_LEVEL_TRACE
        if ( !count )
          FT_TRACE7(( "      none\n" ));
#endif
      }

      if ( localpoints != ALL_POINTS )
        FT_FREE( localpoints );
      FT_FREE( deltas_x );
      FT_FREE( deltas_y );

      offsetToData += tupleDataSize;

      FT_Stream_SeekSet( stream, here );
    }

    FT_TRACE5(( "\n" ));

  Fail2:
    if ( sharedpoints != ALL_POINTS )
      FT_FREE( sharedpoints );
    FT_FREE( tuple_coords );
    FT_FREE( im_start_coords );
    FT_FREE( im_end_coords );

    FT_FRAME_EXIT();

  Fail1:
    FT_FREE( points_org );
    FT_FREE( points_out );
    FT_FREE( has_delta );

    return error;
  }


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    tt_get_var_blend                                                   */
  /*                                                                       */
  /* <Description>                                                         */
  /*    An extended internal version of `TT_Get_MM_Blend' that returns     */
  /*    pointers instead of copying data, without any initialization of    */
  /*    the MM machinery in case it isn't loaded yet.                      */
  /*                                                                       */
  FT_LOCAL_DEF( FT_Error )
  tt_get_var_blend( TT_Face      face,
                    FT_UInt     *num_coords,
                    FT_Fixed*   *coords,
                    FT_Fixed*   *normalizedcoords,
                    FT_MM_Var*  *mm_var )
  {
    if ( face->blend )
    {
      if ( num_coords )
        *num_coords       = face->blend->num_axis;
      if ( coords )
        *coords           = face->blend->coords;
      if ( normalizedcoords )
        *normalizedcoords = face->blend->normalizedcoords;
      if ( mm_var )
        *mm_var           = face->blend->mmvar;
    }
    else
    {
      if ( num_coords )
        *num_coords = 0;
      if ( coords )
        *coords     = NULL;
      if ( mm_var )
        *mm_var     = NULL;
    }

    return FT_Err_Ok;
  }


  static void
  ft_var_done_item_variation_store( TT_Face          face,
                                    GX_ItemVarStore  itemStore )
  {
    FT_Memory  memory = FT_FACE_MEMORY( face );
    FT_UInt    i;


    if ( itemStore->varData )
    {
      for ( i = 0; i < itemStore->dataCount; i++ )
      {
        FT_FREE( itemStore->varData[i].regionIndices );
        FT_FREE( itemStore->varData[i].deltaSet );
      }

      FT_FREE( itemStore->varData );
    }

    if ( itemStore->varRegionList )
    {
      for ( i = 0; i < itemStore->regionCount; i++ )
        FT_FREE( itemStore->varRegionList[i].axisList );

      FT_FREE( itemStore->varRegionList );
    }
  }


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    tt_done_blend                                                      */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Free the blend internal data structure.                            */
  /*                                                                       */
  FT_LOCAL_DEF( void )
  tt_done_blend( TT_Face  face )
  {
    FT_Memory  memory = FT_FACE_MEMORY( face );
    GX_Blend   blend  = face->blend;


    if ( blend )
    {
      FT_UInt  i, num_axes;


      /* blend->num_axis might not be set up yet */
      num_axes = blend->mmvar->num_axis;

      FT_FREE( blend->coords );
      FT_FREE( blend->normalizedcoords );
      FT_FREE( blend->normalized_stylecoords );
      FT_FREE( blend->mmvar );

      if ( blend->avar_segment )
      {
        for ( i = 0; i < num_axes; i++ )
          FT_FREE( blend->avar_segment[i].correspondence );
        FT_FREE( blend->avar_segment );
      }

      if ( blend->hvar_table )
      {
        ft_var_done_item_variation_store( face,
                                          &blend->hvar_table->itemStore );

        FT_FREE( blend->hvar_table->widthMap.innerIndex );
        FT_FREE( blend->hvar_table->widthMap.outerIndex );
        FT_FREE( blend->hvar_table );
      }

      if ( blend->vvar_table )
      {
        ft_var_done_item_variation_store( face,
                                          &blend->vvar_table->itemStore );

        FT_FREE( blend->vvar_table->widthMap.innerIndex );
        FT_FREE( blend->vvar_table->widthMap.outerIndex );
        FT_FREE( blend->vvar_table );
      }

      if ( blend->mvar_table )
      {
        ft_var_done_item_variation_store( face,
                                          &blend->mvar_table->itemStore );

        FT_FREE( blend->mvar_table->values );
        FT_FREE( blend->mvar_table );
      }

      FT_FREE( blend->tuplecoords );
      FT_FREE( blend->glyphoffsets );
      FT_FREE( blend );
    }
  }

#else /* !TT_CONFIG_OPTION_GX_VAR_SUPPORT */

  /* ANSI C doesn't like empty source files */
  typedef int  _tt_gxvar_dummy;

#endif /* !TT_CONFIG_OPTION_GX_VAR_SUPPORT */


/* END */
