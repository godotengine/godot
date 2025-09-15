/****************************************************************************
 *
 * ttgxvar.c
 *
 *   TrueType GX Font Variation loader
 *
 * Copyright (C) 2004-2024 by
 * David Turner, Robert Wilhelm, Werner Lemberg, and George Williams.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


  /**************************************************************************
   *
   * Apple documents the `fvar', `gvar', `cvar', and `avar' tables at
   *
   *   https://developer.apple.com/fonts/TrueType-Reference-Manual/RM06/Chap6[fgca]var.html
   *
   * The documentation for `gvar' is not intelligible; `cvar' refers you
   * to `gvar' and is thus also incomprehensible.
   *
   * The documentation for `avar' appears correct, but Apple has no fonts
   * with an `avar' table, so it is hard to test.
   *
   * Many thanks to John Jenkins (at Apple) in figuring this out.
   *
   *
   * Apple's `kern' table has some references to tuple indices, but as
   * there is no indication where these indices are defined, nor how to
   * interpolate the kerning values (different tuples have different
   * classes) this issue is ignored.
   *
   */


#include <ft2build.h>
#include <freetype/internal/ftdebug.h>
#include FT_CONFIG_CONFIG_H
#include <freetype/internal/ftcalc.h>
#include <freetype/internal/ftstream.h>
#include <freetype/internal/sfnt.h>
#include <freetype/internal/services/svmetric.h>
#include <freetype/tttags.h>
#include <freetype/ttnameid.h>
#include <freetype/ftmm.h>
#include <freetype/ftlist.h>

#include "ttpload.h"
#include "ttgxvar.h"

#include "tterrors.h"


#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT


#define FT_Stream_FTell( stream )                         \
          (FT_ULong)( (stream)->cursor - (stream)->base )
#define FT_Stream_SeekSet( stream, off )                               \
          (stream)->cursor =                                           \
            ( (off) < (FT_ULong)( (stream)->limit - (stream)->base ) ) \
                        ? (stream)->base + (off)                       \
                        : (stream)->limit


  /* some macros we need */
#define FT_fdot14ToFixed( x )                  \
          ( (FT_Fixed)( (FT_ULong)(x) << 2 ) )
#define FT_intToFixed( i )                      \
          ( (FT_Fixed)( (FT_ULong)(i) << 16 ) )
#define FT_fdot6ToFixed( i )                    \
          ( (FT_Fixed)( (FT_ULong)(i) << 10 ) )
#define FT_fixedToInt( x )                          \
          ( (FT_Short)( ( (x) + 0x8000U ) >> 16 ) )
#define FT_fixedToFdot6( x )                    \
          ( (FT_Pos)( ( (x) + 0x200 ) >> 10 ) )


  /**************************************************************************
   *
   * The macro FT_COMPONENT is used in trace mode.  It is an implicit
   * parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log
   * messages during execution.
   */
#undef  FT_COMPONENT
#define FT_COMPONENT  ttgxvar


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                       Internal Routines                       *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/


  /**************************************************************************
   *
   * The macro ALL_POINTS is used in `ft_var_readpackedpoints'.  It
   * indicates that there is a delta for every point without needing to
   * enumerate all of them.
   */

  /* ensure that value `0' has the same width as a pointer */
#define ALL_POINTS  (FT_UShort*)~(FT_PtrDist)0


#define GX_PT_POINTS_ARE_WORDS      0x80U
#define GX_PT_POINT_RUN_COUNT_MASK  0x7FU


  /**************************************************************************
   *
   * @Function:
   *   ft_var_readpackedpoints
   *
   * @Description:
   *   Read a set of points to which the following deltas will apply.
   *   Points are packed with a run length encoding.
   *
   * @Input:
   *   stream ::
   *     The data stream.
   *
   * @Output:
   *   point_cnt ::
   *     The number of points read.  A zero value means that
   *     all points in the glyph will be affected, without
   *     enumerating them individually.
   *
   * @Return:
   *   An array of FT_UShort containing the affected points or the
   *   special value ALL_POINTS.
   */
  static FT_UShort*
  ft_var_readpackedpoints( FT_Stream  stream,
                           FT_UInt   *point_cnt )
  {
    FT_UShort *points = NULL;
    FT_UInt    n;
    FT_UInt    runcnt, cnt;
    FT_UInt    i, j;
    FT_UShort  first;
    FT_Byte*   p;
    FT_Memory  memory = stream->memory;
    FT_Error   error;


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

    if ( FT_QNEW_ARRAY( points, n ) )
      return NULL;

    p     = stream->cursor;
    first = 0;
    i     = 0;
    while ( i < n )
    {
      if ( p >= stream->limit )
        goto Fail;

      runcnt = FT_NEXT_BYTE( p );
      cnt    = runcnt & GX_PT_POINT_RUN_COUNT_MASK;

      /* first point not included in run count */
      cnt++;
      if ( cnt > n - i )
        cnt = n - i;

      if ( runcnt & GX_PT_POINTS_ARE_WORDS )
      {
        if ( 2 * cnt > (FT_UInt)( stream->limit - p ) )
          goto Fail;

        for ( j = 0; j < cnt; j++ )
        {
          first      += FT_NEXT_USHORT( p );
          points[i++] = first;
        }
      }
      else
      {
        if ( cnt > (FT_UInt)( stream->limit - p ) )
          goto Fail;

        for ( j = 0; j < cnt; j++ )
        {
          first      += FT_NEXT_BYTE( p );
          points[i++] = first;
        }
      }
    }

    stream->cursor = p;

    *point_cnt = n;

    return points;

  Fail:
    FT_TRACE1(( "ft_var_readpackedpoints: invalid table\n" ));

    FT_FREE( points );
    return NULL;
  }


#define GX_DT_DELTAS_ARE_ZERO       0x80U
#define GX_DT_DELTAS_ARE_WORDS      0x40U
#define GX_DT_DELTA_RUN_COUNT_MASK  0x3FU


  /**************************************************************************
   *
   * @Function:
   *   ft_var_readpackeddeltas
   *
   * @Description:
   *   Read a set of deltas.  These are packed slightly differently than
   *   points.  In particular there is no overall count.
   *
   * @Input:
   *   stream ::
   *     The data stream.
   *
   *   delta_cnt ::
   *     The number of deltas to be read.
   *
   * @Return:
   *   An array of FT_Fixed containing the deltas for the affected
   *   points.  (This only gets the deltas for one dimension.  It will
   *   generally be called twice, once for x, once for y.  When used in
   *   cvt table, it will only be called once.)
   *
   *   We use FT_Fixed to avoid accumulation errors while summing up all
   *   deltas (the rounding to integer values happens as the very last
   *   step).
   */
  static FT_Fixed*
  ft_var_readpackeddeltas( FT_Stream  stream,
                           FT_UInt    delta_cnt )
  {
    FT_Fixed  *deltas = NULL;
    FT_UInt    runcnt, cnt;
    FT_UInt    i, j;
    FT_Byte*   p;
    FT_Memory  memory = stream->memory;
    FT_Error   error;


    if ( FT_QNEW_ARRAY( deltas, delta_cnt ) )
      return NULL;

    p = stream->cursor;
    i = 0;
    while ( i < delta_cnt )
    {
      if ( p >= stream->limit )
        goto Fail;

      runcnt = FT_NEXT_BYTE( p );
      cnt    = runcnt & GX_DT_DELTA_RUN_COUNT_MASK;

      /* first point not included in run count */
      cnt++;
      if ( cnt > delta_cnt - i )
        cnt = delta_cnt - i;

      if ( runcnt & GX_DT_DELTAS_ARE_ZERO )
      {
        for ( j = 0; j < cnt; j++ )
          deltas[i++] = 0;
      }
      else if ( runcnt & GX_DT_DELTAS_ARE_WORDS )
      {
        if ( 2 * cnt > (FT_UInt)( stream->limit - p ) )
          goto Fail;

        for ( j = 0; j < cnt; j++ )
          deltas[i++] = FT_intToFixed( FT_NEXT_SHORT( p ) );
      }
      else
      {
        if ( cnt > (FT_UInt)( stream->limit - p ) )
          goto Fail;

        for ( j = 0; j < cnt; j++ )
          deltas[i++] = FT_intToFixed( FT_NEXT_CHAR( p ) );
      }
    }

    stream->cursor = p;

    return deltas;

  Fail:
    FT_TRACE1(( "ft_var_readpackeddeltas: invalid table\n" ));

    FT_FREE( deltas );
    return NULL;
  }


  /**************************************************************************
   *
   * @Function:
   *   ft_var_load_avar
   *
   * @Description:
   *   Parse the `avar' table if present.  It need not be, so we return
   *   nothing.
   *
   * @InOut:
   *   face ::
   *     The font face.
   */
  static void
  ft_var_load_avar( TT_Face  face )
  {
    FT_Error   error;
    FT_Stream  stream = FT_FACE_STREAM( face );
    FT_Memory  memory = stream->memory;
    FT_Int     i, j;

    GX_Blend        blend  = face->blend;
    GX_AVarSegment  segment;
    GX_AVarTable    table;

    FT_Long   version;
    FT_Long   axisCount;
    FT_ULong  table_len;

#ifndef TT_CONFIG_OPTION_NO_BORING_EXPANSION
    FT_ULong  table_offset;
    FT_ULong  store_offset;
    FT_ULong  axisMap_offset;
#endif


    FT_TRACE2(( "AVAR " ));

    blend->avar_loaded = TRUE;
    error = face->goto_table( face, TTAG_avar, stream, &table_len );
    if ( error )
    {
      FT_TRACE2(( "is missing\n" ));
      return;
    }

#ifndef TT_CONFIG_OPTION_NO_BORING_EXPANSION
    table_offset = FT_STREAM_POS();
#endif

    if ( FT_FRAME_ENTER( table_len ) )
      return;

    version   = FT_GET_LONG();
    axisCount = FT_GET_LONG();

    if ( version != 0x00010000L
#ifndef TT_CONFIG_OPTION_NO_BORING_EXPANSION
         && version != 0x00020000L
#endif
       )
    {
      FT_TRACE2(( "bad table version\n" ));
      goto Exit;
    }

    FT_TRACE2(( "loaded\n" ));

    if ( axisCount != (FT_Long)blend->mmvar->num_axis )
    {
      FT_TRACE2(( "ft_var_load_avar:"
                  " number of axes in `avar' and `fvar'\n" ));
      FT_TRACE2(( "                  table are different\n" ));
      goto Exit;
    }

    if ( FT_NEW( blend->avar_table ) )
      goto Exit;
    table = blend->avar_table;

    if ( FT_QNEW_ARRAY( table->avar_segment, axisCount ) )
      goto Exit;

    segment = &table->avar_segment[0];
    for ( i = 0; i < axisCount; i++, segment++ )
    {
      FT_TRACE5(( "  axis %d:\n", i ));

      segment->pairCount = FT_GET_USHORT();
      if ( (FT_ULong)segment->pairCount * 4 > table_len                 ||
           FT_QNEW_ARRAY( segment->correspondence, segment->pairCount ) )
      {
        /* Failure.  Free everything we have done so far.  We must do */
        /* it right now since loading the `avar' table is optional.   */

        for ( j = i - 1; j >= 0; j-- )
          FT_FREE( table->avar_segment[j].correspondence );

        FT_FREE( table->avar_segment );
        goto Exit;
      }

      for ( j = 0; j < segment->pairCount; j++ )
      {
        segment->correspondence[j].fromCoord =
          FT_fdot14ToFixed( FT_GET_SHORT() );
        segment->correspondence[j].toCoord =
          FT_fdot14ToFixed( FT_GET_SHORT() );

        FT_TRACE5(( "    mapping %.5f to %.5f\n",
                    (double)segment->correspondence[j].fromCoord / 65536,
                    (double)segment->correspondence[j].toCoord / 65536 ));
      }

      FT_TRACE5(( "\n" ));
    }

#ifndef TT_CONFIG_OPTION_NO_BORING_EXPANSION
    if ( version < 0x00020000L )
      goto Exit;

    axisMap_offset = FT_GET_ULONG();
    store_offset   = FT_GET_ULONG();

    if ( store_offset )
    {
      error = tt_var_load_item_variation_store(
                FT_FACE( face ),
                table_offset + store_offset,
                &table->itemStore );
      if ( error )
        goto Exit;
    }

    if ( axisMap_offset )
    {
      error = tt_var_load_delta_set_index_mapping(
                FT_FACE( face ),
                table_offset + axisMap_offset,
                &table->axisMap,
                &table->itemStore,
                table_len );
      if ( error )
        goto Exit;
    }
#endif


  Exit:
    FT_FRAME_EXIT();
  }


  FT_LOCAL_DEF( FT_Error )
  tt_var_load_item_variation_store( FT_Face          face,      /* TT_Face */
                                    FT_ULong         offset,
                                    GX_ItemVarStore  itemStore )
  {
    TT_Face    ttface = (TT_Face)face;
    FT_Stream  stream = FT_FACE_STREAM( face );
    FT_Memory  memory = stream->memory;

    FT_Error   error;
    FT_UShort  format;
    FT_ULong   region_offset;

    FT_UInt    data_count;
    FT_UShort  axis_count;
    FT_UInt    region_count;

    FT_UInt  i, j;
    FT_Bool  long_words;

    GX_Blend   blend           = ttface->blend;
    FT_ULong*  dataOffsetArray = NULL;


    if ( FT_STREAM_SEEK( offset ) ||
         FT_READ_USHORT( format ) )
      goto Exit;

    if ( format != 1 )
    {
      FT_TRACE2(( "tt_var_load_item_variation_store: bad store format %d\n",
                  format ));
      error = FT_THROW( Invalid_Table );
      goto Exit;
    }

    /* read top level fields */
    if ( FT_READ_ULONG( region_offset ) ||
         FT_READ_USHORT( data_count )   )
      goto Exit;

    /* we need at least one entry in `itemStore->varData' */
    if ( !data_count )
    {
      FT_TRACE2(( "tt_var_load_item_variation_store: missing varData\n" ));
      error = FT_THROW( Invalid_Table );
      goto Exit;
    }

    /* make temporary copy of item variation data offsets; */
    /* we will parse region list first, then come back     */
    if ( FT_QNEW_ARRAY( dataOffsetArray, data_count ) )
      goto Exit;

    for ( i = 0; i < data_count; i++ )
    {
      if ( FT_READ_ULONG( dataOffsetArray[i] ) )
        goto Exit;
    }

    /* parse array of region records (region list) */
    if ( FT_STREAM_SEEK( offset + region_offset ) )
      goto Exit;

    if ( FT_READ_USHORT( axis_count )   ||
         FT_READ_USHORT( region_count ) )
      goto Exit;

    if ( axis_count != (FT_Long)blend->mmvar->num_axis )
    {
      FT_TRACE2(( "tt_var_load_item_variation_store:"
                  " number of axes in item variation store\n" ));
      FT_TRACE2(( "                                 "
                  " and `fvar' table are different\n" ));
      error = FT_THROW( Invalid_Table );
      goto Exit;
    }
    itemStore->axisCount = axis_count;

    /* new constraint in OpenType 1.8.4 */
    if ( region_count >= 32768U )
    {
      FT_TRACE2(( "tt_var_load_item_variation_store:"
                  " too many variation region tables\n" ));
      error = FT_THROW( Invalid_Table );
      goto Exit;
    }

    if ( FT_NEW_ARRAY( itemStore->varRegionList, region_count ) )
      goto Exit;
    itemStore->regionCount = region_count;

    for ( i = 0; i < itemStore->regionCount; i++ )
    {
      GX_AxisCoords  axisCoords;


      if ( FT_NEW_ARRAY( itemStore->varRegionList[i].axisList, axis_count ) )
        goto Exit;

      axisCoords = itemStore->varRegionList[i].axisList;

      for ( j = 0; j < itemStore->axisCount; j++ )
      {
        FT_Int  start, peak, end;


        if ( FT_READ_SHORT( start ) ||
             FT_READ_SHORT( peak )  ||
             FT_READ_SHORT( end )   )
          goto Exit;

        /* immediately tag invalid ranges with special peak = 0 */
        if ( ( start < 0 && end > 0 ) || start > peak || peak > end )
          peak = 0;

        axisCoords[j].startCoord = FT_fdot14ToFixed( start );
        axisCoords[j].peakCoord  = FT_fdot14ToFixed( peak );
        axisCoords[j].endCoord   = FT_fdot14ToFixed( end );
      }
    }

    /* end of region list parse */

    /* use dataOffsetArray now to parse varData items */
    if ( FT_NEW_ARRAY( itemStore->varData, data_count ) )
      goto Exit;
    itemStore->dataCount = data_count;

    for ( i = 0; i < data_count; i++ )
    {
      GX_ItemVarData  varData = &itemStore->varData[i];

      FT_UInt    item_count;
      FT_UShort  word_delta_count;
      FT_UInt    region_idx_count;
      FT_UInt    per_region_size;


      if ( FT_STREAM_SEEK( offset + dataOffsetArray[i] ) )
        goto Exit;

      if ( FT_READ_USHORT( item_count )       ||
           FT_READ_USHORT( word_delta_count ) ||
           FT_READ_USHORT( region_idx_count ) )
        goto Exit;

      long_words        = !!( word_delta_count & 0x8000 );
      word_delta_count &= 0x7FFF;

      /* check some data consistency */
      if ( word_delta_count > region_idx_count )
      {
        FT_TRACE2(( "bad short count %d or region count %d\n",
                    word_delta_count,
                    region_idx_count ));
        error = FT_THROW( Invalid_Table );
        goto Exit;
      }

      if ( region_idx_count > itemStore->regionCount )
      {
        FT_TRACE2(( "inconsistent regionCount %d in varData[%d]\n",
                    region_idx_count,
                    i ));
        error = FT_THROW( Invalid_Table );
        goto Exit;
      }

      /* parse region indices */
      if ( FT_NEW_ARRAY( varData->regionIndices, region_idx_count ) )
        goto Exit;
      varData->regionIdxCount = region_idx_count;
      varData->wordDeltaCount = word_delta_count;
      varData->longWords      = long_words;

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

      per_region_size = word_delta_count + region_idx_count;
      if ( long_words )
        per_region_size *= 2;

      if ( FT_NEW_ARRAY( varData->deltaSet, per_region_size * item_count ) )
        goto Exit;
      if ( FT_Stream_Read( stream,
                           varData->deltaSet,
                           per_region_size * item_count ) )
      {
        FT_TRACE2(( "deltaSet read failed." ));
        error = FT_THROW( Invalid_Table );
        goto Exit;
      }

      varData->itemCount = item_count;
    }

  Exit:
    FT_FREE( dataOffsetArray );

    return error;
  }


  FT_LOCAL_DEF( FT_Error )
  tt_var_load_delta_set_index_mapping( FT_Face            face, /* TT_Face */
                                       FT_ULong           offset,
                                       GX_DeltaSetIdxMap  map,
                                       GX_ItemVarStore    itemStore,
                                       FT_ULong           table_len )
  {
    FT_Stream  stream = FT_FACE_STREAM( face );
    FT_Memory  memory = stream->memory;

    FT_Error  error;

    FT_Byte   format;
    FT_Byte   entryFormat;
    FT_UInt   entrySize;
    FT_UInt   innerBitCount;
    FT_UInt   innerIndexMask;
    FT_ULong  i;
    FT_UInt   j;


    if ( FT_STREAM_SEEK( offset )    ||
         FT_READ_BYTE( format )      ||
         FT_READ_BYTE( entryFormat ) )
      goto Exit;

    if ( format == 0 )
    {
      if ( FT_READ_USHORT( map->mapCount ) )
        goto Exit;
    }
    else if ( format == 1 ) /* new in OpenType 1.9 */
    {
      if ( FT_READ_ULONG( map->mapCount ) )
        goto Exit;
    }
    else
    {
      FT_TRACE2(( "bad map format %d\n", format ));
      error = FT_THROW( Invalid_Table );
      goto Exit;
    }

    if ( entryFormat & 0xC0 )
    {
      FT_TRACE2(( "bad entry format %d\n", format ));
      error = FT_THROW( Invalid_Table );
      goto Exit;
    }

    /* bytes per entry: 1, 2, 3, or 4 */
    entrySize      = ( ( entryFormat & 0x30 ) >> 4 ) + 1;
    innerBitCount  = ( entryFormat & 0x0F ) + 1;
    innerIndexMask = ( 1 << innerBitCount ) - 1;

    /* rough sanity check */
    if ( map->mapCount * entrySize > table_len )
    {
      FT_TRACE1(( "tt_var_load_delta_set_index_mapping:"
                  " invalid number of delta-set index mappings\n" ));
      error = FT_THROW( Invalid_Table );
      goto Exit;
    }

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

      /* new in OpenType 1.8.4 */
      if ( mapData == 0xFFFFFFFFUL )
      {
        /* no variation data for this item */
        map->outerIndex[i] = 0xFFFFU;
        map->innerIndex[i] = 0xFFFFU;

        continue;
      }

      outerIndex = mapData >> innerBitCount;

      if ( outerIndex >= itemStore->dataCount )
      {
        FT_TRACE2(( "outerIndex[%ld] == %d out of range\n",
                    i,
                    outerIndex ));
        error = FT_THROW( Invalid_Table );
        goto Exit;
      }

      map->outerIndex[i] = outerIndex;

      innerIndex = mapData & innerIndexMask;

      if ( innerIndex >= itemStore->varData[outerIndex].itemCount )
      {
        FT_TRACE2(( "innerIndex[%ld] == %d out of range\n",
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


  /**************************************************************************
   *
   * @Function:
   *   ft_var_load_hvvar
   *
   * @Description:
   *   If `vertical' is zero, parse the `HVAR' table and set
   *   `blend->hvar_loaded' to TRUE.  On success, `blend->hvar_checked'
   *   is set to TRUE.
   *
   *   If `vertical' is not zero, parse the `VVAR' table and set
   *   `blend->vvar_loaded' to TRUE.  On success, `blend->vvar_checked'
   *   is set to TRUE.
   *
   *   Some memory may remain allocated on error; it is always freed in
   *   `tt_done_blend', however.
   *
   * @InOut:
   *   face ::
   *     The font face.
   *
   * @Return:
   *   FreeType error code.  0 means success.
   */
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

    error = tt_var_load_item_variation_store(
              FT_FACE( face ),
              table_offset + store_offset,
              &table->itemStore );
    if ( error )
      goto Exit;

    if ( widthMap_offset )
    {
      error = tt_var_load_delta_set_index_mapping(
                FT_FACE( face ),
                table_offset + widthMap_offset,
                &table->widthMap,
                &table->itemStore,
                table_len );
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


  FT_LOCAL_DEF( FT_ItemVarDelta )
  tt_var_get_item_delta( FT_Face          face,        /* TT_Face */
                         GX_ItemVarStore  itemStore,
                         FT_UInt          outerIndex,
                         FT_UInt          innerIndex )
  {
    TT_Face    ttface = (TT_Face)face;
    FT_Stream  stream = FT_FACE_STREAM( face );
    FT_Memory  memory = stream->memory;
    FT_Error   error  = FT_Err_Ok;

    GX_ItemVarData    varData;
    FT_ItemVarDelta*  deltaSet = NULL;
    FT_ItemVarDelta   deltaSetStack[16];

    FT_Fixed*  scalars = NULL;
    FT_Fixed   scalarsStack[16];

    FT_UInt          master, j;
    FT_ItemVarDelta  returnValue = 0;
    FT_UInt          per_region_size;
    FT_Byte*         bytes;


    if ( !ttface->blend || !ttface->blend->normalizedcoords )
      return 0;

    /* OpenType 1.8.4+: No variation data for this item */
    /* as indices have special value 0xFFFF.            */
    if ( outerIndex == 0xFFFF && innerIndex == 0xFFFF )
      return 0;

    /* See pseudo code from `Font Variations Overview' */
    /* in the OpenType specification.                  */

    if ( outerIndex >= itemStore->dataCount )
      return 0; /* Out of range. */

    varData = &itemStore->varData[outerIndex];

    if ( innerIndex >= varData->itemCount )
      return 0; /* Out of range. */

    if ( varData->regionIdxCount == 0 )
      return 0; /* Avoid "applying zero offset to null pointer". */

    if ( varData->regionIdxCount < 16 )
    {
      deltaSet = deltaSetStack;
      scalars  = scalarsStack;
    }
    else
    {
      if ( FT_QNEW_ARRAY( deltaSet, varData->regionIdxCount ) )
        goto Exit;
      if ( FT_QNEW_ARRAY( scalars, varData->regionIdxCount ) )
        goto Exit;
    }

    /* Parse delta set.                                            */
    /*                                                             */
    /* Deltas are (word_delta_count + region_idx_count) bytes each */
    /* if `longWords` isn't set, and twice as much otherwise.      */
    per_region_size = varData->wordDeltaCount + varData->regionIdxCount;
    if ( varData->longWords )
      per_region_size *= 2;

    bytes = varData->deltaSet + per_region_size * innerIndex;

    if ( varData->longWords )
    {
      for ( master = 0; master < varData->wordDeltaCount; master++ )
        deltaSet[master] = FT_NEXT_LONG( bytes );
      for ( ; master < varData->regionIdxCount; master++ )
        deltaSet[master] = FT_NEXT_SHORT( bytes );
    }
    else
    {
      for ( master = 0; master < varData->wordDeltaCount; master++ )
        deltaSet[master] = FT_NEXT_SHORT( bytes );
      for ( ; master < varData->regionIdxCount; master++ )
        deltaSet[master] = FT_NEXT_CHAR( bytes );
    }

    /* outer loop steps through master designs to be blended */
    for ( master = 0; master < varData->regionIdxCount; master++ )
    {
      FT_Fixed  scalar      = 0x10000L;
      FT_UInt   regionIndex = varData->regionIndices[master];

      GX_AxisCoords  axis = itemStore->varRegionList[regionIndex].axisList;


      /* inner loop steps through axes in this region */
      for ( j = 0; j < itemStore->axisCount; j++, axis++ )
      {
        FT_Fixed  ncv = ttface->blend->normalizedcoords[j];


        /* compute the scalar contribution of this axis */
        /* with peak of 0 used for invalid axes         */
        if ( axis->peakCoord == ncv ||
             axis->peakCoord == 0   )
          continue;

        /* ignore this region if coords are out of range */
        else if ( ncv <= axis->startCoord ||
                  ncv >= axis->endCoord   )
        {
          scalar = 0;
          break;
        }

        /* cumulative product of all the axis scalars */
        else if ( ncv < axis->peakCoord )
          scalar = FT_MulDiv( scalar,
                              ncv - axis->startCoord,
                              axis->peakCoord - axis->startCoord );
        else   /* ncv > axis->peakCoord */
          scalar = FT_MulDiv( scalar,
                              axis->endCoord - ncv,
                              axis->endCoord - axis->peakCoord );

      } /* per-axis loop */

      scalars[master] = scalar;

    } /* per-region loop */


    /* Compute the scaled delta for this region.
     *
     * From: https://docs.microsoft.com/en-us/typography/opentype/spec/otvarcommonformats#item-variation-store-header-and-item-variation-data-subtables:
     *
     *   `Fixed` is a 32-bit (16.16) type and, in the general case, requires
     *   32-bit deltas.  As described above, the `DeltaSet` record can
     *   accommodate deltas that are, logically, either 16-bit or 32-bit.
     *   When scaled deltas are applied to `Fixed` values, the `Fixed` value
     *   is treated like a 32-bit integer.
     *
     * `FT_MulAddFix` internally uses 64-bit precision; it thus can handle
     * deltas ranging from small 8-bit to large 32-bit values that are
     * applied to 16.16 `FT_Fixed` / OpenType `Fixed` values.
     */
    returnValue = FT_MulAddFix( scalars, deltaSet, varData->regionIdxCount );

  Exit:
    if ( scalars != scalarsStack )
      FT_FREE( scalars );
    if ( deltaSet != deltaSetStack )
      FT_FREE( deltaSet );

    return returnValue;
  }


  /**************************************************************************
   *
   * @Function:
   *   tt_hvadvance_adjust
   *
   * @Description:
   *   Apply `HVAR' advance width or `VVAR' advance height adjustment of
   *   a given glyph.
   *
   * @Input:
   *   gindex ::
   *     The glyph index.
   *
   *   vertical ::
   *     If set, handle `VVAR' table.
   *
   * @InOut:
   *   face ::
   *     The font face.
   *
   *   adelta ::
   *     Points to width or height value that gets modified.
   */
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
      /* no widthMap data */
      outerIndex = 0;
      innerIndex = gindex;
    }

    delta = tt_var_get_item_delta( FT_FACE( face ),
                                   &table->itemStore,
                                   outerIndex,
                                   innerIndex );

    if ( delta )
    {
      FT_TRACE5(( "%s value %d adjusted by %d unit%s (%s)\n",
                  vertical ? "vertical height" : "horizontal width",
                  *avalue,
                  delta,
                  delta == 1 ? "" : "s",
                  vertical ? "VVAR" : "HVAR" ));

      *avalue = ADD_INT( *avalue, delta );
    }

  Exit:
    return error;
  }


  FT_LOCAL_DEF( FT_Error )
  tt_hadvance_adjust( FT_Face  face,    /* TT_Face */
                      FT_UInt  gindex,
                      FT_Int  *avalue )
  {
    return tt_hvadvance_adjust( (TT_Face)face, gindex, avalue, 0 );
  }


  FT_LOCAL_DEF( FT_Error )
  tt_vadvance_adjust( FT_Face  face,    /* TT_Face */
                      FT_UInt  gindex,
                      FT_Int  *avalue )
  {
    return tt_hvadvance_adjust( (TT_Face)face, gindex, avalue, 1 );
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


  /**************************************************************************
   *
   * @Function:
   *   ft_var_load_mvar
   *
   * @Description:
   *   Parse the `MVAR' table.
   *
   *   Some memory may remain allocated on error; it is always freed in
   *   `tt_done_blend', however.
   *
   * @InOut:
   *   face ::
   *     The font face.
   */
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

    error = tt_var_load_item_variation_store(
              FT_FACE( face ),
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
    limit     = FT_OFFSET( value, blend->mvar_table->valueCount );
    itemStore = &blend->mvar_table->itemStore;

    for ( ; value < limit; value++ )
    {
      value->tag        = FT_GET_ULONG();
      value->outerIndex = FT_GET_USHORT();
      value->innerIndex = FT_GET_USHORT();

      /* new in OpenType 1.8.4 */
      if ( value->outerIndex == 0xFFFFU && value->innerIndex == 0xFFFFU )
      {
        /* no variation data for this item */
        continue;
      }

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
    limit = FT_OFFSET( value, blend->mvar_table->valueCount );

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
  ft_size_reset_iterator( FT_ListNode  node,
                          void*        user )
  {
    FT_Size                       size = (FT_Size)node->data;
    FT_Service_MetricsVariations  var  = (FT_Service_MetricsVariations)user;


    var->size_reset( size );

    return FT_Err_Ok;
  }


  /**************************************************************************
   *
   * @Function:
   *   tt_apply_mvar
   *
   * @Description:
   *   Apply `MVAR' table adjustments.
   *
   * @InOut:
   *   face ::
   *     The font face.
   */
  FT_LOCAL_DEF( void )
  tt_apply_mvar( FT_Face  face )  /* TT_Face */
  {
    TT_Face  ttface = (TT_Face)face;

    GX_Blend  blend = ttface->blend;
    GX_Value  value, limit;

    FT_Short  mvar_hasc_delta = 0;
    FT_Short  mvar_hdsc_delta = 0;
    FT_Short  mvar_hlgp_delta = 0;


    if ( !( ttface->variation_support & TT_FACE_FLAG_VAR_MVAR ) )
      return;

    value = blend->mvar_table->values;
    limit = FT_OFFSET( value, blend->mvar_table->valueCount );

    for ( ; value < limit; value++ )
    {
      FT_Short*  p = ft_var_get_value_pointer( ttface, value->tag );
      FT_Int     delta;


      delta = tt_var_get_item_delta( face,
                                     &blend->mvar_table->itemStore,
                                     value->outerIndex,
                                     value->innerIndex );

      if ( p && delta )
      {
        FT_TRACE5(( "value %c%c%c%c (%d unit%s) adjusted by %d unit%s (MVAR)\n",
                    (FT_Char)( value->tag >> 24 ),
                    (FT_Char)( value->tag >> 16 ),
                    (FT_Char)( value->tag >> 8 ),
                    (FT_Char)( value->tag ),
                    value->unmodified,
                    value->unmodified == 1 ? "" : "s",
                    delta,
                    delta == 1 ? "" : "s" ));

        /* since we handle both signed and unsigned values as FT_Short, */
        /* ensure proper overflow arithmetic                            */
        *p = (FT_Short)( value->unmodified + (FT_Short)delta );

        /* Treat hasc, hdsc and hlgp specially, see below. */
        if ( value->tag == MVAR_TAG_HASC )
          mvar_hasc_delta = (FT_Short)delta;
        else if ( value->tag == MVAR_TAG_HDSC )
          mvar_hdsc_delta = (FT_Short)delta;
        else if ( value->tag == MVAR_TAG_HLGP )
          mvar_hlgp_delta = (FT_Short)delta;
      }
    }

    /* adjust all derived values */
    {
      FT_Service_MetricsVariations  var =
        (FT_Service_MetricsVariations)ttface->face_var;

      /*
       * Apply the deltas of hasc, hdsc and hlgp to the FT_Face's ascender,
       * descender and height attributes, no matter how they were originally
       * computed.
       *
       * (Code that ignores those and accesses the font's metrics values
       * directly is already served by the delta application code above.)
       *
       * The MVAR table supports variations for both typo and win metrics.
       * According to Behdad Esfahbod, the thinking of the working group was
       * that no one uses win metrics anymore for setting line metrics (the
       * specification even calls these metrics "horizontal clipping
       * ascent/descent", probably for their role on the Windows platform in
       * computing clipping boxes), and new fonts should use typo metrics, so
       * typo deltas should be applied to whatever sfnt_load_face decided the
       * line metrics should be.
       *
       * Before, the following led to different line metrics between default
       * outline and instances, visible when e.g. the default outlines were
       * used as the regular face and instances for everything else:
       *
       * 1. sfnt_load_face applied the hhea metrics by default.
       * 2. This code later applied the typo metrics by default, regardless of
       *    whether they were actually changed or the font had the OS/2 table's
       *    fsSelection's bit 7 (USE_TYPO_METRICS) set.
       */
      FT_Short  current_line_gap = face->height - face->ascender +
                                   face->descender;


      face->ascender  = face->ascender + mvar_hasc_delta;
      face->descender = face->descender + mvar_hdsc_delta;
      face->height    = face->ascender - face->descender +
                        current_line_gap + mvar_hlgp_delta;

      face->underline_position  = ttface->postscript.underlinePosition -
                                  ttface->postscript.underlineThickness / 2;
      face->underline_thickness = ttface->postscript.underlineThickness;

      /* iterate over all FT_Size objects and call `var->size_reset' */
      /* to propagate the metrics changes                            */
      if ( var && var->size_reset )
        FT_List_Iterate( &face->sizes_list,
                         ft_size_reset_iterator,
                         (void*)var );
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


  /**************************************************************************
   *
   * @Function:
   *   ft_var_load_gvar
   *
   * @Description:
   *   Parse the `gvar' table if present.  If `fvar' is there, `gvar' had
   *   better be there too.
   *
   * @InOut:
   *   face ::
   *     The font face.
   *
   * @Return:
   *   FreeType error code.  0 means success.
   */
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
    FT_ULong      offsets_len;
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
      FT_TRACE1(( "ft_var_load_gvar:"
                  " number of axes in `gvar' and `cvar'\n" ));
      FT_TRACE1(( "                  table are different\n" ));
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

    /* offsets can be either 2 or 4 bytes                  */
    /* (one more offset than glyphs, to mark size of last) */
    offsets_len = ( gvar_head.glyphCount + 1 ) *
                  ( ( gvar_head.flags & 1 ) ? 4L : 2L );

    /* rough sanity check */
    if (offsets_len > table_len )
    {
      FT_TRACE1(( "ft_var_load_gvar: invalid number of glyphs\n" ));
      error = FT_THROW( Invalid_Table );
      goto Exit;
    }

    FT_TRACE2(( "loaded\n" ));

    blend->gvar_size = table_len;
    offsetToData     = gvar_start + gvar_head.offsetToData;

    FT_TRACE5(( "gvar: there %s %d shared coordinate%s:\n",
                gvar_head.globalCoordCount == 1 ? "is" : "are",
                gvar_head.globalCoordCount,
                gvar_head.globalCoordCount == 1 ? "" : "s" ));

    if ( FT_FRAME_ENTER( offsets_len ) )
      goto Exit;

    /* offsets (one more offset than glyphs, to mark size of last) */
    if ( FT_QNEW_ARRAY( blend->glyphoffsets, gvar_head.glyphCount + 1 ) )
      goto Fail2;

    if ( gvar_head.flags & 1 )
    {
      FT_ULong  limit      = gvar_start + table_len;
      FT_ULong  max_offset = 0;


      for ( i = 0; i <= gvar_head.glyphCount; i++ )
      {
        blend->glyphoffsets[i] = offsetToData + FT_GET_ULONG();

        if ( max_offset <= blend->glyphoffsets[i] )
          max_offset = blend->glyphoffsets[i];
        else
        {
          FT_TRACE2(( "ft_var_load_gvar:"
                      " glyph variation data offset %d not monotonic\n",
                      i ));
          blend->glyphoffsets[i] = max_offset;
        }

        /* use `<', not `<=' */
        if ( limit < blend->glyphoffsets[i] )
        {
          FT_TRACE2(( "ft_var_load_gvar:"
                      " glyph variation data offset %d out of range\n",
                      i ));
          blend->glyphoffsets[i] = limit;
        }
      }
    }
    else
    {
      FT_ULong  limit      = gvar_start + table_len;
      FT_ULong  max_offset = 0;


      for ( i = 0; i <= gvar_head.glyphCount; i++ )
      {
        blend->glyphoffsets[i] = offsetToData + FT_GET_USHORT() * 2;

        if ( max_offset <= blend->glyphoffsets[i] )
          max_offset = blend->glyphoffsets[i];
        else
        {
          FT_TRACE2(( "ft_var_load_gvar:"
                      " glyph variation data offset %d not monotonic\n",
                      i ));
          blend->glyphoffsets[i] = max_offset;
        }

        /* use `<', not `<=' */
        if ( limit < blend->glyphoffsets[i] )
        {
          FT_TRACE2(( "ft_var_load_gvar:"
                      " glyph variation data offset %d out of range\n",
                      i ));
          blend->glyphoffsets[i] = limit;
        }
      }
    }

    blend->gv_glyphcnt = gvar_head.glyphCount;

    FT_FRAME_EXIT();

    if ( gvar_head.globalCoordCount != 0 )
    {
      if ( FT_STREAM_SEEK( gvar_start + gvar_head.offsetToCoord ) ||
           FT_FRAME_ENTER( gvar_head.globalCoordCount *
                           gvar_head.axisCount * 2L )             )
      {
        FT_TRACE2(( "ft_var_load_gvar:"
                    " glyph variation shared tuples missing\n" ));
        goto Fail;
      }

      if ( FT_QNEW_ARRAY( blend->tuplecoords,
                          gvar_head.axisCount * gvar_head.globalCoordCount ) )
        goto Fail2;

      for ( i = 0; i < gvar_head.globalCoordCount; i++ )
      {
        FT_TRACE5(( "  [ " ));
        for ( j = 0; j < (FT_UInt)gvar_head.axisCount; j++ )
        {
          blend->tuplecoords[i * gvar_head.axisCount + j] =
            FT_fdot14ToFixed( FT_GET_SHORT() );
          FT_TRACE5(( "%.5f ",
            (double)blend->tuplecoords[i * gvar_head.axisCount + j] / 65536 ));
        }
        FT_TRACE5(( "]\n" ));
      }

      blend->tuplecount = gvar_head.globalCoordCount;

      FT_TRACE5(( "\n" ));

      FT_FRAME_EXIT();
    }

  Exit:
    return error;

  Fail2:
    FT_FRAME_EXIT();

  Fail:
    FT_FREE( blend->glyphoffsets );
    blend->gv_glyphcnt = 0;
    goto Exit;
  }


  /**************************************************************************
   *
   * @Function:
   *   ft_var_apply_tuple
   *
   * @Description:
   *   Figure out whether a given tuple (design) applies to the current
   *   blend, and if so, what is the scaling factor.
   *
   * @Input:
   *   blend ::
   *     The current blend of the font.
   *
   *   tupleIndex ::
   *     A flag saying whether this is an intermediate
   *     tuple or not.
   *
   *   tuple_coords ::
   *     The coordinates of the tuple in normalized axis
   *     units.
   *
   *   im_start_coords ::
   *     The initial coordinates where this tuple starts
   *     to apply (for intermediate coordinates).
   *
   *   im_end_coords ::
   *     The final coordinates after which this tuple no
   *     longer applies (for intermediate coordinates).
   *
   * @Return:
   *   An FT_Fixed value containing the scaling factor.
   */
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
      FT_Fixed  ncv = blend->normalizedcoords[i];


      FT_TRACE6(( "    axis %d coordinate %.5f:\n", i, (double)ncv / 65536 ));

      /* It's not clear why (for intermediate tuples) we don't need     */
      /* to check against start/end -- the documentation says we don't. */
      /* Similarly, it's unclear why we don't need to scale along the   */
      /* axis.                                                          */

      if ( tuple_coords[i] == ncv )
      {
        FT_TRACE6(( "      tuple coordinate %.5f fits perfectly\n",
                    (double)tuple_coords[i] / 65536 ));
        /* `apply' does not change */
        continue;
      }

      if ( tuple_coords[i] == 0 )
      {
        FT_TRACE6(( "      tuple coordinate is zero, ignore\n" ));
        continue;
      }

      if ( !( tupleIndex & GX_TI_INTERMEDIATE_TUPLE ) )
      {
        /* not an intermediate tuple */

        if ( ( tuple_coords[i] > ncv && ncv > 0 ) ||
             ( tuple_coords[i] < ncv && ncv < 0 ) )
        {
          FT_TRACE6(( "      tuple coordinate %.5f fits\n",
                      (double)tuple_coords[i] / 65536 ));
          apply = FT_MulDiv( apply, ncv, tuple_coords[i] );
        }
        else
        {
          FT_TRACE6(( "      tuple coordinate %.5f is exceeded, stop\n",
                      (double)tuple_coords[i] / 65536 ));
          apply = 0;
          break;
        }
      }
      else
      {
        /* intermediate tuple */

        if ( ncv <= im_start_coords[i] ||
             ncv >= im_end_coords[i]   )
        {
          FT_TRACE6(( "      intermediate tuple range ]%.5f;%.5f[ is exceeded,"
                      " stop\n",
                      (double)im_start_coords[i] / 65536,
                      (double)im_end_coords[i] / 65536 ));
          apply = 0;
          break;
        }

        FT_TRACE6(( "      intermediate tuple range ]%.5f;%.5f[ fits\n",
                    (double)im_start_coords[i] / 65536,
                    (double)im_end_coords[i] / 65536 ));
        if ( ncv < tuple_coords[i] )
          apply = FT_MulDiv( apply,
                             ncv - im_start_coords[i],
                             tuple_coords[i] - im_start_coords[i] );
        else /* ncv > tuple_coords[i] */
          apply = FT_MulDiv( apply,
                             im_end_coords[i] - ncv,
                             im_end_coords[i] - tuple_coords[i] );
      }
    }

    FT_TRACE6(( "    apply factor is %.5f\n", (double)apply / 65536 ));

    return apply;
  }


  /* convert from design coordinates to normalized coordinates */

  static void
  ft_var_to_normalized( TT_Face    face,
                        FT_UInt    num_coords,
                        FT_Fixed*  coords,
                        FT_Fixed*  normalized )
  {
    FT_Error   error  = FT_Err_Ok;
    FT_Memory  memory = face->root.memory;
    FT_UInt    i, j;

    GX_Blend        blend;
    FT_MM_Var*      mmvar;
    FT_Var_Axis*    a;
    GX_AVarSegment  av;

    FT_Fixed*  new_normalized = NULL;
    FT_Fixed*  old_normalized;


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

    a = mmvar->axis;
    for ( i = 0; i < num_coords; i++, a++ )
    {
      FT_Fixed  coord = coords[i];


      FT_TRACE5(( "    %d: %.5f\n", i, (double)coord / 65536 ));
      if ( coord > a->maximum || coord < a->minimum )
      {
        FT_TRACE1(( "ft_var_to_normalized: design coordinate %.5f\n",
                    (double)coord / 65536 ));
        FT_TRACE1(( "                      is out of range [%.5f;%.5f];"
                    " clamping\n",
                    (double)a->minimum / 65536,
                    (double)a->maximum / 65536 ));
      }

      if ( coord > a->def )
        normalized[i] = coord >= a->maximum ?  0x10000L :
                        FT_DivFix( SUB_LONG( coord, a->def ),
                                   SUB_LONG( a->maximum, a->def ) );
      else if ( coord < a->def )
        normalized[i] = coord <= a->minimum ? -0x10000L :
                        FT_DivFix( SUB_LONG( coord, a->def ),
                                   SUB_LONG( a->def, a->minimum ) );
      else
        normalized[i] = 0;
    }

    FT_TRACE5(( "\n" ));

    for ( ; i < mmvar->num_axis; i++ )
      normalized[i] = 0;

    if ( blend->avar_table )
    {
      GX_AVarTable  table = blend->avar_table;


      FT_TRACE5(( "normalized design coordinates"
                  " before applying `avar' data:\n" ));

      if ( table->avar_segment )
      {
        av = table->avar_segment;

        for ( i = 0; i < mmvar->num_axis; i++, av++ )
        {
          for ( j = 1; j < (FT_UInt)av->pairCount; j++ )
          {
            if ( normalized[i] < av->correspondence[j].fromCoord )
            {
              FT_TRACE5(( "  %.5f\n", (double)normalized[i] / 65536 ));

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

      if ( table->itemStore.varData )
      {
        if ( FT_QNEW_ARRAY( new_normalized, mmvar->num_axis ) )
          return;

        /* Install our half-normalized coordinates for the next */
        /* Item Variation Store to work with.                   */
        old_normalized                = face->blend->normalizedcoords;
        face->blend->normalizedcoords = normalized;

        for ( i = 0; i < mmvar->num_axis; i++ )
        {
          FT_Fixed  v          = normalized[i];
          FT_UInt   innerIndex = i;
          FT_UInt   outerIndex = 0;
          FT_Int    delta;


          if ( table->axisMap.innerIndex )
          {
            FT_UInt  idx = i;


            if ( idx >= table->axisMap.mapCount )
              idx = table->axisMap.mapCount - 1;

            outerIndex = table->axisMap.outerIndex[idx];
            innerIndex = table->axisMap.innerIndex[idx];
          }

          delta = tt_var_get_item_delta( FT_FACE( face ),
                                         &table->itemStore,
                                         outerIndex,
                                         innerIndex );

          /* Convert delta in F2DOT14 to 16.16 before adding. */
          v += MUL_INT( delta, 4 );

          /* Clamp value to range [-1, 1]. */
          v = v >=  0x10000L ?  0x10000 : v;
          v = v <= -0x10000L ? -0x10000 : v;

          new_normalized[i] = v;
        }

        for ( i = 0; i < mmvar->num_axis; i++ )
        {
          normalized[i] = new_normalized[i];
        }

        face->blend->normalizedcoords = old_normalized;

        FT_FREE( new_normalized );
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

    for ( i = 0; i < nc; i++ )
      design[i] = coords[i];

    for ( ; i < num_coords; i++ )
      design[i] = 0;

    if ( blend->avar_table && blend->avar_table->avar_segment )
    {
      GX_AVarSegment  av = blend->avar_table->avar_segment;


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

            FT_TRACE5(( "  %.5f\n", (double)design[i] / 65536 ));
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


  /**************************************************************************
   *
   * @Function:
   *   TT_Get_MM_Var
   *
   * @Description:
   *   Check that the font's `fvar' table is valid, parse it, and return
   *   those data.  It also loads (and parses) the `MVAR' table, if
   *   possible.
   *
   * @InOut:
   *   face ::
   *     The font face.
   *     TT_Get_MM_Var initializes the blend structure.
   *
   * @Output:
   *   master ::
   *     The `fvar' data (must be freed by caller).  Can be NULL,
   *     which makes this function simply load MM support.
   *
   * @Return:
   *   FreeType error code.  0 means success.
   */
  FT_LOCAL_DEF( FT_Error )
  TT_Get_MM_Var( FT_Face      face,    /* TT_Face */
                 FT_MM_Var*  *master )
  {
    TT_Face              ttface     = (TT_Face)face;
    FT_Stream            stream     = FT_FACE_STREAM( face );
    FT_Memory            memory     = FT_FACE_MEMORY( face );
    FT_ULong             table_len;
    FT_Error             error      = FT_Err_Ok;
    FT_ULong             fvar_start = 0;
    FT_UInt              i, j;
    FT_MM_Var*           mmvar = NULL;
    FT_Fixed*            next_coords;
    FT_Fixed*            nsc;
    FT_String*           next_name;
    FT_Var_Axis*         a;
    FT_Fixed*            c;
    FT_Var_Named_Style*  ns;
    GX_FVar_Head         fvar_head  = { 0, 0, 0, 0, 0, 0 };
    FT_Bool              usePsName  = 0;
    FT_UInt              num_instances;
    FT_UInt              num_axes;
    FT_UShort*           axis_flags;

    FT_Offset  mmvar_size;
    FT_Offset  axis_flags_size;
    FT_Offset  axis_size;
    FT_Offset  namedstyle_size;
    FT_Offset  next_coords_size;
    FT_Offset  next_name_size;

    FT_Bool  need_init;

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

    /* `num_instances` holds the number of all named instances including  */
    /* the default instance, which might be missing in the table of named */
    /* instances (in 'fvar').  This value is validated in `sfobjs.c` and  */
    /* may be reset to 0 if consistency checks fail.                      */
    num_instances = (FT_UInt)face->style_flags >> 16;

    /* read the font data and set up the internal representation */
    /* if not already done                                       */

    need_init = !ttface->blend;

    if ( need_init )
    {
      FT_TRACE2(( "FVAR " ));

      if ( FT_SET_ERROR( ttface->goto_table( ttface, TTAG_fvar,
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

      /* If `num_instances` is larger, synthetization of the default  */
      /* instance is required.  If `num_instances` is smaller,        */
      /* however, the value has been reset to 0 in `sfnt_init_face`   */
      /* (in `sfobjs.c`); in this case we have underallocated `mmvar` */
      /* structs.                                                     */
      if ( num_instances < fvar_head.instanceCount )
      {
        error = FT_THROW( Invalid_Table );
        goto Exit;
      }

      usePsName = FT_BOOL( fvar_head.instanceSize ==
                           6 + 4 * fvar_head.axisCount );

      FT_TRACE2(( "loaded\n" ));

      FT_TRACE5(( "%d variation ax%s\n",
                  fvar_head.axisCount,
                  fvar_head.axisCount == 1 ? "is" : "es" ));

      if ( FT_NEW( ttface->blend ) )
        goto Exit;

      num_axes                = fvar_head.axisCount;
      ttface->blend->num_axis = num_axes;
    }
    else
      num_axes = ttface->blend->num_axis;

    /* prepare storage area for MM data; this cannot overflow   */
    /* 32-bit arithmetic because of the size limits used in the */
    /* `fvar' table validity check in `sfnt_init_face'          */

    /* the various `*_size' variables, which we also use as     */
    /* offsets into the `mmvar' array, must be multiples of the */
    /* pointer size (except the last one); without such an      */
    /* alignment there might be runtime errors due to           */
    /* misaligned addresses                                     */
#undef  ALIGN_SIZE
#define ALIGN_SIZE( n ) \
          ( ( (n) + sizeof (void*) - 1 ) & ~( sizeof (void*) - 1 ) )

    mmvar_size       = ALIGN_SIZE( sizeof ( FT_MM_Var ) );
    axis_flags_size  = ALIGN_SIZE( num_axes *
                                   sizeof ( FT_UShort ) );
    axis_size        = ALIGN_SIZE( num_axes *
                                   sizeof ( FT_Var_Axis ) );
    namedstyle_size  = ALIGN_SIZE( num_instances *
                                   sizeof ( FT_Var_Named_Style ) );
    next_coords_size = ALIGN_SIZE( num_instances *
                                   num_axes *
                                   sizeof ( FT_Fixed ) );
    next_name_size   = num_axes * 5;

    if ( need_init )
    {
      ttface->blend->mmvar_len = mmvar_size       +
                                 axis_flags_size  +
                                 axis_size        +
                                 namedstyle_size  +
                                 next_coords_size +
                                 next_name_size;

      if ( FT_ALLOC( mmvar, ttface->blend->mmvar_len ) )
        goto Exit;
      ttface->blend->mmvar = mmvar;

      /* set up pointers and offsets into the `mmvar' array; */
      /* the data gets filled in later on                    */

      mmvar->num_axis =
        num_axes;
      mmvar->num_designs =
        ~0U;                   /* meaningless in this context; each glyph */
                               /* may have a different number of designs  */
                               /* (or tuples, as called by Apple)         */
      mmvar->num_namedstyles =
        num_instances;

      /* alas, no public field in `FT_Var_Axis' for axis flags */
      axis_flags =
        (FT_UShort*)( (char*)mmvar + mmvar_size );
      mmvar->axis =
        (FT_Var_Axis*)( (char*)axis_flags + axis_flags_size );
      mmvar->namedstyle =
        (FT_Var_Named_Style*)( (char*)mmvar->axis + axis_size );

      next_coords = (FT_Fixed*)( (char*)mmvar->namedstyle +
                                 namedstyle_size );
      for ( i = 0; i < num_instances; i++ )
      {
        mmvar->namedstyle[i].coords  = next_coords;
        next_coords                 += num_axes;
      }

      next_name = (FT_String*)( (char*)mmvar->namedstyle +
                                namedstyle_size + next_coords_size );
      for ( i = 0; i < num_axes; i++ )
      {
        mmvar->axis[i].name  = next_name;
        next_name           += 5;
      }

      /* now fill in the data */

      if ( FT_STREAM_SEEK( fvar_start + fvar_head.offsetToData ) )
        goto Exit;

      a = mmvar->axis;
      for ( i = 0; i < num_axes; i++ )
      {
        GX_FVar_Axis  axis_rec;

#ifdef FT_DEBUG_LEVEL_TRACE
        int  invalid = 0;
#endif


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

        *axis_flags = axis_rec.flags;

        if ( a->minimum > a->def ||
             a->def > a->maximum )
        {
          a->minimum = a->def;
          a->maximum = a->def;

#ifdef FT_DEBUG_LEVEL_TRACE
          invalid = 1;
#endif
        }

#ifdef FT_DEBUG_LEVEL_TRACE
        if ( i == 0 )
          FT_TRACE5(( "  idx   tag  "
                   /* "  XXX  `XXXX'" */
                      "    minimum     default     maximum   flags\n" ));
                   /* "  XXXX.XXXXX  XXXX.XXXXX  XXXX.XXXXX  0xXXXX" */

        FT_TRACE5(( "  %3d  `%s'"
                    "  %10.5f  %10.5f  %10.5f  0x%04X%s\n",
                    i,
                    a->name,
                    (double)a->minimum / 65536,
                    (double)a->def / 65536,
                    (double)a->maximum / 65536,
                    *axis_flags,
                    invalid ? " (invalid, disabled)" : "" ));
#endif

        a++;
        axis_flags++;
      }

      FT_TRACE5(( "\n" ));

      /* named instance coordinates are stored as design coordinates; */
      /* we have to convert them to normalized coordinates also       */
      if ( FT_NEW_ARRAY( ttface->blend->normalized_stylecoords,
                         num_axes * num_instances ) )
        goto Exit;

      if ( fvar_head.instanceCount && !ttface->blend->avar_loaded )
      {
        FT_ULong  offset = FT_STREAM_POS();


        ft_var_load_avar( ttface );

        if ( FT_STREAM_SEEK( offset ) )
          goto Exit;
      }

      FT_TRACE5(( "%d named instance%s\n",
                  fvar_head.instanceCount,
                  fvar_head.instanceCount == 1 ? "" : "s" ));

      ns  = mmvar->namedstyle;
      nsc = ttface->blend->normalized_stylecoords;
      for ( i = 0; i < fvar_head.instanceCount; i++, ns++ )
      {
        /* PostScript names add 2 bytes to the instance record size */
        if ( FT_FRAME_ENTER( ( usePsName ? 6L : 4L ) +
                             4L * num_axes ) )
          goto Exit;

        ns->strid       =    FT_GET_USHORT();
        (void) /* flags = */ FT_GET_USHORT();

        c = ns->coords;
        for ( j = 0; j < num_axes; j++, c++ )
          *c = FT_GET_LONG();

        /* valid psid values are 6, [256;32767], and 0xFFFF */
        if ( usePsName )
          ns->psid = FT_GET_USHORT();
        else
          ns->psid = 0xFFFF;

#ifdef FT_DEBUG_LEVEL_TRACE
        {
          SFNT_Service  sfnt = (SFNT_Service)ttface->sfnt;

          FT_String*  strname = NULL;
          FT_String*  psname  = NULL;

          FT_ULong  pos;


          pos = FT_STREAM_POS();

          if ( ns->strid != 0xFFFF )
          {
            (void)sfnt->get_name( ttface,
                                  (FT_UShort)ns->strid,
                                  &strname );
            if ( strname && !ft_strcmp( strname, ".notdef" ) )
              strname = NULL;
          }

          if ( ns->psid != 0xFFFF )
          {
            (void)sfnt->get_name( ttface,
                                  (FT_UShort)ns->psid,
                                  &psname );
            if ( psname && !ft_strcmp( psname, ".notdef" ) )
              psname = NULL;
          }

          (void)FT_STREAM_SEEK( pos );

          FT_TRACE5(( "  named instance %d (%s%s%s, %s%s%s)\n",
                      i,
                      strname ? "name: `" : "",
                      strname ? strname : "unnamed",
                      strname ? "'" : "",
                      psname ? "PS name: `" : "",
                      psname ? psname : "no PS name",
                      psname ? "'" : "" ));

          FT_FREE( strname );
          FT_FREE( psname );
        }
#endif /* FT_DEBUG_LEVEL_TRACE */

        ft_var_to_normalized( ttface, num_axes, ns->coords, nsc );
        nsc += num_axes;

        FT_FRAME_EXIT();
      }

      if ( num_instances != fvar_head.instanceCount )
      {
        SFNT_Service  sfnt = (SFNT_Service)ttface->sfnt;

        FT_Int   found, dummy1, dummy2;
        FT_UInt  strid = ~0U;


        /* The default instance is missing in array the    */
        /* of named instances; try to synthesize an entry. */
        /* If this fails, `default_named_instance` remains */
        /* at value zero, which doesn't do any harm.       */
        found = sfnt->get_name_id( ttface,
                                   TT_NAME_ID_TYPOGRAPHIC_SUBFAMILY,
                                   &dummy1,
                                   &dummy2 );
        if ( found )
          strid = TT_NAME_ID_TYPOGRAPHIC_SUBFAMILY;
        else
        {
          found = sfnt->get_name_id( ttface,
                                     TT_NAME_ID_FONT_SUBFAMILY,
                                     &dummy1,
                                     &dummy2 );
          if ( found )
            strid = TT_NAME_ID_FONT_SUBFAMILY;
        }

        if ( found )
        {
          found = sfnt->get_name_id( ttface,
                                     TT_NAME_ID_PS_NAME,
                                     &dummy1,
                                     &dummy2 );
          if ( found )
          {
            FT_TRACE5(( "TT_Get_MM_Var:"
                        " Adding default instance to named instances\n" ));

            /* named instance indices start with value 1 */
            ttface->var_default_named_instance = num_instances;

            ns = &mmvar->namedstyle[fvar_head.instanceCount];

            ns->strid = strid;
            ns->psid  = TT_NAME_ID_PS_NAME;

            a = mmvar->axis;
            c = ns->coords;
            for ( j = 0; j < num_axes; j++, a++, c++ )
              *c = a->def;
          }
        }
      }

      ft_var_load_mvar( ttface );
    }

    /* fill the output array if requested */

    if ( master )
    {
      FT_UInt  n;


      if ( FT_DUP( mmvar, ttface->blend->mmvar, ttface->blend->mmvar_len ) )
        goto Exit;

      axis_flags =
        (FT_UShort*)( (char*)mmvar + mmvar_size );
      mmvar->axis =
        (FT_Var_Axis*)( (char*)axis_flags + axis_flags_size );
      mmvar->namedstyle =
        (FT_Var_Named_Style*)( (char*)mmvar->axis+ axis_size );

      next_coords = (FT_Fixed*)( (char*)mmvar->namedstyle +
                                 namedstyle_size );
      for ( n = 0; n < mmvar->num_namedstyles; n++ )
      {
        mmvar->namedstyle[n].coords  = next_coords;
        next_coords                 += num_axes;
      }

      a         = mmvar->axis;
      next_name = (FT_String*)( (char*)mmvar->namedstyle +
                                namedstyle_size + next_coords_size );
      for ( n = 0; n < num_axes; n++ )
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
        else if ( a->tag == TTAG_ital )
          a->name = (char*)"Italic";

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
    FT_UInt     i;

    FT_Bool     all_design_coords = FALSE;

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
      if ( FT_SET_ERROR( TT_Get_MM_Var( FT_FACE( face ), NULL ) ) )
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

    FT_TRACE5(( "TT_Set_MM_Blend:\n" ));
    FT_TRACE5(( "  normalized design coordinates:\n" ));

    for ( i = 0; i < num_coords; i++ )
    {
      FT_TRACE5(( "    %.5f\n", (double)coords[i] / 65536 ));
      if ( coords[i] < -0x00010000L || coords[i] > 0x00010000L )
      {
        FT_TRACE1(( "TT_Set_MM_Blend: normalized design coordinate %.5f\n",
                    (double)coords[i] / 65536 ));
        FT_TRACE1(( "                 is out of range [-1;1]\n" ));
        error = FT_THROW( Invalid_Argument );
        goto Exit;
      }
    }

    FT_TRACE5(( "\n" ));

    if ( !face->is_cff2 && !blend->glyphoffsets )
    {
      /* While a missing 'gvar' table is acceptable, for example for */
      /* fonts that only vary metrics information or 'COLR' v1       */
      /* `PaintVar*` tables, an incorrect SFNT table offset or size  */
      /* for 'gvar', or an inconsistent 'gvar' table is not.         */
      error = ft_var_load_gvar( face );
      if ( error != FT_Err_Table_Missing && error != FT_Err_Ok )
        goto Exit;
      error = FT_Err_Ok;
    }

    if ( !blend->coords )
    {
      if ( FT_NEW_ARRAY( blend->coords, mmvar->num_axis ) )
        goto Exit;

      /* the first time we have to compute all design coordinates */
      all_design_coords = TRUE;
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
      FT_Bool    have_diff = 0;
      FT_UInt    j;
      FT_Fixed*  c;
      FT_Fixed*  n;


      manageCvt = mcvt_retain;

      for ( i = 0; i < num_coords; i++ )
      {
        if ( blend->normalizedcoords[i] != coords[i] )
        {
          manageCvt = mcvt_load;
          have_diff = 1;
          break;
        }
      }

      if ( !have_diff )
      {
        if ( FT_IS_NAMED_INSTANCE( FT_FACE( face ) ) )
        {
          FT_UInt  instance_index = (FT_UInt)face->root.face_index >> 16;


          c = blend->normalizedcoords + i;
          n = blend->normalized_stylecoords            +
              ( instance_index - 1 ) * mmvar->num_axis +
              i;

          for ( j = i; j < mmvar->num_axis; j++, n++, c++ )
            if ( *c != *n )
              have_diff = 1;
        }
        else
        {
          c = blend->normalizedcoords + i;
          for ( j = i; j < mmvar->num_axis; j++, c++ )
            if ( *c != 0 )
              have_diff = 1;
        }
      }

      /* return value -1 indicates `no change' */
      if ( !have_diff )
      {
        face->doblend = TRUE;

        return -1;
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
    if ( coords )
      FT_MEM_COPY( blend->normalizedcoords,
                   coords,
                   num_coords * sizeof ( FT_Fixed ) );

    if ( set_design_coords )
      ft_var_to_design( face,
                        all_design_coords ? blend->num_axis : num_coords,
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

  Exit:
    return error;
  }


  /**************************************************************************
   *
   * @Function:
   *   TT_Set_MM_Blend
   *
   * @Description:
   *   Set the blend (normalized) coordinates for this instance of the
   *   font.  Check that the `gvar' table is reasonable and does some
   *   initial preparation.
   *
   * @InOut:
   *   face ::
   *     The font.
   *     Initialize the blend structure with `gvar' data.
   *
   * @Input:
   *   num_coords ::
   *     The number of available coordinates.  If it is
   *     larger than the number of axes, ignore the excess
   *     values.  If it is smaller than the number of axes,
   *     use the default value (0) for the remaining axes.
   *
   *   coords ::
   *     An array of `num_coords', each between [-1,1].
   *
   * @Return:
   *   FreeType error code.  0 means success, -1 means success and unchanged
   *   axis values.
   */
  FT_LOCAL_DEF( FT_Error )
  TT_Set_MM_Blend( FT_Face    face,       /* TT_Face */
                   FT_UInt    num_coords,
                   FT_Fixed*  coords )
  {
    return tt_set_mm_blend( (TT_Face)face, num_coords, coords, 1 );
  }


  /**************************************************************************
   *
   * @Function:
   *   TT_Get_MM_Blend
   *
   * @Description:
   *   Get the blend (normalized) coordinates for this instance of the
   *   font.
   *
   * @InOut:
   *   face ::
   *     The font.
   *     Initialize the blend structure with `gvar' data.
   *
   * @Input:
   *   num_coords ::
   *     The number of available coordinates.  If it is
   *     larger than the number of axes, set the excess
   *     values to 0.
   *
   *   coords ::
   *     An array of `num_coords', each between [-1,1].
   *
   * @Return:
   *   FreeType error code.  0 means success, -1 means success and unchanged
   *   axis values.
   */
  FT_LOCAL_DEF( FT_Error )
  TT_Get_MM_Blend( FT_Face    face,       /* TT_Face */
                   FT_UInt    num_coords,
                   FT_Fixed*  coords )
  {
    TT_Face  ttface = (TT_Face)face;

    FT_Error  error = FT_Err_Ok;
    GX_Blend  blend;
    FT_UInt   i, nc;


    if ( !ttface->blend )
    {
      if ( FT_SET_ERROR( TT_Get_MM_Var( face, NULL ) ) )
        return error;
    }

    blend = ttface->blend;

    if ( !blend->coords )
    {
      /* select default instance coordinates */
      /* if no instance is selected yet      */
      if ( FT_SET_ERROR( tt_set_mm_blend( ttface, 0, NULL, 1 ) ) )
        return error;
    }

    nc = num_coords;
    if ( num_coords > blend->num_axis )
    {
      FT_TRACE2(( "TT_Get_MM_Blend:"
                  " only using first %d of %d coordinates\n",
                  blend->num_axis, num_coords ));
      nc = blend->num_axis;
    }

    if ( ttface->doblend )
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


  /**************************************************************************
   *
   * @Function:
   *   TT_Set_Var_Design
   *
   * @Description:
   *   Set the coordinates for the instance, measured in the user
   *   coordinate system.  Parse the `avar' table (if present) to convert
   *   from user to normalized coordinates.
   *
   * @InOut:
   *   face ::
   *     The font face.
   *     Initialize the blend struct with `gvar' data.
   *
   * @Input:
   *   num_coords ::
   *     The number of available coordinates.  If it is
   *     larger than the number of axes, ignore the excess
   *     values.  If it is smaller than the number of axes,
   *     use the default values for the remaining axes.
   *
   *   coords ::
   *     A coordinate array with `num_coords' elements.
   *
   * @Return:
   *   FreeType error code.  0 means success.
   */
  FT_LOCAL_DEF( FT_Error )
  TT_Set_Var_Design( FT_Face    face,       /* TT_Face */
                     FT_UInt    num_coords,
                     FT_Fixed*  coords )
  {
    TT_Face     ttface = (TT_Face)face;
    FT_Error    error  = FT_Err_Ok;
    GX_Blend    blend;
    FT_MM_Var*  mmvar;
    FT_UInt     i;
    FT_Memory   memory = FT_FACE_MEMORY( face );

    FT_Fixed*  c;
    FT_Fixed*  n;
    FT_Fixed*  normalized = NULL;

    FT_Bool  have_diff = 0;


    if ( !ttface->blend )
    {
      if ( FT_SET_ERROR( TT_Get_MM_Var( face, NULL ) ) )
        goto Exit;
    }

    blend = ttface->blend;
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

    c = blend->coords;
    n = coords;
    for ( i = 0; i < num_coords; i++, n++, c++ )
    {
      if ( *c != *n )
      {
        *c        = *n;
        have_diff = 1;
      }
    }

    if ( FT_IS_NAMED_INSTANCE( face ) )
    {
      FT_UInt              instance_index;
      FT_Var_Named_Style*  named_style;


      instance_index = (FT_UInt)face->face_index >> 16;
      named_style    = mmvar->namedstyle + instance_index - 1;

      n = named_style->coords + num_coords;
      for ( ; i < mmvar->num_axis; i++, n++, c++ )
      {
        if ( *c != *n )
        {
          *c        = *n;
          have_diff = 1;
        }
      }
    }
    else
    {
      FT_Var_Axis*  a;


      a = mmvar->axis + num_coords;
      for ( ; i < mmvar->num_axis; i++, a++, c++ )
      {
        if ( *c != a->def )
        {
          *c        = a->def;
          have_diff = 1;
        }
      }
    }

    /* return value -1 indicates `no change';                      */
    /* we can exit early if `normalizedcoords' is already computed */
    if ( blend->normalizedcoords && !have_diff )
      return -1;

    if ( FT_NEW_ARRAY( normalized, mmvar->num_axis ) )
      goto Exit;

    if ( !ttface->blend->avar_loaded )
      ft_var_load_avar( ttface );

    FT_TRACE5(( "TT_Set_Var_Design:\n" ));
    FT_TRACE5(( "  normalized design coordinates:\n" ));
    ft_var_to_normalized( ttface, num_coords, blend->coords, normalized );

    error = tt_set_mm_blend( ttface, mmvar->num_axis, normalized, 0 );
    if ( error )
      goto Exit;

  Exit:
    FT_FREE( normalized );
    return error;
  }


  /**************************************************************************
   *
   * @Function:
   *   TT_Get_Var_Design
   *
   * @Description:
   *   Get the design coordinates of the currently selected interpolated
   *   font.
   *
   * @Input:
   *   face ::
   *     A handle to the source face.
   *
   *   num_coords ::
   *     The number of design coordinates to retrieve.  If it
   *     is larger than the number of axes, set the excess
   *     values to~0.
   *
   * @Output:
   *   coords ::
   *     The design coordinates array.
   *
   * @Return:
   *   FreeType error code.  0~means success.
   */
  FT_LOCAL_DEF( FT_Error )
  TT_Get_Var_Design( FT_Face    face,       /* TT_Face */
                     FT_UInt    num_coords,
                     FT_Fixed*  coords )
  {
    TT_Face   ttface = (TT_Face)face;
    FT_Error  error  = FT_Err_Ok;
    GX_Blend  blend;
    FT_UInt   i, nc;


    if ( !ttface->blend )
    {
      if ( FT_SET_ERROR( TT_Get_MM_Var( face, NULL ) ) )
        return error;
    }

    blend = ttface->blend;

    if ( !blend->coords )
    {
      /* select default instance coordinates */
      /* if no instance is selected yet      */
      if ( FT_SET_ERROR( tt_set_mm_blend( ttface, 0, NULL, 1 ) ) )
        return error;
    }

    nc = num_coords;
    if ( num_coords > blend->num_axis )
    {
      FT_TRACE2(( "TT_Get_Var_Design:"
                  " only using first %d of %d coordinates\n",
                  blend->num_axis, num_coords ));
      nc = blend->num_axis;
    }

    if ( ttface->doblend )
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


  /**************************************************************************
   *
   * @Function:
   *   TT_Set_Named_Instance
   *
   * @Description:
   *   Set the given named instance, also resetting any further
   *   variation.
   *
   * @Input:
   *   face ::
   *     A handle to the source face.
   *
   *   instance_index ::
   *     The instance index, starting with value 1.
   *     Value 0 indicates to not use an instance.
   *
   * @Return:
   *   FreeType error code.  0~means success, -1 means success and unchanged
   *   axis values.
   */
  FT_LOCAL_DEF( FT_Error )
  TT_Set_Named_Instance( FT_Face  face,            /* TT_Face */
                         FT_UInt  instance_index )
  {
    TT_Face     ttface = (TT_Face)face;
    FT_Error    error;
    GX_Blend    blend;
    FT_MM_Var*  mmvar;

    FT_Memory  memory = FT_FACE_MEMORY( face );

    FT_UInt  num_instances;


    if ( !ttface->blend )
    {
      if ( FT_SET_ERROR( TT_Get_MM_Var( face, NULL ) ) )
        goto Exit;
    }

    blend = ttface->blend;
    mmvar = blend->mmvar;

    num_instances = (FT_UInt)face->style_flags >> 16;

    /* `instance_index' starts with value 1, thus `>' */
    if ( instance_index > num_instances )
    {
      error = FT_ERR( Invalid_Argument );
      goto Exit;
    }

    if ( instance_index > 0 )
    {
      SFNT_Service  sfnt = (SFNT_Service)ttface->sfnt;

      FT_Var_Named_Style*  named_style;
      FT_String*           style_name;


      named_style = mmvar->namedstyle + instance_index - 1;

      error = sfnt->get_name( ttface,
                              (FT_UShort)named_style->strid,
                              &style_name );
      if ( error )
        goto Exit;

      /* set (or replace) style name */
      FT_FREE( face->style_name );
      face->style_name = style_name;

      /* finally, select the named instance */
      error = TT_Set_Var_Design( face,
                                 mmvar->num_axis,
                                 named_style->coords );
    }
    else
    {
      /* restore non-VF style name */
      FT_FREE( face->style_name );
      if ( FT_STRDUP( face->style_name, ttface->non_var_style_name ) )
        goto Exit;
      error = TT_Set_Var_Design( face, 0, NULL );
    }

  Exit:
    return error;
  }


  /**************************************************************************
   *
   * @Function:
   *   TT_Get_Default_Named_Instance
   *
   * @Description:
   *   Get the default named instance.
   *
   * @Input:
   *   face ::
   *     A handle to the source face.
   *
   * @Output:
   *   instance_index ::
   *     The default named instance index.
   *
   * @Return:
   *   FreeType error code.  0~means success.
   */
  FT_LOCAL_DEF( FT_Error )
  TT_Get_Default_Named_Instance( FT_Face   face,
                                 FT_UInt  *instance_index )
  {
    TT_Face   ttface = (TT_Face)face;
    FT_Error  error  = FT_Err_Ok;


    if ( !ttface->blend )
    {
      if ( FT_SET_ERROR( TT_Get_MM_Var( face, NULL ) ) )
        goto Exit;
    }

    *instance_index = ttface->var_default_named_instance;

  Exit:
    return error;
  }


  /* This function triggers (lazy) recomputation of the `postscript_name` */
  /* field in `TT_Face`.                                                  */

  FT_LOCAL_DEF( void )
  tt_construct_ps_name( FT_Face  face )
  {
    TT_Face    ttface = (TT_Face)face;
    FT_Memory  memory = FT_FACE_MEMORY( face );


    FT_FREE( ttface->postscript_name );
  }


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                     GX VAR PARSING ROUTINES                   *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/


#ifdef TT_CONFIG_OPTION_BYTECODE_INTERPRETER

  static FT_Error
  tt_cvt_ready_iterator( FT_ListNode  node,
                         void*        user )
  {
    TT_Size  size = (TT_Size)node->data;

    FT_UNUSED( user );


    size->cvt_ready = -1;

    return FT_Err_Ok;
  }

#endif /* TT_CONFIG_OPTION_BYTECODE_INTERPRETER */



  /**************************************************************************
   *
   * @Function:
   *   tt_face_vary_cvt
   *
   * @Description:
   *   Modify the loaded cvt table according to the `cvar' table and the
   *   font's blend.
   *
   * @InOut:
   *   face ::
   *     A handle to the target face object.
   *
   * @Input:
   *   stream ::
   *     A handle to the input stream.
   *
   * @Return:
   *   FreeType error code.  0 means success.
   *
   *   Most errors are ignored.  It is perfectly valid not to have a
   *   `cvar' table even if there is a `gvar' and `fvar' table.
   */
  FT_LOCAL_DEF( FT_Error )
  tt_face_vary_cvt( TT_Face    face,
                    FT_Stream  stream )
  {
#ifdef TT_CONFIG_OPTION_BYTECODE_INTERPRETER

    FT_Error   error;
    FT_Memory  memory = stream->memory;

    FT_Face  root = &face->root;

    FT_ULong  table_start;
    FT_ULong  table_len;

    FT_UInt   tupleCount;
    FT_ULong  offsetToData;

    FT_ULong  here;
    FT_UInt   i, j;

    FT_Fixed*  peak_coords = NULL;
    FT_Fixed*  tuple_coords;
    FT_Fixed*  im_start_coords;
    FT_Fixed*  im_end_coords;

    GX_Blend  blend = face->blend;

    FT_UInt  point_count;
    FT_UInt  spoint_count = 0;

    FT_UShort*  sharedpoints = NULL;
    FT_UShort*  localpoints  = NULL;
    FT_UShort*  points;

    FT_Fixed*  deltas     = NULL;
    FT_Fixed*  cvt_deltas = NULL;


    FT_TRACE2(( "CVAR " ));

    if ( !blend )
    {
      FT_TRACE2(( "\n" ));
      FT_TRACE2(( "tt_face_vary_cvt: no blend specified\n" ));

      return FT_Err_Ok;
    }

    if ( !face->cvt )
    {
      FT_TRACE2(( "\n" ));
      FT_TRACE2(( "tt_face_vary_cvt: no `cvt ' table\n" ));

      return FT_Err_Ok;
    }

    error = face->goto_table( face, TTAG_cvar, stream, &table_len );
    if ( error )
    {
      FT_TRACE2(( "is missing\n" ));

      return FT_Err_Ok;
    }

    if ( FT_FRAME_ENTER( table_len ) )
      return FT_Err_Ok;

    table_start = FT_Stream_FTell( stream );
    if ( FT_GET_LONG() != 0x00010000L )
    {
      FT_TRACE2(( "bad table version\n" ));

      error = FT_Err_Ok;
      goto FExit;
    }

    FT_TRACE2(( "loaded\n" ));

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

    if ( tupleCount & GX_TC_TUPLES_SHARE_POINT_NUMBERS )
    {
      here = FT_Stream_FTell( stream );

      FT_Stream_SeekSet( stream, offsetToData );

      sharedpoints = ft_var_readpackedpoints( stream, &spoint_count );

      offsetToData = FT_Stream_FTell( stream );

      FT_Stream_SeekSet( stream, here );
    }

    FT_TRACE5(( "cvar: there %s %d tuple%s:\n",
                ( tupleCount & GX_TC_TUPLE_COUNT_MASK ) == 1 ? "is" : "are",
                tupleCount & GX_TC_TUPLE_COUNT_MASK,
                ( tupleCount & GX_TC_TUPLE_COUNT_MASK ) == 1 ? "" : "s" ));

    if ( FT_QNEW_ARRAY( peak_coords, 3 * blend->num_axis ) ||
         FT_NEW_ARRAY( cvt_deltas, face->cvt_size )        )
      goto Exit;

    im_start_coords = peak_coords + blend->num_axis;
    im_end_coords = im_start_coords + blend->num_axis;

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
          peak_coords[j] = FT_fdot14ToFixed( FT_GET_SHORT() );
        tuple_coords = peak_coords;
      }
      else if ( ( tupleIndex & GX_TI_TUPLE_INDEX_MASK ) < blend->tuplecount )
        tuple_coords = blend->tuplecoords +
            ( tupleIndex & GX_TI_TUPLE_INDEX_MASK ) * blend->num_axis;
      else
      {
        FT_TRACE2(( "tt_face_vary_cvt:"
                    " invalid tuple index\n" ));

        error = FT_THROW( Invalid_Table );
        goto Exit;
      }

      if ( tupleIndex & GX_TI_INTERMEDIATE_TUPLE )
      {
        for ( j = 0; j < blend->num_axis; j++ )
          im_start_coords[j] = FT_fdot14ToFixed( FT_GET_SHORT() );
        for ( j = 0; j < blend->num_axis; j++ )
          im_end_coords[j] = FT_fdot14ToFixed( FT_GET_SHORT() );
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
        localpoints = ft_var_readpackedpoints( stream, &point_count );
        points      = localpoints;
      }
      else
      {
        localpoints = NULL;
        points      = sharedpoints;
        point_count = spoint_count;
      }

      deltas = ft_var_readpackeddeltas( stream,
                                        point_count == 0 ? face->cvt_size
                                                         : point_count );

      if ( !points || !deltas )
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
          FT_Fixed  old_cvt_delta;


          old_cvt_delta = cvt_deltas[j];
          cvt_deltas[j] = old_cvt_delta + FT_MulFix( deltas[j], apply );

#ifdef FT_DEBUG_LEVEL_TRACE
          if ( old_cvt_delta != cvt_deltas[j] )
          {
            FT_TRACE7(( "      %d: %f -> %f\n",
                        j,
                        (double)( FT_fdot6ToFixed( face->cvt[j] ) +
                                    old_cvt_delta ) / 65536,
                        (double)( FT_fdot6ToFixed( face->cvt[j] ) +
                                    cvt_deltas[j] ) / 65536 ));
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
          int       pindex;
          FT_Fixed  old_cvt_delta;


          pindex = points[j];
          if ( (FT_ULong)pindex >= face->cvt_size )
            continue;

          old_cvt_delta      = cvt_deltas[pindex];
          cvt_deltas[pindex] = old_cvt_delta + FT_MulFix( deltas[j], apply );

#ifdef FT_DEBUG_LEVEL_TRACE
          if ( old_cvt_delta != cvt_deltas[pindex] )
          {
            FT_TRACE7(( "      %d: %f -> %f\n",
                        pindex,
                        (double)( FT_fdot6ToFixed( face->cvt[pindex] ) +
                                    old_cvt_delta ) / 65536,
                        (double)( FT_fdot6ToFixed( face->cvt[pindex] ) +
                                    cvt_deltas[pindex] ) / 65536 ));
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

    for ( i = 0; i < face->cvt_size; i++ )
      face->cvt[i] += FT_fixedToFdot6( cvt_deltas[i] );

    /* Iterate over all `FT_Size` objects and set `cvt_ready` to -1 */
    /* to trigger rescaling of all CVT values.                      */
    FT_List_Iterate( &root->sizes_list,
                     tt_cvt_ready_iterator,
                     NULL );

  Exit:
    if ( sharedpoints != ALL_POINTS )
      FT_FREE( sharedpoints );
    FT_FREE( cvt_deltas );
    FT_FREE( peak_coords );

  FExit:
    FT_FRAME_EXIT();

    return error;

#else /* !TT_CONFIG_OPTION_BYTECODE_INTERPRETER */

    FT_UNUSED( face );
    FT_UNUSED( stream );

    return FT_Err_Ok;

#endif /* !TT_CONFIG_OPTION_BYTECODE_INTERPRETER */

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

  /* modeled after `af_iup_interp', `_iup_worker_interpolate', and   */
  /* `Ins_IUP' with spec differences in handling ill-defined cases.  */
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


  /**************************************************************************
   *
   * @Function:
   *   TT_Vary_Apply_Glyph_Deltas
   *
   * @Description:
   *   Apply the appropriate deltas to the current glyph.
   *
   * @InOut:
   *   loader ::
   *     A handle to the loader object.
   *
   *   outline ::
   *     The outline to change, with appended phantom points.
   *
   * @Output:
   *   unrounded ::
   *     An array with `n_points' elements that is filled with unrounded
   *     point coordinates (in 26.6 format).
   *
   * @Return:
   *   FreeType error code.  0 means success.
   */
  FT_LOCAL_DEF( FT_Error )
  TT_Vary_Apply_Glyph_Deltas( TT_Loader    loader,
                              FT_Outline*  outline,
                              FT_Vector*   unrounded )
  {
    FT_Error   error;
    TT_Face    face        = loader->face;
    FT_Stream  stream      = face->root.stream;
    FT_Memory  memory      = stream->memory;
    FT_UInt    glyph_index = loader->glyph_index;
    FT_UInt    n_points    = (FT_UInt)outline->n_points + 4;

    FT_Vector*  points_org = NULL;  /* coordinates in 16.16 format */
    FT_Vector*  points_out = NULL;  /* coordinates in 16.16 format */
    FT_Bool*    has_delta  = NULL;

    FT_ULong  glyph_start;

    FT_UInt   tupleCount;
    FT_ULong  offsetToData;
    FT_ULong  dataSize;

    FT_ULong  here;
    FT_UInt   i, j;

    FT_Fixed*  peak_coords = NULL;
    FT_Fixed*  tuple_coords;
    FT_Fixed*  im_start_coords;
    FT_Fixed*  im_end_coords;

    GX_Blend  blend = face->blend;

    FT_UInt  point_count;
    FT_UInt  spoint_count = 0;

    FT_UShort*  sharedpoints = NULL;
    FT_UShort*  localpoints  = NULL;
    FT_UShort*  points;

    FT_Fixed*  deltas_x       = NULL;
    FT_Fixed*  deltas_y       = NULL;
    FT_Fixed*  point_deltas_x = NULL;
    FT_Fixed*  point_deltas_y = NULL;


    if ( !face->doblend || !blend )
      return FT_THROW( Invalid_Argument );

    for ( i = 0; i < n_points; i++ )
    {
      unrounded[i].x = INT_TO_F26DOT6( outline->points[i].x );
      unrounded[i].y = INT_TO_F26DOT6( outline->points[i].y );
    }

    if ( glyph_index >= blend->gv_glyphcnt      ||
         blend->glyphoffsets[glyph_index] ==
           blend->glyphoffsets[glyph_index + 1] )
    {
      FT_TRACE2(( "TT_Vary_Apply_Glyph_Deltas:"
                  " no variation data for glyph %d\n", glyph_index ));
      return FT_Err_Ok;
    }

    dataSize = blend->glyphoffsets[glyph_index + 1] -
                 blend->glyphoffsets[glyph_index];

    if ( FT_STREAM_SEEK( blend->glyphoffsets[glyph_index] ) ||
         FT_FRAME_ENTER( dataSize )                         )
      return error;

    glyph_start = FT_Stream_FTell( stream );

    /* each set of glyph variation data is formatted similarly to `cvar' */

    tupleCount   = FT_GET_USHORT();
    offsetToData = FT_GET_USHORT();

    /* rough sanity test */
    if ( offsetToData > dataSize                                ||
         ( tupleCount & GX_TC_TUPLE_COUNT_MASK ) * 4 > dataSize )
    {
      FT_TRACE2(( "TT_Vary_Apply_Glyph_Deltas:"
                  " invalid glyph variation array header\n" ));

      error = FT_THROW( Invalid_Table );
      goto FExit;
    }

    offsetToData += glyph_start;

    if ( tupleCount & GX_TC_TUPLES_SHARE_POINT_NUMBERS )
    {
      here = FT_Stream_FTell( stream );

      FT_Stream_SeekSet( stream, offsetToData );

      sharedpoints = ft_var_readpackedpoints( stream, &spoint_count );

      offsetToData = FT_Stream_FTell( stream );

      FT_Stream_SeekSet( stream, here );
    }

    FT_TRACE5(( "gvar: there %s %d tuple%s:\n",
                ( tupleCount & GX_TC_TUPLE_COUNT_MASK ) == 1 ? "is" : "are",
                tupleCount & GX_TC_TUPLE_COUNT_MASK,
                ( tupleCount & GX_TC_TUPLE_COUNT_MASK ) == 1 ? "" : "s" ));

    if ( FT_QNEW_ARRAY( peak_coords, 3 * blend->num_axis ) ||
         FT_NEW_ARRAY( point_deltas_x, 2 * n_points )      ||
         FT_QNEW_ARRAY( points_org, n_points )             ||
         FT_QNEW_ARRAY( points_out, n_points )             ||
         FT_QNEW_ARRAY( has_delta, n_points )              )
      goto Exit;

    im_start_coords = peak_coords + blend->num_axis;
    im_end_coords   = im_start_coords + blend->num_axis;
    point_deltas_y  = point_deltas_x + n_points;

    for ( j = 0; j < n_points; j++ )
    {
      points_org[j].x = FT_intToFixed( outline->points[j].x );
      points_org[j].y = FT_intToFixed( outline->points[j].y );
    }

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
          peak_coords[j] = FT_fdot14ToFixed( FT_GET_SHORT() );
        tuple_coords = peak_coords;
      }
      else if ( ( tupleIndex & GX_TI_TUPLE_INDEX_MASK ) < blend->tuplecount )
        tuple_coords = blend->tuplecoords +
            ( tupleIndex & GX_TI_TUPLE_INDEX_MASK ) * blend->num_axis;
      else
      {
        FT_TRACE2(( "TT_Vary_Apply_Glyph_Deltas:"
                    " invalid tuple index\n" ));

        error = FT_THROW( Invalid_Table );
        goto Exit;
      }

      if ( tupleIndex & GX_TI_INTERMEDIATE_TUPLE )
      {
        for ( j = 0; j < blend->num_axis; j++ )
          im_start_coords[j] = FT_fdot14ToFixed( FT_GET_SHORT() );
        for ( j = 0; j < blend->num_axis; j++ )
          im_end_coords[j] = FT_fdot14ToFixed( FT_GET_SHORT() );
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
          FT_Fixed  old_point_delta_x = point_deltas_x[j];
          FT_Fixed  old_point_delta_y = point_deltas_y[j];

          FT_Fixed  point_delta_x = FT_MulFix( deltas_x[j], apply );
          FT_Fixed  point_delta_y = FT_MulFix( deltas_y[j], apply );


          point_deltas_x[j] = old_point_delta_x + point_delta_x;
          point_deltas_y[j] = old_point_delta_y + point_delta_y;

#ifdef FT_DEBUG_LEVEL_TRACE
          if ( point_delta_x || point_delta_y )
          {
            FT_TRACE7(( "      %d: (%f, %f) -> (%f, %f)\n",
                        j,
                        (double)( FT_intToFixed( outline->points[j].x ) +
                                    old_point_delta_x ) / 65536,
                        (double)( FT_intToFixed( outline->points[j].y ) +
                                    old_point_delta_y ) / 65536,
                        (double)( FT_intToFixed( outline->points[j].x ) +
                                    point_deltas_x[j] ) / 65536,
                        (double)( FT_intToFixed( outline->points[j].y ) +
                                    point_deltas_y[j] ) / 65536 ));
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
          FT_Fixed  old_point_delta_x = point_deltas_x[j];
          FT_Fixed  old_point_delta_y = point_deltas_y[j];

          FT_Pos  point_delta_x = points_out[j].x - points_org[j].x;
          FT_Pos  point_delta_y = points_out[j].y - points_org[j].y;


          point_deltas_x[j] = old_point_delta_x + point_delta_x;
          point_deltas_y[j] = old_point_delta_y + point_delta_y;

#ifdef FT_DEBUG_LEVEL_TRACE
          if ( point_delta_x || point_delta_y )
          {
            FT_TRACE7(( "      %d: (%f, %f) -> (%f, %f)\n",
                        j,
                        (double)( FT_intToFixed( outline->points[j].x ) +
                                    old_point_delta_x ) / 65536,
                        (double)( FT_intToFixed( outline->points[j].y ) +
                                    old_point_delta_y ) / 65536,
                        (double)( FT_intToFixed( outline->points[j].x ) +
                                    point_deltas_x[j] ) / 65536,
                        (double)( FT_intToFixed( outline->points[j].y ) +
                                    point_deltas_y[j] ) / 65536 ));
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

    /* To avoid double adjustment of advance width or height, */
    /* do not move phantom points if there is HVAR or VVAR    */
    /* support, respectively.                                 */
    if ( face->variation_support & TT_FACE_FLAG_VAR_HADVANCE )
    {
      point_deltas_x[n_points - 4] = 0;
      point_deltas_y[n_points - 4] = 0;
      point_deltas_x[n_points - 3] = 0;
      point_deltas_y[n_points - 3] = 0;
    }
    if ( face->variation_support & TT_FACE_FLAG_VAR_VADVANCE )
    {
      point_deltas_x[n_points - 2] = 0;
      point_deltas_y[n_points - 2] = 0;
      point_deltas_x[n_points - 1] = 0;
      point_deltas_y[n_points - 1] = 0;
    }

    for ( i = 0; i < n_points; i++ )
    {
      unrounded[i].x += FT_fixedToFdot6( point_deltas_x[i] );
      unrounded[i].y += FT_fixedToFdot6( point_deltas_y[i] );

      outline->points[i].x += FT_fixedToInt( point_deltas_x[i] );
      outline->points[i].y += FT_fixedToInt( point_deltas_y[i] );
    }

    /* To avoid double adjustment of advance width or height, */
    /* adjust phantom points only if there is no HVAR or VVAR */
    /* support, respectively.                                 */
    if ( !( face->variation_support & TT_FACE_FLAG_VAR_HADVANCE ) )
    {
      loader->pp1      = outline->points[n_points - 4];
      loader->pp2      = outline->points[n_points - 3];
      loader->linear   = FT_PIX_ROUND( unrounded[n_points - 3].x -
                                       unrounded[n_points - 4].x ) / 64;
    }
    if ( !( face->variation_support & TT_FACE_FLAG_VAR_VADVANCE ) )
    {
      loader->pp3      = outline->points[n_points - 2];
      loader->pp4      = outline->points[n_points - 1];
      loader->vadvance = FT_PIX_ROUND( unrounded[n_points - 1].y -
                                       unrounded[n_points - 2].y ) / 64;
    }

  Exit:
    if ( sharedpoints != ALL_POINTS )
      FT_FREE( sharedpoints );
    FT_FREE( points_org );
    FT_FREE( points_out );
    FT_FREE( has_delta );
    FT_FREE( peak_coords );
    FT_FREE( point_deltas_x );

  FExit:
    FT_FRAME_EXIT();

    return error;
  }


  /**************************************************************************
   *
   * @Function:
   *   tt_get_var_blend
   *
   * @Description:
   *   An extended internal version of `TT_Get_MM_Blend' that returns
   *   pointers instead of copying data, without any initialization of
   *   the MM machinery in case it isn't loaded yet.
   */
  FT_LOCAL_DEF( FT_Error )
  tt_get_var_blend( FT_Face      face,             /* TT_Face */
                    FT_UInt     *num_coords,
                    FT_Fixed*   *coords,
                    FT_Fixed*   *normalizedcoords,
                    FT_MM_Var*  *mm_var )
  {
    TT_Face  ttface = (TT_Face)face;


    if ( ttface->blend )
    {
      if ( num_coords )
        *num_coords       = ttface->blend->num_axis;
      if ( coords )
        *coords           = ttface->blend->coords;
      if ( normalizedcoords )
        *normalizedcoords = ttface->blend->normalizedcoords;
      if ( mm_var )
        *mm_var           = ttface->blend->mmvar;
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


  FT_LOCAL_DEF( void )
  tt_var_done_item_variation_store( FT_Face          face,
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


  FT_LOCAL_DEF( void )
  tt_var_done_delta_set_index_map( FT_Face            face,
                                   GX_DeltaSetIdxMap  deltaSetIdxMap )
  {
    FT_Memory  memory = FT_FACE_MEMORY( face );


    FT_FREE( deltaSetIdxMap->innerIndex );
    FT_FREE( deltaSetIdxMap->outerIndex );
  }


  /**************************************************************************
   *
   * @Function:
   *   tt_done_blend
   *
   * @Description:
   *   Free the blend internal data structure.
   */
  FT_LOCAL_DEF( void )
  tt_done_blend( FT_Face  face )
  {
    TT_Face    ttface = (TT_Face)face;
    FT_Memory  memory = FT_FACE_MEMORY( face );
    GX_Blend   blend  = ttface->blend;


    if ( blend )
    {
      FT_UInt  i, num_axes;


      /* blend->num_axis might not be set up yet */
      num_axes = blend->mmvar->num_axis;

      FT_FREE( blend->coords );
      FT_FREE( blend->normalizedcoords );
      FT_FREE( blend->normalized_stylecoords );
      FT_FREE( blend->mmvar );

      if ( blend->avar_table )
      {
        if ( blend->avar_table->avar_segment )
        {
          for ( i = 0; i < num_axes; i++ )
            FT_FREE( blend->avar_table->avar_segment[i].correspondence );
          FT_FREE( blend->avar_table->avar_segment );
        }

        tt_var_done_item_variation_store( face,
                                          &blend->avar_table->itemStore );

        tt_var_done_delta_set_index_map( face,
                                         &blend->avar_table->axisMap );

        FT_FREE( blend->avar_table );
      }

      if ( blend->hvar_table )
      {
        tt_var_done_item_variation_store( face,
                                          &blend->hvar_table->itemStore );

        tt_var_done_delta_set_index_map( face,
                                         &blend->hvar_table->widthMap );
        FT_FREE( blend->hvar_table );
      }

      if ( blend->vvar_table )
      {
        tt_var_done_item_variation_store( face,
                                          &blend->vvar_table->itemStore );

        tt_var_done_delta_set_index_map( face,
                                         &blend->vvar_table->widthMap );
        FT_FREE( blend->vvar_table );
      }

      if ( blend->mvar_table )
      {
        tt_var_done_item_variation_store( face,
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
  typedef int  tt_gxvar_dummy_;

#endif /* !TT_CONFIG_OPTION_GX_VAR_SUPPORT */


/* END */
