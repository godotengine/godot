/****************************************************************************
 *
 * gxvkern.c
 *
 *   TrueTypeGX/AAT kern table validation (body).
 *
 * Copyright (C) 2004-2019 by
 * suzuki toshiya, Masatake YAMATO, Red Hat K.K.,
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */

/****************************************************************************
 *
 * gxvalid is derived from both gxlayout module and otvalid module.
 * Development of gxlayout is supported by the Information-technology
 * Promotion Agency(IPA), Japan.
 *
 */


#include "gxvalid.h"
#include "gxvcommn.h"

#include FT_SFNT_NAMES_H
#include FT_SERVICE_GX_VALIDATE_H


  /**************************************************************************
   *
   * The macro FT_COMPONENT is used in trace mode.  It is an implicit
   * parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log
   * messages during execution.
   */
#undef  FT_COMPONENT
#define FT_COMPONENT  gxvkern


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                      Data and Types                           *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  typedef enum  GXV_kern_Version_
  {
    KERN_VERSION_CLASSIC = 0x0000,
    KERN_VERSION_NEW     = 0x0001

  } GXV_kern_Version;


  typedef enum GXV_kern_Dialect_
  {
    KERN_DIALECT_UNKNOWN = 0,
    KERN_DIALECT_MS      = FT_VALIDATE_MS,
    KERN_DIALECT_APPLE   = FT_VALIDATE_APPLE,
    KERN_DIALECT_ANY     = FT_VALIDATE_CKERN

  } GXV_kern_Dialect;


  typedef struct  GXV_kern_DataRec_
  {
    GXV_kern_Version  version;
    void             *subtable_data;
    GXV_kern_Dialect  dialect_request;

  } GXV_kern_DataRec, *GXV_kern_Data;


#define GXV_KERN_DATA( field )  GXV_TABLE_DATA( kern, field )

#define KERN_IS_CLASSIC( gxvalid )                               \
          ( KERN_VERSION_CLASSIC == GXV_KERN_DATA( version ) )
#define KERN_IS_NEW( gxvalid )                                   \
          ( KERN_VERSION_NEW     == GXV_KERN_DATA( version ) )

#define KERN_DIALECT( gxvalid )              \
          GXV_KERN_DATA( dialect_request )
#define KERN_ALLOWS_MS( gxvalid )                       \
          ( KERN_DIALECT( gxvalid ) & KERN_DIALECT_MS )
#define KERN_ALLOWS_APPLE( gxvalid )                       \
          ( KERN_DIALECT( gxvalid ) & KERN_DIALECT_APPLE )

#define GXV_KERN_HEADER_SIZE           ( KERN_IS_NEW( gxvalid ) ? 8 : 4 )
#define GXV_KERN_SUBTABLE_HEADER_SIZE  ( KERN_IS_NEW( gxvalid ) ? 8 : 6 )


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                      SUBTABLE VALIDATORS                      *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/


  /* ============================= format 0 ============================== */

  static void
  gxv_kern_subtable_fmt0_pairs_validate( FT_Bytes       table,
                                         FT_Bytes       limit,
                                         FT_UShort      nPairs,
                                         GXV_Validator  gxvalid )
  {
    FT_Bytes   p = table;
    FT_UShort  i;

    FT_UShort  last_gid_left  = 0;
    FT_UShort  last_gid_right = 0;

    FT_UNUSED( limit );


    GXV_NAME_ENTER( "kern format 0 pairs" );

    for ( i = 0; i < nPairs; i++ )
    {
      FT_UShort  gid_left;
      FT_UShort  gid_right;
#ifdef GXV_LOAD_UNUSED_VARS
      FT_Short   kernValue;
#endif


      /* left */
      gid_left  = FT_NEXT_USHORT( p );
      gxv_glyphid_validate( gid_left, gxvalid );

      /* right */
      gid_right = FT_NEXT_USHORT( p );
      gxv_glyphid_validate( gid_right, gxvalid );

      /* Pairs of left and right GIDs must be unique and sorted. */
      GXV_TRACE(( "left gid = %u, right gid = %u\n", gid_left, gid_right ));
      if ( gid_left == last_gid_left )
      {
        if ( last_gid_right < gid_right )
          last_gid_right = gid_right;
        else
          FT_INVALID_DATA;
      }
      else if ( last_gid_left < gid_left )
      {
        last_gid_left  = gid_left;
        last_gid_right = gid_right;
      }
      else
        FT_INVALID_DATA;

      /* skip the kern value */
#ifdef GXV_LOAD_UNUSED_VARS
      kernValue = FT_NEXT_SHORT( p );
#else
      p += 2;
#endif
    }

    GXV_EXIT;
  }

  static void
  gxv_kern_subtable_fmt0_validate( FT_Bytes       table,
                                   FT_Bytes       limit,
                                   GXV_Validator  gxvalid )
  {
    FT_Bytes   p = table + GXV_KERN_SUBTABLE_HEADER_SIZE;

    FT_UShort  nPairs;
    FT_UShort  unitSize;


    GXV_NAME_ENTER( "kern subtable format 0" );

    unitSize = 2 + 2 + 2;
    nPairs   = 0;

    /* nPairs, searchRange, entrySelector, rangeShift */
    GXV_LIMIT_CHECK( 2 + 2 + 2 + 2 );
    gxv_BinSrchHeader_validate( p, limit, &unitSize, &nPairs, gxvalid );
    p += 2 + 2 + 2 + 2;

    gxv_kern_subtable_fmt0_pairs_validate( p, limit, nPairs, gxvalid );

    GXV_EXIT;
  }


  /* ============================= format 1 ============================== */


  typedef struct  GXV_kern_fmt1_StateOptRec_
  {
    FT_UShort  valueTable;
    FT_UShort  valueTable_length;

  } GXV_kern_fmt1_StateOptRec, *GXV_kern_fmt1_StateOptRecData;


  static void
  gxv_kern_subtable_fmt1_valueTable_load( FT_Bytes       table,
                                          FT_Bytes       limit,
                                          GXV_Validator  gxvalid )
  {
    FT_Bytes                       p = table;
    GXV_kern_fmt1_StateOptRecData  optdata =
      (GXV_kern_fmt1_StateOptRecData)gxvalid->statetable.optdata;


    GXV_LIMIT_CHECK( 2 );
    optdata->valueTable = FT_NEXT_USHORT( p );
  }


  /*
   * passed tables_size covers whole StateTable, including kern fmt1 header
   */
  static void
  gxv_kern_subtable_fmt1_subtable_setup( FT_UShort      table_size,
                                         FT_UShort      classTable,
                                         FT_UShort      stateArray,
                                         FT_UShort      entryTable,
                                         FT_UShort*     classTable_length_p,
                                         FT_UShort*     stateArray_length_p,
                                         FT_UShort*     entryTable_length_p,
                                         GXV_Validator  gxvalid )
  {
    FT_UShort  o[4];
    FT_UShort  *l[4];
    FT_UShort  buff[5];

    GXV_kern_fmt1_StateOptRecData  optdata =
      (GXV_kern_fmt1_StateOptRecData)gxvalid->statetable.optdata;


    o[0] = classTable;
    o[1] = stateArray;
    o[2] = entryTable;
    o[3] = optdata->valueTable;
    l[0] = classTable_length_p;
    l[1] = stateArray_length_p;
    l[2] = entryTable_length_p;
    l[3] = &(optdata->valueTable_length);

    gxv_set_length_by_ushort_offset( o, l, buff, 4, table_size, gxvalid );
  }


  /*
   * passed table & limit are of whole StateTable, not including subtables
   */
  static void
  gxv_kern_subtable_fmt1_entry_validate(
    FT_Byte                         state,
    FT_UShort                       flags,
    GXV_StateTable_GlyphOffsetCPtr  glyphOffset_p,
    FT_Bytes                        table,
    FT_Bytes                        limit,
    GXV_Validator                   gxvalid )
  {
#ifdef GXV_LOAD_UNUSED_VARS
    FT_UShort  push;
    FT_UShort  dontAdvance;
#endif
    FT_UShort  valueOffset;
#ifdef GXV_LOAD_UNUSED_VARS
    FT_UShort  kernAction;
    FT_UShort  kernValue;
#endif

    FT_UNUSED( state );
    FT_UNUSED( glyphOffset_p );


#ifdef GXV_LOAD_UNUSED_VARS
    push        = (FT_UShort)( ( flags >> 15 ) & 1      );
    dontAdvance = (FT_UShort)( ( flags >> 14 ) & 1      );
#endif
    valueOffset = (FT_UShort)(   flags         & 0x3FFF );

    {
      GXV_kern_fmt1_StateOptRecData  vt_rec =
        (GXV_kern_fmt1_StateOptRecData)gxvalid->statetable.optdata;
      FT_Bytes  p;


      if ( valueOffset < vt_rec->valueTable )
        FT_INVALID_OFFSET;

      p     = table + valueOffset;
      limit = table + vt_rec->valueTable + vt_rec->valueTable_length;

      GXV_LIMIT_CHECK( 2 + 2 );
#ifdef GXV_LOAD_UNUSED_VARS
      kernAction = FT_NEXT_USHORT( p );
      kernValue  = FT_NEXT_USHORT( p );
#endif
    }
  }


  static void
  gxv_kern_subtable_fmt1_validate( FT_Bytes       table,
                                   FT_Bytes       limit,
                                   GXV_Validator  gxvalid )
  {
    FT_Bytes                   p = table;
    GXV_kern_fmt1_StateOptRec  vt_rec;


    GXV_NAME_ENTER( "kern subtable format 1" );

    gxvalid->statetable.optdata =
      &vt_rec;
    gxvalid->statetable.optdata_load_func =
      gxv_kern_subtable_fmt1_valueTable_load;
    gxvalid->statetable.subtable_setup_func =
      gxv_kern_subtable_fmt1_subtable_setup;
    gxvalid->statetable.entry_glyphoffset_fmt =
      GXV_GLYPHOFFSET_NONE;
    gxvalid->statetable.entry_validate_func =
      gxv_kern_subtable_fmt1_entry_validate;

    gxv_StateTable_validate( p, limit, gxvalid );

    GXV_EXIT;
  }


  /* ================ Data for Class-Based Subtables 2, 3 ================ */

  typedef enum  GXV_kern_ClassSpec_
  {
    GXV_KERN_CLS_L = 0,
    GXV_KERN_CLS_R

  } GXV_kern_ClassSpec;


  /* ============================= format 2 ============================== */

  /* ---------------------- format 2 specific data ----------------------- */

  typedef struct  GXV_kern_subtable_fmt2_DataRec_
  {
    FT_UShort         rowWidth;
    FT_UShort         array;
    FT_UShort         offset_min[2];
    FT_UShort         offset_max[2];
    const FT_String*  class_tag[2];
    GXV_odtect_Range  odtect;

  } GXV_kern_subtable_fmt2_DataRec, *GXV_kern_subtable_fmt2_Data;


#define GXV_KERN_FMT2_DATA( field )                         \
        ( ( (GXV_kern_subtable_fmt2_DataRec *)              \
              ( GXV_KERN_DATA( subtable_data ) ) )->field )


  /* -------------------------- utility functions ----------------------- */

  static void
  gxv_kern_subtable_fmt2_clstbl_validate( FT_Bytes            table,
                                          FT_Bytes            limit,
                                          GXV_kern_ClassSpec  spec,
                                          GXV_Validator       gxvalid )
  {
    const FT_String*  tag    = GXV_KERN_FMT2_DATA( class_tag[spec] );
    GXV_odtect_Range  odtect = GXV_KERN_FMT2_DATA( odtect );

    FT_Bytes   p = table;
    FT_UShort  firstGlyph;
    FT_UShort  nGlyphs;


    GXV_NAME_ENTER( "kern format 2 classTable" );

    GXV_LIMIT_CHECK( 2 + 2 );
    firstGlyph = FT_NEXT_USHORT( p );
    nGlyphs    = FT_NEXT_USHORT( p );
    GXV_TRACE(( " %s firstGlyph=%d, nGlyphs=%d\n",
                tag, firstGlyph, nGlyphs ));

    gxv_glyphid_validate( firstGlyph, gxvalid );
    gxv_glyphid_validate( (FT_UShort)( firstGlyph + nGlyphs - 1 ), gxvalid );

    gxv_array_getlimits_ushort( p, p + ( 2 * nGlyphs ),
                                &( GXV_KERN_FMT2_DATA( offset_min[spec] ) ),
                                &( GXV_KERN_FMT2_DATA( offset_max[spec] ) ),
                                gxvalid );

    gxv_odtect_add_range( table, 2 * nGlyphs, tag, odtect );

    GXV_EXIT;
  }


  static void
  gxv_kern_subtable_fmt2_validate( FT_Bytes       table,
                                   FT_Bytes       limit,
                                   GXV_Validator  gxvalid )
  {
    GXV_ODTECT( 3, odtect );
    GXV_kern_subtable_fmt2_DataRec  fmt2_rec =
      { 0, 0, { 0, 0 }, { 0, 0 }, { "leftClass", "rightClass" }, NULL };

    FT_Bytes   p = table + GXV_KERN_SUBTABLE_HEADER_SIZE;
    FT_UShort  leftOffsetTable;
    FT_UShort  rightOffsetTable;


    GXV_NAME_ENTER( "kern subtable format 2" );

    GXV_ODTECT_INIT( odtect );
    fmt2_rec.odtect = odtect;
    GXV_KERN_DATA( subtable_data ) = &fmt2_rec;

    GXV_LIMIT_CHECK( 2 + 2 + 2 + 2 );
    GXV_KERN_FMT2_DATA( rowWidth ) = FT_NEXT_USHORT( p );
    leftOffsetTable                = FT_NEXT_USHORT( p );
    rightOffsetTable               = FT_NEXT_USHORT( p );
    GXV_KERN_FMT2_DATA( array )    = FT_NEXT_USHORT( p );

    GXV_TRACE(( "rowWidth = %d\n", GXV_KERN_FMT2_DATA( rowWidth ) ));


    GXV_LIMIT_CHECK( leftOffsetTable );
    GXV_LIMIT_CHECK( rightOffsetTable );
    GXV_LIMIT_CHECK( GXV_KERN_FMT2_DATA( array ) );

    gxv_kern_subtable_fmt2_clstbl_validate( table + leftOffsetTable, limit,
                                            GXV_KERN_CLS_L, gxvalid );

    gxv_kern_subtable_fmt2_clstbl_validate( table + rightOffsetTable, limit,
                                            GXV_KERN_CLS_R, gxvalid );

    if ( GXV_KERN_FMT2_DATA( offset_min[GXV_KERN_CLS_L] ) +
           GXV_KERN_FMT2_DATA( offset_min[GXV_KERN_CLS_R] )
         < GXV_KERN_FMT2_DATA( array )                      )
      FT_INVALID_OFFSET;

    gxv_odtect_add_range( table + GXV_KERN_FMT2_DATA( array ),
                          GXV_KERN_FMT2_DATA( offset_max[GXV_KERN_CLS_L] )
                            + GXV_KERN_FMT2_DATA( offset_max[GXV_KERN_CLS_R] )
                            - GXV_KERN_FMT2_DATA( array ),
                          "array", odtect );

    gxv_odtect_validate( odtect, gxvalid );

    GXV_EXIT;
  }


  /* ============================= format 3 ============================== */

  static void
  gxv_kern_subtable_fmt3_validate( FT_Bytes       table,
                                   FT_Bytes       limit,
                                   GXV_Validator  gxvalid )
  {
    FT_Bytes   p = table + GXV_KERN_SUBTABLE_HEADER_SIZE;
    FT_UShort  glyphCount;
    FT_Byte    kernValueCount;
    FT_Byte    leftClassCount;
    FT_Byte    rightClassCount;
    FT_Byte    flags;


    GXV_NAME_ENTER( "kern subtable format 3" );

    GXV_LIMIT_CHECK( 2 + 1 + 1 + 1 + 1 );
    glyphCount      = FT_NEXT_USHORT( p );
    kernValueCount  = FT_NEXT_BYTE( p );
    leftClassCount  = FT_NEXT_BYTE( p );
    rightClassCount = FT_NEXT_BYTE( p );
    flags           = FT_NEXT_BYTE( p );

    if ( gxvalid->face->num_glyphs != glyphCount )
    {
      GXV_TRACE(( "maxGID=%d, but glyphCount=%d\n",
                  gxvalid->face->num_glyphs, glyphCount ));
      GXV_SET_ERR_IF_PARANOID( FT_INVALID_GLYPH_ID );
    }

    if ( flags != 0 )
      GXV_TRACE(( "kern subtable fmt3 has nonzero value"
                  " (%d) in unused flag\n", flags ));
    /*
     * just skip kernValue[kernValueCount]
     */
    GXV_LIMIT_CHECK( 2 * kernValueCount );
    p += 2 * kernValueCount;

    /*
     * check leftClass[gid] < leftClassCount
     */
    {
      FT_Byte  min, max;


      GXV_LIMIT_CHECK( glyphCount );
      gxv_array_getlimits_byte( p, p + glyphCount, &min, &max, gxvalid );
      p += gxvalid->subtable_length;

      if ( leftClassCount < max )
        FT_INVALID_DATA;
    }

    /*
     * check rightClass[gid] < rightClassCount
     */
    {
      FT_Byte  min, max;


      GXV_LIMIT_CHECK( glyphCount );
      gxv_array_getlimits_byte( p, p + glyphCount, &min, &max, gxvalid );
      p += gxvalid->subtable_length;

      if ( rightClassCount < max )
        FT_INVALID_DATA;
    }

    /*
     * check kernIndex[i, j] < kernValueCount
     */
    {
      FT_UShort  i, j;


      for ( i = 0; i < leftClassCount; i++ )
      {
        for ( j = 0; j < rightClassCount; j++ )
        {
          GXV_LIMIT_CHECK( 1 );
          if ( kernValueCount < FT_NEXT_BYTE( p ) )
            FT_INVALID_OFFSET;
        }
      }
    }

    gxvalid->subtable_length = (FT_ULong)( p - table );

    GXV_EXIT;
  }


  static FT_Bool
  gxv_kern_coverage_new_apple_validate( FT_UShort      coverage,
                                        FT_UShort*     format,
                                        GXV_Validator  gxvalid )
  {
    /* new Apple-dialect */
#ifdef GXV_LOAD_TRACE_VARS
    FT_Bool  kernVertical;
    FT_Bool  kernCrossStream;
    FT_Bool  kernVariation;
#endif

    FT_UNUSED( gxvalid );


    /* reserved bits = 0 */
    if ( coverage & 0x1FFC )
      return FALSE;

#ifdef GXV_LOAD_TRACE_VARS
    kernVertical    = FT_BOOL( ( coverage >> 15 ) & 1 );
    kernCrossStream = FT_BOOL( ( coverage >> 14 ) & 1 );
    kernVariation   = FT_BOOL( ( coverage >> 13 ) & 1 );
#endif

    *format = (FT_UShort)( coverage & 0x0003 );

    GXV_TRACE(( "new Apple-dialect: "
                "horizontal=%d, cross-stream=%d, variation=%d, format=%d\n",
                 !kernVertical, kernCrossStream, kernVariation, *format ));

    GXV_TRACE(( "kerning values in Apple format subtable are ignored\n" ));

    return TRUE;
  }


  static FT_Bool
  gxv_kern_coverage_classic_apple_validate( FT_UShort      coverage,
                                            FT_UShort*     format,
                                            GXV_Validator  gxvalid )
  {
    /* classic Apple-dialect */
#ifdef GXV_LOAD_TRACE_VARS
    FT_Bool  horizontal;
    FT_Bool  cross_stream;
#endif


    /* check expected flags, but don't check if MS-dialect is impossible */
    if ( !( coverage & 0xFD00 ) && KERN_ALLOWS_MS( gxvalid ) )
      return FALSE;

    /* reserved bits = 0 */
    if ( coverage & 0x02FC )
      return FALSE;

#ifdef GXV_LOAD_TRACE_VARS
    horizontal   = FT_BOOL( ( coverage >> 15 ) & 1 );
    cross_stream = FT_BOOL( ( coverage >> 13 ) & 1 );
#endif

    *format = (FT_UShort)( coverage & 0x0003 );

    GXV_TRACE(( "classic Apple-dialect: "
                "horizontal=%d, cross-stream=%d, format=%d\n",
                 horizontal, cross_stream, *format ));

    /* format 1 requires GX State Machine, too new for classic */
    if ( *format == 1 )
      return FALSE;

    GXV_TRACE(( "kerning values in Apple format subtable are ignored\n" ));

    return TRUE;
  }


  static FT_Bool
  gxv_kern_coverage_classic_microsoft_validate( FT_UShort      coverage,
                                                FT_UShort*     format,
                                                GXV_Validator  gxvalid )
  {
    /* classic Microsoft-dialect */
#ifdef GXV_LOAD_TRACE_VARS
    FT_Bool  horizontal;
    FT_Bool  minimum;
    FT_Bool  cross_stream;
    FT_Bool  override;
#endif

    FT_UNUSED( gxvalid );


    /* reserved bits = 0 */
    if ( coverage & 0xFDF0 )
      return FALSE;

#ifdef GXV_LOAD_TRACE_VARS
    horizontal   = FT_BOOL(   coverage        & 1 );
    minimum      = FT_BOOL( ( coverage >> 1 ) & 1 );
    cross_stream = FT_BOOL( ( coverage >> 2 ) & 1 );
    override     = FT_BOOL( ( coverage >> 3 ) & 1 );
#endif

    *format = (FT_UShort)( ( coverage >> 8 ) & 0x0003 );

    GXV_TRACE(( "classic Microsoft-dialect: "
                "horizontal=%d, minimum=%d, cross-stream=%d, "
                "override=%d, format=%d\n",
                horizontal, minimum, cross_stream, override, *format ));

    if ( *format == 2 )
      GXV_TRACE((
        "kerning values in Microsoft format 2 subtable are ignored\n" ));

    return TRUE;
  }


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                            MAIN                               *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  static GXV_kern_Dialect
  gxv_kern_coverage_validate( FT_UShort      coverage,
                              FT_UShort*     format,
                              GXV_Validator  gxvalid )
  {
    GXV_kern_Dialect  result = KERN_DIALECT_UNKNOWN;


    GXV_NAME_ENTER( "validating coverage" );

    GXV_TRACE(( "interpret coverage 0x%04x by Apple style\n", coverage ));

    if ( KERN_IS_NEW( gxvalid ) )
    {
      if ( gxv_kern_coverage_new_apple_validate( coverage,
                                                 format,
                                                 gxvalid ) )
      {
        result = KERN_DIALECT_APPLE;
        goto Exit;
      }
    }

    if ( KERN_IS_CLASSIC( gxvalid ) && KERN_ALLOWS_APPLE( gxvalid ) )
    {
      if ( gxv_kern_coverage_classic_apple_validate( coverage,
                                                     format,
                                                     gxvalid ) )
      {
        result = KERN_DIALECT_APPLE;
        goto Exit;
      }
    }

    if ( KERN_IS_CLASSIC( gxvalid ) && KERN_ALLOWS_MS( gxvalid ) )
    {
      if ( gxv_kern_coverage_classic_microsoft_validate( coverage,
                                                         format,
                                                         gxvalid ) )
      {
        result = KERN_DIALECT_MS;
        goto Exit;
      }
    }

    GXV_TRACE(( "cannot interpret coverage, broken kern subtable\n" ));

  Exit:
    GXV_EXIT;
    return result;
  }


  static void
  gxv_kern_subtable_validate( FT_Bytes       table,
                              FT_Bytes       limit,
                              GXV_Validator  gxvalid )
  {
    FT_Bytes   p = table;
#ifdef GXV_LOAD_TRACE_VARS
    FT_UShort  version = 0;    /* MS only: subtable version, unused */
#endif
    FT_ULong   length;         /* MS: 16bit, Apple: 32bit*/
    FT_UShort  coverage;
#ifdef GXV_LOAD_TRACE_VARS
    FT_UShort  tupleIndex = 0; /* Apple only */
#endif
    FT_UShort  u16[2];
    FT_UShort  format = 255;   /* subtable format */


    GXV_NAME_ENTER( "kern subtable" );

    GXV_LIMIT_CHECK( 2 + 2 + 2 );
    u16[0]   = FT_NEXT_USHORT( p ); /* Apple: length_hi MS: version */
    u16[1]   = FT_NEXT_USHORT( p ); /* Apple: length_lo MS: length */
    coverage = FT_NEXT_USHORT( p );

    switch ( gxv_kern_coverage_validate( coverage, &format, gxvalid ) )
    {
    case KERN_DIALECT_MS:
#ifdef GXV_LOAD_TRACE_VARS
      version    = u16[0];
#endif
      length     = u16[1];
#ifdef GXV_LOAD_TRACE_VARS
      tupleIndex = 0;
#endif
      GXV_TRACE(( "Subtable version = %d\n", version ));
      GXV_TRACE(( "Subtable length = %d\n", length ));
      break;

    case KERN_DIALECT_APPLE:
#ifdef GXV_LOAD_TRACE_VARS
      version    = 0;
#endif
      length     = ( (FT_ULong)u16[0] << 16 ) + u16[1];
#ifdef GXV_LOAD_TRACE_VARS
      tupleIndex = 0;
#endif
      GXV_TRACE(( "Subtable length = %d\n", length ));

      if ( KERN_IS_NEW( gxvalid ) )
      {
        GXV_LIMIT_CHECK( 2 );
#ifdef GXV_LOAD_TRACE_VARS
        tupleIndex = FT_NEXT_USHORT( p );
#else
        p += 2;
#endif
        GXV_TRACE(( "Subtable tupleIndex = %d\n", tupleIndex ));
      }
      break;

    default:
      length = u16[1];
      GXV_TRACE(( "cannot detect subtable dialect, "
                  "just skip %d byte\n", length ));
      goto Exit;
    }

    /* formats 1, 2, 3 require the position of the start of this subtable */
    if ( format == 0 )
      gxv_kern_subtable_fmt0_validate( table, table + length, gxvalid );
    else if ( format == 1 )
      gxv_kern_subtable_fmt1_validate( table, table + length, gxvalid );
    else if ( format == 2 )
      gxv_kern_subtable_fmt2_validate( table, table + length, gxvalid );
    else if ( format == 3 )
      gxv_kern_subtable_fmt3_validate( table, table + length, gxvalid );
    else
      FT_INVALID_DATA;

  Exit:
    gxvalid->subtable_length = length;
    GXV_EXIT;
  }


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                         kern TABLE                            *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  static void
  gxv_kern_validate_generic( FT_Bytes          table,
                             FT_Face           face,
                             FT_Bool           classic_only,
                             GXV_kern_Dialect  dialect_request,
                             FT_Validator      ftvalid )
  {
    GXV_ValidatorRec   gxvalidrec;
    GXV_Validator      gxvalid = &gxvalidrec;

    GXV_kern_DataRec   kernrec;
    GXV_kern_Data      kern = &kernrec;

    FT_Bytes           p     = table;
    FT_Bytes           limit = 0;

    FT_ULong           nTables = 0;
    FT_UInt            i;


    gxvalid->root       = ftvalid;
    gxvalid->table_data = kern;
    gxvalid->face       = face;

    FT_TRACE3(( "validating `kern' table\n" ));
    GXV_INIT;
    KERN_DIALECT( gxvalid ) = dialect_request;

    GXV_LIMIT_CHECK( 2 );
    GXV_KERN_DATA( version ) = (GXV_kern_Version)FT_NEXT_USHORT( p );
    GXV_TRACE(( "version 0x%04x (higher 16bit)\n",
                GXV_KERN_DATA( version ) ));

    if ( 0x0001 < GXV_KERN_DATA( version ) )
      FT_INVALID_FORMAT;
    else if ( KERN_IS_CLASSIC( gxvalid ) )
    {
      GXV_LIMIT_CHECK( 2 );
      nTables = FT_NEXT_USHORT( p );
    }
    else if ( KERN_IS_NEW( gxvalid ) )
    {
      if ( classic_only )
        FT_INVALID_FORMAT;

      if ( 0x0000 != FT_NEXT_USHORT( p ) )
        FT_INVALID_FORMAT;

      GXV_LIMIT_CHECK( 4 );
      nTables = FT_NEXT_ULONG( p );
    }

    for ( i = 0; i < nTables; i++ )
    {
      GXV_TRACE(( "validating subtable %d/%d\n", i, nTables ));
      /* p should be 32bit-aligned? */
      gxv_kern_subtable_validate( p, 0, gxvalid );
      p += gxvalid->subtable_length;
    }

    FT_TRACE4(( "\n" ));
  }


  FT_LOCAL_DEF( void )
  gxv_kern_validate( FT_Bytes      table,
                     FT_Face       face,
                     FT_Validator  ftvalid )
  {
    gxv_kern_validate_generic( table, face, 0, KERN_DIALECT_ANY, ftvalid );
  }


  FT_LOCAL_DEF( void )
  gxv_kern_validate_classic( FT_Bytes      table,
                             FT_Face       face,
                             FT_Int        dialect_flags,
                             FT_Validator  ftvalid )
  {
    GXV_kern_Dialect  dialect_request;


    dialect_request = (GXV_kern_Dialect)dialect_flags;
    gxv_kern_validate_generic( table, face, 1, dialect_request, ftvalid );
  }


/* END */
