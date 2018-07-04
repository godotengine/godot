/***************************************************************************/
/*                                                                         */
/*  gxvtrak.c                                                              */
/*                                                                         */
/*    TrueTypeGX/AAT trak table validation (body).                         */
/*                                                                         */
/*  Copyright 2004-2018 by                                                 */
/*  suzuki toshiya, Masatake YAMATO, Red Hat K.K.,                         */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/

/***************************************************************************/
/*                                                                         */
/* gxvalid is derived from both gxlayout module and otvalid module.        */
/* Development of gxlayout is supported by the Information-technology      */
/* Promotion Agency(IPA), Japan.                                           */
/*                                                                         */
/***************************************************************************/


#include "gxvalid.h"
#include "gxvcommn.h"


  /*************************************************************************/
  /*                                                                       */
  /* The macro FT_COMPONENT is used in trace mode.  It is an implicit      */
  /* parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log  */
  /* messages during execution.                                            */
  /*                                                                       */
#undef  FT_COMPONENT
#define FT_COMPONENT  trace_gxvtrak


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                      Data and Types                           *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

    /*
     * referred track table format specification:
     * https://developer.apple.com/fonts/TrueType-Reference-Manual/RM06/Chap6trak.html
     * last update was 1996.
     * ----------------------------------------------
     * [MINIMUM HEADER]: GXV_TRAK_SIZE_MIN
     * version          (fixed:  32bit) = 0x00010000
     * format           (uint16: 16bit) = 0 is only defined (1996)
     * horizOffset      (uint16: 16bit)
     * vertOffset       (uint16: 16bit)
     * reserved         (uint16: 16bit) = 0
     * ----------------------------------------------
     * [VARIABLE BODY]:
     * horizData
     *   header         ( 2 + 2 + 4
     *   trackTable       + nTracks * ( 4 + 2 + 2 )
     *   sizeTable        + nSizes * 4 )
     * ----------------------------------------------
     * vertData
     *   header         ( 2 + 2 + 4
     *   trackTable       + nTracks * ( 4 + 2 + 2 )
     *   sizeTable        + nSizes * 4 )
     * ----------------------------------------------
     */
  typedef struct  GXV_trak_DataRec_
  {
    FT_UShort  trackValueOffset_min;
    FT_UShort  trackValueOffset_max;

  } GXV_trak_DataRec, *GXV_trak_Data;


#define GXV_TRAK_DATA( FIELD )  GXV_TABLE_DATA( trak, FIELD )


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                      UTILITY FUNCTIONS                        *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  static void
  gxv_trak_trackTable_validate( FT_Bytes       table,
                                FT_Bytes       limit,
                                FT_UShort      nTracks,
                                GXV_Validator  gxvalid )
  {
    FT_Bytes  p = table;

    FT_Fixed   track, t;
    FT_UShort  nameIndex;
    FT_UShort  offset;
    FT_UShort  i, j;


    GXV_NAME_ENTER( "trackTable" );

    GXV_TRAK_DATA( trackValueOffset_min ) = 0xFFFFU;
    GXV_TRAK_DATA( trackValueOffset_max ) = 0x0000;

    GXV_LIMIT_CHECK( nTracks * ( 4 + 2 + 2 ) );

    for ( i = 0; i < nTracks; i++ )
    {
      p = table + i * ( 4 + 2 + 2 );
      track     = FT_NEXT_LONG( p );
      nameIndex = FT_NEXT_USHORT( p );
      offset    = FT_NEXT_USHORT( p );

      if ( offset < GXV_TRAK_DATA( trackValueOffset_min ) )
        GXV_TRAK_DATA( trackValueOffset_min ) = offset;
      if ( offset > GXV_TRAK_DATA( trackValueOffset_max ) )
        GXV_TRAK_DATA( trackValueOffset_max ) = offset;

      gxv_sfntName_validate( nameIndex, 256, 32767, gxvalid );

      for ( j = i; j < nTracks; j++ )
      {
         p = table + j * ( 4 + 2 + 2 );
         t = FT_NEXT_LONG( p );
         if ( t == track )
           GXV_TRACE(( "duplicated entries found for track value 0x%x\n",
                        track ));
      }
    }

    gxvalid->subtable_length = (FT_ULong)( p - table );
    GXV_EXIT;
  }


  static void
  gxv_trak_trackData_validate( FT_Bytes       table,
                               FT_Bytes       limit,
                               GXV_Validator  gxvalid )
  {
    FT_Bytes   p = table;
    FT_UShort  nTracks;
    FT_UShort  nSizes;
    FT_ULong   sizeTableOffset;

    GXV_ODTECT( 4, odtect );


    GXV_ODTECT_INIT( odtect );
    GXV_NAME_ENTER( "trackData" );

    /* read the header of trackData */
    GXV_LIMIT_CHECK( 2 + 2 + 4 );
    nTracks         = FT_NEXT_USHORT( p );
    nSizes          = FT_NEXT_USHORT( p );
    sizeTableOffset = FT_NEXT_ULONG( p );

    gxv_odtect_add_range( table, (FT_ULong)( p - table ),
                          "trackData header", odtect );

    /* validate trackTable */
    gxv_trak_trackTable_validate( p, limit, nTracks, gxvalid );
    gxv_odtect_add_range( p, gxvalid->subtable_length,
                          "trackTable", odtect );

    /* sizeTable is array of FT_Fixed, don't check contents */
    p = gxvalid->root->base + sizeTableOffset;
    GXV_LIMIT_CHECK( nSizes * 4 );
    gxv_odtect_add_range( p, nSizes * 4, "sizeTable", odtect );

    /* validate trackValueOffet */
    p = gxvalid->root->base + GXV_TRAK_DATA( trackValueOffset_min );
    if ( limit - p < nTracks * nSizes * 2 )
      GXV_TRACE(( "too short trackValue array\n" ));

    p = gxvalid->root->base + GXV_TRAK_DATA( trackValueOffset_max );
    GXV_LIMIT_CHECK( nSizes * 2 );

    gxv_odtect_add_range( gxvalid->root->base
                            + GXV_TRAK_DATA( trackValueOffset_min ),
                          GXV_TRAK_DATA( trackValueOffset_max )
                            - GXV_TRAK_DATA( trackValueOffset_min )
                            + nSizes * 2,
                          "trackValue array", odtect );

    gxv_odtect_validate( odtect, gxvalid );

    GXV_EXIT;
  }


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                          trak TABLE                           *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  FT_LOCAL_DEF( void )
  gxv_trak_validate( FT_Bytes      table,
                     FT_Face       face,
                     FT_Validator  ftvalid )
  {
    FT_Bytes          p = table;
    FT_Bytes          limit = 0;

    GXV_ValidatorRec  gxvalidrec;
    GXV_Validator     gxvalid = &gxvalidrec;
    GXV_trak_DataRec  trakrec;
    GXV_trak_Data     trak = &trakrec;

    FT_ULong   version;
    FT_UShort  format;
    FT_UShort  horizOffset;
    FT_UShort  vertOffset;
    FT_UShort  reserved;


    GXV_ODTECT( 3, odtect );

    GXV_ODTECT_INIT( odtect );
    gxvalid->root       = ftvalid;
    gxvalid->table_data = trak;
    gxvalid->face       = face;

    limit      = gxvalid->root->limit;

    FT_TRACE3(( "validating `trak' table\n" ));
    GXV_INIT;

    GXV_LIMIT_CHECK( 4 + 2 + 2 + 2 + 2 );
    version     = FT_NEXT_ULONG( p );
    format      = FT_NEXT_USHORT( p );
    horizOffset = FT_NEXT_USHORT( p );
    vertOffset  = FT_NEXT_USHORT( p );
    reserved    = FT_NEXT_USHORT( p );

    GXV_TRACE(( " (version = 0x%08x)\n", version ));
    GXV_TRACE(( " (format = 0x%04x)\n", format ));
    GXV_TRACE(( " (horizOffset = 0x%04x)\n", horizOffset ));
    GXV_TRACE(( " (vertOffset = 0x%04x)\n", vertOffset ));
    GXV_TRACE(( " (reserved = 0x%04x)\n", reserved ));

    /* Version 1.0 (always:1996) */
    if ( version != 0x00010000UL )
      FT_INVALID_FORMAT;

    /* format 0 (always:1996) */
    if ( format != 0x0000 )
      FT_INVALID_FORMAT;

    GXV_32BIT_ALIGNMENT_VALIDATE( horizOffset );
    GXV_32BIT_ALIGNMENT_VALIDATE( vertOffset );

    /* Reserved Fixed Value (always) */
    if ( reserved != 0x0000 )
      FT_INVALID_DATA;

    /* validate trackData */
    if ( 0 < horizOffset )
    {
      gxv_trak_trackData_validate( table + horizOffset, limit, gxvalid );
      gxv_odtect_add_range( table + horizOffset, gxvalid->subtable_length,
                            "horizJustData", odtect );
    }

    if ( 0 < vertOffset )
    {
      gxv_trak_trackData_validate( table + vertOffset, limit, gxvalid );
      gxv_odtect_add_range( table + vertOffset, gxvalid->subtable_length,
                            "vertJustData", odtect );
    }

    gxv_odtect_validate( odtect, gxvalid );

    FT_TRACE4(( "\n" ));
  }


/* END */
