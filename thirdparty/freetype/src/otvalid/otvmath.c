/****************************************************************************
 *
 * otvmath.c
 *
 *   OpenType MATH table validation (body).
 *
 * Copyright (C) 2007-2019 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * Written by George Williams.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#include "otvalid.h"
#include "otvcommn.h"
#include "otvgpos.h"


  /**************************************************************************
   *
   * The macro FT_COMPONENT is used in trace mode.  It is an implicit
   * parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log
   * messages during execution.
   */
#undef  FT_COMPONENT
#define FT_COMPONENT  otvmath



  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                  MATH TYPOGRAPHIC CONSTANTS                   *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  static void
  otv_MathConstants_validate( FT_Bytes       table,
                              OTV_Validator  otvalid )
  {
    FT_Bytes  p = table;
    FT_UInt   i;
    FT_UInt   table_size;

    OTV_OPTIONAL_TABLE( DeviceTableOffset );


    OTV_NAME_ENTER( "MathConstants" );

    /* 56 constants, 51 have device tables */
    OTV_LIMIT_CHECK( 2 * ( 56 + 51 ) );
    table_size = 2 * ( 56 + 51 );

    p += 4 * 2;                 /* First 4 constants have no device tables */
    for ( i = 0; i < 51; i++ )
    {
      p += 2;                                            /* skip the value */
      OTV_OPTIONAL_OFFSET( DeviceTableOffset );
      OTV_SIZE_CHECK( DeviceTableOffset );
      if ( DeviceTableOffset )
        otv_Device_validate( table + DeviceTableOffset, otvalid );
    }

    OTV_EXIT;
  }


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                   MATH ITALICS CORRECTION                     *****/
  /*****                 MATH TOP ACCENT ATTACHMENT                    *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  static void
  otv_MathItalicsCorrectionInfo_validate( FT_Bytes       table,
                                          OTV_Validator  otvalid,
                                          FT_Int         isItalic )
  {
    FT_Bytes  p = table;
    FT_UInt   i, cnt, table_size;

    OTV_OPTIONAL_TABLE( Coverage );
    OTV_OPTIONAL_TABLE( DeviceTableOffset );

    FT_UNUSED( isItalic );  /* only used if tracing is active */


    OTV_NAME_ENTER( isItalic ? "MathItalicsCorrectionInfo"
                             : "MathTopAccentAttachment" );

    OTV_LIMIT_CHECK( 4 );

    OTV_OPTIONAL_OFFSET( Coverage );
    cnt = FT_NEXT_USHORT( p );

    OTV_LIMIT_CHECK( 4 * cnt );
    table_size = 4 + 4 * cnt;

    OTV_SIZE_CHECK( Coverage );
    otv_Coverage_validate( table + Coverage, otvalid, (FT_Int)cnt );

    for ( i = 0; i < cnt; i++ )
    {
      p += 2;                                            /* Skip the value */
      OTV_OPTIONAL_OFFSET( DeviceTableOffset );
      OTV_SIZE_CHECK( DeviceTableOffset );
      if ( DeviceTableOffset )
        otv_Device_validate( table + DeviceTableOffset, otvalid );
    }

    OTV_EXIT;
  }


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                           MATH KERNING                        *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  static void
  otv_MathKern_validate( FT_Bytes       table,
                         OTV_Validator  otvalid )
  {
    FT_Bytes  p = table;
    FT_UInt   i, cnt, table_size;

    OTV_OPTIONAL_TABLE( DeviceTableOffset );


    /* OTV_NAME_ENTER( "MathKern" );*/

    OTV_LIMIT_CHECK( 2 );

    cnt = FT_NEXT_USHORT( p );

    OTV_LIMIT_CHECK( 4 * cnt + 2 );
    table_size = 4 + 4 * cnt;

    /* Heights */
    for ( i = 0; i < cnt; i++ )
    {
      p += 2;                                            /* Skip the value */
      OTV_OPTIONAL_OFFSET( DeviceTableOffset );
      OTV_SIZE_CHECK( DeviceTableOffset );
      if ( DeviceTableOffset )
        otv_Device_validate( table + DeviceTableOffset, otvalid );
    }

    /* One more Kerning value */
    for ( i = 0; i < cnt + 1; i++ )
    {
      p += 2;                                            /* Skip the value */
      OTV_OPTIONAL_OFFSET( DeviceTableOffset );
      OTV_SIZE_CHECK( DeviceTableOffset );
      if ( DeviceTableOffset )
        otv_Device_validate( table + DeviceTableOffset, otvalid );
    }

    OTV_EXIT;
  }


  static void
  otv_MathKernInfo_validate( FT_Bytes       table,
                             OTV_Validator  otvalid )
  {
    FT_Bytes  p = table;
    FT_UInt   i, j, cnt, table_size;

    OTV_OPTIONAL_TABLE( Coverage );
    OTV_OPTIONAL_TABLE( MKRecordOffset );


    OTV_NAME_ENTER( "MathKernInfo" );

    OTV_LIMIT_CHECK( 4 );

    OTV_OPTIONAL_OFFSET( Coverage );
    cnt = FT_NEXT_USHORT( p );

    OTV_LIMIT_CHECK( 8 * cnt );
    table_size = 4 + 8 * cnt;

    OTV_SIZE_CHECK( Coverage );
    otv_Coverage_validate( table + Coverage, otvalid, (FT_Int)cnt );

    for ( i = 0; i < cnt; i++ )
    {
      for ( j = 0; j < 4; j++ )
      {
        OTV_OPTIONAL_OFFSET( MKRecordOffset );
        OTV_SIZE_CHECK( MKRecordOffset );
        if ( MKRecordOffset )
          otv_MathKern_validate( table + MKRecordOffset, otvalid );
      }
    }

    OTV_EXIT;
  }


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                         MATH GLYPH INFO                       *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  static void
  otv_MathGlyphInfo_validate( FT_Bytes       table,
                              OTV_Validator  otvalid )
  {
    FT_Bytes  p = table;
    FT_UInt   MathItalicsCorrectionInfo, MathTopAccentAttachment;
    FT_UInt   ExtendedShapeCoverage, MathKernInfo;


    OTV_NAME_ENTER( "MathGlyphInfo" );

    OTV_LIMIT_CHECK( 8 );

    MathItalicsCorrectionInfo = FT_NEXT_USHORT( p );
    MathTopAccentAttachment   = FT_NEXT_USHORT( p );
    ExtendedShapeCoverage     = FT_NEXT_USHORT( p );
    MathKernInfo              = FT_NEXT_USHORT( p );

    if ( MathItalicsCorrectionInfo )
      otv_MathItalicsCorrectionInfo_validate(
        table + MathItalicsCorrectionInfo, otvalid, TRUE );

    /* Italic correction and Top Accent Attachment have the same format */
    if ( MathTopAccentAttachment )
      otv_MathItalicsCorrectionInfo_validate(
        table + MathTopAccentAttachment, otvalid, FALSE );

    if ( ExtendedShapeCoverage )
    {
      OTV_NAME_ENTER( "ExtendedShapeCoverage" );
      otv_Coverage_validate( table + ExtendedShapeCoverage, otvalid, -1 );
      OTV_EXIT;
    }

    if ( MathKernInfo )
      otv_MathKernInfo_validate( table + MathKernInfo, otvalid );

    OTV_EXIT;
  }


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                    MATH GLYPH CONSTRUCTION                    *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  static void
  otv_GlyphAssembly_validate( FT_Bytes       table,
                              OTV_Validator  otvalid )
  {
    FT_Bytes  p = table;
    FT_UInt   pcnt, table_size;
    FT_UInt   i;

    OTV_OPTIONAL_TABLE( DeviceTableOffset );


    /* OTV_NAME_ENTER( "GlyphAssembly" ); */

    OTV_LIMIT_CHECK( 6 );

    p += 2;                           /* Skip the Italics Correction value */
    OTV_OPTIONAL_OFFSET( DeviceTableOffset );
    pcnt = FT_NEXT_USHORT( p );

    OTV_LIMIT_CHECK( 8 * pcnt );
    table_size = 6 + 8 * pcnt;

    OTV_SIZE_CHECK( DeviceTableOffset );
    if ( DeviceTableOffset )
      otv_Device_validate( table + DeviceTableOffset, otvalid );

    for ( i = 0; i < pcnt; i++ )
    {
      FT_UInt  gid;


      gid = FT_NEXT_USHORT( p );
      if ( gid >= otvalid->glyph_count )
        FT_INVALID_GLYPH_ID;
      p += 2*4;             /* skip the Start, End, Full, and Flags fields */
    }

    /* OTV_EXIT; */
  }


  static void
  otv_MathGlyphConstruction_validate( FT_Bytes       table,
                                      OTV_Validator  otvalid )
  {
    FT_Bytes  p = table;
    FT_UInt   vcnt, table_size;
    FT_UInt   i;

    OTV_OPTIONAL_TABLE( GlyphAssembly );


    /* OTV_NAME_ENTER( "MathGlyphConstruction" ); */

    OTV_LIMIT_CHECK( 4 );

    OTV_OPTIONAL_OFFSET( GlyphAssembly );
    vcnt = FT_NEXT_USHORT( p );

    OTV_LIMIT_CHECK( 4 * vcnt );
    table_size = 4 + 4 * vcnt;

    for ( i = 0; i < vcnt; i++ )
    {
      FT_UInt  gid;


      gid = FT_NEXT_USHORT( p );
      if ( gid >= otvalid->glyph_count )
        FT_INVALID_GLYPH_ID;
      p += 2;                          /* skip the size */
    }

    OTV_SIZE_CHECK( GlyphAssembly );
    if ( GlyphAssembly )
      otv_GlyphAssembly_validate( table+GlyphAssembly, otvalid );

    /* OTV_EXIT; */
  }


  static void
  otv_MathVariants_validate( FT_Bytes       table,
                             OTV_Validator  otvalid )
  {
    FT_Bytes  p = table;
    FT_UInt   vcnt, hcnt, i, table_size;

    OTV_OPTIONAL_TABLE( VCoverage );
    OTV_OPTIONAL_TABLE( HCoverage );
    OTV_OPTIONAL_TABLE( Offset );


    OTV_NAME_ENTER( "MathVariants" );

    OTV_LIMIT_CHECK( 10 );

    p += 2;                       /* Skip the MinConnectorOverlap constant */
    OTV_OPTIONAL_OFFSET( VCoverage );
    OTV_OPTIONAL_OFFSET( HCoverage );
    vcnt = FT_NEXT_USHORT( p );
    hcnt = FT_NEXT_USHORT( p );

    OTV_LIMIT_CHECK( 2 * vcnt + 2 * hcnt );
    table_size = 10 + 2 * vcnt + 2 * hcnt;

    OTV_SIZE_CHECK( VCoverage );
    if ( VCoverage )
      otv_Coverage_validate( table + VCoverage, otvalid, (FT_Int)vcnt );

    OTV_SIZE_CHECK( HCoverage );
    if ( HCoverage )
      otv_Coverage_validate( table + HCoverage, otvalid, (FT_Int)hcnt );

    for ( i = 0; i < vcnt; i++ )
    {
      OTV_OPTIONAL_OFFSET( Offset );
      OTV_SIZE_CHECK( Offset );
      otv_MathGlyphConstruction_validate( table + Offset, otvalid );
    }

    for ( i = 0; i < hcnt; i++ )
    {
      OTV_OPTIONAL_OFFSET( Offset );
      OTV_SIZE_CHECK( Offset );
      otv_MathGlyphConstruction_validate( table + Offset, otvalid );
    }

    OTV_EXIT;
  }


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                          MATH TABLE                           *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  /* sets otvalid->glyph_count */

  FT_LOCAL_DEF( void )
  otv_MATH_validate( FT_Bytes      table,
                     FT_UInt       glyph_count,
                     FT_Validator  ftvalid )
  {
    OTV_ValidatorRec  otvalidrec;
    OTV_Validator     otvalid = &otvalidrec;
    FT_Bytes          p       = table;
    FT_UInt           MathConstants, MathGlyphInfo, MathVariants;


    otvalid->root = ftvalid;

    FT_TRACE3(( "validating MATH table\n" ));
    OTV_INIT;

    OTV_LIMIT_CHECK( 10 );

    if ( FT_NEXT_ULONG( p ) != 0x10000UL )      /* Version */
      FT_INVALID_FORMAT;

    MathConstants = FT_NEXT_USHORT( p );
    MathGlyphInfo = FT_NEXT_USHORT( p );
    MathVariants  = FT_NEXT_USHORT( p );

    otvalid->glyph_count = glyph_count;

    otv_MathConstants_validate( table + MathConstants,
                                otvalid );
    otv_MathGlyphInfo_validate( table + MathGlyphInfo,
                                otvalid );
    otv_MathVariants_validate ( table + MathVariants,
                                otvalid );

    FT_TRACE4(( "\n" ));
  }


/* END */
