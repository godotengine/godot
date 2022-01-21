/****************************************************************************
 *
 * gxvbsln.c
 *
 *   TrueTypeGX/AAT bsln table validation (body).
 *
 * Copyright (C) 2004-2021 by
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


  /**************************************************************************
   *
   * The macro FT_COMPONENT is used in trace mode.  It is an implicit
   * parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log
   * messages during execution.
   */
#undef  FT_COMPONENT
#define FT_COMPONENT  gxvbsln


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                      Data and Types                           *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

#define GXV_BSLN_VALUE_COUNT  32
#define GXV_BSLN_VALUE_EMPTY  0xFFFFU


  typedef struct  GXV_bsln_DataRec_
  {
    FT_Bytes   ctlPoints_p;
    FT_UShort  defaultBaseline;

  } GXV_bsln_DataRec, *GXV_bsln_Data;


#define GXV_BSLN_DATA( field )  GXV_TABLE_DATA( bsln, field )


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                      UTILITY FUNCTIONS                        *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  static void
  gxv_bsln_LookupValue_validate( FT_UShort            glyph,
                                 GXV_LookupValueCPtr  value_p,
                                 GXV_Validator        gxvalid )
  {
    FT_UShort     v = value_p->u;
    FT_UShort*    ctlPoints;

    FT_UNUSED( glyph );


    GXV_NAME_ENTER( "lookup value" );

    if ( v >= GXV_BSLN_VALUE_COUNT )
      FT_INVALID_DATA;

    ctlPoints = (FT_UShort*)GXV_BSLN_DATA( ctlPoints_p );
    if ( ctlPoints && ctlPoints[v] == GXV_BSLN_VALUE_EMPTY )
      FT_INVALID_DATA;

    GXV_EXIT;
  }


  /*
    +===============+ --------+
    | lookup header |         |
    +===============+         |
    | BinSrchHeader |         |
    +===============+         |
    | lastGlyph[0]  |         |
    +---------------+         |
    | firstGlyph[0] |         |    head of lookup table
    +---------------+         |             +
    | offset[0]     |    ->   |          offset            [byte]
    +===============+         |             +
    | lastGlyph[1]  |         | (glyphID - firstGlyph) * 2 [byte]
    +---------------+         |
    | firstGlyph[1] |         |
    +---------------+         |
    | offset[1]     |         |
    +===============+         |
                              |
    ...                       |
                              |
    16bit value array         |
    +===============+         |
    |     value     | <-------+
    ...
  */

  static GXV_LookupValueDesc
  gxv_bsln_LookupFmt4_transit( FT_UShort            relative_gindex,
                               GXV_LookupValueCPtr  base_value_p,
                               FT_Bytes             lookuptbl_limit,
                               GXV_Validator        gxvalid )
  {
    FT_Bytes             p;
    FT_Bytes             limit;
    FT_UShort            offset;
    GXV_LookupValueDesc  value;

    /* XXX: check range ? */
    offset = (FT_UShort)( base_value_p->u +
                          ( relative_gindex * sizeof ( FT_UShort ) ) );

    p     = gxvalid->lookuptbl_head + offset;
    limit = lookuptbl_limit;
    GXV_LIMIT_CHECK( 2 );

    value.u = FT_NEXT_USHORT( p );

    return value;
  }


  static void
  gxv_bsln_parts_fmt0_validate( FT_Bytes       tables,
                                FT_Bytes       limit,
                                GXV_Validator  gxvalid )
  {
    FT_Bytes  p = tables;


    GXV_NAME_ENTER( "parts format 0" );

    /* deltas */
    GXV_LIMIT_CHECK( 2 * GXV_BSLN_VALUE_COUNT );

    gxvalid->table_data = NULL;      /* No ctlPoints here. */

    GXV_EXIT;
  }


  static void
  gxv_bsln_parts_fmt1_validate( FT_Bytes       tables,
                                FT_Bytes       limit,
                                GXV_Validator  gxvalid )
  {
    FT_Bytes  p = tables;


    GXV_NAME_ENTER( "parts format 1" );

    /* deltas */
    gxv_bsln_parts_fmt0_validate( p, limit, gxvalid );

    /* mappingData */
    gxvalid->lookupval_sign   = GXV_LOOKUPVALUE_UNSIGNED;
    gxvalid->lookupval_func   = gxv_bsln_LookupValue_validate;
    gxvalid->lookupfmt4_trans = gxv_bsln_LookupFmt4_transit;
    gxv_LookupTable_validate( p + 2 * GXV_BSLN_VALUE_COUNT,
                              limit,
                              gxvalid );

    GXV_EXIT;
  }


  static void
  gxv_bsln_parts_fmt2_validate( FT_Bytes       tables,
                                FT_Bytes       limit,
                                GXV_Validator  gxvalid )
  {
    FT_Bytes   p = tables;

    FT_UShort  stdGlyph;
    FT_UShort  ctlPoint;
    FT_Int     i;

    FT_UShort  defaultBaseline = GXV_BSLN_DATA( defaultBaseline );


    GXV_NAME_ENTER( "parts format 2" );

    GXV_LIMIT_CHECK( 2 + ( 2 * GXV_BSLN_VALUE_COUNT ) );

    /* stdGlyph */
    stdGlyph = FT_NEXT_USHORT( p );
    GXV_TRACE(( " (stdGlyph = %u)\n", stdGlyph ));

    gxv_glyphid_validate( stdGlyph, gxvalid );

    /* Record the position of ctlPoints */
    GXV_BSLN_DATA( ctlPoints_p ) = p;

    /* ctlPoints */
    for ( i = 0; i < GXV_BSLN_VALUE_COUNT; i++ )
    {
      ctlPoint = FT_NEXT_USHORT( p );
      if ( ctlPoint == GXV_BSLN_VALUE_EMPTY )
      {
        if ( i == defaultBaseline )
          FT_INVALID_DATA;
      }
      else
        gxv_ctlPoint_validate( stdGlyph, ctlPoint, gxvalid );
    }

    GXV_EXIT;
  }


  static void
  gxv_bsln_parts_fmt3_validate( FT_Bytes       tables,
                                FT_Bytes       limit,
                                GXV_Validator  gxvalid)
  {
    FT_Bytes  p = tables;


    GXV_NAME_ENTER( "parts format 3" );

    /* stdGlyph + ctlPoints */
    gxv_bsln_parts_fmt2_validate( p, limit, gxvalid );

    /* mappingData */
    gxvalid->lookupval_sign   = GXV_LOOKUPVALUE_UNSIGNED;
    gxvalid->lookupval_func   = gxv_bsln_LookupValue_validate;
    gxvalid->lookupfmt4_trans = gxv_bsln_LookupFmt4_transit;
    gxv_LookupTable_validate( p + ( 2 + 2 * GXV_BSLN_VALUE_COUNT ),
                              limit,
                              gxvalid );

    GXV_EXIT;
  }


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                         bsln TABLE                            *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  FT_LOCAL_DEF( void )
  gxv_bsln_validate( FT_Bytes      table,
                     FT_Face       face,
                     FT_Validator  ftvalid )
  {
    GXV_ValidatorRec  gxvalidrec;
    GXV_Validator     gxvalid = &gxvalidrec;

    GXV_bsln_DataRec  bslnrec;
    GXV_bsln_Data     bsln = &bslnrec;

    FT_Bytes  p     = table;
    FT_Bytes  limit = 0;

    FT_ULong   version;
    FT_UShort  format;
    FT_UShort  defaultBaseline;

    GXV_Validate_Func  fmt_funcs_table [] =
    {
      gxv_bsln_parts_fmt0_validate,
      gxv_bsln_parts_fmt1_validate,
      gxv_bsln_parts_fmt2_validate,
      gxv_bsln_parts_fmt3_validate,
    };


    gxvalid->root       = ftvalid;
    gxvalid->table_data = bsln;
    gxvalid->face       = face;

    FT_TRACE3(( "validating `bsln' table\n" ));
    GXV_INIT;


    GXV_LIMIT_CHECK( 4 + 2 + 2 );
    version         = FT_NEXT_ULONG( p );
    format          = FT_NEXT_USHORT( p );
    defaultBaseline = FT_NEXT_USHORT( p );

    /* only version 1.0 is defined (1996) */
    if ( version != 0x00010000UL )
      FT_INVALID_FORMAT;

    /* only format 1, 2, 3 are defined (1996) */
    GXV_TRACE(( " (format = %d)\n", format ));
    if ( format > 3 )
      FT_INVALID_FORMAT;

    if ( defaultBaseline > 31 )
      FT_INVALID_FORMAT;

    bsln->defaultBaseline = defaultBaseline;

    fmt_funcs_table[format]( p, limit, gxvalid );

    FT_TRACE4(( "\n" ));
  }


/* arch-tag: ebe81143-fdaa-4c68-a4d1-b57227daa3bc
   (do not change this comment) */


/* END */
