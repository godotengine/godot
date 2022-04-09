/****************************************************************************
 *
 * gxvlcar.c
 *
 *   TrueTypeGX/AAT lcar table validation (body).
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
#define FT_COMPONENT  gxvlcar


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                      Data and Types                           *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  typedef struct  GXV_lcar_DataRec_
  {
    FT_UShort  format;

  } GXV_lcar_DataRec, *GXV_lcar_Data;


#define GXV_LCAR_DATA( FIELD )  GXV_TABLE_DATA( lcar, FIELD )


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                      UTILITY FUNCTIONS                        *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  static void
  gxv_lcar_partial_validate( FT_Short       partial,
                             FT_UShort      glyph,
                             GXV_Validator  gxvalid )
  {
    GXV_NAME_ENTER( "partial" );

    if ( GXV_LCAR_DATA( format ) != 1 )
      goto Exit;

    gxv_ctlPoint_validate( glyph, (FT_UShort)partial, gxvalid );

  Exit:
    GXV_EXIT;
  }


  static void
  gxv_lcar_LookupValue_validate( FT_UShort            glyph,
                                 GXV_LookupValueCPtr  value_p,
                                 GXV_Validator        gxvalid )
  {
    FT_Bytes   p     = gxvalid->root->base + value_p->u;
    FT_Bytes   limit = gxvalid->root->limit;
    FT_UShort  count;
    FT_Short   partial;
    FT_UShort  i;


    GXV_NAME_ENTER( "element in lookupTable" );

    GXV_LIMIT_CHECK( 2 );
    count = FT_NEXT_USHORT( p );

    GXV_LIMIT_CHECK( 2 * count );
    for ( i = 0; i < count; i++ )
    {
      partial = FT_NEXT_SHORT( p );
      gxv_lcar_partial_validate( partial, glyph, gxvalid );
    }

    GXV_EXIT;
  }


  /*
    +------ lcar --------------------+
    |                                |
    |      +===============+         |
    |      | lookup header |         |
    |      +===============+         |
    |      | BinSrchHeader |         |
    |      +===============+         |
    |      | lastGlyph[0]  |         |
    |      +---------------+         |
    |      | firstGlyph[0] |         |  head of lcar sfnt table
    |      +---------------+         |             +
    |      | offset[0]     |    ->   |          offset            [byte]
    |      +===============+         |             +
    |      | lastGlyph[1]  |         | (glyphID - firstGlyph) * 2 [byte]
    |      +---------------+         |
    |      | firstGlyph[1] |         |
    |      +---------------+         |
    |      | offset[1]     |         |
    |      +===============+         |
    |                                |
    |       ....                     |
    |                                |
    |      16bit value array         |
    |      +===============+         |
    +------|     value     | <-------+
    |       ....
    |
    |
    |
    |
    |
    +---->  lcar values...handled by lcar callback function
  */

  static GXV_LookupValueDesc
  gxv_lcar_LookupFmt4_transit( FT_UShort            relative_gindex,
                               GXV_LookupValueCPtr  base_value_p,
                               FT_Bytes             lookuptbl_limit,
                               GXV_Validator        gxvalid )
  {
    FT_Bytes             p;
    FT_Bytes             limit;
    FT_UShort            offset;
    GXV_LookupValueDesc  value;

    FT_UNUSED( lookuptbl_limit );

    /* XXX: check range? */
    offset = (FT_UShort)( base_value_p->u +
                          relative_gindex * sizeof ( FT_UShort ) );
    p      = gxvalid->root->base + offset;
    limit  = gxvalid->root->limit;

    GXV_LIMIT_CHECK ( 2 );
    value.u = FT_NEXT_USHORT( p );

    return value;
  }


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                          lcar TABLE                           *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  FT_LOCAL_DEF( void )
  gxv_lcar_validate( FT_Bytes      table,
                     FT_Face       face,
                     FT_Validator  ftvalid )
  {
    FT_Bytes          p     = table;
    FT_Bytes          limit = 0;
    GXV_ValidatorRec  gxvalidrec;
    GXV_Validator     gxvalid = &gxvalidrec;

    GXV_lcar_DataRec  lcarrec;
    GXV_lcar_Data     lcar = &lcarrec;

    FT_Fixed          version;


    gxvalid->root       = ftvalid;
    gxvalid->table_data = lcar;
    gxvalid->face       = face;

    FT_TRACE3(( "validating `lcar' table\n" ));
    GXV_INIT;

    GXV_LIMIT_CHECK( 4 + 2 );
    version = FT_NEXT_LONG( p );
    GXV_LCAR_DATA( format ) = FT_NEXT_USHORT( p );

    if ( version != 0x00010000UL)
      FT_INVALID_FORMAT;

    if ( GXV_LCAR_DATA( format ) > 1 )
      FT_INVALID_FORMAT;

    gxvalid->lookupval_sign   = GXV_LOOKUPVALUE_UNSIGNED;
    gxvalid->lookupval_func   = gxv_lcar_LookupValue_validate;
    gxvalid->lookupfmt4_trans = gxv_lcar_LookupFmt4_transit;
    gxv_LookupTable_validate( p, limit, gxvalid );

    FT_TRACE4(( "\n" ));
  }


/* END */
