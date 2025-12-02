/****************************************************************************
 *
 * gxvopbd.c
 *
 *   TrueTypeGX/AAT opbd table validation (body).
 *
 * Copyright (C) 2004-2025 by
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
#define FT_COMPONENT  gxvopbd


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                      Data and Types                           *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  typedef struct  GXV_opbd_DataRec_
  {
    FT_UShort  format;
    FT_UShort  valueOffset_min;

  } GXV_opbd_DataRec, *GXV_opbd_Data;


#define GXV_OPBD_DATA( FIELD )  GXV_TABLE_DATA( opbd, FIELD )


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                      UTILITY FUNCTIONS                        *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  static void
  gxv_opbd_LookupValue_validate( FT_UShort            glyph,
                                 GXV_LookupValueCPtr  value_p,
                                 GXV_Validator        gxvalid )
  {
    /* offset in LookupTable is measured from the head of opbd table */
    FT_Bytes   p     = gxvalid->root->base + value_p->u;
    FT_Bytes   limit = gxvalid->root->limit;
    FT_Short   delta_value;
    int        i;


    if ( value_p->u < GXV_OPBD_DATA( valueOffset_min ) )
      GXV_OPBD_DATA( valueOffset_min ) = value_p->u;

    for ( i = 0; i < 4; i++ )
    {
      GXV_LIMIT_CHECK( 2 );
      delta_value = FT_NEXT_SHORT( p );

      if ( GXV_OPBD_DATA( format ) )    /* format 1, value is ctrl pt. */
      {
        if ( delta_value == -1 )
          continue;

        gxv_ctlPoint_validate( glyph, (FT_UShort)delta_value, gxvalid );
      }
      else                              /* format 0, value is distance */
        continue;
    }
  }


  /*
    opbd ---------------------+
                              |
    +===============+         |
    | lookup header |         |
    +===============+         |
    | BinSrchHeader |         |
    +===============+         |
    | lastGlyph[0]  |         |
    +---------------+         |
    | firstGlyph[0] |         |  head of opbd sfnt table
    +---------------+         |             +
    | offset[0]     |    ->   |          offset            [byte]
    +===============+         |             +
    | lastGlyph[1]  |         | (glyphID - firstGlyph) * 4 * sizeof(FT_Short) [byte]
    +---------------+         |
    | firstGlyph[1] |         |
    +---------------+         |
    | offset[1]     |         |
    +===============+         |
                              |
     ....                     |
                              |
    48bit value array         |
    +===============+         |
    |     value     | <-------+
    |               |
    |               |
    |               |
    +---------------+
    .... */

  static GXV_LookupValueDesc
  gxv_opbd_LookupFmt4_transit( FT_UShort            relative_gindex,
                               GXV_LookupValueCPtr  base_value_p,
                               FT_Bytes             lookuptbl_limit,
                               GXV_Validator        gxvalid )
  {
    GXV_LookupValueDesc  value;

    FT_UNUSED( lookuptbl_limit );
    FT_UNUSED( gxvalid );

    /* XXX: check range? */
    value.u = (FT_UShort)( base_value_p->u +
                           relative_gindex * 4 * sizeof ( FT_Short ) );

    return value;
  }


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                         opbd TABLE                            *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  FT_LOCAL_DEF( void )
  gxv_opbd_validate( FT_Bytes      table,
                     FT_Face       face,
                     FT_Validator  ftvalid )
  {
    GXV_ValidatorRec  gxvalidrec;
    GXV_Validator     gxvalid = &gxvalidrec;
    GXV_opbd_DataRec  opbdrec;
    GXV_opbd_Data     opbd  = &opbdrec;
    FT_Bytes          p     = table;
    FT_Bytes          limit = 0;

    FT_ULong  version;


    gxvalid->root       = ftvalid;
    gxvalid->table_data = opbd;
    gxvalid->face       = face;

    FT_TRACE3(( "validating `opbd' table\n" ));
    GXV_INIT;
    GXV_OPBD_DATA( valueOffset_min ) = 0xFFFFU;


    GXV_LIMIT_CHECK( 4 + 2 );
    version                 = FT_NEXT_ULONG( p );
    GXV_OPBD_DATA( format ) = FT_NEXT_USHORT( p );


    /* only 0x00010000 is defined (1996) */
    GXV_TRACE(( "(version=0x%08lx)\n", version ));
    if ( 0x00010000UL != version )
      FT_INVALID_FORMAT;

    /* only values 0 and 1 are defined (1996) */
    GXV_TRACE(( "(format=0x%04x)\n", GXV_OPBD_DATA( format ) ));
    if ( 0x0001 < GXV_OPBD_DATA( format ) )
      FT_INVALID_FORMAT;

    gxvalid->lookupval_sign   = GXV_LOOKUPVALUE_UNSIGNED;
    gxvalid->lookupval_func   = gxv_opbd_LookupValue_validate;
    gxvalid->lookupfmt4_trans = gxv_opbd_LookupFmt4_transit;

    gxv_LookupTable_validate( p, limit, gxvalid );
    p += gxvalid->subtable_length;

    if ( p > table + GXV_OPBD_DATA( valueOffset_min ) )
    {
      GXV_TRACE((
        "found overlap between LookupTable and opbd_value array\n" ));
      FT_INVALID_OFFSET;
    }

    FT_TRACE4(( "\n" ));
  }


/* END */
