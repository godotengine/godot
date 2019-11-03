/****************************************************************************
 *
 * gxvmorx.c
 *
 *   TrueTypeGX/AAT morx table validation (body).
 *
 * Copyright (C) 2005-2019 by
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


#include "gxvmorx.h"


  /**************************************************************************
   *
   * The macro FT_COMPONENT is used in trace mode.  It is an implicit
   * parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log
   * messages during execution.
   */
#undef  FT_COMPONENT
#define FT_COMPONENT  gxvmorx


  static void
  gxv_morx_subtables_validate( FT_Bytes       table,
                               FT_Bytes       limit,
                               FT_UShort      nSubtables,
                               GXV_Validator  gxvalid )
  {
    FT_Bytes  p = table;

    GXV_Validate_Func fmt_funcs_table[] =
    {
      gxv_morx_subtable_type0_validate, /* 0 */
      gxv_morx_subtable_type1_validate, /* 1 */
      gxv_morx_subtable_type2_validate, /* 2 */
      NULL,                             /* 3 */
      gxv_morx_subtable_type4_validate, /* 4 */
      gxv_morx_subtable_type5_validate, /* 5 */

    };

    FT_UShort  i;


    GXV_NAME_ENTER( "subtables in a chain" );

    for ( i = 0; i < nSubtables; i++ )
    {
      GXV_Validate_Func  func;

      FT_ULong  length;
      FT_ULong  coverage;
#ifdef GXV_LOAD_UNUSED_VARS
      FT_ULong  subFeatureFlags;
#endif
      FT_ULong  type;
      FT_ULong  rest;


      GXV_LIMIT_CHECK( 4 + 4 + 4 );
      length          = FT_NEXT_ULONG( p );
      coverage        = FT_NEXT_ULONG( p );
#ifdef GXV_LOAD_UNUSED_VARS
      subFeatureFlags = FT_NEXT_ULONG( p );
#else
      p += 4;
#endif

      GXV_TRACE(( "validating chain subtable %d/%d (%d bytes)\n",
                  i + 1, nSubtables, length ));

      type = coverage & 0x0007;
      rest = length - ( 4 + 4 + 4 );
      GXV_LIMIT_CHECK( rest );

      /* morx coverage consists of mort_coverage & 16bit padding */
      gxv_mort_coverage_validate( (FT_UShort)( ( coverage >> 16 ) | coverage ),
                                  gxvalid );
      if ( type > 5 )
        FT_INVALID_FORMAT;

      func = fmt_funcs_table[type];
      if ( !func )
        GXV_TRACE(( "morx type %d is reserved\n", type ));

      func( p, p + rest, gxvalid );

      /* TODO: subFeatureFlags should be unique in a table? */
      p += rest;
    }

    gxvalid->subtable_length = (FT_ULong)( p - table );

    GXV_EXIT;
  }


  static void
  gxv_morx_chain_validate( FT_Bytes       table,
                           FT_Bytes       limit,
                           GXV_Validator  gxvalid )
  {
    FT_Bytes  p = table;
#ifdef GXV_LOAD_UNUSED_VARS
    FT_ULong  defaultFlags;
#endif
    FT_ULong  chainLength;
    FT_ULong  nFeatureFlags;
    FT_ULong  nSubtables;


    GXV_NAME_ENTER( "morx chain header" );

    GXV_LIMIT_CHECK( 4 + 4 + 4 + 4 );
#ifdef GXV_LOAD_UNUSED_VARS
    defaultFlags  = FT_NEXT_ULONG( p );
#else
    p += 4;
#endif
    chainLength   = FT_NEXT_ULONG( p );
    nFeatureFlags = FT_NEXT_ULONG( p );
    nSubtables    = FT_NEXT_ULONG( p );

    /* feature-array of morx is same with that of mort */
    gxv_mort_featurearray_validate( p, limit, nFeatureFlags, gxvalid );
    p += gxvalid->subtable_length;

    if ( nSubtables >= 0x10000L )
      FT_INVALID_DATA;

    gxv_morx_subtables_validate( p, table + chainLength,
                                 (FT_UShort)nSubtables, gxvalid );

    gxvalid->subtable_length = chainLength;

    /* TODO: defaultFlags should be compared with the flags in tables */

    GXV_EXIT;
  }


  FT_LOCAL_DEF( void )
  gxv_morx_validate( FT_Bytes      table,
                     FT_Face       face,
                     FT_Validator  ftvalid )
  {
    GXV_ValidatorRec  gxvalidrec;
    GXV_Validator     gxvalid = &gxvalidrec;
    FT_Bytes          p     = table;
    FT_Bytes          limit = 0;
    FT_ULong          version;
    FT_ULong          nChains;
    FT_ULong          i;


    gxvalid->root = ftvalid;
    gxvalid->face = face;

    FT_TRACE3(( "validating `morx' table\n" ));
    GXV_INIT;

    GXV_LIMIT_CHECK( 4 + 4 );
    version = FT_NEXT_ULONG( p );
    nChains = FT_NEXT_ULONG( p );

    if ( version != 0x00020000UL )
      FT_INVALID_FORMAT;

    for ( i = 0; i < nChains; i++ )
    {
      GXV_TRACE(( "validating chain %d/%d\n", i + 1, nChains ));
      GXV_32BIT_ALIGNMENT_VALIDATE( p - table );
      gxv_morx_chain_validate( p, limit, gxvalid );
      p += gxvalid->subtable_length;
    }

    FT_TRACE4(( "\n" ));
  }


/* END */
