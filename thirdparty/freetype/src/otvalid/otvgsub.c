/***************************************************************************/
/*                                                                         */
/*  otvgsub.c                                                              */
/*                                                                         */
/*    OpenType GSUB table validation (body).                               */
/*                                                                         */
/*  Copyright 2004-2016 by                                                 */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


#include "otvalid.h"
#include "otvcommn.h"


  /*************************************************************************/
  /*                                                                       */
  /* The macro FT_COMPONENT is used in trace mode.  It is an implicit      */
  /* parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log  */
  /* messages during execution.                                            */
  /*                                                                       */
#undef  FT_COMPONENT
#define FT_COMPONENT  trace_otvgsub


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                  GSUB LOOKUP TYPE 1                           *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  /* uses otvalid->glyph_count */

  static void
  otv_SingleSubst_validate( FT_Bytes       table,
                            OTV_Validator  otvalid )
  {
    FT_Bytes  p = table;
    FT_UInt   SubstFormat;


    OTV_NAME_ENTER( "SingleSubst" );

    OTV_LIMIT_CHECK( 2 );
    SubstFormat = FT_NEXT_USHORT( p );

    OTV_TRACE(( " (format %d)\n", SubstFormat ));

    switch ( SubstFormat )
    {
    case 1:     /* SingleSubstFormat1 */
      {
        FT_Bytes  Coverage;
        FT_Int    DeltaGlyphID;
        FT_Long   idx;


        OTV_LIMIT_CHECK( 4 );
        Coverage     = table + FT_NEXT_USHORT( p );
        DeltaGlyphID = FT_NEXT_SHORT( p );

        otv_Coverage_validate( Coverage, otvalid, -1 );

        idx = (FT_Long)otv_Coverage_get_first( Coverage ) + DeltaGlyphID;
        if ( idx < 0 )
          FT_INVALID_DATA;

        idx = (FT_Long)otv_Coverage_get_last( Coverage ) + DeltaGlyphID;
        if ( (FT_UInt)idx >= otvalid->glyph_count )
          FT_INVALID_DATA;
      }
      break;

    case 2:     /* SingleSubstFormat2 */
      {
        FT_UInt  Coverage, GlyphCount;


        OTV_LIMIT_CHECK( 4 );
        Coverage   = FT_NEXT_USHORT( p );
        GlyphCount = FT_NEXT_USHORT( p );

        OTV_TRACE(( " (GlyphCount = %d)\n", GlyphCount ));

        otv_Coverage_validate( table + Coverage,
                               otvalid,
                               (FT_Int)GlyphCount );

        OTV_LIMIT_CHECK( GlyphCount * 2 );

        /* Substitute */
        for ( ; GlyphCount > 0; GlyphCount-- )
          if ( FT_NEXT_USHORT( p ) >= otvalid->glyph_count )
            FT_INVALID_GLYPH_ID;
      }
      break;

    default:
      FT_INVALID_FORMAT;
    }

    OTV_EXIT;
  }


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                  GSUB LOOKUP TYPE 2                           *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  /* sets otvalid->extra1 (glyph count) */

  static void
  otv_MultipleSubst_validate( FT_Bytes       table,
                              OTV_Validator  otvalid )
  {
    FT_Bytes  p = table;
    FT_UInt   SubstFormat;


    OTV_NAME_ENTER( "MultipleSubst" );

    OTV_LIMIT_CHECK( 2 );
    SubstFormat = FT_NEXT_USHORT( p );

    OTV_TRACE(( " (format %d)\n", SubstFormat ));

    switch ( SubstFormat )
    {
    case 1:
      otvalid->extra1 = otvalid->glyph_count;
      OTV_NEST2( MultipleSubstFormat1, Sequence );
      OTV_RUN( table, otvalid );
      break;

    default:
      FT_INVALID_FORMAT;
    }

    OTV_EXIT;
  }


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                    GSUB LOOKUP TYPE 3                         *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  /* sets otvalid->extra1 (glyph count) */

  static void
  otv_AlternateSubst_validate( FT_Bytes       table,
                               OTV_Validator  otvalid )
  {
    FT_Bytes  p = table;
    FT_UInt   SubstFormat;


    OTV_NAME_ENTER( "AlternateSubst" );

    OTV_LIMIT_CHECK( 2 );
    SubstFormat = FT_NEXT_USHORT( p );

    OTV_TRACE(( " (format %d)\n", SubstFormat ));

    switch ( SubstFormat )
    {
    case 1:
      otvalid->extra1 = otvalid->glyph_count;
      OTV_NEST2( AlternateSubstFormat1, AlternateSet );
      OTV_RUN( table, otvalid );
      break;

    default:
      FT_INVALID_FORMAT;
    }

    OTV_EXIT;
  }


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                    GSUB LOOKUP TYPE 4                         *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

#define LigatureFunc  otv_Ligature_validate

  /* uses otvalid->glyph_count */

  static void
  otv_Ligature_validate( FT_Bytes       table,
                         OTV_Validator  otvalid )
  {
    FT_Bytes  p = table;
    FT_UInt   LigatureGlyph, CompCount;


    OTV_ENTER;

    OTV_LIMIT_CHECK( 4 );
    LigatureGlyph = FT_NEXT_USHORT( p );
    if ( LigatureGlyph >= otvalid->glyph_count )
      FT_INVALID_DATA;

    CompCount = FT_NEXT_USHORT( p );

    OTV_TRACE(( " (CompCount = %d)\n", CompCount ));

    if ( CompCount == 0 )
      FT_INVALID_DATA;

    CompCount--;

    OTV_LIMIT_CHECK( CompCount * 2 );     /* Component */

    /* no need to check the Component glyph indices */

    OTV_EXIT;
  }


  static void
  otv_LigatureSubst_validate( FT_Bytes       table,
                              OTV_Validator  otvalid )
  {
    FT_Bytes  p = table;
    FT_UInt   SubstFormat;


    OTV_NAME_ENTER( "LigatureSubst" );

    OTV_LIMIT_CHECK( 2 );
    SubstFormat = FT_NEXT_USHORT( p );

    OTV_TRACE(( " (format %d)\n", SubstFormat ));

    switch ( SubstFormat )
    {
    case 1:
      OTV_NEST3( LigatureSubstFormat1, LigatureSet, Ligature );
      OTV_RUN( table, otvalid );
      break;

    default:
      FT_INVALID_FORMAT;
    }

    OTV_EXIT;
  }


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                  GSUB LOOKUP TYPE 5                           *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  /* sets otvalid->extra1 (lookup count) */

  static void
  otv_ContextSubst_validate( FT_Bytes       table,
                             OTV_Validator  otvalid )
  {
    FT_Bytes  p = table;
    FT_UInt   SubstFormat;


    OTV_NAME_ENTER( "ContextSubst" );

    OTV_LIMIT_CHECK( 2 );
    SubstFormat = FT_NEXT_USHORT( p );

    OTV_TRACE(( " (format %d)\n", SubstFormat ));

    switch ( SubstFormat )
    {
    case 1:
      /* no need to check glyph indices/classes used as input for these */
      /* context rules since even invalid glyph indices/classes return  */
      /* meaningful results                                             */

      otvalid->extra1 = otvalid->lookup_count;
      OTV_NEST3( ContextSubstFormat1, SubRuleSet, SubRule );
      OTV_RUN( table, otvalid );
      break;

    case 2:
      /* no need to check glyph indices/classes used as input for these */
      /* context rules since even invalid glyph indices/classes return  */
      /* meaningful results                                             */

      OTV_NEST3( ContextSubstFormat2, SubClassSet, SubClassRule );
      OTV_RUN( table, otvalid );
      break;

    case 3:
      OTV_NEST1( ContextSubstFormat3 );
      OTV_RUN( table, otvalid );
      break;

    default:
      FT_INVALID_FORMAT;
    }

    OTV_EXIT;
  }


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                    GSUB LOOKUP TYPE 6                         *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  /* sets otvalid->extra1 (lookup count)            */

  static void
  otv_ChainContextSubst_validate( FT_Bytes       table,
                                  OTV_Validator  otvalid )
  {
    FT_Bytes  p = table;
    FT_UInt   SubstFormat;


    OTV_NAME_ENTER( "ChainContextSubst" );

    OTV_LIMIT_CHECK( 2 );
    SubstFormat = FT_NEXT_USHORT( p );

    OTV_TRACE(( " (format %d)\n", SubstFormat ));

    switch ( SubstFormat )
    {
    case 1:
      /* no need to check glyph indices/classes used as input for these */
      /* context rules since even invalid glyph indices/classes return  */
      /* meaningful results                                             */

      otvalid->extra1 = otvalid->lookup_count;
      OTV_NEST3( ChainContextSubstFormat1,
                 ChainSubRuleSet, ChainSubRule );
      OTV_RUN( table, otvalid );
      break;

    case 2:
      /* no need to check glyph indices/classes used as input for these */
      /* context rules since even invalid glyph indices/classes return  */
      /* meaningful results                                             */

      OTV_NEST3( ChainContextSubstFormat2,
                 ChainSubClassSet, ChainSubClassRule );
      OTV_RUN( table, otvalid );
      break;

    case 3:
      OTV_NEST1( ChainContextSubstFormat3 );
      OTV_RUN( table, otvalid );
      break;

    default:
      FT_INVALID_FORMAT;
    }

    OTV_EXIT;
  }


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                    GSUB LOOKUP TYPE 7                         *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  /* uses otvalid->type_funcs */

  static void
  otv_ExtensionSubst_validate( FT_Bytes       table,
                               OTV_Validator  otvalid )
  {
    FT_Bytes  p = table;
    FT_UInt   SubstFormat;


    OTV_NAME_ENTER( "ExtensionSubst" );

    OTV_LIMIT_CHECK( 2 );
    SubstFormat = FT_NEXT_USHORT( p );

    OTV_TRACE(( " (format %d)\n", SubstFormat ));

    switch ( SubstFormat )
    {
    case 1:     /* ExtensionSubstFormat1 */
      {
        FT_UInt            ExtensionLookupType;
        FT_ULong           ExtensionOffset;
        OTV_Validate_Func  validate;


        OTV_LIMIT_CHECK( 6 );
        ExtensionLookupType = FT_NEXT_USHORT( p );
        ExtensionOffset     = FT_NEXT_ULONG( p );

        if ( ExtensionLookupType == 0 ||
             ExtensionLookupType == 7 ||
             ExtensionLookupType > 8  )
          FT_INVALID_DATA;

        validate = otvalid->type_funcs[ExtensionLookupType - 1];
        validate( table + ExtensionOffset, otvalid );
      }
      break;

    default:
      FT_INVALID_FORMAT;
    }

    OTV_EXIT;
  }


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                    GSUB LOOKUP TYPE 8                         *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  /* uses otvalid->glyph_count */

  static void
  otv_ReverseChainSingleSubst_validate( FT_Bytes       table,
                                        OTV_Validator  otvalid )
  {
    FT_Bytes  p = table, Coverage;
    FT_UInt   SubstFormat;
    FT_UInt   BacktrackGlyphCount, LookaheadGlyphCount, GlyphCount;


    OTV_NAME_ENTER( "ReverseChainSingleSubst" );

    OTV_LIMIT_CHECK( 2 );
    SubstFormat = FT_NEXT_USHORT( p );

    OTV_TRACE(( " (format %d)\n", SubstFormat ));

    switch ( SubstFormat )
    {
    case 1:     /* ReverseChainSingleSubstFormat1 */
      OTV_LIMIT_CHECK( 4 );
      Coverage            = table + FT_NEXT_USHORT( p );
      BacktrackGlyphCount = FT_NEXT_USHORT( p );

      OTV_TRACE(( " (BacktrackGlyphCount = %d)\n", BacktrackGlyphCount ));

      otv_Coverage_validate( Coverage, otvalid, -1 );

      OTV_LIMIT_CHECK( BacktrackGlyphCount * 2 + 2 );

      for ( ; BacktrackGlyphCount > 0; BacktrackGlyphCount-- )
        otv_Coverage_validate( table + FT_NEXT_USHORT( p ), otvalid, -1 );

      LookaheadGlyphCount = FT_NEXT_USHORT( p );

      OTV_TRACE(( " (LookaheadGlyphCount = %d)\n", LookaheadGlyphCount ));

      OTV_LIMIT_CHECK( LookaheadGlyphCount * 2 + 2 );

      for ( ; LookaheadGlyphCount > 0; LookaheadGlyphCount-- )
        otv_Coverage_validate( table + FT_NEXT_USHORT( p ), otvalid, -1 );

      GlyphCount = FT_NEXT_USHORT( p );

      OTV_TRACE(( " (GlyphCount = %d)\n", GlyphCount ));

      if ( GlyphCount != otv_Coverage_get_count( Coverage ) )
        FT_INVALID_DATA;

      OTV_LIMIT_CHECK( GlyphCount * 2 );

      /* Substitute */
      for ( ; GlyphCount > 0; GlyphCount-- )
        if ( FT_NEXT_USHORT( p ) >= otvalid->glyph_count )
          FT_INVALID_DATA;

      break;

    default:
      FT_INVALID_FORMAT;
    }

    OTV_EXIT;
  }


  static const OTV_Validate_Func  otv_gsub_validate_funcs[8] =
  {
    otv_SingleSubst_validate,
    otv_MultipleSubst_validate,
    otv_AlternateSubst_validate,
    otv_LigatureSubst_validate,
    otv_ContextSubst_validate,
    otv_ChainContextSubst_validate,
    otv_ExtensionSubst_validate,
    otv_ReverseChainSingleSubst_validate
  };


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                          GSUB TABLE                           *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  /* sets otvalid->type_count  */
  /* sets otvalid->type_funcs  */
  /* sets otvalid->glyph_count */

  FT_LOCAL_DEF( void )
  otv_GSUB_validate( FT_Bytes      table,
                     FT_UInt       glyph_count,
                     FT_Validator  ftvalid )
  {
    OTV_ValidatorRec  otvalidrec;
    OTV_Validator     otvalid = &otvalidrec;
    FT_Bytes          p       = table;
    FT_UInt           ScriptList, FeatureList, LookupList;


    otvalid->root = ftvalid;

    FT_TRACE3(( "validating GSUB table\n" ));
    OTV_INIT;

    OTV_LIMIT_CHECK( 10 );

    if ( FT_NEXT_ULONG( p ) != 0x10000UL )      /* Version */
      FT_INVALID_FORMAT;

    ScriptList  = FT_NEXT_USHORT( p );
    FeatureList = FT_NEXT_USHORT( p );
    LookupList  = FT_NEXT_USHORT( p );

    otvalid->type_count  = 8;
    otvalid->type_funcs  = (OTV_Validate_Func*)otv_gsub_validate_funcs;
    otvalid->glyph_count = glyph_count;

    otv_LookupList_validate( table + LookupList,
                             otvalid );
    otv_FeatureList_validate( table + FeatureList, table + LookupList,
                              otvalid );
    otv_ScriptList_validate( table + ScriptList, table + FeatureList,
                             otvalid );

    FT_TRACE4(( "\n" ));
  }


/* END */
