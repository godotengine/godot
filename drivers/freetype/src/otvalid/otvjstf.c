/***************************************************************************/
/*                                                                         */
/*  otvjstf.c                                                              */
/*                                                                         */
/*    OpenType JSTF table validation (body).                               */
/*                                                                         */
/*  Copyright 2004, 2007 by                                                */
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
#include "otvgpos.h"


  /*************************************************************************/
  /*                                                                       */
  /* The macro FT_COMPONENT is used in trace mode.  It is an implicit      */
  /* parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log  */
  /* messages during execution.                                            */
  /*                                                                       */
#undef  FT_COMPONENT
#define FT_COMPONENT  trace_otvjstf


#define JstfPriorityFunc  otv_JstfPriority_validate
#define JstfLookupFunc    otv_GPOS_subtable_validate

  /* uses valid->extra1 (GSUB lookup count) */
  /* uses valid->extra2 (GPOS lookup count) */
  /* sets valid->extra1 (counter)           */

  static void
  otv_JstfPriority_validate( FT_Bytes       table,
                             OTV_Validator  valid )
  {
    FT_Bytes  p = table;
    FT_UInt   table_size;
    FT_UInt   gsub_lookup_count, gpos_lookup_count;

    OTV_OPTIONAL_TABLE( ShrinkageEnableGSUB  );
    OTV_OPTIONAL_TABLE( ShrinkageDisableGSUB );
    OTV_OPTIONAL_TABLE( ShrinkageEnableGPOS  );
    OTV_OPTIONAL_TABLE( ShrinkageDisableGPOS );
    OTV_OPTIONAL_TABLE( ExtensionEnableGSUB  );
    OTV_OPTIONAL_TABLE( ExtensionDisableGSUB );
    OTV_OPTIONAL_TABLE( ExtensionEnableGPOS  );
    OTV_OPTIONAL_TABLE( ExtensionDisableGPOS );
    OTV_OPTIONAL_TABLE( ShrinkageJstfMax );
    OTV_OPTIONAL_TABLE( ExtensionJstfMax );


    OTV_ENTER;
    OTV_TRACE(( "JstfPriority table\n" ));

    OTV_LIMIT_CHECK( 20 );

    gsub_lookup_count = valid->extra1;
    gpos_lookup_count = valid->extra2;

    table_size = 20;

    valid->extra1 = gsub_lookup_count;

    OTV_OPTIONAL_OFFSET( ShrinkageEnableGSUB );
    OTV_SIZE_CHECK( ShrinkageEnableGSUB );
    if ( ShrinkageEnableGSUB )
      otv_x_ux( table + ShrinkageEnableGSUB, valid );

    OTV_OPTIONAL_OFFSET( ShrinkageDisableGSUB );
    OTV_SIZE_CHECK( ShrinkageDisableGSUB );
    if ( ShrinkageDisableGSUB )
      otv_x_ux( table + ShrinkageDisableGSUB, valid );

    valid->extra1 = gpos_lookup_count;

    OTV_OPTIONAL_OFFSET( ShrinkageEnableGPOS );
    OTV_SIZE_CHECK( ShrinkageEnableGPOS );
    if ( ShrinkageEnableGPOS )
      otv_x_ux( table + ShrinkageEnableGPOS, valid );

    OTV_OPTIONAL_OFFSET( ShrinkageDisableGPOS );
    OTV_SIZE_CHECK( ShrinkageDisableGPOS );
    if ( ShrinkageDisableGPOS )
      otv_x_ux( table + ShrinkageDisableGPOS, valid );

    OTV_OPTIONAL_OFFSET( ShrinkageJstfMax );
    OTV_SIZE_CHECK( ShrinkageJstfMax );
    if ( ShrinkageJstfMax )
    {
      /* XXX: check lookup types? */
      OTV_NEST2( JstfMax, JstfLookup );
      OTV_RUN( table + ShrinkageJstfMax, valid );
    }

    valid->extra1 = gsub_lookup_count;

    OTV_OPTIONAL_OFFSET( ExtensionEnableGSUB );
    OTV_SIZE_CHECK( ExtensionEnableGSUB );
    if ( ExtensionEnableGSUB )
      otv_x_ux( table + ExtensionEnableGSUB, valid );

    OTV_OPTIONAL_OFFSET( ExtensionDisableGSUB );
    OTV_SIZE_CHECK( ExtensionDisableGSUB );
    if ( ExtensionDisableGSUB )
      otv_x_ux( table + ExtensionDisableGSUB, valid );

    valid->extra1 = gpos_lookup_count;

    OTV_OPTIONAL_OFFSET( ExtensionEnableGPOS );
    OTV_SIZE_CHECK( ExtensionEnableGPOS );
    if ( ExtensionEnableGPOS )
      otv_x_ux( table + ExtensionEnableGPOS, valid );

    OTV_OPTIONAL_OFFSET( ExtensionDisableGPOS );
    OTV_SIZE_CHECK( ExtensionDisableGPOS );
    if ( ExtensionDisableGPOS )
      otv_x_ux( table + ExtensionDisableGPOS, valid );

    OTV_OPTIONAL_OFFSET( ExtensionJstfMax );
    OTV_SIZE_CHECK( ExtensionJstfMax );
    if ( ExtensionJstfMax )
    {
      /* XXX: check lookup types? */
      OTV_NEST2( JstfMax, JstfLookup );
      OTV_RUN( table + ExtensionJstfMax, valid );
    }

    valid->extra1 = gsub_lookup_count;
    valid->extra2 = gpos_lookup_count;

    OTV_EXIT;
  }


  /* sets valid->extra (glyph count)               */
  /* sets valid->func1 (otv_JstfPriority_validate) */

  static void
  otv_JstfScript_validate( FT_Bytes       table,
                           OTV_Validator  valid )
  {
    FT_Bytes  p = table;
    FT_UInt   table_size;
    FT_UInt   JstfLangSysCount;

    OTV_OPTIONAL_TABLE( ExtGlyph );
    OTV_OPTIONAL_TABLE( DefJstfLangSys );


    OTV_NAME_ENTER( "JstfScript" );

    OTV_LIMIT_CHECK( 6 );
    OTV_OPTIONAL_OFFSET( ExtGlyph );
    OTV_OPTIONAL_OFFSET( DefJstfLangSys );
    JstfLangSysCount = FT_NEXT_USHORT( p );

    OTV_TRACE(( " (JstfLangSysCount = %d)\n", JstfLangSysCount ));

    table_size = JstfLangSysCount * 6 + 6;

    OTV_SIZE_CHECK( ExtGlyph );
    if ( ExtGlyph )
    {
      valid->extra1 = valid->glyph_count;
      OTV_NEST1( ExtenderGlyph );
      OTV_RUN( table + ExtGlyph, valid );
    }

    OTV_SIZE_CHECK( DefJstfLangSys );
    if ( DefJstfLangSys )
    {
      OTV_NEST2( JstfLangSys, JstfPriority );
      OTV_RUN( table + DefJstfLangSys, valid );
    }

    OTV_LIMIT_CHECK( 6 * JstfLangSysCount );

    /* JstfLangSysRecord */
    OTV_NEST2( JstfLangSys, JstfPriority );
    for ( ; JstfLangSysCount > 0; JstfLangSysCount-- )
    {
      p += 4;       /* skip JstfLangSysTag */

      OTV_RUN( table + FT_NEXT_USHORT( p ), valid );
    }

    OTV_EXIT;
  }


  /* sets valid->extra1 (GSUB lookup count) */
  /* sets valid->extra2 (GPOS lookup count) */
  /* sets valid->glyph_count                */

  FT_LOCAL_DEF( void )
  otv_JSTF_validate( FT_Bytes      table,
                     FT_Bytes      gsub,
                     FT_Bytes      gpos,
                     FT_UInt       glyph_count,
                     FT_Validator  ftvalid )
  {
    OTV_ValidatorRec  validrec;
    OTV_Validator     valid = &validrec;
    FT_Bytes          p     = table;
    FT_UInt           JstfScriptCount;


    valid->root = ftvalid;

    FT_TRACE3(( "validating JSTF table\n" ));
    OTV_INIT;

    OTV_LIMIT_CHECK( 6 );

    if ( FT_NEXT_ULONG( p ) != 0x10000UL )      /* Version */
      FT_INVALID_FORMAT;

    JstfScriptCount = FT_NEXT_USHORT( p );

    FT_TRACE3(( " (JstfScriptCount = %d)\n", JstfScriptCount ));

    OTV_LIMIT_CHECK( JstfScriptCount * 6 );

    if ( gsub )
      valid->extra1 = otv_GSUBGPOS_get_Lookup_count( gsub );
    else
      valid->extra1 = 0;

    if ( gpos )
      valid->extra2 = otv_GSUBGPOS_get_Lookup_count( gpos );
    else
      valid->extra2 = 0;

    valid->glyph_count = glyph_count;

    /* JstfScriptRecord */
    for ( ; JstfScriptCount > 0; JstfScriptCount-- )
    {
      p += 4;       /* skip JstfScriptTag */

      /* JstfScript */
      otv_JstfScript_validate( table + FT_NEXT_USHORT( p ), valid );
    }

    FT_TRACE4(( "\n" ));
  }


/* END */
