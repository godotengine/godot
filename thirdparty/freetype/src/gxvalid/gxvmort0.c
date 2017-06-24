/***************************************************************************/
/*                                                                         */
/*  gxvmort0.c                                                             */
/*                                                                         */
/*    TrueTypeGX/AAT mort table validation                                 */
/*    body for type0 (Indic Script Rearrangement) subtable.                */
/*                                                                         */
/*  Copyright 2005-2017 by                                                 */
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


#include "gxvmort.h"


  /*************************************************************************/
  /*                                                                       */
  /* The macro FT_COMPONENT is used in trace mode.  It is an implicit      */
  /* parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log  */
  /* messages during execution.                                            */
  /*                                                                       */
#undef  FT_COMPONENT
#define FT_COMPONENT  trace_gxvmort


  static const char* GXV_Mort_IndicScript_Msg[] =
  {
    "no change",
    "Ax => xA",
    "xD => Dx",
    "AxD => DxA",
    "ABx => xAB",
    "ABx => xBA",
    "xCD => CDx",
    "xCD => DCx",
    "AxCD => CDxA",
    "AxCD => DCxA",
    "ABxD => DxAB",
    "ABxD => DxBA",
    "ABxCD => CDxAB",
    "ABxCD => CDxBA",
    "ABxCD => DCxAB",
    "ABxCD => DCxBA",

  };


  static void
  gxv_mort_subtable_type0_entry_validate(
    FT_Byte                         state,
    FT_UShort                       flags,
    GXV_StateTable_GlyphOffsetCPtr  glyphOffset_p,
    FT_Bytes                        table,
    FT_Bytes                        limit,
    GXV_Validator                   gxvalid )
  {
    FT_UShort  markFirst;
    FT_UShort  dontAdvance;
    FT_UShort  markLast;
    FT_UShort  reserved;
    FT_UShort  verb = 0;

    FT_UNUSED( state );
    FT_UNUSED( table );
    FT_UNUSED( limit );

    FT_UNUSED( GXV_Mort_IndicScript_Msg[verb] ); /* for the non-debugging */
    FT_UNUSED( glyphOffset_p );                  /* case                  */


    markFirst   = (FT_UShort)( ( flags >> 15 ) & 1 );
    dontAdvance = (FT_UShort)( ( flags >> 14 ) & 1 );
    markLast    = (FT_UShort)( ( flags >> 13 ) & 1 );

    reserved = (FT_UShort)( flags & 0x1FF0 );
    verb     = (FT_UShort)( flags & 0x000F );

    GXV_TRACE(( "  IndicScript MorphRule for glyphOffset 0x%04x",
                glyphOffset_p->u ));
    GXV_TRACE(( " markFirst=%01d", markFirst ));
    GXV_TRACE(( " dontAdvance=%01d", dontAdvance ));
    GXV_TRACE(( " markLast=%01d", markLast ));
    GXV_TRACE(( " %02d", verb ));
    GXV_TRACE(( " %s\n", GXV_Mort_IndicScript_Msg[verb] ));

    if ( markFirst > 0 && markLast > 0 )
    {
      GXV_TRACE(( "  [odd] a glyph is marked as the first and last"
                  "  in Indic rearrangement\n" ));
      GXV_SET_ERR_IF_PARANOID( FT_INVALID_DATA );
    }

    if ( markFirst > 0 && dontAdvance > 0 )
    {
      GXV_TRACE(( "  [odd] the first glyph is marked as dontAdvance"
                  " in Indic rearrangement\n" ));
      GXV_SET_ERR_IF_PARANOID( FT_INVALID_DATA );
    }

    if ( 0 < reserved )
    {
      GXV_TRACE(( " non-zero bits found in reserved range\n" ));
      GXV_SET_ERR_IF_PARANOID( FT_INVALID_DATA );
    }
    else
      GXV_TRACE(( "\n" ));
  }


  FT_LOCAL_DEF( void )
  gxv_mort_subtable_type0_validate( FT_Bytes       table,
                                    FT_Bytes       limit,
                                    GXV_Validator  gxvalid )
  {
    FT_Bytes  p = table;


    GXV_NAME_ENTER(
      "mort chain subtable type0 (Indic-Script Rearrangement)" );

    GXV_LIMIT_CHECK( GXV_STATETABLE_HEADER_SIZE );

    gxvalid->statetable.optdata               = NULL;
    gxvalid->statetable.optdata_load_func     = NULL;
    gxvalid->statetable.subtable_setup_func   = NULL;
    gxvalid->statetable.entry_glyphoffset_fmt = GXV_GLYPHOFFSET_NONE;
    gxvalid->statetable.entry_validate_func =
      gxv_mort_subtable_type0_entry_validate;

    gxv_StateTable_validate( p, limit, gxvalid );

    GXV_EXIT;
  }


/* END */
