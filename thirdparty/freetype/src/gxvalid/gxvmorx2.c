/***************************************************************************/
/*                                                                         */
/*  gxvmorx2.c                                                             */
/*                                                                         */
/*    TrueTypeGX/AAT morx table validation                                 */
/*    body for type2 (Ligature Substitution) subtable.                     */
/*                                                                         */
/*  Copyright 2005-2016 by                                                 */
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


#include "gxvmorx.h"


  /*************************************************************************/
  /*                                                                       */
  /* The macro FT_COMPONENT is used in trace mode.  It is an implicit      */
  /* parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log  */
  /* messages during execution.                                            */
  /*                                                                       */
#undef  FT_COMPONENT
#define FT_COMPONENT  trace_gxvmorx


  typedef struct  GXV_morx_subtable_type2_StateOptRec_
  {
    FT_ULong  ligActionTable;
    FT_ULong  componentTable;
    FT_ULong  ligatureTable;
    FT_ULong  ligActionTable_length;
    FT_ULong  componentTable_length;
    FT_ULong  ligatureTable_length;

  }  GXV_morx_subtable_type2_StateOptRec,
    *GXV_morx_subtable_type2_StateOptRecData;


#define GXV_MORX_SUBTABLE_TYPE2_HEADER_SIZE \
          ( GXV_XSTATETABLE_HEADER_SIZE + 4 + 4 + 4 )


  static void
  gxv_morx_subtable_type2_opttable_load( FT_Bytes       table,
                                         FT_Bytes       limit,
                                         GXV_Validator  gxvalid )
  {
    FT_Bytes  p = table;

    GXV_morx_subtable_type2_StateOptRecData  optdata =
      (GXV_morx_subtable_type2_StateOptRecData)gxvalid->xstatetable.optdata;


    GXV_LIMIT_CHECK( 4 + 4 + 4 );
    optdata->ligActionTable = FT_NEXT_ULONG( p );
    optdata->componentTable = FT_NEXT_ULONG( p );
    optdata->ligatureTable  = FT_NEXT_ULONG( p );

    GXV_TRACE(( "offset to ligActionTable=0x%08x\n",
                optdata->ligActionTable ));
    GXV_TRACE(( "offset to componentTable=0x%08x\n",
                optdata->componentTable ));
    GXV_TRACE(( "offset to ligatureTable=0x%08x\n",
                optdata->ligatureTable ));
  }


  static void
  gxv_morx_subtable_type2_subtable_setup( FT_ULong       table_size,
                                          FT_ULong       classTable,
                                          FT_ULong       stateArray,
                                          FT_ULong       entryTable,
                                          FT_ULong*      classTable_length_p,
                                          FT_ULong*      stateArray_length_p,
                                          FT_ULong*      entryTable_length_p,
                                          GXV_Validator  gxvalid )
  {
    FT_ULong   o[6];
    FT_ULong*  l[6];
    FT_ULong   buff[7];

    GXV_morx_subtable_type2_StateOptRecData  optdata =
      (GXV_morx_subtable_type2_StateOptRecData)gxvalid->xstatetable.optdata;


    GXV_NAME_ENTER( "subtable boundaries setup" );

    o[0] = classTable;
    o[1] = stateArray;
    o[2] = entryTable;
    o[3] = optdata->ligActionTable;
    o[4] = optdata->componentTable;
    o[5] = optdata->ligatureTable;
    l[0] = classTable_length_p;
    l[1] = stateArray_length_p;
    l[2] = entryTable_length_p;
    l[3] = &(optdata->ligActionTable_length);
    l[4] = &(optdata->componentTable_length);
    l[5] = &(optdata->ligatureTable_length);

    gxv_set_length_by_ulong_offset( o, l, buff, 6, table_size, gxvalid );

    GXV_TRACE(( "classTable: offset=0x%08x length=0x%08x\n",
                classTable, *classTable_length_p ));
    GXV_TRACE(( "stateArray: offset=0x%08x length=0x%08x\n",
                stateArray, *stateArray_length_p ));
    GXV_TRACE(( "entryTable: offset=0x%08x length=0x%08x\n",
                entryTable, *entryTable_length_p ));
    GXV_TRACE(( "ligActionTable: offset=0x%08x length=0x%08x\n",
                optdata->ligActionTable,
                optdata->ligActionTable_length ));
    GXV_TRACE(( "componentTable: offset=0x%08x length=0x%08x\n",
                optdata->componentTable,
                optdata->componentTable_length ));
    GXV_TRACE(( "ligatureTable:  offset=0x%08x length=0x%08x\n",
                optdata->ligatureTable,
                optdata->ligatureTable_length ));

    GXV_EXIT;
  }


#define GXV_MORX_LIGACTION_ENTRY_SIZE  4


  static void
  gxv_morx_subtable_type2_ligActionIndex_validate(
    FT_Bytes       table,
    FT_UShort      ligActionIndex,
    GXV_Validator  gxvalid )
  {
    /* access ligActionTable */
    GXV_morx_subtable_type2_StateOptRecData optdata =
      (GXV_morx_subtable_type2_StateOptRecData)gxvalid->xstatetable.optdata;

    FT_Bytes lat_base  = table + optdata->ligActionTable;
    FT_Bytes p         = lat_base +
                         ligActionIndex * GXV_MORX_LIGACTION_ENTRY_SIZE;
    FT_Bytes lat_limit = lat_base + optdata->ligActionTable;


    if ( p < lat_base )
    {
      GXV_TRACE(( "p < lat_base (%d byte rewind)\n", lat_base - p ));
      FT_INVALID_OFFSET;
    }
    else if ( lat_limit < p )
    {
      GXV_TRACE(( "lat_limit < p (%d byte overrun)\n", p - lat_limit ));
      FT_INVALID_OFFSET;
    }

    {
      /* validate entry in ligActionTable */
      FT_ULong   lig_action;
#ifdef GXV_LOAD_UNUSED_VARS
      FT_UShort  last;
      FT_UShort  store;
#endif
      FT_ULong   offset;
      FT_Long    gid_limit;


      lig_action = FT_NEXT_ULONG( p );
#ifdef GXV_LOAD_UNUSED_VARS
      last       = (FT_UShort)( ( lig_action >> 31 ) & 1 );
      store      = (FT_UShort)( ( lig_action >> 30 ) & 1 );
#endif

      offset = lig_action & 0x3FFFFFFFUL;

      /* this offset is 30-bit signed value to add to GID */
      /* it is different from the location offset in mort */
      if ( ( offset & 0x3FFF0000UL ) == 0x3FFF0000UL )
      { /* negative offset */
        gid_limit = gxvalid->face->num_glyphs -
                    (FT_Long)( offset & 0x0000FFFFUL );
        if ( gid_limit > 0 )
          return;

        GXV_TRACE(( "ligature action table includes"
                    " too negative offset moving all GID"
                    " below defined range: 0x%04x\n",
                    offset & 0xFFFFU ));
        GXV_SET_ERR_IF_PARANOID( FT_INVALID_OFFSET );
      }
      else if ( ( offset & 0x3FFF0000UL ) == 0x00000000UL )
      { /* positive offset */
        if ( (FT_Long)offset < gxvalid->face->num_glyphs )
          return;

        GXV_TRACE(( "ligature action table includes"
                    " too large offset moving all GID"
                    " over defined range: 0x%04x\n",
                    offset & 0xFFFFU ));
        GXV_SET_ERR_IF_PARANOID( FT_INVALID_OFFSET );
      }

      GXV_TRACE(( "ligature action table includes"
                  " invalid offset to add to 16-bit GID:"
                  " 0x%08x\n", offset ));
      GXV_SET_ERR_IF_PARANOID( FT_INVALID_OFFSET );
    }
  }


  static void
  gxv_morx_subtable_type2_entry_validate(
    FT_UShort                       state,
    FT_UShort                       flags,
    GXV_StateTable_GlyphOffsetCPtr  glyphOffset_p,
    FT_Bytes                        table,
    FT_Bytes                        limit,
    GXV_Validator                   gxvalid )
  {
#ifdef GXV_LOAD_UNUSED_VARS
    FT_UShort  setComponent;
    FT_UShort  dontAdvance;
    FT_UShort  performAction;
#endif
    FT_UShort  reserved;
    FT_UShort  ligActionIndex;

    FT_UNUSED( state );
    FT_UNUSED( limit );


#ifdef GXV_LOAD_UNUSED_VARS
    setComponent   = (FT_UShort)( ( flags >> 15 ) & 1 );
    dontAdvance    = (FT_UShort)( ( flags >> 14 ) & 1 );
    performAction  = (FT_UShort)( ( flags >> 13 ) & 1 );
#endif

    reserved       = (FT_UShort)( flags & 0x1FFF );
    ligActionIndex = glyphOffset_p->u;

    if ( reserved > 0 )
      GXV_TRACE(( "  reserved 14bit is non-zero\n" ));

    if ( 0 < ligActionIndex )
      gxv_morx_subtable_type2_ligActionIndex_validate(
        table, ligActionIndex, gxvalid );
  }


  static void
  gxv_morx_subtable_type2_ligatureTable_validate( FT_Bytes       table,
                                                  GXV_Validator  gxvalid )
  {
    GXV_morx_subtable_type2_StateOptRecData  optdata =
      (GXV_morx_subtable_type2_StateOptRecData)gxvalid->xstatetable.optdata;

    FT_Bytes p     = table + optdata->ligatureTable;
    FT_Bytes limit = table + optdata->ligatureTable
                           + optdata->ligatureTable_length;


    GXV_NAME_ENTER( "morx chain subtable type2 - substitutionTable" );

    if ( 0 != optdata->ligatureTable )
    {
      /* Apple does not give specification of ligatureTable format */
      while ( p < limit )
      {
        FT_UShort  lig_gid;


        GXV_LIMIT_CHECK( 2 );
        lig_gid = FT_NEXT_USHORT( p );
        if ( lig_gid < gxvalid->face->num_glyphs )
          GXV_SET_ERR_IF_PARANOID( FT_INVALID_GLYPH_ID );
      }
    }

    GXV_EXIT;
  }


  FT_LOCAL_DEF( void )
  gxv_morx_subtable_type2_validate( FT_Bytes       table,
                                    FT_Bytes       limit,
                                    GXV_Validator  gxvalid )
  {
    FT_Bytes  p = table;

    GXV_morx_subtable_type2_StateOptRec  lig_rec;


    GXV_NAME_ENTER( "morx chain subtable type2 (Ligature Substitution)" );

    GXV_LIMIT_CHECK( GXV_MORX_SUBTABLE_TYPE2_HEADER_SIZE );

    gxvalid->xstatetable.optdata =
      &lig_rec;
    gxvalid->xstatetable.optdata_load_func =
      gxv_morx_subtable_type2_opttable_load;
    gxvalid->xstatetable.subtable_setup_func =
      gxv_morx_subtable_type2_subtable_setup;
    gxvalid->xstatetable.entry_glyphoffset_fmt =
      GXV_GLYPHOFFSET_USHORT;
    gxvalid->xstatetable.entry_validate_func =
      gxv_morx_subtable_type2_entry_validate;

    gxv_XStateTable_validate( p, limit, gxvalid );

#if 0
    p += gxvalid->subtable_length;
#endif
    gxv_morx_subtable_type2_ligatureTable_validate( table, gxvalid );

    GXV_EXIT;
  }


/* END */
