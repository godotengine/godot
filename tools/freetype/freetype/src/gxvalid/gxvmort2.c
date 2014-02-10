/***************************************************************************/
/*                                                                         */
/*  gxvmort2.c                                                             */
/*                                                                         */
/*    TrueTypeGX/AAT mort table validation                                 */
/*    body for type2 (Ligature Substitution) subtable.                     */
/*                                                                         */
/*  Copyright 2005 by suzuki toshiya, Masatake YAMATO, Red Hat K.K.,       */
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


  typedef struct  GXV_mort_subtable_type2_StateOptRec_
  {
    FT_UShort  ligActionTable;
    FT_UShort  componentTable;
    FT_UShort  ligatureTable;
    FT_UShort  ligActionTable_length;
    FT_UShort  componentTable_length;
    FT_UShort  ligatureTable_length;

  }  GXV_mort_subtable_type2_StateOptRec,
    *GXV_mort_subtable_type2_StateOptRecData;

#define GXV_MORT_SUBTABLE_TYPE2_HEADER_SIZE \
          ( GXV_STATETABLE_HEADER_SIZE + 2 + 2 + 2 )


  static void
  gxv_mort_subtable_type2_opttable_load( FT_Bytes       table,
                                         FT_Bytes       limit,
                                         GXV_Validator  valid )
  {
    FT_Bytes p = table;
    GXV_mort_subtable_type2_StateOptRecData  optdata =
      (GXV_mort_subtable_type2_StateOptRecData)valid->statetable.optdata;


    GXV_LIMIT_CHECK( 2 + 2 + 2 );
    optdata->ligActionTable = FT_NEXT_USHORT( p );
    optdata->componentTable = FT_NEXT_USHORT( p );
    optdata->ligatureTable  = FT_NEXT_USHORT( p );

    GXV_TRACE(( "offset to ligActionTable=0x%04x\n",
                optdata->ligActionTable ));
    GXV_TRACE(( "offset to componentTable=0x%04x\n",
                optdata->componentTable ));
    GXV_TRACE(( "offset to ligatureTable=0x%04x\n",
                optdata->ligatureTable ));
  }


  static void
  gxv_mort_subtable_type2_subtable_setup( FT_UShort      table_size,
                                          FT_UShort      classTable,
                                          FT_UShort      stateArray,
                                          FT_UShort      entryTable,
                                          FT_UShort      *classTable_length_p,
                                          FT_UShort      *stateArray_length_p,
                                          FT_UShort      *entryTable_length_p,
                                          GXV_Validator  valid )
  {
    FT_UShort  o[6];
    FT_UShort  *l[6];
    FT_UShort  buff[7];

    GXV_mort_subtable_type2_StateOptRecData  optdata =
      (GXV_mort_subtable_type2_StateOptRecData)valid->statetable.optdata;


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

    gxv_set_length_by_ushort_offset( o, l, buff, 6, table_size, valid );

    GXV_TRACE(( "classTable: offset=0x%04x length=0x%04x\n",
                classTable, *classTable_length_p ));
    GXV_TRACE(( "stateArray: offset=0x%04x length=0x%04x\n",
                stateArray, *stateArray_length_p ));
    GXV_TRACE(( "entryTable: offset=0x%04x length=0x%04x\n",
                entryTable, *entryTable_length_p ));
    GXV_TRACE(( "ligActionTable: offset=0x%04x length=0x%04x\n",
                optdata->ligActionTable,
                optdata->ligActionTable_length ));
    GXV_TRACE(( "componentTable: offset=0x%04x length=0x%04x\n",
                optdata->componentTable,
                optdata->componentTable_length ));
    GXV_TRACE(( "ligatureTable:  offset=0x%04x length=0x%04x\n",
                optdata->ligatureTable,
                optdata->ligatureTable_length ));

    GXV_EXIT;
  }


  static void
  gxv_mort_subtable_type2_ligActionOffset_validate(
    FT_Bytes       table,
    FT_UShort      ligActionOffset,
    GXV_Validator  valid )
  {
    /* access ligActionTable */
    GXV_mort_subtable_type2_StateOptRecData  optdata =
      (GXV_mort_subtable_type2_StateOptRecData)valid->statetable.optdata;

    FT_Bytes lat_base  = table + optdata->ligActionTable;
    FT_Bytes p         = table + ligActionOffset;
    FT_Bytes lat_limit = lat_base + optdata->ligActionTable;


    GXV_32BIT_ALIGNMENT_VALIDATE( ligActionOffset );
    if ( p < lat_base )
    {
      GXV_TRACE(( "too short offset 0x%04x: p < lat_base (%d byte rewind)\n",
                  ligActionOffset, lat_base - p ));

      /* FontValidator, ftxvalidator, ftxdumperfuser warn but continue */
      GXV_SET_ERR_IF_PARANOID( FT_INVALID_OFFSET );
    }
    else if ( lat_limit < p )
    {
      GXV_TRACE(( "too large offset 0x%04x: lat_limit < p (%d byte overrun)\n",
                  ligActionOffset, p - lat_limit ));

      /* FontValidator, ftxvalidator, ftxdumperfuser warn but continue */
      GXV_SET_ERR_IF_PARANOID( FT_INVALID_OFFSET );
    }
    else
    {
      /* validate entry in ligActionTable */
      FT_ULong   lig_action;
#ifdef GXV_LOAD_UNUSED_VARS
      FT_UShort  last;
      FT_UShort  store;
#endif
      FT_ULong   offset;


      lig_action = FT_NEXT_ULONG( p );
#ifdef GXV_LOAD_UNUSED_VARS
      last   = (FT_UShort)( ( lig_action >> 31 ) & 1 );
      store  = (FT_UShort)( ( lig_action >> 30 ) & 1 );
#endif

      /* Apple spec defines this offset as a word offset */
      offset = lig_action & 0x3FFFFFFFUL;
      if ( offset * 2 < optdata->ligatureTable )
      {
        GXV_TRACE(( "too short offset 0x%08x:"
                    " 2 x offset < ligatureTable (%d byte rewind)\n",
                     offset, optdata->ligatureTable - offset * 2 ));

        GXV_SET_ERR_IF_PARANOID( FT_INVALID_OFFSET );
      } else if ( offset * 2 >
                  optdata->ligatureTable + optdata->ligatureTable_length )
      {
        GXV_TRACE(( "too long offset 0x%08x:"
                    " 2 x offset > ligatureTable + ligatureTable_length"
                    " (%d byte overrun)\n",
                     offset,
                     optdata->ligatureTable + optdata->ligatureTable_length
                     - offset * 2 ));

        GXV_SET_ERR_IF_PARANOID( FT_INVALID_OFFSET );
      }
    }
  }


  static void
  gxv_mort_subtable_type2_entry_validate(
    FT_Byte                         state,
    FT_UShort                       flags,
    GXV_StateTable_GlyphOffsetCPtr  glyphOffset_p,
    FT_Bytes                        table,
    FT_Bytes                        limit,
    GXV_Validator                   valid )
  {
#ifdef GXV_LOAD_UNUSED_VARS
    FT_UShort setComponent;
    FT_UShort dontAdvance;
#endif
    FT_UShort offset;

    FT_UNUSED( state );
    FT_UNUSED( glyphOffset_p );
    FT_UNUSED( limit );


#ifdef GXV_LOAD_UNUSED_VARS
    setComponent = (FT_UShort)( ( flags >> 15 ) & 1 );
    dontAdvance  = (FT_UShort)( ( flags >> 14 ) & 1 );
#endif

    offset = (FT_UShort)( flags & 0x3FFFU );

    if ( 0 < offset )
      gxv_mort_subtable_type2_ligActionOffset_validate( table, offset,
                                                        valid );
  }


  static void
  gxv_mort_subtable_type2_ligatureTable_validate( FT_Bytes       table,
                                                  GXV_Validator  valid )
  {
    GXV_mort_subtable_type2_StateOptRecData  optdata =
      (GXV_mort_subtable_type2_StateOptRecData)valid->statetable.optdata;

    FT_Bytes p     = table + optdata->ligatureTable;
    FT_Bytes limit = table + optdata->ligatureTable
                           + optdata->ligatureTable_length;


    GXV_NAME_ENTER( "mort chain subtable type2 - substitutionTable" );
    if ( 0 != optdata->ligatureTable )
    {
      /* Apple does not give specification of ligatureTable format */
      while ( p < limit )
      {
        FT_UShort  lig_gid;


        GXV_LIMIT_CHECK( 2 );
        lig_gid = FT_NEXT_USHORT( p );

        if ( valid->face->num_glyphs < lig_gid )
          GXV_SET_ERR_IF_PARANOID( FT_INVALID_GLYPH_ID );
      }
    }
    GXV_EXIT;
  }


  FT_LOCAL_DEF( void )
  gxv_mort_subtable_type2_validate( FT_Bytes       table,
                                    FT_Bytes       limit,
                                    GXV_Validator  valid )
  {
    FT_Bytes  p = table;

    GXV_mort_subtable_type2_StateOptRec  lig_rec;


    GXV_NAME_ENTER( "mort chain subtable type2 (Ligature Substitution)" );

    GXV_LIMIT_CHECK( GXV_MORT_SUBTABLE_TYPE2_HEADER_SIZE );

    valid->statetable.optdata =
      &lig_rec;
    valid->statetable.optdata_load_func =
      gxv_mort_subtable_type2_opttable_load;
    valid->statetable.subtable_setup_func =
      gxv_mort_subtable_type2_subtable_setup;
    valid->statetable.entry_glyphoffset_fmt =
      GXV_GLYPHOFFSET_NONE;
    valid->statetable.entry_validate_func =
      gxv_mort_subtable_type2_entry_validate;

    gxv_StateTable_validate( p, limit, valid );

    p += valid->subtable_length;
    gxv_mort_subtable_type2_ligatureTable_validate( table, valid );

    valid->subtable_length = p - table;

    GXV_EXIT;
  }


/* END */
