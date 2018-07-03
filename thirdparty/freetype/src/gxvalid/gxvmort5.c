/***************************************************************************/
/*                                                                         */
/*  gxvmort5.c                                                             */
/*                                                                         */
/*    TrueTypeGX/AAT mort table validation                                 */
/*    body for type5 (Contextual Glyph Insertion) subtable.                */
/*                                                                         */
/*  Copyright 2005-2018 by                                                 */
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


  /*
   * mort subtable type5 (Contextual Glyph Insertion)
   * has the format of StateTable with insertion-glyph-list,
   * but without name.  The offset is given by glyphOffset in
   * entryTable.  There is no table location declaration
   * like xxxTable.
   */

  typedef struct  GXV_mort_subtable_type5_StateOptRec_
  {
    FT_UShort   classTable;
    FT_UShort   stateArray;
    FT_UShort   entryTable;

#define GXV_MORT_SUBTABLE_TYPE5_HEADER_SIZE  GXV_STATETABLE_HEADER_SIZE

    FT_UShort*  classTable_length_p;
    FT_UShort*  stateArray_length_p;
    FT_UShort*  entryTable_length_p;

  }  GXV_mort_subtable_type5_StateOptRec,
    *GXV_mort_subtable_type5_StateOptRecData;


  FT_LOCAL_DEF( void )
  gxv_mort_subtable_type5_subtable_setup( FT_UShort      table_size,
                                          FT_UShort      classTable,
                                          FT_UShort      stateArray,
                                          FT_UShort      entryTable,
                                          FT_UShort*     classTable_length_p,
                                          FT_UShort*     stateArray_length_p,
                                          FT_UShort*     entryTable_length_p,
                                          GXV_Validator  gxvalid )
  {
    GXV_mort_subtable_type5_StateOptRecData  optdata =
      (GXV_mort_subtable_type5_StateOptRecData)gxvalid->statetable.optdata;


    gxv_StateTable_subtable_setup( table_size,
                                   classTable,
                                   stateArray,
                                   entryTable,
                                   classTable_length_p,
                                   stateArray_length_p,
                                   entryTable_length_p,
                                   gxvalid );

    optdata->classTable = classTable;
    optdata->stateArray = stateArray;
    optdata->entryTable = entryTable;

    optdata->classTable_length_p = classTable_length_p;
    optdata->stateArray_length_p = stateArray_length_p;
    optdata->entryTable_length_p = entryTable_length_p;
  }


  static void
  gxv_mort_subtable_type5_InsertList_validate( FT_UShort      offset,
                                               FT_UShort      count,
                                               FT_Bytes       table,
                                               FT_Bytes       limit,
                                               GXV_Validator  gxvalid )
  {
    /*
     * We don't know the range of insertion-glyph-list.
     * Set range by whole of state table.
     */
    FT_Bytes  p = table + offset;

    GXV_mort_subtable_type5_StateOptRecData  optdata =
      (GXV_mort_subtable_type5_StateOptRecData)gxvalid->statetable.optdata;

    if ( optdata->classTable < offset                                   &&
         offset < optdata->classTable + *(optdata->classTable_length_p) )
      GXV_TRACE(( " offset runs into ClassTable" ));
    if ( optdata->stateArray < offset                                   &&
         offset < optdata->stateArray + *(optdata->stateArray_length_p) )
      GXV_TRACE(( " offset runs into StateArray" ));
    if ( optdata->entryTable < offset                                   &&
         offset < optdata->entryTable + *(optdata->entryTable_length_p) )
      GXV_TRACE(( " offset runs into EntryTable" ));

#ifndef GXV_LOAD_TRACE_VARS
    GXV_LIMIT_CHECK( count * 2 );
#else
    while ( p < table + offset + ( count * 2 ) )
    {
      FT_UShort insert_glyphID;


      GXV_LIMIT_CHECK( 2 );
      insert_glyphID = FT_NEXT_USHORT( p );
      GXV_TRACE(( " 0x%04x", insert_glyphID ));
    }
    GXV_TRACE(( "\n" ));
#endif
  }


  static void
  gxv_mort_subtable_type5_entry_validate(
    FT_Byte                         state,
    FT_UShort                       flags,
    GXV_StateTable_GlyphOffsetCPtr  glyphOffset,
    FT_Bytes                        table,
    FT_Bytes                        limit,
    GXV_Validator                   gxvalid )
  {
#ifdef GXV_LOAD_UNUSED_VARS
    FT_Bool    setMark;
    FT_Bool    dontAdvance;
    FT_Bool    currentIsKashidaLike;
    FT_Bool    markedIsKashidaLike;
    FT_Bool    currentInsertBefore;
    FT_Bool    markedInsertBefore;
#endif
    FT_Byte    currentInsertCount;
    FT_Byte    markedInsertCount;
    FT_UShort  currentInsertList;
    FT_UShort  markedInsertList;

    FT_UNUSED( state );


#ifdef GXV_LOAD_UNUSED_VARS
    setMark              = FT_BOOL( ( flags >> 15 ) & 1 );
    dontAdvance          = FT_BOOL( ( flags >> 14 ) & 1 );
    currentIsKashidaLike = FT_BOOL( ( flags >> 13 ) & 1 );
    markedIsKashidaLike  = FT_BOOL( ( flags >> 12 ) & 1 );
    currentInsertBefore  = FT_BOOL( ( flags >> 11 ) & 1 );
    markedInsertBefore   = FT_BOOL( ( flags >> 10 ) & 1 );
#endif

    currentInsertCount   = (FT_Byte)( ( flags >> 5 ) & 0x1F   );
    markedInsertCount    = (FT_Byte)(   flags        & 0x001F );

    currentInsertList    = (FT_UShort)( glyphOffset->ul >> 16 );
    markedInsertList     = (FT_UShort)( glyphOffset->ul       );

    if ( 0 != currentInsertList && 0 != currentInsertCount )
    {
      gxv_mort_subtable_type5_InsertList_validate( currentInsertList,
                                                   currentInsertCount,
                                                   table,
                                                   limit,
                                                   gxvalid );
    }

    if ( 0 != markedInsertList && 0 != markedInsertCount )
    {
      gxv_mort_subtable_type5_InsertList_validate( markedInsertList,
                                                   markedInsertCount,
                                                   table,
                                                   limit,
                                                   gxvalid );
    }
  }


  FT_LOCAL_DEF( void )
  gxv_mort_subtable_type5_validate( FT_Bytes       table,
                                    FT_Bytes       limit,
                                    GXV_Validator  gxvalid )
  {
    FT_Bytes  p = table;

    GXV_mort_subtable_type5_StateOptRec      et_rec;
    GXV_mort_subtable_type5_StateOptRecData  et = &et_rec;


    GXV_NAME_ENTER( "mort chain subtable type5 (Glyph Insertion)" );

    GXV_LIMIT_CHECK( GXV_MORT_SUBTABLE_TYPE5_HEADER_SIZE );

    gxvalid->statetable.optdata =
      et;
    gxvalid->statetable.optdata_load_func =
      NULL;
    gxvalid->statetable.subtable_setup_func =
      gxv_mort_subtable_type5_subtable_setup;
    gxvalid->statetable.entry_glyphoffset_fmt =
      GXV_GLYPHOFFSET_ULONG;
    gxvalid->statetable.entry_validate_func =
      gxv_mort_subtable_type5_entry_validate;

    gxv_StateTable_validate( p, limit, gxvalid );

    GXV_EXIT;
  }


/* END */
