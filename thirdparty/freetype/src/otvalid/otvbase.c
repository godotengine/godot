/***************************************************************************/
/*                                                                         */
/*  otvbase.c                                                              */
/*                                                                         */
/*    OpenType BASE table validation (body).                               */
/*                                                                         */
/*  Copyright 2004-2018 by                                                 */
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
#define FT_COMPONENT  trace_otvbase


  static void
  otv_BaseCoord_validate( FT_Bytes       table,
                          OTV_Validator  otvalid )
  {
    FT_Bytes  p = table;
    FT_UInt   BaseCoordFormat;


    OTV_NAME_ENTER( "BaseCoord" );

    OTV_LIMIT_CHECK( 4 );
    BaseCoordFormat = FT_NEXT_USHORT( p );
    p += 2;     /* skip Coordinate */

    OTV_TRACE(( " (format %d)\n", BaseCoordFormat ));

    switch ( BaseCoordFormat )
    {
    case 1:     /* BaseCoordFormat1 */
      break;

    case 2:     /* BaseCoordFormat2 */
      OTV_LIMIT_CHECK( 4 );   /* ReferenceGlyph, BaseCoordPoint */
      break;

    case 3:     /* BaseCoordFormat3 */
      OTV_LIMIT_CHECK( 2 );
      /* DeviceTable */
      otv_Device_validate( table + FT_NEXT_USHORT( p ), otvalid );
      break;

    default:
      FT_INVALID_FORMAT;
    }

    OTV_EXIT;
  }


  static void
  otv_BaseTagList_validate( FT_Bytes       table,
                            OTV_Validator  otvalid )
  {
    FT_Bytes  p = table;
    FT_UInt   BaseTagCount;


    OTV_NAME_ENTER( "BaseTagList" );

    OTV_LIMIT_CHECK( 2 );

    BaseTagCount = FT_NEXT_USHORT( p );

    OTV_TRACE(( " (BaseTagCount = %d)\n", BaseTagCount ));

    OTV_LIMIT_CHECK( BaseTagCount * 4 );          /* BaselineTag */

    OTV_EXIT;
  }


  static void
  otv_BaseValues_validate( FT_Bytes       table,
                           OTV_Validator  otvalid )
  {
    FT_Bytes  p = table;
    FT_UInt   BaseCoordCount;


    OTV_NAME_ENTER( "BaseValues" );

    OTV_LIMIT_CHECK( 4 );

    p             += 2;                     /* skip DefaultIndex */
    BaseCoordCount = FT_NEXT_USHORT( p );

    OTV_TRACE(( " (BaseCoordCount = %d)\n", BaseCoordCount ));

    OTV_LIMIT_CHECK( BaseCoordCount * 2 );

    /* BaseCoord */
    for ( ; BaseCoordCount > 0; BaseCoordCount-- )
      otv_BaseCoord_validate( table + FT_NEXT_USHORT( p ), otvalid );

    OTV_EXIT;
  }


  static void
  otv_MinMax_validate( FT_Bytes       table,
                       OTV_Validator  otvalid )
  {
    FT_Bytes  p = table;
    FT_UInt   table_size;
    FT_UInt   FeatMinMaxCount;

    OTV_OPTIONAL_TABLE( MinCoord );
    OTV_OPTIONAL_TABLE( MaxCoord );


    OTV_NAME_ENTER( "MinMax" );

    OTV_LIMIT_CHECK( 6 );

    OTV_OPTIONAL_OFFSET( MinCoord );
    OTV_OPTIONAL_OFFSET( MaxCoord );
    FeatMinMaxCount = FT_NEXT_USHORT( p );

    OTV_TRACE(( " (FeatMinMaxCount = %d)\n", FeatMinMaxCount ));

    table_size = FeatMinMaxCount * 8 + 6;

    OTV_SIZE_CHECK( MinCoord );
    if ( MinCoord )
      otv_BaseCoord_validate( table + MinCoord, otvalid );

    OTV_SIZE_CHECK( MaxCoord );
    if ( MaxCoord )
      otv_BaseCoord_validate( table + MaxCoord, otvalid );

    OTV_LIMIT_CHECK( FeatMinMaxCount * 8 );

    /* FeatMinMaxRecord */
    for ( ; FeatMinMaxCount > 0; FeatMinMaxCount-- )
    {
      p += 4;                           /* skip FeatureTableTag */

      OTV_OPTIONAL_OFFSET( MinCoord );
      OTV_OPTIONAL_OFFSET( MaxCoord );

      OTV_SIZE_CHECK( MinCoord );
      if ( MinCoord )
        otv_BaseCoord_validate( table + MinCoord, otvalid );

      OTV_SIZE_CHECK( MaxCoord );
      if ( MaxCoord )
        otv_BaseCoord_validate( table + MaxCoord, otvalid );
    }

    OTV_EXIT;
  }


  static void
  otv_BaseScript_validate( FT_Bytes       table,
                           OTV_Validator  otvalid )
  {
    FT_Bytes  p = table;
    FT_UInt   table_size;
    FT_UInt   BaseLangSysCount;

    OTV_OPTIONAL_TABLE( BaseValues    );
    OTV_OPTIONAL_TABLE( DefaultMinMax );


    OTV_NAME_ENTER( "BaseScript" );

    OTV_LIMIT_CHECK( 6 );
    OTV_OPTIONAL_OFFSET( BaseValues    );
    OTV_OPTIONAL_OFFSET( DefaultMinMax );
    BaseLangSysCount = FT_NEXT_USHORT( p );

    OTV_TRACE(( " (BaseLangSysCount = %d)\n", BaseLangSysCount ));

    table_size = BaseLangSysCount * 6 + 6;

    OTV_SIZE_CHECK( BaseValues );
    if ( BaseValues )
      otv_BaseValues_validate( table + BaseValues, otvalid );

    OTV_SIZE_CHECK( DefaultMinMax );
    if ( DefaultMinMax )
      otv_MinMax_validate( table + DefaultMinMax, otvalid );

    OTV_LIMIT_CHECK( BaseLangSysCount * 6 );

    /* BaseLangSysRecord */
    for ( ; BaseLangSysCount > 0; BaseLangSysCount-- )
    {
      p += 4;       /* skip BaseLangSysTag */

      otv_MinMax_validate( table + FT_NEXT_USHORT( p ), otvalid );
    }

    OTV_EXIT;
  }


  static void
  otv_BaseScriptList_validate( FT_Bytes       table,
                               OTV_Validator  otvalid )
  {
    FT_Bytes  p = table;
    FT_UInt   BaseScriptCount;


    OTV_NAME_ENTER( "BaseScriptList" );

    OTV_LIMIT_CHECK( 2 );
    BaseScriptCount = FT_NEXT_USHORT( p );

    OTV_TRACE(( " (BaseScriptCount = %d)\n", BaseScriptCount ));

    OTV_LIMIT_CHECK( BaseScriptCount * 6 );

    /* BaseScriptRecord */
    for ( ; BaseScriptCount > 0; BaseScriptCount-- )
    {
      p += 4;       /* skip BaseScriptTag */

      /* BaseScript */
      otv_BaseScript_validate( table + FT_NEXT_USHORT( p ), otvalid );
    }

    OTV_EXIT;
  }


  static void
  otv_Axis_validate( FT_Bytes       table,
                     OTV_Validator  otvalid )
  {
    FT_Bytes  p = table;
    FT_UInt   table_size;

    OTV_OPTIONAL_TABLE( BaseTagList );


    OTV_NAME_ENTER( "Axis" );

    OTV_LIMIT_CHECK( 4 );
    OTV_OPTIONAL_OFFSET( BaseTagList );

    table_size = 4;

    OTV_SIZE_CHECK( BaseTagList );
    if ( BaseTagList )
      otv_BaseTagList_validate( table + BaseTagList, otvalid );

    /* BaseScriptList */
    otv_BaseScriptList_validate( table + FT_NEXT_USHORT( p ), otvalid );

    OTV_EXIT;
  }


  FT_LOCAL_DEF( void )
  otv_BASE_validate( FT_Bytes      table,
                     FT_Validator  ftvalid )
  {
    OTV_ValidatorRec  otvalidrec;
    OTV_Validator     otvalid = &otvalidrec;
    FT_Bytes          p       = table;
    FT_UInt           table_size;
    FT_UShort         version;

    OTV_OPTIONAL_TABLE( HorizAxis );
    OTV_OPTIONAL_TABLE( VertAxis  );

    OTV_OPTIONAL_TABLE32( itemVarStore );


    otvalid->root = ftvalid;

    FT_TRACE3(( "validating BASE table\n" ));
    OTV_INIT;

    OTV_LIMIT_CHECK( 4 );

    if ( FT_NEXT_USHORT( p ) != 1 )  /* majorVersion */
      FT_INVALID_FORMAT;

    version = FT_NEXT_USHORT( p );   /* minorVersion */

    table_size = 8;
    switch ( version )
    {
    case 0:
      OTV_LIMIT_CHECK( 4 );
      break;

    case 1:
      OTV_LIMIT_CHECK( 8 );
      table_size += 4;
      break;

    default:
      FT_INVALID_FORMAT;
    }

    OTV_OPTIONAL_OFFSET( HorizAxis );
    OTV_SIZE_CHECK( HorizAxis );
    if ( HorizAxis )
      otv_Axis_validate( table + HorizAxis, otvalid );

    OTV_OPTIONAL_OFFSET( VertAxis );
    OTV_SIZE_CHECK( VertAxis );
    if ( VertAxis )
      otv_Axis_validate( table + VertAxis, otvalid );

    if ( version > 0 )
    {
      OTV_OPTIONAL_OFFSET32( itemVarStore );
      OTV_SIZE_CHECK32( itemVarStore );
      if ( itemVarStore )
        OTV_TRACE(( "  [omitting itemVarStore validation]\n" )); /* XXX */
    }

    FT_TRACE4(( "\n" ));
  }


/* END */
