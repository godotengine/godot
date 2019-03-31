/****************************************************************************
 *
 * otvcommn.c
 *
 *   OpenType common tables validation (body).
 *
 * Copyright (C) 2004-2019 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#include "otvcommn.h"


  /**************************************************************************
   *
   * The macro FT_COMPONENT is used in trace mode.  It is an implicit
   * parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log
   * messages during execution.
   */
#undef  FT_COMPONENT
#define FT_COMPONENT  otvcommon


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                       COVERAGE TABLE                          *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  FT_LOCAL_DEF( void )
  otv_Coverage_validate( FT_Bytes       table,
                         OTV_Validator  otvalid,
                         FT_Int         expected_count )
  {
    FT_Bytes  p = table;
    FT_UInt   CoverageFormat;
    FT_UInt   total = 0;


    OTV_NAME_ENTER( "Coverage" );

    OTV_LIMIT_CHECK( 4 );
    CoverageFormat = FT_NEXT_USHORT( p );

    OTV_TRACE(( " (format %d)\n", CoverageFormat ));

    switch ( CoverageFormat )
    {
    case 1:     /* CoverageFormat1 */
      {
        FT_UInt  GlyphCount;
        FT_UInt  i;


        GlyphCount = FT_NEXT_USHORT( p );

        OTV_TRACE(( " (GlyphCount = %d)\n", GlyphCount ));

        OTV_LIMIT_CHECK( GlyphCount * 2 );        /* GlyphArray */

        for ( i = 0; i < GlyphCount; i++ )
        {
          FT_UInt  gid;


          gid = FT_NEXT_USHORT( p );
          if ( gid >= otvalid->glyph_count )
            FT_INVALID_GLYPH_ID;
        }

        total = GlyphCount;
      }
      break;

    case 2:     /* CoverageFormat2 */
      {
        FT_UInt  n, RangeCount;
        FT_UInt  Start, End, StartCoverageIndex, last = 0;


        RangeCount = FT_NEXT_USHORT( p );

        OTV_TRACE(( " (RangeCount = %d)\n", RangeCount ));

        OTV_LIMIT_CHECK( RangeCount * 6 );

        /* RangeRecord */
        for ( n = 0; n < RangeCount; n++ )
        {
          Start              = FT_NEXT_USHORT( p );
          End                = FT_NEXT_USHORT( p );
          StartCoverageIndex = FT_NEXT_USHORT( p );

          if ( Start > End || StartCoverageIndex != total )
            FT_INVALID_DATA;

          if ( End >= otvalid->glyph_count )
            FT_INVALID_GLYPH_ID;

          if ( n > 0 && Start <= last )
            FT_INVALID_DATA;

          total += End - Start + 1;
          last   = End;
        }
      }
      break;

    default:
      FT_INVALID_FORMAT;
    }

    /* Generally, a coverage table offset has an associated count field.  */
    /* The number of glyphs in the table should match this field.  If     */
    /* there is no associated count, a value of -1 tells us not to check. */
    if ( expected_count != -1 && (FT_UInt)expected_count != total )
      FT_INVALID_DATA;

    OTV_EXIT;
  }


  FT_LOCAL_DEF( FT_UInt )
  otv_Coverage_get_first( FT_Bytes  table )
  {
    FT_Bytes  p = table;


    p += 4;     /* skip CoverageFormat and Glyph/RangeCount */

    return FT_NEXT_USHORT( p );
  }


  FT_LOCAL_DEF( FT_UInt )
  otv_Coverage_get_last( FT_Bytes  table )
  {
    FT_Bytes  p = table;
    FT_UInt   CoverageFormat = FT_NEXT_USHORT( p );
    FT_UInt   count          = FT_NEXT_USHORT( p );     /* Glyph/RangeCount */
    FT_UInt   result = 0;


    switch ( CoverageFormat )
    {
    case 1:
      p += ( count - 1 ) * 2;
      result = FT_NEXT_USHORT( p );
      break;

    case 2:
      p += ( count - 1 ) * 6 + 2;
      result = FT_NEXT_USHORT( p );
      break;

    default:
      ;
    }

    return result;
  }


  FT_LOCAL_DEF( FT_UInt )
  otv_Coverage_get_count( FT_Bytes  table )
  {
    FT_Bytes  p              = table;
    FT_UInt   CoverageFormat = FT_NEXT_USHORT( p );
    FT_UInt   count          = FT_NEXT_USHORT( p );     /* Glyph/RangeCount */
    FT_UInt   result         = 0;


    switch ( CoverageFormat )
    {
    case 1:
      return count;

    case 2:
      {
        FT_UInt  Start, End;


        for ( ; count > 0; count-- )
        {
          Start = FT_NEXT_USHORT( p );
          End   = FT_NEXT_USHORT( p );
          p    += 2;                    /* skip StartCoverageIndex */

          result += End - Start + 1;
        }
      }
      break;

    default:
      ;
    }

    return result;
  }


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                   CLASS DEFINITION TABLE                      *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  FT_LOCAL_DEF( void )
  otv_ClassDef_validate( FT_Bytes       table,
                         OTV_Validator  otvalid )
  {
    FT_Bytes  p = table;
    FT_UInt   ClassFormat;


    OTV_NAME_ENTER( "ClassDef" );

    OTV_LIMIT_CHECK( 4 );
    ClassFormat = FT_NEXT_USHORT( p );

    OTV_TRACE(( " (format %d)\n", ClassFormat ));

    switch ( ClassFormat )
    {
    case 1:     /* ClassDefFormat1 */
      {
        FT_UInt  StartGlyph;
        FT_UInt  GlyphCount;


        OTV_LIMIT_CHECK( 4 );

        StartGlyph = FT_NEXT_USHORT( p );
        GlyphCount = FT_NEXT_USHORT( p );

        OTV_TRACE(( " (GlyphCount = %d)\n", GlyphCount ));

        OTV_LIMIT_CHECK( GlyphCount * 2 );    /* ClassValueArray */

        if ( StartGlyph + GlyphCount - 1 >= otvalid->glyph_count )
          FT_INVALID_GLYPH_ID;
      }
      break;

    case 2:     /* ClassDefFormat2 */
      {
        FT_UInt  n, ClassRangeCount;
        FT_UInt  Start, End, last = 0;


        ClassRangeCount = FT_NEXT_USHORT( p );

        OTV_TRACE(( " (ClassRangeCount = %d)\n", ClassRangeCount ));

        OTV_LIMIT_CHECK( ClassRangeCount * 6 );

        /* ClassRangeRecord */
        for ( n = 0; n < ClassRangeCount; n++ )
        {
          Start = FT_NEXT_USHORT( p );
          End   = FT_NEXT_USHORT( p );
          p    += 2;                        /* skip Class */

          if ( Start > End || ( n > 0 && Start <= last ) )
            FT_INVALID_DATA;

          if ( End >= otvalid->glyph_count )
            FT_INVALID_GLYPH_ID;

          last = End;
        }
      }
      break;

    default:
      FT_INVALID_FORMAT;
    }

    /* no need to check glyph indices used as input to class definition   */
    /* tables since even invalid glyph indices return a meaningful result */

    OTV_EXIT;
  }


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                      DEVICE TABLE                             *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  FT_LOCAL_DEF( void )
  otv_Device_validate( FT_Bytes       table,
                       OTV_Validator  otvalid )
  {
    FT_Bytes  p = table;
    FT_UInt   StartSize, EndSize, DeltaFormat, count;


    OTV_NAME_ENTER( "Device" );

    OTV_LIMIT_CHECK( 6 );
    StartSize   = FT_NEXT_USHORT( p );
    EndSize     = FT_NEXT_USHORT( p );
    DeltaFormat = FT_NEXT_USHORT( p );

    if ( DeltaFormat == 0x8000U )
    {
      /* VariationIndex, nothing to do */
    }
    else
    {
      if ( DeltaFormat < 1 || DeltaFormat > 3 )
        FT_INVALID_FORMAT;

      if ( EndSize < StartSize )
        FT_INVALID_DATA;

      count = EndSize - StartSize + 1;
      OTV_LIMIT_CHECK( ( 1 << DeltaFormat ) * count / 8 );  /* DeltaValue */
    }

    OTV_EXIT;
  }


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                         LOOKUPS                               *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  /* uses otvalid->type_count */
  /* uses otvalid->type_funcs */

  FT_LOCAL_DEF( void )
  otv_Lookup_validate( FT_Bytes       table,
                       OTV_Validator  otvalid )
  {
    FT_Bytes           p = table;
    FT_UInt            LookupType, LookupFlag, SubTableCount;
    OTV_Validate_Func  validate;


    OTV_NAME_ENTER( "Lookup" );

    OTV_LIMIT_CHECK( 6 );
    LookupType    = FT_NEXT_USHORT( p );
    LookupFlag    = FT_NEXT_USHORT( p );
    SubTableCount = FT_NEXT_USHORT( p );

    OTV_TRACE(( " (type %d)\n", LookupType ));

    if ( LookupType == 0 || LookupType > otvalid->type_count )
      FT_INVALID_DATA;

    validate = otvalid->type_funcs[LookupType - 1];

    OTV_TRACE(( " (SubTableCount = %d)\n", SubTableCount ));

    OTV_LIMIT_CHECK( SubTableCount * 2 );

    /* SubTable */
    for ( ; SubTableCount > 0; SubTableCount-- )
      validate( table + FT_NEXT_USHORT( p ), otvalid );

    if ( LookupFlag & 0x10 )
      OTV_LIMIT_CHECK( 2 );  /* MarkFilteringSet */

    OTV_EXIT;
  }


  /* uses valid->lookup_count */

  FT_LOCAL_DEF( void )
  otv_LookupList_validate( FT_Bytes       table,
                           OTV_Validator  otvalid )
  {
    FT_Bytes  p = table;
    FT_UInt   LookupCount;


    OTV_NAME_ENTER( "LookupList" );

    OTV_LIMIT_CHECK( 2 );
    LookupCount = FT_NEXT_USHORT( p );

    OTV_TRACE(( " (LookupCount = %d)\n", LookupCount ));

    OTV_LIMIT_CHECK( LookupCount * 2 );

    otvalid->lookup_count = LookupCount;

    /* Lookup */
    for ( ; LookupCount > 0; LookupCount-- )
      otv_Lookup_validate( table + FT_NEXT_USHORT( p ), otvalid );

    OTV_EXIT;
  }


  static FT_UInt
  otv_LookupList_get_count( FT_Bytes  table )
  {
    return FT_NEXT_USHORT( table );
  }


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                        FEATURES                               *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  /* uses otvalid->lookup_count */

  FT_LOCAL_DEF( void )
  otv_Feature_validate( FT_Bytes       table,
                        OTV_Validator  otvalid )
  {
    FT_Bytes  p = table;
    FT_UInt   LookupCount;


    OTV_NAME_ENTER( "Feature" );

    OTV_LIMIT_CHECK( 4 );
    p           += 2;                   /* skip FeatureParams (unused) */
    LookupCount  = FT_NEXT_USHORT( p );

    OTV_TRACE(( " (LookupCount = %d)\n", LookupCount ));

    OTV_LIMIT_CHECK( LookupCount * 2 );

    /* LookupListIndex */
    for ( ; LookupCount > 0; LookupCount-- )
      if ( FT_NEXT_USHORT( p ) >= otvalid->lookup_count )
        FT_INVALID_DATA;

    OTV_EXIT;
  }


  static FT_UInt
  otv_Feature_get_count( FT_Bytes  table )
  {
    return FT_NEXT_USHORT( table );
  }


  /* sets otvalid->lookup_count */

  FT_LOCAL_DEF( void )
  otv_FeatureList_validate( FT_Bytes       table,
                            FT_Bytes       lookups,
                            OTV_Validator  otvalid )
  {
    FT_Bytes  p = table;
    FT_UInt   FeatureCount;


    OTV_NAME_ENTER( "FeatureList" );

    OTV_LIMIT_CHECK( 2 );
    FeatureCount = FT_NEXT_USHORT( p );

    OTV_TRACE(( " (FeatureCount = %d)\n", FeatureCount ));

    OTV_LIMIT_CHECK( FeatureCount * 2 );

    otvalid->lookup_count = otv_LookupList_get_count( lookups );

    /* FeatureRecord */
    for ( ; FeatureCount > 0; FeatureCount-- )
    {
      p += 4;       /* skip FeatureTag */

      /* Feature */
      otv_Feature_validate( table + FT_NEXT_USHORT( p ), otvalid );
    }

    OTV_EXIT;
  }


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                       LANGUAGE SYSTEM                         *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/


  /* uses otvalid->extra1 (number of features) */

  FT_LOCAL_DEF( void )
  otv_LangSys_validate( FT_Bytes       table,
                        OTV_Validator  otvalid )
  {
    FT_Bytes  p = table;
    FT_UInt   ReqFeatureIndex;
    FT_UInt   FeatureCount;


    OTV_NAME_ENTER( "LangSys" );

    OTV_LIMIT_CHECK( 6 );
    p              += 2;                    /* skip LookupOrder (unused) */
    ReqFeatureIndex = FT_NEXT_USHORT( p );
    FeatureCount    = FT_NEXT_USHORT( p );

    OTV_TRACE(( " (ReqFeatureIndex = %d)\n", ReqFeatureIndex ));
    OTV_TRACE(( " (FeatureCount = %d)\n",    FeatureCount    ));

    if ( ReqFeatureIndex != 0xFFFFU && ReqFeatureIndex >= otvalid->extra1 )
      FT_INVALID_DATA;

    OTV_LIMIT_CHECK( FeatureCount * 2 );

    /* FeatureIndex */
    for ( ; FeatureCount > 0; FeatureCount-- )
      if ( FT_NEXT_USHORT( p ) >= otvalid->extra1 )
        FT_INVALID_DATA;

    OTV_EXIT;
  }


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                           SCRIPTS                             *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  FT_LOCAL_DEF( void )
  otv_Script_validate( FT_Bytes       table,
                       OTV_Validator  otvalid )
  {
    FT_UInt   DefaultLangSys, LangSysCount;
    FT_Bytes  p = table;


    OTV_NAME_ENTER( "Script" );

    OTV_LIMIT_CHECK( 4 );
    DefaultLangSys = FT_NEXT_USHORT( p );
    LangSysCount   = FT_NEXT_USHORT( p );

    OTV_TRACE(( " (LangSysCount = %d)\n", LangSysCount ));

    if ( DefaultLangSys != 0 )
      otv_LangSys_validate( table + DefaultLangSys, otvalid );

    OTV_LIMIT_CHECK( LangSysCount * 6 );

    /* LangSysRecord */
    for ( ; LangSysCount > 0; LangSysCount-- )
    {
      p += 4;       /* skip LangSysTag */

      /* LangSys */
      otv_LangSys_validate( table + FT_NEXT_USHORT( p ), otvalid );
    }

    OTV_EXIT;
  }


  /* sets otvalid->extra1 (number of features) */

  FT_LOCAL_DEF( void )
  otv_ScriptList_validate( FT_Bytes       table,
                           FT_Bytes       features,
                           OTV_Validator  otvalid )
  {
    FT_UInt   ScriptCount;
    FT_Bytes  p = table;


    OTV_NAME_ENTER( "ScriptList" );

    OTV_LIMIT_CHECK( 2 );
    ScriptCount = FT_NEXT_USHORT( p );

    OTV_TRACE(( " (ScriptCount = %d)\n", ScriptCount ));

    OTV_LIMIT_CHECK( ScriptCount * 6 );

    otvalid->extra1 = otv_Feature_get_count( features );

    /* ScriptRecord */
    for ( ; ScriptCount > 0; ScriptCount-- )
    {
      p += 4;       /* skip ScriptTag */

      otv_Script_validate( table + FT_NEXT_USHORT( p ), otvalid ); /* Script */
    }

    OTV_EXIT;
  }


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                      UTILITY FUNCTIONS                        *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  /*
     u:   uint16
     ux:  unit16 [x]

     s:   struct
     sx:  struct [x]
     sxy: struct [x], using external y count

     x:   uint16 x

     C:   Coverage

     O:   Offset
     On:  Offset (NULL)
     Ox:  Offset [x]
     Onx: Offset (NULL) [x]
  */

  FT_LOCAL_DEF( void )
  otv_x_Ox( FT_Bytes       table,
            OTV_Validator  otvalid )
  {
    FT_Bytes           p = table;
    FT_UInt            Count;
    OTV_Validate_Func  func;


    OTV_ENTER;

    OTV_LIMIT_CHECK( 2 );
    Count = FT_NEXT_USHORT( p );

    OTV_TRACE(( " (Count = %d)\n", Count ));

    OTV_LIMIT_CHECK( Count * 2 );

    otvalid->nesting_level++;
    func = otvalid->func[otvalid->nesting_level];

    for ( ; Count > 0; Count-- )
      func( table + FT_NEXT_USHORT( p ), otvalid );

    otvalid->nesting_level--;

    OTV_EXIT;
  }


  FT_LOCAL_DEF( void )
  otv_u_C_x_Ox( FT_Bytes       table,
                OTV_Validator  otvalid )
  {
    FT_Bytes           p = table;
    FT_UInt            Count, Coverage;
    OTV_Validate_Func  func;


    OTV_ENTER;

    p += 2;     /* skip Format */

    OTV_LIMIT_CHECK( 4 );
    Coverage = FT_NEXT_USHORT( p );
    Count    = FT_NEXT_USHORT( p );

    OTV_TRACE(( " (Count = %d)\n", Count ));

    otv_Coverage_validate( table + Coverage, otvalid, (FT_Int)Count );

    OTV_LIMIT_CHECK( Count * 2 );

    otvalid->nesting_level++;
    func = otvalid->func[otvalid->nesting_level];

    for ( ; Count > 0; Count-- )
      func( table + FT_NEXT_USHORT( p ), otvalid );

    otvalid->nesting_level--;

    OTV_EXIT;
  }


  /* uses otvalid->extra1 (if > 0: array value limit) */

  FT_LOCAL_DEF( void )
  otv_x_ux( FT_Bytes       table,
            OTV_Validator  otvalid )
  {
    FT_Bytes  p = table;
    FT_UInt   Count;


    OTV_ENTER;

    OTV_LIMIT_CHECK( 2 );
    Count = FT_NEXT_USHORT( p );

    OTV_TRACE(( " (Count = %d)\n", Count ));

    OTV_LIMIT_CHECK( Count * 2 );

    if ( otvalid->extra1 )
    {
      for ( ; Count > 0; Count-- )
        if ( FT_NEXT_USHORT( p ) >= otvalid->extra1 )
          FT_INVALID_DATA;
    }

    OTV_EXIT;
  }


  /* `ux' in the function's name is not really correct since only x-1 */
  /* elements are tested                                              */

  /* uses otvalid->extra1 (array value limit) */

  FT_LOCAL_DEF( void )
  otv_x_y_ux_sy( FT_Bytes       table,
                 OTV_Validator  otvalid )
  {
    FT_Bytes  p = table;
    FT_UInt   Count1, Count2;


    OTV_ENTER;

    OTV_LIMIT_CHECK( 4 );
    Count1 = FT_NEXT_USHORT( p );
    Count2 = FT_NEXT_USHORT( p );

    OTV_TRACE(( " (Count1 = %d)\n", Count1 ));
    OTV_TRACE(( " (Count2 = %d)\n", Count2 ));

    if ( Count1 == 0 )
      FT_INVALID_DATA;

    OTV_LIMIT_CHECK( ( Count1 - 1 ) * 2 + Count2 * 4 );
    p += ( Count1 - 1 ) * 2;

    for ( ; Count2 > 0; Count2-- )
    {
      if ( FT_NEXT_USHORT( p ) >= Count1 )
        FT_INVALID_DATA;

      if ( FT_NEXT_USHORT( p ) >= otvalid->extra1 )
        FT_INVALID_DATA;
    }

    OTV_EXIT;
  }


  /* `uy' in the function's name is not really correct since only y-1 */
  /* elements are tested                                              */

  /* uses otvalid->extra1 (array value limit) */

  FT_LOCAL_DEF( void )
  otv_x_ux_y_uy_z_uz_p_sp( FT_Bytes       table,
                           OTV_Validator  otvalid )
  {
    FT_Bytes  p = table;
    FT_UInt   BacktrackCount, InputCount, LookaheadCount;
    FT_UInt   Count;


    OTV_ENTER;

    OTV_LIMIT_CHECK( 2 );
    BacktrackCount = FT_NEXT_USHORT( p );

    OTV_TRACE(( " (BacktrackCount = %d)\n", BacktrackCount ));

    OTV_LIMIT_CHECK( BacktrackCount * 2 + 2 );
    p += BacktrackCount * 2;

    InputCount = FT_NEXT_USHORT( p );
    if ( InputCount == 0 )
      FT_INVALID_DATA;

    OTV_TRACE(( " (InputCount = %d)\n", InputCount ));

    OTV_LIMIT_CHECK( InputCount * 2 );
    p += ( InputCount - 1 ) * 2;

    LookaheadCount = FT_NEXT_USHORT( p );

    OTV_TRACE(( " (LookaheadCount = %d)\n", LookaheadCount ));

    OTV_LIMIT_CHECK( LookaheadCount * 2 + 2 );
    p += LookaheadCount * 2;

    Count = FT_NEXT_USHORT( p );

    OTV_TRACE(( " (Count = %d)\n", Count ));

    OTV_LIMIT_CHECK( Count * 4 );

    for ( ; Count > 0; Count-- )
    {
      if ( FT_NEXT_USHORT( p ) >= InputCount )
        FT_INVALID_DATA;

      if ( FT_NEXT_USHORT( p ) >= otvalid->extra1 )
        FT_INVALID_DATA;
    }

    OTV_EXIT;
  }


  /* sets otvalid->extra1 (valid->lookup_count) */

  FT_LOCAL_DEF( void )
  otv_u_O_O_x_Onx( FT_Bytes       table,
                   OTV_Validator  otvalid )
  {
    FT_Bytes           p = table;
    FT_UInt            Coverage, ClassDef, ClassSetCount;
    OTV_Validate_Func  func;


    OTV_ENTER;

    p += 2;     /* skip Format */

    OTV_LIMIT_CHECK( 6 );
    Coverage      = FT_NEXT_USHORT( p );
    ClassDef      = FT_NEXT_USHORT( p );
    ClassSetCount = FT_NEXT_USHORT( p );

    OTV_TRACE(( " (ClassSetCount = %d)\n", ClassSetCount ));

    otv_Coverage_validate( table + Coverage, otvalid, -1 );
    otv_ClassDef_validate( table + ClassDef, otvalid );

    OTV_LIMIT_CHECK( ClassSetCount * 2 );

    otvalid->nesting_level++;
    func          = otvalid->func[otvalid->nesting_level];
    otvalid->extra1 = otvalid->lookup_count;

    for ( ; ClassSetCount > 0; ClassSetCount-- )
    {
      FT_UInt  offset = FT_NEXT_USHORT( p );


      if ( offset )
        func( table + offset, otvalid );
    }

    otvalid->nesting_level--;

    OTV_EXIT;
  }


  /* uses otvalid->lookup_count */

  FT_LOCAL_DEF( void )
  otv_u_x_y_Ox_sy( FT_Bytes       table,
                   OTV_Validator  otvalid )
  {
    FT_Bytes  p = table;
    FT_UInt   GlyphCount, Count, count1;


    OTV_ENTER;

    p += 2;     /* skip Format */

    OTV_LIMIT_CHECK( 4 );
    GlyphCount = FT_NEXT_USHORT( p );
    Count      = FT_NEXT_USHORT( p );

    OTV_TRACE(( " (GlyphCount = %d)\n", GlyphCount ));
    OTV_TRACE(( " (Count = %d)\n",      Count      ));

    OTV_LIMIT_CHECK( GlyphCount * 2 + Count * 4 );

    for ( count1 = GlyphCount; count1 > 0; count1-- )
      otv_Coverage_validate( table + FT_NEXT_USHORT( p ), otvalid, -1 );

    for ( ; Count > 0; Count-- )
    {
      if ( FT_NEXT_USHORT( p ) >= GlyphCount )
        FT_INVALID_DATA;

      if ( FT_NEXT_USHORT( p ) >= otvalid->lookup_count )
        FT_INVALID_DATA;
    }

    OTV_EXIT;
  }


  /* sets otvalid->extra1 (valid->lookup_count)    */

  FT_LOCAL_DEF( void )
  otv_u_O_O_O_O_x_Onx( FT_Bytes       table,
                       OTV_Validator  otvalid )
  {
    FT_Bytes           p = table;
    FT_UInt            Coverage;
    FT_UInt            BacktrackClassDef, InputClassDef, LookaheadClassDef;
    FT_UInt            ChainClassSetCount;
    OTV_Validate_Func  func;


    OTV_ENTER;

    p += 2;     /* skip Format */

    OTV_LIMIT_CHECK( 10 );
    Coverage           = FT_NEXT_USHORT( p );
    BacktrackClassDef  = FT_NEXT_USHORT( p );
    InputClassDef      = FT_NEXT_USHORT( p );
    LookaheadClassDef  = FT_NEXT_USHORT( p );
    ChainClassSetCount = FT_NEXT_USHORT( p );

    OTV_TRACE(( " (ChainClassSetCount = %d)\n", ChainClassSetCount ));

    otv_Coverage_validate( table + Coverage, otvalid, -1 );

    otv_ClassDef_validate( table + BacktrackClassDef,  otvalid );
    otv_ClassDef_validate( table + InputClassDef, otvalid );
    otv_ClassDef_validate( table + LookaheadClassDef, otvalid );

    OTV_LIMIT_CHECK( ChainClassSetCount * 2 );

    otvalid->nesting_level++;
    func          = otvalid->func[otvalid->nesting_level];
    otvalid->extra1 = otvalid->lookup_count;

    for ( ; ChainClassSetCount > 0; ChainClassSetCount-- )
    {
      FT_UInt  offset = FT_NEXT_USHORT( p );


      if ( offset )
        func( table + offset, otvalid );
    }

    otvalid->nesting_level--;

    OTV_EXIT;
  }


  /* uses otvalid->lookup_count */

  FT_LOCAL_DEF( void )
  otv_u_x_Ox_y_Oy_z_Oz_p_sp( FT_Bytes       table,
                             OTV_Validator  otvalid )
  {
    FT_Bytes  p = table;
    FT_UInt   BacktrackGlyphCount, InputGlyphCount, LookaheadGlyphCount;
    FT_UInt   count1, count2;


    OTV_ENTER;

    p += 2;     /* skip Format */

    OTV_LIMIT_CHECK( 2 );
    BacktrackGlyphCount = FT_NEXT_USHORT( p );

    OTV_TRACE(( " (BacktrackGlyphCount = %d)\n", BacktrackGlyphCount ));

    OTV_LIMIT_CHECK( BacktrackGlyphCount * 2 + 2 );

    for ( ; BacktrackGlyphCount > 0; BacktrackGlyphCount-- )
      otv_Coverage_validate( table + FT_NEXT_USHORT( p ), otvalid, -1 );

    InputGlyphCount = FT_NEXT_USHORT( p );

    OTV_TRACE(( " (InputGlyphCount = %d)\n", InputGlyphCount ));

    OTV_LIMIT_CHECK( InputGlyphCount * 2 + 2 );

    for ( count1 = InputGlyphCount; count1 > 0; count1-- )
      otv_Coverage_validate( table + FT_NEXT_USHORT( p ), otvalid, -1 );

    LookaheadGlyphCount = FT_NEXT_USHORT( p );

    OTV_TRACE(( " (LookaheadGlyphCount = %d)\n", LookaheadGlyphCount ));

    OTV_LIMIT_CHECK( LookaheadGlyphCount * 2 + 2 );

    for ( ; LookaheadGlyphCount > 0; LookaheadGlyphCount-- )
      otv_Coverage_validate( table + FT_NEXT_USHORT( p ), otvalid, -1 );

    count2 = FT_NEXT_USHORT( p );

    OTV_TRACE(( " (Count = %d)\n", count2 ));

    OTV_LIMIT_CHECK( count2 * 4 );

    for ( ; count2 > 0; count2-- )
    {
      if ( FT_NEXT_USHORT( p ) >= InputGlyphCount )
        FT_INVALID_DATA;

      if ( FT_NEXT_USHORT( p ) >= otvalid->lookup_count )
        FT_INVALID_DATA;
    }

    OTV_EXIT;
  }


  FT_LOCAL_DEF( FT_UInt )
  otv_GSUBGPOS_get_Lookup_count( FT_Bytes  table )
  {
    FT_Bytes  p = table + 8;


    return otv_LookupList_get_count( table + FT_NEXT_USHORT( p ) );
  }


  FT_LOCAL_DEF( FT_UInt )
  otv_GSUBGPOS_have_MarkAttachmentType_flag( FT_Bytes  table )
  {
    FT_Bytes  p, lookup;
    FT_UInt   count;


    if ( !table )
      return 0;

    /* LookupList */
    p      = table + 8;
    table += FT_NEXT_USHORT( p );

    /* LookupCount */
    p     = table;
    count = FT_NEXT_USHORT( p );

    for ( ; count > 0; count-- )
    {
      FT_Bytes  oldp;


      /* Lookup */
      lookup = table + FT_NEXT_USHORT( p );

      oldp = p;

      /* LookupFlag */
      p = lookup + 2;
      if ( FT_NEXT_USHORT( p ) & 0xFF00U )
        return 1;

      p = oldp;
    }

    return 0;
  }


/* END */
