/****************************************************************************
 *
 * otvgpos.c
 *
 *   OpenType GPOS table validation (body).
 *
 * Copyright (C) 2002-2024 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#include "otvalid.h"
#include "otvcommn.h"
#include "otvgpos.h"


  /**************************************************************************
   *
   * The macro FT_COMPONENT is used in trace mode.  It is an implicit
   * parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log
   * messages during execution.
   */
#undef  FT_COMPONENT
#define FT_COMPONENT  otvgpos


  static void
  otv_Anchor_validate( FT_Bytes       table,
                       OTV_Validator  valid );

  static void
  otv_MarkArray_validate( FT_Bytes       table,
                          OTV_Validator  valid );


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                      UTILITY FUNCTIONS                        *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

#define BaseArrayFunc       otv_x_sxy
#define LigatureAttachFunc  otv_x_sxy
#define Mark2ArrayFunc      otv_x_sxy

  /* uses valid->extra1 (counter)                             */
  /* uses valid->extra2 (boolean to handle NULL anchor field) */

  static void
  otv_x_sxy( FT_Bytes       table,
             OTV_Validator  otvalid )
  {
    FT_Bytes  p = table;
    FT_UInt   Count, count1, table_size;


    OTV_ENTER;

    OTV_LIMIT_CHECK( 2 );

    Count = FT_NEXT_USHORT( p );

    OTV_TRACE(( " (Count = %d)\n", Count ));

    OTV_LIMIT_CHECK( Count * otvalid->extra1 * 2 );

    table_size = Count * otvalid->extra1 * 2 + 2;

    for ( ; Count > 0; Count-- )
      for ( count1 = otvalid->extra1; count1 > 0; count1-- )
      {
        OTV_OPTIONAL_TABLE( anchor_offset );


        OTV_OPTIONAL_OFFSET( anchor_offset );

        if ( otvalid->extra2 )
        {
          OTV_SIZE_CHECK( anchor_offset );
          if ( anchor_offset )
            otv_Anchor_validate( table + anchor_offset, otvalid );
        }
        else
          otv_Anchor_validate( table + anchor_offset, otvalid );
      }

    OTV_EXIT;
  }


#define MarkBasePosFormat1Func  otv_u_O_O_u_O_O
#define MarkLigPosFormat1Func   otv_u_O_O_u_O_O
#define MarkMarkPosFormat1Func  otv_u_O_O_u_O_O

  /* sets otvalid->extra1 (class count) */

  static void
  otv_u_O_O_u_O_O( FT_Bytes       table,
                   OTV_Validator  otvalid )
  {
    FT_Bytes           p = table;
    FT_UInt            Coverage1, Coverage2, ClassCount;
    FT_UInt            Array1, Array2;
    OTV_Validate_Func  func;


    OTV_ENTER;

    p += 2;     /* skip PosFormat */

    OTV_LIMIT_CHECK( 10 );
    Coverage1  = FT_NEXT_USHORT( p );
    Coverage2  = FT_NEXT_USHORT( p );
    ClassCount = FT_NEXT_USHORT( p );
    Array1     = FT_NEXT_USHORT( p );
    Array2     = FT_NEXT_USHORT( p );

    otv_Coverage_validate( table + Coverage1, otvalid, -1 );
    otv_Coverage_validate( table + Coverage2, otvalid, -1 );

    otv_MarkArray_validate( table + Array1, otvalid );

    otvalid->nesting_level++;
    func            = otvalid->func[otvalid->nesting_level];
    otvalid->extra1 = ClassCount;

    func( table + Array2, otvalid );

    otvalid->nesting_level--;

    OTV_EXIT;
  }


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                        VALUE RECORDS                          *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  static FT_UInt
  otv_value_length( FT_UInt  format )
  {
    FT_UInt  count;


    count = ( ( format & 0xAA ) >> 1 ) + ( format & 0x55 );
    count = ( ( count  & 0xCC ) >> 2 ) + ( count  & 0x33 );
    count = ( ( count  & 0xF0 ) >> 4 ) + ( count  & 0x0F );

    return count * 2;
  }


  /* uses otvalid->extra3 (pointer to base table) */

  static void
  otv_ValueRecord_validate( FT_Bytes       table,
                            FT_UInt        format,
                            OTV_Validator  otvalid )
  {
    FT_Bytes  p = table;
    FT_UInt   count;

#ifdef FT_DEBUG_LEVEL_TRACE
    FT_Int    loop;
    FT_ULong  res = 0;


    OTV_NAME_ENTER( "ValueRecord" );

    /* display `format' in dual representation */
    for ( loop = 7; loop >= 0; loop-- )
    {
      res <<= 4;
      res  += ( format >> loop ) & 1;
    }

    OTV_TRACE(( " (format 0b%08lx)\n", res ));
#endif

    if ( format >= 0x100 )
      FT_INVALID_FORMAT;

    for ( count = 4; count > 0; count-- )
    {
      if ( format & 1 )
      {
        /* XPlacement, YPlacement, XAdvance, YAdvance */
        OTV_LIMIT_CHECK( 2 );
        p += 2;
      }

      format >>= 1;
    }

    for ( count = 4; count > 0; count-- )
    {
      if ( format & 1 )
      {
        FT_PtrDist  table_size;

        OTV_OPTIONAL_TABLE( device );


        /* XPlaDevice, YPlaDevice, XAdvDevice, YAdvDevice */
        OTV_LIMIT_CHECK( 2 );
        OTV_OPTIONAL_OFFSET( device );

        table_size = p - otvalid->extra3;

        OTV_SIZE_CHECK( device );
        if ( device )
          otv_Device_validate( otvalid->extra3 + device, otvalid );
      }
      format >>= 1;
    }

    OTV_EXIT;
  }


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                           ANCHORS                             *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  static void
  otv_Anchor_validate( FT_Bytes       table,
                       OTV_Validator  otvalid )
  {
    FT_Bytes  p = table;
    FT_UInt   AnchorFormat;


    OTV_NAME_ENTER( "Anchor");

    OTV_LIMIT_CHECK( 6 );
    AnchorFormat = FT_NEXT_USHORT( p );

    OTV_TRACE(( " (format %d)\n", AnchorFormat ));

    p += 4;     /* skip XCoordinate and YCoordinate */

    switch ( AnchorFormat )
    {
    case 1:
      break;

    case 2:
      OTV_LIMIT_CHECK( 2 );  /* AnchorPoint */
      break;

    case 3:
      {
        FT_UInt  table_size;

        OTV_OPTIONAL_TABLE( XDeviceTable );
        OTV_OPTIONAL_TABLE( YDeviceTable );


        OTV_LIMIT_CHECK( 4 );
        OTV_OPTIONAL_OFFSET( XDeviceTable );
        OTV_OPTIONAL_OFFSET( YDeviceTable );

        table_size = 6 + 4;

        OTV_SIZE_CHECK( XDeviceTable );
        if ( XDeviceTable )
          otv_Device_validate( table + XDeviceTable, otvalid );

        OTV_SIZE_CHECK( YDeviceTable );
        if ( YDeviceTable )
          otv_Device_validate( table + YDeviceTable, otvalid );
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
  /*****                         MARK ARRAYS                           *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  static void
  otv_MarkArray_validate( FT_Bytes       table,
                          OTV_Validator  otvalid )
  {
    FT_Bytes  p = table;
    FT_UInt   MarkCount;


    OTV_NAME_ENTER( "MarkArray" );

    OTV_LIMIT_CHECK( 2 );
    MarkCount = FT_NEXT_USHORT( p );

    OTV_TRACE(( " (MarkCount = %d)\n", MarkCount ));

    OTV_LIMIT_CHECK( MarkCount * 4 );

    /* MarkRecord */
    for ( ; MarkCount > 0; MarkCount-- )
    {
      p += 2;   /* skip Class */
      /* MarkAnchor */
      otv_Anchor_validate( table + FT_NEXT_USHORT( p ), otvalid );
    }

    OTV_EXIT;
  }


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                     GPOS LOOKUP TYPE 1                        *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  /* sets otvalid->extra3 (pointer to base table) */

  static void
  otv_SinglePos_validate( FT_Bytes       table,
                          OTV_Validator  otvalid )
  {
    FT_Bytes  p = table;
    FT_UInt   PosFormat;


    OTV_NAME_ENTER( "SinglePos" );

    OTV_LIMIT_CHECK( 2 );
    PosFormat = FT_NEXT_USHORT( p );

    OTV_TRACE(( " (format %d)\n", PosFormat ));

    otvalid->extra3 = table;

    switch ( PosFormat )
    {
    case 1:     /* SinglePosFormat1 */
      {
        FT_UInt  Coverage, ValueFormat;


        OTV_LIMIT_CHECK( 4 );
        Coverage    = FT_NEXT_USHORT( p );
        ValueFormat = FT_NEXT_USHORT( p );

        otv_Coverage_validate( table + Coverage, otvalid, -1 );
        otv_ValueRecord_validate( p, ValueFormat, otvalid ); /* Value */
      }
      break;

    case 2:     /* SinglePosFormat2 */
      {
        FT_UInt  Coverage, ValueFormat, ValueCount, len_value;


        OTV_LIMIT_CHECK( 6 );
        Coverage    = FT_NEXT_USHORT( p );
        ValueFormat = FT_NEXT_USHORT( p );
        ValueCount  = FT_NEXT_USHORT( p );

        OTV_TRACE(( " (ValueCount = %d)\n", ValueCount ));

        len_value = otv_value_length( ValueFormat );

        otv_Coverage_validate( table + Coverage,
                               otvalid,
                               (FT_Int)ValueCount );

        OTV_LIMIT_CHECK( ValueCount * len_value );

        /* Value */
        for ( ; ValueCount > 0; ValueCount-- )
        {
          otv_ValueRecord_validate( p, ValueFormat, otvalid );
          p += len_value;
        }
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
  /*****                     GPOS LOOKUP TYPE 2                        *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  /* sets otvalid->extra3 (pointer to base table) */

  static void
  otv_PairSet_validate( FT_Bytes       table,
                        FT_UInt        format1,
                        FT_UInt        format2,
                        OTV_Validator  otvalid )
  {
    FT_Bytes  p = table;
    FT_UInt   value_len1, value_len2, PairValueCount;


    OTV_NAME_ENTER( "PairSet" );

    otvalid->extra3 = table;

    OTV_LIMIT_CHECK( 2 );
    PairValueCount = FT_NEXT_USHORT( p );

    OTV_TRACE(( " (PairValueCount = %d)\n", PairValueCount ));

    value_len1 = otv_value_length( format1 );
    value_len2 = otv_value_length( format2 );

    OTV_LIMIT_CHECK( PairValueCount * ( value_len1 + value_len2 + 2 ) );

    /* PairValueRecord */
    for ( ; PairValueCount > 0; PairValueCount-- )
    {
      p += 2;       /* skip SecondGlyph */

      if ( format1 )
        otv_ValueRecord_validate( p, format1, otvalid ); /* Value1 */
      p += value_len1;

      if ( format2 )
        otv_ValueRecord_validate( p, format2, otvalid ); /* Value2 */
      p += value_len2;
    }

    OTV_EXIT;
  }


  /* sets otvalid->extra3 (pointer to base table) */

  static void
  otv_PairPos_validate( FT_Bytes       table,
                        OTV_Validator  otvalid )
  {
    FT_Bytes  p = table;
    FT_UInt   PosFormat;


    OTV_NAME_ENTER( "PairPos" );

    OTV_LIMIT_CHECK( 2 );
    PosFormat = FT_NEXT_USHORT( p );

    OTV_TRACE(( " (format %d)\n", PosFormat ));

    switch ( PosFormat )
    {
    case 1:     /* PairPosFormat1 */
      {
        FT_UInt  Coverage, ValueFormat1, ValueFormat2, PairSetCount;


        OTV_LIMIT_CHECK( 8 );
        Coverage     = FT_NEXT_USHORT( p );
        ValueFormat1 = FT_NEXT_USHORT( p );
        ValueFormat2 = FT_NEXT_USHORT( p );
        PairSetCount = FT_NEXT_USHORT( p );

        OTV_TRACE(( " (PairSetCount = %d)\n", PairSetCount ));

        otv_Coverage_validate( table + Coverage, otvalid, -1 );

        OTV_LIMIT_CHECK( PairSetCount * 2 );

        /* PairSetOffset */
        for ( ; PairSetCount > 0; PairSetCount-- )
          otv_PairSet_validate( table + FT_NEXT_USHORT( p ),
                                ValueFormat1, ValueFormat2, otvalid );
      }
      break;

    case 2:     /* PairPosFormat2 */
      {
        FT_UInt  Coverage, ValueFormat1, ValueFormat2, ClassDef1, ClassDef2;
        FT_UInt  ClassCount1, ClassCount2, len_value1, len_value2, count;


        OTV_LIMIT_CHECK( 14 );
        Coverage     = FT_NEXT_USHORT( p );
        ValueFormat1 = FT_NEXT_USHORT( p );
        ValueFormat2 = FT_NEXT_USHORT( p );
        ClassDef1    = FT_NEXT_USHORT( p );
        ClassDef2    = FT_NEXT_USHORT( p );
        ClassCount1  = FT_NEXT_USHORT( p );
        ClassCount2  = FT_NEXT_USHORT( p );

        OTV_TRACE(( " (ClassCount1 = %d)\n", ClassCount1 ));
        OTV_TRACE(( " (ClassCount2 = %d)\n", ClassCount2 ));

        len_value1 = otv_value_length( ValueFormat1 );
        len_value2 = otv_value_length( ValueFormat2 );

        otv_Coverage_validate( table + Coverage, otvalid, -1 );
        otv_ClassDef_validate( table + ClassDef1, otvalid );
        otv_ClassDef_validate( table + ClassDef2, otvalid );

        OTV_LIMIT_CHECK( ClassCount1 * ClassCount2 *
                         ( len_value1 + len_value2 ) );

        otvalid->extra3 = table;

        /* Class1Record */
        for ( ; ClassCount1 > 0; ClassCount1-- )
        {
          /* Class2Record */
          for ( count = ClassCount2; count > 0; count-- )
          {
            if ( ValueFormat1 )
              /* Value1 */
              otv_ValueRecord_validate( p, ValueFormat1, otvalid );
            p += len_value1;

            if ( ValueFormat2 )
              /* Value2 */
              otv_ValueRecord_validate( p, ValueFormat2, otvalid );
            p += len_value2;
          }
        }
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
  /*****                     GPOS LOOKUP TYPE 3                        *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  static void
  otv_CursivePos_validate( FT_Bytes       table,
                           OTV_Validator  otvalid )
  {
    FT_Bytes  p = table;
    FT_UInt   PosFormat;


    OTV_NAME_ENTER( "CursivePos" );

    OTV_LIMIT_CHECK( 2 );
    PosFormat = FT_NEXT_USHORT( p );

    OTV_TRACE(( " (format %d)\n", PosFormat ));

    switch ( PosFormat )
    {
    case 1:     /* CursivePosFormat1 */
      {
        FT_UInt   table_size;
        FT_UInt   Coverage, EntryExitCount;

        OTV_OPTIONAL_TABLE( EntryAnchor );
        OTV_OPTIONAL_TABLE( ExitAnchor  );


        OTV_LIMIT_CHECK( 4 );
        Coverage       = FT_NEXT_USHORT( p );
        EntryExitCount = FT_NEXT_USHORT( p );

        OTV_TRACE(( " (EntryExitCount = %d)\n", EntryExitCount ));

        otv_Coverage_validate( table + Coverage,
                               otvalid,
                               (FT_Int)EntryExitCount );

        OTV_LIMIT_CHECK( EntryExitCount * 4 );

        table_size = EntryExitCount * 4 + 4;

        /* EntryExitRecord */
        for ( ; EntryExitCount > 0; EntryExitCount-- )
        {
          OTV_OPTIONAL_OFFSET( EntryAnchor );
          OTV_OPTIONAL_OFFSET( ExitAnchor  );

          OTV_SIZE_CHECK( EntryAnchor );
          if ( EntryAnchor )
            otv_Anchor_validate( table + EntryAnchor, otvalid );

          OTV_SIZE_CHECK( ExitAnchor );
          if ( ExitAnchor )
            otv_Anchor_validate( table + ExitAnchor, otvalid );
        }
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
  /*****                     GPOS LOOKUP TYPE 4                        *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  /* UNDOCUMENTED (in OpenType 1.5):              */
  /* BaseRecord tables can contain NULL pointers. */

  /* sets otvalid->extra2 (1) */

  static void
  otv_MarkBasePos_validate( FT_Bytes       table,
                            OTV_Validator  otvalid )
  {
    FT_Bytes  p = table;
    FT_UInt   PosFormat;


    OTV_NAME_ENTER( "MarkBasePos" );

    OTV_LIMIT_CHECK( 2 );
    PosFormat = FT_NEXT_USHORT( p );

    OTV_TRACE(( " (format %d)\n", PosFormat ));

    switch ( PosFormat )
    {
    case 1:
      otvalid->extra2 = 1;
      OTV_NEST2( MarkBasePosFormat1, BaseArray );
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
  /*****                     GPOS LOOKUP TYPE 5                        *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  /* sets otvalid->extra2 (1) */

  static void
  otv_MarkLigPos_validate( FT_Bytes       table,
                           OTV_Validator  otvalid )
  {
    FT_Bytes  p = table;
    FT_UInt   PosFormat;


    OTV_NAME_ENTER( "MarkLigPos" );

    OTV_LIMIT_CHECK( 2 );
    PosFormat = FT_NEXT_USHORT( p );

    OTV_TRACE(( " (format %d)\n", PosFormat ));

    switch ( PosFormat )
    {
    case 1:
      otvalid->extra2 = 1;
      OTV_NEST3( MarkLigPosFormat1, LigatureArray, LigatureAttach );
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
  /*****                     GPOS LOOKUP TYPE 6                        *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  /* sets otvalid->extra2 (0) */

  static void
  otv_MarkMarkPos_validate( FT_Bytes       table,
                            OTV_Validator  otvalid )
  {
    FT_Bytes  p = table;
    FT_UInt   PosFormat;


    OTV_NAME_ENTER( "MarkMarkPos" );

    OTV_LIMIT_CHECK( 2 );
    PosFormat = FT_NEXT_USHORT( p );

    OTV_TRACE(( " (format %d)\n", PosFormat ));

    switch ( PosFormat )
    {
    case 1:
      otvalid->extra2 = 0;
      OTV_NEST2( MarkMarkPosFormat1, Mark2Array );
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
  /*****                     GPOS LOOKUP TYPE 7                        *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  /* sets otvalid->extra1 (lookup count) */

  static void
  otv_ContextPos_validate( FT_Bytes       table,
                           OTV_Validator  otvalid )
  {
    FT_Bytes  p = table;
    FT_UInt   PosFormat;


    OTV_NAME_ENTER( "ContextPos" );

    OTV_LIMIT_CHECK( 2 );
    PosFormat = FT_NEXT_USHORT( p );

    OTV_TRACE(( " (format %d)\n", PosFormat ));

    switch ( PosFormat )
    {
    case 1:
      /* no need to check glyph indices/classes used as input for these */
      /* context rules since even invalid glyph indices/classes return  */
      /* meaningful results                                             */

      otvalid->extra1 = otvalid->lookup_count;
      OTV_NEST3( ContextPosFormat1, PosRuleSet, PosRule );
      OTV_RUN( table, otvalid );
      break;

    case 2:
      /* no need to check glyph indices/classes used as input for these */
      /* context rules since even invalid glyph indices/classes return  */
      /* meaningful results                                             */

      OTV_NEST3( ContextPosFormat2, PosClassSet, PosClassRule );
      OTV_RUN( table, otvalid );
      break;

    case 3:
      OTV_NEST1( ContextPosFormat3 );
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
  /*****                     GPOS LOOKUP TYPE 8                        *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  /* sets otvalid->extra1 (lookup count) */

  static void
  otv_ChainContextPos_validate( FT_Bytes       table,
                                OTV_Validator  otvalid )
  {
    FT_Bytes  p = table;
    FT_UInt   PosFormat;


    OTV_NAME_ENTER( "ChainContextPos" );

    OTV_LIMIT_CHECK( 2 );
    PosFormat = FT_NEXT_USHORT( p );

    OTV_TRACE(( " (format %d)\n", PosFormat ));

    switch ( PosFormat )
    {
    case 1:
      /* no need to check glyph indices/classes used as input for these */
      /* context rules since even invalid glyph indices/classes return  */
      /* meaningful results                                             */

      otvalid->extra1 = otvalid->lookup_count;
      OTV_NEST3( ChainContextPosFormat1,
                 ChainPosRuleSet, ChainPosRule );
      OTV_RUN( table, otvalid );
      break;

    case 2:
      /* no need to check glyph indices/classes used as input for these */
      /* context rules since even invalid glyph indices/classes return  */
      /* meaningful results                                             */

      OTV_NEST3( ChainContextPosFormat2,
                 ChainPosClassSet, ChainPosClassRule );
      OTV_RUN( table, otvalid );
      break;

    case 3:
      OTV_NEST1( ChainContextPosFormat3 );
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
  /*****                     GPOS LOOKUP TYPE 9                        *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  /* uses otvalid->type_funcs */

  static void
  otv_ExtensionPos_validate( FT_Bytes       table,
                             OTV_Validator  otvalid )
  {
    FT_Bytes  p = table;
    FT_UInt   PosFormat;


    OTV_NAME_ENTER( "ExtensionPos" );

    OTV_LIMIT_CHECK( 2 );
    PosFormat = FT_NEXT_USHORT( p );

    OTV_TRACE(( " (format %d)\n", PosFormat ));

    switch ( PosFormat )
    {
    case 1:     /* ExtensionPosFormat1 */
      {
        FT_UInt            ExtensionLookupType;
        FT_ULong           ExtensionOffset;
        OTV_Validate_Func  validate;


        OTV_LIMIT_CHECK( 6 );
        ExtensionLookupType = FT_NEXT_USHORT( p );
        ExtensionOffset     = FT_NEXT_ULONG( p );

        if ( ExtensionLookupType == 0 || ExtensionLookupType >= 9 )
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


  static const OTV_Validate_Func  otv_gpos_validate_funcs[9] =
  {
    otv_SinglePos_validate,
    otv_PairPos_validate,
    otv_CursivePos_validate,
    otv_MarkBasePos_validate,
    otv_MarkLigPos_validate,
    otv_MarkMarkPos_validate,
    otv_ContextPos_validate,
    otv_ChainContextPos_validate,
    otv_ExtensionPos_validate
  };


  /* sets otvalid->type_count */
  /* sets otvalid->type_funcs */

  FT_LOCAL_DEF( void )
  otv_GPOS_subtable_validate( FT_Bytes       table,
                              OTV_Validator  otvalid )
  {
    otvalid->type_count = 9;
    otvalid->type_funcs = (OTV_Validate_Func*)otv_gpos_validate_funcs;

    otv_Lookup_validate( table, otvalid );
  }


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                          GPOS TABLE                           *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  /* sets otvalid->glyph_count */

  FT_LOCAL_DEF( void )
  otv_GPOS_validate( FT_Bytes      table,
                     FT_UInt       glyph_count,
                     FT_Validator  ftvalid )
  {
    OTV_ValidatorRec  validrec;
    OTV_Validator     otvalid = &validrec;
    FT_Bytes          p       = table;
    FT_UInt           table_size;
    FT_UShort         version;
    FT_UInt           ScriptList, FeatureList, LookupList;

    OTV_OPTIONAL_TABLE32( featureVariations );


    otvalid->root = ftvalid;

    FT_TRACE3(( "validating GPOS table\n" ));
    OTV_INIT;

    OTV_LIMIT_CHECK( 4 );

    if ( FT_NEXT_USHORT( p ) != 1 )  /* majorVersion */
      FT_INVALID_FORMAT;

    version = FT_NEXT_USHORT( p );   /* minorVersion */

    table_size = 10;
    switch ( version )
    {
    case 0:
      OTV_LIMIT_CHECK( 6 );
      break;

    case 1:
      OTV_LIMIT_CHECK( 10 );
      table_size += 4;
      break;

    default:
      FT_INVALID_FORMAT;
    }

    ScriptList  = FT_NEXT_USHORT( p );
    FeatureList = FT_NEXT_USHORT( p );
    LookupList  = FT_NEXT_USHORT( p );

    otvalid->type_count  = 9;
    otvalid->type_funcs  = (OTV_Validate_Func*)otv_gpos_validate_funcs;
    otvalid->glyph_count = glyph_count;

    otv_LookupList_validate( table + LookupList,
                             otvalid );
    otv_FeatureList_validate( table + FeatureList, table + LookupList,
                              otvalid );
    otv_ScriptList_validate( table + ScriptList, table + FeatureList,
                             otvalid );

    if ( version > 0 )
    {
      OTV_OPTIONAL_OFFSET32( featureVariations );
      OTV_SIZE_CHECK32( featureVariations );
      if ( featureVariations )
        OTV_TRACE(( "  [omitting featureVariations validation]\n" )); /* XXX */
    }

    FT_TRACE4(( "\n" ));
  }


/* END */
