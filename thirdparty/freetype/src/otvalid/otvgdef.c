/****************************************************************************
 *
 * otvgdef.c
 *
 *   OpenType GDEF table validation (body).
 *
 * Copyright (C) 2004-2023 by
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


  /**************************************************************************
   *
   * The macro FT_COMPONENT is used in trace mode.  It is an implicit
   * parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log
   * messages during execution.
   */
#undef  FT_COMPONENT
#define FT_COMPONENT  otvgdef


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                      UTILITY FUNCTIONS                        *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

#define AttachListFunc    otv_O_x_Ox
#define LigCaretListFunc  otv_O_x_Ox

  /* sets valid->extra1 (0)           */

  static void
  otv_O_x_Ox( FT_Bytes       table,
              OTV_Validator  otvalid )
  {
    FT_Bytes           p = table;
    FT_Bytes           Coverage;
    FT_UInt            GlyphCount;
    OTV_Validate_Func  func;


    OTV_ENTER;

    OTV_LIMIT_CHECK( 4 );
    Coverage   = table + FT_NEXT_USHORT( p );
    GlyphCount = FT_NEXT_USHORT( p );

    OTV_TRACE(( " (GlyphCount = %d)\n", GlyphCount ));

    otv_Coverage_validate( Coverage, otvalid, (FT_Int)GlyphCount );
    if ( GlyphCount != otv_Coverage_get_count( Coverage ) )
      FT_INVALID_DATA;

    OTV_LIMIT_CHECK( GlyphCount * 2 );

    otvalid->nesting_level++;
    func            = otvalid->func[otvalid->nesting_level];
    otvalid->extra1 = 0;

    for ( ; GlyphCount > 0; GlyphCount-- )
      func( table + FT_NEXT_USHORT( p ), otvalid );

    otvalid->nesting_level--;

    OTV_EXIT;
  }


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                       LIGATURE CARETS                         *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

#define CaretValueFunc  otv_CaretValue_validate

  static void
  otv_CaretValue_validate( FT_Bytes       table,
                           OTV_Validator  otvalid )
  {
    FT_Bytes  p = table;
    FT_UInt   CaretValueFormat;


    OTV_ENTER;

    OTV_LIMIT_CHECK( 4 );

    CaretValueFormat = FT_NEXT_USHORT( p );

    OTV_TRACE(( " (format = %d)\n", CaretValueFormat ));

    switch ( CaretValueFormat )
    {
    case 1:     /* CaretValueFormat1 */
      /* skip Coordinate, no test */
      break;

    case 2:     /* CaretValueFormat2 */
      /* skip CaretValuePoint, no test */
      break;

    case 3:     /* CaretValueFormat3 */
      p += 2;   /* skip Coordinate */

      OTV_LIMIT_CHECK( 2 );

      /* DeviceTable */
      otv_Device_validate( table + FT_NEXT_USHORT( p ), otvalid );
      break;

    default:
      FT_INVALID_FORMAT;
    }

    OTV_EXIT;
  }


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                       MARK GLYPH SETS                         *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  static void
  otv_MarkGlyphSets_validate( FT_Bytes       table,
                              OTV_Validator  otvalid )
  {
    FT_Bytes  p = table;
    FT_UInt   MarkGlyphSetCount;


    OTV_NAME_ENTER( "MarkGlyphSets" );

    p += 2;     /* skip Format */

    OTV_LIMIT_CHECK( 2 );
    MarkGlyphSetCount = FT_NEXT_USHORT( p );

    OTV_TRACE(( " (MarkGlyphSetCount = %d)\n", MarkGlyphSetCount ));

    OTV_LIMIT_CHECK( MarkGlyphSetCount * 4 );      /* CoverageOffsets */

    for ( ; MarkGlyphSetCount > 0; MarkGlyphSetCount-- )
      otv_Coverage_validate( table + FT_NEXT_ULONG( p ), otvalid, -1 );

    OTV_EXIT;
  }


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                         GDEF TABLE                            *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  /* sets otvalid->glyph_count */

  FT_LOCAL_DEF( void )
  otv_GDEF_validate( FT_Bytes      table,
                     FT_Bytes      gsub,
                     FT_Bytes      gpos,
                     FT_UInt       glyph_count,
                     FT_Validator  ftvalid )
  {
    OTV_ValidatorRec  otvalidrec;
    OTV_Validator     otvalid = &otvalidrec;
    FT_Bytes          p       = table;
    FT_UInt           table_size;
    FT_UShort         version;
    FT_Bool           need_MarkAttachClassDef = 1;

    OTV_OPTIONAL_TABLE( GlyphClassDef );
    OTV_OPTIONAL_TABLE( AttachListOffset );
    OTV_OPTIONAL_TABLE( LigCaretListOffset );
    OTV_OPTIONAL_TABLE( MarkAttachClassDef );
    OTV_OPTIONAL_TABLE( MarkGlyphSetsDef );

    OTV_OPTIONAL_TABLE32( itemVarStore );


    otvalid->root = ftvalid;

    FT_TRACE3(( "validating GDEF table\n" ));
    OTV_INIT;

    OTV_LIMIT_CHECK( 4 );

    if ( FT_NEXT_USHORT( p ) != 1 )  /* majorVersion */
      FT_INVALID_FORMAT;

    version = FT_NEXT_USHORT( p );   /* minorVersion */

    table_size = 10;
    switch ( version )
    {
    case 0:
      /* MarkAttachClassDef has been added to the OpenType */
      /* specification without increasing GDEF's version,  */
      /* so we use this ugly hack to find out whether the  */
      /* table is needed actually.                         */

      need_MarkAttachClassDef = FT_BOOL(
        otv_GSUBGPOS_have_MarkAttachmentType_flag( gsub ) ||
        otv_GSUBGPOS_have_MarkAttachmentType_flag( gpos ) );

      if ( need_MarkAttachClassDef )
      {
        OTV_LIMIT_CHECK( 8 );
        table_size += 2;
      }
      else
        OTV_LIMIT_CHECK( 6 );  /* OpenType < 1.2 */

      break;

    case 2:
      OTV_LIMIT_CHECK( 10 );
      table_size += 4;
      break;

    case 3:
      OTV_LIMIT_CHECK( 14 );
      table_size += 8;
      break;

    default:
      FT_INVALID_FORMAT;
    }

    otvalid->glyph_count = glyph_count;

    OTV_OPTIONAL_OFFSET( GlyphClassDef );
    OTV_SIZE_CHECK( GlyphClassDef );
    if ( GlyphClassDef )
      otv_ClassDef_validate( table + GlyphClassDef, otvalid );

    OTV_OPTIONAL_OFFSET( AttachListOffset );
    OTV_SIZE_CHECK( AttachListOffset );
    if ( AttachListOffset )
    {
      OTV_NEST2( AttachList, AttachPoint );
      OTV_RUN( table + AttachListOffset, otvalid );
    }

    OTV_OPTIONAL_OFFSET( LigCaretListOffset );
    OTV_SIZE_CHECK( LigCaretListOffset );
    if ( LigCaretListOffset )
    {
      OTV_NEST3( LigCaretList, LigGlyph, CaretValue );
      OTV_RUN( table + LigCaretListOffset, otvalid );
    }

    if ( need_MarkAttachClassDef )
    {
      OTV_OPTIONAL_OFFSET( MarkAttachClassDef );
      OTV_SIZE_CHECK( MarkAttachClassDef );
      if ( MarkAttachClassDef )
        otv_ClassDef_validate( table + MarkAttachClassDef, otvalid );
    }

    if ( version > 0 )
    {
      OTV_OPTIONAL_OFFSET( MarkGlyphSetsDef );
      OTV_SIZE_CHECK( MarkGlyphSetsDef );
      if ( MarkGlyphSetsDef )
        otv_MarkGlyphSets_validate( table + MarkGlyphSetsDef, otvalid );
    }

    if ( version > 2 )
    {
      OTV_OPTIONAL_OFFSET32( itemVarStore );
      OTV_SIZE_CHECK32( itemVarStore );
      if ( itemVarStore )
        OTV_TRACE(( "  [omitting itemVarStore validation]\n" )); /* XXX */
    }

    FT_TRACE4(( "\n" ));
  }


/* END */
