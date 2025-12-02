/****************************************************************************
 *
 * ttgpos.c
 *
 *   Routines to parse and access the 'GPOS' table for simple kerning (body).
 *
 * Copyright (C) 2025 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#include <freetype/freetype.h>
#include <freetype/tttables.h>
#include <freetype/tttags.h>

#include <freetype/internal/ftdebug.h>
#include <freetype/internal/ftstream.h>

#include "ttgpos.h"


#ifdef TT_CONFIG_OPTION_GPOS_KERNING


  /**************************************************************************
   *
   * The macro FT_COMPONENT is used in trace mode.  It is an implicit
   * parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log
   * messages during execution.
   */
#undef  FT_COMPONENT
#define FT_COMPONENT  ttgpos


  /*********************************/
  /********                 ********/
  /******** GPOS validation ********/
  /********                 ********/
  /*********************************/

  static FT_Bool
  tt_face_validate_coverage( FT_Byte*  table,
                             FT_Byte*  table_limit,
                             FT_UInt   max_num_coverage_indices )
  {
    FT_UInt  format;

    FT_Byte*  p = table;
    FT_Byte*  limit;

    FT_Long  last_id = -1;


    if ( table_limit < p + 4 )
      return FALSE;

    format = FT_NEXT_USHORT( p );
    if ( format == 1 )
    {
      FT_UInt  glyphCount = FT_NEXT_USHORT( p );


      if ( glyphCount > max_num_coverage_indices )
        return FALSE;

      limit = p + glyphCount * 2;
      if ( table_limit < limit )
        return FALSE;

      while ( p < limit )
      {
        FT_UInt  id = FT_NEXT_USHORT( p );


        if ( last_id >= id )
          return FALSE;
        last_id = id;
      }
    }
    else if ( format == 2 )
    {
      FT_UInt  rangeCount = FT_NEXT_USHORT( p );


      limit = p + rangeCount * 6;
      if ( table_limit < limit )
        return FALSE;

      while ( p < limit )
      {
        FT_UInt  startGlyphID       = FT_NEXT_USHORT( p );
        FT_UInt  endGlyphID         = FT_NEXT_USHORT( p );
        FT_UInt  startCoverageIndex = FT_NEXT_USHORT( p );


        if ( startGlyphID > endGlyphID )
          return FALSE;

        if ( last_id >= startGlyphID )
          return FALSE;
        last_id = endGlyphID;

        /* XXX: Is this modulo 65536 arithmetic? */
        if ( startCoverageIndex + endGlyphID - startGlyphID >=
               max_num_coverage_indices )
          return FALSE;
      }
    }
    else
      return FALSE;

    return TRUE;
  }


  static FT_Bool
  tt_face_validate_class_def( FT_Byte*  table,
                              FT_Byte*  table_limit,
                              FT_UInt   num_classes )
  {
    FT_UInt  format;

    FT_Byte*  p = table;
    FT_Byte*  limit;

    FT_UInt  max_class_value = 0;


    if ( table_limit < p + 2 )
      return FALSE;

    format = FT_NEXT_USHORT( p );
    if ( format == 1 )
    {
      FT_UInt  glyphCount;


      if ( table_limit < p + 4 )
        return FALSE;

      p += 2; /* Skip `startGlyphID`. */

      glyphCount = FT_NEXT_USHORT( p );
      limit      = p + glyphCount * 2;
      if ( table_limit < limit )
        return FALSE;

      while ( p < limit )
      {
        FT_UInt  class_value = FT_NEXT_USHORT( p );


        if ( class_value > max_class_value )
          max_class_value = class_value;
      }
    }
    else if ( format == 2 )
    {
      FT_UInt  classRangeCount;
      FT_Long  last_id = -1;


      if ( table_limit < p + 2 )
        return FALSE;

      classRangeCount = FT_NEXT_USHORT( p );
      limit           = p + classRangeCount * 6;
      if ( table_limit < limit )
        return FALSE;

      while ( p < limit )
      {
        FT_UInt  startGlyphID = FT_NEXT_USHORT( p );
        FT_UInt  endGlyphID   = FT_NEXT_USHORT( p );
        FT_UInt  class_value  = FT_NEXT_USHORT( p );


        if ( startGlyphID > endGlyphID )
          return FALSE;

        if ( last_id >= startGlyphID )
          return FALSE;
        last_id = endGlyphID;

        if ( class_value > max_class_value )
          max_class_value = class_value;
      }
    }
    else
      return FALSE;

    if ( max_class_value + 1 != num_classes )
      return FALSE;

    return TRUE;
  }


  static FT_Bool
  tt_face_validate_feature( FT_Byte*  table,
                            FT_Byte*  table_limit,
                            FT_UInt   use_lookup_table_size,
                            FT_Byte*  use_lookup_table )
  {
    FT_UInt  lookupIndexCount;

    FT_Byte*  p = table;
    FT_Byte*  limit;


    if ( table_limit < p + 4 )
      return FALSE;

    p += 2; /* Skip `featureParamsOffset`. */

    lookupIndexCount = FT_NEXT_USHORT( p );
    limit            = p + lookupIndexCount * 2;
    if ( table_limit < limit )
      return FALSE;

    while ( p < limit )
    {
      FT_UInt  lookup_index = FT_NEXT_USHORT( p );


      if ( lookup_index >= use_lookup_table_size )
        return FALSE;

      use_lookup_table[lookup_index] = TRUE;
    }

    return TRUE;
  }


  static FT_Bool
  tt_face_validate_feature_table( FT_Byte*  table,
                                  FT_Byte*  table_limit,
                                  FT_UInt   use_lookup_table_size,
                                  FT_Byte*  use_lookup_table )
  {
    FT_UInt  featureCount;

    FT_Byte*  p = table;
    FT_Byte*  limit;


    if ( table_limit < p + 2 )
      return FALSE;

    featureCount = FT_NEXT_USHORT( p );
    limit        = p + featureCount * 6;
    if ( table_limit < limit )
      return FALSE;

    /* We completely ignore GPOS script information      */
    /* and collect lookup tables of all 'kern' features. */
    while ( p < limit )
    {
      FT_ULong  featureTag    = FT_NEXT_ULONG( p );
      FT_UInt   featureOffset = FT_NEXT_USHORT( p );


      if ( featureTag == TTAG_kern )
      {
        if ( !tt_face_validate_feature( table + featureOffset,
                                        table_limit,
                                        use_lookup_table_size,
                                        use_lookup_table ) )
          return FALSE;
      }
    }

    return TRUE;
  }


  static FT_Bool
  tt_face_validate_pair_set( FT_Byte*  table,
                             FT_Byte*  table_limit )
  {
    FT_UInt  pairValueCount;

    FT_Byte*  p = table;
    FT_Byte*  limit;

    FT_Long  last_id = -1;


    if ( table_limit < p + 2 )
      return FALSE;

    /* For our purposes, the first value record only contains X advances */
    /* while the second one is empty; a `PairValue` record has thus a    */
    /* size of four bytes.                                               */
    pairValueCount = FT_NEXT_USHORT( p );
    limit          = p + pairValueCount * 4;
    if ( table_limit < limit )
      return FALSE;

    /* We validate the order of `secondGlyph` so that binary search works. */
    while ( p < limit )
    {
      FT_UInt  id = FT_NEXT_USHORT( p );


      if ( last_id >= id )
        return FALSE;

      last_id = id;

      p += 2; /* Skip `valueRecord1`. */
    }

    return TRUE;
  }


  static FT_Bool
  tt_face_validate_pair_pos1( FT_Byte*  table,
                              FT_Byte*  table_limit,
                              FT_Bool*  is_fitting )
  {
    FT_Byte*  coverage;
    FT_UInt   valueFormat1;
    FT_UInt   valueFormat2;

    /* Subtable format is already checked. */
    FT_Byte*  p = table + 2;
    FT_Byte*  limit;


    /* The six bytes for the coverage table offset */
    /* and the value formats are already checked.  */
    coverage = table + FT_NEXT_USHORT( p );

    /* For the limited purpose of accessing the simplest type of kerning */
    /* (similar to what FreeType's 'kern' table handling provides) we    */
    /* only consider tables that contains X advance values for the first */
    /* glyph and no data for the second glyph.                           */
    valueFormat1 = FT_NEXT_USHORT( p );
    valueFormat2 = FT_NEXT_USHORT( p );
    if ( valueFormat1 == 0x4 && valueFormat2 == 0 )
    {
      FT_UInt  pairSetCount;


      if ( table_limit < p + 2 )
        return FALSE;

      pairSetCount = FT_NEXT_USHORT( p );
      limit        = p + pairSetCount * 2;
      if ( table_limit < limit )
        return FALSE;

      if ( !tt_face_validate_coverage( coverage,
                                       table_limit,
                                       pairSetCount ) )
        return FALSE;

      while ( p < limit )
      {
        FT_Byte*  pair_set = table + FT_NEXT_USHORT( p );


        if ( !tt_face_validate_pair_set( pair_set, table_limit ) )
          return FALSE;
      }

      *is_fitting = TRUE;
    }

    return TRUE;
  }


  static FT_Bool
  tt_face_validate_pair_pos2( FT_Byte*  table,
                              FT_Byte*  table_limit,
                              FT_Bool*  is_fitting )
  {
    FT_Byte*  coverage;
    FT_UInt   valueFormat1;
    FT_UInt   valueFormat2;

    /* Subtable format is already checked. */
    FT_Byte*  p = table + 2;
    FT_Byte*  limit;


    /* The six bytes for the coverage table offset */
    /* and the value formats are already checked.  */
    coverage = table + FT_NEXT_USHORT( p );

    valueFormat1 = FT_NEXT_USHORT( p );
    valueFormat2 = FT_NEXT_USHORT( p );
    if ( valueFormat1 == 0x4 && valueFormat2 == 0 )
    {
      FT_Byte*  class_def1;
      FT_Byte*  class_def2;
      FT_UInt   class1Count;
      FT_UInt   class2Count;


      /* The number of coverage indices is not relevant here. */
      if ( !tt_face_validate_coverage( coverage, table_limit, FT_UINT_MAX ) )
        return FALSE;

      if ( table_limit < p + 8 )
        return FALSE;

      class_def1  = table + FT_NEXT_USHORT( p );
      class_def2  = table + FT_NEXT_USHORT( p );
      class1Count = FT_NEXT_USHORT( p );
      class2Count = FT_NEXT_USHORT( p );

      if ( !tt_face_validate_class_def( class_def1,
                                        table_limit,
                                        class1Count ) )
        return FALSE;
      if ( !tt_face_validate_class_def( class_def2,
                                        table_limit,
                                        class2Count ) )
        return FALSE;

      /* For our purposes, the first value record only contains */
      /* X advances while the second one is empty.              */
      limit = p + class1Count * class2Count * 2;
      if ( table_limit < limit )
        return FALSE;

      *is_fitting = TRUE;
    }

    return TRUE;
  }


  /* The return value is the number of fitting subtables. */
  static FT_UInt
  tt_face_validate_lookup_table( FT_Byte*  table,
                                 FT_Byte*  table_limit )
  {
    FT_UInt  lookupType;
    FT_UInt  real_lookupType = 0;
    FT_UInt  subtableCount;

    FT_Byte*  p = table;
    FT_Byte*  limit;

    FT_UInt  num_fitting_subtables = 0;


    if ( table_limit < p + 6 )
      return 0;

    lookupType = FT_NEXT_USHORT( p );

    p += 2; /* Skip `lookupFlag`. */

    subtableCount = FT_NEXT_USHORT( p );
    limit         = p + subtableCount * 2;
    if ( table_limit < limit )
      return 0;

    while ( p < limit )
    {
      FT_Byte*  subtable = table + FT_NEXT_USHORT( p );
      FT_UInt   format;

      FT_Bool  is_fitting = FALSE;


      if ( lookupType == 9 )
      {
        /* Positioning extension. */
        FT_Byte*  q = subtable;


        if ( table_limit < q + 8 )
          return 0;

        if ( FT_NEXT_USHORT( q ) != 1 ) /* format */
          return 0;

        if ( real_lookupType == 0 )
          real_lookupType = FT_NEXT_USHORT( q );
        else if ( real_lookupType != FT_NEXT_USHORT( q ) )
          return 0;

        subtable += FT_PEEK_ULONG( q );
      }
      else
        real_lookupType = lookupType;

      /* Ensure the first eight bytes of the subtable formats. */
      if ( table_limit < subtable + 8 )
        return 0;

      format = FT_PEEK_USHORT( subtable );

      if ( real_lookupType == 2 )
      {
        if ( format == 1 )
        {
          if ( !tt_face_validate_pair_pos1( subtable,
                                            table_limit,
                                            &is_fitting ) )
            return 0;
        }
        else if ( format == 2 )
        {
          if ( !tt_face_validate_pair_pos2( subtable,
                                            table_limit,
                                            &is_fitting ) )
            return 0;
        }
        else
          return 0;
      }
      else
        return 0;

      if ( is_fitting )
        num_fitting_subtables++;
    }

    return num_fitting_subtables;
  }


  static void
  tt_face_get_subtable_offsets( FT_Byte*    table,
                                FT_Byte*    gpos,
                                FT_UInt32*  gpos_lookups_kerning,
                                FT_UInt*    idx )
  {
    FT_UInt  lookupType;
    FT_UInt  subtableCount;

    FT_Byte*  p = table;
    FT_Byte*  limit;


    lookupType = FT_NEXT_USHORT( p );

    p += 2;

    subtableCount = FT_NEXT_USHORT( p );
    limit         = p + subtableCount * 2;
    while ( p < limit )
    {
      FT_Byte*  subtable = table + FT_NEXT_USHORT( p );
      FT_UInt   valueFormat1;
      FT_UInt   valueFormat2;


      if ( lookupType == 9 )
        subtable += FT_PEEK_ULONG( subtable + 4 );

      /* Table offsets for `valueFormat[12]` values */
      /* are identical for both subtable formats.   */
      valueFormat1 = FT_PEEK_USHORT( subtable + 4 );
      valueFormat2 = FT_PEEK_USHORT( subtable + 6 );
      if ( valueFormat1 == 0x4 && valueFormat2 == 0 )
      {
        /* We store offsets relative to the start of the GPOS table. */
        gpos_lookups_kerning[(*idx)++] = (FT_UInt32)( subtable - gpos );
      }
    }
  }


  FT_LOCAL_DEF( FT_Error )
  tt_face_load_gpos( TT_Face    face,
                     FT_Stream  stream )
  {
    FT_Error   error;
    FT_Memory  memory = face->root.memory;

    FT_ULong  gpos_length;
    FT_Byte*  gpos;
    FT_Byte*  gpos_limit;

    FT_UInt32*  gpos_lookups_kerning;

    FT_UInt  featureListOffset;

    FT_UInt   lookupListOffset;
    FT_Byte*  lookup_list;
    FT_UInt   lookupCount;

    FT_UInt  i;

    FT_Byte*  use_lookup_table = NULL;
    FT_UInt   num_fitting_subtables;

    FT_Byte*  p;
    FT_Byte*  limit;


    face->gpos_table               = NULL;
    face->gpos_lookups_kerning     = NULL;
    face->num_gpos_lookups_kerning = 0;

    gpos                 = NULL;
    gpos_lookups_kerning = NULL;

    error = face->goto_table( face, TTAG_GPOS, stream, &gpos_length );
    if ( error )
      goto Fail;

    if ( FT_FRAME_EXTRACT( gpos_length, gpos ) )
      goto Fail;

    if ( gpos_length < 10 )
      goto Fail;

    gpos_limit = gpos + gpos_length;

    /* We first need the number of GPOS lookups. */
    lookupListOffset = FT_PEEK_USHORT( gpos + 8 );

    lookup_list = gpos + lookupListOffset;
    p           = lookup_list;
    if ( gpos_limit < p + 2 )
      goto Fail;

    lookupCount = FT_NEXT_USHORT( p );
    limit       = p + lookupCount * 2;
    if ( gpos_limit < limit )
      goto Fail;

    /* Allocate an auxiliary array for Boolean values that */
    /* gets filled while walking over all 'kern' features. */
    if ( FT_NEW_ARRAY( use_lookup_table, lookupCount ) )
      goto Fail;

    featureListOffset = FT_PEEK_USHORT( gpos + 6 );

    if ( !tt_face_validate_feature_table( gpos + featureListOffset,
                                          gpos_limit,
                                          lookupCount,
                                          use_lookup_table ) )
      goto Fail;

    /* Now walk over all lookup tables and get the */
    /* number of fitting subtables.                */
    num_fitting_subtables = 0;
    for ( i = 0; i < lookupCount; i++ )
    {
      FT_UInt  lookupOffset;


      if ( !use_lookup_table[i] )
        continue;

      lookupOffset = FT_PEEK_USHORT( p + i * 2 );

      num_fitting_subtables +=
        tt_face_validate_lookup_table( lookup_list + lookupOffset,
                                       gpos_limit );

    }

    /* Loop again over all lookup tables and */
    /* collect offsets to those subtables.   */
    if ( num_fitting_subtables )
    {
      FT_UInt  idx;


      if ( FT_QNEW_ARRAY( gpos_lookups_kerning, num_fitting_subtables ) )
        goto Fail;

      idx = 0;
      for ( i = 0; i < lookupCount; i++ )
      {
        FT_UInt  lookupOffset;


        if ( !use_lookup_table[i] )
          continue;

        lookupOffset = FT_PEEK_USHORT( p + i * 2 );

        tt_face_get_subtable_offsets( lookup_list + lookupOffset,
                                      gpos,
                                      gpos_lookups_kerning,
                                      &idx );
      }
    }

    FT_FREE( use_lookup_table );
    use_lookup_table = NULL;

    face->gpos_table               = gpos;
    face->gpos_lookups_kerning     = gpos_lookups_kerning;
    face->num_gpos_lookups_kerning = num_fitting_subtables;

  Exit:
    return error;

  Fail:
    FT_FREE( gpos );
    FT_FREE( gpos_lookups_kerning );
    FT_FREE( use_lookup_table );

    /* If we don't have an explicit error code, set it to a generic value. */
    if ( !error )
      error = FT_THROW( Invalid_Table );

    goto Exit;
  }


  FT_LOCAL_DEF( void )
  tt_face_done_gpos( TT_Face  face )
  {
    FT_Stream  stream = face->root.stream;
    FT_Memory  memory = face->root.memory;


    FT_FRAME_RELEASE( face->gpos_table );
    FT_FREE( face->gpos_lookups_kerning );
  }


  /*********************************/
  /********                 ********/
  /********   GPOS access   ********/
  /********                 ********/
  /*********************************/


  static FT_Long
  tt_face_get_coverage_index( FT_Byte*  table,
                              FT_UInt   glyph_index )
  {
    FT_Byte*  p      = table;
    FT_UInt   format = FT_NEXT_USHORT( p );
    FT_UInt   count  = FT_NEXT_USHORT( p );

    FT_UInt  min, max;


    min = 0;
    max = count;

    if ( format == 1 )
    {
      while ( min < max )
      {
        FT_UInt  mid       = min + ( max - min ) / 2;
        FT_UInt  mid_index = FT_PEEK_USHORT( p + mid * 2 );


        if ( glyph_index > mid_index )
          min = mid + 1;
        else if ( glyph_index < mid_index )
          max = mid;
        else
          return mid;
      }
    }
    else
    {
      while ( min < max )
      {
        FT_UInt  mid          = min + ( max - min ) / 2;
        FT_UInt  startGlyphID = FT_PEEK_USHORT( p + mid * 6 );
        FT_UInt  endGlyphID   = FT_PEEK_USHORT( p + mid * 6 + 2 );


        if ( glyph_index > endGlyphID )
          min = mid + 1;
        else if ( glyph_index < startGlyphID )
          max = mid;
        else
        {
          FT_UInt  startCoverageIndex = FT_PEEK_USHORT( p + mid * 6 + 4 );


          return startCoverageIndex + glyph_index - startGlyphID;
        }
      }
    }

    return -1;
  }


  static FT_UInt
  tt_face_get_class( FT_Byte*  table,
                     FT_UInt   glyph_index )
  {
    FT_Byte*  p      = table;
    FT_UInt   format = FT_NEXT_USHORT( p );


    if ( format == 1 )
    {
      FT_UInt  startGlyphID = FT_NEXT_USHORT( p );
      FT_UInt  glyphCount   = FT_NEXT_USHORT( p );


      /* XXX: Is this modulo 65536 arithmetic? */
      if ( startGlyphID              <= glyph_index &&
           startGlyphID + glyphCount >= glyph_index )
        return FT_PEEK_USHORT( p + ( glyph_index - startGlyphID ) * 2 );
    }
    else
    {
      FT_UInt  count = FT_NEXT_USHORT( p );

      FT_UInt  min, max;


      min = 0;
      max = count;

      while ( min < max )
      {
        FT_UInt  mid          = min + ( max - min ) / 2;
        FT_UInt  startGlyphID = FT_PEEK_USHORT( p + mid * 6 );
        FT_UInt  endGlyphID   = FT_PEEK_USHORT( p + mid * 6 + 2 );


        if ( glyph_index > endGlyphID )
          min = mid + 1;
        else if ( glyph_index < startGlyphID )
          max = mid;
        else
          return FT_PEEK_USHORT( p + mid * 6 + 4 );
      }
    }

    return 0;
  }


  static FT_Bool
  tt_face_get_pair_pos1_kerning( FT_Byte*  table,
                                 FT_UInt   first_glyph,
                                 FT_UInt   second_glyph,
                                 FT_Int*   kerning )
  {
    FT_Byte*  coverage       = table + FT_PEEK_USHORT( table + 2 );
    FT_Long   coverage_index = tt_face_get_coverage_index( coverage,
                                                           first_glyph );

    FT_UInt   pair_set_offset;
    FT_Byte*  p;
    FT_UInt   count;

    FT_UInt  min, max;


    if ( coverage_index < 0 )
      return FALSE;

    pair_set_offset = FT_PEEK_USHORT( table + 10 + coverage_index * 2 );
    p               = table + pair_set_offset;
    count           = FT_NEXT_USHORT( p );

    min = 0;
    max = count;

    while ( min < max )
    {
      FT_UInt  mid       = min + ( max - min ) / 2;
      FT_UInt  mid_index = FT_PEEK_USHORT( p + mid * 4 );


      if ( second_glyph > mid_index )
        min = max + 1;
      else if ( second_glyph < mid_index )
        max = mid;
      else
      {
        *kerning = FT_PEEK_SHORT( p + mid * 4 + 2 );

        return TRUE;
      }
    }

    return FALSE;
  }


  static FT_Bool
  tt_face_get_pair_pos2_kerning( FT_Byte*  table,
                                 FT_UInt   first_glyph,
                                 FT_UInt   second_glyph,
                                 FT_Int*   kerning )
  {
    FT_Byte*  coverage       = table + FT_PEEK_USHORT( table + 2 );
    FT_Long   coverage_index = tt_face_get_coverage_index( coverage,
                                                           first_glyph );

    FT_Byte*  class_def1;
    FT_Byte*  class_def2;
    FT_UInt   first_class;
    FT_UInt   second_class;
    FT_UInt   class2Count;


    if ( coverage_index < 0 )
      return FALSE;

    class_def1 = table + FT_PEEK_USHORT( table + 8 );
    class_def2 = table + FT_PEEK_USHORT( table + 10 );

    class2Count = FT_PEEK_USHORT( table + 14 );

    first_class  = tt_face_get_class( class_def1, first_glyph );
    second_class = tt_face_get_class( class_def2, second_glyph );

    *kerning =
      FT_PEEK_SHORT( table + 16 +
                     ( first_class * class2Count + second_class ) * 2 );

    return TRUE;
  }


  FT_LOCAL_DEF( FT_Int )
  tt_face_get_gpos_kerning( TT_Face  face,
                            FT_UInt  first_glyph,
                            FT_UInt  second_glyph )
  {
    FT_Int  kerning = 0;

    FT_UInt  i;


    /* We only have `PairPos` subtables. */
    for ( i = 0; i < face->num_gpos_lookups_kerning; i++ )
    {
      FT_Byte*  subtable = face->gpos_table + face->gpos_lookups_kerning[i];
      FT_Byte*  p        = subtable;

      FT_UInt  format = FT_NEXT_USHORT( p );


      if ( format == 1 )
      {
        if ( tt_face_get_pair_pos1_kerning( subtable,
                                            first_glyph,
                                            second_glyph,
                                            &kerning ) )
          break;
      }
      else
      {
        if ( tt_face_get_pair_pos2_kerning( subtable,
                                            first_glyph,
                                            second_glyph,
                                            &kerning ) )
         break;
      }
    }

    return kerning;
  }

#else /* !TT_CONFIG_OPTION_GPOS_KERNING */

  /* ANSI C doesn't like empty source files */
  typedef int  tt_gpos_dummy_;

#endif /* !TT_CONFIG_OPTION_GPOS_KERNING */


/* END */
