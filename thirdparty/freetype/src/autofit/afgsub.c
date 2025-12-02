/****************************************************************************
 *
 * afgsub.c
 *
 *   Auto-fitter routines to parse the GSUB table (body).
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

#ifdef FT_CONFIG_OPTION_USE_HARFBUZZ


#include <freetype/freetype.h>
#include <freetype/tttables.h>
#include <freetype/tttags.h>

#include <freetype/internal/ftstream.h>

#include "afglobal.h"
#include "afgsub.h"
#include "aftypes.h"


  /*********************************/
  /********                 ********/
  /******** GSUB validation ********/
  /********                 ********/
  /*********************************/


  static FT_Bool
  af_validate_coverage( FT_Byte*  table,
                        FT_Byte*  table_limit,
                        FT_UInt  *num_glyphs )
  {
    FT_UInt  format;

    FT_Byte*  p     = table;
    FT_UInt   count = 0;


    if ( table_limit < p + 4 )
      return FALSE;

    format = FT_NEXT_USHORT( p );
    if ( format == 1 )
    {
      FT_UInt  glyphCount = FT_NEXT_USHORT( p );


      /* We don't validate glyph IDs. */
      if ( table_limit < p + glyphCount * 2 )
        return FALSE;

      count += glyphCount;
    }
    else if ( format == 2 )
    {
      FT_UInt   rangeCount = FT_NEXT_USHORT( p );
      FT_Byte*  limit      = p + rangeCount * 6;


      if ( table_limit < limit )
        return FALSE;

      while ( p < limit )
      {
        FT_UInt  startGlyphID = FT_NEXT_USHORT( p );
        FT_UInt  endGlyphID   = FT_NEXT_USHORT( p );


        if ( startGlyphID > endGlyphID )
          return FALSE;

        count += endGlyphID - startGlyphID + 1;

        /* We don't validate coverage indices. */
        p += 2;
      }
    }
    else
      return FALSE;

    if ( num_glyphs )
      *num_glyphs = count;

    return TRUE;
  }


  static FT_Bool
  af_validate_single_subst1( FT_Byte*  table,
                             FT_Byte*  table_limit )
  {
    FT_Byte*  coverage;


    /* Subtable format is already checked. */

    /* The four bytes for the coverage table offset */
    /* and the glyph ID delta are already checked.  */
    coverage = table + FT_PEEK_USHORT( table + 2 );
    if ( !af_validate_coverage( coverage, table_limit, NULL ) )
      return FALSE;

    /* We don't validate glyph IDs. */

    return TRUE;
  }


  static FT_Bool
  af_validate_single_subst2( FT_Byte*  table,
                             FT_Byte*  table_limit )
  {
    FT_Byte*  coverage;
    FT_UInt   glyphCount;
    FT_UInt   num_glyphs;

    /* Subtable format is already checked. */
    FT_Byte*  p = table + 2;


    /* The four bytes for the coverage table offset */
    /* and `glyphCount` are already checked.        */
    coverage = table + FT_NEXT_USHORT( p );
    if ( !af_validate_coverage( coverage, table_limit, &num_glyphs ) )
      return FALSE;

    glyphCount = FT_NEXT_USHORT( p );
    /* We don't validate glyph IDs. */
    if ( table_limit < p + glyphCount * 2 )
      return FALSE;

    if ( glyphCount != num_glyphs )
      return FALSE;

    return TRUE;
  }


  static FT_Bool
  af_validate_alternate( FT_Byte*  table,
                         FT_Byte*  table_limit )
  {
    FT_Byte*  coverage;
    FT_UInt   alternateSetCount;
    FT_UInt   num_glyphs;

    /* Subtable format is already checked. */
    FT_Byte*  p = table + 2;
    FT_Byte*  limit;


    /* The four bytes for the coverage table offset */
    /* and `alternateSetCount` are already checked. */
    coverage = table + FT_NEXT_USHORT( p );
    if ( !af_validate_coverage( coverage, table_limit, &num_glyphs ) )
      return FALSE;

    alternateSetCount = FT_NEXT_USHORT( p );
    limit             = p + alternateSetCount * 2;
    if ( table_limit < limit )
      return FALSE;

    if ( alternateSetCount != num_glyphs )
      return FALSE;

    while ( p < limit )
    {
      FT_Byte*  alternate_set;
      FT_UInt   glyphCount;


      alternate_set = table + FT_NEXT_USHORT( p );
      if ( table_limit < alternate_set + 2 )
        return FALSE;

      glyphCount = FT_PEEK_USHORT( alternate_set );
      /* We don't validate glyph IDs. */
      if ( table_limit < alternate_set + 2 + glyphCount * 2 )
        return FALSE;
    }

    return TRUE;
  }


  /* Validate 'SingleSubst' and 'AlternateSubst' lookup tables. */
  static FT_Bool
  af_validate_lookup_table( FT_Byte*  table,
                            FT_Byte*  table_limit )
  {
    FT_UInt  lookupType;
    FT_UInt  real_lookupType = 0;
    FT_UInt  subtableCount;

    FT_Byte*  p = table;
    FT_Byte*  limit;


    if ( table_limit < p + 6 )
      return FALSE;

    lookupType = FT_NEXT_USHORT( p );

    p += 2; /* Skip `lookupFlag`. */

    subtableCount = FT_NEXT_USHORT( p );
    limit         = p + subtableCount * 2;
    if ( table_limit < limit )
      return FALSE;

    while ( p < limit )
    {
      FT_Byte*  subtable = table + FT_NEXT_USHORT( p );
      FT_UInt   format;


      if ( lookupType == 7 )
      {
        /* Substitution extension. */
        FT_Byte*  q = subtable;


        if ( table_limit < q + 8 )
          return FALSE;

        if ( FT_NEXT_USHORT( q ) != 1 ) /* format */
          return FALSE;

        if ( real_lookupType == 0 )
          real_lookupType = FT_NEXT_USHORT( q );
        else if ( real_lookupType != FT_NEXT_USHORT( q ) )
          return FALSE;

        subtable += FT_PEEK_ULONG( q );
      }
      else
        real_lookupType = lookupType;

      /* Ensure the first six bytes of all subtable formats. */
      if ( table_limit < subtable + 6 )
        return FALSE;

      format = FT_PEEK_USHORT( subtable );

      if ( real_lookupType == 1 )
      {
        if ( format == 1 )
        {
          if ( !af_validate_single_subst1( subtable, table_limit ) )
            return FALSE;
        }
        else if ( format == 2 )
        {
          if ( !af_validate_single_subst2( subtable, table_limit ) )
            return FALSE;
        }
        else
          return FALSE;
      }
      else if ( real_lookupType == 3 )
      {
        if ( format == 1 )
        {
          if ( !af_validate_alternate( subtable, table_limit ) )
            return FALSE;
        }
        else
          return FALSE;
      }
      else
        return FALSE;
    }

    return TRUE;
  }


  FT_LOCAL_DEF( void )
  af_parse_gsub( AF_FaceGlobals  globals )
  {
    FT_Error  error = FT_Err_Ok;

    FT_Face    face   = globals->face;
    FT_Memory  memory = face->memory;

    FT_ULong  gsub_length;
    FT_Byte*  gsub;
    FT_Byte*  gsub_limit;

    FT_UInt32*  gsub_lookups_single_alternate;

    FT_UInt   lookupListOffset;
    FT_Byte*  lookup_list;
    FT_UInt   lookupCount;

    FT_UInt  idx;

    FT_Byte*  p;
    FT_Byte*  limit;


    globals->gsub                          = NULL;
    globals->gsub_lookups_single_alternate = NULL;

    /* No error if we can't load or parse GSUB data. */

    gsub                          = NULL;
    gsub_lookups_single_alternate = NULL;

    gsub_length = 0;
    if ( FT_Load_Sfnt_Table( face, TTAG_GSUB, 0, NULL, &gsub_length ) )
      goto Fail;

    if ( FT_QALLOC( gsub, gsub_length ) )
      goto Fail;

    if ( FT_Load_Sfnt_Table( face, TTAG_GSUB, 0, gsub, &gsub_length ) )
      goto Fail;

    if ( gsub_length < 10 )
      goto Fail;

    lookupListOffset = FT_PEEK_USHORT( gsub + 8 );
    if ( gsub_length < lookupListOffset + 2 )
      goto Fail;

    lookupCount = FT_PEEK_USHORT( gsub + lookupListOffset );
    if ( gsub_length < lookupListOffset + 2 + lookupCount * 2 )
      goto Fail;

    if ( FT_NEW_ARRAY( gsub_lookups_single_alternate, lookupCount ) )
      goto Fail;

    gsub_limit  = gsub + gsub_length;
    lookup_list = gsub + lookupListOffset;
    p           = lookup_list + 2;
    limit       = p + lookupCount * 2;
    idx         = 0;
    while ( p < limit )
    {
      FT_UInt  lookupOffset = FT_NEXT_USHORT( p );


      if ( af_validate_lookup_table( lookup_list + lookupOffset,
                                     gsub_limit ) )
      {
        /* We store offsets relative to the start of the GSUB table. */
        gsub_lookups_single_alternate[idx] = lookupListOffset + lookupOffset;
      }

      idx++;
    }

    globals->gsub                          = gsub;
    globals->gsub_lookups_single_alternate = gsub_lookups_single_alternate;

    return;

  Fail:
    FT_FREE( gsub );
    FT_FREE( gsub_lookups_single_alternate );
  }


  /*********************************/
  /********                 ********/
  /********   GSUB access   ********/
  /********                 ********/
  /*********************************/


  static FT_UInt
  af_coverage_format( FT_Byte*  coverage )
  {
    return FT_PEEK_USHORT( coverage );
  }


  static FT_Byte*
  af_coverage_start( FT_Byte*  coverage )
  {
    return coverage + 4;
  }


  static FT_Byte*
  af_coverage_limit( FT_Byte*  coverage )
  {
    if ( af_coverage_format( coverage ) == 1 )
    {
      FT_UInt  glyphCount = FT_PEEK_USHORT( coverage + 2 );


      return af_coverage_start( coverage ) + glyphCount * 2;
    }
    else
    {
      FT_UInt  rangeCount = FT_PEEK_USHORT( coverage + 2 );


      return af_coverage_start( coverage ) + rangeCount * 6;
    }
  }


  typedef struct AF_CoverageIteratorRec_*  AF_CoverageIterator;

  typedef struct  AF_CoverageIteratorRec_
  {
    FT_UInt  format;

    FT_Byte*  p;
    FT_Byte*  limit;

    FT_UInt16  glyph;
    FT_UInt16  glyph_limit;

  } AF_CoverageIteratorRec;


  static FT_Bool
  af_coverage_iterator( AF_CoverageIterator  iter,
                        FT_UInt16*           glyph )
  {
    if ( iter->p >= iter->limit )
      return FALSE;

    if ( iter->format == 1 )
      *glyph = FT_NEXT_USHORT( iter->p );
    else
    {
      if ( iter->glyph > iter->glyph_limit )
      {
        iter->glyph       = FT_NEXT_USHORT( iter->p );
        iter->glyph_limit = FT_NEXT_USHORT( iter->p );

        iter->p += 2;
      }

      *glyph = iter->glyph++;
    }

    return TRUE;
  }


  static AF_CoverageIteratorRec
  af_coverage_iterator_init( FT_Byte*  coverage )
  {
    AF_CoverageIteratorRec  iterator;


    iterator.format      = af_coverage_format( coverage );
    iterator.p           = af_coverage_start( coverage );
    iterator.limit       = af_coverage_limit( coverage );
    iterator.glyph       = 1;
    iterator.glyph_limit = 0;

    return iterator;
  }


  /*
    Because we merge all single and alternate substitution mappings into
    one, large hash, we need the possibility to have multiple glyphs as
    values.  We utilize that we have 32bit integers but only 16bit glyph
    indices, using the following scheme.

    If glyph G maps to a single substitute S, the entry in the map is

      G  ->  S

    If glyph G maps to multiple substitutes S1, S2, ..., Sn, we do

      G                    ->  S1 + ((n - 1) << 16)
      G + (1 << 16)        ->  S2
      G + (2 << 16)        ->  S3
      ...
      G + ((n - 1) << 16)  ->  Sn
  */
  static FT_Error
  af_hash_insert( FT_UInt16  glyph,
                  FT_UInt16  substitute,
                  FT_Hash    map,
                  FT_Memory  memory )
  {
    FT_Error  error;

    size_t*  value = ft_hash_num_lookup( glyph, map );


    if ( !value )
    {
      error = ft_hash_num_insert( glyph, substitute, map, memory );
      if ( error )
        return error;
    }
    else
    {
      /* Get number of substitutes, increased by one... */
      FT_UInt  mask = ( (FT_UInt)*value & 0xFFFF0000U ) + 0x10000U;


      /* ... which becomes the new key mask. */
      error = ft_hash_num_insert( (FT_Int)( glyph | mask ),
                                  substitute,
                                  map,
                                  memory );
      if ( error )
        return error;

      /* Update number of substitutes. */
      *value += 0x10000U;
    }

    return FT_Err_Ok;
  }


  static FT_Error
  af_map_single_subst1( FT_Hash    map,
                        FT_Byte*   table,
                        FT_Memory  memory )
  {
    FT_Error  error;

    FT_Byte*  coverage     = table + FT_PEEK_USHORT( table + 2 );
    FT_UInt   deltaGlyphID = FT_PEEK_USHORT( table + 4 );

    AF_CoverageIteratorRec  iterator = af_coverage_iterator_init( coverage );

    FT_UInt16  glyph;


    while ( af_coverage_iterator( &iterator, &glyph ) )
    {
      /* `deltaGlyphID` requires modulo 65536 arithmetic. */
      FT_UInt16  subst = (FT_UInt16)( ( glyph + deltaGlyphID ) % 0x10000U );


      error = af_hash_insert( glyph, subst, map, memory );
      if ( error )
        return error;
    }

    return FT_Err_Ok;
  }


  static FT_Error
  af_map_single_subst2( FT_Hash    map,
                        FT_Byte*   table,
                        FT_Memory  memory )
  {
    FT_Error  error;

    FT_Byte*  coverage = table + FT_PEEK_USHORT( table + 2 );

    AF_CoverageIteratorRec  iterator = af_coverage_iterator_init( coverage );

    FT_UInt16  glyph;
    FT_Byte*   p = table + 6;


    while ( af_coverage_iterator( &iterator, &glyph ) )
    {
      FT_UInt16  subst = FT_NEXT_USHORT( p );


      error = af_hash_insert( glyph, subst, map, memory );
      if ( error )
        return error;
    }

    return FT_Err_Ok;
  }


  static FT_Error
  af_map_alternate( FT_Hash    map,
                    FT_Byte*   table,
                    FT_Memory  memory )
  {
    FT_Error  error;

    FT_Byte*  coverage = table + FT_PEEK_USHORT( table + 2 );

    AF_CoverageIteratorRec  iterator = af_coverage_iterator_init( coverage );

    FT_UInt16  glyph;
    FT_Byte*   p = table + 6;


    while ( af_coverage_iterator( &iterator, &glyph ) )
    {
      FT_Byte*  alternate_set = table + FT_NEXT_USHORT( p );

      FT_Byte*  q          = alternate_set;
      FT_UInt   glyphCount = FT_NEXT_USHORT( q );

      FT_UInt  i;


      for ( i = 0; i < glyphCount; i++ )
      {
        FT_UInt16  subst = FT_NEXT_USHORT( q );


        error = af_hash_insert( glyph, subst, map, memory );
        if ( error )
          return error;
      }
    }

    return FT_Err_Ok;
  }


  /* Map 'SingleSubst' and 'AlternateSubst' lookup tables. */
  FT_LOCAL_DEF( FT_Error )
  af_map_lookup( AF_FaceGlobals  globals,
                 FT_Hash         map,
                 FT_UInt32       lookup_offset )
  {
    FT_Face    face   = globals->face;
    FT_Memory  memory = face->memory;

    FT_Byte*  table = globals->gsub + lookup_offset;

    FT_UInt  lookupType    = FT_PEEK_USHORT( table );
    FT_UInt  subtableCount = FT_PEEK_USHORT( table + 4 );

    FT_Byte*  p     = table + 6;
    FT_Byte*  limit = p + subtableCount * 2;


    while ( p < limit )
    {
      FT_Error  error;

      FT_UInt  real_lookupType = lookupType;

      FT_Byte*  subtable = table + FT_NEXT_USHORT( p );


      if ( lookupType == 7 )
      {
        FT_Byte*  q = subtable + 2;


        real_lookupType = FT_NEXT_USHORT( q );
        subtable       += FT_PEEK_ULONG( q );
      }

      if ( real_lookupType == 1 )
      {
        FT_UInt  format = FT_PEEK_USHORT( subtable );


        error = ( format == 1 )
                  ? af_map_single_subst1( map, subtable, memory )
                  : af_map_single_subst2( map, subtable, memory );
      }
      else
        error = af_map_alternate( map, subtable, memory );

      if ( error )
        return error;
    }

    return FT_Err_Ok;
  }


#else /* !FT_CONFIG_OPTION_USE_HARFBUZZ */

/* ANSI C doesn't like empty source files */
typedef int  afgsub_dummy_;

#endif /* !FT_CONFIG_OPTION_USE_HARFBUZZ */

/* END */
