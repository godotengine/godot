/****************************************************************************
 *
 * ttkern.c
 *
 *   Load the basic TrueType kerning table.  This doesn't handle
 *   kerning data within the GPOS table at the moment.
 *
 * Copyright (C) 1996-2023 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#include <freetype/internal/ftdebug.h>
#include <freetype/internal/ftstream.h>
#include <freetype/tttags.h>
#include "ttkern.h"

#include "sferrors.h"


  /**************************************************************************
   *
   * The macro FT_COMPONENT is used in trace mode.  It is an implicit
   * parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log
   * messages during execution.
   */
#undef  FT_COMPONENT
#define FT_COMPONENT  ttkern


#undef  TT_KERN_INDEX
#define TT_KERN_INDEX( g1, g2 )  ( ( (FT_ULong)(g1) << 16 ) | (g2) )


  FT_LOCAL_DEF( FT_Error )
  tt_face_load_kern( TT_Face    face,
                     FT_Stream  stream )
  {
    FT_Error   error;
    FT_ULong   table_size;
    FT_Byte*   p;
    FT_Byte*   p_limit;
    FT_UInt    nn, num_tables;
    FT_UInt32  avail = 0, ordered = 0;


    /* the kern table is optional; exit silently if it is missing */
    error = face->goto_table( face, TTAG_kern, stream, &table_size );
    if ( error )
      goto Exit;

    if ( table_size < 4 )  /* the case of a malformed table */
    {
      FT_ERROR(( "tt_face_load_kern:"
                 " kerning table is too small - ignored\n" ));
      error = FT_THROW( Table_Missing );
      goto Exit;
    }

    if ( FT_FRAME_EXTRACT( table_size, face->kern_table ) )
    {
      FT_ERROR(( "tt_face_load_kern:"
                 " could not extract kerning table\n" ));
      goto Exit;
    }

    face->kern_table_size = table_size;

    p       = face->kern_table;
    p_limit = p + table_size;

    p         += 2; /* skip version */
    num_tables = FT_NEXT_USHORT( p );

    if ( num_tables > 32 ) /* we only support up to 32 sub-tables */
      num_tables = 32;

    for ( nn = 0; nn < num_tables; nn++ )
    {
      FT_UInt    num_pairs, length, coverage, format;
      FT_Byte*   p_next;
      FT_UInt32  mask = (FT_UInt32)1UL << nn;


      if ( p + 6 > p_limit )
        break;

      p_next = p;

      p       += 2; /* skip version */
      length   = FT_NEXT_USHORT( p );
      coverage = FT_NEXT_USHORT( p );

      if ( length <= 6 + 8 )
        break;

      p_next += length;

      if ( p_next > p_limit )  /* handle broken table */
        p_next = p_limit;

      format = coverage >> 8;

      /* we currently only support format 0 kerning tables */
      if ( format != 0 )
        goto NextTable;

      /* only use horizontal kerning tables */
      if ( ( coverage & 3U ) != 0x0001 ||
           p + 8 > p_next              )
        goto NextTable;

      num_pairs = FT_NEXT_USHORT( p );
      p        += 6;

      if ( ( p_next - p ) < 6 * (int)num_pairs ) /* handle broken count */
        num_pairs = (FT_UInt)( ( p_next - p ) / 6 );

      avail |= mask;

      /*
       * Now check whether the pairs in this table are ordered.
       * We then can use binary search.
       */
      if ( num_pairs > 0 )
      {
        FT_ULong  count;
        FT_ULong  old_pair;


        old_pair = FT_NEXT_ULONG( p );
        p       += 2;

        for ( count = num_pairs - 1; count > 0; count-- )
        {
          FT_UInt32  cur_pair;


          cur_pair = FT_NEXT_ULONG( p );
          if ( cur_pair < old_pair )
            break;

          p += 2;
          old_pair = cur_pair;
        }

        if ( count == 0 )
          ordered |= mask;
      }

    NextTable:
      p = p_next;
    }

    face->num_kern_tables = nn;
    face->kern_avail_bits = avail;
    face->kern_order_bits = ordered;

  Exit:
    return error;
  }


  FT_LOCAL_DEF( void )
  tt_face_done_kern( TT_Face  face )
  {
    FT_Stream  stream = face->root.stream;


    FT_FRAME_RELEASE( face->kern_table );
    face->kern_table_size = 0;
    face->num_kern_tables = 0;
    face->kern_avail_bits = 0;
    face->kern_order_bits = 0;
  }


  FT_LOCAL_DEF( FT_Int )
  tt_face_get_kerning( TT_Face  face,
                       FT_UInt  left_glyph,
                       FT_UInt  right_glyph )
  {
    FT_Int   result = 0;
    FT_UInt  count, mask;

    FT_Byte*  p;
    FT_Byte*  p_limit;


    if ( !face->kern_table )
      return result;

    p       = face->kern_table;
    p_limit = p + face->kern_table_size;

    p   += 4;
    mask = 0x0001;

    for ( count = face->num_kern_tables;
          count > 0 && p + 6 <= p_limit;
          count--, mask <<= 1 )
    {
      FT_Byte* base     = p;
      FT_Byte* next;
      FT_UInt  version  = FT_NEXT_USHORT( p );
      FT_UInt  length   = FT_NEXT_USHORT( p );
      FT_UInt  coverage = FT_NEXT_USHORT( p );
      FT_UInt  num_pairs;
      FT_Int   value    = 0;

      FT_UNUSED( version );


      next = base + length;

      if ( next > p_limit )  /* handle broken table */
        next = p_limit;

      if ( ( face->kern_avail_bits & mask ) == 0 )
        goto NextTable;

      FT_ASSERT( p + 8 <= next ); /* tested in tt_face_load_kern */

      num_pairs = FT_NEXT_USHORT( p );
      p        += 6;

      if ( ( next - p ) < 6 * (int)num_pairs )  /* handle broken count  */
        num_pairs = (FT_UInt)( ( next - p ) / 6 );

      switch ( coverage >> 8 )
      {
      case 0:
        {
          FT_ULong  key0 = TT_KERN_INDEX( left_glyph, right_glyph );


          if ( face->kern_order_bits & mask )   /* binary search */
          {
            FT_UInt   min = 0;
            FT_UInt   max = num_pairs;


            while ( min < max )
            {
              FT_UInt   mid = ( min + max ) >> 1;
              FT_Byte*  q   = p + 6 * mid;
              FT_ULong  key;


              key = FT_NEXT_ULONG( q );

              if ( key == key0 )
              {
                value = FT_PEEK_SHORT( q );
                goto Found;
              }
              if ( key < key0 )
                min = mid + 1;
              else
                max = mid;
            }
          }
          else /* linear search */
          {
            FT_UInt  count2;


            for ( count2 = num_pairs; count2 > 0; count2-- )
            {
              FT_ULong  key = FT_NEXT_ULONG( p );


              if ( key == key0 )
              {
                value = FT_PEEK_SHORT( p );
                goto Found;
              }
              p += 2;
            }
          }
        }
        break;

       /*
        * We don't support format 2 because we haven't seen a single font
        * using it in real life...
        */

      default:
        ;
      }

      goto NextTable;

    Found:
      if ( coverage & 8 ) /* override or add */
        result = value;
      else
        result += value;

    NextTable:
      p = next;
    }

    return result;
  }

#undef TT_KERN_INDEX

/* END */
