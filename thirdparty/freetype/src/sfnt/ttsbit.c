/****************************************************************************
 *
 * ttsbit.c
 *
 *   TrueType and OpenType embedded bitmap support (body).
 *
 * Copyright (C) 2005-2019 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * Copyright 2013 by Google, Inc.
 * Google Author(s): Behdad Esfahbod.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#include <ft2build.h>
#include FT_INTERNAL_DEBUG_H
#include FT_INTERNAL_STREAM_H
#include FT_TRUETYPE_TAGS_H
#include FT_BITMAP_H


#ifdef TT_CONFIG_OPTION_EMBEDDED_BITMAPS

#include "ttsbit.h"

#include "sferrors.h"

#include "ttmtx.h"
#include "pngshim.h"


  /**************************************************************************
   *
   * The macro FT_COMPONENT is used in trace mode.  It is an implicit
   * parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log
   * messages during execution.
   */
#undef  FT_COMPONENT
#define FT_COMPONENT  ttsbit


  FT_LOCAL_DEF( FT_Error )
  tt_face_load_sbit( TT_Face    face,
                     FT_Stream  stream )
  {
    FT_Error  error;
    FT_ULong  table_size;
    FT_ULong  table_start;


    face->sbit_table       = NULL;
    face->sbit_table_size  = 0;
    face->sbit_table_type  = TT_SBIT_TABLE_TYPE_NONE;
    face->sbit_num_strikes = 0;

    error = face->goto_table( face, TTAG_CBLC, stream, &table_size );
    if ( !error )
      face->sbit_table_type = TT_SBIT_TABLE_TYPE_CBLC;
    else
    {
      error = face->goto_table( face, TTAG_EBLC, stream, &table_size );
      if ( error )
        error = face->goto_table( face, TTAG_bloc, stream, &table_size );
      if ( !error )
        face->sbit_table_type = TT_SBIT_TABLE_TYPE_EBLC;
    }

    if ( error )
    {
      error = face->goto_table( face, TTAG_sbix, stream, &table_size );
      if ( !error )
        face->sbit_table_type = TT_SBIT_TABLE_TYPE_SBIX;
    }
    if ( error )
      goto Exit;

    if ( table_size < 8 )
    {
      FT_ERROR(( "tt_face_load_sbit_strikes: table too short\n" ));
      error = FT_THROW( Invalid_File_Format );
      goto Exit;
    }

    table_start = FT_STREAM_POS();

    switch ( (FT_UInt)face->sbit_table_type )
    {
    case TT_SBIT_TABLE_TYPE_EBLC:
    case TT_SBIT_TABLE_TYPE_CBLC:
      {
        FT_Byte*  p;
        FT_Fixed  version;
        FT_ULong  num_strikes;
        FT_UInt   count;


        if ( FT_FRAME_EXTRACT( table_size, face->sbit_table ) )
          goto Exit;

        face->sbit_table_size = table_size;

        p = face->sbit_table;

        version     = FT_NEXT_LONG( p );
        num_strikes = FT_NEXT_ULONG( p );

        /* there's at least one font (FZShuSong-Z01, version 3)   */
        /* that uses the wrong byte order for the `version' field */
        if ( ( (FT_ULong)version & 0xFFFF0000UL ) != 0x00020000UL &&
             ( (FT_ULong)version & 0x0000FFFFUL ) != 0x00000200UL &&
             ( (FT_ULong)version & 0xFFFF0000UL ) != 0x00030000UL &&
             ( (FT_ULong)version & 0x0000FFFFUL ) != 0x00000300UL )
        {
          error = FT_THROW( Unknown_File_Format );
          goto Exit;
        }

        if ( num_strikes >= 0x10000UL )
        {
          error = FT_THROW( Invalid_File_Format );
          goto Exit;
        }

        /*
         * Count the number of strikes available in the table.  We are a bit
         * paranoid there and don't trust the data.
         */
        count = (FT_UInt)num_strikes;
        if ( 8 + 48UL * count > table_size )
          count = (FT_UInt)( ( table_size - 8 ) / 48 );

        face->sbit_num_strikes = count;
      }
      break;

    case TT_SBIT_TABLE_TYPE_SBIX:
      {
        FT_UShort  version;
        FT_UShort  flags;
        FT_ULong   num_strikes;
        FT_UInt    count;


        if ( FT_FRAME_ENTER( 8 ) )
          goto Exit;

        version     = FT_GET_USHORT();
        flags       = FT_GET_USHORT();
        num_strikes = FT_GET_ULONG();

        FT_FRAME_EXIT();

        if ( version < 1 )
        {
          error = FT_THROW( Unknown_File_Format );
          goto Exit;
        }

        /* Bit 0 must always be `1'.                            */
        /* Bit 1 controls the overlay of bitmaps with outlines. */
        /* All other bits should be zero.                       */
        if ( !( flags == 1 || flags == 3 ) ||
             num_strikes >= 0x10000UL      )
        {
          error = FT_THROW( Invalid_File_Format );
          goto Exit;
        }

        /* we currently don't support bit 1; however, it is better to */
        /* draw at least something...                                 */
        if ( flags == 3 )
          FT_TRACE1(( "tt_face_load_sbit_strikes:"
                      " sbix overlay not supported yet\n"
                      "                          "
                      " expect bad rendering results\n" ));

        /*
         * Count the number of strikes available in the table.  We are a bit
         * paranoid there and don't trust the data.
         */
        count = (FT_UInt)num_strikes;
        if ( 8 + 4UL * count > table_size )
          count = (FT_UInt)( ( table_size - 8 ) / 4 );

        if ( FT_STREAM_SEEK( FT_STREAM_POS() - 8 ) )
          goto Exit;

        face->sbit_table_size = 8 + count * 4;
        if ( FT_FRAME_EXTRACT( face->sbit_table_size, face->sbit_table ) )
          goto Exit;

        face->sbit_num_strikes = count;
      }
      break;

    default:
      /* we ignore unknown table formats */
      error = FT_THROW( Unknown_File_Format );
      break;
    }

    if ( !error )
      FT_TRACE3(( "tt_face_load_sbit_strikes: found %u strikes\n",
                  face->sbit_num_strikes ));

    face->ebdt_start = 0;
    face->ebdt_size  = 0;

    if ( face->sbit_table_type == TT_SBIT_TABLE_TYPE_SBIX )
    {
      /* the `sbix' table is self-contained; */
      /* it has no associated data table     */
      face->ebdt_start = table_start;
      face->ebdt_size  = table_size;
    }
    else if ( face->sbit_table_type != TT_SBIT_TABLE_TYPE_NONE )
    {
      FT_ULong  ebdt_size;


      error = face->goto_table( face, TTAG_CBDT, stream, &ebdt_size );
      if ( error )
        error = face->goto_table( face, TTAG_EBDT, stream, &ebdt_size );
      if ( error )
        error = face->goto_table( face, TTAG_bdat, stream, &ebdt_size );

      if ( !error )
      {
        face->ebdt_start = FT_STREAM_POS();
        face->ebdt_size  = ebdt_size;
      }
    }

    if ( !face->ebdt_size )
    {
      FT_TRACE2(( "tt_face_load_sbit_strikes:"
                  " no embedded bitmap data table found;\n"
                  "                          "
                  " resetting number of strikes to zero\n" ));
      face->sbit_num_strikes = 0;
    }

    return FT_Err_Ok;

  Exit:
    if ( error )
    {
      if ( face->sbit_table )
        FT_FRAME_RELEASE( face->sbit_table );
      face->sbit_table_size = 0;
      face->sbit_table_type = TT_SBIT_TABLE_TYPE_NONE;
    }

    return error;
  }


  FT_LOCAL_DEF( void )
  tt_face_free_sbit( TT_Face  face )
  {
    FT_Stream  stream = face->root.stream;


    FT_FRAME_RELEASE( face->sbit_table );
    face->sbit_table_size  = 0;
    face->sbit_table_type  = TT_SBIT_TABLE_TYPE_NONE;
    face->sbit_num_strikes = 0;
  }


  FT_LOCAL_DEF( FT_Error )
  tt_face_set_sbit_strike( TT_Face          face,
                           FT_Size_Request  req,
                           FT_ULong*        astrike_index )
  {
    return FT_Match_Size( (FT_Face)face, req, 0, astrike_index );
  }


  FT_LOCAL_DEF( FT_Error )
  tt_face_load_strike_metrics( TT_Face           face,
                               FT_ULong          strike_index,
                               FT_Size_Metrics*  metrics )
  {
    /* we have to test for the existence of `sbit_strike_map'    */
    /* because the function gets also used at the very beginning */
    /* to construct `sbit_strike_map' itself                     */
    if ( face->sbit_strike_map )
    {
      if ( strike_index >= (FT_ULong)face->root.num_fixed_sizes )
        return FT_THROW( Invalid_Argument );

      /* map to real index */
      strike_index = face->sbit_strike_map[strike_index];
    }
    else
    {
      if ( strike_index >= (FT_ULong)face->sbit_num_strikes )
        return FT_THROW( Invalid_Argument );
    }

    switch ( (FT_UInt)face->sbit_table_type )
    {
    case TT_SBIT_TABLE_TYPE_EBLC:
    case TT_SBIT_TABLE_TYPE_CBLC:
      {
        FT_Byte*  strike;
        FT_Char   max_before_bl;
        FT_Char   min_after_bl;


        strike = face->sbit_table + 8 + strike_index * 48;

        metrics->x_ppem = (FT_UShort)strike[44];
        metrics->y_ppem = (FT_UShort)strike[45];

        metrics->ascender  = (FT_Char)strike[16] * 64;  /* hori.ascender  */
        metrics->descender = (FT_Char)strike[17] * 64;  /* hori.descender */

        /* Due to fuzzy wording in the EBLC documentation, we find both */
        /* positive and negative values for `descender'.  Additionally, */
        /* many fonts have both `ascender' and `descender' set to zero  */
        /* (which is definitely wrong).  MS Windows simply ignores all  */
        /* those values...  For these reasons we apply some heuristics  */
        /* to get a reasonable, non-zero value for the height.          */

        max_before_bl = (FT_Char)strike[24];
        min_after_bl  = (FT_Char)strike[25];

        if ( metrics->descender > 0 )
        {
          /* compare sign of descender with `min_after_bl' */
          if ( min_after_bl < 0 )
            metrics->descender = -metrics->descender;
        }

        else if ( metrics->descender == 0 )
        {
          if ( metrics->ascender == 0 )
          {
            FT_TRACE2(( "tt_face_load_strike_metrics:"
                        " sanitizing invalid ascender and descender\n"
                        "                            "
                        " values for strike %d (%dppem, %dppem)\n",
                        strike_index,
                        metrics->x_ppem, metrics->y_ppem ));

            /* sanitize buggy ascender and descender values */
            if ( max_before_bl || min_after_bl )
            {
              metrics->ascender  = max_before_bl * 64;
              metrics->descender = min_after_bl * 64;
            }
            else
            {
              metrics->ascender  = metrics->y_ppem * 64;
              metrics->descender = 0;
            }
          }
        }

#if 0
        else
          ; /* if we have a negative descender, simply use it */
#endif

        metrics->height = metrics->ascender - metrics->descender;
        if ( metrics->height == 0 )
        {
          FT_TRACE2(( "tt_face_load_strike_metrics:"
                      " sanitizing invalid height value\n"
                      "                            "
                      " for strike (%d, %d)\n",
                      metrics->x_ppem, metrics->y_ppem ));
          metrics->height    = metrics->y_ppem * 64;
          metrics->descender = metrics->ascender - metrics->height;
        }

        /* Is this correct? */
        metrics->max_advance = ( (FT_Char)strike[22] + /* min_origin_SB  */
                                          strike[18] + /* max_width      */
                                 (FT_Char)strike[23]   /* min_advance_SB */
                                                     ) * 64;

        /* set the scale values (in 16.16 units) so advances */
        /* from the hmtx and vmtx table are scaled correctly */
        metrics->x_scale = FT_MulDiv( metrics->x_ppem,
                                      64 * 0x10000,
                                      face->header.Units_Per_EM );
        metrics->y_scale = FT_MulDiv( metrics->y_ppem,
                                      64 * 0x10000,
                                      face->header.Units_Per_EM );

        return FT_Err_Ok;
      }

    case TT_SBIT_TABLE_TYPE_SBIX:
      {
        FT_Stream       stream = face->root.stream;
        FT_UInt         offset;
        FT_UShort       upem, ppem, resolution;
        TT_HoriHeader  *hori;
        FT_Pos          ppem_; /* to reduce casts */

        FT_Error  error;
        FT_Byte*  p;


        p      = face->sbit_table + 8 + 4 * strike_index;
        offset = FT_NEXT_ULONG( p );

        if ( offset + 4 > face->ebdt_size )
          return FT_THROW( Invalid_File_Format );

        if ( FT_STREAM_SEEK( face->ebdt_start + offset ) ||
             FT_FRAME_ENTER( 4 )                         )
          return error;

        ppem       = FT_GET_USHORT();
        resolution = FT_GET_USHORT();

        FT_UNUSED( resolution ); /* What to do with this? */

        FT_FRAME_EXIT();

        upem = face->header.Units_Per_EM;
        hori = &face->horizontal;

        metrics->x_ppem = ppem;
        metrics->y_ppem = ppem;

        ppem_ = (FT_Pos)ppem;

        metrics->ascender =
          FT_MulDiv( hori->Ascender, ppem_ * 64, upem );
        metrics->descender =
          FT_MulDiv( hori->Descender, ppem_ * 64, upem );
        metrics->height =
          FT_MulDiv( hori->Ascender - hori->Descender + hori->Line_Gap,
                     ppem_ * 64, upem );
        metrics->max_advance =
          FT_MulDiv( hori->advance_Width_Max, ppem_ * 64, upem );

        /* set the scale values (in 16.16 units) so advances */
        /* from the hmtx and vmtx table are scaled correctly */
        metrics->x_scale = FT_MulDiv( metrics->x_ppem,
                                      64 * 0x10000,
                                      face->header.Units_Per_EM );
        metrics->y_scale = FT_MulDiv( metrics->y_ppem,
                                      64 * 0x10000,
                                      face->header.Units_Per_EM );

        return error;
      }

    default:
      return FT_THROW( Unknown_File_Format );
    }
  }


  typedef struct  TT_SBitDecoderRec_
  {
    TT_Face          face;
    FT_Stream        stream;
    FT_Bitmap*       bitmap;
    TT_SBit_Metrics  metrics;
    FT_Bool          metrics_loaded;
    FT_Bool          bitmap_allocated;
    FT_Byte          bit_depth;

    FT_ULong         ebdt_start;
    FT_ULong         ebdt_size;

    FT_ULong         strike_index_array;
    FT_ULong         strike_index_count;
    FT_Byte*         eblc_base;
    FT_Byte*         eblc_limit;

  } TT_SBitDecoderRec, *TT_SBitDecoder;


  static FT_Error
  tt_sbit_decoder_init( TT_SBitDecoder       decoder,
                        TT_Face              face,
                        FT_ULong             strike_index,
                        TT_SBit_MetricsRec*  metrics )
  {
    FT_Error   error  = FT_ERR( Table_Missing );
    FT_Stream  stream = face->root.stream;


    strike_index = face->sbit_strike_map[strike_index];

    if ( !face->ebdt_size )
      goto Exit;
    if ( FT_STREAM_SEEK( face->ebdt_start ) )
      goto Exit;

    decoder->face    = face;
    decoder->stream  = stream;
    decoder->bitmap  = &face->root.glyph->bitmap;
    decoder->metrics = metrics;

    decoder->metrics_loaded   = 0;
    decoder->bitmap_allocated = 0;

    decoder->ebdt_start = face->ebdt_start;
    decoder->ebdt_size  = face->ebdt_size;

    decoder->eblc_base  = face->sbit_table;
    decoder->eblc_limit = face->sbit_table + face->sbit_table_size;

    /* now find the strike corresponding to the index */
    {
      FT_Byte*  p;


      if ( 8 + 48 * strike_index + 3 * 4 + 34 + 1 > face->sbit_table_size )
      {
        error = FT_THROW( Invalid_File_Format );
        goto Exit;
      }

      p = decoder->eblc_base + 8 + 48 * strike_index;

      decoder->strike_index_array = FT_NEXT_ULONG( p );
      p                          += 4;
      decoder->strike_index_count = FT_NEXT_ULONG( p );
      p                          += 34;
      decoder->bit_depth          = *p;

      /* decoder->strike_index_array +                               */
      /*   8 * decoder->strike_index_count > face->sbit_table_size ? */
      if ( decoder->strike_index_array > face->sbit_table_size           ||
           decoder->strike_index_count >
             ( face->sbit_table_size - decoder->strike_index_array ) / 8 )
        error = FT_THROW( Invalid_File_Format );
    }

  Exit:
    return error;
  }


  static void
  tt_sbit_decoder_done( TT_SBitDecoder  decoder )
  {
    FT_UNUSED( decoder );
  }


  static FT_Error
  tt_sbit_decoder_alloc_bitmap( TT_SBitDecoder  decoder,
                                FT_Bool         metrics_only )
  {
    FT_Error    error = FT_Err_Ok;
    FT_UInt     width, height;
    FT_Bitmap*  map = decoder->bitmap;
    FT_ULong    size;


    if ( !decoder->metrics_loaded )
    {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    width  = decoder->metrics->width;
    height = decoder->metrics->height;

    map->width = width;
    map->rows  = height;

    switch ( decoder->bit_depth )
    {
    case 1:
      map->pixel_mode = FT_PIXEL_MODE_MONO;
      map->pitch      = (int)( ( map->width + 7 ) >> 3 );
      map->num_grays  = 2;
      break;

    case 2:
      map->pixel_mode = FT_PIXEL_MODE_GRAY2;
      map->pitch      = (int)( ( map->width + 3 ) >> 2 );
      map->num_grays  = 4;
      break;

    case 4:
      map->pixel_mode = FT_PIXEL_MODE_GRAY4;
      map->pitch      = (int)( ( map->width + 1 ) >> 1 );
      map->num_grays  = 16;
      break;

    case 8:
      map->pixel_mode = FT_PIXEL_MODE_GRAY;
      map->pitch      = (int)( map->width );
      map->num_grays  = 256;
      break;

    case 32:
      map->pixel_mode = FT_PIXEL_MODE_BGRA;
      map->pitch      = (int)( map->width * 4 );
      map->num_grays  = 256;
      break;

    default:
      error = FT_THROW( Invalid_File_Format );
      goto Exit;
    }

    size = map->rows * (FT_ULong)map->pitch;

    /* check that there is no empty image */
    if ( size == 0 )
      goto Exit;     /* exit successfully! */

    if ( metrics_only )
      goto Exit;     /* only metrics are requested */

    error = ft_glyphslot_alloc_bitmap( decoder->face->root.glyph, size );
    if ( error )
      goto Exit;

    decoder->bitmap_allocated = 1;

  Exit:
    return error;
  }


  static FT_Error
  tt_sbit_decoder_load_metrics( TT_SBitDecoder  decoder,
                                FT_Byte*       *pp,
                                FT_Byte*        limit,
                                FT_Bool         big )
  {
    FT_Byte*         p       = *pp;
    TT_SBit_Metrics  metrics = decoder->metrics;


    if ( p + 5 > limit )
      goto Fail;

    metrics->height       = p[0];
    metrics->width        = p[1];
    metrics->horiBearingX = (FT_Char)p[2];
    metrics->horiBearingY = (FT_Char)p[3];
    metrics->horiAdvance  = p[4];

    p += 5;
    if ( big )
    {
      if ( p + 3 > limit )
        goto Fail;

      metrics->vertBearingX = (FT_Char)p[0];
      metrics->vertBearingY = (FT_Char)p[1];
      metrics->vertAdvance  = p[2];

      p += 3;
    }
    else
    {
      /* avoid uninitialized data in case there is no vertical info -- */
      metrics->vertBearingX = 0;
      metrics->vertBearingY = 0;
      metrics->vertAdvance  = 0;
    }

    decoder->metrics_loaded = 1;
    *pp = p;
    return FT_Err_Ok;

  Fail:
    FT_TRACE1(( "tt_sbit_decoder_load_metrics: broken table\n" ));
    return FT_THROW( Invalid_Argument );
  }


  /* forward declaration */
  static FT_Error
  tt_sbit_decoder_load_image( TT_SBitDecoder  decoder,
                              FT_UInt         glyph_index,
                              FT_Int          x_pos,
                              FT_Int          y_pos,
                              FT_UInt         recurse_count,
                              FT_Bool         metrics_only );

  typedef FT_Error  (*TT_SBitDecoder_LoadFunc)(
                      TT_SBitDecoder  decoder,
                      FT_Byte*        p,
                      FT_Byte*        plimit,
                      FT_Int          x_pos,
                      FT_Int          y_pos,
                      FT_UInt         recurse_count );


  static FT_Error
  tt_sbit_decoder_load_byte_aligned( TT_SBitDecoder  decoder,
                                     FT_Byte*        p,
                                     FT_Byte*        limit,
                                     FT_Int          x_pos,
                                     FT_Int          y_pos,
                                     FT_UInt         recurse_count )
  {
    FT_Error    error = FT_Err_Ok;
    FT_Byte*    line;
    FT_Int      pitch, width, height, line_bits, h;
    FT_UInt     bit_height, bit_width;
    FT_Bitmap*  bitmap;

    FT_UNUSED( recurse_count );


    /* check that we can write the glyph into the bitmap */
    bitmap     = decoder->bitmap;
    bit_width  = bitmap->width;
    bit_height = bitmap->rows;
    pitch      = bitmap->pitch;
    line       = bitmap->buffer;

    width  = decoder->metrics->width;
    height = decoder->metrics->height;

    line_bits = width * decoder->bit_depth;

    if ( x_pos < 0 || (FT_UInt)( x_pos + width ) > bit_width   ||
         y_pos < 0 || (FT_UInt)( y_pos + height ) > bit_height )
    {
      FT_TRACE1(( "tt_sbit_decoder_load_byte_aligned:"
                  " invalid bitmap dimensions\n" ));
      error = FT_THROW( Invalid_File_Format );
      goto Exit;
    }

    if ( p + ( ( line_bits + 7 ) >> 3 ) * height > limit )
    {
      FT_TRACE1(( "tt_sbit_decoder_load_byte_aligned: broken bitmap\n" ));
      error = FT_THROW( Invalid_File_Format );
      goto Exit;
    }

    /* now do the blit */
    line  += y_pos * pitch + ( x_pos >> 3 );
    x_pos &= 7;

    if ( x_pos == 0 )  /* the easy one */
    {
      for ( h = height; h > 0; h--, line += pitch )
      {
        FT_Byte*  pwrite = line;
        FT_Int    w;


        for ( w = line_bits; w >= 8; w -= 8 )
        {
          pwrite[0] = (FT_Byte)( pwrite[0] | *p++ );
          pwrite   += 1;
        }

        if ( w > 0 )
          pwrite[0] = (FT_Byte)( pwrite[0] | ( *p++ & ( 0xFF00U >> w ) ) );
      }
    }
    else  /* x_pos > 0 */
    {
      for ( h = height; h > 0; h--, line += pitch )
      {
        FT_Byte*  pwrite = line;
        FT_Int    w;
        FT_UInt   wval = 0;


        for ( w = line_bits; w >= 8; w -= 8 )
        {
          wval       = (FT_UInt)( wval | *p++ );
          pwrite[0]  = (FT_Byte)( pwrite[0] | ( wval >> x_pos ) );
          pwrite    += 1;
          wval     <<= 8;
        }

        if ( w > 0 )
          wval = (FT_UInt)( wval | ( *p++ & ( 0xFF00U >> w ) ) );

        /* all bits read and there are `x_pos + w' bits to be written */

        pwrite[0] = (FT_Byte)( pwrite[0] | ( wval >> x_pos ) );

        if ( x_pos + w > 8 )
        {
          pwrite++;
          wval     <<= 8;
          pwrite[0]  = (FT_Byte)( pwrite[0] | ( wval >> x_pos ) );
        }
      }
    }

  Exit:
    if ( !error )
      FT_TRACE3(( "tt_sbit_decoder_load_byte_aligned: loaded\n" ));
    return error;
  }


  /*
   * Load a bit-aligned bitmap (with pointer `p') into a line-aligned bitmap
   * (with pointer `pwrite').  In the example below, the width is 3 pixel,
   * and `x_pos' is 1 pixel.
   *
   *       p                               p+1
   *     |                               |                               |
   *     | 7   6   5   4   3   2   1   0 | 7   6   5   4   3   2   1   0 |...
   *     |                               |                               |
   *       +-------+   +-------+   +-------+ ...
   *           .           .           .
   *           .           .           .
   *           v           .           .
   *       +-------+       .           .
   * |                               | .
   * | 7   6   5   4   3   2   1   0 | .
   * |                               | .
   *   pwrite              .           .
   *                       .           .
   *                       v           .
   *                   +-------+       .
   *             |                               |
   *             | 7   6   5   4   3   2   1   0 |
   *             |                               |
   *               pwrite+1            .
   *                                   .
   *                                   v
   *                               +-------+
   *                         |                               |
   *                         | 7   6   5   4   3   2   1   0 |
   *                         |                               |
   *                           pwrite+2
   *
   */

  static FT_Error
  tt_sbit_decoder_load_bit_aligned( TT_SBitDecoder  decoder,
                                    FT_Byte*        p,
                                    FT_Byte*        limit,
                                    FT_Int          x_pos,
                                    FT_Int          y_pos,
                                    FT_UInt         recurse_count )
  {
    FT_Error    error = FT_Err_Ok;
    FT_Byte*    line;
    FT_Int      pitch, width, height, line_bits, h, nbits;
    FT_UInt     bit_height, bit_width;
    FT_Bitmap*  bitmap;
    FT_UShort   rval;

    FT_UNUSED( recurse_count );


    /* check that we can write the glyph into the bitmap */
    bitmap     = decoder->bitmap;
    bit_width  = bitmap->width;
    bit_height = bitmap->rows;
    pitch      = bitmap->pitch;
    line       = bitmap->buffer;

    width  = decoder->metrics->width;
    height = decoder->metrics->height;

    line_bits = width * decoder->bit_depth;

    if ( x_pos < 0 || (FT_UInt)( x_pos + width ) > bit_width   ||
         y_pos < 0 || (FT_UInt)( y_pos + height ) > bit_height )
    {
      FT_TRACE1(( "tt_sbit_decoder_load_bit_aligned:"
                  " invalid bitmap dimensions\n" ));
      error = FT_THROW( Invalid_File_Format );
      goto Exit;
    }

    if ( p + ( ( line_bits * height + 7 ) >> 3 ) > limit )
    {
      FT_TRACE1(( "tt_sbit_decoder_load_bit_aligned: broken bitmap\n" ));
      error = FT_THROW( Invalid_File_Format );
      goto Exit;
    }

    if ( !line_bits || !height )
    {
      /* nothing to do */
      goto Exit;
    }

    /* now do the blit */

    /* adjust `line' to point to the first byte of the bitmap */
    line  += y_pos * pitch + ( x_pos >> 3 );
    x_pos &= 7;

    /* the higher byte of `rval' is used as a buffer */
    rval  = 0;
    nbits = 0;

    for ( h = height; h > 0; h--, line += pitch )
    {
      FT_Byte*  pwrite = line;
      FT_Int    w      = line_bits;


      /* handle initial byte (in target bitmap) specially if necessary */
      if ( x_pos )
      {
        w = ( line_bits < 8 - x_pos ) ? line_bits : 8 - x_pos;

        if ( h == height )
        {
          rval  = *p++;
          nbits = x_pos;
        }
        else if ( nbits < w )
        {
          if ( p < limit )
            rval |= *p++;
          nbits += 8 - w;
        }
        else
        {
          rval  >>= 8;
          nbits  -= w;
        }

        *pwrite++ |= ( ( rval >> nbits ) & 0xFF ) &
                     ( ~( 0xFFU << w ) << ( 8 - w - x_pos ) );
        rval     <<= 8;

        w = line_bits - w;
      }

      /* handle medial bytes */
      for ( ; w >= 8; w -= 8 )
      {
        rval      |= *p++;
        *pwrite++ |= ( rval >> nbits ) & 0xFF;

        rval <<= 8;
      }

      /* handle final byte if necessary */
      if ( w > 0 )
      {
        if ( nbits < w )
        {
          if ( p < limit )
            rval |= *p++;
          *pwrite |= ( ( rval >> nbits ) & 0xFF ) & ( 0xFF00U >> w );
          nbits   += 8 - w;

          rval <<= 8;
        }
        else
        {
          *pwrite |= ( ( rval >> nbits ) & 0xFF ) & ( 0xFF00U >> w );
          nbits   -= w;
        }
      }
    }

  Exit:
    if ( !error )
      FT_TRACE3(( "tt_sbit_decoder_load_bit_aligned: loaded\n" ));
    return error;
  }


  static FT_Error
  tt_sbit_decoder_load_compound( TT_SBitDecoder  decoder,
                                 FT_Byte*        p,
                                 FT_Byte*        limit,
                                 FT_Int          x_pos,
                                 FT_Int          y_pos,
                                 FT_UInt         recurse_count )
  {
    FT_Error  error = FT_Err_Ok;
    FT_UInt   num_components, nn;

    FT_Char  horiBearingX = (FT_Char)decoder->metrics->horiBearingX;
    FT_Char  horiBearingY = (FT_Char)decoder->metrics->horiBearingY;
    FT_Byte  horiAdvance  = (FT_Byte)decoder->metrics->horiAdvance;
    FT_Char  vertBearingX = (FT_Char)decoder->metrics->vertBearingX;
    FT_Char  vertBearingY = (FT_Char)decoder->metrics->vertBearingY;
    FT_Byte  vertAdvance  = (FT_Byte)decoder->metrics->vertAdvance;


    if ( p + 2 > limit )
      goto Fail;

    num_components = FT_NEXT_USHORT( p );
    if ( p + 4 * num_components > limit )
    {
      FT_TRACE1(( "tt_sbit_decoder_load_compound: broken table\n" ));
      goto Fail;
    }

    FT_TRACE3(( "tt_sbit_decoder_load_compound: loading %d component%s\n",
                num_components,
                num_components == 1 ? "" : "s" ));

    for ( nn = 0; nn < num_components; nn++ )
    {
      FT_UInt  gindex = FT_NEXT_USHORT( p );
      FT_Char  dx     = FT_NEXT_CHAR( p );
      FT_Char  dy     = FT_NEXT_CHAR( p );


      /* NB: a recursive call */
      error = tt_sbit_decoder_load_image( decoder,
                                          gindex,
                                          x_pos + dx,
                                          y_pos + dy,
                                          recurse_count + 1,
                                          /* request full bitmap image */
                                          FALSE );
      if ( error )
        break;
    }

    FT_TRACE3(( "tt_sbit_decoder_load_compound: done\n" ));

    decoder->metrics->horiBearingX = horiBearingX;
    decoder->metrics->horiBearingY = horiBearingY;
    decoder->metrics->horiAdvance  = horiAdvance;
    decoder->metrics->vertBearingX = vertBearingX;
    decoder->metrics->vertBearingY = vertBearingY;
    decoder->metrics->vertAdvance  = vertAdvance;
    decoder->metrics->width        = (FT_Byte)decoder->bitmap->width;
    decoder->metrics->height       = (FT_Byte)decoder->bitmap->rows;

  Exit:
    return error;

  Fail:
    error = FT_THROW( Invalid_File_Format );
    goto Exit;
  }


#ifdef FT_CONFIG_OPTION_USE_PNG

  static FT_Error
  tt_sbit_decoder_load_png( TT_SBitDecoder  decoder,
                            FT_Byte*        p,
                            FT_Byte*        limit,
                            FT_Int          x_pos,
                            FT_Int          y_pos,
                            FT_UInt         recurse_count )
  {
    FT_Error  error = FT_Err_Ok;
    FT_ULong  png_len;

    FT_UNUSED( recurse_count );


    if ( limit - p < 4 )
    {
      FT_TRACE1(( "tt_sbit_decoder_load_png: broken bitmap\n" ));
      error = FT_THROW( Invalid_File_Format );
      goto Exit;
    }

    png_len = FT_NEXT_ULONG( p );
    if ( (FT_ULong)( limit - p ) < png_len )
    {
      FT_TRACE1(( "tt_sbit_decoder_load_png: broken bitmap\n" ));
      error = FT_THROW( Invalid_File_Format );
      goto Exit;
    }

    error = Load_SBit_Png( decoder->face->root.glyph,
                           x_pos,
                           y_pos,
                           decoder->bit_depth,
                           decoder->metrics,
                           decoder->stream->memory,
                           p,
                           png_len,
                           FALSE,
                           FALSE );

  Exit:
    if ( !error )
      FT_TRACE3(( "tt_sbit_decoder_load_png: loaded\n" ));
    return error;
  }

#endif /* FT_CONFIG_OPTION_USE_PNG */


  static FT_Error
  tt_sbit_decoder_load_bitmap( TT_SBitDecoder  decoder,
                               FT_UInt         glyph_format,
                               FT_ULong        glyph_start,
                               FT_ULong        glyph_size,
                               FT_Int          x_pos,
                               FT_Int          y_pos,
                               FT_UInt         recurse_count,
                               FT_Bool         metrics_only )
  {
    FT_Error   error;
    FT_Stream  stream = decoder->stream;
    FT_Byte*   p;
    FT_Byte*   p_limit;
    FT_Byte*   data;


    /* seek into the EBDT table now */
    if ( !glyph_size                                   ||
         glyph_start + glyph_size > decoder->ebdt_size )
    {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    if ( FT_STREAM_SEEK( decoder->ebdt_start + glyph_start ) ||
         FT_FRAME_EXTRACT( glyph_size, data )                )
      goto Exit;

    p       = data;
    p_limit = p + glyph_size;

    /* read the data, depending on the glyph format */
    switch ( glyph_format )
    {
    case 1:
    case 2:
    case 8:
    case 17:
      error = tt_sbit_decoder_load_metrics( decoder, &p, p_limit, 0 );
      break;

    case 6:
    case 7:
    case 9:
    case 18:
      error = tt_sbit_decoder_load_metrics( decoder, &p, p_limit, 1 );
      break;

    default:
      error = FT_Err_Ok;
    }

    if ( error )
      goto Fail;

    {
      TT_SBitDecoder_LoadFunc  loader;


      switch ( glyph_format )
      {
      case 1:
      case 6:
        loader = tt_sbit_decoder_load_byte_aligned;
        break;

      case 2:
      case 7:
        {
          /* Don't trust `glyph_format'.  For example, Apple's main Korean */
          /* system font, `AppleMyungJo.ttf' (version 7.0d2e6), uses glyph */
          /* format 7, but the data is format 6.  We check whether we have */
          /* an excessive number of bytes in the image: If it is equal to  */
          /* the value for a byte-aligned glyph, use the other loading     */
          /* routine.                                                      */
          /*                                                               */
          /* Note that for some (width,height) combinations, where the     */
          /* width is not a multiple of 8, the sizes for bit- and          */
          /* byte-aligned data are equal, for example (7,7) or (15,6).  We */
          /* then prefer what `glyph_format' specifies.                    */

          FT_UInt  width  = decoder->metrics->width;
          FT_UInt  height = decoder->metrics->height;

          FT_UInt  bit_size  = ( width * height + 7 ) >> 3;
          FT_UInt  byte_size = height * ( ( width + 7 ) >> 3 );


          if ( bit_size < byte_size                  &&
               byte_size == (FT_UInt)( p_limit - p ) )
            loader = tt_sbit_decoder_load_byte_aligned;
          else
            loader = tt_sbit_decoder_load_bit_aligned;
        }
        break;

      case 5:
        loader = tt_sbit_decoder_load_bit_aligned;
        break;

      case 8:
        if ( p + 1 > p_limit )
          goto Fail;

        p += 1;  /* skip padding */
        /* fall-through */

      case 9:
        loader = tt_sbit_decoder_load_compound;
        break;

      case 17: /* small metrics, PNG image data   */
      case 18: /* big metrics, PNG image data     */
      case 19: /* metrics in EBLC, PNG image data */
#ifdef FT_CONFIG_OPTION_USE_PNG
        loader = tt_sbit_decoder_load_png;
        break;
#else
        error = FT_THROW( Unimplemented_Feature );
        goto Fail;
#endif /* FT_CONFIG_OPTION_USE_PNG */

      default:
        error = FT_THROW( Invalid_Table );
        goto Fail;
      }

      if ( !decoder->bitmap_allocated )
      {
        error = tt_sbit_decoder_alloc_bitmap( decoder, metrics_only );

        if ( error )
          goto Fail;
      }

      if ( metrics_only )
        goto Fail; /* this is not an error */

      error = loader( decoder, p, p_limit, x_pos, y_pos, recurse_count );
    }

  Fail:
    FT_FRAME_RELEASE( data );

  Exit:
    return error;
  }


  static FT_Error
  tt_sbit_decoder_load_image( TT_SBitDecoder  decoder,
                              FT_UInt         glyph_index,
                              FT_Int          x_pos,
                              FT_Int          y_pos,
                              FT_UInt         recurse_count,
                              FT_Bool         metrics_only )
  {
    FT_Byte*  p          = decoder->eblc_base + decoder->strike_index_array;
    FT_Byte*  p_limit    = decoder->eblc_limit;
    FT_ULong  num_ranges = decoder->strike_index_count;
    FT_UInt   start, end, index_format, image_format;
    FT_ULong  image_start = 0, image_end = 0, image_offset;


    /* arbitrary recursion limit */
    if ( recurse_count > 100 )
    {
      FT_TRACE4(( "tt_sbit_decoder_load_image:"
                  " recursion depth exceeded\n" ));
      goto Failure;
    }


    /* First, we find the correct strike range that applies to this */
    /* glyph index.                                                 */
    for ( ; num_ranges > 0; num_ranges-- )
    {
      start = FT_NEXT_USHORT( p );
      end   = FT_NEXT_USHORT( p );

      if ( glyph_index >= start && glyph_index <= end )
        goto FoundRange;

      p += 4;  /* ignore index offset */
    }
    goto NoBitmap;

  FoundRange:
    image_offset = FT_NEXT_ULONG( p );

    /* overflow check */
    p = decoder->eblc_base + decoder->strike_index_array;
    if ( image_offset > (FT_ULong)( p_limit - p ) )
      goto Failure;

    p += image_offset;
    if ( p + 8 > p_limit )
      goto NoBitmap;

    /* now find the glyph's location and extend within the ebdt table */
    index_format = FT_NEXT_USHORT( p );
    image_format = FT_NEXT_USHORT( p );
    image_offset = FT_NEXT_ULONG ( p );

    switch ( index_format )
    {
    case 1: /* 4-byte offsets relative to `image_offset' */
      p += 4 * ( glyph_index - start );
      if ( p + 8 > p_limit )
        goto NoBitmap;

      image_start = FT_NEXT_ULONG( p );
      image_end   = FT_NEXT_ULONG( p );

      if ( image_start == image_end )  /* missing glyph */
        goto NoBitmap;
      break;

    case 2: /* big metrics, constant image size */
      {
        FT_ULong  image_size;


        if ( p + 12 > p_limit )
          goto NoBitmap;

        image_size = FT_NEXT_ULONG( p );

        if ( tt_sbit_decoder_load_metrics( decoder, &p, p_limit, 1 ) )
          goto NoBitmap;

        image_start = image_size * ( glyph_index - start );
        image_end   = image_start + image_size;
      }
      break;

    case 3: /* 2-byte offsets relative to 'image_offset' */
      p += 2 * ( glyph_index - start );
      if ( p + 4 > p_limit )
        goto NoBitmap;

      image_start = FT_NEXT_USHORT( p );
      image_end   = FT_NEXT_USHORT( p );

      if ( image_start == image_end )  /* missing glyph */
        goto NoBitmap;
      break;

    case 4: /* sparse glyph array with (glyph,offset) pairs */
      {
        FT_ULong  mm, num_glyphs;


        if ( p + 4 > p_limit )
          goto NoBitmap;

        num_glyphs = FT_NEXT_ULONG( p );

        /* overflow check for p + ( num_glyphs + 1 ) * 4 */
        if ( p + 4 > p_limit                                         ||
             num_glyphs > (FT_ULong)( ( ( p_limit - p ) >> 2 ) - 1 ) )
          goto NoBitmap;

        for ( mm = 0; mm < num_glyphs; mm++ )
        {
          FT_UInt  gindex = FT_NEXT_USHORT( p );


          if ( gindex == glyph_index )
          {
            image_start = FT_NEXT_USHORT( p );
            p          += 2;
            image_end   = FT_PEEK_USHORT( p );
            break;
          }
          p += 2;
        }

        if ( mm >= num_glyphs )
          goto NoBitmap;
      }
      break;

    case 5: /* constant metrics with sparse glyph codes */
    case 19:
      {
        FT_ULong  image_size, mm, num_glyphs;


        if ( p + 16 > p_limit )
          goto NoBitmap;

        image_size = FT_NEXT_ULONG( p );

        if ( tt_sbit_decoder_load_metrics( decoder, &p, p_limit, 1 ) )
          goto NoBitmap;

        num_glyphs = FT_NEXT_ULONG( p );

        /* overflow check for p + 2 * num_glyphs */
        if ( num_glyphs > (FT_ULong)( ( p_limit - p ) >> 1 ) )
          goto NoBitmap;

        for ( mm = 0; mm < num_glyphs; mm++ )
        {
          FT_UInt  gindex = FT_NEXT_USHORT( p );


          if ( gindex == glyph_index )
            break;
        }

        if ( mm >= num_glyphs )
          goto NoBitmap;

        image_start = image_size * mm;
        image_end   = image_start + image_size;
      }
      break;

    default:
      goto NoBitmap;
    }

    if ( image_start > image_end )
      goto NoBitmap;

    image_end  -= image_start;
    image_start = image_offset + image_start;

    FT_TRACE3(( "tt_sbit_decoder_load_image:"
                " found sbit (format %d) for glyph index %d\n",
                image_format, glyph_index ));

    return tt_sbit_decoder_load_bitmap( decoder,
                                        image_format,
                                        image_start,
                                        image_end,
                                        x_pos,
                                        y_pos,
                                        recurse_count,
                                        metrics_only );

  Failure:
    return FT_THROW( Invalid_Table );

  NoBitmap:
    if ( recurse_count )
    {
      FT_TRACE4(( "tt_sbit_decoder_load_image:"
                  " missing subglyph sbit with glyph index %d\n",
                  glyph_index ));
      return FT_THROW( Invalid_Composite );
    }

    FT_TRACE4(( "tt_sbit_decoder_load_image:"
                " no sbit found for glyph index %d\n", glyph_index ));
    return FT_THROW( Missing_Bitmap );
  }


  static FT_Error
  tt_face_load_sbix_image( TT_Face              face,
                           FT_ULong             strike_index,
                           FT_UInt              glyph_index,
                           FT_Stream            stream,
                           FT_Bitmap           *map,
                           TT_SBit_MetricsRec  *metrics,
                           FT_Bool              metrics_only )
  {
    FT_UInt   strike_offset, glyph_start, glyph_end;
    FT_Int    originOffsetX, originOffsetY;
    FT_Tag    graphicType;
    FT_Int    recurse_depth = 0;

    FT_Error  error;
    FT_Byte*  p;

    FT_UNUSED( map );
#ifndef FT_CONFIG_OPTION_USE_PNG
    FT_UNUSED( metrics_only );
#endif


    strike_index = face->sbit_strike_map[strike_index];

    metrics->width  = 0;
    metrics->height = 0;

    p = face->sbit_table + 8 + 4 * strike_index;
    strike_offset = FT_NEXT_ULONG( p );

  retry:
    if ( glyph_index > (FT_UInt)face->root.num_glyphs )
      return FT_THROW( Invalid_Argument );

    if ( strike_offset >= face->ebdt_size                          ||
         face->ebdt_size - strike_offset < 4 + glyph_index * 4 + 8 )
      return FT_THROW( Invalid_File_Format );

    if ( FT_STREAM_SEEK( face->ebdt_start  +
                         strike_offset + 4 +
                         glyph_index * 4   ) ||
         FT_FRAME_ENTER( 8 )                 )
      return error;

    glyph_start = FT_GET_ULONG();
    glyph_end   = FT_GET_ULONG();

    FT_FRAME_EXIT();

    if ( glyph_start == glyph_end )
      return FT_THROW( Missing_Bitmap );
    if ( glyph_start > glyph_end                     ||
         glyph_end - glyph_start < 8                 ||
         face->ebdt_size - strike_offset < glyph_end )
      return FT_THROW( Invalid_File_Format );

    if ( FT_STREAM_SEEK( face->ebdt_start + strike_offset + glyph_start ) ||
         FT_FRAME_ENTER( glyph_end - glyph_start )                        )
      return error;

    originOffsetX = FT_GET_SHORT();
    originOffsetY = FT_GET_SHORT();

    graphicType = FT_GET_TAG4();

    switch ( graphicType )
    {
    case FT_MAKE_TAG( 'd', 'u', 'p', 'e' ):
      if ( recurse_depth < 4 )
      {
        glyph_index = FT_GET_USHORT();
        FT_FRAME_EXIT();
        recurse_depth++;
        goto retry;
      }
      error = FT_THROW( Invalid_File_Format );
      break;

    case FT_MAKE_TAG( 'p', 'n', 'g', ' ' ):
#ifdef FT_CONFIG_OPTION_USE_PNG
      error = Load_SBit_Png( face->root.glyph,
                             0,
                             0,
                             32,
                             metrics,
                             stream->memory,
                             stream->cursor,
                             glyph_end - glyph_start - 8,
                             TRUE,
                             metrics_only );
#else
      error = FT_THROW( Unimplemented_Feature );
#endif
      break;

    case FT_MAKE_TAG( 'j', 'p', 'g', ' ' ):
    case FT_MAKE_TAG( 't', 'i', 'f', 'f' ):
    case FT_MAKE_TAG( 'r', 'g', 'b', 'l' ): /* used on iOS 7.1 */
      error = FT_THROW( Unknown_File_Format );
      break;

    default:
      error = FT_THROW( Unimplemented_Feature );
      break;
    }

    FT_FRAME_EXIT();

    if ( !error )
    {
      FT_Short   abearing;
      FT_UShort  aadvance;


      tt_face_get_metrics( face, FALSE, glyph_index, &abearing, &aadvance );

      metrics->horiBearingX = (FT_Short)originOffsetX;
      metrics->horiBearingY = (FT_Short)( -originOffsetY + metrics->height );
      metrics->horiAdvance  = (FT_UShort)( aadvance *
                                           face->root.size->metrics.x_ppem /
                                           face->header.Units_Per_EM );
    }

    return error;
  }

  FT_LOCAL( FT_Error )
  tt_face_load_sbit_image( TT_Face              face,
                           FT_ULong             strike_index,
                           FT_UInt              glyph_index,
                           FT_UInt              load_flags,
                           FT_Stream            stream,
                           FT_Bitmap           *map,
                           TT_SBit_MetricsRec  *metrics )
  {
    FT_Error  error = FT_Err_Ok;


    switch ( (FT_UInt)face->sbit_table_type )
    {
    case TT_SBIT_TABLE_TYPE_EBLC:
    case TT_SBIT_TABLE_TYPE_CBLC:
      {
        TT_SBitDecoderRec  decoder[1];


        error = tt_sbit_decoder_init( decoder, face, strike_index, metrics );
        if ( !error )
        {
          error = tt_sbit_decoder_load_image(
                    decoder,
                    glyph_index,
                    0,
                    0,
                    0,
                    ( load_flags & FT_LOAD_BITMAP_METRICS_ONLY ) != 0 );
          tt_sbit_decoder_done( decoder );
        }
      }
      break;

    case TT_SBIT_TABLE_TYPE_SBIX:
      error = tt_face_load_sbix_image(
                face,
                strike_index,
                glyph_index,
                stream,
                map,
                metrics,
                ( load_flags & FT_LOAD_BITMAP_METRICS_ONLY ) != 0 );
      break;

    default:
      error = FT_THROW( Unknown_File_Format );
      break;
    }

    /* Flatten color bitmaps if color was not requested. */
    if ( !error                                        &&
         !( load_flags & FT_LOAD_COLOR )               &&
         !( load_flags & FT_LOAD_BITMAP_METRICS_ONLY ) &&
         map->pixel_mode == FT_PIXEL_MODE_BGRA         )
    {
      FT_Bitmap   new_map;
      FT_Library  library = face->root.glyph->library;


      FT_Bitmap_Init( &new_map );

      /* Convert to 8bit grayscale. */
      error = FT_Bitmap_Convert( library, map, &new_map, 1 );
      if ( error )
        FT_Bitmap_Done( library, &new_map );
      else
      {
        map->pixel_mode = new_map.pixel_mode;
        map->pitch      = new_map.pitch;
        map->num_grays  = new_map.num_grays;

        ft_glyphslot_set_bitmap( face->root.glyph, new_map.buffer );
        face->root.glyph->internal->flags |= FT_GLYPH_OWN_BITMAP;
      }
    }

    return error;
  }

#else /* !TT_CONFIG_OPTION_EMBEDDED_BITMAPS */

  /* ANSI C doesn't like empty source files */
  typedef int  _tt_sbit_dummy;

#endif /* !TT_CONFIG_OPTION_EMBEDDED_BITMAPS */


/* END */
