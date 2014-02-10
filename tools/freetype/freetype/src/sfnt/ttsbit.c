/***************************************************************************/
/*                                                                         */
/*  ttsbit.c                                                               */
/*                                                                         */
/*    TrueType and OpenType embedded bitmap support (body).                */
/*                                                                         */
/*  Copyright 2005-2009, 2013 by                                           */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  Copyright 2013 by Google, Inc.                                         */
/*  Google Author(s): Behdad Esfahbod.                                     */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


#include <ft2build.h>
#include FT_INTERNAL_DEBUG_H
#include FT_INTERNAL_STREAM_H
#include FT_TRUETYPE_TAGS_H
#include FT_BITMAP_H
#include "ttsbit.h"

#include "sferrors.h"

#include "pngshim.h"


  /*************************************************************************/
  /*                                                                       */
  /* The macro FT_COMPONENT is used in trace mode.  It is an implicit      */
  /* parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log  */
  /* messages during execution.                                            */
  /*                                                                       */
#undef  FT_COMPONENT
#define FT_COMPONENT  trace_ttsbit


  FT_LOCAL_DEF( FT_Error )
  tt_face_load_eblc( TT_Face    face,
                     FT_Stream  stream )
  {
    FT_Error  error = FT_Err_Ok;
    FT_Fixed  version;
    FT_ULong  num_strikes, table_size;
    FT_Byte*  p;
    FT_Byte*  p_limit;
    FT_UInt   count;


    face->sbit_num_strikes = 0;

    /* this table is optional */
    error = face->goto_table( face, TTAG_CBLC, stream, &table_size );
    if ( error )
      error = face->goto_table( face, TTAG_EBLC, stream, &table_size );
    if ( error )
      error = face->goto_table( face, TTAG_bloc, stream, &table_size );
    if ( error )
      goto Exit;

    if ( table_size < 8 )
    {
      FT_ERROR(( "tt_face_load_sbit_strikes: table too short\n" ));
      error = FT_THROW( Invalid_File_Format );
      goto Exit;
    }

    if ( FT_FRAME_EXTRACT( table_size, face->sbit_table ) )
      goto Exit;

    face->sbit_table_size = table_size;

    p       = face->sbit_table;
    p_limit = p + table_size;

    version     = FT_NEXT_ULONG( p );
    num_strikes = FT_NEXT_ULONG( p );

    if ( version != 0x00020000UL || num_strikes >= 0x10000UL )
    {
      FT_ERROR(( "tt_face_load_sbit_strikes: invalid table version\n" ));
      error = FT_THROW( Invalid_File_Format );
      goto Fail;
    }

    /*
     *  Count the number of strikes available in the table.  We are a bit
     *  paranoid there and don't trust the data.
     */
    count = (FT_UInt)num_strikes;
    if ( 8 + 48UL * count > table_size )
      count = (FT_UInt)( ( p_limit - p ) / 48 );

    face->sbit_num_strikes = count;

    FT_TRACE3(( "sbit_num_strikes: %u\n", count ));
  Exit:
    return error;

  Fail:
    FT_FRAME_RELEASE( face->sbit_table );
    face->sbit_table_size = 0;
    goto Exit;
  }


  FT_LOCAL_DEF( void )
  tt_face_free_eblc( TT_Face  face )
  {
    FT_Stream  stream = face->root.stream;


    FT_FRAME_RELEASE( face->sbit_table );
    face->sbit_table_size  = 0;
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
    FT_Byte*  strike;


    if ( strike_index >= (FT_ULong)face->sbit_num_strikes )
      return FT_THROW( Invalid_Argument );

    strike = face->sbit_table + 8 + strike_index * 48;

    metrics->x_ppem = (FT_UShort)strike[44];
    metrics->y_ppem = (FT_UShort)strike[45];

    metrics->ascender  = (FT_Char)strike[16] << 6;  /* hori.ascender  */
    metrics->descender = (FT_Char)strike[17] << 6;  /* hori.descender */
    metrics->height    = metrics->ascender - metrics->descender;

    /* XXX: Is this correct? */
    metrics->max_advance = ( (FT_Char)strike[22] + /* min_origin_SB  */
                                      strike[18] + /* max_width      */
                             (FT_Char)strike[23]   /* min_advance_SB */
                                                 ) << 6;

    return FT_Err_Ok;
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
    FT_Error   error;
    FT_Stream  stream = face->root.stream;
    FT_ULong   ebdt_size;


    error = face->goto_table( face, TTAG_CBDT, stream, &ebdt_size );
    if ( error )
      error = face->goto_table( face, TTAG_EBDT, stream, &ebdt_size );
    if ( error )
      error = face->goto_table( face, TTAG_bdat, stream, &ebdt_size );
    if ( error )
      goto Exit;

    decoder->face    = face;
    decoder->stream  = stream;
    decoder->bitmap  = &face->root.glyph->bitmap;
    decoder->metrics = metrics;

    decoder->metrics_loaded   = 0;
    decoder->bitmap_allocated = 0;

    decoder->ebdt_start = FT_STREAM_POS();
    decoder->ebdt_size  = ebdt_size;

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

      if ( decoder->strike_index_array > face->sbit_table_size             ||
           decoder->strike_index_array + 8 * decoder->strike_index_count >
             face->sbit_table_size                                         )
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
                                FT_UInt         load_flags )
  {
    FT_Error    error = FT_Err_Ok;
    FT_UInt     width, height;
    FT_Bitmap*  map = decoder->bitmap;
    FT_Long     size;


    if ( !decoder->metrics_loaded )
    {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    width  = decoder->metrics->width;
    height = decoder->metrics->height;

    map->width = (int)width;
    map->rows  = (int)height;

    switch ( decoder->bit_depth )
    {
    case 1:
      map->pixel_mode = FT_PIXEL_MODE_MONO;
      map->pitch      = ( map->width + 7 ) >> 3;
      map->num_grays  = 2;
      break;

    case 2:
      map->pixel_mode = FT_PIXEL_MODE_GRAY2;
      map->pitch      = ( map->width + 3 ) >> 2;
      map->num_grays  = 4;
      break;

    case 4:
      map->pixel_mode = FT_PIXEL_MODE_GRAY4;
      map->pitch      = ( map->width + 1 ) >> 1;
      map->num_grays  = 16;
      break;

    case 8:
      map->pixel_mode = FT_PIXEL_MODE_GRAY;
      map->pitch      = map->width;
      map->num_grays  = 256;
      break;

    case 32:
      if ( load_flags & FT_LOAD_COLOR )
      {
        map->pixel_mode = FT_PIXEL_MODE_BGRA;
        map->pitch      = map->width * 4;
        map->num_grays  = 256;
      }
      else
      {
        map->pixel_mode = FT_PIXEL_MODE_GRAY;
        map->pitch      = map->width;
        map->num_grays  = 256;
      }
      break;

    default:
      error = FT_THROW( Invalid_File_Format );
      goto Exit;
    }

    size = map->rows * map->pitch;

    /* check that there is no empty image */
    if ( size == 0 )
      goto Exit;     /* exit successfully! */

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

    decoder->metrics_loaded = 1;
    *pp = p;
    return FT_Err_Ok;

  Fail:
    FT_TRACE1(( "tt_sbit_decoder_load_metrics: broken table" ));
    return FT_THROW( Invalid_Argument );
  }


  /* forward declaration */
  static FT_Error
  tt_sbit_decoder_load_image( TT_SBitDecoder  decoder,
                              FT_UInt         load_flags,
                              FT_UInt         glyph_index,
                              FT_Int          x_pos,
                              FT_Int          y_pos );

  typedef FT_Error  (*TT_SBitDecoder_LoadFunc)( TT_SBitDecoder  decoder,
                                                FT_UInt         load_flags,
                                                FT_Byte*        p,
                                                FT_Byte*        plimit,
                                                FT_Int          x_pos,
                                                FT_Int          y_pos );


  static FT_Error
  tt_sbit_decoder_load_byte_aligned( TT_SBitDecoder  decoder,
                                     FT_UInt         load_flags,
                                     FT_Byte*        p,
                                     FT_Byte*        limit,
                                     FT_Int          x_pos,
                                     FT_Int          y_pos )
  {
    FT_Error    error = FT_Err_Ok;
    FT_Byte*    line;
    FT_Int      bit_height, bit_width, pitch, width, height, line_bits, h;
    FT_Bitmap*  bitmap;

    FT_UNUSED( load_flags );


    /* check that we can write the glyph into the bitmap */
    bitmap     = decoder->bitmap;
    bit_width  = bitmap->width;
    bit_height = bitmap->rows;
    pitch      = bitmap->pitch;
    line       = bitmap->buffer;

    width  = decoder->metrics->width;
    height = decoder->metrics->height;

    line_bits = width * decoder->bit_depth;

    if ( x_pos < 0 || x_pos + width > bit_width   ||
         y_pos < 0 || y_pos + height > bit_height )
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
                                    FT_UInt         load_flags,
                                    FT_Byte*        p,
                                    FT_Byte*        limit,
                                    FT_Int          x_pos,
                                    FT_Int          y_pos )
  {
    FT_Error    error = FT_Err_Ok;
    FT_Byte*    line;
    FT_Int      bit_height, bit_width, pitch, width, height, line_bits, h, nbits;
    FT_Bitmap*  bitmap;
    FT_UShort   rval;

    FT_UNUSED( load_flags );


    /* check that we can write the glyph into the bitmap */
    bitmap     = decoder->bitmap;
    bit_width  = bitmap->width;
    bit_height = bitmap->rows;
    pitch      = bitmap->pitch;
    line       = bitmap->buffer;

    width  = decoder->metrics->width;
    height = decoder->metrics->height;

    line_bits = width * decoder->bit_depth;

    if ( x_pos < 0 || x_pos + width  > bit_width  ||
         y_pos < 0 || y_pos + height > bit_height )
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
                     ( ~( 0xFF << w ) << ( 8 - w - x_pos ) );
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
                                 FT_UInt         load_flags,
                                 FT_Byte*        p,
                                 FT_Byte*        limit,
                                 FT_Int          x_pos,
                                 FT_Int          y_pos )
  {
    FT_Error  error = FT_Err_Ok;
    FT_UInt   num_components, nn;

    FT_Char  horiBearingX = decoder->metrics->horiBearingX;
    FT_Char  horiBearingY = decoder->metrics->horiBearingY;
    FT_Byte  horiAdvance  = decoder->metrics->horiAdvance;
    FT_Char  vertBearingX = decoder->metrics->vertBearingX;
    FT_Char  vertBearingY = decoder->metrics->vertBearingY;
    FT_Byte  vertAdvance  = decoder->metrics->vertAdvance;


    if ( p + 2 > limit )
      goto Fail;

    num_components = FT_NEXT_USHORT( p );
    if ( p + 4 * num_components > limit )
    {
      FT_TRACE1(( "tt_sbit_decoder_load_compound: broken table\n" ));
      goto Fail;
    }

    FT_TRACE3(( "tt_sbit_decoder_load_compound: loading %d components\n",
                num_components ));

    for ( nn = 0; nn < num_components; nn++ )
    {
      FT_UInt  gindex = FT_NEXT_USHORT( p );
      FT_Byte  dx     = FT_NEXT_BYTE( p );
      FT_Byte  dy     = FT_NEXT_BYTE( p );


      /* NB: a recursive call */
      error = tt_sbit_decoder_load_image( decoder, load_flags, gindex,
                                          x_pos + dx, y_pos + dy );
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
                            FT_UInt         load_flags,
                            FT_Byte*        p,
                            FT_Byte*        limit,
                            FT_Int          x_pos,
                            FT_Int          y_pos )
  {
    FT_Error  error = FT_Err_Ok;
    FT_ULong  png_len;

    FT_UNUSED( load_flags );


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

    error = Load_SBit_Png( decoder->bitmap,
                           x_pos,
                           y_pos,
                           decoder->bit_depth,
                           decoder->metrics,
                           decoder->stream->memory,
                           p,
                           png_len );

  Exit:
    if ( !error )
      FT_TRACE3(( "tt_sbit_decoder_load_png: loaded\n" ));
    return error;
  }

#endif /* FT_CONFIG_OPTION_USE_PNG */


  static FT_Error
  tt_sbit_decoder_load_bitmap( TT_SBitDecoder  decoder,
                               FT_UInt         load_flags,
                               FT_UInt         glyph_format,
                               FT_ULong        glyph_start,
                               FT_ULong        glyph_size,
                               FT_Int          x_pos,
                               FT_Int          y_pos )
  {
    FT_Error   error;
    FT_Stream  stream = decoder->stream;
    FT_Byte*   p;
    FT_Byte*   p_limit;
    FT_Byte*   data;


    /* seek into the EBDT table now */
    if ( glyph_start + glyph_size > decoder->ebdt_size )
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
      case 5:
      case 7:
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

#ifdef FT_CONFIG_OPTION_USE_PNG
      case 17: /* small metrics, PNG image data   */
      case 18: /* big metrics, PNG image data     */
      case 19: /* metrics in EBLC, PNG image data */
        loader = tt_sbit_decoder_load_png;
        break;
#endif /* FT_CONFIG_OPTION_USE_PNG */

      default:
        error = FT_THROW( Invalid_Table );
        goto Fail;
      }

      if ( !decoder->bitmap_allocated )
      {
        error = tt_sbit_decoder_alloc_bitmap( decoder, load_flags );
        if ( error )
          goto Fail;
      }

      if ( decoder->bit_depth == 32                          &&
           decoder->bitmap->pixel_mode != FT_PIXEL_MODE_BGRA )
      {
        /* Flatten color bitmaps if color was not requested. */

        FT_Library library = decoder->face->root.glyph->library;
        FT_Memory  memory  = decoder->stream->memory;

        FT_Bitmap color, *orig;


        if ( decoder->bitmap->pixel_mode != FT_PIXEL_MODE_GRAY ||
             x_pos != 0 || y_pos != 0                          )
        {
          /* Shouldn't happen. */
          error = FT_THROW( Invalid_Table );
          goto Fail;
        }

        FT_Bitmap_New( &color );

        color.rows       = decoder->bitmap->rows;
        color.width      = decoder->bitmap->width;
        color.pitch      = color.width * 4;
        color.pixel_mode = FT_PIXEL_MODE_BGRA;

        if ( FT_ALLOC( color.buffer, color.rows * color.pitch ) )
          goto Fail;

        orig            = decoder->bitmap;
        decoder->bitmap = &color;

        error = loader( decoder, load_flags, p, p_limit, x_pos, y_pos );

        decoder->bitmap = orig;

        /* explicitly test against FT_Err_Ok to avoid compiler warnings */
        /* (we do an assignment within a conditional)                   */
        if ( error                                           ||
             ( error = FT_Bitmap_Convert( library,
                                          &color,
                                          decoder->bitmap,
                                          1 ) ) != FT_Err_Ok )
        {
          FT_Bitmap_Done( library, &color );
          goto Fail;
        }

        FT_Bitmap_Done( library, &color );
      }

      else
        error = loader( decoder, load_flags, p, p_limit, x_pos, y_pos );
    }

  Fail:
    FT_FRAME_RELEASE( data );

  Exit:
    return error;
  }


  static FT_Error
  tt_sbit_decoder_load_image( TT_SBitDecoder  decoder,
                              FT_UInt         load_flags,
                              FT_UInt         glyph_index,
                              FT_Int          x_pos,
                              FT_Int          y_pos )
  {
    /*
     *  First, we find the correct strike range that applies to this
     *  glyph index.
     */

    FT_Byte*  p          = decoder->eblc_base + decoder->strike_index_array;
    FT_Byte*  p_limit    = decoder->eblc_limit;
    FT_ULong  num_ranges = decoder->strike_index_count;
    FT_UInt   start, end, index_format, image_format;
    FT_ULong  image_start = 0, image_end = 0, image_offset;


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
      {
        p += 4 * ( glyph_index - start );
        if ( p + 8 > p_limit )
          goto NoBitmap;

        image_start = FT_NEXT_ULONG( p );
        image_end   = FT_NEXT_ULONG( p );

        if ( image_start == image_end )  /* missing glyph */
          goto NoBitmap;
      }
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
      {
        p += 2 * ( glyph_index - start );
        if ( p + 4 > p_limit )
          goto NoBitmap;

        image_start = FT_NEXT_USHORT( p );
        image_end   = FT_NEXT_USHORT( p );

        if ( image_start == image_end )  /* missing glyph */
          goto NoBitmap;
      }
      break;

    case 4: /* sparse glyph array with (glyph,offset) pairs */
      {
        FT_ULong  mm, num_glyphs;


        if ( p + 4 > p_limit )
          goto NoBitmap;

        num_glyphs = FT_NEXT_ULONG( p );

        /* overflow check for p + ( num_glyphs + 1 ) * 4 */
        if ( num_glyphs > (FT_ULong)( ( ( p_limit - p ) >> 2 ) - 1 ) )
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
                                        load_flags,
                                        image_format,
                                        image_start,
                                        image_end,
                                        x_pos,
                                        y_pos );

  Failure:
    return FT_THROW( Invalid_Table );

  NoBitmap:
    FT_TRACE4(( "tt_sbit_decoder_load_image:"
                " no sbit found for glyph index %d\n", glyph_index ));

    return FT_THROW( Invalid_Argument );
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
    TT_SBitDecoderRec  decoder[1];
    FT_Error           error;

    FT_UNUSED( load_flags );
    FT_UNUSED( stream );
    FT_UNUSED( map );


    error = tt_sbit_decoder_init( decoder, face, strike_index, metrics );
    if ( !error )
    {
      error = tt_sbit_decoder_load_image( decoder,
                                          load_flags,
                                          glyph_index,
                                          0,
                                          0 );
      tt_sbit_decoder_done( decoder );
    }

    return error;
  }


/* EOF */
