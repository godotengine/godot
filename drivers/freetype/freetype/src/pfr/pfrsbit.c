/***************************************************************************/
/*                                                                         */
/*  pfrsbit.c                                                              */
/*                                                                         */
/*    FreeType PFR bitmap loader (body).                                   */
/*                                                                         */
/*  Copyright 2002, 2003, 2006, 2009, 2010, 2013 by                        */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


#include "pfrsbit.h"
#include "pfrload.h"
#include FT_INTERNAL_DEBUG_H
#include FT_INTERNAL_STREAM_H

#include "pfrerror.h"

#undef  FT_COMPONENT
#define FT_COMPONENT  trace_pfr


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                      PFR BIT WRITER                           *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  typedef struct  PFR_BitWriter_
  {
    FT_Byte*  line;      /* current line start                    */
    FT_Int    pitch;     /* line size in bytes                    */
    FT_Int    width;     /* width in pixels/bits                  */
    FT_Int    rows;      /* number of remaining rows to scan      */
    FT_Int    total;     /* total number of bits to draw          */

  } PFR_BitWriterRec, *PFR_BitWriter;


  static void
  pfr_bitwriter_init( PFR_BitWriter  writer,
                      FT_Bitmap*     target,
                      FT_Bool        decreasing )
  {
    writer->line   = target->buffer;
    writer->pitch  = target->pitch;
    writer->width  = target->width;
    writer->rows   = target->rows;
    writer->total  = writer->width * writer->rows;

    if ( !decreasing )
    {
      writer->line += writer->pitch * ( target->rows-1 );
      writer->pitch = -writer->pitch;
    }
  }


  static void
  pfr_bitwriter_decode_bytes( PFR_BitWriter  writer,
                              FT_Byte*       p,
                              FT_Byte*       limit )
  {
    FT_Int    n, reload;
    FT_Int    left = writer->width;
    FT_Byte*  cur  = writer->line;
    FT_UInt   mask = 0x80;
    FT_UInt   val  = 0;
    FT_UInt   c    = 0;


    n = (FT_Int)( limit - p ) * 8;
    if ( n > writer->total )
      n = writer->total;

    reload = n & 7;

    for ( ; n > 0; n-- )
    {
      if ( ( n & 7 ) == reload )
        val = *p++;

      if ( val & 0x80 )
        c |= mask;

      val  <<= 1;
      mask >>= 1;

      if ( --left <= 0 )
      {
        cur[0] = (FT_Byte)c;
        left   = writer->width;
        mask   = 0x80;

        writer->line += writer->pitch;
        cur           = writer->line;
        c             = 0;
      }
      else if ( mask == 0 )
      {
        cur[0] = (FT_Byte)c;
        mask   = 0x80;
        c      = 0;
        cur ++;
      }
    }

    if ( mask != 0x80 )
      cur[0] = (FT_Byte)c;
  }


  static void
  pfr_bitwriter_decode_rle1( PFR_BitWriter  writer,
                             FT_Byte*       p,
                             FT_Byte*       limit )
  {
    FT_Int    n, phase, count, counts[2], reload;
    FT_Int    left = writer->width;
    FT_Byte*  cur  = writer->line;
    FT_UInt   mask = 0x80;
    FT_UInt   c    = 0;


    n = writer->total;

    phase     = 1;
    counts[0] = 0;
    counts[1] = 0;
    count     = 0;
    reload    = 1;

    for ( ; n > 0; n-- )
    {
      if ( reload )
      {
        do
        {
          if ( phase )
          {
            FT_Int  v;


            if ( p >= limit )
              break;

            v         = *p++;
            counts[0] = v >> 4;
            counts[1] = v & 15;
            phase     = 0;
            count     = counts[0];
          }
          else
          {
            phase = 1;
            count = counts[1];
          }

        } while ( count == 0 );
      }

      if ( phase )
        c |= mask;

      mask >>= 1;

      if ( --left <= 0 )
      {
        cur[0] = (FT_Byte) c;
        left   = writer->width;
        mask   = 0x80;

        writer->line += writer->pitch;
        cur           = writer->line;
        c             = 0;
      }
      else if ( mask == 0 )
      {
        cur[0] = (FT_Byte)c;
        mask   = 0x80;
        c      = 0;
        cur ++;
      }

      reload = ( --count <= 0 );
    }

    if ( mask != 0x80 )
      cur[0] = (FT_Byte) c;
  }


  static void
  pfr_bitwriter_decode_rle2( PFR_BitWriter  writer,
                             FT_Byte*       p,
                             FT_Byte*       limit )
  {
    FT_Int    n, phase, count, reload;
    FT_Int    left = writer->width;
    FT_Byte*  cur  = writer->line;
    FT_UInt   mask = 0x80;
    FT_UInt   c    = 0;


    n = writer->total;

    phase  = 1;
    count  = 0;
    reload = 1;

    for ( ; n > 0; n-- )
    {
      if ( reload )
      {
        do
        {
          if ( p >= limit )
            break;

          count = *p++;
          phase = phase ^ 1;

        } while ( count == 0 );
      }

      if ( phase )
        c |= mask;

      mask >>= 1;

      if ( --left <= 0 )
      {
        cur[0] = (FT_Byte) c;
        c      = 0;
        mask   = 0x80;
        left   = writer->width;

        writer->line += writer->pitch;
        cur           = writer->line;
      }
      else if ( mask == 0 )
      {
        cur[0] = (FT_Byte)c;
        c      = 0;
        mask   = 0x80;
        cur ++;
      }

      reload = ( --count <= 0 );
    }

    if ( mask != 0x80 )
      cur[0] = (FT_Byte) c;
  }


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                  BITMAP DATA DECODING                         *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  static void
  pfr_lookup_bitmap_data( FT_Byte*   base,
                          FT_Byte*   limit,
                          FT_UInt    count,
                          FT_UInt    flags,
                          FT_UInt    char_code,
                          FT_ULong*  found_offset,
                          FT_ULong*  found_size )
  {
    FT_UInt   left, right, char_len;
    FT_Bool   two = FT_BOOL( flags & 1 );
    FT_Byte*  buff;


    char_len = 4;
    if ( two )       char_len += 1;
    if ( flags & 2 ) char_len += 1;
    if ( flags & 4 ) char_len += 1;

    left  = 0;
    right = count;

    while ( left < right )
    {
      FT_UInt  middle, code;


      middle = ( left + right ) >> 1;
      buff   = base + middle * char_len;

      /* check that we are not outside of the table -- */
      /* this is possible with broken fonts...         */
      if ( buff + char_len > limit )
        goto Fail;

      if ( two )
        code = PFR_NEXT_USHORT( buff );
      else
        code = PFR_NEXT_BYTE( buff );

      if ( code == char_code )
        goto Found_It;

      if ( code < char_code )
        left = middle;
      else
        right = middle;
    }

  Fail:
    /* Not found */
    *found_size   = 0;
    *found_offset = 0;
    return;

  Found_It:
    if ( flags & 2 )
      *found_size = PFR_NEXT_USHORT( buff );
    else
      *found_size = PFR_NEXT_BYTE( buff );

    if ( flags & 4 )
      *found_offset = PFR_NEXT_ULONG( buff );
    else
      *found_offset = PFR_NEXT_USHORT( buff );
  }


  /* load bitmap metrics.  "*padvance" must be set to the default value */
  /* before calling this function...                                    */
  /*                                                                    */
  static FT_Error
  pfr_load_bitmap_metrics( FT_Byte**  pdata,
                           FT_Byte*   limit,
                           FT_Long    scaled_advance,
                           FT_Long   *axpos,
                           FT_Long   *aypos,
                           FT_UInt   *axsize,
                           FT_UInt   *aysize,
                           FT_Long   *aadvance,
                           FT_UInt   *aformat )
  {
    FT_Error  error = FT_Err_Ok;
    FT_Byte   flags;
    FT_Char   b;
    FT_Byte*  p = *pdata;
    FT_Long   xpos, ypos, advance;
    FT_UInt   xsize, ysize;


    PFR_CHECK( 1 );
    flags = PFR_NEXT_BYTE( p );

    xpos    = 0;
    ypos    = 0;
    xsize   = 0;
    ysize   = 0;
    advance = 0;

    switch ( flags & 3 )
    {
    case 0:
      PFR_CHECK( 1 );
      b    = PFR_NEXT_INT8( p );
      xpos = b >> 4;
      ypos = ( (FT_Char)( b << 4 ) ) >> 4;
      break;

    case 1:
      PFR_CHECK( 2 );
      xpos = PFR_NEXT_INT8( p );
      ypos = PFR_NEXT_INT8( p );
      break;

    case 2:
      PFR_CHECK( 4 );
      xpos = PFR_NEXT_SHORT( p );
      ypos = PFR_NEXT_SHORT( p );
      break;

    case 3:
      PFR_CHECK( 6 );
      xpos = PFR_NEXT_LONG( p );
      ypos = PFR_NEXT_LONG( p );
      break;

    default:
      ;
    }

    flags >>= 2;
    switch ( flags & 3 )
    {
    case 0:
      /* blank image */
      xsize = 0;
      ysize = 0;
      break;

    case 1:
      PFR_CHECK( 1 );
      b     = PFR_NEXT_BYTE( p );
      xsize = ( b >> 4 ) & 0xF;
      ysize = b & 0xF;
      break;

    case 2:
      PFR_CHECK( 2 );
      xsize = PFR_NEXT_BYTE( p );
      ysize = PFR_NEXT_BYTE( p );
      break;

    case 3:
      PFR_CHECK( 4 );
      xsize = PFR_NEXT_USHORT( p );
      ysize = PFR_NEXT_USHORT( p );
      break;

    default:
      ;
    }

    flags >>= 2;
    switch ( flags & 3 )
    {
    case 0:
      advance = scaled_advance;
      break;

    case 1:
      PFR_CHECK( 1 );
      advance = PFR_NEXT_INT8( p ) << 8;
      break;

    case 2:
      PFR_CHECK( 2 );
      advance = PFR_NEXT_SHORT( p );
      break;

    case 3:
      PFR_CHECK( 3 );
      advance = PFR_NEXT_LONG( p );
      break;

    default:
      ;
    }

    *axpos    = xpos;
    *aypos    = ypos;
    *axsize   = xsize;
    *aysize   = ysize;
    *aadvance = advance;
    *aformat  = flags >> 2;
    *pdata    = p;

  Exit:
    return error;

  Too_Short:
    error = FT_THROW( Invalid_Table );
    FT_ERROR(( "pfr_load_bitmap_metrics: invalid glyph data\n" ));
    goto Exit;
  }


  static FT_Error
  pfr_load_bitmap_bits( FT_Byte*    p,
                        FT_Byte*    limit,
                        FT_UInt     format,
                        FT_Bool     decreasing,
                        FT_Bitmap*  target )
  {
    FT_Error          error = FT_Err_Ok;
    PFR_BitWriterRec  writer;


    if ( target->rows > 0 && target->width > 0 )
    {
      pfr_bitwriter_init( &writer, target, decreasing );

      switch ( format )
      {
      case 0: /* packed bits */
        pfr_bitwriter_decode_bytes( &writer, p, limit );
        break;

      case 1: /* RLE1 */
        pfr_bitwriter_decode_rle1( &writer, p, limit );
        break;

      case 2: /* RLE2 */
        pfr_bitwriter_decode_rle2( &writer, p, limit );
        break;

      default:
        FT_ERROR(( "pfr_read_bitmap_data: invalid image type\n" ));
        error = FT_THROW( Invalid_File_Format );
      }
    }

    return error;
  }


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                     BITMAP LOADING                            *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  FT_LOCAL( FT_Error )
  pfr_slot_load_bitmap( PFR_Slot  glyph,
                        PFR_Size  size,
                        FT_UInt   glyph_index )
  {
    FT_Error     error;
    PFR_Face     face   = (PFR_Face) glyph->root.face;
    FT_Stream    stream = face->root.stream;
    PFR_PhyFont  phys   = &face->phy_font;
    FT_ULong     gps_offset;
    FT_ULong     gps_size;
    PFR_Char     character;
    PFR_Strike   strike;


    character = &phys->chars[glyph_index];

    /* Look-up a bitmap strike corresponding to the current */
    /* character dimensions                                 */
    {
      FT_UInt  n;


      strike = phys->strikes;
      for ( n = 0; n < phys->num_strikes; n++ )
      {
        if ( strike->x_ppm == (FT_UInt)size->root.metrics.x_ppem &&
             strike->y_ppm == (FT_UInt)size->root.metrics.y_ppem )
        {
          goto Found_Strike;
        }

        strike++;
      }

      /* couldn't find it */
      return FT_THROW( Invalid_Argument );
    }

  Found_Strike:

    /* Now lookup the glyph's position within the file */
    {
      FT_UInt  char_len;


      char_len = 4;
      if ( strike->flags & 1 ) char_len += 1;
      if ( strike->flags & 2 ) char_len += 1;
      if ( strike->flags & 4 ) char_len += 1;

      /* Access data directly in the frame to speed lookups */
      if ( FT_STREAM_SEEK( phys->bct_offset + strike->bct_offset ) ||
           FT_FRAME_ENTER( char_len * strike->num_bitmaps )        )
        goto Exit;

      pfr_lookup_bitmap_data( stream->cursor,
                              stream->limit,
                              strike->num_bitmaps,
                              strike->flags,
                              character->char_code,
                              &gps_offset,
                              &gps_size );

      FT_FRAME_EXIT();

      if ( gps_size == 0 )
      {
        /* Could not find a bitmap program string for this glyph */
        error = FT_THROW( Invalid_Argument );
        goto Exit;
      }
    }

    /* get the bitmap metrics */
    {
      FT_Long   xpos = 0, ypos = 0, advance = 0;
      FT_UInt   xsize = 0, ysize = 0, format = 0;
      FT_Byte*  p;


      /* compute linear advance */
      advance = character->advance;
      if ( phys->metrics_resolution != phys->outline_resolution )
        advance = FT_MulDiv( advance,
                             phys->outline_resolution,
                             phys->metrics_resolution );

      glyph->root.linearHoriAdvance = advance;

      /* compute default advance, i.e., scaled advance.  This can be */
      /* overridden in the bitmap header of certain glyphs.          */
      advance = FT_MulDiv( (FT_Fixed)size->root.metrics.x_ppem << 8,
                           character->advance,
                           phys->metrics_resolution );

      if ( FT_STREAM_SEEK( face->header.gps_section_offset + gps_offset ) ||
           FT_FRAME_ENTER( gps_size )                                     )
        goto Exit;

      p     = stream->cursor;
      error = pfr_load_bitmap_metrics( &p, stream->limit,
                                       advance,
                                       &xpos, &ypos,
                                       &xsize, &ysize,
                                       &advance, &format );

      /*
       * XXX: on 16bit system, we return an error for huge bitmap
       *      which causes a size truncation, because truncated
       *      size properties makes bitmap glyph broken.
       */
      if ( xpos > FT_INT_MAX || ( ypos + ysize ) > FT_INT_MAX )
      {
        FT_TRACE1(( "pfr_slot_load_bitmap:" ));
        FT_TRACE1(( "huge bitmap glyph %dx%d over FT_GlyphSlot\n",
                     xpos, ypos ));
        error = FT_THROW( Invalid_Pixel_Size );
      }

      if ( !error )
      {
        glyph->root.format = FT_GLYPH_FORMAT_BITMAP;

        /* Set up glyph bitmap and metrics */

        /* XXX: needs casts to fit FT_Bitmap.{width|rows|pitch} */
        glyph->root.bitmap.width      = (FT_Int)xsize;
        glyph->root.bitmap.rows       = (FT_Int)ysize;
        glyph->root.bitmap.pitch      = (FT_Int)( xsize + 7 ) >> 3;
        glyph->root.bitmap.pixel_mode = FT_PIXEL_MODE_MONO;

        /* XXX: needs casts to fit FT_Glyph_Metrics.{width|height} */
        glyph->root.metrics.width        = (FT_Pos)xsize << 6;
        glyph->root.metrics.height       = (FT_Pos)ysize << 6;
        glyph->root.metrics.horiBearingX = xpos << 6;
        glyph->root.metrics.horiBearingY = ypos << 6;
        glyph->root.metrics.horiAdvance  = FT_PIX_ROUND( ( advance >> 2 ) );
        glyph->root.metrics.vertBearingX = - glyph->root.metrics.width >> 1;
        glyph->root.metrics.vertBearingY = 0;
        glyph->root.metrics.vertAdvance  = size->root.metrics.height;

        /* XXX: needs casts fit FT_GlyphSlotRec.bitmap_{left|top} */
        glyph->root.bitmap_left = (FT_Int)xpos;
        glyph->root.bitmap_top  = (FT_Int)(ypos + ysize);

        /* Allocate and read bitmap data */
        {
          FT_ULong  len = glyph->root.bitmap.pitch * ysize;


          error = ft_glyphslot_alloc_bitmap( &glyph->root, len );
          if ( !error )
          {
            error = pfr_load_bitmap_bits(
                      p,
                      stream->limit,
                      format,
                      FT_BOOL(face->header.color_flags & 2),
                      &glyph->root.bitmap );
          }
        }
      }

      FT_FRAME_EXIT();
    }

  Exit:
    return error;
  }

/* END */
