/****************************************************************************
 *
 * pngshim.c
 *
 *   PNG Bitmap glyph support.
 *
 * Copyright (C) 2013-2021 by
 * Google, Inc.
 * Written by Stuart Gill and Behdad Esfahbod.
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
#include FT_CONFIG_STANDARD_LIBRARY_H


#if defined( TT_CONFIG_OPTION_EMBEDDED_BITMAPS ) && \
    defined( FT_CONFIG_OPTION_USE_PNG )

  /* We always include <setjmp.h>, so make libpng shut up! */
#define PNG_SKIP_SETJMP_CHECK 1
#include <png.h>
#include "pngshim.h"

#include "sferrors.h"


  /* This code is freely based on cairo-png.c.  There's so many ways */
  /* to call libpng, and the way cairo does it is defacto standard.  */

  static unsigned int
  multiply_alpha( unsigned int  alpha,
                  unsigned int  color )
  {
    unsigned int  temp = alpha * color + 0x80;


    return ( temp + ( temp >> 8 ) ) >> 8;
  }


  /* Premultiplies data and converts RGBA bytes => BGRA. */
  static void
  premultiply_data( png_structp    png,
                    png_row_infop  row_info,
                    png_bytep      data )
  {
    unsigned int  i = 0, limit;

    /* The `vector_size' attribute was introduced in gcc 3.1, which */
    /* predates clang; the `__BYTE_ORDER__' preprocessor symbol was */
    /* introduced in gcc 4.6 and clang 3.2, respectively.           */
    /* `__builtin_shuffle' for gcc was introduced in gcc 4.7.0.     */
    /*                                                              */
    /* Intel compilers do not currently support __builtin_shuffle;  */

    /* The Intel check must be first. */
#if !defined( __INTEL_COMPILER )                                       && \
    ( ( defined( __GNUC__ )                                &&             \
        ( ( __GNUC__ >= 5 )                              ||               \
        ( ( __GNUC__ == 4 ) && ( __GNUC_MINOR__ >= 7 ) ) ) )         ||   \
      ( defined( __clang__ )                                       &&     \
        ( ( __clang_major__ >= 4 )                               ||       \
        ( ( __clang_major__ == 3 ) && ( __clang_minor__ >= 2 ) ) ) ) ) && \
    defined( __OPTIMIZE__ )                                            && \
    defined( __SSE__ )                                                 && \
    __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__

#ifdef __clang__
    /* the clang documentation doesn't cover the two-argument case of */
    /* `__builtin_shufflevector'; however, it is is implemented since */
    /* version 2.8                                                    */
#define vector_shuffle  __builtin_shufflevector
#else
#define vector_shuffle  __builtin_shuffle
#endif

    typedef unsigned short  v82 __attribute__(( vector_size( 16 ) ));


    if ( row_info->rowbytes > 15 )
    {
      /* process blocks of 16 bytes in one rush, which gives a nice speed-up */
      limit = row_info->rowbytes - 16 + 1;
      for ( ; i < limit; i += 16 )
      {
        unsigned char*  base = &data[i];

        v82  s, s0, s1, a;

        /* clang <= 3.9 can't apply scalar values to vectors */
        /* (or rather, it needs a different syntax)          */
        v82  n0x80 = { 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80 };
        v82  n0xFF = { 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF };
        v82  n8    = { 8, 8, 8, 8, 8, 8, 8, 8 };

        v82  ma = { 1, 1, 3, 3, 5, 5, 7, 7 };
        v82  o1 = { 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF };
        v82  m0 = { 1, 0, 3, 2, 5, 4, 7, 6 };


        ft_memcpy( &s, base, 16 );            /* RGBA RGBA RGBA RGBA */
        s0 = s & n0xFF;                       /*  R B  R B  R B  R B */
        s1 = s >> n8;                         /*  G A  G A  G A  G A */

        a   = vector_shuffle( s1, ma );       /*  A A  A A  A A  A A */
        s1 |= o1;                             /*  G 1  G 1  G 1  G 1 */
        s0  = vector_shuffle( s0, m0 );       /*  B R  B R  B R  B R */

        s0 *= a;
        s1 *= a;
        s0 += n0x80;
        s1 += n0x80;
        s0  = ( s0 + ( s0 >> n8 ) ) >> n8;
        s1  = ( s1 + ( s1 >> n8 ) ) >> n8;

        s = s0 | ( s1 << n8 );
        ft_memcpy( base, &s, 16 );
      }
    }
#endif /* use `vector_size' */

    FT_UNUSED( png );

    limit = row_info->rowbytes;
    for ( ; i < limit; i += 4 )
    {
      unsigned char*  base  = &data[i];
      unsigned int    alpha = base[3];


      if ( alpha == 0 )
        base[0] = base[1] = base[2] = base[3] = 0;

      else
      {
        unsigned int  red   = base[0];
        unsigned int  green = base[1];
        unsigned int  blue  = base[2];


        if ( alpha != 0xFF )
        {
          red   = multiply_alpha( alpha, red   );
          green = multiply_alpha( alpha, green );
          blue  = multiply_alpha( alpha, blue  );
        }

        base[0] = (unsigned char)blue;
        base[1] = (unsigned char)green;
        base[2] = (unsigned char)red;
        base[3] = (unsigned char)alpha;
      }
    }
  }


  /* Converts RGBx bytes to BGRA. */
  static void
  convert_bytes_to_data( png_structp    png,
                         png_row_infop  row_info,
                         png_bytep      data )
  {
    unsigned int  i;

    FT_UNUSED( png );


    for ( i = 0; i < row_info->rowbytes; i += 4 )
    {
      unsigned char*  base  = &data[i];
      unsigned int    red   = base[0];
      unsigned int    green = base[1];
      unsigned int    blue  = base[2];


      base[0] = (unsigned char)blue;
      base[1] = (unsigned char)green;
      base[2] = (unsigned char)red;
      base[3] = 0xFF;
    }
  }


  /* Use error callback to avoid png writing to stderr. */
  static void
  error_callback( png_structp      png,
                  png_const_charp  error_msg )
  {
    FT_Error*  error = (FT_Error*)png_get_error_ptr( png );

    FT_UNUSED( error_msg );


    *error = FT_THROW( Out_Of_Memory );
#ifdef PNG_SETJMP_SUPPORTED
    ft_longjmp( png_jmpbuf( png ), 1 );
#endif
    /* if we get here, then we have no choice but to abort ... */
  }


  /* Use warning callback to avoid png writing to stderr. */
  static void
  warning_callback( png_structp      png,
                    png_const_charp  error_msg )
  {
    FT_UNUSED( png );
    FT_UNUSED( error_msg );

    /* Just ignore warnings. */
  }


  static void
  read_data_from_FT_Stream( png_structp  png,
                            png_bytep    data,
                            png_size_t   length )
  {
    FT_Error   error;
    png_voidp  p      = png_get_io_ptr( png );
    FT_Stream  stream = (FT_Stream)p;


    if ( FT_FRAME_ENTER( length ) )
    {
      FT_Error*  e = (FT_Error*)png_get_error_ptr( png );


      *e = FT_THROW( Invalid_Stream_Read );
      png_error( png, NULL );

      return;
    }

    ft_memcpy( data, stream->cursor, length );

    FT_FRAME_EXIT();
  }


  FT_LOCAL_DEF( FT_Error )
  Load_SBit_Png( FT_GlyphSlot     slot,
                 FT_Int           x_offset,
                 FT_Int           y_offset,
                 FT_Int           pix_bits,
                 TT_SBit_Metrics  metrics,
                 FT_Memory        memory,
                 FT_Byte*         data,
                 FT_UInt          png_len,
                 FT_Bool          populate_map_and_metrics,
                 FT_Bool          metrics_only )
  {
    FT_Bitmap    *map   = &slot->bitmap;
    FT_Error      error = FT_Err_Ok;
    FT_StreamRec  stream;

    png_structp  png;
    png_infop    info;
    png_uint_32  imgWidth, imgHeight;

    int         bitdepth, color_type, interlace;
    FT_Int      i;

    /* `rows` gets modified within a 'setjmp' scope; */
    /* we thus need the `volatile` keyword.          */
    png_byte* *volatile  rows = NULL;


    if ( x_offset < 0 ||
         y_offset < 0 )
    {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    if ( !populate_map_and_metrics                            &&
         ( (FT_UInt)x_offset + metrics->width  > map->width ||
           (FT_UInt)y_offset + metrics->height > map->rows  ||
           pix_bits != 32                                   ||
           map->pixel_mode != FT_PIXEL_MODE_BGRA            ) )
    {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    FT_Stream_OpenMemory( &stream, data, png_len );

    png = png_create_read_struct( PNG_LIBPNG_VER_STRING,
                                  &error,
                                  error_callback,
                                  warning_callback );
    if ( !png )
    {
      error = FT_THROW( Out_Of_Memory );
      goto Exit;
    }

    info = png_create_info_struct( png );
    if ( !info )
    {
      error = FT_THROW( Out_Of_Memory );
      png_destroy_read_struct( &png, NULL, NULL );
      goto Exit;
    }

    if ( ft_setjmp( png_jmpbuf( png ) ) )
    {
      error = FT_THROW( Invalid_File_Format );
      goto DestroyExit;
    }

    png_set_read_fn( png, &stream, read_data_from_FT_Stream );

    png_read_info( png, info );
    png_get_IHDR( png, info,
                  &imgWidth, &imgHeight,
                  &bitdepth, &color_type, &interlace,
                  NULL, NULL );

    if ( error                                        ||
         ( !populate_map_and_metrics                &&
           ( (FT_Int)imgWidth  != metrics->width  ||
             (FT_Int)imgHeight != metrics->height ) ) )
      goto DestroyExit;

    if ( populate_map_and_metrics )
    {
      /* reject too large bitmaps similarly to the rasterizer */
      if ( imgHeight > 0x7FFF || imgWidth > 0x7FFF )
      {
        error = FT_THROW( Array_Too_Large );
        goto DestroyExit;
      }

      metrics->width  = (FT_UShort)imgWidth;
      metrics->height = (FT_UShort)imgHeight;

      map->width      = metrics->width;
      map->rows       = metrics->height;
      map->pixel_mode = FT_PIXEL_MODE_BGRA;
      map->pitch      = (int)( map->width * 4 );
      map->num_grays  = 256;
    }

    /* convert palette/gray image to rgb */
    if ( color_type == PNG_COLOR_TYPE_PALETTE )
      png_set_palette_to_rgb( png );

    /* expand gray bit depth if needed */
    if ( color_type == PNG_COLOR_TYPE_GRAY )
    {
#if PNG_LIBPNG_VER >= 10209
      png_set_expand_gray_1_2_4_to_8( png );
#else
      png_set_gray_1_2_4_to_8( png );
#endif
    }

    /* transform transparency to alpha */
    if ( png_get_valid(png, info, PNG_INFO_tRNS ) )
      png_set_tRNS_to_alpha( png );

    if ( bitdepth == 16 )
      png_set_strip_16( png );

    if ( bitdepth < 8 )
      png_set_packing( png );

    /* convert grayscale to RGB */
    if ( color_type == PNG_COLOR_TYPE_GRAY       ||
         color_type == PNG_COLOR_TYPE_GRAY_ALPHA )
      png_set_gray_to_rgb( png );

    if ( interlace != PNG_INTERLACE_NONE )
      png_set_interlace_handling( png );

    png_set_filler( png, 0xFF, PNG_FILLER_AFTER );

    /* recheck header after setting EXPAND options */
    png_read_update_info(png, info );
    png_get_IHDR( png, info,
                  &imgWidth, &imgHeight,
                  &bitdepth, &color_type, &interlace,
                  NULL, NULL );

    if ( bitdepth != 8                              ||
        !( color_type == PNG_COLOR_TYPE_RGB       ||
           color_type == PNG_COLOR_TYPE_RGB_ALPHA ) )
    {
      error = FT_THROW( Invalid_File_Format );
      goto DestroyExit;
    }

    if ( metrics_only )
      goto DestroyExit;

    switch ( color_type )
    {
    default:
      /* Shouldn't happen, but fall through. */

    case PNG_COLOR_TYPE_RGB_ALPHA:
      png_set_read_user_transform_fn( png, premultiply_data );
      break;

    case PNG_COLOR_TYPE_RGB:
      /* Humm, this smells.  Carry on though. */
      png_set_read_user_transform_fn( png, convert_bytes_to_data );
      break;
    }

    if ( populate_map_and_metrics )
    {
      /* this doesn't overflow: 0x7FFF * 0x7FFF * 4 < 2^32 */
      FT_ULong  size = map->rows * (FT_ULong)map->pitch;


      error = ft_glyphslot_alloc_bitmap( slot, size );
      if ( error )
        goto DestroyExit;
    }

    if ( FT_QNEW_ARRAY( rows, imgHeight ) )
    {
      error = FT_THROW( Out_Of_Memory );
      goto DestroyExit;
    }

    for ( i = 0; i < (FT_Int)imgHeight; i++ )
      rows[i] = map->buffer + ( y_offset + i ) * map->pitch + x_offset * 4;

    png_read_image( png, rows );

    png_read_end( png, info );

  DestroyExit:
    /* even if reading fails with longjmp, rows must be freed */
    FT_FREE( rows );
    png_destroy_read_struct( &png, &info, NULL );
    FT_Stream_Close( &stream );

  Exit:
    return error;
  }

#else /* !(TT_CONFIG_OPTION_EMBEDDED_BITMAPS && FT_CONFIG_OPTION_USE_PNG) */

  /* ANSI C doesn't like empty source files */
  typedef int  _pngshim_dummy;

#endif /* !(TT_CONFIG_OPTION_EMBEDDED_BITMAPS && FT_CONFIG_OPTION_USE_PNG) */


/* END */
