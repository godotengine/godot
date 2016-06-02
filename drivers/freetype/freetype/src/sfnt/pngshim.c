/***************************************************************************/
/*                                                                         */
/*  pngshim.c                                                              */
/*                                                                         */
/*    PNG Bitmap glyph support.                                            */
/*                                                                         */
/*  Copyright 2013 by Google, Inc.                                         */
/*  Written by Stuart Gill and Behdad Esfahbod.                            */
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
#include FT_CONFIG_STANDARD_LIBRARY_H


#ifdef FT_CONFIG_OPTION_USE_PNG

  /* We always include <stjmp.h>, so make libpng shut up! */
#define PNG_SKIP_SETJMP_CHECK 1
#include <png.h>
#include "pngshim.h"

#include "sferrors.h"


  /* This code is freely based on cairo-png.c.  There's so many ways */
  /* to call libpng, and the way cairo does it is defacto standard.  */

  static int
  multiply_alpha( int  alpha,
                  int  color )
  {
    int  temp = ( alpha * color ) + 0x80;


    return ( temp + ( temp >> 8 ) ) >> 8;
  }


  /* Premultiplies data and converts RGBA bytes => native endian. */
  static void
  premultiply_data( png_structp    png,
                    png_row_infop  row_info,
                    png_bytep      data )
  {
    unsigned int  i;

    FT_UNUSED( png );


    for ( i = 0; i < row_info->rowbytes; i += 4 )
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

        base[0] = blue;
        base[1] = green;
        base[2] = red;
        base[3] = alpha;
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


      base[0] = blue;
      base[1] = green;
      base[2] = red;
      base[3] = 0xFF;
    }
  }


  /* Use error callback to avoid png writing to stderr. */
  static void
  error_callback( png_structp      png,
                  png_const_charp  error_msg )
  {
    FT_Error*  error = png_get_error_ptr( png );

    FT_UNUSED( error_msg );


    *error = FT_THROW( Out_Of_Memory );
#ifdef PNG_SETJMP_SUPPORTED
    longjmp( png_jmpbuf( png ), 1 );
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
      FT_Error*  e = png_get_error_ptr( png );


      *e = FT_THROW( Invalid_Stream_Read );
      png_error( png, NULL );

      return;
    }

    memcpy( data, stream->cursor, length );

    FT_FRAME_EXIT();
  }


  static FT_Error
  Load_SBit_Png( FT_Bitmap*       map,
                 FT_Int           x_offset,
                 FT_Int           y_offset,
                 FT_Int           pix_bits,
                 TT_SBit_Metrics  metrics,
                 FT_Memory        memory,
                 FT_Byte*         data,
                 FT_UInt          png_len )
  {
    FT_Error      error = FT_Err_Ok;
    FT_StreamRec  stream;

    png_structp  png;
    png_infop    info;
    png_uint_32  imgWidth, imgHeight;

    int         bitdepth, color_type, interlace;
    FT_Int      i;
    png_byte*  *rows;


    if ( x_offset < 0 || x_offset + metrics->width  > map->width ||
         y_offset < 0 || y_offset + metrics->height > map->rows  ||
         pix_bits != 32 || map->pixel_mode != FT_PIXEL_MODE_BGRA )
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

    if ( error != FT_Err_Ok                   ||
         (FT_Int)imgWidth  != metrics->width  ||
         (FT_Int)imgHeight != metrics->height )
      goto DestroyExit;

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

    if ( FT_NEW_ARRAY( rows, imgHeight ) )
    {
      error = FT_THROW( Out_Of_Memory );
      goto DestroyExit;
    }

    for ( i = 0; i < (FT_Int)imgHeight; i++ )
      rows[i] = map->buffer + ( y_offset + i ) * map->pitch + x_offset * 4;

    png_read_image( png, rows );

    FT_FREE( rows );

    png_read_end( png, info );

  DestroyExit:
    png_destroy_read_struct( &png, &info, NULL );
    FT_Stream_Close( &stream );

  Exit:
    return error;
  }

#endif /* FT_CONFIG_OPTION_USE_PNG */


/* END */
