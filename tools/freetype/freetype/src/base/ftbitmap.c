/***************************************************************************/
/*                                                                         */
/*  ftbitmap.c                                                             */
/*                                                                         */
/*    FreeType utility functions for bitmaps (body).                       */
/*                                                                         */
/*  Copyright 2004-2009, 2011, 2013 by                                     */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
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

#include FT_BITMAP_H
#include FT_IMAGE_H
#include FT_INTERNAL_OBJECTS_H


  static
  const FT_Bitmap  null_bitmap = { 0, 0, 0, 0, 0, 0, 0, 0 };


  /* documentation is in ftbitmap.h */

  FT_EXPORT_DEF( void )
  FT_Bitmap_New( FT_Bitmap  *abitmap )
  {
    *abitmap = null_bitmap;
  }


  /* documentation is in ftbitmap.h */

  FT_EXPORT_DEF( FT_Error )
  FT_Bitmap_Copy( FT_Library        library,
                  const FT_Bitmap  *source,
                  FT_Bitmap        *target)
  {
    FT_Memory  memory = library->memory;
    FT_Error   error  = FT_Err_Ok;
    FT_Int     pitch  = source->pitch;
    FT_ULong   size;


    if ( source == target )
      return FT_Err_Ok;

    if ( source->buffer == NULL )
    {
      *target = *source;

      return FT_Err_Ok;
    }

    if ( pitch < 0 )
      pitch = -pitch;
    size = (FT_ULong)( pitch * source->rows );

    if ( target->buffer )
    {
      FT_Int    target_pitch = target->pitch;
      FT_ULong  target_size;


      if ( target_pitch < 0  )
        target_pitch = -target_pitch;
      target_size = (FT_ULong)( target_pitch * target->rows );

      if ( target_size != size )
        (void)FT_QREALLOC( target->buffer, target_size, size );
    }
    else
      (void)FT_QALLOC( target->buffer, size );

    if ( !error )
    {
      unsigned char *p;


      p = target->buffer;
      *target = *source;
      target->buffer = p;

      FT_MEM_COPY( target->buffer, source->buffer, size );
    }

    return error;
  }


  static FT_Error
  ft_bitmap_assure_buffer( FT_Memory   memory,
                           FT_Bitmap*  bitmap,
                           FT_UInt     xpixels,
                           FT_UInt     ypixels )
  {
    FT_Error        error;
    int             pitch;
    int             new_pitch;
    FT_UInt         bpp;
    FT_Int          i, width, height;
    unsigned char*  buffer = NULL;


    width  = bitmap->width;
    height = bitmap->rows;
    pitch  = bitmap->pitch;
    if ( pitch < 0 )
      pitch = -pitch;

    switch ( bitmap->pixel_mode )
    {
    case FT_PIXEL_MODE_MONO:
      bpp       = 1;
      new_pitch = ( width + xpixels + 7 ) >> 3;
      break;
    case FT_PIXEL_MODE_GRAY2:
      bpp       = 2;
      new_pitch = ( width + xpixels + 3 ) >> 2;
      break;
    case FT_PIXEL_MODE_GRAY4:
      bpp       = 4;
      new_pitch = ( width + xpixels + 1 ) >> 1;
      break;
    case FT_PIXEL_MODE_GRAY:
    case FT_PIXEL_MODE_LCD:
    case FT_PIXEL_MODE_LCD_V:
      bpp       = 8;
      new_pitch = ( width + xpixels );
      break;
    default:
      return FT_THROW( Invalid_Glyph_Format );
    }

    /* if no need to allocate memory */
    if ( ypixels == 0 && new_pitch <= pitch )
    {
      /* zero the padding */
      FT_Int  bit_width = pitch * 8;
      FT_Int  bit_last  = ( width + xpixels ) * bpp;


      if ( bit_last < bit_width )
      {
        FT_Byte*  line  = bitmap->buffer + ( bit_last >> 3 );
        FT_Byte*  end   = bitmap->buffer + pitch;
        FT_Int    shift = bit_last & 7;
        FT_UInt   mask  = 0xFF00U >> shift;
        FT_Int    count = height;


        for ( ; count > 0; count--, line += pitch, end += pitch )
        {
          FT_Byte*  write = line;


          if ( shift > 0 )
          {
            write[0] = (FT_Byte)( write[0] & mask );
            write++;
          }
          if ( write < end )
            FT_MEM_ZERO( write, end-write );
        }
      }

      return FT_Err_Ok;
    }

    if ( FT_QALLOC_MULT( buffer, new_pitch, bitmap->rows + ypixels ) )
      return error;

    if ( bitmap->pitch > 0 )
    {
      FT_Int  len = ( width * bpp + 7 ) >> 3;


      for ( i = 0; i < bitmap->rows; i++ )
        FT_MEM_COPY( buffer + new_pitch * ( ypixels + i ),
                     bitmap->buffer + pitch * i, len );
    }
    else
    {
      FT_Int  len = ( width * bpp + 7 ) >> 3;


      for ( i = 0; i < bitmap->rows; i++ )
        FT_MEM_COPY( buffer + new_pitch * i,
                     bitmap->buffer + pitch * i, len );
    }

    FT_FREE( bitmap->buffer );
    bitmap->buffer = buffer;

    if ( bitmap->pitch < 0 )
      new_pitch = -new_pitch;

    /* set pitch only, width and height are left untouched */
    bitmap->pitch = new_pitch;

    return FT_Err_Ok;
  }


  /* documentation is in ftbitmap.h */

  FT_EXPORT_DEF( FT_Error )
  FT_Bitmap_Embolden( FT_Library  library,
                      FT_Bitmap*  bitmap,
                      FT_Pos      xStrength,
                      FT_Pos      yStrength )
  {
    FT_Error        error;
    unsigned char*  p;
    FT_Int          i, x, y, pitch;
    FT_Int          xstr, ystr;


    if ( !library )
      return FT_THROW( Invalid_Library_Handle );

    if ( !bitmap || !bitmap->buffer )
      return FT_THROW( Invalid_Argument );

    if ( ( ( FT_PIX_ROUND( xStrength ) >> 6 ) > FT_INT_MAX ) ||
         ( ( FT_PIX_ROUND( yStrength ) >> 6 ) > FT_INT_MAX ) )
      return FT_THROW( Invalid_Argument );

    xstr = (FT_Int)FT_PIX_ROUND( xStrength ) >> 6;
    ystr = (FT_Int)FT_PIX_ROUND( yStrength ) >> 6;

    if ( xstr == 0 && ystr == 0 )
      return FT_Err_Ok;
    else if ( xstr < 0 || ystr < 0 )
      return FT_THROW( Invalid_Argument );

    switch ( bitmap->pixel_mode )
    {
    case FT_PIXEL_MODE_GRAY2:
    case FT_PIXEL_MODE_GRAY4:
      {
        FT_Bitmap  tmp;
        FT_Int     align;


        if ( bitmap->pixel_mode == FT_PIXEL_MODE_GRAY2 )
          align = ( bitmap->width + xstr + 3 ) / 4;
        else
          align = ( bitmap->width + xstr + 1 ) / 2;

        FT_Bitmap_New( &tmp );

        error = FT_Bitmap_Convert( library, bitmap, &tmp, align );
        if ( error )
          return error;

        FT_Bitmap_Done( library, bitmap );
        *bitmap = tmp;
      }
      break;

    case FT_PIXEL_MODE_MONO:
      if ( xstr > 8 )
        xstr = 8;
      break;

    case FT_PIXEL_MODE_LCD:
      xstr *= 3;
      break;

    case FT_PIXEL_MODE_LCD_V:
      ystr *= 3;
      break;

    case FT_PIXEL_MODE_BGRA:
      /* We don't embolden color glyphs. */
      return FT_Err_Ok;
    }

    error = ft_bitmap_assure_buffer( library->memory, bitmap, xstr, ystr );
    if ( error )
      return error;

    pitch = bitmap->pitch;
    if ( pitch > 0 )
      p = bitmap->buffer + pitch * ystr;
    else
    {
      pitch = -pitch;
      p = bitmap->buffer + pitch * ( bitmap->rows - 1 );
    }

    /* for each row */
    for ( y = 0; y < bitmap->rows ; y++ )
    {
      /*
       * Horizontally:
       *
       * From the last pixel on, make each pixel or'ed with the
       * `xstr' pixels before it.
       */
      for ( x = pitch - 1; x >= 0; x-- )
      {
        unsigned char tmp;


        tmp = p[x];
        for ( i = 1; i <= xstr; i++ )
        {
          if ( bitmap->pixel_mode == FT_PIXEL_MODE_MONO )
          {
            p[x] |= tmp >> i;

            /* the maximum value of 8 for `xstr' comes from here */
            if ( x > 0 )
              p[x] |= p[x - 1] << ( 8 - i );

#if 0
            if ( p[x] == 0xff )
              break;
#endif
          }
          else
          {
            if ( x - i >= 0 )
            {
              if ( p[x] + p[x - i] > bitmap->num_grays - 1 )
              {
                p[x] = (unsigned char)(bitmap->num_grays - 1);
                break;
              }
              else
              {
                p[x] = (unsigned char)(p[x] + p[x-i]);
                if ( p[x] == bitmap->num_grays - 1 )
                  break;
              }
            }
            else
              break;
          }
        }
      }

      /*
       * Vertically:
       *
       * Make the above `ystr' rows or'ed with it.
       */
      for ( x = 1; x <= ystr; x++ )
      {
        unsigned char*  q;


        q = p - bitmap->pitch * x;
        for ( i = 0; i < pitch; i++ )
          q[i] |= p[i];
      }

      p += bitmap->pitch;
    }

    bitmap->width += xstr;
    bitmap->rows += ystr;

    return FT_Err_Ok;
  }


  FT_Byte
  ft_gray_for_premultiplied_srgb_bgra( const FT_Byte*  bgra )
  {
    FT_Long  a = bgra[3];
    FT_Long  b = bgra[0];
    FT_Long  g = bgra[1];
    FT_Long  r = bgra[2];
    FT_Long  l;


    /*
     * Luminosity for sRGB is defined using ~0.2126,0.7152,0.0722
     * coefficients for RGB channels *on the linear colors*.
     * A gamma of 2.2 is fair to assume.  And then, we need to
     * undo the premultiplication too.
     *
     * http://accessibility.kde.org/hsl-adjusted.php
     *
     * We do the computation with integers only.
     */

    /* Undo premultification, get the number in a 16.16 form. */
    b = FT_MulDiv( b, 65536, a );
    g = FT_MulDiv( g, 65536, a );
    r = FT_MulDiv( r, 65536, a );
    a = a * 256;

    /* Apply gamma of 2.0 instead of 2.2. */
    b = FT_MulFix( b, b );
    g = FT_MulFix( g, g );
    r = FT_MulFix( r, r );

    /* Apply coefficients. */
    b = FT_MulFix( b,  4731 /* 0.0722 * 65536 */ );
    g = FT_MulFix( g, 46871 /* 0.7152 * 65536 */ );
    r = FT_MulFix( r, 13933 /* 0.2126 * 65536 */ );

    l = r + g + b;

    /*
     * Final transparency can be determined this way:
     *
     * - If alpha is zero, we want 0.
     * - If alpha is zero and luminosity is zero, we want 255.
     * - If alpha is zero and luminosity is one, we want 0.
     *
     * So the formula is a * (1 - l).
     */

    return (FT_Byte)( FT_MulFix( 65535 - l, a ) >> 8 );
  }


  /* documentation is in ftbitmap.h */

  FT_EXPORT_DEF( FT_Error )
  FT_Bitmap_Convert( FT_Library        library,
                     const FT_Bitmap  *source,
                     FT_Bitmap        *target,
                     FT_Int            alignment )
  {
    FT_Error   error = FT_Err_Ok;
    FT_Memory  memory;


    if ( !library )
      return FT_THROW( Invalid_Library_Handle );

    memory = library->memory;

    switch ( source->pixel_mode )
    {
    case FT_PIXEL_MODE_MONO:
    case FT_PIXEL_MODE_GRAY:
    case FT_PIXEL_MODE_GRAY2:
    case FT_PIXEL_MODE_GRAY4:
    case FT_PIXEL_MODE_LCD:
    case FT_PIXEL_MODE_LCD_V:
    case FT_PIXEL_MODE_BGRA:
      {
        FT_Int   pad;
        FT_Long  old_size;


        old_size = target->rows * target->pitch;
        if ( old_size < 0 )
          old_size = -old_size;

        target->pixel_mode = FT_PIXEL_MODE_GRAY;
        target->rows       = source->rows;
        target->width      = source->width;

        pad = 0;
        if ( alignment > 0 )
        {
          pad = source->width % alignment;
          if ( pad != 0 )
            pad = alignment - pad;
        }

        target->pitch = source->width + pad;

        if ( target->pitch > 0                                     &&
             (FT_ULong)target->rows > FT_ULONG_MAX / target->pitch )
          return FT_THROW( Invalid_Argument );

        if ( target->rows * target->pitch > old_size             &&
             FT_QREALLOC( target->buffer,
                          old_size, target->rows * target->pitch ) )
          return error;
      }
      break;

    default:
      error = FT_THROW( Invalid_Argument );
    }

    switch ( source->pixel_mode )
    {
    case FT_PIXEL_MODE_MONO:
      {
        FT_Byte*  s = source->buffer;
        FT_Byte*  t = target->buffer;
        FT_Int    i;


        target->num_grays = 2;

        for ( i = source->rows; i > 0; i-- )
        {
          FT_Byte*  ss = s;
          FT_Byte*  tt = t;
          FT_Int    j;


          /* get the full bytes */
          for ( j = source->width >> 3; j > 0; j-- )
          {
            FT_Int  val = ss[0]; /* avoid a byte->int cast on each line */


            tt[0] = (FT_Byte)( ( val & 0x80 ) >> 7 );
            tt[1] = (FT_Byte)( ( val & 0x40 ) >> 6 );
            tt[2] = (FT_Byte)( ( val & 0x20 ) >> 5 );
            tt[3] = (FT_Byte)( ( val & 0x10 ) >> 4 );
            tt[4] = (FT_Byte)( ( val & 0x08 ) >> 3 );
            tt[5] = (FT_Byte)( ( val & 0x04 ) >> 2 );
            tt[6] = (FT_Byte)( ( val & 0x02 ) >> 1 );
            tt[7] = (FT_Byte)(   val & 0x01 );

            tt += 8;
            ss += 1;
          }

          /* get remaining pixels (if any) */
          j = source->width & 7;
          if ( j > 0 )
          {
            FT_Int  val = *ss;


            for ( ; j > 0; j-- )
            {
              tt[0] = (FT_Byte)( ( val & 0x80 ) >> 7);
              val <<= 1;
              tt   += 1;
            }
          }

          s += source->pitch;
          t += target->pitch;
        }
      }
      break;


    case FT_PIXEL_MODE_GRAY:
    case FT_PIXEL_MODE_LCD:
    case FT_PIXEL_MODE_LCD_V:
      {
        FT_Int    width   = source->width;
        FT_Byte*  s       = source->buffer;
        FT_Byte*  t       = target->buffer;
        FT_Int    s_pitch = source->pitch;
        FT_Int    t_pitch = target->pitch;
        FT_Int    i;


        target->num_grays = 256;

        for ( i = source->rows; i > 0; i-- )
        {
          FT_ARRAY_COPY( t, s, width );

          s += s_pitch;
          t += t_pitch;
        }
      }
      break;


    case FT_PIXEL_MODE_GRAY2:
      {
        FT_Byte*  s = source->buffer;
        FT_Byte*  t = target->buffer;
        FT_Int    i;


        target->num_grays = 4;

        for ( i = source->rows; i > 0; i-- )
        {
          FT_Byte*  ss = s;
          FT_Byte*  tt = t;
          FT_Int    j;


          /* get the full bytes */
          for ( j = source->width >> 2; j > 0; j-- )
          {
            FT_Int  val = ss[0];


            tt[0] = (FT_Byte)( ( val & 0xC0 ) >> 6 );
            tt[1] = (FT_Byte)( ( val & 0x30 ) >> 4 );
            tt[2] = (FT_Byte)( ( val & 0x0C ) >> 2 );
            tt[3] = (FT_Byte)( ( val & 0x03 ) );

            ss += 1;
            tt += 4;
          }

          j = source->width & 3;
          if ( j > 0 )
          {
            FT_Int  val = ss[0];


            for ( ; j > 0; j-- )
            {
              tt[0]  = (FT_Byte)( ( val & 0xC0 ) >> 6 );
              val  <<= 2;
              tt    += 1;
            }
          }

          s += source->pitch;
          t += target->pitch;
        }
      }
      break;


    case FT_PIXEL_MODE_GRAY4:
      {
        FT_Byte*  s = source->buffer;
        FT_Byte*  t = target->buffer;
        FT_Int    i;


        target->num_grays = 16;

        for ( i = source->rows; i > 0; i-- )
        {
          FT_Byte*  ss = s;
          FT_Byte*  tt = t;
          FT_Int    j;


          /* get the full bytes */
          for ( j = source->width >> 1; j > 0; j-- )
          {
            FT_Int  val = ss[0];


            tt[0] = (FT_Byte)( ( val & 0xF0 ) >> 4 );
            tt[1] = (FT_Byte)( ( val & 0x0F ) );

            ss += 1;
            tt += 2;
          }

          if ( source->width & 1 )
            tt[0] = (FT_Byte)( ( ss[0] & 0xF0 ) >> 4 );

          s += source->pitch;
          t += target->pitch;
        }
      }
      break;

    case FT_PIXEL_MODE_BGRA:
      {
        FT_Byte*  s       = source->buffer;
        FT_Byte*  t       = target->buffer;
        FT_Int    s_pitch = source->pitch;
        FT_Int    t_pitch = target->pitch;
        FT_Int    i;


        target->num_grays = 256;

        for ( i = source->rows; i > 0; i-- )
        {
          FT_Byte*  ss = s;
          FT_Byte*  tt = t;
          FT_Int    j;


          for ( j = source->width; j > 0; j-- )
          {
            tt[0] = ft_gray_for_premultiplied_srgb_bgra( ss );

            ss += 4;
            tt += 1;
          }

          s += s_pitch;
          t += t_pitch;
        }
      }
      break;

    default:
      ;
    }

    return error;
  }


  /* documentation is in ftbitmap.h */

  FT_EXPORT_DEF( FT_Error )
  FT_GlyphSlot_Own_Bitmap( FT_GlyphSlot  slot )
  {
    if ( slot && slot->format == FT_GLYPH_FORMAT_BITMAP   &&
         !( slot->internal->flags & FT_GLYPH_OWN_BITMAP ) )
    {
      FT_Bitmap  bitmap;
      FT_Error   error;


      FT_Bitmap_New( &bitmap );
      error = FT_Bitmap_Copy( slot->library, &slot->bitmap, &bitmap );
      if ( error )
        return error;

      slot->bitmap = bitmap;
      slot->internal->flags |= FT_GLYPH_OWN_BITMAP;
    }

    return FT_Err_Ok;
  }


  /* documentation is in ftbitmap.h */

  FT_EXPORT_DEF( FT_Error )
  FT_Bitmap_Done( FT_Library  library,
                  FT_Bitmap  *bitmap )
  {
    FT_Memory  memory;


    if ( !library )
      return FT_THROW( Invalid_Library_Handle );

    if ( !bitmap )
      return FT_THROW( Invalid_Argument );

    memory = library->memory;

    FT_FREE( bitmap->buffer );
    *bitmap = null_bitmap;

    return FT_Err_Ok;
  }


/* END */
