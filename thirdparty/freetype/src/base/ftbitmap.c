/****************************************************************************
 *
 * ftbitmap.c
 *
 *   FreeType utility functions for bitmaps (body).
 *
 * Copyright (C) 2004-2020 by
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

#include <freetype/ftbitmap.h>
#include <freetype/ftimage.h>
#include <freetype/internal/ftobjs.h>


  /**************************************************************************
   *
   * The macro FT_COMPONENT is used in trace mode.  It is an implicit
   * parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log
   * messages during execution.
   */
#undef  FT_COMPONENT
#define FT_COMPONENT  bitmap


  static
  const FT_Bitmap  null_bitmap = { 0, 0, 0, NULL, 0, 0, 0, NULL };


  /* documentation is in ftbitmap.h */

  FT_EXPORT_DEF( void )
  FT_Bitmap_Init( FT_Bitmap  *abitmap )
  {
    if ( abitmap )
      *abitmap = null_bitmap;
  }


  /* deprecated function name; retained for ABI compatibility */

  FT_EXPORT_DEF( void )
  FT_Bitmap_New( FT_Bitmap  *abitmap )
  {
    if ( abitmap )
      *abitmap = null_bitmap;
  }


  /* documentation is in ftbitmap.h */

  FT_EXPORT_DEF( FT_Error )
  FT_Bitmap_Copy( FT_Library        library,
                  const FT_Bitmap  *source,
                  FT_Bitmap        *target)
  {
    FT_Memory  memory;
    FT_Error   error  = FT_Err_Ok;

    FT_Int    pitch;
    FT_ULong  size;

    FT_Int  source_pitch_sign, target_pitch_sign;


    if ( !library )
      return FT_THROW( Invalid_Library_Handle );

    if ( !source || !target )
      return FT_THROW( Invalid_Argument );

    if ( source == target )
      return FT_Err_Ok;

    source_pitch_sign = source->pitch < 0 ? -1 : 1;
    target_pitch_sign = target->pitch < 0 ? -1 : 1;

    if ( !source->buffer )
    {
      *target = *source;
      if ( source_pitch_sign != target_pitch_sign )
        target->pitch = -target->pitch;

      return FT_Err_Ok;
    }

    memory = library->memory;
    pitch  = source->pitch;

    if ( pitch < 0 )
      pitch = -pitch;
    size = (FT_ULong)pitch * source->rows;

    if ( target->buffer )
    {
      FT_Int    target_pitch = target->pitch;
      FT_ULong  target_size;


      if ( target_pitch < 0 )
        target_pitch = -target_pitch;
      target_size = (FT_ULong)target_pitch * target->rows;

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

      if ( source_pitch_sign == target_pitch_sign )
        FT_MEM_COPY( target->buffer, source->buffer, size );
      else
      {
        /* take care of bitmap flow */
        FT_UInt   i;
        FT_Byte*  s = source->buffer;
        FT_Byte*  t = target->buffer;


        t += (FT_ULong)pitch * ( target->rows - 1 );

        for ( i = target->rows; i > 0; i-- )
        {
          FT_ARRAY_COPY( t, s, pitch );

          s += pitch;
          t -= pitch;
        }
      }
    }

    return error;
  }


  /* Enlarge `bitmap' horizontally and vertically by `xpixels' */
  /* and `ypixels', respectively.                              */

  static FT_Error
  ft_bitmap_assure_buffer( FT_Memory   memory,
                           FT_Bitmap*  bitmap,
                           FT_UInt     xpixels,
                           FT_UInt     ypixels )
  {
    FT_Error        error;
    unsigned int    pitch;
    unsigned int    new_pitch;
    FT_UInt         bpp;
    FT_UInt         width, height;
    unsigned char*  buffer = NULL;


    width  = bitmap->width;
    height = bitmap->rows;
    pitch  = (unsigned int)FT_ABS( bitmap->pitch );

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
      new_pitch = width + xpixels;
      break;
    default:
      return FT_THROW( Invalid_Glyph_Format );
    }

    /* if no need to allocate memory */
    if ( ypixels == 0 && new_pitch <= pitch )
    {
      /* zero the padding */
      FT_UInt  bit_width = pitch * 8;
      FT_UInt  bit_last  = ( width + xpixels ) * bpp;


      if ( bit_last < bit_width )
      {
        FT_Byte*  line  = bitmap->buffer + ( bit_last >> 3 );
        FT_Byte*  end   = bitmap->buffer + pitch;
        FT_UInt   shift = bit_last & 7;
        FT_UInt   mask  = 0xFF00U >> shift;
        FT_UInt   count = height;


        for ( ; count > 0; count--, line += pitch, end += pitch )
        {
          FT_Byte*  write = line;


          if ( shift > 0 )
          {
            write[0] = (FT_Byte)( write[0] & mask );
            write++;
          }
          if ( write < end )
            FT_MEM_ZERO( write, end - write );
        }
      }

      return FT_Err_Ok;
    }

    /* otherwise allocate new buffer */
    if ( FT_QALLOC_MULT( buffer, bitmap->rows + ypixels, new_pitch ) )
      return error;

    /* new rows get added at the top of the bitmap, */
    /* thus take care of the flow direction         */
    if ( bitmap->pitch > 0 )
    {
      FT_UInt  len = ( width * bpp + 7 ) >> 3;

      unsigned char*  in  = bitmap->buffer;
      unsigned char*  out = buffer;

      unsigned char*  limit = bitmap->buffer + pitch * bitmap->rows;
      unsigned int    delta = new_pitch - len;


      FT_MEM_ZERO( out, new_pitch * ypixels );
      out += new_pitch * ypixels;

      while ( in < limit )
      {
        FT_MEM_COPY( out, in, len );
        in  += pitch;
        out += len;

        /* we use FT_QALLOC_MULT, which doesn't zero out the buffer;      */
        /* consequently, we have to manually zero out the remaining bytes */
        FT_MEM_ZERO( out, delta );
        out += delta;
      }
    }
    else
    {
      FT_UInt  len = ( width * bpp + 7 ) >> 3;

      unsigned char*  in  = bitmap->buffer;
      unsigned char*  out = buffer;

      unsigned char*  limit = bitmap->buffer + pitch * bitmap->rows;
      unsigned int    delta = new_pitch - len;


      while ( in < limit )
      {
        FT_MEM_COPY( out, in, len );
        in  += pitch;
        out += len;

        FT_MEM_ZERO( out, delta );
        out += delta;
      }

      FT_MEM_ZERO( out, new_pitch * ypixels );
    }

    FT_FREE( bitmap->buffer );
    bitmap->buffer = buffer;

    /* set pitch only, width and height are left untouched */
    if ( bitmap->pitch < 0 )
      bitmap->pitch = -(int)new_pitch;
    else
      bitmap->pitch = (int)new_pitch;

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
    FT_Int          i, x, pitch;
    FT_UInt         y;
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


        /* convert to 8bpp */
        FT_Bitmap_Init( &tmp );
        error = FT_Bitmap_Convert( library, bitmap, &tmp, 1 );
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

    error = ft_bitmap_assure_buffer( library->memory, bitmap,
                                     (FT_UInt)xstr, (FT_UInt)ystr );
    if ( error )
      return error;

    /* take care of bitmap flow */
    pitch = bitmap->pitch;
    if ( pitch > 0 )
      p = bitmap->buffer + pitch * ystr;
    else
    {
      pitch = -pitch;
      p = bitmap->buffer + (FT_UInt)pitch * ( bitmap->rows - 1 );
    }

    /* for each row */
    for ( y = 0; y < bitmap->rows; y++ )
    {
      /*
       * Horizontally:
       *
       * From the last pixel on, make each pixel or'ed with the
       * `xstr' pixels before it.
       */
      for ( x = pitch - 1; x >= 0; x-- )
      {
        unsigned char  tmp;


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
            if ( p[x] == 0xFF )
              break;
#endif
          }
          else
          {
            if ( x - i >= 0 )
            {
              if ( p[x] + p[x - i] > bitmap->num_grays - 1 )
              {
                p[x] = (unsigned char)( bitmap->num_grays - 1 );
                break;
              }
              else
              {
                p[x] = (unsigned char)( p[x] + p[x - i] );
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

    bitmap->width += (FT_UInt)xstr;
    bitmap->rows += (FT_UInt)ystr;

    return FT_Err_Ok;
  }


  static FT_Byte
  ft_gray_for_premultiplied_srgb_bgra( const FT_Byte*  bgra )
  {
    FT_UInt  a = bgra[3];
    FT_UInt  l;


    /* Short-circuit transparent color to avoid division by zero. */
    if ( !a )
      return 0;

    /*
     * Luminosity for sRGB is defined using ~0.2126,0.7152,0.0722
     * coefficients for RGB channels *on the linear colors*.
     * A gamma of 2.2 is fair to assume.  And then, we need to
     * undo the premultiplication too.
     *
     *   https://accessibility.kde.org/hsl-adjusted.php
     *
     * We do the computation with integers only, applying a gamma of 2.0.
     * We guarantee 32-bit arithmetic to avoid overflow but the resulting
     * luminosity fits into 16 bits.
     *
     */

    l = (  4732UL /* 0.0722 * 65536 */ * bgra[0] * bgra[0] +
          46871UL /* 0.7152 * 65536 */ * bgra[1] * bgra[1] +
          13933UL /* 0.2126 * 65536 */ * bgra[2] * bgra[2] ) >> 16;

    /*
     * Final transparency can be determined as follows.
     *
     * - If alpha is zero, we want 0.
     * - If alpha is zero and luminosity is zero, we want 255.
     * - If alpha is zero and luminosity is one, we want 0.
     *
     * So the formula is a * (1 - l) = a - l * a.
     *
     * We still need to undo premultiplication by dividing l by a*a.
     *
     */

    return (FT_Byte)( a - l / a );
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

    FT_Byte*  s;
    FT_Byte*  t;


    if ( !library )
      return FT_THROW( Invalid_Library_Handle );

    if ( !source || !target )
      return FT_THROW( Invalid_Argument );

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
        FT_Int    pad, old_target_pitch, target_pitch;
        FT_ULong  old_size;


        old_target_pitch = target->pitch;
        if ( old_target_pitch < 0 )
          old_target_pitch = -old_target_pitch;

        old_size = target->rows * (FT_UInt)old_target_pitch;

        target->pixel_mode = FT_PIXEL_MODE_GRAY;
        target->rows       = source->rows;
        target->width      = source->width;

        pad = 0;
        if ( alignment > 0 )
        {
          pad = (FT_Int)source->width % alignment;
          if ( pad != 0 )
            pad = alignment - pad;
        }

        target_pitch = (FT_Int)source->width + pad;

        if ( target_pitch > 0                                               &&
             (FT_ULong)target->rows > FT_ULONG_MAX / (FT_ULong)target_pitch )
          return FT_THROW( Invalid_Argument );

        if ( FT_QREALLOC( target->buffer,
                          old_size, target->rows * (FT_UInt)target_pitch ) )
          return error;

        target->pitch = target->pitch < 0 ? -target_pitch : target_pitch;
      }
      break;

    default:
      error = FT_THROW( Invalid_Argument );
    }

    s = source->buffer;
    t = target->buffer;

    /* take care of bitmap flow */
    if ( source->pitch < 0 )
      s -= source->pitch * (FT_Int)( source->rows - 1 );
    if ( target->pitch < 0 )
      t -= target->pitch * (FT_Int)( target->rows - 1 );

    switch ( source->pixel_mode )
    {
    case FT_PIXEL_MODE_MONO:
      {
        FT_UInt  i;


        target->num_grays = 2;

        for ( i = source->rows; i > 0; i-- )
        {
          FT_Byte*  ss = s;
          FT_Byte*  tt = t;
          FT_UInt   j;


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
        FT_UInt  width = source->width;
        FT_UInt  i;


        target->num_grays = 256;

        for ( i = source->rows; i > 0; i-- )
        {
          FT_ARRAY_COPY( t, s, width );

          s += source->pitch;
          t += target->pitch;
        }
      }
      break;


    case FT_PIXEL_MODE_GRAY2:
      {
        FT_UInt  i;


        target->num_grays = 4;

        for ( i = source->rows; i > 0; i-- )
        {
          FT_Byte*  ss = s;
          FT_Byte*  tt = t;
          FT_UInt   j;


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
        FT_UInt  i;


        target->num_grays = 16;

        for ( i = source->rows; i > 0; i-- )
        {
          FT_Byte*  ss = s;
          FT_Byte*  tt = t;
          FT_UInt   j;


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
        FT_UInt  i;


        target->num_grays = 256;

        for ( i = source->rows; i > 0; i-- )
        {
          FT_Byte*  ss = s;
          FT_Byte*  tt = t;
          FT_UInt   j;


          for ( j = source->width; j > 0; j-- )
          {
            tt[0] = ft_gray_for_premultiplied_srgb_bgra( ss );

            ss += 4;
            tt += 1;
          }

          s += source->pitch;
          t += target->pitch;
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
  FT_Bitmap_Blend( FT_Library        library,
                   const FT_Bitmap*  source_,
                   const FT_Vector   source_offset_,
                   FT_Bitmap*        target,
                   FT_Vector        *atarget_offset,
                   FT_Color          color )
  {
    FT_Error   error = FT_Err_Ok;
    FT_Memory  memory;

    FT_Bitmap         source_bitmap;
    const FT_Bitmap*  source;

    FT_Vector  source_offset;
    FT_Vector  target_offset;

    FT_Bool  free_source_bitmap          = 0;
    FT_Bool  free_target_bitmap_on_error = 0;

    FT_Pos  source_llx, source_lly, source_urx, source_ury;
    FT_Pos  target_llx, target_lly, target_urx, target_ury;
    FT_Pos  final_llx, final_lly, final_urx, final_ury;

    unsigned int  final_rows, final_width;
    long          x, y;


    if ( !library || !target || !source_ || !atarget_offset )
      return FT_THROW( Invalid_Argument );

    memory = library->memory;

    if ( !( target->pixel_mode == FT_PIXEL_MODE_NONE     ||
            ( target->pixel_mode == FT_PIXEL_MODE_BGRA &&
              target->buffer                           ) ) )
      return FT_THROW( Invalid_Argument );

    if ( source_->pixel_mode == FT_PIXEL_MODE_NONE )
      return FT_Err_Ok;               /* nothing to do */

    /* pitches must have the same sign */
    if ( target->pixel_mode == FT_PIXEL_MODE_BGRA &&
         ( source_->pitch ^ target->pitch ) < 0   )
      return FT_THROW( Invalid_Argument );

    if ( !( source_->width && source_->rows ) )
      return FT_Err_Ok;               /* nothing to do */

    /* assure integer pixel offsets */
    source_offset.x = FT_PIX_FLOOR( source_offset_.x );
    source_offset.y = FT_PIX_FLOOR( source_offset_.y );
    target_offset.x = FT_PIX_FLOOR( atarget_offset->x );
    target_offset.y = FT_PIX_FLOOR( atarget_offset->y );

    /* get source bitmap dimensions */
    source_llx = source_offset.x;
    if ( FT_LONG_MIN + (FT_Pos)( source_->rows << 6 ) + 64 > source_offset.y )
    {
      FT_TRACE5((
        "FT_Bitmap_Blend: y coordinate overflow in source bitmap\n" ));
      return FT_THROW( Invalid_Argument );
    }
    source_lly = source_offset.y - ( source_->rows << 6 );

    if ( FT_LONG_MAX - (FT_Pos)( source_->width << 6 ) - 64 < source_llx )
    {
      FT_TRACE5((
        "FT_Bitmap_Blend: x coordinate overflow in source bitmap\n" ));
      return FT_THROW( Invalid_Argument );
    }
    source_urx = source_llx + ( source_->width << 6 );
    source_ury = source_offset.y;

    /* get target bitmap dimensions */
    if ( target->width && target->rows )
    {
      target_llx = target_offset.x;
      if ( FT_LONG_MIN + (FT_Pos)( target->rows << 6 ) > target_offset.y )
      {
        FT_TRACE5((
          "FT_Bitmap_Blend: y coordinate overflow in target bitmap\n" ));
        return FT_THROW( Invalid_Argument );
      }
      target_lly = target_offset.y - ( target->rows << 6 );

      if ( FT_LONG_MAX - (FT_Pos)( target->width << 6 ) < target_llx )
      {
        FT_TRACE5((
          "FT_Bitmap_Blend: x coordinate overflow in target bitmap\n" ));
        return FT_THROW( Invalid_Argument );
      }
      target_urx = target_llx + ( target->width << 6 );
      target_ury = target_offset.y;
    }
    else
    {
      target_llx = FT_LONG_MAX;
      target_lly = FT_LONG_MAX;
      target_urx = FT_LONG_MIN;
      target_ury = FT_LONG_MIN;
    }

    /* compute final bitmap dimensions */
    final_llx = FT_MIN( source_llx, target_llx );
    final_lly = FT_MIN( source_lly, target_lly );
    final_urx = FT_MAX( source_urx, target_urx );
    final_ury = FT_MAX( source_ury, target_ury );

    final_width = ( final_urx - final_llx ) >> 6;
    final_rows  = ( final_ury - final_lly ) >> 6;

#ifdef FT_DEBUG_LEVEL_TRACE
    FT_TRACE5(( "FT_Bitmap_Blend:\n"
                "  source bitmap: (%ld, %ld) -- (%ld, %ld); %d x %d\n",
      source_llx / 64, source_lly / 64,
      source_urx / 64, source_ury / 64,
      source_->width, source_->rows ));

    if ( target->width && target->rows )
      FT_TRACE5(( "  target bitmap: (%ld, %ld) -- (%ld, %ld); %d x %d\n",
        target_llx / 64, target_lly / 64,
        target_urx / 64, target_ury / 64,
        target->width, target->rows ));
    else
      FT_TRACE5(( "  target bitmap: empty\n" ));

    if ( final_width && final_rows )
      FT_TRACE5(( "  final bitmap: (%ld, %ld) -- (%ld, %ld); %d x %d\n",
        final_llx / 64, final_lly / 64,
        final_urx / 64, final_ury / 64,
        final_width, final_rows ));
    else
      FT_TRACE5(( "  final bitmap: empty\n" ));
#endif /* FT_DEBUG_LEVEL_TRACE */

    if ( !( final_width && final_rows ) )
      return FT_Err_Ok;               /* nothing to do */

    /* for blending, set offset vector of final bitmap */
    /* temporarily to (0,0)                            */
    source_llx -= final_llx;
    source_lly -= final_lly;

    if ( target->width && target->rows )
    {
      target_llx -= final_llx;
      target_lly -= final_lly;
    }

    /* set up target bitmap */
    if ( target->pixel_mode == FT_PIXEL_MODE_NONE )
    {
      /* create new empty bitmap */
      target->width      = final_width;
      target->rows       = final_rows;
      target->pixel_mode = FT_PIXEL_MODE_BGRA;
      target->pitch      = (int)final_width * 4;
      target->num_grays  = 256;

      if ( FT_LONG_MAX / target->pitch < (int)target->rows )
      {
        FT_TRACE5(( "FT_Blend_Bitmap: target bitmap too large (%d x %d)\n",
                     final_width, final_rows ));
        return FT_THROW( Invalid_Argument );
      }

      if ( FT_ALLOC( target->buffer, target->pitch * (int)target->rows ) )
        return error;

      free_target_bitmap_on_error = 1;
    }
    else if ( target->width != final_width ||
              target->rows  != final_rows  )
    {
      /* adjust old bitmap to enlarged size */
      int  pitch, new_pitch;

      unsigned char*  buffer = NULL;


      pitch = target->pitch;

      if ( pitch < 0 )
        pitch = -pitch;

      new_pitch = (int)final_width * 4;

      if ( FT_LONG_MAX / new_pitch < (int)final_rows )
      {
        FT_TRACE5(( "FT_Blend_Bitmap: target bitmap too large (%d x %d)\n",
                     final_width, final_rows ));
        return FT_THROW( Invalid_Argument );
      }

      /* TODO: provide an in-buffer solution for large bitmaps */
      /*       to avoid allocation of a new buffer             */
      if ( FT_ALLOC( buffer, new_pitch * (int)final_rows ) )
        goto Error;

      /* copy data to new buffer */
      x = target_llx >> 6;
      y = target_lly >> 6;

      /* the bitmap flow is from top to bottom, */
      /* but y is measured from bottom to top   */
      if ( target->pitch < 0 )
      {
        /* XXX */
      }
      else
      {
        unsigned char*  p =
          target->buffer;
        unsigned char*  q =
          buffer +
          ( final_rows - y - target->rows ) * new_pitch +
          x * 4;
        unsigned char*  limit_p =
          p + pitch * (int)target->rows;


        while ( p < limit_p )
        {
          FT_MEM_COPY( q, p, pitch );

          p += pitch;
          q += new_pitch;
        }
      }

      FT_FREE( target->buffer );

      target->width = final_width;
      target->rows  = final_rows;

      if ( target->pitch < 0 )
        target->pitch = -new_pitch;
      else
        target->pitch = new_pitch;

      target->buffer = buffer;
    }

    /* adjust source bitmap if necessary */
    if ( source_->pixel_mode != FT_PIXEL_MODE_GRAY )
    {
      FT_Bitmap_Init( &source_bitmap );
      error = FT_Bitmap_Convert( library, source_, &source_bitmap, 1 );
      if ( error )
        goto Error;

      source             = &source_bitmap;
      free_source_bitmap = 1;
    }
    else
      source = source_;

    /* do blending; the code below returns pre-multiplied channels, */
    /* similar to what FreeType gets from `CBDT' tables             */
    x = source_llx >> 6;
    y = source_lly >> 6;

    /* the bitmap flow is from top to bottom, */
    /* but y is measured from bottom to top   */
    if ( target->pitch < 0 )
    {
      /* XXX */
    }
    else
    {
      unsigned char*  p =
        source->buffer;
      unsigned char*  q =
        target->buffer +
        ( target->rows - y - source->rows ) * target->pitch +
        x * 4;
      unsigned char*  limit_p =
        p + source->pitch * (int)source->rows;


      while ( p < limit_p )
      {
        unsigned char*  r       = p;
        unsigned char*  s       = q;
        unsigned char*  limit_r = r + source->width;


        while ( r < limit_r )
        {
          int  aa = *r++;
          int  fa = color.alpha * aa / 255;

          int  fb = color.blue * fa / 255;
          int  fg = color.green * fa / 255;
          int  fr = color.red * fa / 255;

          int  ba2 = 255 - fa;

          int  bb = s[0];
          int  bg = s[1];
          int  br = s[2];
          int  ba = s[3];


          *s++ = (unsigned char)( bb * ba2 / 255 + fb );
          *s++ = (unsigned char)( bg * ba2 / 255 + fg );
          *s++ = (unsigned char)( br * ba2 / 255 + fr );
          *s++ = (unsigned char)( ba * ba2 / 255 + fa );
        }

        p += source->pitch;
        q += target->pitch;
      }
    }

    atarget_offset->x = final_llx;
    atarget_offset->y = final_lly + ( final_rows << 6 );

  Error:
    if ( error && free_target_bitmap_on_error )
      FT_Bitmap_Done( library, target );

    if ( free_source_bitmap )
      FT_Bitmap_Done( library, &source_bitmap );

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


      FT_Bitmap_Init( &bitmap );
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
