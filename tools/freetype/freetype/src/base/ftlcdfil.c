/***************************************************************************/
/*                                                                         */
/*  ftlcdfil.c                                                             */
/*                                                                         */
/*    FreeType API for color filtering of subpixel bitmap glyphs (body).   */
/*                                                                         */
/*  Copyright 2006, 2008-2010, 2013 by                                     */
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

#include FT_LCD_FILTER_H
#include FT_IMAGE_H
#include FT_INTERNAL_OBJECTS_H


#ifdef FT_CONFIG_OPTION_SUBPIXEL_RENDERING

/* define USE_LEGACY to implement the legacy filter */
#define  USE_LEGACY

  /* FIR filter used by the default and light filters */
  static void
  _ft_lcd_filter_fir( FT_Bitmap*      bitmap,
                      FT_Render_Mode  mode,
                      FT_Library      library )
  {
    FT_Byte*  weights = library->lcd_weights;
    FT_UInt   width   = (FT_UInt)bitmap->width;
    FT_UInt   height  = (FT_UInt)bitmap->rows;


    /* horizontal in-place FIR filter */
    if ( mode == FT_RENDER_MODE_LCD && width >= 4 )
    {
      FT_Byte*  line = bitmap->buffer;


      for ( ; height > 0; height--, line += bitmap->pitch )
      {
        FT_UInt  fir[5];
        FT_UInt  val1, xx;


        val1   = line[0];
        fir[0] = weights[2] * val1;
        fir[1] = weights[3] * val1;
        fir[2] = weights[4] * val1;
        fir[3] = 0;
        fir[4] = 0;

        val1    = line[1];
        fir[0] += weights[1] * val1;
        fir[1] += weights[2] * val1;
        fir[2] += weights[3] * val1;
        fir[3] += weights[4] * val1;

        for ( xx = 2; xx < width; xx++ )
        {
          FT_UInt  val, pix;


          val    = line[xx];
          pix    = fir[0] + weights[0] * val;
          fir[0] = fir[1] + weights[1] * val;
          fir[1] = fir[2] + weights[2] * val;
          fir[2] = fir[3] + weights[3] * val;
          fir[3] =          weights[4] * val;

          pix        >>= 8;
          pix         |= -( pix >> 8 );
          line[xx - 2] = (FT_Byte)pix;
        }

        {
          FT_UInt  pix;


          pix          = fir[0] >> 8;
          pix         |= -( pix >> 8 );
          line[xx - 2] = (FT_Byte)pix;

          pix          = fir[1] >> 8;
          pix         |= -( pix >> 8 );
          line[xx - 1] = (FT_Byte)pix;
        }
      }
    }

    /* vertical in-place FIR filter */
    else if ( mode == FT_RENDER_MODE_LCD_V && height >= 4 )
    {
      FT_Byte*  column = bitmap->buffer;
      FT_Int    pitch  = bitmap->pitch;


      for ( ; width > 0; width--, column++ )
      {
        FT_Byte*  col = column;
        FT_UInt   fir[5];
        FT_UInt   val1, yy;


        val1   = col[0];
        fir[0] = weights[2] * val1;
        fir[1] = weights[3] * val1;
        fir[2] = weights[4] * val1;
        fir[3] = 0;
        fir[4] = 0;
        col   += pitch;

        val1    = col[0];
        fir[0] += weights[1] * val1;
        fir[1] += weights[2] * val1;
        fir[2] += weights[3] * val1;
        fir[3] += weights[4] * val1;
        col    += pitch;

        for ( yy = 2; yy < height; yy++ )
        {
          FT_UInt  val, pix;


          val    = col[0];
          pix    = fir[0] + weights[0] * val;
          fir[0] = fir[1] + weights[1] * val;
          fir[1] = fir[2] + weights[2] * val;
          fir[2] = fir[3] + weights[3] * val;
          fir[3] =          weights[4] * val;

          pix           >>= 8;
          pix            |= -( pix >> 8 );
          col[-2 * pitch] = (FT_Byte)pix;
          col            += pitch;
        }

        {
          FT_UInt  pix;


          pix             = fir[0] >> 8;
          pix            |= -( pix >> 8 );
          col[-2 * pitch] = (FT_Byte)pix;

          pix         = fir[1] >> 8;
          pix        |= -( pix >> 8 );
          col[-pitch] = (FT_Byte)pix;
        }
      }
    }
  }


#ifdef USE_LEGACY

  /* intra-pixel filter used by the legacy filter */
  static void
  _ft_lcd_filter_legacy( FT_Bitmap*      bitmap,
                         FT_Render_Mode  mode,
                         FT_Library      library )
  {
    FT_UInt  width  = (FT_UInt)bitmap->width;
    FT_UInt  height = (FT_UInt)bitmap->rows;
    FT_Int   pitch  = bitmap->pitch;

    static const int  filters[3][3] =
    {
      { 65538 * 9/13, 65538 * 1/6, 65538 * 1/13 },
      { 65538 * 3/13, 65538 * 4/6, 65538 * 3/13 },
      { 65538 * 1/13, 65538 * 1/6, 65538 * 9/13 }
    };

    FT_UNUSED( library );


    /* horizontal in-place intra-pixel filter */
    if ( mode == FT_RENDER_MODE_LCD && width >= 3 )
    {
      FT_Byte*  line = bitmap->buffer;


      for ( ; height > 0; height--, line += pitch )
      {
        FT_UInt  xx;


        for ( xx = 0; xx < width; xx += 3 )
        {
          FT_UInt  r = 0;
          FT_UInt  g = 0;
          FT_UInt  b = 0;
          FT_UInt  p;


          p  = line[xx];
          r += filters[0][0] * p;
          g += filters[0][1] * p;
          b += filters[0][2] * p;

          p  = line[xx + 1];
          r += filters[1][0] * p;
          g += filters[1][1] * p;
          b += filters[1][2] * p;

          p  = line[xx + 2];
          r += filters[2][0] * p;
          g += filters[2][1] * p;
          b += filters[2][2] * p;

          line[xx]     = (FT_Byte)( r / 65536 );
          line[xx + 1] = (FT_Byte)( g / 65536 );
          line[xx + 2] = (FT_Byte)( b / 65536 );
        }
      }
    }
    else if ( mode == FT_RENDER_MODE_LCD_V && height >= 3 )
    {
      FT_Byte*  column = bitmap->buffer;


      for ( ; width > 0; width--, column++ )
      {
        FT_Byte*  col     = column;
        FT_Byte*  col_end = col + height * pitch;


        for ( ; col < col_end; col += 3 * pitch )
        {
          FT_UInt  r = 0;
          FT_UInt  g = 0;
          FT_UInt  b = 0;
          FT_UInt  p;


          p  = col[0];
          r += filters[0][0] * p;
          g += filters[0][1] * p;
          b += filters[0][2] * p;

          p  = col[pitch];
          r += filters[1][0] * p;
          g += filters[1][1] * p;
          b += filters[1][2] * p;

          p  = col[pitch * 2];
          r += filters[2][0] * p;
          g += filters[2][1] * p;
          b += filters[2][2] * p;

          col[0]         = (FT_Byte)( r / 65536 );
          col[pitch]     = (FT_Byte)( g / 65536 );
          col[2 * pitch] = (FT_Byte)( b / 65536 );
        }
      }
    }
  }

#endif /* USE_LEGACY */


  FT_EXPORT_DEF( FT_Error )
  FT_Library_SetLcdFilterWeights( FT_Library      library,
                                  unsigned char  *weights )
  {
    if ( !library || !weights )
      return FT_THROW( Invalid_Argument );

    ft_memcpy( library->lcd_weights, weights, 5 );

    return FT_Err_Ok;
  }


  FT_EXPORT_DEF( FT_Error )
  FT_Library_SetLcdFilter( FT_Library    library,
                           FT_LcdFilter  filter )
  {
    static const FT_Byte  light_filter[5] =
                            { 0x00, 0x55, 0x56, 0x55, 0x00 };
    /* the values here sum up to a value larger than 256, */
    /* providing a cheap gamma correction                 */
    static const FT_Byte  default_filter[5] =
                            { 0x10, 0x40, 0x70, 0x40, 0x10 };


    if ( !library )
      return FT_THROW( Invalid_Argument );

    switch ( filter )
    {
    case FT_LCD_FILTER_NONE:
      library->lcd_filter_func = NULL;
      library->lcd_extra       = 0;
      break;

    case FT_LCD_FILTER_DEFAULT:
#if defined( FT_FORCE_LEGACY_LCD_FILTER )

      library->lcd_filter_func = _ft_lcd_filter_legacy;
      library->lcd_extra       = 0;

#elif defined( FT_FORCE_LIGHT_LCD_FILTER )

      ft_memcpy( library->lcd_weights, light_filter, 5 );
      library->lcd_filter_func = _ft_lcd_filter_fir;
      library->lcd_extra       = 2;

#else

      ft_memcpy( library->lcd_weights, default_filter, 5 );
      library->lcd_filter_func = _ft_lcd_filter_fir;
      library->lcd_extra       = 2;

#endif

      break;

    case FT_LCD_FILTER_LIGHT:
      ft_memcpy( library->lcd_weights, light_filter, 5 );
      library->lcd_filter_func = _ft_lcd_filter_fir;
      library->lcd_extra       = 2;
      break;

#ifdef USE_LEGACY

    case FT_LCD_FILTER_LEGACY:
      library->lcd_filter_func = _ft_lcd_filter_legacy;
      library->lcd_extra       = 0;
      break;

#endif

    default:
      return FT_THROW( Invalid_Argument );
    }

    library->lcd_filter = filter;

    return FT_Err_Ok;
  }

#else /* !FT_CONFIG_OPTION_SUBPIXEL_RENDERING */

  FT_EXPORT_DEF( FT_Error )
  FT_Library_SetLcdFilterWeights( FT_Library      library,
                                  unsigned char  *weights )
  {
    FT_UNUSED( library );
    FT_UNUSED( weights );

    return FT_THROW( Unimplemented_Feature );
  }


  FT_EXPORT_DEF( FT_Error )
  FT_Library_SetLcdFilter( FT_Library    library,
                           FT_LcdFilter  filter )
  {
    FT_UNUSED( library );
    FT_UNUSED( filter );

    return FT_THROW( Unimplemented_Feature );
  }

#endif /* !FT_CONFIG_OPTION_SUBPIXEL_RENDERING */


/* END */
