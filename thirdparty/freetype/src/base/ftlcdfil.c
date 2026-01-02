/****************************************************************************
 *
 * ftlcdfil.c
 *
 *   FreeType API for color filtering of subpixel bitmap glyphs (body).
 *
 * Copyright (C) 2006-2025 by
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

#include <freetype/ftlcdfil.h>
#include <freetype/ftimage.h>
#include <freetype/internal/ftobjs.h>


#ifdef FT_CONFIG_OPTION_SUBPIXEL_RENDERING

/* define USE_LEGACY to implement the legacy filter */
#define  USE_LEGACY

#define FT_SHIFTCLAMP( x )  ( x >>= 8, (FT_Byte)( x > 255 ? 255 : x ) )


  /* add padding according to filter weights */
  FT_BASE_DEF( void )
  ft_lcd_padding( FT_BBox*        cbox,
                  FT_GlyphSlot    slot,
                  FT_Render_Mode  mode )
  {
    FT_Byte*                 lcd_weights;
    FT_Bitmap_LcdFilterFunc  lcd_filter_func;


    /* Per-face LCD filtering takes priority if set up. */
    if ( slot->face && slot->face->internal->lcd_filter_func )
    {
      lcd_weights     = slot->face->internal->lcd_weights;
      lcd_filter_func = slot->face->internal->lcd_filter_func;
    }
    else
    {
      lcd_weights     = slot->library->lcd_weights;
      lcd_filter_func = slot->library->lcd_filter_func;
    }

    if ( lcd_filter_func == ft_lcd_filter_fir )
    {
      if ( mode == FT_RENDER_MODE_LCD )
      {
        cbox->xMin -= lcd_weights[0] ? 43 :
                      lcd_weights[1] ? 22 : 0;
        cbox->xMax += lcd_weights[4] ? 43 :
                      lcd_weights[3] ? 22 : 0;
      }
      else if ( mode == FT_RENDER_MODE_LCD_V )
      {
        cbox->yMin -= lcd_weights[0] ? 43 :
                      lcd_weights[1] ? 22 : 0;
        cbox->yMax += lcd_weights[4] ? 43 :
                      lcd_weights[3] ? 22 : 0;
      }
    }
  }


  /* FIR filter used by the default and light filters */
  FT_BASE_DEF( void )
  ft_lcd_filter_fir( FT_Bitmap*           bitmap,
                     FT_LcdFiveTapFilter  weights )
  {
    FT_UInt   width  = (FT_UInt)bitmap->width;
    FT_UInt   height = (FT_UInt)bitmap->rows;
    FT_Int    pitch  = bitmap->pitch;
    FT_Byte*  origin = bitmap->buffer;
    FT_Byte   mode   = bitmap->pixel_mode;


    /* take care of bitmap flow */
    if ( pitch > 0 && height > 0 )
      origin += pitch * (FT_Int)( height - 1 );

    /* horizontal in-place FIR filter */
    if ( mode == FT_PIXEL_MODE_LCD && width >= 2 )
    {
      FT_Byte*  line = origin;


      /* `fir' must be at least 32 bit wide, since the sum of */
      /* the values in `weights' can exceed 0xFF              */

      for ( ; height > 0; height--, line -= pitch )
      {
        FT_UInt  fir[5];
        FT_UInt  val, xx;


        val    = line[0];
        fir[2] = weights[2] * val;
        fir[3] = weights[3] * val;
        fir[4] = weights[4] * val;

        val    = line[1];
        fir[1] = fir[2] + weights[1] * val;
        fir[2] = fir[3] + weights[2] * val;
        fir[3] = fir[4] + weights[3] * val;
        fir[4] =          weights[4] * val;

        for ( xx = 2; xx < width; xx++ )
        {
          val    = line[xx];
          fir[0] = fir[1] + weights[0] * val;
          fir[1] = fir[2] + weights[1] * val;
          fir[2] = fir[3] + weights[2] * val;
          fir[3] = fir[4] + weights[3] * val;
          fir[4] =          weights[4] * val;

          line[xx - 2] = FT_SHIFTCLAMP( fir[0] );
        }

        line[xx - 2] = FT_SHIFTCLAMP( fir[1] );
        line[xx - 1] = FT_SHIFTCLAMP( fir[2] );
      }
    }

    /* vertical in-place FIR filter */
    else if ( mode == FT_PIXEL_MODE_LCD_V && height >= 2 )
    {
      FT_Byte*  column = origin;


      for ( ; width > 0; width--, column++ )
      {
        FT_Byte*  col = column;
        FT_UInt   fir[5];
        FT_UInt   val, yy;


        val    = col[0];
        fir[2] = weights[2] * val;
        fir[3] = weights[3] * val;
        fir[4] = weights[4] * val;
        col   -= pitch;

        val    = col[0];
        fir[1] = fir[2] + weights[1] * val;
        fir[2] = fir[3] + weights[2] * val;
        fir[3] = fir[4] + weights[3] * val;
        fir[4] =          weights[4] * val;
        col   -= pitch;

        for ( yy = 2; yy < height; yy++, col -= pitch )
        {
          val    = col[0];
          fir[0] = fir[1] + weights[0] * val;
          fir[1] = fir[2] + weights[1] * val;
          fir[2] = fir[3] + weights[2] * val;
          fir[3] = fir[4] + weights[3] * val;
          fir[4] =          weights[4] * val;

          col[pitch * 2]  = FT_SHIFTCLAMP( fir[0] );
        }

        col[pitch * 2]  = FT_SHIFTCLAMP( fir[1] );
        col[pitch]      = FT_SHIFTCLAMP( fir[2] );
      }
    }
  }


#ifdef USE_LEGACY

  /* intra-pixel filter used by the legacy filter */
  static void
  _ft_lcd_filter_legacy( FT_Bitmap*      bitmap,
                         FT_Byte*        weights )
  {
    FT_UInt   width  = (FT_UInt)bitmap->width;
    FT_UInt   height = (FT_UInt)bitmap->rows;
    FT_Int    pitch  = bitmap->pitch;
    FT_Byte*  origin = bitmap->buffer;
    FT_Byte   mode   = bitmap->pixel_mode;

    static const unsigned int  filters[3][3] =
    {
      { 65538 * 9/13, 65538 * 1/6, 65538 * 1/13 },
      { 65538 * 3/13, 65538 * 4/6, 65538 * 3/13 },
      { 65538 * 1/13, 65538 * 1/6, 65538 * 9/13 }
    };

    FT_UNUSED( weights );


    /* take care of bitmap flow */
    if ( pitch > 0 && height > 0 )
      origin += pitch * (FT_Int)( height - 1 );

    /* horizontal in-place intra-pixel filter */
    if ( mode == FT_PIXEL_MODE_LCD && width >= 3 )
    {
      FT_Byte*  line = origin;


      for ( ; height > 0; height--, line -= pitch )
      {
        FT_UInt  xx;


        for ( xx = 0; xx < width; xx += 3 )
        {
          FT_UInt  r, g, b;
          FT_UInt  p;


          p  = line[xx];
          r  = filters[0][0] * p;
          g  = filters[0][1] * p;
          b  = filters[0][2] * p;

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
    else if ( mode == FT_PIXEL_MODE_LCD_V && height >= 3 )
    {
      FT_Byte*  column = origin;


      for ( ; width > 0; width--, column++ )
      {
        FT_Byte*  col = column - 2 * pitch;


        for ( ; height > 0; height -= 3, col -= 3 * pitch )
        {
          FT_UInt  r, g, b;
          FT_UInt  p;


          p  = col[0];
          r  = filters[0][0] * p;
          g  = filters[0][1] * p;
          b  = filters[0][2] * p;

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
          col[pitch * 2] = (FT_Byte)( b / 65536 );
        }
      }
    }
  }

#endif /* USE_LEGACY */


  /* documentation in ftlcdfil.h */

  FT_EXPORT_DEF( FT_Error )
  FT_Library_SetLcdFilterWeights( FT_Library      library,
                                  unsigned char  *weights )
  {
    if ( !library )
      return FT_THROW( Invalid_Library_Handle );

    if ( !weights )
      return FT_THROW( Invalid_Argument );

    ft_memcpy( library->lcd_weights, weights, FT_LCD_FILTER_FIVE_TAPS );
    library->lcd_filter_func = ft_lcd_filter_fir;

    return FT_Err_Ok;
  }


  /* documentation in ftlcdfil.h */

  FT_EXPORT_DEF( FT_Error )
  FT_Library_SetLcdFilter( FT_Library    library,
                           FT_LcdFilter  filter )
  {
    static const FT_LcdFiveTapFilter  default_weights =
                   { 0x08, 0x4d, 0x56, 0x4d, 0x08 };
    static const FT_LcdFiveTapFilter  light_weights =
                   { 0x00, 0x55, 0x56, 0x55, 0x00 };


    if ( !library )
      return FT_THROW( Invalid_Library_Handle );

    switch ( filter )
    {
    case FT_LCD_FILTER_NONE:
      library->lcd_filter_func = NULL;
      break;

    case FT_LCD_FILTER_DEFAULT:
      ft_memcpy( library->lcd_weights,
                 default_weights,
                 FT_LCD_FILTER_FIVE_TAPS );
      library->lcd_filter_func = ft_lcd_filter_fir;
      break;

    case FT_LCD_FILTER_LIGHT:
      ft_memcpy( library->lcd_weights,
                 light_weights,
                 FT_LCD_FILTER_FIVE_TAPS );
      library->lcd_filter_func = ft_lcd_filter_fir;
      break;

#ifdef USE_LEGACY

    case FT_LCD_FILTER_LEGACY:
    case FT_LCD_FILTER_LEGACY1:
      library->lcd_filter_func = _ft_lcd_filter_legacy;
      break;

#endif

    default:
      return FT_THROW( Invalid_Argument );
    }

    return FT_Err_Ok;
  }


  FT_EXPORT_DEF( FT_Error )
  FT_Library_SetLcdGeometry( FT_Library  library,
                             FT_Vector   sub[3] )
  {
    FT_UNUSED( library );
    FT_UNUSED( sub );

    return FT_THROW( Unimplemented_Feature );
  }

#else /* !FT_CONFIG_OPTION_SUBPIXEL_RENDERING */

  /* add padding to accommodate outline shifts */
  FT_BASE_DEF( void )
  ft_lcd_padding( FT_BBox*        cbox,
                  FT_GlyphSlot    slot,
                  FT_Render_Mode  mode )
  {
    FT_Vector*  sub = slot->library->lcd_geometry;

    if ( mode == FT_RENDER_MODE_LCD )
    {
      cbox->xMin -= FT_MAX( FT_MAX( sub[0].x, sub[1].x ), sub[2].x );
      cbox->xMax -= FT_MIN( FT_MIN( sub[0].x, sub[1].x ), sub[2].x );
      cbox->yMin -= FT_MAX( FT_MAX( sub[0].y, sub[1].y ), sub[2].y );
      cbox->yMax -= FT_MIN( FT_MIN( sub[0].y, sub[1].y ), sub[2].y );
    }
    else if ( mode == FT_RENDER_MODE_LCD_V )
    {
      cbox->xMin -= FT_MAX( FT_MAX( sub[0].y, sub[1].y ), sub[2].y );
      cbox->xMax -= FT_MIN( FT_MIN( sub[0].y, sub[1].y ), sub[2].y );
      cbox->yMin += FT_MIN( FT_MIN( sub[0].x, sub[1].x ), sub[2].x );
      cbox->yMax += FT_MAX( FT_MAX( sub[0].x, sub[1].x ), sub[2].x );
    }
  }


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


  /* documentation in ftlcdfil.h */

  FT_EXPORT_DEF( FT_Error )
  FT_Library_SetLcdGeometry( FT_Library  library,
                             FT_Vector   sub[3] )
  {
    if ( !library )
      return FT_THROW( Invalid_Library_Handle );

    if ( !sub )
      return FT_THROW( Invalid_Argument );

    ft_memcpy( library->lcd_geometry, sub, 3 * sizeof( FT_Vector ) );

    return FT_Err_Ok;
  }

#endif /* !FT_CONFIG_OPTION_SUBPIXEL_RENDERING */


/* END */
