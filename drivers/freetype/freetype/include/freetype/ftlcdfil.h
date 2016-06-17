/***************************************************************************/
/*                                                                         */
/*  ftlcdfil.h                                                             */
/*                                                                         */
/*    FreeType API for color filtering of subpixel bitmap glyphs           */
/*    (specification).                                                     */
/*                                                                         */
/*  Copyright 2006, 2007, 2008, 2010 by                                    */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


#ifndef __FT_LCD_FILTER_H__
#define __FT_LCD_FILTER_H__

#include <ft2build.h>
#include FT_FREETYPE_H

#ifdef FREETYPE_H
#error "freetype.h of FreeType 1 has been loaded!"
#error "Please fix the directory search order for header files"
#error "so that freetype.h of FreeType 2 is found first."
#endif


FT_BEGIN_HEADER

  /***************************************************************************
   *
   * @section:
   *   lcd_filtering
   *
   * @title:
   *   LCD Filtering
   *
   * @abstract:
   *   Reduce color fringes of LCD-optimized bitmaps.
   *
   * @description:
   *   The @FT_Library_SetLcdFilter API can be used to specify a low-pass
   *   filter which is then applied to LCD-optimized bitmaps generated
   *   through @FT_Render_Glyph.  This is useful to reduce color fringes
   *   which would occur with unfiltered rendering.
   *
   *   Note that no filter is active by default, and that this function is
   *   *not* implemented in default builds of the library.  You need to
   *   #define FT_CONFIG_OPTION_SUBPIXEL_RENDERING in your `ftoption.h' file
   *   in order to activate it.
   *
   *   FreeType generates alpha coverage maps, which are linear by nature.
   *   For instance, the value 0x80 in bitmap representation means that
   *   (within numerical precision) 0x80/0xff fraction of that pixel is
   *   covered by the glyph's outline.  The blending function for placing
   *   text over a background is
   *
   *   {
   *     dst = alpha * src + (1 - alpha) * dst    ,
   *   }
   *
   *   which is known as OVER.  However, when calculating the output of the
   *   OVER operator, the source colors should first be transformed to a
   *   linear color space, then alpha blended in that space, and transformed
   *   back to the output color space.
   *
   *   When linear light blending is used, the default FIR5 filtering
   *   weights (as given by FT_LCD_FILTER_DEFAULT) are no longer optimal, as
   *   they have been designed for black on white rendering while lacking
   *   gamma correction.  To preserve color neutrality, weights for a FIR5
   *   filter should be chosen according to two free parameters `a' and `c',
   *   and the FIR weights should be
   *
   *   {
   *     [a - c, a + c, 2 * a, a + c, a - c]    .
   *   }
   *
   *   This formula generates equal weights for all the color primaries
   *   across the filter kernel, which makes it colorless.  One suggested
   *   set of weights is
   *
   *   {
   *     [0x10, 0x50, 0x60, 0x50, 0x10]    ,
   *   }
   *
   *   where `a' has value 0x30 and `b' value 0x20.  The weights in filter
   *   may have a sum larger than 0x100, which increases coloration slightly
   *   but also improves contrast.
   */


  /****************************************************************************
   *
   * @enum:
   *   FT_LcdFilter
   *
   * @description:
   *   A list of values to identify various types of LCD filters.
   *
   * @values:
   *   FT_LCD_FILTER_NONE ::
   *     Do not perform filtering.  When used with subpixel rendering, this
   *     results in sometimes severe color fringes.
   *
   *   FT_LCD_FILTER_DEFAULT ::
   *     The default filter reduces color fringes considerably, at the cost
   *     of a slight blurriness in the output.
   *
   *   FT_LCD_FILTER_LIGHT ::
   *     The light filter is a variant that produces less blurriness at the
   *     cost of slightly more color fringes than the default one.  It might
   *     be better, depending on taste, your monitor, or your personal vision.
   *
   *   FT_LCD_FILTER_LEGACY ::
   *     This filter corresponds to the original libXft color filter.  It
   *     provides high contrast output but can exhibit really bad color
   *     fringes if glyphs are not extremely well hinted to the pixel grid.
   *     In other words, it only works well if the TrueType bytecode
   *     interpreter is enabled *and* high-quality hinted fonts are used.
   *
   *     This filter is only provided for comparison purposes, and might be
   *     disabled or stay unsupported in the future.
   *
   * @since:
   *   2.3.0
   */
  typedef enum  FT_LcdFilter_
  {
    FT_LCD_FILTER_NONE    = 0,
    FT_LCD_FILTER_DEFAULT = 1,
    FT_LCD_FILTER_LIGHT   = 2,
    FT_LCD_FILTER_LEGACY  = 16,

    FT_LCD_FILTER_MAX   /* do not remove */

  } FT_LcdFilter;


  /**************************************************************************
   *
   * @func:
   *   FT_Library_SetLcdFilter
   *
   * @description:
   *   This function is used to apply color filtering to LCD decimated
   *   bitmaps, like the ones used when calling @FT_Render_Glyph with
   *   @FT_RENDER_MODE_LCD or @FT_RENDER_MODE_LCD_V.
   *
   * @input:
   *   library ::
   *     A handle to the target library instance.
   *
   *   filter ::
   *     The filter type.
   *
   *     You can use @FT_LCD_FILTER_NONE here to disable this feature, or
   *     @FT_LCD_FILTER_DEFAULT to use a default filter that should work
   *     well on most LCD screens.
   *
   * @return:
   *   FreeType error code.  0~means success.
   *
   * @note:
   *   This feature is always disabled by default.  Clients must make an
   *   explicit call to this function with a `filter' value other than
   *   @FT_LCD_FILTER_NONE in order to enable it.
   *
   *   Due to *PATENTS* covering subpixel rendering, this function doesn't
   *   do anything except returning `FT_Err_Unimplemented_Feature' if the
   *   configuration macro FT_CONFIG_OPTION_SUBPIXEL_RENDERING is not
   *   defined in your build of the library, which should correspond to all
   *   default builds of FreeType.
   *
   *   The filter affects glyph bitmaps rendered through @FT_Render_Glyph,
   *   @FT_Outline_Get_Bitmap, @FT_Load_Glyph, and @FT_Load_Char.
   *
   *   It does _not_ affect the output of @FT_Outline_Render and
   *   @FT_Outline_Get_Bitmap.
   *
   *   If this feature is activated, the dimensions of LCD glyph bitmaps are
   *   either larger or taller than the dimensions of the corresponding
   *   outline with regards to the pixel grid.  For example, for
   *   @FT_RENDER_MODE_LCD, the filter adds up to 3~pixels to the left, and
   *   up to 3~pixels to the right.
   *
   *   The bitmap offset values are adjusted correctly, so clients shouldn't
   *   need to modify their layout and glyph positioning code when enabling
   *   the filter.
   *
   * @since:
   *   2.3.0
   */
  FT_EXPORT( FT_Error )
  FT_Library_SetLcdFilter( FT_Library    library,
                           FT_LcdFilter  filter );


  /**************************************************************************
   *
   * @func:
   *   FT_Library_SetLcdFilterWeights
   *
   * @description:
   *   Use this function to override the filter weights selected by
   *   @FT_Library_SetLcdFilter.  By default, FreeType uses the quintuple
   *   (0x00, 0x55, 0x56, 0x55, 0x00) for FT_LCD_FILTER_LIGHT, and (0x10,
   *   0x40, 0x70, 0x40, 0x10) for FT_LCD_FILTER_DEFAULT and
   *   FT_LCD_FILTER_LEGACY.
   *
   * @input:
   *   library ::
   *     A handle to the target library instance.
   *
   *   weights ::
   *     A pointer to an array; the function copies the first five bytes and
   *     uses them to specify the filter weights.
   *
   * @return:
   *   FreeType error code.  0~means success.
   *
   * @note:
   *   Due to *PATENTS* covering subpixel rendering, this function doesn't
   *   do anything except returning `FT_Err_Unimplemented_Feature' if the
   *   configuration macro FT_CONFIG_OPTION_SUBPIXEL_RENDERING is not
   *   defined in your build of the library, which should correspond to all
   *   default builds of FreeType.
   *
   *   This function must be called after @FT_Library_SetLcdFilter to have
   *   any effect.
   *
   * @since:
   *   2.4.0
   */
  FT_EXPORT( FT_Error )
  FT_Library_SetLcdFilterWeights( FT_Library      library,
                                  unsigned char  *weights );

  /* */


FT_END_HEADER

#endif /* __FT_LCD_FILTER_H__ */


/* END */
