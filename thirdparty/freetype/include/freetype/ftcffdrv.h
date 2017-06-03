/***************************************************************************/
/*                                                                         */
/*  ftcffdrv.h                                                             */
/*                                                                         */
/*    FreeType API for controlling the CFF driver (specification only).    */
/*                                                                         */
/*  Copyright 2013-2016 by                                                 */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


#ifndef FTCFFDRV_H_
#define FTCFFDRV_H_

#include <ft2build.h>
#include FT_FREETYPE_H

#ifdef FREETYPE_H
#error "freetype.h of FreeType 1 has been loaded!"
#error "Please fix the directory search order for header files"
#error "so that freetype.h of FreeType 2 is found first."
#endif


FT_BEGIN_HEADER


  /**************************************************************************
   *
   * @section:
   *   cff_driver
   *
   * @title:
   *   The CFF driver
   *
   * @abstract:
   *   Controlling the CFF driver module.
   *
   * @description:
   *   While FreeType's CFF driver doesn't expose API functions by itself,
   *   it is possible to control its behaviour with @FT_Property_Set and
   *   @FT_Property_Get.  The list below gives the available properties
   *   together with the necessary macros and structures.
   *
   *   The CFF driver's module name is `cff'.
   *
   *   *Hinting* *and* *antialiasing* *principles* *of* *the* *new* *engine*
   *
   *   The rasterizer is positioning horizontal features (e.g., ascender
   *   height & x-height, or crossbars) on the pixel grid and minimizing the
   *   amount of antialiasing applied to them, while placing vertical
   *   features (vertical stems) on the pixel grid without hinting, thus
   *   representing the stem position and weight accurately.  Sometimes the
   *   vertical stems may be only partially black.  In this context,
   *   `antialiasing' means that stems are not positioned exactly on pixel
   *   borders, causing a fuzzy appearance.
   *
   *   There are two principles behind this approach.
   *
   *   1) No hinting in the horizontal direction: Unlike `superhinted'
   *   TrueType, which changes glyph widths to accommodate regular
   *   inter-glyph spacing, Adobe's approach is `faithful to the design' in
   *   representing both the glyph width and the inter-glyph spacing
   *   designed for the font.  This makes the screen display as close as it
   *   can be to the result one would get with infinite resolution, while
   *   preserving what is considered the key characteristics of each glyph.
   *   Note that the distances between unhinted and grid-fitted positions at
   *   small sizes are comparable to kerning values and thus would be
   *   noticeable (and distracting) while reading if hinting were applied.
   *
   *   One of the reasons to not hint horizontally is antialiasing for LCD
   *   screens: The pixel geometry of modern displays supplies three
   *   vertical sub-pixels as the eye moves horizontally across each visible
   *   pixel.  On devices where we can be certain this characteristic is
   *   present a rasterizer can take advantage of the sub-pixels to add
   *   increments of weight.  In Western writing systems this turns out to
   *   be the more critical direction anyway; the weights and spacing of
   *   vertical stems (see above) are central to Armenian, Cyrillic, Greek,
   *   and Latin type designs.  Even when the rasterizer uses greyscale
   *   antialiasing instead of color (a necessary compromise when one
   *   doesn't know the screen characteristics), the unhinted vertical
   *   features preserve the design's weight and spacing much better than
   *   aliased type would.
   *
   *   2) Alignment in the vertical direction: Weights and spacing along the
   *   y~axis are less critical; what is much more important is the visual
   *   alignment of related features (like cap-height and x-height).  The
   *   sense of alignment for these is enhanced by the sharpness of grid-fit
   *   edges, while the cruder vertical resolution (full pixels instead of
   *   1/3 pixels) is less of a problem.
   *
   *   On the technical side, horizontal alignment zones for ascender,
   *   x-height, and other important height values (traditionally called
   *   `blue zones') as defined in the font are positioned independently,
   *   each being rounded to the nearest pixel edge, taking care of
   *   overshoot suppression at small sizes, stem darkening, and scaling.
   *
   *   Hstems (this is, hint values defined in the font to help align
   *   horizontal features) that fall within a blue zone are said to be
   *   `captured' and are aligned to that zone.  Uncaptured stems are moved
   *   in one of four ways, top edge up or down, bottom edge up or down.
   *   Unless there are conflicting hstems, the smallest movement is taken
   *   to minimize distortion.
   *
   * @order:
   *   hinting-engine[cff]
   *   no-stem-darkening[cff]
   *   darkening-parameters[cff]
   *
   */


  /**************************************************************************
   *
   * @property:
   *   hinting-engine[cff]
   *
   * @description:
   *   Thanks to Adobe, which contributed a new hinting (and parsing)
   *   engine, an application can select between `freetype' and `adobe' if
   *   compiled with CFF_CONFIG_OPTION_OLD_ENGINE.  If this configuration
   *   macro isn't defined, `hinting-engine' does nothing.
   *
   *   The default engine is `freetype' if CFF_CONFIG_OPTION_OLD_ENGINE is
   *   defined, and `adobe' otherwise.
   *
   *   The following example code demonstrates how to select Adobe's hinting
   *   engine (omitting the error handling).
   *
   *   {
   *     FT_Library  library;
   *     FT_UInt     hinting_engine = FT_CFF_HINTING_ADOBE;
   *
   *
   *     FT_Init_FreeType( &library );
   *
   *     FT_Property_Set( library, "cff",
   *                               "hinting-engine", &hinting_engine );
   *   }
   *
   * @note:
   *   This property can be used with @FT_Property_Get also.
   *
   */


  /**************************************************************************
   *
   * @enum:
   *   FT_CFF_HINTING_XXX
   *
   * @description:
   *   A list of constants used for the @hinting-engine[cff] property to
   *   select the hinting engine for CFF fonts.
   *
   * @values:
   *   FT_CFF_HINTING_FREETYPE ::
   *     Use the old FreeType hinting engine.
   *
   *   FT_CFF_HINTING_ADOBE ::
   *     Use the hinting engine contributed by Adobe.
   *
   */
#define FT_CFF_HINTING_FREETYPE  0
#define FT_CFF_HINTING_ADOBE     1


  /**************************************************************************
   *
   * @property:
   *   no-stem-darkening[cff]
   *
   * @description:
   *   By default, the Adobe CFF engine darkens stems at smaller sizes,
   *   regardless of hinting, to enhance contrast.  This feature requires
   *   a rendering system with proper gamma correction.  Setting this
   *   property, stem darkening gets switched off.
   *
   *   Note that stem darkening is never applied if @FT_LOAD_NO_SCALE is set.
   *
   *   {
   *     FT_Library  library;
   *     FT_Bool     no_stem_darkening = TRUE;
   *
   *
   *     FT_Init_FreeType( &library );
   *
   *     FT_Property_Set( library, "cff",
   *                               "no-stem-darkening", &no_stem_darkening );
   *   }
   *
   * @note:
   *   This property can be used with @FT_Property_Get also.
   *
   */


  /**************************************************************************
   *
   * @property:
   *   darkening-parameters[cff]
   *
   * @description:
   *   By default, the Adobe CFF engine darkens stems as follows (if the
   *   `no-stem-darkening' property isn't set):
   *
   *   {
   *     stem width <= 0.5px:   darkening amount = 0.4px
   *     stem width  = 1px:     darkening amount = 0.275px
   *     stem width  = 1.667px: darkening amount = 0.275px
   *     stem width >= 2.333px: darkening amount = 0px
   *   }
   *
   *   and piecewise linear in-between.  At configuration time, these four
   *   control points can be set with the macro
   *   `CFF_CONFIG_OPTION_DARKENING_PARAMETERS'.  At runtime, the control
   *   points can be changed using the `darkening-parameters' property, as
   *   the following example demonstrates.
   *
   *   {
   *     FT_Library  library;
   *     FT_Int      darken_params[8] = {  500, 300,   // x1, y1
   *                                      1000, 200,   // x2, y2
   *                                      1500, 100,   // x3, y3
   *                                      2000,   0 }; // x4, y4
   *
   *
   *     FT_Init_FreeType( &library );
   *
   *     FT_Property_Set( library, "cff",
   *                               "darkening-parameters", darken_params );
   *   }
   *
   *   The x~values give the stem width, and the y~values the darkening
   *   amount.  The unit is 1000th of pixels.  All coordinate values must be
   *   positive; the x~values must be monotonically increasing; the
   *   y~values must be monotonically decreasing and smaller than or
   *   equal to 500 (corresponding to half a pixel); the slope of each
   *   linear piece must be shallower than -1 (e.g., -.4).
   *
   * @note:
   *   This property can be used with @FT_Property_Get also.
   *
   */

  /* */


FT_END_HEADER


#endif /* FTCFFDRV_H_ */


/* END */
