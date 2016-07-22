/***************************************************************************/
/*                                                                         */
/*  ftttdrv.h                                                              */
/*                                                                         */
/*    FreeType API for controlling the TrueType driver                     */
/*    (specification only).                                                */
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


#ifndef FTTTDRV_H_
#define FTTTDRV_H_

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
   *   tt_driver
   *
   * @title:
   *   The TrueType driver
   *
   * @abstract:
   *   Controlling the TrueType driver module.
   *
   * @description:
   *   While FreeType's TrueType driver doesn't expose API functions by
   *   itself, it is possible to control its behaviour with @FT_Property_Set
   *   and @FT_Property_Get.  The following lists the available properties
   *   together with the necessary macros and structures.
   *
   *   The TrueType driver's module name is `truetype'.
   *
   *   We start with a list of definitions, kindly provided by Greg
   *   Hitchcock.
   *
   *   _Bi-Level_ _Rendering_
   *
   *   Monochromatic rendering, exclusively used in the early days of
   *   TrueType by both Apple and Microsoft.  Microsoft's GDI interface
   *   supported hinting of the right-side bearing point, such that the
   *   advance width could be non-linear.  Most often this was done to
   *   achieve some level of glyph symmetry.  To enable reasonable
   *   performance (e.g., not having to run hinting on all glyphs just to
   *   get the widths) there was a bit in the head table indicating if the
   *   side bearing was hinted, and additional tables, `hdmx' and `LTSH', to
   *   cache hinting widths across multiple sizes and device aspect ratios.
   *
   *   _Font_ _Smoothing_
   *
   *   Microsoft's GDI implementation of anti-aliasing.  Not traditional
   *   anti-aliasing as the outlines were hinted before the sampling.  The
   *   widths matched the bi-level rendering.
   *
   *   _ClearType_ _Rendering_
   *
   *   Technique that uses physical subpixels to improve rendering on LCD
   *   (and other) displays.  Because of the higher resolution, many methods
   *   of improving symmetry in glyphs through hinting the right-side
   *   bearing were no longer necessary.  This lead to what GDI calls
   *   `natural widths' ClearType, see
   *   http://www.beatstamm.com/typography/RTRCh4.htm#Sec21.  Since hinting
   *   has extra resolution, most non-linearity went away, but it is still
   *   possible for hints to change the advance widths in this mode.
   *
   *   _ClearType_ _Compatible_ _Widths_
   *
   *   One of the earliest challenges with ClearType was allowing the
   *   implementation in GDI to be selected without requiring all UI and
   *   documents to reflow.  To address this, a compatible method of
   *   rendering ClearType was added where the font hints are executed once
   *   to determine the width in bi-level rendering, and then re-run in
   *   ClearType, with the difference in widths being absorbed in the font
   *   hints for ClearType (mostly in the white space of hints); see
   *   http://www.beatstamm.com/typography/RTRCh4.htm#Sec20.  Somewhat by
   *   definition, compatible width ClearType allows for non-linear widths,
   *   but only when the bi-level version has non-linear widths.
   *
   *   _ClearType_ _Subpixel_ _Positioning_
   *
   *   One of the nice benefits of ClearType is the ability to more crisply
   *   display fractional widths; unfortunately, the GDI model of integer
   *   bitmaps did not support this.  However, the WPF and Direct Write
   *   frameworks do support fractional widths.  DWrite calls this `natural
   *   mode', not to be confused with GDI's `natural widths'.  Subpixel
   *   positioning, in the current implementation of Direct Write,
   *   unfortunately does not support hinted advance widths, see
   *   http://www.beatstamm.com/typography/RTRCh4.htm#Sec22.  Note that the
   *   TrueType interpreter fully allows the advance width to be adjusted in
   *   this mode, just the DWrite client will ignore those changes.
   *
   *   _ClearType_ _Backwards_ _Compatibility_
   *
   *   This is a set of exceptions made in the TrueType interpreter to
   *   minimize hinting techniques that were problematic with the extra
   *   resolution of ClearType; see
   *   http://www.beatstamm.com/typography/RTRCh4.htm#Sec1 and
   *   http://www.microsoft.com/typography/cleartype/truetypecleartype.aspx.
   *   This technique is not to be confused with ClearType compatible
   *   widths.  ClearType backwards compatibility has no direct impact on
   *   changing advance widths, but there might be an indirect impact on
   *   disabling some deltas.  This could be worked around in backwards
   *   compatibility mode.
   *
   *   _Native_ _ClearType_ _Mode_
   *
   *   (Not to be confused with `natural widths'.)  This mode removes all
   *   the exceptions in the TrueType interpreter when running with
   *   ClearType.  Any issues on widths would still apply, though.
   *
   */


  /**************************************************************************
   *
   * @property:
   *   interpreter-version
   *
   * @description:

   *   Currently, three versions are available, two representing the
   *   bytecode interpreter with subpixel hinting support (old `Infinality'
   *   code and new stripped-down and higher performance `minimal' code) and
   *   one without, respectively.  The default is subpixel support if
   *   TT_CONFIG_OPTION_SUBPIXEL_HINTING is defined, and no subpixel support
   *   otherwise (since it isn't available then).
   *
   *   If subpixel hinting is on, many TrueType bytecode instructions behave
   *   differently compared to B/W or grayscale rendering (except if `native
   *   ClearType' is selected by the font).  Microsoft's main idea is to
   *   render at a much increased horizontal resolution, then sampling down
   *   the created output to subpixel precision.  However, many older fonts
   *   are not suited to this and must be specially taken care of by
   *   applying (hardcoded) tweaks in Microsoft's interpreter.
   *
   *   Details on subpixel hinting and some of the necessary tweaks can be
   *   found in Greg Hitchcock's whitepaper at
   *   `http://www.microsoft.com/typography/cleartype/truetypecleartype.aspx'. 
   *   Note that FreeType currently doesn't really `subpixel hint' (6x1, 6x2,
   *   or 6x5 supersampling) like discussed in the paper.  Depending on the
   *   chosen interpreter, it simply ignores instructions on vertical stems
   *   to arrive at very similar results.
   *
   *   The following example code demonstrates how to deactivate subpixel
   *   hinting (omitting the error handling).
   *
   *   {
   *     FT_Library  library;
   *     FT_Face     face;
   *     FT_UInt     interpreter_version = TT_INTERPRETER_VERSION_35;
   *
   *
   *     FT_Init_FreeType( &library );
   *
   *     FT_Property_Set( library, "truetype",
   *                               "interpreter-version",
   *                               &interpreter_version );
   *   }
   *
   * @note:
   *   This property can be used with @FT_Property_Get also.
   *
   */


  /**************************************************************************
   *
   * @enum:
   *   TT_INTERPRETER_VERSION_XXX
   *
   * @description:
   *   A list of constants used for the @interpreter-version property to
   *   select the hinting engine for Truetype fonts.
   *
   *   The numeric value in the constant names represents the version
   *   number as returned by the `GETINFO' bytecode instruction.
   *
   * @values:
   *   TT_INTERPRETER_VERSION_35 ::
   *     Version~35 corresponds to MS rasterizer v.1.7 as used e.g. in
   *     Windows~98; only grayscale and B/W rasterizing is supported.
   *
   *   TT_INTERPRETER_VERSION_38 ::
   *     Version~38 corresponds to MS rasterizer v.1.9; it is roughly
   *     equivalent to the hinting provided by DirectWrite ClearType (as can
   *     be found, for example, in the Internet Explorer~9 running on
   *     Windows~7).  It is used in FreeType to select the `Infinality'
   *     subpixel hinting code.  The code may be removed in a future
   *     version.
   *
   *   TT_INTERPRETER_VERSION_40 ::
   *     Version~40 corresponds to MS rasterizer v.2.1; it is roughly
   *     equivalent to the hinting provided by DirectWrite ClearType (as can
   *     be found, for example, in Microsoft's Edge Browser on Windows~10). 
   *     It is used in FreeType to select the `minimal' subpixel hinting
   *     code, a stripped-down and higher performance version of the
   *     `Infinality' code.
   *
   * @note:
   *   This property controls the behaviour of the bytecode interpreter
   *   and thus how outlines get hinted.  It does *not* control how glyph
   *   get rasterized!  In particular, it does not control subpixel color
   *   filtering.
   *
   *   If FreeType has not been compiled with the configuration option
   *   FT_CONFIG_OPTION_SUBPIXEL_HINTING, selecting version~38 or~40 causes
   *   an `FT_Err_Unimplemented_Feature' error.
   *
   *   Depending on the graphics framework, Microsoft uses different
   *   bytecode and rendering engines.  As a consequence, the version
   *   numbers returned by a call to the `GETINFO' bytecode instruction are
   *   more convoluted than desired.
   *
   *   Here are two tables that try to shed some light on the possible
   *   values for the MS rasterizer engine, together with the additional
   *   features introduced by it.
   *
   *   {
   *     GETINFO framework               version feature
   *     -------------------------------------------------------------------
   *         3   GDI (Win 3.1),            v1.0  16-bit, first version
   *             TrueImage
   *        33   GDI (Win NT 3.1),         v1.5  32-bit
   *             HP Laserjet
   *        34   GDI (Win 95)              v1.6  font smoothing,
   *                                             new SCANTYPE opcode
   *        35   GDI (Win 98/2000)         v1.7  (UN)SCALED_COMPONENT_OFFSET
   *                                               bits in composite glyphs
   *        36   MGDI (Win CE 2)           v1.6+ classic ClearType
   *        37   GDI (XP and later),       v1.8  ClearType
   *             GDI+ old (before Vista)
   *        38   GDI+ old (Vista, Win 7),  v1.9  subpixel ClearType,
   *             WPF                             Y-direction ClearType,
   *                                             additional error checking
   *        39   DWrite (before Win 8)     v2.0  subpixel ClearType flags
   *                                               in GETINFO opcode,
   *                                             bug fixes
   *        40   GDI+ (after Win 7),       v2.1  Y-direction ClearType flag
   *             DWrite (Win 8)                    in GETINFO opcode,
   *                                             Gray ClearType
   *   }
   *
   *   The `version' field gives a rough orientation only, since some
   *   applications provided certain features much earlier (as an example,
   *   Microsoft Reader used subpixel and Y-direction ClearType already in
   *   Windows 2000).  Similarly, updates to a given framework might include
   *   improved hinting support.
   *
   *   {
   *      version   sampling          rendering        comment
   *               x        y       x           y
   *     --------------------------------------------------------------
   *       v1.0   normal  normal  B/W           B/W    bi-level
   *       v1.6   high    high    gray          gray   grayscale
   *       v1.8   high    normal  color-filter  B/W    (GDI) ClearType
   *       v1.9   high    high    color-filter  gray   Color ClearType
   *       v2.1   high    normal  gray          B/W    Gray ClearType
   *       v2.1   high    high    gray          gray   Gray ClearType
   *   }
   *
   *   Color and Gray ClearType are the two available variants of
   *   `Y-direction ClearType', meaning grayscale rasterization along the
   *   Y-direction; the name used in the TrueType specification for this
   *   feature is `symmetric smoothing'.  `Classic ClearType' is the
   *   original algorithm used before introducing a modified version in
   *   Win~XP.  Another name for v1.6's grayscale rendering is `font
   *   smoothing', and `Color ClearType' is sometimes also called `DWrite
   *   ClearType'.  To differentiate between today's Color ClearType and the
   *   earlier ClearType variant with B/W rendering along the vertical axis,
   *   the latter is sometimes called `GDI ClearType'.
   *
   *   `Normal' and `high' sampling describe the (virtual) resolution to
   *   access the rasterized outline after the hinting process.  `Normal'
   *   means 1 sample per grid line (i.e., B/W).  In the current Microsoft
   *   implementation, `high' means an extra virtual resolution of 16x16 (or
   *   16x1) grid lines per pixel for bytecode instructions like `MIRP'.
   *   After hinting, these 16 grid lines are mapped to 6x5 (or 6x1) grid
   *   lines for color filtering if Color ClearType is activated.
   *
   *   Note that `Gray ClearType' is essentially the same as v1.6's
   *   grayscale rendering.  However, the GETINFO instruction handles it
   *   differently: v1.6 returns bit~12 (hinting for grayscale), while v2.1
   *   returns bits~13 (hinting for ClearType), 18 (symmetrical smoothing),
   *   and~19 (Gray ClearType).  Also, this mode respects bits 2 and~3 for
   *   the version~1 gasp table exclusively (like Color ClearType), while
   *   v1.6 only respects the values of version~0 (bits 0 and~1).
   *
   *   Keep in mind that the features of the above interpreter versions
   *   might not map exactly to FreeType features or behavior because it is
   *   a fundamentally different library with different internals.
   *
   */
#define TT_INTERPRETER_VERSION_35  35
#define TT_INTERPRETER_VERSION_38  38
#define TT_INTERPRETER_VERSION_40  40

 /* */


FT_END_HEADER


#endif /* FTTTDRV_H_ */


/* END */
