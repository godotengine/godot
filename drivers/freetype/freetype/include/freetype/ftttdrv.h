/***************************************************************************/
/*                                                                         */
/*  ftttdrv.h                                                              */
/*                                                                         */
/*    FreeType API for controlling the TrueType driver                     */
/*    (specification only).                                                */
/*                                                                         */
/*  Copyright 2013 by                                                      */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


#ifndef __FTTTDRV_H__
#define __FTTTDRV_H__

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
   */


  /**************************************************************************
   *
   * @property:
   *   interpreter-version
   *
   * @description:
   *   Currently, two versions are available which represent the bytecode
   *   interpreter with and without subpixel hinting support,
   *   respectively.  The default is subpixel support if
   *   TT_CONFIG_OPTION_SUBPIXEL_HINTING is defined, and no subpixel
   *   support otherwise (since it isn't available then).
   *
   *   If subpixel hinting is on, many TrueType bytecode instructions
   *   behave differently compared to B/W or grayscale rendering.  The
   *   main idea is to render at a much increased horizontal resolution,
   *   then sampling down the created output to subpixel precision.
   *   However, many older fonts are not suited to this and must be
   *   specially taken care of by applying (hardcoded) font-specific
   *   tweaks.
   *
   *   Details on subpixel hinting and some of the necessary tweaks can be
   *   found in Greg Hitchcock's whitepaper at
   *   `http://www.microsoft.com/typography/cleartype/truetypecleartype.aspx'.
   *
   *   The following example code demonstrates how to activate subpixel
   *   hinting (omitting the error handling).
   *
   *   {
   *     FT_Library  library;
   *     FT_Face     face;
   *     FT_UInt     interpreter_version = TT_INTERPRETER_VERSION_38;
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
   *     equivalent to the hinting provided by DirectWrite ClearType (as
   *     can be found, for example, in the Internet Explorer~9 running on
   *     Windows~7).
   *
   * @note:
   *   This property controls the behaviour of the bytecode interpreter
   *   and thus how outlines get hinted.  It does *not* control how glyph
   *   get rasterized!  In particular, it does not control subpixel color
   *   filtering.
   *
   *   If FreeType has not been compiled with configuration option
   *   FT_CONFIG_OPTION_SUBPIXEL_HINTING, selecting version~38 causes an
   *   `FT_Err_Unimplemented_Feature' error.
   *
   */
#define TT_INTERPRETER_VERSION_35  35
#define TT_INTERPRETER_VERSION_38  38


 /* */

FT_END_HEADER


#endif /* __FTTTDRV_H__ */


/* END */
