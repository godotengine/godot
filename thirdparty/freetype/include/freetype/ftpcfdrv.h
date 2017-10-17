/***************************************************************************/
/*                                                                         */
/*  ftpcfdrv.h                                                             */
/*                                                                         */
/*    FreeType API for controlling the PCF driver (specification only).    */
/*                                                                         */
/*  Copyright 2017 by                                                      */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


#ifndef FTPCFDRV_H_
#define FTPCFDRV_H_

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
   *   pcf_driver
   *
   * @title:
   *   The PCF driver
   *
   * @abstract:
   *   Controlling the PCF driver module.
   *
   * @description:
   *   While FreeType's PCF driver doesn't expose API functions by itself,
   *   it is possible to control its behaviour with @FT_Property_Set and
   *   @FT_Property_Get.  Right now, there is a single property
   *   `no-long-family-names' available if FreeType is compiled with
   *   PCF_CONFIG_OPTION_LONG_FAMILY_NAMES.
   *
   *   The PCF driver's module name is `pcf'.
   *
   */


  /**************************************************************************
   *
   * @property:
   *   no-long-family-names
   *
   * @description:
   *   If PCF_CONFIG_OPTION_LONG_FAMILY_NAMES is active while compiling
   *   FreeType, the PCF driver constructs long family names.
   *
   *   There are many PCF fonts just called `Fixed' which look completely
   *   different, and which have nothing to do with each other.  When
   *   selecting `Fixed' in KDE or Gnome one gets results that appear rather
   *   random, the style changes often if one changes the size and one
   *   cannot select some fonts at all.  The improve this situation, the PCF
   *   module prepends the foundry name (plus a space) to the family name. 
   *   It also checks whether there are `wide' characters; all put together,
   *   family names like `Sony Fixed' or `Misc Fixed Wide' are constructed.
   *
   *   If `no-long-family-names' is set, this feature gets switched off.
   *
   *   {
   *     FT_Library  library;
   *     FT_Bool     no_long_family_names = TRUE;
   *
   *
   *     FT_Init_FreeType( &library );
   *
   *     FT_Property_Set( library, "pcf",
   *                               "no-long-family-names",
   *                               &no_long_family_names );
   *   }
   *
   * @note:
   *   This property can be used with @FT_Property_Get also.
   *
   *   This property can be set via the `FREETYPE_PROPERTIES' environment
   *   variable (using values 1 and 0 for `on' and `off', respectively).
   *
   */


FT_END_HEADER


#endif /* FTPCFDRV_H_ */


/* END */
