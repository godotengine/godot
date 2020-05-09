/****************************************************************************
 *
 * ftfntfmt.h
 *
 *   Support functions for font formats.
 *
 * Copyright (C) 2002-2020 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#ifndef FTFNTFMT_H_
#define FTFNTFMT_H_

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
   *  font_formats
   *
   * @title:
   *  Font Formats
   *
   * @abstract:
   *  Getting the font format.
   *
   * @description:
   *  The single function in this section can be used to get the font format.
   *  Note that this information is not needed normally; however, there are
   *  special cases (like in PDF devices) where it is important to
   *  differentiate, in spite of FreeType's uniform API.
   *
   */


  /**************************************************************************
   *
   * @function:
   *  FT_Get_Font_Format
   *
   * @description:
   *  Return a string describing the format of a given face.  Possible values
   *  are 'TrueType', 'Type~1', 'BDF', 'PCF', 'Type~42', 'CID~Type~1', 'CFF',
   *  'PFR', and 'Windows~FNT'.
   *
   *  The return value is suitable to be used as an X11 FONT_PROPERTY.
   *
   * @input:
   *  face ::
   *    Input face handle.
   *
   * @return:
   *  Font format string.  `NULL` in case of error.
   *
   * @note:
   *  A deprecated name for the same function is `FT_Get_X11_Font_Format`.
   */
  FT_EXPORT( const char* )
  FT_Get_Font_Format( FT_Face  face );


  /* deprecated */
  FT_EXPORT( const char* )
  FT_Get_X11_Font_Format( FT_Face  face );


  /* */


FT_END_HEADER

#endif /* FTFNTFMT_H_ */


/* END */
