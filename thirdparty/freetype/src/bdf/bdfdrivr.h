/*  bdfdrivr.h

    FreeType font driver for bdf fonts

  Copyright (C) 2001, 2002, 2003, 2004 by
  Francesco Zappa Nardelli

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/


#ifndef BDFDRIVR_H_
#define BDFDRIVR_H_

#include <ft2build.h>
#include FT_INTERNAL_DRIVER_H

#include "bdf.h"


FT_BEGIN_HEADER

#ifdef FT_CONFIG_OPTION_PIC
#error "this module does not support PIC yet"
#endif


  typedef struct  BDF_encoding_el_
  {
    FT_Long    enc;
    FT_UShort  glyph;

  } BDF_encoding_el;


  typedef struct  BDF_FaceRec_
  {
    FT_FaceRec        root;

    char*             charset_encoding;
    char*             charset_registry;

    bdf_font_t*       bdffont;

    BDF_encoding_el*  en_table;

    FT_CharMap        charmap_handle;
    FT_CharMapRec     charmap;  /* a single charmap per face */

    FT_UInt           default_glyph;

  } BDF_FaceRec, *BDF_Face;


  FT_EXPORT_VAR( const FT_Driver_ClassRec )  bdf_driver_class;


FT_END_HEADER


#endif /* BDFDRIVR_H_ */


/* END */
