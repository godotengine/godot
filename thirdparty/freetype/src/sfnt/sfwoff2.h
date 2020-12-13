/****************************************************************************
 *
 * sfwoff2.h
 *
 *   WOFFF2 format management (specification).
 *
 * Copyright (C) 2019-2020 by
 * Nikhil Ramakrishnan, David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#ifndef SFWOFF2_H_
#define SFWOFF2_H_


#include <freetype/internal/sfnt.h>
#include <freetype/internal/ftobjs.h>


FT_BEGIN_HEADER


  /* Leave the first byte open to store `flag_byte'. */
#define WOFF2_FLAGS_TRANSFORM   1 << 8

#define WOFF2_SFNT_HEADER_SIZE  12
#define WOFF2_SFNT_ENTRY_SIZE   16

  /* Suggested maximum size for output. */
#define WOFF2_DEFAULT_MAX_SIZE  30 * 1024 * 1024

  /* 98% of Google Fonts have no glyph above 5k bytes. */
#define WOFF2_DEFAULT_GLYPH_BUF  5120

  /* Composite glyph flags.                                      */
  /* See `CompositeGlyph.java' in `sfntly' for full definitions. */
#define FLAG_ARG_1_AND_2_ARE_WORDS     1 << 0
#define FLAG_WE_HAVE_A_SCALE           1 << 3
#define FLAG_MORE_COMPONENTS           1 << 5
#define FLAG_WE_HAVE_AN_X_AND_Y_SCALE  1 << 6
#define FLAG_WE_HAVE_A_TWO_BY_TWO      1 << 7
#define FLAG_WE_HAVE_INSTRUCTIONS      1 << 8

  /* Simple glyph flags */
#define GLYF_ON_CURVE        1 << 0
#define GLYF_X_SHORT         1 << 1
#define GLYF_Y_SHORT         1 << 2
#define GLYF_REPEAT          1 << 3
#define GLYF_THIS_X_IS_SAME  1 << 4
#define GLYF_THIS_Y_IS_SAME  1 << 5

  /* Other constants */
#define CONTOUR_OFFSET_END_POINT  10


  FT_LOCAL( FT_Error )
  woff2_open_font( FT_Stream  stream,
                   TT_Face    face,
                   FT_Int*    face_index,
                   FT_Long*   num_faces );


FT_END_HEADER

#endif /* SFWOFF2_H_ */


/* END */
