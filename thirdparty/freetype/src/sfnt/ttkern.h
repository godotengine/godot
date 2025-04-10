/****************************************************************************
 *
 * ttkern.h
 *
 *   Load the basic TrueType kerning table.  This doesn't handle
 *   kerning data within the GPOS table at the moment.
 *
 * Copyright (C) 1996-2024 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#ifndef TTKERN_H_
#define TTKERN_H_


#include <freetype/internal/ftstream.h>
#include <freetype/internal/tttypes.h>


FT_BEGIN_HEADER


  FT_LOCAL( FT_Error  )
  tt_face_load_kern( TT_Face    face,
                     FT_Stream  stream );

  FT_LOCAL( void )
  tt_face_done_kern( TT_Face  face );

  FT_LOCAL( FT_Int )
  tt_face_get_kerning( TT_Face     face,
                       FT_UInt     left_glyph,
                       FT_UInt     right_glyph );

#define TT_FACE_HAS_KERNING( face )  ( (face)->kern_avail_bits != 0 )


FT_END_HEADER

#endif /* TTKERN_H_ */


/* END */
