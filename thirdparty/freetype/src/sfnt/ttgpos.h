/****************************************************************************
 *
 * ttgpos.c
 *
 *   Load the TrueType GPOS table.  The only GPOS layout feature this
 *   currently supports is kerning, from x advances in the pair adjustment
 *   layout feature.
 *
 * Copyright (C) 2024 by
 * David Saltzman
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 */


#ifndef TTGPOS_H_
#define TTGPOS_H_


#include <freetype/internal/ftstream.h>
#include <freetype/internal/tttypes.h>


FT_BEGIN_HEADER


#ifdef TT_CONFIG_OPTION_GPOS_KERNING

  FT_LOCAL( FT_Error  )
  tt_face_load_gpos( TT_Face    face,
                     FT_Stream  stream );

  FT_LOCAL( void )
  tt_face_done_gpos( TT_Face  face );

  FT_LOCAL( FT_Int )
  tt_face_get_gpos_kerning( TT_Face  face,
                            FT_UInt  left_glyph,
                            FT_UInt  right_glyph );

#endif /* TT_CONFIG_OPTION_GPOS_KERNING */


FT_END_HEADER

#endif /* TTGPOS_H_ */


/* END */
