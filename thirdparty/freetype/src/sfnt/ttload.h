/****************************************************************************
 *
 * ttload.h
 *
 *   Load the basic TrueType tables, i.e., tables that can be either in
 *   TTF or OTF fonts (specification).
 *
 * Copyright (C) 1996-2023 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#ifndef TTLOAD_H_
#define TTLOAD_H_


#include <freetype/internal/ftstream.h>
#include <freetype/internal/tttypes.h>


FT_BEGIN_HEADER


  FT_LOCAL( TT_Table  )
  tt_face_lookup_table( TT_Face   face,
                        FT_ULong  tag );

  FT_LOCAL( FT_Error )
  tt_face_goto_table( TT_Face    face,
                      FT_ULong   tag,
                      FT_Stream  stream,
                      FT_ULong*  length );


  FT_LOCAL( FT_Error )
  tt_face_load_font_dir( TT_Face    face,
                         FT_Stream  stream );


  FT_LOCAL( FT_Error )
  tt_face_load_any( TT_Face    face,
                    FT_ULong   tag,
                    FT_Long    offset,
                    FT_Byte*   buffer,
                    FT_ULong*  length );


  FT_LOCAL( FT_Error )
  tt_face_load_head( TT_Face    face,
                     FT_Stream  stream );


  FT_LOCAL( FT_Error )
  tt_face_load_cmap( TT_Face    face,
                     FT_Stream  stream );


  FT_LOCAL( FT_Error )
  tt_face_load_maxp( TT_Face    face,
                     FT_Stream  stream );


  FT_LOCAL( FT_Error )
  tt_face_load_name( TT_Face    face,
                     FT_Stream  stream );


  FT_LOCAL( FT_Error )
  tt_face_load_os2( TT_Face    face,
                    FT_Stream  stream );


  FT_LOCAL( FT_Error )
  tt_face_load_post( TT_Face    face,
                     FT_Stream  stream );


  FT_LOCAL( FT_Error )
  tt_face_load_pclt( TT_Face    face,
                     FT_Stream  stream );

  FT_LOCAL( void )
  tt_face_free_name( TT_Face  face );


  FT_LOCAL( FT_Error )
  tt_face_load_gasp( TT_Face    face,
                     FT_Stream  stream );

#ifdef TT_CONFIG_OPTION_EMBEDDED_BITMAPS

  FT_LOCAL( FT_Error )
  tt_face_load_bhed( TT_Face    face,
                     FT_Stream  stream );

#endif /* TT_CONFIG_OPTION_EMBEDDED_BITMAPS */


FT_END_HEADER

#endif /* TTLOAD_H_ */


/* END */
