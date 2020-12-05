/****************************************************************************
 *
 * ttpload.h
 *
 *   TrueType-specific tables loader (specification).
 *
 * Copyright (C) 1996-2020 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#ifndef TTPLOAD_H_
#define TTPLOAD_H_


#include <freetype/internal/tttypes.h>


FT_BEGIN_HEADER


  FT_LOCAL( FT_Error )
  tt_face_load_loca( TT_Face    face,
                     FT_Stream  stream );

  FT_LOCAL( FT_ULong )
  tt_face_get_location( TT_Face   face,
                        FT_UInt   gindex,
                        FT_UInt  *asize );

  FT_LOCAL( void )
  tt_face_done_loca( TT_Face  face );

  FT_LOCAL( FT_Error )
  tt_face_load_cvt( TT_Face    face,
                    FT_Stream  stream );

  FT_LOCAL( FT_Error )
  tt_face_load_fpgm( TT_Face    face,
                     FT_Stream  stream );


  FT_LOCAL( FT_Error )
  tt_face_load_prep( TT_Face    face,
                     FT_Stream  stream );


  FT_LOCAL( FT_Error )
  tt_face_load_hdmx( TT_Face    face,
                     FT_Stream  stream );


  FT_LOCAL( void )
  tt_face_free_hdmx( TT_Face  face );


  FT_LOCAL( FT_Byte* )
  tt_face_get_device_metrics( TT_Face    face,
                              FT_UInt    ppem,
                              FT_UInt    gindex );

FT_END_HEADER

#endif /* TTPLOAD_H_ */


/* END */
