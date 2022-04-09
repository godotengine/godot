/****************************************************************************
 *
 * sfwoff.h
 *
 *   WOFFF format management (specification).
 *
 * Copyright (C) 1996-2021 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#ifndef SFWOFF_H_
#define SFWOFF_H_


#include <freetype/internal/sfnt.h>
#include <freetype/internal/ftobjs.h>


FT_BEGIN_HEADER

#ifdef FT_CONFIG_OPTION_USE_ZLIB

  FT_LOCAL( FT_Error )
  woff_open_font( FT_Stream  stream,
                  TT_Face    face );


#endif

FT_END_HEADER

#endif /* SFWOFF_H_ */


/* END */
