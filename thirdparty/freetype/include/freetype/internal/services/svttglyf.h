/****************************************************************************
 *
 * svttglyf.h
 *
 *   The FreeType TrueType glyph service.
 *
 * Copyright (C) 2007-2025 by
 * David Turner.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */

#ifndef SVTTGLYF_H_
#define SVTTGLYF_H_

#include <freetype/internal/ftserv.h>
#include <freetype/tttables.h>


FT_BEGIN_HEADER


#define FT_SERVICE_ID_TT_GLYF  "tt-glyf"


  typedef FT_ULong
  (*TT_Glyf_GetLocationFunc)( FT_Face    face,
                              FT_UInt    gindex,
                              FT_ULong  *psize );

  FT_DEFINE_SERVICE( TTGlyf )
  {
    TT_Glyf_GetLocationFunc  get_location;
  };


#define FT_DEFINE_SERVICE_TTGLYFREC( class_, get_location_ )  \
  static const FT_Service_TTGlyfRec  class_ =                 \
  {                                                           \
    get_location_                                             \
  };

  /* */


FT_END_HEADER

#endif /* SVTTGLYF_H_ */


/* END */
