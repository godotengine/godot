/****************************************************************************
 *
 * svpostnm.h
 *
 *   The FreeType PostScript name services (specification).
 *
 * Copyright (C) 2003-2023 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#ifndef SVPOSTNM_H_
#define SVPOSTNM_H_

#include <freetype/internal/ftserv.h>


FT_BEGIN_HEADER

  /*
   * A trivial service used to retrieve the PostScript name of a given font
   * when available.  The `get_name' field should never be `NULL`.
   *
   * The corresponding function can return `NULL` to indicate that the
   * PostScript name is not available.
   *
   * The name is owned by the face and will be destroyed with it.
   */

#define FT_SERVICE_ID_POSTSCRIPT_FONT_NAME  "postscript-font-name"


  typedef const char*
  (*FT_PsName_GetFunc)( FT_Face  face );


  FT_DEFINE_SERVICE( PsFontName )
  {
    FT_PsName_GetFunc  get_ps_font_name;
  };


#define FT_DEFINE_SERVICE_PSFONTNAMEREC( class_, get_ps_font_name_ ) \
  static const FT_Service_PsFontNameRec  class_ =                    \
  {                                                                  \
    get_ps_font_name_                                                \
  };

  /* */


FT_END_HEADER


#endif /* SVPOSTNM_H_ */


/* END */
