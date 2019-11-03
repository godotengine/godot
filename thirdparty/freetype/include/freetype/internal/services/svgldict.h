/****************************************************************************
 *
 * svgldict.h
 *
 *   The FreeType glyph dictionary services (specification).
 *
 * Copyright (C) 2003-2019 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#ifndef SVGLDICT_H_
#define SVGLDICT_H_

#include FT_INTERNAL_SERVICE_H


FT_BEGIN_HEADER


  /*
   * A service used to retrieve glyph names, as well as to find the index of
   * a given glyph name in a font.
   *
   */

#define FT_SERVICE_ID_GLYPH_DICT  "glyph-dict"


  typedef FT_Error
  (*FT_GlyphDict_GetNameFunc)( FT_Face     face,
                               FT_UInt     glyph_index,
                               FT_Pointer  buffer,
                               FT_UInt     buffer_max );

  typedef FT_UInt
  (*FT_GlyphDict_NameIndexFunc)( FT_Face           face,
                                 const FT_String*  glyph_name );


  FT_DEFINE_SERVICE( GlyphDict )
  {
    FT_GlyphDict_GetNameFunc    get_name;
    FT_GlyphDict_NameIndexFunc  name_index;  /* optional */
  };


#define FT_DEFINE_SERVICE_GLYPHDICTREC( class_,                        \
                                        get_name_,                     \
                                        name_index_ )                  \
  static const FT_Service_GlyphDictRec  class_ =                       \
  {                                                                    \
    get_name_, name_index_                                             \
  };

  /* */


FT_END_HEADER


#endif /* SVGLDICT_H_ */


/* END */
