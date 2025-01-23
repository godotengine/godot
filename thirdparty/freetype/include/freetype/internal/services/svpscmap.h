/****************************************************************************
 *
 * svpscmap.h
 *
 *   The FreeType PostScript charmap service (specification).
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


#ifndef SVPSCMAP_H_
#define SVPSCMAP_H_

#include <freetype/internal/ftobjs.h>


FT_BEGIN_HEADER


#define FT_SERVICE_ID_POSTSCRIPT_CMAPS  "postscript-cmaps"


  /*
   * Adobe glyph name to unicode value.
   */
  typedef FT_UInt32
  (*PS_Unicode_ValueFunc)( const char*  glyph_name );

  /*
   * Macintosh name id to glyph name.  `NULL` if invalid index.
   */
  typedef const char*
  (*PS_Macintosh_NameFunc)( FT_UInt  name_index );

  /*
   * Adobe standard string ID to glyph name.  `NULL` if invalid index.
   */
  typedef const char*
  (*PS_Adobe_Std_StringsFunc)( FT_UInt  string_index );


  /*
   * Simple unicode -> glyph index charmap built from font glyph names table.
   */
  typedef struct  PS_UniMap_
  {
    FT_UInt32  unicode;      /* bit 31 set: is glyph variant */
    FT_UInt    glyph_index;

  } PS_UniMap;


  typedef struct PS_UnicodesRec_*  PS_Unicodes;

  typedef struct  PS_UnicodesRec_
  {
    FT_CMapRec  cmap;
    FT_UInt     num_maps;
    PS_UniMap*  maps;

  } PS_UnicodesRec;


  /*
   * A function which returns a glyph name for a given index.  Returns
   * `NULL` if invalid index.
   */
  typedef const char*
  (*PS_GetGlyphNameFunc)( FT_Pointer  data,
                          FT_UInt     string_index );

  /*
   * A function used to release the glyph name returned by
   * PS_GetGlyphNameFunc, when needed
   */
  typedef void
  (*PS_FreeGlyphNameFunc)( FT_Pointer  data,
                           const char*  name );

  typedef FT_Error
  (*PS_Unicodes_InitFunc)( FT_Memory             memory,
                           PS_Unicodes           unicodes,
                           FT_UInt               num_glyphs,
                           PS_GetGlyphNameFunc   get_glyph_name,
                           PS_FreeGlyphNameFunc  free_glyph_name,
                           FT_Pointer            glyph_data );

  typedef FT_UInt
  (*PS_Unicodes_CharIndexFunc)( PS_Unicodes  unicodes,
                                FT_UInt32    unicode );

  typedef FT_UInt
  (*PS_Unicodes_CharNextFunc)( PS_Unicodes  unicodes,
                               FT_UInt32   *unicode );


  FT_DEFINE_SERVICE( PsCMaps )
  {
    PS_Unicode_ValueFunc       unicode_value;

    PS_Unicodes_InitFunc       unicodes_init;
    PS_Unicodes_CharIndexFunc  unicodes_char_index;
    PS_Unicodes_CharNextFunc   unicodes_char_next;

    PS_Macintosh_NameFunc      macintosh_name;
    PS_Adobe_Std_StringsFunc   adobe_std_strings;
    const unsigned short*      adobe_std_encoding;
    const unsigned short*      adobe_expert_encoding;
  };


#define FT_DEFINE_SERVICE_PSCMAPSREC( class_,                               \
                                      unicode_value_,                       \
                                      unicodes_init_,                       \
                                      unicodes_char_index_,                 \
                                      unicodes_char_next_,                  \
                                      macintosh_name_,                      \
                                      adobe_std_strings_,                   \
                                      adobe_std_encoding_,                  \
                                      adobe_expert_encoding_ )              \
  static const FT_Service_PsCMapsRec  class_ =                              \
  {                                                                         \
    unicode_value_, unicodes_init_,                                         \
    unicodes_char_index_, unicodes_char_next_, macintosh_name_,             \
    adobe_std_strings_, adobe_std_encoding_, adobe_expert_encoding_         \
  };

  /* */


FT_END_HEADER


#endif /* SVPSCMAP_H_ */


/* END */
