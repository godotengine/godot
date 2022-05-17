/****************************************************************************
 *
 * ttcmap.h
 *
 *   TrueType character mapping table (cmap) support (specification).
 *
 * Copyright (C) 2002-2022 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#ifndef TTCMAP_H_
#define TTCMAP_H_


#include <freetype/internal/tttypes.h>
#include <freetype/internal/ftvalid.h>
#include <freetype/internal/services/svttcmap.h>

FT_BEGIN_HEADER


#define TT_CMAP_FLAG_UNSORTED     1
#define TT_CMAP_FLAG_OVERLAPPING  2

  typedef struct  TT_CMapRec_
  {
    FT_CMapRec  cmap;
    FT_Byte*    data;           /* pointer to in-memory cmap table */
    FT_Int      flags;          /* for format 4 only               */

  } TT_CMapRec, *TT_CMap;

  typedef const struct TT_CMap_ClassRec_*  TT_CMap_Class;


  typedef FT_Error
  (*TT_CMap_ValidateFunc)( FT_Byte*      data,
                           FT_Validator  valid );

  typedef struct  TT_CMap_ClassRec_
  {
    FT_CMap_ClassRec      clazz;
    FT_UInt               format;
    TT_CMap_ValidateFunc  validate;
    TT_CMap_Info_GetFunc  get_cmap_info;

  } TT_CMap_ClassRec;


#define FT_DEFINE_TT_CMAP( class_,             \
                           size_,              \
                           init_,              \
                           done_,              \
                           char_index_,        \
                           char_next_,         \
                           char_var_index_,    \
                           char_var_default_,  \
                           variant_list_,      \
                           charvariant_list_,  \
                           variantchar_list_,  \
                           format_,            \
                           validate_,          \
                           get_cmap_info_ )    \
  FT_CALLBACK_TABLE_DEF                        \
  const TT_CMap_ClassRec  class_ =             \
  {                                            \
    { size_,                                   \
      init_,                                   \
      done_,                                   \
      char_index_,                             \
      char_next_,                              \
      char_var_index_,                         \
      char_var_default_,                       \
      variant_list_,                           \
      charvariant_list_,                       \
      variantchar_list_                        \
    },                                         \
                                               \
    format_,                                   \
    validate_,                                 \
    get_cmap_info_                             \
  };


#undef  TTCMAPCITEM
#define TTCMAPCITEM( a )  FT_CALLBACK_TABLE  const TT_CMap_ClassRec  a;
#include "ttcmapc.h"


  typedef struct  TT_ValidatorRec_
  {
    FT_ValidatorRec  validator;
    FT_UInt          num_glyphs;

  } TT_ValidatorRec, *TT_Validator;


#define TT_VALIDATOR( x )          ( (TT_Validator)( x ) )
#define TT_VALID_GLYPH_COUNT( x )  TT_VALIDATOR( x )->num_glyphs


  FT_CALLBACK_TABLE const TT_CMap_ClassRec  tt_cmap_unicode_class_rec;

  FT_LOCAL( FT_Error )
  tt_face_build_cmaps( TT_Face  face );

  /* used in tt-cmaps service */
  FT_LOCAL( FT_Error )
  tt_get_cmap_info( FT_CharMap    charmap,
                    TT_CMapInfo  *cmap_info );


FT_END_HEADER

#endif /* TTCMAP_H_ */


/* END */
