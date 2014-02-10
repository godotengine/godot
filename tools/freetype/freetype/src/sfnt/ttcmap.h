/***************************************************************************/
/*                                                                         */
/*  ttcmap.h                                                               */
/*                                                                         */
/*    TrueType character mapping table (cmap) support (specification).     */
/*                                                                         */
/*  Copyright 2002-2005, 2009, 2012 by                                     */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


#ifndef __TTCMAP_H__
#define __TTCMAP_H__


#include <ft2build.h>
#include FT_INTERNAL_TRUETYPE_TYPES_H
#include FT_INTERNAL_VALIDATE_H
#include FT_SERVICE_TT_CMAP_H

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


#ifndef FT_CONFIG_OPTION_PIC

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

#else /* FT_CONFIG_OPTION_PIC */

#define FT_DEFINE_TT_CMAP( class_,                      \
                           size_,                       \
                           init_,                       \
                           done_,                       \
                           char_index_,                 \
                           char_next_,                  \
                           char_var_index_,             \
                           char_var_default_,           \
                           variant_list_,               \
                           charvariant_list_,           \
                           variantchar_list_,           \
                           format_,                     \
                           validate_,                   \
                           get_cmap_info_ )             \
  void                                                  \
  FT_Init_Class_ ## class_( TT_CMap_ClassRec*  clazz )  \
  {                                                     \
    clazz->clazz.size             = size_;              \
    clazz->clazz.init             = init_;              \
    clazz->clazz.done             = done_;              \
    clazz->clazz.char_index       = char_index_;        \
    clazz->clazz.char_next        = char_next_;         \
    clazz->clazz.char_var_index   = char_var_index_;    \
    clazz->clazz.char_var_default = char_var_default_;  \
    clazz->clazz.variant_list     = variant_list_;      \
    clazz->clazz.charvariant_list = charvariant_list_;  \
    clazz->clazz.variantchar_list = variantchar_list_;  \
    clazz->format                 = format_;            \
    clazz->validate               = validate_;          \
    clazz->get_cmap_info          = get_cmap_info_;     \
  }

#endif /* FT_CONFIG_OPTION_PIC */


  typedef struct  TT_ValidatorRec_
  {
    FT_ValidatorRec  validator;
    FT_UInt          num_glyphs;

  } TT_ValidatorRec, *TT_Validator;


#define TT_VALIDATOR( x )          ( (TT_Validator)( x ) )
#define TT_VALID_GLYPH_COUNT( x )  TT_VALIDATOR( x )->num_glyphs


  FT_LOCAL( FT_Error )
  tt_face_build_cmaps( TT_Face  face );

  /* used in tt-cmaps service */
  FT_LOCAL( FT_Error )
  tt_get_cmap_info( FT_CharMap    charmap,
                    TT_CMapInfo  *cmap_info );


FT_END_HEADER

#endif /* __TTCMAP_H__ */


/* END */
