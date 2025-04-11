/****************************************************************************
 *
 * t42types.h
 *
 *   Type 42 font data types (specification only).
 *
 * Copyright (C) 2002-2024 by
 * Roberto Alameda.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#ifndef T42TYPES_H_
#define T42TYPES_H_


#include <freetype/freetype.h>
#include <freetype/t1tables.h>
#include <freetype/internal/t1types.h>
#include <freetype/internal/pshints.h>


FT_BEGIN_HEADER


  typedef struct  T42_FaceRec_
  {
    FT_FaceRec      root;
    T1_FontRec      type1;
    const void*     psnames;
    const void*     psaux;
#if 0
    const void*     afm_data;
#endif
    FT_Byte*        ttf_data;
    FT_Long         ttf_size;
    FT_Face         ttf_face;
    FT_CharMapRec   charmaprecs[2];
    FT_CharMap      charmaps[2];
    PS_UnicodesRec  unicode_map;

  } T42_FaceRec, *T42_Face;


FT_END_HEADER

#endif /* T42TYPES_H_ */


/* END */
