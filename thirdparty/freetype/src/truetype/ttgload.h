/****************************************************************************
 *
 * ttgload.h
 *
 *   TrueType Glyph Loader (specification).
 *
 * Copyright (C) 1996-2025 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#ifndef TTGLOAD_H_
#define TTGLOAD_H_


#include "ttobjs.h"

#ifdef TT_USE_BYTECODE_INTERPRETER
#include "ttinterp.h"
#endif


FT_BEGIN_HEADER


  FT_LOCAL( void )
  TT_Init_Glyph_Loading( TT_Face  face );

  FT_LOCAL( void )
  TT_Get_HMetrics( TT_Face     face,
                   FT_UInt     idx,
                   FT_Short*   lsb,
                   FT_UShort*  aw );

  FT_LOCAL( void )
  TT_Get_VMetrics( TT_Face     face,
                   FT_UInt     idx,
                   FT_Pos      yMax,
                   FT_Short*   tsb,
                   FT_UShort*  ah );

  FT_LOCAL( FT_Error )
  TT_Load_Glyph( TT_Size       size,
                 TT_GlyphSlot  glyph,
                 FT_UInt       glyph_index,
                 FT_Int32      load_flags );


FT_END_HEADER

#endif /* TTGLOAD_H_ */


/* END */
