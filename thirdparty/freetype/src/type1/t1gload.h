/****************************************************************************
 *
 * t1gload.h
 *
 *   Type 1 Glyph Loader (specification).
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


#ifndef T1GLOAD_H_
#define T1GLOAD_H_


#include "t1objs.h"


FT_BEGIN_HEADER


  FT_LOCAL( FT_Error )
  T1_Compute_Max_Advance( T1_Face  face,
                          FT_Pos*  max_advance );

  FT_LOCAL( FT_Error )
  T1_Get_Advances( FT_Face    face,
                   FT_UInt    first,
                   FT_UInt    count,
                   FT_Int32   load_flags,
                   FT_Fixed*  advances );

  FT_LOCAL( FT_Error )
  T1_Load_Glyph( FT_GlyphSlot  glyph,
                 FT_Size       size,
                 FT_UInt       glyph_index,
                 FT_Int32      load_flags );


FT_END_HEADER

#endif /* T1GLOAD_H_ */


/* END */
