/****************************************************************************
 *
 * hvfload.h
 *
 *   HVF glyph loading (specification).
 *
 * Copyright (C) 2025-2026 by
 * Apple Inc.
 * written by Deborah Goldsmith <goldsmit@apple.com>
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#ifndef HVFLOAD_H_
#define HVFLOAD_H_


#include <freetype/freetype.h>
#include <freetype/internal/ftobjs.h>


FT_BEGIN_HEADER

  FT_LOCAL( FT_Error )
  hvf_slot_load_glyph( FT_GlyphSlot  glyph,
                       FT_Size       size,
                       FT_UInt       glyph_index,
                       FT_Int32      load_flags );


FT_END_HEADER

#endif /* HVFLOAD_H_ */


/* END */
