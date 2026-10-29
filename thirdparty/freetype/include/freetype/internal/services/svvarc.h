/****************************************************************************
 *
 * svvarc.h
 *
 *   The FreeType VARC service (specification).
 *
 * Copyright (C) 2026 by
 * David Turner, Robert Wilhelm, Werner Lemberg, and Behdad Esfahbod.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#ifndef SVVARC_H_
#define SVVARC_H_


#include <freetype/internal/ftserv.h>


FT_BEGIN_HEADER


#define FT_SERVICE_ID_VARC  "varc"


  typedef FT_Error
  (*FT_VARC_Init_Axes_Func)( FT_Face  face,
                             FT_UInt  axis_count );

  typedef FT_Error
  (*FT_VARC_Load_Func)( FT_Face    face,
                        FT_Stream  stream );

  typedef void
  (*FT_VARC_Done_Func)( FT_Face  face );

  typedef FT_Bool
  (*FT_VARC_Has_Glyph_Func)( FT_Face  face,
                             FT_UInt  glyph_index );

  typedef FT_Error
  (*FT_VARC_Load_Glyph_Func)( FT_Face       face,
                              FT_GlyphSlot  glyph_slot,
                              FT_UInt       glyph_index,
                              FT_Int32      load_flags );


  FT_DEFINE_SERVICE( VARC )
  {
    FT_VARC_Init_Axes_Func   init_axes;
    FT_VARC_Load_Func        load;
    FT_VARC_Done_Func        done;
    FT_VARC_Has_Glyph_Func   has_glyph;
    FT_VARC_Load_Glyph_Func  load_glyph;
  };


#define FT_DEFINE_SERVICE_VARCREC( class_,       \
                                   init_axes_,   \
                                   load_,        \
                                   done_,        \
                                   has_glyph_,   \
                                   load_glyph_ ) \
  static const FT_Service_VARCRec  class_ =      \
  {                                              \
    init_axes_,                                  \
    load_,                                       \
    done_,                                       \
    has_glyph_,                                  \
    load_glyph_                                  \
  };

  /* */


FT_END_HEADER

#endif /* SVVARC_H_ */


/* END */
