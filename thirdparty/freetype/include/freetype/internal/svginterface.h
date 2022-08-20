/****************************************************************************
 *
 * svginterface.h
 *
 *   Interface of ot-svg module (specification only).
 *
 * Copyright (C) 2022 by
 * David Turner, Robert Wilhelm, Werner Lemberg, and Moazin Khatti.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#ifndef SVGINTERFACE_H_
#define SVGINTERFACE_H_

#include <ft2build.h>
#include <freetype/otsvg.h>


FT_BEGIN_HEADER

  typedef FT_Error
  (*Preset_Bitmap_Func)( FT_Module     module,
                         FT_GlyphSlot  slot,
                         FT_Bool       cache );

  typedef struct  SVG_Interface_
  {
    Preset_Bitmap_Func  preset_slot;

  } SVG_Interface;

  typedef SVG_Interface*  SVG_Service;

FT_END_HEADER

#endif /* SVGINTERFACE_H_ */


/* END */
