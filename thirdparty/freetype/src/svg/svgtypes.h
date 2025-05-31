/****************************************************************************
 *
 * svgtypes.h
 *
 *   The FreeType SVG renderer internal types (specification).
 *
 * Copyright (C) 2022-2024 by
 * David Turner, Robert Wilhelm, Werner Lemberg, and Moazin Khatti.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */

#ifndef SVGTYPES_H_
#define SVGTYPES_H_

#include <ft2build.h>
#include <freetype/internal/ftobjs.h>
#include <freetype/ftrender.h>
#include <freetype/otsvg.h>


  typedef struct SVG_RendererRec_
  {
    FT_RendererRec     root;   /* this inherits FT_RendererRec                */
    FT_Bool            loaded;
    FT_Bool            hooks_set;
    SVG_RendererHooks  hooks;  /* this holds hooks for SVG rendering          */
    FT_Pointer         state;  /* a place for hooks to store state, if needed */

  } SVG_RendererRec;

  typedef struct SVG_RendererRec_*  SVG_Renderer;

#endif /* SVGTYPES_H_ */


/* EOF */
