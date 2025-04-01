/****************************************************************************
 *
 * woff2tags.h
 *
 *   WOFF2 Font table tags (specification).
 *
 * Copyright (C) 2019-2024 by
 * Nikhil Ramakrishnan, David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#ifndef WOFF2TAGS_H
#define WOFF2TAGS_H


#include <freetype/internal/ftobjs.h>
#include <freetype/internal/compiler-macros.h>


FT_BEGIN_HEADER

#ifdef FT_CONFIG_OPTION_USE_BROTLI

  FT_LOCAL( FT_Tag )
  woff2_known_tags( FT_Byte  index );

#endif

FT_END_HEADER

#endif /* WOFF2TAGS_H */


/* END */
