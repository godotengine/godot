/****************************************************************************
 *
 * afshaper.h
 *
 *   HarfBuzz interface for accessing OpenType features (specification).
 *
 * Copyright (C) 2013-2025 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#ifndef AFSHAPER_H_
#define AFSHAPER_H_


#include <freetype/freetype.h>


FT_BEGIN_HEADER

#ifdef FT_CONFIG_OPTION_USE_HARFBUZZ
  FT_LOCAL_ARRAY( hb_script_t )
  af_hb_scripts[];
#endif


  FT_Error
  af_shaper_get_coverage( AF_FaceGlobals  globals,
                          AF_StyleClass   style_class,
                          FT_UShort*      gstyles,
                          FT_Bool         default_script );


  void*
  af_shaper_buf_create( AF_FaceGlobals  globals );

  void
  af_shaper_buf_destroy( AF_FaceGlobals  globals,
                         void*           buf );

  const char*
  af_shaper_get_cluster( const char*      p,
                         AF_StyleMetrics  metrics,
                         void*            buf_,
                         unsigned int*    count );

  FT_ULong
  af_shaper_get_elem( AF_StyleMetrics  metrics,
                      void*            buf_,
                      unsigned int     idx,
                      FT_Long*         x_advance,
                      FT_Long*         y_offset );

 /* */

FT_END_HEADER

#endif /* AFSHAPER_H_ */


/* END */
