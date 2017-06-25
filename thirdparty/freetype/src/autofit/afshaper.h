/***************************************************************************/
/*                                                                         */
/*  afshaper.h                                                             */
/*                                                                         */
/*    HarfBuzz interface for accessing OpenType features (specification).  */
/*                                                                         */
/*  Copyright 2013-2017 by                                                 */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


#ifndef AFSHAPER_H_
#define AFSHAPER_H_


#include <ft2build.h>
#include FT_FREETYPE_H


#ifdef FT_CONFIG_OPTION_USE_HARFBUZZ

#include <hb.h>
#include <hb-ot.h>
#include <hb-ft.h>

#endif


FT_BEGIN_HEADER

  FT_Error
  af_shaper_get_coverage( AF_FaceGlobals  globals,
                          AF_StyleClass   style_class,
                          FT_UShort*      gstyles,
                          FT_Bool         default_script );


  void*
  af_shaper_buf_create( FT_Face  face );

  void
  af_shaper_buf_destroy( FT_Face  face,
                         void*    buf );

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
