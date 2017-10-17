/***************************************************************************/
/*                                                                         */
/*  afloader.h                                                             */
/*                                                                         */
/*    Auto-fitter glyph loading routines (specification).                  */
/*                                                                         */
/*  Copyright 2003-2017 by                                                 */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


#ifndef AFLOADER_H_
#define AFLOADER_H_

#include "afhints.h"
#include "afmodule.h"
#include "afglobal.h"


FT_BEGIN_HEADER

  /*
   *  The autofitter module's (global) data structure to communicate with
   *  actual fonts.  If necessary, `local' data like the current face, the
   *  current face's auto-hint data, or the current glyph's parameters
   *  relevant to auto-hinting are `swapped in'.  Cf. functions like
   *  `af_loader_reset' and `af_loader_load_g'.
   */

  typedef struct  AF_LoaderRec_
  {
    /* current face data */
    FT_Face           face;
    AF_FaceGlobals    globals;

    /* current glyph data */
    AF_GlyphHints     hints;
    AF_StyleMetrics   metrics;
    FT_Bool           transformed;
    FT_Matrix         trans_matrix;
    FT_Vector         trans_delta;
    FT_Vector         pp1;
    FT_Vector         pp2;
    /* we don't handle vertical phantom points */

  } AF_LoaderRec, *AF_Loader;


  FT_LOCAL( void )
  af_loader_init( AF_Loader      loader,
                  AF_GlyphHints  hints );


  FT_LOCAL( FT_Error )
  af_loader_reset( AF_Loader  loader,
                   AF_Module  module,
                   FT_Face    face );


  FT_LOCAL( void )
  af_loader_done( AF_Loader  loader );


  FT_LOCAL( FT_Error )
  af_loader_load_glyph( AF_Loader  loader,
                        AF_Module  module,
                        FT_Face    face,
                        FT_UInt    gindex,
                        FT_Int32   load_flags );

  FT_LOCAL_DEF( FT_Int32 )
  af_loader_compute_darkening( AF_Loader  loader,
                               FT_Face    face,
                               FT_Pos     standard_width );

/* */


FT_END_HEADER

#endif /* AFLOADER_H_ */


/* END */
