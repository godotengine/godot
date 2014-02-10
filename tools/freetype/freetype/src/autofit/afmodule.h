/***************************************************************************/
/*                                                                         */
/*  afmodule.h                                                             */
/*                                                                         */
/*    Auto-fitter module implementation (specification).                   */
/*                                                                         */
/*  Copyright 2003, 2004, 2005 by                                          */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


#ifndef __AFMODULE_H__
#define __AFMODULE_H__

#include <ft2build.h>
#include FT_INTERNAL_OBJECTS_H
#include FT_MODULE_H

#include "afloader.h"


FT_BEGIN_HEADER


  /*
   *  This is the `extended' FT_Module structure which holds the
   *  autofitter's global data.  Right before hinting a glyph, the data
   *  specific to the glyph's face (blue zones, stem widths, etc.) are
   *  loaded into `loader' (see function `af_loader_reset').
   */

  typedef struct  AF_ModuleRec_
  {
    FT_ModuleRec  root;

    FT_UInt       fallback_script;

    AF_LoaderRec  loader[1];

  } AF_ModuleRec;


FT_DECLARE_MODULE(autofit_module_class)


FT_END_HEADER

#endif /* __AFMODULE_H__ */


/* END */
