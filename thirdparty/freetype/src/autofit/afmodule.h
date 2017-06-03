/***************************************************************************/
/*                                                                         */
/*  afmodule.h                                                             */
/*                                                                         */
/*    Auto-fitter module implementation (specification).                   */
/*                                                                         */
/*  Copyright 2003-2016 by                                                 */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


#ifndef AFMODULE_H_
#define AFMODULE_H_

#include <ft2build.h>
#include FT_INTERNAL_OBJECTS_H
#include FT_MODULE_H


FT_BEGIN_HEADER


  /*
   *  This is the `extended' FT_Module structure that holds the
   *  autofitter's global data.
   */

  typedef struct  AF_ModuleRec_
  {
    FT_ModuleRec  root;

    FT_UInt       fallback_style;
    FT_UInt       default_script;
#ifdef AF_CONFIG_OPTION_USE_WARPER
    FT_Bool       warping;
#endif
    FT_Bool       no_stem_darkening;
    FT_Int        darken_params[8];

  } AF_ModuleRec, *AF_Module;


FT_DECLARE_MODULE( autofit_module_class )


FT_END_HEADER

#endif /* AFMODULE_H_ */


/* END */
