/****************************************************************************
 *
 * ft-hb.h
 *
 *   FreeType-HarfBuzz bridge (specification).
 *
 * Copyright (C) 2025 by
 * Behdad Esfahbod.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#ifndef FT_HB_H
#define FT_HB_H

#include <freetype/internal/compiler-macros.h>
#include <freetype/freetype.h>


FT_BEGIN_HEADER

#ifdef FT_CONFIG_OPTION_USE_HARFBUZZ

#  include "ft-hb-types.h"

#  ifdef FT_CONFIG_OPTION_USE_HARFBUZZ_DYNAMIC

#    define HB_EXTERN( ret, name, args ) \
              typedef ret (*ft_ ## name ## _func_t) args;
#    include "ft-hb-decls.h"
#    undef HB_EXTERN

  typedef struct ft_hb_funcs_t
  {
#    define HB_EXTERN( ret, name, args ) \
              ft_ ## name ## _func_t  name;
#    include "ft-hb-decls.h"
#    undef HB_EXTERN
  } ft_hb_funcs_t;

  struct  AF_ModuleRec_;

  FT_LOCAL( void )
  ft_hb_funcs_init( struct AF_ModuleRec_  *af_module );

  FT_LOCAL( void )
  ft_hb_funcs_done( struct AF_ModuleRec_  *af_module );

#    define hb( x )  globals->module->hb_funcs->hb_ ## x

#  else /* !FT_CONFIG_OPTION_USE_HARFBUZZ_DYNAMIC */

#    define HB_EXTERN( ret, name, args ) \
              ret name args;
#    include "ft-hb-decls.h"
#    undef HB_EXTERN

#    define hb( x )  hb_ ## x

#  endif /* !FT_CONFIG_OPTION_USE_HARFBUZZ_DYNAMIC */

#endif /* FT_CONFIG_OPTION_USE_HARFBUZZ */


  struct AF_FaceGlobalsRec_;

  FT_LOCAL( FT_Bool )
  ft_hb_enabled( struct AF_FaceGlobalsRec_  *globals );


FT_END_HEADER

#endif /* FT_HB_H */


/* END */
