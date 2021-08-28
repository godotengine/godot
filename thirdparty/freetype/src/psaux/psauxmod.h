/****************************************************************************
 *
 * psauxmod.h
 *
 *   FreeType auxiliary PostScript module implementation (specification).
 *
 * Copyright (C) 2000-2020 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#ifndef PSAUXMOD_H_
#define PSAUXMOD_H_


#include <freetype/ftmodapi.h>

#include <freetype/internal/psaux.h>


FT_BEGIN_HEADER


  FT_CALLBACK_TABLE
  const CFF_Builder_FuncsRec  cff_builder_funcs;

  FT_CALLBACK_TABLE
  const PS_Builder_FuncsRec   ps_builder_funcs;

#ifndef T1_CONFIG_OPTION_NO_AFM
  FT_CALLBACK_TABLE
  const AFM_Parser_FuncsRec  afm_parser_funcs;
#endif

  FT_CALLBACK_TABLE
  const T1_CMap_ClassesRec  t1_cmap_classes;

  FT_CALLBACK_TABLE
  const CFF_Decoder_FuncsRec  cff_decoder_funcs;


  FT_EXPORT_VAR( const FT_Module_Class )  psaux_driver_class;


  FT_DECLARE_MODULE( psaux_module_class )


FT_END_HEADER

#endif /* PSAUXMOD_H_ */


/* END */
