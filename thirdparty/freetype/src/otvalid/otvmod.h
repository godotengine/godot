/***************************************************************************/
/*                                                                         */
/*  otvmod.h                                                               */
/*                                                                         */
/*    FreeType's OpenType validation module implementation                 */
/*    (specification).                                                     */
/*                                                                         */
/*  Copyright 2004-2017 by                                                 */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


#ifndef OTVMOD_H_
#define OTVMOD_H_


#include <ft2build.h>
#include FT_MODULE_H


FT_BEGIN_HEADER

#ifdef FT_CONFIG_OPTION_PIC
#error "this module does not support PIC yet"
#endif


  FT_EXPORT_VAR( const FT_Module_Class )  otv_module_class;


FT_END_HEADER

#endif /* OTVMOD_H_ */


/* END */
