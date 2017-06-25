/***************************************************************************/
/*                                                                         */
/*  t1driver.h                                                             */
/*                                                                         */
/*    High-level Type 1 driver interface (specification).                  */
/*                                                                         */
/*  Copyright 1996-2017 by                                                 */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


#ifndef T1DRIVER_H_
#define T1DRIVER_H_


#include <ft2build.h>
#include FT_INTERNAL_DRIVER_H


FT_BEGIN_HEADER

#ifdef FT_CONFIG_OPTION_PIC
#error "this module does not support PIC yet"
#endif


  FT_EXPORT_VAR( const FT_Driver_ClassRec )  t1_driver_class;


FT_END_HEADER

#endif /* T1DRIVER_H_ */


/* END */
