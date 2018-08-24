/***************************************************************************/
/*                                                                         */
/*  cidriver.h                                                             */
/*                                                                         */
/*    High-level CID driver interface (specification).                     */
/*                                                                         */
/*  Copyright 1996-2018 by                                                 */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


#ifndef CIDRIVER_H_
#define CIDRIVER_H_


#include <ft2build.h>
#include FT_INTERNAL_DRIVER_H


FT_BEGIN_HEADER

#ifdef FT_CONFIG_OPTION_PIC
#error "this module does not support PIC yet"
#endif


  FT_CALLBACK_TABLE
  const FT_Driver_ClassRec  t1cid_driver_class;


FT_END_HEADER

#endif /* CIDRIVER_H_ */


/* END */
