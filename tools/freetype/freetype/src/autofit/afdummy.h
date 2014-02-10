/***************************************************************************/
/*                                                                         */
/*  afdummy.h                                                              */
/*                                                                         */
/*    Auto-fitter dummy routines to be used if no hinting should be        */
/*    performed (specification).                                           */
/*                                                                         */
/*  Copyright 2003-2005, 2011 by                                           */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


#ifndef __AFDUMMY_H__
#define __AFDUMMY_H__

#include "aftypes.h"


FT_BEGIN_HEADER

 /*  A dummy script metrics class used when no hinting should
  *  be performed.  This is the default for non-latin glyphs!
  */

  AF_DECLARE_SCRIPT_CLASS( af_dummy_script_class )

/* */

FT_END_HEADER


#endif /* __AFDUMMY_H__ */


/* END */
