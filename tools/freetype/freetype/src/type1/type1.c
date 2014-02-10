/***************************************************************************/
/*                                                                         */
/*  type1.c                                                                */
/*                                                                         */
/*    FreeType Type 1 driver component (body only).                        */
/*                                                                         */
/*  Copyright 1996-2001 by                                                 */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


#define FT_MAKE_OPTION_SINGLE_OBJECT

#include <ft2build.h>
#include "t1parse.c"
#include "t1load.c"
#include "t1objs.c"
#include "t1driver.c"
#include "t1gload.c"

#ifndef T1_CONFIG_OPTION_NO_AFM
#include "t1afm.c"
#endif


/* END */
