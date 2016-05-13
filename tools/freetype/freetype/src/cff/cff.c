/***************************************************************************/
/*                                                                         */
/*  cff.c                                                                  */
/*                                                                         */
/*    FreeType OpenType driver component (body only).                      */
/*                                                                         */
/*  Copyright 1996-2001, 2002, 2013 by                                     */
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

#include "cffpic.c"
#include "cffdrivr.c"
#include "cffparse.c"
#include "cffload.c"
#include "cffobjs.c"
#include "cffgload.c"
#include "cffcmap.c"

#include "cf2arrst.c"
#include "cf2blues.c"
#include "cf2error.c"
#include "cf2font.c"
#include "cf2ft.c"
#include "cf2hints.c"
#include "cf2intrp.c"
#include "cf2read.c"
#include "cf2stack.c"

/* END */
