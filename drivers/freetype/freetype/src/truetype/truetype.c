/***************************************************************************/
/*                                                                         */
/*  truetype.c                                                             */
/*                                                                         */
/*    FreeType TrueType driver component (body only).                      */
/*                                                                         */
/*  Copyright 1996-2001, 2004, 2006, 2012 by                               */
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
#include "ttpic.c"
#include "ttdriver.c"   /* driver interface    */
#include "ttpload.c"    /* tables loader       */
#include "ttgload.c"    /* glyph loader        */
#include "ttobjs.c"     /* object manager      */

#ifdef TT_USE_BYTECODE_INTERPRETER
#include "ttinterp.c"
#include "ttsubpix.c"
#endif

#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
#include "ttgxvar.c"    /* gx distortable font */
#endif


/* END */
