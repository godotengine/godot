/****************************************************************************
 *
 * truetype.c
 *
 *   FreeType TrueType driver component (body only).
 *
 * Copyright (C) 1996-2019 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#define FT_MAKE_OPTION_SINGLE_OBJECT
#include <ft2build.h>

#include "ttdriver.c"   /* driver interface    */
#include "ttgload.c"    /* glyph loader        */
#include "ttgxvar.c"    /* gx distortable font */
#include "ttinterp.c"
#include "ttobjs.c"     /* object manager      */
#include "ttpload.c"    /* tables loader       */
#include "ttsubpix.c"


/* END */
