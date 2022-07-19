/****************************************************************************
 *
 * psaux.c
 *
 *   FreeType auxiliary PostScript driver component (body only).
 *
 * Copyright (C) 1996-2020 by
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

#include "afmparse.c"
#include "psauxmod.c"
#include "psconv.c"
#include "psobjs.c"
#include "t1cmap.c"
#include "t1decode.c"
#include "cffdecode.c"

#include "psarrst.c"
#include "psblues.c"
#include "pserror.c"
#include "psfont.c"
#include "psft.c"
#include "pshints.c"
#include "psintrp.c"
#include "psread.c"
#include "psstack.c"


/* END */
