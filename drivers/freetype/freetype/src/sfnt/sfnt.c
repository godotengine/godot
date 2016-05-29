/***************************************************************************/
/*                                                                         */
/*  sfnt.c                                                                 */
/*                                                                         */
/*    Single object library component.                                     */
/*                                                                         */
/*  Copyright 1996-2006, 2013 by                                           */
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
#include "sfntpic.c"
#include "ttload.c"
#include "ttmtx.c"
#include "ttcmap.c"
#include "ttkern.c"
#include "sfobjs.c"
#include "sfdriver.c"

#ifdef TT_CONFIG_OPTION_EMBEDDED_BITMAPS
#include "pngshim.c"
#include "ttsbit.c"
#endif

#ifdef TT_CONFIG_OPTION_POSTSCRIPT_NAMES
#include "ttpost.c"
#endif

#ifdef TT_CONFIG_OPTION_BDF
#include "ttbdf.c"
#endif

/* END */
