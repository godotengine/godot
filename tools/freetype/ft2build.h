/***************************************************************************/
/*                                                                         */
/*  ft2build.h                                                             */
/*                                                                         */
/*    FreeType 2 build and setup macros.                                   */
/*    (Generic version)                                                    */
/*                                                                         */
/*  Copyright 1996-2001, 2003, 2006 by                                     */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


 /*
  *  This is a development version of <ft2build.h> that is used
  *  to build the library in debug mode.  Its only difference with
  *  the reference is that it forces the use of the local `ftoption.h'
  *  which contains different settings for all configuration macros.
  *
  *  To use it, you must define the environment variable FT2_BUILD_INCLUDE
  *  to point to the directory containing these two files (`ft2build.h' and
  *  `ftoption.h'), then invoke Jam as usual.
  */

#ifndef __FT2_BUILD_DEVEL_H__
#define __FT2_BUILD_DEVEL_H__

#define  FT_CONFIG_OPTIONS_H   <ftoption.h>
#define FT2_BUILD_LIBRARY
#include <freetype/include/freetype/config/ftheader.h>

#endif /* __FT2_BUILD_DEVEL_H__ */


/* END */
