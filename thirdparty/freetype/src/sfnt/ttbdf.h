/****************************************************************************
 *
 * ttbdf.h
 *
 *   TrueType and OpenType embedded BDF properties (specification).
 *
 * Copyright (C) 2005-2025 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#ifndef TTBDF_H_
#define TTBDF_H_


#include "ttload.h"
#include <freetype/ftbdf.h>


FT_BEGIN_HEADER


#ifdef TT_CONFIG_OPTION_BDF

  FT_LOCAL( void )
  tt_face_free_bdf_props( TT_Face  face );


  FT_LOCAL( FT_Error )
  tt_face_find_bdf_prop( FT_Face           face,
                         const char*       property_name,
                         BDF_PropertyRec  *aprop );

#endif /* TT_CONFIG_OPTION_BDF */


FT_END_HEADER

#endif /* TTBDF_H_ */


/* END */
