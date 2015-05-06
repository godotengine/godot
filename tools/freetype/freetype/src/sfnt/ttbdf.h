/***************************************************************************/
/*                                                                         */
/*  ttbdf.h                                                                */
/*                                                                         */
/*    TrueType and OpenType embedded BDF properties (specification).       */
/*                                                                         */
/*  Copyright 2005 by                                                      */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


#ifndef __TTBDF_H__
#define __TTBDF_H__


#include <ft2build.h>
#include "ttload.h"
#include FT_BDF_H


FT_BEGIN_HEADER


  FT_LOCAL( void )
  tt_face_free_bdf_props( TT_Face  face );


  FT_LOCAL( FT_Error )
  tt_face_find_bdf_prop( TT_Face           face,
                         const char*       property_name,
                         BDF_PropertyRec  *aprop );


FT_END_HEADER

#endif /* __TTBDF_H__ */


/* END */
