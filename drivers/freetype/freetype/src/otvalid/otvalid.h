/***************************************************************************/
/*                                                                         */
/*  otvalid.h                                                              */
/*                                                                         */
/*    OpenType table validation (specification only).                      */
/*                                                                         */
/*  Copyright 2004, 2008 by                                                */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


#ifndef __OTVALID_H__
#define __OTVALID_H__


#include <ft2build.h>
#include FT_FREETYPE_H

#include "otverror.h"           /* must come before FT_INTERNAL_VALIDATE_H */

#include FT_INTERNAL_VALIDATE_H
#include FT_INTERNAL_STREAM_H


FT_BEGIN_HEADER


  FT_LOCAL( void )
  otv_BASE_validate( FT_Bytes      table,
                     FT_Validator  valid );

  /* GSUB and GPOS tables should already be validated; */
  /* if missing, set corresponding argument to 0       */
  FT_LOCAL( void )
  otv_GDEF_validate( FT_Bytes      table,
                     FT_Bytes      gsub,
                     FT_Bytes      gpos,
                     FT_UInt       glyph_count,
                     FT_Validator  valid );

  FT_LOCAL( void )
  otv_GPOS_validate( FT_Bytes      table,
                     FT_UInt       glyph_count,
                     FT_Validator  valid );

  FT_LOCAL( void )
  otv_GSUB_validate( FT_Bytes      table,
                     FT_UInt       glyph_count,
                     FT_Validator  valid );

  /* GSUB and GPOS tables should already be validated; */
  /* if missing, set corresponding argument to 0       */
  FT_LOCAL( void )
  otv_JSTF_validate( FT_Bytes      table,
                     FT_Bytes      gsub,
                     FT_Bytes      gpos,
                     FT_UInt       glyph_count,
                     FT_Validator  valid );

  FT_LOCAL( void )
  otv_MATH_validate( FT_Bytes      table,
                     FT_UInt       glyph_count,
                     FT_Validator  ftvalid );


FT_END_HEADER

#endif /* __OTVALID_H__ */


/* END */
