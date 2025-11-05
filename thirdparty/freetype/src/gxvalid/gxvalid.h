/****************************************************************************
 *
 * gxvalid.h
 *
 *   TrueTypeGX/AAT table validation (specification only).
 *
 * Copyright (C) 2005-2025 by
 * suzuki toshiya, Masatake YAMATO, Red Hat K.K.,
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */

/****************************************************************************
 *
 * gxvalid is derived from both gxlayout module and otvalid module.
 * Development of gxlayout is supported by the Information-technology
 * Promotion Agency(IPA), Japan.
 *
 */


#ifndef GXVALID_H_
#define GXVALID_H_

#include <freetype/freetype.h>

#include "gxverror.h"                     /* must come before `ftvalid.h' */

#include <freetype/internal/ftvalid.h>
#include <freetype/internal/ftstream.h>


FT_BEGIN_HEADER


  FT_LOCAL( void )
  gxv_feat_validate( FT_Bytes      table,
                     FT_Face       face,
                     FT_Validator  valid );


  FT_LOCAL( void )
  gxv_bsln_validate( FT_Bytes      table,
                     FT_Face       face,
                     FT_Validator  valid );


  FT_LOCAL( void )
  gxv_trak_validate( FT_Bytes      table,
                     FT_Face       face,
                     FT_Validator  valid );

  FT_LOCAL( void )
  gxv_just_validate( FT_Bytes      table,
                     FT_Face       face,
                     FT_Validator  valid );

  FT_LOCAL( void )
  gxv_mort_validate( FT_Bytes      table,
                     FT_Face       face,
                     FT_Validator  valid );

  FT_LOCAL( void )
  gxv_morx_validate( FT_Bytes      table,
                     FT_Face       face,
                     FT_Validator  valid );

  FT_LOCAL( void )
  gxv_kern_validate( FT_Bytes      table,
                     FT_Face       face,
                     FT_Validator  valid );

  FT_LOCAL( void )
  gxv_kern_validate_classic( FT_Bytes      table,
                             FT_Face       face,
                             FT_Int        dialect_flags,
                             FT_Validator  valid );

  FT_LOCAL( void )
  gxv_opbd_validate( FT_Bytes      table,
                     FT_Face       face,
                     FT_Validator  valid );

  FT_LOCAL( void )
  gxv_prop_validate( FT_Bytes      table,
                     FT_Face       face,
                     FT_Validator  valid );

  FT_LOCAL( void )
  gxv_lcar_validate( FT_Bytes      table,
                     FT_Face       face,
                     FT_Validator  valid );


FT_END_HEADER


#endif /* GXVALID_H_ */


/* END */
