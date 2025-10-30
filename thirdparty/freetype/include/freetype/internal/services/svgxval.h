/****************************************************************************
 *
 * svgxval.h
 *
 *   FreeType API for validating TrueTypeGX/AAT tables (specification).
 *
 * Copyright (C) 2004-2024 by
 * Masatake YAMATO, Red Hat K.K.,
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


#ifndef SVGXVAL_H_
#define SVGXVAL_H_

#include <freetype/ftgxval.h>
#include <freetype/internal/ftvalid.h>

FT_BEGIN_HEADER


#define FT_SERVICE_ID_GX_VALIDATE           "truetypegx-validate"
#define FT_SERVICE_ID_CLASSICKERN_VALIDATE  "classickern-validate"

  typedef FT_Error
  (*gxv_validate_func)( FT_Face   face,
                        FT_UInt   gx_flags,
                        FT_Bytes  tables[FT_VALIDATE_GX_LENGTH],
                        FT_UInt   table_length );


  typedef FT_Error
  (*ckern_validate_func)( FT_Face   face,
                          FT_UInt   ckern_flags,
                          FT_Bytes  *ckern_table );


  FT_DEFINE_SERVICE( GXvalidate )
  {
    gxv_validate_func  validate;
  };

  FT_DEFINE_SERVICE( CKERNvalidate )
  {
    ckern_validate_func  validate;
  };

  /* */


FT_END_HEADER


#endif /* SVGXVAL_H_ */


/* END */
