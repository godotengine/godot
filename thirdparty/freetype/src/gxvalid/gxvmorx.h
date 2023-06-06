/****************************************************************************
 *
 * gxvmorx.h
 *
 *   TrueTypeGX/AAT common definition for morx table (specification).
 *
 * Copyright (C) 2005-2023 by
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


#ifndef GXVMORX_H_
#define GXVMORX_H_


#include "gxvalid.h"
#include "gxvcommn.h"
#include "gxvmort.h"

#include <freetype/ftsnames.h>


FT_BEGIN_HEADER


  FT_LOCAL( void )
  gxv_morx_subtable_type0_validate( FT_Bytes       table,
                                    FT_Bytes       limit,
                                    GXV_Validator  gxvalid );

  FT_LOCAL( void )
  gxv_morx_subtable_type1_validate( FT_Bytes       table,
                                    FT_Bytes       limit,
                                    GXV_Validator  gxvalid );

  FT_LOCAL( void )
  gxv_morx_subtable_type2_validate( FT_Bytes       table,
                                    FT_Bytes       limit,
                                    GXV_Validator  gxvalid );

  FT_LOCAL( void )
  gxv_morx_subtable_type4_validate( FT_Bytes       table,
                                    FT_Bytes       limit,
                                    GXV_Validator  gxvalid );

  FT_LOCAL( void )
  gxv_morx_subtable_type5_validate( FT_Bytes       table,
                                    FT_Bytes       limit,
                                    GXV_Validator  gxvalid );


FT_END_HEADER

#endif /* GXVMORX_H_ */


/* END */
