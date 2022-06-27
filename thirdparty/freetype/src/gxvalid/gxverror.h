/****************************************************************************
 *
 * gxverror.h
 *
 *   TrueTypeGX/AAT validation module error codes (specification only).
 *
 * Copyright (C) 2004-2022 by
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


  /**************************************************************************
   *
   * This file is used to define the OpenType validation module error
   * enumeration constants.
   *
   */

#ifndef GXVERROR_H_
#define GXVERROR_H_

#include <freetype/ftmoderr.h>

#undef FTERRORS_H_

#undef  FT_ERR_PREFIX
#define FT_ERR_PREFIX  GXV_Err_
#define FT_ERR_BASE    FT_Mod_Err_GXvalid

#include <freetype/fterrors.h>

#endif /* GXVERROR_H_ */


/* END */
