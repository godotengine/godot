/****************************************************************************
 *
 * tterrors.h
 *
 *   TrueType error codes (specification only).
 *
 * Copyright (C) 2001-2021 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


  /**************************************************************************
   *
   * This file is used to define the TrueType error enumeration
   * constants.
   *
   */

#ifndef TTERRORS_H_
#define TTERRORS_H_

#include <freetype/ftmoderr.h>

#undef FTERRORS_H_

#undef  FT_ERR_PREFIX
#define FT_ERR_PREFIX  TT_Err_
#define FT_ERR_BASE    FT_Mod_Err_TrueType

#include <freetype/fterrors.h>

#endif /* TTERRORS_H_ */


/* END */
