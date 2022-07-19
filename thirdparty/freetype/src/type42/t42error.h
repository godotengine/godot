/****************************************************************************
 *
 * t42error.h
 *
 *   Type 42 error codes (specification only).
 *
 * Copyright (C) 2002-2020 by
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
   * This file is used to define the Type 42 error enumeration constants.
   *
   */

#ifndef T42ERROR_H_
#define T42ERROR_H_

#include <freetype/ftmoderr.h>

#undef FTERRORS_H_

#undef  FT_ERR_PREFIX
#define FT_ERR_PREFIX  T42_Err_
#define FT_ERR_BASE    FT_Mod_Err_Type42

#include <freetype/fterrors.h>

#endif /* T42ERROR_H_ */


/* END */
