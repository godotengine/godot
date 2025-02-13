/****************************************************************************
 *
 * cfferrs.h
 *
 *   CFF error codes (specification only).
 *
 * Copyright (C) 2001-2023 by
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
   * This file is used to define the CFF error enumeration constants.
   *
   */

#ifndef CFFERRS_H_
#define CFFERRS_H_

#include <freetype/ftmoderr.h>

#undef FTERRORS_H_

#undef  FT_ERR_PREFIX
#define FT_ERR_PREFIX  CFF_Err_
#define FT_ERR_BASE    FT_Mod_Err_CFF


#include <freetype/fterrors.h>

#endif /* CFFERRS_H_ */


/* END */
