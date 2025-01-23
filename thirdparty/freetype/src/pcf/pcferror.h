/****************************************************************************
 *
 * pcferror.h
 *
 *   PCF error codes (specification only).
 *
 * Copyright 2001, 2012 by
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
   * This file is used to define the PCF error enumeration constants.
   *
   */

#ifndef PCFERROR_H_
#define PCFERROR_H_

#include <freetype/ftmoderr.h>

#undef FTERRORS_H_

#undef  FT_ERR_PREFIX
#define FT_ERR_PREFIX  PCF_Err_
#define FT_ERR_BASE    FT_Mod_Err_PCF

#include <freetype/fterrors.h>

#endif /* PCFERROR_H_ */


/* END */
