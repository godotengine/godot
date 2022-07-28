/****************************************************************************
 *
 * psauxerr.h
 *
 *   PS auxiliary module error codes (specification only).
 *
 * Copyright (C) 2001-2022 by
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
   * This file is used to define the PS auxiliary module error enumeration
   * constants.
   *
   */

#ifndef PSAUXERR_H_
#define PSAUXERR_H_

#include <freetype/ftmoderr.h>

#undef FTERRORS_H_

#undef  FT_ERR_PREFIX
#define FT_ERR_PREFIX  PSaux_Err_
#define FT_ERR_BASE    FT_Mod_Err_PSaux

#include <freetype/fterrors.h>

#endif /* PSAUXERR_H_ */


/* END */
