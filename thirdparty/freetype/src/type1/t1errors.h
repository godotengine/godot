/****************************************************************************
 *
 * t1errors.h
 *
 *   Type 1 error codes (specification only).
 *
 * Copyright (C) 2001-2019 by
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
   * This file is used to define the Type 1 error enumeration constants.
   *
   */

#ifndef T1ERRORS_H_
#define T1ERRORS_H_

#include FT_MODULE_ERRORS_H

#undef FTERRORS_H_

#undef  FT_ERR_PREFIX
#define FT_ERR_PREFIX  T1_Err_
#define FT_ERR_BASE    FT_Mod_Err_Type1

#include FT_ERRORS_H

#endif /* T1ERRORS_H_ */


/* END */
