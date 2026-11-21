/****************************************************************************
 *
 * hvferror.h
 *
 *   HVF error codes (specification only).
 *
 * Copyright (C) 2025-2026 by
 * Apple Inc.
 * written by Deborah Goldsmith <goldsmit@apple.com>
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
   * This file is used to define the HVF error enumeration constants.
   *
   */

#ifndef HVFERROR_H_
#define HVFERROR_H_

#include <freetype/ftmoderr.h>

#undef FTERRORS_H_

#undef  FT_ERR_PREFIX
#define FT_ERR_PREFIX  HVF_Err_
#define FT_ERR_BASE    FT_Mod_Err_HVF


#include <freetype/fterrors.h>

#endif /* HVFERROR_H_ */


/* END */
