/****************************************************************************
 *
 * ftsdferrs.h
 *
 *   Signed Distance Field error codes (specification only).
 *
 * Copyright (C) 2020-2023 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * Written by Anuj Verma.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#ifndef FTSDFERRS_H_
#define FTSDFERRS_H_

#include <freetype/ftmoderr.h>

#undef FTERRORS_H_

#undef  FT_ERR_PREFIX
#define FT_ERR_PREFIX  Sdf_Err_
#define FT_ERR_BASE    FT_Mod_Err_Sdf

#include <freetype/fterrors.h>

#endif /* FTSDFERRS_H_ */


/* END */
