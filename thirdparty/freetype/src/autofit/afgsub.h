/****************************************************************************
 *
 * afgsub.h
 *
 *   Auto-fitter routines to parse the GSUB table (header).
 *
 * Copyright (C) 2025 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */

#ifndef AFGSUB_H_
#define AFGSUB_H_

#include "afglobal.h"


FT_BEGIN_HEADER

  FT_LOCAL( void )
  af_parse_gsub( AF_FaceGlobals  globals );

  FT_LOCAL( FT_Error )
  af_map_lookup( AF_FaceGlobals  globals,
                 FT_Hash         map,
                 FT_UInt32       lookup_offset );

FT_END_HEADER

#endif /* AFGSUB_H_ */

/* END */
