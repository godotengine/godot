/****************************************************************************
 *
 * afdummy.h
 *
 *   Auto-fitter dummy routines to be used if no hinting should be
 *   performed (specification).
 *
 * Copyright (C) 2003-2019 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#ifndef AFDUMMY_H_
#define AFDUMMY_H_

#include "aftypes.h"


FT_BEGIN_HEADER

  /* A dummy writing system used when no hinting should be performed. */

  AF_DECLARE_WRITING_SYSTEM_CLASS( af_dummy_writing_system_class )

/* */

FT_END_HEADER


#endif /* AFDUMMY_H_ */


/* END */
