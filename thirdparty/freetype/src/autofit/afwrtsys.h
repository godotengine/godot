/***************************************************************************/
/*                                                                         */
/*  afwrtsys.h                                                             */
/*                                                                         */
/*    Auto-fitter writing systems (specification only).                    */
/*                                                                         */
/*  Copyright 2013-2017 by                                                 */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


#ifndef AFWRTSYS_H_
#define AFWRTSYS_H_

  /* Since preprocessor directives can't create other preprocessor */
  /* directives, we have to include the header files manually.     */

#include "afdummy.h"
#include "aflatin.h"
#include "afcjk.h"
#include "afindic.h"
#ifdef FT_OPTION_AUTOFIT2
#include "aflatin2.h"
#endif

#endif /* AFWRTSYS_H_ */


  /* The following part can be included multiple times. */
  /* Define `WRITING_SYSTEM' as needed.                 */


  /* Add new writing systems here.  The arguments are the writing system */
  /* name in lowercase and uppercase, respectively.                      */

  WRITING_SYSTEM( dummy,  DUMMY  )
  WRITING_SYSTEM( latin,  LATIN  )
  WRITING_SYSTEM( cjk,    CJK    )
  WRITING_SYSTEM( indic,  INDIC  )
#ifdef FT_OPTION_AUTOFIT2
  WRITING_SYSTEM( latin2, LATIN2 )
#endif


/* END */
