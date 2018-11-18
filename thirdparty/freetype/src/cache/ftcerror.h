/***************************************************************************/
/*                                                                         */
/*  ftcerror.h                                                             */
/*                                                                         */
/*    Caching sub-system error codes (specification only).                 */
/*                                                                         */
/*  Copyright 2001-2018 by                                                 */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


  /*************************************************************************/
  /*                                                                       */
  /* This file is used to define the caching sub-system error enumeration  */
  /* constants.                                                            */
  /*                                                                       */
  /*************************************************************************/

#ifndef FTCERROR_H_
#define FTCERROR_H_

#include FT_MODULE_ERRORS_H

#undef FTERRORS_H_

#undef  FT_ERR_PREFIX
#define FT_ERR_PREFIX  FTC_Err_
#define FT_ERR_BASE    FT_Mod_Err_Cache

#include FT_ERRORS_H

#endif /* FTCERROR_H_ */


/* END */
