/***************************************************************************/
/*                                                                         */
/*  ciderrs.h                                                              */
/*                                                                         */
/*    CID error codes (specification only).                                */
/*                                                                         */
/*  Copyright 2001-2017 by                                                 */
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
  /* This file is used to define the CID error enumeration constants.      */
  /*                                                                       */
  /*************************************************************************/

#ifndef CIDERRS_H_
#define CIDERRS_H_

#include FT_MODULE_ERRORS_H

#undef FTERRORS_H_

#undef  FT_ERR_PREFIX
#define FT_ERR_PREFIX  CID_Err_
#define FT_ERR_BASE    FT_Mod_Err_CID

#include FT_ERRORS_H

#endif /* CIDERRS_H_ */


/* END */
