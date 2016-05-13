/***************************************************************************/
/*                                                                         */
/*  t42error.h                                                             */
/*                                                                         */
/*    Type 42 error codes (specification only).                            */
/*                                                                         */
/*  Copyright 2002, 2003, 2012 by                                          */
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
  /* This file is used to define the Type 42 error enumeration constants.  */
  /*                                                                       */
  /*************************************************************************/

#ifndef __T42ERROR_H__
#define __T42ERROR_H__

#include FT_MODULE_ERRORS_H

#undef __FTERRORS_H__

#undef  FT_ERR_PREFIX
#define FT_ERR_PREFIX  T42_Err_
#define FT_ERR_BASE    FT_Mod_Err_Type42

#include FT_ERRORS_H

#endif /* __T42ERROR_H__ */


/* END */
