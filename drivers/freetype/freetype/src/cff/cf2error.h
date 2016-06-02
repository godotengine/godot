/***************************************************************************/
/*                                                                         */
/*  cf2error.h                                                             */
/*                                                                         */
/*    Adobe's code for error handling (specification).                     */
/*                                                                         */
/*  Copyright 2006-2013 Adobe Systems Incorporated.                        */
/*                                                                         */
/*  This software, and all works of authorship, whether in source or       */
/*  object code form as indicated by the copyright notice(s) included      */
/*  herein (collectively, the "Work") is made available, and may only be   */
/*  used, modified, and distributed under the FreeType Project License,    */
/*  LICENSE.TXT.  Additionally, subject to the terms and conditions of the */
/*  FreeType Project License, each contributor to the Work hereby grants   */
/*  to any individual or legal entity exercising permissions granted by    */
/*  the FreeType Project License and this section (hereafter, "You" or     */
/*  "Your") a perpetual, worldwide, non-exclusive, no-charge,              */
/*  royalty-free, irrevocable (except as stated in this section) patent    */
/*  license to make, have made, use, offer to sell, sell, import, and      */
/*  otherwise transfer the Work, where such license applies only to those  */
/*  patent claims licensable by such contributor that are necessarily      */
/*  infringed by their contribution(s) alone or by combination of their    */
/*  contribution(s) with the Work to which such contribution(s) was        */
/*  submitted.  If You institute patent litigation against any entity      */
/*  (including a cross-claim or counterclaim in a lawsuit) alleging that   */
/*  the Work or a contribution incorporated within the Work constitutes    */
/*  direct or contributory patent infringement, then any patent licenses   */
/*  granted to You under this License for that Work shall terminate as of  */
/*  the date such litigation is filed.                                     */
/*                                                                         */
/*  By using, modifying, or distributing the Work you indicate that you    */
/*  have read and understood the terms and conditions of the               */
/*  FreeType Project License as well as those provided in this section,    */
/*  and you accept them fully.                                             */
/*                                                                         */
/***************************************************************************/


#ifndef __CF2ERROR_H__
#define __CF2ERROR_H__


#include FT_MODULE_ERRORS_H

#undef __FTERRORS_H__

#undef  FT_ERR_PREFIX
#define FT_ERR_PREFIX  CF2_Err_
#define FT_ERR_BASE    FT_Mod_Err_CF2


#include FT_ERRORS_H
#include "cf2ft.h"


FT_BEGIN_HEADER


  /*
   * A poor-man error facility.
   *
   * This code being written in vanilla C, doesn't have the luxury of a
   * language-supported exception mechanism such as the one available in
   * Java.  Instead, we are stuck with using error codes that must be
   * carefully managed and preserved.  However, it is convenient for us to
   * model our error mechanism on a Java-like exception mechanism.
   * When we assign an error code we are thus `throwing' an error.
   *
   * The perservation of an error code is done by coding convention.
   * Upon a function call if the error code is anything other than
   * `FT_Err_Ok', which is guaranteed to be zero, we
   * will return without altering that error.  This will allow the
   * error to propogate and be handled at the appropriate location in
   * the code.
   *
   * This allows a style of code where the error code is initialized
   * up front and a block of calls are made with the error code only
   * being checked after the block.  If a new error occurs, the original
   * error will be preserved and a functional no-op should result in any
   * subsequent function that has an initial error code not equal to
   * `FT_Err_Ok'.
   *
   * Errors are encoded by calling the `FT_THROW' macro.  For example,
   *
   * {
   *   FT_Error  e;
   *
   *
   *   ...
   *   e = FT_THROW( Out_Of_Memory );
   * }
   *
   */


  /* Set error code to a particular value. */
  FT_LOCAL( void )
  cf2_setError( FT_Error*  error,
                FT_Error   value );


  /*
   * A macro that conditionally sets an error code.
   *
   * This macro will first check whether `error' is set;
   * if not, it will set it to `e'.
   *
  */
#define CF2_SET_ERROR( error, e )              \
          cf2_setError( error, FT_THROW( e ) )


FT_END_HEADER


#endif /* __CF2ERROR_H__ */


/* END */
