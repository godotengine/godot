/***************************************************************************/
/*                                                                         */
/*  psintrp.h                                                              */
/*                                                                         */
/*    Adobe's CFF Interpreter (specification).                             */
/*                                                                         */
/*  Copyright 2007-2013 Adobe Systems Incorporated.                        */
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


#ifndef PSINTRP_H_
#define PSINTRP_H_


#include "psft.h"
#include "pshints.h"


FT_BEGIN_HEADER


  FT_LOCAL( void )
  cf2_hintmask_init( CF2_HintMask  hintmask,
                     FT_Error*     error );
  FT_LOCAL( FT_Bool )
  cf2_hintmask_isValid( const CF2_HintMask  hintmask );
  FT_LOCAL( FT_Bool )
  cf2_hintmask_isNew( const CF2_HintMask  hintmask );
  FT_LOCAL( void )
  cf2_hintmask_setNew( CF2_HintMask  hintmask,
                       FT_Bool       val );
  FT_LOCAL( FT_Byte* )
  cf2_hintmask_getMaskPtr( CF2_HintMask  hintmask );
  FT_LOCAL( void )
  cf2_hintmask_setAll( CF2_HintMask  hintmask,
                       size_t        bitCount );

  FT_LOCAL( void )
  cf2_interpT2CharString( CF2_Font              font,
                          CF2_Buffer            charstring,
                          CF2_OutlineCallbacks  callbacks,
                          const FT_Vector*      translation,
                          FT_Bool               doingSeac,
                          CF2_Fixed             curX,
                          CF2_Fixed             curY,
                          CF2_Fixed*            width );


FT_END_HEADER


#endif /* PSINTRP_H_ */


/* END */
