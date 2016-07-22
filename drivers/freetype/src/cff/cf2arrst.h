/***************************************************************************/
/*                                                                         */
/*  cf2arrst.h                                                             */
/*                                                                         */
/*    Adobe's code for Array Stacks (specification).                       */
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


#ifndef CF2ARRST_H_
#define CF2ARRST_H_


#include "cf2error.h"


FT_BEGIN_HEADER


  /* need to define the struct here (not opaque) so it can be allocated by */
  /* clients                                                               */
  typedef struct  CF2_ArrStackRec_
  {
    FT_Memory  memory;
    FT_Error*  error;

    size_t  sizeItem;       /* bytes per element             */
    size_t  allocated;      /* items allocated               */
    size_t  chunk;          /* allocation increment in items */
    size_t  count;          /* number of elements allocated  */
    size_t  totalSize;      /* total bytes allocated         */

    void*  ptr;             /* ptr to data                   */

  } CF2_ArrStackRec, *CF2_ArrStack;


  FT_LOCAL( void )
  cf2_arrstack_init( CF2_ArrStack  arrstack,
                     FT_Memory     memory,
                     FT_Error*     error,
                     size_t        sizeItem );
  FT_LOCAL( void )
  cf2_arrstack_finalize( CF2_ArrStack  arrstack );

  FT_LOCAL( void )
  cf2_arrstack_setCount( CF2_ArrStack  arrstack,
                         size_t        numElements );
  FT_LOCAL( void )
  cf2_arrstack_clear( CF2_ArrStack  arrstack );
  FT_LOCAL( size_t )
  cf2_arrstack_size( const CF2_ArrStack  arrstack );

  FT_LOCAL( void* )
  cf2_arrstack_getBuffer( const CF2_ArrStack  arrstack );
  FT_LOCAL( void* )
  cf2_arrstack_getPointer( const CF2_ArrStack  arrstack,
                           size_t              idx );

  FT_LOCAL( void )
  cf2_arrstack_push( CF2_ArrStack  arrstack,
                     const void*   ptr );


FT_END_HEADER


#endif /* CF2ARRST_H_ */


/* END */
