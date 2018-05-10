/***************************************************************************/
/*                                                                         */
/*  psarrst.c                                                              */
/*                                                                         */
/*    Adobe's code for Array Stacks (body).                                */
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


#include "psft.h"
#include FT_INTERNAL_DEBUG_H

#include "psglue.h"
#include "psarrst.h"

#include "pserror.h"


  /*
   * CF2_ArrStack uses an error pointer, to enable shared errors.
   * Shared errors are necessary when multiple objects allow the program
   * to continue after detecting errors.  Only the first error should be
   * recorded.
   */

  FT_LOCAL_DEF( void )
  cf2_arrstack_init( CF2_ArrStack  arrstack,
                     FT_Memory     memory,
                     FT_Error*     error,
                     size_t        sizeItem )
  {
    FT_ASSERT( arrstack );

    /* initialize the structure */
    arrstack->memory    = memory;
    arrstack->error     = error;
    arrstack->sizeItem  = sizeItem;
    arrstack->allocated = 0;
    arrstack->chunk     = 10;    /* chunks of 10 items */
    arrstack->count     = 0;
    arrstack->totalSize = 0;
    arrstack->ptr       = NULL;
  }


  FT_LOCAL_DEF( void )
  cf2_arrstack_finalize( CF2_ArrStack  arrstack )
  {
    FT_Memory  memory = arrstack->memory;     /* for FT_FREE */


    FT_ASSERT( arrstack );

    arrstack->allocated = 0;
    arrstack->count     = 0;
    arrstack->totalSize = 0;

    /* free the data buffer */
    FT_FREE( arrstack->ptr );
  }


  /* allocate or reallocate the buffer size; */
  /* return false on memory error */
  static FT_Bool
  cf2_arrstack_setNumElements( CF2_ArrStack  arrstack,
                               size_t        numElements )
  {
    FT_ASSERT( arrstack );

    {
      FT_Error   error  = FT_Err_Ok;        /* for FT_REALLOC */
      FT_Memory  memory = arrstack->memory; /* for FT_REALLOC */

      size_t  newSize = numElements * arrstack->sizeItem;


      if ( numElements > FT_LONG_MAX / arrstack->sizeItem )
        goto exit;


      FT_ASSERT( newSize > 0 );   /* avoid realloc with zero size */

      if ( !FT_REALLOC( arrstack->ptr, arrstack->totalSize, newSize ) )
      {
        arrstack->allocated = numElements;
        arrstack->totalSize = newSize;

        if ( arrstack->count > numElements )
        {
          /* we truncated the list! */
          CF2_SET_ERROR( arrstack->error, Stack_Overflow );
          arrstack->count = numElements;
          return FALSE;
        }

        return TRUE;     /* success */
      }
    }

  exit:
    /* if there's not already an error, store this one */
    CF2_SET_ERROR( arrstack->error, Out_Of_Memory );

    return FALSE;
  }


  /* set the count, ensuring allocation is sufficient */
  FT_LOCAL_DEF( void )
  cf2_arrstack_setCount( CF2_ArrStack  arrstack,
                         size_t        numElements )
  {
    FT_ASSERT( arrstack );

    if ( numElements > arrstack->allocated )
    {
      /* expand the allocation first */
      if ( !cf2_arrstack_setNumElements( arrstack, numElements ) )
        return;
    }

    arrstack->count = numElements;
  }


  /* clear the count */
  FT_LOCAL_DEF( void )
  cf2_arrstack_clear( CF2_ArrStack  arrstack )
  {
    FT_ASSERT( arrstack );

    arrstack->count = 0;
  }


  /* current number of items */
  FT_LOCAL_DEF( size_t )
  cf2_arrstack_size( const CF2_ArrStack  arrstack )
  {
    FT_ASSERT( arrstack );

    return arrstack->count;
  }


  FT_LOCAL_DEF( void* )
  cf2_arrstack_getBuffer( const CF2_ArrStack  arrstack )
  {
    FT_ASSERT( arrstack );

    return arrstack->ptr;
  }


  /* return pointer to the given element */
  FT_LOCAL_DEF( void* )
  cf2_arrstack_getPointer( const CF2_ArrStack  arrstack,
                           size_t              idx )
  {
    void*  newPtr;


    FT_ASSERT( arrstack );

    if ( idx >= arrstack->count )
    {
      /* overflow */
      CF2_SET_ERROR( arrstack->error, Stack_Overflow );
      idx = 0;    /* choose safe default */
    }

    newPtr = (FT_Byte*)arrstack->ptr + idx * arrstack->sizeItem;

    return newPtr;
  }


  /* push (append) an element at the end of the list;         */
  /* return false on memory error                             */
  /* TODO: should there be a length param for extra checking? */
  FT_LOCAL_DEF( void )
  cf2_arrstack_push( CF2_ArrStack  arrstack,
                     const void*   ptr )
  {
    FT_ASSERT( arrstack );

    if ( arrstack->count == arrstack->allocated )
    {
      /* grow the buffer by one chunk */
      if ( !cf2_arrstack_setNumElements(
             arrstack, arrstack->allocated + arrstack->chunk ) )
      {
        /* on error, ignore the push */
        return;
      }
    }

    FT_ASSERT( ptr );

    {
      size_t  offset = arrstack->count * arrstack->sizeItem;
      void*   newPtr = (FT_Byte*)arrstack->ptr + offset;


      FT_MEM_COPY( newPtr, ptr, arrstack->sizeItem );
      arrstack->count += 1;
    }
  }


/* END */
