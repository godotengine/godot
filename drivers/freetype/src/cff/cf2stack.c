/***************************************************************************/
/*                                                                         */
/*  cf2stack.c                                                             */
/*                                                                         */
/*    Adobe's code for emulating a CFF stack (body).                       */
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


#include "cf2ft.h"
#include FT_INTERNAL_DEBUG_H

#include "cf2glue.h"
#include "cf2font.h"
#include "cf2stack.h"

#include "cf2error.h"


  /* Allocate and initialize an instance of CF2_Stack.       */
  /* Note: This function returns NULL on error (does not set */
  /* `error').                                               */
  FT_LOCAL_DEF( CF2_Stack )
  cf2_stack_init( FT_Memory  memory,
                  FT_Error*  e )
  {
    FT_Error  error = FT_Err_Ok;     /* for FT_QNEW */

    CF2_Stack  stack = NULL;


    if ( !FT_QNEW( stack ) )
    {
      /* initialize the structure; FT_QNEW zeroes it */
      stack->memory = memory;
      stack->error  = e;
      stack->top    = &stack->buffer[0]; /* empty stack */
    }

    return stack;
  }


  FT_LOCAL_DEF( void )
  cf2_stack_free( CF2_Stack  stack )
  {
    if ( stack )
    {
      FT_Memory  memory = stack->memory;


      /* free the main structure */
      FT_FREE( stack );
    }
  }


  FT_LOCAL_DEF( CF2_UInt )
  cf2_stack_count( CF2_Stack  stack )
  {
    return (CF2_UInt)( stack->top - &stack->buffer[0] );
  }


  FT_LOCAL_DEF( void )
  cf2_stack_pushInt( CF2_Stack  stack,
                     CF2_Int    val )
  {
    if ( stack->top == &stack->buffer[CF2_OPERAND_STACK_SIZE] )
    {
      CF2_SET_ERROR( stack->error, Stack_Overflow );
      return;     /* stack overflow */
    }

    stack->top->u.i  = val;
    stack->top->type = CF2_NumberInt;
    ++stack->top;
  }


  FT_LOCAL_DEF( void )
  cf2_stack_pushFixed( CF2_Stack  stack,
                       CF2_Fixed  val )
  {
    if ( stack->top == &stack->buffer[CF2_OPERAND_STACK_SIZE] )
    {
      CF2_SET_ERROR( stack->error, Stack_Overflow );
      return;     /* stack overflow */
    }

    stack->top->u.r  = val;
    stack->top->type = CF2_NumberFixed;
    ++stack->top;
  }


  /* this function is only allowed to pop an integer type */
  FT_LOCAL_DEF( CF2_Int )
  cf2_stack_popInt( CF2_Stack  stack )
  {
    if ( stack->top == &stack->buffer[0] )
    {
      CF2_SET_ERROR( stack->error, Stack_Underflow );
      return 0;   /* underflow */
    }
    if ( stack->top[-1].type != CF2_NumberInt )
    {
      CF2_SET_ERROR( stack->error, Syntax_Error );
      return 0;   /* type mismatch */
    }

    --stack->top;

    return stack->top->u.i;
  }


  /* Note: type mismatch is silently cast */
  /* TODO: check this                     */
  FT_LOCAL_DEF( CF2_Fixed )
  cf2_stack_popFixed( CF2_Stack  stack )
  {
    if ( stack->top == &stack->buffer[0] )
    {
      CF2_SET_ERROR( stack->error, Stack_Underflow );
      return cf2_intToFixed( 0 );    /* underflow */
    }

    --stack->top;

    switch ( stack->top->type )
    {
    case CF2_NumberInt:
      return cf2_intToFixed( stack->top->u.i );
    case CF2_NumberFrac:
      return cf2_fracToFixed( stack->top->u.f );
    default:
      return stack->top->u.r;
    }
  }


  /* Note: type mismatch is silently cast */
  /* TODO: check this                     */
  FT_LOCAL_DEF( CF2_Fixed )
  cf2_stack_getReal( CF2_Stack  stack,
                     CF2_UInt   idx )
  {
    FT_ASSERT( cf2_stack_count( stack ) <= CF2_OPERAND_STACK_SIZE );

    if ( idx >= cf2_stack_count( stack ) )
    {
      CF2_SET_ERROR( stack->error, Stack_Overflow );
      return cf2_intToFixed( 0 );    /* bounds error */
    }

    switch ( stack->buffer[idx].type )
    {
    case CF2_NumberInt:
      return cf2_intToFixed( stack->buffer[idx].u.i );
    case CF2_NumberFrac:
      return cf2_fracToFixed( stack->buffer[idx].u.f );
    default:
      return stack->buffer[idx].u.r;
    }
  }


  FT_LOCAL( void )
  cf2_stack_roll( CF2_Stack  stack,
                  CF2_Int    count,
                  CF2_Int    shift )
  {
    /* we initialize this variable to avoid compiler warnings */
    CF2_StackNumber  last = { { 0 }, CF2_NumberInt };

    CF2_Int  start_idx, idx, i;


    if ( count < 2 )
      return; /* nothing to do (values 0 and 1), or undefined value */

    if ( (CF2_UInt)count > cf2_stack_count( stack ) )
    {
      CF2_SET_ERROR( stack->error, Stack_Overflow );
      return;
    }

    if ( shift < 0 )
      shift = -( ( -shift ) % count );
    else
      shift %= count;

    if ( shift == 0 )
      return; /* nothing to do */

    /* We use the following algorithm to do the rolling, */
    /* which needs two temporary variables only.         */
    /*                                                   */
    /* Example:                                          */
    /*                                                   */
    /*   count = 8                                       */
    /*   shift = 2                                       */
    /*                                                   */
    /*   stack indices before roll:  7 6 5 4 3 2 1 0     */
    /*   stack indices after roll:   1 0 7 6 5 4 3 2     */
    /*                                                   */
    /* The value of index 0 gets moved to index 2, while */
    /* the old value of index 2 gets moved to index 4,   */
    /* and so on.  We thus have the following copying    */
    /* chains for shift value 2.                         */
    /*                                                   */
    /*   0 -> 2 -> 4 -> 6 -> 0                           */
    /*   1 -> 3 -> 5 -> 7 -> 1                           */
    /*                                                   */
    /* If `count' and `shift' are incommensurable, we    */
    /* have a single chain only.  Otherwise, increase    */
    /* the start index by 1 after the first chain, then  */
    /* do the next chain until all elements in all       */
    /* chains are handled.                               */

    start_idx = -1;
    idx       = -1;
    for ( i = 0; i < count; i++ )
    {
      CF2_StackNumber  tmp;


      if ( start_idx == idx )
      {
        start_idx++;
        idx  = start_idx;
        last = stack->buffer[idx];
      }

      idx += shift;
      if ( idx >= count )
        idx -= count;
      else if ( idx < 0 )
        idx += count;

      tmp                = stack->buffer[idx];
      stack->buffer[idx] = last;
      last               = tmp;
    }
  }


  FT_LOCAL_DEF( void )
  cf2_stack_clear( CF2_Stack  stack )
  {
    stack->top = &stack->buffer[0];
  }


/* END */
