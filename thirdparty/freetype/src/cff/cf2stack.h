/***************************************************************************/
/*                                                                         */
/*  cf2stack.h                                                             */
/*                                                                         */
/*    Adobe's code for emulating a CFF stack (specification).              */
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


#ifndef CF2STACK_H_
#define CF2STACK_H_


FT_BEGIN_HEADER


  /* CFF operand stack; specified maximum of 48 or 192 values */
  typedef struct  CF2_StackNumber_
  {
    union
    {
      CF2_Fixed  r;      /* 16.16 fixed point */
      CF2_Frac   f;      /* 2.30 fixed point (for font matrix) */
      CF2_Int    i;
    } u;

    CF2_NumberType  type;

  } CF2_StackNumber;


  typedef struct  CF2_StackRec_
  {
    FT_Memory         memory;
    FT_Error*         error;
    CF2_StackNumber*  buffer;
    CF2_StackNumber*  top;
    FT_UInt           stackSize;

  } CF2_StackRec, *CF2_Stack;


  FT_LOCAL( CF2_Stack )
  cf2_stack_init( FT_Memory  memory,
                  FT_Error*  error,
                  FT_UInt    stackSize );
  FT_LOCAL( void )
  cf2_stack_free( CF2_Stack  stack );

  FT_LOCAL( CF2_UInt )
  cf2_stack_count( CF2_Stack  stack );

  FT_LOCAL( void )
  cf2_stack_pushInt( CF2_Stack  stack,
                     CF2_Int    val );
  FT_LOCAL( void )
  cf2_stack_pushFixed( CF2_Stack  stack,
                       CF2_Fixed  val );

  FT_LOCAL( CF2_Int )
  cf2_stack_popInt( CF2_Stack  stack );
  FT_LOCAL( CF2_Fixed )
  cf2_stack_popFixed( CF2_Stack  stack );

  FT_LOCAL( CF2_Fixed )
  cf2_stack_getReal( CF2_Stack  stack,
                     CF2_UInt   idx );
  FT_LOCAL( void )
  cf2_stack_setReal( CF2_Stack  stack,
                     CF2_UInt   idx,
                     CF2_Fixed  val );

  FT_LOCAL( void )
  cf2_stack_pop( CF2_Stack  stack,
                 CF2_UInt   num );

  FT_LOCAL( void )
  cf2_stack_roll( CF2_Stack  stack,
                  CF2_Int    count,
                  CF2_Int    idx );

  FT_LOCAL( void )
  cf2_stack_clear( CF2_Stack  stack );


FT_END_HEADER


#endif /* CF2STACK_H_ */


/* END */
