/***************************************************************************/
/*                                                                         */
/*  cf2fixed.h                                                             */
/*                                                                         */
/*    Adobe's code for Fixed Point Mathematics (specification only).       */
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


#ifndef CF2FIXED_H_
#define CF2FIXED_H_


FT_BEGIN_HEADER


  /* rasterizer integer and fixed point arithmetic must be 32-bit */

#define   CF2_Fixed  CF2_F16Dot16
  typedef FT_Int32   CF2_Frac;   /* 2.30 fixed point */


#define CF2_FIXED_MAX      ( (CF2_Fixed)0x7FFFFFFFL )
#define CF2_FIXED_MIN      ( (CF2_Fixed)0x80000000L )
#define CF2_FIXED_ONE      ( (CF2_Fixed)0x10000L )
#define CF2_FIXED_EPSILON  ( (CF2_Fixed)0x0001 )

  /* in C 89, left and right shift of negative numbers is  */
  /* implementation specific behaviour in the general case */

#define cf2_intToFixed( i )                                              \
          ( (CF2_Fixed)( (FT_UInt32)(i) << 16 ) )
#define cf2_fixedToInt( x )                                              \
          ( (FT_Short)( ( (FT_UInt32)(x) + 0x8000U ) >> 16 ) )
#define cf2_fixedRound( x )                                              \
          ( (CF2_Fixed)( ( (FT_UInt32)(x) + 0x8000U ) & 0xFFFF0000UL ) )
#define cf2_doubleToFixed( f )                                           \
          ( (CF2_Fixed)( (f) * 65536.0 + 0.5 ) )
#define cf2_fixedAbs( x )                                                \
          ( (x) < 0 ? NEG_INT32( x ) : (x) )
#define cf2_fixedFloor( x )                                              \
          ( (CF2_Fixed)( (FT_UInt32)(x) & 0xFFFF0000UL ) )
#define cf2_fixedFraction( x )                                           \
          ( (x) - cf2_fixedFloor( x ) )
#define cf2_fracToFixed( x )                                             \
          ( (x) < 0 ? -( ( -(x) + 0x2000 ) >> 14 )                       \
                    :  ( (  (x) + 0x2000 ) >> 14 ) )


  /* signed numeric types */
  typedef enum  CF2_NumberType_
  {
    CF2_NumberFixed,    /* 16.16 */
    CF2_NumberFrac,     /*  2.30 */
    CF2_NumberInt       /* 32.0  */

  } CF2_NumberType;


FT_END_HEADER


#endif /* CF2FIXED_H_ */


/* END */
