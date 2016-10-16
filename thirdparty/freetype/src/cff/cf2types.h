/***************************************************************************/
/*                                                                         */
/*  cf2types.h                                                             */
/*                                                                         */
/*    Adobe's code for defining data types (specification only).           */
/*                                                                         */
/*  Copyright 2011-2013 Adobe Systems Incorporated.                        */
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


#ifndef CF2TYPES_H_
#define CF2TYPES_H_

#include <ft2build.h>
#include FT_FREETYPE_H


FT_BEGIN_HEADER


  /*
   * The data models that we expect to support are as follows:
   *
   *   name  char short int long long-long pointer example
   *  -----------------------------------------------------
   *   ILP32  8    16    32  32     64*      32    32-bit MacOS, x86
   *   LLP64  8    16    32  32     64       64    x64
   *   LP64   8    16    32  64     64       64    64-bit MacOS
   *
   *    *) type may be supported by emulation on a 32-bit architecture
   *
   */


  /* integers at least 32 bits wide */
#define CF2_UInt  FT_UFast
#define CF2_Int   FT_Fast


  /* fixed-float numbers */
  typedef FT_Int32  CF2_F16Dot16;


FT_END_HEADER


#endif /* CF2TYPES_H_ */


/* END */
