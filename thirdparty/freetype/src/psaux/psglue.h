/***************************************************************************/
/*                                                                         */
/*  psglue.h                                                               */
/*                                                                         */
/*    Adobe's code for shared stuff (specification only).                  */
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


#ifndef PSGLUE_H_
#define PSGLUE_H_


/* common includes for other modules */
#include "pserror.h"
#include "psfixed.h"
#include "psarrst.h"
#include "psread.h"


FT_BEGIN_HEADER


  /* rendering parameters */

  /* apply hints to rendered glyphs */
#define CF2_FlagsHinted    1
  /* for testing */
#define CF2_FlagsDarkened  2

  /* type for holding the flags */
  typedef CF2_Int  CF2_RenderingFlags;


  /* elements of a glyph outline */
  typedef enum  CF2_PathOp_
  {
    CF2_PathOpMoveTo = 1,     /* change the current point */
    CF2_PathOpLineTo = 2,     /* line                     */
    CF2_PathOpQuadTo = 3,     /* quadratic curve          */
    CF2_PathOpCubeTo = 4      /* cubic curve              */

  } CF2_PathOp;


  /* a matrix of fixed point values */
  typedef struct  CF2_Matrix_
  {
    CF2_F16Dot16  a;
    CF2_F16Dot16  b;
    CF2_F16Dot16  c;
    CF2_F16Dot16  d;
    CF2_F16Dot16  tx;
    CF2_F16Dot16  ty;

  } CF2_Matrix;


  /* these typedefs are needed by more than one header file */
  /* and gcc compiler doesn't allow redefinition            */
  typedef struct CF2_FontRec_  CF2_FontRec, *CF2_Font;
  typedef struct CF2_HintRec_  CF2_HintRec, *CF2_Hint;


  /* A common structure for all callback parameters.                       */
  /*                                                                       */
  /* Some members may be unused.  For example, `pt0' is not used for       */
  /* `moveTo' and `pt3' is not used for `quadTo'.  The initial point `pt0' */
  /* is included for each path element for generality; curve conversions   */
  /* need it.  The `op' parameter allows one function to handle multiple   */
  /* element types.                                                        */

  typedef struct  CF2_CallbackParamsRec_
  {
    FT_Vector  pt0;
    FT_Vector  pt1;
    FT_Vector  pt2;
    FT_Vector  pt3;

    CF2_Int  op;

  } CF2_CallbackParamsRec, *CF2_CallbackParams;


  /* forward reference */
  typedef struct CF2_OutlineCallbacksRec_  CF2_OutlineCallbacksRec,
                                           *CF2_OutlineCallbacks;

  /* callback function pointers */
  typedef void
  (*CF2_Callback_Type)( CF2_OutlineCallbacks      callbacks,
                        const CF2_CallbackParams  params );


  struct  CF2_OutlineCallbacksRec_
  {
    CF2_Callback_Type  moveTo;
    CF2_Callback_Type  lineTo;
    CF2_Callback_Type  quadTo;
    CF2_Callback_Type  cubeTo;

    CF2_Int  windingMomentum;    /* for winding order detection */

    FT_Memory  memory;
    FT_Error*  error;
  };


FT_END_HEADER


#endif /* PSGLUE_H_ */


/* END */
