/***************************************************************************/
/*                                                                         */
/*  cf2font.h                                                              */
/*                                                                         */
/*    Adobe's code for font instances (specification).                     */
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


#ifndef __CF2FONT_H__
#define __CF2FONT_H__


#include "cf2ft.h"
#include "cf2blues.h"


FT_BEGIN_HEADER


#define CF2_OPERAND_STACK_SIZE  48
#define CF2_MAX_SUBR            10 /* maximum subroutine nesting */


  /* typedef is in `cf2glue.h' */
  struct  CF2_FontRec_
  {
    FT_Memory  memory;
    FT_Error   error;     /* shared error for this instance */

    CF2_RenderingFlags  renderingFlags;

    /* variables that depend on Transform:  */
    /* the following have zero translation; */
    /* inner * outer = font * original      */

    CF2_Matrix  currentTransform;  /* original client matrix           */
    CF2_Matrix  innerTransform;    /* for hinting; erect, scaled       */
    CF2_Matrix  outerTransform;    /* post hinting; includes rotations */
    CF2_Fixed   ppem;              /* transform-dependent              */

    CF2_Int  unitsPerEm;

    CF2_Fixed  syntheticEmboldeningAmountX;   /* character space units */
    CF2_Fixed  syntheticEmboldeningAmountY;   /* character space units */

    /* FreeType related members */
    CF2_OutlineRec  outline;       /* freetype glyph outline functions */
    CFF_Decoder*    decoder;
    CFF_SubFont     lastSubfont;              /* FreeType parsed data; */
                                              /* top font or subfont   */

    /* these flags can vary from one call to the next */
    FT_Bool  hinted;
    FT_Bool  darkened;       /* true if stemDarkened or synthetic bold */
                             /* i.e. darkenX != 0 || darkenY != 0      */
    FT_Bool  stemDarkened;

    /* variables that depend on both FontDict and Transform */
    CF2_Fixed  stdVW;     /* in character space; depends on dict entry */
    CF2_Fixed  stdHW;     /* in character space; depends on dict entry */
    CF2_Fixed  darkenX;                    /* character space units    */
    CF2_Fixed  darkenY;                    /* depends on transform     */
                                           /* and private dict (StdVW) */
    FT_Bool  reverseWinding;               /* darken assuming          */
                                           /* counterclockwise winding */

    CF2_BluesRec  blues;                         /* computed zone data */
  };


  FT_LOCAL( FT_Error )
  cf2_getGlyphOutline( CF2_Font           font,
                       CF2_Buffer         charstring,
                       const CF2_Matrix*  transform,
                       CF2_F16Dot16*      glyphWidth );


FT_END_HEADER


#endif /* __CF2FONT_H__ */


/* END */
