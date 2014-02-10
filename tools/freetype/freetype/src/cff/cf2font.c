/***************************************************************************/
/*                                                                         */
/*  cf2font.c                                                              */
/*                                                                         */
/*    Adobe's code for font instances (body).                              */
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

#include "cf2glue.h"
#include "cf2font.h"
#include "cf2error.h"
#include "cf2intrp.h"


  /* Compute a stem darkening amount in character space. */
  static void
  cf2_computeDarkening( CF2_Fixed   emRatio,
                        CF2_Fixed   ppem,
                        CF2_Fixed   stemWidth,
                        CF2_Fixed*  darkenAmount,
                        CF2_Fixed   boldenAmount,
                        FT_Bool     stemDarkened )
  {
    /* Internal calculations are done in units per thousand for */
    /* convenience.                                             */
    CF2_Fixed  stemWidthPer1000, scaledStem;


    *darkenAmount = 0;

    if ( boldenAmount == 0 && !stemDarkened )
      return;

    /* protect against range problems and divide by zero */
    if ( emRatio < cf2_floatToFixed( .01 ) )
      return;

    if ( stemDarkened )
    {
      /* convert from true character space to 1000 unit character space; */
      /* add synthetic emboldening effect                                */

      /* we have to assure that the computation of `scaledStem' */
      /* and `stemWidthPer1000' don't overflow                  */

      stemWidthPer1000 = FT_MulFix( stemWidth + boldenAmount, emRatio );

      if ( emRatio > CF2_FIXED_ONE                          &&
           stemWidthPer1000 <= ( stemWidth + boldenAmount ) )
      {
        stemWidthPer1000 = 0;                      /* to pacify compiler */
        scaledStem       = cf2_intToFixed( 2333 );
      }
      else
      {
        scaledStem = FT_MulFix( stemWidthPer1000, ppem );

        if ( ppem > CF2_FIXED_ONE           &&
             scaledStem <= stemWidthPer1000 )
          scaledStem = cf2_intToFixed( 2333 );
      }

      /*
       * Total darkening amount is computed in 1000 unit character space
       * using the modified 5 part curve as Avalon rasterizer.
       * The darkening amount is smaller for thicker stems.
       * It becomes zero when the stem is thicker than 2.333 pixels.
       *
       * In Avalon rasterizer,
       *
       *   darkenAmount = 0.5 pixels   if scaledStem <= 0.5 pixels,
       *   darkenAmount = 0.333 pixels if 1 <= scaledStem <= 1.667 pixels,
       *   darkenAmount = 0 pixel      if scaledStem >= 2.333 pixels,
       *
       * and piecewise linear in-between.
       *
       */
      if ( scaledStem < cf2_intToFixed( 500 ) )
        *darkenAmount = FT_DivFix( cf2_intToFixed( 400 ), ppem );

      else if ( scaledStem < cf2_intToFixed( 1000 ) )
        *darkenAmount = FT_DivFix( cf2_intToFixed( 525 ), ppem ) -
                          FT_MulFix( stemWidthPer1000,
                                     cf2_floatToFixed( .25 ) );

      else if ( scaledStem < cf2_intToFixed( 1667 ) )
        *darkenAmount = FT_DivFix( cf2_intToFixed( 275 ), ppem );

      else if ( scaledStem < cf2_intToFixed( 2333 ) )
        *darkenAmount = FT_DivFix( cf2_intToFixed( 963 ), ppem ) -
                          FT_MulFix( stemWidthPer1000,
                                     cf2_floatToFixed( .413 ) );

      /* use half the amount on each side and convert back to true */
      /* character space                                           */
      *darkenAmount = FT_DivFix( *darkenAmount, 2 * emRatio );
    }

    /* add synthetic emboldening effect in character space */
    *darkenAmount += boldenAmount / 2;
  }


  /* set up values for the current FontDict and matrix */

  /* caller's transform is adjusted for subpixel positioning */
  static void
  cf2_font_setup( CF2_Font           font,
                  const CF2_Matrix*  transform )
  {
    /* pointer to parsed font object */
    CFF_Decoder*  decoder = font->decoder;

    FT_Bool  needExtraSetup;

    /* character space units */
    CF2_Fixed  boldenX = font->syntheticEmboldeningAmountX;
    CF2_Fixed  boldenY = font->syntheticEmboldeningAmountY;

    CF2_Fixed  ppem;


    /* clear previous error */
    font->error = FT_Err_Ok;

    /* if a CID fontDict has changed, we need to recompute some cached */
    /* data                                                            */
    needExtraSetup =
      (FT_Bool)( font->lastSubfont != cf2_getSubfont( decoder ) );

    /* if ppem has changed, we need to recompute some cached data         */
    /* note: because of CID font matrix concatenation, ppem and transform */
    /*       do not necessarily track.                                    */
    ppem = cf2_getPpemY( decoder );
    if ( font->ppem != ppem )
    {
      font->ppem     = ppem;
      needExtraSetup = TRUE;
    }

    /* copy hinted flag on each call */
    font->hinted = (FT_Bool)( font->renderingFlags & CF2_FlagsHinted );

    /* determine if transform has changed;       */
    /* include Fontmatrix but ignore translation */
    if ( ft_memcmp( transform,
                    &font->currentTransform,
                    4 * sizeof ( CF2_Fixed ) ) != 0 )
    {
      /* save `key' information for `cache of one' matrix data; */
      /* save client transform, without the translation         */
      font->currentTransform    = *transform;
      font->currentTransform.tx =
      font->currentTransform.ty = cf2_intToFixed( 0 );

      /* TODO: FreeType transform is simple scalar; for now, use identity */
      /*       for outer                                                  */
      font->innerTransform   = *transform;
      font->outerTransform.a =
      font->outerTransform.d = cf2_intToFixed( 1 );
      font->outerTransform.b =
      font->outerTransform.c = cf2_intToFixed( 0 );

      needExtraSetup = TRUE;
    }

    /*
     * font->darkened is set to true if there is a stem darkening request or
     * the font is synthetic emboldened.
     * font->darkened controls whether to adjust blue zones, winding order,
     * and hinting.
     *
     */
    if ( font->stemDarkened != ( font->renderingFlags & CF2_FlagsDarkened ) )
    {
      font->stemDarkened =
        (FT_Bool)( font->renderingFlags & CF2_FlagsDarkened );

      /* blue zones depend on darkened flag */
      needExtraSetup = TRUE;
    }

    /* recompute variables that are dependent on transform or FontDict or */
    /* darken flag                                                        */
    if ( needExtraSetup )
    {
      /* StdVW is found in the private dictionary;                       */
      /* recompute darkening amounts whenever private dictionary or      */
      /* transform change                                                */
      /* Note: a rendering flag turns darkening on or off, so we want to */
      /*       store the `on' amounts;                                   */
      /*       darkening amount is computed in character space           */
      /* TODO: testing size-dependent darkening here;                    */
      /*       what to do for rotations?                                 */

      CF2_Fixed  emRatio;
      CF2_Fixed  stdHW;
      CF2_Int    unitsPerEm = font->unitsPerEm;


      if ( unitsPerEm == 0 )
        unitsPerEm = 1000;

      ppem = FT_MAX( cf2_intToFixed( 4 ),
                     font->ppem ); /* use minimum ppem of 4 */

#if 0
      /* since vstem is measured in the x-direction, we use the `a' member */
      /* of the fontMatrix                                                 */
      emRatio = cf2_fixedFracMul( cf2_intToFixed( 1000 ), fontMatrix->a );
#endif

      /* Freetype does not preserve the fontMatrix when parsing; use */
      /* unitsPerEm instead.                                         */
      /* TODO: check precision of this                               */
      emRatio     = cf2_intToFixed( 1000 ) / unitsPerEm;
      font->stdVW = cf2_getStdVW( decoder );

      if ( font->stdVW <= 0 )
        font->stdVW = FT_DivFix( cf2_intToFixed( 75 ), emRatio );

      if ( boldenX > 0 )
      {
        /* Ensure that boldenX is at least 1 pixel for synthetic bold font */
        /* (similar to what Avalon does)                                   */
        boldenX = FT_MAX( boldenX,
                          FT_DivFix( cf2_intToFixed( unitsPerEm ), ppem ) );

        /* Synthetic emboldening adds at least 1 pixel to darkenX, while */
        /* stem darkening adds at most half pixel.  Since the purpose of */
        /* stem darkening (readability at small sizes) is met with       */
        /* synthetic emboldening, no need to add stem darkening for a    */
        /* synthetic bold font.                                          */
        cf2_computeDarkening( emRatio,
                              ppem,
                              font->stdVW,
                              &font->darkenX,
                              boldenX,
                              FALSE );
      }
      else
        cf2_computeDarkening( emRatio,
                              ppem,
                              font->stdVW,
                              &font->darkenX,
                              0,
                              font->stemDarkened );

#if 0
      /* since hstem is measured in the y-direction, we use the `d' member */
      /* of the fontMatrix                                                 */
      /* TODO: use the same units per em as above; check this              */
      emRatio = cf2_fixedFracMul( cf2_intToFixed( 1000 ), fontMatrix->d );
#endif

      /* set the default stem width, because it must be the same for all */
      /* family members;                                                 */
      /* choose a constant for StdHW that depends on font contrast       */
      stdHW = cf2_getStdHW( decoder );

      if ( stdHW > 0 && font->stdVW > 2 * stdHW )
        font->stdHW = FT_DivFix( cf2_intToFixed( 75 ), emRatio );
      else
      {
        /* low contrast font gets less hstem darkening */
        font->stdHW = FT_DivFix( cf2_intToFixed( 110 ), emRatio );
      }

      cf2_computeDarkening( emRatio,
                            ppem,
                            font->stdHW,
                            &font->darkenY,
                            boldenY,
                            font->stemDarkened );

      if ( font->darkenX != 0 || font->darkenY != 0 )
        font->darkened = TRUE;
      else
        font->darkened = FALSE;

      font->reverseWinding = FALSE; /* initial expectation is CCW */

      /* compute blue zones for this instance */
      cf2_blues_init( &font->blues, font );
    }
  }


  /* equivalent to AdobeGetOutline */
  FT_LOCAL_DEF( FT_Error )
  cf2_getGlyphOutline( CF2_Font           font,
                       CF2_Buffer         charstring,
                       const CF2_Matrix*  transform,
                       CF2_F16Dot16*      glyphWidth )
  {
    FT_Error  lastError = FT_Err_Ok;

    FT_Vector  translation;

#if 0
    FT_Vector  advancePoint;
#endif

    CF2_Fixed  advWidth = 0;
    FT_Bool    needWinding;


    /* Note: use both integer and fraction for outlines.  This allows bbox */
    /*       to come out directly.                                         */

    translation.x = transform->tx;
    translation.y = transform->ty;

    /* set up values based on transform */
    cf2_font_setup( font, transform );
    if ( font->error )
      goto exit;                      /* setup encountered an error */

    /* reset darken direction */
    font->reverseWinding = FALSE;

    /* winding order only affects darkening */
    needWinding = font->darkened;

    while ( 1 )
    {
      /* reset output buffer */
      cf2_outline_reset( &font->outline );

      /* build the outline, passing the full translation */
      cf2_interpT2CharString( font,
                              charstring,
                              (CF2_OutlineCallbacks)&font->outline,
                              &translation,
                              FALSE,
                              0,
                              0,
                              &advWidth );

      if ( font->error )
        goto exit;

      if ( !needWinding )
        break;

      /* check winding order */
      if ( font->outline.root.windingMomentum >= 0 ) /* CFF is CCW */
        break;

      /* invert darkening and render again                            */
      /* TODO: this should be a parameter to getOutline-computeOffset */
      font->reverseWinding = TRUE;

      needWinding = FALSE;    /* exit after next iteration */
    }

    /* finish storing client outline */
    cf2_outline_close( &font->outline );

  exit:
    /* FreeType just wants the advance width; there is no translation */
    *glyphWidth = advWidth;

    /* free resources and collect errors from objects we've used */
    cf2_setError( &font->error, lastError );

    return font->error;
  }


/* END */
