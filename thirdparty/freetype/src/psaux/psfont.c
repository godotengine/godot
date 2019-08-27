/****************************************************************************
 *
 * psfont.c
 *
 *   Adobe's code for font instances (body).
 *
 * Copyright 2007-2014 Adobe Systems Incorporated.
 *
 * This software, and all works of authorship, whether in source or
 * object code form as indicated by the copyright notice(s) included
 * herein (collectively, the "Work") is made available, and may only be
 * used, modified, and distributed under the FreeType Project License,
 * LICENSE.TXT.  Additionally, subject to the terms and conditions of the
 * FreeType Project License, each contributor to the Work hereby grants
 * to any individual or legal entity exercising permissions granted by
 * the FreeType Project License and this section (hereafter, "You" or
 * "Your") a perpetual, worldwide, non-exclusive, no-charge,
 * royalty-free, irrevocable (except as stated in this section) patent
 * license to make, have made, use, offer to sell, sell, import, and
 * otherwise transfer the Work, where such license applies only to those
 * patent claims licensable by such contributor that are necessarily
 * infringed by their contribution(s) alone or by combination of their
 * contribution(s) with the Work to which such contribution(s) was
 * submitted.  If You institute patent litigation against any entity
 * (including a cross-claim or counterclaim in a lawsuit) alleging that
 * the Work or a contribution incorporated within the Work constitutes
 * direct or contributory patent infringement, then any patent licenses
 * granted to You under this License for that Work shall terminate as of
 * the date such litigation is filed.
 *
 * By using, modifying, or distributing the Work you indicate that you
 * have read and understood the terms and conditions of the
 * FreeType Project License as well as those provided in this section,
 * and you accept them fully.
 *
 */


#include <ft2build.h>
#include FT_INTERNAL_CALC_H

#include "psft.h"

#include "psglue.h"
#include "psfont.h"
#include "pserror.h"
#include "psintrp.h"


  /* Compute a stem darkening amount in character space. */
  static void
  cf2_computeDarkening( CF2_Fixed   emRatio,
                        CF2_Fixed   ppem,
                        CF2_Fixed   stemWidth,
                        CF2_Fixed*  darkenAmount,
                        CF2_Fixed   boldenAmount,
                        FT_Bool     stemDarkened,
                        FT_Int*     darkenParams )
  {
    /*
     * Total darkening amount is computed in 1000 unit character space
     * using the modified 5 part curve as Adobe's Avalon rasterizer.
     * The darkening amount is smaller for thicker stems.
     * It becomes zero when the stem is thicker than 2.333 pixels.
     *
     * By default, we use
     *
     *   darkenAmount = 0.4 pixels   if scaledStem <= 0.5 pixels,
     *   darkenAmount = 0.275 pixels if 1 <= scaledStem <= 1.667 pixels,
     *   darkenAmount = 0 pixel      if scaledStem >= 2.333 pixels,
     *
     * and piecewise linear in-between:
     *
     *
     *   darkening
     *       ^
     *       |
     *       |      (x1,y1)
     *       |--------+
     *       |         \
     *       |          \
     *       |           \          (x3,y3)
     *       |            +----------+
     *       |        (x2,y2)         \
     *       |                         \
     *       |                          \
     *       |                           +-----------------
     *       |                         (x4,y4)
     *       +--------------------------------------------->   stem
     *                                                       thickness
     *
     *
     * This corresponds to the following values for the
     * `darkening-parameters' property:
     *
     *   (x1, y1) = (500, 400)
     *   (x2, y2) = (1000, 275)
     *   (x3, y3) = (1667, 275)
     *   (x4, y4) = (2333, 0)
     *
     */

    /* Internal calculations are done in units per thousand for */
    /* convenience. The x axis is scaled stem width in          */
    /* thousandths of a pixel. That is, 1000 is 1 pixel.        */
    /* The y axis is darkening amount in thousandths of a pixel.*/
    /* In the code, below, dividing by ppem and                 */
    /* adjusting for emRatio converts darkenAmount to character */
    /* space (font units).                                      */
    CF2_Fixed  stemWidthPer1000, scaledStem;
    FT_Int     logBase2;


    *darkenAmount = 0;

    if ( boldenAmount == 0 && !stemDarkened )
      return;

    /* protect against range problems and divide by zero */
    if ( emRatio < cf2_doubleToFixed( .01 ) )
      return;

    if ( stemDarkened )
    {
      FT_Int  x1 = darkenParams[0];
      FT_Int  y1 = darkenParams[1];
      FT_Int  x2 = darkenParams[2];
      FT_Int  y2 = darkenParams[3];
      FT_Int  x3 = darkenParams[4];
      FT_Int  y3 = darkenParams[5];
      FT_Int  x4 = darkenParams[6];
      FT_Int  y4 = darkenParams[7];


      /* convert from true character space to 1000 unit character space; */
      /* add synthetic emboldening effect                                */

      /* `stemWidthPer1000' will not overflow for a legitimate font      */

      stemWidthPer1000 = FT_MulFix( stemWidth + boldenAmount, emRatio );

      /* `scaledStem' can easily overflow, so we must clamp its maximum  */
      /* value; the test doesn't need to be precise, but must be         */
      /* conservative.  The clamp value (default 2333) where             */
      /* `darkenAmount' is zero is well below the overflow value of      */
      /* 32767.                                                          */
      /*                                                                 */
      /* FT_MSB computes the integer part of the base 2 logarithm.  The  */
      /* number of bits for the product is 1 or 2 more than the sum of   */
      /* logarithms; remembering that the 16 lowest bits of the fraction */
      /* are dropped this is correct to within a factor of almost 4.     */
      /* For example, 0x80.0000 * 0x80.0000 = 0x4000.0000 is 23+23 and   */
      /* is flagged as possible overflow because 0xFF.FFFF * 0xFF.FFFF = */
      /* 0xFFFF.FE00 is also 23+23.                                      */

      logBase2 = FT_MSB( (FT_UInt32)stemWidthPer1000 ) +
                   FT_MSB( (FT_UInt32)ppem );

      if ( logBase2 >= 46 )
        /* possible overflow */
        scaledStem = cf2_intToFixed( x4 );
      else
        scaledStem = FT_MulFix( stemWidthPer1000, ppem );

      /* now apply the darkening parameters */

      if ( scaledStem < cf2_intToFixed( x1 ) )
        *darkenAmount = FT_DivFix( cf2_intToFixed( y1 ), ppem );

      else if ( scaledStem < cf2_intToFixed( x2 ) )
      {
        FT_Int  xdelta = x2 - x1;
        FT_Int  ydelta = y2 - y1;
        FT_Int  x      = stemWidthPer1000 -
                           FT_DivFix( cf2_intToFixed( x1 ), ppem );


        if ( !xdelta )
          goto Try_x3;

        *darkenAmount = FT_MulDiv( x, ydelta, xdelta ) +
                          FT_DivFix( cf2_intToFixed( y1 ), ppem );
      }

      else if ( scaledStem < cf2_intToFixed( x3 ) )
      {
      Try_x3:
        {
          FT_Int  xdelta = x3 - x2;
          FT_Int  ydelta = y3 - y2;
          FT_Int  x      = stemWidthPer1000 -
                             FT_DivFix( cf2_intToFixed( x2 ), ppem );


          if ( !xdelta )
            goto Try_x4;

          *darkenAmount = FT_MulDiv( x, ydelta, xdelta ) +
                            FT_DivFix( cf2_intToFixed( y2 ), ppem );
        }
      }

      else if ( scaledStem < cf2_intToFixed( x4 ) )
      {
      Try_x4:
        {
          FT_Int  xdelta = x4 - x3;
          FT_Int  ydelta = y4 - y3;
          FT_Int  x      = stemWidthPer1000 -
                             FT_DivFix( cf2_intToFixed( x3 ), ppem );


          if ( !xdelta )
            goto Use_y4;

          *darkenAmount = FT_MulDiv( x, ydelta, xdelta ) +
                            FT_DivFix( cf2_intToFixed( y3 ), ppem );
        }
      }

      else
      {
      Use_y4:
        *darkenAmount = FT_DivFix( cf2_intToFixed( y4 ), ppem );
      }

      /* use half the amount on each side and convert back to true */
      /* character space                                           */
      *darkenAmount = FT_DivFix( *darkenAmount, 2 * emRatio );
    }

    /* add synthetic emboldening effect in character space */
    *darkenAmount += boldenAmount / 2;
  }


  /* set up values for the current FontDict and matrix; */
  /* called for each glyph to be rendered               */

  /* caller's transform is adjusted for subpixel positioning */
  static void
  cf2_font_setup( CF2_Font           font,
                  const CF2_Matrix*  transform )
  {
    /* pointer to parsed font object */
    PS_Decoder*  decoder = font->decoder;

    FT_Bool  needExtraSetup = FALSE;

    CFF_VStoreRec*  vstore;
    FT_Bool         hasVariations = FALSE;

    /* character space units */
    CF2_Fixed  boldenX = font->syntheticEmboldeningAmountX;
    CF2_Fixed  boldenY = font->syntheticEmboldeningAmountY;

    CFF_SubFont  subFont;
    CF2_Fixed    ppem;

    CF2_UInt   lenNormalizedV = 0;
    FT_Fixed*  normalizedV    = NULL;

    /* clear previous error */
    font->error = FT_Err_Ok;

    /* if a CID fontDict has changed, we need to recompute some cached */
    /* data                                                            */
    subFont = cf2_getSubfont( decoder );
    if ( font->lastSubfont != subFont )
    {
      font->lastSubfont = subFont;
      needExtraSetup    = TRUE;
    }

    if ( !font->isT1 )
    {
      /* check for variation vectors */
      vstore        = cf2_getVStore( decoder );
      hasVariations = ( vstore->dataCount != 0 );

      if ( hasVariations )
      {
#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
        FT_Service_CFFLoad  cffload = (FT_Service_CFFLoad)font->cffload;


        /* check whether Private DICT in this subfont needs to be reparsed */
        font->error = cf2_getNormalizedVector( decoder,
                                               &lenNormalizedV,
                                               &normalizedV );
        if ( font->error )
          return;

        if ( cffload->blend_check_vector( &subFont->blend,
                                          subFont->private_dict.vsindex,
                                          lenNormalizedV,
                                          normalizedV ) )
        {
          /* blend has changed, reparse */
          cffload->load_private_dict( decoder->cff,
                                      subFont,
                                      lenNormalizedV,
                                      normalizedV );
          needExtraSetup = TRUE;
        }
#endif

        /* copy from subfont */
        font->blend.font = subFont->blend.font;

        /* clear state of charstring blend */
        font->blend.usedBV = FALSE;

        /* initialize value for charstring */
        font->vsindex = subFont->private_dict.vsindex;

        /* store vector inputs for blends in charstring */
        font->lenNDV = lenNormalizedV;
        font->NDV    = normalizedV;
      }
    }

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
    font->hinted = FT_BOOL( font->renderingFlags & CF2_FlagsHinted );

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
        FT_BOOL( font->renderingFlags & CF2_FlagsDarkened );

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
                              FALSE,
                              font->darkenParams );
      }
      else
        cf2_computeDarkening( emRatio,
                              ppem,
                              font->stdVW,
                              &font->darkenX,
                              0,
                              font->stemDarkened,
                              font->darkenParams );

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

      if ( stdHW > 0 && font->stdVW > MUL_INT32( 2, stdHW ) )
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
                            font->stemDarkened,
                            font->darkenParams );

      if ( font->darkenX != 0 || font->darkenY != 0 )
        font->darkened = TRUE;
      else
        font->darkened = FALSE;

      font->reverseWinding = FALSE; /* initial expectation is CCW */

      /* compute blue zones for this instance */
      cf2_blues_init( &font->blues, font );

    } /* needExtraSetup */
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
