/***************************************************************************/
/*                                                                         */
/*  cf2blues.c                                                             */
/*                                                                         */
/*    Adobe's code for handling Blue Zones (body).                         */
/*                                                                         */
/*  Copyright 2009-2014 Adobe Systems Incorporated.                        */
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

#include "cf2blues.h"
#include "cf2hints.h"
#include "cf2font.h"


  /*************************************************************************/
  /*                                                                       */
  /* The macro FT_COMPONENT is used in trace mode.  It is an implicit      */
  /* parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log  */
  /* messages during execution.                                            */
  /*                                                                       */
#undef  FT_COMPONENT
#define FT_COMPONENT  trace_cf2blues


  /*
   * For blue values, the FreeType parser produces an array of integers,
   * while the Adobe CFF engine produces an array of fixed.
   * Define a macro to convert FreeType to fixed.
   */
#define cf2_blueToFixed( x )  cf2_intToFixed( x )


  FT_LOCAL_DEF( void )
  cf2_blues_init( CF2_Blues  blues,
                  CF2_Font   font )
  {
    /* pointer to parsed font object */
    CFF_Decoder*  decoder = font->decoder;

    CF2_Fixed  zoneHeight;
    CF2_Fixed  maxZoneHeight = 0;
    CF2_Fixed  csUnitsPerPixel;

    size_t  numBlueValues;
    size_t  numOtherBlues;
    size_t  numFamilyBlues;
    size_t  numFamilyOtherBlues;

    FT_Pos*  blueValues;
    FT_Pos*  otherBlues;
    FT_Pos*  familyBlues;
    FT_Pos*  familyOtherBlues;

    size_t     i;
    CF2_Fixed  emBoxBottom, emBoxTop;

#if 0
    CF2_Int  unitsPerEm = font->unitsPerEm;


    if ( unitsPerEm == 0 )
      unitsPerEm = 1000;
#endif

    FT_ZERO( blues );
    blues->scale = font->innerTransform.d;

    cf2_getBlueMetrics( decoder,
                        &blues->blueScale,
                        &blues->blueShift,
                        &blues->blueFuzz );

    cf2_getBlueValues( decoder, &numBlueValues, &blueValues );
    cf2_getOtherBlues( decoder, &numOtherBlues, &otherBlues );
    cf2_getFamilyBlues( decoder, &numFamilyBlues, &familyBlues );
    cf2_getFamilyOtherBlues( decoder, &numFamilyOtherBlues, &familyOtherBlues );

    /*
     * synthetic em box hint heuristic
     *
     * Apply this when ideographic dictionary (LanguageGroup 1) has no
     * real alignment zones.  Adobe tools generate dummy zones at -250 and
     * 1100 for a 1000 unit em.  Fonts with ICF-based alignment zones
     * should not enable the heuristic.  When the heuristic is enabled,
     * the font's blue zones are ignored.
     *
     */

    /* get em box from OS/2 typoAscender/Descender                      */
    /* TODO: FreeType does not parse these metrics.  Skip them for now. */
#if 0
    FCM_getHorizontalLineMetrics( &e,
                                  font->font,
                                  &ascender,
                                  &descender,
                                  &linegap );
    if ( ascender - descender == unitsPerEm )
    {
      emBoxBottom = cf2_intToFixed( descender );
      emBoxTop    = cf2_intToFixed( ascender );
    }
    else
#endif
    {
      emBoxBottom = CF2_ICF_Bottom;
      emBoxTop    = CF2_ICF_Top;
    }

    if ( cf2_getLanguageGroup( decoder ) == 1                   &&
         ( numBlueValues == 0                                 ||
           ( numBlueValues == 4                             &&
             cf2_blueToFixed( blueValues[0] ) < emBoxBottom &&
             cf2_blueToFixed( blueValues[1] ) < emBoxBottom &&
             cf2_blueToFixed( blueValues[2] ) > emBoxTop    &&
             cf2_blueToFixed( blueValues[3] ) > emBoxTop    ) ) )
    {
      /*
       * Construct hint edges suitable for synthetic ghost hints at top
       * and bottom of em box.  +-CF2_MIN_COUNTER allows for unhinted
       * features above or below the last hinted edge.  This also gives a
       * net 1 pixel boost to the height of ideographic glyphs.
       *
       * Note: Adjust synthetic hints outward by epsilon (0x.0001) to
       *       avoid interference.  E.g., some fonts have real hints at
       *       880 and -120.
       */

      blues->emBoxBottomEdge.csCoord = emBoxBottom - CF2_FIXED_EPSILON;
      blues->emBoxBottomEdge.dsCoord = cf2_fixedRound(
                                         FT_MulFix(
                                           blues->emBoxBottomEdge.csCoord,
                                           blues->scale ) ) -
                                       CF2_MIN_COUNTER;
      blues->emBoxBottomEdge.scale   = blues->scale;
      blues->emBoxBottomEdge.flags   = CF2_GhostBottom |
                                       CF2_Locked |
                                       CF2_Synthetic;

      blues->emBoxTopEdge.csCoord = emBoxTop + CF2_FIXED_EPSILON +
                                    2 * font->darkenY;
      blues->emBoxTopEdge.dsCoord = cf2_fixedRound(
                                      FT_MulFix(
                                        blues->emBoxTopEdge.csCoord,
                                        blues->scale ) ) +
                                    CF2_MIN_COUNTER;
      blues->emBoxTopEdge.scale   = blues->scale;
      blues->emBoxTopEdge.flags   = CF2_GhostTop |
                                    CF2_Locked |
                                    CF2_Synthetic;

      blues->doEmBoxHints = TRUE;    /* enable the heuristic */

      return;
    }

    /* copy `BlueValues' and `OtherBlues' to a combined array of top and */
    /* bottom zones                                                      */
    for ( i = 0; i < numBlueValues; i += 2 )
    {
      blues->zone[blues->count].csBottomEdge =
        cf2_blueToFixed( blueValues[i] );
      blues->zone[blues->count].csTopEdge =
        cf2_blueToFixed( blueValues[i + 1] );

      zoneHeight = SUB_INT32( blues->zone[blues->count].csTopEdge,
                              blues->zone[blues->count].csBottomEdge );

      if ( zoneHeight < 0 )
      {
        FT_TRACE4(( "cf2_blues_init: ignoring negative zone height\n" ));
        continue;   /* reject this zone */
      }

      if ( zoneHeight > maxZoneHeight )
      {
        /* take maximum before darkening adjustment      */
        /* so overshoot suppression point doesn't change */
        maxZoneHeight = zoneHeight;
      }

      /* adjust both edges of top zone upward by twice darkening amount */
      if ( i != 0 )
      {
        blues->zone[blues->count].csTopEdge    += 2 * font->darkenY;
        blues->zone[blues->count].csBottomEdge += 2 * font->darkenY;
      }

      /* first `BlueValue' is bottom zone; others are top */
      if ( i == 0 )
      {
        blues->zone[blues->count].bottomZone =
          TRUE;
        blues->zone[blues->count].csFlatEdge =
          blues->zone[blues->count].csTopEdge;
      }
      else
      {
        blues->zone[blues->count].bottomZone =
          FALSE;
        blues->zone[blues->count].csFlatEdge =
          blues->zone[blues->count].csBottomEdge;
      }

      blues->count += 1;
    }

    for ( i = 0; i < numOtherBlues; i += 2 )
    {
      blues->zone[blues->count].csBottomEdge =
        cf2_blueToFixed( otherBlues[i] );
      blues->zone[blues->count].csTopEdge =
        cf2_blueToFixed( otherBlues[i + 1] );

      zoneHeight = SUB_INT32( blues->zone[blues->count].csTopEdge,
                              blues->zone[blues->count].csBottomEdge );

      if ( zoneHeight < 0 )
      {
        FT_TRACE4(( "cf2_blues_init: ignoring negative zone height\n" ));
        continue;   /* reject this zone */
      }

      if ( zoneHeight > maxZoneHeight )
      {
        /* take maximum before darkening adjustment      */
        /* so overshoot suppression point doesn't change */
        maxZoneHeight = zoneHeight;
      }

      /* Note: bottom zones are not adjusted for darkening amount */

      /* all OtherBlues are bottom zone */
      blues->zone[blues->count].bottomZone =
        TRUE;
      blues->zone[blues->count].csFlatEdge =
        blues->zone[blues->count].csTopEdge;

      blues->count += 1;
    }

    /* Adjust for FamilyBlues */

    /* Search for the nearest flat edge in `FamilyBlues' or                */
    /* `FamilyOtherBlues'.  According to the Black Book, any matching edge */
    /* must be within one device pixel                                     */

    csUnitsPerPixel = FT_DivFix( cf2_intToFixed( 1 ), blues->scale );

    /* loop on all zones in this font */
    for ( i = 0; i < blues->count; i++ )
    {
      size_t     j;
      CF2_Fixed  minDiff;
      CF2_Fixed  flatFamilyEdge, diff;
      /* value for this font */
      CF2_Fixed  flatEdge = blues->zone[i].csFlatEdge;


      if ( blues->zone[i].bottomZone )
      {
        /* In a bottom zone, the top edge is the flat edge.             */
        /* Search `FamilyOtherBlues' for bottom zones; look for closest */
        /* Family edge that is within the one pixel threshold.          */

        minDiff = CF2_FIXED_MAX;

        for ( j = 0; j < numFamilyOtherBlues; j += 2 )
        {
          /* top edge */
          flatFamilyEdge = cf2_blueToFixed( familyOtherBlues[j + 1] );

          diff = cf2_fixedAbs( SUB_INT32( flatEdge, flatFamilyEdge ) );

          if ( diff < minDiff && diff < csUnitsPerPixel )
          {
            blues->zone[i].csFlatEdge = flatFamilyEdge;
            minDiff                   = diff;

            if ( diff == 0 )
              break;
          }
        }

        /* check the first member of FamilyBlues, which is a bottom zone */
        if ( numFamilyBlues >= 2 )
        {
          /* top edge */
          flatFamilyEdge = cf2_blueToFixed( familyBlues[1] );

          diff = cf2_fixedAbs( SUB_INT32( flatEdge, flatFamilyEdge ) );

          if ( diff < minDiff && diff < csUnitsPerPixel )
            blues->zone[i].csFlatEdge = flatFamilyEdge;
        }
      }
      else
      {
        /* In a top zone, the bottom edge is the flat edge.                */
        /* Search `FamilyBlues' for top zones; skip first zone, which is a */
        /* bottom zone; look for closest Family edge that is within the    */
        /* one pixel threshold                                             */

        minDiff = CF2_FIXED_MAX;

        for ( j = 2; j < numFamilyBlues; j += 2 )
        {
          /* bottom edge */
          flatFamilyEdge = cf2_blueToFixed( familyBlues[j] );

          /* adjust edges of top zone upward by twice darkening amount */
          flatFamilyEdge += 2 * font->darkenY;      /* bottom edge */

          diff = cf2_fixedAbs( SUB_INT32( flatEdge, flatFamilyEdge ) );

          if ( diff < minDiff && diff < csUnitsPerPixel )
          {
            blues->zone[i].csFlatEdge = flatFamilyEdge;
            minDiff                   = diff;

            if ( diff == 0 )
              break;
          }
        }
      }
    }

    /* TODO: enforce separation of zones, including BlueFuzz */

    /* Adjust BlueScale; similar to AdjustBlueScale() in coretype */
    /* `bcsetup.c'.                                               */

    if ( maxZoneHeight > 0 )
    {
      if ( blues->blueScale > FT_DivFix( cf2_intToFixed( 1 ),
                                         maxZoneHeight ) )
      {
        /* clamp at maximum scale */
        blues->blueScale = FT_DivFix( cf2_intToFixed( 1 ),
                                      maxZoneHeight );
      }

      /*
       * TODO: Revisit the bug fix for 613448.  The minimum scale
       *       requirement catches a number of library fonts.  For
       *       example, with default BlueScale (.039625) and 0.4 minimum,
       *       the test below catches any font with maxZoneHeight < 10.1.
       *       There are library fonts ranging from 2 to 10 that get
       *       caught, including e.g., Eurostile LT Std Medium with
       *       maxZoneHeight of 6.
       *
       */
#if 0
      if ( blueScale < .4 / maxZoneHeight )
      {
        tetraphilia_assert( 0 );
        /* clamp at minimum scale, per bug 0613448 fix */
        blueScale = .4 / maxZoneHeight;
      }
#endif

    }

    /*
     * Suppress overshoot and boost blue zones at small sizes.  Boost
     * amount varies linearly from 0.5 pixel near 0 to 0 pixel at
     * blueScale cutoff.
     * Note: This boost amount is different from the coretype heuristic.
     *
     */

    if ( blues->scale < blues->blueScale )
    {
      blues->suppressOvershoot = TRUE;

      /* Change rounding threshold for `dsFlatEdge'.                    */
      /* Note: constant changed from 0.5 to 0.6 to avoid a problem with */
      /*       10ppem Arial                                             */

      blues->boost = cf2_doubleToFixed( .6 ) -
                       FT_MulDiv( cf2_doubleToFixed ( .6 ),
                                  blues->scale,
                                  blues->blueScale );
      if ( blues->boost > 0x7FFF )
      {
        /* boost must remain less than 0.5, or baseline could go negative */
        blues->boost = 0x7FFF;
      }
    }

    /* boost and darkening have similar effects; don't do both */
    if ( font->stemDarkened )
      blues->boost = 0;

    /* set device space alignment for each zone;    */
    /* apply boost amount before rounding flat edge */

    for ( i = 0; i < blues->count; i++ )
    {
      if ( blues->zone[i].bottomZone )
        blues->zone[i].dsFlatEdge = cf2_fixedRound(
                                      FT_MulFix(
                                        blues->zone[i].csFlatEdge,
                                        blues->scale ) -
                                      blues->boost );
      else
        blues->zone[i].dsFlatEdge = cf2_fixedRound(
                                      FT_MulFix(
                                        blues->zone[i].csFlatEdge,
                                        blues->scale ) +
                                      blues->boost );
    }
  }


  /*
   * Check whether `stemHint' is captured by one of the blue zones.
   *
   * Zero, one or both edges may be valid; only valid edges can be
   * captured.  For compatibility with CoolType, search top and bottom
   * zones in the same pass (see `BlueLock').  If a hint is captured,
   * return true and position the edge(s) in one of 3 ways:
   *
   *  1) If `BlueScale' suppresses overshoot, position the captured edge
   *     at the flat edge of the zone.
   *  2) If overshoot is not suppressed and `BlueShift' requires
   *     overshoot, position the captured edge a minimum of 1 device pixel
   *     from the flat edge.
   *  3) If overshoot is not suppressed or required, position the captured
   *     edge at the nearest device pixel.
   *
   */
  FT_LOCAL_DEF( FT_Bool )
  cf2_blues_capture( const CF2_Blues  blues,
                     CF2_Hint         bottomHintEdge,
                     CF2_Hint         topHintEdge )
  {
    /* TODO: validate? */
    CF2_Fixed  csFuzz = blues->blueFuzz;

    /* new position of captured edge */
    CF2_Fixed  dsNew;

    /* amount that hint is moved when positioned */
    CF2_Fixed  dsMove = 0;

    FT_Bool   captured = FALSE;
    CF2_UInt  i;


    /* assert edge flags are consistent */
    FT_ASSERT( !cf2_hint_isTop( bottomHintEdge ) &&
               !cf2_hint_isBottom( topHintEdge ) );

    /* TODO: search once without blue fuzz for compatibility with coretype? */
    for ( i = 0; i < blues->count; i++ )
    {
      if ( blues->zone[i].bottomZone           &&
           cf2_hint_isBottom( bottomHintEdge ) )
      {
        if ( SUB_INT32( blues->zone[i].csBottomEdge, csFuzz ) <=
               bottomHintEdge->csCoord                           &&
             bottomHintEdge->csCoord <=
               ADD_INT32( blues->zone[i].csTopEdge, csFuzz )     )
        {
          /* bottom edge captured by bottom zone */

          if ( blues->suppressOvershoot )
            dsNew = blues->zone[i].dsFlatEdge;

          else if ( SUB_INT32( blues->zone[i].csTopEdge,
                               bottomHintEdge->csCoord ) >=
                      blues->blueShift )
          {
            /* guarantee minimum of 1 pixel overshoot */
            dsNew = FT_MIN(
                      cf2_fixedRound( bottomHintEdge->dsCoord ),
                      blues->zone[i].dsFlatEdge - cf2_intToFixed( 1 ) );
          }

          else
          {
            /* simply round captured edge */
            dsNew = cf2_fixedRound( bottomHintEdge->dsCoord );
          }

          dsMove   = SUB_INT32( dsNew, bottomHintEdge->dsCoord );
          captured = TRUE;

          break;
        }
      }

      if ( !blues->zone[i].bottomZone && cf2_hint_isTop( topHintEdge ) )
      {
        if ( SUB_INT32( blues->zone[i].csBottomEdge, csFuzz ) <=
               topHintEdge->csCoord                              &&
             topHintEdge->csCoord <=
               ADD_INT32( blues->zone[i].csTopEdge, csFuzz )     )
        {
          /* top edge captured by top zone */

          if ( blues->suppressOvershoot )
            dsNew = blues->zone[i].dsFlatEdge;

          else if ( SUB_INT32( topHintEdge->csCoord,
                               blues->zone[i].csBottomEdge ) >=
                      blues->blueShift )
          {
            /* guarantee minimum of 1 pixel overshoot */
            dsNew = FT_MAX(
                      cf2_fixedRound( topHintEdge->dsCoord ),
                      blues->zone[i].dsFlatEdge + cf2_intToFixed( 1 ) );
          }

          else
          {
            /* simply round captured edge */
            dsNew = cf2_fixedRound( topHintEdge->dsCoord );
          }

          dsMove   = SUB_INT32( dsNew, topHintEdge->dsCoord );
          captured = TRUE;

          break;
        }
      }
    }

    if ( captured )
    {
      /* move both edges and flag them `locked' */
      if ( cf2_hint_isValid( bottomHintEdge ) )
      {
        bottomHintEdge->dsCoord = ADD_INT32( bottomHintEdge->dsCoord,
                                             dsMove );
        cf2_hint_lock( bottomHintEdge );
      }

      if ( cf2_hint_isValid( topHintEdge ) )
      {
        topHintEdge->dsCoord = ADD_INT32( topHintEdge->dsCoord, dsMove );
        cf2_hint_lock( topHintEdge );
      }
    }

    return captured;
  }


/* END */
