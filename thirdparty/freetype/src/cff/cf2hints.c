/***************************************************************************/
/*                                                                         */
/*  cf2hints.c                                                             */
/*                                                                         */
/*    Adobe's code for handling CFF hints (body).                          */
/*                                                                         */
/*  Copyright 2007-2014 Adobe Systems Incorporated.                        */
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
#include "cf2hints.h"
#include "cf2intrp.h"


  /*************************************************************************/
  /*                                                                       */
  /* The macro FT_COMPONENT is used in trace mode.  It is an implicit      */
  /* parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log  */
  /* messages during execution.                                            */
  /*                                                                       */
#undef  FT_COMPONENT
#define FT_COMPONENT  trace_cf2hints


  typedef struct  CF2_HintMoveRec_
  {
    size_t     j;          /* index of upper hint map edge   */
    CF2_Fixed  moveUp;     /* adjustment to optimum position */

  } CF2_HintMoveRec, *CF2_HintMove;


  /* Compute angular momentum for winding order detection.  It is called */
  /* for all lines and curves, but not necessarily in element order.     */
  static CF2_Int
  cf2_getWindingMomentum( CF2_Fixed  x1,
                          CF2_Fixed  y1,
                          CF2_Fixed  x2,
                          CF2_Fixed  y2 )
  {
    /* cross product of pt1 position from origin with pt2 position from  */
    /* pt1; we reduce the precision so that the result fits into 32 bits */

    return ( x1 >> 16 ) * ( SUB_INT32( y2, y1 ) >> 16 ) -
           ( y1 >> 16 ) * ( SUB_INT32( x2, x1 ) >> 16 );
  }


  /*
   * Construct from a StemHint; this is used as a parameter to
   * `cf2_blues_capture'.
   * `hintOrigin' is the character space displacement of a seac accent.
   * Adjust stem hint for darkening here.
   *
   */
  static void
  cf2_hint_init( CF2_Hint            hint,
                 const CF2_ArrStack  stemHintArray,
                 size_t              indexStemHint,
                 const CF2_Font      font,
                 CF2_Fixed           hintOrigin,
                 CF2_Fixed           scale,
                 FT_Bool             bottom )
  {
    CF2_Fixed               width;
    const CF2_StemHintRec*  stemHint;


    FT_ZERO( hint );

    stemHint = (const CF2_StemHintRec*)cf2_arrstack_getPointer(
                                         stemHintArray,
                                         indexStemHint );

    width = SUB_INT32( stemHint->max, stemHint->min );

    if ( width == cf2_intToFixed( -21 ) )
    {
      /* ghost bottom */

      if ( bottom )
      {
        hint->csCoord = stemHint->max;
        hint->flags   = CF2_GhostBottom;
      }
      else
        hint->flags = 0;
    }

    else if ( width == cf2_intToFixed( -20 ) )
    {
      /* ghost top */

      if ( bottom )
        hint->flags = 0;
      else
      {
        hint->csCoord = stemHint->min;
        hint->flags   = CF2_GhostTop;
      }
    }

    else if ( width < 0 )
    {
      /* inverted pair */

      /*
       * Hints with negative widths were produced by an early version of a
       * non-Adobe font tool.  The Type 2 spec allows edge (ghost) hints
       * with negative widths, but says
       *
       *   All other negative widths have undefined meaning.
       *
       * CoolType has a silent workaround that negates the hint width; for
       * permissive mode, we do the same here.
       *
       * Note: Such fonts cannot use ghost hints, but should otherwise work.
       * Note: Some poor hints in our faux fonts can produce negative
       *       widths at some blends.  For example, see a light weight of
       *       `u' in ASerifMM.
       *
       */
      if ( bottom )
      {
        hint->csCoord = stemHint->max;
        hint->flags   = CF2_PairBottom;
      }
      else
      {
        hint->csCoord = stemHint->min;
        hint->flags   = CF2_PairTop;
      }
    }

    else
    {
      /* normal pair */

      if ( bottom )
      {
        hint->csCoord = stemHint->min;
        hint->flags   = CF2_PairBottom;
      }
      else
      {
        hint->csCoord = stemHint->max;
        hint->flags   = CF2_PairTop;
      }
    }

    /* Now that ghost hints have been detected, adjust this edge for      */
    /* darkening.  Bottoms are not changed; tops are incremented by twice */
    /* `darkenY'.                                                         */
    if ( cf2_hint_isTop( hint ) )
      hint->csCoord = ADD_INT32( hint->csCoord, 2 * font->darkenY );

    hint->csCoord = ADD_INT32( hint->csCoord, hintOrigin );
    hint->scale   = scale;
    hint->index   = indexStemHint;   /* index in original stem hint array */

    /* if original stem hint has been used, use the same position */
    if ( hint->flags != 0 && stemHint->used )
    {
      if ( cf2_hint_isTop( hint ) )
        hint->dsCoord = stemHint->maxDS;
      else
        hint->dsCoord = stemHint->minDS;

      cf2_hint_lock( hint );
    }
    else
      hint->dsCoord = FT_MulFix( hint->csCoord, scale );
  }


  /* initialize an invalid hint map element */
  static void
  cf2_hint_initZero( CF2_Hint  hint )
  {
    FT_ZERO( hint );
  }


  FT_LOCAL_DEF( FT_Bool )
  cf2_hint_isValid( const CF2_Hint  hint )
  {
    return (FT_Bool)( hint->flags != 0 );
  }


  static FT_Bool
  cf2_hint_isPair( const CF2_Hint  hint )
  {
    return (FT_Bool)( ( hint->flags                      &
                        ( CF2_PairBottom | CF2_PairTop ) ) != 0 );
  }


  static FT_Bool
  cf2_hint_isPairTop( const CF2_Hint  hint )
  {
    return (FT_Bool)( ( hint->flags & CF2_PairTop ) != 0 );
  }


  FT_LOCAL_DEF( FT_Bool )
  cf2_hint_isTop( const CF2_Hint  hint )
  {
    return (FT_Bool)( ( hint->flags                    &
                        ( CF2_PairTop | CF2_GhostTop ) ) != 0 );
  }


  FT_LOCAL_DEF( FT_Bool )
  cf2_hint_isBottom( const CF2_Hint  hint )
  {
    return (FT_Bool)( ( hint->flags                          &
                        ( CF2_PairBottom | CF2_GhostBottom ) ) != 0 );
  }


  static FT_Bool
  cf2_hint_isLocked( const CF2_Hint  hint )
  {
    return (FT_Bool)( ( hint->flags & CF2_Locked ) != 0 );
  }


  static FT_Bool
  cf2_hint_isSynthetic( const CF2_Hint  hint )
  {
    return (FT_Bool)( ( hint->flags & CF2_Synthetic ) != 0 );
  }


  FT_LOCAL_DEF( void )
  cf2_hint_lock( CF2_Hint  hint )
  {
    hint->flags |= CF2_Locked;
  }


  FT_LOCAL_DEF( void )
  cf2_hintmap_init( CF2_HintMap   hintmap,
                    CF2_Font      font,
                    CF2_HintMap   initialMap,
                    CF2_ArrStack  hintMoves,
                    CF2_Fixed     scale )
  {
    FT_ZERO( hintmap );

    /* copy parameters from font instance */
    hintmap->hinted         = font->hinted;
    hintmap->scale          = scale;
    hintmap->font           = font;
    hintmap->initialHintMap = initialMap;
    /* will clear in `cf2_hintmap_adjustHints' */
    hintmap->hintMoves      = hintMoves;
  }


  static FT_Bool
  cf2_hintmap_isValid( const CF2_HintMap  hintmap )
  {
    return hintmap->isValid;
  }


  /* transform character space coordinate to device space using hint map */
  static CF2_Fixed
  cf2_hintmap_map( CF2_HintMap  hintmap,
                   CF2_Fixed    csCoord )
  {
    if ( hintmap->count == 0 || ! hintmap->hinted )
    {
      /* there are no hints; use uniform scale and zero offset */
      return FT_MulFix( csCoord, hintmap->scale );
    }
    else
    {
      /* start linear search from last hit */
      CF2_UInt  i = hintmap->lastIndex;


      FT_ASSERT( hintmap->lastIndex < CF2_MAX_HINT_EDGES );

      /* search up */
      while ( i < hintmap->count - 1                  &&
              csCoord >= hintmap->edge[i + 1].csCoord )
        i += 1;

      /* search down */
      while ( i > 0 && csCoord < hintmap->edge[i].csCoord )
        i -= 1;

      hintmap->lastIndex = i;

      if ( i == 0 && csCoord < hintmap->edge[0].csCoord )
      {
        /* special case for points below first edge: use uniform scale */
        return ADD_INT32( FT_MulFix( SUB_INT32( csCoord,
                                                hintmap->edge[0].csCoord ),
                                     hintmap->scale ),
                          hintmap->edge[0].dsCoord );
      }
      else
      {
        /*
         * Note: entries with duplicate csCoord are allowed.
         * Use edge[i], the highest entry where csCoord >= entry[i].csCoord
         */
        return ADD_INT32( FT_MulFix( SUB_INT32( csCoord,
                                                hintmap->edge[i].csCoord ),
                                     hintmap->edge[i].scale ),
                          hintmap->edge[i].dsCoord );
      }
    }
  }


  /*
   * This hinting policy moves a hint pair in device space so that one of
   * its two edges is on a device pixel boundary (its fractional part is
   * zero).  `cf2_hintmap_insertHint' guarantees no overlap in CS
   * space.  Ensure here that there is no overlap in DS.
   *
   * In the first pass, edges are adjusted relative to adjacent hints.
   * Those that are below have already been adjusted.  Those that are
   * above have not yet been adjusted.  If a hint above blocks an
   * adjustment to an optimal position, we will try again in a second
   * pass.  The second pass is top-down.
   *
   */

  static void
  cf2_hintmap_adjustHints( CF2_HintMap  hintmap )
  {
    size_t  i, j;


    cf2_arrstack_clear( hintmap->hintMoves );      /* working storage */

    /*
     * First pass is bottom-up (font hint order) without look-ahead.
     * Locked edges are already adjusted.
     * Unlocked edges begin with dsCoord from `initialHintMap'.
     * Save edges that are not optimally adjusted in `hintMoves' array,
     * and process them in second pass.
     */

    for ( i = 0; i < hintmap->count; i++ )
    {
      FT_Bool  isPair = cf2_hint_isPair( &hintmap->edge[i] );


      /* index of upper edge (same value for ghost hint) */
      j = isPair ? i + 1 : i;

      FT_ASSERT( j < hintmap->count );
      FT_ASSERT( cf2_hint_isValid( &hintmap->edge[i] ) );
      FT_ASSERT( cf2_hint_isValid( &hintmap->edge[j] ) );
      FT_ASSERT( cf2_hint_isLocked( &hintmap->edge[i] ) ==
                   cf2_hint_isLocked( &hintmap->edge[j] ) );

      if ( !cf2_hint_isLocked( &hintmap->edge[i] ) )
      {
        /* hint edge is not locked, we can adjust it */
        CF2_Fixed  fracDown = cf2_fixedFraction( hintmap->edge[i].dsCoord );
        CF2_Fixed  fracUp   = cf2_fixedFraction( hintmap->edge[j].dsCoord );

        /* calculate all four possibilities; moves down are negative */
        CF2_Fixed  downMoveDown = 0 - fracDown;
        CF2_Fixed  upMoveDown   = 0 - fracUp;
        CF2_Fixed  downMoveUp   = ( fracDown == 0 )
                                    ? 0
                                    : cf2_intToFixed( 1 ) - fracDown;
        CF2_Fixed  upMoveUp     = ( fracUp == 0 )
                                    ? 0
                                    : cf2_intToFixed( 1 ) - fracUp;

        /* smallest move up */
        CF2_Fixed  moveUp   = FT_MIN( downMoveUp, upMoveUp );
        /* smallest move down */
        CF2_Fixed  moveDown = FT_MAX( downMoveDown, upMoveDown );

        /* final amount to move edge or edge pair */
        CF2_Fixed  move;

        CF2_Fixed  downMinCounter = CF2_MIN_COUNTER;
        CF2_Fixed  upMinCounter   = CF2_MIN_COUNTER;
        FT_Bool    saveEdge       = FALSE;


        /* minimum counter constraint doesn't apply when adjacent edges */
        /* are synthetic                                                */
        /* TODO: doesn't seem a big effect; for now, reduce the code    */
#if 0
        if ( i == 0                                        ||
             cf2_hint_isSynthetic( &hintmap->edge[i - 1] ) )
          downMinCounter = 0;

        if ( j >= hintmap->count - 1                       ||
             cf2_hint_isSynthetic( &hintmap->edge[j + 1] ) )
          upMinCounter = 0;
#endif

        /* is there room to move up?                                    */
        /* there is if we are at top of array or the next edge is at or */
        /* beyond proposed move up?                                     */
        if ( j >= hintmap->count - 1                ||
             hintmap->edge[j + 1].dsCoord >=
               ADD_INT32( hintmap->edge[j].dsCoord,
                          moveUp + upMinCounter )   )
        {
          /* there is room to move up; is there also room to move down? */
          if ( i == 0                                   ||
               hintmap->edge[i - 1].dsCoord <=
                 ADD_INT32( hintmap->edge[i].dsCoord,
                            moveDown - downMinCounter ) )
          {
            /* move smaller absolute amount */
            move = ( -moveDown < moveUp ) ? moveDown : moveUp;  /* optimum */
          }
          else
            move = moveUp;
        }
        else
        {
          /* is there room to move down? */
          if ( i == 0                                   ||
               hintmap->edge[i - 1].dsCoord <=
                 ADD_INT32( hintmap->edge[i].dsCoord,
                            moveDown - downMinCounter ) )
          {
            move     = moveDown;
            /* true if non-optimum move */
            saveEdge = (FT_Bool)( moveUp < -moveDown );
          }
          else
          {
            /* no room to move either way without overlapping or reducing */
            /* the counter too much                                       */
            move     = 0;
            saveEdge = TRUE;
          }
        }

        /* Identify non-moves and moves down that aren't optimal, and save */
        /* them for second pass.                                           */
        /* Do this only if there is an unlocked edge above (which could    */
        /* possibly move).                                                 */
        if ( saveEdge                                    &&
             j < hintmap->count - 1                      &&
             !cf2_hint_isLocked( &hintmap->edge[j + 1] ) )
        {
          CF2_HintMoveRec  savedMove;


          savedMove.j      = j;
          /* desired adjustment in second pass */
          savedMove.moveUp = moveUp - move;

          cf2_arrstack_push( hintmap->hintMoves, &savedMove );
        }

        /* move the edge(s) */
        hintmap->edge[i].dsCoord = ADD_INT32( hintmap->edge[i].dsCoord,
                                              move );
        if ( isPair )
          hintmap->edge[j].dsCoord = ADD_INT32( hintmap->edge[j].dsCoord,
                                                move );
      }

      /* assert there are no overlaps in device space */
      FT_ASSERT( i == 0                                                   ||
                 hintmap->edge[i - 1].dsCoord <= hintmap->edge[i].dsCoord );
      FT_ASSERT( i < j                                                ||
                 hintmap->edge[i].dsCoord <= hintmap->edge[j].dsCoord );

      /* adjust the scales, avoiding divide by zero */
      if ( i > 0 )
      {
        if ( hintmap->edge[i].csCoord != hintmap->edge[i - 1].csCoord )
          hintmap->edge[i - 1].scale =
            FT_DivFix( SUB_INT32( hintmap->edge[i].dsCoord,
                                  hintmap->edge[i - 1].dsCoord ),
                       SUB_INT32( hintmap->edge[i].csCoord,
                                  hintmap->edge[i - 1].csCoord ) );
      }

      if ( isPair )
      {
        if ( hintmap->edge[j].csCoord != hintmap->edge[j - 1].csCoord )
          hintmap->edge[j - 1].scale =
            FT_DivFix( SUB_INT32( hintmap->edge[j].dsCoord,
                                  hintmap->edge[j - 1].dsCoord ),
                       SUB_INT32( hintmap->edge[j].csCoord,
                                  hintmap->edge[j - 1].csCoord ) );

        i += 1;     /* skip upper edge on next loop */
      }
    }

    /* second pass tries to move non-optimal hints up, in case there is */
    /* room now                                                         */
    for ( i = cf2_arrstack_size( hintmap->hintMoves ); i > 0; i-- )
    {
      CF2_HintMove  hintMove = (CF2_HintMove)
                      cf2_arrstack_getPointer( hintmap->hintMoves, i - 1 );


      j = hintMove->j;

      /* this was tested before the push, above */
      FT_ASSERT( j < hintmap->count - 1 );

      /* is there room to move up? */
      if ( hintmap->edge[j + 1].dsCoord >=
             ADD_INT32( hintmap->edge[j].dsCoord,
                        hintMove->moveUp + CF2_MIN_COUNTER ) )
      {
        /* there is more room now, move edge up */
        hintmap->edge[j].dsCoord = ADD_INT32( hintmap->edge[j].dsCoord,
                                              hintMove->moveUp );

        if ( cf2_hint_isPair( &hintmap->edge[j] ) )
        {
          FT_ASSERT( j > 0 );
          hintmap->edge[j - 1].dsCoord =
            ADD_INT32( hintmap->edge[j - 1].dsCoord, hintMove->moveUp );
        }
      }
    }
  }


  /* insert hint edges into map, sorted by csCoord */
  static void
  cf2_hintmap_insertHint( CF2_HintMap  hintmap,
                          CF2_Hint     bottomHintEdge,
                          CF2_Hint     topHintEdge )
  {
    CF2_UInt  indexInsert;

    /* set default values, then check for edge hints */
    FT_Bool   isPair         = TRUE;
    CF2_Hint  firstHintEdge  = bottomHintEdge;
    CF2_Hint  secondHintEdge = topHintEdge;


    /* one or none of the input params may be invalid when dealing with */
    /* edge hints; at least one edge must be valid                      */
    FT_ASSERT( cf2_hint_isValid( bottomHintEdge ) ||
               cf2_hint_isValid( topHintEdge )    );

    /* determine how many and which edges to insert */
    if ( !cf2_hint_isValid( bottomHintEdge ) )
    {
      /* insert only the top edge */
      firstHintEdge = topHintEdge;
      isPair        = FALSE;
    }
    else if ( !cf2_hint_isValid( topHintEdge ) )
    {
      /* insert only the bottom edge */
      isPair = FALSE;
    }

    /* paired edges must be in proper order */
    if ( isPair                                         &&
         topHintEdge->csCoord < bottomHintEdge->csCoord )
      return;

    /* linear search to find index value of insertion point */
    indexInsert = 0;
    for ( ; indexInsert < hintmap->count; indexInsert++ )
    {
      if ( hintmap->edge[indexInsert].csCoord >= firstHintEdge->csCoord )
        break;
    }

    /*
     * Discard any hints that overlap in character space.  Most often, this
     * is while building the initial map, where captured hints from all
     * zones are combined.  Define overlap to include hints that `touch'
     * (overlap zero).  Hiragino Sans/Gothic fonts have numerous hints that
     * touch.  Some fonts have non-ideographic glyphs that overlap our
     * synthetic hints.
     *
     * Overlap also occurs when darkening stem hints that are close.
     *
     */
    if ( indexInsert < hintmap->count )
    {
      /* we are inserting before an existing edge:    */
      /* verify that an existing edge is not the same */
      if ( hintmap->edge[indexInsert].csCoord == firstHintEdge->csCoord )
        return; /* ignore overlapping stem hint */

      /* verify that a new pair does not straddle the next edge */
      if ( isPair                                                        &&
           hintmap->edge[indexInsert].csCoord <= secondHintEdge->csCoord )
        return; /* ignore overlapping stem hint */

      /* verify that we are not inserting between paired edges */
      if ( cf2_hint_isPairTop( &hintmap->edge[indexInsert] ) )
        return; /* ignore overlapping stem hint */
    }

    /* recompute device space locations using initial hint map */
    if ( cf2_hintmap_isValid( hintmap->initialHintMap ) &&
         !cf2_hint_isLocked( firstHintEdge )            )
    {
      if ( isPair )
      {
        /* Use hint map to position the center of stem, and nominal scale */
        /* to position the two edges.  This preserves the stem width.     */
        CF2_Fixed  midpoint =
                     cf2_hintmap_map(
                       hintmap->initialHintMap,
                       ADD_INT32( secondHintEdge->csCoord,
                                  firstHintEdge->csCoord ) / 2 );
        CF2_Fixed  halfWidth =
                     FT_MulFix( SUB_INT32( secondHintEdge->csCoord,
                                           firstHintEdge->csCoord ) / 2,
                                hintmap->scale );


        firstHintEdge->dsCoord  = SUB_INT32( midpoint, halfWidth );
        secondHintEdge->dsCoord = ADD_INT32( midpoint, halfWidth );
      }
      else
        firstHintEdge->dsCoord = cf2_hintmap_map( hintmap->initialHintMap,
                                                  firstHintEdge->csCoord );
    }

    /*
     * Discard any hints that overlap in device space; this can occur
     * because locked hints have been moved to align with blue zones.
     *
     * TODO: Although we might correct this later during adjustment, we
     * don't currently have a way to delete a conflicting hint once it has
     * been inserted.  See v2.030 MinionPro-Regular, 12 ppem darkened,
     * initial hint map for second path, glyph 945 (the perispomeni (tilde)
     * in U+1F6E, Greek omega with psili and perispomeni).  Darkening is
     * 25.  Pair 667,747 initially conflicts in design space with top edge
     * 660.  This is because 667 maps to 7.87, and the top edge was
     * captured by a zone at 8.0.  The pair is later successfully inserted
     * in a zone without the top edge.  In this zone it is adjusted to 8.0,
     * and no longer conflicts with the top edge in design space.  This
     * means it can be included in yet a later zone which does have the top
     * edge hint.  This produces a small mismatch between the first and
     * last points of this path, even though the hint masks are the same.
     * The density map difference is tiny (1/256).
     *
     */

    if ( indexInsert > 0 )
    {
      /* we are inserting after an existing edge */
      if ( firstHintEdge->dsCoord < hintmap->edge[indexInsert - 1].dsCoord )
        return;
    }

    if ( indexInsert < hintmap->count )
    {
      /* we are inserting before an existing edge */
      if ( isPair )
      {
        if ( secondHintEdge->dsCoord > hintmap->edge[indexInsert].dsCoord )
          return;
      }
      else
      {
        if ( firstHintEdge->dsCoord > hintmap->edge[indexInsert].dsCoord )
          return;
      }
    }

    /* make room to insert */
    {
      CF2_UInt  iSrc = hintmap->count - 1;
      CF2_UInt  iDst = isPair ? hintmap->count + 1 : hintmap->count;

      CF2_UInt  count = hintmap->count - indexInsert;


      if ( iDst >= CF2_MAX_HINT_EDGES )
      {
        FT_TRACE4(( "cf2_hintmap_insertHint: too many hintmaps\n" ));
        return;
      }

      while ( count-- )
        hintmap->edge[iDst--] = hintmap->edge[iSrc--];

      /* insert first edge */
      hintmap->edge[indexInsert] = *firstHintEdge;         /* copy struct */
      hintmap->count            += 1;

      if ( isPair )
      {
        /* insert second edge */
        hintmap->edge[indexInsert + 1] = *secondHintEdge;  /* copy struct */
        hintmap->count                += 1;
      }
    }

    return;
  }


  /*
   * Build a map from hints and mask.
   *
   * This function may recur one level if `hintmap->initialHintMap' is not yet
   * valid.
   * If `initialMap' is true, simply build initial map.
   *
   * Synthetic hints are used in two ways.  A hint at zero is inserted, if
   * needed, in the initial hint map, to prevent translations from
   * propagating across the origin.  If synthetic em box hints are enabled
   * for ideographic dictionaries, then they are inserted in all hint
   * maps, including the initial one.
   *
   */
  FT_LOCAL_DEF( void )
  cf2_hintmap_build( CF2_HintMap   hintmap,
                     CF2_ArrStack  hStemHintArray,
                     CF2_ArrStack  vStemHintArray,
                     CF2_HintMask  hintMask,
                     CF2_Fixed     hintOrigin,
                     FT_Bool       initialMap )
  {
    FT_Byte*  maskPtr;

    CF2_Font         font = hintmap->font;
    CF2_HintMaskRec  tempHintMask;

    size_t   bitCount, i;
    FT_Byte  maskByte;


    /* check whether initial map is constructed */
    if ( !initialMap && !cf2_hintmap_isValid( hintmap->initialHintMap ) )
    {
      /* make recursive call with initialHintMap and temporary mask; */
      /* temporary mask will get all bits set, below */
      cf2_hintmask_init( &tempHintMask, hintMask->error );
      cf2_hintmap_build( hintmap->initialHintMap,
                         hStemHintArray,
                         vStemHintArray,
                         &tempHintMask,
                         hintOrigin,
                         TRUE );
    }

    if ( !cf2_hintmask_isValid( hintMask ) )
    {
      /* without a hint mask, assume all hints are active */
      cf2_hintmask_setAll( hintMask,
                           cf2_arrstack_size( hStemHintArray ) +
                             cf2_arrstack_size( vStemHintArray ) );
      if ( !cf2_hintmask_isValid( hintMask ) )
        return;                   /* too many stem hints */
    }

    /* begin by clearing the map */
    hintmap->count     = 0;
    hintmap->lastIndex = 0;

    /* make a copy of the hint mask so we can modify it */
    tempHintMask = *hintMask;
    maskPtr      = cf2_hintmask_getMaskPtr( &tempHintMask );

    /* use the hStem hints only, which are first in the mask */
    bitCount = cf2_arrstack_size( hStemHintArray );

    /* Defense-in-depth.  Should never return here. */
    if ( bitCount > hintMask->bitCount )
      return;

    /* synthetic embox hints get highest priority */
    if ( font->blues.doEmBoxHints )
    {
      CF2_HintRec  dummy;


      cf2_hint_initZero( &dummy );   /* invalid hint map element */

      /* ghost bottom */
      cf2_hintmap_insertHint( hintmap,
                              &font->blues.emBoxBottomEdge,
                              &dummy );
      /* ghost top */
      cf2_hintmap_insertHint( hintmap,
                              &dummy,
                              &font->blues.emBoxTopEdge );
    }

    /* insert hints captured by a blue zone or already locked (higher */
    /* priority)                                                      */
    for ( i = 0, maskByte = 0x80; i < bitCount; i++ )
    {
      if ( maskByte & *maskPtr )
      {
        /* expand StemHint into two `CF2_Hint' elements */
        CF2_HintRec  bottomHintEdge, topHintEdge;


        cf2_hint_init( &bottomHintEdge,
                       hStemHintArray,
                       i,
                       font,
                       hintOrigin,
                       hintmap->scale,
                       TRUE /* bottom */ );
        cf2_hint_init( &topHintEdge,
                       hStemHintArray,
                       i,
                       font,
                       hintOrigin,
                       hintmap->scale,
                       FALSE /* top */ );

        if ( cf2_hint_isLocked( &bottomHintEdge ) ||
             cf2_hint_isLocked( &topHintEdge )    ||
             cf2_blues_capture( &font->blues,
                                &bottomHintEdge,
                                &topHintEdge )   )
        {
          /* insert captured hint into map */
          cf2_hintmap_insertHint( hintmap, &bottomHintEdge, &topHintEdge );

          *maskPtr &= ~maskByte;      /* turn off the bit for this hint */
        }
      }

      if ( ( i & 7 ) == 7 )
      {
        /* move to next mask byte */
        maskPtr++;
        maskByte = 0x80;
      }
      else
        maskByte >>= 1;
    }

    /* initial hint map includes only captured hints plus maybe one at 0 */

    /*
     * TODO: There is a problem here because we are trying to build a
     *       single hint map containing all captured hints.  It is
     *       possible for there to be conflicts between captured hints,
     *       either because of darkening or because the hints are in
     *       separate hint zones (we are ignoring hint zones for the
     *       initial map).  An example of the latter is MinionPro-Regular
     *       v2.030 glyph 883 (Greek Capital Alpha with Psili) at 15ppem.
     *       A stem hint for the psili conflicts with the top edge hint
     *       for the base character.  The stem hint gets priority because
     *       of its sort order.  In glyph 884 (Greek Capital Alpha with
     *       Psili and Oxia), the top of the base character gets a stem
     *       hint, and the psili does not.  This creates different initial
     *       maps for the two glyphs resulting in different renderings of
     *       the base character.  Will probably defer this either as not
     *       worth the cost or as a font bug.  I don't think there is any
     *       good reason for an accent to be captured by an alignment
     *       zone.  -darnold 2/12/10
     */

    if ( initialMap )
    {
      /* Apply a heuristic that inserts a point for (0,0), unless it's     */
      /* already covered by a mapping.  This locks the baseline for glyphs */
      /* that have no baseline hints.                                      */

      if ( hintmap->count == 0                           ||
           hintmap->edge[0].csCoord > 0                  ||
           hintmap->edge[hintmap->count - 1].csCoord < 0 )
      {
        /* all edges are above 0 or all edges are below 0; */
        /* construct a locked edge hint at 0               */

        CF2_HintRec  edge, invalid;


        cf2_hint_initZero( &edge );

        edge.flags = CF2_GhostBottom |
                     CF2_Locked      |
                     CF2_Synthetic;
        edge.scale = hintmap->scale;

        cf2_hint_initZero( &invalid );
        cf2_hintmap_insertHint( hintmap, &edge, &invalid );
      }
    }
    else
    {
      /* insert remaining hints */

      maskPtr = cf2_hintmask_getMaskPtr( &tempHintMask );

      for ( i = 0, maskByte = 0x80; i < bitCount; i++ )
      {
        if ( maskByte & *maskPtr )
        {
          CF2_HintRec  bottomHintEdge, topHintEdge;


          cf2_hint_init( &bottomHintEdge,
                         hStemHintArray,
                         i,
                         font,
                         hintOrigin,
                         hintmap->scale,
                         TRUE /* bottom */ );
          cf2_hint_init( &topHintEdge,
                         hStemHintArray,
                         i,
                         font,
                         hintOrigin,
                         hintmap->scale,
                         FALSE /* top */ );

          cf2_hintmap_insertHint( hintmap, &bottomHintEdge, &topHintEdge );
        }

        if ( ( i & 7 ) == 7 )
        {
          /* move to next mask byte */
          maskPtr++;
          maskByte = 0x80;
        }
        else
          maskByte >>= 1;
      }
    }

    /*
     * Note: The following line is a convenient place to break when
     *       debugging hinting.  Examine `hintmap->edge' for the list of
     *       enabled hints, then step over the call to see the effect of
     *       adjustment.  We stop here first on the recursive call that
     *       creates the initial map, and then on each counter group and
     *       hint zone.
     */

    /* adjust positions of hint edges that are not locked to blue zones */
    cf2_hintmap_adjustHints( hintmap );

    /* save the position of all hints that were used in this hint map; */
    /* if we use them again, we'll locate them in the same position    */
    if ( !initialMap )
    {
      for ( i = 0; i < hintmap->count; i++ )
      {
        if ( !cf2_hint_isSynthetic( &hintmap->edge[i] ) )
        {
          /* Note: include both valid and invalid edges            */
          /* Note: top and bottom edges are copied back separately */
          CF2_StemHint  stemhint = (CF2_StemHint)
                          cf2_arrstack_getPointer( hStemHintArray,
                                                   hintmap->edge[i].index );


          if ( cf2_hint_isTop( &hintmap->edge[i] ) )
            stemhint->maxDS = hintmap->edge[i].dsCoord;
          else
            stemhint->minDS = hintmap->edge[i].dsCoord;

          stemhint->used = TRUE;
        }
      }
    }

    /* hint map is ready to use */
    hintmap->isValid = TRUE;

    /* remember this mask has been used */
    cf2_hintmask_setNew( hintMask, FALSE );
  }


  FT_LOCAL_DEF( void )
  cf2_glyphpath_init( CF2_GlyphPath         glyphpath,
                      CF2_Font              font,
                      CF2_OutlineCallbacks  callbacks,
                      CF2_Fixed             scaleY,
                      /* CF2_Fixed  hShift, */
                      CF2_ArrStack          hStemHintArray,
                      CF2_ArrStack          vStemHintArray,
                      CF2_HintMask          hintMask,
                      CF2_Fixed             hintOriginY,
                      const CF2_Blues       blues,
                      const FT_Vector*      fractionalTranslation )
  {
    FT_ZERO( glyphpath );

    glyphpath->font      = font;
    glyphpath->callbacks = callbacks;

    cf2_arrstack_init( &glyphpath->hintMoves,
                       font->memory,
                       &font->error,
                       sizeof ( CF2_HintMoveRec ) );

    cf2_hintmap_init( &glyphpath->initialHintMap,
                      font,
                      &glyphpath->initialHintMap,
                      &glyphpath->hintMoves,
                      scaleY );
    cf2_hintmap_init( &glyphpath->firstHintMap,
                      font,
                      &glyphpath->initialHintMap,
                      &glyphpath->hintMoves,
                      scaleY );
    cf2_hintmap_init( &glyphpath->hintMap,
                      font,
                      &glyphpath->initialHintMap,
                      &glyphpath->hintMoves,
                      scaleY );

    glyphpath->scaleX = font->innerTransform.a;
    glyphpath->scaleC = font->innerTransform.c;
    glyphpath->scaleY = font->innerTransform.d;

    glyphpath->fractionalTranslation = *fractionalTranslation;

#if 0
    glyphpath->hShift = hShift;       /* for fauxing */
#endif

    glyphpath->hStemHintArray = hStemHintArray;
    glyphpath->vStemHintArray = vStemHintArray;
    glyphpath->hintMask       = hintMask;      /* ptr to current mask */
    glyphpath->hintOriginY    = hintOriginY;
    glyphpath->blues          = blues;
    glyphpath->darken         = font->darkened; /* TODO: should we make copies? */
    glyphpath->xOffset        = font->darkenX;
    glyphpath->yOffset        = font->darkenY;
    glyphpath->miterLimit     = 2 * FT_MAX(
                                     cf2_fixedAbs( glyphpath->xOffset ),
                                     cf2_fixedAbs( glyphpath->yOffset ) );

    /* .1 character space unit */
    glyphpath->snapThreshold = cf2_doubleToFixed( 0.1 );

    glyphpath->moveIsPending = TRUE;
    glyphpath->pathIsOpen    = FALSE;
    glyphpath->pathIsClosing = FALSE;
    glyphpath->elemIsQueued  = FALSE;
  }


  FT_LOCAL_DEF( void )
  cf2_glyphpath_finalize( CF2_GlyphPath  glyphpath )
  {
    cf2_arrstack_finalize( &glyphpath->hintMoves );
  }


  /*
   * Hint point in y-direction and apply outerTransform.
   * Input `current' hint map (which is actually delayed by one element).
   * Input x,y point in Character Space.
   * Output x,y point in Device Space, including translation.
   */
  static void
  cf2_glyphpath_hintPoint( CF2_GlyphPath  glyphpath,
                           CF2_HintMap    hintmap,
                           FT_Vector*     ppt,
                           CF2_Fixed      x,
                           CF2_Fixed      y )
  {
    FT_Vector  pt;   /* hinted point in upright DS */


    pt.x = ADD_INT32( FT_MulFix( glyphpath->scaleX, x ),
                      FT_MulFix( glyphpath->scaleC, y ) );
    pt.y = cf2_hintmap_map( hintmap, y );

    ppt->x = ADD_INT32(
               FT_MulFix( glyphpath->font->outerTransform.a, pt.x ),
               ADD_INT32(
                 FT_MulFix( glyphpath->font->outerTransform.c, pt.y ),
                 glyphpath->fractionalTranslation.x ) );
    ppt->y = ADD_INT32(
               FT_MulFix( glyphpath->font->outerTransform.b, pt.x ),
               ADD_INT32(
                 FT_MulFix( glyphpath->font->outerTransform.d, pt.y ),
                 glyphpath->fractionalTranslation.y ) );
  }


  /*
   * From two line segments, (u1,u2) and (v1,v2), compute a point of
   * intersection on the corresponding lines.
   * Return false if no intersection is found, or if the intersection is
   * too far away from the ends of the line segments, u2 and v1.
   *
   */
  static FT_Bool
  cf2_glyphpath_computeIntersection( CF2_GlyphPath     glyphpath,
                                     const FT_Vector*  u1,
                                     const FT_Vector*  u2,
                                     const FT_Vector*  v1,
                                     const FT_Vector*  v2,
                                     FT_Vector*        intersection )
  {
    /*
     * Let `u' be a zero-based vector from the first segment, `v' from the
     * second segment.
     * Let `w 'be the zero-based vector from `u1' to `v1'.
     * `perp' is the `perpendicular dot product'; see
     * http://mathworld.wolfram.com/PerpDotProduct.html.
     * `s' is the parameter for the parametric line for the first segment
     * (`u').
     *
     * See notation in
     * http://softsurfer.com/Archive/algorithm_0104/algorithm_0104B.htm.
     * Calculations are done in 16.16, but must handle the squaring of
     * line lengths in character space.  We scale all vectors by 1/32 to
     * avoid overflow.  This allows values up to 4095 to be squared.  The
     * scale factor cancels in the divide.
     *
     * TODO: the scale factor could be computed from UnitsPerEm.
     *
     */

#define cf2_perp( a, b )                                    \
          ( FT_MulFix( a.x, b.y ) - FT_MulFix( a.y, b.x ) )

  /* round and divide by 32 */
#define CF2_CS_SCALE( x )         \
          ( ( (x) + 0x10 ) >> 5 )

    FT_Vector  u, v, w;      /* scaled vectors */
    CF2_Fixed  denominator, s;


    u.x = CF2_CS_SCALE( SUB_INT32( u2->x, u1->x ) );
    u.y = CF2_CS_SCALE( SUB_INT32( u2->y, u1->y ) );
    v.x = CF2_CS_SCALE( SUB_INT32( v2->x, v1->x ) );
    v.y = CF2_CS_SCALE( SUB_INT32( v2->y, v1->y ) );
    w.x = CF2_CS_SCALE( SUB_INT32( v1->x, u1->x ) );
    w.y = CF2_CS_SCALE( SUB_INT32( v1->y, u1->y ) );

    denominator = cf2_perp( u, v );

    if ( denominator == 0 )
      return FALSE;           /* parallel or coincident lines */

    s = FT_DivFix( cf2_perp( w, v ), denominator );

    intersection->x = ADD_INT32( u1->x,
                                 FT_MulFix( s, SUB_INT32( u2->x, u1->x ) ) );
    intersection->y = ADD_INT32( u1->y,
                                 FT_MulFix( s, SUB_INT32( u2->y, u1->y ) ) );


    /*
     * Special case snapping for horizontal and vertical lines.
     * This cleans up intersections and reduces problems with winding
     * order detection.
     * Sample case is sbc cd KozGoPr6N-Medium.otf 20 16685.
     * Note: these calculations are in character space.
     *
     */

    if ( u1->x == u2->x                                                &&
         cf2_fixedAbs( SUB_INT32( intersection->x,
                                  u1->x ) ) < glyphpath->snapThreshold )
      intersection->x = u1->x;
    if ( u1->y == u2->y                                                &&
         cf2_fixedAbs( SUB_INT32( intersection->y,
                                  u1->y ) ) < glyphpath->snapThreshold )
      intersection->y = u1->y;

    if ( v1->x == v2->x                                                &&
         cf2_fixedAbs( SUB_INT32( intersection->x,
                                  v1->x ) ) < glyphpath->snapThreshold )
      intersection->x = v1->x;
    if ( v1->y == v2->y                                                &&
         cf2_fixedAbs( SUB_INT32( intersection->y,
                                  v1->y ) ) < glyphpath->snapThreshold )
      intersection->y = v1->y;

    /* limit the intersection distance from midpoint of u2 and v1 */
    if ( cf2_fixedAbs( intersection->x - ADD_INT32( u2->x, v1->x ) / 2 ) >
           glyphpath->miterLimit                                           ||
         cf2_fixedAbs( intersection->y - ADD_INT32( u2->y, v1->y ) / 2 ) >
           glyphpath->miterLimit                                           )
      return FALSE;

    return TRUE;
  }


  /*
   * Push the cached element (glyphpath->prevElem*) to the outline
   * consumer.  When a darkening offset is used, the end point of the
   * cached element may be adjusted to an intersection point or we may
   * synthesize a connecting line to the current element.  If we are
   * closing a subpath, we may also generate a connecting line to the start
   * point.
   *
   * This is where Character Space (CS) is converted to Device Space (DS)
   * using a hint map.  This calculation must use a HintMap that was valid
   * at the time the element was saved.  For the first point in a subpath,
   * that is a saved HintMap.  For most elements, it just means the caller
   * has delayed building a HintMap from the current HintMask.
   *
   * Transform each point with outerTransform and call the outline
   * callbacks.  This is a general 3x3 transform:
   *
   *   x' = a*x + c*y + tx, y' = b*x + d*y + ty
   *
   * but it uses 4 elements from CF2_Font and the translation part
   * from CF2_GlyphPath.
   *
   */
  static void
  cf2_glyphpath_pushPrevElem( CF2_GlyphPath  glyphpath,
                              CF2_HintMap    hintmap,
                              FT_Vector*     nextP0,
                              FT_Vector      nextP1,
                              FT_Bool        close )
  {
    CF2_CallbackParamsRec  params;

    FT_Vector*  prevP0;
    FT_Vector*  prevP1;

    FT_Vector  intersection    = { 0, 0 };
    FT_Bool    useIntersection = FALSE;


    FT_ASSERT( glyphpath->prevElemOp == CF2_PathOpLineTo ||
               glyphpath->prevElemOp == CF2_PathOpCubeTo );

    if ( glyphpath->prevElemOp == CF2_PathOpLineTo )
    {
      prevP0 = &glyphpath->prevElemP0;
      prevP1 = &glyphpath->prevElemP1;
    }
    else
    {
      prevP0 = &glyphpath->prevElemP2;
      prevP1 = &glyphpath->prevElemP3;
    }

    /* optimization: if previous and next elements are offset by the same */
    /* amount, then there will be no gap, and no need to compute an       */
    /* intersection.                                                      */
    if ( prevP1->x != nextP0->x || prevP1->y != nextP0->y )
    {
      /* previous element does not join next element:             */
      /* adjust end point of previous element to the intersection */
      useIntersection = cf2_glyphpath_computeIntersection( glyphpath,
                                                           prevP0,
                                                           prevP1,
                                                           nextP0,
                                                           &nextP1,
                                                           &intersection );
      if ( useIntersection )
      {
        /* modify the last point of the cached element (either line or */
        /* curve)                                                      */
        *prevP1 = intersection;
      }
    }

    params.pt0 = glyphpath->currentDS;

    switch( glyphpath->prevElemOp )
    {
    case CF2_PathOpLineTo:
      params.op = CF2_PathOpLineTo;

      /* note: pt2 and pt3 are unused */

      if ( close )
      {
        /* use first hint map if closing */
        cf2_glyphpath_hintPoint( glyphpath,
                                 &glyphpath->firstHintMap,
                                 &params.pt1,
                                 glyphpath->prevElemP1.x,
                                 glyphpath->prevElemP1.y );
      }
      else
      {
        cf2_glyphpath_hintPoint( glyphpath,
                                 hintmap,
                                 &params.pt1,
                                 glyphpath->prevElemP1.x,
                                 glyphpath->prevElemP1.y );
      }

      /* output only non-zero length lines */
      if ( params.pt0.x != params.pt1.x || params.pt0.y != params.pt1.y )
      {
        glyphpath->callbacks->lineTo( glyphpath->callbacks, &params );

        glyphpath->currentDS = params.pt1;
      }
      break;

    case CF2_PathOpCubeTo:
      params.op = CF2_PathOpCubeTo;

      /* TODO: should we intersect the interior joins (p1-p2 and p2-p3)? */
      cf2_glyphpath_hintPoint( glyphpath,
                               hintmap,
                               &params.pt1,
                               glyphpath->prevElemP1.x,
                               glyphpath->prevElemP1.y );
      cf2_glyphpath_hintPoint( glyphpath,
                               hintmap,
                               &params.pt2,
                               glyphpath->prevElemP2.x,
                               glyphpath->prevElemP2.y );
      cf2_glyphpath_hintPoint( glyphpath,
                               hintmap,
                               &params.pt3,
                               glyphpath->prevElemP3.x,
                               glyphpath->prevElemP3.y );

      glyphpath->callbacks->cubeTo( glyphpath->callbacks, &params );

      glyphpath->currentDS = params.pt3;

      break;
    }

    if ( !useIntersection || close )
    {
      /* insert connecting line between end of previous element and start */
      /* of current one                                                   */
      /* note: at the end of a subpath, we might do both, so use `nextP0' */
      /* before we change it, below                                       */

      if ( close )
      {
        /* if we are closing the subpath, then nextP0 is in the first     */
        /* hint zone                                                      */
        cf2_glyphpath_hintPoint( glyphpath,
                                 &glyphpath->firstHintMap,
                                 &params.pt1,
                                 nextP0->x,
                                 nextP0->y );
      }
      else
      {
        cf2_glyphpath_hintPoint( glyphpath,
                                 hintmap,
                                 &params.pt1,
                                 nextP0->x,
                                 nextP0->y );
      }

      if ( params.pt1.x != glyphpath->currentDS.x ||
           params.pt1.y != glyphpath->currentDS.y )
      {
        /* length is nonzero */
        params.op  = CF2_PathOpLineTo;
        params.pt0 = glyphpath->currentDS;

        /* note: pt2 and pt3 are unused */
        glyphpath->callbacks->lineTo( glyphpath->callbacks, &params );

        glyphpath->currentDS = params.pt1;
      }
    }

    if ( useIntersection )
    {
      /* return intersection point to caller */
      *nextP0 = intersection;
    }
  }


  /* push a MoveTo element based on current point and offset of current */
  /* element                                                            */
  static void
  cf2_glyphpath_pushMove( CF2_GlyphPath  glyphpath,
                          FT_Vector      start )
  {
    CF2_CallbackParamsRec  params;


    params.op  = CF2_PathOpMoveTo;
    params.pt0 = glyphpath->currentDS;

    /* Test if move has really happened yet; it would have called */
    /* `cf2_hintmap_build' to set `isValid'.                   */
    if ( !cf2_hintmap_isValid( &glyphpath->hintMap ) )
    {
      /* we are here iff first subpath is missing a moveto operator: */
      /* synthesize first moveTo to finish initialization of hintMap */
      cf2_glyphpath_moveTo( glyphpath,
                            glyphpath->start.x,
                            glyphpath->start.y );
    }

    cf2_glyphpath_hintPoint( glyphpath,
                             &glyphpath->hintMap,
                             &params.pt1,
                             start.x,
                             start.y );

    /* note: pt2 and pt3 are unused */
    glyphpath->callbacks->moveTo( glyphpath->callbacks, &params );

    glyphpath->currentDS    = params.pt1;
    glyphpath->offsetStart0 = start;
  }


  /*
   * All coordinates are in character space.
   * On input, (x1, y1) and (x2, y2) give line segment.
   * On output, (x, y) give offset vector.
   * We use a piecewise approximation to trig functions.
   *
   * TODO: Offset true perpendicular and proper length
   *       supply the y-translation for hinting here, too,
   *       that adds yOffset unconditionally to *y.
   */
  static void
  cf2_glyphpath_computeOffset( CF2_GlyphPath  glyphpath,
                               CF2_Fixed      x1,
                               CF2_Fixed      y1,
                               CF2_Fixed      x2,
                               CF2_Fixed      y2,
                               CF2_Fixed*     x,
                               CF2_Fixed*     y )
  {
    CF2_Fixed  dx = SUB_INT32( x2, x1 );
    CF2_Fixed  dy = SUB_INT32( y2, y1 );


    /* note: negative offsets don't work here; negate deltas to change */
    /* quadrants, below                                                */
    if ( glyphpath->font->reverseWinding )
    {
      dx = NEG_INT32( dx );
      dy = NEG_INT32( dy );
    }

    *x = *y = 0;

    if ( !glyphpath->darken )
        return;

    /* add momentum for this path element */
    glyphpath->callbacks->windingMomentum =
      ADD_INT32( glyphpath->callbacks->windingMomentum,
                 cf2_getWindingMomentum( x1, y1, x2, y2 ) );

    /* note: allow mixed integer and fixed multiplication here */
    if ( dx >= 0 )
    {
      if ( dy >= 0 )
      {
        /* first quadrant, +x +y */

        if ( dx > MUL_INT32( 2, dy ) )
        {
          /* +x */
          *x = 0;
          *y = 0;
        }
        else if ( dy > MUL_INT32( 2, dx ) )
        {
          /* +y */
          *x = glyphpath->xOffset;
          *y = glyphpath->yOffset;
        }
        else
        {
          /* +x +y */
          *x = FT_MulFix( cf2_doubleToFixed( 0.7 ),
                          glyphpath->xOffset );
          *y = FT_MulFix( cf2_doubleToFixed( 1.0 - 0.7 ),
                          glyphpath->yOffset );
        }
      }
      else
      {
        /* fourth quadrant, +x -y */

        if ( dx > MUL_INT32( -2, dy ) )
        {
          /* +x */
          *x = 0;
          *y = 0;
        }
        else if ( NEG_INT32( dy ) > MUL_INT32( 2, dx ) )
        {
          /* -y */
          *x = NEG_INT32( glyphpath->xOffset );
          *y = glyphpath->yOffset;
        }
        else
        {
          /* +x -y */
          *x = FT_MulFix( cf2_doubleToFixed( -0.7 ),
                          glyphpath->xOffset );
          *y = FT_MulFix( cf2_doubleToFixed( 1.0 - 0.7 ),
                          glyphpath->yOffset );
        }
      }
    }
    else
    {
      if ( dy >= 0 )
      {
        /* second quadrant, -x +y */

        if ( NEG_INT32( dx ) > MUL_INT32( 2, dy ) )
        {
          /* -x */
          *x = 0;
          *y = MUL_INT32( 2, glyphpath->yOffset );
        }
        else if ( dy > MUL_INT32( -2, dx ) )
        {
          /* +y */
          *x = glyphpath->xOffset;
          *y = glyphpath->yOffset;
        }
        else
        {
          /* -x +y */
          *x = FT_MulFix( cf2_doubleToFixed( 0.7 ),
                          glyphpath->xOffset );
          *y = FT_MulFix( cf2_doubleToFixed( 1.0 + 0.7 ),
                          glyphpath->yOffset );
        }
      }
      else
      {
        /* third quadrant, -x -y */

        if ( NEG_INT32( dx ) > MUL_INT32( -2, dy ) )
        {
          /* -x */
          *x = 0;
          *y = MUL_INT32( 2, glyphpath->yOffset );
        }
        else if ( NEG_INT32( dy ) > MUL_INT32( -2, dx ) )
        {
          /* -y */
          *x = NEG_INT32( glyphpath->xOffset );
          *y = glyphpath->yOffset;
        }
        else
        {
          /* -x -y */
          *x = FT_MulFix( cf2_doubleToFixed( -0.7 ),
                          glyphpath->xOffset );
          *y = FT_MulFix( cf2_doubleToFixed( 1.0 + 0.7 ),
                          glyphpath->yOffset );
        }
      }
    }
  }


  /*
   * The functions cf2_glyphpath_{moveTo,lineTo,curveTo,closeOpenPath} are
   * called by the interpreter with Character Space (CS) coordinates.  Each
   * path element is placed into a queue of length one to await the
   * calculation of the following element.  At that time, the darkening
   * offset of the following element is known and joins can be computed,
   * including possible modification of this element, before mapping to
   * Device Space (DS) and passing it on to the outline consumer.
   *
   */
  FT_LOCAL_DEF( void )
  cf2_glyphpath_moveTo( CF2_GlyphPath  glyphpath,
                        CF2_Fixed      x,
                        CF2_Fixed      y )
  {
    cf2_glyphpath_closeOpenPath( glyphpath );

    /* save the parameters of the move for later, when we'll know how to */
    /* offset it;                                                        */
    /* also save last move point */
    glyphpath->currentCS.x = glyphpath->start.x = x;
    glyphpath->currentCS.y = glyphpath->start.y = y;

    glyphpath->moveIsPending = TRUE;

    /* ensure we have a valid map with current mask */
    if ( !cf2_hintmap_isValid( &glyphpath->hintMap ) ||
         cf2_hintmask_isNew( glyphpath->hintMask )   )
      cf2_hintmap_build( &glyphpath->hintMap,
                         glyphpath->hStemHintArray,
                         glyphpath->vStemHintArray,
                         glyphpath->hintMask,
                         glyphpath->hintOriginY,
                         FALSE );

    /* save a copy of current HintMap to use when drawing initial point */
    glyphpath->firstHintMap = glyphpath->hintMap;     /* structure copy */
  }


  FT_LOCAL_DEF( void )
  cf2_glyphpath_lineTo( CF2_GlyphPath  glyphpath,
                        CF2_Fixed      x,
                        CF2_Fixed      y )
  {
    CF2_Fixed  xOffset, yOffset;
    FT_Vector  P0, P1;
    FT_Bool    newHintMap;

    /*
     * New hints will be applied after cf2_glyphpath_pushPrevElem has run.
     * In case this is a synthesized closing line, any new hints should be
     * delayed until this path is closed (`cf2_hintmask_isNew' will be
     * called again before the next line or curve).
     */

    /* true if new hint map not on close */
    newHintMap = cf2_hintmask_isNew( glyphpath->hintMask ) &&
                 !glyphpath->pathIsClosing;

    /*
     * Zero-length lines may occur in the charstring.  Because we cannot
     * compute darkening offsets or intersections from zero-length lines,
     * it is best to remove them and avoid artifacts.  However, zero-length
     * lines in CS at the start of a new hint map can generate non-zero
     * lines in DS due to hint substitution.  We detect a change in hint
     * map here and pass those zero-length lines along.
     */

    /*
     * Note: Find explicitly closed paths here with a conditional
     *       breakpoint using
     *
     *         !gp->pathIsClosing && gp->start.x == x && gp->start.y == y
     *
     */

    if ( glyphpath->currentCS.x == x &&
         glyphpath->currentCS.y == y &&
         !newHintMap                 )
      /*
       * Ignore zero-length lines in CS where the hint map is the same
       * because the line in DS will also be zero length.
       *
       * Ignore zero-length lines when we synthesize a closing line because
       * the close will be handled in cf2_glyphPath_pushPrevElem.
       */
      return;

    cf2_glyphpath_computeOffset( glyphpath,
                                 glyphpath->currentCS.x,
                                 glyphpath->currentCS.y,
                                 x,
                                 y,
                                 &xOffset,
                                 &yOffset );

    /* construct offset points */
    P0.x = ADD_INT32( glyphpath->currentCS.x, xOffset );
    P0.y = ADD_INT32( glyphpath->currentCS.y, yOffset );
    P1.x = ADD_INT32( x, xOffset );
    P1.y = ADD_INT32( y, yOffset );

    if ( glyphpath->moveIsPending )
    {
      /* emit offset 1st point as MoveTo */
      cf2_glyphpath_pushMove( glyphpath, P0 );

      glyphpath->moveIsPending = FALSE;  /* adjust state machine */
      glyphpath->pathIsOpen    = TRUE;

      glyphpath->offsetStart1 = P1;              /* record second point */
    }

    if ( glyphpath->elemIsQueued )
    {
      FT_ASSERT( cf2_hintmap_isValid( &glyphpath->hintMap ) ||
                 glyphpath->hintMap.count == 0              );

      cf2_glyphpath_pushPrevElem( glyphpath,
                                  &glyphpath->hintMap,
                                  &P0,
                                  P1,
                                  FALSE );
    }

    /* queue the current element with offset points */
    glyphpath->elemIsQueued = TRUE;
    glyphpath->prevElemOp   = CF2_PathOpLineTo;
    glyphpath->prevElemP0   = P0;
    glyphpath->prevElemP1   = P1;

    /* update current map */
    if ( newHintMap )
      cf2_hintmap_build( &glyphpath->hintMap,
                         glyphpath->hStemHintArray,
                         glyphpath->vStemHintArray,
                         glyphpath->hintMask,
                         glyphpath->hintOriginY,
                         FALSE );

    glyphpath->currentCS.x = x;     /* pre-offset current point */
    glyphpath->currentCS.y = y;
  }


  FT_LOCAL_DEF( void )
  cf2_glyphpath_curveTo( CF2_GlyphPath  glyphpath,
                         CF2_Fixed      x1,
                         CF2_Fixed      y1,
                         CF2_Fixed      x2,
                         CF2_Fixed      y2,
                         CF2_Fixed      x3,
                         CF2_Fixed      y3 )
  {
    CF2_Fixed  xOffset1, yOffset1, xOffset3, yOffset3;
    FT_Vector  P0, P1, P2, P3;


    /* TODO: ignore zero length portions of curve?? */
    cf2_glyphpath_computeOffset( glyphpath,
                                 glyphpath->currentCS.x,
                                 glyphpath->currentCS.y,
                                 x1,
                                 y1,
                                 &xOffset1,
                                 &yOffset1 );
    cf2_glyphpath_computeOffset( glyphpath,
                                 x2,
                                 y2,
                                 x3,
                                 y3,
                                 &xOffset3,
                                 &yOffset3 );

    /* add momentum from the middle segment */
    glyphpath->callbacks->windingMomentum =
      ADD_INT32( glyphpath->callbacks->windingMomentum,
                 cf2_getWindingMomentum( x1, y1, x2, y2 ) );

    /* construct offset points */
    P0.x = ADD_INT32( glyphpath->currentCS.x, xOffset1 );
    P0.y = ADD_INT32( glyphpath->currentCS.y, yOffset1 );
    P1.x = ADD_INT32( x1, xOffset1 );
    P1.y = ADD_INT32( y1, yOffset1 );
    /* note: preserve angle of final segment by using offset3 at both ends */
    P2.x = ADD_INT32( x2, xOffset3 );
    P2.y = ADD_INT32( y2, yOffset3 );
    P3.x = ADD_INT32( x3, xOffset3 );
    P3.y = ADD_INT32( y3, yOffset3 );

    if ( glyphpath->moveIsPending )
    {
      /* emit offset 1st point as MoveTo */
      cf2_glyphpath_pushMove( glyphpath, P0 );

      glyphpath->moveIsPending = FALSE;
      glyphpath->pathIsOpen    = TRUE;

      glyphpath->offsetStart1 = P1;              /* record second point */
    }

    if ( glyphpath->elemIsQueued )
    {
      FT_ASSERT( cf2_hintmap_isValid( &glyphpath->hintMap ) ||
                 glyphpath->hintMap.count == 0              );

      cf2_glyphpath_pushPrevElem( glyphpath,
                                  &glyphpath->hintMap,
                                  &P0,
                                  P1,
                                  FALSE );
    }

    /* queue the current element with offset points */
    glyphpath->elemIsQueued = TRUE;
    glyphpath->prevElemOp   = CF2_PathOpCubeTo;
    glyphpath->prevElemP0   = P0;
    glyphpath->prevElemP1   = P1;
    glyphpath->prevElemP2   = P2;
    glyphpath->prevElemP3   = P3;

    /* update current map */
    if ( cf2_hintmask_isNew( glyphpath->hintMask ) )
      cf2_hintmap_build( &glyphpath->hintMap,
                         glyphpath->hStemHintArray,
                         glyphpath->vStemHintArray,
                         glyphpath->hintMask,
                         glyphpath->hintOriginY,
                         FALSE );

    glyphpath->currentCS.x = x3;       /* pre-offset current point */
    glyphpath->currentCS.y = y3;
  }


  FT_LOCAL_DEF( void )
  cf2_glyphpath_closeOpenPath( CF2_GlyphPath  glyphpath )
  {
    if ( glyphpath->pathIsOpen )
    {
      /*
       * A closing line in Character Space line is always generated below
       * with `cf2_glyphPath_lineTo'.  It may be ignored later if it turns
       * out to be zero length in Device Space.
       */
      glyphpath->pathIsClosing = TRUE;

      cf2_glyphpath_lineTo( glyphpath,
                            glyphpath->start.x,
                            glyphpath->start.y );

      /* empty the final element from the queue and close the path */
      if ( glyphpath->elemIsQueued )
        cf2_glyphpath_pushPrevElem( glyphpath,
                                    &glyphpath->hintMap,
                                    &glyphpath->offsetStart0,
                                    glyphpath->offsetStart1,
                                    TRUE );

      /* reset state machine */
      glyphpath->moveIsPending = TRUE;
      glyphpath->pathIsOpen    = FALSE;
      glyphpath->pathIsClosing = FALSE;
      glyphpath->elemIsQueued  = FALSE;
    }
  }


/* END */
