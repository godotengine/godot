/***************************************************************************/
/*                                                                         */
/*  aflatin.c                                                              */
/*                                                                         */
/*    Auto-fitter hinting routines for latin writing system (body).        */
/*                                                                         */
/*  Copyright 2003-2018 by                                                 */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


#include <ft2build.h>
#include FT_ADVANCES_H
#include FT_INTERNAL_DEBUG_H

#include "afglobal.h"
#include "afpic.h"
#include "aflatin.h"
#include "aferrors.h"


#ifdef AF_CONFIG_OPTION_USE_WARPER
#include "afwarp.h"
#endif


  /*************************************************************************/
  /*                                                                       */
  /* The macro FT_COMPONENT is used in trace mode.  It is an implicit      */
  /* parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log  */
  /* messages during execution.                                            */
  /*                                                                       */
#undef  FT_COMPONENT
#define FT_COMPONENT  trace_aflatin


  /* needed for computation of round vs. flat segments */
#define FLAT_THRESHOLD( x )  ( x / 14 )


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****            L A T I N   G L O B A L   M E T R I C S            *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/


  /* Find segments and links, compute all stem widths, and initialize */
  /* standard width and height for the glyph with given charcode.     */

  FT_LOCAL_DEF( void )
  af_latin_metrics_init_widths( AF_LatinMetrics  metrics,
                                FT_Face          face )
  {
    /* scan the array of segments in each direction */
    AF_GlyphHintsRec  hints[1];


    FT_TRACE5(( "\n"
                "latin standard widths computation (style `%s')\n"
                "=====================================================\n"
                "\n",
                af_style_names[metrics->root.style_class->style] ));

    af_glyph_hints_init( hints, face->memory );

    metrics->axis[AF_DIMENSION_HORZ].width_count = 0;
    metrics->axis[AF_DIMENSION_VERT].width_count = 0;

    {
      FT_Error            error;
      FT_ULong            glyph_index;
      int                 dim;
      AF_LatinMetricsRec  dummy[1];
      AF_Scaler           scaler = &dummy->root.scaler;

#ifdef FT_CONFIG_OPTION_PIC
      AF_FaceGlobals  globals = metrics->root.globals;
#endif

      AF_StyleClass   style_class  = metrics->root.style_class;
      AF_ScriptClass  script_class = AF_SCRIPT_CLASSES_GET
                                       [style_class->script];

      void*        shaper_buf;
      const char*  p;

#ifdef FT_DEBUG_LEVEL_TRACE
      FT_ULong  ch = 0;
#endif

      p          = script_class->standard_charstring;
      shaper_buf = af_shaper_buf_create( face );

      /*
       * We check a list of standard characters to catch features like
       * `c2sc' (small caps from caps) that don't contain lowercase letters
       * by definition, or other features that mainly operate on numerals.
       * The first match wins.
       */

      glyph_index = 0;
      while ( *p )
      {
        unsigned int  num_idx;

#ifdef FT_DEBUG_LEVEL_TRACE
        const char*  p_old;
#endif


        while ( *p == ' ' )
          p++;

#ifdef FT_DEBUG_LEVEL_TRACE
        p_old = p;
        GET_UTF8_CHAR( ch, p_old );
#endif

        /* reject input that maps to more than a single glyph */
        p = af_shaper_get_cluster( p, &metrics->root, shaper_buf, &num_idx );
        if ( num_idx > 1 )
          continue;

        /* otherwise exit loop if we have a result */
        glyph_index = af_shaper_get_elem( &metrics->root,
                                          shaper_buf,
                                          0,
                                          NULL,
                                          NULL );
        if ( glyph_index )
          break;
      }

      af_shaper_buf_destroy( face, shaper_buf );

      if ( !glyph_index )
        goto Exit;

      FT_TRACE5(( "standard character: U+%04lX (glyph index %d)\n",
                  ch, glyph_index ));

      error = FT_Load_Glyph( face, glyph_index, FT_LOAD_NO_SCALE );
      if ( error || face->glyph->outline.n_points <= 0 )
        goto Exit;

      FT_ZERO( dummy );

      dummy->units_per_em = metrics->units_per_em;

      scaler->x_scale = 0x10000L;
      scaler->y_scale = 0x10000L;
      scaler->x_delta = 0;
      scaler->y_delta = 0;

      scaler->face        = face;
      scaler->render_mode = FT_RENDER_MODE_NORMAL;
      scaler->flags       = 0;

      af_glyph_hints_rescale( hints, (AF_StyleMetrics)dummy );

      error = af_glyph_hints_reload( hints, &face->glyph->outline );
      if ( error )
        goto Exit;

      for ( dim = 0; dim < AF_DIMENSION_MAX; dim++ )
      {
        AF_LatinAxis  axis    = &metrics->axis[dim];
        AF_AxisHints  axhints = &hints->axis[dim];
        AF_Segment    seg, limit, link;
        FT_UInt       num_widths = 0;


        error = af_latin_hints_compute_segments( hints,
                                                 (AF_Dimension)dim );
        if ( error )
          goto Exit;

        /*
         *  We assume that the glyphs selected for the stem width
         *  computation are `featureless' enough so that the linking
         *  algorithm works fine without adjustments of its scoring
         *  function.
         */
        af_latin_hints_link_segments( hints,
                                      0,
                                      NULL,
                                      (AF_Dimension)dim );

        seg   = axhints->segments;
        limit = seg + axhints->num_segments;

        for ( ; seg < limit; seg++ )
        {
          link = seg->link;

          /* we only consider stem segments there! */
          if ( link && link->link == seg && link > seg )
          {
            FT_Pos  dist;


            dist = seg->pos - link->pos;
            if ( dist < 0 )
              dist = -dist;

            if ( num_widths < AF_LATIN_MAX_WIDTHS )
              axis->widths[num_widths++].org = dist;
          }
        }

        /* this also replaces multiple almost identical stem widths */
        /* with a single one (the value 100 is heuristic)           */
        af_sort_and_quantize_widths( &num_widths, axis->widths,
                                     dummy->units_per_em / 100 );
        axis->width_count = num_widths;
      }

    Exit:
      for ( dim = 0; dim < AF_DIMENSION_MAX; dim++ )
      {
        AF_LatinAxis  axis = &metrics->axis[dim];
        FT_Pos        stdw;


        stdw = ( axis->width_count > 0 ) ? axis->widths[0].org
                                         : AF_LATIN_CONSTANT( metrics, 50 );

        /* let's try 20% of the smallest width */
        axis->edge_distance_threshold = stdw / 5;
        axis->standard_width          = stdw;
        axis->extra_light             = 0;

#ifdef FT_DEBUG_LEVEL_TRACE
        {
          FT_UInt  i;


          FT_TRACE5(( "%s widths:\n",
                      dim == AF_DIMENSION_VERT ? "horizontal"
                                               : "vertical" ));

          FT_TRACE5(( "  %d (standard)", axis->standard_width ));
          for ( i = 1; i < axis->width_count; i++ )
            FT_TRACE5(( " %d", axis->widths[i].org ));

          FT_TRACE5(( "\n" ));
        }
#endif
      }
    }

    FT_TRACE5(( "\n" ));

    af_glyph_hints_done( hints );
  }


  static void
  af_latin_sort_blue( FT_UInt        count,
                      AF_LatinBlue*  table )
  {
    FT_UInt       i, j;
    AF_LatinBlue  swap;


    /* we sort from bottom to top */
    for ( i = 1; i < count; i++ )
    {
      for ( j = i; j > 0; j-- )
      {
        FT_Pos  a, b;


        if ( table[j - 1]->flags & ( AF_LATIN_BLUE_TOP     |
                                     AF_LATIN_BLUE_SUB_TOP ) )
          a = table[j - 1]->ref.org;
        else
          a = table[j - 1]->shoot.org;

        if ( table[j]->flags & ( AF_LATIN_BLUE_TOP     |
                                 AF_LATIN_BLUE_SUB_TOP ) )
          b = table[j]->ref.org;
        else
          b = table[j]->shoot.org;

        if ( b >= a )
          break;

        swap         = table[j];
        table[j]     = table[j - 1];
        table[j - 1] = swap;
      }
    }
  }


  /* Find all blue zones.  Flat segments give the reference points, */
  /* round segments the overshoot positions.                        */

  static void
  af_latin_metrics_init_blues( AF_LatinMetrics  metrics,
                               FT_Face          face )
  {
    FT_Pos        flats [AF_BLUE_STRING_MAX_LEN];
    FT_Pos        rounds[AF_BLUE_STRING_MAX_LEN];

    FT_UInt       num_flats;
    FT_UInt       num_rounds;

    AF_LatinBlue  blue;
    FT_Error      error;
    AF_LatinAxis  axis = &metrics->axis[AF_DIMENSION_VERT];
    FT_Outline    outline;

    AF_StyleClass  sc = metrics->root.style_class;

    AF_Blue_Stringset         bss = sc->blue_stringset;
    const AF_Blue_StringRec*  bs  = &af_blue_stringsets[bss];

    FT_Pos  flat_threshold = FLAT_THRESHOLD( metrics->units_per_em );

    void*  shaper_buf;


    /* we walk over the blue character strings as specified in the */
    /* style's entry in the `af_blue_stringset' array              */

    FT_TRACE5(( "latin blue zones computation\n"
                "============================\n"
                "\n" ));

    shaper_buf = af_shaper_buf_create( face );

    for ( ; bs->string != AF_BLUE_STRING_MAX; bs++ )
    {
      const char*  p = &af_blue_strings[bs->string];
      FT_Pos*      blue_ref;
      FT_Pos*      blue_shoot;
      FT_Pos       ascender;
      FT_Pos       descender;


#ifdef FT_DEBUG_LEVEL_TRACE
      {
        FT_Bool  have_flag = 0;


        FT_TRACE5(( "blue zone %d", axis->blue_count ));

        if ( bs->properties )
        {
          FT_TRACE5(( " (" ));

          if ( AF_LATIN_IS_TOP_BLUE( bs ) )
          {
            FT_TRACE5(( "top" ));
            have_flag = 1;
          }
          else if ( AF_LATIN_IS_SUB_TOP_BLUE( bs ) )
          {
            FT_TRACE5(( "sub top" ));
            have_flag = 1;
          }

          if ( AF_LATIN_IS_NEUTRAL_BLUE( bs ) )
          {
            if ( have_flag )
              FT_TRACE5(( ", " ));
            FT_TRACE5(( "neutral" ));
            have_flag = 1;
          }

          if ( AF_LATIN_IS_X_HEIGHT_BLUE( bs ) )
          {
            if ( have_flag )
              FT_TRACE5(( ", " ));
            FT_TRACE5(( "small top" ));
            have_flag = 1;
          }

          if ( AF_LATIN_IS_LONG_BLUE( bs ) )
          {
            if ( have_flag )
              FT_TRACE5(( ", " ));
            FT_TRACE5(( "long" ));
          }

          FT_TRACE5(( ")" ));
        }

        FT_TRACE5(( ":\n" ));
      }
#endif /* FT_DEBUG_LEVEL_TRACE */

      num_flats  = 0;
      num_rounds = 0;
      ascender   = 0;
      descender  = 0;

      while ( *p )
      {
        FT_ULong    glyph_index;
        FT_Long     y_offset;
        FT_Int      best_point, best_contour_first, best_contour_last;
        FT_Vector*  points;

        FT_Pos   best_y_extremum;                      /* same as points.y */
        FT_Bool  best_round = 0;

        unsigned int  i, num_idx;

#ifdef FT_DEBUG_LEVEL_TRACE
        const char*  p_old;
        FT_ULong     ch;
#endif


        while ( *p == ' ' )
          p++;

#ifdef FT_DEBUG_LEVEL_TRACE
        p_old = p;
        GET_UTF8_CHAR( ch, p_old );
#endif

        p = af_shaper_get_cluster( p, &metrics->root, shaper_buf, &num_idx );

        if ( !num_idx )
        {
          FT_TRACE5(( "  U+%04lX unavailable\n", ch ));
          continue;
        }

        if ( AF_LATIN_IS_TOP_BLUE( bs ) )
          best_y_extremum = FT_INT_MIN;
        else
          best_y_extremum = FT_INT_MAX;

        /* iterate over all glyph elements of the character cluster */
        /* and get the data of the `biggest' one                    */
        for ( i = 0; i < num_idx; i++ )
        {
          FT_Pos   best_y;
          FT_Bool  round = 0;


          /* load the character in the face -- skip unknown or empty ones */
          glyph_index = af_shaper_get_elem( &metrics->root,
                                            shaper_buf,
                                            i,
                                            NULL,
                                            &y_offset );
          if ( glyph_index == 0 )
          {
            FT_TRACE5(( "  U+%04lX unavailable\n", ch ));
            continue;
          }

          error   = FT_Load_Glyph( face, glyph_index, FT_LOAD_NO_SCALE );
          outline = face->glyph->outline;
          /* reject glyphs that don't produce any rendering */
          if ( error || outline.n_points <= 2 )
          {
#ifdef FT_DEBUG_LEVEL_TRACE
            if ( num_idx == 1 )
              FT_TRACE5(( "  U+%04lX contains no (usable) outlines\n", ch ));
            else
              FT_TRACE5(( "  component %d of cluster starting with U+%04lX"
                          " contains no (usable) outlines\n", i, ch ));
#endif
            continue;
          }

          /* now compute min or max point indices and coordinates */
          points             = outline.points;
          best_point         = -1;
          best_y             = 0;  /* make compiler happy */
          best_contour_first = 0;  /* ditto */
          best_contour_last  = 0;  /* ditto */

          {
            FT_Int  nn;
            FT_Int  first = 0;
            FT_Int  last  = -1;


            for ( nn = 0; nn < outline.n_contours; first = last + 1, nn++ )
            {
              FT_Int  old_best_point = best_point;
              FT_Int  pp;


              last = outline.contours[nn];

              /* Avoid single-point contours since they are never      */
              /* rasterized.  In some fonts, they correspond to mark   */
              /* attachment points that are way outside of the glyph's */
              /* real outline.                                         */
              if ( last <= first )
                continue;

              if ( AF_LATIN_IS_TOP_BLUE( bs )     ||
                   AF_LATIN_IS_SUB_TOP_BLUE( bs ) )
              {
                for ( pp = first; pp <= last; pp++ )
                {
                  if ( best_point < 0 || points[pp].y > best_y )
                  {
                    best_point = pp;
                    best_y     = points[pp].y;
                    ascender   = FT_MAX( ascender, best_y + y_offset );
                  }
                  else
                    descender = FT_MIN( descender, points[pp].y + y_offset );
                }
              }
              else
              {
                for ( pp = first; pp <= last; pp++ )
                {
                  if ( best_point < 0 || points[pp].y < best_y )
                  {
                    best_point = pp;
                    best_y     = points[pp].y;
                    descender  = FT_MIN( descender, best_y + y_offset );
                  }
                  else
                    ascender = FT_MAX( ascender, points[pp].y + y_offset );
                }
              }

              if ( best_point != old_best_point )
              {
                best_contour_first = first;
                best_contour_last  = last;
              }
            }
          }

          /* now check whether the point belongs to a straight or round   */
          /* segment; we first need to find in which contour the extremum */
          /* lies, then inspect its previous and next points              */
          if ( best_point >= 0 )
          {
            FT_Pos  best_x = points[best_point].x;
            FT_Int  prev, next;
            FT_Int  best_segment_first, best_segment_last;
            FT_Int  best_on_point_first, best_on_point_last;
            FT_Pos  dist;


            best_segment_first = best_point;
            best_segment_last  = best_point;

            if ( FT_CURVE_TAG( outline.tags[best_point] ) == FT_CURVE_TAG_ON )
            {
              best_on_point_first = best_point;
              best_on_point_last  = best_point;
            }
            else
            {
              best_on_point_first = -1;
              best_on_point_last  = -1;
            }

            /* look for the previous and next points on the contour  */
            /* that are not on the same Y coordinate, then threshold */
            /* the `closeness'...                                    */
            prev = best_point;
            next = prev;

            do
            {
              if ( prev > best_contour_first )
                prev--;
              else
                prev = best_contour_last;

              dist = FT_ABS( points[prev].y - best_y );
              /* accept a small distance or a small angle (both values are */
              /* heuristic; value 20 corresponds to approx. 2.9 degrees)   */
              if ( dist > 5 )
                if ( FT_ABS( points[prev].x - best_x ) <= 20 * dist )
                  break;

              best_segment_first = prev;

              if ( FT_CURVE_TAG( outline.tags[prev] ) == FT_CURVE_TAG_ON )
              {
                best_on_point_first = prev;
                if ( best_on_point_last < 0 )
                  best_on_point_last = prev;
              }

            } while ( prev != best_point );

            do
            {
              if ( next < best_contour_last )
                next++;
              else
                next = best_contour_first;

              dist = FT_ABS( points[next].y - best_y );
              if ( dist > 5 )
                if ( FT_ABS( points[next].x - best_x ) <= 20 * dist )
                  break;

              best_segment_last = next;

              if ( FT_CURVE_TAG( outline.tags[next] ) == FT_CURVE_TAG_ON )
              {
                best_on_point_last = next;
                if ( best_on_point_first < 0 )
                  best_on_point_first = next;
              }

            } while ( next != best_point );

            if ( AF_LATIN_IS_LONG_BLUE( bs ) )
            {
              /* If this flag is set, we have an additional constraint to  */
              /* get the blue zone distance: Find a segment of the topmost */
              /* (or bottommost) contour that is longer than a heuristic   */
              /* threshold.  This ensures that small bumps in the outline  */
              /* are ignored (for example, the `vertical serifs' found in  */
              /* many Hebrew glyph designs).                               */

              /* If this segment is long enough, we are done.  Otherwise,  */
              /* search the segment next to the extremum that is long      */
              /* enough, has the same direction, and a not too large       */
              /* vertical distance from the extremum.  Note that the       */
              /* algorithm doesn't check whether the found segment is      */
              /* actually the one (vertically) nearest to the extremum.    */

              /* heuristic threshold value */
              FT_Pos  length_threshold = metrics->units_per_em / 25;


              dist = FT_ABS( points[best_segment_last].x -
                               points[best_segment_first].x );

              if ( dist < length_threshold                       &&
                   best_segment_last - best_segment_first + 2 <=
                     best_contour_last - best_contour_first      )
              {
                /* heuristic threshold value */
                FT_Pos  height_threshold = metrics->units_per_em / 4;

                FT_Int   first;
                FT_Int   last;
                FT_Bool  hit;

                /* we intentionally declare these two variables        */
                /* outside of the loop since various compilers emit    */
                /* incorrect warning messages otherwise, talking about */
                /* `possibly uninitialized variables'                  */
                FT_Int  p_first = 0;            /* make compiler happy */
                FT_Int  p_last  = 0;

                FT_Bool  left2right;


                /* compute direction */
                prev = best_point;

                do
                {
                  if ( prev > best_contour_first )
                    prev--;
                  else
                    prev = best_contour_last;

                  if ( points[prev].x != best_x )
                    break;

                } while ( prev != best_point );

                /* skip glyph for the degenerate case */
                if ( prev == best_point )
                  continue;

                left2right = FT_BOOL( points[prev].x < points[best_point].x );

                first = best_segment_last;
                last  = first;
                hit   = 0;

                do
                {
                  FT_Bool  l2r;
                  FT_Pos   d;


                  if ( !hit )
                  {
                    /* no hit; adjust first point */
                    first = last;

                    /* also adjust first and last on point */
                    if ( FT_CURVE_TAG( outline.tags[first] ) ==
                           FT_CURVE_TAG_ON )
                    {
                      p_first = first;
                      p_last  = first;
                    }
                    else
                    {
                      p_first = -1;
                      p_last  = -1;
                    }

                    hit = 1;
                  }

                  if ( last < best_contour_last )
                    last++;
                  else
                    last = best_contour_first;

                  if ( FT_ABS( best_y - points[first].y ) > height_threshold )
                  {
                    /* vertical distance too large */
                    hit = 0;
                    continue;
                  }

                  /* same test as above */
                  dist = FT_ABS( points[last].y - points[first].y );
                  if ( dist > 5 )
                    if ( FT_ABS( points[last].x - points[first].x ) <=
                           20 * dist )
                    {
                      hit = 0;
                      continue;
                    }

                  if ( FT_CURVE_TAG( outline.tags[last] ) == FT_CURVE_TAG_ON )
                  {
                    p_last = last;
                    if ( p_first < 0 )
                      p_first = last;
                  }

                  l2r = FT_BOOL( points[first].x < points[last].x );
                  d   = FT_ABS( points[last].x - points[first].x );

                  if ( l2r == left2right     &&
                       d >= length_threshold )
                  {
                    /* all constraints are met; update segment after */
                    /* finding its end                               */
                    do
                    {
                      if ( last < best_contour_last )
                        last++;
                      else
                        last = best_contour_first;

                      d = FT_ABS( points[last].y - points[first].y );
                      if ( d > 5 )
                        if ( FT_ABS( points[next].x - points[first].x ) <=
                               20 * dist )
                        {
                          if ( last > best_contour_first )
                            last--;
                          else
                            last = best_contour_last;
                          break;
                        }

                      p_last = last;

                      if ( FT_CURVE_TAG( outline.tags[last] ) ==
                             FT_CURVE_TAG_ON )
                      {
                        p_last = last;
                        if ( p_first < 0 )
                          p_first = last;
                      }

                    } while ( last != best_segment_first );

                    best_y = points[first].y;

                    best_segment_first = first;
                    best_segment_last  = last;

                    best_on_point_first = p_first;
                    best_on_point_last  = p_last;

                    break;
                  }

                } while ( last != best_segment_first );
              }
            }

            /* for computing blue zones, we add the y offset as returned */
            /* by the currently used OpenType feature -- for example,    */
            /* superscript glyphs might be identical to subscript glyphs */
            /* with a vertical shift                                     */
            best_y += y_offset;

#ifdef FT_DEBUG_LEVEL_TRACE
            if ( num_idx == 1 )
              FT_TRACE5(( "  U+%04lX: best_y = %5ld", ch, best_y ));
            else
              FT_TRACE5(( "  component %d of cluster starting with U+%04lX:"
                          " best_y = %5ld", i, ch, best_y ));
#endif

            /* now set the `round' flag depending on the segment's kind: */
            /*                                                           */
            /* - if the horizontal distance between the first and last   */
            /*   `on' point is larger than a heuristic threshold         */
            /*   we have a flat segment                                  */
            /* - if either the first or the last point of the segment is */
            /*   an `off' point, the segment is round, otherwise it is   */
            /*   flat                                                    */
            if ( best_on_point_first >= 0                               &&
                 best_on_point_last >= 0                                &&
                 ( FT_ABS( points[best_on_point_last].x -
                           points[best_on_point_first].x ) ) >
                   flat_threshold                                       )
              round = 0;
            else
              round = FT_BOOL(
                        FT_CURVE_TAG( outline.tags[best_segment_first] ) !=
                          FT_CURVE_TAG_ON                                   ||
                        FT_CURVE_TAG( outline.tags[best_segment_last]  ) !=
                          FT_CURVE_TAG_ON                                   );

            if ( round && AF_LATIN_IS_NEUTRAL_BLUE( bs ) )
            {
              /* only use flat segments for a neutral blue zone */
              FT_TRACE5(( " (round, skipped)\n" ));
              continue;
            }

            FT_TRACE5(( " (%s)\n", round ? "round" : "flat" ));
          }

          if ( AF_LATIN_IS_TOP_BLUE( bs ) )
          {
            if ( best_y > best_y_extremum )
            {
              best_y_extremum = best_y;
              best_round      = round;
            }
          }
          else
          {
            if ( best_y < best_y_extremum )
            {
              best_y_extremum = best_y;
              best_round      = round;
            }
          }

        } /* end for loop */

        if ( !( best_y_extremum == FT_INT_MIN ||
                best_y_extremum == FT_INT_MAX ) )
        {
          if ( best_round )
            rounds[num_rounds++] = best_y_extremum;
          else
            flats[num_flats++]   = best_y_extremum;
        }

      } /* end while loop */

      if ( num_flats == 0 && num_rounds == 0 )
      {
        /*
         *  we couldn't find a single glyph to compute this blue zone,
         *  we will simply ignore it then
         */
        FT_TRACE5(( "  empty\n" ));
        continue;
      }

      /* we have computed the contents of the `rounds' and `flats' tables, */
      /* now determine the reference and overshoot position of the blue -- */
      /* we simply take the median value after a simple sort               */
      af_sort_pos( num_rounds, rounds );
      af_sort_pos( num_flats,  flats );

      blue       = &axis->blues[axis->blue_count];
      blue_ref   = &blue->ref.org;
      blue_shoot = &blue->shoot.org;

      axis->blue_count++;

      if ( num_flats == 0 )
      {
        *blue_ref   =
        *blue_shoot = rounds[num_rounds / 2];
      }
      else if ( num_rounds == 0 )
      {
        *blue_ref   =
        *blue_shoot = flats[num_flats / 2];
      }
      else
      {
        *blue_ref   = flats [num_flats  / 2];
        *blue_shoot = rounds[num_rounds / 2];
      }

      /* there are sometimes problems: if the overshoot position of top     */
      /* zones is under its reference position, or the opposite for bottom  */
      /* zones.  We must thus check everything there and correct the errors */
      if ( *blue_shoot != *blue_ref )
      {
        FT_Pos   ref      = *blue_ref;
        FT_Pos   shoot    = *blue_shoot;
        FT_Bool  over_ref = FT_BOOL( shoot > ref );


        if ( ( AF_LATIN_IS_TOP_BLUE( bs )    ||
               AF_LATIN_IS_SUB_TOP_BLUE( bs) ) ^ over_ref )
        {
          *blue_ref   =
          *blue_shoot = ( shoot + ref ) / 2;

          FT_TRACE5(( "  [overshoot smaller than reference,"
                      " taking mean value]\n" ));
        }
      }

      blue->ascender  = ascender;
      blue->descender = descender;

      blue->flags = 0;
      if ( AF_LATIN_IS_TOP_BLUE( bs ) )
        blue->flags |= AF_LATIN_BLUE_TOP;
      if ( AF_LATIN_IS_SUB_TOP_BLUE( bs ) )
        blue->flags |= AF_LATIN_BLUE_SUB_TOP;
      if ( AF_LATIN_IS_NEUTRAL_BLUE( bs ) )
        blue->flags |= AF_LATIN_BLUE_NEUTRAL;

      /*
       * The following flag is used later to adjust the y and x scales
       * in order to optimize the pixel grid alignment of the top of small
       * letters.
       */
      if ( AF_LATIN_IS_X_HEIGHT_BLUE( bs ) )
        blue->flags |= AF_LATIN_BLUE_ADJUSTMENT;

      FT_TRACE5(( "    -> reference = %ld\n"
                  "       overshoot = %ld\n",
                  *blue_ref, *blue_shoot ));

    } /* end for loop */

    af_shaper_buf_destroy( face, shaper_buf );

    /* we finally check whether blue zones are ordered; */
    /* `ref' and `shoot' values of two blue zones must not overlap */
    if ( axis->blue_count )
    {
      FT_UInt       i;
      AF_LatinBlue  blue_sorted[AF_BLUE_STRINGSET_MAX_LEN + 2];


      for ( i = 0; i < axis->blue_count; i++ )
        blue_sorted[i] = &axis->blues[i];

      /* sort bottoms of blue zones... */
      af_latin_sort_blue( axis->blue_count, blue_sorted );

      /* ...and adjust top values if necessary */
      for ( i = 0; i < axis->blue_count - 1; i++ )
      {
        FT_Pos*  a;
        FT_Pos*  b;

#ifdef FT_DEBUG_LEVEL_TRACE
        FT_Bool  a_is_top = 0;
#endif


        if ( blue_sorted[i]->flags & ( AF_LATIN_BLUE_TOP     |
                                       AF_LATIN_BLUE_SUB_TOP ) )
        {
          a = &blue_sorted[i]->shoot.org;
#ifdef FT_DEBUG_LEVEL_TRACE
          a_is_top = 1;
#endif
        }
        else
          a = &blue_sorted[i]->ref.org;

        if ( blue_sorted[i + 1]->flags & ( AF_LATIN_BLUE_TOP     |
                                           AF_LATIN_BLUE_SUB_TOP ) )
          b = &blue_sorted[i + 1]->shoot.org;
        else
          b = &blue_sorted[i + 1]->ref.org;

        if ( *a > *b )
        {
          *a = *b;
          FT_TRACE5(( "blue zone overlap:"
                      " adjusting %s %d to %ld\n",
                      a_is_top ? "overshoot" : "reference",
                      blue_sorted[i] - axis->blues,
                      *a ));
        }
      }
    }

    FT_TRACE5(( "\n" ));

    return;
  }


  /* Check whether all ASCII digits have the same advance width. */

  FT_LOCAL_DEF( void )
  af_latin_metrics_check_digits( AF_LatinMetrics  metrics,
                                 FT_Face          face )
  {
    FT_Bool   started = 0, same_width = 1;
    FT_Fixed  advance = 0, old_advance = 0;

    void*  shaper_buf;

    /* in all supported charmaps, digits have character codes 0x30-0x39 */
    const char   digits[] = "0 1 2 3 4 5 6 7 8 9";
    const char*  p;


    p          = digits;
    shaper_buf = af_shaper_buf_create( face );

    while ( *p )
    {
      FT_ULong      glyph_index;
      unsigned int  num_idx;


      /* reject input that maps to more than a single glyph */
      p = af_shaper_get_cluster( p, &metrics->root, shaper_buf, &num_idx );
      if ( num_idx > 1 )
        continue;

      glyph_index = af_shaper_get_elem( &metrics->root,
                                        shaper_buf,
                                        0,
                                        &advance,
                                        NULL );
      if ( !glyph_index )
        continue;

      if ( started )
      {
        if ( advance != old_advance )
        {
          same_width = 0;
          break;
        }
      }
      else
      {
        old_advance = advance;
        started     = 1;
      }
    }

    af_shaper_buf_destroy( face, shaper_buf );

    metrics->root.digits_have_same_width = same_width;
  }


  /* Initialize global metrics. */

  FT_LOCAL_DEF( FT_Error )
  af_latin_metrics_init( AF_LatinMetrics  metrics,
                         FT_Face          face )
  {
    FT_CharMap  oldmap = face->charmap;


    metrics->units_per_em = face->units_per_EM;

    if ( !FT_Select_Charmap( face, FT_ENCODING_UNICODE ) )
    {
      af_latin_metrics_init_widths( metrics, face );
      af_latin_metrics_init_blues( metrics, face );
      af_latin_metrics_check_digits( metrics, face );
    }

    FT_Set_Charmap( face, oldmap );
    return FT_Err_Ok;
  }


  /* Adjust scaling value, then scale and shift widths   */
  /* and blue zones (if applicable) for given dimension. */

  static void
  af_latin_metrics_scale_dim( AF_LatinMetrics  metrics,
                              AF_Scaler        scaler,
                              AF_Dimension     dim )
  {
    FT_Fixed      scale;
    FT_Pos        delta;
    AF_LatinAxis  axis;
    FT_UInt       nn;


    if ( dim == AF_DIMENSION_HORZ )
    {
      scale = scaler->x_scale;
      delta = scaler->x_delta;
    }
    else
    {
      scale = scaler->y_scale;
      delta = scaler->y_delta;
    }

    axis = &metrics->axis[dim];

    if ( axis->org_scale == scale && axis->org_delta == delta )
      return;

    axis->org_scale = scale;
    axis->org_delta = delta;

    /*
     * correct X and Y scale to optimize the alignment of the top of small
     * letters to the pixel grid
     */
    {
      AF_LatinAxis  Axis = &metrics->axis[AF_DIMENSION_VERT];
      AF_LatinBlue  blue = NULL;


      for ( nn = 0; nn < Axis->blue_count; nn++ )
      {
        if ( Axis->blues[nn].flags & AF_LATIN_BLUE_ADJUSTMENT )
        {
          blue = &Axis->blues[nn];
          break;
        }
      }

      if ( blue )
      {
        FT_Pos   scaled;
        FT_Pos   threshold;
        FT_Pos   fitted;
        FT_UInt  limit;
        FT_UInt  ppem;


        scaled    = FT_MulFix( blue->shoot.org, scale );
        ppem      = metrics->root.scaler.face->size->metrics.x_ppem;
        limit     = metrics->root.globals->increase_x_height;
        threshold = 40;

        /* if the `increase-x-height' property is active, */
        /* we round up much more often                    */
        if ( limit                                 &&
             ppem <= limit                         &&
             ppem >= AF_PROP_INCREASE_X_HEIGHT_MIN )
          threshold = 52;

        fitted = ( scaled + threshold ) & ~63;

        if ( scaled != fitted )
        {
#if 0
          if ( dim == AF_DIMENSION_HORZ )
          {
            if ( fitted < scaled )
              scale -= scale / 50;  /* scale *= 0.98 */
          }
          else
#endif
          if ( dim == AF_DIMENSION_VERT )
          {
            FT_Pos    max_height;
            FT_Pos    dist;
            FT_Fixed  new_scale;


            new_scale = FT_MulDiv( scale, fitted, scaled );

            /* the scaling should not change the result by more than two pixels */
            max_height = metrics->units_per_em;

            for ( nn = 0; nn < Axis->blue_count; nn++ )
            {
              max_height = FT_MAX( max_height, Axis->blues[nn].ascender );
              max_height = FT_MAX( max_height, -Axis->blues[nn].descender );
            }

            dist  = FT_ABS( FT_MulFix( max_height, new_scale - scale ) );
            dist &= ~127;

            if ( dist == 0 )
            {
              FT_TRACE5((
                "af_latin_metrics_scale_dim:"
                " x height alignment (style `%s'):\n"
                "                           "
                " vertical scaling changed from %.5f to %.5f (by %d%%)\n"
                "\n",
                af_style_names[metrics->root.style_class->style],
                scale / 65536.0,
                new_scale / 65536.0,
                ( fitted - scaled ) * 100 / scaled ));

              scale = new_scale;
            }
#ifdef FT_DEBUG_LEVEL_TRACE
            else
            {
              FT_TRACE5((
                "af_latin_metrics_scale_dim:"
                " x height alignment (style `%s'):\n"
                "                           "
                " excessive vertical scaling abandoned\n"
                "\n",
                af_style_names[metrics->root.style_class->style] ));
            }
#endif
          }
        }
      }
    }

    axis->scale = scale;
    axis->delta = delta;

    if ( dim == AF_DIMENSION_HORZ )
    {
      metrics->root.scaler.x_scale = scale;
      metrics->root.scaler.x_delta = delta;
    }
    else
    {
      metrics->root.scaler.y_scale = scale;
      metrics->root.scaler.y_delta = delta;
    }

    FT_TRACE5(( "%s widths (style `%s')\n",
                dim == AF_DIMENSION_HORZ ? "horizontal" : "vertical",
                af_style_names[metrics->root.style_class->style] ));

    /* scale the widths */
    for ( nn = 0; nn < axis->width_count; nn++ )
    {
      AF_Width  width = axis->widths + nn;


      width->cur = FT_MulFix( width->org, scale );
      width->fit = width->cur;

      FT_TRACE5(( "  %d scaled to %.2f\n",
                  width->org,
                  width->cur / 64.0 ));
    }

    FT_TRACE5(( "\n" ));

    /* an extra-light axis corresponds to a standard width that is */
    /* smaller than 5/8 pixels                                     */
    axis->extra_light =
      (FT_Bool)( FT_MulFix( axis->standard_width, scale ) < 32 + 8 );

#ifdef FT_DEBUG_LEVEL_TRACE
    if ( axis->extra_light )
      FT_TRACE5(( "`%s' style is extra light (at current resolution)\n"
                  "\n",
                  af_style_names[metrics->root.style_class->style] ));
#endif

    if ( dim == AF_DIMENSION_VERT )
    {
#ifdef FT_DEBUG_LEVEL_TRACE
      if ( axis->blue_count )
        FT_TRACE5(( "blue zones (style `%s')\n",
                    af_style_names[metrics->root.style_class->style] ));
#endif

      /* scale the blue zones */
      for ( nn = 0; nn < axis->blue_count; nn++ )
      {
        AF_LatinBlue  blue = &axis->blues[nn];
        FT_Pos        dist;


        blue->ref.cur   = FT_MulFix( blue->ref.org, scale ) + delta;
        blue->ref.fit   = blue->ref.cur;
        blue->shoot.cur = FT_MulFix( blue->shoot.org, scale ) + delta;
        blue->shoot.fit = blue->shoot.cur;
        blue->flags    &= ~AF_LATIN_BLUE_ACTIVE;

        /* a blue zone is only active if it is less than 3/4 pixels tall */
        dist = FT_MulFix( blue->ref.org - blue->shoot.org, scale );
        if ( dist <= 48 && dist >= -48 )
        {
#if 0
          FT_Pos  delta1;
#endif
          FT_Pos  delta2;


          /* use discrete values for blue zone widths */

#if 0

          /* generic, original code */
          delta1 = blue->shoot.org - blue->ref.org;
          delta2 = delta1;
          if ( delta1 < 0 )
            delta2 = -delta2;

          delta2 = FT_MulFix( delta2, scale );

          if ( delta2 < 32 )
            delta2 = 0;
          else if ( delta2 < 64 )
            delta2 = 32 + ( ( ( delta2 - 32 ) + 16 ) & ~31 );
          else
            delta2 = FT_PIX_ROUND( delta2 );

          if ( delta1 < 0 )
            delta2 = -delta2;

          blue->ref.fit   = FT_PIX_ROUND( blue->ref.cur );
          blue->shoot.fit = blue->ref.fit + delta2;

#else

          /* simplified version due to abs(dist) <= 48 */
          delta2 = dist;
          if ( dist < 0 )
            delta2 = -delta2;

          if ( delta2 < 32 )
            delta2 = 0;
          else if ( delta2 < 48 )
            delta2 = 32;
          else
            delta2 = 64;

          if ( dist < 0 )
            delta2 = -delta2;

          blue->ref.fit   = FT_PIX_ROUND( blue->ref.cur );
          blue->shoot.fit = blue->ref.fit - delta2;

#endif

          blue->flags |= AF_LATIN_BLUE_ACTIVE;
        }
      }

      /* use sub-top blue zone only if it doesn't overlap with */
      /* another (non-sup-top) blue zone; otherwise, the       */
      /* effect would be similar to a neutral blue zone, which */
      /* is not desired here                                   */
      for ( nn = 0; nn < axis->blue_count; nn++ )
      {
        AF_LatinBlue  blue = &axis->blues[nn];
        FT_UInt       i;


        if ( !( blue->flags & AF_LATIN_BLUE_SUB_TOP ) )
          continue;
        if ( !( blue->flags & AF_LATIN_BLUE_ACTIVE ) )
          continue;

        for ( i = 0; i < axis->blue_count; i++ )
        {
          AF_LatinBlue  b = &axis->blues[i];


          if ( b->flags & AF_LATIN_BLUE_SUB_TOP )
            continue;
          if ( !( b->flags & AF_LATIN_BLUE_ACTIVE ) )
            continue;

          if ( b->ref.fit <= blue->shoot.fit &&
               b->shoot.fit >= blue->ref.fit )
          {
            blue->flags &= ~AF_LATIN_BLUE_ACTIVE;
            break;
          }
        }
      }

#ifdef FT_DEBUG_LEVEL_TRACE
      for ( nn = 0; nn < axis->blue_count; nn++ )
      {
        AF_LatinBlue  blue = &axis->blues[nn];


        FT_TRACE5(( "  reference %d: %d scaled to %.2f%s\n"
                    "  overshoot %d: %d scaled to %.2f%s\n",
                    nn,
                    blue->ref.org,
                    blue->ref.fit / 64.0,
                    blue->flags & AF_LATIN_BLUE_ACTIVE ? ""
                                                       : " (inactive)",
                    nn,
                    blue->shoot.org,
                    blue->shoot.fit / 64.0,
                    blue->flags & AF_LATIN_BLUE_ACTIVE ? ""
                                                       : " (inactive)" ));
      }
#endif
    }
  }


  /* Scale global values in both directions. */

  FT_LOCAL_DEF( void )
  af_latin_metrics_scale( AF_LatinMetrics  metrics,
                          AF_Scaler        scaler )
  {
    metrics->root.scaler.render_mode = scaler->render_mode;
    metrics->root.scaler.face        = scaler->face;
    metrics->root.scaler.flags       = scaler->flags;

    af_latin_metrics_scale_dim( metrics, scaler, AF_DIMENSION_HORZ );
    af_latin_metrics_scale_dim( metrics, scaler, AF_DIMENSION_VERT );
  }


  /* Extract standard_width from writing system/script specific */
  /* metrics class.                                             */

  FT_LOCAL_DEF( void )
  af_latin_get_standard_widths( AF_LatinMetrics  metrics,
                                FT_Pos*          stdHW,
                                FT_Pos*          stdVW )
  {
    if ( stdHW )
      *stdHW = metrics->axis[AF_DIMENSION_VERT].standard_width;

    if ( stdVW )
      *stdVW = metrics->axis[AF_DIMENSION_HORZ].standard_width;
  }


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****           L A T I N   G L Y P H   A N A L Y S I S             *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/


  /* Walk over all contours and compute its segments. */

  FT_LOCAL_DEF( FT_Error )
  af_latin_hints_compute_segments( AF_GlyphHints  hints,
                                   AF_Dimension   dim )
  {
    AF_LatinMetrics  metrics       = (AF_LatinMetrics)hints->metrics;
    AF_AxisHints     axis          = &hints->axis[dim];
    FT_Memory        memory        = hints->memory;
    FT_Error         error         = FT_Err_Ok;
    AF_Segment       segment       = NULL;
    AF_SegmentRec    seg0;
    AF_Point*        contour       = hints->contours;
    AF_Point*        contour_limit = contour + hints->num_contours;
    AF_Direction     major_dir, segment_dir;

    FT_Pos  flat_threshold = FLAT_THRESHOLD( metrics->units_per_em );


    FT_ZERO( &seg0 );
    seg0.score = 32000;
    seg0.flags = AF_EDGE_NORMAL;

    major_dir   = (AF_Direction)FT_ABS( axis->major_dir );
    segment_dir = major_dir;

    axis->num_segments = 0;

    /* set up (u,v) in each point */
    if ( dim == AF_DIMENSION_HORZ )
    {
      AF_Point  point = hints->points;
      AF_Point  limit = point + hints->num_points;


      for ( ; point < limit; point++ )
      {
        point->u = point->fx;
        point->v = point->fy;
      }
    }
    else
    {
      AF_Point  point = hints->points;
      AF_Point  limit = point + hints->num_points;


      for ( ; point < limit; point++ )
      {
        point->u = point->fy;
        point->v = point->fx;
      }
    }

    /* do each contour separately */
    for ( ; contour < contour_limit; contour++ )
    {
      AF_Point  point   = contour[0];
      AF_Point  last    = point->prev;
      int       on_edge = 0;

      /* we call values measured along a segment (point->v)    */
      /* `coordinates', and values orthogonal to it (point->u) */
      /* `positions'                                           */
      FT_Pos     min_pos      =  32000;
      FT_Pos     max_pos      = -32000;
      FT_Pos     min_coord    =  32000;
      FT_Pos     max_coord    = -32000;
      FT_UShort  min_flags    =  AF_FLAG_NONE;
      FT_UShort  max_flags    =  AF_FLAG_NONE;
      FT_Pos     min_on_coord =  32000;
      FT_Pos     max_on_coord = -32000;

      FT_Bool  passed;

      AF_Segment  prev_segment = NULL;

      FT_Pos     prev_min_pos      = min_pos;
      FT_Pos     prev_max_pos      = max_pos;
      FT_Pos     prev_min_coord    = min_coord;
      FT_Pos     prev_max_coord    = max_coord;
      FT_UShort  prev_min_flags    = min_flags;
      FT_UShort  prev_max_flags    = max_flags;
      FT_Pos     prev_min_on_coord = min_on_coord;
      FT_Pos     prev_max_on_coord = max_on_coord;


      if ( FT_ABS( last->out_dir )  == major_dir &&
           FT_ABS( point->out_dir ) == major_dir )
      {
        /* we are already on an edge, try to locate its start */
        last = point;

        for (;;)
        {
          point = point->prev;
          if ( FT_ABS( point->out_dir ) != major_dir )
          {
            point = point->next;
            break;
          }
          if ( point == last )
            break;
        }
      }

      last   = point;
      passed = 0;

      for (;;)
      {
        FT_Pos  u, v;


        if ( on_edge )
        {
          /* get minimum and maximum position */
          u = point->u;
          if ( u < min_pos )
            min_pos = u;
          if ( u > max_pos )
            max_pos = u;

          /* get minimum and maximum coordinate together with flags */
          v = point->v;
          if ( v < min_coord )
          {
            min_coord = v;
            min_flags = point->flags;
          }
          if ( v > max_coord )
          {
            max_coord = v;
            max_flags = point->flags;
          }

          /* get minimum and maximum coordinate of `on' points */
          if ( !( point->flags & AF_FLAG_CONTROL ) )
          {
            v = point->v;
            if ( v < min_on_coord )
              min_on_coord = v;
            if ( v > max_on_coord )
              max_on_coord = v;
          }

          if ( point->out_dir != segment_dir || point == last )
          {
            /* check whether the new segment's start point is identical to */
            /* the previous segment's end point; for example, this might   */
            /* happen for spikes                                           */

            if ( !prev_segment || segment->first != prev_segment->last )
            {
              /* points are different: we are just leaving an edge, thus */
              /* record a new segment                                    */

              segment->last  = point;
              segment->pos   = (FT_Short)( ( min_pos + max_pos ) >> 1 );
              segment->delta = (FT_Short)( ( max_pos - min_pos ) >> 1 );

              /* a segment is round if either its first or last point */
              /* is a control point, and the length of the on points  */
              /* inbetween doesn't exceed a heuristic limit           */
              if ( ( min_flags | max_flags ) & AF_FLAG_CONTROL      &&
                   ( max_on_coord - min_on_coord ) < flat_threshold )
                segment->flags |= AF_EDGE_ROUND;

              segment->min_coord = (FT_Short)min_coord;
              segment->max_coord = (FT_Short)max_coord;
              segment->height    = segment->max_coord - segment->min_coord;

              prev_segment      = segment;
              prev_min_pos      = min_pos;
              prev_max_pos      = max_pos;
              prev_min_coord    = min_coord;
              prev_max_coord    = max_coord;
              prev_min_flags    = min_flags;
              prev_max_flags    = max_flags;
              prev_min_on_coord = min_on_coord;
              prev_max_on_coord = max_on_coord;
            }
            else
            {
              /* points are the same: we don't create a new segment but */
              /* merge the current segment with the previous one        */

              if ( prev_segment->last->in_dir == point->in_dir )
              {
                /* we have identical directions (this can happen for       */
                /* degenerate outlines that move zig-zag along the main    */
                /* axis without changing the coordinate value of the other */
                /* axis, and where the segments have just been merged):    */
                /* unify segments                                          */

                /* update constraints */

                if ( prev_min_pos < min_pos )
                  min_pos = prev_min_pos;
                if ( prev_max_pos > max_pos )
                  max_pos = prev_max_pos;

                if ( prev_min_coord < min_coord )
                {
                  min_coord = prev_min_coord;
                  min_flags = prev_min_flags;
                }
                if ( prev_max_coord > max_coord )
                {
                  max_coord = prev_max_coord;
                  max_flags = prev_max_flags;
                }

                if ( prev_min_on_coord < min_on_coord )
                  min_on_coord = prev_min_on_coord;
                if ( prev_max_on_coord > max_on_coord )
                  max_on_coord = prev_max_on_coord;

                prev_segment->last  = point;
                prev_segment->pos   = (FT_Short)( ( min_pos +
                                                    max_pos ) >> 1 );
                prev_segment->delta = (FT_Short)( ( max_pos -
                                                    min_pos ) >> 1 );

                if ( ( min_flags | max_flags ) & AF_FLAG_CONTROL      &&
                     ( max_on_coord - min_on_coord ) < flat_threshold )
                  prev_segment->flags |= AF_EDGE_ROUND;
                else
                  prev_segment->flags &= ~AF_EDGE_ROUND;

                prev_segment->min_coord = (FT_Short)min_coord;
                prev_segment->max_coord = (FT_Short)max_coord;
                prev_segment->height    = prev_segment->max_coord -
                                          prev_segment->min_coord;
              }
              else
              {
                /* we have different directions; use the properties of the */
                /* longer segment and discard the other one                */

                if ( FT_ABS( prev_max_coord - prev_min_coord ) >
                     FT_ABS( max_coord - min_coord ) )
                {
                  /* discard current segment */

                  if ( min_pos < prev_min_pos )
                    prev_min_pos = min_pos;
                  if ( max_pos > prev_max_pos )
                    prev_max_pos = max_pos;

                  prev_segment->last  = point;
                  prev_segment->pos   = (FT_Short)( ( prev_min_pos +
                                                      prev_max_pos ) >> 1 );
                  prev_segment->delta = (FT_Short)( ( prev_max_pos -
                                                      prev_min_pos ) >> 1 );
                }
                else
                {
                  /* discard previous segment */

                  if ( prev_min_pos < min_pos )
                    min_pos = prev_min_pos;
                  if ( prev_max_pos > max_pos )
                    max_pos = prev_max_pos;

                  segment->last  = point;
                  segment->pos   = (FT_Short)( ( min_pos + max_pos ) >> 1 );
                  segment->delta = (FT_Short)( ( max_pos - min_pos ) >> 1 );

                  if ( ( min_flags | max_flags ) & AF_FLAG_CONTROL      &&
                       ( max_on_coord - min_on_coord ) < flat_threshold )
                    segment->flags |= AF_EDGE_ROUND;

                  segment->min_coord = (FT_Short)min_coord;
                  segment->max_coord = (FT_Short)max_coord;
                  segment->height    = segment->max_coord -
                                       segment->min_coord;

                  *prev_segment = *segment;

                  prev_min_pos      = min_pos;
                  prev_max_pos      = max_pos;
                  prev_min_coord    = min_coord;
                  prev_max_coord    = max_coord;
                  prev_min_flags    = min_flags;
                  prev_max_flags    = max_flags;
                  prev_min_on_coord = min_on_coord;
                  prev_max_on_coord = max_on_coord;
                }
              }

              axis->num_segments--;
            }

            on_edge = 0;
            segment = NULL;

            /* fall through */
          }
        }

        /* now exit if we are at the start/end point */
        if ( point == last )
        {
          if ( passed )
            break;
          passed = 1;
        }

        /* if we are not on an edge, check whether the major direction */
        /* coincides with the current point's `out' direction, or      */
        /* whether we have a single-point contour                      */
        if ( !on_edge                                  &&
             ( FT_ABS( point->out_dir ) == major_dir ||
               point == point->prev                  ) )
        {
          /* this is the start of a new segment! */
          segment_dir = (AF_Direction)point->out_dir;

          error = af_axis_hints_new_segment( axis, memory, &segment );
          if ( error )
            goto Exit;

          /* clear all segment fields */
          segment[0] = seg0;

          segment->dir   = (FT_Char)segment_dir;
          segment->first = point;
          segment->last  = point;

          /* `af_axis_hints_new_segment' reallocates memory,    */
          /* thus we have to refresh the `prev_segment' pointer */
          if ( prev_segment )
            prev_segment = segment - 1;

          min_pos   = max_pos   = point->u;
          min_coord = max_coord = point->v;
          min_flags = max_flags = point->flags;

          if ( point->flags & AF_FLAG_CONTROL )
          {
            min_on_coord =  32000;
            max_on_coord = -32000;
          }
          else
            min_on_coord = max_on_coord = point->v;

          on_edge = 1;

          if ( point == point->prev )
          {
            /* we have a one-point segment: this is a one-point */
            /* contour with `in' and `out' direction set to     */
            /* AF_DIR_NONE                                      */
            segment->pos = (FT_Short)min_pos;

            if (point->flags & AF_FLAG_CONTROL)
              segment->flags |= AF_EDGE_ROUND;

            segment->min_coord = (FT_Short)point->v;
            segment->max_coord = (FT_Short)point->v;
            segment->height = 0;

            on_edge = 0;
            segment = NULL;
          }
        }

        point = point->next;
      }

    } /* contours */


    /* now slightly increase the height of segments if this makes */
    /* sense -- this is used to better detect and ignore serifs   */
    {
      AF_Segment  segments     = axis->segments;
      AF_Segment  segments_end = segments + axis->num_segments;


      for ( segment = segments; segment < segments_end; segment++ )
      {
        AF_Point  first   = segment->first;
        AF_Point  last    = segment->last;
        FT_Pos    first_v = first->v;
        FT_Pos    last_v  = last->v;


        if ( first_v < last_v )
        {
          AF_Point  p;


          p = first->prev;
          if ( p->v < first_v )
            segment->height = (FT_Short)( segment->height +
                                          ( ( first_v - p->v ) >> 1 ) );

          p = last->next;
          if ( p->v > last_v )
            segment->height = (FT_Short)( segment->height +
                                          ( ( p->v - last_v ) >> 1 ) );
        }
        else
        {
          AF_Point  p;


          p = first->prev;
          if ( p->v > first_v )
            segment->height = (FT_Short)( segment->height +
                                          ( ( p->v - first_v ) >> 1 ) );

          p = last->next;
          if ( p->v < last_v )
            segment->height = (FT_Short)( segment->height +
                                          ( ( last_v - p->v ) >> 1 ) );
        }
      }
    }

  Exit:
    return error;
  }


  /* Link segments to form stems and serifs.  If `width_count' and      */
  /* `widths' are non-zero, use them to fine-tune the scoring function. */

  FT_LOCAL_DEF( void )
  af_latin_hints_link_segments( AF_GlyphHints  hints,
                                FT_UInt        width_count,
                                AF_WidthRec*   widths,
                                AF_Dimension   dim )
  {
    AF_AxisHints  axis          = &hints->axis[dim];
    AF_Segment    segments      = axis->segments;
    AF_Segment    segment_limit = segments + axis->num_segments;
    FT_Pos        len_threshold, len_score, dist_score, max_width;
    AF_Segment    seg1, seg2;


    if ( width_count )
      max_width = widths[width_count - 1].org;
    else
      max_width = 0;

    /* a heuristic value to set up a minimum value for overlapping */
    len_threshold = AF_LATIN_CONSTANT( hints->metrics, 8 );
    if ( len_threshold == 0 )
      len_threshold = 1;

    /* a heuristic value to weight lengths */
    len_score = AF_LATIN_CONSTANT( hints->metrics, 6000 );

    /* a heuristic value to weight distances (no call to    */
    /* AF_LATIN_CONSTANT needed, since we work on multiples */
    /* of the stem width)                                   */
    dist_score = 3000;

    /* now compare each segment to the others */
    for ( seg1 = segments; seg1 < segment_limit; seg1++ )
    {
      if ( seg1->dir != axis->major_dir )
        continue;

      /* search for stems having opposite directions, */
      /* with seg1 to the `left' of seg2              */
      for ( seg2 = segments; seg2 < segment_limit; seg2++ )
      {
        FT_Pos  pos1 = seg1->pos;
        FT_Pos  pos2 = seg2->pos;


        if ( seg1->dir + seg2->dir == 0 && pos2 > pos1 )
        {
          /* compute distance between the two segments */
          FT_Pos  min = seg1->min_coord;
          FT_Pos  max = seg1->max_coord;
          FT_Pos  len;


          if ( min < seg2->min_coord )
            min = seg2->min_coord;

          if ( max > seg2->max_coord )
            max = seg2->max_coord;

          /* compute maximum coordinate difference of the two segments */
          /* (this is, how much they overlap)                          */
          len = max - min;
          if ( len >= len_threshold )
          {
            /*
             *  The score is the sum of two demerits indicating the
             *  `badness' of a fit, measured along the segments' main axis
             *  and orthogonal to it, respectively.
             *
             *  o The less overlapping along the main axis, the worse it
             *    is, causing a larger demerit.
             *
             *  o The nearer the orthogonal distance to a stem width, the
             *    better it is, causing a smaller demerit.  For simplicity,
             *    however, we only increase the demerit for values that
             *    exceed the largest stem width.
             */

            FT_Pos  dist = pos2 - pos1;

            FT_Pos  dist_demerit, score;


            if ( max_width )
            {
              /* distance demerits are based on multiples of `max_width'; */
              /* we scale by 1024 for getting more precision              */
              FT_Pos  delta = ( dist << 10 ) / max_width - ( 1 << 10 );


              if ( delta > 10000 )
                dist_demerit = 32000;
              else if ( delta > 0 )
                dist_demerit = delta * delta / dist_score;
              else
                dist_demerit = 0;
            }
            else
              dist_demerit = dist; /* default if no widths available */

            score = dist_demerit + len_score / len;

            /* and we search for the smallest score */
            if ( score < seg1->score )
            {
              seg1->score = score;
              seg1->link  = seg2;
            }

            if ( score < seg2->score )
            {
              seg2->score = score;
              seg2->link  = seg1;
            }
          }
        }
      }
    }

    /* now compute the `serif' segments, cf. explanations in `afhints.h' */
    for ( seg1 = segments; seg1 < segment_limit; seg1++ )
    {
      seg2 = seg1->link;

      if ( seg2 )
      {
        if ( seg2->link != seg1 )
        {
          seg1->link  = 0;
          seg1->serif = seg2->link;
        }
      }
    }
  }


  /* Link segments to edges, using feature analysis for selection. */

  FT_LOCAL_DEF( FT_Error )
  af_latin_hints_compute_edges( AF_GlyphHints  hints,
                                AF_Dimension   dim )
  {
    AF_AxisHints  axis   = &hints->axis[dim];
    FT_Error      error  = FT_Err_Ok;
    FT_Memory     memory = hints->memory;
    AF_LatinAxis  laxis  = &((AF_LatinMetrics)hints->metrics)->axis[dim];

#ifdef FT_CONFIG_OPTION_PIC
    AF_FaceGlobals  globals = hints->metrics->globals;
#endif

    AF_StyleClass   style_class  = hints->metrics->style_class;
    AF_ScriptClass  script_class = AF_SCRIPT_CLASSES_GET
                                     [style_class->script];

    FT_Bool  top_to_bottom_hinting = 0;

    AF_Segment    segments      = axis->segments;
    AF_Segment    segment_limit = segments + axis->num_segments;
    AF_Segment    seg;

#if 0
    AF_Direction  up_dir;
#endif
    FT_Fixed      scale;
    FT_Pos        edge_distance_threshold;
    FT_Pos        segment_length_threshold;
    FT_Pos        segment_width_threshold;


    axis->num_edges = 0;

    scale = ( dim == AF_DIMENSION_HORZ ) ? hints->x_scale
                                         : hints->y_scale;

#if 0
    up_dir = ( dim == AF_DIMENSION_HORZ ) ? AF_DIR_UP
                                          : AF_DIR_RIGHT;
#endif

    if ( dim == AF_DIMENSION_VERT )
      top_to_bottom_hinting = script_class->top_to_bottom_hinting;

    /*
     *  We ignore all segments that are less than 1 pixel in length
     *  to avoid many problems with serif fonts.  We compute the
     *  corresponding threshold in font units.
     */
    if ( dim == AF_DIMENSION_HORZ )
      segment_length_threshold = FT_DivFix( 64, hints->y_scale );
    else
      segment_length_threshold = 0;

    /*
     *  Similarly, we ignore segments that have a width delta
     *  larger than 0.5px (i.e., a width larger than 1px).
     */
    segment_width_threshold = FT_DivFix( 32, scale );

    /*********************************************************************/
    /*                                                                   */
    /* We begin by generating a sorted table of edges for the current    */
    /* direction.  To do so, we simply scan each segment and try to find */
    /* an edge in our table that corresponds to its position.            */
    /*                                                                   */
    /* If no edge is found, we create and insert a new edge in the       */
    /* sorted table.  Otherwise, we simply add the segment to the edge's */
    /* list which gets processed in the second step to compute the       */
    /* edge's properties.                                                */
    /*                                                                   */
    /* Note that the table of edges is sorted along the segment/edge     */
    /* position.                                                         */
    /*                                                                   */
    /*********************************************************************/

    /* assure that edge distance threshold is at most 0.25px */
    edge_distance_threshold = FT_MulFix( laxis->edge_distance_threshold,
                                         scale );
    if ( edge_distance_threshold > 64 / 4 )
      edge_distance_threshold = 64 / 4;

    edge_distance_threshold = FT_DivFix( edge_distance_threshold,
                                         scale );

    for ( seg = segments; seg < segment_limit; seg++ )
    {
      AF_Edge  found = NULL;
      FT_Int   ee;


      /* ignore too short segments, too wide ones, and, in this loop, */
      /* one-point segments without a direction                       */
      if ( seg->height < segment_length_threshold ||
           seg->delta > segment_width_threshold   ||
           seg->dir == AF_DIR_NONE                )
        continue;

      /* A special case for serif edges: If they are smaller than */
      /* 1.5 pixels we ignore them.                               */
      if ( seg->serif                                     &&
           2 * seg->height < 3 * segment_length_threshold )
        continue;

      /* look for an edge corresponding to the segment */
      for ( ee = 0; ee < axis->num_edges; ee++ )
      {
        AF_Edge  edge = axis->edges + ee;
        FT_Pos   dist;


        dist = seg->pos - edge->fpos;
        if ( dist < 0 )
          dist = -dist;

        if ( dist < edge_distance_threshold && edge->dir == seg->dir )
        {
          found = edge;
          break;
        }
      }

      if ( !found )
      {
        AF_Edge  edge;


        /* insert a new edge in the list and */
        /* sort according to the position    */
        error = af_axis_hints_new_edge( axis, seg->pos,
                                        (AF_Direction)seg->dir,
                                        top_to_bottom_hinting,
                                        memory, &edge );
        if ( error )
          goto Exit;

        /* add the segment to the new edge's list */
        FT_ZERO( edge );

        edge->first    = seg;
        edge->last     = seg;
        edge->dir      = seg->dir;
        edge->fpos     = seg->pos;
        edge->opos     = FT_MulFix( seg->pos, scale );
        edge->pos      = edge->opos;
        seg->edge_next = seg;
      }
      else
      {
        /* if an edge was found, simply add the segment to the edge's */
        /* list                                                       */
        seg->edge_next         = found->first;
        found->last->edge_next = seg;
        found->last            = seg;
      }
    }

    /* we loop again over all segments to catch one-point segments   */
    /* without a direction: if possible, link them to existing edges */
    for ( seg = segments; seg < segment_limit; seg++ )
    {
      AF_Edge  found = NULL;
      FT_Int   ee;


      if ( seg->dir != AF_DIR_NONE )
        continue;

      /* look for an edge corresponding to the segment */
      for ( ee = 0; ee < axis->num_edges; ee++ )
      {
        AF_Edge  edge = axis->edges + ee;
        FT_Pos   dist;


        dist = seg->pos - edge->fpos;
        if ( dist < 0 )
          dist = -dist;

        if ( dist < edge_distance_threshold )
        {
          found = edge;
          break;
        }
      }

      /* one-point segments without a match are ignored */
      if ( found )
      {
        seg->edge_next         = found->first;
        found->last->edge_next = seg;
        found->last            = seg;
      }
    }


    /******************************************************************/
    /*                                                                */
    /* Good, we now compute each edge's properties according to the   */
    /* segments found on its position.  Basically, these are          */
    /*                                                                */
    /*  - the edge's main direction                                   */
    /*  - stem edge, serif edge or both (which defaults to stem then) */
    /*  - rounded edge, straight or both (which defaults to straight) */
    /*  - link for edge                                               */
    /*                                                                */
    /******************************************************************/

    /* first of all, set the `edge' field in each segment -- this is */
    /* required in order to compute edge links                       */

    /*
     * Note that removing this loop and setting the `edge' field of each
     * segment directly in the code above slows down execution speed for
     * some reasons on platforms like the Sun.
     */
    {
      AF_Edge  edges      = axis->edges;
      AF_Edge  edge_limit = edges + axis->num_edges;
      AF_Edge  edge;


      for ( edge = edges; edge < edge_limit; edge++ )
      {
        seg = edge->first;
        if ( seg )
          do
          {
            seg->edge = edge;
            seg       = seg->edge_next;

          } while ( seg != edge->first );
      }

      /* now compute each edge properties */
      for ( edge = edges; edge < edge_limit; edge++ )
      {
        FT_Int  is_round    = 0;  /* does it contain round segments?    */
        FT_Int  is_straight = 0;  /* does it contain straight segments? */
#if 0
        FT_Pos  ups         = 0;  /* number of upwards segments         */
        FT_Pos  downs       = 0;  /* number of downwards segments       */
#endif


        seg = edge->first;

        do
        {
          FT_Bool  is_serif;


          /* check for roundness of segment */
          if ( seg->flags & AF_EDGE_ROUND )
            is_round++;
          else
            is_straight++;

#if 0
          /* check for segment direction */
          if ( seg->dir == up_dir )
            ups   += seg->max_coord - seg->min_coord;
          else
            downs += seg->max_coord - seg->min_coord;
#endif

          /* check for links -- if seg->serif is set, then seg->link must */
          /* be ignored                                                   */
          is_serif = (FT_Bool)( seg->serif               &&
                                seg->serif->edge         &&
                                seg->serif->edge != edge );

          if ( ( seg->link && seg->link->edge ) || is_serif )
          {
            AF_Edge     edge2;
            AF_Segment  seg2;


            edge2 = edge->link;
            seg2  = seg->link;

            if ( is_serif )
            {
              seg2  = seg->serif;
              edge2 = edge->serif;
            }

            if ( edge2 )
            {
              FT_Pos  edge_delta;
              FT_Pos  seg_delta;


              edge_delta = edge->fpos - edge2->fpos;
              if ( edge_delta < 0 )
                edge_delta = -edge_delta;

              seg_delta = seg->pos - seg2->pos;
              if ( seg_delta < 0 )
                seg_delta = -seg_delta;

              if ( seg_delta < edge_delta )
                edge2 = seg2->edge;
            }
            else
              edge2 = seg2->edge;

            if ( is_serif )
            {
              edge->serif   = edge2;
              edge2->flags |= AF_EDGE_SERIF;
            }
            else
              edge->link  = edge2;
          }

          seg = seg->edge_next;

        } while ( seg != edge->first );

        /* set the round/straight flags */
        edge->flags = AF_EDGE_NORMAL;

        if ( is_round > 0 && is_round >= is_straight )
          edge->flags |= AF_EDGE_ROUND;

#if 0
        /* set the edge's main direction */
        edge->dir = AF_DIR_NONE;

        if ( ups > downs )
          edge->dir = (FT_Char)up_dir;

        else if ( ups < downs )
          edge->dir = (FT_Char)-up_dir;

        else if ( ups == downs )
          edge->dir = 0;  /* both up and down! */
#endif

        /* get rid of serifs if link is set                 */
        /* XXX: This gets rid of many unpleasant artefacts! */
        /*      Example: the `c' in cour.pfa at size 13     */

        if ( edge->serif && edge->link )
          edge->serif = NULL;
      }
    }

  Exit:
    return error;
  }


  /* Detect segments and edges for given dimension. */

  FT_LOCAL_DEF( FT_Error )
  af_latin_hints_detect_features( AF_GlyphHints  hints,
                                  FT_UInt        width_count,
                                  AF_WidthRec*   widths,
                                  AF_Dimension   dim )
  {
    FT_Error  error;


    error = af_latin_hints_compute_segments( hints, dim );
    if ( !error )
    {
      af_latin_hints_link_segments( hints, width_count, widths, dim );

      error = af_latin_hints_compute_edges( hints, dim );
    }

    return error;
  }


  /* Compute all edges which lie within blue zones. */

  static void
  af_latin_hints_compute_blue_edges( AF_GlyphHints    hints,
                                     AF_LatinMetrics  metrics )
  {
    AF_AxisHints  axis       = &hints->axis[AF_DIMENSION_VERT];
    AF_Edge       edge       = axis->edges;
    AF_Edge       edge_limit = edge + axis->num_edges;
    AF_LatinAxis  latin      = &metrics->axis[AF_DIMENSION_VERT];
    FT_Fixed      scale      = latin->scale;


    /* compute which blue zones are active, i.e. have their scaled */
    /* size < 3/4 pixels                                           */

    /* for each horizontal edge search the blue zone which is closest */
    for ( ; edge < edge_limit; edge++ )
    {
      FT_UInt   bb;
      AF_Width  best_blue            = NULL;
      FT_Bool   best_blue_is_neutral = 0;
      FT_Pos    best_dist;                 /* initial threshold */


      /* compute the initial threshold as a fraction of the EM size */
      /* (the value 40 is heuristic)                                */
      best_dist = FT_MulFix( metrics->units_per_em / 40, scale );

      /* assure a minimum distance of 0.5px */
      if ( best_dist > 64 / 2 )
        best_dist = 64 / 2;

      for ( bb = 0; bb < latin->blue_count; bb++ )
      {
        AF_LatinBlue  blue = latin->blues + bb;
        FT_Bool       is_top_blue, is_neutral_blue, is_major_dir;


        /* skip inactive blue zones (i.e., those that are too large) */
        if ( !( blue->flags & AF_LATIN_BLUE_ACTIVE ) )
          continue;

        /* if it is a top zone, check for right edges (against the major */
        /* direction); if it is a bottom zone, check for left edges (in  */
        /* the major direction) -- this assumes the TrueType convention  */
        /* for the orientation of contours                               */
        is_top_blue =
          (FT_Byte)( ( blue->flags & ( AF_LATIN_BLUE_TOP     |
                                       AF_LATIN_BLUE_SUB_TOP ) ) != 0 );
        is_neutral_blue =
          (FT_Byte)( ( blue->flags & AF_LATIN_BLUE_NEUTRAL ) != 0);
        is_major_dir =
          FT_BOOL( edge->dir == axis->major_dir );

        /* neutral blue zones are handled for both directions */
        if ( is_top_blue ^ is_major_dir || is_neutral_blue )
        {
          FT_Pos  dist;


          /* first of all, compare it to the reference position */
          dist = edge->fpos - blue->ref.org;
          if ( dist < 0 )
            dist = -dist;

          dist = FT_MulFix( dist, scale );
          if ( dist < best_dist )
          {
            best_dist            = dist;
            best_blue            = &blue->ref;
            best_blue_is_neutral = is_neutral_blue;
          }

          /* now compare it to the overshoot position and check whether */
          /* the edge is rounded, and whether the edge is over the      */
          /* reference position of a top zone, or under the reference   */
          /* position of a bottom zone (provided we don't have a        */
          /* neutral blue zone)                                         */
          if ( edge->flags & AF_EDGE_ROUND &&
               dist != 0                   &&
               !is_neutral_blue            )
          {
            FT_Bool  is_under_ref = FT_BOOL( edge->fpos < blue->ref.org );


            if ( is_top_blue ^ is_under_ref )
            {
              dist = edge->fpos - blue->shoot.org;
              if ( dist < 0 )
                dist = -dist;

              dist = FT_MulFix( dist, scale );
              if ( dist < best_dist )
              {
                best_dist            = dist;
                best_blue            = &blue->shoot;
                best_blue_is_neutral = is_neutral_blue;
              }
            }
          }
        }
      }

      if ( best_blue )
      {
        edge->blue_edge = best_blue;
        if ( best_blue_is_neutral )
          edge->flags |= AF_EDGE_NEUTRAL;
      }
    }
  }


  /* Initalize hinting engine. */

  static FT_Error
  af_latin_hints_init( AF_GlyphHints    hints,
                       AF_LatinMetrics  metrics )
  {
    FT_Render_Mode  mode;
    FT_UInt32       scaler_flags, other_flags;
    FT_Face         face = metrics->root.scaler.face;


    af_glyph_hints_rescale( hints, (AF_StyleMetrics)metrics );

    /*
     *  correct x_scale and y_scale if needed, since they may have
     *  been modified by `af_latin_metrics_scale_dim' above
     */
    hints->x_scale = metrics->axis[AF_DIMENSION_HORZ].scale;
    hints->x_delta = metrics->axis[AF_DIMENSION_HORZ].delta;
    hints->y_scale = metrics->axis[AF_DIMENSION_VERT].scale;
    hints->y_delta = metrics->axis[AF_DIMENSION_VERT].delta;

    /* compute flags depending on render mode, etc. */
    mode = metrics->root.scaler.render_mode;

#if 0 /* #ifdef AF_CONFIG_OPTION_USE_WARPER */
    if ( mode == FT_RENDER_MODE_LCD || mode == FT_RENDER_MODE_LCD_V )
      metrics->root.scaler.render_mode = mode = FT_RENDER_MODE_NORMAL;
#endif

    scaler_flags = hints->scaler_flags;
    other_flags  = 0;

    /*
     *  We snap the width of vertical stems for the monochrome and
     *  horizontal LCD rendering targets only.
     */
    if ( mode == FT_RENDER_MODE_MONO || mode == FT_RENDER_MODE_LCD )
      other_flags |= AF_LATIN_HINTS_HORZ_SNAP;

    /*
     *  We snap the width of horizontal stems for the monochrome and
     *  vertical LCD rendering targets only.
     */
    if ( mode == FT_RENDER_MODE_MONO || mode == FT_RENDER_MODE_LCD_V )
      other_flags |= AF_LATIN_HINTS_VERT_SNAP;

    /*
     *  We adjust stems to full pixels unless in `light' or `lcd' mode.
     */
    if ( mode != FT_RENDER_MODE_LIGHT && mode != FT_RENDER_MODE_LCD )
      other_flags |= AF_LATIN_HINTS_STEM_ADJUST;

    if ( mode == FT_RENDER_MODE_MONO )
      other_flags |= AF_LATIN_HINTS_MONO;

    /*
     *  In `light' or `lcd' mode we disable horizontal hinting completely.
     *  We also do it if the face is italic.
     *
     *  However, if warping is enabled (which only works in `light' hinting
     *  mode), advance widths get adjusted, too.
     */
    if ( mode == FT_RENDER_MODE_LIGHT || mode == FT_RENDER_MODE_LCD ||
         ( face->style_flags & FT_STYLE_FLAG_ITALIC ) != 0          )
      scaler_flags |= AF_SCALER_FLAG_NO_HORIZONTAL;

#ifdef AF_CONFIG_OPTION_USE_WARPER
    /* get (global) warper flag */
    if ( !metrics->root.globals->module->warping )
      scaler_flags |= AF_SCALER_FLAG_NO_WARPER;
#endif

    hints->scaler_flags = scaler_flags;
    hints->other_flags  = other_flags;

    return FT_Err_Ok;
  }


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****        L A T I N   G L Y P H   G R I D - F I T T I N G        *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  /* Snap a given width in scaled coordinates to one of the */
  /* current standard widths.                               */

  static FT_Pos
  af_latin_snap_width( AF_Width  widths,
                       FT_UInt   count,
                       FT_Pos    width )
  {
    FT_UInt  n;
    FT_Pos   best      = 64 + 32 + 2;
    FT_Pos   reference = width;
    FT_Pos   scaled;


    for ( n = 0; n < count; n++ )
    {
      FT_Pos  w;
      FT_Pos  dist;


      w = widths[n].cur;
      dist = width - w;
      if ( dist < 0 )
        dist = -dist;
      if ( dist < best )
      {
        best      = dist;
        reference = w;
      }
    }

    scaled = FT_PIX_ROUND( reference );

    if ( width >= reference )
    {
      if ( width < scaled + 48 )
        width = reference;
    }
    else
    {
      if ( width > scaled - 48 )
        width = reference;
    }

    return width;
  }


  /* Compute the snapped width of a given stem, ignoring very thin ones. */
  /* There is a lot of voodoo in this function; changing the hard-coded  */
  /* parameters influence the whole hinting process.                     */

  static FT_Pos
  af_latin_compute_stem_width( AF_GlyphHints  hints,
                               AF_Dimension   dim,
                               FT_Pos         width,
                               FT_Pos         base_delta,
                               FT_UInt        base_flags,
                               FT_UInt        stem_flags )
  {
    AF_LatinMetrics  metrics  = (AF_LatinMetrics)hints->metrics;
    AF_LatinAxis     axis     = &metrics->axis[dim];
    FT_Pos           dist     = width;
    FT_Int           sign     = 0;
    FT_Int           vertical = ( dim == AF_DIMENSION_VERT );


    if ( !AF_LATIN_HINTS_DO_STEM_ADJUST( hints ) ||
         axis->extra_light                       )
      return width;

    if ( dist < 0 )
    {
      dist = -width;
      sign = 1;
    }

    if ( (  vertical && !AF_LATIN_HINTS_DO_VERT_SNAP( hints ) ) ||
         ( !vertical && !AF_LATIN_HINTS_DO_HORZ_SNAP( hints ) ) )
    {
      /* smooth hinting process: very lightly quantize the stem width */

      /* leave the widths of serifs alone */
      if ( ( stem_flags & AF_EDGE_SERIF ) &&
           vertical                       &&
           ( dist < 3 * 64 )              )
        goto Done_Width;

      else if ( base_flags & AF_EDGE_ROUND )
      {
        if ( dist < 80 )
          dist = 64;
      }
      else if ( dist < 56 )
        dist = 56;

      if ( axis->width_count > 0 )
      {
        FT_Pos  delta;


        /* compare to standard width */
        delta = dist - axis->widths[0].cur;

        if ( delta < 0 )
          delta = -delta;

        if ( delta < 40 )
        {
          dist = axis->widths[0].cur;
          if ( dist < 48 )
            dist = 48;

          goto Done_Width;
        }

        if ( dist < 3 * 64 )
        {
          delta  = dist & 63;
          dist  &= -64;

          if ( delta < 10 )
            dist += delta;

          else if ( delta < 32 )
            dist += 10;

          else if ( delta < 54 )
            dist += 54;

          else
            dist += delta;
        }
        else
        {
          /* A stem's end position depends on two values: the start        */
          /* position and the stem length.  The former gets usually        */
          /* rounded to the grid, while the latter gets rounded also if it */
          /* exceeds a certain length (see below in this function).  This  */
          /* `double rounding' can lead to a great difference to the       */
          /* original, unhinted position; this normally doesn't matter for */
          /* large PPEM values, but for small sizes it can easily make     */
          /* outlines collide.  For this reason, we adjust the stem length */
          /* by a small amount depending on the PPEM value in case the     */
          /* former and latter rounding both point into the same           */
          /* direction.                                                    */

          FT_Pos  bdelta = 0;


          if ( ( ( width > 0 ) && ( base_delta > 0 ) ) ||
               ( ( width < 0 ) && ( base_delta < 0 ) ) )
          {
            FT_UInt  ppem = metrics->root.scaler.face->size->metrics.x_ppem;


            if ( ppem < 10 )
              bdelta = base_delta;
            else if ( ppem < 30 )
              bdelta = ( base_delta * (FT_Pos)( 30 - ppem ) ) / 20;

            if ( bdelta < 0 )
              bdelta = -bdelta;
          }

          dist = ( dist - bdelta + 32 ) & ~63;
        }
      }
    }
    else
    {
      /* strong hinting process: snap the stem width to integer pixels */

      FT_Pos  org_dist = dist;


      dist = af_latin_snap_width( axis->widths, axis->width_count, dist );

      if ( vertical )
      {
        /* in the case of vertical hinting, always round */
        /* the stem heights to integer pixels            */

        if ( dist >= 64 )
          dist = ( dist + 16 ) & ~63;
        else
          dist = 64;
      }
      else
      {
        if ( AF_LATIN_HINTS_DO_MONO( hints ) )
        {
          /* monochrome horizontal hinting: snap widths to integer pixels */
          /* with a different threshold                                   */

          if ( dist < 64 )
            dist = 64;
          else
            dist = ( dist + 32 ) & ~63;
        }
        else
        {
          /* for horizontal anti-aliased hinting, we adopt a more subtle */
          /* approach: we strengthen small stems, round stems whose size */
          /* is between 1 and 2 pixels to an integer, otherwise nothing  */

          if ( dist < 48 )
            dist = ( dist + 64 ) >> 1;

          else if ( dist < 128 )
          {
            /* We only round to an integer width if the corresponding */
            /* distortion is less than 1/4 pixel.  Otherwise this     */
            /* makes everything worse since the diagonals, which are  */
            /* not hinted, appear a lot bolder or thinner than the    */
            /* vertical stems.                                        */

            FT_Pos  delta;


            dist = ( dist + 22 ) & ~63;
            delta = dist - org_dist;
            if ( delta < 0 )
              delta = -delta;

            if ( delta >= 16 )
            {
              dist = org_dist;
              if ( dist < 48 )
                dist = ( dist + 64 ) >> 1;
            }
          }
          else
            /* round otherwise to prevent color fringes in LCD mode */
            dist = ( dist + 32 ) & ~63;
        }
      }
    }

  Done_Width:
    if ( sign )
      dist = -dist;

    return dist;
  }


  /* Align one stem edge relative to the previous stem edge. */

  static void
  af_latin_align_linked_edge( AF_GlyphHints  hints,
                              AF_Dimension   dim,
                              AF_Edge        base_edge,
                              AF_Edge        stem_edge )
  {
    FT_Pos  dist, base_delta;
    FT_Pos  fitted_width;


    dist       = stem_edge->opos - base_edge->opos;
    base_delta = base_edge->pos - base_edge->opos;

    fitted_width = af_latin_compute_stem_width( hints, dim,
                                                dist, base_delta,
                                                base_edge->flags,
                                                stem_edge->flags );


    stem_edge->pos = base_edge->pos + fitted_width;

    FT_TRACE5(( "  LINK: edge %d (opos=%.2f) linked to %.2f,"
                " dist was %.2f, now %.2f\n",
                stem_edge - hints->axis[dim].edges, stem_edge->opos / 64.0,
                stem_edge->pos / 64.0, dist / 64.0, fitted_width / 64.0 ));
  }


  /* Shift the coordinates of the `serif' edge by the same amount */
  /* as the corresponding `base' edge has been moved already.     */

  static void
  af_latin_align_serif_edge( AF_GlyphHints  hints,
                             AF_Edge        base,
                             AF_Edge        serif )
  {
    FT_UNUSED( hints );

    serif->pos = base->pos + ( serif->opos - base->opos );
  }


  /*************************************************************************/
  /*************************************************************************/
  /*************************************************************************/
  /****                                                                 ****/
  /****                    E D G E   H I N T I N G                      ****/
  /****                                                                 ****/
  /*************************************************************************/
  /*************************************************************************/
  /*************************************************************************/


  /* The main grid-fitting routine. */

  static void
  af_latin_hint_edges( AF_GlyphHints  hints,
                       AF_Dimension   dim )
  {
    AF_AxisHints  axis       = &hints->axis[dim];
    AF_Edge       edges      = axis->edges;
    AF_Edge       edge_limit = edges + axis->num_edges;
    FT_PtrDist    n_edges;
    AF_Edge       edge;
    AF_Edge       anchor     = NULL;
    FT_Int        has_serifs = 0;

#ifdef FT_CONFIG_OPTION_PIC
    AF_FaceGlobals  globals = hints->metrics->globals;
#endif

    AF_StyleClass   style_class  = hints->metrics->style_class;
    AF_ScriptClass  script_class = AF_SCRIPT_CLASSES_GET
                                     [style_class->script];

    FT_Bool  top_to_bottom_hinting = 0;

#ifdef FT_DEBUG_LEVEL_TRACE
    FT_UInt  num_actions = 0;
#endif


    FT_TRACE5(( "latin %s edge hinting (style `%s')\n",
                dim == AF_DIMENSION_VERT ? "horizontal" : "vertical",
                af_style_names[hints->metrics->style_class->style] ));

    if ( dim == AF_DIMENSION_VERT )
      top_to_bottom_hinting = script_class->top_to_bottom_hinting;

    /* we begin by aligning all stems relative to the blue zone */
    /* if needed -- that's only for horizontal edges            */

    if ( dim == AF_DIMENSION_VERT && AF_HINTS_DO_BLUES( hints ) )
    {
      for ( edge = edges; edge < edge_limit; edge++ )
      {
        AF_Width  blue;
        AF_Edge   edge1, edge2; /* these edges form the stem to check */


        if ( edge->flags & AF_EDGE_DONE )
          continue;

        edge1 = NULL;
        edge2 = edge->link;

        /*
         *  If a stem contains both a neutral and a non-neutral blue zone,
         *  skip the neutral one.  Otherwise, outlines with different
         *  directions might be incorrectly aligned at the same vertical
         *  position.
         *
         *  If we have two neutral blue zones, skip one of them.
         *
         */
        if ( edge->blue_edge && edge2 && edge2->blue_edge )
        {
          FT_Byte  neutral  = edge->flags  & AF_EDGE_NEUTRAL;
          FT_Byte  neutral2 = edge2->flags & AF_EDGE_NEUTRAL;


          if ( neutral2 )
          {
            edge2->blue_edge = NULL;
            edge2->flags    &= ~AF_EDGE_NEUTRAL;
          }
          else if ( neutral )
          {
            edge->blue_edge = NULL;
            edge->flags    &= ~AF_EDGE_NEUTRAL;
          }
        }

        blue = edge->blue_edge;
        if ( blue )
          edge1 = edge;

        /* flip edges if the other edge is aligned to a blue zone */
        else if ( edge2 && edge2->blue_edge )
        {
          blue  = edge2->blue_edge;
          edge1 = edge2;
          edge2 = edge;
        }

        if ( !edge1 )
          continue;

#ifdef FT_DEBUG_LEVEL_TRACE
        if ( !anchor )
          FT_TRACE5(( "  BLUE_ANCHOR: edge %d (opos=%.2f) snapped to %.2f,"
                      " was %.2f (anchor=edge %d)\n",
                      edge1 - edges, edge1->opos / 64.0, blue->fit / 64.0,
                      edge1->pos / 64.0, edge - edges ));
        else
          FT_TRACE5(( "  BLUE: edge %d (opos=%.2f) snapped to %.2f,"
                      " was %.2f\n",
                      edge1 - edges, edge1->opos / 64.0, blue->fit / 64.0,
                      edge1->pos / 64.0 ));

        num_actions++;
#endif

        edge1->pos    = blue->fit;
        edge1->flags |= AF_EDGE_DONE;

        if ( edge2 && !edge2->blue_edge )
        {
          af_latin_align_linked_edge( hints, dim, edge1, edge2 );
          edge2->flags |= AF_EDGE_DONE;

#ifdef FT_DEBUG_LEVEL_TRACE
          num_actions++;
#endif
        }

        if ( !anchor )
          anchor = edge;
      }
    }

    /* now we align all other stem edges, trying to maintain the */
    /* relative order of stems in the glyph                      */
    for ( edge = edges; edge < edge_limit; edge++ )
    {
      AF_Edge  edge2;


      if ( edge->flags & AF_EDGE_DONE )
        continue;

      /* skip all non-stem edges */
      edge2 = edge->link;
      if ( !edge2 )
      {
        has_serifs++;
        continue;
      }

      /* now align the stem */

      /* this should not happen, but it's better to be safe */
      if ( edge2->blue_edge )
      {
        FT_TRACE5(( "  ASSERTION FAILED for edge %d\n", edge2 - edges ));

        af_latin_align_linked_edge( hints, dim, edge2, edge );
        edge->flags |= AF_EDGE_DONE;

#ifdef FT_DEBUG_LEVEL_TRACE
        num_actions++;
#endif
        continue;
      }

      if ( !anchor )
      {
        /* if we reach this if clause, no stem has been aligned yet */

        FT_Pos  org_len, org_center, cur_len;
        FT_Pos  cur_pos1, error1, error2, u_off, d_off;


        org_len = edge2->opos - edge->opos;
        cur_len = af_latin_compute_stem_width( hints, dim,
                                               org_len, 0,
                                               edge->flags,
                                               edge2->flags );

        /* some voodoo to specially round edges for small stem widths; */
        /* the idea is to align the center of a stem, then shifting    */
        /* the stem edges to suitable positions                        */
        if ( cur_len <= 64 )
        {
          /* width <= 1px */
          u_off = 32;
          d_off = 32;
        }
        else
        {
          /* 1px < width < 1.5px */
          u_off = 38;
          d_off = 26;
        }

        if ( cur_len < 96 )
        {
          org_center = edge->opos + ( org_len >> 1 );
          cur_pos1   = FT_PIX_ROUND( org_center );

          error1 = org_center - ( cur_pos1 - u_off );
          if ( error1 < 0 )
            error1 = -error1;

          error2 = org_center - ( cur_pos1 + d_off );
          if ( error2 < 0 )
            error2 = -error2;

          if ( error1 < error2 )
            cur_pos1 -= u_off;
          else
            cur_pos1 += d_off;

          edge->pos  = cur_pos1 - cur_len / 2;
          edge2->pos = edge->pos + cur_len;
        }
        else
          edge->pos = FT_PIX_ROUND( edge->opos );

        anchor       = edge;
        edge->flags |= AF_EDGE_DONE;

        FT_TRACE5(( "  ANCHOR: edge %d (opos=%.2f) and %d (opos=%.2f)"
                    " snapped to %.2f and %.2f\n",
                    edge - edges, edge->opos / 64.0,
                    edge2 - edges, edge2->opos / 64.0,
                    edge->pos / 64.0, edge2->pos / 64.0 ));

        af_latin_align_linked_edge( hints, dim, edge, edge2 );

#ifdef FT_DEBUG_LEVEL_TRACE
        num_actions += 2;
#endif
      }
      else
      {
        FT_Pos  org_pos, org_len, org_center, cur_len;
        FT_Pos  cur_pos1, cur_pos2, delta1, delta2;


        org_pos    = anchor->pos + ( edge->opos - anchor->opos );
        org_len    = edge2->opos - edge->opos;
        org_center = org_pos + ( org_len >> 1 );

        cur_len = af_latin_compute_stem_width( hints, dim,
                                               org_len, 0,
                                               edge->flags,
                                               edge2->flags );

        if ( edge2->flags & AF_EDGE_DONE )
        {
          FT_TRACE5(( "  ADJUST: edge %d (pos=%.2f) moved to %.2f\n",
                      edge - edges, edge->pos / 64.0,
                      ( edge2->pos - cur_len ) / 64.0 ));

          edge->pos = edge2->pos - cur_len;
        }

        else if ( cur_len < 96 )
        {
          FT_Pos  u_off, d_off;


          cur_pos1 = FT_PIX_ROUND( org_center );

          if ( cur_len <= 64 )
          {
            u_off = 32;
            d_off = 32;
          }
          else
          {
            u_off = 38;
            d_off = 26;
          }

          delta1 = org_center - ( cur_pos1 - u_off );
          if ( delta1 < 0 )
            delta1 = -delta1;

          delta2 = org_center - ( cur_pos1 + d_off );
          if ( delta2 < 0 )
            delta2 = -delta2;

          if ( delta1 < delta2 )
            cur_pos1 -= u_off;
          else
            cur_pos1 += d_off;

          edge->pos  = cur_pos1 - cur_len / 2;
          edge2->pos = cur_pos1 + cur_len / 2;

          FT_TRACE5(( "  STEM: edge %d (opos=%.2f) linked to %d (opos=%.2f)"
                      " snapped to %.2f and %.2f\n",
                      edge - edges, edge->opos / 64.0,
                      edge2 - edges, edge2->opos / 64.0,
                      edge->pos / 64.0, edge2->pos / 64.0 ));
        }

        else
        {
          org_pos    = anchor->pos + ( edge->opos - anchor->opos );
          org_len    = edge2->opos - edge->opos;
          org_center = org_pos + ( org_len >> 1 );

          cur_len    = af_latin_compute_stem_width( hints, dim,
                                                    org_len, 0,
                                                    edge->flags,
                                                    edge2->flags );

          cur_pos1 = FT_PIX_ROUND( org_pos );
          delta1   = cur_pos1 + ( cur_len >> 1 ) - org_center;
          if ( delta1 < 0 )
            delta1 = -delta1;

          cur_pos2 = FT_PIX_ROUND( org_pos + org_len ) - cur_len;
          delta2   = cur_pos2 + ( cur_len >> 1 ) - org_center;
          if ( delta2 < 0 )
            delta2 = -delta2;

          edge->pos  = ( delta1 < delta2 ) ? cur_pos1 : cur_pos2;
          edge2->pos = edge->pos + cur_len;

          FT_TRACE5(( "  STEM: edge %d (opos=%.2f) linked to %d (opos=%.2f)"
                      " snapped to %.2f and %.2f\n",
                      edge - edges, edge->opos / 64.0,
                      edge2 - edges, edge2->opos / 64.0,
                      edge->pos / 64.0, edge2->pos / 64.0 ));
        }

#ifdef FT_DEBUG_LEVEL_TRACE
        num_actions++;
#endif

        edge->flags  |= AF_EDGE_DONE;
        edge2->flags |= AF_EDGE_DONE;

        if ( edge > edges                                             &&
             ( top_to_bottom_hinting ? ( edge->pos > edge[-1].pos )
                                     : ( edge->pos < edge[-1].pos ) ) )
        {
          /* don't move if stem would (almost) disappear otherwise; */
          /* the ad-hoc value 16 corresponds to 1/4px               */
          if ( edge->link && FT_ABS( edge->link->pos - edge[-1].pos ) > 16 )
          {
#ifdef FT_DEBUG_LEVEL_TRACE
            FT_TRACE5(( "  BOUND: edge %d (pos=%.2f) moved to %.2f\n",
                        edge - edges,
                        edge->pos / 64.0,
                        edge[-1].pos / 64.0 ));

            num_actions++;
#endif

            edge->pos = edge[-1].pos;
          }
        }
      }
    }

    /* make sure that lowercase m's maintain their symmetry */

    /* In general, lowercase m's have six vertical edges if they are sans */
    /* serif, or twelve if they are with serifs.  This implementation is  */
    /* based on that assumption, and seems to work very well with most    */
    /* faces.  However, if for a certain face this assumption is not      */
    /* true, the m is just rendered like before.  In addition, any stem   */
    /* correction will only be applied to symmetrical glyphs (even if the */
    /* glyph is not an m), so the potential for unwanted distortion is    */
    /* relatively low.                                                    */

    /* We don't handle horizontal edges since we can't easily assure that */
    /* the third (lowest) stem aligns with the base line; it might end up */
    /* one pixel higher or lower.                                         */

    n_edges = edge_limit - edges;
    if ( dim == AF_DIMENSION_HORZ && ( n_edges == 6 || n_edges == 12 ) )
    {
      AF_Edge  edge1, edge2, edge3;
      FT_Pos   dist1, dist2, span, delta;


      if ( n_edges == 6 )
      {
        edge1 = edges;
        edge2 = edges + 2;
        edge3 = edges + 4;
      }
      else
      {
        edge1 = edges + 1;
        edge2 = edges + 5;
        edge3 = edges + 9;
      }

      dist1 = edge2->opos - edge1->opos;
      dist2 = edge3->opos - edge2->opos;

      span = dist1 - dist2;
      if ( span < 0 )
        span = -span;

      if ( span < 8 )
      {
        delta = edge3->pos - ( 2 * edge2->pos - edge1->pos );
        edge3->pos -= delta;
        if ( edge3->link )
          edge3->link->pos -= delta;

        /* move the serifs along with the stem */
        if ( n_edges == 12 )
        {
          ( edges + 8 )->pos -= delta;
          ( edges + 11 )->pos -= delta;
        }

        edge3->flags |= AF_EDGE_DONE;
        if ( edge3->link )
          edge3->link->flags |= AF_EDGE_DONE;
      }
    }

    if ( has_serifs || !anchor )
    {
      /*
       *  now hint the remaining edges (serifs and single) in order
       *  to complete our processing
       */
      for ( edge = edges; edge < edge_limit; edge++ )
      {
        FT_Pos  delta;


        if ( edge->flags & AF_EDGE_DONE )
          continue;

        delta = 1000;

        if ( edge->serif )
        {
          delta = edge->serif->opos - edge->opos;
          if ( delta < 0 )
            delta = -delta;
        }

        if ( delta < 64 + 16 )
        {
          af_latin_align_serif_edge( hints, edge->serif, edge );
          FT_TRACE5(( "  SERIF: edge %d (opos=%.2f) serif to %d (opos=%.2f)"
                      " aligned to %.2f\n",
                      edge - edges, edge->opos / 64.0,
                      edge->serif - edges, edge->serif->opos / 64.0,
                      edge->pos / 64.0 ));
        }
        else if ( !anchor )
        {
          edge->pos = FT_PIX_ROUND( edge->opos );
          anchor    = edge;
          FT_TRACE5(( "  SERIF_ANCHOR: edge %d (opos=%.2f)"
                      " snapped to %.2f\n",
                      edge-edges, edge->opos / 64.0, edge->pos / 64.0 ));
        }
        else
        {
          AF_Edge  before, after;


          for ( before = edge - 1; before >= edges; before-- )
            if ( before->flags & AF_EDGE_DONE )
              break;

          for ( after = edge + 1; after < edge_limit; after++ )
            if ( after->flags & AF_EDGE_DONE )
              break;

          if ( before >= edges && before < edge   &&
               after < edge_limit && after > edge )
          {
            if ( after->opos == before->opos )
              edge->pos = before->pos;
            else
              edge->pos = before->pos +
                          FT_MulDiv( edge->opos - before->opos,
                                     after->pos - before->pos,
                                     after->opos - before->opos );

            FT_TRACE5(( "  SERIF_LINK1: edge %d (opos=%.2f) snapped to %.2f"
                        " from %d (opos=%.2f)\n",
                        edge - edges, edge->opos / 64.0,
                        edge->pos / 64.0,
                        before - edges, before->opos / 64.0 ));
          }
          else
          {
            edge->pos = anchor->pos +
                        ( ( edge->opos - anchor->opos + 16 ) & ~31 );
            FT_TRACE5(( "  SERIF_LINK2: edge %d (opos=%.2f)"
                        " snapped to %.2f\n",
                        edge - edges, edge->opos / 64.0, edge->pos / 64.0 ));
          }
        }

#ifdef FT_DEBUG_LEVEL_TRACE
        num_actions++;
#endif
        edge->flags |= AF_EDGE_DONE;

        if ( edge > edges                                             &&
             ( top_to_bottom_hinting ? ( edge->pos > edge[-1].pos )
                                     : ( edge->pos < edge[-1].pos ) ) )
        {
          /* don't move if stem would (almost) disappear otherwise; */
          /* the ad-hoc value 16 corresponds to 1/4px               */
          if ( edge->link && FT_ABS( edge->link->pos - edge[-1].pos ) > 16 )
          {
#ifdef FT_DEBUG_LEVEL_TRACE
            FT_TRACE5(( "  BOUND: edge %d (pos=%.2f) moved to %.2f\n",
                        edge - edges,
                        edge->pos / 64.0,
                        edge[-1].pos / 64.0 ));

            num_actions++;
#endif
            edge->pos = edge[-1].pos;
          }
        }

        if ( edge + 1 < edge_limit                                   &&
             edge[1].flags & AF_EDGE_DONE                            &&
             ( top_to_bottom_hinting ? ( edge->pos < edge[1].pos )
                                     : ( edge->pos > edge[1].pos ) ) )
        {
          /* don't move if stem would (almost) disappear otherwise; */
          /* the ad-hoc value 16 corresponds to 1/4px               */
          if ( edge->link && FT_ABS( edge->link->pos - edge[-1].pos ) > 16 )
          {
#ifdef FT_DEBUG_LEVEL_TRACE
            FT_TRACE5(( "  BOUND: edge %d (pos=%.2f) moved to %.2f\n",
                        edge - edges,
                        edge->pos / 64.0,
                        edge[1].pos / 64.0 ));

            num_actions++;
#endif

            edge->pos = edge[1].pos;
          }
        }
      }
    }

#ifdef FT_DEBUG_LEVEL_TRACE
    if ( !num_actions )
      FT_TRACE5(( "  (none)\n" ));
    FT_TRACE5(( "\n" ));
#endif
  }


  /* Apply the complete hinting algorithm to a latin glyph. */

  static FT_Error
  af_latin_hints_apply( FT_UInt          glyph_index,
                        AF_GlyphHints    hints,
                        FT_Outline*      outline,
                        AF_LatinMetrics  metrics )
  {
    FT_Error  error;
    int       dim;

    AF_LatinAxis  axis;


    error = af_glyph_hints_reload( hints, outline );
    if ( error )
      goto Exit;

    /* analyze glyph outline */
    if ( AF_HINTS_DO_HORIZONTAL( hints ) )
    {
      axis  = &metrics->axis[AF_DIMENSION_HORZ];
      error = af_latin_hints_detect_features( hints,
                                              axis->width_count,
                                              axis->widths,
                                              AF_DIMENSION_HORZ );
      if ( error )
        goto Exit;
    }

    if ( AF_HINTS_DO_VERTICAL( hints ) )
    {
      axis  = &metrics->axis[AF_DIMENSION_VERT];
      error = af_latin_hints_detect_features( hints,
                                              axis->width_count,
                                              axis->widths,
                                              AF_DIMENSION_VERT );
      if ( error )
        goto Exit;

      /* apply blue zones to base characters only */
      if ( !( metrics->root.globals->glyph_styles[glyph_index] & AF_NONBASE ) )
        af_latin_hints_compute_blue_edges( hints, metrics );
    }

    /* grid-fit the outline */
    for ( dim = 0; dim < AF_DIMENSION_MAX; dim++ )
    {
#ifdef AF_CONFIG_OPTION_USE_WARPER
      if ( dim == AF_DIMENSION_HORZ                                  &&
           metrics->root.scaler.render_mode == FT_RENDER_MODE_NORMAL &&
           AF_HINTS_DO_WARP( hints )                                 )
      {
        AF_WarperRec  warper;
        FT_Fixed      scale;
        FT_Pos        delta;


        af_warper_compute( &warper, hints, (AF_Dimension)dim,
                           &scale, &delta );
        af_glyph_hints_scale_dim( hints, (AF_Dimension)dim,
                                  scale, delta );
        continue;
      }
#endif /* AF_CONFIG_OPTION_USE_WARPER */

      if ( ( dim == AF_DIMENSION_HORZ && AF_HINTS_DO_HORIZONTAL( hints ) ) ||
           ( dim == AF_DIMENSION_VERT && AF_HINTS_DO_VERTICAL( hints ) )   )
      {
        af_latin_hint_edges( hints, (AF_Dimension)dim );
        af_glyph_hints_align_edge_points( hints, (AF_Dimension)dim );
        af_glyph_hints_align_strong_points( hints, (AF_Dimension)dim );
        af_glyph_hints_align_weak_points( hints, (AF_Dimension)dim );
      }
    }

    af_glyph_hints_save( hints, outline );

  Exit:
    return error;
  }


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****              L A T I N   S C R I P T   C L A S S              *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/


  AF_DEFINE_WRITING_SYSTEM_CLASS(
    af_latin_writing_system_class,

    AF_WRITING_SYSTEM_LATIN,

    sizeof ( AF_LatinMetricsRec ),

    (AF_WritingSystem_InitMetricsFunc) af_latin_metrics_init,        /* style_metrics_init    */
    (AF_WritingSystem_ScaleMetricsFunc)af_latin_metrics_scale,       /* style_metrics_scale   */
    (AF_WritingSystem_DoneMetricsFunc) NULL,                         /* style_metrics_done    */
    (AF_WritingSystem_GetStdWidthsFunc)af_latin_get_standard_widths, /* style_metrics_getstdw */

    (AF_WritingSystem_InitHintsFunc)   af_latin_hints_init,          /* style_hints_init      */
    (AF_WritingSystem_ApplyHintsFunc)  af_latin_hints_apply          /* style_hints_apply     */
  )


/* END */
