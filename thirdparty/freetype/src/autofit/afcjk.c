/****************************************************************************
 *
 * afcjk.c
 *
 *   Auto-fitter hinting routines for CJK writing system (body).
 *
 * Copyright (C) 2006-2019 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */

  /*
   * The algorithm is based on akito's autohint patch, archived at
   *
   * https://web.archive.org/web/20051219160454/http://www.kde.gr.jp:80/~akito/patch/freetype2/2.1.7/
   *
   */

#include <ft2build.h>
#include FT_ADVANCES_H
#include FT_INTERNAL_DEBUG_H

#include "afglobal.h"
#include "aflatin.h"
#include "afcjk.h"


#ifdef AF_CONFIG_OPTION_CJK

#undef AF_CONFIG_OPTION_CJK_BLUE_HANI_VERT

#include "aferrors.h"


#ifdef AF_CONFIG_OPTION_USE_WARPER
#include "afwarp.h"
#endif


  /**************************************************************************
   *
   * The macro FT_COMPONENT is used in trace mode.  It is an implicit
   * parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log
   * messages during execution.
   */
#undef  FT_COMPONENT
#define FT_COMPONENT  afcjk


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****              C J K   G L O B A L   M E T R I C S              *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/


  /* Basically the Latin version with AF_CJKMetrics */
  /* to replace AF_LatinMetrics.                    */

  FT_LOCAL_DEF( void )
  af_cjk_metrics_init_widths( AF_CJKMetrics  metrics,
                              FT_Face        face )
  {
    /* scan the array of segments in each direction */
    AF_GlyphHintsRec  hints[1];


    FT_TRACE5(( "\n"
                "cjk standard widths computation (style `%s')\n"
                "===================================================\n"
                "\n",
                af_style_names[metrics->root.style_class->style] ));

    af_glyph_hints_init( hints, face->memory );

    metrics->axis[AF_DIMENSION_HORZ].width_count = 0;
    metrics->axis[AF_DIMENSION_VERT].width_count = 0;

    {
      FT_Error          error;
      FT_ULong          glyph_index;
      int               dim;
      AF_CJKMetricsRec  dummy[1];
      AF_Scaler         scaler = &dummy->root.scaler;

      AF_StyleClass   style_class  = metrics->root.style_class;
      AF_ScriptClass  script_class = af_script_classes[style_class->script];

      /* If HarfBuzz is not available, we need a pointer to a single */
      /* unsigned long value.                                        */
#ifdef FT_CONFIG_OPTION_USE_HARFBUZZ
      void*     shaper_buf;
#else
      FT_ULong  shaper_buf_;
      void*     shaper_buf = &shaper_buf_;
#endif

      const char*  p;

#ifdef FT_DEBUG_LEVEL_TRACE
      FT_ULong  ch = 0;
#endif

      p = script_class->standard_charstring;

#ifdef FT_CONFIG_OPTION_USE_HARFBUZZ
      shaper_buf = af_shaper_buf_create( face );
#endif

      /* We check a list of standard characters.  The first match wins. */

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
        AF_CJKAxis    axis    = &metrics->axis[dim];
        AF_AxisHints  axhints = &hints->axis[dim];
        AF_Segment    seg, limit, link;
        FT_UInt       num_widths = 0;


        error = af_latin_hints_compute_segments( hints,
                                                 (AF_Dimension)dim );
        if ( error )
          goto Exit;

        /*
         * We assume that the glyphs selected for the stem width
         * computation are `featureless' enough so that the linking
         * algorithm works fine without adjustments of its scoring
         * function.
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

            if ( num_widths < AF_CJK_MAX_WIDTHS )
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
        AF_CJKAxis  axis = &metrics->axis[dim];
        FT_Pos      stdw;


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


  /* Find all blue zones. */

  static void
  af_cjk_metrics_init_blues( AF_CJKMetrics  metrics,
                             FT_Face        face )
  {
    FT_Pos      fills[AF_BLUE_STRING_MAX_LEN];
    FT_Pos      flats[AF_BLUE_STRING_MAX_LEN];

    FT_UInt     num_fills;
    FT_UInt     num_flats;

    FT_Bool     fill;

    AF_CJKBlue  blue;
    FT_Error    error;
    AF_CJKAxis  axis;
    FT_Outline  outline;

    AF_StyleClass  sc = metrics->root.style_class;

    AF_Blue_Stringset         bss = sc->blue_stringset;
    const AF_Blue_StringRec*  bs  = &af_blue_stringsets[bss];

    /* If HarfBuzz is not available, we need a pointer to a single */
    /* unsigned long value.                                        */
#ifdef FT_CONFIG_OPTION_USE_HARFBUZZ
    void*     shaper_buf;
#else
    FT_ULong  shaper_buf_;
    void*     shaper_buf = &shaper_buf_;
#endif


    /* we walk over the blue character strings as specified in the   */
    /* style's entry in the `af_blue_stringset' array, computing its */
    /* extremum points (depending on the string properties)          */

    FT_TRACE5(( "cjk blue zones computation\n"
                "==========================\n"
                "\n" ));

#ifdef FT_CONFIG_OPTION_USE_HARFBUZZ
    shaper_buf = af_shaper_buf_create( face );
#endif

    for ( ; bs->string != AF_BLUE_STRING_MAX; bs++ )
    {
      const char*  p = &af_blue_strings[bs->string];
      FT_Pos*      blue_ref;
      FT_Pos*      blue_shoot;


      if ( AF_CJK_IS_HORIZ_BLUE( bs ) )
        axis = &metrics->axis[AF_DIMENSION_HORZ];
      else
        axis = &metrics->axis[AF_DIMENSION_VERT];

#ifdef FT_DEBUG_LEVEL_TRACE
      {
        FT_String*  cjk_blue_name[4] =
        {
          (FT_String*)"bottom",    /* --   , --  */
          (FT_String*)"top",       /* --   , TOP */
          (FT_String*)"left",      /* HORIZ, --  */
          (FT_String*)"right"      /* HORIZ, TOP */
        };


        FT_TRACE5(( "blue zone %d (%s):\n",
                    axis->blue_count,
                    cjk_blue_name[AF_CJK_IS_HORIZ_BLUE( bs ) |
                                  AF_CJK_IS_TOP_BLUE( bs )   ] ));
      }
#endif /* FT_DEBUG_LEVEL_TRACE */

      num_fills = 0;
      num_flats = 0;

      fill = 1;  /* start with characters that define fill values */
      FT_TRACE5(( "  [overshoot values]\n" ));

      while ( *p )
      {
        FT_ULong    glyph_index;
        FT_Pos      best_pos;       /* same as points.y or points.x, resp. */
        FT_Int      best_point;
        FT_Vector*  points;

        unsigned int  num_idx;

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

        /* switch to characters that define flat values */
        if ( *p == '|' )
        {
          fill = 0;
          FT_TRACE5(( "  [reference values]\n" ));
          p++;
          continue;
        }

        /* reject input that maps to more than a single glyph */
        p = af_shaper_get_cluster( p, &metrics->root, shaper_buf, &num_idx );
        if ( num_idx > 1 )
          continue;

        /* load the character in the face -- skip unknown or empty ones */
        glyph_index = af_shaper_get_elem( &metrics->root,
                                          shaper_buf,
                                          0,
                                          NULL,
                                          NULL );
        if ( glyph_index == 0 )
        {
          FT_TRACE5(( "  U+%04lX unavailable\n", ch ));
          continue;
        }

        error   = FT_Load_Glyph( face, glyph_index, FT_LOAD_NO_SCALE );
        outline = face->glyph->outline;
        if ( error || outline.n_points <= 2 )
        {
          FT_TRACE5(( "  U+%04lX contains no (usable) outlines\n", ch ));
          continue;
        }

        /* now compute min or max point indices and coordinates */
        points     = outline.points;
        best_point = -1;
        best_pos   = 0;  /* make compiler happy */

        {
          FT_Int  nn;
          FT_Int  first = 0;
          FT_Int  last  = -1;


          for ( nn = 0; nn < outline.n_contours; first = last + 1, nn++ )
          {
            FT_Int  pp;


            last = outline.contours[nn];

            /* Avoid single-point contours since they are never rasterized. */
            /* In some fonts, they correspond to mark attachment points     */
            /* which are way outside of the glyph's real outline.           */
            if ( last <= first )
              continue;

            if ( AF_CJK_IS_HORIZ_BLUE( bs ) )
            {
              if ( AF_CJK_IS_RIGHT_BLUE( bs ) )
              {
                for ( pp = first; pp <= last; pp++ )
                  if ( best_point < 0 || points[pp].x > best_pos )
                  {
                    best_point = pp;
                    best_pos   = points[pp].x;
                  }
              }
              else
              {
                for ( pp = first; pp <= last; pp++ )
                  if ( best_point < 0 || points[pp].x < best_pos )
                  {
                    best_point = pp;
                    best_pos   = points[pp].x;
                  }
              }
            }
            else
            {
              if ( AF_CJK_IS_TOP_BLUE( bs ) )
              {
                for ( pp = first; pp <= last; pp++ )
                  if ( best_point < 0 || points[pp].y > best_pos )
                  {
                    best_point = pp;
                    best_pos   = points[pp].y;
                  }
              }
              else
              {
                for ( pp = first; pp <= last; pp++ )
                  if ( best_point < 0 || points[pp].y < best_pos )
                  {
                    best_point = pp;
                    best_pos   = points[pp].y;
                  }
              }
            }
          }

          FT_TRACE5(( "  U+%04lX: best_pos = %5ld\n", ch, best_pos ));
        }

        if ( fill )
          fills[num_fills++] = best_pos;
        else
          flats[num_flats++] = best_pos;

      } /* end while loop */

      if ( num_flats == 0 && num_fills == 0 )
      {
        /*
         * we couldn't find a single glyph to compute this blue zone,
         * we will simply ignore it then
         */
        FT_TRACE5(( "  empty\n" ));
        continue;
      }

      /* we have computed the contents of the `fill' and `flats' tables,   */
      /* now determine the reference and overshoot position of the blue -- */
      /* we simply take the median value after a simple sort               */
      af_sort_pos( num_fills, fills );
      af_sort_pos( num_flats, flats );

      blue       = &axis->blues[axis->blue_count];
      blue_ref   = &blue->ref.org;
      blue_shoot = &blue->shoot.org;

      axis->blue_count++;

      if ( num_flats == 0 )
      {
        *blue_ref   =
        *blue_shoot = fills[num_fills / 2];
      }
      else if ( num_fills == 0 )
      {
        *blue_ref   =
        *blue_shoot = flats[num_flats / 2];
      }
      else
      {
        *blue_ref   = fills[num_fills / 2];
        *blue_shoot = flats[num_flats / 2];
      }

      /* make sure blue_ref >= blue_shoot for top/right or */
      /* vice versa for bottom/left                        */
      if ( *blue_shoot != *blue_ref )
      {
        FT_Pos   ref       = *blue_ref;
        FT_Pos   shoot     = *blue_shoot;
        FT_Bool  under_ref = FT_BOOL( shoot < ref );


        /* AF_CJK_IS_TOP_BLUE covers `right' and `top' */
        if ( AF_CJK_IS_TOP_BLUE( bs ) ^ under_ref )
        {
          *blue_ref   =
          *blue_shoot = ( shoot + ref ) / 2;

          FT_TRACE5(( "  [reference smaller than overshoot,"
                      " taking mean value]\n" ));
        }
      }

      blue->flags = 0;
      if ( AF_CJK_IS_TOP_BLUE( bs ) )
        blue->flags |= AF_CJK_BLUE_TOP;

      FT_TRACE5(( "    -> reference = %ld\n"
                  "       overshoot = %ld\n",
                  *blue_ref, *blue_shoot ));

    } /* end for loop */

    af_shaper_buf_destroy( face, shaper_buf );

    FT_TRACE5(( "\n" ));

    return;
  }


  /* Basically the Latin version with type AF_CJKMetrics for metrics. */

  FT_LOCAL_DEF( void )
  af_cjk_metrics_check_digits( AF_CJKMetrics  metrics,
                               FT_Face        face )
  {
    FT_Bool   started = 0, same_width = 1;
    FT_Fixed  advance = 0, old_advance = 0;

    /* If HarfBuzz is not available, we need a pointer to a single */
    /* unsigned long value.                                        */
#ifdef FT_CONFIG_OPTION_USE_HARFBUZZ
    void*     shaper_buf;
#else
    FT_ULong  shaper_buf_;
    void*     shaper_buf = &shaper_buf_;
#endif

    /* in all supported charmaps, digits have character codes 0x30-0x39 */
    const char   digits[] = "0 1 2 3 4 5 6 7 8 9";
    const char*  p;


    p = digits;

#ifdef FT_CONFIG_OPTION_USE_HARFBUZZ
    shaper_buf = af_shaper_buf_create( face );
#endif

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
  af_cjk_metrics_init( AF_CJKMetrics  metrics,
                       FT_Face        face )
  {
    FT_CharMap  oldmap = face->charmap;


    metrics->units_per_em = face->units_per_EM;

    if ( !FT_Select_Charmap( face, FT_ENCODING_UNICODE ) )
    {
      af_cjk_metrics_init_widths( metrics, face );
      af_cjk_metrics_init_blues( metrics, face );
      af_cjk_metrics_check_digits( metrics, face );
    }

    FT_Set_Charmap( face, oldmap );
    return FT_Err_Ok;
  }


  /* Adjust scaling value, then scale and shift widths   */
  /* and blue zones (if applicable) for given dimension. */

  static void
  af_cjk_metrics_scale_dim( AF_CJKMetrics  metrics,
                            AF_Scaler      scaler,
                            AF_Dimension   dim )
  {
    FT_Fixed    scale;
    FT_Pos      delta;
    AF_CJKAxis  axis;
    FT_UInt     nn;


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

    axis->scale = scale;
    axis->delta = delta;

    /* scale the blue zones */
    for ( nn = 0; nn < axis->blue_count; nn++ )
    {
      AF_CJKBlue  blue = &axis->blues[nn];
      FT_Pos      dist;


      blue->ref.cur   = FT_MulFix( blue->ref.org, scale ) + delta;
      blue->ref.fit   = blue->ref.cur;
      blue->shoot.cur = FT_MulFix( blue->shoot.org, scale ) + delta;
      blue->shoot.fit = blue->shoot.cur;
      blue->flags    &= ~AF_CJK_BLUE_ACTIVE;

      /* a blue zone is only active if it is less than 3/4 pixels tall */
      dist = FT_MulFix( blue->ref.org - blue->shoot.org, scale );
      if ( dist <= 48 && dist >= -48 )
      {
        FT_Pos  delta1, delta2;


        blue->ref.fit  = FT_PIX_ROUND( blue->ref.cur );

        /* shoot is under shoot for cjk */
        delta1 = FT_DivFix( blue->ref.fit, scale ) - blue->shoot.org;
        delta2 = delta1;
        if ( delta1 < 0 )
          delta2 = -delta2;

        delta2 = FT_MulFix( delta2, scale );

        FT_TRACE5(( "delta: %d", delta1 ));
        if ( delta2 < 32 )
          delta2 = 0;
#if 0
        else if ( delta2 < 64 )
          delta2 = 32 + ( ( ( delta2 - 32 ) + 16 ) & ~31 );
#endif
        else
          delta2 = FT_PIX_ROUND( delta2 );
        FT_TRACE5(( "/%d\n", delta2 ));

        if ( delta1 < 0 )
          delta2 = -delta2;

        blue->shoot.fit = blue->ref.fit - delta2;

        FT_TRACE5(( ">> active cjk blue zone %c%d[%ld/%ld]:\n"
                    "     ref:   cur=%.2f fit=%.2f\n"
                    "     shoot: cur=%.2f fit=%.2f\n",
                    ( dim == AF_DIMENSION_HORZ ) ? 'H' : 'V',
                    nn, blue->ref.org, blue->shoot.org,
                    blue->ref.cur / 64.0, blue->ref.fit / 64.0,
                    blue->shoot.cur / 64.0, blue->shoot.fit / 64.0 ));

        blue->flags |= AF_CJK_BLUE_ACTIVE;
      }
    }
  }


  /* Scale global values in both directions. */

  FT_LOCAL_DEF( void )
  af_cjk_metrics_scale( AF_CJKMetrics  metrics,
                        AF_Scaler      scaler )
  {
    /* we copy the whole structure since the x and y scaling values */
    /* are not modified, contrary to e.g. the `latin' auto-hinter   */
    metrics->root.scaler = *scaler;

    af_cjk_metrics_scale_dim( metrics, scaler, AF_DIMENSION_HORZ );
    af_cjk_metrics_scale_dim( metrics, scaler, AF_DIMENSION_VERT );
  }


  /* Extract standard_width from writing system/script specific */
  /* metrics class.                                             */

  FT_LOCAL_DEF( void )
  af_cjk_get_standard_widths( AF_CJKMetrics  metrics,
                              FT_Pos*        stdHW,
                              FT_Pos*        stdVW )
  {
    if ( stdHW )
      *stdHW = metrics->axis[AF_DIMENSION_VERT].standard_width;

    if ( stdVW )
      *stdVW = metrics->axis[AF_DIMENSION_HORZ].standard_width;
  }


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****              C J K   G L Y P H   A N A L Y S I S              *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/


  /* Walk over all contours and compute its segments. */

  static FT_Error
  af_cjk_hints_compute_segments( AF_GlyphHints  hints,
                                 AF_Dimension   dim )
  {
    AF_AxisHints  axis          = &hints->axis[dim];
    AF_Segment    segments      = axis->segments;
    AF_Segment    segment_limit = segments + axis->num_segments;
    FT_Error      error;
    AF_Segment    seg;


    error = af_latin_hints_compute_segments( hints, dim );
    if ( error )
      return error;

    /* a segment is round if it doesn't have successive */
    /* on-curve points.                                 */
    for ( seg = segments; seg < segment_limit; seg++ )
    {
      AF_Point  pt   = seg->first;
      AF_Point  last = seg->last;
      FT_UInt   f0   = pt->flags & AF_FLAG_CONTROL;
      FT_UInt   f1;


      seg->flags &= ~AF_EDGE_ROUND;

      for ( ; pt != last; f0 = f1 )
      {
        pt = pt->next;
        f1 = pt->flags & AF_FLAG_CONTROL;

        if ( !f0 && !f1 )
          break;

        if ( pt == last )
          seg->flags |= AF_EDGE_ROUND;
      }
    }

    return FT_Err_Ok;
  }


  static void
  af_cjk_hints_link_segments( AF_GlyphHints  hints,
                              AF_Dimension   dim )
  {
    AF_AxisHints  axis          = &hints->axis[dim];
    AF_Segment    segments      = axis->segments;
    AF_Segment    segment_limit = segments + axis->num_segments;
    AF_Direction  major_dir     = axis->major_dir;
    AF_Segment    seg1, seg2;
    FT_Pos        len_threshold;
    FT_Pos        dist_threshold;


    len_threshold = AF_LATIN_CONSTANT( hints->metrics, 8 );

    dist_threshold = ( dim == AF_DIMENSION_HORZ ) ? hints->x_scale
                                                  : hints->y_scale;
    dist_threshold = FT_DivFix( 64 * 3, dist_threshold );

    /* now compare each segment to the others */
    for ( seg1 = segments; seg1 < segment_limit; seg1++ )
    {
      if ( seg1->dir != major_dir )
        continue;

      for ( seg2 = segments; seg2 < segment_limit; seg2++ )
        if ( seg2 != seg1 && seg1->dir + seg2->dir == 0 )
        {
          FT_Pos  dist = seg2->pos - seg1->pos;


          if ( dist < 0 )
            continue;

          {
            FT_Pos  min = seg1->min_coord;
            FT_Pos  max = seg1->max_coord;
            FT_Pos  len;


            if ( min < seg2->min_coord )
              min = seg2->min_coord;

            if ( max > seg2->max_coord )
              max = seg2->max_coord;

            len = max - min;
            if ( len >= len_threshold )
            {
              if ( dist * 8 < seg1->score * 9                        &&
                   ( dist * 8 < seg1->score * 7 || seg1->len < len ) )
              {
                seg1->score = dist;
                seg1->len   = len;
                seg1->link  = seg2;
              }

              if ( dist * 8 < seg2->score * 9                        &&
                   ( dist * 8 < seg2->score * 7 || seg2->len < len ) )
              {
                seg2->score = dist;
                seg2->len   = len;
                seg2->link  = seg1;
              }
            }
          }
        }
    }

    /*
     * now compute the `serif' segments
     *
     * In Hanzi, some strokes are wider on one or both of the ends.
     * We either identify the stems on the ends as serifs or remove
     * the linkage, depending on the length of the stems.
     *
     */

    {
      AF_Segment  link1, link2;


      for ( seg1 = segments; seg1 < segment_limit; seg1++ )
      {
        link1 = seg1->link;
        if ( !link1 || link1->link != seg1 || link1->pos <= seg1->pos )
          continue;

        if ( seg1->score >= dist_threshold )
          continue;

        for ( seg2 = segments; seg2 < segment_limit; seg2++ )
        {
          if ( seg2->pos > seg1->pos || seg1 == seg2 )
            continue;

          link2 = seg2->link;
          if ( !link2 || link2->link != seg2 || link2->pos < link1->pos )
            continue;

          if ( seg1->pos == seg2->pos && link1->pos == link2->pos )
            continue;

          if ( seg2->score <= seg1->score || seg1->score * 4 <= seg2->score )
            continue;

          /* seg2 < seg1 < link1 < link2 */

          if ( seg1->len >= seg2->len * 3 )
          {
            AF_Segment  seg;


            for ( seg = segments; seg < segment_limit; seg++ )
            {
              AF_Segment  link = seg->link;


              if ( link == seg2 )
              {
                seg->link  = NULL;
                seg->serif = link1;
              }
              else if ( link == link2 )
              {
                seg->link  = NULL;
                seg->serif = seg1;
              }
            }
          }
          else
          {
            seg1->link = link1->link = NULL;

            break;
          }
        }
      }
    }

    for ( seg1 = segments; seg1 < segment_limit; seg1++ )
    {
      seg2 = seg1->link;

      if ( seg2 )
      {
        if ( seg2->link != seg1 )
        {
          seg1->link = NULL;

          if ( seg2->score < dist_threshold || seg1->score < seg2->score * 4 )
            seg1->serif = seg2->link;
        }
      }
    }
  }


  static FT_Error
  af_cjk_hints_compute_edges( AF_GlyphHints  hints,
                              AF_Dimension   dim )
  {
    AF_AxisHints  axis   = &hints->axis[dim];
    FT_Error      error  = FT_Err_Ok;
    FT_Memory     memory = hints->memory;
    AF_CJKAxis    laxis  = &((AF_CJKMetrics)hints->metrics)->axis[dim];

    AF_Segment    segments      = axis->segments;
    AF_Segment    segment_limit = segments + axis->num_segments;
    AF_Segment    seg;

    FT_Fixed      scale;
    FT_Pos        edge_distance_threshold;


    axis->num_edges = 0;

    scale = ( dim == AF_DIMENSION_HORZ ) ? hints->x_scale
                                         : hints->y_scale;

    /**********************************************************************
     *
     * We begin by generating a sorted table of edges for the current
     * direction.  To do so, we simply scan each segment and try to find
     * an edge in our table that corresponds to its position.
     *
     * If no edge is found, we create and insert a new edge in the
     * sorted table.  Otherwise, we simply add the segment to the edge's
     * list which is then processed in the second step to compute the
     * edge's properties.
     *
     * Note that the edges table is sorted along the segment/edge
     * position.
     *
     */

    edge_distance_threshold = FT_MulFix( laxis->edge_distance_threshold,
                                         scale );
    if ( edge_distance_threshold > 64 / 4 )
      edge_distance_threshold = FT_DivFix( 64 / 4, scale );
    else
      edge_distance_threshold = laxis->edge_distance_threshold;

    for ( seg = segments; seg < segment_limit; seg++ )
    {
      AF_Edge  found = NULL;
      FT_Pos   best  = 0xFFFFU;
      FT_Int   ee;


      /* look for an edge corresponding to the segment */
      for ( ee = 0; ee < axis->num_edges; ee++ )
      {
        AF_Edge  edge = axis->edges + ee;
        FT_Pos   dist;


        if ( edge->dir != seg->dir )
          continue;

        dist = seg->pos - edge->fpos;
        if ( dist < 0 )
          dist = -dist;

        if ( dist < edge_distance_threshold && dist < best )
        {
          AF_Segment  link = seg->link;


          /* check whether all linked segments of the candidate edge */
          /* can make a single edge.                                 */
          if ( link )
          {
            AF_Segment  seg1  = edge->first;
            FT_Pos      dist2 = 0;


            do
            {
              AF_Segment  link1 = seg1->link;


              if ( link1 )
              {
                dist2 = AF_SEGMENT_DIST( link, link1 );
                if ( dist2 >= edge_distance_threshold )
                  break;
              }

            } while ( ( seg1 = seg1->edge_next ) != edge->first );

            if ( dist2 >= edge_distance_threshold )
              continue;
          }

          best  = dist;
          found = edge;
        }
      }

      if ( !found )
      {
        AF_Edge  edge;


        /* insert a new edge in the list and */
        /* sort according to the position    */
        error = af_axis_hints_new_edge( axis, seg->pos,
                                        (AF_Direction)seg->dir, 0,
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

    /*******************************************************************
     *
     * Good, we now compute each edge's properties according to the
     * segments found on its position.  Basically, these are
     *
     * - the edge's main direction
     * - stem edge, serif edge or both (which defaults to stem then)
     * - rounded edge, straight or both (which defaults to straight)
     * - link for edge
     *
     */

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


        seg = edge->first;

        do
        {
          FT_Bool  is_serif;


          /* check for roundness of segment */
          if ( seg->flags & AF_EDGE_ROUND )
            is_round++;
          else
            is_straight++;

          /* check for links -- if seg->serif is set, then seg->link must */
          /* be ignored                                                   */
          is_serif = FT_BOOL( seg->serif && seg->serif->edge != edge );

          if ( seg->link || is_serif )
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

              seg_delta = AF_SEGMENT_DIST( seg, seg2 );

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

  static FT_Error
  af_cjk_hints_detect_features( AF_GlyphHints  hints,
                                AF_Dimension   dim )
  {
    FT_Error  error;


    error = af_cjk_hints_compute_segments( hints, dim );
    if ( !error )
    {
      af_cjk_hints_link_segments( hints, dim );

      error = af_cjk_hints_compute_edges( hints, dim );
    }
    return error;
  }


  /* Compute all edges which lie within blue zones. */

  static void
  af_cjk_hints_compute_blue_edges( AF_GlyphHints  hints,
                                   AF_CJKMetrics  metrics,
                                   AF_Dimension   dim )
  {
    AF_AxisHints  axis       = &hints->axis[dim];
    AF_Edge       edge       = axis->edges;
    AF_Edge       edge_limit = edge + axis->num_edges;
    AF_CJKAxis    cjk        = &metrics->axis[dim];
    FT_Fixed      scale      = cjk->scale;
    FT_Pos        best_dist0;  /* initial threshold */


    /* compute the initial threshold as a fraction of the EM size */
    best_dist0 = FT_MulFix( metrics->units_per_em / 40, scale );

    if ( best_dist0 > 64 / 2 ) /* maximum 1/2 pixel */
      best_dist0 = 64 / 2;

    /* compute which blue zones are active, i.e. have their scaled */
    /* size < 3/4 pixels                                           */

    /* If the distant between an edge and a blue zone is shorter than */
    /* best_dist0, set the blue zone for the edge.  Then search for   */
    /* the blue zone with the smallest best_dist to the edge.         */

    for ( ; edge < edge_limit; edge++ )
    {
      FT_UInt   bb;
      AF_Width  best_blue = NULL;
      FT_Pos    best_dist = best_dist0;


      for ( bb = 0; bb < cjk->blue_count; bb++ )
      {
        AF_CJKBlue  blue = cjk->blues + bb;
        FT_Bool     is_top_right_blue, is_major_dir;


        /* skip inactive blue zones (i.e., those that are too small) */
        if ( !( blue->flags & AF_CJK_BLUE_ACTIVE ) )
          continue;

        /* if it is a top zone, check for right edges -- if it is a bottom */
        /* zone, check for left edges                                      */
        /*                                                                 */
        /* of course, that's for TrueType                                  */
        is_top_right_blue =
          (FT_Byte)( ( blue->flags & AF_CJK_BLUE_TOP ) != 0 );
        is_major_dir =
          FT_BOOL( edge->dir == axis->major_dir );

        /* if it is a top zone, the edge must be against the major    */
        /* direction; if it is a bottom zone, it must be in the major */
        /* direction                                                  */
        if ( is_top_right_blue ^ is_major_dir )
        {
          FT_Pos    dist;
          AF_Width  compare;


          /* Compare the edge to the closest blue zone type */
          if ( FT_ABS( edge->fpos - blue->ref.org ) >
               FT_ABS( edge->fpos - blue->shoot.org ) )
            compare = &blue->shoot;
          else
            compare = &blue->ref;

          dist = edge->fpos - compare->org;
          if ( dist < 0 )
            dist = -dist;

          dist = FT_MulFix( dist, scale );
          if ( dist < best_dist )
          {
            best_dist = dist;
            best_blue = compare;
          }
        }
      }

      if ( best_blue )
        edge->blue_edge = best_blue;
    }
  }


  /* Initalize hinting engine. */

  FT_LOCAL_DEF( FT_Error )
  af_cjk_hints_init( AF_GlyphHints  hints,
                     AF_CJKMetrics  metrics )
  {
    FT_Render_Mode  mode;
    FT_UInt32       scaler_flags, other_flags;


    af_glyph_hints_rescale( hints, (AF_StyleMetrics)metrics );

    /*
     * correct x_scale and y_scale when needed, since they may have
     * been modified af_cjk_scale_dim above
     */
    hints->x_scale = metrics->axis[AF_DIMENSION_HORZ].scale;
    hints->x_delta = metrics->axis[AF_DIMENSION_HORZ].delta;
    hints->y_scale = metrics->axis[AF_DIMENSION_VERT].scale;
    hints->y_delta = metrics->axis[AF_DIMENSION_VERT].delta;

    /* compute flags depending on render mode, etc. */
    mode = metrics->root.scaler.render_mode;

#if 0 /* AF_CONFIG_OPTION_USE_WARPER */
    if ( mode == FT_RENDER_MODE_LCD || mode == FT_RENDER_MODE_LCD_V )
      metrics->root.scaler.render_mode = mode = FT_RENDER_MODE_NORMAL;
#endif

    scaler_flags = hints->scaler_flags;
    other_flags  = 0;

    /*
     * We snap the width of vertical stems for the monochrome and
     * horizontal LCD rendering targets only.
     */
    if ( mode == FT_RENDER_MODE_MONO || mode == FT_RENDER_MODE_LCD )
      other_flags |= AF_LATIN_HINTS_HORZ_SNAP;

    /*
     * We snap the width of horizontal stems for the monochrome and
     * vertical LCD rendering targets only.
     */
    if ( mode == FT_RENDER_MODE_MONO || mode == FT_RENDER_MODE_LCD_V )
      other_flags |= AF_LATIN_HINTS_VERT_SNAP;

    /*
     * We adjust stems to full pixels unless in `light' or `lcd' mode.
     */
    if ( mode != FT_RENDER_MODE_LIGHT && mode != FT_RENDER_MODE_LCD )
      other_flags |= AF_LATIN_HINTS_STEM_ADJUST;

    if ( mode == FT_RENDER_MODE_MONO )
      other_flags |= AF_LATIN_HINTS_MONO;

    scaler_flags |= AF_SCALER_FLAG_NO_ADVANCE;

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
  /*****          C J K   G L Y P H   G R I D - F I T T I N G          *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  /* Snap a given width in scaled coordinates to one of the */
  /* current standard widths.                               */

  static FT_Pos
  af_cjk_snap_width( AF_Width  widths,
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


  /* Compute the snapped width of a given stem.                          */
  /* There is a lot of voodoo in this function; changing the hard-coded  */
  /* parameters influence the whole hinting process.                     */

  static FT_Pos
  af_cjk_compute_stem_width( AF_GlyphHints  hints,
                             AF_Dimension   dim,
                             FT_Pos         width,
                             FT_UInt        base_flags,
                             FT_UInt        stem_flags )
  {
    AF_CJKMetrics  metrics  = (AF_CJKMetrics)hints->metrics;
    AF_CJKAxis     axis     = &metrics->axis[dim];
    FT_Pos         dist     = width;
    FT_Int         sign     = 0;
    FT_Bool        vertical = FT_BOOL( dim == AF_DIMENSION_VERT );

    FT_UNUSED( base_flags );
    FT_UNUSED( stem_flags );


    if ( !AF_LATIN_HINTS_DO_STEM_ADJUST( hints ) )
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

      if ( axis->width_count > 0 )
      {
        if ( FT_ABS( dist - axis->widths[0].cur ) < 40 )
        {
          dist = axis->widths[0].cur;
          if ( dist < 48 )
            dist = 48;

          goto Done_Width;
        }
      }

      if ( dist < 54 )
        dist += ( 54 - dist ) / 2;
      else if ( dist < 3 * 64 )
      {
        FT_Pos  delta;


        delta  = dist & 63;
        dist  &= -64;

        if ( delta < 10 )
          dist += delta;
        else if ( delta < 22 )
          dist += 10;
        else if ( delta < 42 )
          dist += delta;
        else if ( delta < 54 )
          dist += 54;
        else
          dist += delta;
      }
    }
    else
    {
      /* strong hinting process: snap the stem width to integer pixels */

      dist = af_cjk_snap_width( axis->widths, axis->width_count, dist );

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
            dist = ( dist + 22 ) & ~63;
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
  af_cjk_align_linked_edge( AF_GlyphHints  hints,
                            AF_Dimension   dim,
                            AF_Edge        base_edge,
                            AF_Edge        stem_edge )
  {
    FT_Pos  dist = stem_edge->opos - base_edge->opos;

    FT_Pos  fitted_width = af_cjk_compute_stem_width( hints, dim, dist,
                                                      base_edge->flags,
                                                      stem_edge->flags );


    stem_edge->pos = base_edge->pos + fitted_width;

    FT_TRACE5(( "  CJKLINK: edge %d @%d (opos=%.2f) linked to %.2f,"
                " dist was %.2f, now %.2f\n",
                stem_edge - hints->axis[dim].edges, stem_edge->fpos,
                stem_edge->opos / 64.0, stem_edge->pos / 64.0,
                dist / 64.0, fitted_width / 64.0 ));
  }


  /* Shift the coordinates of the `serif' edge by the same amount */
  /* as the corresponding `base' edge has been moved already.     */

  static void
  af_cjk_align_serif_edge( AF_GlyphHints  hints,
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


#define AF_LIGHT_MODE_MAX_HORZ_GAP    9
#define AF_LIGHT_MODE_MAX_VERT_GAP   15
#define AF_LIGHT_MODE_MAX_DELTA_ABS  14


  static FT_Pos
  af_hint_normal_stem( AF_GlyphHints  hints,
                       AF_Edge        edge,
                       AF_Edge        edge2,
                       FT_Pos         anchor,
                       AF_Dimension   dim )
  {
    FT_Pos  org_len, cur_len, org_center;
    FT_Pos  cur_pos1, cur_pos2;
    FT_Pos  d_off1, u_off1, d_off2, u_off2, delta;
    FT_Pos  offset;
    FT_Pos  threshold = 64;


    if ( !AF_LATIN_HINTS_DO_STEM_ADJUST( hints ) )
    {
      if ( ( edge->flags  & AF_EDGE_ROUND ) &&
           ( edge2->flags & AF_EDGE_ROUND ) )
      {
        if ( dim == AF_DIMENSION_VERT )
          threshold = 64 - AF_LIGHT_MODE_MAX_HORZ_GAP;
        else
          threshold = 64 - AF_LIGHT_MODE_MAX_VERT_GAP;
      }
      else
      {
        if ( dim == AF_DIMENSION_VERT )
          threshold = 64 - AF_LIGHT_MODE_MAX_HORZ_GAP / 3;
        else
          threshold = 64 - AF_LIGHT_MODE_MAX_VERT_GAP / 3;
      }
    }

    org_len    = edge2->opos - edge->opos;
    cur_len    = af_cjk_compute_stem_width( hints, dim, org_len,
                                            edge->flags,
                                            edge2->flags );

    org_center = ( edge->opos + edge2->opos ) / 2 + anchor;
    cur_pos1   = org_center - cur_len / 2;
    cur_pos2   = cur_pos1 + cur_len;
    d_off1     = cur_pos1 - FT_PIX_FLOOR( cur_pos1 );
    d_off2     = cur_pos2 - FT_PIX_FLOOR( cur_pos2 );
    u_off1     = 64 - d_off1;
    u_off2     = 64 - d_off2;
    delta      = 0;


    if ( d_off1 == 0 || d_off2 == 0 )
      goto Exit;

    if ( cur_len <= threshold )
    {
      if ( d_off2 < cur_len )
      {
        if ( u_off1 <= d_off2 )
          delta =  u_off1;
        else
          delta = -d_off2;
      }

      goto Exit;
    }

    if ( threshold < 64 )
    {
      if ( d_off1 >= threshold || u_off1 >= threshold ||
           d_off2 >= threshold || u_off2 >= threshold )
        goto Exit;
    }

    offset = cur_len & 63;

    if ( offset < 32 )
    {
      if ( u_off1 <= offset || d_off2 <= offset )
        goto Exit;
    }
    else
      offset = 64 - threshold;

    d_off1 = threshold - u_off1;
    u_off1 = u_off1    - offset;
    u_off2 = threshold - d_off2;
    d_off2 = d_off2    - offset;

    if ( d_off1 <= u_off1 )
      u_off1 = -d_off1;

    if ( d_off2 <= u_off2 )
      u_off2 = -d_off2;

    if ( FT_ABS( u_off1 ) <= FT_ABS( u_off2 ) )
      delta = u_off1;
    else
      delta = u_off2;

  Exit:

#if 1
    if ( !AF_LATIN_HINTS_DO_STEM_ADJUST( hints ) )
    {
      if ( delta > AF_LIGHT_MODE_MAX_DELTA_ABS )
        delta = AF_LIGHT_MODE_MAX_DELTA_ABS;
      else if ( delta < -AF_LIGHT_MODE_MAX_DELTA_ABS )
        delta = -AF_LIGHT_MODE_MAX_DELTA_ABS;
    }
#endif

    cur_pos1 += delta;

    if ( edge->opos < edge2->opos )
    {
      edge->pos  = cur_pos1;
      edge2->pos = cur_pos1 + cur_len;
    }
    else
    {
      edge->pos  = cur_pos1 + cur_len;
      edge2->pos = cur_pos1;
    }

    return delta;
  }


  /* The main grid-fitting routine. */

  static void
  af_cjk_hint_edges( AF_GlyphHints  hints,
                     AF_Dimension   dim )
  {
    AF_AxisHints  axis       = &hints->axis[dim];
    AF_Edge       edges      = axis->edges;
    AF_Edge       edge_limit = edges + axis->num_edges;
    FT_PtrDist    n_edges;
    AF_Edge       edge;
    AF_Edge       anchor   = NULL;
    FT_Pos        delta    = 0;
    FT_Int        skipped  = 0;
    FT_Bool       has_last_stem = FALSE;
    FT_Pos        last_stem_pos = 0;

#ifdef FT_DEBUG_LEVEL_TRACE
    FT_UInt       num_actions = 0;
#endif


    FT_TRACE5(( "cjk %s edge hinting (style `%s')\n",
                dim == AF_DIMENSION_VERT ? "horizontal" : "vertical",
                af_style_names[hints->metrics->style_class->style] ));

    /* we begin by aligning all stems relative to the blue zone */

    if ( AF_HINTS_DO_BLUES( hints ) )
    {
      for ( edge = edges; edge < edge_limit; edge++ )
      {
        AF_Width  blue;
        AF_Edge   edge1, edge2;


        if ( edge->flags & AF_EDGE_DONE )
          continue;

        blue  = edge->blue_edge;
        edge1 = NULL;
        edge2 = edge->link;

        if ( blue )
        {
          edge1 = edge;
        }
        else if ( edge2 && edge2->blue_edge )
        {
          blue  = edge2->blue_edge;
          edge1 = edge2;
          edge2 = edge;
        }

        if ( !edge1 )
          continue;

#ifdef FT_DEBUG_LEVEL_TRACE
        FT_TRACE5(( "  CJKBLUE: edge %d @%d (opos=%.2f) snapped to %.2f,"
                    " was %.2f\n",
                    edge1 - edges, edge1->fpos, edge1->opos / 64.0,
                    blue->fit / 64.0, edge1->pos / 64.0 ));

        num_actions++;
#endif

        edge1->pos    = blue->fit;
        edge1->flags |= AF_EDGE_DONE;

        if ( edge2 && !edge2->blue_edge )
        {
          af_cjk_align_linked_edge( hints, dim, edge1, edge2 );
          edge2->flags |= AF_EDGE_DONE;

#ifdef FT_DEBUG_LEVEL_TRACE
          num_actions++;
#endif
        }

        if ( !anchor )
          anchor = edge;
      }
    }

    /* now we align all stem edges. */
    for ( edge = edges; edge < edge_limit; edge++ )
    {
      AF_Edge  edge2;


      if ( edge->flags & AF_EDGE_DONE )
        continue;

      /* skip all non-stem edges */
      edge2 = edge->link;
      if ( !edge2 )
      {
        skipped++;
        continue;
      }

      /* Some CJK characters have so many stems that
       * the hinter is likely to merge two adjacent ones.
       * To solve this problem, if either edge of a stem
       * is too close to the previous one, we avoid
       * aligning the two edges, but rather interpolate
       * their locations at the end of this function in
       * order to preserve the space between the stems.
       */
      if ( has_last_stem                       &&
           ( edge->pos  < last_stem_pos + 64 ||
             edge2->pos < last_stem_pos + 64 ) )
      {
        skipped++;
        continue;
      }

      /* now align the stem */

      /* this should not happen, but it's better to be safe */
      if ( edge2->blue_edge )
      {
        FT_TRACE5(( "ASSERTION FAILED for edge %d\n", edge2-edges ));

        af_cjk_align_linked_edge( hints, dim, edge2, edge );
        edge->flags |= AF_EDGE_DONE;

#ifdef FT_DEBUG_LEVEL_TRACE
        num_actions++;
#endif

        continue;
      }

      if ( edge2 < edge )
      {
        af_cjk_align_linked_edge( hints, dim, edge2, edge );
        edge->flags |= AF_EDGE_DONE;

#ifdef FT_DEBUG_LEVEL_TRACE
        num_actions++;
#endif

        /* We rarely reaches here it seems;
         * usually the two edges belonging
         * to one stem are marked as DONE together
         */
        has_last_stem = TRUE;
        last_stem_pos = edge->pos;
        continue;
      }

      if ( dim != AF_DIMENSION_VERT && !anchor )
      {

#if 0
        if ( fixedpitch )
        {
          AF_Edge     left  = edge;
          AF_Edge     right = edge_limit - 1;
          AF_EdgeRec  left1, left2, right1, right2;
          FT_Pos      target, center1, center2;
          FT_Pos      delta1, delta2, d1, d2;


          while ( right > left && !right->link )
            right--;

          left1  = *left;
          left2  = *left->link;
          right1 = *right->link;
          right2 = *right;

          delta  = ( ( ( hinter->pp2.x + 32 ) & -64 ) - hinter->pp2.x ) / 2;
          target = left->opos + ( right->opos - left->opos ) / 2 + delta - 16;

          delta1  = delta;
          delta1 += af_hint_normal_stem( hints, left, left->link,
                                         delta1, 0 );

          if ( left->link != right )
            af_hint_normal_stem( hints, right->link, right, delta1, 0 );

          center1 = left->pos + ( right->pos - left->pos ) / 2;

          if ( center1 >= target )
            delta2 = delta - 32;
          else
            delta2 = delta + 32;

          delta2 += af_hint_normal_stem( hints, &left1, &left2, delta2, 0 );

          if ( delta1 != delta2 )
          {
            if ( left->link != right )
              af_hint_normal_stem( hints, &right1, &right2, delta2, 0 );

            center2 = left1.pos + ( right2.pos - left1.pos ) / 2;

            d1 = center1 - target;
            d2 = center2 - target;

            if ( FT_ABS( d2 ) < FT_ABS( d1 ) )
            {
              left->pos       = left1.pos;
              left->link->pos = left2.pos;

              if ( left->link != right )
              {
                right->link->pos = right1.pos;
                right->pos       = right2.pos;
              }

              delta1 = delta2;
            }
          }

          delta               = delta1;
          right->link->flags |= AF_EDGE_DONE;
          right->flags       |= AF_EDGE_DONE;
        }
        else

#endif /* 0 */

          delta = af_hint_normal_stem( hints, edge, edge2, 0,
                                       AF_DIMENSION_HORZ );
      }
      else
        af_hint_normal_stem( hints, edge, edge2, delta, dim );

#if 0
      printf( "stem (%d,%d) adjusted (%.1f,%.1f)\n",
               edge - edges, edge2 - edges,
               ( edge->pos - edge->opos ) / 64.0,
               ( edge2->pos - edge2->opos ) / 64.0 );
#endif

      anchor = edge;
      edge->flags  |= AF_EDGE_DONE;
      edge2->flags |= AF_EDGE_DONE;
      has_last_stem = TRUE;
      last_stem_pos = edge2->pos;
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
      FT_Pos   dist1, dist2, span;


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

      if ( edge1->link == edge1 + 1 &&
           edge2->link == edge2 + 1 &&
           edge3->link == edge3 + 1 && span < 8 )
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

    if ( !skipped )
      goto Exit;

    /*
     * now hint the remaining edges (serifs and single) in order
     * to complete our processing
     */
    for ( edge = edges; edge < edge_limit; edge++ )
    {
      if ( edge->flags & AF_EDGE_DONE )
        continue;

      if ( edge->serif )
      {
        af_cjk_align_serif_edge( hints, edge->serif, edge );
        edge->flags |= AF_EDGE_DONE;
        skipped--;
      }
    }

    if ( !skipped )
      goto Exit;

    for ( edge = edges; edge < edge_limit; edge++ )
    {
      AF_Edge  before, after;


      if ( edge->flags & AF_EDGE_DONE )
        continue;

      before = after = edge;

      while ( --before >= edges )
        if ( before->flags & AF_EDGE_DONE )
          break;

      while ( ++after < edge_limit )
        if ( after->flags & AF_EDGE_DONE )
          break;

      if ( before >= edges || after < edge_limit )
      {
        if ( before < edges )
          af_cjk_align_serif_edge( hints, after, edge );
        else if ( after >= edge_limit )
          af_cjk_align_serif_edge( hints, before, edge );
        else
        {
          if ( after->fpos == before->fpos )
            edge->pos = before->pos;
          else
            edge->pos = before->pos +
                        FT_MulDiv( edge->fpos - before->fpos,
                                   after->pos - before->pos,
                                   after->fpos - before->fpos );
        }
      }
    }

  Exit:

#ifdef FT_DEBUG_LEVEL_TRACE
    if ( !num_actions )
      FT_TRACE5(( "  (none)\n" ));
    FT_TRACE5(( "\n" ));
#endif

    return;
  }


  static void
  af_cjk_align_edge_points( AF_GlyphHints  hints,
                            AF_Dimension   dim )
  {
    AF_AxisHints  axis       = & hints->axis[dim];
    AF_Edge       edges      = axis->edges;
    AF_Edge       edge_limit = edges + axis->num_edges;
    AF_Edge       edge;
    FT_Bool       snapping;


    snapping = FT_BOOL( ( dim == AF_DIMENSION_HORZ             &&
                          AF_LATIN_HINTS_DO_HORZ_SNAP( hints ) )  ||
                        ( dim == AF_DIMENSION_VERT             &&
                          AF_LATIN_HINTS_DO_VERT_SNAP( hints ) )  );

    for ( edge = edges; edge < edge_limit; edge++ )
    {
      /* move the points of each segment     */
      /* in each edge to the edge's position */
      AF_Segment  seg = edge->first;


      if ( snapping )
      {
        do
        {
          AF_Point  point = seg->first;


          for (;;)
          {
            if ( dim == AF_DIMENSION_HORZ )
            {
              point->x      = edge->pos;
              point->flags |= AF_FLAG_TOUCH_X;
            }
            else
            {
              point->y      = edge->pos;
              point->flags |= AF_FLAG_TOUCH_Y;
            }

            if ( point == seg->last )
              break;

            point = point->next;
          }

          seg = seg->edge_next;

        } while ( seg != edge->first );
      }
      else
      {
        FT_Pos  delta = edge->pos - edge->opos;


        do
        {
          AF_Point  point = seg->first;


          for (;;)
          {
            if ( dim == AF_DIMENSION_HORZ )
            {
              point->x     += delta;
              point->flags |= AF_FLAG_TOUCH_X;
            }
            else
            {
              point->y     += delta;
              point->flags |= AF_FLAG_TOUCH_Y;
            }

            if ( point == seg->last )
              break;

            point = point->next;
          }

          seg = seg->edge_next;

        } while ( seg != edge->first );
      }
    }
  }


  /* Apply the complete hinting algorithm to a CJK glyph. */

  FT_LOCAL_DEF( FT_Error )
  af_cjk_hints_apply( FT_UInt        glyph_index,
                      AF_GlyphHints  hints,
                      FT_Outline*    outline,
                      AF_CJKMetrics  metrics )
  {
    FT_Error  error;
    int       dim;

    FT_UNUSED( metrics );
    FT_UNUSED( glyph_index );


    error = af_glyph_hints_reload( hints, outline );
    if ( error )
      goto Exit;

    /* analyze glyph outline */
    if ( AF_HINTS_DO_HORIZONTAL( hints ) )
    {
      error = af_cjk_hints_detect_features( hints, AF_DIMENSION_HORZ );
      if ( error )
        goto Exit;

      af_cjk_hints_compute_blue_edges( hints, metrics, AF_DIMENSION_HORZ );
    }

    if ( AF_HINTS_DO_VERTICAL( hints ) )
    {
      error = af_cjk_hints_detect_features( hints, AF_DIMENSION_VERT );
      if ( error )
        goto Exit;

      af_cjk_hints_compute_blue_edges( hints, metrics, AF_DIMENSION_VERT );
    }

    /* grid-fit the outline */
    for ( dim = 0; dim < AF_DIMENSION_MAX; dim++ )
    {
      if ( ( dim == AF_DIMENSION_HORZ && AF_HINTS_DO_HORIZONTAL( hints ) ) ||
           ( dim == AF_DIMENSION_VERT && AF_HINTS_DO_VERTICAL( hints ) )   )
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

        af_cjk_hint_edges( hints, (AF_Dimension)dim );
        af_cjk_align_edge_points( hints, (AF_Dimension)dim );
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
  /*****                C J K   S C R I P T   C L A S S                *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/


  AF_DEFINE_WRITING_SYSTEM_CLASS(
    af_cjk_writing_system_class,

    AF_WRITING_SYSTEM_CJK,

    sizeof ( AF_CJKMetricsRec ),

    (AF_WritingSystem_InitMetricsFunc) af_cjk_metrics_init,        /* style_metrics_init    */
    (AF_WritingSystem_ScaleMetricsFunc)af_cjk_metrics_scale,       /* style_metrics_scale   */
    (AF_WritingSystem_DoneMetricsFunc) NULL,                       /* style_metrics_done    */
    (AF_WritingSystem_GetStdWidthsFunc)af_cjk_get_standard_widths, /* style_metrics_getstdw */

    (AF_WritingSystem_InitHintsFunc)   af_cjk_hints_init,          /* style_hints_init      */
    (AF_WritingSystem_ApplyHintsFunc)  af_cjk_hints_apply          /* style_hints_apply     */
  )


#else /* !AF_CONFIG_OPTION_CJK */


  AF_DEFINE_WRITING_SYSTEM_CLASS(
    af_cjk_writing_system_class,

    AF_WRITING_SYSTEM_CJK,

    sizeof ( AF_CJKMetricsRec ),

    (AF_WritingSystem_InitMetricsFunc) NULL, /* style_metrics_init    */
    (AF_WritingSystem_ScaleMetricsFunc)NULL, /* style_metrics_scale   */
    (AF_WritingSystem_DoneMetricsFunc) NULL, /* style_metrics_done    */
    (AF_WritingSystem_GetStdWidthsFunc)NULL, /* style_metrics_getstdw */

    (AF_WritingSystem_InitHintsFunc)   NULL, /* style_hints_init      */
    (AF_WritingSystem_ApplyHintsFunc)  NULL  /* style_hints_apply     */
  )


#endif /* !AF_CONFIG_OPTION_CJK */


/* END */
