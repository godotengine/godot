/****************************************************************************
 *
 * pshalgo.c
 *
 *   PostScript hinting algorithm (body).
 *
 * Copyright (C) 2001-2020 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used
 * modified and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#include <ft2build.h>
#include FT_INTERNAL_OBJECTS_H
#include FT_INTERNAL_DEBUG_H
#include FT_INTERNAL_CALC_H
#include "pshalgo.h"

#include "pshnterr.h"


#undef  FT_COMPONENT
#define FT_COMPONENT  pshalgo


#ifdef DEBUG_HINTER
  PSH_Hint_Table  ps_debug_hint_table = NULL;
  PSH_HintFunc    ps_debug_hint_func  = NULL;
  PSH_Glyph       ps_debug_glyph      = NULL;
#endif


#define  COMPUTE_INFLEXS  /* compute inflection points to optimize `S' */
                          /* and similar glyphs                        */


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                  BASIC HINTS RECORDINGS                       *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  /* return true if two stem hints overlap */
  static FT_Int
  psh_hint_overlap( PSH_Hint  hint1,
                    PSH_Hint  hint2 )
  {
    return ADD_INT( hint1->org_pos, hint1->org_len ) >= hint2->org_pos &&
           ADD_INT( hint2->org_pos, hint2->org_len ) >= hint1->org_pos;
  }


  /* destroy hints table */
  static void
  psh_hint_table_done( PSH_Hint_Table  table,
                       FT_Memory       memory )
  {
    FT_FREE( table->zones );
    table->num_zones = 0;
    table->zone      = NULL;

    FT_FREE( table->sort );
    FT_FREE( table->hints );
    table->num_hints   = 0;
    table->max_hints   = 0;
    table->sort_global = NULL;
  }


  /* deactivate all hints in a table */
  static void
  psh_hint_table_deactivate( PSH_Hint_Table  table )
  {
    FT_UInt   count = table->max_hints;
    PSH_Hint  hint  = table->hints;


    for ( ; count > 0; count--, hint++ )
    {
      psh_hint_deactivate( hint );
      hint->order = -1;
    }
  }


  /* internal function to record a new hint */
  static void
  psh_hint_table_record( PSH_Hint_Table  table,
                         FT_UInt         idx )
  {
    PSH_Hint  hint = table->hints + idx;


    if ( idx >= table->max_hints )
    {
      FT_TRACE0(( "psh_hint_table_record: invalid hint index %d\n", idx ));
      return;
    }

    /* ignore active hints */
    if ( psh_hint_is_active( hint ) )
      return;

    psh_hint_activate( hint );

    /* now scan the current active hint set to check */
    /* whether `hint' overlaps with another hint     */
    {
      PSH_Hint*  sorted = table->sort_global;
      FT_UInt    count  = table->num_hints;
      PSH_Hint   hint2;


      hint->parent = NULL;
      for ( ; count > 0; count--, sorted++ )
      {
        hint2 = sorted[0];

        if ( psh_hint_overlap( hint, hint2 ) )
        {
          hint->parent = hint2;
          break;
        }
      }
    }

    if ( table->num_hints < table->max_hints )
      table->sort_global[table->num_hints++] = hint;
    else
      FT_TRACE0(( "psh_hint_table_record: too many sorted hints!  BUG!\n" ));
  }


  static void
  psh_hint_table_record_mask( PSH_Hint_Table  table,
                              PS_Mask         hint_mask )
  {
    FT_Int    mask = 0, val = 0;
    FT_Byte*  cursor = hint_mask->bytes;
    FT_UInt   idx, limit;


    limit = hint_mask->num_bits;

    for ( idx = 0; idx < limit; idx++ )
    {
      if ( mask == 0 )
      {
        val  = *cursor++;
        mask = 0x80;
      }

      if ( val & mask )
        psh_hint_table_record( table, idx );

      mask >>= 1;
    }
  }


  /* create hints table */
  static FT_Error
  psh_hint_table_init( PSH_Hint_Table  table,
                       PS_Hint_Table   hints,
                       PS_Mask_Table   hint_masks,
                       PS_Mask_Table   counter_masks,
                       FT_Memory       memory )
  {
    FT_UInt   count;
    FT_Error  error;

    FT_UNUSED( counter_masks );


    count = hints->num_hints;

    /* allocate our tables */
    if ( FT_NEW_ARRAY( table->sort,  2 * count     ) ||
         FT_NEW_ARRAY( table->hints,     count     ) ||
         FT_NEW_ARRAY( table->zones, 2 * count + 1 ) )
      goto Exit;

    table->max_hints   = count;
    table->sort_global = table->sort + count;
    table->num_hints   = 0;
    table->num_zones   = 0;
    table->zone        = NULL;

    /* initialize the `table->hints' array */
    {
      PSH_Hint  write = table->hints;
      PS_Hint   read  = hints->hints;


      for ( ; count > 0; count--, write++, read++ )
      {
        write->org_pos = read->pos;
        write->org_len = read->len;
        write->flags   = read->flags;
      }
    }

    /* we now need to determine the initial `parent' stems; first  */
    /* activate the hints that are given by the initial hint masks */
    if ( hint_masks )
    {
      PS_Mask  mask = hint_masks->masks;


      count             = hint_masks->num_masks;
      table->hint_masks = hint_masks;

      for ( ; count > 0; count--, mask++ )
        psh_hint_table_record_mask( table, mask );
    }

    /* finally, do a linear parse in case some hints were left alone */
    if ( table->num_hints != table->max_hints )
    {
      FT_UInt  idx;


      FT_TRACE0(( "psh_hint_table_init: missing/incorrect hint masks\n" ));

      count = table->max_hints;
      for ( idx = 0; idx < count; idx++ )
        psh_hint_table_record( table, idx );
    }

  Exit:
    return error;
  }


  static void
  psh_hint_table_activate_mask( PSH_Hint_Table  table,
                                PS_Mask         hint_mask )
  {
    FT_Int    mask = 0, val = 0;
    FT_Byte*  cursor = hint_mask->bytes;
    FT_UInt   idx, limit, count;


    limit = hint_mask->num_bits;
    count = 0;

    psh_hint_table_deactivate( table );

    for ( idx = 0; idx < limit; idx++ )
    {
      if ( mask == 0 )
      {
        val  = *cursor++;
        mask = 0x80;
      }

      if ( val & mask )
      {
        PSH_Hint  hint = &table->hints[idx];


        if ( !psh_hint_is_active( hint ) )
        {
          FT_UInt     count2;

#if 0
          PSH_Hint*  sort = table->sort;
          PSH_Hint   hint2;


          for ( count2 = count; count2 > 0; count2--, sort++ )
          {
            hint2 = sort[0];
            if ( psh_hint_overlap( hint, hint2 ) )
              FT_TRACE0(( "psh_hint_table_activate_mask:"
                          " found overlapping hints\n" ))
          }
#else
          count2 = 0;
#endif

          if ( count2 == 0 )
          {
            psh_hint_activate( hint );
            if ( count < table->max_hints )
              table->sort[count++] = hint;
            else
              FT_TRACE0(( "psh_hint_tableactivate_mask:"
                          " too many active hints\n" ));
          }
        }
      }

      mask >>= 1;
    }
    table->num_hints = count;

    /* now, sort the hints; they are guaranteed to not overlap */
    /* so we can compare their "org_pos" field directly        */
    {
      FT_Int     i1, i2;
      PSH_Hint   hint1, hint2;
      PSH_Hint*  sort = table->sort;


      /* a simple bubble sort will do, since in 99% of cases, the hints */
      /* will be already sorted -- and the sort will be linear          */
      for ( i1 = 1; i1 < (FT_Int)count; i1++ )
      {
        hint1 = sort[i1];
        for ( i2 = i1 - 1; i2 >= 0; i2-- )
        {
          hint2 = sort[i2];

          if ( hint2->org_pos < hint1->org_pos )
            break;

          sort[i2 + 1] = hint2;
          sort[i2]     = hint1;
        }
      }
    }
  }


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****               HINTS GRID-FITTING AND OPTIMIZATION             *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

#if 1
  static FT_Pos
  psh_dimension_quantize_len( PSH_Dimension  dim,
                              FT_Pos         len,
                              FT_Bool        do_snapping )
  {
    if ( len <= 64 )
      len = 64;
    else
    {
      FT_Pos  delta = len - dim->stdw.widths[0].cur;


      if ( delta < 0 )
        delta = -delta;

      if ( delta < 40 )
      {
        len = dim->stdw.widths[0].cur;
        if ( len < 48 )
          len = 48;
      }

      if ( len < 3 * 64 )
      {
        delta = ( len & 63 );
        len  &= -64;

        if ( delta < 10 )
          len += delta;

        else if ( delta < 32 )
          len += 10;

        else if ( delta < 54 )
          len += 54;

        else
          len += delta;
      }
      else
        len = FT_PIX_ROUND( len );
    }

    if ( do_snapping )
      len = FT_PIX_ROUND( len );

    return  len;
  }
#endif /* 0 */


#ifdef DEBUG_HINTER

  static void
  ps_simple_scale( PSH_Hint_Table  table,
                   FT_Fixed        scale,
                   FT_Fixed        delta,
                   FT_Int          dimension )
  {
    FT_UInt  count;


    for ( count = 0; count < table->max_hints; count++ )
    {
      PSH_Hint  hint = table->hints + count;


      hint->cur_pos = FT_MulFix( hint->org_pos, scale ) + delta;
      hint->cur_len = FT_MulFix( hint->org_len, scale );

      if ( ps_debug_hint_func )
        ps_debug_hint_func( hint, dimension );
    }
  }

#endif /* DEBUG_HINTER */


  static FT_Fixed
  psh_hint_snap_stem_side_delta( FT_Fixed  pos,
                                 FT_Fixed  len )
  {
    FT_Fixed  delta1 = FT_PIX_ROUND( pos ) - pos;
    FT_Fixed  delta2 = FT_PIX_ROUND( pos + len ) - pos - len;


    if ( FT_ABS( delta1 ) <= FT_ABS( delta2 ) )
      return delta1;
    else
      return delta2;
  }


  static void
  psh_hint_align( PSH_Hint     hint,
                  PSH_Globals  globals,
                  FT_Int       dimension,
                  PSH_Glyph    glyph )
  {
    PSH_Dimension  dim   = &globals->dimension[dimension];
    FT_Fixed       scale = dim->scale_mult;
    FT_Fixed       delta = dim->scale_delta;


    if ( !psh_hint_is_fitted( hint ) )
    {
      FT_Pos  pos = FT_MulFix( hint->org_pos, scale ) + delta;
      FT_Pos  len = FT_MulFix( hint->org_len, scale );

      FT_Int            do_snapping;
      FT_Pos            fit_len;
      PSH_AlignmentRec  align;


      /* ignore stem alignments when requested through the hint flags */
      if ( ( dimension == 0 && !glyph->do_horz_hints ) ||
           ( dimension == 1 && !glyph->do_vert_hints ) )
      {
        hint->cur_pos = pos;
        hint->cur_len = len;

        psh_hint_set_fitted( hint );
        return;
      }

      /* perform stem snapping when requested - this is necessary
       * for monochrome and LCD hinting modes only
       */
      do_snapping = ( dimension == 0 && glyph->do_horz_snapping ) ||
                    ( dimension == 1 && glyph->do_vert_snapping );

      hint->cur_len = fit_len = len;

      /* check blue zones for horizontal stems */
      align.align     = PSH_BLUE_ALIGN_NONE;
      align.align_bot = align.align_top = 0;

      if ( dimension == 1 )
        psh_blues_snap_stem( &globals->blues,
                             ADD_INT( hint->org_pos, hint->org_len ),
                             hint->org_pos,
                             &align );

      switch ( align.align )
      {
      case PSH_BLUE_ALIGN_TOP:
        /* the top of the stem is aligned against a blue zone */
        hint->cur_pos = align.align_top - fit_len;
        break;

      case PSH_BLUE_ALIGN_BOT:
        /* the bottom of the stem is aligned against a blue zone */
        hint->cur_pos = align.align_bot;
        break;

      case PSH_BLUE_ALIGN_TOP | PSH_BLUE_ALIGN_BOT:
        /* both edges of the stem are aligned against blue zones */
        hint->cur_pos = align.align_bot;
        hint->cur_len = align.align_top - align.align_bot;
        break;

      default:
        {
          PSH_Hint  parent = hint->parent;


          if ( parent )
          {
            FT_Pos  par_org_center, par_cur_center;
            FT_Pos  cur_org_center, cur_delta;


            /* ensure that parent is already fitted */
            if ( !psh_hint_is_fitted( parent ) )
              psh_hint_align( parent, globals, dimension, glyph );

            /* keep original relation between hints, this is, use the */
            /* scaled distance between the centers of the hints to    */
            /* compute the new position                               */
            par_org_center = parent->org_pos + ( parent->org_len >> 1 );
            par_cur_center = parent->cur_pos + ( parent->cur_len >> 1 );
            cur_org_center = hint->org_pos   + ( hint->org_len   >> 1 );

            cur_delta = FT_MulFix( cur_org_center - par_org_center, scale );
            pos       = par_cur_center + cur_delta - ( len >> 1 );
          }

          hint->cur_pos = pos;
          hint->cur_len = fit_len;

          /* Stem adjustment tries to snap stem widths to standard
           * ones.  This is important to prevent unpleasant rounding
           * artefacts.
           */
          if ( glyph->do_stem_adjust )
          {
            if ( len <= 64 )
            {
              /* the stem is less than one pixel; we will center it
               * around the nearest pixel center
               */
              if ( len >= 32 )
              {
                /* This is a special case where we also widen the stem
                 * and align it to the pixel grid.
                 *
                 *   stem_center          = pos + (len/2)
                 *   nearest_pixel_center = FT_ROUND(stem_center-32)+32
                 *   new_pos              = nearest_pixel_center-32
                 *                        = FT_ROUND(stem_center-32)
                 *                        = FT_FLOOR(stem_center-32+32)
                 *                        = FT_FLOOR(stem_center)
                 *   new_len              = 64
                 */
                pos = FT_PIX_FLOOR( pos + ( len >> 1 ) );
                len = 64;
              }
              else if ( len > 0 )
              {
                /* This is a very small stem; we simply align it to the
                 * pixel grid, trying to find the minimum displacement.
                 *
                 * left               = pos
                 * right              = pos + len
                 * left_nearest_edge  = ROUND(pos)
                 * right_nearest_edge = ROUND(right)
                 *
                 * if ( ABS(left_nearest_edge - left) <=
                 *      ABS(right_nearest_edge - right) )
                 *    new_pos = left
                 * else
                 *    new_pos = right
                 */
                FT_Pos  left_nearest  = FT_PIX_ROUND( pos );
                FT_Pos  right_nearest = FT_PIX_ROUND( pos + len );
                FT_Pos  left_disp     = left_nearest - pos;
                FT_Pos  right_disp    = right_nearest - ( pos + len );


                if ( left_disp < 0 )
                  left_disp = -left_disp;
                if ( right_disp < 0 )
                  right_disp = -right_disp;
                if ( left_disp <= right_disp )
                  pos = left_nearest;
                else
                  pos = right_nearest;
              }
              else
              {
                /* this is a ghost stem; we simply round it */
                pos = FT_PIX_ROUND( pos );
              }
            }
            else
            {
              len = psh_dimension_quantize_len( dim, len, 0 );
            }
          }

          /* now that we have a good hinted stem width, try to position */
          /* the stem along a pixel grid integer coordinate             */
          hint->cur_pos = pos + psh_hint_snap_stem_side_delta( pos, len );
          hint->cur_len = len;
        }
      }

      if ( do_snapping )
      {
        pos = hint->cur_pos;
        len = hint->cur_len;

        if ( len < 64 )
          len = 64;
        else
          len = FT_PIX_ROUND( len );

        switch ( align.align )
        {
          case PSH_BLUE_ALIGN_TOP:
            hint->cur_pos = align.align_top - len;
            hint->cur_len = len;
            break;

          case PSH_BLUE_ALIGN_BOT:
            hint->cur_len = len;
            break;

          case PSH_BLUE_ALIGN_BOT | PSH_BLUE_ALIGN_TOP:
            /* don't touch */
            break;


          default:
            hint->cur_len = len;
            if ( len & 64 )
              pos = FT_PIX_FLOOR( pos + ( len >> 1 ) ) + 32;
            else
              pos = FT_PIX_ROUND( pos + ( len >> 1 ) );

            hint->cur_pos = pos - ( len >> 1 );
            hint->cur_len = len;
        }
      }

      psh_hint_set_fitted( hint );

#ifdef DEBUG_HINTER
      if ( ps_debug_hint_func )
        ps_debug_hint_func( hint, dimension );
#endif
    }
  }


#if 0  /* not used for now, experimental */

 /*
  * A variant to perform "light" hinting (i.e. FT_RENDER_MODE_LIGHT)
  * of stems
  */
  static void
  psh_hint_align_light( PSH_Hint     hint,
                        PSH_Globals  globals,
                        FT_Int       dimension,
                        PSH_Glyph    glyph )
  {
    PSH_Dimension  dim   = &globals->dimension[dimension];
    FT_Fixed       scale = dim->scale_mult;
    FT_Fixed       delta = dim->scale_delta;


    if ( !psh_hint_is_fitted( hint ) )
    {
      FT_Pos  pos = FT_MulFix( hint->org_pos, scale ) + delta;
      FT_Pos  len = FT_MulFix( hint->org_len, scale );

      FT_Pos  fit_len;

      PSH_AlignmentRec  align;


      /* ignore stem alignments when requested through the hint flags */
      if ( ( dimension == 0 && !glyph->do_horz_hints ) ||
           ( dimension == 1 && !glyph->do_vert_hints ) )
      {
        hint->cur_pos = pos;
        hint->cur_len = len;

        psh_hint_set_fitted( hint );
        return;
      }

      fit_len = len;

      hint->cur_len = fit_len;

      /* check blue zones for horizontal stems */
      align.align = PSH_BLUE_ALIGN_NONE;
      align.align_bot = align.align_top = 0;

      if ( dimension == 1 )
        psh_blues_snap_stem( &globals->blues,
                             ADD_INT( hint->org_pos, hint->org_len ),
                             hint->org_pos,
                             &align );

      switch ( align.align )
      {
      case PSH_BLUE_ALIGN_TOP:
        /* the top of the stem is aligned against a blue zone */
        hint->cur_pos = align.align_top - fit_len;
        break;

      case PSH_BLUE_ALIGN_BOT:
        /* the bottom of the stem is aligned against a blue zone */
        hint->cur_pos = align.align_bot;
        break;

      case PSH_BLUE_ALIGN_TOP | PSH_BLUE_ALIGN_BOT:
        /* both edges of the stem are aligned against blue zones */
        hint->cur_pos = align.align_bot;
        hint->cur_len = align.align_top - align.align_bot;
        break;

      default:
        {
          PSH_Hint  parent = hint->parent;


          if ( parent )
          {
            FT_Pos  par_org_center, par_cur_center;
            FT_Pos  cur_org_center, cur_delta;


            /* ensure that parent is already fitted */
            if ( !psh_hint_is_fitted( parent ) )
              psh_hint_align_light( parent, globals, dimension, glyph );

            par_org_center = parent->org_pos + ( parent->org_len / 2 );
            par_cur_center = parent->cur_pos + ( parent->cur_len / 2 );
            cur_org_center = hint->org_pos   + ( hint->org_len   / 2 );

            cur_delta = FT_MulFix( cur_org_center - par_org_center, scale );
            pos       = par_cur_center + cur_delta - ( len >> 1 );
          }

          /* Stems less than one pixel wide are easy -- we want to
           * make them as dark as possible, so they must fall within
           * one pixel.  If the stem is split between two pixels
           * then snap the edge that is nearer to the pixel boundary
           * to the pixel boundary.
           */
          if ( len <= 64 )
          {
            if ( ( pos + len + 63 ) / 64  != pos / 64 + 1 )
              pos += psh_hint_snap_stem_side_delta ( pos, len );
          }

          /* Position stems other to minimize the amount of mid-grays.
           * There are, in general, two positions that do this,
           * illustrated as A) and B) below.
           *
           *   +                   +                   +                   +
           *
           * A)             |--------------------------------|
           * B)   |--------------------------------|
           * C)       |--------------------------------|
           *
           * Position A) (split the excess stem equally) should be better
           * for stems of width N + f where f < 0.5.
           *
           * Position B) (split the deficiency equally) should be better
           * for stems of width N + f where f > 0.5.
           *
           * It turns out though that minimizing the total number of lit
           * pixels is also important, so position C), with one edge
           * aligned with a pixel boundary is actually preferable
           * to A).  There are also more possible positions for C) than
           * for A) or B), so it involves less distortion of the overall
           * character shape.
           */
          else /* len > 64 */
          {
            FT_Fixed  frac_len = len & 63;
            FT_Fixed  center = pos + ( len >> 1 );
            FT_Fixed  delta_a, delta_b;


            if ( ( len / 64 ) & 1 )
            {
              delta_a = FT_PIX_FLOOR( center ) + 32 - center;
              delta_b = FT_PIX_ROUND( center ) - center;
            }
            else
            {
              delta_a = FT_PIX_ROUND( center ) - center;
              delta_b = FT_PIX_FLOOR( center ) + 32 - center;
            }

            /* We choose between B) and C) above based on the amount
             * of fractional stem width; for small amounts, choose
             * C) always, for large amounts, B) always, and inbetween,
             * pick whichever one involves less stem movement.
             */
            if ( frac_len < 32 )
            {
              pos += psh_hint_snap_stem_side_delta ( pos, len );
            }
            else if ( frac_len < 48 )
            {
              FT_Fixed  side_delta = psh_hint_snap_stem_side_delta ( pos,
                                                                     len );

              if ( FT_ABS( side_delta ) < FT_ABS( delta_b ) )
                pos += side_delta;
              else
                pos += delta_b;
            }
            else
            {
              pos += delta_b;
            }
          }

          hint->cur_pos = pos;
        }
      }  /* switch */

      psh_hint_set_fitted( hint );

#ifdef DEBUG_HINTER
      if ( ps_debug_hint_func )
        ps_debug_hint_func( hint, dimension );
#endif
    }
  }

#endif /* 0 */


  static void
  psh_hint_table_align_hints( PSH_Hint_Table  table,
                              PSH_Globals     globals,
                              FT_Int          dimension,
                              PSH_Glyph       glyph )
  {
    PSH_Hint       hint;
    FT_UInt        count;

#ifdef DEBUG_HINTER

    PSH_Dimension  dim   = &globals->dimension[dimension];
    FT_Fixed       scale = dim->scale_mult;
    FT_Fixed       delta = dim->scale_delta;


    if ( ps_debug_no_vert_hints && dimension == 0 )
    {
      ps_simple_scale( table, scale, delta, dimension );
      return;
    }

    if ( ps_debug_no_horz_hints && dimension == 1 )
    {
      ps_simple_scale( table, scale, delta, dimension );
      return;
    }

#endif /* DEBUG_HINTER*/

    hint  = table->hints;
    count = table->max_hints;

    for ( ; count > 0; count--, hint++ )
      psh_hint_align( hint, globals, dimension, glyph );
  }


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                POINTS INTERPOLATION ROUTINES                  *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

#define xxDEBUG_ZONES


#ifdef DEBUG_ZONES

#include FT_CONFIG_STANDARD_LIBRARY_H

  static void
  psh_print_zone( PSH_Zone  zone )
  {
    printf( "zone [scale,delta,min,max] = [%.5f,%.2f,%d,%d]\n",
             zone->scale / 65536.0,
             zone->delta / 64.0,
             zone->min,
             zone->max );
  }

#endif /* DEBUG_ZONES */


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                    HINTER GLYPH MANAGEMENT                    *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

#define  psh_corner_is_flat      ft_corner_is_flat
#define  psh_corner_orientation  ft_corner_orientation


#ifdef COMPUTE_INFLEXS

  /* compute all inflex points in a given glyph */
  static void
  psh_glyph_compute_inflections( PSH_Glyph  glyph )
  {
    FT_UInt  n;


    for ( n = 0; n < glyph->num_contours; n++ )
    {
      PSH_Point  first, start, end, before, after;
      FT_Pos     in_x, in_y, out_x, out_y;
      FT_Int     orient_prev, orient_cur;
      FT_Int     finished = 0;


      /* we need at least 4 points to create an inflection point */
      if ( glyph->contours[n].count < 4 )
        continue;

      /* compute first segment in contour */
      first = glyph->contours[n].start;

      start = end = first;
      do
      {
        end = end->next;
        if ( end == first )
          goto Skip;

        in_x = end->org_u - start->org_u;
        in_y = end->org_v - start->org_v;

      } while ( in_x == 0 && in_y == 0 );

      /* extend the segment start whenever possible */
      before = start;
      do
      {
        do
        {
          start  = before;
          before = before->prev;
          if ( before == first )
            goto Skip;

          out_x = start->org_u - before->org_u;
          out_y = start->org_v - before->org_v;

        } while ( out_x == 0 && out_y == 0 );

        orient_prev = psh_corner_orientation( in_x, in_y, out_x, out_y );

      } while ( orient_prev == 0 );

      first = start;
      in_x  = out_x;
      in_y  = out_y;

      /* now, process all segments in the contour */
      do
      {
        /* first, extend current segment's end whenever possible */
        after = end;
        do
        {
          do
          {
            end   = after;
            after = after->next;
            if ( after == first )
              finished = 1;

            out_x = after->org_u - end->org_u;
            out_y = after->org_v - end->org_v;

          } while ( out_x == 0 && out_y == 0 );

          orient_cur = psh_corner_orientation( in_x, in_y, out_x, out_y );

        } while ( orient_cur == 0 );

        if ( ( orient_cur ^ orient_prev ) < 0 )
        {
          do
          {
            psh_point_set_inflex( start );
            start = start->next;
          }
          while ( start != end );

          psh_point_set_inflex( start );
        }

        start       = end;
        end         = after;
        orient_prev = orient_cur;
        in_x        = out_x;
        in_y        = out_y;

      } while ( !finished );

    Skip:
      ;
    }
  }

#endif /* COMPUTE_INFLEXS */


  static void
  psh_glyph_done( PSH_Glyph  glyph )
  {
    FT_Memory  memory = glyph->memory;


    psh_hint_table_done( &glyph->hint_tables[1], memory );
    psh_hint_table_done( &glyph->hint_tables[0], memory );

    FT_FREE( glyph->points );
    FT_FREE( glyph->contours );

    glyph->num_points   = 0;
    glyph->num_contours = 0;

    glyph->memory = NULL;
  }


  static int
  psh_compute_dir( FT_Pos  dx,
                   FT_Pos  dy )
  {
    FT_Pos  ax, ay;
    int     result = PSH_DIR_NONE;


    ax = FT_ABS( dx );
    ay = FT_ABS( dy );

    if ( ay * 12 < ax )
    {
      /* |dy| <<< |dx|  means a near-horizontal segment */
      result = ( dx >= 0 ) ? PSH_DIR_RIGHT : PSH_DIR_LEFT;
    }
    else if ( ax * 12 < ay )
    {
      /* |dx| <<< |dy|  means a near-vertical segment */
      result = ( dy >= 0 ) ? PSH_DIR_UP : PSH_DIR_DOWN;
    }

    return result;
  }


  /* load outline point coordinates into hinter glyph */
  static void
  psh_glyph_load_points( PSH_Glyph  glyph,
                         FT_Int     dimension )
  {
    FT_Vector*  vec   = glyph->outline->points;
    PSH_Point   point = glyph->points;
    FT_UInt     count = glyph->num_points;


    for ( ; count > 0; count--, point++, vec++ )
    {
      point->flags2 = 0;
      point->hint   = NULL;
      if ( dimension == 0 )
      {
        point->org_u = vec->x;
        point->org_v = vec->y;
      }
      else
      {
        point->org_u = vec->y;
        point->org_v = vec->x;
      }

#ifdef DEBUG_HINTER
      point->org_x = vec->x;
      point->org_y = vec->y;
#endif

    }
  }


  /* save hinted point coordinates back to outline */
  static void
  psh_glyph_save_points( PSH_Glyph  glyph,
                         FT_Int     dimension )
  {
    FT_UInt     n;
    PSH_Point   point = glyph->points;
    FT_Vector*  vec   = glyph->outline->points;
    char*       tags  = glyph->outline->tags;


    for ( n = 0; n < glyph->num_points; n++ )
    {
      if ( dimension == 0 )
        vec[n].x = point->cur_u;
      else
        vec[n].y = point->cur_u;

      if ( psh_point_is_strong( point ) )
        tags[n] |= (char)( ( dimension == 0 ) ? 32 : 64 );

#ifdef DEBUG_HINTER

      if ( dimension == 0 )
      {
        point->cur_x   = point->cur_u;
        point->flags_x = point->flags2 | point->flags;
      }
      else
      {
        point->cur_y   = point->cur_u;
        point->flags_y = point->flags2 | point->flags;
      }

#endif

      point++;
    }
  }


  static FT_Error
  psh_glyph_init( PSH_Glyph    glyph,
                  FT_Outline*  outline,
                  PS_Hints     ps_hints,
                  PSH_Globals  globals )
  {
    FT_Error   error;
    FT_Memory  memory;


    /* clear all fields */
    FT_ZERO( glyph );

    memory = glyph->memory = globals->memory;

    /* allocate and setup points + contours arrays */
    if ( FT_NEW_ARRAY( glyph->points,   outline->n_points   ) ||
         FT_NEW_ARRAY( glyph->contours, outline->n_contours ) )
      goto Exit;

    glyph->num_points   = (FT_UInt)outline->n_points;
    glyph->num_contours = (FT_UInt)outline->n_contours;

    {
      FT_UInt      first = 0, next, n;
      PSH_Point    points  = glyph->points;
      PSH_Contour  contour = glyph->contours;


      for ( n = 0; n < glyph->num_contours; n++ )
      {
        FT_UInt    count;
        PSH_Point  point;


        next  = (FT_UInt)outline->contours[n] + 1;
        count = next - first;

        contour->start = points + first;
        contour->count = count;

        if ( count > 0 )
        {
          point = points + first;

          point->prev    = points + next - 1;
          point->contour = contour;

          for ( ; count > 1; count-- )
          {
            point[0].next = point + 1;
            point[1].prev = point;
            point++;
            point->contour = contour;
          }
          point->next = points + first;
        }

        contour++;
        first = next;
      }
    }

    {
      PSH_Point   points = glyph->points;
      PSH_Point   point  = points;
      FT_Vector*  vec    = outline->points;
      FT_UInt     n;


      for ( n = 0; n < glyph->num_points; n++, point++ )
      {
        FT_Int  n_prev = (FT_Int)( point->prev - points );
        FT_Int  n_next = (FT_Int)( point->next - points );
        FT_Pos  dxi, dyi, dxo, dyo;


        if ( !( outline->tags[n] & FT_CURVE_TAG_ON ) )
          point->flags = PSH_POINT_OFF;

        dxi = vec[n].x - vec[n_prev].x;
        dyi = vec[n].y - vec[n_prev].y;

        point->dir_in = (FT_Char)psh_compute_dir( dxi, dyi );

        dxo = vec[n_next].x - vec[n].x;
        dyo = vec[n_next].y - vec[n].y;

        point->dir_out = (FT_Char)psh_compute_dir( dxo, dyo );

        /* detect smooth points */
        if ( point->flags & PSH_POINT_OFF )
          point->flags |= PSH_POINT_SMOOTH;

        else if ( point->dir_in == point->dir_out )
        {
          if ( point->dir_out != PSH_DIR_NONE           ||
               psh_corner_is_flat( dxi, dyi, dxo, dyo ) )
            point->flags |= PSH_POINT_SMOOTH;
        }
      }
    }

    glyph->outline = outline;
    glyph->globals = globals;

#ifdef COMPUTE_INFLEXS
    psh_glyph_load_points( glyph, 0 );
    psh_glyph_compute_inflections( glyph );
#endif /* COMPUTE_INFLEXS */

    /* now deal with hints tables */
    error = psh_hint_table_init( &glyph->hint_tables [0],
                                 &ps_hints->dimension[0].hints,
                                 &ps_hints->dimension[0].masks,
                                 &ps_hints->dimension[0].counters,
                                 memory );
    if ( error )
      goto Exit;

    error = psh_hint_table_init( &glyph->hint_tables [1],
                                 &ps_hints->dimension[1].hints,
                                 &ps_hints->dimension[1].masks,
                                 &ps_hints->dimension[1].counters,
                                 memory );
    if ( error )
      goto Exit;

  Exit:
    return error;
  }


  /* compute all extrema in a glyph for a given dimension */
  static void
  psh_glyph_compute_extrema( PSH_Glyph  glyph )
  {
    FT_UInt  n;


    /* first of all, compute all local extrema */
    for ( n = 0; n < glyph->num_contours; n++ )
    {
      PSH_Point  first = glyph->contours[n].start;
      PSH_Point  point, before, after;


      if ( glyph->contours[n].count == 0 )
        continue;

      point  = first;
      before = point;

      do
      {
        before = before->prev;
        if ( before == first )
          goto Skip;

      } while ( before->org_u == point->org_u );

      first = point = before->next;

      for (;;)
      {
        after = point;
        do
        {
          after = after->next;
          if ( after == first )
            goto Next;

        } while ( after->org_u == point->org_u );

        if ( before->org_u < point->org_u )
        {
          if ( after->org_u < point->org_u )
          {
            /* local maximum */
            goto Extremum;
          }
        }
        else /* before->org_u > point->org_u */
        {
          if ( after->org_u > point->org_u )
          {
            /* local minimum */
          Extremum:
            do
            {
              psh_point_set_extremum( point );
              point = point->next;

            } while ( point != after );
          }
        }

        before = after->prev;
        point  = after;

      } /* for  */

    Next:
      ;
    }

    /* for each extremum, determine its direction along the */
    /* orthogonal axis                                      */
    for ( n = 0; n < glyph->num_points; n++ )
    {
      PSH_Point  point, before, after;


      point  = &glyph->points[n];
      before = point;
      after  = point;

      if ( psh_point_is_extremum( point ) )
      {
        do
        {
          before = before->prev;
          if ( before == point )
            goto Skip;

        } while ( before->org_v == point->org_v );

        do
        {
          after = after->next;
          if ( after == point )
            goto Skip;

        } while ( after->org_v == point->org_v );
      }

      if ( before->org_v < point->org_v &&
           after->org_v  > point->org_v )
      {
        psh_point_set_positive( point );
      }
      else if ( before->org_v > point->org_v &&
                after->org_v  < point->org_v )
      {
        psh_point_set_negative( point );
      }

    Skip:
      ;
    }
  }


  /* major_dir is the direction for points on the bottom/left of the stem; */
  /* Points on the top/right of the stem will have a direction of          */
  /* -major_dir.                                                           */

  static void
  psh_hint_table_find_strong_points( PSH_Hint_Table  table,
                                     PSH_Point       point,
                                     FT_UInt         count,
                                     FT_Int          threshold,
                                     FT_Int          major_dir )
  {
    PSH_Hint*  sort      = table->sort;
    FT_UInt    num_hints = table->num_hints;


    for ( ; count > 0; count--, point++ )
    {
      FT_Int  point_dir = 0;
      FT_Pos  org_u     = point->org_u;


      if ( psh_point_is_strong( point ) )
        continue;

      if ( PSH_DIR_COMPARE( point->dir_in, major_dir ) )
        point_dir = point->dir_in;

      else if ( PSH_DIR_COMPARE( point->dir_out, major_dir ) )
        point_dir = point->dir_out;

      if ( point_dir )
      {
        if ( point_dir == major_dir )
        {
          FT_UInt  nn;


          for ( nn = 0; nn < num_hints; nn++ )
          {
            PSH_Hint  hint = sort[nn];
            FT_Pos    d    = org_u - hint->org_pos;


            if ( d < threshold && -d < threshold )
            {
              psh_point_set_strong( point );
              point->flags2 |= PSH_POINT_EDGE_MIN;
              point->hint    = hint;
              break;
            }
          }
        }
        else if ( point_dir == -major_dir )
        {
          FT_UInt  nn;


          for ( nn = 0; nn < num_hints; nn++ )
          {
            PSH_Hint  hint = sort[nn];
            FT_Pos    d    = org_u - hint->org_pos - hint->org_len;


            if ( d < threshold && -d < threshold )
            {
              psh_point_set_strong( point );
              point->flags2 |= PSH_POINT_EDGE_MAX;
              point->hint    = hint;
              break;
            }
          }
        }
      }

#if 1
      else if ( psh_point_is_extremum( point ) )
      {
        /* treat extrema as special cases for stem edge alignment */
        FT_UInt  nn, min_flag, max_flag;


        if ( major_dir == PSH_DIR_HORIZONTAL )
        {
          min_flag = PSH_POINT_POSITIVE;
          max_flag = PSH_POINT_NEGATIVE;
        }
        else
        {
          min_flag = PSH_POINT_NEGATIVE;
          max_flag = PSH_POINT_POSITIVE;
        }

        if ( point->flags2 & min_flag )
        {
          for ( nn = 0; nn < num_hints; nn++ )
          {
            PSH_Hint  hint = sort[nn];
            FT_Pos    d    = org_u - hint->org_pos;


            if ( d < threshold && -d < threshold )
            {
              point->flags2 |= PSH_POINT_EDGE_MIN;
              point->hint    = hint;
              psh_point_set_strong( point );
              break;
            }
          }
        }
        else if ( point->flags2 & max_flag )
        {
          for ( nn = 0; nn < num_hints; nn++ )
          {
            PSH_Hint  hint = sort[nn];
            FT_Pos    d    = org_u - hint->org_pos - hint->org_len;


            if ( d < threshold && -d < threshold )
            {
              point->flags2 |= PSH_POINT_EDGE_MAX;
              point->hint    = hint;
              psh_point_set_strong( point );
              break;
            }
          }
        }

        if ( !point->hint )
        {
          for ( nn = 0; nn < num_hints; nn++ )
          {
            PSH_Hint  hint = sort[nn];


            if ( org_u >=          hint->org_pos                  &&
                 org_u <= ADD_INT( hint->org_pos, hint->org_len ) )
            {
              point->hint = hint;
              break;
            }
          }
        }
      }

#endif /* 1 */
    }
  }


  /* the accepted shift for strong points in fractional pixels */
#define PSH_STRONG_THRESHOLD  32

  /* the maximum shift value in font units */
#define PSH_STRONG_THRESHOLD_MAXIMUM  30


  /* find strong points in a glyph */
  static void
  psh_glyph_find_strong_points( PSH_Glyph  glyph,
                                FT_Int     dimension )
  {
    /* a point is `strong' if it is located on a stem edge and       */
    /* has an `in' or `out' tangent parallel to the hint's direction */

    PSH_Hint_Table  table     = &glyph->hint_tables[dimension];
    PS_Mask         mask      = table->hint_masks->masks;
    FT_UInt         num_masks = table->hint_masks->num_masks;
    FT_UInt         first     = 0;
    FT_Int          major_dir = ( dimension == 0 ) ? PSH_DIR_VERTICAL
                                                   : PSH_DIR_HORIZONTAL;
    PSH_Dimension   dim       = &glyph->globals->dimension[dimension];
    FT_Fixed        scale     = dim->scale_mult;
    FT_Int          threshold;


    threshold = (FT_Int)FT_DivFix( PSH_STRONG_THRESHOLD, scale );
    if ( threshold > PSH_STRONG_THRESHOLD_MAXIMUM )
      threshold = PSH_STRONG_THRESHOLD_MAXIMUM;

    /* process secondary hints to `selected' points */
    if ( num_masks > 1 && glyph->num_points > 0 )
    {
      /* the `endchar' op can reduce the number of points */
      first = mask->end_point > glyph->num_points
                ? glyph->num_points
                : mask->end_point;
      mask++;
      for ( ; num_masks > 1; num_masks--, mask++ )
      {
        FT_UInt  next = FT_MIN( mask->end_point, glyph->num_points );


        if ( next > first )
        {
          FT_UInt    count = next - first;
          PSH_Point  point = glyph->points + first;


          psh_hint_table_activate_mask( table, mask );

          psh_hint_table_find_strong_points( table, point, count,
                                             threshold, major_dir );
        }
        first = next;
      }
    }

    /* process primary hints for all points */
    if ( num_masks == 1 )
    {
      FT_UInt    count = glyph->num_points;
      PSH_Point  point = glyph->points;


      psh_hint_table_activate_mask( table, table->hint_masks->masks );

      psh_hint_table_find_strong_points( table, point, count,
                                         threshold, major_dir );
    }

    /* now, certain points may have been attached to a hint and */
    /* not marked as strong; update their flags then            */
    {
      FT_UInt    count = glyph->num_points;
      PSH_Point  point = glyph->points;


      for ( ; count > 0; count--, point++ )
        if ( point->hint && !psh_point_is_strong( point ) )
          psh_point_set_strong( point );
    }
  }


  /* find points in a glyph which are in a blue zone and have `in' or */
  /* `out' tangents parallel to the horizontal axis                   */
  static void
  psh_glyph_find_blue_points( PSH_Blues  blues,
                              PSH_Glyph  glyph )
  {
    PSH_Blue_Table  table;
    PSH_Blue_Zone   zone;
    FT_UInt         glyph_count = glyph->num_points;
    FT_UInt         blue_count;
    PSH_Point       point = glyph->points;


    for ( ; glyph_count > 0; glyph_count--, point++ )
    {
      FT_Pos  y;


      /* check tangents */
      if ( !PSH_DIR_COMPARE( point->dir_in,  PSH_DIR_HORIZONTAL ) &&
           !PSH_DIR_COMPARE( point->dir_out, PSH_DIR_HORIZONTAL ) )
        continue;

      /* skip strong points */
      if ( psh_point_is_strong( point ) )
        continue;

      y = point->org_u;

      /* look up top zones */
      table      = &blues->normal_top;
      blue_count = table->count;
      zone       = table->zones;

      for ( ; blue_count > 0; blue_count--, zone++ )
      {
        FT_Pos  delta = y - zone->org_bottom;


        if ( delta < -blues->blue_fuzz )
          break;

        if ( y <= zone->org_top + blues->blue_fuzz )
          if ( blues->no_overshoots || delta <= blues->blue_threshold )
          {
            point->cur_u = zone->cur_bottom;
            psh_point_set_strong( point );
            psh_point_set_fitted( point );
          }
      }

      /* look up bottom zones */
      table      = &blues->normal_bottom;
      blue_count = table->count;
      zone       = table->zones + blue_count - 1;

      for ( ; blue_count > 0; blue_count--, zone-- )
      {
        FT_Pos  delta = zone->org_top - y;


        if ( delta < -blues->blue_fuzz )
          break;

        if ( y >= zone->org_bottom - blues->blue_fuzz )
          if ( blues->no_overshoots || delta < blues->blue_threshold )
          {
            point->cur_u = zone->cur_top;
            psh_point_set_strong( point );
            psh_point_set_fitted( point );
          }
      }
    }
  }


  /* interpolate strong points with the help of hinted coordinates */
  static void
  psh_glyph_interpolate_strong_points( PSH_Glyph  glyph,
                                       FT_Int     dimension )
  {
    PSH_Dimension  dim   = &glyph->globals->dimension[dimension];
    FT_Fixed       scale = dim->scale_mult;

    FT_UInt        count = glyph->num_points;
    PSH_Point      point = glyph->points;


    for ( ; count > 0; count--, point++ )
    {
      PSH_Hint  hint = point->hint;


      if ( hint )
      {
        FT_Pos  delta;


        if ( psh_point_is_edge_min( point ) )
          point->cur_u = hint->cur_pos;

        else if ( psh_point_is_edge_max( point ) )
          point->cur_u = hint->cur_pos + hint->cur_len;

        else
        {
          delta = point->org_u - hint->org_pos;

          if ( delta <= 0 )
            point->cur_u = hint->cur_pos + FT_MulFix( delta, scale );

          else if ( delta >= hint->org_len )
            point->cur_u = hint->cur_pos + hint->cur_len +
                             FT_MulFix( delta - hint->org_len, scale );

          else /* hint->org_len > 0 */
            point->cur_u = hint->cur_pos +
                             FT_MulDiv( delta, hint->cur_len,
                                        hint->org_len );
        }
        psh_point_set_fitted( point );
      }
    }
  }


#define  PSH_MAX_STRONG_INTERNAL  16

  static void
  psh_glyph_interpolate_normal_points( PSH_Glyph  glyph,
                                       FT_Int     dimension )
  {

#if 1
    /* first technique: a point is strong if it is a local extremum */

    PSH_Dimension  dim    = &glyph->globals->dimension[dimension];
    FT_Fixed       scale  = dim->scale_mult;
    FT_Memory      memory = glyph->memory;

    PSH_Point*     strongs     = NULL;
    PSH_Point      strongs_0[PSH_MAX_STRONG_INTERNAL];
    FT_UInt        num_strongs = 0;

    PSH_Point      points = glyph->points;
    PSH_Point      points_end = points + glyph->num_points;
    PSH_Point      point;


    /* first count the number of strong points */
    for ( point = points; point < points_end; point++ )
    {
      if ( psh_point_is_strong( point ) )
        num_strongs++;
    }

    if ( num_strongs == 0 )  /* nothing to do here */
      return;

    /* allocate an array to store a list of points, */
    /* stored in increasing org_u order             */
    if ( num_strongs <= PSH_MAX_STRONG_INTERNAL )
      strongs = strongs_0;
    else
    {
      FT_Error  error;


      if ( FT_NEW_ARRAY( strongs, num_strongs ) )
        return;
    }

    num_strongs = 0;
    for ( point = points; point < points_end; point++ )
    {
      PSH_Point*  insert;


      if ( !psh_point_is_strong( point ) )
        continue;

      for ( insert = strongs + num_strongs; insert > strongs; insert-- )
      {
        if ( insert[-1]->org_u <= point->org_u )
          break;

        insert[0] = insert[-1];
      }
      insert[0] = point;
      num_strongs++;
    }

    /* now try to interpolate all normal points */
    for ( point = points; point < points_end; point++ )
    {
      if ( psh_point_is_strong( point ) )
        continue;

      /* sometimes, some local extrema are smooth points */
      if ( psh_point_is_smooth( point ) )
      {
        if ( point->dir_in == PSH_DIR_NONE   ||
             point->dir_in != point->dir_out )
          continue;

        if ( !psh_point_is_extremum( point ) &&
             !psh_point_is_inflex( point )   )
          continue;

        point->flags &= ~PSH_POINT_SMOOTH;
      }

      /* find best enclosing point coordinates then interpolate */
      {
        PSH_Point   before, after;
        FT_UInt     nn;


        for ( nn = 0; nn < num_strongs; nn++ )
          if ( strongs[nn]->org_u > point->org_u )
            break;

        if ( nn == 0 )  /* point before the first strong point */
        {
          after = strongs[0];

          point->cur_u = after->cur_u +
                           FT_MulFix( point->org_u - after->org_u,
                                      scale );
        }
        else
        {
          before = strongs[nn - 1];

          for ( nn = num_strongs; nn > 0; nn-- )
            if ( strongs[nn - 1]->org_u < point->org_u )
              break;

          if ( nn == num_strongs )  /* point is after last strong point */
          {
            before = strongs[nn - 1];

            point->cur_u = before->cur_u +
                             FT_MulFix( point->org_u - before->org_u,
                                        scale );
          }
          else
          {
            FT_Pos  u;


            after = strongs[nn];

            /* now interpolate point between before and after */
            u = point->org_u;

            if ( u == before->org_u )
              point->cur_u = before->cur_u;

            else if ( u == after->org_u )
              point->cur_u = after->cur_u;

            else
              point->cur_u = before->cur_u +
                               FT_MulDiv( u - before->org_u,
                                          after->cur_u - before->cur_u,
                                          after->org_u - before->org_u );
          }
        }
        psh_point_set_fitted( point );
      }
    }

    if ( strongs != strongs_0 )
      FT_FREE( strongs );

#endif /* 1 */

  }


  /* interpolate other points */
  static void
  psh_glyph_interpolate_other_points( PSH_Glyph  glyph,
                                      FT_Int     dimension )
  {
    PSH_Dimension  dim          = &glyph->globals->dimension[dimension];
    FT_Fixed       scale        = dim->scale_mult;
    FT_Fixed       delta        = dim->scale_delta;
    PSH_Contour    contour      = glyph->contours;
    FT_UInt        num_contours = glyph->num_contours;


    for ( ; num_contours > 0; num_contours--, contour++ )
    {
      PSH_Point  start = contour->start;
      PSH_Point  first, next, point;
      FT_UInt    fit_count;


      /* count the number of strong points in this contour */
      next      = start + contour->count;
      fit_count = 0;
      first     = NULL;

      for ( point = start; point < next; point++ )
        if ( psh_point_is_fitted( point ) )
        {
          if ( !first )
            first = point;

          fit_count++;
        }

      /* if there are less than 2 fitted points in the contour, we */
      /* simply scale and eventually translate the contour points  */
      if ( fit_count < 2 )
      {
        if ( fit_count == 1 )
          delta = first->cur_u - FT_MulFix( first->org_u, scale );

        for ( point = start; point < next; point++ )
          if ( point != first )
            point->cur_u = FT_MulFix( point->org_u, scale ) + delta;

        goto Next_Contour;
      }

      /* there are more than 2 strong points in this contour; we */
      /* need to interpolate weak points between them            */
      start = first;
      do
      {
        /* skip consecutive fitted points */
        for (;;)
        {
          next = first->next;
          if ( next == start )
            goto Next_Contour;

          if ( !psh_point_is_fitted( next ) )
            break;

          first = next;
        }

        /* find next fitted point after unfitted one */
        for (;;)
        {
          next = next->next;
          if ( psh_point_is_fitted( next ) )
            break;
        }

        /* now interpolate between them */
        {
          FT_Pos    org_a, org_ab, cur_a, cur_ab;
          FT_Pos    org_c, org_ac, cur_c;
          FT_Fixed  scale_ab;


          if ( first->org_u <= next->org_u )
          {
            org_a  = first->org_u;
            cur_a  = first->cur_u;
            org_ab = next->org_u - org_a;
            cur_ab = next->cur_u - cur_a;
          }
          else
          {
            org_a  = next->org_u;
            cur_a  = next->cur_u;
            org_ab = first->org_u - org_a;
            cur_ab = first->cur_u - cur_a;
          }

          scale_ab = 0x10000L;
          if ( org_ab > 0 )
            scale_ab = FT_DivFix( cur_ab, org_ab );

          point = first->next;
          do
          {
            org_c  = point->org_u;
            org_ac = org_c - org_a;

            if ( org_ac <= 0 )
            {
              /* on the left of the interpolation zone */
              cur_c = cur_a + FT_MulFix( org_ac, scale );
            }
            else if ( org_ac >= org_ab )
            {
              /* on the right on the interpolation zone */
              cur_c = cur_a + cur_ab + FT_MulFix( org_ac - org_ab, scale );
            }
            else
            {
              /* within the interpolation zone */
              cur_c = cur_a + FT_MulFix( org_ac, scale_ab );
            }

            point->cur_u = cur_c;

            point = point->next;

          } while ( point != next );
        }

        /* keep going until all points in the contours have been processed */
        first = next;

      } while ( first != start );

    Next_Contour:
      ;
    }
  }


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                     HIGH-LEVEL INTERFACE                      *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  FT_Error
  ps_hints_apply( PS_Hints        ps_hints,
                  FT_Outline*     outline,
                  PSH_Globals     globals,
                  FT_Render_Mode  hint_mode )
  {
    PSH_GlyphRec  glyphrec;
    PSH_Glyph     glyph = &glyphrec;
    FT_Error      error;
#ifdef DEBUG_HINTER
    FT_Memory     memory;
#endif
    FT_Int        dimension;


    /* something to do? */
    if ( outline->n_points == 0 || outline->n_contours == 0 )
      return FT_Err_Ok;

#ifdef DEBUG_HINTER

    memory = globals->memory;

    if ( ps_debug_glyph )
    {
      psh_glyph_done( ps_debug_glyph );
      FT_FREE( ps_debug_glyph );
    }

    if ( FT_NEW( glyph ) )
      return error;

    ps_debug_glyph = glyph;

#endif /* DEBUG_HINTER */

    error = psh_glyph_init( glyph, outline, ps_hints, globals );
    if ( error )
      goto Exit;

    /* try to optimize the y_scale so that the top of non-capital letters
     * is aligned on a pixel boundary whenever possible
     */
    {
      PSH_Dimension  dim_x = &glyph->globals->dimension[0];
      PSH_Dimension  dim_y = &glyph->globals->dimension[1];

      FT_Fixed  x_scale = dim_x->scale_mult;
      FT_Fixed  y_scale = dim_y->scale_mult;

      FT_Fixed  old_x_scale = x_scale;
      FT_Fixed  old_y_scale = y_scale;

      FT_Fixed  scaled;
      FT_Fixed  fitted;

      FT_Bool  rescale = FALSE;


      scaled = FT_MulFix( globals->blues.normal_top.zones->org_ref, y_scale );
      fitted = FT_PIX_ROUND( scaled );

      if ( fitted != 0 && scaled != fitted )
      {
        rescale = TRUE;

        y_scale = FT_MulDiv( y_scale, fitted, scaled );

        if ( fitted < scaled )
          x_scale -= x_scale / 50;

        psh_globals_set_scale( glyph->globals, x_scale, y_scale, 0, 0 );
      }

      glyph->do_horz_hints = 1;
      glyph->do_vert_hints = 1;

      glyph->do_horz_snapping = FT_BOOL( hint_mode == FT_RENDER_MODE_MONO ||
                                         hint_mode == FT_RENDER_MODE_LCD  );

      glyph->do_vert_snapping = FT_BOOL( hint_mode == FT_RENDER_MODE_MONO  ||
                                         hint_mode == FT_RENDER_MODE_LCD_V );

      glyph->do_stem_adjust   = FT_BOOL( hint_mode != FT_RENDER_MODE_LIGHT );

      for ( dimension = 0; dimension < 2; dimension++ )
      {
        /* load outline coordinates into glyph */
        psh_glyph_load_points( glyph, dimension );

        /* compute local extrema */
        psh_glyph_compute_extrema( glyph );

        /* compute aligned stem/hints positions */
        psh_hint_table_align_hints( &glyph->hint_tables[dimension],
                                    glyph->globals,
                                    dimension,
                                    glyph );

        /* find strong points, align them, then interpolate others */
        psh_glyph_find_strong_points( glyph, dimension );
        if ( dimension == 1 )
          psh_glyph_find_blue_points( &globals->blues, glyph );
        psh_glyph_interpolate_strong_points( glyph, dimension );
        psh_glyph_interpolate_normal_points( glyph, dimension );
        psh_glyph_interpolate_other_points( glyph, dimension );

        /* save hinted coordinates back to outline */
        psh_glyph_save_points( glyph, dimension );

        if ( rescale )
          psh_globals_set_scale( glyph->globals,
                                 old_x_scale, old_y_scale, 0, 0 );
      }
    }

  Exit:

#ifndef DEBUG_HINTER
    psh_glyph_done( glyph );
#endif

    return error;
  }


/* END */
