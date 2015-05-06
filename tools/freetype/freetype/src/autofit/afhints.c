/***************************************************************************/
/*                                                                         */
/*  afhints.c                                                              */
/*                                                                         */
/*    Auto-fitter hinting routines (body).                                 */
/*                                                                         */
/*  Copyright 2003-2007, 2009-2013 by                                      */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


#include "afhints.h"
#include "aferrors.h"
#include FT_INTERNAL_CALC_H
#include FT_INTERNAL_DEBUG_H


  /*************************************************************************/
  /*                                                                       */
  /* The macro FT_COMPONENT is used in trace mode.  It is an implicit      */
  /* parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log  */
  /* messages during execution.                                            */
  /*                                                                       */
#undef  FT_COMPONENT
#define FT_COMPONENT  trace_afhints


  /* Get new segment for given axis. */

  FT_LOCAL_DEF( FT_Error )
  af_axis_hints_new_segment( AF_AxisHints  axis,
                             FT_Memory     memory,
                             AF_Segment   *asegment )
  {
    FT_Error    error   = FT_Err_Ok;
    AF_Segment  segment = NULL;


    if ( axis->num_segments >= axis->max_segments )
    {
      FT_Int  old_max = axis->max_segments;
      FT_Int  new_max = old_max;
      FT_Int  big_max = (FT_Int)( FT_INT_MAX / sizeof ( *segment ) );


      if ( old_max >= big_max )
      {
        error = FT_THROW( Out_Of_Memory );
        goto Exit;
      }

      new_max += ( new_max >> 2 ) + 4;
      if ( new_max < old_max || new_max > big_max )
        new_max = big_max;

      if ( FT_RENEW_ARRAY( axis->segments, old_max, new_max ) )
        goto Exit;

      axis->max_segments = new_max;
    }

    segment = axis->segments + axis->num_segments++;

  Exit:
    *asegment = segment;
    return error;
  }


  /* Get new edge for given axis, direction, and position. */

  FT_LOCAL( FT_Error )
  af_axis_hints_new_edge( AF_AxisHints  axis,
                          FT_Int        fpos,
                          AF_Direction  dir,
                          FT_Memory     memory,
                          AF_Edge      *anedge )
  {
    FT_Error  error = FT_Err_Ok;
    AF_Edge   edge  = NULL;
    AF_Edge   edges;


    if ( axis->num_edges >= axis->max_edges )
    {
      FT_Int  old_max = axis->max_edges;
      FT_Int  new_max = old_max;
      FT_Int  big_max = (FT_Int)( FT_INT_MAX / sizeof ( *edge ) );


      if ( old_max >= big_max )
      {
        error = FT_THROW( Out_Of_Memory );
        goto Exit;
      }

      new_max += ( new_max >> 2 ) + 4;
      if ( new_max < old_max || new_max > big_max )
        new_max = big_max;

      if ( FT_RENEW_ARRAY( axis->edges, old_max, new_max ) )
        goto Exit;

      axis->max_edges = new_max;
    }

    edges = axis->edges;
    edge  = edges + axis->num_edges;

    while ( edge > edges )
    {
      if ( edge[-1].fpos < fpos )
        break;

      /* we want the edge with same position and minor direction */
      /* to appear before those in the major one in the list     */
      if ( edge[-1].fpos == fpos && dir == axis->major_dir )
        break;

      edge[0] = edge[-1];
      edge--;
    }

    axis->num_edges++;

    FT_ZERO( edge );
    edge->fpos = (FT_Short)fpos;
    edge->dir  = (FT_Char)dir;

  Exit:
    *anedge = edge;
    return error;
  }


#ifdef FT_DEBUG_AUTOFIT

#include FT_CONFIG_STANDARD_LIBRARY_H

  static const char*
  af_dir_str( AF_Direction  dir )
  {
    const char*  result;


    switch ( dir )
    {
    case AF_DIR_UP:
      result = "up";
      break;
    case AF_DIR_DOWN:
      result = "down";
      break;
    case AF_DIR_LEFT:
      result = "left";
      break;
    case AF_DIR_RIGHT:
      result = "right";
      break;
    default:
      result = "none";
    }

    return result;
  }


#define AF_INDEX_NUM( ptr, base )  ( (ptr) ? ( (ptr) - (base) ) : -1 )


#ifdef __cplusplus
  extern "C" {
#endif
  void
  af_glyph_hints_dump_points( AF_GlyphHints  hints )
  {
    AF_Point  points = hints->points;
    AF_Point  limit  = points + hints->num_points;
    AF_Point  point;


    FT_TRACE7(( "Table of points:\n"
                "  [ index |  xorg |  yorg | xscale | yscale"
                " |  xfit |  yfit |  flags ]\n" ));

    for ( point = points; point < limit; point++ )
      FT_TRACE7(( "  [ %5d | %5d | %5d | %6.2f | %6.2f"
                  " | %5.2f | %5.2f | %c%c%c%c%c%c ]\n",
                  point - points,
                  point->fx,
                  point->fy,
                  point->ox / 64.0,
                  point->oy / 64.0,
                  point->x / 64.0,
                  point->y / 64.0,
                  ( point->flags & AF_FLAG_WEAK_INTERPOLATION ) ? 'w' : ' ',
                  ( point->flags & AF_FLAG_INFLECTION )         ? 'i' : ' ',
                  ( point->flags & AF_FLAG_EXTREMA_X )          ? '<' : ' ',
                  ( point->flags & AF_FLAG_EXTREMA_Y )          ? 'v' : ' ',
                  ( point->flags & AF_FLAG_ROUND_X )            ? '(' : ' ',
                  ( point->flags & AF_FLAG_ROUND_Y )            ? 'u' : ' '));
    FT_TRACE7(( "\n" ));
  }
#ifdef __cplusplus
  }
#endif


  static const char*
  af_edge_flags_to_string( AF_Edge_Flags  flags )
  {
    static char  temp[32];
    int          pos = 0;


    if ( flags & AF_EDGE_ROUND )
    {
      ft_memcpy( temp + pos, "round", 5 );
      pos += 5;
    }
    if ( flags & AF_EDGE_SERIF )
    {
      if ( pos > 0 )
        temp[pos++] = ' ';
      ft_memcpy( temp + pos, "serif", 5 );
      pos += 5;
    }
    if ( pos == 0 )
      return "normal";

    temp[pos] = '\0';

    return temp;
  }


  /* Dump the array of linked segments. */

#ifdef __cplusplus
  extern "C" {
#endif
  void
  af_glyph_hints_dump_segments( AF_GlyphHints  hints )
  {
    FT_Int  dimension;


    for ( dimension = 1; dimension >= 0; dimension-- )
    {
      AF_AxisHints  axis     = &hints->axis[dimension];
      AF_Point      points   = hints->points;
      AF_Edge       edges    = axis->edges;
      AF_Segment    segments = axis->segments;
      AF_Segment    limit    = segments + axis->num_segments;
      AF_Segment    seg;


      FT_TRACE7(( "Table of %s segments:\n",
                  dimension == AF_DIMENSION_HORZ ? "vertical"
                                                 : "horizontal" ));
      if ( axis->num_segments )
        FT_TRACE7(( "  [ index |  pos  |  dir  | from"
                    " |  to  | link | serif | edge"
                    " | height | extra |    flags    ]\n" ));
      else
        FT_TRACE7(( "  (none)\n" ));

      for ( seg = segments; seg < limit; seg++ )
        FT_TRACE7(( "  [ %5d | %5.2g | %5s | %4d"
                    " | %4d | %4d | %5d | %4d"
                    " | %6d | %5d | %11s ]\n",
                    seg - segments,
                    dimension == AF_DIMENSION_HORZ
                                 ? (int)seg->first->ox / 64.0
                                 : (int)seg->first->oy / 64.0,
                    af_dir_str( (AF_Direction)seg->dir ),
                    AF_INDEX_NUM( seg->first, points ),
                    AF_INDEX_NUM( seg->last, points ),
                    AF_INDEX_NUM( seg->link, segments ),
                    AF_INDEX_NUM( seg->serif, segments ),
                    AF_INDEX_NUM( seg->edge, edges ),
                    seg->height,
                    seg->height - ( seg->max_coord - seg->min_coord ),
                    af_edge_flags_to_string( (AF_Edge_Flags)seg->flags ) ));
      FT_TRACE7(( "\n" ));
    }
  }
#ifdef __cplusplus
  }
#endif


  /* Fetch number of segments. */

#ifdef __cplusplus
  extern "C" {
#endif
  FT_Error
  af_glyph_hints_get_num_segments( AF_GlyphHints  hints,
                                   FT_Int         dimension,
                                   FT_Int*        num_segments )
  {
    AF_Dimension  dim;
    AF_AxisHints  axis;


    dim = ( dimension == 0 ) ? AF_DIMENSION_HORZ : AF_DIMENSION_VERT;

    axis          = &hints->axis[dim];
    *num_segments = axis->num_segments;

    return FT_Err_Ok;
  }
#ifdef __cplusplus
  }
#endif


  /* Fetch offset of segments into user supplied offset array. */

#ifdef __cplusplus
  extern "C" {
#endif
  FT_Error
  af_glyph_hints_get_segment_offset( AF_GlyphHints  hints,
                                     FT_Int         dimension,
                                     FT_Int         idx,
                                     FT_Pos*        offset )
  {
    AF_Dimension  dim;
    AF_AxisHints  axis;
    AF_Segment    seg;


    if ( !offset )
      return FT_THROW( Invalid_Argument );

    dim = ( dimension == 0 ) ? AF_DIMENSION_HORZ : AF_DIMENSION_VERT;

    axis = &hints->axis[dim];

    if ( idx < 0 || idx >= axis->num_segments )
      return FT_THROW( Invalid_Argument );

    seg     = &axis->segments[idx];
    *offset = ( dim == AF_DIMENSION_HORZ ) ? seg->first->ox
                                           : seg->first->oy;

    return FT_Err_Ok;
  }
#ifdef __cplusplus
  }
#endif


  /* Dump the array of linked edges. */

#ifdef __cplusplus
  extern "C" {
#endif
  void
  af_glyph_hints_dump_edges( AF_GlyphHints  hints )
  {
    FT_Int  dimension;


    for ( dimension = 1; dimension >= 0; dimension-- )
    {
      AF_AxisHints  axis  = &hints->axis[dimension];
      AF_Edge       edges = axis->edges;
      AF_Edge       limit = edges + axis->num_edges;
      AF_Edge       edge;


      /*
       *  note: AF_DIMENSION_HORZ corresponds to _vertical_ edges
       *        since they have a constant X coordinate.
       */
      FT_TRACE7(( "Table of %s edges:\n",
                  dimension == AF_DIMENSION_HORZ ? "vertical"
                                                 : "horizontal" ));
      if ( axis->num_edges )
        FT_TRACE7(( "  [ index |  pos  |  dir  | link"
                    " | serif | blue | opos  |  pos  |    flags    ]\n" ));
      else
        FT_TRACE7(( "  (none)\n" ));

      for ( edge = edges; edge < limit; edge++ )
        FT_TRACE7(( "  [ %5d | %5.2g | %5s | %4d"
                    " | %5d |   %c  | %5.2f | %5.2f | %11s ]\n",
                    edge - edges,
                    (int)edge->opos / 64.0,
                    af_dir_str( (AF_Direction)edge->dir ),
                    AF_INDEX_NUM( edge->link, edges ),
                    AF_INDEX_NUM( edge->serif, edges ),
                    edge->blue_edge ? 'y' : 'n',
                    edge->opos / 64.0,
                    edge->pos / 64.0,
                    af_edge_flags_to_string( (AF_Edge_Flags)edge->flags ) ));
      FT_TRACE7(( "\n" ));
    }
  }
#ifdef __cplusplus
  }
#endif

#else /* !FT_DEBUG_AUTOFIT */

  /* these empty stubs are only used to link the `ftgrid' test program */
  /* if debugging is disabled                                          */

#ifdef __cplusplus
  extern "C" {
#endif

  void
  af_glyph_hints_dump_points( AF_GlyphHints  hints )
  {
    FT_UNUSED( hints );
  }


  void
  af_glyph_hints_dump_segments( AF_GlyphHints  hints )
  {
    FT_UNUSED( hints );
  }


  FT_Error
  af_glyph_hints_get_num_segments( AF_GlyphHints  hints,
                                   FT_Int         dimension,
                                   FT_Int*        num_segments )
  {
    FT_UNUSED( hints );
    FT_UNUSED( dimension );
    FT_UNUSED( num_segments );

    return 0;
  }


  FT_Error
  af_glyph_hints_get_segment_offset( AF_GlyphHints  hints,
                                     FT_Int         dimension,
                                     FT_Int         idx,
                                     FT_Pos*        offset )
  {
    FT_UNUSED( hints );
    FT_UNUSED( dimension );
    FT_UNUSED( idx );
    FT_UNUSED( offset );

    return 0;
  }


  void
  af_glyph_hints_dump_edges( AF_GlyphHints  hints )
  {
    FT_UNUSED( hints );
  }

#ifdef __cplusplus
  }
#endif

#endif /* !FT_DEBUG_AUTOFIT */


  /* Compute the direction value of a given vector. */

  FT_LOCAL_DEF( AF_Direction )
  af_direction_compute( FT_Pos  dx,
                        FT_Pos  dy )
  {
    FT_Pos        ll, ss;  /* long and short arm lengths */
    AF_Direction  dir;     /* candidate direction        */


    if ( dy >= dx )
    {
      if ( dy >= -dx )
      {
        dir = AF_DIR_UP;
        ll  = dy;
        ss  = dx;
      }
      else
      {
        dir = AF_DIR_LEFT;
        ll  = -dx;
        ss  = dy;
      }
    }
    else /* dy < dx */
    {
      if ( dy >= -dx )
      {
        dir = AF_DIR_RIGHT;
        ll  = dx;
        ss  = dy;
      }
      else
      {
        dir = AF_DIR_DOWN;
        ll  = dy;
        ss  = dx;
      }
    }

    /* return no direction if arm lengths differ too much            */
    /* (value 14 is heuristic, corresponding to approx. 4.1 degrees) */
    ss *= 14;
    if ( FT_ABS( ll ) <= FT_ABS( ss ) )
      dir = AF_DIR_NONE;

    return dir;
  }


  FT_LOCAL_DEF( void )
  af_glyph_hints_init( AF_GlyphHints  hints,
                       FT_Memory      memory )
  {
    FT_ZERO( hints );
    hints->memory = memory;
  }


  FT_LOCAL_DEF( void )
  af_glyph_hints_done( AF_GlyphHints  hints )
  {
    FT_Memory  memory = hints->memory;
    int        dim;


    if ( !( hints && hints->memory ) )
      return;

    /*
     *  note that we don't need to free the segment and edge
     *  buffers since they are really within the hints->points array
     */
    for ( dim = 0; dim < AF_DIMENSION_MAX; dim++ )
    {
      AF_AxisHints  axis = &hints->axis[dim];


      axis->num_segments = 0;
      axis->max_segments = 0;
      FT_FREE( axis->segments );

      axis->num_edges = 0;
      axis->max_edges = 0;
      FT_FREE( axis->edges );
    }

    FT_FREE( hints->contours );
    hints->max_contours = 0;
    hints->num_contours = 0;

    FT_FREE( hints->points );
    hints->num_points = 0;
    hints->max_points = 0;

    hints->memory = NULL;
  }


  /* Reset metrics. */

  FT_LOCAL_DEF( void )
  af_glyph_hints_rescale( AF_GlyphHints     hints,
                          AF_ScriptMetrics  metrics )
  {
    hints->metrics      = metrics;
    hints->scaler_flags = metrics->scaler.flags;
  }


  /* Recompute all AF_Point in AF_GlyphHints from the definitions */
  /* in a source outline.                                         */

  FT_LOCAL_DEF( FT_Error )
  af_glyph_hints_reload( AF_GlyphHints  hints,
                         FT_Outline*    outline )
  {
    FT_Error   error   = FT_Err_Ok;
    AF_Point   points;
    FT_UInt    old_max, new_max;
    FT_Fixed   x_scale = hints->x_scale;
    FT_Fixed   y_scale = hints->y_scale;
    FT_Pos     x_delta = hints->x_delta;
    FT_Pos     y_delta = hints->y_delta;
    FT_Memory  memory  = hints->memory;


    hints->num_points   = 0;
    hints->num_contours = 0;

    hints->axis[0].num_segments = 0;
    hints->axis[0].num_edges    = 0;
    hints->axis[1].num_segments = 0;
    hints->axis[1].num_edges    = 0;

    /* first of all, reallocate the contours array if necessary */
    new_max = (FT_UInt)outline->n_contours;
    old_max = hints->max_contours;
    if ( new_max > old_max )
    {
      new_max = ( new_max + 3 ) & ~3; /* round up to a multiple of 4 */

      if ( FT_RENEW_ARRAY( hints->contours, old_max, new_max ) )
        goto Exit;

      hints->max_contours = new_max;
    }

    /*
     *  then reallocate the points arrays if necessary --
     *  note that we reserve two additional point positions, used to
     *  hint metrics appropriately
     */
    new_max = (FT_UInt)( outline->n_points + 2 );
    old_max = hints->max_points;
    if ( new_max > old_max )
    {
      new_max = ( new_max + 2 + 7 ) & ~7; /* round up to a multiple of 8 */

      if ( FT_RENEW_ARRAY( hints->points, old_max, new_max ) )
        goto Exit;

      hints->max_points = new_max;
    }

    hints->num_points   = outline->n_points;
    hints->num_contours = outline->n_contours;

    /* We can't rely on the value of `FT_Outline.flags' to know the fill   */
    /* direction used for a glyph, given that some fonts are broken (e.g., */
    /* the Arphic ones).  We thus recompute it each time we need to.       */
    /*                                                                     */
    hints->axis[AF_DIMENSION_HORZ].major_dir = AF_DIR_UP;
    hints->axis[AF_DIMENSION_VERT].major_dir = AF_DIR_LEFT;

    if ( FT_Outline_Get_Orientation( outline ) == FT_ORIENTATION_POSTSCRIPT )
    {
      hints->axis[AF_DIMENSION_HORZ].major_dir = AF_DIR_DOWN;
      hints->axis[AF_DIMENSION_VERT].major_dir = AF_DIR_RIGHT;
    }

    hints->x_scale = x_scale;
    hints->y_scale = y_scale;
    hints->x_delta = x_delta;
    hints->y_delta = y_delta;

    hints->xmin_delta = 0;
    hints->xmax_delta = 0;

    points = hints->points;
    if ( hints->num_points == 0 )
      goto Exit;

    {
      AF_Point  point;
      AF_Point  point_limit = points + hints->num_points;


      /* compute coordinates & Bezier flags, next and prev */
      {
        FT_Vector*  vec           = outline->points;
        char*       tag           = outline->tags;
        AF_Point    end           = points + outline->contours[0];
        AF_Point    prev          = end;
        FT_Int      contour_index = 0;


        for ( point = points; point < point_limit; point++, vec++, tag++ )
        {
          point->fx = (FT_Short)vec->x;
          point->fy = (FT_Short)vec->y;
          point->ox = point->x = FT_MulFix( vec->x, x_scale ) + x_delta;
          point->oy = point->y = FT_MulFix( vec->y, y_scale ) + y_delta;

          switch ( FT_CURVE_TAG( *tag ) )
          {
          case FT_CURVE_TAG_CONIC:
            point->flags = AF_FLAG_CONIC;
            break;
          case FT_CURVE_TAG_CUBIC:
            point->flags = AF_FLAG_CUBIC;
            break;
          default:
            point->flags = AF_FLAG_NONE;
          }

          point->prev = prev;
          prev->next  = point;
          prev        = point;

          if ( point == end )
          {
            if ( ++contour_index < outline->n_contours )
            {
              end  = points + outline->contours[contour_index];
              prev = end;
            }
          }
        }
      }

      /* set up the contours array */
      {
        AF_Point*  contour       = hints->contours;
        AF_Point*  contour_limit = contour + hints->num_contours;
        short*     end           = outline->contours;
        short      idx           = 0;


        for ( ; contour < contour_limit; contour++, end++ )
        {
          contour[0] = points + idx;
          idx        = (short)( end[0] + 1 );
        }
      }

      /* compute directions of in & out vectors */
      {
        AF_Point      first  = points;
        AF_Point      prev   = NULL;
        FT_Pos        in_x   = 0;
        FT_Pos        in_y   = 0;
        AF_Direction  in_dir = AF_DIR_NONE;


        for ( point = points; point < point_limit; point++ )
        {
          AF_Point  next;
          FT_Pos    out_x, out_y;


          if ( point == first )
          {
            prev   = first->prev;
            in_x   = first->fx - prev->fx;
            in_y   = first->fy - prev->fy;
            in_dir = af_direction_compute( in_x, in_y );
            first  = prev + 1;
          }

          point->in_dir = (FT_Char)in_dir;

          next  = point->next;
          out_x = next->fx - point->fx;
          out_y = next->fy - point->fy;

          in_dir         = af_direction_compute( out_x, out_y );
          point->out_dir = (FT_Char)in_dir;

          /* check for weak points */

          if ( point->flags & AF_FLAG_CONTROL )
          {
          Is_Weak_Point:
            point->flags |= AF_FLAG_WEAK_INTERPOLATION;
          }
          else if ( point->out_dir == point->in_dir )
          {
            if ( point->out_dir != AF_DIR_NONE )
              goto Is_Weak_Point;

            if ( ft_corner_is_flat( in_x, in_y, out_x, out_y ) )
              goto Is_Weak_Point;
          }
          else if ( point->in_dir == -point->out_dir )
            goto Is_Weak_Point;

          in_x = out_x;
          in_y = out_y;
          prev = point;
        }
      }
    }

  Exit:
    return error;
  }


  /* Store the hinted outline in an FT_Outline structure. */

  FT_LOCAL_DEF( void )
  af_glyph_hints_save( AF_GlyphHints  hints,
                       FT_Outline*    outline )
  {
    AF_Point    point = hints->points;
    AF_Point    limit = point + hints->num_points;
    FT_Vector*  vec   = outline->points;
    char*       tag   = outline->tags;


    for ( ; point < limit; point++, vec++, tag++ )
    {
      vec->x = point->x;
      vec->y = point->y;

      if ( point->flags & AF_FLAG_CONIC )
        tag[0] = FT_CURVE_TAG_CONIC;
      else if ( point->flags & AF_FLAG_CUBIC )
        tag[0] = FT_CURVE_TAG_CUBIC;
      else
        tag[0] = FT_CURVE_TAG_ON;
    }
  }


  /****************************************************************
   *
   *                     EDGE POINT GRID-FITTING
   *
   ****************************************************************/


  /* Align all points of an edge to the same coordinate value, */
  /* either horizontally or vertically.                        */

  FT_LOCAL_DEF( void )
  af_glyph_hints_align_edge_points( AF_GlyphHints  hints,
                                    AF_Dimension   dim )
  {
    AF_AxisHints  axis          = & hints->axis[dim];
    AF_Segment    segments      = axis->segments;
    AF_Segment    segment_limit = segments + axis->num_segments;
    AF_Segment    seg;


    if ( dim == AF_DIMENSION_HORZ )
    {
      for ( seg = segments; seg < segment_limit; seg++ )
      {
        AF_Edge   edge = seg->edge;
        AF_Point  point, first, last;


        if ( edge == NULL )
          continue;

        first = seg->first;
        last  = seg->last;
        point = first;
        for (;;)
        {
          point->x      = edge->pos;
          point->flags |= AF_FLAG_TOUCH_X;

          if ( point == last )
            break;

          point = point->next;
        }
      }
    }
    else
    {
      for ( seg = segments; seg < segment_limit; seg++ )
      {
        AF_Edge   edge = seg->edge;
        AF_Point  point, first, last;


        if ( edge == NULL )
          continue;

        first = seg->first;
        last  = seg->last;
        point = first;
        for (;;)
        {
          point->y      = edge->pos;
          point->flags |= AF_FLAG_TOUCH_Y;

          if ( point == last )
            break;

          point = point->next;
        }
      }
    }
  }


  /****************************************************************
   *
   *                    STRONG POINT INTERPOLATION
   *
   ****************************************************************/


  /* Hint the strong points -- this is equivalent to the TrueType `IP' */
  /* hinting instruction.                                              */

  FT_LOCAL_DEF( void )
  af_glyph_hints_align_strong_points( AF_GlyphHints  hints,
                                      AF_Dimension   dim )
  {
    AF_Point      points      = hints->points;
    AF_Point      point_limit = points + hints->num_points;
    AF_AxisHints  axis        = &hints->axis[dim];
    AF_Edge       edges       = axis->edges;
    AF_Edge       edge_limit  = edges + axis->num_edges;
    AF_Flags      touch_flag;


    if ( dim == AF_DIMENSION_HORZ )
      touch_flag = AF_FLAG_TOUCH_X;
    else
      touch_flag  = AF_FLAG_TOUCH_Y;

    if ( edges < edge_limit )
    {
      AF_Point  point;
      AF_Edge   edge;


      for ( point = points; point < point_limit; point++ )
      {
        FT_Pos  u, ou, fu;  /* point position */
        FT_Pos  delta;


        if ( point->flags & touch_flag )
          continue;

        /* if this point is candidate to weak interpolation, we       */
        /* interpolate it after all strong points have been processed */

        if (  ( point->flags & AF_FLAG_WEAK_INTERPOLATION ) &&
             !( point->flags & AF_FLAG_INFLECTION )         )
          continue;

        if ( dim == AF_DIMENSION_VERT )
        {
          u  = point->fy;
          ou = point->oy;
        }
        else
        {
          u  = point->fx;
          ou = point->ox;
        }

        fu = u;

        /* is the point before the first edge? */
        edge  = edges;
        delta = edge->fpos - u;
        if ( delta >= 0 )
        {
          u = edge->pos - ( edge->opos - ou );
          goto Store_Point;
        }

        /* is the point after the last edge? */
        edge  = edge_limit - 1;
        delta = u - edge->fpos;
        if ( delta >= 0 )
        {
          u = edge->pos + ( ou - edge->opos );
          goto Store_Point;
        }

        {
          FT_PtrDist  min, max, mid;
          FT_Pos      fpos;


          /* find enclosing edges */
          min = 0;
          max = edge_limit - edges;

#if 1
          /* for a small number of edges, a linear search is better */
          if ( max <= 8 )
          {
            FT_PtrDist  nn;


            for ( nn = 0; nn < max; nn++ )
              if ( edges[nn].fpos >= u )
                break;

            if ( edges[nn].fpos == u )
            {
              u = edges[nn].pos;
              goto Store_Point;
            }
            min = nn;
          }
          else
#endif
          while ( min < max )
          {
            mid  = ( max + min ) >> 1;
            edge = edges + mid;
            fpos = edge->fpos;

            if ( u < fpos )
              max = mid;
            else if ( u > fpos )
              min = mid + 1;
            else
            {
              /* we are on the edge */
              u = edge->pos;
              goto Store_Point;
            }
          }

          /* point is not on an edge */
          {
            AF_Edge  before = edges + min - 1;
            AF_Edge  after  = edges + min + 0;


            /* assert( before && after && before != after ) */
            if ( before->scale == 0 )
              before->scale = FT_DivFix( after->pos - before->pos,
                                         after->fpos - before->fpos );

            u = before->pos + FT_MulFix( fu - before->fpos,
                                         before->scale );
          }
        }

      Store_Point:
        /* save the point position */
        if ( dim == AF_DIMENSION_HORZ )
          point->x = u;
        else
          point->y = u;

        point->flags |= touch_flag;
      }
    }
  }


  /****************************************************************
   *
   *                    WEAK POINT INTERPOLATION
   *
   ****************************************************************/


  /* Shift the original coordinates of all points between `p1' and */
  /* `p2' to get hinted coordinates, using the same difference as  */
  /* given by `ref'.                                               */

  static void
  af_iup_shift( AF_Point  p1,
                AF_Point  p2,
                AF_Point  ref )
  {
    AF_Point  p;
    FT_Pos    delta = ref->u - ref->v;


    if ( delta == 0 )
      return;

    for ( p = p1; p < ref; p++ )
      p->u = p->v + delta;

    for ( p = ref + 1; p <= p2; p++ )
      p->u = p->v + delta;
  }


  /* Interpolate the original coordinates of all points between `p1' and  */
  /* `p2' to get hinted coordinates, using `ref1' and `ref2' as the       */
  /* reference points.  The `u' and `v' members are the current and       */
  /* original coordinate values, respectively.                            */
  /*                                                                      */
  /* Details can be found in the TrueType bytecode specification.         */

  static void
  af_iup_interp( AF_Point  p1,
                 AF_Point  p2,
                 AF_Point  ref1,
                 AF_Point  ref2 )
  {
    AF_Point  p;
    FT_Pos    u;
    FT_Pos    v1 = ref1->v;
    FT_Pos    v2 = ref2->v;
    FT_Pos    d1 = ref1->u - v1;
    FT_Pos    d2 = ref2->u - v2;


    if ( p1 > p2 )
      return;

    if ( v1 == v2 )
    {
      for ( p = p1; p <= p2; p++ )
      {
        u = p->v;

        if ( u <= v1 )
          u += d1;
        else
          u += d2;

        p->u = u;
      }
      return;
    }

    if ( v1 < v2 )
    {
      for ( p = p1; p <= p2; p++ )
      {
        u = p->v;

        if ( u <= v1 )
          u += d1;
        else if ( u >= v2 )
          u += d2;
        else
          u = ref1->u + FT_MulDiv( u - v1, ref2->u - ref1->u, v2 - v1 );

        p->u = u;
      }
    }
    else
    {
      for ( p = p1; p <= p2; p++ )
      {
        u = p->v;

        if ( u <= v2 )
          u += d2;
        else if ( u >= v1 )
          u += d1;
        else
          u = ref1->u + FT_MulDiv( u - v1, ref2->u - ref1->u, v2 - v1 );

        p->u = u;
      }
    }
  }


  /* Hint the weak points -- this is equivalent to the TrueType `IUP' */
  /* hinting instruction.                                             */

  FT_LOCAL_DEF( void )
  af_glyph_hints_align_weak_points( AF_GlyphHints  hints,
                                    AF_Dimension   dim )
  {
    AF_Point   points        = hints->points;
    AF_Point   point_limit   = points + hints->num_points;
    AF_Point*  contour       = hints->contours;
    AF_Point*  contour_limit = contour + hints->num_contours;
    AF_Flags   touch_flag;
    AF_Point   point;
    AF_Point   end_point;
    AF_Point   first_point;


    /* PASS 1: Move segment points to edge positions */

    if ( dim == AF_DIMENSION_HORZ )
    {
      touch_flag = AF_FLAG_TOUCH_X;

      for ( point = points; point < point_limit; point++ )
      {
        point->u = point->x;
        point->v = point->ox;
      }
    }
    else
    {
      touch_flag = AF_FLAG_TOUCH_Y;

      for ( point = points; point < point_limit; point++ )
      {
        point->u = point->y;
        point->v = point->oy;
      }
    }

    point = points;

    for ( ; contour < contour_limit; contour++ )
    {
      AF_Point  first_touched, last_touched;


      point       = *contour;
      end_point   = point->prev;
      first_point = point;

      /* find first touched point */
      for (;;)
      {
        if ( point > end_point )  /* no touched point in contour */
          goto NextContour;

        if ( point->flags & touch_flag )
          break;

        point++;
      }

      first_touched = point;
      last_touched  = point;

      for (;;)
      {
        FT_ASSERT( point <= end_point                 &&
                   ( point->flags & touch_flag ) != 0 );

        /* skip any touched neighbours */
        while ( point < end_point                    &&
                ( point[1].flags & touch_flag ) != 0 )
          point++;

        last_touched = point;

        /* find the next touched point, if any */
        point++;
        for (;;)
        {
          if ( point > end_point )
            goto EndContour;

          if ( ( point->flags & touch_flag ) != 0 )
            break;

          point++;
        }

        /* interpolate between last_touched and point */
        af_iup_interp( last_touched + 1, point - 1,
                       last_touched, point );
      }

    EndContour:
      /* special case: only one point was touched */
      if ( last_touched == first_touched )
        af_iup_shift( first_point, end_point, first_touched );

      else /* interpolate the last part */
      {
        if ( last_touched < end_point )
          af_iup_interp( last_touched + 1, end_point,
                         last_touched, first_touched );

        if ( first_touched > points )
          af_iup_interp( first_point, first_touched - 1,
                         last_touched, first_touched );
      }

    NextContour:
      ;
    }

    /* now save the interpolated values back to x/y */
    if ( dim == AF_DIMENSION_HORZ )
    {
      for ( point = points; point < point_limit; point++ )
        point->x = point->u;
    }
    else
    {
      for ( point = points; point < point_limit; point++ )
        point->y = point->u;
    }
  }


#ifdef AF_CONFIG_OPTION_USE_WARPER

  /* Apply (small) warp scale and warp delta for given dimension. */

  FT_LOCAL_DEF( void )
  af_glyph_hints_scale_dim( AF_GlyphHints  hints,
                            AF_Dimension   dim,
                            FT_Fixed       scale,
                            FT_Pos         delta )
  {
    AF_Point  points       = hints->points;
    AF_Point  points_limit = points + hints->num_points;
    AF_Point  point;


    if ( dim == AF_DIMENSION_HORZ )
    {
      for ( point = points; point < points_limit; point++ )
        point->x = FT_MulFix( point->fx, scale ) + delta;
    }
    else
    {
      for ( point = points; point < points_limit; point++ )
        point->y = FT_MulFix( point->fy, scale ) + delta;
    }
  }

#endif /* AF_CONFIG_OPTION_USE_WARPER */

/* END */
