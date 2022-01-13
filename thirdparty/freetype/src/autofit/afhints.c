/****************************************************************************
 *
 * afhints.c
 *
 *   Auto-fitter hinting routines (body).
 *
 * Copyright (C) 2003-2021 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#include "afhints.h"
#include "aferrors.h"
#include <freetype/internal/ftcalc.h>
#include <freetype/internal/ftdebug.h>


  /**************************************************************************
   *
   * The macro FT_COMPONENT is used in trace mode.  It is an implicit
   * parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log
   * messages during execution.
   */
#undef  FT_COMPONENT
#define FT_COMPONENT  afhints


  FT_LOCAL_DEF( void )
  af_sort_pos( FT_UInt  count,
               FT_Pos*  table )
  {
    FT_UInt  i, j;
    FT_Pos   swap;


    for ( i = 1; i < count; i++ )
    {
      for ( j = i; j > 0; j-- )
      {
        if ( table[j] >= table[j - 1] )
          break;

        swap         = table[j];
        table[j]     = table[j - 1];
        table[j - 1] = swap;
      }
    }
  }


  FT_LOCAL_DEF( void )
  af_sort_and_quantize_widths( FT_UInt*  count,
                               AF_Width  table,
                               FT_Pos    threshold )
  {
    FT_UInt      i, j;
    FT_UInt      cur_idx;
    FT_Pos       cur_val;
    FT_Pos       sum;
    AF_WidthRec  swap;


    if ( *count == 1 )
      return;

    /* sort */
    for ( i = 1; i < *count; i++ )
    {
      for ( j = i; j > 0; j-- )
      {
        if ( table[j].org >= table[j - 1].org )
          break;

        swap         = table[j];
        table[j]     = table[j - 1];
        table[j - 1] = swap;
      }
    }

    cur_idx = 0;
    cur_val = table[cur_idx].org;

    /* compute and use mean values for clusters not larger than  */
    /* `threshold'; this is very primitive and might not yield   */
    /* the best result, but normally, using reference character  */
    /* `o', `*count' is 2, so the code below is fully sufficient */
    for ( i = 1; i < *count; i++ )
    {
      if ( table[i].org - cur_val > threshold ||
           i == *count - 1                    )
      {
        sum = 0;

        /* fix loop for end of array */
        if ( table[i].org - cur_val <= threshold &&
             i == *count - 1                     )
          i++;

        for ( j = cur_idx; j < i; j++ )
        {
          sum         += table[j].org;
          table[j].org = 0;
        }
        table[cur_idx].org = sum / (FT_Pos)j;

        if ( i < *count - 1 )
        {
          cur_idx = i + 1;
          cur_val = table[cur_idx].org;
        }
      }
    }

    cur_idx = 1;

    /* compress array to remove zero values */
    for ( i = 1; i < *count; i++ )
    {
      if ( table[i].org )
        table[cur_idx++] = table[i];
    }

    *count = cur_idx;
  }

  /* Get new segment for given axis. */

  FT_LOCAL_DEF( FT_Error )
  af_axis_hints_new_segment( AF_AxisHints  axis,
                             FT_Memory     memory,
                             AF_Segment   *asegment )
  {
    FT_Error    error   = FT_Err_Ok;
    AF_Segment  segment = NULL;


    if ( axis->num_segments < AF_SEGMENTS_EMBEDDED )
    {
      if ( !axis->segments )
      {
        axis->segments     = axis->embedded.segments;
        axis->max_segments = AF_SEGMENTS_EMBEDDED;
      }
    }
    else if ( axis->num_segments >= axis->max_segments )
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

      if ( axis->segments == axis->embedded.segments )
      {
        if ( FT_NEW_ARRAY( axis->segments, new_max ) )
          goto Exit;
        ft_memcpy( axis->segments, axis->embedded.segments,
                   sizeof ( axis->embedded.segments ) );
      }
      else
      {
        if ( FT_RENEW_ARRAY( axis->segments, old_max, new_max ) )
          goto Exit;
      }

      axis->max_segments = new_max;
    }

    segment = axis->segments + axis->num_segments++;

  Exit:
    *asegment = segment;
    return error;
  }


  /* Get new edge for given axis, direction, and position, */
  /* without initializing the edge itself.                 */

  FT_LOCAL( FT_Error )
  af_axis_hints_new_edge( AF_AxisHints  axis,
                          FT_Int        fpos,
                          AF_Direction  dir,
                          FT_Bool       top_to_bottom_hinting,
                          FT_Memory     memory,
                          AF_Edge      *anedge )
  {
    FT_Error  error = FT_Err_Ok;
    AF_Edge   edge  = NULL;
    AF_Edge   edges;


    if ( axis->num_edges < AF_EDGES_EMBEDDED )
    {
      if ( !axis->edges )
      {
        axis->edges     = axis->embedded.edges;
        axis->max_edges = AF_EDGES_EMBEDDED;
      }
    }
    else if ( axis->num_edges >= axis->max_edges )
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

      if ( axis->edges == axis->embedded.edges )
      {
        if ( FT_NEW_ARRAY( axis->edges, new_max ) )
          goto Exit;
        ft_memcpy( axis->edges, axis->embedded.edges,
                   sizeof ( axis->embedded.edges ) );
      }
      else
      {
        if ( FT_RENEW_ARRAY( axis->edges, old_max, new_max ) )
          goto Exit;
      }

      axis->max_edges = new_max;
    }

    edges = axis->edges;
    edge  = edges + axis->num_edges;

    while ( edge > edges )
    {
      if ( top_to_bottom_hinting ? ( edge[-1].fpos > fpos )
                                 : ( edge[-1].fpos < fpos ) )
        break;

      /* we want the edge with same position and minor direction */
      /* to appear before those in the major one in the list     */
      if ( edge[-1].fpos == fpos && dir == axis->major_dir )
        break;

      edge[0] = edge[-1];
      edge--;
    }

    axis->num_edges++;

  Exit:
    *anedge = edge;
    return error;
  }


#ifdef FT_DEBUG_AUTOFIT

#include FT_CONFIG_STANDARD_LIBRARY_H

  /* The dump functions are used in the `ftgrid' demo program, too. */
#define AF_DUMP( varformat )          \
          do                          \
          {                           \
            if ( to_stdout )          \
              printf varformat;       \
            else                      \
              FT_TRACE7( varformat ); \
          } while ( 0 )


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


#define AF_INDEX_NUM( ptr, base )  (int)( (ptr) ? ( (ptr) - (base) ) : -1 )


  static char*
  af_print_idx( char* p,
                int   idx )
  {
    if ( idx == -1 )
    {
      p[0] = '-';
      p[1] = '-';
      p[2] = '\0';
    }
    else
      ft_sprintf( p, "%d", idx );

    return p;
  }


  static int
  af_get_segment_index( AF_GlyphHints  hints,
                        int            point_idx,
                        int            dimension )
  {
    AF_AxisHints  axis     = &hints->axis[dimension];
    AF_Point      point    = hints->points + point_idx;
    AF_Segment    segments = axis->segments;
    AF_Segment    limit    = segments + axis->num_segments;
    AF_Segment    segment;


    for ( segment = segments; segment < limit; segment++ )
    {
      if ( segment->first <= segment->last )
      {
        if ( point >= segment->first && point <= segment->last )
          break;
      }
      else
      {
        AF_Point  p = segment->first;


        for (;;)
        {
          if ( point == p )
            goto Exit;

          if ( p == segment->last )
            break;

          p = p->next;
        }
      }
    }

  Exit:
    if ( segment == limit )
      return -1;

    return (int)( segment - segments );
  }


  static int
  af_get_edge_index( AF_GlyphHints  hints,
                     int            segment_idx,
                     int            dimension )
  {
    AF_AxisHints  axis    = &hints->axis[dimension];
    AF_Edge       edges   = axis->edges;
    AF_Segment    segment = axis->segments + segment_idx;


    return segment_idx == -1 ? -1 : AF_INDEX_NUM( segment->edge, edges );
  }


  static int
  af_get_strong_edge_index( AF_GlyphHints  hints,
                            AF_Edge*       strong_edges,
                            int            dimension )
  {
    AF_AxisHints  axis  = &hints->axis[dimension];
    AF_Edge       edges = axis->edges;


    return AF_INDEX_NUM( strong_edges[dimension], edges );
  }


#ifdef __cplusplus
  extern "C" {
#endif
  void
  af_glyph_hints_dump_points( AF_GlyphHints  hints,
                              FT_Bool        to_stdout )
  {
    AF_Point   points  = hints->points;
    AF_Point   limit   = points + hints->num_points;
    AF_Point*  contour = hints->contours;
    AF_Point*  climit  = contour + hints->num_contours;
    AF_Point   point;


    AF_DUMP(( "Table of points:\n" ));

    if ( hints->num_points )
    {
      AF_DUMP(( "  index  hedge  hseg  vedge  vseg  flags "
             /* "  XXXXX  XXXXX XXXXX  XXXXX XXXXX  XXXXXX" */
                "  xorg  yorg  xscale  yscale   xfit    yfit "
             /* " XXXXX XXXXX XXXX.XX XXXX.XX XXXX.XX XXXX.XX" */
                "  hbef  haft  vbef  vaft" ));
             /* " XXXXX XXXXX XXXXX XXXXX" */
    }
    else
      AF_DUMP(( "  (none)\n" ));

    for ( point = points; point < limit; point++ )
    {
      int  point_idx     = AF_INDEX_NUM( point, points );
      int  segment_idx_0 = af_get_segment_index( hints, point_idx, 0 );
      int  segment_idx_1 = af_get_segment_index( hints, point_idx, 1 );

      char  buf1[16], buf2[16], buf3[16], buf4[16];
      char  buf5[16], buf6[16], buf7[16], buf8[16];


      /* insert extra newline at the beginning of a contour */
      if ( contour < climit && *contour == point )
      {
        AF_DUMP(( "\n" ));
        contour++;
      }

      AF_DUMP(( "  %5d  %5s %5s  %5s %5s  %s"
                " %5d %5d %7.2f %7.2f %7.2f %7.2f"
                " %5s %5s %5s %5s\n",
                point_idx,
                af_print_idx( buf1,
                              af_get_edge_index( hints, segment_idx_1, 1 ) ),
                af_print_idx( buf2, segment_idx_1 ),
                af_print_idx( buf3,
                              af_get_edge_index( hints, segment_idx_0, 0 ) ),
                af_print_idx( buf4, segment_idx_0 ),
                ( point->flags & AF_FLAG_NEAR )
                  ? " near "
                  : ( point->flags & AF_FLAG_WEAK_INTERPOLATION )
                    ? " weak "
                    : "strong",

                point->fx,
                point->fy,
                point->ox / 64.0,
                point->oy / 64.0,
                point->x / 64.0,
                point->y / 64.0,

                af_print_idx( buf5, af_get_strong_edge_index( hints,
                                                              point->before,
                                                              1 ) ),
                af_print_idx( buf6, af_get_strong_edge_index( hints,
                                                              point->after,
                                                              1 ) ),
                af_print_idx( buf7, af_get_strong_edge_index( hints,
                                                              point->before,
                                                              0 ) ),
                af_print_idx( buf8, af_get_strong_edge_index( hints,
                                                              point->after,
                                                              0 ) ) ));
    }
    AF_DUMP(( "\n" ));
  }
#ifdef __cplusplus
  }
#endif


  static const char*
  af_edge_flags_to_string( FT_UInt  flags )
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
  af_glyph_hints_dump_segments( AF_GlyphHints  hints,
                                FT_Bool        to_stdout )
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

      char  buf1[16], buf2[16], buf3[16];


      AF_DUMP(( "Table of %s segments:\n",
                dimension == AF_DIMENSION_HORZ ? "vertical"
                                               : "horizontal" ));
      if ( axis->num_segments )
      {
        AF_DUMP(( "  index   pos   delta   dir   from   to "
               /* "  XXXXX  XXXXX  XXXXX  XXXXX  XXXX  XXXX" */
                  "  link  serif  edge"
               /* "  XXXX  XXXXX  XXXX" */
                  "  height  extra     flags\n" ));
               /* "  XXXXXX  XXXXX  XXXXXXXXXXX" */
      }
      else
        AF_DUMP(( "  (none)\n" ));

      for ( seg = segments; seg < limit; seg++ )
        AF_DUMP(( "  %5d  %5d  %5d  %5s  %4d  %4d"
                  "  %4s  %5s  %4s"
                  "  %6d  %5d  %11s\n",
                  AF_INDEX_NUM( seg, segments ),
                  seg->pos,
                  seg->delta,
                  af_dir_str( (AF_Direction)seg->dir ),
                  AF_INDEX_NUM( seg->first, points ),
                  AF_INDEX_NUM( seg->last, points ),

                  af_print_idx( buf1, AF_INDEX_NUM( seg->link, segments ) ),
                  af_print_idx( buf2, AF_INDEX_NUM( seg->serif, segments ) ),
                  af_print_idx( buf3, AF_INDEX_NUM( seg->edge, edges ) ),

                  seg->height,
                  seg->height - ( seg->max_coord - seg->min_coord ),
                  af_edge_flags_to_string( seg->flags ) ));
      AF_DUMP(( "\n" ));
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
                                     FT_Pos        *offset,
                                     FT_Bool       *is_blue,
                                     FT_Pos        *blue_offset )
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

    seg      = &axis->segments[idx];
    *offset  = ( dim == AF_DIMENSION_HORZ ) ? seg->first->fx
                                            : seg->first->fy;
    if ( seg->edge )
      *is_blue = FT_BOOL( seg->edge->blue_edge );
    else
      *is_blue = FALSE;

    if ( *is_blue )
      *blue_offset = seg->edge->blue_edge->org;
    else
      *blue_offset = 0;

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
  af_glyph_hints_dump_edges( AF_GlyphHints  hints,
                             FT_Bool        to_stdout )
  {
    FT_Int  dimension;


    for ( dimension = 1; dimension >= 0; dimension-- )
    {
      AF_AxisHints  axis  = &hints->axis[dimension];
      AF_Edge       edges = axis->edges;
      AF_Edge       limit = edges + axis->num_edges;
      AF_Edge       edge;

      char  buf1[16], buf2[16];


      /*
       * note: AF_DIMENSION_HORZ corresponds to _vertical_ edges
       *       since they have a constant X coordinate.
       */
      if ( dimension == AF_DIMENSION_HORZ )
        AF_DUMP(( "Table of %s edges (1px=%.2fu, 10u=%.2fpx):\n",
                  "vertical",
                  65536.0 * 64.0 / hints->x_scale,
                  10.0 * hints->x_scale / 65536.0 / 64.0 ));
      else
        AF_DUMP(( "Table of %s edges (1px=%.2fu, 10u=%.2fpx):\n",
                  "horizontal",
                  65536.0 * 64.0 / hints->y_scale,
                  10.0 * hints->y_scale / 65536.0 / 64.0 ));

      if ( axis->num_edges )
      {
        AF_DUMP(( "  index    pos     dir   link  serif"
               /* "  XXXXX  XXXX.XX  XXXXX  XXXX  XXXXX" */
                  "  blue    opos     pos       flags\n" ));
               /* "    X   XXXX.XX  XXXX.XX  XXXXXXXXXXX" */
      }
      else
        AF_DUMP(( "  (none)\n" ));

      for ( edge = edges; edge < limit; edge++ )
        AF_DUMP(( "  %5d  %7.2f  %5s  %4s  %5s"
                  "    %c   %7.2f  %7.2f  %11s\n",
                  AF_INDEX_NUM( edge, edges ),
                  (int)edge->opos / 64.0,
                  af_dir_str( (AF_Direction)edge->dir ),
                  af_print_idx( buf1, AF_INDEX_NUM( edge->link, edges ) ),
                  af_print_idx( buf2, AF_INDEX_NUM( edge->serif, edges ) ),

                  edge->blue_edge ? 'y' : 'n',
                  edge->opos / 64.0,
                  edge->pos / 64.0,
                  af_edge_flags_to_string( edge->flags ) ));
      AF_DUMP(( "\n" ));
    }
  }
#ifdef __cplusplus
  }
#endif

#undef AF_DUMP

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
        ll  = -dy;
        ss  = dx;
      }
    }

    /* return no direction if arm lengths do not differ enough       */
    /* (value 14 is heuristic, corresponding to approx. 4.1 degrees) */
    /* the long arm is never negative                                */
    if ( ll <= 14 * FT_ABS( ss ) )
      dir = AF_DIR_NONE;

    return dir;
  }


  FT_LOCAL_DEF( void )
  af_glyph_hints_init( AF_GlyphHints  hints,
                       FT_Memory      memory )
  {
    /* no need to initialize the embedded items */
    FT_MEM_ZERO( hints, sizeof ( *hints ) - sizeof ( hints->embedded ) );
    hints->memory = memory;
  }


  FT_LOCAL_DEF( void )
  af_glyph_hints_done( AF_GlyphHints  hints )
  {
    FT_Memory  memory;
    int        dim;


    if ( !( hints && hints->memory ) )
      return;

    memory = hints->memory;

    /*
     * note that we don't need to free the segment and edge
     * buffers since they are really within the hints->points array
     */
    for ( dim = 0; dim < AF_DIMENSION_MAX; dim++ )
    {
      AF_AxisHints  axis = &hints->axis[dim];


      axis->num_segments = 0;
      axis->max_segments = 0;
      if ( axis->segments != axis->embedded.segments )
        FT_FREE( axis->segments );

      axis->num_edges = 0;
      axis->max_edges = 0;
      if ( axis->edges != axis->embedded.edges )
        FT_FREE( axis->edges );
    }

    if ( hints->contours != hints->embedded.contours )
      FT_FREE( hints->contours );
    hints->max_contours = 0;
    hints->num_contours = 0;

    if ( hints->points != hints->embedded.points )
      FT_FREE( hints->points );
    hints->max_points = 0;
    hints->num_points = 0;

    hints->memory = NULL;
  }


  /* Reset metrics. */

  FT_LOCAL_DEF( void )
  af_glyph_hints_rescale( AF_GlyphHints    hints,
                          AF_StyleMetrics  metrics )
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
    FT_Int     old_max, new_max;
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
    new_max = outline->n_contours;
    old_max = hints->max_contours;

    if ( new_max <= AF_CONTOURS_EMBEDDED )
    {
      if ( !hints->contours )
      {
        hints->contours     = hints->embedded.contours;
        hints->max_contours = AF_CONTOURS_EMBEDDED;
      }
    }
    else if ( new_max > old_max )
    {
      if ( hints->contours == hints->embedded.contours )
        hints->contours = NULL;

      new_max = ( new_max + 3 ) & ~3; /* round up to a multiple of 4 */

      if ( FT_RENEW_ARRAY( hints->contours, old_max, new_max ) )
        goto Exit;

      hints->max_contours = new_max;
    }

    /*
     * then reallocate the points arrays if necessary --
     * note that we reserve two additional point positions, used to
     * hint metrics appropriately
     */
    new_max = outline->n_points + 2;
    old_max = hints->max_points;

    if ( new_max <= AF_POINTS_EMBEDDED )
    {
      if ( !hints->points )
      {
        hints->points     = hints->embedded.points;
        hints->max_points = AF_POINTS_EMBEDDED;
      }
    }
    else if ( new_max > old_max )
    {
      if ( hints->points == hints->embedded.points )
        hints->points = NULL;

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

    points = hints->points;
    if ( hints->num_points == 0 )
      goto Exit;

    {
      AF_Point  point;
      AF_Point  point_limit = points + hints->num_points;

      /* value 20 in `near_limit' is heuristic */
      FT_UInt  units_per_em = hints->metrics->scaler.face->units_per_EM;
      FT_Int   near_limit   = 20 * units_per_em / 2048;


      /* compute coordinates & Bezier flags, next and prev */
      {
        FT_Vector*  vec           = outline->points;
        char*       tag           = outline->tags;
        FT_Short    endpoint      = outline->contours[0];
        AF_Point    end           = points + endpoint;
        AF_Point    prev          = end;
        FT_Int      contour_index = 0;


        for ( point = points; point < point_limit; point++, vec++, tag++ )
        {
          FT_Pos  out_x, out_y;


          point->in_dir  = (FT_Char)AF_DIR_NONE;
          point->out_dir = (FT_Char)AF_DIR_NONE;

          point->fx = (FT_Short)vec->x;
          point->fy = (FT_Short)vec->y;
          point->ox = point->x = FT_MulFix( vec->x, x_scale ) + x_delta;
          point->oy = point->y = FT_MulFix( vec->y, y_scale ) + y_delta;

          end->fx = (FT_Short)outline->points[endpoint].x;
          end->fy = (FT_Short)outline->points[endpoint].y;

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

          out_x = point->fx - prev->fx;
          out_y = point->fy - prev->fy;

          if ( FT_ABS( out_x ) + FT_ABS( out_y ) < near_limit )
            prev->flags |= AF_FLAG_NEAR;

          point->prev = prev;
          prev->next  = point;
          prev        = point;

          if ( point == end )
          {
            if ( ++contour_index < outline->n_contours )
            {
              endpoint = outline->contours[contour_index];
              end      = points + endpoint;
              prev     = end;
            }
          }

#ifdef FT_DEBUG_AUTOFIT
          point->before[0] = NULL;
          point->before[1] = NULL;
          point->after[0]  = NULL;
          point->after[1]  = NULL;
#endif

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

      {
        /*
         * Compute directions of `in' and `out' vectors.
         *
         * Note that distances between points that are very near to each
         * other are accumulated.  In other words, the auto-hinter either
         * prepends the small vectors between near points to the first
         * non-near vector, or the sum of small vector lengths exceeds a
         * threshold, thus `grouping' the small vectors.  All intermediate
         * points are tagged as weak; the directions are adjusted also to
         * be equal to the accumulated one.
         */

        FT_Int  near_limit2 = 2 * near_limit - 1;

        AF_Point*  contour;
        AF_Point*  contour_limit = hints->contours + hints->num_contours;


        for ( contour = hints->contours; contour < contour_limit; contour++ )
        {
          AF_Point  first = *contour;
          AF_Point  next, prev, curr;

          FT_Pos  out_x, out_y;


          /* since the first point of a contour could be part of a */
          /* series of near points, go backwards to find the first */
          /* non-near point and adjust `first'                     */

          point = first;
          prev  = first->prev;

          while ( prev != first )
          {
            out_x = point->fx - prev->fx;
            out_y = point->fy - prev->fy;

            /*
             * We use Taxicab metrics to measure the vector length.
             *
             * Note that the accumulated distances so far could have the
             * opposite direction of the distance measured here.  For this
             * reason we use `near_limit2' for the comparison to get a
             * non-near point even in the worst case.
             */
            if ( FT_ABS( out_x ) + FT_ABS( out_y ) >= near_limit2 )
              break;

            point = prev;
            prev  = prev->prev;
          }

          /* adjust first point */
          first = point;

          /* now loop over all points of the contour to get */
          /* `in' and `out' vector directions               */

          curr = first;

          /*
           * We abuse the `u' and `v' fields to store index deltas to the
           * next and previous non-near point, respectively.
           *
           * To avoid problems with not having non-near points, we point to
           * `first' by default as the next non-near point.
           *
           */
          curr->u  = (FT_Pos)( first - curr );
          first->v = -curr->u;

          out_x = 0;
          out_y = 0;

          next = first;
          do
          {
            AF_Direction  out_dir;


            point = next;
            next  = point->next;

            out_x += next->fx - point->fx;
            out_y += next->fy - point->fy;

            if ( FT_ABS( out_x ) + FT_ABS( out_y ) < near_limit )
            {
              next->flags |= AF_FLAG_WEAK_INTERPOLATION;
              continue;
            }

            curr->u = (FT_Pos)( next - curr );
            next->v = -curr->u;

            out_dir = af_direction_compute( out_x, out_y );

            /* adjust directions for all points inbetween; */
            /* the loop also updates position of `curr'    */
            curr->out_dir = (FT_Char)out_dir;
            for ( curr = curr->next; curr != next; curr = curr->next )
            {
              curr->in_dir  = (FT_Char)out_dir;
              curr->out_dir = (FT_Char)out_dir;
            }
            next->in_dir = (FT_Char)out_dir;

            curr->u  = (FT_Pos)( first - curr );
            first->v = -curr->u;

            out_x = 0;
            out_y = 0;

          } while ( next != first );
        }

        /*
         * The next step is to `simplify' an outline's topology so that we
         * can identify local extrema more reliably: A series of
         * non-horizontal or non-vertical vectors pointing into the same
         * quadrant are handled as a single, long vector.  From a
         * topological point of the view, the intermediate points are of no
         * interest and thus tagged as weak.
         */

        for ( point = points; point < point_limit; point++ )
        {
          if ( point->flags & AF_FLAG_WEAK_INTERPOLATION )
            continue;

          if ( point->in_dir  == AF_DIR_NONE &&
               point->out_dir == AF_DIR_NONE )
          {
            /* check whether both vectors point into the same quadrant */

            FT_Pos  in_x, in_y;
            FT_Pos  out_x, out_y;

            AF_Point  next_u = point + point->u;
            AF_Point  prev_v = point + point->v;


            in_x = point->fx - prev_v->fx;
            in_y = point->fy - prev_v->fy;

            out_x = next_u->fx - point->fx;
            out_y = next_u->fy - point->fy;

            if ( ( in_x ^ out_x ) >= 0 && ( in_y ^ out_y ) >= 0 )
            {
              /* yes, so tag current point as weak */
              /* and update index deltas           */

              point->flags |= AF_FLAG_WEAK_INTERPOLATION;

              prev_v->u = (FT_Pos)( next_u - prev_v );
              next_u->v = -prev_v->u;
            }
          }
        }

        /*
         * Finally, check for remaining weak points.  Everything else not
         * collected in edges so far is then implicitly classified as strong
         * points.
         */

        for ( point = points; point < point_limit; point++ )
        {
          if ( point->flags & AF_FLAG_WEAK_INTERPOLATION )
            continue;

          if ( point->flags & AF_FLAG_CONTROL )
          {
            /* control points are always weak */
          Is_Weak_Point:
            point->flags |= AF_FLAG_WEAK_INTERPOLATION;
          }
          else if ( point->out_dir == point->in_dir )
          {
            if ( point->out_dir != AF_DIR_NONE )
            {
              /* current point lies on a horizontal or          */
              /* vertical segment (but doesn't start or end it) */
              goto Is_Weak_Point;
            }

            {
              AF_Point  next_u = point + point->u;
              AF_Point  prev_v = point + point->v;


              if ( ft_corner_is_flat( point->fx  - prev_v->fx,
                                      point->fy  - prev_v->fy,
                                      next_u->fx - point->fx,
                                      next_u->fy - point->fy ) )
              {
                /* either the `in' or the `out' vector is much more  */
                /* dominant than the other one, so tag current point */
                /* as weak and update index deltas                   */

                prev_v->u = (FT_Pos)( next_u - prev_v );
                next_u->v = -prev_v->u;

                goto Is_Weak_Point;
              }
            }
          }
          else if ( point->in_dir == -point->out_dir )
          {
            /* current point forms a spike */
            goto Is_Weak_Point;
          }
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


        if ( !edge )
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


        if ( !edge )
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
    FT_UInt       touch_flag;


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

        if ( ( point->flags & AF_FLAG_WEAK_INTERPOLATION ) )
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

#ifdef FT_DEBUG_AUTOFIT
          point->before[dim] = edge;
          point->after[dim]  = NULL;
#endif

          goto Store_Point;
        }

        /* is the point after the last edge? */
        edge  = edge_limit - 1;
        delta = u - edge->fpos;
        if ( delta >= 0 )
        {
          u = edge->pos + ( ou - edge->opos );

#ifdef FT_DEBUG_AUTOFIT
          point->before[dim] = NULL;
          point->after[dim]  = edge;
#endif

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

#ifdef FT_DEBUG_AUTOFIT
              point->before[dim] = NULL;
              point->after[dim]  = NULL;
#endif

              goto Store_Point;
            }
          }

          /* point is not on an edge */
          {
            AF_Edge  before = edges + min - 1;
            AF_Edge  after  = edges + min + 0;


#ifdef FT_DEBUG_AUTOFIT
            point->before[dim] = before;
            point->after[dim]  = after;
#endif

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
    FT_Pos    u, v1, v2, u1, u2, d1, d2;


    if ( p1 > p2 )
      return;

    if ( ref1->v > ref2->v )
    {
      p    = ref1;
      ref1 = ref2;
      ref2 = p;
    }

    v1 = ref1->v;
    v2 = ref2->v;
    u1 = ref1->u;
    u2 = ref2->u;
    d1 = u1 - v1;
    d2 = u2 - v2;

    if ( u1 == u2 || v1 == v2 )
    {
      for ( p = p1; p <= p2; p++ )
      {
        u = p->v;

        if ( u <= v1 )
          u += d1;
        else if ( u >= v2 )
          u += d2;
        else
          u = u1;

        p->u = u;
      }
    }
    else
    {
      FT_Fixed  scale = FT_DivFix( u2 - u1, v2 - v1 );


      for ( p = p1; p <= p2; p++ )
      {
        u = p->v;

        if ( u <= v1 )
          u += d1;
        else if ( u >= v2 )
          u += d2;
        else
          u = u1 + FT_MulFix( u - v1, scale );

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
    FT_UInt    touch_flag;
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


/* END */
