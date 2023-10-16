/****************************************************************************
 *
 * afhints.h
 *
 *   Auto-fitter hinting routines (specification).
 *
 * Copyright (C) 2003-2023 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#ifndef AFHINTS_H_
#define AFHINTS_H_

#include "aftypes.h"

FT_BEGIN_HEADER

  /*
   * The definition of outline glyph hints.  These are shared by all
   * writing system analysis routines (until now).
   */

  typedef enum  AF_Dimension_
  {
    AF_DIMENSION_HORZ = 0,  /* x coordinates,                    */
                            /* i.e., vertical segments & edges   */
    AF_DIMENSION_VERT = 1,  /* y coordinates,                    */
                            /* i.e., horizontal segments & edges */

    AF_DIMENSION_MAX  /* do not remove */

  } AF_Dimension;


  /* hint directions -- the values are computed so that two vectors are */
  /* in opposite directions iff `dir1 + dir2 == 0'                      */
  typedef enum  AF_Direction_
  {
    AF_DIR_NONE  =  4,
    AF_DIR_RIGHT =  1,
    AF_DIR_LEFT  = -1,
    AF_DIR_UP    =  2,
    AF_DIR_DOWN  = -2

  } AF_Direction;


  /*
   * The following explanations are mostly taken from the article
   *
   *   Real-Time Grid Fitting of Typographic Outlines
   *
   * by David Turner and Werner Lemberg
   *
   *   https://www.tug.org/TUGboat/Articles/tb24-3/lemberg.pdf
   *
   * with appropriate updates.
   *
   *
   * Segments
   *
   *   `af_{cjk,latin,...}_hints_compute_segments' are the functions to
   *   find segments in an outline.
   *
   *   A segment is a series of at least two consecutive points that are
   *   approximately aligned along a coordinate axis.  The analysis to do
   *   so is specific to a writing system.
   *
   *
   * Edges
   *
   *   `af_{cjk,latin,...}_hints_compute_edges' are the functions to find
   *   edges.
   *
   *   As soon as segments are defined, the auto-hinter groups them into
   *   edges.  An edge corresponds to a single position on the main
   *   dimension that collects one or more segments (allowing for a small
   *   threshold).
   *
   *   As an example, the `latin' writing system first tries to grid-fit
   *   edges, then to align segments on the edges unless it detects that
   *   they form a serif.
   *
   *
   *                     A          H
   *                      |        |
   *                      |        |
   *                      |        |
   *                      |        |
   *        C             |        |             F
   *         +------<-----+        +-----<------+
   *         |             B      G             |
   *         |                                  |
   *         |                                  |
   *         +--------------->------------------+
   *        D                                    E
   *
   *
   * Stems
   *
   *   Stems are detected by `af_{cjk,latin,...}_hint_edges'.
   *
   *   Segments need to be `linked' to other ones in order to detect stems.
   *   A stem is made of two segments that face each other in opposite
   *   directions and that are sufficiently close to each other.  Using
   *   vocabulary from the TrueType specification, stem segments form a
   *   `black distance'.
   *
   *   In the above ASCII drawing, the horizontal segments are BC, DE, and
   *   FG; the vertical segments are AB, CD, EF, and GH.
   *
   *   Each segment has at most one `best' candidate to form a black
   *   distance, or no candidate at all.  Notice that two distinct segments
   *   can have the same candidate, which frequently means a serif.
   *
   *   A stem is recognized by the following condition:
   *
   *     best segment_1 = segment_2 && best segment_2 = segment_1
   *
   *   The best candidate is stored in field `link' in structure
   *   `AF_Segment'.
   *
   *   In the above ASCII drawing, the best candidate for both AB and CD is
   *   GH, while the best candidate for GH is AB.  Similarly, the best
   *   candidate for EF and GH is AB, while the best candidate for AB is
   *   GH.
   *
   *   The detection and handling of stems is dependent on the writing
   *   system.
   *
   *
   * Serifs
   *
   *   Serifs are detected by `af_{cjk,latin,...}_hint_edges'.
   *
   *   In comparison to a stem, a serif (as handled by the auto-hinter
   *   module that takes care of the `latin' writing system) has
   *
   *     best segment_1 = segment_2 && best segment_2 != segment_1
   *
   *   where segment_1 corresponds to the serif segment (CD and EF in the
   *   above ASCII drawing).
   *
   *   The best candidate is stored in field `serif' in structure
   *   `AF_Segment' (and `link' is set to NULL).
   *
   *
   * Touched points
   *
   *   A point is called `touched' if it has been processed somehow by the
   *   auto-hinter.  It basically means that it shouldn't be moved again
   *   (or moved only under certain constraints to preserve the already
   *   applied processing).
   *
   *
   * Flat and round segments
   *
   *   Segments are `round' or `flat', depending on the series of points
   *   that define them.  A segment is round if the next and previous point
   *   of an extremum (which can be either a single point or sequence of
   *   points) are both conic or cubic control points.  Otherwise, a
   *   segment with an extremum is flat.
   *
   *
   * Strong Points
   *
   *   Experience has shown that points not part of an edge need to be
   *   interpolated linearly between their two closest edges, even if these
   *   are not part of the contour of those particular points.  Typical
   *   candidates for this are
   *
   *   - angle points (i.e., points where the `in' and `out' direction
   *     differ greatly)
   *
   *   - inflection points (i.e., where the `in' and `out' angles are the
   *     same, but the curvature changes sign) [currently, such points
   *     aren't handled specially in the auto-hinter]
   *
   *   `af_glyph_hints_align_strong_points' is the function that takes
   *   care of such situations; it is equivalent to the TrueType `IP'
   *   hinting instruction.
   *
   *
   * Weak Points
   *
   *   Other points in the outline must be interpolated using the
   *   coordinates of their previous and next unfitted contour neighbours.
   *   These are called `weak points' and are touched by the function
   *   `af_glyph_hints_align_weak_points', equivalent to the TrueType `IUP'
   *   hinting instruction.  Typical candidates are control points and
   *   points on the contour without a major direction.
   *
   *   The major effect is to reduce possible distortion caused by
   *   alignment of edges and strong points, thus weak points are processed
   *   after strong points.
   */


  /* point hint flags */
#define AF_FLAG_NONE  0

  /* point type flags */
#define AF_FLAG_CONIC    ( 1U << 0 )
#define AF_FLAG_CUBIC    ( 1U << 1 )
#define AF_FLAG_CONTROL  ( AF_FLAG_CONIC | AF_FLAG_CUBIC )

  /* point touch flags */
#define AF_FLAG_TOUCH_X  ( 1U << 2 )
#define AF_FLAG_TOUCH_Y  ( 1U << 3 )

  /* candidates for weak interpolation have this flag set */
#define AF_FLAG_WEAK_INTERPOLATION  ( 1U << 4 )

  /* the distance to the next point is very small */
#define AF_FLAG_NEAR  ( 1U << 5 )


  /* edge hint flags */
#define AF_EDGE_NORMAL  0
#define AF_EDGE_ROUND    ( 1U << 0 )
#define AF_EDGE_SERIF    ( 1U << 1 )
#define AF_EDGE_DONE     ( 1U << 2 )
#define AF_EDGE_NEUTRAL  ( 1U << 3 ) /* edge aligns to a neutral blue zone */


  typedef struct AF_PointRec_*    AF_Point;
  typedef struct AF_SegmentRec_*  AF_Segment;
  typedef struct AF_EdgeRec_*     AF_Edge;


  typedef struct  AF_PointRec_
  {
    FT_UShort  flags;    /* point flags used by hinter   */
    FT_Char    in_dir;   /* direction of inwards vector  */
    FT_Char    out_dir;  /* direction of outwards vector */

    FT_Pos     ox, oy;   /* original, scaled position                   */
    FT_Short   fx, fy;   /* original, unscaled position (in font units) */
    FT_Pos     x, y;     /* current position                            */
    FT_Pos     u, v;     /* current (x,y) or (y,x) depending on context */

    AF_Point   next;     /* next point in contour     */
    AF_Point   prev;     /* previous point in contour */

#ifdef FT_DEBUG_AUTOFIT
    /* track `before' and `after' edges for strong points */
    AF_Edge    before[2];
    AF_Edge    after[2];
#endif

  } AF_PointRec;


  typedef struct  AF_SegmentRec_
  {
    FT_Byte     flags;       /* edge/segment flags for this segment */
    FT_Char     dir;         /* segment direction                   */
    FT_Short    pos;         /* position of segment                 */
    FT_Short    delta;       /* deviation from segment position     */
    FT_Short    min_coord;   /* minimum coordinate of segment       */
    FT_Short    max_coord;   /* maximum coordinate of segment       */
    FT_Short    height;      /* the hinted segment height           */

    AF_Edge     edge;        /* the segment's parent edge           */
    AF_Segment  edge_next;   /* link to next segment in parent edge */

    AF_Segment  link;        /* (stem) link segment        */
    AF_Segment  serif;       /* primary segment for serifs */
    FT_Pos      score;       /* used during stem matching  */
    FT_Pos      len;         /* used during stem matching  */

    AF_Point    first;       /* first point in edge segment */
    AF_Point    last;        /* last point in edge segment  */

  } AF_SegmentRec;


  typedef struct  AF_EdgeRec_
  {
    FT_Short    fpos;       /* original, unscaled position (in font units) */
    FT_Pos      opos;       /* original, scaled position                   */
    FT_Pos      pos;        /* current position                            */

    FT_Byte     flags;      /* edge flags                                   */
    FT_Char     dir;        /* edge direction                               */
    FT_Fixed    scale;      /* used to speed up interpolation between edges */

    AF_Width    blue_edge;  /* non-NULL if this is a blue edge */
    AF_Edge     link;       /* link edge                       */
    AF_Edge     serif;      /* primary edge for serifs         */
    FT_Int      score;      /* used during stem matching       */

    AF_Segment  first;      /* first segment in edge */
    AF_Segment  last;       /* last segment in edge  */

  } AF_EdgeRec;

#define AF_SEGMENTS_EMBEDDED  18   /* number of embedded segments   */
#define AF_EDGES_EMBEDDED     12   /* number of embedded edges      */

  typedef struct  AF_AxisHintsRec_
  {
    FT_UInt       num_segments; /* number of used segments      */
    FT_UInt       max_segments; /* number of allocated segments */
    AF_Segment    segments;     /* segments array               */

    FT_UInt       num_edges;    /* number of used edges      */
    FT_UInt       max_edges;    /* number of allocated edges */
    AF_Edge       edges;        /* edges array               */

    AF_Direction  major_dir;    /* either vertical or horizontal */

    /* two arrays to avoid allocation penalty */
    struct
    {
      AF_SegmentRec  segments[AF_SEGMENTS_EMBEDDED];
      AF_EdgeRec     edges[AF_EDGES_EMBEDDED];
    } embedded;


  } AF_AxisHintsRec, *AF_AxisHints;


#define AF_POINTS_EMBEDDED     96   /* number of embedded points   */
#define AF_CONTOURS_EMBEDDED    8   /* number of embedded contours */

  typedef struct  AF_GlyphHintsRec_
  {
    FT_Memory        memory;

    FT_Fixed         x_scale;
    FT_Pos           x_delta;

    FT_Fixed         y_scale;
    FT_Pos           y_delta;

    FT_Int           max_points;    /* number of allocated points */
    FT_Int           num_points;    /* number of used points      */
    AF_Point         points;        /* points array               */

    FT_Int           max_contours;  /* number of allocated contours */
    FT_Int           num_contours;  /* number of used contours      */
    AF_Point*        contours;      /* contours array               */

    AF_AxisHintsRec  axis[AF_DIMENSION_MAX];

    FT_UInt32        scaler_flags;  /* copy of scaler flags    */
    FT_UInt32        other_flags;   /* free for style-specific */
                                    /* implementations         */
    AF_StyleMetrics  metrics;

    /* Two arrays to avoid allocation penalty.            */
    /* The `embedded' structure must be the last element! */
    struct
    {
      AF_Point       contours[AF_CONTOURS_EMBEDDED];
      AF_PointRec    points[AF_POINTS_EMBEDDED];
    } embedded;

  } AF_GlyphHintsRec;


#define AF_HINTS_TEST_SCALER( h, f )  ( (h)->scaler_flags & (f) )
#define AF_HINTS_TEST_OTHER( h, f )   ( (h)->other_flags  & (f) )


#ifdef FT_DEBUG_AUTOFIT

#define AF_HINTS_DO_HORIZONTAL( h )                                     \
          ( !af_debug_disable_horz_hints_                            && \
            !AF_HINTS_TEST_SCALER( h, AF_SCALER_FLAG_NO_HORIZONTAL ) )

#define AF_HINTS_DO_VERTICAL( h )                                     \
          ( !af_debug_disable_vert_hints_                          && \
            !AF_HINTS_TEST_SCALER( h, AF_SCALER_FLAG_NO_VERTICAL ) )

#define AF_HINTS_DO_BLUES( h )  ( !af_debug_disable_blue_hints_ )

#else /* !FT_DEBUG_AUTOFIT */

#define AF_HINTS_DO_HORIZONTAL( h )                                \
          !AF_HINTS_TEST_SCALER( h, AF_SCALER_FLAG_NO_HORIZONTAL )

#define AF_HINTS_DO_VERTICAL( h )                                \
          !AF_HINTS_TEST_SCALER( h, AF_SCALER_FLAG_NO_VERTICAL )

#define AF_HINTS_DO_BLUES( h )  1

#endif /* !FT_DEBUG_AUTOFIT */


#define AF_HINTS_DO_ADVANCE( h )                                \
          !AF_HINTS_TEST_SCALER( h, AF_SCALER_FLAG_NO_ADVANCE )


  FT_LOCAL( AF_Direction )
  af_direction_compute( FT_Pos  dx,
                        FT_Pos  dy );


  FT_LOCAL( FT_Error )
  af_axis_hints_new_segment( AF_AxisHints  axis,
                             FT_Memory     memory,
                             AF_Segment   *asegment );

  FT_LOCAL( FT_Error)
  af_axis_hints_new_edge( AF_AxisHints  axis,
                          FT_Int        fpos,
                          AF_Direction  dir,
                          FT_Bool       top_to_bottom_hinting,
                          FT_Memory     memory,
                          AF_Edge      *edge );

  FT_LOCAL( void )
  af_glyph_hints_init( AF_GlyphHints  hints,
                       FT_Memory      memory );

  FT_LOCAL( void )
  af_glyph_hints_rescale( AF_GlyphHints    hints,
                          AF_StyleMetrics  metrics );

  FT_LOCAL( FT_Error )
  af_glyph_hints_reload( AF_GlyphHints  hints,
                         FT_Outline*    outline );

  FT_LOCAL( void )
  af_glyph_hints_save( AF_GlyphHints  hints,
                       FT_Outline*    outline );

  FT_LOCAL( void )
  af_glyph_hints_align_edge_points( AF_GlyphHints  hints,
                                    AF_Dimension   dim );

  FT_LOCAL( void )
  af_glyph_hints_align_strong_points( AF_GlyphHints  hints,
                                      AF_Dimension   dim );

  FT_LOCAL( void )
  af_glyph_hints_align_weak_points( AF_GlyphHints  hints,
                                    AF_Dimension   dim );

  FT_LOCAL( void )
  af_glyph_hints_done( AF_GlyphHints  hints );

/* */

#define AF_SEGMENT_LEN( seg )          ( (seg)->max_coord - (seg)->min_coord )

#define AF_SEGMENT_DIST( seg1, seg2 )  ( ( (seg1)->pos > (seg2)->pos )   \
                                           ? (seg1)->pos - (seg2)->pos   \
                                           : (seg2)->pos - (seg1)->pos )


FT_END_HEADER

#endif /* AFHINTS_H_ */


/* END */
