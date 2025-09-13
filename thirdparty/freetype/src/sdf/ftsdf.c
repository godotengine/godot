/****************************************************************************
 *
 * ftsdf.c
 *
 *   Signed Distance Field support for outline fonts (body).
 *
 * Copyright (C) 2020-2024 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * Written by Anuj Verma.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#include <freetype/internal/ftobjs.h>
#include <freetype/internal/ftdebug.h>
#include <freetype/ftoutln.h>
#include <freetype/fttrigon.h>
#include <freetype/ftbitmap.h>
#include "ftsdf.h"

#include "ftsdferrs.h"


  /**************************************************************************
   *
   * A brief technical overview of how the SDF rasterizer works
   * ----------------------------------------------------------
   *
   * [Notes]:
   *   * SDF stands for Signed Distance Field everywhere.
   *
   *   * This renderer generates SDF directly from outlines.  There is
   *     another renderer called 'bsdf', which converts bitmaps to SDF; see
   *     file `ftbsdf.c` for more.
   *
   *   * The basic idea of generating the SDF is taken from Viktor Chlumsky's
   *     research paper.  The paper explains both single and multi-channel
   *     SDF, however, this implementation only generates single-channel SDF.
   *
   *       Chlumsky, Viktor: Shape Decomposition for Multi-channel Distance
   *       Fields.  Master's thesis.  Czech Technical University in Prague,
   *       Faculty of InformationTechnology, 2015.
   *
   *     For more information: https://github.com/Chlumsky/msdfgen
   *
   * ========================================================================
   *
   * Generating SDF from outlines is pretty straightforward.
   *
   * (1) We have a set of contours that make the outline of a shape/glyph.
   *     Each contour comprises of several edges, with three types of edges.
   *
   *     * line segments
   *     * conic Bezier curves
   *     * cubic Bezier curves
   *
   * (2) Apart from the outlines we also have a two-dimensional grid, namely
   *     the bitmap that is used to represent the final SDF data.
   *
   * (3) In order to generate SDF, our task is to find shortest signed
   *     distance from each grid point to the outline.  The 'signed
   *     distance' means that if the grid point is filled by any contour
   *     then its sign is positive, otherwise it is negative.  The pseudo
   *     code is as follows.
   *
   *     ```
   *     foreach grid_point (x, y):
   *     {
   *       int min_dist = INT_MAX;
   *
   *       foreach contour in outline:
   *       {
   *         foreach edge in contour:
   *         {
   *           // get shortest distance from point (x, y) to the edge
   *           d = get_min_dist(x, y, edge);
   *
   *           if (d < min_dist)
   *             min_dist = d;
   *         }
   *
   *         bitmap[x, y] = min_dist;
   *       }
   *     }
   *     ```
   *
   * (4) After running this algorithm the bitmap contains information about
   *     the shortest distance from each point to the outline of the shape.
   *     Of course, while this is the most straightforward way of generating
   *     SDF, we use various optimizations in our implementation.  See the
   *     `sdf_generate_*' functions in this file for all details.
   *
   *     The optimization currently used by default is subdivision; see
   *     function `sdf_generate_subdivision` for more.
   *
   *     Also, to see how we compute the shortest distance from a point to
   *     each type of edge, check out the `get_min_distance_*' functions.
   *
   */


  /**************************************************************************
   *
   * The macro FT_COMPONENT is used in trace mode.  It is an implicit
   * parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log
   * messages during execution.
   */
#undef  FT_COMPONENT
#define FT_COMPONENT  sdf


  /**************************************************************************
   *
   * definitions
   *
   */

  /*
   * If set to 1, the rasterizer uses Newton-Raphson's method for finding
   * the shortest distance from a point to a conic curve.
   *
   * If set to 0, an analytical method gets used instead, which computes the
   * roots of a cubic polynomial to find the shortest distance.  However,
   * the analytical method can currently underflow; we thus use Newton's
   * method by default.
   */
#ifndef USE_NEWTON_FOR_CONIC
#define USE_NEWTON_FOR_CONIC  1
#endif

  /*
   * The number of intervals a Bezier curve gets sampled and checked to find
   * the shortest distance.
   */
#define MAX_NEWTON_DIVISIONS  4

  /*
   * The number of steps of Newton's iterations in each interval of the
   * Bezier curve.  Basically, we run Newton's approximation
   *
   *   x -= Q(t) / Q'(t)
   *
   * for each division to get the shortest distance.
   */
#define MAX_NEWTON_STEPS  4

  /*
   * The epsilon distance (in 16.16 fractional units) used for corner
   * resolving.  If the difference of two distances is less than this value
   * they will be checked for a corner if they are ambiguous.
   */
#define CORNER_CHECK_EPSILON  32

#if 0
  /*
   * Coarse grid dimension.  Will probably be removed in the future because
   * coarse grid optimization is the slowest algorithm.
   */
#define CG_DIMEN  8
#endif


  /**************************************************************************
   *
   * macros
   *
   */

#define MUL_26D6( a, b )  ( ( ( a ) * ( b ) ) / 64 )
#define VEC_26D6_DOT( p, q )  ( MUL_26D6( p.x, q.x ) + \
                                MUL_26D6( p.y, q.y ) )


  /**************************************************************************
   *
   * structures and enums
   *
   */

  /**************************************************************************
   *
   * @Struct:
   *   SDF_TRaster
   *
   * @Description:
   *   This struct is used in place of @FT_Raster and is stored within the
   *   internal FreeType renderer struct.  While rasterizing it is passed to
   *   the @FT_Raster_RenderFunc function, which then can be used however we
   *   want.
   *
   * @Fields:
   *   memory ::
   *     Used internally to allocate intermediate memory while raterizing.
   *
   */
  typedef struct  SDF_TRaster_
  {
    FT_Memory  memory;

  } SDF_TRaster, *SDF_PRaster;


  /**************************************************************************
   *
   * @Enum:
   *   SDF_Edge_Type
   *
   * @Description:
   *   Enumeration of all curve types present in fonts.
   *
   * @Fields:
   *   SDF_EDGE_UNDEFINED ::
   *     Undefined edge, simply used to initialize and detect errors.
   *
   *   SDF_EDGE_LINE ::
   *     Line segment with start and end point.
   *
   *   SDF_EDGE_CONIC ::
   *     A conic/quadratic Bezier curve with start, end, and one control
   *     point.
   *
   *   SDF_EDGE_CUBIC ::
   *     A cubic Bezier curve with start, end, and two control points.
   *
   */
  typedef enum  SDF_Edge_Type_
  {
    SDF_EDGE_UNDEFINED = 0,
    SDF_EDGE_LINE      = 1,
    SDF_EDGE_CONIC     = 2,
    SDF_EDGE_CUBIC     = 3

  } SDF_Edge_Type;


  /**************************************************************************
   *
   * @Enum:
   *   SDF_Contour_Orientation
   *
   * @Description:
   *   Enumeration of all orientation values of a contour.  We determine the
   *   orientation by calculating the area covered by a contour.  Contrary
   *   to values returned by @FT_Outline_Get_Orientation,
   *   `SDF_Contour_Orientation` is independent of the fill rule, which can
   *   be different for different font formats.
   *
   * @Fields:
   *   SDF_ORIENTATION_NONE ::
   *     Undefined orientation, used for initialization and error detection.
   *
   *   SDF_ORIENTATION_CW ::
   *     Clockwise orientation (positive area covered).
   *
   *   SDF_ORIENTATION_CCW ::
   *     Counter-clockwise orientation (negative area covered).
   *
   * @Note:
   *   See @FT_Outline_Get_Orientation for more details.
   *
   */
  typedef enum  SDF_Contour_Orientation_
  {
    SDF_ORIENTATION_NONE = 0,
    SDF_ORIENTATION_CW   = 1,
    SDF_ORIENTATION_CCW  = 2

  } SDF_Contour_Orientation;


  /**************************************************************************
   *
   * @Struct:
   *   SDF_Edge
   *
   * @Description:
   *   Represent an edge of a contour.
   *
   * @Fields:
   *   start_pos ::
   *     Start position of an edge.  Valid for all types of edges.
   *
   *   end_pos ::
   *     Etart position of an edge.  Valid for all types of edges.
   *
   *   control_a ::
   *     A control point of the edge.  Valid only for `SDF_EDGE_CONIC`
   *     and `SDF_EDGE_CUBIC`.
   *
   *   control_b ::
   *     Another control point of the edge.  Valid only for
   *     `SDF_EDGE_CONIC`.
   *
   *   edge_type ::
   *     Type of the edge, see @SDF_Edge_Type for all possible edge types.
   *
   *   next ::
   *     Used to create a singly linked list, which can be interpreted
   *     as a contour.
   *
   */
  typedef struct  SDF_Edge_
  {
    FT_26D6_Vec  start_pos;
    FT_26D6_Vec  end_pos;
    FT_26D6_Vec  control_a;
    FT_26D6_Vec  control_b;

    SDF_Edge_Type  edge_type;

    struct SDF_Edge_*  next;

  } SDF_Edge;


  /**************************************************************************
   *
   * @Struct:
   *   SDF_Contour
   *
   * @Description:
   *   Represent a complete contour, which contains a list of edges.
   *
   * @Fields:
   *   last_pos ::
   *     Contains the value of `end_pos' of the last edge in the list of
   *     edges.  Useful while decomposing the outline with
   *     @FT_Outline_Decompose.
   *
   *   edges ::
   *     Linked list of all the edges that make the contour.
   *
   *   next ::
   *     Used to create a singly linked list, which can be interpreted as a
   *     complete shape or @FT_Outline.
   *
   */
  typedef struct  SDF_Contour_
  {
    FT_26D6_Vec  last_pos;
    SDF_Edge*    edges;

    struct SDF_Contour_*  next;

  } SDF_Contour;


  /**************************************************************************
   *
   * @Struct:
   *   SDF_Shape
   *
   * @Description:
   *   Represent a complete shape, which is the decomposition of
   *   @FT_Outline.
   *
   * @Fields:
   *   memory ::
   *     Used internally to allocate memory.
   *
   *   contours ::
   *     Linked list of all the contours that make the shape.
   *
   */
  typedef struct  SDF_Shape_
  {
    FT_Memory     memory;
    SDF_Contour*  contours;

  } SDF_Shape;


  /**************************************************************************
   *
   * @Struct:
   *   SDF_Signed_Distance
   *
   * @Description:
   *   Represent signed distance of a point, i.e., the distance of the edge
   *   nearest to the point.
   *
   * @Fields:
   *   distance ::
   *     Distance of the point from the nearest edge.  Can be squared or
   *     absolute depending on the `USE_SQUARED_DISTANCES` macro defined in
   *     file `ftsdfcommon.h`.
   *
   *   cross ::
   *     Cross product of the shortest distance vector (i.e., the vector
   *     from the point to the nearest edge) and the direction of the edge
   *     at the nearest point.  This is used to resolve ambiguities of
   *     `sign`.
   *
   *   sign ::
   *     A value used to indicate whether the distance vector is outside or
   *     inside the contour corresponding to the edge.
   *
   * @Note:
   *   `sign` may or may not be correct, therefore it must be checked
   *   properly in case there is an ambiguity.
   *
   */
  typedef struct SDF_Signed_Distance_
  {
    FT_16D16  distance;
    FT_16D16  cross;
    FT_Char   sign;

  } SDF_Signed_Distance;


  /**************************************************************************
   *
   * @Struct:
   *   SDF_Params
   *
   * @Description:
   *   Yet another internal parameters required by the rasterizer.
   *
   * @Fields:
   *   orientation ::
   *     This is not the @SDF_Contour_Orientation value but @FT_Orientation,
   *     which determines whether clockwise-oriented outlines are to be
   *     filled or counter-clockwise-oriented ones.
   *
   *   flip_sign ::
   *     If set to true, flip the sign.  By default the points filled by the
   *     outline are positive.
   *
   *   flip_y ::
   *     If set to true the output bitmap is upside-down.  Can be useful
   *     because OpenGL and DirectX use different coordinate systems for
   *     textures.
   *
   *   overload_sign ::
   *     In the subdivision and bounding box optimization, the default
   *     outside sign is taken as -1.  This parameter can be used to modify
   *     that behaviour.  For example, while generating SDF for a single
   *     counter-clockwise contour, the outside sign should be 1.
   *
   */
  typedef struct SDF_Params_
  {
    FT_Orientation  orientation;
    FT_Bool         flip_sign;
    FT_Bool         flip_y;

    FT_Int  overload_sign;

  } SDF_Params;


  /**************************************************************************
   *
   * constants, initializer, and destructor
   *
   */

  static
  const FT_Vector  zero_vector = { 0, 0 };

  static
  const SDF_Edge  null_edge = { { 0, 0 }, { 0, 0 },
                                { 0, 0 }, { 0, 0 },
                                SDF_EDGE_UNDEFINED, NULL };

  static
  const SDF_Contour  null_contour = { { 0, 0 }, NULL, NULL };

  static
  const SDF_Shape  null_shape = { NULL, NULL };

  static
  const SDF_Signed_Distance  max_sdf = { INT_MAX, 0, 0 };


  /* Create a new @SDF_Edge on the heap and assigns the `edge` */
  /* pointer to the newly allocated memory.                    */
  static FT_Error
  sdf_edge_new( FT_Memory   memory,
                SDF_Edge**  edge )
  {
    FT_Error   error = FT_Err_Ok;
    SDF_Edge*  ptr   = NULL;


    if ( !memory || !edge )
    {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    if ( !FT_QNEW( ptr ) )
    {
      *ptr = null_edge;
      *edge = ptr;
    }

  Exit:
    return error;
  }


  /* Free the allocated `edge` variable. */
  static void
  sdf_edge_done( FT_Memory   memory,
                 SDF_Edge**  edge )
  {
    if ( !memory || !edge || !*edge )
      return;

    FT_FREE( *edge );
  }


  /* Create a new @SDF_Contour on the heap and assign     */
  /* the `contour` pointer to the newly allocated memory. */
  static FT_Error
  sdf_contour_new( FT_Memory      memory,
                   SDF_Contour**  contour )
  {
    FT_Error      error = FT_Err_Ok;
    SDF_Contour*  ptr   = NULL;


    if ( !memory || !contour )
    {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    if ( !FT_QNEW( ptr ) )
    {
      *ptr     = null_contour;
      *contour = ptr;
    }

  Exit:
    return error;
  }


  /* Free the allocated `contour` variable. */
  /* Also free the list of edges.           */
  static void
  sdf_contour_done( FT_Memory      memory,
                    SDF_Contour**  contour )
  {
    SDF_Edge*  edges;
    SDF_Edge*  temp;


    if ( !memory || !contour || !*contour )
      return;

    edges = (*contour)->edges;

    /* release all edges */
    while ( edges )
    {
      temp  = edges;
      edges = edges->next;

      sdf_edge_done( memory, &temp );
    }

    FT_FREE( *contour );
  }


  /* Create a new @SDF_Shape on the heap and assign     */
  /* the `shape` pointer to the newly allocated memory. */
  static FT_Error
  sdf_shape_new( FT_Memory    memory,
                 SDF_Shape**  shape )
  {
    FT_Error    error = FT_Err_Ok;
    SDF_Shape*  ptr   = NULL;


    if ( !memory || !shape )
    {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    if ( !FT_QNEW( ptr ) )
    {
      *ptr        = null_shape;
      ptr->memory = memory;
      *shape      = ptr;
    }

  Exit:
    return error;
  }


  /* Free the allocated `shape` variable. */
  /* Also free the list of contours.      */
  static void
  sdf_shape_done( SDF_Shape**  shape )
  {
    FT_Memory     memory;
    SDF_Contour*  contours;
    SDF_Contour*  temp;


    if ( !shape || !*shape )
      return;

    memory   = (*shape)->memory;
    contours = (*shape)->contours;

    if ( !memory )
      return;

    /* release all contours */
    while ( contours )
    {
      temp     = contours;
      contours = contours->next;

      sdf_contour_done( memory, &temp );
    }

    /* release the allocated shape struct  */
    FT_FREE( *shape );
  }


  /**************************************************************************
   *
   * shape decomposition functions
   *
   */

  /* This function is called when starting a new contour at `to`, */
  /* which gets added to the shape's list.                        */
  static FT_Error
  sdf_move_to( const FT_26D6_Vec* to,
               void*              user )
  {
    SDF_Shape*    shape   = ( SDF_Shape* )user;
    SDF_Contour*  contour = NULL;

    FT_Error   error  = FT_Err_Ok;
    FT_Memory  memory = shape->memory;


    if ( !to || !user )
    {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    FT_CALL( sdf_contour_new( memory, &contour ) );

    contour->last_pos = *to;
    contour->next     = shape->contours;
    shape->contours   = contour;

  Exit:
    return error;
  }


  /* This function is called when there is a line in the      */
  /* contour.  The line starts at the previous edge point and */
  /* stops at `to`.                                           */
  static FT_Error
  sdf_line_to( const FT_26D6_Vec*  to,
               void*               user )
  {
    SDF_Shape*    shape    = ( SDF_Shape* )user;
    SDF_Edge*     edge     = NULL;
    SDF_Contour*  contour  = NULL;

    FT_Error      error    = FT_Err_Ok;
    FT_Memory     memory   = shape->memory;


    if ( !to || !user )
    {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    contour = shape->contours;

    if ( contour->last_pos.x == to->x &&
         contour->last_pos.y == to->y )
      goto Exit;

    FT_CALL( sdf_edge_new( memory, &edge ) );

    edge->edge_type = SDF_EDGE_LINE;
    edge->start_pos = contour->last_pos;
    edge->end_pos   = *to;

    edge->next        = contour->edges;
    contour->edges    = edge;
    contour->last_pos = *to;

  Exit:
    return error;
  }


  /* This function is called when there is a conic Bezier curve   */
  /* in the contour.  The curve starts at the previous edge point */
  /* and stops at `to`, with control point `control_1`.           */
  static FT_Error
  sdf_conic_to( const FT_26D6_Vec*  control_1,
                const FT_26D6_Vec*  to,
                void*               user )
  {
    SDF_Shape*    shape    = ( SDF_Shape* )user;
    SDF_Edge*     edge     = NULL;
    SDF_Contour*  contour  = NULL;

    FT_Error   error  = FT_Err_Ok;
    FT_Memory  memory = shape->memory;


    if ( !control_1 || !to || !user )
    {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    contour = shape->contours;

    /* If the control point coincides with any of the end points */
    /* then it is a line and should be treated as one to avoid   */
    /* unnecessary complexity later in the algorithm.            */
    if ( ( contour->last_pos.x == control_1->x &&
           contour->last_pos.y == control_1->y ) ||
         ( control_1->x == to->x &&
           control_1->y == to->y )               )
    {
      sdf_line_to( to, user );
      goto Exit;
    }

    FT_CALL( sdf_edge_new( memory, &edge ) );

    edge->edge_type = SDF_EDGE_CONIC;
    edge->start_pos = contour->last_pos;
    edge->control_a = *control_1;
    edge->end_pos   = *to;

    edge->next        = contour->edges;
    contour->edges    = edge;
    contour->last_pos = *to;

  Exit:
    return error;
  }


  /* This function is called when there is a cubic Bezier curve   */
  /* in the contour.  The curve starts at the previous edge point */
  /* and stops at `to`, with two control points `control_1` and   */
  /* `control_2`.                                                 */
  static FT_Error
  sdf_cubic_to( const FT_26D6_Vec*  control_1,
                const FT_26D6_Vec*  control_2,
                const FT_26D6_Vec*  to,
                void*               user )
  {
    SDF_Shape*    shape   = ( SDF_Shape* )user;
    SDF_Edge*     edge    = NULL;
    SDF_Contour*  contour = NULL;

    FT_Error   error  = FT_Err_Ok;
    FT_Memory  memory = shape->memory;


    if ( !control_2 || !control_1 || !to || !user )
    {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    contour = shape->contours;

    FT_CALL( sdf_edge_new( memory, &edge ) );

    edge->edge_type = SDF_EDGE_CUBIC;
    edge->start_pos = contour->last_pos;
    edge->control_a = *control_1;
    edge->control_b = *control_2;
    edge->end_pos   = *to;

    edge->next        = contour->edges;
    contour->edges    = edge;
    contour->last_pos = *to;

  Exit:
    return error;
  }


  /* Construct the structure to hold all four outline */
  /* decomposition functions.                         */
  FT_DEFINE_OUTLINE_FUNCS(
    sdf_decompose_funcs,

    (FT_Outline_MoveTo_Func) sdf_move_to,   /* move_to  */
    (FT_Outline_LineTo_Func) sdf_line_to,   /* line_to  */
    (FT_Outline_ConicTo_Func)sdf_conic_to,  /* conic_to */
    (FT_Outline_CubicTo_Func)sdf_cubic_to,  /* cubic_to */

    0,                                      /* shift    */
    0                                       /* delta    */
  )


  /* Decompose `outline` and put it into the `shape` structure.  */
  static FT_Error
  sdf_outline_decompose( FT_Outline*  outline,
                         SDF_Shape*   shape )
  {
    FT_Error  error = FT_Err_Ok;


    if ( !outline || !shape )
    {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    error = FT_Outline_Decompose( outline,
                                  &sdf_decompose_funcs,
                                  (void*)shape );

  Exit:
    return error;
  }


  /**************************************************************************
   *
   * utility functions
   *
   */

  /* Return the control box of an edge.  The control box is a rectangle */
  /* in which all the control points can fit tightly.                   */
  static FT_CBox
  get_control_box( SDF_Edge  edge )
  {
    FT_CBox  cbox   = { 0, 0, 0, 0 };
    FT_Bool  is_set = 0;


    switch ( edge.edge_type )
    {
    case SDF_EDGE_CUBIC:
      cbox.xMin = edge.control_b.x;
      cbox.xMax = edge.control_b.x;
      cbox.yMin = edge.control_b.y;
      cbox.yMax = edge.control_b.y;

      is_set = 1;
      FALL_THROUGH;

    case SDF_EDGE_CONIC:
      if ( is_set )
      {
        cbox.xMin = edge.control_a.x < cbox.xMin
                    ? edge.control_a.x
                    : cbox.xMin;
        cbox.xMax = edge.control_a.x > cbox.xMax
                    ? edge.control_a.x
                    : cbox.xMax;

        cbox.yMin = edge.control_a.y < cbox.yMin
                    ? edge.control_a.y
                    : cbox.yMin;
        cbox.yMax = edge.control_a.y > cbox.yMax
                    ? edge.control_a.y
                    : cbox.yMax;
      }
      else
      {
        cbox.xMin = edge.control_a.x;
        cbox.xMax = edge.control_a.x;
        cbox.yMin = edge.control_a.y;
        cbox.yMax = edge.control_a.y;

        is_set = 1;
      }
      FALL_THROUGH;

    case SDF_EDGE_LINE:
      if ( is_set )
      {
        cbox.xMin = edge.start_pos.x < cbox.xMin
                    ? edge.start_pos.x
                    : cbox.xMin;
        cbox.xMax = edge.start_pos.x > cbox.xMax
                    ? edge.start_pos.x
                    : cbox.xMax;

        cbox.yMin = edge.start_pos.y < cbox.yMin
                    ? edge.start_pos.y
                    : cbox.yMin;
        cbox.yMax = edge.start_pos.y > cbox.yMax
                    ? edge.start_pos.y
                    : cbox.yMax;
      }
      else
      {
        cbox.xMin = edge.start_pos.x;
        cbox.xMax = edge.start_pos.x;
        cbox.yMin = edge.start_pos.y;
        cbox.yMax = edge.start_pos.y;
      }

      cbox.xMin = edge.end_pos.x < cbox.xMin
                  ? edge.end_pos.x
                  : cbox.xMin;
      cbox.xMax = edge.end_pos.x > cbox.xMax
                  ? edge.end_pos.x
                  : cbox.xMax;

      cbox.yMin = edge.end_pos.y < cbox.yMin
                  ? edge.end_pos.y
                  : cbox.yMin;
      cbox.yMax = edge.end_pos.y > cbox.yMax
                  ? edge.end_pos.y
                  : cbox.yMax;

      break;

    default:
      break;
    }

    return cbox;
  }


  /* Return orientation of a single contour.                    */
  /* Note that the orientation is independent of the fill rule! */
  /* So, for TTF a clockwise-oriented contour has to be filled  */
  /* and the opposite for OTF fonts.                            */
  static SDF_Contour_Orientation
  get_contour_orientation ( SDF_Contour*  contour )
  {
    SDF_Edge*  head = NULL;
    FT_26D6    area = 0;


    /* return none if invalid parameters */
    if ( !contour || !contour->edges )
      return SDF_ORIENTATION_NONE;

    head = contour->edges;

    /* Calculate the area of the control box for all edges. */
    while ( head )
    {
      switch ( head->edge_type )
      {
      case SDF_EDGE_LINE:
        area += MUL_26D6( ( head->end_pos.x - head->start_pos.x ),
                          ( head->end_pos.y + head->start_pos.y ) );
        break;

      case SDF_EDGE_CONIC:
        area += MUL_26D6( head->control_a.x - head->start_pos.x,
                          head->control_a.y + head->start_pos.y );
        area += MUL_26D6( head->end_pos.x - head->control_a.x,
                          head->end_pos.y + head->control_a.y );
        break;

      case SDF_EDGE_CUBIC:
        area += MUL_26D6( head->control_a.x - head->start_pos.x,
                          head->control_a.y + head->start_pos.y );
        area += MUL_26D6( head->control_b.x - head->control_a.x,
                          head->control_b.y + head->control_a.y );
        area += MUL_26D6( head->end_pos.x - head->control_b.x,
                          head->end_pos.y + head->control_b.y );
        break;

      default:
        return SDF_ORIENTATION_NONE;
      }

      head = head->next;
    }

    /* Clockwise contours cover a positive area, and counter-clockwise */
    /* contours cover a negative area.                                 */
    if ( area > 0 )
      return SDF_ORIENTATION_CW;
    else
      return SDF_ORIENTATION_CCW;
  }


  /* This function is exactly the same as the one */
  /* in the smooth renderer.  It splits a conic   */
  /* into two conics exactly half way at t = 0.5. */
  static void
  split_conic( FT_26D6_Vec*  base )
  {
    FT_26D6  a, b;


    base[4].x = base[2].x;
    a         = base[0].x + base[1].x;
    b         = base[1].x + base[2].x;
    base[3].x = b / 2;
    base[2].x = ( a + b ) / 4;
    base[1].x = a / 2;

    base[4].y = base[2].y;
    a         = base[0].y + base[1].y;
    b         = base[1].y + base[2].y;
    base[3].y = b / 2;
    base[2].y = ( a + b ) / 4;
    base[1].y = a / 2;
  }


  /* This function is exactly the same as the one */
  /* in the smooth renderer.  It splits a cubic   */
  /* into two cubics exactly half way at t = 0.5. */
  static void
  split_cubic( FT_26D6_Vec*  base )
  {
    FT_26D6  a, b, c;


    base[6].x = base[3].x;
    a         = base[0].x + base[1].x;
    b         = base[1].x + base[2].x;
    c         = base[2].x + base[3].x;
    base[5].x = c / 2;
    c        += b;
    base[4].x = c / 4;
    base[1].x = a / 2;
    a        += b;
    base[2].x = a / 4;
    base[3].x = ( a + c ) / 8;

    base[6].y = base[3].y;
    a         = base[0].y + base[1].y;
    b         = base[1].y + base[2].y;
    c         = base[2].y + base[3].y;
    base[5].y = c / 2;
    c        += b;
    base[4].y = c / 4;
    base[1].y = a / 2;
    a        += b;
    base[2].y = a / 4;
    base[3].y = ( a + c ) / 8;
  }


  /* Split a conic Bezier curve into a number of lines */
  /* and add them to `out'.                            */
  /*                                                   */
  /* This function uses recursion; we thus need        */
  /* parameter `max_splits' for stopping.              */
  static FT_Error
  split_sdf_conic( FT_Memory     memory,
                   FT_26D6_Vec*  control_points,
                   FT_UInt       max_splits,
                   SDF_Edge**    out )
  {
    FT_Error     error = FT_Err_Ok;
    FT_26D6_Vec  cpos[5];
    SDF_Edge*    left,*  right;


    if ( !memory || !out  )
    {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    /* split conic outline */
    cpos[0] = control_points[0];
    cpos[1] = control_points[1];
    cpos[2] = control_points[2];

    split_conic( cpos );

    /* If max number of splits is done */
    /* then stop and add the lines to  */
    /* the list.                       */
    if ( max_splits <= 2 )
      goto Append;

    /* Otherwise keep splitting. */
    FT_CALL( split_sdf_conic( memory, &cpos[0], max_splits / 2, out ) );
    FT_CALL( split_sdf_conic( memory, &cpos[2], max_splits / 2, out ) );

    /* [NOTE]: This is not an efficient way of   */
    /* splitting the curve.  Check the deviation */
    /* instead and stop if the deviation is less */
    /* than a pixel.                             */

    goto Exit;

  Append:
    /* Do allocation and add the lines to the list. */

    FT_CALL( sdf_edge_new( memory, &left ) );
    FT_CALL( sdf_edge_new( memory, &right ) );

    left->start_pos  = cpos[0];
    left->end_pos    = cpos[2];
    left->edge_type  = SDF_EDGE_LINE;

    right->start_pos = cpos[2];
    right->end_pos   = cpos[4];
    right->edge_type = SDF_EDGE_LINE;

    left->next  = right;
    right->next = (*out);
    *out        = left;

  Exit:
    return error;
  }


  /* Split a cubic Bezier curve into a number of lines */
  /* and add them to `out`.                            */
  /*                                                   */
  /* This function uses recursion; we thus need        */
  /* parameter `max_splits' for stopping.              */
  static FT_Error
  split_sdf_cubic( FT_Memory     memory,
                   FT_26D6_Vec*  control_points,
                   FT_UInt       max_splits,
                   SDF_Edge**    out )
  {
    FT_Error       error = FT_Err_Ok;
    FT_26D6_Vec    cpos[7];
    SDF_Edge*      left, *right;
    const FT_26D6  threshold = ONE_PIXEL / 4;


    if ( !memory || !out )
    {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    /* split the cubic */
    cpos[0] = control_points[0];
    cpos[1] = control_points[1];
    cpos[2] = control_points[2];
    cpos[3] = control_points[3];

    /* If the segment is flat enough we won't get any benefit by */
    /* splitting it further, so we can just stop splitting.      */
    /*                                                           */
    /* Check the deviation of the Bezier curve and stop if it is */
    /* smaller than the pre-defined `threshold` value.           */
    if ( FT_ABS( 2 * cpos[0].x - 3 * cpos[1].x + cpos[3].x ) < threshold &&
         FT_ABS( 2 * cpos[0].y - 3 * cpos[1].y + cpos[3].y ) < threshold &&
         FT_ABS( cpos[0].x - 3 * cpos[2].x + 2 * cpos[3].x ) < threshold &&
         FT_ABS( cpos[0].y - 3 * cpos[2].y + 2 * cpos[3].y ) < threshold )
    {
      split_cubic( cpos );
      goto Append;
    }

    split_cubic( cpos );

    /* If max number of splits is done */
    /* then stop and add the lines to  */
    /* the list.                       */
    if ( max_splits <= 2 )
      goto Append;

    /* Otherwise keep splitting. */
    FT_CALL( split_sdf_cubic( memory, &cpos[0], max_splits / 2, out ) );
    FT_CALL( split_sdf_cubic( memory, &cpos[3], max_splits / 2, out ) );

    /* [NOTE]: This is not an efficient way of   */
    /* splitting the curve.  Check the deviation */
    /* instead and stop if the deviation is less */
    /* than a pixel.                             */

    goto Exit;

  Append:
    /* Do allocation and add the lines to the list. */

    FT_CALL( sdf_edge_new( memory, &left) );
    FT_CALL( sdf_edge_new( memory, &right) );

    left->start_pos  = cpos[0];
    left->end_pos    = cpos[3];
    left->edge_type  = SDF_EDGE_LINE;

    right->start_pos = cpos[3];
    right->end_pos   = cpos[6];
    right->edge_type = SDF_EDGE_LINE;

    left->next  = right;
    right->next = (*out);
    *out        = left;

  Exit:
    return error;
  }


  /* Subdivide an entire shape into line segments */
  /* such that it doesn't look visually different */
  /* from the original curve.                     */
  static FT_Error
  split_sdf_shape( SDF_Shape*  shape )
  {
    FT_Error   error = FT_Err_Ok;
    FT_Memory  memory;

    SDF_Contour*  contours;
    SDF_Contour*  new_contours = NULL;


    if ( !shape || !shape->memory )
    {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    contours = shape->contours;
    memory   = shape->memory;

    /* for each contour */
    while ( contours )
    {
      SDF_Edge*  edges     = contours->edges;
      SDF_Edge*  new_edges = NULL;

      SDF_Contour*  tempc;


      /* for each edge */
      while ( edges )
      {
        SDF_Edge*  edge = edges;
        SDF_Edge*  temp;

        switch ( edge->edge_type )
        {
        case SDF_EDGE_LINE:
          /* Just create a duplicate edge in case     */
          /* it is a line.  We can use the same edge. */
          FT_CALL( sdf_edge_new( memory, &temp ) );

          ft_memcpy( temp, edge, sizeof ( *edge ) );

          temp->next = new_edges;
          new_edges  = temp;
          break;

        case SDF_EDGE_CONIC:
          /* Subdivide the curve and add it to the list. */
          {
            FT_26D6_Vec  ctrls[3];
            FT_26D6      dx, dy;
            FT_UInt      num_splits;


            ctrls[0] = edge->start_pos;
            ctrls[1] = edge->control_a;
            ctrls[2] = edge->end_pos;

            dx = FT_ABS( ctrls[2].x + ctrls[0].x - 2 * ctrls[1].x );
            dy = FT_ABS( ctrls[2].y + ctrls[0].y - 2 * ctrls[1].y );
            if ( dx < dy )
              dx = dy;

            /* Calculate the number of necessary bisections.  Each      */
            /* bisection causes a four-fold reduction of the deviation, */
            /* hence we bisect the Bezier curve until the deviation     */
            /* becomes less than 1/8 of a pixel.  For more details      */
            /* check file `ftgrays.c`.                                  */
            num_splits = 1;
            while ( dx > ONE_PIXEL / 8 )
            {
              dx         >>= 2;
              num_splits <<= 1;
            }

            error = split_sdf_conic( memory, ctrls, num_splits, &new_edges );
          }
          break;

        case SDF_EDGE_CUBIC:
          /* Subdivide the curve and add it to the list. */
          {
            FT_26D6_Vec  ctrls[4];


            ctrls[0] = edge->start_pos;
            ctrls[1] = edge->control_a;
            ctrls[2] = edge->control_b;
            ctrls[3] = edge->end_pos;

            error = split_sdf_cubic( memory, ctrls, 32, &new_edges );
          }
          break;

        default:
          error = FT_THROW( Invalid_Argument );
        }

        if ( error != FT_Err_Ok )
          goto Exit;

        edges = edges->next;
      }

      /* add to the contours list */
      FT_CALL( sdf_contour_new( memory, &tempc ) );

      tempc->next  = new_contours;
      tempc->edges = new_edges;
      new_contours = tempc;
      new_edges    = NULL;

      /* deallocate the contour */
      tempc    = contours;
      contours = contours->next;

      sdf_contour_done( memory, &tempc );
    }

    shape->contours = new_contours;

  Exit:
    return error;
  }


  /**************************************************************************
   *
   * for debugging
   *
   */

#ifdef FT_DEBUG_LEVEL_TRACE

  static void
  sdf_shape_dump( SDF_Shape*  shape )
  {
    FT_UInt  num_contours = 0;

    FT_UInt  total_edges = 0;
    FT_UInt  total_lines = 0;
    FT_UInt  total_conic = 0;
    FT_UInt  total_cubic = 0;

    SDF_Contour*  contour_list;


    if ( !shape )
    {
      FT_TRACE5(( "sdf_shape_dump: null shape\n" ));
      return;
    }

    contour_list = shape->contours;

    FT_TRACE5(( "sdf_shape_dump (values are in 26.6 format):\n" ));

    while ( contour_list )
    {
      FT_UInt       num_edges = 0;
      SDF_Edge*     edge_list;
      SDF_Contour*  contour = contour_list;


      FT_TRACE5(( "  Contour %d\n", num_contours ));

      edge_list = contour->edges;

      while ( edge_list )
      {
        SDF_Edge*  edge = edge_list;


        FT_TRACE5(( "  %3d: ", num_edges ));

        switch ( edge->edge_type )
        {
        case SDF_EDGE_LINE:
          FT_TRACE5(( "Line:  (%ld, %ld) -- (%ld, %ld)\n",
                      edge->start_pos.x, edge->start_pos.y,
                      edge->end_pos.x, edge->end_pos.y ));
          total_lines++;
          break;

        case SDF_EDGE_CONIC:
          FT_TRACE5(( "Conic: (%ld, %ld) .. (%ld, %ld) .. (%ld, %ld)\n",
                      edge->start_pos.x, edge->start_pos.y,
                      edge->control_a.x, edge->control_a.y,
                      edge->end_pos.x, edge->end_pos.y ));
          total_conic++;
          break;

        case SDF_EDGE_CUBIC:
          FT_TRACE5(( "Cubic: (%ld, %ld) .. (%ld, %ld)"
                      " .. (%ld, %ld) .. (%ld %ld)\n",
                      edge->start_pos.x, edge->start_pos.y,
                      edge->control_a.x, edge->control_a.y,
                      edge->control_b.x, edge->control_b.y,
                      edge->end_pos.x, edge->end_pos.y ));
          total_cubic++;
          break;

        default:
          break;
        }

        num_edges++;
        total_edges++;
        edge_list = edge_list->next;
      }

      num_contours++;
      contour_list = contour_list->next;
    }

    FT_TRACE5(( "\n" ));
    FT_TRACE5(( "  total number of contours = %d\n", num_contours ));
    FT_TRACE5(( "  total number of edges    = %d\n", total_edges ));
    FT_TRACE5(( "    |__lines = %d\n", total_lines ));
    FT_TRACE5(( "    |__conic = %d\n", total_conic ));
    FT_TRACE5(( "    |__cubic = %d\n", total_cubic ));
  }

#endif /* FT_DEBUG_LEVEL_TRACE */


  /**************************************************************************
   *
   * math functions
   *
   */

#if !USE_NEWTON_FOR_CONIC

  /* [NOTE]: All the functions below down until rasterizer */
  /*         can be avoided if we decide to subdivide the  */
  /*         curve into lines.                             */

  /* This function uses Newton's iteration to find */
  /* the cube root of a fixed-point integer.       */
  static FT_16D16
  cube_root( FT_16D16  val )
  {
    /* [IMPORTANT]: This function is not good as it may */
    /* not break, so use a lookup table instead.  Or we */
    /* can use an algorithm similar to `square_root`.   */

    FT_Int  v, g, c;


    if ( val == 0                  ||
         val == -FT_INT_16D16( 1 ) ||
         val ==  FT_INT_16D16( 1 ) )
      return val;

    v = val < 0 ? -val : val;
    g = square_root( v );
    c = 0;

    while ( 1 )
    {
      c = FT_MulFix( FT_MulFix( g, g ), g ) - v;
      c = FT_DivFix( c, 3 * FT_MulFix( g, g ) );

      g -= c;

      if ( ( c < 0 ? -c : c ) < 30 )
        break;
    }

    return val < 0 ? -g : g;
  }


  /* Calculate the perpendicular by using '1 - base^2'. */
  /* Then use arctan to compute the angle.              */
  static FT_16D16
  arc_cos( FT_16D16  val )
  {
    FT_16D16  p;
    FT_16D16  b   = val;
    FT_16D16  one = FT_INT_16D16( 1 );


    if ( b > one )
      b = one;
    if ( b < -one )
      b = -one;

    p = one - FT_MulFix( b, b );
    p = square_root( p );

    return FT_Atan2( b, p );
  }


  /* Compute roots of a quadratic polynomial, assign them to `out`, */
  /* and return number of real roots.                               */
  /*                                                                */
  /* The procedure can be found at                                  */
  /*                                                                */
  /*   https://mathworld.wolfram.com/QuadraticFormula.html          */
  static FT_UShort
  solve_quadratic_equation( FT_26D6   a,
                            FT_26D6   b,
                            FT_26D6   c,
                            FT_16D16  out[2] )
  {
    FT_16D16  discriminant = 0;


    a = FT_26D6_16D16( a );
    b = FT_26D6_16D16( b );
    c = FT_26D6_16D16( c );

    if ( a == 0 )
    {
      if ( b == 0 )
        return 0;
      else
      {
        out[0] = FT_DivFix( -c, b );

        return 1;
      }
    }

    discriminant = FT_MulFix( b, b ) - 4 * FT_MulFix( a, c );

    if ( discriminant < 0 )
      return 0;
    else if ( discriminant == 0 )
    {
      out[0] = FT_DivFix( -b, 2 * a );

      return 1;
    }
    else
    {
      discriminant = square_root( discriminant );

      out[0] = FT_DivFix( -b + discriminant, 2 * a );
      out[1] = FT_DivFix( -b - discriminant, 2 * a );

      return 2;
    }
  }


  /* Compute roots of a cubic polynomial, assign them to `out`, */
  /* and return number of real roots.                           */
  /*                                                            */
  /* The procedure can be found at                              */
  /*                                                            */
  /*   https://mathworld.wolfram.com/CubicFormula.html          */
  static FT_UShort
  solve_cubic_equation( FT_26D6   a,
                        FT_26D6   b,
                        FT_26D6   c,
                        FT_26D6   d,
                        FT_16D16  out[3] )
  {
    FT_16D16  q = 0;      /* intermediate */
    FT_16D16  r = 0;      /* intermediate */

    FT_16D16  a2 = b;     /* x^2 coefficients */
    FT_16D16  a1 = c;     /* x coefficients   */
    FT_16D16  a0 = d;     /* constant         */

    FT_16D16  q3   = 0;
    FT_16D16  r2   = 0;
    FT_16D16  a23  = 0;
    FT_16D16  a22  = 0;
    FT_16D16  a1x2 = 0;


    /* cutoff value for `a` to be a cubic, otherwise solve quadratic */
    if ( a == 0 || FT_ABS( a ) < 16 )
      return solve_quadratic_equation( b, c, d, out );

    if ( d == 0 )
    {
      out[0] = 0;

      return solve_quadratic_equation( a, b, c, out + 1 ) + 1;
    }

    /* normalize the coefficients; this also makes them 16.16 */
    a2 = FT_DivFix( a2, a );
    a1 = FT_DivFix( a1, a );
    a0 = FT_DivFix( a0, a );

    /* compute intermediates */
    a1x2 = FT_MulFix( a1, a2 );
    a22  = FT_MulFix( a2, a2 );
    a23  = FT_MulFix( a22, a2 );

    q = ( 3 * a1 - a22 ) / 9;
    r = ( 9 * a1x2 - 27 * a0 - 2 * a23 ) / 54;

    /* [BUG]: `q3` and `r2` still cause underflow. */

    q3 = FT_MulFix( q, q );
    q3 = FT_MulFix( q3, q );

    r2 = FT_MulFix( r, r );

    if ( q3 < 0 && r2 < -q3 )
    {
      FT_16D16  t = 0;


      q3 = square_root( -q3 );
      t  = FT_DivFix( r, q3 );

      if ( t > ( 1 << 16 ) )
        t =  ( 1 << 16 );
      if ( t < -( 1 << 16 ) )
        t = -( 1 << 16 );

      t   = arc_cos( t );
      a2 /= 3;
      q   = 2 * square_root( -q );

      out[0] = FT_MulFix( q, FT_Cos( t / 3 ) ) - a2;
      out[1] = FT_MulFix( q, FT_Cos( ( t + FT_ANGLE_PI * 2 ) / 3 ) ) - a2;
      out[2] = FT_MulFix( q, FT_Cos( ( t + FT_ANGLE_PI * 4 ) / 3 ) ) - a2;

      return 3;
    }

    else if ( r2 == -q3 )
    {
      FT_16D16  s = 0;


      s   = cube_root( r );
      a2 /= -3;

      out[0] = a2 + ( 2 * s );
      out[1] = a2 - s;

      return 2;
    }

    else
    {
      FT_16D16  s    = 0;
      FT_16D16  t    = 0;
      FT_16D16  dis  = 0;


      if ( q3 == 0 )
        dis = FT_ABS( r );
      else
        dis = square_root( q3 + r2 );

      s = cube_root( r + dis );
      t = cube_root( r - dis );
      a2 /= -3;
      out[0] = ( a2 + ( s + t ) );

      return 1;
    }
  }

#endif /* !USE_NEWTON_FOR_CONIC */


  /*************************************************************************/
  /*************************************************************************/
  /**                                                                     **/
  /**  RASTERIZER                                                         **/
  /**                                                                     **/
  /*************************************************************************/
  /*************************************************************************/

  /**************************************************************************
   *
   * @Function:
   *   resolve_corner
   *
   * @Description:
   *   At some places on the grid two edges can give opposite directions;
   *   this happens when the closest point is on one of the endpoint.  In
   *   that case we need to check the proper sign.
   *
   *   This can be visualized by an example:
   *
   *   ```
   *                x
   *
   *                   o
   *                  ^ \
   *                 /   \
   *                /     \
   *           (a) /       \  (b)
   *              /         \
   *             /           \
   *            /             v
   *   ```
   *
   *   Suppose `x` is the point whose shortest distance from an arbitrary
   *   contour we want to find out.  It is clear that `o` is the nearest
   *   point on the contour.  Now to determine the sign we do a cross
   *   product of the shortest distance vector and the edge direction, i.e.,
   *
   *   ```
   *   => sign = cross(x - o, direction(a))
   *   ```
   *
   *   Using the right hand thumb rule we can see that the sign will be
   *   positive.
   *
   *   If we use `b', however, we have
   *
   *   ```
   *   => sign = cross(x - o, direction(b))
   *   ```
   *
   *   In this case the sign will be negative.  To determine the correct
   *   sign we thus divide the plane in two halves and check which plane the
   *   point lies in.
   *
   *   ```
   *                   |
   *                x  |
   *                   |
   *                   o
   *                  ^|\
   *                 / | \
   *                /  |  \
   *           (a) /   |   \  (b)
   *              /    |    \
   *             /           \
   *            /             v
   *   ```
   *
   *   We can see that `x` lies in the plane of `a`, so we take the sign
   *   determined by `a`.  This test can be easily done by calculating the
   *   orthogonality and taking the greater one.
   *
   *   The orthogonality is simply the sinus of the two vectors (i.e.,
   *   x - o) and the corresponding direction.  We efficiently pre-compute
   *   the orthogonality with the corresponding `get_min_distance_*`
   *   functions.
   *
   * @Input:
   *   sdf1 ::
   *     First signed distance (can be any of `a` or `b`).
   *
   *   sdf1 ::
   *     Second signed distance (can be any of `a` or `b`).
   *
   * @Return:
   *   The correct signed distance, which is computed by using the above
   *   algorithm.
   *
   * @Note:
   *   The function does not care about the actual distance, it simply
   *   returns the signed distance which has a larger cross product.  As a
   *   consequence, this function should not be used if the two distances
   *   are fairly apart.  In that case simply use the signed distance with
   *   a shorter absolute distance.
   *
   */
  static SDF_Signed_Distance
  resolve_corner( SDF_Signed_Distance  sdf1,
                  SDF_Signed_Distance  sdf2 )
  {
    return FT_ABS( sdf1.cross ) > FT_ABS( sdf2.cross ) ? sdf1 : sdf2;
  }


  /**************************************************************************
   *
   * @Function:
   *   get_min_distance_line
   *
   * @Description:
   *   Find the shortest distance from the `line` segment to a given `point`
   *   and assign it to `out`.  Use it for line segments only.
   *
   * @Input:
   *   line ::
   *     The line segment to which the shortest distance is to be computed.
   *
   *   point ::
   *     Point from which the shortest distance is to be computed.
   *
   * @Output:
   *   out ::
   *     Signed distance from `point` to `line`.
   *
   * @Return:
   *   FreeType error, 0 means success.
   *
   * @Note:
   *   The `line' parameter must have an edge type of `SDF_EDGE_LINE`.
   *
   */
  static FT_Error
  get_min_distance_line( SDF_Edge*             line,
                         FT_26D6_Vec           point,
                         SDF_Signed_Distance*  out )
  {
    /*
     * In order to calculate the shortest distance from a point to
     * a line segment, we do the following.  Let's assume that
     *
     * ```
     * a = start point of the line segment
     * b = end point of the line segment
     * p = point from which shortest distance is to be calculated
     * ```
     *
     * (1) Write the parametric equation of the line.
     *
     *     ```
     *     point_on_line = a + (b - a) * t   (t is the factor)
     *     ```
     *
     * (2) Find the projection of point `p` on the line.  The projection
     *     will be perpendicular to the line, which allows us to get the
     *     solution by making the dot product zero.
     *
     *     ```
     *     (point_on_line - a) . (p - point_on_line) = 0
     *
     *                (point_on_line)
     *      (a) x-------o----------------x (b)
     *                |_|
     *                  |
     *                  |
     *                 (p)
     *     ```
     *
     * (3) Simplification of the above equation yields the factor of
     *     `point_on_line`:
     *
     *     ```
     *     t = ((p - a) . (b - a)) / |b - a|^2
     *     ```
     *
     * (4) We clamp factor `t` between [0.0f, 1.0f] because `point_on_line`
     *     can be outside of the line segment:
     *
     *     ```
     *                                          (point_on_line)
     *     (a) x------------------------x (b) -----o---
     *                                           |_|
     *                                             |
     *                                             |
     *                                            (p)
     *     ```
     *
     * (5) Finally, the distance we are interested in is
     *
     *     ```
     *     |point_on_line - p|
     *     ```
     */

    FT_Error  error = FT_Err_Ok;

    FT_Vector  a;                   /* start position */
    FT_Vector  b;                   /* end position   */
    FT_Vector  p;                   /* current point  */

    FT_26D6_Vec  line_segment;      /* `b` - `a` */
    FT_26D6_Vec  p_sub_a;           /* `p` - `a` */

    FT_26D6   sq_line_length;       /* squared length of `line_segment` */
    FT_16D16  factor;               /* factor of the nearest point      */
    FT_26D6   cross;                /* used to determine sign           */

    FT_16D16_Vec  nearest_point;    /* `point_on_line`       */
    FT_16D16_Vec  nearest_vector;   /* `p` - `nearest_point` */


    if ( !line || !out )
    {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    if ( line->edge_type != SDF_EDGE_LINE )
    {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    a = line->start_pos;
    b = line->end_pos;
    p = point;

    line_segment.x = b.x - a.x;
    line_segment.y = b.y - a.y;

    p_sub_a.x = p.x - a.x;
    p_sub_a.y = p.y - a.y;

    sq_line_length = ( line_segment.x * line_segment.x ) / 64 +
                     ( line_segment.y * line_segment.y ) / 64;

    /* currently factor is 26.6 */
    factor = ( p_sub_a.x * line_segment.x ) / 64 +
             ( p_sub_a.y * line_segment.y ) / 64;

    /* now factor is 16.16 */
    factor = FT_DivFix( factor, sq_line_length );

    /* clamp the factor between 0.0 and 1.0 in fixed-point */
    if ( factor > FT_INT_16D16( 1 ) )
      factor = FT_INT_16D16( 1 );
    if ( factor < 0 )
      factor = 0;

    nearest_point.x = FT_MulFix( FT_26D6_16D16( line_segment.x ),
                                 factor );
    nearest_point.y = FT_MulFix( FT_26D6_16D16( line_segment.y ),
                                 factor );

    nearest_point.x = FT_26D6_16D16( a.x ) + nearest_point.x;
    nearest_point.y = FT_26D6_16D16( a.y ) + nearest_point.y;

    nearest_vector.x = nearest_point.x - FT_26D6_16D16( p.x );
    nearest_vector.y = nearest_point.y - FT_26D6_16D16( p.y );

    cross = FT_MulFix( nearest_vector.x, line_segment.y ) -
            FT_MulFix( nearest_vector.y, line_segment.x );

    /* assign the output */
    out->sign     = cross < 0 ? 1 : -1;
    out->distance = VECTOR_LENGTH_16D16( nearest_vector );

    /* Instead of finding `cross` for checking corner we */
    /* directly set it here.  This is more efficient     */
    /* because if the distance is perpendicular we can   */
    /* directly set it to 1.                             */
    if ( factor != 0 && factor != FT_INT_16D16( 1 ) )
      out->cross = FT_INT_16D16( 1 );
    else
    {
      /* [OPTIMIZATION]: Pre-compute this direction. */
      /* If not perpendicular then compute `cross`.  */
      FT_Vector_NormLen( &line_segment );
      FT_Vector_NormLen( &nearest_vector );

      out->cross = FT_MulFix( line_segment.x, nearest_vector.y ) -
                   FT_MulFix( line_segment.y, nearest_vector.x );
    }

  Exit:
    return error;
  }


  /**************************************************************************
   *
   * @Function:
   *   get_min_distance_conic
   *
   * @Description:
   *   Find the shortest distance from the `conic` Bezier curve to a given
   *   `point` and assign it to `out`.  Use it for conic/quadratic curves
   *   only.
   *
   * @Input:
   *   conic ::
   *     The conic Bezier curve to which the shortest distance is to be
   *     computed.
   *
   *   point ::
   *     Point from which the shortest distance is to be computed.
   *
   * @Output:
   *   out ::
   *     Signed distance from `point` to `conic`.
   *
   * @Return:
   *     FreeType error, 0 means success.
   *
   * @Note:
   *   The `conic` parameter must have an edge type of `SDF_EDGE_CONIC`.
   *
   */

#if !USE_NEWTON_FOR_CONIC

  /*
   * The function uses an analytical method to find the shortest distance
   * which is faster than the Newton-Raphson method, but has underflows at
   * the moment.  Use Newton's method if you can see artifacts in the SDF.
   */
  static FT_Error
  get_min_distance_conic( SDF_Edge*             conic,
                          FT_26D6_Vec           point,
                          SDF_Signed_Distance*  out )
  {
    /*
     * The procedure to find the shortest distance from a point to a
     * quadratic Bezier curve is similar to the line segment algorithm.  The
     * shortest distance is perpendicular to the Bezier curve; the only
     * difference from line is that there can be more than one
     * perpendicular, and we also have to check the endpoints, because the
     * perpendicular may not be the shortest.
     *
     * Let's assume that
     * ```
     * p0 = first endpoint
     * p1 = control point
     * p2 = second endpoint
     * p  = point from which shortest distance is to be calculated
     * ```
     *
     * (1) The equation of a quadratic Bezier curve can be written as
     *
     *     ```
     *     B(t) = (1 - t)^2 * p0 + 2(1 - t)t * p1 + t^2 * p2
     *     ```
     *
     *     with `t` a factor in the range [0.0f, 1.0f].  This equation can
     *     be rewritten as
     *
     *     ```
     *     B(t) = t^2 * (p0 - 2p1 + p2) + 2t * (p1 - p0) + p0
     *     ```
     *
     *     With
     *
     *     ```
     *     A = p0 - 2p1 + p2
     *     B = p1 - p0
     *     ```
     *
     *     we have
     *
     *     ```
     *     B(t) = t^2 * A + 2t * B + p0
     *     ```
     *
     * (2) The derivative of the last equation above is
     *
     *     ```
     *     B'(t) = 2 *(tA + B)
     *     ```
     *
     * (3) To find the shortest distance from `p` to `B(t)` we find the
     *     point on the curve at which the shortest distance vector (i.e.,
     *     `B(t) - p`) and the direction (i.e., `B'(t)`) make 90 degrees.
     *     In other words, we make the dot product zero.
     *
     *     ```
     *     (B(t) - p) . (B'(t)) = 0
     *     (t^2 * A + 2t * B + p0 - p) . (2 * (tA + B)) = 0
     *     ```
     *
     *     After simplifying we get a cubic equation
     *
     *     ```
     *     at^3 + bt^2 + ct + d = 0
     *     ```
     *
     *     with
     *
     *     ```
     *     a = A.A
     *     b = 3A.B
     *     c = 2B.B + A.p0 - A.p
     *     d = p0.B - p.B
     *     ```
     *
     * (4) Now the roots of the equation can be computed using 'Cardano's
     *     Cubic formula'; we clamp the roots in the range [0.0f, 1.0f].
     *
     * [note]: `B` and `B(t)` are different in the above equations.
     */

    FT_Error  error = FT_Err_Ok;

    FT_26D6_Vec  aA, bB;         /* A, B in the above comment             */
    FT_26D6_Vec  nearest_point = { 0, 0 };
                                 /* point on curve nearest to `point`     */
    FT_26D6_Vec  direction;      /* direction of curve at `nearest_point` */

    FT_26D6_Vec  p0, p1, p2;     /* control points of a conic curve       */
    FT_26D6_Vec  p;              /* `point` to which shortest distance    */

    FT_26D6  a, b, c, d;         /* cubic coefficients                    */

    FT_16D16  roots[3] = { 0, 0, 0 };    /* real roots of the cubic eq.   */
    FT_16D16  min_factor;                /* factor at `nearest_point`     */
    FT_16D16  cross;                     /* to determine the sign         */
    FT_16D16  min      = FT_INT_MAX;     /* shortest squared distance     */

    FT_UShort  num_roots;                /* number of real roots of cubic */
    FT_UShort  i;


    if ( !conic || !out )
    {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    if ( conic->edge_type != SDF_EDGE_CONIC )
    {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    p0 = conic->start_pos;
    p1 = conic->control_a;
    p2 = conic->end_pos;
    p  = point;

    /* compute substitution coefficients */
    aA.x = p0.x - 2 * p1.x + p2.x;
    aA.y = p0.y - 2 * p1.y + p2.y;

    bB.x = p1.x - p0.x;
    bB.y = p1.y - p0.y;

    /* compute cubic coefficients */
    a = VEC_26D6_DOT( aA, aA );

    b = 3 * VEC_26D6_DOT( aA, bB );

    c = 2 * VEC_26D6_DOT( bB, bB ) +
            VEC_26D6_DOT( aA, p0 ) -
            VEC_26D6_DOT( aA, p );

    d = VEC_26D6_DOT( p0, bB ) -
        VEC_26D6_DOT( p, bB );

    /* find the roots */
    num_roots = solve_cubic_equation( a, b, c, d, roots );

    if ( num_roots == 0 )
    {
      roots[0]  = 0;
      roots[1]  = FT_INT_16D16( 1 );
      num_roots = 2;
    }

    /* [OPTIMIZATION]: Check the roots, clamp them and discard */
    /*                 duplicate roots.                        */

    /* convert these values to 16.16 for further computation */
    aA.x = FT_26D6_16D16( aA.x );
    aA.y = FT_26D6_16D16( aA.y );

    bB.x = FT_26D6_16D16( bB.x );
    bB.y = FT_26D6_16D16( bB.y );

    p0.x = FT_26D6_16D16( p0.x );
    p0.y = FT_26D6_16D16( p0.y );

    p.x = FT_26D6_16D16( p.x );
    p.y = FT_26D6_16D16( p.y );

    for ( i = 0; i < num_roots; i++ )
    {
      FT_16D16  t    = roots[i];
      FT_16D16  t2   = 0;
      FT_16D16  dist = 0;

      FT_16D16_Vec  curve_point;
      FT_16D16_Vec  dist_vector;

      /*
       * Ideally we should discard the roots which are outside the range
       * [0.0, 1.0] and check the endpoints of the Bezier curve, but Behdad
       * Esfahbod proved the following lemma.
       *
       * Lemma:
       *
       * (1) If the closest point on the curve [0, 1] is to the endpoint at
       *     `t` = 1 and the cubic has no real roots at `t` = 1 then the
       *     cubic must have a real root at some `t` > 1.
       *
       * (2) Similarly, if the closest point on the curve [0, 1] is to the
       *     endpoint at `t` = 0 and the cubic has no real roots at `t` = 0
       *     then the cubic must have a real root at some `t` < 0.
       *
       * Now because of this lemma we only need to clamp the roots and that
       * will take care of the endpoints.
       *
       * For more details see
       *
       *   https://lists.nongnu.org/archive/html/freetype-devel/2020-06/msg00147.html
       */

      if ( t < 0 )
        t = 0;
      if ( t > FT_INT_16D16( 1 ) )
        t = FT_INT_16D16( 1 );

      t2 = FT_MulFix( t, t );

      /* B(t) = t^2 * A + 2t * B + p0 - p */
      curve_point.x = FT_MulFix( aA.x, t2 ) +
                      2 * FT_MulFix( bB.x, t ) + p0.x;
      curve_point.y = FT_MulFix( aA.y, t2 ) +
                      2 * FT_MulFix( bB.y, t ) + p0.y;

      /* `curve_point` - `p` */
      dist_vector.x = curve_point.x - p.x;
      dist_vector.y = curve_point.y - p.y;

      dist = VECTOR_LENGTH_16D16( dist_vector );

      if ( dist < min )
      {
        min           = dist;
        nearest_point = curve_point;
        min_factor    = t;
      }
    }

    /* B'(t) = 2 * (tA + B) */
    direction.x = 2 * FT_MulFix( aA.x, min_factor ) + 2 * bB.x;
    direction.y = 2 * FT_MulFix( aA.y, min_factor ) + 2 * bB.y;

    /* determine the sign */
    cross = FT_MulFix( nearest_point.x - p.x, direction.y ) -
            FT_MulFix( nearest_point.y - p.y, direction.x );

    /* assign the values */
    out->distance = min;
    out->sign     = cross < 0 ? 1 : -1;

    if ( min_factor != 0 && min_factor != FT_INT_16D16( 1 ) )
      out->cross = FT_INT_16D16( 1 );   /* the two are perpendicular */
    else
    {
      /* convert to nearest vector */
      nearest_point.x -= FT_26D6_16D16( p.x );
      nearest_point.y -= FT_26D6_16D16( p.y );

      /* compute `cross` if not perpendicular */
      FT_Vector_NormLen( &direction );
      FT_Vector_NormLen( &nearest_point );

      out->cross = FT_MulFix( direction.x, nearest_point.y ) -
                   FT_MulFix( direction.y, nearest_point.x );
    }

  Exit:
    return error;
  }

#else /* USE_NEWTON_FOR_CONIC */

  /*
   * The function uses Newton's approximation to find the shortest distance,
   * which is a bit slower than the analytical method but doesn't cause
   * underflow.
   */
  static FT_Error
  get_min_distance_conic( SDF_Edge*             conic,
                          FT_26D6_Vec           point,
                          SDF_Signed_Distance*  out )
  {
    /*
     * This method uses Newton-Raphson's approximation to find the shortest
     * distance from a point to a conic curve.  It does not involve solving
     * any cubic equation, that is why there is no risk of underflow.
     *
     * Let's assume that
     *
     * ```
     * p0 = first endpoint
     * p1 = control point
     * p3 = second endpoint
     * p  = point from which shortest distance is to be calculated
     * ```
     *
     * (1) The equation of a quadratic Bezier curve can be written as
     *
     *     ```
     *     B(t) = (1 - t)^2 * p0 + 2(1 - t)t * p1 + t^2 * p2
     *     ```
     *
     *     with `t` the factor in the range [0.0f, 1.0f].  The above
     *     equation can be rewritten as
     *
     *     ```
     *     B(t) = t^2 * (p0 - 2p1 + p2) + 2t * (p1 - p0) + p0
     *     ```
     *
     *     With
     *
     *     ```
     *     A = p0 - 2p1 + p2
     *     B = 2 * (p1 - p0)
     *     ```
     *
     *     we have
     *
     *     ```
     *     B(t) = t^2 * A + t * B + p0
     *     ```
     *
     * (2) The derivative of the above equation is
     *
     *     ```
     *     B'(t) = 2t * A + B
     *     ```
     *
     * (3) The second derivative of the above equation is
     *
     *     ```
     *     B''(t) = 2A
     *     ```
     *
     * (4) The equation `P(t)` of the distance from point `p` to the curve
     *     can be written as
     *
     *     ```
     *     P(t) = t^2 * A + t^2 * B + p0 - p
     *     ```
     *
     *     With
     *
     *     ```
     *     C = p0 - p
     *     ```
     *
     *     we have
     *
     *     ```
     *     P(t) = t^2 * A + t * B + C
     *     ```
     *
     * (5) Finally, the equation of the angle between `B(t)` and `P(t)` can
     *     be written as
     *
     *     ```
     *     Q(t) = P(t) . B'(t)
     *     ```
     *
     * (6) Our task is to find a value of `t` such that the above equation
     *     `Q(t)` becomes zero, that is, the point-to-curve vector makes
     *     90~degrees with the curve.  We solve this with the Newton-Raphson
     *     method.
     *
     * (7) We first assume an arbitrary value of factor `t`, which we then
     *     improve.
     *
     *     ```
     *     t := Q(t) / Q'(t)
     *     ```
     *
     *     Putting the value of `Q(t)` from the above equation gives
     *
     *     ```
     *     t := P(t) . B'(t) / derivative(P(t) . B'(t))
     *     t := P(t) . B'(t) /
     *            (P'(t) . B'(t) + P(t) . B''(t))
     *     ```
     *
     *     Note that `P'(t)` is the same as `B'(t)` because the constant is
     *     gone due to the derivative.
     *
     * (8) Finally we get the equation to improve the factor as
     *
     *     ```
     *     t := P(t) . B'(t) /
     *            (B'(t) . B'(t) + P(t) . B''(t))
     *     ```
     *
     * [note]: `B` and `B(t)` are different in the above equations.
     */

    FT_Error  error = FT_Err_Ok;

    FT_26D6_Vec  aA, bB, cC;     /* A, B, C in the above comment          */
    FT_26D6_Vec  nearest_point = { 0, 0 };
                                 /* point on curve nearest to `point`     */
    FT_26D6_Vec  direction;      /* direction of curve at `nearest_point` */

    FT_26D6_Vec  p0, p1, p2;     /* control points of a conic curve       */
    FT_26D6_Vec  p;              /* `point` to which shortest distance    */

    FT_16D16  min_factor = 0;            /* factor at `nearest_point'     */
    FT_16D16  cross;                     /* to determine the sign         */
    FT_16D16  min        = FT_INT_MAX;   /* shortest squared distance     */

    FT_UShort  iterations;
    FT_UShort  steps;


    if ( !conic || !out )
    {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    if ( conic->edge_type != SDF_EDGE_CONIC )
    {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    p0 = conic->start_pos;
    p1 = conic->control_a;
    p2 = conic->end_pos;
    p  = point;

    /* compute substitution coefficients */
    aA.x = p0.x - 2 * p1.x + p2.x;
    aA.y = p0.y - 2 * p1.y + p2.y;

    bB.x = 2 * ( p1.x - p0.x );
    bB.y = 2 * ( p1.y - p0.y );

    cC.x = p0.x;
    cC.y = p0.y;

    /* do Newton's iterations */
    for ( iterations = 0; iterations <= MAX_NEWTON_DIVISIONS; iterations++ )
    {
      FT_16D16  factor = FT_INT_16D16( iterations ) / MAX_NEWTON_DIVISIONS;
      FT_16D16  factor2;
      FT_16D16  length;

      FT_16D16_Vec  curve_point; /* point on the curve  */
      FT_16D16_Vec  dist_vector; /* `curve_point` - `p` */

      FT_26D6_Vec  d1;           /* first  derivative   */
      FT_26D6_Vec  d2;           /* second derivative   */

      FT_16D16  temp1;
      FT_16D16  temp2;


      for ( steps = 0; steps < MAX_NEWTON_STEPS; steps++ )
      {
        factor2 = FT_MulFix( factor, factor );

        /* B(t) = t^2 * A + t * B + p0 */
        curve_point.x = FT_MulFix( aA.x, factor2 ) +
                        FT_MulFix( bB.x, factor ) + cC.x;
        curve_point.y = FT_MulFix( aA.y, factor2 ) +
                        FT_MulFix( bB.y, factor ) + cC.y;

        /* convert to 16.16 */
        curve_point.x = FT_26D6_16D16( curve_point.x );
        curve_point.y = FT_26D6_16D16( curve_point.y );

        /* P(t) in the comment */
        dist_vector.x = curve_point.x - FT_26D6_16D16( p.x );
        dist_vector.y = curve_point.y - FT_26D6_16D16( p.y );

        length = VECTOR_LENGTH_16D16( dist_vector );

        if ( length < min )
        {
          min           = length;
          min_factor    = factor;
          nearest_point = curve_point;
        }

        /* This is Newton's approximation.          */
        /*                                          */
        /*   t := P(t) . B'(t) /                    */
        /*          (B'(t) . B'(t) + P(t) . B''(t)) */

        /* B'(t) = 2tA + B */
        d1.x = FT_MulFix( aA.x, 2 * factor ) + bB.x;
        d1.y = FT_MulFix( aA.y, 2 * factor ) + bB.y;

        /* B''(t) = 2A */
        d2.x = 2 * aA.x;
        d2.y = 2 * aA.y;

        dist_vector.x /= 1024;
        dist_vector.y /= 1024;

        /* temp1 = P(t) . B'(t) */
        temp1 = VEC_26D6_DOT( dist_vector, d1 );

        /* temp2 = B'(t) . B'(t) + P(t) . B''(t) */
        temp2 = VEC_26D6_DOT( d1, d1 ) +
                VEC_26D6_DOT( dist_vector, d2 );

        factor -= FT_DivFix( temp1, temp2 );

        if ( factor < 0 || factor > FT_INT_16D16( 1 ) )
          break;
      }
    }

    /* B'(t) = 2t * A + B */
    direction.x = 2 * FT_MulFix( aA.x, min_factor ) + bB.x;
    direction.y = 2 * FT_MulFix( aA.y, min_factor ) + bB.y;

    /* determine the sign */
    cross = FT_MulFix( nearest_point.x - FT_26D6_16D16( p.x ),
                       direction.y )                           -
            FT_MulFix( nearest_point.y - FT_26D6_16D16( p.y ),
                       direction.x );

    /* assign the values */
    out->distance = min;
    out->sign     = cross < 0 ? 1 : -1;

    if ( min_factor != 0 && min_factor != FT_INT_16D16( 1 ) )
      out->cross = FT_INT_16D16( 1 );   /* the two are perpendicular */
    else
    {
      /* convert to nearest vector */
      nearest_point.x -= FT_26D6_16D16( p.x );
      nearest_point.y -= FT_26D6_16D16( p.y );

      /* compute `cross` if not perpendicular */
      FT_Vector_NormLen( &direction );
      FT_Vector_NormLen( &nearest_point );

      out->cross = FT_MulFix( direction.x, nearest_point.y ) -
                   FT_MulFix( direction.y, nearest_point.x );
    }

  Exit:
    return error;
  }


#endif /* USE_NEWTON_FOR_CONIC */


  /**************************************************************************
   *
   * @Function:
   *   get_min_distance_cubic
   *
   * @Description:
   *   Find the shortest distance from the `cubic` Bezier curve to a given
   *   `point` and assigns it to `out`.  Use it for cubic curves only.
   *
   * @Input:
   *   cubic ::
   *     The cubic Bezier curve to which the shortest distance is to be
   *     computed.
   *
   *   point ::
   *     Point from which the shortest distance is to be computed.
   *
   * @Output:
   *   out ::
   *     Signed distance from `point` to `cubic`.
   *
   * @Return:
   *   FreeType error, 0 means success.
   *
   * @Note:
   *   The function uses Newton's approximation to find the shortest
   *   distance.  Another way would be to divide the cubic into conic or
   *   subdivide the curve into lines, but that is not implemented.
   *
   *   The `cubic` parameter must have an edge type of `SDF_EDGE_CUBIC`.
   *
   */
  static FT_Error
  get_min_distance_cubic( SDF_Edge*             cubic,
                          FT_26D6_Vec           point,
                          SDF_Signed_Distance*  out )
  {
    /*
     * The procedure to find the shortest distance from a point to a cubic
     * Bezier curve is similar to quadratic curve algorithm.  The only
     * difference is that while calculating factor `t`, instead of a cubic
     * polynomial equation we have to find the roots of a 5th degree
     * polynomial equation.  Solving this would require a significant amount
     * of time, and still the results may not be accurate.  We are thus
     * going to directly approximate the value of `t` using the Newton-Raphson
     * method.
     *
     * Let's assume that
     *
     * ```
     * p0 = first endpoint
     * p1 = first control point
     * p2 = second control point
     * p3 = second endpoint
     * p  = point from which shortest distance is to be calculated
     * ```
     *
     * (1) The equation of a cubic Bezier curve can be written as
     *
     *     ```
     *     B(t) = (1 - t)^3 * p0 + 3(1 - t)^2 t * p1 +
     *              3(1 - t)t^2 * p2 + t^3 * p3
     *     ```
     *
     *     The equation can be expanded and written as
     *
     *     ```
     *     B(t) = t^3 * (-p0 + 3p1 - 3p2 + p3) +
     *              3t^2 * (p0 - 2p1 + p2) + 3t * (-p0 + p1) + p0
     *     ```
     *
     *     With
     *
     *     ```
     *     A = -p0 + 3p1 - 3p2 + p3
     *     B = 3(p0 - 2p1 + p2)
     *     C = 3(-p0 + p1)
     *     ```
     *
     *     we have
     *
     *     ```
     *     B(t) = t^3 * A + t^2 * B + t * C + p0
     *     ```
     *
     * (2) The derivative of the above equation is
     *
     *     ```
     *     B'(t) = 3t^2 * A + 2t * B + C
     *     ```
     *
     * (3) The second derivative of the above equation is
     *
     *     ```
     *     B''(t) = 6t * A + 2B
     *     ```
     *
     * (4) The equation `P(t)` of the distance from point `p` to the curve
     *     can be written as
     *
     *     ```
     *     P(t) = t^3 * A + t^2 * B + t * C + p0 - p
     *     ```
     *
     *     With
     *
     *     ```
     *     D = p0 - p
     *     ```
     *
     *     we have
     *
     *     ```
     *     P(t) = t^3 * A + t^2 * B + t * C + D
     *     ```
     *
     * (5) Finally the equation of the angle between `B(t)` and `P(t)` can
     *     be written as
     *
     *     ```
     *     Q(t) = P(t) . B'(t)
     *     ```
     *
     * (6) Our task is to find a value of `t` such that the above equation
     *     `Q(t)` becomes zero, that is, the point-to-curve vector makes
     *     90~degree with curve.  We solve this with the Newton-Raphson
     *     method.
     *
     * (7) We first assume an arbitrary value of factor `t`, which we then
     *     improve.
     *
     *     ```
     *     t := Q(t) / Q'(t)
     *     ```
     *
     *     Putting the value of `Q(t)` from the above equation gives
     *
     *     ```
     *     t := P(t) . B'(t) / derivative(P(t) . B'(t))
     *     t := P(t) . B'(t) /
     *            (P'(t) . B'(t) + P(t) . B''(t))
     *     ```
     *
     *     Note that `P'(t)` is the same as `B'(t)` because the constant is
     *     gone due to the derivative.
     *
     * (8) Finally we get the equation to improve the factor as
     *
     *     ```
     *     t := P(t) . B'(t) /
     *            (B'(t) . B'( t ) + P(t) . B''(t))
     *     ```
     *
     * [note]: `B` and `B(t)` are different in the above equations.
     */

    FT_Error  error = FT_Err_Ok;

    FT_26D6_Vec   aA, bB, cC, dD; /* A, B, C, D in the above comment       */
    FT_16D16_Vec  nearest_point = { 0, 0 };
                                  /* point on curve nearest to `point`     */
    FT_16D16_Vec  direction;      /* direction of curve at `nearest_point` */

    FT_26D6_Vec  p0, p1, p2, p3;  /* control points of a cubic curve       */
    FT_26D6_Vec  p;               /* `point` to which shortest distance    */

    FT_16D16  min_factor    = 0;            /* factor at shortest distance */
    FT_16D16  min_factor_sq = 0;            /* factor at shortest distance */
    FT_16D16  cross;                        /* to determine the sign       */
    FT_16D16  min           = FT_INT_MAX;   /* shortest distance           */

    FT_UShort  iterations;
    FT_UShort  steps;


    if ( !cubic || !out )
    {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    if ( cubic->edge_type != SDF_EDGE_CUBIC )
    {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    p0 = cubic->start_pos;
    p1 = cubic->control_a;
    p2 = cubic->control_b;
    p3 = cubic->end_pos;
    p  = point;

    /* compute substitution coefficients */
    aA.x = -p0.x + 3 * ( p1.x - p2.x ) + p3.x;
    aA.y = -p0.y + 3 * ( p1.y - p2.y ) + p3.y;

    bB.x = 3 * ( p0.x - 2 * p1.x + p2.x );
    bB.y = 3 * ( p0.y - 2 * p1.y + p2.y );

    cC.x = 3 * ( p1.x - p0.x );
    cC.y = 3 * ( p1.y - p0.y );

    dD.x = p0.x;
    dD.y = p0.y;

    for ( iterations = 0; iterations <= MAX_NEWTON_DIVISIONS; iterations++ )
    {
      FT_16D16  factor  = FT_INT_16D16( iterations ) / MAX_NEWTON_DIVISIONS;

      FT_16D16  factor2;         /* factor^2            */
      FT_16D16  factor3;         /* factor^3            */
      FT_16D16  length;

      FT_16D16_Vec  curve_point; /* point on the curve  */
      FT_16D16_Vec  dist_vector; /* `curve_point' - `p' */

      FT_26D6_Vec  d1;           /* first  derivative   */
      FT_26D6_Vec  d2;           /* second derivative   */

      FT_16D16  temp1;
      FT_16D16  temp2;


      for ( steps = 0; steps < MAX_NEWTON_STEPS; steps++ )
      {
        factor2 = FT_MulFix( factor, factor );
        factor3 = FT_MulFix( factor2, factor );

        /* B(t) = t^3 * A + t^2 * B + t * C + D */
        curve_point.x = FT_MulFix( aA.x, factor3 ) +
                        FT_MulFix( bB.x, factor2 ) +
                        FT_MulFix( cC.x, factor ) + dD.x;
        curve_point.y = FT_MulFix( aA.y, factor3 ) +
                        FT_MulFix( bB.y, factor2 ) +
                        FT_MulFix( cC.y, factor ) + dD.y;

        /* convert to 16.16 */
        curve_point.x = FT_26D6_16D16( curve_point.x );
        curve_point.y = FT_26D6_16D16( curve_point.y );

        /* P(t) in the comment */
        dist_vector.x = curve_point.x - FT_26D6_16D16( p.x );
        dist_vector.y = curve_point.y - FT_26D6_16D16( p.y );

        length = VECTOR_LENGTH_16D16( dist_vector );

        if ( length < min )
        {
          min           = length;
          min_factor    = factor;
          min_factor_sq = factor2;
          nearest_point = curve_point;
        }

        /* This the Newton's approximation.         */
        /*                                          */
        /*   t := P(t) . B'(t) /                    */
        /*          (B'(t) . B'(t) + P(t) . B''(t)) */

        /* B'(t) = 3t^2 * A + 2t * B + C */
        d1.x = FT_MulFix( aA.x, 3 * factor2 ) +
               FT_MulFix( bB.x, 2 * factor ) + cC.x;
        d1.y = FT_MulFix( aA.y, 3 * factor2 ) +
               FT_MulFix( bB.y, 2 * factor ) + cC.y;

        /* B''(t) = 6t * A + 2B */
        d2.x = FT_MulFix( aA.x, 6 * factor ) + 2 * bB.x;
        d2.y = FT_MulFix( aA.y, 6 * factor ) + 2 * bB.y;

        dist_vector.x /= 1024;
        dist_vector.y /= 1024;

        /* temp1 = P(t) . B'(t) */
        temp1 = VEC_26D6_DOT( dist_vector, d1 );

        /* temp2 = B'(t) . B'(t) + P(t) . B''(t) */
        temp2 = VEC_26D6_DOT( d1, d1 ) +
                VEC_26D6_DOT( dist_vector, d2 );

        factor -= FT_DivFix( temp1, temp2 );

        if ( factor < 0 || factor > FT_INT_16D16( 1 ) )
          break;
      }
    }

    /* B'(t) = 3t^2 * A + 2t * B + C */
    direction.x = FT_MulFix( aA.x, 3 * min_factor_sq ) +
                  FT_MulFix( bB.x, 2 * min_factor ) + cC.x;
    direction.y = FT_MulFix( aA.y, 3 * min_factor_sq ) +
                  FT_MulFix( bB.y, 2 * min_factor ) + cC.y;

    /* determine the sign */
    cross = FT_MulFix( nearest_point.x - FT_26D6_16D16( p.x ),
                       direction.y )                           -
            FT_MulFix( nearest_point.y - FT_26D6_16D16( p.y ),
                       direction.x );

    /* assign the values */
    out->distance = min;
    out->sign     = cross < 0 ? 1 : -1;

    if ( min_factor != 0 && min_factor != FT_INT_16D16( 1 ) )
      out->cross = FT_INT_16D16( 1 );   /* the two are perpendicular */
    else
    {
      /* convert to nearest vector */
      nearest_point.x -= FT_26D6_16D16( p.x );
      nearest_point.y -= FT_26D6_16D16( p.y );

      /* compute `cross` if not perpendicular */
      FT_Vector_NormLen( &direction );
      FT_Vector_NormLen( &nearest_point );

      out->cross = FT_MulFix( direction.x, nearest_point.y ) -
                   FT_MulFix( direction.y, nearest_point.x );
    }

  Exit:
    return error;
  }


  /**************************************************************************
   *
   * @Function:
   *   sdf_edge_get_min_distance
   *
   * @Description:
   *   Find shortest distance from `point` to any type of `edge`.  It checks
   *   the edge type and then calls the relevant `get_min_distance_*`
   *   function.
   *
   * @Input:
   *   edge ::
   *     An edge to which the shortest distance is to be computed.
   *
   *   point ::
   *     Point from which the shortest distance is to be computed.
   *
   * @Output:
   *   out ::
   *     Signed distance from `point` to `edge`.
   *
   * @Return:
   *   FreeType error, 0 means success.
   *
   */
  static FT_Error
  sdf_edge_get_min_distance( SDF_Edge*             edge,
                             FT_26D6_Vec           point,
                             SDF_Signed_Distance*  out )
  {
    FT_Error  error = FT_Err_Ok;


    if ( !edge || !out )
    {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    /* edge-specific distance calculation */
    switch ( edge->edge_type )
    {
    case SDF_EDGE_LINE:
      get_min_distance_line( edge, point, out );
      break;

    case SDF_EDGE_CONIC:
      get_min_distance_conic( edge, point, out );
      break;

    case SDF_EDGE_CUBIC:
      get_min_distance_cubic( edge, point, out );
      break;

    default:
      error = FT_THROW( Invalid_Argument );
    }

  Exit:
    return error;
  }


  /* `sdf_generate' is not used at the moment */
#if 0

  #error "DO NOT USE THIS!"
  #error "The function still outputs 16-bit data, which might cause memory"
  #error "corruption.  If required I will add this later."

  /**************************************************************************
   *
   * @Function:
   *   sdf_contour_get_min_distance
   *
   * @Description:
   *   Iterate over all edges that make up the contour, find the shortest
   *   distance from a point to this contour, and assigns result to `out`.
   *
   * @Input:
   *   contour ::
   *     A contour to which the shortest distance is to be computed.
   *
   *   point ::
   *     Point from which the shortest distance is to be computed.
   *
   * @Output:
   *   out ::
   *     Signed distance from the `point' to the `contour'.
   *
   * @Return:
   *   FreeType error, 0 means success.
   *
   * @Note:
   *   The function does not return a signed distance for each edge which
   *   makes up the contour, it simply returns the shortest of all the
   *   edges.
   *
   */
  static FT_Error
  sdf_contour_get_min_distance( SDF_Contour*          contour,
                                FT_26D6_Vec           point,
                                SDF_Signed_Distance*  out )
  {
    FT_Error             error    = FT_Err_Ok;
    SDF_Signed_Distance  min_dist = max_sdf;
    SDF_Edge*            edge_list;


    if ( !contour || !out )
    {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    edge_list = contour->edges;

    /* iterate over all the edges manually */
    while ( edge_list )
    {
      SDF_Signed_Distance  current_dist = max_sdf;
      FT_16D16             diff;


      FT_CALL( sdf_edge_get_min_distance( edge_list,
                                          point,
                                          &current_dist ) );

      if ( current_dist.distance >= 0 )
      {
        diff = current_dist.distance - min_dist.distance;


        if ( FT_ABS( diff ) < CORNER_CHECK_EPSILON )
          min_dist = resolve_corner( min_dist, current_dist );
        else if ( diff < 0 )
          min_dist = current_dist;
      }
      else
        FT_TRACE0(( "sdf_contour_get_min_distance: Overflow.\n" ));

      edge_list = edge_list->next;
    }

    *out = min_dist;

  Exit:
    return error;
  }


  /**************************************************************************
   *
   * @Function:
   *   sdf_generate
   *
   * @Description:
   *   This is the main function that is responsible for generating signed
   *   distance fields.  The function does not align or compute the size of
   *   `bitmap`; therefore the calling application must set up `bitmap`
   *   properly and transform the `shape' appropriately in advance.
   *
   *   Currently we check all pixels against all contours and all edges.
   *
   * @Input:
   *   internal_params ::
   *     Internal parameters and properties required by the rasterizer.  See
   *     @SDF_Params for more.
   *
   *   shape ::
   *     A complete shape which is used to generate SDF.
   *
   *   spread ::
   *     Maximum distances to be allowed in the output bitmap.
   *
   * @Output:
   *   bitmap ::
   *     The output bitmap which will contain the SDF information.
   *
   * @Return:
   *   FreeType error, 0 means success.
   *
   */
  static FT_Error
  sdf_generate( const SDF_Params  internal_params,
                const SDF_Shape*  shape,
                FT_UInt           spread,
                const FT_Bitmap*  bitmap )
  {
    FT_Error  error = FT_Err_Ok;

    FT_UInt  width = 0;
    FT_UInt  rows  = 0;
    FT_UInt  x     = 0;   /* used to loop in x direction, i.e., width     */
    FT_UInt  y     = 0;   /* used to loop in y direction, i.e., rows      */
    FT_UInt  sp_sq = 0;   /* `spread` [* `spread`] as a 16.16 fixed value */

    FT_Short*  buffer;


    if ( !shape || !bitmap )
    {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    if ( spread < MIN_SPREAD || spread > MAX_SPREAD )
    {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    width  = bitmap->width;
    rows   = bitmap->rows;
    buffer = (FT_Short*)bitmap->buffer;

    if ( USE_SQUARED_DISTANCES )
      sp_sq = FT_INT_16D16( spread * spread );
    else
      sp_sq = FT_INT_16D16( spread );

    if ( width == 0 || rows == 0 )
    {
      FT_TRACE0(( "sdf_generate:"
                  " Cannot render glyph with width/height == 0\n" ));
      FT_TRACE0(( "             "
                  " (width, height provided [%d, %d])\n",
                  width, rows ));

      error = FT_THROW( Cannot_Render_Glyph );
      goto Exit;
    }

    /* loop over all rows */
    for ( y = 0; y < rows; y++ )
    {
      /* loop over all pixels of a row */
      for ( x = 0; x < width; x++ )
      {
        /* `grid_point` is the current pixel position; */
        /* our task is to find the shortest distance   */
        /* from this point to the entire shape.        */
        FT_26D6_Vec          grid_point = zero_vector;
        SDF_Signed_Distance  min_dist   = max_sdf;
        SDF_Contour*         contour_list;

        FT_UInt   index;
        FT_Short  value;


        grid_point.x = FT_INT_26D6( x );
        grid_point.y = FT_INT_26D6( y );

        /* This `grid_point' is at the corner, but we */
        /* use the center of the pixel.               */
        grid_point.x += FT_INT_26D6( 1 ) / 2;
        grid_point.y += FT_INT_26D6( 1 ) / 2;

        contour_list = shape->contours;

        /* iterate over all contours manually */
        while ( contour_list )
        {
          SDF_Signed_Distance  current_dist = max_sdf;


          FT_CALL( sdf_contour_get_min_distance( contour_list,
                                                 grid_point,
                                                 &current_dist ) );

          if ( current_dist.distance < min_dist.distance )
            min_dist = current_dist;

          contour_list = contour_list->next;
        }

        /* [OPTIMIZATION]: if (min_dist > sp_sq) then simply clamp  */
        /*                 the value to spread to avoid square_root */

        /* clamp the values to spread */
        if ( min_dist.distance > sp_sq )
          min_dist.distance = sp_sq;

        /* square_root the values and fit in a 6.10 fixed-point */
        if ( USE_SQUARED_DISTANCES )
          min_dist.distance = square_root( min_dist.distance );

        if ( internal_params.orientation == FT_ORIENTATION_FILL_LEFT )
          min_dist.sign = -min_dist.sign;
        if ( internal_params.flip_sign )
          min_dist.sign = -min_dist.sign;

        min_dist.distance /= 64; /* convert from 16.16 to 22.10 */

        value  = min_dist.distance & 0x0000FFFF; /* truncate to 6.10 */
        value *= min_dist.sign;

        if ( internal_params.flip_y )
          index = y * width + x;
        else
          index = ( rows - y - 1 ) * width + x;

        buffer[index] = value;
      }
    }

  Exit:
    return error;
  }

#endif /* 0 */


  /**************************************************************************
   *
   * @Function:
   *   sdf_generate_bounding_box
   *
   * @Description:
   *   This function does basically the same thing as `sdf_generate` above
   *   but more efficiently.
   *
   *   Instead of checking all pixels against all edges, we loop over all
   *   edges and only check pixels around the control box of the edge; the
   *   control box is increased by the spread in all directions.  Anything
   *   outside of the control box that exceeds `spread` doesn't need to be
   *   computed.
   *
   *   Lastly, to determine the sign of unchecked pixels, we do a single
   *   pass of all rows starting with a '+' sign and flipping when we come
   *   across a '-' sign and continue.  This also eliminates the possibility
   *   of overflow because we only check the proximity of the curve.
   *   Therefore we can use squared distanced safely.
   *
   * @Input:
   *   internal_params ::
   *     Internal parameters and properties required by the rasterizer.
   *     See @SDF_Params for more.
   *
   *   shape ::
   *     A complete shape which is used to generate SDF.
   *
   *   spread ::
   *     Maximum distances to be allowed in the output bitmap.
   *
   * @Output:
   *   bitmap ::
   *     The output bitmap which will contain the SDF information.
   *
   * @Return:
   *   FreeType error, 0 means success.
   *
   */
  static FT_Error
  sdf_generate_bounding_box( const SDF_Params  internal_params,
                             const SDF_Shape*  shape,
                             FT_UInt           spread,
                             const FT_Bitmap*  bitmap )
  {
    FT_Error   error  = FT_Err_Ok;
    FT_Memory  memory = NULL;

    FT_Int  width, rows, i, j;
    FT_Int  sp_sq;            /* max value to check   */

    SDF_Contour*   contours;  /* list of all contours */
    FT_SDFFormat*  buffer;    /* the bitmap buffer    */

    /* This buffer has the same size in indices as the    */
    /* bitmap buffer.  When we check a pixel position for */
    /* a shortest distance we keep it in this buffer.     */
    /* This way we can find out which pixel is set,       */
    /* and also determine the signs properly.             */
    SDF_Signed_Distance*  dists = NULL;

    const FT_16D16  fixed_spread = (FT_16D16)FT_INT_16D16( spread );


    if ( !shape || !bitmap )
    {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    if ( spread < MIN_SPREAD || spread > MAX_SPREAD )
    {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    memory = shape->memory;
    if ( !memory )
    {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    if ( FT_ALLOC( dists,
                   bitmap->width * bitmap->rows * sizeof ( *dists ) ) )
      goto Exit;

    contours = shape->contours;
    width    = (FT_Int)bitmap->width;
    rows     = (FT_Int)bitmap->rows;
    buffer   = (FT_SDFFormat*)bitmap->buffer;

    if ( USE_SQUARED_DISTANCES )
      sp_sq = FT_INT_16D16( (FT_Int)( spread * spread ) );
    else
      sp_sq = fixed_spread;

    if ( width == 0 || rows == 0 )
    {
      FT_TRACE0(( "sdf_generate:"
                  " Cannot render glyph with width/height == 0\n" ));
      FT_TRACE0(( "             "
                  " (width, height provided [%d, %d])", width, rows ));

      error = FT_THROW( Cannot_Render_Glyph );
      goto Exit;
    }

    /* loop over all contours */
    while ( contours )
    {
      SDF_Edge*  edges = contours->edges;


      /* loop over all edges */
      while ( edges )
      {
        FT_CBox  cbox;
        FT_Int   x, y;


        /* get the control box and increase it by `spread' */
        cbox = get_control_box( *edges );

        cbox.xMin = ( cbox.xMin - 63 ) / 64 - ( FT_Pos )spread;
        cbox.xMax = ( cbox.xMax + 63 ) / 64 + ( FT_Pos )spread;
        cbox.yMin = ( cbox.yMin - 63 ) / 64 - ( FT_Pos )spread;
        cbox.yMax = ( cbox.yMax + 63 ) / 64 + ( FT_Pos )spread;

        /* now loop over the pixels in the control box. */
        for ( y = cbox.yMin; y < cbox.yMax; y++ )
        {
          for ( x = cbox.xMin; x < cbox.xMax; x++ )
          {
            FT_26D6_Vec          grid_point = zero_vector;
            SDF_Signed_Distance  dist       = max_sdf;
            FT_UInt              index      = 0;
            FT_16D16             diff       = 0;


            if ( x < 0 || x >= width )
              continue;
            if ( y < 0 || y >= rows )
              continue;

            grid_point.x = FT_INT_26D6( x );
            grid_point.y = FT_INT_26D6( y );

            /* This `grid_point` is at the corner, but we */
            /* use the center of the pixel.               */
            grid_point.x += FT_INT_26D6( 1 ) / 2;
            grid_point.y += FT_INT_26D6( 1 ) / 2;

            FT_CALL( sdf_edge_get_min_distance( edges,
                                                grid_point,
                                                &dist ) );

            if ( internal_params.orientation == FT_ORIENTATION_FILL_LEFT )
              dist.sign = -dist.sign;

            /* ignore if the distance is greater than spread;       */
            /* otherwise it creates artifacts due to the wrong sign */
            if ( dist.distance > sp_sq )
              continue;

            /* take the square root of the distance if required */
            if ( USE_SQUARED_DISTANCES )
              dist.distance = square_root( dist.distance );

            if ( internal_params.flip_y )
              index = (FT_UInt)( y * width + x );
            else
              index = (FT_UInt)( ( rows - y - 1 ) * width + x );

            /* check whether the pixel is set or not */
            if ( dists[index].sign == 0 )
              dists[index] = dist;
            else
            {
              diff = FT_ABS( dists[index].distance - dist.distance );

              if ( diff <= CORNER_CHECK_EPSILON )
                dists[index] = resolve_corner( dists[index], dist );
              else if ( dists[index].distance > dist.distance )
                dists[index] = dist;
            }
          }
        }

        edges = edges->next;
      }

      contours = contours->next;
    }

    /* final pass */
    for ( j = 0; j < rows; j++ )
    {
      /* We assume the starting pixel of each row is outside. */
      FT_Char  current_sign = -1;
      FT_UInt  index;


      if ( internal_params.overload_sign != 0 )
        current_sign = internal_params.overload_sign < 0 ? -1 : 1;

      for ( i = 0; i < width; i++ )
      {
        index = (FT_UInt)( j * width + i );

        /* if the pixel is not set                     */
        /* its shortest distance is more than `spread` */
        if ( dists[index].sign == 0 )
          dists[index].distance = fixed_spread;
        else
          current_sign = dists[index].sign;

        /* clamp the values */
        if ( dists[index].distance > fixed_spread )
          dists[index].distance = fixed_spread;

        /* flip sign if required */
        dists[index].distance *= internal_params.flip_sign ? -current_sign
                                                           :  current_sign;

        /* concatenate to appropriate format */
        buffer[index] = map_fixed_to_sdf( dists[index].distance,
                                          fixed_spread );
      }
    }

  Exit:
    FT_FREE( dists );
    return error;
  }


  /**************************************************************************
   *
   * @Function:
   *   sdf_generate_subdivision
   *
   * @Description:
   *   Subdivide the shape into a number of straight lines, then use the
   *   above `sdf_generate_bounding_box` function to generate the SDF.
   *
   *   Note: After calling this function `shape` no longer has the original
   *         edges, it only contains lines.
   *
   * @Input:
   *   internal_params ::
   *     Internal parameters and properties required by the rasterizer.
   *     See @SDF_Params for more.
   *
   *   shape ::
   *     A complete shape which is used to generate SDF.
   *
   *   spread ::
   *     Maximum distances to be allowed inthe output bitmap.
   *
   * @Output:
   *   bitmap ::
   *     The output bitmap which will contain the SDF information.
   *
   * @Return:
   *   FreeType error, 0 means success.
   *
   */
  static FT_Error
  sdf_generate_subdivision( const SDF_Params  internal_params,
                            SDF_Shape*        shape,
                            FT_UInt           spread,
                            const FT_Bitmap*  bitmap )
  {
    /*
     * Thanks to Alexei for providing the idea of this optimization.
     *
     * We take advantage of two facts.
     *
     * (1) Computing the shortest distance from a point to a line segment is
     *     very fast.
     * (2) We don't have to compute the shortest distance for the entire
     *     two-dimensional grid.
     *
     * Both ideas lead to the following optimization.
     *
     * (1) Split the outlines into a number of line segments.
     *
     * (2) For each line segment, only process its neighborhood.
     *
     * (3) Compute the closest distance to the line only for neighborhood
     *     grid points.
     *
     * This greatly reduces the number of grid points to check.
     */

    FT_Error  error = FT_Err_Ok;


    FT_CALL( split_sdf_shape( shape ) );
    FT_CALL( sdf_generate_bounding_box( internal_params,
                                        shape, spread, bitmap ) );

  Exit:
    return error;
  }


  /**************************************************************************
   *
   * @Function:
   *   sdf_generate_with_overlaps
   *
   * @Description:
   *   This function can be used to generate SDF for glyphs with overlapping
   *   contours.  The function generates SDF for contours separately on
   *   separate bitmaps (to generate SDF it uses
   *   `sdf_generate_subdivision`).  At the end it simply combines all the
   *   SDF into the output bitmap; this fixes all the signs and removes
   *   overlaps.
   *
   * @Input:
   *   internal_params ::
   *     Internal parameters and properties required by the rasterizer.  See
   *     @SDF_Params for more.
   *
   *   shape ::
   *     A complete shape which is used to generate SDF.
   *
   *   spread ::
   *     Maximum distances to be allowed in the output bitmap.
   *
   * @Output:
   *   bitmap ::
   *     The output bitmap which will contain the SDF information.
   *
   * @Return:
   *   FreeType error, 0 means success.
   *
   * @Note:
   *   The function cannot generate a proper SDF for glyphs with
   *   self-intersecting contours because we cannot separate them into two
   *   separate bitmaps.  In case of self-intersecting contours it is
   *   necessary to remove the overlaps before generating the SDF.
   *
   */
  static FT_Error
  sdf_generate_with_overlaps( SDF_Params        internal_params,
                              SDF_Shape*        shape,
                              FT_UInt           spread,
                              const FT_Bitmap*  bitmap )
  {
    FT_Error  error = FT_Err_Ok;

    FT_Int      num_contours;        /* total number of contours      */
    FT_Int      i, j;                /* iterators                     */
    FT_Int      width, rows;         /* width and rows of the bitmap  */
    FT_Bitmap*  bitmaps;             /* separate bitmaps for contours */

    SDF_Contour*  contour;           /* temporary variable to iterate */
    SDF_Contour*  temp_contour;      /* temporary contour             */
    SDF_Contour*  head;              /* head of the contour list      */
    SDF_Shape     temp_shape;        /* temporary shape               */

    FT_Memory      memory;           /* to allocate memory            */
    FT_SDFFormat*  t;                /* target bitmap buffer          */
    FT_Bool        flip_sign;        /* flip sign?                    */

    /* orientation of all the separate contours */
    SDF_Contour_Orientation*  orientations;


    bitmaps      = NULL;
    orientations = NULL;
    head         = NULL;

    if ( !shape || !bitmap || !shape->memory )
      return FT_THROW( Invalid_Argument );

    /* Disable `flip_sign` to avoid extra complication */
    /* during the combination phase.                   */
    flip_sign                 = internal_params.flip_sign;
    internal_params.flip_sign = 0;

    contour           = shape->contours;
    memory            = shape->memory;
    temp_shape.memory = memory;
    width             = (FT_Int)bitmap->width;
    rows              = (FT_Int)bitmap->rows;
    num_contours      = 0;

    /* find the number of contours in the shape */
    while ( contour )
    {
      num_contours++;
      contour = contour->next;
    }

    /* allocate the bitmaps to generate SDF for separate contours */
    if ( FT_ALLOC( bitmaps,
                   (FT_UInt)num_contours * sizeof ( *bitmaps ) ) )
      goto Exit;

    /* allocate array to hold orientation for all contours */
    if ( FT_ALLOC( orientations,
                   (FT_UInt)num_contours * sizeof ( *orientations ) ) )
      goto Exit;

    contour = shape->contours;

    /* Iterate over all contours and generate SDF separately. */
    for ( i = 0; i < num_contours; i++ )
    {
      /* initialize the corresponding bitmap */
      FT_Bitmap_Init( &bitmaps[i] );

      bitmaps[i].width      = bitmap->width;
      bitmaps[i].rows       = bitmap->rows;
      bitmaps[i].pitch      = bitmap->pitch;
      bitmaps[i].num_grays  = bitmap->num_grays;
      bitmaps[i].pixel_mode = bitmap->pixel_mode;

      /* allocate memory for the buffer */
      if ( FT_ALLOC( bitmaps[i].buffer,
                     bitmap->rows * (FT_UInt)bitmap->pitch ) )
        goto Exit;

      /* determine the orientation */
      orientations[i] = get_contour_orientation( contour );

      /* The `overload_sign` property is specific to  */
      /* `sdf_generate_bounding_box`.  This basically */
      /* overloads the default sign of the outside    */
      /* pixels, which is necessary for               */
      /* counter-clockwise contours.                  */
      if ( orientations[i] == SDF_ORIENTATION_CCW                   &&
           internal_params.orientation == FT_ORIENTATION_FILL_RIGHT )
        internal_params.overload_sign = 1;
      else if ( orientations[i] == SDF_ORIENTATION_CW                   &&
                internal_params.orientation == FT_ORIENTATION_FILL_LEFT )
        internal_params.overload_sign = 1;
      else
        internal_params.overload_sign = 0;

      /* Make `contour->next` NULL so that there is   */
      /* one contour in the list.  Also hold the next */
      /* contour in a temporary variable so as to     */
      /* restore the original value.                  */
      temp_contour  = contour->next;
      contour->next = NULL;

      /* Use `temp_shape` to hold the new contour. */
      /* Now, `temp_shape` has only one contour.   */
      temp_shape.contours = contour;

      /* finally generate the SDF */
      FT_CALL( sdf_generate_subdivision( internal_params,
                                         &temp_shape,
                                         spread,
                                         &bitmaps[i] ) );

      /* Restore the original `next` variable. */
      contour->next = temp_contour;

      /* Since `split_sdf_shape` deallocated the original */
      /* contours list we need to assign the new value to */
      /* the shape's contour.                             */
      temp_shape.contours->next = head;
      head                      = temp_shape.contours;

      /* Simply flip the orientation in case of post-script fonts */
      /* so as to avoid modificatons in the combining phase.      */
      if ( internal_params.orientation == FT_ORIENTATION_FILL_LEFT )
      {
        if ( orientations[i] == SDF_ORIENTATION_CW )
          orientations[i] = SDF_ORIENTATION_CCW;
        else if ( orientations[i] == SDF_ORIENTATION_CCW )
          orientations[i] = SDF_ORIENTATION_CW;
      }

      contour = contour->next;
    }

    /* assign the new contour list to `shape->contours` */
    shape->contours = head;

    /* cast the output bitmap buffer */
    t = (FT_SDFFormat*)bitmap->buffer;

    /* Iterate over all pixels and combine all separate    */
    /* contours.  These are the rules for combining:       */
    /*                                                     */
    /* (1) For all clockwise contours, compute the largest */
    /*     value.  Name this as `val_c`.                   */
    /* (2) For all counter-clockwise contours, compute the */
    /*     smallest value.  Name this as `val_ac`.         */
    /* (3) Now, finally use the smaller value of `val_c'   */
    /*     and `val_ac'.                                   */
    for ( j = 0; j < rows; j++ )
    {
      for ( i = 0; i < width; i++ )
      {
        FT_Int  id = j * width + i;       /* index of current pixel    */
        FT_Int  c;                        /* contour iterator          */

        FT_SDFFormat  val_c  = 0;         /* max clockwise value       */
        FT_SDFFormat  val_ac = UCHAR_MAX; /* min counter-clockwise val */


        /* iterate through all the contours */
        for ( c = 0; c < num_contours; c++ )
        {
          /* current contour value */
          FT_SDFFormat  temp = ( (FT_SDFFormat*)bitmaps[c].buffer )[id];


          if ( orientations[c] == SDF_ORIENTATION_CW )
            val_c = FT_MAX( val_c, temp );   /* clockwise         */
          else
            val_ac = FT_MIN( val_ac, temp ); /* counter-clockwise */
        }

        /* Finally find the smaller of the two and assign to output. */
        /* Also apply `flip_sign` if set.                            */
        t[id] = FT_MIN( val_c, val_ac );

        if ( flip_sign )
          t[id] = invert_sign( t[id] );
      }
    }

  Exit:
    /* deallocate orientations array */
    if ( orientations )
      FT_FREE( orientations );

    /* deallocate temporary bitmaps */
    if ( bitmaps )
    {
      if ( num_contours == 0 )
        error = FT_THROW( Raster_Corrupted );
      else
      {
        for ( i = 0; i < num_contours; i++ )
          FT_FREE( bitmaps[i].buffer );

        FT_FREE( bitmaps );
      }
    }

    /* restore the `flip_sign` property */
    internal_params.flip_sign = flip_sign;

    return error;
  }


  /**************************************************************************
   *
   * interface functions
   *
   */

  static FT_Error
  sdf_raster_new( void*       memory_,   /* FT_Memory    */
                  FT_Raster*  araster_ ) /* SDF_PRaster* */
  {
    FT_Memory     memory  = (FT_Memory)memory_;
    SDF_PRaster*  araster = (SDF_PRaster*)araster_;


    FT_Error     error;
    SDF_PRaster  raster = NULL;


    if ( !FT_NEW( raster ) )
      raster->memory = memory;

    *araster = raster;

   return error;
  }


  static void
  sdf_raster_reset( FT_Raster       raster,
                    unsigned char*  pool_base,
                    unsigned long   pool_size )
  {
    FT_UNUSED( raster );
    FT_UNUSED( pool_base );
    FT_UNUSED( pool_size );
  }


  static FT_Error
  sdf_raster_set_mode( FT_Raster      raster,
                       unsigned long  mode,
                       void*          args )
  {
    FT_UNUSED( raster );
    FT_UNUSED( mode );
    FT_UNUSED( args );

    return FT_Err_Ok;
  }


  static FT_Error
  sdf_raster_render( FT_Raster                raster,
                     const FT_Raster_Params*  params )
  {
    FT_Error                  error      = FT_Err_Ok;
    SDF_TRaster*              sdf_raster = (SDF_TRaster*)raster;
    FT_Outline*               outline    = NULL;
    const SDF_Raster_Params*  sdf_params = (const SDF_Raster_Params*)params;

    FT_Memory   memory = NULL;
    SDF_Shape*  shape  = NULL;
    SDF_Params  internal_params;


    /* check for valid arguments */
    if ( !sdf_raster || !sdf_params )
    {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    outline = (FT_Outline*)sdf_params->root.source;

    /* check whether outline is valid */
    if ( !outline )
    {
      error = FT_THROW( Invalid_Outline );
      goto Exit;
    }

    /* if the outline is empty, return */
    if ( outline->n_points == 0 || outline->n_contours == 0 )
      goto Exit;

    /* check whether the outline has valid fields */
    if ( !outline->contours || !outline->points )
    {
      error = FT_THROW( Invalid_Outline );
      goto Exit;
    }

    /* check whether spread is set properly */
    if ( sdf_params->spread > MAX_SPREAD ||
         sdf_params->spread < MIN_SPREAD )
    {
      FT_TRACE0(( "sdf_raster_render:"
                  " The `spread' field of `SDF_Raster_Params' is invalid,\n" ));
      FT_TRACE0(( "                  "
                  " the value of this field must be within [%d, %d].\n",
                  MIN_SPREAD, MAX_SPREAD ));
      FT_TRACE0(( "                  "
                  " Also, you must pass `SDF_Raster_Params' instead of\n" ));
      FT_TRACE0(( "                  "
                  " the default `FT_Raster_Params' while calling\n" ));
      FT_TRACE0(( "                  "
                  " this function and set the fields properly.\n" ));

      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    memory = sdf_raster->memory;
    if ( !memory )
    {
      FT_TRACE0(( "sdf_raster_render:"
                  " Raster not setup properly,\n" ));
      FT_TRACE0(( "                  "
                  " unable to find memory handle.\n" ));

      error = FT_THROW( Invalid_Handle );
      goto Exit;
    }

    /* set up the parameters */
    internal_params.orientation   = FT_Outline_Get_Orientation( outline );
    internal_params.flip_sign     = sdf_params->flip_sign;
    internal_params.flip_y        = sdf_params->flip_y;
    internal_params.overload_sign = 0;

    FT_CALL( sdf_shape_new( memory, &shape ) );

    FT_CALL( sdf_outline_decompose( outline, shape ) );

    if ( sdf_params->overlaps )
      FT_CALL( sdf_generate_with_overlaps( internal_params,
                                           shape, sdf_params->spread,
                                           sdf_params->root.target ) );
    else
      FT_CALL( sdf_generate_subdivision( internal_params,
                                         shape, sdf_params->spread,
                                         sdf_params->root.target ) );

    if ( shape )
      sdf_shape_done( &shape );

  Exit:
    return error;
  }


  static void
  sdf_raster_done( FT_Raster  raster )
  {
    FT_Memory  memory = (FT_Memory)((SDF_TRaster*)raster)->memory;


    FT_FREE( raster );
  }


  FT_DEFINE_RASTER_FUNCS(
    ft_sdf_raster,

    FT_GLYPH_FORMAT_OUTLINE,

    (FT_Raster_New_Func)     sdf_raster_new,       /* raster_new      */
    (FT_Raster_Reset_Func)   sdf_raster_reset,     /* raster_reset    */
    (FT_Raster_Set_Mode_Func)sdf_raster_set_mode,  /* raster_set_mode */
    (FT_Raster_Render_Func)  sdf_raster_render,    /* raster_render   */
    (FT_Raster_Done_Func)    sdf_raster_done       /* raster_done     */
  )


/* END */
