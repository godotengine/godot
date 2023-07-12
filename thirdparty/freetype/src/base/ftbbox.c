/****************************************************************************
 *
 * ftbbox.c
 *
 *   FreeType bbox computation (body).
 *
 * Copyright (C) 1996-2023 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used
 * modified and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


  /**************************************************************************
   *
   * This component has a _single_ role: to compute exact outline bounding
   * boxes.
   *
   */


#include <freetype/internal/ftdebug.h>

#include <freetype/ftbbox.h>
#include <freetype/ftimage.h>
#include <freetype/ftoutln.h>
#include <freetype/internal/ftcalc.h>
#include <freetype/internal/ftobjs.h>


  typedef struct  TBBox_Rec_
  {
    FT_Vector  last;
    FT_BBox    bbox;

  } TBBox_Rec;


#define FT_UPDATE_BBOX( p, bbox ) \
  FT_BEGIN_STMNT                  \
    if ( p->x < bbox.xMin )       \
      bbox.xMin = p->x;           \
    if ( p->x > bbox.xMax )       \
      bbox.xMax = p->x;           \
    if ( p->y < bbox.yMin )       \
      bbox.yMin = p->y;           \
    if ( p->y > bbox.yMax )       \
      bbox.yMax = p->y;           \
  FT_END_STMNT

#define CHECK_X( p, bbox )                         \
          ( p->x < bbox.xMin || p->x > bbox.xMax )

#define CHECK_Y( p, bbox )                         \
          ( p->y < bbox.yMin || p->y > bbox.yMax )


  /**************************************************************************
   *
   * @Function:
   *   BBox_Move_To
   *
   * @Description:
   *   This function is used as a `move_to' emitter during
   *   FT_Outline_Decompose().  It simply records the destination point
   *   in `user->last'. We also update bbox in case contour starts with
   *   an implicit `on' point.
   *
   * @Input:
   *   to ::
   *     A pointer to the destination vector.
   *
   * @InOut:
   *   user ::
   *     A pointer to the current walk context.
   *
   * @Return:
   *   Always 0.  Needed for the interface only.
   */
  static int
  BBox_Move_To( FT_Vector*  to,
                TBBox_Rec*  user )
  {
    FT_UPDATE_BBOX( to, user->bbox );

    user->last = *to;

    return 0;
  }


  /**************************************************************************
   *
   * @Function:
   *   BBox_Line_To
   *
   * @Description:
   *   This function is used as a `line_to' emitter during
   *   FT_Outline_Decompose().  It simply records the destination point
   *   in `user->last'; no further computations are necessary because
   *   bbox already contains both explicit ends of the line segment.
   *
   * @Input:
   *   to ::
   *     A pointer to the destination vector.
   *
   * @InOut:
   *   user ::
   *     A pointer to the current walk context.
   *
   * @Return:
   *   Always 0.  Needed for the interface only.
   */
  static int
  BBox_Line_To( FT_Vector*  to,
                TBBox_Rec*  user )
  {
    user->last = *to;

    return 0;
  }


  /**************************************************************************
   *
   * @Function:
   *   BBox_Conic_Check
   *
   * @Description:
   *   Find the extrema of a 1-dimensional conic Bezier curve and update
   *   a bounding range.  This version uses direct computation, as it
   *   doesn't need square roots.
   *
   * @Input:
   *   y1 ::
   *     The start coordinate.
   *
   *   y2 ::
   *     The coordinate of the control point.
   *
   *   y3 ::
   *     The end coordinate.
   *
   * @InOut:
   *   min ::
   *     The address of the current minimum.
   *
   *   max ::
   *     The address of the current maximum.
   */
  static void
  BBox_Conic_Check( FT_Pos   y1,
                    FT_Pos   y2,
                    FT_Pos   y3,
                    FT_Pos*  min,
                    FT_Pos*  max )
  {
    /* This function is only called when a control off-point is outside */
    /* the bbox that contains all on-points.  It finds a local extremum */
    /* within the segment, equal to (y1*y3 - y2*y2)/(y1 - 2*y2 + y3).   */
    /* Or, offsetting from y2, we get                                   */

    y1 -= y2;
    y3 -= y2;
    y2 += FT_MulDiv( y1, y3, y1 + y3 );

    if ( y2 < *min )
      *min = y2;
    if ( y2 > *max )
      *max = y2;
  }


  /**************************************************************************
   *
   * @Function:
   *   BBox_Conic_To
   *
   * @Description:
   *   This function is used as a `conic_to' emitter during
   *   FT_Outline_Decompose().  It checks a conic Bezier curve with the
   *   current bounding box, and computes its extrema if necessary to
   *   update it.
   *
   * @Input:
   *   control ::
   *     A pointer to a control point.
   *
   *   to ::
   *     A pointer to the destination vector.
   *
   * @InOut:
   *   user ::
   *     The address of the current walk context.
   *
   * @Return:
   *   Always 0.  Needed for the interface only.
   *
   * @Note:
   *   In the case of a non-monotonous arc, we compute directly the
   *   extremum coordinates, as it is sufficiently fast.
   */
  static int
  BBox_Conic_To( FT_Vector*  control,
                 FT_Vector*  to,
                 TBBox_Rec*  user )
  {
    /* in case `to' is implicit and not included in bbox yet */
    FT_UPDATE_BBOX( to, user->bbox );

    if ( CHECK_X( control, user->bbox ) )
      BBox_Conic_Check( user->last.x,
                        control->x,
                        to->x,
                        &user->bbox.xMin,
                        &user->bbox.xMax );

    if ( CHECK_Y( control, user->bbox ) )
      BBox_Conic_Check( user->last.y,
                        control->y,
                        to->y,
                        &user->bbox.yMin,
                        &user->bbox.yMax );

    user->last = *to;

    return 0;
  }


  /**************************************************************************
   *
   * @Function:
   *   BBox_Cubic_Check
   *
   * @Description:
   *   Find the extrema of a 1-dimensional cubic Bezier curve and
   *   update a bounding range.  This version uses iterative splitting
   *   because it is faster than the exact solution with square roots.
   *
   * @Input:
   *   p1 ::
   *     The start coordinate.
   *
   *   p2 ::
   *     The coordinate of the first control point.
   *
   *   p3 ::
   *     The coordinate of the second control point.
   *
   *   p4 ::
   *     The end coordinate.
   *
   * @InOut:
   *   min ::
   *     The address of the current minimum.
   *
   *   max ::
   *     The address of the current maximum.
   */
  static FT_Pos
  cubic_peak( FT_Pos  q1,
              FT_Pos  q2,
              FT_Pos  q3,
              FT_Pos  q4 )
  {
    FT_Pos  peak = 0;
    FT_Int  shift;


    /* This function finds a peak of a cubic segment if it is above 0    */
    /* using iterative bisection of the segment, or returns 0.           */
    /* The fixed-point arithmetic of bisection is inherently stable      */
    /* but may loose accuracy in the two lowest bits.  To compensate,    */
    /* we upscale the segment if there is room.  Large values may need   */
    /* to be downscaled to avoid overflows during bisection.             */
    /* It is called with either q2 or q3 positive, which is necessary    */
    /* for the peak to exist and avoids undefined FT_MSB.                */

    shift = 27 - FT_MSB( (FT_UInt32)( FT_ABS( q1 ) |
                                      FT_ABS( q2 ) |
                                      FT_ABS( q3 ) |
                                      FT_ABS( q4 ) ) );

    if ( shift > 0 )
    {
      /* upscaling too much just wastes time */
      if ( shift > 2 )
        shift = 2;

      q1 *= 1 << shift;
      q2 *= 1 << shift;
      q3 *= 1 << shift;
      q4 *= 1 << shift;
    }
    else
    {
      q1 >>= -shift;
      q2 >>= -shift;
      q3 >>= -shift;
      q4 >>= -shift;
    }

    /* for a peak to exist above 0, the cubic segment must have */
    /* at least one of its control off-points above 0.          */
    while ( q2 > 0 || q3 > 0 )
    {
      /* determine which half contains the maximum and split */
      if ( q1 + q2 > q3 + q4 ) /* first half */
      {
        q4 = q4 + q3;
        q3 = q3 + q2;
        q2 = q2 + q1;
        q4 = q4 + q3;
        q3 = q3 + q2;
        q4 = ( q4 + q3 ) >> 3;
        q3 = q3 >> 2;
        q2 = q2 >> 1;
      }
      else                     /* second half */
      {
        q1 = q1 + q2;
        q2 = q2 + q3;
        q3 = q3 + q4;
        q1 = q1 + q2;
        q2 = q2 + q3;
        q1 = ( q1 + q2 ) >> 3;
        q2 = q2 >> 2;
        q3 = q3 >> 1;
      }

      /* check whether either end reached the maximum */
      if ( q1 == q2 && q1 >= q3 )
      {
        peak = q1;
        break;
      }
      if ( q3 == q4 && q2 <= q4 )
      {
        peak = q4;
        break;
      }
    }

    if ( shift > 0 )
      peak >>=  shift;
    else
      peak <<= -shift;

    return peak;
  }


  static void
  BBox_Cubic_Check( FT_Pos   p1,
                    FT_Pos   p2,
                    FT_Pos   p3,
                    FT_Pos   p4,
                    FT_Pos*  min,
                    FT_Pos*  max )
  {
    /* This function is only called when a control off-point is outside  */
    /* the bbox that contains all on-points.  So at least one of the     */
    /* conditions below holds and cubic_peak is called with at least one */
    /* non-zero argument.                                                */

    if ( p2 > *max || p3 > *max )
      *max += cubic_peak( p1 - *max, p2 - *max, p3 - *max, p4 - *max );

    /* now flip the signs to update the minimum */
    if ( p2 < *min || p3 < *min )
      *min -= cubic_peak( *min - p1, *min - p2, *min - p3, *min - p4 );
  }


  /**************************************************************************
   *
   * @Function:
   *   BBox_Cubic_To
   *
   * @Description:
   *   This function is used as a `cubic_to' emitter during
   *   FT_Outline_Decompose().  It checks a cubic Bezier curve with the
   *   current bounding box, and computes its extrema if necessary to
   *   update it.
   *
   * @Input:
   *   control1 ::
   *     A pointer to the first control point.
   *
   *   control2 ::
   *     A pointer to the second control point.
   *
   *   to ::
   *     A pointer to the destination vector.
   *
   * @InOut:
   *   user ::
   *     The address of the current walk context.
   *
   * @Return:
   *   Always 0.  Needed for the interface only.
   *
   * @Note:
   *   In the case of a non-monotonous arc, we don't compute directly
   *   extremum coordinates, we subdivide instead.
   */
  static int
  BBox_Cubic_To( FT_Vector*  control1,
                 FT_Vector*  control2,
                 FT_Vector*  to,
                 TBBox_Rec*  user )
  {
    /* We don't need to check `to' since it is always an on-point,    */
    /* thus within the bbox.  Only segments with an off-point outside */
    /* the bbox can possibly reach new extreme values.                */

    if ( CHECK_X( control1, user->bbox ) ||
         CHECK_X( control2, user->bbox ) )
      BBox_Cubic_Check( user->last.x,
                        control1->x,
                        control2->x,
                        to->x,
                        &user->bbox.xMin,
                        &user->bbox.xMax );

    if ( CHECK_Y( control1, user->bbox ) ||
         CHECK_Y( control2, user->bbox ) )
      BBox_Cubic_Check( user->last.y,
                        control1->y,
                        control2->y,
                        to->y,
                        &user->bbox.yMin,
                        &user->bbox.yMax );

    user->last = *to;

    return 0;
  }


  FT_DEFINE_OUTLINE_FUNCS(
    bbox_interface,

    (FT_Outline_MoveTo_Func) BBox_Move_To,   /* move_to  */
    (FT_Outline_LineTo_Func) BBox_Line_To,   /* line_to  */
    (FT_Outline_ConicTo_Func)BBox_Conic_To,  /* conic_to */
    (FT_Outline_CubicTo_Func)BBox_Cubic_To,  /* cubic_to */
    0,                                       /* shift    */
    0                                        /* delta    */
  )


  /* documentation is in ftbbox.h */

  FT_EXPORT_DEF( FT_Error )
  FT_Outline_Get_BBox( FT_Outline*  outline,
                       FT_BBox     *abbox )
  {
    FT_BBox     cbox = {  0x7FFFFFFFL,  0x7FFFFFFFL,
                         -0x7FFFFFFFL, -0x7FFFFFFFL };
    FT_BBox     bbox = {  0x7FFFFFFFL,  0x7FFFFFFFL,
                         -0x7FFFFFFFL, -0x7FFFFFFFL };
    FT_Vector*  vec;
    FT_UShort   n;


    if ( !abbox )
      return FT_THROW( Invalid_Argument );

    if ( !outline )
      return FT_THROW( Invalid_Outline );

    /* if outline is empty, return (0,0,0,0) */
    if ( outline->n_points == 0 || outline->n_contours <= 0 )
    {
      abbox->xMin = abbox->xMax = 0;
      abbox->yMin = abbox->yMax = 0;

      return 0;
    }

    /* We compute the control box as well as the bounding box of  */
    /* all `on' points in the outline.  Then, if the two boxes    */
    /* coincide, we exit immediately.                             */

    vec = outline->points;

    for ( n = 0; n < outline->n_points; n++ )
    {
      FT_UPDATE_BBOX( vec, cbox );

      if ( FT_CURVE_TAG( outline->tags[n] ) == FT_CURVE_TAG_ON )
        FT_UPDATE_BBOX( vec, bbox );

      vec++;
    }

    /* test two boxes for equality */
    if ( cbox.xMin < bbox.xMin || cbox.xMax > bbox.xMax ||
         cbox.yMin < bbox.yMin || cbox.yMax > bbox.yMax )
    {
      /* the two boxes are different, now walk over the outline to */
      /* get the Bezier arc extrema.                               */

      FT_Error   error;
      TBBox_Rec  user;


      user.bbox = bbox;

      error = FT_Outline_Decompose( outline, &bbox_interface, &user );
      if ( error )
        return error;

      *abbox = user.bbox;
    }
    else
      *abbox = bbox;

    return FT_Err_Ok;
  }


/* END */
