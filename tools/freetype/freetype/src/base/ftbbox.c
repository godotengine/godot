/***************************************************************************/
/*                                                                         */
/*  ftbbox.c                                                               */
/*                                                                         */
/*    FreeType bbox computation (body).                                    */
/*                                                                         */
/*  Copyright 1996-2002, 2004, 2006, 2010, 2013 by                         */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used        */
/*  modified and distributed under the terms of the FreeType project       */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


  /*************************************************************************/
  /*                                                                       */
  /* This component has a _single_ role: to compute exact outline bounding */
  /* boxes.                                                                */
  /*                                                                       */
  /*************************************************************************/


#include <ft2build.h>
#include FT_INTERNAL_DEBUG_H

#include FT_BBOX_H
#include FT_IMAGE_H
#include FT_OUTLINE_H
#include FT_INTERNAL_CALC_H
#include FT_INTERNAL_OBJECTS_H


  typedef struct  TBBox_Rec_
  {
    FT_Vector  last;
    FT_BBox    bbox;

  } TBBox_Rec;


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    BBox_Move_To                                                       */
  /*                                                                       */
  /* <Description>                                                         */
  /*    This function is used as a `move_to' and `line_to' emitter during  */
  /*    FT_Outline_Decompose().  It simply records the destination point   */
  /*    in `user->last'; no further computations are necessary since we    */
  /*    use the cbox as the starting bbox which must be refined.           */
  /*                                                                       */
  /* <Input>                                                               */
  /*    to   :: A pointer to the destination vector.                       */
  /*                                                                       */
  /* <InOut>                                                               */
  /*    user :: A pointer to the current walk context.                     */
  /*                                                                       */
  /* <Return>                                                              */
  /*    Always 0.  Needed for the interface only.                          */
  /*                                                                       */
  static int
  BBox_Move_To( FT_Vector*  to,
                TBBox_Rec*  user )
  {
    user->last = *to;

    return 0;
  }


#define CHECK_X( p, bbox )  \
          ( p->x < bbox.xMin || p->x > bbox.xMax )

#define CHECK_Y( p, bbox )  \
          ( p->y < bbox.yMin || p->y > bbox.yMax )


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    BBox_Conic_Check                                                   */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Finds the extrema of a 1-dimensional conic Bezier curve and update */
  /*    a bounding range.  This version uses direct computation, as it     */
  /*    doesn't need square roots.                                         */
  /*                                                                       */
  /* <Input>                                                               */
  /*    y1  :: The start coordinate.                                       */
  /*                                                                       */
  /*    y2  :: The coordinate of the control point.                        */
  /*                                                                       */
  /*    y3  :: The end coordinate.                                         */
  /*                                                                       */
  /* <InOut>                                                               */
  /*    min :: The address of the current minimum.                         */
  /*                                                                       */
  /*    max :: The address of the current maximum.                         */
  /*                                                                       */
  static void
  BBox_Conic_Check( FT_Pos   y1,
                    FT_Pos   y2,
                    FT_Pos   y3,
                    FT_Pos*  min,
                    FT_Pos*  max )
  {
    if ( y1 <= y3 && y2 == y1 )     /* flat arc */
      goto Suite;

    if ( y1 < y3 )
    {
      if ( y2 >= y1 && y2 <= y3 )   /* ascending arc */
        goto Suite;
    }
    else
    {
      if ( y2 >= y3 && y2 <= y1 )   /* descending arc */
      {
        y2 = y1;
        y1 = y3;
        y3 = y2;
        goto Suite;
      }
    }

    y1 = y3 = y1 - FT_MulDiv( y2 - y1, y2 - y1, y1 - 2*y2 + y3 );

  Suite:
    if ( y1 < *min ) *min = y1;
    if ( y3 > *max ) *max = y3;
  }


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    BBox_Conic_To                                                      */
  /*                                                                       */
  /* <Description>                                                         */
  /*    This function is used as a `conic_to' emitter during               */
  /*    FT_Outline_Decompose().  It checks a conic Bezier curve with the   */
  /*    current bounding box, and computes its extrema if necessary to     */
  /*    update it.                                                         */
  /*                                                                       */
  /* <Input>                                                               */
  /*    control :: A pointer to a control point.                           */
  /*                                                                       */
  /*    to      :: A pointer to the destination vector.                    */
  /*                                                                       */
  /* <InOut>                                                               */
  /*    user    :: The address of the current walk context.                */
  /*                                                                       */
  /* <Return>                                                              */
  /*    Always 0.  Needed for the interface only.                          */
  /*                                                                       */
  /* <Note>                                                                */
  /*    In the case of a non-monotonous arc, we compute directly the       */
  /*    extremum coordinates, as it is sufficiently fast.                  */
  /*                                                                       */
  static int
  BBox_Conic_To( FT_Vector*  control,
                 FT_Vector*  to,
                 TBBox_Rec*  user )
  {
    /* we don't need to check `to' since it is always an `on' point, thus */
    /* within the bbox                                                    */

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


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    BBox_Cubic_Check                                                   */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Finds the extrema of a 1-dimensional cubic Bezier curve and        */
  /*    updates a bounding range.  This version uses splitting because we  */
  /*    don't want to use square roots and extra accuracy.                 */
  /*                                                                       */
  /* <Input>                                                               */
  /*    p1  :: The start coordinate.                                       */
  /*                                                                       */
  /*    p2  :: The coordinate of the first control point.                  */
  /*                                                                       */
  /*    p3  :: The coordinate of the second control point.                 */
  /*                                                                       */
  /*    p4  :: The end coordinate.                                         */
  /*                                                                       */
  /* <InOut>                                                               */
  /*    min :: The address of the current minimum.                         */
  /*                                                                       */
  /*    max :: The address of the current maximum.                         */
  /*                                                                       */

#if 0

  static void
  BBox_Cubic_Check( FT_Pos   p1,
                    FT_Pos   p2,
                    FT_Pos   p3,
                    FT_Pos   p4,
                    FT_Pos*  min,
                    FT_Pos*  max )
  {
    FT_Pos  q1, q2, q3, q4;


    q1 = p1;
    q2 = p2;
    q3 = p3;
    q4 = p4;

    /* for a conic segment to possibly reach new maximum     */
    /* one of its off-points must be above the current value */
    while ( q2 > *max || q3 > *max )
    {
      /* determine which half contains the maximum and split */
      if ( q1 + q2 > q3 + q4 ) /* first half */
      {
        q4 = q4 + q3;
        q3 = q3 + q2;
        q2 = q2 + q1;
        q4 = q4 + q3;
        q3 = q3 + q2;
        q4 = ( q4 + q3 ) / 8;
        q3 = q3 / 4;
        q2 = q2 / 2;
      }
      else                     /* second half */
      {
        q1 = q1 + q2;
        q2 = q2 + q3;
        q3 = q3 + q4;
        q1 = q1 + q2;
        q2 = q2 + q3;
        q1 = ( q1 + q2 ) / 8;
        q2 = q2 / 4;
        q3 = q3 / 2;
      }

      /* check if either end reached the maximum */
      if ( q1 == q2 && q1 >= q3 )
      {
        *max = q1;
        break;
      }
      if ( q3 == q4 && q2 <= q4 )
      {
        *max = q4;
        break;
      }
    }

    q1 = p1;
    q2 = p2;
    q3 = p3;
    q4 = p4;

    /* for a conic segment to possibly reach new minimum     */
    /* one of its off-points must be below the current value */
    while ( q2 < *min || q3 < *min )
    {
      /* determine which half contains the minimum and split */
      if ( q1 + q2 < q3 + q4 ) /* first half */
      {
        q4 = q4 + q3;
        q3 = q3 + q2;
        q2 = q2 + q1;
        q4 = q4 + q3;
        q3 = q3 + q2;
        q4 = ( q4 + q3 ) / 8;
        q3 = q3 / 4;
        q2 = q2 / 2;
      }
      else                     /* second half */
      {
        q1 = q1 + q2;
        q2 = q2 + q3;
        q3 = q3 + q4;
        q1 = q1 + q2;
        q2 = q2 + q3;
        q1 = ( q1 + q2 ) / 8;
        q2 = q2 / 4;
        q3 = q3 / 2;
      }

      /* check if either end reached the minimum */
      if ( q1 == q2 && q1 <= q3 )
      {
        *min = q1;
        break;
      }
      if ( q3 == q4 && q2 >= q4 )
      {
        *min = q4;
        break;
      }
    }
  }

#else

  static void
  test_cubic_extrema( FT_Pos    y1,
                      FT_Pos    y2,
                      FT_Pos    y3,
                      FT_Pos    y4,
                      FT_Fixed  u,
                      FT_Pos*   min,
                      FT_Pos*   max )
  {
 /* FT_Pos    a = y4 - 3*y3 + 3*y2 - y1; */
    FT_Pos    b = y3 - 2*y2 + y1;
    FT_Pos    c = y2 - y1;
    FT_Pos    d = y1;
    FT_Pos    y;
    FT_Fixed  uu;

    FT_UNUSED ( y4 );


    /* The polynomial is                      */
    /*                                        */
    /*    P(x) = a*x^3 + 3b*x^2 + 3c*x + d  , */
    /*                                        */
    /*   dP/dx = 3a*x^2 + 6b*x + 3c         . */
    /*                                        */
    /* However, we also have                  */
    /*                                        */
    /*   dP/dx(u) = 0                       , */
    /*                                        */
    /* which implies by subtraction that      */
    /*                                        */
    /*   P(u) = b*u^2 + 2c*u + d            . */

    if ( u > 0 && u < 0x10000L )
    {
      uu = FT_MulFix( u, u );
      y  = d + FT_MulFix( c, 2*u ) + FT_MulFix( b, uu );

      if ( y < *min ) *min = y;
      if ( y > *max ) *max = y;
    }
  }


  static void
  BBox_Cubic_Check( FT_Pos   y1,
                    FT_Pos   y2,
                    FT_Pos   y3,
                    FT_Pos   y4,
                    FT_Pos*  min,
                    FT_Pos*  max )
  {
    /* always compare first and last points */
    if      ( y1 < *min )  *min = y1;
    else if ( y1 > *max )  *max = y1;

    if      ( y4 < *min )  *min = y4;
    else if ( y4 > *max )  *max = y4;

    /* now, try to see if there are split points here */
    if ( y1 <= y4 )
    {
      /* flat or ascending arc test */
      if ( y1 <= y2 && y2 <= y4 && y1 <= y3 && y3 <= y4 )
        return;
    }
    else /* y1 > y4 */
    {
      /* descending arc test */
      if ( y1 >= y2 && y2 >= y4 && y1 >= y3 && y3 >= y4 )
        return;
    }

    /* There are some split points.  Find them.                        */
    /* We already made sure that a, b, and c below cannot be all zero. */
    {
      FT_Pos    a = y4 - 3*y3 + 3*y2 - y1;
      FT_Pos    b = y3 - 2*y2 + y1;
      FT_Pos    c = y2 - y1;
      FT_Pos    d;
      FT_Fixed  t;
      FT_Int    shift;


      /* We need to solve `ax^2+2bx+c' here, without floating points!      */
      /* The trick is to normalize to a different representation in order  */
      /* to use our 16.16 fixed-point routines.                            */
      /*                                                                   */
      /* We compute FT_MulFix(b,b) and FT_MulFix(a,c) after normalization. */
      /* These values must fit into a single 16.16 value.                  */
      /*                                                                   */
      /* We normalize a, b, and c to `8.16' fixed-point values to ensure   */
      /* that their product is held in a `16.16' value including the sign. */
      /* Necessarily, we need to shift `a', `b', and `c' so that the most  */
      /* significant bit of their absolute values is at position 22.       */
      /*                                                                   */
      /* This also means that we are using 23 bits of precision to compute */
      /* the zeros, independently of the range of the original polynomial  */
      /* coefficients.                                                     */
      /*                                                                   */
      /* This algorithm should ensure reasonably accurate values for the   */
      /* zeros.  Note that they are only expressed with 16 bits when       */
      /* computing the extrema (the zeros need to be in 0..1 exclusive     */
      /* to be considered part of the arc).                                */

      shift = FT_MSB( FT_ABS( a ) | FT_ABS( b ) | FT_ABS( c ) );

      if ( shift > 22 )
      {
        shift -= 22;

        /* this loses some bits of precision, but we use 23 of them */
        /* for the computation anyway                               */
        a >>= shift;
        b >>= shift;
        c >>= shift;
      }
      else
      {
        shift = 22 - shift;

        a <<= shift;
        b <<= shift;
        c <<= shift;
      }

      /* handle a == 0 */
      if ( a == 0 )
      {
        if ( b != 0 )
        {
          t = - FT_DivFix( c, b ) / 2;
          test_cubic_extrema( y1, y2, y3, y4, t, min, max );
        }
      }
      else
      {
        /* solve the equation now */
        d = FT_MulFix( b, b ) - FT_MulFix( a, c );
        if ( d < 0 )
          return;

        if ( d == 0 )
        {
          /* there is a single split point at -b/a */
          t = - FT_DivFix( b, a );
          test_cubic_extrema( y1, y2, y3, y4, t, min, max );
        }
        else
        {
          /* there are two solutions; we need to filter them */
          d = FT_SqrtFixed( (FT_Int32)d );
          t = - FT_DivFix( b - d, a );
          test_cubic_extrema( y1, y2, y3, y4, t, min, max );

          t = - FT_DivFix( b + d, a );
          test_cubic_extrema( y1, y2, y3, y4, t, min, max );
        }
      }
    }
  }

#endif


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    BBox_Cubic_To                                                      */
  /*                                                                       */
  /* <Description>                                                         */
  /*    This function is used as a `cubic_to' emitter during               */
  /*    FT_Outline_Decompose().  It checks a cubic Bezier curve with the   */
  /*    current bounding box, and computes its extrema if necessary to     */
  /*    update it.                                                         */
  /*                                                                       */
  /* <Input>                                                               */
  /*    control1 :: A pointer to the first control point.                  */
  /*                                                                       */
  /*    control2 :: A pointer to the second control point.                 */
  /*                                                                       */
  /*    to       :: A pointer to the destination vector.                   */
  /*                                                                       */
  /* <InOut>                                                               */
  /*    user     :: The address of the current walk context.               */
  /*                                                                       */
  /* <Return>                                                              */
  /*    Always 0.  Needed for the interface only.                          */
  /*                                                                       */
  /* <Note>                                                                */
  /*    In the case of a non-monotonous arc, we don't compute directly     */
  /*    extremum coordinates, we subdivide instead.                        */
  /*                                                                       */
  static int
  BBox_Cubic_To( FT_Vector*  control1,
                 FT_Vector*  control2,
                 FT_Vector*  to,
                 TBBox_Rec*  user )
  {
    /* we don't need to check `to' since it is always an `on' point, thus */
    /* within the bbox                                                    */

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

FT_DEFINE_OUTLINE_FUNCS(bbox_interface,
    (FT_Outline_MoveTo_Func) BBox_Move_To,
    (FT_Outline_LineTo_Func) BBox_Move_To,
    (FT_Outline_ConicTo_Func)BBox_Conic_To,
    (FT_Outline_CubicTo_Func)BBox_Cubic_To,
    0, 0
  )

  /* documentation is in ftbbox.h */

  FT_EXPORT_DEF( FT_Error )
  FT_Outline_Get_BBox( FT_Outline*  outline,
                       FT_BBox     *abbox )
  {
    FT_BBox     cbox;
    FT_BBox     bbox;
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
    bbox.xMin = bbox.xMax = cbox.xMin = cbox.xMax = vec->x;
    bbox.yMin = bbox.yMax = cbox.yMin = cbox.yMax = vec->y;
    vec++;

    for ( n = 1; n < outline->n_points; n++ )
    {
      FT_Pos  x = vec->x;
      FT_Pos  y = vec->y;


      /* update control box */
      if ( x < cbox.xMin ) cbox.xMin = x;
      if ( x > cbox.xMax ) cbox.xMax = x;

      if ( y < cbox.yMin ) cbox.yMin = y;
      if ( y > cbox.yMax ) cbox.yMax = y;

      if ( FT_CURVE_TAG( outline->tags[n] ) == FT_CURVE_TAG_ON )
      {
        /* update bbox for `on' points only */
        if ( x < bbox.xMin ) bbox.xMin = x;
        if ( x > bbox.xMax ) bbox.xMax = x;

        if ( y < bbox.yMin ) bbox.yMin = y;
        if ( y > bbox.yMax ) bbox.yMax = y;
      }

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

#ifdef FT_CONFIG_OPTION_PIC
      FT_Outline_Funcs bbox_interface;
      Init_Class_bbox_interface(&bbox_interface);
#endif

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
