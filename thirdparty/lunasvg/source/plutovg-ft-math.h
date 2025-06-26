/***************************************************************************/
/*                                                                         */
/*  fttrigon.h                                                             */
/*                                                                         */
/*    FreeType trigonometric functions (specification).                    */
/*                                                                         */
/*  Copyright 2001, 2003, 2005, 2007, 2013 by                              */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, FTL.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/

#ifndef PLUTOVG_FT_MATH_H
#define PLUTOVG_FT_MATH_H

#include "plutovg-ft-types.h"

/*************************************************************************/
/*                                                                       */
/* The min and max functions missing in C.  As usual, be careful not to  */
/* write things like PVG_FT_MIN( a++, b++ ) to avoid side effects.           */
/*                                                                       */
#define PVG_FT_MIN( a, b )  ( (a) < (b) ? (a) : (b) )
#define PVG_FT_MAX( a, b )  ( (a) > (b) ? (a) : (b) )

#define PVG_FT_ABS( a )     ( (a) < 0 ? -(a) : (a) )

/*
 * Approximate sqrt(x*x+y*y) using the `alpha max plus beta min'
 * algorithm.  We use alpha = 1, beta = 3/8, giving us results with a
 * largest error less than 7% compared to the exact value.
 */
#define PVG_FT_HYPOT( x, y )                 \
          ( x = PVG_FT_ABS( x ),             \
            y = PVG_FT_ABS( y ),             \
            x > y ? x + ( 3 * y >> 3 )   \
                  : y + ( 3 * x >> 3 ) )

/*************************************************************************/
/*                                                                       */
/* <Function>                                                            */
/*    PVG_FT_MulFix                                                      */
/*                                                                       */
/* <Description>                                                         */
/*    A very simple function used to perform the computation             */
/*    `(a*b)/0x10000' with maximum accuracy.  Most of the time this is   */
/*    used to multiply a given value by a 16.16 fixed-point factor.      */
/*                                                                       */
/* <Input>                                                               */
/*    a :: The first multiplier.                                         */
/*    b :: The second multiplier.  Use a 16.16 factor here whenever      */
/*         possible (see note below).                                    */
/*                                                                       */
/* <Return>                                                              */
/*    The result of `(a*b)/0x10000'.                                     */
/*                                                                       */
/* <Note>                                                                */
/*    This function has been optimized for the case where the absolute   */
/*    value of `a' is less than 2048, and `b' is a 16.16 scaling factor. */
/*    As this happens mainly when scaling from notional units to         */
/*    fractional pixels in FreeType, it resulted in noticeable speed     */
/*    improvements between versions 2.x and 1.x.                         */
/*                                                                       */
/*    As a conclusion, always try to place a 16.16 factor as the         */
/*    _second_ argument of this function; this can make a great          */
/*    difference.                                                        */
/*                                                                       */
PVG_FT_Long
PVG_FT_MulFix( PVG_FT_Long  a,
    PVG_FT_Long  b );

/*************************************************************************/
/*                                                                       */
/* <Function>                                                            */
/*    PVG_FT_MulDiv                                                      */
/*                                                                       */
/* <Description>                                                         */
/*    A very simple function used to perform the computation `(a*b)/c'   */
/*    with maximum accuracy (it uses a 64-bit intermediate integer       */
/*    whenever necessary).                                               */
/*                                                                       */
/*    This function isn't necessarily as fast as some processor specific */
/*    operations, but is at least completely portable.                   */
/*                                                                       */
/* <Input>                                                               */
/*    a :: The first multiplier.                                         */
/*    b :: The second multiplier.                                        */
/*    c :: The divisor.                                                  */
/*                                                                       */
/* <Return>                                                              */
/*    The result of `(a*b)/c'.  This function never traps when trying to */
/*    divide by zero; it simply returns `MaxInt' or `MinInt' depending   */
/*    on the signs of `a' and `b'.                                       */
/*                                                                       */
PVG_FT_Long
PVG_FT_MulDiv( PVG_FT_Long  a,
    PVG_FT_Long  b,
    PVG_FT_Long  c );

/*************************************************************************/
/*                                                                       */
/* <Function>                                                            */
/*    PVG_FT_DivFix                                                      */
/*                                                                       */
/* <Description>                                                         */
/*    A very simple function used to perform the computation             */
/*    `(a*0x10000)/b' with maximum accuracy.  Most of the time, this is  */
/*    used to divide a given value by a 16.16 fixed-point factor.        */
/*                                                                       */
/* <Input>                                                               */
/*    a :: The numerator.                                                */
/*    b :: The denominator.  Use a 16.16 factor here.                    */
/*                                                                       */
/* <Return>                                                              */
/*    The result of `(a*0x10000)/b'.                                     */
/*                                                                       */
PVG_FT_Long
PVG_FT_DivFix( PVG_FT_Long  a,
    PVG_FT_Long  b );



/*************************************************************************/
/*                                                                       */
/* <Section>                                                             */
/*   computations                                                        */
/*                                                                       */
/*************************************************************************/


/*************************************************************************
 *
 * @type:
 *   PVG_FT_Angle
 *
 * @description:
 *   This type is used to model angle values in FreeType.  Note that the
 *   angle is a 16.16 fixed-point value expressed in degrees.
 *
 */
typedef PVG_FT_Fixed  PVG_FT_Angle;


/*************************************************************************
 *
 * @macro:
 *   PVG_FT_ANGLE_PI
 *
 * @description:
 *   The angle pi expressed in @PVG_FT_Angle units.
 *
 */
#define PVG_FT_ANGLE_PI  ( 180L << 16 )


/*************************************************************************
 *
 * @macro:
 *   PVG_FT_ANGLE_2PI
 *
 * @description:
 *   The angle 2*pi expressed in @PVG_FT_Angle units.
 *
 */
#define PVG_FT_ANGLE_2PI  ( PVG_FT_ANGLE_PI * 2 )


/*************************************************************************
 *
 * @macro:
 *   PVG_FT_ANGLE_PI2
 *
 * @description:
 *   The angle pi/2 expressed in @PVG_FT_Angle units.
 *
 */
#define PVG_FT_ANGLE_PI2  ( PVG_FT_ANGLE_PI / 2 )


/*************************************************************************
 *
 * @macro:
 *   PVG_FT_ANGLE_PI4
 *
 * @description:
 *   The angle pi/4 expressed in @PVG_FT_Angle units.
 *
 */
#define PVG_FT_ANGLE_PI4  ( PVG_FT_ANGLE_PI / 4 )


/*************************************************************************
 *
 * @function:
 *   PVG_FT_Sin
 *
 * @description:
 *   Return the sinus of a given angle in fixed-point format.
 *
 * @input:
 *   angle ::
 *     The input angle.
 *
 * @return:
 *   The sinus value.
 *
 * @note:
 *   If you need both the sinus and cosinus for a given angle, use the
 *   function @PVG_FT_Vector_Unit.
 *
 */
PVG_FT_Fixed
PVG_FT_Sin( PVG_FT_Angle  angle );


/*************************************************************************
 *
 * @function:
 *   PVG_FT_Cos
 *
 * @description:
 *   Return the cosinus of a given angle in fixed-point format.
 *
 * @input:
 *   angle ::
 *     The input angle.
 *
 * @return:
 *   The cosinus value.
 *
 * @note:
 *   If you need both the sinus and cosinus for a given angle, use the
 *   function @PVG_FT_Vector_Unit.
 *
 */
PVG_FT_Fixed
PVG_FT_Cos( PVG_FT_Angle  angle );


/*************************************************************************
 *
 * @function:
 *   PVG_FT_Tan
 *
 * @description:
 *   Return the tangent of a given angle in fixed-point format.
 *
 * @input:
 *   angle ::
 *     The input angle.
 *
 * @return:
 *   The tangent value.
 *
 */
PVG_FT_Fixed
PVG_FT_Tan( PVG_FT_Angle  angle );


/*************************************************************************
 *
 * @function:
 *   PVG_FT_Atan2
 *
 * @description:
 *   Return the arc-tangent corresponding to a given vector (x,y) in
 *   the 2d plane.
 *
 * @input:
 *   x ::
 *     The horizontal vector coordinate.
 *
 *   y ::
 *     The vertical vector coordinate.
 *
 * @return:
 *   The arc-tangent value (i.e. angle).
 *
 */
PVG_FT_Angle
PVG_FT_Atan2( PVG_FT_Fixed  x,
    PVG_FT_Fixed  y );


/*************************************************************************
 *
 * @function:
 *   PVG_FT_Angle_Diff
 *
 * @description:
 *   Return the difference between two angles.  The result is always
 *   constrained to the ]-PI..PI] interval.
 *
 * @input:
 *   angle1 ::
 *     First angle.
 *
 *   angle2 ::
 *     Second angle.
 *
 * @return:
 *   Constrained value of `value2-value1'.
 *
 */
PVG_FT_Angle
PVG_FT_Angle_Diff( PVG_FT_Angle  angle1,
    PVG_FT_Angle  angle2 );


/*************************************************************************
 *
 * @function:
 *   PVG_FT_Vector_Unit
 *
 * @description:
 *   Return the unit vector corresponding to a given angle.  After the
 *   call, the value of `vec.x' will be `sin(angle)', and the value of
 *   `vec.y' will be `cos(angle)'.
 *
 *   This function is useful to retrieve both the sinus and cosinus of a
 *   given angle quickly.
 *
 * @output:
 *   vec ::
 *     The address of target vector.
 *
 * @input:
 *   angle ::
 *     The input angle.
 *
 */
void
PVG_FT_Vector_Unit( PVG_FT_Vector*  vec,
    PVG_FT_Angle    angle );


/*************************************************************************
 *
 * @function:
 *   PVG_FT_Vector_Rotate
 *
 * @description:
 *   Rotate a vector by a given angle.
 *
 * @inout:
 *   vec ::
 *     The address of target vector.
 *
 * @input:
 *   angle ::
 *     The input angle.
 *
 */
void
PVG_FT_Vector_Rotate( PVG_FT_Vector*  vec,
    PVG_FT_Angle    angle );


/*************************************************************************
 *
 * @function:
 *   PVG_FT_Vector_Length
 *
 * @description:
 *   Return the length of a given vector.
 *
 * @input:
 *   vec ::
 *     The address of target vector.
 *
 * @return:
 *   The vector length, expressed in the same units that the original
 *   vector coordinates.
 *
 */
PVG_FT_Fixed
PVG_FT_Vector_Length( PVG_FT_Vector*  vec );


/*************************************************************************
 *
 * @function:
 *   PVG_FT_Vector_Polarize
 *
 * @description:
 *   Compute both the length and angle of a given vector.
 *
 * @input:
 *   vec ::
 *     The address of source vector.
 *
 * @output:
 *   length ::
 *     The vector length.
 *
 *   angle ::
 *     The vector angle.
 *
 */
void
PVG_FT_Vector_Polarize( PVG_FT_Vector*  vec,
    PVG_FT_Fixed   *length,
    PVG_FT_Angle   *angle );


/*************************************************************************
 *
 * @function:
 *   PVG_FT_Vector_From_Polar
 *
 * @description:
 *   Compute vector coordinates from a length and angle.
 *
 * @output:
 *   vec ::
 *     The address of source vector.
 *
 * @input:
 *   length ::
 *     The vector length.
 *
 *   angle ::
 *     The vector angle.
 *
 */
void
PVG_FT_Vector_From_Polar( PVG_FT_Vector*  vec,
    PVG_FT_Fixed    length,
    PVG_FT_Angle    angle );

#endif /* PLUTOVG_FT_MATH_H */
