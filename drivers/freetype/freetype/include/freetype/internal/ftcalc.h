/***************************************************************************/
/*                                                                         */
/*  ftcalc.h                                                               */
/*                                                                         */
/*    Arithmetic computations (specification).                             */
/*                                                                         */
/*  Copyright 1996-2006, 2008, 2009, 2012-2013 by                          */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


#ifndef __FTCALC_H__
#define __FTCALC_H__


#include <ft2build.h>
#include FT_FREETYPE_H


FT_BEGIN_HEADER


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    FT_FixedSqrt                                                       */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Computes the square root of a 16.16 fixed-point value.             */
  /*                                                                       */
  /* <Input>                                                               */
  /*    x :: The value to compute the root for.                            */
  /*                                                                       */
  /* <Return>                                                              */
  /*    The result of `sqrt(x)'.                                           */
  /*                                                                       */
  /* <Note>                                                                */
  /*    This function is not very fast.                                    */
  /*                                                                       */
  FT_BASE( FT_Int32 )
  FT_SqrtFixed( FT_Int32  x );


  /*************************************************************************/
  /*                                                                       */
  /* FT_MulDiv() and FT_MulFix() are declared in freetype.h.               */
  /*                                                                       */
  /*************************************************************************/


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    FT_MulDiv_No_Round                                                 */
  /*                                                                       */
  /* <Description>                                                         */
  /*    A very simple function used to perform the computation `(a*b)/c'   */
  /*    (without rounding) with maximum accuracy (it uses a 64-bit         */
  /*    intermediate integer whenever necessary).                          */
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
  FT_BASE( FT_Long )
  FT_MulDiv_No_Round( FT_Long  a,
                      FT_Long  b,
                      FT_Long  c );


  /*
   *  A variant of FT_Matrix_Multiply which scales its result afterwards.
   *  The idea is that both `a' and `b' are scaled by factors of 10 so that
   *  the values are as precise as possible to get a correct result during
   *  the 64bit multiplication.  Let `sa' and `sb' be the scaling factors of
   *  `a' and `b', respectively, then the scaling factor of the result is
   *  `sa*sb'.
   */
  FT_BASE( void )
  FT_Matrix_Multiply_Scaled( const FT_Matrix*  a,
                             FT_Matrix        *b,
                             FT_Long           scaling );


  /*
   *  A variant of FT_Vector_Transform.  See comments for
   *  FT_Matrix_Multiply_Scaled.
   */
  FT_BASE( void )
  FT_Vector_Transform_Scaled( FT_Vector*        vector,
                              const FT_Matrix*  matrix,
                              FT_Long           scaling );


  /*
   *  Return -1, 0, or +1, depending on the orientation of a given corner.
   *  We use the Cartesian coordinate system, with positive vertical values
   *  going upwards.  The function returns +1 if the corner turns to the
   *  left, -1 to the right, and 0 for undecidable cases.
   */
  FT_BASE( FT_Int )
  ft_corner_orientation( FT_Pos  in_x,
                         FT_Pos  in_y,
                         FT_Pos  out_x,
                         FT_Pos  out_y );

  /*
   *  Return TRUE if a corner is flat or nearly flat.  This is equivalent to
   *  saying that the angle difference between the `in' and `out' vectors is
   *  very small.
   */
  FT_BASE( FT_Int )
  ft_corner_is_flat( FT_Pos  in_x,
                     FT_Pos  in_y,
                     FT_Pos  out_x,
                     FT_Pos  out_y );


  /*
   *  Return the most significant bit index.
   */
  FT_BASE( FT_Int )
  FT_MSB( FT_UInt32  z );


  /*
   *  Return sqrt(x*x+y*y), which is the same as `FT_Vector_Length' but uses
   *  two fixed-point arguments instead.
   */
  FT_BASE( FT_Fixed )
  FT_Hypot( FT_Fixed  x,
            FT_Fixed  y );


#define INT_TO_F26DOT6( x )    ( (FT_Long)(x) << 6  )
#define INT_TO_F2DOT14( x )    ( (FT_Long)(x) << 14 )
#define INT_TO_FIXED( x )      ( (FT_Long)(x) << 16 )
#define F2DOT14_TO_FIXED( x )  ( (FT_Long)(x) << 2  )
#define FLOAT_TO_FIXED( x )    ( (FT_Long)( x * 65536.0 ) )
#define FIXED_TO_INT( x )      ( FT_RoundFix( x ) >> 16 )

#define ROUND_F26DOT6( x )     ( x >= 0 ? (    ( (x) + 32 ) & -64 )     \
                                        : ( -( ( 32 - (x) ) & -64 ) ) )


FT_END_HEADER

#endif /* __FTCALC_H__ */


/* END */
