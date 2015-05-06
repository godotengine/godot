/***************************************************************************/
/*                                                                         */
/*  fttrigon.c                                                             */
/*                                                                         */
/*    FreeType trigonometric functions (body).                             */
/*                                                                         */
/*  Copyright 2001-2005, 2012-2013 by                                      */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/

  /*************************************************************************/
  /*                                                                       */
  /* This is a fixed-point CORDIC implementation of trigonometric          */
  /* functions as well as transformations between Cartesian and polar      */
  /* coordinates.  The angles are represented as 16.16 fixed-point values  */
  /* in degrees, i.e., the angular resolution is 2^-16 degrees.  Note that */
  /* only vectors longer than 2^16*180/pi (or at least 22 bits) on a       */
  /* discrete Cartesian grid can have the same or better angular           */
  /* resolution.  Therefore, to maintain this precision, some functions    */
  /* require an interim upscaling of the vectors, whereas others operate   */
  /* with 24-bit long vectors directly.                                    */
  /*                                                                       */
  /*************************************************************************/

#include <ft2build.h>
#include FT_INTERNAL_OBJECTS_H
#include FT_INTERNAL_CALC_H
#include FT_TRIGONOMETRY_H


  /* the Cordic shrink factor 0.858785336480436 * 2^32 */
#define FT_TRIG_SCALE      0xDBD95B16UL

  /* the highest bit in overflow-safe vector components, */
  /* MSB of 0.858785336480436 * sqrt(0.5) * 2^30         */
#define FT_TRIG_SAFE_MSB   29

  /* this table was generated for FT_PI = 180L << 16, i.e. degrees */
#define FT_TRIG_MAX_ITERS  23

  static const FT_Fixed
  ft_trig_arctan_table[] =
  {
    1740967L, 919879L, 466945L, 234379L, 117304L, 58666L, 29335L,
    14668L, 7334L, 3667L, 1833L, 917L, 458L, 229L, 115L,
    57L, 29L, 14L, 7L, 4L, 2L, 1L
  };


#ifdef FT_LONG64

  /* multiply a given value by the CORDIC shrink factor */
  static FT_Fixed
  ft_trig_downscale( FT_Fixed  val )
  {
    FT_Fixed  s;
    FT_Int64  v;


    s   = val;
    val = FT_ABS( val );

    v   = ( val * (FT_Int64)FT_TRIG_SCALE ) + 0x100000000UL;
    val = (FT_Fixed)( v >> 32 );

    return ( s >= 0 ) ? val : -val;
  }

#else /* !FT_LONG64 */

  /* multiply a given value by the CORDIC shrink factor */
  static FT_Fixed
  ft_trig_downscale( FT_Fixed  val )
  {
    FT_Fixed   s;
    FT_UInt32  v1, v2, k1, k2, hi, lo1, lo2, lo3;


    s   = val;
    val = FT_ABS( val );

    v1 = (FT_UInt32)val >> 16;
    v2 = (FT_UInt32)( val & 0xFFFFL );

    k1 = (FT_UInt32)FT_TRIG_SCALE >> 16;           /* constant */
    k2 = (FT_UInt32)( FT_TRIG_SCALE & 0xFFFFL );   /* constant */

    hi   = k1 * v1;
    lo1  = k1 * v2 + k2 * v1;       /* can't overflow */

    lo2  = ( k2 * v2 ) >> 16;
    lo3  = FT_MAX( lo1, lo2 );
    lo1 += lo2;

    hi  += lo1 >> 16;
    if ( lo1 < lo3 )
      hi += (FT_UInt32)0x10000UL;

    val  = (FT_Fixed)hi;

    return ( s >= 0 ) ? val : -val;
  }

#endif /* !FT_LONG64 */


  static FT_Int
  ft_trig_prenorm( FT_Vector*  vec )
  {
    FT_Pos  x, y;
    FT_Int  shift;


    x = vec->x;
    y = vec->y;

    shift = FT_MSB( FT_ABS( x ) | FT_ABS( y ) );

    if ( shift <= FT_TRIG_SAFE_MSB )
    {
      shift  = FT_TRIG_SAFE_MSB - shift;
      vec->x = (FT_Pos)( (FT_ULong)x << shift );
      vec->y = (FT_Pos)( (FT_ULong)y << shift );
    }
    else
    {
      shift -= FT_TRIG_SAFE_MSB;
      vec->x = x >> shift;
      vec->y = y >> shift;
      shift  = -shift;
    }

    return shift;
  }


  static void
  ft_trig_pseudo_rotate( FT_Vector*  vec,
                         FT_Angle    theta )
  {
    FT_Int           i;
    FT_Fixed         x, y, xtemp, b;
    const FT_Fixed  *arctanptr;


    x = vec->x;
    y = vec->y;

    /* Rotate inside [-PI/4,PI/4] sector */
    while ( theta < -FT_ANGLE_PI4 )
    {
      xtemp  =  y;
      y      = -x;
      x      =  xtemp;
      theta +=  FT_ANGLE_PI2;
    }

    while ( theta > FT_ANGLE_PI4 )
    {
      xtemp  = -y;
      y      =  x;
      x      =  xtemp;
      theta -=  FT_ANGLE_PI2;
    }

    arctanptr = ft_trig_arctan_table;

    /* Pseudorotations, with right shifts */
    for ( i = 1, b = 1; i < FT_TRIG_MAX_ITERS; b <<= 1, i++ )
    {
      if ( theta < 0 )
      {
        xtemp  = x + ( ( y + b ) >> i );
        y      = y - ( ( x + b ) >> i );
        x      = xtemp;
        theta += *arctanptr++;
      }
      else
      {
        xtemp  = x - ( ( y + b ) >> i );
        y      = y + ( ( x + b ) >> i );
        x      = xtemp;
        theta -= *arctanptr++;
      }
    }

    vec->x = x;
    vec->y = y;
  }


  static void
  ft_trig_pseudo_polarize( FT_Vector*  vec )
  {
    FT_Angle         theta;
    FT_Int           i;
    FT_Fixed         x, y, xtemp, b;
    const FT_Fixed  *arctanptr;


    x = vec->x;
    y = vec->y;

    /* Get the vector into [-PI/4,PI/4] sector */
    if ( y > x )
    {
      if ( y > -x )
      {
        theta =  FT_ANGLE_PI2;
        xtemp =  y;
        y     = -x;
        x     =  xtemp;
      }
      else
      {
        theta =  y > 0 ? FT_ANGLE_PI : -FT_ANGLE_PI;
        x     = -x;
        y     = -y;
      }
    }
    else
    {
      if ( y < -x )
      {
        theta = -FT_ANGLE_PI2;
        xtemp = -y;
        y     =  x;
        x     =  xtemp;
      }
      else
      {
        theta = 0;
      }
    }

    arctanptr = ft_trig_arctan_table;

    /* Pseudorotations, with right shifts */
    for ( i = 1, b = 1; i < FT_TRIG_MAX_ITERS; b <<= 1, i++ )
    {
      if ( y > 0 )
      {
        xtemp  = x + ( ( y + b ) >> i );
        y      = y - ( ( x + b ) >> i );
        x      = xtemp;
        theta += *arctanptr++;
      }
      else
      {
        xtemp  = x - ( ( y + b ) >> i );
        y      = y + ( ( x + b ) >> i );
        x      = xtemp;
        theta -= *arctanptr++;
      }
    }

    /* round theta */
    if ( theta >= 0 )
      theta = FT_PAD_ROUND( theta, 32 );
    else
      theta = -FT_PAD_ROUND( -theta, 32 );

    vec->x = x;
    vec->y = theta;
  }


  /* documentation is in fttrigon.h */

  FT_EXPORT_DEF( FT_Fixed )
  FT_Cos( FT_Angle  angle )
  {
    FT_Vector  v;


    v.x = FT_TRIG_SCALE >> 8;
    v.y = 0;
    ft_trig_pseudo_rotate( &v, angle );

    return ( v.x + 0x80L ) >> 8;
  }


  /* documentation is in fttrigon.h */

  FT_EXPORT_DEF( FT_Fixed )
  FT_Sin( FT_Angle  angle )
  {
    return FT_Cos( FT_ANGLE_PI2 - angle );
  }


  /* documentation is in fttrigon.h */

  FT_EXPORT_DEF( FT_Fixed )
  FT_Tan( FT_Angle  angle )
  {
    FT_Vector  v;


    v.x = FT_TRIG_SCALE >> 8;
    v.y = 0;
    ft_trig_pseudo_rotate( &v, angle );

    return FT_DivFix( v.y, v.x );
  }


  /* documentation is in fttrigon.h */

  FT_EXPORT_DEF( FT_Angle )
  FT_Atan2( FT_Fixed  dx,
            FT_Fixed  dy )
  {
    FT_Vector  v;


    if ( dx == 0 && dy == 0 )
      return 0;

    v.x = dx;
    v.y = dy;
    ft_trig_prenorm( &v );
    ft_trig_pseudo_polarize( &v );

    return v.y;
  }


  /* documentation is in fttrigon.h */

  FT_EXPORT_DEF( void )
  FT_Vector_Unit( FT_Vector*  vec,
                  FT_Angle    angle )
  {
    vec->x = FT_TRIG_SCALE >> 8;
    vec->y = 0;
    ft_trig_pseudo_rotate( vec, angle );
    vec->x = ( vec->x + 0x80L ) >> 8;
    vec->y = ( vec->y + 0x80L ) >> 8;
  }


  /* these macros return 0 for positive numbers,
     and -1 for negative ones */
#define FT_SIGN_LONG( x )   ( (x) >> ( FT_SIZEOF_LONG * 8 - 1 ) )
#define FT_SIGN_INT( x )    ( (x) >> ( FT_SIZEOF_INT * 8 - 1 ) )
#define FT_SIGN_INT32( x )  ( (x) >> 31 )
#define FT_SIGN_INT16( x )  ( (x) >> 15 )


  /* documentation is in fttrigon.h */

  FT_EXPORT_DEF( void )
  FT_Vector_Rotate( FT_Vector*  vec,
                    FT_Angle    angle )
  {
    FT_Int     shift;
    FT_Vector  v;


    v.x   = vec->x;
    v.y   = vec->y;

    if ( angle && ( v.x != 0 || v.y != 0 ) )
    {
      shift = ft_trig_prenorm( &v );
      ft_trig_pseudo_rotate( &v, angle );
      v.x = ft_trig_downscale( v.x );
      v.y = ft_trig_downscale( v.y );

      if ( shift > 0 )
      {
        FT_Int32  half = (FT_Int32)1L << ( shift - 1 );


        vec->x = ( v.x + half + FT_SIGN_LONG( v.x ) ) >> shift;
        vec->y = ( v.y + half + FT_SIGN_LONG( v.y ) ) >> shift;
      }
      else
      {
        shift  = -shift;
        vec->x = (FT_Pos)( (FT_ULong)v.x << shift );
        vec->y = (FT_Pos)( (FT_ULong)v.y << shift );
      }
    }
  }


  /* documentation is in fttrigon.h */

  FT_EXPORT_DEF( FT_Fixed )
  FT_Vector_Length( FT_Vector*  vec )
  {
    FT_Int     shift;
    FT_Vector  v;


    v = *vec;

    /* handle trivial cases */
    if ( v.x == 0 )
    {
      return FT_ABS( v.y );
    }
    else if ( v.y == 0 )
    {
      return FT_ABS( v.x );
    }

    /* general case */
    shift = ft_trig_prenorm( &v );
    ft_trig_pseudo_polarize( &v );

    v.x = ft_trig_downscale( v.x );

    if ( shift > 0 )
      return ( v.x + ( 1 << ( shift - 1 ) ) ) >> shift;

    return (FT_Fixed)( (FT_UInt32)v.x << -shift );
  }


  /* documentation is in fttrigon.h */

  FT_EXPORT_DEF( void )
  FT_Vector_Polarize( FT_Vector*  vec,
                      FT_Fixed   *length,
                      FT_Angle   *angle )
  {
    FT_Int     shift;
    FT_Vector  v;


    v = *vec;

    if ( v.x == 0 && v.y == 0 )
      return;

    shift = ft_trig_prenorm( &v );
    ft_trig_pseudo_polarize( &v );

    v.x = ft_trig_downscale( v.x );

    *length = ( shift >= 0 ) ?                      ( v.x >>  shift )
                             : (FT_Fixed)( (FT_UInt32)v.x << -shift );
    *angle  = v.y;
  }


  /* documentation is in fttrigon.h */

  FT_EXPORT_DEF( void )
  FT_Vector_From_Polar( FT_Vector*  vec,
                        FT_Fixed    length,
                        FT_Angle    angle )
  {
    vec->x = length;
    vec->y = 0;

    FT_Vector_Rotate( vec, angle );
  }


  /* documentation is in fttrigon.h */

  FT_EXPORT_DEF( FT_Angle )
  FT_Angle_Diff( FT_Angle  angle1,
                 FT_Angle  angle2 )
  {
    FT_Angle  delta = angle2 - angle1;


    delta %= FT_ANGLE_2PI;
    if ( delta < 0 )
      delta += FT_ANGLE_2PI;

    if ( delta > FT_ANGLE_PI )
      delta -= FT_ANGLE_2PI;

    return delta;
  }


/* END */
