/***************************************************************************/
/*                                                                         */
/*  afangles.c                                                             */
/*                                                                         */
/*    Routines used to compute vector angles with limited accuracy         */
/*    and very high speed.  It also contains sorting routines (body).      */
/*                                                                         */
/*  Copyright 2003-2006, 2011-2012 by                                      */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


#include "aftypes.h"


#if 0

  FT_LOCAL_DEF( FT_Int )
  af_corner_is_flat( FT_Pos  x_in,
                     FT_Pos  y_in,
                     FT_Pos  x_out,
                     FT_Pos  y_out )
  {
    FT_Pos  ax = x_in;
    FT_Pos  ay = y_in;

    FT_Pos  d_in, d_out, d_corner;


    if ( ax < 0 )
      ax = -ax;
    if ( ay < 0 )
      ay = -ay;
    d_in = ax + ay;

    ax = x_out;
    if ( ax < 0 )
      ax = -ax;
    ay = y_out;
    if ( ay < 0 )
      ay = -ay;
    d_out = ax + ay;

    ax = x_out + x_in;
    if ( ax < 0 )
      ax = -ax;
    ay = y_out + y_in;
    if ( ay < 0 )
      ay = -ay;
    d_corner = ax + ay;

    return ( d_in + d_out - d_corner ) < ( d_corner >> 4 );
  }


  FT_LOCAL_DEF( FT_Int )
  af_corner_orientation( FT_Pos  x_in,
                         FT_Pos  y_in,
                         FT_Pos  x_out,
                         FT_Pos  y_out )
  {
    FT_Pos  delta;


    delta = x_in * y_out - y_in * x_out;

    if ( delta == 0 )
      return 0;
    else
      return 1 - 2 * ( delta < 0 );
  }

#endif /* 0 */


  /*
   *  We are not using `af_angle_atan' anymore, but we keep the source
   *  code below just in case...
   */


#if 0


  /*
   *  The trick here is to realize that we don't need a very accurate angle
   *  approximation.  We are going to use the result of `af_angle_atan' to
   *  only compare the sign of angle differences, or check whether its
   *  magnitude is very small.
   *
   *  The approximation
   *
   *    dy * PI / (|dx|+|dy|)
   *
   *  should be enough, and much faster to compute.
   */
  FT_LOCAL_DEF( AF_Angle )
  af_angle_atan( FT_Fixed  dx,
                 FT_Fixed  dy )
  {
    AF_Angle  angle;
    FT_Fixed  ax = dx;
    FT_Fixed  ay = dy;


    if ( ax < 0 )
      ax = -ax;
    if ( ay < 0 )
      ay = -ay;

    ax += ay;

    if ( ax == 0 )
      angle = 0;
    else
    {
      angle = ( AF_ANGLE_PI2 * dy ) / ( ax + ay );
      if ( dx < 0 )
      {
        if ( angle >= 0 )
          angle = AF_ANGLE_PI - angle;
        else
          angle = -AF_ANGLE_PI - angle;
      }
    }

    return angle;
  }


#elif 0


  /* the following table has been automatically generated with */
  /* the `mather.py' Python script                             */

#define AF_ATAN_BITS  8

  static const FT_Byte  af_arctan[1L << AF_ATAN_BITS] =
  {
     0,  0,  1,  1,  1,  2,  2,  2,
     3,  3,  3,  3,  4,  4,  4,  5,
     5,  5,  6,  6,  6,  7,  7,  7,
     8,  8,  8,  9,  9,  9, 10, 10,
    10, 10, 11, 11, 11, 12, 12, 12,
    13, 13, 13, 14, 14, 14, 14, 15,
    15, 15, 16, 16, 16, 17, 17, 17,
    18, 18, 18, 18, 19, 19, 19, 20,
    20, 20, 21, 21, 21, 21, 22, 22,
    22, 23, 23, 23, 24, 24, 24, 24,
    25, 25, 25, 26, 26, 26, 26, 27,
    27, 27, 28, 28, 28, 28, 29, 29,
    29, 30, 30, 30, 30, 31, 31, 31,
    31, 32, 32, 32, 33, 33, 33, 33,
    34, 34, 34, 34, 35, 35, 35, 35,
    36, 36, 36, 36, 37, 37, 37, 38,
    38, 38, 38, 39, 39, 39, 39, 40,
    40, 40, 40, 41, 41, 41, 41, 42,
    42, 42, 42, 42, 43, 43, 43, 43,
    44, 44, 44, 44, 45, 45, 45, 45,
    46, 46, 46, 46, 46, 47, 47, 47,
    47, 48, 48, 48, 48, 48, 49, 49,
    49, 49, 50, 50, 50, 50, 50, 51,
    51, 51, 51, 51, 52, 52, 52, 52,
    52, 53, 53, 53, 53, 53, 54, 54,
    54, 54, 54, 55, 55, 55, 55, 55,
    56, 56, 56, 56, 56, 57, 57, 57,
    57, 57, 57, 58, 58, 58, 58, 58,
    59, 59, 59, 59, 59, 59, 60, 60,
    60, 60, 60, 61, 61, 61, 61, 61,
    61, 62, 62, 62, 62, 62, 62, 63,
    63, 63, 63, 63, 63, 64, 64, 64
  };


  FT_LOCAL_DEF( AF_Angle )
  af_angle_atan( FT_Fixed  dx,
                 FT_Fixed  dy )
  {
    AF_Angle  angle;


    /* check trivial cases */
    if ( dy == 0 )
    {
      angle = 0;
      if ( dx < 0 )
        angle = AF_ANGLE_PI;
      return angle;
    }
    else if ( dx == 0 )
    {
      angle = AF_ANGLE_PI2;
      if ( dy < 0 )
        angle = -AF_ANGLE_PI2;
      return angle;
    }

    angle = 0;
    if ( dx < 0 )
    {
      dx = -dx;
      dy = -dy;
      angle = AF_ANGLE_PI;
    }

    if ( dy < 0 )
    {
      FT_Pos  tmp;


      tmp = dx;
      dx  = -dy;
      dy  = tmp;
      angle -= AF_ANGLE_PI2;
    }

    if ( dx == 0 && dy == 0 )
      return 0;

    if ( dx == dy )
      angle += AF_ANGLE_PI4;
    else if ( dx > dy )
      angle += af_arctan[FT_DivFix( dy, dx ) >> ( 16 - AF_ATAN_BITS )];
    else
      angle += AF_ANGLE_PI2 -
               af_arctan[FT_DivFix( dx, dy ) >> ( 16 - AF_ATAN_BITS )];

    if ( angle > AF_ANGLE_PI )
      angle -= AF_ANGLE_2PI;

    return angle;
  }


#endif /* 0 */


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
        table[cur_idx].org = sum / j;

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


/* END */
