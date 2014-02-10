/***************************************************************************/
/*                                                                         */
/*  ftcalc.c                                                               */
/*                                                                         */
/*    Arithmetic computations (body).                                      */
/*                                                                         */
/*  Copyright 1996-2006, 2008, 2012-2013 by                                */
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
  /* Support for 1-complement arithmetic has been totally dropped in this  */
  /* release.  You can still write your own code if you need it.           */
  /*                                                                       */
  /*************************************************************************/

  /*************************************************************************/
  /*                                                                       */
  /* Implementing basic computation routines.                              */
  /*                                                                       */
  /* FT_MulDiv(), FT_MulFix(), FT_DivFix(), FT_RoundFix(), FT_CeilFix(),   */
  /* and FT_FloorFix() are declared in freetype.h.                         */
  /*                                                                       */
  /*************************************************************************/


#include <ft2build.h>
#include FT_GLYPH_H
#include FT_TRIGONOMETRY_H
#include FT_INTERNAL_CALC_H
#include FT_INTERNAL_DEBUG_H
#include FT_INTERNAL_OBJECTS_H

#ifdef FT_MULFIX_INLINED
#undef FT_MulFix
#endif

/* we need to emulate a 64-bit data type if a real one isn't available */

#ifndef FT_LONG64

  typedef struct  FT_Int64_
  {
    FT_UInt32  lo;
    FT_UInt32  hi;

  } FT_Int64;

#endif /* !FT_LONG64 */


  /*************************************************************************/
  /*                                                                       */
  /* The macro FT_COMPONENT is used in trace mode.  It is an implicit      */
  /* parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log  */
  /* messages during execution.                                            */
  /*                                                                       */
#undef  FT_COMPONENT
#define FT_COMPONENT  trace_calc


  /* The following three functions are available regardless of whether */
  /* FT_LONG64 is defined.                                             */

  /* documentation is in freetype.h */

  FT_EXPORT_DEF( FT_Fixed )
  FT_RoundFix( FT_Fixed  a )
  {
    return ( a >= 0 ) ?   ( a + 0x8000L ) & ~0xFFFFL
                      : -((-a + 0x8000L ) & ~0xFFFFL );
  }


  /* documentation is in freetype.h */

  FT_EXPORT_DEF( FT_Fixed )
  FT_CeilFix( FT_Fixed  a )
  {
    return ( a >= 0 ) ?   ( a + 0xFFFFL ) & ~0xFFFFL
                      : -((-a + 0xFFFFL ) & ~0xFFFFL );
  }


  /* documentation is in freetype.h */

  FT_EXPORT_DEF( FT_Fixed )
  FT_FloorFix( FT_Fixed  a )
  {
    return ( a >= 0 ) ?   a & ~0xFFFFL
                      : -((-a) & ~0xFFFFL );
  }


  FT_BASE_DEF ( FT_Int )
  FT_MSB( FT_UInt32 z )
  {
    FT_Int shift = 0;

    /* determine msb bit index in `shift' */
    if ( z >= ( 1L << 16 ) )
    {
      z     >>= 16;
      shift  += 16;
    }
    if ( z >= ( 1L << 8 ) )
    {
      z     >>= 8;
      shift  += 8;
    }
    if ( z >= ( 1L << 4 ) )
    {
      z     >>= 4;
      shift  += 4;
    }
    if ( z >= ( 1L << 2 ) )
    {
      z     >>= 2;
      shift  += 2;
    }
    if ( z >= ( 1L << 1 ) )
    {
      z     >>= 1;
      shift  += 1;
    }

    return shift;
  }


  /* documentation is in ftcalc.h */

  FT_BASE_DEF( FT_Fixed )
  FT_Hypot( FT_Fixed  x,
            FT_Fixed  y )
  {
    FT_Vector  v;


    v.x = x;
    v.y = y;

    return FT_Vector_Length( &v );
  }


#ifdef FT_LONG64


  /* documentation is in freetype.h */

  FT_EXPORT_DEF( FT_Long )
  FT_MulDiv( FT_Long  a,
             FT_Long  b,
             FT_Long  c )
  {
    FT_Int   s;
    FT_Long  d;


    s = 1;
    if ( a < 0 ) { a = -a; s = -1; }
    if ( b < 0 ) { b = -b; s = -s; }
    if ( c < 0 ) { c = -c; s = -s; }

    d = (FT_Long)( c > 0 ? ( (FT_Int64)a * b + ( c >> 1 ) ) / c
                         : 0x7FFFFFFFL );

    return ( s > 0 ) ? d : -d;
  }


  /* documentation is in ftcalc.h */

  FT_BASE_DEF( FT_Long )
  FT_MulDiv_No_Round( FT_Long  a,
                      FT_Long  b,
                      FT_Long  c )
  {
    FT_Int   s;
    FT_Long  d;


    s = 1;
    if ( a < 0 ) { a = -a; s = -1; }
    if ( b < 0 ) { b = -b; s = -s; }
    if ( c < 0 ) { c = -c; s = -s; }

    d = (FT_Long)( c > 0 ? (FT_Int64)a * b / c
                         : 0x7FFFFFFFL );

    return ( s > 0 ) ? d : -d;
  }


  /* documentation is in freetype.h */

  FT_EXPORT_DEF( FT_Long )
  FT_MulFix( FT_Long  a,
             FT_Long  b )
  {
#ifdef FT_MULFIX_ASSEMBLER

    return FT_MULFIX_ASSEMBLER( a, b );

#else

    FT_Int   s = 1;
    FT_Long  c;


    if ( a < 0 )
    {
      a = -a;
      s = -1;
    }

    if ( b < 0 )
    {
      b = -b;
      s = -s;
    }

    c = (FT_Long)( ( (FT_Int64)a * b + 0x8000L ) >> 16 );

    return ( s > 0 ) ? c : -c;

#endif /* FT_MULFIX_ASSEMBLER */
  }


  /* documentation is in freetype.h */

  FT_EXPORT_DEF( FT_Long )
  FT_DivFix( FT_Long  a,
             FT_Long  b )
  {
    FT_Int32   s;
    FT_UInt32  q;


    s = 1;
    if ( a < 0 )
    {
      a = -a;
      s = -1;
    }
    if ( b < 0 )
    {
      b = -b;
      s = -s;
    }

    if ( b == 0 )
      /* check for division by 0 */
      q = 0x7FFFFFFFL;
    else
      /* compute result directly */
      q = (FT_UInt32)( ( ( (FT_UInt64)a << 16 ) + ( b >> 1 ) ) / b );

    return ( s < 0 ? -(FT_Long)q : (FT_Long)q );
  }


#else /* !FT_LONG64 */


  static void
  ft_multo64( FT_UInt32  x,
              FT_UInt32  y,
              FT_Int64  *z )
  {
    FT_UInt32  lo1, hi1, lo2, hi2, lo, hi, i1, i2;


    lo1 = x & 0x0000FFFFU;  hi1 = x >> 16;
    lo2 = y & 0x0000FFFFU;  hi2 = y >> 16;

    lo = lo1 * lo2;
    i1 = lo1 * hi2;
    i2 = lo2 * hi1;
    hi = hi1 * hi2;

    /* Check carry overflow of i1 + i2 */
    i1 += i2;
    hi += (FT_UInt32)( i1 < i2 ) << 16;

    hi += i1 >> 16;
    i1  = i1 << 16;

    /* Check carry overflow of i1 + lo */
    lo += i1;
    hi += ( lo < i1 );

    z->lo = lo;
    z->hi = hi;
  }


  static FT_UInt32
  ft_div64by32( FT_UInt32  hi,
                FT_UInt32  lo,
                FT_UInt32  y )
  {
    FT_UInt32  r, q;
    FT_Int     i;


    q = 0;
    r = hi;

    if ( r >= y )
      return (FT_UInt32)0x7FFFFFFFL;

    i = 32;
    do
    {
      r <<= 1;
      q <<= 1;
      r  |= lo >> 31;

      if ( r >= y )
      {
        r -= y;
        q |= 1;
      }
      lo <<= 1;
    } while ( --i );

    return q;
  }


  static void
  FT_Add64( FT_Int64*  x,
            FT_Int64*  y,
            FT_Int64  *z )
  {
    register FT_UInt32  lo, hi;


    lo = x->lo + y->lo;
    hi = x->hi + y->hi + ( lo < x->lo );

    z->lo = lo;
    z->hi = hi;
  }


  /* documentation is in freetype.h */

  /* The FT_MulDiv function has been optimized thanks to ideas from      */
  /* Graham Asher.  The trick is to optimize computation when everything */
  /* fits within 32-bits (a rather common case).                         */
  /*                                                                     */
  /*  we compute 'a*b+c/2', then divide it by 'c'. (positive values)     */
  /*                                                                     */
  /*  46340 is FLOOR(SQRT(2^31-1)).                                      */
  /*                                                                     */
  /*  if ( a <= 46340 && b <= 46340 ) then ( a*b <= 0x7FFEA810 )         */
  /*                                                                     */
  /*  0x7FFFFFFF - 0x7FFEA810 = 0x157F0                                  */
  /*                                                                     */
  /*  if ( c < 0x157F0*2 ) then ( a*b+c/2 <= 0x7FFFFFFF )                */
  /*                                                                     */
  /*  and 2*0x157F0 = 176096                                             */
  /*                                                                     */

  FT_EXPORT_DEF( FT_Long )
  FT_MulDiv( FT_Long  a,
             FT_Long  b,
             FT_Long  c )
  {
    long  s;


    /* XXX: this function does not allow 64-bit arguments */
    if ( a == 0 || b == c )
      return a;

    s  = a; a = FT_ABS( a );
    s ^= b; b = FT_ABS( b );
    s ^= c; c = FT_ABS( c );

    if ( a <= 46340L && b <= 46340L && c <= 176095L && c > 0 )
      a = ( a * b + ( c >> 1 ) ) / c;

    else if ( (FT_Int32)c > 0 )
    {
      FT_Int64  temp, temp2;


      ft_multo64( (FT_Int32)a, (FT_Int32)b, &temp );

      temp2.hi = 0;
      temp2.lo = (FT_UInt32)(c >> 1);
      FT_Add64( &temp, &temp2, &temp );
      a = ft_div64by32( temp.hi, temp.lo, (FT_Int32)c );
    }
    else
      a = 0x7FFFFFFFL;

    return ( s < 0 ? -a : a );
  }


  FT_BASE_DEF( FT_Long )
  FT_MulDiv_No_Round( FT_Long  a,
                      FT_Long  b,
                      FT_Long  c )
  {
    long  s;


    if ( a == 0 || b == c )
      return a;

    s  = a; a = FT_ABS( a );
    s ^= b; b = FT_ABS( b );
    s ^= c; c = FT_ABS( c );

    if ( a <= 46340L && b <= 46340L && c > 0 )
      a = a * b / c;

    else if ( (FT_Int32)c > 0 )
    {
      FT_Int64  temp;


      ft_multo64( (FT_Int32)a, (FT_Int32)b, &temp );
      a = ft_div64by32( temp.hi, temp.lo, (FT_Int32)c );
    }
    else
      a = 0x7FFFFFFFL;

    return ( s < 0 ? -a : a );
  }


  /* documentation is in freetype.h */

  FT_EXPORT_DEF( FT_Long )
  FT_MulFix( FT_Long  a,
             FT_Long  b )
  {
#ifdef FT_MULFIX_ASSEMBLER

    return FT_MULFIX_ASSEMBLER( a, b );

#elif 0

    /*
     *  This code is nonportable.  See comment below.
     *
     *  However, on a platform where right-shift of a signed quantity fills
     *  the leftmost bits by copying the sign bit, it might be faster.
     */

    FT_Long   sa, sb;
    FT_ULong  ua, ub;


    if ( a == 0 || b == 0x10000L )
      return a;

    /*
     *  This is a clever way of converting a signed number `a' into its
     *  absolute value (stored back into `a') and its sign.  The sign is
     *  stored in `sa'; 0 means `a' was positive or zero, and -1 means `a'
     *  was negative.  (Similarly for `b' and `sb').
     *
     *  Unfortunately, it doesn't work (at least not portably).
     *
     *  It makes the assumption that right-shift on a negative signed value
     *  fills the leftmost bits by copying the sign bit.  This is wrong.
     *  According to K&R 2nd ed, section `A7.8 Shift Operators' on page 206,
     *  the result of right-shift of a negative signed value is
     *  implementation-defined.  At least one implementation fills the
     *  leftmost bits with 0s (i.e., it is exactly the same as an unsigned
     *  right shift).  This means that when `a' is negative, `sa' ends up
     *  with the value 1 rather than -1.  After that, everything else goes
     *  wrong.
     */
    sa = ( a >> ( sizeof ( a ) * 8 - 1 ) );
    a  = ( a ^ sa ) - sa;
    sb = ( b >> ( sizeof ( b ) * 8 - 1 ) );
    b  = ( b ^ sb ) - sb;

    ua = (FT_ULong)a;
    ub = (FT_ULong)b;

    if ( ua <= 2048 && ub <= 1048576L )
      ua = ( ua * ub + 0x8000U ) >> 16;
    else
    {
      FT_ULong  al = ua & 0xFFFFU;


      ua = ( ua >> 16 ) * ub +  al * ( ub >> 16 ) +
           ( ( al * ( ub & 0xFFFFU ) + 0x8000U ) >> 16 );
    }

    sa ^= sb,
    ua  = (FT_ULong)(( ua ^ sa ) - sa);

    return (FT_Long)ua;

#else /* 0 */

    FT_Long   s;
    FT_ULong  ua, ub;


    if ( a == 0 || b == 0x10000L )
      return a;

    s  = a; a = FT_ABS( a );
    s ^= b; b = FT_ABS( b );

    ua = (FT_ULong)a;
    ub = (FT_ULong)b;

    if ( ua <= 2048 && ub <= 1048576L )
      ua = ( ua * ub + 0x8000UL ) >> 16;
    else
    {
      FT_ULong  al = ua & 0xFFFFUL;


      ua = ( ua >> 16 ) * ub +  al * ( ub >> 16 ) +
           ( ( al * ( ub & 0xFFFFUL ) + 0x8000UL ) >> 16 );
    }

    return ( s < 0 ? -(FT_Long)ua : (FT_Long)ua );

#endif /* 0 */

  }


  /* documentation is in freetype.h */

  FT_EXPORT_DEF( FT_Long )
  FT_DivFix( FT_Long  a,
             FT_Long  b )
  {
    FT_Int32   s;
    FT_UInt32  q;


    /* XXX: this function does not allow 64-bit arguments */
    s  = (FT_Int32)a; a = FT_ABS( a );
    s ^= (FT_Int32)b; b = FT_ABS( b );

    if ( (FT_UInt32)b == 0 )
    {
      /* check for division by 0 */
      q = (FT_UInt32)0x7FFFFFFFL;
    }
    else if ( ( a >> 16 ) == 0 )
    {
      /* compute result directly */
      q = (FT_UInt32)( ( (FT_ULong)a << 16 ) + ( b >> 1 ) ) / (FT_UInt32)b;
    }
    else
    {
      /* we need more bits; we have to do it by hand */
      FT_Int64  temp, temp2;


      temp.hi  = (FT_Int32)( a >> 16 );
      temp.lo  = (FT_UInt32)a << 16;
      temp2.hi = 0;
      temp2.lo = (FT_UInt32)( b >> 1 );
      FT_Add64( &temp, &temp2, &temp );
      q = ft_div64by32( temp.hi, temp.lo, (FT_Int32)b );
    }

    return ( s < 0 ? -(FT_Int32)q : (FT_Int32)q );
  }


#if 0

  /* documentation is in ftcalc.h */

  FT_EXPORT_DEF( void )
  FT_MulTo64( FT_Int32   x,
              FT_Int32   y,
              FT_Int64  *z )
  {
    FT_Int32  s;


    s  = x; x = FT_ABS( x );
    s ^= y; y = FT_ABS( y );

    ft_multo64( x, y, z );

    if ( s < 0 )
    {
      z->lo = (FT_UInt32)-(FT_Int32)z->lo;
      z->hi = ~z->hi + !( z->lo );
    }
  }


  /* apparently, the second version of this code is not compiled correctly */
  /* on Mac machines with the MPW C compiler..  tsk, tsk, tsk...           */

#if 1

  FT_EXPORT_DEF( FT_Int32 )
  FT_Div64by32( FT_Int64*  x,
                FT_Int32   y )
  {
    FT_Int32   s;
    FT_UInt32  q, r, i, lo;


    s  = x->hi;
    if ( s < 0 )
    {
      x->lo = (FT_UInt32)-(FT_Int32)x->lo;
      x->hi = ~x->hi + !x->lo;
    }
    s ^= y;  y = FT_ABS( y );

    /* Shortcut */
    if ( x->hi == 0 )
    {
      if ( y > 0 )
        q = x->lo / y;
      else
        q = 0x7FFFFFFFL;

      return ( s < 0 ? -(FT_Int32)q : (FT_Int32)q );
    }

    r  = x->hi;
    lo = x->lo;

    if ( r >= (FT_UInt32)y ) /* we know y is to be treated as unsigned here */
      return ( s < 0 ? 0x80000001UL : 0x7FFFFFFFUL );
                             /* Return Max/Min Int32 if division overflow. */
                             /* This includes division by zero!            */
    q = 0;
    for ( i = 0; i < 32; i++ )
    {
      r <<= 1;
      q <<= 1;
      r  |= lo >> 31;

      if ( r >= (FT_UInt32)y )
      {
        r -= y;
        q |= 1;
      }
      lo <<= 1;
    }

    return ( s < 0 ? -(FT_Int32)q : (FT_Int32)q );
  }

#else /* 0 */

  FT_EXPORT_DEF( FT_Int32 )
  FT_Div64by32( FT_Int64*  x,
                FT_Int32   y )
  {
    FT_Int32   s;
    FT_UInt32  q;


    s  = x->hi;
    if ( s < 0 )
    {
      x->lo = (FT_UInt32)-(FT_Int32)x->lo;
      x->hi = ~x->hi + !x->lo;
    }
    s ^= y;  y = FT_ABS( y );

    /* Shortcut */
    if ( x->hi == 0 )
    {
      if ( y > 0 )
        q = ( x->lo + ( y >> 1 ) ) / y;
      else
        q = 0x7FFFFFFFL;

      return ( s < 0 ? -(FT_Int32)q : (FT_Int32)q );
    }

    q = ft_div64by32( x->hi, x->lo, y );

    return ( s < 0 ? -(FT_Int32)q : (FT_Int32)q );
  }

#endif /* 0 */

#endif /* 0 */


#endif /* FT_LONG64 */


  /* documentation is in ftglyph.h */

  FT_EXPORT_DEF( void )
  FT_Matrix_Multiply( const FT_Matrix*  a,
                      FT_Matrix        *b )
  {
    FT_Fixed  xx, xy, yx, yy;


    if ( !a || !b )
      return;

    xx = FT_MulFix( a->xx, b->xx ) + FT_MulFix( a->xy, b->yx );
    xy = FT_MulFix( a->xx, b->xy ) + FT_MulFix( a->xy, b->yy );
    yx = FT_MulFix( a->yx, b->xx ) + FT_MulFix( a->yy, b->yx );
    yy = FT_MulFix( a->yx, b->xy ) + FT_MulFix( a->yy, b->yy );

    b->xx = xx;  b->xy = xy;
    b->yx = yx;  b->yy = yy;
  }


  /* documentation is in ftglyph.h */

  FT_EXPORT_DEF( FT_Error )
  FT_Matrix_Invert( FT_Matrix*  matrix )
  {
    FT_Pos  delta, xx, yy;


    if ( !matrix )
      return FT_THROW( Invalid_Argument );

    /* compute discriminant */
    delta = FT_MulFix( matrix->xx, matrix->yy ) -
            FT_MulFix( matrix->xy, matrix->yx );

    if ( !delta )
      return FT_THROW( Invalid_Argument );  /* matrix can't be inverted */

    matrix->xy = - FT_DivFix( matrix->xy, delta );
    matrix->yx = - FT_DivFix( matrix->yx, delta );

    xx = matrix->xx;
    yy = matrix->yy;

    matrix->xx = FT_DivFix( yy, delta );
    matrix->yy = FT_DivFix( xx, delta );

    return FT_Err_Ok;
  }


  /* documentation is in ftcalc.h */

  FT_BASE_DEF( void )
  FT_Matrix_Multiply_Scaled( const FT_Matrix*  a,
                             FT_Matrix        *b,
                             FT_Long           scaling )
  {
    FT_Fixed  xx, xy, yx, yy;

    FT_Long   val = 0x10000L * scaling;


    if ( !a || !b )
      return;

    xx = FT_MulDiv( a->xx, b->xx, val ) + FT_MulDiv( a->xy, b->yx, val );
    xy = FT_MulDiv( a->xx, b->xy, val ) + FT_MulDiv( a->xy, b->yy, val );
    yx = FT_MulDiv( a->yx, b->xx, val ) + FT_MulDiv( a->yy, b->yx, val );
    yy = FT_MulDiv( a->yx, b->xy, val ) + FT_MulDiv( a->yy, b->yy, val );

    b->xx = xx;  b->xy = xy;
    b->yx = yx;  b->yy = yy;
  }


  /* documentation is in ftcalc.h */

  FT_BASE_DEF( void )
  FT_Vector_Transform_Scaled( FT_Vector*        vector,
                              const FT_Matrix*  matrix,
                              FT_Long           scaling )
  {
    FT_Pos   xz, yz;

    FT_Long  val = 0x10000L * scaling;


    if ( !vector || !matrix )
      return;

    xz = FT_MulDiv( vector->x, matrix->xx, val ) +
         FT_MulDiv( vector->y, matrix->xy, val );

    yz = FT_MulDiv( vector->x, matrix->yx, val ) +
         FT_MulDiv( vector->y, matrix->yy, val );

    vector->x = xz;
    vector->y = yz;
  }


  /* documentation is in ftcalc.h */

  FT_BASE_DEF( FT_Int32 )
  FT_SqrtFixed( FT_Int32  x )
  {
    FT_UInt32  root, rem_hi, rem_lo, test_div;
    FT_Int     count;


    root = 0;

    if ( x > 0 )
    {
      rem_hi = 0;
      rem_lo = x;
      count  = 24;
      do
      {
        rem_hi   = ( rem_hi << 2 ) | ( rem_lo >> 30 );
        rem_lo <<= 2;
        root   <<= 1;
        test_div = ( root << 1 ) + 1;

        if ( rem_hi >= test_div )
        {
          rem_hi -= test_div;
          root   += 1;
        }
      } while ( --count );
    }

    return (FT_Int32)root;
  }


  /* documentation is in ftcalc.h */

  FT_BASE_DEF( FT_Int )
  ft_corner_orientation( FT_Pos  in_x,
                         FT_Pos  in_y,
                         FT_Pos  out_x,
                         FT_Pos  out_y )
  {
    FT_Long  result; /* avoid overflow on 16-bit system */


    /* deal with the trivial cases quickly */
    if ( in_y == 0 )
    {
      if ( in_x >= 0 )
        result = out_y;
      else
        result = -out_y;
    }
    else if ( in_x == 0 )
    {
      if ( in_y >= 0 )
        result = -out_x;
      else
        result = out_x;
    }
    else if ( out_y == 0 )
    {
      if ( out_x >= 0 )
        result = in_y;
      else
        result = -in_y;
    }
    else if ( out_x == 0 )
    {
      if ( out_y >= 0 )
        result = -in_x;
      else
        result =  in_x;
    }
    else /* general case */
    {
#ifdef FT_LONG64

      FT_Int64  delta = (FT_Int64)in_x * out_y - (FT_Int64)in_y * out_x;


      if ( delta == 0 )
        result = 0;
      else
        result = 1 - 2 * ( delta < 0 );

#else

      FT_Int64  z1, z2;


      /* XXX: this function does not allow 64-bit arguments */
      ft_multo64( (FT_Int32)in_x, (FT_Int32)out_y, &z1 );
      ft_multo64( (FT_Int32)in_y, (FT_Int32)out_x, &z2 );

      if ( z1.hi > z2.hi )
        result = +1;
      else if ( z1.hi < z2.hi )
        result = -1;
      else if ( z1.lo > z2.lo )
        result = +1;
      else if ( z1.lo < z2.lo )
        result = -1;
      else
        result = 0;

#endif
    }

    /* XXX: only the sign of return value, +1/0/-1 must be used */
    return (FT_Int)result;
  }


  /* documentation is in ftcalc.h */

  FT_BASE_DEF( FT_Int )
  ft_corner_is_flat( FT_Pos  in_x,
                     FT_Pos  in_y,
                     FT_Pos  out_x,
                     FT_Pos  out_y )
  {
    FT_Pos  ax = in_x;
    FT_Pos  ay = in_y;

    FT_Pos  d_in, d_out, d_corner;


    if ( ax < 0 )
      ax = -ax;
    if ( ay < 0 )
      ay = -ay;
    d_in = ax + ay;

    ax = out_x;
    if ( ax < 0 )
      ax = -ax;
    ay = out_y;
    if ( ay < 0 )
      ay = -ay;
    d_out = ax + ay;

    ax = out_x + in_x;
    if ( ax < 0 )
      ax = -ax;
    ay = out_y + in_y;
    if ( ay < 0 )
      ay = -ay;
    d_corner = ax + ay;

    return ( d_in + d_out - d_corner ) < ( d_corner >> 4 );
  }


/* END */
