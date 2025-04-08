/****************************************************************************
 *
 * psconv.c
 *
 *   Some convenience conversions (body).
 *
 * Copyright (C) 2006-2024 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#include <freetype/internal/psaux.h>
#include <freetype/internal/ftdebug.h>

#include "psconv.h"
#include "psauxerr.h"


  /**************************************************************************
   *
   * The macro FT_COMPONENT is used in trace mode.  It is an implicit
   * parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log
   * messages during execution.
   */
#undef  FT_COMPONENT
#define FT_COMPONENT  psconv


  /* The following array is used by various functions to quickly convert */
  /* digits (both decimal and non-decimal) into numbers.                 */

#if 'A' == 65
  /* ASCII */

  static const FT_Char  ft_char_table[128] =
  {
    /* 0x00 */
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
     0,  1,  2,  3,  4,  5,  6,  7,  8,  9, -1, -1, -1, -1, -1, -1,
    -1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
    25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, -1, -1, -1, -1, -1,
    -1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
    25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, -1, -1, -1, -1, -1,
  };

  /* no character >= 0x80 can represent a valid number */
#define OP  >=

#endif /* 'A' == 65 */

#if 'A' == 193
  /* EBCDIC */

  static const FT_Char  ft_char_table[128] =
  {
    /* 0x80 */
    -1, 10, 11, 12, 13, 14, 15, 16, 17, 18, -1, -1, -1, -1, -1, -1,
    -1, 19, 20, 21, 22, 23, 24, 25, 26, 27, -1, -1, -1, -1, -1, -1,
    -1, -1, 28, 29, 30, 31, 32, 33, 34, 35, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, 10, 11, 12, 13, 14, 15, 16, 17, 18, -1, -1, -1, -1, -1, -1,
    -1, 19, 20, 21, 22, 23, 24, 25, 26, 27, -1, -1, -1, -1, -1, -1,
    -1, -1, 28, 29, 30, 31, 32, 33, 34, 35, -1, -1, -1, -1, -1, -1,
     0,  1,  2,  3,  4,  5,  6,  7,  8,  9, -1, -1, -1, -1, -1, -1,
  };

  /* no character < 0x80 can represent a valid number */
#define OP  <

#endif /* 'A' == 193 */


  FT_LOCAL_DEF( FT_Long )
  PS_Conv_Strtol( FT_Byte**  cursor,
                  FT_Byte*   limit,
                  FT_Long    base )
  {
    FT_Byte*  p = *cursor;

    FT_Long   num           = 0;
    FT_Bool   sign          = 0;
    FT_Bool   have_overflow = 0;

    FT_Long   num_limit;
    FT_Char   c_limit;


    if ( p >= limit )
      goto Bad;

    if ( base < 2 || base > 36 )
    {
      FT_TRACE4(( "!!!INVALID BASE:!!!" ));
      return 0;
    }

    if ( *p == '-' || *p == '+' )
    {
      sign = FT_BOOL( *p == '-' );

      p++;
      if ( p == limit )
        goto Bad;

      /* only a single sign is allowed */
      if ( *p == '-' || *p == '+' )
        return 0;
    }

    num_limit = 0x7FFFFFFFL / base;
    c_limit   = (FT_Char)( 0x7FFFFFFFL % base );

    for ( ; p < limit; p++ )
    {
      FT_Char  c;


      if ( IS_PS_SPACE( *p ) || *p OP 0x80 )
        break;

      c = ft_char_table[*p & 0x7F];

      if ( c < 0 || c >= base )
        break;

      if ( num > num_limit || ( num == num_limit && c > c_limit ) )
        have_overflow = 1;
      else
        num = num * base + c;
    }

    *cursor = p;

    if ( have_overflow )
    {
      num = 0x7FFFFFFFL;
      FT_TRACE4(( "!!!OVERFLOW:!!!" ));
    }

    if ( sign )
      num = -num;

    return num;

  Bad:
    FT_TRACE4(( "!!!END OF DATA:!!!" ));
    return 0;
  }


  FT_LOCAL_DEF( FT_Long )
  PS_Conv_ToInt( FT_Byte**  cursor,
                 FT_Byte*   limit )

  {
    FT_Byte*  p = *cursor;
    FT_Byte*  curp;

    FT_Long   num;


    curp = p;
    num  = PS_Conv_Strtol( &p, limit, 10 );

    if ( p == curp )
      return 0;

    if ( p < limit && *p == '#' )
    {
      p++;

      curp = p;
      num  = PS_Conv_Strtol( &p, limit, num );

      if ( p == curp )
        return 0;
    }

    *cursor = p;

    return num;
  }


  FT_LOCAL_DEF( FT_Fixed )
  PS_Conv_ToFixed( FT_Byte**  cursor,
                   FT_Byte*   limit,
                   FT_Long    power_ten )
  {
    FT_Byte*  p = *cursor;
    FT_Byte*  curp;

    FT_Fixed  integral = 0;
    FT_Long   decimal  = 0;
    FT_Long   divider  = 1;

    FT_Bool   sign           = 0;
    FT_Bool   have_overflow  = 0;
    FT_Bool   have_underflow = 0;


    if ( p >= limit )
      goto Bad;

    if ( *p == '-' || *p == '+' )
    {
      sign = FT_BOOL( *p == '-' );

      p++;
      if ( p == limit )
        goto Bad;

      /* only a single sign is allowed */
      if ( *p == '-' || *p == '+' )
        return 0;
    }

    /* read the integer part */
    if ( *p != '.' )
    {
      curp     = p;
      integral = PS_Conv_ToInt( &p, limit );

      if ( p == curp )
        return 0;

      if ( integral > 0x7FFF )
        have_overflow = 1;
      else
        integral = (FT_Fixed)( (FT_UInt32)integral << 16 );
    }

    /* read the decimal part */
    if ( p < limit && *p == '.' )
    {
      p++;

      for ( ; p < limit; p++ )
      {
        FT_Char  c;


        if ( IS_PS_SPACE( *p ) || *p OP 0x80 )
          break;

        c = ft_char_table[*p & 0x7F];

        if ( c < 0 || c >= 10 )
          break;

        /* only add digit if we don't overflow */
        if ( divider < 0xCCCCCCCL && decimal < 0xCCCCCCCL )
        {
          decimal = decimal * 10 + c;

          if ( !integral && power_ten > 0 )
            power_ten--;
          else
            divider *= 10;
        }
      }
    }

    /* read exponent, if any */
    if ( p + 1 < limit && ( *p == 'e' || *p == 'E' ) )
    {
      FT_Long  exponent;


      p++;

      curp     = p;
      exponent = PS_Conv_ToInt( &p, limit );

      if ( curp == p )
        return 0;

      /* arbitrarily limit exponent */
      if ( exponent > 1000 )
        have_overflow = 1;
      else if ( exponent < -1000 )
        have_underflow = 1;
      else
        power_ten += exponent;
    }

    *cursor = p;

    if ( !integral && !decimal )
      return 0;

    if ( have_overflow )
      goto Overflow;
    if ( have_underflow )
      goto Underflow;

    while ( power_ten > 0 )
    {
      if ( integral >= 0xCCCCCCCL )
        goto Overflow;
      integral *= 10;

      if ( decimal >= 0xCCCCCCCL )
      {
        if ( divider == 1 )
          goto Overflow;
        divider /= 10;
      }
      else
        decimal *= 10;

      power_ten--;
    }

    while ( power_ten < 0 )
    {
      integral /= 10;
      if ( divider < 0xCCCCCCCL )
        divider *= 10;
      else
        decimal /= 10;

      if ( !integral && !decimal )
        goto Underflow;

      power_ten++;
    }

    if ( decimal )
    {
      decimal = FT_DivFix( decimal, divider );
      /* it's not necessary to check this addition for overflow */
      /* due to the structure of the real number representation */
      integral += decimal;
    }

  Exit:
    if ( sign )
      integral = -integral;

    return integral;

  Bad:
    FT_TRACE4(( "!!!END OF DATA:!!!" ));
    return 0;

  Overflow:
    integral = 0x7FFFFFFFL;
    FT_TRACE4(( "!!!OVERFLOW:!!!" ));
    goto Exit;

  Underflow:
    FT_TRACE4(( "!!!UNDERFLOW:!!!" ));
    return 0;
  }


#if 0
  FT_LOCAL_DEF( FT_UInt )
  PS_Conv_StringDecode( FT_Byte**  cursor,
                        FT_Byte*   limit,
                        FT_Byte*   buffer,
                        FT_Offset  n )
  {
    FT_Byte*  p;
    FT_UInt   r = 0;


    for ( p = *cursor; r < n && p < limit; p++ )
    {
      FT_Byte  b;


      if ( *p != '\\' )
      {
        buffer[r++] = *p;

        continue;
      }

      p++;

      switch ( *p )
      {
      case 'n':
        b = '\n';
        break;
      case 'r':
        b = '\r';
        break;
      case 't':
        b = '\t';
        break;
      case 'b':
        b = '\b';
        break;
      case 'f':
        b = '\f';
        break;
      case '\r':
        p++;
        if ( *p != '\n' )
        {
          b = *p;

          break;
        }
        /* no break */
      case '\n':
        continue;
        break;
      default:
        if ( IS_PS_DIGIT( *p ) )
        {
          b = *p - '0';

          p++;

          if ( IS_PS_DIGIT( *p ) )
          {
            b = b * 8 + *p - '0';

            p++;

            if ( IS_PS_DIGIT( *p ) )
              b = b * 8 + *p - '0';
            else
            {
              buffer[r++] = b;
              b = *p;
            }
          }
          else
          {
            buffer[r++] = b;
            b = *p;
          }
        }
        else
          b = *p;
        break;
      }

      buffer[r++] = b;
    }

    *cursor = p;

    return r;
  }
#endif /* 0 */


  FT_LOCAL_DEF( FT_UInt )
  PS_Conv_ASCIIHexDecode( FT_Byte**  cursor,
                          FT_Byte*   limit,
                          FT_Byte*   buffer,
                          FT_Offset  n )
  {
    FT_Byte*  p;
    FT_UInt   r   = 0;
    FT_UInt   w   = 0;
    FT_UInt   pad = 0x01;


    n *= 2;

#if 1

    p = *cursor;

    if ( p >= limit )
      return 0;

    if ( n > (FT_UInt)( limit - p ) )
      n = (FT_UInt)( limit - p );

    /* we try to process two nibbles at a time to be as fast as possible */
    for ( ; r < n; r++ )
    {
      FT_UInt  c = p[r];


      if ( IS_PS_SPACE( c ) )
        continue;

      if ( c OP 0x80 )
        break;

      c = (FT_UInt)ft_char_table[c & 0x7F];
      if ( c >= 16 )
        break;

      pad = ( pad << 4 ) | c;
      if ( pad & 0x100 )
      {
        buffer[w++] = (FT_Byte)pad;
        pad         = 0x01;
      }
    }

    if ( pad != 0x01 )
      buffer[w++] = (FT_Byte)( pad << 4 );

    *cursor = p + r;

    return w;

#else /* 0 */

    for ( r = 0; r < n; r++ )
    {
      FT_Char  c;


      if ( IS_PS_SPACE( *p ) )
        continue;

      if ( *p OP 0x80 )
        break;

      c = ft_char_table[*p & 0x7F];

      if ( (unsigned)c >= 16 )
        break;

      if ( r & 1 )
      {
        *buffer = (FT_Byte)( *buffer + c );
        buffer++;
      }
      else
        *buffer = (FT_Byte)( c << 4 );

      r++;
    }

    *cursor = p;

    return ( r + 1 ) / 2;

#endif /* 0 */

  }


  FT_LOCAL_DEF( FT_UInt )
  PS_Conv_EexecDecode( FT_Byte**   cursor,
                       FT_Byte*    limit,
                       FT_Byte*    buffer,
                       FT_Offset   n,
                       FT_UShort*  seed )
  {
    FT_Byte*  p;
    FT_UInt   r;
    FT_UInt   s = *seed;


#if 1

    p = *cursor;

    if ( p >= limit )
      return 0;

    if ( n > (FT_UInt)( limit - p ) )
      n = (FT_UInt)( limit - p );

    for ( r = 0; r < n; r++ )
    {
      FT_UInt  val = p[r];
      FT_UInt  b   = ( val ^ ( s >> 8 ) );


      s         = ( (val + s)*52845U + 22719 ) & 0xFFFFU;
      buffer[r] = (FT_Byte) b;
    }

    *cursor = p + n;
    *seed   = (FT_UShort)s;

#else /* 0 */

    for ( r = 0, p = *cursor; r < n && p < limit; r++, p++ )
    {
      FT_Byte  b = (FT_Byte)( *p ^ ( s >> 8 ) );


      s = (FT_UShort)( ( *p + s ) * 52845U + 22719 );
      *buffer++ = b;
    }
    *cursor = p;
    *seed   = s;

#endif /* 0 */

    return r;
  }


/* END */
