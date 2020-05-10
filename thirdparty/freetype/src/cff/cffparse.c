/****************************************************************************
 *
 * cffparse.c
 *
 *   CFF token stream parser (body)
 *
 * Copyright (C) 1996-2020 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#include <ft2build.h>
#include "cffparse.h"
#include FT_INTERNAL_STREAM_H
#include FT_INTERNAL_DEBUG_H
#include FT_INTERNAL_CALC_H
#include FT_INTERNAL_POSTSCRIPT_AUX_H
#include FT_LIST_H

#include "cfferrs.h"
#include "cffload.h"


  /**************************************************************************
   *
   * The macro FT_COMPONENT is used in trace mode.  It is an implicit
   * parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log
   * messages during execution.
   */
#undef  FT_COMPONENT
#define FT_COMPONENT  cffparse


  FT_LOCAL_DEF( FT_Error )
  cff_parser_init( CFF_Parser  parser,
                   FT_UInt     code,
                   void*       object,
                   FT_Library  library,
                   FT_UInt     stackSize,
                   FT_UShort   num_designs,
                   FT_UShort   num_axes )
  {
    FT_Memory  memory = library->memory;    /* for FT_NEW_ARRAY */
    FT_Error   error;                       /* for FT_NEW_ARRAY */


    FT_ZERO( parser );

#if 0
    parser->top         = parser->stack;
#endif
    parser->object_code = code;
    parser->object      = object;
    parser->library     = library;
    parser->num_designs = num_designs;
    parser->num_axes    = num_axes;

    /* allocate the stack buffer */
    if ( FT_NEW_ARRAY( parser->stack, stackSize ) )
    {
      FT_FREE( parser->stack );
      goto Exit;
    }

    parser->stackSize = stackSize;
    parser->top       = parser->stack;    /* empty stack */

  Exit:
    return error;
  }


#ifdef CFF_CONFIG_OPTION_OLD_ENGINE
  static void
  finalize_t2_strings( FT_Memory  memory,
                       void*      data,
                       void*      user )
  {
    CFF_T2_String  t2 = (CFF_T2_String)data;


    FT_UNUSED( user );

    memory->free( memory, t2->start );
    memory->free( memory, data );
  }
#endif /* CFF_CONFIG_OPTION_OLD_ENGINE */


  FT_LOCAL_DEF( void )
  cff_parser_done( CFF_Parser  parser )
  {
    FT_Memory  memory = parser->library->memory;    /* for FT_FREE */


    FT_FREE( parser->stack );

#ifdef CFF_CONFIG_OPTION_OLD_ENGINE
    FT_List_Finalize( &parser->t2_strings,
                      finalize_t2_strings,
                      memory,
                      NULL );
#endif
  }


  /* Assuming `first >= last'. */

  static FT_Error
  cff_parser_within_limits( CFF_Parser  parser,
                            FT_Byte*    first,
                            FT_Byte*    last )
  {
#ifndef CFF_CONFIG_OPTION_OLD_ENGINE

    /* Fast path for regular FreeType builds with the "new" engine; */
    /*   `first >= parser->start' can be assumed.                   */

    FT_UNUSED( first );

    return last < parser->limit ? FT_Err_Ok : FT_THROW( Invalid_Argument );

#else /* CFF_CONFIG_OPTION_OLD_ENGINE */

    FT_ListNode  node;


    if ( first >= parser->start &&
         last  <  parser->limit )
      return FT_Err_Ok;

    node = parser->t2_strings.head;

    while ( node )
    {
      CFF_T2_String  t2 = (CFF_T2_String)node->data;


      if ( first >= t2->start &&
           last  <  t2->limit )
        return FT_Err_Ok;

      node = node->next;
    }

    return FT_THROW( Invalid_Argument );

#endif /* CFF_CONFIG_OPTION_OLD_ENGINE */
  }


  /* read an integer */
  static FT_Long
  cff_parse_integer( CFF_Parser  parser,
                     FT_Byte*    start )
  {
    FT_Byte*  p   = start;
    FT_Int    v   = *p++;
    FT_Long   val = 0;


    if ( v == 28 )
    {
      if ( cff_parser_within_limits( parser, p, p + 1 ) )
        goto Bad;

      val = (FT_Short)( ( (FT_UShort)p[0] << 8 ) | p[1] );
    }
    else if ( v == 29 )
    {
      if ( cff_parser_within_limits( parser, p, p + 3 ) )
        goto Bad;

      val = (FT_Long)( ( (FT_ULong)p[0] << 24 ) |
                       ( (FT_ULong)p[1] << 16 ) |
                       ( (FT_ULong)p[2] <<  8 ) |
                         (FT_ULong)p[3]         );
    }
    else if ( v < 247 )
    {
      val = v - 139;
    }
    else if ( v < 251 )
    {
      if ( cff_parser_within_limits( parser, p, p ) )
        goto Bad;

      val = ( v - 247 ) * 256 + p[0] + 108;
    }
    else
    {
      if ( cff_parser_within_limits( parser, p, p ) )
        goto Bad;

      val = -( v - 251 ) * 256 - p[0] - 108;
    }

  Exit:
    return val;

  Bad:
    val = 0;
    FT_TRACE4(( "!!!END OF DATA:!!!" ));
    goto Exit;
  }


  static const FT_Long power_tens[] =
  {
    1L,
    10L,
    100L,
    1000L,
    10000L,
    100000L,
    1000000L,
    10000000L,
    100000000L,
    1000000000L
  };

  /* maximum values allowed for multiplying      */
  /* with the corresponding `power_tens' element */
  static const FT_Long power_ten_limits[] =
  {
    FT_LONG_MAX / 1L,
    FT_LONG_MAX / 10L,
    FT_LONG_MAX / 100L,
    FT_LONG_MAX / 1000L,
    FT_LONG_MAX / 10000L,
    FT_LONG_MAX / 100000L,
    FT_LONG_MAX / 1000000L,
    FT_LONG_MAX / 10000000L,
    FT_LONG_MAX / 100000000L,
    FT_LONG_MAX / 1000000000L,
  };


  /* read a real */
  static FT_Fixed
  cff_parse_real( CFF_Parser  parser,
                  FT_Byte*    start,
                  FT_Long     power_ten,
                  FT_Long*    scaling )
  {
    FT_Byte*  p = start;
    FT_Int    nib;
    FT_UInt   phase;

    FT_Long   result, number, exponent;
    FT_Int    sign = 0, exponent_sign = 0, have_overflow = 0;
    FT_Long   exponent_add, integer_length, fraction_length;


    if ( scaling )
      *scaling = 0;

    result = 0;

    number   = 0;
    exponent = 0;

    exponent_add    = 0;
    integer_length  = 0;
    fraction_length = 0;

    /* First of all, read the integer part. */
    phase = 4;

    for (;;)
    {
      /* If we entered this iteration with phase == 4, we need to */
      /* read a new byte.  This also skips past the initial 0x1E. */
      if ( phase )
      {
        p++;

        /* Make sure we don't read past the end. */
        if ( cff_parser_within_limits( parser, p, p ) )
          goto Bad;
      }

      /* Get the nibble. */
      nib   = (FT_Int)( p[0] >> phase ) & 0xF;
      phase = 4 - phase;

      if ( nib == 0xE )
        sign = 1;
      else if ( nib > 9 )
        break;
      else
      {
        /* Increase exponent if we can't add the digit. */
        if ( number >= 0xCCCCCCCL )
          exponent_add++;
        /* Skip leading zeros. */
        else if ( nib || number )
        {
          integer_length++;
          number = number * 10 + nib;
        }
      }
    }

    /* Read fraction part, if any. */
    if ( nib == 0xA )
      for (;;)
      {
        /* If we entered this iteration with phase == 4, we need */
        /* to read a new byte.                                   */
        if ( phase )
        {
          p++;

          /* Make sure we don't read past the end. */
          if ( cff_parser_within_limits( parser, p, p ) )
            goto Bad;
        }

        /* Get the nibble. */
        nib   = ( p[0] >> phase ) & 0xF;
        phase = 4 - phase;
        if ( nib >= 10 )
          break;

        /* Skip leading zeros if possible. */
        if ( !nib && !number )
          exponent_add--;
        /* Only add digit if we don't overflow. */
        else if ( number < 0xCCCCCCCL && fraction_length < 9 )
        {
          fraction_length++;
          number = number * 10 + nib;
        }
      }

    /* Read exponent, if any. */
    if ( nib == 12 )
    {
      exponent_sign = 1;
      nib           = 11;
    }

    if ( nib == 11 )
    {
      for (;;)
      {
        /* If we entered this iteration with phase == 4, */
        /* we need to read a new byte.                   */
        if ( phase )
        {
          p++;

          /* Make sure we don't read past the end. */
          if ( cff_parser_within_limits( parser, p, p ) )
            goto Bad;
        }

        /* Get the nibble. */
        nib   = ( p[0] >> phase ) & 0xF;
        phase = 4 - phase;
        if ( nib >= 10 )
          break;

        /* Arbitrarily limit exponent. */
        if ( exponent > 1000 )
          have_overflow = 1;
        else
          exponent = exponent * 10 + nib;
      }

      if ( exponent_sign )
        exponent = -exponent;
    }

    if ( !number )
      goto Exit;

    if ( have_overflow )
    {
      if ( exponent_sign )
        goto Underflow;
      else
        goto Overflow;
    }

    /* We don't check `power_ten' and `exponent_add'. */
    exponent += power_ten + exponent_add;

    if ( scaling )
    {
      /* Only use `fraction_length'. */
      fraction_length += integer_length;
      exponent        += integer_length;

      if ( fraction_length <= 5 )
      {
        if ( number > 0x7FFFL )
        {
          result   = FT_DivFix( number, 10 );
          *scaling = exponent - fraction_length + 1;
        }
        else
        {
          if ( exponent > 0 )
          {
            FT_Long  new_fraction_length, shift;


            /* Make `scaling' as small as possible. */
            new_fraction_length = FT_MIN( exponent, 5 );
            shift               = new_fraction_length - fraction_length;

            if ( shift > 0 )
            {
              exponent -= new_fraction_length;
              number   *= power_tens[shift];
              if ( number > 0x7FFFL )
              {
                number   /= 10;
                exponent += 1;
              }
            }
            else
              exponent -= fraction_length;
          }
          else
            exponent -= fraction_length;

          result   = (FT_Long)( (FT_ULong)number << 16 );
          *scaling = exponent;
        }
      }
      else
      {
        if ( ( number / power_tens[fraction_length - 5] ) > 0x7FFFL )
        {
          result   = FT_DivFix( number, power_tens[fraction_length - 4] );
          *scaling = exponent - 4;
        }
        else
        {
          result   = FT_DivFix( number, power_tens[fraction_length - 5] );
          *scaling = exponent - 5;
        }
      }
    }
    else
    {
      integer_length  += exponent;
      fraction_length -= exponent;

      if ( integer_length > 5 )
        goto Overflow;
      if ( integer_length < -5 )
        goto Underflow;

      /* Remove non-significant digits. */
      if ( integer_length < 0 )
      {
        number          /= power_tens[-integer_length];
        fraction_length += integer_length;
      }

      /* this can only happen if exponent was non-zero */
      if ( fraction_length == 10 )
      {
        number          /= 10;
        fraction_length -= 1;
      }

      /* Convert into 16.16 format. */
      if ( fraction_length > 0 )
      {
        if ( ( number / power_tens[fraction_length] ) > 0x7FFFL )
          goto Exit;

        result = FT_DivFix( number, power_tens[fraction_length] );
      }
      else
      {
        number *= power_tens[-fraction_length];

        if ( number > 0x7FFFL )
          goto Overflow;

        result = (FT_Long)( (FT_ULong)number << 16 );
      }
    }

  Exit:
    if ( sign )
      result = -result;

    return result;

  Overflow:
    result = 0x7FFFFFFFL;
    FT_TRACE4(( "!!!OVERFLOW:!!!" ));
    goto Exit;

  Underflow:
    result = 0;
    FT_TRACE4(( "!!!UNDERFLOW:!!!" ));
    goto Exit;

  Bad:
    result = 0;
    FT_TRACE4(( "!!!END OF DATA:!!!" ));
    goto Exit;
  }


  /* read a number, either integer or real */
  FT_LOCAL_DEF( FT_Long )
  cff_parse_num( CFF_Parser  parser,
                 FT_Byte**   d )
  {
    if ( **d == 30 )
    {
      /* binary-coded decimal is truncated to integer */
      return cff_parse_real( parser, *d, 0, NULL ) >> 16;
    }

    else if ( **d == 255 )
    {
      /* 16.16 fixed point is used internally for CFF2 blend results. */
      /* Since these are trusted values, a limit check is not needed. */

      /* After the 255, 4 bytes give the number.                 */
      /* The blend value is converted to integer, with rounding; */
      /* due to the right-shift we don't need the lowest byte.   */
#if 0
      return (FT_Short)(
               ( ( ( (FT_UInt32)*( d[0] + 1 ) << 24 ) |
                   ( (FT_UInt32)*( d[0] + 2 ) << 16 ) |
                   ( (FT_UInt32)*( d[0] + 3 ) <<  8 ) |
                     (FT_UInt32)*( d[0] + 4 )         ) + 0x8000U ) >> 16 );
#else
      return (FT_Short)(
               ( ( ( (FT_UInt32)*( d[0] + 1 ) << 16 ) |
                   ( (FT_UInt32)*( d[0] + 2 ) <<  8 ) |
                     (FT_UInt32)*( d[0] + 3 )         ) + 0x80U ) >> 8 );
#endif
    }

    else
      return cff_parse_integer( parser, *d );
  }


  /* read a floating point number, either integer or real */
  static FT_Fixed
  do_fixed( CFF_Parser  parser,
            FT_Byte**   d,
            FT_Long     scaling )
  {
    if ( **d == 30 )
      return cff_parse_real( parser, *d, scaling, NULL );
    else
    {
      FT_Long  val = cff_parse_integer( parser, *d );


      if ( scaling )
      {
        if ( FT_ABS( val ) > power_ten_limits[scaling] )
        {
          val = val > 0 ? 0x7FFFFFFFL : -0x7FFFFFFFL;
          goto Overflow;
        }

        val *= power_tens[scaling];
      }

      if ( val > 0x7FFF )
      {
        val = 0x7FFFFFFFL;
        goto Overflow;
      }
      else if ( val < -0x7FFF )
      {
        val = -0x7FFFFFFFL;
        goto Overflow;
      }

      return (FT_Long)( (FT_ULong)val << 16 );

    Overflow:
      FT_TRACE4(( "!!!OVERFLOW:!!!" ));
      return val;
    }
  }


  /* read a floating point number, either integer or real */
  static FT_Fixed
  cff_parse_fixed( CFF_Parser  parser,
                   FT_Byte**   d )
  {
    return do_fixed( parser, d, 0 );
  }


  /* read a floating point number, either integer or real, */
  /* but return `10^scaling' times the number read in      */
  static FT_Fixed
  cff_parse_fixed_scaled( CFF_Parser  parser,
                          FT_Byte**   d,
                          FT_Long     scaling )
  {
    return do_fixed( parser, d, scaling );
  }


  /* read a floating point number, either integer or real,     */
  /* and return it as precise as possible -- `scaling' returns */
  /* the scaling factor (as a power of 10)                     */
  static FT_Fixed
  cff_parse_fixed_dynamic( CFF_Parser  parser,
                           FT_Byte**   d,
                           FT_Long*    scaling )
  {
    FT_ASSERT( scaling );

    if ( **d == 30 )
      return cff_parse_real( parser, *d, 0, scaling );
    else
    {
      FT_Long  number;
      FT_Int   integer_length;


      number = cff_parse_integer( parser, d[0] );

      if ( number > 0x7FFFL )
      {
        for ( integer_length = 5; integer_length < 10; integer_length++ )
          if ( number < power_tens[integer_length] )
            break;

        if ( ( number / power_tens[integer_length - 5] ) > 0x7FFFL )
        {
          *scaling = integer_length - 4;
          return FT_DivFix( number, power_tens[integer_length - 4] );
        }
        else
        {
          *scaling = integer_length - 5;
          return FT_DivFix( number, power_tens[integer_length - 5] );
        }
      }
      else
      {
        *scaling = 0;
        return (FT_Long)( (FT_ULong)number << 16 );
      }
    }
  }


  static FT_Error
  cff_parse_font_matrix( CFF_Parser  parser )
  {
    CFF_FontRecDict  dict   = (CFF_FontRecDict)parser->object;
    FT_Matrix*       matrix = &dict->font_matrix;
    FT_Vector*       offset = &dict->font_offset;
    FT_ULong*        upm    = &dict->units_per_em;
    FT_Byte**        data   = parser->stack;


    if ( parser->top >= parser->stack + 6 )
    {
      FT_Fixed  values[6];
      FT_Long   scalings[6];

      FT_Long  min_scaling, max_scaling;
      int      i;


      dict->has_font_matrix = TRUE;

      /* We expect a well-formed font matrix, this is, the matrix elements */
      /* `xx' and `yy' are of approximately the same magnitude.  To avoid  */
      /* loss of precision, we use the magnitude of the largest matrix     */
      /* element to scale all other elements.  The scaling factor is then  */
      /* contained in the `units_per_em' value.                            */

      max_scaling = FT_LONG_MIN;
      min_scaling = FT_LONG_MAX;

      for ( i = 0; i < 6; i++ )
      {
        values[i] = cff_parse_fixed_dynamic( parser, data++, &scalings[i] );
        if ( values[i] )
        {
          if ( scalings[i] > max_scaling )
            max_scaling = scalings[i];
          if ( scalings[i] < min_scaling )
            min_scaling = scalings[i];
        }
      }

      if ( max_scaling < -9                  ||
           max_scaling > 0                   ||
           ( max_scaling - min_scaling ) < 0 ||
           ( max_scaling - min_scaling ) > 9 )
      {
        FT_TRACE1(( "cff_parse_font_matrix:"
                    " strange scaling values (minimum %d, maximum %d),\n"
                    "                      "
                    " using default matrix\n", min_scaling, max_scaling ));
        goto Unlikely;
      }

      for ( i = 0; i < 6; i++ )
      {
        FT_Fixed  value = values[i];
        FT_Long   divisor, half_divisor;


        if ( !value )
          continue;

        divisor      = power_tens[max_scaling - scalings[i]];
        half_divisor = divisor >> 1;

        if ( value < 0 )
        {
          if ( FT_LONG_MIN + half_divisor < value )
            values[i] = ( value - half_divisor ) / divisor;
          else
            values[i] = FT_LONG_MIN / divisor;
        }
        else
        {
          if ( FT_LONG_MAX - half_divisor > value )
            values[i] = ( value + half_divisor ) / divisor;
          else
            values[i] = FT_LONG_MAX / divisor;
        }
      }

      matrix->xx = values[0];
      matrix->yx = values[1];
      matrix->xy = values[2];
      matrix->yy = values[3];
      offset->x  = values[4];
      offset->y  = values[5];

      *upm = (FT_ULong)power_tens[-max_scaling];

      FT_TRACE4(( " [%f %f %f %f %f %f]\n",
                  (double)matrix->xx / *upm / 65536,
                  (double)matrix->xy / *upm / 65536,
                  (double)matrix->yx / *upm / 65536,
                  (double)matrix->yy / *upm / 65536,
                  (double)offset->x  / *upm / 65536,
                  (double)offset->y  / *upm / 65536 ));

      if ( !FT_Matrix_Check( matrix ) )
      {
        FT_TRACE1(( "cff_parse_font_matrix:"
                    " degenerate values, using default matrix\n" ));
        goto Unlikely;
      }

      return FT_Err_Ok;
    }
    else
      return FT_THROW( Stack_Underflow );

  Unlikely:
    /* Return default matrix in case of unlikely values. */

    matrix->xx = 0x10000L;
    matrix->yx = 0;
    matrix->xy = 0;
    matrix->yy = 0x10000L;
    offset->x  = 0;
    offset->y  = 0;
    *upm       = 1;

    return FT_Err_Ok;
  }


  static FT_Error
  cff_parse_font_bbox( CFF_Parser  parser )
  {
    CFF_FontRecDict  dict = (CFF_FontRecDict)parser->object;
    FT_BBox*         bbox = &dict->font_bbox;
    FT_Byte**        data = parser->stack;
    FT_Error         error;


    error = FT_ERR( Stack_Underflow );

    if ( parser->top >= parser->stack + 4 )
    {
      bbox->xMin = FT_RoundFix( cff_parse_fixed( parser, data++ ) );
      bbox->yMin = FT_RoundFix( cff_parse_fixed( parser, data++ ) );
      bbox->xMax = FT_RoundFix( cff_parse_fixed( parser, data++ ) );
      bbox->yMax = FT_RoundFix( cff_parse_fixed( parser, data   ) );
      error = FT_Err_Ok;

      FT_TRACE4(( " [%d %d %d %d]\n",
                  bbox->xMin / 65536,
                  bbox->yMin / 65536,
                  bbox->xMax / 65536,
                  bbox->yMax / 65536 ));
    }

    return error;
  }


  static FT_Error
  cff_parse_private_dict( CFF_Parser  parser )
  {
    CFF_FontRecDict  dict = (CFF_FontRecDict)parser->object;
    FT_Byte**        data = parser->stack;
    FT_Error         error;


    error = FT_ERR( Stack_Underflow );

    if ( parser->top >= parser->stack + 2 )
    {
      FT_Long  tmp;


      tmp = cff_parse_num( parser, data++ );
      if ( tmp < 0 )
      {
        FT_ERROR(( "cff_parse_private_dict: Invalid dictionary size\n" ));
        error = FT_THROW( Invalid_File_Format );
        goto Fail;
      }
      dict->private_size = (FT_ULong)tmp;

      tmp = cff_parse_num( parser, data );
      if ( tmp < 0 )
      {
        FT_ERROR(( "cff_parse_private_dict: Invalid dictionary offset\n" ));
        error = FT_THROW( Invalid_File_Format );
        goto Fail;
      }
      dict->private_offset = (FT_ULong)tmp;

      FT_TRACE4(( " %lu %lu\n",
                  dict->private_size, dict->private_offset ));

      error = FT_Err_Ok;
    }

  Fail:
    return error;
  }


  /* The `MultipleMaster' operator comes before any  */
  /* top DICT operators that contain T2 charstrings. */

  static FT_Error
  cff_parse_multiple_master( CFF_Parser  parser )
  {
    CFF_FontRecDict  dict = (CFF_FontRecDict)parser->object;
    FT_Error         error;


#ifdef FT_DEBUG_LEVEL_TRACE
    /* beautify tracing message */
    if ( ft_trace_levels[FT_TRACE_COMP( FT_COMPONENT )] < 4 )
      FT_TRACE1(( "Multiple Master CFFs not supported yet,"
                  " handling first master design only\n" ));
    else
      FT_TRACE1(( " (not supported yet,"
                  " handling first master design only)\n" ));
#endif

    error = FT_ERR( Stack_Underflow );

    /* currently, we handle only the first argument */
    if ( parser->top >= parser->stack + 5 )
    {
      FT_Long  num_designs = cff_parse_num( parser, parser->stack );


      if ( num_designs > 16 || num_designs < 2 )
      {
        FT_ERROR(( "cff_parse_multiple_master:"
                   " Invalid number of designs\n" ));
        error = FT_THROW( Invalid_File_Format );
      }
      else
      {
        dict->num_designs   = (FT_UShort)num_designs;
        dict->num_axes      = (FT_UShort)( parser->top - parser->stack - 4 );

        parser->num_designs = dict->num_designs;
        parser->num_axes    = dict->num_axes;

        error = FT_Err_Ok;
      }
    }

    return error;
  }


  static FT_Error
  cff_parse_cid_ros( CFF_Parser  parser )
  {
    CFF_FontRecDict  dict = (CFF_FontRecDict)parser->object;
    FT_Byte**        data = parser->stack;
    FT_Error         error;


    error = FT_ERR( Stack_Underflow );

    if ( parser->top >= parser->stack + 3 )
    {
      dict->cid_registry = (FT_UInt)cff_parse_num( parser, data++ );
      dict->cid_ordering = (FT_UInt)cff_parse_num( parser, data++ );
      if ( **data == 30 )
        FT_TRACE1(( "cff_parse_cid_ros: real supplement is rounded\n" ));
      dict->cid_supplement = cff_parse_num( parser, data );
      if ( dict->cid_supplement < 0 )
        FT_TRACE1(( "cff_parse_cid_ros: negative supplement %d is found\n",
                   dict->cid_supplement ));
      error = FT_Err_Ok;

      FT_TRACE4(( " %d %d %d\n",
                  dict->cid_registry,
                  dict->cid_ordering,
                  dict->cid_supplement ));
    }

    return error;
  }


  static FT_Error
  cff_parse_vsindex( CFF_Parser  parser )
  {
    /* vsindex operator can only be used in a Private DICT */
    CFF_Private  priv = (CFF_Private)parser->object;
    FT_Byte**    data = parser->stack;
    CFF_Blend    blend;
    FT_Error     error;


    if ( !priv || !priv->subfont )
    {
      error = FT_THROW( Invalid_File_Format );
      goto Exit;
    }

    blend = &priv->subfont->blend;

    if ( blend->usedBV )
    {
      FT_ERROR(( " cff_parse_vsindex: vsindex not allowed after blend\n" ));
      error = FT_THROW( Syntax_Error );
      goto Exit;
    }

    priv->vsindex = (FT_UInt)cff_parse_num( parser, data++ );

    FT_TRACE4(( " %d\n", priv->vsindex ));

    error = FT_Err_Ok;

  Exit:
    return error;
  }


  static FT_Error
  cff_parse_blend( CFF_Parser  parser )
  {
    /* blend operator can only be used in a Private DICT */
    CFF_Private  priv = (CFF_Private)parser->object;
    CFF_SubFont  subFont;
    CFF_Blend    blend;
    FT_UInt      numBlends;
    FT_Error     error;


    if ( !priv || !priv->subfont )
    {
      error = FT_THROW( Invalid_File_Format );
      goto Exit;
    }

    subFont = priv->subfont;
    blend   = &subFont->blend;

    if ( cff_blend_check_vector( blend,
                                 priv->vsindex,
                                 subFont->lenNDV,
                                 subFont->NDV ) )
    {
      error = cff_blend_build_vector( blend,
                                      priv->vsindex,
                                      subFont->lenNDV,
                                      subFont->NDV );
      if ( error )
        goto Exit;
    }

    numBlends = (FT_UInt)cff_parse_num( parser, parser->top - 1 );
    if ( numBlends > parser->stackSize )
    {
      FT_ERROR(( "cff_parse_blend: Invalid number of blends\n" ));
      error = FT_THROW( Invalid_File_Format );
      goto Exit;
    }

    FT_TRACE4(( "   %d value%s blended\n",
                numBlends,
                numBlends == 1 ? "" : "s" ));

    error = cff_blend_doBlend( subFont, parser, numBlends );

    blend->usedBV = TRUE;

  Exit:
    return error;
  }


  /* maxstack operator increases parser and operand stacks for CFF2 */
  static FT_Error
  cff_parse_maxstack( CFF_Parser  parser )
  {
    /* maxstack operator can only be used in a Top DICT */
    CFF_FontRecDict  dict  = (CFF_FontRecDict)parser->object;
    FT_Byte**        data  = parser->stack;
    FT_Error         error = FT_Err_Ok;


    if ( !dict )
    {
      error = FT_THROW( Invalid_File_Format );
      goto Exit;
    }

    dict->maxstack = (FT_UInt)cff_parse_num( parser, data++ );
    if ( dict->maxstack > CFF2_MAX_STACK )
      dict->maxstack = CFF2_MAX_STACK;
    if ( dict->maxstack < CFF2_DEFAULT_STACK )
      dict->maxstack = CFF2_DEFAULT_STACK;

    FT_TRACE4(( " %d\n", dict->maxstack ));

  Exit:
    return error;
  }


#define CFF_FIELD_NUM( code, name, id )             \
          CFF_FIELD( code, name, id, cff_kind_num )
#define CFF_FIELD_FIXED( code, name, id )             \
          CFF_FIELD( code, name, id, cff_kind_fixed )
#define CFF_FIELD_FIXED_1000( code, name, id )                 \
          CFF_FIELD( code, name, id, cff_kind_fixed_thousand )
#define CFF_FIELD_STRING( code, name, id )             \
          CFF_FIELD( code, name, id, cff_kind_string )
#define CFF_FIELD_BOOL( code, name, id )             \
          CFF_FIELD( code, name, id, cff_kind_bool )


#undef  CFF_FIELD
#undef  CFF_FIELD_DELTA


#ifndef FT_DEBUG_LEVEL_TRACE


#define CFF_FIELD_CALLBACK( code, name, id ) \
          {                                  \
            cff_kind_callback,               \
            code | CFFCODE,                  \
            0, 0,                            \
            cff_parse_ ## name,              \
            0, 0                             \
          },

#define CFF_FIELD_BLEND( code, id ) \
          {                         \
            cff_kind_blend,         \
            code | CFFCODE,         \
            0, 0,                   \
            cff_parse_blend,        \
            0, 0                    \
          },

#define CFF_FIELD( code, name, id, kind ) \
          {                               \
            kind,                         \
            code | CFFCODE,               \
            FT_FIELD_OFFSET( name ),      \
            FT_FIELD_SIZE( name ),        \
            0, 0, 0                       \
          },

#define CFF_FIELD_DELTA( code, name, max, id ) \
          {                                    \
            cff_kind_delta,                    \
            code | CFFCODE,                    \
            FT_FIELD_OFFSET( name ),           \
            FT_FIELD_SIZE_DELTA( name ),       \
            0,                                 \
            max,                               \
            FT_FIELD_OFFSET( num_ ## name )    \
          },

  static const CFF_Field_Handler  cff_field_handlers[] =
  {

#include "cfftoken.h"

    { 0, 0, 0, 0, 0, 0, 0 }
  };


#else /* FT_DEBUG_LEVEL_TRACE */



#define CFF_FIELD_CALLBACK( code, name, id ) \
          {                                  \
            cff_kind_callback,               \
            code | CFFCODE,                  \
            0, 0,                            \
            cff_parse_ ## name,              \
            0, 0,                            \
            id                               \
          },

#define CFF_FIELD_BLEND( code, id ) \
          {                         \
            cff_kind_blend,         \
            code | CFFCODE,         \
            0, 0,                   \
            cff_parse_blend,        \
            0, 0,                   \
            id                      \
          },

#define CFF_FIELD( code, name, id, kind ) \
          {                               \
            kind,                         \
            code | CFFCODE,               \
            FT_FIELD_OFFSET( name ),      \
            FT_FIELD_SIZE( name ),        \
            0, 0, 0,                      \
            id                            \
          },

#define CFF_FIELD_DELTA( code, name, max, id ) \
          {                                    \
            cff_kind_delta,                    \
            code | CFFCODE,                    \
            FT_FIELD_OFFSET( name ),           \
            FT_FIELD_SIZE_DELTA( name ),       \
            0,                                 \
            max,                               \
            FT_FIELD_OFFSET( num_ ## name ),   \
            id                                 \
          },

  static const CFF_Field_Handler  cff_field_handlers[] =
  {

#include "cfftoken.h"

    { 0, 0, 0, 0, 0, 0, 0, 0 }
  };


#endif /* FT_DEBUG_LEVEL_TRACE */


  FT_LOCAL_DEF( FT_Error )
  cff_parser_run( CFF_Parser  parser,
                  FT_Byte*    start,
                  FT_Byte*    limit )
  {
    FT_Byte*  p     = start;
    FT_Error  error = FT_Err_Ok;

#ifdef CFF_CONFIG_OPTION_OLD_ENGINE
    PSAux_Service  psaux;

    FT_Library  library = parser->library;
    FT_Memory   memory  = library->memory;
#endif

    parser->top    = parser->stack;
    parser->start  = start;
    parser->limit  = limit;
    parser->cursor = start;

    while ( p < limit )
    {
      FT_UInt  v = *p;


      /* Opcode 31 is legacy MM T2 operator, not a number.      */
      /* Opcode 255 is reserved and should not appear in fonts; */
      /* it is used internally for CFF2 blends.                 */
      if ( v >= 27 && v != 31 && v != 255 )
      {
        /* it's a number; we will push its position on the stack */
        if ( (FT_UInt)( parser->top - parser->stack ) >= parser->stackSize )
          goto Stack_Overflow;

        *parser->top++ = p;

        /* now, skip it */
        if ( v == 30 )
        {
          /* skip real number */
          p++;
          for (;;)
          {
            /* An unterminated floating point number at the */
            /* end of a dictionary is invalid but harmless. */
            if ( p >= limit )
              goto Exit;
            v = p[0] >> 4;
            if ( v == 15 )
              break;
            v = p[0] & 0xF;
            if ( v == 15 )
              break;
            p++;
          }
        }
        else if ( v == 28 )
          p += 2;
        else if ( v == 29 )
          p += 4;
        else if ( v > 246 )
          p += 1;
      }
#ifdef CFF_CONFIG_OPTION_OLD_ENGINE
      else if ( v == 31 )
      {
        /* a Type 2 charstring */

        CFF_Decoder  decoder;
        CFF_FontRec  cff_rec;
        FT_Byte*     charstring_base;
        FT_ULong     charstring_len;

        FT_Fixed*     stack;
        FT_ListNode   node;
        CFF_T2_String t2;
        size_t        t2_size;
        FT_Byte*      q;


        charstring_base = ++p;

        /* search `endchar' operator */
        for (;;)
        {
          if ( p >= limit )
            goto Exit;
          if ( *p == 14 )
            break;
          p++;
        }

        charstring_len = (FT_ULong)( p - charstring_base ) + 1;

        /* construct CFF_Decoder object */
        FT_ZERO( &decoder );
        FT_ZERO( &cff_rec );

        cff_rec.top_font.font_dict.num_designs = parser->num_designs;
        cff_rec.top_font.font_dict.num_axes    = parser->num_axes;
        decoder.cff                            = &cff_rec;

        psaux = (PSAux_Service)FT_Get_Module_Interface( library, "psaux" );
        if ( !psaux )
        {
          FT_ERROR(( "cff_parser_run: cannot access `psaux' module\n" ));
          error = FT_THROW( Missing_Module );
          goto Exit;
        }

        error = psaux->cff_decoder_funcs->parse_charstrings_old(
                  &decoder, charstring_base, charstring_len, 1 );
        if ( error )
          goto Exit;

        /* Now copy the stack data in the temporary decoder object,    */
        /* converting it back to charstring number representations     */
        /* (this is ugly, I know).                                     */

        node = (FT_ListNode)memory->alloc( memory,
                                           sizeof ( FT_ListNodeRec ) );
        if ( !node )
          goto Out_Of_Memory_Error;

        FT_List_Add( &parser->t2_strings, node );

        t2 = (CFF_T2_String)memory->alloc( memory,
                                           sizeof ( CFF_T2_StringRec ) );
        if ( !t2 )
          goto Out_Of_Memory_Error;

        node->data = t2;

        /* `5' is the conservative upper bound of required bytes per stack */
        /* element.                                                        */

        t2_size = 5 * ( decoder.top - decoder.stack );

        q = (FT_Byte*)memory->alloc( memory, t2_size );
        if ( !q )
          goto Out_Of_Memory_Error;

        t2->start = q;
        t2->limit = q + t2_size;

        stack = decoder.stack;

        while ( stack < decoder.top )
        {
          FT_ULong  num;
          FT_Bool   neg;


          if ( (FT_UInt)( parser->top - parser->stack ) >= parser->stackSize )
            goto Stack_Overflow;

          *parser->top++ = q;

          if ( *stack < 0 )
          {
            num = (FT_ULong)NEG_LONG( *stack );
            neg = 1;
          }
          else
          {
            num = (FT_ULong)*stack;
            neg = 0;
          }

          if ( num & 0xFFFFU )
          {
            if ( neg )
              num = (FT_ULong)-num;

            *q++ = 255;
            *q++ = ( num & 0xFF000000U ) >> 24;
            *q++ = ( num & 0x00FF0000U ) >> 16;
            *q++ = ( num & 0x0000FF00U ) >>  8;
            *q++ =   num & 0x000000FFU;
          }
          else
          {
            num >>= 16;

            if ( neg )
            {
              if ( num <= 107 )
                *q++ = (FT_Byte)( 139 - num );
              else if ( num <= 1131 )
              {
                *q++ = (FT_Byte)( ( ( num - 108 ) >> 8 ) + 251 );
                *q++ = (FT_Byte)( ( num - 108 ) & 0xFF );
              }
              else
              {
                num = (FT_ULong)-num;

                *q++ = 28;
                *q++ = (FT_Byte)( num >> 8 );
                *q++ = (FT_Byte)( num & 0xFF );
              }
            }
            else
            {
              if ( num <= 107 )
                *q++ = (FT_Byte)( num + 139 );
              else if ( num <= 1131 )
              {
                *q++ = (FT_Byte)( ( ( num - 108 ) >> 8 ) + 247 );
                *q++ = (FT_Byte)( ( num - 108 ) & 0xFF );
              }
              else
              {
                *q++ = 28;
                *q++ = (FT_Byte)( num >> 8 );
                *q++ = (FT_Byte)( num & 0xFF );
              }
            }
          }

          stack++;
        }
      }
#endif /* CFF_CONFIG_OPTION_OLD_ENGINE */
      else
      {
        /* This is not a number, hence it's an operator.  Compute its code */
        /* and look for it in our current list.                            */

        FT_UInt                   code;
        FT_UInt                   num_args;
        const CFF_Field_Handler*  field;


        if ( (FT_UInt)( parser->top - parser->stack ) >= parser->stackSize )
          goto Stack_Overflow;

        num_args     = (FT_UInt)( parser->top - parser->stack );
        *parser->top = p;
        code         = v;

        if ( v == 12 )
        {
          /* two byte operator */
          p++;
          if ( p >= limit )
            goto Syntax_Error;

          code = 0x100 | p[0];
        }
        code = code | parser->object_code;

        for ( field = cff_field_handlers; field->kind; field++ )
        {
          if ( field->code == (FT_Int)code )
          {
            /* we found our field's handler; read it */
            FT_Long   val;
            FT_Byte*  q = (FT_Byte*)parser->object + field->offset;


#ifdef FT_DEBUG_LEVEL_TRACE
            FT_TRACE4(( "  %s", field->id ));
#endif

            /* check that we have enough arguments -- except for */
            /* delta encoded arrays, which can be empty          */
            if ( field->kind != cff_kind_delta && num_args < 1 )
              goto Stack_Underflow;

            switch ( field->kind )
            {
            case cff_kind_bool:
            case cff_kind_string:
            case cff_kind_num:
              val = cff_parse_num( parser, parser->stack );
              goto Store_Number;

            case cff_kind_fixed:
              val = cff_parse_fixed( parser, parser->stack );
              goto Store_Number;

            case cff_kind_fixed_thousand:
              val = cff_parse_fixed_scaled( parser, parser->stack, 3 );

            Store_Number:
              switch ( field->size )
              {
              case (8 / FT_CHAR_BIT):
                *(FT_Byte*)q = (FT_Byte)val;
                break;

              case (16 / FT_CHAR_BIT):
                *(FT_Short*)q = (FT_Short)val;
                break;

              case (32 / FT_CHAR_BIT):
                *(FT_Int32*)q = (FT_Int)val;
                break;

              default:  /* for 64-bit systems */
                *(FT_Long*)q = val;
              }

#ifdef FT_DEBUG_LEVEL_TRACE
              switch ( field->kind )
              {
              case cff_kind_bool:
                FT_TRACE4(( " %s\n", val ? "true" : "false" ));
                break;

              case cff_kind_string:
                FT_TRACE4(( " %ld (SID)\n", val ));
                break;

              case cff_kind_num:
                FT_TRACE4(( " %ld\n", val ));
                break;

              case cff_kind_fixed:
                FT_TRACE4(( " %f\n", (double)val / 65536 ));
                break;

              case cff_kind_fixed_thousand:
                FT_TRACE4(( " %f\n", (double)val / 65536 / 1000 ));

              default:
                ; /* never reached */
              }
#endif

              break;

            case cff_kind_delta:
              {
                FT_Byte*   qcount = (FT_Byte*)parser->object +
                                      field->count_offset;

                FT_Byte**  data = parser->stack;


                if ( num_args > field->array_max )
                  num_args = field->array_max;

                FT_TRACE4(( " [" ));

                /* store count */
                *qcount = (FT_Byte)num_args;

                val = 0;
                while ( num_args > 0 )
                {
                  val = ADD_LONG( val, cff_parse_num( parser, data++ ) );
                  switch ( field->size )
                  {
                  case (8 / FT_CHAR_BIT):
                    *(FT_Byte*)q = (FT_Byte)val;
                    break;

                  case (16 / FT_CHAR_BIT):
                    *(FT_Short*)q = (FT_Short)val;
                    break;

                  case (32 / FT_CHAR_BIT):
                    *(FT_Int32*)q = (FT_Int)val;
                    break;

                  default:  /* for 64-bit systems */
                    *(FT_Long*)q = val;
                  }

                  FT_TRACE4(( " %ld", val ));

                  q += field->size;
                  num_args--;
                }

                FT_TRACE4(( "]\n" ));
              }
              break;

            default:  /* callback or blend */
              error = field->reader( parser );
              if ( error )
                goto Exit;
            }
            goto Found;
          }
        }

        /* this is an unknown operator, or it is unsupported; */
        /* we will ignore it for now.                         */

      Found:
        /* clear stack */
        /* TODO: could clear blend stack here,       */
        /*       but we don't have access to subFont */
        if ( field->kind != cff_kind_blend )
          parser->top = parser->stack;
      }
      p++;
    } /* while ( p < limit ) */

  Exit:
    return error;

#ifdef CFF_CONFIG_OPTION_OLD_ENGINE
  Out_Of_Memory_Error:
    error = FT_THROW( Out_Of_Memory );
    goto Exit;
#endif

  Stack_Overflow:
    error = FT_THROW( Invalid_Argument );
    goto Exit;

  Stack_Underflow:
    error = FT_THROW( Invalid_Argument );
    goto Exit;

  Syntax_Error:
    error = FT_THROW( Invalid_Argument );
    goto Exit;
  }


/* END */
