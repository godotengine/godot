/****************************************************************************
 *
 * sfwoff2.c
 *
 *   WOFF2 format management (base).
 *
 * Copyright (C) 2019-2020 by
 * Nikhil Ramakrishnan, David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */

#include <ft2build.h>
#include "sfwoff2.h"
#include "woff2tags.h"
#include FT_TRUETYPE_TAGS_H
#include FT_INTERNAL_DEBUG_H
#include FT_INTERNAL_STREAM_H


#ifdef FT_CONFIG_OPTION_USE_BROTLI

#include <brotli/decode.h>

#endif


  /**************************************************************************
   *
   * The macro FT_COMPONENT is used in trace mode.  It is an implicit
   * parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log
   * messages during execution.
   */
#undef  FT_COMPONENT
#define FT_COMPONENT  sfwoff2


#define READ_255USHORT( var )  FT_SET_ERROR( Read255UShort( stream, &var ) )

#define READ_BASE128( var )    FT_SET_ERROR( ReadBase128( stream, &var ) )

#define ROUND4( var )          ( ( var + 3 ) & ~3 )

#define WRITE_USHORT( p, v )                \
          do                                \
          {                                 \
            *(p)++ = (FT_Byte)( (v) >> 8 ); \
            *(p)++ = (FT_Byte)( (v) >> 0 ); \
                                            \
          } while ( 0 )

#define WRITE_ULONG( p, v )                  \
          do                                 \
          {                                  \
            *(p)++ = (FT_Byte)( (v) >> 24 ); \
            *(p)++ = (FT_Byte)( (v) >> 16 ); \
            *(p)++ = (FT_Byte)( (v) >>  8 ); \
            *(p)++ = (FT_Byte)( (v) >>  0 ); \
                                             \
          } while ( 0 )

#define WRITE_SHORT( p, v )        \
          do                       \
          {                        \
            *(p)++ = ( (v) >> 8 ); \
            *(p)++ = ( (v) >> 0 ); \
                                   \
          } while ( 0 )

#define WRITE_SFNT_BUF( buf, s ) \
          write_buf( &sfnt, sfnt_size, &dest_offset, buf, s, memory )

#define WRITE_SFNT_BUF_AT( offset, buf, s ) \
          write_buf( &sfnt, sfnt_size, &offset, buf, s, memory )

#define N_CONTOUR_STREAM    0
#define N_POINTS_STREAM     1
#define FLAG_STREAM         2
#define GLYPH_STREAM        3
#define COMPOSITE_STREAM    4
#define BBOX_STREAM         5
#define INSTRUCTION_STREAM  6


  static void
  stream_close( FT_Stream  stream )
  {
    FT_Memory  memory = stream->memory;


    FT_FREE( stream->base );

    stream->size  = 0;
    stream->base  = NULL;
    stream->close = NULL;
  }


  FT_CALLBACK_DEF( int )
  compare_tags( const void*  a,
                const void*  b )
  {
    WOFF2_Table  table1 = *(WOFF2_Table*)a;
    WOFF2_Table  table2 = *(WOFF2_Table*)b;

    FT_ULong  tag1 = table1->Tag;
    FT_ULong  tag2 = table2->Tag;


    if ( tag1 > tag2 )
      return 1;
    else if ( tag1 < tag2 )
      return -1;
    else
      return 0;
  }


  static FT_Error
  Read255UShort( FT_Stream   stream,
                 FT_UShort*  value )
  {
    static const FT_Int  oneMoreByteCode1 = 255;
    static const FT_Int  oneMoreByteCode2 = 254;
    static const FT_Int  wordCode         = 253;
    static const FT_Int  lowestUCode      = 253;

    FT_Error   error        = FT_Err_Ok;
    FT_Byte    code;
    FT_Byte    result_byte  = 0;
    FT_UShort  result_short = 0;


    if ( FT_READ_BYTE( code ) )
      return error;
    if ( code == wordCode )
    {
      /* Read next two bytes and store `FT_UShort' value. */
      if ( FT_READ_USHORT( result_short ) )
        return error;
      *value = result_short;
      return FT_Err_Ok;
    }
    else if ( code == oneMoreByteCode1 )
    {
      if ( FT_READ_BYTE( result_byte ) )
        return error;
      *value = result_byte + lowestUCode;
      return FT_Err_Ok;
    }
    else if ( code == oneMoreByteCode2 )
    {
      if ( FT_READ_BYTE( result_byte ) )
        return error;
      *value = result_byte + lowestUCode * 2;
      return FT_Err_Ok;
    }
    else
    {
      *value = code;
      return FT_Err_Ok;
    }
  }


  static FT_Error
  ReadBase128( FT_Stream  stream,
               FT_ULong*  value )
  {
    FT_ULong  result = 0;
    FT_Int    i;
    FT_Byte   code;
    FT_Error  error  = FT_Err_Ok;


    for ( i = 0; i < 5; ++i )
    {
      code = 0;
      if ( FT_READ_BYTE( code ) )
        return error;

      /* Leading zeros are invalid. */
      if ( i == 0 && code == 0x80 )
        return FT_THROW( Invalid_Table );

      /* If any of top seven bits are set then we're about to overflow. */
      if ( result & 0xfe000000 )
        return FT_THROW( Invalid_Table );

      result = ( result << 7 ) | ( code & 0x7f );

      /* Spin until most significant bit of data byte is false. */
      if ( ( code & 0x80 ) == 0 )
      {
        *value = result;
        return FT_Err_Ok;
      }
    }

    /* Make sure not to exceed the size bound. */
    return FT_THROW( Invalid_Table );
  }


  /* Extend memory of `dst_bytes' buffer and copy data from `src'. */
  static FT_Error
  write_buf( FT_Byte**  dst_bytes,
             FT_ULong*  dst_size,
             FT_ULong*  offset,
             FT_Byte*   src,
             FT_ULong   size,
             FT_Memory  memory )
  {
    FT_Error  error = FT_Err_Ok;
    /* We are reallocating memory for `dst', so its pointer may change. */
    FT_Byte*  dst   = *dst_bytes;


    /* Check whether we are within limits. */
    if ( ( *offset + size ) > WOFF2_DEFAULT_MAX_SIZE  )
      return FT_THROW( Array_Too_Large );

    /* Reallocate `dst'. */
    if ( ( *offset + size ) > *dst_size )
    {
      FT_TRACE6(( "Reallocating %lu to %lu.\n",
                  *dst_size, (*offset + size) ));
      if ( FT_REALLOC( dst,
                       (FT_ULong)( *dst_size ),
                       (FT_ULong)( *offset + size ) ) )
        goto Exit;

      *dst_size = *offset + size;
    }

    /* Copy data. */
    ft_memcpy( dst + *offset, src, size );

    *offset += size;
    /* Set pointer of `dst' to its correct value. */
    *dst_bytes = dst;

  Exit:
    return error;
  }


  /* Pad buffer to closest multiple of 4. */
  static FT_Error
  pad4( FT_Byte**  sfnt_bytes,
        FT_ULong*  sfnt_size,
        FT_ULong*  out_offset,
        FT_Memory  memory )
  {
    FT_Byte*  sfnt        = *sfnt_bytes;
    FT_ULong  dest_offset = *out_offset;

    FT_Byte   zeroes[] = { 0, 0, 0 };
    FT_ULong  pad_bytes;


    if ( dest_offset + 3 < dest_offset )
      return FT_THROW( Invalid_Table );

    pad_bytes = ROUND4( dest_offset ) - dest_offset;
    if ( pad_bytes > 0 )
    {
      if ( WRITE_SFNT_BUF( &zeroes[0], pad_bytes ) )
        return FT_THROW( Invalid_Table );
    }

    *sfnt_bytes = sfnt;
    *out_offset = dest_offset;
    return FT_Err_Ok;
  }


  /* Calculate table checksum of `buf'. */
  static FT_Long
  compute_ULong_sum( FT_Byte*  buf,
                     FT_ULong  size )
  {
    FT_ULong  checksum     = 0;
    FT_ULong  aligned_size = size & ~3;
    FT_ULong  i;
    FT_ULong  v;


    for ( i = 0; i < aligned_size; i += 4 )
      checksum += ( (FT_ULong)buf[i    ] << 24 ) |
                  ( (FT_ULong)buf[i + 1] << 16 ) |
                  ( (FT_ULong)buf[i + 2] <<  8 ) |
                  ( (FT_ULong)buf[i + 3] <<  0 );

    /* If size is not aligned to 4, treat as if it is padded with 0s. */
    if ( size != aligned_size )
    {
      v = 0;
      for ( i = aligned_size ; i < size; ++i )
        v |= (FT_ULong)buf[i] << ( 24 - 8 * ( i & 3 ) );
      checksum += v;
    }

    return checksum;
  }


  static FT_Error
  woff2_decompress( FT_Byte*        dst,
                    FT_ULong        dst_size,
                    const FT_Byte*  src,
                    FT_ULong        src_size )
  {
#ifdef FT_CONFIG_OPTION_USE_BROTLI

    FT_ULong             uncompressed_size = dst_size;
    BrotliDecoderResult  result;


    result = BrotliDecoderDecompress( src_size,
                                      src,
                                      &uncompressed_size,
                                      dst );

    if ( result != BROTLI_DECODER_RESULT_SUCCESS ||
         uncompressed_size != dst_size           )
    {
      FT_ERROR(( "woff2_decompress: Stream length mismatch.\n" ));
      return FT_THROW( Invalid_Table );
    }

    FT_TRACE2(( "woff2_decompress: Brotli stream decompressed.\n" ));
    return FT_Err_Ok;

#else /* !FT_CONFIG_OPTION_USE_BROTLI */

    FT_ERROR(( "woff2_decompress: Brotli support not available.\n" ));
    return FT_THROW( Unimplemented_Feature );

#endif /* !FT_CONFIG_OPTION_USE_BROTLI */
  }


  static WOFF2_Table
  find_table( WOFF2_Table*  tables,
              FT_UShort     num_tables,
              FT_ULong      tag )
  {
    FT_Int  i;


    for ( i = 0; i < num_tables; i++ )
    {
      if ( tables[i]->Tag == tag )
        return tables[i];
    }
    return NULL;
  }


  /* Read `numberOfHMetrics' field from `hhea' table. */
  static FT_Error
  read_num_hmetrics( FT_Stream   stream,
                     FT_UShort*  num_hmetrics )
  {
    FT_Error   error = FT_Err_Ok;
    FT_UShort  num_metrics;


    if ( FT_STREAM_SKIP( 34 )  )
      return FT_THROW( Invalid_Table );

    if ( FT_READ_USHORT( num_metrics ) )
      return FT_THROW( Invalid_Table );

    *num_hmetrics = num_metrics;

    return error;
  }


  /* An auxiliary function for overflow-safe addition. */
  static FT_Int
  with_sign( FT_Byte  flag,
             FT_Int   base_val )
  {
    /* Precondition: 0 <= base_val < 65536 (to avoid overflow). */
    return ( flag & 1 ) ? base_val : -base_val;
  }


  /* An auxiliary function for overflow-safe addition. */
  static FT_Int
  safe_int_addition( FT_Int   a,
                     FT_Int   b,
                     FT_Int*  result )
  {
    if ( ( ( a > 0 ) && ( b > FT_INT_MAX - a ) ) ||
         ( ( a < 0 ) && ( b < FT_INT_MIN - a ) ) )
      return FT_THROW( Invalid_Table );

    *result = a + b;
    return FT_Err_Ok;
  }


  /*
   * Decode variable-length (flag, xCoordinate, yCoordinate) triplet for a
   * simple glyph.  See
   *
   *   https://www.w3.org/TR/WOFF2/#triplet_decoding
   */
  static FT_Error
  triplet_decode( const FT_Byte*  flags_in,
                  const FT_Byte*  in,
                  FT_ULong        in_size,
                  FT_ULong        n_points,
                  WOFF2_Point     result,
                  FT_ULong*       in_bytes_used )
  {
    FT_Int  x = 0;
    FT_Int  y = 0;
    FT_Int  dx;
    FT_Int  dy;
    FT_Int  b0, b1, b2;

    FT_ULong  triplet_index = 0;
    FT_ULong  data_bytes;

    FT_UInt  i;


    if ( n_points > in_size )
      return FT_THROW( Invalid_Table );

    for ( i = 0; i < n_points; ++i )
    {
      FT_Byte  flag     = flags_in[i];
      FT_Bool  on_curve = !( flag >> 7 );


      flag &= 0x7f;
      if ( flag < 84 )
        data_bytes = 1;
      else if ( flag < 120 )
        data_bytes = 2;
      else if ( flag < 124 )
        data_bytes = 3;
      else
        data_bytes = 4;

      /* Overflow checks */
      if ( triplet_index + data_bytes > in_size       ||
           triplet_index + data_bytes < triplet_index )
        return FT_THROW( Invalid_Table );

      if ( flag < 10 )
      {
        dx = 0;
        dy = with_sign( flag,
                        ( ( flag & 14 ) << 7 ) + in[triplet_index] );
      }
      else if ( flag < 20 )
      {
        dx = with_sign( flag,
                        ( ( ( flag - 10 ) & 14 ) << 7 ) +
                          in[triplet_index] );
        dy = 0;
      }
      else if ( flag < 84 )
      {
        b0 = flag - 20;
        b1 = in[triplet_index];
        dx = with_sign( flag,
                        1 + ( b0 & 0x30 ) + ( b1 >> 4 ) );
        dy = with_sign( flag >> 1,
                        1 + ( ( b0 & 0x0c ) << 2 ) + ( b1 & 0x0f ) );
      }
      else if ( flag < 120 )
      {
        b0 = flag - 84;
        dx = with_sign( flag,
                        1 + ( ( b0 / 12 ) << 8 ) + in[triplet_index] );
        dy = with_sign( flag >> 1,
                        1 + ( ( ( b0 % 12 ) >> 2 ) << 8 ) +
                          in[triplet_index + 1] );
      }
      else if ( flag < 124 )
      {
        b2 = in[triplet_index + 1];
        dx = with_sign( flag,
                        ( in[triplet_index] << 4 ) + ( b2 >> 4 ) );
        dy = with_sign( flag >> 1,
                        ( ( b2 & 0x0f ) << 8 ) + in[triplet_index + 2] );
      }
      else
      {
        dx = with_sign( flag,
                        ( in[triplet_index] << 8 ) +
                          in[triplet_index + 1] );
        dy = with_sign( flag >> 1,
                        ( in[triplet_index + 2] << 8 ) +
                          in[triplet_index + 3] );
      }

      triplet_index += data_bytes;

      if ( safe_int_addition( x, dx, &x ) )
        return FT_THROW( Invalid_Table );

      if ( safe_int_addition( y, dy, &y ) )
        return FT_THROW( Invalid_Table );

      result[i].x        = x;
      result[i].y        = y;
      result[i].on_curve = on_curve;
    }

    *in_bytes_used = triplet_index;
    return FT_Err_Ok;
  }


  /* Store decoded points in glyph buffer. */
  static FT_Error
  store_points( FT_ULong           n_points,
                const WOFF2_Point  points,
                FT_UShort          n_contours,
                FT_UShort          instruction_len,
                FT_Byte*           dst,
                FT_ULong           dst_size,
                FT_ULong*          glyph_size )
  {
    FT_UInt   flag_offset  = 10 + ( 2 * n_contours ) + 2 + instruction_len;
    FT_Int    last_flag    = -1;
    FT_Int    repeat_count =  0;
    FT_Int    last_x       =  0;
    FT_Int    last_y       =  0;
    FT_UInt   x_bytes      =  0;
    FT_UInt   y_bytes      =  0;
    FT_UInt   xy_bytes;
    FT_UInt   i;
    FT_UInt   x_offset;
    FT_UInt   y_offset;
    FT_Byte*  pointer;


    for ( i = 0; i < n_points; ++i )
    {
      const WOFF2_PointRec  point = points[i];

      FT_Int  flag = point.on_curve ? GLYF_ON_CURVE : 0;
      FT_Int  dx   = point.x - last_x;
      FT_Int  dy   = point.y - last_y;


      if ( dx == 0 )
        flag |= GLYF_THIS_X_IS_SAME;
      else if ( dx > -256 && dx < 256 )
      {
        flag |= GLYF_X_SHORT | ( dx > 0 ? GLYF_THIS_X_IS_SAME : 0 );
        x_bytes += 1;
      }
      else
        x_bytes += 2;

      if ( dy == 0 )
        flag |= GLYF_THIS_Y_IS_SAME;
      else if ( dy > -256 && dy < 256 )
      {
        flag |= GLYF_Y_SHORT | ( dy > 0 ? GLYF_THIS_Y_IS_SAME : 0 );
        y_bytes += 1;
      }
      else
        y_bytes += 2;

      if ( flag == last_flag && repeat_count != 255 )
      {
        dst[flag_offset - 1] |= GLYF_REPEAT;
        repeat_count++;
      }
      else
      {
        if ( repeat_count != 0 )
        {
          if ( flag_offset >= dst_size )
            return FT_THROW( Invalid_Table );

          dst[flag_offset++] = repeat_count;
        }
        if ( flag_offset >= dst_size )
          return FT_THROW( Invalid_Table );

        dst[flag_offset++] = flag;
        repeat_count       = 0;
      }

      last_x    = point.x;
      last_y    = point.y;
      last_flag = flag;
    }

    if ( repeat_count != 0 )
    {
      if ( flag_offset >= dst_size )
        return FT_THROW( Invalid_Table );

      dst[flag_offset++] = repeat_count;
    }

    xy_bytes = x_bytes + y_bytes;
    if ( xy_bytes < x_bytes                   ||
         flag_offset + xy_bytes < flag_offset ||
         flag_offset + xy_bytes > dst_size    )
      return FT_THROW( Invalid_Table );

    x_offset = flag_offset;
    y_offset = flag_offset + x_bytes;
    last_x = 0;
    last_y = 0;

    for ( i = 0; i < n_points; ++i )
    {
      FT_Int  dx = points[i].x - last_x;
      FT_Int  dy = points[i].y - last_y;


      if ( dx == 0 )
        ;
      else if ( dx > -256 && dx < 256 )
        dst[x_offset++] = FT_ABS( dx );
      else
      {
        pointer = dst + x_offset;
        WRITE_SHORT( pointer, dx );
        x_offset += 2;
      }

      last_x += dx;

      if ( dy == 0 )
        ;
      else if ( dy > -256 && dy < 256 )
        dst[y_offset++] = FT_ABS( dy );
      else
      {
        pointer = dst + y_offset;
        WRITE_SHORT( pointer, dy );
        y_offset += 2;
      }

      last_y += dy;
    }

    *glyph_size = y_offset;
    return FT_Err_Ok;
  }


  static void
  compute_bbox( FT_ULong           n_points,
                const WOFF2_Point  points,
                FT_Byte*           dst,
                FT_UShort*         src_x_min )
  {
    FT_Int  x_min = 0;
    FT_Int  y_min = 0;
    FT_Int  x_max = 0;
    FT_Int  y_max = 0;

    FT_UInt  i;

    FT_ULong  offset;
    FT_Byte*  pointer;


    if ( n_points > 0 )
    {
      x_min = points[0].x;
      y_min = points[0].y;
      x_max = points[0].x;
      y_max = points[0].y;
    }

    for ( i = 1; i < n_points; ++i )
    {
      FT_Int  x = points[i].x;
      FT_Int  y = points[i].y;


      x_min = FT_MIN( x, x_min );
      y_min = FT_MIN( y, y_min );
      x_max = FT_MAX( x, x_max );
      y_max = FT_MAX( y, y_max );
    }

    /* Write values to `glyf' record. */
    offset  = 2;
    pointer = dst + offset;

    WRITE_SHORT( pointer, x_min );
    WRITE_SHORT( pointer, y_min );
    WRITE_SHORT( pointer, x_max );
    WRITE_SHORT( pointer, y_max );

    *src_x_min = (FT_UShort)x_min;
  }


  static FT_Error
  compositeGlyph_size( FT_Stream  stream,
                       FT_ULong   offset,
                       FT_ULong*  size,
                       FT_Bool*   have_instructions )
  {
    FT_Error   error        = FT_Err_Ok;
    FT_ULong   start_offset = offset;
    FT_Bool    we_have_inst = FALSE;
    FT_UShort  flags        = FLAG_MORE_COMPONENTS;


    if ( FT_STREAM_SEEK( start_offset ) )
      goto Exit;
    while ( flags & FLAG_MORE_COMPONENTS )
    {
      FT_ULong  arg_size;


      if ( FT_READ_USHORT( flags ) )
        goto Exit;
      we_have_inst |= ( flags & FLAG_WE_HAVE_INSTRUCTIONS ) != 0;
      /* glyph index */
      arg_size = 2;
      if ( flags & FLAG_ARG_1_AND_2_ARE_WORDS )
        arg_size += 4;
      else
        arg_size += 2;

      if ( flags & FLAG_WE_HAVE_A_SCALE )
        arg_size += 2;
      else if ( flags & FLAG_WE_HAVE_AN_X_AND_Y_SCALE )
        arg_size += 4;
      else if ( flags & FLAG_WE_HAVE_A_TWO_BY_TWO )
        arg_size += 8;

      if ( FT_STREAM_SKIP( arg_size ) )
        goto Exit;
    }

    *size              = FT_STREAM_POS() - start_offset;
    *have_instructions = we_have_inst;

  Exit:
    return error;
  }


  /* Store loca values (provided by `reconstruct_glyf') to output stream. */
  static FT_Error
  store_loca( FT_ULong*  loca_values,
              FT_ULong   loca_values_size,
              FT_UShort  index_format,
              FT_ULong*  checksum,
              FT_Byte**  sfnt_bytes,
              FT_ULong*  sfnt_size,
              FT_ULong*  out_offset,
              FT_Memory  memory )
  {
    FT_Error  error       = FT_Err_Ok;
    FT_Byte*  sfnt        = *sfnt_bytes;
    FT_ULong  dest_offset = *out_offset;

    FT_Byte*  loca_buf = NULL;
    FT_Byte*  dst      = NULL;

    FT_UInt   i = 0;
    FT_ULong  loca_buf_size;

    const FT_ULong  offset_size = index_format ? 4 : 2;


    if ( ( loca_values_size << 2 ) >> 2 != loca_values_size )
      goto Fail;

    loca_buf_size = loca_values_size * offset_size;
    if ( FT_NEW_ARRAY( loca_buf, loca_buf_size ) )
      goto Fail;

    dst = loca_buf;
    for ( i = 0; i < loca_values_size; i++ )
    {
      FT_ULong  value = loca_values[i];


      if ( index_format )
        WRITE_ULONG( dst, value );
      else
        WRITE_USHORT( dst, ( value >> 1 ) );
    }

    *checksum = compute_ULong_sum( loca_buf, loca_buf_size );
    /* Write `loca' table to sfnt buffer. */
    if ( WRITE_SFNT_BUF( loca_buf, loca_buf_size ) )
      goto Fail;

    /* Set pointer `sfnt_bytes' to its correct value. */
    *sfnt_bytes = sfnt;
    *out_offset = dest_offset;

    FT_FREE( loca_buf );
    return error;

  Fail:
    if ( !error )
      error = FT_THROW( Invalid_Table );

    FT_FREE( loca_buf );

    return error;
  }


  static FT_Error
  reconstruct_glyf( FT_Stream    stream,
                    FT_ULong*    glyf_checksum,
                    FT_ULong*    loca_checksum,
                    FT_Byte**    sfnt_bytes,
                    FT_ULong*    sfnt_size,
                    FT_ULong*    out_offset,
                    WOFF2_Info   info,
                    FT_Memory    memory )
  {
    FT_Error  error = FT_Err_Ok;
    FT_Byte*  sfnt  = *sfnt_bytes;

    /* current position in stream */
    const FT_ULong  pos = FT_STREAM_POS();

    FT_UInt  num_substreams = 7;

    FT_UShort  num_glyphs;
    FT_UShort  index_format;
    FT_ULong   expected_loca_length;
    FT_UInt    offset;
    FT_UInt    i;
    FT_ULong   points_size;
    FT_ULong   bitmap_length;
    FT_ULong   glyph_buf_size;
    FT_ULong   bbox_bitmap_offset;

    const FT_ULong  glyf_start  = *out_offset;
    FT_ULong        dest_offset = *out_offset;

    WOFF2_Substream  substreams = NULL;

    FT_ULong*    loca_values  = NULL;
    FT_UShort*   n_points_arr = NULL;
    FT_Byte*     glyph_buf    = NULL;
    WOFF2_Point  points       = NULL;


    if ( FT_NEW_ARRAY( substreams, num_substreams ) )
      goto Fail;

    if ( FT_STREAM_SKIP( 4 ) )
      goto Fail;
    if ( FT_READ_USHORT( num_glyphs ) )
      goto Fail;
    if ( FT_READ_USHORT( index_format ) )
      goto Fail;

    FT_TRACE4(( "num_glyphs = %u; index_format = %u\n",
                num_glyphs, index_format ));

    info->num_glyphs = num_glyphs;

    /* Calculate expected length of loca and compare.          */
    /* See https://www.w3.org/TR/WOFF2/#conform-mustRejectLoca */
    /* index_format = 0 => Short version `loca'.               */
    /* index_format = 1 => Long version `loca'.                */
    expected_loca_length = ( index_format ? 4 : 2 ) *
                             ( (FT_ULong)num_glyphs + 1 );
    if ( info->loca_table->dst_length != expected_loca_length )
      goto Fail;

    offset = ( 2 + num_substreams ) * 4;
    if ( offset > info->glyf_table->TransformLength )
      goto Fail;

    for ( i = 0; i < num_substreams; ++i )
    {
      FT_ULong  substream_size;


      if ( FT_READ_ULONG( substream_size ) )
        goto Fail;
      if ( substream_size > info->glyf_table->TransformLength - offset )
        goto Fail;

      substreams[i].start  = pos + offset;
      substreams[i].offset = pos + offset;
      substreams[i].size   = substream_size;

      FT_TRACE5(( "  Substream %d: offset = %lu; size = %lu;\n",
                  i, substreams[i].offset, substreams[i].size ));
      offset += substream_size;
    }

    if ( FT_NEW_ARRAY( loca_values, num_glyphs + 1 ) )
      goto Fail;

    points_size        = 0;
    bbox_bitmap_offset = substreams[BBOX_STREAM].offset;

    /* Size of bboxBitmap = 4 * floor((numGlyphs + 31) / 32) */
    bitmap_length                   = ( ( num_glyphs + 31 ) >> 5 ) << 2;
    substreams[BBOX_STREAM].offset += bitmap_length;

    glyph_buf_size = WOFF2_DEFAULT_GLYPH_BUF;
    if ( FT_NEW_ARRAY( glyph_buf, glyph_buf_size ) )
      goto Fail;

    if ( FT_NEW_ARRAY( info->x_mins, num_glyphs ) )
      goto Fail;

    for ( i = 0; i < num_glyphs; ++i )
    {
      FT_ULong   glyph_size = 0;
      FT_UShort  n_contours = 0;
      FT_Bool    have_bbox  = FALSE;
      FT_Byte    bbox_bitmap;
      FT_ULong   bbox_offset;
      FT_UShort  x_min      = 0;


      /* Set `have_bbox'. */
      bbox_offset = bbox_bitmap_offset + ( i >> 3 );
      if ( FT_STREAM_SEEK( bbox_offset ) ||
           FT_READ_BYTE( bbox_bitmap )   )
        goto Fail;
      if ( bbox_bitmap & ( 0x80 >> ( i & 7 ) ) )
        have_bbox = TRUE;

      /* Read value from `nContourStream'. */
      if ( FT_STREAM_SEEK( substreams[N_CONTOUR_STREAM].offset ) ||
           FT_READ_USHORT( n_contours )                          )
        goto Fail;
      substreams[N_CONTOUR_STREAM].offset += 2;

      if ( n_contours == 0xffff )
      {
        /* composite glyph */
        FT_Bool    have_instructions = FALSE;
        FT_UShort  instruction_size  = 0;
        FT_ULong   composite_size;
        FT_ULong   size_needed;
        FT_Byte*   pointer           = NULL;


        /* Composite glyphs must have explicit bbox. */
        if ( !have_bbox )
          goto Fail;

        if ( compositeGlyph_size( stream,
                                  substreams[COMPOSITE_STREAM].offset,
                                  &composite_size,
                                  &have_instructions) )
          goto Fail;

        if ( have_instructions )
        {
          if ( FT_STREAM_SEEK( substreams[GLYPH_STREAM].offset ) ||
               READ_255USHORT( instruction_size )                )
            goto Fail;
          substreams[GLYPH_STREAM].offset = FT_STREAM_POS();
        }

        size_needed = 12 + composite_size + instruction_size;
        if ( glyph_buf_size < size_needed )
        {
          if ( FT_RENEW_ARRAY( glyph_buf, glyph_buf_size, size_needed ) )
            goto Fail;
          glyph_buf_size = size_needed;
        }

        pointer = glyph_buf + glyph_size;
        WRITE_USHORT( pointer, n_contours );
        glyph_size += 2;

        /* Read x_min for current glyph. */
        if ( FT_STREAM_SEEK( substreams[BBOX_STREAM].offset ) ||
             FT_READ_USHORT( x_min )                          )
          goto Fail;
        /* No increment here because we read again. */

        if ( FT_STREAM_SEEK( substreams[BBOX_STREAM].offset ) ||
             FT_STREAM_READ( glyph_buf + glyph_size, 8 )      )
          goto Fail;

        substreams[BBOX_STREAM].offset += 8;
        glyph_size                     += 8;

        if ( FT_STREAM_SEEK( substreams[COMPOSITE_STREAM].offset )    ||
             FT_STREAM_READ( glyph_buf + glyph_size, composite_size ) )
          goto Fail;

        substreams[COMPOSITE_STREAM].offset += composite_size;
        glyph_size                          += composite_size;

        if ( have_instructions )
        {
          pointer = glyph_buf + glyph_size;
          WRITE_USHORT( pointer, instruction_size );
          glyph_size += 2;

          if ( FT_STREAM_SEEK( substreams[INSTRUCTION_STREAM].offset )    ||
               FT_STREAM_READ( glyph_buf + glyph_size, instruction_size ) )
            goto Fail;

          substreams[INSTRUCTION_STREAM].offset += instruction_size;
          glyph_size                            += instruction_size;
        }
      }
      else if ( n_contours > 0 )
      {
        /* simple glyph */
        FT_ULong   total_n_points = 0;
        FT_UShort  n_points_contour;
        FT_UInt    j;
        FT_ULong   flag_size;
        FT_ULong   triplet_size;
        FT_ULong   triplet_bytes_used;
        FT_Byte*   flags_buf   = NULL;
        FT_Byte*   triplet_buf = NULL;
        FT_UShort  instruction_size;
        FT_ULong   size_needed;
        FT_Int     end_point;
        FT_UInt    contour_ix;

        FT_Byte*   pointer = NULL;


        if ( FT_NEW_ARRAY( n_points_arr, n_contours ) )
          goto Fail;

        if ( FT_STREAM_SEEK( substreams[N_POINTS_STREAM].offset ) )
          goto Fail;

        for ( j = 0; j < n_contours; ++j )
        {
          if ( READ_255USHORT( n_points_contour ) )
            goto Fail;
          n_points_arr[j] = n_points_contour;
          /* Prevent negative/overflow. */
          if ( total_n_points + n_points_contour < total_n_points )
            goto Fail;
          total_n_points += n_points_contour;
        }
        substreams[N_POINTS_STREAM].offset = FT_STREAM_POS();

        flag_size = total_n_points;
        if ( flag_size > substreams[FLAG_STREAM].size )
          goto Fail;

        flags_buf   = stream->base + substreams[FLAG_STREAM].offset;
        triplet_buf = stream->base + substreams[GLYPH_STREAM].offset;

        if ( substreams[GLYPH_STREAM].size <
               ( substreams[GLYPH_STREAM].offset -
                 substreams[GLYPH_STREAM].start ) )
          goto Fail;

        triplet_size       = substreams[GLYPH_STREAM].size -
                               ( substreams[GLYPH_STREAM].offset -
                                 substreams[GLYPH_STREAM].start );
        triplet_bytes_used = 0;

        /* Create array to store point information. */
        points_size = total_n_points;
        if ( FT_NEW_ARRAY( points, points_size ) )
          goto Fail;

        if ( triplet_decode( flags_buf,
                             triplet_buf,
                             triplet_size,
                             total_n_points,
                             points,
                             &triplet_bytes_used ) )
          goto Fail;

        substreams[FLAG_STREAM].offset  += flag_size;
        substreams[GLYPH_STREAM].offset += triplet_bytes_used;

        if ( FT_STREAM_SEEK( substreams[GLYPH_STREAM].offset ) ||
             READ_255USHORT( instruction_size )                )
          goto Fail;

        substreams[GLYPH_STREAM].offset = FT_STREAM_POS();

        if ( total_n_points >= ( 1 << 27 ) )
          goto Fail;

        size_needed = 12 +
                      ( 2 * n_contours ) +
                      ( 5 * total_n_points ) +
                      instruction_size;
        if ( glyph_buf_size < size_needed )
        {
          if ( FT_RENEW_ARRAY( glyph_buf, glyph_buf_size, size_needed ) )
            goto Fail;
          glyph_buf_size = size_needed;
        }

        pointer = glyph_buf + glyph_size;
        WRITE_USHORT( pointer, n_contours );
        glyph_size += 2;

        if ( have_bbox )
        {
          /* Read x_min for current glyph. */
          if ( FT_STREAM_SEEK( substreams[BBOX_STREAM].offset ) ||
               FT_READ_USHORT( x_min )                          )
            goto Fail;
          /* No increment here because we read again. */

          if ( FT_STREAM_SEEK( substreams[BBOX_STREAM].offset ) ||
               FT_STREAM_READ( glyph_buf + glyph_size, 8 )      )
            goto Fail;
          substreams[BBOX_STREAM].offset += 8;
        }
        else
          compute_bbox( total_n_points, points, glyph_buf, &x_min );

        glyph_size = CONTOUR_OFFSET_END_POINT;

        pointer   = glyph_buf + glyph_size;
        end_point = -1;

        for ( contour_ix = 0; contour_ix < n_contours; ++contour_ix )
        {
          end_point += n_points_arr[contour_ix];
          if ( end_point >= 65536 )
            goto Fail;

          WRITE_SHORT( pointer, end_point );
          glyph_size += 2;
        }

        WRITE_USHORT( pointer, instruction_size );
        glyph_size += 2;

        if ( FT_STREAM_SEEK( substreams[INSTRUCTION_STREAM].offset )    ||
             FT_STREAM_READ( glyph_buf + glyph_size, instruction_size ) )
          goto Fail;

        substreams[INSTRUCTION_STREAM].offset += instruction_size;
        glyph_size                            += instruction_size;

        if ( store_points( total_n_points,
                           points,
                           n_contours,
                           instruction_size,
                           glyph_buf,
                           glyph_buf_size,
                           &glyph_size ) )
          goto Fail;

        FT_FREE( points );
        FT_FREE( n_points_arr );
      }
      else
      {
        /* Empty glyph.          */
        /* Must not have a bbox. */
        if ( have_bbox )
        {
          FT_ERROR(( "Empty glyph has a bbox.\n" ));
          goto Fail;
        }
      }

      loca_values[i] = dest_offset - glyf_start;

      if ( WRITE_SFNT_BUF( glyph_buf, glyph_size ) )
        goto Fail;

      if ( pad4( &sfnt, sfnt_size, &dest_offset, memory ) )
        goto Fail;

      *glyf_checksum += compute_ULong_sum( glyph_buf, glyph_size );

      /* Store x_mins, may be required to reconstruct `hmtx'. */
      if ( n_contours > 0 )
        info->x_mins[i] = x_min;
    }

    info->glyf_table->dst_length = dest_offset - info->glyf_table->dst_offset;
    info->loca_table->dst_offset = dest_offset;

    /* `loca[n]' will be equal to the length of the `glyf' table. */
    loca_values[num_glyphs] = info->glyf_table->dst_length;

    if ( store_loca( loca_values,
                     num_glyphs + 1,
                     index_format,
                     loca_checksum,
                     &sfnt,
                     sfnt_size,
                     &dest_offset,
                     memory ) )
      goto Fail;

    info->loca_table->dst_length = dest_offset - info->loca_table->dst_offset;

    FT_TRACE4(( "  loca table info:\n" ));
    FT_TRACE4(( "    dst_offset = %lu\n", info->loca_table->dst_offset ));
    FT_TRACE4(( "    dst_length = %lu\n", info->loca_table->dst_length ));
    FT_TRACE4(( "    checksum = %09x\n", *loca_checksum ));

    /* Set pointer `sfnt_bytes' to its correct value. */
    *sfnt_bytes = sfnt;
    *out_offset = dest_offset;

    FT_FREE( substreams );
    FT_FREE( loca_values );
    FT_FREE( n_points_arr );
    FT_FREE( glyph_buf );
    FT_FREE( points );

    return error;

  Fail:
    if ( !error )
      error = FT_THROW( Invalid_Table );

    /* Set pointer `sfnt_bytes' to its correct value. */
    *sfnt_bytes = sfnt;

    FT_FREE( substreams );
    FT_FREE( loca_values );
    FT_FREE( n_points_arr );
    FT_FREE( glyph_buf );
    FT_FREE( points );

    return error;
  }


  /* Get `x_mins' for untransformed `glyf' table. */
  static FT_Error
  get_x_mins( FT_Stream     stream,
              WOFF2_Table*  tables,
              FT_UShort     num_tables,
              WOFF2_Info    info,
              FT_Memory     memory )
  {
    FT_UShort  num_glyphs;
    FT_UShort  index_format;
    FT_ULong   glyf_offset;
    FT_UShort  glyf_offset_short;
    FT_ULong   loca_offset;
    FT_Int     i;
    FT_Error   error = FT_Err_Ok;
    FT_ULong   offset_size;

    /* At this point of time those tables might not have been read yet. */
    const WOFF2_Table  maxp_table = find_table( tables, num_tables,
                                                TTAG_maxp );
    const WOFF2_Table  head_table = find_table( tables, num_tables,
                                                TTAG_head );


    if ( !maxp_table )
    {
      FT_ERROR(( "`maxp' table is missing.\n" ));
      return FT_THROW( Invalid_Table );
    }

    if ( !head_table )
    {
      FT_ERROR(( "`head' table is missing.\n" ));
      return FT_THROW( Invalid_Table );
    }

    /* Read `numGlyphs' field from `maxp' table. */
    if ( FT_STREAM_SEEK( maxp_table->src_offset ) || FT_STREAM_SKIP( 8 ) )
      return error;

    if ( FT_READ_USHORT( num_glyphs ) )
      return error;

    info->num_glyphs = num_glyphs;

    /* Read `indexToLocFormat' field from `head' table. */
    if ( FT_STREAM_SEEK( head_table->src_offset ) ||
         FT_STREAM_SKIP( 50 )                     )
      return error;

    if ( FT_READ_USHORT( index_format ) )
      return error;

    offset_size = index_format ? 4 : 2;

    /* Create `x_mins' array. */
    if ( FT_NEW_ARRAY( info->x_mins, num_glyphs ) )
      return error;

    loca_offset = info->loca_table->src_offset;

    for ( i = 0; i < num_glyphs; ++i )
    {
      if ( FT_STREAM_SEEK( loca_offset ) )
        return error;

      loca_offset += offset_size;

      if ( index_format )
      {
        if ( FT_READ_ULONG( glyf_offset ) )
          return error;
      }
      else
      {
        if ( FT_READ_USHORT( glyf_offset_short ) )
          return error;

        glyf_offset = (FT_ULong)( glyf_offset_short );
        glyf_offset = glyf_offset << 1;
      }

      glyf_offset += info->glyf_table->src_offset;

      if ( FT_STREAM_SEEK( glyf_offset ) || FT_STREAM_SKIP( 2 ) )
        return error;

      if ( FT_READ_USHORT( info->x_mins[i] ) )
        return error;
    }

    return error;
  }


  static FT_Error
  reconstruct_hmtx( FT_Stream  stream,
                    FT_UShort  num_glyphs,
                    FT_UShort  num_hmetrics,
                    FT_Short*  x_mins,
                    FT_ULong*  checksum,
                    FT_Byte**  sfnt_bytes,
                    FT_ULong*  sfnt_size,
                    FT_ULong*  out_offset,
                    FT_Memory  memory )
  {
    FT_Error  error       = FT_Err_Ok;
    FT_Byte*  sfnt        = *sfnt_bytes;
    FT_ULong  dest_offset = *out_offset;

    FT_Byte   hmtx_flags;
    FT_Bool   has_proportional_lsbs, has_monospace_lsbs;
    FT_ULong  hmtx_table_size;
    FT_Int    i;

    FT_UShort*  advance_widths = NULL;
    FT_Short*   lsbs           = NULL;
    FT_Byte*    hmtx_table     = NULL;
    FT_Byte*    dst            = NULL;


    if ( FT_READ_BYTE( hmtx_flags ) )
      goto Fail;

    has_proportional_lsbs = ( hmtx_flags & 1 ) == 0;
    has_monospace_lsbs    = ( hmtx_flags & 2 ) == 0;

    /* Bits 2-7 are reserved and MUST be zero. */
    if ( ( hmtx_flags & 0xFC ) != 0 )
      goto Fail;

    /* Are you REALLY transformed? */
    if ( has_proportional_lsbs && has_monospace_lsbs )
      goto Fail;

    /* Cannot have a transformed `hmtx' without `glyf'. */
    if ( ( num_hmetrics > num_glyphs ) ||
         ( num_hmetrics < 1 )          )
      goto Fail;

    /* Must have at least one entry. */
    if ( num_hmetrics < 1 )
      goto Fail;

    if ( FT_NEW_ARRAY( advance_widths, num_hmetrics ) ||
         FT_NEW_ARRAY( lsbs, num_glyphs )             )
      goto Fail;

    /* Read `advanceWidth' stream.  Always present. */
    for ( i = 0; i < num_hmetrics; i++ )
    {
      FT_UShort  advance_width;


      if ( FT_READ_USHORT( advance_width ) )
        goto Fail;

      advance_widths[i] = advance_width;
    }

    /* lsb values for proportional glyphs. */
    for ( i = 0; i < num_hmetrics; i++ )
    {
      FT_Short  lsb;


      if ( has_proportional_lsbs )
      {
        if ( FT_READ_SHORT( lsb ) )
          goto Fail;
      }
      else
        lsb = x_mins[i];

      lsbs[i] = lsb;
    }

    /* lsb values for monospaced glyphs. */
    for ( i = num_hmetrics; i < num_glyphs; i++ )
    {
      FT_Short  lsb;


      if ( has_monospace_lsbs )
      {
        if ( FT_READ_SHORT( lsb ) )
          goto Fail;
      }
      else
        lsb = x_mins[i];

      lsbs[i] = lsb;
    }

    /* Build the hmtx table. */
    hmtx_table_size = 2 * num_hmetrics + 2 * num_glyphs;
    if ( FT_NEW_ARRAY( hmtx_table, hmtx_table_size ) )
      goto Fail;

    dst = hmtx_table;
    FT_TRACE6(( "hmtx values: \n" ));
    for ( i = 0; i < num_glyphs; i++ )
    {
      if ( i < num_hmetrics )
      {
        WRITE_SHORT( dst, advance_widths[i] );
        FT_TRACE6(( "%d ", advance_widths[i] ));
      }

      WRITE_SHORT( dst, lsbs[i] );
      FT_TRACE6(( "%d ", lsbs[i] ));
    }
    FT_TRACE6(( "\n" ));

    *checksum = compute_ULong_sum( hmtx_table, hmtx_table_size );
    /* Write `hmtx' table to sfnt buffer. */
    if ( WRITE_SFNT_BUF( hmtx_table, hmtx_table_size ) )
      goto Fail;

    /* Set pointer `sfnt_bytes' to its correct value. */
    *sfnt_bytes = sfnt;
    *out_offset = dest_offset;

    FT_FREE( advance_widths );
    FT_FREE( lsbs );
    FT_FREE( hmtx_table );

    return error;

  Fail:
    FT_FREE( advance_widths );
    FT_FREE( lsbs );
    FT_FREE( hmtx_table );

    if ( !error )
      error = FT_THROW( Invalid_Table );

    return error;
  }


  static FT_Error
  reconstruct_font( FT_Byte*      transformed_buf,
                    FT_ULong      transformed_buf_size,
                    WOFF2_Table*  indices,
                    WOFF2_Header  woff2,
                    WOFF2_Info    info,
                    FT_Byte**     sfnt_bytes,
                    FT_ULong*     sfnt_size,
                    FT_Memory     memory )
  {
    /* Memory management of `transformed_buf' is handled by the caller. */

    FT_Error   error       = FT_Err_Ok;
    FT_Stream  stream      = NULL;
    FT_Byte*   buf_cursor  = NULL;
    FT_Byte*   table_entry = NULL;

    /* We are reallocating memory for `sfnt', so its pointer may change. */
    FT_Byte*   sfnt = *sfnt_bytes;

    FT_UShort  num_tables  = woff2->num_tables;
    FT_ULong   dest_offset = 12 + num_tables * 16UL;

    FT_ULong   checksum      = 0;
    FT_ULong   loca_checksum = 0;
    FT_Int     nn            = 0;
    FT_UShort  num_hmetrics  = 0;
    FT_ULong   font_checksum = info->header_checksum;
    FT_Bool    is_glyf_xform = FALSE;

    FT_ULong  table_entry_offset = 12;


    /* A few table checks before reconstruction. */
    /* `glyf' must be present with `loca'.       */
    info->glyf_table = find_table( indices, num_tables, TTAG_glyf );
    info->loca_table = find_table( indices, num_tables, TTAG_loca );

    if ( ( info->glyf_table == NULL ) ^ ( info->loca_table == NULL ) )
    {
      FT_ERROR(( "One of `glyf'/`loca' tables missing.\n" ));
      return FT_THROW( Invalid_Table );
    }

    /* Both `glyf' and `loca' must have same transformation. */
    if ( info->glyf_table != NULL )
    {
      if ( ( info->glyf_table->flags & WOFF2_FLAGS_TRANSFORM ) !=
           ( info->loca_table->flags & WOFF2_FLAGS_TRANSFORM ) )
      {
        FT_ERROR(( "Transformation mismatch"
                   " between `glyf' and `loca' table." ));
        return FT_THROW( Invalid_Table );
      }
    }

    /* Create buffer for table entries. */
    if ( FT_NEW_ARRAY( table_entry, 16 ) )
      goto Fail;

    /* Create a stream for the uncompressed buffer. */
    if ( FT_NEW( stream ) )
      goto Fail;
    FT_Stream_OpenMemory( stream, transformed_buf, transformed_buf_size );

    FT_ASSERT( FT_STREAM_POS() == 0 );

    /* Reconstruct/copy tables to output stream. */
    for ( nn = 0; nn < num_tables; nn++ )
    {
      WOFF2_TableRec  table = *( indices[nn] );


      FT_TRACE3(( "Seeking to %d with table size %d.\n",
                  table.src_offset, table.src_length ));
      FT_TRACE3(( "Table tag: %c%c%c%c.\n",
                  (FT_Char)( table.Tag >> 24 ),
                  (FT_Char)( table.Tag >> 16 ),
                  (FT_Char)( table.Tag >> 8  ),
                  (FT_Char)( table.Tag       ) ));

      if ( FT_STREAM_SEEK( table.src_offset ) )
        goto Fail;

      if ( table.src_offset + table.src_length > transformed_buf_size )
        goto Fail;

      /* Get stream size for fields of `hmtx' table. */
      if ( table.Tag == TTAG_hhea )
      {
        if ( read_num_hmetrics( stream, &num_hmetrics ) )
          goto Fail;
      }

      info->num_hmetrics = num_hmetrics;

      checksum = 0;
      if ( ( table.flags & WOFF2_FLAGS_TRANSFORM ) != WOFF2_FLAGS_TRANSFORM )
      {
        /* Check whether `head' is at least 12 bytes. */
        if ( table.Tag == TTAG_head )
        {
          if ( table.src_length < 12 )
            goto Fail;

          buf_cursor = transformed_buf + table.src_offset + 8;
          /* Set checkSumAdjustment = 0 */
          WRITE_ULONG( buf_cursor, 0 );
        }

        table.dst_offset = dest_offset;

        checksum = compute_ULong_sum( transformed_buf + table.src_offset,
                                      table.src_length );
        FT_TRACE4(( "Checksum = %09x.\n", checksum ));

        if ( WRITE_SFNT_BUF( transformed_buf + table.src_offset,
                             table.src_length ) )
          goto Fail;
      }
      else
      {
        FT_TRACE3(( "This table is transformed.\n" ));

        if ( table.Tag == TTAG_glyf )
        {
          is_glyf_xform    = TRUE;
          table.dst_offset = dest_offset;

          if ( reconstruct_glyf( stream,
                                 &checksum,
                                 &loca_checksum,
                                 &sfnt,
                                 sfnt_size,
                                 &dest_offset,
                                 info,
                                 memory ) )
            goto Fail;

          FT_TRACE4(( "Checksum = %09x.\n", checksum ));
        }

        else if ( table.Tag == TTAG_loca )
          checksum = loca_checksum;

        else if ( table.Tag == TTAG_hmtx )
        {
          /* If glyf is not transformed and hmtx is, handle separately. */
          if ( !is_glyf_xform )
          {
            if ( get_x_mins( stream, indices, num_tables, info, memory ) )
              goto Fail;
          }

          table.dst_offset = dest_offset;

          if ( reconstruct_hmtx( stream,
                                 info->num_glyphs,
                                 info->num_hmetrics,
                                 info->x_mins,
                                 &checksum,
                                 &sfnt,
                                 sfnt_size,
                                 &dest_offset,
                                 memory ) )
            goto Fail;
        }
        else
        {
          /* Unknown transform. */
          FT_ERROR(( "Unknown table transform.\n" ));
          goto Fail;
        }
      }

      font_checksum += checksum;

      buf_cursor = &table_entry[0];
      WRITE_ULONG( buf_cursor, table.Tag );
      WRITE_ULONG( buf_cursor, checksum );
      WRITE_ULONG( buf_cursor, table.dst_offset );
      WRITE_ULONG( buf_cursor, table.dst_length );

      WRITE_SFNT_BUF_AT( table_entry_offset, table_entry, 16 );

      /* Update checksum. */
      font_checksum += compute_ULong_sum( table_entry, 16 );

      if ( pad4( &sfnt, sfnt_size, &dest_offset, memory ) )
        goto Fail;

      /* Sanity check. */
      if ( (FT_ULong)( table.dst_offset + table.dst_length ) > dest_offset )
      {
        FT_ERROR(( "Table was partially written.\n" ));
        goto Fail;
      }
    }

    /* Update `head' checkSumAdjustment. */
    info->head_table = find_table( indices, num_tables, TTAG_head );
    if ( !info->head_table )
    {
      FT_ERROR(( "`head' table is missing.\n" ));
      goto Fail;
    }

    if ( info->head_table->dst_length < 12 )
      goto Fail;

    buf_cursor    = sfnt + info->head_table->dst_offset + 8;
    font_checksum = 0xB1B0AFBA - font_checksum;

    WRITE_ULONG( buf_cursor, font_checksum );

    FT_TRACE2(( "Final checksum = %09x.\n", font_checksum ));

    woff2->actual_sfnt_size = dest_offset;

    /* Set pointer of sfnt stream to its correct value. */
    *sfnt_bytes = sfnt;

    FT_FREE( table_entry );
    FT_Stream_Close( stream );
    FT_FREE( stream );

    return error;

  Fail:
    if ( !error )
      error = FT_THROW( Invalid_Table );

    /* Set pointer of sfnt stream to its correct value. */
    *sfnt_bytes = sfnt;

    FT_FREE( table_entry );
    FT_Stream_Close( stream );
    FT_FREE( stream );

    return error;
  }


  /* Replace `face->root.stream' with a stream containing the extracted */
  /* SFNT of a WOFF2 font.                                              */

  FT_LOCAL_DEF( FT_Error )
  woff2_open_font( FT_Stream  stream,
                   TT_Face    face,
                   FT_Int*    face_instance_index,
                   FT_Long*   num_faces )
  {
    FT_Memory  memory = stream->memory;
    FT_Error   error  = FT_Err_Ok;
    FT_Int     face_index;

    WOFF2_HeaderRec  woff2;
    WOFF2_InfoRec    info         = { 0, 0, 0, NULL, NULL, NULL, NULL };
    WOFF2_Table      tables       = NULL;
    WOFF2_Table*     indices      = NULL;
    WOFF2_Table*     temp_indices = NULL;
    WOFF2_Table      last_table;

    FT_Int     nn;
    FT_ULong   j;
    FT_ULong   flags;
    FT_UShort  xform_version;
    FT_ULong   src_offset = 0;

    FT_UInt    glyf_index;
    FT_UInt    loca_index;
    FT_UInt32  file_offset;

    FT_Byte*   sfnt        = NULL;
    FT_Stream  sfnt_stream = NULL;
    FT_Byte*   sfnt_header;
    FT_ULong   sfnt_size;

    FT_Byte*  uncompressed_buf = NULL;

    static const FT_Frame_Field  woff2_header_fields[] =
    {
#undef  FT_STRUCTURE
#define FT_STRUCTURE  WOFF2_HeaderRec

      FT_FRAME_START( 48 ),
        FT_FRAME_ULONG     ( signature ),
        FT_FRAME_ULONG     ( flavor ),
        FT_FRAME_ULONG     ( length ),
        FT_FRAME_USHORT    ( num_tables ),
        FT_FRAME_SKIP_BYTES( 2 ),
        FT_FRAME_ULONG     ( totalSfntSize ),
        FT_FRAME_ULONG     ( totalCompressedSize ),
        FT_FRAME_SKIP_BYTES( 2 * 2 ),
        FT_FRAME_ULONG     ( metaOffset ),
        FT_FRAME_ULONG     ( metaLength ),
        FT_FRAME_ULONG     ( metaOrigLength ),
        FT_FRAME_ULONG     ( privOffset ),
        FT_FRAME_ULONG     ( privLength ),
      FT_FRAME_END
    };


    FT_ASSERT( stream == face->root.stream );
    FT_ASSERT( FT_STREAM_POS() == 0 );

    face_index = FT_ABS( *face_instance_index ) & 0xFFFF;

    /* Read WOFF2 Header. */
    if ( FT_STREAM_READ_FIELDS( woff2_header_fields, &woff2 ) )
      return error;

    FT_TRACE4(( "signature     -> 0x%X\n", woff2.signature ));
    FT_TRACE2(( "flavor        -> 0x%08lx\n", woff2.flavor ));
    FT_TRACE4(( "length        -> %lu\n", woff2.length ));
    FT_TRACE2(( "num_tables    -> %hu\n", woff2.num_tables ));
    FT_TRACE4(( "totalSfntSize -> %lu\n", woff2.totalSfntSize ));
    FT_TRACE4(( "metaOffset    -> %hu\n", woff2.metaOffset ));
    FT_TRACE4(( "metaLength    -> %hu\n", woff2.metaLength ));
    FT_TRACE4(( "privOffset    -> %hu\n", woff2.privOffset ));
    FT_TRACE4(( "privLength    -> %hu\n", woff2.privLength ));

    /* Make sure we don't recurse back here. */
    if ( woff2.flavor == TTAG_wOF2 )
      return FT_THROW( Invalid_Table );

    /* Miscellaneous checks. */
    if ( woff2.length != stream->size                               ||
         woff2.num_tables == 0                                      ||
         48 + woff2.num_tables * 20UL >= woff2.length               ||
         ( woff2.metaOffset == 0 && ( woff2.metaLength != 0     ||
                                      woff2.metaOrigLength != 0 ) ) ||
         ( woff2.metaLength != 0 && woff2.metaOrigLength == 0 )     ||
         ( woff2.metaOffset >= woff2.length )                       ||
         ( woff2.length - woff2.metaOffset < woff2.metaLength )     ||
         ( woff2.privOffset == 0 && woff2.privLength != 0 )         ||
         ( woff2.privOffset >= woff2.length )                       ||
         ( woff2.length - woff2.privOffset < woff2.privLength )     )
    {
      FT_ERROR(( "woff2_open_font: invalid WOFF2 header\n" ));
      return FT_THROW( Invalid_Table );
    }

    FT_TRACE2(( "woff2_open_font: WOFF2 Header is valid.\n" ));

    woff2.ttc_fonts = NULL;

    /* Read table directory. */
    if ( FT_NEW_ARRAY( tables, woff2.num_tables )  ||
         FT_NEW_ARRAY( indices, woff2.num_tables ) )
      goto Exit;

    FT_TRACE2(( "\n"
                "  tag    flags    transform   origLen   transformLen\n"
                "  --------------------------------------------------\n" ));

    for ( nn = 0; nn < woff2.num_tables; nn++ )
    {
      WOFF2_Table  table = tables + nn;


      if ( FT_READ_BYTE( table->FlagByte ) )
        goto Exit;

      if ( ( table->FlagByte & 0x3f ) == 0x3f )
      {
        if ( FT_READ_ULONG( table->Tag ) )
          goto Exit;
      }
      else
      {
        table->Tag = woff2_known_tags( table->FlagByte & 0x3f );
        if ( !table->Tag )
        {
          FT_ERROR(( "woff2_open_font: Unknown table tag." ));
          error = FT_THROW( Invalid_Table );
          goto Exit;
        }
      }

      flags = 0;
      xform_version = ( table->FlagByte >> 6 ) & 0x03;

      /* 0 means xform for glyph/loca, non-0 for others. */
      if ( table->Tag == TTAG_glyf || table->Tag == TTAG_loca )
      {
        if ( xform_version == 0 )
          flags |= WOFF2_FLAGS_TRANSFORM;
      }
      else if ( xform_version != 0 )
        flags |= WOFF2_FLAGS_TRANSFORM;

      flags |= xform_version;

      if ( READ_BASE128( table->dst_length ) )
        goto Exit;

      table->TransformLength = table->dst_length;

      if ( ( flags & WOFF2_FLAGS_TRANSFORM ) != 0 )
      {
        if ( READ_BASE128( table->TransformLength ) )
          goto Exit;

        if ( table->Tag == TTAG_loca && table->TransformLength )
        {
          FT_ERROR(( "woff2_open_font: Invalid loca `transformLength'.\n" ));
          error = FT_THROW( Invalid_Table );
          goto Exit;
        }
      }

      if ( src_offset + table->TransformLength < src_offset )
      {
        FT_ERROR(( "woff2_open_font: invalid WOFF2 table directory.\n" ));
        error = FT_THROW( Invalid_Table );
        goto Exit;
      }

      table->src_offset = src_offset;
      table->src_length = table->TransformLength;
      src_offset       += table->TransformLength;
      table->flags      = flags;

      FT_TRACE2(( "  %c%c%c%c  %08d  %08d    %08ld  %08ld\n",
                  (FT_Char)( table->Tag >> 24 ),
                  (FT_Char)( table->Tag >> 16 ),
                  (FT_Char)( table->Tag >> 8  ),
                  (FT_Char)( table->Tag       ),
                  table->FlagByte & 0x3f,
                  ( table->FlagByte >> 6 ) & 0x03,
                  table->dst_length,
                  table->TransformLength,
                  table->src_length,
                  table->src_offset ));

      indices[nn] = table;
    }

    /* End of last table is uncompressed size. */
    last_table = indices[woff2.num_tables - 1];

    woff2.uncompressed_size = last_table->src_offset +
                              last_table->src_length;
    if ( woff2.uncompressed_size < last_table->src_offset )
    {
      error = FT_THROW( Invalid_Table );
      goto Exit;
    }

    FT_TRACE2(( "Table directory parsed.\n" ));

    /* Check for and read collection directory. */
    woff2.num_fonts      = 1;
    woff2.header_version = 0;

    if ( woff2.flavor == TTAG_ttcf )
    {
      FT_TRACE2(( "Font is a TTC, reading collection directory.\n" ));

      if ( FT_READ_ULONG( woff2.header_version ) )
        goto Exit;

      if ( woff2.header_version != 0x00010000 &&
           woff2.header_version != 0x00020000 )
      {
        error = FT_THROW( Invalid_Table );
        goto Exit;
      }

      if ( READ_255USHORT( woff2.num_fonts ) )
        goto Exit;

      if ( !woff2.num_fonts )
      {
        error = FT_THROW( Invalid_Table );
        goto Exit;
      }

      FT_TRACE4(( "Number of fonts in TTC: %ld\n", woff2.num_fonts ));

      if ( FT_NEW_ARRAY( woff2.ttc_fonts, woff2.num_fonts ) )
        goto Exit;

      for ( nn = 0; nn < woff2.num_fonts; nn++ )
      {
        WOFF2_TtcFont  ttc_font = woff2.ttc_fonts + nn;


        if ( READ_255USHORT( ttc_font->num_tables ) )
          goto Exit;
        if ( FT_READ_ULONG( ttc_font->flavor ) )
          goto Exit;

        if ( FT_NEW_ARRAY( ttc_font->table_indices, ttc_font->num_tables ) )
          goto Exit;

        FT_TRACE5(( "Number of tables in font %d: %ld\n",
                    nn, ttc_font->num_tables ));

#ifdef FT_DEBUG_LEVEL_TRACE
        if ( ttc_font->num_tables )
          FT_TRACE6(( "  Indices: " ));
#endif

        glyf_index = 0;
        loca_index = 0;

        for ( j = 0; j < ttc_font->num_tables; j++ )
        {
          FT_UShort    table_index;
          WOFF2_Table  table;


          if ( READ_255USHORT( table_index ) )
            goto Exit;

          FT_TRACE6(( "%hu ", table_index ));
          if ( table_index >= woff2.num_tables )
          {
            FT_ERROR(( "woff2_open_font: invalid table index\n" ));
            error = FT_THROW( Invalid_Table );
            goto Exit;
          }

          ttc_font->table_indices[j] = table_index;

          table = indices[table_index];
          if ( table->Tag == TTAG_loca )
            loca_index = table_index;
          if ( table->Tag == TTAG_glyf )
            glyf_index = table_index;
        }

#ifdef FT_DEBUG_LEVEL_TRACE
        if ( ttc_font->num_tables )
          FT_TRACE6(( "\n" ));
#endif

        /* glyf and loca must be consecutive */
        if ( glyf_index > 0 || loca_index > 0 )
        {
          if ( glyf_index > loca_index      ||
               loca_index - glyf_index != 1 )
          {
            error = FT_THROW( Invalid_Table );
            goto Exit;
          }
        }
      }

      /* Collection directory reading complete. */
      FT_TRACE2(( "WOFF2 collection directory is valid.\n" ));
    }
    else
      woff2.ttc_fonts = NULL;

    woff2.compressed_offset = FT_STREAM_POS();
    file_offset             = ROUND4( woff2.compressed_offset +
                                      woff2.totalCompressedSize );

    /* Some more checks before we start reading the tables. */
    if ( file_offset > woff2.length )
    {
      error = FT_THROW( Invalid_Table );
      goto Exit;
    }

    if ( woff2.metaOffset )
    {
      if ( file_offset != woff2.metaOffset )
      {
        error = FT_THROW( Invalid_Table );
        goto Exit;
      }
      file_offset = ROUND4(woff2.metaOffset + woff2.metaLength);
    }

    if ( woff2.privOffset )
    {
      if ( file_offset != woff2.privOffset )
      {
        error = FT_THROW( Invalid_Table );
        goto Exit;
      }
      file_offset = ROUND4(woff2.privOffset + woff2.privLength);
    }

    if ( file_offset != ( ROUND4( woff2.length ) ) )
    {
      error = FT_THROW( Invalid_Table );
      goto Exit;
    }

    /* Validate requested face index. */
    *num_faces = woff2.num_fonts;
    /* value -(N+1) requests information on index N */
    if ( *face_instance_index < 0 )
      face_index--;

    if ( face_index >= woff2.num_fonts )
    {
      if ( *face_instance_index >= 0 )
      {
        error = FT_THROW( Invalid_Argument );
        goto Exit;
      }
      else
        face_index = 0;
    }

    /* Only retain tables of the requested face in a TTC. */
    if ( woff2.header_version )
    {
      WOFF2_TtcFont  ttc_font = woff2.ttc_fonts + face_index;


      /* Create a temporary array. */
      if ( FT_NEW_ARRAY( temp_indices,
                         ttc_font->num_tables ) )
        goto Exit;

      FT_TRACE4(( "Storing tables for TTC face index %d.\n", face_index ));
      for ( nn = 0; nn < ttc_font->num_tables; nn++ )
        temp_indices[nn] = indices[ttc_font->table_indices[nn]];

      /* Resize array to required size. */
      if ( FT_RENEW_ARRAY( indices,
                           woff2.num_tables,
                           ttc_font->num_tables ) )
        goto Exit;

      for ( nn = 0; nn < ttc_font->num_tables; nn++ )
        indices[nn] = temp_indices[nn];

      FT_FREE( temp_indices );

      /* Change header values. */
      woff2.flavor     = ttc_font->flavor;
      woff2.num_tables = ttc_font->num_tables;
    }

    /* We need to allocate this much at the minimum. */
    sfnt_size = 12 + woff2.num_tables * 16UL;
    /* This is what we normally expect.                              */
    /* Initially trust `totalSfntSize' and change later as required. */
    if ( woff2.totalSfntSize > sfnt_size )
    {
      /* However, adjust the value to something reasonable. */

      /* Factor 64 is heuristic. */
      if ( ( woff2.totalSfntSize >> 6 ) > woff2.length )
        sfnt_size = woff2.length << 6;
      else
        sfnt_size = woff2.totalSfntSize;

      /* Value 1<<26 = 67108864 is heuristic. */
      if (sfnt_size >= (1 << 26))
        sfnt_size = 1 << 26;

#ifdef FT_DEBUG_LEVEL_TRACE
      if ( sfnt_size != woff2.totalSfntSize )
        FT_TRACE4(( "adjusting estimate of uncompressed font size"
                    " to %lu bytes\n",
                    sfnt_size ));
#endif
    }

    /* Write sfnt header. */
    if ( FT_ALLOC( sfnt, sfnt_size ) ||
         FT_NEW( sfnt_stream )       )
      goto Exit;

    sfnt_header = sfnt;

    WRITE_ULONG( sfnt_header, woff2.flavor );

    if ( woff2.num_tables )
    {
      FT_UInt  searchRange, entrySelector, rangeShift, x;


      x             = woff2.num_tables;
      entrySelector = 0;
      while ( x )
      {
        x            >>= 1;
        entrySelector += 1;
      }
      entrySelector--;

      searchRange = ( 1 << entrySelector ) * 16;
      rangeShift  = ( woff2.num_tables * 16 ) - searchRange;

      WRITE_USHORT( sfnt_header, woff2.num_tables );
      WRITE_USHORT( sfnt_header, searchRange );
      WRITE_USHORT( sfnt_header, entrySelector );
      WRITE_USHORT( sfnt_header, rangeShift );
    }

    info.header_checksum = compute_ULong_sum( sfnt, 12 );

    /* Sort tables by tag. */
    ft_qsort( indices,
              woff2.num_tables,
              sizeof ( WOFF2_Table ),
              compare_tags );

    if ( woff2.uncompressed_size < 1 )
    {
      error = FT_THROW( Invalid_Table );
      goto Exit;
    }

    if ( woff2.uncompressed_size > sfnt_size )
    {
      FT_ERROR(( "woff2_open_font: SFNT table lengths are too large.\n" ));
      error = FT_THROW( Invalid_Table );
      goto Exit;
    }

    /* Allocate memory for uncompressed table data. */
    if ( FT_ALLOC( uncompressed_buf, woff2.uncompressed_size ) ||
         FT_FRAME_ENTER( woff2.totalCompressedSize )           )
      goto Exit;

    /* Uncompress the stream. */
    error = woff2_decompress( uncompressed_buf,
                              woff2.uncompressed_size,
                              stream->cursor,
                              woff2.totalCompressedSize );

    FT_FRAME_EXIT();

    if ( error )
      goto Exit;

    error = reconstruct_font( uncompressed_buf,
                              woff2.uncompressed_size,
                              indices,
                              &woff2,
                              &info,
                              &sfnt,
                              &sfnt_size,
                              memory );

    if ( error )
      goto Exit;

    /* Resize `sfnt' to actual size of sfnt stream. */
    if ( woff2.actual_sfnt_size < sfnt_size )
    {
      FT_TRACE5(( "Trimming sfnt stream from %lu to %lu.\n",
                  sfnt_size, woff2.actual_sfnt_size ));
      if ( FT_REALLOC( sfnt,
                       (FT_ULong)( sfnt_size ),
                       (FT_ULong)( woff2.actual_sfnt_size ) ) )
        goto Exit;
    }

    /* `reconstruct_font' has done all the work. */
    /* Swap out stream and return.               */
    FT_Stream_OpenMemory( sfnt_stream, sfnt, woff2.actual_sfnt_size );
    sfnt_stream->memory = stream->memory;
    sfnt_stream->close  = stream_close;

    FT_Stream_Free(
      face->root.stream,
      ( face->root.face_flags & FT_FACE_FLAG_EXTERNAL_STREAM ) != 0 );

    face->root.stream      = sfnt_stream;
    face->root.face_flags &= ~FT_FACE_FLAG_EXTERNAL_STREAM;

    /* Set face_index to 0 or -1. */
    if ( *face_instance_index >= 0 )
      *face_instance_index = 0;
    else
      *face_instance_index = -1;

    FT_TRACE2(( "woff2_open_font: SFNT synthesized.\n" ));

  Exit:
    FT_FREE( tables );
    FT_FREE( indices );
    FT_FREE( uncompressed_buf );
    FT_FREE( info.x_mins );

    if ( woff2.ttc_fonts )
    {
      WOFF2_TtcFont  ttc_font = woff2.ttc_fonts;


      for ( nn = 0; nn < woff2.num_fonts; nn++ )
      {
        FT_FREE( ttc_font->table_indices );
        ttc_font++;
      }

      FT_FREE( woff2.ttc_fonts );
    }

    if ( error )
    {
      FT_FREE( sfnt );
      if ( sfnt_stream )
      {
        FT_Stream_Close( sfnt_stream );
        FT_FREE( sfnt_stream );
      }
    }

    return error;
  }


#undef READ_255USHORT
#undef READ_BASE128
#undef ROUND4
#undef WRITE_USHORT
#undef WRITE_ULONG
#undef WRITE_SHORT
#undef WRITE_SFNT_BUF
#undef WRITE_SFNT_BUF_AT

#undef N_CONTOUR_STREAM
#undef N_POINTS_STREAM
#undef FLAG_STREAM
#undef GLYPH_STREAM
#undef COMPOSITE_STREAM
#undef BBOX_STREAM
#undef INSTRUCTION_STREAM


/* END */
