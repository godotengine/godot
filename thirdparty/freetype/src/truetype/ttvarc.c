/****************************************************************************
 *
 * ttvarc.c
 *
 *   TrueType and OpenType VARC (Variable Composites) support (body).
 *
 * Copyright (C) 2025-2026 by
 * Behdad Esfahbod, David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


  /**************************************************************************
   *
   * VARC table specification:
   *
   *   https://github.com/harfbuzz/boring-expansion-spec/blob/main/VARC.md
   *
   */


#include <freetype/internal/ftdebug.h>
#include <freetype/internal/ftstream.h>
#include <freetype/internal/ftcalc.h>
#include <freetype/internal/ftobjs.h>
#include <freetype/tttags.h>
#include <freetype/ftmm.h>
#include <freetype/fttrigon.h>
#include <freetype/ftoutln.h>
#include <ft2build.h>

#ifdef TT_CONFIG_OPTION_VARC

#include "ttvarc.h"

#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
#include "ttgxvar.h"
#endif


  /**************************************************************************
   *
   * The macro FT_COMPONENT is used in trace mode.  It is an implicit
   * parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log
   * messages during execution.
   */
#undef  FT_COMPONENT
#define FT_COMPONENT  ttvarc


  FT_LOCAL_DEF( FT_Error )
  tt_face_init_varc_axes( FT_Face  face,
                          FT_UInt  axis_count )
  {
#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
    return tt_var_init_blend( face, axis_count );
#else
    FT_UNUSED( face );
    FT_UNUSED( axis_count );

    return FT_Err_Ok;
#endif
  }


  /* Macro for bounds checking when reading from table */
#define CHECK_TABLE_BOUNDS( p, size )                     \
          ( (FT_Byte*)(p) >=                              \
              (FT_Byte*)varc->table                    && \
            (FT_Byte*)(p) + (size) <=                     \
              (FT_Byte*)varc->table + varc->table_size )

  /* Stack allocation limits for performance. */
#define VARC_STACK_AXIS_COUNT     64
#define VARC_STACK_DELTA_COUNT    128
#define VARC_STACK_COORD_COUNT    64
#define VARC_STACK_INDICES_COUNT  64


  /**************************************************************************
   *
   * Utility Functions
   *
   */

  /**************************************************************************
   *
   * @Function:
   *   read_uint32var
   *
   * @Description:
   *   Read a variable-length uint32 value (1-5 bytes).
   *   Similar to UTF-8 encoding but for integers.
   *
   * @Input:
   *   p ::
   *     Pointer to the byte stream (will be advanced).
   *
   *   limit ::
   *     Maximum valid address in the stream.
   *
   * @Output:
   *   value ::
   *     The decoded uint32 value.
   *
   * @Return:
   *   FreeType error code.  0 means success.
   */
  static FT_Error
  read_uint32var( FT_Byte**  p,
                  FT_Byte*   limit,
                  FT_UInt32* value )
  {
    FT_Byte    b0;
    FT_UInt32  result;


    if ( *p >= limit )
      return FT_THROW( Invalid_Table );

    b0 = *(*p)++;

    /* single byte: 0xxxxxxx */
    if ( b0 < 0x80 )
    {
      *value = b0;
      return FT_Err_Ok;
    }

    /* two bytes: 10xxxxxx xxxxxxxx */
    if ( b0 < 0xC0 )
    {
      if ( *p >= limit )
        return FT_THROW( Invalid_Table );
      result = ( ( b0 - 0x80 ) << 8 ) |
                 *(*p)++;
      *value = result;
      return FT_Err_Ok;
    }

    /* three bytes: 110xxxxx xxxxxxxx xxxxxxxx */
    if ( b0 < 0xE0 )
    {
      if ( *p + 1 >= limit )
        return FT_THROW( Invalid_Table );
      result = ( ( b0 - 0xC0 ) << 16 ) |
               ( (*p)[0]       << 8  ) |
                 (*p)[1];
      *p += 2;
      *value = result;
      return FT_Err_Ok;
    }

    /* four bytes: 1110xxxx xxxxxxxx xxxxxxxx xxxxxxxx */
    if ( b0 < 0xF0 )
    {
      if ( *p + 2 >= limit )
        return FT_THROW( Invalid_Table );
      result = ( ( b0 - 0xE0 ) << 24 ) |
               ( (*p)[0]       << 16 ) |
               ( (*p)[1]       << 8  ) |
                 (*p)[2];
      *p += 3;
      *value = result;
      return FT_Err_Ok;
    }

    /* five bytes: 11110000 xxxxxxxx xxxxxxxx xxxxxxxx xxxxxxxx */
    if ( b0 == 0xF0 )
    {
      if ( *p + 3 >= limit )
        return FT_THROW( Invalid_Table );
      result = ( (*p)[0] << 24 ) |
               ( (*p)[1] << 16 ) |
               ( (*p)[2] << 8  ) |
                 (*p)[3];
      *p += 4;
      *value = result;
      return FT_Err_Ok;
    }

    /* invalid encoding */
    return FT_THROW( Invalid_Table );
  }


  /**************************************************************************
   *
   * @Function:
   *   read_fword
   *
   * @Description:
   *   Read a signed 16-bit FWORD value.
   *
   * @Input:
   *   p ::
   *     Pointer to the byte stream (will be advanced).
   *
   *   limit ::
   *     Maximum valid address in the stream.
   *
   * @Output:
   *   value ::
   *     The decoded FWORD value as 26.6 fixed-point.
   *
   * @Return:
   *   FreeType error code. 0 means success.
   */
  static FT_Error
  read_fword( FT_Byte**  p,
              FT_Byte*   limit,
              FT_Pos*    value )
  {
    FT_Short  val;


    if ( *p + 1 >= limit )
      return FT_THROW( Invalid_Table );

    val = (FT_Short)( ( (FT_UShort)(*p)[0] << 8 ) |
                        (FT_UShort)(*p)[1]        );
    *p += 2;

    /* FWORD is a signed 16-bit integer,         */
    /* shift left by 6 for FT_Pos (26.6 format). */
    *value = (FT_Pos)val * 64;

    return FT_Err_Ok;
  }


  /**************************************************************************
   *
   * @Function:
   *   read_f4dot12
   *
   * @Description:
   *   Read a signed F4DOT12 value (4.12 fixed-point).
   *
   * @Input:
   *   p ::
   *     Pointer to the byte stream (will be advanced).
   *
   *   limit ::
   *     Maximum valid address in the stream.
   *
   * @Output:
   *   value ::
   *     The decoded value as 16.16 fixed-point.
   *
   * @Return:
   *   FreeType error code. 0 means success.
   */
  static FT_Error
  read_f4dot12( FT_Byte**  p,
                FT_Byte*   limit,
                FT_Fixed*  value )
  {
    FT_Short  val;


    if ( *p + 1 >= limit )
      return FT_THROW( Invalid_Table );

    val = (FT_Short)( ( (FT_UShort)(*p)[0] << 8 ) |
                        (FT_UShort)(*p)[1]        );
    *p += 2;

    /* Convert F4DOT12 (4.12) to F16DOT16 (16.16) by shifting left 4. */
    *value = (FT_Fixed)val * 16;

    return FT_Err_Ok;
  }


  /**************************************************************************
   *
   * @Function:
   *   read_f6dot10
   *
   * @Description:
   *   Read a signed F6DOT10 value (6.10 fixed-point).
   *
   * @Input:
   *   p ::
   *     Pointer to the byte stream (will be advanced).
   *
   *   limit ::
   *     Maximum valid address in the stream.
   *
   * @Output:
   *   value ::
   *     The decoded value as 16.16 fixed-point.
   *
   * @Return:
   *   FreeType error code. 0 means success.
   */
  static FT_Error
  read_f6dot10( FT_Byte**  p,
                FT_Byte*   limit,
                FT_Fixed*  value )
  {
    FT_Short  val;


    if ( *p + 1 >= limit )
      return FT_THROW( Invalid_Table );

    val = (FT_Short)( ( (FT_UShort)(*p)[0] << 8 ) |
                        (FT_UShort)(*p)[1]        );
    *p += 2;

    /* Simply shift left by 6. */
    *value = (FT_Fixed)val * 64;

    return FT_Err_Ok;
  }


  /**************************************************************************
   *
   * Coverage Table Functions
   *
   */

  /**************************************************************************
   *
   * @Function:
   *   tt_varc_get_coverage
   *
   * @Description:
   *   Check whether a glyph is covered by the 'VARC' table using binary
   *   search.
   *
   * @Input:
   *   varc ::
   *     The 'VARC' table structure.
   *
   *   glyph_index ::
   *     The glyph ID to check.
   *
   * @Return:
   *   Coverage index if found, -1 otherwise.
   */
  static FT_Int
  tt_varc_get_coverage( TT_Varc  varc,
                        FT_UInt  glyph_index )
  {
    FT_Byte*  p;
    FT_UInt   format;
    FT_UInt   count;


    if ( !varc || !varc->coverage )
      return -1;

    p = varc->coverage;

    /* Check bounds for format and count. */
    if ( !CHECK_TABLE_BOUNDS( p, 4 ) )
      return -1;

    format = ( p[0] << 8 ) | p[1];
    p += 2;

    /* formats 1 and 3: list of 16-bit or 24-bit glyph IDs */
    if ( format == 1 || format == 3 )
    {
      FT_UInt  glyph_size = format == 1 ? 2 : 3;


      if ( !CHECK_TABLE_BOUNDS( p, glyph_size ) )
        return -1;

      count = format == 1 ? FT_NEXT_USHORT( p )
                          : FT_NEXT_UOFF3( p );

      if ( !CHECK_TABLE_BOUNDS( p, count * glyph_size ) )
        return -1;

      /* binary search */
      {
        FT_UInt  min = 0;
        FT_UInt  max = count;


        while ( min < max )
        {
          FT_UInt   mid   = ( min + max ) >> 1;
          FT_Byte*  gid_p = p + mid * glyph_size;
          FT_UInt   gid   = format == 1 ? FT_PEEK_USHORT( gid_p )
                                        : FT_PEEK_UOFF3( gid_p );


          if ( gid == glyph_index )
            return (FT_Int)mid;
          else if ( gid < glyph_index )
            min = mid + 1;
          else
            max = mid;
        }
      }
    }

    /* formats 2 and 4: range list with 16-bit or 24-bit values */
    else if ( format == 2 || format == 4 )
    {
      FT_UInt  value_size  = format == 2 ? 2 : 3;
      FT_UInt  record_size = value_size * 3;
      FT_UInt  min, max;


      if ( !CHECK_TABLE_BOUNDS( p, value_size ) )
        return -1;

      count = format == 2 ? FT_NEXT_USHORT( p )
                          : FT_NEXT_UOFF3( p );

      if ( !CHECK_TABLE_BOUNDS( p, count * record_size ) )
        return -1;

      /* binary search the sorted ranges */
      min = 0;
      max = count;
      while ( min < max )
      {
        FT_UInt   mid   = ( min + max ) >> 1;
        FT_Byte*  rec   = p + mid * record_size;
        FT_UInt   start = format == 2 ? FT_PEEK_USHORT( rec )
                                      : FT_PEEK_UOFF3( rec );
        FT_UInt   end   = format == 2 ? FT_PEEK_USHORT( rec + value_size )
                                      : FT_PEEK_UOFF3( rec + value_size );


        if ( glyph_index < start )
          max = mid;
        else if ( glyph_index > end )
          min = mid + 1;
        else
        {
          FT_Byte*  index_p = rec + value_size * 2;
          FT_UInt   index   = format == 2 ? FT_PEEK_USHORT( index_p )
                                          : FT_PEEK_UOFF3( index_p );


          return (FT_Int)( index + glyph_index - start );
        }
      }
    }

    return -1;
  }


  /**************************************************************************
   *
   * CFF2IndexOf Functions
   *
   */

  /**************************************************************************
   *
   * @Function:
   *   tt_varc_read_cff2_index_entry
   *
   * @Description:
   *   Read entry `index` from a CFF2-style INDEX:
   *
   *     count (uint32) | offSize (uint8) | offset[count + 1] | data
   *
   *   Each offset is `offSize` bytes and 1-based.  On success `*data`
   *   points to the entry's bytes and `*size` is its length.  Every read
   *   and the returned range are validated against `table_limit`; the
   *   caller must pass an `index_data` pointer that lies within the table.
   *
   * @Return:
   *   FreeType error code.  0 means success.
   */
  static FT_Error
  tt_varc_read_cff2_index_entry( FT_Byte*   index_data,
                                 FT_Byte*   table_limit,
                                 FT_UInt    index,
                                 FT_Byte**  data,
                                 FT_UInt*   size )
  {
    FT_Byte*   p = index_data;
    FT_UInt32  count;
    FT_UInt    offSize;
    FT_ULong   offset1 = 0;
    FT_ULong   offset2 = 0;
    FT_Byte*   offset_base;

    FT_UInt  i;


    if ( !p || p + 4 > table_limit )
      return FT_THROW( Invalid_Table );

    count = ( (FT_UInt32)p[0] << 24 ) |
            ( (FT_UInt32)p[1] << 16 ) |
            ( (FT_UInt32)p[2] << 8  ) |
              (FT_UInt32)p[3];
    p += 4;

    if ( p >= table_limit )
      return FT_THROW( Invalid_Table );

    offSize = *p++;
    if ( offSize < 1 || offSize > 4 || index >= count )
      return FT_THROW( Invalid_Table );

    /* division avoids overflow in the offset-array bounds check */
    if ( count >= (FT_UInt32)( ( table_limit - p ) / offSize ) )
      return FT_THROW( Invalid_Table );

    offset_base = p + ( count + 1 ) * offSize;

    p += index * offSize;
    for ( i = 0; i < offSize; i++ )
      offset1 = ( offset1 << 8 ) | p[i];

    p += offSize;
    for ( i = 0; i < offSize; i++ )
      offset2 = ( offset2 << 8 ) | p[i];

    /* offsets are 1-based and must stay within the table */
    if ( offset2 < offset1                                     ||
         offset1 < 1                                           ||
         offset2 - 1 > (FT_ULong)( table_limit - offset_base ) )
      return FT_THROW( Invalid_Table );

    *data = offset_base + offset1 - 1;
    *size = (FT_UInt)( offset2 - offset1 );

    return FT_Err_Ok;
  }


  /**************************************************************************
   *
   * @Function:
   *   tt_varc_get_glyph_record
   *
   * @Description:
   *   Get the glyph record from the `CFF2IndexOf` structure.
   *
   * @Input:
   *   varc ::
   *     The 'VARC' table structure.
   *
   *   glyph_index ::
   *     The glyph ID.
   *
   * @Output:
   *   record_data ::
   *     Pointer to the glyph record data.
   *
   *   record_size ::
   *     Size of the glyph record in bytes.
   *
   * @Return:
   *   FreeType error code.  0 means success.
   */
  static FT_Error
  tt_varc_get_glyph_record( TT_Varc    varc,
                            FT_UInt    glyph_index,
                            FT_Byte**  record_data,
                            FT_UInt*   record_size )
  {
    FT_Error  error;
    FT_Int    coverage_index;


    if ( !varc || !varc->var_composite_glyphs )
      return FT_THROW( Invalid_Table );

    coverage_index = tt_varc_get_coverage( varc, glyph_index );
    if ( coverage_index < 0 )
      return FT_THROW( Invalid_Glyph_Index );

    error = tt_varc_read_cff2_index_entry(
              varc->var_composite_glyphs,
              (FT_Byte*)varc->table + varc->table_size,
              (FT_UInt)coverage_index,
              record_data, record_size );
    if ( error )
      return error;

    /* a component record is at most 64KB */
    if ( *record_size > 0xFFFFU )
      return FT_THROW( Invalid_Table );

    return FT_Err_Ok;
  }


  /**************************************************************************
   *
   * Component Parsing Functions
   *
   */

  /**************************************************************************
   *
   * @Function:
   *   tt_varc_parse_component
   *
   * @Description:
   *   Parse a single VARC component record.
   *
   * @Input:
   *   face ::
   *     The font face.
   *
   *   varc ::
   *     The 'VARC' table structure.
   *
   *   p ::
   *     Pointer to component data (will be advanced).
   *
   *   limit ::
   *     End of component data.
   *
   *   axis_values_buffer ::
   *     A stack buffer to be used for small arrays.
   *
   *   axis_values_buffer_size ::
   *     The size of `axis_values_buffer`.  If the number of parsed axis
   *     values exceeds this size, an array gets allocated on the heap.
   *
   * @Output:
   *   component ::
   *     The parsed component structure.
   *
   * @Return:
   *   FreeType error code.  0 means success.
   */
  static FT_Error
  tt_varc_parse_component( TT_Face           face,
                           TT_Varc           varc,
                           FT_Byte**         p,
                           FT_Byte*          limit,
                           TT_VarcComponent  component,
                           FT_Fixed*         axis_values_buffer,
                           FT_UInt           axis_values_buffer_size )
  {
    FT_Error   error;
    FT_Memory  memory = face->root.memory;

    FT_UInt32  flags32;


    FT_ZERO( component );

    /* read flags (uint32var) */
    error = read_uint32var( p, limit, &flags32 );
    if ( error )
      return error;

    component->flags = flags32;
    /* read GID */
    if ( component->flags & VARC_GID_IS_24BIT )
    {
      if ( *p + 2 >= limit )
        return FT_THROW( Invalid_Table );
      component->gid = ( (*p)[0] << 16 ) |
                       ( (*p)[1] << 8  ) |
                         (*p)[2];
      *p += 3;
    }
    else
    {
      if ( *p + 1 >= limit )
        return FT_THROW( Invalid_Table );
      component->gid = ( (*p)[0] << 8 ) |
                         (*p)[1];
      *p += 2;
    }

    /* validate GID */
    if ( component->gid >= (FT_UInt)face->root.num_glyphs )
    {
      /* For now, don't fail - the font might be using extended GID space. */
      /* return FT_THROW( Invalid_Glyph_Index ); */
    }

    /* read condition index */
    if ( component->flags & VARC_HAVE_CONDITION )
    {
      error = read_uint32var( p, limit, &component->condition_index );
      if ( error )
        return error;
    }

    /* read axis indices index */
    if ( component->flags & VARC_HAVE_AXES )
    {
      error = read_uint32var( p, limit, &component->axis_indices_index );
      if ( error )
        return error;
    }

    /* read axis values (TupleValues) */
    if ( component->flags & VARC_HAVE_AXES )
    {
      FT_UInt   axis_count  = 0;
      FT_Byte*  table_limit = (FT_Byte*)varc->table + varc->table_size;
      FT_Byte*  tuple_data;
      FT_UInt   tuple_size;


      if ( !varc->axis_indices_list )
        return FT_THROW( Invalid_Table );

      error = tt_varc_read_cff2_index_entry( varc->axis_indices_list,
                                             table_limit,
                                             component->axis_indices_index,
                                             &tuple_data, &tuple_size );
      if ( error )
        return error;

      /* decode the TupleValues just to count the axis values */
      {
        FT_Byte*  tuple_limit = tuple_data + tuple_size;
        FT_Byte*  tp          = tuple_data;


        while ( tp < tuple_limit )
        {
          FT_Byte  control = *tp++;
          FT_UInt  cnt     = ( control & 0x3F ) + 1;


          if ( ( control & 0xC0 ) == 0xC0 )
          {
            /* 32-bit values */
            if ( tp + 4 * cnt > tuple_limit )
              break;
            tp += 4 * cnt;
          }
          else if ( control & 0x80 )
            ;  /* zeros - no data bytes */
          else if ( control & 0x40 )
          {
            /* 2-byte values */
            if ( tp + 2 * cnt > tuple_limit )
              break;
            tp += 2 * cnt;
          }
          else
          {
            /* 1-byte values */
            if ( tp + cnt > tuple_limit )
              break;
            tp += cnt;
          }

          axis_count += cnt;
        }
      }

      component->num_axis_values = axis_count;

      if ( axis_count > 0 )
      {
        FT_UInt  i;
        FT_UInt  j;


        /* use caller's stack buffer for small arrays, heap for large */
        if ( axis_count <= axis_values_buffer_size )
        {
          component->axis_values         = axis_values_buffer;
          component->axis_values_on_heap = FALSE;
        }
        else
        {
          if ( FT_NEW_ARRAY( component->axis_values, axis_count ) )
            return error;
          component->axis_values_on_heap = TRUE;
        }

        /* read axis values from TupleValues (packed deltas format) */
        i = 0;
        while ( i < axis_count )
        {
          FT_Byte  control;
          FT_UInt  cnt;


          /* read control byte */
          if ( *p >= limit )
            goto Fail;
          control = *(*p)++;

          /* extract run count (lower 6 bits), add 1 */
          cnt = ( control & 0x3F ) + 1;
          if ( cnt > axis_count - i )
            cnt = axis_count - i;

          /* check top 2 bits for encoding type */
          if ( ( control & 0xC0 ) == 0xC0 )
          {
            /* Both bits set = 32-bit signed integers */
            if ( *p + 4 * cnt > limit )
              goto Fail;
            for ( j = 0; j < cnt; j++ )
            {
              /* read as signed 32-bit integer,     */
              /* reinterpret as F2DOT14 -> F16DOT16 */
              FT_Int32  val = (FT_Int32)( ( (FT_UInt32)(*p)[0] << 24 ) |
                                          ( (FT_UInt32)(*p)[1] << 16 ) |
                                          ( (FT_UInt32)(*p)[2] << 8  ) |
                                            (FT_UInt32)(*p)[3]         );


              component->axis_values[i++] = (FT_Fixed)MUL_INT32( val, 4 );
              *p += 4;
            }
          }
          else if ( control & 0x80 )
          {
            /* bit 7 set = zeros */
            for ( j = 0; j < cnt; j++ )
              component->axis_values[i++] = 0;
          }
          else if ( control & 0x40 )
          {
            /* bit 6 set = 2-byte signed integers */
            if ( *p + 2 * cnt > limit )
              goto Fail;
            for ( j = 0; j < cnt; j++ )
            {
              /* read as signed 16-bit integer,     */
              /* reinterpret as F2DOT14 -> F16DOT16 */
              FT_Short  val = (FT_Short)( ( (FT_UShort)(*p)[0] << 8 ) |
                                            (FT_UShort)(*p)[1]        );


              component->axis_values[i++] = (FT_Fixed)( val * 4 );
              *p += 2;
            }
          }
          else
          {
            /* neither bit set = 1-byte signed integers */
            if ( *p + cnt > limit )
              goto Fail;
            for ( j = 0; j < cnt; j++ )
            {
              /* read signed byte integer,          */
              /* reinterpret as F2DOT14 -> F16DOT16 */
              FT_Char  val = (FT_Char)(*(*p)++);


              component->axis_values[i++] = (FT_Fixed)( val * 4 );
            }
          }
        }
      }
    }

    /* read axis values variation index */
    if ( component->flags & VARC_AXES_HAVE_VARIATION )
    {
      error = read_uint32var( p, limit, &component->axis_values_var_index );
      if ( error )
        goto Fail;
    }

    /* read transform variation index */
    if ( component->flags & VARC_TRANSFORM_HAS_VARIATION )
    {
      error = read_uint32var( p, limit, &component->transform_var_index );
      if ( error )
        goto Fail;
    }

    /* initialize transform to identity */
    component->translate_x = 0;
    component->translate_y = 0;
    component->rotation    = 0;
    component->scale_x     = 0x10000;  /* 1.0 in 16.16 */
    component->scale_y     = 0x10000;  /* 1.0 in 16.16 */
    component->skew_x      = 0;
    component->skew_y      = 0;
    component->tcenter_x   = 0;
    component->tcenter_y   = 0;

    /* read transform components - each field is optional based on flags */
    if ( component->flags & VARC_HAVE_TRANSLATE_X )
    {
      error = read_fword( p, limit, &component->translate_x );
      if ( error )
        goto Fail;
    }

    if ( component->flags & VARC_HAVE_TRANSLATE_Y )
    {
      error = read_fword( p, limit, &component->translate_y );
      if ( error )
        goto Fail;
    }

    if ( component->flags & VARC_HAVE_ROTATION )
    {
      error = read_f4dot12( p, limit, &component->rotation );
      if ( error )
        goto Fail;
    }

    if ( component->flags & VARC_HAVE_SCALE_X )
    {
      error = read_f6dot10( p, limit, &component->scale_x );
      if ( error )
        goto Fail;
    }

    if ( component->flags & VARC_HAVE_SCALE_Y )
    {
      error = read_f6dot10( p, limit, &component->scale_y );
      if ( error )
        goto Fail;
    }
    else if ( component->flags & VARC_HAVE_SCALE_X )
    {
      /* ScaleY defaults to ScaleX if ScaleX is present */
      component->scale_y = component->scale_x;
    }

    if ( component->flags & VARC_HAVE_SKEW_X )
    {
      error = read_f4dot12( p, limit, &component->skew_x );
      if ( error )
        goto Fail;
    }

    if ( component->flags & VARC_HAVE_SKEW_Y )
    {
      error = read_f4dot12( p, limit, &component->skew_y );
      if ( error )
        goto Fail;
    }

    if ( component->flags & VARC_HAVE_TCENTER_X )
    {
      error = read_fword( p, limit, &component->tcenter_x );
      if ( error )
        goto Fail;
    }

    if ( component->flags & VARC_HAVE_TCENTER_Y )
    {
      error = read_fword( p, limit, &component->tcenter_y );
      if ( error )
        goto Fail;
    }

    /* Skip one uint32var for each set reserved flag bit (bits 15-31), */
    /* keeping the byte stream aligned for forward-compatible          */
    /* extensions, as HarfBuzz, skrifa, and fontTools do.              */
    {
      FT_UInt32  reserved = component->flags & VARC_RESERVED_MASK;


      while ( reserved )
      {
        FT_UInt32  ignore;


        error = read_uint32var( p, limit, &ignore );
        if ( error )
          goto Fail;

        reserved &= reserved - 1;
      }
    }

    return FT_Err_Ok;

  Fail:
    /* the axis-value reading loop jumps here on truncated data */
    /* without setting `error`; never report success on failure */
    if ( !error )
      error = FT_THROW( Invalid_Table );
    if ( component->axis_values && component->axis_values_on_heap )
      FT_FREE( component->axis_values );
    return error;
  }


  /**************************************************************************
   *
   * @Function:
   *   tt_varc_free_component
   *
   * @Description:
   *   Free resources allocated for a component.
   *
   * @Input:
   *   face ::
   *     The font face.
   *
   *   component ::
   *     The component to free.
   */
  static void
  tt_varc_free_component( TT_Face           face,
                          TT_VarcComponent  component )
  {
    FT_Memory  memory = face->root.memory;


    if ( component->axis_values && component->axis_values_on_heap )
      FT_FREE( component->axis_values );
  }


  /**************************************************************************
   *
   * MultiItemVariationStore Functions
   *
   */

  /**************************************************************************
   *
   * @Function:
   *   tt_varc_evaluate_region
   *
   * @Description:
   *   Evaluate a variation region against current coordinates.
   *   Return a scalar value (0.0 to 1.0) indicating how much this
   *   region contributes at the current variation location.
   *
   * @Input:
   *   region_data ::
   *     Pointer to region data (axis ranges).
   *
   *   limit ::
   *     End of valid data.
   *
   *   coords ::
   *     Current normalized coordinates (F2DOT14).
   *
   *   coord_count ::
   *     Number of coordinates.
   *
   * @Return:
   *   Scalar value as FT_Fixed (16.16 format), 0 to 0x10000 (1.0).
   */
  static FT_Fixed
  tt_varc_evaluate_region( FT_Byte*   region_data,
                           FT_Byte*   limit,
                           FT_Fixed*  coords,
                           FT_UInt    coord_count )
  {
    FT_Byte*  p      = region_data;
    FT_Fixed  scalar = 0x10000;  /* start with 1.0 in 16.16 */
    FT_UInt   axis_count;

    FT_UInt  i;


    if ( !coords || coord_count == 0 )
      return scalar;  /* no coords available, return 1.0 */

    /* `SparseVariationRegion` table:
     *
     *   regionAxisCount (u16) - number of non-default axes
     *   axisOffsets (u32[regionAxisCount]) - relative to this region
     *   for each sparse axis:
     *     axisIndex (u16)
     *     startCoord (F2DOT14)
     *     peakCoord (F2DOT14)
     *     endCoord (F2DOT14)
     */

    if ( limit - p < 2 )
      return 0;

    /* read sparse axis count (u16) */
    axis_count = ( (FT_UInt)p[0] << 8 ) | p[1];
    p += 2;

    if ( axis_count > (FT_ULong)( limit - p ) / 4 )
      return 0;

    /* process each axis range */
    for ( i = 0; i < axis_count; i++ )
    {
      FT_UInt16  axis_index;
      FT_Short   min_val, peak_val, max_val;
      FT_Fixed   coord;
      FT_Fixed   axis_scalar;
      FT_UInt32  axis_offset;
      FT_Byte*   axis;

      FT_Fixed  min_16;
      FT_Fixed  peak_16;
      FT_Fixed  max_16;


      axis_offset = ( (FT_UInt32)p[0] << 24 ) |
                    ( (FT_UInt32)p[1] << 16 ) |
                    ( (FT_UInt32)p[2] << 8  ) |
                      (FT_UInt32)p[3];
      p += 4;

      if ( axis_offset > (FT_ULong)( limit - region_data )     ||
           (FT_ULong)( limit - region_data ) - axis_offset < 8 )
        return 0;

      axis       = region_data + axis_offset;
      axis_index = ( (FT_UInt16)axis[0] << 8 ) | axis[1];
      axis      += 2;

      /* read min, peak, max (F2DOT14) - each is 2 bytes, */
      /* advance after each read                          */
      min_val  = (FT_Short)( ( (FT_UShort)axis[0] << 8 ) |
                               (FT_UShort)axis[1]        );
      axis += 2;
      peak_val = (FT_Short)( ( (FT_UShort)axis[0] << 8 ) |
                               (FT_UShort)axis[1]        );
      axis += 2;
      max_val  = (FT_Short)( ( (FT_UShort)axis[0] << 8 ) |
                               (FT_UShort)axis[1]        );

      /* get coordinate for this axis */
      if ( axis_index >= coord_count )
        continue;  /* skip unknown axes */

      coord = coords[axis_index];

      /* convert F2DOT14 to F16DOT16 for comparison */
      min_16  = (FT_Fixed)min_val  * 4;
      peak_16 = (FT_Fixed)peak_val * 4;
      max_16  = (FT_Fixed)max_val  * 4;

      /* Evaluate this axis with the sparse-region tent function, */
      /* matching HarfBuzz (`VarRegionAxis::evaluate`) and skrifa */
      /* (`SparseVariationRegion::compute_scalar`).               */
      if ( peak_16 == 0 )
        continue;                /* axis not part of region -> factor 1  */

      if ( coord == peak_16 )
        continue;                /* at peak -> factor 1                  */

      if ( coord == 0 )
        return 0;                /* default location -> region gives 0   */

      if ( min_16 > peak_16             ||
           peak_16 > max_16             ||
           ( min_16 < 0 && max_16 > 0 ) )
        continue;                /* ill-formed axis -> factor 1          */

      if ( coord < min_16 || coord > max_16 )
        return 0;                /* outside range -> region gives 0      */

      if ( coord < peak_16 )
        axis_scalar = FT_DivFix( coord - min_16, peak_16 - min_16 );
      else
        axis_scalar = FT_DivFix( max_16 - coord, max_16 - peak_16 );

      /* multiply into overall scalar */
      scalar = FT_MulFix( scalar, axis_scalar );

      /* early exit if scalar becomes 0 */
      if ( scalar == 0 )
        return 0;
    }

    return scalar;
  }


  /**************************************************************************
   *
   * MultiItemVariationStore Functions
   *
   */

  /**************************************************************************
   *
   * @Function:
   *   tt_varc_read_tuple_deltas
   *
   * @Description:
   *   Read a tuple of delta values from TupleValues (packed deltas).
   *   Return plain integers that will be scaled later.
   *
   * @Input:
   *   data ::
   *     Pointer to TupleValues data.
   *
   *   limit ::
   *     End of valid data.
   *
   *   num_deltas ::
   *     Number of deltas to read.
   *
   * @Output:
   *   deltas ::
   *     Array to store deltas (as plain integers).
   *
   *   bytes_consumed ::
   *     The number of bytes consumed for this tuple.
   *
   * @Return:
   *   FreeType error code.  0 means success.
   */
  static FT_Error
  tt_varc_read_tuple_deltas( FT_Byte*   data,
                             FT_Byte*   limit,
                             FT_UInt    num_deltas,
                             FT_Int32*  deltas,
                             FT_UInt*   bytes_consumed )
  {
    FT_Byte*  p     = data;
    FT_Byte*  start = data;

    FT_UInt  i = 0;


    /* read deltas using TupleValues (packed delta) format */
    while ( i < num_deltas && p < limit )
    {
      FT_Byte  control;
      FT_UInt  cnt, j;


      /* read control byte */
      if ( p >= limit )
        return FT_THROW( Invalid_Table );

      control = *p++;

      /* extract run count (lower 6 bits), add 1 */
      cnt = ( control & 0x3F ) + 1;
      if ( cnt > num_deltas - i )
        cnt = num_deltas - i;

      /* check top 2 bits for encoding type */
      if ( ( control & 0xC0 ) == 0xC0 )
      {
        /* both bits set = 32-bit plain integer values */
        if ( p + 4 * cnt > limit )
          return FT_THROW( Invalid_Table );

        for ( j = 0; j < cnt; j++ )
        {
          /* read as signed 32-bit plain integer */
          FT_Int32  val = (FT_Int32)( ( (FT_UInt32)p[0] << 24 ) |
                                      ( (FT_UInt32)p[1] << 16 ) |
                                      ( (FT_UInt32)p[2] << 8  ) |
                                        (FT_UInt32)p[3]         );


          deltas[i++] = val;  /* store as plain integer */
          p += 4;
        }
      }
      else if ( control & 0x80 )
      {
        /* bit 7 set = zeros */
        for ( j = 0; j < cnt; j++ )
          deltas[i++] = 0;
      }
      else if ( control & 0x40 )
      {
        /* bit 6 set = 2-byte signed integer values */
        if ( p + 2 * cnt > limit )
          return FT_THROW( Invalid_Table );

        for ( j = 0; j < cnt; j++ )
        {
          /* read as signed 16-bit plain integer */
          FT_Short  val = (FT_Short)( ( (FT_UShort)p[0] << 8 ) |
                                        (FT_UShort)p[1]        );


          deltas[i++] = val;  /* store as plain integer */
          p += 2;
        }
      }
      else
      {
        /* neither bit set = 1-byte signed integer values */
        if ( p + cnt > limit )
          return FT_THROW( Invalid_Table );

        for ( j = 0; j < cnt; j++ )
        {
          /* read signed byte as plain integer */
          FT_Char  val = (FT_Char)(*p++);


          deltas[i++] = val;  /* store as plain integer */
        }
      }
    }

    /* if we hit the end before reading all deltas, */
    /* fill remaining with zeros                    */
    while ( i < num_deltas )
      deltas[i++] = 0;

    /* return the number of bytes consumed */
    *bytes_consumed = (FT_UInt)( p - start );

    return FT_Err_Ok;
  }


  /**************************************************************************
   *
   * @Function:
   *   tt_varc_get_item_deltas
   *
   * @Description:
   *   Get a tuple of deltas from the `MultiItemVariationStore`.
   *   Unlike `ItemVariationStore`, which returns a single delta,
   *   `MultiItemVariationStore` returns multiple deltas (a tuple).
   *
   * @Input:
   *   face ::
   *     The font face.
   *
   *   varc ::
   *     The VARC table.
   *
   *   var_index ::
   *     Variation index (outer << 16 | inner).
   *
   *   num_deltas ::
   *     Number of deltas expected in the tuple.
   *
   *   shift ::
   *     Number of bits to preserve from fractional interpolation.
   *     The raw integer deltas are effectively shifted left by this
   *     amount.  Use shift=2 for F2DOT14 data (returns F16DOT16),
   *     shift=4 for F4DOT12 data (returns F16DOT16), etc.
   *     Maximum value is 16.
   *
   *   current_coords ::
   *     Current normalized coordinates (F2DOT14).
   *
   *   num_coords ::
   *     Number of coords in `current_coords`.
   *
   *   condition_defaults ::
   *     Optional defaults for condition values.  If non-NULL, return each
   *     unrounded value's positive sign (0 or 1), instead of scaled deltas.
   *
   * @Output:
   *   deltas ::
   *     Array to store the deltas (must have space for `num_deltas`).
   *     Values are raw integer deltas scaled left by `shift` bits.
   *
   * @Return:
   *   FreeType error code.  0 means success.
   */
  static FT_Error
  tt_varc_get_item_deltas( TT_Face    face,
                           TT_Varc    varc,
                           FT_UInt32  var_index,
                           FT_UInt    num_deltas,
                           FT_Long*   deltas,
                           FT_UInt    shift,
                           FT_Fixed*  current_coords,
                           FT_UInt    num_coords,
                           FT_Short*  condition_defaults )
  {
#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT

    FT_Error   error;
    FT_Memory  memory = face->root.memory;

    FT_Byte*   mvs_data;      /* MultiItemVariationStore */
    FT_Byte*   table_limit;
    FT_UInt16  format;
    FT_UInt32  regions_offset;
    FT_UInt16  data_sets_count;
    FT_UInt    outer, inner;
    FT_Byte*   p;
    FT_Byte*   data_set_offsets;
    FT_UInt32  data_set_offset;

    FT_Byte*   mvd_data;      /* MultiVarData */
    FT_Byte    mvd_format;
    FT_UInt16  region_count;
    FT_Byte*   delta_sets;
    FT_UInt32  delta_sets_offset;
    FT_Byte*   tuple_data;
    FT_UInt    tuple_size;

    FT_Int64   stack_accumulators[VARC_STACK_DELTA_COUNT];
    /* deltas * regions */
    FT_Int32   stack_all_deltas[VARC_STACK_DELTA_COUNT * 16];
    FT_Int64*  accumulators      = NULL;
    FT_Byte*   regions_data      = NULL;
    FT_Byte*   regions_limit     = NULL;
    FT_UInt32  region_list_count = 0;

    FT_UInt    total_deltas;
    FT_Int32*  all_deltas     = NULL;
    FT_UInt    bytes_consumed = 0;
    FT_Byte*   region_indices_ptr;
    FT_UInt    region_idx;

    FT_UInt  i;


    if ( !varc->multi_var_store_loaded || !varc->multi_var_store )
    {
      /* no variation store - return zeros */
      for ( i = 0; i < num_deltas; i++ )
        deltas[i] = condition_defaults ? FT_BOOL( condition_defaults[i] > 0 )
                                       : 0;
      return FT_Err_Ok;
    }

    mvs_data    = varc->multi_var_store;
    table_limit = (FT_Byte*)varc->table + varc->table_size;

    /* split `var_index` into `outer` and `inner` */
    outer = var_index >> 16;
    inner = var_index & 0xFFFF;

    /* parse `MultiItemVariationStore` header */
    p = mvs_data;

    if ( p + 8 > table_limit )
      return FT_THROW( Invalid_Table );

    /* read format (u16) - must be 1 */
    format = ( (FT_UInt16)p[0] << 8 ) | p[1];
    p += 2;

    if ( format != 1 )
      return FT_THROW( Invalid_Table );

    /* read regions offset (u32) */
    regions_offset = ( (FT_UInt32)p[0] << 24 ) |
                     ( (FT_UInt32)p[1] << 16 ) |
                     ( (FT_UInt32)p[2] << 8  ) |
                       (FT_UInt32)p[3];
    p += 4;

    /* read dataSets count (u16) */
    data_sets_count = ( (FT_UInt16)p[0] << 8 ) | p[1];
    p += 2;

    /* validate outer index */
    if ( outer >= data_sets_count )
      return FT_THROW( Invalid_Table );

    /* read data set offset for outer index */
    data_set_offsets = p;
    p                = data_set_offsets + outer * 4;

    if ( p + 4 > table_limit )
      return FT_THROW( Invalid_Table );

    data_set_offset = ( (FT_UInt32)p[0] << 24 ) |
                      ( (FT_UInt32)p[1] << 16 ) |
                      ( (FT_UInt32)p[2] << 8  ) |
                        (FT_UInt32)p[3];

    /* get `MultiVarData` */
    if ( data_set_offset > (FT_UInt32)( table_limit - mvs_data ) )
      return FT_THROW( Invalid_Table );

    mvd_data = mvs_data + data_set_offset;

    if ( table_limit - mvd_data < 7 )
      return FT_THROW( Invalid_Table );

    /* parse `MultiVarData` header */
    p = mvd_data;

    /* read format (u8) - must be 1 */
    mvd_format = *p++;

    if ( mvd_format != 1 )
      return FT_THROW( Invalid_Table );

    /* read `regionIndices` count (u16) */
    region_count = ( (FT_UInt16)p[0] << 8 ) | p[1];
    p += 2;

    if ( region_count > (FT_ULong)( table_limit - p ) / 2 )
      return FT_THROW( Invalid_Table );
    p += region_count * 2;

    if ( table_limit - p < 4 )
      return FT_THROW( Invalid_Table );

    delta_sets_offset = ( (FT_UInt32)p[0] << 24 ) |
                        ( (FT_UInt32)p[1] << 16 ) |
                        ( (FT_UInt32)p[2] << 8  ) |
                          (FT_UInt32)p[3];
    if ( !delta_sets_offset                                           ||
         delta_sets_offset > (FT_ULong)( table_limit - mvd_data )     ||
         (FT_ULong)( table_limit - mvd_data ) - delta_sets_offset < 4 )
      return FT_THROW( Invalid_Table );
    delta_sets = mvd_data + delta_sets_offset;

    /* read tuple from INDEX2 at inner index */
    error = tt_varc_read_cff2_index_entry( delta_sets, table_limit, inner,
                                           &tuple_data, &tuple_size );
    if ( error )
      return error;

    /* allocate 64-bit accumulators for each delta */
    if ( num_deltas <= VARC_STACK_DELTA_COUNT )
      accumulators = stack_accumulators;
    else
    {
      if ( FT_NEW_ARRAY( accumulators, num_deltas ) )
      {
        error = FT_THROW( Out_Of_Memory );
        goto Cleanup;
      }
    }

    /* initialize accumulators to zero */
    FT_ARRAY_ZERO( accumulators, num_deltas );
    if ( condition_defaults )
    {
      for ( i = 0; i < num_deltas; i++ )
      {
#ifdef FT_INT64
        accumulators[i] = (FT_Int64)condition_defaults[i] * 0x10000L;
#else
        accumulators[i].lo = (FT_UInt32)(FT_Int32)condition_defaults[i] << 16;
        accumulators[i].hi = condition_defaults[i] < 0 ? 0xFFFFFFFFU : 0;
#endif
      }
    }

    /* parse `SparseVarRegionList` to get region definitions */
    if ( regions_offset > 0                                     &&
         regions_offset <= (FT_ULong)( table_limit - mvs_data ) )
    {
      regions_data  = mvs_data + regions_offset;
      regions_limit = table_limit;

      /* read region count (u32) */
      if ( regions_limit - regions_data >= 4 )
      {
        region_list_count = ( (FT_UInt32)regions_data[0] << 24 ) |
                            ( (FT_UInt32)regions_data[1] << 16 ) |
                            ( (FT_UInt32)regions_data[2] << 8  ) |
                              (FT_UInt32)regions_data[3];
        regions_data += 4;

        if ( region_list_count >
               (FT_ULong)( regions_limit - regions_data ) / 4 )
          regions_data = NULL;
      }
      else
        regions_data = NULL;
    }

    if ( !regions_data )
    {
      if ( accumulators != stack_accumulators )
        FT_FREE( accumulators );
      error = FT_THROW( Invalid_Table );
      goto Cleanup;
    }

    /* read ALL deltas from the flat tuple */
    /* (region_count * num_deltas values)  */

    /* a wrapped product would under-size `all_deltas` below */
    if ( num_deltas != 0 && region_count > FT_UINT_MAX / num_deltas )
    {
      if ( accumulators != stack_accumulators )
        FT_FREE( accumulators );
      error = FT_THROW( Invalid_Table );
      goto Cleanup;
    }

    total_deltas = region_count * num_deltas;

    /* allocate space for all deltas */
    if ( total_deltas <= VARC_STACK_DELTA_COUNT * 16 )
      all_deltas = stack_all_deltas;
    else
    {
      if ( FT_NEW_ARRAY( all_deltas, total_deltas ) )
      {
        if ( accumulators != stack_accumulators )
          FT_FREE( accumulators );
        error = FT_THROW( Out_Of_Memory );
        goto Cleanup;
      }
    }

    /* read all deltas from TupleValues */
    error = tt_varc_read_tuple_deltas( tuple_data, tuple_data + tuple_size,
                                       total_deltas, all_deltas,
                                       &bytes_consumed );
    if ( error )
    {
      if ( all_deltas != stack_all_deltas )
        FT_FREE( all_deltas );
      if ( accumulators != stack_accumulators )
        FT_FREE( accumulators );
      goto Cleanup;
    }

    /* parse regionIndices and accumulate scaled deltas */

    /* after format(u8) + region_count(u16) */
    region_indices_ptr = mvd_data + 3;

    for ( region_idx = 0; region_idx < region_count; region_idx++ )
    {
      FT_UInt16  region_index;
      FT_Fixed   region_scalar;

      FT_UInt  k;


      /* read region index from `regionIndices` array (u16) */
      if ( region_indices_ptr + 2 > table_limit )
        break;

      region_index        = ( (FT_UInt16)region_indices_ptr[0] << 8 ) |
                            region_indices_ptr[1];
      region_indices_ptr += 2;

      /* evaluate region to get scalar */
      if ( regions_data && current_coords )
      {
        /* SparseVarRegionList format:
         *
         *   u32: regionCount
         *   u32[regionCount]: array of offsets (from start of list)
         *                     to each region
         */

        /* back to start (before count) */
        FT_Byte*  region_list_start = regions_data - 4;
        FT_Byte*  region_offset_ptr;


        if ( region_index < region_list_count                              &&
             region_index < (FT_ULong)( regions_limit - regions_data ) / 4 )
        {
          FT_UInt32  region_offset;


          region_offset_ptr = regions_data + region_index * 4;
          region_offset = ( (FT_UInt32)region_offset_ptr[0] << 24 ) |
                          ( (FT_UInt32)region_offset_ptr[1] << 16 ) |
                          ( (FT_UInt32)region_offset_ptr[2] << 8  ) |
                            (FT_UInt32)region_offset_ptr[3];


          if ( region_offset <
                 (FT_UInt32)( regions_limit - region_list_start ) )
            region_scalar =
              tt_varc_evaluate_region( region_list_start + region_offset,
                                       regions_limit,
                                       current_coords,
                                       num_coords );
          else
            region_scalar = 0;
        }
        else
          region_scalar = 0;
      }
      else
      {
        /* no coords or regions - use 1.0 for first region, 0 for others */
        region_scalar = ( region_idx == 0 ) ? 0x10000 : 0;
      }

      /* scale and accumulate deltas for this region; */
      /* tuple is organized as regions-major:         */
      /*                                              */
      /*   [d0_r0, d1_r0, ..., dN_r0, d0_r1, ...]     */
      if ( region_scalar != 0 )
      {
        FT_UInt  base_offset = region_idx * num_deltas;


        for ( k = 0; k < num_deltas; k++ )
        {
          /* plain integer */
          FT_Int32  region_delta = all_deltas[base_offset + k];

          /* accumulate using 64-bit math: accum += delta * scalar */
#ifdef FT_INT64
          FT_Int64  product = (FT_Int64)region_delta * region_scalar;


          accumulators[k] += product;
#else /* !FT_INT64 */
          /* 32-bit fallback */
          if ( (FT_UInt32)region_delta + 0x8000U <= 0xFFFFU )
          {
            /* fast path - multiplication fits in 32 bits */
            FT_Int32  product = region_delta * region_scalar;


            accumulators[k].lo += (FT_UInt32)product;
            if ( accumulators[k].lo < (FT_UInt32)product )
              accumulators[k].hi += 1;
            if ( product < 0 )
              accumulators[k].hi -= 1;
          }
          else
          {
            /* slow path - full 64-bit signed multiplication */
            FT_UInt32  a = (FT_UInt32)ULABS( region_delta );
            FT_UInt32  b = (FT_UInt32)ULABS( region_scalar );

            FT_UInt32  a_lo = a & 0xFFFF;
            FT_UInt32  a_hi = a >> 16;
            FT_UInt32  b_lo = b & 0xFFFF;
            FT_UInt32  b_hi = b >> 16;

            FT_UInt32  mid = ( a_lo * b_hi ) + ( a_hi * b_lo );
            FT_UInt32  lo  = a_lo * b_lo;
            FT_UInt32  hi  = a_hi * b_hi;


            hi += ( mid >> 16 ) +
                  ( ( ( lo >> 16 ) + ( mid & 0xFFFF ) ) >> 16 );
            lo += ( mid << 16 );

            if ( ( region_delta < 0 ) != ( region_scalar < 0 ) )
            {
              lo = ~lo + 1;
              hi = ~hi + !lo;
            }

            accumulators[k].lo += lo;
            if ( accumulators[k].lo < lo )
              accumulators[k].hi += 1;
            accumulators[k].hi += hi;
          }
#endif /* !FT_INT64 */
        }
      }
    }

    /* free the flat delta array */
    if ( all_deltas != stack_all_deltas )
      FT_FREE( all_deltas );

    /* Convert 64-bit accumulators to deltas with rounding.          */
    /* Accumulators hold: raw_int_delta * region_scalar_16.16        */
    /* We shift right by (16 - shift) to preserve `shift` fractional */
    /* bits from the interpolation.                                  */
    {
      FT_Int   right_shift = 16 - shift;
      FT_Long  rounding    = right_shift > 0 ? ( 1L << ( right_shift - 1 ) )
                                             : 0;


      for ( i = 0; i < num_deltas; i++ )
      {
#ifdef FT_INT64
        if ( condition_defaults )
          deltas[i] = FT_BOOL( accumulators[i] > 0 );
        else
          deltas[i] =
            (FT_Long)( ( accumulators[i] + rounding ) >> right_shift );
#else
        /* 32-bit fallback */
        FT_UInt32  hi = accumulators[i].hi;
        FT_UInt32  lo = accumulators[i].lo;


        if ( condition_defaults )
        {
          deltas[i] = FT_BOOL( !( hi & 0x80000000U ) && ( hi || lo ) );
          continue;
        }

        /* add rounding */
        lo += (FT_UInt32)rounding;
        if ( lo < (FT_UInt32)rounding )
          hi += 1;

        /* shift right by (16 - shift) bits */
        if ( right_shift == 0 )
          deltas[i] = (FT_Long)(FT_Int32)lo;
        else
          deltas[i] = (FT_Long)(FT_Int32)( ( hi << ( 16 + shift ) ) |
                                           ( lo >> right_shift )    );
#endif
      }
    }

    /* free accumulators */
    if ( accumulators != stack_accumulators )
      FT_FREE( accumulators );

    error = FT_Err_Ok;

  Cleanup:
    return error;

#else /* !TT_CONFIG_OPTION_GX_VAR_SUPPORT */

    FT_UNUSED( face );
    FT_UNUSED( varc );
    FT_UNUSED( var_index );
    FT_UNUSED( num_deltas );
    FT_UNUSED( deltas );
    FT_UNUSED( shift );
    FT_UNUSED( current_coords );
    FT_UNUSED( num_coords );
    FT_UNUSED( condition_defaults );

    return FT_THROW( Unimplemented_Feature );

#endif /* !TT_CONFIG_OPTION_GX_VAR_SUPPORT */
  }


  /**************************************************************************
   *
   * Condition Evaluation
   *
   */

  /**************************************************************************
   *
   * @Function:
   *   tt_varc_get_condition
   *
   * @Description:
   *   Look up a 'Condition' by index in the top-level `ConditionList`.
   *   ConditionList:
   *     uint32 count, then Offset32[count] from its start.
   *
   * @Input:
   *   varc ::
   *     The 'VARC' table structure.
   *
   *   cond_index ::
   *     The index into the `ConditionList` structure.
   *
   *   table_limit ::
   *     End of valid table data.
   *
   * @Return:
   *    The condition offset.
   */
  static FT_Byte*
  tt_varc_get_condition( TT_Varc    varc,
                         FT_UInt32  cond_index,
                         FT_Byte*   table_limit )
  {
    FT_Byte*   cl = varc->condition_list;
    FT_UInt32  count;
    FT_Byte*   p;
    FT_UInt32  off;


    if ( !cl || cl + 4 > table_limit )
      return NULL;

    count = ( (FT_UInt32)cl[0] << 24 ) |
            ( (FT_UInt32)cl[1] << 16 ) |
            ( (FT_UInt32)cl[2] << 8  ) |
              (FT_UInt32)cl[3];
    if ( cond_index >= count )
      return NULL;

    /* bound the index with division so `cond_index * 4` cannot overflow */
    if ( cond_index > (FT_UInt32)( ( table_limit - cl - 4 ) / 4 ) )
      return NULL;

    p = cl + 4 + cond_index * 4;
    if ( p + 4 > table_limit )
      return NULL;

    off = ( (FT_UInt32)p[0] << 24 ) |
          ( (FT_UInt32)p[1] << 16 ) |
          ( (FT_UInt32)p[2] << 8  ) |
            (FT_UInt32)p[3];

    if ( off >= (FT_UInt32)( table_limit - cl ) )
      return NULL;

    return cl + off;
  }


  /**************************************************************************
   *
   * @Function:
   *   tt_varc_eval_condition
   *
   * @Description:
   *   Evaluate a 'Condition', matching HarfBuzz/skrifa/fontTools.
   *   Sub-condition offsets in the And/Or/Negate formats are 24-bit and
   *   relative to the start of the containing condition.
   *
   * @Input:
   *   face ::
   *     The font face.
   *
   *   varc ::
   *     The VARC table.
   *
   *   cond ::
   *     Pointer to the Condition.
   *
   *   table_limit ::
   *     End of valid table data.
   *
   *   coords ::
   *     Current normalized coordinates (F2DOT14).
   *
   *   coord_count ::
   *     Number of coordinates.
   *
   *   depth ::
   *     Recursion depth.
   *
   * @Return:
   *   TRUE when the condition is satisfied, FALSE otherwise.
   */
  static FT_Bool
  tt_varc_eval_condition( TT_Face    face,
                          TT_Varc    varc,
                          FT_Byte*   cond,
                          FT_Byte*   table_limit,
                          FT_Fixed*  coords,
                          FT_UInt    num_coords,
                          FT_UInt    depth )
  {
    FT_UInt  format;


    /* A zero condition offset represents True.  Top-level zero offsets */
    /* resolve to the start of the ConditionList.                       */
    if ( cond == varc->condition_list )
      return TRUE;

    if ( !cond                             ||
         depth > TT_VARC_MAX_NESTING_LEVEL ||
         cond + 2 > table_limit            )
      return FALSE;

    format = ( (FT_UInt)cond[0] << 8 ) | cond[1];

    switch ( format )
    {
    case 1:  /* ConditionAxisRange */
      {
        FT_UInt   axis_index;
        FT_Short  min_val, max_val;
        FT_Fixed  coord;


        if ( cond + 8 > table_limit )
          return FALSE;

        axis_index = ( (FT_UInt)cond[2] << 8 ) | cond[3];

        min_val = (FT_Short)( ( (FT_UShort)cond[4] << 8 ) |
                                (FT_UShort)cond[5]        );
        max_val = (FT_Short)( ( (FT_UShort)cond[6] << 8 ) |
                                (FT_UShort)cond[7]        );

        coord = ( coords && axis_index < num_coords ) ? coords[axis_index]
                                                      : 0;

        /* min_val/max_val are F2DOT14; coords are 16.16 (F2DOT14 << 2) */
        return FT_BOOL( coord >= ( (FT_Fixed)min_val * 4 ) &&
                        coord <= ( (FT_Fixed)max_val * 4 ) );
      }

    case 2:  /* ConditionValue */
      {
        FT_Short   default_value;
        FT_UInt32  var_idx;
        FT_Long    positive;


        if ( cond + 8 > table_limit )
          return FALSE;

        default_value = (FT_Short)( ( (FT_UShort)cond[2] << 8 ) |
                                      (FT_UShort)cond[3]        );

        var_idx = ( (FT_UInt32)cond[4] << 24 ) |
                  ( (FT_UInt32)cond[5] << 16 ) |
                  ( (FT_UInt32)cond[6] << 8  ) |
                    (FT_UInt32)cond[7];

        /* Compare the unrounded default plus interpolated delta. */
        /* Even a small positive fraction makes a condition true. */
        positive = FT_BOOL( default_value > 0 );
        if ( var_idx != 0xFFFFFFFFUL )
          (void)tt_varc_get_item_deltas( face, varc, var_idx, 1,
                                         &positive, 0,
                                         coords, num_coords,
                                         &default_value );

        return FT_BOOL( positive );
      }

    case 3:  /* ConditionAnd */
    case 4:  /* ConditionOr */
      {
        FT_UInt   count, i;
        FT_Byte*  p;
        FT_Bool   is_and = FT_BOOL( format == 3 );


        if ( cond + 3 > table_limit )
          return FALSE;

        count = cond[2];
        p     = cond + 3;

        if ( p + (FT_Offset)count * 3 > table_limit )
          return FALSE;

        for ( i = 0; i < count; i++ )
        {
          FT_UInt32  sub_off = ( (FT_UInt32)p[0] << 16 ) |
                               ( (FT_UInt32)p[1] << 8  ) |
                                 (FT_UInt32)p[2];
          FT_Bool    r;


          p += 3;
          r = sub_off == 0 ? TRUE
                           : tt_varc_eval_condition( face, varc,
                                                     cond + sub_off,
                                                     table_limit,
                                                     coords, num_coords,
                                                     depth + 1 );

          if ( is_and && !r )
            return FALSE;
          if ( !is_and && r )
            return TRUE;
        }

        return is_and;
      }

    case 5:  /* ConditionNegate */
      {
        FT_UInt32  sub_off;


        if ( cond + 5 > table_limit )
          return FALSE;

        sub_off = ( (FT_UInt32)cond[2] << 16 ) |
                  ( (FT_UInt32)cond[3] << 8  ) |
                    (FT_UInt32)cond[4];

        if ( sub_off == 0 )
          return FALSE;

        return FT_BOOL( !tt_varc_eval_condition( face, varc, cond + sub_off,
                                                 table_limit, coords,
                                                 num_coords, depth + 1 ) );
      }

    default:
      /* Unknown format: treat as not satisfied (matches fontTools). */
      return FALSE;
    }
  }


  /**************************************************************************
   *
   * Axis Index Reading
   *
   */

  /**************************************************************************
   *
   * @Function:
   *   tt_varc_get_axis_indices
   *
   * @Description:
   *   Read axis indices from the axis indices list (TupleList of integers).
   *
   * @Input:
   *   varc ::
   *     The VARC table.
   *
   *   axis_indices_index ::
   *     Index into the TupleList.
   *
   *   num_indices ::
   *     Number of indices to read.
   *
   * @Output:
   *   indices ::
   *     Array to store axis indices (caller allocates).
   *
   * @Return:
   *   FreeType error code.  0 means success.
   */
  static FT_Error
  tt_varc_get_axis_indices( TT_Varc   varc,
                            FT_UInt   axis_indices_index,
                            FT_UInt   num_indices,
                            FT_UInt*  indices )
  {
    FT_Error  error;

    FT_Byte*  table_limit = (FT_Byte*)varc->table + varc->table_size;
    FT_Byte*  tuple_data;
    FT_UInt   tuple_size;
    FT_Byte*  tuple_limit;

    FT_Byte*  p;
    FT_UInt   i, j;


    if ( !varc || !varc->axis_indices_list || !indices )
      return FT_THROW( Invalid_Argument );

    error = tt_varc_read_cff2_index_entry( varc->axis_indices_list,
                                           table_limit,
                                           axis_indices_index,
                                           &tuple_data, &tuple_size );
    if ( error )
      return error;

    tuple_limit = tuple_data + tuple_size;
    p           = tuple_data;

    /* read indices using TupleValues encoding */
    i = 0;
    while ( i < num_indices && p < tuple_limit )
    {
      FT_Byte  control;
      FT_UInt  cnt;


      control = *p++;
      cnt     = ( control & 0x3F ) + 1;

      if ( cnt > num_indices - i )
        cnt = num_indices - i;

      if ( ( control & 0xC0 ) == 0xC0 )
      {
        /* 32-bit values */
        if ( p + 4 * cnt > tuple_limit )
          return FT_THROW( Invalid_Table );

        for ( j = 0; j < cnt && i < num_indices; j++, i++ )
        {
          indices[i] = ( (FT_UInt)p[0] << 24 ) |
                       ( (FT_UInt)p[1] << 16 ) |
                       ( (FT_UInt)p[2] << 8  ) |
                         (FT_UInt)p[3];
          p += 4;
        }
      }
      else if ( control & 0x80 )
      {
        /* zeros */
        for ( j = 0; j < cnt && i < num_indices; j++, i++ )
          indices[i] = 0;
      }
      else if ( control & 0x40 )
      {
        /* 2-byte values */
        if ( p + 2 * cnt > tuple_limit )
          return FT_THROW( Invalid_Table );

        for ( j = 0; j < cnt && i < num_indices; j++, i++ )
        {
          indices[i] = ( (FT_UInt)p[0] << 8 ) |
                         (FT_UInt)p[1];
          p += 2;
        }
      }
      else
      {
        /* 1-byte values */
        if ( p + cnt > tuple_limit )
          return FT_THROW( Invalid_Table );

        for ( j = 0; j < cnt && i < num_indices; j++, i++ )
          indices[i] = *p++;
      }
    }

    if ( i < num_indices )
      return FT_THROW( Invalid_Table );

    return FT_Err_Ok;
  }


  /**************************************************************************
   *
   * Variation Delta Application Functions
   *
   */

  /**************************************************************************
   *
   * @Function:
   *   tt_varc_apply_axis_deltas
   *
   * @Description:
   *   Apply variation deltas to axis values using the
   *   `MultiItemVariationStore`.
   *
   * @Input:
   *   face ::
   *     The font face.
   *
   *   varc ::
   *     The VARC table.
   *
   *   component ::
   *     The component whose axis values to vary.
   *
   *   current_coords ::
   *     Current normalized coordinates (F2DOT14).
   *
   *   num_coords ::
   *     Number of coords in `current_coords`.
   */
  static void
  tt_varc_apply_axis_deltas( TT_Face           face,
                             TT_Varc           varc,
                             TT_VarcComponent  component,
                             FT_Fixed*         current_coords,
                             FT_UInt           num_coords )
  {
#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT

    FT_Error   error;
    FT_Memory  memory = face->root.memory;

    FT_Long   stack_deltas[VARC_STACK_DELTA_COUNT];
    FT_Long*  deltas = NULL;

    FT_UInt  i;


    if ( !varc->multi_var_store_loaded || component->num_axis_values == 0 )
      return;

    /* use stack allocation for small arrays, heap for large */
    if ( component->num_axis_values <= VARC_STACK_DELTA_COUNT )
      deltas = stack_deltas;
    else
    {
      if ( FT_NEW_ARRAY( deltas, component->num_axis_values ) )
        return;
    }

    /* Get deltas with shift=2: raw F2DOT14 integers are returned  */
    /* already shifted to F16DOT16, matching `axis_values` format. */
    error = tt_varc_get_item_deltas( face, varc,
                                     component->axis_values_var_index,
                                     component->num_axis_values,
                                     deltas, 2,
                                     current_coords,
                                     num_coords, NULL );
    if ( !error )
    {
      for ( i = 0; i < component->num_axis_values; i++ )
        component->axis_values[i] += deltas[i];
    }
#ifdef FT_DEBUG_LEVEL_TRACE
    else
      FT_TRACE2(( "tt_varc_apply_axis_deltas: variation index"
                  " 0x%08lx failed (error 0x%x)\n",
                  (FT_ULong)component->axis_values_var_index, error ));
#endif

    if ( deltas != stack_deltas )
      FT_FREE( deltas );

#else /* !TT_CONFIG_OPTION_GX_VAR_SUPPORT */

    FT_UNUSED( face );
    FT_UNUSED( varc );
    FT_UNUSED( component );
    FT_UNUSED( current_coords );
    FT_UNUSED( num_coords );

#endif /* !TT_CONFIG_OPTION_GX_VAR_SUPPORT */
  }


  /**************************************************************************
   *
   * @Function:
   *   tt_varc_apply_transform_deltas
   *
   * @Description:
   *   Apply variation deltas to transform values using the
   *   `MultiItemVariationStore`.
   *
   * @Input:
   *   face ::
   *     The font face.
   *
   *   varc ::
   *     The VARC table.
   *
   *   component ::
   *     The component whose transform values to vary.
   *
   *   current_coords ::
   *     Current normalized coordinates (F2DOT14).
   *
   *   num_coords ::
   *     Number of coords in `current_coords`.
   */
  static void
  tt_varc_apply_transform_deltas( TT_Face           face,
                                  TT_Varc           varc,
                                  TT_VarcComponent  component,
                                  FT_Fixed*         current_coords,
                                  FT_UInt           num_coords )
  {
#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT

    FT_Error   error;
    FT_Memory  memory = face->root.memory;

    FT_Long   stack_deltas[VARC_STACK_DELTA_COUNT];
    FT_Long*  deltas      = NULL;
    FT_UInt   num_deltas  = 0;
    FT_UInt   delta_index = 0;


    if ( !varc->multi_var_store_loaded )
      return;

    /* count how many transform components are present */
    if ( component->flags & VARC_HAVE_TRANSLATE_X ) num_deltas++;
    if ( component->flags & VARC_HAVE_TRANSLATE_Y ) num_deltas++;
    if ( component->flags & VARC_HAVE_ROTATION )    num_deltas++;
    if ( component->flags & VARC_HAVE_SCALE_X )     num_deltas++;
    if ( component->flags & VARC_HAVE_SCALE_Y )     num_deltas++;
    if ( component->flags & VARC_HAVE_SKEW_X )      num_deltas++;
    if ( component->flags & VARC_HAVE_SKEW_Y )      num_deltas++;
    if ( component->flags & VARC_HAVE_TCENTER_X )   num_deltas++;
    if ( component->flags & VARC_HAVE_TCENTER_Y )   num_deltas++;

    if ( num_deltas == 0 )
      return;

    /* use stack allocation for small arrays, heap for large */
    if ( num_deltas <= VARC_STACK_DELTA_COUNT )
      deltas = stack_deltas;
    else
    {
      if ( FT_NEW_ARRAY( deltas, num_deltas ) )
        return;
    }

    /* Get deltas with shift=4: raw integers shifted left by 4.      */
    /* F4DOT12 fields (rotation, skew) become F16DOT16 directly.     */
    /* FWORD and F6DOT10 fields need an additional << 2 at callsite. */
    error = tt_varc_get_item_deltas( face, varc,
                                     component->transform_var_index,
                                     num_deltas,
                                     deltas, 4,
                                     current_coords,
                                     num_coords, NULL );
    if ( !error )
    {
      /* Apply deltas to transform components in order.             */
      /* Deltas are already shifted left by 4 from get_item_deltas. */
      /* F4DOT12 fields (shift=4): add directly (already F16DOT16). */
      /* FWORD fields (shift=6):   need << 2 more for 26.6 format.  */
      /* F6DOT10 fields (shift=6): need << 2 more for F16DOT16.     */
      delta_index = 0;

      if ( component->flags & VARC_HAVE_TRANSLATE_X )
      {
        /* FWORD: 4+2=6 */
        component->translate_x += deltas[delta_index] * 4;
        delta_index++;
      }

      if ( component->flags & VARC_HAVE_TRANSLATE_Y )
      {
        /* FWORD: 4+2=6 */
        component->translate_y += deltas[delta_index] * 4;
        delta_index++;
      }

      if ( component->flags & VARC_HAVE_ROTATION )
      {
        /* F4DOT12: already 16.16 */
        component->rotation += deltas[delta_index];
        delta_index++;
      }

      if ( component->flags & VARC_HAVE_SCALE_X )
      {
        /* F6DOT10: 4+2=6 */
        component->scale_x += deltas[delta_index] * 4;
        delta_index++;
      }

      if ( component->flags & VARC_HAVE_SCALE_Y )
      {
        /* F6DOT10: 4+2=6 */
        component->scale_y += deltas[delta_index] * 4;
        delta_index++;
      }

      if ( component->flags & VARC_HAVE_SKEW_X )
      {
        /* F4DOT12: already 16.16 */
        component->skew_x += deltas[delta_index];
        delta_index++;
      }

      if ( component->flags & VARC_HAVE_SKEW_Y )
      {
        /* F4DOT12: already 16.16 */
        component->skew_y += deltas[delta_index];
        delta_index++;
      }

      if ( component->flags & VARC_HAVE_TCENTER_X )
      {
        /* FWORD: 4+2=6 */
        component->tcenter_x += deltas[delta_index] * 4;
        delta_index++;
      }

      if ( component->flags & VARC_HAVE_TCENTER_Y )
      {
        /* FWORD: 4+2=6 */
        component->tcenter_y += deltas[delta_index] * 4;
        delta_index++;
      }

      /* When ScaleY is absent it mirrors ScaleX.  Re-apply the mirror */
      /* after variation so a varying uniform scale stays uniform,     */
      /* matching fontTools and skrifa (which redo this post-delta).   */
      if ( ( component->flags & VARC_HAVE_SCALE_X )  &&
           !( component->flags & VARC_HAVE_SCALE_Y ) )
        component->scale_y = component->scale_x;
    }
#ifdef FT_DEBUG_LEVEL_TRACE
    else
      FT_TRACE2(( "tt_varc_apply_transform_deltas: variation index"
                  " 0x%08lx failed (error 0x%x)\n",
                  (FT_ULong)component->transform_var_index, error ));
#endif

    if ( deltas != stack_deltas )
      FT_FREE( deltas );

#else /* !TT_CONFIG_OPTION_GX_VAR_SUPPORT */

    FT_UNUSED( face );
    FT_UNUSED( varc );
    FT_UNUSED( component );
    FT_UNUSED( current_coords );
    FT_UNUSED( num_coords );

#endif /* !TT_CONFIG_OPTION_GX_VAR_SUPPORT */
  }


  /**************************************************************************
   *
   * Transform Matrix Functions
   *
   */

  /**************************************************************************
   *
   * @Function:
   *   tt_varc_build_transform
   *
   * @Description:
   *   Build a transformation matrix from component transform values.
   *
   *   Transformation order (per VARC spec):
   *
   *     1. translate to transformation center (-tCenterX, -tCenterY)
   *     2. apply scale (scaleX, scaleY)
   *     3. apply skew (skewX, skewY)
   *     4. apply rotation (rotation * pi)
   *     5. translate back from center (tCenterX, tCenterY)
   *     6. apply final translation (translateX, translateY)
   *
   * @Input:
   *   component ::
   *     The component with transform values.
   *
   * @Output:
   *   matrix ::
   *     The resulting 2x2 transformation matrix.
   *
   *   offset ::
   *     The translation vector.
   */
  static void
  tt_varc_build_transform( TT_VarcComponent  component,
                           FT_Matrix*        matrix,
                           FT_Vector*        offset )
  {
    FT_Fixed  scale_x  = component->scale_x;
    FT_Fixed  scale_y  = component->scale_y;
    FT_Fixed  skew_x   = component->skew_x;
    FT_Fixed  skew_y   = component->skew_y;
    FT_Fixed  rotation = component->rotation;


    /* Build rotation matrix */
    if ( rotation != 0 )
    {
      FT_Angle   angle;
      FT_Vector  v;


      /* Rotation is F4DOT12 in multiples of pi, converted to F16DOT16 by */
      /* << 4.  FT_ANGLE_PI = 180 * 65536 (degrees), so this converts pi  */
      /* multiples to degrees.                                            */
      angle = FT_MulFix( rotation, FT_ANGLE_PI );

      FT_Vector_Unit( &v, angle );

      /* Rotation matrix: [cos -sin]
                          [sin  cos] */
      matrix->xx =  v.x;
      matrix->xy = -v.y;
      matrix->yx =  v.y;
      matrix->yy =  v.x;
    }
    else
    {
      matrix->xx = 0x10000;
      matrix->xy = 0;
      matrix->yx = 0;
      matrix->yy = 0x10000;
    }

    /* Apply scale - scale each column, not row.                      */
    /* First column (xx, yx) represents X-axis, scaled by `scale_x`.  */
    /* Second column (xy, yy) represents Y-axis, scaled by `scale_y`. */
    if ( scale_x != 0x10000 || scale_y != 0x10000 )
    {
      matrix->xx = FT_MulFix( matrix->xx, scale_x );
      matrix->yx = FT_MulFix( matrix->yx, scale_x );
      matrix->xy = FT_MulFix( matrix->xy, scale_y );
      matrix->yy = FT_MulFix( matrix->yy, scale_y );
    }

    /* apply skew */
    if ( skew_x != 0 || skew_y != 0 )
    {
      FT_Matrix  skew_matrix;
      FT_Fixed   tan_x, tan_y;


      /* Convert skew angles to tangent values.                         */
      /* Skew is in radians * pi, so we need to compute tan(skew * pi). */
      tan_x = skew_x ? FT_Tan( FT_MulFix( skew_x, FT_ANGLE_PI ) ) : 0;
      tan_y = skew_y ? FT_Tan( FT_MulFix( skew_y, FT_ANGLE_PI ) ) : 0;

      /* The transform uses skew(-skewX, skewY), so only the xy term */
      /* (the skewX contribution) is negated; the yx (skewY) term is */
      /* positive.  Matches HarfBuzz/skrifa/fontTools.               */
      skew_matrix.xx =  0x10000;
      skew_matrix.xy = -tan_x;
      skew_matrix.yx =  tan_y;
      skew_matrix.yy =  0x10000;

      /* Apply skew on the RIGHT so the linear part is Rot*Scale*Skew   */
      /* (skew applied to the point first).  `FT_Matrix_Multiply(a, b)` */
      /* computes b = a*b, so this leaves skew_matrix = matrix*skew;    */
      /* copy it back into matrix.                                      */
      FT_Matrix_Multiply( matrix, &skew_matrix );
      *matrix = skew_matrix;
    }

    /* calculate translation with center point transformation */
    offset->x = component->translate_x;
    offset->y = component->translate_y;

    if ( component->tcenter_x != 0 || component->tcenter_y != 0 )
    {
      FT_Vector  center;


      center.x = component->tcenter_x;
      center.y = component->tcenter_y;

      /* offset += center - matrix * center */
      offset->x += center.x - FT_MulFix( matrix->xx, center.x ) -
                              FT_MulFix( matrix->xy, center.y );
      offset->y += center.y - FT_MulFix( matrix->yx, center.x ) -
                              FT_MulFix( matrix->yy, center.y );
    }
  }


  /**************************************************************************
   *
   * Recursion Context Functions
   *
   */

  /**************************************************************************
   *
   * @Function:
   *   tt_varc_context_push
   *
   * @Description:
   *   Push a glyph ID onto the recursion stack.
   *
   * @Input:
   *   context ::
   *     The recursion context.
   *
   *   glyph_index ::
   *     The glyph ID to push.
   *
   * @Return:
   *   FreeType error code.  Returns `FT_Err_Invalid_Table` if cycle
   *   detected or max depth exceeded.
   */
  static FT_Error
  tt_varc_context_push( TT_VarcContext  context,
                        FT_UInt         glyph_index )
  {
    FT_UInt  i;


    /* check max recursion depth */
    if ( context->recursion_depth >= TT_VARC_MAX_NESTING_LEVEL )
    {
      FT_TRACE2(( "tt_varc_context_push: nesting limit reached"
                  " for glyph %u\n",
                  glyph_index ));
      return FT_THROW( Invalid_Table );
    }

    /* check for cycles */
    for ( i = 0; i < context->recursion_depth; i++ )
    {
      if ( context->visited_glyphs[i] == glyph_index )
      {
        FT_TRACE2(( "tt_varc_context_push: cycle detected at glyph %u\n",
                    glyph_index ));
        return FT_THROW( Invalid_Table ); /* cycle detected */
      }
    }

    /* push onto stack */
    context->visited_glyphs[context->recursion_depth] = glyph_index;
    context->recursion_depth++;

    return FT_Err_Ok;
  }


  /**************************************************************************
   *
   * @Function:
   *   tt_varc_context_pop
   *
   * @Description:
   *   Pop a glyph ID from the recursion stack.
   *
   * @Input:
   *   context ::
   *     The recursion context.
   */
  static void
  tt_varc_context_pop( TT_VarcContext  context )
  {
    if ( context->recursion_depth > 0 )
      context->recursion_depth--;
  }


  /**************************************************************************
   *
   * VARC Table Loading Functions
   *
   */

  /**************************************************************************
   *
   * @Function:
   *   tt_face_load_varc
   *
   * @Description:
   *   Load the 'VARC' table from the font file.
   *
   * @Input:
   *   face ::
   *     The font face.
   *
   *   stream ::
   *     The font stream.
   *
   * @Return:
   *   FreeType error code.  0 means success.
   */
  FT_LOCAL_DEF( FT_Error )
  tt_face_load_varc( FT_Face    ftface,
                     FT_Stream  stream )
  {
    TT_Face    face   = (TT_Face)ftface;
    FT_Memory  memory = face->root.memory;
    FT_Error   error;

    FT_Byte*   table = NULL;
    FT_ULong   table_size;
    TT_Varc    varc  = NULL;
    FT_Byte*   p;


    /* Initialize VARC loading flag */
    face->varc_loading_components = FALSE;
    face->varc_context            = NULL;

    FT_TRACE2(( "VARC " ));

    error = face->goto_table( face, TTAG_VARC, stream, &table_size );
    if ( error )
    {
      FT_TRACE2(( "is missing\n" ));
      goto Exit;
    }
    if ( table_size < 24 ) /* minimum header size */
    {
      FT_TRACE2(( "is too short\n" ));
      goto Exit;
    }

    if ( FT_FRAME_EXTRACT( table_size, table ) )
    {
      FT_TRACE2(( "could not be read\n" ));
      goto Exit;
    }

    if ( FT_NEW( varc ) )
    {
      FT_TRACE2(( "could not be loaded\n" ));
      goto Fail;
    }

    varc->table      = table;
    varc->table_size = table_size;

    p = table;

    /* read header */
    varc->version_major = ( p[0] << 8 ) | p[1];
    varc->version_minor = ( p[2] << 8 ) | p[3];
    p += 4;

    /* check version */
    if ( varc->version_major != 1 || varc->version_minor != 0 )
    {
      FT_TRACE2(( "has unsupported version %u.%u\n",
                  varc->version_major, varc->version_minor ));
      error = FT_THROW( Invalid_Table );
      goto Fail;
    }

    /* read offsets */
    varc->coverage_offset = ( (FT_ULong)p[0] << 24 ) |
                            ( (FT_ULong)p[1] << 16 ) |
                            ( (FT_ULong)p[2] << 8  ) |
                              (FT_ULong)p[3];
    p += 4;

    varc->multi_var_store_offset = ( (FT_ULong)p[0] << 24 ) |
                                   ( (FT_ULong)p[1] << 16 ) |
                                   ( (FT_ULong)p[2] << 8  ) |
                                     (FT_ULong)p[3];
    p += 4;

    varc->condition_list_offset = ( (FT_ULong)p[0] << 24 ) |
                                  ( (FT_ULong)p[1] << 16 ) |
                                  ( (FT_ULong)p[2] << 8  ) |
                                    (FT_ULong)p[3];
    p += 4;

    varc->axis_indices_list_offset = ( (FT_ULong)p[0] << 24 ) |
                                     ( (FT_ULong)p[1] << 16 ) |
                                     ( (FT_ULong)p[2] << 8  ) |
                                       (FT_ULong)p[3];
    p += 4;

    varc->var_composite_glyphs_offset = ( (FT_ULong)p[0] << 24 ) |
                                        ( (FT_ULong)p[1] << 16 ) |
                                        ( (FT_ULong)p[2] << 8  ) |
                                          (FT_ULong)p[3];

    /* validate offsets */
    if ( varc->coverage_offset >= table_size             ||
         varc->var_composite_glyphs_offset >= table_size )
    {
      FT_TRACE2(( "has invalid offsets\n" ));
      error = FT_THROW( Invalid_Table );
      goto Fail;
    }

    /* set up pointers */
    varc->coverage             = table + varc->coverage_offset;
    varc->var_composite_glyphs = table + varc->var_composite_glyphs_offset;

    if ( varc->condition_list_offset > 0          &&
         varc->condition_list_offset < table_size )
      varc->condition_list = table + varc->condition_list_offset;
    else
      varc->condition_list = NULL;

    if ( varc->axis_indices_list_offset > 0          &&
         varc->axis_indices_list_offset < table_size )
      varc->axis_indices_list = table + varc->axis_indices_list_offset;
    else
      varc->axis_indices_list = NULL;

#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT

    /* `MultiItemVariationStore` is NOT the same as `ItemVariationStore`! */
    /* `MultiItemVariationStore` stores TUPLES of deltas,                 */
    /*  not single deltas.                                                */
    varc->multi_var_store_loaded = FALSE;
    varc->multi_var_store        = NULL;

    if ( varc->multi_var_store_offset > 0          &&
         varc->multi_var_store_offset < table_size )
    {
      /* point to the `MultiItemVariationStore` data in the 'VARC' table */
      varc->multi_var_store        = table + varc->multi_var_store_offset;
      varc->multi_var_store_loaded = TRUE;

      /* TODO: Implement proper MultiItemVariationStore parser:
       * - Format: u16 (must be 1)
       * - Regions: Offset32 to SparseVarRegionList
       * - DataSets: Array16 of Offset32 to MultiVarData
       * MultiVarData format:
       * - Format: u8 (must be 1)
       * - RegionIndices: Array16 of u16
       * - DeltaSets: INDEX2 of TupleValues (CFF2-style index)
       */
    }

#endif /* TT_CONFIG_OPTION_GX_VAR_SUPPORT */

    face->varc = varc;

    FT_TRACE2(( "loaded\n" ));
    FT_TRACE4(( "  version %u.%u, %lu bytes\n",
                varc->version_major,
                varc->version_minor,
                varc->table_size ));
    FT_TRACE4(( "  offsets: coverage %lu, variation store %lu,\n",
                varc->coverage_offset,
                varc->multi_var_store_offset ));
    FT_TRACE4(( "           conditions %lu, axis indices %lu, glyphs %lu\n",
                varc->condition_list_offset,
                varc->axis_indices_list_offset,
                varc->var_composite_glyphs_offset ));
    goto Exit;

  Fail:
    FT_FREE( varc );
    FT_FRAME_RELEASE( table );

  Exit:
    return error;
  }


  /**************************************************************************
   *
   * @Function:
   *   tt_face_free_varc
   *
   * @Description:
   *   Free the 'VARC' table resources.
   *
   * @Input:
   *   face ::
   *     The font face.
   */
  FT_LOCAL_DEF( void )
  tt_face_free_varc( FT_Face  ftface )
  {
    TT_Face    face   = (TT_Face)ftface;
    FT_Memory  memory = face->root.memory;
    FT_Stream  stream = face->root.stream;
    TT_Varc    varc   = (TT_Varc)face->varc;


    if ( varc )
    {
#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
      /* `MultiItemVariationStore` cleanup.         */
      /* No special cleanup needed -                */
      /* we just point to data in the 'VARC' table. */
      varc->multi_var_store        = NULL;
      varc->multi_var_store_loaded = FALSE;
#endif

      FT_FRAME_RELEASE( varc->table );
      FT_FREE( varc );
      face->varc = NULL;
    }
  }


  /**************************************************************************
   *
   * @Function:
   *   tt_face_has_varc_glyph
   *
   * @Description:
   *   Check whether a glyph has VARC data.
   *
   * @Input:
   *   face ::
   *     The font face.
   *
   *   glyph_index ::
   *     The glyph ID to check.
   *
   * @Return:
   *   TRUE if the glyph has VARC data, FALSE otherwise.
   */
  FT_LOCAL_DEF( FT_Bool )
  tt_face_has_varc_glyph( FT_Face  ftface,
                          FT_UInt  glyph_index )
  {
    TT_Face  face = (TT_Face)ftface;
    TT_Varc  varc = (TT_Varc)face->varc;
    FT_Int   coverage_idx;


    if ( !varc )
      return FALSE;

    coverage_idx = tt_varc_get_coverage( varc, glyph_index );
    return coverage_idx >= 0;
  }


  /**************************************************************************
   *
   * VARC Glyph Loading Functions
   *
   */

#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT

  /**************************************************************************
   *
   * @Function:
   *   tt_varc_set_normalized_coords
   *
   * @Description:
   *   Lightweight coordinate swap for VARC component loading.
   *   Unlike `FT_Set_Var_Blend_Coordinates`, this only sets the normalized
   *   coordinates and invalidates the gvar tuple scalar cache.  It skips
   *   CVT reload, MVAR, auto-hinter invalidation, PS name construction,
   *   and design coordinate computation - none of which are needed when
   *   loading VARC components with `FT_LOAD_NO_HINTING`.
   *
   * @Input:
   *   face ::
   *     The font face.
   *
   *   coords ::
   *     Current normalized coordinates (F2DOT14).
   *
   *   coord_count ::
   *     Number of coordinates.
   */
  static void
  tt_varc_set_normalized_coords( TT_Face    face,
                                 FT_Fixed*  coords,
                                 FT_UInt    num_coords )
  {
    GX_Blend  blend = face->blend;
    FT_UInt   i;


    if ( !blend || !blend->normalizedcoords )
      return;

    if ( num_coords > blend->num_axis )
      num_coords = blend->num_axis;

    FT_MEM_COPY( blend->normalizedcoords,
                 coords,
                 num_coords * sizeof ( FT_Fixed ) );

    /* zero out remaining axes */
    for ( i = num_coords; i < blend->num_axis; i++ )
      blend->normalizedcoords[i] = 0;

    /* update `doblend` flag */
    face->doblend = FALSE;
    for ( i = 0; i < blend->num_axis; i++ )
    {
      if ( blend->normalizedcoords[i] )
      {
        face->doblend = TRUE;
        break;
      }
    }

    /* Update face variation flag so IS_DEFAULT_INSTANCE is correct.   */
    /* Without this, `load_truetype_glyph` skips gvar deltas entirely. */
    if ( face->doblend )
      face->root.face_flags |= FT_FACE_FLAG_VARIATION;
    else
      face->root.face_flags &= ~FT_FACE_FLAG_VARIATION;

    /* invalidate gvar tuple scalar cache */
    for ( i = 0; i < blend->tuplecount; i++ )
      blend->tuplescalars[i] = (FT_Fixed)-0x20000L;
  }

#endif /* TT_CONFIG_OPTION_GX_VAR_SUPPORT */


  /**************************************************************************
   *
   * @Function:
   *   tt_face_load_varc_glyph
   *
   * @Description:
   *   Load a VARC glyph into the glyph slot.
   *
   * @Input:
   *   ftface ::
   *     The font face.
   *
   *   glyph_slot ::
   *     The glyph slot to load into.
   *
   *   glyph_index ::
   *     The glyph ID to load.
   *
   *   load_flags ::
   *     Loading flags.
   *
   * @Return:
   *   FreeType error code.  0 means success.
   */
  FT_LOCAL_DEF( FT_Error )
  tt_face_load_varc_glyph( FT_Face       ftface,
                           FT_GlyphSlot  glyph_slot,
                           FT_UInt       glyph_index,
                           FT_Int32      load_flags )
  {
    TT_Face    face   = (TT_Face)ftface;
    FT_Memory  memory = face->root.memory;
    FT_Error   error;

    TT_Varc   varc = (TT_Varc)face->varc;
    FT_Byte*  record_data;
    FT_UInt   record_size;

    FT_Byte*  p = NULL;
    FT_Byte*  limit;

    TT_VarcContext  context;
    FT_Bool         context_owner = FALSE;

    TT_GlyphSlot    slot = (TT_GlyphSlot)glyph_slot;
    FT_GlyphLoader  gloader;

    FT_Fixed*  parent_coords = NULL;
    FT_UInt    num_coords    = 0;
#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
    FT_Bool    static_axes   = FALSE;
#endif

    FT_Fixed  stack_new_coords[VARC_STACK_COORD_COUNT];
    FT_UInt   stack_axis_indices[VARC_STACK_INDICES_COUNT];
    FT_Fixed  stack_axis_values[VARC_STACK_AXIS_COUNT];

    FT_Matrix  saved_transform_matrix;
    FT_Vector  saved_transform_delta;
    FT_UInt    saved_transform_flags;


    if ( !varc )
      return FT_THROW( Invalid_Table );

    /* get or create recursion context */
    context = (TT_VarcContext)face->varc_context;
    if ( !context )
    {
      /* first call - create new context on heap */
      if ( FT_NEW( context ) )
        return error;

      FT_ZERO( context );
      face->varc_context = context;
      context_owner = TRUE;

      /* save font's current variation coordinates */
      /* for inheritance at depth 1                */
#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT

      {
        FT_MM_Var*  master;


        /* `FT_Get_MM_Var` ensures `face->blend` is initialized */
        /* (it's lazy)                                          */
        error = FT_Get_MM_Var( (FT_Face)face, &master );
        if ( !error && master )
        {
          if ( master->num_axis > 0 )
            context->num_font_coords = master->num_axis;
          FT_Done_MM_Var( face->root.driver->root.library,
                          master );
        }
        else if ( error                                     &&
                  !FT_HAS_MULTIPLE_MASTERS( (FT_Face)face ) )
        {
          if ( !face->blend )
            (void)tt_var_init_gvar( (FT_Face)face );

          if ( face->blend && !face->blend->mmvar )
          {
            (void)tt_get_var_blend( (FT_Face)face,
                                    &context->num_font_coords,
                                    NULL,
                                    NULL,
                                    NULL );
            static_axes = TRUE;
          }
        }
        error = FT_Err_Ok;  /* non-fatal */
      }

      if ( context->num_font_coords > 0 )
      {
        if ( !FT_NEW_ARRAY( context->font_coords,
                            context->num_font_coords ) )
        {
          if ( static_axes )
            FT_MEM_COPY( context->font_coords,
                         face->blend->normalizedcoords,
                         context->num_font_coords * sizeof ( FT_Fixed ) );
          else
            error = FT_Get_Var_Blend_Coordinates(
                      (FT_Face)face,
                      context->num_font_coords,
                      context->font_coords );

          if ( error )
          {
            FT_FREE( context->font_coords );
            context->num_font_coords = 0;
          }
          else
          {
            FT_Generic  saved_autohint = face->root.autohint;


            /* set initial current_coords to font_coords */
            context->current_coords     = context->font_coords;
            context->num_current_coords = context->num_font_coords;

            if ( !static_axes )
            {
              /* Force blend initialization: allocate                */
              /* `normalizedcoords`, load 'gvar', etc.  This ensures */
              /* `tt_varc_set_normalized_coords` can work as a       */
              /* lightweight coordinate swapper later.               */
              face->root.autohint.data      = NULL;
              face->root.autohint.finalizer = NULL;

              FT_Set_Var_Blend_Coordinates( (FT_Face)face,
                                             context->num_font_coords,
                                             context->font_coords );

              face->root.autohint = saved_autohint;
            }
          }
        }
      }

#endif /* TT_CONFIG_OPTION_GX_VAR_SUPPORT */
    }

    /* Save the face's current transform (user's transform at top level,   */
    /* parent's VARC transform at recursive levels).  We restore or clear  */
    /* this before returning so `FT_Load_Glyph`'s post-processing does the */
    /* right thing.                                                        */
    saved_transform_matrix = face->root.internal->transform_matrix;
    saved_transform_delta  = face->root.internal->transform_delta;
    saved_transform_flags  = face->root.internal->transform_flags;

    /* get parent coords from context for delta evaluation */
    parent_coords = context->current_coords;
    num_coords    = context->num_current_coords;

    /* push this glyph onto recursion stack */
    error = tt_varc_context_push( context, glyph_index );
    if ( error )
    {
      if ( context_owner )
        face->varc_context = NULL;
      return error;
    }

    /* get glyph record */
    error = tt_varc_get_glyph_record( varc, glyph_index,
                                      &record_data, &record_size );
    if ( error )
      goto Cleanup;

    p     = record_data;
    limit = record_data + record_size;

    FT_TRACE4(( "VARC glyph %u: nesting level %u, %u-byte record\n",
                glyph_index, context->recursion_depth, record_size ));

    /* Allocate a reusable temp glyph slot for component loading. */
    /* `FT_New_GlyphSlot` prepends to `face->glyph` list, so      */
    /* `FT_Load_Glyph` will load into this slot.  One slot per    */
    /* recursion level suffices.                                  */
    {
      FT_GlyphSlot  temp_glyph;


      error = FT_New_GlyphSlot( (FT_Face)face, &temp_glyph );
      if ( error )
        goto Cleanup;
    }

    /* Accumulate into the slot's own loader, like the glyf composite */
    /* path; the slot owns this storage and frees it on destruction.  */
    gloader = slot->internal->loader;
    FT_GlyphLoader_Rewind( gloader );

    /* parse components until we run out of data */
    while ( p < limit )
    {
      TT_VarcComponentRec  component;
      FT_GlyphSlot         component_slot;

      FT_Matrix  matrix;
      FT_Vector  offset;
      FT_Matrix  composed_matrix;
      FT_Vector  composed_delta;

      FT_Matrix  saved_parent_matrix;
      FT_Vector  saved_parent_delta;
      FT_Bool    saved_has_parent;

      FT_Fixed*  new_coords;
      FT_Bool    new_coords_on_heap;
      FT_Bool    has_axis_override;


      /* parse component */
      error = tt_varc_parse_component( face, varc,
                                       &p, limit,
                                       &component,
                                       stack_axis_values,
                                       VARC_STACK_AXIS_COUNT );
      if ( error )
      {
        /* skip malformed component and continue with others */
        FT_TRACE2(( "tt_face_load_varc_glyph: malformed component"
                    " in glyph %u (error 0x%x)\n",
                    glyph_index, error ));
        break;  /* exit loop - can't safely continue parsing */
                /* if we don't know component size           */
      }

      FT_TRACE5(( "  component glyph %u, flags 0x%08lx\n",
                  component.gid, (FT_ULong)component.flags ));

      /* Evaluate the component's condition, if any.  When it is not  */
      /* satisfied, the component is not rendered, matching HarfBuzz, */
      /* skrifa, and fontTools.                                       */
      if ( component.flags & VARC_HAVE_CONDITION )
      {
        FT_Byte*  table_limit = (FT_Byte*)varc->table + varc->table_size;
        FT_Byte*  cond        =
                    tt_varc_get_condition( varc,
                                           component.condition_index,
                                           table_limit );


#ifdef FT_DEBUG_LEVEL_TRACE
        if ( !cond )
          FT_TRACE2(( "  component glyph %u has invalid condition %lu;"
                      " ignoring condition\n",
                      component.gid,
                      (FT_ULong)component.condition_index ));
#endif
        if ( cond                                                    &&
             !tt_varc_eval_condition( face, varc, cond, table_limit,
                                      parent_coords, num_coords, 0 ) )
        {
          FT_TRACE5(( "    condition %lu is false; skipping component\n",
                      (FT_ULong)component.condition_index ));
          tt_varc_free_component( face, &component );
          continue;
        }
#ifdef FT_DEBUG_LEVEL_TRACE
        else
          FT_TRACE6(( "    condition %lu is true\n",
                      (FT_ULong)component.condition_index ));
#endif
      }

      /* apply variation deltas to axis values if present */
      if ( component.flags & VARC_AXES_HAVE_VARIATION )
        tt_varc_apply_axis_deltas( face, varc, &component,
                                   parent_coords, num_coords );

      /* apply variation deltas to transform values if present */
      if ( component.flags & VARC_TRANSFORM_HAS_VARIATION )
        tt_varc_apply_transform_deltas( face, varc, &component,
                                        parent_coords, num_coords );

      /* build transformation matrix */
      tt_varc_build_transform( &component, &matrix, &offset );

      /* apply axis value overrides if present */
      new_coords         = NULL;
      new_coords_on_heap = FALSE;
      has_axis_override  = FALSE;

#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT

      if ( ( ( component.num_axis_values > 0 && component.axis_values ) ||
             ( component.flags & VARC_RESET_UNSPECIFIED_AXES )          ) &&
           num_coords > 0                                                 &&
           parent_coords                                                  )
      {
        /* allocate new_coords only                  */
        /* (`parent_coords` replaces `saved_coords`) */
        if ( num_coords <= VARC_STACK_COORD_COUNT )
          new_coords = stack_new_coords;
        else
        {
          if ( FT_NEW_ARRAY( new_coords, num_coords ) )
            goto Skip_Axis_Override;
          new_coords_on_heap = TRUE;
        }

        /* start with current coordinates or reset to default */
        if ( component.flags & VARC_RESET_UNSPECIFIED_AXES )
        {
          /* Reset unspecified axes to the font's current instance      */
          /* coordinates, matching HarfBuzz (font->coords), skrifa, and */
          /* fontTools.  Axes explicitly specified below are overridden */
          /* afterwards.                                                */
          FT_UInt  j;
          FT_UInt  ncopy = context->num_font_coords < num_coords
                             ? context->num_font_coords : num_coords;


          if ( context->font_coords )
            FT_MEM_COPY( new_coords, context->font_coords,
                         ncopy * sizeof ( FT_Fixed ) );
          else
            ncopy = 0;

          for ( j = ncopy; j < num_coords; j++ )
            new_coords[j] = 0;  /* default for axes beyond `font_coords` */
        }
        else
        {
          /* inherit from parent coords (handles both depth 1 and > 1) */
          FT_MEM_COPY( new_coords, parent_coords,
                       num_coords * sizeof ( FT_Fixed ) );
        }

        /* read axis indices and apply values */
        if ( component.flags & VARC_HAVE_AXES &&
             varc->axis_indices_list          )
        {
          FT_UInt*  axis_indices = NULL;
          FT_UInt   j;


          /* allocate array for axis indices */
          if ( component.num_axis_values <= VARC_STACK_INDICES_COUNT )
            axis_indices = stack_axis_indices;
          else
          {
            if ( FT_NEW_ARRAY( axis_indices, component.num_axis_values ) )
            {
              if ( new_coords_on_heap )
                FT_FREE( new_coords );
              new_coords = NULL;
              goto Skip_Axis_Override;
            }
          }

          /* read axis indices from TupleList */
          error = tt_varc_get_axis_indices( varc,
                                            component.axis_indices_index,
                                            component.num_axis_values,
                                            axis_indices );
          if ( error )
          {
            FT_TRACE2(( "  component glyph %u has invalid axis indices"
                        " (error 0x%x)\n",
                        component.gid, error ));
            if ( axis_indices != stack_axis_indices )
              FT_FREE( axis_indices );
            if ( new_coords_on_heap )
              FT_FREE( new_coords );
            new_coords = NULL;
            goto Skip_Axis_Override;
          }

          /* apply axis values using the indices */
          for ( j = 0; j < component.num_axis_values; j++ )
          {
            FT_UInt   axis_idx = axis_indices[j];
            FT_Fixed  v        = component.axis_values[j];


            if ( axis_idx < num_coords )
            {
              /* Round to the nearest F2DOT14 grid point (a multiple of 4 */
              /* in 16.16) before the value feeds the leaf's gvar         */
              /* interpolation, matching HarfBuzz (roundf) and skrifa.    */
              if ( v >= 0 )
                v = ( v + 2 ) & ~(FT_Fixed)3;
              else
                v = NEG_LONG( ( NEG_LONG( v ) + 2 ) & ~(FT_Fixed)3 );

              new_coords[axis_idx] = v;
              FT_TRACE7(( "    axis %u: %.5f\n",
                          axis_idx, (double)v / 65536 ));
            }
#ifdef FT_DEBUG_LEVEL_TRACE
            else
              FT_TRACE2(( "  component glyph %u has out-of-range axis %u\n",
                          component.gid, axis_idx ));
#endif
          }

          if ( axis_indices != stack_axis_indices )
            FT_FREE( axis_indices );
        }

        /* Apply the new normalized coordinates.  Use lightweight path  */
        /* that only swaps coords and invalidates the gvar tuple cache. */
        tt_varc_set_normalized_coords( face, new_coords, num_coords );
        has_axis_override = TRUE;
      }

    Skip_Axis_Override:

#endif /* TT_CONFIG_OPTION_GX_VAR_SUPPORT */

      /* Load component - allow recursive VARC loading.              */
      /* Compose transforms: we want (T1*T2)(leaf) not T1(T2(leaf)). */

      /* save current parent transform */
      saved_parent_matrix = context->parent_matrix;
      saved_parent_delta  = context->parent_delta;
      saved_has_parent    = context->has_parent_transform;

      /* compose with parent transform if present */
      if ( context->has_parent_transform )
      {
        /* Compose: final = parent * child.  `FT_Matrix_Multiply(a, b)` */
        /* computes b = a*b, so seed the result with the child matrix   */
        /* and left-multiply by the parent; this matches the            */
        /* composed_delta below (the translation of parent*child) and   */
        /* HarfBuzz/skrifa.                                             */
        composed_matrix = matrix;
        FT_Matrix_Multiply( &context->parent_matrix, &composed_matrix );

        /* transform offset by parent matrix and add parent delta */
        composed_delta.x = FT_MulFix( offset.x, context->parent_matrix.xx ) +
                           FT_MulFix( offset.y, context->parent_matrix.xy ) +
                           context->parent_delta.x;
        composed_delta.y = FT_MulFix( offset.x, context->parent_matrix.yx ) +
                           FT_MulFix( offset.y, context->parent_matrix.yy ) +
                           context->parent_delta.y;
      }
      else
      {
        /* no parent transform, use component transform directly */
        composed_matrix = matrix;
        composed_delta  = offset;
      }

      /* store composed transform in context for child VARC components */
      context->parent_matrix        = composed_matrix;
      context->parent_delta         = composed_delta;
      context->has_parent_transform = TRUE;

      FT_TRACE7(( "    matrix: [ %.5f %.5f; %.5f %.5f ]\n",
                  (double)composed_matrix.xx / 65536,
                  (double)composed_matrix.xy / 65536,
                  (double)composed_matrix.yx / 65536,
                  (double)composed_matrix.yy / 65536 ));
      FT_TRACE7(( "    offset: (%.5f, %.5f)\n",
                  (double)composed_delta.x / 64,
                  (double)composed_delta.y / 64 ));


      /* set child's coords in context for recursive VARC processing */
#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
      if ( has_axis_override )
      {
        context->current_coords     = new_coords;
        context->num_current_coords = num_coords;
      }
#endif

      /* Set composed VARC transform on face so `FT_Load_Glyph` applies    */
      /* it.  The matrix is dimensionless (16.16) and works in any         */
      /* coordinate system.  The delta must match the outline's coordinate */
      /* space: 26.6 device pixels (scaled) or integer font units          */
      /* (NO_SCALE).                                                       */
      {
        FT_Face_Internal  internal = face->root.internal;


        internal->transform_matrix = composed_matrix;
        internal->transform_flags  = 0;

        if ( ( composed_matrix.xy | composed_matrix.yx ) ||
             composed_matrix.xx != 0x10000L              ||
             composed_matrix.yy != 0x10000L              )
          internal->transform_flags |= 1;

        if ( load_flags & FT_LOAD_NO_SCALE )
        {
          /* NO_SCALE: outline in integer font units */
          internal->transform_delta.x = ( composed_delta.x + 32 ) >> 6;
          internal->transform_delta.y = ( composed_delta.y + 32 ) >> 6;
        }
        else
        {
          FT_Size  size = face->root.size;


          if ( size )
          {
            FT_Fixed  x_scale = size->metrics.x_scale;
            FT_Fixed  y_scale = size->metrics.y_scale;


            /* Convert from 26.6 font units to 26.6 device pixels. */
            /* 26.6 * 16.16 = 42.22; >> 22 gives 26.6.             */
#ifdef FT_INT64
            internal->transform_delta.x =
              (FT_Pos)( ( (FT_Int64)composed_delta.x * x_scale +
                          0x200000L ) >> 22 );
            internal->transform_delta.y =
              (FT_Pos)( ( (FT_Int64)composed_delta.y * y_scale +
                          0x200000L ) >> 22 );
#else
            internal->transform_delta.x =
              FT_MulDiv( composed_delta.x, x_scale, 0x400000L );
            internal->transform_delta.y =
              FT_MulDiv( composed_delta.y, y_scale, 0x400000L );
#endif
          }
          else
          {
            internal->transform_delta.x = ( composed_delta.x + 32 ) >> 6;
            internal->transform_delta.y = ( composed_delta.y + 32 ) >> 6;
          }
        }

        if ( internal->transform_delta.x | internal->transform_delta.y )
          internal->transform_flags |= 2;
      }

      /* Load component.  Strip IGNORE_TRANSFORM so our VARC transform    */
      /* is applied by `FT_Load_Glyph`'s post-processing.  Add NO_HINTING */
      /* to avoid auto-fitter reentrancy: the auto-fitter initializes     */
      /* per-face metrics lazily, and `FT_Set_Var_Blend_Coordinates`      */
      /* (called below to change axis values) frees that state, causing   */
      /* use-after-free if the auto-fitter was mid-initialization.        */
      {
        FT_Int32  component_load_flags =
                    ( load_flags & ~(FT_Int32)FT_LOAD_IGNORE_TRANSFORM ) |
                    FT_LOAD_NO_HINTING;


        error = FT_Load_Glyph( (FT_Face)face, component.gid,
                                component_load_flags );
      }

#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
      /* restore parent coordinates */
      if ( has_axis_override )
      {
        tt_varc_set_normalized_coords( face, parent_coords, num_coords );

        context->current_coords     = parent_coords;
        context->num_current_coords = num_coords;

        if ( new_coords_on_heap )
          FT_FREE( new_coords );
      }
#endif

      /* restore parent transform in context */
      context->parent_matrix        = saved_parent_matrix;
      context->parent_delta         = saved_parent_delta;
      context->has_parent_transform = saved_has_parent;

      component_slot = face->root.glyph;

      if ( error )
      {
        FT_TRACE2(( "  component glyph %u failed (error 0x%x); skipped\n",
                    component.gid, error ));
        tt_varc_free_component( face, &component );
        continue;  /* skip failed component, continue with others */
      }

      /* `FT_Load_Glyph` already applied our composed transform.
       *
       * For base glyphs: the driver loaded and scaled, then
       * `FT_Load_Glyph` applied our matrix + delta.
       *
       * For VARC sub-components: the recursive handler processed
       * them and cleared the face transform before returning, so
       * `FT_Load_Glyph`'s post-processing was a no-op.
       */
      if ( component_slot->outline.n_points > 0 )
      {
        FT_Outline*  src = &component_slot->outline;
        FT_Outline*  cur = &gloader->current.outline;


        /* ensure capacity in gloader with geometric growth */
        error = FT_GLYPHLOADER_CHECK_POINTS( gloader,
                                             src->n_points,
                                             src->n_contours );
        if ( error )
        {
          tt_varc_free_component( face, &component );
          goto Cleanup;
        }

        FT_ARRAY_COPY( cur->points, src->points, src->n_points );
        FT_ARRAY_COPY( cur->tags, src->tags, src->n_points );
        FT_ARRAY_COPY( cur->contours, src->contours, src->n_contours );

        cur->n_points   = src->n_points;
        cur->n_contours = src->n_contours;

        FT_GlyphLoader_Add( gloader );

        FT_TRACE5(( "    appended %d points in %d contours\n",
                    src->n_points, src->n_contours ));
      }

      tt_varc_free_component( face, &component );
    }

    /* Hand the loader's accumulated outline to the slot. */
    {
      FT_Outline*  base = &gloader->base.outline;


      if ( base->n_points > 0 )
      {
        slot->outline = *base;
        slot->format  = FT_GLYPH_FORMAT_OUTLINE;
      }

      FT_TRACE4(( "VARC glyph %u: %d points in %d contours\n",
                  glyph_index, base->n_points, base->n_contours ));
    }

    error = FT_Err_Ok;

  Cleanup:
    /* Free the temp glyph slot for this recursion level.             */
    /* `face->root.glyph` points to the temp slot we allocated above; */
    /* `FT_Done_GlyphSlot` removes it and restores the previous head. */
    if ( face->root.glyph != glyph_slot )
      FT_Done_GlyphSlot( face->root.glyph );

    /* pop from recursion stack */
    tt_varc_context_pop( context );

    /* clean up context if we own it */
    if ( context_owner )
    {
#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
      /* Restore font's original coordinates in case we bailed out */
      /* mid-component with overridden axis values still active.   */
      if ( context->font_coords )
        tt_varc_set_normalized_coords( face,
                                       context->font_coords,
                                       context->num_font_coords );
#endif
      if ( context->font_coords )
        FT_FREE( context->font_coords );
      FT_FREE( context );
      face->varc_context = NULL;
    }

    /* Manage face transform for `FT_Load_Glyph`'s post-processing.    */
    /* At top level: restore user's transform so it gets applied once. */
    /* At recursive levels: clear to identity so the parent's          */
    /* `FT_Load_Glyph` doesn't apply twice.                            */
    if ( context_owner )
    {
      face->root.internal->transform_matrix = saved_transform_matrix;
      face->root.internal->transform_delta  = saved_transform_delta;
      face->root.internal->transform_flags  = saved_transform_flags;
    }
    else
    {
      face->root.internal->transform_matrix.xx = 0x10000L;
      face->root.internal->transform_matrix.xy = 0;
      face->root.internal->transform_matrix.yx = 0;
      face->root.internal->transform_matrix.yy = 0x10000L;
      face->root.internal->transform_delta.x   = 0;
      face->root.internal->transform_delta.y   = 0;
      face->root.internal->transform_flags     = 0;
    }

#ifdef FT_DEBUG_LEVEL_TRACE
    if ( error )
      FT_TRACE2(( "tt_face_load_varc_glyph: glyph %u failed"
                  " (error 0x%x)\n",
                  glyph_index, error ));
#endif

    return error;
  }


#endif /* TT_CONFIG_OPTION_VARC */


/* END */
