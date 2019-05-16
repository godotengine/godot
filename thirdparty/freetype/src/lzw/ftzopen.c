/****************************************************************************
 *
 * ftzopen.c
 *
 *   FreeType support for .Z compressed files.
 *
 * This optional component relies on NetBSD's zopen().  It should mainly
 * be used to parse compressed PCF fonts, as found with many X11 server
 * distributions.
 *
 * Copyright (C) 2005-2019 by
 * David Turner.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */

#include "ftzopen.h"
#include FT_INTERNAL_MEMORY_H
#include FT_INTERNAL_STREAM_H
#include FT_INTERNAL_DEBUG_H


  static int
  ft_lzwstate_refill( FT_LzwState  state )
  {
    FT_ULong  count;


    if ( state->in_eof )
      return -1;

    count = FT_Stream_TryRead( state->source,
                               state->buf_tab,
                               state->num_bits );  /* WHY? */

    state->buf_size   = (FT_UInt)count;
    state->buf_total += count;
    state->in_eof     = FT_BOOL( count < state->num_bits );
    state->buf_offset = 0;

    state->buf_size <<= 3;
    if ( state->buf_size > state->num_bits )
      state->buf_size -= state->num_bits - 1;
    else
      return -1; /* not enough data */

    if ( count == 0 )  /* end of file */
      return -1;

    return 0;
  }


  static FT_Int32
  ft_lzwstate_get_code( FT_LzwState  state )
  {
    FT_UInt   num_bits = state->num_bits;
    FT_UInt   offset   = state->buf_offset;
    FT_Byte*  p;
    FT_Int    result;


    if ( state->buf_clear                    ||
         offset >= state->buf_size           ||
         state->free_ent >= state->free_bits )
    {
      if ( state->free_ent >= state->free_bits )
      {
        state->num_bits = ++num_bits;
        if ( num_bits > LZW_MAX_BITS )
          return -1;

        state->free_bits = state->num_bits < state->max_bits
                           ? (FT_UInt)( ( 1UL << num_bits ) - 256 )
                           : state->max_free + 1;
      }

      if ( state->buf_clear )
      {
        state->num_bits  = num_bits = LZW_INIT_BITS;
        state->free_bits = (FT_UInt)( ( 1UL << num_bits ) - 256 );
        state->buf_clear = 0;
      }

      if ( ft_lzwstate_refill( state ) < 0 )
        return -1;

      offset = 0;
    }

    state->buf_offset = offset + num_bits;

    p         = &state->buf_tab[offset >> 3];
    offset   &= 7;
    result    = *p++ >> offset;
    offset    = 8 - offset;
    num_bits -= offset;

    if ( num_bits >= 8 )
    {
      result   |= *p++ << offset;
      offset   += 8;
      num_bits -= 8;
    }
    if ( num_bits > 0 )
      result |= ( *p & LZW_MASK( num_bits ) ) << offset;

    return result;
  }


  /* grow the character stack */
  static int
  ft_lzwstate_stack_grow( FT_LzwState  state )
  {
    if ( state->stack_top >= state->stack_size )
    {
      FT_Memory  memory = state->memory;
      FT_Error   error;
      FT_Offset  old_size = state->stack_size;
      FT_Offset  new_size = old_size;

      new_size = new_size + ( new_size >> 1 ) + 4;

      if ( state->stack == state->stack_0 )
      {
        state->stack = NULL;
        old_size     = 0;
      }

      /* requirement of the character stack larger than 1<<LZW_MAX_BITS */
      /* implies bug in the decompression code                          */
      if ( new_size > ( 1 << LZW_MAX_BITS ) )
      {
        new_size = 1 << LZW_MAX_BITS;
        if ( new_size == old_size )
          return -1;
      }

      if ( FT_RENEW_ARRAY( state->stack, old_size, new_size ) )
        return -1;

      state->stack_size = new_size;
    }
    return 0;
  }


  /* grow the prefix/suffix arrays */
  static int
  ft_lzwstate_prefix_grow( FT_LzwState  state )
  {
    FT_UInt    old_size = state->prefix_size;
    FT_UInt    new_size = old_size;
    FT_Memory  memory   = state->memory;
    FT_Error   error;


    if ( new_size == 0 )  /* first allocation -> 9 bits */
      new_size = 512;
    else
      new_size += new_size >> 2;  /* don't grow too fast */

    /*
     * Note that the `suffix' array is located in the same memory block
     * pointed to by `prefix'.
     *
     * I know that sizeof(FT_Byte) == 1 by definition, but it is clearer
     * to write it literally.
     *
     */
    if ( FT_REALLOC_MULT( state->prefix, old_size, new_size,
                          sizeof ( FT_UShort ) + sizeof ( FT_Byte ) ) )
      return -1;

    /* now adjust `suffix' and move the data accordingly */
    state->suffix = (FT_Byte*)( state->prefix + new_size );

    FT_MEM_MOVE( state->suffix,
                 state->prefix + old_size,
                 old_size * sizeof ( FT_Byte ) );

    state->prefix_size = new_size;
    return 0;
  }


  FT_LOCAL_DEF( void )
  ft_lzwstate_reset( FT_LzwState  state )
  {
    state->in_eof     = 0;
    state->buf_offset = 0;
    state->buf_size   = 0;
    state->buf_clear  = 0;
    state->buf_total  = 0;
    state->stack_top  = 0;
    state->num_bits   = LZW_INIT_BITS;
    state->phase      = FT_LZW_PHASE_START;
  }


  FT_LOCAL_DEF( void )
  ft_lzwstate_init( FT_LzwState  state,
                    FT_Stream    source )
  {
    FT_ZERO( state );

    state->source = source;
    state->memory = source->memory;

    state->prefix      = NULL;
    state->suffix      = NULL;
    state->prefix_size = 0;

    state->stack      = state->stack_0;
    state->stack_size = sizeof ( state->stack_0 );

    ft_lzwstate_reset( state );
  }


  FT_LOCAL_DEF( void )
  ft_lzwstate_done( FT_LzwState  state )
  {
    FT_Memory  memory = state->memory;


    ft_lzwstate_reset( state );

    if ( state->stack != state->stack_0 )
      FT_FREE( state->stack );

    FT_FREE( state->prefix );
    state->suffix = NULL;

    FT_ZERO( state );
  }


#define FTLZW_STACK_PUSH( c )                        \
  FT_BEGIN_STMNT                                     \
    if ( state->stack_top >= state->stack_size &&    \
         ft_lzwstate_stack_grow( state ) < 0   )     \
      goto Eof;                                      \
                                                     \
    state->stack[state->stack_top++] = (FT_Byte)(c); \
  FT_END_STMNT


  FT_LOCAL_DEF( FT_ULong )
  ft_lzwstate_io( FT_LzwState  state,
                  FT_Byte*     buffer,
                  FT_ULong     out_size )
  {
    FT_ULong  result = 0;

    FT_UInt  old_char = state->old_char;
    FT_UInt  old_code = state->old_code;
    FT_UInt  in_code  = state->in_code;


    if ( out_size == 0 )
      goto Exit;

    switch ( state->phase )
    {
    case FT_LZW_PHASE_START:
      {
        FT_Byte   max_bits;
        FT_Int32  c;


        /* skip magic bytes, and read max_bits + block_flag */
        if ( FT_Stream_Seek( state->source, 2 ) != 0               ||
             FT_Stream_TryRead( state->source, &max_bits, 1 ) != 1 )
          goto Eof;

        state->max_bits   = max_bits & LZW_BIT_MASK;
        state->block_mode = max_bits & LZW_BLOCK_MASK;
        state->max_free   = (FT_UInt)( ( 1UL << state->max_bits ) - 256 );

        if ( state->max_bits > LZW_MAX_BITS )
          goto Eof;

        state->num_bits = LZW_INIT_BITS;
        state->free_ent = ( state->block_mode ? LZW_FIRST
                                              : LZW_CLEAR ) - 256;
        in_code  = 0;

        state->free_bits = state->num_bits < state->max_bits
                           ? (FT_UInt)( ( 1UL << state->num_bits ) - 256 )
                           : state->max_free + 1;

        c = ft_lzwstate_get_code( state );
        if ( c < 0 || c > 255 )
          goto Eof;

        old_code = old_char = (FT_UInt)c;

        if ( buffer )
          buffer[result] = (FT_Byte)old_char;

        if ( ++result >= out_size )
          goto Exit;

        state->phase = FT_LZW_PHASE_CODE;
      }
      /* fall-through */

    case FT_LZW_PHASE_CODE:
      {
        FT_Int32  c;
        FT_UInt   code;


      NextCode:
        c = ft_lzwstate_get_code( state );
        if ( c < 0 )
          goto Eof;

        code = (FT_UInt)c;

        if ( code == LZW_CLEAR && state->block_mode )
        {
          /* why not LZW_FIRST-256 ? */
          state->free_ent  = ( LZW_FIRST - 1 ) - 256;
          state->buf_clear = 1;

          /* not quite right, but at least more predictable */
          old_code = 0;
          old_char = 0;

          goto NextCode;
        }

        in_code = code; /* save code for later */

        if ( code >= 256U )
        {
          /* special case for KwKwKwK */
          if ( code - 256U >= state->free_ent )
          {
            /* corrupted LZW stream */
            if ( code - 256U > state->free_ent )
              goto Eof;

            FTLZW_STACK_PUSH( old_char );
            code = old_code;
          }

          while ( code >= 256U )
          {
            if ( !state->prefix )
              goto Eof;

            FTLZW_STACK_PUSH( state->suffix[code - 256] );
            code = state->prefix[code - 256];
          }
        }

        old_char = code;
        FTLZW_STACK_PUSH( old_char );

        state->phase = FT_LZW_PHASE_STACK;
      }
      /* fall-through */

    case FT_LZW_PHASE_STACK:
      {
        while ( state->stack_top > 0 )
        {
          state->stack_top--;

          if ( buffer )
            buffer[result] = state->stack[state->stack_top];

          if ( ++result == out_size )
            goto Exit;
        }

        /* now create new entry */
        if ( state->free_ent < state->max_free )
        {
          if ( state->free_ent >= state->prefix_size &&
               ft_lzwstate_prefix_grow( state ) < 0  )
            goto Eof;

          FT_ASSERT( state->free_ent < state->prefix_size );

          state->prefix[state->free_ent] = (FT_UShort)old_code;
          state->suffix[state->free_ent] = (FT_Byte)  old_char;

          state->free_ent += 1;
        }

        old_code = in_code;

        state->phase = FT_LZW_PHASE_CODE;
        goto NextCode;
      }

    default:  /* state == EOF */
      ;
    }

  Exit:
    state->old_code = old_code;
    state->old_char = old_char;
    state->in_code  = in_code;

    return result;

  Eof:
    state->phase = FT_LZW_PHASE_EOF;
    goto Exit;
  }


/* END */
