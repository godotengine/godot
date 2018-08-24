/***************************************************************************/
/*                                                                         */
/*  ftzopen.h                                                              */
/*                                                                         */
/*    FreeType support for .Z compressed files.                            */
/*                                                                         */
/*  This optional component relies on NetBSD's zopen().  It should mainly  */
/*  be used to parse compressed PCF fonts, as found with many X11 server   */
/*  distributions.                                                         */
/*                                                                         */
/*  Copyright 2005-2018 by                                                 */
/*  David Turner.                                                          */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/

#ifndef FTZOPEN_H_
#define FTZOPEN_H_

#include <ft2build.h>
#include FT_FREETYPE_H


  /*
   *  This is a complete re-implementation of the LZW file reader,
   *  since the old one was incredibly badly written, using
   *  400 KByte of heap memory before decompressing anything.
   *
   */

#define FT_LZW_IN_BUFF_SIZE        64
#define FT_LZW_DEFAULT_STACK_SIZE  64

#define LZW_INIT_BITS     9
#define LZW_MAX_BITS      16

#define LZW_CLEAR         256
#define LZW_FIRST         257

#define LZW_BIT_MASK      0x1F
#define LZW_BLOCK_MASK    0x80
#define LZW_MASK( n )     ( ( 1U << (n) ) - 1U )


  typedef enum  FT_LzwPhase_
  {
    FT_LZW_PHASE_START = 0,
    FT_LZW_PHASE_CODE,
    FT_LZW_PHASE_STACK,
    FT_LZW_PHASE_EOF

  } FT_LzwPhase;


  /*
   *  state of LZW decompressor
   *
   *  small technical note
   *  --------------------
   *
   *  We use a few tricks in this implementation that are explained here to
   *  ease debugging and maintenance.
   *
   *  - First of all, the `prefix' and `suffix' arrays contain the suffix
   *    and prefix for codes over 256; this means that
   *
   *      prefix_of(code) == state->prefix[code-256]
   *      suffix_of(code) == state->suffix[code-256]
   *
   *    Each prefix is a 16-bit code, and each suffix an 8-bit byte.
   *
   *    Both arrays are stored in a single memory block, pointed to by
   *    `state->prefix'.  This means that the following equality is always
   *    true:
   *
   *      state->suffix == (FT_Byte*)(state->prefix + state->prefix_size)
   *
   *    Of course, state->prefix_size is the number of prefix/suffix slots
   *    in the arrays, corresponding to codes 256..255+prefix_size.
   *
   *  - `free_ent' is the index of the next free entry in the `prefix'
   *    and `suffix' arrays.  This means that the corresponding `next free
   *    code' is really `256+free_ent'.
   *
   *    Moreover, `max_free' is the maximum value that `free_ent' can reach.
   *
   *    `max_free' corresponds to `(1 << max_bits) - 256'.  Note that this
   *    value is always <= 0xFF00, which means that both `free_ent' and
   *    `max_free' can be stored in an FT_UInt variable, even on 16-bit
   *    machines.
   *
   *    If `free_ent == max_free', you cannot add new codes to the
   *    prefix/suffix table.
   *
   *  - `num_bits' is the current number of code bits, starting at 9 and
   *    growing each time `free_ent' reaches the value of `free_bits'.  The
   *    latter is computed as follows
   *
   *      if num_bits < max_bits:
   *         free_bits = (1 << num_bits)-256
   *      else:
   *         free_bits = max_free + 1
   *
   *    Since the value of `max_free + 1' can never be reached by
   *    `free_ent', `num_bits' cannot grow larger than `max_bits'.
   */

  typedef struct  FT_LzwStateRec_
  {
    FT_LzwPhase  phase;
    FT_Int       in_eof;

    FT_Byte      buf_tab[16];
    FT_UInt      buf_offset;
    FT_UInt      buf_size;
    FT_Bool      buf_clear;
    FT_Offset    buf_total;

    FT_UInt      max_bits;    /* max code bits, from file header   */
    FT_Int       block_mode;  /* block mode flag, from file header */
    FT_UInt      max_free;    /* (1 << max_bits) - 256             */

    FT_UInt      num_bits;    /* current code bit number */
    FT_UInt      free_ent;    /* index of next free entry */
    FT_UInt      free_bits;   /* if reached by free_ent, increment num_bits */
    FT_UInt      old_code;
    FT_UInt      old_char;
    FT_UInt      in_code;

    FT_UShort*   prefix;      /* always dynamically allocated / reallocated */
    FT_Byte*     suffix;      /* suffix = (FT_Byte*)(prefix + prefix_size)  */
    FT_UInt      prefix_size; /* number of slots in `prefix' or `suffix'    */

    FT_Byte*     stack;       /* character stack */
    FT_UInt      stack_top;
    FT_Offset    stack_size;
    FT_Byte      stack_0[FT_LZW_DEFAULT_STACK_SIZE]; /* minimize heap alloc */

    FT_Stream    source;      /* source stream */
    FT_Memory    memory;

  } FT_LzwStateRec, *FT_LzwState;


  FT_LOCAL( void )
  ft_lzwstate_init( FT_LzwState  state,
                    FT_Stream    source );

  FT_LOCAL( void )
  ft_lzwstate_done( FT_LzwState  state );


  FT_LOCAL( void )
  ft_lzwstate_reset( FT_LzwState  state );


  FT_LOCAL( FT_ULong )
  ft_lzwstate_io( FT_LzwState  state,
                  FT_Byte*     buffer,
                  FT_ULong     out_size );

/* */

#endif /* FTZOPEN_H_ */


/* END */
