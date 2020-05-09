/****************************************************************************
 *
 * ftstream.h
 *
 *   Stream handling (specification).
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


#ifndef FTSTREAM_H_
#define FTSTREAM_H_


#include <ft2build.h>
#include FT_SYSTEM_H
#include FT_INTERNAL_OBJECTS_H


FT_BEGIN_HEADER


  /* format of an 8-bit frame_op value:           */
  /*                                              */
  /* bit  76543210                                */
  /*      xxxxxxes                                */
  /*                                              */
  /* s is set to 1 if the value is signed.        */
  /* e is set to 1 if the value is little-endian. */
  /* xxx is a command.                            */

#define FT_FRAME_OP_SHIFT         2
#define FT_FRAME_OP_SIGNED        1
#define FT_FRAME_OP_LITTLE        2
#define FT_FRAME_OP_COMMAND( x )  ( x >> FT_FRAME_OP_SHIFT )

#define FT_MAKE_FRAME_OP( command, little, sign ) \
          ( ( command << FT_FRAME_OP_SHIFT ) | ( little << 1 ) | sign )

#define FT_FRAME_OP_END    0
#define FT_FRAME_OP_START  1  /* start a new frame     */
#define FT_FRAME_OP_BYTE   2  /* read 1-byte value     */
#define FT_FRAME_OP_SHORT  3  /* read 2-byte value     */
#define FT_FRAME_OP_LONG   4  /* read 4-byte value     */
#define FT_FRAME_OP_OFF3   5  /* read 3-byte value     */
#define FT_FRAME_OP_BYTES  6  /* read a bytes sequence */


  typedef enum  FT_Frame_Op_
  {
    ft_frame_end       = 0,
    ft_frame_start     = FT_MAKE_FRAME_OP( FT_FRAME_OP_START, 0, 0 ),

    ft_frame_byte      = FT_MAKE_FRAME_OP( FT_FRAME_OP_BYTE,  0, 0 ),
    ft_frame_schar     = FT_MAKE_FRAME_OP( FT_FRAME_OP_BYTE,  0, 1 ),

    ft_frame_ushort_be = FT_MAKE_FRAME_OP( FT_FRAME_OP_SHORT, 0, 0 ),
    ft_frame_short_be  = FT_MAKE_FRAME_OP( FT_FRAME_OP_SHORT, 0, 1 ),
    ft_frame_ushort_le = FT_MAKE_FRAME_OP( FT_FRAME_OP_SHORT, 1, 0 ),
    ft_frame_short_le  = FT_MAKE_FRAME_OP( FT_FRAME_OP_SHORT, 1, 1 ),

    ft_frame_ulong_be  = FT_MAKE_FRAME_OP( FT_FRAME_OP_LONG, 0, 0 ),
    ft_frame_long_be   = FT_MAKE_FRAME_OP( FT_FRAME_OP_LONG, 0, 1 ),
    ft_frame_ulong_le  = FT_MAKE_FRAME_OP( FT_FRAME_OP_LONG, 1, 0 ),
    ft_frame_long_le   = FT_MAKE_FRAME_OP( FT_FRAME_OP_LONG, 1, 1 ),

    ft_frame_uoff3_be  = FT_MAKE_FRAME_OP( FT_FRAME_OP_OFF3, 0, 0 ),
    ft_frame_off3_be   = FT_MAKE_FRAME_OP( FT_FRAME_OP_OFF3, 0, 1 ),
    ft_frame_uoff3_le  = FT_MAKE_FRAME_OP( FT_FRAME_OP_OFF3, 1, 0 ),
    ft_frame_off3_le   = FT_MAKE_FRAME_OP( FT_FRAME_OP_OFF3, 1, 1 ),

    ft_frame_bytes     = FT_MAKE_FRAME_OP( FT_FRAME_OP_BYTES, 0, 0 ),
    ft_frame_skip      = FT_MAKE_FRAME_OP( FT_FRAME_OP_BYTES, 0, 1 )

  } FT_Frame_Op;


  typedef struct  FT_Frame_Field_
  {
    FT_Byte    value;
    FT_Byte    size;
    FT_UShort  offset;

  } FT_Frame_Field;


  /* Construct an FT_Frame_Field out of a structure type and a field name. */
  /* The structure type must be set in the FT_STRUCTURE macro before       */
  /* calling the FT_FRAME_START() macro.                                   */
  /*                                                                       */
#define FT_FIELD_SIZE( f )                          \
          (FT_Byte)sizeof ( ((FT_STRUCTURE*)0)->f )

#define FT_FIELD_SIZE_DELTA( f )                       \
          (FT_Byte)sizeof ( ((FT_STRUCTURE*)0)->f[0] )

#define FT_FIELD_OFFSET( f )                         \
          (FT_UShort)( offsetof( FT_STRUCTURE, f ) )

#define FT_FRAME_FIELD( frame_op, field ) \
          {                               \
            frame_op,                     \
            FT_FIELD_SIZE( field ),       \
            FT_FIELD_OFFSET( field )      \
          }

#define FT_MAKE_EMPTY_FIELD( frame_op )  { frame_op, 0, 0 }

#define FT_FRAME_START( size )   { ft_frame_start, 0, size }
#define FT_FRAME_END             { ft_frame_end, 0, 0 }

#define FT_FRAME_LONG( f )       FT_FRAME_FIELD( ft_frame_long_be, f )
#define FT_FRAME_ULONG( f )      FT_FRAME_FIELD( ft_frame_ulong_be, f )
#define FT_FRAME_SHORT( f )      FT_FRAME_FIELD( ft_frame_short_be, f )
#define FT_FRAME_USHORT( f )     FT_FRAME_FIELD( ft_frame_ushort_be, f )
#define FT_FRAME_OFF3( f )       FT_FRAME_FIELD( ft_frame_off3_be, f )
#define FT_FRAME_UOFF3( f )      FT_FRAME_FIELD( ft_frame_uoff3_be, f )
#define FT_FRAME_BYTE( f )       FT_FRAME_FIELD( ft_frame_byte, f )
#define FT_FRAME_CHAR( f )       FT_FRAME_FIELD( ft_frame_schar, f )

#define FT_FRAME_LONG_LE( f )    FT_FRAME_FIELD( ft_frame_long_le, f )
#define FT_FRAME_ULONG_LE( f )   FT_FRAME_FIELD( ft_frame_ulong_le, f )
#define FT_FRAME_SHORT_LE( f )   FT_FRAME_FIELD( ft_frame_short_le, f )
#define FT_FRAME_USHORT_LE( f )  FT_FRAME_FIELD( ft_frame_ushort_le, f )
#define FT_FRAME_OFF3_LE( f )    FT_FRAME_FIELD( ft_frame_off3_le, f )
#define FT_FRAME_UOFF3_LE( f )   FT_FRAME_FIELD( ft_frame_uoff3_le, f )

#define FT_FRAME_SKIP_LONG       { ft_frame_long_be, 0, 0 }
#define FT_FRAME_SKIP_SHORT      { ft_frame_short_be, 0, 0 }
#define FT_FRAME_SKIP_BYTE       { ft_frame_byte, 0, 0 }

#define FT_FRAME_BYTES( field, count ) \
          {                            \
            ft_frame_bytes,            \
            count,                     \
            FT_FIELD_OFFSET( field )   \
          }

#define FT_FRAME_SKIP_BYTES( count )  { ft_frame_skip, count, 0 }


  /**************************************************************************
   *
   * Integer extraction macros -- the 'buffer' parameter must ALWAYS be of
   * type 'char*' or equivalent (1-byte elements).
   */

#define FT_BYTE_( p, i )  ( ((const FT_Byte*)(p))[(i)] )

#define FT_INT16( x )   ( (FT_Int16)(x)  )
#define FT_UINT16( x )  ( (FT_UInt16)(x) )
#define FT_INT32( x )   ( (FT_Int32)(x)  )
#define FT_UINT32( x )  ( (FT_UInt32)(x) )


#define FT_BYTE_U16( p, i, s )  ( FT_UINT16( FT_BYTE_( p, i ) ) << (s) )
#define FT_BYTE_U32( p, i, s )  ( FT_UINT32( FT_BYTE_( p, i ) ) << (s) )


  /*
   *    function      acts on      increases  does range   for    emits
   *                                pointer    checking   frames  error
   *  -------------------------------------------------------------------
   *   FT_PEEK_XXX  buffer pointer      no         no        no     no
   *   FT_NEXT_XXX  buffer pointer     yes         no        no     no
   *   FT_GET_XXX   stream->cursor     yes        yes       yes     no
   *   FT_READ_XXX  stream->pos        yes        yes        no    yes
   */


  /*
   * `FT_PEEK_XXX' are generic macros to get data from a buffer position.  No
   * safety checks are performed.
   */
#define FT_PEEK_SHORT( p )  FT_INT16( FT_BYTE_U16( p, 0, 8 ) | \
                                      FT_BYTE_U16( p, 1, 0 ) )

#define FT_PEEK_USHORT( p )  FT_UINT16( FT_BYTE_U16( p, 0, 8 ) | \
                                        FT_BYTE_U16( p, 1, 0 ) )

#define FT_PEEK_LONG( p )  FT_INT32( FT_BYTE_U32( p, 0, 24 ) | \
                                     FT_BYTE_U32( p, 1, 16 ) | \
                                     FT_BYTE_U32( p, 2,  8 ) | \
                                     FT_BYTE_U32( p, 3,  0 ) )

#define FT_PEEK_ULONG( p )  FT_UINT32( FT_BYTE_U32( p, 0, 24 ) | \
                                       FT_BYTE_U32( p, 1, 16 ) | \
                                       FT_BYTE_U32( p, 2,  8 ) | \
                                       FT_BYTE_U32( p, 3,  0 ) )

#define FT_PEEK_OFF3( p )  FT_INT32( FT_BYTE_U32( p, 0, 16 ) | \
                                     FT_BYTE_U32( p, 1,  8 ) | \
                                     FT_BYTE_U32( p, 2,  0 ) )

#define FT_PEEK_UOFF3( p )  FT_UINT32( FT_BYTE_U32( p, 0, 16 ) | \
                                       FT_BYTE_U32( p, 1,  8 ) | \
                                       FT_BYTE_U32( p, 2,  0 ) )

#define FT_PEEK_SHORT_LE( p )  FT_INT16( FT_BYTE_U16( p, 1, 8 ) | \
                                         FT_BYTE_U16( p, 0, 0 ) )

#define FT_PEEK_USHORT_LE( p )  FT_UINT16( FT_BYTE_U16( p, 1, 8 ) |  \
                                           FT_BYTE_U16( p, 0, 0 ) )

#define FT_PEEK_LONG_LE( p )  FT_INT32( FT_BYTE_U32( p, 3, 24 ) | \
                                        FT_BYTE_U32( p, 2, 16 ) | \
                                        FT_BYTE_U32( p, 1,  8 ) | \
                                        FT_BYTE_U32( p, 0,  0 ) )

#define FT_PEEK_ULONG_LE( p )  FT_UINT32( FT_BYTE_U32( p, 3, 24 ) | \
                                          FT_BYTE_U32( p, 2, 16 ) | \
                                          FT_BYTE_U32( p, 1,  8 ) | \
                                          FT_BYTE_U32( p, 0,  0 ) )

#define FT_PEEK_OFF3_LE( p )  FT_INT32( FT_BYTE_U32( p, 2, 16 ) | \
                                        FT_BYTE_U32( p, 1,  8 ) | \
                                        FT_BYTE_U32( p, 0,  0 ) )

#define FT_PEEK_UOFF3_LE( p )  FT_UINT32( FT_BYTE_U32( p, 2, 16 ) | \
                                          FT_BYTE_U32( p, 1,  8 ) | \
                                          FT_BYTE_U32( p, 0,  0 ) )

  /*
   * `FT_NEXT_XXX' are generic macros to get data from a buffer position
   * which is then increased appropriately.  No safety checks are performed.
   */
#define FT_NEXT_CHAR( buffer )       \
          ( (signed char)*buffer++ )

#define FT_NEXT_BYTE( buffer )         \
          ( (unsigned char)*buffer++ )

#define FT_NEXT_SHORT( buffer )                                   \
          ( (short)( buffer += 2, FT_PEEK_SHORT( buffer - 2 ) ) )

#define FT_NEXT_USHORT( buffer )                                            \
          ( (unsigned short)( buffer += 2, FT_PEEK_USHORT( buffer - 2 ) ) )

#define FT_NEXT_OFF3( buffer )                                  \
          ( (long)( buffer += 3, FT_PEEK_OFF3( buffer - 3 ) ) )

#define FT_NEXT_UOFF3( buffer )                                           \
          ( (unsigned long)( buffer += 3, FT_PEEK_UOFF3( buffer - 3 ) ) )

#define FT_NEXT_LONG( buffer )                                  \
          ( (long)( buffer += 4, FT_PEEK_LONG( buffer - 4 ) ) )

#define FT_NEXT_ULONG( buffer )                                           \
          ( (unsigned long)( buffer += 4, FT_PEEK_ULONG( buffer - 4 ) ) )


#define FT_NEXT_SHORT_LE( buffer )                                   \
          ( (short)( buffer += 2, FT_PEEK_SHORT_LE( buffer - 2 ) ) )

#define FT_NEXT_USHORT_LE( buffer )                                            \
          ( (unsigned short)( buffer += 2, FT_PEEK_USHORT_LE( buffer - 2 ) ) )

#define FT_NEXT_OFF3_LE( buffer )                                  \
          ( (long)( buffer += 3, FT_PEEK_OFF3_LE( buffer - 3 ) ) )

#define FT_NEXT_UOFF3_LE( buffer )                                           \
          ( (unsigned long)( buffer += 3, FT_PEEK_UOFF3_LE( buffer - 3 ) ) )

#define FT_NEXT_LONG_LE( buffer )                                  \
          ( (long)( buffer += 4, FT_PEEK_LONG_LE( buffer - 4 ) ) )

#define FT_NEXT_ULONG_LE( buffer )                                           \
          ( (unsigned long)( buffer += 4, FT_PEEK_ULONG_LE( buffer - 4 ) ) )


  /**************************************************************************
   *
   * The `FT_GET_XXX` macros use an implicit 'stream' variable.
   *
   * Note that a call to `FT_STREAM_SEEK` or `FT_STREAM_POS` has **no**
   * effect on `FT_GET_XXX`!  They operate on `stream->pos`, while
   * `FT_GET_XXX` use `stream->cursor`.
   */
#if 0
#define FT_GET_MACRO( type )    FT_NEXT_ ## type ( stream->cursor )

#define FT_GET_CHAR()       FT_GET_MACRO( CHAR )
#define FT_GET_BYTE()       FT_GET_MACRO( BYTE )
#define FT_GET_SHORT()      FT_GET_MACRO( SHORT )
#define FT_GET_USHORT()     FT_GET_MACRO( USHORT )
#define FT_GET_OFF3()       FT_GET_MACRO( OFF3 )
#define FT_GET_UOFF3()      FT_GET_MACRO( UOFF3 )
#define FT_GET_LONG()       FT_GET_MACRO( LONG )
#define FT_GET_ULONG()      FT_GET_MACRO( ULONG )
#define FT_GET_TAG4()       FT_GET_MACRO( ULONG )

#define FT_GET_SHORT_LE()   FT_GET_MACRO( SHORT_LE )
#define FT_GET_USHORT_LE()  FT_GET_MACRO( USHORT_LE )
#define FT_GET_LONG_LE()    FT_GET_MACRO( LONG_LE )
#define FT_GET_ULONG_LE()   FT_GET_MACRO( ULONG_LE )

#else
#define FT_GET_MACRO( func, type )        ( (type)func( stream ) )

#define FT_GET_CHAR()       FT_GET_MACRO( FT_Stream_GetChar, FT_Char )
#define FT_GET_BYTE()       FT_GET_MACRO( FT_Stream_GetChar, FT_Byte )
#define FT_GET_SHORT()      FT_GET_MACRO( FT_Stream_GetUShort, FT_Short )
#define FT_GET_USHORT()     FT_GET_MACRO( FT_Stream_GetUShort, FT_UShort )
#define FT_GET_OFF3()       FT_GET_MACRO( FT_Stream_GetUOffset, FT_Long )
#define FT_GET_UOFF3()      FT_GET_MACRO( FT_Stream_GetUOffset, FT_ULong )
#define FT_GET_LONG()       FT_GET_MACRO( FT_Stream_GetULong, FT_Long )
#define FT_GET_ULONG()      FT_GET_MACRO( FT_Stream_GetULong, FT_ULong )
#define FT_GET_TAG4()       FT_GET_MACRO( FT_Stream_GetULong, FT_ULong )

#define FT_GET_SHORT_LE()   FT_GET_MACRO( FT_Stream_GetUShortLE, FT_Short )
#define FT_GET_USHORT_LE()  FT_GET_MACRO( FT_Stream_GetUShortLE, FT_UShort )
#define FT_GET_LONG_LE()    FT_GET_MACRO( FT_Stream_GetULongLE, FT_Long )
#define FT_GET_ULONG_LE()   FT_GET_MACRO( FT_Stream_GetULongLE, FT_ULong )
#endif


#define FT_READ_MACRO( func, type, var )        \
          ( var = (type)func( stream, &error ), \
            error != FT_Err_Ok )

  /*
   * The `FT_READ_XXX' macros use implicit `stream' and `error' variables.
   *
   * `FT_READ_XXX' can be controlled with `FT_STREAM_SEEK' and
   * `FT_STREAM_POS'.  They use the full machinery to check whether a read is
   * valid.
   */
#define FT_READ_BYTE( var )       FT_READ_MACRO( FT_Stream_ReadChar, FT_Byte, var )
#define FT_READ_CHAR( var )       FT_READ_MACRO( FT_Stream_ReadChar, FT_Char, var )
#define FT_READ_SHORT( var )      FT_READ_MACRO( FT_Stream_ReadUShort, FT_Short, var )
#define FT_READ_USHORT( var )     FT_READ_MACRO( FT_Stream_ReadUShort, FT_UShort, var )
#define FT_READ_OFF3( var )       FT_READ_MACRO( FT_Stream_ReadUOffset, FT_Long, var )
#define FT_READ_UOFF3( var )      FT_READ_MACRO( FT_Stream_ReadUOffset, FT_ULong, var )
#define FT_READ_LONG( var )       FT_READ_MACRO( FT_Stream_ReadULong, FT_Long, var )
#define FT_READ_ULONG( var )      FT_READ_MACRO( FT_Stream_ReadULong, FT_ULong, var )

#define FT_READ_SHORT_LE( var )   FT_READ_MACRO( FT_Stream_ReadUShortLE, FT_Short, var )
#define FT_READ_USHORT_LE( var )  FT_READ_MACRO( FT_Stream_ReadUShortLE, FT_UShort, var )
#define FT_READ_LONG_LE( var )    FT_READ_MACRO( FT_Stream_ReadULongLE, FT_Long, var )
#define FT_READ_ULONG_LE( var )   FT_READ_MACRO( FT_Stream_ReadULongLE, FT_ULong, var )


#ifndef FT_CONFIG_OPTION_NO_DEFAULT_SYSTEM

  /* initialize a stream for reading a regular system stream */
  FT_BASE( FT_Error )
  FT_Stream_Open( FT_Stream    stream,
                  const char*  filepathname );

#endif /* FT_CONFIG_OPTION_NO_DEFAULT_SYSTEM */


  /* create a new (input) stream from an FT_Open_Args structure */
  FT_BASE( FT_Error )
  FT_Stream_New( FT_Library           library,
                 const FT_Open_Args*  args,
                 FT_Stream           *astream );

  /* free a stream */
  FT_BASE( void )
  FT_Stream_Free( FT_Stream  stream,
                  FT_Int     external );

  /* initialize a stream for reading in-memory data */
  FT_BASE( void )
  FT_Stream_OpenMemory( FT_Stream       stream,
                        const FT_Byte*  base,
                        FT_ULong        size );

  /* close a stream (does not destroy the stream structure) */
  FT_BASE( void )
  FT_Stream_Close( FT_Stream  stream );


  /* seek within a stream. position is relative to start of stream */
  FT_BASE( FT_Error )
  FT_Stream_Seek( FT_Stream  stream,
                  FT_ULong   pos );

  /* skip bytes in a stream */
  FT_BASE( FT_Error )
  FT_Stream_Skip( FT_Stream  stream,
                  FT_Long    distance );

  /* return current stream position */
  FT_BASE( FT_ULong )
  FT_Stream_Pos( FT_Stream  stream );

  /* read bytes from a stream into a user-allocated buffer, returns an */
  /* error if not all bytes could be read.                             */
  FT_BASE( FT_Error )
  FT_Stream_Read( FT_Stream  stream,
                  FT_Byte*   buffer,
                  FT_ULong   count );

  /* read bytes from a stream at a given position */
  FT_BASE( FT_Error )
  FT_Stream_ReadAt( FT_Stream  stream,
                    FT_ULong   pos,
                    FT_Byte*   buffer,
                    FT_ULong   count );

  /* try to read bytes at the end of a stream; return number of bytes */
  /* really available                                                 */
  FT_BASE( FT_ULong )
  FT_Stream_TryRead( FT_Stream  stream,
                     FT_Byte*   buffer,
                     FT_ULong   count );

  /* Enter a frame of `count' consecutive bytes in a stream.  Returns an */
  /* error if the frame could not be read/accessed.  The caller can use  */
  /* the `FT_Stream_GetXXX' functions to retrieve frame data without     */
  /* error checks.                                                       */
  /*                                                                     */
  /* You must _always_ call `FT_Stream_ExitFrame' once you have entered  */
  /* a stream frame!                                                     */
  /*                                                                     */
  /* Nested frames are not permitted.                                    */
  /*                                                                     */
  FT_BASE( FT_Error )
  FT_Stream_EnterFrame( FT_Stream  stream,
                        FT_ULong   count );

  /* exit a stream frame */
  FT_BASE( void )
  FT_Stream_ExitFrame( FT_Stream  stream );


  /* Extract a stream frame.  If the stream is disk-based, a heap block */
  /* is allocated and the frame bytes are read into it.  If the stream  */
  /* is memory-based, this function simply sets a pointer to the data.  */
  /*                                                                    */
  /* Useful to optimize access to memory-based streams transparently.   */
  /*                                                                    */
  /* `FT_Stream_GetXXX' functions can't be used.                        */
  /*                                                                    */
  /* An extracted frame must be `freed' with a call to the function     */
  /* `FT_Stream_ReleaseFrame'.                                          */
  /*                                                                    */
  FT_BASE( FT_Error )
  FT_Stream_ExtractFrame( FT_Stream  stream,
                          FT_ULong   count,
                          FT_Byte**  pbytes );

  /* release an extract frame (see `FT_Stream_ExtractFrame') */
  FT_BASE( void )
  FT_Stream_ReleaseFrame( FT_Stream  stream,
                          FT_Byte**  pbytes );


  /* read a byte from an entered frame */
  FT_BASE( FT_Char )
  FT_Stream_GetChar( FT_Stream  stream );

  /* read a 16-bit big-endian unsigned integer from an entered frame */
  FT_BASE( FT_UShort )
  FT_Stream_GetUShort( FT_Stream  stream );

  /* read a 24-bit big-endian unsigned integer from an entered frame */
  FT_BASE( FT_ULong )
  FT_Stream_GetUOffset( FT_Stream  stream );

  /* read a 32-bit big-endian unsigned integer from an entered frame */
  FT_BASE( FT_ULong )
  FT_Stream_GetULong( FT_Stream  stream );

  /* read a 16-bit little-endian unsigned integer from an entered frame */
  FT_BASE( FT_UShort )
  FT_Stream_GetUShortLE( FT_Stream  stream );

  /* read a 32-bit little-endian unsigned integer from an entered frame */
  FT_BASE( FT_ULong )
  FT_Stream_GetULongLE( FT_Stream  stream );


  /* read a byte from a stream */
  FT_BASE( FT_Char )
  FT_Stream_ReadChar( FT_Stream  stream,
                      FT_Error*  error );

  /* read a 16-bit big-endian unsigned integer from a stream */
  FT_BASE( FT_UShort )
  FT_Stream_ReadUShort( FT_Stream  stream,
                        FT_Error*  error );

  /* read a 24-bit big-endian unsigned integer from a stream */
  FT_BASE( FT_ULong )
  FT_Stream_ReadUOffset( FT_Stream  stream,
                         FT_Error*  error );

  /* read a 32-bit big-endian integer from a stream */
  FT_BASE( FT_ULong )
  FT_Stream_ReadULong( FT_Stream  stream,
                       FT_Error*  error );

  /* read a 16-bit little-endian unsigned integer from a stream */
  FT_BASE( FT_UShort )
  FT_Stream_ReadUShortLE( FT_Stream  stream,
                          FT_Error*  error );

  /* read a 32-bit little-endian unsigned integer from a stream */
  FT_BASE( FT_ULong )
  FT_Stream_ReadULongLE( FT_Stream  stream,
                         FT_Error*  error );

  /* Read a structure from a stream.  The structure must be described */
  /* by an array of FT_Frame_Field records.                           */
  FT_BASE( FT_Error )
  FT_Stream_ReadFields( FT_Stream              stream,
                        const FT_Frame_Field*  fields,
                        void*                  structure );


#define FT_STREAM_POS()           \
          FT_Stream_Pos( stream )

#define FT_STREAM_SEEK( position )                               \
          FT_SET_ERROR( FT_Stream_Seek( stream,                  \
                                        (FT_ULong)(position) ) )

#define FT_STREAM_SKIP( distance )                              \
          FT_SET_ERROR( FT_Stream_Skip( stream,                 \
                                        (FT_Long)(distance) ) )

#define FT_STREAM_READ( buffer, count )                       \
          FT_SET_ERROR( FT_Stream_Read( stream,               \
                                        (FT_Byte*)(buffer),   \
                                        (FT_ULong)(count) ) )

#define FT_STREAM_READ_AT( position, buffer, count )            \
          FT_SET_ERROR( FT_Stream_ReadAt( stream,               \
                                          (FT_ULong)(position), \
                                          (FT_Byte*)(buffer),   \
                                          (FT_ULong)(count) ) )

#define FT_STREAM_READ_FIELDS( fields, object )                          \
          FT_SET_ERROR( FT_Stream_ReadFields( stream, fields, object ) )


#define FT_FRAME_ENTER( size )                                           \
          FT_SET_ERROR(                                                  \
            FT_DEBUG_INNER( FT_Stream_EnterFrame( stream,                \
                                                  (FT_ULong)(size) ) ) )

#define FT_FRAME_EXIT()                                   \
          FT_DEBUG_INNER( FT_Stream_ExitFrame( stream ) )

#define FT_FRAME_EXTRACT( size, bytes )                                       \
          FT_SET_ERROR(                                                       \
            FT_DEBUG_INNER( FT_Stream_ExtractFrame( stream,                   \
                                                    (FT_ULong)(size),         \
                                                    (FT_Byte**)&(bytes) ) ) )

#define FT_FRAME_RELEASE( bytes )                                         \
          FT_DEBUG_INNER( FT_Stream_ReleaseFrame( stream,                 \
                                                  (FT_Byte**)&(bytes) ) )


FT_END_HEADER

#endif /* FTSTREAM_H_ */


/* END */
