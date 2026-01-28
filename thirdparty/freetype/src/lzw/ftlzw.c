/****************************************************************************
 *
 * ftlzw.c
 *
 *   FreeType support for .Z compressed files.
 *
 * This optional component relies on NetBSD's zopen().  It should mainly
 * be used to parse compressed PCF fonts, as found with many X11 server
 * distributions.
 *
 * Copyright (C) 2004-2025 by
 * Albert Chin-A-Young.
 *
 * based on code in `src/gzip/ftgzip.c'
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */

#include <freetype/internal/ftmemory.h>
#include <freetype/internal/ftstream.h>
#include <freetype/internal/ftdebug.h>
#include <freetype/ftlzw.h>
#include FT_CONFIG_STANDARD_LIBRARY_H


#include <freetype/ftmoderr.h>

#undef FTERRORS_H_

#undef  FT_ERR_PREFIX
#define FT_ERR_PREFIX  LZW_Err_
#define FT_ERR_BASE    FT_Mod_Err_LZW

#include <freetype/fterrors.h>


#ifdef FT_CONFIG_OPTION_USE_LZW

#include "ftzopen.h"


/***************************************************************************/
/***************************************************************************/
/*****                                                                 *****/
/*****                  M E M O R Y   M A N A G E M E N T              *****/
/*****                                                                 *****/
/***************************************************************************/
/***************************************************************************/

/***************************************************************************/
/***************************************************************************/
/*****                                                                 *****/
/*****                   F I L E   D E S C R I P T O R                 *****/
/*****                                                                 *****/
/***************************************************************************/
/***************************************************************************/

#define FT_LZW_BUFFER_SIZE  4096

  typedef struct  FT_LZWFileRec_
  {
    FT_Stream       source;         /* parent/source stream        */
    FT_Stream       stream;         /* embedding stream            */
    FT_Memory       memory;         /* memory allocator            */
    FT_LzwStateRec  lzw;            /* lzw decompressor state      */

    FT_Byte         buffer[FT_LZW_BUFFER_SIZE]; /* output buffer      */
    FT_ULong        pos;                        /* position in output */
    FT_Byte*        cursor;
    FT_Byte*        limit;

  } FT_LZWFileRec, *FT_LZWFile;


  /* check and skip .Z header */
  static FT_Error
  ft_lzw_check_header( FT_Stream  stream )
  {
    FT_Error  error;
    FT_Byte   head[2];


    if ( FT_STREAM_SEEK( 0 )       ||
         FT_STREAM_READ( head, 2 ) )
      goto Exit;

    /* head[0] && head[1] are the magic numbers */
    if ( head[0] != 0x1F ||
         head[1] != 0x9D )
      error = FT_THROW( Invalid_File_Format );

  Exit:
    return error;
  }


  static FT_Error
  ft_lzw_file_init( FT_LZWFile  zip,
                    FT_Stream   stream,
                    FT_Stream   source )
  {
    FT_LzwState  lzw   = &zip->lzw;
    FT_Error     error;


    zip->stream = stream;
    zip->source = source;
    zip->memory = stream->memory;

    zip->limit  = zip->buffer + FT_LZW_BUFFER_SIZE;
    zip->cursor = zip->limit;
    zip->pos    = 0;

    /* check and skip .Z header */
    error = ft_lzw_check_header( source );
    if ( error )
      goto Exit;

    /* initialize internal lzw variable */
    ft_lzwstate_init( lzw, source );

  Exit:
    return error;
  }


  static void
  ft_lzw_file_done( FT_LZWFile  zip )
  {
    /* clear the rest */
    ft_lzwstate_done( &zip->lzw );

    zip->memory = NULL;
    zip->source = NULL;
    zip->stream = NULL;
  }


  static FT_Error
  ft_lzw_file_reset( FT_LZWFile  zip )
  {
    FT_Stream  stream = zip->source;
    FT_Error   error;


    if ( !FT_STREAM_SEEK( 0 ) )
    {
      ft_lzwstate_reset( &zip->lzw );

      zip->limit  = zip->buffer + FT_LZW_BUFFER_SIZE;
      zip->cursor = zip->limit;
      zip->pos    = 0;
    }

    return error;
  }


  static FT_Error
  ft_lzw_file_fill_output( FT_LZWFile  zip )
  {
    FT_LzwState  lzw = &zip->lzw;
    FT_ULong     count;
    FT_Error     error = FT_Err_Ok;


    zip->cursor = zip->buffer;

    count = ft_lzwstate_io( lzw, zip->buffer, FT_LZW_BUFFER_SIZE );

    zip->limit = zip->cursor + count;

    if ( count == 0 )
      error = FT_THROW( Invalid_Stream_Operation );

    return error;
  }


  /* fill output buffer; `count' must be <= FT_LZW_BUFFER_SIZE */
  static FT_Error
  ft_lzw_file_skip_output( FT_LZWFile  zip,
                           FT_ULong    count )
  {
    FT_Error  error = FT_Err_Ok;


    /* first, we skip what we can from the output buffer */
    {
      FT_ULong  delta = (FT_ULong)( zip->limit - zip->cursor );


      if ( delta >= count )
        delta = count;

      zip->cursor += delta;
      zip->pos    += delta;

      count -= delta;
    }

    /* next, we skip as many bytes remaining as possible */
    while ( count > 0 )
    {
      FT_ULong  delta = FT_LZW_BUFFER_SIZE;
      FT_ULong  numread;


      if ( delta > count )
        delta = count;

      numread = ft_lzwstate_io( &zip->lzw, NULL, delta );
      if ( numread < delta )
      {
        /* not enough bytes */
        error = FT_THROW( Invalid_Stream_Operation );
        break;
      }

      zip->pos += delta;
      count    -= delta;
    }

    return error;
  }


  static FT_ULong
  ft_lzw_file_io( FT_LZWFile  zip,
                  FT_ULong    pos,
                  FT_Byte*    buffer,
                  FT_ULong    count )
  {
    FT_ULong  result = 0;
    FT_Error  error;


    /* seeking backwards. */
    if ( pos < zip->pos )
    {
      /* If the new position is within the output buffer, simply       */
      /* decrement pointers, otherwise we reset the stream completely! */
      if ( ( zip->pos - pos ) <= (FT_ULong)( zip->cursor - zip->buffer ) )
      {
        zip->cursor -= zip->pos - pos;
        zip->pos     = pos;
      }
      else
      {
        error = ft_lzw_file_reset( zip );
        if ( error )
          goto Exit;
      }
    }

    /* skip unwanted bytes */
    if ( pos > zip->pos )
    {
      error = ft_lzw_file_skip_output( zip, (FT_ULong)( pos - zip->pos ) );
      if ( error )
        goto Exit;
    }

    if ( count == 0 )
      goto Exit;

    /* now read the data */
    for (;;)
    {
      FT_ULong  delta;


      delta = (FT_ULong)( zip->limit - zip->cursor );
      if ( delta >= count )
        delta = count;

      FT_MEM_COPY( buffer + result, zip->cursor, delta );
      result      += delta;
      zip->cursor += delta;
      zip->pos    += delta;

      count -= delta;
      if ( count == 0 )
        break;

      error = ft_lzw_file_fill_output( zip );
      if ( error )
        break;
    }

  Exit:
    return result;
  }


/***************************************************************************/
/***************************************************************************/
/*****                                                                 *****/
/*****            L Z W   E M B E D D I N G   S T R E A M              *****/
/*****                                                                 *****/
/***************************************************************************/
/***************************************************************************/

  static void
  ft_lzw_stream_close( FT_Stream  stream )
  {
    FT_LZWFile  zip    = (FT_LZWFile)stream->descriptor.pointer;
    FT_Memory   memory = stream->memory;


    if ( zip )
    {
      /* finalize lzw file descriptor */
      ft_lzw_file_done( zip );

      FT_FREE( zip );

      stream->descriptor.pointer = NULL;
    }
  }


  static unsigned long
  ft_lzw_stream_io( FT_Stream       stream,
                    unsigned long   offset,
                    unsigned char*  buffer,
                    unsigned long   count )
  {
    FT_LZWFile  zip = (FT_LZWFile)stream->descriptor.pointer;


    return ft_lzw_file_io( zip, offset, buffer, count );
  }


  FT_EXPORT_DEF( FT_Error )
  FT_Stream_OpenLZW( FT_Stream  stream,
                     FT_Stream  source )
  {
    FT_Error    error;
    FT_Memory   memory;
    FT_LZWFile  zip = NULL;


    if ( !stream || !source )
    {
      error = FT_THROW( Invalid_Stream_Handle );
      goto Exit;
    }

    memory = source->memory;

    /*
     * Check the header right now; this prevents allocation of a huge
     * LZWFile object (400 KByte of heap memory) if not necessary.
     *
     * Did I mention that you should never use .Z compressed font
     * files?
     */
    error = ft_lzw_check_header( source );
    if ( error )
      goto Exit;

    FT_ZERO( stream );
    stream->memory = memory;

    if ( !FT_QNEW( zip ) )
    {
      error = ft_lzw_file_init( zip, stream, source );
      if ( error )
      {
        FT_FREE( zip );
        goto Exit;
      }

      stream->descriptor.pointer = zip;
    }

    stream->size  = 0x7FFFFFFFL;  /* don't know the real size! */
    stream->pos   = 0;
    stream->base  = NULL;
    stream->read  = ft_lzw_stream_io;
    stream->close = ft_lzw_stream_close;

  Exit:
    return error;
  }


#include "ftzopen.c"


#else  /* !FT_CONFIG_OPTION_USE_LZW */


  FT_EXPORT_DEF( FT_Error )
  FT_Stream_OpenLZW( FT_Stream  stream,
                     FT_Stream  source )
  {
    FT_UNUSED( stream );
    FT_UNUSED( source );

    return FT_THROW( Unimplemented_Feature );
  }


#endif /* !FT_CONFIG_OPTION_USE_LZW */


/* END */
