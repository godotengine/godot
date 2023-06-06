/****************************************************************************
 *
 * ftgzip.c
 *
 *   FreeType support for .gz compressed files.
 *
 * This optional component relies on zlib.  It should mainly be used to
 * parse compressed PCF fonts, as found with many X11 server
 * distributions.
 *
 * Copyright (C) 2002-2023 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
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
#include <freetype/ftgzip.h>
#include FT_CONFIG_STANDARD_LIBRARY_H


#include <freetype/ftmoderr.h>

#undef FTERRORS_H_

#undef  FT_ERR_PREFIX
#define FT_ERR_PREFIX  Gzip_Err_
#define FT_ERR_BASE    FT_Mod_Err_Gzip

#include <freetype/fterrors.h>


#ifdef FT_CONFIG_OPTION_USE_ZLIB

#ifdef FT_CONFIG_OPTION_SYSTEM_ZLIB

#include <zlib.h>

#else /* !FT_CONFIG_OPTION_SYSTEM_ZLIB */

  /* In this case, we include our own modified sources of the ZLib  */
  /* within the `gzip' component.  The modifications were necessary */
  /* to #include all files without conflicts, as well as preventing */
  /* the definition of `extern' functions that may cause linking    */
  /* conflicts when a program is linked with both FreeType and the  */
  /* original ZLib.                                                 */

#ifndef USE_ZLIB_ZCALLOC
#define MY_ZCALLOC /* prevent all zcalloc() & zfree() in zutil.c */
#endif

  /* Note that our `zlib.h' includes `ftzconf.h' instead of `zconf.h'; */
  /* the main reason is that even a global `zlib.h' includes `zconf.h' */
  /* with                                                              */
  /*                                                                   */
  /*   #include "zconf.h"                                              */
  /*                                                                   */
  /* instead of the expected                                           */
  /*                                                                   */
  /*   #include <zconf.h>                                              */
  /*                                                                   */
  /* so that configuration with `FT_CONFIG_OPTION_SYSTEM_ZLIB' might   */
  /* include the wrong `zconf.h' file, leading to errors.              */

#if defined( __GNUC__ ) ||  defined( __clang__ )
#define ZEXPORT
#define ZEXTERN      static
#endif

#define HAVE_MEMCPY  1
#define Z_SOLO       1
#define Z_FREETYPE   1

#if defined( _MSC_VER )      /* Visual C++ (and Intel C++)   */
  /* We disable the warning `conversion from XXX to YYY,     */
  /* possible loss of data' in order to compile cleanly with */
  /* the maximum level of warnings: zlib is non-FreeType     */
  /* code.                                                   */
#pragma warning( push )
#pragma warning( disable : 4244 )
#endif /* _MSC_VER */

#if defined( __GNUC__ )
#pragma GCC diagnostic push
#ifndef __cplusplus
#pragma GCC diagnostic ignored "-Wstrict-prototypes"
#endif
#pragma GCC diagnostic ignored "-Wimplicit-fallthrough"
#pragma GCC diagnostic ignored "-Wredundant-decls"
#endif

#include "zutil.c"
#include "inffast.c"
#include "inflate.c"
#include "inftrees.c"
#include "adler32.c"
#include "crc32.c"

#if defined( __GNUC__ )
#pragma GCC diagnostic pop
#endif

#if defined( _MSC_VER )
#pragma warning( pop )
#endif

#endif /* !FT_CONFIG_OPTION_SYSTEM_ZLIB */


/***************************************************************************/
/***************************************************************************/
/*****                                                                 *****/
/*****            Z L I B   M E M O R Y   M A N A G E M E N T          *****/
/*****                                                                 *****/
/***************************************************************************/
/***************************************************************************/

  /* it is better to use FreeType memory routines instead of raw
     'malloc/free' */

  static voidpf
  ft_gzip_alloc( voidpf  opaque,
                 uInt    items,
                 uInt    size )
  {
    FT_Memory   memory = (FT_Memory)opaque;
    FT_ULong    sz     = (FT_ULong)size * items;
    FT_Error    error;
    FT_Pointer  p      = NULL;


    /* allocate and zero out */
    FT_MEM_ALLOC( p, sz );
    return p;
  }


  static void
  ft_gzip_free( voidpf  opaque,
                voidpf  address )
  {
    FT_Memory  memory = (FT_Memory)opaque;


    FT_MEM_FREE( address );
  }

/***************************************************************************/
/***************************************************************************/
/*****                                                                 *****/
/*****               Z L I B   F I L E   D E S C R I P T O R           *****/
/*****                                                                 *****/
/***************************************************************************/
/***************************************************************************/

#define FT_GZIP_BUFFER_SIZE  4096

  typedef struct  FT_GZipFileRec_
  {
    FT_Stream  source;         /* parent/source stream        */
    FT_Stream  stream;         /* embedding stream            */
    FT_Memory  memory;         /* memory allocator            */
    z_stream   zstream;        /* zlib input stream           */

    FT_ULong   start;          /* starting position, after .gz header */
    FT_Byte    input[FT_GZIP_BUFFER_SIZE];   /* input read buffer  */

    FT_Byte    buffer[FT_GZIP_BUFFER_SIZE];  /* output buffer      */
    FT_ULong   pos;                          /* position in output */
    FT_Byte*   cursor;
    FT_Byte*   limit;

  } FT_GZipFileRec, *FT_GZipFile;


  /* gzip flag byte */
#define FT_GZIP_ASCII_FLAG   0x01 /* bit 0 set: file probably ascii text */
#define FT_GZIP_HEAD_CRC     0x02 /* bit 1 set: header CRC present */
#define FT_GZIP_EXTRA_FIELD  0x04 /* bit 2 set: extra field present */
#define FT_GZIP_ORIG_NAME    0x08 /* bit 3 set: original file name present */
#define FT_GZIP_COMMENT      0x10 /* bit 4 set: file comment present */
#define FT_GZIP_RESERVED     0xE0 /* bits 5..7: reserved */


  /* check and skip .gz header - we don't support `transparent' compression */
  static FT_Error
  ft_gzip_check_header( FT_Stream  stream )
  {
    FT_Error  error;
    FT_Byte   head[4];


    if ( FT_STREAM_SEEK( 0 )       ||
         FT_STREAM_READ( head, 4 ) )
      goto Exit;

    /* head[0] && head[1] are the magic numbers;    */
    /* head[2] is the method, and head[3] the flags */
    if ( head[0] != 0x1F              ||
         head[1] != 0x8B              ||
         head[2] != Z_DEFLATED        ||
        (head[3] & FT_GZIP_RESERVED)  )
    {
      error = FT_THROW( Invalid_File_Format );
      goto Exit;
    }

    /* skip time, xflags and os code */
    (void)FT_STREAM_SKIP( 6 );

    /* skip the extra field */
    if ( head[3] & FT_GZIP_EXTRA_FIELD )
    {
      FT_UInt  len;


      if ( FT_READ_USHORT_LE( len ) ||
           FT_STREAM_SKIP( len )    )
        goto Exit;
    }

    /* skip original file name */
    if ( head[3] & FT_GZIP_ORIG_NAME )
      for (;;)
      {
        FT_UInt  c;


        if ( FT_READ_BYTE( c ) )
          goto Exit;

        if ( c == 0 )
          break;
      }

    /* skip .gz comment */
    if ( head[3] & FT_GZIP_COMMENT )
      for (;;)
      {
        FT_UInt  c;


        if ( FT_READ_BYTE( c ) )
          goto Exit;

        if ( c == 0 )
          break;
      }

    /* skip CRC */
    if ( head[3] & FT_GZIP_HEAD_CRC )
      if ( FT_STREAM_SKIP( 2 ) )
        goto Exit;

  Exit:
    return error;
  }


  static FT_Error
  ft_gzip_file_init( FT_GZipFile  zip,
                     FT_Stream    stream,
                     FT_Stream    source )
  {
    z_stream*  zstream = &zip->zstream;
    FT_Error   error   = FT_Err_Ok;


    zip->stream = stream;
    zip->source = source;
    zip->memory = stream->memory;

    zip->limit  = zip->buffer + FT_GZIP_BUFFER_SIZE;
    zip->cursor = zip->limit;
    zip->pos    = 0;

    /* check and skip .gz header */
    {
      stream = source;

      error = ft_gzip_check_header( stream );
      if ( error )
        goto Exit;

      zip->start = FT_STREAM_POS();
    }

    /* initialize zlib -- there is no zlib header in the compressed stream */
    zstream->zalloc = ft_gzip_alloc;
    zstream->zfree  = ft_gzip_free;
    zstream->opaque = stream->memory;

    zstream->avail_in = 0;
    zstream->next_in  = zip->buffer;

    if ( inflateInit2( zstream, -MAX_WBITS ) != Z_OK ||
         !zstream->next_in                           )
      error = FT_THROW( Invalid_File_Format );

  Exit:
    return error;
  }


  static void
  ft_gzip_file_done( FT_GZipFile  zip )
  {
    z_stream*  zstream = &zip->zstream;


    inflateEnd( zstream );

    /* clear the rest */
    zstream->zalloc    = NULL;
    zstream->zfree     = NULL;
    zstream->opaque    = NULL;
    zstream->next_in   = NULL;
    zstream->next_out  = NULL;
    zstream->avail_in  = 0;
    zstream->avail_out = 0;

    zip->memory = NULL;
    zip->source = NULL;
    zip->stream = NULL;
  }


  static FT_Error
  ft_gzip_file_reset( FT_GZipFile  zip )
  {
    FT_Stream  stream = zip->source;
    FT_Error   error;


    if ( !FT_STREAM_SEEK( zip->start ) )
    {
      z_stream*  zstream = &zip->zstream;


      inflateReset( zstream );

      zstream->avail_in  = 0;
      zstream->next_in   = zip->input;
      zstream->avail_out = 0;
      zstream->next_out  = zip->buffer;

      zip->limit  = zip->buffer + FT_GZIP_BUFFER_SIZE;
      zip->cursor = zip->limit;
      zip->pos    = 0;
    }

    return error;
  }


  static FT_Error
  ft_gzip_file_fill_input( FT_GZipFile  zip )
  {
    z_stream*  zstream = &zip->zstream;
    FT_Stream  stream  = zip->source;
    FT_ULong   size;


    if ( stream->read )
    {
      size = stream->read( stream, stream->pos, zip->input,
                           FT_GZIP_BUFFER_SIZE );
      if ( size == 0 )
      {
        zip->limit = zip->cursor;
        return FT_THROW( Invalid_Stream_Operation );
      }
    }
    else
    {
      size = stream->size - stream->pos;
      if ( size > FT_GZIP_BUFFER_SIZE )
        size = FT_GZIP_BUFFER_SIZE;

      if ( size == 0 )
      {
        zip->limit = zip->cursor;
        return FT_THROW( Invalid_Stream_Operation );
      }

      FT_MEM_COPY( zip->input, stream->base + stream->pos, size );
    }
    stream->pos += size;

    zstream->next_in  = zip->input;
    zstream->avail_in = size;

    return FT_Err_Ok;
  }


  static FT_Error
  ft_gzip_file_fill_output( FT_GZipFile  zip )
  {
    z_stream*  zstream = &zip->zstream;
    FT_Error   error   = FT_Err_Ok;


    zip->cursor        = zip->buffer;
    zstream->next_out  = zip->cursor;
    zstream->avail_out = FT_GZIP_BUFFER_SIZE;

    while ( zstream->avail_out > 0 )
    {
      int  err;


      if ( zstream->avail_in == 0 )
      {
        error = ft_gzip_file_fill_input( zip );
        if ( error )
          break;
      }

      err = inflate( zstream, Z_NO_FLUSH );

      if ( err == Z_STREAM_END )
      {
        zip->limit = zstream->next_out;
        if ( zip->limit == zip->cursor )
          error = FT_THROW( Invalid_Stream_Operation );
        break;
      }
      else if ( err != Z_OK )
      {
        zip->limit = zip->cursor;
        error      = FT_THROW( Invalid_Stream_Operation );
        break;
      }
    }

    return error;
  }


  /* fill output buffer; `count' must be <= FT_GZIP_BUFFER_SIZE */
  static FT_Error
  ft_gzip_file_skip_output( FT_GZipFile  zip,
                            FT_ULong     count )
  {
    FT_Error  error = FT_Err_Ok;


    for (;;)
    {
      FT_ULong  delta = (FT_ULong)( zip->limit - zip->cursor );


      if ( delta >= count )
        delta = count;

      zip->cursor += delta;
      zip->pos    += delta;

      count -= delta;
      if ( count == 0 )
        break;

      error = ft_gzip_file_fill_output( zip );
      if ( error )
        break;
    }

    return error;
  }


  static FT_ULong
  ft_gzip_file_io( FT_GZipFile  zip,
                   FT_ULong     pos,
                   FT_Byte*     buffer,
                   FT_ULong     count )
  {
    FT_ULong  result = 0;
    FT_Error  error;


    /* Reset inflate stream if we're seeking backwards.        */
    /* Yes, that is not too efficient, but it saves memory :-) */
    if ( pos < zip->pos )
    {
      error = ft_gzip_file_reset( zip );
      if ( error )
        goto Exit;
    }

    /* skip unwanted bytes */
    if ( pos > zip->pos )
    {
      error = ft_gzip_file_skip_output( zip, (FT_ULong)( pos - zip->pos ) );
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

      FT_MEM_COPY( buffer, zip->cursor, delta );
      buffer      += delta;
      result      += delta;
      zip->cursor += delta;
      zip->pos    += delta;

      count -= delta;
      if ( count == 0 )
        break;

      error = ft_gzip_file_fill_output( zip );
      if ( error )
        break;
    }

  Exit:
    return result;
  }


/***************************************************************************/
/***************************************************************************/
/*****                                                                 *****/
/*****               G Z   E M B E D D I N G   S T R E A M             *****/
/*****                                                                 *****/
/***************************************************************************/
/***************************************************************************/

  static void
  ft_gzip_stream_close( FT_Stream  stream )
  {
    FT_GZipFile  zip    = (FT_GZipFile)stream->descriptor.pointer;
    FT_Memory    memory = stream->memory;


    if ( zip )
    {
      /* finalize gzip file descriptor */
      ft_gzip_file_done( zip );

      FT_FREE( zip );

      stream->descriptor.pointer = NULL;
    }

    if ( !stream->read )
      FT_FREE( stream->base );
  }


  static unsigned long
  ft_gzip_stream_io( FT_Stream       stream,
                     unsigned long   offset,
                     unsigned char*  buffer,
                     unsigned long   count )
  {
    FT_GZipFile  zip = (FT_GZipFile)stream->descriptor.pointer;


    return ft_gzip_file_io( zip, offset, buffer, count );
  }


  static FT_ULong
  ft_gzip_get_uncompressed_size( FT_Stream  stream )
  {
    FT_Error  error;
    FT_ULong  old_pos;
    FT_ULong  result = 0;


    old_pos = stream->pos;
    if ( !FT_Stream_Seek( stream, stream->size - 4 ) )
    {
      result = FT_Stream_ReadULongLE( stream, &error );
      if ( error )
        result = 0;

      (void)FT_Stream_Seek( stream, old_pos );
    }

    return result;
  }


  /* documentation is in ftgzip.h */

  FT_EXPORT_DEF( FT_Error )
  FT_Stream_OpenGzip( FT_Stream  stream,
                      FT_Stream  source )
  {
    FT_Error     error;
    FT_Memory    memory;
    FT_GZipFile  zip = NULL;


    if ( !stream || !source )
    {
      error = FT_THROW( Invalid_Stream_Handle );
      goto Exit;
    }

    memory = source->memory;

    /*
     * check the header right now; this prevents allocating un-necessary
     * objects when we don't need them
     */
    error = ft_gzip_check_header( source );
    if ( error )
      goto Exit;

    FT_ZERO( stream );
    stream->memory = memory;

    if ( !FT_QNEW( zip ) )
    {
      error = ft_gzip_file_init( zip, stream, source );
      if ( error )
      {
        FT_FREE( zip );
        goto Exit;
      }

      stream->descriptor.pointer = zip;
    }

    /*
     * We use the following trick to try to dramatically improve the
     * performance while dealing with small files.  If the original stream
     * size is less than a certain threshold, we try to load the whole font
     * file into memory.  This saves us from using the 32KB buffer needed
     * to inflate the file, plus the two 4KB intermediate input/output
     * buffers used in the `FT_GZipFile' structure.
     */
    {
      FT_ULong  zip_size = ft_gzip_get_uncompressed_size( source );


      if ( zip_size != 0 && zip_size < 40 * 1024 )
      {
        FT_Byte*  zip_buff = NULL;


        if ( !FT_QALLOC( zip_buff, zip_size ) )
        {
          FT_ULong  count;


          count = ft_gzip_file_io( zip, 0, zip_buff, zip_size );
          if ( count == zip_size )
          {
            ft_gzip_file_done( zip );
            FT_FREE( zip );

            stream->descriptor.pointer = NULL;

            stream->size  = zip_size;
            stream->pos   = 0;
            stream->base  = zip_buff;
            stream->read  = NULL;
            stream->close = ft_gzip_stream_close;

            goto Exit;
          }

          ft_gzip_file_io( zip, 0, NULL, 0 );
          FT_FREE( zip_buff );
        }
        error = FT_Err_Ok;
      }

      if ( zip_size )
        stream->size = zip_size;
      else
        stream->size  = 0x7FFFFFFFL;  /* don't know the real size! */
    }

    stream->pos   = 0;
    stream->base  = NULL;
    stream->read  = ft_gzip_stream_io;
    stream->close = ft_gzip_stream_close;

  Exit:
    return error;
  }


  /* documentation is in ftgzip.h */

  FT_EXPORT_DEF( FT_Error )
  FT_Gzip_Uncompress( FT_Memory       memory,
                      FT_Byte*        output,
                      FT_ULong*       output_len,
                      const FT_Byte*  input,
                      FT_ULong        input_len )
  {
    z_stream  stream;
    int       err;


    /* check for `input' delayed to `inflate' */

    if ( !memory || !output_len || !output )
      return FT_THROW( Invalid_Argument );

    /* this function is modeled after zlib's `uncompress' function */

    stream.next_in  = (Bytef*)input;
    stream.avail_in = (uInt)input_len;

    stream.next_out  = output;
    stream.avail_out = (uInt)*output_len;

    stream.zalloc = ft_gzip_alloc;
    stream.zfree  = ft_gzip_free;
    stream.opaque = memory;

    err = inflateInit2( &stream, MAX_WBITS|32 );

    if ( err != Z_OK )
      return FT_THROW( Invalid_Argument );

    err = inflate( &stream, Z_FINISH );
    if ( err != Z_STREAM_END )
    {
      inflateEnd( &stream );
      if ( err == Z_OK )
        err = Z_BUF_ERROR;
    }
    else
    {
      *output_len = stream.total_out;

      err = inflateEnd( &stream );
    }

    if ( err == Z_MEM_ERROR )
      return FT_THROW( Out_Of_Memory );

    if ( err == Z_BUF_ERROR )
      return FT_THROW( Array_Too_Large );

    if ( err == Z_DATA_ERROR )
      return FT_THROW( Invalid_Table );

    if ( err == Z_NEED_DICT )
      return FT_THROW( Invalid_Table );

    return FT_Err_Ok;
  }


#else /* !FT_CONFIG_OPTION_USE_ZLIB */

  FT_EXPORT_DEF( FT_Error )
  FT_Stream_OpenGzip( FT_Stream  stream,
                      FT_Stream  source )
  {
    FT_UNUSED( stream );
    FT_UNUSED( source );

    return FT_THROW( Unimplemented_Feature );
  }


  FT_EXPORT_DEF( FT_Error )
  FT_Gzip_Uncompress( FT_Memory       memory,
                      FT_Byte*        output,
                      FT_ULong*       output_len,
                      const FT_Byte*  input,
                      FT_ULong        input_len )
  {
    FT_UNUSED( memory );
    FT_UNUSED( output );
    FT_UNUSED( output_len );
    FT_UNUSED( input );
    FT_UNUSED( input_len );

    return FT_THROW( Unimplemented_Feature );
  }

#endif /* !FT_CONFIG_OPTION_USE_ZLIB */


/* END */
