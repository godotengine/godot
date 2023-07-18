/****************************************************************************
 *
 * sfwoff.c
 *
 *   WOFF format management (base).
 *
 * Copyright (C) 1996-2023 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#include "sfwoff.h"
#include <freetype/tttags.h>
#include <freetype/internal/ftdebug.h>
#include <freetype/internal/ftstream.h>
#include <freetype/ftgzip.h>


#ifdef FT_CONFIG_OPTION_USE_ZLIB


  /**************************************************************************
   *
   * The macro FT_COMPONENT is used in trace mode.  It is an implicit
   * parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log
   * messages during execution.
   */
#undef  FT_COMPONENT
#define FT_COMPONENT  sfwoff


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


  static void
  sfnt_stream_close( FT_Stream  stream )
  {
    FT_Memory  memory = stream->memory;


    FT_FREE( stream->base );

    stream->size  = 0;
    stream->close = NULL;
  }


  FT_COMPARE_DEF( int )
  compare_offsets( const void*  a,
                   const void*  b )
  {
    WOFF_Table  table1 = *(WOFF_Table*)a;
    WOFF_Table  table2 = *(WOFF_Table*)b;

    FT_ULong  offset1 = table1->Offset;
    FT_ULong  offset2 = table2->Offset;


    if ( offset1 > offset2 )
      return 1;
    else if ( offset1 < offset2 )
      return -1;
    else
      return 0;
  }


  /* Replace `face->root.stream' with a stream containing the extracted */
  /* SFNT of a WOFF font.                                               */

  FT_LOCAL_DEF( FT_Error )
  woff_open_font( FT_Stream  stream,
                  TT_Face    face )
  {
    FT_Memory       memory = stream->memory;
    FT_Error        error  = FT_Err_Ok;

    WOFF_HeaderRec  woff;
    WOFF_Table      tables  = NULL;
    WOFF_Table*     indices = NULL;

    FT_ULong        woff_offset;

    FT_Byte*        sfnt        = NULL;
    FT_Stream       sfnt_stream = NULL;

    FT_Byte*        sfnt_header;
    FT_ULong        sfnt_offset;

    FT_Int          nn;
    FT_Tag          old_tag = 0;

    static const FT_Frame_Field  woff_header_fields[] =
    {
#undef  FT_STRUCTURE
#define FT_STRUCTURE  WOFF_HeaderRec

      FT_FRAME_START( 44 ),
        FT_FRAME_ULONG ( signature ),
        FT_FRAME_ULONG ( flavor ),
        FT_FRAME_ULONG ( length ),
        FT_FRAME_USHORT( num_tables ),
        FT_FRAME_USHORT( reserved ),
        FT_FRAME_ULONG ( totalSfntSize ),
        FT_FRAME_USHORT( majorVersion ),
        FT_FRAME_USHORT( minorVersion ),
        FT_FRAME_ULONG ( metaOffset ),
        FT_FRAME_ULONG ( metaLength ),
        FT_FRAME_ULONG ( metaOrigLength ),
        FT_FRAME_ULONG ( privOffset ),
        FT_FRAME_ULONG ( privLength ),
      FT_FRAME_END
    };


    FT_ASSERT( stream == face->root.stream );
    FT_ASSERT( FT_STREAM_POS() == 0 );

    if ( FT_STREAM_READ_FIELDS( woff_header_fields, &woff ) )
      return error;

    /* Make sure we don't recurse back here or hit TTC code. */
    if ( woff.flavor == TTAG_wOFF || woff.flavor == TTAG_ttcf )
      return FT_THROW( Invalid_Table );

    /* Miscellaneous checks. */
    if ( woff.length != stream->size                              ||
         woff.num_tables == 0                                     ||
         44 + woff.num_tables * 20UL >= woff.length               ||
         12 + woff.num_tables * 16UL >= woff.totalSfntSize        ||
         ( woff.totalSfntSize & 3 ) != 0                          ||
         ( woff.metaOffset == 0 && ( woff.metaLength != 0     ||
                                     woff.metaOrigLength != 0 ) ) ||
         ( woff.metaLength != 0 && woff.metaOrigLength == 0 )     ||
         ( woff.privOffset == 0 && woff.privLength != 0 )         )
    {
      FT_ERROR(( "woff_font_open: invalid WOFF header\n" ));
      return FT_THROW( Invalid_Table );
    }

    /* Don't trust `totalSfntSize' before thorough checks. */
    if ( FT_QALLOC( sfnt, 12 ) || FT_NEW( sfnt_stream ) )
      goto Exit;

    sfnt_header = sfnt;

    /* Write sfnt header. */
    {
      FT_UInt  searchRange, entrySelector, rangeShift, x;


      x             = woff.num_tables;
      entrySelector = 0;
      while ( x )
      {
        x            >>= 1;
        entrySelector += 1;
      }
      entrySelector--;

      searchRange = ( 1 << entrySelector ) * 16;
      rangeShift  = woff.num_tables * 16 - searchRange;

      WRITE_ULONG ( sfnt_header, woff.flavor );
      WRITE_USHORT( sfnt_header, woff.num_tables );
      WRITE_USHORT( sfnt_header, searchRange );
      WRITE_USHORT( sfnt_header, entrySelector );
      WRITE_USHORT( sfnt_header, rangeShift );
    }

    /* While the entries in the sfnt header must be sorted by the */
    /* tag value, the tables themselves are not.  We thus have to */
    /* sort them by offset and check that they don't overlap.     */

    if ( FT_QNEW_ARRAY( tables, woff.num_tables )  ||
         FT_QNEW_ARRAY( indices, woff.num_tables ) )
      goto Exit;

    FT_TRACE2(( "\n" ));
    FT_TRACE2(( "  tag    offset    compLen  origLen  checksum\n" ));
    FT_TRACE2(( "  -------------------------------------------\n" ));

    if ( FT_FRAME_ENTER( 20L * woff.num_tables ) )
      goto Exit;

    for ( nn = 0; nn < woff.num_tables; nn++ )
    {
      WOFF_Table  table = tables + nn;

      table->Tag        = FT_GET_TAG4();
      table->Offset     = FT_GET_ULONG();
      table->CompLength = FT_GET_ULONG();
      table->OrigLength = FT_GET_ULONG();
      table->CheckSum   = FT_GET_ULONG();

      FT_TRACE2(( "  %c%c%c%c  %08lx  %08lx  %08lx  %08lx\n",
                  (FT_Char)( table->Tag >> 24 ),
                  (FT_Char)( table->Tag >> 16 ),
                  (FT_Char)( table->Tag >> 8  ),
                  (FT_Char)( table->Tag       ),
                  table->Offset,
                  table->CompLength,
                  table->OrigLength,
                  table->CheckSum ));

      if ( table->Tag <= old_tag )
      {
        FT_FRAME_EXIT();

        FT_ERROR(( "woff_font_open: table tags are not sorted\n" ));
        error = FT_THROW( Invalid_Table );
        goto Exit;
      }

      old_tag     = table->Tag;
      indices[nn] = table;
    }

    FT_FRAME_EXIT();

    /* Sort by offset. */

    ft_qsort( indices,
              woff.num_tables,
              sizeof ( WOFF_Table ),
              compare_offsets );

    /* Check offsets and lengths. */

    woff_offset = 44 + woff.num_tables * 20L;
    sfnt_offset = 12 + woff.num_tables * 16L;

    for ( nn = 0; nn < woff.num_tables; nn++ )
    {
      WOFF_Table  table = indices[nn];


      if ( table->Offset != woff_offset                         ||
           table->CompLength > woff.length                      ||
           table->Offset > woff.length - table->CompLength      ||
           table->OrigLength > woff.totalSfntSize               ||
           sfnt_offset > woff.totalSfntSize - table->OrigLength ||
           table->CompLength > table->OrigLength                )
      {
        FT_ERROR(( "woff_font_open: invalid table offsets\n" ));
        error = FT_THROW( Invalid_Table );
        goto Exit;
      }

      table->OrigOffset = sfnt_offset;

      /* The offsets must be multiples of 4. */
      woff_offset += ( table->CompLength + 3 ) & ~3U;
      sfnt_offset += ( table->OrigLength + 3 ) & ~3U;
    }

    /*
     * Final checks!
     *
     * We don't decode and check the metadata block.
     * We don't check table checksums either.
     * But other than those, I think we implement all
     * `MUST' checks from the spec.
     */

    if ( woff.metaOffset )
    {
      if ( woff.metaOffset != woff_offset                  ||
           woff.metaOffset + woff.metaLength > woff.length )
      {
        FT_ERROR(( "woff_font_open:"
                   " invalid `metadata' offset or length\n" ));
        error = FT_THROW( Invalid_Table );
        goto Exit;
      }

      /* We have padding only ... */
      woff_offset += woff.metaLength;
    }

    if ( woff.privOffset )
    {
      /* ... if it isn't the last block. */
      woff_offset = ( woff_offset + 3 ) & ~3U;

      if ( woff.privOffset != woff_offset                  ||
           woff.privOffset + woff.privLength > woff.length )
      {
        FT_ERROR(( "woff_font_open: invalid `private' offset or length\n" ));
        error = FT_THROW( Invalid_Table );
        goto Exit;
      }

      /* No padding for the last block. */
      woff_offset += woff.privLength;
    }

    if ( sfnt_offset != woff.totalSfntSize ||
         woff_offset != woff.length        )
    {
      FT_ERROR(( "woff_font_open: invalid `sfnt' table structure\n" ));
      error = FT_THROW( Invalid_Table );
      goto Exit;
    }

    /* Now use `totalSfntSize'. */
    if ( FT_QREALLOC( sfnt, 12, woff.totalSfntSize ) )
      goto Exit;

    sfnt_header = sfnt + 12;

    /* Write the tables. */

    for ( nn = 0; nn < woff.num_tables; nn++ )
    {
      WOFF_Table  table = tables + nn;


      /* Write SFNT table entry. */
      WRITE_ULONG( sfnt_header, table->Tag );
      WRITE_ULONG( sfnt_header, table->CheckSum );
      WRITE_ULONG( sfnt_header, table->OrigOffset );
      WRITE_ULONG( sfnt_header, table->OrigLength );

      /* Write table data. */
      if ( FT_STREAM_SEEK( table->Offset )     ||
           FT_FRAME_ENTER( table->CompLength ) )
        goto Exit;

      if ( table->CompLength == table->OrigLength )
      {
        /* Uncompressed data; just copy. */
        ft_memcpy( sfnt + table->OrigOffset,
                   stream->cursor,
                   table->OrigLength );
      }
      else
      {
        /* Uncompress with zlib. */
        FT_ULong  output_len = table->OrigLength;


        error = FT_Gzip_Uncompress( memory,
                                    sfnt + table->OrigOffset, &output_len,
                                    stream->cursor, table->CompLength );
        if ( error )
          goto Exit1;
        if ( output_len != table->OrigLength )
        {
          FT_ERROR(( "woff_font_open: compressed table length mismatch\n" ));
          error = FT_THROW( Invalid_Table );
          goto Exit1;
        }
      }

      FT_FRAME_EXIT();

      /* We don't check whether the padding bytes in the WOFF file are     */
      /* actually '\0'.  For the output, however, we do set them properly. */
      sfnt_offset = table->OrigOffset + table->OrigLength;
      while ( sfnt_offset & 3 )
      {
        sfnt[sfnt_offset] = '\0';
        sfnt_offset++;
      }
    }

    /* Ok!  Finally ready.  Swap out stream and return. */
    FT_Stream_OpenMemory( sfnt_stream, sfnt, woff.totalSfntSize );
    sfnt_stream->memory = stream->memory;
    sfnt_stream->close  = sfnt_stream_close;

    FT_Stream_Free(
      face->root.stream,
      ( face->root.face_flags & FT_FACE_FLAG_EXTERNAL_STREAM ) != 0 );

    face->root.stream = sfnt_stream;

    face->root.face_flags &= ~FT_FACE_FLAG_EXTERNAL_STREAM;

  Exit:
    FT_FREE( tables );
    FT_FREE( indices );

    if ( error )
    {
      FT_FREE( sfnt );
      FT_Stream_Close( sfnt_stream );
      FT_FREE( sfnt_stream );
    }

    return error;

  Exit1:
    FT_FRAME_EXIT();
    goto Exit;
  }


#undef WRITE_USHORT
#undef WRITE_ULONG

#else /* !FT_CONFIG_OPTION_USE_ZLIB */

  /* ANSI C doesn't like empty source files */
  typedef int  sfwoff_dummy_;

#endif /* !FT_CONFIG_OPTION_USE_ZLIB */


/* END */
