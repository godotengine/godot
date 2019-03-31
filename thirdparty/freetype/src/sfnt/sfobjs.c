/****************************************************************************
 *
 * sfobjs.c
 *
 *   SFNT object management (base).
 *
 * Copyright (C) 1996-2019 by
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
#include "sfobjs.h"
#include "ttload.h"
#include "ttcmap.h"
#include "ttkern.h"
#include FT_INTERNAL_SFNT_H
#include FT_INTERNAL_DEBUG_H
#include FT_TRUETYPE_IDS_H
#include FT_TRUETYPE_TAGS_H
#include FT_SERVICE_POSTSCRIPT_CMAPS_H
#include FT_SFNT_NAMES_H
#include FT_GZIP_H

#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
#include FT_SERVICE_MULTIPLE_MASTERS_H
#include FT_SERVICE_METRICS_VARIATIONS_H
#endif

#include "sferrors.h"

#ifdef TT_CONFIG_OPTION_BDF
#include "ttbdf.h"
#endif


  /**************************************************************************
   *
   * The macro FT_COMPONENT is used in trace mode.  It is an implicit
   * parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log
   * messages during execution.
   */
#undef  FT_COMPONENT
#define FT_COMPONENT  sfobjs



  /* convert a UTF-16 name entry to ASCII */
  static FT_String*
  tt_name_ascii_from_utf16( TT_Name    entry,
                            FT_Memory  memory )
  {
    FT_String*  string = NULL;
    FT_UInt     len, code, n;
    FT_Byte*    read   = (FT_Byte*)entry->string;
    FT_Error    error;


    len = (FT_UInt)entry->stringLength / 2;

    if ( FT_NEW_ARRAY( string, len + 1 ) )
      return NULL;

    for ( n = 0; n < len; n++ )
    {
      code = FT_NEXT_USHORT( read );

      if ( code == 0 )
        break;

      if ( code < 32 || code > 127 )
        code = '?';

      string[n] = (char)code;
    }

    string[n] = 0;

    return string;
  }


  /* convert an Apple Roman or symbol name entry to ASCII */
  static FT_String*
  tt_name_ascii_from_other( TT_Name    entry,
                            FT_Memory  memory )
  {
    FT_String*  string = NULL;
    FT_UInt     len, code, n;
    FT_Byte*    read   = (FT_Byte*)entry->string;
    FT_Error    error;


    len = (FT_UInt)entry->stringLength;

    if ( FT_NEW_ARRAY( string, len + 1 ) )
      return NULL;

    for ( n = 0; n < len; n++ )
    {
      code = *read++;

      if ( code == 0 )
        break;

      if ( code < 32 || code > 127 )
        code = '?';

      string[n] = (char)code;
    }

    string[n] = 0;

    return string;
  }


  typedef FT_String*  (*TT_Name_ConvertFunc)( TT_Name    entry,
                                              FT_Memory  memory );


  /* documentation is in sfnt.h */

  FT_LOCAL_DEF( FT_Error )
  tt_face_get_name( TT_Face      face,
                    FT_UShort    nameid,
                    FT_String**  name )
  {
    FT_Memory   memory = face->root.memory;
    FT_Error    error  = FT_Err_Ok;
    FT_String*  result = NULL;
    FT_UShort   n;
    TT_Name     rec;

    FT_Int  found_apple         = -1;
    FT_Int  found_apple_roman   = -1;
    FT_Int  found_apple_english = -1;
    FT_Int  found_win           = -1;
    FT_Int  found_unicode       = -1;

    FT_Bool  is_english = 0;

    TT_Name_ConvertFunc  convert;


    FT_ASSERT( name );

    rec = face->name_table.names;
    for ( n = 0; n < face->num_names; n++, rec++ )
    {
      /* According to the OpenType 1.3 specification, only Microsoft or  */
      /* Apple platform IDs might be used in the `name' table.  The      */
      /* `Unicode' platform is reserved for the `cmap' table, and the    */
      /* `ISO' one is deprecated.                                        */
      /*                                                                 */
      /* However, the Apple TrueType specification doesn't say the same  */
      /* thing and goes to suggest that all Unicode `name' table entries */
      /* should be coded in UTF-16 (in big-endian format I suppose).     */
      /*                                                                 */
      if ( rec->nameID == nameid && rec->stringLength > 0 )
      {
        switch ( rec->platformID )
        {
        case TT_PLATFORM_APPLE_UNICODE:
        case TT_PLATFORM_ISO:
          /* there is `languageID' to check there.  We should use this */
          /* field only as a last solution when nothing else is        */
          /* available.                                                */
          /*                                                           */
          found_unicode = n;
          break;

        case TT_PLATFORM_MACINTOSH:
          /* This is a bit special because some fonts will use either    */
          /* an English language id, or a Roman encoding id, to indicate */
          /* the English version of its font name.                       */
          /*                                                             */
          if ( rec->languageID == TT_MAC_LANGID_ENGLISH )
            found_apple_english = n;
          else if ( rec->encodingID == TT_MAC_ID_ROMAN )
            found_apple_roman = n;
          break;

        case TT_PLATFORM_MICROSOFT:
          /* we only take a non-English name when there is nothing */
          /* else available in the font                            */
          /*                                                       */
          if ( found_win == -1 || ( rec->languageID & 0x3FF ) == 0x009 )
          {
            switch ( rec->encodingID )
            {
            case TT_MS_ID_SYMBOL_CS:
            case TT_MS_ID_UNICODE_CS:
            case TT_MS_ID_UCS_4:
              is_english = FT_BOOL( ( rec->languageID & 0x3FF ) == 0x009 );
              found_win  = n;
              break;

            default:
              ;
            }
          }
          break;

        default:
          ;
        }
      }
    }

    found_apple = found_apple_roman;
    if ( found_apple_english >= 0 )
      found_apple = found_apple_english;

    /* some fonts contain invalid Unicode or Macintosh formatted entries; */
    /* we will thus favor names encoded in Windows formats if available   */
    /* (provided it is an English name)                                   */
    /*                                                                    */
    convert = NULL;
    if ( found_win >= 0 && !( found_apple >= 0 && !is_english ) )
    {
      rec = face->name_table.names + found_win;
      switch ( rec->encodingID )
      {
        /* all Unicode strings are encoded using UTF-16BE */
      case TT_MS_ID_UNICODE_CS:
      case TT_MS_ID_SYMBOL_CS:
        convert = tt_name_ascii_from_utf16;
        break;

      case TT_MS_ID_UCS_4:
        /* Apparently, if this value is found in a name table entry, it is */
        /* documented as `full Unicode repertoire'.  Experience with the   */
        /* MsGothic font shipped with Windows Vista shows that this really */
        /* means UTF-16 encoded names (UCS-4 values are only used within   */
        /* charmaps).                                                      */
        convert = tt_name_ascii_from_utf16;
        break;

      default:
        ;
      }
    }
    else if ( found_apple >= 0 )
    {
      rec     = face->name_table.names + found_apple;
      convert = tt_name_ascii_from_other;
    }
    else if ( found_unicode >= 0 )
    {
      rec     = face->name_table.names + found_unicode;
      convert = tt_name_ascii_from_utf16;
    }

    if ( rec && convert )
    {
      if ( !rec->string )
      {
        FT_Stream  stream = face->name_table.stream;


        if ( FT_QNEW_ARRAY ( rec->string, rec->stringLength ) ||
             FT_STREAM_SEEK( rec->stringOffset )              ||
             FT_STREAM_READ( rec->string, rec->stringLength ) )
        {
          FT_FREE( rec->string );
          rec->stringLength = 0;
          result            = NULL;
          goto Exit;
        }
      }

      result = convert( rec, memory );
    }

  Exit:
    *name = result;
    return error;
  }


  static FT_Encoding
  sfnt_find_encoding( int  platform_id,
                      int  encoding_id )
  {
    typedef struct  TEncoding_
    {
      int          platform_id;
      int          encoding_id;
      FT_Encoding  encoding;

    } TEncoding;

    static
    const TEncoding  tt_encodings[] =
    {
      { TT_PLATFORM_ISO,           -1,                  FT_ENCODING_UNICODE },

      { TT_PLATFORM_APPLE_UNICODE, -1,                  FT_ENCODING_UNICODE },

      { TT_PLATFORM_MACINTOSH,     TT_MAC_ID_ROMAN,     FT_ENCODING_APPLE_ROMAN },

      { TT_PLATFORM_MICROSOFT,     TT_MS_ID_SYMBOL_CS,  FT_ENCODING_MS_SYMBOL },
      { TT_PLATFORM_MICROSOFT,     TT_MS_ID_UCS_4,      FT_ENCODING_UNICODE },
      { TT_PLATFORM_MICROSOFT,     TT_MS_ID_UNICODE_CS, FT_ENCODING_UNICODE },
      { TT_PLATFORM_MICROSOFT,     TT_MS_ID_SJIS,       FT_ENCODING_SJIS },
      { TT_PLATFORM_MICROSOFT,     TT_MS_ID_PRC,        FT_ENCODING_PRC },
      { TT_PLATFORM_MICROSOFT,     TT_MS_ID_BIG_5,      FT_ENCODING_BIG5 },
      { TT_PLATFORM_MICROSOFT,     TT_MS_ID_WANSUNG,    FT_ENCODING_WANSUNG },
      { TT_PLATFORM_MICROSOFT,     TT_MS_ID_JOHAB,      FT_ENCODING_JOHAB }
    };

    const TEncoding  *cur, *limit;


    cur   = tt_encodings;
    limit = cur + sizeof ( tt_encodings ) / sizeof ( tt_encodings[0] );

    for ( ; cur < limit; cur++ )
    {
      if ( cur->platform_id == platform_id )
      {
        if ( cur->encoding_id == encoding_id ||
             cur->encoding_id == -1          )
          return cur->encoding;
      }
    }

    return FT_ENCODING_NONE;
  }


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
    stream->base  = NULL;
    stream->close = NULL;
  }


  FT_CALLBACK_DEF( int )
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

  static FT_Error
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
    FT_ULong        old_tag = 0;

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
    if ( FT_ALLOC( sfnt, 12 + woff.num_tables * 16UL ) ||
         FT_NEW( sfnt_stream )                         )
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

    if ( FT_NEW_ARRAY( tables, woff.num_tables )  ||
         FT_NEW_ARRAY( indices, woff.num_tables ) )
      goto Exit;

    FT_TRACE2(( "\n"
                "  tag    offset    compLen  origLen  checksum\n"
                "  -------------------------------------------\n" ));

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
    if ( FT_REALLOC( sfnt,
                     12 + woff.num_tables * 16UL,
                     woff.totalSfntSize ) )
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
#ifdef FT_CONFIG_OPTION_USE_ZLIB

        /* Uncompress with zlib. */
        FT_ULong  output_len = table->OrigLength;


        error = FT_Gzip_Uncompress( memory,
                                    sfnt + table->OrigOffset, &output_len,
                                    stream->cursor, table->CompLength );
        if ( error )
          goto Exit;
        if ( output_len != table->OrigLength )
        {
          FT_ERROR(( "woff_font_open: compressed table length mismatch\n" ));
          error = FT_THROW( Invalid_Table );
          goto Exit;
        }

#else /* !FT_CONFIG_OPTION_USE_ZLIB */

        error = FT_THROW( Unimplemented_Feature );
        goto Exit;

#endif /* !FT_CONFIG_OPTION_USE_ZLIB */
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
  }


#undef WRITE_USHORT
#undef WRITE_ULONG


  /* Fill in face->ttc_header.  If the font is not a TTC, it is */
  /* synthesized into a TTC with one offset table.              */
  static FT_Error
  sfnt_open_font( FT_Stream  stream,
                  TT_Face    face )
  {
    FT_Memory  memory = stream->memory;
    FT_Error   error;
    FT_ULong   tag, offset;

    static const FT_Frame_Field  ttc_header_fields[] =
    {
#undef  FT_STRUCTURE
#define FT_STRUCTURE  TTC_HeaderRec

      FT_FRAME_START( 8 ),
        FT_FRAME_LONG( version ),
        FT_FRAME_LONG( count   ),  /* this is ULong in the specs */
      FT_FRAME_END
    };


    face->ttc_header.tag     = 0;
    face->ttc_header.version = 0;
    face->ttc_header.count   = 0;

  retry:
    offset = FT_STREAM_POS();

    if ( FT_READ_ULONG( tag ) )
      return error;

    if ( tag == TTAG_wOFF )
    {
      FT_TRACE2(( "sfnt_open_font: file is a WOFF; synthesizing SFNT\n" ));

      if ( FT_STREAM_SEEK( offset ) )
        return error;

      error = woff_open_font( stream, face );
      if ( error )
        return error;

      /* Swap out stream and retry! */
      stream = face->root.stream;
      goto retry;
    }

    if ( tag != 0x00010000UL &&
         tag != TTAG_ttcf    &&
         tag != TTAG_OTTO    &&
         tag != TTAG_true    &&
         tag != TTAG_typ1    &&
         tag != TTAG_0xA5kbd &&
         tag != TTAG_0xA5lst &&
         tag != 0x00020000UL )
    {
      FT_TRACE2(( "  not a font using the SFNT container format\n" ));
      return FT_THROW( Unknown_File_Format );
    }

    face->ttc_header.tag = TTAG_ttcf;

    if ( tag == TTAG_ttcf )
    {
      FT_Int  n;


      FT_TRACE3(( "sfnt_open_font: file is a collection\n" ));

      if ( FT_STREAM_READ_FIELDS( ttc_header_fields, &face->ttc_header ) )
        return error;

      FT_TRACE3(( "                with %ld subfonts\n",
                  face->ttc_header.count ));

      if ( face->ttc_header.count == 0 )
        return FT_THROW( Invalid_Table );

      /* a rough size estimate: let's conservatively assume that there   */
      /* is just a single table info in each subfont header (12 + 16*1 = */
      /* 28 bytes), thus we have (at least) `12 + 4*count' bytes for the */
      /* size of the TTC header plus `28*count' bytes for all subfont    */
      /* headers                                                         */
      if ( (FT_ULong)face->ttc_header.count > stream->size / ( 28 + 4 ) )
        return FT_THROW( Array_Too_Large );

      /* now read the offsets of each font in the file */
      if ( FT_NEW_ARRAY( face->ttc_header.offsets, face->ttc_header.count ) )
        return error;

      if ( FT_FRAME_ENTER( face->ttc_header.count * 4L ) )
        return error;

      for ( n = 0; n < face->ttc_header.count; n++ )
        face->ttc_header.offsets[n] = FT_GET_ULONG();

      FT_FRAME_EXIT();
    }
    else
    {
      FT_TRACE3(( "sfnt_open_font: synthesize TTC\n" ));

      face->ttc_header.version = 1 << 16;
      face->ttc_header.count   = 1;

      if ( FT_NEW( face->ttc_header.offsets ) )
        return error;

      face->ttc_header.offsets[0] = offset;
    }

    return error;
  }


  FT_LOCAL_DEF( FT_Error )
  sfnt_init_face( FT_Stream      stream,
                  TT_Face        face,
                  FT_Int         face_instance_index,
                  FT_Int         num_params,
                  FT_Parameter*  params )
  {
    FT_Error      error;
    FT_Library    library = face->root.driver->root.library;
    SFNT_Service  sfnt;
    FT_Int        face_index;


    /* for now, parameters are unused */
    FT_UNUSED( num_params );
    FT_UNUSED( params );


    sfnt = (SFNT_Service)face->sfnt;
    if ( !sfnt )
    {
      sfnt = (SFNT_Service)FT_Get_Module_Interface( library, "sfnt" );
      if ( !sfnt )
      {
        FT_ERROR(( "sfnt_init_face: cannot access `sfnt' module\n" ));
        return FT_THROW( Missing_Module );
      }

      face->sfnt       = sfnt;
      face->goto_table = sfnt->goto_table;
    }

    FT_FACE_FIND_GLOBAL_SERVICE( face, face->psnames, POSTSCRIPT_CMAPS );

#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
    if ( !face->mm )
    {
      /* we want the MM interface from the `truetype' module only */
      FT_Module  tt_module = FT_Get_Module( library, "truetype" );


      face->mm = ft_module_get_service( tt_module,
                                        FT_SERVICE_ID_MULTI_MASTERS,
                                        0 );
    }

    if ( !face->var )
    {
      /* we want the metrics variations interface */
      /* from the `truetype' module only          */
      FT_Module  tt_module = FT_Get_Module( library, "truetype" );


      face->var = ft_module_get_service( tt_module,
                                         FT_SERVICE_ID_METRICS_VARIATIONS,
                                         0 );
    }
#endif

    FT_TRACE2(( "SFNT driver\n" ));

    error = sfnt_open_font( stream, face );
    if ( error )
      return error;

    /* Stream may have changed in sfnt_open_font. */
    stream = face->root.stream;

    FT_TRACE2(( "sfnt_init_face: %08p (index %d)\n",
                face,
                face_instance_index ));

    face_index = FT_ABS( face_instance_index ) & 0xFFFF;

    /* value -(N+1) requests information on index N */
    if ( face_instance_index < 0 )
      face_index--;

    if ( face_index >= face->ttc_header.count )
    {
      if ( face_instance_index >= 0 )
        return FT_THROW( Invalid_Argument );
      else
        face_index = 0;
    }

    if ( FT_STREAM_SEEK( face->ttc_header.offsets[face_index] ) )
      return error;

    /* check whether we have a valid TrueType file */
    error = sfnt->load_font_dir( face, stream );
    if ( error )
      return error;

#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
    {
      FT_Memory  memory = face->root.memory;

      FT_ULong  fvar_len;

      FT_ULong  version;
      FT_ULong  offset;

      FT_UShort  num_axes;
      FT_UShort  axis_size;
      FT_UShort  num_instances;
      FT_UShort  instance_size;

      FT_Int  instance_index;

      FT_Byte*  default_values  = NULL;
      FT_Byte*  instance_values = NULL;


      instance_index = FT_ABS( face_instance_index ) >> 16;

      /* test whether current face is a GX font with named instances */
      if ( face->goto_table( face, TTAG_fvar, stream, &fvar_len ) ||
           fvar_len < 20                                          ||
           FT_READ_ULONG( version )                               ||
           FT_READ_USHORT( offset )                               ||
           FT_STREAM_SKIP( 2 ) /* reserved */                     ||
           FT_READ_USHORT( num_axes )                             ||
           FT_READ_USHORT( axis_size )                            ||
           FT_READ_USHORT( num_instances )                        ||
           FT_READ_USHORT( instance_size )                        )
      {
        version       = 0;
        offset        = 0;
        num_axes      = 0;
        axis_size     = 0;
        num_instances = 0;
        instance_size = 0;
      }

      /* check that the data is bound by the table length */
      if ( version != 0x00010000UL                    ||
           axis_size != 20                            ||
           num_axes == 0                              ||
           /* `num_axes' limit implied by 16-bit `instance_size' */
           num_axes > 0x3FFE                          ||
           !( instance_size == 4 + 4 * num_axes ||
              instance_size == 6 + 4 * num_axes )     ||
           /* `num_instances' limit implied by limited range of name IDs */
           num_instances > 0x7EFF                     ||
           offset                          +
             axis_size * num_axes          +
             instance_size * num_instances > fvar_len )
        num_instances = 0;
      else
        face->variation_support |= TT_FACE_FLAG_VAR_FVAR;

      /*
       * As documented in the OpenType specification, an entry for the
       * default instance may be omitted in the named instance table.  In
       * particular this means that even if there is no named instance
       * table in the font we actually do have a named instance, namely the
       * default instance.
       *
       * For consistency, we always want the default instance in our list
       * of named instances.  If it is missing, we try to synthesize it
       * later on.  Here, we have to adjust `num_instances' accordingly.
       */

      if ( ( face->variation_support & TT_FACE_FLAG_VAR_FVAR ) &&
           !( FT_ALLOC( default_values, num_axes * 4 )  ||
              FT_ALLOC( instance_values, num_axes * 4 ) )      )
      {
        /* the current stream position is 16 bytes after the table start */
        FT_ULong  array_start = FT_STREAM_POS() - 16 + offset;
        FT_ULong  default_value_offset, instance_offset;

        FT_Byte*  p;
        FT_UInt   i;


        default_value_offset = array_start + 8;
        p                    = default_values;

        for ( i = 0; i < num_axes; i++ )
        {
          (void)FT_STREAM_READ_AT( default_value_offset, p, 4 );

          default_value_offset += axis_size;
          p                    += 4;
        }

        instance_offset = array_start + axis_size * num_axes + 4;

        for ( i = 0; i < num_instances; i++ )
        {
          (void)FT_STREAM_READ_AT( instance_offset,
                                   instance_values,
                                   num_axes * 4 );

          if ( !ft_memcmp( default_values, instance_values, num_axes * 4 ) )
            break;

          instance_offset += instance_size;
        }

        if ( i == num_instances )
        {
          /* no default instance in named instance table; */
          /* we thus have to synthesize it                */
          num_instances++;
        }
      }

      FT_FREE( default_values );
      FT_FREE( instance_values );

      /* we don't support Multiple Master CFFs yet; */
      /* note that `glyf' or `CFF2' have precedence */
      if ( face->goto_table( face, TTAG_glyf, stream, 0 ) &&
           face->goto_table( face, TTAG_CFF2, stream, 0 ) &&
           !face->goto_table( face, TTAG_CFF, stream, 0 ) )
        num_instances = 0;

      /* instance indices in `face_instance_index' start with index 1, */
      /* thus `>' and not `>='                                         */
      if ( instance_index > num_instances )
      {
        if ( face_instance_index >= 0 )
          return FT_THROW( Invalid_Argument );
        else
          num_instances = 0;
      }

      face->root.style_flags = (FT_Long)num_instances << 16;
    }
#endif

    face->root.num_faces  = face->ttc_header.count;
    face->root.face_index = face_instance_index;

    return error;
  }


#define LOAD_( x )                                          \
  do                                                        \
  {                                                         \
    FT_TRACE2(( "`" #x "' " ));                             \
    FT_TRACE3(( "-->\n" ));                                 \
                                                            \
    error = sfnt->load_ ## x( face, stream );               \
                                                            \
    FT_TRACE2(( "%s\n", ( !error )                          \
                        ? "loaded"                          \
                        : FT_ERR_EQ( error, Table_Missing ) \
                          ? "missing"                       \
                          : "failed to load" ));            \
    FT_TRACE3(( "\n" ));                                    \
  } while ( 0 )

#define LOADM_( x, vertical )                               \
  do                                                        \
  {                                                         \
    FT_TRACE2(( "`%s" #x "' ",                              \
                vertical ? "vertical " : "" ));             \
    FT_TRACE3(( "-->\n" ));                                 \
                                                            \
    error = sfnt->load_ ## x( face, stream, vertical );     \
                                                            \
    FT_TRACE2(( "%s\n", ( !error )                          \
                        ? "loaded"                          \
                        : FT_ERR_EQ( error, Table_Missing ) \
                          ? "missing"                       \
                          : "failed to load" ));            \
    FT_TRACE3(( "\n" ));                                    \
  } while ( 0 )

#define GET_NAME( id, field )                                   \
  do                                                            \
  {                                                             \
    error = tt_face_get_name( face, TT_NAME_ID_ ## id, field ); \
    if ( error )                                                \
      goto Exit;                                                \
  } while ( 0 )


  FT_LOCAL_DEF( FT_Error )
  sfnt_load_face( FT_Stream      stream,
                  TT_Face        face,
                  FT_Int         face_instance_index,
                  FT_Int         num_params,
                  FT_Parameter*  params )
  {
    FT_Error      error;
#ifdef TT_CONFIG_OPTION_POSTSCRIPT_NAMES
    FT_Error      psnames_error;
#endif
    FT_Bool       has_outline;
    FT_Bool       is_apple_sbit;
    FT_Bool       is_apple_sbix;
    FT_Bool       has_CBLC;
    FT_Bool       has_CBDT;
    FT_Bool       ignore_typographic_family    = FALSE;
    FT_Bool       ignore_typographic_subfamily = FALSE;

    SFNT_Service  sfnt = (SFNT_Service)face->sfnt;

    FT_UNUSED( face_instance_index );


    /* Check parameters */

    {
      FT_Int  i;


      for ( i = 0; i < num_params; i++ )
      {
        if ( params[i].tag == FT_PARAM_TAG_IGNORE_TYPOGRAPHIC_FAMILY )
          ignore_typographic_family = TRUE;
        else if ( params[i].tag == FT_PARAM_TAG_IGNORE_TYPOGRAPHIC_SUBFAMILY )
          ignore_typographic_subfamily = TRUE;
      }
    }

    /* Load tables */

    /* We now support two SFNT-based bitmapped font formats.  They */
    /* are recognized easily as they do not include a `glyf'       */
    /* table.                                                      */
    /*                                                             */
    /* The first format comes from Apple, and uses a table named   */
    /* `bhed' instead of `head' to store the font header (using    */
    /* the same format).  It also doesn't include horizontal and   */
    /* vertical metrics tables (i.e. `hhea' and `vhea' tables are  */
    /* missing).                                                   */
    /*                                                             */
    /* The other format comes from Microsoft, and is used with     */
    /* WinCE/PocketPC.  It looks like a standard TTF, except that  */
    /* it doesn't contain outlines.                                */
    /*                                                             */

    FT_TRACE2(( "sfnt_load_face: %08p\n\n", face ));

    /* do we have outlines in there? */
#ifdef FT_CONFIG_OPTION_INCREMENTAL
    has_outline = FT_BOOL( face->root.internal->incremental_interface ||
                           tt_face_lookup_table( face, TTAG_glyf )    ||
                           tt_face_lookup_table( face, TTAG_CFF )     ||
                           tt_face_lookup_table( face, TTAG_CFF2 )    );
#else
    has_outline = FT_BOOL( tt_face_lookup_table( face, TTAG_glyf ) ||
                           tt_face_lookup_table( face, TTAG_CFF )  ||
                           tt_face_lookup_table( face, TTAG_CFF2 ) );
#endif

    is_apple_sbit = 0;
    is_apple_sbix = !face->goto_table( face, TTAG_sbix, stream, 0 );

    /* Apple 'sbix' color bitmaps are rendered scaled and then the 'glyf'
     * outline rendered on top.  We don't support that yet, so just ignore
     * the 'glyf' outline and advertise it as a bitmap-only font. */
    if ( is_apple_sbix )
      has_outline = FALSE;

    /* if this font doesn't contain outlines, we try to load */
    /* a `bhed' table                                        */
    if ( !has_outline && sfnt->load_bhed )
    {
      LOAD_( bhed );
      is_apple_sbit = FT_BOOL( !error );
    }

    /* load the font header (`head' table) if this isn't an Apple */
    /* sbit font file                                             */
    if ( !is_apple_sbit || is_apple_sbix )
    {
      LOAD_( head );
      if ( error )
        goto Exit;
    }

    has_CBLC = !face->goto_table( face, TTAG_CBLC, stream, 0 );
    has_CBDT = !face->goto_table( face, TTAG_CBDT, stream, 0 );

    /* Ignore outlines for CBLC/CBDT fonts. */
    if ( has_CBLC || has_CBDT )
      has_outline = FALSE;

    /* OpenType 1.8.2 introduced limits to this value;    */
    /* however, they make sense for older SFNT fonts also */
    if ( face->header.Units_Per_EM <    16 ||
         face->header.Units_Per_EM > 16384 )
    {
      error = FT_THROW( Invalid_Table );

      goto Exit;
    }

    /* the following tables are often not present in embedded TrueType */
    /* fonts within PDF documents, so don't check for them.            */
    LOAD_( maxp );
    LOAD_( cmap );

    /* the following tables are optional in PCL fonts -- */
    /* don't check for errors                            */
    LOAD_( name );
    LOAD_( post );

#ifdef TT_CONFIG_OPTION_POSTSCRIPT_NAMES
    psnames_error = error;
#endif

    /* do not load the metrics headers and tables if this is an Apple */
    /* sbit font file                                                 */
    if ( !is_apple_sbit )
    {
      /* load the `hhea' and `hmtx' tables */
      LOADM_( hhea, 0 );
      if ( !error )
      {
        LOADM_( hmtx, 0 );
        if ( FT_ERR_EQ( error, Table_Missing ) )
        {
          error = FT_THROW( Hmtx_Table_Missing );

#ifdef FT_CONFIG_OPTION_INCREMENTAL
          /* If this is an incrementally loaded font and there are */
          /* overriding metrics, tolerate a missing `hmtx' table.  */
          if ( face->root.internal->incremental_interface          &&
               face->root.internal->incremental_interface->funcs->
                 get_glyph_metrics                                 )
          {
            face->horizontal.number_Of_HMetrics = 0;
            error                               = FT_Err_Ok;
          }
#endif
        }
      }
      else if ( FT_ERR_EQ( error, Table_Missing ) )
      {
        /* No `hhea' table necessary for SFNT Mac fonts. */
        if ( face->format_tag == TTAG_true )
        {
          FT_TRACE2(( "This is an SFNT Mac font.\n" ));

          has_outline = 0;
          error       = FT_Err_Ok;
        }
        else
        {
          error = FT_THROW( Horiz_Header_Missing );

#ifdef FT_CONFIG_OPTION_INCREMENTAL
          /* If this is an incrementally loaded font and there are */
          /* overriding metrics, tolerate a missing `hhea' table.  */
          if ( face->root.internal->incremental_interface          &&
               face->root.internal->incremental_interface->funcs->
                 get_glyph_metrics                                 )
          {
            face->horizontal.number_Of_HMetrics = 0;
            error                               = FT_Err_Ok;
          }
#endif

        }
      }

      if ( error )
        goto Exit;

      /* try to load the `vhea' and `vmtx' tables */
      LOADM_( hhea, 1 );
      if ( !error )
      {
        LOADM_( hmtx, 1 );
        if ( !error )
          face->vertical_info = 1;
      }

      if ( error && FT_ERR_NEQ( error, Table_Missing ) )
        goto Exit;

      LOAD_( os2 );
      if ( error )
      {
        /* we treat the table as missing if there are any errors */
        face->os2.version = 0xFFFFU;
      }
    }

    /* the optional tables */

    /* embedded bitmap support */
    if ( sfnt->load_eblc )
      LOAD_( eblc );

    /* colored glyph support */
    if ( sfnt->load_cpal )
    {
      LOAD_( cpal );
      LOAD_( colr );
    }

    /* consider the pclt, kerning, and gasp tables as optional */
    LOAD_( pclt );
    LOAD_( gasp );
    LOAD_( kern );

    face->root.num_glyphs = face->max_profile.numGlyphs;

    /* Bit 8 of the `fsSelection' field in the `OS/2' table denotes  */
    /* a WWS-only font face.  `WWS' stands for `weight', width', and */
    /* `slope', a term used by Microsoft's Windows Presentation      */
    /* Foundation (WPF).  This flag has been introduced in version   */
    /* 1.5 of the OpenType specification (May 2008).                 */

    face->root.family_name = NULL;
    face->root.style_name  = NULL;
    if ( face->os2.version != 0xFFFFU && face->os2.fsSelection & 256 )
    {
      if ( !ignore_typographic_family )
        GET_NAME( TYPOGRAPHIC_FAMILY, &face->root.family_name );
      if ( !face->root.family_name )
        GET_NAME( FONT_FAMILY, &face->root.family_name );

      if ( !ignore_typographic_subfamily )
        GET_NAME( TYPOGRAPHIC_SUBFAMILY, &face->root.style_name );
      if ( !face->root.style_name )
        GET_NAME( FONT_SUBFAMILY, &face->root.style_name );
    }
    else
    {
      GET_NAME( WWS_FAMILY, &face->root.family_name );
      if ( !face->root.family_name && !ignore_typographic_family )
        GET_NAME( TYPOGRAPHIC_FAMILY, &face->root.family_name );
      if ( !face->root.family_name )
        GET_NAME( FONT_FAMILY, &face->root.family_name );

      GET_NAME( WWS_SUBFAMILY, &face->root.style_name );
      if ( !face->root.style_name && !ignore_typographic_subfamily )
        GET_NAME( TYPOGRAPHIC_SUBFAMILY, &face->root.style_name );
      if ( !face->root.style_name )
        GET_NAME( FONT_SUBFAMILY, &face->root.style_name );
    }

    /* now set up root fields */
    {
      FT_Face  root  = &face->root;
      FT_Long  flags = root->face_flags;


      /**********************************************************************
       *
       * Compute face flags.
       */
      if ( face->sbit_table_type == TT_SBIT_TABLE_TYPE_CBLC ||
           face->sbit_table_type == TT_SBIT_TABLE_TYPE_SBIX ||
           face->colr                                       )
        flags |= FT_FACE_FLAG_COLOR;      /* color glyphs */

      if ( has_outline == TRUE )
        flags |= FT_FACE_FLAG_SCALABLE;   /* scalable outlines */

      /* The sfnt driver only supports bitmap fonts natively, thus we */
      /* don't set FT_FACE_FLAG_HINTER.                               */
      flags |= FT_FACE_FLAG_SFNT       |  /* SFNT file format  */
               FT_FACE_FLAG_HORIZONTAL;   /* horizontal data   */

#ifdef TT_CONFIG_OPTION_POSTSCRIPT_NAMES
      if ( !psnames_error                             &&
           face->postscript.FormatType != 0x00030000L )
        flags |= FT_FACE_FLAG_GLYPH_NAMES;
#endif

      /* fixed width font? */
      if ( face->postscript.isFixedPitch )
        flags |= FT_FACE_FLAG_FIXED_WIDTH;

      /* vertical information? */
      if ( face->vertical_info )
        flags |= FT_FACE_FLAG_VERTICAL;

      /* kerning available ? */
      if ( TT_FACE_HAS_KERNING( face ) )
        flags |= FT_FACE_FLAG_KERNING;

#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
      /* Don't bother to load the tables unless somebody asks for them. */
      /* No need to do work which will (probably) not be used.          */
      if ( face->variation_support & TT_FACE_FLAG_VAR_FVAR )
      {
        if ( tt_face_lookup_table( face, TTAG_glyf ) != 0 &&
             tt_face_lookup_table( face, TTAG_gvar ) != 0 )
          flags |= FT_FACE_FLAG_MULTIPLE_MASTERS;
        if ( tt_face_lookup_table( face, TTAG_CFF2 ) != 0 )
          flags |= FT_FACE_FLAG_MULTIPLE_MASTERS;
      }
#endif

      root->face_flags = flags;

      /**********************************************************************
       *
       * Compute style flags.
       */

      flags = 0;
      if ( has_outline == TRUE && face->os2.version != 0xFFFFU )
      {
        /* We have an OS/2 table; use the `fsSelection' field.  Bit 9 */
        /* indicates an oblique font face.  This flag has been        */
        /* introduced in version 1.5 of the OpenType specification.   */

        if ( face->os2.fsSelection & 512 )       /* bit 9 */
          flags |= FT_STYLE_FLAG_ITALIC;
        else if ( face->os2.fsSelection & 1 )    /* bit 0 */
          flags |= FT_STYLE_FLAG_ITALIC;

        if ( face->os2.fsSelection & 32 )        /* bit 5 */
          flags |= FT_STYLE_FLAG_BOLD;
      }
      else
      {
        /* this is an old Mac font, use the header field */

        if ( face->header.Mac_Style & 1 )
          flags |= FT_STYLE_FLAG_BOLD;

        if ( face->header.Mac_Style & 2 )
          flags |= FT_STYLE_FLAG_ITALIC;
      }

      root->style_flags |= flags;

      /**********************************************************************
       *
       * Polish the charmaps.
       *
       *   Try to set the charmap encoding according to the platform &
       *   encoding ID of each charmap.  Emulate Unicode charmap if one
       *   is missing.
       */

      tt_face_build_cmaps( face );  /* ignore errors */


      /* set the encoding fields */
      {
        FT_Int   m;
#ifdef FT_CONFIG_OPTION_POSTSCRIPT_NAMES
        FT_Bool  has_unicode = FALSE;
#endif


        for ( m = 0; m < root->num_charmaps; m++ )
        {
          FT_CharMap  charmap = root->charmaps[m];


          charmap->encoding = sfnt_find_encoding( charmap->platform_id,
                                                  charmap->encoding_id );

#ifdef FT_CONFIG_OPTION_POSTSCRIPT_NAMES

          if ( charmap->encoding == FT_ENCODING_UNICODE   ||
               charmap->encoding == FT_ENCODING_MS_SYMBOL )  /* PUA */
            has_unicode = TRUE;
        }

        /* synthesize Unicode charmap if one is missing */
        if ( !has_unicode )
        {
          FT_CharMapRec cmaprec;


          cmaprec.face        = root;
          cmaprec.platform_id = TT_PLATFORM_MICROSOFT;
          cmaprec.encoding_id = TT_MS_ID_UNICODE_CS;
          cmaprec.encoding    = FT_ENCODING_UNICODE;


          error = FT_CMap_New( (FT_CMap_Class)&tt_cmap_unicode_class_rec,
                               NULL, &cmaprec, NULL );
          if ( error                                      &&
               FT_ERR_NEQ( error, No_Unicode_Glyph_Name ) &&
               FT_ERR_NEQ( error, Unimplemented_Feature ) )
            goto Exit;
          error = FT_Err_Ok;

#endif /* FT_CONFIG_OPTION_POSTSCRIPT_NAMES */

        }
      }

#ifdef TT_CONFIG_OPTION_EMBEDDED_BITMAPS

      /*
       * Now allocate the root array of FT_Bitmap_Size records and
       * populate them.  Unfortunately, it isn't possible to indicate bit
       * depths in the FT_Bitmap_Size record.  This is a design error.
       */
      {
        FT_UInt  count;


        count = face->sbit_num_strikes;

        if ( count > 0 )
        {
          FT_Memory        memory   = face->root.stream->memory;
          FT_UShort        em_size  = face->header.Units_Per_EM;
          FT_Short         avgwidth = face->os2.xAvgCharWidth;
          FT_Size_Metrics  metrics;

          FT_UInt*  sbit_strike_map = NULL;
          FT_UInt   strike_idx, bsize_idx;


          if ( em_size == 0 || face->os2.version == 0xFFFFU )
          {
            avgwidth = 1;
            em_size = 1;
          }

          /* to avoid invalid strike data in the `available_sizes' field */
          /* of `FT_Face', we map `available_sizes' indices to strike    */
          /* indices                                                     */
          if ( FT_NEW_ARRAY( root->available_sizes, count ) ||
               FT_NEW_ARRAY( sbit_strike_map, count ) )
            goto Exit;

          bsize_idx = 0;
          for ( strike_idx = 0; strike_idx < count; strike_idx++ )
          {
            FT_Bitmap_Size*  bsize = root->available_sizes + bsize_idx;


            error = sfnt->load_strike_metrics( face, strike_idx, &metrics );
            if ( error )
              continue;

            bsize->height = (FT_Short)( metrics.height >> 6 );
            bsize->width  = (FT_Short)(
              ( avgwidth * metrics.x_ppem + em_size / 2 ) / em_size );

            bsize->x_ppem = metrics.x_ppem << 6;
            bsize->y_ppem = metrics.y_ppem << 6;

            /* assume 72dpi */
            bsize->size   = metrics.y_ppem << 6;

            /* only use strikes with valid PPEM values */
            if ( bsize->x_ppem && bsize->y_ppem )
              sbit_strike_map[bsize_idx++] = strike_idx;
          }

          /* reduce array size to the actually used elements */
          (void)FT_RENEW_ARRAY( sbit_strike_map, count, bsize_idx );

          /* from now on, all strike indices are mapped */
          /* using `sbit_strike_map'                    */
          if ( bsize_idx )
          {
            face->sbit_strike_map = sbit_strike_map;

            root->face_flags     |= FT_FACE_FLAG_FIXED_SIZES;
            root->num_fixed_sizes = (FT_Int)bsize_idx;
          }
        }
      }

#endif /* TT_CONFIG_OPTION_EMBEDDED_BITMAPS */

      /* a font with no bitmaps and no outlines is scalable; */
      /* it has only empty glyphs then                       */
      if ( !FT_HAS_FIXED_SIZES( root ) && !FT_IS_SCALABLE( root ) )
        root->face_flags |= FT_FACE_FLAG_SCALABLE;


      /**********************************************************************
       *
       * Set up metrics.
       */
      if ( FT_IS_SCALABLE( root ) )
      {
        /* XXX What about if outline header is missing */
        /*     (e.g. sfnt wrapped bitmap)?             */
        root->bbox.xMin    = face->header.xMin;
        root->bbox.yMin    = face->header.yMin;
        root->bbox.xMax    = face->header.xMax;
        root->bbox.yMax    = face->header.yMax;
        root->units_per_EM = face->header.Units_Per_EM;


        /*
         * Computing the ascender/descender/height is tricky.
         *
         * The OpenType specification v1.8.3 says:
         *
         *   [OS/2's] sTypoAscender, sTypoDescender and sTypoLineGap fields
         *   are intended to allow applications to lay out documents in a
         *   typographically-correct and portable fashion.
         *
         * This is somewhat at odds with the decades of backwards
         * compatibility, operating systems and applications doing whatever
         * they want, not to mention broken fonts.
         *
         * Not all fonts have an OS/2 table; in this case, we take the values
         * in the horizontal header, although there is nothing stopping the
         * values from being unreliable. Even with a OS/2 table, certain fonts
         * set the sTypoAscender, sTypoDescender and sTypoLineGap fields to 0
         * and instead correctly set usWinAscent and usWinDescent.
         *
         * As an example, Arial Narrow is shipped as four files ARIALN.TTF,
         * ARIALNI.TTF, ARIALNB.TTF and ARIALNBI.TTF. Strangely, all fonts have
         * the same values in their sTypo* fields, except ARIALNB.ttf which
         * sets them to 0. All of them have different usWinAscent/Descent
         * values. The OS/2 table therefore cannot be trusted for computing the
         * text height reliably.
         *
         * As a compromise, do the following:
         *
         * 1. If the OS/2 table exists and the fsSelection bit 7 is set
         *    (USE_TYPO_METRICS), trust the font and use the sTypo* metrics.
         * 2. Otherwise, use the `hhea' table's metrics.
         * 3. If they are zero and the OS/2 table exists,
         *    1. use the OS/2 table's sTypo* metrics if they are non-zero.
         *    2. Otherwise, use the OS/2 table's usWin* metrics.
         */

        if ( face->os2.version != 0xFFFFU && face->os2.fsSelection & 128 )
        {
          root->ascender  = face->os2.sTypoAscender;
          root->descender = face->os2.sTypoDescender;
          root->height    = root->ascender - root->descender +
                            face->os2.sTypoLineGap;
        }
        else
        {
          root->ascender  = face->horizontal.Ascender;
          root->descender = face->horizontal.Descender;
          root->height    = root->ascender - root->descender +
                            face->horizontal.Line_Gap;

          if ( !( root->ascender || root->descender ) )
          {
            if ( face->os2.version != 0xFFFFU )
            {
              if ( face->os2.sTypoAscender || face->os2.sTypoDescender )
              {
                root->ascender  = face->os2.sTypoAscender;
                root->descender = face->os2.sTypoDescender;
                root->height    = root->ascender - root->descender +
                                  face->os2.sTypoLineGap;
              }
              else
              {
                root->ascender  =  (FT_Short)face->os2.usWinAscent;
                root->descender = -(FT_Short)face->os2.usWinDescent;
                root->height    =  root->ascender - root->descender;
              }
            }
          }
        }

        root->max_advance_width  =
          (FT_Short)face->horizontal.advance_Width_Max;
        root->max_advance_height =
          (FT_Short)( face->vertical_info ? face->vertical.advance_Height_Max
                                          : root->height );

        /* See https://www.microsoft.com/typography/otspec/post.htm -- */
        /* Adjust underline position from top edge to centre of        */
        /* stroke to convert TrueType meaning to FreeType meaning.     */
        root->underline_position  = face->postscript.underlinePosition -
                                    face->postscript.underlineThickness / 2;
        root->underline_thickness = face->postscript.underlineThickness;
      }

    }

  Exit:
    FT_TRACE2(( "sfnt_load_face: done\n" ));

    return error;
  }


#undef LOAD_
#undef LOADM_
#undef GET_NAME


  FT_LOCAL_DEF( void )
  sfnt_done_face( TT_Face  face )
  {
    FT_Memory     memory;
    SFNT_Service  sfnt;


    if ( !face )
      return;

    memory = face->root.memory;
    sfnt   = (SFNT_Service)face->sfnt;

    if ( sfnt )
    {
      /* destroy the postscript names table if it is loaded */
      if ( sfnt->free_psnames )
        sfnt->free_psnames( face );

      /* destroy the embedded bitmaps table if it is loaded */
      if ( sfnt->free_eblc )
        sfnt->free_eblc( face );

      /* destroy color table data if it is loaded */
      if ( sfnt->free_cpal )
      {
        sfnt->free_cpal( face );
        sfnt->free_colr( face );
      }
    }

#ifdef TT_CONFIG_OPTION_BDF
    /* freeing the embedded BDF properties */
    tt_face_free_bdf_props( face );
#endif

    /* freeing the kerning table */
    tt_face_done_kern( face );

    /* freeing the collection table */
    FT_FREE( face->ttc_header.offsets );
    face->ttc_header.count = 0;

    /* freeing table directory */
    FT_FREE( face->dir_tables );
    face->num_tables = 0;

    {
      FT_Stream  stream = FT_FACE_STREAM( face );


      /* simply release the 'cmap' table frame */
      FT_FRAME_RELEASE( face->cmap_table );
      face->cmap_size = 0;
    }

    face->horz_metrics_size = 0;
    face->vert_metrics_size = 0;

    /* freeing vertical metrics, if any */
    if ( face->vertical_info )
    {
      FT_FREE( face->vertical.long_metrics  );
      FT_FREE( face->vertical.short_metrics );
      face->vertical_info = 0;
    }

    /* freeing the gasp table */
    FT_FREE( face->gasp.gaspRanges );
    face->gasp.numRanges = 0;

    /* freeing the name table */
    if ( sfnt )
      sfnt->free_name( face );

    /* freeing family and style name */
    FT_FREE( face->root.family_name );
    FT_FREE( face->root.style_name );

    /* freeing sbit size table */
    FT_FREE( face->root.available_sizes );
    FT_FREE( face->sbit_strike_map );
    face->root.num_fixed_sizes = 0;

    FT_FREE( face->postscript_name );

#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
    FT_FREE( face->var_postscript_prefix );
#endif

    /* freeing glyph color palette data */
    FT_FREE( face->palette_data.palette_name_ids );
    FT_FREE( face->palette_data.palette_flags );
    FT_FREE( face->palette_data.palette_entry_name_ids );
    FT_FREE( face->palette );

    face->sfnt = NULL;
  }


/* END */
