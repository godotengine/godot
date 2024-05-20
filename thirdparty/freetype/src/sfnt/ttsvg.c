/****************************************************************************
 *
 * ttsvg.c
 *
 *   OpenType SVG Color (specification).
 *
 * Copyright (C) 2022-2023 by
 * David Turner, Robert Wilhelm, Werner Lemberg, and Moazin Khatti.
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
   * 'SVG' table specification:
   *
   *    https://docs.microsoft.com/en-us/typography/opentype/spec/svg
   *
   */

#include <ft2build.h>
#include <freetype/internal/ftstream.h>
#include <freetype/internal/ftobjs.h>
#include <freetype/internal/ftdebug.h>
#include <freetype/tttags.h>
#include <freetype/ftgzip.h>
#include <freetype/otsvg.h>


#ifdef FT_CONFIG_OPTION_SVG

#include "ttsvg.h"


  /* NOTE: These table sizes are given by the specification. */
#define SVG_TABLE_HEADER_SIZE           (10U)
#define SVG_DOCUMENT_RECORD_SIZE        (12U)
#define SVG_DOCUMENT_LIST_MINIMUM_SIZE  (2U + SVG_DOCUMENT_RECORD_SIZE)
#define SVG_MINIMUM_SIZE                (SVG_TABLE_HEADER_SIZE +        \
                                         SVG_DOCUMENT_LIST_MINIMUM_SIZE)


  typedef struct  Svg_
  {
    FT_UShort  version;                 /* table version (starting at 0)  */
    FT_UShort  num_entries;             /* number of SVG document records */

    FT_Byte*  svg_doc_list;  /* pointer to the start of SVG Document List */

    void*     table;                          /* memory that backs up SVG */
    FT_ULong  table_size;

  } Svg;


  /**************************************************************************
   *
   * The macro FT_COMPONENT is used in trace mode.  It is an implicit
   * parameter of the FT_TRACE() and FT_ERROR() macros, usued to print/log
   * messages during execution.
   */
#undef  FT_COMPONENT
#define FT_COMPONENT  ttsvg


  FT_LOCAL_DEF( FT_Error )
  tt_face_load_svg( TT_Face    face,
                    FT_Stream  stream )
  {
    FT_Error   error;
    FT_Memory  memory = face->root.memory;

    FT_ULong  table_size;
    FT_Byte*  table = NULL;
    FT_Byte*  p     = NULL;
    Svg*      svg   = NULL;
    FT_ULong  offsetToSVGDocumentList;


    error = face->goto_table( face, TTAG_SVG, stream, &table_size );
    if ( error )
      goto NoSVG;

    if ( table_size < SVG_MINIMUM_SIZE )
      goto InvalidTable;

    if ( FT_FRAME_EXTRACT( table_size, table ) )
      goto NoSVG;

    /* Allocate memory for the SVG object */
    if ( FT_NEW( svg ) )
      goto NoSVG;

    p                       = table;
    svg->version            = FT_NEXT_USHORT( p );
    offsetToSVGDocumentList = FT_NEXT_ULONG( p );

    if ( offsetToSVGDocumentList < SVG_TABLE_HEADER_SIZE            ||
         offsetToSVGDocumentList > table_size -
                                     SVG_DOCUMENT_LIST_MINIMUM_SIZE )
      goto InvalidTable;

    svg->svg_doc_list = (FT_Byte*)( table + offsetToSVGDocumentList );

    p                = svg->svg_doc_list;
    svg->num_entries = FT_NEXT_USHORT( p );

    FT_TRACE3(( "version: %d\n", svg->version ));
    FT_TRACE3(( "number of entries: %d\n", svg->num_entries ));

    if ( offsetToSVGDocumentList + 2U +
           svg->num_entries * SVG_DOCUMENT_RECORD_SIZE > table_size )
      goto InvalidTable;

    svg->table      = table;
    svg->table_size = table_size;

    face->svg              = svg;
    face->root.face_flags |= FT_FACE_FLAG_SVG;

    return FT_Err_Ok;

  InvalidTable:
    error = FT_THROW( Invalid_Table );

  NoSVG:
    FT_FRAME_RELEASE( table );
    FT_FREE( svg );
    face->svg = NULL;

    return error;
  }


  FT_LOCAL_DEF( void )
  tt_face_free_svg( TT_Face  face )
  {
    FT_Memory  memory = face->root.memory;
    FT_Stream  stream = face->root.stream;

    Svg*  svg = (Svg*)face->svg;


    if ( svg )
    {
      FT_FRAME_RELEASE( svg->table );
      FT_FREE( svg );
    }
  }


  typedef struct  Svg_doc_
  {
    FT_UShort  start_glyph_id;
    FT_UShort  end_glyph_id;

    FT_ULong  offset;
    FT_ULong  length;

  } Svg_doc;


  static Svg_doc
  extract_svg_doc( FT_Byte*  stream )
  {
    Svg_doc  doc;


    doc.start_glyph_id = FT_NEXT_USHORT( stream );
    doc.end_glyph_id   = FT_NEXT_USHORT( stream );

    doc.offset = FT_NEXT_ULONG( stream );
    doc.length = FT_NEXT_ULONG( stream );

    return doc;
  }


  static FT_Int
  compare_svg_doc( Svg_doc  doc,
                   FT_UInt  glyph_index )
  {
    if ( glyph_index < doc.start_glyph_id )
      return -1;
    else if ( glyph_index > doc.end_glyph_id )
      return 1;
    else
      return 0;
  }


  static FT_Error
  find_doc( FT_Byte*    document_records,
            FT_UShort   num_entries,
            FT_UInt     glyph_index,
            FT_ULong   *doc_offset,
            FT_ULong   *doc_length,
            FT_UShort  *start_glyph,
            FT_UShort  *end_glyph )
  {
    FT_Error  error;

    Svg_doc  start_doc;
    Svg_doc  mid_doc = { 0, 0, 0, 0 }; /* pacify compiler */
    Svg_doc  end_doc;

    FT_Bool  found = FALSE;
    FT_UInt  i     = 0;

    FT_UInt  start_index = 0;
    FT_UInt  end_index   = num_entries - 1;
    FT_Int   comp_res;


    /* search algorithm */
    if ( num_entries == 0 )
    {
      error = FT_THROW( Invalid_Table );
      return error;
    }

    start_doc = extract_svg_doc( document_records + start_index * 12 );
    end_doc   = extract_svg_doc( document_records + end_index * 12 );

    if ( ( compare_svg_doc( start_doc, glyph_index ) == -1 ) ||
         ( compare_svg_doc( end_doc, glyph_index ) == 1 )    )
    {
      error = FT_THROW( Invalid_Glyph_Index );
      return error;
    }

    while ( start_index <= end_index )
    {
      i        = ( start_index + end_index ) / 2;
      mid_doc  = extract_svg_doc( document_records + i * 12 );
      comp_res = compare_svg_doc( mid_doc, glyph_index );

      if ( comp_res == 1 )
      {
        start_index = i + 1;
        start_doc   = extract_svg_doc( document_records + start_index * 4 );
      }
      else if ( comp_res == -1 )
      {
        end_index = i - 1;
        end_doc   = extract_svg_doc( document_records + end_index * 4 );
      }
      else
      {
        found = TRUE;
        break;
      }
    }
    /* search algorithm end */

    if ( found != TRUE )
    {
      FT_TRACE5(( "SVG glyph not found\n" ));
      error = FT_THROW( Invalid_Glyph_Index );
    }
    else
    {
      *doc_offset = mid_doc.offset;
      *doc_length = mid_doc.length;

      *start_glyph = mid_doc.start_glyph_id;
      *end_glyph   = mid_doc.end_glyph_id;

      error = FT_Err_Ok;
    }

    return error;
  }


  FT_LOCAL_DEF( FT_Error )
  tt_face_load_svg_doc( FT_GlyphSlot  glyph,
                        FT_UInt       glyph_index )
  {
    FT_Error   error  = FT_Err_Ok;
    TT_Face    face   = (TT_Face)glyph->face;
    FT_Memory  memory = face->root.memory;
    Svg*       svg    = (Svg*)face->svg;

    FT_Byte*  doc_list;
    FT_ULong  doc_limit;

    FT_Byte*   doc;
    FT_ULong   doc_offset;
    FT_ULong   doc_length;
    FT_UShort  doc_start_glyph_id;
    FT_UShort  doc_end_glyph_id;

    FT_SVG_Document  svg_document = (FT_SVG_Document)glyph->other;


    FT_ASSERT( !( svg == NULL ) );

    doc_list = svg->svg_doc_list;

    error = find_doc( doc_list + 2, svg->num_entries, glyph_index,
                                    &doc_offset, &doc_length,
                                    &doc_start_glyph_id, &doc_end_glyph_id );
    if ( error != FT_Err_Ok )
      goto Exit;

    doc_limit = svg->table_size -
                  (FT_ULong)( doc_list - (FT_Byte*)svg->table );
    if ( doc_offset > doc_limit              ||
         doc_length > doc_limit - doc_offset )
    {
      error = FT_THROW( Invalid_Table );
      goto Exit;
    }

    doc = doc_list + doc_offset;

    if ( doc_length > 6 &&
         doc[0] == 0x1F &&
         doc[1] == 0x8B &&
         doc[2] == 0x08 )
    {
#ifdef FT_CONFIG_OPTION_USE_ZLIB

      FT_ULong  uncomp_size;
      FT_Byte*  uncomp_buffer = NULL;


      /*
       * Get the size of the original document.  This helps in allotting the
       * buffer to accommodate the uncompressed version.  The last 4 bytes
       * of the compressed document are equal to the original size modulo
       * 2^32.  Since the size of SVG documents is less than 2^32 bytes we
       * can use this accurately.  The four bytes are stored in
       * little-endian format.
       */
      FT_TRACE4(( "SVG document is GZIP compressed\n" ));
      uncomp_size = (FT_ULong)doc[doc_length - 1] << 24 |
                    (FT_ULong)doc[doc_length - 2] << 16 |
                    (FT_ULong)doc[doc_length - 3] << 8  |
                    (FT_ULong)doc[doc_length - 4];

      if ( FT_QALLOC( uncomp_buffer, uncomp_size ) )
        goto Exit;

      error = FT_Gzip_Uncompress( memory,
                                  uncomp_buffer,
                                  &uncomp_size,
                                  doc,
                                  doc_length );
      if ( error )
      {
        FT_FREE( uncomp_buffer );
        error = FT_THROW( Invalid_Table );
        goto Exit;
      }

      glyph->internal->flags |= FT_GLYPH_OWN_GZIP_SVG;

      doc        = uncomp_buffer;
      doc_length = uncomp_size;

#else /* !FT_CONFIG_OPTION_USE_ZLIB */

      error = FT_THROW( Unimplemented_Feature );
      goto Exit;

#endif /* !FT_CONFIG_OPTION_USE_ZLIB */
    }

    svg_document->svg_document        = doc;
    svg_document->svg_document_length = doc_length;

    svg_document->metrics      = glyph->face->size->metrics;
    svg_document->units_per_EM = glyph->face->units_per_EM;

    svg_document->start_glyph_id = doc_start_glyph_id;
    svg_document->end_glyph_id   = doc_end_glyph_id;

    svg_document->transform.xx = 0x10000;
    svg_document->transform.xy = 0;
    svg_document->transform.yx = 0;
    svg_document->transform.yy = 0x10000;

    svg_document->delta.x = 0;
    svg_document->delta.y = 0;

    FT_TRACE5(( "start_glyph_id: %d\n", doc_start_glyph_id ));
    FT_TRACE5(( "end_glyph_id:   %d\n", doc_end_glyph_id ));
    FT_TRACE5(( "svg_document:\n" ));
    FT_TRACE5(( " %.*s\n", (FT_UInt)doc_length, doc ));

    glyph->other = svg_document;

  Exit:
    return error;
  }

#else /* !FT_CONFIG_OPTION_SVG */

  /* ANSI C doesn't like empty source files */
  typedef int  tt_svg_dummy_;

#endif /* !FT_CONFIG_OPTION_SVG */


/* END */
