/*  pcfread.c

    FreeType font driver for pcf fonts

  Copyright 2000-2010, 2012-2014 by
  Francesco Zappa Nardelli

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/


#include <ft2build.h>

#include FT_INTERNAL_DEBUG_H
#include FT_INTERNAL_STREAM_H
#include FT_INTERNAL_OBJECTS_H

#include "pcf.h"
#include "pcfread.h"

#include "pcferror.h"


  /**************************************************************************
   *
   * The macro FT_COMPONENT is used in trace mode.  It is an implicit
   * parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log
   * messages during execution.
   */
#undef  FT_COMPONENT
#define FT_COMPONENT  pcfread


#ifdef FT_DEBUG_LEVEL_TRACE
  static const char* const  tableNames[] =
  {
    "properties",
    "accelerators",
    "metrics",
    "bitmaps",
    "ink metrics",
    "encodings",
    "swidths",
    "glyph names",
    "BDF accelerators"
  };
#endif


  static
  const FT_Frame_Field  pcf_toc_header[] =
  {
#undef  FT_STRUCTURE
#define FT_STRUCTURE  PCF_TocRec

    FT_FRAME_START( 8 ),
      FT_FRAME_ULONG_LE( version ),
      FT_FRAME_ULONG_LE( count ),
    FT_FRAME_END
  };


  static
  const FT_Frame_Field  pcf_table_header[] =
  {
#undef  FT_STRUCTURE
#define FT_STRUCTURE  PCF_TableRec

    FT_FRAME_START( 16  ),
      FT_FRAME_ULONG_LE( type ),
      FT_FRAME_ULONG_LE( format ),
      FT_FRAME_ULONG_LE( size ),   /* rounded up to a multiple of 4 */
      FT_FRAME_ULONG_LE( offset ),
    FT_FRAME_END
  };


  static FT_Error
  pcf_read_TOC( FT_Stream  stream,
                PCF_Face   face )
  {
    FT_Error   error;
    PCF_Toc    toc = &face->toc;
    PCF_Table  tables;

    FT_Memory  memory = FT_FACE( face )->memory;
    FT_UInt    n;

    FT_ULong   size;


    if ( FT_STREAM_SEEK( 0 )                          ||
         FT_STREAM_READ_FIELDS( pcf_toc_header, toc ) )
      return FT_THROW( Cannot_Open_Resource );

    if ( toc->version != PCF_FILE_VERSION ||
         toc->count   == 0                )
      return FT_THROW( Invalid_File_Format );

    if ( stream->size < 16 )
      return FT_THROW( Invalid_File_Format );

    /* we need 16 bytes per TOC entry, */
    /* and there can be most 9 tables  */
    if ( toc->count > ( stream->size >> 4 ) ||
         toc->count > 9                     )
    {
      FT_TRACE0(( "pcf_read_TOC: adjusting number of tables"
                  " (from %d to %d)\n",
                  toc->count,
                  FT_MIN( stream->size >> 4, 9 ) ));
      toc->count = FT_MIN( stream->size >> 4, 9 );
    }

    if ( FT_NEW_ARRAY( face->toc.tables, toc->count ) )
      return error;

    tables = face->toc.tables;
    for ( n = 0; n < toc->count; n++ )
    {
      if ( FT_STREAM_READ_FIELDS( pcf_table_header, tables ) )
        goto Exit;
      tables++;
    }

    /* Sort tables and check for overlaps.  Because they are almost      */
    /* always ordered already, an in-place bubble sort with simultaneous */
    /* boundary checking seems appropriate.                              */
    tables = face->toc.tables;

    for ( n = 0; n < toc->count - 1; n++ )
    {
      FT_UInt  i, have_change;


      have_change = 0;

      for ( i = 0; i < toc->count - 1 - n; i++ )
      {
        PCF_TableRec  tmp;


        if ( tables[i].offset > tables[i + 1].offset )
        {
          tmp           = tables[i];
          tables[i]     = tables[i + 1];
          tables[i + 1] = tmp;

          have_change = 1;
        }

        if ( ( tables[i].size   > tables[i + 1].offset )                  ||
             ( tables[i].offset > tables[i + 1].offset - tables[i].size ) )
        {
          error = FT_THROW( Invalid_Offset );
          goto Exit;
        }
      }

      if ( !have_change )
        break;
    }

    /*
     * We now check whether the `size' and `offset' values are reasonable:
     * `offset' + `size' must not exceed the stream size.
     *
     * Note, however, that X11's `pcfWriteFont' routine (used by the
     * `bdftopcf' program to create PCF font files) has two special
     * features.
     *
     * - It always assigns the accelerator table a size of 100 bytes in the
     *   TOC, regardless of its real size, which can vary between 34 and 72
     *   bytes.
     *
     * - Due to the way the routine is designed, it ships out the last font
     *   table with its real size, ignoring the TOC's size value.  Since
     *   the TOC size values are always rounded up to a multiple of 4, the
     *   difference can be up to three bytes for all tables except the
     *   accelerator table, for which the difference can be as large as 66
     *   bytes.
     *
     */

    tables = face->toc.tables;
    size   = stream->size;

    for ( n = 0; n < toc->count - 1; n++ )
    {
      /* we need two checks to avoid overflow */
      if ( ( tables->size   > size                ) ||
           ( tables->offset > size - tables->size ) )
      {
        error = FT_THROW( Invalid_Table );
        goto Exit;
      }
      tables++;
    }

    /* only check `tables->offset' for last table element ... */
    if ( ( tables->offset > size ) )
    {
      error = FT_THROW( Invalid_Table );
      goto Exit;
    }
    /* ... and adjust `tables->size' to the real value if necessary */
    if ( tables->size > size - tables->offset )
      tables->size = size - tables->offset;

#ifdef FT_DEBUG_LEVEL_TRACE

    {
      FT_UInt      i, j;
      const char*  name = "?";


      FT_TRACE4(( "pcf_read_TOC:\n" ));

      FT_TRACE4(( "  number of tables: %ld\n", face->toc.count ));

      tables = face->toc.tables;
      for ( i = 0; i < toc->count; i++ )
      {
        for ( j = 0; j < sizeof ( tableNames ) / sizeof ( tableNames[0] );
              j++ )
          if ( tables[i].type == (FT_UInt)( 1 << j ) )
            name = tableNames[j];

        FT_TRACE4(( "  %d: type=%s, format=0x%X,"
                    " size=%ld (0x%lX), offset=%ld (0x%lX)\n",
                    i, name,
                    tables[i].format,
                    tables[i].size, tables[i].size,
                    tables[i].offset, tables[i].offset ));
      }
    }

#endif

    return FT_Err_Ok;

  Exit:
    FT_FREE( face->toc.tables );
    return error;
  }


#define PCF_METRIC_SIZE  12

  static
  const FT_Frame_Field  pcf_metric_header[] =
  {
#undef  FT_STRUCTURE
#define FT_STRUCTURE  PCF_MetricRec

    FT_FRAME_START( PCF_METRIC_SIZE ),
      FT_FRAME_SHORT_LE( leftSideBearing ),
      FT_FRAME_SHORT_LE( rightSideBearing ),
      FT_FRAME_SHORT_LE( characterWidth ),
      FT_FRAME_SHORT_LE( ascent ),
      FT_FRAME_SHORT_LE( descent ),
      FT_FRAME_SHORT_LE( attributes ),
    FT_FRAME_END
  };


  static
  const FT_Frame_Field  pcf_metric_msb_header[] =
  {
#undef  FT_STRUCTURE
#define FT_STRUCTURE  PCF_MetricRec

    FT_FRAME_START( PCF_METRIC_SIZE ),
      FT_FRAME_SHORT( leftSideBearing ),
      FT_FRAME_SHORT( rightSideBearing ),
      FT_FRAME_SHORT( characterWidth ),
      FT_FRAME_SHORT( ascent ),
      FT_FRAME_SHORT( descent ),
      FT_FRAME_SHORT( attributes ),
    FT_FRAME_END
  };


#define PCF_COMPRESSED_METRIC_SIZE  5

  static
  const FT_Frame_Field  pcf_compressed_metric_header[] =
  {
#undef  FT_STRUCTURE
#define FT_STRUCTURE  PCF_Compressed_MetricRec

    FT_FRAME_START( PCF_COMPRESSED_METRIC_SIZE ),
      FT_FRAME_BYTE( leftSideBearing ),
      FT_FRAME_BYTE( rightSideBearing ),
      FT_FRAME_BYTE( characterWidth ),
      FT_FRAME_BYTE( ascent ),
      FT_FRAME_BYTE( descent ),
    FT_FRAME_END
  };


  static FT_Error
  pcf_get_metric( FT_Stream   stream,
                  FT_ULong    format,
                  PCF_Metric  metric )
  {
    FT_Error  error = FT_Err_Ok;


    if ( PCF_FORMAT_MATCH( format, PCF_DEFAULT_FORMAT ) )
    {
      const FT_Frame_Field*  fields;


      /* parsing normal metrics */
      fields = ( PCF_BYTE_ORDER( format ) == MSBFirst )
               ? pcf_metric_msb_header
               : pcf_metric_header;

      /* the following sets `error' but doesn't return in case of failure */
      (void)FT_STREAM_READ_FIELDS( fields, metric );
    }
    else
    {
      PCF_Compressed_MetricRec  compr;


      /* parsing compressed metrics */
      if ( FT_STREAM_READ_FIELDS( pcf_compressed_metric_header, &compr ) )
        goto Exit;

      metric->leftSideBearing  = (FT_Short)( compr.leftSideBearing  - 0x80 );
      metric->rightSideBearing = (FT_Short)( compr.rightSideBearing - 0x80 );
      metric->characterWidth   = (FT_Short)( compr.characterWidth   - 0x80 );
      metric->ascent           = (FT_Short)( compr.ascent           - 0x80 );
      metric->descent          = (FT_Short)( compr.descent          - 0x80 );
      metric->attributes       = 0;
    }

    FT_TRACE5(( " width=%d,"
                " lsb=%d, rsb=%d,"
                " ascent=%d, descent=%d,"
                " attributes=%d\n",
                metric->characterWidth,
                metric->leftSideBearing,
                metric->rightSideBearing,
                metric->ascent,
                metric->descent,
                metric->attributes ));

  Exit:
    return error;
  }


  static FT_Error
  pcf_seek_to_table_type( FT_Stream  stream,
                          PCF_Table  tables,
                          FT_ULong   ntables, /* same as PCF_Toc->count */
                          FT_ULong   type,
                          FT_ULong  *aformat,
                          FT_ULong  *asize )
  {
    FT_Error  error = FT_ERR( Invalid_File_Format );
    FT_ULong  i;


    for ( i = 0; i < ntables; i++ )
      if ( tables[i].type == type )
      {
        if ( stream->pos > tables[i].offset )
        {
          error = FT_THROW( Invalid_Stream_Skip );
          goto Fail;
        }

        if ( FT_STREAM_SKIP( tables[i].offset - stream->pos ) )
        {
          error = FT_THROW( Invalid_Stream_Skip );
          goto Fail;
        }

        *asize   = tables[i].size;
        *aformat = tables[i].format;

        return FT_Err_Ok;
      }

  Fail:
    *asize = 0;
    return error;
  }


  static FT_Bool
  pcf_has_table_type( PCF_Table  tables,
                      FT_ULong   ntables, /* same as PCF_Toc->count */
                      FT_ULong   type )
  {
    FT_ULong  i;


    for ( i = 0; i < ntables; i++ )
      if ( tables[i].type == type )
        return TRUE;

    return FALSE;
  }


#define PCF_PROPERTY_SIZE  9

  static
  const FT_Frame_Field  pcf_property_header[] =
  {
#undef  FT_STRUCTURE
#define FT_STRUCTURE  PCF_ParsePropertyRec

    FT_FRAME_START( PCF_PROPERTY_SIZE ),
      FT_FRAME_LONG_LE( name ),
      FT_FRAME_BYTE   ( isString ),
      FT_FRAME_LONG_LE( value ),
    FT_FRAME_END
  };


  static
  const FT_Frame_Field  pcf_property_msb_header[] =
  {
#undef  FT_STRUCTURE
#define FT_STRUCTURE  PCF_ParsePropertyRec

    FT_FRAME_START( PCF_PROPERTY_SIZE ),
      FT_FRAME_LONG( name ),
      FT_FRAME_BYTE( isString ),
      FT_FRAME_LONG( value ),
    FT_FRAME_END
  };


  FT_LOCAL_DEF( PCF_Property )
  pcf_find_property( PCF_Face          face,
                     const FT_String*  prop )
  {
    PCF_Property  properties = face->properties;
    FT_Bool       found      = 0;
    int           i;


    for ( i = 0; i < face->nprops && !found; i++ )
    {
      if ( !ft_strcmp( properties[i].name, prop ) )
        found = 1;
    }

    if ( found )
      return properties + i - 1;
    else
      return NULL;
  }


  static FT_Error
  pcf_get_properties( FT_Stream  stream,
                      PCF_Face   face )
  {
    PCF_ParseProperty  props      = NULL;
    PCF_Property       properties = NULL;
    FT_ULong           nprops, orig_nprops, i;
    FT_ULong           format, size;
    FT_Error           error;
    FT_Memory          memory     = FT_FACE( face )->memory;
    FT_ULong           string_size;
    FT_String*         strings    = NULL;


    error = pcf_seek_to_table_type( stream,
                                    face->toc.tables,
                                    face->toc.count,
                                    PCF_PROPERTIES,
                                    &format,
                                    &size );
    if ( error )
      goto Bail;

    if ( FT_READ_ULONG_LE( format ) )
      goto Bail;

    FT_TRACE4(( "pcf_get_properties:\n"
                "  format: 0x%lX (%s)\n",
                format,
                PCF_BYTE_ORDER( format ) == MSBFirst ? "MSB" : "LSB" ));

    if ( !PCF_FORMAT_MATCH( format, PCF_DEFAULT_FORMAT ) )
      goto Bail;

    if ( PCF_BYTE_ORDER( format ) == MSBFirst )
      (void)FT_READ_ULONG( orig_nprops );
    else
      (void)FT_READ_ULONG_LE( orig_nprops );
    if ( error )
      goto Bail;

    FT_TRACE4(( "  number of properties: %ld\n", orig_nprops ));

    /* rough estimate */
    if ( orig_nprops > size / PCF_PROPERTY_SIZE )
    {
      error = FT_THROW( Invalid_Table );
      goto Bail;
    }

    /* as a heuristic limit to avoid excessive allocation in */
    /* gzip bombs (i.e., very small, invalid input data that */
    /* pretends to expand to an insanely large file) we only */
    /* load the first 256 properties                         */
    if ( orig_nprops > 256 )
    {
      FT_TRACE0(( "pcf_get_properties:"
                  " only loading first 256 properties\n" ));
      nprops = 256;
    }
    else
      nprops = orig_nprops;

    face->nprops = (int)nprops;

    if ( FT_NEW_ARRAY( props, nprops ) )
      goto Bail;

    for ( i = 0; i < nprops; i++ )
    {
      if ( PCF_BYTE_ORDER( format ) == MSBFirst )
      {
        if ( FT_STREAM_READ_FIELDS( pcf_property_msb_header, props + i ) )
          goto Bail;
      }
      else
      {
        if ( FT_STREAM_READ_FIELDS( pcf_property_header, props + i ) )
          goto Bail;
      }
    }

    /* this skip will only work if we really have an extremely large */
    /* number of properties; it will fail for fake data, avoiding an */
    /* unnecessarily large allocation later on                       */
    if ( FT_STREAM_SKIP( ( orig_nprops - nprops ) * PCF_PROPERTY_SIZE ) )
    {
      error = FT_THROW( Invalid_Stream_Skip );
      goto Bail;
    }

    /* pad the property array                                            */
    /*                                                                   */
    /* clever here - nprops is the same as the number of odd-units read, */
    /* as only isStringProp are odd length   (Keith Packard)             */
    /*                                                                   */
    if ( orig_nprops & 3 )
    {
      i = 4 - ( orig_nprops & 3 );
      if ( FT_STREAM_SKIP( i ) )
      {
        error = FT_THROW( Invalid_Stream_Skip );
        goto Bail;
      }
    }

    if ( PCF_BYTE_ORDER( format ) == MSBFirst )
      (void)FT_READ_ULONG( string_size );
    else
      (void)FT_READ_ULONG_LE( string_size );
    if ( error )
      goto Bail;

    FT_TRACE4(( "  string size: %ld\n", string_size ));

    /* rough estimate */
    if ( string_size > size - orig_nprops * PCF_PROPERTY_SIZE )
    {
      error = FT_THROW( Invalid_Table );
      goto Bail;
    }

    /* the strings in the `strings' array are PostScript strings, */
    /* which can have a maximum length of 65536 characters each   */
    if ( string_size > 16777472 )   /* 256 * (65536 + 1) */
    {
      FT_TRACE0(( "pcf_get_properties:"
                  " loading only 16777472 bytes of strings array\n" ));
      string_size = 16777472;
    }

    /* allocate one more byte so that we have a final null byte */
    if ( FT_NEW_ARRAY( strings, string_size + 1 ) )
      goto Bail;

    error = FT_Stream_Read( stream, (FT_Byte*)strings, string_size );
    if ( error )
      goto Bail;

    if ( FT_NEW_ARRAY( properties, nprops ) )
      goto Bail;

    face->properties = properties;

    FT_TRACE4(( "\n" ));
    for ( i = 0; i < nprops; i++ )
    {
      FT_Long  name_offset = props[i].name;


      if ( ( name_offset < 0 )                     ||
           ( (FT_ULong)name_offset > string_size ) )
      {
        error = FT_THROW( Invalid_Offset );
        goto Bail;
      }

      if ( FT_STRDUP( properties[i].name, strings + name_offset ) )
        goto Bail;

      FT_TRACE4(( "  %s:", properties[i].name ));

      properties[i].isString = props[i].isString;

      if ( props[i].isString )
      {
        FT_Long  value_offset = props[i].value;


        if ( ( value_offset < 0 )                     ||
             ( (FT_ULong)value_offset > string_size ) )
        {
          error = FT_THROW( Invalid_Offset );
          goto Bail;
        }

        if ( FT_STRDUP( properties[i].value.atom, strings + value_offset ) )
          goto Bail;

        FT_TRACE4(( " `%s'\n", properties[i].value.atom ));
      }
      else
      {
        properties[i].value.l = props[i].value;

        FT_TRACE4(( " %d\n", properties[i].value.l ));
      }
    }

    error = FT_Err_Ok;

  Bail:
    FT_FREE( props );
    FT_FREE( strings );

    return error;
  }


  static FT_Error
  pcf_get_metrics( FT_Stream  stream,
                   PCF_Face   face )
  {
    FT_Error    error;
    FT_Memory   memory  = FT_FACE( face )->memory;
    FT_ULong    format, size;
    PCF_Metric  metrics = NULL;
    FT_ULong    nmetrics, orig_nmetrics, i;


    error = pcf_seek_to_table_type( stream,
                                    face->toc.tables,
                                    face->toc.count,
                                    PCF_METRICS,
                                    &format,
                                    &size );
    if ( error )
      return error;

    if ( FT_READ_ULONG_LE( format ) )
      goto Bail;

    FT_TRACE4(( "pcf_get_metrics:\n"
                "  format: 0x%lX (%s, %s)\n",
                format,
                PCF_BYTE_ORDER( format ) == MSBFirst ? "MSB" : "LSB",
                PCF_FORMAT_MATCH( format, PCF_COMPRESSED_METRICS ) ?
                  "compressed" : "uncompressed" ));

    if ( !PCF_FORMAT_MATCH( format, PCF_DEFAULT_FORMAT )     &&
         !PCF_FORMAT_MATCH( format, PCF_COMPRESSED_METRICS ) )
      return FT_THROW( Invalid_File_Format );

    if ( PCF_FORMAT_MATCH( format, PCF_DEFAULT_FORMAT ) )
    {
      if ( PCF_BYTE_ORDER( format ) == MSBFirst )
        (void)FT_READ_ULONG( orig_nmetrics );
      else
        (void)FT_READ_ULONG_LE( orig_nmetrics );
    }
    else
    {
      if ( PCF_BYTE_ORDER( format ) == MSBFirst )
        (void)FT_READ_USHORT( orig_nmetrics );
      else
        (void)FT_READ_USHORT_LE( orig_nmetrics );
    }
    if ( error )
      return FT_THROW( Invalid_File_Format );

    FT_TRACE4(( "  number of metrics: %ld\n", orig_nmetrics ));

    /* rough estimate */
    if ( PCF_FORMAT_MATCH( format, PCF_DEFAULT_FORMAT ) )
    {
      if ( orig_nmetrics > size / PCF_METRIC_SIZE )
        return FT_THROW( Invalid_Table );
    }
    else
    {
      if ( orig_nmetrics > size / PCF_COMPRESSED_METRIC_SIZE )
        return FT_THROW( Invalid_Table );
    }

    if ( !orig_nmetrics )
      return FT_THROW( Invalid_Table );

    /* PCF is a format from ancient times; Unicode was in its       */
    /* infancy, and widely used two-byte character sets for CJK     */
    /* scripts (Big 5, GB 2312, JIS X 0208, etc.) did have at most  */
    /* 15000 characters.  Even the more exotic CNS 11643 and CCCII  */
    /* standards, which were essentially three-byte character sets, */
    /* provided less then 65536 assigned characters.                */
    /*                                                              */
    /* While technically possible to have a larger number of glyphs */
    /* in PCF files, we thus limit the number to 65536.             */
    if ( orig_nmetrics > 65536 )
    {
      FT_TRACE0(( "pcf_get_metrics:"
                  " only loading first 65536 metrics\n" ));
      nmetrics = 65536;
    }
    else
      nmetrics = orig_nmetrics;

    face->nmetrics = nmetrics;

    if ( FT_NEW_ARRAY( face->metrics, nmetrics ) )
      return error;

    metrics = face->metrics;

    FT_TRACE4(( "\n" ));
    for ( i = 0; i < nmetrics; i++, metrics++ )
    {
      FT_TRACE5(( "  idx %ld:", i ));
      error = pcf_get_metric( stream, format, metrics );

      metrics->bits = 0;

      if ( error )
        break;

      /* sanity checks -- those values are used in `PCF_Glyph_Load' to     */
      /* compute a glyph's bitmap dimensions, thus setting them to zero in */
      /* case of an error disables this particular glyph only              */
      if ( metrics->rightSideBearing < metrics->leftSideBearing ||
           metrics->ascent < -metrics->descent                  )
      {
        metrics->characterWidth   = 0;
        metrics->leftSideBearing  = 0;
        metrics->rightSideBearing = 0;
        metrics->ascent           = 0;
        metrics->descent          = 0;

        FT_TRACE0(( "pcf_get_metrics:"
                    " invalid metrics for glyph %d\n", i ));
      }
    }

    if ( error )
      FT_FREE( face->metrics );

  Bail:
    return error;
  }


  static FT_Error
  pcf_get_bitmaps( FT_Stream  stream,
                   PCF_Face   face )
  {
    FT_Error   error;
    FT_Memory  memory  = FT_FACE( face )->memory;
    FT_ULong*  offsets = NULL;
    FT_ULong   bitmapSizes[GLYPHPADOPTIONS];
    FT_ULong   format, size;
    FT_ULong   nbitmaps, orig_nbitmaps, i, sizebitmaps = 0;


    error = pcf_seek_to_table_type( stream,
                                    face->toc.tables,
                                    face->toc.count,
                                    PCF_BITMAPS,
                                    &format,
                                    &size );
    if ( error )
      return error;

    error = FT_Stream_EnterFrame( stream, 8 );
    if ( error )
      return error;

    format = FT_GET_ULONG_LE();
    if ( PCF_BYTE_ORDER( format ) == MSBFirst )
      orig_nbitmaps = FT_GET_ULONG();
    else
      orig_nbitmaps = FT_GET_ULONG_LE();

    FT_Stream_ExitFrame( stream );

    FT_TRACE4(( "pcf_get_bitmaps:\n"
                "  format: 0x%lX\n"
                "          (%s, %s,\n"
                "           padding=%d bit%s, scanning=%d bit%s)\n",
                format,
                PCF_BYTE_ORDER( format ) == MSBFirst
                  ? "most significant byte first"
                  : "least significant byte first",
                PCF_BIT_ORDER( format ) == MSBFirst
                  ? "most significant bit first"
                  : "least significant bit first",
                8 << PCF_GLYPH_PAD_INDEX( format ),
                ( 8 << PCF_GLYPH_PAD_INDEX( format ) ) == 1 ? "" : "s",
                8 << PCF_SCAN_UNIT_INDEX( format ),
                ( 8 << PCF_SCAN_UNIT_INDEX( format ) ) == 1 ? "" : "s" ));

    if ( !PCF_FORMAT_MATCH( format, PCF_DEFAULT_FORMAT ) )
      return FT_THROW( Invalid_File_Format );

    FT_TRACE4(( "  number of bitmaps: %ld\n", orig_nbitmaps ));

    /* see comment in `pcf_get_metrics' */
    if ( orig_nbitmaps > 65536 )
    {
      FT_TRACE0(( "pcf_get_bitmaps:"
                  " only loading first 65536 bitmaps\n" ));
      nbitmaps = 65536;
    }
    else
      nbitmaps = orig_nbitmaps;

    if ( nbitmaps != face->nmetrics )
      return FT_THROW( Invalid_File_Format );

    if ( FT_NEW_ARRAY( offsets, nbitmaps ) )
      return error;

    FT_TRACE5(( "\n" ));
    for ( i = 0; i < nbitmaps; i++ )
    {
      if ( PCF_BYTE_ORDER( format ) == MSBFirst )
        (void)FT_READ_ULONG( offsets[i] );
      else
        (void)FT_READ_ULONG_LE( offsets[i] );

      FT_TRACE5(( "  bitmap %lu: offset %lu (0x%lX)\n",
                  i, offsets[i], offsets[i] ));
    }
    if ( error )
      goto Bail;

    for ( i = 0; i < GLYPHPADOPTIONS; i++ )
    {
      if ( PCF_BYTE_ORDER( format ) == MSBFirst )
        (void)FT_READ_ULONG( bitmapSizes[i] );
      else
        (void)FT_READ_ULONG_LE( bitmapSizes[i] );
      if ( error )
        goto Bail;

      sizebitmaps = bitmapSizes[PCF_GLYPH_PAD_INDEX( format )];

      FT_TRACE4(( "  %ld-bit padding implies a size of %lu\n",
                  8 << i, bitmapSizes[i] ));
    }

    FT_TRACE4(( "  %lu bitmaps, using %ld-bit padding\n",
                nbitmaps,
                8 << PCF_GLYPH_PAD_INDEX( format ) ));
    FT_TRACE4(( "  bitmap size: %lu\n", sizebitmaps ));

    FT_UNUSED( sizebitmaps );       /* only used for debugging */

    /* right now, we only check the bitmap offsets; */
    /* actual bitmaps are only loaded on demand     */
    for ( i = 0; i < nbitmaps; i++ )
    {
      /* rough estimate */
      if ( offsets[i] > size )
      {
        FT_TRACE0(( "pcf_get_bitmaps:"
                    " invalid offset to bitmap data of glyph %lu\n", i ));
      }
      else
        face->metrics[i].bits = stream->pos + offsets[i];
    }

    face->bitmapsFormat = format;

  Bail:
    FT_FREE( offsets );
    return error;
  }


  /*
   * This file uses X11 terminology for PCF data; an `encoding' in X11 speak
   * is the same as a character code in FreeType speak.
   */
#define PCF_ENC_SIZE  10

  static
  const FT_Frame_Field  pcf_enc_header[] =
  {
#undef  FT_STRUCTURE
#define FT_STRUCTURE  PCF_EncRec

    FT_FRAME_START( PCF_ENC_SIZE ),
      FT_FRAME_USHORT_LE( firstCol ),
      FT_FRAME_USHORT_LE( lastCol ),
      FT_FRAME_USHORT_LE( firstRow ),
      FT_FRAME_USHORT_LE( lastRow ),
      FT_FRAME_USHORT_LE( defaultChar ),
    FT_FRAME_END
  };


  static
  const FT_Frame_Field  pcf_enc_msb_header[] =
  {
#undef  FT_STRUCTURE
#define FT_STRUCTURE  PCF_EncRec

    FT_FRAME_START( PCF_ENC_SIZE ),
      FT_FRAME_USHORT( firstCol ),
      FT_FRAME_USHORT( lastCol ),
      FT_FRAME_USHORT( firstRow ),
      FT_FRAME_USHORT( lastRow ),
      FT_FRAME_USHORT( defaultChar ),
    FT_FRAME_END
  };


  static FT_Error
  pcf_get_encodings( FT_Stream  stream,
                     PCF_Face   face )
  {
    FT_Error    error;
    FT_Memory   memory = FT_FACE( face )->memory;
    FT_ULong    format, size;
    PCF_Enc     enc = &face->enc;
    FT_ULong    nencoding;
    FT_UShort*  offset;
    FT_UShort   defaultCharRow, defaultCharCol;
    FT_UShort   encodingOffset, defaultCharEncodingOffset;
    FT_UShort   i, j;
    FT_Byte*    pos;


    error = pcf_seek_to_table_type( stream,
                                    face->toc.tables,
                                    face->toc.count,
                                    PCF_BDF_ENCODINGS,
                                    &format,
                                    &size );
    if ( error )
      goto Bail;

    if ( FT_READ_ULONG_LE( format ) )
      goto Bail;

    FT_TRACE4(( "pcf_get_encodings:\n"
                "  format: 0x%lX (%s)\n",
                format,
                PCF_BYTE_ORDER( format ) == MSBFirst ? "MSB" : "LSB" ));

    if ( !PCF_FORMAT_MATCH( format, PCF_DEFAULT_FORMAT ) &&
         !PCF_FORMAT_MATCH( format, PCF_BDF_ENCODINGS )  )
      return FT_THROW( Invalid_File_Format );

    if ( PCF_BYTE_ORDER( format ) == MSBFirst )
    {
      if ( FT_STREAM_READ_FIELDS( pcf_enc_msb_header, enc ) )
        goto Bail;
    }
    else
    {
      if ( FT_STREAM_READ_FIELDS( pcf_enc_header, enc ) )
        goto Bail;
    }

    FT_TRACE4(( "  firstCol 0x%X, lastCol 0x%X\n"
                "  firstRow 0x%X, lastRow 0x%X\n"
                "  defaultChar 0x%X\n",
                enc->firstCol, enc->lastCol,
                enc->firstRow, enc->lastRow,
                enc->defaultChar ));

    /* sanity checks; we limit numbers of rows and columns to 256 */
    if ( enc->firstCol > enc->lastCol ||
         enc->lastCol  > 0xFF         ||
         enc->firstRow > enc->lastRow ||
         enc->lastRow  > 0xFF         )
      return FT_THROW( Invalid_Table );

    nencoding = (FT_ULong)( enc->lastCol - enc->firstCol + 1 ) *
                (FT_ULong)( enc->lastRow - enc->firstRow + 1 );

    if ( FT_NEW_ARRAY( enc->offset, nencoding ) )
      goto Bail;

    error = FT_Stream_EnterFrame( stream, 2 * nencoding );
    if ( error )
      goto Exit;

    FT_TRACE5(( "\n" ));

    defaultCharRow = enc->defaultChar >> 8;
    defaultCharCol = enc->defaultChar & 0xFF;

    /* validate default character */
    if ( defaultCharRow < enc->firstRow ||
         defaultCharRow > enc->lastRow  ||
         defaultCharCol < enc->firstCol ||
         defaultCharCol > enc->lastCol  )
    {
      enc->defaultChar = enc->firstRow * 256U + enc->firstCol;
      FT_TRACE0(( "pcf_get_encodings:"
                  " Invalid default character set to %u\n",
                  enc->defaultChar ));

      defaultCharRow = enc->firstRow;
      defaultCharCol = enc->firstCol;
    }

    /* FreeType mandates that glyph index 0 is the `undefined glyph',  */
    /* which PCF calls the `default character'.  For this reason, we   */
    /* swap the positions of glyph index 0 and the index corresponding */
    /* to `defaultChar' in case they are different.                    */

    /* `stream->cursor' still points at the beginning of the frame; */
    /* we can thus easily get the offset to the default character   */
    pos = stream->cursor +
            2 * ( ( defaultCharRow - enc->firstRow ) *
                  ( enc->lastCol - enc->firstCol + 1 ) +
                    defaultCharCol - enc->firstCol       );

    if ( PCF_BYTE_ORDER( format ) == MSBFirst )
      defaultCharEncodingOffset = FT_PEEK_USHORT( pos );
    else
      defaultCharEncodingOffset = FT_PEEK_USHORT_LE( pos );

    if ( defaultCharEncodingOffset >= face->nmetrics )
    {
      FT_TRACE0(( "pcf_get_encodings:"
                  " Invalid glyph index for default character,"
                  " setting to zero\n" ));
      defaultCharEncodingOffset = 0;
    }

    if ( defaultCharEncodingOffset )
    {
      /* do the swapping */
      PCF_MetricRec  tmp = face->metrics[defaultCharEncodingOffset];


      face->metrics[defaultCharEncodingOffset] = face->metrics[0];
      face->metrics[0]                         = tmp;
    }

    offset = enc->offset;
    for ( i = enc->firstRow; i <= enc->lastRow; i++ )
    {
      for ( j = enc->firstCol; j <= enc->lastCol; j++ )
      {
        /* X11's reference implementation uses the equivalent to  */
        /* `FT_GET_SHORT', however PCF fonts with more than 32768 */
        /* characters (e.g., `unifont.pcf') clearly show that an  */
        /* unsigned value is needed.                              */
        if ( PCF_BYTE_ORDER( format ) == MSBFirst )
          encodingOffset = FT_GET_USHORT();
        else
          encodingOffset = FT_GET_USHORT_LE();

        if ( encodingOffset != 0xFFFFU )
        {
          if ( encodingOffset == defaultCharEncodingOffset )
            encodingOffset = 0;
          else if ( encodingOffset == 0 )
            encodingOffset = defaultCharEncodingOffset;
        }

        *offset++ = encodingOffset;
      }
    }
    FT_Stream_ExitFrame( stream );

    return error;

  Exit:
    FT_FREE( enc->offset );

  Bail:
    return error;
  }


  static
  const FT_Frame_Field  pcf_accel_header[] =
  {
#undef  FT_STRUCTURE
#define FT_STRUCTURE  PCF_AccelRec

    FT_FRAME_START( 20 ),
      FT_FRAME_BYTE      ( noOverlap ),
      FT_FRAME_BYTE      ( constantMetrics ),
      FT_FRAME_BYTE      ( terminalFont ),
      FT_FRAME_BYTE      ( constantWidth ),
      FT_FRAME_BYTE      ( inkInside ),
      FT_FRAME_BYTE      ( inkMetrics ),
      FT_FRAME_BYTE      ( drawDirection ),
      FT_FRAME_SKIP_BYTES( 1 ),
      FT_FRAME_LONG_LE   ( fontAscent ),
      FT_FRAME_LONG_LE   ( fontDescent ),
      FT_FRAME_LONG_LE   ( maxOverlap ),
    FT_FRAME_END
  };


  static
  const FT_Frame_Field  pcf_accel_msb_header[] =
  {
#undef  FT_STRUCTURE
#define FT_STRUCTURE  PCF_AccelRec

    FT_FRAME_START( 20 ),
      FT_FRAME_BYTE      ( noOverlap ),
      FT_FRAME_BYTE      ( constantMetrics ),
      FT_FRAME_BYTE      ( terminalFont ),
      FT_FRAME_BYTE      ( constantWidth ),
      FT_FRAME_BYTE      ( inkInside ),
      FT_FRAME_BYTE      ( inkMetrics ),
      FT_FRAME_BYTE      ( drawDirection ),
      FT_FRAME_SKIP_BYTES( 1 ),
      FT_FRAME_LONG      ( fontAscent ),
      FT_FRAME_LONG      ( fontDescent ),
      FT_FRAME_LONG      ( maxOverlap ),
    FT_FRAME_END
  };


  static FT_Error
  pcf_get_accel( FT_Stream  stream,
                 PCF_Face   face,
                 FT_ULong   type )
  {
    FT_ULong   format, size;
    FT_Error   error;
    PCF_Accel  accel = &face->accel;


    error = pcf_seek_to_table_type( stream,
                                    face->toc.tables,
                                    face->toc.count,
                                    type,
                                    &format,
                                    &size );
    if ( error )
      goto Bail;

    if ( FT_READ_ULONG_LE( format ) )
      goto Bail;

    FT_TRACE4(( "pcf_get_accel%s:\n"
                "  format: 0x%lX (%s, %s)\n",
                type == PCF_BDF_ACCELERATORS ? " (getting BDF accelerators)"
                                             : "",
                format,
                PCF_BYTE_ORDER( format ) == MSBFirst ? "MSB" : "LSB",
                PCF_FORMAT_MATCH( format, PCF_ACCEL_W_INKBOUNDS ) ?
                  "accelerated" : "not accelerated" ));

    if ( !PCF_FORMAT_MATCH( format, PCF_DEFAULT_FORMAT )    &&
         !PCF_FORMAT_MATCH( format, PCF_ACCEL_W_INKBOUNDS ) )
      goto Bail;

    if ( PCF_BYTE_ORDER( format ) == MSBFirst )
    {
      if ( FT_STREAM_READ_FIELDS( pcf_accel_msb_header, accel ) )
        goto Bail;
    }
    else
    {
      if ( FT_STREAM_READ_FIELDS( pcf_accel_header, accel ) )
        goto Bail;
    }

    FT_TRACE5(( "  noOverlap=%s, constantMetrics=%s,"
                " terminalFont=%s, constantWidth=%s\n"
                "  inkInside=%s, inkMetrics=%s, drawDirection=%s\n"
                "  fontAscent=%ld, fontDescent=%ld, maxOverlap=%ld\n",
                accel->noOverlap ? "yes" : "no",
                accel->constantMetrics ? "yes" : "no",
                accel->terminalFont ? "yes" : "no",
                accel->constantWidth ? "yes" : "no",
                accel->inkInside ? "yes" : "no",
                accel->inkMetrics ? "yes" : "no",
                accel->drawDirection ? "RTL" : "LTR",
                accel->fontAscent,
                accel->fontDescent,
                accel->maxOverlap ));

    /* sanity checks */
    if ( FT_ABS( accel->fontAscent ) > 0x7FFF )
    {
      accel->fontAscent = accel->fontAscent < 0 ? -0x7FFF : 0x7FFF;
      FT_TRACE0(( "pfc_get_accel: clamping font ascent to value %d\n",
                  accel->fontAscent ));
    }
    if ( FT_ABS( accel->fontDescent ) > 0x7FFF )
    {
      accel->fontDescent = accel->fontDescent < 0 ? -0x7FFF : 0x7FFF;
      FT_TRACE0(( "pfc_get_accel: clamping font descent to value %d\n",
                  accel->fontDescent ));
    }

    FT_TRACE5(( "  minbounds:" ));
    error = pcf_get_metric( stream,
                            format & ( ~PCF_FORMAT_MASK ),
                            &(accel->minbounds) );
    if ( error )
      goto Bail;

    FT_TRACE5(( "  maxbounds:" ));
    error = pcf_get_metric( stream,
                            format & ( ~PCF_FORMAT_MASK ),
                            &(accel->maxbounds) );
    if ( error )
      goto Bail;

    if ( PCF_FORMAT_MATCH( format, PCF_ACCEL_W_INKBOUNDS ) )
    {
      FT_TRACE5(( "  ink minbounds:" ));
      error = pcf_get_metric( stream,
                              format & ( ~PCF_FORMAT_MASK ),
                              &(accel->ink_minbounds) );
      if ( error )
        goto Bail;

      FT_TRACE5(( "  ink maxbounds:" ));
      error = pcf_get_metric( stream,
                              format & ( ~PCF_FORMAT_MASK ),
                              &(accel->ink_maxbounds) );
      if ( error )
        goto Bail;
    }
    else
    {
      accel->ink_minbounds = accel->minbounds;
      accel->ink_maxbounds = accel->maxbounds;
    }

  Bail:
    return error;
  }


  static FT_Error
  pcf_interpret_style( PCF_Face  pcf )
  {
    FT_Error   error  = FT_Err_Ok;
    FT_Face    face   = FT_FACE( pcf );
    FT_Memory  memory = face->memory;

    PCF_Property  prop;

    size_t  nn, len;
    char*   strings[4] = { NULL, NULL, NULL, NULL };
    size_t  lengths[4];


    face->style_flags = 0;

    prop = pcf_find_property( pcf, "SLANT" );
    if ( prop && prop->isString                                       &&
         ( *(prop->value.atom) == 'O' || *(prop->value.atom) == 'o' ||
           *(prop->value.atom) == 'I' || *(prop->value.atom) == 'i' ) )
    {
      face->style_flags |= FT_STYLE_FLAG_ITALIC;
      strings[2] = ( *(prop->value.atom) == 'O' ||
                     *(prop->value.atom) == 'o' ) ? (char *)"Oblique"
                                                  : (char *)"Italic";
    }

    prop = pcf_find_property( pcf, "WEIGHT_NAME" );
    if ( prop && prop->isString                                       &&
         ( *(prop->value.atom) == 'B' || *(prop->value.atom) == 'b' ) )
    {
      face->style_flags |= FT_STYLE_FLAG_BOLD;
      strings[1] = (char*)"Bold";
    }

    prop = pcf_find_property( pcf, "SETWIDTH_NAME" );
    if ( prop && prop->isString                                        &&
         *(prop->value.atom)                                           &&
         !( *(prop->value.atom) == 'N' || *(prop->value.atom) == 'n' ) )
      strings[3] = (char*)( prop->value.atom );

    prop = pcf_find_property( pcf, "ADD_STYLE_NAME" );
    if ( prop && prop->isString                                        &&
         *(prop->value.atom)                                           &&
         !( *(prop->value.atom) == 'N' || *(prop->value.atom) == 'n' ) )
      strings[0] = (char*)( prop->value.atom );

    for ( len = 0, nn = 0; nn < 4; nn++ )
    {
      lengths[nn] = 0;
      if ( strings[nn] )
      {
        lengths[nn] = ft_strlen( strings[nn] );
        len        += lengths[nn] + 1;
      }
    }

    if ( len == 0 )
    {
      strings[0] = (char*)"Regular";
      lengths[0] = ft_strlen( strings[0] );
      len        = lengths[0] + 1;
    }

    {
      char*  s;


      if ( FT_ALLOC( face->style_name, len ) )
        return error;

      s = face->style_name;

      for ( nn = 0; nn < 4; nn++ )
      {
        char*  src = strings[nn];


        len = lengths[nn];

        if ( !src )
          continue;

        /* separate elements with a space */
        if ( s != face->style_name )
          *s++ = ' ';

        ft_memcpy( s, src, len );

        /* need to convert spaces to dashes for */
        /* add_style_name and setwidth_name     */
        if ( nn == 0 || nn == 3 )
        {
          size_t  mm;


          for ( mm = 0; mm < len; mm++ )
            if ( s[mm] == ' ' )
              s[mm] = '-';
        }

        s += len;
      }
      *s = 0;
    }

    return error;
  }


  FT_LOCAL_DEF( FT_Error )
  pcf_load_font( FT_Stream  stream,
                 PCF_Face   face,
                 FT_Long    face_index )
  {
    FT_Face    root   = FT_FACE( face );
    FT_Error   error;
    FT_Memory  memory = FT_FACE( face )->memory;
    FT_Bool    hasBDFAccelerators;


    error = pcf_read_TOC( stream, face );
    if ( error )
      goto Exit;

    root->num_faces  = 1;
    root->face_index = 0;

    /* If we are performing a simple font format check, exit immediately. */
    if ( face_index < 0 )
      return FT_Err_Ok;

    error = pcf_get_properties( stream, face );
    if ( error )
      goto Exit;

    /* Use the old accelerators if no BDF accelerators are in the file. */
    hasBDFAccelerators = pcf_has_table_type( face->toc.tables,
                                             face->toc.count,
                                             PCF_BDF_ACCELERATORS );
    if ( !hasBDFAccelerators )
    {
      error = pcf_get_accel( stream, face, PCF_ACCELERATORS );
      if ( error )
        goto Exit;
    }

    /* metrics */
    error = pcf_get_metrics( stream, face );
    if ( error )
      goto Exit;

    /* bitmaps */
    error = pcf_get_bitmaps( stream, face );
    if ( error )
      goto Exit;

    /* encodings */
    error = pcf_get_encodings( stream, face );
    if ( error )
      goto Exit;

    /* BDF style accelerators (i.e. bounds based on encoded glyphs) */
    if ( hasBDFAccelerators )
    {
      error = pcf_get_accel( stream, face, PCF_BDF_ACCELERATORS );
      if ( error )
        goto Exit;
    }

    /* XXX: TO DO: inkmetrics and glyph_names are missing */

    /* now construct the face object */
    {
      PCF_Property  prop;


      root->face_flags |= FT_FACE_FLAG_FIXED_SIZES |
                          FT_FACE_FLAG_HORIZONTAL;

      if ( face->accel.constantWidth )
        root->face_flags |= FT_FACE_FLAG_FIXED_WIDTH;

      if ( FT_SET_ERROR( pcf_interpret_style( face ) ) )
        goto Exit;

      prop = pcf_find_property( face, "FAMILY_NAME" );
      if ( prop && prop->isString )
      {

#ifdef PCF_CONFIG_OPTION_LONG_FAMILY_NAMES

        PCF_Driver  driver = (PCF_Driver)FT_FACE_DRIVER( face );


        if ( !driver->no_long_family_names )
        {
          /* Prepend the foundry name plus a space to the family name.     */
          /* There are many fonts just called `Fixed' which look           */
          /* completely different, and which have nothing to do with each  */
          /* other.  When selecting `Fixed' in KDE or Gnome one gets       */
          /* results that appear rather random, the style changes often if */
          /* one changes the size and one cannot select some fonts at all. */
          /*                                                               */
          /* We also check whether we have `wide' characters; all put      */
          /* together, we get family names like `Sony Fixed' or `Misc      */
          /* Fixed Wide'.                                                  */

          PCF_Property  foundry_prop, point_size_prop, average_width_prop;

          int  l    = ft_strlen( prop->value.atom ) + 1;
          int  wide = 0;


          foundry_prop       = pcf_find_property( face, "FOUNDRY" );
          point_size_prop    = pcf_find_property( face, "POINT_SIZE" );
          average_width_prop = pcf_find_property( face, "AVERAGE_WIDTH" );

          if ( point_size_prop && average_width_prop )
          {
            if ( average_width_prop->value.l >= point_size_prop->value.l )
            {
              /* This font is at least square shaped or even wider */
              wide = 1;
              l   += ft_strlen( " Wide" );
            }
          }

          if ( foundry_prop && foundry_prop->isString )
          {
            l += ft_strlen( foundry_prop->value.atom ) + 1;

            if ( FT_NEW_ARRAY( root->family_name, l ) )
              goto Exit;

            ft_strcpy( root->family_name, foundry_prop->value.atom );
            ft_strcat( root->family_name, " " );
            ft_strcat( root->family_name, prop->value.atom );
          }
          else
          {
            if ( FT_NEW_ARRAY( root->family_name, l ) )
              goto Exit;

            ft_strcpy( root->family_name, prop->value.atom );
          }

          if ( wide )
            ft_strcat( root->family_name, " Wide" );
        }
        else

#endif /* PCF_CONFIG_OPTION_LONG_FAMILY_NAMES */

        {
          if ( FT_STRDUP( root->family_name, prop->value.atom ) )
            goto Exit;
        }
      }
      else
        root->family_name = NULL;

      root->num_glyphs = (FT_Long)face->nmetrics;

      root->num_fixed_sizes = 1;
      if ( FT_NEW_ARRAY( root->available_sizes, 1 ) )
        goto Exit;

      {
        FT_Bitmap_Size*  bsize = root->available_sizes;
        FT_Short         resolution_x = 0, resolution_y = 0;


        FT_ZERO( bsize );

        /* for simplicity, we take absolute values of integer properties */

#if 0
        bsize->height = face->accel.maxbounds.ascent << 6;
#endif

#ifdef FT_DEBUG_LEVEL_TRACE
        if ( face->accel.fontAscent + face->accel.fontDescent < 0 )
          FT_TRACE0(( "pcf_load_font: negative height\n" ));
#endif
        if ( FT_ABS( face->accel.fontAscent +
                     face->accel.fontDescent ) > 0x7FFF )
        {
          bsize->height = 0x7FFF;
          FT_TRACE0(( "pcf_load_font: clamping height to value %d\n",
                      bsize->height ));
        }
        else
          bsize->height = FT_ABS( (FT_Short)( face->accel.fontAscent +
                                              face->accel.fontDescent ) );

        prop = pcf_find_property( face, "AVERAGE_WIDTH" );
        if ( prop )
        {
#ifdef FT_DEBUG_LEVEL_TRACE
          if ( prop->value.l < 0 )
            FT_TRACE0(( "pcf_load_font: negative average width\n" ));
#endif
          if ( ( FT_ABS( prop->value.l ) > 0x7FFFL * 10 - 5 ) )
          {
            bsize->width = 0x7FFF;
            FT_TRACE0(( "pcf_load_font: clamping average width to value %d\n",
                        bsize->width ));
          }
          else
            bsize->width = FT_ABS( (FT_Short)( ( prop->value.l + 5 ) / 10 ) );
        }
        else
        {
          /* this is a heuristical value */
          bsize->width = (FT_Short)FT_MulDiv( bsize->height, 2, 3 );
        }

        prop = pcf_find_property( face, "POINT_SIZE" );
        if ( prop )
        {
#ifdef FT_DEBUG_LEVEL_TRACE
          if ( prop->value.l < 0 )
            FT_TRACE0(( "pcf_load_font: negative point size\n" ));
#endif
          /* convert from 722.7 decipoints to 72 points per inch */
          if ( FT_ABS( prop->value.l ) > 0x504C2L ) /* 0x7FFF * 72270/7200 */
          {
            bsize->size = 0x7FFF;
            FT_TRACE0(( "pcf_load_font: clamping point size to value %d\n",
                        bsize->size ));
          }
          else
            bsize->size = FT_MulDiv( FT_ABS( prop->value.l ),
                                     64 * 7200,
                                     72270L );
        }

        prop = pcf_find_property( face, "PIXEL_SIZE" );
        if ( prop )
        {
#ifdef FT_DEBUG_LEVEL_TRACE
          if ( prop->value.l < 0 )
            FT_TRACE0(( "pcf_load_font: negative pixel size\n" ));
#endif
          if ( FT_ABS( prop->value.l ) > 0x7FFF )
          {
            bsize->y_ppem = 0x7FFF << 6;
            FT_TRACE0(( "pcf_load_font: clamping pixel size to value %d\n",
                        bsize->y_ppem ));
          }
          else
            bsize->y_ppem = FT_ABS( (FT_Short)prop->value.l ) << 6;
        }

        prop = pcf_find_property( face, "RESOLUTION_X" );
        if ( prop )
        {
#ifdef FT_DEBUG_LEVEL_TRACE
          if ( prop->value.l < 0 )
            FT_TRACE0(( "pcf_load_font: negative X resolution\n" ));
#endif
          if ( FT_ABS( prop->value.l ) > 0x7FFF )
          {
            resolution_x = 0x7FFF;
            FT_TRACE0(( "pcf_load_font: clamping X resolution to value %d\n",
                        resolution_x ));
          }
          else
            resolution_x = FT_ABS( (FT_Short)prop->value.l );
        }

        prop = pcf_find_property( face, "RESOLUTION_Y" );
        if ( prop )
        {
#ifdef FT_DEBUG_LEVEL_TRACE
          if ( prop->value.l < 0 )
            FT_TRACE0(( "pcf_load_font: negative Y resolution\n" ));
#endif
          if ( FT_ABS( prop->value.l ) > 0x7FFF )
          {
            resolution_y = 0x7FFF;
            FT_TRACE0(( "pcf_load_font: clamping Y resolution to value %d\n",
                        resolution_y ));
          }
          else
            resolution_y = FT_ABS( (FT_Short)prop->value.l );
        }

        if ( bsize->y_ppem == 0 )
        {
          bsize->y_ppem = bsize->size;
          if ( resolution_y )
            bsize->y_ppem = FT_MulDiv( bsize->y_ppem, resolution_y, 72 );
        }
        if ( resolution_x && resolution_y )
          bsize->x_ppem = FT_MulDiv( bsize->y_ppem,
                                     resolution_x,
                                     resolution_y );
        else
          bsize->x_ppem = bsize->y_ppem;
      }

      /* set up charset */
      {
        PCF_Property  charset_registry, charset_encoding;


        charset_registry = pcf_find_property( face, "CHARSET_REGISTRY" );
        charset_encoding = pcf_find_property( face, "CHARSET_ENCODING" );

        if ( charset_registry && charset_registry->isString &&
             charset_encoding && charset_encoding->isString )
        {
          if ( FT_STRDUP( face->charset_encoding,
                          charset_encoding->value.atom ) ||
               FT_STRDUP( face->charset_registry,
                          charset_registry->value.atom ) )
            goto Exit;
        }
      }
    }

  Exit:
    if ( error )
    {
      /* This is done to respect the behaviour of the original */
      /* PCF font driver.                                      */
      error = FT_THROW( Invalid_File_Format );
    }

    return error;
  }


/* END */
