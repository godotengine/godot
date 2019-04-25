/*  pcf.h

  FreeType font driver for pcf fonts

  Copyright (C) 2000, 2001, 2002, 2003, 2006, 2010 by
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


#ifndef PCF_H_
#define PCF_H_


#include <ft2build.h>
#include FT_INTERNAL_DRIVER_H
#include FT_INTERNAL_STREAM_H


FT_BEGIN_HEADER

  typedef struct  PCF_TableRec_
  {
    FT_ULong  type;
    FT_ULong  format;
    FT_ULong  size;
    FT_ULong  offset;

  } PCF_TableRec, *PCF_Table;


  typedef struct  PCF_TocRec_
  {
    FT_ULong   version;
    FT_ULong   count;
    PCF_Table  tables;

  } PCF_TocRec, *PCF_Toc;


  typedef struct  PCF_ParsePropertyRec_
  {
    FT_Long  name;
    FT_Byte  isString;
    FT_Long  value;

  } PCF_ParsePropertyRec, *PCF_ParseProperty;


  typedef struct  PCF_PropertyRec_
  {
    FT_String*  name;
    FT_Byte     isString;

    union
    {
      FT_String*  atom;
      FT_Long     l;
      FT_ULong    ul;

    } value;

  } PCF_PropertyRec, *PCF_Property;


  typedef struct  PCF_Compressed_MetricRec_
  {
    FT_Byte  leftSideBearing;
    FT_Byte  rightSideBearing;
    FT_Byte  characterWidth;
    FT_Byte  ascent;
    FT_Byte  descent;

  } PCF_Compressed_MetricRec, *PCF_Compressed_Metric;


  typedef struct  PCF_MetricRec_
  {
    FT_Short  leftSideBearing;
    FT_Short  rightSideBearing;
    FT_Short  characterWidth;
    FT_Short  ascent;
    FT_Short  descent;
    FT_Short  attributes;
    FT_ULong  bits;

  } PCF_MetricRec, *PCF_Metric;


  typedef struct  PCF_AccelRec_
  {
    FT_Byte        noOverlap;
    FT_Byte        constantMetrics;
    FT_Byte        terminalFont;
    FT_Byte        constantWidth;
    FT_Byte        inkInside;
    FT_Byte        inkMetrics;
    FT_Byte        drawDirection;
    FT_Long        fontAscent;
    FT_Long        fontDescent;
    FT_Long        maxOverlap;
    PCF_MetricRec  minbounds;
    PCF_MetricRec  maxbounds;
    PCF_MetricRec  ink_minbounds;
    PCF_MetricRec  ink_maxbounds;

  } PCF_AccelRec, *PCF_Accel;


  typedef struct  PCF_EncodingRec_
  {
    FT_Long    enc;
    FT_UShort  glyph;

  } PCF_EncodingRec, *PCF_Encoding;


  typedef struct  PCF_FaceRec_
  {
    FT_FaceRec     root;

    FT_StreamRec   comp_stream;
    FT_Stream      comp_source;

    char*          charset_encoding;
    char*          charset_registry;

    PCF_TocRec     toc;
    PCF_AccelRec   accel;

    int            nprops;
    PCF_Property   properties;

    FT_ULong       nmetrics;
    PCF_Metric     metrics;
    FT_ULong       nencodings;
    PCF_Encoding   encodings;

    FT_Short       defaultChar;

    FT_ULong       bitmapsFormat;

    FT_CharMap     charmap_handle;
    FT_CharMapRec  charmap;  /* a single charmap per face */

  } PCF_FaceRec, *PCF_Face;


  typedef struct  PCF_DriverRec_
  {
    FT_DriverRec  root;

    FT_Bool  no_long_family_names;

  } PCF_DriverRec, *PCF_Driver;


  /* macros for pcf font format */

#define LSBFirst  0
#define MSBFirst  1

#define PCF_FILE_VERSION        ( ( 'p' << 24 ) | \
                                  ( 'c' << 16 ) | \
                                  ( 'f' <<  8 ) | 1 )
#define PCF_FORMAT_MASK         0xFFFFFF00UL

#define PCF_DEFAULT_FORMAT      0x00000000UL
#define PCF_INKBOUNDS           0x00000200UL
#define PCF_ACCEL_W_INKBOUNDS   0x00000100UL
#define PCF_COMPRESSED_METRICS  0x00000100UL

#define PCF_FORMAT_MATCH( a, b ) \
          ( ( (a) & PCF_FORMAT_MASK ) == ( (b) & PCF_FORMAT_MASK ) )

#define PCF_GLYPH_PAD_MASK  ( 3 << 0 )
#define PCF_BYTE_MASK       ( 1 << 2 )
#define PCF_BIT_MASK        ( 1 << 3 )
#define PCF_SCAN_UNIT_MASK  ( 3 << 4 )

#define PCF_BYTE_ORDER( f ) \
          ( ( (f) & PCF_BYTE_MASK ) ? MSBFirst : LSBFirst )
#define PCF_BIT_ORDER( f ) \
          ( ( (f) & PCF_BIT_MASK ) ? MSBFirst : LSBFirst )
#define PCF_GLYPH_PAD_INDEX( f ) \
          ( (f) & PCF_GLYPH_PAD_MASK )
#define PCF_GLYPH_PAD( f ) \
          ( 1 << PCF_GLYPH_PAD_INDEX( f ) )
#define PCF_SCAN_UNIT_INDEX( f ) \
          ( ( (f) & PCF_SCAN_UNIT_MASK ) >> 4 )
#define PCF_SCAN_UNIT( f ) \
          ( 1 << PCF_SCAN_UNIT_INDEX( f ) )
#define PCF_FORMAT_BITS( f )             \
          ( (f) & ( PCF_GLYPH_PAD_MASK | \
                    PCF_BYTE_MASK      | \
                    PCF_BIT_MASK       | \
                    PCF_SCAN_UNIT_MASK ) )

#define PCF_SIZE_TO_INDEX( s )  ( (s) == 4 ? 2 : (s) == 2 ? 1 : 0 )
#define PCF_INDEX_TO_SIZE( b )  ( 1 << b )

#define PCF_FORMAT( bit, byte, glyph, scan )          \
          ( ( PCF_SIZE_TO_INDEX( scan )      << 4 ) | \
            ( ( (bit)  == MSBFirst ? 1 : 0 ) << 3 ) | \
            ( ( (byte) == MSBFirst ? 1 : 0 ) << 2 ) | \
            ( PCF_SIZE_TO_INDEX( glyph )     << 0 ) )

#define PCF_PROPERTIES        ( 1 << 0 )
#define PCF_ACCELERATORS      ( 1 << 1 )
#define PCF_METRICS           ( 1 << 2 )
#define PCF_BITMAPS           ( 1 << 3 )
#define PCF_INK_METRICS       ( 1 << 4 )
#define PCF_BDF_ENCODINGS     ( 1 << 5 )
#define PCF_SWIDTHS           ( 1 << 6 )
#define PCF_GLYPH_NAMES       ( 1 << 7 )
#define PCF_BDF_ACCELERATORS  ( 1 << 8 )

#define GLYPHPADOPTIONS  4 /* I'm not sure about this */

  FT_LOCAL( FT_Error )
  pcf_load_font( FT_Stream  stream,
                 PCF_Face   face,
                 FT_Long    face_index );

FT_END_HEADER

#endif /* PCF_H_ */


/* END */
