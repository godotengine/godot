/*
 * Copyright 2000 Computing Research Labs, New Mexico State University
 * Copyright 2001-2004, 2011 Francesco Zappa Nardelli
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE COMPUTING RESEARCH LAB OR NEW MEXICO STATE UNIVERSITY BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT
 * OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR
 * THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */


#ifndef BDF_H_
#define BDF_H_


/*
 * Based on bdf.h,v 1.16 2000/03/16 20:08:51 mleisher
 */

#include <freetype/internal/ftobjs.h>
#include <freetype/internal/ftstream.h>
#include <freetype/internal/fthash.h>


FT_BEGIN_HEADER


/* Imported from bdfP.h */

#define _bdf_glyph_modified( map, e )                     \
          ( (map)[(e) >> 5] & ( 1UL << ( (e) & 31 ) ) )
#define _bdf_set_glyph_modified( map, e )                 \
          ( (map)[(e) >> 5] |= ( 1UL << ( (e) & 31 ) ) )
#define _bdf_clear_glyph_modified( map, e )               \
          ( (map)[(e) >> 5] &= ~( 1UL << ( (e) & 31 ) ) )

/* end of bdfP.h */


  /**************************************************************************
   *
   * BDF font options macros and types.
   *
   */


#define BDF_CORRECT_METRICS  0x01 /* Correct invalid metrics when loading. */
#define BDF_KEEP_COMMENTS    0x02 /* Preserve the font comments.           */
#define BDF_KEEP_UNENCODED   0x04 /* Keep the unencoded glyphs.            */
#define BDF_PROPORTIONAL     0x08 /* Font has proportional spacing.        */
#define BDF_MONOWIDTH        0x10 /* Font has mono width.                  */
#define BDF_CHARCELL         0x20 /* Font has charcell spacing.            */

#define BDF_ALL_SPACING  ( BDF_PROPORTIONAL | \
                           BDF_MONOWIDTH    | \
                           BDF_CHARCELL     )

#define BDF_DEFAULT_LOAD_OPTIONS  ( BDF_CORRECT_METRICS | \
                                    BDF_KEEP_COMMENTS   | \
                                    BDF_KEEP_UNENCODED  | \
                                    BDF_PROPORTIONAL    )


  typedef struct  bdf_options_t_
  {
    int            correct_metrics;
    int            keep_unencoded;
    int            keep_comments;
    int            font_spacing;

  } bdf_options_t;


  /* Callback function type for unknown configuration options. */
  typedef int
  (*bdf_options_callback_t)( bdf_options_t*  opts,
                             char**          params,
                             unsigned long   nparams,
                             void*           client_data );


  /**************************************************************************
   *
   * BDF font property macros and types.
   *
   */


#define BDF_ATOM      1
#define BDF_INTEGER   2
#define BDF_CARDINAL  3


  /* This structure represents a particular property of a font. */
  /* There are a set of defaults and each font has their own.   */
  typedef struct  bdf_property_t_
  {
    const char*  name;         /* Name of the property.   */
    int          format;       /* Format of the property. */
    int          builtin;      /* A builtin property.     */
    union
    {
      char*          atom;
      long           l;
      unsigned long  ul;

    } value;             /* Value of the property.  */

  } bdf_property_t;


  /**************************************************************************
   *
   * BDF font metric and glyph types.
   *
   */


  typedef struct  bdf_bbx_t_
  {
    unsigned short  width;
    unsigned short  height;

    short           x_offset;
    short           y_offset;

    short           ascent;
    short           descent;

  } bdf_bbx_t;


  typedef struct  bdf_glyph_t_
  {
    char*           name;        /* Glyph name.                          */
    unsigned long   encoding;    /* Glyph encoding.                      */
    unsigned short  swidth;      /* Scalable width.                      */
    unsigned short  dwidth;      /* Device width.                        */
    bdf_bbx_t       bbx;         /* Glyph bounding box.                  */
    unsigned char*  bitmap;      /* Glyph bitmap.                        */
    unsigned long   bpr;         /* Number of bytes used per row.        */
    unsigned short  bytes;       /* Number of bytes used for the bitmap. */

  } bdf_glyph_t;


  typedef struct  bdf_font_t_
  {
    char*            name;           /* Name of the font.                   */
    bdf_bbx_t        bbx;            /* Font bounding box.                  */

    unsigned long    point_size;     /* Point size of the font.             */
    unsigned long    resolution_x;   /* Font horizontal resolution.         */
    unsigned long    resolution_y;   /* Font vertical resolution.           */

    int              spacing;        /* Font spacing value.                 */

    unsigned short   monowidth;      /* Logical width for monowidth font.   */

    unsigned long    default_char;   /* Encoding of the default glyph.      */

    long             font_ascent;    /* Font ascent.                        */
    long             font_descent;   /* Font descent.                       */

    unsigned long    glyphs_size;    /* Glyph structures allocated.         */
    unsigned long    glyphs_used;    /* Glyph structures used.              */
    bdf_glyph_t*     glyphs;         /* Glyphs themselves.                  */

    unsigned long    unencoded_size; /* Unencoded glyph struct. allocated.  */
    unsigned long    unencoded_used; /* Unencoded glyph struct. used.       */
    bdf_glyph_t*     unencoded;      /* Unencoded glyphs themselves.        */

    unsigned long    props_size;     /* Font properties allocated.          */
    unsigned long    props_used;     /* Font properties used.               */
    bdf_property_t*  props;          /* Font properties themselves.         */

    char*            comments;       /* Font comments.                      */
    unsigned long    comments_len;   /* Length of comment string.           */

    void*            internal;       /* Internal data for the font.         */

    unsigned short   bpp;            /* Bits per pixel.                     */

    FT_Memory        memory;

    bdf_property_t*  user_props;
    unsigned long    nuser_props;
    FT_HashRec       proptbl;

  } bdf_font_t;


  /**************************************************************************
   *
   * Types for load/save callbacks.
   *
   */


  /* Error codes. */
#define BDF_MISSING_START       -1
#define BDF_MISSING_FONTNAME    -2
#define BDF_MISSING_SIZE        -3
#define BDF_MISSING_CHARS       -4
#define BDF_MISSING_STARTCHAR   -5
#define BDF_MISSING_ENCODING    -6
#define BDF_MISSING_BBX         -7

#define BDF_OUT_OF_MEMORY      -20

#define BDF_INVALID_LINE      -100


  /**************************************************************************
   *
   * BDF font API.
   *
   */

  FT_LOCAL( FT_Error )
  bdf_load_font( FT_Stream       stream,
                 FT_Memory       memory,
                 bdf_options_t*  opts,
                 bdf_font_t*    *font );

  FT_LOCAL( void )
  bdf_free_font( bdf_font_t*  font );

  FT_LOCAL( bdf_property_t * )
  bdf_get_font_property( bdf_font_t*  font,
                         const char*  name );


FT_END_HEADER


#endif /* BDF_H_ */


/* END */
