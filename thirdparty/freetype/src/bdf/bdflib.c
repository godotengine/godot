/*
 * Copyright 2000 Computing Research Labs, New Mexico State University
 * Copyright 2001-2014
 *   Francesco Zappa Nardelli
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

  /**************************************************************************
   *
   * This file is based on bdf.c,v 1.22 2000/03/16 20:08:50
   *
   * taken from Mark Leisher's xmbdfed package
   *
   */



#include <freetype/freetype.h>
#include <freetype/internal/ftdebug.h>
#include <freetype/internal/ftstream.h>
#include <freetype/internal/ftobjs.h>

#include "bdf.h"
#include "bdferror.h"


  /**************************************************************************
   *
   * The macro FT_COMPONENT is used in trace mode.  It is an implicit
   * parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log
   * messages during execution.
   */
#undef  FT_COMPONENT
#define FT_COMPONENT  bdflib


  /**************************************************************************
   *
   * Builtin BDF font properties.
   *
   */

  /* List of most properties that might appear in a font.  Doesn't include */
  /* the RAW_* and AXIS_* properties in X11R6 polymorphic fonts.           */

  static const bdf_property_t  bdf_properties_[] =
  {
    { "ADD_STYLE_NAME",          BDF_ATOM,     1, { 0 } },
    { "AVERAGE_WIDTH",           BDF_INTEGER,  1, { 0 } },
    { "AVG_CAPITAL_WIDTH",       BDF_INTEGER,  1, { 0 } },
    { "AVG_LOWERCASE_WIDTH",     BDF_INTEGER,  1, { 0 } },
    { "CAP_HEIGHT",              BDF_INTEGER,  1, { 0 } },
    { "CHARSET_COLLECTIONS",     BDF_ATOM,     1, { 0 } },
    { "CHARSET_ENCODING",        BDF_ATOM,     1, { 0 } },
    { "CHARSET_REGISTRY",        BDF_ATOM,     1, { 0 } },
    { "COPYRIGHT",               BDF_ATOM,     1, { 0 } },
    { "DEFAULT_CHAR",            BDF_CARDINAL, 1, { 0 } },
    { "DESTINATION",             BDF_CARDINAL, 1, { 0 } },
    { "DEVICE_FONT_NAME",        BDF_ATOM,     1, { 0 } },
    { "END_SPACE",               BDF_INTEGER,  1, { 0 } },
    { "FACE_NAME",               BDF_ATOM,     1, { 0 } },
    { "FAMILY_NAME",             BDF_ATOM,     1, { 0 } },
    { "FIGURE_WIDTH",            BDF_INTEGER,  1, { 0 } },
    { "FONT",                    BDF_ATOM,     1, { 0 } },
    { "FONTNAME_REGISTRY",       BDF_ATOM,     1, { 0 } },
    { "FONT_ASCENT",             BDF_INTEGER,  1, { 0 } },
    { "FONT_DESCENT",            BDF_INTEGER,  1, { 0 } },
    { "FOUNDRY",                 BDF_ATOM,     1, { 0 } },
    { "FULL_NAME",               BDF_ATOM,     1, { 0 } },
    { "ITALIC_ANGLE",            BDF_INTEGER,  1, { 0 } },
    { "MAX_SPACE",               BDF_INTEGER,  1, { 0 } },
    { "MIN_SPACE",               BDF_INTEGER,  1, { 0 } },
    { "NORM_SPACE",              BDF_INTEGER,  1, { 0 } },
    { "NOTICE",                  BDF_ATOM,     1, { 0 } },
    { "PIXEL_SIZE",              BDF_INTEGER,  1, { 0 } },
    { "POINT_SIZE",              BDF_INTEGER,  1, { 0 } },
    { "QUAD_WIDTH",              BDF_INTEGER,  1, { 0 } },
    { "RAW_ASCENT",              BDF_INTEGER,  1, { 0 } },
    { "RAW_AVERAGE_WIDTH",       BDF_INTEGER,  1, { 0 } },
    { "RAW_AVG_CAPITAL_WIDTH",   BDF_INTEGER,  1, { 0 } },
    { "RAW_AVG_LOWERCASE_WIDTH", BDF_INTEGER,  1, { 0 } },
    { "RAW_CAP_HEIGHT",          BDF_INTEGER,  1, { 0 } },
    { "RAW_DESCENT",             BDF_INTEGER,  1, { 0 } },
    { "RAW_END_SPACE",           BDF_INTEGER,  1, { 0 } },
    { "RAW_FIGURE_WIDTH",        BDF_INTEGER,  1, { 0 } },
    { "RAW_MAX_SPACE",           BDF_INTEGER,  1, { 0 } },
    { "RAW_MIN_SPACE",           BDF_INTEGER,  1, { 0 } },
    { "RAW_NORM_SPACE",          BDF_INTEGER,  1, { 0 } },
    { "RAW_PIXEL_SIZE",          BDF_INTEGER,  1, { 0 } },
    { "RAW_POINT_SIZE",          BDF_INTEGER,  1, { 0 } },
    { "RAW_PIXELSIZE",           BDF_INTEGER,  1, { 0 } },
    { "RAW_POINTSIZE",           BDF_INTEGER,  1, { 0 } },
    { "RAW_QUAD_WIDTH",          BDF_INTEGER,  1, { 0 } },
    { "RAW_SMALL_CAP_SIZE",      BDF_INTEGER,  1, { 0 } },
    { "RAW_STRIKEOUT_ASCENT",    BDF_INTEGER,  1, { 0 } },
    { "RAW_STRIKEOUT_DESCENT",   BDF_INTEGER,  1, { 0 } },
    { "RAW_SUBSCRIPT_SIZE",      BDF_INTEGER,  1, { 0 } },
    { "RAW_SUBSCRIPT_X",         BDF_INTEGER,  1, { 0 } },
    { "RAW_SUBSCRIPT_Y",         BDF_INTEGER,  1, { 0 } },
    { "RAW_SUPERSCRIPT_SIZE",    BDF_INTEGER,  1, { 0 } },
    { "RAW_SUPERSCRIPT_X",       BDF_INTEGER,  1, { 0 } },
    { "RAW_SUPERSCRIPT_Y",       BDF_INTEGER,  1, { 0 } },
    { "RAW_UNDERLINE_POSITION",  BDF_INTEGER,  1, { 0 } },
    { "RAW_UNDERLINE_THICKNESS", BDF_INTEGER,  1, { 0 } },
    { "RAW_X_HEIGHT",            BDF_INTEGER,  1, { 0 } },
    { "RELATIVE_SETWIDTH",       BDF_CARDINAL, 1, { 0 } },
    { "RELATIVE_WEIGHT",         BDF_CARDINAL, 1, { 0 } },
    { "RESOLUTION",              BDF_INTEGER,  1, { 0 } },
    { "RESOLUTION_X",            BDF_CARDINAL, 1, { 0 } },
    { "RESOLUTION_Y",            BDF_CARDINAL, 1, { 0 } },
    { "SETWIDTH_NAME",           BDF_ATOM,     1, { 0 } },
    { "SLANT",                   BDF_ATOM,     1, { 0 } },
    { "SMALL_CAP_SIZE",          BDF_INTEGER,  1, { 0 } },
    { "SPACING",                 BDF_ATOM,     1, { 0 } },
    { "STRIKEOUT_ASCENT",        BDF_INTEGER,  1, { 0 } },
    { "STRIKEOUT_DESCENT",       BDF_INTEGER,  1, { 0 } },
    { "SUBSCRIPT_SIZE",          BDF_INTEGER,  1, { 0 } },
    { "SUBSCRIPT_X",             BDF_INTEGER,  1, { 0 } },
    { "SUBSCRIPT_Y",             BDF_INTEGER,  1, { 0 } },
    { "SUPERSCRIPT_SIZE",        BDF_INTEGER,  1, { 0 } },
    { "SUPERSCRIPT_X",           BDF_INTEGER,  1, { 0 } },
    { "SUPERSCRIPT_Y",           BDF_INTEGER,  1, { 0 } },
    { "UNDERLINE_POSITION",      BDF_INTEGER,  1, { 0 } },
    { "UNDERLINE_THICKNESS",     BDF_INTEGER,  1, { 0 } },
    { "WEIGHT",                  BDF_CARDINAL, 1, { 0 } },
    { "WEIGHT_NAME",             BDF_ATOM,     1, { 0 } },
    { "X_HEIGHT",                BDF_INTEGER,  1, { 0 } },
    { "_MULE_BASELINE_OFFSET",   BDF_INTEGER,  1, { 0 } },
    { "_MULE_RELATIVE_COMPOSE",  BDF_INTEGER,  1, { 0 } },
  };

  static const unsigned long
  num_bdf_properties_ = sizeof ( bdf_properties_ ) /
                        sizeof ( bdf_properties_[0] );

  /* Auto correction messages. */
#define ACMSG1   "FONT_ASCENT property missing.  " \
                 "Added `FONT_ASCENT %hd'.\n"
#define ACMSG2   "FONT_DESCENT property missing.  " \
                 "Added `FONT_DESCENT %hd'.\n"
#define ACMSG3   "Font width != actual width.  Old: %d New: %d.\n"
#define ACMSG4   "Font left bearing != actual left bearing.  " \
                 "Old: %hd New: %hd.\n"
#define ACMSG5   "Font ascent != actual ascent.  Old: %hd New: %hd.\n"
#define ACMSG6   "Font descent != actual descent.  Old: %d New: %d.\n"
#define ACMSG7   "Font height != actual height. Old: %d New: %d.\n"
#define ACMSG8   "Glyph scalable width (SWIDTH) adjustments made.\n"
#define ACMSG9   "SWIDTH field missing at line %lu.  Set automatically.\n"
#define ACMSG10  "DWIDTH field missing at line %lu.  Set to glyph width.\n"
#define ACMSG11  "SIZE bits per pixel field adjusted to %hd.\n"
#define ACMSG13  "Glyph %lu extra rows removed.\n"
#define ACMSG14  "Glyph %lu extra columns removed.\n"
#define ACMSG15  "Incorrect glyph count: %lu indicated but %lu found.\n"
#define ACMSG16  "Glyph %lu missing columns padded with zero bits.\n"
#define ACMSG17  "Adjusting number of glyphs to %lu.\n"

  /* Error messages. */
#define ERRMSG1  "[line %lu] Missing `%s' line.\n"
#define ERRMSG2  "[line %lu] Font header corrupted or missing fields.\n"
#define ERRMSG3  "[line %lu] Font glyphs corrupted or missing fields.\n"
#define ERRMSG4  "[line %lu] BBX too big.\n"
#define ERRMSG5  "[line %lu] `%s' value too big.\n"
#define ERRMSG6  "[line %lu] Input line too long.\n"
#define ERRMSG7  "[line %lu] Font name too long.\n"
#define ERRMSG8  "[line %lu] Invalid `%s' value.\n"
#define ERRMSG9  "[line %lu] Invalid keyword.\n"

  /* Debug messages. */
#define DBGMSG1  "  [%6lu] %s" /* no \n */
#define DBGMSG2  " (0x%lX)\n"


  /**************************************************************************
   *
   * Utility types and functions.
   *
   */


  /* Structure used while loading BDF fonts. */

  typedef struct  bdf_parse_t__
  {
    unsigned long   flags;
    unsigned long   cnt;
    unsigned long   row;

    short           minlb;
    short           maxlb;
    short           maxrb;
    short           maxas;
    short           maxds;

    short           rbearing;

    char*           glyph_name;
    long            glyph_enc;

    bdf_glyph_t*    glyph;
    bdf_font_t*     font;

    FT_Memory       memory;
    unsigned long   size;        /* the stream size */

  } bdf_parse_t_;


  /* Function type for parsing lines of a BDF font. */

  typedef FT_Error
  (*bdf_line_func_t_)( char*          line,
                       unsigned long  linelen,
                       unsigned long  lineno,
                       bdf_parse_t_*  p,
                       void*          next );


#define setsbit( m, cc ) \
          ( m[(FT_Byte)(cc) >> 3] |= (FT_Byte)( 1 << ( (cc) & 7 ) ) )
#define sbitset( m, cc ) \
          ( m[(FT_Byte)(cc) >> 3]  & ( 1 << ( (cc) & 7 ) ) )


  static char*
  bdf_strtok_( char*  line,
               int    delim )
  {
    while ( *line && *line != delim )
      line++;

    if ( *line )
      *line++ = '\0';

    while ( *line && *line == delim )
      line++;

    return line;
  }


  /* XXX: make this work with EBCDIC also */

  static const unsigned char  a2i[128] =
  {
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
    0x08, 0x09, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
  };

  static const unsigned char  ddigits[32] =
  {
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0x03,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  };


  /* Routine to convert a decimal ASCII string to an unsigned long integer. */
  static unsigned long
  bdf_atoul_( const char*  s )
  {
    unsigned long  v;


    for ( v = 0; sbitset( ddigits, *s ); s++ )
    {
      if ( v < ( FT_ULONG_MAX - 9 ) / 10 )
        v = v * 10 + a2i[(int)*s];
      else
      {
        v = FT_ULONG_MAX;
        break;
      }
    }

    return v;
  }


  /* Routine to convert a decimal ASCII string to a signed long integer. */
  static long
  bdf_atol_( const char*  s )
  {
    long  v, neg;


    /* Check for a minus sign. */
    if ( *s == '-' )
    {
      s++;
      neg = -1;
    }
    else
      neg =  1;

    for ( v = 0; sbitset( ddigits, *s ); s++ )
    {
      if ( v < ( FT_LONG_MAX - 9 ) / 10 )
        v = v * 10 + a2i[(int)*s];
      else
      {
        v = FT_LONG_MAX;
        break;
      }
    }

    return neg * v;
  }


  /* Routine to convert a decimal ASCII string to an unsigned short integer. */
  static unsigned short
  bdf_atous_( const char*  s )
  {
    unsigned short  v;


    for ( v = 0; sbitset( ddigits, *s ); s++ )
    {
      if ( v < ( FT_USHORT_MAX - 9 ) / 10 )
        v = (unsigned short)( v * 10 + a2i[(int)*s] );
      else
      {
        v = FT_USHORT_MAX;
        break;
      }
    }

    return v;
  }


  /* Routine to convert a decimal ASCII string to a signed short integer. */
  static short
  bdf_atos_( const char*  s )
  {
    short  v, neg;


    /* Check for a minus. */
    if ( *s == '-' )
    {
      s++;
      neg = -1;
    }
    else
      neg =  1;

    for ( v = 0; sbitset( ddigits, *s ); s++ )
    {
      if ( v < ( SHRT_MAX - 9 ) / 10 )
        v = (short)( v * 10 + a2i[(int)*s] );
      else
      {
        v = SHRT_MAX;
        break;
      }
    }

    return neg * v;
  }


  /* Routine to compare two glyphs by encoding so they can be sorted. */
  FT_COMPARE_DEF( int )
  by_encoding( const void*  a,
               const void*  b )
  {
    bdf_glyph_t  *c1, *c2;


    c1 = (bdf_glyph_t *)a;
    c2 = (bdf_glyph_t *)b;

    if ( c1->encoding < c2->encoding )
      return -1;

    if ( c1->encoding > c2->encoding )
      return 1;

    return 0;
  }


  static FT_Error
  bdf_create_property( const char*  name,
                       int          format,
                       bdf_font_t*  font )
  {
    size_t           n;
    bdf_property_t*  p;
    FT_Memory        memory = font->memory;
    FT_Error         error  = FT_Err_Ok;


    /* First check whether the property has        */
    /* already been added or not.  If it has, then */
    /* simply ignore it.                           */
    if ( ft_hash_str_lookup( name, &(font->proptbl) ) )
      goto Exit;

    if ( FT_QRENEW_ARRAY( font->user_props,
                          font->nuser_props,
                          font->nuser_props + 1 ) )
      goto Exit;

    p = font->user_props + font->nuser_props;

    if ( FT_STRDUP( p->name, name ) )
      goto Exit;

    p->format     = format;
    p->builtin    = 0;
    p->value.atom = NULL;  /* nothing is ever stored here */

    n = num_bdf_properties_ + font->nuser_props;

    error = ft_hash_str_insert( p->name, n, &(font->proptbl), memory );
    if ( error )
      goto Exit;

    font->nuser_props++;

  Exit:
    return error;
  }


  static bdf_property_t*
  bdf_get_property( const char*  name,
                    bdf_font_t*  font )
  {
    size_t*  propid;


    if ( name == NULL || *name == 0 )
      return NULL;

    if ( ( propid = ft_hash_str_lookup( name, &(font->proptbl) ) ) == NULL )
      return NULL;

    if ( *propid >= num_bdf_properties_ )
      return font->user_props + ( *propid - num_bdf_properties_ );

    return (bdf_property_t*)bdf_properties_ + *propid;
  }


  /**************************************************************************
   *
   * BDF font file parsing flags and functions.
   *
   */


  /* Parse flags. */

#define BDF_START_      0x0001U
#define BDF_FONT_NAME_  0x0002U
#define BDF_SIZE_       0x0004U
#define BDF_FONT_BBX_   0x0008U
#define BDF_PROPS_      0x0010U
#define BDF_GLYPHS_     0x0020U
#define BDF_GLYPH_      0x0040U
#define BDF_ENCODING_   0x0080U
#define BDF_SWIDTH_     0x0100U
#define BDF_DWIDTH_     0x0200U
#define BDF_BBX_        0x0400U
#define BDF_BITMAP_     0x0800U

#define BDF_GLYPH_BITS_ ( BDF_GLYPH_    | \
                          BDF_ENCODING_ | \
                          BDF_SWIDTH_   | \
                          BDF_DWIDTH_   | \
                          BDF_BBX_      | \
                          BDF_BITMAP_   )


  static FT_Error
  bdf_add_comment_( bdf_font_t*    font,
                    const char*    comment,
                    unsigned long  len )
  {
    char*      cp;
    FT_Memory  memory = font->memory;
    FT_Error   error  = FT_Err_Ok;


    /* Skip keyword COMMENT. */
    comment += 7;
    len     -= 7;

    if ( len == 0 )
      goto Exit;

    if ( FT_QRENEW_ARRAY( font->comments,
                          font->comments_len,
                          font->comments_len + len + 1 ) )
      goto Exit;

    cp = font->comments + font->comments_len;

    FT_MEM_COPY( cp, comment, len );
    cp[len] = '\0';

    font->comments_len += len + 1;

  Exit:
    return error;
  }


  /* Determine whether the property is an atom or not.  If it is, then */
  /* clean it up so the double quotes are removed if they exist.       */
  static int
  bdf_is_atom_( char*          line,
                unsigned long  linelen,
                char**         name,
                char**         value,
                bdf_font_t*    font )
  {
    int              hold;
    char             *sp, *ep;
    bdf_property_t*  p;


    sp = ep = line;

    while ( *ep && *ep != ' ' )
      ep++;

    hold = *ep;
    *ep  = '\0';

    p = bdf_get_property( sp, font );

    /* If the property exists and is not an atom, just return here. */
    if ( p && p->format != BDF_ATOM )
    {
      *ep = (char)hold;  /* Undo NUL-termination. */
      return 0;
    }

    *name = sp;

    /* The property is an atom.  Trim all leading and trailing whitespace */
    /* and double quotes for the atom value.                              */
    sp = ep;
    ep = line + linelen;

    /* Trim the leading whitespace if it exists. */
    if ( sp < ep )
      do
         sp++;
      while ( *sp == ' ' );

    /* Trim the leading double quote if it exists. */
    if ( *sp == '"' )
      sp++;

    *value = sp;

    /* Trim the trailing whitespace if it exists. */
    if ( sp < ep )
      do
        *ep-- = '\0';
      while ( *ep == ' ' );

    /* Trim the trailing double quote if it exists. */
    if ( *ep  == '"' )
      *ep = '\0';

    return 1;
  }


  static FT_Error
  bdf_add_property_( bdf_font_t*    font,
                     const char*    name,
                     char*          value,
                     unsigned long  lineno )
  {
    size_t*         propid;
    bdf_property_t  *prop, *fp;
    FT_Memory       memory = font->memory;
    FT_Error        error  = FT_Err_Ok;

    FT_UNUSED( lineno );        /* only used in debug mode */


    /* First, check whether the property already exists in the font. */
    if ( ( propid = ft_hash_str_lookup( name, font->internal ) ) != NULL )
    {
      /* The property already exists in the font, so simply replace */
      /* the value of the property with the current value.          */
      fp = font->props + *propid;

      switch ( fp->format )
      {
      case BDF_ATOM:
        /* Delete the current atom if it exists. */
        FT_FREE( fp->value.atom );

        if ( value && value[0] != 0 )
        {
          if ( FT_STRDUP( fp->value.atom, value ) )
            goto Exit;
        }
        break;

      case BDF_INTEGER:
        fp->value.l = bdf_atol_( value );
        break;

      case BDF_CARDINAL:
        fp->value.ul = bdf_atoul_( value );
        break;

      default:
        ;
      }

      goto Exit;
    }

    /* See whether this property type exists yet or not. */
    /* If not, create it.                                */
    propid = ft_hash_str_lookup( name, &(font->proptbl) );
    if ( !propid )
    {
      error = bdf_create_property( name, BDF_ATOM, font );
      if ( error )
        goto Exit;
      propid = ft_hash_str_lookup( name, &(font->proptbl) );
    }

    /* Allocate another property if this is overflowing. */
    if ( font->props_used == font->props_size )
    {
      if ( FT_QRENEW_ARRAY( font->props,
                            font->props_size,
                            font->props_size + 1 ) )
        goto Exit;

      font->props_size++;
    }

    if ( *propid >= num_bdf_properties_ )
      prop = font->user_props + ( *propid - num_bdf_properties_ );
    else
      prop = (bdf_property_t*)bdf_properties_ + *propid;

    fp = font->props + font->props_used;

    fp->name    = prop->name;
    fp->format  = prop->format;
    fp->builtin = prop->builtin;

    switch ( prop->format )
    {
    case BDF_ATOM:
      fp->value.atom = NULL;
      if ( value && value[0] )
      {
        if ( FT_STRDUP( fp->value.atom, value ) )
          goto Exit;
      }
      break;

    case BDF_INTEGER:
      fp->value.l = bdf_atol_( value );
      break;

    case BDF_CARDINAL:
      fp->value.ul = bdf_atoul_( value );
      break;
    }

    /* Add the property to the font property table. */
    error = ft_hash_str_insert( fp->name,
                                font->props_used,
                                font->internal,
                                memory );
    if ( error )
      goto Exit;

    font->props_used++;

  Exit:
    return error;
  }


  static FT_Error
  bdf_parse_end_( char*          line,
                  unsigned long  linelen,
                  unsigned long  lineno,
                  bdf_parse_t_*  p,
                  void*          next )
  {
    /* a no-op; we ignore everything after `ENDFONT' */

    FT_UNUSED( line );
    FT_UNUSED( linelen );
    FT_UNUSED( lineno );
    FT_UNUSED( p );
    FT_UNUSED( next );

    return FT_Err_Ok;
  }


  /* Line function prototypes. */
  static FT_Error
  bdf_parse_start_( char*          line,
                    unsigned long  linelen,
                    unsigned long  lineno,
                    bdf_parse_t_*  p,
                    void*          next );


  static FT_Error
  bdf_parse_glyphs_( char*          line,
                     unsigned long  linelen,
                     unsigned long  lineno,
                     bdf_parse_t_*  p,
                     void*          next );


  /* Aggressively parse the glyph bitmaps. */
  static FT_Error
  bdf_parse_bitmap_( char*          line,
                     unsigned long  linelen,
                     unsigned long  lineno,
                     bdf_parse_t_*  p,
                     void*          next )
  {
    bdf_glyph_t*    glyph = p->glyph;
    unsigned char*  bp;
    unsigned long   i, nibbles;
    int             x;

    FT_UNUSED( lineno );        /* only used in debug mode */


    nibbles = glyph->bpr << 1;
    bp      = glyph->bitmap + p->row * glyph->bpr;

    if ( nibbles > linelen )
    {
      FT_TRACE2(( "bdf_parse_bitmap_: " ACMSG16, glyph->encoding ));
      nibbles = linelen;
    }

    for ( i = 0; i < nibbles; i++ )
    {
      /* char to hex without checks */
      x  = line[i];
      x += 9 * ( x & 0x40 ) >> 6;  /* for [A-Fa-f] */
      x &= 0x0F;

      if ( i & 1 )
        *bp++ |= x;
      else
        *bp = (unsigned char)( x << 4 );
    }

    p->row++;

    /* When done, go back to parsing glyphs */
    if ( p->row >= (unsigned long)glyph->bbx.height )
      *(bdf_line_func_t_*)next = bdf_parse_glyphs_;

    return FT_Err_Ok;
  }


  /* Actually parse the glyph info. */
  static FT_Error
  bdf_parse_glyphs_( char*          line,
                     unsigned long  linelen,
                     unsigned long  lineno,
                     bdf_parse_t_*  p,
                     void*          next )
  {
    bdf_font_t*   font   = p->font;
    bdf_glyph_t*  glyph;
    FT_Memory     memory = font->memory;
    FT_Error      error  = FT_Err_Ok;

    FT_UNUSED( lineno );        /* only used in debug mode */


    /* Check for a comment. */
    if ( ft_strncmp( line, "COMMENT", 7 ) == 0 )
    {
      if ( p->flags & BDF_KEEP_COMMENTS )
        error = bdf_add_comment_( font, line, linelen );

      goto Exit;
    }

    /* Check for the ENDFONT field. */
    if ( ft_strncmp( line, "ENDFONT", 7 ) == 0 )
    {
      if ( p->flags & BDF_GLYPH_BITS_ )
      {
        /* Missing ENDCHAR field. */
        FT_ERROR(( "bdf_parse_glyphs_: " ERRMSG1, lineno, "ENDCHAR" ));
        error = FT_THROW( Corrupted_Font_Glyphs );
        goto Exit;
      }

      /* Sort the glyphs by encoding. */
      ft_qsort( (char *)font->glyphs,
                font->glyphs_used,
                sizeof ( bdf_glyph_t ),
                by_encoding );

      p->flags &= ~BDF_START_;

      *(bdf_line_func_t_*)next = bdf_parse_end_;

      goto Exit;
    }

    /* Check for the ENDCHAR field. */
    if ( ft_strncmp( line, "ENDCHAR", 7 ) == 0 )
    {
      /* Free unused glyph_name */
      FT_FREE( p->glyph_name );

      p->glyph_enc = 0;
      p->flags    &= ~BDF_GLYPH_BITS_;

      goto Exit;
    }

    /* Check whether a glyph is being scanned but should be */
    /* ignored because it is an unencoded glyph.            */
    if ( p->flags & BDF_GLYPH_              &&
         p->glyph_enc            == -1      &&
         !( p->flags & BDF_KEEP_UNENCODED ) )
      goto Exit;

    /* Check for the STARTCHAR field. */
    if ( ft_strncmp( line, "STARTCHAR ", 10 ) == 0 )
    {
      if ( p->flags & BDF_GLYPH_BITS_ )
      {
        /* Missing ENDCHAR field. */
        FT_ERROR(( "bdf_parse_glyphs_: " ERRMSG1, lineno, "ENDCHAR" ));
        error = FT_THROW( Missing_Startchar_Field );
        goto Exit;
      }

      line = bdf_strtok_( line, ' ' );

      if ( FT_STRDUP( p->glyph_name, line ) )
        goto Exit;

      p->flags |= BDF_GLYPH_;

      FT_TRACE4(( DBGMSG1, lineno, line ));

      goto Exit;
    }

    /* Check for the ENCODING field. */
    if ( ft_strncmp( line, "ENCODING ", 9 ) == 0 )
    {
      if ( !( p->flags & BDF_GLYPH_ ) )
      {
        /* Missing STARTCHAR field. */
        FT_ERROR(( "bdf_parse_glyphs_: " ERRMSG1, lineno, "STARTCHAR" ));
        error = FT_THROW( Missing_Startchar_Field );
        goto Exit;
      }

      line = bdf_strtok_( line, ' ' );

      p->glyph_enc = bdf_atol_( line );

      /* Normalize negative encoding values.  The specification only */
      /* allows -1, but we can be more generous here.                */
      if ( p->glyph_enc < -1 )
        p->glyph_enc = -1;

      /* Check for alternative encoding format. */
      line = bdf_strtok_( line, ' ' );

      if ( p->glyph_enc == -1 && *line )
        p->glyph_enc = bdf_atol_( line );

      if ( p->glyph_enc < -1 || p->glyph_enc >= 0x110000L )
        p->glyph_enc = -1;

      FT_TRACE4(( DBGMSG2, p->glyph_enc ));

      if ( p->glyph_enc >= 0 )
      {
        /* Make sure there are enough glyphs allocated in case the */
        /* number of characters happen to be wrong.                */
        if ( font->glyphs_used == font->glyphs_size )
        {
          if ( FT_RENEW_ARRAY( font->glyphs,
                               font->glyphs_size,
                               font->glyphs_size + 64 ) )
            goto Exit;

          font->glyphs_size += 64;
        }

        glyph           = font->glyphs + font->glyphs_used++;
        glyph->name     = p->glyph_name;
        glyph->encoding = (unsigned long)p->glyph_enc;
      }
      else if ( p->flags & BDF_KEEP_UNENCODED )
      {
        /* Allocate the next unencoded glyph. */
        if ( font->unencoded_used == font->unencoded_size )
        {
          if ( FT_RENEW_ARRAY( font->unencoded ,
                               font->unencoded_size,
                               font->unencoded_size + 4 ) )
            goto Exit;

          font->unencoded_size += 4;
        }

        glyph           = font->unencoded + font->unencoded_used;
        glyph->name     = p->glyph_name;
        glyph->encoding = font->unencoded_used++;
      }
      else
      {
        /* Free up the glyph name if the unencoded shouldn't be */
        /* kept.                                                */
        FT_FREE( p->glyph_name );
        glyph = NULL;
      }

      p->glyph_name = NULL;
      p->glyph      = glyph;
      p->flags     |= BDF_ENCODING_;

      goto Exit;
    }

    if ( !( p->flags & BDF_ENCODING_ ) )
      goto Missing_Encoding;

    /* Point at the glyph being constructed. */
    glyph = p->glyph;

    /* Expect the SWIDTH (scalable width) field next. */
    if ( ft_strncmp( line, "SWIDTH ", 7 ) == 0 )
    {
      line          = bdf_strtok_( line, ' ' );
      glyph->swidth = bdf_atous_( line );

      p->flags |= BDF_SWIDTH_;
      goto Exit;
    }

    /* Expect the DWIDTH (device width) field next. */
    if ( ft_strncmp( line, "DWIDTH ", 7 ) == 0 )
    {
      line          = bdf_strtok_( line, ' ' );
      glyph->dwidth = bdf_atous_( line );

      if ( !( p->flags & BDF_SWIDTH_ ) )
      {
        /* Missing SWIDTH field.  Emit an auto correction message and set */
        /* the scalable width from the device width.                      */
        FT_TRACE2(( "bdf_parse_glyphs_: " ACMSG9, lineno ));

        glyph->swidth = (unsigned short)FT_MulDiv(
                          glyph->dwidth, 72000L,
                          (FT_Long)( font->point_size *
                                     font->resolution_x ) );
      }

      p->flags |= BDF_DWIDTH_;
      goto Exit;
    }

    /* Do not leak the bitmap or reset its size */
    if ( p->flags & BDF_BITMAP_ )
      goto Exit;

    /* Expect the BBX field next. */
    if ( ft_strncmp( line, "BBX ", 4 ) == 0 )
    {
      line                = bdf_strtok_( line, ' ' );
      glyph->bbx.width    = bdf_atous_( line );
      line                = bdf_strtok_( line, ' ' );
      glyph->bbx.height   = bdf_atous_( line );
      line                = bdf_strtok_( line, ' ' );
      glyph->bbx.x_offset = bdf_atos_( line );
      line                = bdf_strtok_( line, ' ' );
      glyph->bbx.y_offset = bdf_atos_( line );

      /* Generate the ascent and descent of the character. */
      glyph->bbx.ascent  = (short)( glyph->bbx.height + glyph->bbx.y_offset );
      glyph->bbx.descent = (short)( -glyph->bbx.y_offset );

      /* Determine the overall font bounding box as the characters are */
      /* loaded so corrections can be done later if indicated.         */
      p->maxas    = (short)FT_MAX( glyph->bbx.ascent, p->maxas );
      p->maxds    = (short)FT_MAX( glyph->bbx.descent, p->maxds );

      p->rbearing = (short)( glyph->bbx.width + glyph->bbx.x_offset );

      p->maxrb    = (short)FT_MAX( p->rbearing, p->maxrb );
      p->minlb    = (short)FT_MIN( glyph->bbx.x_offset, p->minlb );
      p->maxlb    = (short)FT_MAX( glyph->bbx.x_offset, p->maxlb );

      if ( !( p->flags & BDF_DWIDTH_ ) )
      {
        /* Missing DWIDTH field.  Emit an auto correction message and set */
        /* the device width to the glyph width.                           */
        FT_TRACE2(( "bdf_parse_glyphs_: " ACMSG10, lineno ));
        glyph->dwidth = glyph->bbx.width;
      }

      /* If the BDF_CORRECT_METRICS flag is set, then adjust the SWIDTH */
      /* value if necessary.                                            */
      if ( p->flags & BDF_CORRECT_METRICS )
      {
        /* Determine the point size of the glyph. */
        unsigned short  sw = (unsigned short)FT_MulDiv(
                               glyph->dwidth, 72000L,
                               (FT_Long)( font->point_size *
                                          font->resolution_x ) );


        if ( sw != glyph->swidth )
        {
          glyph->swidth = sw;

          FT_TRACE2(( "bdf_parse_glyphs_: " ACMSG8 ));
        }
      }

      p->flags |= BDF_BBX_;
      goto Exit;
    }

    /* And finally, gather up the bitmap. */
    if ( ft_strncmp( line, "BITMAP", 6 ) == 0 )
    {
      unsigned long  bitmap_size;


      if ( !( p->flags & BDF_BBX_ ) )
      {
        /* Missing BBX field. */
        FT_ERROR(( "bdf_parse_glyphs_: " ERRMSG1, lineno, "BBX" ));
        error = FT_THROW( Missing_Bbx_Field );
        goto Exit;
      }

      /* Allocate enough space for the bitmap. */
      glyph->bpr = ( glyph->bbx.width * p->font->bpp + 7 ) >> 3;

      bitmap_size = glyph->bpr * glyph->bbx.height;
      if ( glyph->bpr > 0xFFFFU || bitmap_size > 0xFFFFU )
      {
        FT_ERROR(( "bdf_parse_glyphs_: " ERRMSG4, lineno ));
        error = FT_THROW( Bbx_Too_Big );
        goto Exit;
      }
      else
        glyph->bytes = (unsigned short)bitmap_size;

      if ( !bitmap_size || FT_ALLOC( glyph->bitmap, glyph->bytes ) )
        goto Exit;

      p->row    = 0;
      p->flags |= BDF_BITMAP_;
      *(bdf_line_func_t_*)next = bdf_parse_bitmap_;

      goto Exit;
    }

    FT_ERROR(( "bdf_parse_glyphs_: " ERRMSG9, lineno ));
    error = FT_THROW( Invalid_File_Format );
    goto Exit;

  Missing_Encoding:
    /* Missing ENCODING field. */
    FT_ERROR(( "bdf_parse_glyphs_: " ERRMSG1, lineno, "ENCODING" ));
    error = FT_THROW( Missing_Encoding_Field );

  Exit:
    if ( error && ( p->flags & BDF_GLYPH_ ) )
      FT_FREE( p->glyph_name );

    return error;
  }


  /* Load the font properties. */
  static FT_Error
  bdf_parse_properties_( char*          line,
                         unsigned long  linelen,
                         unsigned long  lineno,
                         bdf_parse_t_*  p,
                         void*          next )
  {
    bdf_font_t*  font  = p->font;
    FT_Error     error = FT_Err_Ok;
    char*        name;
    char*        value;

    FT_UNUSED( lineno );


    /* Check for a comment. */
    if ( ft_strncmp( line, "COMMENT", 7 ) == 0 )
    {
      if ( p->flags & BDF_KEEP_COMMENTS )
        error = bdf_add_comment_( font, line, linelen );

      goto Exit;
    }

    /* Check for the end of the properties. */
    if ( ft_strncmp( line, "ENDPROPERTIES", 13 ) == 0 )
    {
      *(bdf_line_func_t_*)next = bdf_parse_start_;

      goto Exit;
    }

    /* Ignore the _XFREE86_GLYPH_RANGES properties. */
    if ( ft_strncmp( line, "_XFREE86_GLYPH_RANGES", 21 ) == 0 )
      goto Exit;

    if ( bdf_is_atom_( line, linelen, &name, &value, p->font ) )
    {
      error = bdf_add_property_( font, name, value, lineno );
      if ( error )
        goto Exit;
    }
    else
    {
      value = bdf_strtok_( line, ' ' );

      error = bdf_add_property_( font, line, value, lineno );
      if ( error )
        goto Exit;
    }

  Exit:
    return error;
  }


  /* Load the font header. */
  static FT_Error
  bdf_parse_start_( char*          line,
                    unsigned long  linelen,
                    unsigned long  lineno,
                    bdf_parse_t_*  p,
                    void*          next )
  {
    bdf_font_t*  font;
    FT_Memory    memory = p->memory;
    FT_Error     error  = FT_Err_Ok;

    FT_UNUSED( lineno );            /* only used in debug mode */


    /* The first line must be STARTFONT.       */
    /* Otherwise, reject the font immediately. */
    if ( !( p->flags & BDF_START_ ) )
    {
      if ( ft_strncmp( line, "STARTFONT", 9 ) != 0 )
      {
        error = FT_THROW( Missing_Startfont_Field );
        goto Exit;
      }

      p->flags |= BDF_START_;

      if ( FT_NEW( p->font ) )
        goto Exit;

      p->font->memory = memory;

      goto Exit;
    }

    /* Point at the font being constructed. */
    font = p->font;

    /* Check for a comment. */
    if ( ft_strncmp( line, "COMMENT", 7 ) == 0 )
    {
      if ( p->flags & BDF_KEEP_COMMENTS )
        error = bdf_add_comment_( font, line, linelen );

      goto Exit;
    }

    /* Check for the start of the properties. */
    if ( !( p->flags & BDF_PROPS_ )                       &&
         ft_strncmp( line, "STARTPROPERTIES ", 16 ) == 0 )
    {
      line             = bdf_strtok_( line, ' ' );
      font->props_size = bdf_atoul_( line );

      if ( font->props_size < 2 )
        font->props_size = 2;

      /* We need at least 4 bytes per property. */
      if ( font->props_size > p->size / 4 )
      {
        font->props_size = 0;

        FT_ERROR(( "bdf_parse_start_: " ERRMSG5, lineno, "STARTPROPERTIES" ));
        error = FT_THROW( Invalid_Argument );
        goto Exit;
      }

      if ( FT_NEW_ARRAY( font->props, font->props_size ) )
      {
        font->props_size = 0;
        goto Exit;
      }

      if ( FT_QNEW( font->internal ) )
        goto Exit;
      error = ft_hash_str_init( font->internal, memory );
      if ( error )
        goto Exit;

      /* preset common properties */
      {
        bdf_property_t*  prop    = (bdf_property_t*)bdf_properties_;
        FT_Hash          proptbl = &font->proptbl;
        size_t           i;


        error = ft_hash_str_init( proptbl, memory );
        if ( error )
          goto Exit;
        for ( i = 0; i < num_bdf_properties_; i++, prop++ )
        {
          error = ft_hash_str_insert( prop->name, i, proptbl, memory );
          if ( error )
            goto Exit;
        }
      }

      p->flags |= BDF_PROPS_;

      *(bdf_line_func_t_*)next = bdf_parse_properties_;

      goto Exit;
    }

    /* Check for the FONTBOUNDINGBOX field. */
    if ( ft_strncmp( line, "FONTBOUNDINGBOX ", 16 ) == 0 )
    {
      line               = bdf_strtok_( line, ' ' );
      font->bbx.width    = bdf_atous_( line );
      line               = bdf_strtok_( line, ' ' );
      font->bbx.height   = bdf_atous_( line );
      line               = bdf_strtok_( line, ' ' );
      font->bbx.x_offset = bdf_atos_( line );
      line               = bdf_strtok_( line, ' ' );
      font->bbx.y_offset = bdf_atos_( line );

      font->bbx.ascent  = (short)( font->bbx.height +
                                      font->bbx.y_offset );

      font->bbx.descent = (short)( -font->bbx.y_offset );

      p->flags |= BDF_FONT_BBX_;

      goto Exit;
    }

    /* The next thing to check for is the FONT field. */
    if ( ft_strncmp( line, "FONT ", 5 ) == 0 )
    {
      int  i;


      line = bdf_strtok_( line, ' ' );

      /* Allowing multiple `FONT' lines (which is invalid) doesn't hurt... */
      FT_FREE( font->name );

      if ( FT_STRDUP( font->name, line ) )
        goto Exit;

      /* If the font name is an XLFD name, set the spacing to the one in */
      /* the font name after the 11th dash.                              */
      for ( i = 0; i < 11; i++ )
      {
        while ( *line && *line != '-' )
          line++;
        if ( *line )
          line++;
      }

      switch ( *line )
      {
      case 'C':
      case 'c':
        font->spacing = BDF_CHARCELL;
        break;
      case 'M':
      case 'm':
        font->spacing = BDF_MONOWIDTH;
        break;
      case 'P':
      case 'p':
      default:
        font->spacing = BDF_PROPORTIONAL;
        break;
      }

      p->flags |= BDF_FONT_NAME_;

      goto Exit;
    }

    /* Check for the SIZE field. */
    if ( ft_strncmp( line, "SIZE ", 5 ) == 0 )
    {
      line               = bdf_strtok_( line, ' ' );
      font->point_size   = bdf_atoul_( line );
      line               = bdf_strtok_( line, ' ' );
      font->resolution_x = bdf_atoul_( line );
      line               = bdf_strtok_( line, ' ' );
      font->resolution_y = bdf_atoul_( line );

      /* Check for the bits per pixel field. */
      line = bdf_strtok_( line, ' ' );
      if ( *line )
      {
        unsigned short bpp;


        bpp = bdf_atous_( line );

        /* Only values 1, 2, 4, 8 are allowed for greymap fonts. */
        if ( bpp > 4 )
          font->bpp = 8;
        else if ( bpp > 2 )
          font->bpp = 4;
        else if ( bpp > 1 )
          font->bpp = 2;
        else
          font->bpp = 1;

        if ( font->bpp != bpp )
          FT_TRACE2(( "bdf_parse_start_: " ACMSG11, font->bpp ));
      }
      else
        font->bpp = 1;

      p->flags |= BDF_SIZE_;

      goto Exit;
    }

    /* Check for the CHARS field */
    if ( ft_strncmp( line, "CHARS ", 6 ) == 0 )
    {
      /* Check the header for completeness before parsing glyphs. */
      if ( !( p->flags & BDF_FONT_NAME_ ) )
      {
        /* Missing the FONT field. */
        FT_ERROR(( "bdf_parse_start_: " ERRMSG1, lineno, "FONT" ));
        error = FT_THROW( Missing_Font_Field );
        goto Exit;
      }
      if ( !( p->flags & BDF_SIZE_ ) )
      {
        /* Missing the SIZE field. */
        FT_ERROR(( "bdf_parse_start_: " ERRMSG1, lineno, "SIZE" ));
        error = FT_THROW( Missing_Size_Field );
        goto Exit;
      }
      if ( !( p->flags & BDF_FONT_BBX_ ) )
      {
        /* Missing the FONTBOUNDINGBOX field. */
        FT_ERROR(( "bdf_parse_start_: " ERRMSG1, lineno, "FONTBOUNDINGBOX" ));
        error = FT_THROW( Missing_Fontboundingbox_Field );
        goto Exit;
      }

      line   = bdf_strtok_( line, ' ' );
      p->cnt = font->glyphs_size = bdf_atoul_( line );

      /* We need at least 20 bytes per glyph. */
      if ( p->cnt > p->size / 20 )
      {
        p->cnt = font->glyphs_size = p->size / 20;
        FT_TRACE2(( "bdf_parse_start_: " ACMSG17, p->cnt ));
      }

      /* Make sure the number of glyphs is non-zero. */
      if ( p->cnt == 0 )
        font->glyphs_size = 64;

      /* Limit ourselves to 1,114,112 glyphs in the font (this is the */
      /* number of code points available in Unicode).                 */
      if ( p->cnt >= 0x110000UL )
      {
        FT_ERROR(( "bdf_parse_start_: " ERRMSG5, lineno, "CHARS" ));
        error = FT_THROW( Invalid_Argument );
        goto Exit;
      }

      if ( FT_NEW_ARRAY( font->glyphs, font->glyphs_size ) )
        goto Exit;

      p->flags |= BDF_GLYPHS_;

      *(bdf_line_func_t_*)next = bdf_parse_glyphs_;

      goto Exit;
    }

    FT_ERROR(( "bdf_parse_start_: " ERRMSG9, lineno ));
    error = FT_THROW( Invalid_File_Format );

  Exit:
    return error;
  }


  static FT_Error
  bdf_readstream_( FT_Stream       stream,
                   bdf_parse_t_*   p,
                   unsigned long*  lno )
  {
    bdf_line_func_t_  cb = bdf_parse_start_;
    unsigned long     lineno, buf_size;
    unsigned long     bytes, start, end, cursor, avail;
    char*             buf    = NULL;
    FT_Memory         memory = stream->memory;
    FT_Error          error  = FT_Err_Ok;


    /* initial size and allocation of the input buffer */
    buf_size = 1024;

    if ( FT_QALLOC( buf, buf_size ) )
      goto Exit;

    lineno = 1;
    start  = 0;
    cursor = 0;

  Refill:
    bytes = FT_Stream_TryRead( stream,
                               (FT_Byte*)buf + cursor, buf_size - cursor );
    avail = cursor + bytes;

    while ( bytes )
    {
      /* try to find the start of the line */
      while ( start < avail && buf[start] < ' ' )
        start++;

      /* try to find the end of the line */
      end = start + 1;
      while (   end < avail && buf[end] >= ' ' )
        end++;

      /* if we hit the end of the buffer, try shifting its content */
      /* or even resizing it                                       */
      if ( end >= avail )
      {
        if ( start == 0 )
        {
          /* this line is definitely too long; try resizing the input */
          /* buffer a bit to handle it.                               */
          FT_ULong  new_size;


          if ( buf_size >= 65536UL )  /* limit ourselves to 64KByte */
          {
            FT_ERROR(( "bdf_readstream_: " ERRMSG6, lineno ));
            error = FT_THROW( Invalid_File_Format );

            goto Exit;
          }

          new_size = buf_size * 4;
          if ( FT_QREALLOC( buf, buf_size, new_size ) )
            goto Exit;

          cursor   = avail;
          buf_size = new_size;
        }
        else
        {
          cursor = avail - start;

          FT_MEM_MOVE( buf, buf + start, cursor );

          start  = 0;
        }
        goto Refill;
      }

      /* NUL-terminate the line. */
      buf[end] = 0;

      if ( buf[start] != '#' )
      {
        error = (*cb)( buf + start, end - start, lineno, p, (void*)&cb );
        if ( error )
          break;
      }

      lineno  += 1;
      start    = end + 1;
    }

    *lno = lineno;

  Exit:
    FT_FREE( buf );
    return error;
  }


  /**************************************************************************
   *
   * API.
   *
   */


  FT_LOCAL_DEF( FT_Error )
  bdf_load_font( FT_Stream       stream,
                 FT_Memory       memory,
                 unsigned long   flags,
                 bdf_font_t*    *font )
  {
    unsigned long  lineno = 0; /* make compiler happy */
    bdf_parse_t_   *p     = NULL;

    FT_Error  error = FT_Err_Ok;


    if ( FT_NEW( p ) )
      goto Exit;

    p->flags  = flags;   /* comments, metrics, unencoded */
    p->minlb  = 32767;
    p->size   = stream->size;
    p->memory = memory;  /* only during font creation */

    error = bdf_readstream_( stream, p, &lineno );
    if ( error )
      goto Fail;

    if ( p->font )
    {
      /* If the number of glyphs loaded is not that of the original count, */
      /* indicate the difference.                                          */
      if ( p->cnt != p->font->glyphs_used + p->font->unencoded_used )
      {
        FT_TRACE2(( "bdf_load_font: " ACMSG15, p->cnt,
                    p->font->glyphs_used + p->font->unencoded_used ));
      }

      /* Once the font has been loaded, adjust the overall font metrics if */
      /* necessary.                                                        */
      if ( p->flags & BDF_CORRECT_METRICS &&
           ( p->font->glyphs_used > 0 || p->font->unencoded_used > 0 ) )
      {
        if ( p->maxrb - p->minlb != p->font->bbx.width )
        {
          FT_TRACE2(( "bdf_load_font: " ACMSG3,
                      p->font->bbx.width, p->maxrb - p->minlb ));
          p->font->bbx.width = (unsigned short)( p->maxrb - p->minlb );
        }

        if ( p->font->bbx.x_offset != p->minlb )
        {
          FT_TRACE2(( "bdf_load_font: " ACMSG4,
                      p->font->bbx.x_offset, p->minlb ));
          p->font->bbx.x_offset = p->minlb;
        }

        if ( p->font->bbx.ascent != p->maxas )
        {
          FT_TRACE2(( "bdf_load_font: " ACMSG5,
                      p->font->bbx.ascent, p->maxas ));
          p->font->bbx.ascent = p->maxas;
        }

        if ( p->font->bbx.descent != p->maxds )
        {
          FT_TRACE2(( "bdf_load_font: " ACMSG6,
                      p->font->bbx.descent, p->maxds ));
          p->font->bbx.descent  = p->maxds;
          p->font->bbx.y_offset = (short)( -p->maxds );
        }

        if ( p->maxas + p->maxds != p->font->bbx.height )
        {
          FT_TRACE2(( "bdf_load_font: " ACMSG7,
                      p->font->bbx.height, p->maxas + p->maxds ));
          p->font->bbx.height = (unsigned short)( p->maxas + p->maxds );
        }
      }
    }

    if ( p->flags & BDF_START_ )
    {
      /* The ENDFONT field was never reached or did not exist. */
      if ( !( p->flags & BDF_GLYPHS_ ) )
      {
        /* Error happened while parsing header. */
        FT_ERROR(( "bdf_load_font: " ERRMSG2, lineno ));
        error = FT_THROW( Corrupted_Font_Header );
        goto Fail;
      }
      else
      {
        /* Error happened when parsing glyphs. */
        FT_ERROR(( "bdf_load_font: " ERRMSG3, lineno ));
        error = FT_THROW( Corrupted_Font_Glyphs );
        goto Fail;
      }
    }

    if ( !p->font && !error )
      error = FT_THROW( Invalid_File_Format );

    *font = p->font;

  Exit:
    if ( p )
    {
      FT_FREE( p->glyph_name );
      FT_FREE( p );
    }

    return error;

  Fail:
    bdf_free_font( p->font );

    FT_FREE( p->font );

    goto Exit;
  }


  FT_LOCAL_DEF( void )
  bdf_free_font( bdf_font_t*  font )
  {
    bdf_property_t*  prop;
    unsigned long    i;
    bdf_glyph_t*     glyphs;
    FT_Memory        memory;


    if ( font == NULL )
      return;

    memory = font->memory;

    FT_FREE( font->name );

    /* Free up the internal hash table of property names. */
    if ( font->internal )
    {
      ft_hash_str_free( font->internal, memory );
      FT_FREE( font->internal );
    }

    /* Free up the comment info. */
    FT_FREE( font->comments );

    /* Free up the properties. */
    for ( i = 0; i < font->props_size; i++ )
    {
      if ( font->props[i].format == BDF_ATOM )
        FT_FREE( font->props[i].value.atom );
    }

    FT_FREE( font->props );

    /* Free up the character info. */
    for ( i = 0, glyphs = font->glyphs;
          i < font->glyphs_used; i++, glyphs++ )
    {
      FT_FREE( glyphs->name );
      FT_FREE( glyphs->bitmap );
    }

    for ( i = 0, glyphs = font->unencoded; i < font->unencoded_used;
          i++, glyphs++ )
    {
      FT_FREE( glyphs->name );
      FT_FREE( glyphs->bitmap );
    }

    FT_FREE( font->glyphs );
    FT_FREE( font->unencoded );

    /* bdf_cleanup */
    ft_hash_str_free( &(font->proptbl), memory );

    /* Free up the user defined properties. */
    for ( prop = font->user_props, i = 0;
          i < font->nuser_props; i++, prop++ )
      FT_FREE( prop->name );

    FT_FREE( font->user_props );

    /* FREE( font ); */ /* XXX Fixme */
  }


  FT_LOCAL_DEF( bdf_property_t * )
  bdf_get_font_property( bdf_font_t*  font,
                         const char*  name )
  {
    size_t*  propid;


    if ( font == NULL || font->props_size == 0 || name == NULL || *name == 0 )
      return 0;

    propid = ft_hash_str_lookup( name, font->internal );

    return propid ? ( font->props + *propid ) : NULL;
  }


/* END */
