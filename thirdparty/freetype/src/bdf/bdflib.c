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
   * Default BDF font options.
   *
   */


  static const bdf_options_t  _bdf_opts =
  {
    1,                /* Correct metrics.               */
    1,                /* Preserve unencoded glyphs.     */
    0,                /* Preserve comments.             */
    BDF_PROPORTIONAL  /* Default spacing.               */
  };


  /**************************************************************************
   *
   * Builtin BDF font properties.
   *
   */

  /* List of most properties that might appear in a font.  Doesn't include */
  /* the RAW_* and AXIS_* properties in X11R6 polymorphic fonts.           */

  static const bdf_property_t  _bdf_properties[] =
  {
    { "ADD_STYLE_NAME",          BDF_ATOM,     1, { 0 } },
    { "AVERAGE_WIDTH",           BDF_INTEGER,  1, { 0 } },
    { "AVG_CAPITAL_WIDTH",       BDF_INTEGER,  1, { 0 } },
    { "AVG_LOWERCASE_WIDTH",     BDF_INTEGER,  1, { 0 } },
    { "CAP_HEIGHT",              BDF_INTEGER,  1, { 0 } },
    { "CHARSET_COLLECTIONS",     BDF_ATOM,     1, { 0 } },
    { "CHARSET_ENCODING",        BDF_ATOM,     1, { 0 } },
    { "CHARSET_REGISTRY",        BDF_ATOM,     1, { 0 } },
    { "COMMENT",                 BDF_ATOM,     1, { 0 } },
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
  _num_bdf_properties = sizeof ( _bdf_properties ) /
                        sizeof ( _bdf_properties[0] );


  /* An auxiliary macro to parse properties, to be used in conditionals. */
  /* It behaves like `strncmp' but also tests the following character    */
  /* whether it is a whitespace or null.                                 */
  /* `property' is a constant string of length `n' to compare with.      */
#define _bdf_strncmp( name, property, n )      \
          ( ft_strncmp( name, property, n ) || \
            !( name[n] == ' '  ||              \
               name[n] == '\0' ||              \
               name[n] == '\n' ||              \
               name[n] == '\r' ||              \
               name[n] == '\t' )            )

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
#define ACMSG9   "SWIDTH field missing at line %ld.  Set automatically.\n"
#define ACMSG10  "DWIDTH field missing at line %ld.  Set to glyph width.\n"
#define ACMSG11  "SIZE bits per pixel field adjusted to %hd.\n"
#define ACMSG13  "Glyph %lu extra rows removed.\n"
#define ACMSG14  "Glyph %lu extra columns removed.\n"
#define ACMSG15  "Incorrect glyph count: %ld indicated but %ld found.\n"
#define ACMSG16  "Glyph %lu missing columns padded with zero bits.\n"
#define ACMSG17  "Adjusting number of glyphs to %ld.\n"

  /* Error messages. */
#define ERRMSG1  "[line %ld] Missing `%s' line.\n"
#define ERRMSG2  "[line %ld] Font header corrupted or missing fields.\n"
#define ERRMSG3  "[line %ld] Font glyphs corrupted or missing fields.\n"
#define ERRMSG4  "[line %ld] BBX too big.\n"
#define ERRMSG5  "[line %ld] `%s' value too big.\n"
#define ERRMSG6  "[line %ld] Input line too long.\n"
#define ERRMSG7  "[line %ld] Font name too long.\n"
#define ERRMSG8  "[line %ld] Invalid `%s' value.\n"
#define ERRMSG9  "[line %ld] Invalid keyword.\n"

  /* Debug messages. */
#define DBGMSG1  "  [%6ld] %s" /* no \n */
#define DBGMSG2  " (0x%lX)\n"


  /**************************************************************************
   *
   * Utility types and functions.
   *
   */


  /* Function type for parsing lines of a BDF font. */

  typedef FT_Error
  (*_bdf_line_func_t)( char*          line,
                       unsigned long  linelen,
                       unsigned long  lineno,
                       void*          call_data,
                       void*          client_data );


  /* List structure for splitting lines into fields. */

  typedef struct  _bdf_list_t_
  {
    char**         field;
    unsigned long  size;
    unsigned long  used;
    FT_Memory      memory;

  } _bdf_list_t;


  /* Structure used while loading BDF fonts. */

  typedef struct  _bdf_parse_t_
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

    bdf_font_t*     font;
    bdf_options_t*  opts;

    _bdf_list_t     list;

    FT_Memory       memory;
    unsigned long   size;        /* the stream size */

  } _bdf_parse_t;


#define setsbit( m, cc ) \
          ( m[(FT_Byte)(cc) >> 3] |= (FT_Byte)( 1 << ( (cc) & 7 ) ) )
#define sbitset( m, cc ) \
          ( m[(FT_Byte)(cc) >> 3]  & ( 1 << ( (cc) & 7 ) ) )


  static void
  _bdf_list_init( _bdf_list_t*  list,
                  FT_Memory     memory )
  {
    FT_ZERO( list );
    list->memory = memory;
  }


  static void
  _bdf_list_done( _bdf_list_t*  list )
  {
    FT_Memory  memory = list->memory;


    if ( memory )
    {
      FT_FREE( list->field );
      FT_ZERO( list );
    }
  }


  static FT_Error
  _bdf_list_ensure( _bdf_list_t*   list,
                    unsigned long  num_items ) /* same as _bdf_list_t.used */
  {
    FT_Error  error = FT_Err_Ok;


    if ( num_items > list->size )
    {
      unsigned long  oldsize = list->size; /* same as _bdf_list_t.size */
      unsigned long  newsize = oldsize + ( oldsize >> 1 ) + 5;
      unsigned long  bigsize = (unsigned long)( FT_INT_MAX / sizeof ( char* ) );
      FT_Memory      memory  = list->memory;


      if ( oldsize == bigsize )
      {
        error = FT_THROW( Out_Of_Memory );
        goto Exit;
      }
      else if ( newsize < oldsize || newsize > bigsize )
        newsize = bigsize;

      if ( FT_QRENEW_ARRAY( list->field, oldsize, newsize ) )
        goto Exit;

      list->size = newsize;
    }

  Exit:
    return error;
  }


  static void
  _bdf_list_shift( _bdf_list_t*   list,
                   unsigned long  n )
  {
    unsigned long  i, u;


    if ( list == NULL || list->used == 0 || n == 0 )
      return;

    if ( n >= list->used )
    {
      list->used = 0;
      return;
    }

    for ( u = n, i = 0; u < list->used; i++, u++ )
      list->field[i] = list->field[u];
    list->used -= n;
  }


  /* An empty string for empty fields. */

  static const char  empty[] = "";      /* XXX eliminate this */


  static char *
  _bdf_list_join( _bdf_list_t*    list,
                  int             c,
                  unsigned long  *alen )
  {
    unsigned long  i, j;
    char*          dp;


    *alen = 0;

    if ( list == NULL || list->used == 0 )
      return 0;

    dp = list->field[0];
    for ( i = j = 0; i < list->used; i++ )
    {
      char*  fp = list->field[i];


      while ( *fp )
        dp[j++] = *fp++;

      if ( i + 1 < list->used )
        dp[j++] = (char)c;
    }
    if ( dp != empty )
      dp[j] = 0;

    *alen = j;
    return dp;
  }


  /* The code below ensures that we have at least 4 + 1 `field' */
  /* elements in `list' (which are possibly NULL) so that we    */
  /* don't have to check the number of fields in most cases.    */

  static FT_Error
  _bdf_list_split( _bdf_list_t*   list,
                   const char*    separators,
                   char*          line,
                   unsigned long  linelen )
  {
    unsigned long  final_empty;
    int            mult;
    const char     *sp, *end;
    char           *ep;
    char           seps[32];
    FT_Error       error = FT_Err_Ok;


    /* Initialize the list. */
    list->used = 0;
    if ( list->size )
    {
      list->field[0] = (char*)empty;
      list->field[1] = (char*)empty;
      list->field[2] = (char*)empty;
      list->field[3] = (char*)empty;
      list->field[4] = (char*)empty;
    }

    /* If the line is empty, then simply return. */
    if ( linelen == 0 || line[0] == 0 )
      goto Exit;

    /* In the original code, if the `separators' parameter is NULL or */
    /* empty, the list is split into individual bytes.  We don't need */
    /* this, so an error is signaled.                                 */
    if ( separators == NULL || *separators == 0 )
    {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    /* Prepare the separator bitmap. */
    FT_MEM_ZERO( seps, 32 );

    /* If the very last character of the separator string is a plus, then */
    /* set the `mult' flag to indicate that multiple separators should be */
    /* collapsed into one.                                                */
    for ( mult = 0, sp = separators; sp && *sp; sp++ )
    {
      if ( *sp == '+' && *( sp + 1 ) == 0 )
        mult = 1;
      else
        setsbit( seps, *sp );
    }

    /* Break the line up into fields. */
    for ( final_empty = 0, sp = ep = line, end = sp + linelen;
          sp < end && *sp; )
    {
      /* Collect everything that is not a separator. */
      for ( ; *ep && !sbitset( seps, *ep ); ep++ )
        ;

      /* Resize the list if necessary. */
      if ( list->used == list->size )
      {
        error = _bdf_list_ensure( list, list->used + 1 );
        if ( error )
          goto Exit;
      }

      /* Assign the field appropriately. */
      list->field[list->used++] = ( ep > sp ) ? (char*)sp : (char*)empty;

      sp = ep;

      if ( mult )
      {
        /* If multiple separators should be collapsed, do it now by */
        /* setting all the separator characters to 0.               */
        for ( ; *ep && sbitset( seps, *ep ); ep++ )
          *ep = 0;
      }
      else if ( *ep != 0 )
        /* Don't collapse multiple separators by making them 0, so just */
        /* make the one encountered 0.                                  */
        *ep++ = 0;

      final_empty = ( ep > sp && *ep == 0 );
      sp = ep;
    }

    /* Finally, NULL-terminate the list. */
    if ( list->used + final_empty >= list->size )
    {
      error = _bdf_list_ensure( list, list->used + final_empty + 1 );
      if ( error )
        goto Exit;
    }

    if ( final_empty )
      list->field[list->used++] = (char*)empty;

    list->field[list->used] = NULL;

  Exit:
    return error;
  }


#define NO_SKIP  256  /* this value cannot be stored in a 'char' */


  static FT_Error
  _bdf_readstream( FT_Stream         stream,
                   _bdf_line_func_t  callback,
                   void*             client_data,
                   unsigned long    *lno )
  {
    _bdf_line_func_t  cb;
    unsigned long     lineno, buf_size;
    int               refill, hold, to_skip;
    ptrdiff_t         bytes, start, end, cursor, avail;
    char*             buf    = NULL;
    FT_Memory         memory = stream->memory;
    FT_Error          error  = FT_Err_Ok;


    if ( callback == NULL )
    {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    /* initial size and allocation of the input buffer */
    buf_size = 1024;

    if ( FT_QALLOC( buf, buf_size ) )
      goto Exit;

    cb      = callback;
    lineno  = 1;
    buf[0]  = 0;
    start   = 0;
    avail   = 0;
    cursor  = 0;
    refill  = 1;
    to_skip = NO_SKIP;
    bytes   = 0;        /* make compiler happy */

    for (;;)
    {
      if ( refill )
      {
        bytes  = (ptrdiff_t)FT_Stream_TryRead(
                   stream, (FT_Byte*)buf + cursor,
                   buf_size - (unsigned long)cursor );
        avail  = cursor + bytes;
        cursor = 0;
        refill = 0;
      }

      end = start;

      /* should we skip an optional character like \n or \r? */
      if ( start < avail && buf[start] == to_skip )
      {
        start  += 1;
        to_skip = NO_SKIP;
        continue;
      }

      /* try to find the end of the line */
      while ( end < avail && buf[end] != '\n' && buf[end] != '\r' )
        end++;

      /* if we hit the end of the buffer, try shifting its content */
      /* or even resizing it                                       */
      if ( end >= avail )
      {
        if ( bytes == 0 )
        {
          /* last line in file doesn't end in \r or \n; */
          /* ignore it then exit                        */
          if ( lineno == 1 )
            error = FT_THROW( Missing_Startfont_Field );
          break;
        }

        if ( start == 0 )
        {
          /* this line is definitely too long; try resizing the input */
          /* buffer a bit to handle it.                               */
          FT_ULong  new_size;


          if ( buf_size >= 65536UL )  /* limit ourselves to 64KByte */
          {
            if ( lineno == 1 )
              error = FT_THROW( Missing_Startfont_Field );
            else
            {
              FT_ERROR(( "_bdf_readstream: " ERRMSG6, lineno ));
              error = FT_THROW( Invalid_Argument );
            }
            goto Exit;
          }

          new_size = buf_size * 2;
          if ( FT_QREALLOC( buf, buf_size, new_size ) )
            goto Exit;

          cursor   = avail;
          buf_size = new_size;
        }
        else
        {
          bytes = avail - start;

          FT_MEM_MOVE( buf, buf + start, bytes );

          cursor = bytes;
          start  = 0;
        }
        refill = 1;
        continue;
      }

      /* Temporarily NUL-terminate the line. */
      hold     = buf[end];
      buf[end] = 0;

      /* XXX: Use encoding independent value for 0x1A */
      if ( buf[start] != '#' && buf[start] != 0x1A && end > start )
      {
        error = (*cb)( buf + start, (unsigned long)( end - start ), lineno,
                       (void*)&cb, client_data );
        /* Redo if we have encountered CHARS without properties. */
        if ( error == -1 )
          error = (*cb)( buf + start, (unsigned long)( end - start ), lineno,
                         (void*)&cb, client_data );
        if ( error )
          break;
      }

      lineno  += 1;
      buf[end] = (char)hold;
      start    = end + 1;

      if ( hold == '\n' )
        to_skip = '\r';
      else if ( hold == '\r' )
        to_skip = '\n';
      else
        to_skip = NO_SKIP;
    }

    *lno = lineno;

  Exit:
    FT_FREE( buf );
    return error;
  }


  /* XXX: make this work with EBCDIC also */

  static const unsigned char  a2i[128] =
  {
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
  };

  static const unsigned char  ddigits[32] =
  {
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0x03,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  };

  static const unsigned char  hdigits[32] =
  {
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0x03,
    0x7E, 0x00, 0x00, 0x00, 0x7E, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  };


  /* Routine to convert a decimal ASCII string to an unsigned long integer. */
  static unsigned long
  _bdf_atoul( const char*  s )
  {
    unsigned long  v;


    if ( s == NULL || *s == 0 )
      return 0;

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
  _bdf_atol( const char*  s )
  {
    long  v, neg;


    if ( s == NULL || *s == 0 )
      return 0;

    /* Check for a minus sign. */
    neg = 0;
    if ( *s == '-' )
    {
      s++;
      neg = 1;
    }

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

    return ( !neg ) ? v : -v;
  }


  /* Routine to convert a decimal ASCII string to an unsigned short integer. */
  static unsigned short
  _bdf_atous( const char*  s )
  {
    unsigned short  v;


    if ( s == NULL || *s == 0 )
      return 0;

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
  _bdf_atos( const char*  s )
  {
    short  v, neg;


    if ( s == NULL || *s == 0 )
      return 0;

    /* Check for a minus. */
    neg = 0;
    if ( *s == '-' )
    {
      s++;
      neg = 1;
    }

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

    return (short)( ( !neg ) ? v : -v );
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

    n = ft_strlen( name ) + 1;
    if ( n > FT_LONG_MAX )
      return FT_THROW( Invalid_Argument );

    if ( FT_QALLOC( p->name, n ) )
      goto Exit;

    FT_MEM_COPY( (char *)p->name, name, n );

    p->format     = format;
    p->builtin    = 0;
    p->value.atom = NULL;  /* nothing is ever stored here */

    n = _num_bdf_properties + font->nuser_props;

    error = ft_hash_str_insert( p->name, n, &(font->proptbl), memory );
    if ( error )
      goto Exit;

    font->nuser_props++;

  Exit:
    return error;
  }


  FT_LOCAL_DEF( bdf_property_t* )
  bdf_get_property( char*        name,
                    bdf_font_t*  font )
  {
    size_t*  propid;


    if ( name == NULL || *name == 0 )
      return 0;

    if ( ( propid = ft_hash_str_lookup( name, &(font->proptbl) ) ) == NULL )
      return 0;

    if ( *propid >= _num_bdf_properties )
      return font->user_props + ( *propid - _num_bdf_properties );

    return (bdf_property_t*)_bdf_properties + *propid;
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

#define BDF_SWIDTH_ADJ_  0x1000U

#define BDF_GLYPH_BITS_ ( BDF_GLYPH_    | \
                          BDF_ENCODING_ | \
                          BDF_SWIDTH_   | \
                          BDF_DWIDTH_   | \
                          BDF_BBX_      | \
                          BDF_BITMAP_   )

#define BDF_GLYPH_WIDTH_CHECK_   0x40000000UL
#define BDF_GLYPH_HEIGHT_CHECK_  0x80000000UL


  static FT_Error
  _bdf_add_comment( bdf_font_t*    font,
                    char*          comment,
                    unsigned long  len )
  {
    char*      cp;
    FT_Memory  memory = font->memory;
    FT_Error   error  = FT_Err_Ok;


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


  /* Set the spacing from the font name if it exists, or set it to the */
  /* default specified in the options.                                 */
  static FT_Error
  _bdf_set_default_spacing( bdf_font_t*     font,
                            bdf_options_t*  opts,
                            unsigned long   lineno )
  {
    size_t       len;
    char         name[256];
    _bdf_list_t  list;
    FT_Memory    memory;
    FT_Error     error = FT_Err_Ok;

    FT_UNUSED( lineno );        /* only used in debug mode */


    if ( font == NULL || font->name == NULL || font->name[0] == 0 )
    {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    memory = font->memory;

    _bdf_list_init( &list, memory );

    font->spacing = opts->font_spacing;

    len = ft_strlen( font->name ) + 1;
    /* Limit ourselves to 256 characters in the font name. */
    if ( len >= 256 )
    {
      FT_ERROR(( "_bdf_set_default_spacing: " ERRMSG7, lineno ));
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    FT_MEM_COPY( name, font->name, len );

    error = _bdf_list_split( &list, "-", name, (unsigned long)len );
    if ( error )
      goto Fail;

    if ( list.used == 15 )
    {
      switch ( list.field[11][0] )
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
        font->spacing = BDF_PROPORTIONAL;
        break;
      }
    }

  Fail:
    _bdf_list_done( &list );

  Exit:
    return error;
  }


  /* Determine whether the property is an atom or not.  If it is, then */
  /* clean it up so the double quotes are removed if they exist.       */
  static int
  _bdf_is_atom( char*          line,
                unsigned long  linelen,
                char**         name,
                char**         value,
                bdf_font_t*    font )
  {
    int              hold;
    char             *sp, *ep;
    bdf_property_t*  p;


    *name = sp = ep = line;

    while ( *ep && *ep != ' ' && *ep != '\t' )
      ep++;

    hold = -1;
    if ( *ep )
    {
      hold = *ep;
      *ep  = 0;
    }

    p = bdf_get_property( sp, font );

    /* Restore the character that was saved before any return can happen. */
    if ( hold != -1 )
      *ep = (char)hold;

    /* If the property exists and is not an atom, just return here. */
    if ( p && p->format != BDF_ATOM )
      return 0;

    /* The property is an atom.  Trim all leading and trailing whitespace */
    /* and double quotes for the atom value.                              */
    sp = ep;
    ep = line + linelen;

    /* Trim the leading whitespace if it exists. */
    if ( *sp )
      *sp++ = 0;
    while ( *sp                           &&
            ( *sp == ' ' || *sp == '\t' ) )
      sp++;

    /* Trim the leading double quote if it exists. */
    if ( *sp == '"' )
      sp++;
    *value = sp;

    /* Trim the trailing whitespace if it exists. */
    while ( ep > sp                                       &&
            ( *( ep - 1 ) == ' ' || *( ep - 1 ) == '\t' ) )
      *--ep = 0;

    /* Trim the trailing double quote if it exists. */
    if ( ep > sp && *( ep - 1 ) == '"' )
      *--ep = 0;

    return 1;
  }


  static FT_Error
  _bdf_add_property( bdf_font_t*    font,
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
    if ( ( propid = ft_hash_str_lookup( name,
                                        (FT_Hash)font->internal ) ) != NULL )
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
        fp->value.l = _bdf_atol( value );
        break;

      case BDF_CARDINAL:
        fp->value.ul = _bdf_atoul( value );
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

    if ( *propid >= _num_bdf_properties )
      prop = font->user_props + ( *propid - _num_bdf_properties );
    else
      prop = (bdf_property_t*)_bdf_properties + *propid;

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
      fp->value.l = _bdf_atol( value );
      break;

    case BDF_CARDINAL:
      fp->value.ul = _bdf_atoul( value );
      break;
    }

    /* If the property happens to be a comment, then it doesn't need */
    /* to be added to the internal hash table.                       */
    if ( _bdf_strncmp( name, "COMMENT", 7 ) != 0 )
    {
      /* Add the property to the font property table. */
      error = ft_hash_str_insert( fp->name,
                                  font->props_used,
                                  (FT_Hash)font->internal,
                                  memory );
      if ( error )
        goto Exit;
    }

    font->props_used++;

    /* Some special cases need to be handled here.  The DEFAULT_CHAR       */
    /* property needs to be located if it exists in the property list, the */
    /* FONT_ASCENT and FONT_DESCENT need to be assigned if they are        */
    /* present, and the SPACING property should override the default       */
    /* spacing.                                                            */
    if ( _bdf_strncmp( name, "DEFAULT_CHAR", 12 ) == 0 )
      font->default_char = fp->value.ul;
    else if ( _bdf_strncmp( name, "FONT_ASCENT", 11 ) == 0 )
      font->font_ascent = fp->value.l;
    else if ( _bdf_strncmp( name, "FONT_DESCENT", 12 ) == 0 )
      font->font_descent = fp->value.l;
    else if ( _bdf_strncmp( name, "SPACING", 7 ) == 0 )
    {
      if ( !fp->value.atom )
      {
        FT_ERROR(( "_bdf_add_property: " ERRMSG8, lineno, "SPACING" ));
        error = FT_THROW( Invalid_File_Format );
        goto Exit;
      }

      if ( fp->value.atom[0] == 'p' || fp->value.atom[0] == 'P' )
        font->spacing = BDF_PROPORTIONAL;
      else if ( fp->value.atom[0] == 'm' || fp->value.atom[0] == 'M' )
        font->spacing = BDF_MONOWIDTH;
      else if ( fp->value.atom[0] == 'c' || fp->value.atom[0] == 'C' )
        font->spacing = BDF_CHARCELL;
    }

  Exit:
    return error;
  }


  static const unsigned char nibble_mask[8] =
  {
    0xFF, 0x80, 0xC0, 0xE0, 0xF0, 0xF8, 0xFC, 0xFE
  };


  static FT_Error
  _bdf_parse_end( char*          line,
                  unsigned long  linelen,
                  unsigned long  lineno,
                  void*          call_data,
                  void*          client_data )
  {
    /* a no-op; we ignore everything after `ENDFONT' */

    FT_UNUSED( line );
    FT_UNUSED( linelen );
    FT_UNUSED( lineno );
    FT_UNUSED( call_data );
    FT_UNUSED( client_data );

    return FT_Err_Ok;
  }


  /* Actually parse the glyph info and bitmaps. */
  static FT_Error
  _bdf_parse_glyphs( char*          line,
                     unsigned long  linelen,
                     unsigned long  lineno,
                     void*          call_data,
                     void*          client_data )
  {
    int                c, mask_index;
    char*              s;
    unsigned char*     bp;
    unsigned long      i, slen, nibbles;

    _bdf_line_func_t*  next;
    _bdf_parse_t*      p;
    bdf_glyph_t*       glyph;
    bdf_font_t*        font;

    FT_Memory          memory;
    FT_Error           error = FT_Err_Ok;

    FT_UNUSED( lineno );        /* only used in debug mode */


    next = (_bdf_line_func_t *)call_data;
    p    = (_bdf_parse_t *)    client_data;

    font   = p->font;
    memory = font->memory;

    /* Check for a comment. */
    if ( _bdf_strncmp( line, "COMMENT", 7 ) == 0 )
    {
      if ( p->opts->keep_comments )
      {
        linelen -= 7;

        s = line + 7;
        if ( *s != 0 )
        {
          s++;
          linelen--;
        }
        error = _bdf_add_comment( p->font, s, linelen );
      }
      goto Exit;
    }

    /* The very first thing expected is the number of glyphs. */
    if ( !( p->flags & BDF_GLYPHS_ ) )
    {
      if ( _bdf_strncmp( line, "CHARS", 5 ) != 0 )
      {
        FT_ERROR(( "_bdf_parse_glyphs: " ERRMSG1, lineno, "CHARS" ));
        error = FT_THROW( Missing_Chars_Field );
        goto Exit;
      }

      error = _bdf_list_split( &p->list, " +", line, linelen );
      if ( error )
        goto Exit;
      p->cnt = font->glyphs_size = _bdf_atoul( p->list.field[1] );

      /* We need at least 20 bytes per glyph. */
      if ( p->cnt > p->size / 20 )
      {
        p->cnt = font->glyphs_size = p->size / 20;
        FT_TRACE2(( "_bdf_parse_glyphs: " ACMSG17, p->cnt ));
      }

      /* Make sure the number of glyphs is non-zero. */
      if ( p->cnt == 0 )
        font->glyphs_size = 64;

      /* Limit ourselves to 1,114,112 glyphs in the font (this is the */
      /* number of code points available in Unicode).                 */
      if ( p->cnt >= 0x110000UL )
      {
        FT_ERROR(( "_bdf_parse_glyphs: " ERRMSG5, lineno, "CHARS" ));
        error = FT_THROW( Invalid_Argument );
        goto Exit;
      }

      if ( FT_NEW_ARRAY( font->glyphs, font->glyphs_size ) )
        goto Exit;

      p->flags |= BDF_GLYPHS_;

      goto Exit;
    }

    /* Check for the ENDFONT field. */
    if ( _bdf_strncmp( line, "ENDFONT", 7 ) == 0 )
    {
      if ( p->flags & BDF_GLYPH_BITS_ )
      {
        /* Missing ENDCHAR field. */
        FT_ERROR(( "_bdf_parse_glyphs: " ERRMSG1, lineno, "ENDCHAR" ));
        error = FT_THROW( Corrupted_Font_Glyphs );
        goto Exit;
      }

      /* Sort the glyphs by encoding. */
      ft_qsort( (char *)font->glyphs,
                font->glyphs_used,
                sizeof ( bdf_glyph_t ),
                by_encoding );

      p->flags &= ~BDF_START_;
      *next     = _bdf_parse_end;

      goto Exit;
    }

    /* Check for the ENDCHAR field. */
    if ( _bdf_strncmp( line, "ENDCHAR", 7 ) == 0 )
    {
      p->glyph_enc = 0;
      p->flags    &= ~BDF_GLYPH_BITS_;

      goto Exit;
    }

    /* Check whether a glyph is being scanned but should be */
    /* ignored because it is an unencoded glyph.            */
    if ( ( p->flags & BDF_GLYPH_ )     &&
         p->glyph_enc            == -1 &&
         p->opts->keep_unencoded == 0  )
      goto Exit;

    /* Check for the STARTCHAR field. */
    if ( _bdf_strncmp( line, "STARTCHAR", 9 ) == 0 )
    {
      if ( p->flags & BDF_GLYPH_BITS_ )
      {
        /* Missing ENDCHAR field. */
        FT_ERROR(( "_bdf_parse_glyphs: " ERRMSG1, lineno, "ENDCHAR" ));
        error = FT_THROW( Missing_Startchar_Field );
        goto Exit;
      }

      /* Set the character name in the parse info first until the */
      /* encoding can be checked for an unencoded character.      */
      FT_FREE( p->glyph_name );

      error = _bdf_list_split( &p->list, " +", line, linelen );
      if ( error )
        goto Exit;

      _bdf_list_shift( &p->list, 1 );

      s = _bdf_list_join( &p->list, ' ', &slen );

      if ( !s )
      {
        FT_ERROR(( "_bdf_parse_glyphs: " ERRMSG8, lineno, "STARTCHAR" ));
        error = FT_THROW( Invalid_File_Format );
        goto Exit;
      }

      if ( FT_QALLOC( p->glyph_name, slen + 1 ) )
        goto Exit;

      FT_MEM_COPY( p->glyph_name, s, slen + 1 );

      p->flags |= BDF_GLYPH_;

      FT_TRACE4(( DBGMSG1, lineno, s ));

      goto Exit;
    }

    /* Check for the ENCODING field. */
    if ( _bdf_strncmp( line, "ENCODING", 8 ) == 0 )
    {
      if ( !( p->flags & BDF_GLYPH_ ) )
      {
        /* Missing STARTCHAR field. */
        FT_ERROR(( "_bdf_parse_glyphs: " ERRMSG1, lineno, "STARTCHAR" ));
        error = FT_THROW( Missing_Startchar_Field );
        goto Exit;
      }

      error = _bdf_list_split( &p->list, " +", line, linelen );
      if ( error )
        goto Exit;

      p->glyph_enc = _bdf_atol( p->list.field[1] );

      /* Normalize negative encoding values.  The specification only */
      /* allows -1, but we can be more generous here.                */
      if ( p->glyph_enc < -1 )
        p->glyph_enc = -1;

      /* Check for alternative encoding format. */
      if ( p->glyph_enc == -1 && p->list.used > 2 )
        p->glyph_enc = _bdf_atol( p->list.field[2] );

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

        /* Reset the initial glyph info. */
        p->glyph_name = NULL;
      }
      else
      {
        /* Unencoded glyph.  Check whether it should */
        /* be added or not.                          */
        if ( p->opts->keep_unencoded )
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

          /* Reset the initial glyph info. */
          p->glyph_name = NULL;
        }
        else
        {
          /* Free up the glyph name if the unencoded shouldn't be */
          /* kept.                                                */
          FT_FREE( p->glyph_name );
        }
      }

      /* Clear the flags that might be added when width and height are */
      /* checked for consistency.                                      */
      p->flags &= ~( BDF_GLYPH_WIDTH_CHECK_ | BDF_GLYPH_HEIGHT_CHECK_ );

      p->flags |= BDF_ENCODING_;

      goto Exit;
    }

    if ( !( p->flags & BDF_ENCODING_ ) )
      goto Missing_Encoding;

    /* Point at the glyph being constructed. */
    if ( p->glyph_enc == -1 )
      glyph = font->unencoded + ( font->unencoded_used - 1 );
    else
      glyph = font->glyphs + ( font->glyphs_used - 1 );

    /* Check whether a bitmap is being constructed. */
    if ( p->flags & BDF_BITMAP_ )
    {
      /* If there are more rows than are specified in the glyph metrics, */
      /* ignore the remaining lines.                                     */
      if ( p->row >= (unsigned long)glyph->bbx.height )
      {
        if ( !( p->flags & BDF_GLYPH_HEIGHT_CHECK_ ) )
        {
          FT_TRACE2(( "_bdf_parse_glyphs: " ACMSG13, glyph->encoding ));
          p->flags |= BDF_GLYPH_HEIGHT_CHECK_;
        }

        goto Exit;
      }

      /* Only collect the number of nibbles indicated by the glyph     */
      /* metrics.  If there are more columns, they are simply ignored. */
      nibbles = glyph->bpr << 1;
      bp      = glyph->bitmap + p->row * glyph->bpr;

      for ( i = 0; i < nibbles; i++ )
      {
        c = line[i];
        if ( !sbitset( hdigits, c ) )
          break;
        *bp = (FT_Byte)( ( *bp << 4 ) + a2i[c] );
        if ( i + 1 < nibbles && ( i & 1 ) )
          *++bp = 0;
      }

      /* If any line has not enough columns,            */
      /* indicate they have been padded with zero bits. */
      if ( i < nibbles                            &&
           !( p->flags & BDF_GLYPH_WIDTH_CHECK_ ) )
      {
        FT_TRACE2(( "_bdf_parse_glyphs: " ACMSG16, glyph->encoding ));
        p->flags       |= BDF_GLYPH_WIDTH_CHECK_;
      }

      /* Remove possible garbage at the right. */
      mask_index = ( glyph->bbx.width * p->font->bpp ) & 7;
      if ( glyph->bbx.width )
        *bp &= nibble_mask[mask_index];

      /* If any line has extra columns, indicate they have been removed. */
      if ( i == nibbles                           &&
           sbitset( hdigits, line[nibbles] )      &&
           !( p->flags & BDF_GLYPH_WIDTH_CHECK_ ) )
      {
        FT_TRACE2(( "_bdf_parse_glyphs: " ACMSG14, glyph->encoding ));
        p->flags       |= BDF_GLYPH_WIDTH_CHECK_;
      }

      p->row++;
      goto Exit;
    }

    /* Expect the SWIDTH (scalable width) field next. */
    if ( _bdf_strncmp( line, "SWIDTH", 6 ) == 0 )
    {
      error = _bdf_list_split( &p->list, " +", line, linelen );
      if ( error )
        goto Exit;

      glyph->swidth = _bdf_atous( p->list.field[1] );
      p->flags |= BDF_SWIDTH_;

      goto Exit;
    }

    /* Expect the DWIDTH (device width) field next. */
    if ( _bdf_strncmp( line, "DWIDTH", 6 ) == 0 )
    {
      error = _bdf_list_split( &p->list, " +", line, linelen );
      if ( error )
        goto Exit;

      glyph->dwidth = _bdf_atous( p->list.field[1] );

      if ( !( p->flags & BDF_SWIDTH_ ) )
      {
        /* Missing SWIDTH field.  Emit an auto correction message and set */
        /* the scalable width from the device width.                      */
        FT_TRACE2(( "_bdf_parse_glyphs: " ACMSG9, lineno ));

        glyph->swidth = (unsigned short)FT_MulDiv(
                          glyph->dwidth, 72000L,
                          (FT_Long)( font->point_size *
                                     font->resolution_x ) );
      }

      p->flags |= BDF_DWIDTH_;
      goto Exit;
    }

    /* Expect the BBX field next. */
    if ( _bdf_strncmp( line, "BBX", 3 ) == 0 )
    {
      error = _bdf_list_split( &p->list, " +", line, linelen );
      if ( error )
        goto Exit;

      glyph->bbx.width    = _bdf_atous( p->list.field[1] );
      glyph->bbx.height   = _bdf_atous( p->list.field[2] );
      glyph->bbx.x_offset = _bdf_atos( p->list.field[3] );
      glyph->bbx.y_offset = _bdf_atos( p->list.field[4] );

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
        FT_TRACE2(( "_bdf_parse_glyphs: " ACMSG10, lineno ));
        glyph->dwidth = glyph->bbx.width;
      }

      /* If the BDF_CORRECT_METRICS flag is set, then adjust the SWIDTH */
      /* value if necessary.                                            */
      if ( p->opts->correct_metrics )
      {
        /* Determine the point size of the glyph. */
        unsigned short  sw = (unsigned short)FT_MulDiv(
                               glyph->dwidth, 72000L,
                               (FT_Long)( font->point_size *
                                          font->resolution_x ) );


        if ( sw != glyph->swidth )
        {
          glyph->swidth = sw;

          p->flags       |= BDF_SWIDTH_ADJ_;
        }
      }

      p->flags |= BDF_BBX_;
      goto Exit;
    }

    /* And finally, gather up the bitmap. */
    if ( _bdf_strncmp( line, "BITMAP", 6 ) == 0 )
    {
      unsigned long  bitmap_size;


      if ( !( p->flags & BDF_BBX_ ) )
      {
        /* Missing BBX field. */
        FT_ERROR(( "_bdf_parse_glyphs: " ERRMSG1, lineno, "BBX" ));
        error = FT_THROW( Missing_Bbx_Field );
        goto Exit;
      }

      /* Allocate enough space for the bitmap. */
      glyph->bpr = ( glyph->bbx.width * p->font->bpp + 7 ) >> 3;

      bitmap_size = glyph->bpr * glyph->bbx.height;
      if ( glyph->bpr > 0xFFFFU || bitmap_size > 0xFFFFU )
      {
        FT_ERROR(( "_bdf_parse_glyphs: " ERRMSG4, lineno ));
        error = FT_THROW( Bbx_Too_Big );
        goto Exit;
      }
      else
        glyph->bytes = (unsigned short)bitmap_size;

      if ( FT_ALLOC( glyph->bitmap, glyph->bytes ) )
        goto Exit;

      p->row    = 0;
      p->flags |= BDF_BITMAP_;

      goto Exit;
    }

    FT_ERROR(( "_bdf_parse_glyphs: " ERRMSG9, lineno ));
    error = FT_THROW( Invalid_File_Format );
    goto Exit;

  Missing_Encoding:
    /* Missing ENCODING field. */
    FT_ERROR(( "_bdf_parse_glyphs: " ERRMSG1, lineno, "ENCODING" ));
    error = FT_THROW( Missing_Encoding_Field );

  Exit:
    if ( error && ( p->flags & BDF_GLYPH_ ) )
      FT_FREE( p->glyph_name );

    return error;
  }


  /* Load the font properties. */
  static FT_Error
  _bdf_parse_properties( char*          line,
                         unsigned long  linelen,
                         unsigned long  lineno,
                         void*          call_data,
                         void*          client_data )
  {
    unsigned long      vlen;
    _bdf_line_func_t*  next;
    _bdf_parse_t*      p;
    char*              name;
    char*              value;
    char               nbuf[128];
    FT_Error           error = FT_Err_Ok;

    FT_UNUSED( lineno );


    next = (_bdf_line_func_t *)call_data;
    p    = (_bdf_parse_t *)    client_data;

    /* Check for the end of the properties. */
    if ( _bdf_strncmp( line, "ENDPROPERTIES", 13 ) == 0 )
    {
      /* If the FONT_ASCENT or FONT_DESCENT properties have not been      */
      /* encountered yet, then make sure they are added as properties and */
      /* make sure they are set from the font bounding box info.          */
      /*                                                                  */
      /* This is *always* done regardless of the options, because X11     */
      /* requires these two fields to compile fonts.                      */
      if ( bdf_get_font_property( p->font, "FONT_ASCENT" ) == 0 )
      {
        p->font->font_ascent = p->font->bbx.ascent;
        ft_sprintf( nbuf, "%hd", p->font->bbx.ascent );
        error = _bdf_add_property( p->font, "FONT_ASCENT",
                                   nbuf, lineno );
        if ( error )
          goto Exit;

        FT_TRACE2(( "_bdf_parse_properties: " ACMSG1, p->font->bbx.ascent ));
      }

      if ( bdf_get_font_property( p->font, "FONT_DESCENT" ) == 0 )
      {
        p->font->font_descent = p->font->bbx.descent;
        ft_sprintf( nbuf, "%hd", p->font->bbx.descent );
        error = _bdf_add_property( p->font, "FONT_DESCENT",
                                   nbuf, lineno );
        if ( error )
          goto Exit;

        FT_TRACE2(( "_bdf_parse_properties: " ACMSG2, p->font->bbx.descent ));
      }

      p->flags &= ~BDF_PROPS_;
      *next     = _bdf_parse_glyphs;

      goto Exit;
    }

    /* Ignore the _XFREE86_GLYPH_RANGES properties. */
    if ( _bdf_strncmp( line, "_XFREE86_GLYPH_RANGES", 21 ) == 0 )
      goto Exit;

    /* Handle COMMENT fields and properties in a special way to preserve */
    /* the spacing.                                                      */
    if ( _bdf_strncmp( line, "COMMENT", 7 ) == 0 )
    {
      name = value = line;
      value += 7;
      if ( *value )
        *value++ = 0;
      error = _bdf_add_property( p->font, name, value, lineno );
      if ( error )
        goto Exit;
    }
    else if ( _bdf_is_atom( line, linelen, &name, &value, p->font ) )
    {
      error = _bdf_add_property( p->font, name, value, lineno );
      if ( error )
        goto Exit;
    }
    else
    {
      error = _bdf_list_split( &p->list, " +", line, linelen );
      if ( error )
        goto Exit;
      name = p->list.field[0];

      _bdf_list_shift( &p->list, 1 );
      value = _bdf_list_join( &p->list, ' ', &vlen );

      error = _bdf_add_property( p->font, name, value, lineno );
      if ( error )
        goto Exit;
    }

  Exit:
    return error;
  }


  /* Load the font header. */
  static FT_Error
  _bdf_parse_start( char*          line,
                    unsigned long  linelen,
                    unsigned long  lineno,
                    void*          call_data,
                    void*          client_data )
  {
    unsigned long      slen;
    _bdf_line_func_t*  next;
    _bdf_parse_t*      p;
    bdf_font_t*        font;
    char               *s;

    FT_Memory          memory = NULL;
    FT_Error           error  = FT_Err_Ok;

    FT_UNUSED( lineno );            /* only used in debug mode */


    next = (_bdf_line_func_t *)call_data;
    p    = (_bdf_parse_t *)    client_data;

    if ( p->font )
      memory = p->font->memory;

    /* Check for a comment.  This is done to handle those fonts that have */
    /* comments before the STARTFONT line for some reason.                */
    if ( _bdf_strncmp( line, "COMMENT", 7 ) == 0 )
    {
      if ( p->opts->keep_comments && p->font )
      {
        linelen -= 7;

        s = line + 7;
        if ( *s != 0 )
        {
          s++;
          linelen--;
        }
        error = _bdf_add_comment( p->font, s, linelen );
      }
      goto Exit;
    }

    if ( !( p->flags & BDF_START_ ) )
    {
      memory = p->memory;

      if ( _bdf_strncmp( line, "STARTFONT", 9 ) != 0 )
      {
        /* we don't emit an error message since this code gets */
        /* explicitly caught one level higher                  */
        error = FT_THROW( Missing_Startfont_Field );
        goto Exit;
      }

      p->flags = BDF_START_;
      font = p->font = NULL;

      if ( FT_NEW( font ) )
        goto Exit;
      p->font = font;

      font->memory = p->memory;

      { /* setup */
        size_t           i;
        bdf_property_t*  prop;


        error = ft_hash_str_init( &(font->proptbl), memory );
        if ( error )
          goto Exit;
        for ( i = 0, prop = (bdf_property_t*)_bdf_properties;
              i < _num_bdf_properties; i++, prop++ )
        {
          error = ft_hash_str_insert( prop->name, i,
                                      &(font->proptbl), memory );
          if ( error )
            goto Exit;
        }
      }

      if ( FT_QALLOC( p->font->internal, sizeof ( FT_HashRec ) ) )
        goto Exit;
      error = ft_hash_str_init( (FT_Hash)p->font->internal, memory );
      if ( error )
        goto Exit;
      p->font->spacing      = p->opts->font_spacing;
      p->font->default_char = ~0UL;

      goto Exit;
    }

    /* Check for the start of the properties. */
    if ( _bdf_strncmp( line, "STARTPROPERTIES", 15 ) == 0 )
    {
      if ( !( p->flags & BDF_FONT_BBX_ ) )
      {
        /* Missing the FONTBOUNDINGBOX field. */
        FT_ERROR(( "_bdf_parse_start: " ERRMSG1, lineno, "FONTBOUNDINGBOX" ));
        error = FT_THROW( Missing_Fontboundingbox_Field );
        goto Exit;
      }

      error = _bdf_list_split( &p->list, " +", line, linelen );
      if ( error )
        goto Exit;

      /* at this point, `p->font' can't be NULL */
      p->cnt = p->font->props_size = _bdf_atoul( p->list.field[1] );
      /* We need at least 4 bytes per property. */
      if ( p->cnt > p->size / 4 )
      {
        p->font->props_size = 0;

        FT_ERROR(( "_bdf_parse_glyphs: " ERRMSG5, lineno, "STARTPROPERTIES" ));
        error = FT_THROW( Invalid_Argument );
        goto Exit;
      }

      if ( FT_NEW_ARRAY( p->font->props, p->cnt ) )
      {
        p->font->props_size = 0;
        goto Exit;
      }

      p->flags |= BDF_PROPS_;
      *next     = _bdf_parse_properties;

      goto Exit;
    }

    /* Check for the FONTBOUNDINGBOX field. */
    if ( _bdf_strncmp( line, "FONTBOUNDINGBOX", 15 ) == 0 )
    {
      if ( !( p->flags & BDF_SIZE_ ) )
      {
        /* Missing the SIZE field. */
        FT_ERROR(( "_bdf_parse_start: " ERRMSG1, lineno, "SIZE" ));
        error = FT_THROW( Missing_Size_Field );
        goto Exit;
      }

      error = _bdf_list_split( &p->list, " +", line, linelen );
      if ( error )
        goto Exit;

      p->font->bbx.width  = _bdf_atous( p->list.field[1] );
      p->font->bbx.height = _bdf_atous( p->list.field[2] );

      p->font->bbx.x_offset = _bdf_atos( p->list.field[3] );
      p->font->bbx.y_offset = _bdf_atos( p->list.field[4] );

      p->font->bbx.ascent  = (short)( p->font->bbx.height +
                                      p->font->bbx.y_offset );

      p->font->bbx.descent = (short)( -p->font->bbx.y_offset );

      p->flags |= BDF_FONT_BBX_;

      goto Exit;
    }

    /* The next thing to check for is the FONT field. */
    if ( _bdf_strncmp( line, "FONT", 4 ) == 0 )
    {
      error = _bdf_list_split( &p->list, " +", line, linelen );
      if ( error )
        goto Exit;
      _bdf_list_shift( &p->list, 1 );

      s = _bdf_list_join( &p->list, ' ', &slen );

      if ( !s )
      {
        FT_ERROR(( "_bdf_parse_start: " ERRMSG8, lineno, "FONT" ));
        error = FT_THROW( Invalid_File_Format );
        goto Exit;
      }

      /* Allowing multiple `FONT' lines (which is invalid) doesn't hurt... */
      FT_FREE( p->font->name );

      if ( FT_QALLOC( p->font->name, slen + 1 ) )
        goto Exit;
      FT_MEM_COPY( p->font->name, s, slen + 1 );

      /* If the font name is an XLFD name, set the spacing to the one in  */
      /* the font name.  If there is no spacing fall back on the default. */
      error = _bdf_set_default_spacing( p->font, p->opts, lineno );
      if ( error )
        goto Exit;

      p->flags |= BDF_FONT_NAME_;

      goto Exit;
    }

    /* Check for the SIZE field. */
    if ( _bdf_strncmp( line, "SIZE", 4 ) == 0 )
    {
      if ( !( p->flags & BDF_FONT_NAME_ ) )
      {
        /* Missing the FONT field. */
        FT_ERROR(( "_bdf_parse_start: " ERRMSG1, lineno, "FONT" ));
        error = FT_THROW( Missing_Font_Field );
        goto Exit;
      }

      error = _bdf_list_split( &p->list, " +", line, linelen );
      if ( error )
        goto Exit;

      p->font->point_size   = _bdf_atoul( p->list.field[1] );
      p->font->resolution_x = _bdf_atoul( p->list.field[2] );
      p->font->resolution_y = _bdf_atoul( p->list.field[3] );

      /* Check for the bits per pixel field. */
      if ( p->list.used == 5 )
      {
        unsigned short bpp;


        bpp = _bdf_atous( p->list.field[4] );

        /* Only values 1, 2, 4, 8 are allowed for greymap fonts. */
        if ( bpp > 4 )
          p->font->bpp = 8;
        else if ( bpp > 2 )
          p->font->bpp = 4;
        else if ( bpp > 1 )
          p->font->bpp = 2;
        else
          p->font->bpp = 1;

        if ( p->font->bpp != bpp )
          FT_TRACE2(( "_bdf_parse_start: " ACMSG11, p->font->bpp ));
      }
      else
        p->font->bpp = 1;

      p->flags |= BDF_SIZE_;

      goto Exit;
    }

    /* Check for the CHARS field -- font properties are optional */
    if ( _bdf_strncmp( line, "CHARS", 5 ) == 0 )
    {
      char  nbuf[128];


      if ( !( p->flags & BDF_FONT_BBX_ ) )
      {
        /* Missing the FONTBOUNDINGBOX field. */
        FT_ERROR(( "_bdf_parse_start: " ERRMSG1, lineno, "FONTBOUNDINGBOX" ));
        error = FT_THROW( Missing_Fontboundingbox_Field );
        goto Exit;
      }

      /* Add the two standard X11 properties which are required */
      /* for compiling fonts.                                   */
      p->font->font_ascent = p->font->bbx.ascent;
      ft_sprintf( nbuf, "%hd", p->font->bbx.ascent );
      error = _bdf_add_property( p->font, "FONT_ASCENT",
                                 nbuf, lineno );
      if ( error )
        goto Exit;
      FT_TRACE2(( "_bdf_parse_properties: " ACMSG1, p->font->bbx.ascent ));

      p->font->font_descent = p->font->bbx.descent;
      ft_sprintf( nbuf, "%hd", p->font->bbx.descent );
      error = _bdf_add_property( p->font, "FONT_DESCENT",
                                 nbuf, lineno );
      if ( error )
        goto Exit;
      FT_TRACE2(( "_bdf_parse_properties: " ACMSG2, p->font->bbx.descent ));

      *next = _bdf_parse_glyphs;

      /* A special return value. */
      error = -1;
      goto Exit;
    }

    FT_ERROR(( "_bdf_parse_start: " ERRMSG9, lineno ));
    error = FT_THROW( Invalid_File_Format );

  Exit:
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
                 bdf_options_t*  opts,
                 bdf_font_t*    *font )
  {
    unsigned long  lineno = 0; /* make compiler happy */
    _bdf_parse_t   *p     = NULL;

    FT_Error  error = FT_Err_Ok;


    if ( FT_NEW( p ) )
      goto Exit;

    p->opts   = (bdf_options_t*)( opts ? opts : &_bdf_opts );
    p->minlb  = 32767;
    p->size   = stream->size;
    p->memory = memory;  /* only during font creation */

    _bdf_list_init( &p->list, memory );

    error = _bdf_readstream( stream, _bdf_parse_start,
                             (void *)p, &lineno );
    if ( error )
      goto Fail;

    if ( p->font )
    {
      /* If the font is not proportional, set the font's monowidth */
      /* field to the width of the font bounding box.              */

      if ( p->font->spacing != BDF_PROPORTIONAL )
        p->font->monowidth = p->font->bbx.width;

      /* If the number of glyphs loaded is not that of the original count, */
      /* indicate the difference.                                          */
      if ( p->cnt != p->font->glyphs_used + p->font->unencoded_used )
      {
        FT_TRACE2(( "bdf_load_font: " ACMSG15, p->cnt,
                    p->font->glyphs_used + p->font->unencoded_used ));
      }

      /* Once the font has been loaded, adjust the overall font metrics if */
      /* necessary.                                                        */
      if ( p->opts->correct_metrics != 0 &&
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

        if ( p->flags & BDF_SWIDTH_ADJ_ )
          FT_TRACE2(( "bdf_load_font: " ACMSG8 ));
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
      _bdf_list_done( &p->list );

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
      ft_hash_str_free( (FT_Hash)font->internal, memory );
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

    propid = ft_hash_str_lookup( name, (FT_Hash)font->internal );

    return propid ? ( font->props + *propid ) : 0;
  }


/* END */
