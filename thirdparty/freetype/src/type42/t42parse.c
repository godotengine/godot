/****************************************************************************
 *
 * t42parse.c
 *
 *   Type 42 font parser (body).
 *
 * Copyright (C) 2002-2023 by
 * Roberto Alameda.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#include "t42parse.h"
#include "t42error.h"
#include <freetype/internal/ftdebug.h>
#include <freetype/internal/ftstream.h>
#include <freetype/internal/psaux.h>


  /**************************************************************************
   *
   * The macro FT_COMPONENT is used in trace mode.  It is an implicit
   * parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log
   * messages during execution.
   */
#undef  FT_COMPONENT
#define FT_COMPONENT  t42


  static void
  t42_parse_font_matrix( T42_Face    face,
                         T42_Loader  loader );
  static void
  t42_parse_encoding( T42_Face    face,
                      T42_Loader  loader );

  static void
  t42_parse_charstrings( T42_Face    face,
                         T42_Loader  loader );

  static void
  t42_parse_sfnts( T42_Face    face,
                   T42_Loader  loader );


  /* as Type42 fonts have no Private dict,         */
  /* we set the last argument of T1_FIELD_XXX to 0 */
  static const
  T1_FieldRec  t42_keywords[] =
  {

#undef  FT_STRUCTURE
#define FT_STRUCTURE  T1_FontInfo
#undef  T1CODE
#define T1CODE        T1_FIELD_LOCATION_FONT_INFO

    T1_FIELD_STRING( "version",            version,             0 )
    T1_FIELD_STRING( "Notice",             notice,              0 )
    T1_FIELD_STRING( "FullName",           full_name,           0 )
    T1_FIELD_STRING( "FamilyName",         family_name,         0 )
    T1_FIELD_STRING( "Weight",             weight,              0 )
    T1_FIELD_NUM   ( "ItalicAngle",        italic_angle,        0 )
    T1_FIELD_BOOL  ( "isFixedPitch",       is_fixed_pitch,      0 )
    T1_FIELD_NUM   ( "UnderlinePosition",  underline_position,  0 )
    T1_FIELD_NUM   ( "UnderlineThickness", underline_thickness, 0 )

#undef  FT_STRUCTURE
#define FT_STRUCTURE  PS_FontExtraRec
#undef  T1CODE
#define T1CODE        T1_FIELD_LOCATION_FONT_EXTRA

    T1_FIELD_NUM   ( "FSType",             fs_type,             0 )

#undef  FT_STRUCTURE
#define FT_STRUCTURE  T1_FontRec
#undef  T1CODE
#define T1CODE        T1_FIELD_LOCATION_FONT_DICT

    T1_FIELD_KEY  ( "FontName",    font_name,    0 )
    T1_FIELD_NUM  ( "PaintType",   paint_type,   0 )
    T1_FIELD_NUM  ( "FontType",    font_type,    0 )
    T1_FIELD_FIXED( "StrokeWidth", stroke_width, 0 )

#undef  FT_STRUCTURE
#define FT_STRUCTURE  FT_BBox
#undef  T1CODE
#define T1CODE        T1_FIELD_LOCATION_BBOX

    T1_FIELD_BBOX( "FontBBox", xMin, 0 )

    T1_FIELD_CALLBACK( "FontMatrix",  t42_parse_font_matrix, 0 )
    T1_FIELD_CALLBACK( "Encoding",    t42_parse_encoding,    0 )
    T1_FIELD_CALLBACK( "CharStrings", t42_parse_charstrings, 0 )
    T1_FIELD_CALLBACK( "sfnts",       t42_parse_sfnts,       0 )

    { 0, T1_FIELD_LOCATION_CID_INFO, T1_FIELD_TYPE_NONE, 0, 0, 0, 0, 0, 0 }
  };


#define T1_Add_Table( p, i, o, l )  (p)->funcs.add( (p), i, o, l )
#define T1_Release_Table( p )          \
          do                           \
          {                            \
            if ( (p)->funcs.release )  \
              (p)->funcs.release( p ); \
          } while ( 0 )

#define T1_Skip_Spaces( p )    (p)->root.funcs.skip_spaces( &(p)->root )
#define T1_Skip_PS_Token( p )  (p)->root.funcs.skip_PS_token( &(p)->root )

#define T1_ToInt( p )                          \
          (p)->root.funcs.to_int( &(p)->root )
#define T1_ToBytes( p, b, m, n, d )                          \
          (p)->root.funcs.to_bytes( &(p)->root, b, m, n, d )

#define T1_ToFixedArray( p, m, f, t )                           \
          (p)->root.funcs.to_fixed_array( &(p)->root, m, f, t )
#define T1_ToToken( p, t )                          \
          (p)->root.funcs.to_token( &(p)->root, t )

#define T1_Load_Field( p, f, o, m, pf )                         \
          (p)->root.funcs.load_field( &(p)->root, f, o, m, pf )
#define T1_Load_Field_Table( p, f, o, m, pf )                         \
          (p)->root.funcs.load_field_table( &(p)->root, f, o, m, pf )


  /********************* Parsing Functions ******************/

  FT_LOCAL_DEF( FT_Error )
  t42_parser_init( T42_Parser     parser,
                   FT_Stream      stream,
                   FT_Memory      memory,
                   PSAux_Service  psaux )
  {
    FT_Error  error = FT_Err_Ok;
    FT_Long   size;


    psaux->ps_parser_funcs->init( &parser->root, NULL, NULL, memory );

    parser->stream    = stream;
    parser->base_len  = 0;
    parser->base_dict = NULL;
    parser->in_memory = 0;

    /********************************************************************
     *
     * Here a short summary of what is going on:
     *
     *   When creating a new Type 42 parser, we try to locate and load
     *   the base dictionary, loading the whole font into memory.
     *
     *   When `loading' the base dictionary, we only set up pointers
     *   in the case of a memory-based stream.  Otherwise, we allocate
     *   and load the base dictionary in it.
     *
     *   parser->in_memory is set if we have a memory stream.
     */

    if ( FT_STREAM_SEEK( 0L ) ||
         FT_FRAME_ENTER( 17 ) )
      goto Exit;

    if ( ft_memcmp( stream->cursor, "%!PS-TrueTypeFont", 17 ) != 0 )
    {
      FT_TRACE2(( "  not a Type42 font\n" ));
      error = FT_THROW( Unknown_File_Format );
    }

    FT_FRAME_EXIT();

    if ( error || FT_STREAM_SEEK( 0 ) )
      goto Exit;

    size = (FT_Long)stream->size;

    /* now, try to load `size' bytes of the `base' dictionary we */
    /* found previously                                          */

    /* if it is a memory-based resource, set up pointers */
    if ( !stream->read )
    {
      parser->base_dict = (FT_Byte*)stream->base + stream->pos;
      parser->base_len  = size;
      parser->in_memory = 1;

      /* check that the `size' field is valid */
      if ( FT_STREAM_SKIP( size ) )
        goto Exit;
    }
    else
    {
      /* read segment in memory */
      if ( FT_QALLOC( parser->base_dict, size )      ||
           FT_STREAM_READ( parser->base_dict, size ) )
        goto Exit;

      parser->base_len = size;
    }

    parser->root.base   = parser->base_dict;
    parser->root.cursor = parser->base_dict;
    parser->root.limit  = parser->root.cursor + parser->base_len;

  Exit:
    if ( error && !parser->in_memory )
      FT_FREE( parser->base_dict );

    return error;
  }


  FT_LOCAL_DEF( void )
  t42_parser_done( T42_Parser  parser )
  {
    FT_Memory  memory = parser->root.memory;


    /* free the base dictionary only when we have a disk stream */
    if ( !parser->in_memory )
      FT_FREE( parser->base_dict );

    if ( parser->root.funcs.done )
      parser->root.funcs.done( &parser->root );
  }


  static int
  t42_is_space( FT_Byte  c )
  {
    return ( c == ' '  || c == '\t'              ||
             c == '\r' || c == '\n' || c == '\f' ||
             c == '\0'                           );
  }


  static void
  t42_parse_font_matrix( T42_Face    face,
                         T42_Loader  loader )
  {
    T42_Parser  parser = &loader->parser;
    FT_Matrix*  matrix = &face->type1.font_matrix;
    FT_Vector*  offset = &face->type1.font_offset;
    FT_Fixed    temp[6];
    FT_Fixed    temp_scale;
    FT_Int      result;


    result = T1_ToFixedArray( parser, 6, temp, 0 );

    if ( result < 6 )
    {
      parser->root.error = FT_THROW( Invalid_File_Format );
      return;
    }

    temp_scale = FT_ABS( temp[3] );

    if ( temp_scale == 0 )
    {
      FT_ERROR(( "t42_parse_font_matrix: invalid font matrix\n" ));
      parser->root.error = FT_THROW( Invalid_File_Format );
      return;
    }

    /* atypical case */
    if ( temp_scale != 0x10000L )
    {
      temp[0] = FT_DivFix( temp[0], temp_scale );
      temp[1] = FT_DivFix( temp[1], temp_scale );
      temp[2] = FT_DivFix( temp[2], temp_scale );
      temp[4] = FT_DivFix( temp[4], temp_scale );
      temp[5] = FT_DivFix( temp[5], temp_scale );
      temp[3] = temp[3] < 0 ? -0x10000L : 0x10000L;
    }

    matrix->xx = temp[0];
    matrix->yx = temp[1];
    matrix->xy = temp[2];
    matrix->yy = temp[3];

    if ( !FT_Matrix_Check( matrix ) )
    {
      FT_ERROR(( "t42_parse_font_matrix: invalid font matrix\n" ));
      parser->root.error = FT_THROW( Invalid_File_Format );
      return;
    }

    /* note that the offsets must be expressed in integer font units */
    offset->x = temp[4] >> 16;
    offset->y = temp[5] >> 16;
  }


  static void
  t42_parse_encoding( T42_Face    face,
                      T42_Loader  loader )
  {
    T42_Parser  parser = &loader->parser;
    FT_Byte*    cur;
    FT_Byte*    limit  = parser->root.limit;

    PSAux_Service  psaux  = (PSAux_Service)face->psaux;


    T1_Skip_Spaces( parser );
    cur = parser->root.cursor;
    if ( cur >= limit )
    {
      FT_ERROR(( "t42_parse_encoding: out of bounds\n" ));
      parser->root.error = FT_THROW( Invalid_File_Format );
      return;
    }

    /* if we have a number or `[', the encoding is an array, */
    /* and we must load it now                               */
    if ( ft_isdigit( *cur ) || *cur == '[' )
    {
      T1_Encoding  encode          = &face->type1.encoding;
      FT_Int       count, n;
      PS_Table     char_table      = &loader->encoding_table;
      FT_Memory    memory          = parser->root.memory;
      FT_Error     error;
      FT_Bool      only_immediates = 0;


      /* read the number of entries in the encoding; should be 256 */
      if ( *cur == '[' )
      {
        count           = 256;
        only_immediates = 1;
        parser->root.cursor++;
      }
      else
        count = (FT_Int)T1_ToInt( parser );

      /* only composite fonts (which we don't support) */
      /* can have larger values                        */
      if ( count > 256 )
      {
        FT_ERROR(( "t42_parse_encoding: invalid encoding array size\n" ));
        parser->root.error = FT_THROW( Invalid_File_Format );
        return;
      }

      T1_Skip_Spaces( parser );
      if ( parser->root.cursor >= limit )
        return;

      /* PostScript happily allows overwriting of encoding arrays */
      if ( encode->char_index )
      {
        FT_FREE( encode->char_index );
        FT_FREE( encode->char_name );
        T1_Release_Table( char_table );
      }

      /* we use a T1_Table to store our charnames */
      loader->num_chars = encode->num_chars = count;
      if ( FT_QNEW_ARRAY( encode->char_index, count )    ||
           FT_QNEW_ARRAY( encode->char_name,  count )    ||
           FT_SET_ERROR( psaux->ps_table_funcs->init(
                           char_table, count, memory ) ) )
      {
        parser->root.error = error;
        return;
      }

      /* We need to `zero' out encoding_table.elements */
      for ( n = 0; n < count; n++ )
        (void)T1_Add_Table( char_table, n, ".notdef", 8 );

      /* Now we need to read records of the form                */
      /*                                                        */
      /*   ... charcode /charname ...                           */
      /*                                                        */
      /* for each entry in our table.                           */
      /*                                                        */
      /* We simply look for a number followed by an immediate   */
      /* name.  Note that this ignores correctly the sequence   */
      /* that is often seen in type42 fonts:                    */
      /*                                                        */
      /*   0 1 255 { 1 index exch /.notdef put } for dup        */
      /*                                                        */
      /* used to clean the encoding array before anything else. */
      /*                                                        */
      /* Alternatively, if the array is directly given as       */
      /*                                                        */
      /*   /Encoding [ ... ]                                    */
      /*                                                        */
      /* we only read immediates.                               */

      n = 0;
      T1_Skip_Spaces( parser );

      while ( parser->root.cursor < limit )
      {
        cur = parser->root.cursor;

        /* we stop when we encounter `def' or `]' */
        if ( *cur == 'd' && cur + 3 < limit )
        {
          if ( cur[1] == 'e'          &&
               cur[2] == 'f'          &&
               t42_is_space( cur[3] ) )
          {
            FT_TRACE6(( "encoding end\n" ));
            cur += 3;
            break;
          }
        }
        if ( *cur == ']' )
        {
          FT_TRACE6(( "encoding end\n" ));
          cur++;
          break;
        }

        /* check whether we have found an entry */
        if ( ft_isdigit( *cur ) || only_immediates )
        {
          FT_Int  charcode;


          if ( only_immediates )
            charcode = n;
          else
          {
            charcode = (FT_Int)T1_ToInt( parser );
            T1_Skip_Spaces( parser );

            /* protect against invalid charcode */
            if ( cur == parser->root.cursor )
            {
              parser->root.error = FT_THROW( Unknown_File_Format );
              return;
            }
          }

          cur = parser->root.cursor;

          if ( cur + 2 < limit && *cur == '/' && n < count )
          {
            FT_UInt  len;


            cur++;

            parser->root.cursor = cur;
            T1_Skip_PS_Token( parser );
            if ( parser->root.cursor >= limit )
              return;
            if ( parser->root.error )
              return;

            len = (FT_UInt)( parser->root.cursor - cur );

            parser->root.error = T1_Add_Table( char_table, charcode,
                                               cur, len + 1 );
            if ( parser->root.error )
              return;
            char_table->elements[charcode][len] = '\0';

            n++;
          }
          else if ( only_immediates )
          {
            /* Since the current position is not updated for           */
            /* immediates-only mode we would get an infinite loop if   */
            /* we don't do anything here.                              */
            /*                                                         */
            /* This encoding array is not valid according to the       */
            /* type42 specification (it might be an encoding for a CID */
            /* type42 font, however), so we conclude that this font is */
            /* NOT a type42 font.                                      */
            parser->root.error = FT_THROW( Unknown_File_Format );
            return;
          }
        }
        else
        {
          T1_Skip_PS_Token( parser );
          if ( parser->root.error )
            return;
        }

        T1_Skip_Spaces( parser );
      }

      face->type1.encoding_type = T1_ENCODING_TYPE_ARRAY;
      parser->root.cursor       = cur;
    }

    /* Otherwise, we should have either `StandardEncoding', */
    /* `ExpertEncoding', or `ISOLatin1Encoding'             */
    else
    {
      if ( cur + 17 < limit                                            &&
           ft_strncmp( (const char*)cur, "StandardEncoding", 16 ) == 0 )
        face->type1.encoding_type = T1_ENCODING_TYPE_STANDARD;

      else if ( cur + 15 < limit                                          &&
                ft_strncmp( (const char*)cur, "ExpertEncoding", 14 ) == 0 )
        face->type1.encoding_type = T1_ENCODING_TYPE_EXPERT;

      else if ( cur + 18 < limit                                             &&
                ft_strncmp( (const char*)cur, "ISOLatin1Encoding", 17 ) == 0 )
        face->type1.encoding_type = T1_ENCODING_TYPE_ISOLATIN1;

      else
        parser->root.error = FT_ERR( Ignore );
    }
  }


  typedef enum  T42_Load_Status_
  {
    BEFORE_START,
    BEFORE_TABLE_DIR,
    OTHER_TABLES

  } T42_Load_Status;


  static void
  t42_parse_sfnts( T42_Face    face,
                   T42_Loader  loader )
  {
    T42_Parser  parser = &loader->parser;
    FT_Memory   memory = parser->root.memory;
    FT_Byte*    cur;
    FT_Byte*    limit  = parser->root.limit;
    FT_Error    error;
    FT_Int      num_tables = 0;
    FT_Long     ttf_count;
    FT_Long     ttf_reserved;

    FT_ULong    n, string_size, old_string_size, real_size;
    FT_Byte*    string_buf = NULL;
    FT_Bool     allocated  = 0;

    T42_Load_Status  status;

    /** There should only be one sfnts array, but free any previous. */
    FT_FREE( face->ttf_data );
    face->ttf_size = 0;

    /* The format is                                */
    /*                                              */
    /*   /sfnts [ <hexstring> <hexstring> ... ] def */
    /*                                              */
    /* or                                           */
    /*                                              */
    /*   /sfnts [                                   */
    /*      <num_bin_bytes> RD <binary data>        */
    /*      <num_bin_bytes> RD <binary data>        */
    /*      ...                                     */
    /*   ] def                                      */
    /*                                              */
    /* with exactly one space after the `RD' token. */

    T1_Skip_Spaces( parser );

    if ( parser->root.cursor >= limit || *parser->root.cursor++ != '[' )
    {
      FT_ERROR(( "t42_parse_sfnts: can't find begin of sfnts vector\n" ));
      error = FT_THROW( Invalid_File_Format );
      goto Fail;
    }

    T1_Skip_Spaces( parser );
    status          = BEFORE_START;
    string_size     = 0;
    old_string_size = 0;
    ttf_count       = 0;
    ttf_reserved    = 12;
    if ( FT_QALLOC( face->ttf_data, ttf_reserved ) )
      goto Fail;

    FT_TRACE2(( "\n" ));
    FT_TRACE2(( "t42_parse_sfnts:\n" ));

    while ( parser->root.cursor < limit )
    {
      FT_ULong  size;


      cur = parser->root.cursor;

      if ( *cur == ']' )
      {
        parser->root.cursor++;
        face->ttf_size = ttf_count;
        goto Exit;
      }

      else if ( *cur == '<' )
      {
        if ( string_buf && !allocated )
        {
          FT_ERROR(( "t42_parse_sfnts: "
                     "can't handle mixed binary and hex strings\n" ));
          error = FT_THROW( Invalid_File_Format );
          goto Fail;
        }

        T1_Skip_PS_Token( parser );
        if ( parser->root.error )
          goto Exit;

        /* don't include delimiters */
        string_size = (FT_ULong)( ( parser->root.cursor - cur - 2 + 1 ) / 2 );
        if ( !string_size )
        {
          FT_ERROR(( "t42_parse_sfnts: invalid data in sfnts array\n" ));
          error = FT_THROW( Invalid_File_Format );
          goto Fail;
        }
        if ( FT_QREALLOC( string_buf, old_string_size, string_size ) )
          goto Fail;

        allocated = 1;

        parser->root.cursor = cur;
        (void)T1_ToBytes( parser, string_buf, string_size, &real_size, 1 );
        old_string_size = string_size;
        string_size     = real_size;
      }

      else if ( ft_isdigit( *cur ) )
      {
        FT_Long  tmp;


        if ( allocated )
        {
          FT_ERROR(( "t42_parse_sfnts: "
                     "can't handle mixed binary and hex strings\n" ));
          error = FT_THROW( Invalid_File_Format );
          goto Fail;
        }

        tmp = T1_ToInt( parser );
        if ( tmp < 0 )
        {
          FT_ERROR(( "t42_parse_sfnts: invalid string size\n" ));
          error = FT_THROW( Invalid_File_Format );
          goto Fail;
        }
        else
          string_size = (FT_ULong)tmp;

        T1_Skip_PS_Token( parser );             /* `RD' */
        if ( parser->root.error )
          return;

        string_buf = parser->root.cursor + 1;   /* one space after `RD' */

        if ( (FT_ULong)( limit - parser->root.cursor ) <= string_size )
        {
          FT_ERROR(( "t42_parse_sfnts: too much binary data\n" ));
          error = FT_THROW( Invalid_File_Format );
          goto Fail;
        }
        else
          parser->root.cursor += string_size + 1;
      }

      if ( !string_buf )
      {
        FT_ERROR(( "t42_parse_sfnts: invalid data in sfnts array\n" ));
        error = FT_THROW( Invalid_File_Format );
        goto Fail;
      }

      /* A string can have a trailing zero (odd) byte for padding. */
      /* Ignore it.                                                */
      if ( ( string_size & 1 ) && string_buf[string_size - 1] == 0 )
        string_size--;

      if ( !string_size )
      {
        FT_ERROR(( "t42_parse_sfnts: invalid string\n" ));
        error = FT_THROW( Invalid_File_Format );
        goto Fail;
      }

      FT_TRACE2(( "  PS string size %5lu bytes, offset 0x%08lx (%lu)\n",
                  string_size, ttf_count, ttf_count ));

      /* The whole TTF is now loaded into `string_buf'.  We are */
      /* checking its contents while copying it to `ttf_data'.  */

      size = (FT_ULong)( limit - parser->root.cursor );

      for ( n = 0; n < string_size; n++ )
      {
        switch ( status )
        {
        case BEFORE_START:
          /* load offset table, 12 bytes */
          if ( ttf_count < 12 )
          {
            face->ttf_data[ttf_count++] = string_buf[n];
            continue;
          }
          else
          {
            FT_Long ttf_reserved_prev = ttf_reserved;


            num_tables   = 16 * face->ttf_data[4] + face->ttf_data[5];
            status       = BEFORE_TABLE_DIR;
            ttf_reserved = 12 + 16 * num_tables;

            FT_TRACE2(( "  SFNT directory contains %d tables\n",
                        num_tables ));

            if ( (FT_Long)size < ttf_reserved )
            {
              FT_ERROR(( "t42_parse_sfnts: invalid data in sfnts array\n" ));
              error = FT_THROW( Invalid_File_Format );
              goto Fail;
            }

            if ( FT_QREALLOC( face->ttf_data, ttf_reserved_prev,
                              ttf_reserved ) )
              goto Fail;
          }
          FALL_THROUGH;

        case BEFORE_TABLE_DIR:
          /* the offset table is read; read the table directory */
          if ( ttf_count < ttf_reserved )
          {
            face->ttf_data[ttf_count++] = string_buf[n];
            continue;
          }
          else
          {
            int       i;
            FT_ULong  len;
            FT_Long ttf_reserved_prev = ttf_reserved;


            FT_TRACE2(( "\n" ));
            FT_TRACE2(( "  table    length\n" ));
            FT_TRACE2(( "  ------------------------------\n" ));

            for ( i = 0; i < num_tables; i++ )
            {
              FT_Byte*  p = face->ttf_data + 12 + 16 * i + 12;


              len = FT_PEEK_ULONG( p );
              FT_TRACE2(( "   %4i  0x%08lx (%lu)\n", i, len, len ));

              if ( len > size                               ||
                   ttf_reserved > (FT_Long)( size - len ) )
              {
                FT_ERROR(( "t42_parse_sfnts:"
                           " invalid data in sfnts array\n" ));
                error = FT_THROW( Invalid_File_Format );
                goto Fail;
              }

              /* Pad to a 4-byte boundary length */
              ttf_reserved += (FT_Long)( ( len + 3 ) & ~3U );
            }
            ttf_reserved += 1;

            status = OTHER_TABLES;

            FT_TRACE2(( "\n" ));
            FT_TRACE2(( "  allocating %ld bytes\n", ttf_reserved ));
            FT_TRACE2(( "\n" ));

            if ( FT_QREALLOC( face->ttf_data, ttf_reserved_prev,
                              ttf_reserved ) )
              goto Fail;
          }
          FALL_THROUGH;

        case OTHER_TABLES:
          /* all other tables are just copied */
          if ( ttf_count >= ttf_reserved )
          {
            FT_ERROR(( "t42_parse_sfnts: too much binary data\n" ));
            error = FT_THROW( Invalid_File_Format );
            goto Fail;
          }
          face->ttf_data[ttf_count++] = string_buf[n];
        }
      }

      T1_Skip_Spaces( parser );
    }

    /* if control reaches this point, the format was not valid */
    error = FT_THROW( Invalid_File_Format );

  Fail:
    parser->root.error = error;

  Exit:
    if ( parser->root.error )
    {
      FT_FREE( face->ttf_data );
      face->ttf_size = 0;
    }
    if ( allocated )
      FT_FREE( string_buf );
  }


  static void
  t42_parse_charstrings( T42_Face    face,
                         T42_Loader  loader )
  {
    T42_Parser     parser       = &loader->parser;
    PS_Table       code_table   = &loader->charstrings;
    PS_Table       name_table   = &loader->glyph_names;
    PS_Table       swap_table   = &loader->swap_table;
    FT_Memory      memory       = parser->root.memory;
    FT_Error       error;

    PSAux_Service  psaux        = (PSAux_Service)face->psaux;

    FT_Byte*       cur;
    FT_Byte*       limit        = parser->root.limit;
    FT_Int         n;
    FT_Int         notdef_index = 0;
    FT_Byte        notdef_found = 0;


    T1_Skip_Spaces( parser );

    if ( parser->root.cursor >= limit )
    {
      FT_ERROR(( "t42_parse_charstrings: out of bounds\n" ));
      error = FT_THROW( Invalid_File_Format );
      goto Fail;
    }

    if ( ft_isdigit( *parser->root.cursor ) )
    {
      loader->num_glyphs = T1_ToInt( parser );
      if ( parser->root.error )
        return;
      if ( loader->num_glyphs < 0 )
      {
        FT_ERROR(( "t42_parse_encoding: invalid number of glyphs\n" ));
        error = FT_THROW( Invalid_File_Format );
        goto Fail;
      }

      /* we certainly need more than 4 bytes per glyph */
      if ( loader->num_glyphs > ( limit - parser->root.cursor ) >> 2 )
      {
        FT_TRACE0(( "t42_parse_charstrings: adjusting number of glyphs"
                    " (from %d to %ld)\n",
                    loader->num_glyphs,
                    ( limit - parser->root.cursor ) >> 2 ));
        loader->num_glyphs = ( limit - parser->root.cursor ) >> 2;
      }

    }
    else if ( *parser->root.cursor == '<' )
    {
      /* We have `<< ... >>'.  Count the number of `/' in the dictionary */
      /* to get its size.                                                */
      FT_Int  count = 0;


      T1_Skip_PS_Token( parser );
      if ( parser->root.error )
        return;
      T1_Skip_Spaces( parser );
      cur = parser->root.cursor;

      while ( parser->root.cursor < limit )
      {
        if ( *parser->root.cursor == '/' )
          count++;
        else if ( *parser->root.cursor == '>' )
        {
          loader->num_glyphs  = count;
          parser->root.cursor = cur;        /* rewind */
          break;
        }
        T1_Skip_PS_Token( parser );
        if ( parser->root.error )
          return;
        T1_Skip_Spaces( parser );
      }
    }
    else
    {
      FT_ERROR(( "t42_parse_charstrings: invalid token\n" ));
      error = FT_THROW( Invalid_File_Format );
      goto Fail;
    }

    if ( parser->root.cursor >= limit )
    {
      FT_ERROR(( "t42_parse_charstrings: out of bounds\n" ));
      error = FT_THROW( Invalid_File_Format );
      goto Fail;
    }

    /* initialize tables */

    /* contrary to Type1, we disallow multiple CharStrings arrays */
    if ( swap_table->init )
    {
      FT_ERROR(( "t42_parse_charstrings:"
                 " only one CharStrings array allowed\n" ));
      error = FT_THROW( Invalid_File_Format );
      goto Fail;
    }

    error = psaux->ps_table_funcs->init( code_table,
                                         loader->num_glyphs,
                                         memory );
    if ( error )
      goto Fail;

    error = psaux->ps_table_funcs->init( name_table,
                                         loader->num_glyphs,
                                         memory );
    if ( error )
      goto Fail;

    /* Initialize table for swapping index notdef_index and */
    /* index 0 names and codes (if necessary).              */

    error = psaux->ps_table_funcs->init( swap_table, 4, memory );
    if ( error )
      goto Fail;

    n = 0;

    for (;;)
    {
      /* We support two formats.                     */
      /*                                             */
      /*   `/glyphname' + index [+ `def']            */
      /*   `(glyphname)' [+ `cvn'] + index [+ `def'] */
      /*                                             */
      /* The latter format gets created by the       */
      /* LilyPond typesetting program.               */

      T1_Skip_Spaces( parser );

      cur = parser->root.cursor;
      if ( cur >= limit )
        break;

      /* We stop when we find an `end' keyword or '>' */
      if ( *cur   == 'e'          &&
           cur + 3 < limit        &&
           cur[1] == 'n'          &&
           cur[2] == 'd'          &&
           t42_is_space( cur[3] ) )
        break;
      if ( *cur == '>' )
        break;

      T1_Skip_PS_Token( parser );
      if ( parser->root.cursor >= limit )
      {
        FT_ERROR(( "t42_parse_charstrings: out of bounds\n" ));
        error = FT_THROW( Invalid_File_Format );
        goto Fail;
      }
      if ( parser->root.error )
        return;

      if ( *cur == '/' || *cur == '(' )
      {
        FT_UInt  len;
        FT_Bool  have_literal = FT_BOOL( *cur == '(' );


        if ( cur + ( have_literal ? 3 : 2 ) >= limit )
        {
          FT_ERROR(( "t42_parse_charstrings: out of bounds\n" ));
          error = FT_THROW( Invalid_File_Format );
          goto Fail;
        }

        cur++;                              /* skip `/' */
        len = (FT_UInt)( parser->root.cursor - cur );
        if ( have_literal )
          len--;

        error = T1_Add_Table( name_table, n, cur, len + 1 );
        if ( error )
          goto Fail;

        /* add a trailing zero to the name table */
        name_table->elements[n][len] = '\0';

        /* record index of /.notdef */
        if ( *cur == '.'                                                &&
             ft_strcmp( ".notdef",
                        (const char*)( name_table->elements[n] ) ) == 0 )
        {
          notdef_index = n;
          notdef_found = 1;
        }

        T1_Skip_Spaces( parser );

        if ( have_literal )
          T1_Skip_PS_Token( parser );

        cur = parser->root.cursor;

        (void)T1_ToInt( parser );
        if ( parser->root.cursor >= limit )
        {
          FT_ERROR(( "t42_parse_charstrings: out of bounds\n" ));
          error = FT_THROW( Invalid_File_Format );
          goto Fail;
        }

        len = (FT_UInt)( parser->root.cursor - cur );

        error = T1_Add_Table( code_table, n, cur, len + 1 );
        if ( error )
          goto Fail;

        code_table->elements[n][len] = '\0';

        n++;
        if ( n >= loader->num_glyphs )
          break;
      }
    }

    loader->num_glyphs = n;

    if ( !notdef_found )
    {
      FT_ERROR(( "t42_parse_charstrings: no /.notdef glyph\n" ));
      error = FT_THROW( Invalid_File_Format );
      goto Fail;
    }

    /* if /.notdef does not occupy index 0, do our magic. */
    if ( ft_strcmp( ".notdef", (const char*)name_table->elements[0] ) )
    {
      /* Swap glyph in index 0 with /.notdef glyph.  First, add index 0  */
      /* name and code entries to swap_table.  Then place notdef_index   */
      /* name and code entries into swap_table.  Then swap name and code */
      /* entries at indices notdef_index and 0 using values stored in    */
      /* swap_table.                                                     */

      /* Index 0 name */
      error = T1_Add_Table( swap_table, 0,
                            name_table->elements[0],
                            name_table->lengths [0] );
      if ( error )
        goto Fail;

      /* Index 0 code */
      error = T1_Add_Table( swap_table, 1,
                            code_table->elements[0],
                            code_table->lengths [0] );
      if ( error )
        goto Fail;

      /* Index notdef_index name */
      error = T1_Add_Table( swap_table, 2,
                            name_table->elements[notdef_index],
                            name_table->lengths [notdef_index] );
      if ( error )
        goto Fail;

      /* Index notdef_index code */
      error = T1_Add_Table( swap_table, 3,
                            code_table->elements[notdef_index],
                            code_table->lengths [notdef_index] );
      if ( error )
        goto Fail;

      error = T1_Add_Table( name_table, notdef_index,
                            swap_table->elements[0],
                            swap_table->lengths [0] );
      if ( error )
        goto Fail;

      error = T1_Add_Table( code_table, notdef_index,
                            swap_table->elements[1],
                            swap_table->lengths [1] );
      if ( error )
        goto Fail;

      error = T1_Add_Table( name_table, 0,
                            swap_table->elements[2],
                            swap_table->lengths [2] );
      if ( error )
        goto Fail;

      error = T1_Add_Table( code_table, 0,
                            swap_table->elements[3],
                            swap_table->lengths [3] );
      if ( error )
        goto Fail;

    }

    return;

  Fail:
    parser->root.error = error;
  }


  static FT_Error
  t42_load_keyword( T42_Face    face,
                    T42_Loader  loader,
                    T1_Field    field )
  {
    FT_Error  error;
    void*     dummy_object;
    void**    objects;
    FT_UInt   max_objects = 0;


    /* if the keyword has a dedicated callback, call it */
    if ( field->type == T1_FIELD_TYPE_CALLBACK )
    {
      field->reader( (FT_Face)face, loader );
      error = loader->parser.root.error;
      goto Exit;
    }

    /* now the keyword is either a simple field or a table of fields; */
    /* we are now going to take care of it                            */

    switch ( field->location )
    {
    case T1_FIELD_LOCATION_FONT_INFO:
      dummy_object = &face->type1.font_info;
      break;

    case T1_FIELD_LOCATION_FONT_EXTRA:
      dummy_object = &face->type1.font_extra;
      break;

    case T1_FIELD_LOCATION_BBOX:
      dummy_object = &face->type1.font_bbox;
      break;

    default:
      dummy_object = &face->type1;
    }

    objects = &dummy_object;

    if ( field->type == T1_FIELD_TYPE_INTEGER_ARRAY ||
         field->type == T1_FIELD_TYPE_FIXED_ARRAY   )
      error = T1_Load_Field_Table( &loader->parser, field,
                                   objects, max_objects, 0 );
    else
      error = T1_Load_Field( &loader->parser, field,
                             objects, max_objects, 0 );

   Exit:
    return error;
  }


  FT_LOCAL_DEF( FT_Error )
  t42_parse_dict( T42_Face    face,
                  T42_Loader  loader,
                  FT_Byte*    base,
                  FT_Long     size )
  {
    T42_Parser  parser     = &loader->parser;
    FT_Byte*    limit;
    FT_Int      n_keywords = (FT_Int)( sizeof ( t42_keywords ) /
                                         sizeof ( t42_keywords[0] ) );


    parser->root.cursor = base;
    parser->root.limit  = base + size;
    parser->root.error  = FT_Err_Ok;

    limit = parser->root.limit;

    T1_Skip_Spaces( parser );

    while ( parser->root.cursor < limit )
    {
      FT_Byte*  cur;


      cur = parser->root.cursor;

      /* look for `FontDirectory' which causes problems for some fonts */
      if ( *cur == 'F' && cur + 25 < limit                    &&
           ft_strncmp( (char*)cur, "FontDirectory", 13 ) == 0 )
      {
        FT_Byte*  cur2;


        /* skip the `FontDirectory' keyword */
        T1_Skip_PS_Token( parser );
        T1_Skip_Spaces  ( parser );
        cur = cur2 = parser->root.cursor;

        /* look up the `known' keyword */
        while ( cur < limit )
        {
          if ( *cur == 'k' && cur + 5 < limit             &&
                ft_strncmp( (char*)cur, "known", 5 ) == 0 )
            break;

          T1_Skip_PS_Token( parser );
          if ( parser->root.error )
            goto Exit;
          T1_Skip_Spaces  ( parser );
          cur = parser->root.cursor;
        }

        if ( cur < limit )
        {
          T1_TokenRec  token;


          /* skip the `known' keyword and the token following it */
          T1_Skip_PS_Token( parser );
          T1_ToToken( parser, &token );

          /* if the last token was an array, skip it! */
          if ( token.type == T1_TOKEN_TYPE_ARRAY )
            cur2 = parser->root.cursor;
        }
        parser->root.cursor = cur2;
      }

      /* look for immediates */
      else if ( *cur == '/' && cur + 2 < limit )
      {
        FT_UInt  len;


        cur++;

        parser->root.cursor = cur;
        T1_Skip_PS_Token( parser );
        if ( parser->root.error )
          goto Exit;

        len = (FT_UInt)( parser->root.cursor - cur );

        if ( len > 0 && len < 22 && parser->root.cursor < limit )
        {
          int  i;


          /* now compare the immediate name to the keyword table */

          /* loop through all known keywords */
          for ( i = 0; i < n_keywords; i++ )
          {
            T1_Field  keyword = (T1_Field)&t42_keywords[i];
            FT_Byte   *name   = (FT_Byte*)keyword->ident;


            if ( !name )
              continue;

            if ( cur[0] == name[0]                      &&
                 len == ft_strlen( (const char *)name ) &&
                 ft_memcmp( cur, name, len ) == 0       )
            {
              /* we found it -- run the parsing callback! */
              parser->root.error = t42_load_keyword( face,
                                                     loader,
                                                     keyword );
              if ( parser->root.error )
                return parser->root.error;
              break;
            }
          }
        }
      }
      else
      {
        T1_Skip_PS_Token( parser );
        if ( parser->root.error )
          goto Exit;
      }

      T1_Skip_Spaces( parser );
    }

  Exit:
    return parser->root.error;
  }


  FT_LOCAL_DEF( void )
  t42_loader_init( T42_Loader  loader,
                   T42_Face    face )
  {
    FT_UNUSED( face );

    FT_ZERO( loader );
    loader->num_glyphs = 0;
    loader->num_chars  = 0;

    /* initialize the tables -- simply set their `init' field to 0 */
    loader->encoding_table.init = 0;
    loader->charstrings.init    = 0;
    loader->glyph_names.init    = 0;
  }


  FT_LOCAL_DEF( void )
  t42_loader_done( T42_Loader  loader )
  {
    T42_Parser  parser = &loader->parser;


    /* finalize tables */
    T1_Release_Table( &loader->encoding_table );
    T1_Release_Table( &loader->charstrings );
    T1_Release_Table( &loader->glyph_names );
    T1_Release_Table( &loader->swap_table );

    /* finalize parser */
    t42_parser_done( parser );
  }


/* END */
