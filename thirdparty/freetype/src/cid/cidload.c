/****************************************************************************
 *
 * cidload.c
 *
 *   CID-keyed Type1 font loader (body).
 *
 * Copyright (C) 1996-2024 by
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
#include <freetype/internal/ftdebug.h>
#include FT_CONFIG_CONFIG_H
#include <freetype/ftmm.h>
#include <freetype/internal/t1types.h>
#include <freetype/internal/psaux.h>

#include "cidload.h"

#include "ciderrs.h"


  /**************************************************************************
   *
   * The macro FT_COMPONENT is used in trace mode.  It is an implicit
   * parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log
   * messages during execution.
   */
#undef  FT_COMPONENT
#define FT_COMPONENT  cidload


  /* read a single offset */
  FT_LOCAL_DEF( FT_ULong )
  cid_get_offset( FT_Byte*  *start,
                  FT_UInt    offsize )
  {
    FT_ULong  result;
    FT_Byte*  p = *start;


    for ( result = 0; offsize > 0; offsize-- )
    {
      result <<= 8;
      result  |= *p++;
    }

    *start = p;
    return result;
  }


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                    TYPE 1 SYMBOL PARSING                      *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/


  static FT_Error
  cid_load_keyword( CID_Face        face,
                    CID_Loader*     loader,
                    const T1_Field  keyword )
  {
    FT_Error      error;
    CID_Parser*   parser = &loader->parser;
    FT_Byte*      object;
    void*         dummy_object;
    CID_FaceInfo  cid = &face->cid;


    /* if the keyword has a dedicated callback, call it */
    if ( keyword->type == T1_FIELD_TYPE_CALLBACK )
    {
      FT_TRACE4(( "  %s", keyword->ident ));

      keyword->reader( (FT_Face)face, parser );
      error = parser->root.error;
      goto Exit;
    }

    /* we must now compute the address of our target object */
    switch ( keyword->location )
    {
    case T1_FIELD_LOCATION_CID_INFO:
      object = (FT_Byte*)cid;
      break;

    case T1_FIELD_LOCATION_FONT_INFO:
      object = (FT_Byte*)&cid->font_info;
      break;

    case T1_FIELD_LOCATION_FONT_EXTRA:
      object = (FT_Byte*)&face->font_extra;
      break;

    case T1_FIELD_LOCATION_BBOX:
      object = (FT_Byte*)&cid->font_bbox;
      break;

    default:
      {
        CID_FaceDict  dict;


        if ( parser->num_dict >= cid->num_dicts )
        {
          FT_ERROR(( "cid_load_keyword: invalid use of `%s'\n",
                     keyword->ident ));
          error = FT_THROW( Syntax_Error );
          goto Exit;
        }

        dict = cid->font_dicts + parser->num_dict;
        switch ( keyword->location )
        {
        case T1_FIELD_LOCATION_PRIVATE:
          object = (FT_Byte*)&dict->private_dict;
          break;

        default:
          object = (FT_Byte*)dict;
        }
      }
    }

    FT_TRACE4(( "  %s", keyword->ident ));

    dummy_object = object;

    /* now, load the keyword data in the object's field(s) */
    if ( keyword->type == T1_FIELD_TYPE_INTEGER_ARRAY ||
         keyword->type == T1_FIELD_TYPE_FIXED_ARRAY   )
      error = cid_parser_load_field_table( &loader->parser, keyword,
                                           &dummy_object );
    else
      error = cid_parser_load_field( &loader->parser,
                                     keyword, &dummy_object );

    FT_TRACE4(( "\n" ));

  Exit:
    return error;
  }


  FT_CALLBACK_DEF( void )
  cid_parse_font_matrix( FT_Face  face,     /* CID_Face */
                         void*    parser_ )
  {
    CID_Face      cidface = (CID_Face)face;
    CID_Parser*   parser  = (CID_Parser*)parser_;
    CID_FaceDict  dict;
    FT_Fixed      temp[6];
    FT_Fixed      temp_scale;


    if ( parser->num_dict < cidface->cid.num_dicts )
    {
      FT_Matrix*  matrix;
      FT_Vector*  offset;
      FT_Int      result;


      dict   = cidface->cid.font_dicts + parser->num_dict;
      matrix = &dict->font_matrix;
      offset = &dict->font_offset;

      /* input is scaled by 1000 to accommodate default FontMatrix */
      result = cid_parser_to_fixed_array( parser, 6, temp, 3 );

      if ( result < 6 )
      {
        FT_ERROR(( "cid_parse_font_matrix: not enough matrix elements\n" ));
        goto Exit;
      }

      FT_TRACE4(( " [%f %f %f %f %f %f]\n",
                  (double)temp[0] / 65536 / 1000,
                  (double)temp[1] / 65536 / 1000,
                  (double)temp[2] / 65536 / 1000,
                  (double)temp[3] / 65536 / 1000,
                  (double)temp[4] / 65536 / 1000,
                  (double)temp[5] / 65536 / 1000 ));

      temp_scale = FT_ABS( temp[3] );

      if ( temp_scale == 0 )
      {
        FT_ERROR(( "cid_parse_font_matrix: invalid font matrix\n" ));
        goto Exit;
      }

      /* atypical case */
      if ( temp_scale != 0x10000L )
      {
        /* set units per EM based on FontMatrix values */
        face->units_per_EM = (FT_UShort)FT_DivFix( 1000, temp_scale );

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
        FT_ERROR(( "t1_parse_font_matrix: invalid font matrix\n" ));
        parser->root.error = FT_THROW( Invalid_File_Format );
        goto Exit;
      }

      /* note that the font offsets are expressed in integer font units */
      offset->x  = temp[4] >> 16;
      offset->y  = temp[5] >> 16;
    }

  Exit:
    return;
  }


  FT_CALLBACK_DEF( void )
  parse_fd_array( FT_Face  face,     /* CID_Face */
                  void*    parser_ )
  {
    CID_Face      cidface = (CID_Face)face;
    CID_Parser*   parser  = (CID_Parser*)parser_;
    CID_FaceInfo  cid     = &cidface->cid;
    FT_Memory     memory  = FT_FACE_MEMORY( face );
    FT_Stream     stream  = parser->stream;
    FT_Error      error   = FT_Err_Ok;
    FT_Long       num_dicts, max_dicts;


    num_dicts = cid_parser_to_int( parser );
    if ( num_dicts < 0 || num_dicts > FT_INT_MAX )
    {
      FT_ERROR(( "parse_fd_array: invalid number of dictionaries\n" ));
      goto Exit;
    }

    FT_TRACE4(( " %ld\n", num_dicts ));

    /*
     * A single entry in the FDArray must (at least) contain the following
     * structure elements.
     *
     *   %ADOBeginFontDict              18
     *   X dict begin                   13
     *     /FontMatrix [X X X X]        22
     *     /Private X dict begin        22
     *     end                           4
     *   end                             4
     *   %ADOEndFontDict                16
     *
     * This needs 18+13+22+22+4+4+16=99 bytes or more.  Normally, you also
     * need a `dup X' at the very beginning and a `put' at the end, so a
     * rough guess using 100 bytes as the minimum is justified.
     */
    max_dicts = (FT_Long)( stream->size / 100 );
    if ( num_dicts > max_dicts )
    {
      FT_TRACE0(( "parse_fd_array: adjusting FDArray size"
                  " (from %ld to %ld)\n",
                  num_dicts, max_dicts ));
      num_dicts = max_dicts;
    }

    if ( !cid->font_dicts )
    {
      FT_UInt  n;


      if ( FT_NEW_ARRAY( cid->font_dicts, num_dicts ) )
        goto Exit;

      cid->num_dicts = num_dicts;

      /* set some default values (the same as for Type 1 fonts) */
      for ( n = 0; n < cid->num_dicts; n++ )
      {
        CID_FaceDict  dict = cid->font_dicts + n;


        dict->private_dict.blue_shift       = 7;
        dict->private_dict.blue_fuzz        = 1;
        dict->private_dict.lenIV            = 4;
        dict->private_dict.expansion_factor = (FT_Fixed)( 0.06 * 0x10000L );
        dict->private_dict.blue_scale       = (FT_Fixed)(
                                                0.039625 * 0x10000L * 1000 );
      }
    }

  Exit:
    return;
  }


  /* By mistake, `expansion_factor' appears both in PS_PrivateRec */
  /* and CID_FaceDictRec (both are public header files and can't  */
  /* be thus changed).  We simply copy the value.                 */

  FT_CALLBACK_DEF( void )
  parse_expansion_factor( FT_Face  face,    /* CID_Face */
                          void*    parser_ )
  {
    CID_Face      cidface = (CID_Face)face;
    CID_Parser*   parser  = (CID_Parser*)parser_;
    CID_FaceDict  dict;


    if ( parser->num_dict < cidface->cid.num_dicts )
    {
      dict = cidface->cid.font_dicts + parser->num_dict;

      dict->expansion_factor              = cid_parser_to_fixed( parser, 0 );
      dict->private_dict.expansion_factor = dict->expansion_factor;

      FT_TRACE4(( "%ld\n", dict->expansion_factor ));
    }

    return;
  }


  /* By mistake, `CID_FaceDictRec' doesn't contain a field for the */
  /* `FontName' keyword.  FreeType doesn't need it, but it is nice */
  /* to catch it for producing better trace output.                */

  FT_CALLBACK_DEF( void )
  parse_font_name( FT_Face  face,     /* CID_Face */
                   void*    parser_ )
  {
#ifdef FT_DEBUG_LEVEL_TRACE
    CID_Face      cidface = (CID_Face)face;
    CID_Parser*   parser  = (CID_Parser*)parser_;


    if ( parser->num_dict < cidface->cid.num_dicts )
    {
      T1_TokenRec  token;
      FT_UInt      len;


      cid_parser_to_token( parser, &token );

      len = (FT_UInt)( token.limit - token.start );
      if ( len )
        FT_TRACE4(( " %.*s\n", len, token.start ));
      else
        FT_TRACE4(( " <no value>\n" ));
    }
#else
    FT_UNUSED( face );
    FT_UNUSED( parser_ );
#endif

    return;
  }


  static
  const T1_FieldRec  cid_field_records[] =
  {

#include "cidtoken.h"

    T1_FIELD_CALLBACK( "FDArray",         parse_fd_array, 0 )
    T1_FIELD_CALLBACK( "FontMatrix",      cid_parse_font_matrix, 0 )
    T1_FIELD_CALLBACK( "ExpansionFactor", parse_expansion_factor, 0 )
    T1_FIELD_CALLBACK( "FontName",        parse_font_name, 0 )

    T1_FIELD_ZERO
  };


  static FT_Error
  cid_parse_dict( CID_Face     face,
                  CID_Loader*  loader,
                  FT_Byte*     base,
                  FT_ULong     size )
  {
    CID_Parser*  parser = &loader->parser;


    parser->root.cursor = base;
    parser->root.limit  = base + size;
    parser->root.error  = FT_Err_Ok;

    {
      FT_Byte*  cur   = base;
      FT_Byte*  limit = cur + size;


      for (;;)
      {
        FT_Byte*  newlimit;


        parser->root.cursor = cur;
        cid_parser_skip_spaces( parser );

        if ( parser->root.cursor >= limit )
          newlimit = limit - 1 - 17;
        else
          newlimit = parser->root.cursor - 17;

        /* look for `%ADOBeginFontDict' */
        for ( ; cur < newlimit; cur++ )
        {
          if ( *cur == '%'                                            &&
               ft_strncmp( (char*)cur, "%ADOBeginFontDict", 17 ) == 0 )
          {
            /* if /FDArray was found, then cid->num_dicts is > 0, and */
            /* we can start increasing parser->num_dict               */
            if ( face->cid.num_dicts > 0 )
            {
              parser->num_dict++;

#ifdef FT_DEBUG_LEVEL_TRACE
              FT_TRACE4(( " FontDict %u", parser->num_dict ));
              if ( parser->num_dict > face->cid.num_dicts )
                FT_TRACE4(( " (ignored)" ));
              FT_TRACE4(( "\n" ));
#endif
            }
          }
        }

        cur = parser->root.cursor;
        /* no error can occur in cid_parser_skip_spaces */
        if ( cur >= limit )
          break;

        cid_parser_skip_PS_token( parser );
        if ( parser->root.cursor >= limit || parser->root.error )
          break;

        /* look for immediates */
        if ( *cur == '/' && cur + 2 < limit )
        {
          FT_UInt  len;


          cur++;
          len = (FT_UInt)( parser->root.cursor - cur );

          if ( len > 0 && len < 22 )
          {
            /* now compare the immediate name to the keyword table */
            T1_Field  keyword = (T1_Field)cid_field_records;


            while ( keyword->len )
            {
              FT_Byte*  name = (FT_Byte*)keyword->ident;


              if ( keyword->len == len              &&
                   ft_memcmp( cur, name, len ) == 0 )
              {
                /* we found it - run the parsing callback */
                parser->root.error = cid_load_keyword( face,
                                                       loader,
                                                       keyword );
                if ( parser->root.error )
                  return parser->root.error;
                break;
              }

              keyword++;
            }
          }
        }

        cur = parser->root.cursor;
      }

      if ( !face->cid.num_dicts )
      {
        FT_ERROR(( "cid_parse_dict: No font dictionary found\n" ));
        return FT_THROW( Invalid_File_Format );
      }
    }

    return parser->root.error;
  }


  /* read the subrmap and the subrs of each font dict */
  static FT_Error
  cid_read_subrs( CID_Face  face )
  {
    CID_FaceInfo   cid    = &face->cid;
    FT_Memory      memory = face->root.memory;
    FT_Stream      stream = face->cid_stream;
    FT_Error       error;
    FT_UInt        n;
    CID_Subrs      subr;
    FT_UInt        max_offsets = 0;
    FT_ULong*      offsets = NULL;
    PSAux_Service  psaux = (PSAux_Service)face->psaux;


    if ( FT_NEW_ARRAY( face->subrs, cid->num_dicts ) )
      goto Exit;

    subr = face->subrs;
    for ( n = 0; n < cid->num_dicts; n++, subr++ )
    {
      CID_FaceDict  dict  = cid->font_dicts + n;
      FT_Int        lenIV = dict->private_dict.lenIV;
      FT_UInt       count, num_subrs = dict->num_subrs;
      FT_ULong      data_len;
      FT_Byte*      p;


      if ( !num_subrs )
        continue;

      /* reallocate offsets array if needed */
      if ( num_subrs + 1 > max_offsets )
      {
        FT_UInt  new_max = FT_PAD_CEIL( num_subrs + 1, 4 );


        if ( new_max <= max_offsets )
        {
          error = FT_THROW( Syntax_Error );
          goto Fail;
        }

        if ( FT_QRENEW_ARRAY( offsets, max_offsets, new_max ) )
          goto Fail;

        max_offsets = new_max;
      }

      /* read the subrmap's offsets */
      if ( FT_STREAM_SEEK( cid->data_offset + dict->subrmap_offset ) ||
           FT_FRAME_ENTER( ( num_subrs + 1 ) * dict->sd_bytes )      )
        goto Fail;

      p = (FT_Byte*)stream->cursor;
      for ( count = 0; count <= num_subrs; count++ )
        offsets[count] = cid_get_offset( &p, dict->sd_bytes );

      FT_FRAME_EXIT();

      /* offsets must be ordered */
      for ( count = 1; count <= num_subrs; count++ )
        if ( offsets[count - 1] > offsets[count] )
        {
          FT_ERROR(( "cid_read_subrs: offsets are not ordered\n" ));
          error = FT_THROW( Invalid_File_Format );
          goto Fail;
        }

      if ( offsets[num_subrs] > stream->size - cid->data_offset )
      {
        FT_ERROR(( "cid_read_subrs: too large `subrs' offsets\n" ));
        error = FT_THROW( Invalid_File_Format );
        goto Fail;
      }

      /* now, compute the size of subrs charstrings, */
      /* allocate, and read them                     */
      data_len = offsets[num_subrs] - offsets[0];

      if ( FT_QNEW_ARRAY( subr->code, num_subrs + 1 ) ||
           FT_QALLOC( subr->code[0], data_len )       )
        goto Fail;

      if ( FT_STREAM_SEEK( cid->data_offset + offsets[0] ) ||
           FT_STREAM_READ( subr->code[0], data_len )       )
        goto Fail;

      /* set up pointers */
      for ( count = 1; count <= num_subrs; count++ )
      {
        FT_ULong  len;


        len               = offsets[count] - offsets[count - 1];
        subr->code[count] = subr->code[count - 1] + len;
      }

      /* decrypt subroutines, but only if lenIV >= 0 */
      if ( lenIV >= 0 )
      {
        for ( count = 0; count < num_subrs; count++ )
        {
          FT_ULong  len;


          len = offsets[count + 1] - offsets[count];
          psaux->t1_decrypt( subr->code[count], len, 4330 );
        }
      }

      subr->num_subrs = (FT_Int)num_subrs;
    }

  Exit:
    FT_FREE( offsets );
    return error;

  Fail:
    if ( face->subrs )
    {
      for ( n = 0; n < cid->num_dicts; n++ )
      {
        if ( face->subrs[n].code )
          FT_FREE( face->subrs[n].code[0] );

        FT_FREE( face->subrs[n].code );
      }
      FT_FREE( face->subrs );
    }
    goto Exit;
  }


  static void
  cid_init_loader( CID_Loader*  loader,
                   CID_Face     face )
  {
    FT_UNUSED( face );

    FT_ZERO( loader );
  }


  static  void
  cid_done_loader( CID_Loader*  loader )
  {
    CID_Parser*  parser = &loader->parser;


    /* finalize parser */
    cid_parser_done( parser );
  }


  static FT_Error
  cid_hex_to_binary( FT_Byte*   data,
                     FT_ULong   data_len,
                     FT_ULong   offset,
                     CID_Face   face,
                     FT_ULong*  data_written )
  {
    FT_Stream  stream = face->root.stream;
    FT_Error   error;

    FT_Byte    buffer[256];
    FT_Byte   *p, *plimit;
    FT_Byte   *d = data, *dlimit;
    FT_Byte    val;

    FT_Bool    upper_nibble, done;


    if ( FT_STREAM_SEEK( offset ) )
      goto Exit;

    dlimit = d + data_len;
    p      = buffer;
    plimit = p;

    upper_nibble = 1;
    done         = 0;

    while ( d < dlimit )
    {
      if ( p >= plimit )
      {
        FT_ULong  oldpos = FT_STREAM_POS();
        FT_ULong  size   = stream->size - oldpos;


        if ( size == 0 )
        {
          error = FT_THROW( Syntax_Error );
          goto Exit;
        }

        if ( FT_STREAM_READ( buffer, 256 > size ? size : 256 ) )
          goto Exit;
        p      = buffer;
        plimit = p + FT_STREAM_POS() - oldpos;
      }

      if ( ft_isdigit( *p ) )
        val = (FT_Byte)( *p - '0' );
      else if ( *p >= 'a' && *p <= 'f' )
        val = (FT_Byte)( *p - 'a' + 10 );
      else if ( *p >= 'A' && *p <= 'F' )
        val = (FT_Byte)( *p - 'A' + 10 );
      else if ( *p == ' '  ||
                *p == '\t' ||
                *p == '\r' ||
                *p == '\n' ||
                *p == '\f' ||
                *p == '\0' )
      {
        p++;
        continue;
      }
      else if ( *p == '>' )
      {
        val  = 0;
        done = 1;
      }
      else
      {
        error = FT_THROW( Syntax_Error );
        goto Exit;
      }

      if ( upper_nibble )
        *d = (FT_Byte)( val << 4 );
      else
      {
        *d = (FT_Byte)( *d + val );
        d++;
      }

      upper_nibble = (FT_Byte)( 1 - upper_nibble );

      if ( done )
        break;

      p++;
    }

    error = FT_Err_Ok;

  Exit:
    *data_written = (FT_ULong)( d - data );
    return error;
  }


  FT_LOCAL_DEF( FT_Error )
  cid_face_open( CID_Face  face,
                 FT_Int    face_index )
  {
    CID_Loader   loader;
    CID_Parser*  parser;
    FT_Memory    memory = face->root.memory;
    FT_Error     error;
    FT_UInt      n;

    CID_FaceInfo  cid = &face->cid;

    FT_ULong  binary_length;


    cid_init_loader( &loader, face );

    parser = &loader.parser;
    error = cid_parser_new( parser, face->root.stream, face->root.memory,
                            (PSAux_Service)face->psaux );
    if ( error )
      goto Exit;

    error = cid_parse_dict( face, &loader,
                            parser->postscript,
                            parser->postscript_len );
    if ( error )
      goto Exit;

    if ( face_index < 0 )
      goto Exit;

    if ( FT_NEW( face->cid_stream ) )
      goto Exit;

    if ( parser->binary_length )
    {
      if ( parser->binary_length >
             face->root.stream->size - parser->data_offset )
      {
        FT_TRACE0(( "cid_face_open: adjusting length of binary data\n" ));
        FT_TRACE0(( "               (from %lu to %lu bytes)\n",
                    parser->binary_length,
                    face->root.stream->size - parser->data_offset ));
        parser->binary_length = face->root.stream->size -
                                parser->data_offset;
      }

      /* we must convert the data section from hexadecimal to binary */
      if ( FT_QALLOC( face->binary_data, parser->binary_length )   ||
           FT_SET_ERROR( cid_hex_to_binary( face->binary_data,
                                            parser->binary_length,
                                            parser->data_offset,
                                            face,
                                            &binary_length ) )     )
        goto Exit;

      FT_Stream_OpenMemory( face->cid_stream,
                            face->binary_data, binary_length );
      cid->data_offset = 0;
    }
    else
    {
      *face->cid_stream = *face->root.stream;
      cid->data_offset  = loader.parser.data_offset;
    }

    /* sanity tests */

    if ( cid->gd_bytes == 0 )
    {
      FT_ERROR(( "cid_face_open:"
                 " Invalid `GDBytes' value\n" ));
      error = FT_THROW( Invalid_File_Format );
      goto Exit;
    }

    /* allow at most 32bit offsets */
    if ( cid->fd_bytes > 4 || cid->gd_bytes > 4 )
    {
      FT_ERROR(( "cid_face_open:"
                 " Values of `FDBytes' or `GDBytes' larger than 4\n" ));
      FT_ERROR(( "               "
                 " are not supported\n" ));
      error = FT_THROW( Invalid_File_Format );
      goto Exit;
    }

    binary_length = face->cid_stream->size - cid->data_offset;

    if ( cid->cidmap_offset > binary_length )
    {
      FT_ERROR(( "cid_face_open: Invalid `CIDMapOffset' value\n" ));
      error = FT_THROW( Invalid_File_Format );
      goto Exit;
    }

    /* the initial pre-check prevents the multiplication overflow */
    if ( cid->cid_count > FT_ULONG_MAX / 8                    ||
         cid->cid_count * ( cid->fd_bytes + cid->gd_bytes ) >
           binary_length - cid->cidmap_offset                 )
    {
      FT_ERROR(( "cid_face_open: Invalid `CIDCount' value\n" ));
      error = FT_THROW( Invalid_File_Format );
      goto Exit;
    }


    for ( n = 0; n < cid->num_dicts; n++ )
    {
      CID_FaceDict  dict = cid->font_dicts + n;


      /* the upper limits are ad-hoc values */
      if ( dict->private_dict.blue_shift > 1000 ||
           dict->private_dict.blue_shift < 0    )
      {
        FT_TRACE2(( "cid_face_open:"
                    " setting unlikely BlueShift value %d to default (7)\n",
                    dict->private_dict.blue_shift ));
        dict->private_dict.blue_shift = 7;
      }

      if ( dict->private_dict.blue_fuzz > 1000 ||
           dict->private_dict.blue_fuzz < 0    )
      {
        FT_TRACE2(( "cid_face_open:"
                    " setting unlikely BlueFuzz value %d to default (1)\n",
                    dict->private_dict.blue_fuzz ));
        dict->private_dict.blue_fuzz = 1;
      }

      if ( dict->num_subrs && dict->sd_bytes == 0 )
      {
        FT_ERROR(( "cid_face_open: Invalid `SDBytes' value\n" ));
        error = FT_THROW( Invalid_File_Format );
        goto Exit;
      }

      if ( dict->sd_bytes > 4 )
      {
        FT_ERROR(( "cid_face_open:"
                   " Values of `SDBytes' larger than 4"
                   " are not supported\n" ));
        error = FT_THROW( Invalid_File_Format );
        goto Exit;
      }

      if ( dict->subrmap_offset > binary_length )
      {
        FT_ERROR(( "cid_face_open: Invalid `SubrMapOffset' value\n" ));
        error = FT_THROW( Invalid_File_Format );
        goto Exit;
      }

      /* the initial pre-check prevents the multiplication overflow */
      if ( dict->num_subrs > FT_UINT_MAX / 4      ||
           dict->num_subrs * dict->sd_bytes >
             binary_length - dict->subrmap_offset )
      {
        FT_ERROR(( "cid_face_open: Invalid `SubrCount' value\n" ));
        error = FT_THROW( Invalid_File_Format );
        goto Exit;
      }
    }

    /* we can now safely proceed */
    error = cid_read_subrs( face );

  Exit:
    cid_done_loader( &loader );
    return error;
  }


/* END */
