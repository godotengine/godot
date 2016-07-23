/***************************************************************************/
/*                                                                         */
/*  cidparse.c                                                             */
/*                                                                         */
/*    CID-keyed Type1 parser (body).                                       */
/*                                                                         */
/*  Copyright 1996-2016 by                                                 */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


#include <ft2build.h>
#include FT_INTERNAL_DEBUG_H
#include FT_INTERNAL_OBJECTS_H
#include FT_INTERNAL_STREAM_H

#include "cidparse.h"

#include "ciderrs.h"


  /*************************************************************************/
  /*                                                                       */
  /* The macro FT_COMPONENT is used in trace mode.  It is an implicit      */
  /* parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log  */
  /* messages during execution.                                            */
  /*                                                                       */
#undef  FT_COMPONENT
#define FT_COMPONENT  trace_cidparse


  /*************************************************************************/
  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                    INPUT STREAM PARSER                        *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/
  /*************************************************************************/


#define STARTDATA      "StartData"
#define STARTDATA_LEN  ( sizeof ( STARTDATA ) - 1 )
#define SFNTS          "/sfnts"
#define SFNTS_LEN      ( sizeof ( SFNTS ) - 1 )


  FT_LOCAL_DEF( FT_Error )
  cid_parser_new( CID_Parser*    parser,
                  FT_Stream      stream,
                  FT_Memory      memory,
                  PSAux_Service  psaux )
  {
    FT_Error  error;
    FT_ULong  base_offset, offset, ps_len;
    FT_Byte   *cur, *limit;
    FT_Byte   *arg1, *arg2;


    FT_MEM_ZERO( parser, sizeof ( *parser ) );
    psaux->ps_parser_funcs->init( &parser->root, 0, 0, memory );

    parser->stream = stream;

    base_offset = FT_STREAM_POS();

    /* first of all, check the font format in the header */
    if ( FT_FRAME_ENTER( 31 ) )
      goto Exit;

    if ( ft_strncmp( (char *)stream->cursor,
                     "%!PS-Adobe-3.0 Resource-CIDFont", 31 ) )
    {
      FT_TRACE2(( "  not a CID-keyed font\n" ));
      error = FT_THROW( Unknown_File_Format );
    }

    FT_FRAME_EXIT();
    if ( error )
      goto Exit;

  Again:
    /* now, read the rest of the file until we find */
    /* `StartData' or `/sfnts'                      */
    {
      /*
       * The algorithm is as follows (omitting the case with less than 256
       * bytes to fill for simplicity).
       *
       * 1. Fill the buffer with 256 + STARTDATA_LEN bytes.
       *
       * 2. Search for the STARTDATA and SFNTS strings at positions
       *    buffer[0], buffer[1], ...,
       *    buffer[255 + STARTDATA_LEN - SFNTS_LEN].
       *
       * 3. Move the last STARTDATA_LEN bytes to buffer[0].
       *
       * 4. Fill the buffer with 256 bytes, starting at STARTDATA_LEN.
       *
       * 5. Repeat with step 2.
       *
       */
      FT_Byte  buffer[256 + STARTDATA_LEN + 1];

      /* values for the first loop */
      FT_ULong  read_len    = 256 + STARTDATA_LEN;
      FT_ULong  read_offset = 0;
      FT_Byte*  p           = buffer;


      for ( offset = FT_STREAM_POS(); ; offset += 256 )
      {
        FT_ULong  stream_len;


        stream_len = stream->size - FT_STREAM_POS();

        read_len = FT_MIN( read_len, stream_len );
        if ( FT_STREAM_READ( p, read_len ) )
          goto Exit;

        /* ensure that we do not compare with data beyond the buffer */
        p[read_len] = '\0';

        limit = p + read_len - SFNTS_LEN;

        for ( p = buffer; p < limit; p++ )
        {
          if ( p[0] == 'S'                                           &&
               ft_strncmp( (char*)p, STARTDATA, STARTDATA_LEN ) == 0 )
          {
            /* save offset of binary data after `StartData' */
            offset += (FT_ULong)( p - buffer ) + STARTDATA_LEN;
            goto Found;
          }
          else if ( p[1] == 's'                                   &&
                    ft_strncmp( (char*)p, SFNTS, SFNTS_LEN ) == 0 )
          {
            offset += (FT_ULong)( p - buffer ) + SFNTS_LEN;
            goto Found;
          }
        }

        if ( read_offset + read_len < STARTDATA_LEN )
        {
          FT_TRACE2(( "cid_parser_new: no `StartData' keyword found\n" ));
          error = FT_THROW( Invalid_File_Format );
          goto Exit;
        }

        FT_MEM_MOVE( buffer,
                     buffer + read_offset + read_len - STARTDATA_LEN,
                     STARTDATA_LEN );

        /* values for the next loop */
        read_len    = 256;
        read_offset = STARTDATA_LEN;
        p           = buffer + read_offset;
      }
    }

  Found:
    /* We have found the start of the binary data or the `/sfnts' token. */
    /* Now rewind and extract the frame corresponding to this PostScript */
    /* section.                                                          */

    ps_len = offset - base_offset;
    if ( FT_STREAM_SEEK( base_offset )                  ||
         FT_FRAME_EXTRACT( ps_len, parser->postscript ) )
      goto Exit;

    parser->data_offset    = offset;
    parser->postscript_len = ps_len;
    parser->root.base      = parser->postscript;
    parser->root.cursor    = parser->postscript;
    parser->root.limit     = parser->root.cursor + ps_len;
    parser->num_dict       = -1;

    /* Finally, we check whether `StartData' or `/sfnts' was real --  */
    /* it could be in a comment or string.  We also get the arguments */
    /* of `StartData' to find out whether the data is represented in  */
    /* binary or hex format.                                          */

    arg1 = parser->root.cursor;
    cid_parser_skip_PS_token( parser );
    cid_parser_skip_spaces  ( parser );
    arg2 = parser->root.cursor;
    cid_parser_skip_PS_token( parser );
    cid_parser_skip_spaces  ( parser );

    limit = parser->root.limit;
    cur   = parser->root.cursor;

    while ( cur < limit - SFNTS_LEN )
    {
      if ( parser->root.error )
      {
        error = parser->root.error;
        goto Exit;
      }

      if ( cur[0] == 'S'                                           &&
           cur < limit - STARTDATA_LEN                             &&
           ft_strncmp( (char*)cur, STARTDATA, STARTDATA_LEN ) == 0 )
      {
        if ( ft_strncmp( (char*)arg1, "(Hex)", 5 ) == 0 )
        {
          FT_Long  tmp = ft_atol( (const char *)arg2 );


          if ( tmp < 0 )
          {
            FT_ERROR(( "cid_parser_new: invalid length of hex data\n" ));
            error = FT_THROW( Invalid_File_Format );
          }
          else
            parser->binary_length = (FT_ULong)tmp;
        }

        goto Exit;
      }
      else if ( cur[1] == 's'                                   &&
                ft_strncmp( (char*)cur, SFNTS, SFNTS_LEN ) == 0 )
      {
        FT_TRACE2(( "cid_parser_new: cannot handle Type 11 fonts\n" ));
        error = FT_THROW( Unknown_File_Format );
        goto Exit;
      }

      cid_parser_skip_PS_token( parser );
      cid_parser_skip_spaces  ( parser );
      arg1 = arg2;
      arg2 = cur;
      cur  = parser->root.cursor;
    }

    /* we haven't found the correct `StartData'; go back and continue */
    /* searching                                                      */
    FT_FRAME_RELEASE( parser->postscript );
    if ( !FT_STREAM_SEEK( offset ) )
      goto Again;

  Exit:
    return error;
  }


#undef STARTDATA
#undef STARTDATA_LEN
#undef SFNTS
#undef SFNTS_LEN


  FT_LOCAL_DEF( void )
  cid_parser_done( CID_Parser*  parser )
  {
    /* always free the private dictionary */
    if ( parser->postscript )
    {
      FT_Stream  stream = parser->stream;


      FT_FRAME_RELEASE( parser->postscript );
    }
    parser->root.funcs.done( &parser->root );
  }


/* END */
