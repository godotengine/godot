/****************************************************************************
 *
 * cidparse.h
 *
 *   CID-keyed Type1 parser (specification).
 *
 * Copyright (C) 1996-2021 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#ifndef CIDPARSE_H_
#define CIDPARSE_H_


#include <freetype/internal/t1types.h>
#include <freetype/internal/ftstream.h>
#include <freetype/internal/psaux.h>


FT_BEGIN_HEADER


  /**************************************************************************
   *
   * @Struct:
   *   CID_Parser
   *
   * @Description:
   *   A CID_Parser is an object used to parse a Type 1 fonts very
   *   quickly.
   *
   * @Fields:
   *   root ::
   *     The root PS_ParserRec fields.
   *
   *   stream ::
   *     The current input stream.
   *
   *   postscript ::
   *     A pointer to the data to be parsed.
   *
   *   postscript_len ::
   *     The length of the data to be parsed.
   *
   *   data_offset ::
   *     The start position of the binary data (i.e., the
   *     end of the data to be parsed.
   *
   *   binary_length ::
   *     The length of the data after the `StartData'
   *     command if the data format is hexadecimal.
   *
   *   cid ::
   *     A structure which holds the information about
   *     the current font.
   *
   *   num_dict ::
   *     The number of font dictionaries.
   */
  typedef struct  CID_Parser_
  {
    PS_ParserRec  root;
    FT_Stream     stream;

    FT_Byte*      postscript;
    FT_ULong      postscript_len;

    FT_ULong      data_offset;

    FT_ULong      binary_length;

    CID_FaceInfo  cid;
    FT_UInt       num_dict;

  } CID_Parser;


  FT_LOCAL( FT_Error )
  cid_parser_new( CID_Parser*    parser,
                  FT_Stream      stream,
                  FT_Memory      memory,
                  PSAux_Service  psaux );

  FT_LOCAL( void )
  cid_parser_done( CID_Parser*  parser );


  /**************************************************************************
   *
   *                           PARSING ROUTINES
   *
   */

#define cid_parser_skip_spaces( p )                 \
          (p)->root.funcs.skip_spaces( &(p)->root )
#define cid_parser_skip_PS_token( p )                 \
          (p)->root.funcs.skip_PS_token( &(p)->root )

#define cid_parser_to_int( p )       (p)->root.funcs.to_int( &(p)->root )
#define cid_parser_to_fixed( p, t )  (p)->root.funcs.to_fixed( &(p)->root, t )

#define cid_parser_to_coord_array( p, m, c )                 \
          (p)->root.funcs.to_coord_array( &(p)->root, m, c )
#define cid_parser_to_fixed_array( p, m, f, t )                 \
          (p)->root.funcs.to_fixed_array( &(p)->root, m, f, t )
#define cid_parser_to_token( p, t )                 \
          (p)->root.funcs.to_token( &(p)->root, t )
#define cid_parser_to_token_array( p, t, m, c )                 \
          (p)->root.funcs.to_token_array( &(p)->root, t, m, c )

#define cid_parser_load_field( p, f, o )                       \
          (p)->root.funcs.load_field( &(p)->root, f, o, 0, 0 )
#define cid_parser_load_field_table( p, f, o )                       \
          (p)->root.funcs.load_field_table( &(p)->root, f, o, 0, 0 )


FT_END_HEADER

#endif /* CIDPARSE_H_ */


/* END */
