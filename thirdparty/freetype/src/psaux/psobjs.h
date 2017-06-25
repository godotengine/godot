/***************************************************************************/
/*                                                                         */
/*  psobjs.h                                                               */
/*                                                                         */
/*    Auxiliary functions for PostScript fonts (specification).            */
/*                                                                         */
/*  Copyright 1996-2017 by                                                 */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


#ifndef PSOBJS_H_
#define PSOBJS_H_


#include <ft2build.h>
#include FT_INTERNAL_POSTSCRIPT_AUX_H


FT_BEGIN_HEADER


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                             T1_TABLE                          *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/


  FT_CALLBACK_TABLE
  const PS_Table_FuncsRec    ps_table_funcs;

  FT_CALLBACK_TABLE
  const PS_Parser_FuncsRec   ps_parser_funcs;

  FT_CALLBACK_TABLE
  const T1_Builder_FuncsRec  t1_builder_funcs;


  FT_LOCAL( FT_Error )
  ps_table_new( PS_Table   table,
                FT_Int     count,
                FT_Memory  memory );

  FT_LOCAL( FT_Error )
  ps_table_add( PS_Table  table,
                FT_Int    idx,
                void*     object,
                FT_UInt   length );

  FT_LOCAL( void )
  ps_table_done( PS_Table  table );


  FT_LOCAL( void )
  ps_table_release( PS_Table  table );


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                            T1 PARSER                          *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/


  FT_LOCAL( void )
  ps_parser_skip_spaces( PS_Parser  parser );

  FT_LOCAL( void )
  ps_parser_skip_PS_token( PS_Parser  parser );

  FT_LOCAL( void )
  ps_parser_to_token( PS_Parser  parser,
                      T1_Token   token );

  FT_LOCAL( void )
  ps_parser_to_token_array( PS_Parser  parser,
                            T1_Token   tokens,
                            FT_UInt    max_tokens,
                            FT_Int*    pnum_tokens );

  FT_LOCAL( FT_Error )
  ps_parser_load_field( PS_Parser       parser,
                        const T1_Field  field,
                        void**          objects,
                        FT_UInt         max_objects,
                        FT_ULong*       pflags );

  FT_LOCAL( FT_Error )
  ps_parser_load_field_table( PS_Parser       parser,
                              const T1_Field  field,
                              void**          objects,
                              FT_UInt         max_objects,
                              FT_ULong*       pflags );

  FT_LOCAL( FT_Long )
  ps_parser_to_int( PS_Parser  parser );


  FT_LOCAL( FT_Error )
  ps_parser_to_bytes( PS_Parser  parser,
                      FT_Byte*   bytes,
                      FT_Offset  max_bytes,
                      FT_ULong*  pnum_bytes,
                      FT_Bool    delimiters );


  FT_LOCAL( FT_Fixed )
  ps_parser_to_fixed( PS_Parser  parser,
                      FT_Int     power_ten );


  FT_LOCAL( FT_Int )
  ps_parser_to_coord_array( PS_Parser  parser,
                            FT_Int     max_coords,
                            FT_Short*  coords );

  FT_LOCAL( FT_Int )
  ps_parser_to_fixed_array( PS_Parser  parser,
                            FT_Int     max_values,
                            FT_Fixed*  values,
                            FT_Int     power_ten );


  FT_LOCAL( void )
  ps_parser_init( PS_Parser  parser,
                  FT_Byte*   base,
                  FT_Byte*   limit,
                  FT_Memory  memory );

  FT_LOCAL( void )
  ps_parser_done( PS_Parser  parser );


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                            T1 BUILDER                         *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  FT_LOCAL( void )
  t1_builder_init( T1_Builder    builder,
                   FT_Face       face,
                   FT_Size       size,
                   FT_GlyphSlot  glyph,
                   FT_Bool       hinting );

  FT_LOCAL( void )
  t1_builder_done( T1_Builder  builder );

  FT_LOCAL( FT_Error )
  t1_builder_check_points( T1_Builder  builder,
                           FT_Int      count );

  FT_LOCAL( void )
  t1_builder_add_point( T1_Builder  builder,
                        FT_Pos      x,
                        FT_Pos      y,
                        FT_Byte     flag );

  FT_LOCAL( FT_Error )
  t1_builder_add_point1( T1_Builder  builder,
                         FT_Pos      x,
                         FT_Pos      y );

  FT_LOCAL( FT_Error )
  t1_builder_add_contour( T1_Builder  builder );


  FT_LOCAL( FT_Error )
  t1_builder_start_point( T1_Builder  builder,
                          FT_Pos      x,
                          FT_Pos      y );


  FT_LOCAL( void )
  t1_builder_close_contour( T1_Builder  builder );


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                            OTHER                              *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  FT_LOCAL( void )
  t1_decrypt( FT_Byte*   buffer,
              FT_Offset  length,
              FT_UShort  seed );


FT_END_HEADER

#endif /* PSOBJS_H_ */


/* END */
