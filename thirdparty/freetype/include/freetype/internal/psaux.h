/****************************************************************************
 *
 * psaux.h
 *
 *   Auxiliary functions and data structures related to PostScript fonts
 *   (specification).
 *
 * Copyright (C) 1996-2019 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#ifndef PSAUX_H_
#define PSAUX_H_


#include <ft2build.h>
#include FT_INTERNAL_OBJECTS_H
#include FT_INTERNAL_TYPE1_TYPES_H
#include FT_INTERNAL_HASH_H
#include FT_INTERNAL_TRUETYPE_TYPES_H
#include FT_SERVICE_POSTSCRIPT_CMAPS_H
#include FT_INTERNAL_CFF_TYPES_H
#include FT_INTERNAL_CFF_OBJECTS_TYPES_H



FT_BEGIN_HEADER


  /**************************************************************************
   *
   * PostScript modules driver class.
   */
  typedef struct  PS_DriverRec_
  {
    FT_DriverRec  root;

    FT_UInt   hinting_engine;
    FT_Bool   no_stem_darkening;
    FT_Int    darken_params[8];
    FT_Int32  random_seed;

  } PS_DriverRec, *PS_Driver;


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                             T1_TABLE                          *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/


  typedef struct PS_TableRec_*              PS_Table;
  typedef const struct PS_Table_FuncsRec_*  PS_Table_Funcs;


  /**************************************************************************
   *
   * @struct:
   *   PS_Table_FuncsRec
   *
   * @description:
   *   A set of function pointers to manage PS_Table objects.
   *
   * @fields:
   *   table_init ::
   *     Used to initialize a table.
   *
   *   table_done ::
   *     Finalizes resp. destroy a given table.
   *
   *   table_add ::
   *     Adds a new object to a table.
   *
   *   table_release ::
   *     Releases table data, then finalizes it.
   */
  typedef struct  PS_Table_FuncsRec_
  {
    FT_Error
    (*init)( PS_Table   table,
             FT_Int     count,
             FT_Memory  memory );

    void
    (*done)( PS_Table  table );

    FT_Error
    (*add)( PS_Table  table,
            FT_Int    idx,
            void*     object,
            FT_UInt   length );

    void
    (*release)( PS_Table  table );

  } PS_Table_FuncsRec;


  /**************************************************************************
   *
   * @struct:
   *   PS_TableRec
   *
   * @description:
   *   A PS_Table is a simple object used to store an array of objects in a
   *   single memory block.
   *
   * @fields:
   *   block ::
   *     The address in memory of the growheap's block.  This can change
   *     between two object adds, due to reallocation.
   *
   *   cursor ::
   *     The current top of the grow heap within its block.
   *
   *   capacity ::
   *     The current size of the heap block.  Increments by 1kByte chunks.
   *
   *   init ::
   *     Set to 0xDEADBEEF if 'elements' and 'lengths' have been allocated.
   *
   *   max_elems ::
   *     The maximum number of elements in table.
   *
   *   num_elems ::
   *     The current number of elements in table.
   *
   *   elements ::
   *     A table of element addresses within the block.
   *
   *   lengths ::
   *     A table of element sizes within the block.
   *
   *   memory ::
   *     The object used for memory operations (alloc/realloc).
   *
   *   funcs ::
   *     A table of method pointers for this object.
   */
  typedef struct  PS_TableRec_
  {
    FT_Byte*           block;          /* current memory block           */
    FT_Offset          cursor;         /* current cursor in memory block */
    FT_Offset          capacity;       /* current size of memory block   */
    FT_ULong           init;

    FT_Int             max_elems;
    FT_Int             num_elems;
    FT_Byte**          elements;       /* addresses of table elements */
    FT_UInt*           lengths;        /* lengths of table elements   */

    FT_Memory          memory;
    PS_Table_FuncsRec  funcs;

  } PS_TableRec;


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                       T1 FIELDS & TOKENS                      *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  typedef struct PS_ParserRec_*  PS_Parser;

  typedef struct T1_TokenRec_*   T1_Token;

  typedef struct T1_FieldRec_*   T1_Field;


  /* simple enumeration type used to identify token types */
  typedef enum  T1_TokenType_
  {
    T1_TOKEN_TYPE_NONE = 0,
    T1_TOKEN_TYPE_ANY,
    T1_TOKEN_TYPE_STRING,
    T1_TOKEN_TYPE_ARRAY,
    T1_TOKEN_TYPE_KEY, /* aka `name' */

    /* do not remove */
    T1_TOKEN_TYPE_MAX

  } T1_TokenType;


  /* a simple structure used to identify tokens */
  typedef struct  T1_TokenRec_
  {
    FT_Byte*      start;   /* first character of token in input stream */
    FT_Byte*      limit;   /* first character after the token          */
    T1_TokenType  type;    /* type of token                            */

  } T1_TokenRec;


  /* enumeration type used to identify object fields */
  typedef enum  T1_FieldType_
  {
    T1_FIELD_TYPE_NONE = 0,
    T1_FIELD_TYPE_BOOL,
    T1_FIELD_TYPE_INTEGER,
    T1_FIELD_TYPE_FIXED,
    T1_FIELD_TYPE_FIXED_1000,
    T1_FIELD_TYPE_STRING,
    T1_FIELD_TYPE_KEY,
    T1_FIELD_TYPE_BBOX,
    T1_FIELD_TYPE_MM_BBOX,
    T1_FIELD_TYPE_INTEGER_ARRAY,
    T1_FIELD_TYPE_FIXED_ARRAY,
    T1_FIELD_TYPE_CALLBACK,

    /* do not remove */
    T1_FIELD_TYPE_MAX

  } T1_FieldType;


  typedef enum  T1_FieldLocation_
  {
    T1_FIELD_LOCATION_CID_INFO,
    T1_FIELD_LOCATION_FONT_DICT,
    T1_FIELD_LOCATION_FONT_EXTRA,
    T1_FIELD_LOCATION_FONT_INFO,
    T1_FIELD_LOCATION_PRIVATE,
    T1_FIELD_LOCATION_BBOX,
    T1_FIELD_LOCATION_LOADER,
    T1_FIELD_LOCATION_FACE,
    T1_FIELD_LOCATION_BLEND,

    /* do not remove */
    T1_FIELD_LOCATION_MAX

  } T1_FieldLocation;


  typedef void
  (*T1_Field_ParseFunc)( FT_Face     face,
                         FT_Pointer  parser );


  /* structure type used to model object fields */
  typedef struct  T1_FieldRec_
  {
    const char*         ident;        /* field identifier               */
    T1_FieldLocation    location;
    T1_FieldType        type;         /* type of field                  */
    T1_Field_ParseFunc  reader;
    FT_UInt             offset;       /* offset of field in object      */
    FT_Byte             size;         /* size of field in bytes         */
    FT_UInt             array_max;    /* maximum number of elements for */
                                      /* array                          */
    FT_UInt             count_offset; /* offset of element count for    */
                                      /* arrays; must not be zero if in */
                                      /* use -- in other words, a       */
                                      /* `num_FOO' element must not     */
                                      /* start the used structure if we */
                                      /* parse a `FOO' array            */
    FT_UInt             dict;         /* where we expect it             */
  } T1_FieldRec;

#define T1_FIELD_DICT_FONTDICT ( 1 << 0 ) /* also FontInfo and FDArray */
#define T1_FIELD_DICT_PRIVATE  ( 1 << 1 )



#define T1_NEW_SIMPLE_FIELD( _ident, _type, _fname, _dict ) \
          {                                                 \
            _ident, T1CODE, _type,                          \
            0,                                              \
            FT_FIELD_OFFSET( _fname ),                      \
            FT_FIELD_SIZE( _fname ),                        \
            0, 0,                                           \
            _dict                                           \
          },

#define T1_NEW_CALLBACK_FIELD( _ident, _reader, _dict ) \
          {                                             \
            _ident, T1CODE, T1_FIELD_TYPE_CALLBACK,     \
            (T1_Field_ParseFunc)_reader,                \
            0, 0,                                       \
            0, 0,                                       \
            _dict                                       \
          },

#define T1_NEW_TABLE_FIELD( _ident, _type, _fname, _max, _dict ) \
          {                                                      \
            _ident, T1CODE, _type,                               \
            0,                                                   \
            FT_FIELD_OFFSET( _fname ),                           \
            FT_FIELD_SIZE_DELTA( _fname ),                       \
            _max,                                                \
            FT_FIELD_OFFSET( num_ ## _fname ),                   \
            _dict                                                \
          },

#define T1_NEW_TABLE_FIELD2( _ident, _type, _fname, _max, _dict ) \
          {                                                       \
            _ident, T1CODE, _type,                                \
            0,                                                    \
            FT_FIELD_OFFSET( _fname ),                            \
            FT_FIELD_SIZE_DELTA( _fname ),                        \
            _max, 0,                                              \
            _dict                                                 \
          },


#define T1_FIELD_BOOL( _ident, _fname, _dict )                             \
          T1_NEW_SIMPLE_FIELD( _ident, T1_FIELD_TYPE_BOOL, _fname, _dict )

#define T1_FIELD_NUM( _ident, _fname, _dict )                                 \
          T1_NEW_SIMPLE_FIELD( _ident, T1_FIELD_TYPE_INTEGER, _fname, _dict )

#define T1_FIELD_FIXED( _ident, _fname, _dict )                             \
          T1_NEW_SIMPLE_FIELD( _ident, T1_FIELD_TYPE_FIXED, _fname, _dict )

#define T1_FIELD_FIXED_1000( _ident, _fname, _dict )                     \
          T1_NEW_SIMPLE_FIELD( _ident, T1_FIELD_TYPE_FIXED_1000, _fname, \
                               _dict )

#define T1_FIELD_STRING( _ident, _fname, _dict )                             \
          T1_NEW_SIMPLE_FIELD( _ident, T1_FIELD_TYPE_STRING, _fname, _dict )

#define T1_FIELD_KEY( _ident, _fname, _dict )                             \
          T1_NEW_SIMPLE_FIELD( _ident, T1_FIELD_TYPE_KEY, _fname, _dict )

#define T1_FIELD_BBOX( _ident, _fname, _dict )                             \
          T1_NEW_SIMPLE_FIELD( _ident, T1_FIELD_TYPE_BBOX, _fname, _dict )


#define T1_FIELD_NUM_TABLE( _ident, _fname, _fmax, _dict )         \
          T1_NEW_TABLE_FIELD( _ident, T1_FIELD_TYPE_INTEGER_ARRAY, \
                              _fname, _fmax, _dict )

#define T1_FIELD_FIXED_TABLE( _ident, _fname, _fmax, _dict )     \
          T1_NEW_TABLE_FIELD( _ident, T1_FIELD_TYPE_FIXED_ARRAY, \
                              _fname, _fmax, _dict )

#define T1_FIELD_NUM_TABLE2( _ident, _fname, _fmax, _dict )         \
          T1_NEW_TABLE_FIELD2( _ident, T1_FIELD_TYPE_INTEGER_ARRAY, \
                               _fname, _fmax, _dict )

#define T1_FIELD_FIXED_TABLE2( _ident, _fname, _fmax, _dict )     \
          T1_NEW_TABLE_FIELD2( _ident, T1_FIELD_TYPE_FIXED_ARRAY, \
                               _fname, _fmax, _dict )

#define T1_FIELD_CALLBACK( _ident, _name, _dict )       \
          T1_NEW_CALLBACK_FIELD( _ident, _name, _dict )


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                            T1 PARSER                          *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  typedef const struct PS_Parser_FuncsRec_*  PS_Parser_Funcs;

  typedef struct  PS_Parser_FuncsRec_
  {
    void
    (*init)( PS_Parser  parser,
             FT_Byte*   base,
             FT_Byte*   limit,
             FT_Memory  memory );

    void
    (*done)( PS_Parser  parser );

    void
    (*skip_spaces)( PS_Parser  parser );
    void
    (*skip_PS_token)( PS_Parser  parser );

    FT_Long
    (*to_int)( PS_Parser  parser );
    FT_Fixed
    (*to_fixed)( PS_Parser  parser,
                 FT_Int     power_ten );

    FT_Error
    (*to_bytes)( PS_Parser  parser,
                 FT_Byte*   bytes,
                 FT_Offset  max_bytes,
                 FT_ULong*  pnum_bytes,
                 FT_Bool    delimiters );

    FT_Int
    (*to_coord_array)( PS_Parser  parser,
                       FT_Int     max_coords,
                       FT_Short*  coords );
    FT_Int
    (*to_fixed_array)( PS_Parser  parser,
                       FT_Int     max_values,
                       FT_Fixed*  values,
                       FT_Int     power_ten );

    void
    (*to_token)( PS_Parser  parser,
                 T1_Token   token );
    void
    (*to_token_array)( PS_Parser  parser,
                       T1_Token   tokens,
                       FT_UInt    max_tokens,
                       FT_Int*    pnum_tokens );

    FT_Error
    (*load_field)( PS_Parser       parser,
                   const T1_Field  field,
                   void**          objects,
                   FT_UInt         max_objects,
                   FT_ULong*       pflags );

    FT_Error
    (*load_field_table)( PS_Parser       parser,
                         const T1_Field  field,
                         void**          objects,
                         FT_UInt         max_objects,
                         FT_ULong*       pflags );

  } PS_Parser_FuncsRec;


  /**************************************************************************
   *
   * @struct:
   *   PS_ParserRec
   *
   * @description:
   *   A PS_Parser is an object used to parse a Type 1 font very quickly.
   *
   * @fields:
   *   cursor ::
   *     The current position in the text.
   *
   *   base ::
   *     Start of the processed text.
   *
   *   limit ::
   *     End of the processed text.
   *
   *   error ::
   *     The last error returned.
   *
   *   memory ::
   *     The object used for memory operations (alloc/realloc).
   *
   *   funcs ::
   *     A table of functions for the parser.
   */
  typedef struct  PS_ParserRec_
  {
    FT_Byte*   cursor;
    FT_Byte*   base;
    FT_Byte*   limit;
    FT_Error   error;
    FT_Memory  memory;

    PS_Parser_FuncsRec  funcs;

  } PS_ParserRec;


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                         PS BUILDER                            *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/


  typedef struct PS_Builder_  PS_Builder;
  typedef const struct PS_Builder_FuncsRec_*  PS_Builder_Funcs;

  typedef struct  PS_Builder_FuncsRec_
  {
    void
    (*init)( PS_Builder*  ps_builder,
             void*        builder,
             FT_Bool      is_t1 );

    void
    (*done)( PS_Builder*  builder );

  } PS_Builder_FuncsRec;


  /**************************************************************************
   *
   * @struct:
   *   PS_Builder
   *
   * @description:
   *    A structure used during glyph loading to store its outline.
   *
   * @fields:
   *   memory ::
   *     The current memory object.
   *
   *   face ::
   *     The current face object.
   *
   *   glyph ::
   *     The current glyph slot.
   *
   *   loader ::
   *     XXX
   *
   *   base ::
   *     The base glyph outline.
   *
   *   current ::
   *     The current glyph outline.
   *
   *   pos_x ::
   *     The horizontal translation (if composite glyph).
   *
   *   pos_y ::
   *     The vertical translation (if composite glyph).
   *
   *   left_bearing ::
   *     The left side bearing point.
   *
   *   advance ::
   *     The horizontal advance vector.
   *
   *   bbox ::
   *     Unused.
   *
   *   path_begun ::
   *     A flag which indicates that a new path has begun.
   *
   *   load_points ::
   *     If this flag is not set, no points are loaded.
   *
   *   no_recurse ::
   *     Set but not used.
   *
   *   metrics_only ::
   *     A boolean indicating that we only want to compute the metrics of a
   *     given glyph, not load all of its points.
   *
   *   is_t1 ::
   *     Set if current font type is Type 1.
   *
   *   funcs ::
   *     An array of function pointers for the builder.
   */
  struct  PS_Builder_
  {
    FT_Memory       memory;
    FT_Face         face;
    CFF_GlyphSlot   glyph;
    FT_GlyphLoader  loader;
    FT_Outline*     base;
    FT_Outline*     current;

    FT_Pos*  pos_x;
    FT_Pos*  pos_y;

    FT_Vector*  left_bearing;
    FT_Vector*  advance;

    FT_BBox*  bbox;          /* bounding box */
    FT_Bool   path_begun;
    FT_Bool   load_points;
    FT_Bool   no_recurse;

    FT_Bool  metrics_only;
    FT_Bool  is_t1;

    PS_Builder_FuncsRec  funcs;

  };


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                            PS DECODER                         *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

#define PS_MAX_OPERANDS        48
#define PS_MAX_SUBRS_CALLS     16   /* maximum subroutine nesting;         */
                                    /* only 10 are allowed but there exist */
                                    /* fonts like `HiraKakuProN-W3.ttf'    */
                                    /* (Hiragino Kaku Gothic ProN W3;      */
                                    /* 8.2d6e1; 2014-12-19) that exceed    */
                                    /* this limit                          */

  /* execution context charstring zone */

  typedef struct  PS_Decoder_Zone_
  {
    FT_Byte*  base;
    FT_Byte*  limit;
    FT_Byte*  cursor;

  } PS_Decoder_Zone;


  typedef FT_Error
  (*CFF_Decoder_Get_Glyph_Callback)( TT_Face    face,
                                     FT_UInt    glyph_index,
                                     FT_Byte**  pointer,
                                     FT_ULong*  length );

  typedef void
  (*CFF_Decoder_Free_Glyph_Callback)( TT_Face    face,
                                      FT_Byte**  pointer,
                                      FT_ULong   length );


  typedef struct  PS_Decoder_
  {
    PS_Builder  builder;

    FT_Fixed   stack[PS_MAX_OPERANDS + 1];
    FT_Fixed*  top;

    PS_Decoder_Zone   zones[PS_MAX_SUBRS_CALLS + 1];
    PS_Decoder_Zone*  zone;

    FT_Int     flex_state;
    FT_Int     num_flex_vectors;
    FT_Vector  flex_vectors[7];

    CFF_Font     cff;
    CFF_SubFont  current_subfont; /* for current glyph_index */
    FT_Generic*  cf2_instance;

    FT_Pos*  glyph_width;
    FT_Bool  width_only;
    FT_Int   num_hints;

    FT_UInt  num_locals;
    FT_UInt  num_globals;

    FT_Int  locals_bias;
    FT_Int  globals_bias;

    FT_Byte**  locals;
    FT_Byte**  globals;

    FT_Byte**  glyph_names;   /* for pure CFF fonts only  */
    FT_UInt    num_glyphs;    /* number of glyphs in font */

    FT_Render_Mode  hint_mode;

    FT_Bool  seac;

    CFF_Decoder_Get_Glyph_Callback   get_glyph_callback;
    CFF_Decoder_Free_Glyph_Callback  free_glyph_callback;

    /* Type 1 stuff */
    FT_Service_PsCMaps  psnames;      /* for seac */

    FT_Int    lenIV;         /* internal for sub routine calls   */
    FT_UInt*  locals_len;    /* array of subrs length (optional) */
    FT_Hash   locals_hash;   /* used if `num_subrs' was massaged */

    FT_Matrix  font_matrix;
    FT_Vector  font_offset;

    PS_Blend  blend;         /* for multiple master support */

    FT_Long*  buildchar;
    FT_UInt   len_buildchar;

  } PS_Decoder;


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                         T1 BUILDER                            *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/


  typedef struct T1_BuilderRec_*  T1_Builder;


  typedef FT_Error
  (*T1_Builder_Check_Points_Func)( T1_Builder  builder,
                                   FT_Int      count );

  typedef void
  (*T1_Builder_Add_Point_Func)( T1_Builder  builder,
                                FT_Pos      x,
                                FT_Pos      y,
                                FT_Byte     flag );

  typedef FT_Error
  (*T1_Builder_Add_Point1_Func)( T1_Builder  builder,
                                 FT_Pos      x,
                                 FT_Pos      y );

  typedef FT_Error
  (*T1_Builder_Add_Contour_Func)( T1_Builder  builder );

  typedef FT_Error
  (*T1_Builder_Start_Point_Func)( T1_Builder  builder,
                                  FT_Pos      x,
                                  FT_Pos      y );

  typedef void
  (*T1_Builder_Close_Contour_Func)( T1_Builder  builder );


  typedef const struct T1_Builder_FuncsRec_*  T1_Builder_Funcs;

  typedef struct  T1_Builder_FuncsRec_
  {
    void
    (*init)( T1_Builder    builder,
             FT_Face       face,
             FT_Size       size,
             FT_GlyphSlot  slot,
             FT_Bool       hinting );

    void
    (*done)( T1_Builder   builder );

    T1_Builder_Check_Points_Func   check_points;
    T1_Builder_Add_Point_Func      add_point;
    T1_Builder_Add_Point1_Func     add_point1;
    T1_Builder_Add_Contour_Func    add_contour;
    T1_Builder_Start_Point_Func    start_point;
    T1_Builder_Close_Contour_Func  close_contour;

  } T1_Builder_FuncsRec;


  /* an enumeration type to handle charstring parsing states */
  typedef enum  T1_ParseState_
  {
    T1_Parse_Start,
    T1_Parse_Have_Width,
    T1_Parse_Have_Moveto,
    T1_Parse_Have_Path

  } T1_ParseState;


  /**************************************************************************
   *
   * @struct:
   *   T1_BuilderRec
   *
   * @description:
   *    A structure used during glyph loading to store its outline.
   *
   * @fields:
   *   memory ::
   *     The current memory object.
   *
   *   face ::
   *     The current face object.
   *
   *   glyph ::
   *     The current glyph slot.
   *
   *   loader ::
   *     XXX
   *
   *   base ::
   *     The base glyph outline.
   *
   *   current ::
   *     The current glyph outline.
   *
   *   max_points ::
   *     maximum points in builder outline
   *
   *   max_contours ::
   *     Maximum number of contours in builder outline.
   *
   *   pos_x ::
   *     The horizontal translation (if composite glyph).
   *
   *   pos_y ::
   *     The vertical translation (if composite glyph).
   *
   *   left_bearing ::
   *     The left side bearing point.
   *
   *   advance ::
   *     The horizontal advance vector.
   *
   *   bbox ::
   *     Unused.
   *
   *   parse_state ::
   *     An enumeration which controls the charstring parsing state.
   *
   *   load_points ::
   *     If this flag is not set, no points are loaded.
   *
   *   no_recurse ::
   *     Set but not used.
   *
   *   metrics_only ::
   *     A boolean indicating that we only want to compute the metrics of a
   *     given glyph, not load all of its points.
   *
   *   funcs ::
   *     An array of function pointers for the builder.
   */
  typedef struct  T1_BuilderRec_
  {
    FT_Memory       memory;
    FT_Face         face;
    FT_GlyphSlot    glyph;
    FT_GlyphLoader  loader;
    FT_Outline*     base;
    FT_Outline*     current;

    FT_Pos          pos_x;
    FT_Pos          pos_y;

    FT_Vector       left_bearing;
    FT_Vector       advance;

    FT_BBox         bbox;          /* bounding box */
    T1_ParseState   parse_state;
    FT_Bool         load_points;
    FT_Bool         no_recurse;

    FT_Bool         metrics_only;

    void*           hints_funcs;    /* hinter-specific */
    void*           hints_globals;  /* hinter-specific */

    T1_Builder_FuncsRec  funcs;

  } T1_BuilderRec;


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                         T1 DECODER                            *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

#if 0

  /**************************************************************************
   *
   * T1_MAX_SUBRS_CALLS details the maximum number of nested sub-routine
   * calls during glyph loading.
   */
#define T1_MAX_SUBRS_CALLS  8


  /**************************************************************************
   *
   * T1_MAX_CHARSTRING_OPERANDS is the charstring stack's capacity.  A
   * minimum of 16 is required.
   */
#define T1_MAX_CHARSTRINGS_OPERANDS  32

#endif /* 0 */


  typedef struct  T1_Decoder_ZoneRec_
  {
    FT_Byte*  cursor;
    FT_Byte*  base;
    FT_Byte*  limit;

  } T1_Decoder_ZoneRec, *T1_Decoder_Zone;


  typedef struct T1_DecoderRec_*              T1_Decoder;
  typedef const struct T1_Decoder_FuncsRec_*  T1_Decoder_Funcs;


  typedef FT_Error
  (*T1_Decoder_Callback)( T1_Decoder  decoder,
                          FT_UInt     glyph_index );


  typedef struct  T1_Decoder_FuncsRec_
  {
    FT_Error
    (*init)( T1_Decoder           decoder,
             FT_Face              face,
             FT_Size              size,
             FT_GlyphSlot         slot,
             FT_Byte**            glyph_names,
             PS_Blend             blend,
             FT_Bool              hinting,
             FT_Render_Mode       hint_mode,
             T1_Decoder_Callback  callback );

    void
    (*done)( T1_Decoder  decoder );

#ifdef T1_CONFIG_OPTION_OLD_ENGINE
    FT_Error
    (*parse_charstrings_old)( T1_Decoder  decoder,
                              FT_Byte*    base,
                              FT_UInt     len );
#else
    FT_Error
    (*parse_metrics)( T1_Decoder  decoder,
                      FT_Byte*    base,
                      FT_UInt     len );
#endif

    FT_Error
    (*parse_charstrings)( PS_Decoder*  decoder,
                          FT_Byte*     charstring_base,
                          FT_ULong     charstring_len );


  } T1_Decoder_FuncsRec;


  typedef struct  T1_DecoderRec_
  {
    T1_BuilderRec        builder;

    FT_Long              stack[T1_MAX_CHARSTRINGS_OPERANDS];
    FT_Long*             top;

    T1_Decoder_ZoneRec   zones[T1_MAX_SUBRS_CALLS + 1];
    T1_Decoder_Zone      zone;

    FT_Service_PsCMaps   psnames;      /* for seac */
    FT_UInt              num_glyphs;
    FT_Byte**            glyph_names;

    FT_Int               lenIV;        /* internal for sub routine calls */
    FT_Int               num_subrs;
    FT_Byte**            subrs;
    FT_UInt*             subrs_len;    /* array of subrs length (optional) */
    FT_Hash              subrs_hash;   /* used if `num_subrs' was massaged */

    FT_Matrix            font_matrix;
    FT_Vector            font_offset;

    FT_Int               flex_state;
    FT_Int               num_flex_vectors;
    FT_Vector            flex_vectors[7];

    PS_Blend             blend;       /* for multiple master support */

    FT_Render_Mode       hint_mode;

    T1_Decoder_Callback  parse_callback;
    T1_Decoder_FuncsRec  funcs;

    FT_Long*             buildchar;
    FT_UInt              len_buildchar;

    FT_Bool              seac;

    FT_Generic           cf2_instance;

  } T1_DecoderRec;


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                        CFF BUILDER                            *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/


  typedef struct CFF_Builder_  CFF_Builder;


  typedef FT_Error
  (*CFF_Builder_Check_Points_Func)( CFF_Builder*  builder,
                                    FT_Int        count );

  typedef void
  (*CFF_Builder_Add_Point_Func)( CFF_Builder*  builder,
                                 FT_Pos        x,
                                 FT_Pos        y,
                                 FT_Byte       flag );
  typedef FT_Error
  (*CFF_Builder_Add_Point1_Func)( CFF_Builder*  builder,
                                  FT_Pos        x,
                                  FT_Pos        y );
  typedef FT_Error
  (*CFF_Builder_Start_Point_Func)( CFF_Builder*  builder,
                                   FT_Pos        x,
                                   FT_Pos        y );
  typedef void
  (*CFF_Builder_Close_Contour_Func)( CFF_Builder*  builder );

  typedef FT_Error
  (*CFF_Builder_Add_Contour_Func)( CFF_Builder*  builder );

  typedef const struct CFF_Builder_FuncsRec_*  CFF_Builder_Funcs;

  typedef struct  CFF_Builder_FuncsRec_
  {
    void
    (*init)( CFF_Builder*   builder,
             TT_Face        face,
             CFF_Size       size,
             CFF_GlyphSlot  glyph,
             FT_Bool        hinting );

    void
    (*done)( CFF_Builder*  builder );

    CFF_Builder_Check_Points_Func   check_points;
    CFF_Builder_Add_Point_Func      add_point;
    CFF_Builder_Add_Point1_Func     add_point1;
    CFF_Builder_Add_Contour_Func    add_contour;
    CFF_Builder_Start_Point_Func    start_point;
    CFF_Builder_Close_Contour_Func  close_contour;

  } CFF_Builder_FuncsRec;


  /**************************************************************************
   *
   * @struct:
   *   CFF_Builder
   *
   * @description:
   *    A structure used during glyph loading to store its outline.
   *
   * @fields:
   *   memory ::
   *     The current memory object.
   *
   *   face ::
   *     The current face object.
   *
   *   glyph ::
   *     The current glyph slot.
   *
   *   loader ::
   *     The current glyph loader.
   *
   *   base ::
   *     The base glyph outline.
   *
   *   current ::
   *     The current glyph outline.
   *
   *   pos_x ::
   *     The horizontal translation (if composite glyph).
   *
   *   pos_y ::
   *     The vertical translation (if composite glyph).
   *
   *   left_bearing ::
   *     The left side bearing point.
   *
   *   advance ::
   *     The horizontal advance vector.
   *
   *   bbox ::
   *     Unused.
   *
   *   path_begun ::
   *     A flag which indicates that a new path has begun.
   *
   *   load_points ::
   *     If this flag is not set, no points are loaded.
   *
   *   no_recurse ::
   *     Set but not used.
   *
   *   metrics_only ::
   *     A boolean indicating that we only want to compute the metrics of a
   *     given glyph, not load all of its points.
   *
   *   hints_funcs ::
   *     Auxiliary pointer for hinting.
   *
   *   hints_globals ::
   *     Auxiliary pointer for hinting.
   *
   *   funcs ::
   *     A table of method pointers for this object.
   */
  struct  CFF_Builder_
  {
    FT_Memory       memory;
    TT_Face         face;
    CFF_GlyphSlot   glyph;
    FT_GlyphLoader  loader;
    FT_Outline*     base;
    FT_Outline*     current;

    FT_Pos  pos_x;
    FT_Pos  pos_y;

    FT_Vector  left_bearing;
    FT_Vector  advance;

    FT_BBox  bbox;          /* bounding box */

    FT_Bool  path_begun;
    FT_Bool  load_points;
    FT_Bool  no_recurse;

    FT_Bool  metrics_only;

    void*  hints_funcs;     /* hinter-specific */
    void*  hints_globals;   /* hinter-specific */

    CFF_Builder_FuncsRec  funcs;
  };


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                        CFF DECODER                            *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/


#define CFF_MAX_OPERANDS        48
#define CFF_MAX_SUBRS_CALLS     16  /* maximum subroutine nesting;         */
                                    /* only 10 are allowed but there exist */
                                    /* fonts like `HiraKakuProN-W3.ttf'    */
                                    /* (Hiragino Kaku Gothic ProN W3;      */
                                    /* 8.2d6e1; 2014-12-19) that exceed    */
                                    /* this limit                          */
#define CFF_MAX_TRANS_ELEMENTS  32

  /* execution context charstring zone */

  typedef struct  CFF_Decoder_Zone_
  {
    FT_Byte*  base;
    FT_Byte*  limit;
    FT_Byte*  cursor;

  } CFF_Decoder_Zone;


  typedef struct  CFF_Decoder_
  {
    CFF_Builder  builder;
    CFF_Font     cff;

    FT_Fixed   stack[CFF_MAX_OPERANDS + 1];
    FT_Fixed*  top;

    CFF_Decoder_Zone   zones[CFF_MAX_SUBRS_CALLS + 1];
    CFF_Decoder_Zone*  zone;

    FT_Int     flex_state;
    FT_Int     num_flex_vectors;
    FT_Vector  flex_vectors[7];

    FT_Pos  glyph_width;
    FT_Pos  nominal_width;

    FT_Bool   read_width;
    FT_Bool   width_only;
    FT_Int    num_hints;
    FT_Fixed  buildchar[CFF_MAX_TRANS_ELEMENTS];

    FT_UInt  num_locals;
    FT_UInt  num_globals;

    FT_Int  locals_bias;
    FT_Int  globals_bias;

    FT_Byte**  locals;
    FT_Byte**  globals;

    FT_Byte**  glyph_names;   /* for pure CFF fonts only  */
    FT_UInt    num_glyphs;    /* number of glyphs in font */

    FT_Render_Mode  hint_mode;

    FT_Bool  seac;

    CFF_SubFont  current_subfont; /* for current glyph_index */

    CFF_Decoder_Get_Glyph_Callback   get_glyph_callback;
    CFF_Decoder_Free_Glyph_Callback  free_glyph_callback;

  } CFF_Decoder;


  typedef const struct CFF_Decoder_FuncsRec_*  CFF_Decoder_Funcs;

  typedef struct  CFF_Decoder_FuncsRec_
  {
    void
    (*init)( CFF_Decoder*                     decoder,
             TT_Face                          face,
             CFF_Size                         size,
             CFF_GlyphSlot                    slot,
             FT_Bool                          hinting,
             FT_Render_Mode                   hint_mode,
             CFF_Decoder_Get_Glyph_Callback   get_callback,
             CFF_Decoder_Free_Glyph_Callback  free_callback );

    FT_Error
    (*prepare)( CFF_Decoder*  decoder,
                CFF_Size      size,
                FT_UInt       glyph_index );

#ifdef CFF_CONFIG_OPTION_OLD_ENGINE
    FT_Error
    (*parse_charstrings_old)( CFF_Decoder*  decoder,
                              FT_Byte*      charstring_base,
                              FT_ULong      charstring_len,
                              FT_Bool       in_dict );
#endif

    FT_Error
    (*parse_charstrings)( PS_Decoder*  decoder,
                          FT_Byte*     charstring_base,
                          FT_ULong     charstring_len );

  } CFF_Decoder_FuncsRec;


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                            AFM PARSER                         *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  typedef struct AFM_ParserRec_*  AFM_Parser;

  typedef struct  AFM_Parser_FuncsRec_
  {
    FT_Error
    (*init)( AFM_Parser  parser,
             FT_Memory   memory,
             FT_Byte*    base,
             FT_Byte*    limit );

    void
    (*done)( AFM_Parser  parser );

    FT_Error
    (*parse)( AFM_Parser  parser );

  } AFM_Parser_FuncsRec;


  typedef struct AFM_StreamRec_*  AFM_Stream;


  /**************************************************************************
   *
   * @struct:
   *   AFM_ParserRec
   *
   * @description:
   *   An AFM_Parser is a parser for the AFM files.
   *
   * @fields:
   *   memory ::
   *     The object used for memory operations (alloc and realloc).
   *
   *   stream ::
   *     This is an opaque object.
   *
   *   FontInfo ::
   *     The result will be stored here.
   *
   *   get_index ::
   *     A user provided function to get a glyph index by its name.
   */
  typedef struct  AFM_ParserRec_
  {
    FT_Memory     memory;
    AFM_Stream    stream;

    AFM_FontInfo  FontInfo;

    FT_Int
    (*get_index)( const char*  name,
                  FT_Offset    len,
                  void*        user_data );

    void*         user_data;

  } AFM_ParserRec;


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                     TYPE1 CHARMAPS                            *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  typedef const struct T1_CMap_ClassesRec_*  T1_CMap_Classes;

  typedef struct T1_CMap_ClassesRec_
  {
    FT_CMap_Class  standard;
    FT_CMap_Class  expert;
    FT_CMap_Class  custom;
    FT_CMap_Class  unicode;

  } T1_CMap_ClassesRec;


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                        PSAux Module Interface                 *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  typedef struct  PSAux_ServiceRec_
  {
    /* don't use `PS_Table_Funcs' and friends to avoid compiler warnings */
    const PS_Table_FuncsRec*    ps_table_funcs;
    const PS_Parser_FuncsRec*   ps_parser_funcs;
    const T1_Builder_FuncsRec*  t1_builder_funcs;
    const T1_Decoder_FuncsRec*  t1_decoder_funcs;

    void
    (*t1_decrypt)( FT_Byte*   buffer,
                   FT_Offset  length,
                   FT_UShort  seed );

    FT_UInt32
    (*cff_random)( FT_UInt32  r );

    void
    (*ps_decoder_init)( PS_Decoder*  ps_decoder,
                        void*        decoder,
                        FT_Bool      is_t1 );

    void
    (*t1_make_subfont)( FT_Face      face,
                        PS_Private   priv,
                        CFF_SubFont  subfont );

    T1_CMap_Classes  t1_cmap_classes;

    /* fields after this comment line were added after version 2.1.10 */
    const AFM_Parser_FuncsRec*  afm_parser_funcs;

    const CFF_Decoder_FuncsRec*  cff_decoder_funcs;

  } PSAux_ServiceRec, *PSAux_Service;

  /* backward compatible type definition */
  typedef PSAux_ServiceRec   PSAux_Interface;


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                 Some convenience functions                    *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

#define IS_PS_NEWLINE( ch ) \
  ( (ch) == '\r' ||         \
    (ch) == '\n' )

#define IS_PS_SPACE( ch )  \
  ( (ch) == ' '         || \
    IS_PS_NEWLINE( ch ) || \
    (ch) == '\t'        || \
    (ch) == '\f'        || \
    (ch) == '\0' )

#define IS_PS_SPECIAL( ch )       \
  ( (ch) == '/'                || \
    (ch) == '(' || (ch) == ')' || \
    (ch) == '<' || (ch) == '>' || \
    (ch) == '[' || (ch) == ']' || \
    (ch) == '{' || (ch) == '}' || \
    (ch) == '%'                )

#define IS_PS_DELIM( ch )  \
  ( IS_PS_SPACE( ch )   || \
    IS_PS_SPECIAL( ch ) )

#define IS_PS_DIGIT( ch )        \
  ( (ch) >= '0' && (ch) <= '9' )

#define IS_PS_XDIGIT( ch )            \
  ( IS_PS_DIGIT( ch )              || \
    ( (ch) >= 'A' && (ch) <= 'F' ) || \
    ( (ch) >= 'a' && (ch) <= 'f' ) )

#define IS_PS_BASE85( ch )       \
  ( (ch) >= '!' && (ch) <= 'u' )

#define IS_PS_TOKEN( cur, limit, token )                                \
  ( (char)(cur)[0] == (token)[0]                                     && \
    ( (cur) + sizeof ( (token) ) == (limit) ||                          \
      ( (cur) + sizeof( (token) ) < (limit)          &&                 \
        IS_PS_DELIM( (cur)[sizeof ( (token) ) - 1] ) ) )             && \
    ft_strncmp( (char*)(cur), (token), sizeof ( (token) ) - 1 ) == 0 )


FT_END_HEADER

#endif /* PSAUX_H_ */


/* END */
