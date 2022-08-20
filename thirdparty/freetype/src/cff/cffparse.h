/****************************************************************************
 *
 * cffparse.h
 *
 *   CFF token stream parser (specification)
 *
 * Copyright (C) 1996-2022 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#ifndef CFFPARSE_H_
#define CFFPARSE_H_


#include <freetype/internal/cfftypes.h>
#include <freetype/internal/ftobjs.h>


FT_BEGIN_HEADER


  /* CFF uses constant parser stack size; */
  /* CFF2 can increase from default 193   */
#define CFF_MAX_STACK_DEPTH  96

  /*
   * There are plans to remove the `maxstack' operator in a forthcoming
   * revision of the CFF2 specification, increasing the (then static) stack
   * size to 513.  By making the default stack size equal to the maximum
   * stack size, the operator is essentially disabled, which has the
   * desired effect in FreeType.
   */
#define CFF2_MAX_STACK      513
#define CFF2_DEFAULT_STACK  513

#define CFF_CODE_TOPDICT    0x1000
#define CFF_CODE_PRIVATE    0x2000
#define CFF2_CODE_TOPDICT   0x3000
#define CFF2_CODE_FONTDICT  0x4000
#define CFF2_CODE_PRIVATE   0x5000


  typedef struct  CFF_ParserRec_
  {
    FT_Library  library;
    FT_Byte*    start;
    FT_Byte*    limit;
    FT_Byte*    cursor;

    FT_Byte**   stack;
    FT_Byte**   top;
    FT_UInt     stackSize;  /* allocated size */

#ifdef CFF_CONFIG_OPTION_OLD_ENGINE
    FT_ListRec  t2_strings;
#endif /* CFF_CONFIG_OPTION_OLD_ENGINE */

    FT_UInt     object_code;
    void*       object;

    FT_UShort   num_designs; /* a copy of `CFF_FontRecDict->num_designs' */
    FT_UShort   num_axes;    /* a copy of `CFF_FontRecDict->num_axes'    */

  } CFF_ParserRec, *CFF_Parser;


  FT_LOCAL( FT_Long )
  cff_parse_num( CFF_Parser  parser,
                 FT_Byte**   d );

  FT_LOCAL( FT_Error )
  cff_parser_init( CFF_Parser  parser,
                   FT_UInt     code,
                   void*       object,
                   FT_Library  library,
                   FT_UInt     stackSize,
                   FT_UShort   num_designs,
                   FT_UShort   num_axes );

  FT_LOCAL( void )
  cff_parser_done( CFF_Parser  parser );

  FT_LOCAL( FT_Error )
  cff_parser_run( CFF_Parser  parser,
                  FT_Byte*    start,
                  FT_Byte*    limit );


  enum
  {
    cff_kind_none = 0,
    cff_kind_num,
    cff_kind_fixed,
    cff_kind_fixed_thousand,
    cff_kind_string,
    cff_kind_bool,
    cff_kind_delta,
    cff_kind_callback,
    cff_kind_blend,

    cff_kind_max  /* do not remove */
  };


  /* now generate handlers for the most simple fields */
  typedef FT_Error  (*CFF_Field_Reader)( CFF_Parser  parser );

  typedef struct  CFF_Field_Handler_
  {
    int               kind;
    int               code;
    FT_UInt           offset;
    FT_Byte           size;
    CFF_Field_Reader  reader;
    FT_UInt           array_max;
    FT_UInt           count_offset;

#ifdef FT_DEBUG_LEVEL_TRACE
    const char*       id;
#endif

  } CFF_Field_Handler;


FT_END_HEADER


#ifdef CFF_CONFIG_OPTION_OLD_ENGINE
  typedef struct  CFF_T2_String_
  {
    FT_Byte*  start;
    FT_Byte*  limit;

  } CFF_T2_StringRec, *CFF_T2_String;
#endif /* CFF_CONFIG_OPTION_OLD_ENGINE */

#endif /* CFFPARSE_H_ */


/* END */
