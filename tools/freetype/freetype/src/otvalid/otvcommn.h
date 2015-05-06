/***************************************************************************/
/*                                                                         */
/*  otvcommn.h                                                             */
/*                                                                         */
/*    OpenType common tables validation (specification).                   */
/*                                                                         */
/*  Copyright 2004, 2005, 2007, 2009 by                                    */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


#ifndef __OTVCOMMN_H__
#define __OTVCOMMN_H__


#include <ft2build.h>
#include "otvalid.h"
#include FT_INTERNAL_DEBUG_H


FT_BEGIN_HEADER


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                         VALIDATION                            *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  typedef struct OTV_ValidatorRec_*  OTV_Validator;

  typedef void  (*OTV_Validate_Func)( FT_Bytes       table,
                                      OTV_Validator  valid );

  typedef struct  OTV_ValidatorRec_
  {
    FT_Validator        root;
    FT_UInt             type_count;
    OTV_Validate_Func*  type_funcs;

    FT_UInt             lookup_count;
    FT_UInt             glyph_count;

    FT_UInt             nesting_level;

    OTV_Validate_Func   func[3];

    FT_UInt             extra1;     /* for passing parameters */
    FT_UInt             extra2;
    FT_Bytes            extra3;

#ifdef FT_DEBUG_LEVEL_TRACE
    FT_UInt             debug_indent;
    const FT_String*    debug_function_name[3];
#endif

  } OTV_ValidatorRec;


#undef  FT_INVALID_
#define FT_INVALID_( _prefix, _error )                         \
          ft_validator_error( valid->root, _prefix ## _error )

#define OTV_OPTIONAL_TABLE( _table )  FT_UShort  _table;      \
                                      FT_Bytes   _table ## _p

#define OTV_OPTIONAL_OFFSET( _offset )           \
          FT_BEGIN_STMNT                         \
            _offset ## _p = p;                   \
            _offset       = FT_NEXT_USHORT( p ); \
          FT_END_STMNT

#define OTV_LIMIT_CHECK( _count )                    \
          FT_BEGIN_STMNT                             \
            if ( p + (_count) > valid->root->limit ) \
              FT_INVALID_TOO_SHORT;                  \
          FT_END_STMNT

#define OTV_SIZE_CHECK( _size )                                     \
          FT_BEGIN_STMNT                                            \
            if ( _size > 0 && _size < table_size )                  \
            {                                                       \
              if ( valid->root->level == FT_VALIDATE_PARANOID )     \
                FT_INVALID_OFFSET;                                  \
              else                                                  \
              {                                                     \
                /* strip off `const' */                             \
                FT_Byte*  pp = (FT_Byte*)_size ## _p;               \
                                                                    \
                                                                    \
                FT_TRACE3(( "\n"                                    \
                            "Invalid offset to optional table `%s'" \
                            " set to zero.\n"                       \
                            "\n", #_size ));                        \
                                                                    \
                /* always assume 16bit entities */                  \
                _size = pp[0] = pp[1] = 0;                          \
              }                                                     \
            }                                                       \
          FT_END_STMNT


#define  OTV_NAME_(x)  #x
#define  OTV_NAME(x)   OTV_NAME_(x)

#define  OTV_FUNC_(x)  x##Func
#define  OTV_FUNC(x)   OTV_FUNC_(x)

#ifdef FT_DEBUG_LEVEL_TRACE

#define OTV_NEST1( x )                                     \
          FT_BEGIN_STMNT                                   \
            valid->nesting_level          = 0;             \
            valid->func[0]                = OTV_FUNC( x ); \
            valid->debug_function_name[0] = OTV_NAME( x ); \
          FT_END_STMNT

#define OTV_NEST2( x, y )                                  \
          FT_BEGIN_STMNT                                   \
            valid->nesting_level          = 0;             \
            valid->func[0]                = OTV_FUNC( x ); \
            valid->func[1]                = OTV_FUNC( y ); \
            valid->debug_function_name[0] = OTV_NAME( x ); \
            valid->debug_function_name[1] = OTV_NAME( y ); \
          FT_END_STMNT

#define OTV_NEST3( x, y, z )                               \
          FT_BEGIN_STMNT                                   \
            valid->nesting_level          = 0;             \
            valid->func[0]                = OTV_FUNC( x ); \
            valid->func[1]                = OTV_FUNC( y ); \
            valid->func[2]                = OTV_FUNC( z ); \
            valid->debug_function_name[0] = OTV_NAME( x ); \
            valid->debug_function_name[1] = OTV_NAME( y ); \
            valid->debug_function_name[2] = OTV_NAME( z ); \
          FT_END_STMNT

#define OTV_INIT  valid->debug_indent = 0

#define OTV_ENTER                                                            \
          FT_BEGIN_STMNT                                                     \
            valid->debug_indent += 2;                                        \
            FT_TRACE4(( "%*.s", valid->debug_indent, 0 ));                   \
            FT_TRACE4(( "%s table\n",                                        \
                        valid->debug_function_name[valid->nesting_level] )); \
          FT_END_STMNT

#define OTV_NAME_ENTER( name )                             \
          FT_BEGIN_STMNT                                   \
            valid->debug_indent += 2;                      \
            FT_TRACE4(( "%*.s", valid->debug_indent, 0 )); \
            FT_TRACE4(( "%s table\n", name ));             \
          FT_END_STMNT

#define OTV_EXIT  valid->debug_indent -= 2

#define OTV_TRACE( s )                                     \
          FT_BEGIN_STMNT                                   \
            FT_TRACE4(( "%*.s", valid->debug_indent, 0 )); \
            FT_TRACE4( s );                                \
          FT_END_STMNT

#else   /* !FT_DEBUG_LEVEL_TRACE */

#define OTV_NEST1( x )                            \
          FT_BEGIN_STMNT                          \
            valid->nesting_level = 0;             \
            valid->func[0]       = OTV_FUNC( x ); \
          FT_END_STMNT

#define OTV_NEST2( x, y )                         \
          FT_BEGIN_STMNT                          \
            valid->nesting_level = 0;             \
            valid->func[0]       = OTV_FUNC( x ); \
            valid->func[1]       = OTV_FUNC( y ); \
          FT_END_STMNT

#define OTV_NEST3( x, y, z )                      \
          FT_BEGIN_STMNT                          \
            valid->nesting_level = 0;             \
            valid->func[0]       = OTV_FUNC( x ); \
            valid->func[1]       = OTV_FUNC( y ); \
            valid->func[2]       = OTV_FUNC( z ); \
          FT_END_STMNT

#define OTV_INIT                do { } while ( 0 )
#define OTV_ENTER               do { } while ( 0 )
#define OTV_NAME_ENTER( name )  do { } while ( 0 )
#define OTV_EXIT                do { } while ( 0 )

#define OTV_TRACE( s )          do { } while ( 0 )

#endif  /* !FT_DEBUG_LEVEL_TRACE */


#define OTV_RUN  valid->func[0]


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                       COVERAGE TABLE                          *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  FT_LOCAL( void )
  otv_Coverage_validate( FT_Bytes       table,
                         OTV_Validator  valid,
                         FT_Int         expected_count );

  /* return first covered glyph */
  FT_LOCAL( FT_UInt )
  otv_Coverage_get_first( FT_Bytes  table );

  /* return last covered glyph */
  FT_LOCAL( FT_UInt )
  otv_Coverage_get_last( FT_Bytes  table );

  /* return number of covered glyphs */
  FT_LOCAL( FT_UInt )
  otv_Coverage_get_count( FT_Bytes  table );


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                  CLASS DEFINITION TABLE                       *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  FT_LOCAL( void )
  otv_ClassDef_validate( FT_Bytes       table,
                         OTV_Validator  valid );


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                      DEVICE TABLE                             *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  FT_LOCAL( void )
  otv_Device_validate( FT_Bytes       table,
                       OTV_Validator  valid );


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                           LOOKUPS                             *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  FT_LOCAL( void )
  otv_Lookup_validate( FT_Bytes       table,
                       OTV_Validator  valid );

  FT_LOCAL( void )
  otv_LookupList_validate( FT_Bytes       table,
                           OTV_Validator  valid );


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                        FEATURES                               *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  FT_LOCAL( void )
  otv_Feature_validate( FT_Bytes       table,
                        OTV_Validator  valid );

  /* lookups must already be validated */
  FT_LOCAL( void )
  otv_FeatureList_validate( FT_Bytes       table,
                            FT_Bytes       lookups,
                            OTV_Validator  valid );


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                       LANGUAGE SYSTEM                         *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  FT_LOCAL( void )
  otv_LangSys_validate( FT_Bytes       table,
                        OTV_Validator  valid );


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                           SCRIPTS                             *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  FT_LOCAL( void )
  otv_Script_validate( FT_Bytes       table,
                       OTV_Validator  valid );

  /* features must already be validated */
  FT_LOCAL( void )
  otv_ScriptList_validate( FT_Bytes       table,
                           FT_Bytes       features,
                           OTV_Validator  valid );


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                      UTILITY FUNCTIONS                        *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

#define ChainPosClassSetFunc  otv_x_Ox
#define ChainPosRuleSetFunc   otv_x_Ox
#define ChainSubClassSetFunc  otv_x_Ox
#define ChainSubRuleSetFunc   otv_x_Ox
#define JstfLangSysFunc       otv_x_Ox
#define JstfMaxFunc           otv_x_Ox
#define LigGlyphFunc          otv_x_Ox
#define LigatureArrayFunc     otv_x_Ox
#define LigatureSetFunc       otv_x_Ox
#define PosClassSetFunc       otv_x_Ox
#define PosRuleSetFunc        otv_x_Ox
#define SubClassSetFunc       otv_x_Ox
#define SubRuleSetFunc        otv_x_Ox

  FT_LOCAL( void )
  otv_x_Ox ( FT_Bytes       table,
             OTV_Validator  valid );

#define AlternateSubstFormat1Func     otv_u_C_x_Ox
#define ChainContextPosFormat1Func    otv_u_C_x_Ox
#define ChainContextSubstFormat1Func  otv_u_C_x_Ox
#define ContextPosFormat1Func         otv_u_C_x_Ox
#define ContextSubstFormat1Func       otv_u_C_x_Ox
#define LigatureSubstFormat1Func      otv_u_C_x_Ox
#define MultipleSubstFormat1Func      otv_u_C_x_Ox

  FT_LOCAL( void )
  otv_u_C_x_Ox( FT_Bytes       table,
                OTV_Validator  valid );

#define AlternateSetFunc     otv_x_ux
#define AttachPointFunc      otv_x_ux
#define ExtenderGlyphFunc    otv_x_ux
#define JstfGPOSModListFunc  otv_x_ux
#define JstfGSUBModListFunc  otv_x_ux
#define SequenceFunc         otv_x_ux

  FT_LOCAL( void )
  otv_x_ux( FT_Bytes       table,
            OTV_Validator  valid );

#define PosClassRuleFunc  otv_x_y_ux_sy
#define PosRuleFunc       otv_x_y_ux_sy
#define SubClassRuleFunc  otv_x_y_ux_sy
#define SubRuleFunc       otv_x_y_ux_sy

  FT_LOCAL( void )
  otv_x_y_ux_sy( FT_Bytes       table,
                 OTV_Validator  valid );

#define ChainPosClassRuleFunc  otv_x_ux_y_uy_z_uz_p_sp
#define ChainPosRuleFunc       otv_x_ux_y_uy_z_uz_p_sp
#define ChainSubClassRuleFunc  otv_x_ux_y_uy_z_uz_p_sp
#define ChainSubRuleFunc       otv_x_ux_y_uy_z_uz_p_sp

  FT_LOCAL( void )
  otv_x_ux_y_uy_z_uz_p_sp( FT_Bytes       table,
                           OTV_Validator  valid );

#define ContextPosFormat2Func    otv_u_O_O_x_Onx
#define ContextSubstFormat2Func  otv_u_O_O_x_Onx

  FT_LOCAL( void )
  otv_u_O_O_x_Onx( FT_Bytes       table,
                   OTV_Validator  valid );

#define ContextPosFormat3Func    otv_u_x_y_Ox_sy
#define ContextSubstFormat3Func  otv_u_x_y_Ox_sy

  FT_LOCAL( void )
  otv_u_x_y_Ox_sy( FT_Bytes       table,
                   OTV_Validator  valid );

#define ChainContextPosFormat2Func    otv_u_O_O_O_O_x_Onx
#define ChainContextSubstFormat2Func  otv_u_O_O_O_O_x_Onx

  FT_LOCAL( void )
  otv_u_O_O_O_O_x_Onx( FT_Bytes       table,
                       OTV_Validator  valid );

#define ChainContextPosFormat3Func    otv_u_x_Ox_y_Oy_z_Oz_p_sp
#define ChainContextSubstFormat3Func  otv_u_x_Ox_y_Oy_z_Oz_p_sp

  FT_LOCAL( void )
  otv_u_x_Ox_y_Oy_z_Oz_p_sp( FT_Bytes       table,
                             OTV_Validator  valid );


  FT_LOCAL( FT_UInt )
  otv_GSUBGPOS_get_Lookup_count( FT_Bytes  table );

  FT_LOCAL( FT_UInt )
  otv_GSUBGPOS_have_MarkAttachmentType_flag( FT_Bytes  table );

 /* */

FT_END_HEADER

#endif /* __OTVCOMMN_H__ */


/* END */
