/***************************************************************************/
/*                                                                         */
/*  aftypes.h                                                              */
/*                                                                         */
/*    Auto-fitter types (specification only).                              */
/*                                                                         */
/*  Copyright 2003-2009, 2011-2012 by                                      */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


  /*************************************************************************
   *
   *  The auto-fitter is a complete rewrite of the old auto-hinter.
   *  Its main feature is the ability to differentiate between different
   *  scripts in order to apply language-specific rules.
   *
   *  The code has also been compartmentized into several entities that
   *  should make algorithmic experimentation easier than with the old
   *  code.
   *
   *  Finally, we get rid of the Catharon license, since this code is
   *  released under the FreeType one.
   *
   *************************************************************************/


#ifndef __AFTYPES_H__
#define __AFTYPES_H__

#include <ft2build.h>

#include FT_FREETYPE_H
#include FT_OUTLINE_H
#include FT_INTERNAL_OBJECTS_H
#include FT_INTERNAL_DEBUG_H


FT_BEGIN_HEADER

  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                    D E B U G G I N G                          *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

#ifdef FT_DEBUG_AUTOFIT

#include FT_CONFIG_STANDARD_LIBRARY_H

extern int    _af_debug_disable_horz_hints;
extern int    _af_debug_disable_vert_hints;
extern int    _af_debug_disable_blue_hints;
extern void*  _af_debug_hints;

#endif /* FT_DEBUG_AUTOFIT */


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                 U T I L I T Y   S T U F F                     *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  typedef struct  AF_WidthRec_
  {
    FT_Pos  org;  /* original position/width in font units              */
    FT_Pos  cur;  /* current/scaled position/width in device sub-pixels */
    FT_Pos  fit;  /* current/fitted position/width in device sub-pixels */

  } AF_WidthRec, *AF_Width;


  FT_LOCAL( void )
  af_sort_pos( FT_UInt  count,
               FT_Pos*  table );

  FT_LOCAL( void )
  af_sort_and_quantize_widths( FT_UInt*  count,
                               AF_Width  widths,
                               FT_Pos    threshold );


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                   A N G L E   T Y P E S                       *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  /*
   *  The auto-fitter doesn't need a very high angular accuracy;
   *  this allows us to speed up some computations considerably with a
   *  light Cordic algorithm (see afangles.c).
   */

  typedef FT_Int  AF_Angle;


#define AF_ANGLE_PI   256
#define AF_ANGLE_2PI  ( AF_ANGLE_PI * 2 )
#define AF_ANGLE_PI2  ( AF_ANGLE_PI / 2 )
#define AF_ANGLE_PI4  ( AF_ANGLE_PI / 4 )


#if 0
  /*
   *  compute the angle of a given 2-D vector
   */
  FT_LOCAL( AF_Angle )
  af_angle_atan( FT_Pos  dx,
                 FT_Pos  dy );


  /*
   *  compute `angle2 - angle1'; the result is always within
   *  the range [-AF_ANGLE_PI .. AF_ANGLE_PI - 1]
   */
  FT_LOCAL( AF_Angle )
  af_angle_diff( AF_Angle  angle1,
                 AF_Angle  angle2 );
#endif /* 0 */


#define AF_ANGLE_DIFF( result, angle1, angle2 ) \
  FT_BEGIN_STMNT                                \
    AF_Angle  _delta = (angle2) - (angle1);     \
                                                \
                                                \
    _delta %= AF_ANGLE_2PI;                     \
    if ( _delta < 0 )                           \
      _delta += AF_ANGLE_2PI;                   \
                                                \
    if ( _delta > AF_ANGLE_PI )                 \
      _delta -= AF_ANGLE_2PI;                   \
                                                \
    result = _delta;                            \
  FT_END_STMNT


  /*  opaque handle to glyph-specific hints -- see `afhints.h' for more
   *  details
   */
  typedef struct AF_GlyphHintsRec_*  AF_GlyphHints;


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                       S C A L E R S                           *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  /*
   *  A scaler models the target pixel device that will receive the
   *  auto-hinted glyph image.
   */

  typedef enum  AF_ScalerFlags_
  {
    AF_SCALER_FLAG_NO_HORIZONTAL = 1,  /* disable horizontal hinting */
    AF_SCALER_FLAG_NO_VERTICAL   = 2,  /* disable vertical hinting   */
    AF_SCALER_FLAG_NO_ADVANCE    = 4   /* disable advance hinting    */

  } AF_ScalerFlags;


  typedef struct  AF_ScalerRec_
  {
    FT_Face         face;        /* source font face                        */
    FT_Fixed        x_scale;     /* from font units to 1/64th device pixels */
    FT_Fixed        y_scale;     /* from font units to 1/64th device pixels */
    FT_Pos          x_delta;     /* in 1/64th device pixels                 */
    FT_Pos          y_delta;     /* in 1/64th device pixels                 */
    FT_Render_Mode  render_mode; /* monochrome, anti-aliased, LCD, etc.     */
    FT_UInt32       flags;       /* additional control flags, see above     */

  } AF_ScalerRec, *AF_Scaler;


#define AF_SCALER_EQUAL_SCALES( a, b )      \
          ( (a)->x_scale == (b)->x_scale && \
            (a)->y_scale == (b)->y_scale && \
            (a)->x_delta == (b)->x_delta && \
            (a)->y_delta == (b)->y_delta )


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                       S C R I P T S                           *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  /*
   *  The list of known scripts.  Each different script corresponds to the
   *  following information:
   *
   *   - A set of Unicode ranges to test whether the face supports the
   *     script.
   *
   *   - A specific global analyzer that will compute global metrics
   *     specific to the script.
   *
   *   - A specific glyph analyzer that will compute segments and
   *     edges for each glyph covered by the script.
   *
   *   - A specific grid-fitting algorithm that will distort the
   *     scaled glyph outline according to the results of the glyph
   *     analyzer.
   *
   *  Note that a given analyzer and/or grid-fitting algorithm can be
   *  used by more than one script.
   */

  typedef enum  AF_Script_
  {
    AF_SCRIPT_DUMMY = 0,
    AF_SCRIPT_LATIN = 1,
    AF_SCRIPT_CJK   = 2,
    AF_SCRIPT_INDIC = 3,
#ifdef FT_OPTION_AUTOFIT2
    AF_SCRIPT_LATIN2 = 4,
#endif

    /* add new scripts here.  Don't forget to update the list in */
    /* `afglobal.c'.                                             */

    AF_SCRIPT_MAX   /* do not remove */

  } AF_Script;


  typedef struct AF_ScriptClassRec_ const*  AF_ScriptClass;
  typedef struct AF_FaceGlobalsRec_*        AF_FaceGlobals;

  typedef struct  AF_ScriptMetricsRec_
  {
    AF_ScriptClass  clazz;
    AF_ScalerRec    scaler;
    FT_Bool         digits_have_same_width;

    AF_FaceGlobals  globals;    /* to access properties */

  } AF_ScriptMetricsRec, *AF_ScriptMetrics;


  /*  This function parses an FT_Face to compute global metrics for
   *  a specific script.
   */
  typedef FT_Error
  (*AF_Script_InitMetricsFunc)( AF_ScriptMetrics  metrics,
                                FT_Face           face );

  typedef void
  (*AF_Script_ScaleMetricsFunc)( AF_ScriptMetrics  metrics,
                                 AF_Scaler         scaler );

  typedef void
  (*AF_Script_DoneMetricsFunc)( AF_ScriptMetrics  metrics );


  typedef FT_Error
  (*AF_Script_InitHintsFunc)( AF_GlyphHints     hints,
                              AF_ScriptMetrics  metrics );

  typedef void
  (*AF_Script_ApplyHintsFunc)( AF_GlyphHints     hints,
                               FT_Outline*       outline,
                               AF_ScriptMetrics  metrics );


  typedef struct  AF_Script_UniRangeRec_
  {
    FT_UInt32  first;
    FT_UInt32  last;

  } AF_Script_UniRangeRec;

#define AF_UNIRANGE_REC( a, b ) { (FT_UInt32)(a), (FT_UInt32)(b) }

  typedef const AF_Script_UniRangeRec  *AF_Script_UniRange;


  typedef struct  AF_ScriptClassRec_
  {
    AF_Script           script;
    AF_Script_UniRange  script_uni_ranges; /* last must be { 0, 0 }        */
    FT_UInt32           standard_char;     /* for default width and height */

    FT_Offset                   script_metrics_size;
    AF_Script_InitMetricsFunc   script_metrics_init;
    AF_Script_ScaleMetricsFunc  script_metrics_scale;
    AF_Script_DoneMetricsFunc   script_metrics_done;

    AF_Script_InitHintsFunc     script_hints_init;
    AF_Script_ApplyHintsFunc    script_hints_apply;

  } AF_ScriptClassRec;


  /* Declare and define vtables for classes */
#ifndef FT_CONFIG_OPTION_PIC

#define AF_DECLARE_SCRIPT_CLASS( script_class ) \
  FT_CALLBACK_TABLE const AF_ScriptClassRec     \
  script_class;

#define AF_DEFINE_SCRIPT_CLASS( script_class, script_, ranges, def_char,   \
                                m_size,                                    \
                                m_init, m_scale, m_done, h_init, h_apply ) \
  FT_CALLBACK_TABLE_DEF const AF_ScriptClassRec  script_class =            \
  {                                                                        \
    script_,                                                               \
    ranges,                                                                \
    def_char,                                                              \
                                                                           \
    m_size,                                                                \
                                                                           \
    m_init,                                                                \
    m_scale,                                                               \
    m_done,                                                                \
                                                                           \
    h_init,                                                                \
    h_apply                                                                \
  };

#else /* FT_CONFIG_OPTION_PIC */

#define AF_DECLARE_SCRIPT_CLASS( script_class )             \
  FT_LOCAL( void )                                          \
  FT_Init_Class_ ## script_class( AF_ScriptClassRec*  ac );

#define AF_DEFINE_SCRIPT_CLASS( script_class, script_, ranges, def_char,   \
                                m_size,                                    \
                                m_init, m_scale, m_done, h_init, h_apply ) \
  FT_LOCAL_DEF( void )                                                     \
  FT_Init_Class_ ## script_class( AF_ScriptClassRec*  ac )                 \
  {                                                                        \
    ac->script               = script_;                                    \
    ac->script_uni_ranges    = ranges;                                     \
    ac->default_char         = def_char;                                   \
                                                                           \
    ac->script_metrics_size  = m_size;                                     \
                                                                           \
    ac->script_metrics_init  = m_init;                                     \
    ac->script_metrics_scale = m_scale;                                    \
    ac->script_metrics_done  = m_done;                                     \
                                                                           \
    ac->script_hints_init    = h_init;                                     \
    ac->script_hints_apply   = h_apply;                                    \
  }

#endif /* FT_CONFIG_OPTION_PIC */


/* */

FT_END_HEADER

#endif /* __AFTYPES_H__ */


/* END */
