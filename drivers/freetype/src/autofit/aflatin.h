/***************************************************************************/
/*                                                                         */
/*  aflatin.h                                                              */
/*                                                                         */
/*    Auto-fitter hinting routines for latin script (specification).       */
/*                                                                         */
/*  Copyright 2003-2007, 2009, 2011-2012 by                                */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


#ifndef __AFLATIN_H__
#define __AFLATIN_H__

#include "afhints.h"


FT_BEGIN_HEADER


  /* the latin-specific script class */

  AF_DECLARE_SCRIPT_CLASS( af_latin_script_class )


  /* constants are given with units_per_em == 2048 in mind */
#define AF_LATIN_CONSTANT( metrics, c )                                      \
  ( ( (c) * (FT_Long)( (AF_LatinMetrics)(metrics) )->units_per_em ) / 2048 )


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****            L A T I N   G L O B A L   M E T R I C S            *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/


  /*
   *  The following declarations could be embedded in the file `aflatin.c';
   *  they have been made semi-public to allow alternate script hinters to
   *  re-use some of them.
   */


  /* Latin (global) metrics management */

  enum
  {
    AF_LATIN_BLUE_CAPITAL_TOP,
    AF_LATIN_BLUE_CAPITAL_BOTTOM,
    AF_LATIN_BLUE_SMALL_F_TOP,
    AF_LATIN_BLUE_SMALL_TOP,
    AF_LATIN_BLUE_SMALL_BOTTOM,
    AF_LATIN_BLUE_SMALL_MINOR,

    AF_LATIN_BLUE_MAX
  };


#define AF_LATIN_IS_TOP_BLUE( b )  ( (b) == AF_LATIN_BLUE_CAPITAL_TOP || \
                                     (b) == AF_LATIN_BLUE_SMALL_F_TOP || \
                                     (b) == AF_LATIN_BLUE_SMALL_TOP   )

#define AF_LATIN_MAX_WIDTHS  16
#define AF_LATIN_MAX_BLUES   AF_LATIN_BLUE_MAX


  enum
  {
    AF_LATIN_BLUE_ACTIVE     = 1 << 0,  /* set if zone height is <= 3/4px */
    AF_LATIN_BLUE_TOP        = 1 << 1,  /* result of AF_LATIN_IS_TOP_BLUE */
    AF_LATIN_BLUE_ADJUSTMENT = 1 << 2,  /* used for scale adjustment      */
                                        /* optimization                   */
    AF_LATIN_BLUE_FLAG_MAX
  };


  typedef struct  AF_LatinBlueRec_
  {
    AF_WidthRec  ref;
    AF_WidthRec  shoot;
    FT_UInt      flags;

  } AF_LatinBlueRec, *AF_LatinBlue;


  typedef struct  AF_LatinAxisRec_
  {
    FT_Fixed         scale;
    FT_Pos           delta;

    FT_UInt          width_count;                 /* number of used widths */
    AF_WidthRec      widths[AF_LATIN_MAX_WIDTHS]; /* widths array          */
    FT_Pos           edge_distance_threshold;   /* used for creating edges */
    FT_Pos           standard_width;         /* the default stem thickness */
    FT_Bool          extra_light;         /* is standard width very light? */

    /* ignored for horizontal metrics */
    FT_UInt          blue_count;
    AF_LatinBlueRec  blues[AF_LATIN_BLUE_MAX];

    FT_Fixed         org_scale;
    FT_Pos           org_delta;

  } AF_LatinAxisRec, *AF_LatinAxis;


  typedef struct  AF_LatinMetricsRec_
  {
    AF_ScriptMetricsRec  root;
    FT_UInt              units_per_em;
    AF_LatinAxisRec      axis[AF_DIMENSION_MAX];

  } AF_LatinMetricsRec, *AF_LatinMetrics;


  FT_LOCAL( FT_Error )
  af_latin_metrics_init( AF_LatinMetrics  metrics,
                         FT_Face          face );

  FT_LOCAL( void )
  af_latin_metrics_scale( AF_LatinMetrics  metrics,
                          AF_Scaler        scaler );

  FT_LOCAL( void )
  af_latin_metrics_init_widths( AF_LatinMetrics  metrics,
                                FT_Face          face );

  FT_LOCAL( void )
  af_latin_metrics_check_digits( AF_LatinMetrics  metrics,
                                 FT_Face          face );


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****           L A T I N   G L Y P H   A N A L Y S I S             *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  enum
  {
    AF_LATIN_HINTS_HORZ_SNAP   = 1 << 0, /* enable stem width snapping  */
    AF_LATIN_HINTS_VERT_SNAP   = 1 << 1, /* enable stem height snapping */
    AF_LATIN_HINTS_STEM_ADJUST = 1 << 2, /* enable stem width/height    */
                                         /* adjustment                  */
    AF_LATIN_HINTS_MONO        = 1 << 3  /* indicate monochrome         */
                                         /* rendering                   */
  };


#define AF_LATIN_HINTS_DO_HORZ_SNAP( h )             \
  AF_HINTS_TEST_OTHER( h, AF_LATIN_HINTS_HORZ_SNAP )

#define AF_LATIN_HINTS_DO_VERT_SNAP( h )             \
  AF_HINTS_TEST_OTHER( h, AF_LATIN_HINTS_VERT_SNAP )

#define AF_LATIN_HINTS_DO_STEM_ADJUST( h )             \
  AF_HINTS_TEST_OTHER( h, AF_LATIN_HINTS_STEM_ADJUST )

#define AF_LATIN_HINTS_DO_MONO( h )             \
  AF_HINTS_TEST_OTHER( h, AF_LATIN_HINTS_MONO )


  /*
   *  The next functions shouldn't normally be exported.  However, other
   *  scripts might like to use these functions as-is.
   */
  FT_LOCAL( FT_Error )
  af_latin_hints_compute_segments( AF_GlyphHints  hints,
                                   AF_Dimension   dim );

  FT_LOCAL( void )
  af_latin_hints_link_segments( AF_GlyphHints  hints,
                                AF_Dimension   dim );

  FT_LOCAL( FT_Error )
  af_latin_hints_compute_edges( AF_GlyphHints  hints,
                                AF_Dimension   dim );

  FT_LOCAL( FT_Error )
  af_latin_hints_detect_features( AF_GlyphHints  hints,
                                  AF_Dimension   dim );

/* */

FT_END_HEADER

#endif /* __AFLATIN_H__ */


/* END */
