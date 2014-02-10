/***************************************************************************/
/*                                                                         */
/*  afcjk.h                                                                */
/*                                                                         */
/*    Auto-fitter hinting routines for CJK script (specification).         */
/*                                                                         */
/*  Copyright 2006, 2007, 2011, 2012 by                                    */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


#ifndef __AFCJK_H__
#define __AFCJK_H__

#include "afhints.h"
#include "aflatin.h"


FT_BEGIN_HEADER


  /* the CJK-specific script class */

  AF_DECLARE_SCRIPT_CLASS( af_cjk_script_class )

  /* CJK (global) metrics management */

  /*
   *  CJK glyphs tend to fill the square.  So we have both vertical and
   *  horizontal blue zones.  But some glyphs have flat bounding strokes that
   *  leave some space between neighbour glyphs.
   */
  enum
  {
    AF_CJK_BLUE_TOP,
    AF_CJK_BLUE_BOTTOM,
    AF_CJK_BLUE_LEFT,
    AF_CJK_BLUE_RIGHT,

    AF_CJK_BLUE_MAX
  };


#define AF_CJK_MAX_WIDTHS  16
#define AF_CJK_MAX_BLUES   AF_CJK_BLUE_MAX


  enum
  {
    AF_CJK_BLUE_ACTIVE     = 1 << 0,
    AF_CJK_BLUE_IS_TOP     = 1 << 1,
    AF_CJK_BLUE_IS_RIGHT   = 1 << 2,
    AF_CJK_BLUE_ADJUSTMENT = 1 << 3,  /* used for scale adjustment */
                                      /* optimization              */
    AF_CJK_BLUE_FLAG_MAX
  };


  typedef struct  AF_CJKBlueRec_
  {
    AF_WidthRec  ref;
    AF_WidthRec  shoot; /* undershoot */
    FT_UInt      flags;

  } AF_CJKBlueRec, *AF_CJKBlue;


  typedef struct  AF_CJKAxisRec_
  {
    FT_Fixed       scale;
    FT_Pos         delta;

    FT_UInt        width_count;
    AF_WidthRec    widths[AF_CJK_MAX_WIDTHS];
    FT_Pos         edge_distance_threshold;
    FT_Pos         standard_width;
    FT_Bool        extra_light;

    /* used for horizontal metrics too for CJK */
    FT_Bool        control_overshoot;
    FT_UInt        blue_count;
    AF_CJKBlueRec  blues[AF_CJK_BLUE_MAX];

    FT_Fixed       org_scale;
    FT_Pos         org_delta;

  } AF_CJKAxisRec, *AF_CJKAxis;


  typedef struct  AF_CJKMetricsRec_
  {
    AF_ScriptMetricsRec  root;
    FT_UInt              units_per_em;
    AF_CJKAxisRec        axis[AF_DIMENSION_MAX];

  } AF_CJKMetricsRec, *AF_CJKMetrics;


#ifdef AF_CONFIG_OPTION_CJK
  FT_LOCAL( FT_Error )
  af_cjk_metrics_init( AF_CJKMetrics  metrics,
                       FT_Face        face );

  FT_LOCAL( void )
  af_cjk_metrics_scale( AF_CJKMetrics  metrics,
                        AF_Scaler      scaler );

  FT_LOCAL( FT_Error )
  af_cjk_hints_init( AF_GlyphHints  hints,
                     AF_CJKMetrics  metrics );

  FT_LOCAL( FT_Error )
  af_cjk_hints_apply( AF_GlyphHints  hints,
                      FT_Outline*    outline,
                      AF_CJKMetrics  metrics );

  /* shared; called from afindic.c */
  FT_LOCAL( void )
  af_cjk_metrics_check_digits( AF_CJKMetrics  metrics,
                               FT_Face        face );

  FT_LOCAL( void )
  af_cjk_metrics_init_widths( AF_CJKMetrics  metrics,
                              FT_Face        face );
#endif /* AF_CONFIG_OPTION_CJK */


/* */

FT_END_HEADER

#endif /* __AFCJK_H__ */


/* END */
