/****************************************************************************
 *
 * afdummy.c
 *
 *   Auto-fitter dummy routines to be used if no hinting should be
 *   performed (body).
 *
 * Copyright (C) 2003-2024 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#include "afdummy.h"
#include "afhints.h"
#include "aferrors.h"


  static FT_Error
  af_dummy_hints_init( AF_GlyphHints    hints,
                       AF_StyleMetrics  metrics )
  {
    af_glyph_hints_rescale( hints, metrics );

    hints->x_scale = metrics->scaler.x_scale;
    hints->y_scale = metrics->scaler.y_scale;
    hints->x_delta = metrics->scaler.x_delta;
    hints->y_delta = metrics->scaler.y_delta;

    return FT_Err_Ok;
  }


  static FT_Error
  af_dummy_hints_apply( FT_UInt          glyph_index,
                        AF_GlyphHints    hints,
                        FT_Outline*      outline,
                        AF_StyleMetrics  metrics )
  {
    FT_Error  error;

    FT_UNUSED( glyph_index );
    FT_UNUSED( metrics );


    error = af_glyph_hints_reload( hints, outline );
    if ( !error )
      af_glyph_hints_save( hints, outline );

    return error;
  }


  AF_DEFINE_WRITING_SYSTEM_CLASS(
    af_dummy_writing_system_class,

    AF_WRITING_SYSTEM_DUMMY,

    sizeof ( AF_StyleMetricsRec ),

    (AF_WritingSystem_InitMetricsFunc) NULL,                /* style_metrics_init    */
    (AF_WritingSystem_ScaleMetricsFunc)NULL,                /* style_metrics_scale   */
    (AF_WritingSystem_DoneMetricsFunc) NULL,                /* style_metrics_done    */
    (AF_WritingSystem_GetStdWidthsFunc)NULL,                /* style_metrics_getstdw */

    (AF_WritingSystem_InitHintsFunc)   af_dummy_hints_init, /* style_hints_init      */
    (AF_WritingSystem_ApplyHintsFunc)  af_dummy_hints_apply /* style_hints_apply     */
  )


/* END */
