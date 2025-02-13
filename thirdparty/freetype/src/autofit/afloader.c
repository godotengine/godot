/****************************************************************************
 *
 * afloader.c
 *
 *   Auto-fitter glyph loading routines (body).
 *
 * Copyright (C) 2003-2023 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#include "afglobal.h"
#include "afloader.h"
#include "afhints.h"
#include "aferrors.h"
#include "afmodule.h"

#include <freetype/internal/ftcalc.h>


  /* Initialize glyph loader. */

  FT_LOCAL_DEF( void )
  af_loader_init( AF_Loader      loader,
                  AF_GlyphHints  hints )
  {
    FT_ZERO( loader );

    loader->hints = hints;
  }


  /* Reset glyph loader and compute globals if necessary. */

  FT_LOCAL_DEF( FT_Error )
  af_loader_reset( AF_Loader  loader,
                   AF_Module  module,
                   FT_Face    face )
  {
    FT_Error  error = FT_Err_Ok;


    loader->face    = face;
    loader->globals = (AF_FaceGlobals)face->autohint.data;

    if ( !loader->globals )
    {
      error = af_face_globals_new( face, &loader->globals, module );
      if ( !error )
      {
        face->autohint.data      = (FT_Pointer)loader->globals;
        face->autohint.finalizer = af_face_globals_free;
      }
    }

    return error;
  }


  /* Finalize glyph loader. */

  FT_LOCAL_DEF( void )
  af_loader_done( AF_Loader  loader )
  {
    loader->face    = NULL;
    loader->globals = NULL;
    loader->hints   = NULL;
  }


#define af_intToFixed( i ) \
          ( (FT_Fixed)( (FT_UInt32)(i) << 16 ) )
#define af_fixedToInt( x ) \
          ( (FT_Short)( ( (FT_UInt32)(x) + 0x8000U ) >> 16 ) )
#define af_floatToFixed( f ) \
          ( (FT_Fixed)( (f) * 65536.0 + 0.5 ) )


  static FT_Error
  af_loader_embolden_glyph_in_slot( AF_Loader        loader,
                                    FT_Face          face,
                                    AF_StyleMetrics  style_metrics )
  {
    FT_Error  error = FT_Err_Ok;

    FT_GlyphSlot           slot    = face->glyph;
    AF_FaceGlobals         globals = loader->globals;
    AF_WritingSystemClass  writing_system_class;

    FT_Size_Metrics*  size_metrics = &face->size->internal->autohint_metrics;

    FT_Pos  stdVW = 0;
    FT_Pos  stdHW = 0;

    FT_Bool  size_changed = size_metrics->x_ppem !=
                              globals->stem_darkening_for_ppem;

    FT_Fixed  em_size  = af_intToFixed( face->units_per_EM );

    FT_Matrix  scale_down_matrix = { 0x10000L, 0, 0, 0x10000L };


    /* Skip stem darkening for broken fonts. */
    if ( !face->units_per_EM )
    {
      error = FT_ERR( Corrupted_Font_Header );
      goto Exit;
    }

    /*
     * We depend on the writing system (script analyzers) to supply
     * standard widths for the script of the glyph we are looking at.  If
     * it can't deliver, stem darkening is disabled.
     */
    writing_system_class =
      af_writing_system_classes[style_metrics->style_class->writing_system];

    if ( writing_system_class->style_metrics_getstdw )
      writing_system_class->style_metrics_getstdw( style_metrics,
                                                   &stdHW,
                                                   &stdVW );
    else
    {
      error = FT_ERR( Unimplemented_Feature );
      goto Exit;
    }

    if ( size_changed                                               ||
         ( stdVW > 0 && stdVW != globals->standard_vertical_width ) )
    {
      FT_Fixed  darken_by_font_units_x, darken_x;


      darken_by_font_units_x =
         af_loader_compute_darkening( loader,
                                      face,
                                      stdVW ) ;
      darken_x = FT_MulFix( darken_by_font_units_x,
                            size_metrics->x_scale );

      globals->standard_vertical_width = stdVW;
      globals->stem_darkening_for_ppem = size_metrics->x_ppem;
      globals->darken_x                = af_fixedToInt( darken_x );
    }

    if ( size_changed                                                 ||
         ( stdHW > 0 && stdHW != globals->standard_horizontal_width ) )
    {
      FT_Fixed  darken_by_font_units_y, darken_y;


      darken_by_font_units_y =
         af_loader_compute_darkening( loader,
                                      face,
                                      stdHW ) ;
      darken_y = FT_MulFix( darken_by_font_units_y,
                            size_metrics->y_scale );

      globals->standard_horizontal_width = stdHW;
      globals->stem_darkening_for_ppem   = size_metrics->x_ppem;
      globals->darken_y                  = af_fixedToInt( darken_y );

      /*
       * Scale outlines down on the Y-axis to keep them inside their blue
       * zones.  The stronger the emboldening, the stronger the downscaling
       * (plus heuristical padding to prevent outlines still falling out
       * their zones due to rounding).
       *
       * Reason: `FT_Outline_Embolden' works by shifting the rightmost
       * points of stems farther to the right, and topmost points farther
       * up.  This positions points on the Y-axis outside their
       * pre-computed blue zones and leads to distortion when applying the
       * hints in the code further below.  Code outside this emboldening
       * block doesn't know we are presenting it with modified outlines the
       * analyzer didn't see!
       *
       * An unfortunate side effect of downscaling is that the emboldening
       * effect is slightly decreased.  The loss becomes more pronounced
       * versus the CFF driver at smaller sizes, e.g., at 9ppem and below.
       */
      globals->scale_down_factor =
        FT_DivFix( em_size - ( darken_by_font_units_y + af_intToFixed( 8 ) ),
                   em_size );
    }

    FT_Outline_EmboldenXY( &slot->outline,
                           globals->darken_x,
                           globals->darken_y );

    scale_down_matrix.yy = globals->scale_down_factor;
    FT_Outline_Transform( &slot->outline, &scale_down_matrix );

  Exit:
    return error;
  }


  /* Load the glyph at index into the current slot of a face and hint it. */

  FT_LOCAL_DEF( FT_Error )
  af_loader_load_glyph( AF_Loader  loader,
                        AF_Module  module,
                        FT_Face    face,
                        FT_UInt    glyph_index,
                        FT_Int32   load_flags )
  {
    FT_Error  error;

    FT_Size           size          = face->size;
    FT_Size_Internal  size_internal = size->internal;
    FT_GlyphSlot      slot          = face->glyph;
    FT_Slot_Internal  slot_internal = slot->internal;
    FT_GlyphLoader    gloader       = slot_internal->loader;

    AF_GlyphHints          hints         = loader->hints;
    AF_ScalerRec           scaler;
    AF_StyleMetrics        style_metrics;
    FT_UInt                style_options = AF_STYLE_NONE_DFLT;
    AF_StyleClass          style_class;
    AF_WritingSystemClass  writing_system_class;


    FT_ZERO( &scaler );

    if ( !size_internal->autohint_metrics.x_scale                          ||
         size_internal->autohint_mode != FT_LOAD_TARGET_MODE( load_flags ) )
    {
      /* switching between hinting modes usually means different scaling */
      /* values; this later on enforces recomputation of everything      */
      /* related to the current size                                     */

      size_internal->autohint_mode    = FT_LOAD_TARGET_MODE( load_flags );
      size_internal->autohint_metrics = size->metrics;

#ifdef AF_CONFIG_OPTION_TT_SIZE_METRICS
      {
        FT_Size_Metrics*  size_metrics = &size_internal->autohint_metrics;


        /* set metrics to integer values and adjust scaling accordingly; */
        /* this is the same setup as with TrueType fonts, cf. function   */
        /* `tt_size_reset' in file `ttobjs.c'                            */
        size_metrics->ascender  = FT_PIX_ROUND(
                                    FT_MulFix( face->ascender,
                                               size_metrics->y_scale ) );
        size_metrics->descender = FT_PIX_ROUND(
                                    FT_MulFix( face->descender,
                                               size_metrics->y_scale ) );
        size_metrics->height    = FT_PIX_ROUND(
                                    FT_MulFix( face->height,
                                               size_metrics->y_scale ) );

        size_metrics->x_scale     = FT_DivFix( size_metrics->x_ppem << 6,
                                               face->units_per_EM );
        size_metrics->y_scale     = FT_DivFix( size_metrics->y_ppem << 6,
                                               face->units_per_EM );
        size_metrics->max_advance = FT_PIX_ROUND(
                                      FT_MulFix( face->max_advance_width,
                                                 size_metrics->x_scale ) );
      }
#endif /* AF_CONFIG_OPTION_TT_SIZE_METRICS */
    }

    /*
     * TODO: This code currently doesn't support fractional advance widths,
     * i.e., placing hinted glyphs at anything other than integer
     * x-positions.  This is only relevant for the warper code, which
     * scales and shifts glyphs to optimize blackness of stems (hinting on
     * the x-axis by nature places things on pixel integers, hinting on the
     * y-axis only, i.e., LIGHT mode, doesn't touch the x-axis).  The delta
     * values of the scaler would need to be adjusted.
     */
    scaler.face    = face;
    scaler.x_scale = size_internal->autohint_metrics.x_scale;
    scaler.x_delta = 0;
    scaler.y_scale = size_internal->autohint_metrics.y_scale;
    scaler.y_delta = 0;

    scaler.render_mode = FT_LOAD_TARGET_MODE( load_flags );
    scaler.flags       = 0;

    /* note that the fallback style can't be changed anymore */
    /* after the first call of `af_loader_load_glyph'        */
    error = af_loader_reset( loader, module, face );
    if ( error )
      goto Exit;

    /*
     * Glyphs (really code points) are assigned to scripts.  Script
     * analysis is done lazily: For each glyph that passes through here,
     * the corresponding script analyzer is called, but returns immediately
     * if it has been run already.
     */
    error = af_face_globals_get_metrics( loader->globals, glyph_index,
                                         style_options, &style_metrics );
    if ( error )
      goto Exit;

    style_class          = style_metrics->style_class;
    writing_system_class =
      af_writing_system_classes[style_class->writing_system];

    loader->metrics = style_metrics;

    if ( writing_system_class->style_metrics_scale )
      writing_system_class->style_metrics_scale( style_metrics, &scaler );
    else
      style_metrics->scaler = scaler;

    if ( writing_system_class->style_hints_init )
    {
      error = writing_system_class->style_hints_init( hints,
                                                      style_metrics );
      if ( error )
        goto Exit;
    }

    /*
     * Do the main work of `af_loader_load_glyph'.  Note that we never have
     * to deal with composite glyphs as those get loaded into
     * FT_GLYPH_FORMAT_OUTLINE by the recursed `FT_Load_Glyph' function.
     * In the rare cases where FT_LOAD_NO_RECURSE is set, it implies
     * FT_LOAD_NO_SCALE and as such the auto-hinter is never called.
     */
    load_flags |=  FT_LOAD_NO_SCALE         |
                   FT_LOAD_IGNORE_TRANSFORM |
                   FT_LOAD_LINEAR_DESIGN;
    load_flags &= ~FT_LOAD_RENDER;

    error = FT_Load_Glyph( face, glyph_index, load_flags );
    if ( error )
      goto Exit;

    /*
     * Apply stem darkening (emboldening) here before hints are applied to
     * the outline.  Glyphs are scaled down proportionally to the
     * emboldening so that curve points don't fall outside their
     * precomputed blue zones.
     *
     * Any emboldening done by the font driver (e.g., the CFF driver)
     * doesn't reach here because the autohinter loads the unprocessed
     * glyphs in font units for analysis (functions `af_*_metrics_init_*')
     * and then above to prepare it for the rasterizers by itself,
     * independently of the font driver.  So emboldening must be done here,
     * within the autohinter.
     *
     * All glyphs to be autohinted pass through here one by one.  The
     * standard widths can therefore change from one glyph to the next,
     * depending on what script a glyph is assigned to (each script has its
     * own set of standard widths and other metrics).  The darkening amount
     * must therefore be recomputed for each size and
     * `standard_{vertical,horizontal}_width' change.
     *
     * Ignore errors and carry on without emboldening.
     *
     */

    /* stem darkening only works well in `light' mode */
    if ( scaler.render_mode == FT_RENDER_MODE_LIGHT    &&
         ( !face->internal->no_stem_darkening        ||
           ( face->internal->no_stem_darkening < 0 &&
             !module->no_stem_darkening            ) ) )
      af_loader_embolden_glyph_in_slot( loader, face, style_metrics );

    loader->transformed = slot_internal->glyph_transformed;
    if ( loader->transformed )
    {
      FT_Matrix  inverse;


      loader->trans_matrix = slot_internal->glyph_matrix;
      loader->trans_delta  = slot_internal->glyph_delta;

      inverse = loader->trans_matrix;
      if ( !FT_Matrix_Invert( &inverse ) )
        FT_Vector_Transform( &loader->trans_delta, &inverse );
    }

    switch ( slot->format )
    {
    case FT_GLYPH_FORMAT_OUTLINE:
      /* translate the loaded glyph when an internal transform is needed */
      if ( loader->transformed )
        FT_Outline_Translate( &slot->outline,
                              loader->trans_delta.x,
                              loader->trans_delta.y );

      /* compute original horizontal phantom points */
      /* (and ignore vertical ones)                 */
      loader->pp1.x = hints->x_delta;
      loader->pp1.y = hints->y_delta;
      loader->pp2.x = FT_MulFix( slot->metrics.horiAdvance,
                                 hints->x_scale ) + hints->x_delta;
      loader->pp2.y = hints->y_delta;

      /* be sure to check for spacing glyphs */
      if ( slot->outline.n_points == 0 )
        goto Hint_Metrics;

      /* now load the slot image into the auto-outline */
      /* and run the automatic hinting process         */
      if ( writing_system_class->style_hints_apply )
      {
        error = writing_system_class->style_hints_apply(
                  glyph_index,
                  hints,
                  &gloader->base.outline,
                  style_metrics );
        if ( error )
          goto Exit;
      }

      /* we now need to adjust the metrics according to the change in */
      /* width/positioning that occurred during the hinting process   */
      if ( scaler.render_mode != FT_RENDER_MODE_LIGHT )
      {
        AF_AxisHints  axis  = &hints->axis[AF_DIMENSION_HORZ];


        if ( axis->num_edges > 1 && AF_HINTS_DO_ADVANCE( hints ) )
        {
          AF_Edge  edge1 = axis->edges;         /* leftmost edge  */
          AF_Edge  edge2 = edge1 +
                           axis->num_edges - 1; /* rightmost edge */

          FT_Pos  old_rsb = loader->pp2.x - edge2->opos;
          /* loader->pp1.x is always zero at this point of time */
          FT_Pos  old_lsb = edge1->opos;     /* - loader->pp1.x */
          FT_Pos  new_lsb = edge1->pos;

          /* remember unhinted values to later account */
          /* for rounding errors                       */
          FT_Pos  pp1x_uh = new_lsb    - old_lsb;
          FT_Pos  pp2x_uh = edge2->pos + old_rsb;


          /* prefer too much space over too little space */
          /* for very small sizes                        */

          if ( old_lsb < 24 )
            pp1x_uh -= 8;

          if ( old_rsb < 24 )
            pp2x_uh += 8;

          loader->pp1.x = FT_PIX_ROUND( pp1x_uh );
          loader->pp2.x = FT_PIX_ROUND( pp2x_uh );

          if ( loader->pp1.x >= new_lsb && old_lsb > 0 )
            loader->pp1.x -= 64;

          if ( loader->pp2.x <= edge2->pos && old_rsb > 0 )
            loader->pp2.x += 64;

          slot->lsb_delta = loader->pp1.x - pp1x_uh;
          slot->rsb_delta = loader->pp2.x - pp2x_uh;
        }
        else
        {
          FT_Pos  pp1x = loader->pp1.x;
          FT_Pos  pp2x = loader->pp2.x;


          loader->pp1.x = FT_PIX_ROUND( pp1x );
          loader->pp2.x = FT_PIX_ROUND( pp2x );

          slot->lsb_delta = loader->pp1.x - pp1x;
          slot->rsb_delta = loader->pp2.x - pp2x;
        }
      }
      /* `light' mode uses integer advance widths */
      /* but sets `lsb_delta' and `rsb_delta'     */
      else
      {
        FT_Pos  pp1x = loader->pp1.x;
        FT_Pos  pp2x = loader->pp2.x;


        loader->pp1.x = FT_PIX_ROUND( pp1x );
        loader->pp2.x = FT_PIX_ROUND( pp2x );

        slot->lsb_delta = loader->pp1.x - pp1x;
        slot->rsb_delta = loader->pp2.x - pp2x;
      }

      break;

    default:
      /* we don't support other formats (yet?) */
      error = FT_THROW( Unimplemented_Feature );
    }

  Hint_Metrics:
    {
      FT_BBox    bbox;
      FT_Vector  vvector;


      vvector.x = slot->metrics.vertBearingX - slot->metrics.horiBearingX;
      vvector.y = slot->metrics.vertBearingY - slot->metrics.horiBearingY;
      vvector.x = FT_MulFix( vvector.x, style_metrics->scaler.x_scale );
      vvector.y = FT_MulFix( vvector.y, style_metrics->scaler.y_scale );

      /* transform the hinted outline if needed */
      if ( loader->transformed )
      {
        FT_Outline_Transform( &gloader->base.outline, &loader->trans_matrix );
        FT_Vector_Transform( &vvector, &loader->trans_matrix );
      }

      /* we must translate our final outline by -pp1.x and compute */
      /* the new metrics                                           */
      if ( loader->pp1.x )
        FT_Outline_Translate( &gloader->base.outline, -loader->pp1.x, 0 );

      FT_Outline_Get_CBox( &gloader->base.outline, &bbox );

      bbox.xMin = FT_PIX_FLOOR( bbox.xMin );
      bbox.yMin = FT_PIX_FLOOR( bbox.yMin );
      bbox.xMax = FT_PIX_CEIL(  bbox.xMax );
      bbox.yMax = FT_PIX_CEIL(  bbox.yMax );

      slot->metrics.width        = bbox.xMax - bbox.xMin;
      slot->metrics.height       = bbox.yMax - bbox.yMin;
      slot->metrics.horiBearingX = bbox.xMin;
      slot->metrics.horiBearingY = bbox.yMax;

      slot->metrics.vertBearingX = FT_PIX_FLOOR( bbox.xMin + vvector.x );
      slot->metrics.vertBearingY = FT_PIX_FLOOR( bbox.yMax + vvector.y );

      /* for mono-width fonts (like Andale, Courier, etc.) we need */
      /* to keep the original rounded advance width; ditto for     */
      /* digits if all have the same advance width                 */
      if ( scaler.render_mode != FT_RENDER_MODE_LIGHT                       &&
           ( FT_IS_FIXED_WIDTH( slot->face )                              ||
             ( af_face_globals_is_digit( loader->globals, glyph_index ) &&
               style_metrics->digits_have_same_width                    ) ) )
      {
        slot->metrics.horiAdvance =
          FT_MulFix( slot->metrics.horiAdvance,
                     style_metrics->scaler.x_scale );

        /* Set delta values to 0.  Otherwise code that uses them is */
        /* going to ruin the fixed advance width.                   */
        slot->lsb_delta = 0;
        slot->rsb_delta = 0;
      }
      else
      {
        /* non-spacing glyphs must stay as-is */
        if ( slot->metrics.horiAdvance )
          slot->metrics.horiAdvance = loader->pp2.x - loader->pp1.x;
      }

      slot->metrics.vertAdvance = FT_MulFix( slot->metrics.vertAdvance,
                                             style_metrics->scaler.y_scale );

      slot->metrics.horiAdvance = FT_PIX_ROUND( slot->metrics.horiAdvance );
      slot->metrics.vertAdvance = FT_PIX_ROUND( slot->metrics.vertAdvance );

      slot->format  = FT_GLYPH_FORMAT_OUTLINE;
    }

  Exit:
    return error;
  }


  /*
   * Compute amount of font units the face should be emboldened by, in
   * analogy to the CFF driver's `cf2_computeDarkening' function.  See there
   * for details of the algorithm.
   *
   * XXX: Currently a crude adaption of the original algorithm.  Do better?
   */
  FT_LOCAL_DEF( FT_Fixed )
  af_loader_compute_darkening( AF_Loader  loader,
                               FT_Face    face,
                               FT_Pos     standard_width )
  {
    AF_Module  module = loader->globals->module;

    FT_UShort  units_per_EM;
    FT_Fixed   ppem, em_ratio;
    FT_Fixed   stem_width, stem_width_per_1000, scaled_stem, darken_amount;
    FT_Int     log_base_2;
    FT_Int     x1, y1, x2, y2, x3, y3, x4, y4;


    ppem         = FT_MAX( af_intToFixed( 4 ),
                           af_intToFixed( face->size->metrics.x_ppem ) );
    units_per_EM = face->units_per_EM;

    em_ratio = FT_DivFix( af_intToFixed( 1000 ),
                          af_intToFixed ( units_per_EM ) );
    if ( em_ratio < af_floatToFixed( .01 ) )
    {
      /* If something goes wrong, don't embolden. */
      return 0;
    }

    x1 = module->darken_params[0];
    y1 = module->darken_params[1];
    x2 = module->darken_params[2];
    y2 = module->darken_params[3];
    x3 = module->darken_params[4];
    y3 = module->darken_params[5];
    x4 = module->darken_params[6];
    y4 = module->darken_params[7];

    if ( standard_width <= 0 )
    {
      stem_width          = af_intToFixed( 75 ); /* taken from cf2font.c */
      stem_width_per_1000 = stem_width;
    }
    else
    {
      stem_width          = af_intToFixed( standard_width );
      stem_width_per_1000 = FT_MulFix( stem_width, em_ratio );
    }

    log_base_2 = FT_MSB( (FT_UInt32)stem_width_per_1000 ) +
                 FT_MSB( (FT_UInt32)ppem );

    if ( log_base_2 >= 46 )
    {
      /* possible overflow */
      scaled_stem = af_intToFixed( x4 );
    }
    else
      scaled_stem = FT_MulFix( stem_width_per_1000, ppem );

    /* now apply the darkening parameters */
    if ( scaled_stem < af_intToFixed( x1 ) )
      darken_amount = FT_DivFix( af_intToFixed( y1 ), ppem );

    else if ( scaled_stem < af_intToFixed( x2 ) )
    {
      FT_Int  xdelta = x2 - x1;
      FT_Int  ydelta = y2 - y1;
      FT_Int  x      = stem_width_per_1000 -
                       FT_DivFix( af_intToFixed( x1 ), ppem );


      if ( !xdelta )
        goto Try_x3;

      darken_amount = FT_MulDiv( x, ydelta, xdelta ) +
                      FT_DivFix( af_intToFixed( y1 ), ppem );
    }

    else if ( scaled_stem < af_intToFixed( x3 ) )
    {
    Try_x3:
      {
        FT_Int  xdelta = x3 - x2;
        FT_Int  ydelta = y3 - y2;
        FT_Int  x      = stem_width_per_1000 -
                         FT_DivFix( af_intToFixed( x2 ), ppem );


        if ( !xdelta )
          goto Try_x4;

        darken_amount = FT_MulDiv( x, ydelta, xdelta ) +
                        FT_DivFix( af_intToFixed( y2 ), ppem );
      }
    }

    else if ( scaled_stem < af_intToFixed( x4 ) )
    {
    Try_x4:
      {
        FT_Int  xdelta = x4 - x3;
        FT_Int  ydelta = y4 - y3;
        FT_Int  x      = stem_width_per_1000 -
                         FT_DivFix( af_intToFixed( x3 ), ppem );


        if ( !xdelta )
          goto Use_y4;

        darken_amount = FT_MulDiv( x, ydelta, xdelta ) +
                        FT_DivFix( af_intToFixed( y3 ), ppem );
      }
    }

    else
    {
    Use_y4:
      darken_amount = FT_DivFix( af_intToFixed( y4 ), ppem );
    }

    /* Convert darken_amount from per 1000 em to true character space. */
    return FT_DivFix( darken_amount, em_ratio );
  }


/* END */
