/****************************************************************************
 *
 * hvfload.c
 *
 *   HVF glyph loading (body).
 *
 * Copyright (C) 2025-2026 by
 * Apple Inc.
 * written by Deborah Goldsmith <goldsmit@apple.com>
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#include <freetype/internal/ftdebug.h>
#include <freetype/internal/ftstream.h>
#include <freetype/internal/ftgloadr.h>
#include <freetype/ftoutln.h>

#include "hvfload.h"
#include "hvfobjs.h"
#include "hvferror.h"


#undef  FT_COMPONENT
#define FT_COMPONENT  hvfload


#ifdef FT_CONFIG_OPTION_HVF

  /**************************************************************************
   *
   * HVF Coordinate Conversion Macros
   *
   */

  /* Convert HVF coordinates to FreeType 26.6 fixed point */
  /* using a pre-calculated scale factor.                 */
#define HVF_COORD_TO_FIXED( coord, scale_fixed )      \
          ( (FT_F26Dot6)( (coord) * (scale_fixed) ) )

  /* Pre-calculate scale factor for efficient coordinate conversion.   */
  /* FreeType scale factors are 16.16 fixed point; convert to floating */
  /* point for HVF.                                                    */
#define FT_SCALE_TO_HVF_SCALE_FIXED( scale, apply_scaling ) \
          ( (apply_scaling) ? (HVFXYCoord)(scale) / 65536.0 \
                            : (HVFXYCoord)64.0 )

  /* Cache management: clear cache after this many glyph loads. */
  /* TODO: Future enhancement: make this a settable property on */
  /*       the HVF driver.                                      */
#define HVF_CACHE_CLEAR_COUNT  17


  /**************************************************************************
   *
   * HVF axis values setup with TrueType integration
   *
   */
  static FT_Error
  hvf_set_variation_axes( HVF_Face  face )
  {
    FT_Error  error = FT_Err_Ok;

    int  axis_count;

    FT_UInt  i;
    FT_UInt  max_axes;

    HVFAxisValue  axis_value;


    /* Only the HVF API calls need `@available` protection. */
    if ( HVF_IS_AVAILABLE )
    {
      /* Get HVF axis count for current part. */
      axis_count =
        HVF_render_part_axis_count( (HVFPartRenderer*)face->renderer );
      if ( axis_count <= 0 )
      {
        /* No axes for this part - this is normal for non-variable glyphs. */
        return FT_Err_Ok;
      }

      /* Calculate maximum axes to set - only set axes that have */
      /* pre-stored coordinates.                                 */
      /*                                                         */
      /* Note: `HVF_set_render_part` already sets all axes to    */
      /*       default (0.0), so we only need to override axes   */
      /*       that have actual variation coordinates.           */
      max_axes = 0;
      if ( face->axis_coords && face->num_axes > 0 )
      {
        max_axes = face->num_axes;
        if ( max_axes > (FT_UInt)axis_count )
          max_axes = (FT_UInt)axis_count;
      }

      /* Set only axes with pre-stored coordinates */
      /* (`HVF_set_render_part` handles defaults). */
      for ( i = 0; i < max_axes; i++ )
      {
        int  hvf_result;


        /* Use pre-converted HVF axis coordinate directly */
        /* (no conversion needed).                        */
        axis_value = ((HVFAxisValue*)face->axis_coords)[i];
        FT_TRACE5(( "hvf_set_variation_axes:"
                    " axis %u = %f (pre-converted HVF coord)\n",
                    i, axis_value ));

        /* Set the axis value in HVF renderer. */
        hvf_result = HVF_set_axis_value( (HVFPartRenderer*)face->renderer,
                                         (int)i,
                                         axis_value );
#ifdef FT_DEBUG_LEVEL_TRACE
        if ( hvf_result != 0 )
        {
          /* Continue with other axes even if one fails. */
          FT_TRACE1(( "hvf_set_variation_axes:"
                      " HVF_set_axis_value failed for axis %u (error %d)\n",
                      i, hvf_result ));
        }
#else
        FT_UNUSED( hvf_result );
#endif
      }

      /* Remaining axes already set to defaults by `HVF_set_render_part` - */
      /* no action needed.                                                 */
      FT_TRACE3(( "hvf_set_variation_axes:"
                  " configured %u of %d axis value%s\n"
                  "                       "
                  " using pre-converted HVF coordinates\n",
                  max_axes, axis_count, axis_count == 1 ? "" : "s" ));
    }
    else
    {
      FT_TRACE3(( "hvf_set_variation_axes:"
                  " HVF not available at runtime\n" ));
      return FT_THROW( Unimplemented_Feature );
    }

    return error;
  }


  /**************************************************************************
   *
   * HVF instruction callback - using `FT_GlyphLoader`
   *
   */
  static HVFPartRenderAction
  hvf_render_callback( HVFPartRenderInstruction        instruction,
                       const HVFPartRenderParameters*  params,
                       void*                           user_data )
  {
    HVF_RenderContext*  ctx = (HVF_RenderContext*)user_data;

    FT_Error  error;


    switch ( instruction )
    {
    case HVFPartRenderInstructionBeginPart:
      FT_TRACE5(( "hvf_render: BeginPart (part %zu)\n",
                  params->beginPart.partInfo.partId ));
      return HVFPartRenderActionContinue;

    case HVFPartRenderInstructionBeginPath:
      FT_TRACE5(( "hvf_render: BeginPath\n" ));
      ctx->path_begun = 1;
      return HVFPartRenderActionContinue;

    case HVFPartRenderInstructionAddPoint:
      {
        FT_Int  n;


        /* Move-to or line-to point with pre-calculated scaling factors. */
        error = FT_GLYPHLOADER_CHECK_POINTS( ctx->loader, 1, 0 );
        if ( error )
          return HVFPartRenderActionStop;

        n = ctx->outline->n_points;

        ctx->outline->points[n].x =
          HVF_COORD_TO_FIXED( params->addPoint.pt.x, ctx->x_scale_fixed );
        ctx->outline->points[n].y =
          HVF_COORD_TO_FIXED( params->addPoint.pt.y, ctx->y_scale_fixed );
        ctx->outline->tags[n] = FT_CURVE_TAG_ON;
        ctx->outline->n_points++;

        FT_TRACE5(( "hvf_render: AddPoint (%.2f, %.2f)\n",
                    (double)ctx->outline->points[n].x / 64,
                    (double)ctx->outline->points[n].y / 64 ));
        return HVFPartRenderActionContinue;
      }

    case HVFPartRenderInstructionAddQuad:
      {
        FT_Int  n;


        /* Quadratic curve with pre-calculated scaling factors. */
        error = FT_GLYPHLOADER_CHECK_POINTS( ctx->loader, 2, 0 );
        if ( error )
          return HVFPartRenderActionStop;

        n = ctx->outline->n_points;

        /* Control point. */
        ctx->outline->points[n].x =
          HVF_COORD_TO_FIXED( params->addQuad.offpt.x, ctx->x_scale_fixed );
        ctx->outline->points[n].y =
          HVF_COORD_TO_FIXED( params->addQuad.offpt.y, ctx->y_scale_fixed );
        ctx->outline->tags[n] = FT_CURVE_TAG_CONIC;

        /* End point. */
        ctx->outline->points[n + 1].x =
          HVF_COORD_TO_FIXED( params->addQuad.onpt.x, ctx->x_scale_fixed );
        ctx->outline->points[n + 1].y =
          HVF_COORD_TO_FIXED( params->addQuad.onpt.y, ctx->y_scale_fixed );
        ctx->outline->tags[n + 1] = FT_CURVE_TAG_ON;

        ctx->outline->n_points += 2;

        FT_TRACE5(( "hvf_render: AddQuad (%.2f,%.2f) (%.2f,%.2f)\n",
                    (double)ctx->outline->points[n].x / 64,
                    (double)ctx->outline->points[n].y / 64,
                    (double)ctx->outline->points[n + 1].x / 64,
                    (double)ctx->outline->points[n + 1].y / 64 ));
        return HVFPartRenderActionContinue;
      }

    case HVFPartRenderInstructionAddCubic:
      {
        FT_Int  n;


        /* Cubic curve with pre-calculated scaling factors. */
        error = FT_GLYPHLOADER_CHECK_POINTS( ctx->loader, 3, 0 );
        if ( error )
          return HVFPartRenderActionStop;

        n = ctx->outline->n_points;

        /* First control point */
        ctx->outline->points[n].x =
          HVF_COORD_TO_FIXED( params->addCubic.cp1.x, ctx->x_scale_fixed );
        ctx->outline->points[n].y =
          HVF_COORD_TO_FIXED( params->addCubic.cp1.y, ctx->y_scale_fixed );
        ctx->outline->tags[n] = FT_CURVE_TAG_CUBIC;

        /* Second control point */
        ctx->outline->points[n + 1].x =
          HVF_COORD_TO_FIXED( params->addCubic.cp2.x, ctx->x_scale_fixed );
        ctx->outline->points[n + 1].y =
          HVF_COORD_TO_FIXED( params->addCubic.cp2.y, ctx->y_scale_fixed );
        ctx->outline->tags[n + 1] = FT_CURVE_TAG_CUBIC;

        /* End point */
        ctx->outline->points[n + 2].x =
          HVF_COORD_TO_FIXED( params->addCubic.onpt.x, ctx->x_scale_fixed );
        ctx->outline->points[n + 2].y =
          HVF_COORD_TO_FIXED( params->addCubic.onpt.y, ctx->y_scale_fixed );
        ctx->outline->tags[n + 2] = FT_CURVE_TAG_ON;

        ctx->outline->n_points += 3;

        FT_TRACE5(( "hvf_render:"
                    " AddCubic (%.2f,%.2f) (%.2f,%.2f) (%.2f,%.2f)\n",
                    (double)ctx->outline->points[n].x / 64,
                    (double)ctx->outline->points[n].y / 64,
                    (double)ctx->outline->points[n + 1].x / 64,
                    (double)ctx->outline->points[n + 1].y / 64,
                    (double)ctx->outline->points[n + 2].x / 64,
                    (double)ctx->outline->points[n + 2].y / 64 ));
        return HVFPartRenderActionContinue;
      }

    case HVFPartRenderInstructionClosePath:
      if ( ctx->path_begun && ctx->outline->n_points > 0 )
      {
        /* Check space for contour. */
        error = FT_GLYPHLOADER_CHECK_POINTS( ctx->loader, 0, 1 );
        if ( error )
          return HVFPartRenderActionStop;
        
        /* Close current contour. */
        ctx->outline->contours[ctx->outline->n_contours++] =
          (FT_UShort)( ctx->outline->n_points - 1 );
        
        ctx->path_begun = 0;
      }
      
      FT_TRACE5(( "hvf_render: ClosePath\n" ));
      return HVFPartRenderActionContinue;

    case HVFPartRenderInstructionEndPath:
      FT_TRACE5(( "hvf_render: EndPath\n" ));
      return HVFPartRenderActionContinue;

    case HVFPartRenderInstructionEndPart:
      FT_TRACE5(( "hvf_render: EndPart (part %zu)\n",
                  params->endPart.partInfo.partId ));
      return HVFPartRenderActionContinue;

    case HVFPartRenderInstructionStop:
      FT_TRACE5(( "hvf_render: Stop\n" ));
      return HVFPartRenderActionStop;

    default:
      FT_TRACE1(( "hvf_render: Unknown instruction %d\n", instruction ));
      return HVFPartRenderActionStop;
    }
  }


  /**************************************************************************
   *
   * @Function:
   *   hvf_slot_load_glyph
   *
   * @Description:
   *   Load a glyph using HVF rendering with TrueType integration.
   *
   * @Input:
   *   glyph ::
   *     The glyph slot.
   *
   *   size ::
   *     The size object.
   *
   *   glyph_index ::
   *     The glyph index.
   *
   *   load_flags ::
   *     Load flags.
   *
   * @Return:
   *   FreeType error code.  0 means success.
   */
  FT_LOCAL_DEF( FT_Error )
  hvf_slot_load_glyph( FT_GlyphSlot  glyph,
                       FT_Size       size,
                       FT_UInt       glyph_index,
                       FT_Int32      load_flags )
  {
    HVF_Face           face   = (HVF_Face)glyph->face;
    FT_GlyphLoader     loader = glyph->internal->loader;
    HVF_RenderContext  context;

    FT_Error  error = FT_Err_Ok;

    FT_Bool  apply_scaling = !( load_flags & FT_LOAD_NO_SCALE );


    if ( !face->renderer )
    {
      FT_ERROR(( "hvf_slot_load_glyph: HVF renderer not initialized\n" ));
      return FT_THROW( Invalid_Handle );
    }

    if ( glyph_index >= (FT_UInt)face->root.root.num_glyphs )
      return FT_THROW( Invalid_Glyph_Index );

    /* Initialize render context. */
    context.loader     = loader;
    context.path_begun = 0;

    /* Pre-calculate scaling factors based on `FT_LOAD_NO_SCALE` flag. */
    context.x_scale_fixed =
      FT_SCALE_TO_HVF_SCALE_FIXED( size->metrics.x_scale, apply_scaling );
    context.y_scale_fixed =
      FT_SCALE_TO_HVF_SCALE_FIXED( size->metrics.y_scale, apply_scaling );
    
    FT_TRACE3(( "hvf_slot_load_glyph:"
                " %s\n"
                "                    "
                " (x_scale_fixed=%.1f, y_scale_fixed=%.1f)\n",
                apply_scaling ? "scaling enabled"
                              : "FT_LOAD_NO_SCALE - no scaling applied",
                context.x_scale_fixed,
                context.y_scale_fixed ));

    /* Initialize glyph loader. */
    FT_GlyphLoader_Rewind( loader );
    context.outline = &loader->current.outline;

    /* Only the HVF API calls need `@available` protection. */
    if ( HVF_IS_AVAILABLE )
    {
      int  hvf_result;


      /* Configure HVF renderer for this glyph. */
      hvf_result = HVF_set_render_part( (HVFPartRenderer*)face->renderer,
                                        (HVFPartIndex)glyph_index );
      if ( hvf_result != 0 )
      {
        FT_TRACE1(( "hvf_slot_load_glyph:"
                    " HVF_set_render_part failed (error %d)\n",
                    hvf_result ));
        return FT_THROW( Invalid_Glyph_Index );
      }

      /* Set variation axis values for this glyph */
      /* (using TT_Face infrastructure).          */
      error = hvf_set_variation_axes( face );
#ifdef FT_DEBUG_LEVEL_TRACE
      if ( error )
      {
        /* Continue with rendering even if axis setting fails. */
        FT_TRACE1(( "hvf_slot_load_glyph:"
                    " hvf_set_variation_axes failed (error %d)\n", error ));
      }
#endif
      error = FT_Err_Ok;  /* axis failure is non-fatal */

      /* Render the glyph using HVF. */
      hvf_result = HVF_render_current_part( (HVFPartRenderer*)face->renderer,
                                            hvf_render_callback,
                                            &context );
      if ( hvf_result != 0 )
      {
        FT_TRACE1(( "hvf_slot_load_glyph:"
                    " HVF_render_current_part failed (%d)\n", hvf_result ));
        return FT_THROW( Invalid_Glyph_Index );
      }
    }
    else
    {
      FT_TRACE3(( "hvf_slot_load_glyph: HVF not available at runtime\n" ));
      return FT_THROW( Unimplemented_Feature );
    }

    /* Close any open path. */
    if ( context.path_begun && context.outline->n_points > 0 )
    {
      /* Check space for contour. */
      error = FT_GLYPHLOADER_CHECK_POINTS( loader, 0, 1 );
      if ( !error )
      {
        context.outline->contours[context.outline->n_contours++] =
          (FT_UShort)( context.outline->n_points - 1 );
      }
    }

    /* Finalize glyph data with `FT_GlyphLoader`. */
    FT_GlyphLoader_Add( loader );

    /* Set up glyph slot. */
    glyph->format  = FT_GLYPH_FORMAT_OUTLINE;
    glyph->outline = loader->base.outline;

    /* Nearly all HVF glyphs have overlapping contours, */
    /* since it is a stroke-based format                */
    glyph->outline.flags |= FT_OUTLINE_OVERLAP;

    /* Set glyph metrics - get from TrueType infrastructure when possible. */
    {
      FT_BBox       bbox;
      FT_UShort     advance_width = 0;
      FT_Short      left_bearing  = 0;

      TT_Face       tt_face = (TT_Face)face;
      SFNT_Service  sfnt    = (SFNT_Service)tt_face->sfnt;

      FT_Bool       has_vertical_info;
      FT_Pos        top;      /* vertical top side bearing */
      FT_Pos        advance;  /* vertical advance height   */


      /* Get horizontal metrics from TrueType tables if available. */
      if ( sfnt && glyph_index < (FT_UInt)face->root.root.num_glyphs )
        sfnt->get_metrics( tt_face, 0, glyph_index,
                           &left_bearing, &advance_width );
      else
      {
        /* Fallback to reasonable defaults. */
        advance_width = (FT_UShort)( size->metrics.x_ppem );
        left_bearing  = 0;
      }

      glyph->metrics.horiAdvance  = advance_width;
      glyph->linearHoriAdvance    = advance_width;

      /* Get vertical metrics from vmtx table or synthesize them,     */
      /* consistent with TrueType and CFF drivers for SFNT-based fonts. */
      has_vertical_info =
        FT_BOOL( tt_face->vertical_info                   &&
                 tt_face->vertical.number_Of_VMetrics > 0 );

      if ( has_vertical_info )
      {
        FT_Short   tsb = 0;
        FT_UShort  ah  = 0;


        sfnt->get_metrics( tt_face, 1, glyph_index, &tsb, &ah );

        top     = tsb;
        advance = ah;
      }
      else
      {
        /* Use OS/2 or horizontal header for vertical advance. */
        if ( tt_face->os2.version != 0xFFFFU )
          advance = (FT_Pos)( tt_face->os2.sTypoAscender -
                              tt_face->os2.sTypoDescender );
        else
          advance = (FT_Pos)( tt_face->horizontal.Ascender -
                              tt_face->horizontal.Descender );

        top = 0;  /* computed after bbox is known */
      }

      glyph->linearVertAdvance = advance;

      /* Scale table-derived metrics if needed.                     */
      /* Outline coordinates are already in the target coordinate   */
      /* space (scaled in callback, or 26.6 font units for NO_SCALE) */
      /* so bbox-derived metrics need no additional scaling.         */
      if ( apply_scaling )
      {
        FT_Fixed  x_scale = size->metrics.x_scale;
        FT_Fixed  y_scale = size->metrics.y_scale;


        glyph->metrics.horiAdvance = FT_MulFix( glyph->metrics.horiAdvance,
                                                 x_scale );
        top     = FT_MulFix( top, y_scale );
        advance = FT_MulFix( advance, y_scale );
      }

      /* Compute width, height, and bearing metrics from the glyph  */
      /* outline bounding box, consistent with TrueType and CFF     */
      /* drivers.                                                    */
      FT_Outline_Get_CBox( &glyph->outline, &bbox );

      glyph->metrics.width        = bbox.xMax - bbox.xMin;
      glyph->metrics.height       = bbox.yMax - bbox.yMin;
      glyph->metrics.horiBearingX = bbox.xMin;
      glyph->metrics.horiBearingY = bbox.yMax;

      /* Finalize vertical metrics.                                   */
      /* vertBearingX centers the glyph horizontally for vertical     */
      /* layout, matching both TrueType and CFF drivers.              */
      /* When no vmtx data is available, vertBearingY centers the     */
      /* glyph vertically within the advance height.                  */
      if ( !has_vertical_info )
        top = ( advance - glyph->metrics.height ) / 2;

      glyph->metrics.vertBearingX = glyph->metrics.horiBearingX -
                                      glyph->metrics.horiAdvance / 2;
      glyph->metrics.vertBearingY = top;
      glyph->metrics.vertAdvance  = advance;
    }

    /* Cache management - clear cache every */
    /* `HVF_CACHE_CLEAR_COUNT` glyphs.      */
    face->cache_count++;
    if ( face->cache_count >= HVF_CACHE_CLEAR_COUNT )
    {
      if ( HVF_IS_AVAILABLE )
      {
        HVF_clear_part_cache( (HVFPartRenderer*)face->renderer );
        face->cache_count = 0;
        FT_TRACE3(( "hvf_slot_load_glyph: cleared HVF cache\n" ));
      }
#ifdef FT_DEBUG_LEVEL_TRACE
      else
        FT_TRACE3(( "hvf_slot_load_glyph:"
                    " HVF not available for cache clearing\n" ));
#endif
    }

    FT_TRACE2(( "hvf_slot_load_glyph: glyph %u loaded successfully\n",
                glyph_index ));

    return error;
  }

#else /* !FT_CONFIG_OPTION_HVF */

  FT_LOCAL_DEF( FT_Error )
  hvf_slot_load_glyph( FT_GlyphSlot  glyph,
                       FT_Size       size,
                       FT_UInt       glyph_index,
                       FT_Int32      load_flags )
  {
    FT_UNUSED( glyph );
    FT_UNUSED( size );
    FT_UNUSED( glyph_index );
    FT_UNUSED( load_flags );

    return FT_THROW( Unimplemented_Feature );
  }

#endif /* !FT_CONFIG_OPTION_HVF */


/* END */
