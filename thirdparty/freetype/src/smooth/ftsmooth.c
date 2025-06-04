/****************************************************************************
 *
 * ftsmooth.c
 *
 *   Anti-aliasing renderer interface (body).
 *
 * Copyright (C) 2000-2024 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#include <freetype/internal/ftdebug.h>
#include <freetype/internal/ftobjs.h>
#include <freetype/ftoutln.h>
#include "ftsmooth.h"
#include "ftgrays.h"

#include "ftsmerrs.h"


  /* sets render-specific mode */
  static FT_Error
  ft_smooth_set_mode( FT_Renderer  render,
                      FT_ULong     mode_tag,
                      FT_Pointer   data )
  {
    /* we simply pass it to the raster */
    return render->clazz->raster_class->raster_set_mode( render->raster,
                                                         mode_tag,
                                                         data );
  }

  /* transform a given glyph image */
  static FT_Error
  ft_smooth_transform( FT_Renderer       render,
                       FT_GlyphSlot      slot,
                       const FT_Matrix*  matrix,
                       const FT_Vector*  delta )
  {
    FT_Error  error = FT_Err_Ok;


    if ( slot->format != render->glyph_format )
    {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    if ( matrix )
      FT_Outline_Transform( &slot->outline, matrix );

    if ( delta )
      FT_Outline_Translate( &slot->outline, delta->x, delta->y );

  Exit:
    return error;
  }


  /* return the glyph's control box */
  static void
  ft_smooth_get_cbox( FT_Renderer   render,
                      FT_GlyphSlot  slot,
                      FT_BBox*      cbox )
  {
    FT_ZERO( cbox );

    if ( slot->format == render->glyph_format )
      FT_Outline_Get_CBox( &slot->outline, cbox );
  }

  typedef struct TOrigin_
  {
    unsigned char*  origin;  /* pixmap origin at the bottom-left */
    int             pitch;   /* pitch to go down one row */

  } TOrigin;

#ifndef FT_CONFIG_OPTION_SUBPIXEL_RENDERING

  /* initialize renderer -- init its raster */
  static FT_Error
  ft_smooth_init( FT_Module  module )   /* FT_Renderer */
  {
    FT_Renderer  render = (FT_Renderer)module;

    FT_Vector*  sub = render->root.library->lcd_geometry;


    /* set up default subpixel geometry for striped RGB panels. */
    sub[0].x = -21;
    sub[0].y = 0;
    sub[1].x = 0;
    sub[1].y = 0;
    sub[2].x = 21;
    sub[2].y = 0;

    render->clazz->raster_class->raster_reset( render->raster, NULL, 0 );

    return 0;
  }


  /* This function writes every third byte in direct rendering mode */
  static void
  ft_smooth_lcd_spans( int             y,
                       int             count,
                       const FT_Span*  spans,
                       void*           target_ )   /* TOrigin* */
  {
    TOrigin*  target = (TOrigin*)target_;

    unsigned char*  dst_line = target->origin - y * target->pitch;
    unsigned char*  dst;
    unsigned short  w;


    for ( ; count--; spans++ )
      for ( dst = dst_line + spans->x * 3, w = spans->len; w--; dst += 3 )
        *dst = spans->coverage;
  }


  static FT_Error
  ft_smooth_raster_lcd( FT_Renderer  render,
                        FT_Outline*  outline,
                        FT_Bitmap*   bitmap )
  {
    FT_Error      error = FT_Err_Ok;
    FT_Vector*    sub   = render->root.library->lcd_geometry;
    FT_Pos        x, y;

    FT_Raster_Params   params;
    TOrigin            target;


    /* Render 3 separate coverage bitmaps, shifting the outline.  */
    /* Set up direct rendering to record them on each third byte. */
    params.source     = outline;
    params.flags      = FT_RASTER_FLAG_AA | FT_RASTER_FLAG_DIRECT;
    params.gray_spans = ft_smooth_lcd_spans;
    params.user       = &target;

    params.clip_box.xMin = 0;
    params.clip_box.yMin = 0;
    params.clip_box.xMax = bitmap->width;
    params.clip_box.yMax = bitmap->rows;

    if ( bitmap->pitch < 0 )
      target.origin = bitmap->buffer;
    else
      target.origin = bitmap->buffer
                      + ( bitmap->rows - 1 ) * (unsigned int)bitmap->pitch;

    target.pitch = bitmap->pitch;

    FT_Outline_Translate( outline,
                          -sub[0].x,
                          -sub[0].y );
    error = render->raster_render( render->raster, &params );
    x = sub[0].x;
    y = sub[0].y;
    if ( error )
      goto Exit;

    target.origin++;
    FT_Outline_Translate( outline,
                          sub[0].x - sub[1].x,
                          sub[0].y - sub[1].y );
    error = render->raster_render( render->raster, &params );
    x = sub[1].x;
    y = sub[1].y;
    if ( error )
      goto Exit;

    target.origin++;
    FT_Outline_Translate( outline,
                          sub[1].x - sub[2].x,
                          sub[1].y - sub[2].y );
    error = render->raster_render( render->raster, &params );
    x = sub[2].x;
    y = sub[2].y;

  Exit:
    FT_Outline_Translate( outline, x, y );

    return error;
  }


  static FT_Error
  ft_smooth_raster_lcdv( FT_Renderer  render,
                         FT_Outline*  outline,
                         FT_Bitmap*   bitmap )
  {
    FT_Error     error = FT_Err_Ok;
    int          pitch = bitmap->pitch;
    FT_Vector*   sub   = render->root.library->lcd_geometry;
    FT_Pos       x, y;

    FT_Raster_Params  params;


    params.target = bitmap;
    params.source = outline;
    params.flags  = FT_RASTER_FLAG_AA;

    /* Render 3 separate coverage bitmaps, shifting the outline. */
    /* Notice that the subpixel geometry vectors are rotated.    */
    /* Triple the pitch to render on each third row.            */
    bitmap->pitch *= 3;
    bitmap->rows  /= 3;

    FT_Outline_Translate( outline,
                          -sub[0].y,
                          sub[0].x );
    error = render->raster_render( render->raster, &params );
    x = sub[0].y;
    y = -sub[0].x;
    if ( error )
      goto Exit;

    bitmap->buffer += pitch;
    FT_Outline_Translate( outline,
                          sub[0].y - sub[1].y,
                          sub[1].x - sub[0].x );
    error = render->raster_render( render->raster, &params );
    x = sub[1].y;
    y = -sub[1].x;
    bitmap->buffer -= pitch;
    if ( error )
      goto Exit;

    bitmap->buffer += 2 * pitch;
    FT_Outline_Translate( outline,
                          sub[1].y - sub[2].y,
                          sub[2].x - sub[1].x );
    error = render->raster_render( render->raster, &params );
    x = sub[2].y;
    y = -sub[2].x;
    bitmap->buffer -= 2 * pitch;

  Exit:
    FT_Outline_Translate( outline, x, y );

    bitmap->pitch /= 3;
    bitmap->rows  *= 3;

    return error;
  }

#else   /* FT_CONFIG_OPTION_SUBPIXEL_RENDERING */

  /* initialize renderer -- init its raster */
  static FT_Error
  ft_smooth_init( FT_Module  module )   /* FT_Renderer */
  {
    FT_Renderer  render = (FT_Renderer)module;


    /* set up default LCD filtering */
    FT_Library_SetLcdFilter( render->root.library, FT_LCD_FILTER_DEFAULT );

    render->clazz->raster_class->raster_reset( render->raster, NULL, 0 );

    return 0;
  }


  static FT_Error
  ft_smooth_raster_lcd( FT_Renderer  render,
                        FT_Outline*  outline,
                        FT_Bitmap*   bitmap )
  {
    FT_Error    error      = FT_Err_Ok;
    FT_Vector*  points     = outline->points;
    FT_Vector*  points_end = FT_OFFSET( points, outline->n_points );
    FT_Vector*  vec;

    FT_Raster_Params  params;


    params.target = bitmap;
    params.source = outline;
    params.flags  = FT_RASTER_FLAG_AA;

    /* implode outline */
    for ( vec = points; vec < points_end; vec++ )
      vec->x *= 3;

    /* render outline into the bitmap */
    error = render->raster_render( render->raster, &params );

    /* deflate outline */
    for ( vec = points; vec < points_end; vec++ )
      vec->x /= 3;

    return error;
  }


  static FT_Error
  ft_smooth_raster_lcdv( FT_Renderer  render,
                         FT_Outline*  outline,
                         FT_Bitmap*   bitmap )
  {
    FT_Error    error      = FT_Err_Ok;
    FT_Vector*  points     = outline->points;
    FT_Vector*  points_end = FT_OFFSET( points, outline->n_points );
    FT_Vector*  vec;

    FT_Raster_Params  params;


    params.target = bitmap;
    params.source = outline;
    params.flags  = FT_RASTER_FLAG_AA;

    /* implode outline */
    for ( vec = points; vec < points_end; vec++ )
      vec->y *= 3;

    /* render outline into the bitmap */
    error = render->raster_render( render->raster, &params );

    /* deflate outline */
    for ( vec = points; vec < points_end; vec++ )
      vec->y /= 3;

    return error;
  }

#endif  /* FT_CONFIG_OPTION_SUBPIXEL_RENDERING */

/* Oversampling scale to be used in rendering overlaps */
#define SCALE  ( 1 << 2 )

  /* This function averages inflated spans in direct rendering mode */
  static void
  ft_smooth_overlap_spans( int             y,
                           int             count,
                           const FT_Span*  spans,
                           void*           target_ )
  {
    TOrigin*  target = (TOrigin*)target_;


    unsigned char*  dst = target->origin - ( y / SCALE ) * target->pitch;
    unsigned short  x;
    unsigned int    cover, sum;


    /* When accumulating the oversampled spans we need to assure that  */
    /* fully covered pixels are equal to 255 and do not overflow.      */
    /* It is important that the SCALE is a power of 2, each subpixel   */
    /* cover can also reach a power of 2 after rounding, and the total */
    /* is clamped to 255 when it adds up to 256.                       */
    for ( ; count--; spans++ )
    {
      cover = ( spans->coverage + SCALE * SCALE / 2 ) / ( SCALE * SCALE );
      for ( x = 0; x < spans->len; x++ )
      {
        sum                           = dst[( spans->x + x ) / SCALE] + cover;
        dst[( spans->x + x ) / SCALE] = (unsigned char)( sum - ( sum >> 8 ) );
      }
    }
  }


  static FT_Error
  ft_smooth_raster_overlap( FT_Renderer  render,
                            FT_Outline*  outline,
                            FT_Bitmap*   bitmap )
  {
    FT_Error    error      = FT_Err_Ok;
    FT_Vector*  points     = outline->points;
    FT_Vector*  points_end = FT_OFFSET( points, outline->n_points );
    FT_Vector*  vec;

    FT_Raster_Params   params;
    TOrigin            target;


    /* Reject outlines that are too wide for 16-bit FT_Span.       */
    /* Other limits are applied upstream with the same error code. */
    if ( bitmap->width * SCALE > 0x7FFF )
      return FT_THROW( Raster_Overflow );

    /* Set up direct rendering to average oversampled spans. */
    params.source     = outline;
    params.flags      = FT_RASTER_FLAG_AA | FT_RASTER_FLAG_DIRECT;
    params.gray_spans = ft_smooth_overlap_spans;
    params.user       = &target;

    params.clip_box.xMin = 0;
    params.clip_box.yMin = 0;
    params.clip_box.xMax = bitmap->width * SCALE;
    params.clip_box.yMax = bitmap->rows  * SCALE;

    if ( bitmap->pitch < 0 )
      target.origin = bitmap->buffer;
    else
      target.origin = bitmap->buffer
                      + ( bitmap->rows - 1 ) * (unsigned int)bitmap->pitch;

    target.pitch = bitmap->pitch;

    /* inflate outline */
    for ( vec = points; vec < points_end; vec++ )
    {
      vec->x *= SCALE;
      vec->y *= SCALE;
    }

    /* render outline into the bitmap */
    error = render->raster_render( render->raster, &params );

    /* deflate outline */
    for ( vec = points; vec < points_end; vec++ )
    {
      vec->x /= SCALE;
      vec->y /= SCALE;
    }

    return error;
  }

#undef SCALE

  static FT_Error
  ft_smooth_render( FT_Renderer       render,
                    FT_GlyphSlot      slot,
                    FT_Render_Mode    mode,
                    const FT_Vector*  origin )
  {
    FT_Error     error   = FT_Err_Ok;
    FT_Outline*  outline = &slot->outline;
    FT_Bitmap*   bitmap  = &slot->bitmap;
    FT_Memory    memory  = render->root.memory;
    FT_Pos       x_shift = 0;
    FT_Pos       y_shift = 0;


    /* check glyph image format */
    if ( slot->format != render->glyph_format )
    {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    /* check mode */
    if ( mode != FT_RENDER_MODE_NORMAL &&
         mode != FT_RENDER_MODE_LIGHT  &&
         mode != FT_RENDER_MODE_LCD    &&
         mode != FT_RENDER_MODE_LCD_V  )
    {
      error = FT_THROW( Cannot_Render_Glyph );
      goto Exit;
    }

    /* release old bitmap buffer */
    if ( slot->internal->flags & FT_GLYPH_OWN_BITMAP )
    {
      FT_FREE( bitmap->buffer );
      slot->internal->flags &= ~FT_GLYPH_OWN_BITMAP;
    }

    if ( ft_glyphslot_preset_bitmap( slot, mode, origin ) )
    {
      error = FT_THROW( Raster_Overflow );
      goto Exit;
    }

    if ( !bitmap->rows || !bitmap->pitch )
      goto Exit;

    /* allocate new one */
    if ( FT_ALLOC_MULT( bitmap->buffer, bitmap->rows, bitmap->pitch ) )
      goto Exit;

    slot->internal->flags |= FT_GLYPH_OWN_BITMAP;

    x_shift = 64 * -slot->bitmap_left;
    y_shift = 64 * -slot->bitmap_top;
    if ( bitmap->pixel_mode == FT_PIXEL_MODE_LCD_V )
      y_shift += 64 * (FT_Int)bitmap->rows / 3;
    else
      y_shift += 64 * (FT_Int)bitmap->rows;

    if ( origin )
    {
      x_shift += origin->x;
      y_shift += origin->y;
    }

    /* translate outline to render it into the bitmap */
    if ( x_shift || y_shift )
      FT_Outline_Translate( outline, x_shift, y_shift );

    if ( mode == FT_RENDER_MODE_NORMAL ||
         mode == FT_RENDER_MODE_LIGHT  )
    {
      if ( outline->flags & FT_OUTLINE_OVERLAP )
        error = ft_smooth_raster_overlap( render, outline, bitmap );
      else
      {
        FT_Raster_Params  params;


        params.target = bitmap;
        params.source = outline;
        params.flags  = FT_RASTER_FLAG_AA;

        error = render->raster_render( render->raster, &params );
      }
    }
    else
    {
      if ( mode == FT_RENDER_MODE_LCD )
        error = ft_smooth_raster_lcd ( render, outline, bitmap );
      else if ( mode == FT_RENDER_MODE_LCD_V )
        error = ft_smooth_raster_lcdv( render, outline, bitmap );

#ifdef FT_CONFIG_OPTION_SUBPIXEL_RENDERING

      /* finally apply filtering */
      {
        FT_Byte*                 lcd_weights;
        FT_Bitmap_LcdFilterFunc  lcd_filter_func;


        /* Per-face LCD filtering takes priority if set up. */
        if ( slot->face && slot->face->internal->lcd_filter_func )
        {
          lcd_weights     = slot->face->internal->lcd_weights;
          lcd_filter_func = slot->face->internal->lcd_filter_func;
        }
        else
        {
          lcd_weights     = slot->library->lcd_weights;
          lcd_filter_func = slot->library->lcd_filter_func;
        }

        if ( lcd_filter_func )
          lcd_filter_func( bitmap, lcd_weights );
      }

#endif /* FT_CONFIG_OPTION_SUBPIXEL_RENDERING */

    }

  Exit:
    if ( !error )
    {
      /* everything is fine; the glyph is now officially a bitmap */
      slot->format = FT_GLYPH_FORMAT_BITMAP;
    }
    else if ( slot->internal->flags & FT_GLYPH_OWN_BITMAP )
    {
      FT_FREE( bitmap->buffer );
      slot->internal->flags &= ~FT_GLYPH_OWN_BITMAP;
    }

    if ( x_shift || y_shift )
      FT_Outline_Translate( outline, -x_shift, -y_shift );

    return error;
  }


  FT_DEFINE_RENDERER(
    ft_smooth_renderer_class,

      FT_MODULE_RENDERER,
      sizeof ( FT_RendererRec ),

      "smooth",
      0x10000L,
      0x20000L,

      NULL,    /* module specific interface */

      (FT_Module_Constructor)ft_smooth_init,  /* module_init   */
      (FT_Module_Destructor) NULL,            /* module_done   */
      (FT_Module_Requester)  NULL,            /* get_interface */

    FT_GLYPH_FORMAT_OUTLINE,

    (FT_Renderer_RenderFunc)   ft_smooth_render,     /* render_glyph    */
    (FT_Renderer_TransformFunc)ft_smooth_transform,  /* transform_glyph */
    (FT_Renderer_GetCBoxFunc)  ft_smooth_get_cbox,   /* get_glyph_cbox  */
    (FT_Renderer_SetModeFunc)  ft_smooth_set_mode,   /* set_mode        */

    (FT_Raster_Funcs*)&ft_grays_raster               /* raster_class    */
  )


/* END */
