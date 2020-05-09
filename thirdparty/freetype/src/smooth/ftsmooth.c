/****************************************************************************
 *
 * ftsmooth.c
 *
 *   Anti-aliasing renderer interface (body).
 *
 * Copyright (C) 2000-2020 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#include <ft2build.h>
#include FT_INTERNAL_DEBUG_H
#include FT_INTERNAL_OBJECTS_H
#include FT_OUTLINE_H
#include "ftsmooth.h"
#include "ftgrays.h"

#include "ftsmerrs.h"


  /* initialize renderer -- init its raster */
  static FT_Error
  ft_smooth_init( FT_Renderer  render )
  {

#ifndef FT_CONFIG_OPTION_SUBPIXEL_RENDERING

    FT_Vector*  sub = render->root.library->lcd_geometry;


    /* set up default subpixel geometry for striped RGB panels. */
    sub[0].x = -21;
    sub[0].y = 0;
    sub[1].x = 0;
    sub[1].y = 0;
    sub[2].x = 21;
    sub[2].y = 0;

#elif 0   /* or else, once ClearType patents expire */

    FT_Library_SetLcdFilter( render->root.library, FT_LCD_FILTER_DEFAULT );

#endif

    render->clazz->raster_class->raster_reset( render->raster, NULL, 0 );

    return 0;
  }


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


  /* convert a slot's glyph image into a bitmap */
  static FT_Error
  ft_smooth_render_generic( FT_Renderer       render,
                            FT_GlyphSlot      slot,
                            FT_Render_Mode    mode,
                            const FT_Vector*  origin,
                            FT_Render_Mode    required_mode )
  {
    FT_Error     error   = FT_Err_Ok;
    FT_Outline*  outline = &slot->outline;
    FT_Bitmap*   bitmap  = &slot->bitmap;
    FT_Memory    memory  = render->root.memory;
    FT_Pos       x_shift = 0;
    FT_Pos       y_shift = 0;
    FT_Int       hmul    = ( mode == FT_RENDER_MODE_LCD );
    FT_Int       vmul    = ( mode == FT_RENDER_MODE_LCD_V );

    FT_Raster_Params  params;


    /* check glyph image format */
    if ( slot->format != render->glyph_format )
    {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    /* check mode */
    if ( mode != required_mode )
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

    /* set up parameters */
    params.target = bitmap;
    params.source = outline;
    params.flags  = FT_RASTER_FLAG_AA;

#ifdef FT_CONFIG_OPTION_SUBPIXEL_RENDERING

    /* implode outline if needed */
    {
      FT_Vector*  points     = outline->points;
      FT_Vector*  points_end = FT_OFFSET( points, outline->n_points );
      FT_Vector*  vec;


      if ( hmul )
        for ( vec = points; vec < points_end; vec++ )
          vec->x *= 3;

      if ( vmul )
        for ( vec = points; vec < points_end; vec++ )
          vec->y *= 3;
    }

    /* render outline into the bitmap */
    error = render->raster_render( render->raster, &params );

    /* deflate outline if needed */
    {
      FT_Vector*  points     = outline->points;
      FT_Vector*  points_end = FT_OFFSET( points, outline->n_points );
      FT_Vector*  vec;


      if ( hmul )
        for ( vec = points; vec < points_end; vec++ )
          vec->x /= 3;

      if ( vmul )
        for ( vec = points; vec < points_end; vec++ )
          vec->y /= 3;
    }

    if ( error )
      goto Exit;

    /* finally apply filtering */
    if ( hmul || vmul )
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

#else /* !FT_CONFIG_OPTION_SUBPIXEL_RENDERING */

    if ( hmul )  /* lcd */
    {
      FT_Byte*  line;
      FT_Byte*  temp = NULL;
      FT_UInt   i, j;

      unsigned int  height = bitmap->rows;
      unsigned int  width  = bitmap->width;
      int           pitch  = bitmap->pitch;

      FT_Vector*  sub = slot->library->lcd_geometry;


      /* Render 3 separate monochrome bitmaps, shifting the outline.  */
      width /= 3;

      FT_Outline_Translate( outline,
                            -sub[0].x,
                            -sub[0].y );
      error = render->raster_render( render->raster, &params );
      if ( error )
        goto Exit;

      bitmap->buffer += width;
      FT_Outline_Translate( outline,
                            sub[0].x - sub[1].x,
                            sub[0].y - sub[1].y );
      error = render->raster_render( render->raster, &params );
      bitmap->buffer -= width;
      if ( error )
        goto Exit;

      bitmap->buffer += 2 * width;
      FT_Outline_Translate( outline,
                            sub[1].x - sub[2].x,
                            sub[1].y - sub[2].y );
      error = render->raster_render( render->raster, &params );
      bitmap->buffer -= 2 * width;
      if ( error )
        goto Exit;

      x_shift -= sub[2].x;
      y_shift -= sub[2].y;

      /* XXX: Rearrange the bytes according to FT_PIXEL_MODE_LCD.    */
      /* XXX: It is more efficient to render every third byte above. */

      if ( FT_ALLOC( temp, (FT_ULong)pitch ) )
        goto Exit;

      for ( i = 0; i < height; i++ )
      {
        line = bitmap->buffer + i * (FT_ULong)pitch;
        for ( j = 0; j < width; j++ )
        {
          temp[3 * j    ] = line[j];
          temp[3 * j + 1] = line[j + width];
          temp[3 * j + 2] = line[j + width + width];
        }
        FT_MEM_COPY( line, temp, pitch );
      }

      FT_FREE( temp );
    }
    else if ( vmul )  /* lcd_v */
    {
      int  pitch  = bitmap->pitch;

      FT_Vector*  sub = slot->library->lcd_geometry;


      /* Render 3 separate monochrome bitmaps, shifting the outline. */
      /* Notice that the subpixel geometry vectors are rotated.      */
      /* Triple the pitch to render on each third row.               */
      bitmap->pitch *= 3;
      bitmap->rows  /= 3;

      FT_Outline_Translate( outline,
                            -sub[0].y,
                            sub[0].x );
      error = render->raster_render( render->raster, &params );
      if ( error )
        goto Exit;

      bitmap->buffer += pitch;
      FT_Outline_Translate( outline,
                            sub[0].y - sub[1].y,
                            sub[1].x - sub[0].x );
      error = render->raster_render( render->raster, &params );
      bitmap->buffer -= pitch;
      if ( error )
        goto Exit;

      bitmap->buffer += 2 * pitch;
      FT_Outline_Translate( outline,
                            sub[1].y - sub[2].y,
                            sub[2].x - sub[1].x );
      error = render->raster_render( render->raster, &params );
      bitmap->buffer -= 2 * pitch;
      if ( error )
        goto Exit;

      x_shift -= sub[2].y;
      y_shift += sub[2].x;

      bitmap->pitch /= 3;
      bitmap->rows  *= 3;
    }
    else  /* grayscale */
      error = render->raster_render( render->raster, &params );

#endif /* !FT_CONFIG_OPTION_SUBPIXEL_RENDERING */

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


  /* convert a slot's glyph image into a bitmap */
  static FT_Error
  ft_smooth_render( FT_Renderer       render,
                    FT_GlyphSlot      slot,
                    FT_Render_Mode    mode,
                    const FT_Vector*  origin )
  {
    if ( mode == FT_RENDER_MODE_LIGHT )
      mode = FT_RENDER_MODE_NORMAL;

    return ft_smooth_render_generic( render, slot, mode, origin,
                                     FT_RENDER_MODE_NORMAL );
  }


  /* convert a slot's glyph image into a horizontal LCD bitmap */
  static FT_Error
  ft_smooth_render_lcd( FT_Renderer       render,
                        FT_GlyphSlot      slot,
                        FT_Render_Mode    mode,
                        const FT_Vector*  origin )
  {
    return ft_smooth_render_generic( render, slot, mode, origin,
                                     FT_RENDER_MODE_LCD );
  }


  /* convert a slot's glyph image into a vertical LCD bitmap */
  static FT_Error
  ft_smooth_render_lcd_v( FT_Renderer       render,
                          FT_GlyphSlot      slot,
                          FT_Render_Mode    mode,
                          const FT_Vector*  origin )
  {
    return ft_smooth_render_generic( render, slot, mode, origin,
                                     FT_RENDER_MODE_LCD_V );
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


  FT_DEFINE_RENDERER(
    ft_smooth_lcd_renderer_class,

      FT_MODULE_RENDERER,
      sizeof ( FT_RendererRec ),

      "smooth-lcd",
      0x10000L,
      0x20000L,

      NULL,    /* module specific interface */

      (FT_Module_Constructor)ft_smooth_init,  /* module_init   */
      (FT_Module_Destructor) NULL,            /* module_done   */
      (FT_Module_Requester)  NULL,            /* get_interface */

    FT_GLYPH_FORMAT_OUTLINE,

    (FT_Renderer_RenderFunc)   ft_smooth_render_lcd,  /* render_glyph    */
    (FT_Renderer_TransformFunc)ft_smooth_transform,   /* transform_glyph */
    (FT_Renderer_GetCBoxFunc)  ft_smooth_get_cbox,    /* get_glyph_cbox  */
    (FT_Renderer_SetModeFunc)  ft_smooth_set_mode,    /* set_mode        */

    (FT_Raster_Funcs*)&ft_grays_raster                /* raster_class    */
  )


  FT_DEFINE_RENDERER(
    ft_smooth_lcdv_renderer_class,

      FT_MODULE_RENDERER,
      sizeof ( FT_RendererRec ),

      "smooth-lcdv",
      0x10000L,
      0x20000L,

      NULL,    /* module specific interface */

      (FT_Module_Constructor)ft_smooth_init,  /* module_init   */
      (FT_Module_Destructor) NULL,            /* module_done   */
      (FT_Module_Requester)  NULL,            /* get_interface */

    FT_GLYPH_FORMAT_OUTLINE,

    (FT_Renderer_RenderFunc)   ft_smooth_render_lcd_v,  /* render_glyph    */
    (FT_Renderer_TransformFunc)ft_smooth_transform,     /* transform_glyph */
    (FT_Renderer_GetCBoxFunc)  ft_smooth_get_cbox,      /* get_glyph_cbox  */
    (FT_Renderer_SetModeFunc)  ft_smooth_set_mode,      /* set_mode        */

    (FT_Raster_Funcs*)&ft_grays_raster                  /* raster_class    */
  )


/* END */
