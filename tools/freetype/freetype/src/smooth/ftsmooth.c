/***************************************************************************/
/*                                                                         */
/*  ftsmooth.c                                                             */
/*                                                                         */
/*    Anti-aliasing renderer interface (body).                             */
/*                                                                         */
/*  Copyright 2000-2006, 2009-2013 by                                      */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


#include <ft2build.h>
#include FT_INTERNAL_DEBUG_H
#include FT_INTERNAL_OBJECTS_H
#include FT_OUTLINE_H
#include "ftsmooth.h"
#include "ftgrays.h"
#include "ftspic.h"

#include "ftsmerrs.h"


  /* initialize renderer -- init its raster */
  static FT_Error
  ft_smooth_init( FT_Renderer  render )
  {
    FT_Library  library = FT_MODULE_LIBRARY( render );


    render->clazz->raster_class->raster_reset( render->raster,
                                               library->raster_pool,
                                               library->raster_pool_size );

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
    FT_MEM_ZERO( cbox, sizeof ( *cbox ) );

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
    FT_Error     error;
    FT_Outline*  outline = NULL;
    FT_BBox      cbox;
    FT_Pos       width, height, pitch;
#ifndef FT_CONFIG_OPTION_SUBPIXEL_RENDERING
    FT_Pos       height_org, width_org;
#endif
    FT_Bitmap*   bitmap  = &slot->bitmap;
    FT_Memory    memory  = render->root.memory;
    FT_Int       hmul    = mode == FT_RENDER_MODE_LCD;
    FT_Int       vmul    = mode == FT_RENDER_MODE_LCD_V;
    FT_Pos       x_shift = 0;
    FT_Pos       y_shift = 0;
    FT_Pos       x_left, y_top;

    FT_Raster_Params  params;

    FT_Bool  have_translated_origin = FALSE;
    FT_Bool  have_outline_shifted   = FALSE;
    FT_Bool  have_buffer            = FALSE;


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

    outline = &slot->outline;

    /* translate the outline to the new origin if needed */
    if ( origin )
    {
      FT_Outline_Translate( outline, origin->x, origin->y );
      have_translated_origin = TRUE;
    }

    /* compute the control box, and grid fit it */
    FT_Outline_Get_CBox( outline, &cbox );

    cbox.xMin = FT_PIX_FLOOR( cbox.xMin );
    cbox.yMin = FT_PIX_FLOOR( cbox.yMin );
    cbox.xMax = FT_PIX_CEIL( cbox.xMax );
    cbox.yMax = FT_PIX_CEIL( cbox.yMax );

    if ( cbox.xMin < 0 && cbox.xMax > FT_INT_MAX + cbox.xMin )
    {
      FT_ERROR(( "ft_smooth_render_generic: glyph too large:"
                 " xMin = %d, xMax = %d\n",
                 cbox.xMin >> 6, cbox.xMax >> 6 ));
      error = FT_THROW( Raster_Overflow );
      goto Exit;
    }
    else
      width = ( cbox.xMax - cbox.xMin ) >> 6;

    if ( cbox.yMin < 0 && cbox.yMax > FT_INT_MAX + cbox.yMin )
    {
      FT_ERROR(( "ft_smooth_render_generic: glyph too large:"
                 " yMin = %d, yMax = %d\n",
                 cbox.yMin >> 6, cbox.yMax >> 6 ));
      error = FT_THROW( Raster_Overflow );
      goto Exit;
    }
    else
      height = ( cbox.yMax - cbox.yMin ) >> 6;

#ifndef FT_CONFIG_OPTION_SUBPIXEL_RENDERING
    width_org  = width;
    height_org = height;
#endif

    /* release old bitmap buffer */
    if ( slot->internal->flags & FT_GLYPH_OWN_BITMAP )
    {
      FT_FREE( bitmap->buffer );
      slot->internal->flags &= ~FT_GLYPH_OWN_BITMAP;
    }

    /* allocate new one */
    pitch = width;
    if ( hmul )
    {
      width = width * 3;
      pitch = FT_PAD_CEIL( width, 4 );
    }

    if ( vmul )
      height *= 3;

    x_shift = (FT_Int) cbox.xMin;
    y_shift = (FT_Int) cbox.yMin;
    x_left  = (FT_Int)( cbox.xMin >> 6 );
    y_top   = (FT_Int)( cbox.yMax >> 6 );

#ifdef FT_CONFIG_OPTION_SUBPIXEL_RENDERING

    if ( slot->library->lcd_filter_func )
    {
      FT_Int  extra = slot->library->lcd_extra;


      if ( hmul )
      {
        x_shift -= 64 * ( extra >> 1 );
        width   += 3 * extra;
        pitch    = FT_PAD_CEIL( width, 4 );
        x_left  -= extra >> 1;
      }

      if ( vmul )
      {
        y_shift -= 64 * ( extra >> 1 );
        height  += 3 * extra;
        y_top   += extra >> 1;
      }
    }

#endif

#if FT_UINT_MAX > 0xFFFFU

    /* Required check is (pitch * height < FT_ULONG_MAX),        */
    /* but we care realistic cases only.  Always pitch <= width. */
    if ( width > 0x7FFF || height > 0x7FFF )
    {
      FT_ERROR(( "ft_smooth_render_generic: glyph too large: %u x %u\n",
                 width, height ));
      error = FT_THROW( Raster_Overflow );
      goto Exit;
    }

#endif

    bitmap->pixel_mode = FT_PIXEL_MODE_GRAY;
    bitmap->num_grays  = 256;
    bitmap->width      = width;
    bitmap->rows       = height;
    bitmap->pitch      = pitch;

    /* translate outline to render it into the bitmap */
    FT_Outline_Translate( outline, -x_shift, -y_shift );
    have_outline_shifted = TRUE;

    if ( FT_ALLOC( bitmap->buffer, (FT_ULong)pitch * height ) )
      goto Exit;
    else
      have_buffer = TRUE;

    slot->internal->flags |= FT_GLYPH_OWN_BITMAP;

    /* set up parameters */
    params.target = bitmap;
    params.source = outline;
    params.flags  = FT_RASTER_FLAG_AA;

#ifdef FT_CONFIG_OPTION_SUBPIXEL_RENDERING

    /* implode outline if needed */
    {
      FT_Vector*  points     = outline->points;
      FT_Vector*  points_end = points + outline->n_points;
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
      FT_Vector*  points_end = points + outline->n_points;
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

    if ( slot->library->lcd_filter_func )
      slot->library->lcd_filter_func( bitmap, mode, slot->library );

#else /* !FT_CONFIG_OPTION_SUBPIXEL_RENDERING */

    /* render outline into bitmap */
    error = render->raster_render( render->raster, &params );
    if ( error )
      goto Exit;

    /* expand it horizontally */
    if ( hmul )
    {
      FT_Byte*  line = bitmap->buffer;
      FT_UInt   hh;


      for ( hh = height_org; hh > 0; hh--, line += pitch )
      {
        FT_UInt   xx;
        FT_Byte*  end = line + width;


        for ( xx = width_org; xx > 0; xx-- )
        {
          FT_UInt  pixel = line[xx-1];


          end[-3] = (FT_Byte)pixel;
          end[-2] = (FT_Byte)pixel;
          end[-1] = (FT_Byte)pixel;
          end    -= 3;
        }
      }
    }

    /* expand it vertically */
    if ( vmul )
    {
      FT_Byte*  read  = bitmap->buffer + ( height - height_org ) * pitch;
      FT_Byte*  write = bitmap->buffer;
      FT_UInt   hh;


      for ( hh = height_org; hh > 0; hh-- )
      {
        ft_memcpy( write, read, pitch );
        write += pitch;

        ft_memcpy( write, read, pitch );
        write += pitch;

        ft_memcpy( write, read, pitch );
        write += pitch;
        read  += pitch;
      }
    }

#endif /* !FT_CONFIG_OPTION_SUBPIXEL_RENDERING */

    /*
     * XXX: on 16bit system, we return an error for huge bitmap
     * to prevent an overflow.
     */
    if ( x_left > FT_INT_MAX || y_top > FT_INT_MAX )
    {
      error = FT_THROW( Invalid_Pixel_Size );
      goto Exit;
    }

    slot->format      = FT_GLYPH_FORMAT_BITMAP;
    slot->bitmap_left = (FT_Int)x_left;
    slot->bitmap_top  = (FT_Int)y_top;

    /* everything is fine; don't deallocate buffer */
    have_buffer = FALSE;

    error = FT_Err_Ok;

  Exit:
    if ( have_outline_shifted )
      FT_Outline_Translate( outline, x_shift, y_shift );
    if ( have_translated_origin )
      FT_Outline_Translate( outline, -origin->x, -origin->y );
    if ( have_buffer )
    {
      FT_FREE( bitmap->buffer );
      slot->internal->flags &= ~FT_GLYPH_OWN_BITMAP;
    }

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
    FT_Error  error;

    error = ft_smooth_render_generic( render, slot, mode, origin,
                                      FT_RENDER_MODE_LCD );
    if ( !error )
      slot->bitmap.pixel_mode = FT_PIXEL_MODE_LCD;

    return error;
  }


  /* convert a slot's glyph image into a vertical LCD bitmap */
  static FT_Error
  ft_smooth_render_lcd_v( FT_Renderer       render,
                          FT_GlyphSlot      slot,
                          FT_Render_Mode    mode,
                          const FT_Vector*  origin )
  {
    FT_Error  error;

    error = ft_smooth_render_generic( render, slot, mode, origin,
                                      FT_RENDER_MODE_LCD_V );
    if ( !error )
      slot->bitmap.pixel_mode = FT_PIXEL_MODE_LCD_V;

    return error;
  }


  FT_DEFINE_RENDERER( ft_smooth_renderer_class,

      FT_MODULE_RENDERER,
      sizeof ( FT_RendererRec ),

      "smooth",
      0x10000L,
      0x20000L,

      0,    /* module specific interface */

      (FT_Module_Constructor)ft_smooth_init,
      (FT_Module_Destructor) 0,
      (FT_Module_Requester)  0
    ,

    FT_GLYPH_FORMAT_OUTLINE,

    (FT_Renderer_RenderFunc)   ft_smooth_render,
    (FT_Renderer_TransformFunc)ft_smooth_transform,
    (FT_Renderer_GetCBoxFunc)  ft_smooth_get_cbox,
    (FT_Renderer_SetModeFunc)  ft_smooth_set_mode,

    (FT_Raster_Funcs*)    &FT_GRAYS_RASTER_GET
  )


  FT_DEFINE_RENDERER( ft_smooth_lcd_renderer_class,

      FT_MODULE_RENDERER,
      sizeof ( FT_RendererRec ),

      "smooth-lcd",
      0x10000L,
      0x20000L,

      0,    /* module specific interface */

      (FT_Module_Constructor)ft_smooth_init,
      (FT_Module_Destructor) 0,
      (FT_Module_Requester)  0
    ,

    FT_GLYPH_FORMAT_OUTLINE,

    (FT_Renderer_RenderFunc)   ft_smooth_render_lcd,
    (FT_Renderer_TransformFunc)ft_smooth_transform,
    (FT_Renderer_GetCBoxFunc)  ft_smooth_get_cbox,
    (FT_Renderer_SetModeFunc)  ft_smooth_set_mode,

    (FT_Raster_Funcs*)    &FT_GRAYS_RASTER_GET
  )

  FT_DEFINE_RENDERER( ft_smooth_lcdv_renderer_class,

      FT_MODULE_RENDERER,
      sizeof ( FT_RendererRec ),

      "smooth-lcdv",
      0x10000L,
      0x20000L,

      0,    /* module specific interface */

      (FT_Module_Constructor)ft_smooth_init,
      (FT_Module_Destructor) 0,
      (FT_Module_Requester)  0
    ,

    FT_GLYPH_FORMAT_OUTLINE,

    (FT_Renderer_RenderFunc)   ft_smooth_render_lcd_v,
    (FT_Renderer_TransformFunc)ft_smooth_transform,
    (FT_Renderer_GetCBoxFunc)  ft_smooth_get_cbox,
    (FT_Renderer_SetModeFunc)  ft_smooth_set_mode,

    (FT_Raster_Funcs*)    &FT_GRAYS_RASTER_GET
  )


/* END */
