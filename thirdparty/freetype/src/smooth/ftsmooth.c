/***************************************************************************/
/*                                                                         */
/*  ftsmooth.c                                                             */
/*                                                                         */
/*    Anti-aliasing renderer interface (body).                             */
/*                                                                         */
/*  Copyright 2000-2017 by                                                 */
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
    FT_Error     error;
    FT_Outline*  outline = &slot->outline;
    FT_Bitmap*   bitmap  = &slot->bitmap;
    FT_Memory    memory  = render->root.memory;
    FT_BBox      cbox;
    FT_Pos       x_shift = 0;
    FT_Pos       y_shift = 0;
    FT_Pos       x_left, y_top;
    FT_Pos       width, height, pitch;
#ifndef FT_CONFIG_OPTION_SUBPIXEL_RENDERING
    FT_Pos       height_org, width_org;
#endif
    FT_Int       hmul    = ( mode == FT_RENDER_MODE_LCD );
    FT_Int       vmul    = ( mode == FT_RENDER_MODE_LCD_V );

    FT_Raster_Params  params;

    FT_Bool  have_outline_shifted = FALSE;
    FT_Bool  have_buffer          = FALSE;

#ifdef FT_CONFIG_OPTION_SUBPIXEL_RENDERING

    FT_Int                   lcd_extra          = 0;
    FT_LcdFiveTapFilter      lcd_weights        = { 0 };
    FT_Bool                  have_custom_weight = FALSE;
    FT_Bitmap_LcdFilterFunc  lcd_filter_func    = NULL;


    if ( slot->face )
    {
      FT_Char  i;


      for ( i = 0; i < FT_LCD_FILTER_FIVE_TAPS; i++ )
        if ( slot->face->internal->lcd_weights[i] != 0 )
        {
          have_custom_weight = TRUE;
          break;
        }
    }

    /*
     * The LCD filter can be set library-wide and per-face.  Face overrides
     * library.  If the face filter weights are all zero (the default), it
     * means that the library default should be used.
     */
    if ( have_custom_weight )
    {
      /*
       * A per-font filter is set.  It always uses the default 5-tap
       * in-place FIR filter that needs 2 extra pixels.
       */
      ft_memcpy( lcd_weights,
                 slot->face->internal->lcd_weights,
                 FT_LCD_FILTER_FIVE_TAPS );
      lcd_filter_func = ft_lcd_filter_fir;
      lcd_extra       = 2;
    }
    else
    {
      /*
       * The face's lcd_weights is {0, 0, 0, 0, 0}, meaning `use library
       * default'.  If the library is set to use no LCD filtering
       * (lcd_filter_func == NULL), `lcd_filter_func' here is also set to
       * NULL and the tests further below pass over the filtering process.
       */
      ft_memcpy( lcd_weights,
                 slot->library->lcd_weights,
                 FT_LCD_FILTER_FIVE_TAPS );
      lcd_filter_func = slot->library->lcd_filter_func;
      lcd_extra       = slot->library->lcd_extra;
    }

#endif /*FT_CONFIG_OPTION_SUBPIXEL_RENDERING */

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

    if ( origin )
    {
      x_shift = origin->x;
      y_shift = origin->y;
    }

    /* compute the control box, and grid fit it */
    /* taking into account the origin shift     */
    FT_Outline_Get_CBox( outline, &cbox );

    cbox.xMin = FT_PIX_FLOOR( cbox.xMin + x_shift );
    cbox.yMin = FT_PIX_FLOOR( cbox.yMin + y_shift );
    cbox.xMax = FT_PIX_CEIL( cbox.xMax + x_shift );
    cbox.yMax = FT_PIX_CEIL( cbox.yMax + y_shift );

    x_shift -= cbox.xMin;
    y_shift -= cbox.yMin;

    x_left  = cbox.xMin >> 6;
    y_top   = cbox.yMax >> 6;

    width  = (FT_ULong)( cbox.xMax - cbox.xMin ) >> 6;
    height = (FT_ULong)( cbox.yMax - cbox.yMin ) >> 6;

#ifndef FT_CONFIG_OPTION_SUBPIXEL_RENDERING
    width_org  = width;
    height_org = height;
#endif

    pitch = width;
    if ( hmul )
    {
      width *= 3;
      pitch  = FT_PAD_CEIL( width, 4 );
    }

    if ( vmul )
      height *= 3;

#ifdef FT_CONFIG_OPTION_SUBPIXEL_RENDERING
    if ( lcd_filter_func )
    {
      if ( hmul )
      {
        x_shift += 64 * ( lcd_extra >> 1 );
        x_left  -= lcd_extra >> 1;
        width   += 3 * lcd_extra;
        pitch    = FT_PAD_CEIL( width, 4 );
      }

      if ( vmul )
      {
        y_shift += 64 * ( lcd_extra >> 1 );
        y_top   += lcd_extra >> 1;
        height  += 3 * lcd_extra;
      }
    }
#endif

    /*
     * XXX: on 16bit system, we return an error for huge bitmap
     * to prevent an overflow.
     */
    if ( x_left > FT_INT_MAX || y_top > FT_INT_MAX ||
         x_left < FT_INT_MIN || y_top < FT_INT_MIN )
    {
      error = FT_THROW( Invalid_Pixel_Size );
      goto Exit;
    }

    /* Required check is (pitch * height < FT_ULONG_MAX),        */
    /* but we care realistic cases only.  Always pitch <= width. */
    if ( width > 0x7FFF || height > 0x7FFF )
    {
      FT_ERROR(( "ft_smooth_render_generic: glyph too large: %u x %u\n",
                 width, height ));
      error = FT_THROW( Raster_Overflow );
      goto Exit;
    }

    /* release old bitmap buffer */
    if ( slot->internal->flags & FT_GLYPH_OWN_BITMAP )
    {
      FT_FREE( bitmap->buffer );
      slot->internal->flags &= ~FT_GLYPH_OWN_BITMAP;
    }

    /* allocate new one */
    if ( FT_ALLOC( bitmap->buffer, (FT_ULong)( pitch * height ) ) )
      goto Exit;
    else
      have_buffer = TRUE;

    slot->internal->flags |= FT_GLYPH_OWN_BITMAP;

    slot->format      = FT_GLYPH_FORMAT_BITMAP;
    slot->bitmap_left = (FT_Int)x_left;
    slot->bitmap_top  = (FT_Int)y_top;

    bitmap->pixel_mode = FT_PIXEL_MODE_GRAY;
    bitmap->num_grays  = 256;
    bitmap->width      = (unsigned int)width;
    bitmap->rows       = (unsigned int)height;
    bitmap->pitch      = pitch;

    /* translate outline to render it into the bitmap */
    if ( x_shift || y_shift )
    {
      FT_Outline_Translate( outline, x_shift, y_shift );
      have_outline_shifted = TRUE;
    }

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

    if ( lcd_filter_func )
      lcd_filter_func( bitmap, mode, lcd_weights );

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

    /* everything is fine; don't deallocate buffer */
    have_buffer = FALSE;

    error = FT_Err_Ok;

  Exit:
    if ( have_outline_shifted )
      FT_Outline_Translate( outline, -x_shift, -y_shift );
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

    (FT_Raster_Funcs*)&FT_GRAYS_RASTER_GET           /* raster_class    */
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

    (FT_Raster_Funcs*)&FT_GRAYS_RASTER_GET            /* raster_class    */
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

    (FT_Raster_Funcs*)&FT_GRAYS_RASTER_GET              /* raster_class    */
  )


/* END */
