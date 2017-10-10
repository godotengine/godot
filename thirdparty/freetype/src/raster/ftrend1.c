/***************************************************************************/
/*                                                                         */
/*  ftrend1.c                                                              */
/*                                                                         */
/*    The FreeType glyph rasterizer interface (body).                      */
/*                                                                         */
/*  Copyright 1996-2017 by                                                 */
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
#include "ftrend1.h"
#include "ftraster.h"
#include "rastpic.h"

#include "rasterrs.h"


  /* initialize renderer -- init its raster */
  static FT_Error
  ft_raster1_init( FT_Renderer  render )
  {
    render->clazz->raster_class->raster_reset( render->raster, NULL, 0 );

    return FT_Err_Ok;
  }


  /* set render-specific mode */
  static FT_Error
  ft_raster1_set_mode( FT_Renderer  render,
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
  ft_raster1_transform( FT_Renderer       render,
                        FT_GlyphSlot      slot,
                        const FT_Matrix*  matrix,
                        const FT_Vector*  delta )
  {
    FT_Error error = FT_Err_Ok;


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
  ft_raster1_get_cbox( FT_Renderer   render,
                       FT_GlyphSlot  slot,
                       FT_BBox*      cbox )
  {
    FT_ZERO( cbox );

    if ( slot->format == render->glyph_format )
      FT_Outline_Get_CBox( &slot->outline, cbox );
  }


  /* convert a slot's glyph image into a bitmap */
  static FT_Error
  ft_raster1_render( FT_Renderer       render,
                     FT_GlyphSlot      slot,
                     FT_Render_Mode    mode,
                     const FT_Vector*  origin )
  {
    FT_Error     error;
    FT_Outline*  outline;
    FT_BBox      cbox, cbox0;
    FT_UInt      width, height, pitch;
    FT_Bitmap*   bitmap;
    FT_Memory    memory;

    FT_Raster_Params  params;


    /* check glyph image format */
    if ( slot->format != render->glyph_format )
    {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    /* check rendering mode */
    if ( mode != FT_RENDER_MODE_MONO )
    {
      /* raster1 is only capable of producing monochrome bitmaps */
      return FT_THROW( Cannot_Render_Glyph );
    }

    outline = &slot->outline;

    /* translate the outline to the new origin if needed */
    if ( origin )
      FT_Outline_Translate( outline, origin->x, origin->y );

    /* compute the control box, and grid fit it */
    FT_Outline_Get_CBox( outline, &cbox0 );

    /* undocumented but confirmed: bbox values get rounded */
#if 1
    cbox.xMin = FT_PIX_ROUND( cbox0.xMin );
    cbox.yMin = FT_PIX_ROUND( cbox0.yMin );
    cbox.xMax = FT_PIX_ROUND( cbox0.xMax );
    cbox.yMax = FT_PIX_ROUND( cbox0.yMax );
#else
    cbox.xMin = FT_PIX_FLOOR( cbox.xMin );
    cbox.yMin = FT_PIX_FLOOR( cbox.yMin );
    cbox.xMax = FT_PIX_CEIL( cbox.xMax );
    cbox.yMax = FT_PIX_CEIL( cbox.yMax );
#endif

    /* If either `width' or `height' round to 0, try    */
    /* explicitly rounding up/down.  In the case of     */
    /* glyphs containing only one very narrow feature,  */
    /* this gives the drop-out compensation in the scan */
    /* conversion code a chance to do its stuff.        */
    width  = (FT_UInt)( ( cbox.xMax - cbox.xMin ) >> 6 );
    if ( width == 0 )
    {
      cbox.xMin = FT_PIX_FLOOR( cbox0.xMin );
      cbox.xMax = FT_PIX_CEIL( cbox0.xMax );

      width = (FT_UInt)( ( cbox.xMax - cbox.xMin ) >> 6 );
    }

    height = (FT_UInt)( ( cbox.yMax - cbox.yMin ) >> 6 );
    if ( height == 0 )
    {
      cbox.yMin = FT_PIX_FLOOR( cbox0.yMin );
      cbox.yMax = FT_PIX_CEIL( cbox0.yMax );

      height = (FT_UInt)( ( cbox.yMax - cbox.yMin ) >> 6 );
    }

    if ( width > FT_USHORT_MAX || height > FT_USHORT_MAX )
    {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    bitmap = &slot->bitmap;
    memory = render->root.memory;

    /* release old bitmap buffer */
    if ( slot->internal->flags & FT_GLYPH_OWN_BITMAP )
    {
      FT_FREE( bitmap->buffer );
      slot->internal->flags &= ~FT_GLYPH_OWN_BITMAP;
    }

    pitch              = ( ( width + 15 ) >> 4 ) << 1;
    bitmap->pixel_mode = FT_PIXEL_MODE_MONO;

    bitmap->width = width;
    bitmap->rows  = height;
    bitmap->pitch = (int)pitch;

    if ( FT_ALLOC_MULT( bitmap->buffer, height, pitch ) )
      goto Exit;

    slot->internal->flags |= FT_GLYPH_OWN_BITMAP;

    /* translate outline to render it into the bitmap */
    FT_Outline_Translate( outline, -cbox.xMin, -cbox.yMin );

    /* set up parameters */
    params.target = bitmap;
    params.source = outline;
    params.flags  = 0;

    /* render outline into the bitmap */
    error = render->raster_render( render->raster, &params );

    FT_Outline_Translate( outline, cbox.xMin, cbox.yMin );

    if ( error )
      goto Exit;

    slot->format      = FT_GLYPH_FORMAT_BITMAP;
    slot->bitmap_left = (FT_Int)( cbox.xMin >> 6 );
    slot->bitmap_top  = (FT_Int)( cbox.yMax >> 6 );

  Exit:
    return error;
  }


  FT_DEFINE_RENDERER(
    ft_raster1_renderer_class,

      FT_MODULE_RENDERER,
      sizeof ( FT_RendererRec ),

      "raster1",
      0x10000L,
      0x20000L,

      NULL,    /* module specific interface */

      (FT_Module_Constructor)ft_raster1_init,  /* module_init   */
      (FT_Module_Destructor) NULL,             /* module_done   */
      (FT_Module_Requester)  NULL,             /* get_interface */

    FT_GLYPH_FORMAT_OUTLINE,

    (FT_Renderer_RenderFunc)   ft_raster1_render,     /* render_glyph    */
    (FT_Renderer_TransformFunc)ft_raster1_transform,  /* transform_glyph */
    (FT_Renderer_GetCBoxFunc)  ft_raster1_get_cbox,   /* get_glyph_cbox  */
    (FT_Renderer_SetModeFunc)  ft_raster1_set_mode,   /* set_mode        */

    (FT_Raster_Funcs*)&FT_STANDARD_RASTER_GET         /* raster_class    */
  )


/* END */
