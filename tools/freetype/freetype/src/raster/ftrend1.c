/***************************************************************************/
/*                                                                         */
/*  ftrend1.c                                                              */
/*                                                                         */
/*    The FreeType glyph rasterizer interface (body).                      */
/*                                                                         */
/*  Copyright 1996-2003, 2005, 2006, 2011, 2013 by                         */
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
    FT_Library  library = FT_MODULE_LIBRARY( render );


    render->clazz->raster_class->raster_reset( render->raster,
                                               library->raster_pool,
                                               library->raster_pool_size );

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
    FT_MEM_ZERO( cbox, sizeof ( *cbox ) );

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
    FT_BBox      cbox;
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
#ifndef FT_CONFIG_OPTION_PIC
    if ( mode != FT_RENDER_MODE_MONO )
    {
      /* raster1 is only capable of producing monochrome bitmaps */
      if ( render->clazz == &ft_raster1_renderer_class )
        return FT_THROW( Cannot_Render_Glyph );
    }
    else
    {
      /* raster5 is only capable of producing 5-gray-levels bitmaps */
      if ( render->clazz == &ft_raster5_renderer_class )
        return FT_THROW( Cannot_Render_Glyph );
    }
#else /* FT_CONFIG_OPTION_PIC */
    /* When PIC is enabled, we cannot get to the class object      */
    /* so instead we check the final character in the class name   */
    /* ("raster5" or "raster1"). Yes this is a hack.               */
    /* The "correct" thing to do is have different render function */
    /* for each of the classes.                                    */
    if ( mode != FT_RENDER_MODE_MONO )
    {
      /* raster1 is only capable of producing monochrome bitmaps */
      if ( render->clazz->root.module_name[6] == '1' )
        return FT_THROW( Cannot_Render_Glyph );
    }
    else
    {
      /* raster5 is only capable of producing 5-gray-levels bitmaps */
      if ( render->clazz->root.module_name[6] == '5' )
        return FT_THROW( Cannot_Render_Glyph );
    }
#endif /* FT_CONFIG_OPTION_PIC */

    outline = &slot->outline;

    /* translate the outline to the new origin if needed */
    if ( origin )
      FT_Outline_Translate( outline, origin->x, origin->y );

    /* compute the control box, and grid fit it */
    FT_Outline_Get_CBox( outline, &cbox );

    /* undocumented but confirmed: bbox values get rounded */
#if 1
    cbox.xMin = FT_PIX_ROUND( cbox.xMin );
    cbox.yMin = FT_PIX_ROUND( cbox.yMin );
    cbox.xMax = FT_PIX_ROUND( cbox.xMax );
    cbox.yMax = FT_PIX_ROUND( cbox.yMax );
#else
    cbox.xMin = FT_PIX_FLOOR( cbox.xMin );
    cbox.yMin = FT_PIX_FLOOR( cbox.yMin );
    cbox.xMax = FT_PIX_CEIL( cbox.xMax );
    cbox.yMax = FT_PIX_CEIL( cbox.yMax );
#endif

    width  = (FT_UInt)( ( cbox.xMax - cbox.xMin ) >> 6 );
    height = (FT_UInt)( ( cbox.yMax - cbox.yMin ) >> 6 );

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

    /* allocate new one, depends on pixel format */
    if ( !( mode & FT_RENDER_MODE_MONO ) )
    {
      /* we pad to 32 bits, only for backwards compatibility with FT 1.x */
      pitch              = FT_PAD_CEIL( width, 4 );
      bitmap->pixel_mode = FT_PIXEL_MODE_GRAY;
      bitmap->num_grays  = 256;
    }
    else
    {
      pitch              = ( ( width + 15 ) >> 4 ) << 1;
      bitmap->pixel_mode = FT_PIXEL_MODE_MONO;
    }

    bitmap->width = width;
    bitmap->rows  = height;
    bitmap->pitch = pitch;

    if ( FT_ALLOC_MULT( bitmap->buffer, pitch, height ) )
      goto Exit;

    slot->internal->flags |= FT_GLYPH_OWN_BITMAP;

    /* translate outline to render it into the bitmap */
    FT_Outline_Translate( outline, -cbox.xMin, -cbox.yMin );

    /* set up parameters */
    params.target = bitmap;
    params.source = outline;
    params.flags  = 0;

    if ( bitmap->pixel_mode == FT_PIXEL_MODE_GRAY )
      params.flags |= FT_RASTER_FLAG_AA;

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


  FT_DEFINE_RENDERER( ft_raster1_renderer_class,

      FT_MODULE_RENDERER,
      sizeof ( FT_RendererRec ),

      "raster1",
      0x10000L,
      0x20000L,

      0,    /* module specific interface */

      (FT_Module_Constructor)ft_raster1_init,
      (FT_Module_Destructor) 0,
      (FT_Module_Requester)  0
    ,

    FT_GLYPH_FORMAT_OUTLINE,

    (FT_Renderer_RenderFunc)   ft_raster1_render,
    (FT_Renderer_TransformFunc)ft_raster1_transform,
    (FT_Renderer_GetCBoxFunc)  ft_raster1_get_cbox,
    (FT_Renderer_SetModeFunc)  ft_raster1_set_mode,

    (FT_Raster_Funcs*)    &FT_STANDARD_RASTER_GET
  )


  /* This renderer is _NOT_ part of the default modules; you will need */
  /* to register it by hand in your application.  It should only be    */
  /* used for backwards-compatibility with FT 1.x anyway.              */
  /*                                                                   */
  FT_DEFINE_RENDERER( ft_raster5_renderer_class,

      FT_MODULE_RENDERER,
      sizeof ( FT_RendererRec ),

      "raster5",
      0x10000L,
      0x20000L,

      0,    /* module specific interface */

      (FT_Module_Constructor)ft_raster1_init,
      (FT_Module_Destructor) 0,
      (FT_Module_Requester)  0
    ,

    FT_GLYPH_FORMAT_OUTLINE,

    (FT_Renderer_RenderFunc)   ft_raster1_render,
    (FT_Renderer_TransformFunc)ft_raster1_transform,
    (FT_Renderer_GetCBoxFunc)  ft_raster1_get_cbox,
    (FT_Renderer_SetModeFunc)  ft_raster1_set_mode,

    (FT_Raster_Funcs*)    &FT_STANDARD_RASTER_GET
  )


/* END */
