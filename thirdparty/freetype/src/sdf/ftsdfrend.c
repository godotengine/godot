/****************************************************************************
 *
 * ftsdfrend.c
 *
 *   Signed Distance Field renderer interface (body).
 *
 * Copyright (C) 2020-2025 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * Written by Anuj Verma.
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
#include <freetype/internal/services/svprop.h>
#include <freetype/ftoutln.h>
#include <freetype/ftbitmap.h>
#include "ftsdfrend.h"
#include "ftsdf.h"

#include "ftsdferrs.h"


  /**************************************************************************
   *
   * The macro FT_COMPONENT is used in trace mode.  It is an implicit
   * parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log
   * messages during execution.
   */
#undef  FT_COMPONENT
#define FT_COMPONENT  sdf


  /**************************************************************************
   *
   * macros and default property values
   *
   */
#define SDF_RENDERER( rend )  ( (SDF_Renderer)rend )


  /**************************************************************************
   *
   * for setting properties
   *
   */

  /* property setter function */
  static FT_Error
  sdf_property_set( FT_Module    module,
                    const char*  property_name,
                    const void*  value,
                    FT_Bool      value_is_string )
  {
    FT_Error      error  = FT_Err_Ok;
    SDF_Renderer  render = SDF_RENDERER( FT_RENDERER( module ) );

    FT_UNUSED( value_is_string );


    if ( ft_strcmp( property_name, "spread" ) == 0 )
    {
      FT_Int  val = *(const FT_Int*)value;


      if ( val > MAX_SPREAD || val < MIN_SPREAD )
      {
        FT_TRACE0(( "[sdf] sdf_property_set:"
                    " the `spread' property can have a value\n" ));
        FT_TRACE0(( "                       "
                    " within range [%d, %d] (value provided: %d)\n",
                    MIN_SPREAD, MAX_SPREAD, val ));

        error = FT_THROW( Invalid_Argument );
        goto Exit;
      }

      render->spread = (FT_UInt)val;
      FT_TRACE7(( "[sdf] sdf_property_set:"
                  " updated property `spread' to %d\n", val ));
    }

    else if ( ft_strcmp( property_name, "flip_sign" ) == 0 )
    {
      FT_Int  val = *(const FT_Int*)value;


      render->flip_sign = val ? 1 : 0;
      FT_TRACE7(( "[sdf] sdf_property_set:"
                  " updated property `flip_sign' to %d\n", val ));
    }

    else if ( ft_strcmp( property_name, "flip_y" ) == 0 )
    {
      FT_Int  val = *(const FT_Int*)value;


      render->flip_y = val ? 1 : 0;
      FT_TRACE7(( "[sdf] sdf_property_set:"
                  " updated property `flip_y' to %d\n", val ));
    }

    else if ( ft_strcmp( property_name, "overlaps" ) == 0 )
    {
      FT_Bool  val = *(const FT_Bool*)value;


      render->overlaps = val;
      FT_TRACE7(( "[sdf] sdf_property_set:"
                  " updated property `overlaps' to %d\n", val ));
    }

    else
    {
      FT_TRACE0(( "[sdf] sdf_property_set:"
                  " missing property `%s'\n", property_name ));
      error = FT_THROW( Missing_Property );
    }

  Exit:
    return error;
  }


  /* property getter function */
  static FT_Error
  sdf_property_get( FT_Module    module,
                    const char*  property_name,
                    void*        value )
  {
    FT_Error      error  = FT_Err_Ok;
    SDF_Renderer  render = SDF_RENDERER( FT_RENDERER( module ) );


    if ( ft_strcmp( property_name, "spread" ) == 0 )
    {
      FT_UInt*  val = (FT_UInt*)value;


      *val = render->spread;
    }

    else if ( ft_strcmp( property_name, "flip_sign" ) == 0 )
    {
      FT_Int*  val = (FT_Int*)value;


      *val = render->flip_sign;
    }

    else if ( ft_strcmp( property_name, "flip_y" ) == 0 )
    {
      FT_Int*  val = (FT_Int*)value;


      *val = render->flip_y;
    }

    else if ( ft_strcmp( property_name, "overlaps" ) == 0 )
    {
      FT_Int*  val = (FT_Int*)value;


      *val = render->overlaps;
    }

    else
    {
      FT_TRACE0(( "[sdf] sdf_property_get:"
                  " missing property `%s'\n", property_name ));
      error = FT_THROW( Missing_Property );
    }

    return error;
  }


  FT_DEFINE_SERVICE_PROPERTIESREC(
    sdf_service_properties,

    (FT_Properties_SetFunc)sdf_property_set,        /* set_property */
    (FT_Properties_GetFunc)sdf_property_get )       /* get_property */


  FT_DEFINE_SERVICEDESCREC1(
    sdf_services,

    FT_SERVICE_ID_PROPERTIES, &sdf_service_properties )


  static FT_Module_Interface
  ft_sdf_requester( FT_Module    module,
                    const char*  module_interface )
  {
    FT_UNUSED( module );

    return ft_service_list_lookup( sdf_services, module_interface );
  }


  /*************************************************************************/
  /*************************************************************************/
  /**                                                                     **/
  /**  OUTLINE TO SDF CONVERTER                                           **/
  /**                                                                     **/
  /*************************************************************************/
  /*************************************************************************/

  /**************************************************************************
   *
   * interface functions
   *
   */

  static FT_Error
  ft_sdf_init( FT_Module  module )   /* SDF_Renderer */
  {
    SDF_Renderer  sdf_render = SDF_RENDERER( module );


    sdf_render->spread    = DEFAULT_SPREAD;
    sdf_render->flip_sign = 0;
    sdf_render->flip_y    = 0;
    sdf_render->overlaps  = 0;

    return FT_Err_Ok;
  }


  static void
  ft_sdf_done( FT_Module  module )
  {
    FT_UNUSED( module );
  }


  /* generate signed distance field from a glyph's slot image */
  static FT_Error
  ft_sdf_render( FT_Renderer       module,
                 FT_GlyphSlot      slot,
                 FT_Render_Mode    mode,
                 const FT_Vector*  origin )
  {
    FT_Error     error   = FT_Err_Ok;
    FT_Outline*  outline = &slot->outline;
    FT_Bitmap*   bitmap  = &slot->bitmap;
    FT_Memory    memory  = NULL;
    FT_Renderer  render  = NULL;

    FT_Pos  x_shift = 0;
    FT_Pos  y_shift = 0;

    FT_Pos  x_pad = 0;
    FT_Pos  y_pad = 0;

    SDF_Raster_Params  params;
    SDF_Renderer       sdf_module = SDF_RENDERER( module );


    render = &sdf_module->root;
    memory = render->root.memory;

    /* check whether slot format is correct before rendering */
    if ( slot->format != render->glyph_format )
    {
      error = FT_THROW( Invalid_Glyph_Format );
      goto Exit;
    }

    /* check whether render mode is correct */
    if ( mode != FT_RENDER_MODE_SDF )
    {
      error = FT_THROW( Cannot_Render_Glyph );
      goto Exit;
    }

    /* deallocate the previously allocated bitmap */
    if ( slot->internal->flags & FT_GLYPH_OWN_BITMAP )
    {
      FT_FREE( bitmap->buffer );
      slot->internal->flags &= ~FT_GLYPH_OWN_BITMAP;
    }

    /* preset the bitmap using the glyph's outline;         */
    /* the sdf bitmap is similar to an anti-aliased bitmap  */
    /* with a slightly bigger size and different pixel mode */
    if ( ft_glyphslot_preset_bitmap( slot, FT_RENDER_MODE_NORMAL, origin ) )
    {
      error = FT_THROW( Raster_Overflow );
      goto Exit;
    }

    /* nothing to render */
    if ( !bitmap->rows || !bitmap->pitch )
      goto Exit;

    /* the padding will simply be equal to the `spread' */
    x_pad = sdf_module->spread;
    y_pad = sdf_module->spread;

    /* apply the padding; will be in all the directions */
    bitmap->rows  += y_pad * 2;
    bitmap->width += x_pad * 2;

    /* ignore the pitch, pixel mode and set custom */
    bitmap->pixel_mode = FT_PIXEL_MODE_GRAY;
    bitmap->pitch      = (int)( bitmap->width );
    bitmap->num_grays  = 255;

    /* allocate new buffer */
    if ( FT_ALLOC_MULT( bitmap->buffer, bitmap->rows, bitmap->pitch ) )
      goto Exit;

    slot->internal->flags |= FT_GLYPH_OWN_BITMAP;

    slot->bitmap_top  += y_pad;
    slot->bitmap_left -= x_pad;

    x_shift  = 64 * -slot->bitmap_left;
    y_shift  = 64 * -slot->bitmap_top;
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
    params.root.target = bitmap;
    params.root.source = outline;
    params.root.flags  = FT_RASTER_FLAG_SDF;
    params.spread      = sdf_module->spread;
    params.flip_sign   = sdf_module->flip_sign;
    params.flip_y      = sdf_module->flip_y;
    params.overlaps    = sdf_module->overlaps;

    /* render the outline */
    error = render->raster_render( render->raster,
                                   (const FT_Raster_Params*)&params );

    /* transform the outline back to the original state */
    if ( x_shift || y_shift )
      FT_Outline_Translate( outline, -x_shift, -y_shift );

  Exit:
    if ( !error )
    {
      /* the glyph is successfully rendered to a bitmap */
      slot->format = FT_GLYPH_FORMAT_BITMAP;
    }
    else if ( slot->internal->flags & FT_GLYPH_OWN_BITMAP )
    {
      FT_FREE( bitmap->buffer );
      slot->internal->flags &= ~FT_GLYPH_OWN_BITMAP;
    }

    return error;
  }


  /* transform the glyph using matrix and/or delta */
  static FT_Error
  ft_sdf_transform( FT_Renderer       render,
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


  /* return the control box of a glyph's outline */
  static void
  ft_sdf_get_cbox( FT_Renderer   render,
                   FT_GlyphSlot  slot,
                   FT_BBox*      cbox )
  {
    FT_ZERO( cbox );

    if ( slot->format == render->glyph_format )
      FT_Outline_Get_CBox( &slot->outline, cbox );
  }


  /* set render specific modes or attributes */
  static FT_Error
  ft_sdf_set_mode( FT_Renderer  render,
                   FT_ULong     mode_tag,
                   FT_Pointer   data )
  {
    /* pass it to the rasterizer */
    return render->clazz->raster_class->raster_set_mode( render->raster,
                                                         mode_tag,
                                                         data );
  }


  FT_DEFINE_RENDERER(
    ft_sdf_renderer_class,

    FT_MODULE_RENDERER,
    sizeof ( SDF_Renderer_Module ),

    "sdf",
    0x10000L,
    0x20000L,

    NULL,

    (FT_Module_Constructor)ft_sdf_init,
    (FT_Module_Destructor) ft_sdf_done,
    (FT_Module_Requester)  ft_sdf_requester,

    FT_GLYPH_FORMAT_OUTLINE,

    (FT_Renderer_RenderFunc)   ft_sdf_render,     /* render_glyph    */
    (FT_Renderer_TransformFunc)ft_sdf_transform,  /* transform_glyph */
    (FT_Renderer_GetCBoxFunc)  ft_sdf_get_cbox,   /* get_glyph_cbox  */
    (FT_Renderer_SetModeFunc)  ft_sdf_set_mode,   /* set_mode        */

    (FT_Raster_Funcs*)&ft_sdf_raster              /* raster_class    */
  )


  /*************************************************************************/
  /*************************************************************************/
  /**                                                                     **/
  /**  BITMAP TO SDF CONVERTER                                            **/
  /**                                                                     **/
  /*************************************************************************/
  /*************************************************************************/

  /* generate signed distance field from glyph's bitmap */
  static FT_Error
  ft_bsdf_render( FT_Renderer       module,
                  FT_GlyphSlot      slot,
                  FT_Render_Mode    mode,
                  const FT_Vector*  origin )
  {
    FT_Error   error  = FT_Err_Ok;
    FT_Memory  memory = NULL;

    FT_Bitmap*   bitmap  = &slot->bitmap;
    FT_Renderer  render  = NULL;
    FT_Bitmap    target;

    FT_Pos  x_pad = 0;
    FT_Pos  y_pad = 0;

    SDF_Raster_Params  params;
    SDF_Renderer       sdf_module = SDF_RENDERER( module );


    /* initialize the bitmap in case any error occurs */
    FT_Bitmap_Init( &target );

    render = &sdf_module->root;
    memory = render->root.memory;

    /* check whether slot format is correct before rendering */
    if ( slot->format != render->glyph_format )
    {
      error = FT_THROW( Invalid_Glyph_Format );
      goto Exit;
    }

    /* check whether render mode is correct */
    if ( mode != FT_RENDER_MODE_SDF )
    {
      error = FT_THROW( Cannot_Render_Glyph );
      goto Exit;
    }

    if ( origin )
    {
      FT_ERROR(( "ft_bsdf_render: can't translate the bitmap\n" ));

      error = FT_THROW( Unimplemented_Feature );
      goto Exit;
    }

    /* nothing to render */
    if ( !bitmap->rows || !bitmap->pitch )
      goto Exit;

    /* Do not generate SDF if the bitmap is not owned by the       */
    /* glyph: it might be that the source buffer is already freed. */
    if ( !( slot->internal->flags & FT_GLYPH_OWN_BITMAP ) )
    {
      FT_ERROR(( "ft_bsdf_render: can't generate SDF from"
                 " unowned source bitmap\n" ));

      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    FT_Bitmap_New( &target );

    /* padding will simply be equal to `spread` */
    x_pad = sdf_module->spread;
    y_pad = sdf_module->spread;

    /* apply padding, which extends to all directions */
    target.rows  = bitmap->rows  + y_pad * 2;
    target.width = bitmap->width + x_pad * 2;

    /* set up the target bitmap */
    target.pixel_mode = FT_PIXEL_MODE_GRAY;
    target.pitch      = (int)( target.width );
    target.num_grays  = 255;

    if ( FT_ALLOC_MULT( target.buffer, target.rows, target.pitch ) )
      goto Exit;

    /* set up parameters */
    params.root.target = &target;
    params.root.source = bitmap;
    params.root.flags  = FT_RASTER_FLAG_SDF;
    params.spread      = sdf_module->spread;
    params.flip_sign   = sdf_module->flip_sign;
    params.flip_y      = sdf_module->flip_y;

    error = render->raster_render( render->raster,
                                   (const FT_Raster_Params*)&params );

  Exit:
    if ( !error )
    {
      /* the glyph is successfully converted to a SDF */
      if ( slot->internal->flags & FT_GLYPH_OWN_BITMAP )
        FT_FREE( bitmap->buffer );

      slot->bitmap       = target;
      slot->bitmap_top  += y_pad;
      slot->bitmap_left -= x_pad;

      if ( target.buffer )
        slot->internal->flags |= FT_GLYPH_OWN_BITMAP;
    }
    else if ( target.buffer )
      FT_FREE( target.buffer );

    return error;
  }


  FT_DEFINE_RENDERER(
    ft_bitmap_sdf_renderer_class,

    FT_MODULE_RENDERER,
    sizeof ( SDF_Renderer_Module ),

    "bsdf",
    0x10000L,
    0x20000L,

    NULL,

    (FT_Module_Constructor)ft_sdf_init,
    (FT_Module_Destructor) ft_sdf_done,
    (FT_Module_Requester)  ft_sdf_requester,

    FT_GLYPH_FORMAT_BITMAP,

    (FT_Renderer_RenderFunc)   ft_bsdf_render,    /* render_glyph    */
    (FT_Renderer_TransformFunc)ft_sdf_transform,  /* transform_glyph */
    (FT_Renderer_GetCBoxFunc)  ft_sdf_get_cbox,   /* get_glyph_cbox  */
    (FT_Renderer_SetModeFunc)  ft_sdf_set_mode,   /* set_mode        */

    (FT_Raster_Funcs*)&ft_bitmap_sdf_raster       /* raster_class    */
  )


/* END */
