/***************************************************************************/
/*                                                                         */
/*  ftspic.c                                                               */
/*                                                                         */
/*    The FreeType position independent code services for smooth module.   */
/*                                                                         */
/*  Copyright 2009-2017 by                                                 */
/*  Oran Agra and Mickey Gabel.                                            */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


#include <ft2build.h>
#include FT_FREETYPE_H
#include FT_INTERNAL_OBJECTS_H
#include "ftspic.h"
#include "ftsmerrs.h"


#ifdef FT_CONFIG_OPTION_PIC

  /* forward declaration of PIC init functions from ftgrays.c */
  void
  FT_Init_Class_ft_grays_raster( FT_Raster_Funcs*  funcs );


  void
  ft_smooth_renderer_class_pic_free( FT_Library  library )
  {
    FT_PIC_Container*  pic_container = &library->pic_container;
    FT_Memory          memory        = library->memory;


    if ( pic_container->smooth )
    {
      SmoothPIC*  container = (SmoothPIC*)pic_container->smooth;


      if ( --container->ref_count )
        return;

      FT_FREE( container );
      pic_container->smooth = NULL;
    }
  }


  FT_Error
  ft_smooth_renderer_class_pic_init( FT_Library  library )
  {
    FT_PIC_Container*  pic_container = &library->pic_container;
    FT_Error           error         = FT_Err_Ok;
    SmoothPIC*         container     = NULL;
    FT_Memory          memory        = library->memory;


    /* since this function also serve smooth_lcd and smooth_lcdv renderers,
       it implements reference counting */
    if ( pic_container->smooth )
    {
      ((SmoothPIC*)pic_container->smooth)->ref_count++;
      return error;
    }

    /* allocate pointer, clear and set global container pointer */
    if ( FT_ALLOC( container, sizeof ( *container ) ) )
      return error;
    FT_MEM_SET( container, 0, sizeof ( *container ) );
    pic_container->smooth = container;

    container->ref_count = 1;

    /* initialize pointer table -                       */
    /* this is how the module usually expects this data */
    FT_Init_Class_ft_grays_raster( &container->ft_grays_raster );

    return error;
  }


  /* re-route these init and free functions to the above functions */
  FT_Error
  ft_smooth_lcd_renderer_class_pic_init( FT_Library  library )
  {
    return ft_smooth_renderer_class_pic_init( library );
  }


  void
  ft_smooth_lcd_renderer_class_pic_free( FT_Library  library )
  {
    ft_smooth_renderer_class_pic_free( library );
  }


  FT_Error
  ft_smooth_lcdv_renderer_class_pic_init( FT_Library  library )
  {
    return ft_smooth_renderer_class_pic_init( library );
  }


  void
  ft_smooth_lcdv_renderer_class_pic_free( FT_Library  library )
  {
    ft_smooth_renderer_class_pic_free( library );
  }

#endif /* FT_CONFIG_OPTION_PIC */


/* END */
