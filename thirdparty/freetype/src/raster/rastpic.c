/***************************************************************************/
/*                                                                         */
/*  rastpic.c                                                              */
/*                                                                         */
/*    The FreeType position independent code services for raster module.   */
/*                                                                         */
/*  Copyright 2009-2016 by                                                 */
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
#include "rastpic.h"
#include "rasterrs.h"


#ifdef FT_CONFIG_OPTION_PIC

  /* forward declaration of PIC init functions from ftraster.c */
  void
  FT_Init_Class_ft_standard_raster( FT_Raster_Funcs*  funcs );


  void
  ft_raster1_renderer_class_pic_free( FT_Library  library )
  {
    FT_PIC_Container*  pic_container = &library->pic_container;
    FT_Memory          memory        = library->memory;


    if ( pic_container->raster )
    {
      RasterPIC*  container = (RasterPIC*)pic_container->raster;


      if ( --container->ref_count )
        return;
      FT_FREE( container );
      pic_container->raster = NULL;
    }
  }


  FT_Error
  ft_raster1_renderer_class_pic_init( FT_Library  library )
  {
    FT_PIC_Container*  pic_container = &library->pic_container;
    FT_Error           error         = FT_Err_Ok;
    RasterPIC*         container     = NULL;
    FT_Memory          memory        = library->memory;


    /* XXX: since this function also served the no longer available  */
    /*      raster5 renderer it uses reference counting, which could */
    /*      be removed now                                           */
    if ( pic_container->raster )
    {
      ((RasterPIC*)pic_container->raster)->ref_count++;
      return error;
    }

    /* allocate pointer, clear and set global container pointer */
    if ( FT_ALLOC( container, sizeof ( *container ) ) )
      return error;
    FT_MEM_SET( container, 0, sizeof ( *container ) );
    pic_container->raster = container;

    container->ref_count = 1;

    /* initialize pointer table -                       */
    /* this is how the module usually expects this data */
    FT_Init_Class_ft_standard_raster( &container->ft_standard_raster );

    return error;
  }

#endif /* FT_CONFIG_OPTION_PIC */


/* END */
