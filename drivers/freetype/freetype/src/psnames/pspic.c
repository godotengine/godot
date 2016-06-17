/***************************************************************************/
/*                                                                         */
/*  pspic.c                                                                */
/*                                                                         */
/*    The FreeType position independent code services for psnames module.  */
/*                                                                         */
/*  Copyright 2009, 2010, 2012, 2013 by                                    */
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
#include "pspic.h"
#include "psnamerr.h"


#ifdef FT_CONFIG_OPTION_PIC

  /* forward declaration of PIC init functions from psmodule.c */
  FT_Error
  FT_Create_Class_pscmaps_services( FT_Library           library,
                                    FT_ServiceDescRec**  output_class );
  void
  FT_Destroy_Class_pscmaps_services( FT_Library          library,
                                     FT_ServiceDescRec*  clazz );

  void
  FT_Init_Class_pscmaps_interface( FT_Library              library,
                                   FT_Service_PsCMapsRec*  clazz );


  void
  psnames_module_class_pic_free( FT_Library  library )
  {
    FT_PIC_Container*  pic_container = &library->pic_container;
    FT_Memory          memory        = library->memory;


    if ( pic_container->psnames )
    {
      PSModulePIC*  container = (PSModulePIC*)pic_container->psnames;


      if ( container->pscmaps_services )
        FT_Destroy_Class_pscmaps_services( library,
                                           container->pscmaps_services );
      container->pscmaps_services = NULL;
      FT_FREE( container );
      pic_container->psnames = NULL;
    }
  }


  FT_Error
  psnames_module_class_pic_init( FT_Library  library )
  {
    FT_PIC_Container*  pic_container = &library->pic_container;
    FT_Error           error         = FT_Err_Ok;
    PSModulePIC*       container     = NULL;
    FT_Memory          memory        = library->memory;


    /* allocate pointer, clear and set global container pointer */
    if ( FT_ALLOC( container, sizeof ( *container ) ) )
      return error;
    FT_MEM_SET( container, 0, sizeof ( *container ) );
    pic_container->psnames = container;

    /* initialize pointer table -                       */
    /* this is how the module usually expects this data */
    error = FT_Create_Class_pscmaps_services(
              library, &container->pscmaps_services );
    if ( error )
      goto Exit;
    FT_Init_Class_pscmaps_interface( library,
                                     &container->pscmaps_interface );

  Exit:
    if ( error )
      psnames_module_class_pic_free( library );
    return error;
  }


#endif /* FT_CONFIG_OPTION_PIC */


/* END */
