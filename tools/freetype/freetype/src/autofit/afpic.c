/***************************************************************************/
/*                                                                         */
/*  afpic.c                                                                */
/*                                                                         */
/*    The FreeType position independent code services for autofit module.  */
/*                                                                         */
/*  Copyright 2009-2013 by                                                 */
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
#include "afpic.h"
#include "aferrors.h"


#ifdef FT_CONFIG_OPTION_PIC

  /* forward declaration of PIC init functions from afmodule.c */
  FT_Error
  FT_Create_Class_af_services( FT_Library           library,
                               FT_ServiceDescRec**  output_class );

  void
  FT_Destroy_Class_af_services( FT_Library          library,
                                FT_ServiceDescRec*  clazz );

  void
  FT_Init_Class_af_service_properties( FT_Service_PropertiesRec*  clazz );

  void FT_Init_Class_af_autofitter_interface(
    FT_Library                   library,
    FT_AutoHinter_InterfaceRec*  clazz );


  /* forward declaration of PIC init functions from script classes */
#include "aflatin.h"
#ifdef FT_OPTION_AUTOFIT2
#include "aflatin2.h"
#endif
#include "afcjk.h"
#include "afdummy.h"
#include "afindic.h"


  void
  autofit_module_class_pic_free( FT_Library  library )
  {
    FT_PIC_Container*  pic_container = &library->pic_container;
    FT_Memory          memory        = library->memory;


    if ( pic_container->autofit )
    {
      AFModulePIC*  container = (AFModulePIC*)pic_container->autofit;


      if ( container->af_services )
        FT_Destroy_Class_af_services( library,
                                      container->af_services );
      container->af_services = NULL;

      FT_FREE( container );
      pic_container->autofit = NULL;
    }
  }


  FT_Error
  autofit_module_class_pic_init( FT_Library  library )
  {
    FT_PIC_Container*  pic_container = &library->pic_container;
    FT_UInt            ss;
    FT_Error           error         = FT_Err_Ok;
    AFModulePIC*       container     = NULL;
    FT_Memory          memory        = library->memory;


    /* allocate pointer, clear and set global container pointer */
    if ( FT_ALLOC ( container, sizeof ( *container ) ) )
      return error;
    FT_MEM_SET( container, 0, sizeof ( *container ) );
    pic_container->autofit = container;

    /* initialize pointer table -                       */
    /* this is how the module usually expects this data */
    error = FT_Create_Class_af_services( library,
                                         &container->af_services );
    if ( error )
      goto Exit;

    FT_Init_Class_af_service_properties( &container->af_service_properties );

    for ( ss = 0 ; ss < AF_SCRIPT_CLASSES_REC_COUNT ; ss++ )
    {
      container->af_script_classes[ss] =
        &container->af_script_classes_rec[ss];
    }
    container->af_script_classes[AF_SCRIPT_CLASSES_COUNT - 1] = NULL;

    /* add call to initialization function when you add new scripts */
    ss = 0;
    FT_Init_Class_af_dummy_script_class(
      &container->af_script_classes_rec[ss++] );
#ifdef FT_OPTION_AUTOFIT2
    FT_Init_Class_af_latin2_script_class(
      &container->af_script_classes_rec[ss++] );
#endif
    FT_Init_Class_af_latin_script_class(
      &container->af_script_classes_rec[ss++] );
    FT_Init_Class_af_cjk_script_class(
      &container->af_script_classes_rec[ss++] );
    FT_Init_Class_af_indic_script_class(
      &container->af_script_classes_rec[ss++] );

    FT_Init_Class_af_autofitter_interface(
      library, &container->af_autofitter_interface );

  Exit:
    if ( error )
      autofit_module_class_pic_free( library );
    return error;
  }

#endif /* FT_CONFIG_OPTION_PIC */


/* END */
