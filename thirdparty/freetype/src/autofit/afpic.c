/***************************************************************************/
/*                                                                         */
/*  afpic.c                                                                */
/*                                                                         */
/*    The FreeType position independent code services for autofit module.  */
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
#include "afpic.h"
#include "afglobal.h"
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


  /* forward declaration of PIC init functions from writing system classes */
#undef  WRITING_SYSTEM
#define WRITING_SYSTEM( ws, WS )  /* empty */

#include "afwrtsys.h"


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

    for ( ss = 0; ss < AF_WRITING_SYSTEM_MAX; ss++ )
      container->af_writing_system_classes[ss] =
        &container->af_writing_system_classes_rec[ss];
    container->af_writing_system_classes[AF_WRITING_SYSTEM_MAX] = NULL;

    for ( ss = 0; ss < AF_SCRIPT_MAX; ss++ )
      container->af_script_classes[ss] =
        &container->af_script_classes_rec[ss];
    container->af_script_classes[AF_SCRIPT_MAX] = NULL;

    for ( ss = 0; ss < AF_STYLE_MAX; ss++ )
      container->af_style_classes[ss] =
        &container->af_style_classes_rec[ss];
    container->af_style_classes[AF_STYLE_MAX] = NULL;

#undef  WRITING_SYSTEM
#define WRITING_SYSTEM( ws, WS )                             \
        FT_Init_Class_af_ ## ws ## _writing_system_class(    \
          &container->af_writing_system_classes_rec[ss++] );

    ss = 0;
#include "afwrtsys.h"

#undef  SCRIPT
#define SCRIPT( s, S, d, h, H, sss )                 \
        FT_Init_Class_af_ ## s ## _script_class(     \
          &container->af_script_classes_rec[ss++] );

    ss = 0;
#include "afscript.h"

#undef  STYLE
#define STYLE( s, S, d, ws, sc, bss, c )            \
        FT_Init_Class_af_ ## s ## _style_class(     \
          &container->af_style_classes_rec[ss++] );

    ss = 0;
#include "afstyles.h"

    FT_Init_Class_af_autofitter_interface(
      library, &container->af_autofitter_interface );

  Exit:
    if ( error )
      autofit_module_class_pic_free( library );
    return error;
  }

#endif /* FT_CONFIG_OPTION_PIC */


/* END */
