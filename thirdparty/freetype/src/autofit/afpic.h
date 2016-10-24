/***************************************************************************/
/*                                                                         */
/*  afpic.h                                                                */
/*                                                                         */
/*    The FreeType position independent code services for autofit module.  */
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


#ifndef AFPIC_H_
#define AFPIC_H_


#include FT_INTERNAL_PIC_H


#ifndef FT_CONFIG_OPTION_PIC

#define AF_SERVICES_GET                af_services
#define AF_SERVICE_PROPERTIES_GET      af_service_properties

#define AF_WRITING_SYSTEM_CLASSES_GET  af_writing_system_classes
#define AF_SCRIPT_CLASSES_GET          af_script_classes
#define AF_STYLE_CLASSES_GET           af_style_classes
#define AF_INTERFACE_GET               af_autofitter_interface

#else /* FT_CONFIG_OPTION_PIC */

  /* some include files required for members of AFModulePIC */
#include FT_SERVICE_PROPERTIES_H

#include "aftypes.h"


FT_BEGIN_HEADER

  typedef struct  AFModulePIC_
  {
    FT_ServiceDescRec*          af_services;
    FT_Service_PropertiesRec    af_service_properties;

    AF_WritingSystemClass       af_writing_system_classes
                                  [AF_WRITING_SYSTEM_MAX + 1];
    AF_WritingSystemClassRec    af_writing_system_classes_rec
                                  [AF_WRITING_SYSTEM_MAX];

    AF_ScriptClass              af_script_classes
                                  [AF_SCRIPT_MAX + 1];
    AF_ScriptClassRec           af_script_classes_rec
                                  [AF_SCRIPT_MAX];

    AF_StyleClass               af_style_classes
                                  [AF_STYLE_MAX + 1];
    AF_StyleClassRec            af_style_classes_rec
                                  [AF_STYLE_MAX];

    FT_AutoHinter_InterfaceRec  af_autofitter_interface;

  } AFModulePIC;


#define GET_PIC( lib )  \
          ( (AFModulePIC*)((lib)->pic_container.autofit) )

#define AF_SERVICES_GET  \
          ( GET_PIC( library )->af_services )
#define AF_SERVICE_PROPERTIES_GET  \
          ( GET_PIC( library )->af_service_properties )

#define AF_WRITING_SYSTEM_CLASSES_GET  \
          ( GET_PIC( FT_FACE_LIBRARY( globals->face ) )->af_writing_system_classes )
#define AF_SCRIPT_CLASSES_GET  \
          ( GET_PIC( FT_FACE_LIBRARY( globals->face ) )->af_script_classes )
#define AF_STYLE_CLASSES_GET  \
          ( GET_PIC( FT_FACE_LIBRARY( globals->face ) )->af_style_classes )
#define AF_INTERFACE_GET  \
          ( GET_PIC( library )->af_autofitter_interface )


  /* see afpic.c for the implementation */
  void
  autofit_module_class_pic_free( FT_Library  library );

  FT_Error
  autofit_module_class_pic_init( FT_Library  library );

FT_END_HEADER

#endif /* FT_CONFIG_OPTION_PIC */

 /* */

#endif /* AFPIC_H_ */


/* END */
