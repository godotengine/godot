/***************************************************************************/
/*                                                                         */
/*  pspic.h                                                                */
/*                                                                         */
/*    The FreeType position independent code services for psnames module.  */
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


#ifndef PSPIC_H_
#define PSPIC_H_


#include FT_INTERNAL_PIC_H


#ifndef FT_CONFIG_OPTION_PIC

#define PSCMAPS_SERVICES_GET   pscmaps_services
#define PSCMAPS_INTERFACE_GET  pscmaps_interface

#else /* FT_CONFIG_OPTION_PIC */

#include FT_SERVICE_POSTSCRIPT_CMAPS_H


FT_BEGIN_HEADER

  typedef struct  PSModulePIC_
  {
    FT_ServiceDescRec*     pscmaps_services;
    FT_Service_PsCMapsRec  pscmaps_interface;

  } PSModulePIC;


#define GET_PIC( lib )                                     \
          ( (PSModulePIC*)((lib)->pic_container.psnames) )
#define PSCMAPS_SERVICES_GET   ( GET_PIC( library )->pscmaps_services )
#define PSCMAPS_INTERFACE_GET  ( GET_PIC( library )->pscmaps_interface )


  /* see pspic.c for the implementation */
  void
  psnames_module_class_pic_free( FT_Library  library );

  FT_Error
  psnames_module_class_pic_init( FT_Library  library );

FT_END_HEADER

#endif /* FT_CONFIG_OPTION_PIC */

 /* */

#endif /* PSPIC_H_ */


/* END */
