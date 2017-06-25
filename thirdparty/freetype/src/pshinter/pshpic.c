/***************************************************************************/
/*                                                                         */
/*  pshpic.c                                                               */
/*                                                                         */
/*    The FreeType position independent code services for pshinter module. */
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
#include "pshpic.h"
#include "pshnterr.h"


#ifdef FT_CONFIG_OPTION_PIC

  /* forward declaration of PIC init functions from pshmod.c */
  void
  FT_Init_Class_pshinter_interface( FT_Library           library,
                                    PSHinter_Interface*  clazz );

  void
  pshinter_module_class_pic_free( FT_Library  library )
  {
    FT_PIC_Container*  pic_container = &library->pic_container;
    FT_Memory          memory        = library->memory;


    if ( pic_container->pshinter )
    {
      FT_FREE( pic_container->pshinter );
      pic_container->pshinter = NULL;
    }
  }


  FT_Error
  pshinter_module_class_pic_init( FT_Library  library )
  {
    FT_PIC_Container*  pic_container = &library->pic_container;
    FT_Error           error         = FT_Err_Ok;
    PSHinterPIC*       container     = NULL;
    FT_Memory          memory        = library->memory;


    /* allocate pointer, clear and set global container pointer */
    if ( FT_ALLOC( container, sizeof ( *container ) ) )
      return error;
    FT_MEM_SET( container, 0, sizeof ( *container ) );
    pic_container->pshinter = container;

    /* add call to initialization function when you add new scripts */
    FT_Init_Class_pshinter_interface(
      library, &container->pshinter_interface );

    if ( error )
      pshinter_module_class_pic_free( library );

    return error;
  }

#endif /* FT_CONFIG_OPTION_PIC */


/* END */
