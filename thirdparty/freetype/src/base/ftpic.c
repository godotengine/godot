/***************************************************************************/
/*                                                                         */
/*  ftpic.c                                                                */
/*                                                                         */
/*    The FreeType position independent code services (body).              */
/*                                                                         */
/*  Copyright 2009-2018 by                                                 */
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
#include "basepic.h"

#ifdef FT_CONFIG_OPTION_PIC

  /* documentation is in ftpic.h */

  FT_BASE_DEF( FT_Error )
  ft_pic_container_init( FT_Library  library )
  {
    FT_PIC_Container*  pic_container = &library->pic_container;
    FT_Error           error;


    FT_MEM_SET( pic_container, 0, sizeof ( *pic_container ) );

    error = ft_base_pic_init( library );
    if ( error )
      return error;

    return FT_Err_Ok;
  }


  /* Destroy the contents of the container. */
  FT_BASE_DEF( void )
  ft_pic_container_destroy( FT_Library  library )
  {
    ft_base_pic_free( library );
  }

#endif /* FT_CONFIG_OPTION_PIC */


/* END */
