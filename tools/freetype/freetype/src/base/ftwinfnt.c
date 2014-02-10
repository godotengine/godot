/***************************************************************************/
/*                                                                         */
/*  ftwinfnt.c                                                             */
/*                                                                         */
/*    FreeType API for accessing Windows FNT specific info (body).         */
/*                                                                         */
/*  Copyright 2003, 2004 by                                                */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


#include <ft2build.h>
#include FT_WINFONTS_H
#include FT_INTERNAL_OBJECTS_H
#include FT_SERVICE_WINFNT_H


  /* documentation is in ftwinfnt.h */

  FT_EXPORT_DEF( FT_Error )
  FT_Get_WinFNT_Header( FT_Face               face,
                        FT_WinFNT_HeaderRec  *header )
  {
    FT_Service_WinFnt  service;
    FT_Error           error;


    error = FT_ERR( Invalid_Argument );

    if ( face != NULL )
    {
      FT_FACE_LOOKUP_SERVICE( face, service, WINFNT );

      if ( service != NULL )
      {
        error = service->get_header( face, header );
      }
    }

    return error;
  }


/* END */
