/***************************************************************************/
/*                                                                         */
/*  ftwinfnt.c                                                             */
/*                                                                         */
/*    FreeType API for accessing Windows FNT specific info (body).         */
/*                                                                         */
/*  Copyright 2003-2016 by                                                 */
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
#include FT_INTERNAL_DEBUG_H
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


    if ( !face )
      return FT_THROW( Invalid_Face_Handle );

    if ( !header )
      return FT_THROW( Invalid_Argument );

    FT_FACE_LOOKUP_SERVICE( face, service, WINFNT );

    if ( service )
      error = service->get_header( face, header );
    else
      error = FT_THROW( Invalid_Argument );

    return error;
  }


/* END */
