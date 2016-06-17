/***************************************************************************/
/*                                                                         */
/*  ftbdf.c                                                                */
/*                                                                         */
/*    FreeType API for accessing BDF-specific strings (body).              */
/*                                                                         */
/*  Copyright 2002-2004, 2013 by                                           */
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
#include FT_INTERNAL_OBJECTS_H
#include FT_SERVICE_BDF_H


  /* documentation is in ftbdf.h */

  FT_EXPORT_DEF( FT_Error )
  FT_Get_BDF_Charset_ID( FT_Face       face,
                         const char*  *acharset_encoding,
                         const char*  *acharset_registry )
  {
    FT_Error     error;
    const char*  encoding = NULL;
    const char*  registry = NULL;


    error = FT_ERR( Invalid_Argument );

    if ( face )
    {
      FT_Service_BDF  service;


      FT_FACE_FIND_SERVICE( face, service, BDF );

      if ( service && service->get_charset_id )
        error = service->get_charset_id( face, &encoding, &registry );
    }

    if ( acharset_encoding )
      *acharset_encoding = encoding;

    if ( acharset_registry )
      *acharset_registry = registry;

    return error;
  }


  /* documentation is in ftbdf.h */

  FT_EXPORT_DEF( FT_Error )
  FT_Get_BDF_Property( FT_Face           face,
                       const char*       prop_name,
                       BDF_PropertyRec  *aproperty )
  {
    FT_Error  error;


    error = FT_ERR( Invalid_Argument );

    aproperty->type = BDF_PROPERTY_TYPE_NONE;

    if ( face )
    {
      FT_Service_BDF  service;


      FT_FACE_FIND_SERVICE( face, service, BDF );

      if ( service && service->get_property )
        error = service->get_property( face, prop_name, aproperty );
    }

    return  error;
  }


/* END */
