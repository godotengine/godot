/****************************************************************************
 *
 * ftbdf.c
 *
 *   FreeType API for accessing BDF-specific strings (body).
 *
 * Copyright (C) 2002-2022 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#include <freetype/internal/ftdebug.h>

#include <freetype/internal/ftobjs.h>
#include <freetype/internal/services/svbdf.h>


  /* documentation is in ftbdf.h */

  FT_EXPORT_DEF( FT_Error )
  FT_Get_BDF_Charset_ID( FT_Face       face,
                         const char*  *acharset_encoding,
                         const char*  *acharset_registry )
  {
    FT_Error     error;
    const char*  encoding = NULL;
    const char*  registry = NULL;

    FT_Service_BDF  service;


    if ( !face )
      return FT_THROW( Invalid_Face_Handle );

    FT_FACE_FIND_SERVICE( face, service, BDF );

    if ( service && service->get_charset_id )
      error = service->get_charset_id( face, &encoding, &registry );
    else
      error = FT_THROW( Invalid_Argument );

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

    FT_Service_BDF  service;


    if ( !face )
      return FT_THROW( Invalid_Face_Handle );

    if ( !aproperty )
      return FT_THROW( Invalid_Argument );

    aproperty->type = BDF_PROPERTY_TYPE_NONE;

    FT_FACE_FIND_SERVICE( face, service, BDF );

    if ( service && service->get_property )
      error = service->get_property( face, prop_name, aproperty );
    else
      error = FT_THROW( Invalid_Argument );

    return error;
  }


/* END */
