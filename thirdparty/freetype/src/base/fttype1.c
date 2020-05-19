/****************************************************************************
 *
 * fttype1.c
 *
 *   FreeType utility file for PS names support (body).
 *
 * Copyright (C) 2002-2020 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#include <ft2build.h>
#include FT_INTERNAL_DEBUG_H
#include FT_INTERNAL_OBJECTS_H
#include FT_INTERNAL_SERVICE_H
#include FT_SERVICE_POSTSCRIPT_INFO_H


  /* documentation is in t1tables.h */

  FT_EXPORT_DEF( FT_Error )
  FT_Get_PS_Font_Info( FT_Face          face,
                       PS_FontInfoRec*  afont_info )
  {
    FT_Error           error;
    FT_Service_PsInfo  service;


    if ( !face )
      return FT_THROW( Invalid_Face_Handle );

    if ( !afont_info )
      return FT_THROW( Invalid_Argument );

    FT_FACE_FIND_SERVICE( face, service, POSTSCRIPT_INFO );

    if ( service && service->ps_get_font_info )
      error = service->ps_get_font_info( face, afont_info );
    else
      error = FT_THROW( Invalid_Argument );

    return error;
  }


  /* documentation is in t1tables.h */

  FT_EXPORT_DEF( FT_Int )
  FT_Has_PS_Glyph_Names( FT_Face  face )
  {
    FT_Int             result = 0;
    FT_Service_PsInfo  service;


    if ( face )
    {
      FT_FACE_FIND_SERVICE( face, service, POSTSCRIPT_INFO );

      if ( service && service->ps_has_glyph_names )
        result = service->ps_has_glyph_names( face );
    }

    return result;
  }


  /* documentation is in t1tables.h */

  FT_EXPORT_DEF( FT_Error )
  FT_Get_PS_Font_Private( FT_Face         face,
                          PS_PrivateRec*  afont_private )
  {
    FT_Error           error;
    FT_Service_PsInfo  service;


    if ( !face )
      return FT_THROW( Invalid_Face_Handle );

    if ( !afont_private )
      return FT_THROW( Invalid_Argument );

    FT_FACE_FIND_SERVICE( face, service, POSTSCRIPT_INFO );

    if ( service && service->ps_get_font_private )
      error = service->ps_get_font_private( face, afont_private );
    else
      error = FT_THROW( Invalid_Argument );

    return error;
  }


  /* documentation is in t1tables.h */

  FT_EXPORT_DEF( FT_Long )
  FT_Get_PS_Font_Value( FT_Face       face,
                        PS_Dict_Keys  key,
                        FT_UInt       idx,
                        void         *value,
                        FT_Long       value_len )
  {
    FT_Int             result  = 0;
    FT_Service_PsInfo  service = NULL;


    if ( face )
    {
      FT_FACE_FIND_SERVICE( face, service, POSTSCRIPT_INFO );

      if ( service && service->ps_get_font_value )
        result = service->ps_get_font_value( face, key, idx,
                                             value, value_len );
    }

    return result;
  }


/* END */
