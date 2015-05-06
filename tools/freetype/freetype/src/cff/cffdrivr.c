/***************************************************************************/
/*                                                                         */
/*  cffdrivr.c                                                             */
/*                                                                         */
/*    OpenType font driver implementation (body).                          */
/*                                                                         */
/*  Copyright 1996-2013 by                                                 */
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
#include FT_FREETYPE_H
#include FT_INTERNAL_DEBUG_H
#include FT_INTERNAL_STREAM_H
#include FT_INTERNAL_SFNT_H
#include FT_SERVICE_CID_H
#include FT_SERVICE_POSTSCRIPT_INFO_H
#include FT_SERVICE_POSTSCRIPT_NAME_H
#include FT_SERVICE_TT_CMAP_H

#include "cffdrivr.h"
#include "cffgload.h"
#include "cffload.h"
#include "cffcmap.h"
#include "cffparse.h"

#include "cfferrs.h"
#include "cffpic.h"

#include FT_SERVICE_XFREE86_NAME_H
#include FT_SERVICE_GLYPH_DICT_H
#include FT_SERVICE_PROPERTIES_H
#include FT_CFF_DRIVER_H


  /*************************************************************************/
  /*                                                                       */
  /* The macro FT_COMPONENT is used in trace mode.  It is an implicit      */
  /* parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log  */
  /* messages during execution.                                            */
  /*                                                                       */
#undef  FT_COMPONENT
#define FT_COMPONENT  trace_cffdriver


  /*************************************************************************/
  /*************************************************************************/
  /*************************************************************************/
  /****                                                                 ****/
  /****                                                                 ****/
  /****                          F A C E S                              ****/
  /****                                                                 ****/
  /****                                                                 ****/
  /*************************************************************************/
  /*************************************************************************/
  /*************************************************************************/


#undef  PAIR_TAG
#define PAIR_TAG( left, right )  ( ( (FT_ULong)left << 16 ) | \
                                     (FT_ULong)right        )


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    cff_get_kerning                                                    */
  /*                                                                       */
  /* <Description>                                                         */
  /*    A driver method used to return the kerning vector between two      */
  /*    glyphs of the same face.                                           */
  /*                                                                       */
  /* <Input>                                                               */
  /*    face        :: A handle to the source face object.                 */
  /*                                                                       */
  /*    left_glyph  :: The index of the left glyph in the kern pair.       */
  /*                                                                       */
  /*    right_glyph :: The index of the right glyph in the kern pair.      */
  /*                                                                       */
  /* <Output>                                                              */
  /*    kerning     :: The kerning vector.  This is in font units for      */
  /*                   scalable formats, and in pixels for fixed-sizes     */
  /*                   formats.                                            */
  /*                                                                       */
  /* <Return>                                                              */
  /*    FreeType error code.  0 means success.                             */
  /*                                                                       */
  /* <Note>                                                                */
  /*    Only horizontal layouts (left-to-right & right-to-left) are        */
  /*    supported by this function.  Other layouts, or more sophisticated  */
  /*    kernings, are out of scope of this method (the basic driver        */
  /*    interface is meant to be simple).                                  */
  /*                                                                       */
  /*    They can be implemented by format-specific interfaces.             */
  /*                                                                       */
  FT_CALLBACK_DEF( FT_Error )
  cff_get_kerning( FT_Face     ttface,          /* TT_Face */
                   FT_UInt     left_glyph,
                   FT_UInt     right_glyph,
                   FT_Vector*  kerning )
  {
    TT_Face       face = (TT_Face)ttface;
    SFNT_Service  sfnt = (SFNT_Service)face->sfnt;


    kerning->x = 0;
    kerning->y = 0;

    if ( sfnt )
      kerning->x = sfnt->get_kerning( face, left_glyph, right_glyph );

    return FT_Err_Ok;
  }


#undef PAIR_TAG


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    cff_glyph_load                                                     */
  /*                                                                       */
  /* <Description>                                                         */
  /*    A driver method used to load a glyph within a given glyph slot.    */
  /*                                                                       */
  /* <Input>                                                               */
  /*    slot        :: A handle to the target slot object where the glyph  */
  /*                   will be loaded.                                     */
  /*                                                                       */
  /*    size        :: A handle to the source face size at which the glyph */
  /*                   must be scaled, loaded, etc.                        */
  /*                                                                       */
  /*    glyph_index :: The index of the glyph in the font file.            */
  /*                                                                       */
  /*    load_flags  :: A flag indicating what to load for this glyph.  The */
  /*                   FT_LOAD_??? constants can be used to control the    */
  /*                   glyph loading process (e.g., whether the outline    */
  /*                   should be scaled, whether to load bitmaps or not,   */
  /*                   whether to hint the outline, etc).                  */
  /*                                                                       */
  /* <Return>                                                              */
  /*    FreeType error code.  0 means success.                             */
  /*                                                                       */
  FT_CALLBACK_DEF( FT_Error )
  cff_glyph_load( FT_GlyphSlot  cffslot,      /* CFF_GlyphSlot */
                  FT_Size       cffsize,      /* CFF_Size      */
                  FT_UInt       glyph_index,
                  FT_Int32      load_flags )
  {
    FT_Error       error;
    CFF_GlyphSlot  slot = (CFF_GlyphSlot)cffslot;
    CFF_Size       size = (CFF_Size)cffsize;


    if ( !slot )
      return FT_THROW( Invalid_Slot_Handle );

    /* check whether we want a scaled outline or bitmap */
    if ( !size )
      load_flags |= FT_LOAD_NO_SCALE | FT_LOAD_NO_HINTING;

    /* reset the size object if necessary */
    if ( load_flags & FT_LOAD_NO_SCALE )
      size = NULL;

    if ( size )
    {
      /* these two objects must have the same parent */
      if ( cffsize->face != cffslot->face )
        return FT_THROW( Invalid_Face_Handle );
    }

    /* now load the glyph outline if necessary */
    error = cff_slot_load( slot, size, glyph_index, load_flags );

    /* force drop-out mode to 2 - irrelevant now */
    /* slot->outline.dropout_mode = 2; */

    return error;
  }


  FT_CALLBACK_DEF( FT_Error )
  cff_get_advances( FT_Face    face,
                    FT_UInt    start,
                    FT_UInt    count,
                    FT_Int32   flags,
                    FT_Fixed*  advances )
  {
    FT_UInt       nn;
    FT_Error      error = FT_Err_Ok;
    FT_GlyphSlot  slot  = face->glyph;


    flags |= (FT_UInt32)FT_LOAD_ADVANCE_ONLY;

    for ( nn = 0; nn < count; nn++ )
    {
      error = cff_glyph_load( slot, face->size, start + nn, flags );
      if ( error )
        break;

      advances[nn] = ( flags & FT_LOAD_VERTICAL_LAYOUT )
                     ? slot->linearVertAdvance
                     : slot->linearHoriAdvance;
    }

    return error;
  }


  /*
   *  GLYPH DICT SERVICE
   *
   */

  static FT_Error
  cff_get_glyph_name( CFF_Face    face,
                      FT_UInt     glyph_index,
                      FT_Pointer  buffer,
                      FT_UInt     buffer_max )
  {
    CFF_Font    font   = (CFF_Font)face->extra.data;
    FT_String*  gname;
    FT_UShort   sid;
    FT_Error    error;


    if ( !font->psnames )
    {
      FT_ERROR(( "cff_get_glyph_name:"
                 " cannot get glyph name from CFF & CEF fonts\n"
                 "                   "
                 " without the `PSNames' module\n" ));
      error = FT_THROW( Missing_Module );
      goto Exit;
    }

    /* first, locate the sid in the charset table */
    sid = font->charset.sids[glyph_index];

    /* now, lookup the name itself */
    gname = cff_index_get_sid_string( font, sid );

    if ( gname )
      FT_STRCPYN( buffer, gname, buffer_max );

    error = FT_Err_Ok;

  Exit:
    return error;
  }


  static FT_UInt
  cff_get_name_index( CFF_Face    face,
                      FT_String*  glyph_name )
  {
    CFF_Font            cff;
    CFF_Charset         charset;
    FT_Service_PsCMaps  psnames;
    FT_String*          name;
    FT_UShort           sid;
    FT_UInt             i;


    cff     = (CFF_FontRec *)face->extra.data;
    charset = &cff->charset;

    FT_FACE_FIND_GLOBAL_SERVICE( face, psnames, POSTSCRIPT_CMAPS );
    if ( !psnames )
      return 0;

    for ( i = 0; i < cff->num_glyphs; i++ )
    {
      sid = charset->sids[i];

      if ( sid > 390 )
        name = cff_index_get_string( cff, sid - 391 );
      else
        name = (FT_String *)psnames->adobe_std_strings( sid );

      if ( !name )
        continue;

      if ( !ft_strcmp( glyph_name, name ) )
        return i;
    }

    return 0;
  }


  FT_DEFINE_SERVICE_GLYPHDICTREC(
    cff_service_glyph_dict,
    (FT_GlyphDict_GetNameFunc)  cff_get_glyph_name,
    (FT_GlyphDict_NameIndexFunc)cff_get_name_index
  )


  /*
   *  POSTSCRIPT INFO SERVICE
   *
   */

  static FT_Int
  cff_ps_has_glyph_names( FT_Face  face )
  {
    return ( face->face_flags & FT_FACE_FLAG_GLYPH_NAMES ) > 0;
  }


  static FT_Error
  cff_ps_get_font_info( CFF_Face         face,
                        PS_FontInfoRec*  afont_info )
  {
    CFF_Font  cff   = (CFF_Font)face->extra.data;
    FT_Error  error = FT_Err_Ok;


    if ( cff && cff->font_info == NULL )
    {
      CFF_FontRecDict  dict   = &cff->top_font.font_dict;
      PS_FontInfoRec  *font_info = NULL;
      FT_Memory        memory = face->root.memory;


      if ( FT_ALLOC( font_info, sizeof ( *font_info ) ) )
        goto Fail;

      font_info->version     = cff_index_get_sid_string( cff,
                                                         dict->version );
      font_info->notice      = cff_index_get_sid_string( cff,
                                                         dict->notice );
      font_info->full_name   = cff_index_get_sid_string( cff,
                                                         dict->full_name );
      font_info->family_name = cff_index_get_sid_string( cff,
                                                         dict->family_name );
      font_info->weight      = cff_index_get_sid_string( cff,
                                                         dict->weight );
      font_info->italic_angle        = dict->italic_angle;
      font_info->is_fixed_pitch      = dict->is_fixed_pitch;
      font_info->underline_position  = (FT_Short)dict->underline_position;
      font_info->underline_thickness = (FT_Short)dict->underline_thickness;

      cff->font_info = font_info;
    }

    if ( cff )
      *afont_info = *cff->font_info;

  Fail:
    return error;
  }


  FT_DEFINE_SERVICE_PSINFOREC(
    cff_service_ps_info,
    (PS_GetFontInfoFunc)   cff_ps_get_font_info,
    (PS_GetFontExtraFunc)  NULL,
    (PS_HasGlyphNamesFunc) cff_ps_has_glyph_names,
    (PS_GetFontPrivateFunc)NULL,        /* unsupported with CFF fonts */
    (PS_GetFontValueFunc)  NULL         /* not implemented            */
  )


  /*
   *  POSTSCRIPT NAME SERVICE
   *
   */

  static const char*
  cff_get_ps_name( CFF_Face  face )
  {
    CFF_Font  cff = (CFF_Font)face->extra.data;


    return (const char*)cff->font_name;
  }


  FT_DEFINE_SERVICE_PSFONTNAMEREC(
    cff_service_ps_name,
    (FT_PsName_GetFunc)cff_get_ps_name
  )


  /*
   * TT CMAP INFO
   *
   * If the charmap is a synthetic Unicode encoding cmap or
   * a Type 1 standard (or expert) encoding cmap, hide TT CMAP INFO
   * service defined in SFNT module.
   *
   * Otherwise call the service function in the sfnt module.
   *
   */
  static FT_Error
  cff_get_cmap_info( FT_CharMap    charmap,
                     TT_CMapInfo  *cmap_info )
  {
    FT_CMap   cmap  = FT_CMAP( charmap );
    FT_Error  error = FT_Err_Ok;

    FT_Face     face    = FT_CMAP_FACE( cmap );
    FT_Library  library = FT_FACE_LIBRARY( face );


    cmap_info->language = 0;
    cmap_info->format   = 0;

    if ( cmap->clazz != &CFF_CMAP_ENCODING_CLASS_REC_GET &&
         cmap->clazz != &CFF_CMAP_UNICODE_CLASS_REC_GET  )
    {
      FT_Module           sfnt    = FT_Get_Module( library, "sfnt" );
      FT_Service_TTCMaps  service =
        (FT_Service_TTCMaps)ft_module_get_service( sfnt,
                                                   FT_SERVICE_ID_TT_CMAP );


      if ( service && service->get_cmap_info )
        error = service->get_cmap_info( charmap, cmap_info );
    }

    return error;
  }


  FT_DEFINE_SERVICE_TTCMAPSREC(
    cff_service_get_cmap_info,
    (TT_CMap_Info_GetFunc)cff_get_cmap_info
  )


  /*
   *  CID INFO SERVICE
   *
   */
  static FT_Error
  cff_get_ros( CFF_Face      face,
               const char*  *registry,
               const char*  *ordering,
               FT_Int       *supplement )
  {
    FT_Error  error = FT_Err_Ok;
    CFF_Font  cff   = (CFF_Font)face->extra.data;


    if ( cff )
    {
      CFF_FontRecDict  dict = &cff->top_font.font_dict;


      if ( dict->cid_registry == 0xFFFFU )
      {
        error = FT_THROW( Invalid_Argument );
        goto Fail;
      }

      if ( registry )
      {
        if ( cff->registry == NULL )
          cff->registry = cff_index_get_sid_string( cff,
                                                    dict->cid_registry );
        *registry = cff->registry;
      }

      if ( ordering )
      {
        if ( cff->ordering == NULL )
          cff->ordering = cff_index_get_sid_string( cff,
                                                    dict->cid_ordering );
        *ordering = cff->ordering;
      }

      /*
       * XXX: According to Adobe TechNote #5176, the supplement in CFF
       *      can be a real number. We truncate it to fit public API
       *      since freetype-2.3.6.
       */
      if ( supplement )
      {
        if ( dict->cid_supplement < FT_INT_MIN ||
             dict->cid_supplement > FT_INT_MAX )
          FT_TRACE1(( "cff_get_ros: too large supplement %d is truncated\n",
                      dict->cid_supplement ));
        *supplement = (FT_Int)dict->cid_supplement;
      }
    }

  Fail:
    return error;
  }


  static FT_Error
  cff_get_is_cid( CFF_Face  face,
                  FT_Bool  *is_cid )
  {
    FT_Error  error = FT_Err_Ok;
    CFF_Font  cff   = (CFF_Font)face->extra.data;


    *is_cid = 0;

    if ( cff )
    {
      CFF_FontRecDict  dict = &cff->top_font.font_dict;


      if ( dict->cid_registry != 0xFFFFU )
        *is_cid = 1;
    }

    return error;
  }


  static FT_Error
  cff_get_cid_from_glyph_index( CFF_Face  face,
                                FT_UInt   glyph_index,
                                FT_UInt  *cid )
  {
    FT_Error  error = FT_Err_Ok;
    CFF_Font  cff;


    cff = (CFF_Font)face->extra.data;

    if ( cff )
    {
      FT_UInt          c;
      CFF_FontRecDict  dict = &cff->top_font.font_dict;


      if ( dict->cid_registry == 0xFFFFU )
      {
        error = FT_THROW( Invalid_Argument );
        goto Fail;
      }

      if ( glyph_index > cff->num_glyphs )
      {
        error = FT_THROW( Invalid_Argument );
        goto Fail;
      }

      c = cff->charset.sids[glyph_index];

      if ( cid )
        *cid = c;
    }

  Fail:
    return error;
  }


  FT_DEFINE_SERVICE_CIDREC(
    cff_service_cid_info,
    (FT_CID_GetRegistryOrderingSupplementFunc)cff_get_ros,
    (FT_CID_GetIsInternallyCIDKeyedFunc)      cff_get_is_cid,
    (FT_CID_GetCIDFromGlyphIndexFunc)         cff_get_cid_from_glyph_index
  )


  /*
   *  PROPERTY SERVICE
   *
   */
  static FT_Error
  cff_property_set( FT_Module    module,         /* CFF_Driver */
                    const char*  property_name,
                    const void*  value )
  {
    FT_Error    error  = FT_Err_Ok;
    CFF_Driver  driver = (CFF_Driver)module;


    if ( !ft_strcmp( property_name, "hinting-engine" ) )
    {
      FT_UInt*  hinting_engine = (FT_UInt*)value;


#ifndef CFF_CONFIG_OPTION_OLD_ENGINE
      if ( *hinting_engine != FT_CFF_HINTING_ADOBE )
        error = FT_ERR( Unimplemented_Feature );
      else
#endif
        driver->hinting_engine = *hinting_engine;

      return error;
    }
    else if ( !ft_strcmp( property_name, "no-stem-darkening" ) )
    {
      FT_Bool*  no_stem_darkening = (FT_Bool*)value;


      driver->no_stem_darkening = *no_stem_darkening;

      return error;
    }

    FT_TRACE0(( "cff_property_set: missing property `%s'\n",
                property_name ));
    return FT_THROW( Missing_Property );
  }


  static FT_Error
  cff_property_get( FT_Module    module,         /* CFF_Driver */
                    const char*  property_name,
                    const void*  value )
  {
    FT_Error    error  = FT_Err_Ok;
    CFF_Driver  driver = (CFF_Driver)module;

    FT_UInt  hinting_engine    = driver->hinting_engine;
    FT_Bool  no_stem_darkening = driver->no_stem_darkening;


    if ( !ft_strcmp( property_name, "hinting-engine" ) )
    {
      FT_UInt*  val = (FT_UInt*)value;


      *val = hinting_engine;

      return error;
    }
    else if ( !ft_strcmp( property_name, "no-stem-darkening" ) )
    {
      FT_Bool*  val = (FT_Bool*)value;


      *val = no_stem_darkening;

      return error;
    }

    FT_TRACE0(( "cff_property_get: missing property `%s'\n",
                property_name ));
    return FT_THROW( Missing_Property );
  }


  FT_DEFINE_SERVICE_PROPERTIESREC(
    cff_service_properties,
    (FT_Properties_SetFunc)cff_property_set,
    (FT_Properties_GetFunc)cff_property_get )


  /*************************************************************************/
  /*************************************************************************/
  /*************************************************************************/
  /****                                                                 ****/
  /****                                                                 ****/
  /****                D R I V E R  I N T E R F A C E                   ****/
  /****                                                                 ****/
  /****                                                                 ****/
  /*************************************************************************/
  /*************************************************************************/
  /*************************************************************************/

#ifndef FT_CONFIG_OPTION_NO_GLYPH_NAMES
  FT_DEFINE_SERVICEDESCREC7(
    cff_services,
    FT_SERVICE_ID_XF86_NAME,            FT_XF86_FORMAT_CFF,
    FT_SERVICE_ID_POSTSCRIPT_INFO,      &CFF_SERVICE_PS_INFO_GET,
    FT_SERVICE_ID_POSTSCRIPT_FONT_NAME, &CFF_SERVICE_PS_NAME_GET,
    FT_SERVICE_ID_GLYPH_DICT,           &CFF_SERVICE_GLYPH_DICT_GET,
    FT_SERVICE_ID_TT_CMAP,              &CFF_SERVICE_GET_CMAP_INFO_GET,
    FT_SERVICE_ID_CID,                  &CFF_SERVICE_CID_INFO_GET,
    FT_SERVICE_ID_PROPERTIES,           &CFF_SERVICE_PROPERTIES_GET
  )
#else
  FT_DEFINE_SERVICEDESCREC6(
    cff_services,
    FT_SERVICE_ID_XF86_NAME,            FT_XF86_FORMAT_CFF,
    FT_SERVICE_ID_POSTSCRIPT_INFO,      &CFF_SERVICE_PS_INFO_GET,
    FT_SERVICE_ID_POSTSCRIPT_FONT_NAME, &CFF_SERVICE_PS_NAME_GET,
    FT_SERVICE_ID_TT_CMAP,              &CFF_SERVICE_GET_CMAP_INFO_GET,
    FT_SERVICE_ID_CID,                  &CFF_SERVICE_CID_INFO_GET,
    FT_SERVICE_ID_PROPERTIES,           &CFF_SERVICE_PROPERTIES_GET
  )
#endif


  FT_CALLBACK_DEF( FT_Module_Interface )
  cff_get_interface( FT_Module    driver,       /* CFF_Driver */
                     const char*  module_interface )
  {
    FT_Library           library;
    FT_Module            sfnt;
    FT_Module_Interface  result;


    /* CFF_SERVICES_GET derefers `library' in PIC mode */
#ifdef FT_CONFIG_OPTION_PIC
    if ( !driver )
      return NULL;
    library = driver->library;
    if ( !library )
      return NULL;
#endif

    result = ft_service_list_lookup( CFF_SERVICES_GET, module_interface );
    if ( result != NULL )
      return result;

    /* `driver' is not yet evaluated in non-PIC mode */
#ifndef FT_CONFIG_OPTION_PIC
    if ( !driver )
      return NULL;
    library = driver->library;
    if ( !library )
      return NULL;
#endif

    /* we pass our request to the `sfnt' module */
    sfnt = FT_Get_Module( library, "sfnt" );

    return sfnt ? sfnt->clazz->get_interface( sfnt, module_interface ) : 0;
  }


  /* The FT_DriverInterface structure is defined in ftdriver.h. */

#ifdef TT_CONFIG_OPTION_EMBEDDED_BITMAPS
#define CFF_SIZE_SELECT cff_size_select
#else
#define CFF_SIZE_SELECT 0
#endif

  FT_DEFINE_DRIVER(
    cff_driver_class,

      FT_MODULE_FONT_DRIVER       |
      FT_MODULE_DRIVER_SCALABLE   |
      FT_MODULE_DRIVER_HAS_HINTER,

      sizeof ( CFF_DriverRec ),
      "cff",
      0x10000L,
      0x20000L,

      0,   /* module-specific interface */

      cff_driver_init,
      cff_driver_done,
      cff_get_interface,

    /* now the specific driver fields */
    sizeof ( TT_FaceRec ),
    sizeof ( CFF_SizeRec ),
    sizeof ( CFF_GlyphSlotRec ),

    cff_face_init,
    cff_face_done,
    cff_size_init,
    cff_size_done,
    cff_slot_init,
    cff_slot_done,

    cff_glyph_load,

    cff_get_kerning,
    0,                       /* FT_Face_AttachFunc */
    cff_get_advances,

    cff_size_request,

    CFF_SIZE_SELECT
  )


/* END */
