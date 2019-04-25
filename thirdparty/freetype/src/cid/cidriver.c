/***************************************************************************/
/*                                                                         */
/*  cidriver.c                                                             */
/*                                                                         */
/*    CID driver interface (body).                                         */
/*                                                                         */
/*  Copyright 1996-2018 by                                                 */
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
#include "cidriver.h"
#include "cidgload.h"
#include FT_INTERNAL_DEBUG_H
#include FT_INTERNAL_POSTSCRIPT_PROPS_H

#include "ciderrs.h"

#include FT_SERVICE_POSTSCRIPT_NAME_H
#include FT_SERVICE_FONT_FORMAT_H
#include FT_SERVICE_POSTSCRIPT_INFO_H
#include FT_SERVICE_CID_H
#include FT_SERVICE_PROPERTIES_H
#include FT_DRIVER_H

#include FT_INTERNAL_POSTSCRIPT_AUX_H


  /*************************************************************************/
  /*                                                                       */
  /* The macro FT_COMPONENT is used in trace mode.  It is an implicit      */
  /* parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log  */
  /* messages during execution.                                            */
  /*                                                                       */
#undef  FT_COMPONENT
#define FT_COMPONENT  trace_ciddriver


  /*
   *  POSTSCRIPT NAME SERVICE
   *
   */

  static const char*
  cid_get_postscript_name( CID_Face  face )
  {
    const char*  result = face->cid.cid_font_name;


    if ( result && result[0] == '/' )
      result++;

    return result;
  }


  static const FT_Service_PsFontNameRec  cid_service_ps_name =
  {
    (FT_PsName_GetFunc)cid_get_postscript_name    /* get_ps_font_name */
  };


  /*
   *  POSTSCRIPT INFO SERVICE
   *
   */

  static FT_Error
  cid_ps_get_font_info( FT_Face          face,
                        PS_FontInfoRec*  afont_info )
  {
    *afont_info = ((CID_Face)face)->cid.font_info;

    return FT_Err_Ok;
  }

  static FT_Error
  cid_ps_get_font_extra( FT_Face          face,
                        PS_FontExtraRec*  afont_extra )
  {
    *afont_extra = ((CID_Face)face)->font_extra;

    return FT_Err_Ok;
  }

  static const FT_Service_PsInfoRec  cid_service_ps_info =
  {
    (PS_GetFontInfoFunc)   cid_ps_get_font_info,   /* ps_get_font_info    */
    (PS_GetFontExtraFunc)  cid_ps_get_font_extra,  /* ps_get_font_extra   */
    /* unsupported with CID fonts */
    (PS_HasGlyphNamesFunc) NULL,                   /* ps_has_glyph_names  */
    /* unsupported                */
    (PS_GetFontPrivateFunc)NULL,                   /* ps_get_font_private */
    /* not implemented            */
    (PS_GetFontValueFunc)  NULL                    /* ps_get_font_value   */
  };


  /*
   *  CID INFO SERVICE
   *
   */
  static FT_Error
  cid_get_ros( CID_Face      face,
               const char*  *registry,
               const char*  *ordering,
               FT_Int       *supplement )
  {
    CID_FaceInfo  cid = &face->cid;


    if ( registry )
      *registry = cid->registry;

    if ( ordering )
      *ordering = cid->ordering;

    if ( supplement )
      *supplement = cid->supplement;

    return FT_Err_Ok;
  }


  static FT_Error
  cid_get_is_cid( CID_Face  face,
                  FT_Bool  *is_cid )
  {
    FT_Error  error = FT_Err_Ok;
    FT_UNUSED( face );


    if ( is_cid )
      *is_cid = 1; /* cid driver is only used for CID keyed fonts */

    return error;
  }


  static FT_Error
  cid_get_cid_from_glyph_index( CID_Face  face,
                                FT_UInt   glyph_index,
                                FT_UInt  *cid )
  {
    FT_Error  error = FT_Err_Ok;
    FT_UNUSED( face );


    if ( cid )
      *cid = glyph_index; /* identity mapping */

    return error;
  }


  static const FT_Service_CIDRec  cid_service_cid_info =
  {
    (FT_CID_GetRegistryOrderingSupplementFunc)
      cid_get_ros,                             /* get_ros                  */
    (FT_CID_GetIsInternallyCIDKeyedFunc)
      cid_get_is_cid,                          /* get_is_cid               */
    (FT_CID_GetCIDFromGlyphIndexFunc)
      cid_get_cid_from_glyph_index             /* get_cid_from_glyph_index */
  };


  /*
   *  PROPERTY SERVICE
   *
   */

  FT_DEFINE_SERVICE_PROPERTIESREC(
    cid_service_properties,

    (FT_Properties_SetFunc)ps_property_set,      /* set_property */
    (FT_Properties_GetFunc)ps_property_get )     /* get_property */


  /*
   *  SERVICE LIST
   *
   */

  static const FT_ServiceDescRec  cid_services[] =
  {
    { FT_SERVICE_ID_FONT_FORMAT,          FT_FONT_FORMAT_CID },
    { FT_SERVICE_ID_POSTSCRIPT_FONT_NAME, &cid_service_ps_name },
    { FT_SERVICE_ID_POSTSCRIPT_INFO,      &cid_service_ps_info },
    { FT_SERVICE_ID_CID,                  &cid_service_cid_info },
    { FT_SERVICE_ID_PROPERTIES,           &cid_service_properties },
    { NULL, NULL }
  };


  FT_CALLBACK_DEF( FT_Module_Interface )
  cid_get_interface( FT_Module    module,
                     const char*  cid_interface )
  {
    FT_UNUSED( module );

    return ft_service_list_lookup( cid_services, cid_interface );
  }



  FT_CALLBACK_TABLE_DEF
  const FT_Driver_ClassRec  t1cid_driver_class =
  {
    {
      FT_MODULE_FONT_DRIVER       |
      FT_MODULE_DRIVER_SCALABLE   |
      FT_MODULE_DRIVER_HAS_HINTER,
      sizeof ( PS_DriverRec ),

      "t1cid",   /* module name           */
      0x10000L,  /* version 1.0 of driver */
      0x20000L,  /* requires FreeType 2.0 */

      NULL,    /* module-specific interface */

      cid_driver_init,          /* FT_Module_Constructor  module_init   */
      cid_driver_done,          /* FT_Module_Destructor   module_done   */
      cid_get_interface         /* FT_Module_Requester    get_interface */
    },

    sizeof ( CID_FaceRec ),
    sizeof ( CID_SizeRec ),
    sizeof ( CID_GlyphSlotRec ),

    cid_face_init,              /* FT_Face_InitFunc  init_face */
    cid_face_done,              /* FT_Face_DoneFunc  done_face */
    cid_size_init,              /* FT_Size_InitFunc  init_size */
    cid_size_done,              /* FT_Size_DoneFunc  done_size */
    cid_slot_init,              /* FT_Slot_InitFunc  init_slot */
    cid_slot_done,              /* FT_Slot_DoneFunc  done_slot */

    cid_slot_load_glyph,        /* FT_Slot_LoadFunc  load_glyph */

    NULL,                       /* FT_Face_GetKerningFunc   get_kerning  */
    NULL,                       /* FT_Face_AttachFunc       attach_file  */
    NULL,                       /* FT_Face_GetAdvancesFunc  get_advances */

    cid_size_request,           /* FT_Size_RequestFunc  request_size */
    NULL                        /* FT_Size_SelectFunc   select_size  */
  };


/* END */
