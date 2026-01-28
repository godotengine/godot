/****************************************************************************
 *
 * cidriver.c
 *
 *   CID driver interface (body).
 *
 * Copyright (C) 1996-2025 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#include "cidriver.h"
#include "cidgload.h"
#include <freetype/internal/ftdebug.h>
#include <freetype/internal/ftpsprop.h>

#include "ciderrs.h"

#include <freetype/internal/services/svpostnm.h>
#include <freetype/internal/services/svfntfmt.h>
#include <freetype/internal/services/svpsinfo.h>
#include <freetype/internal/services/svcid.h>
#include <freetype/internal/services/svprop.h>
#include <freetype/ftdriver.h>

#include <freetype/internal/psaux.h>


  /**************************************************************************
   *
   * The macro FT_COMPONENT is used in trace mode.  It is an implicit
   * parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log
   * messages during execution.
   */
#undef  FT_COMPONENT
#define FT_COMPONENT  ciddriver


  /*
   * POSTSCRIPT NAME SERVICE
   *
   */

  FT_CALLBACK_DEF( const char* )
  cid_get_postscript_name( FT_Face  face )    /* CID_Face */
  {
    CID_Face     cidface = (CID_Face)face;
    const char*  result  = cidface->cid.cid_font_name;


    if ( result && result[0] == '/' )
      result++;

    return result;
  }


  static const FT_Service_PsFontNameRec  cid_service_ps_name =
  {
    (FT_PsName_GetFunc)cid_get_postscript_name    /* get_ps_font_name */
  };


  /*
   * POSTSCRIPT INFO SERVICE
   *
   */

  FT_CALLBACK_DEF( FT_Error )
  cid_ps_get_font_info( FT_Face          face,        /* CID_Face */
                        PS_FontInfoRec*  afont_info )
  {
    *afont_info = ( (CID_Face)face )->cid.font_info;

    return FT_Err_Ok;
  }


  FT_CALLBACK_DEF( FT_Error )
  cid_ps_get_font_extra( FT_Face           face,         /* CID_Face */
                         PS_FontExtraRec*  afont_extra )
  {
    *afont_extra = ( (CID_Face)face )->font_extra;

    return FT_Err_Ok;
  }


  static const FT_Service_PsInfoRec  cid_service_ps_info =
  {
    cid_ps_get_font_info,   /* PS_GetFontInfoFunc    ps_get_font_info    */
    cid_ps_get_font_extra,  /* PS_GetFontExtraFunc   ps_get_font_extra   */
    /* unsupported with CID fonts */
    NULL,                   /* PS_HasGlyphNamesFunc  ps_has_glyph_names  */
    /* unsupported                */
    NULL,                   /* PS_GetFontPrivateFunc ps_get_font_private */
    /* not implemented            */
    NULL                    /* PS_GetFontValueFunc   ps_get_font_value   */
  };


  /*
   * CID INFO SERVICE
   *
   */
  FT_CALLBACK_DEF( FT_Error )
  cid_get_ros( FT_Face       face,        /* CID_Face */
               const char*  *registry,
               const char*  *ordering,
               FT_Int       *supplement )
  {
    CID_Face      cidface = (CID_Face)face;
    CID_FaceInfo  cid     = &cidface->cid;


    if ( registry )
      *registry = cid->registry;

    if ( ordering )
      *ordering = cid->ordering;

    if ( supplement )
      *supplement = cid->supplement;

    return FT_Err_Ok;
  }


  FT_CALLBACK_DEF( FT_Error )
  cid_get_is_cid( FT_Face   face,    /* CID_Face */
                  FT_Bool  *is_cid )
  {
    FT_Error  error = FT_Err_Ok;
    FT_UNUSED( face );


    /*
     * XXX: If the ROS is Adobe-Identity-H or -V,
     * the font has no reliable information about
     * its glyph collection.  Should we not set
     * *is_cid in such cases?
     */
    if ( is_cid )
      *is_cid = 1;

    return error;
  }


  FT_CALLBACK_DEF( FT_Error )
  cid_get_cid_from_glyph_index( FT_Face   face,        /* CID_Face */
                                FT_UInt   glyph_index,
                                FT_UInt  *cid )
  {
    FT_Error  error   = FT_Err_Ok;
    CID_Face  cidface = (CID_Face)face;


    /*
     * Currently, FreeType does not support incrementally-defined, CID-keyed
     * fonts that store the glyph description data in a `/GlyphDirectory`
     * array or dictionary.  Fonts loaded by the incremental loading feature
     * are thus not handled here.
     */
    error = cid_compute_fd_and_offsets( cidface, glyph_index,
                                        NULL, NULL, NULL );
    if ( error )
      *cid = 0;
    else
      *cid = glyph_index;

    return error;
  }


  static const FT_Service_CIDRec  cid_service_cid_info =
  {
    cid_get_ros,
      /* FT_CID_GetRegistryOrderingSupplementFunc get_ros                  */
    cid_get_is_cid,
      /* FT_CID_GetIsInternallyCIDKeyedFunc       get_is_cid               */
    cid_get_cid_from_glyph_index
      /* FT_CID_GetCIDFromGlyphIndexFunc          get_cid_from_glyph_index */
  };


  /*
   * PROPERTY SERVICE
   *
   */

  FT_DEFINE_SERVICE_PROPERTIESREC(
    cid_service_properties,

    ps_property_set,  /* FT_Properties_SetFunc set_property */
    ps_property_get   /* FT_Properties_GetFunc get_property */
  )

  /*
   * SERVICE LIST
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
