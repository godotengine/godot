/***************************************************************************/
/*                                                                         */
/*  t42drivr.c                                                             */
/*                                                                         */
/*    High-level Type 42 driver interface (body).                          */
/*                                                                         */
/*  Copyright 2002-2004, 2006, 2007, 2009, 2011, 2013 by                   */
/*  Roberto Alameda.                                                       */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


  /*************************************************************************/
  /*                                                                       */
  /* This driver implements Type42 fonts as described in the               */
  /* Technical Note #5012 from Adobe, with these limitations:              */
  /*                                                                       */
  /* 1) CID Fonts are not currently supported.                             */
  /* 2) Incremental fonts making use of the GlyphDirectory keyword         */
  /*    will be loaded, but the rendering will be using the TrueType       */
  /*    tables.                                                            */
  /* 3) As for Type1 fonts, CDevProc is not supported.                     */
  /* 4) The Metrics dictionary is not supported.                           */
  /* 5) AFM metrics are not supported.                                     */
  /*                                                                       */
  /* In other words, this driver supports Type42 fonts derived from        */
  /* TrueType fonts in a non-CID manner, as done by usual conversion       */
  /* programs.                                                             */
  /*                                                                       */
  /*************************************************************************/


#include "t42drivr.h"
#include "t42objs.h"
#include "t42error.h"
#include FT_INTERNAL_DEBUG_H

#include FT_SERVICE_XFREE86_NAME_H
#include FT_SERVICE_GLYPH_DICT_H
#include FT_SERVICE_POSTSCRIPT_NAME_H
#include FT_SERVICE_POSTSCRIPT_INFO_H

#undef  FT_COMPONENT
#define FT_COMPONENT  trace_t42


  /*
   *
   *  GLYPH DICT SERVICE
   *
   */

  static FT_Error
  t42_get_glyph_name( T42_Face    face,
                      FT_UInt     glyph_index,
                      FT_Pointer  buffer,
                      FT_UInt     buffer_max )
  {
    FT_STRCPYN( buffer, face->type1.glyph_names[glyph_index], buffer_max );

    return FT_Err_Ok;
  }


  static FT_UInt
  t42_get_name_index( T42_Face    face,
                      FT_String*  glyph_name )
  {
    FT_Int  i;


    for ( i = 0; i < face->type1.num_glyphs; i++ )
    {
      FT_String*  gname = face->type1.glyph_names[i];


      if ( glyph_name[0] == gname[0] && !ft_strcmp( glyph_name, gname ) )
        return (FT_UInt)ft_atol( (const char *)face->type1.charstrings[i] );
    }

    return 0;
  }


  static const FT_Service_GlyphDictRec  t42_service_glyph_dict =
  {
    (FT_GlyphDict_GetNameFunc)  t42_get_glyph_name,
    (FT_GlyphDict_NameIndexFunc)t42_get_name_index
  };


  /*
   *
   *  POSTSCRIPT NAME SERVICE
   *
   */

  static const char*
  t42_get_ps_font_name( T42_Face  face )
  {
    return (const char*)face->type1.font_name;
  }


  static const FT_Service_PsFontNameRec  t42_service_ps_font_name =
  {
    (FT_PsName_GetFunc)t42_get_ps_font_name
  };


  /*
   *
   *  POSTSCRIPT INFO SERVICE
   *
   */

  static FT_Error
  t42_ps_get_font_info( FT_Face          face,
                        PS_FontInfoRec*  afont_info )
  {
    *afont_info = ((T42_Face)face)->type1.font_info;

    return FT_Err_Ok;
  }


  static FT_Error
  t42_ps_get_font_extra( FT_Face           face,
                         PS_FontExtraRec*  afont_extra )
  {
    *afont_extra = ((T42_Face)face)->type1.font_extra;

    return FT_Err_Ok;
  }


  static FT_Int
  t42_ps_has_glyph_names( FT_Face  face )
  {
    FT_UNUSED( face );

    return 1;
  }


  static FT_Error
  t42_ps_get_font_private( FT_Face         face,
                           PS_PrivateRec*  afont_private )
  {
    *afont_private = ((T42_Face)face)->type1.private_dict;

    return FT_Err_Ok;
  }


  static const FT_Service_PsInfoRec  t42_service_ps_info =
  {
    (PS_GetFontInfoFunc)   t42_ps_get_font_info,
    (PS_GetFontExtraFunc)  t42_ps_get_font_extra,
    (PS_HasGlyphNamesFunc) t42_ps_has_glyph_names,
    (PS_GetFontPrivateFunc)t42_ps_get_font_private,
    (PS_GetFontValueFunc)  NULL             /* not implemented */
  };


  /*
   *
   *  SERVICE LIST
   *
   */

  static const FT_ServiceDescRec  t42_services[] =
  {
    { FT_SERVICE_ID_GLYPH_DICT,           &t42_service_glyph_dict },
    { FT_SERVICE_ID_POSTSCRIPT_FONT_NAME, &t42_service_ps_font_name },
    { FT_SERVICE_ID_POSTSCRIPT_INFO,      &t42_service_ps_info },
    { FT_SERVICE_ID_XF86_NAME,            FT_XF86_FORMAT_TYPE_42 },
    { NULL, NULL }
  };


  FT_CALLBACK_DEF( FT_Module_Interface )
  T42_Get_Interface( FT_Module         module,
                     const FT_String*  t42_interface )
  {
    FT_UNUSED( module );

    return ft_service_list_lookup( t42_services, t42_interface );
  }


  const FT_Driver_ClassRec  t42_driver_class =
  {
    {
      FT_MODULE_FONT_DRIVER       |
      FT_MODULE_DRIVER_SCALABLE   |
#ifdef TT_USE_BYTECODE_INTERPRETER
      FT_MODULE_DRIVER_HAS_HINTER,
#else
      0,
#endif

      sizeof ( T42_DriverRec ),

      "type42",
      0x10000L,
      0x20000L,

      0,    /* format interface */

      T42_Driver_Init,
      T42_Driver_Done,
      T42_Get_Interface,
    },

    sizeof ( T42_FaceRec ),
    sizeof ( T42_SizeRec ),
    sizeof ( T42_GlyphSlotRec ),

    T42_Face_Init,
    T42_Face_Done,
    T42_Size_Init,
    T42_Size_Done,
    T42_GlyphSlot_Init,
    T42_GlyphSlot_Done,

    T42_GlyphSlot_Load,

    0,                 /* FT_Face_GetKerningFunc  */
    0,                 /* FT_Face_AttachFunc      */

    0,                 /* FT_Face_GetAdvancesFunc */
    T42_Size_Request,
    T42_Size_Select
  };


/* END */
