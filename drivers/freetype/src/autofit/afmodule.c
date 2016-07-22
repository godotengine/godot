/***************************************************************************/
/*                                                                         */
/*  afmodule.c                                                             */
/*                                                                         */
/*    Auto-fitter module implementation (body).                            */
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


#include "afglobal.h"
#include "afmodule.h"
#include "afloader.h"
#include "aferrors.h"
#include "afpic.h"

#ifdef FT_DEBUG_AUTOFIT

#ifndef FT_MAKE_OPTION_SINGLE_OBJECT

#ifdef __cplusplus
  extern "C" {
#endif
  extern void
  af_glyph_hints_dump_segments( AF_GlyphHints  hints,
                                FT_Bool        to_stdout );
  extern void
  af_glyph_hints_dump_points( AF_GlyphHints  hints,
                              FT_Bool        to_stdout );
  extern void
  af_glyph_hints_dump_edges( AF_GlyphHints  hints,
                             FT_Bool        to_stdout );
#ifdef __cplusplus
  }
#endif

#endif

  int  _af_debug_disable_horz_hints;
  int  _af_debug_disable_vert_hints;
  int  _af_debug_disable_blue_hints;

  /* we use a global object instead of a local one for debugging */
  AF_GlyphHintsRec  _af_debug_hints_rec[1];

  void*  _af_debug_hints = _af_debug_hints_rec;
#endif

#include FT_INTERNAL_OBJECTS_H
#include FT_INTERNAL_DEBUG_H
#include FT_AUTOHINTER_H
#include FT_SERVICE_PROPERTIES_H


  /*************************************************************************/
  /*                                                                       */
  /* The macro FT_COMPONENT is used in trace mode.  It is an implicit      */
  /* parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log  */
  /* messages during execution.                                            */
  /*                                                                       */
#undef  FT_COMPONENT
#define FT_COMPONENT  trace_afmodule


  static FT_Error
  af_property_get_face_globals( FT_Face          face,
                                AF_FaceGlobals*  aglobals,
                                AF_Module        module )
  {
    FT_Error        error = FT_Err_Ok;
    AF_FaceGlobals  globals;


    if ( !face )
      return FT_THROW( Invalid_Face_Handle );

    globals = (AF_FaceGlobals)face->autohint.data;
    if ( !globals )
    {
      /* trigger computation of the global style data */
      /* in case it hasn't been done yet              */
      error = af_face_globals_new( face, &globals, module );
      if ( !error )
      {
        face->autohint.data =
          (FT_Pointer)globals;
        face->autohint.finalizer =
          (FT_Generic_Finalizer)af_face_globals_free;
      }
    }

    if ( !error )
      *aglobals = globals;

    return error;
  }


  static FT_Error
  af_property_set( FT_Module    ft_module,
                   const char*  property_name,
                   const void*  value )
  {
    FT_Error   error  = FT_Err_Ok;
    AF_Module  module = (AF_Module)ft_module;


    if ( !ft_strcmp( property_name, "fallback-script" ) )
    {
      FT_UInt*  fallback_script = (FT_UInt*)value;

      FT_UInt  ss;


      /* We translate the fallback script to a fallback style that uses */
      /* `fallback-script' as its script and `AF_COVERAGE_NONE' as its  */
      /* coverage value.                                                */
      for ( ss = 0; AF_STYLE_CLASSES_GET[ss]; ss++ )
      {
        AF_StyleClass  style_class = AF_STYLE_CLASSES_GET[ss];


        if ( (FT_UInt)style_class->script == *fallback_script &&
             style_class->coverage == AF_COVERAGE_DEFAULT     )
        {
          module->fallback_style = ss;
          break;
        }
      }

      if ( !AF_STYLE_CLASSES_GET[ss] )
      {
        FT_TRACE0(( "af_property_set: Invalid value %d for property `%s'\n",
                    fallback_script, property_name ));
        return FT_THROW( Invalid_Argument );
      }

      return error;
    }
    else if ( !ft_strcmp( property_name, "default-script" ) )
    {
      FT_UInt*  default_script = (FT_UInt*)value;


      module->default_script = *default_script;

      return error;
    }
    else if ( !ft_strcmp( property_name, "increase-x-height" ) )
    {
      FT_Prop_IncreaseXHeight*  prop = (FT_Prop_IncreaseXHeight*)value;
      AF_FaceGlobals            globals;


      error = af_property_get_face_globals( prop->face, &globals, module );
      if ( !error )
        globals->increase_x_height = prop->limit;

      return error;
    }
#ifdef AF_CONFIG_OPTION_USE_WARPER
    else if ( !ft_strcmp( property_name, "warping" ) )
    {
      FT_Bool*  warping = (FT_Bool*)value;


      module->warping = *warping;

      return error;
    }
#endif /* AF_CONFIG_OPTION_USE_WARPER */
    else if ( !ft_strcmp( property_name, "darkening-parameters" ) )
    {
      FT_Int*  darken_params = (FT_Int*)value;

      FT_Int  x1 = darken_params[0];
      FT_Int  y1 = darken_params[1];
      FT_Int  x2 = darken_params[2];
      FT_Int  y2 = darken_params[3];
      FT_Int  x3 = darken_params[4];
      FT_Int  y3 = darken_params[5];
      FT_Int  x4 = darken_params[6];
      FT_Int  y4 = darken_params[7];


      if ( x1 < 0   || x2 < 0   || x3 < 0   || x4 < 0   ||
           y1 < 0   || y2 < 0   || y3 < 0   || y4 < 0   ||
           x1 > x2  || x2 > x3  || x3 > x4              ||
           y1 > 500 || y2 > 500 || y3 > 500 || y4 > 500 )
        return FT_THROW( Invalid_Argument );

      module->darken_params[0] = x1;
      module->darken_params[1] = y1;
      module->darken_params[2] = x2;
      module->darken_params[3] = y2;
      module->darken_params[4] = x3;
      module->darken_params[5] = y3;
      module->darken_params[6] = x4;
      module->darken_params[7] = y4;

      return error;
    }
    else if ( !ft_strcmp( property_name, "no-stem-darkening" ) )
    {
      FT_Bool*  no_stem_darkening = (FT_Bool*)value;


      module->no_stem_darkening = *no_stem_darkening;

      return error;
    }

    FT_TRACE0(( "af_property_set: missing property `%s'\n",
                property_name ));
    return FT_THROW( Missing_Property );
  }


  static FT_Error
  af_property_get( FT_Module    ft_module,
                   const char*  property_name,
                   void*        value )
  {
    FT_Error   error          = FT_Err_Ok;
    AF_Module  module         = (AF_Module)ft_module;
    FT_UInt    fallback_style = module->fallback_style;
    FT_UInt    default_script = module->default_script;
#ifdef AF_CONFIG_OPTION_USE_WARPER
    FT_Bool    warping        = module->warping;
#endif


    if ( !ft_strcmp( property_name, "glyph-to-script-map" ) )
    {
      FT_Prop_GlyphToScriptMap*  prop = (FT_Prop_GlyphToScriptMap*)value;
      AF_FaceGlobals             globals;


      error = af_property_get_face_globals( prop->face, &globals, module );
      if ( !error )
        prop->map = globals->glyph_styles;

      return error;
    }
    else if ( !ft_strcmp( property_name, "fallback-script" ) )
    {
      FT_UInt*  val = (FT_UInt*)value;

      AF_StyleClass  style_class = AF_STYLE_CLASSES_GET[fallback_style];


      *val = style_class->script;

      return error;
    }
    else if ( !ft_strcmp( property_name, "default-script" ) )
    {
      FT_UInt*  val = (FT_UInt*)value;


      *val = default_script;

      return error;
    }
    else if ( !ft_strcmp( property_name, "increase-x-height" ) )
    {
      FT_Prop_IncreaseXHeight*  prop = (FT_Prop_IncreaseXHeight*)value;
      AF_FaceGlobals            globals;


      error = af_property_get_face_globals( prop->face, &globals, module );
      if ( !error )
        prop->limit = globals->increase_x_height;

      return error;
    }
#ifdef AF_CONFIG_OPTION_USE_WARPER
    else if ( !ft_strcmp( property_name, "warping" ) )
    {
      FT_Bool*  val = (FT_Bool*)value;


      *val = warping;

      return error;
    }
#endif /* AF_CONFIG_OPTION_USE_WARPER */
    else if ( !ft_strcmp( property_name, "darkening-parameters" ) )
    {
      FT_Int*  darken_params = module->darken_params;
      FT_Int*  val           = (FT_Int*)value;


      val[0] = darken_params[0];
      val[1] = darken_params[1];
      val[2] = darken_params[2];
      val[3] = darken_params[3];
      val[4] = darken_params[4];
      val[5] = darken_params[5];
      val[6] = darken_params[6];
      val[7] = darken_params[7];

      return error;
    }
    else if ( !ft_strcmp( property_name, "no-stem-darkening" ) )
    {
      FT_Bool   no_stem_darkening = module->no_stem_darkening;
      FT_Bool*  val               = (FT_Bool*)value;


      *val = no_stem_darkening;

      return error;
    }

    FT_TRACE0(( "af_property_get: missing property `%s'\n",
                property_name ));
    return FT_THROW( Missing_Property );
  }


  FT_DEFINE_SERVICE_PROPERTIESREC(
    af_service_properties,
    (FT_Properties_SetFunc)af_property_set,        /* set_property */
    (FT_Properties_GetFunc)af_property_get )       /* get_property */


  FT_DEFINE_SERVICEDESCREC1(
    af_services,
    FT_SERVICE_ID_PROPERTIES, &AF_SERVICE_PROPERTIES_GET )


  FT_CALLBACK_DEF( FT_Module_Interface )
  af_get_interface( FT_Module    module,
                    const char*  module_interface )
  {
    /* AF_SERVICES_GET dereferences `library' in PIC mode */
#ifdef FT_CONFIG_OPTION_PIC
    FT_Library  library;


    if ( !module )
      return NULL;
    library = module->library;
    if ( !library )
      return NULL;
#else
    FT_UNUSED( module );
#endif

    return ft_service_list_lookup( AF_SERVICES_GET, module_interface );
  }


  FT_CALLBACK_DEF( FT_Error )
  af_autofitter_init( FT_Module  ft_module )      /* AF_Module */
  {
    AF_Module  module = (AF_Module)ft_module;


    module->fallback_style    = AF_STYLE_FALLBACK;
    module->default_script    = AF_SCRIPT_DEFAULT;
#ifdef AF_CONFIG_OPTION_USE_WARPER
    module->warping           = 0;
#endif
    module->no_stem_darkening = TRUE;

    module->darken_params[0]  = CFF_CONFIG_OPTION_DARKENING_PARAMETER_X1;
    module->darken_params[1]  = CFF_CONFIG_OPTION_DARKENING_PARAMETER_Y1;
    module->darken_params[2]  = CFF_CONFIG_OPTION_DARKENING_PARAMETER_X2;
    module->darken_params[3]  = CFF_CONFIG_OPTION_DARKENING_PARAMETER_Y2;
    module->darken_params[4]  = CFF_CONFIG_OPTION_DARKENING_PARAMETER_X3;
    module->darken_params[5]  = CFF_CONFIG_OPTION_DARKENING_PARAMETER_Y3;
    module->darken_params[6]  = CFF_CONFIG_OPTION_DARKENING_PARAMETER_X4;
    module->darken_params[7]  = CFF_CONFIG_OPTION_DARKENING_PARAMETER_Y4;

    return FT_Err_Ok;
  }


  FT_CALLBACK_DEF( void )
  af_autofitter_done( FT_Module  ft_module )      /* AF_Module */
  {
    FT_UNUSED( ft_module );

#ifdef FT_DEBUG_AUTOFIT
    if ( _af_debug_hints_rec->memory )
      af_glyph_hints_done( _af_debug_hints_rec );
#endif
  }


  FT_CALLBACK_DEF( FT_Error )
  af_autofitter_load_glyph( AF_Module     module,
                            FT_GlyphSlot  slot,
                            FT_Size       size,
                            FT_UInt       glyph_index,
                            FT_Int32      load_flags )
  {
    FT_Error   error  = FT_Err_Ok;
    FT_Memory  memory = module->root.library->memory;

#ifdef FT_DEBUG_AUTOFIT

    /* in debug mode, we use a global object that survives this routine */

    AF_GlyphHints  hints = _af_debug_hints_rec;
    AF_LoaderRec   loader[1];

    FT_UNUSED( size );


    if ( hints->memory )
      af_glyph_hints_done( hints );

    af_glyph_hints_init( hints, memory );
    af_loader_init( loader, hints );

    error = af_loader_load_glyph( loader, module, slot->face,
                                  glyph_index, load_flags );

    af_glyph_hints_dump_points( hints, 0 );
    af_glyph_hints_dump_segments( hints, 0 );
    af_glyph_hints_dump_edges( hints, 0 );

    af_loader_done( loader );

    return error;

#else /* !FT_DEBUG_AUTOFIT */

    AF_GlyphHintsRec  hints[1];
    AF_LoaderRec      loader[1];

    FT_UNUSED( size );


    af_glyph_hints_init( hints, memory );
    af_loader_init( loader, hints );

    error = af_loader_load_glyph( loader, module, slot->face,
                                  glyph_index, load_flags );

    af_loader_done( loader );
    af_glyph_hints_done( hints );

    return error;

#endif /* !FT_DEBUG_AUTOFIT */
  }


  FT_DEFINE_AUTOHINTER_INTERFACE(
    af_autofitter_interface,
    NULL,                                                    /* reset_face */
    NULL,                                              /* get_global_hints */
    NULL,                                             /* done_global_hints */
    (FT_AutoHinter_GlyphLoadFunc)af_autofitter_load_glyph )  /* load_glyph */


  FT_DEFINE_MODULE(
    autofit_module_class,

    FT_MODULE_HINTER,
    sizeof ( AF_ModuleRec ),

    "autofitter",
    0x10000L,   /* version 1.0 of the autofitter  */
    0x20000L,   /* requires FreeType 2.0 or above */

    (const void*)&AF_INTERFACE_GET,

    (FT_Module_Constructor)af_autofitter_init,
    (FT_Module_Destructor) af_autofitter_done,
    (FT_Module_Requester)  af_get_interface )


/* END */
