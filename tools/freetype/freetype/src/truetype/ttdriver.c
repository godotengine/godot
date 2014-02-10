/***************************************************************************/
/*                                                                         */
/*  ttdriver.c                                                             */
/*                                                                         */
/*    TrueType font driver implementation (body).                          */
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
#include FT_INTERNAL_DEBUG_H
#include FT_INTERNAL_STREAM_H
#include FT_INTERNAL_SFNT_H
#include FT_SERVICE_XFREE86_NAME_H

#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
#include FT_MULTIPLE_MASTERS_H
#include FT_SERVICE_MULTIPLE_MASTERS_H
#endif

#include FT_SERVICE_TRUETYPE_ENGINE_H
#include FT_SERVICE_TRUETYPE_GLYF_H
#include FT_SERVICE_PROPERTIES_H
#include FT_TRUETYPE_DRIVER_H

#include "ttdriver.h"
#include "ttgload.h"
#include "ttpload.h"

#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
#include "ttgxvar.h"
#endif

#include "tterrors.h"

#include "ttpic.h"

  /*************************************************************************/
  /*                                                                       */
  /* The macro FT_COMPONENT is used in trace mode.  It is an implicit      */
  /* parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log  */
  /* messages during execution.                                            */
  /*                                                                       */
#undef  FT_COMPONENT
#define FT_COMPONENT  trace_ttdriver


  /*
   *  PROPERTY SERVICE
   *
   */
  static FT_Error
  tt_property_set( FT_Module    module,         /* TT_Driver */
                   const char*  property_name,
                   const void*  value )
  {
    FT_Error   error  = FT_Err_Ok;
    TT_Driver  driver = (TT_Driver)module;


    if ( !ft_strcmp( property_name, "interpreter-version" ) )
    {
      FT_UInt*  interpreter_version = (FT_UInt*)value;


#ifndef TT_CONFIG_OPTION_SUBPIXEL_HINTING
      if ( *interpreter_version != TT_INTERPRETER_VERSION_35 )
        error = FT_ERR( Unimplemented_Feature );
      else
#endif
        driver->interpreter_version = *interpreter_version;

      return error;
    }

    FT_TRACE0(( "tt_property_set: missing property `%s'\n",
                property_name ));
    return FT_THROW( Missing_Property );
  }


  static FT_Error
  tt_property_get( FT_Module    module,         /* TT_Driver */
                   const char*  property_name,
                   const void*  value )
  {
    FT_Error   error  = FT_Err_Ok;
    TT_Driver  driver = (TT_Driver)module;

    FT_UInt  interpreter_version = driver->interpreter_version;


    if ( !ft_strcmp( property_name, "interpreter-version" ) )
    {
      FT_UInt*  val = (FT_UInt*)value;


      *val = interpreter_version;

      return error;
    }

    FT_TRACE0(( "tt_property_get: missing property `%s'\n",
                property_name ));
    return FT_THROW( Missing_Property );
  }


  FT_DEFINE_SERVICE_PROPERTIESREC(
    tt_service_properties,
    (FT_Properties_SetFunc)tt_property_set,
    (FT_Properties_GetFunc)tt_property_get )


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
  /*    tt_get_kerning                                                     */
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
  static FT_Error
  tt_get_kerning( FT_Face     ttface,          /* TT_Face */
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

    return 0;
  }


#undef PAIR_TAG


  static FT_Error
  tt_get_advances( FT_Face    ttface,
                   FT_UInt    start,
                   FT_UInt    count,
                   FT_Int32   flags,
                   FT_Fixed  *advances )
  {
    FT_UInt  nn;
    TT_Face  face  = (TT_Face) ttface;


    /* XXX: TODO: check for sbits */

    if ( flags & FT_LOAD_VERTICAL_LAYOUT )
    {
      for ( nn = 0; nn < count; nn++ )
      {
        FT_Short   tsb;
        FT_UShort  ah;


        TT_Get_VMetrics( face, start + nn, &tsb, &ah );
        advances[nn] = ah;
      }
    }
    else
    {
      for ( nn = 0; nn < count; nn++ )
      {
        FT_Short   lsb;
        FT_UShort  aw;


        TT_Get_HMetrics( face, start + nn, &lsb, &aw );
        advances[nn] = aw;
      }
    }

    return FT_Err_Ok;
  }

  /*************************************************************************/
  /*************************************************************************/
  /*************************************************************************/
  /****                                                                 ****/
  /****                                                                 ****/
  /****                           S I Z E S                             ****/
  /****                                                                 ****/
  /****                                                                 ****/
  /*************************************************************************/
  /*************************************************************************/
  /*************************************************************************/


#ifdef TT_CONFIG_OPTION_EMBEDDED_BITMAPS

  static FT_Error
  tt_size_select( FT_Size   size,
                  FT_ULong  strike_index )
  {
    TT_Face   ttface = (TT_Face)size->face;
    TT_Size   ttsize = (TT_Size)size;
    FT_Error  error  = FT_Err_Ok;


    ttsize->strike_index = strike_index;

    if ( FT_IS_SCALABLE( size->face ) )
    {
      /* use the scaled metrics, even when tt_size_reset fails */
      FT_Select_Metrics( size->face, strike_index );

      tt_size_reset( ttsize );
    }
    else
    {
      SFNT_Service      sfnt    = (SFNT_Service) ttface->sfnt;
      FT_Size_Metrics*  metrics = &size->metrics;


      error = sfnt->load_strike_metrics( ttface, strike_index, metrics );
      if ( error )
        ttsize->strike_index = 0xFFFFFFFFUL;
    }

    return error;
  }

#endif /* TT_CONFIG_OPTION_EMBEDDED_BITMAPS */


  static FT_Error
  tt_size_request( FT_Size          size,
                   FT_Size_Request  req )
  {
    TT_Size   ttsize = (TT_Size)size;
    FT_Error  error  = FT_Err_Ok;


#ifdef TT_CONFIG_OPTION_EMBEDDED_BITMAPS

    if ( FT_HAS_FIXED_SIZES( size->face ) )
    {
      TT_Face       ttface = (TT_Face)size->face;
      SFNT_Service  sfnt   = (SFNT_Service) ttface->sfnt;
      FT_ULong      strike_index;


      error = sfnt->set_sbit_strike( ttface, req, &strike_index );

      if ( error )
        ttsize->strike_index = 0xFFFFFFFFUL;
      else
        return tt_size_select( size, strike_index );
    }

#endif /* TT_CONFIG_OPTION_EMBEDDED_BITMAPS */

    FT_Request_Metrics( size->face, req );

    if ( FT_IS_SCALABLE( size->face ) )
    {
      error = tt_size_reset( ttsize );
      ttsize->root.metrics = ttsize->metrics;
    }

    return error;
  }


  /*************************************************************************/
  /*                                                                       */
  /* <Function>                                                            */
  /*    tt_glyph_load                                                      */
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
  /*                   FT_LOAD_XXX constants can be used to control the    */
  /*                   glyph loading process (e.g., whether the outline    */
  /*                   should be scaled, whether to load bitmaps or not,   */
  /*                   whether to hint the outline, etc).                  */
  /*                                                                       */
  /* <Return>                                                              */
  /*    FreeType error code.  0 means success.                             */
  /*                                                                       */
  static FT_Error
  tt_glyph_load( FT_GlyphSlot  ttslot,      /* TT_GlyphSlot */
                 FT_Size       ttsize,      /* TT_Size      */
                 FT_UInt       glyph_index,
                 FT_Int32      load_flags )
  {
    TT_GlyphSlot  slot = (TT_GlyphSlot)ttslot;
    TT_Size       size = (TT_Size)ttsize;
    FT_Face       face = ttslot->face;
    FT_Error      error;


    if ( !slot )
      return FT_THROW( Invalid_Slot_Handle );

    if ( !size )
      return FT_THROW( Invalid_Size_Handle );

    if ( !face )
      return FT_THROW( Invalid_Argument );

#ifdef FT_CONFIG_OPTION_INCREMENTAL
    if ( glyph_index >= (FT_UInt)face->num_glyphs &&
         !face->internal->incremental_interface   )
#else
    if ( glyph_index >= (FT_UInt)face->num_glyphs )
#endif
      return FT_THROW( Invalid_Argument );

    if ( load_flags & FT_LOAD_NO_HINTING )
    {
      /* both FT_LOAD_NO_HINTING and FT_LOAD_NO_AUTOHINT   */
      /* are necessary to disable hinting for tricky fonts */

      if ( FT_IS_TRICKY( face ) )
        load_flags &= ~FT_LOAD_NO_HINTING;

      if ( load_flags & FT_LOAD_NO_AUTOHINT )
        load_flags |= FT_LOAD_NO_HINTING;
    }

    if ( load_flags & ( FT_LOAD_NO_RECURSE | FT_LOAD_NO_SCALE ) )
    {
      load_flags |= FT_LOAD_NO_BITMAP | FT_LOAD_NO_SCALE;

      if ( !FT_IS_TRICKY( face ) )
        load_flags |= FT_LOAD_NO_HINTING;
    }

    /* now load the glyph outline if necessary */
    error = TT_Load_Glyph( size, slot, glyph_index, load_flags );

    /* force drop-out mode to 2 - irrelevant now */
    /* slot->outline.dropout_mode = 2; */

    return error;
  }


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

#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
  FT_DEFINE_SERVICE_MULTIMASTERSREC(
    tt_service_gx_multi_masters,
    (FT_Get_MM_Func)        NULL,
    (FT_Set_MM_Design_Func) NULL,
    (FT_Set_MM_Blend_Func)  TT_Set_MM_Blend,
    (FT_Get_MM_Var_Func)    TT_Get_MM_Var,
    (FT_Set_Var_Design_Func)TT_Set_Var_Design )
#endif

  static const FT_Service_TrueTypeEngineRec  tt_service_truetype_engine =
  {
#ifdef TT_USE_BYTECODE_INTERPRETER

#ifdef TT_CONFIG_OPTION_UNPATENTED_HINTING
    FT_TRUETYPE_ENGINE_TYPE_UNPATENTED
#else
    FT_TRUETYPE_ENGINE_TYPE_PATENTED
#endif

#else /* !TT_USE_BYTECODE_INTERPRETER */

    FT_TRUETYPE_ENGINE_TYPE_NONE

#endif /* TT_USE_BYTECODE_INTERPRETER */
  };

  FT_DEFINE_SERVICE_TTGLYFREC(
    tt_service_truetype_glyf,
    (TT_Glyf_GetLocationFunc)tt_face_get_location )

#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
  FT_DEFINE_SERVICEDESCREC5(
    tt_services,
    FT_SERVICE_ID_XF86_NAME,       FT_XF86_FORMAT_TRUETYPE,
    FT_SERVICE_ID_MULTI_MASTERS,   &TT_SERVICE_GX_MULTI_MASTERS_GET,
    FT_SERVICE_ID_TRUETYPE_ENGINE, &tt_service_truetype_engine,
    FT_SERVICE_ID_TT_GLYF,         &TT_SERVICE_TRUETYPE_GLYF_GET,
    FT_SERVICE_ID_PROPERTIES,      &TT_SERVICE_PROPERTIES_GET )
#else
  FT_DEFINE_SERVICEDESCREC4(
    tt_services,
    FT_SERVICE_ID_XF86_NAME,       FT_XF86_FORMAT_TRUETYPE,
    FT_SERVICE_ID_TRUETYPE_ENGINE, &tt_service_truetype_engine,
    FT_SERVICE_ID_TT_GLYF,         &TT_SERVICE_TRUETYPE_GLYF_GET,
    FT_SERVICE_ID_PROPERTIES,      &TT_SERVICE_PROPERTIES_GET )
#endif


  FT_CALLBACK_DEF( FT_Module_Interface )
  tt_get_interface( FT_Module    driver,    /* TT_Driver */
                    const char*  tt_interface )
  {
    FT_Library           library;
    FT_Module_Interface  result;
    FT_Module            sfntd;
    SFNT_Service         sfnt;


    /* TT_SERVICES_GET derefers `library' in PIC mode */
#ifdef FT_CONFIG_OPTION_PIC
    if ( !driver )
      return NULL;
    library = driver->library;
    if ( !library )
      return NULL;
#endif

    result = ft_service_list_lookup( TT_SERVICES_GET, tt_interface );
    if ( result != NULL )
      return result;

#ifndef FT_CONFIG_OPTION_PIC
    if ( !driver )
      return NULL;
    library = driver->library;
    if ( !library )
      return NULL;
#endif

    /* only return the default interface from the SFNT module */
    sfntd = FT_Get_Module( library, "sfnt" );
    if ( sfntd )
    {
      sfnt = (SFNT_Service)( sfntd->clazz->module_interface );
      if ( sfnt )
        return sfnt->get_interface( driver, tt_interface );
    }

    return 0;
  }


  /* The FT_DriverInterface structure is defined in ftdriver.h. */

#ifdef TT_USE_BYTECODE_INTERPRETER
#define TT_HINTER_FLAG  FT_MODULE_DRIVER_HAS_HINTER
#else
#define TT_HINTER_FLAG  0
#endif

#ifdef TT_CONFIG_OPTION_EMBEDDED_BITMAPS
#define TT_SIZE_SELECT  tt_size_select
#else
#define TT_SIZE_SELECT  0
#endif

  FT_DEFINE_DRIVER(
    tt_driver_class,

      FT_MODULE_FONT_DRIVER     |
      FT_MODULE_DRIVER_SCALABLE |
      TT_HINTER_FLAG,

      sizeof ( TT_DriverRec ),

      "truetype",      /* driver name                           */
      0x10000L,        /* driver version == 1.0                 */
      0x20000L,        /* driver requires FreeType 2.0 or above */

      (void*)0,        /* driver specific interface */

      tt_driver_init,
      tt_driver_done,
      tt_get_interface,

    sizeof ( TT_FaceRec ),
    sizeof ( TT_SizeRec ),
    sizeof ( FT_GlyphSlotRec ),

    tt_face_init,
    tt_face_done,
    tt_size_init,
    tt_size_done,
    tt_slot_init,
    0,                       /* FT_Slot_DoneFunc */

    tt_glyph_load,

    tt_get_kerning,
    0,                       /* FT_Face_AttachFunc */
    tt_get_advances,

    tt_size_request,
    TT_SIZE_SELECT
  )


/* END */
