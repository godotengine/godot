/****************************************************************************
 *
 * ttdriver.c
 *
 *   TrueType font driver implementation (body).
 *
 * Copyright (C) 1996-2024 by
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
#include <freetype/internal/ftstream.h>
#include <freetype/internal/sfnt.h>
#include <freetype/internal/services/svfntfmt.h>

#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
#include <freetype/ftmm.h>
#include <freetype/internal/services/svmm.h>
#include <freetype/internal/services/svmetric.h>
#endif

#include <freetype/internal/services/svtteng.h>
#include <freetype/internal/services/svttglyf.h>
#include <freetype/internal/services/svprop.h>
#include <freetype/ftdriver.h>

#include "ttdriver.h"
#include "ttgload.h"
#include "ttpload.h"

#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
#include "ttgxvar.h"
#endif

#include "tterrors.h"


  /**************************************************************************
   *
   * The macro FT_COMPONENT is used in trace mode.  It is an implicit
   * parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log
   * messages during execution.
   */
#undef  FT_COMPONENT
#define FT_COMPONENT  ttdriver


  /*
   * PROPERTY SERVICE
   *
   */
  FT_CALLBACK_DEF( FT_Error )
  tt_property_set( FT_Module    module,         /* TT_Driver */
                   const char*  property_name,
                   const void*  value,
                   FT_Bool      value_is_string )
  {
    FT_Error   error  = FT_Err_Ok;
    TT_Driver  driver = (TT_Driver)module;

#ifndef FT_CONFIG_OPTION_ENVIRONMENT_PROPERTIES
    FT_UNUSED( value_is_string );
#endif


    if ( !ft_strcmp( property_name, "interpreter-version" ) )
    {
      FT_UInt  interpreter_version;


#ifdef FT_CONFIG_OPTION_ENVIRONMENT_PROPERTIES
      if ( value_is_string )
      {
        const char*  s = (const char*)value;


        interpreter_version = (FT_UInt)ft_strtol( s, NULL, 10 );
      }
      else
#endif
      {
        FT_UInt*  iv = (FT_UInt*)value;


        interpreter_version = *iv;
      }

      switch ( interpreter_version )
      {
      case TT_INTERPRETER_VERSION_35:
        driver->interpreter_version = TT_INTERPRETER_VERSION_35;
        break;

      case TT_INTERPRETER_VERSION_38:
      case TT_INTERPRETER_VERSION_40:
#ifdef TT_SUPPORT_SUBPIXEL_HINTING_MINIMAL
        driver->interpreter_version = TT_INTERPRETER_VERSION_40;
      break;
#endif

      default:
        error = FT_ERR( Unimplemented_Feature );
      }

      return error;
    }

    FT_TRACE2(( "tt_property_set: missing property `%s'\n",
                property_name ));
    return FT_THROW( Missing_Property );
  }


  FT_CALLBACK_DEF( FT_Error )
  tt_property_get( FT_Module    module,         /* TT_Driver */
                   const char*  property_name,
                   void*        value )
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

    FT_TRACE2(( "tt_property_get: missing property `%s'\n",
                property_name ));
    return FT_THROW( Missing_Property );
  }


  FT_DEFINE_SERVICE_PROPERTIESREC(
    tt_service_properties,

    tt_property_set,  /* FT_Properties_SetFunc set_property */
    tt_property_get   /* FT_Properties_GetFunc get_property */
  )


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


  /**************************************************************************
   *
   * @Function:
   *   tt_get_kerning
   *
   * @Description:
   *   A driver method used to return the kerning vector between two
   *   glyphs of the same face.
   *
   * @Input:
   *   face ::
   *     A handle to the source face object.
   *
   *   left_glyph ::
   *     The index of the left glyph in the kern pair.
   *
   *   right_glyph ::
   *     The index of the right glyph in the kern pair.
   *
   * @Output:
   *   kerning ::
   *     The kerning vector.  This is in font units for
   *     scalable formats, and in pixels for fixed-sizes
   *     formats.
   *
   * @Return:
   *   FreeType error code.  0 means success.
   *
   * @Note:
   *   Only horizontal layouts (left-to-right & right-to-left) are
   *   supported by this function.  Other layouts, or more sophisticated
   *   kernings, are out of scope of this method (the basic driver
   *   interface is meant to be simple).
   *
   *   They can be implemented by format-specific interfaces.
   */
  FT_CALLBACK_DEF( FT_Error )
  tt_get_kerning( FT_Face     face,        /* TT_Face */
                  FT_UInt     left_glyph,
                  FT_UInt     right_glyph,
                  FT_Vector*  kerning )
  {
    TT_Face       ttface = (TT_Face)face;
    SFNT_Service  sfnt   = (SFNT_Service)ttface->sfnt;


    kerning->x = 0;
    kerning->y = 0;

    if ( sfnt )
    {
      /* Use 'kern' table if available since that can be faster; otherwise */
      /* use GPOS kerning pairs if available.                              */
      if ( ttface->kern_avail_bits != 0 )
        kerning->x = sfnt->get_kerning( ttface,
                                        left_glyph,
                                        right_glyph );
#ifdef TT_CONFIG_OPTION_GPOS_KERNING
      else if ( ttface->gpos_kerning_available )
        kerning->x = sfnt->get_gpos_kerning( ttface,
                                             left_glyph,
                                             right_glyph );
#endif
    }

    return 0;
  }


  FT_CALLBACK_DEF( FT_Error )
  tt_get_advances( FT_Face    face,      /* TT_Face */
                   FT_UInt    start,
                   FT_UInt    count,
                   FT_Int32   flags,
                   FT_Fixed  *advances )
  {
    FT_UInt  nn;
    TT_Face  ttface = (TT_Face)face;


    /* XXX: TODO: check for sbits */

    if ( flags & FT_LOAD_VERTICAL_LAYOUT )
    {
#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
      /* no fast retrieval for blended MM fonts without VVAR table */
      if ( ( FT_IS_NAMED_INSTANCE( face ) || FT_IS_VARIATION( face ) ) &&
           !( ttface->variation_support & TT_FACE_FLAG_VAR_VADVANCE )  )
        return FT_THROW( Unimplemented_Feature );
#endif

      for ( nn = 0; nn < count; nn++ )
      {
        FT_Short   tsb;
        FT_UShort  ah;


        /* since we don't need `tsb', we use zero for `yMax' parameter */
        TT_Get_VMetrics( ttface, start + nn, 0, &tsb, &ah );
        advances[nn] = ah;
      }
    }
    else
    {
#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
      /* no fast retrieval for blended MM fonts without HVAR table */
      if ( ( FT_IS_NAMED_INSTANCE( face ) || FT_IS_VARIATION( face ) ) &&
           !( ttface->variation_support & TT_FACE_FLAG_VAR_HADVANCE )  )
        return FT_THROW( Unimplemented_Feature );
#endif

      for ( nn = 0; nn < count; nn++ )
      {
        FT_Short   lsb;
        FT_UShort  aw;


        TT_Get_HMetrics( ttface, start + nn, &lsb, &aw );
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

  FT_CALLBACK_DEF( FT_Error )
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

      tt_size_reset( ttsize ); /* ignore return value */
    }
    else
    {
      SFNT_Service      sfnt         = (SFNT_Service)ttface->sfnt;
      FT_Size_Metrics*  size_metrics = &size->metrics;


      error = sfnt->load_strike_metrics( ttface,
                                         strike_index,
                                         size_metrics );
      if ( error )
        ttsize->strike_index = 0xFFFFFFFFUL;
    }

    return error;
  }

#endif /* TT_CONFIG_OPTION_EMBEDDED_BITMAPS */


  FT_CALLBACK_DEF( FT_Error )
  tt_size_request( FT_Size          size,
                   FT_Size_Request  req )
  {
    TT_Size   ttsize = (TT_Size)size;
    FT_Error  error  = FT_Err_Ok;


#ifdef TT_CONFIG_OPTION_EMBEDDED_BITMAPS

    if ( FT_HAS_FIXED_SIZES( size->face ) )
    {
      TT_Face       ttface = (TT_Face)size->face;
      SFNT_Service  sfnt   = (SFNT_Service)ttface->sfnt;
      FT_ULong      strike_index;


      error = sfnt->set_sbit_strike( ttface, req, &strike_index );

      if ( error )
        ttsize->strike_index = 0xFFFFFFFFUL;
      else
        return tt_size_select( size, strike_index );
    }

#endif /* TT_CONFIG_OPTION_EMBEDDED_BITMAPS */

    {
      FT_Error  err = FT_Request_Metrics( size->face, req );


      if ( err )
      {
        error = err;
        goto Exit;
      }
    }

    if ( FT_IS_SCALABLE( size->face ) )
    {
      error = tt_size_reset( ttsize );

#ifdef TT_USE_BYTECODE_INTERPRETER
      /* for the `MPS' bytecode instruction we need the point size */
      if ( !error )
      {
        FT_UInt  resolution =
                   ttsize->metrics->x_ppem > ttsize->metrics->y_ppem
                     ? req->horiResolution
                     : req->vertResolution;


        /* if we don't have a resolution value, assume 72dpi */
        if ( req->type == FT_SIZE_REQUEST_TYPE_SCALES ||
             !resolution                              )
          resolution = 72;

        ttsize->point_size = FT_MulDiv( ttsize->ttmetrics.ppem,
                                        64 * 72,
                                        resolution );
      }
#endif
    }

  Exit:
    return error;
  }


  /**************************************************************************
   *
   * @Function:
   *   tt_glyph_load
   *
   * @Description:
   *   A driver method used to load a glyph within a given glyph slot.
   *
   * @Input:
   *   slot ::
   *     A handle to the target slot object where the glyph
   *     will be loaded.
   *
   *   size ::
   *     A handle to the source face size at which the glyph
   *     must be scaled, loaded, etc.
   *
   *   glyph_index ::
   *     The index of the glyph in the font file.
   *
   *   load_flags ::
   *     A flag indicating what to load for this glyph.  The
   *     FT_LOAD_XXX constants can be used to control the
   *     glyph loading process (e.g., whether the outline
   *     should be scaled, whether to load bitmaps or not,
   *     whether to hint the outline, etc).
   *
   * @Return:
   *   FreeType error code.  0 means success.
   */
  FT_CALLBACK_DEF( FT_Error )
  tt_glyph_load( FT_GlyphSlot  slot,        /* TT_GlyphSlot */
                 FT_Size       size,        /* TT_Size      */
                 FT_UInt       glyph_index,
                 FT_Int32      load_flags )
  {
    TT_GlyphSlot  ttslot = (TT_GlyphSlot)slot;
    TT_Size       ttsize = (TT_Size)size;
    FT_Face       face   = ttslot->face;
    FT_Error      error;


    if ( !slot )
      return FT_THROW( Invalid_Slot_Handle );

    if ( !size )
      return FT_THROW( Invalid_Size_Handle );

    if ( !face )
      return FT_THROW( Invalid_Face_Handle );

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

    /* use hinted metrics only if we load a glyph with hinting */
    ttsize->metrics = ( load_flags & FT_LOAD_NO_HINTING )
                        ? &size->metrics
                        : &ttsize->hinted_metrics;

    /* now fill in the glyph slot with outline/bitmap/layered */
    error = TT_Load_Glyph( ttsize, ttslot, glyph_index, load_flags );

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

    NULL,                  /* FT_Get_MM_Func              get_mm                     */
    NULL,                  /* FT_Set_MM_Design_Func       set_mm_design              */
    TT_Set_MM_Blend,       /* FT_Set_MM_Blend_Func        set_mm_blend               */
    TT_Get_MM_Blend,       /* FT_Get_MM_Blend_Func        get_mm_blend               */
    TT_Get_MM_Var,         /* FT_Get_MM_Var_Func          get_mm_var                 */
    TT_Set_Var_Design,     /* FT_Set_Var_Design_Func      set_var_design             */
    TT_Get_Var_Design,     /* FT_Get_Var_Design_Func      get_var_design             */
    TT_Set_Named_Instance, /* FT_Set_Named_Instance_Func  set_named_instance         */
    TT_Get_Default_Named_Instance,
                    /* FT_Get_Default_Named_Instance_Func get_default_named_instance */
    NULL,                  /* FT_Set_MM_WeightVector_Func set_mm_weightvector        */
    NULL,                  /* FT_Get_MM_WeightVector_Func get_mm_weightvector        */

    tt_construct_ps_name,  /* FT_Construct_PS_Name_Func   construct_ps_name          */
    tt_var_load_delta_set_index_mapping,
                    /* FT_Var_Load_Delta_Set_Idx_Map_Func load_delta_set_idx_map     */
    tt_var_load_item_variation_store,
                    /* FT_Var_Load_Item_Var_Store_Func    load_item_variation_store  */
    tt_var_get_item_delta, /* FT_Var_Get_Item_Delta_Func  get_item_delta             */
    tt_var_done_item_variation_store,
                    /* FT_Var_Done_Item_Var_Store_Func    done_item_variation_store  */
    tt_var_done_delta_set_index_map,
                    /* FT_Var_Done_Delta_Set_Idx_Map_Func done_delta_set_index_map   */
    tt_get_var_blend,      /* FT_Get_Var_Blend_Func       get_var_blend              */
    tt_done_blend          /* FT_Done_Blend_Func          done_blend                 */
  )

  FT_DEFINE_SERVICE_METRICSVARIATIONSREC(
    tt_service_metrics_variations,

    tt_hadvance_adjust,   /* FT_HAdvance_Adjust_Func hadvance_adjust */
    NULL,                 /* FT_LSB_Adjust_Func      lsb_adjust      */
    NULL,                 /* FT_RSB_Adjust_Func      rsb_adjust      */

    tt_vadvance_adjust,   /* FT_VAdvance_Adjust_Func vadvance_adjust */
    NULL,                 /* FT_TSB_Adjust_Func      tsb_adjust      */
    NULL,                 /* FT_BSB_Adjust_Func      bsb_adjust      */
    NULL,                 /* FT_VOrg_Adjust_Func     vorg_adjust     */

    tt_apply_mvar,        /* FT_Metrics_Adjust_Func  metrics_adjust  */
    tt_size_reset_height  /* FT_Size_Reset_Func      size_reset      */
  )

#endif /* TT_CONFIG_OPTION_GX_VAR_SUPPORT */


  static const FT_Service_TrueTypeEngineRec  tt_service_truetype_engine =
  {
#ifdef TT_USE_BYTECODE_INTERPRETER

    FT_TRUETYPE_ENGINE_TYPE_PATENTED

#else /* !TT_USE_BYTECODE_INTERPRETER */

    FT_TRUETYPE_ENGINE_TYPE_NONE

#endif /* TT_USE_BYTECODE_INTERPRETER */
  };


  FT_DEFINE_SERVICE_TTGLYFREC(
    tt_service_truetype_glyf,

    (TT_Glyf_GetLocationFunc)tt_face_get_location      /* get_location */
  )


#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
  FT_DEFINE_SERVICEDESCREC6(
    tt_services,

    FT_SERVICE_ID_FONT_FORMAT,        FT_FONT_FORMAT_TRUETYPE,
    FT_SERVICE_ID_MULTI_MASTERS,      &tt_service_gx_multi_masters,
    FT_SERVICE_ID_METRICS_VARIATIONS, &tt_service_metrics_variations,
    FT_SERVICE_ID_TRUETYPE_ENGINE,    &tt_service_truetype_engine,
    FT_SERVICE_ID_TT_GLYF,            &tt_service_truetype_glyf,
    FT_SERVICE_ID_PROPERTIES,         &tt_service_properties )
#else
  FT_DEFINE_SERVICEDESCREC4(
    tt_services,

    FT_SERVICE_ID_FONT_FORMAT,     FT_FONT_FORMAT_TRUETYPE,
    FT_SERVICE_ID_TRUETYPE_ENGINE, &tt_service_truetype_engine,
    FT_SERVICE_ID_TT_GLYF,         &tt_service_truetype_glyf,
    FT_SERVICE_ID_PROPERTIES,      &tt_service_properties )
#endif


  FT_CALLBACK_DEF( FT_Module_Interface )
  tt_get_interface( FT_Module    driver,    /* TT_Driver */
                    const char*  tt_interface )
  {
    FT_Library           library;
    FT_Module_Interface  result;
    FT_Module            sfntd;
    SFNT_Service         sfnt;


    result = ft_service_list_lookup( tt_services, tt_interface );
    if ( result )
      return result;

    if ( !driver )
      return NULL;
    library = driver->library;
    if ( !library )
      return NULL;

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

      NULL,    /* module-specific interface */

      tt_driver_init,           /* FT_Module_Constructor  module_init   */
      tt_driver_done,           /* FT_Module_Destructor   module_done   */
      tt_get_interface,         /* FT_Module_Requester    get_interface */

    sizeof ( TT_FaceRec ),
    sizeof ( TT_SizeRec ),
    sizeof ( FT_GlyphSlotRec ),

    tt_face_init,               /* FT_Face_InitFunc  init_face */
    tt_face_done,               /* FT_Face_DoneFunc  done_face */
    tt_size_init,               /* FT_Size_InitFunc  init_size */
    tt_size_done,               /* FT_Size_DoneFunc  done_size */
    tt_slot_init,               /* FT_Slot_InitFunc  init_slot */
    NULL,                       /* FT_Slot_DoneFunc  done_slot */

    tt_glyph_load,              /* FT_Slot_LoadFunc  load_glyph */

    tt_get_kerning,             /* FT_Face_GetKerningFunc   get_kerning  */
    NULL,                       /* FT_Face_AttachFunc       attach_file  */
    tt_get_advances,            /* FT_Face_GetAdvancesFunc  get_advances */

    tt_size_request,            /* FT_Size_RequestFunc  request_size */
    TT_SIZE_SELECT              /* FT_Size_SelectFunc   select_size  */
  )


/* END */
