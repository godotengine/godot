/****************************************************************************
 *
 * hvfdrv.c
 *
 *   HVF font driver implementation (body).
 *
 * Copyright (C) 2025-2026 by
 * Apple Inc.
 * written by Deborah Goldsmith <goldsmit@apple.com>
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
#include <freetype/tttags.h>
#include <freetype/internal/tttypes.h>  /* For TT_Face */
#include <freetype/internal/services/svfntfmt.h>

#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
#include <freetype/ftmm.h>
#include <freetype/internal/services/svmm.h>
#include <freetype/internal/services/svmetric.h>
#endif

#include "hvfdrv.h"
#include "hvfobjs.h"
#include "hvfload.h"
#include "hvferror.h"


  /**************************************************************************
   *
   * The macro FT_COMPONENT is used in trace mode.  It is an implicit
   * parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log
   * messages during execution.
   */
#undef  FT_COMPONENT
#define FT_COMPONENT  hvfdrv


#ifdef FT_CONFIG_OPTION_HVF

  /**************************************************************************
   *
   *                          FACE  MANAGEMENT
   *
   */

  /**************************************************************************
   *
   * @Function:
   *   hvf_get_kerning
   *
   * @Description:
   *   Get kerning vector between two glyphs.  Delegates to SFNT service
   *   since HVF fonts are TrueType/OpenType fonts.
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
   *     The kerning vector in font units.
   *
   * @Return:
   *   FreeType error code.  0 means success.
   */
  static FT_Error
  hvf_get_kerning( FT_Face     face,
                   FT_UInt     left_glyph,
                   FT_UInt     right_glyph,
                   FT_Vector*  kerning )
  {
    TT_Face       tt_face = (TT_Face)face;
    SFNT_Service  sfnt    = (SFNT_Service)tt_face->sfnt;


    kerning->x = 0;
    kerning->y = 0;

    if ( sfnt )
    {
      /* Use 'kern' table if available since that can be faster; otherwise */
      /* use GPOS kerning pairs if available.                              */
      if ( tt_face->kern_avail_bits )
        kerning->x = sfnt->get_kerning( tt_face,
                                        left_glyph,
                                        right_glyph );
#ifdef TT_CONFIG_OPTION_GPOS_KERNING
      else if ( tt_face->num_gpos_lookups_kerning )
        kerning->x = sfnt->get_gpos_kerning( tt_face,
                                             left_glyph,
                                             right_glyph );
#endif
    }

    return FT_Err_Ok;
  }


  /**************************************************************************
   *
   * @Function:
   *   hvf_get_advances
   *
   * @Description:
   *   Get advance widths/heights for a range of glyphs.  Delegates to
   *   TrueType infrastructure since HVF fonts are TrueType/OpenType fonts.
   *
   * @Input:
   *   face ::
   *     A handle to the source face object.
   *
   *   start ::
   *     The first glyph index.
   *
   *   count ::
   *     The number of advance values to retrieve.
   *
   *   flags ::
   *     Loading flags (vertical/horizontal layout).
   *
   * @Output:
   *   advances ::
   *     The advance values in 16.16 format.
   *
   * @Return:
   *   FreeType error code.  0 means success.
   */
  static FT_Error
  hvf_get_advances( FT_Face    face,
                    FT_UInt    start,
                    FT_UInt    count,
                    FT_Int32   flags,
                    FT_Fixed  *advances )
  {
    TT_Face  tt_face = (TT_Face)face;
    FT_Bool  horz;
    FT_UInt  nn;


    if ( !FT_IS_SFNT( face ) )
      return FT_THROW( Unimplemented_Feature );

    horz = !( flags & FT_LOAD_VERTICAL_LAYOUT );

    if ( horz )
    {
      /* Check availability of horizontal metrics. */
      if ( !tt_face->horizontal.number_Of_HMetrics )
        return FT_THROW( Unimplemented_Feature );

#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
      /* No fast retrieval for blended MM fonts without 'HVAR' table. */
      if ( ( FT_IS_NAMED_INSTANCE( face ) || FT_IS_VARIATION( face ) ) &&
           !( tt_face->variation_support & TT_FACE_FLAG_VAR_HADVANCE ) )
        return FT_THROW( Unimplemented_Feature );
#endif
    }
    else  /* vertical */
    {
      /* Check whether we have data from the 'vmtx' table at all. */
      if ( !tt_face->vertical_info )
        return FT_THROW( Unimplemented_Feature );

#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
      /* No fast retrieval for blended MM fonts without 'VVAR' table. */
      if ( ( FT_IS_NAMED_INSTANCE( face ) || FT_IS_VARIATION( face ) ) &&
           !( tt_face->variation_support & TT_FACE_FLAG_VAR_VADVANCE ) )
        return FT_THROW( Unimplemented_Feature );
#endif
    }

    /* Proceed to fast advances. */
    for ( nn = 0; nn < count; nn++ )
    {
      FT_UShort  aw;
      FT_Short   dummy;


      ( (SFNT_Service)tt_face->sfnt )->get_metrics( tt_face,
                                                    !horz,
                                                    start + nn,
                                                    &dummy,
                                                    &aw );

      FT_TRACE5(( "  idx %u: advance %s %d font unit%s\n",
                  start + nn,
                  horz ? "width" : "height",
                  aw,
                  aw == 1 ? "" : "s" ));
      advances[nn] = aw;
    }

    return FT_Err_Ok;
  }


  /**************************************************************************
   *
   *                          SERVICE DEFINITIONS
   *
   */

#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT

  /* Service delegation functions following CFF module pattern. */

  FT_CALLBACK_DEF( FT_Error )
  hvf_set_mm_blend( FT_Face    face,        /* HVF_Face */
                    FT_UInt    num_coords,
                    FT_Fixed*  coords )
  {
    TT_Face                  tt_face  = (TT_Face)face;
    FT_Service_MultiMasters  mm       = (FT_Service_MultiMasters)tt_face->mm;
    HVF_Face                 hvf_face = (HVF_Face)face;

    FT_Error  error;


    /* Call the underlying service. */
    error = mm->set_mm_blend( face, num_coords, coords );
    
    /* error == -1 means "no change"; early exit. */
    if ( error == -1 )
      return FT_Err_Ok;
    
    /* If successful, refresh our cached HVF coordinates. */
    if ( !error )
    {
      FT_Error  refresh_error = hvf_refresh_axis_coordinates( hvf_face );


#ifdef FT_DEBUG_LEVEL_TRACE
      if ( refresh_error )
      {
        /* Don't fail the entire operation, just log the issue. */
        FT_TRACE3(( "hvf_set_mm_blend:"
                    " failed to refresh HVF coordinates (error %d)\n",
                    refresh_error ));
      }
#else
      FT_UNUSED( refresh_error );
#endif
    }

    return error;
  }


  FT_CALLBACK_DEF( FT_Error )
  hvf_get_mm_blend( FT_Face    face,        /* HVF_Face */
                    FT_UInt    num_coords,
                    FT_Fixed*  coords )
  {
    TT_Face                  tt_face = (TT_Face)face;
    FT_Service_MultiMasters  mm      = (FT_Service_MultiMasters)tt_face->mm;


    return mm->get_mm_blend( face, num_coords, coords );
  }


  FT_CALLBACK_DEF( FT_Error )
  hvf_get_mm_var( FT_Face      face,    /* HVF_Face */
                  FT_MM_Var**  master )
  {
    TT_Face                  tt_face = (TT_Face)face;
    FT_Service_MultiMasters  mm      = (FT_Service_MultiMasters)tt_face->mm;


    return mm->get_mm_var( face, master );
  }


  FT_CALLBACK_DEF( FT_Error )
  hvf_set_var_design( FT_Face    face,       /* HVF_Face */
                      FT_UInt    num_coords,
                      FT_Fixed*  coords )
  {
    TT_Face                  tt_face  = (TT_Face)face;
    FT_Service_MultiMasters  mm       = (FT_Service_MultiMasters)tt_face->mm;
    HVF_Face                 hvf_face = (HVF_Face)face;

    FT_Error  error;


    /* Call the underlying service. */
    error = mm->set_var_design( face, num_coords, coords );

    /* error == -1 means "no change"; early exit. */
    if ( error == -1 )
      return FT_Err_Ok;
    
    /* If successful, refresh our cached HVF coordinates. */
    if ( !error || error == -2 )
    {
      FT_Error  refresh_error = hvf_refresh_axis_coordinates( hvf_face );


#ifdef FT_DEBUG_LEVEL_TRACE
      if ( refresh_error )
      {
        /* Don't fail the entire operation, just log the issue. */
        FT_TRACE3(( "hvf_set_var_design:"
                    " failed to refresh HVF coordinates (error %d)\n",
                    refresh_error ));
      }
#else
      FT_UNUSED( refresh_error );
#endif
    }

    return error;
  }


  FT_CALLBACK_DEF( FT_Error )
  hvf_get_var_design( FT_Face    face,       /* HVF_Face */
                      FT_UInt    num_coords,
                      FT_Fixed*  coords )
  {
    TT_Face                  tt_face = (TT_Face)face;
    FT_Service_MultiMasters  mm      = (FT_Service_MultiMasters)tt_face->mm;


    return mm->get_var_design( face, num_coords, coords );
  }


  FT_CALLBACK_DEF( FT_Error )
  hvf_set_named_instance( FT_Face  face,            /* HVF_Face */
                          FT_UInt  instance_index )
  {
    TT_Face                  tt_face  = (TT_Face)face;
    FT_Service_MultiMasters  mm       = (FT_Service_MultiMasters)tt_face->mm;
    HVF_Face                 hvf_face = (HVF_Face)face;

    FT_Error  error;


    /* Call the underlying service. */
    error = mm->set_named_instance( face, instance_index );

    /* error == -1 means "no change"; early exit. */
    if ( error == -1 )
      return FT_Err_Ok;
    
    /* If successful, refresh our cached HVF coordinates. */
    if ( !error )
    {
      FT_Error  refresh_error = hvf_refresh_axis_coordinates( hvf_face );


#ifdef FT_DEBUG_LEVEL_TRACE
      if ( refresh_error )
      {
        /* Don't fail the entire operation, just log the issue. */
        FT_TRACE3(( "hvf_set_named_instance:"
                    " failed to refresh HVF coordinates (error %d)\n",
                    refresh_error ));
      }
#else
      FT_UNUSED( refresh_error );
#endif
    }

    return error;
  }


  FT_CALLBACK_DEF( FT_Error )
  hvf_get_default_named_instance( FT_Face   face,            /* HVF_Face */
                                  FT_UInt  *instance_index )
  {
    TT_Face                  tt_face = (TT_Face)face;
    FT_Service_MultiMasters  mm      = (FT_Service_MultiMasters)tt_face->mm;


    return mm->get_default_named_instance( face, instance_index );
  }


  FT_CALLBACK_DEF( void )
  hvf_construct_ps_name( FT_Face  face )  /* HVF_Face */
  {
    TT_Face                  tt_face = (TT_Face)face;
    FT_Service_MultiMasters  mm      = (FT_Service_MultiMasters)tt_face->mm;


    mm->construct_ps_name( face );
  }


  FT_CALLBACK_DEF( FT_Error )
  hvf_load_delta_set_index_mapping( FT_Face            face,   /* HVF_Face */
                                    FT_ULong           offset,
                                    GX_DeltaSetIdxMap  map,
                                    GX_ItemVarStore    itemStore,
                                    FT_ULong           table_len )
  {
    TT_Face                  tt_face = (TT_Face)face;
    FT_Service_MultiMasters  mm      = (FT_Service_MultiMasters)tt_face->mm;


    return mm->load_delta_set_idx_map( face, offset, map,
                                       itemStore, table_len );
  }


  FT_CALLBACK_DEF( FT_Error )
  hvf_load_item_variation_store( FT_Face          face,       /* HVF_Face */
                                 FT_ULong         offset,
                                 GX_ItemVarStore  itemStore )
  {
    TT_Face                  tt_face = (TT_Face)face;
    FT_Service_MultiMasters  mm      = (FT_Service_MultiMasters)tt_face->mm;


    return mm->load_item_var_store( face, offset, itemStore );
  }


  FT_CALLBACK_DEF( FT_ItemVarDelta )
  hvf_get_item_delta( FT_Face          face,        /* HVF_Face */
                      GX_ItemVarStore  itemStore,
                      FT_UInt          outerIndex,
                      FT_UInt          innerIndex )
  {
    TT_Face                  tt_face = (TT_Face)face;
    FT_Service_MultiMasters  mm      = (FT_Service_MultiMasters)tt_face->mm;


    return mm->get_item_delta( face, itemStore, outerIndex, innerIndex );
  }


  FT_CALLBACK_DEF( void )
  hvf_done_item_variation_store( FT_Face          face,       /* HVF_Face */
                                 GX_ItemVarStore  itemStore )
  {
    TT_Face                  tt_face = (TT_Face)face;
    FT_Service_MultiMasters  mm      = (FT_Service_MultiMasters)tt_face->mm;


    mm->done_item_var_store( face, itemStore );
  }


  FT_CALLBACK_DEF( void )
  hvf_done_delta_set_index_map( FT_Face            face,       /* HVF_Face */
                                GX_DeltaSetIdxMap  deltaSetIdxMap )
  {
    TT_Face                  tt_face = (TT_Face)face;
    FT_Service_MultiMasters  mm      = (FT_Service_MultiMasters)tt_face->mm;


    mm->done_delta_set_idx_map( face, deltaSetIdxMap );
  }


  FT_CALLBACK_DEF( FT_Error )
  hvf_get_var_blend( FT_Face      face,              /* HVF_Face */
                     FT_UInt*     num_coords,
                     FT_Fixed**   coords,
                     FT_Fixed**   normalizedcoords,
                     FT_MM_Var**  mm_var )
  {
    TT_Face                  tt_face = (TT_Face)face;
    FT_Service_MultiMasters  mm      = (FT_Service_MultiMasters)tt_face->mm;


    return mm->get_var_blend( face, num_coords, coords,
                              normalizedcoords, mm_var );
  }


  FT_CALLBACK_DEF( void )
  hvf_done_blend( FT_Face  face )  /* HVF_Face */
  {
    TT_Face                  tt_face = (TT_Face)face;
    FT_Service_MultiMasters  mm      = (FT_Service_MultiMasters)tt_face->mm;


    mm->done_blend( face );
  }


  FT_CALLBACK_DEF( FT_Error )
  hvf_hadvance_adjust( FT_Face  face,    /* HVF_Face */
                       FT_UInt  gindex,
                       FT_Int   *avalue )
  {
    TT_Face                       tt_face = (TT_Face)face;
    FT_Service_MetricsVariations  var     = (FT_Service_MetricsVariations)
                                            tt_face->tt_var;


    return var->hadvance_adjust( face, gindex, avalue );
  }


  FT_CALLBACK_DEF( FT_Error )
  hvf_vadvance_adjust( FT_Face  face,    /* HVF_Face */
                       FT_UInt  gindex,
                       FT_Int   *avalue )
  {
    TT_Face                       tt_face = (TT_Face)face;
    FT_Service_MetricsVariations  var     = (FT_Service_MetricsVariations)
                                            tt_face->tt_var;


    return var->vadvance_adjust( face, gindex, avalue );
  }


  FT_CALLBACK_DEF( void )
  hvf_apply_mvar( FT_Face  face )    /* HVF_Face */
  {
    TT_Face                       tt_face = (TT_Face)face;
    FT_Service_MetricsVariations  var     = (FT_Service_MetricsVariations)
                                            tt_face->tt_var;


    var->metrics_adjust( face );
  }


  FT_CALLBACK_DEF( void )
  hvf_size_reset_height( FT_Size  size )
  {
    /* Delegate to size's face services. */
    TT_Face                       tt_face = (TT_Face)size->face;
    FT_Service_MetricsVariations  var     = (FT_Service_MetricsVariations)
                                            tt_face->tt_var;


    if ( var->size_reset )
      var->size_reset( size );
  }


  FT_DEFINE_SERVICE_MULTIMASTERSREC(
    hvf_service_gx_multi_masters,

    NULL,                /* FT_Get_MM_Func         get_mm                     */
    NULL,                /* FT_Set_MM_Design_Func  set_mm_design              */
    hvf_set_mm_blend,    /* FT_Set_MM_Blend_Func   set_mm_blend               */
    hvf_get_mm_blend,    /* FT_Get_MM_Blend_Func   get_mm_blend               */
    hvf_get_mm_var,      /* FT_Get_MM_Var_Func     get_mm_var                 */
    hvf_set_var_design,  /* FT_Set_Var_Design_Func set_var_design             */
    hvf_get_var_design,  /* FT_Get_Var_Design_Func get_var_design             */
    hvf_set_named_instance,
             /* FT_Set_Named_Instance_Func         set_named_instance         */
    hvf_get_default_named_instance,
             /* FT_Get_Default_Named_Instance_Func get_default_named_instance */
    NULL,    /* FT_Set_MM_WeightVector_Func        set_mm_weightvector        */
    NULL,    /* FT_Get_MM_WeightVector_Func        get_mm_weightvector        */

    hvf_construct_ps_name,
             /* FT_Construct_PS_Name_Func          construct_ps_name          */
    hvf_load_delta_set_index_mapping,
             /* FT_Var_Load_Delta_Set_Idx_Map_Func load_delta_set_idx_map     */
    hvf_load_item_variation_store,
             /* FT_Var_Load_Item_Var_Store_Func    load_item_variation_store  */
    hvf_get_item_delta,
             /* FT_Var_Get_Item_Delta_Func         get_item_delta             */
    hvf_done_item_variation_store,
             /* FT_Var_Done_Item_Var_Store_Func    done_item_variation_store  */
    hvf_done_delta_set_index_map,
             /* FT_Var_Done_Delta_Set_Idx_Map_Func done_delta_set_index_map   */
    hvf_get_var_blend,   /* FT_Get_Var_Blend_Func  get_var_blend              */
    hvf_done_blend       /* FT_Done_Blend_Func     done_blend                 */
  )


  FT_DEFINE_SERVICE_METRICSVARIATIONSREC(
    hvf_service_metrics_variations,

    hvf_hadvance_adjust,  /* FT_HAdvance_Adjust_Func hadvance_adjust */
    NULL,                 /* FT_LSB_Adjust_Func      lsb_adjust      */
    NULL,                 /* FT_RSB_Adjust_Func      rsb_adjust      */

    hvf_vadvance_adjust,  /* FT_VAdvance_Adjust_Func vadvance_adjust */
    NULL,                 /* FT_TSB_Adjust_Func      tsb_adjust      */
    NULL,                 /* FT_BSB_Adjust_Func      bsb_adjust      */
    NULL,                 /* FT_VOrg_Adjust_Func     vorg_adjust     */

    hvf_apply_mvar,       /* FT_Metrics_Adjust_Func  metrics_adjust  */
    hvf_size_reset_height /* FT_Size_Reset_Func      size_reset      */
  )

#endif /* TT_CONFIG_OPTION_GX_VAR_SUPPORT */


#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
  FT_DEFINE_SERVICEDESCREC3(
    hvf_services,

    FT_SERVICE_ID_FONT_FORMAT,        FT_FONT_FORMAT_HVF,
    FT_SERVICE_ID_MULTI_MASTERS,      &hvf_service_gx_multi_masters,
    FT_SERVICE_ID_METRICS_VARIATIONS, &hvf_service_metrics_variations )
#else
  FT_DEFINE_SERVICEDESCREC1(
    hvf_services,

    FT_SERVICE_ID_FONT_FORMAT, FT_FONT_FORMAT_HVF )
#endif


  FT_CALLBACK_DEF( FT_Module_Interface )
  hvf_get_interface( FT_Module    driver,    /* HVF_Driver */
                     const char*  hvf_interface )
  {
    FT_Library           library;
    FT_Module_Interface  result;
    FT_Module            sfntd;
    SFNT_Service         sfnt;


    FT_TRACE5(( "hvf_get_interface: Looking up service %s\n",
                hvf_interface ));

    result = ft_service_list_lookup( hvf_services, hvf_interface );
    if ( result )
      return result;

    FT_TRACE3(( "hvf_get_interface: Did not find service %s; falling back\n",
                hvf_interface ));

    if ( !driver )
      return NULL;

    library = driver->library;
    if ( !library )
      return NULL;

    /* Only return the default interface from the SFNT module. */
    sfntd = FT_Get_Module( library, "sfnt" );
    if ( sfntd )
    {
      sfnt = (SFNT_Service)( sfntd->clazz->module_interface );
      if ( sfnt )
        return sfnt->get_interface( driver, hvf_interface );
    }

    return NULL;
  }

#endif /* FT_CONFIG_OPTION_HVF */


  /**************************************************************************
   *
   *                          DRIVER INTERFACE
   *
   */

#ifdef FT_CONFIG_OPTION_HVF
#define PUT_HVF_MODULE( a )  a
#else
#define PUT_HVF_MODULE( a )  NULL
#endif

  FT_DEFINE_DRIVER(
    hvf_driver_class,

      FT_MODULE_FONT_DRIVER        |
      FT_MODULE_DRIVER_SCALABLE    |
      FT_MODULE_DRIVER_HAS_HINTER,

      sizeof ( FT_DriverRec ),
      "hvf",           /* driver name                           */
      0x10000L,        /* driver version == 1.0                 */
      0x20000L,        /* driver requires FreeType 2.0 or above */

      NULL,    /* module-specific interface */

      NULL,                        /* FT_Module_Constructor  module_init   */
      NULL,                        /* FT_Module_Destructor   module_done   */
      PUT_HVF_MODULE( hvf_get_interface ),
                                   /* FT_Module_Requester    get_interface */

    sizeof ( HVF_FaceRec ),
    sizeof ( FT_SizeRec ),
    sizeof ( FT_GlyphSlotRec ),

    hvf_face_init,                          /* FT_Face_InitFunc  init_face */
    PUT_HVF_MODULE( hvf_face_done ),        /* FT_Face_DoneFunc  done_face */
    NULL,                                   /* FT_Size_InitFunc  init_size */
    NULL,                                   /* FT_Size_DoneFunc  done_size */
    NULL,                                   /* FT_Slot_InitFunc  init_slot */
    NULL,                                   /* FT_Slot_DoneFunc  done_slot */

    hvf_slot_load_glyph,                   /* FT_Slot_LoadFunc  load_glyph */

    PUT_HVF_MODULE( hvf_get_kerning ),
                                  /* FT_Face_GetKerningFunc   get_kerning  */
    NULL,                         /* FT_Face_AttachFunc       attach_file  */
    PUT_HVF_MODULE( hvf_get_advances ),
                                  /* FT_Face_GetAdvancesFunc  get_advances */

    NULL,                             /* FT_Size_RequestFunc  request_size */
    NULL                              /* FT_Size_SelectFunc   select_size  */
  )


/* END */
