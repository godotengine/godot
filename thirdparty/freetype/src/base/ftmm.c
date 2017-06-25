/***************************************************************************/
/*                                                                         */
/*  ftmm.c                                                                 */
/*                                                                         */
/*    Multiple Master font support (body).                                 */
/*                                                                         */
/*  Copyright 1996-2017 by                                                 */
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

#include FT_MULTIPLE_MASTERS_H
#include FT_INTERNAL_OBJECTS_H
#include FT_SERVICE_MULTIPLE_MASTERS_H
#include FT_SERVICE_METRICS_VARIATIONS_H


  /*************************************************************************/
  /*                                                                       */
  /* The macro FT_COMPONENT is used in trace mode.  It is an implicit      */
  /* parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log  */
  /* messages during execution.                                            */
  /*                                                                       */
#undef  FT_COMPONENT
#define FT_COMPONENT  trace_mm


  static FT_Error
  ft_face_get_mm_service( FT_Face                   face,
                          FT_Service_MultiMasters  *aservice )
  {
    FT_Error  error;


    *aservice = NULL;

    if ( !face )
      return FT_THROW( Invalid_Face_Handle );

    error = FT_ERR( Invalid_Argument );

    if ( FT_HAS_MULTIPLE_MASTERS( face ) )
    {
      FT_FACE_LOOKUP_SERVICE( face,
                              *aservice,
                              MULTI_MASTERS );

      if ( *aservice )
        error = FT_Err_Ok;
    }

    return error;
  }


  static FT_Error
  ft_face_get_mvar_service( FT_Face                        face,
                            FT_Service_MetricsVariations  *aservice )
  {
    FT_Error  error;


    *aservice = NULL;

    if ( !face )
      return FT_THROW( Invalid_Face_Handle );

    error = FT_ERR( Invalid_Argument );

    if ( FT_HAS_MULTIPLE_MASTERS( face ) )
    {
      FT_FACE_LOOKUP_SERVICE( face,
                              *aservice,
                              METRICS_VARIATIONS );

      if ( *aservice )
        error = FT_Err_Ok;
    }

    return error;
  }


  /* documentation is in ftmm.h */

  FT_EXPORT_DEF( FT_Error )
  FT_Get_Multi_Master( FT_Face           face,
                       FT_Multi_Master  *amaster )
  {
    FT_Error                 error;
    FT_Service_MultiMasters  service;


    /* check of `face' delayed to `ft_face_get_mm_service' */

    if ( !amaster )
      return FT_THROW( Invalid_Argument );

    error = ft_face_get_mm_service( face, &service );
    if ( !error )
    {
      error = FT_ERR( Invalid_Argument );
      if ( service->get_mm )
        error = service->get_mm( face, amaster );
    }

    return error;
  }


  /* documentation is in ftmm.h */

  FT_EXPORT_DEF( FT_Error )
  FT_Get_MM_Var( FT_Face      face,
                 FT_MM_Var*  *amaster )
  {
    FT_Error                 error;
    FT_Service_MultiMasters  service;


    /* check of `face' delayed to `ft_face_get_mm_service' */

    if ( !amaster )
      return FT_THROW( Invalid_Argument );

    error = ft_face_get_mm_service( face, &service );
    if ( !error )
    {
      error = FT_ERR( Invalid_Argument );
      if ( service->get_mm_var )
        error = service->get_mm_var( face, amaster );
    }

    return error;
  }


  /* documentation is in ftmm.h */

  FT_EXPORT_DEF( FT_Error )
  FT_Set_MM_Design_Coordinates( FT_Face   face,
                                FT_UInt   num_coords,
                                FT_Long*  coords )
  {
    FT_Error                 error;
    FT_Service_MultiMasters  service;


    /* check of `face' delayed to `ft_face_get_mm_service' */

    if ( !coords )
      return FT_THROW( Invalid_Argument );

    error = ft_face_get_mm_service( face, &service );
    if ( !error )
    {
      error = FT_ERR( Invalid_Argument );
      if ( service->set_mm_design )
        error = service->set_mm_design( face, num_coords, coords );
    }

    /* enforce recomputation of auto-hinting data */
    if ( !error && face->autohint.finalizer )
    {
      face->autohint.finalizer( face->autohint.data );
      face->autohint.data = NULL;
    }

    return error;
  }


  /* documentation is in ftmm.h */

  FT_EXPORT_DEF( FT_Error )
  FT_Set_Var_Design_Coordinates( FT_Face    face,
                                 FT_UInt    num_coords,
                                 FT_Fixed*  coords )
  {
    FT_Error                      error;
    FT_Service_MultiMasters       service_mm   = NULL;
    FT_Service_MetricsVariations  service_mvar = NULL;


    /* check of `face' delayed to `ft_face_get_mm_service' */

    if ( !coords )
      return FT_THROW( Invalid_Argument );

    error = ft_face_get_mm_service( face, &service_mm );
    if ( !error )
    {
      error = FT_ERR( Invalid_Argument );
      if ( service_mm->set_var_design )
        error = service_mm->set_var_design( face, num_coords, coords );
    }

    if ( !error )
    {
      (void)ft_face_get_mvar_service( face, &service_mvar );

      if ( service_mvar && service_mvar->metrics_adjust )
        service_mvar->metrics_adjust( face );
    }

    /* enforce recomputation of auto-hinting data */
    if ( !error && face->autohint.finalizer )
    {
      face->autohint.finalizer( face->autohint.data );
      face->autohint.data = NULL;
    }

    return error;
  }


  /* documentation is in ftmm.h */

  FT_EXPORT_DEF( FT_Error )
  FT_Get_Var_Design_Coordinates( FT_Face    face,
                                 FT_UInt    num_coords,
                                 FT_Fixed*  coords )
  {
    FT_Error                 error;
    FT_Service_MultiMasters  service;


    /* check of `face' delayed to `ft_face_get_mm_service' */

    if ( !coords )
      return FT_THROW( Invalid_Argument );

    error = ft_face_get_mm_service( face, &service );
    if ( !error )
    {
      error = FT_ERR( Invalid_Argument );
      if ( service->get_var_design )
        error = service->get_var_design( face, num_coords, coords );
    }

    return error;
  }


  /* documentation is in ftmm.h */

  FT_EXPORT_DEF( FT_Error )
  FT_Set_MM_Blend_Coordinates( FT_Face    face,
                               FT_UInt    num_coords,
                               FT_Fixed*  coords )
  {
    FT_Error                      error;
    FT_Service_MultiMasters       service_mm   = NULL;
    FT_Service_MetricsVariations  service_mvar = NULL;


    /* check of `face' delayed to `ft_face_get_mm_service' */

    if ( !coords )
      return FT_THROW( Invalid_Argument );

    error = ft_face_get_mm_service( face, &service_mm );
    if ( !error )
    {
      error = FT_ERR( Invalid_Argument );
      if ( service_mm->set_mm_blend )
        error = service_mm->set_mm_blend( face, num_coords, coords );
    }

    if ( !error )
    {
      (void)ft_face_get_mvar_service( face, &service_mvar );

      if ( service_mvar && service_mvar->metrics_adjust )
        service_mvar->metrics_adjust( face );
    }

    /* enforce recomputation of auto-hinting data */
    if ( !error && face->autohint.finalizer )
    {
      face->autohint.finalizer( face->autohint.data );
      face->autohint.data = NULL;
    }

    return error;
  }


  /* documentation is in ftmm.h */

  /* This is exactly the same as the previous function.  It exists for */
  /* orthogonality.                                                    */

  FT_EXPORT_DEF( FT_Error )
  FT_Set_Var_Blend_Coordinates( FT_Face    face,
                                FT_UInt    num_coords,
                                FT_Fixed*  coords )
  {
    FT_Error                      error;
    FT_Service_MultiMasters       service_mm   = NULL;
    FT_Service_MetricsVariations  service_mvar = NULL;


    /* check of `face' delayed to `ft_face_get_mm_service' */

    if ( !coords )
      return FT_THROW( Invalid_Argument );

    error = ft_face_get_mm_service( face, &service_mm );
    if ( !error )
    {
      error = FT_ERR( Invalid_Argument );
      if ( service_mm->set_mm_blend )
        error = service_mm->set_mm_blend( face, num_coords, coords );
    }

    if ( !error )
    {
      (void)ft_face_get_mvar_service( face, &service_mvar );

      if ( service_mvar && service_mvar->metrics_adjust )
        service_mvar->metrics_adjust( face );
    }

    /* enforce recomputation of auto-hinting data */
    if ( !error && face->autohint.finalizer )
    {
      face->autohint.finalizer( face->autohint.data );
      face->autohint.data = NULL;
    }

    return error;
  }


  /* documentation is in ftmm.h */

  FT_EXPORT_DEF( FT_Error )
  FT_Get_MM_Blend_Coordinates( FT_Face    face,
                               FT_UInt    num_coords,
                               FT_Fixed*  coords )
  {
    FT_Error                 error;
    FT_Service_MultiMasters  service;


    /* check of `face' delayed to `ft_face_get_mm_service' */

    if ( !coords )
      return FT_THROW( Invalid_Argument );

    error = ft_face_get_mm_service( face, &service );
    if ( !error )
    {
      error = FT_ERR( Invalid_Argument );
      if ( service->get_mm_blend )
        error = service->get_mm_blend( face, num_coords, coords );
    }

    return error;
  }


  /* documentation is in ftmm.h */

  /* This is exactly the same as the previous function.  It exists for */
  /* orthogonality.                                                    */

  FT_EXPORT_DEF( FT_Error )
  FT_Get_Var_Blend_Coordinates( FT_Face    face,
                                FT_UInt    num_coords,
                                FT_Fixed*  coords )
  {
    FT_Error                 error;
    FT_Service_MultiMasters  service;


    /* check of `face' delayed to `ft_face_get_mm_service' */

    if ( !coords )
      return FT_THROW( Invalid_Argument );

    error = ft_face_get_mm_service( face, &service );
    if ( !error )
    {
      error = FT_ERR( Invalid_Argument );
      if ( service->get_mm_blend )
        error = service->get_mm_blend( face, num_coords, coords );
    }

    return error;
  }


/* END */
