/***************************************************************************/
/*                                                                         */
/*  pfrdrivr.c                                                             */
/*                                                                         */
/*    FreeType PFR driver interface (body).                                */
/*                                                                         */
/*  Copyright 2002-2004, 2006, 2008, 2010, 2011, 2013 by                   */
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
#include FT_SERVICE_PFR_H
#include FT_SERVICE_XFREE86_NAME_H
#include "pfrdrivr.h"
#include "pfrobjs.h"

#include "pfrerror.h"


  FT_CALLBACK_DEF( FT_Error )
  pfr_get_kerning( FT_Face     pfrface,     /* PFR_Face */
                   FT_UInt     left,
                   FT_UInt     right,
                   FT_Vector  *avector )
  {
    PFR_Face     face = (PFR_Face)pfrface;
    PFR_PhyFont  phys = &face->phy_font;


    pfr_face_get_kerning( pfrface, left, right, avector );

    /* convert from metrics to outline units when necessary */
    if ( phys->outline_resolution != phys->metrics_resolution )
    {
      if ( avector->x != 0 )
        avector->x = FT_MulDiv( avector->x, phys->outline_resolution,
                                            phys->metrics_resolution );

      if ( avector->y != 0 )
        avector->y = FT_MulDiv( avector->x, phys->outline_resolution,
                                            phys->metrics_resolution );
    }

    return FT_Err_Ok;
  }


 /*
  *  PFR METRICS SERVICE
  *
  */

  FT_CALLBACK_DEF( FT_Error )
  pfr_get_advance( FT_Face   pfrface,       /* PFR_Face */
                   FT_UInt   gindex,
                   FT_Pos   *anadvance )
  {
    PFR_Face  face  = (PFR_Face)pfrface;
    FT_Error  error = FT_ERR( Invalid_Argument );


    *anadvance = 0;

    if ( !gindex )
      goto Exit;

    gindex--;

    if ( face )
    {
      PFR_PhyFont  phys = &face->phy_font;


      if ( gindex < phys->num_chars )
      {
        *anadvance = phys->chars[gindex].advance;
        error      = FT_Err_Ok;
      }
    }

  Exit:
    return error;
  }


  FT_CALLBACK_DEF( FT_Error )
  pfr_get_metrics( FT_Face    pfrface,      /* PFR_Face */
                   FT_UInt   *anoutline_resolution,
                   FT_UInt   *ametrics_resolution,
                   FT_Fixed  *ametrics_x_scale,
                   FT_Fixed  *ametrics_y_scale )
  {
    PFR_Face     face = (PFR_Face)pfrface;
    PFR_PhyFont  phys = &face->phy_font;
    FT_Fixed     x_scale, y_scale;
    FT_Size      size = face->root.size;


    if ( anoutline_resolution )
      *anoutline_resolution = phys->outline_resolution;

    if ( ametrics_resolution )
      *ametrics_resolution = phys->metrics_resolution;

    x_scale = 0x10000L;
    y_scale = 0x10000L;

    if ( size )
    {
      x_scale = FT_DivFix( size->metrics.x_ppem << 6,
                           phys->metrics_resolution );

      y_scale = FT_DivFix( size->metrics.y_ppem << 6,
                           phys->metrics_resolution );
    }

    if ( ametrics_x_scale )
      *ametrics_x_scale = x_scale;

    if ( ametrics_y_scale )
      *ametrics_y_scale = y_scale;

    return FT_Err_Ok;
  }


  FT_CALLBACK_TABLE_DEF
  const FT_Service_PfrMetricsRec  pfr_metrics_service_rec =
  {
    pfr_get_metrics,
    pfr_face_get_kerning,
    pfr_get_advance
  };


 /*
  *  SERVICE LIST
  *
  */

  static const FT_ServiceDescRec  pfr_services[] =
  {
    { FT_SERVICE_ID_PFR_METRICS, &pfr_metrics_service_rec },
    { FT_SERVICE_ID_XF86_NAME,   FT_XF86_FORMAT_PFR },
    { NULL, NULL }
  };


  FT_CALLBACK_DEF( FT_Module_Interface )
  pfr_get_service( FT_Module         module,
                   const FT_String*  service_id )
  {
    FT_UNUSED( module );

    return ft_service_list_lookup( pfr_services, service_id );
  }


  FT_CALLBACK_TABLE_DEF
  const FT_Driver_ClassRec  pfr_driver_class =
  {
    {
      FT_MODULE_FONT_DRIVER     |
      FT_MODULE_DRIVER_SCALABLE,

      sizeof ( FT_DriverRec ),

      "pfr",
      0x10000L,
      0x20000L,

      NULL,

      0,                /* FT_Module_Constructor */
      0,                /* FT_Module_Destructor  */
      pfr_get_service
    },

    sizeof ( PFR_FaceRec ),
    sizeof ( PFR_SizeRec ),
    sizeof ( PFR_SlotRec ),

    pfr_face_init,
    pfr_face_done,
    0,                  /* FT_Size_InitFunc */
    0,                  /* FT_Size_DoneFunc */
    pfr_slot_init,
    pfr_slot_done,

    pfr_slot_load,

    pfr_get_kerning,
    0,                  /* FT_Face_AttachFunc      */
    0,                  /* FT_Face_GetAdvancesFunc */
    0,                  /* FT_Size_RequestFunc     */
    0,                  /* FT_Size_SelectFunc      */
  };


/* END */
