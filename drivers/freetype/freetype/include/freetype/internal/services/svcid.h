/***************************************************************************/
/*                                                                         */
/*  svcid.h                                                                */
/*                                                                         */
/*    The FreeType CID font services (specification).                      */
/*                                                                         */
/*  Copyright 2007, 2009, 2012 by Derek Clegg, Michael Toftdal.            */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


#ifndef __SVCID_H__
#define __SVCID_H__

#include FT_INTERNAL_SERVICE_H


FT_BEGIN_HEADER


#define FT_SERVICE_ID_CID  "CID"

  typedef FT_Error
  (*FT_CID_GetRegistryOrderingSupplementFunc)( FT_Face       face,
                                               const char*  *registry,
                                               const char*  *ordering,
                                               FT_Int       *supplement );
  typedef FT_Error
  (*FT_CID_GetIsInternallyCIDKeyedFunc)( FT_Face   face,
                                         FT_Bool  *is_cid );
  typedef FT_Error
  (*FT_CID_GetCIDFromGlyphIndexFunc)( FT_Face   face,
                                      FT_UInt   glyph_index,
                                      FT_UInt  *cid );

  FT_DEFINE_SERVICE( CID )
  {
    FT_CID_GetRegistryOrderingSupplementFunc  get_ros;
    FT_CID_GetIsInternallyCIDKeyedFunc        get_is_cid;
    FT_CID_GetCIDFromGlyphIndexFunc           get_cid_from_glyph_index;
  };


#ifndef FT_CONFIG_OPTION_PIC

#define FT_DEFINE_SERVICE_CIDREC( class_,                                   \
                                  get_ros_,                                 \
                                  get_is_cid_,                              \
                                  get_cid_from_glyph_index_ )               \
  static const FT_Service_CIDRec class_ =                                   \
  {                                                                         \
    get_ros_, get_is_cid_, get_cid_from_glyph_index_                        \
  };

#else /* FT_CONFIG_OPTION_PIC */

#define FT_DEFINE_SERVICE_CIDREC( class_,                                   \
                                  get_ros_,                                 \
                                  get_is_cid_,                              \
                                  get_cid_from_glyph_index_ )               \
  void                                                                      \
  FT_Init_Class_ ## class_( FT_Library          library,                    \
                            FT_Service_CIDRec*  clazz )                     \
  {                                                                         \
    FT_UNUSED( library );                                                   \
                                                                            \
    clazz->get_ros                  = get_ros_;                             \
    clazz->get_is_cid               = get_is_cid_;                          \
    clazz->get_cid_from_glyph_index = get_cid_from_glyph_index_;            \
  }

#endif /* FT_CONFIG_OPTION_PIC */

  /* */


FT_END_HEADER


#endif /* __SVCID_H__ */


/* END */
