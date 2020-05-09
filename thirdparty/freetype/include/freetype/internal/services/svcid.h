/****************************************************************************
 *
 * svcid.h
 *
 *   The FreeType CID font services (specification).
 *
 * Copyright (C) 2007-2020 by
 * Derek Clegg and Michael Toftdal.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#ifndef SVCID_H_
#define SVCID_H_

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


#define FT_DEFINE_SERVICE_CIDREC( class_,                                   \
                                  get_ros_,                                 \
                                  get_is_cid_,                              \
                                  get_cid_from_glyph_index_ )               \
  static const FT_Service_CIDRec class_ =                                   \
  {                                                                         \
    get_ros_, get_is_cid_, get_cid_from_glyph_index_                        \
  };

  /* */


FT_END_HEADER


#endif /* SVCID_H_ */


/* END */
