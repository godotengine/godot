/***************************************************************************/
/*                                                                         */
/*  svttcmap.h                                                             */
/*                                                                         */
/*    The FreeType TrueType/sfnt cmap extra information service.           */
/*                                                                         */
/*  Copyright 2003 by                                                      */
/*  Masatake YAMATO, Redhat K.K.                                           */
/*                                                                         */
/*  Copyright 2003, 2008, 2009, 2012 by                                    */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/

/* Development of this service is support of
   Information-technology Promotion Agency, Japan. */

#ifndef __SVTTCMAP_H__
#define __SVTTCMAP_H__

#include FT_INTERNAL_SERVICE_H
#include FT_TRUETYPE_TABLES_H


FT_BEGIN_HEADER


#define FT_SERVICE_ID_TT_CMAP  "tt-cmaps"


  /*************************************************************************/
  /*                                                                       */
  /* <Struct>                                                              */
  /*    TT_CMapInfo                                                        */
  /*                                                                       */
  /* <Description>                                                         */
  /*    A structure used to store TrueType/sfnt specific cmap information  */
  /*    which is not covered by the generic @FT_CharMap structure.  This   */
  /*    structure can be accessed with the @FT_Get_TT_CMap_Info function.  */
  /*                                                                       */
  /* <Fields>                                                              */
  /*    language ::                                                        */
  /*      The language ID used in Mac fonts.  Definitions of values are in */
  /*      freetype/ttnameid.h.                                             */
  /*                                                                       */
  /*    format ::                                                          */
  /*      The cmap format.  OpenType 1.5 defines the formats 0 (byte       */
  /*      encoding table), 2~(high-byte mapping through table), 4~(segment */
  /*      mapping to delta values), 6~(trimmed table mapping), 8~(mixed    */
  /*      16-bit and 32-bit coverage), 10~(trimmed array), 12~(segmented   */
  /*      coverage), and 14 (Unicode Variation Sequences).                 */
  /*                                                                       */
  typedef struct  TT_CMapInfo_
  {
    FT_ULong  language;
    FT_Long   format;

  } TT_CMapInfo;


  typedef FT_Error
  (*TT_CMap_Info_GetFunc)( FT_CharMap    charmap,
                           TT_CMapInfo  *cmap_info );


  FT_DEFINE_SERVICE( TTCMaps )
  {
    TT_CMap_Info_GetFunc  get_cmap_info;
  };

#ifndef FT_CONFIG_OPTION_PIC

#define FT_DEFINE_SERVICE_TTCMAPSREC( class_, get_cmap_info_ )  \
  static const FT_Service_TTCMapsRec  class_ =                  \
  {                                                             \
    get_cmap_info_                                              \
  };

#else /* FT_CONFIG_OPTION_PIC */

#define FT_DEFINE_SERVICE_TTCMAPSREC( class_, get_cmap_info_ )  \
  void                                                          \
  FT_Init_Class_ ## class_( FT_Library              library,    \
                            FT_Service_TTCMapsRec*  clazz )     \
  {                                                             \
    FT_UNUSED( library );                                       \
                                                                \
    clazz->get_cmap_info = get_cmap_info_;                      \
  }

#endif /* FT_CONFIG_OPTION_PIC */

  /* */


FT_END_HEADER

#endif /* __SVTTCMAP_H__ */


/* END */
