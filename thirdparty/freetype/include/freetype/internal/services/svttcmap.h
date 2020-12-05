/****************************************************************************
 *
 * svttcmap.h
 *
 *   The FreeType TrueType/sfnt cmap extra information service.
 *
 * Copyright (C) 2003-2020 by
 * Masatake YAMATO, Redhat K.K.,
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */

/* Development of this service is support of
   Information-technology Promotion Agency, Japan. */

#ifndef SVTTCMAP_H_
#define SVTTCMAP_H_

#include <freetype/internal/ftserv.h>
#include <freetype/tttables.h>


FT_BEGIN_HEADER


#define FT_SERVICE_ID_TT_CMAP  "tt-cmaps"


  /**************************************************************************
   *
   * @struct:
   *   TT_CMapInfo
   *
   * @description:
   *   A structure used to store TrueType/sfnt specific cmap information
   *   which is not covered by the generic @FT_CharMap structure.  This
   *   structure can be accessed with the @FT_Get_TT_CMap_Info function.
   *
   * @fields:
   *   language ::
   *     The language ID used in Mac fonts.  Definitions of values are in
   *     `ttnameid.h`.
   *
   *   format ::
   *     The cmap format.  OpenType 1.6 defines the formats 0 (byte encoding
   *     table), 2~(high-byte mapping through table), 4~(segment mapping to
   *     delta values), 6~(trimmed table mapping), 8~(mixed 16-bit and 32-bit
   *     coverage), 10~(trimmed array), 12~(segmented coverage), 13~(last
   *     resort font), and 14 (Unicode Variation Sequences).
   */
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


#define FT_DEFINE_SERVICE_TTCMAPSREC( class_, get_cmap_info_ )  \
  static const FT_Service_TTCMapsRec  class_ =                  \
  {                                                             \
    get_cmap_info_                                              \
  };

  /* */


FT_END_HEADER

#endif /* SVTTCMAP_H_ */


/* END */
