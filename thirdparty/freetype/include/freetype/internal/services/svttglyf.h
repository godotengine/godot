/***************************************************************************/
/*                                                                         */
/*  svttglyf.h                                                             */
/*                                                                         */
/*    The FreeType TrueType glyph service.                                 */
/*                                                                         */
/*  Copyright 2007-2017 by                                                 */
/*  David Turner.                                                          */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/

#ifndef SVTTGLYF_H_
#define SVTTGLYF_H_

#include FT_INTERNAL_SERVICE_H
#include FT_TRUETYPE_TABLES_H


FT_BEGIN_HEADER


#define FT_SERVICE_ID_TT_GLYF  "tt-glyf"


  typedef FT_ULong
  (*TT_Glyf_GetLocationFunc)( FT_Face    face,
                              FT_UInt    gindex,
                              FT_ULong  *psize );

  FT_DEFINE_SERVICE( TTGlyf )
  {
    TT_Glyf_GetLocationFunc  get_location;
  };


#ifndef FT_CONFIG_OPTION_PIC

#define FT_DEFINE_SERVICE_TTGLYFREC( class_, get_location_ )  \
  static const FT_Service_TTGlyfRec  class_ =                 \
  {                                                           \
    get_location_                                             \
  };

#else /* FT_CONFIG_OPTION_PIC */

#define FT_DEFINE_SERVICE_TTGLYFREC( class_, get_location_ )  \
  void                                                        \
  FT_Init_Class_ ## class_( FT_Service_TTGlyfRec*  clazz )    \
  {                                                           \
    clazz->get_location = get_location_;                      \
  }

#endif /* FT_CONFIG_OPTION_PIC */

  /* */


FT_END_HEADER

#endif /* SVTTGLYF_H_ */


/* END */
