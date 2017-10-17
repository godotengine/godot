/***************************************************************************/
/*                                                                         */
/*  pfrcmap.h                                                              */
/*                                                                         */
/*    FreeType PFR cmap handling (specification).                          */
/*                                                                         */
/*  Copyright 2002-2017 by                                                 */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


#ifndef PFRCMAP_H_
#define PFRCMAP_H_

#include <ft2build.h>
#include FT_INTERNAL_OBJECTS_H
#include "pfrtypes.h"


FT_BEGIN_HEADER

  typedef struct  PFR_CMapRec_
  {
    FT_CMapRec  cmap;
    FT_UInt     num_chars;
    PFR_Char    chars;

  } PFR_CMapRec, *PFR_CMap;


  FT_CALLBACK_TABLE const FT_CMap_ClassRec  pfr_cmap_class_rec;

FT_END_HEADER


#endif /* PFRCMAP_H_ */


/* END */
