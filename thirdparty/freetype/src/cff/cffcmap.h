/****************************************************************************
 *
 * cffcmap.h
 *
 *   CFF character mapping table (cmap) support (specification).
 *
 * Copyright (C) 2002-2024 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#ifndef CFFCMAP_H_
#define CFFCMAP_H_

#include <freetype/internal/cffotypes.h>

FT_BEGIN_HEADER


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****          TYPE1 STANDARD (AND EXPERT) ENCODING CMAPS           *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  /* standard (and expert) encoding cmaps */
  typedef struct CFF_CMapStdRec_*  CFF_CMapStd;

  typedef struct  CFF_CMapStdRec_
  {
    FT_CMapRec  cmap;
    FT_UShort*  gids;   /* up to 256 elements */

  } CFF_CMapStdRec;


  FT_DECLARE_CMAP_CLASS( cff_cmap_encoding_class_rec )


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****               CFF SYNTHETIC UNICODE ENCODING CMAP             *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  /* unicode (synthetic) cmaps */

  FT_DECLARE_CMAP_CLASS( cff_cmap_unicode_class_rec )


FT_END_HEADER

#endif /* CFFCMAP_H_ */


/* END */
