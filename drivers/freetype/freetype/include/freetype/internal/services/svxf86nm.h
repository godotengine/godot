/***************************************************************************/
/*                                                                         */
/*  svxf86nm.h                                                             */
/*                                                                         */
/*    The FreeType XFree86 services (specification only).                  */
/*                                                                         */
/*  Copyright 2003 by                                                      */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


#ifndef __SVXF86NM_H__
#define __SVXF86NM_H__

#include FT_INTERNAL_SERVICE_H


FT_BEGIN_HEADER


  /*
   *  A trivial service used to return the name of a face's font driver,
   *  according to the XFree86 nomenclature.  Note that the service data
   *  is a simple constant string pointer.
   */

#define FT_SERVICE_ID_XF86_NAME  "xf86-driver-name"

#define FT_XF86_FORMAT_TRUETYPE  "TrueType"
#define FT_XF86_FORMAT_TYPE_1    "Type 1"
#define FT_XF86_FORMAT_BDF       "BDF"
#define FT_XF86_FORMAT_PCF       "PCF"
#define FT_XF86_FORMAT_TYPE_42   "Type 42"
#define FT_XF86_FORMAT_CID       "CID Type 1"
#define FT_XF86_FORMAT_CFF       "CFF"
#define FT_XF86_FORMAT_PFR       "PFR"
#define FT_XF86_FORMAT_WINFNT    "Windows FNT"

  /* */


FT_END_HEADER


#endif /* __SVXF86NM_H__ */


/* END */
