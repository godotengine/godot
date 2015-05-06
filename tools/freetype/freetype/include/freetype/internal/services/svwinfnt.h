/***************************************************************************/
/*                                                                         */
/*  svwinfnt.h                                                             */
/*                                                                         */
/*    The FreeType Windows FNT/FONT service (specification).               */
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


#ifndef __SVWINFNT_H__
#define __SVWINFNT_H__

#include FT_INTERNAL_SERVICE_H
#include FT_WINFONTS_H


FT_BEGIN_HEADER


#define FT_SERVICE_ID_WINFNT  "winfonts"

  typedef FT_Error
  (*FT_WinFnt_GetHeaderFunc)( FT_Face               face,
                              FT_WinFNT_HeaderRec  *aheader );


  FT_DEFINE_SERVICE( WinFnt )
  {
    FT_WinFnt_GetHeaderFunc  get_header;
  };

  /* */


FT_END_HEADER


#endif /* __SVWINFNT_H__ */


/* END */
