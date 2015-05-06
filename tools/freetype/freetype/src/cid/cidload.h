/***************************************************************************/
/*                                                                         */
/*  cidload.h                                                              */
/*                                                                         */
/*    CID-keyed Type1 font loader (specification).                         */
/*                                                                         */
/*  Copyright 1996-2001, 2002, 2003, 2004 by                               */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


#ifndef __CIDLOAD_H__
#define __CIDLOAD_H__


#include <ft2build.h>
#include FT_INTERNAL_STREAM_H
#include "cidparse.h"


FT_BEGIN_HEADER


  typedef struct  CID_Loader_
  {
    CID_Parser  parser;          /* parser used to read the stream */
    FT_Int      num_chars;       /* number of characters in encoding */

  } CID_Loader;


  FT_LOCAL( FT_Long )
  cid_get_offset( FT_Byte**  start,
                  FT_Byte    offsize );

  FT_LOCAL( FT_Error )
  cid_face_open( CID_Face  face,
                 FT_Int    face_index );


FT_END_HEADER

#endif /* __CIDLOAD_H__ */


/* END */
