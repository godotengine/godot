/***************************************************************************/
/*                                                                         */
/*  cidgload.h                                                             */
/*                                                                         */
/*    OpenType Glyph Loader (specification).                               */
/*                                                                         */
/*  Copyright 1996-2016 by                                                 */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


#ifndef CIDGLOAD_H_
#define CIDGLOAD_H_


#include <ft2build.h>
#include "cidobjs.h"


FT_BEGIN_HEADER


#if 0

  /* Compute the maximum advance width of a font through quick parsing */
  FT_LOCAL( FT_Error )
  cid_face_compute_max_advance( CID_Face  face,
                                FT_Int*   max_advance );

#endif /* 0 */

  FT_LOCAL( FT_Error )
  cid_slot_load_glyph( FT_GlyphSlot  glyph,         /* CID_Glyph_Slot */
                       FT_Size       size,          /* CID_Size       */
                       FT_UInt       glyph_index,
                       FT_Int32      load_flags );


FT_END_HEADER

#endif /* CIDGLOAD_H_ */


/* END */
