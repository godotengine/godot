/***************************************************************************/
/*                                                                         */
/*  ftpatent.c                                                             */
/*                                                                         */
/*    FreeType API for checking patented TrueType bytecode instructions    */
/*    (body).  Obsolete, retained for backward compatibility.              */
/*                                                                         */
/*  Copyright 2007-2018 by                                                 */
/*  David Turner.                                                          */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/

#include <ft2build.h>
#include FT_FREETYPE_H
#include FT_TRUETYPE_TAGS_H
#include FT_INTERNAL_OBJECTS_H
#include FT_INTERNAL_STREAM_H
#include FT_SERVICE_SFNT_H
#include FT_SERVICE_TRUETYPE_GLYF_H


  /* documentation is in freetype.h */

  FT_EXPORT_DEF( FT_Bool )
  FT_Face_CheckTrueTypePatents( FT_Face  face )
  {
    FT_UNUSED( face );

    return FALSE;
  }


  /* documentation is in freetype.h */

  FT_EXPORT_DEF( FT_Bool )
  FT_Face_SetUnpatentedHinting( FT_Face  face,
                                FT_Bool  value )
  {
    FT_UNUSED( face );
    FT_UNUSED( value );

    return FALSE;
  }

/* END */
