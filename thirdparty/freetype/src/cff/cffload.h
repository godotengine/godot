/****************************************************************************
 *
 * cffload.h
 *
 *   OpenType & CFF data/program tables loader (specification).
 *
 * Copyright (C) 1996-2019 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#ifndef CFFLOAD_H_
#define CFFLOAD_H_


#include <ft2build.h>
#include FT_INTERNAL_CFF_TYPES_H
#include "cffparse.h"
#include FT_INTERNAL_CFF_OBJECTS_TYPES_H  /* for CFF_Face */


FT_BEGIN_HEADER

  FT_LOCAL( FT_UShort )
  cff_get_standard_encoding( FT_UInt  charcode );


  FT_LOCAL( FT_String* )
  cff_index_get_string( CFF_Font  font,
                        FT_UInt   element );

  FT_LOCAL( FT_String* )
  cff_index_get_sid_string( CFF_Font  font,
                            FT_UInt   sid );


  FT_LOCAL( FT_Error )
  cff_index_access_element( CFF_Index  idx,
                            FT_UInt    element,
                            FT_Byte**  pbytes,
                            FT_ULong*  pbyte_len );

  FT_LOCAL( void )
  cff_index_forget_element( CFF_Index  idx,
                            FT_Byte**  pbytes );

  FT_LOCAL( FT_String* )
  cff_index_get_name( CFF_Font  font,
                      FT_UInt   element );


  FT_LOCAL( FT_UInt )
  cff_charset_cid_to_gindex( CFF_Charset  charset,
                             FT_UInt      cid );


  FT_LOCAL( FT_Error )
  cff_font_load( FT_Library  library,
                 FT_Stream   stream,
                 FT_Int      face_index,
                 CFF_Font    font,
                 CFF_Face    face,
                 FT_Bool     pure_cff,
                 FT_Bool     cff2 );

  FT_LOCAL( void )
  cff_font_done( CFF_Font  font );


  FT_LOCAL( FT_Error )
  cff_load_private_dict( CFF_Font     font,
                         CFF_SubFont  subfont,
                         FT_UInt      lenNDV,
                         FT_Fixed*    NDV );

  FT_LOCAL( FT_Byte )
  cff_fd_select_get( CFF_FDSelect  fdselect,
                     FT_UInt       glyph_index );

  FT_LOCAL( FT_Bool )
  cff_blend_check_vector( CFF_Blend  blend,
                          FT_UInt    vsindex,
                          FT_UInt    lenNDV,
                          FT_Fixed*  NDV );

  FT_LOCAL( FT_Error )
  cff_blend_build_vector( CFF_Blend  blend,
                          FT_UInt    vsindex,
                          FT_UInt    lenNDV,
                          FT_Fixed*  NDV );

  FT_LOCAL( void )
  cff_blend_clear( CFF_SubFont  subFont );

  FT_LOCAL( FT_Error )
  cff_blend_doBlend( CFF_SubFont  subfont,
                     CFF_Parser   parser,
                     FT_UInt      numBlends );

#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
  FT_LOCAL( FT_Error )
  cff_get_var_blend( CFF_Face     face,
                     FT_UInt     *num_coords,
                     FT_Fixed*   *coords,
                     FT_Fixed*   *normalizedcoords,
                     FT_MM_Var*  *mm_var );

  FT_LOCAL( void )
  cff_done_blend( CFF_Face  face );
#endif


FT_END_HEADER

#endif /* CFFLOAD_H_ */


/* END */
