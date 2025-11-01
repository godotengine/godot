/****************************************************************************
 *
 * svcfftl.h
 *
 *   The FreeType CFF tables loader service (specification).
 *
 * Copyright (C) 2017-2025 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#ifndef SVCFFTL_H_
#define SVCFFTL_H_

#include <freetype/internal/ftserv.h>
#include <freetype/internal/cfftypes.h>


FT_BEGIN_HEADER


#define FT_SERVICE_ID_CFF_LOAD  "cff-load"


  typedef FT_UShort
  (*FT_Get_Standard_Encoding_Func)( FT_UInt  charcode );

  typedef FT_Error
  (*FT_Load_Private_Dict_Func)( CFF_Font     font,
                                CFF_SubFont  subfont,
                                FT_UInt      lenNDV,
                                FT_Fixed*    NDV );

  typedef FT_Byte
  (*FT_FD_Select_Get_Func)( CFF_FDSelect  fdselect,
                            FT_UInt       glyph_index );

  typedef FT_Bool
  (*FT_Blend_Check_Vector_Func)( CFF_Blend  blend,
                                 FT_UInt    vsindex,
                                 FT_UInt    lenNDV,
                                 FT_Fixed*  NDV );

  typedef FT_Error
  (*FT_Blend_Build_Vector_Func)( CFF_Blend  blend,
                                 FT_UInt    vsindex,
                                 FT_UInt    lenNDV,
                                 FT_Fixed*  NDV );


  FT_DEFINE_SERVICE( CFFLoad )
  {
    FT_Get_Standard_Encoding_Func  get_standard_encoding;
    FT_Load_Private_Dict_Func      load_private_dict;
    FT_FD_Select_Get_Func          fd_select_get;
    FT_Blend_Check_Vector_Func     blend_check_vector;
    FT_Blend_Build_Vector_Func     blend_build_vector;
  };


#define FT_DEFINE_SERVICE_CFFLOADREC( class_,                  \
                                      get_standard_encoding_,  \
                                      load_private_dict_,      \
                                      fd_select_get_,          \
                                      blend_check_vector_,     \
                                      blend_build_vector_ )    \
  static const FT_Service_CFFLoadRec  class_ =                 \
  {                                                            \
    get_standard_encoding_,                                    \
    load_private_dict_,                                        \
    fd_select_get_,                                            \
    blend_check_vector_,                                       \
    blend_build_vector_                                        \
  };


FT_END_HEADER


#endif


/* END */
