/****************************************************************************
 *
 * svmm.h
 *
 *   The FreeType Multiple Masters and GX var services (specification).
 *
 * Copyright (C) 2003-2024 by
 * David Turner, Robert Wilhelm, Werner Lemberg, and Dominik Röttsches.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#ifndef SVMM_H_
#define SVMM_H_

#include <freetype/ftmm.h>
#include <freetype/internal/ftserv.h>
#include <freetype/internal/ftmmtypes.h>


FT_BEGIN_HEADER


  /*
   * A service used to manage multiple-masters data in a given face.
   *
   * See the related APIs in `ftmm.h' (FT_MULTIPLE_MASTERS_H).
   *
   */

#define FT_SERVICE_ID_MULTI_MASTERS  "multi-masters"


  typedef FT_Error
  (*FT_Get_MM_Func)( FT_Face           face,
                     FT_Multi_Master*  master );

  typedef FT_Error
  (*FT_Get_MM_Var_Func)( FT_Face      face,
                         FT_MM_Var*  *master );

  typedef FT_Error
  (*FT_Set_MM_Design_Func)( FT_Face   face,
                            FT_UInt   num_coords,
                            FT_Long*  coords );

  /* use return value -1 to indicate that the new coordinates  */
  /* are equal to the current ones; no changes are thus needed */
  typedef FT_Error
  (*FT_Set_Var_Design_Func)( FT_Face    face,
                             FT_UInt    num_coords,
                             FT_Fixed*  coords );

  /* use return value -1 to indicate that the new coordinates  */
  /* are equal to the current ones; no changes are thus needed */
  typedef FT_Error
  (*FT_Set_MM_Blend_Func)( FT_Face    face,
                           FT_UInt    num_coords,
                           FT_Fixed*  coords );

  typedef FT_Error
  (*FT_Get_Var_Design_Func)( FT_Face    face,
                             FT_UInt    num_coords,
                             FT_Fixed*  coords );

  typedef FT_Error
  (*FT_Set_Named_Instance_Func)( FT_Face  face,
                                 FT_UInt  instance_index );

  typedef FT_Error
  (*FT_Get_Default_Named_Instance_Func)( FT_Face   face,
                                         FT_UInt  *instance_index );

  typedef FT_Error
  (*FT_Get_MM_Blend_Func)( FT_Face    face,
                           FT_UInt    num_coords,
                           FT_Fixed*  coords );

  typedef FT_Error
  (*FT_Get_Var_Blend_Func)( FT_Face      face,
                            FT_UInt     *num_coords,
                            FT_Fixed*   *coords,
                            FT_Fixed*   *normalizedcoords,
                            FT_MM_Var*  *mm_var );

  typedef void
  (*FT_Done_Blend_Func)( FT_Face  face );

  typedef FT_Error
  (*FT_Set_MM_WeightVector_Func)( FT_Face    face,
                                  FT_UInt    len,
                                  FT_Fixed*  weight_vector );

  typedef FT_Error
  (*FT_Get_MM_WeightVector_Func)( FT_Face    face,
                                  FT_UInt*   len,
                                  FT_Fixed*  weight_vector );

  typedef void
  (*FT_Construct_PS_Name_Func)( FT_Face  face );

  typedef FT_Error
  (*FT_Var_Load_Delta_Set_Idx_Map_Func)( FT_Face            face,
                                         FT_ULong           offset,
                                         GX_DeltaSetIdxMap  map,
                                         GX_ItemVarStore    itemStore,
                                         FT_ULong           table_len );

  typedef FT_Error
  (*FT_Var_Load_Item_Var_Store_Func)( FT_Face          face,
                                      FT_ULong         offset,
                                      GX_ItemVarStore  itemStore );

  typedef FT_ItemVarDelta
  (*FT_Var_Get_Item_Delta_Func)( FT_Face          face,
                                 GX_ItemVarStore  itemStore,
                                 FT_UInt          outerIndex,
                                 FT_UInt          innerIndex );

  typedef void
  (*FT_Var_Done_Item_Var_Store_Func)( FT_Face          face,
                                      GX_ItemVarStore  itemStore );

  typedef void
  (*FT_Var_Done_Delta_Set_Idx_Map_Func)( FT_Face            face,
                                         GX_DeltaSetIdxMap  deltaSetIdxMap );


  FT_DEFINE_SERVICE( MultiMasters )
  {
    FT_Get_MM_Func                        get_mm;
    FT_Set_MM_Design_Func                 set_mm_design;
    FT_Set_MM_Blend_Func                  set_mm_blend;
    FT_Get_MM_Blend_Func                  get_mm_blend;
    FT_Get_MM_Var_Func                    get_mm_var;
    FT_Set_Var_Design_Func                set_var_design;
    FT_Get_Var_Design_Func                get_var_design;
    FT_Set_Named_Instance_Func            set_named_instance;
    FT_Get_Default_Named_Instance_Func    get_default_named_instance;
    FT_Set_MM_WeightVector_Func           set_mm_weightvector;
    FT_Get_MM_WeightVector_Func           get_mm_weightvector;

    /* for internal use; only needed for code sharing between modules */
    FT_Construct_PS_Name_Func             construct_ps_name;
    FT_Var_Load_Delta_Set_Idx_Map_Func    load_delta_set_idx_map;
    FT_Var_Load_Item_Var_Store_Func       load_item_var_store;
    FT_Var_Get_Item_Delta_Func            get_item_delta;
    FT_Var_Done_Item_Var_Store_Func       done_item_var_store;
    FT_Var_Done_Delta_Set_Idx_Map_Func    done_delta_set_idx_map;
    FT_Get_Var_Blend_Func                 get_var_blend;
    FT_Done_Blend_Func                    done_blend;
  };


#define FT_DEFINE_SERVICE_MULTIMASTERSREC( class_,                      \
                                           get_mm_,                     \
                                           set_mm_design_,              \
                                           set_mm_blend_,               \
                                           get_mm_blend_,               \
                                           get_mm_var_,                 \
                                           set_var_design_,             \
                                           get_var_design_,             \
                                           set_named_instance_,         \
                                           get_default_named_instance_, \
                                           set_mm_weightvector_,        \
                                           get_mm_weightvector_,        \
                                                                        \
                                           construct_ps_name_,          \
                                           load_delta_set_idx_map_,     \
                                           load_item_var_store_,        \
                                           get_item_delta_,             \
                                           done_item_var_store_,        \
                                           done_delta_set_idx_map_,     \
                                           get_var_blend_,              \
                                           done_blend_ )                \
  static const FT_Service_MultiMastersRec  class_ =                     \
  {                                                                     \
    get_mm_,                                                            \
    set_mm_design_,                                                     \
    set_mm_blend_,                                                      \
    get_mm_blend_,                                                      \
    get_mm_var_,                                                        \
    set_var_design_,                                                    \
    get_var_design_,                                                    \
    set_named_instance_,                                                \
    get_default_named_instance_,                                        \
    set_mm_weightvector_,                                               \
    get_mm_weightvector_,                                               \
                                                                        \
    construct_ps_name_,                                                 \
    load_delta_set_idx_map_,                                            \
    load_item_var_store_,                                               \
    get_item_delta_,                                                    \
    done_item_var_store_,                                               \
    done_delta_set_idx_map_,                                            \
    get_var_blend_,                                                     \
    done_blend_                                                         \
  };

  /* */


FT_END_HEADER

#endif /* SVMM_H_ */


/* END */
