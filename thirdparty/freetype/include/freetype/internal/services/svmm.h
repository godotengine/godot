/***************************************************************************/
/*                                                                         */
/*  svmm.h                                                                 */
/*                                                                         */
/*    The FreeType Multiple Masters and GX var services (specification).   */
/*                                                                         */
/*  Copyright 2003-2017 by                                                 */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


#ifndef SVMM_H_
#define SVMM_H_

#include FT_INTERNAL_SERVICE_H


FT_BEGIN_HEADER


  /*
   *  A service used to manage multiple-masters data in a given face.
   *
   *  See the related APIs in `ftmm.h' (FT_MULTIPLE_MASTERS_H).
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

  typedef FT_Error
  (*FT_Set_Var_Design_Func)( FT_Face    face,
                             FT_UInt    num_coords,
                             FT_Fixed*  coords );

  typedef FT_Error
  (*FT_Set_MM_Blend_Func)( FT_Face   face,
                           FT_UInt   num_coords,
                           FT_Long*  coords );

  typedef FT_Error
  (*FT_Get_Var_Design_Func)( FT_Face    face,
                             FT_UInt    num_coords,
                             FT_Fixed*  coords );

  typedef FT_Error
  (*FT_Get_MM_Blend_Func)( FT_Face   face,
                           FT_UInt   num_coords,
                           FT_Long*  coords );

  typedef FT_Error
  (*FT_Get_Var_Blend_Func)( FT_Face      face,
                            FT_UInt     *num_coords,
                            FT_Fixed*   *coords,
                            FT_Fixed*   *normalizedcoords,
                            FT_MM_Var*  *mm_var );

  typedef void
  (*FT_Done_Blend_Func)( FT_Face );


  FT_DEFINE_SERVICE( MultiMasters )
  {
    FT_Get_MM_Func          get_mm;
    FT_Set_MM_Design_Func   set_mm_design;
    FT_Set_MM_Blend_Func    set_mm_blend;
    FT_Get_MM_Blend_Func    get_mm_blend;
    FT_Get_MM_Var_Func      get_mm_var;
    FT_Set_Var_Design_Func  set_var_design;
    FT_Get_Var_Design_Func  get_var_design;

    /* for internal use; only needed for code sharing between modules */
    FT_Get_Var_Blend_Func   get_var_blend;
    FT_Done_Blend_Func      done_blend;
  };


#ifndef FT_CONFIG_OPTION_PIC

#define FT_DEFINE_SERVICE_MULTIMASTERSREC( class_,           \
                                           get_mm_,          \
                                           set_mm_design_,   \
                                           set_mm_blend_,    \
                                           get_mm_blend_,    \
                                           get_mm_var_,      \
                                           set_var_design_,  \
                                           get_var_design_,  \
                                           get_var_blend_,   \
                                           done_blend_     ) \
  static const FT_Service_MultiMastersRec  class_ =          \
  {                                                          \
    get_mm_,                                                 \
    set_mm_design_,                                          \
    set_mm_blend_,                                           \
    get_mm_blend_,                                           \
    get_mm_var_,                                             \
    set_var_design_,                                         \
    get_var_design_,                                         \
    get_var_blend_,                                          \
    done_blend_                                              \
  };

#else /* FT_CONFIG_OPTION_PIC */

#define FT_DEFINE_SERVICE_MULTIMASTERSREC( class_,               \
                                           get_mm_,              \
                                           set_mm_design_,       \
                                           set_mm_blend_,        \
                                           get_mm_blend_,        \
                                           get_mm_var_,          \
                                           set_var_design_,      \
                                           get_var_design_,      \
                                           get_var_blend_,       \
                                           done_blend_ )         \
  void                                                           \
  FT_Init_Class_ ## class_( FT_Service_MultiMastersRec*  clazz ) \
  {                                                              \
    clazz->get_mm         = get_mm_;                             \
    clazz->set_mm_design  = set_mm_design_;                      \
    clazz->set_mm_blend   = set_mm_blend_;                       \
    clazz->get_mm_blend   = get_mm_blend_;                       \
    clazz->get_mm_var     = get_mm_var_;                         \
    clazz->set_var_design = set_var_design_;                     \
    clazz->get_var_design = get_var_design_;                     \
    clazz->get_var_blend  = get_var_blend_;                      \
    clazz->done_blend     = done_blend_;                         \
  }

#endif /* FT_CONFIG_OPTION_PIC */

  /* */


FT_END_HEADER

#endif /* SVMM_H_ */


/* END */
