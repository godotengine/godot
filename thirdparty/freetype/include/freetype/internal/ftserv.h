/***************************************************************************/
/*                                                                         */
/*  ftserv.h                                                               */
/*                                                                         */
/*    The FreeType services (specification only).                          */
/*                                                                         */
/*  Copyright 2003-2016 by                                                 */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/

  /*************************************************************************/
  /*                                                                       */
  /*  Each module can export one or more `services'.  Each service is      */
  /*  identified by a constant string and modeled by a pointer; the latter */
  /*  generally corresponds to a structure containing function pointers.   */
  /*                                                                       */
  /*  Note that a service's data cannot be a mere function pointer because */
  /*  in C it is possible that function pointers might be implemented      */
  /*  differently than data pointers (e.g. 48 bits instead of 32).         */
  /*                                                                       */
  /*************************************************************************/


#ifndef FTSERV_H_
#define FTSERV_H_


FT_BEGIN_HEADER

  /*
   * @macro:
   *   FT_FACE_FIND_SERVICE
   *
   * @description:
   *   This macro is used to look up a service from a face's driver module.
   *
   * @input:
   *   face ::
   *     The source face handle.
   *
   *   id ::
   *     A string describing the service as defined in the service's
   *     header files (e.g. FT_SERVICE_ID_MULTI_MASTERS which expands to
   *     `multi-masters').  It is automatically prefixed with
   *     `FT_SERVICE_ID_'.
   *
   * @output:
   *   ptr ::
   *     A variable that receives the service pointer.  Will be NULL
   *     if not found.
   */
#ifdef __cplusplus

#define FT_FACE_FIND_SERVICE( face, ptr, id )                               \
  FT_BEGIN_STMNT                                                            \
    FT_Module    module = FT_MODULE( FT_FACE( face )->driver );             \
    FT_Pointer   _tmp_  = NULL;                                             \
    FT_Pointer*  _pptr_ = (FT_Pointer*)&(ptr);                              \
                                                                            \
                                                                            \
    if ( module->clazz->get_interface )                                     \
      _tmp_ = module->clazz->get_interface( module, FT_SERVICE_ID_ ## id ); \
    *_pptr_ = _tmp_;                                                        \
  FT_END_STMNT

#else /* !C++ */

#define FT_FACE_FIND_SERVICE( face, ptr, id )                               \
  FT_BEGIN_STMNT                                                            \
    FT_Module   module = FT_MODULE( FT_FACE( face )->driver );              \
    FT_Pointer  _tmp_  = NULL;                                              \
                                                                            \
    if ( module->clazz->get_interface )                                     \
      _tmp_ = module->clazz->get_interface( module, FT_SERVICE_ID_ ## id ); \
    ptr = _tmp_;                                                            \
  FT_END_STMNT

#endif /* !C++ */


  /*
   * @macro:
   *   FT_FACE_FIND_GLOBAL_SERVICE
   *
   * @description:
   *   This macro is used to look up a service from all modules.
   *
   * @input:
   *   face ::
   *     The source face handle.
   *
   *   id ::
   *     A string describing the service as defined in the service's
   *     header files (e.g. FT_SERVICE_ID_MULTI_MASTERS which expands to
   *     `multi-masters').  It is automatically prefixed with
   *     `FT_SERVICE_ID_'.
   *
   * @output:
   *   ptr ::
   *     A variable that receives the service pointer.  Will be NULL
   *     if not found.
   */
#ifdef __cplusplus

#define FT_FACE_FIND_GLOBAL_SERVICE( face, ptr, id )               \
  FT_BEGIN_STMNT                                                   \
    FT_Module    module = FT_MODULE( FT_FACE( face )->driver );    \
    FT_Pointer   _tmp_;                                            \
    FT_Pointer*  _pptr_ = (FT_Pointer*)&(ptr);                     \
                                                                   \
                                                                   \
    _tmp_ = ft_module_get_service( module, FT_SERVICE_ID_ ## id ); \
    *_pptr_ = _tmp_;                                               \
  FT_END_STMNT

#else /* !C++ */

#define FT_FACE_FIND_GLOBAL_SERVICE( face, ptr, id )               \
  FT_BEGIN_STMNT                                                   \
    FT_Module   module = FT_MODULE( FT_FACE( face )->driver );     \
    FT_Pointer  _tmp_;                                             \
                                                                   \
                                                                   \
    _tmp_ = ft_module_get_service( module, FT_SERVICE_ID_ ## id ); \
    ptr   = _tmp_;                                                 \
  FT_END_STMNT

#endif /* !C++ */


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****         S E R V I C E   D E S C R I P T O R S                 *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  /*
   *  The following structure is used to _describe_ a given service
   *  to the library.  This is useful to build simple static service lists.
   */
  typedef struct  FT_ServiceDescRec_
  {
    const char*  serv_id;     /* service name         */
    const void*  serv_data;   /* service pointer/data */

  } FT_ServiceDescRec;

  typedef const FT_ServiceDescRec*  FT_ServiceDesc;


  /*************************************************************************/
  /*                                                                       */
  /* <Macro>                                                               */
  /*    FT_DEFINE_SERVICEDESCREC1                                          */
  /*    FT_DEFINE_SERVICEDESCREC2                                          */
  /*    FT_DEFINE_SERVICEDESCREC3                                          */
  /*    FT_DEFINE_SERVICEDESCREC4                                          */
  /*    FT_DEFINE_SERVICEDESCREC5                                          */
  /*    FT_DEFINE_SERVICEDESCREC6                                          */
  /*    FT_DEFINE_SERVICEDESCREC7                                          */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Used to initialize an array of FT_ServiceDescRec structures.       */
  /*                                                                       */
  /*    When FT_CONFIG_OPTION_PIC is defined a `create' function needs to  */
  /*    be called with a pointer to return an allocated array.  As soon as */
  /*    it is no longer needed, a `destroy' function needs to be called to */
  /*    release that allocation.                                           */
  /*                                                                       */
  /*    These functions should be manually called from the `pic_init' and  */
  /*    `pic_free' functions of your module (see FT_DEFINE_MODULE).        */
  /*                                                                       */
  /*    When FT_CONFIG_OPTION_PIC is not defined the array will be         */
  /*    allocated in the global scope (or the scope where the macro is     */
  /*    used).                                                             */
  /*                                                                       */
#ifndef FT_CONFIG_OPTION_PIC

#define FT_DEFINE_SERVICEDESCREC1( class_,                                  \
                                   serv_id_1, serv_data_1 )                 \
  static const FT_ServiceDescRec  class_[] =                                \
  {                                                                         \
    { serv_id_1, serv_data_1 },                                             \
    { NULL, NULL }                                                          \
  };

#define FT_DEFINE_SERVICEDESCREC2( class_,                                  \
                                   serv_id_1, serv_data_1,                  \
                                   serv_id_2, serv_data_2 )                 \
  static const FT_ServiceDescRec  class_[] =                                \
  {                                                                         \
    { serv_id_1, serv_data_1 },                                             \
    { serv_id_2, serv_data_2 },                                             \
    { NULL, NULL }                                                          \
  };

#define FT_DEFINE_SERVICEDESCREC3( class_,                                  \
                                   serv_id_1, serv_data_1,                  \
                                   serv_id_2, serv_data_2,                  \
                                   serv_id_3, serv_data_3 )                 \
  static const FT_ServiceDescRec  class_[] =                                \
  {                                                                         \
    { serv_id_1, serv_data_1 },                                             \
    { serv_id_2, serv_data_2 },                                             \
    { serv_id_3, serv_data_3 },                                             \
    { NULL, NULL }                                                          \
  };

#define FT_DEFINE_SERVICEDESCREC4( class_,                                  \
                                   serv_id_1, serv_data_1,                  \
                                   serv_id_2, serv_data_2,                  \
                                   serv_id_3, serv_data_3,                  \
                                   serv_id_4, serv_data_4 )                 \
  static const FT_ServiceDescRec  class_[] =                                \
  {                                                                         \
    { serv_id_1, serv_data_1 },                                             \
    { serv_id_2, serv_data_2 },                                             \
    { serv_id_3, serv_data_3 },                                             \
    { serv_id_4, serv_data_4 },                                             \
    { NULL, NULL }                                                          \
  };

#define FT_DEFINE_SERVICEDESCREC5( class_,                                  \
                                   serv_id_1, serv_data_1,                  \
                                   serv_id_2, serv_data_2,                  \
                                   serv_id_3, serv_data_3,                  \
                                   serv_id_4, serv_data_4,                  \
                                   serv_id_5, serv_data_5 )                 \
  static const FT_ServiceDescRec  class_[] =                                \
  {                                                                         \
    { serv_id_1, serv_data_1 },                                             \
    { serv_id_2, serv_data_2 },                                             \
    { serv_id_3, serv_data_3 },                                             \
    { serv_id_4, serv_data_4 },                                             \
    { serv_id_5, serv_data_5 },                                             \
    { NULL, NULL }                                                          \
  };

#define FT_DEFINE_SERVICEDESCREC6( class_,                                  \
                                   serv_id_1, serv_data_1,                  \
                                   serv_id_2, serv_data_2,                  \
                                   serv_id_3, serv_data_3,                  \
                                   serv_id_4, serv_data_4,                  \
                                   serv_id_5, serv_data_5,                  \
                                   serv_id_6, serv_data_6 )                 \
  static const FT_ServiceDescRec  class_[] =                                \
  {                                                                         \
    { serv_id_1, serv_data_1 },                                             \
    { serv_id_2, serv_data_2 },                                             \
    { serv_id_3, serv_data_3 },                                             \
    { serv_id_4, serv_data_4 },                                             \
    { serv_id_5, serv_data_5 },                                             \
    { serv_id_6, serv_data_6 },                                             \
    { NULL, NULL }                                                          \
  };

#define FT_DEFINE_SERVICEDESCREC7( class_,                                  \
                                   serv_id_1, serv_data_1,                  \
                                   serv_id_2, serv_data_2,                  \
                                   serv_id_3, serv_data_3,                  \
                                   serv_id_4, serv_data_4,                  \
                                   serv_id_5, serv_data_5,                  \
                                   serv_id_6, serv_data_6,                  \
                                   serv_id_7, serv_data_7 )                 \
  static const FT_ServiceDescRec  class_[] =                                \
  {                                                                         \
    { serv_id_1, serv_data_1 },                                             \
    { serv_id_2, serv_data_2 },                                             \
    { serv_id_3, serv_data_3 },                                             \
    { serv_id_4, serv_data_4 },                                             \
    { serv_id_5, serv_data_5 },                                             \
    { serv_id_6, serv_data_6 },                                             \
    { serv_id_7, serv_data_7 },                                             \
    { NULL, NULL }                                                          \
  };

#else /* FT_CONFIG_OPTION_PIC */

#define FT_DEFINE_SERVICEDESCREC1( class_,                                  \
                                   serv_id_1, serv_data_1 )                 \
  void                                                                      \
  FT_Destroy_Class_ ## class_( FT_Library          library,                 \
                               FT_ServiceDescRec*  clazz )                  \
  {                                                                         \
    FT_Memory  memory = library->memory;                                    \
                                                                            \
                                                                            \
    if ( clazz )                                                            \
      FT_FREE( clazz );                                                     \
  }                                                                         \
                                                                            \
  FT_Error                                                                  \
  FT_Create_Class_ ## class_( FT_Library           library,                 \
                              FT_ServiceDescRec**  output_class )           \
  {                                                                         \
    FT_ServiceDescRec*  clazz  = NULL;                                      \
    FT_Error            error;                                              \
    FT_Memory           memory = library->memory;                           \
                                                                            \
                                                                            \
    if ( FT_ALLOC( clazz, sizeof ( *clazz ) * 2 ) )                         \
      return error;                                                         \
                                                                            \
    clazz[0].serv_id   = serv_id_1;                                         \
    clazz[0].serv_data = serv_data_1;                                       \
    clazz[1].serv_id   = NULL;                                              \
    clazz[1].serv_data = NULL;                                              \
                                                                            \
    *output_class = clazz;                                                  \
                                                                            \
    return FT_Err_Ok;                                                       \
  }

#define FT_DEFINE_SERVICEDESCREC2( class_,                                  \
                                   serv_id_1, serv_data_1,                  \
                                   serv_id_2, serv_data_2 )                 \
  void                                                                      \
  FT_Destroy_Class_ ## class_( FT_Library          library,                 \
                               FT_ServiceDescRec*  clazz )                  \
  {                                                                         \
    FT_Memory  memory = library->memory;                                    \
                                                                            \
                                                                            \
    if ( clazz )                                                            \
      FT_FREE( clazz );                                                     \
  }                                                                         \
                                                                            \
  FT_Error                                                                  \
  FT_Create_Class_ ## class_( FT_Library           library,                 \
                              FT_ServiceDescRec**  output_class )           \
  {                                                                         \
    FT_ServiceDescRec*  clazz  = NULL;                                      \
    FT_Error            error;                                              \
    FT_Memory           memory = library->memory;                           \
                                                                            \
                                                                            \
    if ( FT_ALLOC( clazz, sizeof ( *clazz ) * 3 ) )                         \
      return error;                                                         \
                                                                            \
    clazz[0].serv_id   = serv_id_1;                                         \
    clazz[0].serv_data = serv_data_1;                                       \
    clazz[1].serv_id   = serv_id_2;                                         \
    clazz[1].serv_data = serv_data_2;                                       \
    clazz[2].serv_id   = NULL;                                              \
    clazz[2].serv_data = NULL;                                              \
                                                                            \
    *output_class = clazz;                                                  \
                                                                            \
    return FT_Err_Ok;                                                       \
  }

#define FT_DEFINE_SERVICEDESCREC3( class_,                                  \
                                   serv_id_1, serv_data_1,                  \
                                   serv_id_2, serv_data_2,                  \
                                   serv_id_3, serv_data_3 )                 \
  void                                                                      \
  FT_Destroy_Class_ ## class_( FT_Library          library,                 \
                               FT_ServiceDescRec*  clazz )                  \
  {                                                                         \
    FT_Memory  memory = library->memory;                                    \
                                                                            \
                                                                            \
    if ( clazz )                                                            \
      FT_FREE( clazz );                                                     \
  }                                                                         \
                                                                            \
  FT_Error                                                                  \
  FT_Create_Class_ ## class_( FT_Library           library,                 \
                              FT_ServiceDescRec**  output_class )           \
  {                                                                         \
    FT_ServiceDescRec*  clazz  = NULL;                                      \
    FT_Error            error;                                              \
    FT_Memory           memory = library->memory;                           \
                                                                            \
                                                                            \
    if ( FT_ALLOC( clazz, sizeof ( *clazz ) * 4 ) )                         \
      return error;                                                         \
                                                                            \
    clazz[0].serv_id   = serv_id_1;                                         \
    clazz[0].serv_data = serv_data_1;                                       \
    clazz[1].serv_id   = serv_id_2;                                         \
    clazz[1].serv_data = serv_data_2;                                       \
    clazz[2].serv_id   = serv_id_3;                                         \
    clazz[2].serv_data = serv_data_3;                                       \
    clazz[3].serv_id   = NULL;                                              \
    clazz[3].serv_data = NULL;                                              \
                                                                            \
    *output_class = clazz;                                                  \
                                                                            \
    return FT_Err_Ok;                                                       \
  }

#define FT_DEFINE_SERVICEDESCREC4( class_,                                  \
                                   serv_id_1, serv_data_1,                  \
                                   serv_id_2, serv_data_2,                  \
                                   serv_id_3, serv_data_3,                  \
                                   serv_id_4, serv_data_4 )                 \
  void                                                                      \
  FT_Destroy_Class_ ## class_( FT_Library          library,                 \
                               FT_ServiceDescRec*  clazz )                  \
  {                                                                         \
    FT_Memory  memory = library->memory;                                    \
                                                                            \
                                                                            \
    if ( clazz )                                                            \
      FT_FREE( clazz );                                                     \
  }                                                                         \
                                                                            \
  FT_Error                                                                  \
  FT_Create_Class_ ## class_( FT_Library           library,                 \
                              FT_ServiceDescRec**  output_class )           \
  {                                                                         \
    FT_ServiceDescRec*  clazz  = NULL;                                      \
    FT_Error            error;                                              \
    FT_Memory           memory = library->memory;                           \
                                                                            \
                                                                            \
    if ( FT_ALLOC( clazz, sizeof ( *clazz ) * 5 ) )                         \
      return error;                                                         \
                                                                            \
    clazz[0].serv_id   = serv_id_1;                                         \
    clazz[0].serv_data = serv_data_1;                                       \
    clazz[1].serv_id   = serv_id_2;                                         \
    clazz[1].serv_data = serv_data_2;                                       \
    clazz[2].serv_id   = serv_id_3;                                         \
    clazz[2].serv_data = serv_data_3;                                       \
    clazz[3].serv_id   = serv_id_4;                                         \
    clazz[3].serv_data = serv_data_4;                                       \
    clazz[4].serv_id   = NULL;                                              \
    clazz[4].serv_data = NULL;                                              \
                                                                            \
    *output_class = clazz;                                                  \
                                                                            \
    return FT_Err_Ok;                                                       \
  }

#define FT_DEFINE_SERVICEDESCREC5( class_,                                  \
                                   serv_id_1, serv_data_1,                  \
                                   serv_id_2, serv_data_2,                  \
                                   serv_id_3, serv_data_3,                  \
                                   serv_id_4, serv_data_4,                  \
                                   serv_id_5, serv_data_5 )                 \
  void                                                                      \
  FT_Destroy_Class_ ## class_( FT_Library          library,                 \
                               FT_ServiceDescRec*  clazz )                  \
  {                                                                         \
    FT_Memory  memory = library->memory;                                    \
                                                                            \
                                                                            \
    if ( clazz )                                                            \
      FT_FREE( clazz );                                                     \
  }                                                                         \
                                                                            \
  FT_Error                                                                  \
  FT_Create_Class_ ## class_( FT_Library           library,                 \
                              FT_ServiceDescRec**  output_class )           \
  {                                                                         \
    FT_ServiceDescRec*  clazz  = NULL;                                      \
    FT_Error            error;                                              \
    FT_Memory           memory = library->memory;                           \
                                                                            \
                                                                            \
    if ( FT_ALLOC( clazz, sizeof ( *clazz ) * 6 ) )                         \
      return error;                                                         \
                                                                            \
    clazz[0].serv_id   = serv_id_1;                                         \
    clazz[0].serv_data = serv_data_1;                                       \
    clazz[1].serv_id   = serv_id_2;                                         \
    clazz[1].serv_data = serv_data_2;                                       \
    clazz[2].serv_id   = serv_id_3;                                         \
    clazz[2].serv_data = serv_data_3;                                       \
    clazz[3].serv_id   = serv_id_4;                                         \
    clazz[3].serv_data = serv_data_4;                                       \
    clazz[4].serv_id   = serv_id_5;                                         \
    clazz[4].serv_data = serv_data_5;                                       \
    clazz[5].serv_id   = NULL;                                              \
    clazz[5].serv_data = NULL;                                              \
                                                                            \
    *output_class = clazz;                                                  \
                                                                            \
    return FT_Err_Ok;                                                       \
  }

#define FT_DEFINE_SERVICEDESCREC6( class_,                                  \
                                   serv_id_1, serv_data_1,                  \
                                   serv_id_2, serv_data_2,                  \
                                   serv_id_3, serv_data_3,                  \
                                   serv_id_4, serv_data_4,                  \
                                   serv_id_5, serv_data_5,                  \
                                   serv_id_6, serv_data_6 )                 \
  void                                                                      \
  FT_Destroy_Class_ ## class_( FT_Library          library,                 \
                               FT_ServiceDescRec*  clazz )                  \
  {                                                                         \
    FT_Memory  memory = library->memory;                                    \
                                                                            \
                                                                            \
    if ( clazz )                                                            \
      FT_FREE( clazz );                                                     \
  }                                                                         \
                                                                            \
  FT_Error                                                                  \
  FT_Create_Class_ ## class_( FT_Library           library,                 \
                              FT_ServiceDescRec**  output_class)            \
  {                                                                         \
    FT_ServiceDescRec*  clazz  = NULL;                                      \
    FT_Error            error;                                              \
    FT_Memory           memory = library->memory;                           \
                                                                            \
                                                                            \
    if ( FT_ALLOC( clazz, sizeof ( *clazz ) * 7 ) )                         \
      return error;                                                         \
                                                                            \
    clazz[0].serv_id   = serv_id_1;                                         \
    clazz[0].serv_data = serv_data_1;                                       \
    clazz[1].serv_id   = serv_id_2;                                         \
    clazz[1].serv_data = serv_data_2;                                       \
    clazz[2].serv_id   = serv_id_3;                                         \
    clazz[2].serv_data = serv_data_3;                                       \
    clazz[3].serv_id   = serv_id_4;                                         \
    clazz[3].serv_data = serv_data_4;                                       \
    clazz[4].serv_id   = serv_id_5;                                         \
    clazz[4].serv_data = serv_data_5;                                       \
    clazz[5].serv_id   = serv_id_6;                                         \
    clazz[5].serv_data = serv_data_6;                                       \
    clazz[6].serv_id   = NULL;                                              \
    clazz[6].serv_data = NULL;                                              \
                                                                            \
    *output_class = clazz;                                                  \
                                                                            \
    return FT_Err_Ok;                                                       \
  }

#define FT_DEFINE_SERVICEDESCREC7( class_,                                  \
                                   serv_id_1, serv_data_1,                  \
                                   serv_id_2, serv_data_2,                  \
                                   serv_id_3, serv_data_3,                  \
                                   serv_id_4, serv_data_4,                  \
                                   serv_id_5, serv_data_5,                  \
                                   serv_id_6, serv_data_6,                  \
                                   serv_id_7, serv_data_7 )                 \
  void                                                                      \
  FT_Destroy_Class_ ## class_( FT_Library          library,                 \
                               FT_ServiceDescRec*  clazz )                  \
  {                                                                         \
    FT_Memory  memory = library->memory;                                    \
                                                                            \
                                                                            \
    if ( clazz )                                                            \
      FT_FREE( clazz );                                                     \
  }                                                                         \
                                                                            \
  FT_Error                                                                  \
  FT_Create_Class_ ## class_( FT_Library           library,                 \
                              FT_ServiceDescRec**  output_class)            \
  {                                                                         \
    FT_ServiceDescRec*  clazz  = NULL;                                      \
    FT_Error            error;                                              \
    FT_Memory           memory = library->memory;                           \
                                                                            \
                                                                            \
    if ( FT_ALLOC( clazz, sizeof ( *clazz ) * 8 ) )                         \
      return error;                                                         \
                                                                            \
    clazz[0].serv_id   = serv_id_1;                                         \
    clazz[0].serv_data = serv_data_1;                                       \
    clazz[1].serv_id   = serv_id_2;                                         \
    clazz[1].serv_data = serv_data_2;                                       \
    clazz[2].serv_id   = serv_id_3;                                         \
    clazz[2].serv_data = serv_data_3;                                       \
    clazz[3].serv_id   = serv_id_4;                                         \
    clazz[3].serv_data = serv_data_4;                                       \
    clazz[4].serv_id   = serv_id_5;                                         \
    clazz[4].serv_data = serv_data_5;                                       \
    clazz[5].serv_id   = serv_id_6;                                         \
    clazz[5].serv_data = serv_data_6;                                       \
    clazz[6].serv_id   = serv_id_7;                                         \
    clazz[6].serv_data = serv_data_7;                                       \
    clazz[7].serv_id   = NULL;                                              \
    clazz[7].serv_data = NULL;                                              \
                                                                            \
    *output_class = clazz;                                                  \
                                                                            \
    return FT_Err_Ok;                                                       \
  }

#endif /* FT_CONFIG_OPTION_PIC */


  /*
   *  Parse a list of FT_ServiceDescRec descriptors and look for
   *  a specific service by ID.  Note that the last element in the
   *  array must be { NULL, NULL }, and that the function should
   *  return NULL if the service isn't available.
   *
   *  This function can be used by modules to implement their
   *  `get_service' method.
   */
  FT_BASE( FT_Pointer )
  ft_service_list_lookup( FT_ServiceDesc  service_descriptors,
                          const char*     service_id );


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****             S E R V I C E S   C A C H E                       *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  /*
   *  This structure is used to store a cache for several frequently used
   *  services.  It is the type of `face->internal->services'.  You
   *  should only use FT_FACE_LOOKUP_SERVICE to access it.
   *
   *  All fields should have the type FT_Pointer to relax compilation
   *  dependencies.  We assume the developer isn't completely stupid.
   *
   *  Each field must be named `service_XXXX' where `XXX' corresponds to
   *  the correct FT_SERVICE_ID_XXXX macro.  See the definition of
   *  FT_FACE_LOOKUP_SERVICE below how this is implemented.
   *
   */
  typedef struct  FT_ServiceCacheRec_
  {
    FT_Pointer  service_POSTSCRIPT_FONT_NAME;
    FT_Pointer  service_MULTI_MASTERS;
    FT_Pointer  service_GLYPH_DICT;
    FT_Pointer  service_PFR_METRICS;
    FT_Pointer  service_WINFNT;

  } FT_ServiceCacheRec, *FT_ServiceCache;


  /*
   *  A magic number used within the services cache.
   */

  /* ensure that value `1' has the same width as a pointer */
#define FT_SERVICE_UNAVAILABLE  ((FT_Pointer)~(FT_PtrDist)1)


  /*
   * @macro:
   *   FT_FACE_LOOKUP_SERVICE
   *
   * @description:
   *   This macro is used to lookup a service from a face's driver module
   *   using its cache.
   *
   * @input:
   *   face::
   *     The source face handle containing the cache.
   *
   *   field ::
   *     The field name in the cache.
   *
   *   id ::
   *     The service ID.
   *
   * @output:
   *   ptr ::
   *     A variable receiving the service data.  NULL if not available.
   */
#ifdef __cplusplus

#define FT_FACE_LOOKUP_SERVICE( face, ptr, id )                \
  FT_BEGIN_STMNT                                               \
    FT_Pointer   svc;                                          \
    FT_Pointer*  Pptr = (FT_Pointer*)&(ptr);                   \
                                                               \
                                                               \
    svc = FT_FACE( face )->internal->services. service_ ## id; \
    if ( svc == FT_SERVICE_UNAVAILABLE )                       \
      svc = NULL;                                              \
    else if ( svc == NULL )                                    \
    {                                                          \
      FT_FACE_FIND_SERVICE( face, svc, id );                   \
                                                               \
      FT_FACE( face )->internal->services. service_ ## id =    \
        (FT_Pointer)( svc != NULL ? svc                        \
                                  : FT_SERVICE_UNAVAILABLE );  \
    }                                                          \
    *Pptr = svc;                                               \
  FT_END_STMNT

#else /* !C++ */

#define FT_FACE_LOOKUP_SERVICE( face, ptr, id )                \
  FT_BEGIN_STMNT                                               \
    FT_Pointer  svc;                                           \
                                                               \
                                                               \
    svc = FT_FACE( face )->internal->services. service_ ## id; \
    if ( svc == FT_SERVICE_UNAVAILABLE )                       \
      svc = NULL;                                              \
    else if ( svc == NULL )                                    \
    {                                                          \
      FT_FACE_FIND_SERVICE( face, svc, id );                   \
                                                               \
      FT_FACE( face )->internal->services. service_ ## id =    \
        (FT_Pointer)( svc != NULL ? svc                        \
                                  : FT_SERVICE_UNAVAILABLE );  \
    }                                                          \
    ptr = svc;                                                 \
  FT_END_STMNT

#endif /* !C++ */

  /*
   *  A macro used to define new service structure types.
   */

#define FT_DEFINE_SERVICE( name )            \
  typedef struct FT_Service_ ## name ## Rec_ \
    FT_Service_ ## name ## Rec ;             \
  typedef struct FT_Service_ ## name ## Rec_ \
    const * FT_Service_ ## name ;            \
  struct FT_Service_ ## name ## Rec_

  /* */

  /*
   *  The header files containing the services.
   */

#define FT_SERVICE_BDF_H                <freetype/internal/services/svbdf.h>
#define FT_SERVICE_CID_H                <freetype/internal/services/svcid.h>
#define FT_SERVICE_GLYPH_DICT_H         <freetype/internal/services/svgldict.h>
#define FT_SERVICE_GX_VALIDATE_H        <freetype/internal/services/svgxval.h>
#define FT_SERVICE_KERNING_H            <freetype/internal/services/svkern.h>
#define FT_SERVICE_MULTIPLE_MASTERS_H   <freetype/internal/services/svmm.h>
#define FT_SERVICE_OPENTYPE_VALIDATE_H  <freetype/internal/services/svotval.h>
#define FT_SERVICE_PFR_H                <freetype/internal/services/svpfr.h>
#define FT_SERVICE_POSTSCRIPT_CMAPS_H   <freetype/internal/services/svpscmap.h>
#define FT_SERVICE_POSTSCRIPT_INFO_H    <freetype/internal/services/svpsinfo.h>
#define FT_SERVICE_POSTSCRIPT_NAME_H    <freetype/internal/services/svpostnm.h>
#define FT_SERVICE_PROPERTIES_H         <freetype/internal/services/svprop.h>
#define FT_SERVICE_SFNT_H               <freetype/internal/services/svsfnt.h>
#define FT_SERVICE_TRUETYPE_ENGINE_H    <freetype/internal/services/svtteng.h>
#define FT_SERVICE_TT_CMAP_H            <freetype/internal/services/svttcmap.h>
#define FT_SERVICE_WINFNT_H             <freetype/internal/services/svwinfnt.h>
#define FT_SERVICE_FONT_FORMAT_H        <freetype/internal/services/svfntfmt.h>
#define FT_SERVICE_TRUETYPE_GLYF_H      <freetype/internal/services/svttglyf.h>

 /* */

FT_END_HEADER

#endif /* FTSERV_H_ */


/* END */
