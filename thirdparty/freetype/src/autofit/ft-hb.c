/****************************************************************************
 *
 * ft-hb.c
 *
 *   FreeType-HarfBuzz bridge (body).
 *
 * Copyright (C) 2025 by
 * Behdad Esfahbod.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#if !defined( _WIN32 ) && !defined( _GNU_SOURCE )
#  define _GNU_SOURCE  1  /* for RTLD_DEFAULT */
#endif

#include <freetype/freetype.h>
#include <freetype/internal/ftmemory.h>

#include "afglobal.h"

#include "ft-hb.h"


#if defined( FT_CONFIG_OPTION_USE_HARFBUZZ )         && \
    defined( FT_CONFIG_OPTION_USE_HARFBUZZ_DYNAMIC )

#ifndef FT_LIBHARFBUZZ
#  ifdef _WIN32
#    define FT_LIBHARFBUZZ "libharfbuzz-0.dll"
#  else
#    ifdef __APPLE__
#      define FT_LIBHARFBUZZ "libharfbuzz.0.dylib"
#    else
#      define FT_LIBHARFBUZZ "libharfbuzz.so.0"
#    endif
#  endif
#endif

#ifdef _WIN32

#  include <windows.h>

#else /* !_WIN32 */

#  include <dlfcn.h>

  /* The GCC pragma suppresses the warning "ISO C forbids     */
  /* assignment between function pointer and 'void *'", which */
  /* inevitably gets emitted with `-Wpedantic`; see the man   */
  /* page of function `dlsym` for more information.           */
#  if defined( __GNUC__ )
#    pragma GCC diagnostic push
#    ifndef __cplusplus
#      pragma GCC diagnostic ignored "-Wpedantic"
#    endif
#  endif

#endif /* !_WIN32 */


  FT_LOCAL_DEF( void )
  ft_hb_funcs_init( struct AF_ModuleRec_  *af_module )
  {
    FT_Memory  memory = af_module->root.memory;
    FT_Error   error;

    ft_hb_funcs_t                *funcs           = NULL;
    ft_hb_version_atleast_func_t  version_atleast = NULL;

#ifdef _WIN32
    HANDLE  lib;
#  define DLSYM( lib, name ) \
            (ft_ ## name ## _func_t)GetProcAddress( lib, #name )
#else
    void  *lib;
#  define DLSYM( lib, name ) \
            (ft_ ## name ## _func_t)dlsym( lib, #name )
#endif


    af_module->hb_funcs = NULL;

    if ( FT_NEW( funcs ) )
      return;
    FT_ZERO( funcs );

#ifdef _WIN32

    lib = LoadLibraryA( FT_LIBHARFBUZZ );
    if ( !lib )
      goto Fail;
    version_atleast = DLSYM( lib, hb_version_atleast );

#else /* !_WIN32 */

#  ifdef RTLD_DEFAULT
#    define FT_RTLD_FLAGS RTLD_LAZY | RTLD_GLOBAL
    lib             = RTLD_DEFAULT;
    version_atleast = DLSYM( lib, hb_version_atleast );
#  else
#    define FT_RTLD_FLAGS RTLD_LAZY
#  endif

    if ( !version_atleast )
    {
      /* Load the HarfBuzz library.
       *
       * We never close the library, since we opened it with RTLD_GLOBAL.
       * This is important for the case where we are using HarfBuzz as a
       * shared library, and we want to use the symbols from the library in
       * other shared libraries or clients.  HarfBuzz holds onto global
       * variables, and closing the library will cause them to be
       * invalidated.
       */
      lib = dlopen( FT_LIBHARFBUZZ, FT_RTLD_FLAGS );
      if ( !lib )
        goto Fail;
      version_atleast = DLSYM( lib, hb_version_atleast );
    }

#endif /* !_WIN32 */

    if ( !version_atleast )
      goto Fail;

    /* Load all symbols we use. */
#define HB_EXTERN( ret, name, args )  \
  {                                   \
    funcs->name = DLSYM( lib, name ); \
    if ( !funcs->name )               \
      goto Fail;                      \
  }
#include "ft-hb-decls.h"
#undef HB_EXTERN

#undef DLSYM

    af_module->hb_funcs = funcs;
    return;

  Fail:
    if ( funcs )
      FT_FREE( funcs );
  }


  FT_LOCAL_DEF( void )
  ft_hb_funcs_done( struct AF_ModuleRec_  *af_module )
  {
    FT_Memory  memory = af_module->root.memory;


    if ( af_module->hb_funcs )
    {
      FT_FREE( af_module->hb_funcs );
      af_module->hb_funcs = NULL;
    }
  }


  FT_LOCAL_DEF( FT_Bool )
  ft_hb_enabled( struct AF_FaceGlobalsRec_  *globals )
  {
    return globals->module->hb_funcs != NULL;
  }

#ifndef _WIN32
#  if defined( __GNUC__ )
#    pragma GCC diagnostic pop
#  endif
#endif

#else /* !FT_CONFIG_OPTION_USE_HARFBUZZ_DYNAMIC */

  FT_LOCAL_DEF( FT_Bool )
  ft_hb_enabled( struct AF_FaceGlobalsRec_  *globals )
  {
    FT_UNUSED( globals );

#ifdef FT_CONFIG_OPTION_USE_HARFBUZZ
    return TRUE;
#else
    return FALSE;
#endif
  }

#endif /* !FT_CONFIG_OPTION_USE_HARFBUZZ_DYNAMIC */


/* END */
