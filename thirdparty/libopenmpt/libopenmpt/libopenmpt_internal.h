/*
 * libopenmpt_internal.h
 * ---------------------
 * Purpose: libopenmpt internal interface configuration, overruling the public interface configuration (only used and needed when building libopenmpt)
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */

#ifndef LIBOPENMPT_INTERNAL_H
#define LIBOPENMPT_INTERNAL_H

#include "libopenmpt_config.h"

#if defined(NO_LIBOPENMPT_C)
#undef LIBOPENMPT_API
#define LIBOPENMPT_API     LIBOPENMPT_API_HELPER_LOCAL
#endif

#if defined(NO_LIBOPENMPT_CXX)
#undef LIBOPENMPT_CXX_API
#define LIBOPENMPT_CXX_API LIBOPENMPT_API_HELPER_LOCAL
#endif

#ifdef __cplusplus
#if defined(LIBOPENMPT_BUILD_DLL) || defined(LIBOPENMPT_USE_DLL)
#if defined(_MSC_VER) && !defined(_DLL)
/* #pragma message( "libopenmpt C++ interface is disabled if libopenmpt is built as a DLL and the runtime is statically linked. This is not supported by microsoft and cannot possibly work. Ever." ) */
#undef LIBOPENMPT_CXX_API
#define LIBOPENMPT_CXX_API LIBOPENMPT_API_HELPER_LOCAL
#endif
#endif
#endif


#endif /* LIBOPENMPT_INTERNAL_H */
