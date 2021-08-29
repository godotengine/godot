#ifndef HEADER_CURL_MEMORY_H
#define HEADER_CURL_MEMORY_H
/***************************************************************************
 *                                  _   _ ____  _
 *  Project                     ___| | | |  _ \| |
 *                             / __| | | | |_) | |
 *                            | (__| |_| |  _ <| |___
 *                             \___|\___/|_| \_\_____|
 *
 * Copyright (C) 1998 - 2020, Daniel Stenberg, <daniel@haxx.se>, et al.
 *
 * This software is licensed as described in the file COPYING, which
 * you should have received as part of this distribution. The terms
 * are also available at https://curl.se/docs/copyright.html.
 *
 * You may opt to use, copy, modify, merge, publish, distribute and/or sell
 * copies of the Software, and permit persons to whom the Software is
 * furnished to do so, under the terms of the COPYING file.
 *
 * This software is distributed on an "AS IS" basis, WITHOUT WARRANTY OF ANY
 * KIND, either express or implied.
 *
 ***************************************************************************/

/*
 * Nasty internal details ahead...
 *
 * File curl_memory.h must be included by _all_ *.c source files
 * that use memory related functions strdup, malloc, calloc, realloc
 * or free, and given source file is used to build libcurl library.
 * It should be included immediately before memdebug.h as the last files
 * included to avoid undesired interaction with other memory function
 * headers in dependent libraries.
 *
 * There is nearly no exception to above rule. All libcurl source
 * files in 'lib' subdirectory as well as those living deep inside
 * 'packages' subdirectories and linked together in order to build
 * libcurl library shall follow it.
 *
 * File lib/strdup.c is an exception, given that it provides a strdup
 * clone implementation while using malloc. Extra care needed inside
 * this one.
 *
 * The need for curl_memory.h inclusion is due to libcurl's feature
 * of allowing library user to provide memory replacement functions,
 * memory callbacks, at runtime with curl_global_init_mem()
 *
 * Any *.c source file used to build libcurl library that does not
 * include curl_memory.h and uses any memory function of the five
 * mentioned above will compile without any indication, but it will
 * trigger weird memory related issues at runtime.
 *
 * OTOH some source files from 'lib' subdirectory may additionally be
 * used directly as source code when using some curlx_ functions by
 * third party programs that don't even use libcurl at all. When using
 * these source files in this way it is necessary these are compiled
 * with CURLX_NO_MEMORY_CALLBACKS defined, in order to ensure that no
 * attempt of calling libcurl's memory callbacks is done from code
 * which can not use this machinery.
 *
 * Notice that libcurl's 'memory tracking' system works chaining into
 * the memory callback machinery. This implies that when compiling
 * 'lib' source files with CURLX_NO_MEMORY_CALLBACKS defined this file
 * disengages usage of libcurl's 'memory tracking' system, defining
 * MEMDEBUG_NODEFINES and overriding CURLDEBUG purpose.
 *
 * CURLX_NO_MEMORY_CALLBACKS takes precedence over CURLDEBUG. This is
 * done in order to allow building a 'memory tracking' enabled libcurl
 * and at the same time allow building programs which do not use it.
 *
 * Programs and libraries in 'tests' subdirectories have specific
 * purposes and needs, and as such each one will use whatever fits
 * best, depending additionally whether it links with libcurl or not.
 *
 * Caveat emptor. Proper curlx_* separation is a work in progress
 * the same as CURLX_NO_MEMORY_CALLBACKS usage, some adjustments may
 * still be required. IOW don't use them yet, there are sharp edges.
 */

#ifdef HEADER_CURL_MEMDEBUG_H
#error "Header memdebug.h shall not be included before curl_memory.h"
#endif

#ifndef CURLX_NO_MEMORY_CALLBACKS

#ifndef CURL_DID_MEMORY_FUNC_TYPEDEFS /* only if not already done */
/*
 * The following memory function replacement typedef's are COPIED from
 * curl/curl.h and MUST match the originals. We copy them to avoid having to
 * include curl/curl.h here. We avoid that include since it includes stdio.h
 * and other headers that may get messed up with defines done here.
 */
typedef void *(*curl_malloc_callback)(size_t size);
typedef void (*curl_free_callback)(void *ptr);
typedef void *(*curl_realloc_callback)(void *ptr, size_t size);
typedef char *(*curl_strdup_callback)(const char *str);
typedef void *(*curl_calloc_callback)(size_t nmemb, size_t size);
#define CURL_DID_MEMORY_FUNC_TYPEDEFS
#endif

extern curl_malloc_callback Curl_cmalloc;
extern curl_free_callback Curl_cfree;
extern curl_realloc_callback Curl_crealloc;
extern curl_strdup_callback Curl_cstrdup;
extern curl_calloc_callback Curl_ccalloc;
#if defined(WIN32) && defined(UNICODE)
extern curl_wcsdup_callback Curl_cwcsdup;
#endif

#ifndef CURLDEBUG

/*
 * libcurl's 'memory tracking' system defines strdup, malloc, calloc,
 * realloc and free, along with others, in memdebug.h in a different
 * way although still using memory callbacks forward declared above.
 * When using the 'memory tracking' system (CURLDEBUG defined) we do
 * not define here the five memory functions given that definitions
 * from memdebug.h are the ones that shall be used.
 */

#undef strdup
#define strdup(ptr) Curl_cstrdup(ptr)
#undef malloc
#define malloc(size) Curl_cmalloc(size)
#undef calloc
#define calloc(nbelem,size) Curl_ccalloc(nbelem, size)
#undef realloc
#define realloc(ptr,size) Curl_crealloc(ptr, size)
#undef free
#define free(ptr) Curl_cfree(ptr)

#ifdef WIN32
#  ifdef UNICODE
#    undef wcsdup
#    define wcsdup(ptr) Curl_cwcsdup(ptr)
#    undef _wcsdup
#    define _wcsdup(ptr) Curl_cwcsdup(ptr)
#    undef _tcsdup
#    define _tcsdup(ptr) Curl_cwcsdup(ptr)
#  else
#    undef _tcsdup
#    define _tcsdup(ptr) Curl_cstrdup(ptr)
#  endif
#endif

#endif /* CURLDEBUG */

#else /* CURLX_NO_MEMORY_CALLBACKS */

#ifndef MEMDEBUG_NODEFINES
#define MEMDEBUG_NODEFINES
#endif

#endif /* CURLX_NO_MEMORY_CALLBACKS */

#endif /* HEADER_CURL_MEMORY_H */
