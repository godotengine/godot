/**************************************************************************
 *
 * Copyright 2009 VMware, Inc.
 * All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sub license, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL
 * THE COPYRIGHT HOLDERS, AUTHORS AND/OR ITS SUPPLIERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
 * USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial portions
 * of the Software.
 *
 **************************************************************************/


#ifndef U_DL_H_
#define U_DL_H_


#include "detect_os.h"

#ifdef __cplusplus
extern "C" {
#endif

#if DETECT_OS_WINDOWS
#  define UTIL_DL_EXT ".dll"
#  define UTIL_DL_PREFIX ""
#elif DETECT_OS_APPLE
#  define UTIL_DL_EXT ".dylib"
#  define UTIL_DL_PREFIX "lib"
#else
#  define UTIL_DL_EXT ".so"
#  define UTIL_DL_PREFIX "lib"
#endif


struct util_dl_library;


typedef void (*util_dl_proc)(void);


/**
 * Open a library dynamically.
 */
struct util_dl_library *
util_dl_open(const char *filename);


/**
 * Lookup a function in a library.
 */
util_dl_proc
util_dl_get_proc_address(struct util_dl_library *library,
                         const char *procname);


/**
 * Close a library.
 */
void
util_dl_close(struct util_dl_library *library);


/**
 * Return most recent error message.
 */
const char *
util_dl_error(void);

#ifdef __cplusplus
}
#endif

#endif /* U_DL_H_ */
