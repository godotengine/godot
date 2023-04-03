/**************************************************************************
 *
 * Copyright 2007-2013 VMware, Inc.
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
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial portions
 * of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
 * IN NO EVENT SHALL VMWARE AND/OR ITS SUPPLIERS BE LIABLE FOR
 * ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 **************************************************************************/

#include "no_extern_c.h"

#ifndef _C99_COMPAT_H_
#define _C99_COMPAT_H_


/*
 * MSVC hacks.
 */
#if defined(_MSC_VER)

#  if _MSC_VER < 1900
#    error "Microsoft Visual Studio 2015 or higher required"
#  endif

   /*
    * XXX: MSVC has a `__restrict` keyword, but it also has a
    * `__declspec(restrict)` modifier, so it is impossible to define a
    * `restrict` macro without interfering with the latter.  Furthermore the
    * MSVC standard library uses __declspec(restrict) under the _CRTRESTRICT
    * macro.  For now resolve this issue by redefining _CRTRESTRICT, but going
    * forward we should probably should stop using restrict, especially
    * considering that our code does not obbey strict aliasing rules any way.
    */
#  include <crtdefs.h>
#  undef _CRTRESTRICT
#  define _CRTRESTRICT
#endif


/*
 * C99 restrict keyword
 *
 * See also:
 * - http://cellperformance.beyond3d.com/articles/2006/05/demystifying-the-restrict-keyword.html
 */
#ifndef restrict
#  ifndef __cplusplus
     /* Use C99 restrict keyword */
#  elif defined(__GNUC__)
#    define restrict __restrict__
#  elif defined(_MSC_VER)
#    define restrict __restrict
#  else
#    define restrict /* */
#  endif
#endif


/* Simple test case for debugging */
#if 0
static inline const char *
test_c99_compat_h(const void * restrict a,
                  const void * restrict b)
{
   return __func__;
}
#endif


#endif /* _C99_COMPAT_H_ */
