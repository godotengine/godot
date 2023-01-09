/**************************************************************************
 *
 * Copyright 2020 Red Hat
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
 * IN NO EVENT SHALL THE AUTHORS AND/OR ITS SUPPLIERS BE LIABLE FOR
 * ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 **************************************************************************/

#include "util/detect.h"

static inline void *
util_memset32(void *s, uint32_t ui, size_t n)
{
#if DETECT_CC_GCC && DETECT_ARCH_X86_64
   long d0, d1;
   __asm__ volatile("rep\n\t"
                    "stosl"
                    : "=&c" (d0), "=&D" (d1)
                    : "a" (ui), "1" (s), "0" (n)
                    : "memory");
   return s;
#else
   uint32_t *xs = (uint32_t *)s;
   while (n--)
      *xs++ = ui;
   return s;
#endif
}

static inline void *
util_memset64(void *s, uint64_t ui, size_t n)
{
#if DETECT_CC_GCC && DETECT_ARCH_X86_64
   long d0, d1;
   __asm__ volatile("rep\n\t"
                    "stosq"
                    : "=&c" (d0), "=&D" (d1)
                    : "a" (ui), "1" (s), "0" (n)
                    : "memory");
   return s;
#else
   uint64_t *xs = (uint64_t *)s;
   while (n--)
      *xs++ = ui;
   return s;
#endif

}
