/**************************************************************************
 *
 * Copyright 2022 Yonggang Luo
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

/**
 * @file
 * Compiler configuration defines.
 *
 * See:
 * - http://gcc.gnu.org/onlinedocs/cpp/Common-Predefined-Macros.html
 * - echo | gcc -dM -E - | sort
 * - http://msdn.microsoft.com/en-us/library/b0084kay.aspx
 * - https://sourceforge.net/p/predef/wiki/Home/
 */

#ifndef UTIL_DETECT_CC_H_
#define UTIL_DETECT_CC_H_

/*
 * Compiler
 */

#if defined(__GNUC__)
#define DETECT_CC_GCC 1
#define DETECT_CC_GCC_VERSION (__GNUC__ * 100 + __GNUC_MINOR__)
#endif

/*
 * Meaning of _MSC_VER value:
 * - 1800: Visual Studio 2013
 * - 1700: Visual Studio 2012
 * - 1600: Visual Studio 2010
 * - 1500: Visual Studio 2008
 * - 1400: Visual C++ 2005
 * - 1310: Visual C++ .NET 2003
 * - 1300: Visual C++ .NET 2002
 *
 * __MSC__ seems to be an old macro -- it is not pre-defined on recent MSVC
 * versions.
 */
#if defined(_MSC_VER) || defined(__MSC__)
#define DETECT_CC_MSVC 1
#endif

#if defined(__ICL)
#define DETECT_CC_ICL 1
#endif

#ifndef DETECT_CC_GCC
#define DETECT_CC_GCC 0
#endif

#ifndef DETECT_CC_GCC_VERSION
#define DETECT_CC_GCC_VERSION 0
#endif

#ifndef DETECT_CC_MSVC
#define DETECT_CC_MSVC 0
#endif

#ifndef DETECT_CC_ICL
#define DETECT_CC_ICL 0
#endif

#endif /* UTIL_DETECT_CC_H_ */
