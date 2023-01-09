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
 * Mesa configuration defines.
 *
 * This header file sets several defines based on the compiler, processor
 * architecture, and operating system being used. These defines are comes
 * from corresponding headers.
 *
 * See:
 * - http://gcc.gnu.org/onlinedocs/cpp/Common-Predefined-Macros.html
 * - echo | gcc -dM -E - | sort
 * - http://msdn.microsoft.com/en-us/library/b0084kay.aspx
 * - https://sourceforge.net/p/predef/wiki/Home/
 */

#ifndef UTIL_DETECT_H_
#define UTIL_DETECT_H_

#include <limits.h>

/*
 * Compiler detection
 */
#include "util/detect_cc.h"

/*
 * Processor architecture detection
 */
#include "util/detect_arch.h"

/*
 * Endian detection detection
 */
#include "util/u_endian.h"

/*
 * Operating system family detection
 */
#include "util/detect_os.h"

#endif /* UTIL_DETECT_H_ */
