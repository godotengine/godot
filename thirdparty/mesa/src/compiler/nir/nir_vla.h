/**************************************************************************
 *
 * Copyright 2015 VMware, Inc.
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

#ifndef NIR_VLA_H
#define NIR_VLA_H

#include "c99_alloca.h"


/* Declare a variable length array, with no initialization */
#define NIR_VLA(_type, _name, _length) \
   _type *_name = alloca((_length) * sizeof *_name)


/* Declare a variable length array, and initialize it with the given byte.
 *
 * _length is evaluated twice, so expressions with side-effects must be
 * avoided.
 */
#define NIR_VLA_FILL(_type, _name, _length, _byte) \
   _type *_name = memset(alloca((_length) * sizeof *_name), _byte, (_length) * sizeof *_name)


/* Declare a variable length array, and zero it.
 *
 * Just like NIR_VLA_FILL, _length is evaluated twice, so expressions with
 * side-effects must be avoided.
 */
#define NIR_VLA_ZERO(_type, _name, _length) \
   NIR_VLA_FILL(_type, _name, _length, 0)

#endif /* NIR_VLA_H */
