/*
 * Mesa 3-D graphics library
 *
 * Copyright (C) 2018-2019 Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef _DOUBLE_H_
#define _DOUBLE_H_


#ifdef __cplusplus
extern "C" {
#endif

/*
 * This API is no more than a wrapper to the counterpart softfloat.h
 * calls. Still, softfloat.h conversion API is meant to be kept private. In
 * other words, only use the API published here, instead of calling directly
 * the softfloat.h one.
 */

float _mesa_double_to_float(double val);
float _mesa_double_to_float_rtz(double val);

static inline float
_mesa_double_to_float_rtne(double val)
{
   return _mesa_double_to_float(val);
}

#ifdef __cplusplus
} /* extern C */
#endif

#endif /* _DOUBLE_H_ */
