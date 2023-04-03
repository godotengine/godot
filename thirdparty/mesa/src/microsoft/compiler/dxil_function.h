/*
 * Copyright © Microsoft Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef DXIL_FUNCTION_H
#define DXIL_FUNCTION_H

#define DXIL_FUNC_PARAM_INT64 'l'
#define DXIL_FUNC_PARAM_INT32 'i'
#define DXIL_FUNC_PARAM_INT16 'h'
#define DXIL_FUNC_PARAM_INT8  'c'
#define DXIL_FUNC_PARAM_BOOL  'b'

#define DXIL_FUNC_PARAM_FLOAT64 'g'
#define DXIL_FUNC_PARAM_FLOAT32 'f'
#define DXIL_FUNC_PARAM_FLOAT16 'e'
#define DXIL_FUNC_PARAM_HANDLE  '@'
#define DXIL_FUNC_PARAM_POINTER '*'
#define DXIL_FUNC_PARAM_VOID 'v'
#define DXIL_FUNC_PARAM_FROM_OVERLOAD 'O'
#define DXIL_FUNC_PARAM_RESRET 'R'
#define DXIL_FUNC_PARAM_CBUF_RET 'B'
#define DXIL_FUNC_PARAM_DIM 'D'
#define DXIL_FUNC_PARAM_SPLIT_DOUBLE 'G'
#define DXIL_FUNC_PARAM_SAMPLE_POS 'S'
#define DXIL_FUNC_PARAM_RES_BIND '#'
#define DXIL_FUNC_PARAM_RES_PROPS 'P'

#include "dxil_module.h"
#include "util/rb_tree.h"

const char *dxil_overload_suffix( enum overload_type overload);

const struct dxil_type *
dxil_get_overload_type(struct dxil_module *mod, enum overload_type overload);

/* These functions implement a generic method for declaring functions
 * The input parameters types are given as a string using the characters
 * given above as identifyer for the types. Only scalars and pointers to
 * scalars are implemented at this point.
 *
 * Examples:
 *
 * Call:             dxil_alloc_func(mod, "storeData.f32", "v", "icf");
 * Result function:  void storeData.f32(int32, int8, float32)
 *
 * Call:             dxil_alloc_func(mod, "storeData.f32", "e", "*icf");
 * Result function:  float16 storeData.f32(int32 *, int8, float32)
 *
 * Call:             dxil_alloc_func(mod, "storeData.f32", "*h", "b*f");
 * Result function:  float16 storeData.f32(bool *, float32 *)
 *
 */

const struct dxil_func *
dxil_alloc_func(struct dxil_module *mod, const char *name, enum overload_type overload,
                const char *retval_type_descr, const char *param_descr, enum dxil_attr_kind attr);

/* For specifically constructed return types one can also create the return type
 * seperately and pass it as paramaer
 */
const struct dxil_func *
dxil_alloc_func_with_rettype(struct dxil_module *mod, const char *name, enum overload_type overload,
                             const struct dxil_type *retval_type, const char *param_descr,
                             enum dxil_attr_kind attr);

/* This call should be the usual entry point to allocate a new function type.
 * 'name' must either be in the list of predefined functions, or a function
 * with 'name' and 'overload' must already be allocated by using one of the above
 * function calls. The allocated functions are searched for by using an rb_tree, so
 * the search complexity should be O(log(n)).
 */
const struct dxil_func *
dxil_get_function(struct dxil_module *mod, const char *name,
                  enum overload_type overload);

#endif // DXIL_FUNCTION_H
