/*
 * Copyright © 2016 Intel Corporation
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef _SPIRV_INFO_H_
#define _SPIRV_INFO_H_

#include "spirv.h"

const char *spirv_addressingmodel_to_string(SpvAddressingModel model);
const char *spirv_builtin_to_string(SpvBuiltIn builtin);
const char *spirv_capability_to_string(SpvCapability cap);
const char *spirv_decoration_to_string(SpvDecoration dec);
const char *spirv_dim_to_string(SpvDim dim);
const char *spirv_executionmode_to_string(SpvExecutionMode mode);
const char *spirv_executionmodel_to_string(SpvExecutionModel model);
const char *spirv_imageformat_to_string(SpvImageFormat format);
const char *spirv_imageoperands_to_string(SpvImageOperandsMask op);
const char *spirv_memorymodel_to_string(SpvMemoryModel cap);
const char *spirv_op_to_string(SpvOp op);
const char *spirv_storageclass_to_string(SpvStorageClass sc);
const char *spirv_fproundingmode_to_string(SpvFPRoundingMode sc);

#endif /* SPIRV_INFO_H */
