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

#ifndef DXIL_VERSIONS_H
#define DXIL_VERSIONS_H

#ifdef __cplusplus
extern "C" {
#endif

enum dxil_shader_model {
   SHADER_MODEL_6_0 = 0x60000,
   SHADER_MODEL_6_1,
   SHADER_MODEL_6_2,
   SHADER_MODEL_6_3,
   SHADER_MODEL_6_4,
   SHADER_MODEL_6_5,
   SHADER_MODEL_6_6,
   SHADER_MODEL_6_7,
};

enum dxil_validator_version {
   NO_DXIL_VALIDATION,
   DXIL_VALIDATOR_1_0 = 0x10000,
   DXIL_VALIDATOR_1_1,
   DXIL_VALIDATOR_1_2,
   DXIL_VALIDATOR_1_3,
   DXIL_VALIDATOR_1_4,
   DXIL_VALIDATOR_1_5,
   DXIL_VALIDATOR_1_6,
   DXIL_VALIDATOR_1_7,
};

#ifdef __cplusplus
}
#endif

#endif
