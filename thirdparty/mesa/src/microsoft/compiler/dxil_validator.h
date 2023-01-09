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

#ifndef DXIL_VALIDATOR_H
#define DXIL_VALIDATOR_H

#include "dxil_versions.h"

#include <stddef.h>

struct dxil_validator;

#ifdef __cplusplus
extern "C" {
#endif

struct dxil_validator *
dxil_create_validator(const void *ctx);

void
dxil_destroy_validator(struct dxil_validator *val);

bool
dxil_validate_module(struct dxil_validator *val, void *data,
                     size_t size, char **error);

char *
dxil_disasm_module(struct dxil_validator *val, void *data, size_t size);

enum dxil_validator_version
dxil_get_validator_version(struct dxil_validator *val);

#ifdef __cplusplus
}
#endif

#endif
