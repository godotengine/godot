/*
 * Copyright © 2018 Intel Corporation
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

#ifndef NIR_DEREF_H
#define NIR_DEREF_H

#include "nir.h"
#include "nir_builder.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
   /** Short path so we can keep it on the stack most of the time. */
   nir_deref_instr *_short_path[7];

   /** A null-terminated array view of a deref chain
    *
    * The first element of this array will be the variable dereference
    * followed by every deref_instr on the path to the final one.  The last
    * element in the array is a NULL pointer which acts as a terminator.
    */
   nir_deref_instr **path;
} nir_deref_path;

typedef struct {
   nir_deref_instr *instr;
   nir_deref_path *_path;
} nir_deref_and_path;

void nir_deref_path_init(nir_deref_path *path,
                         nir_deref_instr *deref, void *mem_ctx);
void nir_deref_path_finish(nir_deref_path *path);

unsigned nir_deref_instr_get_const_offset(nir_deref_instr *deref,
                                          glsl_type_size_align_func size_align);

nir_ssa_def *nir_build_deref_offset(nir_builder *b, nir_deref_instr *deref,
                                    glsl_type_size_align_func size_align);

nir_deref_path *nir_get_deref_path(void *mem_ctx, nir_deref_and_path *deref);

typedef enum {
   nir_derefs_do_not_alias     = 0,
   nir_derefs_equal_bit        = (1 << 0),
   nir_derefs_may_alias_bit    = (1 << 1),
   nir_derefs_a_contains_b_bit = (1 << 2),
   nir_derefs_b_contains_a_bit = (1 << 3),
} nir_deref_compare_result;

nir_deref_compare_result nir_compare_deref_paths(nir_deref_path *a_path, nir_deref_path *b_path);
nir_deref_compare_result nir_compare_derefs(nir_deref_instr *a, nir_deref_instr *b);
nir_deref_compare_result nir_compare_derefs_and_paths(void *mem_ctx,
                                                      nir_deref_and_path *a,
                                                      nir_deref_and_path *b);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* NIR_DEREF_H */
