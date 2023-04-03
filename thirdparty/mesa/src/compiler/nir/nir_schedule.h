/*
 * Copyright © 2014 Connor Abbott
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

#ifndef NIR_SCHEDULE_H
#define NIR_SCHEDULE_H

#include "nir.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Struct filled in by the intrinsic_cb callback of nir_schedule_options to
 * specify a backend-specific dependency on an intrinsic.
 */
typedef struct nir_schedule_dependency {
   /* Which class of dependency this is. The meanings of the classes are
    * specific to the backend. This must be less than
    * NIR_SCHEDULE_N_DEPENDENCY_CLASSES.
    */
   int klass;
   /* The type of dependency */
   enum {
      NIR_SCHEDULE_READ_DEPENDENCY,
      NIR_SCHEDULE_WRITE_DEPENDENCY,
   } type;
} nir_schedule_dependency;

typedef struct nir_schedule_options {
   /* On some hardware with some stages the inputs and outputs to the shader
    * share the same memory. In that case the scheduler needs to ensure that
    * all output writes are scheduled after all of the input writes to avoid
    * overwriting them. This is a bitmask of stages that need that.
    */
   unsigned stages_with_shared_io_memory;
   /* The approximate amount of register pressure at which point the scheduler
    * will try to reduce register usage.
    */
   int threshold;
   /* If set, instead of trying to optimise parallelism, the scheduler will try
    * to always minimise register pressure. This can be used as a fallback when
    * register allocation fails so that it can at least try to generate a
    * working shader even if it’s inefficient.
    */
   bool fallback;
   /* Callback used to add custom dependencies on intrinsics. If it returns
    * true then a dependency should be added and dep is filled in to describe
    * it.
    */
   bool (* intrinsic_cb)(nir_intrinsic_instr *intr,
                         nir_schedule_dependency *dep,
                         void *user_data);

   /* Data to pass to the intrinsic callback */
   void *intrinsic_cb_data;

   /* Callback used to specify instruction delays */
   unsigned (* instr_delay_cb)(nir_instr *instr, void *user_data);

   /* Data to pass to the instruction delay callback */
   void *instr_delay_cb_data;

} nir_schedule_options;

void nir_schedule(nir_shader *shader, const nir_schedule_options *options);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* NIR_SCHEDULE_H */
