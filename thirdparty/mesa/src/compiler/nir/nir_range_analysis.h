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
#ifndef _NIR_RANGE_ANALYSIS_H_
#define _NIR_RANGE_ANALYSIS_H_

#include "nir.h"

enum PACKED ssa_ranges {
   unknown = 0,
   lt_zero,
   le_zero,
   gt_zero,
   ge_zero,
   ne_zero,
   eq_zero,
   last_range = eq_zero
};

struct ssa_result_range {
   enum ssa_ranges range;

   /** A floating-point value that can only have integer values. */
   bool is_integral;

   /** A floating-point value that cannot be NaN. */
   bool is_a_number;

   /** Is the value known to be a finite number? */
   bool is_finite;
};

#ifdef __cplusplus
extern "C" {
#endif

extern struct ssa_result_range
nir_analyze_range(struct hash_table *range_ht,
                  const nir_alu_instr *instr, unsigned src);

uint64_t nir_ssa_def_bits_used(const nir_ssa_def *def);

#ifdef __cplusplus
}
#endif
#endif /* _NIR_RANGE_ANALYSIS_H_ */
