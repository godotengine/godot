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

#ifndef NIR_INSTR_SET_H
#define NIR_INSTR_SET_H

#include "nir.h"

/**
 * This file defines functions for creating, destroying, and manipulating an
 * "instruction set," which is an abstraction for finding duplicate
 * instructions using a hash set. Note that the question of whether an
 * instruction is actually a duplicate (e.g. whether it has any side effects)
 * is handled transparently. The user can pass any instruction to
 * nir_instr_set_add_or_rewrite() and nir_instr_set_remove(), and if the
 * instruction isn't safe to rewrite or isn't supported, it's silently
 * removed.
 */

/*@{*/

/** Creates an instruction set, using a given ralloc mem_ctx */
struct set *nir_instr_set_create(void *mem_ctx);

/** Destroys an instruction set. */
void nir_instr_set_destroy(struct set *instr_set);

/**
 * Adds an instruction to an instruction set if it doesn't exist. If it
 * does already exist, rewrites all uses of it to point to the other
 * already-inserted instruction. Returns 'true' if the uses of the instruction
 * were rewritten. Otherwise, replaces the already-inserted instruction
 * with the new one.
 *
 * If cond_function() is given, only rewrites uses if
 * cond_function(old_instr, new_instr) returns true.
 */
bool nir_instr_set_add_or_rewrite(struct set *instr_set, nir_instr *instr,
                                  bool (*cond_function)(const nir_instr *a,
                                                        const nir_instr *b));

/**
 * Removes an instruction from an instruction set, so that other instructions
 * won't be merged with it.
 */
void nir_instr_set_remove(struct set *instr_set, nir_instr *instr);

/*@}*/

#endif /* NIR_INSTR_SET_H */
