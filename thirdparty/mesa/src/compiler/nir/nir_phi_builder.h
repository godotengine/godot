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

#ifndef NIR_PHI_BUILDER_H
#define NIR_PHI_BUILDER_H

#include "nir.h"

/** A helper for placing phi nodes in a NIR shader
 *
 * Basic usage goes something like this:
 *
 *     each variable, var, has:
 *         a bitset var.defs of blocks where the variable is defined
 *         a struct nir_phi_builder_value *pb_val
 *
 *     // initialize bitsets
 *     foreach block:
 *         foreach def of variable var:
 *             var.defs[def.block] = true;
 *
 *     // initialize phi builder
 *     pb = nir_phi_builder_create()
 *     foreach var:
 *         var.pb_val = nir_phi_builder_add_value(pb, var.defs)
 *
 *     // Visit each block.  This needs to visit dominators first;
 *     // nir_foreach_block() will be ok.
 *
 *     foreach block:
 *         foreach instruction:
 *             foreach use of variable var:
 *                 replace use with nir_phi_builder_get_block_def(var.pb_val)
 *             foreach def of variable var:
 *                 create ssa def, register with
 *     nir_phi_builder_set_block_def(var.pb_val)
 *
 *     nir_phi_builder_finish(pb)
 */
struct nir_phi_builder;

struct nir_phi_builder_value;

/* Create a new phi builder.
 *
 * While this is fairly cheap, it does allocate some memory and walk the list
 * of blocks so it's recommended that you only call it once and use it to
 * build phis for several values.
 */
struct nir_phi_builder *nir_phi_builder_create(nir_function_impl *impl);

/* Register a value with the builder.
 *
 * The 'defs' parameter specifies a bitset of blocks in which the given value
 * is defined.  This is used to determine where to place the phi nodes.
 */
struct nir_phi_builder_value *
nir_phi_builder_add_value(struct nir_phi_builder *pb, unsigned num_components,
                          unsigned bit_size, const BITSET_WORD *defs);

/* Register a definition for the given value and block.
 *
 * It is safe to call this function as many times as you wish for any given
 * block/value pair.  However, it always replaces whatever was there
 * previously even if that definition is from a phi node.  The phi builder
 * always uses the latest information it has, so you must be careful about the
 * order in which you register definitions.  The final value at the end of the
 * block must be the last value registered.
 */
void
nir_phi_builder_value_set_block_def(struct nir_phi_builder_value *val,
                                    nir_block *block, nir_ssa_def *def);

/* Get the definition for the given value in the given block.
 *
 * This definition will always be the latest definition known for the given
 * block.  If no definition is immediately available, it will crawl up the
 * dominance tree and insert phi nodes as needed until it finds one.  In the
 * case that no suitable definition is found, it will return the result of a
 * nir_ssa_undef_instr with the correct number of components.
 *
 * Because this function only uses the latest available information for any
 * given block, you must have already finished registering definitions for any
 * blocks that dominate the current block in order to get the correct result.
 */
nir_ssa_def *
nir_phi_builder_value_get_block_def(struct nir_phi_builder_value *val,
                                    nir_block *block);

/* Finish building phi nodes and free the builder.
 *
 * This function does far more than just free memory.  Prior to calling
 * nir_phi_builder_finish, no phi nodes have actually been inserted in the
 * program.  This function is what finishes setting up phi node sources and
 * adds the phi nodes to the program.
 */
void nir_phi_builder_finish(struct nir_phi_builder *pb);

#endif /* NIR_PHI_BUILDER_H */
