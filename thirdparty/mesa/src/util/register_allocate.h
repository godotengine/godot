/*
 * Copyright © 2010 Intel Corporation
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
 *
 * Authors:
 *    Eric Anholt <eric@anholt.net>
 *
 */

#ifndef REGISTER_ALLOCATE_H
#define REGISTER_ALLOCATE_H

#include <stdbool.h>
#include "util/bitset.h"

#ifdef __cplusplus
extern "C" {
#endif


struct ra_class;
struct ra_regs;

struct blob;
struct blob_reader;

/* @{
 * Register set setup.
 *
 * This should be done once at backend initializaion, as
 * ra_set_finalize is O(r^2*c^2).  The registers may be virtual
 * registers, such as aligned register pairs that conflict with the
 * two real registers from which they are composed.
 */
struct ra_regs *ra_alloc_reg_set(void *mem_ctx, unsigned int count,
                                 bool need_conflict_lists);
void ra_set_allocate_round_robin(struct ra_regs *regs);
struct ra_class *ra_alloc_reg_class(struct ra_regs *regs);
struct ra_class *ra_alloc_contig_reg_class(struct ra_regs *regs, int contig_len);
unsigned int ra_class_index(struct ra_class *c);
void ra_add_reg_conflict(struct ra_regs *regs,
                         unsigned int r1, unsigned int r2);
void ra_add_transitive_reg_conflict(struct ra_regs *regs,
                                    unsigned int base_reg, unsigned int reg);

void
ra_add_transitive_reg_pair_conflict(struct ra_regs *regs,
                                    unsigned int base_reg, unsigned int reg0, unsigned int reg1);

void ra_make_reg_conflicts_transitive(struct ra_regs *regs, unsigned int reg);
void ra_class_add_reg(struct ra_class *c, unsigned int reg);
struct ra_class *ra_get_class_from_index(struct ra_regs *regs, unsigned int c);
void ra_set_num_conflicts(struct ra_regs *regs, unsigned int class_a,
                          unsigned int class_b, unsigned int num_conflicts);
void ra_set_finalize(struct ra_regs *regs, unsigned int **conflicts);

void ra_set_serialize(const struct ra_regs *regs, struct blob *blob);
struct ra_regs *ra_set_deserialize(void *mem_ctx, struct blob_reader *blob);
/** @} */

/** @{ Interference graph setup.
 *
 * Each interference graph node is a virtual variable in the IL.  It
 * is up to the user to ra_set_node_class() for the virtual variable,
 * and compute live ranges and ra_node_interfere() between conflicting
 * live ranges. Note that an interference *must not* be added between
 * two nodes if their classes haven't been assigned yet. The user
 * should set the class of each node before building the interference
 * graph.
 */
struct ra_graph *ra_alloc_interference_graph(struct ra_regs *regs,
                                             unsigned int count);
void ra_resize_interference_graph(struct ra_graph *g, unsigned int count);
void ra_set_node_class(struct ra_graph *g, unsigned int n, struct ra_class *c);
struct ra_class *ra_get_node_class(struct ra_graph *g, unsigned int n);
unsigned int ra_add_node(struct ra_graph *g, struct ra_class *c);

/** @{ Register selection callback.
 *
 * The register allocator can use either one of two built-in register
 * selection behaviors (ie. lowest-available or round-robin), or the
 * user can implement it's own selection policy by setting an register
 * selection callback.  The parameters to the callback are:
 *
 *  - n       the graph node, ie. the virtual variable to select a
 *            register for
 *  - regs    bitset of available registers to choose; this bitset
 *            contains *all* registers, but registers of different
 *            classes will not have their corresponding bit set.
 *  - data    callback data specified in ra_set_select_reg_callback()
 */
typedef unsigned int (*ra_select_reg_callback)(
      unsigned int n,        /* virtual variable to choose a physical reg for */
      BITSET_WORD *regs,     /* available physical regs to choose from */
      void *data);

void ra_set_select_reg_callback(struct ra_graph *g,
                                ra_select_reg_callback callback,
                                void *data);
void ra_add_node_interference(struct ra_graph *g,
                              unsigned int n1, unsigned int n2);
void ra_reset_node_interference(struct ra_graph *g, unsigned int n);
/** @} */

/** @{ Graph-coloring register allocation */
bool ra_allocate(struct ra_graph *g);

#define NO_REG ~0U
/**
 * Returns NO_REG for a node that has not (yet) been assigned.
 */
unsigned int ra_get_node_reg(struct ra_graph *g, unsigned int n);
void ra_set_node_reg(struct ra_graph * g, unsigned int n, unsigned int reg);
void ra_set_node_spill_cost(struct ra_graph *g, unsigned int n, float cost);
int ra_get_best_spill_node(struct ra_graph *g);
/** @} */


#ifdef __cplusplus
}  // extern "C"
#endif

#endif /* REGISTER_ALLOCATE_H */
