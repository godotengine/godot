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

#ifndef REGISTER_ALLOCATE_INTERNAL_H
#define REGISTER_ALLOCATE_INTERNAL_H

#include <stdbool.h>
#include "util/bitset.h"
#include "util/u_dynarray.h"

#ifdef __cplusplus
extern "C" {

#define class klass
#endif

struct ra_reg {
   BITSET_WORD *conflicts;
   struct util_dynarray conflict_list;
};

struct ra_regs {
   struct ra_reg *regs;
   unsigned int count;

   struct ra_class **classes;
   unsigned int class_count;

   bool round_robin;
};

struct ra_class {
   struct ra_regs *regset;

   /**
    * Bitset indicating which registers belong to this class.
    *
    * (If bit N is set, then register N belongs to this class.)
    */
   BITSET_WORD *regs;

   /**
    * Number of regs after each bit in *regs that are also conflicted by an
    * allocation to that reg for this class.
    */
   int contig_len;

   /**
    * p(B) in Runeson/Nyström paper.
    *
    * This is "how many regs are in the set."
    */
   unsigned int p;

   /**
    * q(B,C) (indexed by C, B is this register class) in
    * Runeson/Nyström paper.  This is "how many registers of B could
    * the worst choice register from C conflict with".
    */
   unsigned int *q;

   int index;
};

struct ra_node {
   /** @{
    *
    * List of which nodes this node interferes with.  This should be
    * symmetric with the other node.
    */
   struct util_dynarray adjacency_list;
   /** @} */

   unsigned int class;

   /* Client-assigned register, if assigned, or NO_REG. */
   unsigned int forced_reg;

   /* Register, if assigned, or NO_REG. */
   unsigned int reg;

   /**
    * The q total, as defined in the Runeson/Nyström paper, for all the
    * interfering nodes not in the stack.
    */
   unsigned int q_total;

   /* For an implementation that needs register spilling, this is the
    * approximate cost of spilling this node.
    */
   float spill_cost;

   /* Temporary data for the algorithm to scratch around in */
   struct {
      /**
       * Temporary version of q_total which we decrement as things are placed
       * into the stack.
       */
      unsigned int q_total;
   } tmp;
};

struct ra_graph {
   struct ra_regs *regs;
   /**
    * the variables that need register allocation.
    */
   struct ra_node *nodes;
   BITSET_WORD *adjacency;
   unsigned int count; /**< count of nodes. */

   unsigned int alloc; /**< count of nodes allocated. */

   ra_select_reg_callback select_reg_callback;
   void *select_reg_callback_data;

   /* Temporary data for the algorithm to scratch around in */
   struct {
      unsigned int *stack;
      unsigned int stack_count;

      /** Bit-set indicating, for each register, if it's in the stack */
      BITSET_WORD *in_stack;

      /** Bit-set indicating, for each register, if it pre-assigned */
      BITSET_WORD *reg_assigned;

      /** Bit-set indicating, for each register, the value of the pq test */
      BITSET_WORD *pq_test;

      /** For each BITSET_WORD, the minimum q value or ~0 if unknown */
      unsigned int *min_q_total;

      /*
       * * For each BITSET_WORD, the node with the minimum q_total if
       * min_q_total[i] != ~0.
       */
      unsigned int *min_q_node;

      /**
       * Tracks the start of the set of optimistically-colored registers in the
       * stack.
       */
      unsigned int stack_optimistic_start;
   } tmp;
};

bool ra_class_allocations_conflict(struct ra_class *c1, unsigned int r1,
                                   struct ra_class *c2, unsigned int r2);

#ifdef __cplusplus
} /* extern "C" */

#undef class
#endif

#endif /* REGISTER_ALLOCATE_INTERNAL_H  */
