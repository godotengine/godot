/*
 * Copyright © 2019 Broadcom
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

#include "nir_schedule.h"
#include "util/dag.h"
#include "util/u_dynarray.h"

/** @file
 *
 * Implements basic-block-level prepass instruction scheduling in NIR to
 * manage register pressure.
 *
 * This is based on the Goodman/Hsu paper (1988, cached copy at
 * https://people.freedesktop.org/~anholt/scheduling-goodman-hsu.pdf).  We
 * make up the DDG for NIR (which can be mostly done using the NIR def/use
 * chains for SSA instructions, plus some edges for ordering register writes
 * vs reads, and some more for ordering intrinsics).  Then we pick heads off
 * of the DDG using their heuristic to emit the NIR instructions back into the
 * block in their new order.
 *
 * The hard case for prepass scheduling on GPUs seems to always be consuming
 * texture/ubo results.  The register pressure heuristic doesn't want to pick
 * an instr that starts consuming texture results because it usually won't be
 * the only usage, so that instruction will increase pressure.
 *
 * If you try to force consumption of tex results always, then in a case where
 * single sample is used for many outputs, you'll end up picking every other
 * user and expanding register pressure.  The partially_evaluated_path flag
 * helps tremendously, in that if you happen for whatever reason to pick a
 * texture sample's output, then you'll try to finish off that sample.  Future
 * work may include doing some local search before locking in a choice, to try
 * to more reliably find the case where just a few choices going against the
 * heuristic can manage to free the whole vector.
 */

static bool debug;

/**
 * Represents a node in the DDG for a NIR instruction.
 */
typedef struct {
   struct dag_node dag; /* must be first for our u_dynarray_foreach */
   nir_instr *instr;
   bool partially_evaluated_path;

   /* Approximate estimate of the delay between starting this instruction and
    * its results being available.
    *
    * Accuracy is not too important, given that we're prepass scheduling here
    * and just trying to reduce excess dependencies introduced by a register
    * allocator by stretching out the live intervals of expensive
    * instructions.
    */
   uint32_t delay;

   /* Cost of the maximum-delay path from this node to the leaves. */
   uint32_t max_delay;

   /* scoreboard->time value when this instruction can be scheduled without
    * any stalls expected.
    */
   uint32_t ready_time;
} nir_schedule_node;

typedef struct {
   struct dag *dag;

   nir_shader *shader;

   /* Mapping from nir_register * or nir_ssa_def * to a struct set of
    * instructions remaining to be scheduled using the register.
    */
   struct hash_table *remaining_uses;

   /* Map from nir_instr to nir_schedule_node * */
   struct hash_table *instr_map;

   /* Set of nir_register * or nir_ssa_def * that have had any instruction
    * scheduled on them.
    */
   struct set *live_values;

   /* An abstract approximation of the number of nir_scheduler_node->delay
    * units since the start of the shader.
    */
   uint32_t time;

   /* Number of channels currently used by the NIR instructions that have been
    * scheduled.
    */
   int pressure;

   /* Options specified by the backend */
   const nir_schedule_options *options;
} nir_schedule_scoreboard;

/* When walking the instructions in reverse, we use this flag to swap
 * before/after in add_dep().
 */
enum direction { F, R };

struct nir_schedule_class_dep {
   int klass;
   nir_schedule_node *node;
   struct nir_schedule_class_dep *next;
};

typedef struct {
   nir_schedule_scoreboard *scoreboard;

   /* Map from nir_register to nir_schedule_node * */
   struct hash_table *reg_map;

   /* Scheduler nodes for last instruction involved in some class of dependency.
    */
   nir_schedule_node *load_input;
   nir_schedule_node *store_shared;
   nir_schedule_node *unknown_intrinsic;
   nir_schedule_node *discard;
   nir_schedule_node *jump;

   struct nir_schedule_class_dep *class_deps;

   enum direction dir;
} nir_deps_state;

static void *
_mesa_hash_table_search_data(struct hash_table *ht, void *key)
{
   struct hash_entry *entry = _mesa_hash_table_search(ht, key);
   if (!entry)
      return NULL;
   return entry->data;
}

static nir_schedule_node *
nir_schedule_get_node(struct hash_table *instr_map, nir_instr *instr)
{
   return _mesa_hash_table_search_data(instr_map, instr);
}

static struct set *
nir_schedule_scoreboard_get_src(nir_schedule_scoreboard *scoreboard, nir_src *src)
{
   if (src->is_ssa) {
      return _mesa_hash_table_search_data(scoreboard->remaining_uses, src->ssa);
   } else {
      return _mesa_hash_table_search_data(scoreboard->remaining_uses,
                                          src->reg.reg);
   }
}

static int
nir_schedule_def_pressure(nir_ssa_def *def)
{
   return def->num_components;
}

static int
nir_schedule_src_pressure(nir_src *src)
{
   if (src->is_ssa)
      return nir_schedule_def_pressure(src->ssa);
   else
      return src->reg.reg->num_components;
}

static int
nir_schedule_dest_pressure(nir_dest *dest)
{
   if (dest->is_ssa)
      return nir_schedule_def_pressure(&dest->ssa);
   else
      return dest->reg.reg->num_components;
}

/**
 * Adds a dependency such that @after must appear in the final program after
 * @before.
 *
 * We add @before as a child of @after, so that DAG heads are the outputs of
 * the program and we make our scheduling decisions bottom to top.
 */
static void
add_dep(nir_deps_state *state,
        nir_schedule_node *before,
        nir_schedule_node *after)
{
   if (!before || !after)
      return;

   assert(before != after);

   if (state->dir == F)
      dag_add_edge(&before->dag, &after->dag, 0);
   else
      dag_add_edge(&after->dag, &before->dag, 0);
}


static void
add_read_dep(nir_deps_state *state,
             nir_schedule_node *before,
             nir_schedule_node *after)
{
   add_dep(state, before, after);
}

static void
add_write_dep(nir_deps_state *state,
              nir_schedule_node **before,
              nir_schedule_node *after)
{
   add_dep(state, *before, after);
   *before = after;
}

static bool
nir_schedule_reg_src_deps(nir_src *src, void *in_state)
{
   nir_deps_state *state = in_state;

   if (src->is_ssa)
      return true;

   struct hash_entry *entry = _mesa_hash_table_search(state->reg_map,
                                                      src->reg.reg);
   if (!entry)
      return true;
   nir_schedule_node *dst_n = entry->data;

   nir_schedule_node *src_n =
      nir_schedule_get_node(state->scoreboard->instr_map,
                            src->parent_instr);

   add_dep(state, dst_n, src_n);

   return true;
}

static bool
nir_schedule_reg_dest_deps(nir_dest *dest, void *in_state)
{
   nir_deps_state *state = in_state;

   if (dest->is_ssa)
      return true;

   nir_schedule_node *dest_n =
      nir_schedule_get_node(state->scoreboard->instr_map,
                            dest->reg.parent_instr);

   struct hash_entry *entry = _mesa_hash_table_search(state->reg_map,
                                                      dest->reg.reg);
   if (!entry) {
      _mesa_hash_table_insert(state->reg_map, dest->reg.reg, dest_n);
      return true;
   }
   nir_schedule_node **before = (nir_schedule_node **)&entry->data;

   add_write_dep(state, before, dest_n);

   return true;
}

static bool
nir_schedule_ssa_deps(nir_ssa_def *def, void *in_state)
{
   nir_deps_state *state = in_state;
   struct hash_table *instr_map = state->scoreboard->instr_map;
   nir_schedule_node *def_n = nir_schedule_get_node(instr_map, def->parent_instr);

   nir_foreach_use(src, def) {
      nir_schedule_node *use_n = nir_schedule_get_node(instr_map,
                                                       src->parent_instr);

      add_read_dep(state, def_n, use_n);
   }

   return true;
}

static struct nir_schedule_class_dep *
nir_schedule_get_class_dep(nir_deps_state *state,
                           int klass)
{
   for (struct nir_schedule_class_dep *class_dep = state->class_deps;
        class_dep != NULL;
        class_dep = class_dep->next) {
      if (class_dep->klass == klass)
         return class_dep;
   }

   struct nir_schedule_class_dep *class_dep =
      ralloc(state->reg_map, struct nir_schedule_class_dep);

   class_dep->klass = klass;
   class_dep->node = NULL;
   class_dep->next = state->class_deps;

   state->class_deps = class_dep;

   return class_dep;
}

static void
nir_schedule_intrinsic_deps(nir_deps_state *state,
                            nir_intrinsic_instr *instr)
{
   nir_schedule_node *n = nir_schedule_get_node(state->scoreboard->instr_map,
                                                &instr->instr);
   const nir_schedule_options *options = state->scoreboard->options;
   nir_schedule_dependency dep;

   if (options->intrinsic_cb &&
       options->intrinsic_cb(instr, &dep, options->intrinsic_cb_data)) {
      struct nir_schedule_class_dep *class_dep =
         nir_schedule_get_class_dep(state, dep.klass);

      switch (dep.type) {
      case NIR_SCHEDULE_READ_DEPENDENCY:
         add_read_dep(state, class_dep->node, n);
         break;
      case NIR_SCHEDULE_WRITE_DEPENDENCY:
         add_write_dep(state, &class_dep->node, n);
         break;
      }
   }

   switch (instr->intrinsic) {
   case nir_intrinsic_load_uniform:
   case nir_intrinsic_load_ubo:
   case nir_intrinsic_load_front_face:
      break;

   case nir_intrinsic_discard:
   case nir_intrinsic_discard_if:
   case nir_intrinsic_demote:
   case nir_intrinsic_demote_if:
   case nir_intrinsic_terminate:
   case nir_intrinsic_terminate_if:
      /* We are adding two dependencies:
       *
       * * A individual one that we could use to add a read_dep while handling
       *   nir_instr_type_tex
       *
       * * Include it on the unknown intrinsic set, as we want discard to be
       *   serialized in in the same order relative to intervening stores or
       *   atomic accesses to SSBOs and images
       */
      add_write_dep(state, &state->discard, n);
      add_write_dep(state, &state->unknown_intrinsic, n);
      break;

   case nir_intrinsic_store_output:
      /* For some hardware and stages, output stores affect the same shared
       * memory as input loads.
       */
      if ((state->scoreboard->options->stages_with_shared_io_memory &
           (1 << state->scoreboard->shader->info.stage)))
         add_write_dep(state, &state->load_input, n);

      /* Make sure that preceding discards stay before the store_output */
      add_read_dep(state, state->discard, n);

      break;

   case nir_intrinsic_load_input:
   case nir_intrinsic_load_per_vertex_input:
      add_read_dep(state, state->load_input, n);
      break;

   case nir_intrinsic_load_shared:
   case nir_intrinsic_load_shared2_amd:
      /* Don't move load_shared beyond a following store_shared, as it could
       * change their value
       */
      add_read_dep(state, state->store_shared, n);
      break;

   case nir_intrinsic_store_shared:
   case nir_intrinsic_store_shared2_amd:
      add_write_dep(state, &state->store_shared, n);
      break;

   case nir_intrinsic_control_barrier:
   case nir_intrinsic_memory_barrier_shared:
   case nir_intrinsic_group_memory_barrier:
   /* A generic memory barrier can be emitted when multiple synchronization
    * semantics are involved, including shared memory.
    */
   case nir_intrinsic_memory_barrier:
      add_write_dep(state, &state->store_shared, n);

      /* Serialize against ssbos/atomics/etc. */
      add_write_dep(state, &state->unknown_intrinsic, n);
      break;

   case nir_intrinsic_scoped_barrier: {
      const nir_variable_mode modes = nir_intrinsic_memory_modes(instr);

      if (modes & nir_var_mem_shared)
         add_write_dep(state, &state->store_shared, n);

      /* Serialize against other categories. */
      add_write_dep(state, &state->unknown_intrinsic, n);

      break;
   }

   default:
      /* Attempt to handle other intrinsics that we haven't individually
       * categorized by serializing them in the same order relative to each
       * other.
       */
      add_write_dep(state, &state->unknown_intrinsic, n);
      break;
   }
}

/**
 * Common code for dependencies that need to be tracked both forward and
 * backward.
 *
 * This is for things like "all reads of r4 have to happen between the r4
 * writes that surround them".
 */
static void
nir_schedule_calculate_deps(nir_deps_state *state, nir_schedule_node *n)
{
   nir_instr *instr = n->instr;

   /* For NIR SSA defs, we only need to do a single pass of making the uses
    * depend on the def.
    */
   if (state->dir == F)
      nir_foreach_ssa_def(instr, nir_schedule_ssa_deps, state);

   /* For NIR regs, track the last writer in the scheduler state so that we
    * can keep the writes in order and let reads get reordered only between
    * each write.
    */
   nir_foreach_src(instr, nir_schedule_reg_src_deps, state);

   nir_foreach_dest(instr, nir_schedule_reg_dest_deps, state);

   /* Make sure any other instructions keep their positions relative to
    * jumps.
    */
   if (instr->type != nir_instr_type_jump)
      add_read_dep(state, state->jump, n);

   switch (instr->type) {
   case nir_instr_type_ssa_undef:
   case nir_instr_type_load_const:
   case nir_instr_type_alu:
   case nir_instr_type_deref:
      break;

   case nir_instr_type_tex:
      /* Don't move texture ops before a discard, as that could increase
       * memory bandwidth for reading the discarded samples.
       */
      add_read_dep(state, state->discard, n);
      break;

   case nir_instr_type_jump:
      add_write_dep(state, &state->jump, n);
      break;

   case nir_instr_type_call:
      unreachable("Calls should have been lowered");
      break;

   case nir_instr_type_parallel_copy:
      unreachable("Parallel copies should have been lowered");
      break;

   case nir_instr_type_phi:
      unreachable("nir_schedule() should be called after lowering from SSA");
      break;

   case nir_instr_type_intrinsic:
      nir_schedule_intrinsic_deps(state, nir_instr_as_intrinsic(instr));
      break;
   }
}

static void
calculate_forward_deps(nir_schedule_scoreboard *scoreboard, nir_block *block)
{
   nir_deps_state state = {
      .scoreboard = scoreboard,
      .dir = F,
      .reg_map = _mesa_pointer_hash_table_create(NULL),
   };

   nir_foreach_instr(instr, block) {
      nir_schedule_node *node = nir_schedule_get_node(scoreboard->instr_map,
                                                      instr);
      nir_schedule_calculate_deps(&state, node);
   }

   ralloc_free(state.reg_map);
}

static void
calculate_reverse_deps(nir_schedule_scoreboard *scoreboard, nir_block *block)
{
   nir_deps_state state = {
      .scoreboard = scoreboard,
      .dir = R,
      .reg_map = _mesa_pointer_hash_table_create(NULL),
   };

   nir_foreach_instr_reverse(instr, block) {
      nir_schedule_node *node = nir_schedule_get_node(scoreboard->instr_map,
                                                      instr);
      nir_schedule_calculate_deps(&state, node);
   }

   ralloc_free(state.reg_map);
}

typedef struct {
   nir_schedule_scoreboard *scoreboard;
   int regs_freed;
} nir_schedule_regs_freed_state;

static bool
nir_schedule_regs_freed_src_cb(nir_src *src, void *in_state)
{
   nir_schedule_regs_freed_state *state = in_state;
   nir_schedule_scoreboard *scoreboard = state->scoreboard;
   struct set *remaining_uses = nir_schedule_scoreboard_get_src(scoreboard, src);

   if (remaining_uses->entries == 1 &&
       _mesa_set_search(remaining_uses, src->parent_instr)) {
      state->regs_freed += nir_schedule_src_pressure(src);
   }

   return true;
}

static bool
nir_schedule_regs_freed_def_cb(nir_ssa_def *def, void *in_state)
{
   nir_schedule_regs_freed_state *state = in_state;

   state->regs_freed -= nir_schedule_def_pressure(def);

   return true;
}

static bool
nir_schedule_regs_freed_dest_cb(nir_dest *dest, void *in_state)
{
   nir_schedule_regs_freed_state *state = in_state;
   nir_schedule_scoreboard *scoreboard = state->scoreboard;

   if (dest->is_ssa)
      return true;

   nir_register *reg = dest->reg.reg;

   /* Only the first def of a reg counts against register pressure. */
   if (!_mesa_set_search(scoreboard->live_values, reg))
      state->regs_freed -= nir_schedule_dest_pressure(dest);

   return true;
}

static int
nir_schedule_regs_freed(nir_schedule_scoreboard *scoreboard, nir_schedule_node *n)
{
   nir_schedule_regs_freed_state state = {
      .scoreboard = scoreboard,
   };

   nir_foreach_src(n->instr, nir_schedule_regs_freed_src_cb, &state);

   nir_foreach_ssa_def(n->instr, nir_schedule_regs_freed_def_cb, &state);

   nir_foreach_dest(n->instr, nir_schedule_regs_freed_dest_cb, &state);

   return state.regs_freed;
}

/**
 * Chooses an instruction that will minimise the register pressure as much as
 * possible. This should only be used as a fallback when the regular scheduling
 * generates a shader whose register allocation fails.
 */
static nir_schedule_node *
nir_schedule_choose_instruction_fallback(nir_schedule_scoreboard *scoreboard)
{
   nir_schedule_node *chosen = NULL;

   /* Find the leader in the ready (shouldn't-stall) set with the mininum
    * cost.
    */
   list_for_each_entry(nir_schedule_node, n, &scoreboard->dag->heads, dag.link) {
      if (scoreboard->time < n->ready_time)
         continue;

      if (!chosen || chosen->max_delay > n->max_delay)
         chosen = n;
   }
   if (chosen) {
      if (debug) {
         fprintf(stderr, "chose (ready fallback):          ");
         nir_print_instr(chosen->instr, stderr);
         fprintf(stderr, "\n");
      }

      return chosen;
   }

   /* Otherwise, choose the leader with the minimum cost. */
   list_for_each_entry(nir_schedule_node, n, &scoreboard->dag->heads, dag.link) {
      if (!chosen || chosen->max_delay > n->max_delay)
         chosen = n;
   }
   if (debug) {
      fprintf(stderr, "chose (leader fallback):         ");
      nir_print_instr(chosen->instr, stderr);
      fprintf(stderr, "\n");
   }

   return chosen;
}

/**
 * Chooses an instruction to schedule using the Goodman/Hsu (1988) CSP (Code
 * Scheduling for Parallelism) heuristic.
 *
 * Picks an instruction on the critical that's ready to execute without
 * stalls, if possible, otherwise picks the instruction on the critical path.
 */
static nir_schedule_node *
nir_schedule_choose_instruction_csp(nir_schedule_scoreboard *scoreboard)
{
   nir_schedule_node *chosen = NULL;

   /* Find the leader in the ready (shouldn't-stall) set with the maximum
    * cost.
    */
   list_for_each_entry(nir_schedule_node, n, &scoreboard->dag->heads, dag.link) {
      if (scoreboard->time < n->ready_time)
         continue;

      if (!chosen || chosen->max_delay < n->max_delay)
         chosen = n;
   }
   if (chosen) {
      if (debug) {
         fprintf(stderr, "chose (ready):          ");
         nir_print_instr(chosen->instr, stderr);
         fprintf(stderr, "\n");
      }

      return chosen;
   }

   /* Otherwise, choose the leader with the maximum cost. */
   list_for_each_entry(nir_schedule_node, n, &scoreboard->dag->heads, dag.link) {
      if (!chosen || chosen->max_delay < n->max_delay)
         chosen = n;
   }
   if (debug) {
      fprintf(stderr, "chose (leader):         ");
      nir_print_instr(chosen->instr, stderr);
      fprintf(stderr, "\n");
   }

   return chosen;
}

/**
 * Chooses an instruction to schedule using the Goodman/Hsu (1988) CSR (Code
 * Scheduling for Register pressure) heuristic.
 */
static nir_schedule_node *
nir_schedule_choose_instruction_csr(nir_schedule_scoreboard *scoreboard)
{
   nir_schedule_node *chosen = NULL;

   /* Find a ready inst with regs freed and pick the one with max cost. */
   list_for_each_entry(nir_schedule_node, n, &scoreboard->dag->heads, dag.link) {
      if (n->ready_time > scoreboard->time)
         continue;

      int regs_freed = nir_schedule_regs_freed(scoreboard, n);

      if (regs_freed > 0 && (!chosen || chosen->max_delay < n->max_delay)) {
         chosen = n;
      }
   }
   if (chosen) {
      if (debug) {
         fprintf(stderr, "chose (freed+ready):    ");
         nir_print_instr(chosen->instr, stderr);
         fprintf(stderr, "\n");
      }

      return chosen;
   }

   /* Find a leader with regs freed and pick the one with max cost. */
   list_for_each_entry(nir_schedule_node, n, &scoreboard->dag->heads, dag.link) {
      int regs_freed = nir_schedule_regs_freed(scoreboard, n);

      if (regs_freed > 0 && (!chosen || chosen->max_delay < n->max_delay)) {
         chosen = n;
      }
   }
   if (chosen) {
      if (debug) {
         fprintf(stderr, "chose (regs freed):     ");
         nir_print_instr(chosen->instr, stderr);
         fprintf(stderr, "\n");
      }

      return chosen;
   }

   /* Find a partially evaluated path and try to finish it off */
   list_for_each_entry(nir_schedule_node, n, &scoreboard->dag->heads, dag.link) {
      if (n->partially_evaluated_path &&
          (!chosen || chosen->max_delay < n->max_delay)) {
         chosen = n;
      }
   }
   if (chosen) {
      if (debug) {
         fprintf(stderr, "chose (partial path):   ");
         nir_print_instr(chosen->instr, stderr);
         fprintf(stderr, "\n");
      }

      return chosen;
   }

   /* Contra the paper, pick a leader with no effect on used regs.  This may
    * open up new opportunities, as otherwise a single-operand instr consuming
    * a value will tend to block finding freeing that value.  This had a
    * massive effect on reducing spilling on V3D.
    *
    * XXX: Should this prioritize ready?
    */
   list_for_each_entry(nir_schedule_node, n, &scoreboard->dag->heads, dag.link) {
      if (nir_schedule_regs_freed(scoreboard, n) != 0)
         continue;

      if (!chosen || chosen->max_delay < n->max_delay)
         chosen = n;
   }
   if (chosen) {
      if (debug) {
         fprintf(stderr, "chose (regs no-op):     ");
         nir_print_instr(chosen->instr, stderr);
         fprintf(stderr, "\n");
      }

      return chosen;
   }

   /* Pick the max delay of the remaining ready set. */
   list_for_each_entry(nir_schedule_node, n, &scoreboard->dag->heads, dag.link) {
      if (n->ready_time > scoreboard->time)
         continue;
      if (!chosen || chosen->max_delay < n->max_delay)
         chosen = n;
   }
   if (chosen) {
      if (debug) {
         fprintf(stderr, "chose (ready max delay):   ");
         nir_print_instr(chosen->instr, stderr);
         fprintf(stderr, "\n");
      }
      return chosen;
   }

   /* Pick the max delay of the remaining leaders. */
   list_for_each_entry(nir_schedule_node, n, &scoreboard->dag->heads, dag.link) {
      if (!chosen || chosen->max_delay < n->max_delay)
         chosen = n;
   }

   if (debug) {
      fprintf(stderr, "chose (max delay):         ");
      nir_print_instr(chosen->instr, stderr);
      fprintf(stderr, "\n");
   }

   return chosen;
}

static void
dump_state(nir_schedule_scoreboard *scoreboard)
{
   list_for_each_entry(nir_schedule_node, n, &scoreboard->dag->heads, dag.link) {
      fprintf(stderr, "maxdel %5d ", n->max_delay);
      nir_print_instr(n->instr, stderr);
      fprintf(stderr, "\n");

      util_dynarray_foreach(&n->dag.edges, struct dag_edge, edge) {
         nir_schedule_node *child = (nir_schedule_node *)edge->child;

         fprintf(stderr, " -> (%d parents) ", child->dag.parent_count);
         nir_print_instr(child->instr, stderr);
         fprintf(stderr, "\n");
      }
   }
}

static void
nir_schedule_mark_use(nir_schedule_scoreboard *scoreboard,
                      void *reg_or_def,
                      nir_instr *reg_or_def_parent,
                      int pressure)
{
   /* Make the value live if it's the first time it's been used. */
   if (!_mesa_set_search(scoreboard->live_values, reg_or_def)) {
      _mesa_set_add(scoreboard->live_values, reg_or_def);
      scoreboard->pressure += pressure;
   }

   /* Make the value dead if it's the last remaining use.  Be careful when one
    * instruction uses a value twice to not decrement pressure twice.
    */
   struct set *remaining_uses =
      _mesa_hash_table_search_data(scoreboard->remaining_uses, reg_or_def);
   struct set_entry *entry = _mesa_set_search(remaining_uses, reg_or_def_parent);
   if (entry) {
      _mesa_set_remove(remaining_uses, entry);

      if (remaining_uses->entries == 0)
         scoreboard->pressure -= pressure;
   }
}

static bool
nir_schedule_mark_src_scheduled(nir_src *src, void *state)
{
   nir_schedule_scoreboard *scoreboard = state;
   struct set *remaining_uses = nir_schedule_scoreboard_get_src(scoreboard, src);

   struct set_entry *entry = _mesa_set_search(remaining_uses,
                                              src->parent_instr);
   if (entry) {
      /* Once we've used an SSA value in one instruction, bump the priority of
       * the other uses so the SSA value can get fully consumed.
       *
       * We don't do this for registers, and it's would be a hassle and it's
       * unclear if that would help or not.  Also, skip it for constants, as
       * they're often folded as immediates into backend instructions and have
       * many unrelated instructions all referencing the same value (0).
       */
      if (src->is_ssa &&
          src->ssa->parent_instr->type != nir_instr_type_load_const) {
         nir_foreach_use(other_src, src->ssa) {
            if (other_src->parent_instr == src->parent_instr)
               continue;

            nir_schedule_node *n =
               nir_schedule_get_node(scoreboard->instr_map,
                                     other_src->parent_instr);

            if (n && !n->partially_evaluated_path) {
               if (debug) {
                  fprintf(stderr, "  New partially evaluated path: ");
                  nir_print_instr(n->instr, stderr);
                  fprintf(stderr, "\n");
               }

               n->partially_evaluated_path = true;
            }
         }
      }
   }

   nir_schedule_mark_use(scoreboard,
                         src->is_ssa ? (void *)src->ssa : (void *)src->reg.reg,
                         src->parent_instr,
                         nir_schedule_src_pressure(src));

   return true;
}

static bool
nir_schedule_mark_def_scheduled(nir_ssa_def *def, void *state)
{
   nir_schedule_scoreboard *scoreboard = state;

   nir_schedule_mark_use(scoreboard, def, def->parent_instr,
                         nir_schedule_def_pressure(def));

   return true;
}

static bool
nir_schedule_mark_dest_scheduled(nir_dest *dest, void *state)
{
   nir_schedule_scoreboard *scoreboard = state;

   /* SSA defs were handled in nir_schedule_mark_def_scheduled()
    */
   if (dest->is_ssa)
      return true;

   /* XXX: This is not actually accurate for regs -- the last use of a reg may
    * have a live interval that extends across control flow.  We should
    * calculate the live ranges of regs, and have scheduler nodes for the CF
    * nodes that also "use" the reg.
    */
   nir_schedule_mark_use(scoreboard, dest->reg.reg,
                         dest->reg.parent_instr,
                         nir_schedule_dest_pressure(dest));

   return true;
}

static void
nir_schedule_mark_node_scheduled(nir_schedule_scoreboard *scoreboard,
                                 nir_schedule_node *n)
{
   nir_foreach_src(n->instr, nir_schedule_mark_src_scheduled, scoreboard);
   nir_foreach_ssa_def(n->instr, nir_schedule_mark_def_scheduled, scoreboard);
   nir_foreach_dest(n->instr, nir_schedule_mark_dest_scheduled, scoreboard);

   util_dynarray_foreach(&n->dag.edges, struct dag_edge, edge) {
      nir_schedule_node *child = (nir_schedule_node *)edge->child;

      child->ready_time = MAX2(child->ready_time,
                               scoreboard->time + n->delay);

      if (child->dag.parent_count == 1) {
         if (debug) {
            fprintf(stderr, "  New DAG head: ");
            nir_print_instr(child->instr, stderr);
            fprintf(stderr, "\n");
         }
      }
   }

   dag_prune_head(scoreboard->dag, &n->dag);

   scoreboard->time = MAX2(n->ready_time, scoreboard->time);
   scoreboard->time++;
}

static void
nir_schedule_instructions(nir_schedule_scoreboard *scoreboard, nir_block *block)
{
   while (!list_is_empty(&scoreboard->dag->heads)) {
      if (debug) {
         fprintf(stderr, "current list:\n");
         dump_state(scoreboard);
      }

      nir_schedule_node *chosen;
      if (scoreboard->options->fallback)
         chosen = nir_schedule_choose_instruction_fallback(scoreboard);
      else if (scoreboard->pressure < scoreboard->options->threshold)
         chosen = nir_schedule_choose_instruction_csp(scoreboard);
      else
         chosen = nir_schedule_choose_instruction_csr(scoreboard);

      /* Now that we've scheduled a new instruction, some of its children may
       * be promoted to the list of instructions ready to be scheduled.
       */
      nir_schedule_mark_node_scheduled(scoreboard, chosen);

      /* Move the instruction to the end (so our first chosen instructions are
       * the start of the program).
       */
      exec_node_remove(&chosen->instr->node);
      exec_list_push_tail(&block->instr_list, &chosen->instr->node);

      if (debug)
         fprintf(stderr, "\n");
   }
}

static uint32_t
nir_schedule_get_delay(nir_schedule_scoreboard *scoreboard, nir_instr *instr)
{
   if (scoreboard->options->instr_delay_cb) {
      void *cb_data = scoreboard->options->instr_delay_cb_data;
      return scoreboard->options->instr_delay_cb(instr, cb_data);
   }

   switch (instr->type) {
   case nir_instr_type_ssa_undef:
   case nir_instr_type_load_const:
   case nir_instr_type_alu:
   case nir_instr_type_deref:
   case nir_instr_type_jump:
   case nir_instr_type_parallel_copy:
   case nir_instr_type_call:
   case nir_instr_type_phi:
      return 1;

   case nir_instr_type_intrinsic:
      switch (nir_instr_as_intrinsic(instr)->intrinsic) {
      case nir_intrinsic_load_ubo:
      case nir_intrinsic_load_ssbo:
      case nir_intrinsic_load_scratch:
      case nir_intrinsic_load_shared:
      case nir_intrinsic_image_load:
         return 50;
      default:
         return 1;
      }
      break;

   case nir_instr_type_tex:
      /* Pick some large number to try to fetch textures early and sample them
       * late.
       */
      return 100;
   }

   return 0;
}

static void
nir_schedule_dag_max_delay_cb(struct dag_node *node, void *state)
{
   nir_schedule_node *n = (nir_schedule_node *)node;
   uint32_t max_delay = 0;

   util_dynarray_foreach(&n->dag.edges, struct dag_edge, edge) {
      nir_schedule_node *child = (nir_schedule_node *)edge->child;
      max_delay = MAX2(child->max_delay, max_delay);
   }

   n->max_delay = MAX2(n->max_delay, max_delay + n->delay);
 }

static void
nir_schedule_block(nir_schedule_scoreboard *scoreboard, nir_block *block)
{
   void *mem_ctx = ralloc_context(NULL);
   scoreboard->instr_map = _mesa_pointer_hash_table_create(mem_ctx);

   scoreboard->dag = dag_create(mem_ctx);

   nir_foreach_instr(instr, block) {
      nir_schedule_node *n =
         rzalloc(mem_ctx, nir_schedule_node);

      n->instr = instr;
      n->delay = nir_schedule_get_delay(scoreboard, instr);
      dag_init_node(scoreboard->dag, &n->dag);

      _mesa_hash_table_insert(scoreboard->instr_map, instr, n);
   }

   calculate_forward_deps(scoreboard, block);
   calculate_reverse_deps(scoreboard, block);

   dag_traverse_bottom_up(scoreboard->dag, nir_schedule_dag_max_delay_cb, NULL);

   nir_schedule_instructions(scoreboard, block);

   ralloc_free(mem_ctx);
   scoreboard->instr_map = NULL;
}

static bool
nir_schedule_ssa_def_init_scoreboard(nir_ssa_def *def, void *state)
{
   nir_schedule_scoreboard *scoreboard = state;
   struct set *def_uses = _mesa_pointer_set_create(scoreboard);

   _mesa_hash_table_insert(scoreboard->remaining_uses, def, def_uses);

   _mesa_set_add(def_uses, def->parent_instr);

   nir_foreach_use(src, def) {
      _mesa_set_add(def_uses, src->parent_instr);
   }

   /* XXX: Handle if uses */

   return true;
}

static nir_schedule_scoreboard *
nir_schedule_get_scoreboard(nir_shader *shader,
                            const nir_schedule_options *options)
{
   nir_schedule_scoreboard *scoreboard = rzalloc(NULL, nir_schedule_scoreboard);

   scoreboard->shader = shader;
   scoreboard->live_values = _mesa_pointer_set_create(scoreboard);
   scoreboard->remaining_uses = _mesa_pointer_hash_table_create(scoreboard);
   scoreboard->options = options;
   scoreboard->pressure = 0;

   nir_foreach_function(function, shader) {
      nir_foreach_register(reg, &function->impl->registers) {
         struct set *register_uses =
            _mesa_pointer_set_create(scoreboard);

         _mesa_hash_table_insert(scoreboard->remaining_uses, reg, register_uses);

         nir_foreach_use(src, reg) {
            _mesa_set_add(register_uses, src->parent_instr);
         }

         /* XXX: Handle if uses */

         nir_foreach_def(dest, reg) {
            _mesa_set_add(register_uses, dest->reg.parent_instr);
         }
      }

      nir_foreach_block(block, function->impl) {
         nir_foreach_instr(instr, block) {
            nir_foreach_ssa_def(instr, nir_schedule_ssa_def_init_scoreboard,
                                scoreboard);
         }

         /* XXX: We're ignoring if uses, which may prioritize scheduling other
          * uses of the if src even when it doesn't help.  That's not many
          * values, though, so meh.
          */
      }
   }

   return scoreboard;
}

static void
nir_schedule_validate_uses(nir_schedule_scoreboard *scoreboard)
{
#ifdef NDEBUG
   return;
#endif

   bool any_uses = false;

   hash_table_foreach(scoreboard->remaining_uses, entry) {
      struct set *remaining_uses = entry->data;

      set_foreach(remaining_uses, instr_entry) {
         if (!any_uses) {
            fprintf(stderr, "Tracked uses remain after scheduling.  "
                    "Affected instructions: \n");
            any_uses = true;
         }
         nir_print_instr(instr_entry->key, stderr);
         fprintf(stderr, "\n");
      }
   }

   assert(!any_uses);
}

/**
 * Schedules the NIR instructions to try to decrease stalls (for example,
 * delaying texture reads) while managing register pressure.
 *
 * The threshold represents "number of NIR register/SSA def channels live
 * before switching the scheduling heuristic to reduce register pressure",
 * since most of our GPU architectures are scalar (extending to vector with a
 * flag wouldn't be hard).  This number should be a bit below the number of
 * registers available (counting any that may be occupied by system value
 * payload values, for example), since the heuristic may not always be able to
 * free a register immediately.  The amount below the limit is up to you to
 * tune.
 */
void
nir_schedule(nir_shader *shader,
             const nir_schedule_options *options)
{
   nir_schedule_scoreboard *scoreboard = nir_schedule_get_scoreboard(shader,
                                                                     options);

   if (debug) {
      fprintf(stderr, "NIR shader before scheduling:\n");
      nir_print_shader(shader, stderr);
   }

   nir_foreach_function(function, shader) {
      if (!function->impl)
         continue;

      nir_foreach_block(block, function->impl) {
         nir_schedule_block(scoreboard, block);
      }
   }

   nir_schedule_validate_uses(scoreboard);

   ralloc_free(scoreboard);
}
