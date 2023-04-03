/*
 * Copyright © 2014 Intel Corporation
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
 *    Jason Ekstrand (jason@jlekstrand.net)
 *
 */

#include "nir.h"
#include "nir_builder.h"
#include "nir_deref.h"
#include "nir_phi_builder.h"
#include "nir_vla.h"


struct deref_node {
   struct deref_node *parent;
   const struct glsl_type *type;

   bool lower_to_ssa;

   /* Only valid for things that end up in the direct list.
    * Note that multiple nir_deref_instrs may correspond to this node, but
    * they will all be equivalent, so any is as good as the other.
    */
   nir_deref_path path;
   struct exec_node direct_derefs_link;

   struct set *loads;
   struct set *stores;
   struct set *copies;

   struct nir_phi_builder_value *pb_value;

   /* True if this node is fully direct.  If set, it must be in the children
    * array of its parent.
    */
   bool is_direct;

   /* Set on a root node for a variable to indicate that variable is used by a
    * cast or passed through some other sequence of instructions that are not
    * derefs.
    */
   bool has_complex_use;

   struct deref_node *wildcard;
   struct deref_node *indirect;
   struct deref_node *children[0];
};

#define UNDEF_NODE ((struct deref_node *)(uintptr_t)1)

struct lower_variables_state {
   nir_shader *shader;
   void *dead_ctx;
   nir_function_impl *impl;

   /* A hash table mapping variables to deref_node data */
   struct hash_table *deref_var_nodes;

   /* A hash table mapping fully-qualified direct dereferences, i.e.
    * dereferences with no indirect or wildcard array dereferences, to
    * deref_node data.
    *
    * At the moment, we only lower loads, stores, and copies that can be
    * trivially lowered to loads and stores, i.e. copies with no indirects
    * and no wildcards.  If a part of a variable that is being loaded from
    * and/or stored into is also involved in a copy operation with
    * wildcards, then we lower that copy operation to loads and stores, but
    * otherwise we leave copies with wildcards alone. Since the only derefs
    * used in these loads, stores, and trivial copies are ones with no
    * wildcards and no indirects, these are precisely the derefs that we
    * can actually consider lowering.
    */
   struct exec_list direct_deref_nodes;

   /* Controls whether get_deref_node will add variables to the
    * direct_deref_nodes table.  This is turned on when we are initially
    * scanning for load/store instructions.  It is then turned off so we
    * don't accidentally change the direct_deref_nodes table while we're
    * iterating throug it.
    */
   bool add_to_direct_deref_nodes;

   struct nir_phi_builder *phi_builder;
};

static struct deref_node *
deref_node_create(struct deref_node *parent,
                  const struct glsl_type *type,
                  bool is_direct, void *mem_ctx)
{
   size_t size = sizeof(struct deref_node) +
                 glsl_get_length(type) * sizeof(struct deref_node *);

   struct deref_node *node = rzalloc_size(mem_ctx, size);
   node->type = type;
   node->parent = parent;
   exec_node_init(&node->direct_derefs_link);
   node->is_direct = is_direct;

   return node;
}

/* Returns the deref node associated with the given variable.  This will be
 * the root of the tree representing all of the derefs of the given variable.
 */
static struct deref_node *
get_deref_node_for_var(nir_variable *var, struct lower_variables_state *state)
{
   struct deref_node *node;

   struct hash_entry *var_entry =
      _mesa_hash_table_search(state->deref_var_nodes, var);

   if (var_entry) {
      return var_entry->data;
   } else {
      node = deref_node_create(NULL, var->type, true, state->dead_ctx);
      _mesa_hash_table_insert(state->deref_var_nodes, var, node);
      return node;
   }
}

/* Gets the deref_node for the given deref chain and creates it if it
 * doesn't yet exist.  If the deref is fully-qualified and direct and
 * state->add_to_direct_deref_nodes is true, it will be added to the hash
 * table of of fully-qualified direct derefs.
 */
static struct deref_node *
get_deref_node_recur(nir_deref_instr *deref,
                     struct lower_variables_state *state)
{
   if (deref->deref_type == nir_deref_type_var)
      return get_deref_node_for_var(deref->var, state);

   if (deref->deref_type == nir_deref_type_cast)
      return NULL;

   struct deref_node *parent =
      get_deref_node_recur(nir_deref_instr_parent(deref), state);
   if (parent == NULL)
      return NULL;

   if (parent == UNDEF_NODE)
      return UNDEF_NODE;

   switch (deref->deref_type) {
   case nir_deref_type_struct:
      assert(glsl_type_is_struct_or_ifc(parent->type));
      assert(deref->strct.index < glsl_get_length(parent->type));

      if (parent->children[deref->strct.index] == NULL) {
         parent->children[deref->strct.index] =
            deref_node_create(parent, deref->type, parent->is_direct,
                              state->dead_ctx);
      }

      return parent->children[deref->strct.index];

   case nir_deref_type_array: {
      if (nir_src_is_const(deref->arr.index)) {
         uint32_t index = nir_src_as_uint(deref->arr.index);
         /* This is possible if a loop unrolls and generates an
          * out-of-bounds offset.  We need to handle this at least
          * somewhat gracefully.
          */
         if (index >= glsl_get_length(parent->type))
            return UNDEF_NODE;

         if (parent->children[index] == NULL) {
            parent->children[index] =
               deref_node_create(parent, deref->type, parent->is_direct,
                                 state->dead_ctx);
         }

         return parent->children[index];
      } else {
         if (parent->indirect == NULL) {
            parent->indirect =
               deref_node_create(parent, deref->type, false, state->dead_ctx);
         }

         return parent->indirect;
      }
      break;
   }

   case nir_deref_type_array_wildcard:
      if (parent->wildcard == NULL) {
         parent->wildcard =
            deref_node_create(parent, deref->type, false, state->dead_ctx);
      }

      return parent->wildcard;

   default:
      unreachable("Invalid deref type");
   }
}

static struct deref_node *
get_deref_node(nir_deref_instr *deref, struct lower_variables_state *state)
{
   /* This pass only works on local variables.  Just ignore any derefs with
    * a non-local mode.
    */
   if (!nir_deref_mode_must_be(deref, nir_var_function_temp))
      return NULL;

   struct deref_node *node = get_deref_node_recur(deref, state);
   if (!node)
      return NULL;

   /* Insert the node in the direct derefs list.  We only do this if it's not
    * already in the list and we only bother for deref nodes which are used
    * directly in a load or store.
    */
   if (node != UNDEF_NODE && node->is_direct &&
       state->add_to_direct_deref_nodes &&
       node->direct_derefs_link.next == NULL) {
      nir_deref_path_init(&node->path, deref, state->dead_ctx);
      assert(deref->var != NULL);
      exec_list_push_tail(&state->direct_deref_nodes,
                          &node->direct_derefs_link);
   }

   return node;
}

/* \sa foreach_deref_node_match */
static void
foreach_deref_node_worker(struct deref_node *node, nir_deref_instr **path,
                          void (* cb)(struct deref_node *node,
                                      struct lower_variables_state *state),
                          struct lower_variables_state *state)
{
   if (*path == NULL) {
      cb(node, state);
      return;
   }

   switch ((*path)->deref_type) {
   case nir_deref_type_struct:
      if (node->children[(*path)->strct.index]) {
         foreach_deref_node_worker(node->children[(*path)->strct.index],
                                   path + 1, cb, state);
      }
      return;

   case nir_deref_type_array: {
      uint32_t index = nir_src_as_uint((*path)->arr.index);

      if (node->children[index]) {
         foreach_deref_node_worker(node->children[index],
                                   path + 1, cb, state);
      }

      if (node->wildcard) {
         foreach_deref_node_worker(node->wildcard,
                                   path + 1, cb, state);
      }
      return;
   }

   default:
      unreachable("Unsupported deref type");
   }
}

/* Walks over every "matching" deref_node and calls the callback.  A node
 * is considered to "match" if either refers to that deref or matches up t
 * a wildcard.  In other words, the following would match a[6].foo[3].bar:
 *
 * a[6].foo[3].bar
 * a[*].foo[3].bar
 * a[6].foo[*].bar
 * a[*].foo[*].bar
 *
 * The given deref must be a full-length and fully qualified (no wildcards
 * or indirects) deref chain.
 */
static void
foreach_deref_node_match(nir_deref_path *path,
                         void (* cb)(struct deref_node *node,
                                     struct lower_variables_state *state),
                         struct lower_variables_state *state)
{
   assert(path->path[0]->deref_type == nir_deref_type_var);
   struct deref_node *node = get_deref_node_for_var(path->path[0]->var, state);

   if (node == NULL)
      return;

   foreach_deref_node_worker(node, &path->path[1], cb, state);
}

/* \sa deref_may_be_aliased */
static bool
path_may_be_aliased_node(struct deref_node *node, nir_deref_instr **path,
                         struct lower_variables_state *state)
{
   if (*path == NULL)
      return false;

   switch ((*path)->deref_type) {
   case nir_deref_type_struct:
      if (node->children[(*path)->strct.index]) {
         return path_may_be_aliased_node(node->children[(*path)->strct.index],
                                         path + 1, state);
      } else {
         return false;
      }

   case nir_deref_type_array: {
      if (!nir_src_is_const((*path)->arr.index))
         return true;

      uint32_t index = nir_src_as_uint((*path)->arr.index);

      /* If there is an indirect at this level, we're aliased. */
      if (node->indirect)
         return true;

      if (node->children[index] &&
          path_may_be_aliased_node(node->children[index],
                                   path + 1, state))
         return true;

      if (node->wildcard &&
          path_may_be_aliased_node(node->wildcard, path + 1, state))
         return true;

      return false;
   }

   default:
      unreachable("Unsupported deref type");
   }
}

/* Returns true if there are no indirects that can ever touch this deref.
 *
 * For example, if the given deref is a[6].foo, then any uses of a[i].foo
 * would cause this to return false, but a[i].bar would not affect it
 * because it's a different structure member.  A var_copy involving of
 * a[*].bar also doesn't affect it because that can be lowered to entirely
 * direct load/stores.
 *
 * We only support asking this question about fully-qualified derefs.
 * Obviously, it's pointless to ask this about indirects, but we also
 * rule-out wildcards.  Handling Wildcard dereferences would involve
 * checking each array index to make sure that there aren't any indirect
 * references.
 */
static bool
path_may_be_aliased(nir_deref_path *path,
                    struct lower_variables_state *state)
{
   assert(path->path[0]->deref_type == nir_deref_type_var);
   nir_variable *var = path->path[0]->var;
   struct deref_node *var_node = get_deref_node_for_var(var, state);

   /* First see if this variable is ever used by anything other than a
    * load/store.  If there's even so much as a cast in the way, we have to
    * assume aliasing and bail.
    */
   if (var_node->has_complex_use)
      return true;

   return path_may_be_aliased_node(var_node, &path->path[1], state);
}

static void
register_complex_use(nir_deref_instr *deref,
                     struct lower_variables_state *state)
{
   assert(deref->deref_type == nir_deref_type_var);
   struct deref_node *node = get_deref_node_for_var(deref->var, state);
   if (node == NULL)
      return;

   node->has_complex_use = true;
}

static bool
register_load_instr(nir_intrinsic_instr *load_instr,
                    struct lower_variables_state *state)
{
   nir_deref_instr *deref = nir_src_as_deref(load_instr->src[0]);
   struct deref_node *node = get_deref_node(deref, state);
   if (node == NULL)
      return false;

   /* Replace out-of-bounds load derefs with an undef, so that they don't get
    * left around when a driver has lowered all indirects and thus doesn't
    * expect any array derefs at all after vars_to_ssa.
    */
   if (node == UNDEF_NODE) {
      nir_ssa_undef_instr *undef =
         nir_ssa_undef_instr_create(state->shader,
                                    load_instr->num_components,
                                    load_instr->dest.ssa.bit_size);

      nir_instr_insert_before(&load_instr->instr, &undef->instr);
      nir_instr_remove(&load_instr->instr);

      nir_ssa_def_rewrite_uses(&load_instr->dest.ssa, &undef->def);
      return true;
   }

   if (node->loads == NULL)
      node->loads = _mesa_pointer_set_create(state->dead_ctx);

   _mesa_set_add(node->loads, load_instr);

   return false;
}

static bool
register_store_instr(nir_intrinsic_instr *store_instr,
                     struct lower_variables_state *state)
{
   nir_deref_instr *deref = nir_src_as_deref(store_instr->src[0]);
   struct deref_node *node = get_deref_node(deref, state);

   /* Drop out-of-bounds store derefs, so that they don't get left around when a
    * driver has lowered all indirects and thus doesn't expect any array derefs
    * at all after vars_to_ssa.
    */
   if (node == UNDEF_NODE) {
      nir_instr_remove(&store_instr->instr);
      return true;
   }

   if (node == NULL)
      return false;

   if (node->stores == NULL)
      node->stores = _mesa_pointer_set_create(state->dead_ctx);

   _mesa_set_add(node->stores, store_instr);

   return false;
}

static void
register_copy_instr(nir_intrinsic_instr *copy_instr,
                    struct lower_variables_state *state)
{
   for (unsigned idx = 0; idx < 2; idx++) {
      nir_deref_instr *deref = nir_src_as_deref(copy_instr->src[idx]);
      struct deref_node *node = get_deref_node(deref, state);
      if (node == NULL || node == UNDEF_NODE)
         continue;

      if (node->copies == NULL)
         node->copies = _mesa_pointer_set_create(state->dead_ctx);

      _mesa_set_add(node->copies, copy_instr);
   }
}

static bool
register_variable_uses(nir_function_impl *impl,
                       struct lower_variables_state *state)
{
   bool progress = false;

   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         switch (instr->type) {
         case nir_instr_type_deref: {
            nir_deref_instr *deref = nir_instr_as_deref(instr);

            if (deref->deref_type == nir_deref_type_var &&
                nir_deref_instr_has_complex_use(deref, 0))
               register_complex_use(deref, state);

            break;
         }

         case nir_instr_type_intrinsic: {
            nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);

            switch (intrin->intrinsic) {
            case nir_intrinsic_load_deref:
               progress = register_load_instr(intrin, state) || progress;
               break;

            case nir_intrinsic_store_deref:
               progress = register_store_instr(intrin, state) || progress;
               break;

            case nir_intrinsic_copy_deref:
               register_copy_instr(intrin, state);
               break;

            default:
               continue;
            }
            break;
         }

         default:
            break;
         }
      }
   }
   return progress;
}

/* Walks over all of the copy instructions to or from the given deref_node
 * and lowers them to load/store intrinsics.
 */
static void
lower_copies_to_load_store(struct deref_node *node,
                           struct lower_variables_state *state)
{
   if (!node->copies)
      return;

   nir_builder b;
   nir_builder_init(&b, state->impl);

   set_foreach(node->copies, copy_entry) {
      nir_intrinsic_instr *copy = (void *)copy_entry->key;

      nir_lower_deref_copy_instr(&b, copy);

      for (unsigned i = 0; i < 2; ++i) {
         nir_deref_instr *arg_deref = nir_src_as_deref(copy->src[i]);
         struct deref_node *arg_node = get_deref_node(arg_deref, state);

         /* Only bother removing copy entries for other nodes */
         if (arg_node == NULL || arg_node == node)
            continue;

         struct set_entry *arg_entry = _mesa_set_search(arg_node->copies, copy);
         assert(arg_entry);
         _mesa_set_remove(arg_node->copies, arg_entry);
      }

      nir_instr_remove(&copy->instr);
   }

   node->copies = NULL;
}

/* Performs variable renaming
 *
 * This algorithm is very similar to the one outlined in "Efficiently
 * Computing Static Single Assignment Form and the Control Dependence
 * Graph" by Cytron et al.  The primary difference is that we only put one
 * SSA def on the stack per block.
 */
static bool
rename_variables(struct lower_variables_state *state)
{
   nir_builder b;
   nir_builder_init(&b, state->impl);

   nir_foreach_block(block, state->impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;

         nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);

         switch (intrin->intrinsic) {
         case nir_intrinsic_load_deref: {
            nir_deref_instr *deref = nir_src_as_deref(intrin->src[0]);
            if (!nir_deref_mode_must_be(deref, nir_var_function_temp))
               continue;

            struct deref_node *node = get_deref_node(deref, state);
            if (node == NULL)
               continue;

            /* Should have been removed before rename_variables(). */
            assert(node != UNDEF_NODE);

            if (!node->lower_to_ssa)
               continue;

            nir_alu_instr *mov = nir_alu_instr_create(state->shader,
                                                      nir_op_mov);
            mov->src[0].src = nir_src_for_ssa(
               nir_phi_builder_value_get_block_def(node->pb_value, block));
            for (unsigned i = intrin->num_components; i < NIR_MAX_VEC_COMPONENTS; i++)
               mov->src[0].swizzle[i] = 0;

            assert(intrin->dest.is_ssa);

            mov->dest.write_mask = (1 << intrin->num_components) - 1;
            nir_ssa_dest_init(&mov->instr, &mov->dest.dest,
                              intrin->num_components,
                              intrin->dest.ssa.bit_size, NULL);

            nir_instr_insert_before(&intrin->instr, &mov->instr);
            nir_instr_remove(&intrin->instr);

            nir_ssa_def_rewrite_uses(&intrin->dest.ssa,
                                     &mov->dest.dest.ssa);
            break;
         }

         case nir_intrinsic_store_deref: {
            nir_deref_instr *deref = nir_src_as_deref(intrin->src[0]);
            if (!nir_deref_mode_must_be(deref, nir_var_function_temp))
               continue;

            struct deref_node *node = get_deref_node(deref, state);
            if (node == NULL)
               continue;

            /* Should have been removed before rename_variables(). */
            assert(node != UNDEF_NODE);

            assert(intrin->src[1].is_ssa);
            nir_ssa_def *value = intrin->src[1].ssa;

            if (!node->lower_to_ssa)
               continue;

            assert(intrin->num_components ==
                   glsl_get_vector_elements(node->type));

            nir_ssa_def *new_def;
            b.cursor = nir_before_instr(&intrin->instr);

            unsigned wrmask = nir_intrinsic_write_mask(intrin);
            if (wrmask == (1 << intrin->num_components) - 1) {
               /* Whole variable store - just copy the source.  Note that
                * intrin->num_components and value->num_components
                * may differ.
                */
               unsigned swiz[NIR_MAX_VEC_COMPONENTS];
               for (unsigned i = 0; i < NIR_MAX_VEC_COMPONENTS; i++)
                  swiz[i] = i < intrin->num_components ? i : 0;

               new_def = nir_swizzle(&b, value, swiz,
                                     intrin->num_components);
            } else {
               nir_ssa_def *old_def =
                  nir_phi_builder_value_get_block_def(node->pb_value, block);
               /* For writemasked store_var intrinsics, we combine the newly
                * written values with the existing contents of unwritten
                * channels, creating a new SSA value for the whole vector.
                */
               nir_ssa_scalar srcs[NIR_MAX_VEC_COMPONENTS];
               for (unsigned i = 0; i < intrin->num_components; i++) {
                  if (wrmask & (1 << i)) {
                     srcs[i] = nir_get_ssa_scalar(value, i);
                  } else {
                     srcs[i] = nir_get_ssa_scalar(old_def, i);
                  }
               }
               new_def = nir_vec_scalars(&b, srcs, intrin->num_components);
            }

            assert(new_def->num_components == intrin->num_components);

            nir_phi_builder_value_set_block_def(node->pb_value, block, new_def);
            nir_instr_remove(&intrin->instr);
            break;
         }

         default:
            break;
         }
      }
   }

   return true;
}

/** Implements a pass to lower variable uses to SSA values
 *
 * This path walks the list of instructions and tries to lower as many
 * local variable load/store operations to SSA defs and uses as it can.
 * The process involves four passes:
 *
 *  1) Iterate over all of the instructions and mark where each local
 *     variable deref is used in a load, store, or copy.  While we're at
 *     it, we keep track of all of the fully-qualified (no wildcards) and
 *     fully-direct references we see and store them in the
 *     direct_deref_nodes hash table.
 *
 *  2) Walk over the list of fully-qualified direct derefs generated in
 *     the previous pass.  For each deref, we determine if it can ever be
 *     aliased, i.e. if there is an indirect reference anywhere that may
 *     refer to it.  If it cannot be aliased, we mark it for lowering to an
 *     SSA value.  At this point, we lower any var_copy instructions that
 *     use the given deref to load/store operations.
 *
 *  3) Walk over the list of derefs we plan to lower to SSA values and
 *     insert phi nodes as needed.
 *
 *  4) Perform "variable renaming" by replacing the load/store instructions
 *     with SSA definitions and SSA uses.
 */
static bool
nir_lower_vars_to_ssa_impl(nir_function_impl *impl)
{
   struct lower_variables_state state;

   state.shader = impl->function->shader;
   state.dead_ctx = ralloc_context(state.shader);
   state.impl = impl;

   state.deref_var_nodes = _mesa_pointer_hash_table_create(state.dead_ctx);
   exec_list_make_empty(&state.direct_deref_nodes);

   /* Build the initial deref structures and direct_deref_nodes table */
   state.add_to_direct_deref_nodes = true;

   bool progress = register_variable_uses(impl, &state);

   nir_metadata_require(impl, nir_metadata_block_index);

   /* We're about to iterate through direct_deref_nodes.  Don't modify it. */
   state.add_to_direct_deref_nodes = false;

   foreach_list_typed_safe(struct deref_node, node, direct_derefs_link,
                           &state.direct_deref_nodes) {
      nir_deref_path *path = &node->path;

      assert(path->path[0]->deref_type == nir_deref_type_var);

      /* We don't build deref nodes for non-local variables */
      assert(path->path[0]->var->data.mode == nir_var_function_temp);

      if (path_may_be_aliased(path, &state)) {
         exec_node_remove(&node->direct_derefs_link);
         continue;
      }

      node->lower_to_ssa = true;
      progress = true;

      foreach_deref_node_match(path, lower_copies_to_load_store, &state);
   }

   if (!progress) {
      nir_metadata_preserve(impl, nir_metadata_all);
      return false;
   }

   nir_metadata_require(impl, nir_metadata_dominance);

   /* We may have lowered some copy instructions to load/store
    * instructions.  The uses from the copy instructions hav already been
    * removed but we need to rescan to ensure that the uses from the newly
    * added load/store instructions are registered.  We need this
    * information for phi node insertion below.
    */
   register_variable_uses(impl, &state);

   state.phi_builder = nir_phi_builder_create(state.impl);

   BITSET_WORD *store_blocks =
      ralloc_array(state.dead_ctx, BITSET_WORD,
                   BITSET_WORDS(state.impl->num_blocks));
   foreach_list_typed(struct deref_node, node, direct_derefs_link,
                      &state.direct_deref_nodes) {
      if (!node->lower_to_ssa)
         continue;

      memset(store_blocks, 0,
             BITSET_WORDS(state.impl->num_blocks) * sizeof(*store_blocks));

      assert(node->path.path[0]->var->constant_initializer == NULL &&
             node->path.path[0]->var->pointer_initializer == NULL);

      if (node->stores) {
         set_foreach(node->stores, store_entry) {
            nir_intrinsic_instr *store =
               (nir_intrinsic_instr *)store_entry->key;
            BITSET_SET(store_blocks, store->instr.block->index);
         }
      }

      node->pb_value =
         nir_phi_builder_add_value(state.phi_builder,
                                   glsl_get_vector_elements(node->type),
                                   glsl_get_bit_size(node->type),
                                   store_blocks);
   }

   rename_variables(&state);

   nir_phi_builder_finish(state.phi_builder);

   nir_metadata_preserve(impl, nir_metadata_block_index |
                               nir_metadata_dominance);

   ralloc_free(state.dead_ctx);

   return progress;
}

bool
nir_lower_vars_to_ssa(nir_shader *shader)
{
   bool progress = false;

   nir_foreach_function(function, shader) {
      if (function->impl)
         progress |= nir_lower_vars_to_ssa_impl(function->impl);
   }

   return progress;
}
