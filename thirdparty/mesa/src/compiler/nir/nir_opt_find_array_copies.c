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

#include "nir.h"
#include "nir_builder.h"
#include "nir_deref.h"

struct match_node {
   /* Note: these fields are only valid for leaf nodes */

   unsigned next_array_idx;
   int src_wildcard_idx;
   nir_deref_path first_src_path;

   /* The index of the first read of the source path that's part of the copy
    * we're matching. If the last write to the source path is after this, we
    * would get a different result from reading it at the end and we can't
    * emit the copy.
    */
   unsigned first_src_read;

   /* The last time there was a write to this node. */
   unsigned last_overwritten;

   /* The last time there was a write to this node which successfully advanced
    * next_array_idx. This helps us catch any intervening aliased writes.
    */
   unsigned last_successful_write;

   unsigned num_children;
   struct match_node *children[];
};

struct match_state {
   /* Map from nir_variable * -> match_node */
   struct hash_table *var_nodes;
   /* Map from cast nir_deref_instr * -> match_node */
   struct hash_table *cast_nodes;

   unsigned cur_instr;

   nir_builder builder;

   void *dead_ctx;
};

static struct match_node *
create_match_node(const struct glsl_type *type, struct match_state *state)
{
   unsigned num_children = 0;
   if (glsl_type_is_array_or_matrix(type)) {
      /* One for wildcards */
      num_children = glsl_get_length(type) + 1;
   } else if (glsl_type_is_struct_or_ifc(type)) {
      num_children = glsl_get_length(type);
   }

   struct match_node *node = rzalloc_size(state->dead_ctx,
                                          sizeof(struct match_node) +
                                          num_children * sizeof(struct match_node *));
   node->num_children = num_children;
   node->src_wildcard_idx = -1;
   node->first_src_read = UINT32_MAX;
   return node;
}

static struct match_node *
node_for_deref(nir_deref_instr *instr, struct match_node *parent,
               struct match_state *state)
{
   unsigned idx;
   switch (instr->deref_type) {
   case nir_deref_type_var: {
      struct hash_entry *entry =
         _mesa_hash_table_search(state->var_nodes, instr->var);
      if (entry) {
         return entry->data;
      } else {
         struct match_node *node = create_match_node(instr->type, state);
         _mesa_hash_table_insert(state->var_nodes, instr->var, node);
         return node;
      }
   }

   case nir_deref_type_cast: {
      struct hash_entry *entry =
         _mesa_hash_table_search(state->cast_nodes, instr);
      if (entry) {
         return entry->data;
      } else {
         struct match_node *node = create_match_node(instr->type, state);
         _mesa_hash_table_insert(state->cast_nodes, instr, node);
         return node;
      }
   }

   case nir_deref_type_array_wildcard:
      idx = parent->num_children - 1;
      break;

   case nir_deref_type_array:
      if (nir_src_is_const(instr->arr.index)) {
         idx = nir_src_as_uint(instr->arr.index);
         assert(idx < parent->num_children - 1);
      } else {
         idx = parent->num_children - 1;
      }
      break;

   case nir_deref_type_struct:
      idx = instr->strct.index;
      break;

   default:
      unreachable("bad deref type");
   }

   assert(idx < parent->num_children);
   if (parent->children[idx]) {
      return parent->children[idx];
   } else {
      struct match_node *node = create_match_node(instr->type, state);
      parent->children[idx] = node;
      return node;
   }
}

static struct match_node *
node_for_wildcard(const struct glsl_type *type, struct match_node *parent,
                  struct match_state *state)
{
   assert(glsl_type_is_array_or_matrix(type));
   unsigned idx = glsl_get_length(type);

   if (parent->children[idx]) {
      return parent->children[idx];
   } else {
      struct match_node *node =
         create_match_node(glsl_get_array_element(type), state);
      parent->children[idx] = node;
      return node;
   }
}

static struct match_node *
node_for_path(nir_deref_path *path, struct match_state *state)
{
   struct match_node *node = NULL;
   for (nir_deref_instr **instr = path->path; *instr; instr++)
      node = node_for_deref(*instr, node, state);

   return node;
}

static struct match_node *
node_for_path_with_wildcard(nir_deref_path *path, unsigned wildcard_idx,
                            struct match_state *state)
{
   struct match_node *node = NULL;
   unsigned idx = 0;
   for (nir_deref_instr **instr = path->path; *instr; instr++, idx++) {
      if (idx == wildcard_idx)
         node = node_for_wildcard((*(instr - 1))->type, node, state);
      else
         node = node_for_deref(*instr, node, state);
   }

   return node;
}

typedef void (*match_cb)(struct match_node *, struct match_state *);

static void
_foreach_child(match_cb cb, struct match_node *node, struct match_state *state)
{
   if (node->num_children == 0) {
      cb(node, state);
   } else {
      for (unsigned i = 0; i < node->num_children; i++) {
         if (node->children[i])
            _foreach_child(cb, node->children[i], state);
      }
   }
}

static void
_foreach_aliasing(nir_deref_instr **deref, match_cb cb,
                  struct match_node *node, struct match_state *state)
{
   if (*deref == NULL) {
      cb(node, state);
      return;
   }

   switch ((*deref)->deref_type) {
   case nir_deref_type_struct: {
      struct match_node *child = node->children[(*deref)->strct.index];
      if (child)
         _foreach_aliasing(deref + 1, cb, child, state);
      return;
   }

   case nir_deref_type_array:
   case nir_deref_type_array_wildcard: {
      if ((*deref)->deref_type == nir_deref_type_array_wildcard ||
          !nir_src_is_const((*deref)->arr.index)) {
         /* This access may touch any index, so we have to visit all of
          * them.
          */
         for (unsigned i = 0; i < node->num_children; i++) {
            if (node->children[i])
               _foreach_aliasing(deref + 1, cb, node->children[i], state);
         }
      } else {
         /* Visit the wildcard entry if any */
         if (node->children[node->num_children - 1]) {
            _foreach_aliasing(deref + 1, cb,
                              node->children[node->num_children - 1], state);
         }

         unsigned index = nir_src_as_uint((*deref)->arr.index);
         /* Check that the index is in-bounds */
         if (index < node->num_children - 1 && node->children[index])
            _foreach_aliasing(deref + 1, cb, node->children[index], state);
      }
      return;
   }

   case nir_deref_type_cast:
      _foreach_child(cb, node, state);
      return;

   default:
      unreachable("bad deref type");
   }
}

/* Given a deref path, find all the leaf deref nodes that alias it. */

static void
foreach_aliasing_node(nir_deref_path *path,
                      match_cb cb,
                      struct match_state *state)
{
   if (path->path[0]->deref_type == nir_deref_type_var) {
      struct hash_entry *entry = _mesa_hash_table_search(state->var_nodes,
                                                         path->path[0]->var);
      if (entry)
         _foreach_aliasing(&path->path[1], cb, entry->data, state);

      hash_table_foreach(state->cast_nodes, entry)
         _foreach_child(cb, entry->data, state);
   } else {
      /* Casts automatically alias anything that isn't a cast */
      assert(path->path[0]->deref_type == nir_deref_type_cast);
      hash_table_foreach(state->var_nodes, entry)
         _foreach_child(cb, entry->data, state);

      /* Casts alias other casts if the casts are different or if they're the
       * same and the path from the cast may alias as per the usual rules.
       */
      hash_table_foreach(state->cast_nodes, entry) {
         const nir_deref_instr *cast = entry->key;
         assert(cast->deref_type == nir_deref_type_cast);
         if (cast == path->path[0])
            _foreach_aliasing(&path->path[1], cb, entry->data, state);
         else
            _foreach_child(cb, entry->data, state);
      }
   }
}

static nir_deref_instr *
build_wildcard_deref(nir_builder *b, nir_deref_path *path,
                     unsigned wildcard_idx)
{
   assert(path->path[wildcard_idx]->deref_type == nir_deref_type_array);

   nir_deref_instr *tail =
      nir_build_deref_array_wildcard(b, path->path[wildcard_idx - 1]);

   for (unsigned i = wildcard_idx + 1; path->path[i]; i++)
      tail = nir_build_deref_follower(b, tail, path->path[i]);

   return tail;
}

static void
clobber(struct match_node *node, struct match_state *state)
{
   node->last_overwritten = state->cur_instr;
}

static bool
try_match_deref(nir_deref_path *base_path, int *path_array_idx,
                nir_deref_path *deref_path, int arr_idx,
                nir_deref_instr *dst)
{
   for (int i = 0; ; i++) {
      nir_deref_instr *b = base_path->path[i];
      nir_deref_instr *d = deref_path->path[i];
      /* They have to be the same length */
      if ((b == NULL) != (d == NULL))
         return false;

      if (b == NULL)
         break;

      /* This can happen if one is a deref_array and the other a wildcard */
      if (b->deref_type != d->deref_type)
         return false;;

      switch (b->deref_type) {
      case nir_deref_type_var:
         if (b->var != d->var)
            return false;
         continue;

      case nir_deref_type_array:
         assert(b->arr.index.is_ssa && d->arr.index.is_ssa);
         const bool const_b_idx = nir_src_is_const(b->arr.index);
         const bool const_d_idx = nir_src_is_const(d->arr.index);
         const unsigned b_idx = const_b_idx ? nir_src_as_uint(b->arr.index) : 0;
         const unsigned d_idx = const_d_idx ? nir_src_as_uint(d->arr.index) : 0;

         /* If we don't have an index into the path yet or if this entry in
          * the path is at the array index, see if this is a candidate.  We're
          * looking for an index which is zero in the base deref and arr_idx
          * in the search deref and has a matching array size.
          */
         if ((*path_array_idx < 0 || *path_array_idx == i) &&
             const_b_idx && b_idx == 0 &&
             const_d_idx && d_idx == arr_idx &&
             glsl_get_length(nir_deref_instr_parent(b)->type) ==
             glsl_get_length(nir_deref_instr_parent(dst)->type)) {
            *path_array_idx = i;
            continue;
         }

         /* We're at the array index but not a candidate */
         if (*path_array_idx == i)
            return false;

         /* If we're not the path array index, we must match exactly.  We
          * could probably just compare SSA values and trust in copy
          * propagation but doing it ourselves means this pass can run a bit
          * earlier.
          */
         if (b->arr.index.ssa == d->arr.index.ssa ||
             (const_b_idx && const_d_idx && b_idx == d_idx))
            continue;

         return false;

      case nir_deref_type_array_wildcard:
         continue;

      case nir_deref_type_struct:
         if (b->strct.index != d->strct.index)
            return false;
         continue;

      default:
         unreachable("Invalid deref type in a path");
      }
   }

   /* If we got here without failing, we've matched.  However, it isn't an
    * array match unless we found an altered array index.
    */
   return *path_array_idx > 0;
}

static void
handle_read(nir_deref_instr *src, struct match_state *state)
{
   /* We only need to create an entry for sources that might be used to form
    * an array copy. Hence no indirects or indexing into a vector.
    */
   if (nir_deref_instr_has_indirect(src) ||
       nir_deref_instr_is_known_out_of_bounds(src) ||
       (src->deref_type == nir_deref_type_array &&
        glsl_type_is_vector(nir_src_as_deref(src->parent)->type)))
      return;

   nir_deref_path src_path;
   nir_deref_path_init(&src_path, src, state->dead_ctx);

   /* Create a node for this source if it doesn't exist. The point of this is
    * to know which nodes aliasing a given store we actually need to care
    * about, to avoid creating an excessive amount of nodes.
    */
   node_for_path(&src_path, state);
}

/* The core implementation, which is used for both copies and writes. Return
 * true if a copy is created.
 */
static bool
handle_write(nir_deref_instr *dst, nir_deref_instr *src,
             unsigned write_index, unsigned read_index,
             struct match_state *state)
{
   nir_builder *b = &state->builder;

   nir_deref_path dst_path;
   nir_deref_path_init(&dst_path, dst, state->dead_ctx);

   unsigned idx = 0;
   for (nir_deref_instr **instr = dst_path.path; *instr; instr++, idx++) {
      if ((*instr)->deref_type != nir_deref_type_array)
         continue;

      /* Get the entry where the index is replaced by a wildcard, so that we
       * hopefully can keep matching an array copy.
       */
      struct match_node *dst_node =
         node_for_path_with_wildcard(&dst_path, idx, state);

      if (!src)
         goto reset;

      if (nir_src_as_uint((*instr)->arr.index) != dst_node->next_array_idx)
         goto reset;

      if (dst_node->next_array_idx == 0) {
         /* At this point there may be multiple source indices which are zero,
          * so we can't pin down the actual source index. Just store it and
          * move on.
          */
         nir_deref_path_init(&dst_node->first_src_path, src, state->dead_ctx);
      } else {
         nir_deref_path src_path;
         nir_deref_path_init(&src_path, src, state->dead_ctx);
         bool result = try_match_deref(&dst_node->first_src_path,
                                       &dst_node->src_wildcard_idx,
                                       &src_path, dst_node->next_array_idx,
                                       *instr);
         nir_deref_path_finish(&src_path);
         if (!result)
            goto reset;
      }

      /* Check if an aliasing write clobbered the array after the last normal
       * write. For example, with a sequence like this:
       *
       * dst[0][*] = src[0][*];
       * dst[0][0] = 0; // invalidates the array copy dst[*][*] = src[*][*]
       * dst[1][*] = src[1][*];
       *
       * Note that the second write wouldn't reset the entry for dst[*][*]
       * by itself, but it'll be caught by this check when processing the
       * third copy.
       */
      if (dst_node->last_successful_write < dst_node->last_overwritten)
         goto reset;

      dst_node->last_successful_write = write_index;

      /* In this case we've successfully processed an array element. Check if
       * this is the last, so that we can emit an array copy.
       */
      dst_node->next_array_idx++;
      dst_node->first_src_read = MIN2(dst_node->first_src_read, read_index);
      if (dst_node->next_array_idx > 1 &&
          dst_node->next_array_idx == glsl_get_length((*(instr - 1))->type)) {
         /* Make sure that nothing was overwritten. */
         struct match_node *src_node =
            node_for_path_with_wildcard(&dst_node->first_src_path,
                                        dst_node->src_wildcard_idx,
                                        state);

         if (src_node->last_overwritten <= dst_node->first_src_read) {
            nir_copy_deref(b, build_wildcard_deref(b, &dst_path, idx),
                              build_wildcard_deref(b, &dst_node->first_src_path,
                                                   dst_node->src_wildcard_idx));
            foreach_aliasing_node(&dst_path, clobber, state);
            return true;
         }
      } else {
         continue;
      }

reset:
      dst_node->next_array_idx = 0;
      dst_node->src_wildcard_idx = -1;
      dst_node->last_successful_write = 0;
      dst_node->first_src_read = UINT32_MAX;
   }

   /* Mark everything aliasing dst_path as clobbered. This needs to happen
    * last since in the loop above we need to know what last clobbered
    * dst_node and this overwrites that.
    */
   foreach_aliasing_node(&dst_path, clobber, state);

   return false;
}

static bool
opt_find_array_copies_block(nir_builder *b, nir_block *block,
                            struct match_state *state)
{
   bool progress = false;

   unsigned next_index = 0;

   _mesa_hash_table_clear(state->var_nodes, NULL);
   _mesa_hash_table_clear(state->cast_nodes, NULL);

   nir_foreach_instr(instr, block) {
      if (instr->type != nir_instr_type_intrinsic)
         continue;

      /* Index the instructions before we do anything else. */
      instr->index = next_index++;

      /* Save the index of this instruction */
      state->cur_instr = instr->index;

      nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);

      if (intrin->intrinsic == nir_intrinsic_load_deref) {
         handle_read(nir_src_as_deref(intrin->src[0]), state);
         continue;
      }

      if (intrin->intrinsic != nir_intrinsic_copy_deref &&
          intrin->intrinsic != nir_intrinsic_store_deref)
         continue;

      nir_deref_instr *dst_deref = nir_src_as_deref(intrin->src[0]);

      /* The destination must be local.  If we see a non-local store, we
       * continue on because it won't affect local stores or read-only
       * variables.
       */
      if (!nir_deref_mode_may_be(dst_deref, nir_var_function_temp))
         continue;

      if (!nir_deref_mode_must_be(dst_deref, nir_var_function_temp)) {
         /* This only happens if we have something that might be a local store
          * but we don't know.  In this case, clear everything.
          */
         nir_deref_path dst_path;
         nir_deref_path_init(&dst_path, dst_deref, state->dead_ctx);
         foreach_aliasing_node(&dst_path, clobber, state);
         continue;
      }

      /* If there are any known out-of-bounds writes, then we can just skip
       * this write as it's undefined and won't contribute to building up an
       * array copy anyways.
       */
      if (nir_deref_instr_is_known_out_of_bounds(dst_deref))
         continue;

      nir_deref_instr *src_deref;
      unsigned load_index = 0;
      if (intrin->intrinsic == nir_intrinsic_copy_deref) {
         src_deref = nir_src_as_deref(intrin->src[1]);
         load_index = intrin->instr.index;
      } else {
         assert(intrin->intrinsic == nir_intrinsic_store_deref);
         nir_intrinsic_instr *load = nir_src_as_intrinsic(intrin->src[1]);
         if (load == NULL || load->intrinsic != nir_intrinsic_load_deref) {
            src_deref = NULL;
         } else {
            src_deref = nir_src_as_deref(load->src[0]);
            load_index = load->instr.index;
         }

         if (nir_intrinsic_write_mask(intrin) !=
             (1 << glsl_get_components(dst_deref->type)) - 1) {
            src_deref = NULL;
         }
      }

      /* The source must be either local or something that's guaranteed to be
       * read-only.
       */
      if (src_deref &&
          !nir_deref_mode_must_be(src_deref, nir_var_function_temp |
                                             nir_var_read_only_modes)) {
         src_deref = NULL;
      }

      /* There must be no indirects in the source or destination and no known
       * out-of-bounds accesses in the source, and the copy must be fully
       * qualified, or else we can't build up the array copy. We handled
       * out-of-bounds accesses to the dest above. The types must match, since
       * copy_deref currently can't bitcast mismatched deref types.
       */
      if (src_deref &&
          (nir_deref_instr_has_indirect(src_deref) ||
           nir_deref_instr_is_known_out_of_bounds(src_deref) ||
           nir_deref_instr_has_indirect(dst_deref) ||
           !glsl_type_is_vector_or_scalar(src_deref->type) ||
           glsl_get_bare_type(src_deref->type) !=
           glsl_get_bare_type(dst_deref->type))) {
         src_deref = NULL;
      }

      state->builder.cursor = nir_after_instr(instr);
      progress |= handle_write(dst_deref, src_deref, instr->index,
                               load_index, state);
   }

   return progress;
}

static bool
opt_find_array_copies_impl(nir_function_impl *impl)
{
   nir_builder b;
   nir_builder_init(&b, impl);

   bool progress = false;

   struct match_state s;
   s.dead_ctx = ralloc_context(NULL);
   s.var_nodes = _mesa_pointer_hash_table_create(s.dead_ctx);
   s.cast_nodes = _mesa_pointer_hash_table_create(s.dead_ctx);
   nir_builder_init(&s.builder, impl);

   nir_foreach_block(block, impl) {
      if (opt_find_array_copies_block(&b, block, &s))
         progress = true;
   }

   ralloc_free(s.dead_ctx);

   if (progress) {
      nir_metadata_preserve(impl, nir_metadata_block_index |
                                  nir_metadata_dominance);
   } else {
      nir_metadata_preserve(impl, nir_metadata_all);
   }

   return progress;
}

/**
 * This peephole optimization looks for a series of load/store_deref or
 * copy_deref instructions that copy an array from one variable to another and
 * turns it into a copy_deref that copies the entire array.  The pattern it
 * looks for is extremely specific but it's good enough to pick up on the
 * input array copies in DXVK and should also be able to pick up the sequence
 * generated by spirv_to_nir for a OpLoad of a large composite followed by
 * OpStore.
 *
 * TODO: Support out-of-order copies.
 */
bool
nir_opt_find_array_copies(nir_shader *shader)
{
   bool progress = false;

   nir_foreach_function(function, shader) {
      if (function->impl && opt_find_array_copies_impl(function->impl))
         progress = true;
   }

   return progress;
}
