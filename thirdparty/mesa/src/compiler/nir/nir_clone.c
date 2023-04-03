/*
 * Copyright © 2015 Red Hat
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
#include "nir_control_flow.h"
#include "nir_xfb_info.h"

/* Secret Decoder Ring:
 *   clone_foo():
 *        Allocate and clone a foo.
 *   __clone_foo():
 *        Clone body of foo (ie. parent class, embedded struct, etc)
 */

typedef struct {
   /* True if we are cloning an entire shader. */
   bool global_clone;

   /* If true allows the clone operation to fall back to the original pointer
    * if no clone pointer is found in the remap table.  This allows us to
    * clone a loop body without having to add srcs from outside the loop to
    * the remap table. This is useful for loop unrolling.
    */
   bool allow_remap_fallback;

   /* maps orig ptr -> cloned ptr: */
   struct hash_table *remap_table;

   /* List of phi sources. */
   struct list_head phi_srcs;

   /* new shader object, used as memctx for just about everything else: */
   nir_shader *ns;
} clone_state;

static void
init_clone_state(clone_state *state, struct hash_table *remap_table,
                 bool global, bool allow_remap_fallback)
{
   state->global_clone = global;
   state->allow_remap_fallback = allow_remap_fallback;

   if (remap_table) {
      state->remap_table = remap_table;
   } else {
      state->remap_table = _mesa_pointer_hash_table_create(NULL);
   }

   list_inithead(&state->phi_srcs);
}

static void
free_clone_state(clone_state *state)
{
   _mesa_hash_table_destroy(state->remap_table, NULL);
}

static inline void *
_lookup_ptr(clone_state *state, const void *ptr, bool global)
{
   struct hash_entry *entry;

   if (!ptr)
      return NULL;

   if (!state->global_clone && global)
      return (void *)ptr;

   if (unlikely(!state->remap_table)) {
      assert(state->allow_remap_fallback);
      return (void *)ptr;
   }

   entry = _mesa_hash_table_search(state->remap_table, ptr);
   if (!entry) {
      assert(state->allow_remap_fallback);
      return (void *)ptr;
   }

   return entry->data;
}

static void
add_remap(clone_state *state, void *nptr, const void *ptr)
{
   _mesa_hash_table_insert(state->remap_table, ptr, nptr);
}

static void *
remap_local(clone_state *state, const void *ptr)
{
   return _lookup_ptr(state, ptr, false);
}

static void *
remap_global(clone_state *state, const void *ptr)
{
   return _lookup_ptr(state, ptr, true);
}

static nir_register *
remap_reg(clone_state *state, const nir_register *reg)
{
   return _lookup_ptr(state, reg, false);
}

static nir_variable *
remap_var(clone_state *state, const nir_variable *var)
{
   return _lookup_ptr(state, var, nir_variable_is_global(var));
}

nir_constant *
nir_constant_clone(const nir_constant *c, nir_variable *nvar)
{
   nir_constant *nc = ralloc(nvar, nir_constant);

   memcpy(nc->values, c->values, sizeof(nc->values));
   nc->num_elements = c->num_elements;
   nc->elements = ralloc_array(nvar, nir_constant *, c->num_elements);
   for (unsigned i = 0; i < c->num_elements; i++) {
      nc->elements[i] = nir_constant_clone(c->elements[i], nvar);
   }

   return nc;
}

/* NOTE: for cloning nir_variables, bypass nir_variable_create to avoid
 * having to deal with locals and globals separately:
 */
nir_variable *
nir_variable_clone(const nir_variable *var, nir_shader *shader)
{
   nir_variable *nvar = rzalloc(shader, nir_variable);

   nvar->type = var->type;
   nvar->name = ralloc_strdup(nvar, var->name);
   nvar->data = var->data;
   nvar->num_state_slots = var->num_state_slots;
   if (var->num_state_slots) {
      nvar->state_slots = ralloc_array(nvar, nir_state_slot, var->num_state_slots);
      memcpy(nvar->state_slots, var->state_slots,
             var->num_state_slots * sizeof(nir_state_slot));
   }
   if (var->constant_initializer) {
      nvar->constant_initializer =
         nir_constant_clone(var->constant_initializer, nvar);
   }
   nvar->interface_type = var->interface_type;

   nvar->num_members = var->num_members;
   if (var->num_members) {
      nvar->members = ralloc_array(nvar, struct nir_variable_data,
                                   var->num_members);
      memcpy(nvar->members, var->members,
             var->num_members * sizeof(*var->members));
   }

   return nvar;
}

static nir_variable *
clone_variable(clone_state *state, const nir_variable *var)
{
   nir_variable *nvar = nir_variable_clone(var, state->ns);
   add_remap(state, nvar, var);

   return nvar;
}

/* clone list of nir_variable: */
static void
clone_var_list(clone_state *state, struct exec_list *dst,
               const struct exec_list *list)
{
   exec_list_make_empty(dst);
   foreach_list_typed(nir_variable, var, node, list) {
      nir_variable *nvar = clone_variable(state, var);
      exec_list_push_tail(dst, &nvar->node);
   }
}

/* NOTE: for cloning nir_registers, bypass nir_global/local_reg_create()
 * to avoid having to deal with locals and globals separately:
 */
static nir_register *
clone_register(clone_state *state, const nir_register *reg)
{
   nir_register *nreg = rzalloc(state->ns, nir_register);
   add_remap(state, nreg, reg);

   nreg->num_components = reg->num_components;
   nreg->bit_size = reg->bit_size;
   nreg->num_array_elems = reg->num_array_elems;
   nreg->index = reg->index;

   /* reconstructing uses/defs/if_uses handled by nir_instr_insert() */
   list_inithead(&nreg->uses);
   list_inithead(&nreg->defs);
   list_inithead(&nreg->if_uses);

   return nreg;
}

/* clone list of nir_register: */
static void
clone_reg_list(clone_state *state, struct exec_list *dst,
               const struct exec_list *list)
{
   exec_list_make_empty(dst);
   foreach_list_typed(nir_register, reg, node, list) {
      nir_register *nreg = clone_register(state, reg);
      exec_list_push_tail(dst, &nreg->node);
   }
}

static void
__clone_src(clone_state *state, void *ninstr_or_if,
            nir_src *nsrc, const nir_src *src)
{
   nsrc->is_ssa = src->is_ssa;
   if (src->is_ssa) {
      nsrc->ssa = remap_local(state, src->ssa);
   } else {
      nsrc->reg.reg = remap_reg(state, src->reg.reg);
      if (src->reg.indirect) {
         nsrc->reg.indirect = gc_alloc(state->ns->gctx, nir_src, 1);
         __clone_src(state, ninstr_or_if, nsrc->reg.indirect, src->reg.indirect);
      }
      nsrc->reg.base_offset = src->reg.base_offset;
   }
}

static void
__clone_dst(clone_state *state, nir_instr *ninstr,
            nir_dest *ndst, const nir_dest *dst)
{
   ndst->is_ssa = dst->is_ssa;
   if (dst->is_ssa) {
      nir_ssa_dest_init(ninstr, ndst, dst->ssa.num_components,
                        dst->ssa.bit_size, NULL);
      if (likely(state->remap_table))
         add_remap(state, &ndst->ssa, &dst->ssa);
   } else {
      ndst->reg.reg = remap_reg(state, dst->reg.reg);
      if (dst->reg.indirect) {
         ndst->reg.indirect = gc_alloc(state->ns->gctx, nir_src, 1);
         __clone_src(state, ninstr, ndst->reg.indirect, dst->reg.indirect);
      }
      ndst->reg.base_offset = dst->reg.base_offset;
   }
}

static nir_alu_instr *
clone_alu(clone_state *state, const nir_alu_instr *alu)
{
   nir_alu_instr *nalu = nir_alu_instr_create(state->ns, alu->op);
   nalu->exact = alu->exact;
   nalu->no_signed_wrap = alu->no_signed_wrap;
   nalu->no_unsigned_wrap = alu->no_unsigned_wrap;

   __clone_dst(state, &nalu->instr, &nalu->dest.dest, &alu->dest.dest);
   nalu->dest.saturate = alu->dest.saturate;
   nalu->dest.write_mask = alu->dest.write_mask;

   for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; i++) {
      __clone_src(state, &nalu->instr, &nalu->src[i].src, &alu->src[i].src);
      nalu->src[i].negate = alu->src[i].negate;
      nalu->src[i].abs = alu->src[i].abs;
      memcpy(nalu->src[i].swizzle, alu->src[i].swizzle,
             sizeof(nalu->src[i].swizzle));
   }

   return nalu;
}

nir_alu_instr *
nir_alu_instr_clone(nir_shader *shader, const nir_alu_instr *orig)
{
   clone_state state = {
      .allow_remap_fallback = true,
      .ns = shader,
   };
   return clone_alu(&state, orig);
}

static nir_deref_instr *
clone_deref_instr(clone_state *state, const nir_deref_instr *deref)
{
   nir_deref_instr *nderef =
      nir_deref_instr_create(state->ns, deref->deref_type);

   __clone_dst(state, &nderef->instr, &nderef->dest, &deref->dest);

   nderef->modes = deref->modes;
   nderef->type = deref->type;

   if (deref->deref_type == nir_deref_type_var) {
      nderef->var = remap_var(state, deref->var);
      return nderef;
   }

   __clone_src(state, &nderef->instr, &nderef->parent, &deref->parent);

   switch (deref->deref_type) {
   case nir_deref_type_struct:
      nderef->strct.index = deref->strct.index;
      break;

   case nir_deref_type_array:
   case nir_deref_type_ptr_as_array:
      __clone_src(state, &nderef->instr,
                  &nderef->arr.index, &deref->arr.index);
      nderef->arr.in_bounds = deref->arr.in_bounds;
      break;

   case nir_deref_type_array_wildcard:
      /* Nothing to do */
      break;

   case nir_deref_type_cast:
      nderef->cast.ptr_stride = deref->cast.ptr_stride;
      nderef->cast.align_mul = deref->cast.align_mul;
      nderef->cast.align_offset = deref->cast.align_offset;
      break;

   default:
      unreachable("Invalid instruction deref type");
   }

   return nderef;
}

static nir_intrinsic_instr *
clone_intrinsic(clone_state *state, const nir_intrinsic_instr *itr)
{
   nir_intrinsic_instr *nitr =
      nir_intrinsic_instr_create(state->ns, itr->intrinsic);

   unsigned num_srcs = nir_intrinsic_infos[itr->intrinsic].num_srcs;

   if (nir_intrinsic_infos[itr->intrinsic].has_dest)
      __clone_dst(state, &nitr->instr, &nitr->dest, &itr->dest);

   nitr->num_components = itr->num_components;
   memcpy(nitr->const_index, itr->const_index, sizeof(nitr->const_index));

   for (unsigned i = 0; i < num_srcs; i++)
      __clone_src(state, &nitr->instr, &nitr->src[i], &itr->src[i]);

   return nitr;
}

static nir_load_const_instr *
clone_load_const(clone_state *state, const nir_load_const_instr *lc)
{
   nir_load_const_instr *nlc =
      nir_load_const_instr_create(state->ns, lc->def.num_components,
                                  lc->def.bit_size);

   memcpy(&nlc->value, &lc->value, sizeof(*nlc->value) * lc->def.num_components);

   add_remap(state, &nlc->def, &lc->def);

   return nlc;
}

static nir_ssa_undef_instr *
clone_ssa_undef(clone_state *state, const nir_ssa_undef_instr *sa)
{
   nir_ssa_undef_instr *nsa =
      nir_ssa_undef_instr_create(state->ns, sa->def.num_components,
                                 sa->def.bit_size);

   add_remap(state, &nsa->def, &sa->def);

   return nsa;
}

static nir_tex_instr *
clone_tex(clone_state *state, const nir_tex_instr *tex)
{
   nir_tex_instr *ntex = nir_tex_instr_create(state->ns, tex->num_srcs);

   ntex->sampler_dim = tex->sampler_dim;
   ntex->dest_type = tex->dest_type;
   ntex->op = tex->op;
   __clone_dst(state, &ntex->instr, &ntex->dest, &tex->dest);
   for (unsigned i = 0; i < ntex->num_srcs; i++) {
      ntex->src[i].src_type = tex->src[i].src_type;
      __clone_src(state, &ntex->instr, &ntex->src[i].src, &tex->src[i].src);
   }
   ntex->coord_components = tex->coord_components;
   ntex->is_array = tex->is_array;
   ntex->array_is_lowered_cube = tex->array_is_lowered_cube;
   ntex->is_shadow = tex->is_shadow;
   ntex->is_new_style_shadow = tex->is_new_style_shadow;
   ntex->is_sparse = tex->is_sparse;
   ntex->component = tex->component;
   memcpy(ntex->tg4_offsets, tex->tg4_offsets, sizeof(tex->tg4_offsets));

   ntex->texture_index = tex->texture_index;
   ntex->sampler_index = tex->sampler_index;

   ntex->texture_non_uniform = tex->texture_non_uniform;
   ntex->sampler_non_uniform = tex->sampler_non_uniform;

   return ntex;
}

static nir_phi_instr *
clone_phi(clone_state *state, const nir_phi_instr *phi, nir_block *nblk)
{
   nir_phi_instr *nphi = nir_phi_instr_create(state->ns);

   __clone_dst(state, &nphi->instr, &nphi->dest, &phi->dest);

   /* Cloning a phi node is a bit different from other instructions.  The
    * sources of phi instructions are the only time where we can use an SSA
    * def before it is defined.  In order to handle this, we just copy over
    * the sources from the old phi instruction directly and then fix them up
    * in a second pass once all the instrutions in the function have been
    * properly cloned.
    *
    * In order to ensure that the copied sources (which are the same as the
    * old phi instruction's sources for now) don't get inserted into the old
    * shader's use-def lists, we have to add the phi instruction *before* we
    * set up its sources.
    */
   nir_instr_insert_after_block(nblk, &nphi->instr);

   nir_foreach_phi_src(src, phi) {
      nir_phi_src *nsrc = nir_phi_instr_add_src(nphi, src->pred, src->src);

      /* Stash it in the list of phi sources.  We'll walk this list and fix up
       * sources at the very end of clone_function_impl.
       */
      list_add(&nsrc->src.use_link, &state->phi_srcs);
   }

   return nphi;
}

static nir_jump_instr *
clone_jump(clone_state *state, const nir_jump_instr *jmp)
{
   /* These aren't handled because they require special block linking */
   assert(jmp->type != nir_jump_goto && jmp->type != nir_jump_goto_if);

   nir_jump_instr *njmp = nir_jump_instr_create(state->ns, jmp->type);

   return njmp;
}

static nir_call_instr *
clone_call(clone_state *state, const nir_call_instr *call)
{
   nir_function *ncallee = remap_global(state, call->callee);
   nir_call_instr *ncall = nir_call_instr_create(state->ns, ncallee);

   for (unsigned i = 0; i < ncall->num_params; i++)
      __clone_src(state, ncall, &ncall->params[i], &call->params[i]);

   return ncall;
}

static nir_instr *
clone_instr(clone_state *state, const nir_instr *instr)
{
   switch (instr->type) {
   case nir_instr_type_alu:
      return &clone_alu(state, nir_instr_as_alu(instr))->instr;
   case nir_instr_type_deref:
      return &clone_deref_instr(state, nir_instr_as_deref(instr))->instr;
   case nir_instr_type_intrinsic:
      return &clone_intrinsic(state, nir_instr_as_intrinsic(instr))->instr;
   case nir_instr_type_load_const:
      return &clone_load_const(state, nir_instr_as_load_const(instr))->instr;
   case nir_instr_type_ssa_undef:
      return &clone_ssa_undef(state, nir_instr_as_ssa_undef(instr))->instr;
   case nir_instr_type_tex:
      return &clone_tex(state, nir_instr_as_tex(instr))->instr;
   case nir_instr_type_phi:
      unreachable("Cannot clone phis with clone_instr");
   case nir_instr_type_jump:
      return &clone_jump(state, nir_instr_as_jump(instr))->instr;
   case nir_instr_type_call:
      return &clone_call(state, nir_instr_as_call(instr))->instr;
   case nir_instr_type_parallel_copy:
      unreachable("Cannot clone parallel copies");
   default:
      unreachable("bad instr type");
      return NULL;
   }
}

nir_instr *
nir_instr_clone(nir_shader *shader, const nir_instr *orig)
{
   clone_state state = {
      .allow_remap_fallback = true,
      .ns = shader,
   };
   return clone_instr(&state, orig);
}

nir_instr *
nir_instr_clone_deep(nir_shader *shader, const nir_instr *orig,
                     struct hash_table *remap_table)
{
   clone_state state = {
      .allow_remap_fallback = true,
      .ns = shader,
      .remap_table = remap_table,
   };
   return clone_instr(&state, orig);
}

static nir_block *
clone_block(clone_state *state, struct exec_list *cf_list, const nir_block *blk)
{
   /* Don't actually create a new block.  Just use the one from the tail of
    * the list.  NIR guarantees that the tail of the list is a block and that
    * no two blocks are side-by-side in the IR;  It should be empty.
    */
   nir_block *nblk =
      exec_node_data(nir_block, exec_list_get_tail(cf_list), cf_node.node);
   assert(nblk->cf_node.type == nir_cf_node_block);
   assert(exec_list_is_empty(&nblk->instr_list));

   /* We need this for phi sources */
   add_remap(state, nblk, blk);

   nir_foreach_instr(instr, blk) {
      if (instr->type == nir_instr_type_phi) {
         /* Phi instructions are a bit of a special case when cloning because
          * we don't want inserting the instruction to automatically handle
          * use/defs for us.  Instead, we need to wait until all the
          * blocks/instructions are in so that we can set their sources up.
          */
         clone_phi(state, nir_instr_as_phi(instr), nblk);
      } else {
         nir_instr *ninstr = clone_instr(state, instr);
         nir_instr_insert_after_block(nblk, ninstr);
      }
   }

   return nblk;
}

static void
clone_cf_list(clone_state *state, struct exec_list *dst,
              const struct exec_list *list);

static nir_if *
clone_if(clone_state *state, struct exec_list *cf_list, const nir_if *i)
{
   nir_if *ni = nir_if_create(state->ns);
   ni->control = i->control;

   __clone_src(state, ni, &ni->condition, &i->condition);

   nir_cf_node_insert_end(cf_list, &ni->cf_node);

   clone_cf_list(state, &ni->then_list, &i->then_list);
   clone_cf_list(state, &ni->else_list, &i->else_list);

   return ni;
}

static nir_loop *
clone_loop(clone_state *state, struct exec_list *cf_list, const nir_loop *loop)
{
   nir_loop *nloop = nir_loop_create(state->ns);
   nloop->control = loop->control;
   nloop->partially_unrolled = loop->partially_unrolled;

   nir_cf_node_insert_end(cf_list, &nloop->cf_node);

   clone_cf_list(state, &nloop->body, &loop->body);

   return nloop;
}

/* clone list of nir_cf_node: */
static void
clone_cf_list(clone_state *state, struct exec_list *dst,
              const struct exec_list *list)
{
   foreach_list_typed(nir_cf_node, cf, node, list) {
      switch (cf->type) {
      case nir_cf_node_block:
         clone_block(state, dst, nir_cf_node_as_block(cf));
         break;
      case nir_cf_node_if:
         clone_if(state, dst, nir_cf_node_as_if(cf));
         break;
      case nir_cf_node_loop:
         clone_loop(state, dst, nir_cf_node_as_loop(cf));
         break;
      default:
         unreachable("bad cf type");
      }
   }
}

/* After we've cloned almost everything, we have to walk the list of phi
 * sources and fix them up.  Thanks to loops, the block and SSA value for a
 * phi source may not be defined when we first encounter it.  Instead, we
 * add it to the phi_srcs list and we fix it up here.
 */
static void
fixup_phi_srcs(clone_state *state)
{
   list_for_each_entry_safe(nir_phi_src, src, &state->phi_srcs, src.use_link) {
      src->pred = remap_local(state, src->pred);

      /* Remove from this list */
      list_del(&src->src.use_link);

      if (src->src.is_ssa) {
         src->src.ssa = remap_local(state, src->src.ssa);
         list_addtail(&src->src.use_link, &src->src.ssa->uses);
      } else {
         src->src.reg.reg = remap_reg(state, src->src.reg.reg);
         list_addtail(&src->src.use_link, &src->src.reg.reg->uses);
      }
   }
   assert(list_is_empty(&state->phi_srcs));
}

void
nir_cf_list_clone(nir_cf_list *dst, nir_cf_list *src, nir_cf_node *parent,
                  struct hash_table *remap_table)
{
   exec_list_make_empty(&dst->list);
   dst->impl = src->impl;

   if (exec_list_is_empty(&src->list))
      return;

   clone_state state;
   init_clone_state(&state, remap_table, false, true);

   /* We use the same shader */
   state.ns = src->impl->function->shader;

   /* The control-flow code assumes that the list of cf_nodes always starts
    * and ends with a block.  We start by adding an empty block.
    */
   nir_block *nblk = nir_block_create(state.ns);
   nblk->cf_node.parent = parent;
   exec_list_push_tail(&dst->list, &nblk->cf_node.node);

   clone_cf_list(&state, &dst->list, &src->list);

   fixup_phi_srcs(&state);

   if (!remap_table)
      free_clone_state(&state);
}

static nir_function_impl *
clone_function_impl(clone_state *state, const nir_function_impl *fi)
{
   nir_function_impl *nfi = nir_function_impl_create_bare(state->ns);

   if (fi->preamble)
      nfi->preamble = remap_global(state, fi->preamble);

   clone_var_list(state, &nfi->locals, &fi->locals);
   clone_reg_list(state, &nfi->registers, &fi->registers);
   nfi->reg_alloc = fi->reg_alloc;

   assert(list_is_empty(&state->phi_srcs));

   clone_cf_list(state, &nfi->body, &fi->body);

   fixup_phi_srcs(state);

   /* All metadata is invalidated in the cloning process */
   nfi->valid_metadata = 0;

   return nfi;
}

nir_function_impl *
nir_function_impl_clone(nir_shader *shader, const nir_function_impl *fi)
{
   clone_state state;
   init_clone_state(&state, NULL, false, false);

   state.ns = shader;

   nir_function_impl *nfi = clone_function_impl(&state, fi);

   free_clone_state(&state);

   return nfi;
}

static nir_function *
clone_function(clone_state *state, const nir_function *fxn, nir_shader *ns)
{
   assert(ns == state->ns);
   nir_function *nfxn = nir_function_create(ns, fxn->name);

   /* Needed for call instructions */
   add_remap(state, nfxn, fxn);

   nfxn->num_params = fxn->num_params;
   if (fxn->num_params) {
           nfxn->params = ralloc_array(state->ns, nir_parameter, fxn->num_params);
           memcpy(nfxn->params, fxn->params, sizeof(nir_parameter) * fxn->num_params);
   }
   nfxn->is_entrypoint = fxn->is_entrypoint;
   nfxn->is_preamble = fxn->is_preamble;

   /* At first glance, it looks like we should clone the function_impl here.
    * However, call instructions need to be able to reference at least the
    * function and those will get processed as we clone the function_impls.
    * We stop here and do function_impls as a second pass.
    */

   return nfxn;
}

nir_shader *
nir_shader_clone(void *mem_ctx, const nir_shader *s)
{
   clone_state state;
   init_clone_state(&state, NULL, true, false);

   nir_shader *ns = nir_shader_create(mem_ctx, s->info.stage, s->options, NULL);
   state.ns = ns;

   clone_var_list(&state, &ns->variables, &s->variables);

   /* Go through and clone functions */
   foreach_list_typed(nir_function, fxn, node, &s->functions)
      clone_function(&state, fxn, ns);

   /* Only after all functions are cloned can we clone the actual function
    * implementations.  This is because nir_call_instrs and preambles need to
    * reference the functions of other functions and we don't know what order
    * the functions will have in the list.
    */
   nir_foreach_function(fxn, s) {
      nir_function *nfxn = remap_global(&state, fxn);
      nfxn->impl = clone_function_impl(&state, fxn->impl);
      nfxn->impl->function = nfxn;
   }

   ns->info = s->info;
   ns->info.name = ralloc_strdup(ns, ns->info.name);
   if (ns->info.label)
      ns->info.label = ralloc_strdup(ns, ns->info.label);

   ns->num_inputs = s->num_inputs;
   ns->num_uniforms = s->num_uniforms;
   ns->num_outputs = s->num_outputs;
   ns->scratch_size = s->scratch_size;

   ns->constant_data_size = s->constant_data_size;
   if (s->constant_data_size > 0) {
      ns->constant_data = ralloc_size(ns, s->constant_data_size);
      memcpy(ns->constant_data, s->constant_data, s->constant_data_size);
   }

   if (s->xfb_info) {
      size_t size = nir_xfb_info_size(s->xfb_info->output_count);
      ns->xfb_info = ralloc_size(ns, size);
      memcpy(ns->xfb_info, s->xfb_info, size);
   }

   free_clone_state(&state);

   return ns;
}

/** Overwrites dst and replaces its contents with src
 *
 * Everything ralloc parented to dst and src itself (but not its children)
 * will be freed.
 *
 * This should only be used by test code which needs to swap out shaders with
 * a cloned or deserialized version.
 */
void
nir_shader_replace(nir_shader *dst, nir_shader *src)
{
   /* Delete all of dest's ralloc children */
   void *dead_ctx = ralloc_context(NULL);
   ralloc_adopt(dead_ctx, dst);
   ralloc_free(dead_ctx);

   /* Re-parent all of src's ralloc children to dst */
   ralloc_adopt(dst, src);

   memcpy(dst, src, sizeof(*dst));

   /* We have to move all the linked lists over separately because we need the
    * pointers in the list elements to point to the lists in dst and not src.
    */
   exec_list_move_nodes_to(&src->variables, &dst->variables);

   /* Now move the functions over.  This takes a tiny bit more work */
   exec_list_move_nodes_to(&src->functions, &dst->functions);
   nir_foreach_function(function, dst)
      function->shader = dst;

   ralloc_free(src);
}
