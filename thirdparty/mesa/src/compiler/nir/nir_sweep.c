/*
 * Copyright © 2015 Intel Corporation
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

/**
 * \file nir_sweep.c
 *
 * The nir_sweep() pass performs a mark and sweep pass over a nir_shader's associated
 * memory - anything still connected to the program will be kept, and any dead memory
 * we dropped on the floor will be freed.
 *
 * The expectation is that drivers should call this when finished compiling the shader
 * (after any optimization, lowering, and so on).  However, it's also fine to call it
 * earlier, and even many times, trading CPU cycles for memory savings.
 */

#define steal_list(mem_ctx, type, list) \
   foreach_list_typed(type, obj, node, list) { ralloc_steal(mem_ctx, obj); }

static void sweep_cf_node(nir_shader *nir, nir_cf_node *cf_node);

static bool
sweep_src_indirect(nir_src *src, void *nir)
{
   if (!src->is_ssa && src->reg.indirect)
      gc_mark_live(((nir_shader*)nir)->gctx, src->reg.indirect);

   return true;
}

static bool
sweep_dest_indirect(nir_dest *dest, void *nir)
{
   if (!dest->is_ssa && dest->reg.indirect)
      gc_mark_live(((nir_shader*)nir)->gctx, dest->reg.indirect);

   return true;
}

static void
sweep_block(nir_shader *nir, nir_block *block)
{
   ralloc_steal(nir, block);

   /* sweep_impl will mark all metadata invalid.  We can safely release all of
    * this here.
    */
   ralloc_free(block->live_in);
   block->live_in = NULL;

   ralloc_free(block->live_out);
   block->live_out = NULL;

   nir_foreach_instr(instr, block) {
      gc_mark_live(nir->gctx, instr);

      switch (instr->type) {
      case nir_instr_type_tex:
         gc_mark_live(nir->gctx, nir_instr_as_tex(instr)->src);
         break;
      case nir_instr_type_phi:
         nir_foreach_phi_src(src, nir_instr_as_phi(instr))
            gc_mark_live(nir->gctx, src);
         break;
      default:
         break;
      }

      nir_foreach_src(instr, sweep_src_indirect, nir);
      nir_foreach_dest(instr, sweep_dest_indirect, nir);
   }
}

static void
sweep_if(nir_shader *nir, nir_if *iff)
{
   ralloc_steal(nir, iff);

   foreach_list_typed(nir_cf_node, cf_node, node, &iff->then_list) {
      sweep_cf_node(nir, cf_node);
   }

   foreach_list_typed(nir_cf_node, cf_node, node, &iff->else_list) {
      sweep_cf_node(nir, cf_node);
   }
}

static void
sweep_loop(nir_shader *nir, nir_loop *loop)
{
   ralloc_steal(nir, loop);

   foreach_list_typed(nir_cf_node, cf_node, node, &loop->body) {
      sweep_cf_node(nir, cf_node);
   }
}

static void
sweep_cf_node(nir_shader *nir, nir_cf_node *cf_node)
{
   switch (cf_node->type) {
   case nir_cf_node_block:
      sweep_block(nir, nir_cf_node_as_block(cf_node));
      break;
   case nir_cf_node_if:
      sweep_if(nir, nir_cf_node_as_if(cf_node));
      break;
   case nir_cf_node_loop:
      sweep_loop(nir, nir_cf_node_as_loop(cf_node));
      break;
   default:
      unreachable("Invalid CF node type");
   }
}

static void
sweep_impl(nir_shader *nir, nir_function_impl *impl)
{
   ralloc_steal(nir, impl);

   steal_list(nir, nir_variable, &impl->locals);
   steal_list(nir, nir_register, &impl->registers);

   foreach_list_typed(nir_cf_node, cf_node, node, &impl->body) {
      sweep_cf_node(nir, cf_node);
   }

   sweep_block(nir, impl->end_block);

   /* Wipe out all the metadata, if any. */
   nir_metadata_preserve(impl, nir_metadata_none);
}

static void
sweep_function(nir_shader *nir, nir_function *f)
{
   ralloc_steal(nir, f);
   ralloc_steal(nir, f->params);

   if (f->impl)
      sweep_impl(nir, f->impl);
}

void
nir_sweep(nir_shader *nir)
{
   void *rubbish = ralloc_context(NULL);

   struct list_head instr_gc_list;
   list_inithead(&instr_gc_list);

   /* First, move ownership of all the memory to a temporary context; assume dead. */
   ralloc_adopt(rubbish, nir);

   /* Start sweeping */
   gc_sweep_start(nir->gctx);

   ralloc_steal(nir, nir->gctx);
   ralloc_steal(nir, (char *)nir->info.name);
   if (nir->info.label)
      ralloc_steal(nir, (char *)nir->info.label);

   /* Variables and registers are not dead.  Steal them back. */
   steal_list(nir, nir_variable, &nir->variables);

   /* Recurse into functions, stealing their contents back. */
   foreach_list_typed(nir_function, func, node, &nir->functions) {
      sweep_function(nir, func);
   }

   ralloc_steal(nir, nir->constant_data);
   ralloc_steal(nir, nir->xfb_info);
   ralloc_steal(nir, nir->printf_info);
   for (int i = 0; i < nir->printf_info_count; i++) {
      ralloc_steal(nir, nir->printf_info[i].arg_sizes);
      ralloc_steal(nir, nir->printf_info[i].strings);
   }

   /* Free everything we didn't steal back. */
   gc_sweep_end(nir->gctx);
   ralloc_free(rubbish);
}
