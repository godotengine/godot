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

#include "nir.h"
#include "nir_builder.h"

static void
build_constant_load(nir_builder *b, nir_deref_instr *deref, nir_constant *c)
{
   if (glsl_type_is_vector_or_scalar(deref->type)) {
      nir_load_const_instr *load =
         nir_load_const_instr_create(b->shader,
                                     glsl_get_vector_elements(deref->type),
                                     glsl_get_bit_size(deref->type));
      memcpy(load->value, c->values, sizeof(*load->value) * load->def.num_components);
      nir_builder_instr_insert(b, &load->instr);
      nir_store_deref(b, deref, &load->def, ~0);
   } else if (glsl_type_is_struct_or_ifc(deref->type)) {
      unsigned len = glsl_get_length(deref->type);
      for (unsigned i = 0; i < len; i++) {
         build_constant_load(b, nir_build_deref_struct(b, deref, i),
                             c->elements[i]);
      }
   } else {
      assert(glsl_type_is_array(deref->type) ||
             glsl_type_is_matrix(deref->type));
      unsigned len = glsl_get_length(deref->type);
      for (unsigned i = 0; i < len; i++) {
         build_constant_load(b,
                             nir_build_deref_array_imm(b, deref, i),
                             c->elements[i]);
      }
   }
}

static bool
lower_const_initializer(struct nir_builder *b, struct exec_list *var_list,
                        nir_variable_mode modes)
{
   bool progress = false;

   b->cursor = nir_before_cf_list(&b->impl->body);

   nir_foreach_variable_in_list(var, var_list) {
      if (!(var->data.mode & modes))
         continue;

      if (var->constant_initializer) {
         build_constant_load(b, nir_build_deref_var(b, var),
                             var->constant_initializer);

         progress = true;
         var->constant_initializer = NULL;
      } else if (var->pointer_initializer) {
         nir_deref_instr *src_deref = nir_build_deref_var(b, var->pointer_initializer);
         nir_deref_instr *dst_deref = nir_build_deref_var(b, var);

         /* Note that this stores a pointer to src into dst */
         nir_store_deref(b, dst_deref, &src_deref->dest.ssa, ~0);

         progress = true;
         var->pointer_initializer = NULL;
      }

   }

   return progress;
}

bool
nir_lower_variable_initializers(nir_shader *shader, nir_variable_mode modes)
{
   bool progress = false;

   /* Only some variables have initializers that we want to lower.  Others
    * such as uniforms have initializers which are useful later during linking
    * so we want to skip over those.  Restrict to only variable types where
    * initializers make sense so that callers can use nir_var_all.
    */
   modes &= nir_var_shader_out |
            nir_var_shader_temp |
            nir_var_function_temp |
            nir_var_system_value;

   nir_foreach_function(function, shader) {
      if (!function->impl)
	 continue;

      bool impl_progress = false;

      nir_builder builder;
      nir_builder_init(&builder, function->impl);

      if ((modes & ~nir_var_function_temp) && function->is_entrypoint) {
         impl_progress |= lower_const_initializer(&builder,
                                                  &shader->variables,
                                                  modes);
      }

      if (modes & nir_var_function_temp) {
         impl_progress |= lower_const_initializer(&builder,
                                                  &function->impl->locals,
                                                  nir_var_function_temp);
      }

      if (impl_progress) {
         progress = true;
         nir_metadata_preserve(function->impl, nir_metadata_block_index |
                                               nir_metadata_dominance |
                                               nir_metadata_live_ssa_defs);
      } else {
         nir_metadata_preserve(function->impl, nir_metadata_all);
      }
   }

   return progress;
}

/* Zero initialize shared_size bytes of shared memory by splitting work writes
 * of chunk_size bytes among the invocations.
 *
 * Used for implementing VK_KHR_zero_initialize_workgroup_memory.
 */
bool
nir_zero_initialize_shared_memory(nir_shader *shader,
                                  const unsigned shared_size,
                                  const unsigned chunk_size)
{
   assert(shared_size > 0);
   assert(chunk_size > 0);
   assert(chunk_size % 4 == 0);

   nir_builder b;
   nir_builder_init(&b, nir_shader_get_entrypoint(shader));
   b.cursor = nir_before_cf_list(&b.impl->body);

   assert(!shader->info.workgroup_size_variable);
   const unsigned local_count = shader->info.workgroup_size[0] *
                                shader->info.workgroup_size[1] *
                                shader->info.workgroup_size[2];

   /* The initialization logic is simplified if we can always split the memory
    * in full chunk_size units.
    */
   assert(shared_size % chunk_size == 0);

   const unsigned chunk_comps = chunk_size / 4;

   nir_variable *it = nir_local_variable_create(b.impl, glsl_uint_type(),
                                                "zero_init_iterator");
   nir_ssa_def *local_index = nir_load_local_invocation_index(&b);
   nir_ssa_def *first_offset = nir_imul_imm(&b, local_index, chunk_size);
   nir_store_var(&b, it, first_offset, 0x1);

   nir_loop *loop = nir_push_loop(&b);
   {
      nir_ssa_def *offset = nir_load_var(&b, it);

      nir_push_if(&b, nir_uge(&b, offset, nir_imm_int(&b, shared_size)));
      {
         nir_jump(&b, nir_jump_break);
      }
      nir_pop_if(&b, NULL);

      nir_store_shared(&b, nir_imm_zero(&b, chunk_comps, 32), offset,
                       .align_mul=chunk_size,
                       .write_mask=((1 << chunk_comps) - 1));

      nir_ssa_def *new_offset = nir_iadd_imm(&b, offset, chunk_size * local_count);
      nir_store_var(&b, it, new_offset, 0x1);
   }
   nir_pop_loop(&b, loop);

   if (shader->options->use_scoped_barrier) {
      nir_scoped_barrier(&b, NIR_SCOPE_WORKGROUP, NIR_SCOPE_WORKGROUP,
                         NIR_MEMORY_ACQ_REL, nir_var_mem_shared);
   } else {
      nir_memory_barrier_shared(&b);
   }

   nir_metadata_preserve(nir_shader_get_entrypoint(shader), nir_metadata_none);

   return true;
}
