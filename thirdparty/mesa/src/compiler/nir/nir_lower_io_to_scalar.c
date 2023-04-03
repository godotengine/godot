/*
 * Copyright © 2016 Broadcom
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

/** @file nir_lower_io_to_scalar.c
 *
 * Replaces nir_load_input/nir_store_output operations with num_components !=
 * 1 with individual per-channel operations.
 */

static void
set_io_semantics(nir_intrinsic_instr *scalar_intr,
                 nir_intrinsic_instr *vec_intr, unsigned component)
{
   nir_io_semantics sem = nir_intrinsic_io_semantics(vec_intr);
   sem.gs_streams = (sem.gs_streams >> (component * 2)) & 0x3;
   nir_intrinsic_set_io_semantics(scalar_intr, sem);
}

static void
lower_load_input_to_scalar(nir_builder *b, nir_intrinsic_instr *intr)
{
   b->cursor = nir_before_instr(&intr->instr);

   assert(intr->dest.is_ssa);

   nir_ssa_def *loads[NIR_MAX_VEC_COMPONENTS];

   for (unsigned i = 0; i < intr->num_components; i++) {
      nir_intrinsic_instr *chan_intr =
         nir_intrinsic_instr_create(b->shader, intr->intrinsic);
      nir_ssa_dest_init(&chan_intr->instr, &chan_intr->dest,
                        1, intr->dest.ssa.bit_size, NULL);
      chan_intr->num_components = 1;

      nir_intrinsic_set_base(chan_intr, nir_intrinsic_base(intr));
      nir_intrinsic_set_component(chan_intr, nir_intrinsic_component(intr) + i);
      nir_intrinsic_set_dest_type(chan_intr, nir_intrinsic_dest_type(intr));
      set_io_semantics(chan_intr, intr, i);
      /* offset and vertex (if needed) */
      for (unsigned j = 0; j < nir_intrinsic_infos[intr->intrinsic].num_srcs; ++j)
         nir_src_copy(&chan_intr->src[j], &intr->src[j], &chan_intr->instr);

      nir_builder_instr_insert(b, &chan_intr->instr);

      loads[i] = &chan_intr->dest.ssa;
   }

   nir_ssa_def_rewrite_uses(&intr->dest.ssa,
                            nir_vec(b, loads, intr->num_components));
   nir_instr_remove(&intr->instr);
}

static void
lower_load_to_scalar(nir_builder *b, nir_intrinsic_instr *intr)
{
   b->cursor = nir_before_instr(&intr->instr);

   assert(intr->dest.is_ssa);

   nir_ssa_def *loads[NIR_MAX_VEC_COMPONENTS];
   nir_ssa_def *base_offset = nir_get_io_offset_src(intr)->ssa;

   for (unsigned i = 0; i < intr->num_components; i++) {
      nir_intrinsic_instr *chan_intr =
         nir_intrinsic_instr_create(b->shader, intr->intrinsic);
      nir_ssa_dest_init(&chan_intr->instr, &chan_intr->dest,
                        1, intr->dest.ssa.bit_size, NULL);
      chan_intr->num_components = 1;

      nir_intrinsic_set_align_offset(chan_intr,
                                     (nir_intrinsic_align_offset(intr) +
                                      i * (intr->dest.ssa.bit_size / 8)) % nir_intrinsic_align_mul(intr));
      nir_intrinsic_set_align_mul(chan_intr, nir_intrinsic_align_mul(intr));
      if (nir_intrinsic_has_access(intr))
         nir_intrinsic_set_access(chan_intr, nir_intrinsic_access(intr));
      if (nir_intrinsic_has_range(intr))
         nir_intrinsic_set_range(chan_intr, nir_intrinsic_range(intr));
      if (nir_intrinsic_has_range_base(intr))
         nir_intrinsic_set_range_base(chan_intr, nir_intrinsic_range_base(intr));
      if (nir_intrinsic_has_base(intr))
         nir_intrinsic_set_base(chan_intr, nir_intrinsic_base(intr));
      for (unsigned j = 0; j < nir_intrinsic_infos[intr->intrinsic].num_srcs - 1; j++)
         nir_src_copy(&chan_intr->src[j], &intr->src[j], &chan_intr->instr);

      /* increment offset per component */
      nir_ssa_def *offset = nir_iadd_imm(b, base_offset, i * (intr->dest.ssa.bit_size / 8));
      *nir_get_io_offset_src(chan_intr) = nir_src_for_ssa(offset);

      nir_builder_instr_insert(b, &chan_intr->instr);

      loads[i] = &chan_intr->dest.ssa;
   }

   nir_ssa_def_rewrite_uses(&intr->dest.ssa,
                            nir_vec(b, loads, intr->num_components));
   nir_instr_remove(&intr->instr);
}

static void
lower_store_output_to_scalar(nir_builder *b, nir_intrinsic_instr *intr)
{
   b->cursor = nir_before_instr(&intr->instr);

   nir_ssa_def *value = nir_ssa_for_src(b, intr->src[0], intr->num_components);

   for (unsigned i = 0; i < intr->num_components; i++) {
      if (!(nir_intrinsic_write_mask(intr) & (1 << i)))
         continue;

      nir_intrinsic_instr *chan_intr =
         nir_intrinsic_instr_create(b->shader, intr->intrinsic);
      chan_intr->num_components = 1;

      nir_intrinsic_set_base(chan_intr, nir_intrinsic_base(intr));
      nir_intrinsic_set_write_mask(chan_intr, 0x1);
      nir_intrinsic_set_component(chan_intr, nir_intrinsic_component(intr) + i);
      nir_intrinsic_set_src_type(chan_intr, nir_intrinsic_src_type(intr));
      set_io_semantics(chan_intr, intr, i);

      if (nir_intrinsic_has_io_xfb(intr)) {
         /* Scalarize transform feedback info. */
         unsigned component = nir_intrinsic_component(chan_intr);

         for (unsigned c = 0; c <= component; c++) {
            nir_io_xfb xfb = c < 2 ? nir_intrinsic_io_xfb(intr) :
                                     nir_intrinsic_io_xfb2(intr);

            if (component < c + xfb.out[c % 2].num_components) {
               nir_io_xfb scalar_xfb;

               memset(&scalar_xfb, 0, sizeof(scalar_xfb));
               scalar_xfb.out[component % 2].num_components = 1;
               scalar_xfb.out[component % 2].buffer = xfb.out[c % 2].buffer;
               scalar_xfb.out[component % 2].offset = xfb.out[c % 2].offset +
                                                      component - c;
               if (component < 2)
                  nir_intrinsic_set_io_xfb(chan_intr, scalar_xfb);
               else
                  nir_intrinsic_set_io_xfb2(chan_intr, scalar_xfb);
               break;
            }
         }
      }

      /* value */
      chan_intr->src[0] = nir_src_for_ssa(nir_channel(b, value, i));
      /* offset and vertex (if needed) */
      for (unsigned j = 1; j < nir_intrinsic_infos[intr->intrinsic].num_srcs; ++j)
         nir_src_copy(&chan_intr->src[j], &intr->src[j], &chan_intr->instr);

      nir_builder_instr_insert(b, &chan_intr->instr);
   }

   nir_instr_remove(&intr->instr);
}

static void
lower_store_to_scalar(nir_builder *b, nir_intrinsic_instr *intr)
{
   b->cursor = nir_before_instr(&intr->instr);

   nir_ssa_def *value = nir_ssa_for_src(b, intr->src[0], intr->num_components);
   nir_ssa_def *base_offset = nir_get_io_offset_src(intr)->ssa;

   /* iterate wrmask instead of num_components to handle split components */
   u_foreach_bit(i, nir_intrinsic_write_mask(intr)) {
      nir_intrinsic_instr *chan_intr =
         nir_intrinsic_instr_create(b->shader, intr->intrinsic);
      chan_intr->num_components = 1;

      nir_intrinsic_set_write_mask(chan_intr, 0x1);
      nir_intrinsic_set_align_offset(chan_intr,
                                     (nir_intrinsic_align_offset(intr) +
                                      i * (value->bit_size / 8)) % nir_intrinsic_align_mul(intr));
      nir_intrinsic_set_align_mul(chan_intr, nir_intrinsic_align_mul(intr));
      if (nir_intrinsic_has_access(intr))
         nir_intrinsic_set_access(chan_intr, nir_intrinsic_access(intr));
      if (nir_intrinsic_has_base(intr))
         nir_intrinsic_set_base(chan_intr, nir_intrinsic_base(intr));

      /* value */
      chan_intr->src[0] = nir_src_for_ssa(nir_channel(b, value, i));
      for (unsigned j = 1; j < nir_intrinsic_infos[intr->intrinsic].num_srcs - 1; j++)
         nir_src_copy(&chan_intr->src[j], &intr->src[j], &chan_intr->instr);

      /* increment offset per component */
      nir_ssa_def *offset = nir_iadd_imm(b, base_offset, i * (value->bit_size / 8));
      *nir_get_io_offset_src(chan_intr) = nir_src_for_ssa(offset);

      nir_builder_instr_insert(b, &chan_intr->instr);
   }

   nir_instr_remove(&intr->instr);
}

static bool
nir_lower_io_to_scalar_instr(nir_builder *b, nir_instr *instr, void *data)
{
   nir_variable_mode mask = *(nir_variable_mode *)data;

   if (instr->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);

   if (intr->num_components == 1)
      return false;

   if ((intr->intrinsic == nir_intrinsic_load_input ||
        intr->intrinsic == nir_intrinsic_load_per_vertex_input) &&
       (mask & nir_var_shader_in)) {
      lower_load_input_to_scalar(b, intr);
      return true;
   }

   if (intr->intrinsic == nir_intrinsic_load_per_vertex_output &&
      (mask & nir_var_shader_out)) {
      lower_load_input_to_scalar(b, intr);
      return true;
   }

   if ((intr->intrinsic == nir_intrinsic_load_ubo && (mask & nir_var_mem_ubo)) ||
       (intr->intrinsic == nir_intrinsic_load_ssbo && (mask & nir_var_mem_ssbo)) ||
       (intr->intrinsic == nir_intrinsic_load_global && (mask & nir_var_mem_global)) ||
       (intr->intrinsic == nir_intrinsic_load_shared && (mask & nir_var_mem_shared))) {
      lower_load_to_scalar(b, intr);
      return true;
   }

   if ((intr->intrinsic == nir_intrinsic_store_output ||
        intr->intrinsic == nir_intrinsic_store_per_vertex_output) &&
       mask & nir_var_shader_out) {
      lower_store_output_to_scalar(b, intr);
      return true;
   }

   if ((intr->intrinsic == nir_intrinsic_store_ssbo && (mask & nir_var_mem_ssbo)) ||
       (intr->intrinsic == nir_intrinsic_store_global && (mask & nir_var_mem_global)) ||
       (intr->intrinsic == nir_intrinsic_store_shared && (mask & nir_var_mem_shared))) {
      lower_store_to_scalar(b, intr);
      return true;
   }

   return false;
}

void
nir_lower_io_to_scalar(nir_shader *shader, nir_variable_mode mask)
{
   nir_shader_instructions_pass(shader,
                                nir_lower_io_to_scalar_instr,
                                nir_metadata_block_index |
                                nir_metadata_dominance,
                                &mask);
}

static nir_variable **
get_channel_variables(struct hash_table *ht, nir_variable *var)
{
   nir_variable **chan_vars;
   struct hash_entry *entry = _mesa_hash_table_search(ht, var);
   if (!entry) {
      chan_vars = (nir_variable **) calloc(4, sizeof(nir_variable *));
      _mesa_hash_table_insert(ht, var, chan_vars);
   } else {
      chan_vars = (nir_variable **) entry->data;
   }

   return chan_vars;
}

/*
 * Note that the src deref that we are cloning is the head of the
 * chain of deref instructions from the original intrinsic, but
 * the dst we are cloning to is the tail (because chains of deref
 * instructions are created back to front)
 */

static nir_deref_instr *
clone_deref_array(nir_builder *b, nir_deref_instr *dst_tail,
                  const nir_deref_instr *src_head)
{
   const nir_deref_instr *parent = nir_deref_instr_parent(src_head);

   if (!parent)
      return dst_tail;

   assert(src_head->deref_type == nir_deref_type_array);

   dst_tail = clone_deref_array(b, dst_tail, parent);

   return nir_build_deref_array(b, dst_tail,
                                nir_ssa_for_src(b, src_head->arr.index, 1));
}

static void
lower_load_to_scalar_early(nir_builder *b, nir_intrinsic_instr *intr,
                           nir_variable *var, struct hash_table *split_inputs,
                           struct hash_table *split_outputs)
{
   b->cursor = nir_before_instr(&intr->instr);

   assert(intr->dest.is_ssa);

   nir_ssa_def *loads[NIR_MAX_VEC_COMPONENTS];

   nir_variable **chan_vars;
   if (var->data.mode == nir_var_shader_in) {
      chan_vars = get_channel_variables(split_inputs, var);
   } else {
      chan_vars = get_channel_variables(split_outputs, var);
   }

   for (unsigned i = 0; i < intr->num_components; i++) {
      nir_variable *chan_var = chan_vars[var->data.location_frac + i];
      if (!chan_vars[var->data.location_frac + i]) {
         chan_var = nir_variable_clone(var, b->shader);
         chan_var->data.location_frac =  var->data.location_frac + i;
         chan_var->type = glsl_channel_type(chan_var->type);

         chan_vars[var->data.location_frac + i] = chan_var;

         nir_shader_add_variable(b->shader, chan_var);
      }

      nir_intrinsic_instr *chan_intr =
         nir_intrinsic_instr_create(b->shader, intr->intrinsic);
      nir_ssa_dest_init(&chan_intr->instr, &chan_intr->dest,
                        1, intr->dest.ssa.bit_size, NULL);
      chan_intr->num_components = 1;

      nir_deref_instr *deref = nir_build_deref_var(b, chan_var);

      deref = clone_deref_array(b, deref, nir_src_as_deref(intr->src[0]));

      chan_intr->src[0] = nir_src_for_ssa(&deref->dest.ssa);

      if (intr->intrinsic == nir_intrinsic_interp_deref_at_offset ||
          intr->intrinsic == nir_intrinsic_interp_deref_at_sample ||
          intr->intrinsic == nir_intrinsic_interp_deref_at_vertex)
         nir_src_copy(&chan_intr->src[1], &intr->src[1], &chan_intr->instr);

      nir_builder_instr_insert(b, &chan_intr->instr);

      loads[i] = &chan_intr->dest.ssa;
   }

   nir_ssa_def_rewrite_uses(&intr->dest.ssa,
                            nir_vec(b, loads, intr->num_components));

   /* Remove the old load intrinsic */
   nir_instr_remove(&intr->instr);
}

static void
lower_store_output_to_scalar_early(nir_builder *b, nir_intrinsic_instr *intr,
                                   nir_variable *var,
                                   struct hash_table *split_outputs)
{
   b->cursor = nir_before_instr(&intr->instr);

   nir_ssa_def *value = nir_ssa_for_src(b, intr->src[1], intr->num_components);

   nir_variable **chan_vars = get_channel_variables(split_outputs, var);
   for (unsigned i = 0; i < intr->num_components; i++) {
      if (!(nir_intrinsic_write_mask(intr) & (1 << i)))
         continue;

      nir_variable *chan_var = chan_vars[var->data.location_frac + i];
      if (!chan_vars[var->data.location_frac + i]) {
         chan_var = nir_variable_clone(var, b->shader);
         chan_var->data.location_frac =  var->data.location_frac + i;
         chan_var->type = glsl_channel_type(chan_var->type);

         chan_vars[var->data.location_frac + i] = chan_var;

         nir_shader_add_variable(b->shader, chan_var);
      }

      nir_intrinsic_instr *chan_intr =
         nir_intrinsic_instr_create(b->shader, intr->intrinsic);
      chan_intr->num_components = 1;

      nir_intrinsic_set_write_mask(chan_intr, 0x1);

      nir_deref_instr *deref = nir_build_deref_var(b, chan_var);

      deref = clone_deref_array(b, deref, nir_src_as_deref(intr->src[0]));

      chan_intr->src[0] = nir_src_for_ssa(&deref->dest.ssa);
      chan_intr->src[1] = nir_src_for_ssa(nir_channel(b, value, i));

      nir_builder_instr_insert(b, &chan_intr->instr);
   }

   /* Remove the old store intrinsic */
   nir_instr_remove(&intr->instr);
}

struct io_to_scalar_early_state {
   struct hash_table *split_inputs, *split_outputs;
   nir_variable_mode mask;
};

static bool
nir_lower_io_to_scalar_early_instr(nir_builder *b, nir_instr *instr, void *data)
{
   struct io_to_scalar_early_state *state = data;

   if (instr->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);

   if (intr->num_components == 1)
      return false;

   if (intr->intrinsic != nir_intrinsic_load_deref &&
       intr->intrinsic != nir_intrinsic_store_deref &&
       intr->intrinsic != nir_intrinsic_interp_deref_at_centroid &&
       intr->intrinsic != nir_intrinsic_interp_deref_at_sample &&
       intr->intrinsic != nir_intrinsic_interp_deref_at_offset &&
       intr->intrinsic != nir_intrinsic_interp_deref_at_vertex)
      return false;

   nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
   if (!nir_deref_mode_is_one_of(deref, state->mask))
      return false;

   nir_variable *var = nir_deref_instr_get_variable(deref);
   nir_variable_mode mode = var->data.mode;

   /* TODO: add patch support */
   if (var->data.patch)
      return false;

   /* TODO: add doubles support */
   if (glsl_type_is_64bit(glsl_without_array(var->type)))
      return false;

   if (!(b->shader->info.stage == MESA_SHADER_VERTEX &&
         mode == nir_var_shader_in) &&
       var->data.location < VARYING_SLOT_VAR0 &&
       var->data.location >= 0)
      return false;

   /* Don't bother splitting if we can't opt away any unused
    * components.
    */
   if (var->data.always_active_io)
      return false;

   if (var->data.must_be_shader_input)
      return false;

   /* Skip types we cannot split */
   if (glsl_type_is_matrix(glsl_without_array(var->type)) ||
       glsl_type_is_struct_or_ifc(glsl_without_array(var->type)))
      return false;

   switch (intr->intrinsic) {
   case nir_intrinsic_interp_deref_at_centroid:
   case nir_intrinsic_interp_deref_at_sample:
   case nir_intrinsic_interp_deref_at_offset:
   case nir_intrinsic_interp_deref_at_vertex:
   case nir_intrinsic_load_deref:
      if ((state->mask & nir_var_shader_in && mode == nir_var_shader_in) ||
          (state->mask & nir_var_shader_out && mode == nir_var_shader_out)) {
         lower_load_to_scalar_early(b, intr, var, state->split_inputs,
                                    state->split_outputs);
         return true;
      }
      break;
   case nir_intrinsic_store_deref:
      if (state->mask & nir_var_shader_out &&
          mode == nir_var_shader_out) {
         lower_store_output_to_scalar_early(b, intr, var, state->split_outputs);
         return true;
      }
      break;
   default:
      break;
   }

   return false;
}

/*
 * This function is intended to be called earlier than nir_lower_io_to_scalar()
 * i.e. before nir_lower_io() is called.
 */
bool
nir_lower_io_to_scalar_early(nir_shader *shader, nir_variable_mode mask)
{
   struct io_to_scalar_early_state state = {
      .split_inputs = _mesa_pointer_hash_table_create(NULL),
      .split_outputs = _mesa_pointer_hash_table_create(NULL),
      .mask = mask
   };

   bool progress = nir_shader_instructions_pass(shader,
                                                nir_lower_io_to_scalar_early_instr,
                                                nir_metadata_block_index |
                                                nir_metadata_dominance,
                                                &state);

   /* Remove old input from the shaders inputs list */
   hash_table_foreach(state.split_inputs, entry) {
      nir_variable *var = (nir_variable *) entry->key;
      exec_node_remove(&var->node);

      free(entry->data);
   }

   /* Remove old output from the shaders outputs list */
   hash_table_foreach(state.split_outputs, entry) {
      nir_variable *var = (nir_variable *) entry->key;
      exec_node_remove(&var->node);

      free(entry->data);
   }

   _mesa_hash_table_destroy(state.split_inputs, NULL);
   _mesa_hash_table_destroy(state.split_outputs, NULL);

   nir_remove_dead_derefs(shader);

   return progress;
}
