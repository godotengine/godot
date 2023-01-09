/*
 * Copyright © 2022 Collabora Ltd
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
 *    Gert Wollny <gert.wollny@collabora.com>
 */

#include "nir.h"
#include "nir_builder.h"

#include "nir_deref.h"
#include "util/hash_table.h"

/* This pass splits stores to and loads from 64 bit vec3
 * and vec4 local variables to use at most vec2, and it also
 * splits phi nodes accordingly.
 *
 * Arrays of vec3 and vec4 are handled directly, arrays of arrays
 * are lowered to arrays on the fly.
 */

static bool
nir_split_64bit_vec3_and_vec4_filter(const nir_instr *instr,
                                     const void *data)
{
   switch (instr->type) {
   case  nir_instr_type_intrinsic: {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);

      switch (intr->intrinsic) {
      case nir_intrinsic_load_deref: {
         if (nir_dest_bit_size(intr->dest) != 64)
            return false;
         nir_variable *var = nir_intrinsic_get_var(intr, 0);
         if (var->data.mode != nir_var_function_temp)
            return false;
         return nir_dest_num_components(intr->dest) >= 3;
      }
      case nir_intrinsic_store_deref: {
         if (nir_src_bit_size(intr->src[1]) != 64)
            return false;
         nir_variable *var = nir_intrinsic_get_var(intr, 0);
         if (var->data.mode != nir_var_function_temp)
            return false;
         return nir_src_num_components(intr->src[1]) >= 3;
      default:
            return false;
         }
      }
   }
   case nir_instr_type_phi: {
         nir_phi_instr *phi = nir_instr_as_phi(instr);
         if (nir_dest_bit_size(phi->dest) != 64)
            return false;
         return nir_dest_num_components(phi->dest) >= 3;
      }

   default:
      return false;
   }
}

typedef struct {
   nir_variable *xy;
   nir_variable *zw;
} variable_pair;


static nir_ssa_def *
merge_to_vec3_or_vec4(nir_builder *b, nir_ssa_def *load1,
                      nir_ssa_def *load2)
{
   assert(load2->num_components > 0 && load2->num_components < 3);

   if (load2->num_components == 1)
      return nir_vec3(b, nir_channel(b, load1, 0),
                      nir_channel(b, load1, 1),
                      nir_channel(b, load2, 0));
   else
      return nir_vec4(b, nir_channel(b, load1, 0),
                      nir_channel(b, load1, 1),
                      nir_channel(b, load2, 0),
                      nir_channel(b, load2, 1));
}

static nir_ssa_def *
get_linear_array_offset(nir_builder *b, nir_deref_instr *deref)
{
   nir_deref_path path;
   nir_deref_path_init(&path, deref, NULL);

   nir_ssa_def *offset = nir_imm_intN_t(b, 0, deref->dest.ssa.bit_size);
   for (nir_deref_instr **p = &path.path[1]; *p; p++) {
      switch ((*p)->deref_type) {
      case nir_deref_type_array: {
         nir_ssa_def *index = nir_ssa_for_src(b, (*p)->arr.index, 1);
         int stride = glsl_array_size((*p)->type);
         if (stride >= 0)
            offset = nir_iadd(b, offset, nir_amul_imm(b, index, stride));
         else
            offset = nir_iadd(b, offset, index);
         break;
      }
      default:
         unreachable("Not part of the path");
      }
   }
   nir_deref_path_finish(&path);
   return offset;
}


static variable_pair *
get_var_pair(nir_builder *b, nir_variable *old_var,
             struct hash_table  *split_vars)
{
   variable_pair *new_var = NULL;
   unsigned old_components = glsl_get_components(
                                glsl_without_array_or_matrix(old_var->type));

   assert(old_components > 2 && old_components <= 4);

   struct hash_entry *entry = _mesa_hash_table_search(split_vars, old_var);
   if (!entry) {
      new_var = (variable_pair *)calloc(1, sizeof(variable_pair));
      new_var->xy = nir_variable_clone(old_var, b->shader);
      new_var->zw = nir_variable_clone(old_var, b->shader);
      new_var->xy->type = glsl_dvec_type(2);
      new_var->zw->type = glsl_dvec_type(old_components - 2);

      if (glsl_type_is_array(old_var->type)) {
         unsigned array_size = glsl_get_aoa_size(old_var->type);
         new_var->xy->type = glsl_array_type(new_var->xy->type,
                                             array_size, 0);
         new_var->zw->type = glsl_array_type(new_var->zw->type,
                                             array_size, 0);
      }

      exec_list_push_tail(&b->impl->locals, &new_var->xy->node);
      exec_list_push_tail(&b->impl->locals, &new_var->zw->node);

      _mesa_hash_table_insert(split_vars, old_var, new_var);
   } else
       new_var = (variable_pair *)entry->data;
   return new_var;
}

static nir_ssa_def *
split_load_deref(nir_builder *b, nir_intrinsic_instr *intr,
                 nir_ssa_def *offset, struct hash_table *split_vars)
{
   nir_variable *old_var = nir_intrinsic_get_var(intr, 0);
   unsigned old_components = glsl_get_components(
                                glsl_without_array_or_matrix(old_var->type));

   variable_pair *vars = get_var_pair(b, old_var, split_vars);

   nir_deref_instr *deref1 = nir_build_deref_var(b, vars->xy);
   nir_deref_instr *deref2 = nir_build_deref_var(b, vars->zw);

   if (offset) {
      deref1 = nir_build_deref_array(b, deref1, offset);
      deref2 = nir_build_deref_array(b, deref2, offset);
   }

   nir_ssa_def *load1 = nir_build_load_deref(b, 2, 64, &deref1->dest.ssa, 0);
   nir_ssa_def *load2 = nir_build_load_deref(b, old_components - 2, 64,
                                             &deref2->dest.ssa, 0);

   return merge_to_vec3_or_vec4(b, load1, load2);
}

static nir_ssa_def *
split_store_deref(nir_builder *b, nir_intrinsic_instr *intr,
                  nir_ssa_def *offset, struct hash_table *split_vars)
{
   nir_variable *old_var = nir_intrinsic_get_var(intr, 0);

   variable_pair *vars = get_var_pair(b, old_var, split_vars);

   nir_deref_instr *deref_xy = nir_build_deref_var(b, vars->xy);
   nir_deref_instr *deref_zw = nir_build_deref_var(b, vars->zw);

   if (offset) {
      deref_xy = nir_build_deref_array(b, deref_xy, offset);
      deref_zw = nir_build_deref_array(b, deref_zw, offset);
   }

   int write_mask_xy = nir_intrinsic_write_mask(intr) & 3;
   if (write_mask_xy) {
      nir_ssa_def *src_xy = nir_channels(b, intr->src[1].ssa, 3);
      nir_build_store_deref(b, &deref_xy->dest.ssa, src_xy, write_mask_xy);
   }

   int write_mask_zw = nir_intrinsic_write_mask(intr) & 0xc;
   if (write_mask_zw) {
      nir_ssa_def *src_zw = nir_channels(b, intr->src[1].ssa, write_mask_zw);
      nir_build_store_deref(b, &deref_zw->dest.ssa, src_zw, write_mask_zw >> 2);
   }

   return NIR_LOWER_INSTR_PROGRESS_REPLACE;
}

static nir_ssa_def *
split_phi(nir_builder *b, nir_phi_instr *phi)
{
   nir_op vec_op = nir_op_vec(phi->dest.ssa.num_components);

   nir_alu_instr *vec = nir_alu_instr_create(b->shader, vec_op);
   nir_ssa_dest_init(&vec->instr, &vec->dest.dest,
                     phi->dest.ssa.num_components,
                     64, NULL);
   vec->dest.write_mask = (1 << phi->dest.ssa.num_components) - 1;

   int num_comp[2] = {2, phi->dest.ssa.num_components - 2};

   nir_phi_instr *new_phi[2];

   for (unsigned i = 0; i < 2; i++) {
      new_phi[i] = nir_phi_instr_create(b->shader);
      nir_ssa_dest_init(&new_phi[i]->instr, &new_phi[i]->dest, num_comp[i],
                        phi->dest.ssa.bit_size, NULL);

      nir_foreach_phi_src(src, phi) {
         /* Insert at the end of the predecessor but before the jump
          * (This was inspired by nir_lower_phi_to_scalar) */
         nir_instr *pred_last_instr = nir_block_last_instr(src->pred);

         if (pred_last_instr && pred_last_instr->type == nir_instr_type_jump)
            b->cursor = nir_before_instr(pred_last_instr);
         else
            b->cursor = nir_after_block(src->pred);

         nir_ssa_def *new_src = nir_channels(b, src->src.ssa,
                                             ((1 << num_comp[i]) - 1) << (2 * i));

         nir_phi_instr_add_src(new_phi[i], src->pred, nir_src_for_ssa(new_src));
      }
      nir_instr_insert_before(&phi->instr, &new_phi[i]->instr);
   }

   b->cursor = nir_after_instr(&phi->instr);
   return merge_to_vec3_or_vec4(b, &new_phi[0]->dest.ssa, &new_phi[1]->dest.ssa);
};

static nir_ssa_def *
nir_split_64bit_vec3_and_vec4_impl(nir_builder *b, nir_instr *instr, void *d)
{
   struct hash_table *split_vars = (struct hash_table *)d;

   switch (instr->type) {

   case nir_instr_type_intrinsic: {

      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      switch (intr->intrinsic) {

      case nir_intrinsic_load_deref: {
         nir_deref_instr *deref =
               nir_instr_as_deref(intr->src[0].ssa->parent_instr);
         if (deref->deref_type == nir_deref_type_var)
            return split_load_deref(b, intr, NULL, split_vars);
         else if (deref->deref_type == nir_deref_type_array) {
            return split_load_deref(b, intr, get_linear_array_offset(b, deref), split_vars);
         }
         else
            unreachable("Only splitting of loads from vars and arrays");
      }

      case nir_intrinsic_store_deref: {
         nir_deref_instr *deref =
               nir_instr_as_deref(intr->src[0].ssa->parent_instr);
         if (deref->deref_type == nir_deref_type_var)
            return split_store_deref(b, intr, NULL, split_vars);
         else if (deref->deref_type == nir_deref_type_array)
            return split_store_deref(b, intr,  get_linear_array_offset(b, deref), split_vars);
         else
            unreachable("Only splitting of stores to vars and arrays");
         }

      default:
         unreachable("Only splitting load_deref and store_deref");
      }
   }

   case nir_instr_type_phi: {
      nir_phi_instr *phi = nir_instr_as_phi(instr);
      return split_phi(b, phi);
   }

   default:
      unreachable("Only splitting load_deref/store_deref and phi");
   }

   return NULL;
}


bool
nir_split_64bit_vec3_and_vec4(nir_shader *sh)
{
   struct hash_table *split_vars = _mesa_pointer_hash_table_create(NULL);

   bool progress =
         nir_shader_lower_instructions(sh,
                                       nir_split_64bit_vec3_and_vec4_filter,
                                       nir_split_64bit_vec3_and_vec4_impl,
                                       split_vars);

   _mesa_hash_table_destroy(split_vars, NULL);
   return progress;
}
