/*
 * Copyright © 2019 Intel Corporation
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
#include "util/u_dynarray.h"

/** @file nir_lower_io_to_vector.c
 *
 * Merges compatible input/output variables residing in different components
 * of the same location. It's expected that further passes such as
 * nir_lower_io_to_temporaries will combine loads and stores of the merged
 * variables, producing vector nir_load_input/nir_store_output instructions
 * when all is said and done.
 */

/* FRAG_RESULT_MAX+1 instead of just FRAG_RESULT_MAX because of how this pass
 * handles dual source blending */
#define MAX_SLOTS MAX2(VARYING_SLOT_TESS_MAX, FRAG_RESULT_MAX+1)

static unsigned
get_slot(const nir_variable *var)
{
   /* This handling of dual-source blending might not be correct when more than
    * one render target is supported, but it seems no driver supports more than
    * one. */
   return var->data.location + var->data.index;
}

static const struct glsl_type *
get_per_vertex_type(const nir_shader *shader, const nir_variable *var,
                    unsigned *num_vertices)
{
   if (nir_is_arrayed_io(var, shader->info.stage)) {
      assert(glsl_type_is_array(var->type));
      if (num_vertices)
         *num_vertices = glsl_get_length(var->type);
      return glsl_get_array_element(var->type);
   } else {
      if (num_vertices)
         *num_vertices = 0;
      return var->type;
   }
}

static const struct glsl_type *
resize_array_vec_type(const struct glsl_type *type, unsigned num_components)
{
   if (glsl_type_is_array(type)) {
      const struct glsl_type *arr_elem =
         resize_array_vec_type(glsl_get_array_element(type), num_components);
      return glsl_array_type(arr_elem, glsl_get_length(type), 0);
   } else {
      assert(glsl_type_is_vector_or_scalar(type));
      return glsl_vector_type(glsl_get_base_type(type), num_components);
   }
}

static bool
variables_can_merge(const nir_shader *shader,
                    const nir_variable *a, const nir_variable *b,
                    bool same_array_structure)
{
   if (a->data.compact || b->data.compact)
      return false;

   if (a->data.per_view || b->data.per_view)
      return false;

   const struct glsl_type *a_type_tail = a->type;
   const struct glsl_type *b_type_tail = b->type;

   if (nir_is_arrayed_io(a, shader->info.stage) !=
       nir_is_arrayed_io(b, shader->info.stage))
      return false;

   /* They must have the same array structure */
   if (same_array_structure) {
      while (glsl_type_is_array(a_type_tail)) {
         if (!glsl_type_is_array(b_type_tail))
            return false;

         if (glsl_get_length(a_type_tail) != glsl_get_length(b_type_tail))
            return false;

         a_type_tail = glsl_get_array_element(a_type_tail);
         b_type_tail = glsl_get_array_element(b_type_tail);
      }
      if (glsl_type_is_array(b_type_tail))
         return false;
   } else {
      a_type_tail = glsl_without_array(a_type_tail);
      b_type_tail = glsl_without_array(b_type_tail);
   }

   if (!glsl_type_is_vector_or_scalar(a_type_tail) ||
       !glsl_type_is_vector_or_scalar(b_type_tail))
      return false;

   if (glsl_get_base_type(a_type_tail) != glsl_get_base_type(b_type_tail))
      return false;

   /* TODO: add 64/16bit support ? */
   if (glsl_get_bit_size(a_type_tail) != 32)
      return false;

   assert(a->data.mode == b->data.mode);
   if (shader->info.stage == MESA_SHADER_FRAGMENT &&
       a->data.mode == nir_var_shader_in &&
       (a->data.interpolation != b->data.interpolation ||
        a->data.centroid != b->data.centroid ||
        a->data.sample != b->data.sample))
      return false;

   if (shader->info.stage == MESA_SHADER_FRAGMENT &&
       a->data.mode == nir_var_shader_out &&
       a->data.index != b->data.index)
      return false;

   /* It's tricky to merge XFB-outputs correctly, because we need there
    * to not be any overlaps when we get to
    * nir_gather_xfb_info_with_varyings later on. We'll end up
    * triggering an assert there if we merge here.
    */
   if ((shader->info.stage == MESA_SHADER_VERTEX ||
        shader->info.stage == MESA_SHADER_TESS_EVAL ||
        shader->info.stage == MESA_SHADER_GEOMETRY) &&
       a->data.mode == nir_var_shader_out &&
       (a->data.explicit_xfb_buffer || b->data.explicit_xfb_buffer))
      return false;

   return true;
}

static const struct glsl_type *
get_flat_type(const nir_shader *shader, nir_variable *old_vars[MAX_SLOTS][4],
              unsigned *loc, nir_variable **first_var, unsigned *num_vertices)
{
   unsigned todo = 1;
   unsigned slots = 0;
   unsigned num_vars = 0;
   enum glsl_base_type base;
   *num_vertices = 0;
   *first_var = NULL;

   while (todo) {
      assert(*loc < MAX_SLOTS);
      for (unsigned frac = 0; frac < 4; frac++) {
         nir_variable *var = old_vars[*loc][frac];
         if (!var)
            continue;
         if ((*first_var &&
              !variables_can_merge(shader, var, *first_var, false)) ||
             var->data.compact) {
            (*loc)++;
            return NULL;
         }

         if (!*first_var) {
            if (!glsl_type_is_vector_or_scalar(glsl_without_array(var->type))) {
               (*loc)++;
               return NULL;
            }
            *first_var = var;
            base = glsl_get_base_type(
               glsl_without_array(get_per_vertex_type(shader, var, NULL)));
         }

         bool vs_in = shader->info.stage == MESA_SHADER_VERTEX &&
                      var->data.mode == nir_var_shader_in;
         unsigned var_slots = glsl_count_attribute_slots(
            get_per_vertex_type(shader, var, num_vertices), vs_in);
         todo = MAX2(todo, var_slots);
         num_vars++;
      }
      todo--;
      slots++;
      (*loc)++;
   }

   if (num_vars <= 1)
      return NULL;

   if (slots == 1)
      return glsl_vector_type(base, 4);
   else
      return glsl_array_type(glsl_vector_type(base, 4), slots, 0);
}

static bool
create_new_io_vars(nir_shader *shader, nir_variable_mode mode,
                   nir_variable *new_vars[MAX_SLOTS][4],
                   bool flat_vars[MAX_SLOTS],
                   struct util_dynarray *demote_vars)
{
   nir_variable *old_vars[MAX_SLOTS][4] = {{0}};

   bool has_io_var = false;
   nir_foreach_variable_with_modes(var, shader, mode) {
      unsigned frac = var->data.location_frac;
      old_vars[get_slot(var)][frac] = var;
      has_io_var = true;
   }

   if (!has_io_var)
      return false;

   bool merged_any_vars = false;

   for (unsigned loc = 0; loc < MAX_SLOTS; loc++) {
      unsigned frac = 0;
      while (frac < 4) {
         nir_variable *first_var = old_vars[loc][frac];
         if (!first_var) {
            frac++;
            continue;
         }

         int first = frac;
         bool found_merge = false;

         while (frac < 4) {
            nir_variable *var = old_vars[loc][frac];
            if (!var)
               break;

            if (var != first_var) {
               if (!variables_can_merge(shader, first_var, var, true))
                  break;

               found_merge = true;
            }

            const unsigned num_components =
               glsl_get_components(glsl_without_array(var->type));
            if (!num_components) {
               assert(frac == 0);
               frac++;
               break; /* The type was a struct. */
            }

            /* We had better not have any overlapping vars */
            for (unsigned i = 1; i < num_components; i++)
               assert(old_vars[loc][frac + i] == NULL);

            frac += num_components;
         }

         if (!found_merge)
            continue;

         merged_any_vars = true;

         nir_variable *var = nir_variable_clone(old_vars[loc][first], shader);
         var->data.location_frac = first;
         var->type = resize_array_vec_type(var->type, frac - first);

         nir_shader_add_variable(shader, var);
         for (unsigned i = first; i < frac; i++) {
            new_vars[loc][i] = var;
            if (old_vars[loc][i]) {
               util_dynarray_append(demote_vars, nir_variable *, old_vars[loc][i]);
               old_vars[loc][i] = NULL;
            }
         }

         old_vars[loc][first] = var;
      }
   }

   /* "flat" mode: tries to ensure there is at most one variable per slot by
    * merging variables into vec4s
    */
   for (unsigned loc = 0; loc < MAX_SLOTS;) {
      nir_variable *first_var;
      unsigned num_vertices;
      unsigned new_loc = loc;
      const struct glsl_type *flat_type =
         get_flat_type(shader, old_vars, &new_loc, &first_var, &num_vertices);
      if (flat_type) {
         merged_any_vars = true;

         nir_variable *var = nir_variable_clone(first_var, shader);
         var->data.location_frac = 0;
         if (num_vertices)
            var->type = glsl_array_type(flat_type, num_vertices, 0);
         else
            var->type = flat_type;

         nir_shader_add_variable(shader, var);
         unsigned num_slots = MAX2(glsl_get_length(flat_type), 1);
         for (unsigned i = 0; i < num_slots; i++) {
            for (unsigned j = 0; j < 4; j++)
               new_vars[loc + i][j] = var;
            flat_vars[loc + i] = true;
         }
      }
      loc = new_loc;
   }

   return merged_any_vars;
}

static nir_deref_instr *
build_array_deref_of_new_var(nir_builder *b, nir_variable *new_var,
                             nir_deref_instr *leader)
{
   if (leader->deref_type == nir_deref_type_var)
      return nir_build_deref_var(b, new_var);

   nir_deref_instr *parent =
      build_array_deref_of_new_var(b, new_var, nir_deref_instr_parent(leader));

   return nir_build_deref_follower(b, parent, leader);
}

static nir_ssa_def *
build_array_index(nir_builder *b, nir_deref_instr *deref, nir_ssa_def *base,
                  bool vs_in, bool per_vertex)
{
   switch (deref->deref_type) {
   case nir_deref_type_var:
      return base;
   case nir_deref_type_array: {
      nir_ssa_def *index = nir_i2iN(b, deref->arr.index.ssa,
                                   deref->dest.ssa.bit_size);

      if (nir_deref_instr_parent(deref)->deref_type == nir_deref_type_var &&
          per_vertex)
         return base;

      return nir_iadd(
         b, build_array_index(b, nir_deref_instr_parent(deref), base, vs_in, per_vertex),
         nir_amul_imm(b, index, glsl_count_attribute_slots(deref->type, vs_in)));
   }
   default:
      unreachable("Invalid deref instruction type");
   }
}

static nir_deref_instr *
build_array_deref_of_new_var_flat(nir_shader *shader,
                                  nir_builder *b, nir_variable *new_var,
                                  nir_deref_instr *leader, unsigned base)
{
   nir_deref_instr *deref = nir_build_deref_var(b, new_var);

   bool per_vertex = nir_is_arrayed_io(new_var, shader->info.stage);
   if (per_vertex) {
      nir_deref_path path;
      nir_deref_path_init(&path, leader, NULL);

      assert(path.path[0]->deref_type == nir_deref_type_var);
      nir_deref_instr *p = path.path[1];
      nir_deref_path_finish(&path);

      nir_ssa_def *index = p->arr.index.ssa;
      deref = nir_build_deref_array(b, deref, index);
   }

   if (!glsl_type_is_array(deref->type))
      return deref;

   bool vs_in = shader->info.stage == MESA_SHADER_VERTEX &&
                new_var->data.mode == nir_var_shader_in;
   return nir_build_deref_array(b, deref,
      build_array_index(b, leader, nir_imm_int(b, base), vs_in, per_vertex));
}

ASSERTED static bool
nir_shader_can_read_output(const shader_info *info)
{
   switch (info->stage) {
   case MESA_SHADER_TESS_CTRL:
   case MESA_SHADER_FRAGMENT:
      return true;

   case MESA_SHADER_TASK:
   case MESA_SHADER_MESH:
      /* TODO(mesh): This will not be allowed on EXT. */
      return true;

   default:
      return false;
   }
}

static bool
nir_lower_io_to_vector_impl(nir_function_impl *impl, nir_variable_mode modes)
{
   assert(!(modes & ~(nir_var_shader_in | nir_var_shader_out)));

   nir_builder b;
   nir_builder_init(&b, impl);

   nir_metadata_require(impl, nir_metadata_dominance);

   struct util_dynarray demote_vars;
   util_dynarray_init(&demote_vars, NULL);

   nir_shader *shader = impl->function->shader;
   nir_variable *new_inputs[MAX_SLOTS][4] = {{0}};
   nir_variable *new_outputs[MAX_SLOTS][4] = {{0}};
   bool flat_inputs[MAX_SLOTS] = {0};
   bool flat_outputs[MAX_SLOTS] = {0};

   if (modes & nir_var_shader_in) {
      /* Vertex shaders support overlapping inputs.  We don't do those */
      assert(b.shader->info.stage != MESA_SHADER_VERTEX);

      /* If we don't actually merge any variables, remove that bit from modes
       * so we don't bother doing extra non-work.
       */
      if (!create_new_io_vars(shader, nir_var_shader_in,
                              new_inputs, flat_inputs, &demote_vars))
         modes &= ~nir_var_shader_in;
   }

   if (modes & nir_var_shader_out) {
      /* If we don't actually merge any variables, remove that bit from modes
       * so we don't bother doing extra non-work.
       */
      if (!create_new_io_vars(shader, nir_var_shader_out,
                              new_outputs, flat_outputs, &demote_vars))
         modes &= ~nir_var_shader_out;
   }

   if (!modes)
      return false;

   bool progress = false;

   /* Actually lower all the IO load/store intrinsics.  Load instructions are
    * lowered to a vector load and an ALU instruction to grab the channels we
    * want.  Outputs are lowered to a write-masked store of the vector output.
    * For non-TCS outputs, we then run nir_lower_io_to_temporaries at the end
    * to clean up the partial writes.
    */
   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;

         nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);

         switch (intrin->intrinsic) {
         case nir_intrinsic_load_deref:
         case nir_intrinsic_interp_deref_at_centroid:
         case nir_intrinsic_interp_deref_at_sample:
         case nir_intrinsic_interp_deref_at_offset:
         case nir_intrinsic_interp_deref_at_vertex: {
            nir_deref_instr *old_deref = nir_src_as_deref(intrin->src[0]);
            if (!nir_deref_mode_is_one_of(old_deref, modes))
               break;

            if (nir_deref_mode_is(old_deref, nir_var_shader_out))
               assert(nir_shader_can_read_output(&b.shader->info));

            nir_variable *old_var = nir_deref_instr_get_variable(old_deref);

            const unsigned loc = get_slot(old_var);
            const unsigned old_frac = old_var->data.location_frac;
            nir_variable *new_var = old_var->data.mode == nir_var_shader_in ?
                                    new_inputs[loc][old_frac] :
                                    new_outputs[loc][old_frac];
            bool flat = old_var->data.mode == nir_var_shader_in ?
                        flat_inputs[loc] : flat_outputs[loc];
            if (!new_var)
               break;

            const unsigned new_frac = new_var->data.location_frac;

            nir_component_mask_t vec4_comp_mask =
               ((1 << intrin->num_components) - 1) << old_frac;

            b.cursor = nir_before_instr(&intrin->instr);

            /* Rewrite the load to use the new variable and only select a
             * portion of the result.
             */
            nir_deref_instr *new_deref;
            if (flat) {
               new_deref = build_array_deref_of_new_var_flat(
                  shader, &b, new_var, old_deref, loc - get_slot(new_var));
            } else {
               assert(get_slot(new_var) == loc);
               new_deref = build_array_deref_of_new_var(&b, new_var, old_deref);
               assert(glsl_type_is_vector(new_deref->type));
            }
            nir_instr_rewrite_src(&intrin->instr, &intrin->src[0],
                                  nir_src_for_ssa(&new_deref->dest.ssa));

            intrin->num_components =
               glsl_get_components(new_deref->type);
            intrin->dest.ssa.num_components = intrin->num_components;

            b.cursor = nir_after_instr(&intrin->instr);

            nir_ssa_def *new_vec = nir_channels(&b, &intrin->dest.ssa,
                                                vec4_comp_mask >> new_frac);
            nir_ssa_def_rewrite_uses_after(&intrin->dest.ssa,
                                           new_vec,
                                           new_vec->parent_instr);

            progress = true;
            break;
         }

         case nir_intrinsic_store_deref: {
            nir_deref_instr *old_deref = nir_src_as_deref(intrin->src[0]);
            if (!nir_deref_mode_is(old_deref, nir_var_shader_out))
               break;

            nir_variable *old_var = nir_deref_instr_get_variable(old_deref);

            const unsigned loc = get_slot(old_var);
            const unsigned old_frac = old_var->data.location_frac;
            nir_variable *new_var = new_outputs[loc][old_frac];
            bool flat = flat_outputs[loc];
            if (!new_var)
               break;

            const unsigned new_frac = new_var->data.location_frac;

            b.cursor = nir_before_instr(&intrin->instr);

            /* Rewrite the store to be a masked store to the new variable */
            nir_deref_instr *new_deref;
            if (flat) {
               new_deref = build_array_deref_of_new_var_flat(
                  shader, &b, new_var, old_deref, loc - get_slot(new_var));
            } else {
               assert(get_slot(new_var) == loc);
               new_deref = build_array_deref_of_new_var(&b, new_var, old_deref);
               assert(glsl_type_is_vector(new_deref->type));
            }
            nir_instr_rewrite_src(&intrin->instr, &intrin->src[0],
                                  nir_src_for_ssa(&new_deref->dest.ssa));

            intrin->num_components =
               glsl_get_components(new_deref->type);

            nir_component_mask_t old_wrmask = nir_intrinsic_write_mask(intrin);

            assert(intrin->src[1].is_ssa);
            nir_ssa_def *old_value = intrin->src[1].ssa;
            nir_ssa_scalar comps[4];
            for (unsigned c = 0; c < intrin->num_components; c++) {
               if (new_frac + c >= old_frac &&
                   (old_wrmask & 1 << (new_frac + c - old_frac))) {
                  comps[c] = nir_get_ssa_scalar(old_value,
                                         new_frac + c - old_frac);
               } else {
                  comps[c] = nir_get_ssa_scalar(nir_ssa_undef(&b, old_value->num_components,
                                                              old_value->bit_size), 0);
               }
            }
            nir_ssa_def *new_value = nir_vec_scalars(&b, comps, intrin->num_components);
            nir_instr_rewrite_src(&intrin->instr, &intrin->src[1],
                                  nir_src_for_ssa(new_value));

            nir_intrinsic_set_write_mask(intrin,
                                         old_wrmask << (old_frac - new_frac));

            progress = true;
            break;
         }

         default:
            break;
         }
      }
   }

   /* Demote the old var to a global, so that things like
    * nir_lower_io_to_temporaries() don't trigger on it.
    */
   util_dynarray_foreach(&demote_vars, nir_variable *, varp) {
      (*varp)->data.mode = nir_var_shader_temp;
   }
   nir_fixup_deref_modes(b.shader);
   util_dynarray_fini(&demote_vars);

   if (progress) {
      nir_metadata_preserve(impl, nir_metadata_block_index |
                                  nir_metadata_dominance);
   }

   return progress;
}

bool
nir_lower_io_to_vector(nir_shader *shader, nir_variable_mode modes)
{
   bool progress = false;

   nir_foreach_function(function, shader) {
      if (function->impl)
         progress |= nir_lower_io_to_vector_impl(function->impl, modes);
   }

   return progress;
}

static bool
nir_vectorize_tess_levels_impl(nir_function_impl *impl)
{
   bool progress = false;
   nir_builder b;
   nir_builder_init(&b, impl);

   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;

         nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
         if (intrin->intrinsic != nir_intrinsic_load_deref &&
             intrin->intrinsic != nir_intrinsic_store_deref)
            continue;

         nir_deref_instr *deref = nir_src_as_deref(intrin->src[0]);
         if (!nir_deref_mode_is(deref, nir_var_shader_out))
            continue;

         nir_variable *var = nir_deref_instr_get_variable(deref);
         if (var->data.location != VARYING_SLOT_TESS_LEVEL_OUTER &&
             var->data.location != VARYING_SLOT_TESS_LEVEL_INNER)
            continue;

         assert(deref->deref_type == nir_deref_type_array);
         assert(nir_src_is_const(deref->arr.index));
         unsigned index = nir_src_as_uint(deref->arr.index);
         unsigned vec_size = glsl_get_vector_elements(var->type);

         b.cursor = nir_before_instr(instr);
         nir_ssa_def *new_deref = &nir_build_deref_var(&b, var)->dest.ssa;
         nir_instr_rewrite_src(instr, &intrin->src[0], nir_src_for_ssa(new_deref));

         nir_deref_instr_remove_if_unused(deref);

         intrin->num_components = vec_size;

         /* Handle out of bounds access. */
         if (index >= vec_size) {
            if (intrin->intrinsic == nir_intrinsic_load_deref) {
               /* Return undef from out of bounds loads. */
               b.cursor = nir_after_instr(instr);
               nir_ssa_def *val = &intrin->dest.ssa;
               nir_ssa_def *u = nir_ssa_undef(&b, val->num_components, val->bit_size);
               nir_ssa_def_rewrite_uses(val, u);
            }

            /* Finally, remove the out of bounds access. */
            nir_instr_remove(instr);
            progress = true;
            continue;
         }

         if (intrin->intrinsic == nir_intrinsic_store_deref) {
            nir_intrinsic_set_write_mask(intrin, 1 << index);
            nir_ssa_def *new_val = nir_ssa_undef(&b, intrin->num_components, 32);
            new_val = nir_vector_insert_imm(&b, new_val, intrin->src[1].ssa, index);
            nir_instr_rewrite_src(instr, &intrin->src[1], nir_src_for_ssa(new_val));
         } else {
            b.cursor = nir_after_instr(instr);
            nir_ssa_def *val = &intrin->dest.ssa;
            val->num_components = intrin->num_components;
            nir_ssa_def *comp = nir_channel(&b, val, index);
            nir_ssa_def_rewrite_uses_after(val, comp, comp->parent_instr);
         }

         progress = true;
      }
   }

   if (progress)
      nir_metadata_preserve(impl, nir_metadata_block_index | nir_metadata_dominance);
   else
      nir_metadata_preserve(impl, nir_metadata_all);

   return progress;
}

/* Make the tess factor variables vectors instead of compact arrays, so accesses
 * can be combined by nir_opt_cse()/nir_opt_combine_stores().
 */
bool
nir_vectorize_tess_levels(nir_shader *shader)
{
   bool progress = false;

   nir_foreach_shader_out_variable(var, shader) {
      if (var->data.location == VARYING_SLOT_TESS_LEVEL_OUTER ||
          var->data.location == VARYING_SLOT_TESS_LEVEL_INNER) {
         var->type = glsl_vector_type(GLSL_TYPE_FLOAT, glsl_get_length(var->type));
         var->data.compact = false;
         progress = true;
      }
   }

   nir_foreach_function(function, shader) {
      if (function->impl)
         progress |= nir_vectorize_tess_levels_impl(function->impl);
   }

   return progress;
}
