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
#include "nir_builder.h"
#include "util/set.h"
#include "util/hash_table.h"

/* This file contains various little helpers for doing simple linking in
 * NIR.  Eventually, we'll probably want a full-blown varying packing
 * implementation in here.  Right now, it just deletes unused things.
 */

/**
 * Returns the bits in the inputs_read, or outputs_written
 * bitfield corresponding to this variable.
 */
static uint64_t
get_variable_io_mask(nir_variable *var, gl_shader_stage stage)
{
   if (var->data.location < 0)
      return 0;

   unsigned location = var->data.patch ?
      var->data.location - VARYING_SLOT_PATCH0 : var->data.location;

   assert(var->data.mode == nir_var_shader_in ||
          var->data.mode == nir_var_shader_out);
   assert(var->data.location >= 0);
   assert(location < 64);

   const struct glsl_type *type = var->type;
   if (nir_is_arrayed_io(var, stage) || var->data.per_view) {
      assert(glsl_type_is_array(type));
      type = glsl_get_array_element(type);
   }

   unsigned slots = glsl_count_attribute_slots(type, false);
   return BITFIELD64_MASK(slots) << location;
}

static bool
is_non_generic_patch_var(nir_variable *var)
{
   return var->data.location == VARYING_SLOT_TESS_LEVEL_INNER ||
          var->data.location == VARYING_SLOT_TESS_LEVEL_OUTER ||
          var->data.location == VARYING_SLOT_BOUNDING_BOX0 ||
          var->data.location == VARYING_SLOT_BOUNDING_BOX1;
}

static uint8_t
get_num_components(nir_variable *var)
{
   if (glsl_type_is_struct_or_ifc(glsl_without_array(var->type)))
      return 4;

   return glsl_get_vector_elements(glsl_without_array(var->type));
}

static void
tcs_add_output_reads(nir_shader *shader, uint64_t *read, uint64_t *patches_read)
{
   nir_foreach_function(function, shader) {
      if (!function->impl)
         continue;

      nir_foreach_block(block, function->impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
            if (intrin->intrinsic != nir_intrinsic_load_deref)
               continue;

            nir_deref_instr *deref = nir_src_as_deref(intrin->src[0]);
            if (!nir_deref_mode_is(deref, nir_var_shader_out))
               continue;

            nir_variable *var = nir_deref_instr_get_variable(deref);
            for (unsigned i = 0; i < get_num_components(var); i++) {
               if (var->data.patch) {
                  if (is_non_generic_patch_var(var))
                     continue;

                  patches_read[var->data.location_frac + i] |=
                     get_variable_io_mask(var, shader->info.stage);
               } else {
                  read[var->data.location_frac + i] |=
                     get_variable_io_mask(var, shader->info.stage);
               }
            }
         }
      }
   }
}

/**
 * Helper for removing unused shader I/O variables, by demoting them to global
 * variables (which may then by dead code eliminated).
 *
 * Example usage is:
 *
 * progress = nir_remove_unused_io_vars(producer, nir_var_shader_out,
 *                                      read, patches_read) ||
 *                                      progress;
 *
 * The "used" should be an array of 4 uint64_ts (probably of VARYING_BIT_*)
 * representing each .location_frac used.  Note that for vector variables,
 * only the first channel (.location_frac) is examined for deciding if the
 * variable is used!
 */
bool
nir_remove_unused_io_vars(nir_shader *shader,
                          nir_variable_mode mode,
                          uint64_t *used_by_other_stage,
                          uint64_t *used_by_other_stage_patches)
{
   bool progress = false;
   uint64_t *used;

   assert(mode == nir_var_shader_in || mode == nir_var_shader_out);

   nir_foreach_variable_with_modes_safe(var, shader, mode) {
      if (var->data.patch)
         used = used_by_other_stage_patches;
      else
         used = used_by_other_stage;

      if (var->data.location < VARYING_SLOT_VAR0 && var->data.location >= 0)
         if (shader->info.stage != MESA_SHADER_MESH || var->data.location != VARYING_SLOT_PRIMITIVE_ID)
            continue;

      if (var->data.always_active_io)
         continue;

      if (var->data.explicit_xfb_buffer)
         continue;

      uint64_t other_stage = used[var->data.location_frac];

      if (!(other_stage & get_variable_io_mask(var, shader->info.stage))) {
         /* This one is invalid, make it a global variable instead */
         if (shader->info.stage == MESA_SHADER_MESH &&
               (shader->info.outputs_read & BITFIELD64_BIT(var->data.location)))
            var->data.mode = nir_var_mem_shared;
         else
            var->data.mode = nir_var_shader_temp;
         var->data.location = 0;

         progress = true;
      }
   }

   nir_function_impl *impl = nir_shader_get_entrypoint(shader);
   if (progress) {
      nir_metadata_preserve(impl, nir_metadata_dominance |
                            nir_metadata_block_index);
      nir_fixup_deref_modes(shader);
   } else {
      nir_metadata_preserve(impl, nir_metadata_all);
   }

   return progress;
}

bool
nir_remove_unused_varyings(nir_shader *producer, nir_shader *consumer)
{
   assert(producer->info.stage != MESA_SHADER_FRAGMENT);
   assert(consumer->info.stage != MESA_SHADER_VERTEX);

   uint64_t read[4] = { 0 }, written[4] = { 0 };
   uint64_t patches_read[4] = { 0 }, patches_written[4] = { 0 };

   nir_foreach_shader_out_variable(var, producer) {
      for (unsigned i = 0; i < get_num_components(var); i++) {
         if (var->data.patch) {
            if (is_non_generic_patch_var(var))
               continue;

            patches_written[var->data.location_frac + i] |=
               get_variable_io_mask(var, producer->info.stage);
         } else {
            written[var->data.location_frac + i] |=
               get_variable_io_mask(var, producer->info.stage);
         }
      }
   }

   nir_foreach_shader_in_variable(var, consumer) {
      for (unsigned i = 0; i < get_num_components(var); i++) {
         if (var->data.patch) {
            if (is_non_generic_patch_var(var))
               continue;

            patches_read[var->data.location_frac + i] |=
               get_variable_io_mask(var, consumer->info.stage);
         } else {
            read[var->data.location_frac + i] |=
               get_variable_io_mask(var, consumer->info.stage);
         }
      }
   }

   /* Each TCS invocation can read data written by other TCS invocations,
    * so even if the outputs are not used by the TES we must also make
    * sure they are not read by the TCS before demoting them to globals.
    */
   if (producer->info.stage == MESA_SHADER_TESS_CTRL)
      tcs_add_output_reads(producer, read, patches_read);

   bool progress = false;
   progress = nir_remove_unused_io_vars(producer, nir_var_shader_out, read,
                                        patches_read);

   progress = nir_remove_unused_io_vars(consumer, nir_var_shader_in, written,
                                        patches_written) || progress;

   return progress;
}

static uint8_t
get_interp_type(nir_variable *var, const struct glsl_type *type,
                bool default_to_smooth_interp)
{
   if (var->data.per_primitive)
      return INTERP_MODE_NONE;
   if (glsl_type_is_integer(type))
      return INTERP_MODE_FLAT;
   else if (var->data.interpolation != INTERP_MODE_NONE)
      return var->data.interpolation;
   else if (default_to_smooth_interp)
      return INTERP_MODE_SMOOTH;
   else
      return INTERP_MODE_NONE;
}

#define INTERPOLATE_LOC_SAMPLE 0
#define INTERPOLATE_LOC_CENTROID 1
#define INTERPOLATE_LOC_CENTER 2

static uint8_t
get_interp_loc(nir_variable *var)
{
   if (var->data.sample)
      return INTERPOLATE_LOC_SAMPLE;
   else if (var->data.centroid)
      return INTERPOLATE_LOC_CENTROID;
   else
      return INTERPOLATE_LOC_CENTER;
}

static bool
is_packing_supported_for_type(const struct glsl_type *type)
{
   /* We ignore complex types such as arrays, matrices, structs and bitsizes
    * other then 32bit. All other vector types should have been split into
    * scalar variables by the lower_io_to_scalar pass. The only exception
    * should be OpenGL xfb varyings.
    * TODO: add support for more complex types?
    */
   return glsl_type_is_scalar(type) && glsl_type_is_32bit(type);
}

struct assigned_comps
{
   uint8_t comps;
   uint8_t interp_type;
   uint8_t interp_loc;
   bool is_32bit;
   bool is_mediump;
   bool is_per_primitive;
};

/* Packing arrays and dual slot varyings is difficult so to avoid complex
 * algorithms this function just assigns them their existing location for now.
 * TODO: allow better packing of complex types.
 */
static void
get_unmoveable_components_masks(nir_shader *shader,
                                nir_variable_mode mode,
                                struct assigned_comps *comps,
                                gl_shader_stage stage,
                                bool default_to_smooth_interp)
{
   nir_foreach_variable_with_modes_safe(var, shader, mode) {
      assert(var->data.location >= 0);

      /* Only remap things that aren't built-ins. */
      if (var->data.location >= VARYING_SLOT_VAR0 &&
          var->data.location - VARYING_SLOT_VAR0 < MAX_VARYINGS_INCL_PATCH) {

         const struct glsl_type *type = var->type;
         if (nir_is_arrayed_io(var, stage) || var->data.per_view) {
            assert(glsl_type_is_array(type));
            type = glsl_get_array_element(type);
         }

         /* If we can pack this varying then don't mark the components as
          * used.
          */
         if (is_packing_supported_for_type(type) &&
             !var->data.always_active_io)
            continue;

         unsigned location = var->data.location - VARYING_SLOT_VAR0;

         unsigned elements =
            glsl_type_is_vector_or_scalar(glsl_without_array(type)) ?
            glsl_get_vector_elements(glsl_without_array(type)) : 4;

         bool dual_slot = glsl_type_is_dual_slot(glsl_without_array(type));
         unsigned slots = glsl_count_attribute_slots(type, false);
         unsigned dmul = glsl_type_is_64bit(glsl_without_array(type)) ? 2 : 1;
         unsigned comps_slot2 = 0;
         for (unsigned i = 0; i < slots; i++) {
            if (dual_slot) {
               if (i & 1) {
                  comps[location + i].comps |= ((1 << comps_slot2) - 1);
               } else {
                  unsigned num_comps = 4 - var->data.location_frac;
                  comps_slot2 = (elements * dmul) - num_comps;

                  /* Assume ARB_enhanced_layouts packing rules for doubles */
                  assert(var->data.location_frac == 0 ||
                         var->data.location_frac == 2);
                  assert(comps_slot2 <= 4);

                  comps[location + i].comps |=
                     ((1 << num_comps) - 1) << var->data.location_frac;
               }
            } else {
               comps[location + i].comps |=
                  ((1 << (elements * dmul)) - 1) << var->data.location_frac;
            }

            comps[location + i].interp_type =
               get_interp_type(var, type, default_to_smooth_interp);
            comps[location + i].interp_loc = get_interp_loc(var);
            comps[location + i].is_32bit =
               glsl_type_is_32bit(glsl_without_array(type));
            comps[location + i].is_mediump =
               var->data.precision == GLSL_PRECISION_MEDIUM ||
               var->data.precision == GLSL_PRECISION_LOW;
            comps[location + i].is_per_primitive = var->data.per_primitive;
         }
      }
   }
}

struct varying_loc
{
   uint8_t component;
   uint32_t location;
};

static void
mark_all_used_slots(nir_variable *var, uint64_t *slots_used,
                    uint64_t slots_used_mask, unsigned num_slots)
{
   unsigned loc_offset = var->data.patch ? VARYING_SLOT_PATCH0 : 0;

   slots_used[var->data.patch ? 1 : 0] |= slots_used_mask &
      BITFIELD64_RANGE(var->data.location - loc_offset, num_slots);
}

static void
mark_used_slot(nir_variable *var, uint64_t *slots_used, unsigned offset)
{
   unsigned loc_offset = var->data.patch ? VARYING_SLOT_PATCH0 : 0;

   slots_used[var->data.patch ? 1 : 0] |=
      BITFIELD64_BIT(var->data.location - loc_offset + offset);
}

static void
remap_slots_and_components(nir_shader *shader, nir_variable_mode mode,
                           struct varying_loc (*remap)[4],
                           uint64_t *slots_used, uint64_t *out_slots_read,
                           uint32_t *p_slots_used, uint32_t *p_out_slots_read)
 {
   const gl_shader_stage stage = shader->info.stage;
   uint64_t out_slots_read_tmp[2] = {0};
   uint64_t slots_used_tmp[2] = {0};

   /* We don't touch builtins so just copy the bitmask */
   slots_used_tmp[0] = *slots_used & BITFIELD64_RANGE(0, VARYING_SLOT_VAR0);

   nir_foreach_variable_with_modes(var, shader, mode) {
      assert(var->data.location >= 0);

      /* Only remap things that aren't built-ins */
      if (var->data.location >= VARYING_SLOT_VAR0 &&
          var->data.location - VARYING_SLOT_VAR0 < MAX_VARYINGS_INCL_PATCH) {

         const struct glsl_type *type = var->type;
         if (nir_is_arrayed_io(var, stage) || var->data.per_view) {
            assert(glsl_type_is_array(type));
            type = glsl_get_array_element(type);
         }

         unsigned num_slots = glsl_count_attribute_slots(type, false);
         bool used_across_stages = false;
         bool outputs_read = false;

         unsigned location = var->data.location - VARYING_SLOT_VAR0;
         struct varying_loc *new_loc = &remap[location][var->data.location_frac];

         unsigned loc_offset = var->data.patch ? VARYING_SLOT_PATCH0 : 0;
         uint64_t used = var->data.patch ? *p_slots_used : *slots_used;
         uint64_t outs_used =
            var->data.patch ? *p_out_slots_read : *out_slots_read;
         uint64_t slots =
            BITFIELD64_RANGE(var->data.location - loc_offset, num_slots);

         if (slots & used)
            used_across_stages = true;

         if (slots & outs_used)
            outputs_read = true;

         if (new_loc->location) {
            var->data.location = new_loc->location;
            var->data.location_frac = new_loc->component;
         }

         if (var->data.always_active_io) {
            /* We can't apply link time optimisations (specifically array
             * splitting) to these so we need to copy the existing mask
             * otherwise we will mess up the mask for things like partially
             * marked arrays.
             */
            if (used_across_stages)
               mark_all_used_slots(var, slots_used_tmp, used, num_slots);

            if (outputs_read) {
               mark_all_used_slots(var, out_slots_read_tmp, outs_used,
                                   num_slots);
            }
         } else {
            for (unsigned i = 0; i < num_slots; i++) {
               if (used_across_stages)
                  mark_used_slot(var, slots_used_tmp, i);

               if (outputs_read)
                  mark_used_slot(var, out_slots_read_tmp, i);
            }
         }
      }
   }

   *slots_used = slots_used_tmp[0];
   *out_slots_read = out_slots_read_tmp[0];
   *p_slots_used = slots_used_tmp[1];
   *p_out_slots_read = out_slots_read_tmp[1];
}

struct varying_component {
   nir_variable *var;
   uint8_t interp_type;
   uint8_t interp_loc;
   bool is_32bit;
   bool is_patch;
   bool is_per_primitive;
   bool is_mediump;
   bool is_intra_stage_only;
   bool initialised;
};

static int
cmp_varying_component(const void *comp1_v, const void *comp2_v)
{
   struct varying_component *comp1 = (struct varying_component *) comp1_v;
   struct varying_component *comp2 = (struct varying_component *) comp2_v;

   /* We want patches to be order at the end of the array */
   if (comp1->is_patch != comp2->is_patch)
      return comp1->is_patch ? 1 : -1;

   /* Sort per-primitive outputs after per-vertex ones to allow
    * better compaction when they are mixed in the shader's source.
    */
   if (comp1->is_per_primitive != comp2->is_per_primitive)
      return comp1->is_per_primitive ? 1 : -1;

   /* We want to try to group together TCS outputs that are only read by other
    * TCS invocations and not consumed by the follow stage.
    */
   if (comp1->is_intra_stage_only != comp2->is_intra_stage_only)
      return comp1->is_intra_stage_only ? 1 : -1;

   /* Group mediump varyings together. */
   if (comp1->is_mediump != comp2->is_mediump)
      return comp1->is_mediump ? 1 : -1;

   /* We can only pack varyings with matching interpolation types so group
    * them together.
    */
   if (comp1->interp_type != comp2->interp_type)
      return comp1->interp_type - comp2->interp_type;

   /* Interpolation loc must match also. */
   if (comp1->interp_loc != comp2->interp_loc)
      return comp1->interp_loc - comp2->interp_loc;

   /* If everything else matches just use the original location to sort */
   const struct nir_variable_data *const data1 = &comp1->var->data;
   const struct nir_variable_data *const data2 = &comp2->var->data;
   if (data1->location != data2->location)
      return data1->location - data2->location;
   return (int)data1->location_frac - (int)data2->location_frac;
}

static void
gather_varying_component_info(nir_shader *producer, nir_shader *consumer,
                              struct varying_component **varying_comp_info,
                              unsigned *varying_comp_info_size,
                              bool default_to_smooth_interp)
{
   unsigned store_varying_info_idx[MAX_VARYINGS_INCL_PATCH][4] = {{0}};
   unsigned num_of_comps_to_pack = 0;

   /* Count the number of varying that can be packed and create a mapping
    * of those varyings to the array we will pass to qsort.
    */
   nir_foreach_shader_out_variable(var, producer) {

      /* Only remap things that aren't builtins. */
      if (var->data.location >= VARYING_SLOT_VAR0 &&
          var->data.location - VARYING_SLOT_VAR0 < MAX_VARYINGS_INCL_PATCH) {

         /* We can't repack xfb varyings. */
         if (var->data.always_active_io)
            continue;

         const struct glsl_type *type = var->type;
         if (nir_is_arrayed_io(var, producer->info.stage) || var->data.per_view) {
            assert(glsl_type_is_array(type));
            type = glsl_get_array_element(type);
         }

         if (!is_packing_supported_for_type(type))
            continue;

         unsigned loc = var->data.location - VARYING_SLOT_VAR0;
         store_varying_info_idx[loc][var->data.location_frac] =
            ++num_of_comps_to_pack;
      }
   }

   *varying_comp_info_size = num_of_comps_to_pack;
   *varying_comp_info = rzalloc_array(NULL, struct varying_component,
                                      num_of_comps_to_pack);

   nir_function_impl *impl = nir_shader_get_entrypoint(consumer);

   /* Walk over the shader and populate the varying component info array */
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;

         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_load_deref &&
             intr->intrinsic != nir_intrinsic_interp_deref_at_centroid &&
             intr->intrinsic != nir_intrinsic_interp_deref_at_sample &&
             intr->intrinsic != nir_intrinsic_interp_deref_at_offset &&
             intr->intrinsic != nir_intrinsic_interp_deref_at_vertex)
            continue;

         nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
         if (!nir_deref_mode_is(deref, nir_var_shader_in))
            continue;

         /* We only remap things that aren't builtins. */
         nir_variable *in_var = nir_deref_instr_get_variable(deref);
         if (in_var->data.location < VARYING_SLOT_VAR0)
            continue;

         unsigned location = in_var->data.location - VARYING_SLOT_VAR0;
         if (location >= MAX_VARYINGS_INCL_PATCH)
            continue;

         unsigned var_info_idx =
            store_varying_info_idx[location][in_var->data.location_frac];
         if (!var_info_idx)
            continue;

         struct varying_component *vc_info =
            &(*varying_comp_info)[var_info_idx-1];

         if (!vc_info->initialised) {
            const struct glsl_type *type = in_var->type;
            if (nir_is_arrayed_io(in_var, consumer->info.stage) ||
                in_var->data.per_view) {
               assert(glsl_type_is_array(type));
               type = glsl_get_array_element(type);
            }

            vc_info->var = in_var;
            vc_info->interp_type =
               get_interp_type(in_var, type, default_to_smooth_interp);
            vc_info->interp_loc = get_interp_loc(in_var);
            vc_info->is_32bit = glsl_type_is_32bit(type);
            vc_info->is_patch = in_var->data.patch;
            vc_info->is_per_primitive = in_var->data.per_primitive;
            vc_info->is_mediump = !producer->options->linker_ignore_precision &&
               (in_var->data.precision == GLSL_PRECISION_MEDIUM ||
                in_var->data.precision == GLSL_PRECISION_LOW);
            vc_info->is_intra_stage_only = false;
            vc_info->initialised = true;
         }
      }
   }

   /* Walk over the shader and populate the varying component info array
    * for varyings which are read by other TCS instances but are not consumed
    * by the TES.
    */
   if (producer->info.stage == MESA_SHADER_TESS_CTRL) {
      impl = nir_shader_get_entrypoint(producer);

      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_load_deref)
               continue;

            nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
            if (!nir_deref_mode_is(deref, nir_var_shader_out))
               continue;

            /* We only remap things that aren't builtins. */
            nir_variable *out_var = nir_deref_instr_get_variable(deref);
            if (out_var->data.location < VARYING_SLOT_VAR0)
               continue;

            unsigned location = out_var->data.location - VARYING_SLOT_VAR0;
            if (location >= MAX_VARYINGS_INCL_PATCH)
               continue;

            unsigned var_info_idx =
               store_varying_info_idx[location][out_var->data.location_frac];
            if (!var_info_idx) {
               /* Something went wrong, the shader interfaces didn't match, so
                * abandon packing. This can happen for example when the
                * inputs are scalars but the outputs are struct members.
                */
               *varying_comp_info_size = 0;
               break;
            }

            struct varying_component *vc_info =
               &(*varying_comp_info)[var_info_idx-1];

            if (!vc_info->initialised) {
               const struct glsl_type *type = out_var->type;
               if (nir_is_arrayed_io(out_var, producer->info.stage)) {
                  assert(glsl_type_is_array(type));
                  type = glsl_get_array_element(type);
               }

               vc_info->var = out_var;
               vc_info->interp_type =
                  get_interp_type(out_var, type, default_to_smooth_interp);
               vc_info->interp_loc = get_interp_loc(out_var);
               vc_info->is_32bit = glsl_type_is_32bit(type);
               vc_info->is_patch = out_var->data.patch;
               vc_info->is_per_primitive = out_var->data.per_primitive;
               vc_info->is_mediump = !producer->options->linker_ignore_precision &&
                  (out_var->data.precision == GLSL_PRECISION_MEDIUM ||
                   out_var->data.precision == GLSL_PRECISION_LOW);
               vc_info->is_intra_stage_only = true;
               vc_info->initialised = true;
            }
         }
      }
   }

   for (unsigned i = 0; i < *varying_comp_info_size; i++ ) {
      struct varying_component *vc_info = &(*varying_comp_info)[i];
      if (!vc_info->initialised) {
         /* Something went wrong, the shader interfaces didn't match, so
          * abandon packing. This can happen for example when the outputs are
          * scalars but the inputs are struct members.
          */
         *varying_comp_info_size = 0;
         break;
      }
   }
}

static bool
allow_pack_interp_type(nir_pack_varying_options options, int type)
{
   int sel;

   switch (type) {
   case INTERP_MODE_NONE:
      sel = nir_pack_varying_interp_mode_none;
      break;
   case INTERP_MODE_SMOOTH:
      sel = nir_pack_varying_interp_mode_smooth;
      break;
   case INTERP_MODE_FLAT:
      sel = nir_pack_varying_interp_mode_flat;
      break;
   case INTERP_MODE_NOPERSPECTIVE:
      sel = nir_pack_varying_interp_mode_noperspective;
      break;
   default:
      return false;
   }

   return options & sel;
}

static bool
allow_pack_interp_loc(nir_pack_varying_options options, int loc)
{
   int sel;

   switch (loc) {
   case INTERPOLATE_LOC_SAMPLE:
      sel = nir_pack_varying_interp_loc_sample;
      break;
   case INTERPOLATE_LOC_CENTROID:
      sel = nir_pack_varying_interp_loc_centroid;
      break;
   case INTERPOLATE_LOC_CENTER:
      sel = nir_pack_varying_interp_loc_center;
      break;
   default:
      return false;
   }

   return options & sel;
}

static void
assign_remap_locations(struct varying_loc (*remap)[4],
                       struct assigned_comps *assigned_comps,
                       struct varying_component *info,
                       unsigned *cursor, unsigned *comp,
                       unsigned max_location,
                       nir_pack_varying_options options)
{
   unsigned tmp_cursor = *cursor;
   unsigned tmp_comp = *comp;

   for (; tmp_cursor < max_location; tmp_cursor++) {

      if (assigned_comps[tmp_cursor].comps) {
         /* Don't pack per-primitive and per-vertex varyings together. */
         if (assigned_comps[tmp_cursor].is_per_primitive != info->is_per_primitive) {
            tmp_comp = 0;
            continue;
         }

         /* We can only pack varyings with matching precision. */
         if (assigned_comps[tmp_cursor].is_mediump != info->is_mediump) {
            tmp_comp = 0;
            continue;
         }

         /* We can only pack varyings with matching interpolation type
          * if driver does not support it.
          */
         if (assigned_comps[tmp_cursor].interp_type != info->interp_type &&
             (!allow_pack_interp_type(options, assigned_comps[tmp_cursor].interp_type) ||
              !allow_pack_interp_type(options, info->interp_type))) {
            tmp_comp = 0;
            continue;
         }

         /* We can only pack varyings with matching interpolation location
          * if driver does not support it.
          */
         if (assigned_comps[tmp_cursor].interp_loc != info->interp_loc &&
             (!allow_pack_interp_loc(options, assigned_comps[tmp_cursor].interp_loc) ||
              !allow_pack_interp_loc(options, info->interp_loc))) {
            tmp_comp = 0;
            continue;
         }

         /* We can only pack varyings with matching types, and the current
          * algorithm only supports packing 32-bit.
          */
         if (!assigned_comps[tmp_cursor].is_32bit) {
            tmp_comp = 0;
            continue;
         }

         while (tmp_comp < 4 &&
                (assigned_comps[tmp_cursor].comps & (1 << tmp_comp))) {
            tmp_comp++;
         }
      }

      if (tmp_comp == 4) {
         tmp_comp = 0;
         continue;
      }

      unsigned location = info->var->data.location - VARYING_SLOT_VAR0;

      /* Once we have assigned a location mark it as used */
      assigned_comps[tmp_cursor].comps |= (1 << tmp_comp);
      assigned_comps[tmp_cursor].interp_type = info->interp_type;
      assigned_comps[tmp_cursor].interp_loc = info->interp_loc;
      assigned_comps[tmp_cursor].is_32bit = info->is_32bit;
      assigned_comps[tmp_cursor].is_mediump = info->is_mediump;
      assigned_comps[tmp_cursor].is_per_primitive = info->is_per_primitive;

      /* Assign remap location */
      remap[location][info->var->data.location_frac].component = tmp_comp++;
      remap[location][info->var->data.location_frac].location =
         tmp_cursor + VARYING_SLOT_VAR0;

      break;
   }

   *cursor = tmp_cursor;
   *comp = tmp_comp;
}

/* If there are empty components in the slot compact the remaining components
 * as close to component 0 as possible. This will make it easier to fill the
 * empty components with components from a different slot in a following pass.
 */
static void
compact_components(nir_shader *producer, nir_shader *consumer,
                   struct assigned_comps *assigned_comps,
                   bool default_to_smooth_interp)
{
   struct varying_loc remap[MAX_VARYINGS_INCL_PATCH][4] = {{{0}, {0}}};
   struct varying_component *varying_comp_info;
   unsigned varying_comp_info_size;

   /* Gather varying component info */
   gather_varying_component_info(producer, consumer, &varying_comp_info,
                                 &varying_comp_info_size,
                                 default_to_smooth_interp);

   /* Sort varying components. */
   qsort(varying_comp_info, varying_comp_info_size,
         sizeof(struct varying_component), cmp_varying_component);

   nir_pack_varying_options options = consumer->options->pack_varying_options;

   unsigned cursor = 0;
   unsigned comp = 0;

   /* Set the remap array based on the sorted components */
   for (unsigned i = 0; i < varying_comp_info_size; i++ ) {
      struct varying_component *info = &varying_comp_info[i];

      assert(info->is_patch || cursor < MAX_VARYING);
      if (info->is_patch) {
         /* The list should be sorted with all non-patch inputs first followed
          * by patch inputs.  When we hit our first patch input, we need to
          * reset the cursor to MAX_VARYING so we put them in the right slot.
          */
         if (cursor < MAX_VARYING) {
            cursor = MAX_VARYING;
            comp = 0;
         }

         assign_remap_locations(remap, assigned_comps, info,
                                &cursor, &comp, MAX_VARYINGS_INCL_PATCH,
                                options);
      } else {
         assign_remap_locations(remap, assigned_comps, info,
                                &cursor, &comp, MAX_VARYING,
                                options);

         /* Check if we failed to assign a remap location. This can happen if
          * for example there are a bunch of unmovable components with
          * mismatching interpolation types causing us to skip over locations
          * that would have been useful for packing later components.
          * The solution is to iterate over the locations again (this should
          * happen very rarely in practice).
          */
         if (cursor == MAX_VARYING) {
            cursor = 0;
            comp = 0;
            assign_remap_locations(remap, assigned_comps, info,
                                   &cursor, &comp, MAX_VARYING,
                                   options);
         }
      }
   }

   ralloc_free(varying_comp_info);

   uint64_t zero = 0;
   uint32_t zero32 = 0;
   remap_slots_and_components(consumer, nir_var_shader_in, remap,
                              &consumer->info.inputs_read, &zero,
                              &consumer->info.patch_inputs_read, &zero32);
   remap_slots_and_components(producer, nir_var_shader_out, remap,
                              &producer->info.outputs_written,
                              &producer->info.outputs_read,
                              &producer->info.patch_outputs_written,
                              &producer->info.patch_outputs_read);
}

/* We assume that this has been called more-or-less directly after
 * remove_unused_varyings.  At this point, all of the varyings that we
 * aren't going to be using have been completely removed and the
 * inputs_read and outputs_written fields in nir_shader_info reflect
 * this.  Therefore, the total set of valid slots is the OR of the two
 * sets of varyings;  this accounts for varyings which one side may need
 * to read/write even if the other doesn't.  This can happen if, for
 * instance, an array is used indirectly from one side causing it to be
 * unsplittable but directly from the other.
 */
void
nir_compact_varyings(nir_shader *producer, nir_shader *consumer,
                     bool default_to_smooth_interp)
{
   assert(producer->info.stage != MESA_SHADER_FRAGMENT);
   assert(consumer->info.stage != MESA_SHADER_VERTEX);

   struct assigned_comps assigned_comps[MAX_VARYINGS_INCL_PATCH] = {{0}};

   get_unmoveable_components_masks(producer, nir_var_shader_out,
                                   assigned_comps,
                                   producer->info.stage,
                                   default_to_smooth_interp);
   get_unmoveable_components_masks(consumer, nir_var_shader_in,
                                   assigned_comps,
                                   consumer->info.stage,
                                   default_to_smooth_interp);

   compact_components(producer, consumer, assigned_comps,
                      default_to_smooth_interp);
}

/*
 * Mark XFB varyings as always_active_io in the consumer so the linking opts
 * don't touch them.
 */
void
nir_link_xfb_varyings(nir_shader *producer, nir_shader *consumer)
{
   nir_variable *input_vars[MAX_VARYING][4] = { 0 };

   nir_foreach_shader_in_variable(var, consumer) {
      if (var->data.location >= VARYING_SLOT_VAR0 &&
          var->data.location - VARYING_SLOT_VAR0 < MAX_VARYING) {

         unsigned location = var->data.location - VARYING_SLOT_VAR0;
         input_vars[location][var->data.location_frac] = var;
      }
   }

   nir_foreach_shader_out_variable(var, producer) {
      if (var->data.location >= VARYING_SLOT_VAR0 &&
          var->data.location - VARYING_SLOT_VAR0 < MAX_VARYING) {

         if (!var->data.always_active_io)
            continue;

         unsigned location = var->data.location - VARYING_SLOT_VAR0;
         if (input_vars[location][var->data.location_frac]) {
            input_vars[location][var->data.location_frac]->data.always_active_io = true;
         }
      }
   }
}

static bool
does_varying_match(nir_variable *out_var, nir_variable *in_var)
{
   return in_var->data.location == out_var->data.location &&
          in_var->data.location_frac == out_var->data.location_frac;
}

static nir_variable *
get_matching_input_var(nir_shader *consumer, nir_variable *out_var)
{
   nir_foreach_shader_in_variable(var, consumer) {
      if (does_varying_match(out_var, var))
         return var;
   }

   return NULL;
}

static bool
can_replace_varying(nir_variable *out_var)
{
   /* Skip types that require more complex handling.
    * TODO: add support for these types.
    */
   if (glsl_type_is_array(out_var->type) ||
       glsl_type_is_dual_slot(out_var->type) ||
       glsl_type_is_matrix(out_var->type) ||
       glsl_type_is_struct_or_ifc(out_var->type))
      return false;

   /* Limit this pass to scalars for now to keep things simple. Most varyings
    * should have been lowered to scalars at this point anyway.
    */
   if (!glsl_type_is_scalar(out_var->type))
      return false;

   if (out_var->data.location < VARYING_SLOT_VAR0 ||
       out_var->data.location - VARYING_SLOT_VAR0 >= MAX_VARYING)
      return false;

   return true;
}

static bool
replace_varying_input_by_constant_load(nir_shader *shader,
                                       nir_intrinsic_instr *store_intr)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(shader);

   nir_builder b;
   nir_builder_init(&b, impl);

   nir_variable *out_var =
      nir_deref_instr_get_variable(nir_src_as_deref(store_intr->src[0]));

   bool progress = false;
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;

         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_load_deref)
            continue;

         nir_deref_instr *in_deref = nir_src_as_deref(intr->src[0]);
         if (!nir_deref_mode_is(in_deref, nir_var_shader_in))
            continue;

         nir_variable *in_var = nir_deref_instr_get_variable(in_deref);

         if (!does_varying_match(out_var, in_var))
            continue;

         b.cursor = nir_before_instr(instr);

         nir_load_const_instr *out_const =
            nir_instr_as_load_const(store_intr->src[1].ssa->parent_instr);

         /* Add new const to replace the input */
         nir_ssa_def *nconst = nir_build_imm(&b, store_intr->num_components,
                                             intr->dest.ssa.bit_size,
                                             out_const->value);

         nir_ssa_def_rewrite_uses(&intr->dest.ssa, nconst);

         progress = true;
      }
   }

   return progress;
}

static bool
replace_duplicate_input(nir_shader *shader, nir_variable *input_var,
                         nir_intrinsic_instr *dup_store_intr)
{
   assert(input_var);

   nir_function_impl *impl = nir_shader_get_entrypoint(shader);

   nir_builder b;
   nir_builder_init(&b, impl);

   nir_variable *dup_out_var =
      nir_deref_instr_get_variable(nir_src_as_deref(dup_store_intr->src[0]));

   bool progress = false;
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;

         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_load_deref)
            continue;

         nir_deref_instr *in_deref = nir_src_as_deref(intr->src[0]);
         if (!nir_deref_mode_is(in_deref, nir_var_shader_in))
            continue;

         nir_variable *in_var = nir_deref_instr_get_variable(in_deref);

         if (!does_varying_match(dup_out_var, in_var) ||
             in_var->data.interpolation != input_var->data.interpolation ||
             get_interp_loc(in_var) != get_interp_loc(input_var))
            continue;

         b.cursor = nir_before_instr(instr);

         nir_ssa_def *load = nir_load_var(&b, input_var);
         nir_ssa_def_rewrite_uses(&intr->dest.ssa, load);

         progress = true;
      }
   }

   return progress;
}

static bool
is_direct_uniform_load(nir_ssa_def *def, nir_ssa_scalar *s)
{
   /* def is sure to be scalar as can_replace_varying() filter out vector case. */
   assert(def->num_components == 1);

   /* Uniform load may hide behind some move instruction for converting
    * vector to scalar:
    *
    *     vec1 32 ssa_1 = deref_var &color (uniform vec3)
    *     vec3 32 ssa_2 = intrinsic load_deref (ssa_1) (0)
    *     vec1 32 ssa_3 = mov ssa_2.x
    *     vec1 32 ssa_4 = deref_var &color_out (shader_out float)
    *     intrinsic store_deref (ssa_4, ssa_3) (1, 0)
    */
   *s = nir_ssa_scalar_resolved(def, 0);

   nir_ssa_def *ssa = s->def;
   if (ssa->parent_instr->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(ssa->parent_instr);
   if (intr->intrinsic != nir_intrinsic_load_deref)
      return false;

   nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
   /* TODO: support nir_var_mem_ubo. */
   if (!nir_deref_mode_is(deref, nir_var_uniform))
      return false;

   /* Does not support indirect uniform load. */
   return !nir_deref_instr_has_indirect(deref);
}

static nir_variable *
get_uniform_var_in_consumer(nir_shader *consumer,
                            nir_variable *var_in_producer)
{
   /* Find if uniform already exists in consumer. */
   nir_variable *new_var = NULL;
   nir_foreach_uniform_variable(v, consumer) {
      if (!strcmp(var_in_producer->name, v->name)) {
         new_var = v;
         break;
      }
   }

   /* Create a variable if not exist. */
   if (!new_var) {
      new_var = nir_variable_clone(var_in_producer, consumer);
      nir_shader_add_variable(consumer, new_var);
   }

   return new_var;
}

static nir_deref_instr *
clone_deref_instr(nir_builder *b, nir_variable *var, nir_deref_instr *deref)
{
   if (deref->deref_type == nir_deref_type_var)
       return nir_build_deref_var(b, var);

   nir_deref_instr *parent_deref = nir_deref_instr_parent(deref);
   nir_deref_instr *parent = clone_deref_instr(b, var, parent_deref);

   /* Build array and struct deref instruction.
    * "deref" instr is sure to be direct (see is_direct_uniform_load()).
    */
   switch (deref->deref_type) {
   case nir_deref_type_array: {
      nir_load_const_instr *index =
         nir_instr_as_load_const(deref->arr.index.ssa->parent_instr);
      return nir_build_deref_array_imm(b, parent, index->value->i64);
   }
   case nir_deref_type_ptr_as_array: {
      nir_load_const_instr *index =
         nir_instr_as_load_const(deref->arr.index.ssa->parent_instr);
      nir_ssa_def *ssa = nir_imm_intN_t(b, index->value->i64,
                                        parent->dest.ssa.bit_size);
      return nir_build_deref_ptr_as_array(b, parent, ssa);
   }
   case nir_deref_type_struct:
      return nir_build_deref_struct(b, parent, deref->strct.index);
   default:
      unreachable("invalid type");
      return NULL;
   }
}

static bool
replace_varying_input_by_uniform_load(nir_shader *shader,
                                      nir_intrinsic_instr *store_intr,
                                      nir_ssa_scalar *scalar)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(shader);

   nir_builder b;
   nir_builder_init(&b, impl);

   nir_variable *out_var =
      nir_deref_instr_get_variable(nir_src_as_deref(store_intr->src[0]));

   nir_intrinsic_instr *load = nir_instr_as_intrinsic(scalar->def->parent_instr);
   nir_deref_instr *deref = nir_src_as_deref(load->src[0]);
   nir_variable *uni_var = nir_deref_instr_get_variable(deref);
   uni_var = get_uniform_var_in_consumer(shader, uni_var);

   bool progress = false;
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;

         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_load_deref)
            continue;

         nir_deref_instr *in_deref = nir_src_as_deref(intr->src[0]);
         if (!nir_deref_mode_is(in_deref, nir_var_shader_in))
            continue;

         nir_variable *in_var = nir_deref_instr_get_variable(in_deref);

         if (!does_varying_match(out_var, in_var))
            continue;

         b.cursor = nir_before_instr(instr);

         /* Clone instructions start from deref load to variable deref. */
         nir_deref_instr *uni_deref = clone_deref_instr(&b, uni_var, deref);
         nir_ssa_def *uni_def = nir_load_deref(&b, uni_deref);

         /* Add a vector to scalar move if uniform is a vector. */
         if (uni_def->num_components > 1) {
            nir_alu_src src = {0};
            src.src = nir_src_for_ssa(uni_def);
            src.swizzle[0] = scalar->comp;
            uni_def = nir_mov_alu(&b, src, 1);
         }

         /* Replace load input with load uniform. */
         nir_ssa_def_rewrite_uses(&intr->dest.ssa, uni_def);

         progress = true;
      }
   }

   return progress;
}

/* The GLSL ES 3.20 spec says:
 *
 * "The precision of a vertex output does not need to match the precision of
 * the corresponding fragment input. The minimum precision at which vertex
 * outputs are interpolated is the minimum of the vertex output precision and
 * the fragment input precision, with the exception that for highp,
 * implementations do not have to support full IEEE 754 precision." (9.1 "Input
 * Output Matching by Name in Linked Programs")
 *
 * To implement this, when linking shaders we will take the minimum precision
 * qualifier (allowing drivers to interpolate at lower precision). For
 * input/output between non-fragment stages (e.g. VERTEX to GEOMETRY), the spec
 * requires we use the *last* specified precision if there is a conflict.
 *
 * Precisions are ordered as (NONE, HIGH, MEDIUM, LOW). If either precision is
 * NONE, we'll return the other precision, since there is no conflict.
 * Otherwise for fragment interpolation, we'll pick the smallest of (HIGH,
 * MEDIUM, LOW) by picking the maximum of the raw values - note the ordering is
 * "backwards". For non-fragment stages, we'll pick the latter precision to
 * comply with the spec. (Note that the order matters.)
 *
 * For streamout, "Variables declared with lowp or mediump precision are
 * promoted to highp before being written." (12.2 "Transform Feedback", p. 341
 * of OpenGL ES 3.2 specification). So drivers should promote them
 * the transform feedback memory store, but not the output store.
 */

static unsigned
nir_link_precision(unsigned producer, unsigned consumer, bool fs)
{
   if (producer == GLSL_PRECISION_NONE)
      return consumer;
   else if (consumer == GLSL_PRECISION_NONE)
      return producer;
   else
      return fs ? MAX2(producer, consumer) : consumer;
}

static nir_variable *
find_consumer_variable(const nir_shader *consumer,
                       const nir_variable *producer_var)
{
   nir_foreach_variable_with_modes(var, consumer, nir_var_shader_in) {
      if (var->data.location == producer_var->data.location &&
          var->data.location_frac == producer_var->data.location_frac)
         return var;
   }
   return NULL;
}

void
nir_link_varying_precision(nir_shader *producer, nir_shader *consumer)
{
   bool frag = consumer->info.stage == MESA_SHADER_FRAGMENT;

   nir_foreach_shader_out_variable(producer_var, producer) {
      /* Skip if the slot is not assigned */
      if (producer_var->data.location < 0)
         continue;

      nir_variable *consumer_var = find_consumer_variable(consumer,
            producer_var);

      /* Skip if the variable will be eliminated */
      if (!consumer_var)
         continue;

      /* Now we have a pair of variables. Let's pick the smaller precision. */
      unsigned precision_1 = producer_var->data.precision;
      unsigned precision_2 = consumer_var->data.precision;
      unsigned minimum = nir_link_precision(precision_1, precision_2, frag);

      /* Propagate the new precision */
      producer_var->data.precision = consumer_var->data.precision = minimum;
   }
}

bool
nir_link_opt_varyings(nir_shader *producer, nir_shader *consumer)
{
   /* TODO: Add support for more shader stage combinations */
   if (consumer->info.stage != MESA_SHADER_FRAGMENT ||
       (producer->info.stage != MESA_SHADER_VERTEX &&
        producer->info.stage != MESA_SHADER_TESS_EVAL))
      return false;

   bool progress = false;

   nir_function_impl *impl = nir_shader_get_entrypoint(producer);

   struct hash_table *varying_values = _mesa_pointer_hash_table_create(NULL);

   /* If we find a store in the last block of the producer we can be sure this
    * is the only possible value for this output.
    */
   nir_block *last_block = nir_impl_last_block(impl);
   nir_foreach_instr_reverse(instr, last_block) {
      if (instr->type != nir_instr_type_intrinsic)
         continue;

      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);

      if (intr->intrinsic != nir_intrinsic_store_deref)
         continue;

      nir_deref_instr *out_deref = nir_src_as_deref(intr->src[0]);
      if (!nir_deref_mode_is(out_deref, nir_var_shader_out))
         continue;

      nir_variable *out_var = nir_deref_instr_get_variable(out_deref);
      if (!can_replace_varying(out_var))
         continue;

      nir_ssa_def *ssa = intr->src[1].ssa;
      if (ssa->parent_instr->type == nir_instr_type_load_const) {
         progress |= replace_varying_input_by_constant_load(consumer, intr);
         continue;
      }

      nir_ssa_scalar uni_scalar;
      if (is_direct_uniform_load(ssa, &uni_scalar)) {
         if (consumer->options->lower_varying_from_uniform) {
            progress |= replace_varying_input_by_uniform_load(consumer, intr,
                                                              &uni_scalar);
            continue;
         } else {
            nir_variable *in_var = get_matching_input_var(consumer, out_var);
            /* The varying is loaded from same uniform, so no need to do any
             * interpolation. Mark it as flat explicitly.
             */
            if (!consumer->options->no_integers &&
                in_var && in_var->data.interpolation <= INTERP_MODE_NOPERSPECTIVE) {
               in_var->data.interpolation = INTERP_MODE_FLAT;
               out_var->data.interpolation = INTERP_MODE_FLAT;
            }
         }
      }

      struct hash_entry *entry = _mesa_hash_table_search(varying_values, ssa);
      if (entry) {
         progress |= replace_duplicate_input(consumer,
                                             (nir_variable *) entry->data,
                                             intr);
      } else {
         nir_variable *in_var = get_matching_input_var(consumer, out_var);
         if (in_var) {
            _mesa_hash_table_insert(varying_values, ssa, in_var);
         }
      }
   }

   _mesa_hash_table_destroy(varying_values, NULL);

   return progress;
}

/* TODO any better helper somewhere to sort a list? */

static void
insert_sorted(struct exec_list *var_list, nir_variable *new_var)
{
   nir_foreach_variable_in_list(var, var_list) {
      /* Use the `per_primitive` bool to sort per-primitive variables
       * to the end of the list, so they get the last driver locations
       * by nir_assign_io_var_locations.
       *
       * This is done because AMD HW requires that per-primitive outputs
       * are the last params.
       * In the future we can add an option for this, if needed by other HW.
       */
      if (new_var->data.per_primitive < var->data.per_primitive ||
          (new_var->data.per_primitive == var->data.per_primitive &&
           (var->data.location > new_var->data.location ||
            (var->data.location == new_var->data.location &&
             var->data.location_frac > new_var->data.location_frac)))) {
         exec_node_insert_node_before(&var->node, &new_var->node);
         return;
      }
   }
   exec_list_push_tail(var_list, &new_var->node);
}

static void
sort_varyings(nir_shader *shader, nir_variable_mode mode,
              struct exec_list *sorted_list)
{
   exec_list_make_empty(sorted_list);
   nir_foreach_variable_with_modes_safe(var, shader, mode) {
      exec_node_remove(&var->node);
      insert_sorted(sorted_list, var);
   }
}

void
nir_assign_io_var_locations(nir_shader *shader, nir_variable_mode mode,
                            unsigned *size, gl_shader_stage stage)
{
   unsigned location = 0;
   unsigned assigned_locations[VARYING_SLOT_TESS_MAX];
   uint64_t processed_locs[2] = {0};

   struct exec_list io_vars;
   sort_varyings(shader, mode, &io_vars);

   int ASSERTED last_loc = 0;
   bool ASSERTED last_per_prim = false;
   bool last_partial = false;
   nir_foreach_variable_in_list(var, &io_vars) {
      const struct glsl_type *type = var->type;
      if (nir_is_arrayed_io(var, stage)) {
         assert(glsl_type_is_array(type));
         type = glsl_get_array_element(type);
      }

      int base;
      if (var->data.mode == nir_var_shader_in && stage == MESA_SHADER_VERTEX)
         base = VERT_ATTRIB_GENERIC0;
      else if (var->data.mode == nir_var_shader_out &&
               stage == MESA_SHADER_FRAGMENT)
         base = FRAG_RESULT_DATA0;
      else
         base = VARYING_SLOT_VAR0;

      unsigned var_size, driver_size;
      if (var->data.compact) {
         /* If we are inside a partial compact,
          * don't allow another compact to be in this slot
          * if it starts at component 0.
          */
         if (last_partial && var->data.location_frac == 0) {
            location++;
         }

         /* compact variables must be arrays of scalars */
         assert(!var->data.per_view);
         assert(glsl_type_is_array(type));
         assert(glsl_type_is_scalar(glsl_get_array_element(type)));
         unsigned start = 4 * location + var->data.location_frac;
         unsigned end = start + glsl_get_length(type);
         var_size = driver_size = end / 4 - location;
         last_partial = end % 4 != 0;
      } else {
         /* Compact variables bypass the normal varying compacting pass,
          * which means they cannot be in the same vec4 slot as a normal
          * variable. If part of the current slot is taken up by a compact
          * variable, we need to go to the next one.
          */
         if (last_partial) {
            location++;
            last_partial = false;
         }

         /* per-view variables have an extra array dimension, which is ignored
          * when counting user-facing slots (var->data.location), but *not*
          * with driver slots (var->data.driver_location). That is, each user
          * slot maps to multiple driver slots.
          */
         driver_size = glsl_count_attribute_slots(type, false);
         if (var->data.per_view) {
            assert(glsl_type_is_array(type));
            var_size =
               glsl_count_attribute_slots(glsl_get_array_element(type), false);
         } else {
            var_size = driver_size;
         }
      }

      /* Builtins don't allow component packing so we only need to worry about
       * user defined varyings sharing the same location.
       */
      bool processed = false;
      if (var->data.location >= base) {
         unsigned glsl_location = var->data.location - base;

         for (unsigned i = 0; i < var_size; i++) {
            if (processed_locs[var->data.index] &
                ((uint64_t)1 << (glsl_location + i)))
               processed = true;
            else
               processed_locs[var->data.index] |=
                  ((uint64_t)1 << (glsl_location + i));
         }
      }

      /* Because component packing allows varyings to share the same location
       * we may have already have processed this location.
       */
      if (processed) {
         /* TODO handle overlapping per-view variables */
         assert(!var->data.per_view);
         unsigned driver_location = assigned_locations[var->data.location];
         var->data.driver_location = driver_location;

         /* An array may be packed such that is crosses multiple other arrays
          * or variables, we need to make sure we have allocated the elements
          * consecutively if the previously proccessed var was shorter than
          * the current array we are processing.
          *
          * NOTE: The code below assumes the var list is ordered in ascending
          * location order, but per-vertex/per-primitive outputs may be
          * grouped separately.
          */
         assert(last_loc <= var->data.location ||
                last_per_prim != var->data.per_primitive);
         last_loc = var->data.location;
         last_per_prim = var->data.per_primitive;
         unsigned last_slot_location = driver_location + var_size;
         if (last_slot_location > location) {
            unsigned num_unallocated_slots = last_slot_location - location;
            unsigned first_unallocated_slot = var_size - num_unallocated_slots;
            for (unsigned i = first_unallocated_slot; i < var_size; i++) {
               assigned_locations[var->data.location + i] = location;
               location++;
            }
         }
         continue;
      }

      for (unsigned i = 0; i < var_size; i++) {
         assigned_locations[var->data.location + i] = location + i;
      }

      var->data.driver_location = location;
      location += driver_size;
   }

   if (last_partial)
      location++;

   exec_list_append(&shader->variables, &io_vars);
   *size = location;
}

static uint64_t
get_linked_variable_location(unsigned location, bool patch)
{
   if (!patch)
      return location;

   /* Reserve locations 0...3 for special patch variables
    * like tess factors and bounding boxes, and the generic patch
    * variables will come after them.
    */
   if (location >= VARYING_SLOT_PATCH0)
      return location - VARYING_SLOT_PATCH0 + 4;
   else if (location >= VARYING_SLOT_TESS_LEVEL_OUTER &&
            location <= VARYING_SLOT_BOUNDING_BOX1)
      return location - VARYING_SLOT_TESS_LEVEL_OUTER;
   else
      unreachable("Unsupported variable in get_linked_variable_location.");
}

static uint64_t
get_linked_variable_io_mask(nir_variable *variable, gl_shader_stage stage)
{
   const struct glsl_type *type = variable->type;

   if (nir_is_arrayed_io(variable, stage)) {
      assert(glsl_type_is_array(type));
      type = glsl_get_array_element(type);
   }

   unsigned slots = glsl_count_attribute_slots(type, false);
   if (variable->data.compact) {
      unsigned component_count = variable->data.location_frac + glsl_get_length(type);
      slots = DIV_ROUND_UP(component_count, 4);
   }

   uint64_t mask = u_bit_consecutive64(0, slots);
   return mask;
}

nir_linked_io_var_info
nir_assign_linked_io_var_locations(nir_shader *producer, nir_shader *consumer)
{
   assert(producer);
   assert(consumer);

   uint64_t producer_output_mask = 0;
   uint64_t producer_patch_output_mask = 0;

   nir_foreach_shader_out_variable(variable, producer) {
      uint64_t mask = get_linked_variable_io_mask(variable, producer->info.stage);
      uint64_t loc = get_linked_variable_location(variable->data.location, variable->data.patch);

      if (variable->data.patch)
         producer_patch_output_mask |= mask << loc;
      else
         producer_output_mask |= mask << loc;
   }

   uint64_t consumer_input_mask = 0;
   uint64_t consumer_patch_input_mask = 0;

   nir_foreach_shader_in_variable(variable, consumer) {
      uint64_t mask = get_linked_variable_io_mask(variable, consumer->info.stage);
      uint64_t loc = get_linked_variable_location(variable->data.location, variable->data.patch);

      if (variable->data.patch)
         consumer_patch_input_mask |= mask << loc;
      else
         consumer_input_mask |= mask << loc;
   }

   uint64_t io_mask = producer_output_mask | consumer_input_mask;
   uint64_t patch_io_mask = producer_patch_output_mask | consumer_patch_input_mask;

   nir_foreach_shader_out_variable(variable, producer) {
      uint64_t loc = get_linked_variable_location(variable->data.location, variable->data.patch);

      if (variable->data.patch)
         variable->data.driver_location = util_bitcount64(patch_io_mask & u_bit_consecutive64(0, loc));
      else
         variable->data.driver_location = util_bitcount64(io_mask & u_bit_consecutive64(0, loc));
   }

   nir_foreach_shader_in_variable(variable, consumer) {
      uint64_t loc = get_linked_variable_location(variable->data.location, variable->data.patch);

      if (variable->data.patch)
         variable->data.driver_location = util_bitcount64(patch_io_mask & u_bit_consecutive64(0, loc));
      else
         variable->data.driver_location = util_bitcount64(io_mask & u_bit_consecutive64(0, loc));
   }

   nir_linked_io_var_info result = {
      .num_linked_io_vars = util_bitcount64(io_mask),
      .num_linked_patch_io_vars = util_bitcount64(patch_io_mask),
   };

   return result;
}
