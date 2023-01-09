/* SPDX-License-Identifier: MIT */

#include "nir.h"
#include "nir_builder.h"

static bool
var_is_inline_sampler(const nir_variable *var)
{
   if (var->data.mode != nir_var_uniform)
      return false;

   return glsl_type_is_sampler(var->type) &&
          var->data.sampler.is_inline_sampler;
}

static bool
inline_sampler_vars_equal(const nir_variable *a, const nir_variable *b)
{
   assert(var_is_inline_sampler(a) && var_is_inline_sampler(b));

   if (a == b)
      return true;

   return a->data.sampler.addressing_mode == b->data.sampler.addressing_mode &&
          a->data.sampler.normalized_coordinates == b->data.sampler.normalized_coordinates &&
          a->data.sampler.filter_mode == b->data.sampler.filter_mode;
}

static nir_variable *
find_identical_inline_sampler(nir_shader *nir,
                              struct exec_list *inline_samplers,
                              nir_variable *sampler)
{
   nir_foreach_variable_in_list(var, inline_samplers) {
      if (inline_sampler_vars_equal(var, sampler))
         return var;
   }

   nir_foreach_uniform_variable(var, nir) {
      if (!var_is_inline_sampler(var) ||
          !inline_sampler_vars_equal(var, sampler))
         continue;

      exec_node_remove(&var->node);
      exec_list_push_tail(inline_samplers, &var->node);
      return var;
   }
   unreachable("Should have at least found the input sampler");
}

static bool
nir_dedup_inline_samplers_instr(nir_builder *b,
                                nir_instr *instr,
                                void *cb_data)
{
   struct exec_list *inline_samplers = cb_data;

   if (instr->type != nir_instr_type_deref)
      return false;

   nir_deref_instr *deref = nir_instr_as_deref(instr);
   if (deref->deref_type != nir_deref_type_var)
      return false;

   nir_variable *sampler = nir_deref_instr_get_variable(deref);
   if (!var_is_inline_sampler(sampler))
      return false;

   nir_variable *replacement =
      find_identical_inline_sampler(b->shader, inline_samplers, sampler);
   deref->var = replacement;
   return true;
}

/** De-duplicates inline sampler variables
 *
 * Any dead or redundant inline sampler variables are removed any live inline
 * sampler variables are placed at the end of the variables list.
 */
bool
nir_dedup_inline_samplers(nir_shader *nir)
{
   struct exec_list inline_samplers;
   exec_list_make_empty(&inline_samplers);

   nir_shader_instructions_pass(nir, nir_dedup_inline_samplers_instr,
                                nir_metadata_block_index |
                                nir_metadata_dominance,
                                &inline_samplers);

   /* If we found any inline samplers in the instructions pass, they'll now be
    * in the inline_samplers list.
    */
   bool progress = !exec_list_is_empty(&inline_samplers);

   /* Remove any dead samplers */
   nir_foreach_uniform_variable_safe(var, nir) {
      if (var_is_inline_sampler(var)) {
         exec_node_remove(&var->node);
         progress = true;
      }
   }

   exec_node_insert_list_after(exec_list_get_tail(&nir->variables),
                               &inline_samplers);

   return progress;
}

bool
nir_lower_cl_images(nir_shader *shader, bool lower_image_derefs, bool lower_sampler_derefs)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(shader);

   ASSERTED int last_loc = -1;
   int num_rd_images = 0, num_wr_images = 0;
   nir_foreach_variable_with_modes(var, shader, nir_var_image | nir_var_uniform) {
      if (!glsl_type_is_image(var->type) && !glsl_type_is_texture(var->type))
         continue;

      /* Assume they come in order */
      assert(var->data.location > last_loc);
      last_loc = var->data.location;

      assert(glsl_type_is_image(var->type) || var->data.access & ACCESS_NON_WRITEABLE);
      if (var->data.access & ACCESS_NON_WRITEABLE)
         var->data.driver_location = num_rd_images++;
      else
         var->data.driver_location = num_wr_images++;
      var->data.binding = var->data.driver_location;
   }
   shader->info.num_textures = num_rd_images;
   BITSET_ZERO(shader->info.textures_used);
   if (num_rd_images)
      BITSET_SET_RANGE(shader->info.textures_used, 0, num_rd_images - 1);

   BITSET_ZERO(shader->info.images_used);
   if (num_wr_images)
      BITSET_SET_RANGE(shader->info.images_used, 0, num_wr_images - 1);
   shader->info.num_images = num_wr_images;

   last_loc = -1;
   int num_samplers = 0;
   nir_foreach_uniform_variable(var, shader) {
      if (var->type == glsl_bare_sampler_type()) {
         /* Assume they come in order */
         assert(var->data.location > last_loc);
         last_loc = var->data.location;
         var->data.driver_location = num_samplers++;
      } else {
         /* CL shouldn't have any sampled images */
         assert(!glsl_type_is_sampler(var->type));
      }
   }
   BITSET_ZERO(shader->info.samplers_used);
   if (num_samplers)
      BITSET_SET_RANGE(shader->info.samplers_used, 0, num_samplers - 1);

   nir_builder b;
   nir_builder_init(&b, impl);

   /* don't need any lowering if we can keep the derefs */
   if (!lower_image_derefs && !lower_sampler_derefs) {
      nir_metadata_preserve(impl, nir_metadata_all);
      return false;
   }

   bool progress = false;
   nir_foreach_block_reverse(block, impl) {
      nir_foreach_instr_reverse_safe(instr, block) {
         switch (instr->type) {
         case nir_instr_type_deref: {
            nir_deref_instr *deref = nir_instr_as_deref(instr);
            if (deref->deref_type != nir_deref_type_var)
               break;

            if (!glsl_type_is_image(deref->type) &&
                !glsl_type_is_texture(deref->type) &&
                !glsl_type_is_sampler(deref->type))
               break;

            if (!lower_image_derefs && glsl_type_is_image(deref->type))
               break;

            if (!lower_sampler_derefs &&
                (glsl_type_is_sampler(deref->type) || glsl_type_is_texture(deref->type)))
               break;

            b.cursor = nir_instr_remove(&deref->instr);
            nir_ssa_def *loc =
               nir_imm_intN_t(&b, deref->var->data.driver_location,
                                  deref->dest.ssa.bit_size);
            nir_ssa_def_rewrite_uses(&deref->dest.ssa, loc);
            progress = true;
            break;
         }

         case nir_instr_type_tex: {
            if (!lower_sampler_derefs)
               break;

            nir_tex_instr *tex = nir_instr_as_tex(instr);
            unsigned count = 0;
            for (unsigned i = 0; i < tex->num_srcs; i++) {
               if (tex->src[i].src_type == nir_tex_src_texture_deref ||
                   tex->src[i].src_type == nir_tex_src_sampler_deref) {
                  nir_deref_instr *deref = nir_src_as_deref(tex->src[i].src);
                  if (deref->deref_type == nir_deref_type_var) {
                     /* In this case, we know the actual variable */
                     if (tex->src[i].src_type == nir_tex_src_texture_deref)
                        tex->texture_index = deref->var->data.driver_location;
                     else
                        tex->sampler_index = deref->var->data.driver_location;
                     /* This source gets discarded */
                     nir_instr_rewrite_src(&tex->instr, &tex->src[i].src,
                                           NIR_SRC_INIT);
                     continue;
                  } else {
                     assert(tex->src[i].src.is_ssa);
                     b.cursor = nir_before_instr(&tex->instr);
                     /* Back-ends expect a 32-bit thing, not 64-bit */
                     nir_ssa_def *offset = nir_u2u32(&b, tex->src[i].src.ssa);
                     if (tex->src[i].src_type == nir_tex_src_texture_deref)
                        tex->src[count].src_type = nir_tex_src_texture_offset;
                     else
                        tex->src[count].src_type = nir_tex_src_sampler_offset;
                     nir_instr_rewrite_src(&tex->instr, &tex->src[count].src,
                                           nir_src_for_ssa(offset));
                  }
               } else {
                  /* If we've removed a source, move this one down */
                  if (count != i) {
                     assert(count < i);
                     tex->src[count].src_type = tex->src[i].src_type;
                     nir_instr_move_src(&tex->instr, &tex->src[count].src,
                                        &tex->src[i].src);
                  }
               }
               count++;
            }
            tex->num_srcs = count;
            progress = true;
            break;
         }

         case nir_instr_type_intrinsic: {
            nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
            switch (intrin->intrinsic) {
            case nir_intrinsic_image_deref_load:
            case nir_intrinsic_image_deref_store:
            case nir_intrinsic_image_deref_atomic_add:
            case nir_intrinsic_image_deref_atomic_imin:
            case nir_intrinsic_image_deref_atomic_umin:
            case nir_intrinsic_image_deref_atomic_imax:
            case nir_intrinsic_image_deref_atomic_umax:
            case nir_intrinsic_image_deref_atomic_and:
            case nir_intrinsic_image_deref_atomic_or:
            case nir_intrinsic_image_deref_atomic_xor:
            case nir_intrinsic_image_deref_atomic_exchange:
            case nir_intrinsic_image_deref_atomic_comp_swap:
            case nir_intrinsic_image_deref_atomic_fadd:
            case nir_intrinsic_image_deref_atomic_inc_wrap:
            case nir_intrinsic_image_deref_atomic_dec_wrap:
            case nir_intrinsic_image_deref_size:
            case nir_intrinsic_image_deref_samples: {
               if (!lower_image_derefs)
                  break;

               assert(intrin->src[0].is_ssa);
               b.cursor = nir_before_instr(&intrin->instr);
               /* Back-ends expect a 32-bit thing, not 64-bit */
               nir_ssa_def *offset = nir_u2u32(&b, intrin->src[0].ssa);
               nir_rewrite_image_intrinsic(intrin, offset, false);
               progress = true;
               break;
            }

            default:
               break;
            }
            break;
         }

         default:
            break;
         }
      }
   }

   if (progress) {
      nir_metadata_preserve(impl, nir_metadata_block_index |
                                  nir_metadata_dominance);
   } else {
      nir_metadata_preserve(impl, nir_metadata_all);
   }

   return progress;
}
