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

struct var_info {
   nir_variable *var;

   bool is_constant;
   bool found_read;
   bool duplicate;

   /* Block that has all the variable stores.  All the blocks with reads
    * should be dominated by this block.
    */
   nir_block *block;

   /* If is_constant, hold the collected constant data for this var. */
   uint32_t constant_data_size;
   void *constant_data;
};

static int
var_info_cmp(const void *_a, const void *_b)
{
   const struct var_info *a = _a;
   const struct var_info *b = _b;
   uint32_t a_size = a->constant_data_size;
   uint32_t b_size = b->constant_data_size;

   if (a->is_constant != b->is_constant) {
      return (int)a->is_constant - (int)b->is_constant;
   } else if (a_size < b_size) {
      return -1;
   } else if (a_size > b_size) {
      return 1;
   } else if (a_size == 0) {
      /* Don't call memcmp with invalid pointers. */
      return 0;
   } else {
      return memcmp(a->constant_data, b->constant_data, a_size);
   }
}

static nir_ssa_def *
build_constant_load(nir_builder *b, nir_deref_instr *deref,
                    glsl_type_size_align_func size_align)
{
   nir_variable *var = nir_deref_instr_get_variable(deref);

   const unsigned bit_size = glsl_get_bit_size(deref->type);
   const unsigned num_components = glsl_get_vector_elements(deref->type);

   UNUSED unsigned var_size, var_align;
   size_align(var->type, &var_size, &var_align);
   assert(var->data.location % var_align == 0);

   UNUSED unsigned deref_size, deref_align;
   size_align(deref->type, &deref_size, &deref_align);

   nir_ssa_def *src = nir_build_deref_offset(b, deref, size_align);
   nir_ssa_def *load =
      nir_load_constant(b, num_components, bit_size, src,
                        .base = var->data.location,
                        .range = var_size,
                        .align_mul = deref_align,
                        .align_offset = 0);

   if (load->bit_size < 8) {
      /* Booleans are special-cased to be 32-bit */
      assert(glsl_type_is_boolean(deref->type));
      assert(deref_size == num_components * 4);
      load->bit_size = 32;
      return nir_b2b1(b, load);
   } else {
      assert(deref_size == num_components * bit_size / 8);
      return load;
   }
}

static void
handle_constant_store(void *mem_ctx, struct var_info *info,
                      nir_deref_instr *deref, nir_const_value *val,
                      unsigned writemask,
                      glsl_type_size_align_func size_align)
{
   assert(!nir_deref_instr_has_indirect(deref));
   const unsigned bit_size = glsl_get_bit_size(deref->type);
   const unsigned num_components = glsl_get_vector_elements(deref->type);

   if (info->constant_data_size == 0) {
      unsigned var_size, var_align;
      size_align(info->var->type, &var_size, &var_align);
      info->constant_data_size = var_size;
      info->constant_data = rzalloc_size(mem_ctx, var_size);
   }

   const unsigned offset = nir_deref_instr_get_const_offset(deref, size_align);
   if (offset >= info->constant_data_size)
      return;

   char *dst = (char *)info->constant_data + offset;

   for (unsigned i = 0; i < num_components; i++) {
      if (!(writemask & (1 << i)))
         continue;

      switch (bit_size) {
      case 1:
         /* Booleans are special-cased to be 32-bit */
         ((int32_t *)dst)[i] = -(int)val[i].b;
         break;

      case 8:
         ((uint8_t *)dst)[i] = val[i].u8;
         break;

      case 16:
         ((uint16_t *)dst)[i] = val[i].u16;
         break;

      case 32:
         ((uint32_t *)dst)[i] = val[i].u32;
         break;

      case 64:
         ((uint64_t *)dst)[i] = val[i].u64;
         break;

      default:
         unreachable("Invalid bit size");
      }
   }
}

/** Lower large constant variables to shader constant data
 *
 * This pass looks for large (type_size(var->type) > threshold) variables
 * which are statically constant and moves them into shader constant data.
 * This is especially useful when large tables are baked into the shader
 * source code because they can be moved into a UBO by the driver to reduce
 * register pressure and make indirect access cheaper.
 */
bool
nir_opt_large_constants(nir_shader *shader,
                        glsl_type_size_align_func size_align,
                        unsigned threshold)
{
   /* Default to a natural alignment if none is provided */
   if (size_align == NULL)
      size_align = glsl_get_natural_size_align_bytes;

   /* This only works with a single entrypoint */
   nir_function_impl *impl = nir_shader_get_entrypoint(shader);

   unsigned num_locals = nir_function_impl_index_vars(impl);

   if (num_locals == 0) {
      nir_shader_preserve_all_metadata(shader);
      return false;
   }

   struct var_info *var_infos = ralloc_array(NULL, struct var_info, num_locals);
   nir_foreach_function_temp_variable(var, impl) {
      var_infos[var->index] = (struct var_info) {
         .var = var,
         .is_constant = true,
         .found_read = false,
      };
   }

   nir_metadata_require(impl, nir_metadata_dominance);

   /* First, walk through the shader and figure out what variables we can
    * lower to the constant blob.
    */
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_deref) {
            /* If we ever see a complex use of a deref_var, we have to assume
             * that variable is non-constant because we can't guarantee we
             * will find all of the writers of that variable.
             */
            nir_deref_instr *deref = nir_instr_as_deref(instr);
            if (deref->deref_type == nir_deref_type_var &&
                deref->var->data.mode == nir_var_function_temp &&
                nir_deref_instr_has_complex_use(deref, 0))
               var_infos[deref->var->index].is_constant = false;
            continue;
         }

         if (instr->type != nir_instr_type_intrinsic)
            continue;

         nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);

         bool src_is_const = false;
         nir_deref_instr *src_deref = NULL, *dst_deref = NULL;
         unsigned writemask = 0;
         switch (intrin->intrinsic) {
         case nir_intrinsic_store_deref:
            dst_deref = nir_src_as_deref(intrin->src[0]);
            src_is_const = nir_src_is_const(intrin->src[1]);
            writemask = nir_intrinsic_write_mask(intrin);
            break;

         case nir_intrinsic_load_deref:
            src_deref = nir_src_as_deref(intrin->src[0]);
            break;

         case nir_intrinsic_copy_deref:
            assert(!"Lowering of copy_deref with large constants is prohibited");
            break;

         default:
            continue;
         }

         if (dst_deref && nir_deref_mode_must_be(dst_deref, nir_var_function_temp)) {
            nir_variable *var = nir_deref_instr_get_variable(dst_deref);
            if (var == NULL)
               continue;

            assert(var->data.mode == nir_var_function_temp);

            struct var_info *info = &var_infos[var->index];
            if (!info->is_constant)
               continue;

            if (!info->block)
               info->block = block;

            /* We only consider variables constant if they only have constant
             * stores, all the stores come before any reads, and all stores
             * come from the same block.  We also can't handle indirect stores.
             */
            if (!src_is_const || info->found_read || block != info->block ||
                nir_deref_instr_has_indirect(dst_deref)) {
               info->is_constant = false;
            } else {
               nir_const_value *val = nir_src_as_const_value(intrin->src[1]);
               handle_constant_store(var_infos, info, dst_deref, val, writemask,
                                     size_align);
            }
         }

         if (src_deref && nir_deref_mode_must_be(src_deref, nir_var_function_temp)) {
            nir_variable *var = nir_deref_instr_get_variable(src_deref);
            if (var == NULL)
               continue;

            assert(var->data.mode == nir_var_function_temp);

            /* We only consider variables constant if all the reads are
             * dominated by the block that writes to it.
             */
            struct var_info *info = &var_infos[var->index];
            if (!info->is_constant)
               continue;

            if (!info->block || !nir_block_dominates(info->block, block))
               info->is_constant = false;

            info->found_read = true;
         }
      }
   }

   /* Allocate constant data space for each variable that just has constant
    * data.  We sort them by size and content so we can easily find
    * duplicates.
    */
   const unsigned old_constant_data_size = shader->constant_data_size;
   qsort(var_infos, num_locals, sizeof(struct var_info), var_info_cmp);
   for (int i = 0; i < num_locals; i++) {
      struct var_info *info = &var_infos[i];

      /* Fix up indices after we sorted. */
      info->var->index = i;

      if (!info->is_constant)
         continue;

      unsigned var_size, var_align;
      size_align(info->var->type, &var_size, &var_align);
      if (var_size <= threshold || !info->found_read) {
         /* Don't bother lowering small stuff or data that's never read */
         info->is_constant = false;
         continue;
      }

      if (i > 0 && var_info_cmp(info, &var_infos[i - 1]) == 0) {
         info->var->data.location = var_infos[i - 1].var->data.location;
         info->duplicate = true;
      } else {
         info->var->data.location = ALIGN_POT(shader->constant_data_size, var_align);
         shader->constant_data_size = info->var->data.location + var_size;
      }
   }

   if (shader->constant_data_size == old_constant_data_size) {
      nir_shader_preserve_all_metadata(shader);
      ralloc_free(var_infos);
      return false;
   }

   assert(shader->constant_data_size > old_constant_data_size);
   shader->constant_data = rerzalloc_size(shader, shader->constant_data,
                                          old_constant_data_size,
                                          shader->constant_data_size);
   for (int i = 0; i < num_locals; i++) {
      struct var_info *info = &var_infos[i];
      if (!info->duplicate && info->is_constant) {
         memcpy((char *)shader->constant_data + info->var->data.location,
                info->constant_data, info->constant_data_size);
      }
   }

   nir_builder b;
   nir_builder_init(&b, impl);

   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;

         nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);

         switch (intrin->intrinsic) {
         case nir_intrinsic_load_deref: {
            nir_deref_instr *deref = nir_src_as_deref(intrin->src[0]);
            if (!nir_deref_mode_is(deref, nir_var_function_temp))
               continue;

            nir_variable *var = nir_deref_instr_get_variable(deref);
            if (var == NULL)
               continue;

            struct var_info *info = &var_infos[var->index];
            if (info->is_constant) {
               b.cursor = nir_after_instr(&intrin->instr);
               nir_ssa_def *val = build_constant_load(&b, deref, size_align);
               nir_ssa_def_rewrite_uses(&intrin->dest.ssa,
                                        val);
               nir_instr_remove(&intrin->instr);
               nir_deref_instr_remove_if_unused(deref);
            }
            break;
         }

         case nir_intrinsic_store_deref: {
            nir_deref_instr *deref = nir_src_as_deref(intrin->src[0]);
            if (!nir_deref_mode_is(deref, nir_var_function_temp))
               continue;

            nir_variable *var = nir_deref_instr_get_variable(deref);
            if (var == NULL)
               continue;

            struct var_info *info = &var_infos[var->index];
            if (info->is_constant) {
               nir_instr_remove(&intrin->instr);
               nir_deref_instr_remove_if_unused(deref);
            }
            break;
         }
         case nir_intrinsic_copy_deref:
         default:
            continue;
         }
      }
   }

   /* Clean up the now unused variables */
   for (int i = 0; i < num_locals; i++) {
      struct var_info *info = &var_infos[i];
      if (info->is_constant)
         exec_node_remove(&info->var->node);
   }

   ralloc_free(var_infos);

   nir_metadata_preserve(impl, nir_metadata_block_index |
                               nir_metadata_dominance);
   return true;
}
