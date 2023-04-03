/*
 * Copyright © 2020 Intel Corporation
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

#include "nir_builder.h"

static bool
opt_memcpy_deref_cast(nir_intrinsic_instr *cpy, nir_src *deref_src)
{
   assert(cpy->intrinsic == nir_intrinsic_memcpy_deref);

   nir_deref_instr *cast = nir_src_as_deref(*deref_src);
   if (cast == NULL || cast->deref_type != nir_deref_type_cast)
      return false;

   /* We always have to replace the source with a deref, not a bare uint
    * pointer.  If it's the first deref in the chain, bail.
    */
   nir_deref_instr *parent = nir_src_as_deref(cast->parent);
   if (parent == NULL)
      return false;

   /* If it has useful alignment information, we want to keep that */
   if (cast->cast.align_mul > 0)
      return false;

   /* Casts to uint8 or int8 never do us any good; get rid of them */
   if (cast->type == glsl_int8_t_type() ||
       cast->type == glsl_uint8_t_type()) {
      nir_instr_rewrite_src(&cpy->instr, deref_src,
                            nir_src_for_ssa(&parent->dest.ssa));
      return true;
   }

   int64_t parent_type_size = glsl_get_explicit_size(parent->type, false);
   if (parent_type_size < 0)
      return false;

   if (!nir_src_is_const(cpy->src[2]))
      return false;

   /* We don't want to get rid of the cast if the resulting type would be
    * smaller than the amount of data we're copying.
    */
   if (nir_src_as_uint(cpy->src[2]) < (uint64_t)parent_type_size)
      return false;

   nir_instr_rewrite_src(&cpy->instr, deref_src,
                         nir_src_for_ssa(&parent->dest.ssa));
   return true;
}

static bool
type_is_tightly_packed(const struct glsl_type *type, unsigned *size_out)
{
   unsigned size = 0;
   if (glsl_type_is_struct_or_ifc(type)) {
      unsigned num_fields = glsl_get_length(type);
      for (unsigned i = 0; i < num_fields; i++) {
         const struct glsl_struct_field *field =
            glsl_get_struct_field_data(type, i);

         if (field->offset < 0 || field->offset != size)
            return false;

         unsigned field_size;
         if (!type_is_tightly_packed(field->type, &field_size))
            return false;

         size = field->offset + field_size;
      }
   } else if (glsl_type_is_array_or_matrix(type)) {
      if (glsl_type_is_unsized_array(type))
         return false;

      unsigned stride = glsl_get_explicit_stride(type);
      if (stride == 0)
         return false;

      const struct glsl_type *elem_type = glsl_get_array_element(type);

      unsigned elem_size;
      if (!type_is_tightly_packed(elem_type, &elem_size))
         return false;

      if (elem_size != stride)
         return false;

      size = stride * glsl_get_length(type);
   } else {
      assert(glsl_type_is_vector_or_scalar(type));
      if (glsl_get_explicit_stride(type) > 0)
         return false;

      if (glsl_type_is_boolean(type))
         return false;

      size = glsl_get_explicit_size(type, false);
   }

   if (size_out)
      *size_out = size;
   return true;
}

static bool
try_lower_memcpy(nir_builder *b, nir_intrinsic_instr *cpy,
                 struct set *complex_vars)
{
   nir_deref_instr *dst = nir_src_as_deref(cpy->src[0]);
   nir_deref_instr *src = nir_src_as_deref(cpy->src[1]);

   /* A self-copy can always be eliminated */
   if (dst == src) {
      nir_instr_remove(&cpy->instr);
      return true;
   }

   if (!nir_src_is_const(cpy->src[2]))
      return false;

   uint64_t size = nir_src_as_uint(cpy->src[2]);
   if (size == 0) {
      nir_instr_remove(&cpy->instr);
      return true;
   }

   if (glsl_type_is_vector_or_scalar(src->type) &&
       glsl_type_is_vector_or_scalar(dst->type) &&
       glsl_get_explicit_size(dst->type, false) == size &&
       glsl_get_explicit_size(src->type, false) == size) {
      b->cursor = nir_instr_remove(&cpy->instr);
      nir_ssa_def *data =
         nir_load_deref_with_access(b, src, nir_intrinsic_src_access(cpy));
      data = nir_bitcast_vector(b, data, glsl_get_bit_size(dst->type));
      assert(data->num_components == glsl_get_vector_elements(dst->type));
      nir_store_deref_with_access(b, dst, data, ~0 /* write mask */,
                                  nir_intrinsic_dst_access(cpy));
      return true;
   }

   unsigned type_size;
   if (dst->type == src->type &&
       type_is_tightly_packed(dst->type, &type_size) &&
       type_size == size) {
      b->cursor = nir_instr_remove(&cpy->instr);
      nir_copy_deref_with_access(b, dst, src,
                                 nir_intrinsic_dst_access(cpy),
                                 nir_intrinsic_src_access(cpy));
      return true;
   }

   /* If one of the two types is tightly packed and happens to equal the
    * memcpy size, then we can get the memcpy by casting to that type and
    * doing a deref copy.
    *
    * However, if we blindly apply this logic, we may end up with extra casts
    * where we don't want them. The whole point of converting memcpy to
    * copy_deref is in the hopes that nir_opt_copy_prop_vars or
    * nir_lower_vars_to_ssa will get rid of the copy and those passes don't
    * handle casts well. Heuristically, only do this optimization if the
    * tightly packed type is on a deref with nir_var_function_temp so we stick
    * the cast on the other mode.
    */
   if (dst->modes == nir_var_function_temp &&
       type_is_tightly_packed(dst->type, &type_size) &&
       type_size == size) {
      b->cursor = nir_instr_remove(&cpy->instr);
      src = nir_build_deref_cast(b, &src->dest.ssa,
                                 src->modes, dst->type, 0);
      nir_copy_deref_with_access(b, dst, src,
                                 nir_intrinsic_dst_access(cpy),
                                 nir_intrinsic_src_access(cpy));
      return true;
   }

   /* If we can get at the variable AND the only complex use of that variable
    * is as a memcpy destination, then we don't have to care about any empty
    * space in the variable.  In particular, we know that the variable is never
    * cast to any other type and it's never used as a memcpy source so nothing
    * can see any padding bytes.  This holds even if some other memcpy only
    * writes to part of the variable.
    */
   if (dst->deref_type == nir_deref_type_var &&
       dst->modes == nir_var_function_temp &&
       _mesa_set_search(complex_vars, dst->var) == NULL &&
       glsl_get_explicit_size(dst->type, false) <= size) {
      b->cursor = nir_instr_remove(&cpy->instr);
      src = nir_build_deref_cast(b, &src->dest.ssa,
                                 src->modes, dst->type, 0);
      nir_copy_deref_with_access(b, dst, src,
                                 nir_intrinsic_dst_access(cpy),
                                 nir_intrinsic_src_access(cpy));
      return true;
   }

   if (src->modes == nir_var_function_temp &&
       type_is_tightly_packed(src->type, &type_size) &&
       type_size == size) {
      b->cursor = nir_instr_remove(&cpy->instr);
      dst = nir_build_deref_cast(b, &dst->dest.ssa,
                                 dst->modes, src->type, 0);
      nir_copy_deref_with_access(b, dst, src,
                                 nir_intrinsic_dst_access(cpy),
                                 nir_intrinsic_src_access(cpy));
      return true;
   }

   return false;
}

static bool
opt_memcpy_impl(nir_function_impl *impl)
{
   bool progress = false;

   nir_builder b;
   nir_builder_init(&b, impl);

   struct set *complex_vars = _mesa_pointer_set_create(NULL);

   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_deref)
            continue;

         nir_deref_instr *deref = nir_instr_as_deref(instr);
         if (deref->deref_type != nir_deref_type_var)
            continue;

         nir_deref_instr_has_complex_use_options opts =
            nir_deref_instr_has_complex_use_allow_memcpy_dst;
         if (nir_deref_instr_has_complex_use(deref, opts))
            _mesa_set_add(complex_vars, deref->var);
      }
   }

   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;

         nir_intrinsic_instr *cpy = nir_instr_as_intrinsic(instr);
         if (cpy->intrinsic != nir_intrinsic_memcpy_deref)
            continue;

         while (opt_memcpy_deref_cast(cpy, &cpy->src[0]))
            progress = true;
         while (opt_memcpy_deref_cast(cpy, &cpy->src[1]))
            progress = true;

         if (try_lower_memcpy(&b, cpy, complex_vars)) {
            progress = true;
            continue;
         }
      }
   }

   _mesa_set_destroy(complex_vars, NULL);

   if (progress) {
      nir_metadata_preserve(impl, nir_metadata_block_index |
                                  nir_metadata_dominance);
   } else {
      nir_metadata_preserve(impl, nir_metadata_all);
   }

   return progress;
}

bool
nir_opt_memcpy(nir_shader *shader)
{
   bool progress = false;

   nir_foreach_function(function, shader) {
      if (function->impl && opt_memcpy_impl(function->impl))
         progress = true;
   }

   return progress;
}
