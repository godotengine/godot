/*
 * Copyright © 2014 Intel Corporation
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
 *    Jason Ekstrand (jason@jlekstrand.net)
 *
 */

#include "nir.h"
#include "nir_builder.h"
#include "nir_deref.h"
#include "compiler/nir_types.h"

/*
 * Lowers all copy intrinsics to sequences of load/store intrinsics.
 */

static nir_deref_instr *
build_deref_to_next_wildcard(nir_builder *b,
                             nir_deref_instr *parent,
                             nir_deref_instr ***deref_arr)
{
   for (; **deref_arr; (*deref_arr)++) {
      if ((**deref_arr)->deref_type == nir_deref_type_array_wildcard)
         return parent;

      parent = nir_build_deref_follower(b, parent, **deref_arr);
   }

   assert(**deref_arr == NULL);
   *deref_arr = NULL;
   return parent;
}

static void
emit_deref_copy_load_store(nir_builder *b,
                           nir_deref_instr *dst_deref,
                           nir_deref_instr **dst_deref_arr,
                           nir_deref_instr *src_deref,
                           nir_deref_instr **src_deref_arr,
                           enum gl_access_qualifier dst_access,
                           enum gl_access_qualifier src_access)
{
   if (dst_deref_arr || src_deref_arr) {
      assert(dst_deref_arr && src_deref_arr);
      dst_deref = build_deref_to_next_wildcard(b, dst_deref, &dst_deref_arr);
      src_deref = build_deref_to_next_wildcard(b, src_deref, &src_deref_arr);
   }

   if (dst_deref_arr || src_deref_arr) {
      assert(dst_deref_arr && src_deref_arr);
      assert((*dst_deref_arr)->deref_type == nir_deref_type_array_wildcard);
      assert((*src_deref_arr)->deref_type == nir_deref_type_array_wildcard);

      unsigned length = glsl_get_length(src_deref->type);
      /* The wildcards should represent the same number of elements */
      assert(length == glsl_get_length(dst_deref->type));
      assert(length > 0);

      for (unsigned i = 0; i < length; i++) {
         emit_deref_copy_load_store(b,
                                    nir_build_deref_array_imm(b, dst_deref, i),
                                    dst_deref_arr + 1,
                                    nir_build_deref_array_imm(b, src_deref, i),
                                    src_deref_arr + 1, dst_access, src_access);
      }
   } else {
      assert(glsl_get_bare_type(dst_deref->type) ==
             glsl_get_bare_type(src_deref->type));
      assert(glsl_type_is_vector_or_scalar(dst_deref->type));

      nir_store_deref_with_access(b, dst_deref,
                                  nir_load_deref_with_access(b, src_deref, src_access),
                                  ~0, src_access);
   }
}

void
nir_lower_deref_copy_instr(nir_builder *b, nir_intrinsic_instr *copy)
{
   /* Unfortunately, there's just no good way to handle wildcards except to
    * flip the chain around and walk the list from variable to final pointer.
    */
   assert(copy->src[0].is_ssa && copy->src[1].is_ssa);
   nir_deref_instr *dst = nir_instr_as_deref(copy->src[0].ssa->parent_instr);
   nir_deref_instr *src = nir_instr_as_deref(copy->src[1].ssa->parent_instr);

   nir_deref_path dst_path, src_path;
   nir_deref_path_init(&dst_path, dst, NULL);
   nir_deref_path_init(&src_path, src, NULL);

   b->cursor = nir_before_instr(&copy->instr);
   emit_deref_copy_load_store(b, dst_path.path[0], &dst_path.path[1],
                                 src_path.path[0], &src_path.path[1],
                                 nir_intrinsic_dst_access(copy),
                                 nir_intrinsic_src_access(copy));

   nir_deref_path_finish(&dst_path);
   nir_deref_path_finish(&src_path);
}

static bool
lower_var_copies_impl(nir_function_impl *impl)
{
   bool progress = false;

   nir_builder b;
   nir_builder_init(&b, impl);

   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;

         nir_intrinsic_instr *copy = nir_instr_as_intrinsic(instr);
         if (copy->intrinsic != nir_intrinsic_copy_deref)
            continue;

         nir_lower_deref_copy_instr(&b, copy);

         nir_instr_remove(&copy->instr);
         nir_deref_instr_remove_if_unused(nir_src_as_deref(copy->src[0]));
         nir_deref_instr_remove_if_unused(nir_src_as_deref(copy->src[1]));

         progress = true;
         nir_instr_free(&copy->instr);
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

/* Lowers every copy_var instruction in the program to a sequence of
 * load/store instructions.
 */
bool
nir_lower_var_copies(nir_shader *shader)
{
   bool progress = false;

   nir_foreach_function(function, shader) {
      if (function->impl)
         progress |= lower_var_copies_impl(function->impl);
   }

   return progress;
}
