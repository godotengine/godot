/*
 * Copyright © 2021 Igalia S.L.
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
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "nir.h"
#include "nir_builder.h"

/* Some hardware doesn't have a way to check if invocation was demoted,
 * in such case we have to track it ourselves.
 * OpIsHelperInvocationEXT is specified as:
 *
 *  "An invocation is currently a helper invocation if it was originally
 *   invoked as a helper invocation or if it has been demoted to a helper
 *   invocation by OpDemoteToHelperInvocationEXT."
 *
 * Therefore we:
 * - Set gl_IsHelperInvocationEXT = gl_HelperInvocation
 * - Add "gl_IsHelperInvocationEXT = true" right before each demote
 * - Add "gl_IsHelperInvocationEXT = gl_IsHelperInvocationEXT || condition"
 *   right before each demote_if
 */

static bool
nir_lower_load_and_store_is_helper(nir_builder *b, nir_instr *instr, void *data)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
   nir_deref_instr *is_helper_deref = (nir_deref_instr*) data;

   switch (intrin->intrinsic) {
   case nir_intrinsic_demote: {
      b->cursor = nir_before_instr(instr);
      nir_store_deref(b, is_helper_deref, nir_imm_bool(b, true), 1);
      return true;
   }
   case nir_intrinsic_demote_if: {
      b->cursor = nir_before_instr(instr);
      nir_ssa_def *current_is_helper = nir_load_deref(b, is_helper_deref);
      nir_ssa_def *updated_is_helper = nir_ior(b, current_is_helper, intrin->src[0].ssa);
      nir_store_deref(b, is_helper_deref, updated_is_helper, 1);
      return true;
   }
   case nir_intrinsic_is_helper_invocation: {
      b->cursor = nir_before_instr(instr);
      nir_ssa_def *is_helper = nir_load_deref(b, is_helper_deref);
      nir_ssa_def_rewrite_uses(&intrin->dest.ssa, is_helper);
      nir_instr_remove_v(instr);
      return true;
   }
   default:
      return false;
   }
}

static bool
has_is_helper_invocation(nir_shader *shader)
{
   nir_foreach_function(function, shader) {
      if (!function->impl)
         continue;

      nir_foreach_block_safe(block, function->impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
            if (intrin->intrinsic == nir_intrinsic_is_helper_invocation)
               return true;
         }
      }
   }

   return false;
}

bool
nir_lower_is_helper_invocation(nir_shader *shader)
{
   if (shader->info.stage != MESA_SHADER_FRAGMENT)
      return false;

   if (!has_is_helper_invocation(shader))
      return false;

   nir_function_impl *entrypoint = nir_shader_get_entrypoint(shader);

   nir_builder b;
   nir_builder_init(&b, entrypoint);
   b.cursor = nir_before_cf_list(&entrypoint->body);

   nir_variable *is_helper = nir_local_variable_create(entrypoint,
                                          glsl_bool_type(),
                                          "gl_IsHelperInvocationEXT");

   nir_ssa_def *started_as_helper = shader->options->lower_helper_invocation ?
      nir_build_lowered_load_helper_invocation(&b) :
      nir_load_helper_invocation(&b, 1);

   nir_deref_instr *is_helper_deref = nir_build_deref_var(&b, is_helper);
   nir_store_deref(&b, is_helper_deref, started_as_helper, 1);

   return nir_shader_instructions_pass(shader,
                                       nir_lower_load_and_store_is_helper,
                                       nir_metadata_all,
                                       is_helper_deref);
}
