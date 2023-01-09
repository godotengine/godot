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
 *    Connor Abbott (cwabbott0@gmail.com)
 *
 */

#include "nir.h"

static bool
deref_used_for_not_store(nir_deref_instr *deref)
{
   nir_foreach_use(src, &deref->dest.ssa) {
      switch (src->parent_instr->type) {
      case nir_instr_type_deref:
         if (deref_used_for_not_store(nir_instr_as_deref(src->parent_instr)))
            return true;
         break;

      case nir_instr_type_intrinsic: {
         nir_intrinsic_instr *intrin =
            nir_instr_as_intrinsic(src->parent_instr);
         /* The first source of copy and store intrinsics is the deref to
          * write.  Don't record those.
          */
         if ((intrin->intrinsic != nir_intrinsic_store_deref &&
              intrin->intrinsic != nir_intrinsic_copy_deref) ||
             src != &intrin->src[0])
            return true;
         break;
      }

      default:
         /* If it's used by any other instruction type (most likely a texture
          * or call instruction), consider it used.
          */
         return true;
      }
   }

   return false;
}

static void
add_var_use_deref(nir_deref_instr *deref, struct set *live)
{
   if (deref->deref_type != nir_deref_type_var)
      return;

   /* Since these local variables don't escape the shader, writing doesn't
    * make them live.  Only keep them if they are used by some intrinsic.
    */
   if ((deref->var->data.mode & (nir_var_function_temp |
                                 nir_var_shader_temp)) &&
       !deref_used_for_not_store(deref))
      return;

   /*
    * Shared memory blocks (interface type) alias each other, so be
    * conservative in that case.
    */
   if ((deref->var->data.mode & nir_var_mem_shared) &&
       !glsl_type_is_interface(deref->var->type) &&
       !deref_used_for_not_store(deref))
      return;

   nir_variable *var = deref->var;
   do {
      _mesa_set_add(live, var);
      /* Also mark the chain of variables used to initialize it. */
      var = var->pointer_initializer;
   } while (var);
}

static void
add_var_use_shader(nir_shader *shader, struct set *live, nir_variable_mode modes)
{
   nir_foreach_function(function, shader) {
      if (function->impl) {
         nir_foreach_block(block, function->impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type == nir_instr_type_deref)
                  add_var_use_deref(nir_instr_as_deref(instr), live);
            }
         }
      }
   }
}

static void
remove_dead_var_writes(nir_shader *shader)
{
   nir_foreach_function(function, shader) {
      if (!function->impl)
         continue;

      nir_foreach_block(block, function->impl) {
         nir_foreach_instr_safe(instr, block) {
            switch (instr->type) {
            case nir_instr_type_deref: {
               nir_deref_instr *deref = nir_instr_as_deref(instr);
               if (deref->deref_type == nir_deref_type_cast &&
                   !nir_deref_instr_parent(deref))
                  continue;

               nir_variable_mode parent_modes;
               if (deref->deref_type == nir_deref_type_var)
                  parent_modes = deref->var->data.mode;
               else
                  parent_modes = nir_deref_instr_parent(deref)->modes;

               /* If the parent mode is 0, then it references a dead variable.
                * Flag this deref as dead and remove it.
                */
               if (parent_modes == 0) {
                  deref->modes = 0;
                  nir_instr_remove(&deref->instr);
               }
               break;
            }

            case nir_instr_type_intrinsic: {
               nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
               if (intrin->intrinsic != nir_intrinsic_copy_deref &&
                   intrin->intrinsic != nir_intrinsic_store_deref)
                  break;

               if (nir_src_as_deref(intrin->src[0])->modes == 0)
                  nir_instr_remove(instr);
               break;
            }

            default:
               break; /* Nothing to do */
            }
         }
      }
   }
}

static bool
remove_dead_vars(struct exec_list *var_list, nir_variable_mode modes,
                 struct set *live, const nir_remove_dead_variables_options *opts)
{
   bool progress = false;

   nir_foreach_variable_in_list_safe(var, var_list) {
      if (!(var->data.mode & modes))
         continue;

      if (opts && opts->can_remove_var &&
          !opts->can_remove_var(var, opts->can_remove_var_data))
         continue;

      struct set_entry *entry = _mesa_set_search(live, var);
      if (entry == NULL) {
         /* Mark this variable as used by setting the mode to 0 */
         var->data.mode = 0;
         exec_node_remove(&var->node);
         progress = true;
      }
   }

   return progress;
}

bool
nir_remove_dead_variables(nir_shader *shader, nir_variable_mode modes,
                          const nir_remove_dead_variables_options *opts)
{
   bool progress = false;
   struct set *live = _mesa_pointer_set_create(NULL);

   add_var_use_shader(shader, live, modes);

   if (modes & ~nir_var_function_temp) {
      progress = remove_dead_vars(&shader->variables, modes,
                                  live, opts) || progress;
   }

   if (modes & nir_var_function_temp) {
      nir_foreach_function(function, shader) {
         if (function->impl) {
            if (remove_dead_vars(&function->impl->locals,
                                 nir_var_function_temp,
                                 live, opts))
               progress = true;
         }
      }
   }

   _mesa_set_destroy(live, NULL);

   nir_foreach_function(function, shader) {
      if (!function->impl)
         continue;

      if (progress) {
         remove_dead_var_writes(shader);
         nir_metadata_preserve(function->impl, nir_metadata_block_index |
                                               nir_metadata_dominance);
      } else {
         nir_metadata_preserve(function->impl, nir_metadata_all);
      }
   }

   return progress;
}
