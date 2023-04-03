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

/*
 * This lowering pass detects when a global variable is only being used by
 * one function and makes it local to that function
 */

#include "nir.h"

static void
register_var_use(nir_variable *var, nir_function_impl *impl,
                 struct hash_table *var_func_table)
{
   if (var->data.mode != nir_var_shader_temp)
      return;

   struct hash_entry *entry =
      _mesa_hash_table_search(var_func_table, var);

   if (entry) {
      if (entry->data != impl)
         entry->data = NULL;
   } else {
      _mesa_hash_table_insert(var_func_table, var, impl);
   }
}

static bool
mark_global_var_uses_block(nir_block *block, nir_function_impl *impl,
                           struct hash_table *var_func_table)
{
   nir_foreach_instr(instr, block) {
      if (instr->type ==  nir_instr_type_deref) {
         nir_deref_instr *deref = nir_instr_as_deref(instr);
         if (deref->deref_type == nir_deref_type_var)
            register_var_use(deref->var, impl, var_func_table);
      }
   }

   return true;
}

bool
nir_lower_global_vars_to_local(nir_shader *shader)
{
   bool progress = false;

   /* A hash table keyed on variable pointers that stores the unique
    * nir_function_impl that uses the given variable.  If a variable is
    * used in multiple functions, the data for the given key will be NULL.
    */
   struct hash_table *var_func_table = _mesa_pointer_hash_table_create(NULL);

   nir_foreach_function(function, shader) {
      if (function->impl) {
         nir_foreach_block(block, function->impl)
            mark_global_var_uses_block(block, function->impl, var_func_table);
      }
   }

   nir_foreach_variable_with_modes_safe(var, shader, nir_var_shader_temp) {
      struct hash_entry *entry = _mesa_hash_table_search(var_func_table, var);
      if (!entry)
         continue;

      nir_function_impl *impl = entry->data;

      if (impl != NULL) {
         exec_node_remove(&var->node);
         var->data.mode = nir_var_function_temp;
         exec_list_push_tail(&impl->locals, &var->node);
         nir_metadata_preserve(impl, nir_metadata_block_index |
                                     nir_metadata_dominance |
                                     nir_metadata_live_ssa_defs);
         progress = true;
      }
   }

   _mesa_hash_table_destroy(var_func_table, NULL);

   if (progress)
      nir_fixup_deref_modes(shader);

   nir_foreach_function(function, shader) {
      if (function->impl)
         nir_metadata_preserve(function->impl, nir_metadata_all);
   }

   return progress;
}
