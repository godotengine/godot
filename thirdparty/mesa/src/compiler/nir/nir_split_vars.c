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
#include "nir_vla.h"

#include "util/set.h"
#include "util/u_math.h"

static struct set *
get_complex_used_vars(nir_shader *shader, void *mem_ctx)
{
   struct set *complex_vars = _mesa_pointer_set_create(mem_ctx);

   nir_foreach_function(function, shader) {
      if (!function->impl)
         continue;

      nir_foreach_block(block, function->impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_deref)
               continue;

            nir_deref_instr *deref = nir_instr_as_deref(instr);

            /* We only need to consider var derefs because
             * nir_deref_instr_has_complex_use is recursive.
             */
            if (deref->deref_type == nir_deref_type_var &&
                nir_deref_instr_has_complex_use(deref, 0))
               _mesa_set_add(complex_vars, deref->var);
         }
      }
   }

   return complex_vars;
}

struct split_var_state {
   void *mem_ctx;

   nir_shader *shader;
   nir_function_impl *impl;

   nir_variable *base_var;
};

struct field {
   struct field *parent;

   const struct glsl_type *type;

   unsigned num_fields;
   struct field *fields;

   nir_variable *var;
};

static int
num_array_levels_in_array_of_vector_type(const struct glsl_type *type)
{
   int num_levels = 0;
   while (true) {
      if (glsl_type_is_array_or_matrix(type)) {
         num_levels++;
         type = glsl_get_array_element(type);
      } else if (glsl_type_is_vector_or_scalar(type)) {
         return num_levels;
      } else {
         /* Not an array of vectors */
         return -1;
      }
   }
}

static void
init_field_for_type(struct field *field, struct field *parent,
                    const struct glsl_type *type,
                    const char *name,
                    struct split_var_state *state)
{
   *field = (struct field) {
      .parent = parent,
      .type = type,
   };

   const struct glsl_type *struct_type = glsl_without_array(type);
   if (glsl_type_is_struct_or_ifc(struct_type)) {
      field->num_fields = glsl_get_length(struct_type),
      field->fields = ralloc_array(state->mem_ctx, struct field,
                                   field->num_fields);
      for (unsigned i = 0; i < field->num_fields; i++) {
         char *field_name = NULL;
         if (name) {
            field_name = ralloc_asprintf(state->mem_ctx, "%s_%s", name,
                                         glsl_get_struct_elem_name(struct_type, i));
         } else {
            field_name = ralloc_asprintf(state->mem_ctx, "{unnamed %s}_%s",
                                         glsl_get_type_name(struct_type),
                                         glsl_get_struct_elem_name(struct_type, i));
         }
         init_field_for_type(&field->fields[i], field,
                             glsl_get_struct_field(struct_type, i),
                             field_name, state);
      }
   } else {
      const struct glsl_type *var_type = type;
      for (struct field *f = field->parent; f; f = f->parent)
         var_type = glsl_type_wrap_in_arrays(var_type, f->type);

      nir_variable_mode mode = state->base_var->data.mode;
      if (mode == nir_var_function_temp) {
         field->var = nir_local_variable_create(state->impl, var_type, name);
      } else {
         field->var = nir_variable_create(state->shader, mode, var_type, name);
      }
      field->var->data.ray_query = state->base_var->data.ray_query;
   }
}

static bool
split_var_list_structs(nir_shader *shader,
                       nir_function_impl *impl,
                       struct exec_list *vars,
                       nir_variable_mode mode,
                       struct hash_table *var_field_map,
                       struct set **complex_vars,
                       void *mem_ctx)
{
   struct split_var_state state = {
      .mem_ctx = mem_ctx,
      .shader = shader,
      .impl = impl,
   };

   struct exec_list split_vars;
   exec_list_make_empty(&split_vars);

   /* To avoid list confusion (we'll be adding things as we split variables),
    * pull all of the variables we plan to split off of the list
    */
   nir_foreach_variable_in_list_safe(var, vars) {
      if (var->data.mode != mode)
         continue;

      if (!glsl_type_is_struct_or_ifc(glsl_without_array(var->type)))
         continue;

      if (*complex_vars == NULL)
         *complex_vars = get_complex_used_vars(shader, mem_ctx);

      /* We can't split a variable that's referenced with deref that has any
       * sort of complex usage.
       */
      if (_mesa_set_search(*complex_vars, var))
         continue;

      exec_node_remove(&var->node);
      exec_list_push_tail(&split_vars, &var->node);
   }

   nir_foreach_variable_in_list(var, &split_vars) {
      state.base_var = var;

      struct field *root_field = ralloc(mem_ctx, struct field);
      init_field_for_type(root_field, NULL, var->type, var->name, &state);
      _mesa_hash_table_insert(var_field_map, var, root_field);
   }

   return !exec_list_is_empty(&split_vars);
}

static void
split_struct_derefs_impl(nir_function_impl *impl,
                         struct hash_table *var_field_map,
                         nir_variable_mode modes,
                         void *mem_ctx)
{
   nir_builder b;
   nir_builder_init(&b, impl);

   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_deref)
            continue;

         nir_deref_instr *deref = nir_instr_as_deref(instr);
         if (!nir_deref_mode_may_be(deref, modes))
            continue;

         /* Clean up any dead derefs we find lying around.  They may refer to
          * variables we're planning to split.
          */
         if (nir_deref_instr_remove_if_unused(deref))
            continue;

         if (!glsl_type_is_vector_or_scalar(deref->type))
            continue;

         nir_variable *base_var = nir_deref_instr_get_variable(deref);
         /* If we can't chase back to the variable, then we're a complex use.
          * This should have been detected by get_complex_used_vars() and the
          * variable should not have been split.  However, we have no way of
          * knowing that here, so we just have to trust it.
          */
         if (base_var == NULL)
            continue;

         struct hash_entry *entry =
            _mesa_hash_table_search(var_field_map, base_var);
         if (!entry)
            continue;

         struct field *root_field = entry->data;

         nir_deref_path path;
         nir_deref_path_init(&path, deref, mem_ctx);

         struct field *tail_field = root_field;
         for (unsigned i = 0; path.path[i]; i++) {
            if (path.path[i]->deref_type != nir_deref_type_struct)
               continue;

            assert(i > 0);
            assert(glsl_type_is_struct_or_ifc(path.path[i - 1]->type));
            assert(path.path[i - 1]->type ==
                   glsl_without_array(tail_field->type));

            tail_field = &tail_field->fields[path.path[i]->strct.index];
         }
         nir_variable *split_var = tail_field->var;

         nir_deref_instr *new_deref = NULL;
         for (unsigned i = 0; path.path[i]; i++) {
            nir_deref_instr *p = path.path[i];
            b.cursor = nir_after_instr(&p->instr);

            switch (p->deref_type) {
            case nir_deref_type_var:
               assert(new_deref == NULL);
               new_deref = nir_build_deref_var(&b, split_var);
               break;

            case nir_deref_type_array:
            case nir_deref_type_array_wildcard:
               new_deref = nir_build_deref_follower(&b, new_deref, p);
               break;

            case nir_deref_type_struct:
               /* Nothing to do; we're splitting structs */
               break;

            default:
               unreachable("Invalid deref type in path");
            }
         }

         assert(new_deref->type == deref->type);
         nir_ssa_def_rewrite_uses(&deref->dest.ssa,
                                  &new_deref->dest.ssa);
         nir_deref_instr_remove_if_unused(deref);
      }
   }
}

/** A pass for splitting structs into multiple variables
 *
 * This pass splits arrays of structs into multiple variables, one for each
 * (possibly nested) structure member.  After this pass completes, no
 * variables of the given mode will contain a struct type.
 */
bool
nir_split_struct_vars(nir_shader *shader, nir_variable_mode modes)
{
   void *mem_ctx = ralloc_context(NULL);
   struct hash_table *var_field_map =
      _mesa_pointer_hash_table_create(mem_ctx);
   struct set *complex_vars = NULL;

   assert((modes & (nir_var_shader_temp | nir_var_ray_hit_attrib | nir_var_function_temp)) == modes);

   bool has_global_splits = false;
   nir_variable_mode global_modes = modes & (nir_var_shader_temp | nir_var_ray_hit_attrib);
   if (global_modes) {
      has_global_splits = split_var_list_structs(shader, NULL,
                                                 &shader->variables,
                                                 global_modes,
                                                 var_field_map,
                                                 &complex_vars,
                                                 mem_ctx);
   }

   bool progress = false;
   nir_foreach_function(function, shader) {
      if (!function->impl)
         continue;

      bool has_local_splits = false;
      if (modes & nir_var_function_temp) {
         has_local_splits = split_var_list_structs(shader, function->impl,
                                                   &function->impl->locals,
                                                   nir_var_function_temp,
                                                   var_field_map,
                                                   &complex_vars,
                                                   mem_ctx);
      }

      if (has_global_splits || has_local_splits) {
         split_struct_derefs_impl(function->impl, var_field_map,
                                  modes, mem_ctx);

         nir_metadata_preserve(function->impl, nir_metadata_block_index |
                                               nir_metadata_dominance);
         progress = true;
      } else {
         nir_metadata_preserve(function->impl, nir_metadata_all);
      }
   }

   ralloc_free(mem_ctx);

   return progress;
}

struct array_level_info {
   unsigned array_len;
   bool split;
};

struct array_split {
   /* Only set if this is the tail end of the splitting */
   nir_variable *var;

   unsigned num_splits;
   struct array_split *splits;
};

struct array_var_info {
   nir_variable *base_var;

   const struct glsl_type *split_var_type;

   bool split_var;
   struct array_split root_split;

   unsigned num_levels;
   struct array_level_info levels[0];
};

static bool
init_var_list_array_infos(nir_shader *shader,
                          struct exec_list *vars,
                          nir_variable_mode mode,
                          struct hash_table *var_info_map,
                          struct set **complex_vars,
                          void *mem_ctx)
{
   bool has_array = false;

   nir_foreach_variable_in_list(var, vars) {
      if (var->data.mode != mode)
         continue;

      int num_levels = num_array_levels_in_array_of_vector_type(var->type);
      if (num_levels <= 0)
         continue;

      if (*complex_vars == NULL)
         *complex_vars = get_complex_used_vars(shader, mem_ctx);

      /* We can't split a variable that's referenced with deref that has any
       * sort of complex usage.
       */
      if (_mesa_set_search(*complex_vars, var))
         continue;

      struct array_var_info *info =
         rzalloc_size(mem_ctx, sizeof(*info) +
                               num_levels * sizeof(info->levels[0]));

      info->base_var = var;
      info->num_levels = num_levels;

      const struct glsl_type *type = var->type;
      for (int i = 0; i < num_levels; i++) {
         info->levels[i].array_len = glsl_get_length(type);
         type = glsl_get_array_element(type);

         /* All levels start out initially as split */
         info->levels[i].split = true;
      }

      _mesa_hash_table_insert(var_info_map, var, info);
      has_array = true;
   }

   return has_array;
}

static struct array_var_info *
get_array_var_info(nir_variable *var,
                   struct hash_table *var_info_map)
{
   struct hash_entry *entry =
      _mesa_hash_table_search(var_info_map, var);
   return entry ? entry->data : NULL;
}

static struct array_var_info *
get_array_deref_info(nir_deref_instr *deref,
                     struct hash_table *var_info_map,
                     nir_variable_mode modes)
{
   if (!nir_deref_mode_may_be(deref, modes))
      return NULL;

   nir_variable *var = nir_deref_instr_get_variable(deref);
   if (var == NULL)
      return NULL;

   return get_array_var_info(var, var_info_map);
}

static void
mark_array_deref_used(nir_deref_instr *deref,
                      struct hash_table *var_info_map,
                      nir_variable_mode modes,
                      void *mem_ctx)
{
   struct array_var_info *info =
      get_array_deref_info(deref, var_info_map, modes);
   if (!info)
      return;

   nir_deref_path path;
   nir_deref_path_init(&path, deref, mem_ctx);

   /* Walk the path and look for indirects.  If we have an array deref with an
    * indirect, mark the given level as not being split.
    */
   for (unsigned i = 0; i < info->num_levels; i++) {
      nir_deref_instr *p = path.path[i + 1];
      if (p->deref_type == nir_deref_type_array &&
          !nir_src_is_const(p->arr.index))
         info->levels[i].split = false;
   }
}

static void
mark_array_usage_impl(nir_function_impl *impl,
                      struct hash_table *var_info_map,
                      nir_variable_mode modes,
                      void *mem_ctx)
{
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;

         nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
         switch (intrin->intrinsic) {
         case nir_intrinsic_copy_deref:
            mark_array_deref_used(nir_src_as_deref(intrin->src[1]),
                                  var_info_map, modes, mem_ctx);
            FALLTHROUGH;

         case nir_intrinsic_load_deref:
         case nir_intrinsic_store_deref:
            mark_array_deref_used(nir_src_as_deref(intrin->src[0]),
                                  var_info_map, modes, mem_ctx);
            break;

         default:
            break;
         }
      }
   }
}

static void
create_split_array_vars(struct array_var_info *var_info,
                        unsigned level,
                        struct array_split *split,
                        const char *name,
                        nir_shader *shader,
                        nir_function_impl *impl,
                        void *mem_ctx)
{
   while (level < var_info->num_levels && !var_info->levels[level].split) {
      name = ralloc_asprintf(mem_ctx, "%s[*]", name);
      level++;
   }

   if (level == var_info->num_levels) {
      /* We add parens to the variable name so it looks like "(foo[2][*])" so
       * that further derefs will look like "(foo[2][*])[ssa_6]"
       */
      name = ralloc_asprintf(mem_ctx, "(%s)", name);

      nir_variable_mode mode = var_info->base_var->data.mode;
      if (mode == nir_var_function_temp) {
         split->var = nir_local_variable_create(impl,
                                                var_info->split_var_type, name);
      } else {
         split->var = nir_variable_create(shader, mode,
                                          var_info->split_var_type, name);
      }
      split->var->data.ray_query = var_info->base_var->data.ray_query;
   } else {
      assert(var_info->levels[level].split);
      split->num_splits = var_info->levels[level].array_len;
      split->splits = rzalloc_array(mem_ctx, struct array_split,
                                    split->num_splits);
      for (unsigned i = 0; i < split->num_splits; i++) {
         create_split_array_vars(var_info, level + 1, &split->splits[i],
                                 ralloc_asprintf(mem_ctx, "%s[%d]", name, i),
                                 shader, impl, mem_ctx);
      }
   }
}

static bool
split_var_list_arrays(nir_shader *shader,
                      nir_function_impl *impl,
                      struct exec_list *vars,
                      nir_variable_mode mode,
                      struct hash_table *var_info_map,
                      void *mem_ctx)
{
   struct exec_list split_vars;
   exec_list_make_empty(&split_vars);

   nir_foreach_variable_in_list_safe(var, vars) {
      if (var->data.mode != mode)
         continue;

      struct array_var_info *info = get_array_var_info(var, var_info_map);
      if (!info)
         continue;

      bool has_split = false;
      const struct glsl_type *split_type =
         glsl_without_array_or_matrix(var->type);
      for (int i = info->num_levels - 1; i >= 0; i--) {
         if (info->levels[i].split) {
            has_split = true;
            continue;
         }

         /* If the original type was a matrix type, we'd like to keep that so
          * we don't convert matrices into arrays.
          */
         if (i == info->num_levels - 1 &&
             glsl_type_is_matrix(glsl_without_array(var->type))) {
            split_type = glsl_matrix_type(glsl_get_base_type(split_type),
                                          glsl_get_components(split_type),
                                          info->levels[i].array_len);
         } else {
            split_type = glsl_array_type(split_type, info->levels[i].array_len, 0);
         }
      }

      if (has_split) {
         info->split_var_type = split_type;
         /* To avoid list confusion (we'll be adding things as we split
          * variables), pull all of the variables we plan to split off of the
          * main variable list.
          */
         exec_node_remove(&var->node);
         exec_list_push_tail(&split_vars, &var->node);
      } else {
         assert(split_type == glsl_get_bare_type(var->type));
         /* If we're not modifying this variable, delete the info so we skip
          * it faster in later passes.
          */
         _mesa_hash_table_remove_key(var_info_map, var);
      }
   }

   nir_foreach_variable_in_list(var, &split_vars) {
      struct array_var_info *info = get_array_var_info(var, var_info_map);
      create_split_array_vars(info, 0, &info->root_split, var->name,
                              shader, impl, mem_ctx);
   }

   return !exec_list_is_empty(&split_vars);
}

static bool
deref_has_split_wildcard(nir_deref_path *path,
                         struct array_var_info *info)
{
   if (info == NULL)
      return false;

   assert(path->path[0]->var == info->base_var);
   for (unsigned i = 0; i < info->num_levels; i++) {
      if (path->path[i + 1]->deref_type == nir_deref_type_array_wildcard &&
          info->levels[i].split)
         return true;
   }

   return false;
}

static bool
array_path_is_out_of_bounds(nir_deref_path *path,
                            struct array_var_info *info)
{
   if (info == NULL)
      return false;

   assert(path->path[0]->var == info->base_var);
   for (unsigned i = 0; i < info->num_levels; i++) {
      nir_deref_instr *p = path->path[i + 1];
      if (p->deref_type == nir_deref_type_array_wildcard)
         continue;

      if (nir_src_is_const(p->arr.index) &&
          nir_src_as_uint(p->arr.index) >= info->levels[i].array_len)
         return true;
   }

   return false;
}

static void
emit_split_copies(nir_builder *b,
                  struct array_var_info *dst_info, nir_deref_path *dst_path,
                  unsigned dst_level, nir_deref_instr *dst,
                  struct array_var_info *src_info, nir_deref_path *src_path,
                  unsigned src_level, nir_deref_instr *src)
{
   nir_deref_instr *dst_p, *src_p;

   while ((dst_p = dst_path->path[dst_level + 1])) {
      if (dst_p->deref_type == nir_deref_type_array_wildcard)
         break;

      dst = nir_build_deref_follower(b, dst, dst_p);
      dst_level++;
   }

   while ((src_p = src_path->path[src_level + 1])) {
      if (src_p->deref_type == nir_deref_type_array_wildcard)
         break;

      src = nir_build_deref_follower(b, src, src_p);
      src_level++;
   }

   if (src_p == NULL || dst_p == NULL) {
      assert(src_p == NULL && dst_p == NULL);
      nir_copy_deref(b, dst, src);
   } else {
      assert(dst_p->deref_type == nir_deref_type_array_wildcard &&
             src_p->deref_type == nir_deref_type_array_wildcard);

      if ((dst_info && dst_info->levels[dst_level].split) ||
          (src_info && src_info->levels[src_level].split)) {
         /* There are no indirects at this level on one of the source or the
          * destination so we are lowering it.
          */
         assert(glsl_get_length(dst_path->path[dst_level]->type) ==
                glsl_get_length(src_path->path[src_level]->type));
         unsigned len = glsl_get_length(dst_path->path[dst_level]->type);
         for (unsigned i = 0; i < len; i++) {
            emit_split_copies(b, dst_info, dst_path, dst_level + 1,
                              nir_build_deref_array_imm(b, dst, i),
                              src_info, src_path, src_level + 1,
                              nir_build_deref_array_imm(b, src, i));
         }
      } else {
         /* Neither side is being split so we just keep going */
         emit_split_copies(b, dst_info, dst_path, dst_level + 1,
                           nir_build_deref_array_wildcard(b, dst),
                           src_info, src_path, src_level + 1,
                           nir_build_deref_array_wildcard(b, src));
      }
   }
}

static void
split_array_copies_impl(nir_function_impl *impl,
                        struct hash_table *var_info_map,
                        nir_variable_mode modes,
                        void *mem_ctx)
{
   nir_builder b;
   nir_builder_init(&b, impl);

   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;

         nir_intrinsic_instr *copy = nir_instr_as_intrinsic(instr);
         if (copy->intrinsic != nir_intrinsic_copy_deref)
            continue;

         nir_deref_instr *dst_deref = nir_src_as_deref(copy->src[0]);
         nir_deref_instr *src_deref = nir_src_as_deref(copy->src[1]);

         struct array_var_info *dst_info =
            get_array_deref_info(dst_deref, var_info_map, modes);
         struct array_var_info *src_info =
            get_array_deref_info(src_deref, var_info_map, modes);

         if (!src_info && !dst_info)
            continue;

         nir_deref_path dst_path, src_path;
         nir_deref_path_init(&dst_path, dst_deref, mem_ctx);
         nir_deref_path_init(&src_path, src_deref, mem_ctx);

         if (!deref_has_split_wildcard(&dst_path, dst_info) &&
             !deref_has_split_wildcard(&src_path, src_info))
            continue;

         b.cursor = nir_instr_remove(&copy->instr);

         emit_split_copies(&b, dst_info, &dst_path, 0, dst_path.path[0],
                               src_info, &src_path, 0, src_path.path[0]);
      }
   }
}

static void
split_array_access_impl(nir_function_impl *impl,
                        struct hash_table *var_info_map,
                        nir_variable_mode modes,
                        void *mem_ctx)
{
   nir_builder b;
   nir_builder_init(&b, impl);

   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type == nir_instr_type_deref) {
            /* Clean up any dead derefs we find lying around.  They may refer
             * to variables we're planning to split.
             */
            nir_deref_instr *deref = nir_instr_as_deref(instr);
            if (nir_deref_mode_may_be(deref, modes))
               nir_deref_instr_remove_if_unused(deref);
            continue;
         }

         if (instr->type != nir_instr_type_intrinsic)
            continue;

         nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
         if (intrin->intrinsic != nir_intrinsic_load_deref &&
             intrin->intrinsic != nir_intrinsic_store_deref &&
             intrin->intrinsic != nir_intrinsic_copy_deref)
            continue;

         const unsigned num_derefs =
            intrin->intrinsic == nir_intrinsic_copy_deref ? 2 : 1;

         for (unsigned d = 0; d < num_derefs; d++) {
            nir_deref_instr *deref = nir_src_as_deref(intrin->src[d]);

            struct array_var_info *info =
               get_array_deref_info(deref, var_info_map, modes);
            if (!info)
               continue;

            nir_deref_path path;
            nir_deref_path_init(&path, deref, mem_ctx);

            b.cursor = nir_before_instr(&intrin->instr);

            if (array_path_is_out_of_bounds(&path, info)) {
               /* If one of the derefs is out-of-bounds, we just delete the
                * instruction.  If a destination is out of bounds, then it may
                * have been in-bounds prior to shrinking so we don't want to
                * accidentally stomp something.  However, we've already proven
                * that it will never be read so it's safe to delete.  If a
                * source is out of bounds then it is loading random garbage.
                * For loads, we replace their uses with an undef instruction
                * and for copies we just delete the copy since it was writing
                * undefined garbage anyway and we may as well leave the random
                * garbage in the destination alone.
                */
               if (intrin->intrinsic == nir_intrinsic_load_deref) {
                  nir_ssa_def *u =
                     nir_ssa_undef(&b, intrin->dest.ssa.num_components,
                                       intrin->dest.ssa.bit_size);
                  nir_ssa_def_rewrite_uses(&intrin->dest.ssa,
                                           u);
               }
               nir_instr_remove(&intrin->instr);
               for (unsigned i = 0; i < num_derefs; i++)
                  nir_deref_instr_remove_if_unused(nir_src_as_deref(intrin->src[i]));
               break;
            }

            struct array_split *split = &info->root_split;
            for (unsigned i = 0; i < info->num_levels; i++) {
               if (info->levels[i].split) {
                  nir_deref_instr *p = path.path[i + 1];
                  unsigned index = nir_src_as_uint(p->arr.index);
                  assert(index < info->levels[i].array_len);
                  split = &split->splits[index];
               }
            }
            assert(!split->splits && split->var);

            nir_deref_instr *new_deref = nir_build_deref_var(&b, split->var);
            for (unsigned i = 0; i < info->num_levels; i++) {
               if (!info->levels[i].split) {
                  new_deref = nir_build_deref_follower(&b, new_deref,
                                                       path.path[i + 1]);
               }
            }
            assert(new_deref->type == deref->type);

            /* Rewrite the deref source to point to the split one */
            nir_instr_rewrite_src(&intrin->instr, &intrin->src[d],
                                  nir_src_for_ssa(&new_deref->dest.ssa));
            nir_deref_instr_remove_if_unused(deref);
         }
      }
   }
}

/** A pass for splitting arrays of vectors into multiple variables
 *
 * This pass looks at arrays (possibly multiple levels) of vectors (not
 * structures or other types) and tries to split them into piles of variables,
 * one for each array element.  The heuristic used is simple: If a given array
 * level is never used with an indirect, that array level will get split.
 *
 * This pass probably could handles structures easily enough but making a pass
 * that could see through an array of structures of arrays would be difficult
 * so it's best to just run nir_split_struct_vars first.
 */
bool
nir_split_array_vars(nir_shader *shader, nir_variable_mode modes)
{
   void *mem_ctx = ralloc_context(NULL);
   struct hash_table *var_info_map = _mesa_pointer_hash_table_create(mem_ctx);
   struct set *complex_vars = NULL;

   assert((modes & (nir_var_shader_temp | nir_var_ray_hit_attrib | nir_var_function_temp)) == modes);

   bool has_global_array = false;
   if (modes & (nir_var_shader_temp | nir_var_ray_hit_attrib)) {
      has_global_array = init_var_list_array_infos(shader,
                                                   &shader->variables,
                                                   modes,
                                                   var_info_map,
                                                   &complex_vars,
                                                   mem_ctx);
   }

   bool has_any_array = false;
   nir_foreach_function(function, shader) {
      if (!function->impl)
         continue;

      bool has_local_array = false;
      if (modes & nir_var_function_temp) {
         has_local_array = init_var_list_array_infos(shader,
                                                     &function->impl->locals,
                                                     nir_var_function_temp,
                                                     var_info_map,
                                                     &complex_vars,
                                                     mem_ctx);
      }

      if (has_global_array || has_local_array) {
         has_any_array = true;
         mark_array_usage_impl(function->impl, var_info_map, modes, mem_ctx);
      }
   }

   /* If we failed to find any arrays of arrays, bail early. */
   if (!has_any_array) {
      ralloc_free(mem_ctx);
      nir_shader_preserve_all_metadata(shader);
      return false;
   }

   bool has_global_splits = false;
   if (modes & (nir_var_shader_temp | nir_var_ray_hit_attrib)) {
      has_global_splits = split_var_list_arrays(shader, NULL,
                                                &shader->variables,
                                                modes,
                                                var_info_map, mem_ctx);
   }

   bool progress = false;
   nir_foreach_function(function, shader) {
      if (!function->impl)
         continue;

      bool has_local_splits = false;
      if (modes & nir_var_function_temp) {
         has_local_splits = split_var_list_arrays(shader, function->impl,
                                                  &function->impl->locals,
                                                  nir_var_function_temp,
                                                  var_info_map, mem_ctx);
      }

      if (has_global_splits || has_local_splits) {
         split_array_copies_impl(function->impl, var_info_map, modes, mem_ctx);
         split_array_access_impl(function->impl, var_info_map, modes, mem_ctx);

         nir_metadata_preserve(function->impl, nir_metadata_block_index |
                                               nir_metadata_dominance);
         progress = true;
      } else {
         nir_metadata_preserve(function->impl, nir_metadata_all);
      }
   }

   ralloc_free(mem_ctx);

   return progress;
}

struct array_level_usage {
   unsigned array_len;

   /* The value UINT_MAX will be used to indicate an indirect */
   unsigned max_read;
   unsigned max_written;

   /* True if there is a copy that isn't to/from a shrinkable array */
   bool has_external_copy;
   struct set *levels_copied;
};

struct vec_var_usage {
   /* Convenience set of all components this variable has */
   nir_component_mask_t all_comps;

   nir_component_mask_t comps_read;
   nir_component_mask_t comps_written;

   nir_component_mask_t comps_kept;

   /* True if there is a copy that isn't to/from a shrinkable vector */
   bool has_external_copy;
   bool has_complex_use;
   struct set *vars_copied;

   unsigned num_levels;
   struct array_level_usage levels[0];
};

static struct vec_var_usage *
get_vec_var_usage(nir_variable *var,
                  struct hash_table *var_usage_map,
                  bool add_usage_entry, void *mem_ctx)
{
   struct hash_entry *entry = _mesa_hash_table_search(var_usage_map, var);
   if (entry)
      return entry->data;

   if (!add_usage_entry)
      return NULL;

   /* Check to make sure that we are working with an array of vectors.  We
    * don't bother to shrink single vectors because we figure that we can
    * clean it up better with SSA than by inserting piles of vecN instructions
    * to compact results.
    */
   int num_levels = num_array_levels_in_array_of_vector_type(var->type);
   if (num_levels < 1)
      return NULL; /* Not an array of vectors */

   struct vec_var_usage *usage =
      rzalloc_size(mem_ctx, sizeof(*usage) +
                            num_levels * sizeof(usage->levels[0]));

   usage->num_levels = num_levels;
   const struct glsl_type *type = var->type;
   for (unsigned i = 0; i < num_levels; i++) {
      usage->levels[i].array_len = glsl_get_length(type);
      type = glsl_get_array_element(type);
   }
   assert(glsl_type_is_vector_or_scalar(type));

   usage->all_comps = (1 << glsl_get_components(type)) - 1;

   _mesa_hash_table_insert(var_usage_map, var, usage);

   return usage;
}

static struct vec_var_usage *
get_vec_deref_usage(nir_deref_instr *deref,
                    struct hash_table *var_usage_map,
                    nir_variable_mode modes,
                    bool add_usage_entry, void *mem_ctx)
{
   if (!nir_deref_mode_may_be(deref, modes))
      return NULL;

   nir_variable *var = nir_deref_instr_get_variable(deref);
   if (var == NULL)
      return NULL;

   return get_vec_var_usage(nir_deref_instr_get_variable(deref),
                            var_usage_map, add_usage_entry, mem_ctx);
}

static void
mark_deref_if_complex(nir_deref_instr *deref,
                      struct hash_table *var_usage_map,
                      nir_variable_mode modes,
                      void *mem_ctx)
{
   /* Only bother with var derefs because nir_deref_instr_has_complex_use is
    * recursive.
    */
   if (deref->deref_type != nir_deref_type_var)
      return;

   if (!(deref->var->data.mode & modes))
      return;

   if (!nir_deref_instr_has_complex_use(deref, 0))
      return;

   struct vec_var_usage *usage =
      get_vec_var_usage(deref->var, var_usage_map, true, mem_ctx);
   if (!usage)
      return;

   usage->has_complex_use = true;
}

static void
mark_deref_used(nir_deref_instr *deref,
                nir_component_mask_t comps_read,
                nir_component_mask_t comps_written,
                nir_deref_instr *copy_deref,
                struct hash_table *var_usage_map,
                nir_variable_mode modes,
                void *mem_ctx)
{
   if (!nir_deref_mode_may_be(deref, modes))
      return;

   nir_variable *var = nir_deref_instr_get_variable(deref);
   if (var == NULL)
      return;

   struct vec_var_usage *usage =
      get_vec_var_usage(var, var_usage_map, true, mem_ctx);
   if (!usage)
      return;

   usage->comps_read |= comps_read & usage->all_comps;
   usage->comps_written |= comps_written & usage->all_comps;

   struct vec_var_usage *copy_usage = NULL;
   if (copy_deref) {
      copy_usage = get_vec_deref_usage(copy_deref, var_usage_map, modes,
                                       true, mem_ctx);
      if (copy_usage) {
         if (usage->vars_copied == NULL) {
            usage->vars_copied = _mesa_pointer_set_create(mem_ctx);
         }
         _mesa_set_add(usage->vars_copied, copy_usage);
      } else {
         usage->has_external_copy = true;
      }
   }

   nir_deref_path path;
   nir_deref_path_init(&path, deref, mem_ctx);

   nir_deref_path copy_path;
   if (copy_usage)
      nir_deref_path_init(&copy_path, copy_deref, mem_ctx);

   unsigned copy_i = 0;
   for (unsigned i = 0; i < usage->num_levels; i++) {
      struct array_level_usage *level = &usage->levels[i];
      nir_deref_instr *deref = path.path[i + 1];
      assert(deref->deref_type == nir_deref_type_array ||
             deref->deref_type == nir_deref_type_array_wildcard);

      unsigned max_used;
      if (deref->deref_type == nir_deref_type_array) {
         max_used = nir_src_is_const(deref->arr.index) ?
                    nir_src_as_uint(deref->arr.index) : UINT_MAX;
      } else {
         /* For wildcards, we read or wrote the whole thing. */
         assert(deref->deref_type == nir_deref_type_array_wildcard);
         max_used = level->array_len - 1;

         if (copy_usage) {
            /* Match each wildcard level with the level on copy_usage */
            for (; copy_path.path[copy_i + 1]; copy_i++) {
               if (copy_path.path[copy_i + 1]->deref_type ==
                   nir_deref_type_array_wildcard)
                  break;
            }
            struct array_level_usage *copy_level =
               &copy_usage->levels[copy_i++];

            if (level->levels_copied == NULL) {
               level->levels_copied = _mesa_pointer_set_create(mem_ctx);
            }
            _mesa_set_add(level->levels_copied, copy_level);
         } else {
            /* We have a wildcard and it comes from a variable we aren't
             * tracking; flag it and we'll know to not shorten this array.
             */
            level->has_external_copy = true;
         }
      }

      if (comps_written)
         level->max_written = MAX2(level->max_written, max_used);
      if (comps_read)
         level->max_read = MAX2(level->max_read, max_used);
   }
}

static bool
src_is_load_deref(nir_src src, nir_src deref_src)
{
   nir_intrinsic_instr *load = nir_src_as_intrinsic(src);
   if (load == NULL || load->intrinsic != nir_intrinsic_load_deref)
      return false;

   assert(load->src[0].is_ssa);

   return load->src[0].ssa == deref_src.ssa;
}

/* Returns all non-self-referential components of a store instruction.  A
 * component is self-referential if it comes from the same component of a load
 * instruction on the same deref.  If the only data in a particular component
 * of a variable came directly from that component then it's undefined.  The
 * only way to get defined data into a component of a variable is for it to
 * get written there by something outside or from a different component.
 *
 * This is a fairly common pattern in shaders that come from either GLSL IR or
 * GLSLang because both glsl_to_nir and GLSLang implement write-masking with
 * load-vec-store.
 */
static nir_component_mask_t
get_non_self_referential_store_comps(nir_intrinsic_instr *store)
{
   nir_component_mask_t comps = nir_intrinsic_write_mask(store);

   assert(store->src[1].is_ssa);
   nir_instr *src_instr = store->src[1].ssa->parent_instr;
   if (src_instr->type != nir_instr_type_alu)
      return comps;

   nir_alu_instr *src_alu = nir_instr_as_alu(src_instr);

   if (src_alu->op == nir_op_mov) {
      /* If it's just a swizzle of a load from the same deref, discount any
       * channels that don't move in the swizzle.
       */
      if (src_is_load_deref(src_alu->src[0].src, store->src[0])) {
         for (unsigned i = 0; i < NIR_MAX_VEC_COMPONENTS; i++) {
            if (src_alu->src[0].swizzle[i] == i)
               comps &= ~(1u << i);
         }
      }
   } else if (nir_op_is_vec(src_alu->op)) {
      /* If it's a vec, discount any channels that are just loads from the
       * same deref put in the same spot.
       */
      for (unsigned i = 0; i < nir_op_infos[src_alu->op].num_inputs; i++) {
         if (src_is_load_deref(src_alu->src[i].src, store->src[0]) &&
             src_alu->src[i].swizzle[0] == i)
            comps &= ~(1u << i);
      }
   }

   return comps;
}

static void
find_used_components_impl(nir_function_impl *impl,
                          struct hash_table *var_usage_map,
                          nir_variable_mode modes,
                          void *mem_ctx)
{
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_deref) {
            mark_deref_if_complex(nir_instr_as_deref(instr),
                                  var_usage_map, modes, mem_ctx);
         }

         if (instr->type != nir_instr_type_intrinsic)
            continue;

         nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
         switch (intrin->intrinsic) {
         case nir_intrinsic_load_deref:
            mark_deref_used(nir_src_as_deref(intrin->src[0]),
                            nir_ssa_def_components_read(&intrin->dest.ssa), 0,
                            NULL, var_usage_map, modes, mem_ctx);
            break;

         case nir_intrinsic_store_deref:
            mark_deref_used(nir_src_as_deref(intrin->src[0]),
                            0, get_non_self_referential_store_comps(intrin),
                            NULL, var_usage_map, modes, mem_ctx);
            break;

         case nir_intrinsic_copy_deref: {
            /* Just mark everything used for copies. */
            nir_deref_instr *dst = nir_src_as_deref(intrin->src[0]);
            nir_deref_instr *src = nir_src_as_deref(intrin->src[1]);
            mark_deref_used(dst, 0, ~0, src, var_usage_map, modes, mem_ctx);
            mark_deref_used(src, ~0, 0, dst, var_usage_map, modes, mem_ctx);
            break;
         }

         default:
            break;
         }
      }
   }
}

static bool
shrink_vec_var_list(struct exec_list *vars,
                    nir_variable_mode mode,
                    struct hash_table *var_usage_map)
{
   /* Initialize the components kept field of each variable.  This is the
    * AND of the components written and components read.  If a component is
    * written but never read, it's dead.  If it is read but never written,
    * then all values read are undefined garbage and we may as well not read
    * them.
    *
    * The same logic applies to the array length.  We make the array length
    * the minimum needed required length between read and write and plan to
    * discard any OOB access.  The one exception here is indirect writes
    * because we don't know where they will land and we can't shrink an array
    * with indirect writes because previously in-bounds writes may become
    * out-of-bounds and have undefined behavior.
    *
    * Also, if we have a copy that to/from something we can't shrink, we need
    * to leave components and array_len of any wildcards alone.
    */
   nir_foreach_variable_in_list(var, vars) {
      if (var->data.mode != mode)
         continue;

      struct vec_var_usage *usage =
         get_vec_var_usage(var, var_usage_map, false, NULL);
      if (!usage)
         continue;

      assert(usage->comps_kept == 0);
      if (usage->has_external_copy || usage->has_complex_use)
         usage->comps_kept = usage->all_comps;
      else
         usage->comps_kept = usage->comps_read & usage->comps_written;

      for (unsigned i = 0; i < usage->num_levels; i++) {
         struct array_level_usage *level = &usage->levels[i];
         assert(level->array_len > 0);

         if (level->max_written == UINT_MAX || level->has_external_copy ||
             usage->has_complex_use)
            continue; /* Can't shrink */

         unsigned max_used = MIN2(level->max_read, level->max_written);
         level->array_len = MIN2(max_used, level->array_len - 1) + 1;
      }
   }

   /* In order for variable copies to work, we have to have the same data type
    * on the source and the destination.  In order to satisfy this, we run a
    * little fixed-point algorithm to transitively ensure that we get enough
    * components and array elements for this to hold for all copies.
    */
   bool fp_progress;
   do {
      fp_progress = false;
      nir_foreach_variable_in_list(var, vars) {
         if (var->data.mode != mode)
            continue;

         struct vec_var_usage *var_usage =
            get_vec_var_usage(var, var_usage_map, false, NULL);
         if (!var_usage || !var_usage->vars_copied)
            continue;

         set_foreach(var_usage->vars_copied, copy_entry) {
            struct vec_var_usage *copy_usage = (void *)copy_entry->key;
            if (copy_usage->comps_kept != var_usage->comps_kept) {
               nir_component_mask_t comps_kept =
                  (var_usage->comps_kept | copy_usage->comps_kept);
               var_usage->comps_kept = comps_kept;
               copy_usage->comps_kept = comps_kept;
               fp_progress = true;
            }
         }

         for (unsigned i = 0; i < var_usage->num_levels; i++) {
            struct array_level_usage *var_level = &var_usage->levels[i];
            if (!var_level->levels_copied)
               continue;

            set_foreach(var_level->levels_copied, copy_entry) {
               struct array_level_usage *copy_level = (void *)copy_entry->key;
               if (var_level->array_len != copy_level->array_len) {
                  unsigned array_len =
                     MAX2(var_level->array_len, copy_level->array_len);
                  var_level->array_len = array_len;
                  copy_level->array_len = array_len;
                  fp_progress = true;
               }
            }
         }
      }
   } while (fp_progress);

   bool vars_shrunk = false;
   nir_foreach_variable_in_list_safe(var, vars) {
      if (var->data.mode != mode)
         continue;

      struct vec_var_usage *usage =
         get_vec_var_usage(var, var_usage_map, false, NULL);
      if (!usage)
         continue;

      bool shrunk = false;
      const struct glsl_type *vec_type = var->type;
      for (unsigned i = 0; i < usage->num_levels; i++) {
         /* If we've reduced the array to zero elements at some level, just
          * set comps_kept to 0 and delete the variable.
          */
         if (usage->levels[i].array_len == 0) {
            usage->comps_kept = 0;
            break;
         }

         assert(usage->levels[i].array_len <= glsl_get_length(vec_type));
         if (usage->levels[i].array_len < glsl_get_length(vec_type))
            shrunk = true;
         vec_type = glsl_get_array_element(vec_type);
      }
      assert(glsl_type_is_vector_or_scalar(vec_type));

      assert(usage->comps_kept == (usage->comps_kept & usage->all_comps));
      if (usage->comps_kept != usage->all_comps)
         shrunk = true;

      if (usage->comps_kept == 0) {
         /* This variable is dead, remove it */
         vars_shrunk = true;
         exec_node_remove(&var->node);
         continue;
      }

      if (!shrunk) {
         /* This variable doesn't need to be shrunk.  Remove it from the
          * hash table so later steps will ignore it.
          */
         _mesa_hash_table_remove_key(var_usage_map, var);
         continue;
      }

      /* Build the new var type */
      unsigned new_num_comps = util_bitcount(usage->comps_kept);
      const struct glsl_type *new_type =
         glsl_vector_type(glsl_get_base_type(vec_type), new_num_comps);
      for (int i = usage->num_levels - 1; i >= 0; i--) {
         assert(usage->levels[i].array_len > 0);
         /* If the original type was a matrix type, we'd like to keep that so
          * we don't convert matrices into arrays.
          */
         if (i == usage->num_levels - 1 &&
             glsl_type_is_matrix(glsl_without_array(var->type)) &&
             new_num_comps > 1 && usage->levels[i].array_len > 1) {
            new_type = glsl_matrix_type(glsl_get_base_type(new_type),
                                        new_num_comps,
                                        usage->levels[i].array_len);
         } else {
            new_type = glsl_array_type(new_type, usage->levels[i].array_len, 0);
         }
      }
      var->type = new_type;

      vars_shrunk = true;
   }

   return vars_shrunk;
}

static bool
vec_deref_is_oob(nir_deref_instr *deref,
                 struct vec_var_usage *usage)
{
   nir_deref_path path;
   nir_deref_path_init(&path, deref, NULL);

   bool oob = false;
   for (unsigned i = 0; i < usage->num_levels; i++) {
      nir_deref_instr *p = path.path[i + 1];
      if (p->deref_type == nir_deref_type_array_wildcard)
         continue;

      if (nir_src_is_const(p->arr.index) &&
          nir_src_as_uint(p->arr.index) >= usage->levels[i].array_len) {
         oob = true;
         break;
      }
   }

   nir_deref_path_finish(&path);

   return oob;
}

static bool
vec_deref_is_dead_or_oob(nir_deref_instr *deref,
                         struct hash_table *var_usage_map,
                         nir_variable_mode modes)
{
   struct vec_var_usage *usage =
      get_vec_deref_usage(deref, var_usage_map, modes, false, NULL);
   if (!usage)
      return false;

   return usage->comps_kept == 0 || vec_deref_is_oob(deref, usage);
}

static void
shrink_vec_var_access_impl(nir_function_impl *impl,
                           struct hash_table *var_usage_map,
                           nir_variable_mode modes)
{
   nir_builder b;
   nir_builder_init(&b, impl);

   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         switch (instr->type) {
         case nir_instr_type_deref: {
            nir_deref_instr *deref = nir_instr_as_deref(instr);
            if (!nir_deref_mode_may_be(deref, modes))
               break;

            /* Clean up any dead derefs we find lying around.  They may refer
             * to variables we've deleted.
             */
            if (nir_deref_instr_remove_if_unused(deref))
               break;

            /* Update the type in the deref to keep the types consistent as
             * you walk down the chain.  We don't need to check if this is one
             * of the derefs we're shrinking because this is a no-op if it
             * isn't.  The worst that could happen is that we accidentally fix
             * an invalid deref.
             */
            if (deref->deref_type == nir_deref_type_var) {
               deref->type = deref->var->type;
            } else if (deref->deref_type == nir_deref_type_array ||
                       deref->deref_type == nir_deref_type_array_wildcard) {
               nir_deref_instr *parent = nir_deref_instr_parent(deref);
               assert(glsl_type_is_array(parent->type) ||
                      glsl_type_is_matrix(parent->type));
               deref->type = glsl_get_array_element(parent->type);
            }
            break;
         }

         case nir_instr_type_intrinsic: {
            nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);

            /* If we have a copy whose source or destination has been deleted
             * because we determined the variable was dead, then we just
             * delete the copy instruction.  If the source variable was dead
             * then it was writing undefined garbage anyway and if it's the
             * destination variable that's dead then the write isn't needed.
             */
            if (intrin->intrinsic == nir_intrinsic_copy_deref) {
               nir_deref_instr *dst = nir_src_as_deref(intrin->src[0]);
               nir_deref_instr *src = nir_src_as_deref(intrin->src[1]);
               if (vec_deref_is_dead_or_oob(dst, var_usage_map, modes) ||
                   vec_deref_is_dead_or_oob(src, var_usage_map, modes)) {
                  nir_instr_remove(&intrin->instr);
                  nir_deref_instr_remove_if_unused(dst);
                  nir_deref_instr_remove_if_unused(src);
               }
               continue;
            }

            if (intrin->intrinsic != nir_intrinsic_load_deref &&
                intrin->intrinsic != nir_intrinsic_store_deref)
               continue;

            nir_deref_instr *deref = nir_src_as_deref(intrin->src[0]);
            if (!nir_deref_mode_may_be(deref, modes))
               continue;

            struct vec_var_usage *usage =
               get_vec_deref_usage(deref, var_usage_map, modes, false, NULL);
            if (!usage)
               continue;

            if (usage->comps_kept == 0 || vec_deref_is_oob(deref, usage)) {
               if (intrin->intrinsic == nir_intrinsic_load_deref) {
                  nir_ssa_def *u =
                     nir_ssa_undef(&b, intrin->dest.ssa.num_components,
                                       intrin->dest.ssa.bit_size);
                  nir_ssa_def_rewrite_uses(&intrin->dest.ssa,
                                           u);
               }
               nir_instr_remove(&intrin->instr);
               nir_deref_instr_remove_if_unused(deref);
               continue;
            }

            /* If we're not dropping any components, there's no need to
             * compact vectors.
             */
            if (usage->comps_kept == usage->all_comps)
               continue;

            if (intrin->intrinsic == nir_intrinsic_load_deref) {
               b.cursor = nir_after_instr(&intrin->instr);

               nir_ssa_def *undef =
                  nir_ssa_undef(&b, 1, intrin->dest.ssa.bit_size);
               nir_ssa_def *vec_srcs[NIR_MAX_VEC_COMPONENTS];
               unsigned c = 0;
               for (unsigned i = 0; i < intrin->num_components; i++) {
                  if (usage->comps_kept & (1u << i))
                     vec_srcs[i] = nir_channel(&b, &intrin->dest.ssa, c++);
                  else
                     vec_srcs[i] = undef;
               }
               nir_ssa_def *vec = nir_vec(&b, vec_srcs, intrin->num_components);

               nir_ssa_def_rewrite_uses_after(&intrin->dest.ssa,
                                              vec,
                                              vec->parent_instr);

               /* The SSA def is now only used by the swizzle.  It's safe to
                * shrink the number of components.
                */
               assert(list_length(&intrin->dest.ssa.uses) == c);
               intrin->num_components = c;
               intrin->dest.ssa.num_components = c;
            } else {
               nir_component_mask_t write_mask =
                  nir_intrinsic_write_mask(intrin);

               unsigned swizzle[NIR_MAX_VEC_COMPONENTS];
               nir_component_mask_t new_write_mask = 0;
               unsigned c = 0;
               for (unsigned i = 0; i < intrin->num_components; i++) {
                  if (usage->comps_kept & (1u << i)) {
                     swizzle[c] = i;
                     if (write_mask & (1u << i))
                        new_write_mask |= 1u << c;
                     c++;
                  }
               }

               b.cursor = nir_before_instr(&intrin->instr);

               nir_ssa_def *swizzled =
                  nir_swizzle(&b, intrin->src[1].ssa, swizzle, c);

               /* Rewrite to use the compacted source */
               nir_instr_rewrite_src(&intrin->instr, &intrin->src[1],
                                     nir_src_for_ssa(swizzled));
               nir_intrinsic_set_write_mask(intrin, new_write_mask);
               intrin->num_components = c;
            }
            break;
         }

         default:
            break;
         }
      }
   }
}

static bool
function_impl_has_vars_with_modes(nir_function_impl *impl,
                                  nir_variable_mode modes)
{
   nir_shader *shader = impl->function->shader;

   if (modes & ~nir_var_function_temp) {
      nir_foreach_variable_with_modes(var, shader,
                                      modes & ~nir_var_function_temp)
         return true;
   }

   if ((modes & nir_var_function_temp) && !exec_list_is_empty(&impl->locals))
      return true;

   return false;
}

/** Attempt to shrink arrays of vectors
 *
 * This pass looks at variables which contain a vector or an array (possibly
 * multiple dimensions) of vectors and attempts to lower to a smaller vector
 * or array.  If the pass can prove that a component of a vector (or array of
 * vectors) is never really used, then that component will be removed.
 * Similarly, the pass attempts to shorten arrays based on what elements it
 * can prove are never read or never contain valid data.
 */
bool
nir_shrink_vec_array_vars(nir_shader *shader, nir_variable_mode modes)
{
   assert((modes & (nir_var_shader_temp | nir_var_function_temp)) == modes);

   void *mem_ctx = ralloc_context(NULL);

   struct hash_table *var_usage_map =
      _mesa_pointer_hash_table_create(mem_ctx);

   bool has_vars_to_shrink = false;
   nir_foreach_function(function, shader) {
      if (!function->impl)
         continue;

      /* Don't even bother crawling the IR if we don't have any variables.
       * Given that this pass deletes any unused variables, it's likely that
       * we will be in this scenario eventually.
       */
      if (function_impl_has_vars_with_modes(function->impl, modes)) {
         has_vars_to_shrink = true;
         find_used_components_impl(function->impl, var_usage_map,
                                   modes, mem_ctx);
      }
   }
   if (!has_vars_to_shrink) {
      ralloc_free(mem_ctx);
      nir_shader_preserve_all_metadata(shader);
      return false;
   }

   bool globals_shrunk = false;
   if (modes & nir_var_shader_temp) {
      globals_shrunk = shrink_vec_var_list(&shader->variables,
                                           nir_var_shader_temp,
                                           var_usage_map);
   }

   bool progress = false;
   nir_foreach_function(function, shader) {
      if (!function->impl)
         continue;

      bool locals_shrunk = false;
      if (modes & nir_var_function_temp) {
         locals_shrunk = shrink_vec_var_list(&function->impl->locals,
                                             nir_var_function_temp,
                                             var_usage_map);
      }

      if (globals_shrunk || locals_shrunk) {
         shrink_vec_var_access_impl(function->impl, var_usage_map, modes);

         nir_metadata_preserve(function->impl, nir_metadata_block_index |
                                               nir_metadata_dominance);
         progress = true;
      } else {
         nir_metadata_preserve(function->impl, nir_metadata_all);
      }
   }

   ralloc_free(mem_ctx);

   return progress;
}
