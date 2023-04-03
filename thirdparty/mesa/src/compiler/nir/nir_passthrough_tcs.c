/*
 * Copyright © 2022 Google, Inc.
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

/*
 * A helper to create a passthrough TCS shader for drivers that cannot handle
 * having a TES without TCS.
 *
 * Uses the load_tess_level_outer_default and load_tess_level_inner_default
 * intrinsics to load the gl_TessLevelOuter and gl_TessLevelInner values,
 * so driver will somehow need to implement those to load the values set
 * by pipe_context::set_tess_state() or similar.
 */

nir_shader *
nir_create_passthrough_tcs_impl(const nir_shader_compiler_options *options,
                                unsigned *locations, unsigned num_locations,
                                uint8_t patch_vertices)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_TESS_CTRL, options,
                                                  "tcs passthrough");

   unsigned num_inputs = 0;
   unsigned num_outputs = 0;

   nir_variable *in_inner =
      nir_variable_create(b.shader, nir_var_system_value, glsl_vec_type(2),
                          "tess inner default");
   in_inner->data.location = SYSTEM_VALUE_TESS_LEVEL_INNER_DEFAULT;

   nir_variable *out_inner =
      nir_variable_create(b.shader, nir_var_shader_out, glsl_vec_type(2),
                          "tess inner");
   out_inner->data.location = VARYING_SLOT_TESS_LEVEL_INNER;
   out_inner->data.driver_location = num_outputs++;

   nir_ssa_def *inner = nir_load_var(&b, in_inner);
   nir_store_var(&b, out_inner, inner, 0x3);

   nir_variable *in_outer =
      nir_variable_create(b.shader, nir_var_system_value, glsl_vec4_type(),
                          "tess outer default");
   in_outer->data.location = SYSTEM_VALUE_TESS_LEVEL_OUTER_DEFAULT;

   nir_variable *out_outer =
      nir_variable_create(b.shader, nir_var_shader_out, glsl_vec4_type(),
                          "tess outer");
   out_outer->data.location = VARYING_SLOT_TESS_LEVEL_OUTER;
   out_outer->data.driver_location = num_outputs++;

   nir_ssa_def *outer = nir_load_var(&b, in_outer);
   nir_store_var(&b, out_outer, outer, 0xf);

   nir_ssa_def *id = nir_load_invocation_id(&b);
   for (unsigned i = 0; i < num_locations; i++) {
      const struct glsl_type *type;
      unsigned semantic = locations[i];
      if (semantic < VARYING_SLOT_VAR31 && semantic != VARYING_SLOT_EDGE)
         type = glsl_array_type(glsl_vec4_type(), 0, 0);
      else if (semantic >= VARYING_SLOT_VAR0_16BIT)
         type = glsl_array_type(glsl_vector_type(GLSL_TYPE_FLOAT16, 4), 0, 0);
      else
         continue;

      char name[10];
      snprintf(name, sizeof(name), "in_%d", i);
      nir_variable *in = nir_variable_create(b.shader, nir_var_shader_in, type, name);
      in->data.location = semantic;
      in->data.driver_location = num_inputs++;

      snprintf(name, sizeof(name), "out_%d", i);
      nir_variable *out = nir_variable_create(b.shader, nir_var_shader_out, type, name);
      out->data.location = semantic;
      out->data.driver_location = num_outputs++;

      /* no need to use copy_var to save a lower pass */
      nir_ssa_def *value = nir_load_array_var(&b, in, id);
      nir_store_array_var(&b, out, id, value, 0xf);
   }

   b.shader->num_inputs = num_inputs;
   b.shader->num_outputs = num_outputs;

   b.shader->info.tess.tcs_vertices_out = patch_vertices;

   nir_validate_shader(b.shader, "in nir_create_passthrough_tcs");

   return b.shader;
}


nir_shader *
nir_create_passthrough_tcs(const nir_shader_compiler_options *options,
                           const nir_shader *vs, uint8_t patch_vertices)
{
   unsigned locations[MAX_VARYING];
   unsigned num_outputs = 0;

   nir_foreach_shader_out_variable (var, vs) {
      assert(num_outputs < ARRAY_SIZE(locations));
      locations[num_outputs++] = var->data.location;
   }

   return nir_create_passthrough_tcs_impl(options, locations, num_outputs, patch_vertices);
}
