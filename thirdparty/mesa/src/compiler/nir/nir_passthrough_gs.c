/*
 * Copyright © 2022 Collabora Ltc.
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
 * A helper to create a passthrough GS shader for drivers that needs to lower
 * some rendering tasks to the GS.
 */

nir_shader *
nir_create_passthrough_gs(const nir_shader_compiler_options *options,
                          const nir_shader *prev_stage,
                          enum shader_prim primitive_type,
                          unsigned vertices)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_GEOMETRY,
                                                  options,
                                                  "gs passthrough");

   nir_shader *nir = b.shader;
   nir->info.gs.input_primitive = primitive_type;
   nir->info.gs.output_primitive = primitive_type;
   nir->info.gs.vertices_in = vertices;
   nir->info.gs.vertices_out = vertices;
   nir->info.gs.invocations = 1;
   nir->info.gs.active_stream_mask = 1;

   nir_variable *in_vars[VARYING_SLOT_MAX];
   nir_variable *out_vars[VARYING_SLOT_MAX];
   unsigned num_vars = 0;

   /* Create input/output variables. */
   nir_foreach_shader_out_variable(var, prev_stage) {
      assert(!var->data.patch);

      char name[100];
      if (var->name)
         snprintf(name, sizeof(name), "in_%s", var->name);
      else
         snprintf(name, sizeof(name), "in_%d", var->data.driver_location);

      nir_variable *in = nir_variable_create(nir, nir_var_shader_in,
                                             glsl_array_type(var->type,
                                                             vertices,
                                                             false),
                                             name);
      in->data.location = var->data.location;
      in->data.location_frac = var->data.location_frac;
      in->data.driver_location = var->data.driver_location;
      in->data.interpolation = var->data.interpolation;
      in->data.compact = var->data.compact;

      if (var->name)
         snprintf(name, sizeof(name), "out_%s", var->name);
      else
         snprintf(name, sizeof(name), "out_%d", var->data.driver_location);

      nir_variable *out = nir_variable_create(nir, nir_var_shader_out,
                                              var->type, name);
      out->data.location = var->data.location;
      out->data.location_frac = var->data.location_frac;
      out->data.driver_location = var->data.driver_location;
      out->data.interpolation = var->data.interpolation;
      out->data.compact = var->data.compact;

      in_vars[num_vars] = in;
      out_vars[num_vars++] = out;
   }

   for (unsigned i = 0; i < vertices; ++i) {
      /* Copy inputs to outputs. */
      for (unsigned j = 0; j < num_vars; ++j) {
         /* no need to use copy_var to save a lower pass */
         nir_ssa_def *value = nir_load_array_var_imm(&b, in_vars[j], i);
         nir_store_var(&b, out_vars[j], value,
                       (1u << value->num_components) - 1);
      }
      nir_emit_vertex(&b, 0);
   }

   nir_end_primitive(&b, 0);
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
   nir_validate_shader(nir, "in nir_create_passthrough_gs");

   return nir;
}
