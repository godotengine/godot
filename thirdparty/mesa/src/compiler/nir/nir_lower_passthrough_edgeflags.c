/*
 * Copyright © 2015 Red Hat
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

static void
lower_impl(nir_function_impl *impl)
{
   nir_shader *shader = impl->function->shader;
   nir_builder b;
   nir_variable *in, *out;
   nir_ssa_def *def;

   nir_builder_init(&b, impl);
   b.cursor = nir_before_cf_list(&impl->body);

   /* The edge flag is the last input in st/mesa.  This code is also called by
    * i965 which calls it before any input locations are assigned.
    */
   assert(shader->num_inputs == 0 ||
          shader->num_inputs == util_bitcount64(shader->info.inputs_read));

   /* Lowered IO only uses intrinsics. It doesn't use variables. */
   if (shader->info.io_lowered) {
      assert(shader->num_outputs ==
             util_bitcount64(shader->info.outputs_written));

      /* Load an edge flag. */
      nir_io_semantics load_sem = {0};
      load_sem.location = VERT_ATTRIB_EDGEFLAG;
      load_sem.num_slots = 1;

      nir_ssa_def *load =
         nir_load_input(&b, 1, 32, nir_imm_int(&b, 0),
                        .base = shader->num_inputs++,
                        .component = 0,
                        .dest_type = nir_type_float32,
                        .io_semantics = load_sem);

      /* Store an edge flag. */
      nir_io_semantics semantics = {0};
      semantics.location = VARYING_SLOT_EDGE;
      semantics.num_slots = 1;

      nir_store_output(&b, load, nir_imm_int(&b, 0),
                       .base = shader->num_outputs++,
                       .component = 0,
                       .io_semantics = semantics,
                       .src_type = nir_type_float32,
                       .write_mask = 0x1);

      nir_metadata_preserve(impl, nir_metadata_block_index |
                                  nir_metadata_dominance);
      return;
   }

   in  = nir_variable_create(shader, nir_var_shader_in,
                             glsl_vec4_type(), "edgeflag_in");
   in->data.location = VERT_ATTRIB_EDGEFLAG;

   in->data.driver_location = shader->num_inputs++;
   shader->info.inputs_read |= VERT_BIT_EDGEFLAG;

   out = nir_variable_create(shader, nir_var_shader_out,
                             glsl_vec4_type(), "edgeflag_out");
   out->data.location = VARYING_SLOT_EDGE;
   shader->info.outputs_written |= VARYING_BIT_EDGE;

   def = nir_load_var(&b, in);
   nir_store_var(&b, out, def, 0xf);

   nir_metadata_preserve(impl, nir_metadata_block_index |
                               nir_metadata_dominance);
}

void nir_lower_passthrough_edgeflags(nir_shader *shader)
{
   assert(shader->info.stage == MESA_SHADER_VERTEX);

   shader->info.vs.needs_edge_flag = true;

   lower_impl(nir_shader_get_entrypoint(shader));
}
