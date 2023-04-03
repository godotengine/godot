/*
 * Copyright © 2015 Intel Corporation
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

static nir_intrinsic_instr *
as_intrinsic(nir_instr *instr, nir_intrinsic_op op)
{
   if (instr->type != nir_instr_type_intrinsic)
      return NULL;

   nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
   if (intrin->intrinsic != op)
      return NULL;

   return intrin;
}

static nir_intrinsic_instr *
as_set_vertex_and_primitive_count(nir_instr *instr)
{
   return as_intrinsic(instr, nir_intrinsic_set_vertex_and_primitive_count);
}

/**
 * Count the number of vertices/primitives emitted by a geometry shader per stream.
 * If a constant number of vertices is emitted, the output is set to
 * that number, otherwise it is unknown at compile time and the
 * result will be -1.
 *
 * This only works if you've used nir_lower_gs_intrinsics() to do vertex
 * counting at the NIR level.
 */
void
nir_gs_count_vertices_and_primitives(const nir_shader *shader,
                                     int *out_vtxcnt,
                                     int *out_prmcnt,
                                     unsigned num_streams)
{
   assert(num_streams);

   int vtxcnt_arr[4] = {-1, -1, -1, -1};
   int prmcnt_arr[4] = {-1, -1, -1, -1};
   bool cnt_found[4] = {false, false, false, false};

   nir_foreach_function(function, shader) {
      if (!function->impl)
         continue;

      /* set_vertex_and_primitive_count intrinsics only appear in predecessors of the
       * end block.  So we don't need to walk all of them.
       */
      set_foreach(function->impl->end_block->predecessors, entry) {
         nir_block *block = (nir_block *) entry->key;

         nir_foreach_instr_reverse(instr, block) {
            nir_intrinsic_instr *intrin = as_set_vertex_and_primitive_count(instr);
            if (!intrin)
               continue;

            unsigned stream = nir_intrinsic_stream_id(intrin);
            if (stream >= num_streams)
               continue;

            int vtxcnt = -1;
            int prmcnt = -1;

            /* If the number of vertices/primitives is compile-time known, we use that,
             * otherwise we leave it at -1 which means that it's unknown.
             */
            if (nir_src_is_const(intrin->src[0]))
               vtxcnt = nir_src_as_int(intrin->src[0]);
            if (nir_src_is_const(intrin->src[1]))
               prmcnt = nir_src_as_int(intrin->src[1]);

            /* We've found contradictory set_vertex_and_primitive_count intrinsics.
             * This can happen if there are early-returns in main() and
             * different paths emit different numbers of vertices.
             */
            if (cnt_found[stream] && vtxcnt != vtxcnt_arr[stream])
               vtxcnt = -1;
            if (cnt_found[stream] && prmcnt != prmcnt_arr[stream])
               prmcnt = -1;

            vtxcnt_arr[stream] = vtxcnt;
            prmcnt_arr[stream] = prmcnt;
            cnt_found[stream] = true;
         }
      }
   }

   if (out_vtxcnt)
      memcpy(out_vtxcnt, vtxcnt_arr, num_streams * sizeof(int));
   if (out_prmcnt)
      memcpy(out_prmcnt, prmcnt_arr, num_streams * sizeof(int));
}
