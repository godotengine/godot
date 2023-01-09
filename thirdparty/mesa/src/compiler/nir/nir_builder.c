/*
 * Copyright © 2014-2015 Broadcom
 * Copyright © 2021 Google
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

void
nir_builder_init(nir_builder *build, nir_function_impl *impl)
{
   memset(build, 0, sizeof(*build));
   build->exact = false;
   build->impl = impl;
   build->shader = impl->function->shader;
}

nir_builder MUST_CHECK PRINTFLIKE(3, 4)
nir_builder_init_simple_shader(gl_shader_stage stage,
                               const nir_shader_compiler_options *options,
                               const char *name, ...)
{
   nir_builder b;

   memset(&b, 0, sizeof(b));
   b.shader = nir_shader_create(NULL, stage, options, NULL);

   if (name) {
      va_list args;
      va_start(args, name);
      b.shader->info.name = ralloc_vasprintf(b.shader, name, args);
      va_end(args);
   }

   nir_function *func = nir_function_create(b.shader, "main");
   func->is_entrypoint = true;
   b.exact = false;
   b.impl = nir_function_impl_create(func);
   b.cursor = nir_after_cf_list(&b.impl->body);

   /* Simple shaders are typically internal, e.g. blit shaders */
   b.shader->info.internal = true;

   return b;
}

nir_ssa_def *
nir_builder_alu_instr_finish_and_insert(nir_builder *build, nir_alu_instr *instr)
{
   const nir_op_info *op_info = &nir_op_infos[instr->op];

   instr->exact = build->exact;

   /* Guess the number of components the destination temporary should have
    * based on our input sizes, if it's not fixed for the op.
    */
   unsigned num_components = op_info->output_size;
   if (num_components == 0) {
      for (unsigned i = 0; i < op_info->num_inputs; i++) {
         if (op_info->input_sizes[i] == 0)
            num_components = MAX2(num_components,
                                  instr->src[i].src.ssa->num_components);
      }
   }
   assert(num_components != 0);

   /* Figure out the bitwidth based on the source bitwidth if the instruction
    * is variable-width.
    */
   unsigned bit_size = nir_alu_type_get_type_size(op_info->output_type);
   if (bit_size == 0) {
      for (unsigned i = 0; i < op_info->num_inputs; i++) {
         unsigned src_bit_size = instr->src[i].src.ssa->bit_size;
         if (nir_alu_type_get_type_size(op_info->input_types[i]) == 0) {
            if (bit_size)
               assert(src_bit_size == bit_size);
            else
               bit_size = src_bit_size;
         } else {
            assert(src_bit_size ==
               nir_alu_type_get_type_size(op_info->input_types[i]));
         }
      }
   }

   /* When in doubt, assume 32. */
   if (bit_size == 0)
      bit_size = 32;

   /* Make sure we don't swizzle from outside of our source vector (like if a
    * scalar value was passed into a multiply with a vector).
    */
   for (unsigned i = 0; i < op_info->num_inputs; i++) {
      for (unsigned j = instr->src[i].src.ssa->num_components;
           j < NIR_MAX_VEC_COMPONENTS; j++) {
         instr->src[i].swizzle[j] = instr->src[i].src.ssa->num_components - 1;
      }
   }

   nir_ssa_dest_init(&instr->instr, &instr->dest.dest, num_components,
                     bit_size, NULL);
   instr->dest.write_mask = (1 << num_components) - 1;

   nir_builder_instr_insert(build, &instr->instr);

   return &instr->dest.dest.ssa;
}

nir_ssa_def *
nir_build_alu(nir_builder *build, nir_op op, nir_ssa_def *src0,
              nir_ssa_def *src1, nir_ssa_def *src2, nir_ssa_def *src3)
{
   nir_alu_instr *instr = nir_alu_instr_create(build->shader, op);
   if (!instr)
      return NULL;

   instr->src[0].src = nir_src_for_ssa(src0);
   if (src1)
      instr->src[1].src = nir_src_for_ssa(src1);
   if (src2)
      instr->src[2].src = nir_src_for_ssa(src2);
   if (src3)
      instr->src[3].src = nir_src_for_ssa(src3);

   return nir_builder_alu_instr_finish_and_insert(build, instr);
}

nir_ssa_def *
nir_build_alu1(nir_builder *build, nir_op op, nir_ssa_def *src0)
{
   nir_alu_instr *instr = nir_alu_instr_create(build->shader, op);
   if (!instr)
      return NULL;

   instr->src[0].src = nir_src_for_ssa(src0);

   return nir_builder_alu_instr_finish_and_insert(build, instr);
}

nir_ssa_def *
nir_build_alu2(nir_builder *build, nir_op op, nir_ssa_def *src0,
              nir_ssa_def *src1)
{
   nir_alu_instr *instr = nir_alu_instr_create(build->shader, op);
   if (!instr)
      return NULL;

   instr->src[0].src = nir_src_for_ssa(src0);
   instr->src[1].src = nir_src_for_ssa(src1);

   return nir_builder_alu_instr_finish_and_insert(build, instr);
}

nir_ssa_def *
nir_build_alu3(nir_builder *build, nir_op op, nir_ssa_def *src0,
              nir_ssa_def *src1, nir_ssa_def *src2)
{
   nir_alu_instr *instr = nir_alu_instr_create(build->shader, op);
   if (!instr)
      return NULL;

   instr->src[0].src = nir_src_for_ssa(src0);
   instr->src[1].src = nir_src_for_ssa(src1);
   instr->src[2].src = nir_src_for_ssa(src2);

   return nir_builder_alu_instr_finish_and_insert(build, instr);
}

nir_ssa_def *
nir_build_alu4(nir_builder *build, nir_op op, nir_ssa_def *src0,
              nir_ssa_def *src1, nir_ssa_def *src2, nir_ssa_def *src3)
{
   nir_alu_instr *instr = nir_alu_instr_create(build->shader, op);
   if (!instr)
      return NULL;

   instr->src[0].src = nir_src_for_ssa(src0);
   instr->src[1].src = nir_src_for_ssa(src1);
   instr->src[2].src = nir_src_for_ssa(src2);
   instr->src[3].src = nir_src_for_ssa(src3);

   return nir_builder_alu_instr_finish_and_insert(build, instr);
}

/* for the couple special cases with more than 4 src args: */
nir_ssa_def *
nir_build_alu_src_arr(nir_builder *build, nir_op op, nir_ssa_def **srcs)
{
   const nir_op_info *op_info = &nir_op_infos[op];
   nir_alu_instr *instr = nir_alu_instr_create(build->shader, op);
   if (!instr)
      return NULL;

   for (unsigned i = 0; i < op_info->num_inputs; i++)
      instr->src[i].src = nir_src_for_ssa(srcs[i]);

   return nir_builder_alu_instr_finish_and_insert(build, instr);
}

nir_ssa_def *
nir_build_tex_deref_instr(nir_builder *build, nir_texop op,
                          nir_deref_instr *texture,
                          nir_deref_instr *sampler,
                          unsigned num_extra_srcs,
                          const nir_tex_src *extra_srcs)
{
   assert(texture != NULL);
   assert(glsl_type_is_texture(texture->type) ||
          glsl_type_is_sampler(texture->type));

   const unsigned num_srcs = 1 + (sampler != NULL) + num_extra_srcs;

   nir_tex_instr *tex = nir_tex_instr_create(build->shader, num_srcs);
   tex->op = op;
   tex->sampler_dim = glsl_get_sampler_dim(texture->type);
   tex->is_array = glsl_sampler_type_is_array(texture->type);
   tex->is_shadow = false;

   switch (op) {
   case nir_texop_txs:
   case nir_texop_texture_samples:
   case nir_texop_query_levels:
   case nir_texop_txf_ms_mcs_intel:
   case nir_texop_fragment_mask_fetch_amd:
   case nir_texop_descriptor_amd:
      tex->dest_type = nir_type_int32;
      break;
   case nir_texop_lod:
      tex->dest_type = nir_type_float32;
      break;
   case nir_texop_samples_identical:
      tex->dest_type = nir_type_bool1;
      break;
   default:
      assert(!nir_tex_instr_is_query(tex));
      tex->dest_type = nir_get_nir_type_for_glsl_base_type(
         glsl_get_sampler_result_type(texture->type));
      break;
   }

   unsigned src_idx = 0;
   tex->src[src_idx++] = (nir_tex_src) {
      .src_type = nir_tex_src_texture_deref,
      .src = nir_src_for_ssa(&texture->dest.ssa),
   };
   if (sampler != NULL) {
      assert(glsl_type_is_sampler(sampler->type));
      tex->src[src_idx++] = (nir_tex_src) {
         .src_type = nir_tex_src_sampler_deref,
         .src = nir_src_for_ssa(&sampler->dest.ssa),
      };
   }
   for (unsigned i = 0; i < num_extra_srcs; i++) {
      switch (extra_srcs[i].src_type) {
      case nir_tex_src_coord:
         tex->coord_components = nir_src_num_components(extra_srcs[i].src);
         assert(tex->coord_components == tex->is_array +
                glsl_get_sampler_dim_coordinate_components(tex->sampler_dim));
         break;

      case nir_tex_src_lod:
         assert(tex->sampler_dim == GLSL_SAMPLER_DIM_1D ||
                tex->sampler_dim == GLSL_SAMPLER_DIM_2D ||
                tex->sampler_dim == GLSL_SAMPLER_DIM_3D ||
                tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE);
         break;

      case nir_tex_src_ms_index:
         assert(tex->sampler_dim == GLSL_SAMPLER_DIM_MS);
         break;

      case nir_tex_src_comparator:
         /* Assume 1-component shadow for the builder helper */
         tex->is_shadow = true;
         tex->is_new_style_shadow = true;
         break;

      case nir_tex_src_texture_deref:
      case nir_tex_src_sampler_deref:
      case nir_tex_src_texture_offset:
      case nir_tex_src_sampler_offset:
      case nir_tex_src_texture_handle:
      case nir_tex_src_sampler_handle:
         unreachable("Texture and sampler must be provided directly as derefs");
         break;

      default:
         break;
      }

      tex->src[src_idx++] = extra_srcs[i];
   }
   assert(src_idx == num_srcs);

   nir_ssa_dest_init(&tex->instr, &tex->dest,
                     nir_tex_instr_dest_size(tex),
                     nir_alu_type_get_type_size(tex->dest_type),
                     NULL);
   nir_builder_instr_insert(build, &tex->instr);

   return &tex->dest.ssa;
}

nir_ssa_def *
nir_vec_scalars(nir_builder *build, nir_ssa_scalar *comp, unsigned num_components)
{
   nir_op op = nir_op_vec(num_components);
   nir_alu_instr *instr = nir_alu_instr_create(build->shader, op);
   if (!instr)
      return NULL;

   for (unsigned i = 0; i < num_components; i++) {
      instr->src[i].src = nir_src_for_ssa(comp[i].def);
      instr->src[i].swizzle[0] = comp[i].comp;
   }
   instr->exact = build->exact;

   /* Note: not reusing nir_builder_alu_instr_finish_and_insert() because it
    * can't re-guess the num_components when num_components == 1 (nir_op_mov).
    */
   nir_ssa_dest_init(&instr->instr, &instr->dest.dest, num_components,
                     comp[0].def->bit_size, NULL);
   instr->dest.write_mask = (1 << num_components) - 1;

   nir_builder_instr_insert(build, &instr->instr);

   return &instr->dest.dest.ssa;
}

/**
 * Turns a nir_src into a nir_ssa_def * so it can be passed to
 * nir_build_alu()-based builder calls.
 *
 * See nir_ssa_for_alu_src() for alu instructions.
 */
nir_ssa_def *
nir_ssa_for_src(nir_builder *build, nir_src src, int num_components)
{
   if (src.is_ssa && src.ssa->num_components == num_components)
      return src.ssa;

   assert((unsigned)num_components <= nir_src_num_components(src));

   nir_alu_src alu = { NIR_SRC_INIT };
   alu.src = src;
   for (int j = 0; j < NIR_MAX_VEC_COMPONENTS; j++)
      alu.swizzle[j] = j;

   return nir_mov_alu(build, alu, num_components);
}

/**
 * Similar to nir_ssa_for_src(), but for alu srcs, respecting the
 * nir_alu_src's swizzle.
 */
nir_ssa_def *
nir_ssa_for_alu_src(nir_builder *build, nir_alu_instr *instr, unsigned srcn)
{
   if (nir_alu_src_is_trivial_ssa(instr, srcn))
      return instr->src[srcn].src.ssa;

   nir_alu_src *src = &instr->src[srcn];
   unsigned num_components = nir_ssa_alu_instr_src_components(instr, srcn);
   return nir_mov_alu(build, *src, num_components);
}

/* Generic builder for system values. */
nir_ssa_def *
nir_load_system_value(nir_builder *build, nir_intrinsic_op op, int index,
                      unsigned num_components, unsigned bit_size)
{
   nir_intrinsic_instr *load = nir_intrinsic_instr_create(build->shader, op);
   if (nir_intrinsic_infos[op].dest_components > 0)
      assert(num_components == nir_intrinsic_infos[op].dest_components);
   else
      load->num_components = num_components;
   load->const_index[0] = index;

   nir_ssa_dest_init(&load->instr, &load->dest,
                     num_components, bit_size, NULL);
   nir_builder_instr_insert(build, &load->instr);
   return &load->dest.ssa;
}

void
nir_builder_instr_insert(nir_builder *build, nir_instr *instr)
{
   nir_instr_insert(build->cursor, instr);

   if (build->update_divergence)
      nir_update_instr_divergence(build->shader, instr);

   /* Move the cursor forward. */
   build->cursor = nir_after_instr(instr);
}

void
nir_builder_cf_insert(nir_builder *build, nir_cf_node *cf)
{
   nir_cf_node_insert(build->cursor, cf);
}

bool
nir_builder_is_inside_cf(nir_builder *build, nir_cf_node *cf_node)
{
   nir_block *block = nir_cursor_current_block(build->cursor);
   for (nir_cf_node *n = &block->cf_node; n; n = n->parent) {
      if (n == cf_node)
         return true;
   }
   return false;
}

nir_if *
nir_push_if_src(nir_builder *build, nir_src condition)
{
   nir_if *nif = nir_if_create(build->shader);
   nif->condition = condition;
   nir_builder_cf_insert(build, &nif->cf_node);
   build->cursor = nir_before_cf_list(&nif->then_list);
   return nif;
}

nir_if *
nir_push_if(nir_builder *build, nir_ssa_def *condition)
{
   return nir_push_if_src(build, nir_src_for_ssa(condition));
}

nir_if *
nir_push_else(nir_builder *build, nir_if *nif)
{
   if (nif) {
      assert(nir_builder_is_inside_cf(build, &nif->cf_node));
   } else {
      nir_block *block = nir_cursor_current_block(build->cursor);
      nif = nir_cf_node_as_if(block->cf_node.parent);
   }
   build->cursor = nir_before_cf_list(&nif->else_list);
   return nif;
}

void
nir_pop_if(nir_builder *build, nir_if *nif)
{
   if (nif) {
      assert(nir_builder_is_inside_cf(build, &nif->cf_node));
   } else {
      nir_block *block = nir_cursor_current_block(build->cursor);
      nif = nir_cf_node_as_if(block->cf_node.parent);
   }
   build->cursor = nir_after_cf_node(&nif->cf_node);
}

nir_ssa_def *
nir_if_phi(nir_builder *build, nir_ssa_def *then_def, nir_ssa_def *else_def)
{
   nir_block *block = nir_cursor_current_block(build->cursor);
   nir_if *nif = nir_cf_node_as_if(nir_cf_node_prev(&block->cf_node));

   nir_phi_instr *phi = nir_phi_instr_create(build->shader);
   nir_phi_instr_add_src(phi, nir_if_last_then_block(nif), nir_src_for_ssa(then_def));
   nir_phi_instr_add_src(phi, nir_if_last_else_block(nif), nir_src_for_ssa(else_def));

   assert(then_def->num_components == else_def->num_components);
   assert(then_def->bit_size == else_def->bit_size);
   nir_ssa_dest_init(&phi->instr, &phi->dest,
                     then_def->num_components, then_def->bit_size, NULL);

   nir_builder_instr_insert(build, &phi->instr);

   return &phi->dest.ssa;
}

nir_loop *
nir_push_loop(nir_builder *build)
{
   nir_loop *loop = nir_loop_create(build->shader);
   nir_builder_cf_insert(build, &loop->cf_node);
   build->cursor = nir_before_cf_list(&loop->body);
   return loop;
}

void
nir_pop_loop(nir_builder *build, nir_loop *loop)
{
   if (loop) {
      assert(nir_builder_is_inside_cf(build, &loop->cf_node));
   } else {
      nir_block *block = nir_cursor_current_block(build->cursor);
      loop = nir_cf_node_as_loop(block->cf_node.parent);
   }
   build->cursor = nir_after_cf_node(&loop->cf_node);
}

nir_ssa_def *
nir_compare_func(nir_builder *b, enum compare_func func,
                 nir_ssa_def *src0, nir_ssa_def *src1)
{
   switch (func) {
   case COMPARE_FUNC_NEVER:
      return nir_imm_int(b, 0);
   case COMPARE_FUNC_ALWAYS:
      return nir_imm_int(b, ~0);
   case COMPARE_FUNC_EQUAL:
      return nir_feq(b, src0, src1);
   case COMPARE_FUNC_NOTEQUAL:
      return nir_fneu(b, src0, src1);
   case COMPARE_FUNC_GREATER:
      return nir_flt(b, src1, src0);
   case COMPARE_FUNC_GEQUAL:
      return nir_fge(b, src0, src1);
   case COMPARE_FUNC_LESS:
      return nir_flt(b, src0, src1);
   case COMPARE_FUNC_LEQUAL:
      return nir_fge(b, src1, src0);
   }
   unreachable("bad compare func");
}

nir_ssa_def *
nir_type_convert(nir_builder *b,
                 nir_ssa_def *src,
                 nir_alu_type src_type,
                 nir_alu_type dest_type,
                 nir_rounding_mode rnd)
{
   assert(nir_alu_type_get_type_size(src_type) == 0 ||
          nir_alu_type_get_type_size(src_type) == src->bit_size);

   const nir_alu_type dst_base =
      (nir_alu_type) nir_alu_type_get_base_type(dest_type);

   const nir_alu_type src_base =
      (nir_alu_type) nir_alu_type_get_base_type(src_type);

   /* b2b and f2b use the regular type conversion path, but i2b is implemented
    * as src != 0.
    */
   if (dst_base == nir_type_bool && (src_base == nir_type_int ||
                                     src_base == nir_type_uint)) {
      nir_op opcode;

      const unsigned dst_bit_size = nir_alu_type_get_type_size(dest_type);

      switch (dst_bit_size) {
      case 1:  opcode = nir_op_ine;   break;
      case 8:  opcode = nir_op_ine8;  break;
      case 16: opcode = nir_op_ine16; break;
      case 32: opcode = nir_op_ine32; break;
      default: unreachable("Invalid Boolean size.");
      }

      return nir_build_alu(b, opcode, src,
                           nir_imm_zero(b, src->num_components, src->bit_size),
                           NULL, NULL);
   } else {
      src_type = (nir_alu_type) (src_type | src->bit_size);

      nir_op opcode =
         nir_type_conversion_op(src_type, dest_type, rnd);
      if (opcode == nir_op_mov)
         return src;

      return nir_build_alu(b, opcode, src, NULL, NULL, NULL);
   }
}

nir_ssa_def *
nir_gen_rect_vertices(nir_builder *b, nir_ssa_def *z, nir_ssa_def *w)
{
   if (!z)
      z = nir_imm_float(b, 0.0);
   if (!w)
      w = nir_imm_float(b, 1.0);

   nir_ssa_def *vertex_id;
   if (b->shader->options->vertex_id_zero_based)
      vertex_id = nir_load_vertex_id_zero_base(b);
   else
      vertex_id = nir_load_vertex_id(b);

   /* vertex 0: -1.0, -1.0
    * vertex 1: -1.0,  1.0
    * vertex 2:  1.0, -1.0
    * vertex 3:  1.0,  1.0
    *
    * so:
    *
    * channel 0 is vertex_id < 2 ? -1.0 :  1.0
    * channel 1 is vertex_id & 1 ?  1.0 : -1.0
    */

   nir_ssa_def *c0cmp = nir_ilt(b, vertex_id, nir_imm_int(b, 2));
   nir_ssa_def *c1cmp = nir_test_mask(b, vertex_id, 1);

   nir_ssa_def *comp[4];
   comp[0] = nir_bcsel(b, c0cmp, nir_imm_float(b, -1.0), nir_imm_float(b, 1.0));
   comp[1] = nir_bcsel(b, c1cmp, nir_imm_float(b, 1.0), nir_imm_float(b, -1.0));
   comp[2] = z;
   comp[3] = w;

   return nir_vec(b, comp, 4);
}
