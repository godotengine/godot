/*
 * Copyright (C) 2019-2021 Collabora, Ltd.
 * Copyright (C) 2019 Alyssa Rosenzweig
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

/**
 * @file
 *
 * Implements the fragment pipeline (blending and writeout) in software, to be
 * run as a dedicated "blend shader" stage on Midgard/Bifrost, or as a fragment
 * shader variant on typical GPUs. This pass is useful if hardware lacks
 * fixed-function blending in part or in full.
 */

#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_format_convert.h"
#include "nir_lower_blend.h"

/* Given processed factors, combine them per a blend function */

static nir_ssa_def *
nir_blend_func(
   nir_builder *b,
   enum blend_func func,
   nir_ssa_def *src, nir_ssa_def *dst)
{
   switch (func) {
   case BLEND_FUNC_ADD:
      return nir_fadd(b, src, dst);
   case BLEND_FUNC_SUBTRACT:
      return nir_fsub(b, src, dst);
   case BLEND_FUNC_REVERSE_SUBTRACT:
      return nir_fsub(b, dst, src);
   case BLEND_FUNC_MIN:
      return nir_fmin(b, src, dst);
   case BLEND_FUNC_MAX:
      return nir_fmax(b, src, dst);
   }

   unreachable("Invalid blend function");
}

/* Does this blend function multiply by a blend factor? */

static bool
nir_blend_factored(enum blend_func func)
{
   switch (func) {
   case BLEND_FUNC_ADD:
   case BLEND_FUNC_SUBTRACT:
   case BLEND_FUNC_REVERSE_SUBTRACT:
      return true;
   default:
      return false;
   }
}

/* Compute a src_alpha_saturate factor */
static nir_ssa_def *
nir_alpha_saturate(
   nir_builder *b,
   nir_ssa_def *src, nir_ssa_def *dst,
   unsigned chan)
{
   nir_ssa_def *Asrc = nir_channel(b, src, 3);
   nir_ssa_def *Adst = nir_channel(b, dst, 3);
   nir_ssa_def *one = nir_imm_floatN_t(b, 1.0, src->bit_size);
   nir_ssa_def *Adsti = nir_fsub(b, one, Adst);

   return (chan < 3) ? nir_fmin(b, Asrc, Adsti) : one;
}

/* Returns a scalar single factor, unmultiplied */

static nir_ssa_def *
nir_blend_factor_value(
   nir_builder *b,
   nir_ssa_def *src, nir_ssa_def *src1, nir_ssa_def *dst, nir_ssa_def *bconst,
   unsigned chan,
   enum blend_factor factor)
{
   switch (factor) {
   case BLEND_FACTOR_ZERO:
      return nir_imm_floatN_t(b, 0.0, src->bit_size);
   case BLEND_FACTOR_SRC_COLOR:
      return nir_channel(b, src, chan);
   case BLEND_FACTOR_SRC1_COLOR:
      return nir_channel(b, src1, chan);
   case BLEND_FACTOR_DST_COLOR:
      return nir_channel(b, dst, chan);
   case BLEND_FACTOR_SRC_ALPHA:
      return nir_channel(b, src, 3);
   case BLEND_FACTOR_SRC1_ALPHA:
      return nir_channel(b, src1, 3);
   case BLEND_FACTOR_DST_ALPHA:
      return nir_channel(b, dst, 3);
   case BLEND_FACTOR_CONSTANT_COLOR:
      return nir_channel(b, bconst, chan);
   case BLEND_FACTOR_CONSTANT_ALPHA:
      return nir_channel(b, bconst, 3);
   case BLEND_FACTOR_SRC_ALPHA_SATURATE:
      return nir_alpha_saturate(b, src, dst, chan);
   }

   unreachable("Invalid blend factor");
}

static nir_ssa_def *
nir_fsat_signed(nir_builder *b, nir_ssa_def *x)
{
   return nir_fclamp(b, x, nir_imm_floatN_t(b, -1.0, x->bit_size),
                           nir_imm_floatN_t(b, +1.0, x->bit_size));
}

static nir_ssa_def *
nir_fsat_to_format(nir_builder *b, nir_ssa_def *x, enum pipe_format format)
{
   if (util_format_is_unorm(format))
      return nir_fsat(b, x);
   else if (util_format_is_snorm(format))
      return nir_fsat_signed(b, x);
   else
      return x;
}

/*
 * The spec says we need to clamp blend factors. However, we don't want to clamp
 * unnecessarily, as the clamp might not be optimized out. Check whether
 * clamping a blend factor is needed.
 */
static bool
should_clamp_factor(enum blend_factor factor, bool inverted, bool snorm)
{
   switch (factor) {
   case BLEND_FACTOR_ZERO:
      /* 0, 1 are in [0, 1] and [-1, 1] */
      return false;

   case BLEND_FACTOR_SRC_COLOR:
   case BLEND_FACTOR_SRC1_COLOR:
   case BLEND_FACTOR_DST_COLOR:
   case BLEND_FACTOR_SRC_ALPHA:
   case BLEND_FACTOR_SRC1_ALPHA:
   case BLEND_FACTOR_DST_ALPHA:
      /* Colours are already clamped. For unorm, the complement of something
       * clamped is still clamped. But for snorm, this is not true. Clamp for
       * snorm only.
       */
      return inverted && snorm;

   case BLEND_FACTOR_CONSTANT_COLOR:
   case BLEND_FACTOR_CONSTANT_ALPHA:
      /* Constant colours are not yet clamped */
      return true;

   case BLEND_FACTOR_SRC_ALPHA_SATURATE:
      /* For unorm, this is in bounds (and hence so is its complement). For
       * snorm, it may not be.
       */
      return snorm;
   }

   unreachable("invalid blend factor");
}

static nir_ssa_def *
nir_blend_factor(
   nir_builder *b,
   nir_ssa_def *raw_scalar,
   nir_ssa_def *src, nir_ssa_def *src1, nir_ssa_def *dst, nir_ssa_def *bconst,
   unsigned chan,
   enum blend_factor factor,
   bool inverted,
   enum pipe_format format)
{
   nir_ssa_def *f =
      nir_blend_factor_value(b, src, src1, dst, bconst, chan, factor);

   if (inverted)
      f = nir_fadd_imm(b, nir_fneg(b, f), 1.0);

   if (should_clamp_factor(factor, inverted, util_format_is_snorm(format)))
      f = nir_fsat_to_format(b, f, format);

   return nir_fmul(b, raw_scalar, f);
}

/* Given a colormask, "blend" with the destination */

static nir_ssa_def *
nir_color_mask(
   nir_builder *b,
   unsigned mask,
   nir_ssa_def *src,
   nir_ssa_def *dst)
{
   return nir_vec4(b,
         nir_channel(b, (mask & (1 << 0)) ? src : dst, 0),
         nir_channel(b, (mask & (1 << 1)) ? src : dst, 1),
         nir_channel(b, (mask & (1 << 2)) ? src : dst, 2),
         nir_channel(b, (mask & (1 << 3)) ? src : dst, 3));
}

static nir_ssa_def *
nir_logicop_func(
   nir_builder *b,
   unsigned func,
   nir_ssa_def *src, nir_ssa_def *dst, nir_ssa_def *bitmask)
{
   switch (func) {
   case PIPE_LOGICOP_CLEAR:
      return nir_imm_ivec4(b, 0, 0, 0, 0);
   case PIPE_LOGICOP_NOR:
      return nir_ixor(b, nir_ior(b, src, dst), bitmask);
   case PIPE_LOGICOP_AND_INVERTED:
      return nir_iand(b, nir_ixor(b, src, bitmask), dst);
   case PIPE_LOGICOP_COPY_INVERTED:
      return nir_ixor(b, src, bitmask);
   case PIPE_LOGICOP_AND_REVERSE:
      return nir_iand(b, src, nir_ixor(b, dst, bitmask));
   case PIPE_LOGICOP_INVERT:
      return nir_ixor(b, dst, bitmask);
   case PIPE_LOGICOP_XOR:
      return nir_ixor(b, src, dst);
   case PIPE_LOGICOP_NAND:
      return nir_ixor(b, nir_iand(b, src, dst), bitmask);
   case PIPE_LOGICOP_AND:
      return nir_iand(b, src, dst);
   case PIPE_LOGICOP_EQUIV:
      return nir_ixor(b, nir_ixor(b, src, dst), bitmask);
   case PIPE_LOGICOP_NOOP:
      return dst;
   case PIPE_LOGICOP_OR_INVERTED:
      return nir_ior(b, nir_ixor(b, src, bitmask), dst);
   case PIPE_LOGICOP_COPY:
      return src;
   case PIPE_LOGICOP_OR_REVERSE:
      return nir_ior(b, src, nir_ixor(b, dst, bitmask));
   case PIPE_LOGICOP_OR:
      return nir_ior(b, src, dst);
   case PIPE_LOGICOP_SET:
      return nir_imm_ivec4(b, ~0, ~0, ~0, ~0);
   }

   unreachable("Invalid logciop function");
}

static nir_ssa_def *
nir_blend_logicop(
   nir_builder *b,
   const nir_lower_blend_options *options,
   unsigned rt,
   nir_ssa_def *src, nir_ssa_def *dst)
{
   unsigned bit_size = src->bit_size;

   enum pipe_format format = options->format[rt];
   const struct util_format_description *format_desc =
      util_format_description(format);

   /* From section 17.3.9 ("Logical Operation") of the OpenGL 4.6 core spec:
    *
    *    Logical operation has no effect on a floating-point destination color
    *    buffer, or when FRAMEBUFFER_SRGB is enabled and the value of
    *    FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING for the framebuffer attachment
    *    corresponding to the destination buffer is SRGB (see section 9.2.3).
    *    However, if logical operation is enabled, blending is still disabled.
    */
   if (util_format_is_float(format) || util_format_is_srgb(format))
      return src;

   if (bit_size != 32) {
      src = nir_f2f32(b, src);
      dst = nir_f2f32(b, dst);
   }

   assert(src->num_components <= 4);
   assert(dst->num_components <= 4);

   unsigned bits[4];
   for (int i = 0; i < 4; ++i)
       bits[i] = format_desc->channel[i].size;

   if (util_format_is_unorm(format)) {
      src = nir_format_float_to_unorm(b, src, bits);
      dst = nir_format_float_to_unorm(b, dst, bits);
   } else if (util_format_is_snorm(format)) {
      src = nir_format_float_to_snorm(b, src, bits);
      dst = nir_format_float_to_snorm(b, dst, bits);
   } else {
      assert(util_format_is_pure_integer(format));
   }

   nir_const_value mask[4];
   for (int i = 0; i < 4; ++i)
      mask[i] = nir_const_value_for_int(BITFIELD_MASK(bits[i]), 32);

   nir_ssa_def *out = nir_logicop_func(b, options->logicop_func, src, dst,
                                       nir_build_imm(b, 4, 32, mask));

   if (util_format_is_unorm(format)) {
      out = nir_format_unorm_to_float(b, out, bits);
   } else if (util_format_is_snorm(format)) {
      /* Sign extend before converting so the i2f in snorm_to_float works */
      out = nir_format_sign_extend_ivec(b, out, bits);
      out = nir_format_snorm_to_float(b, out, bits);
   } else {
      assert(util_format_is_pure_integer(format));
   }

   if (bit_size == 16)
      out = nir_f2f16(b, out);

   return out;
}

static bool
channel_exists(const struct util_format_description *desc, unsigned i)
{
   return (i < desc->nr_channels) &&
          desc->channel[i].type != UTIL_FORMAT_TYPE_VOID;
}

/* Given a blend state, the source color, and the destination color,
 * return the blended color
 */

static nir_ssa_def *
nir_blend(
   nir_builder *b,
   const nir_lower_blend_options *options,
   unsigned rt,
   nir_ssa_def *src, nir_ssa_def *src1, nir_ssa_def *dst)
{
   /* Grab the blend constant ahead of time */
   nir_ssa_def *bconst;
   if (options->scalar_blend_const) {
      bconst = nir_vec4(b,
                        nir_load_blend_const_color_r_float(b),
                        nir_load_blend_const_color_g_float(b),
                        nir_load_blend_const_color_b_float(b),
                        nir_load_blend_const_color_a_float(b));
   } else {
      bconst = nir_load_blend_const_color_rgba(b);
   }

   if (src->bit_size == 16)
      bconst = nir_f2f16(b, bconst);

   /* Fixed-point framebuffers require their inputs clamped. */
   enum pipe_format format = options->format[rt];

   /* From section 17.3.6 "Blending" of the OpenGL 4.5 spec:
    *
    *     If the color buffer is fixed-point, the components of the source and
    *     destination values and blend factors are each clamped to [0, 1] or
    *     [-1, 1] respectively for an unsigned normalized or signed normalized
    *     color buffer prior to evaluating the blend equation. If the color
    *     buffer is floating-point, no clamping occurs.
    *
    * Blend factors are clamped at the time of their use to ensure we properly
    * clamp negative constant colours with signed normalized formats and
    * ONE_MINUS_CONSTANT_* factors. Notice that -1 is in [-1, 1] but 1 - (-1) =
    * 2 is not in [-1, 1] and should be clamped to 1.
    */
   src = nir_fsat_to_format(b, src, format);

   if (src1)
      src1 = nir_fsat_to_format(b, src1, format);

   /* DST_ALPHA reads back 1.0 if there is no alpha channel */
   const struct util_format_description *desc =
      util_format_description(format);

   nir_ssa_def *zero = nir_imm_floatN_t(b, 0.0, dst->bit_size);
   nir_ssa_def *one = nir_imm_floatN_t(b, 1.0, dst->bit_size);

   dst = nir_vec4(b,
         channel_exists(desc, 0) ? nir_channel(b, dst, 0) : zero,
         channel_exists(desc, 1) ? nir_channel(b, dst, 1) : zero,
         channel_exists(desc, 2) ? nir_channel(b, dst, 2) : zero,
         channel_exists(desc, 3) ? nir_channel(b, dst, 3) : one);

   /* We blend per channel and recombine later */
   nir_ssa_def *channels[4];

   for (unsigned c = 0; c < 4; ++c) {
      /* Decide properties based on channel */
      nir_lower_blend_channel chan =
         (c < 3) ? options->rt[rt].rgb : options->rt[rt].alpha;

      nir_ssa_def *psrc = nir_channel(b, src, c);
      nir_ssa_def *pdst = nir_channel(b, dst, c);

      if (nir_blend_factored(chan.func)) {
         psrc = nir_blend_factor(
                   b, psrc,
                   src, src1, dst, bconst, c,
                   chan.src_factor, chan.invert_src_factor, format);

         pdst = nir_blend_factor(
                   b, pdst,
                   src, src1, dst, bconst, c,
                   chan.dst_factor, chan.invert_dst_factor, format);
      }

      channels[c] = nir_blend_func(b, chan.func, psrc, pdst);
   }

   return nir_vec(b, channels, 4);
}

static int
color_index_for_var(const nir_variable *var)
{
   if (var->data.location != FRAG_RESULT_COLOR &&
       var->data.location < FRAG_RESULT_DATA0)
      return -1;

   return (var->data.location == FRAG_RESULT_COLOR) ? 0 :
          (var->data.location - FRAG_RESULT_DATA0);
}

/*
 * Test if the blending options for a given channel encode the "replace" blend
 * mode: dest = source. In this case, blending may be specially optimized.
 */
static bool
nir_blend_replace_channel(const nir_lower_blend_channel *c)
{
   return (c->func == BLEND_FUNC_ADD) &&
          (c->src_factor == BLEND_FACTOR_ZERO && c->invert_src_factor) &&
          (c->dst_factor == BLEND_FACTOR_ZERO && !c->invert_dst_factor);
}

static bool
nir_blend_replace_rt(const nir_lower_blend_rt *rt)
{
   return nir_blend_replace_channel(&rt->rgb) &&
          nir_blend_replace_channel(&rt->alpha);
}

static bool
nir_lower_blend_store(nir_builder *b, nir_intrinsic_instr *store,
                      const nir_lower_blend_options *options)
{
   assert(store->intrinsic == nir_intrinsic_store_deref);

   nir_variable *var = nir_intrinsic_get_var(store, 0);
   int rt = color_index_for_var(var);

   /* No blend lowering requested on this RT */
   if (rt < 0 || options->format[rt] == PIPE_FORMAT_NONE)
      return false;

   b->cursor = nir_before_instr(&store->instr);

   /* Grab the input color.  We always want 4 channels during blend.  Dead
    * code will clean up any channels we don't need.
    */
   assert(store->src[1].is_ssa);
   nir_ssa_def *src = nir_pad_vector(b, store->src[1].ssa, 4);

   /* Grab the previous fragment color */
   var->data.fb_fetch_output = true;
   b->shader->info.outputs_read |= BITFIELD64_BIT(var->data.location);
   b->shader->info.fs.uses_fbfetch_output = true;
   nir_ssa_def *dst = nir_pad_vector(b, nir_load_var(b, var), 4);

   /* Blend the two colors per the passed options. We only call nir_blend if
    * blending is enabled with a blend mode other than replace (independent of
    * the color mask). That avoids unnecessary fsat instructions in the common
    * case where blending is disabled at an API level, but the driver calls
    * nir_blend (possibly for color masking).
    */
   nir_ssa_def *blended = src;

   if (options->logicop_enable) {
      blended = nir_blend_logicop(b, options, rt, src, dst);
   } else if (!util_format_is_pure_integer(options->format[rt]) &&
              !nir_blend_replace_rt(&options->rt[rt])) {
      assert(!util_format_is_scaled(options->format[rt]));
      blended = nir_blend(b, options, rt, src, options->src1, dst);
   }

   /* Apply a colormask if necessary */
   if (options->rt[rt].colormask != BITFIELD_MASK(4))
      blended = nir_color_mask(b, options->rt[rt].colormask, blended, dst);

   const unsigned num_components = glsl_get_vector_elements(var->type);

   /* Shave off any components we don't want to store */
   blended = nir_trim_vector(b, blended, num_components);

   /* Grow or shrink the store destination as needed */
   store->num_components = num_components;
   store->dest.ssa.num_components = num_components;
   nir_intrinsic_set_write_mask(store, nir_intrinsic_write_mask(store) &
                                       nir_component_mask(num_components));

   /* Write out the final color instead of the input */
   nir_instr_rewrite_src_ssa(&store->instr, &store->src[1], blended);
   return true;
}

static bool
nir_lower_blend_instr(nir_builder *b, nir_instr *instr, void *data)
{
   const nir_lower_blend_options *options = data;

   switch (instr->type) {
   case nir_instr_type_deref: {
      /* Fix up output deref types, as needed */
      nir_deref_instr *deref = nir_instr_as_deref(instr);
      if (!nir_deref_mode_is(deref, nir_var_shader_out))
         return false;

      /* Indirects must be already lowered and output variables split */
      assert(deref->deref_type == nir_deref_type_var);

      if (deref->type == deref->var->type)
         return false;

      deref->type = deref->var->type;
      return true;
   }

   case nir_instr_type_intrinsic: {
      nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
      if (intrin->intrinsic != nir_intrinsic_load_deref &&
          intrin->intrinsic != nir_intrinsic_store_deref)
         return false;

      nir_deref_instr *deref = nir_src_as_deref(intrin->src[0]);
      if (!nir_deref_mode_is(deref, nir_var_shader_out))
         return false;

      assert(glsl_type_is_vector_or_scalar(deref->type));

      if (intrin->intrinsic == nir_intrinsic_load_deref) {
         /* We need to fix up framebuffer if num_components changed */
         const unsigned num_components = glsl_get_vector_elements(deref->type);
         if (intrin->num_components == num_components)
            return false;

         b->cursor = nir_after_instr(&intrin->instr);

         assert(intrin->dest.is_ssa);
         nir_ssa_def *val = nir_resize_vector(b, &intrin->dest.ssa,
                                              num_components);
         intrin->num_components = num_components,
         nir_ssa_def_rewrite_uses_after(&intrin->dest.ssa, val,
                                        val->parent_instr);
         return true;
      } else {
         return nir_lower_blend_store(b, intrin, options);
      }
   }

   default:
      return false;
   }
}

/** Lower blending to framebuffer fetch and some math
 *
 * This pass requires that indirects are lowered and output variables split
 * so that we have a single output variable for each RT.  We could go to the
 * effort of handling arrays (possibly of arrays) but, given that we need
 * indirects lowered anyway (we need constant indices to look up blend
 * functions and formats), we may as well require variables to be split.
 * This can be done by calling nir_lower_io_arrays_to_elements_no_indirect().
 */
void
nir_lower_blend(nir_shader *shader, const nir_lower_blend_options *options)
{
   assert(shader->info.stage == MESA_SHADER_FRAGMENT);

   /* Re-type any blended output variables to have the same number of
    * components as the image format.  The GL 4.6 Spec says:
    *
    *    "If a fragment shader writes to none of gl_FragColor, gl_FragData,
    *    nor any user-defined output variables, the values of the fragment
    *    colors following shader execution are undefined, and may differ for
    *    each fragment color.  If some, but not all elements of gl_FragData or
    *    of theser-defined output variables are written, the values of
    *    fragment colors corresponding to unwritten elements orariables are
    *    similarly undefined."
    *
    * Note the phrase "following shader execution".  Those color values are
    * then supposed to go into blending which may, depending on the blend
    * mode, apply constraints that result in well-defined rendering.  It's
    * fine if we have to pad out a value with undef but we then need to blend
    * that garbage value to ensure correct results.
    *
    * This may also, depending on output format, be a small optimization
    * allowing NIR to dead-code unused calculations.
    */
   nir_foreach_shader_out_variable(var, shader) {
      int rt = color_index_for_var(var);

      /* No blend lowering requested on this RT */
      if (rt < 0 || options->format[rt] == PIPE_FORMAT_NONE)
         continue;

      const unsigned num_format_components =
         util_format_get_nr_components(options->format[rt]);

      /* Indirects must be already lowered and output variables split */
      assert(glsl_type_is_vector_or_scalar(var->type));
      var->type = glsl_replace_vector_type(var->type, num_format_components);
   }

   nir_shader_instructions_pass(shader, nir_lower_blend_instr,
                                nir_metadata_block_index |
                                nir_metadata_dominance,
                                (void *)options);
}
