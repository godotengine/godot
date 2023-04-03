/*
 * Copyright © 2016 Intel Corporation
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

#include <math.h>
#include "vtn_private.h"
#include "spirv_info.h"

/*
 * Normally, column vectors in SPIR-V correspond to a single NIR SSA
 * definition. But for matrix multiplies, we want to do one routine for
 * multiplying a matrix by a matrix and then pretend that vectors are matrices
 * with one column. So we "wrap" these things, and unwrap the result before we
 * send it off.
 */

static struct vtn_ssa_value *
wrap_matrix(struct vtn_builder *b, struct vtn_ssa_value *val)
{
   if (val == NULL)
      return NULL;

   if (glsl_type_is_matrix(val->type))
      return val;

   struct vtn_ssa_value *dest = rzalloc(b, struct vtn_ssa_value);
   dest->type = glsl_get_bare_type(val->type);
   dest->elems = ralloc_array(b, struct vtn_ssa_value *, 1);
   dest->elems[0] = val;

   return dest;
}

static struct vtn_ssa_value *
unwrap_matrix(struct vtn_ssa_value *val)
{
   if (glsl_type_is_matrix(val->type))
         return val;

   return val->elems[0];
}

static struct vtn_ssa_value *
matrix_multiply(struct vtn_builder *b,
                struct vtn_ssa_value *_src0, struct vtn_ssa_value *_src1)
{

   struct vtn_ssa_value *src0 = wrap_matrix(b, _src0);
   struct vtn_ssa_value *src1 = wrap_matrix(b, _src1);
   struct vtn_ssa_value *src0_transpose = wrap_matrix(b, _src0->transposed);
   struct vtn_ssa_value *src1_transpose = wrap_matrix(b, _src1->transposed);

   unsigned src0_rows = glsl_get_vector_elements(src0->type);
   unsigned src0_columns = glsl_get_matrix_columns(src0->type);
   unsigned src1_columns = glsl_get_matrix_columns(src1->type);

   const struct glsl_type *dest_type;
   if (src1_columns > 1) {
      dest_type = glsl_matrix_type(glsl_get_base_type(src0->type),
                                   src0_rows, src1_columns);
   } else {
      dest_type = glsl_vector_type(glsl_get_base_type(src0->type), src0_rows);
   }
   struct vtn_ssa_value *dest = vtn_create_ssa_value(b, dest_type);

   dest = wrap_matrix(b, dest);

   bool transpose_result = false;
   if (src0_transpose && src1_transpose) {
      /* transpose(A) * transpose(B) = transpose(B * A) */
      src1 = src0_transpose;
      src0 = src1_transpose;
      src0_transpose = NULL;
      src1_transpose = NULL;
      transpose_result = true;
   }

   if (src0_transpose && !src1_transpose &&
       glsl_get_base_type(src0->type) == GLSL_TYPE_FLOAT) {
      /* We already have the rows of src0 and the columns of src1 available,
       * so we can just take the dot product of each row with each column to
       * get the result.
       */

      for (unsigned i = 0; i < src1_columns; i++) {
         nir_ssa_def *vec_src[4];
         for (unsigned j = 0; j < src0_rows; j++) {
            vec_src[j] = nir_fdot(&b->nb, src0_transpose->elems[j]->def,
                                          src1->elems[i]->def);
         }
         dest->elems[i]->def = nir_vec(&b->nb, vec_src, src0_rows);
      }
   } else {
      /* We don't handle the case where src1 is transposed but not src0, since
       * the general case only uses individual components of src1 so the
       * optimizer should chew through the transpose we emitted for src1.
       */

      for (unsigned i = 0; i < src1_columns; i++) {
         /* dest[i] = sum(src0[j] * src1[i][j] for all j) */
         dest->elems[i]->def =
            nir_fmul(&b->nb, src0->elems[src0_columns - 1]->def,
                     nir_channel(&b->nb, src1->elems[i]->def, src0_columns - 1));
         for (int j = src0_columns - 2; j >= 0; j--) {
            dest->elems[i]->def =
               nir_ffma(&b->nb, src0->elems[j]->def,
                                nir_channel(&b->nb, src1->elems[i]->def, j),
                                dest->elems[i]->def);
         }
      }
   }

   dest = unwrap_matrix(dest);

   if (transpose_result)
      dest = vtn_ssa_transpose(b, dest);

   return dest;
}

static struct vtn_ssa_value *
mat_times_scalar(struct vtn_builder *b,
                 struct vtn_ssa_value *mat,
                 nir_ssa_def *scalar)
{
   struct vtn_ssa_value *dest = vtn_create_ssa_value(b, mat->type);
   for (unsigned i = 0; i < glsl_get_matrix_columns(mat->type); i++) {
      if (glsl_base_type_is_integer(glsl_get_base_type(mat->type)))
         dest->elems[i]->def = nir_imul(&b->nb, mat->elems[i]->def, scalar);
      else
         dest->elems[i]->def = nir_fmul(&b->nb, mat->elems[i]->def, scalar);
   }

   return dest;
}

nir_ssa_def *
vtn_mediump_downconvert(struct vtn_builder *b, enum glsl_base_type base_type, nir_ssa_def *def)
{
   if (def->bit_size == 16)
      return def;

   switch (base_type) {
   case GLSL_TYPE_FLOAT:
      return nir_f2fmp(&b->nb, def);
   case GLSL_TYPE_INT:
   case GLSL_TYPE_UINT:
      return nir_i2imp(&b->nb, def);
   /* Workaround for 3DMark Wild Life which has RelaxedPrecision on
    * OpLogical* operations (which is forbidden by spec).
    */
   case GLSL_TYPE_BOOL:
      return def;
   default:
      unreachable("bad relaxed precision input type");
   }
}

struct vtn_ssa_value *
vtn_mediump_downconvert_value(struct vtn_builder *b, struct vtn_ssa_value *src)
{
   if (!src)
      return src;

   struct vtn_ssa_value *srcmp = vtn_create_ssa_value(b, src->type);

   if (src->transposed) {
      srcmp->transposed = vtn_mediump_downconvert_value(b, src->transposed);
   } else {
      enum glsl_base_type base_type = glsl_get_base_type(src->type);

      if (glsl_type_is_vector_or_scalar(src->type)) {
         srcmp->def = vtn_mediump_downconvert(b, base_type, src->def);
      } else {
         assert(glsl_get_base_type(src->type) == GLSL_TYPE_FLOAT);
         for (int i = 0; i < glsl_get_matrix_columns(src->type); i++)
            srcmp->elems[i]->def = vtn_mediump_downconvert(b, base_type, src->elems[i]->def);
      }
   }

   return srcmp;
}

static struct vtn_ssa_value *
vtn_handle_matrix_alu(struct vtn_builder *b, SpvOp opcode,
                      struct vtn_ssa_value *src0, struct vtn_ssa_value *src1)
{
   switch (opcode) {
   case SpvOpFNegate: {
      struct vtn_ssa_value *dest = vtn_create_ssa_value(b, src0->type);
      unsigned cols = glsl_get_matrix_columns(src0->type);
      for (unsigned i = 0; i < cols; i++)
         dest->elems[i]->def = nir_fneg(&b->nb, src0->elems[i]->def);
      return dest;
   }

   case SpvOpFAdd: {
      struct vtn_ssa_value *dest = vtn_create_ssa_value(b, src0->type);
      unsigned cols = glsl_get_matrix_columns(src0->type);
      for (unsigned i = 0; i < cols; i++)
         dest->elems[i]->def =
            nir_fadd(&b->nb, src0->elems[i]->def, src1->elems[i]->def);
      return dest;
   }

   case SpvOpFSub: {
      struct vtn_ssa_value *dest = vtn_create_ssa_value(b, src0->type);
      unsigned cols = glsl_get_matrix_columns(src0->type);
      for (unsigned i = 0; i < cols; i++)
         dest->elems[i]->def =
            nir_fsub(&b->nb, src0->elems[i]->def, src1->elems[i]->def);
      return dest;
   }

   case SpvOpTranspose:
      return vtn_ssa_transpose(b, src0);

   case SpvOpMatrixTimesScalar:
      if (src0->transposed) {
         return vtn_ssa_transpose(b, mat_times_scalar(b, src0->transposed,
                                                         src1->def));
      } else {
         return mat_times_scalar(b, src0, src1->def);
      }
      break;

   case SpvOpVectorTimesMatrix:
   case SpvOpMatrixTimesVector:
   case SpvOpMatrixTimesMatrix:
      if (opcode == SpvOpVectorTimesMatrix) {
         return matrix_multiply(b, vtn_ssa_transpose(b, src1), src0);
      } else {
         return matrix_multiply(b, src0, src1);
      }
      break;

   default: vtn_fail_with_opcode("unknown matrix opcode", opcode);
   }
}

static nir_alu_type
convert_op_src_type(SpvOp opcode)
{
   switch (opcode) {
   case SpvOpFConvert:
   case SpvOpConvertFToS:
   case SpvOpConvertFToU:
      return nir_type_float;
   case SpvOpSConvert:
   case SpvOpConvertSToF:
   case SpvOpSatConvertSToU:
      return nir_type_int;
   case SpvOpUConvert:
   case SpvOpConvertUToF:
   case SpvOpSatConvertUToS:
      return nir_type_uint;
   default:
      unreachable("Unhandled conversion op");
   }
}

static nir_alu_type
convert_op_dst_type(SpvOp opcode)
{
   switch (opcode) {
   case SpvOpFConvert:
   case SpvOpConvertSToF:
   case SpvOpConvertUToF:
      return nir_type_float;
   case SpvOpSConvert:
   case SpvOpConvertFToS:
   case SpvOpSatConvertUToS:
      return nir_type_int;
   case SpvOpUConvert:
   case SpvOpConvertFToU:
   case SpvOpSatConvertSToU:
      return nir_type_uint;
   default:
      unreachable("Unhandled conversion op");
   }
}

nir_op
vtn_nir_alu_op_for_spirv_opcode(struct vtn_builder *b,
                                SpvOp opcode, bool *swap, bool *exact,
                                unsigned src_bit_size, unsigned dst_bit_size)
{
   /* Indicates that the first two arguments should be swapped.  This is
    * used for implementing greater-than and less-than-or-equal.
    */
   *swap = false;

   *exact = false;

   switch (opcode) {
   case SpvOpSNegate:            return nir_op_ineg;
   case SpvOpFNegate:            return nir_op_fneg;
   case SpvOpNot:                return nir_op_inot;
   case SpvOpIAdd:               return nir_op_iadd;
   case SpvOpFAdd:               return nir_op_fadd;
   case SpvOpISub:               return nir_op_isub;
   case SpvOpFSub:               return nir_op_fsub;
   case SpvOpIMul:               return nir_op_imul;
   case SpvOpFMul:               return nir_op_fmul;
   case SpvOpUDiv:               return nir_op_udiv;
   case SpvOpSDiv:               return nir_op_idiv;
   case SpvOpFDiv:               return nir_op_fdiv;
   case SpvOpUMod:               return nir_op_umod;
   case SpvOpSMod:               return nir_op_imod;
   case SpvOpFMod:               return nir_op_fmod;
   case SpvOpSRem:               return nir_op_irem;
   case SpvOpFRem:               return nir_op_frem;

   case SpvOpShiftRightLogical:     return nir_op_ushr;
   case SpvOpShiftRightArithmetic:  return nir_op_ishr;
   case SpvOpShiftLeftLogical:      return nir_op_ishl;
   case SpvOpLogicalOr:             return nir_op_ior;
   case SpvOpLogicalEqual:          return nir_op_ieq;
   case SpvOpLogicalNotEqual:       return nir_op_ine;
   case SpvOpLogicalAnd:            return nir_op_iand;
   case SpvOpLogicalNot:            return nir_op_inot;
   case SpvOpBitwiseOr:             return nir_op_ior;
   case SpvOpBitwiseXor:            return nir_op_ixor;
   case SpvOpBitwiseAnd:            return nir_op_iand;
   case SpvOpSelect:                return nir_op_bcsel;
   case SpvOpIEqual:                return nir_op_ieq;

   case SpvOpBitFieldInsert:        return nir_op_bitfield_insert;
   case SpvOpBitFieldSExtract:      return nir_op_ibitfield_extract;
   case SpvOpBitFieldUExtract:      return nir_op_ubitfield_extract;
   case SpvOpBitReverse:            return nir_op_bitfield_reverse;

   case SpvOpUCountLeadingZerosINTEL: return nir_op_uclz;
   /* SpvOpUCountTrailingZerosINTEL is handled elsewhere. */
   case SpvOpAbsISubINTEL:          return nir_op_uabs_isub;
   case SpvOpAbsUSubINTEL:          return nir_op_uabs_usub;
   case SpvOpIAddSatINTEL:          return nir_op_iadd_sat;
   case SpvOpUAddSatINTEL:          return nir_op_uadd_sat;
   case SpvOpIAverageINTEL:         return nir_op_ihadd;
   case SpvOpUAverageINTEL:         return nir_op_uhadd;
   case SpvOpIAverageRoundedINTEL:  return nir_op_irhadd;
   case SpvOpUAverageRoundedINTEL:  return nir_op_urhadd;
   case SpvOpISubSatINTEL:          return nir_op_isub_sat;
   case SpvOpUSubSatINTEL:          return nir_op_usub_sat;
   case SpvOpIMul32x16INTEL:        return nir_op_imul_32x16;
   case SpvOpUMul32x16INTEL:        return nir_op_umul_32x16;

   /* The ordered / unordered operators need special implementation besides
    * the logical operator to use since they also need to check if operands are
    * ordered.
    */
   case SpvOpFOrdEqual:                            *exact = true;  return nir_op_feq;
   case SpvOpFUnordEqual:                          *exact = true;  return nir_op_feq;
   case SpvOpINotEqual:                                            return nir_op_ine;
   case SpvOpLessOrGreater:                        /* Deprecated, use OrdNotEqual */
   case SpvOpFOrdNotEqual:                         *exact = true;  return nir_op_fneu;
   case SpvOpFUnordNotEqual:                       *exact = true;  return nir_op_fneu;
   case SpvOpULessThan:                                            return nir_op_ult;
   case SpvOpSLessThan:                                            return nir_op_ilt;
   case SpvOpFOrdLessThan:                         *exact = true;  return nir_op_flt;
   case SpvOpFUnordLessThan:                       *exact = true;  return nir_op_flt;
   case SpvOpUGreaterThan:          *swap = true;                  return nir_op_ult;
   case SpvOpSGreaterThan:          *swap = true;                  return nir_op_ilt;
   case SpvOpFOrdGreaterThan:       *swap = true;  *exact = true;  return nir_op_flt;
   case SpvOpFUnordGreaterThan:     *swap = true;  *exact = true;  return nir_op_flt;
   case SpvOpULessThanEqual:        *swap = true;                  return nir_op_uge;
   case SpvOpSLessThanEqual:        *swap = true;                  return nir_op_ige;
   case SpvOpFOrdLessThanEqual:     *swap = true;  *exact = true;  return nir_op_fge;
   case SpvOpFUnordLessThanEqual:   *swap = true;  *exact = true;  return nir_op_fge;
   case SpvOpUGreaterThanEqual:                                    return nir_op_uge;
   case SpvOpSGreaterThanEqual:                                    return nir_op_ige;
   case SpvOpFOrdGreaterThanEqual:                 *exact = true;  return nir_op_fge;
   case SpvOpFUnordGreaterThanEqual:               *exact = true;  return nir_op_fge;

   /* Conversions: */
   case SpvOpQuantizeToF16:         return nir_op_fquantize2f16;
   case SpvOpUConvert:
   case SpvOpConvertFToU:
   case SpvOpConvertFToS:
   case SpvOpConvertSToF:
   case SpvOpConvertUToF:
   case SpvOpSConvert:
   case SpvOpFConvert: {
      nir_alu_type src_type = convert_op_src_type(opcode) | src_bit_size;
      nir_alu_type dst_type = convert_op_dst_type(opcode) | dst_bit_size;
      return nir_type_conversion_op(src_type, dst_type, nir_rounding_mode_undef);
   }

   case SpvOpPtrCastToGeneric:   return nir_op_mov;
   case SpvOpGenericCastToPtr:   return nir_op_mov;

   /* Derivatives: */
   case SpvOpDPdx:         return nir_op_fddx;
   case SpvOpDPdy:         return nir_op_fddy;
   case SpvOpDPdxFine:     return nir_op_fddx_fine;
   case SpvOpDPdyFine:     return nir_op_fddy_fine;
   case SpvOpDPdxCoarse:   return nir_op_fddx_coarse;
   case SpvOpDPdyCoarse:   return nir_op_fddy_coarse;

   case SpvOpIsNormal:     return nir_op_fisnormal;
   case SpvOpIsFinite:     return nir_op_fisfinite;

   default:
      vtn_fail("No NIR equivalent: %u", opcode);
   }
}

static void
handle_no_contraction(struct vtn_builder *b, UNUSED struct vtn_value *val,
                      UNUSED int member, const struct vtn_decoration *dec,
                      UNUSED void *_void)
{
   vtn_assert(dec->scope == VTN_DEC_DECORATION);
   if (dec->decoration != SpvDecorationNoContraction)
      return;

   b->nb.exact = true;
}

void
vtn_handle_no_contraction(struct vtn_builder *b, struct vtn_value *val)
{
   vtn_foreach_decoration(b, val, handle_no_contraction, NULL);
}

nir_rounding_mode
vtn_rounding_mode_to_nir(struct vtn_builder *b, SpvFPRoundingMode mode)
{
   switch (mode) {
   case SpvFPRoundingModeRTE:
      return nir_rounding_mode_rtne;
   case SpvFPRoundingModeRTZ:
      return nir_rounding_mode_rtz;
   case SpvFPRoundingModeRTP:
      vtn_fail_if(b->shader->info.stage != MESA_SHADER_KERNEL,
                  "FPRoundingModeRTP is only supported in kernels");
      return nir_rounding_mode_ru;
   case SpvFPRoundingModeRTN:
      vtn_fail_if(b->shader->info.stage != MESA_SHADER_KERNEL,
                  "FPRoundingModeRTN is only supported in kernels");
      return nir_rounding_mode_rd;
   default:
      vtn_fail("Unsupported rounding mode: %s",
               spirv_fproundingmode_to_string(mode));
      break;
   }
}

struct conversion_opts {
   nir_rounding_mode rounding_mode;
   bool saturate;
};

static void
handle_conversion_opts(struct vtn_builder *b, UNUSED struct vtn_value *val,
                       UNUSED int member,
                       const struct vtn_decoration *dec, void *_opts)
{
   struct conversion_opts *opts = _opts;

   switch (dec->decoration) {
   case SpvDecorationFPRoundingMode:
      opts->rounding_mode = vtn_rounding_mode_to_nir(b, dec->operands[0]);
      break;

   case SpvDecorationSaturatedConversion:
      vtn_fail_if(b->shader->info.stage != MESA_SHADER_KERNEL,
                  "Saturated conversions are only allowed in kernels");
      opts->saturate = true;
      break;

   default:
      break;
   }
}

static void
handle_no_wrap(UNUSED struct vtn_builder *b, UNUSED struct vtn_value *val,
               UNUSED int member,
               const struct vtn_decoration *dec, void *_alu)
{
   nir_alu_instr *alu = _alu;
   switch (dec->decoration) {
   case SpvDecorationNoSignedWrap:
      alu->no_signed_wrap = true;
      break;
   case SpvDecorationNoUnsignedWrap:
      alu->no_unsigned_wrap = true;
      break;
   default:
      /* Do nothing. */
      break;
   }
}

static void
vtn_value_is_relaxed_precision_cb(struct vtn_builder *b,
                          struct vtn_value *val, int member,
                          const struct vtn_decoration *dec, void *void_ctx)
{
   bool *relaxed_precision = void_ctx;
   switch (dec->decoration) {
   case SpvDecorationRelaxedPrecision:
      *relaxed_precision = true;
      break;

   default:
      break;
   }
}

bool
vtn_value_is_relaxed_precision(struct vtn_builder *b, struct vtn_value *val)
{
   bool result = false;
   vtn_foreach_decoration(b, val,
                          vtn_value_is_relaxed_precision_cb, &result);
   return result;
}

static bool
vtn_alu_op_mediump_16bit(struct vtn_builder *b, SpvOp opcode, struct vtn_value *dest_val)
{
   if (!b->options->mediump_16bit_alu || !vtn_value_is_relaxed_precision(b, dest_val))
      return false;

   switch (opcode) {
   case SpvOpDPdx:
   case SpvOpDPdy:
   case SpvOpDPdxFine:
   case SpvOpDPdyFine:
   case SpvOpDPdxCoarse:
   case SpvOpDPdyCoarse:
   case SpvOpFwidth:
   case SpvOpFwidthFine:
   case SpvOpFwidthCoarse:
      return b->options->mediump_16bit_derivatives;
   default:
      return true;
   }
}

static nir_ssa_def *
vtn_mediump_upconvert(struct vtn_builder *b, enum glsl_base_type base_type, nir_ssa_def *def)
{
   if (def->bit_size != 16)
      return def;

   switch (base_type) {
   case GLSL_TYPE_FLOAT:
      return nir_f2f32(&b->nb, def);
   case GLSL_TYPE_INT:
      return nir_i2i32(&b->nb, def);
   case GLSL_TYPE_UINT:
      return nir_u2u32(&b->nb, def);
   default:
      unreachable("bad relaxed precision output type");
   }
}

void
vtn_mediump_upconvert_value(struct vtn_builder *b, struct vtn_ssa_value *value)
{
   enum glsl_base_type base_type = glsl_get_base_type(value->type);

   if (glsl_type_is_vector_or_scalar(value->type)) {
      value->def = vtn_mediump_upconvert(b, base_type, value->def);
   } else {
      for (int i = 0; i < glsl_get_matrix_columns(value->type); i++)
         value->elems[i]->def = vtn_mediump_upconvert(b, base_type, value->elems[i]->def);
   }
}

void
vtn_handle_alu(struct vtn_builder *b, SpvOp opcode,
               const uint32_t *w, unsigned count)
{
   struct vtn_value *dest_val = vtn_untyped_value(b, w[2]);
   const struct glsl_type *dest_type = vtn_get_type(b, w[1])->type;

   vtn_handle_no_contraction(b, dest_val);
   bool mediump_16bit = vtn_alu_op_mediump_16bit(b, opcode, dest_val);

   /* Collect the various SSA sources */
   const unsigned num_inputs = count - 3;
   struct vtn_ssa_value *vtn_src[4] = { NULL, };
   for (unsigned i = 0; i < num_inputs; i++) {
      vtn_src[i] = vtn_ssa_value(b, w[i + 3]);
      if (mediump_16bit)
         vtn_src[i] = vtn_mediump_downconvert_value(b, vtn_src[i]);
   }

   if (glsl_type_is_matrix(vtn_src[0]->type) ||
       (num_inputs >= 2 && glsl_type_is_matrix(vtn_src[1]->type))) {
      struct vtn_ssa_value *dest = vtn_handle_matrix_alu(b, opcode, vtn_src[0], vtn_src[1]);

      if (mediump_16bit)
         vtn_mediump_upconvert_value(b, dest);

      vtn_push_ssa_value(b, w[2], dest);
      b->nb.exact = b->exact;
      return;
   }

   struct vtn_ssa_value *dest = vtn_create_ssa_value(b, dest_type);
   nir_ssa_def *src[4] = { NULL, };
   for (unsigned i = 0; i < num_inputs; i++) {
      vtn_assert(glsl_type_is_vector_or_scalar(vtn_src[i]->type));
      src[i] = vtn_src[i]->def;
   }

   switch (opcode) {
   case SpvOpAny:
      dest->def = nir_bany(&b->nb, src[0]);
      break;

   case SpvOpAll:
      dest->def = nir_ball(&b->nb, src[0]);
      break;

   case SpvOpOuterProduct: {
      for (unsigned i = 0; i < src[1]->num_components; i++) {
         dest->elems[i]->def =
            nir_fmul(&b->nb, src[0], nir_channel(&b->nb, src[1], i));
      }
      break;
   }

   case SpvOpDot:
      dest->def = nir_fdot(&b->nb, src[0], src[1]);
      break;

   case SpvOpIAddCarry:
      vtn_assert(glsl_type_is_struct_or_ifc(dest_type));
      dest->elems[0]->def = nir_iadd(&b->nb, src[0], src[1]);
      dest->elems[1]->def = nir_uadd_carry(&b->nb, src[0], src[1]);
      break;

   case SpvOpISubBorrow:
      vtn_assert(glsl_type_is_struct_or_ifc(dest_type));
      dest->elems[0]->def = nir_isub(&b->nb, src[0], src[1]);
      dest->elems[1]->def = nir_usub_borrow(&b->nb, src[0], src[1]);
      break;

   case SpvOpUMulExtended: {
      vtn_assert(glsl_type_is_struct_or_ifc(dest_type));
      if (src[0]->bit_size == 32) {
         nir_ssa_def *umul = nir_umul_2x32_64(&b->nb, src[0], src[1]);
         dest->elems[0]->def = nir_unpack_64_2x32_split_x(&b->nb, umul);
         dest->elems[1]->def = nir_unpack_64_2x32_split_y(&b->nb, umul);
      } else {
         dest->elems[0]->def = nir_imul(&b->nb, src[0], src[1]);
         dest->elems[1]->def = nir_umul_high(&b->nb, src[0], src[1]);
      }
      break;
   }

   case SpvOpSMulExtended: {
      vtn_assert(glsl_type_is_struct_or_ifc(dest_type));
      if (src[0]->bit_size == 32) {
         nir_ssa_def *umul = nir_imul_2x32_64(&b->nb, src[0], src[1]);
         dest->elems[0]->def = nir_unpack_64_2x32_split_x(&b->nb, umul);
         dest->elems[1]->def = nir_unpack_64_2x32_split_y(&b->nb, umul);
      } else {
         dest->elems[0]->def = nir_imul(&b->nb, src[0], src[1]);
         dest->elems[1]->def = nir_imul_high(&b->nb, src[0], src[1]);
      }
      break;
   }

   case SpvOpFwidth:
      dest->def = nir_fadd(&b->nb,
                               nir_fabs(&b->nb, nir_fddx(&b->nb, src[0])),
                               nir_fabs(&b->nb, nir_fddy(&b->nb, src[0])));
      break;
   case SpvOpFwidthFine:
      dest->def = nir_fadd(&b->nb,
                               nir_fabs(&b->nb, nir_fddx_fine(&b->nb, src[0])),
                               nir_fabs(&b->nb, nir_fddy_fine(&b->nb, src[0])));
      break;
   case SpvOpFwidthCoarse:
      dest->def = nir_fadd(&b->nb,
                               nir_fabs(&b->nb, nir_fddx_coarse(&b->nb, src[0])),
                               nir_fabs(&b->nb, nir_fddy_coarse(&b->nb, src[0])));
      break;

   case SpvOpVectorTimesScalar:
      /* The builder will take care of splatting for us. */
      dest->def = nir_fmul(&b->nb, src[0], src[1]);
      break;

   case SpvOpIsNan: {
      const bool save_exact = b->nb.exact;

      b->nb.exact = true;
      dest->def = nir_fneu(&b->nb, src[0], src[0]);
      b->nb.exact = save_exact;
      break;
   }

   case SpvOpOrdered: {
      const bool save_exact = b->nb.exact;

      b->nb.exact = true;
      dest->def = nir_iand(&b->nb, nir_feq(&b->nb, src[0], src[0]),
                                   nir_feq(&b->nb, src[1], src[1]));
      b->nb.exact = save_exact;
      break;
   }

   case SpvOpUnordered: {
      const bool save_exact = b->nb.exact;

      b->nb.exact = true;
      dest->def = nir_ior(&b->nb, nir_fneu(&b->nb, src[0], src[0]),
                                  nir_fneu(&b->nb, src[1], src[1]));
      b->nb.exact = save_exact;
      break;
   }

   case SpvOpIsInf: {
      nir_ssa_def *inf = nir_imm_floatN_t(&b->nb, INFINITY, src[0]->bit_size);
      dest->def = nir_ieq(&b->nb, nir_fabs(&b->nb, src[0]), inf);
      break;
   }

   case SpvOpFUnordEqual: {
      const bool save_exact = b->nb.exact;

      b->nb.exact = true;

      /* This could also be implemented as !(a < b || b < a).  If one or both
       * of the source are numbers, later optimization passes can easily
       * eliminate the isnan() checks.  This may trim the sequence down to a
       * single (a == b) operation.  Otherwise, the optimizer can transform
       * whatever is left to !(a < b || b < a).  Since some applications will
       * open-code this sequence, these optimizations are needed anyway.
       */
      dest->def =
         nir_ior(&b->nb,
                 nir_feq(&b->nb, src[0], src[1]),
                 nir_ior(&b->nb,
                         nir_fneu(&b->nb, src[0], src[0]),
                         nir_fneu(&b->nb, src[1], src[1])));

      b->nb.exact = save_exact;
      break;
   }

   case SpvOpFUnordLessThan:
   case SpvOpFUnordGreaterThan:
   case SpvOpFUnordLessThanEqual:
   case SpvOpFUnordGreaterThanEqual: {
      bool swap;
      bool unused_exact;
      unsigned src_bit_size = glsl_get_bit_size(vtn_src[0]->type);
      unsigned dst_bit_size = glsl_get_bit_size(dest_type);
      nir_op op = vtn_nir_alu_op_for_spirv_opcode(b, opcode, &swap,
                                                  &unused_exact,
                                                  src_bit_size, dst_bit_size);

      if (swap) {
         nir_ssa_def *tmp = src[0];
         src[0] = src[1];
         src[1] = tmp;
      }

      const bool save_exact = b->nb.exact;

      b->nb.exact = true;

      /* Use the property FUnordLessThan(a, b) ≡ !FOrdGreaterThanEqual(a, b). */
      switch (op) {
      case nir_op_fge: op = nir_op_flt; break;
      case nir_op_flt: op = nir_op_fge; break;
      default: unreachable("Impossible opcode.");
      }

      dest->def =
         nir_inot(&b->nb,
                  nir_build_alu(&b->nb, op, src[0], src[1], NULL, NULL));

      b->nb.exact = save_exact;
      break;
   }

   case SpvOpLessOrGreater:
   case SpvOpFOrdNotEqual: {
      /* For all the SpvOpFOrd* comparisons apart from NotEqual, the value
       * from the ALU will probably already be false if the operands are not
       * ordered so we don’t need to handle it specially.
       */
      const bool save_exact = b->nb.exact;

      b->nb.exact = true;

      /* This could also be implemented as (a < b || b < a).  If one or both
       * of the source are numbers, later optimization passes can easily
       * eliminate the isnan() checks.  This may trim the sequence down to a
       * single (a != b) operation.  Otherwise, the optimizer can transform
       * whatever is left to (a < b || b < a).  Since some applications will
       * open-code this sequence, these optimizations are needed anyway.
       */
      dest->def =
         nir_iand(&b->nb,
                  nir_fneu(&b->nb, src[0], src[1]),
                  nir_iand(&b->nb,
                          nir_feq(&b->nb, src[0], src[0]),
                          nir_feq(&b->nb, src[1], src[1])));

      b->nb.exact = save_exact;
      break;
   }

   case SpvOpUConvert:
   case SpvOpConvertFToU:
   case SpvOpConvertFToS:
   case SpvOpConvertSToF:
   case SpvOpConvertUToF:
   case SpvOpSConvert:
   case SpvOpFConvert:
   case SpvOpSatConvertSToU:
   case SpvOpSatConvertUToS: {
      unsigned src_bit_size = src[0]->bit_size;
      unsigned dst_bit_size = glsl_get_bit_size(dest_type);
      nir_alu_type src_type = convert_op_src_type(opcode) | src_bit_size;
      nir_alu_type dst_type = convert_op_dst_type(opcode) | dst_bit_size;

      struct conversion_opts opts = {
         .rounding_mode = nir_rounding_mode_undef,
         .saturate = false,
      };
      vtn_foreach_decoration(b, dest_val, handle_conversion_opts, &opts);

      if (opcode == SpvOpSatConvertSToU || opcode == SpvOpSatConvertUToS)
         opts.saturate = true;

      if (b->shader->info.stage == MESA_SHADER_KERNEL) {
         if (opts.rounding_mode == nir_rounding_mode_undef && !opts.saturate) {
            dest->def = nir_type_convert(&b->nb, src[0], src_type, dst_type,
                                         nir_rounding_mode_undef);
         } else {
            dest->def = nir_convert_alu_types(&b->nb, dst_bit_size, src[0],
                                              src_type, dst_type,
                                              opts.rounding_mode, opts.saturate);
         }
      } else {
         vtn_fail_if(opts.rounding_mode != nir_rounding_mode_undef &&
                     dst_type != nir_type_float16,
                     "Rounding modes are only allowed on conversions to "
                     "16-bit float types");
         dest->def = nir_type_convert(&b->nb, src[0], src_type, dst_type,
                                      opts.rounding_mode);
      }
      break;
   }

   case SpvOpBitFieldInsert:
   case SpvOpBitFieldSExtract:
   case SpvOpBitFieldUExtract:
   case SpvOpShiftLeftLogical:
   case SpvOpShiftRightArithmetic:
   case SpvOpShiftRightLogical: {
      bool swap;
      bool exact;
      unsigned src0_bit_size = glsl_get_bit_size(vtn_src[0]->type);
      unsigned dst_bit_size = glsl_get_bit_size(dest_type);
      nir_op op = vtn_nir_alu_op_for_spirv_opcode(b, opcode, &swap, &exact,
                                                  src0_bit_size, dst_bit_size);

      assert(!exact);

      assert (op == nir_op_ushr || op == nir_op_ishr || op == nir_op_ishl ||
              op == nir_op_bitfield_insert || op == nir_op_ubitfield_extract ||
              op == nir_op_ibitfield_extract);

      for (unsigned i = 0; i < nir_op_infos[op].num_inputs; i++) {
         unsigned src_bit_size =
            nir_alu_type_get_type_size(nir_op_infos[op].input_types[i]);
         if (src_bit_size == 0)
            continue;
         if (src_bit_size != src[i]->bit_size) {
            assert(src_bit_size == 32);
            /* Convert the Shift, Offset and Count  operands to 32 bits, which is the bitsize
             * supported by the NIR instructions. See discussion here:
             *
             * https://lists.freedesktop.org/archives/mesa-dev/2018-April/193026.html
             */
            src[i] = nir_u2u32(&b->nb, src[i]);
         }
      }
      dest->def = nir_build_alu(&b->nb, op, src[0], src[1], src[2], src[3]);
      break;
   }

   case SpvOpSignBitSet:
      dest->def = nir_i2b(&b->nb,
         nir_ushr(&b->nb, src[0], nir_imm_int(&b->nb, src[0]->bit_size - 1)));
      break;

   case SpvOpUCountTrailingZerosINTEL:
      dest->def = nir_umin(&b->nb,
                               nir_find_lsb(&b->nb, src[0]),
                               nir_imm_int(&b->nb, 32u));
      break;

   case SpvOpBitCount: {
      /* bit_count always returns int32, but the SPIR-V opcode just says the return
       * value needs to be big enough to store the number of bits.
       */
      dest->def = nir_u2uN(&b->nb, nir_bit_count(&b->nb, src[0]), glsl_get_bit_size(dest_type));
      break;
   }

   case SpvOpSDotKHR:
   case SpvOpUDotKHR:
   case SpvOpSUDotKHR:
   case SpvOpSDotAccSatKHR:
   case SpvOpUDotAccSatKHR:
   case SpvOpSUDotAccSatKHR:
      unreachable("Should have called vtn_handle_integer_dot instead.");

   default: {
      bool swap;
      bool exact;
      unsigned src_bit_size = glsl_get_bit_size(vtn_src[0]->type);
      unsigned dst_bit_size = glsl_get_bit_size(dest_type);
      nir_op op = vtn_nir_alu_op_for_spirv_opcode(b, opcode, &swap,
                                                  &exact,
                                                  src_bit_size, dst_bit_size);

      if (swap) {
         nir_ssa_def *tmp = src[0];
         src[0] = src[1];
         src[1] = tmp;
      }

      switch (op) {
      case nir_op_ishl:
      case nir_op_ishr:
      case nir_op_ushr:
         if (src[1]->bit_size != 32)
            src[1] = nir_u2u32(&b->nb, src[1]);
         break;
      default:
         break;
      }

      const bool save_exact = b->nb.exact;

      if (exact)
         b->nb.exact = true;

      dest->def = nir_build_alu(&b->nb, op, src[0], src[1], src[2], src[3]);

      b->nb.exact = save_exact;
      break;
   } /* default */
   }

   switch (opcode) {
   case SpvOpIAdd:
   case SpvOpIMul:
   case SpvOpISub:
   case SpvOpShiftLeftLogical:
   case SpvOpSNegate: {
      nir_alu_instr *alu = nir_instr_as_alu(dest->def->parent_instr);
      vtn_foreach_decoration(b, dest_val, handle_no_wrap, alu);
      break;
   }
   default:
      /* Do nothing. */
      break;
   }

   if (mediump_16bit)
      vtn_mediump_upconvert_value(b, dest);
   vtn_push_ssa_value(b, w[2], dest);

   b->nb.exact = b->exact;
}

void
vtn_handle_integer_dot(struct vtn_builder *b, SpvOp opcode,
                       const uint32_t *w, unsigned count)
{
   struct vtn_value *dest_val = vtn_untyped_value(b, w[2]);
   const struct glsl_type *dest_type = vtn_get_type(b, w[1])->type;
   const unsigned dest_size = glsl_get_bit_size(dest_type);

   vtn_handle_no_contraction(b, dest_val);

   /* Collect the various SSA sources.
    *
    * Due to the optional "Packed Vector Format" field, determine number of
    * inputs from the opcode.  This differs from vtn_handle_alu.
    */
   const unsigned num_inputs = (opcode == SpvOpSDotAccSatKHR ||
                                opcode == SpvOpUDotAccSatKHR ||
                                opcode == SpvOpSUDotAccSatKHR) ? 3 : 2;

   vtn_assert(count >= num_inputs + 3);

   struct vtn_ssa_value *vtn_src[3] = { NULL, };
   nir_ssa_def *src[3] = { NULL, };

   for (unsigned i = 0; i < num_inputs; i++) {
      vtn_src[i] = vtn_ssa_value(b, w[i + 3]);
      src[i] = vtn_src[i]->def;

      vtn_assert(glsl_type_is_vector_or_scalar(vtn_src[i]->type));
   }

   /* For all of the opcodes *except* SpvOpSUDotKHR and SpvOpSUDotAccSatKHR,
    * the SPV_KHR_integer_dot_product spec says:
    *
    *    _Vector 1_ and _Vector 2_ must have the same type.
    *
    * The practical requirement is the same bit-size and the same number of
    * components.
    */
   vtn_fail_if(glsl_get_bit_size(vtn_src[0]->type) !=
               glsl_get_bit_size(vtn_src[1]->type) ||
               glsl_get_vector_elements(vtn_src[0]->type) !=
               glsl_get_vector_elements(vtn_src[1]->type),
               "Vector 1 and vector 2 source of opcode %s must have the same "
               "type",
               spirv_op_to_string(opcode));

   if (num_inputs == 3) {
      /* The SPV_KHR_integer_dot_product spec says:
       *
       *    The type of Accumulator must be the same as Result Type.
       *
       * The handling of SpvOpSDotAccSatKHR and friends with the packed 4x8
       * types (far below) assumes these types have the same size.
       */
      vtn_fail_if(dest_type != vtn_src[2]->type,
                  "Accumulator type must be the same as Result Type for "
                  "opcode %s",
                  spirv_op_to_string(opcode));
   }

   unsigned packed_bit_size = 8;
   if (glsl_type_is_vector(vtn_src[0]->type)) {
      /* FINISHME: Is this actually as good or better for platforms that don't
       * have the special instructions (i.e., one or both of has_dot_4x8 or
       * has_sudot_4x8 is false)?
       */
      if (glsl_get_vector_elements(vtn_src[0]->type) == 4 &&
          glsl_get_bit_size(vtn_src[0]->type) == 8 &&
          glsl_get_bit_size(dest_type) <= 32) {
         src[0] = nir_pack_32_4x8(&b->nb, src[0]);
         src[1] = nir_pack_32_4x8(&b->nb, src[1]);
      } else if (glsl_get_vector_elements(vtn_src[0]->type) == 2 &&
                 glsl_get_bit_size(vtn_src[0]->type) == 16 &&
                 glsl_get_bit_size(dest_type) <= 32 &&
                 opcode != SpvOpSUDotKHR &&
                 opcode != SpvOpSUDotAccSatKHR) {
         src[0] = nir_pack_32_2x16(&b->nb, src[0]);
         src[1] = nir_pack_32_2x16(&b->nb, src[1]);
         packed_bit_size = 16;
      }
   } else if (glsl_type_is_scalar(vtn_src[0]->type) &&
              glsl_type_is_32bit(vtn_src[0]->type)) {
      /* The SPV_KHR_integer_dot_product spec says:
       *
       *    When _Vector 1_ and _Vector 2_ are scalar integer types, _Packed
       *    Vector Format_ must be specified to select how the integers are to
       *    be interpreted as vectors.
       *
       * The "Packed Vector Format" value follows the last input.
       */
      vtn_assert(count == (num_inputs + 4));
      const SpvPackedVectorFormat pack_format = w[num_inputs + 3];
      vtn_fail_if(pack_format != SpvPackedVectorFormatPackedVectorFormat4x8BitKHR,
                  "Unsupported vector packing format %d for opcode %s",
                  pack_format, spirv_op_to_string(opcode));
   } else {
      vtn_fail_with_opcode("Invalid source types.", opcode);
   }

   nir_ssa_def *dest = NULL;

   if (src[0]->num_components > 1) {
      nir_ssa_def *(*src0_conversion)(nir_builder *, nir_ssa_def *, unsigned);
      nir_ssa_def *(*src1_conversion)(nir_builder *, nir_ssa_def *, unsigned);

      switch (opcode) {
      case SpvOpSDotKHR:
      case SpvOpSDotAccSatKHR:
         src0_conversion = nir_i2iN;
         src1_conversion = nir_i2iN;
         break;

      case SpvOpUDotKHR:
      case SpvOpUDotAccSatKHR:
         src0_conversion = nir_u2uN;
         src1_conversion = nir_u2uN;
         break;

      case SpvOpSUDotKHR:
      case SpvOpSUDotAccSatKHR:
         src0_conversion = nir_i2iN;
         src1_conversion = nir_u2uN;
         break;

      default:
         unreachable("Invalid opcode.");
      }

      /* The SPV_KHR_integer_dot_product spec says:
       *
       *    All components of the input vectors are sign-extended to the bit
       *    width of the result's type. The sign-extended input vectors are
       *    then multiplied component-wise and all components of the vector
       *    resulting from the component-wise multiplication are added
       *    together. The resulting value will equal the low-order N bits of
       *    the correct result R, where N is the result width and R is
       *    computed with enough precision to avoid overflow and underflow.
       */
      const unsigned vector_components =
         glsl_get_vector_elements(vtn_src[0]->type);

      for (unsigned i = 0; i < vector_components; i++) {
         nir_ssa_def *const src0 =
            src0_conversion(&b->nb, nir_channel(&b->nb, src[0], i), dest_size);

         nir_ssa_def *const src1 =
            src1_conversion(&b->nb, nir_channel(&b->nb, src[1], i), dest_size);

         nir_ssa_def *const mul_result = nir_imul(&b->nb, src0, src1);

         dest = (i == 0) ? mul_result : nir_iadd(&b->nb, dest, mul_result);
      }

      if (num_inputs == 3) {
         /* For SpvOpSDotAccSatKHR, the SPV_KHR_integer_dot_product spec says:
          *
          *    Signed integer dot product of _Vector 1_ and _Vector 2_ and
          *    signed saturating addition of the result with _Accumulator_.
          *
          * For SpvOpUDotAccSatKHR, the SPV_KHR_integer_dot_product spec says:
          *
          *    Unsigned integer dot product of _Vector 1_ and _Vector 2_ and
          *    unsigned saturating addition of the result with _Accumulator_.
          *
          * For SpvOpSUDotAccSatKHR, the SPV_KHR_integer_dot_product spec says:
          *
          *    Mixed-signedness integer dot product of _Vector 1_ and _Vector
          *    2_ and signed saturating addition of the result with
          *    _Accumulator_.
          */
         dest = (opcode == SpvOpUDotAccSatKHR)
            ? nir_uadd_sat(&b->nb, dest, src[2])
            : nir_iadd_sat(&b->nb, dest, src[2]);
      }
   } else {
      assert(src[0]->num_components == 1 && src[1]->num_components == 1);
      assert(src[0]->bit_size == 32 && src[1]->bit_size == 32);

      nir_ssa_def *const zero = nir_imm_zero(&b->nb, 1, 32);
      bool is_signed = opcode == SpvOpSDotKHR || opcode == SpvOpSUDotKHR ||
                       opcode == SpvOpSDotAccSatKHR || opcode == SpvOpSUDotAccSatKHR;

      if (packed_bit_size == 16) {
         switch (opcode) {
         case SpvOpSDotKHR:
            dest = nir_sdot_2x16_iadd(&b->nb, src[0], src[1], zero);
            break;
         case SpvOpUDotKHR:
            dest = nir_udot_2x16_uadd(&b->nb, src[0], src[1], zero);
            break;
         case SpvOpSDotAccSatKHR:
            if (dest_size == 32)
               dest = nir_sdot_2x16_iadd_sat(&b->nb, src[0], src[1], src[2]);
            else
               dest = nir_sdot_2x16_iadd(&b->nb, src[0], src[1], zero);
            break;
         case SpvOpUDotAccSatKHR:
            if (dest_size == 32)
               dest = nir_udot_2x16_uadd_sat(&b->nb, src[0], src[1], src[2]);
            else
               dest = nir_udot_2x16_uadd(&b->nb, src[0], src[1], zero);
            break;
         default:
            unreachable("Invalid opcode.");
         }
      } else {
         switch (opcode) {
         case SpvOpSDotKHR:
            dest = nir_sdot_4x8_iadd(&b->nb, src[0], src[1], zero);
            break;
         case SpvOpUDotKHR:
            dest = nir_udot_4x8_uadd(&b->nb, src[0], src[1], zero);
            break;
         case SpvOpSUDotKHR:
            dest = nir_sudot_4x8_iadd(&b->nb, src[0], src[1], zero);
            break;
         case SpvOpSDotAccSatKHR:
            if (dest_size == 32)
               dest = nir_sdot_4x8_iadd_sat(&b->nb, src[0], src[1], src[2]);
            else
               dest = nir_sdot_4x8_iadd(&b->nb, src[0], src[1], zero);
            break;
         case SpvOpUDotAccSatKHR:
            if (dest_size == 32)
               dest = nir_udot_4x8_uadd_sat(&b->nb, src[0], src[1], src[2]);
            else
               dest = nir_udot_4x8_uadd(&b->nb, src[0], src[1], zero);
            break;
         case SpvOpSUDotAccSatKHR:
            if (dest_size == 32)
               dest = nir_sudot_4x8_iadd_sat(&b->nb, src[0], src[1], src[2]);
            else
               dest = nir_sudot_4x8_iadd(&b->nb, src[0], src[1], zero);
            break;
         default:
            unreachable("Invalid opcode.");
         }
      }

      if (dest_size != 32) {
         /* When the accumulator is 32-bits, a NIR dot-product with saturate
          * is generated above.  In all other cases a regular dot-product is
          * generated above, and separate addition with saturate is generated
          * here.
          *
          * The SPV_KHR_integer_dot_product spec says:
          *
          *    If any of the multiplications or additions, with the exception
          *    of the final accumulation, overflow or underflow, the result of
          *    the instruction is undefined.
          *
          * Therefore it is safe to cast the dot-product result down to the
          * size of the accumulator before doing the addition.  Since the
          * result of the dot-product cannot overflow 32-bits, this is also
          * safe to cast up.
          */
         if (num_inputs == 3) {
            dest = is_signed
               ? nir_iadd_sat(&b->nb, nir_i2iN(&b->nb, dest, dest_size), src[2])
               : nir_uadd_sat(&b->nb, nir_u2uN(&b->nb, dest, dest_size), src[2]);
         } else {
            dest = is_signed
               ? nir_i2iN(&b->nb, dest, dest_size)
               : nir_u2uN(&b->nb, dest, dest_size);
         }
      }
   }

   vtn_push_nir_ssa(b, w[2], dest);

   b->nb.exact = b->exact;
}

void
vtn_handle_bitcast(struct vtn_builder *b, const uint32_t *w, unsigned count)
{
   vtn_assert(count == 4);
   /* From the definition of OpBitcast in the SPIR-V 1.2 spec:
    *
    *    "If Result Type has the same number of components as Operand, they
    *    must also have the same component width, and results are computed per
    *    component.
    *
    *    If Result Type has a different number of components than Operand, the
    *    total number of bits in Result Type must equal the total number of
    *    bits in Operand. Let L be the type, either Result Type or Operand’s
    *    type, that has the larger number of components. Let S be the other
    *    type, with the smaller number of components. The number of components
    *    in L must be an integer multiple of the number of components in S.
    *    The first component (that is, the only or lowest-numbered component)
    *    of S maps to the first components of L, and so on, up to the last
    *    component of S mapping to the last components of L. Within this
    *    mapping, any single component of S (mapping to multiple components of
    *    L) maps its lower-ordered bits to the lower-numbered components of L."
    */

   struct vtn_type *type = vtn_get_type(b, w[1]);
   struct nir_ssa_def *src = vtn_get_nir_ssa(b, w[3]);

   vtn_fail_if(src->num_components * src->bit_size !=
               glsl_get_vector_elements(type->type) * glsl_get_bit_size(type->type),
               "Source and destination of OpBitcast must have the same "
               "total number of bits");
   nir_ssa_def *val =
      nir_bitcast_vector(&b->nb, src, glsl_get_bit_size(type->type));
   vtn_push_nir_ssa(b, w[2], val);
}
