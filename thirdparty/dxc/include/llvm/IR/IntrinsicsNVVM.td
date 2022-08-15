//===- IntrinsicsNVVM.td - Defines NVVM intrinsics ---------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines all of the NVVM-specific intrinsics for use with NVPTX.
//
//===----------------------------------------------------------------------===//

def llvm_anyi64ptr_ty     : LLVMAnyPointerType<llvm_i64_ty>;     // (space)i64*

//
// MISC
//

  def int_nvvm_clz_i : GCCBuiltin<"__nvvm_clz_i">,
      Intrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrNoMem]>;
  def int_nvvm_clz_ll : GCCBuiltin<"__nvvm_clz_ll">,
      Intrinsic<[llvm_i32_ty], [llvm_i64_ty], [IntrNoMem]>;

  def int_nvvm_popc_i : GCCBuiltin<"__nvvm_popc_i">,
      Intrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrNoMem]>;
  def int_nvvm_popc_ll : GCCBuiltin<"__nvvm_popc_ll">,
      Intrinsic<[llvm_i32_ty], [llvm_i64_ty], [IntrNoMem]>;

  def int_nvvm_prmt : GCCBuiltin<"__nvvm_prmt">,
      Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
        [IntrNoMem, Commutative]>;

//
// Min Max
//

  def int_nvvm_min_i : GCCBuiltin<"__nvvm_min_i">,
      Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty],
        [IntrNoMem, Commutative]>;
  def int_nvvm_min_ui : GCCBuiltin<"__nvvm_min_ui">,
      Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty],
        [IntrNoMem, Commutative]>;

  def int_nvvm_min_ll : GCCBuiltin<"__nvvm_min_ll">,
      Intrinsic<[llvm_i64_ty], [llvm_i64_ty, llvm_i64_ty],
        [IntrNoMem, Commutative]>;
  def int_nvvm_min_ull : GCCBuiltin<"__nvvm_min_ull">,
      Intrinsic<[llvm_i64_ty], [llvm_i64_ty, llvm_i64_ty],
        [IntrNoMem, Commutative]>;

  def int_nvvm_max_i : GCCBuiltin<"__nvvm_max_i">,
      Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty],
        [IntrNoMem, Commutative]>;
  def int_nvvm_max_ui : GCCBuiltin<"__nvvm_max_ui">,
      Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty],
        [IntrNoMem, Commutative]>;

  def int_nvvm_max_ll : GCCBuiltin<"__nvvm_max_ll">,
      Intrinsic<[llvm_i64_ty], [llvm_i64_ty, llvm_i64_ty],
        [IntrNoMem, Commutative]>;
  def int_nvvm_max_ull : GCCBuiltin<"__nvvm_max_ull">,
      Intrinsic<[llvm_i64_ty], [llvm_i64_ty, llvm_i64_ty],
        [IntrNoMem, Commutative]>;

  def int_nvvm_fmin_f : GCCBuiltin<"__nvvm_fmin_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty],
        [IntrNoMem, Commutative]>;
  def int_nvvm_fmin_ftz_f : GCCBuiltin<"__nvvm_fmin_ftz_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty],
        [IntrNoMem, Commutative]>;

  def int_nvvm_fmax_f : GCCBuiltin<"__nvvm_fmax_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty]
        , [IntrNoMem, Commutative]>;
  def int_nvvm_fmax_ftz_f : GCCBuiltin<"__nvvm_fmax_ftz_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty],
        [IntrNoMem, Commutative]>;

  def int_nvvm_fmin_d : GCCBuiltin<"__nvvm_fmin_d">,
      Intrinsic<[llvm_double_ty], [llvm_double_ty, llvm_double_ty],
        [IntrNoMem, Commutative]>;
  def int_nvvm_fmax_d : GCCBuiltin<"__nvvm_fmax_d">,
      Intrinsic<[llvm_double_ty], [llvm_double_ty, llvm_double_ty],
        [IntrNoMem, Commutative]>;

//
// Multiplication
//

  def int_nvvm_mulhi_i : GCCBuiltin<"__nvvm_mulhi_i">,
      Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty],
        [IntrNoMem, Commutative]>;
  def int_nvvm_mulhi_ui : GCCBuiltin<"__nvvm_mulhi_ui">,
      Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty],
        [IntrNoMem, Commutative]>;

  def int_nvvm_mulhi_ll : GCCBuiltin<"__nvvm_mulhi_ll">,
      Intrinsic<[llvm_i64_ty], [llvm_i64_ty, llvm_i64_ty],
        [IntrNoMem, Commutative]>;
  def int_nvvm_mulhi_ull : GCCBuiltin<"__nvvm_mulhi_ull">,
      Intrinsic<[llvm_i64_ty], [llvm_i64_ty, llvm_i64_ty],
        [IntrNoMem, Commutative]>;

  def int_nvvm_mul_rn_ftz_f : GCCBuiltin<"__nvvm_mul_rn_ftz_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty],
        [IntrNoMem, Commutative]>;
  def int_nvvm_mul_rn_f : GCCBuiltin<"__nvvm_mul_rn_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty],
        [IntrNoMem, Commutative]>;
  def int_nvvm_mul_rz_ftz_f : GCCBuiltin<"__nvvm_mul_rz_ftz_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty],
        [IntrNoMem, Commutative]>;
  def int_nvvm_mul_rz_f : GCCBuiltin<"__nvvm_mul_rz_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty],
        [IntrNoMem, Commutative]>;
  def int_nvvm_mul_rm_ftz_f : GCCBuiltin<"__nvvm_mul_rm_ftz_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty],
        [IntrNoMem, Commutative]>;
  def int_nvvm_mul_rm_f : GCCBuiltin<"__nvvm_mul_rm_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty],
        [IntrNoMem, Commutative]>;
  def int_nvvm_mul_rp_ftz_f : GCCBuiltin<"__nvvm_mul_rp_ftz_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty],
        [IntrNoMem, Commutative]>;
  def int_nvvm_mul_rp_f : GCCBuiltin<"__nvvm_mul_rp_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty],
        [IntrNoMem, Commutative]>;

  def int_nvvm_mul_rn_d : GCCBuiltin<"__nvvm_mul_rn_d">,
      Intrinsic<[llvm_double_ty], [llvm_double_ty, llvm_double_ty],
        [IntrNoMem, Commutative]>;
  def int_nvvm_mul_rz_d : GCCBuiltin<"__nvvm_mul_rz_d">,
      Intrinsic<[llvm_double_ty], [llvm_double_ty, llvm_double_ty],
        [IntrNoMem, Commutative]>;
  def int_nvvm_mul_rm_d : GCCBuiltin<"__nvvm_mul_rm_d">,
      Intrinsic<[llvm_double_ty], [llvm_double_ty, llvm_double_ty],
        [IntrNoMem, Commutative]>;
  def int_nvvm_mul_rp_d : GCCBuiltin<"__nvvm_mul_rp_d">,
      Intrinsic<[llvm_double_ty], [llvm_double_ty, llvm_double_ty],
        [IntrNoMem, Commutative]>;

  def int_nvvm_mul24_i : GCCBuiltin<"__nvvm_mul24_i">,
      Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty],
        [IntrNoMem, Commutative]>;
  def int_nvvm_mul24_ui : GCCBuiltin<"__nvvm_mul24_ui">,
      Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty],
        [IntrNoMem, Commutative]>;

//
// Div
//

  def int_nvvm_div_approx_ftz_f : GCCBuiltin<"__nvvm_div_approx_ftz_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty],
        [IntrNoMem, Commutative]>;
  def int_nvvm_div_approx_f : GCCBuiltin<"__nvvm_div_approx_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty],
        [IntrNoMem, Commutative]>;

  def int_nvvm_div_rn_ftz_f : GCCBuiltin<"__nvvm_div_rn_ftz_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty],
        [IntrNoMem, Commutative]>;
  def int_nvvm_div_rn_f : GCCBuiltin<"__nvvm_div_rn_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty],
        [IntrNoMem, Commutative]>;

  def int_nvvm_div_rz_ftz_f : GCCBuiltin<"__nvvm_div_rz_ftz_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty],
        [IntrNoMem, Commutative]>;
  def int_nvvm_div_rz_f : GCCBuiltin<"__nvvm_div_rz_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty],
        [IntrNoMem, Commutative]>;

  def int_nvvm_div_rm_ftz_f : GCCBuiltin<"__nvvm_div_rm_ftz_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty],
        [IntrNoMem, Commutative]>;
  def int_nvvm_div_rm_f : GCCBuiltin<"__nvvm_div_rm_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty],
        [IntrNoMem, Commutative]>;

  def int_nvvm_div_rp_ftz_f : GCCBuiltin<"__nvvm_div_rp_ftz_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty],
        [IntrNoMem, Commutative]>;
  def int_nvvm_div_rp_f : GCCBuiltin<"__nvvm_div_rp_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty],
        [IntrNoMem, Commutative]>;

  def int_nvvm_div_rn_d : GCCBuiltin<"__nvvm_div_rn_d">,
      Intrinsic<[llvm_double_ty], [llvm_double_ty, llvm_double_ty],
        [IntrNoMem, Commutative]>;
  def int_nvvm_div_rz_d : GCCBuiltin<"__nvvm_div_rz_d">,
      Intrinsic<[llvm_double_ty], [llvm_double_ty, llvm_double_ty],
        [IntrNoMem, Commutative]>;
  def int_nvvm_div_rm_d : GCCBuiltin<"__nvvm_div_rm_d">,
      Intrinsic<[llvm_double_ty], [llvm_double_ty, llvm_double_ty],
        [IntrNoMem, Commutative]>;
  def int_nvvm_div_rp_d : GCCBuiltin<"__nvvm_div_rp_d">,
      Intrinsic<[llvm_double_ty], [llvm_double_ty, llvm_double_ty],
        [IntrNoMem, Commutative]>;

//
// Brev
//

  def int_nvvm_brev32 : GCCBuiltin<"__nvvm_brev32">,
      Intrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrNoMem]>;
  def int_nvvm_brev64 : GCCBuiltin<"__nvvm_brev64">,
      Intrinsic<[llvm_i64_ty], [llvm_i64_ty], [IntrNoMem]>;

//
// Sad
//

  def int_nvvm_sad_i : GCCBuiltin<"__nvvm_sad_i">,
      Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
        [IntrNoMem, Commutative]>;
  def int_nvvm_sad_ui : GCCBuiltin<"__nvvm_sad_ui">,
      Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
        [IntrNoMem, Commutative]>;

//
// Floor  Ceil
//

  def int_nvvm_floor_ftz_f : GCCBuiltin<"__nvvm_floor_ftz_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty], [IntrNoMem]>;
  def int_nvvm_floor_f : GCCBuiltin<"__nvvm_floor_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty], [IntrNoMem]>;
  def int_nvvm_floor_d : GCCBuiltin<"__nvvm_floor_d">,
      Intrinsic<[llvm_double_ty], [llvm_double_ty], [IntrNoMem]>;

  def int_nvvm_ceil_ftz_f : GCCBuiltin<"__nvvm_ceil_ftz_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty], [IntrNoMem]>;
  def int_nvvm_ceil_f : GCCBuiltin<"__nvvm_ceil_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty], [IntrNoMem]>;
  def int_nvvm_ceil_d : GCCBuiltin<"__nvvm_ceil_d">,
      Intrinsic<[llvm_double_ty], [llvm_double_ty], [IntrNoMem]>;

//
// Abs
//

  def int_nvvm_abs_i : GCCBuiltin<"__nvvm_abs_i">,
      Intrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrNoMem]>;
  def int_nvvm_abs_ll : GCCBuiltin<"__nvvm_abs_ll">,
      Intrinsic<[llvm_i64_ty], [llvm_i64_ty], [IntrNoMem]>;

  def int_nvvm_fabs_ftz_f : GCCBuiltin<"__nvvm_fabs_ftz_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty], [IntrNoMem]>;
  def int_nvvm_fabs_f : GCCBuiltin<"__nvvm_fabs_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty], [IntrNoMem]>;

  def int_nvvm_fabs_d : GCCBuiltin<"__nvvm_fabs_d">,
      Intrinsic<[llvm_double_ty], [llvm_double_ty], [IntrNoMem]>;

//
// Round
//

  def int_nvvm_round_ftz_f : GCCBuiltin<"__nvvm_round_ftz_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty], [IntrNoMem]>;
  def int_nvvm_round_f : GCCBuiltin<"__nvvm_round_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty], [IntrNoMem]>;

  def int_nvvm_round_d : GCCBuiltin<"__nvvm_round_d">,
      Intrinsic<[llvm_double_ty], [llvm_double_ty], [IntrNoMem]>;

//
// Trunc
//

  def int_nvvm_trunc_ftz_f : GCCBuiltin<"__nvvm_trunc_ftz_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty], [IntrNoMem]>;
  def int_nvvm_trunc_f : GCCBuiltin<"__nvvm_trunc_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty], [IntrNoMem]>;

  def int_nvvm_trunc_d : GCCBuiltin<"__nvvm_trunc_d">,
      Intrinsic<[llvm_double_ty], [llvm_double_ty], [IntrNoMem]>;

//
// Saturate
//

  def int_nvvm_saturate_ftz_f : GCCBuiltin<"__nvvm_saturate_ftz_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty], [IntrNoMem]>;
  def int_nvvm_saturate_f : GCCBuiltin<"__nvvm_saturate_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty], [IntrNoMem]>;

  def int_nvvm_saturate_d : GCCBuiltin<"__nvvm_saturate_d">,
      Intrinsic<[llvm_double_ty], [llvm_double_ty], [IntrNoMem]>;

//
// Exp2  Log2
//

  def int_nvvm_ex2_approx_ftz_f : GCCBuiltin<"__nvvm_ex2_approx_ftz_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty], [IntrNoMem]>;
  def int_nvvm_ex2_approx_f : GCCBuiltin<"__nvvm_ex2_approx_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty], [IntrNoMem]>;
  def int_nvvm_ex2_approx_d : GCCBuiltin<"__nvvm_ex2_approx_d">,
      Intrinsic<[llvm_double_ty], [llvm_double_ty], [IntrNoMem]>;

  def int_nvvm_lg2_approx_ftz_f : GCCBuiltin<"__nvvm_lg2_approx_ftz_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty], [IntrNoMem]>;
  def int_nvvm_lg2_approx_f : GCCBuiltin<"__nvvm_lg2_approx_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty], [IntrNoMem]>;
  def int_nvvm_lg2_approx_d : GCCBuiltin<"__nvvm_lg2_approx_d">,
      Intrinsic<[llvm_double_ty], [llvm_double_ty], [IntrNoMem]>;

//
// Sin  Cos
//

  def int_nvvm_sin_approx_ftz_f : GCCBuiltin<"__nvvm_sin_approx_ftz_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty], [IntrNoMem]>;
  def int_nvvm_sin_approx_f : GCCBuiltin<"__nvvm_sin_approx_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty], [IntrNoMem]>;

  def int_nvvm_cos_approx_ftz_f : GCCBuiltin<"__nvvm_cos_approx_ftz_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty], [IntrNoMem]>;
  def int_nvvm_cos_approx_f : GCCBuiltin<"__nvvm_cos_approx_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty], [IntrNoMem]>;

//
// Fma
//

  def int_nvvm_fma_rn_ftz_f : GCCBuiltin<"__nvvm_fma_rn_ftz_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty, llvm_float_ty],
        [IntrNoMem, Commutative]>;
  def int_nvvm_fma_rn_f : GCCBuiltin<"__nvvm_fma_rn_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty, llvm_float_ty],
        [IntrNoMem, Commutative]>;
  def int_nvvm_fma_rz_ftz_f : GCCBuiltin<"__nvvm_fma_rz_ftz_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty, llvm_float_ty],
        [IntrNoMem, Commutative]>;
  def int_nvvm_fma_rz_f : GCCBuiltin<"__nvvm_fma_rz_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty, llvm_float_ty],
        [IntrNoMem, Commutative]>;
  def int_nvvm_fma_rm_ftz_f : GCCBuiltin<"__nvvm_fma_rm_ftz_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty, llvm_float_ty],
        [IntrNoMem, Commutative]>;
  def int_nvvm_fma_rm_f : GCCBuiltin<"__nvvm_fma_rm_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty, llvm_float_ty],
        [IntrNoMem, Commutative]>;
  def int_nvvm_fma_rp_ftz_f : GCCBuiltin<"__nvvm_fma_rp_ftz_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty, llvm_float_ty],
        [IntrNoMem, Commutative]>;
  def int_nvvm_fma_rp_f : GCCBuiltin<"__nvvm_fma_rp_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty, llvm_float_ty],
        [IntrNoMem, Commutative]>;

  def int_nvvm_fma_rn_d : GCCBuiltin<"__nvvm_fma_rn_d">,
      Intrinsic<[llvm_double_ty],
        [llvm_double_ty, llvm_double_ty, llvm_double_ty],
        [IntrNoMem, Commutative]>;
  def int_nvvm_fma_rz_d : GCCBuiltin<"__nvvm_fma_rz_d">,
      Intrinsic<[llvm_double_ty],
        [llvm_double_ty, llvm_double_ty, llvm_double_ty],
        [IntrNoMem, Commutative]>;
  def int_nvvm_fma_rm_d : GCCBuiltin<"__nvvm_fma_rm_d">,
      Intrinsic<[llvm_double_ty],
        [llvm_double_ty, llvm_double_ty, llvm_double_ty],
        [IntrNoMem, Commutative]>;
  def int_nvvm_fma_rp_d : GCCBuiltin<"__nvvm_fma_rp_d">,
      Intrinsic<[llvm_double_ty],
        [llvm_double_ty, llvm_double_ty, llvm_double_ty],
        [IntrNoMem, Commutative]>;

//
// Rcp
//

  def int_nvvm_rcp_rn_ftz_f : GCCBuiltin<"__nvvm_rcp_rn_ftz_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty], [IntrNoMem]>;
  def int_nvvm_rcp_rn_f : GCCBuiltin<"__nvvm_rcp_rn_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty], [IntrNoMem]>;
  def int_nvvm_rcp_rz_ftz_f : GCCBuiltin<"__nvvm_rcp_rz_ftz_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty], [IntrNoMem]>;
  def int_nvvm_rcp_rz_f : GCCBuiltin<"__nvvm_rcp_rz_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty], [IntrNoMem]>;
  def int_nvvm_rcp_rm_ftz_f : GCCBuiltin<"__nvvm_rcp_rm_ftz_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty], [IntrNoMem]>;
  def int_nvvm_rcp_rm_f : GCCBuiltin<"__nvvm_rcp_rm_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty], [IntrNoMem]>;
  def int_nvvm_rcp_rp_ftz_f : GCCBuiltin<"__nvvm_rcp_rp_ftz_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty], [IntrNoMem]>;
  def int_nvvm_rcp_rp_f : GCCBuiltin<"__nvvm_rcp_rp_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty], [IntrNoMem]>;

  def int_nvvm_rcp_rn_d : GCCBuiltin<"__nvvm_rcp_rn_d">,
      Intrinsic<[llvm_double_ty], [llvm_double_ty], [IntrNoMem]>;
  def int_nvvm_rcp_rz_d : GCCBuiltin<"__nvvm_rcp_rz_d">,
      Intrinsic<[llvm_double_ty], [llvm_double_ty], [IntrNoMem]>;
  def int_nvvm_rcp_rm_d : GCCBuiltin<"__nvvm_rcp_rm_d">,
      Intrinsic<[llvm_double_ty], [llvm_double_ty], [IntrNoMem]>;
  def int_nvvm_rcp_rp_d : GCCBuiltin<"__nvvm_rcp_rp_d">,
      Intrinsic<[llvm_double_ty], [llvm_double_ty], [IntrNoMem]>;

  def int_nvvm_rcp_approx_ftz_d : GCCBuiltin<"__nvvm_rcp_approx_ftz_d">,
      Intrinsic<[llvm_double_ty], [llvm_double_ty], [IntrNoMem]>;

//
// Sqrt
//

  def int_nvvm_sqrt_f : GCCBuiltin<"__nvvm_sqrt_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty], [IntrNoMem]>;
  def int_nvvm_sqrt_rn_ftz_f : GCCBuiltin<"__nvvm_sqrt_rn_ftz_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty], [IntrNoMem]>;
  def int_nvvm_sqrt_rn_f : GCCBuiltin<"__nvvm_sqrt_rn_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty], [IntrNoMem]>;
  def int_nvvm_sqrt_rz_ftz_f : GCCBuiltin<"__nvvm_sqrt_rz_ftz_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty], [IntrNoMem]>;
  def int_nvvm_sqrt_rz_f : GCCBuiltin<"__nvvm_sqrt_rz_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty], [IntrNoMem]>;
  def int_nvvm_sqrt_rm_ftz_f : GCCBuiltin<"__nvvm_sqrt_rm_ftz_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty], [IntrNoMem]>;
  def int_nvvm_sqrt_rm_f : GCCBuiltin<"__nvvm_sqrt_rm_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty], [IntrNoMem]>;
  def int_nvvm_sqrt_rp_ftz_f : GCCBuiltin<"__nvvm_sqrt_rp_ftz_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty], [IntrNoMem]>;
  def int_nvvm_sqrt_rp_f : GCCBuiltin<"__nvvm_sqrt_rp_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty], [IntrNoMem]>;
  def int_nvvm_sqrt_approx_ftz_f : GCCBuiltin<"__nvvm_sqrt_approx_ftz_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty], [IntrNoMem]>;
  def int_nvvm_sqrt_approx_f : GCCBuiltin<"__nvvm_sqrt_approx_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty], [IntrNoMem]>;

  def int_nvvm_sqrt_rn_d : GCCBuiltin<"__nvvm_sqrt_rn_d">,
      Intrinsic<[llvm_double_ty], [llvm_double_ty], [IntrNoMem]>;
  def int_nvvm_sqrt_rz_d : GCCBuiltin<"__nvvm_sqrt_rz_d">,
      Intrinsic<[llvm_double_ty], [llvm_double_ty], [IntrNoMem]>;
  def int_nvvm_sqrt_rm_d : GCCBuiltin<"__nvvm_sqrt_rm_d">,
      Intrinsic<[llvm_double_ty], [llvm_double_ty], [IntrNoMem]>;
  def int_nvvm_sqrt_rp_d : GCCBuiltin<"__nvvm_sqrt_rp_d">,
      Intrinsic<[llvm_double_ty], [llvm_double_ty], [IntrNoMem]>;

//
// Rsqrt
//

  def int_nvvm_rsqrt_approx_ftz_f : GCCBuiltin<"__nvvm_rsqrt_approx_ftz_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty], [IntrNoMem]>;
  def int_nvvm_rsqrt_approx_f : GCCBuiltin<"__nvvm_rsqrt_approx_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty], [IntrNoMem]>;
  def int_nvvm_rsqrt_approx_d : GCCBuiltin<"__nvvm_rsqrt_approx_d">,
      Intrinsic<[llvm_double_ty], [llvm_double_ty], [IntrNoMem]>;

//
// Add
//

  def int_nvvm_add_rn_ftz_f : GCCBuiltin<"__nvvm_add_rn_ftz_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty],
        [IntrNoMem, Commutative]>;
  def int_nvvm_add_rn_f : GCCBuiltin<"__nvvm_add_rn_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty],
        [IntrNoMem, Commutative]>;
  def int_nvvm_add_rz_ftz_f : GCCBuiltin<"__nvvm_add_rz_ftz_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty],
        [IntrNoMem, Commutative]>;
  def int_nvvm_add_rz_f : GCCBuiltin<"__nvvm_add_rz_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty],
        [IntrNoMem, Commutative]>;
  def int_nvvm_add_rm_ftz_f : GCCBuiltin<"__nvvm_add_rm_ftz_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty],
        [IntrNoMem, Commutative]>;
  def int_nvvm_add_rm_f : GCCBuiltin<"__nvvm_add_rm_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty],
        [IntrNoMem, Commutative]>;
  def int_nvvm_add_rp_ftz_f : GCCBuiltin<"__nvvm_add_rp_ftz_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty],
        [IntrNoMem, Commutative]>;
  def int_nvvm_add_rp_f : GCCBuiltin<"__nvvm_add_rp_f">,
      Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty],
        [IntrNoMem, Commutative]>;

  def int_nvvm_add_rn_d : GCCBuiltin<"__nvvm_add_rn_d">,
      Intrinsic<[llvm_double_ty], [llvm_double_ty, llvm_double_ty],
        [IntrNoMem, Commutative]>;
  def int_nvvm_add_rz_d : GCCBuiltin<"__nvvm_add_rz_d">,
      Intrinsic<[llvm_double_ty], [llvm_double_ty, llvm_double_ty],
        [IntrNoMem, Commutative]>;
  def int_nvvm_add_rm_d : GCCBuiltin<"__nvvm_add_rm_d">,
      Intrinsic<[llvm_double_ty], [llvm_double_ty, llvm_double_ty],
        [IntrNoMem, Commutative]>;
  def int_nvvm_add_rp_d : GCCBuiltin<"__nvvm_add_rp_d">,
      Intrinsic<[llvm_double_ty], [llvm_double_ty, llvm_double_ty],
        [IntrNoMem, Commutative]>;

//
// Convert
//

  def int_nvvm_d2f_rn_ftz : GCCBuiltin<"__nvvm_d2f_rn_ftz">,
      Intrinsic<[llvm_float_ty], [llvm_double_ty], [IntrNoMem]>;
  def int_nvvm_d2f_rn : GCCBuiltin<"__nvvm_d2f_rn">,
      Intrinsic<[llvm_float_ty], [llvm_double_ty], [IntrNoMem]>;
  def int_nvvm_d2f_rz_ftz : GCCBuiltin<"__nvvm_d2f_rz_ftz">,
      Intrinsic<[llvm_float_ty], [llvm_double_ty], [IntrNoMem]>;
  def int_nvvm_d2f_rz : GCCBuiltin<"__nvvm_d2f_rz">,
      Intrinsic<[llvm_float_ty], [llvm_double_ty], [IntrNoMem]>;
  def int_nvvm_d2f_rm_ftz : GCCBuiltin<"__nvvm_d2f_rm_ftz">,
      Intrinsic<[llvm_float_ty], [llvm_double_ty], [IntrNoMem]>;
  def int_nvvm_d2f_rm : GCCBuiltin<"__nvvm_d2f_rm">,
      Intrinsic<[llvm_float_ty], [llvm_double_ty], [IntrNoMem]>;
  def int_nvvm_d2f_rp_ftz : GCCBuiltin<"__nvvm_d2f_rp_ftz">,
      Intrinsic<[llvm_float_ty], [llvm_double_ty], [IntrNoMem]>;
  def int_nvvm_d2f_rp : GCCBuiltin<"__nvvm_d2f_rp">,
      Intrinsic<[llvm_float_ty], [llvm_double_ty], [IntrNoMem]>;

  def int_nvvm_d2i_rn : GCCBuiltin<"__nvvm_d2i_rn">,
      Intrinsic<[llvm_i32_ty], [llvm_double_ty], [IntrNoMem]>;
  def int_nvvm_d2i_rz : GCCBuiltin<"__nvvm_d2i_rz">,
      Intrinsic<[llvm_i32_ty], [llvm_double_ty], [IntrNoMem]>;
  def int_nvvm_d2i_rm : GCCBuiltin<"__nvvm_d2i_rm">,
      Intrinsic<[llvm_i32_ty], [llvm_double_ty], [IntrNoMem]>;
  def int_nvvm_d2i_rp : GCCBuiltin<"__nvvm_d2i_rp">,
      Intrinsic<[llvm_i32_ty], [llvm_double_ty], [IntrNoMem]>;

  def int_nvvm_d2ui_rn : GCCBuiltin<"__nvvm_d2ui_rn">,
      Intrinsic<[llvm_i32_ty], [llvm_double_ty], [IntrNoMem]>;
  def int_nvvm_d2ui_rz : GCCBuiltin<"__nvvm_d2ui_rz">,
      Intrinsic<[llvm_i32_ty], [llvm_double_ty], [IntrNoMem]>;
  def int_nvvm_d2ui_rm : GCCBuiltin<"__nvvm_d2ui_rm">,
      Intrinsic<[llvm_i32_ty], [llvm_double_ty], [IntrNoMem]>;
  def int_nvvm_d2ui_rp : GCCBuiltin<"__nvvm_d2ui_rp">,
      Intrinsic<[llvm_i32_ty], [llvm_double_ty], [IntrNoMem]>;

  def int_nvvm_i2d_rn : GCCBuiltin<"__nvvm_i2d_rn">,
      Intrinsic<[llvm_double_ty], [llvm_i32_ty], [IntrNoMem]>;
  def int_nvvm_i2d_rz : GCCBuiltin<"__nvvm_i2d_rz">,
      Intrinsic<[llvm_double_ty], [llvm_i32_ty], [IntrNoMem]>;
  def int_nvvm_i2d_rm : GCCBuiltin<"__nvvm_i2d_rm">,
      Intrinsic<[llvm_double_ty], [llvm_i32_ty], [IntrNoMem]>;
  def int_nvvm_i2d_rp : GCCBuiltin<"__nvvm_i2d_rp">,
      Intrinsic<[llvm_double_ty], [llvm_i32_ty], [IntrNoMem]>;

  def int_nvvm_ui2d_rn : GCCBuiltin<"__nvvm_ui2d_rn">,
      Intrinsic<[llvm_double_ty], [llvm_i32_ty], [IntrNoMem]>;
  def int_nvvm_ui2d_rz : GCCBuiltin<"__nvvm_ui2d_rz">,
      Intrinsic<[llvm_double_ty], [llvm_i32_ty], [IntrNoMem]>;
  def int_nvvm_ui2d_rm : GCCBuiltin<"__nvvm_ui2d_rm">,
      Intrinsic<[llvm_double_ty], [llvm_i32_ty], [IntrNoMem]>;
  def int_nvvm_ui2d_rp : GCCBuiltin<"__nvvm_ui2d_rp">,
      Intrinsic<[llvm_double_ty], [llvm_i32_ty], [IntrNoMem]>;

  def int_nvvm_f2i_rn_ftz : GCCBuiltin<"__nvvm_f2i_rn_ftz">,
      Intrinsic<[llvm_i32_ty], [llvm_float_ty], [IntrNoMem]>;
  def int_nvvm_f2i_rn : GCCBuiltin<"__nvvm_f2i_rn">,
      Intrinsic<[llvm_i32_ty], [llvm_float_ty], [IntrNoMem]>;
  def int_nvvm_f2i_rz_ftz : GCCBuiltin<"__nvvm_f2i_rz_ftz">,
      Intrinsic<[llvm_i32_ty], [llvm_float_ty], [IntrNoMem]>;
  def int_nvvm_f2i_rz : GCCBuiltin<"__nvvm_f2i_rz">,
      Intrinsic<[llvm_i32_ty], [llvm_float_ty], [IntrNoMem]>;
  def int_nvvm_f2i_rm_ftz : GCCBuiltin<"__nvvm_f2i_rm_ftz">,
      Intrinsic<[llvm_i32_ty], [llvm_float_ty], [IntrNoMem]>;
  def int_nvvm_f2i_rm : GCCBuiltin<"__nvvm_f2i_rm">,
      Intrinsic<[llvm_i32_ty], [llvm_float_ty], [IntrNoMem]>;
  def int_nvvm_f2i_rp_ftz : GCCBuiltin<"__nvvm_f2i_rp_ftz">,
      Intrinsic<[llvm_i32_ty], [llvm_float_ty], [IntrNoMem]>;
  def int_nvvm_f2i_rp : GCCBuiltin<"__nvvm_f2i_rp">,
      Intrinsic<[llvm_i32_ty], [llvm_float_ty], [IntrNoMem]>;

  def int_nvvm_f2ui_rn_ftz : GCCBuiltin<"__nvvm_f2ui_rn_ftz">,
      Intrinsic<[llvm_i32_ty], [llvm_float_ty], [IntrNoMem]>;
  def int_nvvm_f2ui_rn : GCCBuiltin<"__nvvm_f2ui_rn">,
      Intrinsic<[llvm_i32_ty], [llvm_float_ty], [IntrNoMem]>;
  def int_nvvm_f2ui_rz_ftz : GCCBuiltin<"__nvvm_f2ui_rz_ftz">,
      Intrinsic<[llvm_i32_ty], [llvm_float_ty], [IntrNoMem]>;
  def int_nvvm_f2ui_rz : GCCBuiltin<"__nvvm_f2ui_rz">,
      Intrinsic<[llvm_i32_ty], [llvm_float_ty], [IntrNoMem]>;
  def int_nvvm_f2ui_rm_ftz : GCCBuiltin<"__nvvm_f2ui_rm_ftz">,
      Intrinsic<[llvm_i32_ty], [llvm_float_ty], [IntrNoMem]>;
  def int_nvvm_f2ui_rm : GCCBuiltin<"__nvvm_f2ui_rm">,
      Intrinsic<[llvm_i32_ty], [llvm_float_ty], [IntrNoMem]>;
  def int_nvvm_f2ui_rp_ftz : GCCBuiltin<"__nvvm_f2ui_rp_ftz">,
      Intrinsic<[llvm_i32_ty], [llvm_float_ty], [IntrNoMem]>;
  def int_nvvm_f2ui_rp : GCCBuiltin<"__nvvm_f2ui_rp">,
      Intrinsic<[llvm_i32_ty], [llvm_float_ty], [IntrNoMem]>;

  def int_nvvm_i2f_rn : GCCBuiltin<"__nvvm_i2f_rn">,
      Intrinsic<[llvm_float_ty], [llvm_i32_ty], [IntrNoMem]>;
  def int_nvvm_i2f_rz : GCCBuiltin<"__nvvm_i2f_rz">,
      Intrinsic<[llvm_float_ty], [llvm_i32_ty], [IntrNoMem]>;
  def int_nvvm_i2f_rm : GCCBuiltin<"__nvvm_i2f_rm">,
      Intrinsic<[llvm_float_ty], [llvm_i32_ty], [IntrNoMem]>;
  def int_nvvm_i2f_rp : GCCBuiltin<"__nvvm_i2f_rp">,
      Intrinsic<[llvm_float_ty], [llvm_i32_ty], [IntrNoMem]>;

  def int_nvvm_ui2f_rn : GCCBuiltin<"__nvvm_ui2f_rn">,
      Intrinsic<[llvm_float_ty], [llvm_i32_ty], [IntrNoMem]>;
  def int_nvvm_ui2f_rz : GCCBuiltin<"__nvvm_ui2f_rz">,
      Intrinsic<[llvm_float_ty], [llvm_i32_ty], [IntrNoMem]>;
  def int_nvvm_ui2f_rm : GCCBuiltin<"__nvvm_ui2f_rm">,
      Intrinsic<[llvm_float_ty], [llvm_i32_ty], [IntrNoMem]>;
  def int_nvvm_ui2f_rp : GCCBuiltin<"__nvvm_ui2f_rp">,
      Intrinsic<[llvm_float_ty], [llvm_i32_ty], [IntrNoMem]>;

  def int_nvvm_lohi_i2d : GCCBuiltin<"__nvvm_lohi_i2d">,
      Intrinsic<[llvm_double_ty], [llvm_i32_ty, llvm_i32_ty],
        [IntrNoMem, Commutative]>;

  def int_nvvm_d2i_lo : GCCBuiltin<"__nvvm_d2i_lo">,
      Intrinsic<[llvm_i32_ty], [llvm_double_ty], [IntrNoMem]>;
  def int_nvvm_d2i_hi : GCCBuiltin<"__nvvm_d2i_hi">,
      Intrinsic<[llvm_i32_ty], [llvm_double_ty], [IntrNoMem]>;

  def int_nvvm_f2ll_rn_ftz : GCCBuiltin<"__nvvm_f2ll_rn_ftz">,
      Intrinsic<[llvm_i64_ty], [llvm_float_ty], [IntrNoMem]>;
  def int_nvvm_f2ll_rn : GCCBuiltin<"__nvvm_f2ll_rn">,
      Intrinsic<[llvm_i64_ty], [llvm_float_ty], [IntrNoMem]>;
  def int_nvvm_f2ll_rz_ftz : GCCBuiltin<"__nvvm_f2ll_rz_ftz">,
      Intrinsic<[llvm_i64_ty], [llvm_float_ty], [IntrNoMem]>;
  def int_nvvm_f2ll_rz : GCCBuiltin<"__nvvm_f2ll_rz">,
      Intrinsic<[llvm_i64_ty], [llvm_float_ty], [IntrNoMem]>;
  def int_nvvm_f2ll_rm_ftz : GCCBuiltin<"__nvvm_f2ll_rm_ftz">,
      Intrinsic<[llvm_i64_ty], [llvm_float_ty], [IntrNoMem]>;
  def int_nvvm_f2ll_rm : GCCBuiltin<"__nvvm_f2ll_rm">,
      Intrinsic<[llvm_i64_ty], [llvm_float_ty], [IntrNoMem]>;
  def int_nvvm_f2ll_rp_ftz : GCCBuiltin<"__nvvm_f2ll_rp_ftz">,
      Intrinsic<[llvm_i64_ty], [llvm_float_ty], [IntrNoMem]>;
  def int_nvvm_f2ll_rp : GCCBuiltin<"__nvvm_f2ll_rp">,
      Intrinsic<[llvm_i64_ty], [llvm_float_ty], [IntrNoMem]>;

  def int_nvvm_f2ull_rn_ftz : GCCBuiltin<"__nvvm_f2ull_rn_ftz">,
      Intrinsic<[llvm_i64_ty], [llvm_float_ty], [IntrNoMem]>;
  def int_nvvm_f2ull_rn : GCCBuiltin<"__nvvm_f2ull_rn">,
      Intrinsic<[llvm_i64_ty], [llvm_float_ty], [IntrNoMem]>;
  def int_nvvm_f2ull_rz_ftz : GCCBuiltin<"__nvvm_f2ull_rz_ftz">,
      Intrinsic<[llvm_i64_ty], [llvm_float_ty], [IntrNoMem]>;
  def int_nvvm_f2ull_rz : GCCBuiltin<"__nvvm_f2ull_rz">,
      Intrinsic<[llvm_i64_ty], [llvm_float_ty], [IntrNoMem]>;
  def int_nvvm_f2ull_rm_ftz : GCCBuiltin<"__nvvm_f2ull_rm_ftz">,
      Intrinsic<[llvm_i64_ty], [llvm_float_ty], [IntrNoMem]>;
  def int_nvvm_f2ull_rm : GCCBuiltin<"__nvvm_f2ull_rm">,
      Intrinsic<[llvm_i64_ty], [llvm_float_ty], [IntrNoMem]>;
  def int_nvvm_f2ull_rp_ftz : GCCBuiltin<"__nvvm_f2ull_rp_ftz">,
      Intrinsic<[llvm_i64_ty], [llvm_float_ty], [IntrNoMem]>;
  def int_nvvm_f2ull_rp : GCCBuiltin<"__nvvm_f2ull_rp">,
      Intrinsic<[llvm_i64_ty], [llvm_float_ty], [IntrNoMem]>;

  def int_nvvm_d2ll_rn : GCCBuiltin<"__nvvm_d2ll_rn">,
      Intrinsic<[llvm_i64_ty], [llvm_double_ty], [IntrNoMem]>;
  def int_nvvm_d2ll_rz : GCCBuiltin<"__nvvm_d2ll_rz">,
      Intrinsic<[llvm_i64_ty], [llvm_double_ty], [IntrNoMem]>;
  def int_nvvm_d2ll_rm : GCCBuiltin<"__nvvm_d2ll_rm">,
      Intrinsic<[llvm_i64_ty], [llvm_double_ty], [IntrNoMem]>;
  def int_nvvm_d2ll_rp : GCCBuiltin<"__nvvm_d2ll_rp">,
      Intrinsic<[llvm_i64_ty], [llvm_double_ty], [IntrNoMem]>;

  def int_nvvm_d2ull_rn : GCCBuiltin<"__nvvm_d2ull_rn">,
      Intrinsic<[llvm_i64_ty], [llvm_double_ty], [IntrNoMem]>;
  def int_nvvm_d2ull_rz : GCCBuiltin<"__nvvm_d2ull_rz">,
      Intrinsic<[llvm_i64_ty], [llvm_double_ty], [IntrNoMem]>;
  def int_nvvm_d2ull_rm : GCCBuiltin<"__nvvm_d2ull_rm">,
      Intrinsic<[llvm_i64_ty], [llvm_double_ty], [IntrNoMem]>;
  def int_nvvm_d2ull_rp : GCCBuiltin<"__nvvm_d2ull_rp">,
      Intrinsic<[llvm_i64_ty], [llvm_double_ty], [IntrNoMem]>;

  def int_nvvm_ll2f_rn : GCCBuiltin<"__nvvm_ll2f_rn">,
      Intrinsic<[llvm_float_ty], [llvm_i64_ty], [IntrNoMem]>;
  def int_nvvm_ll2f_rz : GCCBuiltin<"__nvvm_ll2f_rz">,
      Intrinsic<[llvm_float_ty], [llvm_i64_ty], [IntrNoMem]>;
  def int_nvvm_ll2f_rm : GCCBuiltin<"__nvvm_ll2f_rm">,
      Intrinsic<[llvm_float_ty], [llvm_i64_ty], [IntrNoMem]>;
  def int_nvvm_ll2f_rp : GCCBuiltin<"__nvvm_ll2f_rp">,
      Intrinsic<[llvm_float_ty], [llvm_i64_ty], [IntrNoMem]>;
  def int_nvvm_ull2f_rn : GCCBuiltin<"__nvvm_ull2f_rn">,
      Intrinsic<[llvm_float_ty], [llvm_i64_ty], [IntrNoMem]>;
  def int_nvvm_ull2f_rz : GCCBuiltin<"__nvvm_ull2f_rz">,
      Intrinsic<[llvm_float_ty], [llvm_i64_ty], [IntrNoMem]>;
  def int_nvvm_ull2f_rm : GCCBuiltin<"__nvvm_ull2f_rm">,
      Intrinsic<[llvm_float_ty], [llvm_i64_ty], [IntrNoMem]>;
  def int_nvvm_ull2f_rp : GCCBuiltin<"__nvvm_ull2f_rp">,
      Intrinsic<[llvm_float_ty], [llvm_i64_ty], [IntrNoMem]>;

  def int_nvvm_ll2d_rn : GCCBuiltin<"__nvvm_ll2d_rn">,
      Intrinsic<[llvm_double_ty], [llvm_i64_ty], [IntrNoMem]>;
  def int_nvvm_ll2d_rz : GCCBuiltin<"__nvvm_ll2d_rz">,
      Intrinsic<[llvm_double_ty], [llvm_i64_ty], [IntrNoMem]>;
  def int_nvvm_ll2d_rm : GCCBuiltin<"__nvvm_ll2d_rm">,
      Intrinsic<[llvm_double_ty], [llvm_i64_ty], [IntrNoMem]>;
  def int_nvvm_ll2d_rp : GCCBuiltin<"__nvvm_ll2d_rp">,
      Intrinsic<[llvm_double_ty], [llvm_i64_ty], [IntrNoMem]>;
  def int_nvvm_ull2d_rn : GCCBuiltin<"__nvvm_ull2d_rn">,
      Intrinsic<[llvm_double_ty], [llvm_i64_ty], [IntrNoMem]>;
  def int_nvvm_ull2d_rz : GCCBuiltin<"__nvvm_ull2d_rz">,
      Intrinsic<[llvm_double_ty], [llvm_i64_ty], [IntrNoMem]>;
  def int_nvvm_ull2d_rm : GCCBuiltin<"__nvvm_ull2d_rm">,
      Intrinsic<[llvm_double_ty], [llvm_i64_ty], [IntrNoMem]>;
  def int_nvvm_ull2d_rp : GCCBuiltin<"__nvvm_ull2d_rp">,
      Intrinsic<[llvm_double_ty], [llvm_i64_ty], [IntrNoMem]>;

  def int_nvvm_f2h_rn_ftz : GCCBuiltin<"__nvvm_f2h_rn_ftz">,
      Intrinsic<[llvm_i16_ty], [llvm_float_ty], [IntrNoMem]>;
  def int_nvvm_f2h_rn : GCCBuiltin<"__nvvm_f2h_rn">,
      Intrinsic<[llvm_i16_ty], [llvm_float_ty], [IntrNoMem]>;

  def int_nvvm_h2f : GCCBuiltin<"__nvvm_h2f">,
      Intrinsic<[llvm_float_ty], [llvm_i16_ty], [IntrNoMem]>;

//
// Bitcast
//

  def int_nvvm_bitcast_f2i : GCCBuiltin<"__nvvm_bitcast_f2i">,
      Intrinsic<[llvm_i32_ty], [llvm_float_ty], [IntrNoMem]>;
  def int_nvvm_bitcast_i2f : GCCBuiltin<"__nvvm_bitcast_i2f">,
      Intrinsic<[llvm_float_ty], [llvm_i32_ty], [IntrNoMem]>;

  def int_nvvm_bitcast_ll2d : GCCBuiltin<"__nvvm_bitcast_ll2d">,
      Intrinsic<[llvm_double_ty], [llvm_i64_ty], [IntrNoMem]>;
  def int_nvvm_bitcast_d2ll : GCCBuiltin<"__nvvm_bitcast_d2ll">,
      Intrinsic<[llvm_i64_ty], [llvm_double_ty], [IntrNoMem]>;


// Atomic not available as an llvm intrinsic.
  def int_nvvm_atomic_load_add_f32 : Intrinsic<[llvm_float_ty],
          [LLVMAnyPointerType<llvm_float_ty>, llvm_float_ty],
                                      [IntrReadWriteArgMem, NoCapture<0>]>;
  def int_nvvm_atomic_load_inc_32 : Intrinsic<[llvm_i32_ty],
          [LLVMAnyPointerType<llvm_i32_ty>, llvm_i32_ty],
                                      [IntrReadWriteArgMem, NoCapture<0>]>;
  def int_nvvm_atomic_load_dec_32 : Intrinsic<[llvm_i32_ty],
          [LLVMAnyPointerType<llvm_i32_ty>, llvm_i32_ty],
                                      [IntrReadWriteArgMem, NoCapture<0>]>;

// Bar.Sync
  def int_cuda_syncthreads : GCCBuiltin<"__syncthreads">,
      Intrinsic<[], [], [IntrNoDuplicate]>;
  def int_nvvm_barrier0 : GCCBuiltin<"__nvvm_bar0">,
      Intrinsic<[], [], [IntrNoDuplicate]>;
  def int_nvvm_barrier0_popc : GCCBuiltin<"__nvvm_bar0_popc">,
      Intrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrNoDuplicate]>;
  def int_nvvm_barrier0_and : GCCBuiltin<"__nvvm_bar0_and">,
      Intrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrNoDuplicate]>;
  def int_nvvm_barrier0_or : GCCBuiltin<"__nvvm_bar0_or">,
      Intrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrNoDuplicate]>;

  // Membar
  def int_nvvm_membar_cta : GCCBuiltin<"__nvvm_membar_cta">,
      Intrinsic<[], [], []>;
  def int_nvvm_membar_gl : GCCBuiltin<"__nvvm_membar_gl">,
      Intrinsic<[], [], []>;
  def int_nvvm_membar_sys : GCCBuiltin<"__nvvm_membar_sys">,
      Intrinsic<[], [], []>;


// Accessing special registers
  def int_nvvm_read_ptx_sreg_tid_x :
      Intrinsic<[llvm_i32_ty], [], [IntrNoMem]>,
      GCCBuiltin<"__nvvm_read_ptx_sreg_tid_x">;
  def int_nvvm_read_ptx_sreg_tid_y :
      Intrinsic<[llvm_i32_ty], [], [IntrNoMem]>,
      GCCBuiltin<"__nvvm_read_ptx_sreg_tid_y">;
  def int_nvvm_read_ptx_sreg_tid_z :
      Intrinsic<[llvm_i32_ty], [], [IntrNoMem]>,
      GCCBuiltin<"__nvvm_read_ptx_sreg_tid_z">;

  def int_nvvm_read_ptx_sreg_ntid_x :
      Intrinsic<[llvm_i32_ty], [], [IntrNoMem]>,
      GCCBuiltin<"__nvvm_read_ptx_sreg_ntid_x">;
  def int_nvvm_read_ptx_sreg_ntid_y :
      Intrinsic<[llvm_i32_ty], [], [IntrNoMem]>,
      GCCBuiltin<"__nvvm_read_ptx_sreg_ntid_y">;
  def int_nvvm_read_ptx_sreg_ntid_z :
      Intrinsic<[llvm_i32_ty], [], [IntrNoMem]>,
      GCCBuiltin<"__nvvm_read_ptx_sreg_ntid_z">;

  def int_nvvm_read_ptx_sreg_ctaid_x :
      Intrinsic<[llvm_i32_ty], [], [IntrNoMem]>,
      GCCBuiltin<"__nvvm_read_ptx_sreg_ctaid_x">;
  def int_nvvm_read_ptx_sreg_ctaid_y :
      Intrinsic<[llvm_i32_ty], [], [IntrNoMem]>,
      GCCBuiltin<"__nvvm_read_ptx_sreg_ctaid_y">;
  def int_nvvm_read_ptx_sreg_ctaid_z :
      Intrinsic<[llvm_i32_ty], [], [IntrNoMem]>,
      GCCBuiltin<"__nvvm_read_ptx_sreg_ctaid_z">;

  def int_nvvm_read_ptx_sreg_nctaid_x :
      Intrinsic<[llvm_i32_ty], [], [IntrNoMem]>,
      GCCBuiltin<"__nvvm_read_ptx_sreg_nctaid_x">;
  def int_nvvm_read_ptx_sreg_nctaid_y :
      Intrinsic<[llvm_i32_ty], [], [IntrNoMem]>,
      GCCBuiltin<"__nvvm_read_ptx_sreg_nctaid_y">;
  def int_nvvm_read_ptx_sreg_nctaid_z :
      Intrinsic<[llvm_i32_ty], [], [IntrNoMem]>,
      GCCBuiltin<"__nvvm_read_ptx_sreg_nctaid_z">;

  def int_nvvm_read_ptx_sreg_warpsize :
      Intrinsic<[llvm_i32_ty], [], [IntrNoMem]>,
      GCCBuiltin<"__nvvm_read_ptx_sreg_warpsize">;


// Generated within nvvm. Use for ldu on sm_20 or later
def int_nvvm_ldu_global_i : Intrinsic<[llvm_anyint_ty],
  [LLVMAnyPointerType<LLVMMatchType<0>>, llvm_i32_ty],
  [IntrReadMem, NoCapture<0>],
  "llvm.nvvm.ldu.global.i">;
def int_nvvm_ldu_global_f : Intrinsic<[llvm_anyfloat_ty],
  [LLVMAnyPointerType<LLVMMatchType<0>>, llvm_i32_ty],
  [IntrReadMem, NoCapture<0>],
  "llvm.nvvm.ldu.global.f">;
def int_nvvm_ldu_global_p : Intrinsic<[llvm_anyptr_ty],
  [LLVMAnyPointerType<LLVMMatchType<0>>, llvm_i32_ty],
  [IntrReadMem, NoCapture<0>],
  "llvm.nvvm.ldu.global.p">;

// Generated within nvvm. Use for ldg on sm_35 or later
def int_nvvm_ldg_global_i : Intrinsic<[llvm_anyint_ty],
  [LLVMAnyPointerType<LLVMMatchType<0>>, llvm_i32_ty],
  [IntrReadMem, NoCapture<0>],
  "llvm.nvvm.ldg.global.i">;
def int_nvvm_ldg_global_f : Intrinsic<[llvm_anyfloat_ty],
  [LLVMAnyPointerType<LLVMMatchType<0>>, llvm_i32_ty],
  [IntrReadMem, NoCapture<0>],
  "llvm.nvvm.ldg.global.f">;
def int_nvvm_ldg_global_p : Intrinsic<[llvm_anyptr_ty],
  [LLVMAnyPointerType<LLVMMatchType<0>>, llvm_i32_ty],
  [IntrReadMem, NoCapture<0>],
  "llvm.nvvm.ldg.global.p">;

// Use for generic pointers
// - These intrinsics are used to convert address spaces.
// - The input pointer and output pointer must have the same type, except for
//   the address-space. (This restriction is not enforced here as there is
//   currently no way to describe it).
// - This complements the llvm bitcast, which can be used to cast one type
//   of pointer to another type of pointer, while the address space remains
//   the same.
def int_nvvm_ptr_local_to_gen: Intrinsic<[llvm_anyptr_ty],
                 [llvm_anyptr_ty], [IntrNoMem],
                 "llvm.nvvm.ptr.local.to.gen">;
def int_nvvm_ptr_shared_to_gen: Intrinsic<[llvm_anyptr_ty],
                 [llvm_anyptr_ty], [IntrNoMem],
                 "llvm.nvvm.ptr.shared.to.gen">;
def int_nvvm_ptr_global_to_gen: Intrinsic<[llvm_anyptr_ty],
                 [llvm_anyptr_ty], [IntrNoMem],
                 "llvm.nvvm.ptr.global.to.gen">;
def int_nvvm_ptr_constant_to_gen: Intrinsic<[llvm_anyptr_ty],
                 [llvm_anyptr_ty], [IntrNoMem],
                 "llvm.nvvm.ptr.constant.to.gen">;

def int_nvvm_ptr_gen_to_global: Intrinsic<[llvm_anyptr_ty],
                 [llvm_anyptr_ty], [IntrNoMem],
                 "llvm.nvvm.ptr.gen.to.global">;
def int_nvvm_ptr_gen_to_shared: Intrinsic<[llvm_anyptr_ty],
                 [llvm_anyptr_ty], [IntrNoMem],
                 "llvm.nvvm.ptr.gen.to.shared">;
def int_nvvm_ptr_gen_to_local: Intrinsic<[llvm_anyptr_ty],
                 [llvm_anyptr_ty], [IntrNoMem],
                 "llvm.nvvm.ptr.gen.to.local">;
def int_nvvm_ptr_gen_to_constant: Intrinsic<[llvm_anyptr_ty],
                 [llvm_anyptr_ty], [IntrNoMem],
                 "llvm.nvvm.ptr.gen.to.constant">;

// Used in nvvm internally to help address space opt and ptx code generation
// This is for params that are passed to kernel functions by pointer by-val.
def int_nvvm_ptr_gen_to_param: Intrinsic<[llvm_anyptr_ty],
                                     [llvm_anyptr_ty],
                                   [IntrNoMem],
                                   "llvm.nvvm.ptr.gen.to.param">;

// Move intrinsics, used in nvvm internally

def int_nvvm_move_i16 : Intrinsic<[llvm_i16_ty], [llvm_i16_ty], [IntrNoMem],
  "llvm.nvvm.move.i16">;
def int_nvvm_move_i32 : Intrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrNoMem],
  "llvm.nvvm.move.i32">;
def int_nvvm_move_i64 : Intrinsic<[llvm_i64_ty], [llvm_i64_ty], [IntrNoMem],
  "llvm.nvvm.move.i64">;
def int_nvvm_move_float : Intrinsic<[llvm_float_ty], [llvm_float_ty],
  [IntrNoMem], "llvm.nvvm.move.float">;
def int_nvvm_move_double : Intrinsic<[llvm_double_ty], [llvm_double_ty],
  [IntrNoMem], "llvm.nvvm.move.double">;
def int_nvvm_move_ptr : Intrinsic<[llvm_anyptr_ty], [llvm_anyptr_ty],
  [IntrNoMem, NoCapture<0>], "llvm.nvvm.move.ptr">;


// For getting the handle from a texture or surface variable
def int_nvvm_texsurf_handle
  : Intrinsic<[llvm_i64_ty], [llvm_metadata_ty, llvm_anyi64ptr_ty],
              [IntrNoMem], "llvm.nvvm.texsurf.handle">;
def int_nvvm_texsurf_handle_internal
  : Intrinsic<[llvm_i64_ty], [llvm_anyptr_ty],
              [IntrNoMem], "llvm.nvvm.texsurf.handle.internal">;

/// Error / Warn
def int_nvvm_compiler_error :
    Intrinsic<[], [llvm_anyptr_ty], [], "llvm.nvvm.compiler.error">;
def int_nvvm_compiler_warn :
    Intrinsic<[], [llvm_anyptr_ty], [], "llvm.nvvm.compiler.warn">;

def int_nvvm_reflect :
  Intrinsic<[llvm_i32_ty], [llvm_anyptr_ty], [IntrNoMem], "llvm.nvvm.reflect">;

// isspacep.{const, global, local, shared}
def int_nvvm_isspacep_const
  : Intrinsic<[llvm_i1_ty], [llvm_ptr_ty], [IntrNoMem],
              "llvm.nvvm.isspacep.const">,
    GCCBuiltin<"__nvvm_isspacep_const">;
def int_nvvm_isspacep_global
  : Intrinsic<[llvm_i1_ty], [llvm_ptr_ty], [IntrNoMem],
              "llvm.nvvm.isspacep.global">,
    GCCBuiltin<"__nvvm_isspacep_global">;
def int_nvvm_isspacep_local
  : Intrinsic<[llvm_i1_ty], [llvm_ptr_ty], [IntrNoMem],
              "llvm.nvvm.isspacep.local">,
    GCCBuiltin<"__nvvm_isspacep_local">;
def int_nvvm_isspacep_shared
  : Intrinsic<[llvm_i1_ty], [llvm_ptr_ty], [IntrNoMem],
              "llvm.nvvm.isspacep.shared">,
    GCCBuiltin<"__nvvm_isspacep_shared">;

// Environment register read
def int_nvvm_read_ptx_sreg_envreg0
  : Intrinsic<[llvm_i32_ty], [], [IntrNoMem],
              "llvm.nvvm.read.ptx.sreg.envreg0">,
    GCCBuiltin<"__nvvm_read_ptx_sreg_envreg0">;
def int_nvvm_read_ptx_sreg_envreg1
  : Intrinsic<[llvm_i32_ty], [], [IntrNoMem],
              "llvm.nvvm.read.ptx.sreg.envreg1">,
    GCCBuiltin<"__nvvm_read_ptx_sreg_envreg1">;
def int_nvvm_read_ptx_sreg_envreg2
  : Intrinsic<[llvm_i32_ty], [], [IntrNoMem],
              "llvm.nvvm.read.ptx.sreg.envreg2">,
    GCCBuiltin<"__nvvm_read_ptx_sreg_envreg2">;
def int_nvvm_read_ptx_sreg_envreg3
  : Intrinsic<[llvm_i32_ty], [], [IntrNoMem],
              "llvm.nvvm.read.ptx.sreg.envreg3">,
    GCCBuiltin<"__nvvm_read_ptx_sreg_envreg3">;
def int_nvvm_read_ptx_sreg_envreg4
  : Intrinsic<[llvm_i32_ty], [], [IntrNoMem],
              "llvm.nvvm.read.ptx.sreg.envreg4">,
    GCCBuiltin<"__nvvm_read_ptx_sreg_envreg4">;
def int_nvvm_read_ptx_sreg_envreg5
  : Intrinsic<[llvm_i32_ty], [], [IntrNoMem],
              "llvm.nvvm.read.ptx.sreg.envreg5">,
    GCCBuiltin<"__nvvm_read_ptx_sreg_envreg5">;
def int_nvvm_read_ptx_sreg_envreg6
  : Intrinsic<[llvm_i32_ty], [], [IntrNoMem],
              "llvm.nvvm.read.ptx.sreg.envreg6">,
    GCCBuiltin<"__nvvm_read_ptx_sreg_envreg6">;
def int_nvvm_read_ptx_sreg_envreg7
  : Intrinsic<[llvm_i32_ty], [], [IntrNoMem],
              "llvm.nvvm.read.ptx.sreg.envreg7">,
    GCCBuiltin<"__nvvm_read_ptx_sreg_envreg7">;
def int_nvvm_read_ptx_sreg_envreg8
  : Intrinsic<[llvm_i32_ty], [], [IntrNoMem],
              "llvm.nvvm.read.ptx.sreg.envreg8">,
    GCCBuiltin<"__nvvm_read_ptx_sreg_envreg8">;
def int_nvvm_read_ptx_sreg_envreg9
  : Intrinsic<[llvm_i32_ty], [], [IntrNoMem],
              "llvm.nvvm.read.ptx.sreg.envreg9">,
    GCCBuiltin<"__nvvm_read_ptx_sreg_envreg9">;
def int_nvvm_read_ptx_sreg_envreg10
  : Intrinsic<[llvm_i32_ty], [], [IntrNoMem],
              "llvm.nvvm.read.ptx.sreg.envreg10">,
    GCCBuiltin<"__nvvm_read_ptx_sreg_envreg10">;
def int_nvvm_read_ptx_sreg_envreg11
  : Intrinsic<[llvm_i32_ty], [], [IntrNoMem],
              "llvm.nvvm.read.ptx.sreg.envreg11">,
    GCCBuiltin<"__nvvm_read_ptx_sreg_envreg11">;
def int_nvvm_read_ptx_sreg_envreg12
  : Intrinsic<[llvm_i32_ty], [], [IntrNoMem],
              "llvm.nvvm.read.ptx.sreg.envreg12">,
    GCCBuiltin<"__nvvm_read_ptx_sreg_envreg12">;
def int_nvvm_read_ptx_sreg_envreg13
  : Intrinsic<[llvm_i32_ty], [], [IntrNoMem],
              "llvm.nvvm.read.ptx.sreg.envreg13">,
    GCCBuiltin<"__nvvm_read_ptx_sreg_envreg13">;
def int_nvvm_read_ptx_sreg_envreg14
  : Intrinsic<[llvm_i32_ty], [], [IntrNoMem],
              "llvm.nvvm.read.ptx.sreg.envreg14">,
    GCCBuiltin<"__nvvm_read_ptx_sreg_envreg14">;
def int_nvvm_read_ptx_sreg_envreg15
  : Intrinsic<[llvm_i32_ty], [], [IntrNoMem],
              "llvm.nvvm.read.ptx.sreg.envreg15">,
    GCCBuiltin<"__nvvm_read_ptx_sreg_envreg15">;
def int_nvvm_read_ptx_sreg_envreg16
  : Intrinsic<[llvm_i32_ty], [], [IntrNoMem],
              "llvm.nvvm.read.ptx.sreg.envreg16">,
    GCCBuiltin<"__nvvm_read_ptx_sreg_envreg16">;
def int_nvvm_read_ptx_sreg_envreg17
  : Intrinsic<[llvm_i32_ty], [], [IntrNoMem],
              "llvm.nvvm.read.ptx.sreg.envreg17">,
    GCCBuiltin<"__nvvm_read_ptx_sreg_envreg17">;
def int_nvvm_read_ptx_sreg_envreg18
  : Intrinsic<[llvm_i32_ty], [], [IntrNoMem],
              "llvm.nvvm.read.ptx.sreg.envreg18">,
    GCCBuiltin<"__nvvm_read_ptx_sreg_envreg18">;
def int_nvvm_read_ptx_sreg_envreg19
  : Intrinsic<[llvm_i32_ty], [], [IntrNoMem],
              "llvm.nvvm.read.ptx.sreg.envreg19">,
    GCCBuiltin<"__nvvm_read_ptx_sreg_envreg19">;
def int_nvvm_read_ptx_sreg_envreg20
  : Intrinsic<[llvm_i32_ty], [], [IntrNoMem],
              "llvm.nvvm.read.ptx.sreg.envreg20">,
    GCCBuiltin<"__nvvm_read_ptx_sreg_envreg20">;
def int_nvvm_read_ptx_sreg_envreg21
  : Intrinsic<[llvm_i32_ty], [], [IntrNoMem],
              "llvm.nvvm.read.ptx.sreg.envreg21">,
    GCCBuiltin<"__nvvm_read_ptx_sreg_envreg21">;
def int_nvvm_read_ptx_sreg_envreg22
  : Intrinsic<[llvm_i32_ty], [], [IntrNoMem],
              "llvm.nvvm.read.ptx.sreg.envreg22">,
    GCCBuiltin<"__nvvm_read_ptx_sreg_envreg22">;
def int_nvvm_read_ptx_sreg_envreg23
  : Intrinsic<[llvm_i32_ty], [], [IntrNoMem],
              "llvm.nvvm.read.ptx.sreg.envreg23">,
    GCCBuiltin<"__nvvm_read_ptx_sreg_envreg23">;
def int_nvvm_read_ptx_sreg_envreg24
  : Intrinsic<[llvm_i32_ty], [], [IntrNoMem],
              "llvm.nvvm.read.ptx.sreg.envreg24">,
    GCCBuiltin<"__nvvm_read_ptx_sreg_envreg24">;
def int_nvvm_read_ptx_sreg_envreg25
  : Intrinsic<[llvm_i32_ty], [], [IntrNoMem],
              "llvm.nvvm.read.ptx.sreg.envreg25">,
    GCCBuiltin<"__nvvm_read_ptx_sreg_envreg25">;
def int_nvvm_read_ptx_sreg_envreg26
  : Intrinsic<[llvm_i32_ty], [], [IntrNoMem],
              "llvm.nvvm.read.ptx.sreg.envreg26">,
    GCCBuiltin<"__nvvm_read_ptx_sreg_envreg26">;
def int_nvvm_read_ptx_sreg_envreg27
  : Intrinsic<[llvm_i32_ty], [], [IntrNoMem],
              "llvm.nvvm.read.ptx.sreg.envreg27">,
    GCCBuiltin<"__nvvm_read_ptx_sreg_envreg27">;
def int_nvvm_read_ptx_sreg_envreg28
  : Intrinsic<[llvm_i32_ty], [], [IntrNoMem],
              "llvm.nvvm.read.ptx.sreg.envreg28">,
    GCCBuiltin<"__nvvm_read_ptx_sreg_envreg28">;
def int_nvvm_read_ptx_sreg_envreg29
  : Intrinsic<[llvm_i32_ty], [], [IntrNoMem],
              "llvm.nvvm.read.ptx.sreg.envreg29">,
    GCCBuiltin<"__nvvm_read_ptx_sreg_envreg29">;
def int_nvvm_read_ptx_sreg_envreg30
  : Intrinsic<[llvm_i32_ty], [], [IntrNoMem],
              "llvm.nvvm.read.ptx.sreg.envreg30">,
    GCCBuiltin<"__nvvm_read_ptx_sreg_envreg30">;
def int_nvvm_read_ptx_sreg_envreg31
  : Intrinsic<[llvm_i32_ty], [], [IntrNoMem],
              "llvm.nvvm.read.ptx.sreg.envreg31">,
    GCCBuiltin<"__nvvm_read_ptx_sreg_envreg31">;


// Texture Fetch
// texmode_independent
def int_nvvm_tex_1d_v4f32_s32
  : Intrinsic<[llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_i32_ty], [],
              "llvm.nvvm.tex.1d.v4f32.s32">;
def int_nvvm_tex_1d_v4f32_f32
  : Intrinsic<[llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.1d.v4f32.f32">;
def int_nvvm_tex_1d_level_v4f32_f32
  : Intrinsic<[llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.1d.level.v4f32.f32">;
def int_nvvm_tex_1d_grad_v4f32_f32
  : Intrinsic<[llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_float_ty, llvm_float_ty,
               llvm_float_ty], [],
              "llvm.nvvm.tex.1d.grad.v4f32.f32">;
def int_nvvm_tex_1d_v4s32_s32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_i32_ty], [],
              "llvm.nvvm.tex.1d.v4s32.s32">;
def int_nvvm_tex_1d_v4s32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.1d.v4s32.f32">;
def int_nvvm_tex_1d_level_v4s32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.1d.level.v4s32.f32">;
def int_nvvm_tex_1d_grad_v4s32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_float_ty, llvm_float_ty,
               llvm_float_ty], [],
              "llvm.nvvm.tex.1d.grad.v4s32.f32">;
def int_nvvm_tex_1d_v4u32_s32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_i32_ty], [],
              "llvm.nvvm.tex.1d.v4u32.s32">;
def int_nvvm_tex_1d_v4u32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.1d.v4u32.f32">;
def int_nvvm_tex_1d_level_v4u32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.1d.level.v4u32.f32">;
def int_nvvm_tex_1d_grad_v4u32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_float_ty, llvm_float_ty,
               llvm_float_ty], [],
              "llvm.nvvm.tex.1d.grad.v4u32.f32">;

def int_nvvm_tex_1d_array_v4f32_s32
  : Intrinsic<[llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.tex.1d.array.v4f32.s32">;
def int_nvvm_tex_1d_array_v4f32_f32
  : Intrinsic<[llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_i32_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.1d.array.v4f32.f32">;
def int_nvvm_tex_1d_array_level_v4f32_f32
  : Intrinsic<[llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_i32_ty, llvm_float_ty,
               llvm_float_ty], [],
              "llvm.nvvm.tex.1d.array.level.v4f32.f32">;
def int_nvvm_tex_1d_array_grad_v4f32_f32
  : Intrinsic<[llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_i32_ty, llvm_float_ty,
               llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.1d.array.grad.v4f32.f32">;
def int_nvvm_tex_1d_array_v4s32_s32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.tex.1d.array.v4s32.s32">;
def int_nvvm_tex_1d_array_v4s32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_i32_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.1d.array.v4s32.f32">;
def int_nvvm_tex_1d_array_level_v4s32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_i32_ty, llvm_float_ty,
               llvm_float_ty], [],
              "llvm.nvvm.tex.1d.array.level.v4s32.f32">;
def int_nvvm_tex_1d_array_grad_v4s32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_i32_ty, llvm_float_ty,
               llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.1d.array.grad.v4s32.f32">;
def int_nvvm_tex_1d_array_v4u32_s32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.tex.1d.array.v4u32.s32">;
def int_nvvm_tex_1d_array_v4u32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_i32_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.1d.array.v4u32.f32">;
def int_nvvm_tex_1d_array_level_v4u32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_i32_ty, llvm_float_ty,
               llvm_float_ty], [],
              "llvm.nvvm.tex.1d.array.level.v4u32.f32">;
def int_nvvm_tex_1d_array_grad_v4u32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_i32_ty, llvm_float_ty,
               llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.1d.array.grad.v4u32.f32">;

def int_nvvm_tex_2d_v4f32_s32
  : Intrinsic<[llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.tex.2d.v4f32.s32">;
def int_nvvm_tex_2d_v4f32_f32
  : Intrinsic<[llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.2d.v4f32.f32">;
def int_nvvm_tex_2d_level_v4f32_f32
  : Intrinsic<[llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_float_ty, llvm_float_ty,
               llvm_float_ty], [],
              "llvm.nvvm.tex.2d.level.v4f32.f32">;
def int_nvvm_tex_2d_grad_v4f32_f32
  : Intrinsic<[llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_float_ty, llvm_float_ty,
               llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.2d.grad.v4f32.f32">;
def int_nvvm_tex_2d_v4s32_s32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.tex.2d.v4s32.s32">;
def int_nvvm_tex_2d_v4s32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.2d.v4s32.f32">;
def int_nvvm_tex_2d_level_v4s32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_float_ty, llvm_float_ty,
               llvm_float_ty], [],
              "llvm.nvvm.tex.2d.level.v4s32.f32">;
def int_nvvm_tex_2d_grad_v4s32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_float_ty, llvm_float_ty,
               llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.2d.grad.v4s32.f32">;
def int_nvvm_tex_2d_v4u32_s32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.tex.2d.v4u32.s32">;
def int_nvvm_tex_2d_v4u32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.2d.v4u32.f32">;
def int_nvvm_tex_2d_level_v4u32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_float_ty, llvm_float_ty,
               llvm_float_ty], [],
              "llvm.nvvm.tex.2d.level.v4u32.f32">;
def int_nvvm_tex_2d_grad_v4u32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_float_ty, llvm_float_ty,
               llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.2d.grad.v4u32.f32">;

def int_nvvm_tex_2d_array_v4f32_s32
  : Intrinsic<[llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
               llvm_i32_ty], [],
              "llvm.nvvm.tex.2d.array.v4f32.s32">;
def int_nvvm_tex_2d_array_v4f32_f32
  : Intrinsic<[llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_i32_ty, llvm_float_ty,
               llvm_float_ty], [],
              "llvm.nvvm.tex.2d.array.v4f32.f32">;
def int_nvvm_tex_2d_array_level_v4f32_f32
  : Intrinsic<[llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_i32_ty, llvm_float_ty,
               llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.2d.array.level.v4f32.f32">;
def int_nvvm_tex_2d_array_grad_v4f32_f32
  : Intrinsic<[llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_i32_ty, llvm_float_ty,
               llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty,
               llvm_float_ty], [],
              "llvm.nvvm.tex.2d.array.grad.v4f32.f32">;
def int_nvvm_tex_2d_array_v4s32_s32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
               llvm_i32_ty], [],
              "llvm.nvvm.tex.2d.array.v4s32.s32">;
def int_nvvm_tex_2d_array_v4s32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_i32_ty, llvm_float_ty,
               llvm_float_ty], [],
              "llvm.nvvm.tex.2d.array.v4s32.f32">;
def int_nvvm_tex_2d_array_level_v4s32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_i32_ty, llvm_float_ty,
               llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.2d.array.level.v4s32.f32">;
def int_nvvm_tex_2d_array_grad_v4s32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_i32_ty, llvm_float_ty,
               llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty,
               llvm_float_ty], [],
              "llvm.nvvm.tex.2d.array.grad.v4s32.f32">;
def int_nvvm_tex_2d_array_v4u32_s32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
               llvm_i32_ty], [],
              "llvm.nvvm.tex.2d.array.v4u32.s32">;
def int_nvvm_tex_2d_array_v4u32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_i32_ty, llvm_float_ty,
               llvm_float_ty], [],
              "llvm.nvvm.tex.2d.array.v4u32.f32">;
def int_nvvm_tex_2d_array_level_v4u32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_i32_ty, llvm_float_ty,
               llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.2d.array.level.v4u32.f32">;
def int_nvvm_tex_2d_array_grad_v4u32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_i32_ty, llvm_float_ty,
               llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty,
               llvm_float_ty], [],
              "llvm.nvvm.tex.2d.array.grad.v4u32.f32">;

def int_nvvm_tex_3d_v4f32_s32
  : Intrinsic<[llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [], "llvm.nvvm.tex.3d.v4f32.s32">;
def int_nvvm_tex_3d_v4f32_f32
  : Intrinsic<[llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_float_ty, llvm_float_ty,
               llvm_float_ty], [],
              "llvm.nvvm.tex.3d.v4f32.f32">;
def int_nvvm_tex_3d_level_v4f32_f32
  : Intrinsic<[llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_float_ty, llvm_float_ty,
               llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.3d.level.v4f32.f32">;
def int_nvvm_tex_3d_grad_v4f32_f32
  : Intrinsic<[llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_float_ty, llvm_float_ty,
               llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty,
               llvm_float_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.3d.grad.v4f32.f32">;
def int_nvvm_tex_3d_v4s32_s32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [], "llvm.nvvm.tex.3d.v4s32.s32">;
def int_nvvm_tex_3d_v4s32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_float_ty, llvm_float_ty,
               llvm_float_ty], [],
              "llvm.nvvm.tex.3d.v4s32.f32">;
def int_nvvm_tex_3d_level_v4s32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_float_ty, llvm_float_ty,
               llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.3d.level.v4s32.f32">;
def int_nvvm_tex_3d_grad_v4s32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_float_ty, llvm_float_ty,
               llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty,
               llvm_float_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.3d.grad.v4s32.f32">;
def int_nvvm_tex_3d_v4u32_s32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [], "llvm.nvvm.tex.3d.v4u32.s32">;
def int_nvvm_tex_3d_v4u32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_float_ty, llvm_float_ty,
               llvm_float_ty], [],
              "llvm.nvvm.tex.3d.v4u32.f32">;
def int_nvvm_tex_3d_level_v4u32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_float_ty, llvm_float_ty,
               llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.3d.level.v4u32.f32">;
def int_nvvm_tex_3d_grad_v4u32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_float_ty, llvm_float_ty,
               llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty,
               llvm_float_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.3d.grad.v4u32.f32">;

def int_nvvm_tex_cube_v4f32_f32
  : Intrinsic<[llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty],
              [llvm_i64_ty, llvm_i64_ty,
               llvm_float_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.cube.v4f32.f32">;
def int_nvvm_tex_cube_level_v4f32_f32
  : Intrinsic<[llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty],
              [llvm_i64_ty, llvm_i64_ty,
               llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.cube.level.v4f32.f32">;
def int_nvvm_tex_cube_v4s32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i64_ty,
               llvm_float_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.cube.v4s32.f32">;
def int_nvvm_tex_cube_level_v4s32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i64_ty,
               llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.cube.level.v4s32.f32">;
def int_nvvm_tex_cube_v4u32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i64_ty,
               llvm_float_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.cube.v4u32.f32">;
def int_nvvm_tex_cube_level_v4u32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i64_ty,
               llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.cube.level.v4u32.f32">;

def int_nvvm_tex_cube_array_v4f32_f32
  : Intrinsic<[llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_i32_ty,
               llvm_float_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.cube.array.v4f32.f32">;
def int_nvvm_tex_cube_array_level_v4f32_f32
  : Intrinsic<[llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_i32_ty,
               llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.cube.array.level.v4f32.f32">;
def int_nvvm_tex_cube_array_v4s32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_i32_ty,
               llvm_float_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.cube.array.v4s32.f32">;
def int_nvvm_tex_cube_array_level_v4s32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_i32_ty,
               llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.cube.array.level.v4s32.f32">;
def int_nvvm_tex_cube_array_v4u32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_i32_ty,
               llvm_float_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.cube.array.v4u32.f32">;
def int_nvvm_tex_cube_array_level_v4u32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_i32_ty,
               llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.cube.array.level.v4u32.f32">;

def int_nvvm_tld4_r_2d_v4f32_f32
  : Intrinsic<[llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tld4.r.2d.v4f32.f32">;
def int_nvvm_tld4_g_2d_v4f32_f32
  : Intrinsic<[llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tld4.g.2d.v4f32.f32">;
def int_nvvm_tld4_b_2d_v4f32_f32
  : Intrinsic<[llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tld4.b.2d.v4f32.f32">;
def int_nvvm_tld4_a_2d_v4f32_f32
  : Intrinsic<[llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tld4.a.2d.v4f32.f32">;
def int_nvvm_tld4_r_2d_v4s32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tld4.r.2d.v4s32.f32">;
def int_nvvm_tld4_g_2d_v4s32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tld4.g.2d.v4s32.f32">;
def int_nvvm_tld4_b_2d_v4s32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tld4.b.2d.v4s32.f32">;
def int_nvvm_tld4_a_2d_v4s32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tld4.a.2d.v4s32.f32">;
def int_nvvm_tld4_r_2d_v4u32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tld4.r.2d.v4u32.f32">;
def int_nvvm_tld4_g_2d_v4u32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tld4.g.2d.v4u32.f32">;
def int_nvvm_tld4_b_2d_v4u32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tld4.b.2d.v4u32.f32">;
def int_nvvm_tld4_a_2d_v4u32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i64_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tld4.a.2d.v4u32.f32">;


// texmode_unified
def int_nvvm_tex_unified_1d_v4f32_s32
  : Intrinsic<[llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty],
              [llvm_i64_ty, llvm_i32_ty], [],
              "llvm.nvvm.tex.unified.1d.v4f32.s32">;
def int_nvvm_tex_unified_1d_v4f32_f32
  : Intrinsic<[llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty],
              [llvm_i64_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.unified.1d.v4f32.f32">;
def int_nvvm_tex_unified_1d_level_v4f32_f32
  : Intrinsic<[llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty],
              [llvm_i64_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.unified.1d.level.v4f32.f32">;
def int_nvvm_tex_unified_1d_grad_v4f32_f32
  : Intrinsic<[llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty],
              [llvm_i64_ty, llvm_float_ty, llvm_float_ty,
               llvm_float_ty], [],
              "llvm.nvvm.tex.unified.1d.grad.v4f32.f32">;
def int_nvvm_tex_unified_1d_v4s32_s32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty], [],
              "llvm.nvvm.tex.unified.1d.v4s32.s32">;
def int_nvvm_tex_unified_1d_v4s32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.unified.1d.v4s32.f32">;
def int_nvvm_tex_unified_1d_level_v4s32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.unified.1d.level.v4s32.f32">;
def int_nvvm_tex_unified_1d_grad_v4s32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_float_ty, llvm_float_ty,
               llvm_float_ty], [],
              "llvm.nvvm.tex.unified.1d.grad.v4s32.f32">;
def int_nvvm_tex_unified_1d_v4u32_s32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty], [],
              "llvm.nvvm.tex.unified.1d.v4u32.s32">;
def int_nvvm_tex_unified_1d_v4u32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.unified.1d.v4u32.f32">;
def int_nvvm_tex_unified_1d_level_v4u32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.unified.1d.level.v4u32.f32">;
def int_nvvm_tex_unified_1d_grad_v4u32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_float_ty, llvm_float_ty,
               llvm_float_ty], [],
              "llvm.nvvm.tex.unified.1d.grad.v4u32.f32">;

def int_nvvm_tex_unified_1d_array_v4f32_s32
  : Intrinsic<[llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.tex.unified.1d.array.v4f32.s32">;
def int_nvvm_tex_unified_1d_array_v4f32_f32
  : Intrinsic<[llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.unified.1d.array.v4f32.f32">;
def int_nvvm_tex_unified_1d_array_level_v4f32_f32
  : Intrinsic<[llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_float_ty,
               llvm_float_ty], [],
              "llvm.nvvm.tex.unified.1d.array.level.v4f32.f32">;
def int_nvvm_tex_unified_1d_array_grad_v4f32_f32
  : Intrinsic<[llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_float_ty,
               llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.unified.1d.array.grad.v4f32.f32">;
def int_nvvm_tex_unified_1d_array_v4s32_s32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.tex.unified.1d.array.v4s32.s32">;
def int_nvvm_tex_unified_1d_array_v4s32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.unified.1d.array.v4s32.f32">;
def int_nvvm_tex_unified_1d_array_level_v4s32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_float_ty,
               llvm_float_ty], [],
              "llvm.nvvm.tex.unified.1d.array.level.v4s32.f32">;
def int_nvvm_tex_unified_1d_array_grad_v4s32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_float_ty,
               llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.unified.1d.array.grad.v4s32.f32">;
def int_nvvm_tex_unified_1d_array_v4u32_s32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.tex.unified.1d.array.v4u32.s32">;
def int_nvvm_tex_unified_1d_array_v4u32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.unified.1d.array.v4u32.f32">;
def int_nvvm_tex_unified_1d_array_level_v4u32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_float_ty,
               llvm_float_ty], [],
              "llvm.nvvm.tex.unified.1d.array.level.v4u32.f32">;
def int_nvvm_tex_unified_1d_array_grad_v4u32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_float_ty,
               llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.unified.1d.array.grad.v4u32.f32">;

def int_nvvm_tex_unified_2d_v4f32_s32
  : Intrinsic<[llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.tex.unified.2d.v4f32.s32">;
def int_nvvm_tex_unified_2d_v4f32_f32
  : Intrinsic<[llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty],
              [llvm_i64_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.unified.2d.v4f32.f32">;
def int_nvvm_tex_unified_2d_level_v4f32_f32
  : Intrinsic<[llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty],
              [llvm_i64_ty, llvm_float_ty, llvm_float_ty,
               llvm_float_ty], [],
              "llvm.nvvm.tex.unified.2d.level.v4f32.f32">;
def int_nvvm_tex_unified_2d_grad_v4f32_f32
  : Intrinsic<[llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty],
              [llvm_i64_ty, llvm_float_ty, llvm_float_ty,
               llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.unified.2d.grad.v4f32.f32">;
def int_nvvm_tex_unified_2d_v4s32_s32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.tex.unified.2d.v4s32.s32">;
def int_nvvm_tex_unified_2d_v4s32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.unified.2d.v4s32.f32">;
def int_nvvm_tex_unified_2d_level_v4s32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_float_ty, llvm_float_ty,
               llvm_float_ty], [],
              "llvm.nvvm.tex.unified.2d.level.v4s32.f32">;
def int_nvvm_tex_unified_2d_grad_v4s32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_float_ty, llvm_float_ty,
               llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.unified.2d.grad.v4s32.f32">;
def int_nvvm_tex_unified_2d_v4u32_s32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.tex.unified.2d.v4u32.s32">;
def int_nvvm_tex_unified_2d_v4u32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.unified.2d.v4u32.f32">;
def int_nvvm_tex_unified_2d_level_v4u32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_float_ty, llvm_float_ty,
               llvm_float_ty], [],
              "llvm.nvvm.tex.unified.2d.level.v4u32.f32">;
def int_nvvm_tex_unified_2d_grad_v4u32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_float_ty, llvm_float_ty,
               llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.unified.2d.grad.v4u32.f32">;

def int_nvvm_tex_unified_2d_array_v4f32_s32
  : Intrinsic<[llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
               llvm_i32_ty], [],
              "llvm.nvvm.tex.unified.2d.array.v4f32.s32">;
def int_nvvm_tex_unified_2d_array_v4f32_f32
  : Intrinsic<[llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_float_ty,
               llvm_float_ty], [],
              "llvm.nvvm.tex.unified.2d.array.v4f32.f32">;
def int_nvvm_tex_unified_2d_array_level_v4f32_f32
  : Intrinsic<[llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_float_ty,
               llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.unified.2d.array.level.v4f32.f32">;
def int_nvvm_tex_unified_2d_array_grad_v4f32_f32
  : Intrinsic<[llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_float_ty,
               llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty,
               llvm_float_ty], [],
              "llvm.nvvm.tex.unified.2d.array.grad.v4f32.f32">;
def int_nvvm_tex_unified_2d_array_v4s32_s32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
               llvm_i32_ty], [],
              "llvm.nvvm.tex.unified.2d.array.v4s32.s32">;
def int_nvvm_tex_unified_2d_array_v4s32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_float_ty,
               llvm_float_ty], [],
              "llvm.nvvm.tex.unified.2d.array.v4s32.f32">;
def int_nvvm_tex_unified_2d_array_level_v4s32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_float_ty,
               llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.unified.2d.array.level.v4s32.f32">;
def int_nvvm_tex_unified_2d_array_grad_v4s32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_float_ty,
               llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty,
               llvm_float_ty], [],
              "llvm.nvvm.tex.unified.2d.array.grad.v4s32.f32">;
def int_nvvm_tex_unified_2d_array_v4u32_s32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
               llvm_i32_ty], [],
              "llvm.nvvm.tex.unified.2d.array.v4u32.s32">;
def int_nvvm_tex_unified_2d_array_v4u32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_float_ty,
               llvm_float_ty], [],
              "llvm.nvvm.tex.unified.2d.array.v4u32.f32">;
def int_nvvm_tex_unified_2d_array_level_v4u32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_float_ty,
               llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.unified.2d.array.level.v4u32.f32">;
def int_nvvm_tex_unified_2d_array_grad_v4u32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_float_ty,
               llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty,
               llvm_float_ty], [],
              "llvm.nvvm.tex.unified.2d.array.grad.v4u32.f32">;

def int_nvvm_tex_unified_3d_v4f32_s32
  : Intrinsic<[llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [], "llvm.nvvm.tex.unified.3d.v4f32.s32">;
def int_nvvm_tex_unified_3d_v4f32_f32
  : Intrinsic<[llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty],
              [llvm_i64_ty, llvm_float_ty, llvm_float_ty,
               llvm_float_ty], [],
              "llvm.nvvm.tex.unified.3d.v4f32.f32">;
def int_nvvm_tex_unified_3d_level_v4f32_f32
  : Intrinsic<[llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty],
              [llvm_i64_ty, llvm_float_ty, llvm_float_ty,
               llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.unified.3d.level.v4f32.f32">;
def int_nvvm_tex_unified_3d_grad_v4f32_f32
  : Intrinsic<[llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty],
              [llvm_i64_ty, llvm_float_ty, llvm_float_ty,
               llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty,
               llvm_float_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.unified.3d.grad.v4f32.f32">;
def int_nvvm_tex_unified_3d_v4s32_s32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [], "llvm.nvvm.tex.unified.3d.v4s32.s32">;
def int_nvvm_tex_unified_3d_v4s32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_float_ty, llvm_float_ty,
               llvm_float_ty], [],
              "llvm.nvvm.tex.unified.3d.v4s32.f32">;
def int_nvvm_tex_unified_3d_level_v4s32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_float_ty, llvm_float_ty,
               llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.unified.3d.level.v4s32.f32">;
def int_nvvm_tex_unified_3d_grad_v4s32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_float_ty, llvm_float_ty,
               llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty,
               llvm_float_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.unified.3d.grad.v4s32.f32">;
def int_nvvm_tex_unified_3d_v4u32_s32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [], "llvm.nvvm.tex.unified.3d.v4u32.s32">;
def int_nvvm_tex_unified_3d_v4u32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_float_ty, llvm_float_ty,
               llvm_float_ty], [],
              "llvm.nvvm.tex.unified.3d.v4u32.f32">;
def int_nvvm_tex_unified_3d_level_v4u32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_float_ty, llvm_float_ty,
               llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.unified.3d.level.v4u32.f32">;
def int_nvvm_tex_unified_3d_grad_v4u32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_float_ty, llvm_float_ty,
               llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty,
               llvm_float_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.unified.3d.grad.v4u32.f32">;

def int_nvvm_tex_unified_cube_v4f32_f32
  : Intrinsic<[llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty],
              [llvm_i64_ty,
               llvm_float_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.unified.cube.v4f32.f32">;
def int_nvvm_tex_unified_cube_level_v4f32_f32
  : Intrinsic<[llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty],
              [llvm_i64_ty,
               llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.unified.cube.level.v4f32.f32">;
def int_nvvm_tex_unified_cube_v4s32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty,
               llvm_float_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.unified.cube.v4s32.f32">;
def int_nvvm_tex_unified_cube_level_v4s32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty,
               llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.unified.cube.level.v4s32.f32">;
def int_nvvm_tex_unified_cube_v4u32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty,
               llvm_float_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.unified.cube.v4u32.f32">;
def int_nvvm_tex_unified_cube_level_v4u32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty,
               llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.unified.cube.level.v4u32.f32">;

def int_nvvm_tex_unified_cube_array_v4f32_f32
  : Intrinsic<[llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty],
              [llvm_i64_ty, llvm_i32_ty,
               llvm_float_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.unified.cube.array.v4f32.f32">;
def int_nvvm_tex_unified_cube_array_level_v4f32_f32
  : Intrinsic<[llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty],
              [llvm_i64_ty, llvm_i32_ty,
               llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.unified.cube.array.level.v4f32.f32">;
def int_nvvm_tex_unified_cube_array_v4s32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty,
               llvm_float_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.unified.cube.array.v4s32.f32">;
def int_nvvm_tex_unified_cube_array_level_v4s32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty,
               llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.unified.cube.array.level.v4s32.f32">;
def int_nvvm_tex_unified_cube_array_v4u32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty,
               llvm_float_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.unified.cube.array.v4u32.f32">;
def int_nvvm_tex_unified_cube_array_level_v4u32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty,
               llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tex.unified.cube.array.level.v4u32.f32">;

def int_nvvm_tld4_unified_r_2d_v4f32_f32
  : Intrinsic<[llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty],
              [llvm_i64_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tld4.unified.r.2d.v4f32.f32">;
def int_nvvm_tld4_unified_g_2d_v4f32_f32
  : Intrinsic<[llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty],
              [llvm_i64_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tld4.unified.g.2d.v4f32.f32">;
def int_nvvm_tld4_unified_b_2d_v4f32_f32
  : Intrinsic<[llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty],
              [llvm_i64_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tld4.unified.b.2d.v4f32.f32">;
def int_nvvm_tld4_unified_a_2d_v4f32_f32
  : Intrinsic<[llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty],
              [llvm_i64_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tld4.unified.a.2d.v4f32.f32">;
def int_nvvm_tld4_unified_r_2d_v4s32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tld4.unified.r.2d.v4s32.f32">;
def int_nvvm_tld4_unified_g_2d_v4s32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tld4.unified.g.2d.v4s32.f32">;
def int_nvvm_tld4_unified_b_2d_v4s32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tld4.unified.b.2d.v4s32.f32">;
def int_nvvm_tld4_unified_a_2d_v4s32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tld4.unified.a.2d.v4s32.f32">;
def int_nvvm_tld4_unified_r_2d_v4u32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tld4.unified.r.2d.v4u32.f32">;
def int_nvvm_tld4_unified_g_2d_v4u32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tld4.unified.g.2d.v4u32.f32">;
def int_nvvm_tld4_unified_b_2d_v4u32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tld4.unified.b.2d.v4u32.f32">;
def int_nvvm_tld4_unified_a_2d_v4u32_f32
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_float_ty, llvm_float_ty], [],
              "llvm.nvvm.tld4.unified.a.2d.v4u32.f32">;


//=== Surface Load
// .clamp variants
def int_nvvm_suld_1d_i8_clamp
  : Intrinsic<[llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.1d.i8.clamp">;
def int_nvvm_suld_1d_i16_clamp
  : Intrinsic<[llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.1d.i16.clamp">;
def int_nvvm_suld_1d_i32_clamp
  : Intrinsic<[llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.1d.i32.clamp">;
def int_nvvm_suld_1d_i64_clamp
  : Intrinsic<[llvm_i64_ty],
              [llvm_i64_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.1d.i64.clamp">;
def int_nvvm_suld_1d_v2i8_clamp
  : Intrinsic<[llvm_i16_ty, llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.1d.v2i8.clamp">;
def int_nvvm_suld_1d_v2i16_clamp
  : Intrinsic<[llvm_i16_ty, llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.1d.v2i16.clamp">;
def int_nvvm_suld_1d_v2i32_clamp
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.1d.v2i32.clamp">;
def int_nvvm_suld_1d_v2i64_clamp
  : Intrinsic<[llvm_i64_ty, llvm_i64_ty],
              [llvm_i64_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.1d.v2i64.clamp">;
def int_nvvm_suld_1d_v4i8_clamp
  : Intrinsic<[llvm_i16_ty, llvm_i16_ty, llvm_i16_ty, llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.1d.v4i8.clamp">;
def int_nvvm_suld_1d_v4i16_clamp
  : Intrinsic<[llvm_i16_ty, llvm_i16_ty, llvm_i16_ty, llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.1d.v4i16.clamp">;
def int_nvvm_suld_1d_v4i32_clamp
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.1d.v4i32.clamp">;

def int_nvvm_suld_1d_array_i8_clamp
  : Intrinsic<[llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.1d.array.i8.clamp">;
def int_nvvm_suld_1d_array_i16_clamp
  : Intrinsic<[llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.1d.array.i16.clamp">;
def int_nvvm_suld_1d_array_i32_clamp
  : Intrinsic<[llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.1d.array.i32.clamp">;
def int_nvvm_suld_1d_array_i64_clamp
  : Intrinsic<[llvm_i64_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.1d.array.i64.clamp">;
def int_nvvm_suld_1d_array_v2i8_clamp
  : Intrinsic<[llvm_i16_ty, llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.1d.array.v2i8.clamp">;
def int_nvvm_suld_1d_array_v2i16_clamp
  : Intrinsic<[llvm_i16_ty, llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.1d.array.v2i16.clamp">;
def int_nvvm_suld_1d_array_v2i32_clamp
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.1d.array.v2i32.clamp">;
def int_nvvm_suld_1d_array_v2i64_clamp
  : Intrinsic<[llvm_i64_ty, llvm_i64_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.1d.array.v2i64.clamp">;
def int_nvvm_suld_1d_array_v4i8_clamp
  : Intrinsic<[llvm_i16_ty, llvm_i16_ty, llvm_i16_ty, llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.1d.array.v4i8.clamp">;
def int_nvvm_suld_1d_array_v4i16_clamp
  : Intrinsic<[llvm_i16_ty, llvm_i16_ty, llvm_i16_ty, llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.1d.array.v4i16.clamp">;
def int_nvvm_suld_1d_array_v4i32_clamp
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.1d.array.v4i32.clamp">;

def int_nvvm_suld_2d_i8_clamp
  : Intrinsic<[llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.2d.i8.clamp">;
def int_nvvm_suld_2d_i16_clamp
  : Intrinsic<[llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.2d.i16.clamp">;
def int_nvvm_suld_2d_i32_clamp
  : Intrinsic<[llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.2d.i32.clamp">;
def int_nvvm_suld_2d_i64_clamp
  : Intrinsic<[llvm_i64_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.2d.i64.clamp">;
def int_nvvm_suld_2d_v2i8_clamp
  : Intrinsic<[llvm_i16_ty, llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.2d.v2i8.clamp">;
def int_nvvm_suld_2d_v2i16_clamp
  : Intrinsic<[llvm_i16_ty, llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.2d.v2i16.clamp">;
def int_nvvm_suld_2d_v2i32_clamp
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.2d.v2i32.clamp">;
def int_nvvm_suld_2d_v2i64_clamp
  : Intrinsic<[llvm_i64_ty, llvm_i64_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.2d.v2i64.clamp">;
def int_nvvm_suld_2d_v4i8_clamp
  : Intrinsic<[llvm_i16_ty, llvm_i16_ty, llvm_i16_ty, llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.2d.v4i8.clamp">;
def int_nvvm_suld_2d_v4i16_clamp
  : Intrinsic<[llvm_i16_ty, llvm_i16_ty, llvm_i16_ty, llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.2d.v4i16.clamp">;
def int_nvvm_suld_2d_v4i32_clamp
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.2d.v4i32.clamp">;

def int_nvvm_suld_2d_array_i8_clamp
  : Intrinsic<[llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.2d.array.i8.clamp">;
def int_nvvm_suld_2d_array_i16_clamp
  : Intrinsic<[llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.2d.array.i16.clamp">;
def int_nvvm_suld_2d_array_i32_clamp
  : Intrinsic<[llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.2d.array.i32.clamp">;
def int_nvvm_suld_2d_array_i64_clamp
  : Intrinsic<[llvm_i64_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.2d.array.i64.clamp">;
def int_nvvm_suld_2d_array_v2i8_clamp
  : Intrinsic<[llvm_i16_ty, llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.2d.array.v2i8.clamp">;
def int_nvvm_suld_2d_array_v2i16_clamp
  : Intrinsic<[llvm_i16_ty, llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.2d.array.v2i16.clamp">;
def int_nvvm_suld_2d_array_v2i32_clamp
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.2d.array.v2i32.clamp">;
def int_nvvm_suld_2d_array_v2i64_clamp
  : Intrinsic<[llvm_i64_ty, llvm_i64_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.2d.array.v2i64.clamp">;
def int_nvvm_suld_2d_array_v4i8_clamp
  : Intrinsic<[llvm_i16_ty, llvm_i16_ty, llvm_i16_ty, llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.2d.array.v4i8.clamp">;
def int_nvvm_suld_2d_array_v4i16_clamp
  : Intrinsic<[llvm_i16_ty, llvm_i16_ty, llvm_i16_ty, llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.2d.array.v4i16.clamp">;
def int_nvvm_suld_2d_array_v4i32_clamp
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.2d.array.v4i32.clamp">;

def int_nvvm_suld_3d_i8_clamp
  : Intrinsic<[llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.3d.i8.clamp">;
def int_nvvm_suld_3d_i16_clamp
  : Intrinsic<[llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.3d.i16.clamp">;
def int_nvvm_suld_3d_i32_clamp
  : Intrinsic<[llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.3d.i32.clamp">;
def int_nvvm_suld_3d_i64_clamp
  : Intrinsic<[llvm_i64_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.3d.i64.clamp">;
def int_nvvm_suld_3d_v2i8_clamp
  : Intrinsic<[llvm_i16_ty, llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.3d.v2i8.clamp">;
def int_nvvm_suld_3d_v2i16_clamp
  : Intrinsic<[llvm_i16_ty, llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.3d.v2i16.clamp">;
def int_nvvm_suld_3d_v2i32_clamp
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.3d.v2i32.clamp">;
def int_nvvm_suld_3d_v2i64_clamp
  : Intrinsic<[llvm_i64_ty, llvm_i64_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.3d.v2i64.clamp">;
def int_nvvm_suld_3d_v4i8_clamp
  : Intrinsic<[llvm_i16_ty, llvm_i16_ty, llvm_i16_ty, llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.3d.v4i8.clamp">;
def int_nvvm_suld_3d_v4i16_clamp
  : Intrinsic<[llvm_i16_ty, llvm_i16_ty, llvm_i16_ty, llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.3d.v4i16.clamp">;
def int_nvvm_suld_3d_v4i32_clamp
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.3d.v4i32.clamp">;

// .trap variants
def int_nvvm_suld_1d_i8_trap
  : Intrinsic<[llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.1d.i8.trap">;
def int_nvvm_suld_1d_i16_trap
  : Intrinsic<[llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.1d.i16.trap">;
def int_nvvm_suld_1d_i32_trap
  : Intrinsic<[llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.1d.i32.trap">;
def int_nvvm_suld_1d_i64_trap
  : Intrinsic<[llvm_i64_ty],
              [llvm_i64_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.1d.i64.trap">;
def int_nvvm_suld_1d_v2i8_trap
  : Intrinsic<[llvm_i16_ty, llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.1d.v2i8.trap">;
def int_nvvm_suld_1d_v2i16_trap
  : Intrinsic<[llvm_i16_ty, llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.1d.v2i16.trap">;
def int_nvvm_suld_1d_v2i32_trap
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.1d.v2i32.trap">;
def int_nvvm_suld_1d_v2i64_trap
  : Intrinsic<[llvm_i64_ty, llvm_i64_ty],
              [llvm_i64_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.1d.v2i64.trap">;
def int_nvvm_suld_1d_v4i8_trap
  : Intrinsic<[llvm_i16_ty, llvm_i16_ty, llvm_i16_ty, llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.1d.v4i8.trap">;
def int_nvvm_suld_1d_v4i16_trap
  : Intrinsic<[llvm_i16_ty, llvm_i16_ty, llvm_i16_ty, llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.1d.v4i16.trap">;
def int_nvvm_suld_1d_v4i32_trap
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.1d.v4i32.trap">;

def int_nvvm_suld_1d_array_i8_trap
  : Intrinsic<[llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.1d.array.i8.trap">;
def int_nvvm_suld_1d_array_i16_trap
  : Intrinsic<[llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.1d.array.i16.trap">;
def int_nvvm_suld_1d_array_i32_trap
  : Intrinsic<[llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.1d.array.i32.trap">;
def int_nvvm_suld_1d_array_i64_trap
  : Intrinsic<[llvm_i64_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.1d.array.i64.trap">;
def int_nvvm_suld_1d_array_v2i8_trap
  : Intrinsic<[llvm_i16_ty, llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.1d.array.v2i8.trap">;
def int_nvvm_suld_1d_array_v2i16_trap
  : Intrinsic<[llvm_i16_ty, llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.1d.array.v2i16.trap">;
def int_nvvm_suld_1d_array_v2i32_trap
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.1d.array.v2i32.trap">;
def int_nvvm_suld_1d_array_v2i64_trap
  : Intrinsic<[llvm_i64_ty, llvm_i64_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.1d.array.v2i64.trap">;
def int_nvvm_suld_1d_array_v4i8_trap
  : Intrinsic<[llvm_i16_ty, llvm_i16_ty, llvm_i16_ty, llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.1d.array.v4i8.trap">;
def int_nvvm_suld_1d_array_v4i16_trap
  : Intrinsic<[llvm_i16_ty, llvm_i16_ty, llvm_i16_ty, llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.1d.array.v4i16.trap">;
def int_nvvm_suld_1d_array_v4i32_trap
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.1d.array.v4i32.trap">;

def int_nvvm_suld_2d_i8_trap
  : Intrinsic<[llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.2d.i8.trap">;
def int_nvvm_suld_2d_i16_trap
  : Intrinsic<[llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.2d.i16.trap">;
def int_nvvm_suld_2d_i32_trap
  : Intrinsic<[llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.2d.i32.trap">;
def int_nvvm_suld_2d_i64_trap
  : Intrinsic<[llvm_i64_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.2d.i64.trap">;
def int_nvvm_suld_2d_v2i8_trap
  : Intrinsic<[llvm_i16_ty, llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.2d.v2i8.trap">;
def int_nvvm_suld_2d_v2i16_trap
  : Intrinsic<[llvm_i16_ty, llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.2d.v2i16.trap">;
def int_nvvm_suld_2d_v2i32_trap
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.2d.v2i32.trap">;
def int_nvvm_suld_2d_v2i64_trap
  : Intrinsic<[llvm_i64_ty, llvm_i64_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.2d.v2i64.trap">;
def int_nvvm_suld_2d_v4i8_trap
  : Intrinsic<[llvm_i16_ty, llvm_i16_ty, llvm_i16_ty, llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.2d.v4i8.trap">;
def int_nvvm_suld_2d_v4i16_trap
  : Intrinsic<[llvm_i16_ty, llvm_i16_ty, llvm_i16_ty, llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.2d.v4i16.trap">;
def int_nvvm_suld_2d_v4i32_trap
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.2d.v4i32.trap">;

def int_nvvm_suld_2d_array_i8_trap
  : Intrinsic<[llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.2d.array.i8.trap">;
def int_nvvm_suld_2d_array_i16_trap
  : Intrinsic<[llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.2d.array.i16.trap">;
def int_nvvm_suld_2d_array_i32_trap
  : Intrinsic<[llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.2d.array.i32.trap">;
def int_nvvm_suld_2d_array_i64_trap
  : Intrinsic<[llvm_i64_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.2d.array.i64.trap">;
def int_nvvm_suld_2d_array_v2i8_trap
  : Intrinsic<[llvm_i16_ty, llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.2d.array.v2i8.trap">;
def int_nvvm_suld_2d_array_v2i16_trap
  : Intrinsic<[llvm_i16_ty, llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.2d.array.v2i16.trap">;
def int_nvvm_suld_2d_array_v2i32_trap
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.2d.array.v2i32.trap">;
def int_nvvm_suld_2d_array_v2i64_trap
  : Intrinsic<[llvm_i64_ty, llvm_i64_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.2d.array.v2i64.trap">;
def int_nvvm_suld_2d_array_v4i8_trap
  : Intrinsic<[llvm_i16_ty, llvm_i16_ty, llvm_i16_ty, llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.2d.array.v4i8.trap">;
def int_nvvm_suld_2d_array_v4i16_trap
  : Intrinsic<[llvm_i16_ty, llvm_i16_ty, llvm_i16_ty, llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.2d.array.v4i16.trap">;
def int_nvvm_suld_2d_array_v4i32_trap
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.2d.array.v4i32.trap">;

def int_nvvm_suld_3d_i8_trap
  : Intrinsic<[llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.3d.i8.trap">;
def int_nvvm_suld_3d_i16_trap
  : Intrinsic<[llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.3d.i16.trap">;
def int_nvvm_suld_3d_i32_trap
  : Intrinsic<[llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.3d.i32.trap">;
def int_nvvm_suld_3d_i64_trap
  : Intrinsic<[llvm_i64_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.3d.i64.trap">;
def int_nvvm_suld_3d_v2i8_trap
  : Intrinsic<[llvm_i16_ty, llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.3d.v2i8.trap">;
def int_nvvm_suld_3d_v2i16_trap
  : Intrinsic<[llvm_i16_ty, llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.3d.v2i16.trap">;
def int_nvvm_suld_3d_v2i32_trap
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.3d.v2i32.trap">;
def int_nvvm_suld_3d_v2i64_trap
  : Intrinsic<[llvm_i64_ty, llvm_i64_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.3d.v2i64.trap">;
def int_nvvm_suld_3d_v4i8_trap
  : Intrinsic<[llvm_i16_ty, llvm_i16_ty, llvm_i16_ty, llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.3d.v4i8.trap">;
def int_nvvm_suld_3d_v4i16_trap
  : Intrinsic<[llvm_i16_ty, llvm_i16_ty, llvm_i16_ty, llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.3d.v4i16.trap">;
def int_nvvm_suld_3d_v4i32_trap
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.3d.v4i32.trap">;

// .zero variants
def int_nvvm_suld_1d_i8_zero
  : Intrinsic<[llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.1d.i8.zero">;
def int_nvvm_suld_1d_i16_zero
  : Intrinsic<[llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.1d.i16.zero">;
def int_nvvm_suld_1d_i32_zero
  : Intrinsic<[llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.1d.i32.zero">;
def int_nvvm_suld_1d_i64_zero
  : Intrinsic<[llvm_i64_ty],
              [llvm_i64_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.1d.i64.zero">;
def int_nvvm_suld_1d_v2i8_zero
  : Intrinsic<[llvm_i16_ty, llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.1d.v2i8.zero">;
def int_nvvm_suld_1d_v2i16_zero
  : Intrinsic<[llvm_i16_ty, llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.1d.v2i16.zero">;
def int_nvvm_suld_1d_v2i32_zero
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.1d.v2i32.zero">;
def int_nvvm_suld_1d_v2i64_zero
  : Intrinsic<[llvm_i64_ty, llvm_i64_ty],
              [llvm_i64_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.1d.v2i64.zero">;
def int_nvvm_suld_1d_v4i8_zero
  : Intrinsic<[llvm_i16_ty, llvm_i16_ty, llvm_i16_ty, llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.1d.v4i8.zero">;
def int_nvvm_suld_1d_v4i16_zero
  : Intrinsic<[llvm_i16_ty, llvm_i16_ty, llvm_i16_ty, llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.1d.v4i16.zero">;
def int_nvvm_suld_1d_v4i32_zero
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.1d.v4i32.zero">;

def int_nvvm_suld_1d_array_i8_zero
  : Intrinsic<[llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.1d.array.i8.zero">;
def int_nvvm_suld_1d_array_i16_zero
  : Intrinsic<[llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.1d.array.i16.zero">;
def int_nvvm_suld_1d_array_i32_zero
  : Intrinsic<[llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.1d.array.i32.zero">;
def int_nvvm_suld_1d_array_i64_zero
  : Intrinsic<[llvm_i64_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.1d.array.i64.zero">;
def int_nvvm_suld_1d_array_v2i8_zero
  : Intrinsic<[llvm_i16_ty, llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.1d.array.v2i8.zero">;
def int_nvvm_suld_1d_array_v2i16_zero
  : Intrinsic<[llvm_i16_ty, llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.1d.array.v2i16.zero">;
def int_nvvm_suld_1d_array_v2i32_zero
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.1d.array.v2i32.zero">;
def int_nvvm_suld_1d_array_v2i64_zero
  : Intrinsic<[llvm_i64_ty, llvm_i64_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.1d.array.v2i64.zero">;
def int_nvvm_suld_1d_array_v4i8_zero
  : Intrinsic<[llvm_i16_ty, llvm_i16_ty, llvm_i16_ty, llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.1d.array.v4i8.zero">;
def int_nvvm_suld_1d_array_v4i16_zero
  : Intrinsic<[llvm_i16_ty, llvm_i16_ty, llvm_i16_ty, llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.1d.array.v4i16.zero">;
def int_nvvm_suld_1d_array_v4i32_zero
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.1d.array.v4i32.zero">;

def int_nvvm_suld_2d_i8_zero
  : Intrinsic<[llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.2d.i8.zero">;
def int_nvvm_suld_2d_i16_zero
  : Intrinsic<[llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.2d.i16.zero">;
def int_nvvm_suld_2d_i32_zero
  : Intrinsic<[llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.2d.i32.zero">;
def int_nvvm_suld_2d_i64_zero
  : Intrinsic<[llvm_i64_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.2d.i64.zero">;
def int_nvvm_suld_2d_v2i8_zero
  : Intrinsic<[llvm_i16_ty, llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.2d.v2i8.zero">;
def int_nvvm_suld_2d_v2i16_zero
  : Intrinsic<[llvm_i16_ty, llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.2d.v2i16.zero">;
def int_nvvm_suld_2d_v2i32_zero
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.2d.v2i32.zero">;
def int_nvvm_suld_2d_v2i64_zero
  : Intrinsic<[llvm_i64_ty, llvm_i64_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.2d.v2i64.zero">;
def int_nvvm_suld_2d_v4i8_zero
  : Intrinsic<[llvm_i16_ty, llvm_i16_ty, llvm_i16_ty, llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.2d.v4i8.zero">;
def int_nvvm_suld_2d_v4i16_zero
  : Intrinsic<[llvm_i16_ty, llvm_i16_ty, llvm_i16_ty, llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.2d.v4i16.zero">;
def int_nvvm_suld_2d_v4i32_zero
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.2d.v4i32.zero">;

def int_nvvm_suld_2d_array_i8_zero
  : Intrinsic<[llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.2d.array.i8.zero">;
def int_nvvm_suld_2d_array_i16_zero
  : Intrinsic<[llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.2d.array.i16.zero">;
def int_nvvm_suld_2d_array_i32_zero
  : Intrinsic<[llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.2d.array.i32.zero">;
def int_nvvm_suld_2d_array_i64_zero
  : Intrinsic<[llvm_i64_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.2d.array.i64.zero">;
def int_nvvm_suld_2d_array_v2i8_zero
  : Intrinsic<[llvm_i16_ty, llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.2d.array.v2i8.zero">;
def int_nvvm_suld_2d_array_v2i16_zero
  : Intrinsic<[llvm_i16_ty, llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.2d.array.v2i16.zero">;
def int_nvvm_suld_2d_array_v2i32_zero
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.2d.array.v2i32.zero">;
def int_nvvm_suld_2d_array_v2i64_zero
  : Intrinsic<[llvm_i64_ty, llvm_i64_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.2d.array.v2i64.zero">;
def int_nvvm_suld_2d_array_v4i8_zero
  : Intrinsic<[llvm_i16_ty, llvm_i16_ty, llvm_i16_ty, llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.2d.array.v4i8.zero">;
def int_nvvm_suld_2d_array_v4i16_zero
  : Intrinsic<[llvm_i16_ty, llvm_i16_ty, llvm_i16_ty, llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.2d.array.v4i16.zero">;
def int_nvvm_suld_2d_array_v4i32_zero
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.2d.array.v4i32.zero">;

def int_nvvm_suld_3d_i8_zero
  : Intrinsic<[llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.3d.i8.zero">;
def int_nvvm_suld_3d_i16_zero
  : Intrinsic<[llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.3d.i16.zero">;
def int_nvvm_suld_3d_i32_zero
  : Intrinsic<[llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.3d.i32.zero">;
def int_nvvm_suld_3d_i64_zero
  : Intrinsic<[llvm_i64_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.3d.i64.zero">;
def int_nvvm_suld_3d_v2i8_zero
  : Intrinsic<[llvm_i16_ty, llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.3d.v2i8.zero">;
def int_nvvm_suld_3d_v2i16_zero
  : Intrinsic<[llvm_i16_ty, llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.3d.v2i16.zero">;
def int_nvvm_suld_3d_v2i32_zero
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.3d.v2i32.zero">;
def int_nvvm_suld_3d_v2i64_zero
  : Intrinsic<[llvm_i64_ty, llvm_i64_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.3d.v2i64.zero">;
def int_nvvm_suld_3d_v4i8_zero
  : Intrinsic<[llvm_i16_ty, llvm_i16_ty, llvm_i16_ty, llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.3d.v4i8.zero">;
def int_nvvm_suld_3d_v4i16_zero
  : Intrinsic<[llvm_i16_ty, llvm_i16_ty, llvm_i16_ty, llvm_i16_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.3d.v4i16.zero">;
def int_nvvm_suld_3d_v4i32_zero
  : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.suld.3d.v4i32.zero">;

//===- Texture Query ------------------------------------------------------===//

def int_nvvm_txq_channel_order
  : Intrinsic<[llvm_i32_ty], [llvm_i64_ty], [IntrNoMem],
              "llvm.nvvm.txq.channel.order">,
    GCCBuiltin<"__nvvm_txq_channel_order">;
def int_nvvm_txq_channel_data_type
  : Intrinsic<[llvm_i32_ty], [llvm_i64_ty], [IntrNoMem],
              "llvm.nvvm.txq.channel.data.type">,
    GCCBuiltin<"__nvvm_txq_channel_data_type">;
def int_nvvm_txq_width
  : Intrinsic<[llvm_i32_ty], [llvm_i64_ty], [IntrNoMem],
              "llvm.nvvm.txq.width">,
    GCCBuiltin<"__nvvm_txq_width">;
def int_nvvm_txq_height
  : Intrinsic<[llvm_i32_ty], [llvm_i64_ty], [IntrNoMem],
              "llvm.nvvm.txq.height">,
    GCCBuiltin<"__nvvm_txq_height">;
def int_nvvm_txq_depth
  : Intrinsic<[llvm_i32_ty], [llvm_i64_ty], [IntrNoMem],
              "llvm.nvvm.txq.depth">,
    GCCBuiltin<"__nvvm_txq_depth">;
def int_nvvm_txq_array_size
  : Intrinsic<[llvm_i32_ty], [llvm_i64_ty], [IntrNoMem],
              "llvm.nvvm.txq.array.size">,
    GCCBuiltin<"__nvvm_txq_array_size">;
def int_nvvm_txq_num_samples
  : Intrinsic<[llvm_i32_ty], [llvm_i64_ty], [IntrNoMem],
              "llvm.nvvm.txq.num.samples">,
    GCCBuiltin<"__nvvm_txq_num_samples">;
def int_nvvm_txq_num_mipmap_levels
  : Intrinsic<[llvm_i32_ty], [llvm_i64_ty], [IntrNoMem],
              "llvm.nvvm.txq.num.mipmap.levels">,
    GCCBuiltin<"__nvvm_txq_num_mipmap_levels">;

//===- Surface Query ------------------------------------------------------===//

def int_nvvm_suq_channel_order
  : Intrinsic<[llvm_i32_ty], [llvm_i64_ty], [IntrNoMem],
              "llvm.nvvm.suq.channel.order">,
    GCCBuiltin<"__nvvm_suq_channel_order">;
def int_nvvm_suq_channel_data_type
  : Intrinsic<[llvm_i32_ty], [llvm_i64_ty], [IntrNoMem],
              "llvm.nvvm.suq.channel.data.type">,
    GCCBuiltin<"__nvvm_suq_channel_data_type">;
def int_nvvm_suq_width
  : Intrinsic<[llvm_i32_ty], [llvm_i64_ty], [IntrNoMem],
              "llvm.nvvm.suq.width">,
    GCCBuiltin<"__nvvm_suq_width">;
def int_nvvm_suq_height
  : Intrinsic<[llvm_i32_ty], [llvm_i64_ty], [IntrNoMem],
              "llvm.nvvm.suq.height">,
    GCCBuiltin<"__nvvm_suq_height">;
def int_nvvm_suq_depth
  : Intrinsic<[llvm_i32_ty], [llvm_i64_ty], [IntrNoMem],
              "llvm.nvvm.suq.depth">,
    GCCBuiltin<"__nvvm_suq_depth">;
def int_nvvm_suq_array_size
  : Intrinsic<[llvm_i32_ty], [llvm_i64_ty], [IntrNoMem],
              "llvm.nvvm.suq.array.size">,
    GCCBuiltin<"__nvvm_suq_array_size">;


//===- Handle Query -------------------------------------------------------===//

def int_nvvm_istypep_sampler
  : Intrinsic<[llvm_i1_ty], [llvm_i64_ty], [IntrNoMem],
              "llvm.nvvm.istypep.sampler">,
    GCCBuiltin<"__nvvm_istypep_sampler">;
def int_nvvm_istypep_surface
  : Intrinsic<[llvm_i1_ty], [llvm_i64_ty], [IntrNoMem],
              "llvm.nvvm.istypep.surface">,
    GCCBuiltin<"__nvvm_istypep_surface">;
def int_nvvm_istypep_texture
  : Intrinsic<[llvm_i1_ty], [llvm_i64_ty], [IntrNoMem],
              "llvm.nvvm.istypep.texture">,
    GCCBuiltin<"__nvvm_istypep_texture">;



//===- Surface Stores -----------------------------------------------------===//

// Unformatted
// .clamp variant
def int_nvvm_sust_b_1d_i8_clamp
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.1d.i8.clamp">,
    GCCBuiltin<"__nvvm_sust_b_1d_i8_clamp">;
def int_nvvm_sust_b_1d_i16_clamp
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.1d.i16.clamp">,
    GCCBuiltin<"__nvvm_sust_b_1d_i16_clamp">;
def int_nvvm_sust_b_1d_i32_clamp
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.sust.b.1d.i32.clamp">,
    GCCBuiltin<"__nvvm_sust_b_1d_i32_clamp">;
def int_nvvm_sust_b_1d_i64_clamp
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i64_ty], [],
              "llvm.nvvm.sust.b.1d.i64.clamp">,
    GCCBuiltin<"__nvvm_sust_b_1d_i64_clamp">;
def int_nvvm_sust_b_1d_v2i8_clamp
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.1d.v2i8.clamp">,
    GCCBuiltin<"__nvvm_sust_b_1d_v2i8_clamp">;
def int_nvvm_sust_b_1d_v2i16_clamp
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.1d.v2i16.clamp">,
    GCCBuiltin<"__nvvm_sust_b_1d_v2i16_clamp">;
def int_nvvm_sust_b_1d_v2i32_clamp
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.sust.b.1d.v2i32.clamp">,
    GCCBuiltin<"__nvvm_sust_b_1d_v2i32_clamp">;
def int_nvvm_sust_b_1d_v2i64_clamp
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i64_ty, llvm_i64_ty], [],
              "llvm.nvvm.sust.b.1d.v2i64.clamp">,
    GCCBuiltin<"__nvvm_sust_b_1d_v2i64_clamp">;
def int_nvvm_sust_b_1d_v4i8_clamp
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i16_ty,
                   llvm_i16_ty, llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.1d.v4i8.clamp">,
    GCCBuiltin<"__nvvm_sust_b_1d_v4i8_clamp">;
def int_nvvm_sust_b_1d_v4i16_clamp
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i16_ty,
                   llvm_i16_ty, llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.1d.v4i16.clamp">,
    GCCBuiltin<"__nvvm_sust_b_1d_v4i16_clamp">;
def int_nvvm_sust_b_1d_v4i32_clamp
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.sust.b.1d.v4i32.clamp">,
    GCCBuiltin<"__nvvm_sust_b_1d_v4i32_clamp">;


def int_nvvm_sust_b_1d_array_i8_clamp
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.1d.array.i8.clamp">,
    GCCBuiltin<"__nvvm_sust_b_1d_array_i8_clamp">;
def int_nvvm_sust_b_1d_array_i16_clamp
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.1d.array.i16.clamp">,
    GCCBuiltin<"__nvvm_sust_b_1d_array_i16_clamp">;
def int_nvvm_sust_b_1d_array_i32_clamp
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.sust.b.1d.array.i32.clamp">,
    GCCBuiltin<"__nvvm_sust_b_1d_array_i32_clamp">;
def int_nvvm_sust_b_1d_array_i64_clamp
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i64_ty], [],
              "llvm.nvvm.sust.b.1d.array.i64.clamp">,
    GCCBuiltin<"__nvvm_sust_b_1d_array_i64_clamp">;
def int_nvvm_sust_b_1d_array_v2i8_clamp
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.1d.array.v2i8.clamp">,
    GCCBuiltin<"__nvvm_sust_b_1d_array_v2i8_clamp">;
def int_nvvm_sust_b_1d_array_v2i16_clamp
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.1d.array.v2i16.clamp">,
    GCCBuiltin<"__nvvm_sust_b_1d_array_v2i16_clamp">;
def int_nvvm_sust_b_1d_array_v2i32_clamp
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.sust.b.1d.array.v2i32.clamp">,
    GCCBuiltin<"__nvvm_sust_b_1d_array_v2i32_clamp">;
def int_nvvm_sust_b_1d_array_v2i64_clamp
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i64_ty, llvm_i64_ty], [],
              "llvm.nvvm.sust.b.1d.array.v2i64.clamp">,
    GCCBuiltin<"__nvvm_sust_b_1d_array_v2i64_clamp">;
def int_nvvm_sust_b_1d_array_v4i8_clamp
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i16_ty,
                   llvm_i16_ty, llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.1d.array.v4i8.clamp">,
    GCCBuiltin<"__nvvm_sust_b_1d_array_v4i8_clamp">;
def int_nvvm_sust_b_1d_array_v4i16_clamp
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i16_ty,
                   llvm_i16_ty, llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.1d.array.v4i16.clamp">,
    GCCBuiltin<"__nvvm_sust_b_1d_array_v4i16_clamp">;
def int_nvvm_sust_b_1d_array_v4i32_clamp
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.sust.b.1d.array.v4i32.clamp">,
    GCCBuiltin<"__nvvm_sust_b_1d_array_v4i32_clamp">;


def int_nvvm_sust_b_2d_i8_clamp
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.2d.i8.clamp">,
    GCCBuiltin<"__nvvm_sust_b_2d_i8_clamp">;
def int_nvvm_sust_b_2d_i16_clamp
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.2d.i16.clamp">,
    GCCBuiltin<"__nvvm_sust_b_2d_i16_clamp">;
def int_nvvm_sust_b_2d_i32_clamp
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.sust.b.2d.i32.clamp">,
    GCCBuiltin<"__nvvm_sust_b_2d_i32_clamp">;
def int_nvvm_sust_b_2d_i64_clamp
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i64_ty], [],
              "llvm.nvvm.sust.b.2d.i64.clamp">,
    GCCBuiltin<"__nvvm_sust_b_2d_i64_clamp">;
def int_nvvm_sust_b_2d_v2i8_clamp
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.2d.v2i8.clamp">,
    GCCBuiltin<"__nvvm_sust_b_2d_v2i8_clamp">;
def int_nvvm_sust_b_2d_v2i16_clamp
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.2d.v2i16.clamp">,
    GCCBuiltin<"__nvvm_sust_b_2d_v2i16_clamp">;
def int_nvvm_sust_b_2d_v2i32_clamp
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.sust.b.2d.v2i32.clamp">,
    GCCBuiltin<"__nvvm_sust_b_2d_v2i32_clamp">;
def int_nvvm_sust_b_2d_v2i64_clamp
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i64_ty, llvm_i64_ty], [],
              "llvm.nvvm.sust.b.2d.v2i64.clamp">,
    GCCBuiltin<"__nvvm_sust_b_2d_v2i64_clamp">;
def int_nvvm_sust_b_2d_v4i8_clamp
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i16_ty,
                   llvm_i16_ty, llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.2d.v4i8.clamp">,
    GCCBuiltin<"__nvvm_sust_b_2d_v4i8_clamp">;
def int_nvvm_sust_b_2d_v4i16_clamp
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i16_ty,
                   llvm_i16_ty, llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.2d.v4i16.clamp">,
    GCCBuiltin<"__nvvm_sust_b_2d_v4i16_clamp">;
def int_nvvm_sust_b_2d_v4i32_clamp
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.sust.b.2d.v4i32.clamp">,
    GCCBuiltin<"__nvvm_sust_b_2d_v4i32_clamp">;


def int_nvvm_sust_b_2d_array_i8_clamp
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i32_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.2d.array.i8.clamp">,
    GCCBuiltin<"__nvvm_sust_b_2d_array_i8_clamp">;
def int_nvvm_sust_b_2d_array_i16_clamp
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i32_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.2d.array.i16.clamp">,
    GCCBuiltin<"__nvvm_sust_b_2d_array_i16_clamp">;
def int_nvvm_sust_b_2d_array_i32_clamp
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.sust.b.2d.array.i32.clamp">,
    GCCBuiltin<"__nvvm_sust_b_2d_array_i32_clamp">;
def int_nvvm_sust_b_2d_array_i64_clamp
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i32_ty, llvm_i64_ty], [],
              "llvm.nvvm.sust.b.2d.array.i64.clamp">,
    GCCBuiltin<"__nvvm_sust_b_2d_array_i64_clamp">;
def int_nvvm_sust_b_2d_array_v2i8_clamp
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.2d.array.v2i8.clamp">,
    GCCBuiltin<"__nvvm_sust_b_2d_array_v2i8_clamp">;
def int_nvvm_sust_b_2d_array_v2i16_clamp
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.2d.array.v2i16.clamp">,
    GCCBuiltin<"__nvvm_sust_b_2d_array_v2i16_clamp">;
def int_nvvm_sust_b_2d_array_v2i32_clamp
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.sust.b.2d.array.v2i32.clamp">,
    GCCBuiltin<"__nvvm_sust_b_2d_array_v2i32_clamp">;
def int_nvvm_sust_b_2d_array_v2i64_clamp
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i64_ty, llvm_i64_ty], [],
              "llvm.nvvm.sust.b.2d.array.v2i64.clamp">,
    GCCBuiltin<"__nvvm_sust_b_2d_array_v2i64_clamp">;
def int_nvvm_sust_b_2d_array_v4i8_clamp
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i16_ty, llvm_i16_ty, llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.2d.array.v4i8.clamp">,
    GCCBuiltin<"__nvvm_sust_b_2d_array_v4i8_clamp">;
def int_nvvm_sust_b_2d_array_v4i16_clamp
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i16_ty, llvm_i16_ty, llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.2d.array.v4i16.clamp">,
    GCCBuiltin<"__nvvm_sust_b_2d_array_v4i16_clamp">;
def int_nvvm_sust_b_2d_array_v4i32_clamp
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.sust.b.2d.array.v4i32.clamp">,
    GCCBuiltin<"__nvvm_sust_b_2d_array_v4i32_clamp">;


def int_nvvm_sust_b_3d_i8_clamp
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i32_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.3d.i8.clamp">,
    GCCBuiltin<"__nvvm_sust_b_3d_i8_clamp">;
def int_nvvm_sust_b_3d_i16_clamp
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i32_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.3d.i16.clamp">,
    GCCBuiltin<"__nvvm_sust_b_3d_i16_clamp">;
def int_nvvm_sust_b_3d_i32_clamp
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.sust.b.3d.i32.clamp">,
    GCCBuiltin<"__nvvm_sust_b_3d_i32_clamp">;
def int_nvvm_sust_b_3d_i64_clamp
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i32_ty, llvm_i64_ty], [],
              "llvm.nvvm.sust.b.3d.i64.clamp">,
    GCCBuiltin<"__nvvm_sust_b_3d_i64_clamp">;
def int_nvvm_sust_b_3d_v2i8_clamp
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.3d.v2i8.clamp">,
    GCCBuiltin<"__nvvm_sust_b_3d_v2i8_clamp">;
def int_nvvm_sust_b_3d_v2i16_clamp
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.3d.v2i16.clamp">,
    GCCBuiltin<"__nvvm_sust_b_3d_v2i16_clamp">;
def int_nvvm_sust_b_3d_v2i32_clamp
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.sust.b.3d.v2i32.clamp">,
    GCCBuiltin<"__nvvm_sust_b_3d_v2i32_clamp">;
def int_nvvm_sust_b_3d_v2i64_clamp
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i64_ty, llvm_i64_ty], [],
              "llvm.nvvm.sust.b.3d.v2i64.clamp">,
    GCCBuiltin<"__nvvm_sust_b_3d_v2i64_clamp">;
def int_nvvm_sust_b_3d_v4i8_clamp
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i16_ty, llvm_i16_ty, llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.3d.v4i8.clamp">,
    GCCBuiltin<"__nvvm_sust_b_3d_v4i8_clamp">;
def int_nvvm_sust_b_3d_v4i16_clamp
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i16_ty, llvm_i16_ty, llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.3d.v4i16.clamp">,
    GCCBuiltin<"__nvvm_sust_b_3d_v4i16_clamp">;
def int_nvvm_sust_b_3d_v4i32_clamp
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.sust.b.3d.v4i32.clamp">,
    GCCBuiltin<"__nvvm_sust_b_3d_v4i32_clamp">;


// .trap variant
def int_nvvm_sust_b_1d_i8_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.1d.i8.trap">,
    GCCBuiltin<"__nvvm_sust_b_1d_i8_trap">;
def int_nvvm_sust_b_1d_i16_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.1d.i16.trap">,
    GCCBuiltin<"__nvvm_sust_b_1d_i16_trap">;
def int_nvvm_sust_b_1d_i32_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.sust.b.1d.i32.trap">,
    GCCBuiltin<"__nvvm_sust_b_1d_i32_trap">;
def int_nvvm_sust_b_1d_i64_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i64_ty], [],
              "llvm.nvvm.sust.b.1d.i64.trap">,
    GCCBuiltin<"__nvvm_sust_b_1d_i64_trap">;
def int_nvvm_sust_b_1d_v2i8_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.1d.v2i8.trap">,
    GCCBuiltin<"__nvvm_sust_b_1d_v2i8_trap">;
def int_nvvm_sust_b_1d_v2i16_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.1d.v2i16.trap">,
    GCCBuiltin<"__nvvm_sust_b_1d_v2i16_trap">;
def int_nvvm_sust_b_1d_v2i32_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.sust.b.1d.v2i32.trap">,
    GCCBuiltin<"__nvvm_sust_b_1d_v2i32_trap">;
def int_nvvm_sust_b_1d_v2i64_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i64_ty, llvm_i64_ty], [],
              "llvm.nvvm.sust.b.1d.v2i64.trap">,
    GCCBuiltin<"__nvvm_sust_b_1d_v2i64_trap">;
def int_nvvm_sust_b_1d_v4i8_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i16_ty,
                   llvm_i16_ty, llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.1d.v4i8.trap">,
    GCCBuiltin<"__nvvm_sust_b_1d_v4i8_trap">;
def int_nvvm_sust_b_1d_v4i16_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i16_ty,
                   llvm_i16_ty, llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.1d.v4i16.trap">,
    GCCBuiltin<"__nvvm_sust_b_1d_v4i16_trap">;
def int_nvvm_sust_b_1d_v4i32_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.sust.b.1d.v4i32.trap">,
    GCCBuiltin<"__nvvm_sust_b_1d_v4i32_trap">;


def int_nvvm_sust_b_1d_array_i8_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.1d.array.i8.trap">,
    GCCBuiltin<"__nvvm_sust_b_1d_array_i8_trap">;
def int_nvvm_sust_b_1d_array_i16_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.1d.array.i16.trap">,
    GCCBuiltin<"__nvvm_sust_b_1d_array_i16_trap">;
def int_nvvm_sust_b_1d_array_i32_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.sust.b.1d.array.i32.trap">,
    GCCBuiltin<"__nvvm_sust_b_1d_array_i32_trap">;
def int_nvvm_sust_b_1d_array_i64_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i64_ty], [],
              "llvm.nvvm.sust.b.1d.array.i64.trap">,
    GCCBuiltin<"__nvvm_sust_b_1d_array_i64_trap">;
def int_nvvm_sust_b_1d_array_v2i8_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.1d.array.v2i8.trap">,
    GCCBuiltin<"__nvvm_sust_b_1d_array_v2i8_trap">;
def int_nvvm_sust_b_1d_array_v2i16_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.1d.array.v2i16.trap">,
    GCCBuiltin<"__nvvm_sust_b_1d_array_v2i16_trap">;
def int_nvvm_sust_b_1d_array_v2i32_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.sust.b.1d.array.v2i32.trap">,
    GCCBuiltin<"__nvvm_sust_b_1d_array_v2i32_trap">;
def int_nvvm_sust_b_1d_array_v2i64_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i64_ty, llvm_i64_ty], [],
              "llvm.nvvm.sust.b.1d.array.v2i64.trap">,
    GCCBuiltin<"__nvvm_sust_b_1d_array_v2i64_trap">;
def int_nvvm_sust_b_1d_array_v4i8_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i16_ty,
                   llvm_i16_ty, llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.1d.array.v4i8.trap">,
    GCCBuiltin<"__nvvm_sust_b_1d_array_v4i8_trap">;
def int_nvvm_sust_b_1d_array_v4i16_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i16_ty,
                   llvm_i16_ty, llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.1d.array.v4i16.trap">,
    GCCBuiltin<"__nvvm_sust_b_1d_array_v4i16_trap">;
def int_nvvm_sust_b_1d_array_v4i32_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.sust.b.1d.array.v4i32.trap">,
    GCCBuiltin<"__nvvm_sust_b_1d_array_v4i32_trap">;


def int_nvvm_sust_b_2d_i8_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.2d.i8.trap">,
    GCCBuiltin<"__nvvm_sust_b_2d_i8_trap">;
def int_nvvm_sust_b_2d_i16_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.2d.i16.trap">,
    GCCBuiltin<"__nvvm_sust_b_2d_i16_trap">;
def int_nvvm_sust_b_2d_i32_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.sust.b.2d.i32.trap">,
    GCCBuiltin<"__nvvm_sust_b_2d_i32_trap">;
def int_nvvm_sust_b_2d_i64_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i64_ty], [],
              "llvm.nvvm.sust.b.2d.i64.trap">,
    GCCBuiltin<"__nvvm_sust_b_2d_i64_trap">;
def int_nvvm_sust_b_2d_v2i8_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.2d.v2i8.trap">,
    GCCBuiltin<"__nvvm_sust_b_2d_v2i8_trap">;
def int_nvvm_sust_b_2d_v2i16_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.2d.v2i16.trap">,
    GCCBuiltin<"__nvvm_sust_b_2d_v2i16_trap">;
def int_nvvm_sust_b_2d_v2i32_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.sust.b.2d.v2i32.trap">,
    GCCBuiltin<"__nvvm_sust_b_2d_v2i32_trap">;
def int_nvvm_sust_b_2d_v2i64_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i64_ty, llvm_i64_ty], [],
              "llvm.nvvm.sust.b.2d.v2i64.trap">,
    GCCBuiltin<"__nvvm_sust_b_2d_v2i64_trap">;
def int_nvvm_sust_b_2d_v4i8_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i16_ty,
                   llvm_i16_ty, llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.2d.v4i8.trap">,
    GCCBuiltin<"__nvvm_sust_b_2d_v4i8_trap">;
def int_nvvm_sust_b_2d_v4i16_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i16_ty,
                   llvm_i16_ty, llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.2d.v4i16.trap">,
    GCCBuiltin<"__nvvm_sust_b_2d_v4i16_trap">;
def int_nvvm_sust_b_2d_v4i32_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.sust.b.2d.v4i32.trap">,
    GCCBuiltin<"__nvvm_sust_b_2d_v4i32_trap">;


def int_nvvm_sust_b_2d_array_i8_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i32_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.2d.array.i8.trap">,
    GCCBuiltin<"__nvvm_sust_b_2d_array_i8_trap">;
def int_nvvm_sust_b_2d_array_i16_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i32_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.2d.array.i16.trap">,
    GCCBuiltin<"__nvvm_sust_b_2d_array_i16_trap">;
def int_nvvm_sust_b_2d_array_i32_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.sust.b.2d.array.i32.trap">,
    GCCBuiltin<"__nvvm_sust_b_2d_array_i32_trap">;
def int_nvvm_sust_b_2d_array_i64_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i32_ty, llvm_i64_ty], [],
              "llvm.nvvm.sust.b.2d.array.i64.trap">,
    GCCBuiltin<"__nvvm_sust_b_2d_array_i64_trap">;
def int_nvvm_sust_b_2d_array_v2i8_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.2d.array.v2i8.trap">,
    GCCBuiltin<"__nvvm_sust_b_2d_array_v2i8_trap">;
def int_nvvm_sust_b_2d_array_v2i16_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.2d.array.v2i16.trap">,
    GCCBuiltin<"__nvvm_sust_b_2d_array_v2i16_trap">;
def int_nvvm_sust_b_2d_array_v2i32_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.sust.b.2d.array.v2i32.trap">,
    GCCBuiltin<"__nvvm_sust_b_2d_array_v2i32_trap">;
def int_nvvm_sust_b_2d_array_v2i64_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i64_ty, llvm_i64_ty], [],
              "llvm.nvvm.sust.b.2d.array.v2i64.trap">,
    GCCBuiltin<"__nvvm_sust_b_2d_array_v2i64_trap">;
def int_nvvm_sust_b_2d_array_v4i8_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i16_ty, llvm_i16_ty, llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.2d.array.v4i8.trap">,
    GCCBuiltin<"__nvvm_sust_b_2d_array_v4i8_trap">;
def int_nvvm_sust_b_2d_array_v4i16_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i16_ty, llvm_i16_ty, llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.2d.array.v4i16.trap">,
    GCCBuiltin<"__nvvm_sust_b_2d_array_v4i16_trap">;
def int_nvvm_sust_b_2d_array_v4i32_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.sust.b.2d.array.v4i32.trap">,
    GCCBuiltin<"__nvvm_sust_b_2d_array_v4i32_trap">;


def int_nvvm_sust_b_3d_i8_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i32_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.3d.i8.trap">,
    GCCBuiltin<"__nvvm_sust_b_3d_i8_trap">;
def int_nvvm_sust_b_3d_i16_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i32_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.3d.i16.trap">,
    GCCBuiltin<"__nvvm_sust_b_3d_i16_trap">;
def int_nvvm_sust_b_3d_i32_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.sust.b.3d.i32.trap">,
    GCCBuiltin<"__nvvm_sust_b_3d_i32_trap">;
def int_nvvm_sust_b_3d_i64_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i32_ty, llvm_i64_ty], [],
              "llvm.nvvm.sust.b.3d.i64.trap">,
    GCCBuiltin<"__nvvm_sust_b_3d_i64_trap">;
def int_nvvm_sust_b_3d_v2i8_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.3d.v2i8.trap">,
    GCCBuiltin<"__nvvm_sust_b_3d_v2i8_trap">;
def int_nvvm_sust_b_3d_v2i16_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.3d.v2i16.trap">,
    GCCBuiltin<"__nvvm_sust_b_3d_v2i16_trap">;
def int_nvvm_sust_b_3d_v2i32_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.sust.b.3d.v2i32.trap">,
    GCCBuiltin<"__nvvm_sust_b_3d_v2i32_trap">;
def int_nvvm_sust_b_3d_v2i64_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i64_ty, llvm_i64_ty], [],
              "llvm.nvvm.sust.b.3d.v2i64.trap">,
    GCCBuiltin<"__nvvm_sust_b_3d_v2i64_trap">;
def int_nvvm_sust_b_3d_v4i8_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i16_ty, llvm_i16_ty, llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.3d.v4i8.trap">,
    GCCBuiltin<"__nvvm_sust_b_3d_v4i8_trap">;
def int_nvvm_sust_b_3d_v4i16_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i16_ty, llvm_i16_ty, llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.3d.v4i16.trap">,
    GCCBuiltin<"__nvvm_sust_b_3d_v4i16_trap">;
def int_nvvm_sust_b_3d_v4i32_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.sust.b.3d.v4i32.trap">,
    GCCBuiltin<"__nvvm_sust_b_3d_v4i32_trap">;


// .zero variant
def int_nvvm_sust_b_1d_i8_zero
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.1d.i8.zero">,
    GCCBuiltin<"__nvvm_sust_b_1d_i8_zero">;
def int_nvvm_sust_b_1d_i16_zero
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.1d.i16.zero">,
    GCCBuiltin<"__nvvm_sust_b_1d_i16_zero">;
def int_nvvm_sust_b_1d_i32_zero
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.sust.b.1d.i32.zero">,
    GCCBuiltin<"__nvvm_sust_b_1d_i32_zero">;
def int_nvvm_sust_b_1d_i64_zero
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i64_ty], [],
              "llvm.nvvm.sust.b.1d.i64.zero">,
    GCCBuiltin<"__nvvm_sust_b_1d_i64_zero">;
def int_nvvm_sust_b_1d_v2i8_zero
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.1d.v2i8.zero">,
    GCCBuiltin<"__nvvm_sust_b_1d_v2i8_zero">;
def int_nvvm_sust_b_1d_v2i16_zero
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.1d.v2i16.zero">,
    GCCBuiltin<"__nvvm_sust_b_1d_v2i16_zero">;
def int_nvvm_sust_b_1d_v2i32_zero
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.sust.b.1d.v2i32.zero">,
    GCCBuiltin<"__nvvm_sust_b_1d_v2i32_zero">;
def int_nvvm_sust_b_1d_v2i64_zero
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i64_ty, llvm_i64_ty], [],
              "llvm.nvvm.sust.b.1d.v2i64.zero">,
    GCCBuiltin<"__nvvm_sust_b_1d_v2i64_zero">;
def int_nvvm_sust_b_1d_v4i8_zero
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i16_ty,
                   llvm_i16_ty, llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.1d.v4i8.zero">,
    GCCBuiltin<"__nvvm_sust_b_1d_v4i8_zero">;
def int_nvvm_sust_b_1d_v4i16_zero
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i16_ty,
                   llvm_i16_ty, llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.1d.v4i16.zero">,
    GCCBuiltin<"__nvvm_sust_b_1d_v4i16_zero">;
def int_nvvm_sust_b_1d_v4i32_zero
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.sust.b.1d.v4i32.zero">,
    GCCBuiltin<"__nvvm_sust_b_1d_v4i32_zero">;


def int_nvvm_sust_b_1d_array_i8_zero
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.1d.array.i8.zero">,
    GCCBuiltin<"__nvvm_sust_b_1d_array_i8_zero">;
def int_nvvm_sust_b_1d_array_i16_zero
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.1d.array.i16.zero">,
    GCCBuiltin<"__nvvm_sust_b_1d_array_i16_zero">;
def int_nvvm_sust_b_1d_array_i32_zero
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.sust.b.1d.array.i32.zero">,
    GCCBuiltin<"__nvvm_sust_b_1d_array_i32_zero">;
def int_nvvm_sust_b_1d_array_i64_zero
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i64_ty], [],
              "llvm.nvvm.sust.b.1d.array.i64.zero">,
    GCCBuiltin<"__nvvm_sust_b_1d_array_i64_zero">;
def int_nvvm_sust_b_1d_array_v2i8_zero
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.1d.array.v2i8.zero">,
    GCCBuiltin<"__nvvm_sust_b_1d_array_v2i8_zero">;
def int_nvvm_sust_b_1d_array_v2i16_zero
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.1d.array.v2i16.zero">,
    GCCBuiltin<"__nvvm_sust_b_1d_array_v2i16_zero">;
def int_nvvm_sust_b_1d_array_v2i32_zero
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.sust.b.1d.array.v2i32.zero">,
    GCCBuiltin<"__nvvm_sust_b_1d_array_v2i32_zero">;
def int_nvvm_sust_b_1d_array_v2i64_zero
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i64_ty, llvm_i64_ty], [],
              "llvm.nvvm.sust.b.1d.array.v2i64.zero">,
    GCCBuiltin<"__nvvm_sust_b_1d_array_v2i64_zero">;
def int_nvvm_sust_b_1d_array_v4i8_zero
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i16_ty,
                   llvm_i16_ty, llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.1d.array.v4i8.zero">,
    GCCBuiltin<"__nvvm_sust_b_1d_array_v4i8_zero">;
def int_nvvm_sust_b_1d_array_v4i16_zero
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i16_ty,
                   llvm_i16_ty, llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.1d.array.v4i16.zero">,
    GCCBuiltin<"__nvvm_sust_b_1d_array_v4i16_zero">;
def int_nvvm_sust_b_1d_array_v4i32_zero
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.sust.b.1d.array.v4i32.zero">,
    GCCBuiltin<"__nvvm_sust_b_1d_array_v4i32_zero">;


def int_nvvm_sust_b_2d_i8_zero
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.2d.i8.zero">,
    GCCBuiltin<"__nvvm_sust_b_2d_i8_zero">;
def int_nvvm_sust_b_2d_i16_zero
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.2d.i16.zero">,
    GCCBuiltin<"__nvvm_sust_b_2d_i16_zero">;
def int_nvvm_sust_b_2d_i32_zero
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.sust.b.2d.i32.zero">,
    GCCBuiltin<"__nvvm_sust_b_2d_i32_zero">;
def int_nvvm_sust_b_2d_i64_zero
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i64_ty], [],
              "llvm.nvvm.sust.b.2d.i64.zero">,
    GCCBuiltin<"__nvvm_sust_b_2d_i64_zero">;
def int_nvvm_sust_b_2d_v2i8_zero
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.2d.v2i8.zero">,
    GCCBuiltin<"__nvvm_sust_b_2d_v2i8_zero">;
def int_nvvm_sust_b_2d_v2i16_zero
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.2d.v2i16.zero">,
    GCCBuiltin<"__nvvm_sust_b_2d_v2i16_zero">;
def int_nvvm_sust_b_2d_v2i32_zero
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.sust.b.2d.v2i32.zero">,
    GCCBuiltin<"__nvvm_sust_b_2d_v2i32_zero">;
def int_nvvm_sust_b_2d_v2i64_zero
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i64_ty, llvm_i64_ty], [],
              "llvm.nvvm.sust.b.2d.v2i64.zero">,
    GCCBuiltin<"__nvvm_sust_b_2d_v2i64_zero">;
def int_nvvm_sust_b_2d_v4i8_zero
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i16_ty,
                   llvm_i16_ty, llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.2d.v4i8.zero">,
    GCCBuiltin<"__nvvm_sust_b_2d_v4i8_zero">;
def int_nvvm_sust_b_2d_v4i16_zero
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i16_ty,
                   llvm_i16_ty, llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.2d.v4i16.zero">,
    GCCBuiltin<"__nvvm_sust_b_2d_v4i16_zero">;
def int_nvvm_sust_b_2d_v4i32_zero
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.sust.b.2d.v4i32.zero">,
    GCCBuiltin<"__nvvm_sust_b_2d_v4i32_zero">;


def int_nvvm_sust_b_2d_array_i8_zero
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i32_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.2d.array.i8.zero">,
    GCCBuiltin<"__nvvm_sust_b_2d_array_i8_zero">;
def int_nvvm_sust_b_2d_array_i16_zero
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i32_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.2d.array.i16.zero">,
    GCCBuiltin<"__nvvm_sust_b_2d_array_i16_zero">;
def int_nvvm_sust_b_2d_array_i32_zero
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.sust.b.2d.array.i32.zero">,
    GCCBuiltin<"__nvvm_sust_b_2d_array_i32_zero">;
def int_nvvm_sust_b_2d_array_i64_zero
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i32_ty, llvm_i64_ty], [],
              "llvm.nvvm.sust.b.2d.array.i64.zero">,
    GCCBuiltin<"__nvvm_sust_b_2d_array_i64_zero">;
def int_nvvm_sust_b_2d_array_v2i8_zero
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.2d.array.v2i8.zero">,
    GCCBuiltin<"__nvvm_sust_b_2d_array_v2i8_zero">;
def int_nvvm_sust_b_2d_array_v2i16_zero
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.2d.array.v2i16.zero">,
    GCCBuiltin<"__nvvm_sust_b_2d_array_v2i16_zero">;
def int_nvvm_sust_b_2d_array_v2i32_zero
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.sust.b.2d.array.v2i32.zero">,
    GCCBuiltin<"__nvvm_sust_b_2d_array_v2i32_zero">;
def int_nvvm_sust_b_2d_array_v2i64_zero
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i64_ty, llvm_i64_ty], [],
              "llvm.nvvm.sust.b.2d.array.v2i64.zero">,
    GCCBuiltin<"__nvvm_sust_b_2d_array_v2i64_zero">;
def int_nvvm_sust_b_2d_array_v4i8_zero
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i16_ty, llvm_i16_ty, llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.2d.array.v4i8.zero">,
    GCCBuiltin<"__nvvm_sust_b_2d_array_v4i8_zero">;
def int_nvvm_sust_b_2d_array_v4i16_zero
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i16_ty, llvm_i16_ty, llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.2d.array.v4i16.zero">,
    GCCBuiltin<"__nvvm_sust_b_2d_array_v4i16_zero">;
def int_nvvm_sust_b_2d_array_v4i32_zero
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.sust.b.2d.array.v4i32.zero">,
    GCCBuiltin<"__nvvm_sust_b_2d_array_v4i32_zero">;


def int_nvvm_sust_b_3d_i8_zero
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i32_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.3d.i8.zero">,
    GCCBuiltin<"__nvvm_sust_b_3d_i8_zero">;
def int_nvvm_sust_b_3d_i16_zero
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i32_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.3d.i16.zero">,
    GCCBuiltin<"__nvvm_sust_b_3d_i16_zero">;
def int_nvvm_sust_b_3d_i32_zero
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.sust.b.3d.i32.zero">,
    GCCBuiltin<"__nvvm_sust_b_3d_i32_zero">;
def int_nvvm_sust_b_3d_i64_zero
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i32_ty, llvm_i64_ty], [],
              "llvm.nvvm.sust.b.3d.i64.zero">,
    GCCBuiltin<"__nvvm_sust_b_3d_i64_zero">;
def int_nvvm_sust_b_3d_v2i8_zero
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.3d.v2i8.zero">,
    GCCBuiltin<"__nvvm_sust_b_3d_v2i8_zero">;
def int_nvvm_sust_b_3d_v2i16_zero
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.3d.v2i16.zero">,
    GCCBuiltin<"__nvvm_sust_b_3d_v2i16_zero">;
def int_nvvm_sust_b_3d_v2i32_zero
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.sust.b.3d.v2i32.zero">,
    GCCBuiltin<"__nvvm_sust_b_3d_v2i32_zero">;
def int_nvvm_sust_b_3d_v2i64_zero
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i64_ty, llvm_i64_ty], [],
              "llvm.nvvm.sust.b.3d.v2i64.zero">,
    GCCBuiltin<"__nvvm_sust_b_3d_v2i64_zero">;
def int_nvvm_sust_b_3d_v4i8_zero
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i16_ty, llvm_i16_ty, llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.3d.v4i8.zero">,
    GCCBuiltin<"__nvvm_sust_b_3d_v4i8_zero">;
def int_nvvm_sust_b_3d_v4i16_zero
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i16_ty, llvm_i16_ty, llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.b.3d.v4i16.zero">,
    GCCBuiltin<"__nvvm_sust_b_3d_v4i16_zero">;
def int_nvvm_sust_b_3d_v4i32_zero
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.sust.b.3d.v4i32.zero">,
    GCCBuiltin<"__nvvm_sust_b_3d_v4i32_zero">;



// Formatted

def int_nvvm_sust_p_1d_i8_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.p.1d.i8.trap">,
    GCCBuiltin<"__nvvm_sust_p_1d_i8_trap">;
def int_nvvm_sust_p_1d_i16_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.p.1d.i16.trap">,
    GCCBuiltin<"__nvvm_sust_p_1d_i16_trap">;
def int_nvvm_sust_p_1d_i32_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.sust.p.1d.i32.trap">,
    GCCBuiltin<"__nvvm_sust_p_1d_i32_trap">;
def int_nvvm_sust_p_1d_v2i8_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.p.1d.v2i8.trap">,
    GCCBuiltin<"__nvvm_sust_p_1d_v2i8_trap">;
def int_nvvm_sust_p_1d_v2i16_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.p.1d.v2i16.trap">,
    GCCBuiltin<"__nvvm_sust_p_1d_v2i16_trap">;
def int_nvvm_sust_p_1d_v2i32_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.sust.p.1d.v2i32.trap">,
    GCCBuiltin<"__nvvm_sust_p_1d_v2i32_trap">;
def int_nvvm_sust_p_1d_v4i8_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i16_ty,
                   llvm_i16_ty, llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.p.1d.v4i8.trap">,
    GCCBuiltin<"__nvvm_sust_p_1d_v4i8_trap">;
def int_nvvm_sust_p_1d_v4i16_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i16_ty,
                   llvm_i16_ty, llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.p.1d.v4i16.trap">,
    GCCBuiltin<"__nvvm_sust_p_1d_v4i16_trap">;
def int_nvvm_sust_p_1d_v4i32_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.sust.p.1d.v4i32.trap">,
    GCCBuiltin<"__nvvm_sust_p_1d_v4i32_trap">;


def int_nvvm_sust_p_1d_array_i8_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.p.1d.array.i8.trap">,
    GCCBuiltin<"__nvvm_sust_p_1d_array_i8_trap">;
def int_nvvm_sust_p_1d_array_i16_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.p.1d.array.i16.trap">,
    GCCBuiltin<"__nvvm_sust_p_1d_array_i16_trap">;
def int_nvvm_sust_p_1d_array_i32_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.sust.p.1d.array.i32.trap">,
    GCCBuiltin<"__nvvm_sust_p_1d_array_i32_trap">;
def int_nvvm_sust_p_1d_array_v2i8_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.p.1d.array.v2i8.trap">,
    GCCBuiltin<"__nvvm_sust_p_1d_array_v2i8_trap">;
def int_nvvm_sust_p_1d_array_v2i16_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.p.1d.array.v2i16.trap">,
    GCCBuiltin<"__nvvm_sust_p_1d_array_v2i16_trap">;
def int_nvvm_sust_p_1d_array_v2i32_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.sust.p.1d.array.v2i32.trap">,
    GCCBuiltin<"__nvvm_sust_p_1d_array_v2i32_trap">;
def int_nvvm_sust_p_1d_array_v4i8_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i16_ty,
                   llvm_i16_ty, llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.p.1d.array.v4i8.trap">,
    GCCBuiltin<"__nvvm_sust_p_1d_array_v4i8_trap">;
def int_nvvm_sust_p_1d_array_v4i16_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i16_ty,
                   llvm_i16_ty, llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.p.1d.array.v4i16.trap">,
    GCCBuiltin<"__nvvm_sust_p_1d_array_v4i16_trap">;
def int_nvvm_sust_p_1d_array_v4i32_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.sust.p.1d.array.v4i32.trap">,
    GCCBuiltin<"__nvvm_sust_p_1d_array_v4i32_trap">;


def int_nvvm_sust_p_2d_i8_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.p.2d.i8.trap">,
    GCCBuiltin<"__nvvm_sust_p_2d_i8_trap">;
def int_nvvm_sust_p_2d_i16_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.p.2d.i16.trap">,
    GCCBuiltin<"__nvvm_sust_p_2d_i16_trap">;
def int_nvvm_sust_p_2d_i32_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.sust.p.2d.i32.trap">,
    GCCBuiltin<"__nvvm_sust_p_2d_i32_trap">;
def int_nvvm_sust_p_2d_v2i8_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.p.2d.v2i8.trap">,
    GCCBuiltin<"__nvvm_sust_p_2d_v2i8_trap">;
def int_nvvm_sust_p_2d_v2i16_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.p.2d.v2i16.trap">,
    GCCBuiltin<"__nvvm_sust_p_2d_v2i16_trap">;
def int_nvvm_sust_p_2d_v2i32_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.sust.p.2d.v2i32.trap">,
    GCCBuiltin<"__nvvm_sust_p_2d_v2i32_trap">;
def int_nvvm_sust_p_2d_v4i8_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i16_ty,
                   llvm_i16_ty, llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.p.2d.v4i8.trap">,
    GCCBuiltin<"__nvvm_sust_p_2d_v4i8_trap">;
def int_nvvm_sust_p_2d_v4i16_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i16_ty,
                   llvm_i16_ty, llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.p.2d.v4i16.trap">,
    GCCBuiltin<"__nvvm_sust_p_2d_v4i16_trap">;
def int_nvvm_sust_p_2d_v4i32_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.sust.p.2d.v4i32.trap">,
    GCCBuiltin<"__nvvm_sust_p_2d_v4i32_trap">;


def int_nvvm_sust_p_2d_array_i8_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i32_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.p.2d.array.i8.trap">,
    GCCBuiltin<"__nvvm_sust_p_2d_array_i8_trap">;
def int_nvvm_sust_p_2d_array_i16_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i32_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.p.2d.array.i16.trap">,
    GCCBuiltin<"__nvvm_sust_p_2d_array_i16_trap">;
def int_nvvm_sust_p_2d_array_i32_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.sust.p.2d.array.i32.trap">,
    GCCBuiltin<"__nvvm_sust_p_2d_array_i32_trap">;
def int_nvvm_sust_p_2d_array_v2i8_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.p.2d.array.v2i8.trap">,
    GCCBuiltin<"__nvvm_sust_p_2d_array_v2i8_trap">;
def int_nvvm_sust_p_2d_array_v2i16_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.p.2d.array.v2i16.trap">,
    GCCBuiltin<"__nvvm_sust_p_2d_array_v2i16_trap">;
def int_nvvm_sust_p_2d_array_v2i32_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.sust.p.2d.array.v2i32.trap">,
    GCCBuiltin<"__nvvm_sust_p_2d_array_v2i32_trap">;
def int_nvvm_sust_p_2d_array_v4i8_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i16_ty, llvm_i16_ty, llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.p.2d.array.v4i8.trap">,
    GCCBuiltin<"__nvvm_sust_p_2d_array_v4i8_trap">;
def int_nvvm_sust_p_2d_array_v4i16_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i16_ty, llvm_i16_ty, llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.p.2d.array.v4i16.trap">,
    GCCBuiltin<"__nvvm_sust_p_2d_array_v4i16_trap">;
def int_nvvm_sust_p_2d_array_v4i32_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.sust.p.2d.array.v4i32.trap">,
    GCCBuiltin<"__nvvm_sust_p_2d_array_v4i32_trap">;


def int_nvvm_sust_p_3d_i8_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i32_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.p.3d.i8.trap">,
    GCCBuiltin<"__nvvm_sust_p_3d_i8_trap">;
def int_nvvm_sust_p_3d_i16_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i32_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.p.3d.i16.trap">,
    GCCBuiltin<"__nvvm_sust_p_3d_i16_trap">;
def int_nvvm_sust_p_3d_i32_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.sust.p.3d.i32.trap">,
    GCCBuiltin<"__nvvm_sust_p_3d_i32_trap">;
def int_nvvm_sust_p_3d_v2i8_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.p.3d.v2i8.trap">,
    GCCBuiltin<"__nvvm_sust_p_3d_v2i8_trap">;
def int_nvvm_sust_p_3d_v2i16_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.p.3d.v2i16.trap">,
    GCCBuiltin<"__nvvm_sust_p_3d_v2i16_trap">;
def int_nvvm_sust_p_3d_v2i32_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.sust.p.3d.v2i32.trap">,
    GCCBuiltin<"__nvvm_sust_p_3d_v2i32_trap">;
def int_nvvm_sust_p_3d_v4i8_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i16_ty, llvm_i16_ty, llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.p.3d.v4i8.trap">,
    GCCBuiltin<"__nvvm_sust_p_3d_v4i8_trap">;
def int_nvvm_sust_p_3d_v4i16_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i16_ty, llvm_i16_ty, llvm_i16_ty, llvm_i16_ty], [],
              "llvm.nvvm.sust.p.3d.v4i16.trap">,
    GCCBuiltin<"__nvvm_sust_p_3d_v4i16_trap">;
def int_nvvm_sust_p_3d_v4i32_trap
  : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                   llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [],
              "llvm.nvvm.sust.p.3d.v4i32.trap">,
    GCCBuiltin<"__nvvm_sust_p_3d_v4i32_trap">;


def int_nvvm_rotate_b32
  : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty],
              [IntrNoMem], "llvm.nvvm.rotate.b32">,
              GCCBuiltin<"__nvvm_rotate_b32">;

def int_nvvm_rotate_b64
  :Intrinsic<[llvm_i64_ty], [llvm_i64_ty, llvm_i32_ty],
             [IntrNoMem], "llvm.nvvm.rotate.b64">,
             GCCBuiltin<"__nvvm_rotate_b64">;

def int_nvvm_rotate_right_b64
  : Intrinsic<[llvm_i64_ty], [llvm_i64_ty, llvm_i32_ty],
              [IntrNoMem], "llvm.nvvm.rotate.right.b64">,
              GCCBuiltin<"__nvvm_rotate_right_b64">;

def int_nvvm_swap_lo_hi_b64
  : Intrinsic<[llvm_i64_ty], [llvm_i64_ty],
              [IntrNoMem], "llvm.nvvm.swap.lo.hi.b64">,
              GCCBuiltin<"__nvvm_swap_lo_hi_b64">;


// Old PTX back-end intrinsics retained here for backwards-compatibility

multiclass PTXReadSpecialRegisterIntrinsic_v4i32<string prefix> {
// FIXME: Do we need the 128-bit integer type version?
//    def _r64   : Intrinsic<[llvm_i128_ty],   [], [IntrNoMem]>;

// FIXME: Enable this once v4i32 support is enabled in back-end.
//    def _v4i16 : Intrinsic<[llvm_v4i32_ty], [], [IntrNoMem]>;

  def _x     : Intrinsic<[llvm_i32_ty], [], [IntrNoMem]>,
               GCCBuiltin<!strconcat(prefix, "_x")>;
  def _y     : Intrinsic<[llvm_i32_ty], [], [IntrNoMem]>,
               GCCBuiltin<!strconcat(prefix, "_y")>;
  def _z     : Intrinsic<[llvm_i32_ty], [], [IntrNoMem]>,
               GCCBuiltin<!strconcat(prefix, "_z")>;
  def _w     : Intrinsic<[llvm_i32_ty], [], [IntrNoMem]>,
               GCCBuiltin<!strconcat(prefix, "_w")>;
}

class PTXReadSpecialRegisterIntrinsic_r32<string name>
  : Intrinsic<[llvm_i32_ty], [], [IntrNoMem]>,
    GCCBuiltin<name>;

class PTXReadSpecialRegisterIntrinsic_r64<string name>
  : Intrinsic<[llvm_i64_ty], [], [IntrNoMem]>,
    GCCBuiltin<name>;

defm int_ptx_read_tid        : PTXReadSpecialRegisterIntrinsic_v4i32
                               <"__builtin_ptx_read_tid">;
defm int_ptx_read_ntid       : PTXReadSpecialRegisterIntrinsic_v4i32
                               <"__builtin_ptx_read_ntid">;

def int_ptx_read_laneid      : PTXReadSpecialRegisterIntrinsic_r32
                               <"__builtin_ptx_read_laneid">;
def int_ptx_read_warpid      : PTXReadSpecialRegisterIntrinsic_r32
                               <"__builtin_ptx_read_warpid">;
def int_ptx_read_nwarpid     : PTXReadSpecialRegisterIntrinsic_r32
                               <"__builtin_ptx_read_nwarpid">;

defm int_ptx_read_ctaid      : PTXReadSpecialRegisterIntrinsic_v4i32
                               <"__builtin_ptx_read_ctaid">;
defm int_ptx_read_nctaid     : PTXReadSpecialRegisterIntrinsic_v4i32
                               <"__builtin_ptx_read_nctaid">;

def int_ptx_read_smid        : PTXReadSpecialRegisterIntrinsic_r32
                               <"__builtin_ptx_read_smid">;
def int_ptx_read_nsmid       : PTXReadSpecialRegisterIntrinsic_r32
                               <"__builtin_ptx_read_nsmid">;
def int_ptx_read_gridid      : PTXReadSpecialRegisterIntrinsic_r32
                               <"__builtin_ptx_read_gridid">;

def int_ptx_read_lanemask_eq : PTXReadSpecialRegisterIntrinsic_r32
                               <"__builtin_ptx_read_lanemask_eq">;
def int_ptx_read_lanemask_le : PTXReadSpecialRegisterIntrinsic_r32
                               <"__builtin_ptx_read_lanemask_le">;
def int_ptx_read_lanemask_lt : PTXReadSpecialRegisterIntrinsic_r32
                               <"__builtin_ptx_read_lanemask_lt">;
def int_ptx_read_lanemask_ge : PTXReadSpecialRegisterIntrinsic_r32
                               <"__builtin_ptx_read_lanemask_ge">;
def int_ptx_read_lanemask_gt : PTXReadSpecialRegisterIntrinsic_r32
                               <"__builtin_ptx_read_lanemask_gt">;

def int_ptx_read_clock       : PTXReadSpecialRegisterIntrinsic_r32
                               <"__builtin_ptx_read_clock">;
def int_ptx_read_clock64     : PTXReadSpecialRegisterIntrinsic_r64
                               <"__builtin_ptx_read_clock64">;

def int_ptx_read_pm0         : PTXReadSpecialRegisterIntrinsic_r32
                               <"__builtin_ptx_read_pm0">;
def int_ptx_read_pm1         : PTXReadSpecialRegisterIntrinsic_r32
                               <"__builtin_ptx_read_pm1">;
def int_ptx_read_pm2         : PTXReadSpecialRegisterIntrinsic_r32
                               <"__builtin_ptx_read_pm2">;
def int_ptx_read_pm3         : PTXReadSpecialRegisterIntrinsic_r32
                               <"__builtin_ptx_read_pm3">;

def int_ptx_bar_sync : Intrinsic<[], [llvm_i32_ty], []>,
                       GCCBuiltin<"__builtin_ptx_bar_sync">;
