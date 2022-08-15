//===- IntrinsicsAMDGPU.td - Defines AMDGPU intrinsics -----*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines all of the R600-specific intrinsics.
//
//===----------------------------------------------------------------------===//

let TargetPrefix = "r600" in {

class R600ReadPreloadRegisterIntrinsic<string name>
  : Intrinsic<[llvm_i32_ty], [], [IntrNoMem]>,
    GCCBuiltin<name>;

multiclass R600ReadPreloadRegisterIntrinsic_xyz<string prefix> {
  def _x : R600ReadPreloadRegisterIntrinsic<!strconcat(prefix, "_x")>;
  def _y : R600ReadPreloadRegisterIntrinsic<!strconcat(prefix, "_y")>;
  def _z : R600ReadPreloadRegisterIntrinsic<!strconcat(prefix, "_z")>;
}

defm int_r600_read_global_size : R600ReadPreloadRegisterIntrinsic_xyz <
                                       "__builtin_r600_read_global_size">;
defm int_r600_read_local_size : R600ReadPreloadRegisterIntrinsic_xyz <
                                       "__builtin_r600_read_local_size">;
defm int_r600_read_ngroups : R600ReadPreloadRegisterIntrinsic_xyz <
                                       "__builtin_r600_read_ngroups">;
defm int_r600_read_tgid : R600ReadPreloadRegisterIntrinsic_xyz <
                                       "__builtin_r600_read_tgid">;
defm int_r600_read_tidig : R600ReadPreloadRegisterIntrinsic_xyz <
                                       "__builtin_r600_read_tidig">;
} // End TargetPrefix = "r600"

let TargetPrefix = "AMDGPU" in {

class AMDGPUReadPreloadRegisterIntrinsic<string name>
  : Intrinsic<[llvm_i32_ty], [], [IntrNoMem]>,
    GCCBuiltin<name>;

def int_AMDGPU_div_scale : GCCBuiltin<"__builtin_amdgpu_div_scale">,
  // 1st parameter: Numerator
  // 2nd parameter: Denominator
  // 3rd parameter: Constant to select select between first and
  //                second. (0 = first, 1 = second).
  Intrinsic<[llvm_anyfloat_ty, llvm_i1_ty],
            [LLVMMatchType<0>, LLVMMatchType<0>, llvm_i1_ty],
            [IntrNoMem]>;

def int_AMDGPU_div_fmas : GCCBuiltin<"__builtin_amdgpu_div_fmas">,
  Intrinsic<[llvm_anyfloat_ty],
            [LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>, llvm_i1_ty],
            [IntrNoMem]>;

def int_AMDGPU_div_fixup : GCCBuiltin<"__builtin_amdgpu_div_fixup">,
  Intrinsic<[llvm_anyfloat_ty],
            [LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>],
            [IntrNoMem]>;

def int_AMDGPU_trig_preop : GCCBuiltin<"__builtin_amdgpu_trig_preop">,
  Intrinsic<[llvm_anyfloat_ty], [LLVMMatchType<0>, llvm_i32_ty],
            [IntrNoMem]>;

def int_AMDGPU_rcp : GCCBuiltin<"__builtin_amdgpu_rcp">,
  Intrinsic<[llvm_anyfloat_ty], [LLVMMatchType<0>], [IntrNoMem]>;

def int_AMDGPU_rsq : GCCBuiltin<"__builtin_amdgpu_rsq">,
  Intrinsic<[llvm_anyfloat_ty], [LLVMMatchType<0>], [IntrNoMem]>;

def int_AMDGPU_rsq_clamped : GCCBuiltin<"__builtin_amdgpu_rsq_clamped">,
  Intrinsic<[llvm_anyfloat_ty], [LLVMMatchType<0>], [IntrNoMem]>;

def int_AMDGPU_ldexp : GCCBuiltin<"__builtin_amdgpu_ldexp">,
  Intrinsic<[llvm_anyfloat_ty], [LLVMMatchType<0>, llvm_i32_ty], [IntrNoMem]>;

def int_AMDGPU_class : GCCBuiltin<"__builtin_amdgpu_class">,
  Intrinsic<[llvm_i1_ty], [llvm_anyfloat_ty, llvm_i32_ty], [IntrNoMem]>;

def int_AMDGPU_read_workdim : AMDGPUReadPreloadRegisterIntrinsic <
                                       "__builtin_amdgpu_read_workdim">;

} // End TargetPrefix = "AMDGPU"
