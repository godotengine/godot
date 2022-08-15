//===- IntrinsicsHexagon.td - Defines Hexagon intrinsics ---*- tablegen -*-===//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines all of the Hexagon-specific intrinsics.
//
//===----------------------------------------------------------------------===//

let TargetPrefix = "hexagon" in {
  /// Hexagon_Intrinsic - Base class for all Hexagon intrinsics.
  class Hexagon_Intrinsic<string GCCIntSuffix, list<LLVMType> ret_types,
                              list<LLVMType> param_types,
                              list<IntrinsicProperty> properties>
    : GCCBuiltin<!strconcat("__builtin_", GCCIntSuffix)>,
      Intrinsic<ret_types, param_types, properties>;
}

//===----------------------------------------------------------------------===//
//
// DEF_FUNCTION_TYPE_1(QI_ftype_MEM,BT_BOOL,BT_PTR) ->
// Hexagon_qi_mem_Intrinsic<string GCCIntSuffix>
//
class Hexagon_qi_mem_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i1_ty], [llvm_ptr_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_1(void_ftype_SI,BT_VOID,BT_INT) ->
// Hexagon_void_si_Intrinsic<string GCCIntSuffix>
//
class Hexagon_void_si_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_void_ty], [llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_1(HI_ftype_SI,BT_I16,BT_INT) ->
// Hexagon_hi_si_Intrinsic<string GCCIntSuffix>
//
class Hexagon_hi_si_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i16_ty], [llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_1(SI_ftype_SI,BT_INT,BT_INT) ->
// Hexagon_si_si_Intrinsic<string GCCIntSuffix>
//
class Hexagon_si_si_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i32_ty], [llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_1(DI_ftype_SI,BT_LONGLONG,BT_INT) ->
// Hexagon_di_si_Intrinsic<string GCCIntSuffix>
//
class Hexagon_di_si_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i64_ty], [llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_1(SI_ftype_DI,BT_INT,BT_LONGLONG) ->
// Hexagon_si_di_Intrinsic<string GCCIntSuffix>
//
class Hexagon_si_di_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i32_ty], [llvm_i64_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_1(DI_ftype_DI,BT_LONGLONG,BT_LONGLONG) ->
// Hexagon_di_di_Intrinsic<string GCCIntSuffix>
//
class Hexagon_di_di_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i64_ty], [llvm_i64_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_1(QI_ftype_QI,BT_BOOL,BT_BOOL) ->
// Hexagon_qi_qi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_qi_qi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i1_ty], [llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_1(QI_ftype_SI,BT_BOOL,BT_INT) ->
// Hexagon_qi_si_Intrinsic<string GCCIntSuffix>
//
class Hexagon_qi_si_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i1_ty], [llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_1(DI_ftype_QI,BT_LONGLONG,BT_BOOL) ->
// Hexagon_di_qi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_di_qi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i64_ty], [llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_1(SI_ftype_QI,BT_INT,BT_BOOL) ->
// Hexagon_si_qi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_si_qi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i32_ty], [llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_2(QI_ftype_SISI,BT_BOOL,BT_INT,BT_INT) ->
// Hexagon_qi_sisi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_qi_sisi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i1_ty], [llvm_i32_ty, llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_2(void_ftype_SISI,BT_VOID,BT_INT,BT_INT) ->
// Hexagon_void_sisi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_void_sisi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_void_ty], [llvm_i32_ty, llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_2(SI_ftype_SISI,BT_INT,BT_INT,BT_INT) ->
// Hexagon_si_sisi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_si_sisi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_2(USI_ftype_SISI,BT_UINT,BT_INT,BT_INT) ->
// Hexagon_usi_sisi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_usi_sisi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_2(DI_ftype_SISI,BT_LONGLONG,BT_INT,BT_INT) ->
// Hexagon_di_sisi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_di_sisi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i64_ty], [llvm_i32_ty, llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_2(UDI_ftype_SISI,BT_ULONGLONG,BT_INT,BT_INT) ->
// Hexagon_udi_sisi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_udi_sisi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i64_ty], [llvm_i32_ty, llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_2(DI_ftype_SIDI,BT_LONGLONG,BT_INT,BT_LONGLONG) ->
// Hexagon_di_sidi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_di_sidi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i64_ty], [llvm_i32_ty, llvm_i64_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_2(DI_ftype_DISI,BT_LONGLONG,BT_LONGLONG,BT_INT) ->
// Hexagon_di_disi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_di_disi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i64_ty], [llvm_i64_ty, llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_2(SI_ftype_SIDI,BT_INT,BT_INT,BT_LONGLONG) ->
// Hexagon_si_sidi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_si_sidi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i32_ty], [llvm_i32_ty, llvm_i64_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_2(SI_ftype_DIDI,BT_INT,BT_LONGLONG,BT_LONGLONG) ->
// Hexagon_si_didi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_si_didi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i32_ty], [llvm_i64_ty, llvm_i64_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_2(DI_ftype_DIDI,BT_LONGLONG,BT_LONGLONG,BT_LONGLONG) ->
// Hexagon_di_didi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_di_didi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i64_ty], [llvm_i64_ty, llvm_i64_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_2(UDI_ftype_DIDI,BT_ULONGLONG,BT_LONGLONG,BT_LONGLONG) ->
// Hexagon_udi_didi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_udi_didi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i64_ty], [llvm_i64_ty, llvm_i64_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_2(SI_ftype_DISI,BT_INT,BT_LONGLONG,BT_INT) ->
// Hexagon_si_disi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_si_disi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i32_ty], [llvm_i64_ty, llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_2(QI_ftype_DIDI,BT_BOOL,BT_LONGLONG,BT_LONGLONG) ->
// Hexagon_qi_didi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_qi_didi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i1_ty], [llvm_i64_ty, llvm_i64_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_2(QI_ftype_SIDI,BT_BOOL,BT_INT,BT_LONGLONG) ->
// Hexagon_qi_didi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_qi_sidi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i1_ty], [llvm_i32_ty, llvm_i64_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_2(QI_ftype_DISI,BT_BOOL,BT_LONGLONG,BT_INT) ->
// Hexagon_qi_disi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_qi_disi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i1_ty], [llvm_i64_ty, llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_2(QI_ftype_QIQI,BT_BOOL,BT_BOOL,BT_BOOL) ->
// Hexagon_qi_qiqi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_qi_qiqi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i1_ty], [llvm_i32_ty, llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_2(QI_ftype_QIQIQI,BT_BOOL,BT_BOOL,BT_BOOL) ->
// Hexagon_qi_qiqiqi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_qi_qiqiqi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i1_ty], [llvm_i1_ty, llvm_i1_ty, llvm_i1_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_2(SI_ftype_QIQI,BT_INT,BT_BOOL,BT_BOOL) ->
// Hexagon_si_qiqi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_si_qiqi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_2(SI_ftype_QISI,BT_INT,BT_BOOL,BT_INT) ->
// Hexagon_si_qisi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_si_qisi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i32_ty], [llvm_i1_ty, llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_3(void_ftype_SISISI,BT_VOID,BT_INT,BT_INT,BT_INT) ->
// Hexagon_void_sisisi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_void_sisisi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_void_ty], [llvm_i32_ty, llvm_i32_ty,
                           llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_3(SI_ftype_SISISI,BT_INT,BT_INT,BT_INT,BT_INT) ->
// Hexagon_si_sisisi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_si_sisisi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty,
                           llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_3(DI_ftype_SISISI,BT_LONGLONG,BT_INT,BT_INT,BT_INT) ->
// Hexagon_di_sisisi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_di_sisisi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i64_ty], [llvm_i32_ty, llvm_i32_ty,
                           llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_3(SI_ftype_DISISI,BT_INT,BT_LONGLONG,BT_INT,BT_INT) ->
// Hexagon_si_disisi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_si_disisi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i32_ty], [llvm_i64_ty, llvm_i32_ty,
                           llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_3(DI_ftype_DISISI,BT_LONGLONG,BT_LONGLONG,BT_INT,BT_INT) ->
// Hexagon_di_disisi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_di_disisi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i64_ty], [llvm_i64_ty, llvm_i32_ty,
                           llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_3(SI_ftype_SIDISI,BT_INT,BT_INT,BT_LONGLONG,BT_INT) ->
// Hexagon_si_sidisi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_si_sidisi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i32_ty], [llvm_i32_ty, llvm_i64_ty,
                           llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_3(DI_ftype_DIDISI,BT_LONGLONG,BT_LONGLONG,
//                     BT_LONGLONG,BT_INT) ->
// Hexagon_di_didisi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_di_didisi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i64_ty], [llvm_i64_ty, llvm_i64_ty,
                           llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_3(SI_ftype_SIDIDI,BT_INT,BT_INT,BT_LONGLONG,BT_LONGLONG) ->
// Hexagon_si_sididi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_si_sididi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i32_ty], [llvm_i32_ty, llvm_i64_ty,
                           llvm_i64_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_3(DI_ftype_DIDIDI,BT_LONGLONG,BT_LONGLONG,BT_LONGLONG,
//                     BT_LONGLONG) ->
// Hexagon_di_dididi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_di_dididi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i64_ty], [llvm_i64_ty, llvm_i64_ty,
                           llvm_i64_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_3(SI_ftype_SISIDI,BT_INT,BT_INT,BT_INT,BT_LONGLONG) ->
// Hexagon_si_sisidi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_si_sisidi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty,
                           llvm_i64_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_3(SI_ftype_QISISI,BT_INT,BT_BOOL,BT_INT,BT_INT) ->
// Hexagon_si_qisisi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_si_qisisi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty,
                           llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_3(DI_ftype_QISISI,BT_LONGLONG,BT_BOOL,BT_INT,BT_INT) ->
// Hexagon_di_qisisi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_di_qisisi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i64_ty], [llvm_i1_ty, llvm_i32_ty,
                           llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_3(DI_ftype_QIDIDI,BT_LONGLONG,BT_BOOL,BT_LONGLONG,
//                     BT_LONGLONG) ->
// Hexagon_di_qididi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_di_qididi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i64_ty], [llvm_i32_ty, llvm_i64_ty,
                           llvm_i64_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_3(DI_ftype_DIDIQI,BT_LONGLONG,BT_LONGLONG,BT_LONGLONG,
//                     BT_BOOL) ->
// Hexagon_di_didiqi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_di_didiqi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i64_ty], [llvm_i64_ty, llvm_i64_ty,
                           llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_4(SI_ftype_SISISISI,BT_INT,BT_INT,BT_INT,BT_INT,BT_INT) ->
// Hexagon_si_sisisisi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_si_sisisisi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty,
                           llvm_i32_ty, llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_4(DI_ftype_DIDISISI,BT_LONGLONG,BT_LONGLONG,
//                     BT_LONGLONG,BT_INT,BT_INT) ->
// Hexagon_di_didisisi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_di_didisisi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i64_ty], [llvm_i64_ty, llvm_i64_ty,
                           llvm_i32_ty, llvm_i32_ty],
                          [IntrNoMem]>;

class Hexagon_mem_memmemsi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_ptr_ty], [llvm_ptr_ty, llvm_ptr_ty,
                           llvm_i32_ty],
                          [IntrReadWriteArgMem]>;

class Hexagon_mem_memsisi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_ptr_ty], [llvm_ptr_ty, llvm_i32_ty,
                           llvm_i32_ty],
                          [IntrReadWriteArgMem]>;

class Hexagon_mem_memdisi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_ptr_ty], [llvm_ptr_ty, llvm_i64_ty,
                           llvm_i32_ty],
                          [IntrReadWriteArgMem]>;

class Hexagon_mem_memmemsisi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_ptr_ty], [llvm_ptr_ty, llvm_ptr_ty,
                           llvm_i32_ty, llvm_i32_ty],
                          [IntrReadWriteArgMem]>;

class Hexagon_mem_memsisisi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_ptr_ty], [llvm_ptr_ty, llvm_i32_ty,
                           llvm_i32_ty, llvm_i32_ty],
                          [IntrReadWriteArgMem]>;

class Hexagon_mem_memdisisi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_ptr_ty], [llvm_ptr_ty, llvm_i64_ty,
                           llvm_i32_ty, llvm_i32_ty],
                          [IntrReadWriteArgMem]>;

//
// Hexagon_sf_df_Intrinsic<string GCCIntSuffix>
//
class Hexagon_sf_si_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_float_ty], [llvm_i32_ty],
                          [IntrNoMem, Throws]>;
//
// Hexagon_sf_df_Intrinsic<string GCCIntSuffix>
//
class Hexagon_sf_df_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_float_ty], [llvm_double_ty],
                          [IntrNoMem]>;
//
// Hexagon_sf_di_Intrinsic<string GCCIntSuffix>
//
class Hexagon_sf_di_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_float_ty], [llvm_i64_ty],
                          [IntrNoMem]>;
//
// Hexagon_df_sf_Intrinsic<string GCCIntSuffix>
//
class Hexagon_df_sf_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_double_ty], [llvm_float_ty],
                          [IntrNoMem]>;
//
// Hexagon_di_sf_Intrinsic<string GCCIntSuffix>
//
class Hexagon_di_sf_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i64_ty], [llvm_float_ty],
                          [IntrNoMem]>;
//
// Hexagon_sf_sf_Intrinsic<string GCCIntSuffix>
//
class Hexagon_sf_sf_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_float_ty], [llvm_float_ty],
                          [IntrNoMem]>;
//
// Hexagon_si_sf_Intrinsic<string GCCIntSuffix>
//
class Hexagon_si_sf_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i32_ty], [llvm_float_ty],
                          [IntrNoMem]>;
//
// Hexagon_si_df_Intrinsic<string GCCIntSuffix>
//
class Hexagon_si_df_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i32_ty], [llvm_double_ty],
                          [IntrNoMem]>;
//
// Hexagon_sf_sfsf_Intrinsic<string GCCIntSuffix>
//
class Hexagon_sf_sfsf_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_float_ty], [llvm_float_ty, llvm_float_ty],
                          [IntrNoMem, Throws]>;
//
// Hexagon_si_sfsf_Intrinsic<string GCCIntSuffix>
//
class Hexagon_si_sfsf_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i32_ty], [llvm_float_ty, llvm_float_ty],
                          [IntrNoMem, Throws]>;
//
// Hexagon_si_sfsi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_si_sfsi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i32_ty], [llvm_float_ty, llvm_i32_ty],
                          [IntrNoMem, Throws]>;
//
// Hexagon_qi_sfqi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_qi_sfqi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i1_ty], [llvm_float_ty, llvm_i32_ty],
                          [IntrNoMem]>;
//
// Hexagon_sf_sfsfsf_Intrinsic<string GCCIntSuffix>
//
class Hexagon_sf_sfsfsf_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_float_ty], [llvm_float_ty, llvm_float_ty,
                                            llvm_float_ty],
                          [IntrNoMem, Throws]>;
//
// Hexagon_sf_sfsfsfqi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_sf_sfsfsfqi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_float_ty], [llvm_float_ty, llvm_float_ty,
                                            llvm_float_ty,
                           llvm_i32_ty],
                          [IntrNoMem, Throws]>;
//
// Hexagon_di_dididi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_di_dididisi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i64_ty], [llvm_i64_ty, llvm_i64_ty,
                           llvm_i64_ty, llvm_i32_ty],
                          [IntrNoMem]>;
//
// Hexagon_df_si_Intrinsic<string GCCIntSuffix>
//
class Hexagon_df_si_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_double_ty], [llvm_i32_ty],
                          [IntrNoMem, Throws]>;
//
// Hexagon_df_di_Intrinsic<string GCCIntSuffix>
//
class Hexagon_df_di_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_double_ty], [llvm_i64_ty],
                          [IntrNoMem]>;
//
// Hexagon_di_df_Intrinsic<string GCCIntSuffix>
//
class Hexagon_di_df_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i64_ty], [llvm_double_ty],
                          [IntrNoMem]>;
//
// Hexagon_df_df_Intrinsic<string GCCIntSuffix>
//
class Hexagon_df_df_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_double_ty], [llvm_double_ty],
                          [IntrNoMem]>;
//
// Hexagon_df_dfdf_Intrinsic<string GCCIntSuffix>
//
class Hexagon_df_dfdf_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_double_ty], [llvm_double_ty, llvm_double_ty],
                          [IntrNoMem, Throws]>;
//
// Hexagon_si_dfdf_Intrinsic<string GCCIntSuffix>
//
class Hexagon_si_dfdf_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i32_ty], [llvm_double_ty, llvm_double_ty],
                          [IntrNoMem, Throws]>;
//
// Hexagon_si_dfsi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_si_dfsi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i32_ty], [llvm_double_ty, llvm_i32_ty],
                          [IntrNoMem, Throws]>;
//
//
// Hexagon_df_dfdfdf_Intrinsic<string GCCIntSuffix>
//
class Hexagon_df_dfdfdf_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_double_ty], [llvm_double_ty, llvm_double_ty,
                                             llvm_double_ty],
                          [IntrNoMem, Throws]>;
//
// Hexagon_df_dfdfdf_Intrinsic<string GCCIntSuffix>
//
class Hexagon_df_dfdfdfqi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_double_ty], [llvm_double_ty, llvm_double_ty,
                                             llvm_double_ty,
                          llvm_i32_ty],
                          [IntrNoMem, Throws]>;


// This one below will not be auto-generated,
// so make sure, you don't overwrite this one.
//
// BUILTIN_INFO(SI_to_SXTHI_asrh,SI_ftype_SI,1)
//
def int_hexagon_SI_to_SXTHI_asrh :
Hexagon_si_si_Intrinsic<"SI_to_SXTHI_asrh">;
//
// BUILTIN_INFO_NONCONST(brev_ldd,PTR_ftype_PTRPTRSI,3)
//
def int_hexagon_brev_ldd :
Hexagon_mem_memmemsi_Intrinsic<"brev_ldd">;
//
// BUILTIN_INFO_NONCONST(brev_ldw,PTR_ftype_PTRPTRSI,3)
//
def int_hexagon_brev_ldw :
Hexagon_mem_memmemsi_Intrinsic<"brev_ldw">;
//
// BUILTIN_INFO_NONCONST(brev_ldh,PTR_ftype_PTRPTRSI,3)
//
def int_hexagon_brev_ldh :
Hexagon_mem_memmemsi_Intrinsic<"brev_ldh">;
//
// BUILTIN_INFO_NONCONST(brev_lduh,PTR_ftype_PTRPTRSI,3)
//
def int_hexagon_brev_lduh :
Hexagon_mem_memmemsi_Intrinsic<"brev_lduh">;
//
// BUILTIN_INFO_NONCONST(brev_ldb,PTR_ftype_PTRPTRSI,3)
//
def int_hexagon_brev_ldb :
Hexagon_mem_memmemsi_Intrinsic<"brev_ldb">;
//
// BUILTIN_INFO_NONCONST(brev_ldub,PTR_ftype_PTRPTRSI,3)
//
def int_hexagon_brev_ldub :
Hexagon_mem_memmemsi_Intrinsic<"brev_ldub">;
//
// BUILTIN_INFO_NONCONST(circ_ldd,PTR_ftype_PTRPTRSISI,4)
//
def int_hexagon_circ_ldd :
Hexagon_mem_memmemsisi_Intrinsic<"circ_ldd">;
//
// BUILTIN_INFO_NONCONST(circ_ldw,PTR_ftype_PTRPTRSISI,4)
//
def int_hexagon_circ_ldw :
Hexagon_mem_memmemsisi_Intrinsic<"circ_ldw">;
//
// BUILTIN_INFO_NONCONST(circ_ldh,PTR_ftype_PTRPTRSISI,4)
//
def int_hexagon_circ_ldh :
Hexagon_mem_memmemsisi_Intrinsic<"circ_ldh">;
//
// BUILTIN_INFO_NONCONST(circ_lduh,PTR_ftype_PTRPTRSISI,4)
//
def int_hexagon_circ_lduh :
Hexagon_mem_memmemsisi_Intrinsic<"circ_lduh">;
//
// BUILTIN_INFO_NONCONST(circ_ldb,PTR_ftype_PTRPTRSISI,4)
//
def int_hexagon_circ_ldb :
Hexagon_mem_memmemsisi_Intrinsic<"circ_ldb">;
//
// BUILTIN_INFO_NONCONST(circ_ldub,PTR_ftype_PTRPTRSISI,4)
//
def int_hexagon_circ_ldub :
Hexagon_mem_memmemsisi_Intrinsic<"circ_ldub">;

//
// BUILTIN_INFO_NONCONST(brev_stb,PTR_ftype_PTRSISI,3)
//
def int_hexagon_brev_stb :
Hexagon_mem_memsisi_Intrinsic<"brev_stb">;
//
// BUILTIN_INFO_NONCONST(brev_sthhi,PTR_ftype_PTRSISI,3)
//
def int_hexagon_brev_sthhi :
Hexagon_mem_memsisi_Intrinsic<"brev_sthhi">;
//
// BUILTIN_INFO_NONCONST(brev_sth,PTR_ftype_PTRSISI,3)
//
def int_hexagon_brev_sth :
Hexagon_mem_memsisi_Intrinsic<"brev_sth">;
//
// BUILTIN_INFO_NONCONST(brev_stw,PTR_ftype_PTRSISI,3)
//
def int_hexagon_brev_stw :
Hexagon_mem_memsisi_Intrinsic<"brev_stw">;
//
// BUILTIN_INFO_NONCONST(brev_std,PTR_ftype_PTRSISI,3)
//
def int_hexagon_brev_std :
Hexagon_mem_memdisi_Intrinsic<"brev_std">;
//
// BUILTIN_INFO_NONCONST(circ_std,PTR_ftype_PTRDISISI,4)
//
def int_hexagon_circ_std :
Hexagon_mem_memdisisi_Intrinsic<"circ_std">;
//
// BUILTIN_INFO_NONCONST(circ_stw,PTR_ftype_PTRSISISI,4)
//
def int_hexagon_circ_stw :
Hexagon_mem_memsisisi_Intrinsic<"circ_stw">;
//
// BUILTIN_INFO_NONCONST(circ_sth,PTR_ftype_PTRSISISI,4)
//
def int_hexagon_circ_sth :
Hexagon_mem_memsisisi_Intrinsic<"circ_sth">;
//
// BUILTIN_INFO_NONCONST(circ_sthhi,PTR_ftype_PTRSISISI,4)
//
def int_hexagon_circ_sthhi :
Hexagon_mem_memsisisi_Intrinsic<"circ_sthhi">;
//
// BUILTIN_INFO_NONCONST(circ_stb,PTR_ftype_PTRSISISI,4)
//
def int_hexagon_circ_stb :
Hexagon_mem_memsisisi_Intrinsic<"circ_stb">;


//
// BUILTIN_INFO(HEXAGON.C2_cmpeq,QI_ftype_SISI,2)
//
def int_hexagon_C2_cmpeq :
Hexagon_si_sisi_Intrinsic<"HEXAGON_C2_cmpeq">;
//
// BUILTIN_INFO(HEXAGON.C2_cmpgt,QI_ftype_SISI,2)
//
def int_hexagon_C2_cmpgt :
Hexagon_si_sisi_Intrinsic<"HEXAGON_C2_cmpgt">;
//
// BUILTIN_INFO(HEXAGON.C2_cmpgtu,QI_ftype_SISI,2)
//
def int_hexagon_C2_cmpgtu :
Hexagon_si_sisi_Intrinsic<"HEXAGON_C2_cmpgtu">;
//
// BUILTIN_INFO(HEXAGON.C2_cmpeqp,QI_ftype_DIDI,2)
//
def int_hexagon_C2_cmpeqp :
Hexagon_si_didi_Intrinsic<"HEXAGON_C2_cmpeqp">;
//
// BUILTIN_INFO(HEXAGON.C2_cmpgtp,QI_ftype_DIDI,2)
//
def int_hexagon_C2_cmpgtp :
Hexagon_si_didi_Intrinsic<"HEXAGON_C2_cmpgtp">;
//
// BUILTIN_INFO(HEXAGON.C2_cmpgtup,QI_ftype_DIDI,2)
//
def int_hexagon_C2_cmpgtup :
Hexagon_si_didi_Intrinsic<"HEXAGON_C2_cmpgtup">;
//
// BUILTIN_INFO(HEXAGON.A4_rcmpeqi,SI_ftype_SISI,2)
//
def int_hexagon_A4_rcmpeqi :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A4_rcmpeqi">;
//
// BUILTIN_INFO(HEXAGON.A4_rcmpneqi,SI_ftype_SISI,2)
//
def int_hexagon_A4_rcmpneqi :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A4_rcmpneqi">;
//
// BUILTIN_INFO(HEXAGON.A4_rcmpeq,SI_ftype_SISI,2)
//
def int_hexagon_A4_rcmpeq :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A4_rcmpeq">;
//
// BUILTIN_INFO(HEXAGON.A4_rcmpneq,SI_ftype_SISI,2)
//
def int_hexagon_A4_rcmpneq :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A4_rcmpneq">;
//
// BUILTIN_INFO(HEXAGON.C2_bitsset,QI_ftype_SISI,2)
//
def int_hexagon_C2_bitsset :
Hexagon_si_sisi_Intrinsic<"HEXAGON_C2_bitsset">;
//
// BUILTIN_INFO(HEXAGON.C2_bitsclr,QI_ftype_SISI,2)
//
def int_hexagon_C2_bitsclr :
Hexagon_si_sisi_Intrinsic<"HEXAGON_C2_bitsclr">;
//
// BUILTIN_INFO(HEXAGON.C4_nbitsset,QI_ftype_SISI,2)
//
def int_hexagon_C4_nbitsset :
Hexagon_si_sisi_Intrinsic<"HEXAGON_C4_nbitsset">;
//
// BUILTIN_INFO(HEXAGON.C4_nbitsclr,QI_ftype_SISI,2)
//
def int_hexagon_C4_nbitsclr :
Hexagon_si_sisi_Intrinsic<"HEXAGON_C4_nbitsclr">;
//
// BUILTIN_INFO(HEXAGON.C2_cmpeqi,QI_ftype_SISI,2)
//
def int_hexagon_C2_cmpeqi :
Hexagon_si_sisi_Intrinsic<"HEXAGON_C2_cmpeqi">;
//
// BUILTIN_INFO(HEXAGON.C2_cmpgti,QI_ftype_SISI,2)
//
def int_hexagon_C2_cmpgti :
Hexagon_si_sisi_Intrinsic<"HEXAGON_C2_cmpgti">;
//
// BUILTIN_INFO(HEXAGON.C2_cmpgtui,QI_ftype_SISI,2)
//
def int_hexagon_C2_cmpgtui :
Hexagon_si_sisi_Intrinsic<"HEXAGON_C2_cmpgtui">;
//
// BUILTIN_INFO(HEXAGON.C2_cmpgei,QI_ftype_SISI,2)
//
def int_hexagon_C2_cmpgei :
Hexagon_si_sisi_Intrinsic<"HEXAGON_C2_cmpgei">;
//
// BUILTIN_INFO(HEXAGON.C2_cmpgeui,QI_ftype_SISI,2)
//
def int_hexagon_C2_cmpgeui :
Hexagon_si_sisi_Intrinsic<"HEXAGON_C2_cmpgeui">;
//
// BUILTIN_INFO(HEXAGON.C2_cmplt,QI_ftype_SISI,2)
//
def int_hexagon_C2_cmplt :
Hexagon_si_sisi_Intrinsic<"HEXAGON_C2_cmplt">;
//
// BUILTIN_INFO(HEXAGON.C2_cmpltu,QI_ftype_SISI,2)
//
def int_hexagon_C2_cmpltu :
Hexagon_si_sisi_Intrinsic<"HEXAGON_C2_cmpltu">;
//
// BUILTIN_INFO(HEXAGON.C2_bitsclri,QI_ftype_SISI,2)
//
def int_hexagon_C2_bitsclri :
Hexagon_si_sisi_Intrinsic<"HEXAGON_C2_bitsclri">;
//
// BUILTIN_INFO(HEXAGON.C4_nbitsclri,QI_ftype_SISI,2)
//
def int_hexagon_C4_nbitsclri :
Hexagon_si_sisi_Intrinsic<"HEXAGON_C4_nbitsclri">;
//
// BUILTIN_INFO(HEXAGON.C4_cmpneqi,QI_ftype_SISI,2)
//
def int_hexagon_C4_cmpneqi :
Hexagon_si_sisi_Intrinsic<"HEXAGON_C4_cmpneqi">;
//
// BUILTIN_INFO(HEXAGON.C4_cmpltei,QI_ftype_SISI,2)
//
def int_hexagon_C4_cmpltei :
Hexagon_si_sisi_Intrinsic<"HEXAGON_C4_cmpltei">;
//
// BUILTIN_INFO(HEXAGON.C4_cmplteui,QI_ftype_SISI,2)
//
def int_hexagon_C4_cmplteui :
Hexagon_si_sisi_Intrinsic<"HEXAGON_C4_cmplteui">;
//
// BUILTIN_INFO(HEXAGON.C4_cmpneq,QI_ftype_SISI,2)
//
def int_hexagon_C4_cmpneq :
Hexagon_si_sisi_Intrinsic<"HEXAGON_C4_cmpneq">;
//
// BUILTIN_INFO(HEXAGON.C4_cmplte,QI_ftype_SISI,2)
//
def int_hexagon_C4_cmplte :
Hexagon_si_sisi_Intrinsic<"HEXAGON_C4_cmplte">;
//
// BUILTIN_INFO(HEXAGON.C4_cmplteu,QI_ftype_SISI,2)
//
def int_hexagon_C4_cmplteu :
Hexagon_si_sisi_Intrinsic<"HEXAGON_C4_cmplteu">;
//
// BUILTIN_INFO(HEXAGON.C2_and,QI_ftype_QIQI,2)
//
def int_hexagon_C2_and :
Hexagon_si_sisi_Intrinsic<"HEXAGON_C2_and">;
//
// BUILTIN_INFO(HEXAGON.C2_or,QI_ftype_QIQI,2)
//
def int_hexagon_C2_or :
Hexagon_si_sisi_Intrinsic<"HEXAGON_C2_or">;
//
// BUILTIN_INFO(HEXAGON.C2_xor,QI_ftype_QIQI,2)
//
def int_hexagon_C2_xor :
Hexagon_si_sisi_Intrinsic<"HEXAGON_C2_xor">;
//
// BUILTIN_INFO(HEXAGON.C2_andn,QI_ftype_QIQI,2)
//
def int_hexagon_C2_andn :
Hexagon_si_sisi_Intrinsic<"HEXAGON_C2_andn">;
//
// BUILTIN_INFO(HEXAGON.C2_not,QI_ftype_QI,1)
//
def int_hexagon_C2_not :
Hexagon_si_si_Intrinsic<"HEXAGON_C2_not">;
//
// BUILTIN_INFO(HEXAGON.C2_orn,QI_ftype_QIQI,2)
//
def int_hexagon_C2_orn :
Hexagon_si_sisi_Intrinsic<"HEXAGON_C2_orn">;
//
// BUILTIN_INFO(HEXAGON.C4_and_and,QI_ftype_QIQIQI,3)
//
def int_hexagon_C4_and_and :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_C4_and_and">;
//
// BUILTIN_INFO(HEXAGON.C4_and_or,QI_ftype_QIQIQI,3)
//
def int_hexagon_C4_and_or :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_C4_and_or">;
//
// BUILTIN_INFO(HEXAGON.C4_or_and,QI_ftype_QIQIQI,3)
//
def int_hexagon_C4_or_and :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_C4_or_and">;
//
// BUILTIN_INFO(HEXAGON.C4_or_or,QI_ftype_QIQIQI,3)
//
def int_hexagon_C4_or_or :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_C4_or_or">;
//
// BUILTIN_INFO(HEXAGON.C4_and_andn,QI_ftype_QIQIQI,3)
//
def int_hexagon_C4_and_andn :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_C4_and_andn">;
//
// BUILTIN_INFO(HEXAGON.C4_and_orn,QI_ftype_QIQIQI,3)
//
def int_hexagon_C4_and_orn :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_C4_and_orn">;
//
// BUILTIN_INFO(HEXAGON.C4_or_andn,QI_ftype_QIQIQI,3)
//
def int_hexagon_C4_or_andn :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_C4_or_andn">;
//
// BUILTIN_INFO(HEXAGON.C4_or_orn,QI_ftype_QIQIQI,3)
//
def int_hexagon_C4_or_orn :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_C4_or_orn">;
//
// BUILTIN_INFO(HEXAGON.C2_pxfer_map,QI_ftype_QI,1)
//
def int_hexagon_C2_pxfer_map :
Hexagon_si_qi_Intrinsic<"HEXAGON_C2_pxfer_map">;
//
// BUILTIN_INFO(HEXAGON.C2_any8,QI_ftype_QI,1)
//
def int_hexagon_C2_any8 :
Hexagon_si_qi_Intrinsic<"HEXAGON_C2_any8">;
//
// BUILTIN_INFO(HEXAGON.C2_all8,QI_ftype_QI,1)
//
def int_hexagon_C2_all8 :
Hexagon_si_qi_Intrinsic<"HEXAGON_C2_all8">;
//
// BUILTIN_INFO(HEXAGON.C2_vitpack,SI_ftype_QIQI,2)
//
def int_hexagon_C2_vitpack :
Hexagon_si_qiqi_Intrinsic<"HEXAGON_C2_vitpack">;
//
// BUILTIN_INFO(HEXAGON.C2_mux,SI_ftype_QISISI,3)
//
def int_hexagon_C2_mux :
Hexagon_si_qisisi_Intrinsic<"HEXAGON_C2_mux">;
//
// BUILTIN_INFO(HEXAGON.C2_muxii,SI_ftype_QISISI,3)
//
def int_hexagon_C2_muxii :
Hexagon_si_qisisi_Intrinsic<"HEXAGON_C2_muxii">;
//
// BUILTIN_INFO(HEXAGON.C2_muxir,SI_ftype_QISISI,3)
//
def int_hexagon_C2_muxir :
Hexagon_si_qisisi_Intrinsic<"HEXAGON_C2_muxir">;
//
// BUILTIN_INFO(HEXAGON.C2_muxri,SI_ftype_QISISI,3)
//
def int_hexagon_C2_muxri :
Hexagon_si_qisisi_Intrinsic<"HEXAGON_C2_muxri">;
//
// BUILTIN_INFO(HEXAGON.C2_vmux,DI_ftype_QIDIDI,3)
//
def int_hexagon_C2_vmux :
Hexagon_di_qididi_Intrinsic<"HEXAGON_C2_vmux">;
//
// BUILTIN_INFO(HEXAGON.C2_mask,DI_ftype_QI,1)
//
def int_hexagon_C2_mask :
Hexagon_di_qi_Intrinsic<"HEXAGON_C2_mask">;
//
// BUILTIN_INFO(HEXAGON.A2_vcmpbeq,QI_ftype_DIDI,2)
//
def int_hexagon_A2_vcmpbeq :
Hexagon_si_didi_Intrinsic<"HEXAGON_A2_vcmpbeq">;
//
// BUILTIN_INFO(HEXAGON.A4_vcmpbeqi,QI_ftype_DISI,2)
//
def int_hexagon_A4_vcmpbeqi :
Hexagon_si_disi_Intrinsic<"HEXAGON_A4_vcmpbeqi">;
//
// BUILTIN_INFO(HEXAGON.A4_vcmpbeq_any,QI_ftype_DIDI,2)
//
def int_hexagon_A4_vcmpbeq_any :
Hexagon_si_didi_Intrinsic<"HEXAGON_A4_vcmpbeq_any">;
//
// BUILTIN_INFO(HEXAGON.A2_vcmpbgtu,QI_ftype_DIDI,2)
//
def int_hexagon_A2_vcmpbgtu :
Hexagon_si_didi_Intrinsic<"HEXAGON_A2_vcmpbgtu">;
//
// BUILTIN_INFO(HEXAGON.A4_vcmpbgtui,QI_ftype_DISI,2)
//
def int_hexagon_A4_vcmpbgtui :
Hexagon_si_disi_Intrinsic<"HEXAGON_A4_vcmpbgtui">;
//
// BUILTIN_INFO(HEXAGON.A4_vcmpbgt,QI_ftype_DIDI,2)
//
def int_hexagon_A4_vcmpbgt :
Hexagon_si_didi_Intrinsic<"HEXAGON_A4_vcmpbgt">;
//
// BUILTIN_INFO(HEXAGON.A4_vcmpbgti,QI_ftype_DISI,2)
//
def int_hexagon_A4_vcmpbgti :
Hexagon_si_disi_Intrinsic<"HEXAGON_A4_vcmpbgti">;
//
// BUILTIN_INFO(HEXAGON.A4_cmpbeq,QI_ftype_SISI,2)
//
def int_hexagon_A4_cmpbeq :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A4_cmpbeq">;
//
// BUILTIN_INFO(HEXAGON.A4_cmpbeqi,QI_ftype_SISI,2)
//
def int_hexagon_A4_cmpbeqi :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A4_cmpbeqi">;
//
// BUILTIN_INFO(HEXAGON.A4_cmpbgtu,QI_ftype_SISI,2)
//
def int_hexagon_A4_cmpbgtu :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A4_cmpbgtu">;
//
// BUILTIN_INFO(HEXAGON.A4_cmpbgtui,QI_ftype_SISI,2)
//
def int_hexagon_A4_cmpbgtui :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A4_cmpbgtui">;
//
// BUILTIN_INFO(HEXAGON.A4_cmpbgt,QI_ftype_SISI,2)
//
def int_hexagon_A4_cmpbgt :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A4_cmpbgt">;
//
// BUILTIN_INFO(HEXAGON.A4_cmpbgti,QI_ftype_SISI,2)
//
def int_hexagon_A4_cmpbgti :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A4_cmpbgti">;
//
// BUILTIN_INFO(HEXAGON.A2_vcmpheq,QI_ftype_DIDI,2)
//
def int_hexagon_A2_vcmpheq :
Hexagon_si_didi_Intrinsic<"HEXAGON_A2_vcmpheq">;
//
// BUILTIN_INFO(HEXAGON.A2_vcmphgt,QI_ftype_DIDI,2)
//
def int_hexagon_A2_vcmphgt :
Hexagon_si_didi_Intrinsic<"HEXAGON_A2_vcmphgt">;
//
// BUILTIN_INFO(HEXAGON.A2_vcmphgtu,QI_ftype_DIDI,2)
//
def int_hexagon_A2_vcmphgtu :
Hexagon_si_didi_Intrinsic<"HEXAGON_A2_vcmphgtu">;
//
// BUILTIN_INFO(HEXAGON.A4_vcmpheqi,QI_ftype_DISI,2)
//
def int_hexagon_A4_vcmpheqi :
Hexagon_si_disi_Intrinsic<"HEXAGON_A4_vcmpheqi">;
//
// BUILTIN_INFO(HEXAGON.A4_vcmphgti,QI_ftype_DISI,2)
//
def int_hexagon_A4_vcmphgti :
Hexagon_si_disi_Intrinsic<"HEXAGON_A4_vcmphgti">;
//
// BUILTIN_INFO(HEXAGON.A4_vcmphgtui,QI_ftype_DISI,2)
//
def int_hexagon_A4_vcmphgtui :
Hexagon_si_disi_Intrinsic<"HEXAGON_A4_vcmphgtui">;
//
// BUILTIN_INFO(HEXAGON.A4_cmpheq,QI_ftype_SISI,2)
//
def int_hexagon_A4_cmpheq :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A4_cmpheq">;
//
// BUILTIN_INFO(HEXAGON.A4_cmphgt,QI_ftype_SISI,2)
//
def int_hexagon_A4_cmphgt :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A4_cmphgt">;
//
// BUILTIN_INFO(HEXAGON.A4_cmphgtu,QI_ftype_SISI,2)
//
def int_hexagon_A4_cmphgtu :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A4_cmphgtu">;
//
// BUILTIN_INFO(HEXAGON.A4_cmpheqi,QI_ftype_SISI,2)
//
def int_hexagon_A4_cmpheqi :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A4_cmpheqi">;
//
// BUILTIN_INFO(HEXAGON.A4_cmphgti,QI_ftype_SISI,2)
//
def int_hexagon_A4_cmphgti :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A4_cmphgti">;
//
// BUILTIN_INFO(HEXAGON.A4_cmphgtui,QI_ftype_SISI,2)
//
def int_hexagon_A4_cmphgtui :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A4_cmphgtui">;
//
// BUILTIN_INFO(HEXAGON.A2_vcmpweq,QI_ftype_DIDI,2)
//
def int_hexagon_A2_vcmpweq :
Hexagon_si_didi_Intrinsic<"HEXAGON_A2_vcmpweq">;
//
// BUILTIN_INFO(HEXAGON.A2_vcmpwgt,QI_ftype_DIDI,2)
//
def int_hexagon_A2_vcmpwgt :
Hexagon_si_didi_Intrinsic<"HEXAGON_A2_vcmpwgt">;
//
// BUILTIN_INFO(HEXAGON.A2_vcmpwgtu,QI_ftype_DIDI,2)
//
def int_hexagon_A2_vcmpwgtu :
Hexagon_si_didi_Intrinsic<"HEXAGON_A2_vcmpwgtu">;
//
// BUILTIN_INFO(HEXAGON.A4_vcmpweqi,QI_ftype_DISI,2)
//
def int_hexagon_A4_vcmpweqi :
Hexagon_si_disi_Intrinsic<"HEXAGON_A4_vcmpweqi">;
//
// BUILTIN_INFO(HEXAGON.A4_vcmpwgti,QI_ftype_DISI,2)
//
def int_hexagon_A4_vcmpwgti :
Hexagon_si_disi_Intrinsic<"HEXAGON_A4_vcmpwgti">;
//
// BUILTIN_INFO(HEXAGON.A4_vcmpwgtui,QI_ftype_DISI,2)
//
def int_hexagon_A4_vcmpwgtui :
Hexagon_si_disi_Intrinsic<"HEXAGON_A4_vcmpwgtui">;
//
// BUILTIN_INFO(HEXAGON.A4_boundscheck,QI_ftype_SIDI,2)
//
def int_hexagon_A4_boundscheck :
Hexagon_si_sidi_Intrinsic<"HEXAGON_A4_boundscheck">;
//
// BUILTIN_INFO(HEXAGON.A4_tlbmatch,QI_ftype_DISI,2)
//
def int_hexagon_A4_tlbmatch :
Hexagon_si_disi_Intrinsic<"HEXAGON_A4_tlbmatch">;
//
// BUILTIN_INFO(HEXAGON.C2_tfrpr,SI_ftype_QI,1)
//
def int_hexagon_C2_tfrpr :
Hexagon_si_qi_Intrinsic<"HEXAGON_C2_tfrpr">;
//
// BUILTIN_INFO(HEXAGON.C2_tfrrp,QI_ftype_SI,1)
//
def int_hexagon_C2_tfrrp :
Hexagon_si_si_Intrinsic<"HEXAGON_C2_tfrrp">;
//
// BUILTIN_INFO(HEXAGON.C4_fastcorner9,QI_ftype_QIQI,2)
//
def int_hexagon_C4_fastcorner9 :
Hexagon_si_qiqi_Intrinsic<"HEXAGON_C4_fastcorner9">;
//
// BUILTIN_INFO(HEXAGON.C4_fastcorner9_not,QI_ftype_QIQI,2)
//
def int_hexagon_C4_fastcorner9_not :
Hexagon_si_qiqi_Intrinsic<"HEXAGON_C4_fastcorner9_not">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_acc_hh_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_acc_hh_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpy_acc_hh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_acc_hh_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_acc_hh_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpy_acc_hh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_acc_hl_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_acc_hl_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpy_acc_hl_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_acc_hl_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_acc_hl_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpy_acc_hl_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_acc_lh_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_acc_lh_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpy_acc_lh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_acc_lh_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_acc_lh_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpy_acc_lh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_acc_ll_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_acc_ll_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpy_acc_ll_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_acc_ll_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_acc_ll_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpy_acc_ll_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_nac_hh_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_nac_hh_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpy_nac_hh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_nac_hh_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_nac_hh_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpy_nac_hh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_nac_hl_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_nac_hl_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpy_nac_hl_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_nac_hl_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_nac_hl_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpy_nac_hl_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_nac_lh_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_nac_lh_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpy_nac_lh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_nac_lh_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_nac_lh_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpy_nac_lh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_nac_ll_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_nac_ll_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpy_nac_ll_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_nac_ll_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_nac_ll_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpy_nac_ll_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_acc_sat_hh_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_acc_sat_hh_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpy_acc_sat_hh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_acc_sat_hh_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_acc_sat_hh_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpy_acc_sat_hh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_acc_sat_hl_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_acc_sat_hl_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpy_acc_sat_hl_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_acc_sat_hl_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_acc_sat_hl_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpy_acc_sat_hl_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_acc_sat_lh_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_acc_sat_lh_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpy_acc_sat_lh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_acc_sat_lh_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_acc_sat_lh_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpy_acc_sat_lh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_acc_sat_ll_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_acc_sat_ll_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpy_acc_sat_ll_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_acc_sat_ll_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_acc_sat_ll_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpy_acc_sat_ll_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_nac_sat_hh_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_nac_sat_hh_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpy_nac_sat_hh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_nac_sat_hh_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_nac_sat_hh_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpy_nac_sat_hh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_nac_sat_hl_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_nac_sat_hl_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpy_nac_sat_hl_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_nac_sat_hl_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_nac_sat_hl_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpy_nac_sat_hl_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_nac_sat_lh_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_nac_sat_lh_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpy_nac_sat_lh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_nac_sat_lh_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_nac_sat_lh_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpy_nac_sat_lh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_nac_sat_ll_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_nac_sat_ll_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpy_nac_sat_ll_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_nac_sat_ll_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_nac_sat_ll_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpy_nac_sat_ll_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_hh_s0,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_hh_s0 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_hh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_hh_s1,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_hh_s1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_hh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_hl_s0,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_hl_s0 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_hl_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_hl_s1,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_hl_s1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_hl_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_lh_s0,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_lh_s0 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_lh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_lh_s1,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_lh_s1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_lh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_ll_s0,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_ll_s0 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_ll_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_ll_s1,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_ll_s1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_ll_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_sat_hh_s0,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_sat_hh_s0 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_sat_hh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_sat_hh_s1,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_sat_hh_s1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_sat_hh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_sat_hl_s0,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_sat_hl_s0 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_sat_hl_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_sat_hl_s1,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_sat_hl_s1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_sat_hl_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_sat_lh_s0,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_sat_lh_s0 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_sat_lh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_sat_lh_s1,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_sat_lh_s1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_sat_lh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_sat_ll_s0,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_sat_ll_s0 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_sat_ll_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_sat_ll_s1,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_sat_ll_s1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_sat_ll_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_rnd_hh_s0,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_rnd_hh_s0 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_rnd_hh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_rnd_hh_s1,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_rnd_hh_s1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_rnd_hh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_rnd_hl_s0,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_rnd_hl_s0 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_rnd_hl_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_rnd_hl_s1,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_rnd_hl_s1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_rnd_hl_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_rnd_lh_s0,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_rnd_lh_s0 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_rnd_lh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_rnd_lh_s1,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_rnd_lh_s1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_rnd_lh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_rnd_ll_s0,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_rnd_ll_s0 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_rnd_ll_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_rnd_ll_s1,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_rnd_ll_s1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_rnd_ll_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_sat_rnd_hh_s0,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_sat_rnd_hh_s0 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_sat_rnd_hh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_sat_rnd_hh_s1,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_sat_rnd_hh_s1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_sat_rnd_hh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_sat_rnd_hl_s0,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_sat_rnd_hl_s0 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_sat_rnd_hl_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_sat_rnd_hl_s1,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_sat_rnd_hl_s1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_sat_rnd_hl_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_sat_rnd_lh_s0,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_sat_rnd_lh_s0 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_sat_rnd_lh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_sat_rnd_lh_s1,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_sat_rnd_lh_s1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_sat_rnd_lh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_sat_rnd_ll_s0,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_sat_rnd_ll_s0 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_sat_rnd_ll_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_sat_rnd_ll_s1,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_sat_rnd_ll_s1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_sat_rnd_ll_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_acc_hh_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyd_acc_hh_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_mpyd_acc_hh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_acc_hh_s1,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyd_acc_hh_s1 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_mpyd_acc_hh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_acc_hl_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyd_acc_hl_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_mpyd_acc_hl_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_acc_hl_s1,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyd_acc_hl_s1 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_mpyd_acc_hl_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_acc_lh_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyd_acc_lh_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_mpyd_acc_lh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_acc_lh_s1,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyd_acc_lh_s1 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_mpyd_acc_lh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_acc_ll_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyd_acc_ll_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_mpyd_acc_ll_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_acc_ll_s1,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyd_acc_ll_s1 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_mpyd_acc_ll_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_nac_hh_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyd_nac_hh_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_mpyd_nac_hh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_nac_hh_s1,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyd_nac_hh_s1 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_mpyd_nac_hh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_nac_hl_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyd_nac_hl_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_mpyd_nac_hl_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_nac_hl_s1,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyd_nac_hl_s1 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_mpyd_nac_hl_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_nac_lh_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyd_nac_lh_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_mpyd_nac_lh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_nac_lh_s1,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyd_nac_lh_s1 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_mpyd_nac_lh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_nac_ll_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyd_nac_ll_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_mpyd_nac_ll_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_nac_ll_s1,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyd_nac_ll_s1 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_mpyd_nac_ll_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_hh_s0,DI_ftype_SISI,2)
//
def int_hexagon_M2_mpyd_hh_s0 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_mpyd_hh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_hh_s1,DI_ftype_SISI,2)
//
def int_hexagon_M2_mpyd_hh_s1 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_mpyd_hh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_hl_s0,DI_ftype_SISI,2)
//
def int_hexagon_M2_mpyd_hl_s0 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_mpyd_hl_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_hl_s1,DI_ftype_SISI,2)
//
def int_hexagon_M2_mpyd_hl_s1 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_mpyd_hl_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_lh_s0,DI_ftype_SISI,2)
//
def int_hexagon_M2_mpyd_lh_s0 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_mpyd_lh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_lh_s1,DI_ftype_SISI,2)
//
def int_hexagon_M2_mpyd_lh_s1 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_mpyd_lh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_ll_s0,DI_ftype_SISI,2)
//
def int_hexagon_M2_mpyd_ll_s0 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_mpyd_ll_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_ll_s1,DI_ftype_SISI,2)
//
def int_hexagon_M2_mpyd_ll_s1 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_mpyd_ll_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_rnd_hh_s0,DI_ftype_SISI,2)
//
def int_hexagon_M2_mpyd_rnd_hh_s0 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_mpyd_rnd_hh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_rnd_hh_s1,DI_ftype_SISI,2)
//
def int_hexagon_M2_mpyd_rnd_hh_s1 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_mpyd_rnd_hh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_rnd_hl_s0,DI_ftype_SISI,2)
//
def int_hexagon_M2_mpyd_rnd_hl_s0 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_mpyd_rnd_hl_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_rnd_hl_s1,DI_ftype_SISI,2)
//
def int_hexagon_M2_mpyd_rnd_hl_s1 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_mpyd_rnd_hl_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_rnd_lh_s0,DI_ftype_SISI,2)
//
def int_hexagon_M2_mpyd_rnd_lh_s0 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_mpyd_rnd_lh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_rnd_lh_s1,DI_ftype_SISI,2)
//
def int_hexagon_M2_mpyd_rnd_lh_s1 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_mpyd_rnd_lh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_rnd_ll_s0,DI_ftype_SISI,2)
//
def int_hexagon_M2_mpyd_rnd_ll_s0 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_mpyd_rnd_ll_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_rnd_ll_s1,DI_ftype_SISI,2)
//
def int_hexagon_M2_mpyd_rnd_ll_s1 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_mpyd_rnd_ll_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_acc_hh_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpyu_acc_hh_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpyu_acc_hh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_acc_hh_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpyu_acc_hh_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpyu_acc_hh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_acc_hl_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpyu_acc_hl_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpyu_acc_hl_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_acc_hl_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpyu_acc_hl_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpyu_acc_hl_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_acc_lh_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpyu_acc_lh_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpyu_acc_lh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_acc_lh_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpyu_acc_lh_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpyu_acc_lh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_acc_ll_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpyu_acc_ll_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpyu_acc_ll_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_acc_ll_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpyu_acc_ll_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpyu_acc_ll_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_nac_hh_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpyu_nac_hh_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpyu_nac_hh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_nac_hh_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpyu_nac_hh_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpyu_nac_hh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_nac_hl_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpyu_nac_hl_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpyu_nac_hl_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_nac_hl_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpyu_nac_hl_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpyu_nac_hl_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_nac_lh_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpyu_nac_lh_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpyu_nac_lh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_nac_lh_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpyu_nac_lh_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpyu_nac_lh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_nac_ll_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpyu_nac_ll_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpyu_nac_ll_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_nac_ll_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpyu_nac_ll_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_mpyu_nac_ll_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_hh_s0,USI_ftype_SISI,2)
//
def int_hexagon_M2_mpyu_hh_s0 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpyu_hh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_hh_s1,USI_ftype_SISI,2)
//
def int_hexagon_M2_mpyu_hh_s1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpyu_hh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_hl_s0,USI_ftype_SISI,2)
//
def int_hexagon_M2_mpyu_hl_s0 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpyu_hl_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_hl_s1,USI_ftype_SISI,2)
//
def int_hexagon_M2_mpyu_hl_s1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpyu_hl_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_lh_s0,USI_ftype_SISI,2)
//
def int_hexagon_M2_mpyu_lh_s0 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpyu_lh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_lh_s1,USI_ftype_SISI,2)
//
def int_hexagon_M2_mpyu_lh_s1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpyu_lh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_ll_s0,USI_ftype_SISI,2)
//
def int_hexagon_M2_mpyu_ll_s0 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpyu_ll_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_ll_s1,USI_ftype_SISI,2)
//
def int_hexagon_M2_mpyu_ll_s1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpyu_ll_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_acc_hh_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyud_acc_hh_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_mpyud_acc_hh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_acc_hh_s1,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyud_acc_hh_s1 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_mpyud_acc_hh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_acc_hl_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyud_acc_hl_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_mpyud_acc_hl_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_acc_hl_s1,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyud_acc_hl_s1 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_mpyud_acc_hl_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_acc_lh_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyud_acc_lh_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_mpyud_acc_lh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_acc_lh_s1,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyud_acc_lh_s1 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_mpyud_acc_lh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_acc_ll_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyud_acc_ll_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_mpyud_acc_ll_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_acc_ll_s1,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyud_acc_ll_s1 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_mpyud_acc_ll_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_nac_hh_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyud_nac_hh_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_mpyud_nac_hh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_nac_hh_s1,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyud_nac_hh_s1 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_mpyud_nac_hh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_nac_hl_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyud_nac_hl_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_mpyud_nac_hl_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_nac_hl_s1,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyud_nac_hl_s1 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_mpyud_nac_hl_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_nac_lh_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyud_nac_lh_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_mpyud_nac_lh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_nac_lh_s1,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyud_nac_lh_s1 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_mpyud_nac_lh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_nac_ll_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyud_nac_ll_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_mpyud_nac_ll_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_nac_ll_s1,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyud_nac_ll_s1 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_mpyud_nac_ll_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_hh_s0,UDI_ftype_SISI,2)
//
def int_hexagon_M2_mpyud_hh_s0 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_mpyud_hh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_hh_s1,UDI_ftype_SISI,2)
//
def int_hexagon_M2_mpyud_hh_s1 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_mpyud_hh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_hl_s0,UDI_ftype_SISI,2)
//
def int_hexagon_M2_mpyud_hl_s0 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_mpyud_hl_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_hl_s1,UDI_ftype_SISI,2)
//
def int_hexagon_M2_mpyud_hl_s1 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_mpyud_hl_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_lh_s0,UDI_ftype_SISI,2)
//
def int_hexagon_M2_mpyud_lh_s0 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_mpyud_lh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_lh_s1,UDI_ftype_SISI,2)
//
def int_hexagon_M2_mpyud_lh_s1 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_mpyud_lh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_ll_s0,UDI_ftype_SISI,2)
//
def int_hexagon_M2_mpyud_ll_s0 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_mpyud_ll_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_ll_s1,UDI_ftype_SISI,2)
//
def int_hexagon_M2_mpyud_ll_s1 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_mpyud_ll_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpysmi,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpysmi :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpysmi">;
//
// BUILTIN_INFO(HEXAGON.M2_macsip,SI_ftype_SISISI,3)
//
def int_hexagon_M2_macsip :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_macsip">;
//
// BUILTIN_INFO(HEXAGON.M2_macsin,SI_ftype_SISISI,3)
//
def int_hexagon_M2_macsin :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_macsin">;
//
// BUILTIN_INFO(HEXAGON.M2_dpmpyss_s0,DI_ftype_SISI,2)
//
def int_hexagon_M2_dpmpyss_s0 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_dpmpyss_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_dpmpyss_acc_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_dpmpyss_acc_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_dpmpyss_acc_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_dpmpyss_nac_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_dpmpyss_nac_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_dpmpyss_nac_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_dpmpyuu_s0,UDI_ftype_SISI,2)
//
def int_hexagon_M2_dpmpyuu_s0 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_dpmpyuu_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_dpmpyuu_acc_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_dpmpyuu_acc_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_dpmpyuu_acc_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_dpmpyuu_nac_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_dpmpyuu_nac_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_dpmpyuu_nac_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_up,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_up :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_up">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_up_s1,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_up_s1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_up_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_up_s1_sat,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_up_s1_sat :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpy_up_s1_sat">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_up,USI_ftype_SISI,2)
//
def int_hexagon_M2_mpyu_up :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpyu_up">;
//
// BUILTIN_INFO(HEXAGON.M2_mpysu_up,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpysu_up :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpysu_up">;
//
// BUILTIN_INFO(HEXAGON.M2_dpmpyss_rnd_s0,SI_ftype_SISI,2)
//
def int_hexagon_M2_dpmpyss_rnd_s0 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_dpmpyss_rnd_s0">;
//
// BUILTIN_INFO(HEXAGON.M4_mac_up_s1_sat,SI_ftype_SISISI,3)
//
def int_hexagon_M4_mac_up_s1_sat :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M4_mac_up_s1_sat">;
//
// BUILTIN_INFO(HEXAGON.M4_nac_up_s1_sat,SI_ftype_SISISI,3)
//
def int_hexagon_M4_nac_up_s1_sat :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M4_nac_up_s1_sat">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyi,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpyi :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpyi">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyui,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpyui :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_mpyui">;
//
// BUILTIN_INFO(HEXAGON.M2_maci,SI_ftype_SISISI,3)
//
def int_hexagon_M2_maci :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_maci">;
//
// BUILTIN_INFO(HEXAGON.M2_acci,SI_ftype_SISISI,3)
//
def int_hexagon_M2_acci :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_acci">;
//
// BUILTIN_INFO(HEXAGON.M2_accii,SI_ftype_SISISI,3)
//
def int_hexagon_M2_accii :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_accii">;
//
// BUILTIN_INFO(HEXAGON.M2_nacci,SI_ftype_SISISI,3)
//
def int_hexagon_M2_nacci :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_nacci">;
//
// BUILTIN_INFO(HEXAGON.M2_naccii,SI_ftype_SISISI,3)
//
def int_hexagon_M2_naccii :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_naccii">;
//
// BUILTIN_INFO(HEXAGON.M2_subacc,SI_ftype_SISISI,3)
//
def int_hexagon_M2_subacc :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_subacc">;
//
// BUILTIN_INFO(HEXAGON.M4_mpyrr_addr,SI_ftype_SISISI,3)
//
def int_hexagon_M4_mpyrr_addr :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M4_mpyrr_addr">;
//
// BUILTIN_INFO(HEXAGON.M4_mpyri_addr_u2,SI_ftype_SISISI,3)
//
def int_hexagon_M4_mpyri_addr_u2 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M4_mpyri_addr_u2">;
//
// BUILTIN_INFO(HEXAGON.M4_mpyri_addr,SI_ftype_SISISI,3)
//
def int_hexagon_M4_mpyri_addr :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M4_mpyri_addr">;
//
// BUILTIN_INFO(HEXAGON.M4_mpyri_addi,SI_ftype_SISISI,3)
//
def int_hexagon_M4_mpyri_addi :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M4_mpyri_addi">;
//
// BUILTIN_INFO(HEXAGON.M4_mpyrr_addi,SI_ftype_SISISI,3)
//
def int_hexagon_M4_mpyrr_addi :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M4_mpyrr_addi">;
//
// BUILTIN_INFO(HEXAGON.M2_vmpy2s_s0,DI_ftype_SISI,2)
//
def int_hexagon_M2_vmpy2s_s0 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_vmpy2s_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_vmpy2s_s1,DI_ftype_SISI,2)
//
def int_hexagon_M2_vmpy2s_s1 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_vmpy2s_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_vmac2s_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_vmac2s_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_vmac2s_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_vmac2s_s1,DI_ftype_DISISI,3)
//
def int_hexagon_M2_vmac2s_s1 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_vmac2s_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_vmpy2su_s0,DI_ftype_SISI,2)
//
def int_hexagon_M2_vmpy2su_s0 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_vmpy2su_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_vmpy2su_s1,DI_ftype_SISI,2)
//
def int_hexagon_M2_vmpy2su_s1 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_vmpy2su_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_vmac2su_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_vmac2su_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_vmac2su_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_vmac2su_s1,DI_ftype_DISISI,3)
//
def int_hexagon_M2_vmac2su_s1 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_vmac2su_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_vmpy2s_s0pack,SI_ftype_SISI,2)
//
def int_hexagon_M2_vmpy2s_s0pack :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_vmpy2s_s0pack">;
//
// BUILTIN_INFO(HEXAGON.M2_vmpy2s_s1pack,SI_ftype_SISI,2)
//
def int_hexagon_M2_vmpy2s_s1pack :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_vmpy2s_s1pack">;
//
// BUILTIN_INFO(HEXAGON.M2_vmac2,DI_ftype_DISISI,3)
//
def int_hexagon_M2_vmac2 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_vmac2">;
//
// BUILTIN_INFO(HEXAGON.M2_vmpy2es_s0,DI_ftype_DIDI,2)
//
def int_hexagon_M2_vmpy2es_s0 :
Hexagon_di_didi_Intrinsic<"HEXAGON_M2_vmpy2es_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_vmpy2es_s1,DI_ftype_DIDI,2)
//
def int_hexagon_M2_vmpy2es_s1 :
Hexagon_di_didi_Intrinsic<"HEXAGON_M2_vmpy2es_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_vmac2es_s0,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_vmac2es_s0 :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M2_vmac2es_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_vmac2es_s1,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_vmac2es_s1 :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M2_vmac2es_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_vmac2es,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_vmac2es :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M2_vmac2es">;
//
// BUILTIN_INFO(HEXAGON.M2_vrmac_s0,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_vrmac_s0 :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M2_vrmac_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_vrmpy_s0,DI_ftype_DIDI,2)
//
def int_hexagon_M2_vrmpy_s0 :
Hexagon_di_didi_Intrinsic<"HEXAGON_M2_vrmpy_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_vdmpyrs_s0,SI_ftype_DIDI,2)
//
def int_hexagon_M2_vdmpyrs_s0 :
Hexagon_si_didi_Intrinsic<"HEXAGON_M2_vdmpyrs_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_vdmpyrs_s1,SI_ftype_DIDI,2)
//
def int_hexagon_M2_vdmpyrs_s1 :
Hexagon_si_didi_Intrinsic<"HEXAGON_M2_vdmpyrs_s1">;
//
// BUILTIN_INFO(HEXAGON.M5_vrmpybuu,DI_ftype_DIDI,2)
//
def int_hexagon_M5_vrmpybuu :
Hexagon_di_didi_Intrinsic<"HEXAGON_M5_vrmpybuu">;
//
// BUILTIN_INFO(HEXAGON.M5_vrmacbuu,DI_ftype_DIDIDI,3)
//
def int_hexagon_M5_vrmacbuu :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M5_vrmacbuu">;
//
// BUILTIN_INFO(HEXAGON.M5_vrmpybsu,DI_ftype_DIDI,2)
//
def int_hexagon_M5_vrmpybsu :
Hexagon_di_didi_Intrinsic<"HEXAGON_M5_vrmpybsu">;
//
// BUILTIN_INFO(HEXAGON.M5_vrmacbsu,DI_ftype_DIDIDI,3)
//
def int_hexagon_M5_vrmacbsu :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M5_vrmacbsu">;
//
// BUILTIN_INFO(HEXAGON.M5_vmpybuu,DI_ftype_SISI,2)
//
def int_hexagon_M5_vmpybuu :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M5_vmpybuu">;
//
// BUILTIN_INFO(HEXAGON.M5_vmpybsu,DI_ftype_SISI,2)
//
def int_hexagon_M5_vmpybsu :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M5_vmpybsu">;
//
// BUILTIN_INFO(HEXAGON.M5_vmacbuu,DI_ftype_DISISI,3)
//
def int_hexagon_M5_vmacbuu :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M5_vmacbuu">;
//
// BUILTIN_INFO(HEXAGON.M5_vmacbsu,DI_ftype_DISISI,3)
//
def int_hexagon_M5_vmacbsu :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M5_vmacbsu">;
//
// BUILTIN_INFO(HEXAGON.M5_vdmpybsu,DI_ftype_DIDI,2)
//
def int_hexagon_M5_vdmpybsu :
Hexagon_di_didi_Intrinsic<"HEXAGON_M5_vdmpybsu">;
//
// BUILTIN_INFO(HEXAGON.M5_vdmacbsu,DI_ftype_DIDIDI,3)
//
def int_hexagon_M5_vdmacbsu :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M5_vdmacbsu">;
//
// BUILTIN_INFO(HEXAGON.M2_vdmacs_s0,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_vdmacs_s0 :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M2_vdmacs_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_vdmacs_s1,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_vdmacs_s1 :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M2_vdmacs_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_vdmpys_s0,DI_ftype_DIDI,2)
//
def int_hexagon_M2_vdmpys_s0 :
Hexagon_di_didi_Intrinsic<"HEXAGON_M2_vdmpys_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_vdmpys_s1,DI_ftype_DIDI,2)
//
def int_hexagon_M2_vdmpys_s1 :
Hexagon_di_didi_Intrinsic<"HEXAGON_M2_vdmpys_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_cmpyrs_s0,SI_ftype_SISI,2)
//
def int_hexagon_M2_cmpyrs_s0 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_cmpyrs_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_cmpyrs_s1,SI_ftype_SISI,2)
//
def int_hexagon_M2_cmpyrs_s1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_cmpyrs_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_cmpyrsc_s0,SI_ftype_SISI,2)
//
def int_hexagon_M2_cmpyrsc_s0 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_cmpyrsc_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_cmpyrsc_s1,SI_ftype_SISI,2)
//
def int_hexagon_M2_cmpyrsc_s1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_cmpyrsc_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_cmacs_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_cmacs_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_cmacs_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_cmacs_s1,DI_ftype_DISISI,3)
//
def int_hexagon_M2_cmacs_s1 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_cmacs_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_cmacsc_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_cmacsc_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_cmacsc_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_cmacsc_s1,DI_ftype_DISISI,3)
//
def int_hexagon_M2_cmacsc_s1 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_cmacsc_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_cmpys_s0,DI_ftype_SISI,2)
//
def int_hexagon_M2_cmpys_s0 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_cmpys_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_cmpys_s1,DI_ftype_SISI,2)
//
def int_hexagon_M2_cmpys_s1 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_cmpys_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_cmpysc_s0,DI_ftype_SISI,2)
//
def int_hexagon_M2_cmpysc_s0 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_cmpysc_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_cmpysc_s1,DI_ftype_SISI,2)
//
def int_hexagon_M2_cmpysc_s1 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_cmpysc_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_cnacs_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_cnacs_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_cnacs_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_cnacs_s1,DI_ftype_DISISI,3)
//
def int_hexagon_M2_cnacs_s1 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_cnacs_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_cnacsc_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_cnacsc_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_cnacsc_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_cnacsc_s1,DI_ftype_DISISI,3)
//
def int_hexagon_M2_cnacsc_s1 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_cnacsc_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_vrcmpys_s1,DI_ftype_DISI,2)
//
def int_hexagon_M2_vrcmpys_s1 :
Hexagon_di_disi_Intrinsic<"HEXAGON_M2_vrcmpys_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_vrcmpys_acc_s1,DI_ftype_DIDISI,3)
//
def int_hexagon_M2_vrcmpys_acc_s1 :
Hexagon_di_didisi_Intrinsic<"HEXAGON_M2_vrcmpys_acc_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_vrcmpys_s1rp,SI_ftype_DISI,2)
//
def int_hexagon_M2_vrcmpys_s1rp :
Hexagon_si_disi_Intrinsic<"HEXAGON_M2_vrcmpys_s1rp">;
//
// BUILTIN_INFO(HEXAGON.M2_mmacls_s0,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_mmacls_s0 :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M2_mmacls_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mmacls_s1,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_mmacls_s1 :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M2_mmacls_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mmachs_s0,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_mmachs_s0 :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M2_mmachs_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mmachs_s1,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_mmachs_s1 :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M2_mmachs_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mmpyl_s0,DI_ftype_DIDI,2)
//
def int_hexagon_M2_mmpyl_s0 :
Hexagon_di_didi_Intrinsic<"HEXAGON_M2_mmpyl_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mmpyl_s1,DI_ftype_DIDI,2)
//
def int_hexagon_M2_mmpyl_s1 :
Hexagon_di_didi_Intrinsic<"HEXAGON_M2_mmpyl_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mmpyh_s0,DI_ftype_DIDI,2)
//
def int_hexagon_M2_mmpyh_s0 :
Hexagon_di_didi_Intrinsic<"HEXAGON_M2_mmpyh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mmpyh_s1,DI_ftype_DIDI,2)
//
def int_hexagon_M2_mmpyh_s1 :
Hexagon_di_didi_Intrinsic<"HEXAGON_M2_mmpyh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mmacls_rs0,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_mmacls_rs0 :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M2_mmacls_rs0">;
//
// BUILTIN_INFO(HEXAGON.M2_mmacls_rs1,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_mmacls_rs1 :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M2_mmacls_rs1">;
//
// BUILTIN_INFO(HEXAGON.M2_mmachs_rs0,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_mmachs_rs0 :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M2_mmachs_rs0">;
//
// BUILTIN_INFO(HEXAGON.M2_mmachs_rs1,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_mmachs_rs1 :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M2_mmachs_rs1">;
//
// BUILTIN_INFO(HEXAGON.M2_mmpyl_rs0,DI_ftype_DIDI,2)
//
def int_hexagon_M2_mmpyl_rs0 :
Hexagon_di_didi_Intrinsic<"HEXAGON_M2_mmpyl_rs0">;
//
// BUILTIN_INFO(HEXAGON.M2_mmpyl_rs1,DI_ftype_DIDI,2)
//
def int_hexagon_M2_mmpyl_rs1 :
Hexagon_di_didi_Intrinsic<"HEXAGON_M2_mmpyl_rs1">;
//
// BUILTIN_INFO(HEXAGON.M2_mmpyh_rs0,DI_ftype_DIDI,2)
//
def int_hexagon_M2_mmpyh_rs0 :
Hexagon_di_didi_Intrinsic<"HEXAGON_M2_mmpyh_rs0">;
//
// BUILTIN_INFO(HEXAGON.M2_mmpyh_rs1,DI_ftype_DIDI,2)
//
def int_hexagon_M2_mmpyh_rs1 :
Hexagon_di_didi_Intrinsic<"HEXAGON_M2_mmpyh_rs1">;
//
// BUILTIN_INFO(HEXAGON.M4_vrmpyeh_s0,DI_ftype_DIDI,2)
//
def int_hexagon_M4_vrmpyeh_s0 :
Hexagon_di_didi_Intrinsic<"HEXAGON_M4_vrmpyeh_s0">;
//
// BUILTIN_INFO(HEXAGON.M4_vrmpyeh_s1,DI_ftype_DIDI,2)
//
def int_hexagon_M4_vrmpyeh_s1 :
Hexagon_di_didi_Intrinsic<"HEXAGON_M4_vrmpyeh_s1">;
//
// BUILTIN_INFO(HEXAGON.M4_vrmpyeh_acc_s0,DI_ftype_DIDIDI,3)
//
def int_hexagon_M4_vrmpyeh_acc_s0 :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M4_vrmpyeh_acc_s0">;
//
// BUILTIN_INFO(HEXAGON.M4_vrmpyeh_acc_s1,DI_ftype_DIDIDI,3)
//
def int_hexagon_M4_vrmpyeh_acc_s1 :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M4_vrmpyeh_acc_s1">;
//
// BUILTIN_INFO(HEXAGON.M4_vrmpyoh_s0,DI_ftype_DIDI,2)
//
def int_hexagon_M4_vrmpyoh_s0 :
Hexagon_di_didi_Intrinsic<"HEXAGON_M4_vrmpyoh_s0">;
//
// BUILTIN_INFO(HEXAGON.M4_vrmpyoh_s1,DI_ftype_DIDI,2)
//
def int_hexagon_M4_vrmpyoh_s1 :
Hexagon_di_didi_Intrinsic<"HEXAGON_M4_vrmpyoh_s1">;
//
// BUILTIN_INFO(HEXAGON.M4_vrmpyoh_acc_s0,DI_ftype_DIDIDI,3)
//
def int_hexagon_M4_vrmpyoh_acc_s0 :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M4_vrmpyoh_acc_s0">;
//
// BUILTIN_INFO(HEXAGON.M4_vrmpyoh_acc_s1,DI_ftype_DIDIDI,3)
//
def int_hexagon_M4_vrmpyoh_acc_s1 :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M4_vrmpyoh_acc_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_hmmpyl_rs1,SI_ftype_SISI,2)
//
def int_hexagon_M2_hmmpyl_rs1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_hmmpyl_rs1">;
//
// BUILTIN_INFO(HEXAGON.M2_hmmpyh_rs1,SI_ftype_SISI,2)
//
def int_hexagon_M2_hmmpyh_rs1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_hmmpyh_rs1">;
//
// BUILTIN_INFO(HEXAGON.M2_hmmpyl_s1,SI_ftype_SISI,2)
//
def int_hexagon_M2_hmmpyl_s1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_hmmpyl_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_hmmpyh_s1,SI_ftype_SISI,2)
//
def int_hexagon_M2_hmmpyh_s1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON_M2_hmmpyh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mmaculs_s0,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_mmaculs_s0 :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M2_mmaculs_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mmaculs_s1,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_mmaculs_s1 :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M2_mmaculs_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mmacuhs_s0,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_mmacuhs_s0 :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M2_mmacuhs_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mmacuhs_s1,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_mmacuhs_s1 :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M2_mmacuhs_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mmpyul_s0,DI_ftype_DIDI,2)
//
def int_hexagon_M2_mmpyul_s0 :
Hexagon_di_didi_Intrinsic<"HEXAGON_M2_mmpyul_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mmpyul_s1,DI_ftype_DIDI,2)
//
def int_hexagon_M2_mmpyul_s1 :
Hexagon_di_didi_Intrinsic<"HEXAGON_M2_mmpyul_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mmpyuh_s0,DI_ftype_DIDI,2)
//
def int_hexagon_M2_mmpyuh_s0 :
Hexagon_di_didi_Intrinsic<"HEXAGON_M2_mmpyuh_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mmpyuh_s1,DI_ftype_DIDI,2)
//
def int_hexagon_M2_mmpyuh_s1 :
Hexagon_di_didi_Intrinsic<"HEXAGON_M2_mmpyuh_s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mmaculs_rs0,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_mmaculs_rs0 :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M2_mmaculs_rs0">;
//
// BUILTIN_INFO(HEXAGON.M2_mmaculs_rs1,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_mmaculs_rs1 :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M2_mmaculs_rs1">;
//
// BUILTIN_INFO(HEXAGON.M2_mmacuhs_rs0,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_mmacuhs_rs0 :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M2_mmacuhs_rs0">;
//
// BUILTIN_INFO(HEXAGON.M2_mmacuhs_rs1,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_mmacuhs_rs1 :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M2_mmacuhs_rs1">;
//
// BUILTIN_INFO(HEXAGON.M2_mmpyul_rs0,DI_ftype_DIDI,2)
//
def int_hexagon_M2_mmpyul_rs0 :
Hexagon_di_didi_Intrinsic<"HEXAGON_M2_mmpyul_rs0">;
//
// BUILTIN_INFO(HEXAGON.M2_mmpyul_rs1,DI_ftype_DIDI,2)
//
def int_hexagon_M2_mmpyul_rs1 :
Hexagon_di_didi_Intrinsic<"HEXAGON_M2_mmpyul_rs1">;
//
// BUILTIN_INFO(HEXAGON.M2_mmpyuh_rs0,DI_ftype_DIDI,2)
//
def int_hexagon_M2_mmpyuh_rs0 :
Hexagon_di_didi_Intrinsic<"HEXAGON_M2_mmpyuh_rs0">;
//
// BUILTIN_INFO(HEXAGON.M2_mmpyuh_rs1,DI_ftype_DIDI,2)
//
def int_hexagon_M2_mmpyuh_rs1 :
Hexagon_di_didi_Intrinsic<"HEXAGON_M2_mmpyuh_rs1">;
//
// BUILTIN_INFO(HEXAGON.M2_vrcmaci_s0,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_vrcmaci_s0 :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M2_vrcmaci_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_vrcmacr_s0,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_vrcmacr_s0 :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M2_vrcmacr_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_vrcmaci_s0c,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_vrcmaci_s0c :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M2_vrcmaci_s0c">;
//
// BUILTIN_INFO(HEXAGON.M2_vrcmacr_s0c,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_vrcmacr_s0c :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M2_vrcmacr_s0c">;
//
// BUILTIN_INFO(HEXAGON.M2_cmaci_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_cmaci_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_cmaci_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_cmacr_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_cmacr_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M2_cmacr_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_vrcmpyi_s0,DI_ftype_DIDI,2)
//
def int_hexagon_M2_vrcmpyi_s0 :
Hexagon_di_didi_Intrinsic<"HEXAGON_M2_vrcmpyi_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_vrcmpyr_s0,DI_ftype_DIDI,2)
//
def int_hexagon_M2_vrcmpyr_s0 :
Hexagon_di_didi_Intrinsic<"HEXAGON_M2_vrcmpyr_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_vrcmpyi_s0c,DI_ftype_DIDI,2)
//
def int_hexagon_M2_vrcmpyi_s0c :
Hexagon_di_didi_Intrinsic<"HEXAGON_M2_vrcmpyi_s0c">;
//
// BUILTIN_INFO(HEXAGON.M2_vrcmpyr_s0c,DI_ftype_DIDI,2)
//
def int_hexagon_M2_vrcmpyr_s0c :
Hexagon_di_didi_Intrinsic<"HEXAGON_M2_vrcmpyr_s0c">;
//
// BUILTIN_INFO(HEXAGON.M2_cmpyi_s0,DI_ftype_SISI,2)
//
def int_hexagon_M2_cmpyi_s0 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_cmpyi_s0">;
//
// BUILTIN_INFO(HEXAGON.M2_cmpyr_s0,DI_ftype_SISI,2)
//
def int_hexagon_M2_cmpyr_s0 :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M2_cmpyr_s0">;
//
// BUILTIN_INFO(HEXAGON.M4_cmpyi_wh,SI_ftype_DISI,2)
//
def int_hexagon_M4_cmpyi_wh :
Hexagon_si_disi_Intrinsic<"HEXAGON_M4_cmpyi_wh">;
//
// BUILTIN_INFO(HEXAGON.M4_cmpyr_wh,SI_ftype_DISI,2)
//
def int_hexagon_M4_cmpyr_wh :
Hexagon_si_disi_Intrinsic<"HEXAGON_M4_cmpyr_wh">;
//
// BUILTIN_INFO(HEXAGON.M4_cmpyi_whc,SI_ftype_DISI,2)
//
def int_hexagon_M4_cmpyi_whc :
Hexagon_si_disi_Intrinsic<"HEXAGON_M4_cmpyi_whc">;
//
// BUILTIN_INFO(HEXAGON.M4_cmpyr_whc,SI_ftype_DISI,2)
//
def int_hexagon_M4_cmpyr_whc :
Hexagon_si_disi_Intrinsic<"HEXAGON_M4_cmpyr_whc">;
//
// BUILTIN_INFO(HEXAGON.M2_vcmpy_s0_sat_i,DI_ftype_DIDI,2)
//
def int_hexagon_M2_vcmpy_s0_sat_i :
Hexagon_di_didi_Intrinsic<"HEXAGON_M2_vcmpy_s0_sat_i">;
//
// BUILTIN_INFO(HEXAGON.M2_vcmpy_s0_sat_r,DI_ftype_DIDI,2)
//
def int_hexagon_M2_vcmpy_s0_sat_r :
Hexagon_di_didi_Intrinsic<"HEXAGON_M2_vcmpy_s0_sat_r">;
//
// BUILTIN_INFO(HEXAGON.M2_vcmpy_s1_sat_i,DI_ftype_DIDI,2)
//
def int_hexagon_M2_vcmpy_s1_sat_i :
Hexagon_di_didi_Intrinsic<"HEXAGON_M2_vcmpy_s1_sat_i">;
//
// BUILTIN_INFO(HEXAGON.M2_vcmpy_s1_sat_r,DI_ftype_DIDI,2)
//
def int_hexagon_M2_vcmpy_s1_sat_r :
Hexagon_di_didi_Intrinsic<"HEXAGON_M2_vcmpy_s1_sat_r">;
//
// BUILTIN_INFO(HEXAGON.M2_vcmac_s0_sat_i,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_vcmac_s0_sat_i :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M2_vcmac_s0_sat_i">;
//
// BUILTIN_INFO(HEXAGON.M2_vcmac_s0_sat_r,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_vcmac_s0_sat_r :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M2_vcmac_s0_sat_r">;
//
// BUILTIN_INFO(HEXAGON.S2_vcrotate,DI_ftype_DISI,2)
//
def int_hexagon_S2_vcrotate :
Hexagon_di_disi_Intrinsic<"HEXAGON_S2_vcrotate">;
//
// BUILTIN_INFO(HEXAGON.S4_vrcrotate_acc,DI_ftype_DIDISISI,4)
//
def int_hexagon_S4_vrcrotate_acc :
Hexagon_di_didisisi_Intrinsic<"HEXAGON_S4_vrcrotate_acc">;
//
// BUILTIN_INFO(HEXAGON.S4_vrcrotate,DI_ftype_DISISI,3)
//
def int_hexagon_S4_vrcrotate :
Hexagon_di_disisi_Intrinsic<"HEXAGON_S4_vrcrotate">;
//
// BUILTIN_INFO(HEXAGON.S2_vcnegh,DI_ftype_DISI,2)
//
def int_hexagon_S2_vcnegh :
Hexagon_di_disi_Intrinsic<"HEXAGON_S2_vcnegh">;
//
// BUILTIN_INFO(HEXAGON.S2_vrcnegh,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_vrcnegh :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_vrcnegh">;
//
// BUILTIN_INFO(HEXAGON.M4_pmpyw,DI_ftype_SISI,2)
//
def int_hexagon_M4_pmpyw :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M4_pmpyw">;
//
// BUILTIN_INFO(HEXAGON.M4_vpmpyh,DI_ftype_SISI,2)
//
def int_hexagon_M4_vpmpyh :
Hexagon_di_sisi_Intrinsic<"HEXAGON_M4_vpmpyh">;
//
// BUILTIN_INFO(HEXAGON.M4_pmpyw_acc,DI_ftype_DISISI,3)
//
def int_hexagon_M4_pmpyw_acc :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M4_pmpyw_acc">;
//
// BUILTIN_INFO(HEXAGON.M4_vpmpyh_acc,DI_ftype_DISISI,3)
//
def int_hexagon_M4_vpmpyh_acc :
Hexagon_di_disisi_Intrinsic<"HEXAGON_M4_vpmpyh_acc">;
//
// BUILTIN_INFO(HEXAGON.A2_add,SI_ftype_SISI,2)
//
def int_hexagon_A2_add :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_add">;
//
// BUILTIN_INFO(HEXAGON.A2_sub,SI_ftype_SISI,2)
//
def int_hexagon_A2_sub :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_sub">;
//
// BUILTIN_INFO(HEXAGON.A2_addsat,SI_ftype_SISI,2)
//
def int_hexagon_A2_addsat :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_addsat">;
//
// BUILTIN_INFO(HEXAGON.A2_subsat,SI_ftype_SISI,2)
//
def int_hexagon_A2_subsat :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_subsat">;
//
// BUILTIN_INFO(HEXAGON.A2_addi,SI_ftype_SISI,2)
//
def int_hexagon_A2_addi :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_addi">;
//
// BUILTIN_INFO(HEXAGON.A2_addh_l16_ll,SI_ftype_SISI,2)
//
def int_hexagon_A2_addh_l16_ll :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_addh_l16_ll">;
//
// BUILTIN_INFO(HEXAGON.A2_addh_l16_hl,SI_ftype_SISI,2)
//
def int_hexagon_A2_addh_l16_hl :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_addh_l16_hl">;
//
// BUILTIN_INFO(HEXAGON.A2_addh_l16_sat_ll,SI_ftype_SISI,2)
//
def int_hexagon_A2_addh_l16_sat_ll :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_addh_l16_sat_ll">;
//
// BUILTIN_INFO(HEXAGON.A2_addh_l16_sat_hl,SI_ftype_SISI,2)
//
def int_hexagon_A2_addh_l16_sat_hl :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_addh_l16_sat_hl">;
//
// BUILTIN_INFO(HEXAGON.A2_subh_l16_ll,SI_ftype_SISI,2)
//
def int_hexagon_A2_subh_l16_ll :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_subh_l16_ll">;
//
// BUILTIN_INFO(HEXAGON.A2_subh_l16_hl,SI_ftype_SISI,2)
//
def int_hexagon_A2_subh_l16_hl :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_subh_l16_hl">;
//
// BUILTIN_INFO(HEXAGON.A2_subh_l16_sat_ll,SI_ftype_SISI,2)
//
def int_hexagon_A2_subh_l16_sat_ll :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_subh_l16_sat_ll">;
//
// BUILTIN_INFO(HEXAGON.A2_subh_l16_sat_hl,SI_ftype_SISI,2)
//
def int_hexagon_A2_subh_l16_sat_hl :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_subh_l16_sat_hl">;
//
// BUILTIN_INFO(HEXAGON.A2_addh_h16_ll,SI_ftype_SISI,2)
//
def int_hexagon_A2_addh_h16_ll :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_addh_h16_ll">;
//
// BUILTIN_INFO(HEXAGON.A2_addh_h16_lh,SI_ftype_SISI,2)
//
def int_hexagon_A2_addh_h16_lh :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_addh_h16_lh">;
//
// BUILTIN_INFO(HEXAGON.A2_addh_h16_hl,SI_ftype_SISI,2)
//
def int_hexagon_A2_addh_h16_hl :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_addh_h16_hl">;
//
// BUILTIN_INFO(HEXAGON.A2_addh_h16_hh,SI_ftype_SISI,2)
//
def int_hexagon_A2_addh_h16_hh :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_addh_h16_hh">;
//
// BUILTIN_INFO(HEXAGON.A2_addh_h16_sat_ll,SI_ftype_SISI,2)
//
def int_hexagon_A2_addh_h16_sat_ll :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_addh_h16_sat_ll">;
//
// BUILTIN_INFO(HEXAGON.A2_addh_h16_sat_lh,SI_ftype_SISI,2)
//
def int_hexagon_A2_addh_h16_sat_lh :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_addh_h16_sat_lh">;
//
// BUILTIN_INFO(HEXAGON.A2_addh_h16_sat_hl,SI_ftype_SISI,2)
//
def int_hexagon_A2_addh_h16_sat_hl :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_addh_h16_sat_hl">;
//
// BUILTIN_INFO(HEXAGON.A2_addh_h16_sat_hh,SI_ftype_SISI,2)
//
def int_hexagon_A2_addh_h16_sat_hh :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_addh_h16_sat_hh">;
//
// BUILTIN_INFO(HEXAGON.A2_subh_h16_ll,SI_ftype_SISI,2)
//
def int_hexagon_A2_subh_h16_ll :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_subh_h16_ll">;
//
// BUILTIN_INFO(HEXAGON.A2_subh_h16_lh,SI_ftype_SISI,2)
//
def int_hexagon_A2_subh_h16_lh :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_subh_h16_lh">;
//
// BUILTIN_INFO(HEXAGON.A2_subh_h16_hl,SI_ftype_SISI,2)
//
def int_hexagon_A2_subh_h16_hl :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_subh_h16_hl">;
//
// BUILTIN_INFO(HEXAGON.A2_subh_h16_hh,SI_ftype_SISI,2)
//
def int_hexagon_A2_subh_h16_hh :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_subh_h16_hh">;
//
// BUILTIN_INFO(HEXAGON.A2_subh_h16_sat_ll,SI_ftype_SISI,2)
//
def int_hexagon_A2_subh_h16_sat_ll :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_subh_h16_sat_ll">;
//
// BUILTIN_INFO(HEXAGON.A2_subh_h16_sat_lh,SI_ftype_SISI,2)
//
def int_hexagon_A2_subh_h16_sat_lh :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_subh_h16_sat_lh">;
//
// BUILTIN_INFO(HEXAGON.A2_subh_h16_sat_hl,SI_ftype_SISI,2)
//
def int_hexagon_A2_subh_h16_sat_hl :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_subh_h16_sat_hl">;
//
// BUILTIN_INFO(HEXAGON.A2_subh_h16_sat_hh,SI_ftype_SISI,2)
//
def int_hexagon_A2_subh_h16_sat_hh :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_subh_h16_sat_hh">;
//
// BUILTIN_INFO(HEXAGON.A2_aslh,SI_ftype_SI,1)
//
def int_hexagon_A2_aslh :
Hexagon_si_si_Intrinsic<"HEXAGON_A2_aslh">;
//
// BUILTIN_INFO(HEXAGON.A2_asrh,SI_ftype_SI,1)
//
def int_hexagon_A2_asrh :
Hexagon_si_si_Intrinsic<"HEXAGON_A2_asrh">;
//
// BUILTIN_INFO(HEXAGON.A2_addp,DI_ftype_DIDI,2)
//
def int_hexagon_A2_addp :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_addp">;
//
// BUILTIN_INFO(HEXAGON.A2_addpsat,DI_ftype_DIDI,2)
//
def int_hexagon_A2_addpsat :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_addpsat">;
//
// BUILTIN_INFO(HEXAGON.A2_addsp,DI_ftype_SIDI,2)
//
def int_hexagon_A2_addsp :
Hexagon_di_sidi_Intrinsic<"HEXAGON_A2_addsp">;
//
// BUILTIN_INFO(HEXAGON.A2_subp,DI_ftype_DIDI,2)
//
def int_hexagon_A2_subp :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_subp">;
//
// BUILTIN_INFO(HEXAGON.A2_neg,SI_ftype_SI,1)
//
def int_hexagon_A2_neg :
Hexagon_si_si_Intrinsic<"HEXAGON_A2_neg">;
//
// BUILTIN_INFO(HEXAGON.A2_negsat,SI_ftype_SI,1)
//
def int_hexagon_A2_negsat :
Hexagon_si_si_Intrinsic<"HEXAGON_A2_negsat">;
//
// BUILTIN_INFO(HEXAGON.A2_abs,SI_ftype_SI,1)
//
def int_hexagon_A2_abs :
Hexagon_si_si_Intrinsic<"HEXAGON_A2_abs">;
//
// BUILTIN_INFO(HEXAGON.A2_abssat,SI_ftype_SI,1)
//
def int_hexagon_A2_abssat :
Hexagon_si_si_Intrinsic<"HEXAGON_A2_abssat">;
//
// BUILTIN_INFO(HEXAGON.A2_vconj,DI_ftype_DI,1)
//
def int_hexagon_A2_vconj :
Hexagon_di_di_Intrinsic<"HEXAGON_A2_vconj">;
//
// BUILTIN_INFO(HEXAGON.A2_negp,DI_ftype_DI,1)
//
def int_hexagon_A2_negp :
Hexagon_di_di_Intrinsic<"HEXAGON_A2_negp">;
//
// BUILTIN_INFO(HEXAGON.A2_absp,DI_ftype_DI,1)
//
def int_hexagon_A2_absp :
Hexagon_di_di_Intrinsic<"HEXAGON_A2_absp">;
//
// BUILTIN_INFO(HEXAGON.A2_max,SI_ftype_SISI,2)
//
def int_hexagon_A2_max :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_max">;
//
// BUILTIN_INFO(HEXAGON.A2_maxu,USI_ftype_SISI,2)
//
def int_hexagon_A2_maxu :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_maxu">;
//
// BUILTIN_INFO(HEXAGON.A2_min,SI_ftype_SISI,2)
//
def int_hexagon_A2_min :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_min">;
//
// BUILTIN_INFO(HEXAGON.A2_minu,USI_ftype_SISI,2)
//
def int_hexagon_A2_minu :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_minu">;
//
// BUILTIN_INFO(HEXAGON.A2_maxp,DI_ftype_DIDI,2)
//
def int_hexagon_A2_maxp :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_maxp">;
//
// BUILTIN_INFO(HEXAGON.A2_maxup,UDI_ftype_DIDI,2)
//
def int_hexagon_A2_maxup :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_maxup">;
//
// BUILTIN_INFO(HEXAGON.A2_minp,DI_ftype_DIDI,2)
//
def int_hexagon_A2_minp :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_minp">;
//
// BUILTIN_INFO(HEXAGON.A2_minup,UDI_ftype_DIDI,2)
//
def int_hexagon_A2_minup :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_minup">;
//
// BUILTIN_INFO(HEXAGON.A2_tfr,SI_ftype_SI,1)
//
def int_hexagon_A2_tfr :
Hexagon_si_si_Intrinsic<"HEXAGON_A2_tfr">;
//
// BUILTIN_INFO(HEXAGON.A2_tfrsi,SI_ftype_SI,1)
//
def int_hexagon_A2_tfrsi :
Hexagon_si_si_Intrinsic<"HEXAGON_A2_tfrsi">;
//
// BUILTIN_INFO(HEXAGON.A2_tfrp,DI_ftype_DI,1)
//
def int_hexagon_A2_tfrp :
Hexagon_di_di_Intrinsic<"HEXAGON_A2_tfrp">;
//
// BUILTIN_INFO(HEXAGON.A2_tfrpi,DI_ftype_SI,1)
//
def int_hexagon_A2_tfrpi :
Hexagon_di_di_Intrinsic<"HEXAGON_A2_tfrpi">;
//
// BUILTIN_INFO(HEXAGON.A2_zxtb,SI_ftype_SI,1)
//
def int_hexagon_A2_zxtb :
Hexagon_si_si_Intrinsic<"HEXAGON_A2_zxtb">;
//
// BUILTIN_INFO(HEXAGON.A2_sxtb,SI_ftype_SI,1)
//
def int_hexagon_A2_sxtb :
Hexagon_si_si_Intrinsic<"HEXAGON_A2_sxtb">;
//
// BUILTIN_INFO(HEXAGON.A2_zxth,SI_ftype_SI,1)
//
def int_hexagon_A2_zxth :
Hexagon_si_si_Intrinsic<"HEXAGON_A2_zxth">;
//
// BUILTIN_INFO(HEXAGON.A2_sxth,SI_ftype_SI,1)
//
def int_hexagon_A2_sxth :
Hexagon_si_si_Intrinsic<"HEXAGON_A2_sxth">;
//
// BUILTIN_INFO(HEXAGON.A2_combinew,DI_ftype_SISI,2)
//
def int_hexagon_A2_combinew :
Hexagon_di_sisi_Intrinsic<"HEXAGON_A2_combinew">;
//
// BUILTIN_INFO(HEXAGON.A4_combineri,DI_ftype_SISI,2)
//
def int_hexagon_A4_combineri :
Hexagon_di_sisi_Intrinsic<"HEXAGON_A4_combineri">;
//
// BUILTIN_INFO(HEXAGON.A4_combineir,DI_ftype_SISI,2)
//
def int_hexagon_A4_combineir :
Hexagon_di_sisi_Intrinsic<"HEXAGON_A4_combineir">;
//
// BUILTIN_INFO(HEXAGON.A2_combineii,DI_ftype_SISI,2)
//
def int_hexagon_A2_combineii :
Hexagon_di_sisi_Intrinsic<"HEXAGON_A2_combineii">;
//
// BUILTIN_INFO(HEXAGON.A2_combine_hh,SI_ftype_SISI,2)
//
def int_hexagon_A2_combine_hh :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_combine_hh">;
//
// BUILTIN_INFO(HEXAGON.A2_combine_hl,SI_ftype_SISI,2)
//
def int_hexagon_A2_combine_hl :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_combine_hl">;
//
// BUILTIN_INFO(HEXAGON.A2_combine_lh,SI_ftype_SISI,2)
//
def int_hexagon_A2_combine_lh :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_combine_lh">;
//
// BUILTIN_INFO(HEXAGON.A2_combine_ll,SI_ftype_SISI,2)
//
def int_hexagon_A2_combine_ll :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_combine_ll">;
//
// BUILTIN_INFO(HEXAGON.A2_tfril,SI_ftype_SISI,2)
//
def int_hexagon_A2_tfril :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_tfril">;
//
// BUILTIN_INFO(HEXAGON.A2_tfrih,SI_ftype_SISI,2)
//
def int_hexagon_A2_tfrih :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_tfrih">;
//
// BUILTIN_INFO(HEXAGON.A2_and,SI_ftype_SISI,2)
//
def int_hexagon_A2_and :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_and">;
//
// BUILTIN_INFO(HEXAGON.A2_or,SI_ftype_SISI,2)
//
def int_hexagon_A2_or :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_or">;
//
// BUILTIN_INFO(HEXAGON.A2_xor,SI_ftype_SISI,2)
//
def int_hexagon_A2_xor :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_xor">;
//
// BUILTIN_INFO(HEXAGON.A2_not,SI_ftype_SI,1)
//
def int_hexagon_A2_not :
Hexagon_si_si_Intrinsic<"HEXAGON_A2_not">;
//
// BUILTIN_INFO(HEXAGON.M2_xor_xacc,SI_ftype_SISISI,3)
//
def int_hexagon_M2_xor_xacc :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M2_xor_xacc">;
//
// BUILTIN_INFO(HEXAGON.M4_xor_xacc,DI_ftype_DIDIDI,3)
//
def int_hexagon_M4_xor_xacc :
Hexagon_di_dididi_Intrinsic<"HEXAGON_M4_xor_xacc">;
//
// BUILTIN_INFO(HEXAGON.A4_andn,SI_ftype_SISI,2)
//
def int_hexagon_A4_andn :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A4_andn">;
//
// BUILTIN_INFO(HEXAGON.A4_orn,SI_ftype_SISI,2)
//
def int_hexagon_A4_orn :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A4_orn">;
//
// BUILTIN_INFO(HEXAGON.A4_andnp,DI_ftype_DIDI,2)
//
def int_hexagon_A4_andnp :
Hexagon_di_didi_Intrinsic<"HEXAGON_A4_andnp">;
//
// BUILTIN_INFO(HEXAGON.A4_ornp,DI_ftype_DIDI,2)
//
def int_hexagon_A4_ornp :
Hexagon_di_didi_Intrinsic<"HEXAGON_A4_ornp">;
//
// BUILTIN_INFO(HEXAGON.S4_addaddi,SI_ftype_SISISI,3)
//
def int_hexagon_S4_addaddi :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S4_addaddi">;
//
// BUILTIN_INFO(HEXAGON.S4_subaddi,SI_ftype_SISISI,3)
//
def int_hexagon_S4_subaddi :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S4_subaddi">;
//
// BUILTIN_INFO(HEXAGON.M4_and_and,SI_ftype_SISISI,3)
//
def int_hexagon_M4_and_and :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M4_and_and">;
//
// BUILTIN_INFO(HEXAGON.M4_and_andn,SI_ftype_SISISI,3)
//
def int_hexagon_M4_and_andn :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M4_and_andn">;
//
// BUILTIN_INFO(HEXAGON.M4_and_or,SI_ftype_SISISI,3)
//
def int_hexagon_M4_and_or :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M4_and_or">;
//
// BUILTIN_INFO(HEXAGON.M4_and_xor,SI_ftype_SISISI,3)
//
def int_hexagon_M4_and_xor :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M4_and_xor">;
//
// BUILTIN_INFO(HEXAGON.M4_or_and,SI_ftype_SISISI,3)
//
def int_hexagon_M4_or_and :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M4_or_and">;
//
// BUILTIN_INFO(HEXAGON.M4_or_andn,SI_ftype_SISISI,3)
//
def int_hexagon_M4_or_andn :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M4_or_andn">;
//
// BUILTIN_INFO(HEXAGON.M4_or_or,SI_ftype_SISISI,3)
//
def int_hexagon_M4_or_or :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M4_or_or">;
//
// BUILTIN_INFO(HEXAGON.M4_or_xor,SI_ftype_SISISI,3)
//
def int_hexagon_M4_or_xor :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M4_or_xor">;
//
// BUILTIN_INFO(HEXAGON.S4_or_andix,SI_ftype_SISISI,3)
//
def int_hexagon_S4_or_andix :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S4_or_andix">;
//
// BUILTIN_INFO(HEXAGON.S4_or_andi,SI_ftype_SISISI,3)
//
def int_hexagon_S4_or_andi :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S4_or_andi">;
//
// BUILTIN_INFO(HEXAGON.S4_or_ori,SI_ftype_SISISI,3)
//
def int_hexagon_S4_or_ori :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S4_or_ori">;
//
// BUILTIN_INFO(HEXAGON.M4_xor_and,SI_ftype_SISISI,3)
//
def int_hexagon_M4_xor_and :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M4_xor_and">;
//
// BUILTIN_INFO(HEXAGON.M4_xor_or,SI_ftype_SISISI,3)
//
def int_hexagon_M4_xor_or :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M4_xor_or">;
//
// BUILTIN_INFO(HEXAGON.M4_xor_andn,SI_ftype_SISISI,3)
//
def int_hexagon_M4_xor_andn :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_M4_xor_andn">;
//
// BUILTIN_INFO(HEXAGON.A2_subri,SI_ftype_SISI,2)
//
def int_hexagon_A2_subri :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_subri">;
//
// BUILTIN_INFO(HEXAGON.A2_andir,SI_ftype_SISI,2)
//
def int_hexagon_A2_andir :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_andir">;
//
// BUILTIN_INFO(HEXAGON.A2_orir,SI_ftype_SISI,2)
//
def int_hexagon_A2_orir :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_orir">;
//
// BUILTIN_INFO(HEXAGON.A2_andp,DI_ftype_DIDI,2)
//
def int_hexagon_A2_andp :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_andp">;
//
// BUILTIN_INFO(HEXAGON.A2_orp,DI_ftype_DIDI,2)
//
def int_hexagon_A2_orp :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_orp">;
//
// BUILTIN_INFO(HEXAGON.A2_xorp,DI_ftype_DIDI,2)
//
def int_hexagon_A2_xorp :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_xorp">;
//
// BUILTIN_INFO(HEXAGON.A2_notp,DI_ftype_DI,1)
//
def int_hexagon_A2_notp :
Hexagon_di_di_Intrinsic<"HEXAGON_A2_notp">;
//
// BUILTIN_INFO(HEXAGON.A2_sxtw,DI_ftype_SI,1)
//
def int_hexagon_A2_sxtw :
Hexagon_di_si_Intrinsic<"HEXAGON_A2_sxtw">;
//
// BUILTIN_INFO(HEXAGON.A2_sat,SI_ftype_DI,1)
//
def int_hexagon_A2_sat :
Hexagon_si_di_Intrinsic<"HEXAGON_A2_sat">;
//
// BUILTIN_INFO(HEXAGON.A2_roundsat,SI_ftype_DI,1)
//
def int_hexagon_A2_roundsat :
Hexagon_si_di_Intrinsic<"HEXAGON_A2_roundsat">;
//
// BUILTIN_INFO(HEXAGON.A2_sath,SI_ftype_SI,1)
//
def int_hexagon_A2_sath :
Hexagon_si_si_Intrinsic<"HEXAGON_A2_sath">;
//
// BUILTIN_INFO(HEXAGON.A2_satuh,SI_ftype_SI,1)
//
def int_hexagon_A2_satuh :
Hexagon_si_si_Intrinsic<"HEXAGON_A2_satuh">;
//
// BUILTIN_INFO(HEXAGON.A2_satub,SI_ftype_SI,1)
//
def int_hexagon_A2_satub :
Hexagon_si_si_Intrinsic<"HEXAGON_A2_satub">;
//
// BUILTIN_INFO(HEXAGON.A2_satb,SI_ftype_SI,1)
//
def int_hexagon_A2_satb :
Hexagon_si_si_Intrinsic<"HEXAGON_A2_satb">;
//
// BUILTIN_INFO(HEXAGON.A2_vaddub,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vaddub :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vaddub">;
//
// BUILTIN_INFO(HEXAGON.A2_vaddb_map,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vaddb_map :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vaddb_map">;
//
// BUILTIN_INFO(HEXAGON.A2_vaddubs,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vaddubs :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vaddubs">;
//
// BUILTIN_INFO(HEXAGON.A2_vaddh,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vaddh :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vaddh">;
//
// BUILTIN_INFO(HEXAGON.A2_vaddhs,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vaddhs :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vaddhs">;
//
// BUILTIN_INFO(HEXAGON.A2_vadduhs,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vadduhs :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vadduhs">;
//
// BUILTIN_INFO(HEXAGON.A5_vaddhubs,SI_ftype_DIDI,2)
//
def int_hexagon_A5_vaddhubs :
Hexagon_si_didi_Intrinsic<"HEXAGON_A5_vaddhubs">;
//
// BUILTIN_INFO(HEXAGON.A2_vaddw,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vaddw :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vaddw">;
//
// BUILTIN_INFO(HEXAGON.A2_vaddws,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vaddws :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vaddws">;
//
// BUILTIN_INFO(HEXAGON.S4_vxaddsubw,DI_ftype_DIDI,2)
//
def int_hexagon_S4_vxaddsubw :
Hexagon_di_didi_Intrinsic<"HEXAGON_S4_vxaddsubw">;
//
// BUILTIN_INFO(HEXAGON.S4_vxsubaddw,DI_ftype_DIDI,2)
//
def int_hexagon_S4_vxsubaddw :
Hexagon_di_didi_Intrinsic<"HEXAGON_S4_vxsubaddw">;
//
// BUILTIN_INFO(HEXAGON.S4_vxaddsubh,DI_ftype_DIDI,2)
//
def int_hexagon_S4_vxaddsubh :
Hexagon_di_didi_Intrinsic<"HEXAGON_S4_vxaddsubh">;
//
// BUILTIN_INFO(HEXAGON.S4_vxsubaddh,DI_ftype_DIDI,2)
//
def int_hexagon_S4_vxsubaddh :
Hexagon_di_didi_Intrinsic<"HEXAGON_S4_vxsubaddh">;
//
// BUILTIN_INFO(HEXAGON.S4_vxaddsubhr,DI_ftype_DIDI,2)
//
def int_hexagon_S4_vxaddsubhr :
Hexagon_di_didi_Intrinsic<"HEXAGON_S4_vxaddsubhr">;
//
// BUILTIN_INFO(HEXAGON.S4_vxsubaddhr,DI_ftype_DIDI,2)
//
def int_hexagon_S4_vxsubaddhr :
Hexagon_di_didi_Intrinsic<"HEXAGON_S4_vxsubaddhr">;
//
// BUILTIN_INFO(HEXAGON.A2_svavgh,SI_ftype_SISI,2)
//
def int_hexagon_A2_svavgh :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_svavgh">;
//
// BUILTIN_INFO(HEXAGON.A2_svavghs,SI_ftype_SISI,2)
//
def int_hexagon_A2_svavghs :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_svavghs">;
//
// BUILTIN_INFO(HEXAGON.A2_svnavgh,SI_ftype_SISI,2)
//
def int_hexagon_A2_svnavgh :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_svnavgh">;
//
// BUILTIN_INFO(HEXAGON.A2_svaddh,SI_ftype_SISI,2)
//
def int_hexagon_A2_svaddh :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_svaddh">;
//
// BUILTIN_INFO(HEXAGON.A2_svaddhs,SI_ftype_SISI,2)
//
def int_hexagon_A2_svaddhs :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_svaddhs">;
//
// BUILTIN_INFO(HEXAGON.A2_svadduhs,SI_ftype_SISI,2)
//
def int_hexagon_A2_svadduhs :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_svadduhs">;
//
// BUILTIN_INFO(HEXAGON.A2_svsubh,SI_ftype_SISI,2)
//
def int_hexagon_A2_svsubh :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_svsubh">;
//
// BUILTIN_INFO(HEXAGON.A2_svsubhs,SI_ftype_SISI,2)
//
def int_hexagon_A2_svsubhs :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_svsubhs">;
//
// BUILTIN_INFO(HEXAGON.A2_svsubuhs,SI_ftype_SISI,2)
//
def int_hexagon_A2_svsubuhs :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A2_svsubuhs">;
//
// BUILTIN_INFO(HEXAGON.A2_vraddub,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vraddub :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vraddub">;
//
// BUILTIN_INFO(HEXAGON.A2_vraddub_acc,DI_ftype_DIDIDI,3)
//
def int_hexagon_A2_vraddub_acc :
Hexagon_di_dididi_Intrinsic<"HEXAGON_A2_vraddub_acc">;
//
// BUILTIN_INFO(HEXAGON.M2_vraddh,SI_ftype_DIDI,2)
//
def int_hexagon_M2_vraddh :
Hexagon_si_didi_Intrinsic<"HEXAGON_M2_vraddh">;
//
// BUILTIN_INFO(HEXAGON.M2_vradduh,SI_ftype_DIDI,2)
//
def int_hexagon_M2_vradduh :
Hexagon_si_didi_Intrinsic<"HEXAGON_M2_vradduh">;
//
// BUILTIN_INFO(HEXAGON.A2_vsubub,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vsubub :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vsubub">;
//
// BUILTIN_INFO(HEXAGON.A2_vsubb_map,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vsubb_map :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vsubb_map">;
//
// BUILTIN_INFO(HEXAGON.A2_vsububs,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vsububs :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vsububs">;
//
// BUILTIN_INFO(HEXAGON.A2_vsubh,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vsubh :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vsubh">;
//
// BUILTIN_INFO(HEXAGON.A2_vsubhs,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vsubhs :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vsubhs">;
//
// BUILTIN_INFO(HEXAGON.A2_vsubuhs,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vsubuhs :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vsubuhs">;
//
// BUILTIN_INFO(HEXAGON.A2_vsubw,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vsubw :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vsubw">;
//
// BUILTIN_INFO(HEXAGON.A2_vsubws,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vsubws :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vsubws">;
//
// BUILTIN_INFO(HEXAGON.A2_vabsh,DI_ftype_DI,1)
//
def int_hexagon_A2_vabsh :
Hexagon_di_di_Intrinsic<"HEXAGON_A2_vabsh">;
//
// BUILTIN_INFO(HEXAGON.A2_vabshsat,DI_ftype_DI,1)
//
def int_hexagon_A2_vabshsat :
Hexagon_di_di_Intrinsic<"HEXAGON_A2_vabshsat">;
//
// BUILTIN_INFO(HEXAGON.A2_vabsw,DI_ftype_DI,1)
//
def int_hexagon_A2_vabsw :
Hexagon_di_di_Intrinsic<"HEXAGON_A2_vabsw">;
//
// BUILTIN_INFO(HEXAGON.A2_vabswsat,DI_ftype_DI,1)
//
def int_hexagon_A2_vabswsat :
Hexagon_di_di_Intrinsic<"HEXAGON_A2_vabswsat">;
//
// BUILTIN_INFO(HEXAGON.M2_vabsdiffw,DI_ftype_DIDI,2)
//
def int_hexagon_M2_vabsdiffw :
Hexagon_di_didi_Intrinsic<"HEXAGON_M2_vabsdiffw">;
//
// BUILTIN_INFO(HEXAGON.M2_vabsdiffh,DI_ftype_DIDI,2)
//
def int_hexagon_M2_vabsdiffh :
Hexagon_di_didi_Intrinsic<"HEXAGON_M2_vabsdiffh">;
//
// BUILTIN_INFO(HEXAGON.A2_vrsadub,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vrsadub :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vrsadub">;
//
// BUILTIN_INFO(HEXAGON.A2_vrsadub_acc,DI_ftype_DIDIDI,3)
//
def int_hexagon_A2_vrsadub_acc :
Hexagon_di_dididi_Intrinsic<"HEXAGON_A2_vrsadub_acc">;
//
// BUILTIN_INFO(HEXAGON.A2_vavgub,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vavgub :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vavgub">;
//
// BUILTIN_INFO(HEXAGON.A2_vavguh,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vavguh :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vavguh">;
//
// BUILTIN_INFO(HEXAGON.A2_vavgh,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vavgh :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vavgh">;
//
// BUILTIN_INFO(HEXAGON.A2_vnavgh,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vnavgh :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vnavgh">;
//
// BUILTIN_INFO(HEXAGON.A2_vavgw,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vavgw :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vavgw">;
//
// BUILTIN_INFO(HEXAGON.A2_vnavgw,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vnavgw :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vnavgw">;
//
// BUILTIN_INFO(HEXAGON.A2_vavgwr,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vavgwr :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vavgwr">;
//
// BUILTIN_INFO(HEXAGON.A2_vnavgwr,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vnavgwr :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vnavgwr">;
//
// BUILTIN_INFO(HEXAGON.A2_vavgwcr,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vavgwcr :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vavgwcr">;
//
// BUILTIN_INFO(HEXAGON.A2_vnavgwcr,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vnavgwcr :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vnavgwcr">;
//
// BUILTIN_INFO(HEXAGON.A2_vavghcr,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vavghcr :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vavghcr">;
//
// BUILTIN_INFO(HEXAGON.A2_vnavghcr,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vnavghcr :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vnavghcr">;
//
// BUILTIN_INFO(HEXAGON.A2_vavguw,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vavguw :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vavguw">;
//
// BUILTIN_INFO(HEXAGON.A2_vavguwr,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vavguwr :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vavguwr">;
//
// BUILTIN_INFO(HEXAGON.A2_vavgubr,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vavgubr :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vavgubr">;
//
// BUILTIN_INFO(HEXAGON.A2_vavguhr,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vavguhr :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vavguhr">;
//
// BUILTIN_INFO(HEXAGON.A2_vavghr,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vavghr :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vavghr">;
//
// BUILTIN_INFO(HEXAGON.A2_vnavghr,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vnavghr :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vnavghr">;
//
// BUILTIN_INFO(HEXAGON.A4_round_ri,SI_ftype_SISI,2)
//
def int_hexagon_A4_round_ri :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A4_round_ri">;
//
// BUILTIN_INFO(HEXAGON.A4_round_rr,SI_ftype_SISI,2)
//
def int_hexagon_A4_round_rr :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A4_round_rr">;
//
// BUILTIN_INFO(HEXAGON.A4_round_ri_sat,SI_ftype_SISI,2)
//
def int_hexagon_A4_round_ri_sat :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A4_round_ri_sat">;
//
// BUILTIN_INFO(HEXAGON.A4_round_rr_sat,SI_ftype_SISI,2)
//
def int_hexagon_A4_round_rr_sat :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A4_round_rr_sat">;
//
// BUILTIN_INFO(HEXAGON.A4_cround_ri,SI_ftype_SISI,2)
//
def int_hexagon_A4_cround_ri :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A4_cround_ri">;
//
// BUILTIN_INFO(HEXAGON.A4_cround_rr,SI_ftype_SISI,2)
//
def int_hexagon_A4_cround_rr :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A4_cround_rr">;
//
// BUILTIN_INFO(HEXAGON.A4_vrminh,DI_ftype_DIDISI,3)
//
def int_hexagon_A4_vrminh :
Hexagon_di_didisi_Intrinsic<"HEXAGON_A4_vrminh">;
//
// BUILTIN_INFO(HEXAGON.A4_vrmaxh,DI_ftype_DIDISI,3)
//
def int_hexagon_A4_vrmaxh :
Hexagon_di_didisi_Intrinsic<"HEXAGON_A4_vrmaxh">;
//
// BUILTIN_INFO(HEXAGON.A4_vrminuh,DI_ftype_DIDISI,3)
//
def int_hexagon_A4_vrminuh :
Hexagon_di_didisi_Intrinsic<"HEXAGON_A4_vrminuh">;
//
// BUILTIN_INFO(HEXAGON.A4_vrmaxuh,DI_ftype_DIDISI,3)
//
def int_hexagon_A4_vrmaxuh :
Hexagon_di_didisi_Intrinsic<"HEXAGON_A4_vrmaxuh">;
//
// BUILTIN_INFO(HEXAGON.A4_vrminw,DI_ftype_DIDISI,3)
//
def int_hexagon_A4_vrminw :
Hexagon_di_didisi_Intrinsic<"HEXAGON_A4_vrminw">;
//
// BUILTIN_INFO(HEXAGON.A4_vrmaxw,DI_ftype_DIDISI,3)
//
def int_hexagon_A4_vrmaxw :
Hexagon_di_didisi_Intrinsic<"HEXAGON_A4_vrmaxw">;
//
// BUILTIN_INFO(HEXAGON.A4_vrminuw,DI_ftype_DIDISI,3)
//
def int_hexagon_A4_vrminuw :
Hexagon_di_didisi_Intrinsic<"HEXAGON_A4_vrminuw">;
//
// BUILTIN_INFO(HEXAGON.A4_vrmaxuw,DI_ftype_DIDISI,3)
//
def int_hexagon_A4_vrmaxuw :
Hexagon_di_didisi_Intrinsic<"HEXAGON_A4_vrmaxuw">;
//
// BUILTIN_INFO(HEXAGON.A2_vminb,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vminb :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vminb">;
//
// BUILTIN_INFO(HEXAGON.A2_vmaxb,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vmaxb :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vmaxb">;
//
// BUILTIN_INFO(HEXAGON.A2_vminub,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vminub :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vminub">;
//
// BUILTIN_INFO(HEXAGON.A2_vmaxub,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vmaxub :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vmaxub">;
//
// BUILTIN_INFO(HEXAGON.A2_vminh,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vminh :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vminh">;
//
// BUILTIN_INFO(HEXAGON.A2_vmaxh,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vmaxh :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vmaxh">;
//
// BUILTIN_INFO(HEXAGON.A2_vminuh,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vminuh :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vminuh">;
//
// BUILTIN_INFO(HEXAGON.A2_vmaxuh,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vmaxuh :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vmaxuh">;
//
// BUILTIN_INFO(HEXAGON.A2_vminw,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vminw :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vminw">;
//
// BUILTIN_INFO(HEXAGON.A2_vmaxw,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vmaxw :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vmaxw">;
//
// BUILTIN_INFO(HEXAGON.A2_vminuw,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vminuw :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vminuw">;
//
// BUILTIN_INFO(HEXAGON.A2_vmaxuw,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vmaxuw :
Hexagon_di_didi_Intrinsic<"HEXAGON_A2_vmaxuw">;
//
// BUILTIN_INFO(HEXAGON.A4_modwrapu,SI_ftype_SISI,2)
//
def int_hexagon_A4_modwrapu :
Hexagon_si_sisi_Intrinsic<"HEXAGON_A4_modwrapu">;
//
// BUILTIN_INFO(HEXAGON.F2_sfadd,SF_ftype_SFSF,2)
//
def int_hexagon_F2_sfadd :
Hexagon_sf_sfsf_Intrinsic<"HEXAGON_F2_sfadd">;
//
// BUILTIN_INFO(HEXAGON.F2_sfsub,SF_ftype_SFSF,2)
//
def int_hexagon_F2_sfsub :
Hexagon_sf_sfsf_Intrinsic<"HEXAGON_F2_sfsub">;
//
// BUILTIN_INFO(HEXAGON.F2_sfmpy,SF_ftype_SFSF,2)
//
def int_hexagon_F2_sfmpy :
Hexagon_sf_sfsf_Intrinsic<"HEXAGON_F2_sfmpy">;
//
// BUILTIN_INFO(HEXAGON.F2_sffma,SF_ftype_SFSFSF,3)
//
def int_hexagon_F2_sffma :
Hexagon_sf_sfsfsf_Intrinsic<"HEXAGON_F2_sffma">;
//
// BUILTIN_INFO(HEXAGON.F2_sffma_sc,SF_ftype_SFSFSFQI,4)
//
def int_hexagon_F2_sffma_sc :
Hexagon_sf_sfsfsfqi_Intrinsic<"HEXAGON_F2_sffma_sc">;
//
// BUILTIN_INFO(HEXAGON.F2_sffms,SF_ftype_SFSFSF,3)
//
def int_hexagon_F2_sffms :
Hexagon_sf_sfsfsf_Intrinsic<"HEXAGON_F2_sffms">;
//
// BUILTIN_INFO(HEXAGON.F2_sffma_lib,SF_ftype_SFSFSF,3)
//
def int_hexagon_F2_sffma_lib :
Hexagon_sf_sfsfsf_Intrinsic<"HEXAGON_F2_sffma_lib">;
//
// BUILTIN_INFO(HEXAGON.F2_sffms_lib,SF_ftype_SFSFSF,3)
//
def int_hexagon_F2_sffms_lib :
Hexagon_sf_sfsfsf_Intrinsic<"HEXAGON_F2_sffms_lib">;
//
// BUILTIN_INFO(HEXAGON.F2_sfcmpeq,QI_ftype_SFSF,2)
//
def int_hexagon_F2_sfcmpeq :
Hexagon_si_sfsf_Intrinsic<"HEXAGON_F2_sfcmpeq">;
//
// BUILTIN_INFO(HEXAGON.F2_sfcmpgt,QI_ftype_SFSF,2)
//
def int_hexagon_F2_sfcmpgt :
Hexagon_si_sfsf_Intrinsic<"HEXAGON_F2_sfcmpgt">;
//
// BUILTIN_INFO(HEXAGON.F2_sfcmpge,QI_ftype_SFSF,2)
//
def int_hexagon_F2_sfcmpge :
Hexagon_si_sfsf_Intrinsic<"HEXAGON_F2_sfcmpge">;
//
// BUILTIN_INFO(HEXAGON.F2_sfcmpuo,QI_ftype_SFSF,2)
//
def int_hexagon_F2_sfcmpuo :
Hexagon_si_sfsf_Intrinsic<"HEXAGON_F2_sfcmpuo">;
//
// BUILTIN_INFO(HEXAGON.F2_sfmax,SF_ftype_SFSF,2)
//
def int_hexagon_F2_sfmax :
Hexagon_sf_sfsf_Intrinsic<"HEXAGON_F2_sfmax">;
//
// BUILTIN_INFO(HEXAGON.F2_sfmin,SF_ftype_SFSF,2)
//
def int_hexagon_F2_sfmin :
Hexagon_sf_sfsf_Intrinsic<"HEXAGON_F2_sfmin">;
//
// BUILTIN_INFO(HEXAGON.F2_sfclass,QI_ftype_SFSI,2)
//
def int_hexagon_F2_sfclass :
Hexagon_si_sfsi_Intrinsic<"HEXAGON_F2_sfclass">;
//
// BUILTIN_INFO(HEXAGON.F2_sfimm_p,SF_ftype_SI,1)
//
def int_hexagon_F2_sfimm_p :
Hexagon_sf_si_Intrinsic<"HEXAGON_F2_sfimm_p">;
//
// BUILTIN_INFO(HEXAGON.F2_sfimm_n,SF_ftype_SI,1)
//
def int_hexagon_F2_sfimm_n :
Hexagon_sf_si_Intrinsic<"HEXAGON_F2_sfimm_n">;
//
// BUILTIN_INFO(HEXAGON.F2_sffixupn,SF_ftype_SFSF,2)
//
def int_hexagon_F2_sffixupn :
Hexagon_sf_sfsf_Intrinsic<"HEXAGON_F2_sffixupn">;
//
// BUILTIN_INFO(HEXAGON.F2_sffixupd,SF_ftype_SFSF,2)
//
def int_hexagon_F2_sffixupd :
Hexagon_sf_sfsf_Intrinsic<"HEXAGON_F2_sffixupd">;
//
// BUILTIN_INFO(HEXAGON.F2_sffixupr,SF_ftype_SF,1)
//
def int_hexagon_F2_sffixupr :
Hexagon_sf_sf_Intrinsic<"HEXAGON_F2_sffixupr">;
//
// BUILTIN_INFO(HEXAGON.F2_dfcmpeq,QI_ftype_DFDF,2)
//
def int_hexagon_F2_dfcmpeq :
Hexagon_si_dfdf_Intrinsic<"HEXAGON_F2_dfcmpeq">;
//
// BUILTIN_INFO(HEXAGON.F2_dfcmpgt,QI_ftype_DFDF,2)
//
def int_hexagon_F2_dfcmpgt :
Hexagon_si_dfdf_Intrinsic<"HEXAGON_F2_dfcmpgt">;
//
// BUILTIN_INFO(HEXAGON.F2_dfcmpge,QI_ftype_DFDF,2)
//
def int_hexagon_F2_dfcmpge :
Hexagon_si_dfdf_Intrinsic<"HEXAGON_F2_dfcmpge">;
//
// BUILTIN_INFO(HEXAGON.F2_dfcmpuo,QI_ftype_DFDF,2)
//
def int_hexagon_F2_dfcmpuo :
Hexagon_si_dfdf_Intrinsic<"HEXAGON_F2_dfcmpuo">;
//
// BUILTIN_INFO(HEXAGON.F2_dfclass,QI_ftype_DFSI,2)
//
def int_hexagon_F2_dfclass :
Hexagon_si_dfsi_Intrinsic<"HEXAGON_F2_dfclass">;
//
// BUILTIN_INFO(HEXAGON.F2_dfimm_p,DF_ftype_SI,1)
//
def int_hexagon_F2_dfimm_p :
Hexagon_df_si_Intrinsic<"HEXAGON_F2_dfimm_p">;
//
// BUILTIN_INFO(HEXAGON.F2_dfimm_n,DF_ftype_SI,1)
//
def int_hexagon_F2_dfimm_n :
Hexagon_df_si_Intrinsic<"HEXAGON_F2_dfimm_n">;
//
// BUILTIN_INFO(HEXAGON.F2_conv_sf2df,DF_ftype_SF,1)
//
def int_hexagon_F2_conv_sf2df :
Hexagon_df_sf_Intrinsic<"HEXAGON_F2_conv_sf2df">;
//
// BUILTIN_INFO(HEXAGON.F2_conv_df2sf,SF_ftype_DF,1)
//
def int_hexagon_F2_conv_df2sf :
Hexagon_sf_df_Intrinsic<"HEXAGON_F2_conv_df2sf">;
//
// BUILTIN_INFO(HEXAGON.F2_conv_uw2sf,SF_ftype_SI,1)
//
def int_hexagon_F2_conv_uw2sf :
Hexagon_sf_si_Intrinsic<"HEXAGON_F2_conv_uw2sf">;
//
// BUILTIN_INFO(HEXAGON.F2_conv_uw2df,DF_ftype_SI,1)
//
def int_hexagon_F2_conv_uw2df :
Hexagon_df_si_Intrinsic<"HEXAGON_F2_conv_uw2df">;
//
// BUILTIN_INFO(HEXAGON.F2_conv_w2sf,SF_ftype_SI,1)
//
def int_hexagon_F2_conv_w2sf :
Hexagon_sf_si_Intrinsic<"HEXAGON_F2_conv_w2sf">;
//
// BUILTIN_INFO(HEXAGON.F2_conv_w2df,DF_ftype_SI,1)
//
def int_hexagon_F2_conv_w2df :
Hexagon_df_si_Intrinsic<"HEXAGON_F2_conv_w2df">;
//
// BUILTIN_INFO(HEXAGON.F2_conv_ud2sf,SF_ftype_DI,1)
//
def int_hexagon_F2_conv_ud2sf :
Hexagon_sf_di_Intrinsic<"HEXAGON_F2_conv_ud2sf">;
//
// BUILTIN_INFO(HEXAGON.F2_conv_ud2df,DF_ftype_DI,1)
//
def int_hexagon_F2_conv_ud2df :
Hexagon_df_di_Intrinsic<"HEXAGON_F2_conv_ud2df">;
//
// BUILTIN_INFO(HEXAGON.F2_conv_d2sf,SF_ftype_DI,1)
//
def int_hexagon_F2_conv_d2sf :
Hexagon_sf_di_Intrinsic<"HEXAGON_F2_conv_d2sf">;
//
// BUILTIN_INFO(HEXAGON.F2_conv_d2df,DF_ftype_DI,1)
//
def int_hexagon_F2_conv_d2df :
Hexagon_df_di_Intrinsic<"HEXAGON_F2_conv_d2df">;
//
// BUILTIN_INFO(HEXAGON.F2_conv_sf2uw,SI_ftype_SF,1)
//
def int_hexagon_F2_conv_sf2uw :
Hexagon_si_sf_Intrinsic<"HEXAGON_F2_conv_sf2uw">;
//
// BUILTIN_INFO(HEXAGON.F2_conv_sf2w,SI_ftype_SF,1)
//
def int_hexagon_F2_conv_sf2w :
Hexagon_si_sf_Intrinsic<"HEXAGON_F2_conv_sf2w">;
//
// BUILTIN_INFO(HEXAGON.F2_conv_sf2ud,DI_ftype_SF,1)
//
def int_hexagon_F2_conv_sf2ud :
Hexagon_di_sf_Intrinsic<"HEXAGON_F2_conv_sf2ud">;
//
// BUILTIN_INFO(HEXAGON.F2_conv_sf2d,DI_ftype_SF,1)
//
def int_hexagon_F2_conv_sf2d :
Hexagon_di_sf_Intrinsic<"HEXAGON_F2_conv_sf2d">;
//
// BUILTIN_INFO(HEXAGON.F2_conv_df2uw,SI_ftype_DF,1)
//
def int_hexagon_F2_conv_df2uw :
Hexagon_si_df_Intrinsic<"HEXAGON_F2_conv_df2uw">;
//
// BUILTIN_INFO(HEXAGON.F2_conv_df2w,SI_ftype_DF,1)
//
def int_hexagon_F2_conv_df2w :
Hexagon_si_df_Intrinsic<"HEXAGON_F2_conv_df2w">;
//
// BUILTIN_INFO(HEXAGON.F2_conv_df2ud,DI_ftype_DF,1)
//
def int_hexagon_F2_conv_df2ud :
Hexagon_di_df_Intrinsic<"HEXAGON_F2_conv_df2ud">;
//
// BUILTIN_INFO(HEXAGON.F2_conv_df2d,DI_ftype_DF,1)
//
def int_hexagon_F2_conv_df2d :
Hexagon_di_df_Intrinsic<"HEXAGON_F2_conv_df2d">;
//
// BUILTIN_INFO(HEXAGON.F2_conv_sf2uw_chop,SI_ftype_SF,1)
//
def int_hexagon_F2_conv_sf2uw_chop :
Hexagon_si_sf_Intrinsic<"HEXAGON_F2_conv_sf2uw_chop">;
//
// BUILTIN_INFO(HEXAGON.F2_conv_sf2w_chop,SI_ftype_SF,1)
//
def int_hexagon_F2_conv_sf2w_chop :
Hexagon_si_sf_Intrinsic<"HEXAGON_F2_conv_sf2w_chop">;
//
// BUILTIN_INFO(HEXAGON.F2_conv_sf2ud_chop,DI_ftype_SF,1)
//
def int_hexagon_F2_conv_sf2ud_chop :
Hexagon_di_sf_Intrinsic<"HEXAGON_F2_conv_sf2ud_chop">;
//
// BUILTIN_INFO(HEXAGON.F2_conv_sf2d_chop,DI_ftype_SF,1)
//
def int_hexagon_F2_conv_sf2d_chop :
Hexagon_di_sf_Intrinsic<"HEXAGON_F2_conv_sf2d_chop">;
//
// BUILTIN_INFO(HEXAGON.F2_conv_df2uw_chop,SI_ftype_DF,1)
//
def int_hexagon_F2_conv_df2uw_chop :
Hexagon_si_df_Intrinsic<"HEXAGON_F2_conv_df2uw_chop">;
//
// BUILTIN_INFO(HEXAGON.F2_conv_df2w_chop,SI_ftype_DF,1)
//
def int_hexagon_F2_conv_df2w_chop :
Hexagon_si_df_Intrinsic<"HEXAGON_F2_conv_df2w_chop">;
//
// BUILTIN_INFO(HEXAGON.F2_conv_df2ud_chop,DI_ftype_DF,1)
//
def int_hexagon_F2_conv_df2ud_chop :
Hexagon_di_df_Intrinsic<"HEXAGON_F2_conv_df2ud_chop">;
//
// BUILTIN_INFO(HEXAGON.F2_conv_df2d_chop,DI_ftype_DF,1)
//
def int_hexagon_F2_conv_df2d_chop :
Hexagon_di_df_Intrinsic<"HEXAGON_F2_conv_df2d_chop">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_r_r,SI_ftype_SISI,2)
//
def int_hexagon_S2_asr_r_r :
Hexagon_si_sisi_Intrinsic<"HEXAGON_S2_asr_r_r">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_r_r,SI_ftype_SISI,2)
//
def int_hexagon_S2_asl_r_r :
Hexagon_si_sisi_Intrinsic<"HEXAGON_S2_asl_r_r">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_r_r,SI_ftype_SISI,2)
//
def int_hexagon_S2_lsr_r_r :
Hexagon_si_sisi_Intrinsic<"HEXAGON_S2_lsr_r_r">;
//
// BUILTIN_INFO(HEXAGON.S2_lsl_r_r,SI_ftype_SISI,2)
//
def int_hexagon_S2_lsl_r_r :
Hexagon_si_sisi_Intrinsic<"HEXAGON_S2_lsl_r_r">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_r_p,DI_ftype_DISI,2)
//
def int_hexagon_S2_asr_r_p :
Hexagon_di_disi_Intrinsic<"HEXAGON_S2_asr_r_p">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_r_p,DI_ftype_DISI,2)
//
def int_hexagon_S2_asl_r_p :
Hexagon_di_disi_Intrinsic<"HEXAGON_S2_asl_r_p">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_r_p,DI_ftype_DISI,2)
//
def int_hexagon_S2_lsr_r_p :
Hexagon_di_disi_Intrinsic<"HEXAGON_S2_lsr_r_p">;
//
// BUILTIN_INFO(HEXAGON.S2_lsl_r_p,DI_ftype_DISI,2)
//
def int_hexagon_S2_lsl_r_p :
Hexagon_di_disi_Intrinsic<"HEXAGON_S2_lsl_r_p">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_r_r_acc,SI_ftype_SISISI,3)
//
def int_hexagon_S2_asr_r_r_acc :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S2_asr_r_r_acc">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_r_r_acc,SI_ftype_SISISI,3)
//
def int_hexagon_S2_asl_r_r_acc :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S2_asl_r_r_acc">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_r_r_acc,SI_ftype_SISISI,3)
//
def int_hexagon_S2_lsr_r_r_acc :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S2_lsr_r_r_acc">;
//
// BUILTIN_INFO(HEXAGON.S2_lsl_r_r_acc,SI_ftype_SISISI,3)
//
def int_hexagon_S2_lsl_r_r_acc :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S2_lsl_r_r_acc">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_r_p_acc,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_asr_r_p_acc :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_asr_r_p_acc">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_r_p_acc,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_asl_r_p_acc :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_asl_r_p_acc">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_r_p_acc,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_lsr_r_p_acc :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_lsr_r_p_acc">;
//
// BUILTIN_INFO(HEXAGON.S2_lsl_r_p_acc,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_lsl_r_p_acc :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_lsl_r_p_acc">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_r_r_nac,SI_ftype_SISISI,3)
//
def int_hexagon_S2_asr_r_r_nac :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S2_asr_r_r_nac">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_r_r_nac,SI_ftype_SISISI,3)
//
def int_hexagon_S2_asl_r_r_nac :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S2_asl_r_r_nac">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_r_r_nac,SI_ftype_SISISI,3)
//
def int_hexagon_S2_lsr_r_r_nac :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S2_lsr_r_r_nac">;
//
// BUILTIN_INFO(HEXAGON.S2_lsl_r_r_nac,SI_ftype_SISISI,3)
//
def int_hexagon_S2_lsl_r_r_nac :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S2_lsl_r_r_nac">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_r_p_nac,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_asr_r_p_nac :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_asr_r_p_nac">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_r_p_nac,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_asl_r_p_nac :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_asl_r_p_nac">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_r_p_nac,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_lsr_r_p_nac :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_lsr_r_p_nac">;
//
// BUILTIN_INFO(HEXAGON.S2_lsl_r_p_nac,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_lsl_r_p_nac :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_lsl_r_p_nac">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_r_r_and,SI_ftype_SISISI,3)
//
def int_hexagon_S2_asr_r_r_and :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S2_asr_r_r_and">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_r_r_and,SI_ftype_SISISI,3)
//
def int_hexagon_S2_asl_r_r_and :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S2_asl_r_r_and">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_r_r_and,SI_ftype_SISISI,3)
//
def int_hexagon_S2_lsr_r_r_and :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S2_lsr_r_r_and">;
//
// BUILTIN_INFO(HEXAGON.S2_lsl_r_r_and,SI_ftype_SISISI,3)
//
def int_hexagon_S2_lsl_r_r_and :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S2_lsl_r_r_and">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_r_r_or,SI_ftype_SISISI,3)
//
def int_hexagon_S2_asr_r_r_or :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S2_asr_r_r_or">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_r_r_or,SI_ftype_SISISI,3)
//
def int_hexagon_S2_asl_r_r_or :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S2_asl_r_r_or">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_r_r_or,SI_ftype_SISISI,3)
//
def int_hexagon_S2_lsr_r_r_or :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S2_lsr_r_r_or">;
//
// BUILTIN_INFO(HEXAGON.S2_lsl_r_r_or,SI_ftype_SISISI,3)
//
def int_hexagon_S2_lsl_r_r_or :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S2_lsl_r_r_or">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_r_p_and,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_asr_r_p_and :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_asr_r_p_and">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_r_p_and,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_asl_r_p_and :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_asl_r_p_and">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_r_p_and,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_lsr_r_p_and :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_lsr_r_p_and">;
//
// BUILTIN_INFO(HEXAGON.S2_lsl_r_p_and,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_lsl_r_p_and :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_lsl_r_p_and">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_r_p_or,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_asr_r_p_or :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_asr_r_p_or">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_r_p_or,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_asl_r_p_or :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_asl_r_p_or">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_r_p_or,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_lsr_r_p_or :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_lsr_r_p_or">;
//
// BUILTIN_INFO(HEXAGON.S2_lsl_r_p_or,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_lsl_r_p_or :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_lsl_r_p_or">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_r_p_xor,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_asr_r_p_xor :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_asr_r_p_xor">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_r_p_xor,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_asl_r_p_xor :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_asl_r_p_xor">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_r_p_xor,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_lsr_r_p_xor :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_lsr_r_p_xor">;
//
// BUILTIN_INFO(HEXAGON.S2_lsl_r_p_xor,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_lsl_r_p_xor :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_lsl_r_p_xor">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_r_r_sat,SI_ftype_SISI,2)
//
def int_hexagon_S2_asr_r_r_sat :
Hexagon_si_sisi_Intrinsic<"HEXAGON_S2_asr_r_r_sat">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_r_r_sat,SI_ftype_SISI,2)
//
def int_hexagon_S2_asl_r_r_sat :
Hexagon_si_sisi_Intrinsic<"HEXAGON_S2_asl_r_r_sat">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_i_r,SI_ftype_SISI,2)
//
def int_hexagon_S2_asr_i_r :
Hexagon_si_sisi_Intrinsic<"HEXAGON_S2_asr_i_r">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_i_r,SI_ftype_SISI,2)
//
def int_hexagon_S2_lsr_i_r :
Hexagon_si_sisi_Intrinsic<"HEXAGON_S2_lsr_i_r">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_i_r,SI_ftype_SISI,2)
//
def int_hexagon_S2_asl_i_r :
Hexagon_si_sisi_Intrinsic<"HEXAGON_S2_asl_i_r">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_i_p,DI_ftype_DISI,2)
//
def int_hexagon_S2_asr_i_p :
Hexagon_di_disi_Intrinsic<"HEXAGON_S2_asr_i_p">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_i_p,DI_ftype_DISI,2)
//
def int_hexagon_S2_lsr_i_p :
Hexagon_di_disi_Intrinsic<"HEXAGON_S2_lsr_i_p">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_i_p,DI_ftype_DISI,2)
//
def int_hexagon_S2_asl_i_p :
Hexagon_di_disi_Intrinsic<"HEXAGON_S2_asl_i_p">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_i_r_acc,SI_ftype_SISISI,3)
//
def int_hexagon_S2_asr_i_r_acc :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S2_asr_i_r_acc">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_i_r_acc,SI_ftype_SISISI,3)
//
def int_hexagon_S2_lsr_i_r_acc :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S2_lsr_i_r_acc">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_i_r_acc,SI_ftype_SISISI,3)
//
def int_hexagon_S2_asl_i_r_acc :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S2_asl_i_r_acc">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_i_p_acc,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_asr_i_p_acc :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_asr_i_p_acc">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_i_p_acc,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_lsr_i_p_acc :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_lsr_i_p_acc">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_i_p_acc,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_asl_i_p_acc :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_asl_i_p_acc">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_i_r_nac,SI_ftype_SISISI,3)
//
def int_hexagon_S2_asr_i_r_nac :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S2_asr_i_r_nac">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_i_r_nac,SI_ftype_SISISI,3)
//
def int_hexagon_S2_lsr_i_r_nac :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S2_lsr_i_r_nac">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_i_r_nac,SI_ftype_SISISI,3)
//
def int_hexagon_S2_asl_i_r_nac :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S2_asl_i_r_nac">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_i_p_nac,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_asr_i_p_nac :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_asr_i_p_nac">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_i_p_nac,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_lsr_i_p_nac :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_lsr_i_p_nac">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_i_p_nac,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_asl_i_p_nac :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_asl_i_p_nac">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_i_r_xacc,SI_ftype_SISISI,3)
//
def int_hexagon_S2_lsr_i_r_xacc :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S2_lsr_i_r_xacc">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_i_r_xacc,SI_ftype_SISISI,3)
//
def int_hexagon_S2_asl_i_r_xacc :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S2_asl_i_r_xacc">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_i_p_xacc,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_lsr_i_p_xacc :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_lsr_i_p_xacc">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_i_p_xacc,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_asl_i_p_xacc :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_asl_i_p_xacc">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_i_r_and,SI_ftype_SISISI,3)
//
def int_hexagon_S2_asr_i_r_and :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S2_asr_i_r_and">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_i_r_and,SI_ftype_SISISI,3)
//
def int_hexagon_S2_lsr_i_r_and :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S2_lsr_i_r_and">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_i_r_and,SI_ftype_SISISI,3)
//
def int_hexagon_S2_asl_i_r_and :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S2_asl_i_r_and">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_i_r_or,SI_ftype_SISISI,3)
//
def int_hexagon_S2_asr_i_r_or :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S2_asr_i_r_or">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_i_r_or,SI_ftype_SISISI,3)
//
def int_hexagon_S2_lsr_i_r_or :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S2_lsr_i_r_or">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_i_r_or,SI_ftype_SISISI,3)
//
def int_hexagon_S2_asl_i_r_or :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S2_asl_i_r_or">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_i_p_and,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_asr_i_p_and :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_asr_i_p_and">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_i_p_and,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_lsr_i_p_and :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_lsr_i_p_and">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_i_p_and,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_asl_i_p_and :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_asl_i_p_and">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_i_p_or,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_asr_i_p_or :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_asr_i_p_or">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_i_p_or,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_lsr_i_p_or :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_lsr_i_p_or">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_i_p_or,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_asl_i_p_or :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_asl_i_p_or">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_i_r_sat,SI_ftype_SISI,2)
//
def int_hexagon_S2_asl_i_r_sat :
Hexagon_si_sisi_Intrinsic<"HEXAGON_S2_asl_i_r_sat">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_i_r_rnd,SI_ftype_SISI,2)
//
def int_hexagon_S2_asr_i_r_rnd :
Hexagon_si_sisi_Intrinsic<"HEXAGON_S2_asr_i_r_rnd">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_i_r_rnd_goodsyntax,SI_ftype_SISI,2)
//
def int_hexagon_S2_asr_i_r_rnd_goodsyntax :
Hexagon_si_sisi_Intrinsic<"HEXAGON_S2_asr_i_r_rnd_goodsyntax">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_i_p_rnd,DI_ftype_DISI,2)
//
def int_hexagon_S2_asr_i_p_rnd :
Hexagon_di_disi_Intrinsic<"HEXAGON_S2_asr_i_p_rnd">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_i_p_rnd_goodsyntax,DI_ftype_DISI,2)
//
def int_hexagon_S2_asr_i_p_rnd_goodsyntax :
Hexagon_di_disi_Intrinsic<"HEXAGON_S2_asr_i_p_rnd_goodsyntax">;
//
// BUILTIN_INFO(HEXAGON.S4_lsli,SI_ftype_SISI,2)
//
def int_hexagon_S4_lsli :
Hexagon_si_sisi_Intrinsic<"HEXAGON_S4_lsli">;
//
// BUILTIN_INFO(HEXAGON.S2_addasl_rrri,SI_ftype_SISISI,3)
//
def int_hexagon_S2_addasl_rrri :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S2_addasl_rrri">;
//
// BUILTIN_INFO(HEXAGON.S4_andi_asl_ri,SI_ftype_SISISI,3)
//
def int_hexagon_S4_andi_asl_ri :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S4_andi_asl_ri">;
//
// BUILTIN_INFO(HEXAGON.S4_ori_asl_ri,SI_ftype_SISISI,3)
//
def int_hexagon_S4_ori_asl_ri :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S4_ori_asl_ri">;
//
// BUILTIN_INFO(HEXAGON.S4_addi_asl_ri,SI_ftype_SISISI,3)
//
def int_hexagon_S4_addi_asl_ri :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S4_addi_asl_ri">;
//
// BUILTIN_INFO(HEXAGON.S4_subi_asl_ri,SI_ftype_SISISI,3)
//
def int_hexagon_S4_subi_asl_ri :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S4_subi_asl_ri">;
//
// BUILTIN_INFO(HEXAGON.S4_andi_lsr_ri,SI_ftype_SISISI,3)
//
def int_hexagon_S4_andi_lsr_ri :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S4_andi_lsr_ri">;
//
// BUILTIN_INFO(HEXAGON.S4_ori_lsr_ri,SI_ftype_SISISI,3)
//
def int_hexagon_S4_ori_lsr_ri :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S4_ori_lsr_ri">;
//
// BUILTIN_INFO(HEXAGON.S4_addi_lsr_ri,SI_ftype_SISISI,3)
//
def int_hexagon_S4_addi_lsr_ri :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S4_addi_lsr_ri">;
//
// BUILTIN_INFO(HEXAGON.S4_subi_lsr_ri,SI_ftype_SISISI,3)
//
def int_hexagon_S4_subi_lsr_ri :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S4_subi_lsr_ri">;
//
// BUILTIN_INFO(HEXAGON.S2_valignib,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_valignib :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_valignib">;
//
// BUILTIN_INFO(HEXAGON.S2_valignrb,DI_ftype_DIDIQI,3)
//
def int_hexagon_S2_valignrb :
Hexagon_di_didiqi_Intrinsic<"HEXAGON_S2_valignrb">;
//
// BUILTIN_INFO(HEXAGON.S2_vspliceib,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_vspliceib :
Hexagon_di_didisi_Intrinsic<"HEXAGON_S2_vspliceib">;
//
// BUILTIN_INFO(HEXAGON.S2_vsplicerb,DI_ftype_DIDIQI,3)
//
def int_hexagon_S2_vsplicerb :
Hexagon_di_didiqi_Intrinsic<"HEXAGON_S2_vsplicerb">;
//
// BUILTIN_INFO(HEXAGON.S2_vsplatrh,DI_ftype_SI,1)
//
def int_hexagon_S2_vsplatrh :
Hexagon_di_si_Intrinsic<"HEXAGON_S2_vsplatrh">;
//
// BUILTIN_INFO(HEXAGON.S2_vsplatrb,SI_ftype_SI,1)
//
def int_hexagon_S2_vsplatrb :
Hexagon_si_si_Intrinsic<"HEXAGON_S2_vsplatrb">;
//
// BUILTIN_INFO(HEXAGON.S2_insert,SI_ftype_SISISISI,4)
//
def int_hexagon_S2_insert :
Hexagon_si_sisisisi_Intrinsic<"HEXAGON_S2_insert">;
//
// BUILTIN_INFO(HEXAGON.S2_tableidxb_goodsyntax,SI_ftype_SISISISI,4)
//
def int_hexagon_S2_tableidxb_goodsyntax :
Hexagon_si_sisisisi_Intrinsic<"HEXAGON_S2_tableidxb_goodsyntax">;
//
// BUILTIN_INFO(HEXAGON.S2_tableidxh_goodsyntax,SI_ftype_SISISISI,4)
//
def int_hexagon_S2_tableidxh_goodsyntax :
Hexagon_si_sisisisi_Intrinsic<"HEXAGON_S2_tableidxh_goodsyntax">;
//
// BUILTIN_INFO(HEXAGON.S2_tableidxw_goodsyntax,SI_ftype_SISISISI,4)
//
def int_hexagon_S2_tableidxw_goodsyntax :
Hexagon_si_sisisisi_Intrinsic<"HEXAGON_S2_tableidxw_goodsyntax">;
//
// BUILTIN_INFO(HEXAGON.S2_tableidxd_goodsyntax,SI_ftype_SISISISI,4)
//
def int_hexagon_S2_tableidxd_goodsyntax :
Hexagon_si_sisisisi_Intrinsic<"HEXAGON_S2_tableidxd_goodsyntax">;
//
// BUILTIN_INFO(HEXAGON.A4_bitspliti,DI_ftype_SISI,2)
//
def int_hexagon_A4_bitspliti :
Hexagon_di_sisi_Intrinsic<"HEXAGON_A4_bitspliti">;
//
// BUILTIN_INFO(HEXAGON.A4_bitsplit,DI_ftype_SISI,2)
//
def int_hexagon_A4_bitsplit :
Hexagon_di_sisi_Intrinsic<"HEXAGON_A4_bitsplit">;
//
// BUILTIN_INFO(HEXAGON.S4_extract,SI_ftype_SISISI,3)
//
def int_hexagon_S4_extract :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S4_extract">;
//
// BUILTIN_INFO(HEXAGON.S2_extractu,SI_ftype_SISISI,3)
//
def int_hexagon_S2_extractu :
Hexagon_si_sisisi_Intrinsic<"HEXAGON_S2_extractu">;
//
// BUILTIN_INFO(HEXAGON.S2_insertp,DI_ftype_DIDISISI,4)
//
def int_hexagon_S2_insertp :
Hexagon_di_didisisi_Intrinsic<"HEXAGON_S2_insertp">;
//
// BUILTIN_INFO(HEXAGON.S4_extractp,DI_ftype_DISISI,3)
//
def int_hexagon_S4_extractp :
Hexagon_di_disisi_Intrinsic<"HEXAGON_S4_extractp">;
//
// BUILTIN_INFO(HEXAGON.S2_extractup,DI_ftype_DISISI,3)
//
def int_hexagon_S2_extractup :
Hexagon_di_disisi_Intrinsic<"HEXAGON_S2_extractup">;
//
// BUILTIN_INFO(HEXAGON.S2_insert_rp,SI_ftype_SISIDI,3)
//
def int_hexagon_S2_insert_rp :
Hexagon_si_sisidi_Intrinsic<"HEXAGON_S2_insert_rp">;
//
// BUILTIN_INFO(HEXAGON.S4_extract_rp,SI_ftype_SIDI,2)
//
def int_hexagon_S4_extract_rp :
Hexagon_si_sidi_Intrinsic<"HEXAGON_S4_extract_rp">;
//
// BUILTIN_INFO(HEXAGON.S2_extractu_rp,SI_ftype_SIDI,2)
//
def int_hexagon_S2_extractu_rp :
Hexagon_si_sidi_Intrinsic<"HEXAGON_S2_extractu_rp">;
//
// BUILTIN_INFO(HEXAGON.S2_insertp_rp,DI_ftype_DIDIDI,3)
//
def int_hexagon_S2_insertp_rp :
Hexagon_di_dididi_Intrinsic<"HEXAGON_S2_insertp_rp">;
//
// BUILTIN_INFO(HEXAGON.S4_extractp_rp,DI_ftype_DIDI,2)
//
def int_hexagon_S4_extractp_rp :
Hexagon_di_didi_Intrinsic<"HEXAGON_S4_extractp_rp">;
//
// BUILTIN_INFO(HEXAGON.S2_extractup_rp,DI_ftype_DIDI,2)
//
def int_hexagon_S2_extractup_rp :
Hexagon_di_didi_Intrinsic<"HEXAGON_S2_extractup_rp">;
//
// BUILTIN_INFO(HEXAGON.S2_tstbit_i,QI_ftype_SISI,2)
//
def int_hexagon_S2_tstbit_i :
Hexagon_si_sisi_Intrinsic<"HEXAGON_S2_tstbit_i">;
//
// BUILTIN_INFO(HEXAGON.S4_ntstbit_i,QI_ftype_SISI,2)
//
def int_hexagon_S4_ntstbit_i :
Hexagon_si_sisi_Intrinsic<"HEXAGON_S4_ntstbit_i">;
//
// BUILTIN_INFO(HEXAGON.S2_setbit_i,SI_ftype_SISI,2)
//
def int_hexagon_S2_setbit_i :
Hexagon_si_sisi_Intrinsic<"HEXAGON_S2_setbit_i">;
//
// BUILTIN_INFO(HEXAGON.S2_togglebit_i,SI_ftype_SISI,2)
//
def int_hexagon_S2_togglebit_i :
Hexagon_si_sisi_Intrinsic<"HEXAGON_S2_togglebit_i">;
//
// BUILTIN_INFO(HEXAGON.S2_clrbit_i,SI_ftype_SISI,2)
//
def int_hexagon_S2_clrbit_i :
Hexagon_si_sisi_Intrinsic<"HEXAGON_S2_clrbit_i">;
//
// BUILTIN_INFO(HEXAGON.S2_tstbit_r,QI_ftype_SISI,2)
//
def int_hexagon_S2_tstbit_r :
Hexagon_si_sisi_Intrinsic<"HEXAGON_S2_tstbit_r">;
//
// BUILTIN_INFO(HEXAGON.S4_ntstbit_r,QI_ftype_SISI,2)
//
def int_hexagon_S4_ntstbit_r :
Hexagon_si_sisi_Intrinsic<"HEXAGON_S4_ntstbit_r">;
//
// BUILTIN_INFO(HEXAGON.S2_setbit_r,SI_ftype_SISI,2)
//
def int_hexagon_S2_setbit_r :
Hexagon_si_sisi_Intrinsic<"HEXAGON_S2_setbit_r">;
//
// BUILTIN_INFO(HEXAGON.S2_togglebit_r,SI_ftype_SISI,2)
//
def int_hexagon_S2_togglebit_r :
Hexagon_si_sisi_Intrinsic<"HEXAGON_S2_togglebit_r">;
//
// BUILTIN_INFO(HEXAGON.S2_clrbit_r,SI_ftype_SISI,2)
//
def int_hexagon_S2_clrbit_r :
Hexagon_si_sisi_Intrinsic<"HEXAGON_S2_clrbit_r">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_i_vh,DI_ftype_DISI,2)
//
def int_hexagon_S2_asr_i_vh :
Hexagon_di_disi_Intrinsic<"HEXAGON_S2_asr_i_vh">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_i_vh,DI_ftype_DISI,2)
//
def int_hexagon_S2_lsr_i_vh :
Hexagon_di_disi_Intrinsic<"HEXAGON_S2_lsr_i_vh">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_i_vh,DI_ftype_DISI,2)
//
def int_hexagon_S2_asl_i_vh :
Hexagon_di_disi_Intrinsic<"HEXAGON_S2_asl_i_vh">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_r_vh,DI_ftype_DISI,2)
//
def int_hexagon_S2_asr_r_vh :
Hexagon_di_disi_Intrinsic<"HEXAGON_S2_asr_r_vh">;
//
// BUILTIN_INFO(HEXAGON.S5_asrhub_rnd_sat_goodsyntax,SI_ftype_DISI,2)
//
def int_hexagon_S5_asrhub_rnd_sat_goodsyntax :
Hexagon_si_disi_Intrinsic<"HEXAGON_S5_asrhub_rnd_sat_goodsyntax">;
//
// BUILTIN_INFO(HEXAGON.S5_asrhub_sat,SI_ftype_DISI,2)
//
def int_hexagon_S5_asrhub_sat :
Hexagon_si_disi_Intrinsic<"HEXAGON_S5_asrhub_sat">;
//
// BUILTIN_INFO(HEXAGON.S5_vasrhrnd_goodsyntax,DI_ftype_DISI,2)
//
def int_hexagon_S5_vasrhrnd_goodsyntax :
Hexagon_di_disi_Intrinsic<"HEXAGON_S5_vasrhrnd_goodsyntax">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_r_vh,DI_ftype_DISI,2)
//
def int_hexagon_S2_asl_r_vh :
Hexagon_di_disi_Intrinsic<"HEXAGON_S2_asl_r_vh">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_r_vh,DI_ftype_DISI,2)
//
def int_hexagon_S2_lsr_r_vh :
Hexagon_di_disi_Intrinsic<"HEXAGON_S2_lsr_r_vh">;
//
// BUILTIN_INFO(HEXAGON.S2_lsl_r_vh,DI_ftype_DISI,2)
//
def int_hexagon_S2_lsl_r_vh :
Hexagon_di_disi_Intrinsic<"HEXAGON_S2_lsl_r_vh">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_i_vw,DI_ftype_DISI,2)
//
def int_hexagon_S2_asr_i_vw :
Hexagon_di_disi_Intrinsic<"HEXAGON_S2_asr_i_vw">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_i_svw_trun,SI_ftype_DISI,2)
//
def int_hexagon_S2_asr_i_svw_trun :
Hexagon_si_disi_Intrinsic<"HEXAGON_S2_asr_i_svw_trun">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_r_svw_trun,SI_ftype_DISI,2)
//
def int_hexagon_S2_asr_r_svw_trun :
Hexagon_si_disi_Intrinsic<"HEXAGON_S2_asr_r_svw_trun">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_i_vw,DI_ftype_DISI,2)
//
def int_hexagon_S2_lsr_i_vw :
Hexagon_di_disi_Intrinsic<"HEXAGON_S2_lsr_i_vw">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_i_vw,DI_ftype_DISI,2)
//
def int_hexagon_S2_asl_i_vw :
Hexagon_di_disi_Intrinsic<"HEXAGON_S2_asl_i_vw">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_r_vw,DI_ftype_DISI,2)
//
def int_hexagon_S2_asr_r_vw :
Hexagon_di_disi_Intrinsic<"HEXAGON_S2_asr_r_vw">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_r_vw,DI_ftype_DISI,2)
//
def int_hexagon_S2_asl_r_vw :
Hexagon_di_disi_Intrinsic<"HEXAGON_S2_asl_r_vw">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_r_vw,DI_ftype_DISI,2)
//
def int_hexagon_S2_lsr_r_vw :
Hexagon_di_disi_Intrinsic<"HEXAGON_S2_lsr_r_vw">;
//
// BUILTIN_INFO(HEXAGON.S2_lsl_r_vw,DI_ftype_DISI,2)
//
def int_hexagon_S2_lsl_r_vw :
Hexagon_di_disi_Intrinsic<"HEXAGON_S2_lsl_r_vw">;
//
// BUILTIN_INFO(HEXAGON.S2_vrndpackwh,SI_ftype_DI,1)
//
def int_hexagon_S2_vrndpackwh :
Hexagon_si_di_Intrinsic<"HEXAGON_S2_vrndpackwh">;
//
// BUILTIN_INFO(HEXAGON.S2_vrndpackwhs,SI_ftype_DI,1)
//
def int_hexagon_S2_vrndpackwhs :
Hexagon_si_di_Intrinsic<"HEXAGON_S2_vrndpackwhs">;
//
// BUILTIN_INFO(HEXAGON.S2_vsxtbh,DI_ftype_SI,1)
//
def int_hexagon_S2_vsxtbh :
Hexagon_di_si_Intrinsic<"HEXAGON_S2_vsxtbh">;
//
// BUILTIN_INFO(HEXAGON.S2_vzxtbh,DI_ftype_SI,1)
//
def int_hexagon_S2_vzxtbh :
Hexagon_di_si_Intrinsic<"HEXAGON_S2_vzxtbh">;
//
// BUILTIN_INFO(HEXAGON.S2_vsathub,SI_ftype_DI,1)
//
def int_hexagon_S2_vsathub :
Hexagon_si_di_Intrinsic<"HEXAGON_S2_vsathub">;
//
// BUILTIN_INFO(HEXAGON.S2_svsathub,SI_ftype_SI,1)
//
def int_hexagon_S2_svsathub :
Hexagon_si_si_Intrinsic<"HEXAGON_S2_svsathub">;
//
// BUILTIN_INFO(HEXAGON.S2_svsathb,SI_ftype_SI,1)
//
def int_hexagon_S2_svsathb :
Hexagon_si_si_Intrinsic<"HEXAGON_S2_svsathb">;
//
// BUILTIN_INFO(HEXAGON.S2_vsathb,SI_ftype_DI,1)
//
def int_hexagon_S2_vsathb :
Hexagon_si_di_Intrinsic<"HEXAGON_S2_vsathb">;
//
// BUILTIN_INFO(HEXAGON.S2_vtrunohb,SI_ftype_DI,1)
//
def int_hexagon_S2_vtrunohb :
Hexagon_si_di_Intrinsic<"HEXAGON_S2_vtrunohb">;
//
// BUILTIN_INFO(HEXAGON.S2_vtrunewh,DI_ftype_DIDI,2)
//
def int_hexagon_S2_vtrunewh :
Hexagon_di_didi_Intrinsic<"HEXAGON_S2_vtrunewh">;
//
// BUILTIN_INFO(HEXAGON.S2_vtrunowh,DI_ftype_DIDI,2)
//
def int_hexagon_S2_vtrunowh :
Hexagon_di_didi_Intrinsic<"HEXAGON_S2_vtrunowh">;
//
// BUILTIN_INFO(HEXAGON.S2_vtrunehb,SI_ftype_DI,1)
//
def int_hexagon_S2_vtrunehb :
Hexagon_si_di_Intrinsic<"HEXAGON_S2_vtrunehb">;
//
// BUILTIN_INFO(HEXAGON.S2_vsxthw,DI_ftype_SI,1)
//
def int_hexagon_S2_vsxthw :
Hexagon_di_si_Intrinsic<"HEXAGON_S2_vsxthw">;
//
// BUILTIN_INFO(HEXAGON.S2_vzxthw,DI_ftype_SI,1)
//
def int_hexagon_S2_vzxthw :
Hexagon_di_si_Intrinsic<"HEXAGON_S2_vzxthw">;
//
// BUILTIN_INFO(HEXAGON.S2_vsatwh,SI_ftype_DI,1)
//
def int_hexagon_S2_vsatwh :
Hexagon_si_di_Intrinsic<"HEXAGON_S2_vsatwh">;
//
// BUILTIN_INFO(HEXAGON.S2_vsatwuh,SI_ftype_DI,1)
//
def int_hexagon_S2_vsatwuh :
Hexagon_si_di_Intrinsic<"HEXAGON_S2_vsatwuh">;
//
// BUILTIN_INFO(HEXAGON.S2_packhl,DI_ftype_SISI,2)
//
def int_hexagon_S2_packhl :
Hexagon_di_sisi_Intrinsic<"HEXAGON_S2_packhl">;
//
// BUILTIN_INFO(HEXAGON.A2_swiz,SI_ftype_SI,1)
//
def int_hexagon_A2_swiz :
Hexagon_si_si_Intrinsic<"HEXAGON_A2_swiz">;
//
// BUILTIN_INFO(HEXAGON.S2_vsathub_nopack,DI_ftype_DI,1)
//
def int_hexagon_S2_vsathub_nopack :
Hexagon_di_di_Intrinsic<"HEXAGON_S2_vsathub_nopack">;
//
// BUILTIN_INFO(HEXAGON.S2_vsathb_nopack,DI_ftype_DI,1)
//
def int_hexagon_S2_vsathb_nopack :
Hexagon_di_di_Intrinsic<"HEXAGON_S2_vsathb_nopack">;
//
// BUILTIN_INFO(HEXAGON.S2_vsatwh_nopack,DI_ftype_DI,1)
//
def int_hexagon_S2_vsatwh_nopack :
Hexagon_di_di_Intrinsic<"HEXAGON_S2_vsatwh_nopack">;
//
// BUILTIN_INFO(HEXAGON.S2_vsatwuh_nopack,DI_ftype_DI,1)
//
def int_hexagon_S2_vsatwuh_nopack :
Hexagon_di_di_Intrinsic<"HEXAGON_S2_vsatwuh_nopack">;
//
// BUILTIN_INFO(HEXAGON.S2_shuffob,DI_ftype_DIDI,2)
//
def int_hexagon_S2_shuffob :
Hexagon_di_didi_Intrinsic<"HEXAGON_S2_shuffob">;
//
// BUILTIN_INFO(HEXAGON.S2_shuffeb,DI_ftype_DIDI,2)
//
def int_hexagon_S2_shuffeb :
Hexagon_di_didi_Intrinsic<"HEXAGON_S2_shuffeb">;
//
// BUILTIN_INFO(HEXAGON.S2_shuffoh,DI_ftype_DIDI,2)
//
def int_hexagon_S2_shuffoh :
Hexagon_di_didi_Intrinsic<"HEXAGON_S2_shuffoh">;
//
// BUILTIN_INFO(HEXAGON.S2_shuffeh,DI_ftype_DIDI,2)
//
def int_hexagon_S2_shuffeh :
Hexagon_di_didi_Intrinsic<"HEXAGON_S2_shuffeh">;
//
// BUILTIN_INFO(HEXAGON.S5_popcountp,SI_ftype_DI,1)
//
def int_hexagon_S5_popcountp :
Hexagon_si_di_Intrinsic<"HEXAGON_S5_popcountp">;
//
// BUILTIN_INFO(HEXAGON.S4_parity,SI_ftype_SISI,2)
//
def int_hexagon_S4_parity :
Hexagon_si_sisi_Intrinsic<"HEXAGON_S4_parity">;
//
// BUILTIN_INFO(HEXAGON.S2_parityp,SI_ftype_DIDI,2)
//
def int_hexagon_S2_parityp :
Hexagon_si_didi_Intrinsic<"HEXAGON_S2_parityp">;
//
// BUILTIN_INFO(HEXAGON.S2_lfsp,DI_ftype_DIDI,2)
//
def int_hexagon_S2_lfsp :
Hexagon_di_didi_Intrinsic<"HEXAGON_S2_lfsp">;
//
// BUILTIN_INFO(HEXAGON.S2_clbnorm,SI_ftype_SI,1)
//
def int_hexagon_S2_clbnorm :
Hexagon_si_si_Intrinsic<"HEXAGON_S2_clbnorm">;
//
// BUILTIN_INFO(HEXAGON.S4_clbaddi,SI_ftype_SISI,2)
//
def int_hexagon_S4_clbaddi :
Hexagon_si_sisi_Intrinsic<"HEXAGON_S4_clbaddi">;
//
// BUILTIN_INFO(HEXAGON.S4_clbpnorm,SI_ftype_DI,1)
//
def int_hexagon_S4_clbpnorm :
Hexagon_si_di_Intrinsic<"HEXAGON_S4_clbpnorm">;
//
// BUILTIN_INFO(HEXAGON.S4_clbpaddi,SI_ftype_DISI,2)
//
def int_hexagon_S4_clbpaddi :
Hexagon_si_disi_Intrinsic<"HEXAGON_S4_clbpaddi">;
//
// BUILTIN_INFO(HEXAGON.S2_clb,SI_ftype_SI,1)
//
def int_hexagon_S2_clb :
Hexagon_si_si_Intrinsic<"HEXAGON_S2_clb">;
//
// BUILTIN_INFO(HEXAGON.S2_cl0,SI_ftype_SI,1)
//
def int_hexagon_S2_cl0 :
Hexagon_si_si_Intrinsic<"HEXAGON_S2_cl0">;
//
// BUILTIN_INFO(HEXAGON.S2_cl1,SI_ftype_SI,1)
//
def int_hexagon_S2_cl1 :
Hexagon_si_si_Intrinsic<"HEXAGON_S2_cl1">;
//
// BUILTIN_INFO(HEXAGON.S2_clbp,SI_ftype_DI,1)
//
def int_hexagon_S2_clbp :
Hexagon_si_di_Intrinsic<"HEXAGON_S2_clbp">;
//
// BUILTIN_INFO(HEXAGON.S2_cl0p,SI_ftype_DI,1)
//
def int_hexagon_S2_cl0p :
Hexagon_si_di_Intrinsic<"HEXAGON_S2_cl0p">;
//
// BUILTIN_INFO(HEXAGON.S2_cl1p,SI_ftype_DI,1)
//
def int_hexagon_S2_cl1p :
Hexagon_si_di_Intrinsic<"HEXAGON_S2_cl1p">;
//
// BUILTIN_INFO(HEXAGON.S2_brev,SI_ftype_SI,1)
//
def int_hexagon_S2_brev :
Hexagon_si_si_Intrinsic<"HEXAGON_S2_brev">;
//
// BUILTIN_INFO(HEXAGON.S2_brevp,DI_ftype_DI,1)
//
def int_hexagon_S2_brevp :
Hexagon_di_di_Intrinsic<"HEXAGON_S2_brevp">;
//
// BUILTIN_INFO(HEXAGON.S2_ct0,SI_ftype_SI,1)
//
def int_hexagon_S2_ct0 :
Hexagon_si_si_Intrinsic<"HEXAGON_S2_ct0">;
//
// BUILTIN_INFO(HEXAGON.S2_ct1,SI_ftype_SI,1)
//
def int_hexagon_S2_ct1 :
Hexagon_si_si_Intrinsic<"HEXAGON_S2_ct1">;
//
// BUILTIN_INFO(HEXAGON.S2_ct0p,SI_ftype_DI,1)
//
def int_hexagon_S2_ct0p :
Hexagon_si_di_Intrinsic<"HEXAGON_S2_ct0p">;
//
// BUILTIN_INFO(HEXAGON.S2_ct1p,SI_ftype_DI,1)
//
def int_hexagon_S2_ct1p :
Hexagon_si_di_Intrinsic<"HEXAGON_S2_ct1p">;
//
// BUILTIN_INFO(HEXAGON.S2_interleave,DI_ftype_DI,1)
//
def int_hexagon_S2_interleave :
Hexagon_di_di_Intrinsic<"HEXAGON_S2_interleave">;
//
// BUILTIN_INFO(HEXAGON.S2_deinterleave,DI_ftype_DI,1)
//
def int_hexagon_S2_deinterleave :
Hexagon_di_di_Intrinsic<"HEXAGON_S2_deinterleave">;

def llvm_ptr32_ty : LLVMPointerType<llvm_i32_ty>;
def llvm_ptr64_ty : LLVMPointerType<llvm_i64_ty>;

// Mark locked loads as read/write to prevent any accidental reordering.
def int_hexagon_L2_loadw_locked :
Hexagon_Intrinsic<"HEXAGON_L2_loadw_locked", [llvm_i32_ty], [llvm_ptr32_ty],
      [IntrReadWriteArgMem, NoCapture<0>]>;
def int_hexagon_L4_loadd_locked :
Hexagon_Intrinsic<"HEXAGON_L4_loadd_locked", [llvm_i64_ty], [llvm_ptr64_ty],
      [IntrReadWriteArgMem, NoCapture<0>]>;

def int_hexagon_S2_storew_locked :
Hexagon_Intrinsic<"HEXAGON_S2_storew_locked", [llvm_i32_ty],
      [llvm_ptr32_ty, llvm_i32_ty], [IntrReadWriteArgMem, NoCapture<0>]>;
def int_hexagon_S4_stored_locked :
Hexagon_Intrinsic<"HEXAGON_S4_stored_locked", [llvm_i32_ty],
      [llvm_ptr64_ty, llvm_i64_ty], [IntrReadWriteArgMem, NoCapture<0>]>;
