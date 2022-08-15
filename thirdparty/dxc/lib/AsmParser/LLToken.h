//===- LLToken.h - Token Codes for LLVM Assembly Files ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the enums for the .ll lexer.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_ASMPARSER_LLTOKEN_H
#define LLVM_LIB_ASMPARSER_LLTOKEN_H

namespace llvm {
namespace lltok {
  enum Kind {
    // Markers
    Eof, Error,

    // Tokens with no info.
    dotdotdot,         // ...
    equal, comma,      // =  ,
    star,              // *
    lsquare, rsquare,  // [  ]
    lbrace, rbrace,    // {  }
    less, greater,     // <  >
    lparen, rparen,    // (  )
    exclaim,           // !
    bar,               // |

    kw_x,
    kw_true,    kw_false,
    kw_declare, kw_define,
    kw_global,  kw_constant,

    kw_private,
    kw_internal,
    kw_linkonce, kw_linkonce_odr,
    kw_weak, // Used as a linkage, and a modifier for "cmpxchg".
    kw_weak_odr, kw_appending,
    kw_dllimport, kw_dllexport, kw_common, kw_available_externally,
    kw_default, kw_hidden, kw_protected,
    kw_unnamed_addr,
    kw_externally_initialized,
    kw_extern_weak,
    kw_external, kw_thread_local,
    kw_localdynamic, kw_initialexec, kw_localexec,
    kw_zeroinitializer,
    kw_undef, kw_null,
    kw_to,
    kw_tail,
    kw_musttail,
    kw_target,
    kw_triple,
    kw_unwind,
    kw_deplibs,                 // FIXME: Remove in 4.0
    kw_datalayout,
    kw_volatile,
    kw_atomic,
    kw_unordered, kw_monotonic, kw_acquire, kw_release, kw_acq_rel, kw_seq_cst,
    kw_singlethread,
    kw_nnan,
    kw_ninf,
    kw_nsz,
    kw_arcp,
    kw_fast,
    kw_nuw,
    kw_nsw,
    kw_exact,
    kw_inbounds,
    kw_align,
    kw_addrspace,
    kw_section,
    kw_alias,
    kw_module,
    kw_asm,
    kw_sideeffect,
    kw_alignstack,
    kw_inteldialect,
    kw_gc,
    kw_prefix,
    kw_prologue,
    kw_c,

    kw_cc, kw_ccc, kw_fastcc, kw_coldcc,
    kw_intel_ocl_bicc,
    kw_x86_stdcallcc, kw_x86_fastcallcc, kw_x86_thiscallcc, kw_x86_vectorcallcc,
    kw_arm_apcscc, kw_arm_aapcscc, kw_arm_aapcs_vfpcc,
    kw_msp430_intrcc,
    kw_ptx_kernel, kw_ptx_device,
    kw_spir_kernel, kw_spir_func,
    kw_x86_64_sysvcc, kw_x86_64_win64cc,
    kw_webkit_jscc, kw_anyregcc,
    kw_preserve_mostcc, kw_preserve_allcc,
    kw_ghccc,

    // Attributes:
    kw_attributes,
    kw_alwaysinline,
    kw_argmemonly,
    kw_sanitize_address,
    kw_builtin,
    kw_byval,
    kw_inalloca,
    kw_cold,
    kw_convergent,
    kw_dereferenceable,
    kw_dereferenceable_or_null,
    kw_inlinehint,
    kw_inreg,
    kw_jumptable,
    kw_minsize,
    kw_naked,
    kw_nest,
    kw_noalias,
    kw_nobuiltin,
    kw_nocapture,
    kw_noduplicate,
    kw_noimplicitfloat,
    kw_noinline,
    kw_nonlazybind,
    kw_nonnull,
    kw_noredzone,
    kw_noreturn,
    kw_nounwind,
    kw_optnone,
    kw_optsize,
    kw_readnone,
    kw_readonly,
    kw_returned,
    kw_returns_twice,
    kw_signext,
    kw_ssp,
    kw_sspreq,
    kw_sspstrong,
    kw_safestack,
    kw_sret,
    kw_sanitize_thread,
    kw_sanitize_memory,
    kw_uwtable,
    kw_zeroext,

    kw_type,
    kw_opaque,

    kw_comdat,

    // Comdat types
    kw_any,
    kw_exactmatch,
    kw_largest,
    kw_noduplicates,
    kw_samesize,

    kw_eq, kw_ne, kw_slt, kw_sgt, kw_sle, kw_sge, kw_ult, kw_ugt, kw_ule,
    kw_uge, kw_oeq, kw_one, kw_olt, kw_ogt, kw_ole, kw_oge, kw_ord, kw_uno,
    kw_ueq, kw_une,

    // atomicrmw operations that aren't also instruction keywords.
    kw_xchg, kw_nand, kw_max, kw_min, kw_umax, kw_umin,

    // Instruction Opcodes (Opcode in UIntVal).
    kw_add,  kw_fadd, kw_sub,  kw_fsub, kw_mul,  kw_fmul,
    kw_udiv, kw_sdiv, kw_fdiv,
    kw_urem, kw_srem, kw_frem, kw_shl,  kw_lshr, kw_ashr,
    kw_and,  kw_or,   kw_xor,  kw_icmp, kw_fcmp,

    kw_phi, kw_call,
    kw_trunc, kw_zext, kw_sext, kw_fptrunc, kw_fpext, kw_uitofp, kw_sitofp,
    kw_fptoui, kw_fptosi, kw_inttoptr, kw_ptrtoint, kw_bitcast,
    kw_addrspacecast,
    kw_select, kw_va_arg,

    kw_landingpad, kw_personality, kw_cleanup, kw_catch, kw_filter,

    kw_ret, kw_br, kw_switch, kw_indirectbr, kw_invoke, kw_resume,
    kw_unreachable,

    kw_alloca, kw_load, kw_store, kw_fence, kw_cmpxchg, kw_atomicrmw,
    kw_getelementptr,

    kw_extractelement, kw_insertelement, kw_shufflevector,
    kw_extractvalue, kw_insertvalue, kw_blockaddress,

    // Metadata types.
    kw_distinct,

    // Use-list order directives.
    kw_uselistorder, kw_uselistorder_bb,

    // Unsigned Valued tokens (UIntVal).
    GlobalID,          // @42
    LocalVarID,        // %42
    AttrGrpID,         // #42

    // String valued tokens (StrVal).
    LabelStr,          // foo:
    GlobalVar,         // @foo @"foo"
    ComdatVar,         // $foo
    LocalVar,          // %foo %"foo"
    MetadataVar,       // !foo
    StringConstant,    // "foo"
    DwarfTag,          // DW_TAG_foo
    DwarfAttEncoding,  // DW_ATE_foo
    DwarfVirtuality,   // DW_VIRTUALITY_foo
    DwarfLang,         // DW_LANG_foo
    DwarfOp,           // DW_OP_foo
    DIFlag,            // DIFlagFoo

    // Type valued tokens (TyVal).
    Type,

    APFloat,  // APFloatVal
    APSInt // APSInt
  };
} // end namespace lltok
} // end namespace llvm

#endif
