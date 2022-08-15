//===--- HLSLOptions.td - Options for HLSL --------------------------------===//
///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// HLSLOptions.td                                                            //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
//  This file defines the options accepted by HLSL.                          //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
// Include the common option parsing interfaces.                             //

include "llvm/Option/OptParser.td"

//////////////////////////////////////////////////////////////////////////////
// Flags

// DriverOption - The option is a "driver" option, and should not be forwarded
// to other tools.
def DriverOption : OptionFlag;

// CoreOption - This is considered a "core" HLSL option.
def CoreOption : OptionFlag;

// ISenseOption - This option is only supported for IntelliSense.
def ISenseOption : OptionFlag;

// RewriteOption - This is considered a "rewriter" HLSL option.
def RewriteOption : OptionFlag;

//////////////////////////////////////////////////////////////////////////////
// Groups

// Meta-group for options which are only used for compilation,
// and not linking etc.
def CompileOnly_Group     : OptionGroup<"<CompileOnly group>">;

def Action_Group          : OptionGroup<"<action group>">;

def I_Group               : OptionGroup<"<I group>">, Group<CompileOnly_Group>;
def M_Group               : OptionGroup<"<M group>">, Group<CompileOnly_Group>;
def T_Group               : OptionGroup<"<T group>">;
def O_Group               : OptionGroup<"<O group>">, Group<CompileOnly_Group>;
def R_Group               : OptionGroup<"<R group>">, Group<CompileOnly_Group>;
def R_value_Group         : OptionGroup<"<R (with value) group>">, Group<R_Group>;
def W_Group               : OptionGroup<"<W group>">, Group<CompileOnly_Group>, HelpText<"Warning Options">;
def W_value_Group         : OptionGroup<"<W (with value) group>">, Group<W_Group>, HelpText<"Warning Options">;
def d_Group               : OptionGroup<"<d group>">;
def f_Group               : OptionGroup<"<f group>">, Group<CompileOnly_Group>;
def f_clang_Group         : OptionGroup<"<f (clang-only) group>">, Group<CompileOnly_Group>;
def g_Group               : OptionGroup<"<g group>">;
def g_flags_Group         : OptionGroup<"<g flags group>">;
def i_Group               : OptionGroup<"<i group>">, Group<CompileOnly_Group>;
def clang_i_Group         : OptionGroup<"<clang i group>">, Group<i_Group>;
def m_Group               : OptionGroup<"<m group>">, Group<CompileOnly_Group>;
def u_Group               : OptionGroup<"<u group>">;

// Developer Driver Options

def internal_Group : OptionGroup<"<clang internal options>">;
def internal_driver_Group : OptionGroup<"<clang driver internal options>">,
  Group<internal_Group>, HelpText<"DRIVER OPTIONS">;
def internal_debug_Group :
  OptionGroup<"<clang debug/development internal options>">,
  Group<internal_Group>, HelpText<"DEBUG/DEVELOPMENT OPTIONS">;

def hlslcomp_Group : OptionGroup<"HLSL Compilation">, HelpText<"Compilation Options">;
def hlsloptz_Group : OptionGroup<"HLSL Optimization">, HelpText<"Optimization Options">;
def hlslutil_Group : OptionGroup<"HLSL Utility">, HelpText<"Utility Options">;
def hlslcore_Group : OptionGroup<"HLSL Core">, HelpText<"Common Options">;
def hlslrewrite_Group : OptionGroup<"HLSL Rewriter">, HelpText<"Rewriter Options">;

def spirv_Group : OptionGroup<"SPIR-V CodeGen">, HelpText<"SPIR-V CodeGen Options">; // SPIRV Change

//////////////////////////////////////////////////////////////////////////////
// Options

// The internal option ID must be a valid C++ identifier and results in a
// hlsl::options::OPT_XX enum constant for XX.
//
// See tools/clang/include/clang/Driver/Options.td for more notes.
// These options are based on those used by clang.
// fxc-based options kept for compatiblity are below.
//
// The general approach is to include only things that are in use, in the
// same order as in Options.td.

def D : JoinedOrSeparate<["-", "/"], "D">, Group<hlslcomp_Group>, Flags<[CoreOption, RewriteOption]>,
    HelpText<"Define macro">;
def H : Flag<["-"], "H">, Flags<[CoreOption]>, Group<hlslcomp_Group>,
    HelpText<"Show header includes and nesting depth">;
def I : JoinedOrSeparate<["-", "/"], "I">, Group<hlslcomp_Group>, Flags<[CoreOption, RewriteOption]>,
    HelpText<"Add directory to include search path">;
def O0 : Flag<["-", "/"], "O0">, Group<hlsloptz_Group>, Flags<[CoreOption]>,
    HelpText<"Optimization Level 0">;
def O1 : Flag<["-", "/"], "O1">, Group<hlsloptz_Group>, Flags<[CoreOption]>,
    HelpText<"Optimization Level 1">;
def O2 : Flag<["-", "/"], "O2">, Group<hlsloptz_Group>, Flags<[CoreOption]>,
    HelpText<"Optimization Level 2">;
def O3 : Flag<["-", "/"], "O3">, Group<hlsloptz_Group>, Flags<[CoreOption]>,
    HelpText<"Optimization Level 3 (Default)">;
def Odump : Flag<["-", "/"], "Odump">, Group<hlslcomp_Group>, Flags<[CoreOption]>,
    HelpText<"Print the optimizer commands.">;
def Qunused_arguments : Flag<["-"], "Qunused-arguments">, Group<hlslcore_Group>, Flags<[CoreOption]>,
  HelpText<"Don't emit warning for unused driver arguments">;
def Wall : Flag<["-"], "Wall">, Group<W_Group>, Flags<[CoreOption]>;
def Wdeprecated : Flag<["-"], "Wdeprecated">, Group<W_Group>, Flags<[CoreOption]>;
def W_Joined : Joined<["-"], "W">, Group<W_Group>, Flags<[CoreOption]>,
  MetaVarName<"[no-]<warning>">, HelpText<"Enable/Disable the specified warning">;
def d_Flag : Flag<["-"], "d">, Group<d_Group>;
def d_Joined : Joined<["-"], "d">, Group<d_Group>;
//def fcolor_diagnostics : Flag<["-"], "fcolor-diagnostics">, Group<hlslcomp_Group>,
//  Flags<[CoreOption, DriverOption]>, HelpText<"Use colors in diagnostics">;
def fdiagnostics_color : Flag<["-"], "fdiagnostics-color">, Group<hlslcomp_Group>,
  Flags<[CoreOption, DriverOption]>;
def fdiagnostics_color_EQ : Joined<["-"], "fdiagnostics-color=">, Group<hlslcomp_Group>;
//def fansi_escape_codes : Flag<["-"], "fansi-escape-codes">, Group<f_Group>, // HLSL Change: disable ANSI escape codes
//  Flags<[CoreOption, CC1Option]>, HelpText<"Use ANSI escape codes for diagnostics">;
def fconstexpr_depth_EQ : Joined<["-"], "fconstexpr-depth=">, Group<f_Group>;
def fconstexpr_steps_EQ : Joined<["-"], "fconstexpr-steps=">, Group<f_Group>;
def fconstexpr_backtrace_limit_EQ : Joined<["-"], "fconstexpr-backtrace-limit=">,
                                    Group<f_Group>;
def fdiagnostics_show_option : Flag<["-"], "fdiagnostics-show-option">, Group<hlslcomp_Group>,
    Flags<[CoreOption]>, HelpText<"Print option name with mappable diagnostics">;
def fno_diagnostics_show_option : Flag<["-"], "fno-diagnostics-show-option">, Group<hlslcomp_Group>,
    Flags<[CoreOption]>, HelpText<"Do not print option name with mappable diagnostics">;
def fdiagnostics_show_category_EQ : Joined<["-"], "fdiagnostics-show-category=">, Group<hlslcomp_Group>;
def ferror_limit_EQ : Joined<["-"], "ferror-limit=">, Group<hlslcomp_Group>, Flags<[CoreOption]>;

def import_binding_table : JoinedOrSeparate<["-", "/"], "import-binding-table">, Group<hlsloptz_Group>, Flags<[CoreOption, DriverOption, HelpHidden]>,
  HelpText<"Import a binding table to specify resource bindings.">;
def binding_table_define : Separate<["-", "/"], "binding-table-define">, Group<hlsloptz_Group>, Flags<[CoreOption, DriverOption, HelpHidden]>,
  HelpText<"Import a binding table from a define to specify resource bindings.">;

def funsafe_math_optimizations : Flag<["-"], "funsafe-math-optimizations">,
  Group<hlsloptz_Group>;
def fno_unsafe_math_optimizations : Flag<["-"], "fno-unsafe-math-optimizations">,
  Group<hlsloptz_Group>;
def fassociative_math : Flag<["-"], "fassociative-math">, Group<hlsloptz_Group>;
def fno_associative_math : Flag<["-"], "fno-associative-math">, Group<hlsloptz_Group>;
//def freciprocal_math :
//  Flag<["-"], "freciprocal-math">, Group<hlsloptz_Group>, Flags<[CoreOption]>,
//  HelpText<"Allow division operations to be reassociated">;
def fno_reciprocal_math : Flag<["-"], "fno-reciprocal-math">, Group<hlsloptz_Group>;
def ffinite_math_only : Flag<["-"], "ffinite-math-only">, Group<hlsloptz_Group>, Flags<[CoreOption]>;
def fno_finite_math_only : Flag<["-"], "fno-finite-math-only">, Group<hlsloptz_Group>;
def fsigned_zeros : Flag<["-"], "fsigned-zeros">, Group<hlsloptz_Group>;
//def fno_signed_zeros :
//  Flag<["-"], "fno-signed-zeros">, Group<hlsloptz_Group>, Flags<[CoreOption]>,
//  HelpText<"Allow optimizations that ignore the sign of floating point zeros">;
def fhonor_nans : Flag<["-"], "fhonor-nans">, Group<hlsloptz_Group>;
def fno_honor_nans : Flag<["-"], "fno-honor-nans">, Group<hlsloptz_Group>;
def fhonor_infinities : Flag<["-"], "fhonor-infinities">, Group<hlsloptz_Group>;
def fno_honor_infinities : Flag<["-"], "fno-honor-infinities">, Group<hlsloptz_Group>;
//def ftrapping_math : Flag<["-"], "ftrapping-math">, Group<f_Group>;
//def fno_trapping_math : Flag<["-"], "fno-trapping-math">, Group<f_Group>;
//def ffp_contract : Joined<["-"], "ffp-contract=">, Group<hlsloptz_Group>,
//  Flags<[CoreOption]>, HelpText<"Form fused FP ops (e.g. FMAs): fast (everywhere)"
//  " | on (according to FP_CONTRACT pragma, default) | off (never fuse)">;
def flimited_precision_EQ : Joined<["-"], "flimited-precision=">, Group<hlsloptz_Group>;
def memdep_block_scan_limit : Separate<["-", "/"], "memdep-block-scan-limit">, Group<hlsloptz_Group>, Flags<[CoreOption, DriverOption, HelpHidden]>,
  HelpText<"The number of instructions to scan in a block in memory dependency analysis.">;
def opt_disable : Separate<["-", "/"], "opt-disable">, Group<hlsloptz_Group>, Flags<[CoreOption, DriverOption, HelpHidden]>,
  HelpText<"Disable this optimization.">;
def opt_enable : Separate<["-", "/"], "opt-enable">, Group<hlsloptz_Group>, Flags<[CoreOption, DriverOption, HelpHidden]>,
  HelpText<"Enable this optimization.">;
def opt_select : MultiArg<["-", "/"], "opt-select", 2>, MetaVarName<"<opt> <variant>">, Group<hlsloptz_Group>, Flags<[CoreOption, DriverOption, HelpHidden]>,
  HelpText<"Select this optimization variant.">;

/*
def fno_caret_diagnostics : Flag<["-"], "fno-caret-diagnostics">, Group<hlslcomp_Group>,
 Flags<[CoreOption]>;
def fno_color_diagnostics : Flag<["-"], "fno-color-diagnostics">, Group<hlslcomp_Group>,
  Flags<[CoreOption]>;
def fno_diagnostics_color : Flag<["-"], "fno-diagnostics-color">, Group<hlslcomp_Group>,
  Flags<[CoreOption, DriverOption]>;
def fno_show_column : Flag<["-"], "fno-show-column">, Group<hlslcomp_Group>, Flags<[CoreOption]>,
  HelpText<"Do not include column number on diagnostics">;
def fno_spell_checking : Flag<["-"], "fno-spell-checking">, Group<hlslcomp_Group>,
  Flags<[CoreOption]>, HelpText<"Disable spell-checking">;
def fshow_column : Flag<["-"], "fshow-column">, Group<hlslcomp_Group>, Flags<[CoreOption]>;
def fshow_source_location : Flag<["-"], "fshow-source-location">, Group<hlslcomp_Group>;
def fspell_checking : Flag<["-"], "fspell-checking">, Group<hlslcomp_Group>;
def fspell_checking_limit_EQ : Joined<["-"], "fspell-checking-limit=">, Group<hlslcomp_Group>;
def funroll_loops : Flag<["-"], "funroll-loops">, Group<hlsloptz_Group>,
  HelpText<"Turn on loop unroller">, Flags<[CoreOption]>;
def fno_unroll_loops : Flag<["-"], "fno-unroll-loops">, Group<hlsloptz_Group>,
  HelpText<"Turn off loop unroller">, Flags<[CoreOption]>;
def freroll_loops : Flag<["-"], "freroll-loops">, Group<hlsloptz_Group>,
  HelpText<"Turn on loop reroller">, Flags<[CoreOption]>;
def fno_reroll_loops : Flag<["-"], "fno-reroll-loops">, Group<hlsloptz_Group>,
  HelpText<"Turn off loop reroller">;
*/
def help : Flag<["-", "--", "/"], "help">, Flags<[DriverOption]>, Group<hlslcore_Group>,
  HelpText<"Display available options">;
/*
def imacros : JoinedOrSeparate<["-", "--"], "imacros">, Group<clang_i_Group>, Flags<[CoreOption]>,
  HelpText<"Include macros from file before parsing">, MetaVarName<"<file>">;
def include_ : JoinedOrSeparate<["-", "--"], "include">, Group<clang_i_Group>, EnumName<"include">,
    MetaVarName<"<file>">, HelpText<"Include file before parsing">, Flags<[CoreOption]>;
def mllvm : Separate<["-"], "mllvm">, Flags<[CoreOption]>,
  HelpText<"Additional arguments to forward to LLVM's option processing">;
def mms_bitfields : Flag<["-"], "mms-bitfields">, Group<m_Group>, Flags<[CoreOption]>,
  HelpText<"Set the default structure layout to be compatible with the Microsoft compiler standard">;
def o : JoinedOrSeparate<["-"], "o">, Flags<[DriverOption, RenderAsInput, CoreOption]>,
  HelpText<"Write output to <file>">, MetaVarName<"<file>">;
def time : Flag<["-"], "time">,
  HelpText<"Time individual commands">;
// def U : JoinedOrSeparate<["-"], "U">, Group<CompileOnly_Group>, Flags<[CoreOption]>;
def w : Flag<["-"], "w">, HelpText<"Suppress all warnings">, Flags<[CoreOption]>;
def working_directory : JoinedOrSeparate<["-"], "working-directory">, Flags<[CoreOption]>,
  MetaVarName<"<dir>">, HelpText<"Resolve file paths relative to the specified directory">;
def working_directory_EQ : Joined<["-"], "working-directory=">, Flags<[CoreOption]>,
  MetaVarName<"<dir>">, Alias<working_directory>;
*/

// Double dash options, which are usually an alias for one of the previous
// options.

def _all_warnings : Flag<["--"], "all-warnings">, Flags<[CoreOption]>, Alias<Wall>;
def _help_hidden : Flag<["--"], "help-hidden">, Flags<[DriverOption]>;
def _help_question : Flag<["-", "/"], "?">, Flags<[DriverOption]>, Alias<help>;
def _version : Flag<["--"], "version">, Group<hlslcore_Group>, Flags<[DriverOption]>,
  HelpText<"Display compiler version information">;

//////////////////////////////////////////////////////////////////////////////
// New HLSL-specific flags.

def ast_dump : Flag<["-", "/"], "ast-dump">, Flags<[CoreOption, DriverOption, HelpHidden]>,
  HelpText<"Dumps the parsed Abstract Syntax Tree.">; // should not be core, but handy workaround until explicit API written
def dump_dependencies : Flag<["-", "/"], "M">, Flags<[CoreOption, DriverOption]>,
  HelpText<"Dumps the list of the compilation target dependencies.">;
def write_dependencies : Flag<["-", "/"], "MD">, Flags<[CoreOption, DriverOption]>,
  HelpText<"Write a file with .d extension that will contain the list of the compilation target dependencies.">;
def write_dependencies_to : JoinedOrSeparate<["-", "/"], "MF">, MetaVarName<"<file>">, Flags<[CoreOption, DriverOption]>,
  HelpText<"Write the specfied file that will contain the list of the compilation target dependencies.">;
def external_lib : Separate<["-", "/"], "external">, Group<hlslcore_Group>, Flags<[DriverOption, RewriteOption, HelpHidden]>,
  HelpText<"External DLL name to load for compiler support">;
def external_fn : Separate<["-", "/"], "external-fn">, Group<hlslcore_Group>, Flags<[DriverOption, RewriteOption, HelpHidden]>,
  HelpText<"External function name to load for compiler support">;
def fcgl : Flag<["-", "/"], "fcgl">, Group<hlslcore_Group>, Flags<[CoreOption, HelpHidden]>,
  HelpText<"Generate high-level code only">;
def preserve_intermediate_values : Flag<["-", "/"], "preserve-intermediate-values">, Group<hlslcore_Group>, Flags<[CoreOption, HelpHidden]>,
  HelpText<"Preserve intermediate values to help shader debugging">;
def flegacy_macro_expansion : Flag<["-", "/"], "flegacy-macro-expansion">, Group<hlslcomp_Group>, Flags<[CoreOption, RewriteOption, DriverOption]>,
    HelpText<"Expand the operands before performing token-pasting operation (fxc behavior)">;
def flegacy_resource_reservation : Flag<["-", "/"], "flegacy-resource-reservation">, Group<hlslcomp_Group>, Flags<[CoreOption, DriverOption]>,
    HelpText<"Reserve unused explicit register assignments for compatibility with shader model 5.0 and below">;
def no_legacy_cbuf_layout : Flag<["-", "/"], "no-legacy-cbuf-layout">, Group<hlslcomp_Group>, Flags<[CoreOption]>,
  HelpText<"Do not use legacy cbuffer load">;
def not_use_legacy_cbuf_load_ : Flag<["-", "/"], "not_use_legacy_cbuf_load">, Group<hlslcomp_Group>, Flags<[CoreOption, HelpHidden]>,
  HelpText<"Do not use legacy cbuffer load">;
def pack_prefix_stable : Flag<["-", "/"], "pack-prefix-stable">, Group<hlslcomp_Group>, Flags<[CoreOption]>,
  HelpText<"(default) Pack signatures preserving prefix-stable property - appended elements will not disturb placement of prior elements">;
def pack_prefix_stable_ : Flag<["-", "/"], "pack_prefix_stable">, Group<hlslcomp_Group>, Flags<[CoreOption, HelpHidden]>,
  HelpText<"(default) Pack signatures preserving prefix-stable property - appended elements will not disturb placement of prior elements">;
def pack_optimized : Flag<["-", "/"], "pack-optimized">, Group<hlslcomp_Group>, Flags<[CoreOption]>,
  HelpText<"Optimize signature packing assuming identical signature provided for each connecting stage">;
def pack_optimized_ : Flag<["-", "/"], "pack_optimized">, Group<hlslcomp_Group>, Flags<[CoreOption, HelpHidden]>,
  HelpText<"Optimize signature packing assuming identical signature provided for each connecting stage">;
def hlsl_version : Separate<["-", "/"], "HV">, Group<hlslcomp_Group>, Flags<[CoreOption, RewriteOption]>,
  HelpText<"HLSL version (2016, 2017, 2018, 2021). Default is 2018">;
def no_warnings : Flag<["-", "/"], "no-warnings">, Group<hlslcomp_Group>, Flags<[CoreOption, RewriteOption]>,
  HelpText<"Suppress warnings">;
def rootsig_define : Separate<["-", "/"], "rootsig-define">, Group<hlslcomp_Group>, Flags<[CoreOption]>,
  HelpText<"Read root signature from a #define">;
def enable_16bit_types: Flag<["-", "/"], "enable-16bit-types">, Flags<[CoreOption, RewriteOption, DriverOption]>, Group<hlslcomp_Group>,
  HelpText<"Enable 16bit types and disable min precision types. Available in HLSL 2018 and shader model 6.2">;
def ignore_line_directives : Flag<["-", "/"], "ignore-line-directives">, HelpText<"Ignore line directives">, Flags<[CoreOption]>, Group<hlslcomp_Group>;
def auto_binding_space : Separate<["-", "/"], "auto-binding-space">, Group<hlslcomp_Group>, Flags<[CoreOption]>,
  HelpText<"Set auto binding space - enables auto resource binding in libraries">;
def exports : Separate<["-", "/"], "exports">, Group<hlslcomp_Group>, Flags<[CoreOption]>,
  HelpText<"Specify exports when compiling a library: export1[[,export1_clone,...]=internal_name][;...]">;
def export_shaders_only : Flag<["-", "/"], "export-shaders-only">, Group<hlslcomp_Group>, Flags<[CoreOption]>,
  HelpText<"Only export shaders when compiling a library">;
def default_linkage : Separate<["-", "/"], "default-linkage">, Group<hlslcomp_Group>, Flags<[CoreOption]>,
  HelpText<"Set default linkage for non-shader functions when compiling or linking to a library target (internal, external)">;
def precise_output : Separate<["-", "/"], "precise-output">, Group<hlslcomp_Group>, Flags<[CoreOption, HelpHidden]>,
    HelpText<"Mark output semantic as precise">;
def encoding : Separate<["-", "/"], "encoding">, Group<hlslcomp_Group>, Flags<[CoreOption, RewriteOption, DriverOption]>,
  HelpText<"Set default encoding for source inputs and text outputs (utf8|utf16(win)|utf32(*nix)|wide) default=utf8">;
def validator_version : Separate<["-", "/"], "validator-version">, Group<hlslcomp_Group>, Flags<[CoreOption, HelpHidden]>,
  HelpText<"Override validator version for module.  Format: <major.minor> ; Default: DXIL.dll version or current internal version.">;
def print_after_all : Flag<["-", "/"], "print-after-all">, Group<hlslcomp_Group>, Flags<[CoreOption, HelpHidden]>,
  HelpText<"Print LLVM IR after each pass.">;
def print_after : Separate<["-", "/"], "print-after">, Group<hlslcomp_Group>, Flags<[CoreOption, HelpHidden]>,
  HelpText<"Print LLVM IR after a specific pass.">;
def ignore_opt_semdefs : Flag<["-", "/"], "ignore-opt-semdefs">, Group<hlslcomp_Group>, Flags<[CoreOption, HelpHidden]>,
  HelpText<"Ignore optional semantic defines which are not required to ensure program correctness.">;
def ignore_semdef : Separate<["-", "/"], "ignore-semdef">, Group<hlslcomp_Group>, Flags<[CoreOption, HelpHidden]>,
  HelpText<"Ignore the semantic define passed as a value to this flag.">;
def override_semdef : Separate<["-", "/"], "override-semdef">, Group<hlslcomp_Group>, Flags<[CoreOption, HelpHidden]>,
  HelpText<"Override the value of semantic define with the one passed with this flag.">;
def force_zero_store_lifetimes : Flag<["-", "/"], "force-zero-store-lifetimes">, Group<hlslcomp_Group>, Flags<[CoreOption, HelpHidden]>,
  HelpText<"Instead of generating lifetime intrinsics (SM >= 6.6) or storing undef (SM < 6.6), force fall back to storing zeroinitializer.">;
def enable_lifetime_markers : Flag<["-", "/"], "enable-lifetime-markers">, Group<hlslcomp_Group>, Flags<[CoreOption]>,
  HelpText<"Enable generation of lifetime markers">;
def disable_lifetime_markers : Flag<["-", "/"], "disable-lifetime-markers">, Group<hlslcomp_Group>, Flags<[CoreOption, HelpHidden]>,
  HelpText<"Disable generation of lifetime markers where they would be otherwise (6.6+)">;
def enable_templates: Flag<["-", "/"], "enable-templates">, Group<hlslcomp_Group>, Flags<[CoreOption, HelpHidden]>,
  HelpText<"Enable template support for HLSL.">;
def enable_operator_overloading: Flag<["-", "/"], "enable-operator-overloading">, Group<hlslcomp_Group>, Flags<[CoreOption, HelpHidden]>,
  HelpText<"Enable operator overloading support for HLSL.">;
def strict_udt_casting: Flag<["-", "/"], "strict-udt-casting">, Group<hlslcomp_Group>, Flags<[CoreOption, HelpHidden]>,
  HelpText<"Require explicit casts between different structure types with compatible layouts.">;
def enable_payload_qualifiers : Flag<["-", "/"], "enable-payload-qualifiers">, Group<hlslcomp_Group>, Flags<[CoreOption, RewriteOption, DriverOption]>,
  HelpText<"Enables support for payload access qualifiers for raytracing payloads in SM 6.6.">;
def disable_payload_qualifiers : Flag<["-", "/"], "disable-payload-qualifiers">, Group<hlslcomp_Group>, Flags<[CoreOption, RewriteOption, DriverOption]>,
  HelpText<"Disables support for payload access qualifiers for raytracing payloads in SM 6.7.">;
def disable_exception_handling : Flag<["-", "/"], "disable-exception-handling">, Group<hlslcomp_Group>, Flags<[DriverOption, HelpHidden]>,
  HelpText<"Disable dxc handling of exceptions">;

// Used with API only
def skip_serialization : Flag<["-", "/"], "skip-serialization">, Group<hlslcore_Group>, Flags<[CoreOption, HelpHidden]>,
  HelpText<"Return a module interface instead of serialized output">;

// SPIRV Change Starts
def spirv : Flag<["-"], "spirv">, Group<spirv_Group>, Flags<[CoreOption, DriverOption]>,
  HelpText<"Generate SPIR-V code">;
def fvk_stage_io_order_EQ : Joined<["-"], "fvk-stage-io-order=">, Group<spirv_Group>, Flags<[CoreOption, DriverOption, HelpHidden]>,
  HelpText<"Specify Vulkan stage I/O location assignment order">;
def fvk_b_shift : MultiArg<["-"], "fvk-b-shift", 2>, MetaVarName<"<shift> <space>">, Group<spirv_Group>, Flags<[CoreOption, DriverOption]>,
  HelpText<"Specify Vulkan binding number shift for b-type register">;
def fvk_t_shift : MultiArg<["-"], "fvk-t-shift", 2>, MetaVarName<"<shift> <space>">, Group<spirv_Group>, Flags<[CoreOption, DriverOption]>,
  HelpText<"Specify Vulkan binding number shift for t-type register">;
def fvk_s_shift : MultiArg<["-"], "fvk-s-shift", 2>, MetaVarName<"<shift> <space>">, Group<spirv_Group>, Flags<[CoreOption, DriverOption]>,
  HelpText<"Specify Vulkan binding number shift for s-type register">;
def fvk_u_shift : MultiArg<["-"], "fvk-u-shift", 2>, MetaVarName<"<shift> <space>">, Group<spirv_Group>, Flags<[CoreOption, DriverOption]>,
  HelpText<"Specify Vulkan binding number shift for u-type register">;
def fvk_bind_globals : MultiArg<["-"], "fvk-bind-globals", 2>, MetaVarName<"<binding> <set>">, Group<spirv_Group>, Flags<[CoreOption, DriverOption]>,
  HelpText<"Specify Vulkan binding number and set number for the $Globals cbuffer">;
def fvk_bind_register : MultiArg<["-"], "fvk-bind-register", 4>, MetaVarName<"<type-number> <space> <binding> <set>">, Group<spirv_Group>, Flags<[CoreOption, DriverOption]>,
  HelpText<"Specify Vulkan descriptor set and binding for a specific register">;
def vkbr : MultiArg<["-"], "vkbr", 4>, Flags<[CoreOption, DriverOption]>, Alias<fvk_bind_register>;
def fvk_invert_y: Flag<["-"], "fvk-invert-y">, Group<spirv_Group>, Flags<[CoreOption, DriverOption]>,
  HelpText<"Negate SV_Position.y before writing to stage output in VS/DS/GS to accommodate Vulkan's coordinate system">;
def fvk_use_dx_position_w: Flag<["-"], "fvk-use-dx-position-w">, Group<spirv_Group>, Flags<[CoreOption, DriverOption]>,
  HelpText<"Reciprocate SV_Position.w after reading from stage input in PS to accommodate the difference between Vulkan and DirectX">;
def fvk_support_nonzero_base_instance: Flag<["-"], "fvk-support-nonzero-base-instance">, Group<spirv_Group>, Flags<[CoreOption, DriverOption]>,
  HelpText<"Follow Vulkan spec to use gl_BaseInstance as the first vertex instance, which makes SV_InstanceID = gl_InstanceIndex - gl_BaseInstance (without this option, SV_InstanceID = gl_InstanceIndex)">;
def fvk_use_gl_layout: Flag<["-"], "fvk-use-gl-layout">, Group<spirv_Group>, Flags<[CoreOption, DriverOption]>,
  HelpText<"Use strict OpenGL std140/std430 memory layout for Vulkan resources">;
def fvk_use_dx_layout: Flag<["-"], "fvk-use-dx-layout">, Group<spirv_Group>, Flags<[CoreOption, DriverOption]>,
  HelpText<"Use DirectX memory layout for Vulkan resources">;
def fvk_use_scalar_layout: Flag<["-"], "fvk-use-scalar-layout">, Group<spirv_Group>, Flags<[CoreOption, DriverOption]>,
  HelpText<"Use scalar memory layout for Vulkan resources">;
def fspv_use_legacy_buffer_matrix_order: Flag<["-"], "fspv-use-legacy-buffer-matrix-order">, Group<spirv_Group>, Flags<[CoreOption, DriverOption]>,
  HelpText<"Assume the legacy matrix order (row major) when accessing raw buffers (e.g., ByteAdddressBuffer)">;
def fspv_reflect: Flag<["-"], "fspv-reflect">, Group<spirv_Group>, Flags<[CoreOption, DriverOption]>,
  HelpText<"Emit additional SPIR-V instructions to aid reflection">;
def fspv_debug_EQ : Joined<["-"], "fspv-debug=">, Group<spirv_Group>, Flags<[CoreOption, DriverOption]>,
  HelpText<"Specify whitelist of debug info category (file -> source -> line, tool)">;
def fspv_extension_EQ : Joined<["-"], "fspv-extension=">, Group<spirv_Group>, Flags<[CoreOption, DriverOption]>,
  HelpText<"Specify SPIR-V extension permitted to use">;
def fspv_target_env_EQ : Joined<["-"], "fspv-target-env=">, Group<spirv_Group>, Flags<[CoreOption, DriverOption]>,
  HelpText<"Specify the target environment: vulkan1.0 (default), vulkan1.1, vulkan1.1spirv1.4, vulkan1.2, vulkan1.3, or universal1.5">;
def fspv_flatten_resource_arrays: Flag<["-"], "fspv-flatten-resource-arrays">, Group<spirv_Group>, Flags<[CoreOption, DriverOption]>,
  HelpText<"Flatten arrays of resources so each array element takes one binding number">;
def fspv_reduce_load_size: Flag<["-"], "fspv-reduce-load-size">, Group<spirv_Group>, Flags<[CoreOption, DriverOption]>,
  HelpText<"Replaces loads of composite objects to reduce memory pressure for the loads">;
def fspv_fix_func_call_arguments: Flag<["-"], "fspv-fix-func-call-arguments">, Group<spirv_Group>, Flags<[CoreOption, DriverOption, HelpHidden]>,
  HelpText<"Fix function call arguments which are not memory objects">;
def fspv_entrypoint_name_EQ : Joined<["-"], "fspv-entrypoint-name=">, Group<spirv_Group>, Flags<[CoreOption, DriverOption]>,
  HelpText<"Specify the SPIR-V entry point name. Defaults to the HLSL entry point name.">;
def fvk_auto_shift_bindings: Flag<["-"], "fvk-auto-shift-bindings">, Group<spirv_Group>, Flags<[CoreOption, DriverOption]>,
  HelpText<"Apply fvk-*-shift to resources without an explicit register assignment.">;
def Wno_vk_ignored_features : Joined<["-"], "Wno-vk-ignored-features">, Group<spirv_Group>, Flags<[CoreOption, DriverOption, HelpHidden]>,
  HelpText<"Do not emit warnings for ingored features resulting from no Vulkan support">;
def Wno_vk_emulated_features : Joined<["-"], "Wno-vk-emulated-features">, Group<spirv_Group>, Flags<[CoreOption, DriverOption, HelpHidden]>,
  HelpText<"Do not emit warnings for emulated features resulting from no direct mapping">;
def fspv_print_all: Flag<["-"], "fspv-print-all">, Group<spirv_Group>, Flags<[CoreOption, DriverOption]>,
  HelpText<"Print the SPIR-V module before each pass and after the last one. Useful for debugging SPIR-V legalization and optimization passes.">;
def Oconfig : CommaJoined<["-"], "Oconfig=">, Group<spirv_Group>, Flags<[CoreOption]>,
  HelpText<"Specify a comma-separated list of SPIRV-Tools passes to customize optimization configuration (see http://khr.io/hlsl2spirv#optimization)">;
// SPIRV Change Ends

//////////////////////////////////////////////////////////////////////////////
// fxc-based flags that don't match those previously defined.

def target_profile : JoinedOrSeparate<["-", "/"], "T">, Flags<[CoreOption]>, Group<hlslcomp_Group>, MetaVarName<"<profile>">,
  /* <py::lines('VALRULE-TEXT')>hctdb_instrhelp.get_target_profiles()</py>*/
  // VALRULE-TEXT:BEGIN
  HelpText<"Set target profile. \n\t<profile>: ps_6_0, ps_6_1, ps_6_2, ps_6_3, ps_6_4, ps_6_5, ps_6_6, ps_6_7, \n\t\t vs_6_0, vs_6_1, vs_6_2, vs_6_3, vs_6_4, vs_6_5, vs_6_6, vs_6_7, \n\t\t gs_6_0, gs_6_1, gs_6_2, gs_6_3, gs_6_4, gs_6_5, gs_6_6, gs_6_7, \n\t\t hs_6_0, hs_6_1, hs_6_2, hs_6_3, hs_6_4, hs_6_5, hs_6_6, hs_6_7, \n\t\t ds_6_0, ds_6_1, ds_6_2, ds_6_3, ds_6_4, ds_6_5, ds_6_6, ds_6_7, \n\t\t cs_6_0, cs_6_1, cs_6_2, cs_6_3, cs_6_4, cs_6_5, cs_6_6, cs_6_7, \n\t\t lib_6_1, lib_6_2, lib_6_3, lib_6_4, lib_6_5, lib_6_6, lib_6_7, \n\t\t ms_6_5, ms_6_6, ms_6_7, \n\t\t as_6_5, as_6_6, as_6_7, \n\t\t ">;
  // VALRULE-TEXT:END
def entrypoint :  JoinedOrSeparate<["-", "/"], "E">, Flags<[CoreOption, RewriteOption]>, Group<hlslcomp_Group>,
  HelpText<"Entry point name">;
// /I <include> - already defined above
def _vi : Flag<["-", "/"], "Vi">, Alias<H>, Flags<[CoreOption]>, Group<hlslcomp_Group>,
  HelpText<"Display details about the include process.">;
def Od : Flag<["-", "/"], "Od">, Flags<[CoreOption]>, Group<hlslcomp_Group>,
  HelpText<"Disable optimizations">;
def _SLASH_WX : Flag<["-", "/"], "WX">, Flags<[CoreOption]>, Group<hlslcomp_Group>,
  HelpText<"Treat warnings as errors">;
def VD : Flag<["-", "/"], "Vd">, Flags<[CoreOption]>, Group<hlslcomp_Group>,
  HelpText<"Disable validation">;
def _SLASH_Zi : Flag<["-", "/"], "Zi">, Flags<[CoreOption]>, Group<hlslcomp_Group>,
  HelpText<"Enable debug information. Cannot be used together with -Zs">;
def Zs : Flag<["-", "/"], "Zs">, Flags<[CoreOption]>, Group<hlslcomp_Group>,
  HelpText<"Generate small PDB with just sources and compile options. Cannot be used together with -Zi">;
def recompile : Flag<["-", "/"], "recompile">, Flags<[CoreOption]>, Group<hlslcomp_Group>,
  HelpText<"recompile from DXIL container with Debug Info or Debug Info bitcode file">;
def Zpr : Flag<["-", "/"], "Zpr">, Flags<[CoreOption]>, Group<hlslcomp_Group>,
  HelpText<"Pack matrices in row-major order">;
def Zpc : Flag<["-", "/"], "Zpc">, Flags<[CoreOption]>, Group<hlslcomp_Group>,
  HelpText<"Pack matrices in column-major order">;
def Zss : Flag<["-", "/"], "Zss">, Flags<[CoreOption]>, Group<hlslcomp_Group>,
  HelpText<"Compute Shader Hash considering source information">;
def Zsb : Flag<["-", "/"], "Zsb">, Flags<[CoreOption]>, Group<hlslcomp_Group>,
  HelpText<"Compute Shader Hash considering only output binary">;

// deprecated /Gpp def Gpp : Flag<["-", "/"], "Gpp">, HelpText<"Force partial precision">;
def Gfa : Flag<["-", "/"], "Gfa">, HelpText<"Avoid flow control constructs">, Flags<[CoreOption]>, Group<hlslcomp_Group>;
def Gfp : Flag<["-", "/"], "Gfp">, HelpText<"Prefer flow control constructs">, Flags<[CoreOption]>, Group<hlslcomp_Group>;
// /Gdp - disable effect performance mode - unsupported
def Gec : Flag<["-", "/"], "Gec">, HelpText<"Enable backward compatibility mode">, Flags<[CoreOption]>, Group<hlslcomp_Group>;
def Ges : Flag<["-", "/"], "Ges">, HelpText<"Enable strict mode">, Flags<[CoreOption]>, Group<hlslcomp_Group>;
def Gis : Flag<["-", "/"], "Gis">, HelpText<"Force IEEE strictness">, Flags<[CoreOption]>, Group<hlslcomp_Group>;

def denorm : JoinedOrSeparate<["-", "/"], "denorm">, HelpText<"select denormal value options (any, preserve, ftz). any is the default.">, Flags<[CoreOption]>, Group<hlslcomp_Group>;

def Fo : JoinedOrSeparate<["-", "/"], "Fo">, MetaVarName<"<file>">, HelpText<"Output object file">, Flags<[CoreOption, RewriteOption, DriverOption]>, Group<hlslcomp_Group>;
// def Fl : JoinedOrSeparate<["-", "/"], "Fl">, MetaVarName<"<file>">, HelpText<"Output a library">;
def Fc : JoinedOrSeparate<["-", "/"], "Fc">, MetaVarName<"<file>">, HelpText<"Output assembly code listing file">, Flags<[DriverOption]>, Group<hlslcomp_Group>;
//def Fx : JoinedOrSeparate<["-", "/"], "Fx">, MetaVarName<"<file>">, HelpText<"Output assembly code and hex listing file">;
def Fh : JoinedOrSeparate<["-", "/"], "Fh">, MetaVarName<"<file>">, HelpText<"Output header file containing object code">, Flags<[DriverOption]>, Group<hlslcomp_Group>;
def Fe : JoinedOrSeparate<["-", "/"], "Fe">, MetaVarName<"<file>">, HelpText<"Output warnings and errors to the given file">, Flags<[CoreOption, DriverOption]>, Group<hlslcomp_Group>;
def Fd : JoinedOrSeparate<["-", "/"], "Fd">, MetaVarName<"<file>">,
  HelpText<"Write debug information to the given file, or automatically named file in directory when ending in '\\'">,
  Flags<[CoreOption, DriverOption]>, Group<hlslcomp_Group>;
def Fre : Separate<["-", "/"], "Fre">, MetaVarName<"<file>">, HelpText<"Output reflection to the given file">, Flags<[CoreOption, DriverOption]>, Group<hlslcomp_Group>;
def Frs : Separate<["-", "/"], "Frs">, MetaVarName<"<file>">, HelpText<"Output root signature to the given file">, Flags<[CoreOption, DriverOption]>, Group<hlslcomp_Group>;
def Fsh : Separate<["-", "/"], "Fsh">, MetaVarName<"<file>">, HelpText<"Output shader hash to the given file">, Flags<[CoreOption, DriverOption]>, Group<hlslcomp_Group>;

def Vn : JoinedOrSeparate<["-", "/"], "Vn">, MetaVarName<"<name>">, HelpText<"Use <name> as variable name in header file">, Flags<[DriverOption]>, Group<hlslcomp_Group>;
def Cc : Flag<["-", "/"], "Cc">, HelpText<"Output color coded assembly listings">, Group<hlslcomp_Group>, Flags<[DriverOption]>;
def Ni : Flag<["-", "/"], "Ni">, HelpText<"Output instruction numbers in assembly listings">, Group<hlslcomp_Group>, Flags<[DriverOption]>;
def No : Flag<["-", "/"], "No">, HelpText<"Output instruction byte offsets in assembly listings">, Group<hlslcomp_Group>, Flags<[DriverOption]>;
def Lx : Flag<["-", "/"], "Lx">, HelpText<"Output hexadecimal literals">, Group<hlslcomp_Group>, Flags<[DriverOption]>;

// In place of 'E' for clang; fxc uses 'E' for entry point.
def P : Separate<["-", "/"], "P">, Flags<[CoreOption, DriverOption]>, Group<hlslutil_Group>,
  HelpText<"Preprocess to file (must be used alone)">;

// @<file> - options response file

def dumpbin : Flag<["-", "/"], "dumpbin">, Flags<[DriverOption]>, Group<hlslutil_Group>,
  HelpText<"Load a binary file rather than compiling">;
def link : Flag<["-", "/"], "link">, Flags<[DriverOption]>, Group<hlslutil_Group>,
  HelpText<"Link list of libraries provided in <inputs> argument separated by ';'">;
def Qstrip_reflect : Flag<["-", "/"], "Qstrip_reflect">, Flags<[CoreOption, DriverOption]>, Group<hlslutil_Group>,
  HelpText<"Strip reflection data from shader bytecode  (must be used with /Fo <file>)">;
def Qstrip_debug : Flag<["-", "/"], "Qstrip_debug">, Flags<[CoreOption, DriverOption]>, Group<hlslutil_Group>,
  HelpText<"Strip debug information from 4_0+ shader bytecode  (must be used with /Fo <file>)">;
def Qembed_debug : Flag<["-", "/"], "Qembed_debug">, Flags<[CoreOption]>, Group<hlslutil_Group>,
  HelpText<"Embed PDB in shader container (must be used with /Zi)">;
def Qstrip_priv : Flag<["-", "/"], "Qstrip_priv">, Flags<[CoreOption, DriverOption]>, Group<hlslutil_Group>,
  HelpText<"Strip private data from shader bytecode  (must be used with /Fo <file>)">;
def Qsource_in_debug_module : Flag<["-", "/"], "Qsource_in_debug_module">, Flags<[CoreOption, HelpHidden]>, Group<hlslutil_Group>,
  HelpText<"Generate old PDB format.">;
def Qpdb_in_private : Flag<["-", "/"], "Qpdb_in_private">, Flags<[CoreOption, HelpHidden]>, Group<hlslutil_Group>,
  HelpText<"Store PDB in private user data.">;

def Qstrip_rootsignature : Flag<["-", "/"], "Qstrip_rootsignature">, Flags<[CoreOption, DriverOption]>, Group<hlslutil_Group>, HelpText<"Strip root signature data from shader bytecode  (must be used with /Fo <file>)">;
def setrootsignature     : JoinedOrSeparate<["-", "/"], "setrootsignature">,     MetaVarName<"<file>">, Flags<[CoreOption, DriverOption]>, Group<hlslutil_Group>, HelpText<"Attach root signature to shader bytecode">;
def extractrootsignature : Flag<["-", "/"], "extractrootsignature">, Flags<[DriverOption]>, Group<hlslutil_Group>, HelpText<"Extract root signature from shader bytecode (must be used with /Fo <file>)">;
def verifyrootsignature  : JoinedOrSeparate<["-", "/"], "verifyrootsignature">,  MetaVarName<"<file>">, Flags<[DriverOption]>, Group<hlslutil_Group>, HelpText<"Verify shader bytecode with root signature">;
def force_rootsig_ver    : JoinedOrSeparate<["-", "/"], "force-rootsig-ver">,    Flags<[CoreOption]>, MetaVarName<"<profile>">, Group<hlslcomp_Group>, HelpText<"force root signature version (rootsig_1_1 if omitted)">;
def force_rootsig_ver_    : JoinedOrSeparate<["-", "/"], "force_rootsig_ver">,    Flags<[CoreOption, HelpHidden]>, MetaVarName<"<profile>">, Group<hlslcomp_Group>, HelpText<"force root signature version (rootsig_1_1 if omitted)">;

// Temporary TEST options, until reflection inside DXIL part can always be stripped
def Qkeep_reflect_in_dxil : Flag<["-", "/"], "Qkeep_reflect_in_dxil">,
  Flags<[CoreOption, HelpHidden]>, Group<hlslutil_Group>,
  HelpText<"Keep reflection data in shader bytecode">;
def Qstrip_reflect_from_dxil : Flag<["-", "/"], "Qstrip_reflect_from_dxil">,
  Flags<[CoreOption, HelpHidden]>, Group<hlslutil_Group>,
  HelpText<"Strip reflection data from shader bytecode  (must be used with /Fo <file>)">;

def enable_short_circuit : Flag<["-", "/"], "enable-short-circuit">, Flags<[CoreOption, HelpHidden]>, Group<hlslcomp_Group>,
  HelpText<"Operators '&&', '||', and '?:' will short circuit, only accepting scalar inputs.  Use and(), or(), and select() intrinsics in place of non-short-circuiting vector operators.">;

def enable_bitfields : Flag<["-", "/"], "enable-bitfields">, Flags<[CoreOption, HelpHidden]>, Group<hlslcomp_Group>,
  HelpText<"Enables bitfields on struct members">;

/*
def shtemplate : JoinedOrSeparate<["-", "/"], "shtemplate">, MetaVarName<"<file>">, Group<hlslcomp_Group>,
  HelpText<"Template shader file for merging/matching resources">;
def mergeUAVs : JoinedOrSeparate<["-", "/"], "mergeUAVs">, MetaVarName<"<file>">, Group<hlslcomp_Group>,
  HelpText<"Merge UAV slots of template shader and current shader">;
def matchUAVs : JoinedOrSeparate<["-", "/"], "matchUAVs">, MetaVarName<"<file>">, Group<hlslcomp_Group>,
  HelpText<"Match template shader UAV slots in current shader">;
def enable_unbounded_descriptor_tables : Flag<["-", "/"], "enable_unbounded_descriptor_tables">, Flags<[CoreOption]>, Group<hlslcomp_Group>,
  HelpText<"Enables unbounded descriptor tables">;
*/
def res_may_alias : Flag<["-", "/"], "res-may-alias">, Flags<[CoreOption]>, Group<hlslcomp_Group>,
  HelpText<"Assume that UAVs/SRVs may alias">;
def res_may_alias_ : Flag<["-", "/"], "res_may_alias">, Flags<[CoreOption, HelpHidden]>, Group<hlslcomp_Group>,
  HelpText<"Assume that UAVs/SRVs may alias">;
def all_resources_bound : Flag<["-", "/"], "all-resources-bound">, Flags<[CoreOption]>, Group<hlslcomp_Group>,
  HelpText<"Enables agressive flattening">;
def all_resources_bound_ : Flag<["-", "/"], "all_resources_bound">, Flags<[CoreOption, HelpHidden]>, Group<hlslcomp_Group>,
  HelpText<"Enables agressive flattening">;

def setprivate : JoinedOrSeparate<["-", "/"], "setprivate">, Flags<[CoreOption, DriverOption]>, MetaVarName<"<file>">, Group<hlslutil_Group>,
  HelpText<"Private data to add to compiled shader blob">;
def getprivate : JoinedOrSeparate<["-", "/"], "getprivate">, Flags<[DriverOption]>, MetaVarName<"<file>">, Group<hlslutil_Group>,
  HelpText<"Save private data from shader blob">;

def nologo : Flag<["-", "/"], "nologo">, Group<hlslcore_Group>, Flags<[DriverOption, HelpHidden]>,
  HelpText<"Suppress copyright message">;

//////////////////////////////////////////////////////////////////////////////
// Rewriter Options

def rw_unchanged : Flag<["-", "/"], "unchanged">, Group<hlslrewrite_Group>, Flags<[RewriteOption]>,
  HelpText<"Rewrite HLSL, without changes.">;
def rw_skip_function_body : Flag<["-", "/"], "skip-fn-body">, Group<hlslrewrite_Group>, Flags<[RewriteOption]>,
  HelpText<"Translate function definitions to declarations">;
def rw_skip_static : Flag<["-", "/"], "skip-static">, Group<hlslrewrite_Group>, Flags<[RewriteOption]>,
  HelpText<"Remove static functions and globals when used with -skip-fn-body">;
def rw_global_extern_by_default : Flag<["-", "/"], "global-extern-by-default">, Group<hlslrewrite_Group>, Flags<[RewriteOption]>,
  HelpText<"Set extern on non-static globals">;
def rw_keep_user_macro : Flag<["-", "/"], "keep-user-macro">, Group<hlslrewrite_Group>, Flags<[RewriteOption]>,
  HelpText<"Write out user defines after rewritten HLSL">;
def rw_extract_entry_uniforms : Flag<["-", "/"], "extract-entry-uniforms">, Group<hlslrewrite_Group>, Flags<[RewriteOption]>,
  HelpText<"Move uniform parameters from entry point to global scope">;
def rw_remove_unused_globals : Flag<["-", "/"], "remove-unused-globals">, Group<hlslrewrite_Group>, Flags<[RewriteOption]>,
  HelpText<"Remove unused static globals and functions">;
def rw_remove_unused_functions : Flag<["-", "/"], "remove-unused-functions">, Group<hlslrewrite_Group>, Flags<[RewriteOption]>,
  HelpText<"Remove unused functions and types">;
def rw_line_directive : Flag<["-", "/"], "line-directive">, Group<hlslrewrite_Group>, Flags<[RewriteOption]>,
  HelpText<"Add line directive">;
def rw_decl_global_cb : Flag<["-", "/"], "decl-global-cb">, Group<hlslrewrite_Group>, Flags<[RewriteOption]>,
  HelpText<"Collect all global constants outside cbuffer declarations into cbuffer GlobalCB { ... }. Still experimental, not all dependency scenarios handled.">;
// Also removed: compress, decompress, /Gch (child effect), /Gpp (partial precision)
// /Op - no support for preshaders.
