/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Option Parsing Definitions                                                 *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/////////
// Prefixes

#ifdef PREFIX
#define COMMA ,
PREFIX(prefix_0, {0})
PREFIX(prefix_3, {"-" COMMA 0})
PREFIX(prefix_4, {"-" COMMA "--" COMMA "/" COMMA 0})
PREFIX(prefix_1, {"-" COMMA "/" COMMA 0})
PREFIX(prefix_2, {"--" COMMA 0})
#undef COMMA
#endif

/////////
// Groups

#ifdef OPTION
OPTION(0, "<action group>", Action_Group, Group, INVALID, INVALID, 0, 0, 0, 0, 0)
OPTION(0, "<CompileOnly group>", CompileOnly_Group, Group, INVALID, INVALID, 0, 0, 0, 0, 0)
OPTION(0, "<I group>", I_Group, Group, CompileOnly_Group, INVALID, 0, 0, 0, 0, 0)
OPTION(0, "<M group>", M_Group, Group, CompileOnly_Group, INVALID, 0, 0, 0, 0, 0)
OPTION(0, "<O group>", O_Group, Group, CompileOnly_Group, INVALID, 0, 0, 0, 0, 0)
OPTION(0, "<R group>", R_Group, Group, CompileOnly_Group, INVALID, 0, 0, 0, 0, 0)
OPTION(0, "<R (with value) group>", R_value_Group, Group, R_Group, INVALID, 0, 0, 0, 0, 0)
OPTION(0, "<T group>", T_Group, Group, INVALID, INVALID, 0, 0, 0, 0, 0)
OPTION(0, "<W group>", W_Group, Group, CompileOnly_Group, INVALID, 0, 0, 0,
       "Warning Options", 0)
OPTION(0, "<W (with value) group>", W_value_Group, Group, W_Group, INVALID, 0, 0, 0,
       "Warning Options", 0)
OPTION(0, "<clang i group>", clang_i_Group, Group, i_Group, INVALID, 0, 0, 0, 0, 0)
OPTION(0, "<d group>", d_Group, Group, INVALID, INVALID, 0, 0, 0, 0, 0)
OPTION(0, "<f group>", f_Group, Group, CompileOnly_Group, INVALID, 0, 0, 0, 0, 0)
OPTION(0, "<f (clang-only) group>", f_clang_Group, Group, CompileOnly_Group, INVALID, 0, 0, 0, 0, 0)
OPTION(0, "<g group>", g_Group, Group, INVALID, INVALID, 0, 0, 0, 0, 0)
OPTION(0, "<g flags group>", g_flags_Group, Group, INVALID, INVALID, 0, 0, 0, 0, 0)
OPTION(0, "HLSL Compilation", hlslcomp_Group, Group, INVALID, INVALID, 0, 0, 0,
       "Compilation Options", 0)
OPTION(0, "HLSL Core", hlslcore_Group, Group, INVALID, INVALID, 0, 0, 0,
       "Common Options", 0)
OPTION(0, "HLSL Optimization", hlsloptz_Group, Group, INVALID, INVALID, 0, 0, 0,
       "Optimization Options", 0)
OPTION(0, "HLSL Rewriter", hlslrewrite_Group, Group, INVALID, INVALID, 0, 0, 0,
       "Rewriter Options", 0)
OPTION(0, "HLSL Utility", hlslutil_Group, Group, INVALID, INVALID, 0, 0, 0,
       "Utility Options", 0)
OPTION(0, "<i group>", i_Group, Group, CompileOnly_Group, INVALID, 0, 0, 0, 0, 0)
OPTION(0, "<clang internal options>", internal_Group, Group, INVALID, INVALID, 0, 0, 0, 0, 0)
OPTION(0, "<clang debug/development internal options>", internal_debug_Group, Group, internal_Group, INVALID, 0, 0, 0,
       "DEBUG/DEVELOPMENT OPTIONS", 0)
OPTION(0, "<clang driver internal options>", internal_driver_Group, Group, internal_Group, INVALID, 0, 0, 0,
       "DRIVER OPTIONS", 0)
OPTION(0, "<m group>", m_Group, Group, CompileOnly_Group, INVALID, 0, 0, 0, 0, 0)
OPTION(0, "SPIR-V CodeGen", spirv_Group, Group, INVALID, INVALID, 0, 0, 0,
       "SPIR-V CodeGen Options", 0)
OPTION(0, "<u group>", u_Group, Group, INVALID, INVALID, 0, 0, 0, 0, 0)

//////////
// Options

OPTION(prefix_0, "<input>", INPUT, Input, INVALID, INVALID, 0, 0, 0, 0, 0)
OPTION(prefix_0, "<unknown>", UNKNOWN, Unknown, INVALID, INVALID, 0, 0, 0, 0, 0)
OPTION(prefix_1, "?", _help_question, Flag, INVALID, help, 0, DriverOption, 0, 0, 0)
OPTION(prefix_1, "all-resources-bound", all_resources_bound, Flag, hlslcomp_Group, INVALID, 0, CoreOption, 0,
       "Enables agressive flattening", 0)
OPTION(prefix_2, "all-warnings", _all_warnings, Flag, INVALID, Wall, 0, CoreOption, 0, 0, 0)
OPTION(prefix_1, "all_resources_bound", all_resources_bound_, Flag, hlslcomp_Group, INVALID, 0, CoreOption | HelpHidden, 0,
       "Enables agressive flattening", 0)
OPTION(prefix_1, "ast-dump", ast_dump, Flag, INVALID, INVALID, 0, CoreOption | DriverOption | HelpHidden, 0,
       "Dumps the parsed Abstract Syntax Tree.", 0)
OPTION(prefix_1, "auto-binding-space", auto_binding_space, Separate, hlslcomp_Group, INVALID, 0, CoreOption, 0,
       "Set auto binding space - enables auto resource binding in libraries", 0)
OPTION(prefix_1, "binding-table-define", binding_table_define, Separate, hlsloptz_Group, INVALID, 0, CoreOption | DriverOption | HelpHidden, 0,
       "Import a binding table from a define to specify resource bindings.", 0)
OPTION(prefix_1, "Cc", Cc, Flag, hlslcomp_Group, INVALID, 0, DriverOption, 0,
       "Output color coded assembly listings", 0)
OPTION(prefix_1, "decl-global-cb", rw_decl_global_cb, Flag, hlslrewrite_Group, INVALID, 0, RewriteOption, 0,
       "Collect all global constants outside cbuffer declarations into cbuffer GlobalCB { ... }. Still experimental, not all dependency scenarios handled.", 0)
OPTION(prefix_1, "default-linkage", default_linkage, Separate, hlslcomp_Group, INVALID, 0, CoreOption, 0,
       "Set default linkage for non-shader functions when compiling or linking to a library target (internal, external)", 0)
OPTION(prefix_1, "denorm", denorm, JoinedOrSeparate, hlslcomp_Group, INVALID, 0, CoreOption, 0,
       "select denormal value options (any, preserve, ftz). any is the default.", 0)
OPTION(prefix_1, "disable-exception-handling", disable_exception_handling, Flag, hlslcomp_Group, INVALID, 0, DriverOption | HelpHidden, 0,
       "Disable dxc handling of exceptions", 0)
OPTION(prefix_1, "disable-lifetime-markers", disable_lifetime_markers, Flag, hlslcomp_Group, INVALID, 0, CoreOption | HelpHidden, 0,
       "Disable generation of lifetime markers where they would be otherwise (6.6+)", 0)
OPTION(prefix_1, "disable-payload-qualifiers", disable_payload_qualifiers, Flag, hlslcomp_Group, INVALID, 0, CoreOption | RewriteOption | DriverOption, 0,
       "Disables support for payload access qualifiers for raytracing payloads in SM 6.7.", 0)
OPTION(prefix_1, "dumpbin", dumpbin, Flag, hlslutil_Group, INVALID, 0, DriverOption, 0,
       "Load a binary file rather than compiling", 0)
OPTION(prefix_1, "D", D, JoinedOrSeparate, hlslcomp_Group, INVALID, 0, CoreOption | RewriteOption, 0,
       "Define macro", 0)
OPTION(prefix_3, "d", d_Flag, Flag, d_Group, INVALID, 0, 0, 0, 0, 0)
OPTION(prefix_3, "d", d_Joined, Joined, d_Group, INVALID, 0, 0, 0, 0, 0)
OPTION(prefix_1, "enable-16bit-types", enable_16bit_types, Flag, hlslcomp_Group, INVALID, 0, CoreOption | RewriteOption | DriverOption, 0,
       "Enable 16bit types and disable min precision types. Available in HLSL 2018 and shader model 6.2", 0)
OPTION(prefix_1, "enable-bitfields", enable_bitfields, Flag, hlslcomp_Group, INVALID, 0, CoreOption | HelpHidden, 0,
       "Enables bitfields on struct members", 0)
OPTION(prefix_1, "enable-lifetime-markers", enable_lifetime_markers, Flag, hlslcomp_Group, INVALID, 0, CoreOption, 0,
       "Enable generation of lifetime markers", 0)
OPTION(prefix_1, "enable-operator-overloading", enable_operator_overloading, Flag, hlslcomp_Group, INVALID, 0, CoreOption | HelpHidden, 0,
       "Enable operator overloading support for HLSL.", 0)
OPTION(prefix_1, "enable-payload-qualifiers", enable_payload_qualifiers, Flag, hlslcomp_Group, INVALID, 0, CoreOption | RewriteOption | DriverOption, 0,
       "Enables support for payload access qualifiers for raytracing payloads in SM 6.6.", 0)
OPTION(prefix_1, "enable-short-circuit", enable_short_circuit, Flag, hlslcomp_Group, INVALID, 0, CoreOption | HelpHidden, 0,
       "Operators '&&', '||', and '?:' will short circuit, only accepting scalar inputs.  Use and(), or(), and select() intrinsics in place of non-short-circuiting vector operators.", 0)
OPTION(prefix_1, "enable-templates", enable_templates, Flag, hlslcomp_Group, INVALID, 0, CoreOption | HelpHidden, 0,
       "Enable template support for HLSL.", 0)
OPTION(prefix_1, "encoding", encoding, Separate, hlslcomp_Group, INVALID, 0, CoreOption | RewriteOption | DriverOption, 0,
       "Set default encoding for source inputs and text outputs (utf8|utf16(win)|utf32(*nix)|wide) default=utf8", 0)
OPTION(prefix_1, "export-shaders-only", export_shaders_only, Flag, hlslcomp_Group, INVALID, 0, CoreOption, 0,
       "Only export shaders when compiling a library", 0)
OPTION(prefix_1, "exports", exports, Separate, hlslcomp_Group, INVALID, 0, CoreOption, 0,
       "Specify exports when compiling a library: export1[[,export1_clone,...]=internal_name][;...]", 0)
OPTION(prefix_1, "external-fn", external_fn, Separate, hlslcore_Group, INVALID, 0, DriverOption | RewriteOption | HelpHidden, 0,
       "External function name to load for compiler support", 0)
OPTION(prefix_1, "external", external_lib, Separate, hlslcore_Group, INVALID, 0, DriverOption | RewriteOption | HelpHidden, 0,
       "External DLL name to load for compiler support", 0)
OPTION(prefix_1, "extract-entry-uniforms", rw_extract_entry_uniforms, Flag, hlslrewrite_Group, INVALID, 0, RewriteOption, 0,
       "Move uniform parameters from entry point to global scope", 0)
OPTION(prefix_1, "extractrootsignature", extractrootsignature, Flag, hlslutil_Group, INVALID, 0, DriverOption, 0,
       "Extract root signature from shader bytecode (must be used with /Fo <file>)", 0)
OPTION(prefix_1, "E", entrypoint, JoinedOrSeparate, hlslcomp_Group, INVALID, 0, CoreOption | RewriteOption, 0,
       "Entry point name", 0)
OPTION(prefix_3, "fassociative-math", fassociative_math, Flag, hlsloptz_Group, INVALID, 0, 0, 0, 0, 0)
OPTION(prefix_1, "fcgl", fcgl, Flag, hlslcore_Group, INVALID, 0, CoreOption | HelpHidden, 0,
       "Generate high-level code only", 0)
OPTION(prefix_3, "fconstexpr-backtrace-limit=", fconstexpr_backtrace_limit_EQ, Joined, f_Group, INVALID, 0, 0, 0, 0, 0)
OPTION(prefix_3, "fconstexpr-depth=", fconstexpr_depth_EQ, Joined, f_Group, INVALID, 0, 0, 0, 0, 0)
OPTION(prefix_3, "fconstexpr-steps=", fconstexpr_steps_EQ, Joined, f_Group, INVALID, 0, 0, 0, 0, 0)
OPTION(prefix_1, "Fc", Fc, JoinedOrSeparate, hlslcomp_Group, INVALID, 0, DriverOption, 0,
       "Output assembly code listing file", "<file>")
OPTION(prefix_3, "fdiagnostics-color=", fdiagnostics_color_EQ, Joined, hlslcomp_Group, INVALID, 0, 0, 0, 0, 0)
OPTION(prefix_3, "fdiagnostics-color", fdiagnostics_color, Flag, hlslcomp_Group, INVALID, 0, CoreOption | DriverOption, 0, 0, 0)
OPTION(prefix_3, "fdiagnostics-show-category=", fdiagnostics_show_category_EQ, Joined, hlslcomp_Group, INVALID, 0, 0, 0, 0, 0)
OPTION(prefix_3, "fdiagnostics-show-option", fdiagnostics_show_option, Flag, hlslcomp_Group, INVALID, 0, CoreOption, 0,
       "Print option name with mappable diagnostics", 0)
OPTION(prefix_1, "Fd", Fd, JoinedOrSeparate, hlslcomp_Group, INVALID, 0, CoreOption | DriverOption, 0,
       "Write debug information to the given file, or automatically named file in directory when ending in '\\'", "<file>")
OPTION(prefix_3, "ferror-limit=", ferror_limit_EQ, Joined, hlslcomp_Group, INVALID, 0, CoreOption, 0, 0, 0)
OPTION(prefix_1, "Fe", Fe, JoinedOrSeparate, hlslcomp_Group, INVALID, 0, CoreOption | DriverOption, 0,
       "Output warnings and errors to the given file", "<file>")
OPTION(prefix_3, "ffinite-math-only", ffinite_math_only, Flag, hlsloptz_Group, INVALID, 0, CoreOption, 0, 0, 0)
OPTION(prefix_3, "fhonor-infinities", fhonor_infinities, Flag, hlsloptz_Group, INVALID, 0, 0, 0, 0, 0)
OPTION(prefix_3, "fhonor-nans", fhonor_nans, Flag, hlsloptz_Group, INVALID, 0, 0, 0, 0, 0)
OPTION(prefix_1, "Fh", Fh, JoinedOrSeparate, hlslcomp_Group, INVALID, 0, DriverOption, 0,
       "Output header file containing object code", "<file>")
OPTION(prefix_1, "flegacy-macro-expansion", flegacy_macro_expansion, Flag, hlslcomp_Group, INVALID, 0, CoreOption | RewriteOption | DriverOption, 0,
       "Expand the operands before performing token-pasting operation (fxc behavior)", 0)
OPTION(prefix_1, "flegacy-resource-reservation", flegacy_resource_reservation, Flag, hlslcomp_Group, INVALID, 0, CoreOption | DriverOption, 0,
       "Reserve unused explicit register assignments for compatibility with shader model 5.0 and below", 0)
OPTION(prefix_3, "flimited-precision=", flimited_precision_EQ, Joined, hlsloptz_Group, INVALID, 0, 0, 0, 0, 0)
OPTION(prefix_3, "fno-associative-math", fno_associative_math, Flag, hlsloptz_Group, INVALID, 0, 0, 0, 0, 0)
OPTION(prefix_3, "fno-diagnostics-show-option", fno_diagnostics_show_option, Flag, hlslcomp_Group, INVALID, 0, CoreOption, 0,
       "Do not print option name with mappable diagnostics", 0)
OPTION(prefix_3, "fno-finite-math-only", fno_finite_math_only, Flag, hlsloptz_Group, INVALID, 0, 0, 0, 0, 0)
OPTION(prefix_3, "fno-honor-infinities", fno_honor_infinities, Flag, hlsloptz_Group, INVALID, 0, 0, 0, 0, 0)
OPTION(prefix_3, "fno-honor-nans", fno_honor_nans, Flag, hlsloptz_Group, INVALID, 0, 0, 0, 0, 0)
OPTION(prefix_3, "fno-reciprocal-math", fno_reciprocal_math, Flag, hlsloptz_Group, INVALID, 0, 0, 0, 0, 0)
OPTION(prefix_3, "fno-unsafe-math-optimizations", fno_unsafe_math_optimizations, Flag, hlsloptz_Group, INVALID, 0, 0, 0, 0, 0)
OPTION(prefix_1, "force-rootsig-ver", force_rootsig_ver, JoinedOrSeparate, hlslcomp_Group, INVALID, 0, CoreOption, 0,
       "force root signature version (rootsig_1_1 if omitted)", "<profile>")
OPTION(prefix_1, "force-zero-store-lifetimes", force_zero_store_lifetimes, Flag, hlslcomp_Group, INVALID, 0, CoreOption | HelpHidden, 0,
       "Instead of generating lifetime intrinsics (SM >= 6.6) or storing undef (SM < 6.6), force fall back to storing zeroinitializer.", 0)
OPTION(prefix_1, "force_rootsig_ver", force_rootsig_ver_, JoinedOrSeparate, hlslcomp_Group, INVALID, 0, CoreOption | HelpHidden, 0,
       "force root signature version (rootsig_1_1 if omitted)", "<profile>")
OPTION(prefix_1, "Fo", Fo, JoinedOrSeparate, hlslcomp_Group, INVALID, 0, CoreOption | RewriteOption | DriverOption, 0,
       "Output object file", "<file>")
OPTION(prefix_1, "Fre", Fre, Separate, hlslcomp_Group, INVALID, 0, CoreOption | DriverOption, 0,
       "Output reflection to the given file", "<file>")
OPTION(prefix_1, "Frs", Frs, Separate, hlslcomp_Group, INVALID, 0, CoreOption | DriverOption, 0,
       "Output root signature to the given file", "<file>")
OPTION(prefix_1, "Fsh", Fsh, Separate, hlslcomp_Group, INVALID, 0, CoreOption | DriverOption, 0,
       "Output shader hash to the given file", "<file>")
OPTION(prefix_3, "fsigned-zeros", fsigned_zeros, Flag, hlsloptz_Group, INVALID, 0, 0, 0, 0, 0)
OPTION(prefix_3, "fspv-debug=", fspv_debug_EQ, Joined, spirv_Group, INVALID, 0, CoreOption | DriverOption, 0,
       "Specify whitelist of debug info category (file -> source -> line, tool)", 0)
OPTION(prefix_3, "fspv-entrypoint-name=", fspv_entrypoint_name_EQ, Joined, spirv_Group, INVALID, 0, CoreOption | DriverOption, 0,
       "Specify the SPIR-V entry point name. Defaults to the HLSL entry point name.", 0)
OPTION(prefix_3, "fspv-extension=", fspv_extension_EQ, Joined, spirv_Group, INVALID, 0, CoreOption | DriverOption, 0,
       "Specify SPIR-V extension permitted to use", 0)
OPTION(prefix_3, "fspv-fix-func-call-arguments", fspv_fix_func_call_arguments, Flag, spirv_Group, INVALID, 0, CoreOption | DriverOption | HelpHidden, 0,
       "Fix function call arguments which are not memory objects", 0)
OPTION(prefix_3, "fspv-flatten-resource-arrays", fspv_flatten_resource_arrays, Flag, spirv_Group, INVALID, 0, CoreOption | DriverOption, 0,
       "Flatten arrays of resources so each array element takes one binding number", 0)
OPTION(prefix_3, "fspv-print-all", fspv_print_all, Flag, spirv_Group, INVALID, 0, CoreOption | DriverOption, 0,
       "Print the SPIR-V module before each pass and after the last one. Useful for debugging SPIR-V legalization and optimization passes.", 0)
OPTION(prefix_3, "fspv-reduce-load-size", fspv_reduce_load_size, Flag, spirv_Group, INVALID, 0, CoreOption | DriverOption, 0,
       "Replaces loads of composite objects to reduce memory pressure for the loads", 0)
OPTION(prefix_3, "fspv-reflect", fspv_reflect, Flag, spirv_Group, INVALID, 0, CoreOption | DriverOption, 0,
       "Emit additional SPIR-V instructions to aid reflection", 0)
OPTION(prefix_3, "fspv-target-env=", fspv_target_env_EQ, Joined, spirv_Group, INVALID, 0, CoreOption | DriverOption, 0,
       "Specify the target environment: vulkan1.0 (default), vulkan1.1, vulkan1.1spirv1.4, vulkan1.2, vulkan1.3, or universal1.5", 0)
OPTION(prefix_3, "fspv-use-legacy-buffer-matrix-order", fspv_use_legacy_buffer_matrix_order, Flag, spirv_Group, INVALID, 0, CoreOption | DriverOption, 0,
       "Assume the legacy matrix order (row major) when accessing raw buffers (e.g., ByteAdddressBuffer)", 0)
OPTION(prefix_3, "funsafe-math-optimizations", funsafe_math_optimizations, Flag, hlsloptz_Group, INVALID, 0, 0, 0, 0, 0)
OPTION(prefix_3, "fvk-auto-shift-bindings", fvk_auto_shift_bindings, Flag, spirv_Group, INVALID, 0, CoreOption | DriverOption, 0,
       "Apply fvk-*-shift to resources without an explicit register assignment.", 0)
OPTION(prefix_3, "fvk-b-shift", fvk_b_shift, MultiArg, spirv_Group, INVALID, 0, CoreOption | DriverOption, 2,
       "Specify Vulkan binding number shift for b-type register", "<shift> <space>")
OPTION(prefix_3, "fvk-bind-globals", fvk_bind_globals, MultiArg, spirv_Group, INVALID, 0, CoreOption | DriverOption, 2,
       "Specify Vulkan binding number and set number for the $Globals cbuffer", "<binding> <set>")
OPTION(prefix_3, "fvk-bind-register", fvk_bind_register, MultiArg, spirv_Group, INVALID, 0, CoreOption | DriverOption, 4,
       "Specify Vulkan descriptor set and binding for a specific register", "<type-number> <space> <binding> <set>")
OPTION(prefix_3, "fvk-invert-y", fvk_invert_y, Flag, spirv_Group, INVALID, 0, CoreOption | DriverOption, 0,
       "Negate SV_Position.y before writing to stage output in VS/DS/GS to accommodate Vulkan's coordinate system", 0)
OPTION(prefix_3, "fvk-s-shift", fvk_s_shift, MultiArg, spirv_Group, INVALID, 0, CoreOption | DriverOption, 2,
       "Specify Vulkan binding number shift for s-type register", "<shift> <space>")
OPTION(prefix_3, "fvk-stage-io-order=", fvk_stage_io_order_EQ, Joined, spirv_Group, INVALID, 0, CoreOption | DriverOption | HelpHidden, 0,
       "Specify Vulkan stage I/O location assignment order", 0)
OPTION(prefix_3, "fvk-support-nonzero-base-instance", fvk_support_nonzero_base_instance, Flag, spirv_Group, INVALID, 0, CoreOption | DriverOption, 0,
       "Follow Vulkan spec to use gl_BaseInstance as the first vertex instance, which makes SV_InstanceID = gl_InstanceIndex - gl_BaseInstance (without this option, SV_InstanceID = gl_InstanceIndex)", 0)
OPTION(prefix_3, "fvk-t-shift", fvk_t_shift, MultiArg, spirv_Group, INVALID, 0, CoreOption | DriverOption, 2,
       "Specify Vulkan binding number shift for t-type register", "<shift> <space>")
OPTION(prefix_3, "fvk-u-shift", fvk_u_shift, MultiArg, spirv_Group, INVALID, 0, CoreOption | DriverOption, 2,
       "Specify Vulkan binding number shift for u-type register", "<shift> <space>")
OPTION(prefix_3, "fvk-use-dx-layout", fvk_use_dx_layout, Flag, spirv_Group, INVALID, 0, CoreOption | DriverOption, 0,
       "Use DirectX memory layout for Vulkan resources", 0)
OPTION(prefix_3, "fvk-use-dx-position-w", fvk_use_dx_position_w, Flag, spirv_Group, INVALID, 0, CoreOption | DriverOption, 0,
       "Reciprocate SV_Position.w after reading from stage input in PS to accommodate the difference between Vulkan and DirectX", 0)
OPTION(prefix_3, "fvk-use-gl-layout", fvk_use_gl_layout, Flag, spirv_Group, INVALID, 0, CoreOption | DriverOption, 0,
       "Use strict OpenGL std140/std430 memory layout for Vulkan resources", 0)
OPTION(prefix_3, "fvk-use-scalar-layout", fvk_use_scalar_layout, Flag, spirv_Group, INVALID, 0, CoreOption | DriverOption, 0,
       "Use scalar memory layout for Vulkan resources", 0)
OPTION(prefix_1, "Gec", Gec, Flag, hlslcomp_Group, INVALID, 0, CoreOption, 0,
       "Enable backward compatibility mode", 0)
OPTION(prefix_1, "Ges", Ges, Flag, hlslcomp_Group, INVALID, 0, CoreOption, 0,
       "Enable strict mode", 0)
OPTION(prefix_1, "getprivate", getprivate, JoinedOrSeparate, hlslutil_Group, INVALID, 0, DriverOption, 0,
       "Save private data from shader blob", "<file>")
OPTION(prefix_1, "Gfa", Gfa, Flag, hlslcomp_Group, INVALID, 0, CoreOption, 0,
       "Avoid flow control constructs", 0)
OPTION(prefix_1, "Gfp", Gfp, Flag, hlslcomp_Group, INVALID, 0, CoreOption, 0,
       "Prefer flow control constructs", 0)
OPTION(prefix_1, "Gis", Gis, Flag, hlslcomp_Group, INVALID, 0, CoreOption, 0,
       "Force IEEE strictness", 0)
OPTION(prefix_1, "global-extern-by-default", rw_global_extern_by_default, Flag, hlslrewrite_Group, INVALID, 0, RewriteOption, 0,
       "Set extern on non-static globals", 0)
OPTION(prefix_2, "help-hidden", _help_hidden, Flag, INVALID, INVALID, 0, DriverOption, 0, 0, 0)
OPTION(prefix_4, "help", help, Flag, hlslcore_Group, INVALID, 0, DriverOption, 0,
       "Display available options", 0)
OPTION(prefix_1, "HV", hlsl_version, Separate, hlslcomp_Group, INVALID, 0, CoreOption | RewriteOption, 0,
       "HLSL version (2016, 2017, 2018, 2021). Default is 2018", 0)
OPTION(prefix_3, "H", H, Flag, hlslcomp_Group, INVALID, 0, CoreOption, 0,
       "Show header includes and nesting depth", 0)
OPTION(prefix_1, "ignore-line-directives", ignore_line_directives, Flag, hlslcomp_Group, INVALID, 0, CoreOption, 0,
       "Ignore line directives", 0)
OPTION(prefix_1, "ignore-opt-semdefs", ignore_opt_semdefs, Flag, hlslcomp_Group, INVALID, 0, CoreOption | HelpHidden, 0,
       "Ignore optional semantic defines which are not required to ensure program correctness.", 0)
OPTION(prefix_1, "ignore-semdef", ignore_semdef, Separate, hlslcomp_Group, INVALID, 0, CoreOption | HelpHidden, 0,
       "Ignore the semantic define passed as a value to this flag.", 0)
OPTION(prefix_1, "import-binding-table", import_binding_table, JoinedOrSeparate, hlsloptz_Group, INVALID, 0, CoreOption | DriverOption | HelpHidden, 0,
       "Import a binding table to specify resource bindings.", 0)
OPTION(prefix_1, "I", I, JoinedOrSeparate, hlslcomp_Group, INVALID, 0, CoreOption | RewriteOption, 0,
       "Add directory to include search path", 0)
OPTION(prefix_1, "keep-user-macro", rw_keep_user_macro, Flag, hlslrewrite_Group, INVALID, 0, RewriteOption, 0,
       "Write out user defines after rewritten HLSL", 0)
OPTION(prefix_1, "line-directive", rw_line_directive, Flag, hlslrewrite_Group, INVALID, 0, RewriteOption, 0,
       "Add line directive", 0)
OPTION(prefix_1, "link", link, Flag, hlslutil_Group, INVALID, 0, DriverOption, 0,
       "Link list of libraries provided in <inputs> argument separated by ';'", 0)
OPTION(prefix_1, "Lx", Lx, Flag, hlslcomp_Group, INVALID, 0, DriverOption, 0,
       "Output hexadecimal literals", 0)
OPTION(prefix_1, "MD", write_dependencies, Flag, INVALID, INVALID, 0, CoreOption | DriverOption, 0,
       "Write a file with .d extension that will contain the list of the compilation target dependencies.", 0)
OPTION(prefix_1, "memdep-block-scan-limit", memdep_block_scan_limit, Separate, hlsloptz_Group, INVALID, 0, CoreOption | DriverOption | HelpHidden, 0,
       "The number of instructions to scan in a block in memory dependency analysis.", 0)
OPTION(prefix_1, "MF", write_dependencies_to, JoinedOrSeparate, INVALID, INVALID, 0, CoreOption | DriverOption, 0,
       "Write the specfied file that will contain the list of the compilation target dependencies.", "<file>")
OPTION(prefix_1, "M", dump_dependencies, Flag, INVALID, INVALID, 0, CoreOption | DriverOption, 0,
       "Dumps the list of the compilation target dependencies.", 0)
OPTION(prefix_1, "Ni", Ni, Flag, hlslcomp_Group, INVALID, 0, DriverOption, 0,
       "Output instruction numbers in assembly listings", 0)
OPTION(prefix_1, "no-legacy-cbuf-layout", no_legacy_cbuf_layout, Flag, hlslcomp_Group, INVALID, 0, CoreOption, 0,
       "Do not use legacy cbuffer load", 0)
OPTION(prefix_1, "no-warnings", no_warnings, Flag, hlslcomp_Group, INVALID, 0, CoreOption | RewriteOption, 0,
       "Suppress warnings", 0)
OPTION(prefix_1, "nologo", nologo, Flag, hlslcore_Group, INVALID, 0, DriverOption | HelpHidden, 0,
       "Suppress copyright message", 0)
OPTION(prefix_1, "not_use_legacy_cbuf_load", not_use_legacy_cbuf_load_, Flag, hlslcomp_Group, INVALID, 0, CoreOption | HelpHidden, 0,
       "Do not use legacy cbuffer load", 0)
OPTION(prefix_1, "No", No, Flag, hlslcomp_Group, INVALID, 0, DriverOption, 0,
       "Output instruction byte offsets in assembly listings", 0)
OPTION(prefix_1, "O0", O0, Flag, hlsloptz_Group, INVALID, 0, CoreOption, 0,
       "Optimization Level 0", 0)
OPTION(prefix_1, "O1", O1, Flag, hlsloptz_Group, INVALID, 0, CoreOption, 0,
       "Optimization Level 1", 0)
OPTION(prefix_1, "O2", O2, Flag, hlsloptz_Group, INVALID, 0, CoreOption, 0,
       "Optimization Level 2", 0)
OPTION(prefix_1, "O3", O3, Flag, hlsloptz_Group, INVALID, 0, CoreOption, 0,
       "Optimization Level 3 (Default)", 0)
OPTION(prefix_3, "Oconfig=", Oconfig, CommaJoined, spirv_Group, INVALID, 0, CoreOption, 0,
       "Specify a comma-separated list of SPIRV-Tools passes to customize optimization configuration (see http://khr.io/hlsl2spirv#optimization)", 0)
OPTION(prefix_1, "Odump", Odump, Flag, hlslcomp_Group, INVALID, 0, CoreOption, 0,
       "Print the optimizer commands.", 0)
OPTION(prefix_1, "Od", Od, Flag, hlslcomp_Group, INVALID, 0, CoreOption, 0,
       "Disable optimizations", 0)
OPTION(prefix_1, "opt-disable", opt_disable, Separate, hlsloptz_Group, INVALID, 0, CoreOption | DriverOption | HelpHidden, 0,
       "Disable this optimization.", 0)
OPTION(prefix_1, "opt-enable", opt_enable, Separate, hlsloptz_Group, INVALID, 0, CoreOption | DriverOption | HelpHidden, 0,
       "Enable this optimization.", 0)
OPTION(prefix_1, "opt-select", opt_select, MultiArg, hlsloptz_Group, INVALID, 0, CoreOption | DriverOption | HelpHidden, 2,
       "Select this optimization variant.", "<opt> <variant>")
OPTION(prefix_1, "override-semdef", override_semdef, Separate, hlslcomp_Group, INVALID, 0, CoreOption | HelpHidden, 0,
       "Override the value of semantic define with the one passed with this flag.", 0)
OPTION(prefix_1, "pack-optimized", pack_optimized, Flag, hlslcomp_Group, INVALID, 0, CoreOption, 0,
       "Optimize signature packing assuming identical signature provided for each connecting stage", 0)
OPTION(prefix_1, "pack-prefix-stable", pack_prefix_stable, Flag, hlslcomp_Group, INVALID, 0, CoreOption, 0,
       "(default) Pack signatures preserving prefix-stable property - appended elements will not disturb placement of prior elements", 0)
OPTION(prefix_1, "pack_optimized", pack_optimized_, Flag, hlslcomp_Group, INVALID, 0, CoreOption | HelpHidden, 0,
       "Optimize signature packing assuming identical signature provided for each connecting stage", 0)
OPTION(prefix_1, "pack_prefix_stable", pack_prefix_stable_, Flag, hlslcomp_Group, INVALID, 0, CoreOption | HelpHidden, 0,
       "(default) Pack signatures preserving prefix-stable property - appended elements will not disturb placement of prior elements", 0)
OPTION(prefix_1, "precise-output", precise_output, Separate, hlslcomp_Group, INVALID, 0, CoreOption | HelpHidden, 0,
       "Mark output semantic as precise", 0)
OPTION(prefix_1, "preserve-intermediate-values", preserve_intermediate_values, Flag, hlslcore_Group, INVALID, 0, CoreOption | HelpHidden, 0,
       "Preserve intermediate values to help shader debugging", 0)
OPTION(prefix_1, "print-after-all", print_after_all, Flag, hlslcomp_Group, INVALID, 0, CoreOption | HelpHidden, 0,
       "Print LLVM IR after each pass.", 0)
OPTION(prefix_1, "print-after", print_after, Separate, hlslcomp_Group, INVALID, 0, CoreOption | HelpHidden, 0,
       "Print LLVM IR after a specific pass.", 0)
OPTION(prefix_1, "P", P, Separate, hlslutil_Group, INVALID, 0, CoreOption | DriverOption, 0,
       "Preprocess to file (must be used alone)", 0)
OPTION(prefix_1, "Qembed_debug", Qembed_debug, Flag, hlslutil_Group, INVALID, 0, CoreOption, 0,
       "Embed PDB in shader container (must be used with /Zi)", 0)
OPTION(prefix_1, "Qkeep_reflect_in_dxil", Qkeep_reflect_in_dxil, Flag, hlslutil_Group, INVALID, 0, CoreOption | HelpHidden, 0,
       "Keep reflection data in shader bytecode", 0)
OPTION(prefix_1, "Qpdb_in_private", Qpdb_in_private, Flag, hlslutil_Group, INVALID, 0, CoreOption | HelpHidden, 0,
       "Store PDB in private user data.", 0)
OPTION(prefix_1, "Qsource_in_debug_module", Qsource_in_debug_module, Flag, hlslutil_Group, INVALID, 0, CoreOption | HelpHidden, 0,
       "Generate old PDB format.", 0)
OPTION(prefix_1, "Qstrip_debug", Qstrip_debug, Flag, hlslutil_Group, INVALID, 0, CoreOption | DriverOption, 0,
       "Strip debug information from 4_0+ shader bytecode  (must be used with /Fo <file>)", 0)
OPTION(prefix_1, "Qstrip_priv", Qstrip_priv, Flag, hlslutil_Group, INVALID, 0, CoreOption | DriverOption, 0,
       "Strip private data from shader bytecode  (must be used with /Fo <file>)", 0)
OPTION(prefix_1, "Qstrip_reflect_from_dxil", Qstrip_reflect_from_dxil, Flag, hlslutil_Group, INVALID, 0, CoreOption | HelpHidden, 0,
       "Strip reflection data from shader bytecode  (must be used with /Fo <file>)", 0)
OPTION(prefix_1, "Qstrip_reflect", Qstrip_reflect, Flag, hlslutil_Group, INVALID, 0, CoreOption | DriverOption, 0,
       "Strip reflection data from shader bytecode  (must be used with /Fo <file>)", 0)
OPTION(prefix_1, "Qstrip_rootsignature", Qstrip_rootsignature, Flag, hlslutil_Group, INVALID, 0, CoreOption | DriverOption, 0,
       "Strip root signature data from shader bytecode  (must be used with /Fo <file>)", 0)
OPTION(prefix_3, "Qunused-arguments", Qunused_arguments, Flag, hlslcore_Group, INVALID, 0, CoreOption, 0,
       "Don't emit warning for unused driver arguments", 0)
OPTION(prefix_1, "recompile", recompile, Flag, hlslcomp_Group, INVALID, 0, CoreOption, 0,
       "recompile from DXIL container with Debug Info or Debug Info bitcode file", 0)
OPTION(prefix_1, "remove-unused-functions", rw_remove_unused_functions, Flag, hlslrewrite_Group, INVALID, 0, RewriteOption, 0,
       "Remove unused functions and types", 0)
OPTION(prefix_1, "remove-unused-globals", rw_remove_unused_globals, Flag, hlslrewrite_Group, INVALID, 0, RewriteOption, 0,
       "Remove unused static globals and functions", 0)
OPTION(prefix_1, "res-may-alias", res_may_alias, Flag, hlslcomp_Group, INVALID, 0, CoreOption, 0,
       "Assume that UAVs/SRVs may alias", 0)
OPTION(prefix_1, "res_may_alias", res_may_alias_, Flag, hlslcomp_Group, INVALID, 0, CoreOption | HelpHidden, 0,
       "Assume that UAVs/SRVs may alias", 0)
OPTION(prefix_1, "rootsig-define", rootsig_define, Separate, hlslcomp_Group, INVALID, 0, CoreOption, 0,
       "Read root signature from a #define", 0)
OPTION(prefix_1, "setprivate", setprivate, JoinedOrSeparate, hlslutil_Group, INVALID, 0, CoreOption | DriverOption, 0,
       "Private data to add to compiled shader blob", "<file>")
OPTION(prefix_1, "setrootsignature", setrootsignature, JoinedOrSeparate, hlslutil_Group, INVALID, 0, CoreOption | DriverOption, 0,
       "Attach root signature to shader bytecode", "<file>")
OPTION(prefix_1, "skip-fn-body", rw_skip_function_body, Flag, hlslrewrite_Group, INVALID, 0, RewriteOption, 0,
       "Translate function definitions to declarations", 0)
OPTION(prefix_1, "skip-serialization", skip_serialization, Flag, hlslcore_Group, INVALID, 0, CoreOption | HelpHidden, 0,
       "Return a module interface instead of serialized output", 0)
OPTION(prefix_1, "skip-static", rw_skip_static, Flag, hlslrewrite_Group, INVALID, 0, RewriteOption, 0,
       "Remove static functions and globals when used with -skip-fn-body", 0)
OPTION(prefix_3, "spirv", spirv, Flag, spirv_Group, INVALID, 0, CoreOption | DriverOption, 0,
       "Generate SPIR-V code", 0)
OPTION(prefix_1, "strict-udt-casting", strict_udt_casting, Flag, hlslcomp_Group, INVALID, 0, CoreOption | HelpHidden, 0,
       "Require explicit casts between different structure types with compatible layouts.", 0)
OPTION(prefix_1, "T", target_profile, JoinedOrSeparate, hlslcomp_Group, INVALID, 0, CoreOption, 0,
       "Set target profile. \n\t<profile>: ps_6_0, ps_6_1, ps_6_2, ps_6_3, ps_6_4, ps_6_5, ps_6_6, ps_6_7, \n\t\t vs_6_0, vs_6_1, vs_6_2, vs_6_3, vs_6_4, vs_6_5, vs_6_6, vs_6_7, \n\t\t gs_6_0, gs_6_1, gs_6_2, gs_6_3, gs_6_4, gs_6_5, gs_6_6, gs_6_7, \n\t\t hs_6_0, hs_6_1, hs_6_2, hs_6_3, hs_6_4, hs_6_5, hs_6_6, hs_6_7, \n\t\t ds_6_0, ds_6_1, ds_6_2, ds_6_3, ds_6_4, ds_6_5, ds_6_6, ds_6_7, \n\t\t cs_6_0, cs_6_1, cs_6_2, cs_6_3, cs_6_4, cs_6_5, cs_6_6, cs_6_7, \n\t\t lib_6_1, lib_6_2, lib_6_3, lib_6_4, lib_6_5, lib_6_6, lib_6_7, \n\t\t ms_6_5, ms_6_6, ms_6_7, \n\t\t as_6_5, as_6_6, as_6_7, \n\t\t ", "<profile>")
OPTION(prefix_1, "unchanged", rw_unchanged, Flag, hlslrewrite_Group, INVALID, 0, RewriteOption, 0,
       "Rewrite HLSL, without changes.", 0)
OPTION(prefix_1, "validator-version", validator_version, Separate, hlslcomp_Group, INVALID, 0, CoreOption | HelpHidden, 0,
       "Override validator version for module.  Format: <major.minor> ; Default: DXIL.dll version or current internal version.", 0)
OPTION(prefix_1, "Vd", VD, Flag, hlslcomp_Group, INVALID, 0, CoreOption, 0,
       "Disable validation", 0)
OPTION(prefix_1, "verifyrootsignature", verifyrootsignature, JoinedOrSeparate, hlslutil_Group, INVALID, 0, DriverOption, 0,
       "Verify shader bytecode with root signature", "<file>")
OPTION(prefix_2, "version", _version, Flag, hlslcore_Group, INVALID, 0, DriverOption, 0,
       "Display compiler version information", 0)
OPTION(prefix_1, "Vi", _vi, Flag, hlslcomp_Group, H, 0, CoreOption, 0,
       "Display details about the include process.", 0)
OPTION(prefix_3, "vkbr", vkbr, MultiArg, INVALID, fvk_bind_register, 0, CoreOption | DriverOption, 4, 0, 0)
OPTION(prefix_1, "Vn", Vn, JoinedOrSeparate, hlslcomp_Group, INVALID, 0, DriverOption, 0,
       "Use <name> as variable name in header file", "<name>")
OPTION(prefix_3, "Wall", Wall, Flag, W_Group, INVALID, 0, CoreOption, 0, 0, 0)
OPTION(prefix_3, "Wdeprecated", Wdeprecated, Flag, W_Group, INVALID, 0, CoreOption, 0, 0, 0)
OPTION(prefix_3, "Wno-vk-emulated-features", Wno_vk_emulated_features, Joined, spirv_Group, INVALID, 0, CoreOption | DriverOption | HelpHidden, 0,
       "Do not emit warnings for emulated features resulting from no direct mapping", 0)
OPTION(prefix_3, "Wno-vk-ignored-features", Wno_vk_ignored_features, Joined, spirv_Group, INVALID, 0, CoreOption | DriverOption | HelpHidden, 0,
       "Do not emit warnings for ingored features resulting from no Vulkan support", 0)
OPTION(prefix_1, "WX", _SLASH_WX, Flag, hlslcomp_Group, INVALID, 0, CoreOption, 0,
       "Treat warnings as errors", 0)
OPTION(prefix_3, "W", W_Joined, Joined, W_Group, INVALID, 0, CoreOption, 0,
       "Enable/Disable the specified warning", "[no-]<warning>")
OPTION(prefix_1, "Zi", _SLASH_Zi, Flag, hlslcomp_Group, INVALID, 0, CoreOption, 0,
       "Enable debug information. Cannot be used together with -Zs", 0)
OPTION(prefix_1, "Zpc", Zpc, Flag, hlslcomp_Group, INVALID, 0, CoreOption, 0,
       "Pack matrices in column-major order", 0)
OPTION(prefix_1, "Zpr", Zpr, Flag, hlslcomp_Group, INVALID, 0, CoreOption, 0,
       "Pack matrices in row-major order", 0)
OPTION(prefix_1, "Zsb", Zsb, Flag, hlslcomp_Group, INVALID, 0, CoreOption, 0,
       "Compute Shader Hash considering only output binary", 0)
OPTION(prefix_1, "Zss", Zss, Flag, hlslcomp_Group, INVALID, 0, CoreOption, 0,
       "Compute Shader Hash considering source information", 0)
OPTION(prefix_1, "Zs", Zs, Flag, hlslcomp_Group, INVALID, 0, CoreOption, 0,
       "Generate small PDB with just sources and compile options. Cannot be used together with -Zi", 0)
#endif
