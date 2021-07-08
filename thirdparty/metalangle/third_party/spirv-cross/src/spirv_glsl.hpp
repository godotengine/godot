/*
 * Copyright 2015-2020 Arm Limited
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SPIRV_CROSS_GLSL_HPP
#define SPIRV_CROSS_GLSL_HPP

#include "GLSL.std.450.h"
#include "spirv_cross.hpp"
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace SPIRV_CROSS_NAMESPACE
{
enum PlsFormat
{
	PlsNone = 0,

	PlsR11FG11FB10F,
	PlsR32F,
	PlsRG16F,
	PlsRGB10A2,
	PlsRGBA8,
	PlsRG16,

	PlsRGBA8I,
	PlsRG16I,

	PlsRGB10A2UI,
	PlsRGBA8UI,
	PlsRG16UI,
	PlsR32UI
};

struct PlsRemap
{
	uint32_t id;
	PlsFormat format;
};

enum AccessChainFlagBits
{
	ACCESS_CHAIN_INDEX_IS_LITERAL_BIT = 1 << 0,
	ACCESS_CHAIN_CHAIN_ONLY_BIT = 1 << 1,
	ACCESS_CHAIN_PTR_CHAIN_BIT = 1 << 2,
	ACCESS_CHAIN_SKIP_REGISTER_EXPRESSION_READ_BIT = 1 << 3,
	ACCESS_CHAIN_LITERAL_MSB_FORCE_ID = 1 << 4
};
typedef uint32_t AccessChainFlags;

class CompilerGLSL : public Compiler
{
public:
	struct Options
	{
		// The shading language version. Corresponds to #version $VALUE.
		uint32_t version = 450;

		// Emit the OpenGL ES shading language instead of desktop OpenGL.
		bool es = false;

		// Debug option to always emit temporary variables for all expressions.
		bool force_temporary = false;

		// If true, Vulkan GLSL features are used instead of GL-compatible features.
		// Mostly useful for debugging SPIR-V files.
		bool vulkan_semantics = false;

		// If true, gl_PerVertex is explicitly redeclared in vertex, geometry and tessellation shaders.
		// The members of gl_PerVertex is determined by which built-ins are declared by the shader.
		// This option is ignored in ES versions, as redeclaration in ES is not required, and it depends on a different extension
		// (EXT_shader_io_blocks) which makes things a bit more fuzzy.
		bool separate_shader_objects = false;

		// Flattens multidimensional arrays, e.g. float foo[a][b][c] into single-dimensional arrays,
		// e.g. float foo[a * b * c].
		// This function does not change the actual SPIRType of any object.
		// Only the generated code, including declarations of interface variables are changed to be single array dimension.
		bool flatten_multidimensional_arrays = false;

		// For older desktop GLSL targets than version 420, the
		// GL_ARB_shading_language_420pack extensions is used to be able to support
		// layout(binding) on UBOs and samplers.
		// If disabled on older targets, binding decorations will be stripped.
		bool enable_420pack_extension = true;

		// In non-Vulkan GLSL, emit push constant blocks as UBOs rather than plain uniforms.
		bool emit_push_constant_as_uniform_buffer = false;

		// Always emit uniform blocks as plain uniforms, regardless of the GLSL version, even when UBOs are supported.
		// Does not apply to shader storage or push constant blocks.
		bool emit_uniform_buffer_as_plain_uniforms = false;

		// Emit OpLine directives if present in the module.
		// May not correspond exactly to original source, but should be a good approximation.
		bool emit_line_directives = false;

		// In cases where readonly/writeonly decoration are not used at all,
		// we try to deduce which qualifier(s) we should actually used, since actually emitting
		// read-write decoration is very rare, and older glslang/HLSL compilers tend to just emit readwrite as a matter of fact.
		// The default (true) is to enable automatic deduction for these cases, but if you trust the decorations set
		// by the SPIR-V, it's recommended to set this to false.
		bool enable_storage_image_qualifier_deduction = true;

		// On some targets (WebGPU), uninitialized variables are banned.
		// If this is enabled, all variables (temporaries, Private, Function)
		// which would otherwise be uninitialized will now be initialized to 0 instead.
		bool force_zero_initialized_variables = false;

		enum Precision
		{
			DontCare,
			Lowp,
			Mediump,
			Highp
		};

		struct
		{
			// GLSL: In vertex shaders, rewrite [0, w] depth (Vulkan/D3D style) to [-w, w] depth (GL style).
			// MSL: In vertex shaders, rewrite [-w, w] depth (GL style) to [0, w] depth.
			// HLSL: In vertex shaders, rewrite [-w, w] depth (GL style) to [0, w] depth.
			bool fixup_clipspace = false;

			// Inverts gl_Position.y or equivalent.
			bool flip_vert_y = false;

			// GLSL only, for HLSL version of this option, see CompilerHLSL.
			// If true, the backend will assume that InstanceIndex will need to apply
			// a base instance offset. Set to false if you know you will never use base instance
			// functionality as it might remove some internal uniforms.
			bool support_nonzero_base_instance = true;
		} vertex;

		struct
		{
			// Add precision mediump float in ES targets when emitting GLES source.
			// Add precision highp int in ES targets when emitting GLES source.
			Precision default_float_precision = Mediump;
			Precision default_int_precision = Highp;
		} fragment;
	};

	void remap_pixel_local_storage(std::vector<PlsRemap> inputs, std::vector<PlsRemap> outputs)
	{
		pls_inputs = std::move(inputs);
		pls_outputs = std::move(outputs);
		remap_pls_variables();
	}

	// Redirect a subpassInput reading from input_attachment_index to instead load its value from
	// the color attachment at location = color_location. Requires ESSL.
	void remap_ext_framebuffer_fetch(uint32_t input_attachment_index, uint32_t color_location);

	explicit CompilerGLSL(std::vector<uint32_t> spirv_)
	    : Compiler(std::move(spirv_))
	{
		init();
	}

	CompilerGLSL(const uint32_t *ir_, size_t word_count)
	    : Compiler(ir_, word_count)
	{
		init();
	}

	explicit CompilerGLSL(const ParsedIR &ir_)
	    : Compiler(ir_)
	{
		init();
	}

	explicit CompilerGLSL(ParsedIR &&ir_)
	    : Compiler(std::move(ir_))
	{
		init();
	}

	const Options &get_common_options() const
	{
		return options;
	}

	void set_common_options(const Options &opts)
	{
		options = opts;
	}

	std::string compile() override;

	// Returns the current string held in the conversion buffer. Useful for
	// capturing what has been converted so far when compile() throws an error.
	std::string get_partial_source();

	// Adds a line to be added right after #version in GLSL backend.
	// This is useful for enabling custom extensions which are outside the scope of SPIRV-Cross.
	// This can be combined with variable remapping.
	// A new-line will be added.
	//
	// While add_header_line() is a more generic way of adding arbitrary text to the header
	// of a GLSL file, require_extension() should be used when adding extensions since it will
	// avoid creating collisions with SPIRV-Cross generated extensions.
	//
	// Code added via add_header_line() is typically backend-specific.
	void add_header_line(const std::string &str);

	// Adds an extension which is required to run this shader, e.g.
	// require_extension("GL_KHR_my_extension");
	void require_extension(const std::string &ext);

	// Legacy GLSL compatibility method.
	// Takes a uniform or push constant variable and flattens it into a (i|u)vec4 array[N]; array instead.
	// For this to work, all types in the block must be the same basic type, e.g. mixing vec2 and vec4 is fine, but
	// mixing int and float is not.
	// The name of the uniform array will be the same as the interface block name.
	void flatten_buffer_block(VariableID id);

	// After compilation, query if a variable ID was used as a depth resource.
	// This is meaningful for MSL since descriptor types depend on this knowledge.
	// Cases which return true:
	// - Images which are declared with depth = 1 image type.
	// - Samplers which are statically used at least once with Dref opcodes.
	// - Images which are statically used at least once with Dref opcodes.
	bool variable_is_depth_or_compare(VariableID id) const;

protected:
	void reset();
	void emit_function(SPIRFunction &func, const Bitset &return_flags);

	bool has_extension(const std::string &ext) const;
	void require_extension_internal(const std::string &ext);

	// Virtualize methods which need to be overridden by subclass targets like C++ and such.
	virtual void emit_function_prototype(SPIRFunction &func, const Bitset &return_flags);

	SPIRBlock *current_emitting_block = nullptr;
	SPIRBlock *current_emitting_switch = nullptr;
	bool current_emitting_switch_fallthrough = false;

	virtual void emit_instruction(const Instruction &instr);
	void emit_block_instructions(SPIRBlock &block);
	virtual void emit_glsl_op(uint32_t result_type, uint32_t result_id, uint32_t op, const uint32_t *args,
	                          uint32_t count);
	virtual void emit_spv_amd_shader_ballot_op(uint32_t result_type, uint32_t result_id, uint32_t op,
	                                           const uint32_t *args, uint32_t count);
	virtual void emit_spv_amd_shader_explicit_vertex_parameter_op(uint32_t result_type, uint32_t result_id, uint32_t op,
	                                                              const uint32_t *args, uint32_t count);
	virtual void emit_spv_amd_shader_trinary_minmax_op(uint32_t result_type, uint32_t result_id, uint32_t op,
	                                                   const uint32_t *args, uint32_t count);
	virtual void emit_spv_amd_gcn_shader_op(uint32_t result_type, uint32_t result_id, uint32_t op, const uint32_t *args,
	                                        uint32_t count);
	virtual void emit_header();
	void emit_line_directive(uint32_t file_id, uint32_t line_literal);
	void build_workgroup_size(SmallVector<std::string> &arguments, const SpecializationConstant &x,
	                          const SpecializationConstant &y, const SpecializationConstant &z);

	virtual void emit_sampled_image_op(uint32_t result_type, uint32_t result_id, uint32_t image_id, uint32_t samp_id);
	virtual void emit_texture_op(const Instruction &i);
	virtual std::string to_texture_op(const Instruction &i, bool *forward,
	                                  SmallVector<uint32_t> &inherited_expressions);
	virtual void emit_subgroup_op(const Instruction &i);
	virtual std::string type_to_glsl(const SPIRType &type, uint32_t id = 0);
	virtual std::string builtin_to_glsl(spv::BuiltIn builtin, spv::StorageClass storage);
	virtual void emit_struct_member(const SPIRType &type, uint32_t member_type_id, uint32_t index,
	                                const std::string &qualifier = "", uint32_t base_offset = 0);
	virtual void emit_struct_padding_target(const SPIRType &type);
	virtual std::string image_type_glsl(const SPIRType &type, uint32_t id = 0);
	std::string constant_expression(const SPIRConstant &c);
	std::string constant_op_expression(const SPIRConstantOp &cop);
	virtual std::string constant_expression_vector(const SPIRConstant &c, uint32_t vector);
	virtual void emit_fixup();
	virtual std::string variable_decl(const SPIRType &type, const std::string &name, uint32_t id = 0);
	virtual std::string to_func_call_arg(const SPIRFunction::Parameter &arg, uint32_t id);
	virtual std::string to_function_name(VariableID img, const SPIRType &imgtype, bool is_fetch, bool is_gather,
	                                     bool is_proj, bool has_array_offsets, bool has_offset, bool has_grad,
	                                     bool has_dref, uint32_t lod, uint32_t minlod);
	virtual std::string to_function_args(VariableID img, const SPIRType &imgtype, bool is_fetch, bool is_gather,
	                                     bool is_proj, uint32_t coord, uint32_t coord_components, uint32_t dref,
	                                     uint32_t grad_x, uint32_t grad_y, uint32_t lod, uint32_t coffset,
	                                     uint32_t offset, uint32_t bias, uint32_t comp, uint32_t sample,
	                                     uint32_t minlod, bool *p_forward);
	virtual void emit_buffer_block(const SPIRVariable &type);
	virtual void emit_push_constant_block(const SPIRVariable &var);
	virtual void emit_uniform(const SPIRVariable &var);
	virtual std::string unpack_expression_type(std::string expr_str, const SPIRType &type, uint32_t physical_type_id,
	                                           bool packed_type, bool row_major);

	virtual bool builtin_translates_to_nonarray(spv::BuiltIn builtin) const;

	void emit_copy_logical_type(uint32_t lhs_id, uint32_t lhs_type_id, uint32_t rhs_id, uint32_t rhs_type_id,
	                            SmallVector<uint32_t> chain);

	StringStream<> buffer;

	template <typename T>
	inline void statement_inner(T &&t)
	{
		buffer << std::forward<T>(t);
		statement_count++;
	}

	template <typename T, typename... Ts>
	inline void statement_inner(T &&t, Ts &&... ts)
	{
		buffer << std::forward<T>(t);
		statement_count++;
		statement_inner(std::forward<Ts>(ts)...);
	}

	template <typename... Ts>
	inline void statement(Ts &&... ts)
	{
		if (is_forcing_recompilation())
		{
			// Do not bother emitting code while force_recompile is active.
			// We will compile again.
			statement_count++;
			return;
		}

		if (redirect_statement)
		{
			redirect_statement->push_back(join(std::forward<Ts>(ts)...));
			statement_count++;
		}
		else
		{
			for (uint32_t i = 0; i < indent; i++)
				buffer << "    ";
			statement_inner(std::forward<Ts>(ts)...);
			buffer << '\n';
		}
	}

	template <typename... Ts>
	inline void statement_no_indent(Ts &&... ts)
	{
		auto old_indent = indent;
		indent = 0;
		statement(std::forward<Ts>(ts)...);
		indent = old_indent;
	}

	// Used for implementing continue blocks where
	// we want to obtain a list of statements we can merge
	// on a single line separated by comma.
	SmallVector<std::string> *redirect_statement = nullptr;
	const SPIRBlock *current_continue_block = nullptr;

	void begin_scope();
	void end_scope();
	void end_scope(const std::string &trailer);
	void end_scope_decl();
	void end_scope_decl(const std::string &decl);

	Options options;

	virtual std::string type_to_array_glsl(
	    const SPIRType &type); // Allow Metal to use the array<T> template to make arrays a value type
	std::string to_array_size(const SPIRType &type, uint32_t index);
	uint32_t to_array_size_literal(const SPIRType &type, uint32_t index) const;
	uint32_t to_array_size_literal(const SPIRType &type) const;
	virtual std::string variable_decl(const SPIRVariable &variable); // Threadgroup arrays can't have a wrapper type
	std::string variable_decl_function_local(SPIRVariable &variable);

	void add_local_variable_name(uint32_t id);
	void add_resource_name(uint32_t id);
	void add_member_name(SPIRType &type, uint32_t name);
	void add_function_overload(const SPIRFunction &func);

	virtual bool is_non_native_row_major_matrix(uint32_t id);
	virtual bool member_is_non_native_row_major_matrix(const SPIRType &type, uint32_t index);
	bool member_is_remapped_physical_type(const SPIRType &type, uint32_t index) const;
	bool member_is_packed_physical_type(const SPIRType &type, uint32_t index) const;
	virtual std::string convert_row_major_matrix(std::string exp_str, const SPIRType &exp_type,
	                                             uint32_t physical_type_id, bool is_packed);

	std::unordered_set<std::string> local_variable_names;
	std::unordered_set<std::string> resource_names;
	std::unordered_set<std::string> block_input_names;
	std::unordered_set<std::string> block_output_names;
	std::unordered_set<std::string> block_ubo_names;
	std::unordered_set<std::string> block_ssbo_names;
	std::unordered_set<std::string> block_names; // A union of all block_*_names.
	std::unordered_map<std::string, std::unordered_set<uint64_t>> function_overloads;
	std::unordered_map<uint32_t, std::string> preserved_aliases;
	void preserve_alias_on_reset(uint32_t id);
	void reset_name_caches();

	bool processing_entry_point = false;

	// Can be overriden by subclass backends for trivial things which
	// shouldn't need polymorphism.
	struct BackendVariations
	{
		std::string discard_literal = "discard";
		std::string demote_literal = "demote";
		std::string null_pointer_literal = "";
		bool float_literal_suffix = false;
		bool double_literal_suffix = true;
		bool uint32_t_literal_suffix = true;
		bool long_long_literal_suffix = false;
		const char *basic_int_type = "int";
		const char *basic_uint_type = "uint";
		const char *basic_int8_type = "int8_t";
		const char *basic_uint8_type = "uint8_t";
		const char *basic_int16_type = "int16_t";
		const char *basic_uint16_type = "uint16_t";
		const char *int16_t_literal_suffix = "s";
		const char *uint16_t_literal_suffix = "us";
		const char *nonuniform_qualifier = "nonuniformEXT";
		const char *boolean_mix_function = "mix";
		bool swizzle_is_function = false;
		bool shared_is_implied = false;
		bool unsized_array_supported = true;
		bool explicit_struct_type = false;
		bool use_initializer_list = false;
		bool use_typed_initializer_list = false;
		bool can_declare_struct_inline = true;
		bool can_declare_arrays_inline = true;
		bool native_row_major_matrix = true;
		bool use_constructor_splatting = true;
		bool allow_precision_qualifiers = false;
		bool can_swizzle_scalar = false;
		bool force_gl_in_out_block = false;
		bool can_return_array = true;
		bool allow_truncated_access_chain = false;
		bool supports_extensions = false;
		bool supports_empty_struct = false;
		bool array_is_value_type = true;
		bool comparison_image_samples_scalar = false;
		bool native_pointers = false;
		bool support_small_type_sampling_result = false;
		bool support_case_fallthrough = true;
		bool use_array_constructor = false;
	} backend;

	void emit_struct(SPIRType &type);
	void emit_resources();
	void emit_buffer_block_native(const SPIRVariable &var);
	void emit_buffer_reference_block(SPIRType &type, bool forward_declaration);
	void emit_buffer_block_legacy(const SPIRVariable &var);
	void emit_buffer_block_flattened(const SPIRVariable &type);
	void emit_declared_builtin_block(spv::StorageClass storage, spv::ExecutionModel model);
	bool should_force_emit_builtin_block(spv::StorageClass storage);
	void emit_push_constant_block_vulkan(const SPIRVariable &var);
	void emit_push_constant_block_glsl(const SPIRVariable &var);
	void emit_interface_block(const SPIRVariable &type);
	void emit_flattened_io_block(const SPIRVariable &var, const char *qual);
	void emit_block_chain(SPIRBlock &block);
	void emit_hoisted_temporaries(SmallVector<std::pair<TypeID, ID>> &temporaries);
	std::string constant_value_macro_name(uint32_t id);
	void emit_constant(const SPIRConstant &constant);
	void emit_specialization_constant_op(const SPIRConstantOp &constant);
	std::string emit_continue_block(uint32_t continue_block, bool follow_true_block, bool follow_false_block);
	bool attempt_emit_loop_header(SPIRBlock &block, SPIRBlock::Method method);

	void branch(BlockID from, BlockID to);
	void branch_to_continue(BlockID from, BlockID to);
	void branch(BlockID from, uint32_t cond, BlockID true_block, BlockID false_block);
	void flush_phi(BlockID from, BlockID to);
	void flush_variable_declaration(uint32_t id);
	void flush_undeclared_variables(SPIRBlock &block);
	void emit_variable_temporary_copies(const SPIRVariable &var);

	bool should_dereference(uint32_t id);
	bool should_forward(uint32_t id) const;
	bool should_suppress_usage_tracking(uint32_t id) const;
	void emit_mix_op(uint32_t result_type, uint32_t id, uint32_t left, uint32_t right, uint32_t lerp);
	void emit_nminmax_op(uint32_t result_type, uint32_t id, uint32_t op0, uint32_t op1, GLSLstd450 op);
	bool to_trivial_mix_op(const SPIRType &type, std::string &op, uint32_t left, uint32_t right, uint32_t lerp);
	void emit_quaternary_func_op(uint32_t result_type, uint32_t result_id, uint32_t op0, uint32_t op1, uint32_t op2,
	                             uint32_t op3, const char *op);
	void emit_trinary_func_op(uint32_t result_type, uint32_t result_id, uint32_t op0, uint32_t op1, uint32_t op2,
	                          const char *op);
	void emit_binary_func_op(uint32_t result_type, uint32_t result_id, uint32_t op0, uint32_t op1, const char *op);

	void emit_unary_func_op_cast(uint32_t result_type, uint32_t result_id, uint32_t op0, const char *op,
	                             SPIRType::BaseType input_type, SPIRType::BaseType expected_result_type);
	void emit_binary_func_op_cast(uint32_t result_type, uint32_t result_id, uint32_t op0, uint32_t op1, const char *op,
	                              SPIRType::BaseType input_type, bool skip_cast_if_equal_type);
	void emit_binary_func_op_cast_clustered(uint32_t result_type, uint32_t result_id, uint32_t op0, uint32_t op1,
	                                        const char *op, SPIRType::BaseType input_type);
	void emit_trinary_func_op_cast(uint32_t result_type, uint32_t result_id, uint32_t op0, uint32_t op1, uint32_t op2,
	                               const char *op, SPIRType::BaseType input_type);
	void emit_trinary_func_op_bitextract(uint32_t result_type, uint32_t result_id, uint32_t op0, uint32_t op1,
	                                     uint32_t op2, const char *op, SPIRType::BaseType expected_result_type,
	                                     SPIRType::BaseType input_type0, SPIRType::BaseType input_type1,
	                                     SPIRType::BaseType input_type2);
	void emit_bitfield_insert_op(uint32_t result_type, uint32_t result_id, uint32_t op0, uint32_t op1, uint32_t op2,
	                             uint32_t op3, const char *op, SPIRType::BaseType offset_count_type);

	void emit_unary_func_op(uint32_t result_type, uint32_t result_id, uint32_t op0, const char *op);
	void emit_unrolled_unary_op(uint32_t result_type, uint32_t result_id, uint32_t operand, const char *op);
	void emit_binary_op(uint32_t result_type, uint32_t result_id, uint32_t op0, uint32_t op1, const char *op);
	void emit_unrolled_binary_op(uint32_t result_type, uint32_t result_id, uint32_t op0, uint32_t op1, const char *op,
	                             bool negate, SPIRType::BaseType expected_type);
	void emit_binary_op_cast(uint32_t result_type, uint32_t result_id, uint32_t op0, uint32_t op1, const char *op,
	                         SPIRType::BaseType input_type, bool skip_cast_if_equal_type);

	SPIRType binary_op_bitcast_helper(std::string &cast_op0, std::string &cast_op1, SPIRType::BaseType &input_type,
	                                  uint32_t op0, uint32_t op1, bool skip_cast_if_equal_type);

	std::string to_ternary_expression(const SPIRType &result_type, uint32_t select, uint32_t true_value,
	                                  uint32_t false_value);

	void emit_unary_op(uint32_t result_type, uint32_t result_id, uint32_t op0, const char *op);
	bool expression_is_forwarded(uint32_t id) const;
	bool expression_suppresses_usage_tracking(uint32_t id) const;
	SPIRExpression &emit_op(uint32_t result_type, uint32_t result_id, const std::string &rhs, bool forward_rhs,
	                        bool suppress_usage_tracking = false);

	void access_chain_internal_append_index(std::string &expr, uint32_t base, const SPIRType *type,
	                                        AccessChainFlags flags, bool &access_chain_is_arrayed, uint32_t index);

	std::string access_chain_internal(uint32_t base, const uint32_t *indices, uint32_t count, AccessChainFlags flags,
	                                  AccessChainMeta *meta);

	std::string access_chain(uint32_t base, const uint32_t *indices, uint32_t count, const SPIRType &target_type,
	                         AccessChainMeta *meta = nullptr, bool ptr_chain = false);

	std::string flattened_access_chain(uint32_t base, const uint32_t *indices, uint32_t count,
	                                   const SPIRType &target_type, uint32_t offset, uint32_t matrix_stride,
	                                   uint32_t array_stride, bool need_transpose);
	std::string flattened_access_chain_struct(uint32_t base, const uint32_t *indices, uint32_t count,
	                                          const SPIRType &target_type, uint32_t offset);
	std::string flattened_access_chain_matrix(uint32_t base, const uint32_t *indices, uint32_t count,
	                                          const SPIRType &target_type, uint32_t offset, uint32_t matrix_stride,
	                                          bool need_transpose);
	std::string flattened_access_chain_vector(uint32_t base, const uint32_t *indices, uint32_t count,
	                                          const SPIRType &target_type, uint32_t offset, uint32_t matrix_stride,
	                                          bool need_transpose);
	std::pair<std::string, uint32_t> flattened_access_chain_offset(const SPIRType &basetype, const uint32_t *indices,
	                                                               uint32_t count, uint32_t offset,
	                                                               uint32_t word_stride, bool *need_transpose = nullptr,
	                                                               uint32_t *matrix_stride = nullptr,
	                                                               uint32_t *array_stride = nullptr,
	                                                               bool ptr_chain = false);

	const char *index_to_swizzle(uint32_t index);
	std::string remap_swizzle(const SPIRType &result_type, uint32_t input_components, const std::string &expr);
	std::string declare_temporary(uint32_t type, uint32_t id);
	void emit_uninitialized_temporary(uint32_t type, uint32_t id);
	SPIRExpression &emit_uninitialized_temporary_expression(uint32_t type, uint32_t id);
	void append_global_func_args(const SPIRFunction &func, uint32_t index, SmallVector<std::string> &arglist);
	std::string to_expression(uint32_t id, bool register_expression_read = true);
	std::string to_composite_constructor_expression(uint32_t id);
	std::string to_rerolled_array_expression(const std::string &expr, const SPIRType &type);
	std::string to_enclosed_expression(uint32_t id, bool register_expression_read = true);
	std::string to_unpacked_expression(uint32_t id, bool register_expression_read = true);
	std::string to_unpacked_row_major_matrix_expression(uint32_t id);
	std::string to_enclosed_unpacked_expression(uint32_t id, bool register_expression_read = true);
	std::string to_dereferenced_expression(uint32_t id, bool register_expression_read = true);
	std::string to_pointer_expression(uint32_t id, bool register_expression_read = true);
	std::string to_enclosed_pointer_expression(uint32_t id, bool register_expression_read = true);
	std::string to_extract_component_expression(uint32_t id, uint32_t index);
	std::string enclose_expression(const std::string &expr);
	std::string dereference_expression(const SPIRType &expression_type, const std::string &expr);
	std::string address_of_expression(const std::string &expr);
	void strip_enclosed_expression(std::string &expr);
	std::string to_member_name(const SPIRType &type, uint32_t index);
	virtual std::string to_member_reference(uint32_t base, const SPIRType &type, uint32_t index, bool ptr_chain);
	std::string type_to_glsl_constructor(const SPIRType &type);
	std::string argument_decl(const SPIRFunction::Parameter &arg);
	virtual std::string to_qualifiers_glsl(uint32_t id);
	const char *to_precision_qualifiers_glsl(uint32_t id);
	virtual const char *to_storage_qualifiers_glsl(const SPIRVariable &var);
	const char *flags_to_qualifiers_glsl(const SPIRType &type, const Bitset &flags);
	const char *format_to_glsl(spv::ImageFormat format);
	virtual std::string layout_for_member(const SPIRType &type, uint32_t index);
	virtual std::string to_interpolation_qualifiers(const Bitset &flags);
	std::string layout_for_variable(const SPIRVariable &variable);
	std::string to_combined_image_sampler(VariableID image_id, VariableID samp_id);
	virtual bool skip_argument(uint32_t id) const;
	virtual void emit_array_copy(const std::string &lhs, uint32_t rhs_id, spv::StorageClass lhs_storage,
	                             spv::StorageClass rhs_storage);
	virtual void emit_block_hints(const SPIRBlock &block);
	virtual std::string to_initializer_expression(const SPIRVariable &var);
	virtual std::string to_zero_initialized_expression(uint32_t type_id);
	bool type_can_zero_initialize(const SPIRType &type) const;

	bool buffer_is_packing_standard(const SPIRType &type, BufferPackingStandard packing,
	                                uint32_t *failed_index = nullptr, uint32_t start_offset = 0,
	                                uint32_t end_offset = ~(0u));
	std::string buffer_to_packing_standard(const SPIRType &type, bool support_std430_without_scalar_layout);

	uint32_t type_to_packed_base_size(const SPIRType &type, BufferPackingStandard packing);
	uint32_t type_to_packed_alignment(const SPIRType &type, const Bitset &flags, BufferPackingStandard packing);
	uint32_t type_to_packed_array_stride(const SPIRType &type, const Bitset &flags, BufferPackingStandard packing);
	uint32_t type_to_packed_size(const SPIRType &type, const Bitset &flags, BufferPackingStandard packing);

	std::string bitcast_glsl(const SPIRType &result_type, uint32_t arg);
	virtual std::string bitcast_glsl_op(const SPIRType &result_type, const SPIRType &argument_type);

	std::string bitcast_expression(SPIRType::BaseType target_type, uint32_t arg);
	std::string bitcast_expression(const SPIRType &target_type, SPIRType::BaseType expr_type, const std::string &expr);

	std::string build_composite_combiner(uint32_t result_type, const uint32_t *elems, uint32_t length);
	bool remove_duplicate_swizzle(std::string &op);
	bool remove_unity_swizzle(uint32_t base, std::string &op);

	// Can modify flags to remote readonly/writeonly if image type
	// and force recompile.
	bool check_atomic_image(uint32_t id);

	virtual void replace_illegal_names();
	void replace_illegal_names(const std::unordered_set<std::string> &keywords);
	virtual void emit_entry_point_declarations();

	void replace_fragment_output(SPIRVariable &var);
	void replace_fragment_outputs();
	bool check_explicit_lod_allowed(uint32_t lod);
	std::string legacy_tex_op(const std::string &op, const SPIRType &imgtype, uint32_t lod, uint32_t id);

	uint32_t indent = 0;

	std::unordered_set<uint32_t> emitted_functions;

	// Ensure that we declare phi-variable copies even if the original declaration isn't deferred
	std::unordered_set<uint32_t> flushed_phi_variables;

	std::unordered_set<uint32_t> flattened_buffer_blocks;
	std::unordered_set<uint32_t> flattened_structs;

	std::string load_flattened_struct(SPIRVariable &var);
	std::string to_flattened_struct_member(const SPIRVariable &var, uint32_t index);
	void store_flattened_struct(SPIRVariable &var, uint32_t value);

	// Usage tracking. If a temporary is used more than once, use the temporary instead to
	// avoid AST explosion when SPIRV is generated with pure SSA and doesn't write stuff to variables.
	std::unordered_map<uint32_t, uint32_t> expression_usage_counts;
	void track_expression_read(uint32_t id);

	SmallVector<std::string> forced_extensions;
	SmallVector<std::string> header_lines;

	// Used when expressions emit extra opcodes with their own unique IDs,
	// and we need to reuse the IDs across recompilation loops.
	// Currently used by NMin/Max/Clamp implementations.
	std::unordered_map<uint32_t, uint32_t> extra_sub_expressions;

	uint32_t statement_count = 0;

	inline bool is_legacy() const
	{
		return (options.es && options.version < 300) || (!options.es && options.version < 130);
	}

	inline bool is_legacy_es() const
	{
		return options.es && options.version < 300;
	}

	inline bool is_legacy_desktop() const
	{
		return !options.es && options.version < 130;
	}

	bool args_will_forward(uint32_t id, const uint32_t *args, uint32_t num_args, bool pure);
	void register_call_out_argument(uint32_t id);
	void register_impure_function_call();
	void register_control_dependent_expression(uint32_t expr);

	// GL_EXT_shader_pixel_local_storage support.
	std::vector<PlsRemap> pls_inputs;
	std::vector<PlsRemap> pls_outputs;
	std::string pls_decl(const PlsRemap &variable);
	const char *to_pls_qualifiers_glsl(const SPIRVariable &variable);
	void emit_pls();
	void remap_pls_variables();

	// GL_EXT_shader_framebuffer_fetch support.
	std::vector<std::pair<uint32_t, uint32_t>> subpass_to_framebuffer_fetch_attachment;
	std::unordered_set<uint32_t> inout_color_attachments;
	bool subpass_input_is_framebuffer_fetch(uint32_t id) const;
	void emit_inout_fragment_outputs_copy_to_subpass_inputs();
	const SPIRVariable *find_subpass_input_by_attachment_index(uint32_t index) const;
	const SPIRVariable *find_color_output_by_location(uint32_t location) const;

	// A variant which takes two sets of name. The secondary is only used to verify there are no collisions,
	// but the set is not updated when we have found a new name.
	// Used primarily when adding block interface names.
	void add_variable(std::unordered_set<std::string> &variables_primary,
	                  const std::unordered_set<std::string> &variables_secondary, std::string &name);

	void check_function_call_constraints(const uint32_t *args, uint32_t length);
	void handle_invalid_expression(uint32_t id);
	void find_static_extensions();

	std::string emit_for_loop_initializers(const SPIRBlock &block);
	void emit_while_loop_initializers(const SPIRBlock &block);
	bool for_loop_initializers_are_same_type(const SPIRBlock &block);
	bool optimize_read_modify_write(const SPIRType &type, const std::string &lhs, const std::string &rhs);
	void fixup_image_load_store_access();

	bool type_is_empty(const SPIRType &type);

	virtual void declare_undefined_values();

	static std::string sanitize_underscores(const std::string &str);

	bool can_use_io_location(spv::StorageClass storage, bool block);
	const Instruction *get_next_instruction_in_block(const Instruction &instr);
	static uint32_t mask_relevant_memory_semantics(uint32_t semantics);

	std::string convert_half_to_string(const SPIRConstant &value, uint32_t col, uint32_t row);
	std::string convert_float_to_string(const SPIRConstant &value, uint32_t col, uint32_t row);
	std::string convert_double_to_string(const SPIRConstant &value, uint32_t col, uint32_t row);

	std::string convert_separate_image_to_expression(uint32_t id);

	// Builtins in GLSL are always specific signedness, but the SPIR-V can declare them
	// as either unsigned or signed.
	// Sometimes we will need to automatically perform bitcasts on load and store to make this work.
	virtual void bitcast_to_builtin_store(uint32_t target_id, std::string &expr, const SPIRType &expr_type);
	virtual void bitcast_from_builtin_load(uint32_t source_id, std::string &expr, const SPIRType &expr_type);
	void unroll_array_from_complex_load(uint32_t target_id, uint32_t source_id, std::string &expr);
	void convert_non_uniform_expression(const SPIRType &type, std::string &expr);

	void handle_store_to_invariant_variable(uint32_t store_id, uint32_t value_id);
	void disallow_forwarding_in_expression_chain(const SPIRExpression &expr);

	bool expression_is_constant_null(uint32_t id) const;
	virtual void emit_store_statement(uint32_t lhs_expression, uint32_t rhs_expression);

	uint32_t get_integer_width_for_instruction(const Instruction &instr) const;
	uint32_t get_integer_width_for_glsl_instruction(GLSLstd450 op, const uint32_t *arguments, uint32_t length) const;

	bool variable_is_lut(const SPIRVariable &var) const;

	char current_locale_radix_character = '.';

	void fixup_type_alias();
	void reorder_type_alias();

	void propagate_nonuniform_qualifier(uint32_t id);

	static const char *vector_swizzle(int vecsize, int index);

private:
	void init();
};
} // namespace SPIRV_CROSS_NAMESPACE

#endif
