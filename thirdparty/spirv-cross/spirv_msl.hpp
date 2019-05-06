/*
 * Copyright 2016-2019 The Brenwill Workshop Ltd.
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

#ifndef SPIRV_CROSS_MSL_HPP
#define SPIRV_CROSS_MSL_HPP

#include "spirv_glsl.hpp"
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>

namespace SPIRV_CROSS_NAMESPACE
{

// Indicates the format of the vertex attribute. Currently limited to specifying
// if the attribute is an 8-bit unsigned integer, 16-bit unsigned integer, or
// some other format.
enum MSLVertexFormat
{
	MSL_VERTEX_FORMAT_OTHER = 0,
	MSL_VERTEX_FORMAT_UINT8 = 1,
	MSL_VERTEX_FORMAT_UINT16 = 2,
	MSL_VERTEX_FORMAT_INT_MAX = 0x7fffffff
};

// Defines MSL characteristics of a vertex attribute at a particular location.
// After compilation, it is possible to query whether or not this location was used.
struct MSLVertexAttr
{
	uint32_t location = 0;
	uint32_t msl_buffer = 0;
	uint32_t msl_offset = 0;
	uint32_t msl_stride = 0;
	bool per_instance = false;
	MSLVertexFormat format = MSL_VERTEX_FORMAT_OTHER;
	spv::BuiltIn builtin = spv::BuiltInMax;
};

// Matches the binding index of a MSL resource for a binding within a descriptor set.
// Taken together, the stage, desc_set and binding combine to form a reference to a resource
// descriptor used in a particular shading stage.
// If using MSL 2.0 argument buffers, and the descriptor set is not marked as a discrete descriptor set,
// the binding reference we remap to will become an [[id(N)]] attribute within
// the "descriptor set" argument buffer structure.
// For resources which are bound in the "classic" MSL 1.0 way or discrete descriptors, the remap will become a
// [[buffer(N)]], [[texture(N)]] or [[sampler(N)]] depending on the resource types used.
struct MSLResourceBinding
{
	spv::ExecutionModel stage = spv::ExecutionModelMax;
	uint32_t desc_set = 0;
	uint32_t binding = 0;
	uint32_t msl_buffer = 0;
	uint32_t msl_texture = 0;
	uint32_t msl_sampler = 0;
};

enum MSLSamplerCoord
{
	MSL_SAMPLER_COORD_NORMALIZED = 0,
	MSL_SAMPLER_COORD_PIXEL = 1,
	MSL_SAMPLER_INT_MAX = 0x7fffffff
};

enum MSLSamplerFilter
{
	MSL_SAMPLER_FILTER_NEAREST = 0,
	MSL_SAMPLER_FILTER_LINEAR = 1,
	MSL_SAMPLER_FILTER_INT_MAX = 0x7fffffff
};

enum MSLSamplerMipFilter
{
	MSL_SAMPLER_MIP_FILTER_NONE = 0,
	MSL_SAMPLER_MIP_FILTER_NEAREST = 1,
	MSL_SAMPLER_MIP_FILTER_LINEAR = 2,
	MSL_SAMPLER_MIP_FILTER_INT_MAX = 0x7fffffff
};

enum MSLSamplerAddress
{
	MSL_SAMPLER_ADDRESS_CLAMP_TO_ZERO = 0,
	MSL_SAMPLER_ADDRESS_CLAMP_TO_EDGE = 1,
	MSL_SAMPLER_ADDRESS_CLAMP_TO_BORDER = 2,
	MSL_SAMPLER_ADDRESS_REPEAT = 3,
	MSL_SAMPLER_ADDRESS_MIRRORED_REPEAT = 4,
	MSL_SAMPLER_ADDRESS_INT_MAX = 0x7fffffff
};

enum MSLSamplerCompareFunc
{
	MSL_SAMPLER_COMPARE_FUNC_NEVER = 0,
	MSL_SAMPLER_COMPARE_FUNC_LESS = 1,
	MSL_SAMPLER_COMPARE_FUNC_LESS_EQUAL = 2,
	MSL_SAMPLER_COMPARE_FUNC_GREATER = 3,
	MSL_SAMPLER_COMPARE_FUNC_GREATER_EQUAL = 4,
	MSL_SAMPLER_COMPARE_FUNC_EQUAL = 5,
	MSL_SAMPLER_COMPARE_FUNC_NOT_EQUAL = 6,
	MSL_SAMPLER_COMPARE_FUNC_ALWAYS = 7,
	MSL_SAMPLER_COMPARE_FUNC_INT_MAX = 0x7fffffff
};

enum MSLSamplerBorderColor
{
	MSL_SAMPLER_BORDER_COLOR_TRANSPARENT_BLACK = 0,
	MSL_SAMPLER_BORDER_COLOR_OPAQUE_BLACK = 1,
	MSL_SAMPLER_BORDER_COLOR_OPAQUE_WHITE = 2,
	MSL_SAMPLER_BORDER_COLOR_INT_MAX = 0x7fffffff
};

struct MSLConstexprSampler
{
	MSLSamplerCoord coord = MSL_SAMPLER_COORD_NORMALIZED;
	MSLSamplerFilter min_filter = MSL_SAMPLER_FILTER_NEAREST;
	MSLSamplerFilter mag_filter = MSL_SAMPLER_FILTER_NEAREST;
	MSLSamplerMipFilter mip_filter = MSL_SAMPLER_MIP_FILTER_NONE;
	MSLSamplerAddress s_address = MSL_SAMPLER_ADDRESS_CLAMP_TO_EDGE;
	MSLSamplerAddress t_address = MSL_SAMPLER_ADDRESS_CLAMP_TO_EDGE;
	MSLSamplerAddress r_address = MSL_SAMPLER_ADDRESS_CLAMP_TO_EDGE;
	MSLSamplerCompareFunc compare_func = MSL_SAMPLER_COMPARE_FUNC_NEVER;
	MSLSamplerBorderColor border_color = MSL_SAMPLER_BORDER_COLOR_TRANSPARENT_BLACK;
	float lod_clamp_min = 0.0f;
	float lod_clamp_max = 1000.0f;
	int max_anisotropy = 1;

	bool compare_enable = false;
	bool lod_clamp_enable = false;
	bool anisotropy_enable = false;
};

// Tracks the type ID and member index of a struct member
using MSLStructMemberKey = uint64_t;

// Special constant used in a MSLResourceBinding desc_set
// element to indicate the bindings for the push constants.
static const uint32_t kPushConstDescSet = ~(0u);

// Special constant used in a MSLResourceBinding binding
// element to indicate the bindings for the push constants.
static const uint32_t kPushConstBinding = 0;

// Special constant used in a MSLResourceBinding binding
// element to indicate the buffer binding for swizzle buffers.
static const uint32_t kSwizzleBufferBinding = ~(1u);

static const uint32_t kMaxArgumentBuffers = 8;

// Decompiles SPIR-V to Metal Shading Language
class CompilerMSL : public CompilerGLSL
{
public:
	// Options for compiling to Metal Shading Language
	struct Options
	{
		typedef enum
		{
			iOS = 0,
			macOS = 1
		} Platform;

		Platform platform = macOS;
		uint32_t msl_version = make_msl_version(1, 2);
		uint32_t texel_buffer_texture_width = 4096; // Width of 2D Metal textures used as 1D texel buffers
		uint32_t swizzle_buffer_index = 30;
		uint32_t indirect_params_buffer_index = 29;
		uint32_t shader_output_buffer_index = 28;
		uint32_t shader_patch_output_buffer_index = 27;
		uint32_t shader_tess_factor_buffer_index = 26;
		uint32_t shader_input_wg_index = 0;
		bool enable_point_size_builtin = true;
		bool disable_rasterization = false;
		bool capture_output_to_buffer = false;
		bool swizzle_texture_samples = false;
		bool tess_domain_origin_lower_left = false;

		// Enable use of MSL 2.0 indirect argument buffers.
		// MSL 2.0 must also be enabled.
		bool argument_buffers = false;

		// Fragment output in MSL must have at least as many components as the render pass.
		// Add support to explicit pad out components.
		bool pad_fragment_output_components = false;

		// Requires MSL 2.1, use the native support for texel buffers.
		bool texture_buffer_native = false;

		bool is_ios()
		{
			return platform == iOS;
		}

		bool is_macos()
		{
			return platform == macOS;
		}

		void set_msl_version(uint32_t major, uint32_t minor = 0, uint32_t patch = 0)
		{
			msl_version = make_msl_version(major, minor, patch);
		}

		bool supports_msl_version(uint32_t major, uint32_t minor = 0, uint32_t patch = 0)
		{
			return msl_version >= make_msl_version(major, minor, patch);
		}

		static uint32_t make_msl_version(uint32_t major, uint32_t minor = 0, uint32_t patch = 0)
		{
			return (major * 10000) + (minor * 100) + patch;
		}
	};

	const Options &get_msl_options() const
	{
		return msl_options;
	}

	void set_msl_options(const Options &opts)
	{
		msl_options = opts;
	}

	// Provide feedback to calling API to allow runtime to disable pipeline
	// rasterization if vertex shader requires rasterization to be disabled.
	bool get_is_rasterization_disabled() const
	{
		return is_rasterization_disabled && (get_entry_point().model == spv::ExecutionModelVertex ||
		                                     get_entry_point().model == spv::ExecutionModelTessellationControl ||
		                                     get_entry_point().model == spv::ExecutionModelTessellationEvaluation);
	}

	// Provide feedback to calling API to allow it to pass an auxiliary
	// swizzle buffer if the shader needs it.
	bool needs_swizzle_buffer() const
	{
		return used_swizzle_buffer;
	}

	// Provide feedback to calling API to allow it to pass an output
	// buffer if the shader needs it.
	bool needs_output_buffer() const
	{
		return capture_output_to_buffer && stage_out_var_id != 0;
	}

	// Provide feedback to calling API to allow it to pass a patch output
	// buffer if the shader needs it.
	bool needs_patch_output_buffer() const
	{
		return capture_output_to_buffer && patch_stage_out_var_id != 0;
	}

	// Provide feedback to calling API to allow it to pass an input threadgroup
	// buffer if the shader needs it.
	bool needs_input_threadgroup_mem() const
	{
		return capture_output_to_buffer && stage_in_var_id != 0;
	}

	explicit CompilerMSL(std::vector<uint32_t> spirv);
	CompilerMSL(const uint32_t *ir, size_t word_count);
	explicit CompilerMSL(const ParsedIR &ir);
	explicit CompilerMSL(ParsedIR &&ir);

	// attr is a vertex attribute binding used to match
	// vertex content locations to MSL attributes. If vertex attributes are provided,
	// is_msl_vertex_attribute_used() will return true after calling ::compile() if
	// the location was used by the MSL code.
	void add_msl_vertex_attribute(const MSLVertexAttr &attr);

	// resource is a resource binding to indicate the MSL buffer,
	// texture or sampler index to use for a particular SPIR-V description set
	// and binding. If resource bindings are provided,
	// is_msl_resource_binding_used() will return true after calling ::compile() if
	// the set/binding combination was used by the MSL code.
	void add_msl_resource_binding(const MSLResourceBinding &resource);

	// When using MSL argument buffers, we can force "classic" MSL 1.0 binding schemes for certain descriptor sets.
	// This corresponds to VK_KHR_push_descriptor in Vulkan.
	void add_discrete_descriptor_set(uint32_t desc_set);

	// Query after compilation is done. This allows you to check if a location or set/binding combination was used by the shader.
	bool is_msl_vertex_attribute_used(uint32_t location);
	bool is_msl_resource_binding_used(spv::ExecutionModel model, uint32_t set, uint32_t binding);

	// Compiles the SPIR-V code into Metal Shading Language.
	std::string compile() override;

	// Remap a sampler with ID to a constexpr sampler.
	// Older iOS targets must use constexpr samplers in certain cases (PCF),
	// so a static sampler must be used.
	// The sampler will not consume a binding, but be declared in the entry point as a constexpr sampler.
	// This can be used on both combined image/samplers (sampler2D) or standalone samplers.
	// The remapped sampler must not be an array of samplers.
	void remap_constexpr_sampler(uint32_t id, const MSLConstexprSampler &sampler);

	// If using CompilerMSL::Options::pad_fragment_output_components, override the number of components we expect
	// to use for a particular location. The default is 4 if number of components is not overridden.
	void set_fragment_output_components(uint32_t location, uint32_t components);

protected:
	// An enum of SPIR-V functions that are implemented in additional
	// source code that is added to the shader if necessary.
	enum SPVFuncImpl
	{
		SPVFuncImplNone,
		SPVFuncImplMod,
		SPVFuncImplRadians,
		SPVFuncImplDegrees,
		SPVFuncImplFindILsb,
		SPVFuncImplFindSMsb,
		SPVFuncImplFindUMsb,
		SPVFuncImplSSign,
		SPVFuncImplArrayCopyMultidimBase,
		// Unfortunately, we cannot use recursive templates in the MSL compiler properly,
		// so stamp out variants up to some arbitrary maximum.
		SPVFuncImplArrayCopy = SPVFuncImplArrayCopyMultidimBase + 1,
		SPVFuncImplArrayOfArrayCopy2Dim = SPVFuncImplArrayCopyMultidimBase + 2,
		SPVFuncImplArrayOfArrayCopy3Dim = SPVFuncImplArrayCopyMultidimBase + 3,
		SPVFuncImplArrayOfArrayCopy4Dim = SPVFuncImplArrayCopyMultidimBase + 4,
		SPVFuncImplArrayOfArrayCopy5Dim = SPVFuncImplArrayCopyMultidimBase + 5,
		SPVFuncImplArrayOfArrayCopy6Dim = SPVFuncImplArrayCopyMultidimBase + 6,
		SPVFuncImplTexelBufferCoords,
		SPVFuncImplInverse4x4,
		SPVFuncImplInverse3x3,
		SPVFuncImplInverse2x2,
		SPVFuncImplRowMajor2x3,
		SPVFuncImplRowMajor2x4,
		SPVFuncImplRowMajor3x2,
		SPVFuncImplRowMajor3x4,
		SPVFuncImplRowMajor4x2,
		SPVFuncImplRowMajor4x3,
		SPVFuncImplTextureSwizzle,
		SPVFuncImplSubgroupBallot,
		SPVFuncImplSubgroupBallotBitExtract,
		SPVFuncImplSubgroupBallotFindLSB,
		SPVFuncImplSubgroupBallotFindMSB,
		SPVFuncImplSubgroupBallotBitCount,
		SPVFuncImplSubgroupAllEqual,
		SPVFuncImplArrayCopyMultidimMax = 6
	};

	void emit_binary_unord_op(uint32_t result_type, uint32_t result_id, uint32_t op0, uint32_t op1, const char *op);
	void emit_instruction(const Instruction &instr) override;
	void emit_glsl_op(uint32_t result_type, uint32_t result_id, uint32_t op, const uint32_t *args,
	                  uint32_t count) override;
	void emit_header() override;
	void emit_function_prototype(SPIRFunction &func, const Bitset &return_flags) override;
	void emit_sampled_image_op(uint32_t result_type, uint32_t result_id, uint32_t image_id, uint32_t samp_id) override;
	void emit_subgroup_op(const Instruction &i) override;
	void emit_fixup() override;
	std::string to_struct_member(const SPIRType &type, uint32_t member_type_id, uint32_t index,
	                             const std::string &qualifier = "");
	void emit_struct_member(const SPIRType &type, uint32_t member_type_id, uint32_t index,
	                        const std::string &qualifier = "", uint32_t base_offset = 0) override;
	std::string type_to_glsl(const SPIRType &type, uint32_t id = 0) override;
	std::string image_type_glsl(const SPIRType &type, uint32_t id = 0) override;
	std::string sampler_type(const SPIRType &type);
	std::string builtin_to_glsl(spv::BuiltIn builtin, spv::StorageClass storage) override;
	size_t get_declared_struct_member_size(const SPIRType &struct_type, uint32_t index) const override;
	std::string to_func_call_arg(uint32_t id) override;
	std::string to_name(uint32_t id, bool allow_alias = true) const override;
	std::string to_function_name(uint32_t img, const SPIRType &imgtype, bool is_fetch, bool is_gather, bool is_proj,
	                             bool has_array_offsets, bool has_offset, bool has_grad, bool has_dref,
	                             uint32_t lod) override;
	std::string to_function_args(uint32_t img, const SPIRType &imgtype, bool is_fetch, bool is_gather, bool is_proj,
	                             uint32_t coord, uint32_t coord_components, uint32_t dref, uint32_t grad_x,
	                             uint32_t grad_y, uint32_t lod, uint32_t coffset, uint32_t offset, uint32_t bias,
	                             uint32_t comp, uint32_t sample, bool *p_forward) override;
	std::string to_initializer_expression(const SPIRVariable &var) override;
	std::string unpack_expression_type(std::string expr_str, const SPIRType &type, uint32_t packed_type_id) override;
	std::string bitcast_glsl_op(const SPIRType &result_type, const SPIRType &argument_type) override;
	bool skip_argument(uint32_t id) const override;
	std::string to_member_reference(uint32_t base, const SPIRType &type, uint32_t index, bool ptr_chain) override;
	std::string to_qualifiers_glsl(uint32_t id) override;
	void replace_illegal_names() override;
	void declare_undefined_values() override;
	void declare_constant_arrays();
	bool is_patch_block(const SPIRType &type);
	bool is_non_native_row_major_matrix(uint32_t id) override;
	bool member_is_non_native_row_major_matrix(const SPIRType &type, uint32_t index) override;
	std::string convert_row_major_matrix(std::string exp_str, const SPIRType &exp_type, bool is_packed) override;

	void preprocess_op_codes();
	void localize_global_variables();
	void extract_global_variables_from_functions();
	void mark_packable_structs();
	void mark_as_packable(SPIRType &type);

	std::unordered_map<uint32_t, std::set<uint32_t>> function_global_vars;
	void extract_global_variables_from_function(uint32_t func_id, std::set<uint32_t> &added_arg_ids,
	                                            std::unordered_set<uint32_t> &global_var_ids,
	                                            std::unordered_set<uint32_t> &processed_func_ids);
	uint32_t add_interface_block(spv::StorageClass storage, bool patch = false);
	uint32_t add_interface_block_pointer(uint32_t ib_var_id, spv::StorageClass storage);

	void add_variable_to_interface_block(spv::StorageClass storage, const std::string &ib_var_ref, SPIRType &ib_type,
	                                     SPIRVariable &var, bool strip_array);
	void add_composite_variable_to_interface_block(spv::StorageClass storage, const std::string &ib_var_ref,
	                                               SPIRType &ib_type, SPIRVariable &var, bool strip_array);
	void add_plain_variable_to_interface_block(spv::StorageClass storage, const std::string &ib_var_ref,
	                                           SPIRType &ib_type, SPIRVariable &var, bool strip_array);
	void add_plain_member_variable_to_interface_block(spv::StorageClass storage, const std::string &ib_var_ref,
	                                                  SPIRType &ib_type, SPIRVariable &var, uint32_t index,
	                                                  bool strip_array);
	void add_composite_member_variable_to_interface_block(spv::StorageClass storage, const std::string &ib_var_ref,
	                                                      SPIRType &ib_type, SPIRVariable &var, uint32_t index,
	                                                      bool strip_array);
	uint32_t get_accumulated_member_location(const SPIRVariable &var, uint32_t mbr_idx, bool strip_array);
	void add_tess_level_input_to_interface_block(const std::string &ib_var_ref, SPIRType &ib_type, SPIRVariable &var);

	void fix_up_interface_member_indices(spv::StorageClass storage, uint32_t ib_type_id);

	void mark_location_as_used_by_shader(uint32_t location, spv::StorageClass storage);
	uint32_t ensure_correct_builtin_type(uint32_t type_id, spv::BuiltIn builtin);
	uint32_t ensure_correct_attribute_type(uint32_t type_id, uint32_t location);

	void emit_custom_functions();
	void emit_resources();
	void emit_specialization_constants_and_structs();
	void emit_interface_block(uint32_t ib_var_id);
	bool maybe_emit_array_assignment(uint32_t id_lhs, uint32_t id_rhs);
	void add_convert_row_major_matrix_function(uint32_t cols, uint32_t rows);
	void fix_up_shader_inputs_outputs();

	std::string func_type_decl(SPIRType &type);
	std::string entry_point_args_classic(bool append_comma);
	std::string entry_point_args_argument_buffer(bool append_comma);
	std::string entry_point_arg_stage_in();
	void entry_point_args_builtin(std::string &args);
	void entry_point_args_discrete_descriptors(std::string &args);
	std::string to_qualified_member_name(const SPIRType &type, uint32_t index);
	std::string ensure_valid_name(std::string name, std::string pfx);
	std::string to_sampler_expression(uint32_t id);
	std::string to_swizzle_expression(uint32_t id);
	std::string builtin_qualifier(spv::BuiltIn builtin);
	std::string builtin_type_decl(spv::BuiltIn builtin);
	std::string built_in_func_arg(spv::BuiltIn builtin, bool prefix_comma);
	std::string member_attribute_qualifier(const SPIRType &type, uint32_t index);
	std::string argument_decl(const SPIRFunction::Parameter &arg);
	std::string round_fp_tex_coords(std::string tex_coords, bool coord_is_fp);
	uint32_t get_metal_resource_index(SPIRVariable &var, SPIRType::BaseType basetype);
	uint32_t get_ordered_member_location(uint32_t type_id, uint32_t index, uint32_t *comp = nullptr);
	size_t get_declared_struct_member_alignment(const SPIRType &struct_type, uint32_t index) const;
	std::string to_component_argument(uint32_t id);
	void align_struct(SPIRType &ib_type);
	bool is_member_packable(SPIRType &ib_type, uint32_t index);
	MSLStructMemberKey get_struct_member_key(uint32_t type_id, uint32_t index);
	std::string get_argument_address_space(const SPIRVariable &argument);
	std::string get_type_address_space(const SPIRType &type, uint32_t id);
	SPIRType &get_stage_in_struct_type();
	SPIRType &get_stage_out_struct_type();
	SPIRType &get_patch_stage_in_struct_type();
	SPIRType &get_patch_stage_out_struct_type();
	std::string get_tess_factor_struct_name();
	void emit_atomic_func_op(uint32_t result_type, uint32_t result_id, const char *op, uint32_t mem_order_1,
	                         uint32_t mem_order_2, bool has_mem_order_2, uint32_t op0, uint32_t op1 = 0,
	                         bool op1_is_pointer = false, bool op1_is_literal = false, uint32_t op2 = 0);
	const char *get_memory_order(uint32_t spv_mem_sem);
	void add_pragma_line(const std::string &line);
	void add_typedef_line(const std::string &line);
	void emit_barrier(uint32_t id_exe_scope, uint32_t id_mem_scope, uint32_t id_mem_sem);
	void emit_array_copy(const std::string &lhs, uint32_t rhs_id) override;
	void build_implicit_builtins();
	void emit_entry_point_declarations() override;
	uint32_t builtin_frag_coord_id = 0;
	uint32_t builtin_sample_id_id = 0;
	uint32_t builtin_vertex_idx_id = 0;
	uint32_t builtin_base_vertex_id = 0;
	uint32_t builtin_instance_idx_id = 0;
	uint32_t builtin_base_instance_id = 0;
	uint32_t builtin_invocation_id_id = 0;
	uint32_t builtin_primitive_id_id = 0;
	uint32_t builtin_subgroup_invocation_id_id = 0;
	uint32_t builtin_subgroup_size_id = 0;
	uint32_t swizzle_buffer_id = 0;

	void bitcast_to_builtin_store(uint32_t target_id, std::string &expr, const SPIRType &expr_type) override;
	void bitcast_from_builtin_load(uint32_t source_id, std::string &expr, const SPIRType &expr_type) override;
	void emit_store_statement(uint32_t lhs_expression, uint32_t rhs_expression) override;

	void analyze_sampled_image_usage();

	bool emit_tessellation_access_chain(const uint32_t *ops, uint32_t length);
	bool is_out_of_bounds_tessellation_level(uint32_t id_lhs);

	Options msl_options;
	std::set<SPVFuncImpl> spv_function_implementations;
	std::unordered_map<uint32_t, MSLVertexAttr> vtx_attrs_by_location;
	std::unordered_map<uint32_t, MSLVertexAttr> vtx_attrs_by_builtin;
	std::unordered_set<uint32_t> vtx_attrs_in_use;
	std::unordered_map<uint32_t, uint32_t> fragment_output_components;
	std::unordered_map<MSLStructMemberKey, uint32_t> struct_member_padding;
	std::set<std::string> pragma_lines;
	std::set<std::string> typedef_lines;
	SmallVector<uint32_t> vars_needing_early_declaration;

	SmallVector<std::pair<MSLResourceBinding, bool>> resource_bindings;
	uint32_t next_metal_resource_index_buffer = 0;
	uint32_t next_metal_resource_index_texture = 0;
	uint32_t next_metal_resource_index_sampler = 0;

	uint32_t stage_in_var_id = 0;
	uint32_t stage_out_var_id = 0;
	uint32_t patch_stage_in_var_id = 0;
	uint32_t patch_stage_out_var_id = 0;
	uint32_t stage_in_ptr_var_id = 0;
	uint32_t stage_out_ptr_var_id = 0;
	bool has_sampled_images = false;
	bool needs_vertex_idx_arg = false;
	bool needs_instance_idx_arg = false;
	bool is_rasterization_disabled = false;
	bool capture_output_to_buffer = false;
	bool needs_swizzle_buffer_def = false;
	bool used_swizzle_buffer = false;
	bool added_builtin_tess_level = false;
	bool needs_subgroup_invocation_id = false;
	std::string qual_pos_var_name;
	std::string stage_in_var_name = "in";
	std::string stage_out_var_name = "out";
	std::string patch_stage_in_var_name = "patchIn";
	std::string patch_stage_out_var_name = "patchOut";
	std::string sampler_name_suffix = "Smplr";
	std::string swizzle_name_suffix = "Swzl";
	std::string input_wg_var_name = "gl_in";
	std::string output_buffer_var_name = "spvOut";
	std::string patch_output_buffer_var_name = "spvPatchOut";
	std::string tess_factor_buffer_var_name = "spvTessLevel";
	spv::Op previous_instruction_opcode = spv::OpNop;

	std::unordered_map<uint32_t, MSLConstexprSampler> constexpr_samplers;
	SmallVector<uint32_t> buffer_arrays;

	uint32_t argument_buffer_ids[kMaxArgumentBuffers];
	uint32_t argument_buffer_discrete_mask = 0;
	void analyze_argument_buffers();
	bool descriptor_set_is_argument_buffer(uint32_t desc_set) const;

	uint32_t get_target_components_for_fragment_location(uint32_t location) const;
	uint32_t build_extended_vector_type(uint32_t type_id, uint32_t components);

	bool suppress_missing_prototypes = false;

	// OpcodeHandler that handles several MSL preprocessing operations.
	struct OpCodePreprocessor : OpcodeHandler
	{
		OpCodePreprocessor(CompilerMSL &compiler_)
		    : compiler(compiler_)
		{
		}

		bool handle(spv::Op opcode, const uint32_t *args, uint32_t length) override;
		CompilerMSL::SPVFuncImpl get_spv_func_impl(spv::Op opcode, const uint32_t *args);
		void check_resource_write(uint32_t var_id);

		CompilerMSL &compiler;
		std::unordered_map<uint32_t, uint32_t> result_types;
		bool suppress_missing_prototypes = false;
		bool uses_atomics = false;
		bool uses_resource_write = false;
		bool needs_subgroup_invocation_id = false;
	};

	// OpcodeHandler that scans for uses of sampled images
	struct SampledImageScanner : OpcodeHandler
	{
		SampledImageScanner(CompilerMSL &compiler_)
		    : compiler(compiler_)
		{
		}

		bool handle(spv::Op opcode, const uint32_t *args, uint32_t) override;

		CompilerMSL &compiler;
	};

	// Sorts the members of a SPIRType and associated Meta info based on a settable sorting
	// aspect, which defines which aspect of the struct members will be used to sort them.
	// Regardless of the sorting aspect, built-in members always appear at the end of the struct.
	struct MemberSorter
	{
		enum SortAspect
		{
			Location,
			LocationReverse,
			Offset,
			OffsetThenLocationReverse,
			Alphabetical
		};

		void sort();
		bool operator()(uint32_t mbr_idx1, uint32_t mbr_idx2);
		MemberSorter(SPIRType &t, Meta &m, SortAspect sa);

		SPIRType &type;
		Meta &meta;
		SortAspect sort_aspect;
	};
};
} // namespace SPIRV_CROSS_NAMESPACE

#endif
