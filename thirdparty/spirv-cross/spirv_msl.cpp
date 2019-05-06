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

#include "spirv_msl.hpp"
#include "GLSL.std.450.h"

#include <algorithm>
#include <assert.h>
#include <numeric>

using namespace spv;
using namespace SPIRV_CROSS_NAMESPACE;
using namespace std;

static const uint32_t k_unknown_location = ~0u;
static const uint32_t k_unknown_component = ~0u;

CompilerMSL::CompilerMSL(std::vector<uint32_t> spirv_)
    : CompilerGLSL(move(spirv_))
{
}

CompilerMSL::CompilerMSL(const uint32_t *ir_, size_t word_count)
    : CompilerGLSL(ir_, word_count)
{
}

CompilerMSL::CompilerMSL(const ParsedIR &ir_)
    : CompilerGLSL(ir_)
{
}

CompilerMSL::CompilerMSL(ParsedIR &&ir_)
    : CompilerGLSL(std::move(ir_))
{
}

void CompilerMSL::add_msl_vertex_attribute(const MSLVertexAttr &va)
{
	vtx_attrs_by_location[va.location] = va;
	if (va.builtin != BuiltInMax && !vtx_attrs_by_builtin.count(va.builtin))
		vtx_attrs_by_builtin[va.builtin] = va;
}

void CompilerMSL::add_msl_resource_binding(const MSLResourceBinding &binding)
{
	resource_bindings.push_back({ binding, false });
}

void CompilerMSL::add_discrete_descriptor_set(uint32_t desc_set)
{
	if (desc_set < kMaxArgumentBuffers)
		argument_buffer_discrete_mask |= 1u << desc_set;
}

bool CompilerMSL::is_msl_vertex_attribute_used(uint32_t location)
{
	return vtx_attrs_in_use.count(location) != 0;
}

bool CompilerMSL::is_msl_resource_binding_used(ExecutionModel model, uint32_t desc_set, uint32_t binding)
{
	auto itr = find_if(begin(resource_bindings), end(resource_bindings),
	                   [&](const std::pair<MSLResourceBinding, bool> &resource) -> bool {
		                   return model == resource.first.stage && desc_set == resource.first.desc_set &&
		                          binding == resource.first.binding;
	                   });
	return itr != end(resource_bindings) && itr->second;
}

void CompilerMSL::set_fragment_output_components(uint32_t location, uint32_t components)
{
	fragment_output_components[location] = components;
}

void CompilerMSL::build_implicit_builtins()
{
	bool need_sample_pos = active_input_builtins.get(BuiltInSamplePosition);
	bool need_vertex_params = capture_output_to_buffer && get_execution_model() == ExecutionModelVertex;
	bool need_tesc_params = get_execution_model() == ExecutionModelTessellationControl;
	bool need_subgroup_mask =
	    active_input_builtins.get(BuiltInSubgroupEqMask) || active_input_builtins.get(BuiltInSubgroupGeMask) ||
	    active_input_builtins.get(BuiltInSubgroupGtMask) || active_input_builtins.get(BuiltInSubgroupLeMask) ||
	    active_input_builtins.get(BuiltInSubgroupLtMask);
	bool need_subgroup_ge_mask = !msl_options.is_ios() && (active_input_builtins.get(BuiltInSubgroupGeMask) ||
	                                                       active_input_builtins.get(BuiltInSubgroupGtMask));
	if (need_subpass_input || need_sample_pos || need_subgroup_mask || need_vertex_params || need_tesc_params ||
	    needs_subgroup_invocation_id)
	{
		bool has_frag_coord = false;
		bool has_sample_id = false;
		bool has_vertex_idx = false;
		bool has_base_vertex = false;
		bool has_instance_idx = false;
		bool has_base_instance = false;
		bool has_invocation_id = false;
		bool has_primitive_id = false;
		bool has_subgroup_invocation_id = false;
		bool has_subgroup_size = false;

		ir.for_each_typed_id<SPIRVariable>([&](uint32_t, SPIRVariable &var) {
			if (var.storage != StorageClassInput || !ir.meta[var.self].decoration.builtin)
				return;

			BuiltIn builtin = ir.meta[var.self].decoration.builtin_type;
			if (need_subpass_input && builtin == BuiltInFragCoord)
			{
				builtin_frag_coord_id = var.self;
				has_frag_coord = true;
			}

			if (need_sample_pos && builtin == BuiltInSampleId)
			{
				builtin_sample_id_id = var.self;
				has_sample_id = true;
			}

			if (need_vertex_params)
			{
				switch (builtin)
				{
				case BuiltInVertexIndex:
					builtin_vertex_idx_id = var.self;
					has_vertex_idx = true;
					break;
				case BuiltInBaseVertex:
					builtin_base_vertex_id = var.self;
					has_base_vertex = true;
					break;
				case BuiltInInstanceIndex:
					builtin_instance_idx_id = var.self;
					has_instance_idx = true;
					break;
				case BuiltInBaseInstance:
					builtin_base_instance_id = var.self;
					has_base_instance = true;
					break;
				default:
					break;
				}
			}

			if (need_tesc_params)
			{
				switch (builtin)
				{
				case BuiltInInvocationId:
					builtin_invocation_id_id = var.self;
					has_invocation_id = true;
					break;
				case BuiltInPrimitiveId:
					builtin_primitive_id_id = var.self;
					has_primitive_id = true;
					break;
				default:
					break;
				}
			}

			if ((need_subgroup_mask || needs_subgroup_invocation_id) && builtin == BuiltInSubgroupLocalInvocationId)
			{
				builtin_subgroup_invocation_id_id = var.self;
				has_subgroup_invocation_id = true;
			}

			if (need_subgroup_ge_mask && builtin == BuiltInSubgroupSize)
			{
				builtin_subgroup_size_id = var.self;
				has_subgroup_size = true;
			}
		});

		if (!has_frag_coord && need_subpass_input)
		{
			uint32_t offset = ir.increase_bound_by(3);
			uint32_t type_id = offset;
			uint32_t type_ptr_id = offset + 1;
			uint32_t var_id = offset + 2;

			// Create gl_FragCoord.
			SPIRType vec4_type;
			vec4_type.basetype = SPIRType::Float;
			vec4_type.width = 32;
			vec4_type.vecsize = 4;
			set<SPIRType>(type_id, vec4_type);

			SPIRType vec4_type_ptr;
			vec4_type_ptr = vec4_type;
			vec4_type_ptr.pointer = true;
			vec4_type_ptr.parent_type = type_id;
			vec4_type_ptr.storage = StorageClassInput;
			auto &ptr_type = set<SPIRType>(type_ptr_id, vec4_type_ptr);
			ptr_type.self = type_id;

			set<SPIRVariable>(var_id, type_ptr_id, StorageClassInput);
			set_decoration(var_id, DecorationBuiltIn, BuiltInFragCoord);
			builtin_frag_coord_id = var_id;
		}

		if (!has_sample_id && need_sample_pos)
		{
			uint32_t offset = ir.increase_bound_by(3);
			uint32_t type_id = offset;
			uint32_t type_ptr_id = offset + 1;
			uint32_t var_id = offset + 2;

			// Create gl_SampleID.
			SPIRType uint_type;
			uint_type.basetype = SPIRType::UInt;
			uint_type.width = 32;
			set<SPIRType>(type_id, uint_type);

			SPIRType uint_type_ptr;
			uint_type_ptr = uint_type;
			uint_type_ptr.pointer = true;
			uint_type_ptr.parent_type = type_id;
			uint_type_ptr.storage = StorageClassInput;
			auto &ptr_type = set<SPIRType>(type_ptr_id, uint_type_ptr);
			ptr_type.self = type_id;

			set<SPIRVariable>(var_id, type_ptr_id, StorageClassInput);
			set_decoration(var_id, DecorationBuiltIn, BuiltInSampleId);
			builtin_sample_id_id = var_id;
		}

		if (need_vertex_params && (!has_vertex_idx || !has_base_vertex || !has_instance_idx || !has_base_instance))
		{
			uint32_t offset = ir.increase_bound_by(2);
			uint32_t type_id = offset;
			uint32_t type_ptr_id = offset + 1;

			SPIRType uint_type;
			uint_type.basetype = SPIRType::UInt;
			uint_type.width = 32;
			set<SPIRType>(type_id, uint_type);

			SPIRType uint_type_ptr;
			uint_type_ptr = uint_type;
			uint_type_ptr.pointer = true;
			uint_type_ptr.parent_type = type_id;
			uint_type_ptr.storage = StorageClassInput;
			auto &ptr_type = set<SPIRType>(type_ptr_id, uint_type_ptr);
			ptr_type.self = type_id;

			if (!has_vertex_idx)
			{
				uint32_t var_id = ir.increase_bound_by(1);

				// Create gl_VertexIndex.
				set<SPIRVariable>(var_id, type_ptr_id, StorageClassInput);
				set_decoration(var_id, DecorationBuiltIn, BuiltInVertexIndex);
				builtin_vertex_idx_id = var_id;
			}
			if (!has_base_vertex)
			{
				uint32_t var_id = ir.increase_bound_by(1);

				// Create gl_BaseVertex.
				set<SPIRVariable>(var_id, type_ptr_id, StorageClassInput);
				set_decoration(var_id, DecorationBuiltIn, BuiltInBaseVertex);
				builtin_base_vertex_id = var_id;
			}
			if (!has_instance_idx)
			{
				uint32_t var_id = ir.increase_bound_by(1);

				// Create gl_InstanceIndex.
				set<SPIRVariable>(var_id, type_ptr_id, StorageClassInput);
				set_decoration(var_id, DecorationBuiltIn, BuiltInInstanceIndex);
				builtin_instance_idx_id = var_id;
			}
			if (!has_base_instance)
			{
				uint32_t var_id = ir.increase_bound_by(1);

				// Create gl_BaseInstance.
				set<SPIRVariable>(var_id, type_ptr_id, StorageClassInput);
				set_decoration(var_id, DecorationBuiltIn, BuiltInBaseInstance);
				builtin_base_instance_id = var_id;
			}
		}

		if (need_tesc_params && (!has_invocation_id || !has_primitive_id))
		{
			uint32_t offset = ir.increase_bound_by(2);
			uint32_t type_id = offset;
			uint32_t type_ptr_id = offset + 1;

			SPIRType uint_type;
			uint_type.basetype = SPIRType::UInt;
			uint_type.width = 32;
			set<SPIRType>(type_id, uint_type);

			SPIRType uint_type_ptr;
			uint_type_ptr = uint_type;
			uint_type_ptr.pointer = true;
			uint_type_ptr.parent_type = type_id;
			uint_type_ptr.storage = StorageClassInput;
			auto &ptr_type = set<SPIRType>(type_ptr_id, uint_type_ptr);
			ptr_type.self = type_id;

			if (!has_invocation_id)
			{
				uint32_t var_id = ir.increase_bound_by(1);

				// Create gl_InvocationID.
				set<SPIRVariable>(var_id, type_ptr_id, StorageClassInput);
				set_decoration(var_id, DecorationBuiltIn, BuiltInInvocationId);
				builtin_invocation_id_id = var_id;
			}
			if (!has_primitive_id)
			{
				uint32_t var_id = ir.increase_bound_by(1);

				// Create gl_PrimitiveID.
				set<SPIRVariable>(var_id, type_ptr_id, StorageClassInput);
				set_decoration(var_id, DecorationBuiltIn, BuiltInPrimitiveId);
				builtin_primitive_id_id = var_id;
			}
		}

		if (!has_subgroup_invocation_id && (need_subgroup_mask || needs_subgroup_invocation_id))
		{
			uint32_t offset = ir.increase_bound_by(3);
			uint32_t type_id = offset;
			uint32_t type_ptr_id = offset + 1;
			uint32_t var_id = offset + 2;

			// Create gl_SubgroupInvocationID.
			SPIRType uint_type;
			uint_type.basetype = SPIRType::UInt;
			uint_type.width = 32;
			set<SPIRType>(type_id, uint_type);

			SPIRType uint_type_ptr;
			uint_type_ptr = uint_type;
			uint_type_ptr.pointer = true;
			uint_type_ptr.parent_type = type_id;
			uint_type_ptr.storage = StorageClassInput;
			auto &ptr_type = set<SPIRType>(type_ptr_id, uint_type_ptr);
			ptr_type.self = type_id;

			set<SPIRVariable>(var_id, type_ptr_id, StorageClassInput);
			set_decoration(var_id, DecorationBuiltIn, BuiltInSubgroupLocalInvocationId);
			builtin_subgroup_invocation_id_id = var_id;
		}

		if (!has_subgroup_size && need_subgroup_ge_mask)
		{
			uint32_t offset = ir.increase_bound_by(3);
			uint32_t type_id = offset;
			uint32_t type_ptr_id = offset + 1;
			uint32_t var_id = offset + 2;

			// Create gl_SubgroupSize.
			SPIRType uint_type;
			uint_type.basetype = SPIRType::UInt;
			uint_type.width = 32;
			set<SPIRType>(type_id, uint_type);

			SPIRType uint_type_ptr;
			uint_type_ptr = uint_type;
			uint_type_ptr.pointer = true;
			uint_type_ptr.parent_type = type_id;
			uint_type_ptr.storage = StorageClassInput;
			auto &ptr_type = set<SPIRType>(type_ptr_id, uint_type_ptr);
			ptr_type.self = type_id;

			set<SPIRVariable>(var_id, type_ptr_id, StorageClassInput);
			set_decoration(var_id, DecorationBuiltIn, BuiltInSubgroupSize);
			builtin_subgroup_size_id = var_id;
		}
	}

	if (needs_swizzle_buffer_def)
	{
		uint32_t offset = ir.increase_bound_by(4);
		uint32_t type_id = offset;
		uint32_t type_ptr_id = offset + 1;
		uint32_t type_ptr_ptr_id = offset + 2;
		uint32_t var_id = offset + 3;

		// Create a buffer to hold extra data, including the swizzle constants.
		SPIRType uint_type;
		uint_type.basetype = SPIRType::UInt;
		uint_type.width = 32;
		set<SPIRType>(type_id, uint_type);

		SPIRType uint_type_pointer = uint_type;
		uint_type_pointer.pointer = true;
		uint_type_pointer.pointer_depth = 1;
		uint_type_pointer.parent_type = type_id;
		uint_type_pointer.storage = StorageClassUniform;
		set<SPIRType>(type_ptr_id, uint_type_pointer);
		set_decoration(type_ptr_id, DecorationArrayStride, 4);

		SPIRType uint_type_pointer2 = uint_type_pointer;
		uint_type_pointer2.pointer_depth++;
		uint_type_pointer2.parent_type = type_ptr_id;
		set<SPIRType>(type_ptr_ptr_id, uint_type_pointer2);

		set<SPIRVariable>(var_id, type_ptr_ptr_id, StorageClassUniformConstant);
		set_name(var_id, "spvSwizzleConstants");
		// This should never match anything.
		set_decoration(var_id, DecorationDescriptorSet, 0xFFFFFFFE);
		set_decoration(var_id, DecorationBinding, msl_options.swizzle_buffer_index);
		swizzle_buffer_id = var_id;
	}
}

static string create_sampler_address(const char *prefix, MSLSamplerAddress addr)
{
	switch (addr)
	{
	case MSL_SAMPLER_ADDRESS_CLAMP_TO_EDGE:
		return join(prefix, "address::clamp_to_edge");
	case MSL_SAMPLER_ADDRESS_CLAMP_TO_ZERO:
		return join(prefix, "address::clamp_to_zero");
	case MSL_SAMPLER_ADDRESS_CLAMP_TO_BORDER:
		return join(prefix, "address::clamp_to_border");
	case MSL_SAMPLER_ADDRESS_REPEAT:
		return join(prefix, "address::repeat");
	case MSL_SAMPLER_ADDRESS_MIRRORED_REPEAT:
		return join(prefix, "address::mirrored_repeat");
	default:
		SPIRV_CROSS_THROW("Invalid sampler addressing mode.");
	}
}

SPIRType &CompilerMSL::get_stage_in_struct_type()
{
	auto &si_var = get<SPIRVariable>(stage_in_var_id);
	return get_variable_data_type(si_var);
}

SPIRType &CompilerMSL::get_stage_out_struct_type()
{
	auto &so_var = get<SPIRVariable>(stage_out_var_id);
	return get_variable_data_type(so_var);
}

SPIRType &CompilerMSL::get_patch_stage_in_struct_type()
{
	auto &si_var = get<SPIRVariable>(patch_stage_in_var_id);
	return get_variable_data_type(si_var);
}

SPIRType &CompilerMSL::get_patch_stage_out_struct_type()
{
	auto &so_var = get<SPIRVariable>(patch_stage_out_var_id);
	return get_variable_data_type(so_var);
}

std::string CompilerMSL::get_tess_factor_struct_name()
{
	if (get_entry_point().flags.get(ExecutionModeTriangles))
		return "MTLTriangleTessellationFactorsHalf";
	return "MTLQuadTessellationFactorsHalf";
}

void CompilerMSL::emit_entry_point_declarations()
{
	// FIXME: Get test coverage here ...

	// Emit constexpr samplers here.
	for (auto &samp : constexpr_samplers)
	{
		auto &var = get<SPIRVariable>(samp.first);
		auto &type = get<SPIRType>(var.basetype);
		if (type.basetype == SPIRType::Sampler)
			add_resource_name(samp.first);

		SmallVector<string> args;
		auto &s = samp.second;

		if (s.coord != MSL_SAMPLER_COORD_NORMALIZED)
			args.push_back("coord::pixel");

		if (s.min_filter == s.mag_filter)
		{
			if (s.min_filter != MSL_SAMPLER_FILTER_NEAREST)
				args.push_back("filter::linear");
		}
		else
		{
			if (s.min_filter != MSL_SAMPLER_FILTER_NEAREST)
				args.push_back("min_filter::linear");
			if (s.mag_filter != MSL_SAMPLER_FILTER_NEAREST)
				args.push_back("mag_filter::linear");
		}

		switch (s.mip_filter)
		{
		case MSL_SAMPLER_MIP_FILTER_NONE:
			// Default
			break;
		case MSL_SAMPLER_MIP_FILTER_NEAREST:
			args.push_back("mip_filter::nearest");
			break;
		case MSL_SAMPLER_MIP_FILTER_LINEAR:
			args.push_back("mip_filter::linear");
			break;
		default:
			SPIRV_CROSS_THROW("Invalid mip filter.");
		}

		if (s.s_address == s.t_address && s.s_address == s.r_address)
		{
			if (s.s_address != MSL_SAMPLER_ADDRESS_CLAMP_TO_EDGE)
				args.push_back(create_sampler_address("", s.s_address));
		}
		else
		{
			if (s.s_address != MSL_SAMPLER_ADDRESS_CLAMP_TO_EDGE)
				args.push_back(create_sampler_address("s_", s.s_address));
			if (s.t_address != MSL_SAMPLER_ADDRESS_CLAMP_TO_EDGE)
				args.push_back(create_sampler_address("t_", s.t_address));
			if (s.r_address != MSL_SAMPLER_ADDRESS_CLAMP_TO_EDGE)
				args.push_back(create_sampler_address("r_", s.r_address));
		}

		if (s.compare_enable)
		{
			switch (s.compare_func)
			{
			case MSL_SAMPLER_COMPARE_FUNC_ALWAYS:
				args.push_back("compare_func::always");
				break;
			case MSL_SAMPLER_COMPARE_FUNC_NEVER:
				args.push_back("compare_func::never");
				break;
			case MSL_SAMPLER_COMPARE_FUNC_EQUAL:
				args.push_back("compare_func::equal");
				break;
			case MSL_SAMPLER_COMPARE_FUNC_NOT_EQUAL:
				args.push_back("compare_func::not_equal");
				break;
			case MSL_SAMPLER_COMPARE_FUNC_LESS:
				args.push_back("compare_func::less");
				break;
			case MSL_SAMPLER_COMPARE_FUNC_LESS_EQUAL:
				args.push_back("compare_func::less_equal");
				break;
			case MSL_SAMPLER_COMPARE_FUNC_GREATER:
				args.push_back("compare_func::greater");
				break;
			case MSL_SAMPLER_COMPARE_FUNC_GREATER_EQUAL:
				args.push_back("compare_func::greater_equal");
				break;
			default:
				SPIRV_CROSS_THROW("Invalid sampler compare function.");
			}
		}

		if (s.s_address == MSL_SAMPLER_ADDRESS_CLAMP_TO_BORDER || s.t_address == MSL_SAMPLER_ADDRESS_CLAMP_TO_BORDER ||
		    s.r_address == MSL_SAMPLER_ADDRESS_CLAMP_TO_BORDER)
		{
			switch (s.border_color)
			{
			case MSL_SAMPLER_BORDER_COLOR_OPAQUE_BLACK:
				args.push_back("border_color::opaque_black");
				break;
			case MSL_SAMPLER_BORDER_COLOR_OPAQUE_WHITE:
				args.push_back("border_color::opaque_white");
				break;
			case MSL_SAMPLER_BORDER_COLOR_TRANSPARENT_BLACK:
				args.push_back("border_color::transparent_black");
				break;
			default:
				SPIRV_CROSS_THROW("Invalid sampler border color.");
			}
		}

		if (s.anisotropy_enable)
			args.push_back(join("max_anisotropy(", s.max_anisotropy, ")"));
		if (s.lod_clamp_enable)
		{
			args.push_back(join("lod_clamp(", convert_to_string(s.lod_clamp_min, current_locale_radix_character), ", ",
			                    convert_to_string(s.lod_clamp_max, current_locale_radix_character), ")"));
		}

		statement("constexpr sampler ",
		          type.basetype == SPIRType::SampledImage ? to_sampler_expression(samp.first) : to_name(samp.first),
		          "(", merge(args), ");");
	}

	// Emit buffer arrays here.
	for (uint32_t array_id : buffer_arrays)
	{
		const auto &var = get<SPIRVariable>(array_id);
		const auto &type = get_variable_data_type(var);
		string name = to_name(array_id);
		statement(get_argument_address_space(var) + " " + type_to_glsl(type) + "* " + name + "[] =");
		begin_scope();
		for (uint32_t i = 0; i < type.array[0]; ++i)
			statement(name + "_" + convert_to_string(i) + ",");
		end_scope_decl();
		statement_no_indent("");
	}
	// For some reason, without this, we end up emitting the arrays twice.
	buffer_arrays.clear();
}

string CompilerMSL::compile()
{
	// Do not deal with GLES-isms like precision, older extensions and such.
	options.vulkan_semantics = true;
	options.es = false;
	options.version = 450;
	backend.null_pointer_literal = "nullptr";
	backend.float_literal_suffix = false;
	backend.uint32_t_literal_suffix = true;
	backend.int16_t_literal_suffix = "";
	backend.uint16_t_literal_suffix = "u";
	backend.basic_int_type = "int";
	backend.basic_uint_type = "uint";
	backend.basic_int8_type = "char";
	backend.basic_uint8_type = "uchar";
	backend.basic_int16_type = "short";
	backend.basic_uint16_type = "ushort";
	backend.discard_literal = "discard_fragment()";
	backend.swizzle_is_function = false;
	backend.shared_is_implied = false;
	backend.use_initializer_list = true;
	backend.use_typed_initializer_list = true;
	backend.native_row_major_matrix = false;
	backend.unsized_array_supported = false;
	backend.can_declare_arrays_inline = false;
	backend.can_return_array = false;
	backend.boolean_mix_support = false;
	backend.allow_truncated_access_chain = true;
	backend.array_is_value_type = false;
	backend.comparison_image_samples_scalar = true;
	backend.native_pointers = true;
	backend.nonuniform_qualifier = "";

	capture_output_to_buffer = msl_options.capture_output_to_buffer;
	is_rasterization_disabled = msl_options.disable_rasterization || capture_output_to_buffer;

	fixup_type_alias();
	replace_illegal_names();

	struct_member_padding.clear();

	build_function_control_flow_graphs_and_analyze();
	update_active_builtins();
	analyze_image_and_sampler_usage();
	analyze_sampled_image_usage();
	preprocess_op_codes();
	build_implicit_builtins();

	fixup_image_load_store_access();

	set_enabled_interface_variables(get_active_interface_variables());
	if (swizzle_buffer_id)
		active_interface_variables.insert(swizzle_buffer_id);

	// Create structs to hold input, output and uniform variables.
	// Do output first to ensure out. is declared at top of entry function.
	qual_pos_var_name = "";
	stage_out_var_id = add_interface_block(StorageClassOutput);
	patch_stage_out_var_id = add_interface_block(StorageClassOutput, true);
	stage_in_var_id = add_interface_block(StorageClassInput);
	if (get_execution_model() == ExecutionModelTessellationEvaluation)
		patch_stage_in_var_id = add_interface_block(StorageClassInput, true);

	if (get_execution_model() == ExecutionModelTessellationControl)
		stage_out_ptr_var_id = add_interface_block_pointer(stage_out_var_id, StorageClassOutput);
	if (is_tessellation_shader())
		stage_in_ptr_var_id = add_interface_block_pointer(stage_in_var_id, StorageClassInput);

	// Metal vertex functions that define no output must disable rasterization and return void.
	if (!stage_out_var_id)
		is_rasterization_disabled = true;

	// Convert the use of global variables to recursively-passed function parameters
	localize_global_variables();
	extract_global_variables_from_functions();

	// Mark any non-stage-in structs to be tightly packed.
	mark_packable_structs();
	reorder_type_alias();

	// Add fixup hooks required by shader inputs and outputs. This needs to happen before
	// the loop, so the hooks aren't added multiple times.
	fix_up_shader_inputs_outputs();

	// If we are using argument buffers, we create argument buffer structures for them here.
	// These buffers will be used in the entry point, not the individual resources.
	if (msl_options.argument_buffers)
	{
		if (!msl_options.supports_msl_version(2, 0))
			SPIRV_CROSS_THROW("Argument buffers can only be used with MSL 2.0 and up.");
		analyze_argument_buffers();
	}

	uint32_t pass_count = 0;
	do
	{
		if (pass_count >= 3)
			SPIRV_CROSS_THROW("Over 3 compilation loops detected. Must be a bug!");

		reset();

		// Start bindings at zero.
		next_metal_resource_index_buffer = 0;
		next_metal_resource_index_texture = 0;
		next_metal_resource_index_sampler = 0;

		// Move constructor for this type is broken on GCC 4.9 ...
		buffer.reset();

		emit_header();
		emit_specialization_constants_and_structs();
		emit_resources();
		emit_custom_functions();
		emit_function(get<SPIRFunction>(ir.default_entry_point), Bitset());

		pass_count++;
	} while (is_forcing_recompilation());

	return buffer.str();
}

// Register the need to output any custom functions.
void CompilerMSL::preprocess_op_codes()
{
	OpCodePreprocessor preproc(*this);
	traverse_all_reachable_opcodes(get<SPIRFunction>(ir.default_entry_point), preproc);

	suppress_missing_prototypes = preproc.suppress_missing_prototypes;

	if (preproc.uses_atomics)
	{
		add_header_line("#include <metal_atomic>");
		add_pragma_line("#pragma clang diagnostic ignored \"-Wunused-variable\"");
	}

	// Metal vertex functions that write to resources must disable rasterization and return void.
	if (preproc.uses_resource_write)
		is_rasterization_disabled = true;

	// Tessellation control shaders are run as compute functions in Metal, and so
	// must capture their output to a buffer.
	if (get_execution_model() == ExecutionModelTessellationControl)
	{
		is_rasterization_disabled = true;
		capture_output_to_buffer = true;
	}

	if (preproc.needs_subgroup_invocation_id)
		needs_subgroup_invocation_id = true;
}

// Move the Private and Workgroup global variables to the entry function.
// Non-constant variables cannot have global scope in Metal.
void CompilerMSL::localize_global_variables()
{
	auto &entry_func = get<SPIRFunction>(ir.default_entry_point);
	auto iter = global_variables.begin();
	while (iter != global_variables.end())
	{
		uint32_t v_id = *iter;
		auto &var = get<SPIRVariable>(v_id);
		if (var.storage == StorageClassPrivate || var.storage == StorageClassWorkgroup)
		{
			if (!variable_is_lut(var))
				entry_func.add_local_variable(v_id);
			iter = global_variables.erase(iter);
		}
		else
			iter++;
	}
}

// For any global variable accessed directly by a function,
// extract that variable and add it as an argument to that function.
void CompilerMSL::extract_global_variables_from_functions()
{
	// Uniforms
	unordered_set<uint32_t> global_var_ids;
	ir.for_each_typed_id<SPIRVariable>([&](uint32_t, SPIRVariable &var) {
		if (var.storage == StorageClassInput || var.storage == StorageClassOutput ||
		    var.storage == StorageClassUniform || var.storage == StorageClassUniformConstant ||
		    var.storage == StorageClassPushConstant || var.storage == StorageClassStorageBuffer)
		{
			global_var_ids.insert(var.self);
		}
	});

	// Local vars that are declared in the main function and accessed directly by a function
	auto &entry_func = get<SPIRFunction>(ir.default_entry_point);
	for (auto &var : entry_func.local_variables)
		if (get<SPIRVariable>(var).storage != StorageClassFunction)
			global_var_ids.insert(var);

	std::set<uint32_t> added_arg_ids;
	unordered_set<uint32_t> processed_func_ids;
	extract_global_variables_from_function(ir.default_entry_point, added_arg_ids, global_var_ids, processed_func_ids);
}

// MSL does not support the use of global variables for shader input content.
// For any global variable accessed directly by the specified function, extract that variable,
// add it as an argument to that function, and the arg to the added_arg_ids collection.
void CompilerMSL::extract_global_variables_from_function(uint32_t func_id, std::set<uint32_t> &added_arg_ids,
                                                         unordered_set<uint32_t> &global_var_ids,
                                                         unordered_set<uint32_t> &processed_func_ids)
{
	// Avoid processing a function more than once
	if (processed_func_ids.find(func_id) != processed_func_ids.end())
	{
		// Return function global variables
		added_arg_ids = function_global_vars[func_id];
		return;
	}

	processed_func_ids.insert(func_id);

	auto &func = get<SPIRFunction>(func_id);

	// Recursively establish global args added to functions on which we depend.
	for (auto block : func.blocks)
	{
		auto &b = get<SPIRBlock>(block);
		for (auto &i : b.ops)
		{
			auto ops = stream(i);
			auto op = static_cast<Op>(i.op);

			switch (op)
			{
			case OpLoad:
			case OpInBoundsAccessChain:
			case OpAccessChain:
			case OpPtrAccessChain:
			{
				uint32_t base_id = ops[2];
				if (global_var_ids.find(base_id) != global_var_ids.end())
					added_arg_ids.insert(base_id);

				auto &type = get<SPIRType>(ops[0]);
				if (type.basetype == SPIRType::Image && type.image.dim == DimSubpassData)
				{
					// Implicitly reads gl_FragCoord.
					assert(builtin_frag_coord_id != 0);
					added_arg_ids.insert(builtin_frag_coord_id);
				}

				break;
			}

			case OpFunctionCall:
			{
				// First see if any of the function call args are globals
				for (uint32_t arg_idx = 3; arg_idx < i.length; arg_idx++)
				{
					uint32_t arg_id = ops[arg_idx];
					if (global_var_ids.find(arg_id) != global_var_ids.end())
						added_arg_ids.insert(arg_id);
				}

				// Then recurse into the function itself to extract globals used internally in the function
				uint32_t inner_func_id = ops[2];
				std::set<uint32_t> inner_func_args;
				extract_global_variables_from_function(inner_func_id, inner_func_args, global_var_ids,
				                                       processed_func_ids);
				added_arg_ids.insert(inner_func_args.begin(), inner_func_args.end());
				break;
			}

			case OpStore:
			{
				uint32_t base_id = ops[0];
				if (global_var_ids.find(base_id) != global_var_ids.end())
					added_arg_ids.insert(base_id);
				break;
			}

			case OpSelect:
			{
				uint32_t base_id = ops[3];
				if (global_var_ids.find(base_id) != global_var_ids.end())
					added_arg_ids.insert(base_id);
				base_id = ops[4];
				if (global_var_ids.find(base_id) != global_var_ids.end())
					added_arg_ids.insert(base_id);
				break;
			}

			default:
				break;
			}

			// TODO: Add all other operations which can affect memory.
			// We should consider a more unified system here to reduce boiler-plate.
			// This kind of analysis is done in several places ...
		}
	}

	function_global_vars[func_id] = added_arg_ids;

	// Add the global variables as arguments to the function
	if (func_id != ir.default_entry_point)
	{
		bool added_in = false;
		bool added_out = false;
		for (uint32_t arg_id : added_arg_ids)
		{
			auto &var = get<SPIRVariable>(arg_id);
			uint32_t type_id = var.basetype;
			auto *p_type = &get<SPIRType>(type_id);
			BuiltIn bi_type = BuiltIn(get_decoration(arg_id, DecorationBuiltIn));

			if (((is_tessellation_shader() && var.storage == StorageClassInput) ||
			     (get_execution_model() == ExecutionModelTessellationControl && var.storage == StorageClassOutput)) &&
			    !(has_decoration(arg_id, DecorationPatch) || is_patch_block(*p_type)) &&
			    (!is_builtin_variable(var) || bi_type == BuiltInPosition || bi_type == BuiltInPointSize ||
			     bi_type == BuiltInClipDistance || bi_type == BuiltInCullDistance ||
			     p_type->basetype == SPIRType::Struct))
			{
				// Tessellation control shaders see inputs and per-vertex outputs as arrays.
				// Similarly, tessellation evaluation shaders see per-vertex inputs as arrays.
				// We collected them into a structure; we must pass the array of this
				// structure to the function.
				std::string name;
				if (var.storage == StorageClassInput)
				{
					if (added_in)
						continue;
					name = input_wg_var_name;
					arg_id = stage_in_ptr_var_id;
					added_in = true;
				}
				else if (var.storage == StorageClassOutput)
				{
					if (added_out)
						continue;
					name = "gl_out";
					arg_id = stage_out_ptr_var_id;
					added_out = true;
				}
				type_id = get<SPIRVariable>(arg_id).basetype;
				p_type = &get<SPIRType>(type_id);
				uint32_t next_id = ir.increase_bound_by(1);
				func.add_parameter(type_id, next_id, true);
				set<SPIRVariable>(next_id, type_id, StorageClassFunction, 0, arg_id);

				set_name(next_id, name);
			}
			else if (is_builtin_variable(var) && p_type->basetype == SPIRType::Struct)
			{
				// Get the pointee type
				type_id = get_pointee_type_id(type_id);
				p_type = &get<SPIRType>(type_id);

				uint32_t mbr_idx = 0;
				for (auto &mbr_type_id : p_type->member_types)
				{
					BuiltIn builtin = BuiltInMax;
					bool is_builtin = is_member_builtin(*p_type, mbr_idx, &builtin);
					if (is_builtin && has_active_builtin(builtin, var.storage))
					{
						// Add a arg variable with the same type and decorations as the member
						uint32_t next_ids = ir.increase_bound_by(2);
						uint32_t ptr_type_id = next_ids + 0;
						uint32_t var_id = next_ids + 1;

						// Make sure we have an actual pointer type,
						// so that we will get the appropriate address space when declaring these builtins.
						auto &ptr = set<SPIRType>(ptr_type_id, get<SPIRType>(mbr_type_id));
						ptr.self = mbr_type_id;
						ptr.storage = var.storage;
						ptr.pointer = true;
						ptr.parent_type = mbr_type_id;

						func.add_parameter(mbr_type_id, var_id, true);
						set<SPIRVariable>(var_id, ptr_type_id, StorageClassFunction);
						ir.meta[var_id].decoration = ir.meta[type_id].members[mbr_idx];
					}
					mbr_idx++;
				}
			}
			else
			{
				uint32_t next_id = ir.increase_bound_by(1);
				func.add_parameter(type_id, next_id, true);
				set<SPIRVariable>(next_id, type_id, StorageClassFunction, 0, arg_id);

				// Ensure the existing variable has a valid name and the new variable has all the same meta info
				set_name(arg_id, ensure_valid_name(to_name(arg_id), "v"));
				ir.meta[next_id] = ir.meta[arg_id];
			}
		}
	}
}

// For all variables that are some form of non-input-output interface block, mark that all the structs
// that are recursively contained within the type referenced by that variable should be packed tightly.
void CompilerMSL::mark_packable_structs()
{
	ir.for_each_typed_id<SPIRVariable>([&](uint32_t, SPIRVariable &var) {
		if (var.storage != StorageClassFunction && !is_hidden_variable(var))
		{
			auto &type = this->get<SPIRType>(var.basetype);
			if (type.pointer &&
			    (type.storage == StorageClassUniform || type.storage == StorageClassUniformConstant ||
			     type.storage == StorageClassPushConstant || type.storage == StorageClassStorageBuffer) &&
			    (has_decoration(type.self, DecorationBlock) || has_decoration(type.self, DecorationBufferBlock)))
				mark_as_packable(type);
		}
	});
}

// If the specified type is a struct, it and any nested structs
// are marked as packable with the SPIRVCrossDecorationPacked decoration,
void CompilerMSL::mark_as_packable(SPIRType &type)
{
	// If this is not the base type (eg. it's a pointer or array), tunnel down
	if (type.parent_type)
	{
		mark_as_packable(get<SPIRType>(type.parent_type));
		return;
	}

	if (type.basetype == SPIRType::Struct)
	{
		set_extended_decoration(type.self, SPIRVCrossDecorationPacked);

		// Recurse
		size_t mbr_cnt = type.member_types.size();
		for (uint32_t mbr_idx = 0; mbr_idx < mbr_cnt; mbr_idx++)
		{
			uint32_t mbr_type_id = type.member_types[mbr_idx];
			auto &mbr_type = get<SPIRType>(mbr_type_id);
			mark_as_packable(mbr_type);
			if (mbr_type.type_alias)
			{
				auto &mbr_type_alias = get<SPIRType>(mbr_type.type_alias);
				mark_as_packable(mbr_type_alias);
			}
		}
	}
}

// If a vertex attribute exists at the location, it is marked as being used by this shader
void CompilerMSL::mark_location_as_used_by_shader(uint32_t location, StorageClass storage)
{
	if ((get_execution_model() == ExecutionModelVertex || is_tessellation_shader()) && (storage == StorageClassInput))
		vtx_attrs_in_use.insert(location);
}

uint32_t CompilerMSL::get_target_components_for_fragment_location(uint32_t location) const
{
	auto itr = fragment_output_components.find(location);
	if (itr == end(fragment_output_components))
		return 4;
	else
		return itr->second;
}

uint32_t CompilerMSL::build_extended_vector_type(uint32_t type_id, uint32_t components)
{
	uint32_t new_type_id = ir.increase_bound_by(1);
	auto &type = set<SPIRType>(new_type_id, get<SPIRType>(type_id));
	type.vecsize = components;
	type.self = new_type_id;
	type.parent_type = type_id;
	type.pointer = false;

	return new_type_id;
}

void CompilerMSL::add_plain_variable_to_interface_block(StorageClass storage, const string &ib_var_ref,
                                                        SPIRType &ib_type, SPIRVariable &var, bool strip_array)
{
	bool is_builtin = is_builtin_variable(var);
	BuiltIn builtin = BuiltIn(get_decoration(var.self, DecorationBuiltIn));
	bool is_flat = has_decoration(var.self, DecorationFlat);
	bool is_noperspective = has_decoration(var.self, DecorationNoPerspective);
	bool is_centroid = has_decoration(var.self, DecorationCentroid);
	bool is_sample = has_decoration(var.self, DecorationSample);

	// Add a reference to the variable type to the interface struct.
	uint32_t ib_mbr_idx = uint32_t(ib_type.member_types.size());
	uint32_t type_id = ensure_correct_builtin_type(var.basetype, builtin);
	var.basetype = type_id;

	type_id = get_pointee_type_id(var.basetype);
	if (strip_array && is_array(get<SPIRType>(type_id)))
		type_id = get<SPIRType>(type_id).parent_type;
	auto &type = get<SPIRType>(type_id);
	uint32_t target_components = 0;
	uint32_t type_components = type.vecsize;
	bool padded_output = false;

	// Check if we need to pad fragment output to match a certain number of components.
	if (get_decoration_bitset(var.self).get(DecorationLocation) && msl_options.pad_fragment_output_components &&
	    get_entry_point().model == ExecutionModelFragment && storage == StorageClassOutput)
	{
		uint32_t locn = get_decoration(var.self, DecorationLocation);
		target_components = get_target_components_for_fragment_location(locn);
		if (type_components < target_components)
		{
			// Make a new type here.
			type_id = build_extended_vector_type(type_id, target_components);
			padded_output = true;
		}
	}

	ib_type.member_types.push_back(type_id);

	// Give the member a name
	string mbr_name = ensure_valid_name(to_expression(var.self), "m");
	set_member_name(ib_type.self, ib_mbr_idx, mbr_name);

	// Update the original variable reference to include the structure reference
	string qual_var_name = ib_var_ref + "." + mbr_name;
	auto &entry_func = get<SPIRFunction>(ir.default_entry_point);

	if (padded_output)
	{
		entry_func.add_local_variable(var.self);
		vars_needing_early_declaration.push_back(var.self);

		entry_func.fixup_hooks_out.push_back([=, &var]() {
			SPIRType &padded_type = this->get<SPIRType>(type_id);
			statement(qual_var_name, " = ", remap_swizzle(padded_type, type_components, to_name(var.self)), ";");
		});
	}
	else if (!strip_array)
		ir.meta[var.self].decoration.qualified_alias = qual_var_name;

	if (var.storage == StorageClassOutput && var.initializer != 0)
	{
		entry_func.fixup_hooks_in.push_back(
		    [=, &var]() { statement(qual_var_name, " = ", to_expression(var.initializer), ";"); });
	}

	// Copy the variable location from the original variable to the member
	if (get_decoration_bitset(var.self).get(DecorationLocation))
	{
		uint32_t locn = get_decoration(var.self, DecorationLocation);
		if (storage == StorageClassInput && (get_execution_model() == ExecutionModelVertex || is_tessellation_shader()))
		{
			type_id = ensure_correct_attribute_type(var.basetype, locn);
			var.basetype = type_id;
			type_id = get_pointee_type_id(type_id);
			if (strip_array && is_array(get<SPIRType>(type_id)))
				type_id = get<SPIRType>(type_id).parent_type;
			ib_type.member_types[ib_mbr_idx] = type_id;
		}
		set_member_decoration(ib_type.self, ib_mbr_idx, DecorationLocation, locn);
		mark_location_as_used_by_shader(locn, storage);
	}
	else if (is_builtin && is_tessellation_shader() && vtx_attrs_by_builtin.count(builtin))
	{
		uint32_t locn = vtx_attrs_by_builtin[builtin].location;
		set_member_decoration(ib_type.self, ib_mbr_idx, DecorationLocation, locn);
		mark_location_as_used_by_shader(locn, storage);
	}

	if (get_decoration_bitset(var.self).get(DecorationComponent))
	{
		uint32_t comp = get_decoration(var.self, DecorationComponent);
		set_member_decoration(ib_type.self, ib_mbr_idx, DecorationComponent, comp);
	}

	if (get_decoration_bitset(var.self).get(DecorationIndex))
	{
		uint32_t index = get_decoration(var.self, DecorationIndex);
		set_member_decoration(ib_type.self, ib_mbr_idx, DecorationIndex, index);
	}

	// Mark the member as builtin if needed
	if (is_builtin)
	{
		set_member_decoration(ib_type.self, ib_mbr_idx, DecorationBuiltIn, builtin);
		if (builtin == BuiltInPosition && storage == StorageClassOutput)
			qual_pos_var_name = qual_var_name;
	}

	// Copy interpolation decorations if needed
	if (is_flat)
		set_member_decoration(ib_type.self, ib_mbr_idx, DecorationFlat);
	if (is_noperspective)
		set_member_decoration(ib_type.self, ib_mbr_idx, DecorationNoPerspective);
	if (is_centroid)
		set_member_decoration(ib_type.self, ib_mbr_idx, DecorationCentroid);
	if (is_sample)
		set_member_decoration(ib_type.self, ib_mbr_idx, DecorationSample);

	set_extended_member_decoration(ib_type.self, ib_mbr_idx, SPIRVCrossDecorationInterfaceOrigID, var.self);
}

void CompilerMSL::add_composite_variable_to_interface_block(StorageClass storage, const string &ib_var_ref,
                                                            SPIRType &ib_type, SPIRVariable &var, bool strip_array)
{
	auto &entry_func = get<SPIRFunction>(ir.default_entry_point);
	auto &var_type = strip_array ? get_variable_element_type(var) : get_variable_data_type(var);
	uint32_t elem_cnt = 0;

	if (is_matrix(var_type))
	{
		if (is_array(var_type))
			SPIRV_CROSS_THROW("MSL cannot emit arrays-of-matrices in input and output variables.");

		elem_cnt = var_type.columns;
	}
	else if (is_array(var_type))
	{
		if (var_type.array.size() != 1)
			SPIRV_CROSS_THROW("MSL cannot emit arrays-of-arrays in input and output variables.");

		elem_cnt = to_array_size_literal(var_type);
	}

	bool is_builtin = is_builtin_variable(var);
	BuiltIn builtin = BuiltIn(get_decoration(var.self, DecorationBuiltIn));
	bool is_flat = has_decoration(var.self, DecorationFlat);
	bool is_noperspective = has_decoration(var.self, DecorationNoPerspective);
	bool is_centroid = has_decoration(var.self, DecorationCentroid);
	bool is_sample = has_decoration(var.self, DecorationSample);

	auto *usable_type = &var_type;
	if (usable_type->pointer)
		usable_type = &get<SPIRType>(usable_type->parent_type);
	while (is_array(*usable_type) || is_matrix(*usable_type))
		usable_type = &get<SPIRType>(usable_type->parent_type);

	// If a builtin, force it to have the proper name.
	if (is_builtin)
		set_name(var.self, builtin_to_glsl(builtin, StorageClassFunction));

	entry_func.add_local_variable(var.self);

	// We need to declare the variable early and at entry-point scope.
	vars_needing_early_declaration.push_back(var.self);

	for (uint32_t i = 0; i < elem_cnt; i++)
	{
		// Add a reference to the variable type to the interface struct.
		uint32_t ib_mbr_idx = uint32_t(ib_type.member_types.size());

		uint32_t target_components = 0;
		bool padded_output = false;
		uint32_t type_id = usable_type->self;

		// Check if we need to pad fragment output to match a certain number of components.
		if (get_decoration_bitset(var.self).get(DecorationLocation) && msl_options.pad_fragment_output_components &&
		    get_entry_point().model == ExecutionModelFragment && storage == StorageClassOutput)
		{
			uint32_t locn = get_decoration(var.self, DecorationLocation) + i;
			target_components = get_target_components_for_fragment_location(locn);
			if (usable_type->vecsize < target_components)
			{
				// Make a new type here.
				type_id = build_extended_vector_type(usable_type->self, target_components);
				padded_output = true;
			}
		}

		ib_type.member_types.push_back(get_pointee_type_id(type_id));

		// Give the member a name
		string mbr_name = ensure_valid_name(join(to_expression(var.self), "_", i), "m");
		set_member_name(ib_type.self, ib_mbr_idx, mbr_name);

		// There is no qualified alias since we need to flatten the internal array on return.
		if (get_decoration_bitset(var.self).get(DecorationLocation))
		{
			uint32_t locn = get_decoration(var.self, DecorationLocation) + i;
			if (storage == StorageClassInput &&
			    (get_execution_model() == ExecutionModelVertex || is_tessellation_shader()))
			{
				var.basetype = ensure_correct_attribute_type(var.basetype, locn);
				uint32_t mbr_type_id = ensure_correct_attribute_type(usable_type->self, locn);
				ib_type.member_types[ib_mbr_idx] = mbr_type_id;
			}
			set_member_decoration(ib_type.self, ib_mbr_idx, DecorationLocation, locn);
			mark_location_as_used_by_shader(locn, storage);
		}
		else if (is_builtin && is_tessellation_shader() && vtx_attrs_by_builtin.count(builtin))
		{
			uint32_t locn = vtx_attrs_by_builtin[builtin].location + i;
			set_member_decoration(ib_type.self, ib_mbr_idx, DecorationLocation, locn);
			mark_location_as_used_by_shader(locn, storage);
		}

		if (get_decoration_bitset(var.self).get(DecorationIndex))
		{
			uint32_t index = get_decoration(var.self, DecorationIndex);
			set_member_decoration(ib_type.self, ib_mbr_idx, DecorationIndex, index);
		}

		// Copy interpolation decorations if needed
		if (is_flat)
			set_member_decoration(ib_type.self, ib_mbr_idx, DecorationFlat);
		if (is_noperspective)
			set_member_decoration(ib_type.self, ib_mbr_idx, DecorationNoPerspective);
		if (is_centroid)
			set_member_decoration(ib_type.self, ib_mbr_idx, DecorationCentroid);
		if (is_sample)
			set_member_decoration(ib_type.self, ib_mbr_idx, DecorationSample);

		set_extended_member_decoration(ib_type.self, ib_mbr_idx, SPIRVCrossDecorationInterfaceOrigID, var.self);

		if (!strip_array)
		{
			switch (storage)
			{
			case StorageClassInput:
				entry_func.fixup_hooks_in.push_back(
				    [=, &var]() { statement(to_name(var.self), "[", i, "] = ", ib_var_ref, ".", mbr_name, ";"); });
				break;

			case StorageClassOutput:
				entry_func.fixup_hooks_out.push_back([=, &var]() {
					if (padded_output)
					{
						auto &padded_type = this->get<SPIRType>(type_id);
						statement(
						    ib_var_ref, ".", mbr_name, " = ",
						    remap_swizzle(padded_type, usable_type->vecsize, join(to_name(var.self), "[", i, "]")),
						    ";");
					}
					else
						statement(ib_var_ref, ".", mbr_name, " = ", to_name(var.self), "[", i, "];");
				});
				break;

			default:
				break;
			}
		}
	}
}

uint32_t CompilerMSL::get_accumulated_member_location(const SPIRVariable &var, uint32_t mbr_idx, bool strip_array)
{
	auto &type = strip_array ? get_variable_element_type(var) : get_variable_data_type(var);
	uint32_t location = get_decoration(var.self, DecorationLocation);

	for (uint32_t i = 0; i < mbr_idx; i++)
	{
		auto &mbr_type = get<SPIRType>(type.member_types[i]);

		// Start counting from any place we have a new location decoration.
		if (has_member_decoration(type.self, mbr_idx, DecorationLocation))
			location = get_member_decoration(type.self, mbr_idx, DecorationLocation);

		uint32_t location_count = 1;

		if (mbr_type.columns > 1)
			location_count = mbr_type.columns;

		if (!mbr_type.array.empty())
			for (uint32_t j = 0; j < uint32_t(mbr_type.array.size()); j++)
				location_count *= to_array_size_literal(mbr_type, j);

		location += location_count;
	}

	return location;
}

void CompilerMSL::add_composite_member_variable_to_interface_block(StorageClass storage, const string &ib_var_ref,
                                                                   SPIRType &ib_type, SPIRVariable &var,
                                                                   uint32_t mbr_idx, bool strip_array)
{
	auto &entry_func = get<SPIRFunction>(ir.default_entry_point);
	auto &var_type = strip_array ? get_variable_element_type(var) : get_variable_data_type(var);

	BuiltIn builtin;
	bool is_builtin = is_member_builtin(var_type, mbr_idx, &builtin);
	bool is_flat =
	    has_member_decoration(var_type.self, mbr_idx, DecorationFlat) || has_decoration(var.self, DecorationFlat);
	bool is_noperspective = has_member_decoration(var_type.self, mbr_idx, DecorationNoPerspective) ||
	                        has_decoration(var.self, DecorationNoPerspective);
	bool is_centroid = has_member_decoration(var_type.self, mbr_idx, DecorationCentroid) ||
	                   has_decoration(var.self, DecorationCentroid);
	bool is_sample =
	    has_member_decoration(var_type.self, mbr_idx, DecorationSample) || has_decoration(var.self, DecorationSample);

	uint32_t mbr_type_id = var_type.member_types[mbr_idx];
	auto &mbr_type = get<SPIRType>(mbr_type_id);
	uint32_t elem_cnt = 0;

	if (is_matrix(mbr_type))
	{
		if (is_array(mbr_type))
			SPIRV_CROSS_THROW("MSL cannot emit arrays-of-matrices in input and output variables.");

		elem_cnt = mbr_type.columns;
	}
	else if (is_array(mbr_type))
	{
		if (mbr_type.array.size() != 1)
			SPIRV_CROSS_THROW("MSL cannot emit arrays-of-arrays in input and output variables.");

		elem_cnt = to_array_size_literal(mbr_type);
	}

	auto *usable_type = &mbr_type;
	if (usable_type->pointer)
		usable_type = &get<SPIRType>(usable_type->parent_type);
	while (is_array(*usable_type) || is_matrix(*usable_type))
		usable_type = &get<SPIRType>(usable_type->parent_type);

	for (uint32_t i = 0; i < elem_cnt; i++)
	{
		// Add a reference to the variable type to the interface struct.
		uint32_t ib_mbr_idx = uint32_t(ib_type.member_types.size());
		ib_type.member_types.push_back(usable_type->self);

		// Give the member a name
		string mbr_name = ensure_valid_name(join(to_qualified_member_name(var_type, mbr_idx), "_", i), "m");
		set_member_name(ib_type.self, ib_mbr_idx, mbr_name);

		if (has_member_decoration(var_type.self, mbr_idx, DecorationLocation))
		{
			uint32_t locn = get_member_decoration(var_type.self, mbr_idx, DecorationLocation) + i;
			set_member_decoration(ib_type.self, ib_mbr_idx, DecorationLocation, locn);
			mark_location_as_used_by_shader(locn, storage);
		}
		else if (has_decoration(var.self, DecorationLocation))
		{
			uint32_t locn = get_accumulated_member_location(var, mbr_idx, strip_array) + i;
			set_member_decoration(ib_type.self, ib_mbr_idx, DecorationLocation, locn);
			mark_location_as_used_by_shader(locn, storage);
		}
		else if (is_builtin && is_tessellation_shader() && vtx_attrs_by_builtin.count(builtin))
		{
			uint32_t locn = vtx_attrs_by_builtin[builtin].location + i;
			set_member_decoration(ib_type.self, ib_mbr_idx, DecorationLocation, locn);
			mark_location_as_used_by_shader(locn, storage);
		}

		if (has_member_decoration(var_type.self, mbr_idx, DecorationComponent))
			SPIRV_CROSS_THROW("DecorationComponent on matrices and arrays make little sense.");

		// Copy interpolation decorations if needed
		if (is_flat)
			set_member_decoration(ib_type.self, ib_mbr_idx, DecorationFlat);
		if (is_noperspective)
			set_member_decoration(ib_type.self, ib_mbr_idx, DecorationNoPerspective);
		if (is_centroid)
			set_member_decoration(ib_type.self, ib_mbr_idx, DecorationCentroid);
		if (is_sample)
			set_member_decoration(ib_type.self, ib_mbr_idx, DecorationSample);

		set_extended_member_decoration(ib_type.self, ib_mbr_idx, SPIRVCrossDecorationInterfaceOrigID, var.self);
		set_extended_member_decoration(ib_type.self, ib_mbr_idx, SPIRVCrossDecorationInterfaceMemberIndex, mbr_idx);

		// Unflatten or flatten from [[stage_in]] or [[stage_out]] as appropriate.
		if (!strip_array)
		{
			switch (storage)
			{
			case StorageClassInput:
				entry_func.fixup_hooks_in.push_back([=, &var, &var_type]() {
					statement(to_name(var.self), ".", to_member_name(var_type, mbr_idx), "[", i, "] = ", ib_var_ref,
					          ".", mbr_name, ";");
				});
				break;

			case StorageClassOutput:
				entry_func.fixup_hooks_out.push_back([=, &var, &var_type]() {
					statement(ib_var_ref, ".", mbr_name, " = ", to_name(var.self), ".",
					          to_member_name(var_type, mbr_idx), "[", i, "];");
				});
				break;

			default:
				break;
			}
		}
	}
}

void CompilerMSL::add_plain_member_variable_to_interface_block(StorageClass storage, const string &ib_var_ref,
                                                               SPIRType &ib_type, SPIRVariable &var, uint32_t mbr_idx,
                                                               bool strip_array)
{
	auto &var_type = strip_array ? get_variable_element_type(var) : get_variable_data_type(var);
	auto &entry_func = get<SPIRFunction>(ir.default_entry_point);

	BuiltIn builtin = BuiltInMax;
	bool is_builtin = is_member_builtin(var_type, mbr_idx, &builtin);
	bool is_flat =
	    has_member_decoration(var_type.self, mbr_idx, DecorationFlat) || has_decoration(var.self, DecorationFlat);
	bool is_noperspective = has_member_decoration(var_type.self, mbr_idx, DecorationNoPerspective) ||
	                        has_decoration(var.self, DecorationNoPerspective);
	bool is_centroid = has_member_decoration(var_type.self, mbr_idx, DecorationCentroid) ||
	                   has_decoration(var.self, DecorationCentroid);
	bool is_sample =
	    has_member_decoration(var_type.self, mbr_idx, DecorationSample) || has_decoration(var.self, DecorationSample);

	// Add a reference to the member to the interface struct.
	uint32_t mbr_type_id = var_type.member_types[mbr_idx];
	uint32_t ib_mbr_idx = uint32_t(ib_type.member_types.size());
	mbr_type_id = ensure_correct_builtin_type(mbr_type_id, builtin);
	var_type.member_types[mbr_idx] = mbr_type_id;
	ib_type.member_types.push_back(mbr_type_id);

	// Give the member a name
	string mbr_name = ensure_valid_name(to_qualified_member_name(var_type, mbr_idx), "m");
	set_member_name(ib_type.self, ib_mbr_idx, mbr_name);

	// Update the original variable reference to include the structure reference
	string qual_var_name = ib_var_ref + "." + mbr_name;

	if (is_builtin && !strip_array)
	{
		// For the builtin gl_PerVertex, we cannot treat it as a block anyways,
		// so redirect to qualified name.
		set_member_qualified_name(var_type.self, mbr_idx, qual_var_name);
	}
	else if (!strip_array)
	{
		// Unflatten or flatten from [[stage_in]] or [[stage_out]] as appropriate.
		switch (storage)
		{
		case StorageClassInput:
			entry_func.fixup_hooks_in.push_back([=, &var, &var_type]() {
				statement(to_name(var.self), ".", to_member_name(var_type, mbr_idx), " = ", qual_var_name, ";");
			});
			break;

		case StorageClassOutput:
			entry_func.fixup_hooks_out.push_back([=, &var, &var_type]() {
				statement(qual_var_name, " = ", to_name(var.self), ".", to_member_name(var_type, mbr_idx), ";");
			});
			break;

		default:
			break;
		}
	}

	// Copy the variable location from the original variable to the member
	if (has_member_decoration(var_type.self, mbr_idx, DecorationLocation))
	{
		uint32_t locn = get_member_decoration(var_type.self, mbr_idx, DecorationLocation);
		if (storage == StorageClassInput && (get_execution_model() == ExecutionModelVertex || is_tessellation_shader()))
		{
			mbr_type_id = ensure_correct_attribute_type(mbr_type_id, locn);
			var_type.member_types[mbr_idx] = mbr_type_id;
			ib_type.member_types[ib_mbr_idx] = mbr_type_id;
		}
		set_member_decoration(ib_type.self, ib_mbr_idx, DecorationLocation, locn);
		mark_location_as_used_by_shader(locn, storage);
	}
	else if (has_decoration(var.self, DecorationLocation))
	{
		// The block itself might have a location and in this case, all members of the block
		// receive incrementing locations.
		uint32_t locn = get_accumulated_member_location(var, mbr_idx, strip_array);
		if (storage == StorageClassInput && (get_execution_model() == ExecutionModelVertex || is_tessellation_shader()))
		{
			mbr_type_id = ensure_correct_attribute_type(mbr_type_id, locn);
			var_type.member_types[mbr_idx] = mbr_type_id;
			ib_type.member_types[ib_mbr_idx] = mbr_type_id;
		}
		set_member_decoration(ib_type.self, ib_mbr_idx, DecorationLocation, locn);
		mark_location_as_used_by_shader(locn, storage);
	}
	else if (is_builtin && is_tessellation_shader() && vtx_attrs_by_builtin.count(builtin))
	{
		uint32_t locn = 0;
		auto builtin_itr = vtx_attrs_by_builtin.find(builtin);
		if (builtin_itr != end(vtx_attrs_by_builtin))
			locn = builtin_itr->second.location;
		set_member_decoration(ib_type.self, ib_mbr_idx, DecorationLocation, locn);
		mark_location_as_used_by_shader(locn, storage);
	}

	// Copy the component location, if present.
	if (has_member_decoration(var_type.self, mbr_idx, DecorationComponent))
	{
		uint32_t comp = get_member_decoration(var_type.self, mbr_idx, DecorationComponent);
		set_member_decoration(ib_type.self, ib_mbr_idx, DecorationComponent, comp);
	}

	// Mark the member as builtin if needed
	if (is_builtin)
	{
		set_member_decoration(ib_type.self, ib_mbr_idx, DecorationBuiltIn, builtin);
		if (builtin == BuiltInPosition && storage == StorageClassOutput)
			qual_pos_var_name = qual_var_name;
	}

	// Copy interpolation decorations if needed
	if (is_flat)
		set_member_decoration(ib_type.self, ib_mbr_idx, DecorationFlat);
	if (is_noperspective)
		set_member_decoration(ib_type.self, ib_mbr_idx, DecorationNoPerspective);
	if (is_centroid)
		set_member_decoration(ib_type.self, ib_mbr_idx, DecorationCentroid);
	if (is_sample)
		set_member_decoration(ib_type.self, ib_mbr_idx, DecorationSample);

	set_extended_member_decoration(ib_type.self, ib_mbr_idx, SPIRVCrossDecorationInterfaceOrigID, var.self);
	set_extended_member_decoration(ib_type.self, ib_mbr_idx, SPIRVCrossDecorationInterfaceMemberIndex, mbr_idx);
}

// In Metal, the tessellation levels are stored as tightly packed half-precision floating point values.
// But, stage-in attribute offsets and strides must be multiples of four, so we can't pass the levels
// individually. Therefore, we must pass them as vectors. Triangles get a single float4, with the outer
// levels in 'xyz' and the inner level in 'w'. Quads get a float4 containing the outer levels and a
// float2 containing the inner levels.
void CompilerMSL::add_tess_level_input_to_interface_block(const std::string &ib_var_ref, SPIRType &ib_type,
                                                          SPIRVariable &var)
{
	auto &entry_func = get<SPIRFunction>(ir.default_entry_point);
	auto &var_type = get_variable_element_type(var);

	BuiltIn builtin = BuiltIn(get_decoration(var.self, DecorationBuiltIn));

	// Force the variable to have the proper name.
	set_name(var.self, builtin_to_glsl(builtin, StorageClassFunction));

	if (get_entry_point().flags.get(ExecutionModeTriangles))
	{
		// Triangles are tricky, because we want only one member in the struct.

		// We need to declare the variable early and at entry-point scope.
		entry_func.add_local_variable(var.self);
		vars_needing_early_declaration.push_back(var.self);

		string mbr_name = "gl_TessLevel";

		// If we already added the other one, we can skip this step.
		if (!added_builtin_tess_level)
		{
			// Add a reference to the variable type to the interface struct.
			uint32_t ib_mbr_idx = uint32_t(ib_type.member_types.size());

			uint32_t type_id = build_extended_vector_type(var_type.self, 4);

			ib_type.member_types.push_back(type_id);

			// Give the member a name
			set_member_name(ib_type.self, ib_mbr_idx, mbr_name);

			// There is no qualified alias since we need to flatten the internal array on return.
			if (get_decoration_bitset(var.self).get(DecorationLocation))
			{
				uint32_t locn = get_decoration(var.self, DecorationLocation);
				set_member_decoration(ib_type.self, ib_mbr_idx, DecorationLocation, locn);
				mark_location_as_used_by_shader(locn, StorageClassInput);
			}
			else if (vtx_attrs_by_builtin.count(builtin))
			{
				uint32_t locn = vtx_attrs_by_builtin[builtin].location;
				set_member_decoration(ib_type.self, ib_mbr_idx, DecorationLocation, locn);
				mark_location_as_used_by_shader(locn, StorageClassInput);
			}

			added_builtin_tess_level = true;
		}

		switch (builtin)
		{
		case BuiltInTessLevelOuter:
			entry_func.fixup_hooks_in.push_back([=, &var]() {
				statement(to_name(var.self), "[0] = ", ib_var_ref, ".", mbr_name, ".x;");
				statement(to_name(var.self), "[1] = ", ib_var_ref, ".", mbr_name, ".y;");
				statement(to_name(var.self), "[2] = ", ib_var_ref, ".", mbr_name, ".z;");
			});
			break;

		case BuiltInTessLevelInner:
			entry_func.fixup_hooks_in.push_back(
			    [=, &var]() { statement(to_name(var.self), "[0] = ", ib_var_ref, ".", mbr_name, ".w;"); });
			break;

		default:
			assert(false);
			break;
		}
	}
	else
	{
		// Add a reference to the variable type to the interface struct.
		uint32_t ib_mbr_idx = uint32_t(ib_type.member_types.size());

		uint32_t type_id = build_extended_vector_type(var_type.self, builtin == BuiltInTessLevelOuter ? 4 : 2);
		// Change the type of the variable, too.
		uint32_t ptr_type_id = ir.increase_bound_by(1);
		auto &new_var_type = set<SPIRType>(ptr_type_id, get<SPIRType>(type_id));
		new_var_type.pointer = true;
		new_var_type.storage = StorageClassInput;
		new_var_type.parent_type = type_id;
		var.basetype = ptr_type_id;

		ib_type.member_types.push_back(type_id);

		// Give the member a name
		string mbr_name = to_expression(var.self);
		set_member_name(ib_type.self, ib_mbr_idx, mbr_name);

		// Since vectors can be indexed like arrays, there is no need to unpack this. We can
		// just refer to the vector directly. So give it a qualified alias.
		string qual_var_name = ib_var_ref + "." + mbr_name;
		ir.meta[var.self].decoration.qualified_alias = qual_var_name;

		if (get_decoration_bitset(var.self).get(DecorationLocation))
		{
			uint32_t locn = get_decoration(var.self, DecorationLocation);
			set_member_decoration(ib_type.self, ib_mbr_idx, DecorationLocation, locn);
			mark_location_as_used_by_shader(locn, StorageClassInput);
		}
		else if (vtx_attrs_by_builtin.count(builtin))
		{
			uint32_t locn = vtx_attrs_by_builtin[builtin].location;
			set_member_decoration(ib_type.self, ib_mbr_idx, DecorationLocation, locn);
			mark_location_as_used_by_shader(locn, StorageClassInput);
		}
	}
}

void CompilerMSL::add_variable_to_interface_block(StorageClass storage, const string &ib_var_ref, SPIRType &ib_type,
                                                  SPIRVariable &var, bool strip_array)
{
	auto &entry_func = get<SPIRFunction>(ir.default_entry_point);
	// Tessellation control I/O variables and tessellation evaluation per-point inputs are
	// usually declared as arrays. In these cases, we want to add the element type to the
	// interface block, since in Metal it's the interface block itself which is arrayed.
	auto &var_type = strip_array ? get_variable_element_type(var) : get_variable_data_type(var);
	bool is_builtin = is_builtin_variable(var);
	auto builtin = BuiltIn(get_decoration(var.self, DecorationBuiltIn));

	if (var_type.basetype == SPIRType::Struct)
	{
		if (!is_builtin_type(var_type) && (!capture_output_to_buffer || storage == StorageClassInput) && !strip_array)
		{
			// For I/O blocks or structs, we will need to pass the block itself around
			// to functions if they are used globally in leaf functions.
			// Rather than passing down member by member,
			// we unflatten I/O blocks while running the shader,
			// and pass the actual struct type down to leaf functions.
			// We then unflatten inputs, and flatten outputs in the "fixup" stages.
			entry_func.add_local_variable(var.self);
			vars_needing_early_declaration.push_back(var.self);
		}

		if (capture_output_to_buffer && storage != StorageClassInput && !has_decoration(var_type.self, DecorationBlock))
		{
			// In Metal tessellation shaders, the interface block itself is arrayed. This makes things
			// very complicated, since stage-in structures in MSL don't support nested structures.
			// Luckily, for stage-out when capturing output, we can avoid this and just add
			// composite members directly, because the stage-out structure is stored to a buffer,
			// not returned.
			add_plain_variable_to_interface_block(storage, ib_var_ref, ib_type, var, strip_array);
		}
		else
		{
			// Flatten the struct members into the interface struct
			for (uint32_t mbr_idx = 0; mbr_idx < uint32_t(var_type.member_types.size()); mbr_idx++)
			{
				builtin = BuiltInMax;
				is_builtin = is_member_builtin(var_type, mbr_idx, &builtin);
				auto &mbr_type = get<SPIRType>(var_type.member_types[mbr_idx]);

				if (!is_builtin || has_active_builtin(builtin, storage))
				{
					if ((!is_builtin ||
					     (storage == StorageClassInput && get_execution_model() != ExecutionModelFragment)) &&
					    (storage == StorageClassInput || storage == StorageClassOutput) &&
					    (is_matrix(mbr_type) || is_array(mbr_type)))
					{
						add_composite_member_variable_to_interface_block(storage, ib_var_ref, ib_type, var, mbr_idx,
						                                                 strip_array);
					}
					else
					{
						add_plain_member_variable_to_interface_block(storage, ib_var_ref, ib_type, var, mbr_idx,
						                                             strip_array);
					}
				}
			}
		}
	}
	else if (get_execution_model() == ExecutionModelTessellationEvaluation && storage == StorageClassInput &&
	         !strip_array && is_builtin && (builtin == BuiltInTessLevelOuter || builtin == BuiltInTessLevelInner))
	{
		add_tess_level_input_to_interface_block(ib_var_ref, ib_type, var);
	}
	else if (var_type.basetype == SPIRType::Boolean || var_type.basetype == SPIRType::Char ||
	         type_is_integral(var_type) || type_is_floating_point(var_type) || var_type.basetype == SPIRType::Boolean)
	{
		if (!is_builtin || has_active_builtin(builtin, storage))
		{
			// MSL does not allow matrices or arrays in input or output variables, so need to handle it specially.
			if ((!is_builtin || (storage == StorageClassInput && get_execution_model() != ExecutionModelFragment)) &&
			    (storage == StorageClassInput || (storage == StorageClassOutput && !capture_output_to_buffer)) &&
			    (is_matrix(var_type) || is_array(var_type)))
			{
				add_composite_variable_to_interface_block(storage, ib_var_ref, ib_type, var, strip_array);
			}
			else
			{
				add_plain_variable_to_interface_block(storage, ib_var_ref, ib_type, var, strip_array);
			}
		}
	}
}

// Fix up the mapping of variables to interface member indices, which is used to compile access chains
// for per-vertex variables in a tessellation control shader.
void CompilerMSL::fix_up_interface_member_indices(StorageClass storage, uint32_t ib_type_id)
{
	// Only needed for tessellation shaders.
	if (get_execution_model() != ExecutionModelTessellationControl &&
	    !(get_execution_model() == ExecutionModelTessellationEvaluation && storage == StorageClassInput))
		return;

	bool in_array = false;
	for (uint32_t i = 0; i < ir.meta[ib_type_id].members.size(); i++)
	{
		auto &mbr_dec = ir.meta[ib_type_id].members[i];
		uint32_t var_id = mbr_dec.extended.ib_orig_id;
		if (!var_id)
			continue;
		auto &var = get<SPIRVariable>(var_id);

		// Unfortunately, all this complexity is needed to handle flattened structs and/or
		// arrays.
		if (storage == StorageClassInput)
		{
			auto &type = get_variable_element_type(var);
			if (is_array(type) || is_matrix(type))
			{
				if (in_array)
					continue;
				in_array = true;
				set_extended_decoration(var_id, SPIRVCrossDecorationInterfaceMemberIndex, i);
			}
			else
			{
				if (type.basetype == SPIRType::Struct)
				{
					uint32_t mbr_idx =
					    get_extended_member_decoration(ib_type_id, i, SPIRVCrossDecorationInterfaceMemberIndex);
					auto &mbr_type = get<SPIRType>(type.member_types[mbr_idx]);

					if (is_array(mbr_type) || is_matrix(mbr_type))
					{
						if (in_array)
							continue;
						in_array = true;
						set_extended_member_decoration(var_id, mbr_idx, SPIRVCrossDecorationInterfaceMemberIndex, i);
					}
					else
					{
						in_array = false;
						set_extended_member_decoration(var_id, mbr_idx, SPIRVCrossDecorationInterfaceMemberIndex, i);
					}
				}
				else
				{
					in_array = false;
					set_extended_decoration(var_id, SPIRVCrossDecorationInterfaceMemberIndex, i);
				}
			}
		}
		else
			set_extended_decoration(var_id, SPIRVCrossDecorationInterfaceMemberIndex, i);
	}
}

// Add an interface structure for the type of storage, which is either StorageClassInput or StorageClassOutput.
// Returns the ID of the newly added variable, or zero if no variable was added.
uint32_t CompilerMSL::add_interface_block(StorageClass storage, bool patch)
{
	// Accumulate the variables that should appear in the interface struct
	SmallVector<SPIRVariable *> vars;
	bool incl_builtins = (storage == StorageClassOutput || is_tessellation_shader());

	ir.for_each_typed_id<SPIRVariable>([&](uint32_t var_id, SPIRVariable &var) {
		auto &type = this->get<SPIRType>(var.basetype);
		BuiltIn bi_type = BuiltIn(get_decoration(var_id, DecorationBuiltIn));
		if (var.storage == storage && interface_variable_exists_in_entry_point(var.self) &&
		    !is_hidden_variable(var, incl_builtins) && type.pointer &&
		    (has_decoration(var_id, DecorationPatch) || is_patch_block(type)) == patch &&
		    (!is_builtin_variable(var) || bi_type == BuiltInPosition || bi_type == BuiltInPointSize ||
		     bi_type == BuiltInClipDistance || bi_type == BuiltInCullDistance || bi_type == BuiltInLayer ||
		     bi_type == BuiltInViewportIndex || bi_type == BuiltInFragDepth || bi_type == BuiltInSampleMask ||
		     (get_execution_model() == ExecutionModelTessellationEvaluation &&
		      (bi_type == BuiltInTessLevelOuter || bi_type == BuiltInTessLevelInner))))
		{
			vars.push_back(&var);
		}
	});

	// If no variables qualify, leave.
	// For patch input in a tessellation evaluation shader, the per-vertex stage inputs
	// are included in a special patch control point array.
	if (vars.empty() && !(storage == StorageClassInput && patch && stage_in_var_id))
		return 0;

	// Add a new typed variable for this interface structure.
	// The initializer expression is allocated here, but populated when the function
	// declaraion is emitted, because it is cleared after each compilation pass.
	uint32_t next_id = ir.increase_bound_by(3);
	uint32_t ib_type_id = next_id++;
	auto &ib_type = set<SPIRType>(ib_type_id);
	ib_type.basetype = SPIRType::Struct;
	ib_type.storage = storage;
	set_decoration(ib_type_id, DecorationBlock);

	uint32_t ib_var_id = next_id++;
	auto &var = set<SPIRVariable>(ib_var_id, ib_type_id, storage, 0);
	var.initializer = next_id++;

	string ib_var_ref;
	auto &entry_func = get<SPIRFunction>(ir.default_entry_point);
	switch (storage)
	{
	case StorageClassInput:
		ib_var_ref = patch ? patch_stage_in_var_name : stage_in_var_name;
		if (get_execution_model() == ExecutionModelTessellationControl)
		{
			// Add a hook to populate the shared workgroup memory containing
			// the gl_in array.
			entry_func.fixup_hooks_in.push_back([=]() {
				// Can't use PatchVertices yet; the hook for that may not have run yet.
				statement("if (", to_expression(builtin_invocation_id_id), " < ", "spvIndirectParams[0])");
				statement("    ", input_wg_var_name, "[", to_expression(builtin_invocation_id_id), "] = ", ib_var_ref,
				          ";");
				statement("threadgroup_barrier(mem_flags::mem_threadgroup);");
				statement("if (", to_expression(builtin_invocation_id_id), " >= ", get_entry_point().output_vertices,
				          ")");
				statement("    return;");
			});
		}
		break;

	case StorageClassOutput:
	{
		ib_var_ref = patch ? patch_stage_out_var_name : stage_out_var_name;

		// Add the output interface struct as a local variable to the entry function.
		// If the entry point should return the output struct, set the entry function
		// to return the output interface struct, otherwise to return nothing.
		// Indicate the output var requires early initialization.
		bool ep_should_return_output = !get_is_rasterization_disabled();
		uint32_t rtn_id = ep_should_return_output ? ib_var_id : 0;
		if (!capture_output_to_buffer)
		{
			entry_func.add_local_variable(ib_var_id);
			for (auto &blk_id : entry_func.blocks)
			{
				auto &blk = get<SPIRBlock>(blk_id);
				if (blk.terminator == SPIRBlock::Return)
					blk.return_value = rtn_id;
			}
			vars_needing_early_declaration.push_back(ib_var_id);
		}
		else
		{
			switch (get_execution_model())
			{
			case ExecutionModelVertex:
			case ExecutionModelTessellationEvaluation:
				// Instead of declaring a struct variable to hold the output and then
				// copying that to the output buffer, we'll declare the output variable
				// as a reference to the final output element in the buffer. Then we can
				// avoid the extra copy.
				entry_func.fixup_hooks_in.push_back([=]() {
					if (stage_out_var_id)
					{
						// The first member of the indirect buffer is always the number of vertices
						// to draw.
						statement("device ", to_name(ir.default_entry_point), "_", ib_var_ref, "& ", ib_var_ref, " = ",
						          output_buffer_var_name, "[(", to_expression(builtin_instance_idx_id), " - ",
						          to_expression(builtin_base_instance_id), ") * spvIndirectParams[0] + ",
						          to_expression(builtin_vertex_idx_id), " - ", to_expression(builtin_base_vertex_id),
						          "];");
					}
				});
				break;
			case ExecutionModelTessellationControl:
				if (patch)
					entry_func.fixup_hooks_in.push_back([=]() {
						statement("device ", to_name(ir.default_entry_point), "_", ib_var_ref, "& ", ib_var_ref, " = ",
						          patch_output_buffer_var_name, "[", to_expression(builtin_primitive_id_id), "];");
					});
				else
					entry_func.fixup_hooks_in.push_back([=]() {
						statement("device ", to_name(ir.default_entry_point), "_", ib_var_ref, "* gl_out = &",
						          output_buffer_var_name, "[", to_expression(builtin_primitive_id_id), " * ",
						          get_entry_point().output_vertices, "];");
					});
				break;
			default:
				break;
			}
		}
		break;
	}

	default:
		break;
	}

	set_name(ib_type_id, to_name(ir.default_entry_point) + "_" + ib_var_ref);
	set_name(ib_var_id, ib_var_ref);

	for (auto p_var : vars)
	{
		bool strip_array =
		    (get_execution_model() == ExecutionModelTessellationControl ||
		     (get_execution_model() == ExecutionModelTessellationEvaluation && storage == StorageClassInput)) &&
		    !patch;
		add_variable_to_interface_block(storage, ib_var_ref, ib_type, *p_var, strip_array);
	}

	// Sort the members of the structure by their locations.
	MemberSorter member_sorter(ib_type, ir.meta[ib_type_id], MemberSorter::Location);
	member_sorter.sort();

	// The member indices were saved to the original variables, but after the members
	// were sorted, those indices are now likely incorrect. Fix those up now.
	if (!patch)
		fix_up_interface_member_indices(storage, ib_type_id);

	// For patch inputs, add one more member, holding the array of control point data.
	if (get_execution_model() == ExecutionModelTessellationEvaluation && storage == StorageClassInput && patch &&
	    stage_in_var_id)
	{
		uint32_t pcp_type_id = ir.increase_bound_by(1);
		auto &pcp_type = set<SPIRType>(pcp_type_id, ib_type);
		pcp_type.basetype = SPIRType::ControlPointArray;
		pcp_type.parent_type = pcp_type.type_alias = get_stage_in_struct_type().self;
		pcp_type.storage = storage;
		ir.meta[pcp_type_id] = ir.meta[ib_type.self];
		uint32_t mbr_idx = uint32_t(ib_type.member_types.size());
		ib_type.member_types.push_back(pcp_type_id);
		set_member_name(ib_type.self, mbr_idx, "gl_in");
	}

	return ib_var_id;
}

uint32_t CompilerMSL::add_interface_block_pointer(uint32_t ib_var_id, StorageClass storage)
{
	if (!ib_var_id)
		return 0;

	uint32_t ib_ptr_var_id;
	uint32_t next_id = ir.increase_bound_by(3);
	auto &ib_type = expression_type(ib_var_id);
	if (get_execution_model() == ExecutionModelTessellationControl)
	{
		// Tessellation control per-vertex I/O is presented as an array, so we must
		// do the same with our struct here.
		uint32_t ib_ptr_type_id = next_id++;
		auto &ib_ptr_type = set<SPIRType>(ib_ptr_type_id, ib_type);
		ib_ptr_type.parent_type = ib_ptr_type.type_alias = ib_type.self;
		ib_ptr_type.pointer = true;
		ib_ptr_type.storage = storage == StorageClassInput ? StorageClassWorkgroup : StorageClassStorageBuffer;
		ir.meta[ib_ptr_type_id] = ir.meta[ib_type.self];
		// To ensure that get_variable_data_type() doesn't strip off the pointer,
		// which we need, use another pointer.
		uint32_t ib_ptr_ptr_type_id = next_id++;
		auto &ib_ptr_ptr_type = set<SPIRType>(ib_ptr_ptr_type_id, ib_ptr_type);
		ib_ptr_ptr_type.parent_type = ib_ptr_type_id;
		ib_ptr_ptr_type.type_alias = ib_type.self;
		ib_ptr_ptr_type.storage = StorageClassFunction;
		ir.meta[ib_ptr_ptr_type_id] = ir.meta[ib_type.self];

		ib_ptr_var_id = next_id;
		set<SPIRVariable>(ib_ptr_var_id, ib_ptr_ptr_type_id, StorageClassFunction, 0);
		set_name(ib_ptr_var_id, storage == StorageClassInput ? input_wg_var_name : "gl_out");
	}
	else
	{
		// Tessellation evaluation per-vertex inputs are also presented as arrays.
		// But, in Metal, this array uses a very special type, 'patch_control_point<T>',
		// which is a container that can be used to access the control point data.
		// To represent this, a special 'ControlPointArray' type has been added to the
		// SPIRV-Cross type system. It should only be generated by and seen in the MSL
		// backend (i.e. this one).
		uint32_t pcp_type_id = next_id++;
		auto &pcp_type = set<SPIRType>(pcp_type_id, ib_type);
		pcp_type.basetype = SPIRType::ControlPointArray;
		pcp_type.parent_type = pcp_type.type_alias = ib_type.self;
		pcp_type.storage = storage;
		ir.meta[pcp_type_id] = ir.meta[ib_type.self];

		ib_ptr_var_id = next_id;
		set<SPIRVariable>(ib_ptr_var_id, pcp_type_id, storage, 0);
		set_name(ib_ptr_var_id, "gl_in");
		ir.meta[ib_ptr_var_id].decoration.qualified_alias = join(patch_stage_in_var_name, ".gl_in");
	}
	return ib_ptr_var_id;
}

// Ensure that the type is compatible with the builtin.
// If it is, simply return the given type ID.
// Otherwise, create a new type, and return it's ID.
uint32_t CompilerMSL::ensure_correct_builtin_type(uint32_t type_id, BuiltIn builtin)
{
	auto &type = get<SPIRType>(type_id);

	if ((builtin == BuiltInSampleMask && is_array(type)) ||
	    ((builtin == BuiltInLayer || builtin == BuiltInViewportIndex) && type.basetype != SPIRType::UInt))
	{
		uint32_t next_id = ir.increase_bound_by(type.pointer ? 2 : 1);
		uint32_t base_type_id = next_id++;
		auto &base_type = set<SPIRType>(base_type_id);
		base_type.basetype = SPIRType::UInt;
		base_type.width = 32;

		if (!type.pointer)
			return base_type_id;

		uint32_t ptr_type_id = next_id++;
		auto &ptr_type = set<SPIRType>(ptr_type_id);
		ptr_type = base_type;
		ptr_type.pointer = true;
		ptr_type.storage = type.storage;
		ptr_type.parent_type = base_type_id;
		return ptr_type_id;
	}

	return type_id;
}

// Ensure that the type is compatible with the vertex attribute.
// If it is, simply return the given type ID.
// Otherwise, create a new type, and return its ID.
uint32_t CompilerMSL::ensure_correct_attribute_type(uint32_t type_id, uint32_t location)
{
	auto &type = get<SPIRType>(type_id);

	auto p_va = vtx_attrs_by_location.find(location);
	if (p_va == end(vtx_attrs_by_location))
		return type_id;

	switch (p_va->second.format)
	{
	case MSL_VERTEX_FORMAT_UINT8:
	{
		switch (type.basetype)
		{
		case SPIRType::UByte:
		case SPIRType::UShort:
		case SPIRType::UInt:
			return type_id;
		case SPIRType::Short:
		case SPIRType::Int:
			break;
		default:
			SPIRV_CROSS_THROW("Vertex attribute type mismatch between host and shader");
		}
		uint32_t next_id = ir.increase_bound_by(type.pointer ? 2 : 1);
		uint32_t base_type_id = next_id++;
		auto &base_type = set<SPIRType>(base_type_id);
		base_type = type;
		base_type.basetype = type.basetype == SPIRType::Short ? SPIRType::UShort : SPIRType::UInt;
		base_type.pointer = false;

		if (!type.pointer)
			return base_type_id;

		uint32_t ptr_type_id = next_id++;
		auto &ptr_type = set<SPIRType>(ptr_type_id);
		ptr_type = base_type;
		ptr_type.pointer = true;
		ptr_type.storage = type.storage;
		ptr_type.parent_type = base_type_id;
		return ptr_type_id;
	}

	case MSL_VERTEX_FORMAT_UINT16:
	{
		switch (type.basetype)
		{
		case SPIRType::UShort:
		case SPIRType::UInt:
			return type_id;
		case SPIRType::Int:
			break;
		default:
			SPIRV_CROSS_THROW("Vertex attribute type mismatch between host and shader");
		}
		uint32_t next_id = ir.increase_bound_by(type.pointer ? 2 : 1);
		uint32_t base_type_id = next_id++;
		auto &base_type = set<SPIRType>(base_type_id);
		base_type = type;
		base_type.basetype = SPIRType::UInt;
		base_type.pointer = false;

		if (!type.pointer)
			return base_type_id;

		uint32_t ptr_type_id = next_id++;
		auto &ptr_type = set<SPIRType>(ptr_type_id);
		ptr_type = base_type;
		ptr_type.pointer = true;
		ptr_type.storage = type.storage;
		ptr_type.parent_type = base_type_id;
		return ptr_type_id;
	}

	default:
	case MSL_VERTEX_FORMAT_OTHER:
		break;
	}

	return type_id;
}

// Sort the members of the struct type by offset, and pack and then pad members where needed
// to align MSL members with SPIR-V offsets. The struct members are iterated twice. Packing
// occurs first, followed by padding, because packing a member reduces both its size and its
// natural alignment, possibly requiring a padding member to be added ahead of it.
void CompilerMSL::align_struct(SPIRType &ib_type)
{
	uint32_t &ib_type_id = ib_type.self;

	// Sort the members of the interface structure by their offset.
	// They should already be sorted per SPIR-V spec anyway.
	MemberSorter member_sorter(ib_type, ir.meta[ib_type_id], MemberSorter::Offset);
	member_sorter.sort();

	uint32_t mbr_cnt = uint32_t(ib_type.member_types.size());

	// Test the alignment of each member, and if a member should be closer to the previous
	// member than the default spacing expects, it is likely that the previous member is in
	// a packed format. If so, and the previous member is packable, pack it.
	// For example...this applies to any 3-element vector that is followed by a scalar.
	uint32_t curr_offset = 0;
	for (uint32_t mbr_idx = 0; mbr_idx < mbr_cnt; mbr_idx++)
	{
		if (is_member_packable(ib_type, mbr_idx))
		{
			set_extended_member_decoration(ib_type_id, mbr_idx, SPIRVCrossDecorationPacked);
			set_extended_member_decoration(ib_type_id, mbr_idx, SPIRVCrossDecorationPackedType,
			                               ib_type.member_types[mbr_idx]);
		}

		// Align current offset to the current member's default alignment.
		size_t align_mask = get_declared_struct_member_alignment(ib_type, mbr_idx) - 1;
		uint32_t aligned_curr_offset = uint32_t((curr_offset + align_mask) & ~align_mask);

		// Fetch the member offset as declared in the SPIRV.
		uint32_t mbr_offset = get_member_decoration(ib_type_id, mbr_idx, DecorationOffset);
		if (mbr_offset > aligned_curr_offset)
		{
			// Since MSL and SPIR-V have slightly different struct member alignment and
			// size rules, we'll pad to standard C-packing rules. If the member is farther
			// away than C-packing, expects, add an inert padding member before the the member.
			MSLStructMemberKey key = get_struct_member_key(ib_type_id, mbr_idx);
			struct_member_padding[key] = mbr_offset - curr_offset;
		}

		// Increment the current offset to be positioned immediately after the current member.
		// Don't do this for the last member since it can be unsized, and it is not relevant for padding purposes here.
		if (mbr_idx + 1 < mbr_cnt)
			curr_offset = mbr_offset + uint32_t(get_declared_struct_member_size(ib_type, mbr_idx));
	}
}

// Returns whether the specified struct member supports a packable type
// variation that is smaller than the unpacked variation of that type.
bool CompilerMSL::is_member_packable(SPIRType &ib_type, uint32_t index)
{
	// We've already marked it as packable
	if (has_extended_member_decoration(ib_type.self, index, SPIRVCrossDecorationPacked))
		return true;

	auto &mbr_type = get<SPIRType>(ib_type.member_types[index]);

	uint32_t component_size = mbr_type.width / 8;
	uint32_t unpacked_mbr_size;
	if (mbr_type.vecsize == 3)
		unpacked_mbr_size = component_size * (mbr_type.vecsize + 1) * mbr_type.columns;
	else
		unpacked_mbr_size = component_size * mbr_type.vecsize * mbr_type.columns;

	// Special case for packing. Check for float[] or vec2[] in std140 layout. Here we actually need to pad out instead,
	// but we will use the same mechanism.
	if (is_array(mbr_type) && (is_scalar(mbr_type) || is_vector(mbr_type)) && mbr_type.vecsize <= 2 &&
	    type_struct_member_array_stride(ib_type, index) == 4 * component_size)
	{
		return true;
	}

	// Check for array of struct, where the SPIR-V declares an array stride which is larger than the struct itself.
	// This can happen for struct A { float a }; A a[]; in std140 layout.
	// TODO: Emit a padded struct which can be used for this purpose.
	if (is_array(mbr_type) && mbr_type.basetype == SPIRType::Struct)
	{
		size_t declared_struct_size = get_declared_struct_size(mbr_type);
		size_t alignment = get_declared_struct_member_alignment(ib_type, index);
		declared_struct_size = (declared_struct_size + alignment - 1) & ~(alignment - 1);
		if (type_struct_member_array_stride(ib_type, index) > declared_struct_size)
			return true;
	}

	// TODO: Another sanity check for matrices. We currently do not support std140 matrices which need to be padded out per column.
	//if (is_matrix(mbr_type) && mbr_type.vecsize <= 2 && type_struct_member_matrix_stride(ib_type, index) == 16)
	//	SPIRV_CROSS_THROW("Currently cannot support matrices with small vector size in std140 layout.");

	// Only vectors or 3-row matrices need to be packed.
	if (mbr_type.vecsize == 1 || (is_matrix(mbr_type) && mbr_type.vecsize != 3))
		return false;

	// Only row-major matrices need to be packed.
	if (is_matrix(mbr_type) && !has_member_decoration(ib_type.self, index, DecorationRowMajor))
		return false;

	if (is_array(mbr_type))
	{
		// If member is an array, and the array stride is larger than the type needs, don't pack it.
		// Take into consideration multi-dimentional arrays.
		uint32_t md_elem_cnt = 1;
		size_t last_elem_idx = mbr_type.array.size() - 1;
		for (uint32_t i = 0; i < last_elem_idx; i++)
			md_elem_cnt *= max(to_array_size_literal(mbr_type, i), 1u);

		uint32_t unpacked_array_stride = unpacked_mbr_size * md_elem_cnt;
		uint32_t array_stride = type_struct_member_array_stride(ib_type, index);
		return unpacked_array_stride > array_stride;
	}
	else
	{
		uint32_t mbr_offset_curr = get_member_decoration(ib_type.self, index, DecorationOffset);
		// For vectors, pack if the member's offset doesn't conform to the
		// type's usual alignment. For example, a float3 at offset 4.
		if (!is_matrix(mbr_type) && (mbr_offset_curr % unpacked_mbr_size))
			return true;
		// Pack if there is not enough space between this member and next.
		// If last member, only pack if it's a row-major matrix.
		if (index < ib_type.member_types.size() - 1)
		{
			uint32_t mbr_offset_next = get_member_decoration(ib_type.self, index + 1, DecorationOffset);
			return unpacked_mbr_size > mbr_offset_next - mbr_offset_curr;
		}
		else
			return is_matrix(mbr_type);
	}
}

// Returns a combination of type ID and member index for use as hash key
MSLStructMemberKey CompilerMSL::get_struct_member_key(uint32_t type_id, uint32_t index)
{
	MSLStructMemberKey k = type_id;
	k <<= 32;
	k += index;
	return k;
}

void CompilerMSL::emit_store_statement(uint32_t lhs_expression, uint32_t rhs_expression)
{
	if (!has_extended_decoration(lhs_expression, SPIRVCrossDecorationPacked) ||
	    get_extended_decoration(lhs_expression, SPIRVCrossDecorationPackedType) == 0)
	{
		CompilerGLSL::emit_store_statement(lhs_expression, rhs_expression);
	}
	else
	{
		// Special handling when storing to a float[] or float2[] in std140 layout.

		auto &type = get<SPIRType>(get_extended_decoration(lhs_expression, SPIRVCrossDecorationPackedType));
		string lhs = to_dereferenced_expression(lhs_expression);
		string rhs = to_pointer_expression(rhs_expression);

		// Unpack the expression so we can store to it with a float or float2.
		// It's still an l-value, so it's fine. Most other unpacking of expressions turn them into r-values instead.
		if (is_scalar(type) && is_array(type))
			lhs = enclose_expression(lhs) + ".x";
		else if (is_vector(type) && type.vecsize == 2 && is_array(type))
			lhs = enclose_expression(lhs) + ".xy";

		if (!optimize_read_modify_write(expression_type(rhs_expression), lhs, rhs))
			statement(lhs, " = ", rhs, ";");
		register_write(lhs_expression);
	}
}

// Converts the format of the current expression from packed to unpacked,
// by wrapping the expression in a constructor of the appropriate type.
string CompilerMSL::unpack_expression_type(string expr_str, const SPIRType &type, uint32_t packed_type_id)
{
	const SPIRType *packed_type = nullptr;
	if (packed_type_id)
		packed_type = &get<SPIRType>(packed_type_id);

	// float[] and float2[] cases are really just padding, so directly swizzle from the backing float4 instead.
	if (packed_type && is_array(*packed_type) && is_scalar(*packed_type))
		return enclose_expression(expr_str) + ".x";
	else if (packed_type && is_array(*packed_type) && is_vector(*packed_type) && packed_type->vecsize == 2)
		return enclose_expression(expr_str) + ".xy";
	else
		return join(type_to_glsl(type), "(", expr_str, ")");
}

// Emits the file header info
void CompilerMSL::emit_header()
{
	// This particular line can be overridden during compilation, so make it a flag and not a pragma line.
	if (suppress_missing_prototypes)
		statement("#pragma clang diagnostic ignored \"-Wmissing-prototypes\"");
	for (auto &pragma : pragma_lines)
		statement(pragma);

	if (!pragma_lines.empty() || suppress_missing_prototypes)
		statement("");

	statement("#include <metal_stdlib>");
	statement("#include <simd/simd.h>");

	for (auto &header : header_lines)
		statement(header);

	statement("");
	statement("using namespace metal;");
	statement("");

	for (auto &td : typedef_lines)
		statement(td);

	if (!typedef_lines.empty())
		statement("");
}

void CompilerMSL::add_pragma_line(const string &line)
{
	auto rslt = pragma_lines.insert(line);
	if (rslt.second)
		force_recompile();
}

void CompilerMSL::add_typedef_line(const string &line)
{
	auto rslt = typedef_lines.insert(line);
	if (rslt.second)
		force_recompile();
}

// Emits any needed custom function bodies.
void CompilerMSL::emit_custom_functions()
{
	for (uint32_t i = SPVFuncImplArrayCopyMultidimMax; i >= 2; i--)
		if (spv_function_implementations.count(static_cast<SPVFuncImpl>(SPVFuncImplArrayCopyMultidimBase + i)))
			spv_function_implementations.insert(static_cast<SPVFuncImpl>(SPVFuncImplArrayCopyMultidimBase + i - 1));

	for (auto &spv_func : spv_function_implementations)
	{
		switch (spv_func)
		{
		case SPVFuncImplMod:
			statement("// Implementation of the GLSL mod() function, which is slightly different than Metal fmod()");
			statement("template<typename Tx, typename Ty>");
			statement("Tx mod(Tx x, Ty y)");
			begin_scope();
			statement("return x - y * floor(x / y);");
			end_scope();
			statement("");
			break;

		case SPVFuncImplRadians:
			statement("// Implementation of the GLSL radians() function");
			statement("template<typename T>");
			statement("T radians(T d)");
			begin_scope();
			statement("return d * T(0.01745329251);");
			end_scope();
			statement("");
			break;

		case SPVFuncImplDegrees:
			statement("// Implementation of the GLSL degrees() function");
			statement("template<typename T>");
			statement("T degrees(T r)");
			begin_scope();
			statement("return r * T(57.2957795131);");
			end_scope();
			statement("");
			break;

		case SPVFuncImplFindILsb:
			statement("// Implementation of the GLSL findLSB() function");
			statement("template<typename T>");
			statement("T findLSB(T x)");
			begin_scope();
			statement("return select(ctz(x), T(-1), x == T(0));");
			end_scope();
			statement("");
			break;

		case SPVFuncImplFindUMsb:
			statement("// Implementation of the unsigned GLSL findMSB() function");
			statement("template<typename T>");
			statement("T findUMSB(T x)");
			begin_scope();
			statement("return select(clz(T(0)) - (clz(x) + T(1)), T(-1), x == T(0));");
			end_scope();
			statement("");
			break;

		case SPVFuncImplFindSMsb:
			statement("// Implementation of the signed GLSL findMSB() function");
			statement("template<typename T>");
			statement("T findSMSB(T x)");
			begin_scope();
			statement("T v = select(x, T(-1) - x, x < T(0));");
			statement("return select(clz(T(0)) - (clz(v) + T(1)), T(-1), v == T(0));");
			end_scope();
			statement("");
			break;

		case SPVFuncImplSSign:
			statement("// Implementation of the GLSL sign() function for integer types");
			statement("template<typename T, typename E = typename enable_if<is_integral<T>::value>::type>");
			statement("T sign(T x)");
			begin_scope();
			statement("return select(select(select(x, T(0), x == T(0)), T(1), x > T(0)), T(-1), x < T(0));");
			end_scope();
			statement("");
			break;

		case SPVFuncImplArrayCopy:
			statement("// Implementation of an array copy function to cover GLSL's ability to copy an array via "
			          "assignment.");
			statement("template<typename T, uint N>");
			statement("void spvArrayCopyFromStack1(thread T (&dst)[N], thread const T (&src)[N])");
			begin_scope();
			statement("for (uint i = 0; i < N; dst[i] = src[i], i++);");
			end_scope();
			statement("");

			statement("template<typename T, uint N>");
			statement("void spvArrayCopyFromConstant1(thread T (&dst)[N], constant T (&src)[N])");
			begin_scope();
			statement("for (uint i = 0; i < N; dst[i] = src[i], i++);");
			end_scope();
			statement("");
			break;

		case SPVFuncImplArrayOfArrayCopy2Dim:
		case SPVFuncImplArrayOfArrayCopy3Dim:
		case SPVFuncImplArrayOfArrayCopy4Dim:
		case SPVFuncImplArrayOfArrayCopy5Dim:
		case SPVFuncImplArrayOfArrayCopy6Dim:
		{
			static const char *function_name_tags[] = {
				"FromStack",
				"FromConstant",
			};

			static const char *src_address_space[] = {
				"thread const",
				"constant",
			};

			for (uint32_t variant = 0; variant < 2; variant++)
			{
				uint32_t dimensions = spv_func - SPVFuncImplArrayCopyMultidimBase;
				string tmp = "template<typename T";
				for (uint8_t i = 0; i < dimensions; i++)
				{
					tmp += ", uint ";
					tmp += 'A' + i;
				}
				tmp += ">";
				statement(tmp);

				string array_arg;
				for (uint8_t i = 0; i < dimensions; i++)
				{
					array_arg += "[";
					array_arg += 'A' + i;
					array_arg += "]";
				}

				statement("void spvArrayCopy", function_name_tags[variant], dimensions, "(thread T (&dst)", array_arg,
				          ", ", src_address_space[variant], " T (&src)", array_arg, ")");

				begin_scope();
				statement("for (uint i = 0; i < A; i++)");
				begin_scope();
				statement("spvArrayCopy", function_name_tags[variant], dimensions - 1, "(dst[i], src[i]);");
				end_scope();
				end_scope();
				statement("");
			}
			break;
		}

		case SPVFuncImplTexelBufferCoords:
		{
			string tex_width_str = convert_to_string(msl_options.texel_buffer_texture_width);
			statement("// Returns 2D texture coords corresponding to 1D texel buffer coords");
			statement("uint2 spvTexelBufferCoord(uint tc)");
			begin_scope();
			statement(join("return uint2(tc % ", tex_width_str, ", tc / ", tex_width_str, ");"));
			end_scope();
			statement("");
			break;
		}

		case SPVFuncImplInverse4x4:
			statement("// Returns the determinant of a 2x2 matrix.");
			statement("inline float spvDet2x2(float a1, float a2, float b1, float b2)");
			begin_scope();
			statement("return a1 * b2 - b1 * a2;");
			end_scope();
			statement("");

			statement("// Returns the determinant of a 3x3 matrix.");
			statement("inline float spvDet3x3(float a1, float a2, float a3, float b1, float b2, float b3, float c1, "
			          "float c2, float c3)");
			begin_scope();
			statement("return a1 * spvDet2x2(b2, b3, c2, c3) - b1 * spvDet2x2(a2, a3, c2, c3) + c1 * spvDet2x2(a2, a3, "
			          "b2, b3);");
			end_scope();
			statement("");
			statement("// Returns the inverse of a matrix, by using the algorithm of calculating the classical");
			statement("// adjoint and dividing by the determinant. The contents of the matrix are changed.");
			statement("float4x4 spvInverse4x4(float4x4 m)");
			begin_scope();
			statement("float4x4 adj;	// The adjoint matrix (inverse after dividing by determinant)");
			statement_no_indent("");
			statement("// Create the transpose of the cofactors, as the classical adjoint of the matrix.");
			statement("adj[0][0] =  spvDet3x3(m[1][1], m[1][2], m[1][3], m[2][1], m[2][2], m[2][3], m[3][1], m[3][2], "
			          "m[3][3]);");
			statement("adj[0][1] = -spvDet3x3(m[0][1], m[0][2], m[0][3], m[2][1], m[2][2], m[2][3], m[3][1], m[3][2], "
			          "m[3][3]);");
			statement("adj[0][2] =  spvDet3x3(m[0][1], m[0][2], m[0][3], m[1][1], m[1][2], m[1][3], m[3][1], m[3][2], "
			          "m[3][3]);");
			statement("adj[0][3] = -spvDet3x3(m[0][1], m[0][2], m[0][3], m[1][1], m[1][2], m[1][3], m[2][1], m[2][2], "
			          "m[2][3]);");
			statement_no_indent("");
			statement("adj[1][0] = -spvDet3x3(m[1][0], m[1][2], m[1][3], m[2][0], m[2][2], m[2][3], m[3][0], m[3][2], "
			          "m[3][3]);");
			statement("adj[1][1] =  spvDet3x3(m[0][0], m[0][2], m[0][3], m[2][0], m[2][2], m[2][3], m[3][0], m[3][2], "
			          "m[3][3]);");
			statement("adj[1][2] = -spvDet3x3(m[0][0], m[0][2], m[0][3], m[1][0], m[1][2], m[1][3], m[3][0], m[3][2], "
			          "m[3][3]);");
			statement("adj[1][3] =  spvDet3x3(m[0][0], m[0][2], m[0][3], m[1][0], m[1][2], m[1][3], m[2][0], m[2][2], "
			          "m[2][3]);");
			statement_no_indent("");
			statement("adj[2][0] =  spvDet3x3(m[1][0], m[1][1], m[1][3], m[2][0], m[2][1], m[2][3], m[3][0], m[3][1], "
			          "m[3][3]);");
			statement("adj[2][1] = -spvDet3x3(m[0][0], m[0][1], m[0][3], m[2][0], m[2][1], m[2][3], m[3][0], m[3][1], "
			          "m[3][3]);");
			statement("adj[2][2] =  spvDet3x3(m[0][0], m[0][1], m[0][3], m[1][0], m[1][1], m[1][3], m[3][0], m[3][1], "
			          "m[3][3]);");
			statement("adj[2][3] = -spvDet3x3(m[0][0], m[0][1], m[0][3], m[1][0], m[1][1], m[1][3], m[2][0], m[2][1], "
			          "m[2][3]);");
			statement_no_indent("");
			statement("adj[3][0] = -spvDet3x3(m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2], m[3][0], m[3][1], "
			          "m[3][2]);");
			statement("adj[3][1] =  spvDet3x3(m[0][0], m[0][1], m[0][2], m[2][0], m[2][1], m[2][2], m[3][0], m[3][1], "
			          "m[3][2]);");
			statement("adj[3][2] = -spvDet3x3(m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[3][0], m[3][1], "
			          "m[3][2]);");
			statement("adj[3][3] =  spvDet3x3(m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], "
			          "m[2][2]);");
			statement_no_indent("");
			statement("// Calculate the determinant as a combination of the cofactors of the first row.");
			statement("float det = (adj[0][0] * m[0][0]) + (adj[0][1] * m[1][0]) + (adj[0][2] * m[2][0]) + (adj[0][3] "
			          "* m[3][0]);");
			statement_no_indent("");
			statement("// Divide the classical adjoint matrix by the determinant.");
			statement("// If determinant is zero, matrix is not invertable, so leave it unchanged.");
			statement("return (det != 0.0f) ? (adj * (1.0f / det)) : m;");
			end_scope();
			statement("");
			break;

		case SPVFuncImplInverse3x3:
			if (spv_function_implementations.count(SPVFuncImplInverse4x4) == 0)
			{
				statement("// Returns the determinant of a 2x2 matrix.");
				statement("inline float spvDet2x2(float a1, float a2, float b1, float b2)");
				begin_scope();
				statement("return a1 * b2 - b1 * a2;");
				end_scope();
				statement("");
			}

			statement("// Returns the inverse of a matrix, by using the algorithm of calculating the classical");
			statement("// adjoint and dividing by the determinant. The contents of the matrix are changed.");
			statement("float3x3 spvInverse3x3(float3x3 m)");
			begin_scope();
			statement("float3x3 adj;	// The adjoint matrix (inverse after dividing by determinant)");
			statement_no_indent("");
			statement("// Create the transpose of the cofactors, as the classical adjoint of the matrix.");
			statement("adj[0][0] =  spvDet2x2(m[1][1], m[1][2], m[2][1], m[2][2]);");
			statement("adj[0][1] = -spvDet2x2(m[0][1], m[0][2], m[2][1], m[2][2]);");
			statement("adj[0][2] =  spvDet2x2(m[0][1], m[0][2], m[1][1], m[1][2]);");
			statement_no_indent("");
			statement("adj[1][0] = -spvDet2x2(m[1][0], m[1][2], m[2][0], m[2][2]);");
			statement("adj[1][1] =  spvDet2x2(m[0][0], m[0][2], m[2][0], m[2][2]);");
			statement("adj[1][2] = -spvDet2x2(m[0][0], m[0][2], m[1][0], m[1][2]);");
			statement_no_indent("");
			statement("adj[2][0] =  spvDet2x2(m[1][0], m[1][1], m[2][0], m[2][1]);");
			statement("adj[2][1] = -spvDet2x2(m[0][0], m[0][1], m[2][0], m[2][1]);");
			statement("adj[2][2] =  spvDet2x2(m[0][0], m[0][1], m[1][0], m[1][1]);");
			statement_no_indent("");
			statement("// Calculate the determinant as a combination of the cofactors of the first row.");
			statement("float det = (adj[0][0] * m[0][0]) + (adj[0][1] * m[1][0]) + (adj[0][2] * m[2][0]);");
			statement_no_indent("");
			statement("// Divide the classical adjoint matrix by the determinant.");
			statement("// If determinant is zero, matrix is not invertable, so leave it unchanged.");
			statement("return (det != 0.0f) ? (adj * (1.0f / det)) : m;");
			end_scope();
			statement("");
			break;

		case SPVFuncImplInverse2x2:
			statement("// Returns the inverse of a matrix, by using the algorithm of calculating the classical");
			statement("// adjoint and dividing by the determinant. The contents of the matrix are changed.");
			statement("float2x2 spvInverse2x2(float2x2 m)");
			begin_scope();
			statement("float2x2 adj;	// The adjoint matrix (inverse after dividing by determinant)");
			statement_no_indent("");
			statement("// Create the transpose of the cofactors, as the classical adjoint of the matrix.");
			statement("adj[0][0] =  m[1][1];");
			statement("adj[0][1] = -m[0][1];");
			statement_no_indent("");
			statement("adj[1][0] = -m[1][0];");
			statement("adj[1][1] =  m[0][0];");
			statement_no_indent("");
			statement("// Calculate the determinant as a combination of the cofactors of the first row.");
			statement("float det = (adj[0][0] * m[0][0]) + (adj[0][1] * m[1][0]);");
			statement_no_indent("");
			statement("// Divide the classical adjoint matrix by the determinant.");
			statement("// If determinant is zero, matrix is not invertable, so leave it unchanged.");
			statement("return (det != 0.0f) ? (adj * (1.0f / det)) : m;");
			end_scope();
			statement("");
			break;

		case SPVFuncImplRowMajor2x3:
			statement("// Implementation of a conversion of matrix content from RowMajor to ColumnMajor organization.");
			statement("float2x3 spvConvertFromRowMajor2x3(float2x3 m)");
			begin_scope();
			statement("return float2x3(float3(m[0][0], m[0][2], m[1][1]), float3(m[0][1], m[1][0], m[1][2]));");
			end_scope();
			statement("");
			break;

		case SPVFuncImplRowMajor2x4:
			statement("// Implementation of a conversion of matrix content from RowMajor to ColumnMajor organization.");
			statement("float2x4 spvConvertFromRowMajor2x4(float2x4 m)");
			begin_scope();
			statement("return float2x4(float4(m[0][0], m[0][2], m[1][0], m[1][2]), float4(m[0][1], m[0][3], m[1][1], "
			          "m[1][3]));");
			end_scope();
			statement("");
			break;

		case SPVFuncImplRowMajor3x2:
			statement("// Implementation of a conversion of matrix content from RowMajor to ColumnMajor organization.");
			statement("float3x2 spvConvertFromRowMajor3x2(float3x2 m)");
			begin_scope();
			statement("return float3x2(float2(m[0][0], m[1][1]), float2(m[0][1], m[2][0]), float2(m[1][0], m[2][1]));");
			end_scope();
			statement("");
			break;

		case SPVFuncImplRowMajor3x4:
			statement("// Implementation of a conversion of matrix content from RowMajor to ColumnMajor organization.");
			statement("float3x4 spvConvertFromRowMajor3x4(float3x4 m)");
			begin_scope();
			statement("return float3x4(float4(m[0][0], m[0][3], m[1][2], m[2][1]), float4(m[0][1], m[1][0], m[1][3], "
			          "m[2][2]), float4(m[0][2], m[1][1], m[2][0], m[2][3]));");
			end_scope();
			statement("");
			break;

		case SPVFuncImplRowMajor4x2:
			statement("// Implementation of a conversion of matrix content from RowMajor to ColumnMajor organization.");
			statement("float4x2 spvConvertFromRowMajor4x2(float4x2 m)");
			begin_scope();
			statement("return float4x2(float2(m[0][0], m[2][0]), float2(m[0][1], m[2][1]), float2(m[1][0], m[3][0]), "
			          "float2(m[1][1], m[3][1]));");
			end_scope();
			statement("");
			break;

		case SPVFuncImplRowMajor4x3:
			statement("// Implementation of a conversion of matrix content from RowMajor to ColumnMajor organization.");
			statement("float4x3 spvConvertFromRowMajor4x3(float4x3 m)");
			begin_scope();
			statement("return float4x3(float3(m[0][0], m[1][1], m[2][2]), float3(m[0][1], m[1][2], m[3][0]), "
			          "float3(m[0][2], m[2][0], m[3][1]), float3(m[1][0], m[2][1], m[3][2]));");
			end_scope();
			statement("");
			break;

		case SPVFuncImplTextureSwizzle:
			statement("enum class spvSwizzle : uint");
			begin_scope();
			statement("none = 0,");
			statement("zero,");
			statement("one,");
			statement("red,");
			statement("green,");
			statement("blue,");
			statement("alpha");
			end_scope_decl();
			statement("");
			statement("template<typename T> struct spvRemoveReference { typedef T type; };");
			statement("template<typename T> struct spvRemoveReference<thread T&> { typedef T type; };");
			statement("template<typename T> struct spvRemoveReference<thread T&&> { typedef T type; };");
			statement("template<typename T> inline constexpr thread T&& spvForward(thread typename "
			          "spvRemoveReference<T>::type& x)");
			begin_scope();
			statement("return static_cast<thread T&&>(x);");
			end_scope();
			statement("template<typename T> inline constexpr thread T&& spvForward(thread typename "
			          "spvRemoveReference<T>::type&& x)");
			begin_scope();
			statement("return static_cast<thread T&&>(x);");
			end_scope();
			statement("");
			statement("template<typename T>");
			statement("inline T spvGetSwizzle(vec<T, 4> x, T c, spvSwizzle s)");
			begin_scope();
			statement("switch (s)");
			begin_scope();
			statement("case spvSwizzle::none:");
			statement("    return c;");
			statement("case spvSwizzle::zero:");
			statement("    return 0;");
			statement("case spvSwizzle::one:");
			statement("    return 1;");
			statement("case spvSwizzle::red:");
			statement("    return x.r;");
			statement("case spvSwizzle::green:");
			statement("    return x.g;");
			statement("case spvSwizzle::blue:");
			statement("    return x.b;");
			statement("case spvSwizzle::alpha:");
			statement("    return x.a;");
			end_scope();
			end_scope();
			statement("");
			statement("// Wrapper function that swizzles texture samples and fetches.");
			statement("template<typename T>");
			statement("inline vec<T, 4> spvTextureSwizzle(vec<T, 4> x, uint s)");
			begin_scope();
			statement("if (!s)");
			statement("    return x;");
			statement("return vec<T, 4>(spvGetSwizzle(x, x.r, spvSwizzle((s >> 0) & 0xFF)), "
			          "spvGetSwizzle(x, x.g, spvSwizzle((s >> 8) & 0xFF)), spvGetSwizzle(x, x.b, spvSwizzle((s >> 16) "
			          "& 0xFF)), "
			          "spvGetSwizzle(x, x.a, spvSwizzle((s >> 24) & 0xFF)));");
			end_scope();
			statement("");
			statement("template<typename T>");
			statement("inline T spvTextureSwizzle(T x, uint s)");
			begin_scope();
			statement("return spvTextureSwizzle(vec<T, 4>(x, 0, 0, 1), s).x;");
			end_scope();
			statement("");
			statement("// Wrapper function that swizzles texture gathers.");
			statement("template<typename T, typename Tex, typename... Ts>");
			statement(
			    "inline vec<T, 4> spvGatherSwizzle(sampler s, const thread Tex& t, Ts... params, component c, uint sw) "
			    "METAL_CONST_ARG(c)");
			begin_scope();
			statement("if (sw)");
			begin_scope();
			statement("switch (spvSwizzle((sw >> (uint(c) * 8)) & 0xFF))");
			begin_scope();
			statement("case spvSwizzle::none:");
			statement("    break;");
			statement("case spvSwizzle::zero:");
			statement("    return vec<T, 4>(0, 0, 0, 0);");
			statement("case spvSwizzle::one:");
			statement("    return vec<T, 4>(1, 1, 1, 1);");
			statement("case spvSwizzle::red:");
			statement("    return t.gather(s, spvForward<Ts>(params)..., component::x);");
			statement("case spvSwizzle::green:");
			statement("    return t.gather(s, spvForward<Ts>(params)..., component::y);");
			statement("case spvSwizzle::blue:");
			statement("    return t.gather(s, spvForward<Ts>(params)..., component::z);");
			statement("case spvSwizzle::alpha:");
			statement("    return t.gather(s, spvForward<Ts>(params)..., component::w);");
			end_scope();
			end_scope();
			// texture::gather insists on its component parameter being a constant
			// expression, so we need this silly workaround just to compile the shader.
			statement("switch (c)");
			begin_scope();
			statement("case component::x:");
			statement("    return t.gather(s, spvForward<Ts>(params)..., component::x);");
			statement("case component::y:");
			statement("    return t.gather(s, spvForward<Ts>(params)..., component::y);");
			statement("case component::z:");
			statement("    return t.gather(s, spvForward<Ts>(params)..., component::z);");
			statement("case component::w:");
			statement("    return t.gather(s, spvForward<Ts>(params)..., component::w);");
			end_scope();
			end_scope();
			statement("");
			statement("// Wrapper function that swizzles depth texture gathers.");
			statement("template<typename T, typename Tex, typename... Ts>");
			statement(
			    "inline vec<T, 4> spvGatherCompareSwizzle(sampler s, const thread Tex& t, Ts... params, uint sw) ");
			begin_scope();
			statement("if (sw)");
			begin_scope();
			statement("switch (spvSwizzle(sw & 0xFF))");
			begin_scope();
			statement("case spvSwizzle::none:");
			statement("case spvSwizzle::red:");
			statement("    break;");
			statement("case spvSwizzle::zero:");
			statement("case spvSwizzle::green:");
			statement("case spvSwizzle::blue:");
			statement("case spvSwizzle::alpha:");
			statement("    return vec<T, 4>(0, 0, 0, 0);");
			statement("case spvSwizzle::one:");
			statement("    return vec<T, 4>(1, 1, 1, 1);");
			end_scope();
			end_scope();
			statement("return t.gather_compare(s, spvForward<Ts>(params)...);");
			end_scope();
			statement("");
			break;

		case SPVFuncImplSubgroupBallot:
			statement("inline uint4 spvSubgroupBallot(bool value)");
			begin_scope();
			statement("simd_vote vote = simd_ballot(value);");
			statement("// simd_ballot() returns a 64-bit integer-like object, but");
			statement("// SPIR-V callers expect a uint4. We must convert.");
			statement("// FIXME: This won't include higher bits if Apple ever supports");
			statement("// 128 lanes in an SIMD-group.");
			statement("return uint4((uint)((simd_vote::vote_t)vote & 0xFFFFFFFF), (uint)(((simd_vote::vote_t)vote >> "
			          "32) & 0xFFFFFFFF), 0, 0);");
			end_scope();
			statement("");
			break;

		case SPVFuncImplSubgroupBallotBitExtract:
			statement("inline bool spvSubgroupBallotBitExtract(uint4 ballot, uint bit)");
			begin_scope();
			statement("return !!extract_bits(ballot[bit / 32], bit % 32, 1);");
			end_scope();
			statement("");
			break;

		case SPVFuncImplSubgroupBallotFindLSB:
			statement("inline uint spvSubgroupBallotFindLSB(uint4 ballot)");
			begin_scope();
			statement("return select(ctz(ballot.x), select(32 + ctz(ballot.y), select(64 + ctz(ballot.z), select(96 + "
			          "ctz(ballot.w), uint(-1), ballot.w == 0), ballot.z == 0), ballot.y == 0), ballot.x == 0);");
			end_scope();
			statement("");
			break;

		case SPVFuncImplSubgroupBallotFindMSB:
			statement("inline uint spvSubgroupBallotFindMSB(uint4 ballot)");
			begin_scope();
			statement("return select(128 - (clz(ballot.w) + 1), select(96 - (clz(ballot.z) + 1), select(64 - "
			          "(clz(ballot.y) + 1), select(32 - (clz(ballot.x) + 1), uint(-1), ballot.x == 0), ballot.y == 0), "
			          "ballot.z == 0), ballot.w == 0);");
			end_scope();
			statement("");
			break;

		case SPVFuncImplSubgroupBallotBitCount:
			statement("inline uint spvSubgroupBallotBitCount(uint4 ballot)");
			begin_scope();
			statement("return popcount(ballot.x) + popcount(ballot.y) + popcount(ballot.z) + popcount(ballot.w);");
			end_scope();
			statement("");
			statement("inline uint spvSubgroupBallotInclusiveBitCount(uint4 ballot, uint gl_SubgroupInvocationID)");
			begin_scope();
			statement("uint4 mask = uint4(extract_bits(0xFFFFFFFF, 0, min(gl_SubgroupInvocationID + 1, 32u)), "
			          "extract_bits(0xFFFFFFFF, 0, (uint)max((int)gl_SubgroupInvocationID + 1 - 32, 0)), "
			          "uint2(0));");
			statement("return spvSubgroupBallotBitCount(ballot & mask);");
			end_scope();
			statement("");
			statement("inline uint spvSubgroupBallotExclusiveBitCount(uint4 ballot, uint gl_SubgroupInvocationID)");
			begin_scope();
			statement("uint4 mask = uint4(extract_bits(0xFFFFFFFF, 0, min(gl_SubgroupInvocationID, 32u)), "
			          "extract_bits(0xFFFFFFFF, 0, (uint)max((int)gl_SubgroupInvocationID - 32, 0)), uint2(0));");
			statement("return spvSubgroupBallotBitCount(ballot & mask);");
			end_scope();
			statement("");
			break;

		case SPVFuncImplSubgroupAllEqual:
			// Metal doesn't provide a function to evaluate this directly. But, we can
			// implement this by comparing every thread's value to one thread's value
			// (in this case, the value of the first active thread). Then, by the transitive
			// property of equality, if all comparisons return true, then they are all equal.
			statement("template<typename T>");
			statement("inline bool spvSubgroupAllEqual(T value)");
			begin_scope();
			statement("return simd_all(value == simd_broadcast_first(value));");
			end_scope();
			statement("");
			statement("template<>");
			statement("inline bool spvSubgroupAllEqual(bool value)");
			begin_scope();
			statement("return simd_all(value) || !simd_any(value);");
			end_scope();
			statement("");
			break;

		default:
			break;
		}
	}
}

// Undefined global memory is not allowed in MSL.
// Declare constant and init to zeros. Use {}, as global constructors can break Metal.
void CompilerMSL::declare_undefined_values()
{
	bool emitted = false;
	ir.for_each_typed_id<SPIRUndef>([&](uint32_t, SPIRUndef &undef) {
		auto &type = this->get<SPIRType>(undef.basetype);
		statement("constant ", variable_decl(type, to_name(undef.self), undef.self), " = {};");
		emitted = true;
	});

	if (emitted)
		statement("");
}

void CompilerMSL::declare_constant_arrays()
{
	// MSL cannot declare arrays inline (except when declaring a variable), so we must move them out to
	// global constants directly, so we are able to use constants as variable expressions.
	bool emitted = false;

	ir.for_each_typed_id<SPIRConstant>([&](uint32_t, SPIRConstant &c) {
		if (c.specialization)
			return;

		auto &type = this->get<SPIRType>(c.constant_type);
		if (!type.array.empty())
		{
			auto name = to_name(c.self);
			statement("constant ", variable_decl(type, name), " = ", constant_expression(c), ";");
			emitted = true;
		}
	});

	if (emitted)
		statement("");
}

void CompilerMSL::emit_resources()
{
	declare_constant_arrays();
	declare_undefined_values();

	// Emit the special [[stage_in]] and [[stage_out]] interface blocks which we created.
	emit_interface_block(stage_out_var_id);
	emit_interface_block(patch_stage_out_var_id);
	emit_interface_block(stage_in_var_id);
	emit_interface_block(patch_stage_in_var_id);
}

// Emit declarations for the specialization Metal function constants
void CompilerMSL::emit_specialization_constants_and_structs()
{
	SpecializationConstant wg_x, wg_y, wg_z;
	uint32_t workgroup_size_id = get_work_group_size_specialization_constants(wg_x, wg_y, wg_z);
	bool emitted = false;

	unordered_set<uint32_t> declared_structs;

	for (auto &id_ : ir.ids_for_constant_or_type)
	{
		auto &id = ir.ids[id_];

		if (id.get_type() == TypeConstant)
		{
			auto &c = id.get<SPIRConstant>();

			if (c.self == workgroup_size_id)
			{
				// TODO: This can be expressed as a [[threads_per_threadgroup]] input semantic, but we need to know
				// the work group size at compile time in SPIR-V, and [[threads_per_threadgroup]] would need to be passed around as a global.
				// The work group size may be a specialization constant.
				statement("constant uint3 ", builtin_to_glsl(BuiltInWorkgroupSize, StorageClassWorkgroup),
				          " [[maybe_unused]] = ", constant_expression(get<SPIRConstant>(workgroup_size_id)), ";");
				emitted = true;
			}
			else if (c.specialization)
			{
				auto &type = get<SPIRType>(c.constant_type);
				string sc_type_name = type_to_glsl(type);
				string sc_name = to_name(c.self);
				string sc_tmp_name = sc_name + "_tmp";

				// Function constants are only supported in MSL 1.2 and later.
				// If we don't support it just declare the "default" directly.
				// This "default" value can be overridden to the true specialization constant by the API user.
				// Specialization constants which are used as array length expressions cannot be function constants in MSL,
				// so just fall back to macros.
				if (msl_options.supports_msl_version(1, 2) && has_decoration(c.self, DecorationSpecId) &&
				    !c.is_used_as_array_length)
				{
					uint32_t constant_id = get_decoration(c.self, DecorationSpecId);
					// Only scalar, non-composite values can be function constants.
					statement("constant ", sc_type_name, " ", sc_tmp_name, " [[function_constant(", constant_id,
					          ")]];");
					statement("constant ", sc_type_name, " ", sc_name, " = is_function_constant_defined(", sc_tmp_name,
					          ") ? ", sc_tmp_name, " : ", constant_expression(c), ";");
				}
				else if (has_decoration(c.self, DecorationSpecId))
				{
					// Fallback to macro overrides.
					c.specialization_constant_macro_name =
					    constant_value_macro_name(get_decoration(c.self, DecorationSpecId));

					statement("#ifndef ", c.specialization_constant_macro_name);
					statement("#define ", c.specialization_constant_macro_name, " ", constant_expression(c));
					statement("#endif");
					statement("constant ", sc_type_name, " ", sc_name, " = ", c.specialization_constant_macro_name,
					          ";");
				}
				else
				{
					// Composite specialization constants must be built from other specialization constants.
					statement("constant ", sc_type_name, " ", sc_name, " = ", constant_expression(c), ";");
				}
				emitted = true;
			}
		}
		else if (id.get_type() == TypeConstantOp)
		{
			auto &c = id.get<SPIRConstantOp>();
			auto &type = get<SPIRType>(c.basetype);
			auto name = to_name(c.self);
			statement("constant ", variable_decl(type, name), " = ", constant_op_expression(c), ";");
			emitted = true;
		}
		else if (id.get_type() == TypeType)
		{
			// Output non-builtin interface structs. These include local function structs
			// and structs nested within uniform and read-write buffers.
			auto &type = id.get<SPIRType>();
			uint32_t type_id = type.self;

			bool is_struct = (type.basetype == SPIRType::Struct) && type.array.empty();
			bool is_block =
			    has_decoration(type.self, DecorationBlock) || has_decoration(type.self, DecorationBufferBlock);

			bool is_builtin_block = is_block && is_builtin_type(type);
			bool is_declarable_struct = is_struct && !is_builtin_block;

			// We'll declare this later.
			if (stage_out_var_id && get_stage_out_struct_type().self == type_id)
				is_declarable_struct = false;
			if (patch_stage_out_var_id && get_patch_stage_out_struct_type().self == type_id)
				is_declarable_struct = false;
			if (stage_in_var_id && get_stage_in_struct_type().self == type_id)
				is_declarable_struct = false;
			if (patch_stage_in_var_id && get_patch_stage_in_struct_type().self == type_id)
				is_declarable_struct = false;

			// Align and emit declarable structs...but avoid declaring each more than once.
			if (is_declarable_struct && declared_structs.count(type_id) == 0)
			{
				if (emitted)
					statement("");
				emitted = false;

				declared_structs.insert(type_id);

				if (has_extended_decoration(type_id, SPIRVCrossDecorationPacked))
					align_struct(type);

				// Make sure we declare the underlying struct type, and not the "decorated" type with pointers, etc.
				emit_struct(get<SPIRType>(type_id));
			}
		}
	}

	if (emitted)
		statement("");
}

void CompilerMSL::emit_binary_unord_op(uint32_t result_type, uint32_t result_id, uint32_t op0, uint32_t op1,
                                       const char *op)
{
	bool forward = should_forward(op0) && should_forward(op1);
	emit_op(result_type, result_id,
	        join("(isunordered(", to_enclosed_unpacked_expression(op0), ", ", to_enclosed_unpacked_expression(op1),
	             ") || ", to_enclosed_unpacked_expression(op0), " ", op, " ", to_enclosed_unpacked_expression(op1),
	             ")"),
	        forward);

	inherit_expression_dependencies(result_id, op0);
	inherit_expression_dependencies(result_id, op1);
}

bool CompilerMSL::emit_tessellation_access_chain(const uint32_t *ops, uint32_t length)
{
	// If this is a per-vertex output, remap it to the I/O array buffer.
	auto *var = maybe_get<SPIRVariable>(ops[2]);
	BuiltIn bi_type = BuiltIn(get_decoration(ops[2], DecorationBuiltIn));
	if (var &&
	    (var->storage == StorageClassInput ||
	     (get_execution_model() == ExecutionModelTessellationControl && var->storage == StorageClassOutput)) &&
	    !(has_decoration(ops[2], DecorationPatch) || is_patch_block(get_variable_data_type(*var))) &&
	    (!is_builtin_variable(*var) || bi_type == BuiltInPosition || bi_type == BuiltInPointSize ||
	     bi_type == BuiltInClipDistance || bi_type == BuiltInCullDistance ||
	     get_variable_data_type(*var).basetype == SPIRType::Struct))
	{
		AccessChainMeta meta;
		SmallVector<uint32_t> indices;
		uint32_t next_id = ir.increase_bound_by(2);

		indices.reserve(length - 3 + 1);
		uint32_t type_id = next_id++;
		SPIRType new_uint_type;
		new_uint_type.basetype = SPIRType::UInt;
		new_uint_type.width = 32;
		set<SPIRType>(type_id, new_uint_type);

		indices.push_back(ops[3]);

		uint32_t const_mbr_id = next_id++;
		uint32_t index = get_extended_decoration(ops[2], SPIRVCrossDecorationInterfaceMemberIndex);
		uint32_t ptr = var->storage == StorageClassInput ? stage_in_ptr_var_id : stage_out_ptr_var_id;
		if (var->storage == StorageClassInput || has_decoration(get_variable_element_type(*var).self, DecorationBlock))
		{
			uint32_t i = 4;
			auto *type = &get_variable_element_type(*var);
			if (index == uint32_t(-1) && length >= 5)
			{
				// Maybe this is a struct type in the input class, in which case
				// we put it as a decoration on the corresponding member.
				index = get_extended_member_decoration(ops[2], get_constant(ops[4]).scalar(),
				                                       SPIRVCrossDecorationInterfaceMemberIndex);
				assert(index != uint32_t(-1));
				i++;
				type = &get<SPIRType>(type->member_types[get_constant(ops[4]).scalar()]);
			}
			// In this case, we flattened structures and arrays, so now we have to
			// combine the following indices. If we encounter a non-constant index,
			// we're hosed.
			for (; i < length; ++i)
			{
				if (!is_array(*type) && !is_matrix(*type) && type->basetype != SPIRType::Struct)
					break;

				auto &c = get_constant(ops[i]);
				index += c.scalar();
				if (type->parent_type)
					type = &get<SPIRType>(type->parent_type);
				else if (type->basetype == SPIRType::Struct)
					type = &get<SPIRType>(type->member_types[c.scalar()]);
			}
			// If the access chain terminates at a composite type, the composite
			// itself might be copied. In that case, we must unflatten it.
			if (is_matrix(*type) || is_array(*type) || type->basetype == SPIRType::Struct)
			{
				std::string temp_name = join(to_name(var->self), "_", ops[1]);
				statement(variable_decl(*type, temp_name, var->self), ";");
				// Set up the initializer for this temporary variable.
				indices.push_back(const_mbr_id);
				if (type->basetype == SPIRType::Struct)
				{
					for (uint32_t j = 0; j < type->member_types.size(); j++)
					{
						index = get_extended_member_decoration(ops[2], j, SPIRVCrossDecorationInterfaceMemberIndex);
						const auto &mbr_type = get<SPIRType>(type->member_types[j]);
						if (is_matrix(mbr_type))
						{
							for (uint32_t k = 0; k < mbr_type.columns; k++, index++)
							{
								set<SPIRConstant>(const_mbr_id, type_id, index, false);
								auto e = access_chain(ptr, indices.data(), uint32_t(indices.size()), mbr_type, nullptr,
								                      true);
								statement(temp_name, ".", to_member_name(*type, j), "[", k, "] = ", e, ";");
							}
						}
						else if (is_array(mbr_type))
						{
							for (uint32_t k = 0; k < mbr_type.array[0]; k++, index++)
							{
								set<SPIRConstant>(const_mbr_id, type_id, index, false);
								auto e = access_chain(ptr, indices.data(), uint32_t(indices.size()), mbr_type, nullptr,
								                      true);
								statement(temp_name, ".", to_member_name(*type, j), "[", k, "] = ", e, ";");
							}
						}
						else
						{
							set<SPIRConstant>(const_mbr_id, type_id, index, false);
							auto e =
							    access_chain(ptr, indices.data(), uint32_t(indices.size()), mbr_type, nullptr, true);
							statement(temp_name, ".", to_member_name(*type, j), " = ", e, ";");
						}
					}
				}
				else if (is_matrix(*type))
				{
					for (uint32_t j = 0; j < type->columns; j++, index++)
					{
						set<SPIRConstant>(const_mbr_id, type_id, index, false);
						auto e = access_chain(ptr, indices.data(), uint32_t(indices.size()), *type, nullptr, true);
						statement(temp_name, "[", j, "] = ", e, ";");
					}
				}
				else // Must be an array
				{
					assert(is_array(*type));
					for (uint32_t j = 0; j < type->array[0]; j++, index++)
					{
						set<SPIRConstant>(const_mbr_id, type_id, index, false);
						auto e = access_chain(ptr, indices.data(), uint32_t(indices.size()), *type, nullptr, true);
						statement(temp_name, "[", j, "] = ", e, ";");
					}
				}

				// This needs to be a variable instead of an expression so we don't
				// try to dereference this as a variable pointer.
				set<SPIRVariable>(ops[1], ops[0], var->storage);
				ir.meta[ops[1]] = ir.meta[ops[2]];
				set_name(ops[1], temp_name);
				if (has_decoration(var->self, DecorationInvariant))
					set_decoration(ops[1], DecorationInvariant);
				for (uint32_t j = 2; j < length; j++)
					inherit_expression_dependencies(ops[1], ops[j]);
				return true;
			}
			else
			{
				set<SPIRConstant>(const_mbr_id, type_id, index, false);
				indices.push_back(const_mbr_id);

				if (i < length)
					indices.insert(indices.end(), ops + i, ops + length);
			}
		}
		else
		{
			assert(index != uint32_t(-1));
			set<SPIRConstant>(const_mbr_id, type_id, index, false);
			indices.push_back(const_mbr_id);

			indices.insert(indices.end(), ops + 4, ops + length);
		}

		// We use the pointer to the base of the input/output array here,
		// so this is always a pointer chain.
		auto e = access_chain(ptr, indices.data(), uint32_t(indices.size()), get<SPIRType>(ops[0]), &meta, true);
		auto &expr = set<SPIRExpression>(ops[1], move(e), ops[0], should_forward(ops[2]));
		expr.loaded_from = var->self;
		expr.need_transpose = meta.need_transpose;
		expr.access_chain = true;

		// Mark the result as being packed if necessary.
		if (meta.storage_is_packed)
			set_extended_decoration(ops[1], SPIRVCrossDecorationPacked);
		if (meta.storage_packed_type != 0)
			set_extended_decoration(ops[1], SPIRVCrossDecorationPackedType, meta.storage_packed_type);
		if (meta.storage_is_invariant)
			set_decoration(ops[1], DecorationInvariant);

		for (uint32_t i = 2; i < length; i++)
		{
			inherit_expression_dependencies(ops[1], ops[i]);
			add_implied_read_expression(expr, ops[i]);
		}

		return true;
	}

	// If this is the inner tessellation level, and we're tessellating triangles,
	// drop the last index. It isn't an array in this case, so we can't have an
	// array reference here. We need to make this ID a variable instead of an
	// expression so we don't try to dereference it as a variable pointer.
	// Don't do this if the index is a constant 1, though. We need to drop stores
	// to that one.
	auto *m = ir.find_meta(var ? var->self : 0);
	if (get_execution_model() == ExecutionModelTessellationControl && var && m &&
	    m->decoration.builtin_type == BuiltInTessLevelInner && get_entry_point().flags.get(ExecutionModeTriangles))
	{
		auto *c = maybe_get<SPIRConstant>(ops[3]);
		if (c && c->scalar() == 1)
			return false;
		auto &dest_var = set<SPIRVariable>(ops[1], *var);
		dest_var.basetype = ops[0];
		ir.meta[ops[1]] = ir.meta[ops[2]];
		inherit_expression_dependencies(ops[1], ops[2]);
		return true;
	}

	return false;
}

bool CompilerMSL::is_out_of_bounds_tessellation_level(uint32_t id_lhs)
{
	if (!get_entry_point().flags.get(ExecutionModeTriangles))
		return false;

	// In SPIR-V, TessLevelInner always has two elements and TessLevelOuter always has
	// four. This is true even if we are tessellating triangles. This allows clients
	// to use a single tessellation control shader with multiple tessellation evaluation
	// shaders.
	// In Metal, however, only the first element of TessLevelInner and the first three
	// of TessLevelOuter are accessible. This stems from how in Metal, the tessellation
	// levels must be stored to a dedicated buffer in a particular format that depends
	// on the patch type. Therefore, in Triangles mode, any access to the second
	// inner level or the fourth outer level must be dropped.
	const auto *e = maybe_get<SPIRExpression>(id_lhs);
	if (!e || !e->access_chain)
		return false;
	BuiltIn builtin = BuiltIn(get_decoration(e->loaded_from, DecorationBuiltIn));
	if (builtin != BuiltInTessLevelInner && builtin != BuiltInTessLevelOuter)
		return false;
	auto *c = maybe_get<SPIRConstant>(e->implied_read_expressions[1]);
	if (!c)
		return false;
	return (builtin == BuiltInTessLevelInner && c->scalar() == 1) ||
	       (builtin == BuiltInTessLevelOuter && c->scalar() == 3);
}

// Override for MSL-specific syntax instructions
void CompilerMSL::emit_instruction(const Instruction &instruction)
{
#define MSL_BOP(op) emit_binary_op(ops[0], ops[1], ops[2], ops[3], #op)
#define MSL_BOP_CAST(op, type) \
	emit_binary_op_cast(ops[0], ops[1], ops[2], ops[3], #op, type, opcode_is_sign_invariant(opcode))
#define MSL_UOP(op) emit_unary_op(ops[0], ops[1], ops[2], #op)
#define MSL_QFOP(op) emit_quaternary_func_op(ops[0], ops[1], ops[2], ops[3], ops[4], ops[5], #op)
#define MSL_TFOP(op) emit_trinary_func_op(ops[0], ops[1], ops[2], ops[3], ops[4], #op)
#define MSL_BFOP(op) emit_binary_func_op(ops[0], ops[1], ops[2], ops[3], #op)
#define MSL_BFOP_CAST(op, type) \
	emit_binary_func_op_cast(ops[0], ops[1], ops[2], ops[3], #op, type, opcode_is_sign_invariant(opcode))
#define MSL_UFOP(op) emit_unary_func_op(ops[0], ops[1], ops[2], #op)
#define MSL_UNORD_BOP(op) emit_binary_unord_op(ops[0], ops[1], ops[2], ops[3], #op)

	auto ops = stream(instruction);
	auto opcode = static_cast<Op>(instruction.op);

	// If we need to do implicit bitcasts, make sure we do it with the correct type.
	uint32_t integer_width = get_integer_width_for_instruction(instruction);
	auto int_type = to_signed_basetype(integer_width);
	auto uint_type = to_unsigned_basetype(integer_width);

	switch (opcode)
	{

	// Comparisons
	case OpIEqual:
		MSL_BOP_CAST(==, int_type);
		break;

	case OpLogicalEqual:
	case OpFOrdEqual:
		MSL_BOP(==);
		break;

	case OpINotEqual:
		MSL_BOP_CAST(!=, int_type);
		break;

	case OpLogicalNotEqual:
	case OpFOrdNotEqual:
		MSL_BOP(!=);
		break;

	case OpUGreaterThan:
		MSL_BOP_CAST(>, uint_type);
		break;

	case OpSGreaterThan:
		MSL_BOP_CAST(>, int_type);
		break;

	case OpFOrdGreaterThan:
		MSL_BOP(>);
		break;

	case OpUGreaterThanEqual:
		MSL_BOP_CAST(>=, uint_type);
		break;

	case OpSGreaterThanEqual:
		MSL_BOP_CAST(>=, int_type);
		break;

	case OpFOrdGreaterThanEqual:
		MSL_BOP(>=);
		break;

	case OpULessThan:
		MSL_BOP_CAST(<, uint_type);
		break;

	case OpSLessThan:
		MSL_BOP_CAST(<, int_type);
		break;

	case OpFOrdLessThan:
		MSL_BOP(<);
		break;

	case OpULessThanEqual:
		MSL_BOP_CAST(<=, uint_type);
		break;

	case OpSLessThanEqual:
		MSL_BOP_CAST(<=, int_type);
		break;

	case OpFOrdLessThanEqual:
		MSL_BOP(<=);
		break;

	case OpFUnordEqual:
		MSL_UNORD_BOP(==);
		break;

	case OpFUnordNotEqual:
		MSL_UNORD_BOP(!=);
		break;

	case OpFUnordGreaterThan:
		MSL_UNORD_BOP(>);
		break;

	case OpFUnordGreaterThanEqual:
		MSL_UNORD_BOP(>=);
		break;

	case OpFUnordLessThan:
		MSL_UNORD_BOP(<);
		break;

	case OpFUnordLessThanEqual:
		MSL_UNORD_BOP(<=);
		break;

	// Derivatives
	case OpDPdx:
	case OpDPdxFine:
	case OpDPdxCoarse:
		MSL_UFOP(dfdx);
		register_control_dependent_expression(ops[1]);
		break;

	case OpDPdy:
	case OpDPdyFine:
	case OpDPdyCoarse:
		MSL_UFOP(dfdy);
		register_control_dependent_expression(ops[1]);
		break;

	case OpFwidth:
	case OpFwidthCoarse:
	case OpFwidthFine:
		MSL_UFOP(fwidth);
		register_control_dependent_expression(ops[1]);
		break;

	// Bitfield
	case OpBitFieldInsert:
		MSL_QFOP(insert_bits);
		break;

	case OpBitFieldSExtract:
	case OpBitFieldUExtract:
		MSL_TFOP(extract_bits);
		break;

	case OpBitReverse:
		MSL_UFOP(reverse_bits);
		break;

	case OpBitCount:
		MSL_UFOP(popcount);
		break;

	case OpFRem:
		MSL_BFOP(fmod);
		break;

	// Atomics
	case OpAtomicExchange:
	{
		uint32_t result_type = ops[0];
		uint32_t id = ops[1];
		uint32_t ptr = ops[2];
		uint32_t mem_sem = ops[4];
		uint32_t val = ops[5];
		emit_atomic_func_op(result_type, id, "atomic_exchange_explicit", mem_sem, mem_sem, false, ptr, val);
		break;
	}

	case OpAtomicCompareExchange:
	{
		uint32_t result_type = ops[0];
		uint32_t id = ops[1];
		uint32_t ptr = ops[2];
		uint32_t mem_sem_pass = ops[4];
		uint32_t mem_sem_fail = ops[5];
		uint32_t val = ops[6];
		uint32_t comp = ops[7];
		emit_atomic_func_op(result_type, id, "atomic_compare_exchange_weak_explicit", mem_sem_pass, mem_sem_fail, true,
		                    ptr, comp, true, false, val);
		break;
	}

	case OpAtomicCompareExchangeWeak:
		SPIRV_CROSS_THROW("OpAtomicCompareExchangeWeak is only supported in kernel profile.");

	case OpAtomicLoad:
	{
		uint32_t result_type = ops[0];
		uint32_t id = ops[1];
		uint32_t ptr = ops[2];
		uint32_t mem_sem = ops[4];
		emit_atomic_func_op(result_type, id, "atomic_load_explicit", mem_sem, mem_sem, false, ptr, 0);
		break;
	}

	case OpAtomicStore:
	{
		uint32_t result_type = expression_type(ops[0]).self;
		uint32_t id = ops[0];
		uint32_t ptr = ops[0];
		uint32_t mem_sem = ops[2];
		uint32_t val = ops[3];
		emit_atomic_func_op(result_type, id, "atomic_store_explicit", mem_sem, mem_sem, false, ptr, val);
		break;
	}

#define MSL_AFMO_IMPL(op, valsrc, valconst)                                                                      \
	do                                                                                                           \
	{                                                                                                            \
		uint32_t result_type = ops[0];                                                                           \
		uint32_t id = ops[1];                                                                                    \
		uint32_t ptr = ops[2];                                                                                   \
		uint32_t mem_sem = ops[4];                                                                               \
		uint32_t val = valsrc;                                                                                   \
		emit_atomic_func_op(result_type, id, "atomic_fetch_" #op "_explicit", mem_sem, mem_sem, false, ptr, val, \
		                    false, valconst);                                                                    \
	} while (false)

#define MSL_AFMO(op) MSL_AFMO_IMPL(op, ops[5], false)
#define MSL_AFMIO(op) MSL_AFMO_IMPL(op, 1, true)

	case OpAtomicIIncrement:
		MSL_AFMIO(add);
		break;

	case OpAtomicIDecrement:
		MSL_AFMIO(sub);
		break;

	case OpAtomicIAdd:
		MSL_AFMO(add);
		break;

	case OpAtomicISub:
		MSL_AFMO(sub);
		break;

	case OpAtomicSMin:
	case OpAtomicUMin:
		MSL_AFMO(min);
		break;

	case OpAtomicSMax:
	case OpAtomicUMax:
		MSL_AFMO(max);
		break;

	case OpAtomicAnd:
		MSL_AFMO(and);
		break;

	case OpAtomicOr:
		MSL_AFMO(or);
		break;

	case OpAtomicXor:
		MSL_AFMO(xor);
		break;

	// Images

	// Reads == Fetches in Metal
	case OpImageRead:
	{
		// Mark that this shader reads from this image
		uint32_t img_id = ops[2];
		auto &type = expression_type(img_id);
		if (type.image.dim != DimSubpassData)
		{
			auto *p_var = maybe_get_backing_variable(img_id);
			if (p_var && has_decoration(p_var->self, DecorationNonReadable))
			{
				unset_decoration(p_var->self, DecorationNonReadable);
				force_recompile();
			}
		}

		emit_texture_op(instruction);
		break;
	}

	case OpImageWrite:
	{
		uint32_t img_id = ops[0];
		uint32_t coord_id = ops[1];
		uint32_t texel_id = ops[2];
		const uint32_t *opt = &ops[3];
		uint32_t length = instruction.length - 3;

		// Bypass pointers because we need the real image struct
		auto &type = expression_type(img_id);
		auto &img_type = get<SPIRType>(type.self);

		// Ensure this image has been marked as being written to and force a
		// recommpile so that the image type output will include write access
		auto *p_var = maybe_get_backing_variable(img_id);
		if (p_var && has_decoration(p_var->self, DecorationNonWritable))
		{
			unset_decoration(p_var->self, DecorationNonWritable);
			force_recompile();
		}

		bool forward = false;
		uint32_t bias = 0;
		uint32_t lod = 0;
		uint32_t flags = 0;

		if (length)
		{
			flags = *opt++;
			length--;
		}

		auto test = [&](uint32_t &v, uint32_t flag) {
			if (length && (flags & flag))
			{
				v = *opt++;
				length--;
			}
		};

		test(bias, ImageOperandsBiasMask);
		test(lod, ImageOperandsLodMask);

		auto &texel_type = expression_type(texel_id);
		auto store_type = texel_type;
		store_type.vecsize = 4;

		statement(join(
		    to_expression(img_id), ".write(", remap_swizzle(store_type, texel_type.vecsize, to_expression(texel_id)),
		    ", ",
		    to_function_args(img_id, img_type, true, false, false, coord_id, 0, 0, 0, 0, lod, 0, 0, 0, 0, 0, &forward),
		    ");"));

		if (p_var && variable_storage_is_aliased(*p_var))
			flush_all_aliased_variables();

		break;
	}

	case OpImageQuerySize:
	case OpImageQuerySizeLod:
	{
		uint32_t rslt_type_id = ops[0];
		auto &rslt_type = get<SPIRType>(rslt_type_id);

		uint32_t id = ops[1];

		uint32_t img_id = ops[2];
		string img_exp = to_expression(img_id);
		auto &img_type = expression_type(img_id);
		Dim img_dim = img_type.image.dim;
		bool img_is_array = img_type.image.arrayed;

		if (img_type.basetype != SPIRType::Image)
			SPIRV_CROSS_THROW("Invalid type for OpImageQuerySize.");

		string lod;
		if (opcode == OpImageQuerySizeLod)
		{
			// LOD index defaults to zero, so don't bother outputing level zero index
			string decl_lod = to_expression(ops[3]);
			if (decl_lod != "0")
				lod = decl_lod;
		}

		string expr = type_to_glsl(rslt_type) + "(";
		expr += img_exp + ".get_width(" + lod + ")";

		if (img_dim == Dim2D || img_dim == DimCube || img_dim == Dim3D)
			expr += ", " + img_exp + ".get_height(" + lod + ")";

		if (img_dim == Dim3D)
			expr += ", " + img_exp + ".get_depth(" + lod + ")";

		if (img_is_array)
			expr += ", " + img_exp + ".get_array_size()";

		expr += ")";

		emit_op(rslt_type_id, id, expr, should_forward(img_id));

		break;
	}

	case OpImageQueryLod:
		SPIRV_CROSS_THROW("MSL does not support textureQueryLod().");

#define MSL_ImgQry(qrytype)                                                                 \
	do                                                                                      \
	{                                                                                       \
		uint32_t rslt_type_id = ops[0];                                                     \
		auto &rslt_type = get<SPIRType>(rslt_type_id);                                      \
		uint32_t id = ops[1];                                                               \
		uint32_t img_id = ops[2];                                                           \
		string img_exp = to_expression(img_id);                                             \
		string expr = type_to_glsl(rslt_type) + "(" + img_exp + ".get_num_" #qrytype "())"; \
		emit_op(rslt_type_id, id, expr, should_forward(img_id));                            \
	} while (false)

	case OpImageQueryLevels:
		MSL_ImgQry(mip_levels);
		break;

	case OpImageQuerySamples:
		MSL_ImgQry(samples);
		break;

	case OpImage:
	{
		uint32_t result_type = ops[0];
		uint32_t id = ops[1];
		auto *combined = maybe_get<SPIRCombinedImageSampler>(ops[2]);

		if (combined)
		{
			auto &e = emit_op(result_type, id, to_expression(combined->image), true, true);
			auto *var = maybe_get_backing_variable(combined->image);
			if (var)
				e.loaded_from = var->self;
		}
		else
		{
			auto &e = emit_op(result_type, id, to_expression(ops[2]), true, true);
			auto *var = maybe_get_backing_variable(ops[2]);
			if (var)
				e.loaded_from = var->self;
		}
		break;
	}

	case OpImageTexelPointer:
		SPIRV_CROSS_THROW("MSL does not support atomic operations on images or texel buffers.");

	// Casting
	case OpQuantizeToF16:
	{
		uint32_t result_type = ops[0];
		uint32_t id = ops[1];
		uint32_t arg = ops[2];

		string exp;
		auto &type = get<SPIRType>(result_type);

		switch (type.vecsize)
		{
		case 1:
			exp = join("float(half(", to_expression(arg), "))");
			break;
		case 2:
			exp = join("float2(half2(", to_expression(arg), "))");
			break;
		case 3:
			exp = join("float3(half3(", to_expression(arg), "))");
			break;
		case 4:
			exp = join("float4(half4(", to_expression(arg), "))");
			break;
		default:
			SPIRV_CROSS_THROW("Illegal argument to OpQuantizeToF16.");
		}

		emit_op(result_type, id, exp, should_forward(arg));
		break;
	}

	case OpInBoundsAccessChain:
	case OpAccessChain:
	case OpPtrAccessChain:
		if (is_tessellation_shader())
		{
			if (!emit_tessellation_access_chain(ops, instruction.length))
				CompilerGLSL::emit_instruction(instruction);
		}
		else
			CompilerGLSL::emit_instruction(instruction);
		break;

	case OpStore:
		if (is_out_of_bounds_tessellation_level(ops[0]))
			break;

		if (maybe_emit_array_assignment(ops[0], ops[1]))
			break;

		CompilerGLSL::emit_instruction(instruction);
		break;

	// Compute barriers
	case OpMemoryBarrier:
		emit_barrier(0, ops[0], ops[1]);
		break;

	case OpControlBarrier:
		// In GLSL a memory barrier is often followed by a control barrier.
		// But in MSL, memory barriers are also control barriers, so don't
		// emit a simple control barrier if a memory barrier has just been emitted.
		if (previous_instruction_opcode != OpMemoryBarrier)
			emit_barrier(ops[0], ops[1], ops[2]);
		break;

	case OpVectorTimesMatrix:
	case OpMatrixTimesVector:
	{
		// If the matrix needs transpose and it is square or packed, just flip the multiply order.
		uint32_t mtx_id = ops[opcode == OpMatrixTimesVector ? 2 : 3];
		auto *e = maybe_get<SPIRExpression>(mtx_id);
		auto &t = expression_type(mtx_id);
		bool is_packed = has_extended_decoration(mtx_id, SPIRVCrossDecorationPacked);
		if (e && e->need_transpose && (t.columns == t.vecsize || is_packed))
		{
			e->need_transpose = false;

			// This is important for matrices. Packed matrices
			// are generally transposed, so unpacking using a constructor argument
			// will result in an error.
			// The simplest solution for now is to just avoid unpacking the matrix in this operation.
			unset_extended_decoration(mtx_id, SPIRVCrossDecorationPacked);

			emit_binary_op(ops[0], ops[1], ops[3], ops[2], "*");
			if (is_packed)
				set_extended_decoration(mtx_id, SPIRVCrossDecorationPacked);
			e->need_transpose = true;
		}
		else
			MSL_BOP(*);
		break;
	}

		// OpOuterProduct

	case OpIAddCarry:
	case OpISubBorrow:
	{
		uint32_t result_type = ops[0];
		uint32_t result_id = ops[1];
		uint32_t op0 = ops[2];
		uint32_t op1 = ops[3];
		forced_temporaries.insert(result_id);
		auto &type = get<SPIRType>(result_type);
		statement(variable_decl(type, to_name(result_id)), ";");
		set<SPIRExpression>(result_id, to_name(result_id), result_type, true);

		auto &res_type = get<SPIRType>(type.member_types[1]);
		if (opcode == OpIAddCarry)
		{
			statement(to_expression(result_id), ".", to_member_name(type, 0), " = ", to_enclosed_expression(op0), " + ",
			          to_enclosed_expression(op1), ";");
			statement(to_expression(result_id), ".", to_member_name(type, 1), " = select(", type_to_glsl(res_type),
			          "(1), ", type_to_glsl(res_type), "(0), ", to_expression(result_id), ".", to_member_name(type, 0),
			          " >= max(", to_expression(op0), ", ", to_expression(op1), "));");
		}
		else
		{
			statement(to_expression(result_id), ".", to_member_name(type, 0), " = ", to_enclosed_expression(op0), " - ",
			          to_enclosed_expression(op1), ";");
			statement(to_expression(result_id), ".", to_member_name(type, 1), " = select(", type_to_glsl(res_type),
			          "(1), ", type_to_glsl(res_type), "(0), ", to_enclosed_expression(op0),
			          " >= ", to_enclosed_expression(op1), ");");
		}
		break;
	}

	case OpUMulExtended:
	case OpSMulExtended:
	{
		uint32_t result_type = ops[0];
		uint32_t result_id = ops[1];
		uint32_t op0 = ops[2];
		uint32_t op1 = ops[3];
		forced_temporaries.insert(result_id);
		auto &type = get<SPIRType>(result_type);
		statement(variable_decl(type, to_name(result_id)), ";");
		set<SPIRExpression>(result_id, to_name(result_id), result_type, true);

		statement(to_expression(result_id), ".", to_member_name(type, 0), " = ", to_enclosed_expression(op0), " * ",
		          to_enclosed_expression(op1), ";");
		statement(to_expression(result_id), ".", to_member_name(type, 1), " = mulhi(", to_expression(op0), ", ",
		          to_expression(op1), ");");
		break;
	}

	default:
		CompilerGLSL::emit_instruction(instruction);
		break;
	}

	previous_instruction_opcode = opcode;
}

void CompilerMSL::emit_barrier(uint32_t id_exe_scope, uint32_t id_mem_scope, uint32_t id_mem_sem)
{
	if (get_execution_model() != ExecutionModelGLCompute && get_execution_model() != ExecutionModelTessellationControl)
		return;

	uint32_t exe_scope = id_exe_scope ? get<SPIRConstant>(id_exe_scope).scalar() : uint32_t(ScopeInvocation);
	uint32_t mem_scope = id_mem_scope ? get<SPIRConstant>(id_mem_scope).scalar() : uint32_t(ScopeInvocation);
	// Use the wider of the two scopes (smaller value)
	exe_scope = min(exe_scope, mem_scope);

	string bar_stmt;
	if ((msl_options.is_ios() && msl_options.supports_msl_version(1, 2)) || msl_options.supports_msl_version(2))
		bar_stmt = exe_scope < ScopeSubgroup ? "threadgroup_barrier" : "simdgroup_barrier";
	else
		bar_stmt = "threadgroup_barrier";
	bar_stmt += "(";

	uint32_t mem_sem = id_mem_sem ? get<SPIRConstant>(id_mem_sem).scalar() : uint32_t(MemorySemanticsMaskNone);

	// Use the | operator to combine flags if we can.
	if (msl_options.supports_msl_version(1, 2))
	{
		string mem_flags = "";
		// For tesc shaders, this also affects objects in the Output storage class.
		// Since in Metal, these are placed in a device buffer, we have to sync device memory here.
		if (get_execution_model() == ExecutionModelTessellationControl ||
		    (mem_sem & (MemorySemanticsUniformMemoryMask | MemorySemanticsCrossWorkgroupMemoryMask)))
			mem_flags += "mem_flags::mem_device";
		if (mem_sem & (MemorySemanticsSubgroupMemoryMask | MemorySemanticsWorkgroupMemoryMask |
		               MemorySemanticsAtomicCounterMemoryMask))
		{
			if (!mem_flags.empty())
				mem_flags += " | ";
			mem_flags += "mem_flags::mem_threadgroup";
		}
		if (mem_sem & MemorySemanticsImageMemoryMask)
		{
			if (!mem_flags.empty())
				mem_flags += " | ";
			mem_flags += "mem_flags::mem_texture";
		}

		if (mem_flags.empty())
			mem_flags = "mem_flags::mem_none";

		bar_stmt += mem_flags;
	}
	else
	{
		if ((mem_sem & (MemorySemanticsUniformMemoryMask | MemorySemanticsCrossWorkgroupMemoryMask)) &&
		    (mem_sem & (MemorySemanticsSubgroupMemoryMask | MemorySemanticsWorkgroupMemoryMask |
		                MemorySemanticsAtomicCounterMemoryMask)))
			bar_stmt += "mem_flags::mem_device_and_threadgroup";
		else if (mem_sem & (MemorySemanticsUniformMemoryMask | MemorySemanticsCrossWorkgroupMemoryMask))
			bar_stmt += "mem_flags::mem_device";
		else if (mem_sem & (MemorySemanticsSubgroupMemoryMask | MemorySemanticsWorkgroupMemoryMask |
		                    MemorySemanticsAtomicCounterMemoryMask))
			bar_stmt += "mem_flags::mem_threadgroup";
		else if (mem_sem & MemorySemanticsImageMemoryMask)
			bar_stmt += "mem_flags::mem_texture";
		else
			bar_stmt += "mem_flags::mem_none";
	}

	if (msl_options.is_ios() && (msl_options.supports_msl_version(2) && !msl_options.supports_msl_version(2, 1)))
	{
		bar_stmt += ", ";

		switch (mem_scope)
		{
		case ScopeCrossDevice:
		case ScopeDevice:
			bar_stmt += "memory_scope_device";
			break;

		case ScopeSubgroup:
		case ScopeInvocation:
			bar_stmt += "memory_scope_simdgroup";
			break;

		case ScopeWorkgroup:
		default:
			bar_stmt += "memory_scope_threadgroup";
			break;
		}
	}

	bar_stmt += ");";

	statement(bar_stmt);

	assert(current_emitting_block);
	flush_control_dependent_expressions(current_emitting_block->self);
	flush_all_active_variables();
}

void CompilerMSL::emit_array_copy(const string &lhs, uint32_t rhs_id)
{
	// Assignment from an array initializer is fine.
	auto &type = expression_type(rhs_id);
	auto *var = maybe_get_backing_variable(rhs_id);

	// Unfortunately, we cannot template on address space in MSL,
	// so explicit address space redirection it is ...
	bool is_constant = false;
	if (ir.ids[rhs_id].get_type() == TypeConstant)
	{
		is_constant = true;
	}
	else if (var && var->remapped_variable && var->statically_assigned &&
	         ir.ids[var->static_expression].get_type() == TypeConstant)
	{
		is_constant = true;
	}

	// For the case where we have OpLoad triggering an array copy,
	// we cannot easily detect this case ahead of time since it's
	// context dependent. We might have to force a recompile here
	// if this is the only use of array copies in our shader.
	if (type.array.size() > 1)
	{
		if (type.array.size() > SPVFuncImplArrayCopyMultidimMax)
			SPIRV_CROSS_THROW("Cannot support this many dimensions for arrays of arrays.");
		auto func = static_cast<SPVFuncImpl>(SPVFuncImplArrayCopyMultidimBase + type.array.size());
		if (spv_function_implementations.count(func) == 0)
		{
			spv_function_implementations.insert(func);
			suppress_missing_prototypes = true;
			force_recompile();
		}
	}
	else if (spv_function_implementations.count(SPVFuncImplArrayCopy) == 0)
	{
		spv_function_implementations.insert(SPVFuncImplArrayCopy);
		suppress_missing_prototypes = true;
		force_recompile();
	}

	const char *tag = is_constant ? "FromConstant" : "FromStack";
	statement("spvArrayCopy", tag, type.array.size(), "(", lhs, ", ", to_expression(rhs_id), ");");
}

// Since MSL does not allow arrays to be copied via simple variable assignment,
// if the LHS and RHS represent an assignment of an entire array, it must be
// implemented by calling an array copy function.
// Returns whether the struct assignment was emitted.
bool CompilerMSL::maybe_emit_array_assignment(uint32_t id_lhs, uint32_t id_rhs)
{
	// We only care about assignments of an entire array
	auto &type = expression_type(id_rhs);
	if (type.array.size() == 0)
		return false;

	auto *var = maybe_get<SPIRVariable>(id_lhs);

	// Is this a remapped, static constant? Don't do anything.
	if (var && var->remapped_variable && var->statically_assigned)
		return true;

	if (ir.ids[id_rhs].get_type() == TypeConstant && var && var->deferred_declaration)
	{
		// Special case, if we end up declaring a variable when assigning the constant array,
		// we can avoid the copy by directly assigning the constant expression.
		// This is likely necessary to be able to use a variable as a true look-up table, as it is unlikely
		// the compiler will be able to optimize the spvArrayCopy() into a constant LUT.
		// After a variable has been declared, we can no longer assign constant arrays in MSL unfortunately.
		statement(to_expression(id_lhs), " = ", constant_expression(get<SPIRConstant>(id_rhs)), ";");
		return true;
	}

	// Ensure the LHS variable has been declared
	auto *p_v_lhs = maybe_get_backing_variable(id_lhs);
	if (p_v_lhs)
		flush_variable_declaration(p_v_lhs->self);

	emit_array_copy(to_expression(id_lhs), id_rhs);
	register_write(id_lhs);

	return true;
}

// Emits one of the atomic functions. In MSL, the atomic functions operate on pointers
void CompilerMSL::emit_atomic_func_op(uint32_t result_type, uint32_t result_id, const char *op, uint32_t mem_order_1,
                                      uint32_t mem_order_2, bool has_mem_order_2, uint32_t obj, uint32_t op1,
                                      bool op1_is_pointer, bool op1_is_literal, uint32_t op2)
{
	forced_temporaries.insert(result_id);

	string exp = string(op) + "(";

	auto &type = get_pointee_type(expression_type(obj));
	exp += "(volatile ";
	auto *var = maybe_get_backing_variable(obj);
	if (!var)
		SPIRV_CROSS_THROW("No backing variable for atomic operation.");
	exp += get_argument_address_space(*var);
	exp += " atomic_";
	exp += type_to_glsl(type);
	exp += "*)";

	exp += "&";
	exp += to_enclosed_expression(obj);

	bool is_atomic_compare_exchange_strong = op1_is_pointer && op1;

	if (is_atomic_compare_exchange_strong)
	{
		assert(strcmp(op, "atomic_compare_exchange_weak_explicit") == 0);
		assert(op2);
		assert(has_mem_order_2);
		exp += ", &";
		exp += to_name(result_id);
		exp += ", ";
		exp += to_expression(op2);
		exp += ", ";
		exp += get_memory_order(mem_order_1);
		exp += ", ";
		exp += get_memory_order(mem_order_2);
		exp += ")";

		// MSL only supports the weak atomic compare exchange, so emit a CAS loop here.
		// The MSL function returns false if the atomic write fails OR the comparison test fails,
		// so we must validate that it wasn't the comparison test that failed before continuing
		// the CAS loop, otherwise it will loop infinitely, with the comparison test always failing.
		// The function updates the comparitor value from the memory value, so the additional
		// comparison test evaluates the memory value against the expected value.
		statement(variable_decl(type, to_name(result_id)), ";");
		statement("do");
		begin_scope();
		statement(to_name(result_id), " = ", to_expression(op1), ";");
		end_scope_decl(join("while (!", exp, " && ", to_name(result_id), " == ", to_enclosed_expression(op1), ")"));
		set<SPIRExpression>(result_id, to_name(result_id), result_type, true);
	}
	else
	{
		assert(strcmp(op, "atomic_compare_exchange_weak_explicit") != 0);
		if (op1)
		{
			if (op1_is_literal)
				exp += join(", ", op1);
			else
				exp += ", " + to_expression(op1);
		}
		if (op2)
			exp += ", " + to_expression(op2);

		exp += string(", ") + get_memory_order(mem_order_1);
		if (has_mem_order_2)
			exp += string(", ") + get_memory_order(mem_order_2);

		exp += ")";
		emit_op(result_type, result_id, exp, false);
	}

	flush_all_atomic_capable_variables();
}

// Metal only supports relaxed memory order for now
const char *CompilerMSL::get_memory_order(uint32_t)
{
	return "memory_order_relaxed";
}

// Override for MSL-specific extension syntax instructions
void CompilerMSL::emit_glsl_op(uint32_t result_type, uint32_t id, uint32_t eop, const uint32_t *args, uint32_t count)
{
	auto op = static_cast<GLSLstd450>(eop);

	// If we need to do implicit bitcasts, make sure we do it with the correct type.
	uint32_t integer_width = get_integer_width_for_glsl_instruction(op, args, count);
	auto int_type = to_signed_basetype(integer_width);
	auto uint_type = to_unsigned_basetype(integer_width);

	switch (op)
	{
	case GLSLstd450Atan2:
		emit_binary_func_op(result_type, id, args[0], args[1], "atan2");
		break;
	case GLSLstd450InverseSqrt:
		emit_unary_func_op(result_type, id, args[0], "rsqrt");
		break;
	case GLSLstd450RoundEven:
		emit_unary_func_op(result_type, id, args[0], "rint");
		break;

	case GLSLstd450FindSMsb:
		emit_unary_func_op_cast(result_type, id, args[0], "findSMSB", int_type, int_type);
		break;

	case GLSLstd450FindUMsb:
		emit_unary_func_op_cast(result_type, id, args[0], "findUMSB", uint_type, uint_type);
		break;

	case GLSLstd450PackSnorm4x8:
		emit_unary_func_op(result_type, id, args[0], "pack_float_to_snorm4x8");
		break;
	case GLSLstd450PackUnorm4x8:
		emit_unary_func_op(result_type, id, args[0], "pack_float_to_unorm4x8");
		break;
	case GLSLstd450PackSnorm2x16:
		emit_unary_func_op(result_type, id, args[0], "pack_float_to_snorm2x16");
		break;
	case GLSLstd450PackUnorm2x16:
		emit_unary_func_op(result_type, id, args[0], "pack_float_to_unorm2x16");
		break;

	case GLSLstd450PackHalf2x16:
	{
		auto expr = join("as_type<uint>(half2(", to_expression(args[0]), "))");
		emit_op(result_type, id, expr, should_forward(args[0]));
		inherit_expression_dependencies(id, args[0]);
		break;
	}

	case GLSLstd450UnpackSnorm4x8:
		emit_unary_func_op(result_type, id, args[0], "unpack_snorm4x8_to_float");
		break;
	case GLSLstd450UnpackUnorm4x8:
		emit_unary_func_op(result_type, id, args[0], "unpack_unorm4x8_to_float");
		break;
	case GLSLstd450UnpackSnorm2x16:
		emit_unary_func_op(result_type, id, args[0], "unpack_snorm2x16_to_float");
		break;
	case GLSLstd450UnpackUnorm2x16:
		emit_unary_func_op(result_type, id, args[0], "unpack_unorm2x16_to_float");
		break;

	case GLSLstd450UnpackHalf2x16:
	{
		auto expr = join("float2(as_type<half2>(", to_expression(args[0]), "))");
		emit_op(result_type, id, expr, should_forward(args[0]));
		inherit_expression_dependencies(id, args[0]);
		break;
	}

	case GLSLstd450PackDouble2x32:
		emit_unary_func_op(result_type, id, args[0], "unsupported_GLSLstd450PackDouble2x32"); // Currently unsupported
		break;
	case GLSLstd450UnpackDouble2x32:
		emit_unary_func_op(result_type, id, args[0], "unsupported_GLSLstd450UnpackDouble2x32"); // Currently unsupported
		break;

	case GLSLstd450MatrixInverse:
	{
		auto &mat_type = get<SPIRType>(result_type);
		switch (mat_type.columns)
		{
		case 2:
			emit_unary_func_op(result_type, id, args[0], "spvInverse2x2");
			break;
		case 3:
			emit_unary_func_op(result_type, id, args[0], "spvInverse3x3");
			break;
		case 4:
			emit_unary_func_op(result_type, id, args[0], "spvInverse4x4");
			break;
		default:
			break;
		}
		break;
	}

	case GLSLstd450FMin:
		// If the result type isn't float, don't bother calling the specific
		// precise::/fast:: version. Metal doesn't have those for half and
		// double types.
		if (get<SPIRType>(result_type).basetype != SPIRType::Float)
			emit_binary_func_op(result_type, id, args[0], args[1], "min");
		else
			emit_binary_func_op(result_type, id, args[0], args[1], "fast::min");
		break;

	case GLSLstd450FMax:
		if (get<SPIRType>(result_type).basetype != SPIRType::Float)
			emit_binary_func_op(result_type, id, args[0], args[1], "max");
		else
			emit_binary_func_op(result_type, id, args[0], args[1], "fast::max");
		break;

	case GLSLstd450FClamp:
		// TODO: If args[1] is 0 and args[2] is 1, emit a saturate() call.
		if (get<SPIRType>(result_type).basetype != SPIRType::Float)
			emit_trinary_func_op(result_type, id, args[0], args[1], args[2], "clamp");
		else
			emit_trinary_func_op(result_type, id, args[0], args[1], args[2], "fast::clamp");
		break;

	case GLSLstd450NMin:
		if (get<SPIRType>(result_type).basetype != SPIRType::Float)
			emit_binary_func_op(result_type, id, args[0], args[1], "min");
		else
			emit_binary_func_op(result_type, id, args[0], args[1], "precise::min");
		break;

	case GLSLstd450NMax:
		if (get<SPIRType>(result_type).basetype != SPIRType::Float)
			emit_binary_func_op(result_type, id, args[0], args[1], "max");
		else
			emit_binary_func_op(result_type, id, args[0], args[1], "precise::max");
		break;

	case GLSLstd450NClamp:
		// TODO: If args[1] is 0 and args[2] is 1, emit a saturate() call.
		if (get<SPIRType>(result_type).basetype != SPIRType::Float)
			emit_trinary_func_op(result_type, id, args[0], args[1], args[2], "clamp");
		else
			emit_trinary_func_op(result_type, id, args[0], args[1], args[2], "precise::clamp");
		break;

		// TODO:
		//        GLSLstd450InterpolateAtCentroid (centroid_no_perspective qualifier)
		//        GLSLstd450InterpolateAtSample (sample_no_perspective qualifier)
		//        GLSLstd450InterpolateAtOffset

	default:
		CompilerGLSL::emit_glsl_op(result_type, id, eop, args, count);
		break;
	}
}

// Emit a structure declaration for the specified interface variable.
void CompilerMSL::emit_interface_block(uint32_t ib_var_id)
{
	if (ib_var_id)
	{
		auto &ib_var = get<SPIRVariable>(ib_var_id);
		auto &ib_type = get_variable_data_type(ib_var);
		assert(ib_type.basetype == SPIRType::Struct && !ib_type.member_types.empty());
		emit_struct(ib_type);
	}
}

// Emits the declaration signature of the specified function.
// If this is the entry point function, Metal-specific return value and function arguments are added.
void CompilerMSL::emit_function_prototype(SPIRFunction &func, const Bitset &)
{
	if (func.self != ir.default_entry_point)
		add_function_overload(func);

	local_variable_names = resource_names;
	string decl;

	processing_entry_point = (func.self == ir.default_entry_point);

	auto &type = get<SPIRType>(func.return_type);

	if (type.array.empty())
	{
		decl += func_type_decl(type);
	}
	else
	{
		// We cannot return arrays in MSL, so "return" through an out variable.
		decl = "void";
	}

	decl += " ";
	decl += to_name(func.self);
	decl += "(";

	if (!type.array.empty())
	{
		// Fake arrays returns by writing to an out array instead.
		decl += "thread ";
		decl += type_to_glsl(type);
		decl += " (&SPIRV_Cross_return_value)";
		decl += type_to_array_glsl(type);
		if (!func.arguments.empty())
			decl += ", ";
	}

	if (processing_entry_point)
	{
		if (msl_options.argument_buffers)
			decl += entry_point_args_argument_buffer(!func.arguments.empty());
		else
			decl += entry_point_args_classic(!func.arguments.empty());

		// If entry point function has variables that require early declaration,
		// ensure they each have an empty initializer, creating one if needed.
		// This is done at this late stage because the initialization expression
		// is cleared after each compilation pass.
		for (auto var_id : vars_needing_early_declaration)
		{
			auto &ed_var = get<SPIRVariable>(var_id);
			uint32_t &initializer = ed_var.initializer;
			if (!initializer)
				initializer = ir.increase_bound_by(1);

			// Do not override proper initializers.
			if (ir.ids[initializer].get_type() == TypeNone || ir.ids[initializer].get_type() == TypeExpression)
				set<SPIRExpression>(ed_var.initializer, "{}", ed_var.basetype, true);
		}
	}

	for (auto &arg : func.arguments)
	{
		uint32_t name_id = arg.id;

		auto *var = maybe_get<SPIRVariable>(arg.id);
		if (var)
		{
			// If we need to modify the name of the variable, make sure we modify the original variable.
			// Our alias is just a shadow variable.
			if (arg.alias_global_variable && var->basevariable)
				name_id = var->basevariable;

			var->parameter = &arg; // Hold a pointer to the parameter so we can invalidate the readonly field if needed.
		}

		add_local_variable_name(name_id);

		decl += argument_decl(arg);

		// Manufacture automatic sampler arg for SampledImage texture
		auto &arg_type = get<SPIRType>(arg.type);
		if (arg_type.basetype == SPIRType::SampledImage && arg_type.image.dim != DimBuffer)
			decl += join(", thread const ", sampler_type(arg_type), " ", to_sampler_expression(arg.id));

		// Manufacture automatic swizzle arg.
		if (msl_options.swizzle_texture_samples && has_sampled_images && is_sampled_image_type(arg_type))
		{
			bool arg_is_array = !arg_type.array.empty();
			decl += join(", constant uint32_t", arg_is_array ? "* " : "& ", to_swizzle_expression(arg.id));
		}

		if (&arg != &func.arguments.back())
			decl += ", ";
	}

	decl += ")";
	statement(decl);
}

// Returns the texture sampling function string for the specified image and sampling characteristics.
string CompilerMSL::to_function_name(uint32_t img, const SPIRType &imgtype, bool is_fetch, bool is_gather, bool, bool,
                                     bool has_offset, bool, bool has_dref, uint32_t)
{
	// Special-case gather. We have to alter the component being looked up
	// in the swizzle case.
	if (msl_options.swizzle_texture_samples && is_gather)
	{
		string fname = imgtype.image.depth ? "spvGatherCompareSwizzle" : "spvGatherSwizzle";
		fname += "<" + type_to_glsl(get<SPIRType>(imgtype.image.type)) + ", metal::" + type_to_glsl(imgtype);
		// Add the arg types ourselves. Yes, this sucks, but Clang can't
		// deduce template pack parameters in the middle of an argument list.
		switch (imgtype.image.dim)
		{
		case Dim2D:
			fname += ", float2";
			if (imgtype.image.arrayed)
				fname += ", uint";
			if (imgtype.image.depth)
				fname += ", float";
			if (!imgtype.image.depth || has_offset)
				fname += ", int2";
			break;
		case DimCube:
			fname += ", float3";
			if (imgtype.image.arrayed)
				fname += ", uint";
			if (imgtype.image.depth)
				fname += ", float";
			break;
		default:
			SPIRV_CROSS_THROW("Invalid texture dimension for gather op.");
		}
		fname += ">";
		return fname;
	}

	auto *combined = maybe_get<SPIRCombinedImageSampler>(img);

	// Texture reference
	string fname = to_expression(combined ? combined->image : img) + ".";
	if (msl_options.swizzle_texture_samples && !is_gather && is_sampled_image_type(imgtype))
		fname = "spvTextureSwizzle(" + fname;

	// Texture function and sampler
	if (is_fetch)
		fname += "read";
	else if (is_gather)
		fname += "gather";
	else
		fname += "sample";

	if (has_dref)
		fname += "_compare";

	return fname;
}

// Returns the function args for a texture sampling function for the specified image and sampling characteristics.
string CompilerMSL::to_function_args(uint32_t img, const SPIRType &imgtype, bool is_fetch, bool is_gather, bool is_proj,
                                     uint32_t coord, uint32_t, uint32_t dref, uint32_t grad_x, uint32_t grad_y,
                                     uint32_t lod, uint32_t coffset, uint32_t offset, uint32_t bias, uint32_t comp,
                                     uint32_t sample, bool *p_forward)
{
	string farg_str;
	if (!is_fetch)
		farg_str += to_sampler_expression(img);

	if (msl_options.swizzle_texture_samples && is_gather)
	{
		if (!farg_str.empty())
			farg_str += ", ";

		auto *combined = maybe_get<SPIRCombinedImageSampler>(img);
		farg_str += to_expression(combined ? combined->image : img);
	}

	// Texture coordinates
	bool forward = should_forward(coord);
	auto coord_expr = to_enclosed_expression(coord);
	auto &coord_type = expression_type(coord);
	bool coord_is_fp = type_is_floating_point(coord_type);
	bool is_cube_fetch = false;

	string tex_coords = coord_expr;
	uint32_t alt_coord_component = 0;

	switch (imgtype.image.dim)
	{

	case Dim1D:
		if (coord_type.vecsize > 1)
			tex_coords = enclose_expression(tex_coords) + ".x";

		if (is_fetch)
			tex_coords = "uint(" + round_fp_tex_coords(tex_coords, coord_is_fp) + ")";

		alt_coord_component = 1;
		break;

	case DimBuffer:
		if (coord_type.vecsize > 1)
			tex_coords = enclose_expression(tex_coords) + ".x";

		if (msl_options.texture_buffer_native)
		{
			tex_coords = "uint(" + round_fp_tex_coords(tex_coords, coord_is_fp) + ")";
		}
		else
		{
			// Metal texel buffer textures are 2D, so convert 1D coord to 2D.
			if (is_fetch)
				tex_coords = "spvTexelBufferCoord(" + round_fp_tex_coords(tex_coords, coord_is_fp) + ")";
		}

		alt_coord_component = 1;
		break;

	case DimSubpassData:
		if (imgtype.image.ms)
			tex_coords = "uint2(gl_FragCoord.xy)";
		else
			tex_coords = join("uint2(gl_FragCoord.xy), 0");
		break;

	case Dim2D:
		if (coord_type.vecsize > 2)
			tex_coords = enclose_expression(tex_coords) + ".xy";

		if (is_fetch)
			tex_coords = "uint2(" + round_fp_tex_coords(tex_coords, coord_is_fp) + ")";

		alt_coord_component = 2;
		break;

	case Dim3D:
		if (coord_type.vecsize > 3)
			tex_coords = enclose_expression(tex_coords) + ".xyz";

		if (is_fetch)
			tex_coords = "uint3(" + round_fp_tex_coords(tex_coords, coord_is_fp) + ")";

		alt_coord_component = 3;
		break;

	case DimCube:
		if (is_fetch)
		{
			is_cube_fetch = true;
			tex_coords += ".xy";
			tex_coords = "uint2(" + round_fp_tex_coords(tex_coords, coord_is_fp) + ")";
		}
		else
		{
			if (coord_type.vecsize > 3)
				tex_coords = enclose_expression(tex_coords) + ".xyz";
		}

		alt_coord_component = 3;
		break;

	default:
		break;
	}

	if (is_fetch && offset)
	{
		// Fetch offsets must be applied directly to the coordinate.
		forward = forward && should_forward(offset);
		auto &type = expression_type(offset);
		if (type.basetype != SPIRType::UInt)
			tex_coords += " + " + bitcast_expression(SPIRType::UInt, offset);
		else
			tex_coords += " + " + to_enclosed_expression(offset);
	}
	else if (is_fetch && coffset)
	{
		// Fetch offsets must be applied directly to the coordinate.
		forward = forward && should_forward(coffset);
		auto &type = expression_type(coffset);
		if (type.basetype != SPIRType::UInt)
			tex_coords += " + " + bitcast_expression(SPIRType::UInt, coffset);
		else
			tex_coords += " + " + to_enclosed_expression(coffset);
	}

	// If projection, use alt coord as divisor
	if (is_proj)
		tex_coords += " / " + to_extract_component_expression(coord, alt_coord_component);

	if (!farg_str.empty())
		farg_str += ", ";
	farg_str += tex_coords;

	// If fetch from cube, add face explicitly
	if (is_cube_fetch)
	{
		// Special case for cube arrays, face and layer are packed in one dimension.
		if (imgtype.image.arrayed)
			farg_str += ", uint(" + to_extract_component_expression(coord, 2) + ") % 6u";
		else
			farg_str += ", uint(" + round_fp_tex_coords(to_extract_component_expression(coord, 2), coord_is_fp) + ")";
	}

	// If array, use alt coord
	if (imgtype.image.arrayed)
	{
		// Special case for cube arrays, face and layer are packed in one dimension.
		if (imgtype.image.dim == DimCube && is_fetch)
			farg_str += ", uint(" + to_extract_component_expression(coord, 2) + ") / 6u";
		else
			farg_str += ", uint(" +
			            round_fp_tex_coords(to_extract_component_expression(coord, alt_coord_component), coord_is_fp) +
			            ")";
	}

	// Depth compare reference value
	if (dref)
	{
		forward = forward && should_forward(dref);
		farg_str += ", ";

		if (is_proj)
			farg_str +=
			    to_enclosed_expression(dref) + " / " + to_extract_component_expression(coord, alt_coord_component);
		else
			farg_str += to_expression(dref);

		if (msl_options.is_macos() && (grad_x || grad_y))
		{
			// For sample compare, MSL does not support gradient2d for all targets (only iOS apparently according to docs).
			// However, the most common case here is to have a constant gradient of 0, as that is the only way to express
			// LOD == 0 in GLSL with sampler2DArrayShadow (cascaded shadow mapping).
			// We will detect a compile-time constant 0 value for gradient and promote that to level(0) on MSL.
			bool constant_zero_x = !grad_x || expression_is_constant_null(grad_x);
			bool constant_zero_y = !grad_y || expression_is_constant_null(grad_y);
			if (constant_zero_x && constant_zero_y)
			{
				lod = 0;
				grad_x = 0;
				grad_y = 0;
				farg_str += ", level(0)";
			}
			else
			{
				SPIRV_CROSS_THROW("Using non-constant 0.0 gradient() qualifier for sample_compare. This is not "
				                  "supported in MSL macOS.");
			}
		}

		if (msl_options.is_macos() && bias)
		{
			// Bias is not supported either on macOS with sample_compare.
			// Verify it is compile-time zero, and drop the argument.
			if (expression_is_constant_null(bias))
			{
				bias = 0;
			}
			else
			{
				SPIRV_CROSS_THROW(
				    "Using non-constant 0.0 bias() qualifier for sample_compare. This is not supported in MSL macOS.");
			}
		}
	}

	// LOD Options
	// Metal does not support LOD for 1D textures.
	if (bias && imgtype.image.dim != Dim1D)
	{
		forward = forward && should_forward(bias);
		farg_str += ", bias(" + to_expression(bias) + ")";
	}

	// Metal does not support LOD for 1D textures.
	if (lod && imgtype.image.dim != Dim1D)
	{
		forward = forward && should_forward(lod);
		if (is_fetch)
		{
			farg_str += ", " + to_expression(lod);
		}
		else
		{
			farg_str += ", level(" + to_expression(lod) + ")";
		}
	}
	else if (is_fetch && !lod && imgtype.image.dim != Dim1D && imgtype.image.dim != DimBuffer && !imgtype.image.ms &&
	         imgtype.image.sampled != 2)
	{
		// Lod argument is optional in OpImageFetch, but we require a LOD value, pick 0 as the default.
		// Check for sampled type as well, because is_fetch is also used for OpImageRead in MSL.
		farg_str += ", 0";
	}

	// Metal does not support LOD for 1D textures.
	if ((grad_x || grad_y) && imgtype.image.dim != Dim1D)
	{
		forward = forward && should_forward(grad_x);
		forward = forward && should_forward(grad_y);
		string grad_opt;
		switch (imgtype.image.dim)
		{
		case Dim2D:
			grad_opt = "2d";
			break;
		case Dim3D:
			grad_opt = "3d";
			break;
		case DimCube:
			grad_opt = "cube";
			break;
		default:
			grad_opt = "unsupported_gradient_dimension";
			break;
		}
		farg_str += ", gradient" + grad_opt + "(" + to_expression(grad_x) + ", " + to_expression(grad_y) + ")";
	}

	// Add offsets
	string offset_expr;
	if (coffset && !is_fetch)
	{
		forward = forward && should_forward(coffset);
		offset_expr = to_expression(coffset);
	}
	else if (offset && !is_fetch)
	{
		forward = forward && should_forward(offset);
		offset_expr = to_expression(offset);
	}

	if (!offset_expr.empty())
	{
		switch (imgtype.image.dim)
		{
		case Dim2D:
			if (coord_type.vecsize > 2)
				offset_expr = enclose_expression(offset_expr) + ".xy";

			farg_str += ", " + offset_expr;
			break;

		case Dim3D:
			if (coord_type.vecsize > 3)
				offset_expr = enclose_expression(offset_expr) + ".xyz";

			farg_str += ", " + offset_expr;
			break;

		default:
			break;
		}
	}

	if (comp)
	{
		// If 2D has gather component, ensure it also has an offset arg
		if (imgtype.image.dim == Dim2D && offset_expr.empty())
			farg_str += ", int2(0)";

		forward = forward && should_forward(comp);
		farg_str += ", " + to_component_argument(comp);
	}

	if (sample)
	{
		farg_str += ", ";
		farg_str += to_expression(sample);
	}

	if (msl_options.swizzle_texture_samples && is_sampled_image_type(imgtype))
	{
		// Add the swizzle constant from the swizzle buffer.
		if (!is_gather)
			farg_str += ")";
		farg_str += ", " + to_swizzle_expression(img);
		used_swizzle_buffer = true;
	}

	*p_forward = forward;

	return farg_str;
}

// If the texture coordinates are floating point, invokes MSL round() function to round them.
string CompilerMSL::round_fp_tex_coords(string tex_coords, bool coord_is_fp)
{
	return coord_is_fp ? ("round(" + tex_coords + ")") : tex_coords;
}

// Returns a string to use in an image sampling function argument.
// The ID must be a scalar constant.
string CompilerMSL::to_component_argument(uint32_t id)
{
	if (ir.ids[id].get_type() != TypeConstant)
	{
		SPIRV_CROSS_THROW("ID " + to_string(id) + " is not an OpConstant.");
		return "component::x";
	}

	uint32_t component_index = get<SPIRConstant>(id).scalar();
	switch (component_index)
	{
	case 0:
		return "component::x";
	case 1:
		return "component::y";
	case 2:
		return "component::z";
	case 3:
		return "component::w";

	default:
		SPIRV_CROSS_THROW("The value (" + to_string(component_index) + ") of OpConstant ID " + to_string(id) +
		                  " is not a valid Component index, which must be one of 0, 1, 2, or 3.");
		return "component::x";
	}
}

// Establish sampled image as expression object and assign the sampler to it.
void CompilerMSL::emit_sampled_image_op(uint32_t result_type, uint32_t result_id, uint32_t image_id, uint32_t samp_id)
{
	set<SPIRCombinedImageSampler>(result_id, result_type, image_id, samp_id);
}

// Returns a string representation of the ID, usable as a function arg.
// Manufacture automatic sampler arg for SampledImage texture.
string CompilerMSL::to_func_call_arg(uint32_t id)
{
	string arg_str;

	auto *c = maybe_get<SPIRConstant>(id);
	if (c && !get<SPIRType>(c->constant_type).array.empty())
	{
		// If we are passing a constant array directly to a function for some reason,
		// the callee will expect an argument in thread const address space
		// (since we can only bind to arrays with references in MSL).
		// To resolve this, we must emit a copy in this address space.
		// This kind of code gen should be rare enough that performance is not a real concern.
		// Inline the SPIR-V to avoid this kind of suboptimal codegen.
		//
		// We risk calling this inside a continue block (invalid code),
		// so just create a thread local copy in the current function.
		arg_str = join("_", id, "_array_copy");
		auto &constants = current_function->constant_arrays_needed_on_stack;
		auto itr = find(begin(constants), end(constants), id);
		if (itr == end(constants))
		{
			force_recompile();
			constants.push_back(id);
		}
	}
	else
		arg_str = CompilerGLSL::to_func_call_arg(id);

	// Manufacture automatic sampler arg if the arg is a SampledImage texture.
	auto &type = expression_type(id);
	if (type.basetype == SPIRType::SampledImage && type.image.dim != DimBuffer)
	{
		// Need to check the base variable in case we need to apply a qualified alias.
		uint32_t var_id = 0;
		auto *sampler_var = maybe_get<SPIRVariable>(id);
		if (sampler_var)
			var_id = sampler_var->basevariable;

		arg_str += ", " + to_sampler_expression(var_id ? var_id : id);
	}

	if (msl_options.swizzle_texture_samples && has_sampled_images && is_sampled_image_type(type))
	{
		// Need to check the base variable in case we need to apply a qualified alias.
		uint32_t var_id = 0;
		auto *sampler_var = maybe_get<SPIRVariable>(id);
		if (sampler_var)
			var_id = sampler_var->basevariable;

		arg_str += ", " + to_swizzle_expression(var_id ? var_id : id);
	}

	return arg_str;
}

// If the ID represents a sampled image that has been assigned a sampler already,
// generate an expression for the sampler, otherwise generate a fake sampler name
// by appending a suffix to the expression constructed from the ID.
string CompilerMSL::to_sampler_expression(uint32_t id)
{
	auto *combined = maybe_get<SPIRCombinedImageSampler>(id);
	auto expr = to_expression(combined ? combined->image : id);
	auto index = expr.find_first_of('[');

	uint32_t samp_id = 0;
	if (combined)
		samp_id = combined->sampler;

	if (index == string::npos)
		return samp_id ? to_expression(samp_id) : expr + sampler_name_suffix;
	else
	{
		auto image_expr = expr.substr(0, index);
		auto array_expr = expr.substr(index);
		return samp_id ? to_expression(samp_id) : (image_expr + sampler_name_suffix + array_expr);
	}
}

string CompilerMSL::to_swizzle_expression(uint32_t id)
{
	auto *combined = maybe_get<SPIRCombinedImageSampler>(id);

	auto expr = to_expression(combined ? combined->image : id);
	auto index = expr.find_first_of('[');

	// If an image is part of an argument buffer translate this to a legal identifier.
	for (auto &c : expr)
		if (c == '.')
			c = '_';

	if (index == string::npos)
		return expr + swizzle_name_suffix;
	else
	{
		auto image_expr = expr.substr(0, index);
		auto array_expr = expr.substr(index);
		return image_expr + swizzle_name_suffix + array_expr;
	}
}

// Checks whether the type is a Block all of whose members have DecorationPatch.
bool CompilerMSL::is_patch_block(const SPIRType &type)
{
	if (!has_decoration(type.self, DecorationBlock))
		return false;

	for (uint32_t i = 0; i < type.member_types.size(); i++)
	{
		if (!has_member_decoration(type.self, i, DecorationPatch))
			return false;
	}

	return true;
}

// Checks whether the ID is a row_major matrix that requires conversion before use
bool CompilerMSL::is_non_native_row_major_matrix(uint32_t id)
{
	// Natively supported row-major matrices do not need to be converted.
	if (backend.native_row_major_matrix)
		return false;

	// Non-matrix or column-major matrix types do not need to be converted.
	if (!has_decoration(id, DecorationRowMajor))
		return false;

	// Generate a function that will swap matrix elements from row-major to column-major.
	// Packed row-matrix should just use transpose() function.
	if (!has_extended_decoration(id, SPIRVCrossDecorationPacked))
	{
		const auto type = expression_type(id);
		add_convert_row_major_matrix_function(type.columns, type.vecsize);
	}

	return true;
}

// Checks whether the member is a row_major matrix that requires conversion before use
bool CompilerMSL::member_is_non_native_row_major_matrix(const SPIRType &type, uint32_t index)
{
	// Natively supported row-major matrices do not need to be converted.
	if (backend.native_row_major_matrix)
		return false;

	// Non-matrix or column-major matrix types do not need to be converted.
	if (!has_member_decoration(type.self, index, DecorationRowMajor))
		return false;

	// Generate a function that will swap matrix elements from row-major to column-major.
	// Packed row-matrix should just use transpose() function.
	if (!has_extended_member_decoration(type.self, index, SPIRVCrossDecorationPacked))
	{
		const auto mbr_type = get<SPIRType>(type.member_types[index]);
		add_convert_row_major_matrix_function(mbr_type.columns, mbr_type.vecsize);
	}

	return true;
}

// Adds a function suitable for converting a non-square row-major matrix to a column-major matrix.
void CompilerMSL::add_convert_row_major_matrix_function(uint32_t cols, uint32_t rows)
{
	SPVFuncImpl spv_func;
	if (cols == rows) // Square matrix...just use transpose() function
		return;
	else if (cols == 2 && rows == 3)
		spv_func = SPVFuncImplRowMajor2x3;
	else if (cols == 2 && rows == 4)
		spv_func = SPVFuncImplRowMajor2x4;
	else if (cols == 3 && rows == 2)
		spv_func = SPVFuncImplRowMajor3x2;
	else if (cols == 3 && rows == 4)
		spv_func = SPVFuncImplRowMajor3x4;
	else if (cols == 4 && rows == 2)
		spv_func = SPVFuncImplRowMajor4x2;
	else if (cols == 4 && rows == 3)
		spv_func = SPVFuncImplRowMajor4x3;
	else
		SPIRV_CROSS_THROW("Could not convert row-major matrix.");

	auto rslt = spv_function_implementations.insert(spv_func);
	if (rslt.second)
	{
		suppress_missing_prototypes = true;
		force_recompile();
	}
}

// Wraps the expression string in a function call that converts the
// row_major matrix result of the expression to a column_major matrix.
string CompilerMSL::convert_row_major_matrix(string exp_str, const SPIRType &exp_type, bool is_packed)
{
	strip_enclosed_expression(exp_str);

	string func_name;

	// Square and packed matrices can just use transpose
	if (exp_type.columns == exp_type.vecsize || is_packed)
		func_name = "transpose";
	else
		func_name = string("spvConvertFromRowMajor") + to_string(exp_type.columns) + "x" + to_string(exp_type.vecsize);

	return join(func_name, "(", exp_str, ")");
}

// Called automatically at the end of the entry point function
void CompilerMSL::emit_fixup()
{
	if ((get_execution_model() == ExecutionModelVertex ||
	     get_execution_model() == ExecutionModelTessellationEvaluation) &&
	    stage_out_var_id && !qual_pos_var_name.empty() && !capture_output_to_buffer)
	{
		if (options.vertex.fixup_clipspace)
			statement(qual_pos_var_name, ".z = (", qual_pos_var_name, ".z + ", qual_pos_var_name,
			          ".w) * 0.5;       // Adjust clip-space for Metal");

		if (options.vertex.flip_vert_y)
			statement(qual_pos_var_name, ".y = -(", qual_pos_var_name, ".y);", "    // Invert Y-axis for Metal");
	}
}

// Return a string defining a structure member, with padding and packing.
string CompilerMSL::to_struct_member(const SPIRType &type, uint32_t member_type_id, uint32_t index,
                                     const string &qualifier)
{
	auto &membertype = get<SPIRType>(member_type_id);

	// If this member requires padding to maintain alignment, emit a dummy padding member.
	MSLStructMemberKey key = get_struct_member_key(type.self, index);
	uint32_t pad_len = struct_member_padding[key];
	if (pad_len > 0)
		statement("char _m", index, "_pad", "[", to_string(pad_len), "];");

	// If this member is packed, mark it as so.
	string pack_pfx = "";

	const SPIRType *effective_membertype = &membertype;
	SPIRType override_type;

	uint32_t orig_id = 0;
	if (has_extended_member_decoration(type.self, index, SPIRVCrossDecorationInterfaceOrigID))
		orig_id = get_extended_member_decoration(type.self, index, SPIRVCrossDecorationInterfaceOrigID);

	if (member_is_packed_type(type, index))
	{
		// If we're packing a matrix, output an appropriate typedef
		if (membertype.basetype == SPIRType::Struct)
		{
			pack_pfx = "/* FIXME: A padded struct is needed here. If you see this message, file a bug! */ ";
		}
		else if (membertype.vecsize > 1 && membertype.columns > 1)
		{
			pack_pfx = "packed_";
			string base_type = membertype.width == 16 ? "half" : "float";
			string td_line = "typedef ";
			td_line += base_type + to_string(membertype.vecsize) + "x" + to_string(membertype.columns);
			td_line += " " + pack_pfx;
			td_line += base_type + to_string(membertype.columns) + "x" + to_string(membertype.vecsize);
			td_line += ";";
			add_typedef_line(td_line);
		}
		else if (is_array(membertype) && membertype.vecsize <= 2 && membertype.basetype != SPIRType::Struct)
		{
			// A "packed" float array, but we pad here instead to 4-vector.
			override_type = membertype;
			override_type.vecsize = 4;
			effective_membertype = &override_type;
		}
		else
			pack_pfx = "packed_";
	}

	// Very specifically, image load-store in argument buffers are disallowed on MSL on iOS.
	if (msl_options.is_ios() && membertype.basetype == SPIRType::Image && membertype.image.sampled == 2)
	{
		if (!has_decoration(orig_id, DecorationNonWritable))
			SPIRV_CROSS_THROW("Writable images are not allowed in argument buffers on iOS.");
	}

	// Array information is baked into these types.
	string array_type;
	if (membertype.basetype != SPIRType::Image && membertype.basetype != SPIRType::Sampler &&
	    membertype.basetype != SPIRType::SampledImage)
	{
		array_type = type_to_array_glsl(membertype);
	}

	return join(pack_pfx, type_to_glsl(*effective_membertype, orig_id), " ", qualifier, to_member_name(type, index),
	            member_attribute_qualifier(type, index), array_type, ";");
}

// Emit a structure member, padding and packing to maintain the correct memeber alignments.
void CompilerMSL::emit_struct_member(const SPIRType &type, uint32_t member_type_id, uint32_t index,
                                     const string &qualifier, uint32_t)
{
	statement(to_struct_member(type, member_type_id, index, qualifier));
}

// Return a MSL qualifier for the specified function attribute member
string CompilerMSL::member_attribute_qualifier(const SPIRType &type, uint32_t index)
{
	auto &execution = get_entry_point();

	uint32_t mbr_type_id = type.member_types[index];
	auto &mbr_type = get<SPIRType>(mbr_type_id);

	BuiltIn builtin = BuiltInMax;
	bool is_builtin = is_member_builtin(type, index, &builtin);

	if (has_extended_member_decoration(type.self, index, SPIRVCrossDecorationArgumentBufferID))
		return join(" [[id(", get_extended_member_decoration(type.self, index, SPIRVCrossDecorationArgumentBufferID),
		            ")]]");

	// Vertex function inputs
	if (execution.model == ExecutionModelVertex && type.storage == StorageClassInput)
	{
		if (is_builtin)
		{
			switch (builtin)
			{
			case BuiltInVertexId:
			case BuiltInVertexIndex:
			case BuiltInBaseVertex:
			case BuiltInInstanceId:
			case BuiltInInstanceIndex:
			case BuiltInBaseInstance:
				return string(" [[") + builtin_qualifier(builtin) + "]]";

			case BuiltInDrawIndex:
				SPIRV_CROSS_THROW("DrawIndex is not supported in MSL.");

			default:
				return "";
			}
		}
		uint32_t locn = get_ordered_member_location(type.self, index);
		if (locn != k_unknown_location)
			return string(" [[attribute(") + convert_to_string(locn) + ")]]";
	}

	// Vertex and tessellation evaluation function outputs
	if ((execution.model == ExecutionModelVertex || execution.model == ExecutionModelTessellationEvaluation) &&
	    type.storage == StorageClassOutput)
	{
		if (is_builtin)
		{
			switch (builtin)
			{
			case BuiltInPointSize:
				// Only mark the PointSize builtin if really rendering points.
				// Some shaders may include a PointSize builtin even when used to render
				// non-point topologies, and Metal will reject this builtin when compiling
				// the shader into a render pipeline that uses a non-point topology.
				return msl_options.enable_point_size_builtin ? (string(" [[") + builtin_qualifier(builtin) + "]]") : "";

			case BuiltInViewportIndex:
				if (!msl_options.supports_msl_version(2, 0))
					SPIRV_CROSS_THROW("ViewportIndex requires Metal 2.0.");
				/* fallthrough */
			case BuiltInPosition:
			case BuiltInLayer:
			case BuiltInClipDistance:
				return string(" [[") + builtin_qualifier(builtin) + "]]" + (mbr_type.array.empty() ? "" : " ");

			default:
				return "";
			}
		}
		uint32_t comp;
		uint32_t locn = get_ordered_member_location(type.self, index, &comp);
		if (locn != k_unknown_location)
		{
			if (comp != k_unknown_component)
				return string(" [[user(locn") + convert_to_string(locn) + "_" + convert_to_string(comp) + ")]]";
			else
				return string(" [[user(locn") + convert_to_string(locn) + ")]]";
		}
	}

	// Tessellation control function inputs
	if (execution.model == ExecutionModelTessellationControl && type.storage == StorageClassInput)
	{
		if (is_builtin)
		{
			switch (builtin)
			{
			case BuiltInInvocationId:
			case BuiltInPrimitiveId:
			case BuiltInSubgroupLocalInvocationId: // FIXME: Should work in any stage
			case BuiltInSubgroupSize: // FIXME: Should work in any stage
				return string(" [[") + builtin_qualifier(builtin) + "]]" + (mbr_type.array.empty() ? "" : " ");
			case BuiltInPatchVertices:
				return "";
			// Others come from stage input.
			default:
				break;
			}
		}
		uint32_t locn = get_ordered_member_location(type.self, index);
		if (locn != k_unknown_location)
			return string(" [[attribute(") + convert_to_string(locn) + ")]]";
	}

	// Tessellation control function outputs
	if (execution.model == ExecutionModelTessellationControl && type.storage == StorageClassOutput)
	{
		// For this type of shader, we always arrange for it to capture its
		// output to a buffer. For this reason, qualifiers are irrelevant here.
		return "";
	}

	// Tessellation evaluation function inputs
	if (execution.model == ExecutionModelTessellationEvaluation && type.storage == StorageClassInput)
	{
		if (is_builtin)
		{
			switch (builtin)
			{
			case BuiltInPrimitiveId:
			case BuiltInTessCoord:
				return string(" [[") + builtin_qualifier(builtin) + "]]";
			case BuiltInPatchVertices:
				return "";
			// Others come from stage input.
			default:
				break;
			}
		}
		// The special control point array must not be marked with an attribute.
		if (get_type(type.member_types[index]).basetype == SPIRType::ControlPointArray)
			return "";
		uint32_t locn = get_ordered_member_location(type.self, index);
		if (locn != k_unknown_location)
			return string(" [[attribute(") + convert_to_string(locn) + ")]]";
	}

	// Tessellation evaluation function outputs were handled above.

	// Fragment function inputs
	if (execution.model == ExecutionModelFragment && type.storage == StorageClassInput)
	{
		string quals = "";
		if (is_builtin)
		{
			switch (builtin)
			{
			case BuiltInFrontFacing:
			case BuiltInPointCoord:
			case BuiltInFragCoord:
			case BuiltInSampleId:
			case BuiltInSampleMask:
			case BuiltInLayer:
				quals = builtin_qualifier(builtin);

			default:
				break;
			}
		}
		else
		{
			uint32_t comp;
			uint32_t locn = get_ordered_member_location(type.self, index, &comp);
			if (locn != k_unknown_location)
			{
				if (comp != k_unknown_component)
					quals = string("user(locn") + convert_to_string(locn) + "_" + convert_to_string(comp) + ")";
				else
					quals = string("user(locn") + convert_to_string(locn) + ")";
			}
		}
		// Don't bother decorating integers with the 'flat' attribute; it's
		// the default (in fact, the only option). Also don't bother with the
		// FragCoord builtin; it's always noperspective on Metal.
		if (!type_is_integral(mbr_type) && (!is_builtin || builtin != BuiltInFragCoord))
		{
			if (has_member_decoration(type.self, index, DecorationFlat))
			{
				if (!quals.empty())
					quals += ", ";
				quals += "flat";
			}
			else if (has_member_decoration(type.self, index, DecorationCentroid))
			{
				if (!quals.empty())
					quals += ", ";
				if (has_member_decoration(type.self, index, DecorationNoPerspective))
					quals += "centroid_no_perspective";
				else
					quals += "centroid_perspective";
			}
			else if (has_member_decoration(type.self, index, DecorationSample))
			{
				if (!quals.empty())
					quals += ", ";
				if (has_member_decoration(type.self, index, DecorationNoPerspective))
					quals += "sample_no_perspective";
				else
					quals += "sample_perspective";
			}
			else if (has_member_decoration(type.self, index, DecorationNoPerspective))
			{
				if (!quals.empty())
					quals += ", ";
				quals += "center_no_perspective";
			}
		}
		if (!quals.empty())
			return " [[" + quals + "]]";
	}

	// Fragment function outputs
	if (execution.model == ExecutionModelFragment && type.storage == StorageClassOutput)
	{
		if (is_builtin)
		{
			switch (builtin)
			{
			case BuiltInSampleMask:
			case BuiltInFragDepth:
				return string(" [[") + builtin_qualifier(builtin) + "]]";

			default:
				return "";
			}
		}
		uint32_t locn = get_ordered_member_location(type.self, index);
		if (locn != k_unknown_location && has_member_decoration(type.self, index, DecorationIndex))
			return join(" [[color(", locn, "), index(", get_member_decoration(type.self, index, DecorationIndex),
			            ")]]");
		else if (locn != k_unknown_location)
			return join(" [[color(", locn, ")]]");
		else if (has_member_decoration(type.self, index, DecorationIndex))
			return join(" [[index(", get_member_decoration(type.self, index, DecorationIndex), ")]]");
		else
			return "";
	}

	// Compute function inputs
	if (execution.model == ExecutionModelGLCompute && type.storage == StorageClassInput)
	{
		if (is_builtin)
		{
			switch (builtin)
			{
			case BuiltInGlobalInvocationId:
			case BuiltInWorkgroupId:
			case BuiltInNumWorkgroups:
			case BuiltInLocalInvocationId:
			case BuiltInLocalInvocationIndex:
			case BuiltInNumSubgroups:
			case BuiltInSubgroupId:
			case BuiltInSubgroupLocalInvocationId: // FIXME: Should work in any stage
			case BuiltInSubgroupSize: // FIXME: Should work in any stage
				return string(" [[") + builtin_qualifier(builtin) + "]]";

			default:
				return "";
			}
		}
	}

	return "";
}

// Returns the location decoration of the member with the specified index in the specified type.
// If the location of the member has been explicitly set, that location is used. If not, this
// function assumes the members are ordered in their location order, and simply returns the
// index as the location.
uint32_t CompilerMSL::get_ordered_member_location(uint32_t type_id, uint32_t index, uint32_t *comp)
{
	auto &m = ir.meta[type_id];
	if (index < m.members.size())
	{
		auto &dec = m.members[index];
		if (comp)
		{
			if (dec.decoration_flags.get(DecorationComponent))
				*comp = dec.component;
			else
				*comp = k_unknown_component;
		}
		if (dec.decoration_flags.get(DecorationLocation))
			return dec.location;
	}

	return index;
}

// Returns the type declaration for a function, including the
// entry type if the current function is the entry point function
string CompilerMSL::func_type_decl(SPIRType &type)
{
	// The regular function return type. If not processing the entry point function, that's all we need
	string return_type = type_to_glsl(type) + type_to_array_glsl(type);
	if (!processing_entry_point)
		return return_type;

	// If an outgoing interface block has been defined, and it should be returned, override the entry point return type
	bool ep_should_return_output = !get_is_rasterization_disabled();
	if (stage_out_var_id && ep_should_return_output)
		return_type = type_to_glsl(get_stage_out_struct_type()) + type_to_array_glsl(type);

	// Prepend a entry type, based on the execution model
	string entry_type;
	auto &execution = get_entry_point();
	switch (execution.model)
	{
	case ExecutionModelVertex:
		entry_type = "vertex";
		break;
	case ExecutionModelTessellationEvaluation:
		if (!msl_options.supports_msl_version(1, 2))
			SPIRV_CROSS_THROW("Tessellation requires Metal 1.2.");
		if (execution.flags.get(ExecutionModeIsolines))
			SPIRV_CROSS_THROW("Metal does not support isoline tessellation.");
		if (msl_options.is_ios())
			entry_type =
			    join("[[ patch(", execution.flags.get(ExecutionModeTriangles) ? "triangle" : "quad", ") ]] vertex");
		else
			entry_type = join("[[ patch(", execution.flags.get(ExecutionModeTriangles) ? "triangle" : "quad", ", ",
			                  execution.output_vertices, ") ]] vertex");
		break;
	case ExecutionModelFragment:
		entry_type =
		    execution.flags.get(ExecutionModeEarlyFragmentTests) ? "[[ early_fragment_tests ]] fragment" : "fragment";
		break;
	case ExecutionModelTessellationControl:
		if (!msl_options.supports_msl_version(1, 2))
			SPIRV_CROSS_THROW("Tessellation requires Metal 1.2.");
		if (execution.flags.get(ExecutionModeIsolines))
			SPIRV_CROSS_THROW("Metal does not support isoline tessellation.");
		/* fallthrough */
	case ExecutionModelGLCompute:
	case ExecutionModelKernel:
		entry_type = "kernel";
		break;
	default:
		entry_type = "unknown";
		break;
	}

	return entry_type + " " + return_type;
}

// In MSL, address space qualifiers are required for all pointer or reference variables
string CompilerMSL::get_argument_address_space(const SPIRVariable &argument)
{
	const auto &type = get<SPIRType>(argument.basetype);

	switch (type.storage)
	{
	case StorageClassWorkgroup:
		return "threadgroup";

	case StorageClassStorageBuffer:
	{
		// For arguments from variable pointers, we use the write count deduction, so
		// we should not assume any constness here. Only for global SSBOs.
		bool readonly = false;
		if (has_decoration(type.self, DecorationBlock))
			readonly = ir.get_buffer_block_flags(argument).get(DecorationNonWritable);

		return readonly ? "const device" : "device";
	}

	case StorageClassUniform:
	case StorageClassUniformConstant:
	case StorageClassPushConstant:
		if (type.basetype == SPIRType::Struct)
		{
			bool ssbo = has_decoration(type.self, DecorationBufferBlock);
			if (ssbo)
			{
				bool readonly = ir.get_buffer_block_flags(argument).get(DecorationNonWritable);
				return readonly ? "const device" : "device";
			}
			else
				return "constant";
		}
		break;

	case StorageClassFunction:
	case StorageClassGeneric:
		// No address space for plain values.
		return type.pointer ? "thread" : "";

	case StorageClassInput:
		if (get_execution_model() == ExecutionModelTessellationControl && argument.basevariable == stage_in_ptr_var_id)
			return "threadgroup";
		break;

	case StorageClassOutput:
		if (capture_output_to_buffer)
			return "device";
		break;

	default:
		break;
	}

	return "thread";
}

string CompilerMSL::get_type_address_space(const SPIRType &type, uint32_t id)
{
	switch (type.storage)
	{
	case StorageClassWorkgroup:
		return "threadgroup";

	case StorageClassStorageBuffer:
	{
		// This can be called for variable pointer contexts as well, so be very careful about which method we choose.
		Bitset flags;
		if (ir.ids[id].get_type() == TypeVariable && has_decoration(type.self, DecorationBlock))
			flags = get_buffer_block_flags(id);
		else
			flags = get_decoration_bitset(id);

		return flags.get(DecorationNonWritable) ? "const device" : "device";
	}

	case StorageClassUniform:
	case StorageClassUniformConstant:
	case StorageClassPushConstant:
		if (type.basetype == SPIRType::Struct)
		{
			bool ssbo = has_decoration(type.self, DecorationBufferBlock);
			if (ssbo)
			{
				// This can be called for variable pointer contexts as well, so be very careful about which method we choose.
				Bitset flags;
				if (ir.ids[id].get_type() == TypeVariable && has_decoration(type.self, DecorationBlock))
					flags = get_buffer_block_flags(id);
				else
					flags = get_decoration_bitset(id);

				return flags.get(DecorationNonWritable) ? "const device" : "device";
			}
			else
				return "constant";
		}
		else
			return "constant";

	case StorageClassFunction:
	case StorageClassGeneric:
		// No address space for plain values.
		return type.pointer ? "thread" : "";

	case StorageClassOutput:
		if (capture_output_to_buffer)
			return "device";
		break;

	default:
		break;
	}

	return "thread";
}

string CompilerMSL::entry_point_arg_stage_in()
{
	string decl;

	// Stage-in structure
	uint32_t stage_in_id;
	if (get_execution_model() == ExecutionModelTessellationEvaluation)
		stage_in_id = patch_stage_in_var_id;
	else
		stage_in_id = stage_in_var_id;

	if (stage_in_id)
	{
		auto &var = get<SPIRVariable>(stage_in_id);
		auto &type = get_variable_data_type(var);

		add_resource_name(var.self);
		decl = join(type_to_glsl(type), " ", to_name(var.self), " [[stage_in]]");
	}

	return decl;
}

void CompilerMSL::entry_point_args_builtin(string &ep_args)
{
	// Builtin variables
	ir.for_each_typed_id<SPIRVariable>([&](uint32_t var_id, SPIRVariable &var) {
		BuiltIn bi_type = ir.meta[var_id].decoration.builtin_type;

		// Don't emit SamplePosition as a separate parameter. In the entry
		// point, we get that by calling get_sample_position() on the sample ID.
		if (var.storage == StorageClassInput && is_builtin_variable(var) &&
		    get_variable_data_type(var).basetype != SPIRType::Struct &&
		    get_variable_data_type(var).basetype != SPIRType::ControlPointArray)
		{
			if (bi_type != BuiltInSamplePosition && bi_type != BuiltInHelperInvocation &&
			    bi_type != BuiltInPatchVertices && bi_type != BuiltInTessLevelInner &&
			    bi_type != BuiltInTessLevelOuter && bi_type != BuiltInPosition && bi_type != BuiltInPointSize &&
			    bi_type != BuiltInClipDistance && bi_type != BuiltInCullDistance && bi_type != BuiltInSubgroupEqMask &&
			    bi_type != BuiltInSubgroupGeMask && bi_type != BuiltInSubgroupGtMask &&
			    bi_type != BuiltInSubgroupLeMask && bi_type != BuiltInSubgroupLtMask)
			{
				if (!ep_args.empty())
					ep_args += ", ";

				ep_args += builtin_type_decl(bi_type) + " " + to_expression(var_id);
				ep_args += " [[" + builtin_qualifier(bi_type) + "]]";
			}
		}
	});

	// Vertex and instance index built-ins
	if (needs_vertex_idx_arg)
		ep_args += built_in_func_arg(BuiltInVertexIndex, !ep_args.empty());

	if (needs_instance_idx_arg)
		ep_args += built_in_func_arg(BuiltInInstanceIndex, !ep_args.empty());

	if (capture_output_to_buffer)
	{
		// Add parameters to hold the indirect draw parameters and the shader output. This has to be handled
		// specially because it needs to be a pointer, not a reference.
		if (stage_out_var_id)
		{
			if (!ep_args.empty())
				ep_args += ", ";
			ep_args += join("device ", type_to_glsl(get_stage_out_struct_type()), "* ", output_buffer_var_name,
			                " [[buffer(", msl_options.shader_output_buffer_index, ")]]");
		}

		if (get_execution_model() == ExecutionModelTessellationControl)
		{
			if (!ep_args.empty())
				ep_args += ", ";
			ep_args +=
			    join("constant uint* spvIndirectParams [[buffer(", msl_options.indirect_params_buffer_index, ")]]");
		}
		else if (stage_out_var_id)
		{
			if (!ep_args.empty())
				ep_args += ", ";
			ep_args +=
			    join("device uint* spvIndirectParams [[buffer(", msl_options.indirect_params_buffer_index, ")]]");
		}

		// Tessellation control shaders get three additional parameters:
		// a buffer to hold the per-patch data, a buffer to hold the per-patch
		// tessellation levels, and a block of workgroup memory to hold the
		// input control point data.
		if (get_execution_model() == ExecutionModelTessellationControl)
		{
			if (patch_stage_out_var_id)
			{
				if (!ep_args.empty())
					ep_args += ", ";
				ep_args +=
				    join("device ", type_to_glsl(get_patch_stage_out_struct_type()), "* ", patch_output_buffer_var_name,
				         " [[buffer(", convert_to_string(msl_options.shader_patch_output_buffer_index), ")]]");
			}
			if (!ep_args.empty())
				ep_args += ", ";
			ep_args += join("device ", get_tess_factor_struct_name(), "* ", tess_factor_buffer_var_name, " [[buffer(",
			                convert_to_string(msl_options.shader_tess_factor_buffer_index), ")]]");
			if (stage_in_var_id)
			{
				if (!ep_args.empty())
					ep_args += ", ";
				ep_args += join("threadgroup ", type_to_glsl(get_stage_in_struct_type()), "* ", input_wg_var_name,
				                " [[threadgroup(", convert_to_string(msl_options.shader_input_wg_index), ")]]");
			}
		}
	}
}

string CompilerMSL::entry_point_args_argument_buffer(bool append_comma)
{
	string ep_args = entry_point_arg_stage_in();

	for (uint32_t i = 0; i < kMaxArgumentBuffers; i++)
	{
		uint32_t id = argument_buffer_ids[i];
		if (id == 0)
			continue;

		add_resource_name(id);
		auto &var = get<SPIRVariable>(id);
		auto &type = get_variable_data_type(var);

		if (!ep_args.empty())
			ep_args += ", ";

		ep_args += get_argument_address_space(var) + " " + type_to_glsl(type) + "& " + to_name(id);
		ep_args += " [[buffer(" + convert_to_string(i) + ")]]";

		// Makes it more practical for testing, since the push constant block can occupy the first available
		// buffer slot if it's not bound explicitly.
		next_metal_resource_index_buffer = i + 1;
	}

	entry_point_args_discrete_descriptors(ep_args);
	entry_point_args_builtin(ep_args);

	if (!ep_args.empty() && append_comma)
		ep_args += ", ";

	return ep_args;
}

void CompilerMSL::entry_point_args_discrete_descriptors(string &ep_args)
{
	// Output resources, sorted by resource index & type
	// We need to sort to work around a bug on macOS 10.13 with NVidia drivers where switching between shaders
	// with different order of buffers can result in issues with buffer assignments inside the driver.
	struct Resource
	{
		SPIRVariable *var;
		string name;
		SPIRType::BaseType basetype;
		uint32_t index;
	};

	SmallVector<Resource> resources;

	ir.for_each_typed_id<SPIRVariable>([&](uint32_t, SPIRVariable &var) {
		if ((var.storage == StorageClassUniform || var.storage == StorageClassUniformConstant ||
		     var.storage == StorageClassPushConstant || var.storage == StorageClassStorageBuffer) &&
		    !is_hidden_variable(var))
		{
			auto &type = get_variable_data_type(var);
			uint32_t var_id = var.self;

			if (var.storage != StorageClassPushConstant)
			{
				uint32_t desc_set = get_decoration(var_id, DecorationDescriptorSet);
				if (descriptor_set_is_argument_buffer(desc_set))
					return;
			}

			if (type.basetype == SPIRType::SampledImage)
			{
				add_resource_name(var_id);
				resources.push_back(
				    { &var, to_name(var_id), SPIRType::Image, get_metal_resource_index(var, SPIRType::Image) });

				if (type.image.dim != DimBuffer && constexpr_samplers.count(var_id) == 0)
				{
					resources.push_back({ &var, to_sampler_expression(var_id), SPIRType::Sampler,
					                      get_metal_resource_index(var, SPIRType::Sampler) });
				}
			}
			else if (constexpr_samplers.count(var_id) == 0)
			{
				// constexpr samplers are not declared as resources.
				add_resource_name(var_id);
				resources.push_back(
				    { &var, to_name(var_id), type.basetype, get_metal_resource_index(var, type.basetype) });
			}
		}
	});

	sort(resources.begin(), resources.end(), [](const Resource &lhs, const Resource &rhs) {
		return tie(lhs.basetype, lhs.index) < tie(rhs.basetype, rhs.index);
	});

	for (auto &r : resources)
	{
		auto &var = *r.var;
		auto &type = get_variable_data_type(var);

		uint32_t var_id = var.self;

		switch (r.basetype)
		{
		case SPIRType::Struct:
		{
			auto &m = ir.meta[type.self];
			if (m.members.size() == 0)
				break;
			if (!type.array.empty())
			{
				if (type.array.size() > 1)
					SPIRV_CROSS_THROW("Arrays of arrays of buffers are not supported.");

				// Metal doesn't directly support this, so we must expand the
				// array. We'll declare a local array to hold these elements
				// later.
				uint32_t array_size = to_array_size_literal(type);

				if (array_size == 0)
					SPIRV_CROSS_THROW("Unsized arrays of buffers are not supported in MSL.");

				buffer_arrays.push_back(var_id);
				for (uint32_t i = 0; i < array_size; ++i)
				{
					if (!ep_args.empty())
						ep_args += ", ";
					ep_args += get_argument_address_space(var) + " " + type_to_glsl(type) + "* " + r.name + "_" +
					           convert_to_string(i);
					ep_args += " [[buffer(" + convert_to_string(r.index + i) + ")]]";
				}
			}
			else
			{
				if (!ep_args.empty())
					ep_args += ", ";
				ep_args += get_argument_address_space(var) + " " + type_to_glsl(type) + "& " + r.name;
				ep_args += " [[buffer(" + convert_to_string(r.index) + ")]]";
			}
			break;
		}
		case SPIRType::Sampler:
			if (!ep_args.empty())
				ep_args += ", ";
			ep_args += sampler_type(type) + " " + r.name;
			ep_args += " [[sampler(" + convert_to_string(r.index) + ")]]";
			break;
		case SPIRType::Image:
			if (!ep_args.empty())
				ep_args += ", ";
			ep_args += image_type_glsl(type, var_id) + " " + r.name;
			ep_args += " [[texture(" + convert_to_string(r.index) + ")]]";
			break;
		default:
			if (!ep_args.empty())
				ep_args += ", ";
			ep_args += type_to_glsl(type, var_id) + " " + r.name;
			ep_args += " [[buffer(" + convert_to_string(r.index) + ")]]";
			break;
		}
	}
}

// Returns a string containing a comma-delimited list of args for the entry point function
// This is the "classic" method of MSL 1 when we don't have argument buffer support.
string CompilerMSL::entry_point_args_classic(bool append_comma)
{
	string ep_args = entry_point_arg_stage_in();
	entry_point_args_discrete_descriptors(ep_args);
	entry_point_args_builtin(ep_args);

	if (!ep_args.empty() && append_comma)
		ep_args += ", ";

	return ep_args;
}

void CompilerMSL::fix_up_shader_inputs_outputs()
{
	// Look for sampled images. Add hooks to set up the swizzle constants.
	ir.for_each_typed_id<SPIRVariable>([&](uint32_t, SPIRVariable &var) {
		auto &type = get_variable_data_type(var);

		uint32_t var_id = var.self;

		if ((var.storage == StorageClassUniform || var.storage == StorageClassUniformConstant ||
		     var.storage == StorageClassPushConstant || var.storage == StorageClassStorageBuffer) &&
		    !is_hidden_variable(var))
		{
			if (msl_options.swizzle_texture_samples && has_sampled_images && is_sampled_image_type(type))
			{
				auto &entry_func = this->get<SPIRFunction>(ir.default_entry_point);
				entry_func.fixup_hooks_in.push_back([this, &type, &var, var_id]() {
					bool is_array_type = !type.array.empty();

					uint32_t desc_set = get_decoration(var_id, DecorationDescriptorSet);
					if (descriptor_set_is_argument_buffer(desc_set))
					{
						statement("constant uint32_t", is_array_type ? "* " : "& ", to_swizzle_expression(var_id),
						          is_array_type ? " = &" : " = ", to_name(argument_buffer_ids[desc_set]),
						          ".spvSwizzleConstants", "[",
						          convert_to_string(get_metal_resource_index(var, SPIRType::Image)), "];");
					}
					else
					{
						// If we have an array of images, we need to be able to index into it, so take a pointer instead.
						statement("constant uint32_t", is_array_type ? "* " : "& ", to_swizzle_expression(var_id),
						          is_array_type ? " = &" : " = ", to_name(swizzle_buffer_id), "[",
						          convert_to_string(get_metal_resource_index(var, SPIRType::Image)), "];");
					}
				});
			}
		}
	});

	// Builtin variables
	ir.for_each_typed_id<SPIRVariable>([&](uint32_t, SPIRVariable &var) {
		uint32_t var_id = var.self;
		BuiltIn bi_type = ir.meta[var_id].decoration.builtin_type;

		if (var.storage == StorageClassInput && is_builtin_variable(var))
		{
			auto &entry_func = this->get<SPIRFunction>(ir.default_entry_point);
			switch (bi_type)
			{
			case BuiltInSamplePosition:
				entry_func.fixup_hooks_in.push_back([=]() {
					statement(builtin_type_decl(bi_type), " ", to_expression(var_id), " = get_sample_position(",
					          to_expression(builtin_sample_id_id), ");");
				});
				break;
			case BuiltInHelperInvocation:
				if (msl_options.is_ios())
					SPIRV_CROSS_THROW("simd_is_helper_thread() is only supported on macOS.");
				else if (msl_options.is_macos() && !msl_options.supports_msl_version(2, 1))
					SPIRV_CROSS_THROW("simd_is_helper_thread() requires version 2.1 on macOS.");

				entry_func.fixup_hooks_in.push_back([=]() {
					statement(builtin_type_decl(bi_type), " ", to_expression(var_id), " = simd_is_helper_thread();");
				});
				break;
			case BuiltInPatchVertices:
				if (get_execution_model() == ExecutionModelTessellationEvaluation)
					entry_func.fixup_hooks_in.push_back([=]() {
						statement(builtin_type_decl(bi_type), " ", to_expression(var_id), " = ",
						          to_expression(patch_stage_in_var_id), ".gl_in.size();");
					});
				else
					entry_func.fixup_hooks_in.push_back([=]() {
						statement(builtin_type_decl(bi_type), " ", to_expression(var_id), " = spvIndirectParams[0];");
					});
				break;
			case BuiltInTessCoord:
				// Emit a fixup to account for the shifted domain. Don't do this for triangles;
				// MoltenVK will just reverse the winding order instead.
				if (msl_options.tess_domain_origin_lower_left && !get_entry_point().flags.get(ExecutionModeTriangles))
				{
					string tc = to_expression(var_id);
					entry_func.fixup_hooks_in.push_back([=]() { statement(tc, ".y = 1.0 - ", tc, ".y;"); });
				}
				break;
			case BuiltInSubgroupEqMask:
				if (msl_options.is_ios())
					SPIRV_CROSS_THROW("Subgroup ballot functionality is unavailable on iOS.");
				if (!msl_options.supports_msl_version(2, 1))
					SPIRV_CROSS_THROW("Subgroup ballot functionality requires Metal 2.1.");
				entry_func.fixup_hooks_in.push_back([=]() {
					statement(builtin_type_decl(bi_type), " ", to_expression(var_id), " = ",
					          builtin_subgroup_invocation_id_id, " > 32 ? uint4(0, (1 << (",
					          to_expression(builtin_subgroup_invocation_id_id), " - 32)), uint2(0)) : uint4(1 << ",
					          to_expression(builtin_subgroup_invocation_id_id), ", uint3(0));");
				});
				break;
			case BuiltInSubgroupGeMask:
				if (msl_options.is_ios())
					SPIRV_CROSS_THROW("Subgroup ballot functionality is unavailable on iOS.");
				if (!msl_options.supports_msl_version(2, 1))
					SPIRV_CROSS_THROW("Subgroup ballot functionality requires Metal 2.1.");
				entry_func.fixup_hooks_in.push_back([=]() {
					// Case where index < 32, size < 32:
					// mask0 = bfe(0xFFFFFFFF, index, size - index);
					// mask1 = bfe(0xFFFFFFFF, 0, 0); // Gives 0
					// Case where index < 32 but size >= 32:
					// mask0 = bfe(0xFFFFFFFF, index, 32 - index);
					// mask1 = bfe(0xFFFFFFFF, 0, size - 32);
					// Case where index >= 32:
					// mask0 = bfe(0xFFFFFFFF, 32, 0); // Gives 0
					// mask1 = bfe(0xFFFFFFFF, index - 32, size - index);
					// This is expressed without branches to avoid divergent
					// control flow--hence the complicated min/max expressions.
					// This is further complicated by the fact that if you attempt
					// to bfe out-of-bounds on Metal, undefined behavior is the
					// result.
					statement(builtin_type_decl(bi_type), " ", to_expression(var_id),
					          " = uint4(extract_bits(0xFFFFFFFF, min(",
					          to_expression(builtin_subgroup_invocation_id_id), ", 32u), (uint)max(min((int)",
					          to_expression(builtin_subgroup_size_id), ", 32) - (int)",
					          to_expression(builtin_subgroup_invocation_id_id),
					          ", 0)), extract_bits(0xFFFFFFFF, (uint)max((int)",
					          to_expression(builtin_subgroup_invocation_id_id), " - 32, 0), (uint)max((int)",
					          to_expression(builtin_subgroup_size_id), " - (int)max(",
					          to_expression(builtin_subgroup_invocation_id_id), ", 32u), 0)), uint2(0));");
				});
				break;
			case BuiltInSubgroupGtMask:
				if (msl_options.is_ios())
					SPIRV_CROSS_THROW("Subgroup ballot functionality is unavailable on iOS.");
				if (!msl_options.supports_msl_version(2, 1))
					SPIRV_CROSS_THROW("Subgroup ballot functionality requires Metal 2.1.");
				entry_func.fixup_hooks_in.push_back([=]() {
					// The same logic applies here, except now the index is one
					// more than the subgroup invocation ID.
					statement(builtin_type_decl(bi_type), " ", to_expression(var_id),
					          " = uint4(extract_bits(0xFFFFFFFF, min(",
					          to_expression(builtin_subgroup_invocation_id_id), " + 1, 32u), (uint)max(min((int)",
					          to_expression(builtin_subgroup_size_id), ", 32) - (int)",
					          to_expression(builtin_subgroup_invocation_id_id),
					          " - 1, 0)), extract_bits(0xFFFFFFFF, (uint)max((int)",
					          to_expression(builtin_subgroup_invocation_id_id), " + 1 - 32, 0), (uint)max((int)",
					          to_expression(builtin_subgroup_size_id), " - (int)max(",
					          to_expression(builtin_subgroup_invocation_id_id), " + 1, 32u), 0)), uint2(0));");
				});
				break;
			case BuiltInSubgroupLeMask:
				if (msl_options.is_ios())
					SPIRV_CROSS_THROW("Subgroup ballot functionality is unavailable on iOS.");
				if (!msl_options.supports_msl_version(2, 1))
					SPIRV_CROSS_THROW("Subgroup ballot functionality requires Metal 2.1.");
				entry_func.fixup_hooks_in.push_back([=]() {
					statement(builtin_type_decl(bi_type), " ", to_expression(var_id),
					          " = uint4(extract_bits(0xFFFFFFFF, 0, min(",
					          to_expression(builtin_subgroup_invocation_id_id),
					          " + 1, 32u)), extract_bits(0xFFFFFFFF, 0, (uint)max((int)",
					          to_expression(builtin_subgroup_invocation_id_id), " + 1 - 32, 0)), uint2(0));");
				});
				break;
			case BuiltInSubgroupLtMask:
				if (msl_options.is_ios())
					SPIRV_CROSS_THROW("Subgroup ballot functionality is unavailable on iOS.");
				if (!msl_options.supports_msl_version(2, 1))
					SPIRV_CROSS_THROW("Subgroup ballot functionality requires Metal 2.1.");
				entry_func.fixup_hooks_in.push_back([=]() {
					statement(builtin_type_decl(bi_type), " ", to_expression(var_id),
					          " = uint4(extract_bits(0xFFFFFFFF, 0, min(",
					          to_expression(builtin_subgroup_invocation_id_id),
					          ", 32u)), extract_bits(0xFFFFFFFF, 0, (uint)max((int)",
					          to_expression(builtin_subgroup_invocation_id_id), " - 32, 0)), uint2(0));");
				});
				break;
			default:
				break;
			}
		}
	});
}

// Returns the Metal index of the resource of the specified type as used by the specified variable.
uint32_t CompilerMSL::get_metal_resource_index(SPIRVariable &var, SPIRType::BaseType basetype)
{
	auto &execution = get_entry_point();
	auto &var_dec = ir.meta[var.self].decoration;
	uint32_t var_desc_set = (var.storage == StorageClassPushConstant) ? kPushConstDescSet : var_dec.set;
	uint32_t var_binding = (var.storage == StorageClassPushConstant) ? kPushConstBinding : var_dec.binding;

	// If a matching binding has been specified, find and use it
	auto itr = find_if(begin(resource_bindings), end(resource_bindings),
	                   [&](const pair<MSLResourceBinding, bool> &resource) -> bool {
		                   return var_desc_set == resource.first.desc_set && var_binding == resource.first.binding &&
		                          execution.model == resource.first.stage;
	                   });

	if (itr != end(resource_bindings))
	{
		itr->second = true;
		switch (basetype)
		{
		case SPIRType::Image:
			return itr->first.msl_texture;
		case SPIRType::Sampler:
			return itr->first.msl_sampler;
		default:
			return itr->first.msl_buffer;
		}
	}

	// If there is no explicit mapping of bindings to MSL, use the declared binding.
	if (has_decoration(var.self, DecorationBinding))
	{
		var_binding = get_decoration(var.self, DecorationBinding);
		// Avoid emitting sentinel bindings.
		if (var_binding < 0x80000000u)
			return var_binding;
	}

	uint32_t binding_stride = 1;
	auto &type = get<SPIRType>(var.basetype);
	for (uint32_t i = 0; i < uint32_t(type.array.size()); i++)
		binding_stride *= type.array_size_literal[i] ? type.array[i] : get<SPIRConstant>(type.array[i]).scalar();

	// If a binding has not been specified, revert to incrementing resource indices
	uint32_t resource_index;
	switch (basetype)
	{
	case SPIRType::Image:
		resource_index = next_metal_resource_index_texture;
		next_metal_resource_index_texture += binding_stride;
		break;
	case SPIRType::Sampler:
		resource_index = next_metal_resource_index_sampler;
		next_metal_resource_index_sampler += binding_stride;
		break;
	default:
		resource_index = next_metal_resource_index_buffer;
		next_metal_resource_index_buffer += binding_stride;
		break;
	}
	return resource_index;
}

string CompilerMSL::argument_decl(const SPIRFunction::Parameter &arg)
{
	auto &var = get<SPIRVariable>(arg.id);
	auto &type = get_variable_data_type(var);
	auto &var_type = get<SPIRType>(arg.type);
	StorageClass storage = var_type.storage;
	bool is_pointer = var_type.pointer;

	// If we need to modify the name of the variable, make sure we use the original variable.
	// Our alias is just a shadow variable.
	uint32_t name_id = var.self;
	if (arg.alias_global_variable && var.basevariable)
		name_id = var.basevariable;

	bool constref = !arg.alias_global_variable && is_pointer && arg.write_count == 0;

	bool type_is_image = type.basetype == SPIRType::Image || type.basetype == SPIRType::SampledImage ||
	                     type.basetype == SPIRType::Sampler;

	// Arrays of images/samplers in MSL are always const.
	if (!type.array.empty() && type_is_image)
		constref = true;

	string decl;
	if (constref)
		decl += "const ";

	bool builtin = is_builtin_variable(var);
	if (var.basevariable == stage_in_ptr_var_id || var.basevariable == stage_out_ptr_var_id)
		decl += type_to_glsl(type, arg.id);
	else if (builtin)
		decl += builtin_type_decl(static_cast<BuiltIn>(get_decoration(arg.id, DecorationBuiltIn)));
	else if ((storage == StorageClassUniform || storage == StorageClassStorageBuffer) && is_array(type))
		decl += join(type_to_glsl(type, arg.id), "*");
	else
		decl += type_to_glsl(type, arg.id);

	bool opaque_handle = storage == StorageClassUniformConstant;

	string address_space = get_argument_address_space(var);

	if (!builtin && !opaque_handle && !is_pointer &&
	    (storage == StorageClassFunction || storage == StorageClassGeneric))
	{
		// If the argument is a pure value and not an opaque type, we will pass by value.
		if (is_array(type))
		{
			// We are receiving an array by value. This is problematic.
			// We cannot be sure of the target address space since we are supposed to receive a copy,
			// but this is not possible with MSL without some extra work.
			// We will have to assume we're getting a reference in thread address space.
			// If we happen to get a reference in constant address space, the caller must emit a copy and pass that.
			// Thread const therefore becomes the only logical choice, since we cannot "create" a constant array from
			// non-constant arrays, but we can create thread const from constant.
			decl = string("thread const ") + decl;
			decl += " (&";
			decl += to_expression(name_id);
			decl += ")";
			decl += type_to_array_glsl(type);
		}
		else
		{
			if (!address_space.empty())
				decl = join(address_space, " ", decl);
			decl += " ";
			decl += to_expression(name_id);
		}
	}
	else if (is_array(type) && !type_is_image)
	{
		// Arrays of images and samplers are special cased.
		if (!address_space.empty())
			decl = join(address_space, " ", decl);

		if (msl_options.argument_buffers)
		{
			// An awkward case where we need to emit *more* address space declarations (yay!).
			// An example is where we pass down an array of buffer pointers to leaf functions.
			// It's a constant array containing pointers to constants.
			// The pointer array is always constant however. E.g.
			// device SSBO * constant (&array)[N].
			// const device SSBO * constant (&array)[N].
			// constant SSBO * constant (&array)[N].
			// However, this only matters for argument buffers, since for MSL 1.0 style codegen,
			// we emit the buffer array on stack instead, and that seems to work just fine apparently.
			if (storage == StorageClassUniform || storage == StorageClassStorageBuffer)
				decl += " constant";
		}

		decl += " (&";
		decl += to_expression(name_id);
		decl += ")";
		decl += type_to_array_glsl(type);
	}
	else if (!opaque_handle)
	{
		// If this is going to be a reference to a variable pointer, the address space
		// for the reference has to go before the '&', but after the '*'.
		if (!address_space.empty())
		{
			if (decl.back() == '*')
				decl += join(" ", address_space, " ");
			else
				decl = join(address_space, " ", decl);
		}
		decl += "&";
		decl += " ";
		decl += to_expression(name_id);
	}
	else
	{
		if (!address_space.empty())
			decl = join(address_space, " ", decl);
		decl += " ";
		decl += to_expression(name_id);
	}

	return decl;
}

// If we're currently in the entry point function, and the object
// has a qualified name, use it, otherwise use the standard name.
string CompilerMSL::to_name(uint32_t id, bool allow_alias) const
{
	if (current_function && (current_function->self == ir.default_entry_point))
	{
		auto *m = ir.find_meta(id);
		if (m && !m->decoration.qualified_alias.empty())
			return m->decoration.qualified_alias;
	}
	return Compiler::to_name(id, allow_alias);
}

// Returns a name that combines the name of the struct with the name of the member, except for Builtins
string CompilerMSL::to_qualified_member_name(const SPIRType &type, uint32_t index)
{
	// Don't qualify Builtin names because they are unique and are treated as such when building expressions
	BuiltIn builtin = BuiltInMax;
	if (is_member_builtin(type, index, &builtin))
		return builtin_to_glsl(builtin, type.storage);

	// Strip any underscore prefix from member name
	string mbr_name = to_member_name(type, index);
	size_t startPos = mbr_name.find_first_not_of("_");
	mbr_name = (startPos != string::npos) ? mbr_name.substr(startPos) : "";
	return join(to_name(type.self), "_", mbr_name);
}

// Ensures that the specified name is permanently usable by prepending a prefix
// if the first chars are _ and a digit, which indicate a transient name.
string CompilerMSL::ensure_valid_name(string name, string pfx)
{
	return (name.size() >= 2 && name[0] == '_' && isdigit(name[1])) ? (pfx + name) : name;
}

// Replace all names that match MSL keywords or Metal Standard Library functions.
void CompilerMSL::replace_illegal_names()
{
	// FIXME: MSL and GLSL are doing two different things here.
	// Agree on convention and remove this override.
	static const unordered_set<string> keywords = {
		"kernel",
		"vertex",
		"fragment",
		"compute",
		"bias",
		"assert",
		"VARIABLE_TRACEPOINT",
		"STATIC_DATA_TRACEPOINT",
		"STATIC_DATA_TRACEPOINT_V",
		"METAL_ALIGN",
		"METAL_ASM",
		"METAL_CONST",
		"METAL_DEPRECATED",
		"METAL_ENABLE_IF",
		"METAL_FUNC",
		"METAL_INTERNAL",
		"METAL_NON_NULL_RETURN",
		"METAL_NORETURN",
		"METAL_NOTHROW",
		"METAL_PURE",
		"METAL_UNAVAILABLE",
		"METAL_IMPLICIT",
		"METAL_EXPLICIT",
		"METAL_CONST_ARG",
		"METAL_ARG_UNIFORM",
		"METAL_ZERO_ARG",
		"METAL_VALID_LOD_ARG",
		"METAL_VALID_LEVEL_ARG",
		"METAL_VALID_STORE_ORDER",
		"METAL_VALID_LOAD_ORDER",
		"METAL_VALID_COMPARE_EXCHANGE_FAILURE_ORDER",
		"METAL_COMPATIBLE_COMPARE_EXCHANGE_ORDERS",
		"METAL_VALID_RENDER_TARGET",
		"is_function_constant_defined",
		"CHAR_BIT",
		"SCHAR_MAX",
		"SCHAR_MIN",
		"UCHAR_MAX",
		"CHAR_MAX",
		"CHAR_MIN",
		"USHRT_MAX",
		"SHRT_MAX",
		"SHRT_MIN",
		"UINT_MAX",
		"INT_MAX",
		"INT_MIN",
		"FLT_DIG",
		"FLT_MANT_DIG",
		"FLT_MAX_10_EXP",
		"FLT_MAX_EXP",
		"FLT_MIN_10_EXP",
		"FLT_MIN_EXP",
		"FLT_RADIX",
		"FLT_MAX",
		"FLT_MIN",
		"FLT_EPSILON",
		"FP_ILOGB0",
		"FP_ILOGBNAN",
		"MAXFLOAT",
		"HUGE_VALF",
		"INFINITY",
		"NAN",
		"M_E_F",
		"M_LOG2E_F",
		"M_LOG10E_F",
		"M_LN2_F",
		"M_LN10_F",
		"M_PI_F",
		"M_PI_2_F",
		"M_PI_4_F",
		"M_1_PI_F",
		"M_2_PI_F",
		"M_2_SQRTPI_F",
		"M_SQRT2_F",
		"M_SQRT1_2_F",
		"HALF_DIG",
		"HALF_MANT_DIG",
		"HALF_MAX_10_EXP",
		"HALF_MAX_EXP",
		"HALF_MIN_10_EXP",
		"HALF_MIN_EXP",
		"HALF_RADIX",
		"HALF_MAX",
		"HALF_MIN",
		"HALF_EPSILON",
		"MAXHALF",
		"HUGE_VALH",
		"M_E_H",
		"M_LOG2E_H",
		"M_LOG10E_H",
		"M_LN2_H",
		"M_LN10_H",
		"M_PI_H",
		"M_PI_2_H",
		"M_PI_4_H",
		"M_1_PI_H",
		"M_2_PI_H",
		"M_2_SQRTPI_H",
		"M_SQRT2_H",
		"M_SQRT1_2_H",
		"DBL_DIG",
		"DBL_MANT_DIG",
		"DBL_MAX_10_EXP",
		"DBL_MAX_EXP",
		"DBL_MIN_10_EXP",
		"DBL_MIN_EXP",
		"DBL_RADIX",
		"DBL_MAX",
		"DBL_MIN",
		"DBL_EPSILON",
		"HUGE_VAL",
		"M_E",
		"M_LOG2E",
		"M_LOG10E",
		"M_LN2",
		"M_LN10",
		"M_PI",
		"M_PI_2",
		"M_PI_4",
		"M_1_PI",
		"M_2_PI",
		"M_2_SQRTPI",
		"M_SQRT2",
		"M_SQRT1_2",
		"quad_broadcast",
	};

	static const unordered_set<string> illegal_func_names = {
		"main",
		"saturate",
		"assert",
		"VARIABLE_TRACEPOINT",
		"STATIC_DATA_TRACEPOINT",
		"STATIC_DATA_TRACEPOINT_V",
		"METAL_ALIGN",
		"METAL_ASM",
		"METAL_CONST",
		"METAL_DEPRECATED",
		"METAL_ENABLE_IF",
		"METAL_FUNC",
		"METAL_INTERNAL",
		"METAL_NON_NULL_RETURN",
		"METAL_NORETURN",
		"METAL_NOTHROW",
		"METAL_PURE",
		"METAL_UNAVAILABLE",
		"METAL_IMPLICIT",
		"METAL_EXPLICIT",
		"METAL_CONST_ARG",
		"METAL_ARG_UNIFORM",
		"METAL_ZERO_ARG",
		"METAL_VALID_LOD_ARG",
		"METAL_VALID_LEVEL_ARG",
		"METAL_VALID_STORE_ORDER",
		"METAL_VALID_LOAD_ORDER",
		"METAL_VALID_COMPARE_EXCHANGE_FAILURE_ORDER",
		"METAL_COMPATIBLE_COMPARE_EXCHANGE_ORDERS",
		"METAL_VALID_RENDER_TARGET",
		"is_function_constant_defined",
		"CHAR_BIT",
		"SCHAR_MAX",
		"SCHAR_MIN",
		"UCHAR_MAX",
		"CHAR_MAX",
		"CHAR_MIN",
		"USHRT_MAX",
		"SHRT_MAX",
		"SHRT_MIN",
		"UINT_MAX",
		"INT_MAX",
		"INT_MIN",
		"FLT_DIG",
		"FLT_MANT_DIG",
		"FLT_MAX_10_EXP",
		"FLT_MAX_EXP",
		"FLT_MIN_10_EXP",
		"FLT_MIN_EXP",
		"FLT_RADIX",
		"FLT_MAX",
		"FLT_MIN",
		"FLT_EPSILON",
		"FP_ILOGB0",
		"FP_ILOGBNAN",
		"MAXFLOAT",
		"HUGE_VALF",
		"INFINITY",
		"NAN",
		"M_E_F",
		"M_LOG2E_F",
		"M_LOG10E_F",
		"M_LN2_F",
		"M_LN10_F",
		"M_PI_F",
		"M_PI_2_F",
		"M_PI_4_F",
		"M_1_PI_F",
		"M_2_PI_F",
		"M_2_SQRTPI_F",
		"M_SQRT2_F",
		"M_SQRT1_2_F",
		"HALF_DIG",
		"HALF_MANT_DIG",
		"HALF_MAX_10_EXP",
		"HALF_MAX_EXP",
		"HALF_MIN_10_EXP",
		"HALF_MIN_EXP",
		"HALF_RADIX",
		"HALF_MAX",
		"HALF_MIN",
		"HALF_EPSILON",
		"MAXHALF",
		"HUGE_VALH",
		"M_E_H",
		"M_LOG2E_H",
		"M_LOG10E_H",
		"M_LN2_H",
		"M_LN10_H",
		"M_PI_H",
		"M_PI_2_H",
		"M_PI_4_H",
		"M_1_PI_H",
		"M_2_PI_H",
		"M_2_SQRTPI_H",
		"M_SQRT2_H",
		"M_SQRT1_2_H",
		"DBL_DIG",
		"DBL_MANT_DIG",
		"DBL_MAX_10_EXP",
		"DBL_MAX_EXP",
		"DBL_MIN_10_EXP",
		"DBL_MIN_EXP",
		"DBL_RADIX",
		"DBL_MAX",
		"DBL_MIN",
		"DBL_EPSILON",
		"HUGE_VAL",
		"M_E",
		"M_LOG2E",
		"M_LOG10E",
		"M_LN2",
		"M_LN10",
		"M_PI",
		"M_PI_2",
		"M_PI_4",
		"M_1_PI",
		"M_2_PI",
		"M_2_SQRTPI",
		"M_SQRT2",
		"M_SQRT1_2",
	};

	ir.for_each_typed_id<SPIRVariable>([&](uint32_t self, SPIRVariable &) {
		auto &dec = ir.meta[self].decoration;
		if (keywords.find(dec.alias) != end(keywords))
			dec.alias += "0";
	});

	ir.for_each_typed_id<SPIRFunction>([&](uint32_t self, SPIRFunction &) {
		auto &dec = ir.meta[self].decoration;
		if (illegal_func_names.find(dec.alias) != end(illegal_func_names))
			dec.alias += "0";
	});

	ir.for_each_typed_id<SPIRType>([&](uint32_t self, SPIRType &) {
		for (auto &mbr_dec : ir.meta[self].members)
			if (keywords.find(mbr_dec.alias) != end(keywords))
				mbr_dec.alias += "0";
	});

	for (auto &entry : ir.entry_points)
	{
		// Change both the entry point name and the alias, to keep them synced.
		string &ep_name = entry.second.name;
		if (illegal_func_names.find(ep_name) != end(illegal_func_names))
			ep_name += "0";

		// Always write this because entry point might have been renamed earlier.
		ir.meta[entry.first].decoration.alias = ep_name;
	}

	CompilerGLSL::replace_illegal_names();
}

string CompilerMSL::to_member_reference(uint32_t base, const SPIRType &type, uint32_t index, bool ptr_chain)
{
	auto *var = maybe_get<SPIRVariable>(base);
	// If this is a buffer array, we have to dereference the buffer pointers.
	// Otherwise, if this is a pointer expression, dereference it.

	bool declared_as_pointer = false;

	if (var)
	{
		bool is_buffer_variable = var->storage == StorageClassUniform || var->storage == StorageClassStorageBuffer;
		declared_as_pointer = is_buffer_variable && is_array(get<SPIRType>(var->basetype));
	}

	if (declared_as_pointer || (!ptr_chain && should_dereference(base)))
		return join("->", to_member_name(type, index));
	else
		return join(".", to_member_name(type, index));
}

string CompilerMSL::to_qualifiers_glsl(uint32_t id)
{
	string quals;

	auto &type = expression_type(id);
	if (type.storage == StorageClassWorkgroup)
		quals += "threadgroup ";

	return quals;
}

// The optional id parameter indicates the object whose type we are trying
// to find the description for. It is optional. Most type descriptions do not
// depend on a specific object's use of that type.
string CompilerMSL::type_to_glsl(const SPIRType &type, uint32_t id)
{
	string type_name;

	// Pointer?
	if (type.pointer)
	{
		type_name = join(get_type_address_space(type, id), " ", type_to_glsl(get<SPIRType>(type.parent_type), id));
		switch (type.basetype)
		{
		case SPIRType::Image:
		case SPIRType::SampledImage:
		case SPIRType::Sampler:
			// These are handles.
			break;
		default:
			// Anything else can be a raw pointer.
			type_name += "*";
			break;
		}
		return type_name;
	}

	switch (type.basetype)
	{
	case SPIRType::Struct:
		// Need OpName lookup here to get a "sensible" name for a struct.
		return to_name(type.self);

	case SPIRType::Image:
	case SPIRType::SampledImage:
		return image_type_glsl(type, id);

	case SPIRType::Sampler:
		return sampler_type(type);

	case SPIRType::Void:
		return "void";

	case SPIRType::AtomicCounter:
		return "atomic_uint";

	case SPIRType::ControlPointArray:
		return join("patch_control_point<", type_to_glsl(get<SPIRType>(type.parent_type), id), ">");

	// Scalars
	case SPIRType::Boolean:
		type_name = "bool";
		break;
	case SPIRType::Char:
	case SPIRType::SByte:
		type_name = "char";
		break;
	case SPIRType::UByte:
		type_name = "uchar";
		break;
	case SPIRType::Short:
		type_name = "short";
		break;
	case SPIRType::UShort:
		type_name = "ushort";
		break;
	case SPIRType::Int:
		type_name = "int";
		break;
	case SPIRType::UInt:
		type_name = "uint";
		break;
	case SPIRType::Int64:
		type_name = "long"; // Currently unsupported
		break;
	case SPIRType::UInt64:
		type_name = "size_t";
		break;
	case SPIRType::Half:
		type_name = "half";
		break;
	case SPIRType::Float:
		type_name = "float";
		break;
	case SPIRType::Double:
		type_name = "double"; // Currently unsupported
		break;

	default:
		return "unknown_type";
	}

	// Matrix?
	if (type.columns > 1)
		type_name += to_string(type.columns) + "x";

	// Vector or Matrix?
	if (type.vecsize > 1)
		type_name += to_string(type.vecsize);

	return type_name;
}

std::string CompilerMSL::sampler_type(const SPIRType &type)
{
	if (!type.array.empty())
	{
		if (!msl_options.supports_msl_version(2))
			SPIRV_CROSS_THROW("MSL 2.0 or greater is required for arrays of samplers.");

		if (type.array.size() > 1)
			SPIRV_CROSS_THROW("Arrays of arrays of samplers are not supported in MSL.");

		// Arrays of samplers in MSL must be declared with a special array<T, N> syntax ala C++11 std::array.
		uint32_t array_size = to_array_size_literal(type);
		if (array_size == 0)
			SPIRV_CROSS_THROW("Unsized array of samplers is not supported in MSL.");

		auto &parent = get<SPIRType>(get_pointee_type(type).parent_type);
		return join("array<", sampler_type(parent), ", ", array_size, ">");
	}
	else
		return "sampler";
}

// Returns an MSL string describing the SPIR-V image type
string CompilerMSL::image_type_glsl(const SPIRType &type, uint32_t id)
{
	auto *var = maybe_get<SPIRVariable>(id);
	if (var && var->basevariable)
	{
		// For comparison images, check against the base variable,
		// and not the fake ID which might have been generated for this variable.
		id = var->basevariable;
	}

	if (!type.array.empty())
	{
		uint32_t major = 2, minor = 0;
		if (msl_options.is_ios())
		{
			major = 1;
			minor = 2;
		}
		if (!msl_options.supports_msl_version(major, minor))
		{
			if (msl_options.is_ios())
				SPIRV_CROSS_THROW("MSL 1.2 or greater is required for arrays of textures.");
			else
				SPIRV_CROSS_THROW("MSL 2.0 or greater is required for arrays of textures.");
		}

		if (type.array.size() > 1)
			SPIRV_CROSS_THROW("Arrays of arrays of textures are not supported in MSL.");

		// Arrays of images in MSL must be declared with a special array<T, N> syntax ala C++11 std::array.
		uint32_t array_size = to_array_size_literal(type);
		if (array_size == 0)
			SPIRV_CROSS_THROW("Unsized array of images is not supported in MSL.");

		auto &parent = get<SPIRType>(get_pointee_type(type).parent_type);
		return join("array<", image_type_glsl(parent, id), ", ", array_size, ">");
	}

	string img_type_name;

	// Bypass pointers because we need the real image struct
	auto &img_type = get<SPIRType>(type.self).image;
	if (image_is_comparison(type, id))
	{
		switch (img_type.dim)
		{
		case Dim1D:
			img_type_name += "depth1d_unsupported_by_metal";
			break;
		case Dim2D:
			if (img_type.ms && img_type.arrayed)
			{
				if (!msl_options.supports_msl_version(2, 1))
					SPIRV_CROSS_THROW("Multisampled array textures are supported from 2.1.");
				img_type_name += "depth2d_ms_array";
			}
			else if (img_type.ms)
				img_type_name += "depth2d_ms";
			else if (img_type.arrayed)
				img_type_name += "depth2d_array";
			else
				img_type_name += "depth2d";
			break;
		case Dim3D:
			img_type_name += "depth3d_unsupported_by_metal";
			break;
		case DimCube:
			img_type_name += (img_type.arrayed ? "depthcube_array" : "depthcube");
			break;
		default:
			img_type_name += "unknown_depth_texture_type";
			break;
		}
	}
	else
	{
		switch (img_type.dim)
		{
		case Dim1D:
			img_type_name += (img_type.arrayed ? "texture1d_array" : "texture1d");
			break;
		case DimBuffer:
			if (img_type.ms || img_type.arrayed)
				SPIRV_CROSS_THROW("Cannot use texel buffers with multisampling or array layers.");

			if (msl_options.texture_buffer_native)
			{
				if (!msl_options.supports_msl_version(2, 1))
					SPIRV_CROSS_THROW("Native texture_buffer type is only supported in MSL 2.1.");
				img_type_name = "texture_buffer";
			}
			else
				img_type_name += "texture2d";
			break;
		case Dim2D:
		case DimSubpassData:
			if (img_type.ms && img_type.arrayed)
			{
				if (!msl_options.supports_msl_version(2, 1))
					SPIRV_CROSS_THROW("Multisampled array textures are supported from 2.1.");
				img_type_name += "texture2d_ms_array";
			}
			else if (img_type.ms)
				img_type_name += "texture2d_ms";
			else if (img_type.arrayed)
				img_type_name += "texture2d_array";
			else
				img_type_name += "texture2d";
			break;
		case Dim3D:
			img_type_name += "texture3d";
			break;
		case DimCube:
			img_type_name += (img_type.arrayed ? "texturecube_array" : "texturecube");
			break;
		default:
			img_type_name += "unknown_texture_type";
			break;
		}
	}

	// Append the pixel type
	img_type_name += "<";
	img_type_name += type_to_glsl(get<SPIRType>(img_type.type));

	// For unsampled images, append the sample/read/write access qualifier.
	// For kernel images, the access qualifier my be supplied directly by SPIR-V.
	// Otherwise it may be set based on whether the image is read from or written to within the shader.
	if (type.basetype == SPIRType::Image && type.image.sampled == 2 && type.image.dim != DimSubpassData)
	{
		switch (img_type.access)
		{
		case AccessQualifierReadOnly:
			img_type_name += ", access::read";
			break;

		case AccessQualifierWriteOnly:
			img_type_name += ", access::write";
			break;

		case AccessQualifierReadWrite:
			img_type_name += ", access::read_write";
			break;

		default:
		{
			auto *p_var = maybe_get_backing_variable(id);
			if (p_var && p_var->basevariable)
				p_var = maybe_get<SPIRVariable>(p_var->basevariable);
			if (p_var && !has_decoration(p_var->self, DecorationNonWritable))
			{
				img_type_name += ", access::";

				if (!has_decoration(p_var->self, DecorationNonReadable))
					img_type_name += "read_";

				img_type_name += "write";
			}
			break;
		}
		}
	}

	img_type_name += ">";

	return img_type_name;
}

void CompilerMSL::emit_subgroup_op(const Instruction &i)
{
	const uint32_t *ops = stream(i);
	auto op = static_cast<Op>(i.op);

	// Metal 2.0 is required. iOS only supports quad ops. macOS only supports
	// broadcast and shuffle on 10.13 (2.0), with full support in 10.14 (2.1).
	// Note that iOS makes no distinction between a quad-group and a subgroup;
	// all subgroups are quad-groups there.
	if (!msl_options.supports_msl_version(2))
		SPIRV_CROSS_THROW("Subgroups are only supported in Metal 2.0 and up.");

	if (msl_options.is_ios())
	{
		switch (op)
		{
		default:
			SPIRV_CROSS_THROW("iOS only supports quad-group operations.");
		case OpGroupNonUniformBroadcast:
		case OpGroupNonUniformShuffle:
		case OpGroupNonUniformShuffleXor:
		case OpGroupNonUniformShuffleUp:
		case OpGroupNonUniformShuffleDown:
		case OpGroupNonUniformQuadSwap:
		case OpGroupNonUniformQuadBroadcast:
			break;
		}
	}

	if (msl_options.is_macos() && !msl_options.supports_msl_version(2, 1))
	{
		switch (op)
		{
		default:
			SPIRV_CROSS_THROW("Subgroup ops beyond broadcast and shuffle on macOS require Metal 2.0 and up.");
		case OpGroupNonUniformBroadcast:
		case OpGroupNonUniformShuffle:
		case OpGroupNonUniformShuffleXor:
		case OpGroupNonUniformShuffleUp:
		case OpGroupNonUniformShuffleDown:
			break;
		}
	}

	uint32_t result_type = ops[0];
	uint32_t id = ops[1];

	auto scope = static_cast<Scope>(get<SPIRConstant>(ops[2]).scalar());
	if (scope != ScopeSubgroup)
		SPIRV_CROSS_THROW("Only subgroup scope is supported.");

	switch (op)
	{
	case OpGroupNonUniformElect:
		emit_op(result_type, id, "simd_is_first()", true);
		break;

	case OpGroupNonUniformBroadcast:
		emit_binary_func_op(result_type, id, ops[3], ops[4],
		                    msl_options.is_ios() ? "quad_broadcast" : "simd_broadcast");
		break;

	case OpGroupNonUniformBroadcastFirst:
		emit_unary_func_op(result_type, id, ops[3], "simd_broadcast_first");
		break;

	case OpGroupNonUniformBallot:
		emit_unary_func_op(result_type, id, ops[3], "spvSubgroupBallot");
		break;

	case OpGroupNonUniformInverseBallot:
		emit_binary_func_op(result_type, id, ops[3], builtin_subgroup_invocation_id_id, "spvSubgroupBallotBitExtract");
		break;

	case OpGroupNonUniformBallotBitExtract:
		emit_binary_func_op(result_type, id, ops[3], ops[4], "spvSubgroupBallotBitExtract");
		break;

	case OpGroupNonUniformBallotFindLSB:
		emit_unary_func_op(result_type, id, ops[3], "spvSubgroupBallotFindLSB");
		break;

	case OpGroupNonUniformBallotFindMSB:
		emit_unary_func_op(result_type, id, ops[3], "spvSubgroupBallotFindMSB");
		break;

	case OpGroupNonUniformBallotBitCount:
	{
		auto operation = static_cast<GroupOperation>(ops[3]);
		if (operation == GroupOperationReduce)
			emit_unary_func_op(result_type, id, ops[4], "spvSubgroupBallotBitCount");
		else if (operation == GroupOperationInclusiveScan)
			emit_binary_func_op(result_type, id, ops[4], builtin_subgroup_invocation_id_id,
			                    "spvSubgroupBallotInclusiveBitCount");
		else if (operation == GroupOperationExclusiveScan)
			emit_binary_func_op(result_type, id, ops[4], builtin_subgroup_invocation_id_id,
			                    "spvSubgroupBallotExclusiveBitCount");
		else
			SPIRV_CROSS_THROW("Invalid BitCount operation.");
		break;
	}

	case OpGroupNonUniformShuffle:
		emit_binary_func_op(result_type, id, ops[3], ops[4], msl_options.is_ios() ? "quad_shuffle" : "simd_shuffle");
		break;

	case OpGroupNonUniformShuffleXor:
		emit_binary_func_op(result_type, id, ops[3], ops[4],
		                    msl_options.is_ios() ? "quad_shuffle_xor" : "simd_shuffle_xor");
		break;

	case OpGroupNonUniformShuffleUp:
		emit_binary_func_op(result_type, id, ops[3], ops[4],
		                    msl_options.is_ios() ? "quad_shuffle_up" : "simd_shuffle_up");
		break;

	case OpGroupNonUniformShuffleDown:
		emit_binary_func_op(result_type, id, ops[3], ops[4],
		                    msl_options.is_ios() ? "quad_shuffle_down" : "simd_shuffle_down");
		break;

	case OpGroupNonUniformAll:
		emit_unary_func_op(result_type, id, ops[3], "simd_all");
		break;

	case OpGroupNonUniformAny:
		emit_unary_func_op(result_type, id, ops[3], "simd_any");
		break;

	case OpGroupNonUniformAllEqual:
		emit_unary_func_op(result_type, id, ops[3], "spvSubgroupAllEqual");
		break;

		// clang-format off
#define MSL_GROUP_OP(op, msl_op) \
case OpGroupNonUniform##op: \
	{ \
		auto operation = static_cast<GroupOperation>(ops[3]); \
		if (operation == GroupOperationReduce) \
			emit_unary_func_op(result_type, id, ops[4], "simd_" #msl_op); \
		else if (operation == GroupOperationInclusiveScan) \
			emit_unary_func_op(result_type, id, ops[4], "simd_prefix_inclusive_" #msl_op); \
		else if (operation == GroupOperationExclusiveScan) \
			emit_unary_func_op(result_type, id, ops[4], "simd_prefix_exclusive_" #msl_op); \
		else if (operation == GroupOperationClusteredReduce) \
		{ \
			/* Only cluster sizes of 4 are supported. */ \
			uint32_t cluster_size = get<SPIRConstant>(ops[5]).scalar(); \
			if (cluster_size != 4) \
				SPIRV_CROSS_THROW("Metal only supports quad ClusteredReduce."); \
			emit_unary_func_op(result_type, id, ops[4], "quad_" #msl_op); \
		} \
		else \
			SPIRV_CROSS_THROW("Invalid group operation."); \
		break; \
	}
	MSL_GROUP_OP(FAdd, sum)
	MSL_GROUP_OP(FMul, product)
	MSL_GROUP_OP(IAdd, sum)
	MSL_GROUP_OP(IMul, product)
#undef MSL_GROUP_OP
	// The others, unfortunately, don't support InclusiveScan or ExclusiveScan.
#define MSL_GROUP_OP(op, msl_op) \
case OpGroupNonUniform##op: \
	{ \
		auto operation = static_cast<GroupOperation>(ops[3]); \
		if (operation == GroupOperationReduce) \
			emit_unary_func_op(result_type, id, ops[4], "simd_" #msl_op); \
		else if (operation == GroupOperationInclusiveScan) \
			SPIRV_CROSS_THROW("Metal doesn't support InclusiveScan for OpGroupNonUniform" #op "."); \
		else if (operation == GroupOperationExclusiveScan) \
			SPIRV_CROSS_THROW("Metal doesn't support ExclusiveScan for OpGroupNonUniform" #op "."); \
		else if (operation == GroupOperationClusteredReduce) \
		{ \
			/* Only cluster sizes of 4 are supported. */ \
			uint32_t cluster_size = get<SPIRConstant>(ops[5]).scalar(); \
			if (cluster_size != 4) \
				SPIRV_CROSS_THROW("Metal only supports quad ClusteredReduce."); \
			emit_unary_func_op(result_type, id, ops[4], "quad_" #msl_op); \
		} \
		else \
			SPIRV_CROSS_THROW("Invalid group operation."); \
		break; \
	}
	MSL_GROUP_OP(FMin, min)
	MSL_GROUP_OP(FMax, max)
	MSL_GROUP_OP(SMin, min)
	MSL_GROUP_OP(SMax, max)
	MSL_GROUP_OP(UMin, min)
	MSL_GROUP_OP(UMax, max)
	MSL_GROUP_OP(BitwiseAnd, and)
	MSL_GROUP_OP(BitwiseOr, or)
	MSL_GROUP_OP(BitwiseXor, xor)
	MSL_GROUP_OP(LogicalAnd, and)
	MSL_GROUP_OP(LogicalOr, or)
	MSL_GROUP_OP(LogicalXor, xor)
		// clang-format on

	case OpGroupNonUniformQuadSwap:
	{
		// We can implement this easily based on the following table giving
		// the target lane ID from the direction and current lane ID:
		//        Direction
		//      | 0 | 1 | 2 |
		//   ---+---+---+---+
		// L 0  | 1   2   3
		// a 1  | 0   3   2
		// n 2  | 3   0   1
		// e 3  | 2   1   0
		// Notice that target = source ^ (direction + 1).
		uint32_t mask = get<SPIRConstant>(ops[4]).scalar() + 1;
		uint32_t mask_id = ir.increase_bound_by(1);
		set<SPIRConstant>(mask_id, expression_type_id(ops[4]), mask, false);
		emit_binary_func_op(result_type, id, ops[3], mask_id, "quad_shuffle_xor");
		break;
	}

	case OpGroupNonUniformQuadBroadcast:
		emit_binary_func_op(result_type, id, ops[3], ops[4], "quad_broadcast");
		break;

	default:
		SPIRV_CROSS_THROW("Invalid opcode for subgroup.");
	}

	register_control_dependent_expression(id);
}

string CompilerMSL::bitcast_glsl_op(const SPIRType &out_type, const SPIRType &in_type)
{
	if (out_type.basetype == in_type.basetype)
		return "";

	assert(out_type.basetype != SPIRType::Boolean);
	assert(in_type.basetype != SPIRType::Boolean);

	bool integral_cast = type_is_integral(out_type) && type_is_integral(in_type);
	bool same_size_cast = out_type.width == in_type.width;

	if (integral_cast && same_size_cast)
	{
		// Trivial bitcast case, casts between integers.
		return type_to_glsl(out_type);
	}
	else
	{
		// Fall back to the catch-all bitcast in MSL.
		return "as_type<" + type_to_glsl(out_type) + ">";
	}
}

// Returns an MSL string identifying the name of a SPIR-V builtin.
// Output builtins are qualified with the name of the stage out structure.
string CompilerMSL::builtin_to_glsl(BuiltIn builtin, StorageClass storage)
{
	switch (builtin)
	{

	// Override GLSL compiler strictness
	case BuiltInVertexId:
		return "gl_VertexID";
	case BuiltInInstanceId:
		return "gl_InstanceID";
	case BuiltInVertexIndex:
		return "gl_VertexIndex";
	case BuiltInInstanceIndex:
		return "gl_InstanceIndex";
	case BuiltInBaseVertex:
		return "gl_BaseVertex";
	case BuiltInBaseInstance:
		return "gl_BaseInstance";
	case BuiltInDrawIndex:
		SPIRV_CROSS_THROW("DrawIndex is not supported in MSL.");

	// When used in the entry function, output builtins are qualified with output struct name.
	// Test storage class as NOT Input, as output builtins might be part of generic type.
	// Also don't do this for tessellation control shaders.
	case BuiltInViewportIndex:
		if (!msl_options.supports_msl_version(2, 0))
			SPIRV_CROSS_THROW("ViewportIndex requires Metal 2.0.");
		/* fallthrough */
	case BuiltInPosition:
	case BuiltInPointSize:
	case BuiltInClipDistance:
	case BuiltInCullDistance:
	case BuiltInLayer:
	case BuiltInFragDepth:
	case BuiltInSampleMask:
		if (get_execution_model() == ExecutionModelTessellationControl)
			break;
		if (storage != StorageClassInput && current_function && (current_function->self == ir.default_entry_point))
			return stage_out_var_name + "." + CompilerGLSL::builtin_to_glsl(builtin, storage);

		break;

	case BuiltInTessLevelOuter:
		if (get_execution_model() == ExecutionModelTessellationEvaluation)
		{
			if (storage != StorageClassOutput && !get_entry_point().flags.get(ExecutionModeTriangles) &&
			    current_function && (current_function->self == ir.default_entry_point))
				return join(patch_stage_in_var_name, ".", CompilerGLSL::builtin_to_glsl(builtin, storage));
			else
				break;
		}
		if (storage != StorageClassInput && current_function && (current_function->self == ir.default_entry_point))
			return join(tess_factor_buffer_var_name, "[", to_expression(builtin_primitive_id_id),
			            "].edgeTessellationFactor");
		break;

	case BuiltInTessLevelInner:
		if (get_execution_model() == ExecutionModelTessellationEvaluation)
		{
			if (storage != StorageClassOutput && !get_entry_point().flags.get(ExecutionModeTriangles) &&
			    current_function && (current_function->self == ir.default_entry_point))
				return join(patch_stage_in_var_name, ".", CompilerGLSL::builtin_to_glsl(builtin, storage));
			else
				break;
		}
		if (storage != StorageClassInput && current_function && (current_function->self == ir.default_entry_point))
			return join(tess_factor_buffer_var_name, "[", to_expression(builtin_primitive_id_id),
			            "].insideTessellationFactor");
		break;

	default:
		break;
	}

	return CompilerGLSL::builtin_to_glsl(builtin, storage);
}

// Returns an MSL string attribute qualifer for a SPIR-V builtin
string CompilerMSL::builtin_qualifier(BuiltIn builtin)
{
	auto &execution = get_entry_point();

	switch (builtin)
	{
	// Vertex function in
	case BuiltInVertexId:
		return "vertex_id";
	case BuiltInVertexIndex:
		return "vertex_id";
	case BuiltInBaseVertex:
		return "base_vertex";
	case BuiltInInstanceId:
		return "instance_id";
	case BuiltInInstanceIndex:
		return "instance_id";
	case BuiltInBaseInstance:
		return "base_instance";
	case BuiltInDrawIndex:
		SPIRV_CROSS_THROW("DrawIndex is not supported in MSL.");

	// Vertex function out
	case BuiltInClipDistance:
		return "clip_distance";
	case BuiltInPointSize:
		return "point_size";
	case BuiltInPosition:
		return "position";
	case BuiltInLayer:
		return "render_target_array_index";
	case BuiltInViewportIndex:
		if (!msl_options.supports_msl_version(2, 0))
			SPIRV_CROSS_THROW("ViewportIndex requires Metal 2.0.");
		return "viewport_array_index";

	// Tess. control function in
	case BuiltInInvocationId:
		return "thread_index_in_threadgroup";
	case BuiltInPatchVertices:
		// Shouldn't be reached.
		SPIRV_CROSS_THROW("PatchVertices is derived from the auxiliary buffer in MSL.");
	case BuiltInPrimitiveId:
		switch (execution.model)
		{
		case ExecutionModelTessellationControl:
			return "threadgroup_position_in_grid";
		case ExecutionModelTessellationEvaluation:
			return "patch_id";
		default:
			SPIRV_CROSS_THROW("PrimitiveId is not supported in this execution model.");
		}

	// Tess. control function out
	case BuiltInTessLevelOuter:
	case BuiltInTessLevelInner:
		// Shouldn't be reached.
		SPIRV_CROSS_THROW("Tessellation levels are handled specially in MSL.");

	// Tess. evaluation function in
	case BuiltInTessCoord:
		return "position_in_patch";

	// Fragment function in
	case BuiltInFrontFacing:
		return "front_facing";
	case BuiltInPointCoord:
		return "point_coord";
	case BuiltInFragCoord:
		return "position";
	case BuiltInSampleId:
		return "sample_id";
	case BuiltInSampleMask:
		return "sample_mask";
	case BuiltInSamplePosition:
		// Shouldn't be reached.
		SPIRV_CROSS_THROW("Sample position is retrieved by a function in MSL.");

	// Fragment function out
	case BuiltInFragDepth:
		if (execution.flags.get(ExecutionModeDepthGreater))
			return "depth(greater)";
		else if (execution.flags.get(ExecutionModeDepthLess))
			return "depth(less)";
		else
			return "depth(any)";

	// Compute function in
	case BuiltInGlobalInvocationId:
		return "thread_position_in_grid";

	case BuiltInWorkgroupId:
		return "threadgroup_position_in_grid";

	case BuiltInNumWorkgroups:
		return "threadgroups_per_grid";

	case BuiltInLocalInvocationId:
		return "thread_position_in_threadgroup";

	case BuiltInLocalInvocationIndex:
		return "thread_index_in_threadgroup";

	case BuiltInSubgroupSize:
		return "thread_execution_width";

	case BuiltInNumSubgroups:
		if (!msl_options.supports_msl_version(2))
			SPIRV_CROSS_THROW("Subgroup builtins require Metal 2.0.");
		return msl_options.is_ios() ? "quadgroups_per_threadgroup" : "simdgroups_per_threadgroup";

	case BuiltInSubgroupId:
		if (!msl_options.supports_msl_version(2))
			SPIRV_CROSS_THROW("Subgroup builtins require Metal 2.0.");
		return msl_options.is_ios() ? "quadgroup_index_in_threadgroup" : "simdgroup_index_in_threadgroup";

	case BuiltInSubgroupLocalInvocationId:
		if (!msl_options.supports_msl_version(2))
			SPIRV_CROSS_THROW("Subgroup builtins require Metal 2.0.");
		return msl_options.is_ios() ? "thread_index_in_quadgroup" : "thread_index_in_simdgroup";

	case BuiltInSubgroupEqMask:
	case BuiltInSubgroupGeMask:
	case BuiltInSubgroupGtMask:
	case BuiltInSubgroupLeMask:
	case BuiltInSubgroupLtMask:
		// Shouldn't be reached.
		SPIRV_CROSS_THROW("Subgroup ballot masks are handled specially in MSL.");

	default:
		return "unsupported-built-in";
	}
}

// Returns an MSL string type declaration for a SPIR-V builtin
string CompilerMSL::builtin_type_decl(BuiltIn builtin)
{
	const SPIREntryPoint &execution = get_entry_point();
	switch (builtin)
	{
	// Vertex function in
	case BuiltInVertexId:
		return "uint";
	case BuiltInVertexIndex:
		return "uint";
	case BuiltInBaseVertex:
		return "uint";
	case BuiltInInstanceId:
		return "uint";
	case BuiltInInstanceIndex:
		return "uint";
	case BuiltInBaseInstance:
		return "uint";
	case BuiltInDrawIndex:
		SPIRV_CROSS_THROW("DrawIndex is not supported in MSL.");

	// Vertex function out
	case BuiltInClipDistance:
		return "float";
	case BuiltInPointSize:
		return "float";
	case BuiltInPosition:
		return "float4";
	case BuiltInLayer:
		return "uint";
	case BuiltInViewportIndex:
		if (!msl_options.supports_msl_version(2, 0))
			SPIRV_CROSS_THROW("ViewportIndex requires Metal 2.0.");
		return "uint";

	// Tess. control function in
	case BuiltInInvocationId:
		return "uint";
	case BuiltInPatchVertices:
		return "uint";
	case BuiltInPrimitiveId:
		return "uint";

	// Tess. control function out
	case BuiltInTessLevelInner:
		if (execution.model == ExecutionModelTessellationEvaluation)
			return !execution.flags.get(ExecutionModeTriangles) ? "float2" : "float";
		return "half";
	case BuiltInTessLevelOuter:
		if (execution.model == ExecutionModelTessellationEvaluation)
			return !execution.flags.get(ExecutionModeTriangles) ? "float4" : "float";
		return "half";

	// Tess. evaluation function in
	case BuiltInTessCoord:
		return execution.flags.get(ExecutionModeTriangles) ? "float3" : "float2";

	// Fragment function in
	case BuiltInFrontFacing:
		return "bool";
	case BuiltInPointCoord:
		return "float2";
	case BuiltInFragCoord:
		return "float4";
	case BuiltInSampleId:
		return "uint";
	case BuiltInSampleMask:
		return "uint";
	case BuiltInSamplePosition:
		return "float2";

	// Fragment function out
	case BuiltInFragDepth:
		return "float";

	// Compute function in
	case BuiltInGlobalInvocationId:
	case BuiltInLocalInvocationId:
	case BuiltInNumWorkgroups:
	case BuiltInWorkgroupId:
		return "uint3";
	case BuiltInLocalInvocationIndex:
	case BuiltInNumSubgroups:
	case BuiltInSubgroupId:
	case BuiltInSubgroupSize:
	case BuiltInSubgroupLocalInvocationId:
		return "uint";
	case BuiltInSubgroupEqMask:
	case BuiltInSubgroupGeMask:
	case BuiltInSubgroupGtMask:
	case BuiltInSubgroupLeMask:
	case BuiltInSubgroupLtMask:
		return "uint4";

	case BuiltInHelperInvocation:
		return "bool";

	default:
		return "unsupported-built-in-type";
	}
}

// Returns the declaration of a built-in argument to a function
string CompilerMSL::built_in_func_arg(BuiltIn builtin, bool prefix_comma)
{
	string bi_arg;
	if (prefix_comma)
		bi_arg += ", ";

	bi_arg += builtin_type_decl(builtin);
	bi_arg += " " + builtin_to_glsl(builtin, StorageClassInput);
	bi_arg += " [[" + builtin_qualifier(builtin) + "]]";

	return bi_arg;
}

// Returns the byte size of a struct member.
size_t CompilerMSL::get_declared_struct_member_size(const SPIRType &struct_type, uint32_t index) const
{
	auto &type = get<SPIRType>(struct_type.member_types[index]);

	switch (type.basetype)
	{
	case SPIRType::Unknown:
	case SPIRType::Void:
	case SPIRType::AtomicCounter:
	case SPIRType::Image:
	case SPIRType::SampledImage:
	case SPIRType::Sampler:
		SPIRV_CROSS_THROW("Querying size of opaque object.");

	default:
	{
		// For arrays, we can use ArrayStride to get an easy check.
		// Runtime arrays will have zero size so force to min of one.
		if (!type.array.empty())
		{
			uint32_t array_size = to_array_size_literal(type);
			return type_struct_member_array_stride(struct_type, index) * max(array_size, 1u);
		}

		if (type.basetype == SPIRType::Struct)
		{
			// The size of a struct in Metal is aligned up to its natural alignment.
			auto size = get_declared_struct_size(type);
			auto alignment = get_declared_struct_member_alignment(struct_type, index);
			return (size + alignment - 1) & ~(alignment - 1);
		}

		uint32_t component_size = type.width / 8;
		uint32_t vecsize = type.vecsize;
		uint32_t columns = type.columns;

		// An unpacked 3-element vector or matrix column is the same memory size as a 4-element.
		if (vecsize == 3 && !has_extended_member_decoration(struct_type.self, index, SPIRVCrossDecorationPacked))
			vecsize = 4;

		return component_size * vecsize * columns;
	}
	}
}

// Returns the byte alignment of a struct member.
size_t CompilerMSL::get_declared_struct_member_alignment(const SPIRType &struct_type, uint32_t index) const
{
	auto &type = get<SPIRType>(struct_type.member_types[index]);

	switch (type.basetype)
	{
	case SPIRType::Unknown:
	case SPIRType::Void:
	case SPIRType::AtomicCounter:
	case SPIRType::Image:
	case SPIRType::SampledImage:
	case SPIRType::Sampler:
		SPIRV_CROSS_THROW("Querying alignment of opaque object.");

	case SPIRType::Struct:
	{
		// In MSL, a struct's alignment is equal to the maximum alignment of any of its members.
		uint32_t alignment = 1;
		for (uint32_t i = 0; i < type.member_types.size(); i++)
			alignment = max(alignment, uint32_t(get_declared_struct_member_alignment(type, i)));
		return alignment;
	}

	default:
	{
		// Alignment of packed type is the same as the underlying component or column size.
		// Alignment of unpacked type is the same as the vector size.
		// Alignment of 3-elements vector is the same as 4-elements (including packed using column).
		if (member_is_packed_type(struct_type, index))
		{
			// This is getting pretty complicated.
			// The special case of array of float/float2 needs to be handled here.
			uint32_t packed_type_id =
			    get_extended_member_decoration(struct_type.self, index, SPIRVCrossDecorationPackedType);
			const SPIRType *packed_type = packed_type_id != 0 ? &get<SPIRType>(packed_type_id) : nullptr;
			if (packed_type && is_array(*packed_type) && !is_matrix(*packed_type) &&
			    packed_type->basetype != SPIRType::Struct)
				return (packed_type->width / 8) * 4;
			else
				return (type.width / 8) * (type.columns == 3 ? 4 : type.columns);
		}
		else
			return (type.width / 8) * (type.vecsize == 3 ? 4 : type.vecsize);
	}
	}
}

bool CompilerMSL::skip_argument(uint32_t) const
{
	return false;
}

void CompilerMSL::analyze_sampled_image_usage()
{
	if (msl_options.swizzle_texture_samples)
	{
		SampledImageScanner scanner(*this);
		traverse_all_reachable_opcodes(get<SPIRFunction>(ir.default_entry_point), scanner);
	}
}

bool CompilerMSL::SampledImageScanner::handle(spv::Op opcode, const uint32_t *args, uint32_t length)
{
	switch (opcode)
	{
	case OpLoad:
	case OpImage:
	case OpSampledImage:
	{
		if (length < 3)
			return false;

		uint32_t result_type = args[0];
		auto &type = compiler.get<SPIRType>(result_type);
		if ((type.basetype != SPIRType::Image && type.basetype != SPIRType::SampledImage) || type.image.sampled != 1)
			return true;

		uint32_t id = args[1];
		compiler.set<SPIRExpression>(id, "", result_type, true);
		break;
	}
	case OpImageSampleExplicitLod:
	case OpImageSampleProjExplicitLod:
	case OpImageSampleDrefExplicitLod:
	case OpImageSampleProjDrefExplicitLod:
	case OpImageSampleImplicitLod:
	case OpImageSampleProjImplicitLod:
	case OpImageSampleDrefImplicitLod:
	case OpImageSampleProjDrefImplicitLod:
	case OpImageFetch:
	case OpImageGather:
	case OpImageDrefGather:
		compiler.has_sampled_images =
		    compiler.has_sampled_images || compiler.is_sampled_image_type(compiler.expression_type(args[2]));
		compiler.needs_swizzle_buffer_def = compiler.needs_swizzle_buffer_def || compiler.has_sampled_images;
		break;
	default:
		break;
	}
	return true;
}

bool CompilerMSL::OpCodePreprocessor::handle(Op opcode, const uint32_t *args, uint32_t length)
{
	// Since MSL exists in a single execution scope, function prototype declarations are not
	// needed, and clutter the output. If secondary functions are output (either as a SPIR-V
	// function implementation or as indicated by the presence of OpFunctionCall), then set
	// suppress_missing_prototypes to suppress compiler warnings of missing function prototypes.

	// Mark if the input requires the implementation of an SPIR-V function that does not exist in Metal.
	SPVFuncImpl spv_func = get_spv_func_impl(opcode, args);
	if (spv_func != SPVFuncImplNone)
	{
		compiler.spv_function_implementations.insert(spv_func);
		suppress_missing_prototypes = true;
	}

	switch (opcode)
	{

	case OpFunctionCall:
		suppress_missing_prototypes = true;
		break;

	case OpImageWrite:
		uses_resource_write = true;
		break;

	case OpStore:
		check_resource_write(args[0]);
		break;

	case OpAtomicExchange:
	case OpAtomicCompareExchange:
	case OpAtomicCompareExchangeWeak:
	case OpAtomicIIncrement:
	case OpAtomicIDecrement:
	case OpAtomicIAdd:
	case OpAtomicISub:
	case OpAtomicSMin:
	case OpAtomicUMin:
	case OpAtomicSMax:
	case OpAtomicUMax:
	case OpAtomicAnd:
	case OpAtomicOr:
	case OpAtomicXor:
		uses_atomics = true;
		check_resource_write(args[2]);
		break;

	case OpAtomicLoad:
		uses_atomics = true;
		break;

	case OpGroupNonUniformInverseBallot:
		needs_subgroup_invocation_id = true;
		break;

	case OpGroupNonUniformBallotBitCount:
		if (args[3] != GroupOperationReduce)
			needs_subgroup_invocation_id = true;
		break;

	default:
		break;
	}

	// If it has one, keep track of the instruction's result type, mapped by ID
	uint32_t result_type, result_id;
	if (compiler.instruction_to_result_type(result_type, result_id, opcode, args, length))
		result_types[result_id] = result_type;

	return true;
}

// If the variable is a Uniform or StorageBuffer, mark that a resource has been written to.
void CompilerMSL::OpCodePreprocessor::check_resource_write(uint32_t var_id)
{
	auto *p_var = compiler.maybe_get_backing_variable(var_id);
	StorageClass sc = p_var ? p_var->storage : StorageClassMax;
	if (sc == StorageClassUniform || sc == StorageClassStorageBuffer)
		uses_resource_write = true;
}

// Returns an enumeration of a SPIR-V function that needs to be output for certain Op codes.
CompilerMSL::SPVFuncImpl CompilerMSL::OpCodePreprocessor::get_spv_func_impl(Op opcode, const uint32_t *args)
{
	switch (opcode)
	{
	case OpFMod:
		return SPVFuncImplMod;

	case OpFunctionCall:
	{
		auto &return_type = compiler.get<SPIRType>(args[0]);
		if (return_type.array.size() > 1)
		{
			if (return_type.array.size() > SPVFuncImplArrayCopyMultidimMax)
				SPIRV_CROSS_THROW("Cannot support this many dimensions for arrays of arrays.");
			return static_cast<SPVFuncImpl>(SPVFuncImplArrayCopyMultidimBase + return_type.array.size());
		}
		else if (return_type.array.size() > 0)
			return SPVFuncImplArrayCopy;

		break;
	}

	case OpStore:
	{
		// Get the result type of the RHS. Since this is run as a pre-processing stage,
		// we must extract the result type directly from the Instruction, rather than the ID.
		uint32_t id_lhs = args[0];
		uint32_t id_rhs = args[1];

		const SPIRType *type = nullptr;
		if (compiler.ir.ids[id_rhs].get_type() != TypeNone)
		{
			// Could be a constant, or similar.
			type = &compiler.expression_type(id_rhs);
		}
		else
		{
			// Or ... an expression.
			uint32_t tid = result_types[id_rhs];
			if (tid)
				type = &compiler.get<SPIRType>(tid);
		}

		auto *var = compiler.maybe_get<SPIRVariable>(id_lhs);

		// Are we simply assigning to a statically assigned variable which takes a constant?
		// Don't bother emitting this function.
		bool static_expression_lhs =
		    var && var->storage == StorageClassFunction && var->statically_assigned && var->remapped_variable;
		if (type && compiler.is_array(*type) && !static_expression_lhs)
		{
			if (type->array.size() > 1)
			{
				if (type->array.size() > SPVFuncImplArrayCopyMultidimMax)
					SPIRV_CROSS_THROW("Cannot support this many dimensions for arrays of arrays.");
				return static_cast<SPVFuncImpl>(SPVFuncImplArrayCopyMultidimBase + type->array.size());
			}
			else
				return SPVFuncImplArrayCopy;
		}

		break;
	}

	case OpImageFetch:
	case OpImageRead:
	case OpImageWrite:
	{
		// Retrieve the image type, and if it's a Buffer, emit a texel coordinate function
		uint32_t tid = result_types[args[opcode == OpImageWrite ? 0 : 2]];
		if (tid && compiler.get<SPIRType>(tid).image.dim == DimBuffer && !compiler.msl_options.texture_buffer_native)
			return SPVFuncImplTexelBufferCoords;

		if (opcode == OpImageFetch && compiler.msl_options.swizzle_texture_samples)
			return SPVFuncImplTextureSwizzle;

		break;
	}

	case OpImageSampleExplicitLod:
	case OpImageSampleProjExplicitLod:
	case OpImageSampleDrefExplicitLod:
	case OpImageSampleProjDrefExplicitLod:
	case OpImageSampleImplicitLod:
	case OpImageSampleProjImplicitLod:
	case OpImageSampleDrefImplicitLod:
	case OpImageSampleProjDrefImplicitLod:
	case OpImageGather:
	case OpImageDrefGather:
		if (compiler.msl_options.swizzle_texture_samples)
			return SPVFuncImplTextureSwizzle;
		break;

	case OpCompositeConstruct:
	{
		auto &type = compiler.get<SPIRType>(args[0]);
		if (type.array.size() > 1) // We need to use copies to build the composite.
			return static_cast<SPVFuncImpl>(SPVFuncImplArrayCopyMultidimBase + type.array.size() - 1);
		break;
	}

	case OpExtInst:
	{
		uint32_t extension_set = args[2];
		if (compiler.get<SPIRExtension>(extension_set).ext == SPIRExtension::GLSL)
		{
			GLSLstd450 op_450 = static_cast<GLSLstd450>(args[3]);
			switch (op_450)
			{
			case GLSLstd450Radians:
				return SPVFuncImplRadians;
			case GLSLstd450Degrees:
				return SPVFuncImplDegrees;
			case GLSLstd450FindILsb:
				return SPVFuncImplFindILsb;
			case GLSLstd450FindSMsb:
				return SPVFuncImplFindSMsb;
			case GLSLstd450FindUMsb:
				return SPVFuncImplFindUMsb;
			case GLSLstd450SSign:
				return SPVFuncImplSSign;
			case GLSLstd450MatrixInverse:
			{
				auto &mat_type = compiler.get<SPIRType>(args[0]);
				switch (mat_type.columns)
				{
				case 2:
					return SPVFuncImplInverse2x2;
				case 3:
					return SPVFuncImplInverse3x3;
				case 4:
					return SPVFuncImplInverse4x4;
				default:
					break;
				}
				break;
			}
			default:
				break;
			}
		}
		break;
	}

	case OpGroupNonUniformBallot:
		return SPVFuncImplSubgroupBallot;

	case OpGroupNonUniformInverseBallot:
	case OpGroupNonUniformBallotBitExtract:
		return SPVFuncImplSubgroupBallotBitExtract;

	case OpGroupNonUniformBallotFindLSB:
		return SPVFuncImplSubgroupBallotFindLSB;

	case OpGroupNonUniformBallotFindMSB:
		return SPVFuncImplSubgroupBallotFindMSB;

	case OpGroupNonUniformBallotBitCount:
		return SPVFuncImplSubgroupBallotBitCount;

	case OpGroupNonUniformAllEqual:
		return SPVFuncImplSubgroupAllEqual;

	default:
		break;
	}
	return SPVFuncImplNone;
}

// Sort both type and meta member content based on builtin status (put builtins at end),
// then by the required sorting aspect.
void CompilerMSL::MemberSorter::sort()
{
	// Create a temporary array of consecutive member indices and sort it based on how
	// the members should be reordered, based on builtin and sorting aspect meta info.
	size_t mbr_cnt = type.member_types.size();
	SmallVector<uint32_t> mbr_idxs(mbr_cnt);
	iota(mbr_idxs.begin(), mbr_idxs.end(), 0); // Fill with consecutive indices
	std::sort(mbr_idxs.begin(), mbr_idxs.end(), *this); // Sort member indices based on sorting aspect

	// Move type and meta member info to the order defined by the sorted member indices.
	// This is done by creating temporary copies of both member types and meta, and then
	// copying back to the original content at the sorted indices.
	auto mbr_types_cpy = type.member_types;
	auto mbr_meta_cpy = meta.members;
	for (uint32_t mbr_idx = 0; mbr_idx < mbr_cnt; mbr_idx++)
	{
		type.member_types[mbr_idx] = mbr_types_cpy[mbr_idxs[mbr_idx]];
		meta.members[mbr_idx] = mbr_meta_cpy[mbr_idxs[mbr_idx]];
	}
}

// Sort first by builtin status (put builtins at end), then by the sorting aspect.
bool CompilerMSL::MemberSorter::operator()(uint32_t mbr_idx1, uint32_t mbr_idx2)
{
	auto &mbr_meta1 = meta.members[mbr_idx1];
	auto &mbr_meta2 = meta.members[mbr_idx2];
	if (mbr_meta1.builtin != mbr_meta2.builtin)
		return mbr_meta2.builtin;
	else
		switch (sort_aspect)
		{
		case Location:
			return mbr_meta1.location < mbr_meta2.location;
		case LocationReverse:
			return mbr_meta1.location > mbr_meta2.location;
		case Offset:
			return mbr_meta1.offset < mbr_meta2.offset;
		case OffsetThenLocationReverse:
			return (mbr_meta1.offset < mbr_meta2.offset) ||
			       ((mbr_meta1.offset == mbr_meta2.offset) && (mbr_meta1.location > mbr_meta2.location));
		case Alphabetical:
			return mbr_meta1.alias < mbr_meta2.alias;
		default:
			return false;
		}
}

CompilerMSL::MemberSorter::MemberSorter(SPIRType &t, Meta &m, SortAspect sa)
    : type(t)
    , meta(m)
    , sort_aspect(sa)
{
	// Ensure enough meta info is available
	meta.members.resize(max(type.member_types.size(), meta.members.size()));
}

void CompilerMSL::remap_constexpr_sampler(uint32_t id, const MSLConstexprSampler &sampler)
{
	auto &type = get<SPIRType>(get<SPIRVariable>(id).basetype);
	if (type.basetype != SPIRType::SampledImage && type.basetype != SPIRType::Sampler)
		SPIRV_CROSS_THROW("Can only remap SampledImage and Sampler type.");
	if (!type.array.empty())
		SPIRV_CROSS_THROW("Can not remap array of samplers.");
	constexpr_samplers[id] = sampler;
}

void CompilerMSL::bitcast_from_builtin_load(uint32_t source_id, std::string &expr, const SPIRType &expr_type)
{
	auto *var = maybe_get_backing_variable(source_id);
	if (var)
		source_id = var->self;

	// Only interested in standalone builtin variables.
	if (!has_decoration(source_id, DecorationBuiltIn))
		return;

	auto builtin = static_cast<BuiltIn>(get_decoration(source_id, DecorationBuiltIn));
	auto expected_type = expr_type.basetype;
	switch (builtin)
	{
	case BuiltInGlobalInvocationId:
	case BuiltInLocalInvocationId:
	case BuiltInWorkgroupId:
	case BuiltInLocalInvocationIndex:
	case BuiltInWorkgroupSize:
	case BuiltInNumWorkgroups:
	case BuiltInLayer:
	case BuiltInViewportIndex:
		expected_type = SPIRType::UInt;
		break;

	case BuiltInTessLevelInner:
	case BuiltInTessLevelOuter:
		if (get_execution_model() == ExecutionModelTessellationControl)
			expected_type = SPIRType::Half;
		break;

	default:
		break;
	}

	if (expected_type != expr_type.basetype)
		expr = bitcast_expression(expr_type, expected_type, expr);

	if (builtin == BuiltInTessCoord && get_entry_point().flags.get(ExecutionModeQuads) && expr_type.vecsize == 3)
	{
		// In SPIR-V, this is always a vec3, even for quads. In Metal, though, it's a float2 for quads.
		// The code is expecting a float3, so we need to widen this.
		expr = join("float3(", expr, ", 0)");
	}
}

void CompilerMSL::bitcast_to_builtin_store(uint32_t target_id, std::string &expr, const SPIRType &expr_type)
{
	auto *var = maybe_get_backing_variable(target_id);
	if (var)
		target_id = var->self;

	// Only interested in standalone builtin variables.
	if (!has_decoration(target_id, DecorationBuiltIn))
		return;

	auto builtin = static_cast<BuiltIn>(get_decoration(target_id, DecorationBuiltIn));
	auto expected_type = expr_type.basetype;
	switch (builtin)
	{
	case BuiltInLayer:
	case BuiltInViewportIndex:
		expected_type = SPIRType::UInt;
		break;

	case BuiltInTessLevelInner:
	case BuiltInTessLevelOuter:
		expected_type = SPIRType::Half;
		break;

	default:
		break;
	}

	if (expected_type != expr_type.basetype)
	{
		if (expected_type == SPIRType::Half && expr_type.basetype == SPIRType::Float)
		{
			// These are of different widths, so we cannot do a straight bitcast.
			expr = join("half(", expr, ")");
		}
		else
		{
			auto type = expr_type;
			type.basetype = expected_type;
			expr = bitcast_expression(type, expr_type.basetype, expr);
		}
	}
}

std::string CompilerMSL::to_initializer_expression(const SPIRVariable &var)
{
	// We risk getting an array initializer here with MSL. If we have an array.
	// FIXME: We cannot handle non-constant arrays being initialized.
	// We will need to inject spvArrayCopy here somehow ...
	auto &type = get<SPIRType>(var.basetype);
	if (ir.ids[var.initializer].get_type() == TypeConstant &&
	    (!type.array.empty() || type.basetype == SPIRType::Struct))
		return constant_expression(get<SPIRConstant>(var.initializer));
	else
		return CompilerGLSL::to_initializer_expression(var);
}

bool CompilerMSL::descriptor_set_is_argument_buffer(uint32_t desc_set) const
{
	if (!msl_options.argument_buffers)
		return false;
	if (desc_set >= kMaxArgumentBuffers)
		return false;

	return (argument_buffer_discrete_mask & (1u << desc_set)) == 0;
}

void CompilerMSL::analyze_argument_buffers()
{
	// Gather all used resources and sort them out into argument buffers.
	// Each argument buffer corresponds to a descriptor set in SPIR-V.
	// The [[id(N)]] values used correspond to the resource mapping we have for MSL.
	// Otherwise, the binding number is used, but this is generally not safe some types like
	// combined image samplers and arrays of resources. Metal needs different indices here,
	// while SPIR-V can have one descriptor set binding. To use argument buffers in practice,
	// you will need to use the remapping from the API.
	for (auto &id : argument_buffer_ids)
		id = 0;

	// Output resources, sorted by resource index & type.
	struct Resource
	{
		SPIRVariable *var;
		string name;
		SPIRType::BaseType basetype;
		uint32_t index;
	};
	SmallVector<Resource> resources_in_set[kMaxArgumentBuffers];

	bool set_needs_swizzle_buffer[kMaxArgumentBuffers] = {};

	ir.for_each_typed_id<SPIRVariable>([&](uint32_t self, SPIRVariable &var) {
		if ((var.storage == StorageClassUniform || var.storage == StorageClassUniformConstant ||
		     var.storage == StorageClassStorageBuffer) &&
		    !is_hidden_variable(var))
		{
			uint32_t desc_set = get_decoration(self, DecorationDescriptorSet);
			// Ignore if it's part of a push descriptor set.
			if (!descriptor_set_is_argument_buffer(desc_set))
				return;

			uint32_t var_id = var.self;
			auto &type = get_variable_data_type(var);

			if (desc_set >= kMaxArgumentBuffers)
				SPIRV_CROSS_THROW("Descriptor set index is out of range.");

			if (type.basetype == SPIRType::SampledImage)
			{
				add_resource_name(var_id);

				uint32_t image_resource_index = get_metal_resource_index(var, SPIRType::Image);
				uint32_t sampler_resource_index = get_metal_resource_index(var, SPIRType::Sampler);

				// Avoid trivial conflicts where we didn't remap.
				// This will let us at least compile test cases without having to instrument remaps.
				if (sampler_resource_index == image_resource_index)
					sampler_resource_index += type.array.empty() ? 1 : to_array_size_literal(type);

				resources_in_set[desc_set].push_back({ &var, to_name(var_id), SPIRType::Image, image_resource_index });

				if (type.image.dim != DimBuffer && constexpr_samplers.count(var_id) == 0)
				{
					resources_in_set[desc_set].push_back(
					    { &var, to_sampler_expression(var_id), SPIRType::Sampler, sampler_resource_index });
				}
			}
			else if (constexpr_samplers.count(var_id) == 0)
			{
				// constexpr samplers are not declared as resources.
				add_resource_name(var_id);
				resources_in_set[desc_set].push_back(
				    { &var, to_name(var_id), type.basetype, get_metal_resource_index(var, type.basetype) });
			}

			// Check if this descriptor set needs a swizzle buffer.
			if (needs_swizzle_buffer_def && is_sampled_image_type(type))
				set_needs_swizzle_buffer[desc_set] = true;
		}
	});

	if (needs_swizzle_buffer_def)
	{
		uint32_t swizzle_buffer_type_id = 0;

		// We might have to add a swizzle buffer resource to the set.
		for (uint32_t desc_set = 0; desc_set < kMaxArgumentBuffers; desc_set++)
		{
			if (!set_needs_swizzle_buffer[desc_set])
				continue;

			if (swizzle_buffer_type_id == 0)
			{
				uint32_t offset = ir.increase_bound_by(2);
				uint32_t type_id = offset;
				swizzle_buffer_type_id = offset + 1;

				// Create a buffer to hold extra data, including the swizzle constants.
				SPIRType uint_type;
				uint_type.basetype = SPIRType::UInt;
				uint_type.width = 32;
				set<SPIRType>(type_id, uint_type);

				SPIRType uint_type_pointer = uint_type;
				uint_type_pointer.pointer = true;
				uint_type_pointer.pointer_depth = 1;
				uint_type_pointer.parent_type = type_id;
				uint_type_pointer.storage = StorageClassUniform;
				set<SPIRType>(swizzle_buffer_type_id, uint_type_pointer);
				set_decoration(swizzle_buffer_type_id, DecorationArrayStride, 4);
			}

			uint32_t var_id = ir.increase_bound_by(1);
			auto &var = set<SPIRVariable>(var_id, swizzle_buffer_type_id, StorageClassUniformConstant);
			set_name(var_id, "spvSwizzleConstants");
			set_decoration(var_id, DecorationDescriptorSet, desc_set);
			set_decoration(var_id, DecorationBinding, kSwizzleBufferBinding);
			resources_in_set[desc_set].push_back(
			    { &var, to_name(var_id), SPIRType::UInt, get_metal_resource_index(var, SPIRType::UInt) });
		}
	}

	for (uint32_t desc_set = 0; desc_set < kMaxArgumentBuffers; desc_set++)
	{
		auto &resources = resources_in_set[desc_set];
		if (resources.empty())
			continue;

		assert(descriptor_set_is_argument_buffer(desc_set));

		uint32_t next_id = ir.increase_bound_by(3);
		uint32_t type_id = next_id + 1;
		uint32_t ptr_type_id = next_id + 2;
		argument_buffer_ids[desc_set] = next_id;

		auto &buffer_type = set<SPIRType>(type_id);
		buffer_type.storage = StorageClassUniform;
		buffer_type.basetype = SPIRType::Struct;
		set_name(type_id, join("spvDescriptorSetBuffer", desc_set));

		auto &ptr_type = set<SPIRType>(ptr_type_id);
		ptr_type = buffer_type;
		ptr_type.pointer = true;
		ptr_type.pointer_depth = 1;
		ptr_type.parent_type = type_id;

		uint32_t buffer_variable_id = next_id;
		set<SPIRVariable>(buffer_variable_id, ptr_type_id, StorageClassUniform);
		set_name(buffer_variable_id, join("spvDescriptorSet", desc_set));

		// Ids must be emitted in ID order.
		sort(begin(resources), end(resources), [&](const Resource &lhs, const Resource &rhs) -> bool {
			return tie(lhs.index, lhs.basetype) < tie(rhs.index, rhs.basetype);
		});

		uint32_t member_index = 0;
		for (auto &resource : resources)
		{
			auto &var = *resource.var;
			auto &type = get_variable_data_type(var);
			string mbr_name = ensure_valid_name(resource.name, "m");
			set_member_name(buffer_type.self, member_index, mbr_name);

			if (resource.basetype == SPIRType::Sampler && type.basetype != SPIRType::Sampler)
			{
				// Have to synthesize a sampler type here.

				bool type_is_array = !type.array.empty();
				uint32_t sampler_type_id = ir.increase_bound_by(type_is_array ? 2 : 1);
				auto &new_sampler_type = set<SPIRType>(sampler_type_id);
				new_sampler_type.basetype = SPIRType::Sampler;
				new_sampler_type.storage = StorageClassUniformConstant;

				if (type_is_array)
				{
					uint32_t sampler_type_array_id = sampler_type_id + 1;
					auto &sampler_type_array = set<SPIRType>(sampler_type_array_id);
					sampler_type_array = new_sampler_type;
					sampler_type_array.array = type.array;
					sampler_type_array.array_size_literal = type.array_size_literal;
					sampler_type_array.parent_type = sampler_type_id;
					buffer_type.member_types.push_back(sampler_type_array_id);
				}
				else
					buffer_type.member_types.push_back(sampler_type_id);
			}
			else
			{
				if (resource.basetype == SPIRType::Image || resource.basetype == SPIRType::Sampler ||
				    resource.basetype == SPIRType::SampledImage)
				{
					// Drop pointer information when we emit the resources into a struct.
					buffer_type.member_types.push_back(get_variable_data_type_id(var));
					set_qualified_name(var.self, join(to_name(buffer_variable_id), ".", mbr_name));
				}
				else
				{
					// Resources will be declared as pointers not references, so automatically dereference as appropriate.
					buffer_type.member_types.push_back(var.basetype);
					if (type.array.empty())
						set_qualified_name(var.self, join("(*", to_name(buffer_variable_id), ".", mbr_name, ")"));
					else
						set_qualified_name(var.self, join(to_name(buffer_variable_id), ".", mbr_name));
				}
			}

			set_extended_member_decoration(buffer_type.self, member_index, SPIRVCrossDecorationArgumentBufferID,
			                               resource.index);
			set_extended_member_decoration(buffer_type.self, member_index, SPIRVCrossDecorationInterfaceOrigID,
			                               var.self);
			member_index++;
		}
	}
}
