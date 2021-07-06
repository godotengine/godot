/*
 * Copyright 2016-2020 The Brenwill Workshop Ltd.
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
static const char *force_inline = "static inline __attribute__((always_inline))";

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
	StageSetBinding tuple = { binding.stage, binding.desc_set, binding.binding };
	resource_bindings[tuple] = { binding, false };
}

void CompilerMSL::add_dynamic_buffer(uint32_t desc_set, uint32_t binding, uint32_t index)
{
	SetBindingPair pair = { desc_set, binding };
	buffers_requiring_dynamic_offset[pair] = { index, 0 };
}

void CompilerMSL::add_inline_uniform_block(uint32_t desc_set, uint32_t binding)
{
	SetBindingPair pair = { desc_set, binding };
	inline_uniform_blocks.insert(pair);
}

void CompilerMSL::add_discrete_descriptor_set(uint32_t desc_set)
{
	if (desc_set < kMaxArgumentBuffers)
		argument_buffer_discrete_mask |= 1u << desc_set;
}

void CompilerMSL::set_argument_buffer_device_address_space(uint32_t desc_set, bool device_storage)
{
	if (desc_set < kMaxArgumentBuffers)
	{
		if (device_storage)
			argument_buffer_device_storage_mask |= 1u << desc_set;
		else
			argument_buffer_device_storage_mask &= ~(1u << desc_set);
	}
}

bool CompilerMSL::is_msl_vertex_attribute_used(uint32_t location)
{
	return vtx_attrs_in_use.count(location) != 0;
}

bool CompilerMSL::is_msl_resource_binding_used(ExecutionModel model, uint32_t desc_set, uint32_t binding) const
{
	StageSetBinding tuple = { model, desc_set, binding };
	auto itr = resource_bindings.find(tuple);
	return itr != end(resource_bindings) && itr->second.second;
}

uint32_t CompilerMSL::get_automatic_msl_resource_binding(uint32_t id) const
{
	return get_extended_decoration(id, SPIRVCrossDecorationResourceIndexPrimary);
}

uint32_t CompilerMSL::get_automatic_msl_resource_binding_secondary(uint32_t id) const
{
	return get_extended_decoration(id, SPIRVCrossDecorationResourceIndexSecondary);
}

uint32_t CompilerMSL::get_automatic_msl_resource_binding_tertiary(uint32_t id) const
{
	return get_extended_decoration(id, SPIRVCrossDecorationResourceIndexTertiary);
}

uint32_t CompilerMSL::get_automatic_msl_resource_binding_quaternary(uint32_t id) const
{
	return get_extended_decoration(id, SPIRVCrossDecorationResourceIndexQuaternary);
}

void CompilerMSL::set_fragment_output_components(uint32_t location, uint32_t components)
{
	fragment_output_components[location] = components;
}

bool CompilerMSL::builtin_translates_to_nonarray(spv::BuiltIn builtin) const
{
	return (builtin == BuiltInSampleMask);
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
	bool need_multiview = get_execution_model() == ExecutionModelVertex && !msl_options.view_index_from_device_index &&
	                      (msl_options.multiview || active_input_builtins.get(BuiltInViewIndex));
	bool need_dispatch_base =
	    msl_options.dispatch_base && get_execution_model() == ExecutionModelGLCompute &&
	    (active_input_builtins.get(BuiltInWorkgroupId) || active_input_builtins.get(BuiltInGlobalInvocationId));
	if (need_subpass_input || need_sample_pos || need_subgroup_mask || need_vertex_params || need_tesc_params ||
	    need_multiview || need_dispatch_base || needs_subgroup_invocation_id)
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
		bool has_view_idx = false;
		uint32_t workgroup_id_type = 0;

		ir.for_each_typed_id<SPIRVariable>([&](uint32_t, SPIRVariable &var) {
			if (var.storage != StorageClassInput || !ir.meta[var.self].decoration.builtin)
				return;

			// Use Metal's native frame-buffer fetch API for subpass inputs.
			BuiltIn builtin = ir.meta[var.self].decoration.builtin_type;
			if (need_subpass_input && (!msl_options.is_ios() || !msl_options.ios_use_framebuffer_fetch_subpasses) &&
			    builtin == BuiltInFragCoord)
			{
				mark_implicit_builtin(StorageClassInput, BuiltInFragCoord, var.self);
				builtin_frag_coord_id = var.self;
				has_frag_coord = true;
			}

			if (need_sample_pos && builtin == BuiltInSampleId)
			{
				builtin_sample_id_id = var.self;
				mark_implicit_builtin(StorageClassInput, BuiltInSampleId, var.self);
				has_sample_id = true;
			}

			if (need_vertex_params)
			{
				switch (builtin)
				{
				case BuiltInVertexIndex:
					builtin_vertex_idx_id = var.self;
					mark_implicit_builtin(StorageClassInput, BuiltInVertexIndex, var.self);
					has_vertex_idx = true;
					break;
				case BuiltInBaseVertex:
					builtin_base_vertex_id = var.self;
					mark_implicit_builtin(StorageClassInput, BuiltInBaseVertex, var.self);
					has_base_vertex = true;
					break;
				case BuiltInInstanceIndex:
					builtin_instance_idx_id = var.self;
					mark_implicit_builtin(StorageClassInput, BuiltInInstanceIndex, var.self);
					has_instance_idx = true;
					break;
				case BuiltInBaseInstance:
					builtin_base_instance_id = var.self;
					mark_implicit_builtin(StorageClassInput, BuiltInBaseInstance, var.self);
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
					mark_implicit_builtin(StorageClassInput, BuiltInInvocationId, var.self);
					has_invocation_id = true;
					break;
				case BuiltInPrimitiveId:
					builtin_primitive_id_id = var.self;
					mark_implicit_builtin(StorageClassInput, BuiltInPrimitiveId, var.self);
					has_primitive_id = true;
					break;
				default:
					break;
				}
			}

			if ((need_subgroup_mask || needs_subgroup_invocation_id) && builtin == BuiltInSubgroupLocalInvocationId)
			{
				builtin_subgroup_invocation_id_id = var.self;
				mark_implicit_builtin(StorageClassInput, BuiltInSubgroupLocalInvocationId, var.self);
				has_subgroup_invocation_id = true;
			}

			if (need_subgroup_ge_mask && builtin == BuiltInSubgroupSize)
			{
				builtin_subgroup_size_id = var.self;
				mark_implicit_builtin(StorageClassInput, BuiltInSubgroupSize, var.self);
				has_subgroup_size = true;
			}

			if (need_multiview)
			{
				switch (builtin)
				{
				case BuiltInInstanceIndex:
					// The view index here is derived from the instance index.
					builtin_instance_idx_id = var.self;
					mark_implicit_builtin(StorageClassInput, BuiltInInstanceIndex, var.self);
					has_instance_idx = true;
					break;
				case BuiltInViewIndex:
					builtin_view_idx_id = var.self;
					mark_implicit_builtin(StorageClassInput, BuiltInViewIndex, var.self);
					has_view_idx = true;
					break;
				default:
					break;
				}
			}

			// The base workgroup needs to have the same type and vector size
			// as the workgroup or invocation ID, so keep track of the type that
			// was used.
			if (need_dispatch_base && workgroup_id_type == 0 &&
			    (builtin == BuiltInWorkgroupId || builtin == BuiltInGlobalInvocationId))
				workgroup_id_type = var.basetype;
		});

		// Use Metal's native frame-buffer fetch API for subpass inputs.
		if (!has_frag_coord && (!msl_options.is_ios() || !msl_options.ios_use_framebuffer_fetch_subpasses) &&
		    need_subpass_input)
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
			mark_implicit_builtin(StorageClassInput, BuiltInFragCoord, var_id);
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
			mark_implicit_builtin(StorageClassInput, BuiltInSampleId, var_id);
		}

		if ((need_vertex_params && (!has_vertex_idx || !has_base_vertex || !has_instance_idx || !has_base_instance)) ||
		    (need_multiview && (!has_instance_idx || !has_view_idx)))
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

			if (need_vertex_params && !has_vertex_idx)
			{
				uint32_t var_id = ir.increase_bound_by(1);

				// Create gl_VertexIndex.
				set<SPIRVariable>(var_id, type_ptr_id, StorageClassInput);
				set_decoration(var_id, DecorationBuiltIn, BuiltInVertexIndex);
				builtin_vertex_idx_id = var_id;
				mark_implicit_builtin(StorageClassInput, BuiltInVertexIndex, var_id);
			}

			if (need_vertex_params && !has_base_vertex)
			{
				uint32_t var_id = ir.increase_bound_by(1);

				// Create gl_BaseVertex.
				set<SPIRVariable>(var_id, type_ptr_id, StorageClassInput);
				set_decoration(var_id, DecorationBuiltIn, BuiltInBaseVertex);
				builtin_base_vertex_id = var_id;
				mark_implicit_builtin(StorageClassInput, BuiltInBaseVertex, var_id);
			}

			if (!has_instance_idx) // Needed by both multiview and tessellation
			{
				uint32_t var_id = ir.increase_bound_by(1);

				// Create gl_InstanceIndex.
				set<SPIRVariable>(var_id, type_ptr_id, StorageClassInput);
				set_decoration(var_id, DecorationBuiltIn, BuiltInInstanceIndex);
				builtin_instance_idx_id = var_id;
				mark_implicit_builtin(StorageClassInput, BuiltInInstanceIndex, var_id);
			}

			if (need_vertex_params && !has_base_instance)
			{
				uint32_t var_id = ir.increase_bound_by(1);

				// Create gl_BaseInstance.
				set<SPIRVariable>(var_id, type_ptr_id, StorageClassInput);
				set_decoration(var_id, DecorationBuiltIn, BuiltInBaseInstance);
				builtin_base_instance_id = var_id;
				mark_implicit_builtin(StorageClassInput, BuiltInBaseInstance, var_id);
			}

			if (need_multiview)
			{
				// Multiview shaders are not allowed to write to gl_Layer, ostensibly because
				// it is implicitly written from gl_ViewIndex, but we have to do that explicitly.
				// Note that we can't just abuse gl_ViewIndex for this purpose: it's an input, but
				// gl_Layer is an output in vertex-pipeline shaders.
				uint32_t type_ptr_out_id = ir.increase_bound_by(2);
				SPIRType uint_type_ptr_out;
				uint_type_ptr_out = uint_type;
				uint_type_ptr_out.pointer = true;
				uint_type_ptr_out.parent_type = type_id;
				uint_type_ptr_out.storage = StorageClassOutput;
				auto &ptr_out_type = set<SPIRType>(type_ptr_out_id, uint_type_ptr_out);
				ptr_out_type.self = type_id;
				uint32_t var_id = type_ptr_out_id + 1;
				set<SPIRVariable>(var_id, type_ptr_out_id, StorageClassOutput);
				set_decoration(var_id, DecorationBuiltIn, BuiltInLayer);
				builtin_layer_id = var_id;
				mark_implicit_builtin(StorageClassOutput, BuiltInLayer, var_id);
			}

			if (need_multiview && !has_view_idx)
			{
				uint32_t var_id = ir.increase_bound_by(1);

				// Create gl_ViewIndex.
				set<SPIRVariable>(var_id, type_ptr_id, StorageClassInput);
				set_decoration(var_id, DecorationBuiltIn, BuiltInViewIndex);
				builtin_view_idx_id = var_id;
				mark_implicit_builtin(StorageClassInput, BuiltInViewIndex, var_id);
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
				mark_implicit_builtin(StorageClassInput, BuiltInInvocationId, var_id);
			}

			if (!has_primitive_id)
			{
				uint32_t var_id = ir.increase_bound_by(1);

				// Create gl_PrimitiveID.
				set<SPIRVariable>(var_id, type_ptr_id, StorageClassInput);
				set_decoration(var_id, DecorationBuiltIn, BuiltInPrimitiveId);
				builtin_primitive_id_id = var_id;
				mark_implicit_builtin(StorageClassInput, BuiltInPrimitiveId, var_id);
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
			mark_implicit_builtin(StorageClassInput, BuiltInSubgroupLocalInvocationId, var_id);
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
			mark_implicit_builtin(StorageClassInput, BuiltInSubgroupSize, var_id);
		}

		if (need_dispatch_base)
		{
			uint32_t var_id;
			if (msl_options.supports_msl_version(1, 2))
			{
				// If we have MSL 1.2, we can (ab)use the [[grid_origin]] builtin
				// to convey this information and save a buffer slot.
				uint32_t offset = ir.increase_bound_by(1);
				var_id = offset;

				set<SPIRVariable>(var_id, workgroup_id_type, StorageClassInput);
				set_extended_decoration(var_id, SPIRVCrossDecorationBuiltInDispatchBase);
				get_entry_point().interface_variables.push_back(var_id);
			}
			else
			{
				// Otherwise, we need to fall back to a good ol' fashioned buffer.
				uint32_t offset = ir.increase_bound_by(2);
				var_id = offset;
				uint32_t type_id = offset + 1;

				SPIRType var_type = get<SPIRType>(workgroup_id_type);
				var_type.storage = StorageClassUniform;
				set<SPIRType>(type_id, var_type);

				set<SPIRVariable>(var_id, type_id, StorageClassUniform);
				// This should never match anything.
				set_decoration(var_id, DecorationDescriptorSet, ~(5u));
				set_decoration(var_id, DecorationBinding, msl_options.indirect_params_buffer_index);
				set_extended_decoration(var_id, SPIRVCrossDecorationResourceIndexPrimary,
				                        msl_options.indirect_params_buffer_index);
			}
			set_name(var_id, "spvDispatchBase");
			builtin_dispatch_base_id = var_id;
		}
	}

	if (needs_swizzle_buffer_def)
	{
		uint32_t var_id = build_constant_uint_array_pointer();
		set_name(var_id, "spvSwizzleConstants");
		// This should never match anything.
		set_decoration(var_id, DecorationDescriptorSet, kSwizzleBufferBinding);
		set_decoration(var_id, DecorationBinding, msl_options.swizzle_buffer_index);
		set_extended_decoration(var_id, SPIRVCrossDecorationResourceIndexPrimary, msl_options.swizzle_buffer_index);
		swizzle_buffer_id = var_id;
	}

	if (!buffers_requiring_array_length.empty())
	{
		uint32_t var_id = build_constant_uint_array_pointer();
		set_name(var_id, "spvBufferSizeConstants");
		// This should never match anything.
		set_decoration(var_id, DecorationDescriptorSet, kBufferSizeBufferBinding);
		set_decoration(var_id, DecorationBinding, msl_options.buffer_size_buffer_index);
		set_extended_decoration(var_id, SPIRVCrossDecorationResourceIndexPrimary, msl_options.buffer_size_buffer_index);
		buffer_size_buffer_id = var_id;
	}

	if (needs_view_mask_buffer())
	{
		uint32_t var_id = build_constant_uint_array_pointer();
		set_name(var_id, "spvViewMask");
		// This should never match anything.
		set_decoration(var_id, DecorationDescriptorSet, ~(4u));
		set_decoration(var_id, DecorationBinding, msl_options.view_mask_buffer_index);
		set_extended_decoration(var_id, SPIRVCrossDecorationResourceIndexPrimary, msl_options.view_mask_buffer_index);
		view_mask_buffer_id = var_id;
	}

	if (!buffers_requiring_dynamic_offset.empty())
	{
		uint32_t var_id = build_constant_uint_array_pointer();
		set_name(var_id, "spvDynamicOffsets");
		// This should never match anything.
		set_decoration(var_id, DecorationDescriptorSet, ~(5u));
		set_decoration(var_id, DecorationBinding, msl_options.dynamic_offsets_buffer_index);
		set_extended_decoration(var_id, SPIRVCrossDecorationResourceIndexPrimary,
		                        msl_options.dynamic_offsets_buffer_index);
		dynamic_offsets_buffer_id = var_id;
	}
}

// Checks if the specified builtin variable (e.g. gl_InstanceIndex) is marked as active.
// If not, it marks it as active and forces a recompilation.
// This might be used when the optimization of inactive builtins was too optimistic (e.g. when "spvOut" is emitted).
void CompilerMSL::ensure_builtin(spv::StorageClass storage, spv::BuiltIn builtin)
{
	Bitset *active_builtins = nullptr;
	switch (storage)
	{
	case StorageClassInput:
		active_builtins = &active_input_builtins;
		break;

	case StorageClassOutput:
		active_builtins = &active_output_builtins;
		break;

	default:
		break;
	}

	// At this point, the specified builtin variable must have already been declared in the entry point.
	// If not, mark as active and force recompile.
	if (active_builtins != nullptr && !active_builtins->get(builtin))
	{
		active_builtins->set(builtin);
		force_recompile();
	}
}

void CompilerMSL::mark_implicit_builtin(StorageClass storage, BuiltIn builtin, uint32_t id)
{
	Bitset *active_builtins = nullptr;
	switch (storage)
	{
	case StorageClassInput:
		active_builtins = &active_input_builtins;
		break;

	case StorageClassOutput:
		active_builtins = &active_output_builtins;
		break;

	default:
		break;
	}

	assert(active_builtins != nullptr);
	active_builtins->set(builtin);

	auto &var = get_entry_point().interface_variables;
	if (find(begin(var), end(var), VariableID(id)) == end(var))
		var.push_back(id);
}

uint32_t CompilerMSL::build_constant_uint_array_pointer()
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
	return var_id;
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
	// Constant arrays of non-primitive types (i.e. matrices) won't link properly into Metal libraries
	declare_complex_constant_arrays();

	// Emit constexpr samplers here.
	for (auto &samp : constexpr_samplers_by_id)
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

		// If we would emit no arguments, then omit the parentheses entirely. Otherwise,
		// we'll wind up with a "most vexing parse" situation.
		if (args.empty())
			statement("constexpr sampler ",
			          type.basetype == SPIRType::SampledImage ? to_sampler_expression(samp.first) : to_name(samp.first),
			          ";");
		else
			statement("constexpr sampler ",
			          type.basetype == SPIRType::SampledImage ? to_sampler_expression(samp.first) : to_name(samp.first),
			          "(", merge(args), ");");
	}

	// Emit dynamic buffers here.
	for (auto &dynamic_buffer : buffers_requiring_dynamic_offset)
	{
		if (!dynamic_buffer.second.second)
		{
			// Could happen if no buffer was used at requested binding point.
			continue;
		}

		const auto &var = get<SPIRVariable>(dynamic_buffer.second.second);
		uint32_t var_id = var.self;
		const auto &type = get_variable_data_type(var);
		string name = to_name(var.self);
		uint32_t desc_set = get_decoration(var.self, DecorationDescriptorSet);
		uint32_t arg_id = argument_buffer_ids[desc_set];
		uint32_t base_index = dynamic_buffer.second.first;

		if (!type.array.empty())
		{
			// This is complicated, because we need to support arrays of arrays.
			// And it's even worse if the outermost dimension is a runtime array, because now
			// all this complicated goop has to go into the shader itself. (FIXME)
			if (!type.array[type.array.size() - 1])
				SPIRV_CROSS_THROW("Runtime arrays with dynamic offsets are not supported yet.");
			else
			{
				is_using_builtin_array = true;
				statement(get_argument_address_space(var), " ", type_to_glsl(type), "* ", to_restrict(var_id), name,
				          type_to_array_glsl(type), " =");

				uint32_t dim = uint32_t(type.array.size());
				uint32_t j = 0;
				for (SmallVector<uint32_t> indices(type.array.size());
				     indices[type.array.size() - 1] < to_array_size_literal(type); j++)
				{
					while (dim > 0)
					{
						begin_scope();
						--dim;
					}

					string arrays;
					for (uint32_t i = uint32_t(type.array.size()); i; --i)
						arrays += join("[", indices[i - 1], "]");
					statement("(", get_argument_address_space(var), " ", type_to_glsl(type), "* ",
					          to_restrict(var_id, false), ")((", get_argument_address_space(var), " char* ",
					          to_restrict(var_id, false), ")", to_name(arg_id), ".", ensure_valid_name(name, "m"),
					          arrays, " + ", to_name(dynamic_offsets_buffer_id), "[", base_index + j, "]),");

					while (++indices[dim] >= to_array_size_literal(type, dim) && dim < type.array.size() - 1)
					{
						end_scope(",");
						indices[dim++] = 0;
					}
				}
				end_scope_decl();
				statement_no_indent("");
				is_using_builtin_array = false;
			}
		}
		else
		{
			statement(get_argument_address_space(var), " auto& ", to_restrict(var_id), name, " = *(",
			          get_argument_address_space(var), " ", type_to_glsl(type), "* ", to_restrict(var_id, false), ")((",
			          get_argument_address_space(var), " char* ", to_restrict(var_id, false), ")", to_name(arg_id), ".",
			          ensure_valid_name(name, "m"), " + ", to_name(dynamic_offsets_buffer_id), "[", base_index, "]);");
		}
	}

	// Emit buffer arrays here.
	for (uint32_t array_id : buffer_arrays)
	{
		const auto &var = get<SPIRVariable>(array_id);
		const auto &type = get_variable_data_type(var);
		const auto &buffer_type = get_variable_element_type(var);
		string name = to_name(array_id);
		statement(get_argument_address_space(var), " ", type_to_glsl(buffer_type), "* ", to_restrict(array_id), name,
		          "[] =");
		begin_scope();
		for (uint32_t i = 0; i < to_array_size_literal(type); ++i)
			statement(name, "_", i, ",");
		end_scope_decl();
		statement_no_indent("");
	}
	// For some reason, without this, we end up emitting the arrays twice.
	buffer_arrays.clear();

	// Emit disabled fragment outputs.
	std::sort(disabled_frag_outputs.begin(), disabled_frag_outputs.end());
	for (uint32_t var_id : disabled_frag_outputs)
	{
		auto &var = get<SPIRVariable>(var_id);
		add_local_variable_name(var_id);
		statement(variable_decl(var), ";");
		var.deferred_declaration = false;
	}
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
	backend.uint16_t_literal_suffix = "";
	backend.basic_int_type = "int";
	backend.basic_uint_type = "uint";
	backend.basic_int8_type = "char";
	backend.basic_uint8_type = "uchar";
	backend.basic_int16_type = "short";
	backend.basic_uint16_type = "ushort";
	backend.discard_literal = "discard_fragment()";
	backend.demote_literal = "unsupported-demote";
	backend.boolean_mix_function = "select";
	backend.swizzle_is_function = false;
	backend.shared_is_implied = false;
	backend.use_initializer_list = true;
	backend.use_typed_initializer_list = true;
	backend.native_row_major_matrix = false;
	backend.unsized_array_supported = false;
	backend.can_declare_arrays_inline = false;
	backend.allow_truncated_access_chain = true;
	backend.comparison_image_samples_scalar = true;
	backend.native_pointers = true;
	backend.nonuniform_qualifier = "";
	backend.support_small_type_sampling_result = true;
	backend.supports_empty_struct = true;

	// Allow Metal to use the array<T> template unless we force it off.
	backend.can_return_array = !msl_options.force_native_arrays;
	backend.array_is_value_type = !msl_options.force_native_arrays;

	capture_output_to_buffer = msl_options.capture_output_to_buffer;
	is_rasterization_disabled = msl_options.disable_rasterization || capture_output_to_buffer;

	// Initialize array here rather than constructor, MSVC 2013 workaround.
	for (auto &id : next_metal_resource_ids)
		id = 0;

	fixup_type_alias();
	replace_illegal_names();

	build_function_control_flow_graphs_and_analyze();
	update_active_builtins();
	analyze_image_and_sampler_usage();
	analyze_sampled_image_usage();
	analyze_interlocked_resource_usage();
	preprocess_op_codes();
	build_implicit_builtins();

	fixup_image_load_store_access();

	set_enabled_interface_variables(get_active_interface_variables());
	if (msl_options.force_active_argument_buffer_resources)
		activate_argument_buffer_resources();

	if (swizzle_buffer_id)
		active_interface_variables.insert(swizzle_buffer_id);
	if (buffer_size_buffer_id)
		active_interface_variables.insert(buffer_size_buffer_id);
	if (view_mask_buffer_id)
		active_interface_variables.insert(view_mask_buffer_id);
	if (dynamic_offsets_buffer_id)
		active_interface_variables.insert(dynamic_offsets_buffer_id);
	if (builtin_layer_id)
		active_interface_variables.insert(builtin_layer_id);
	if (builtin_dispatch_base_id && !msl_options.supports_msl_version(1, 2))
		active_interface_variables.insert(builtin_dispatch_base_id);

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
		for (auto &id : next_metal_resource_ids)
			id = 0;

		// Move constructor for this type is broken on GCC 4.9 ...
		buffer.reset();

		emit_header();
		emit_custom_templates();
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
			case OpArrayLength:
			{
				uint32_t base_id = ops[2];
				if (global_var_ids.find(base_id) != global_var_ids.end())
					added_arg_ids.insert(base_id);

				// Use Metal's native frame-buffer fetch API for subpass inputs.
				auto &type = get<SPIRType>(ops[0]);
				if (type.basetype == SPIRType::Image && type.image.dim == DimSubpassData &&
				    (!msl_options.is_ios() || !msl_options.ios_use_framebuffer_fetch_subpasses))
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

			// Emulate texture2D atomic operations
			case OpImageTexelPointer:
			{
				// When using the pointer, we need to know which variable it is actually loaded from.
				uint32_t base_id = ops[2];
				auto *var = maybe_get_backing_variable(base_id);
				if (var && atomic_image_vars.count(var->self))
				{
					if (global_var_ids.find(base_id) != global_var_ids.end())
						added_arg_ids.insert(base_id);
				}
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
// are marked as packable with the SPIRVCrossDecorationBufferBlockRepacked decoration,
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
		set_extended_decoration(type.self, SPIRVCrossDecorationBufferBlockRepacked);

		// Recurse
		uint32_t mbr_cnt = uint32_t(type.member_types.size());
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
                                                        SPIRType &ib_type, SPIRVariable &var, InterfaceBlockMeta &meta)
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
	if (meta.strip_array && is_array(get<SPIRType>(type_id)))
		type_id = get<SPIRType>(type_id).parent_type;
	auto &type = get<SPIRType>(type_id);
	uint32_t target_components = 0;
	uint32_t type_components = type.vecsize;

	bool padded_output = false;
	bool padded_input = false;
	uint32_t start_component = 0;

	auto &entry_func = get<SPIRFunction>(ir.default_entry_point);

	// Deal with Component decorations.
	InterfaceBlockMeta::LocationMeta *location_meta = nullptr;
	if (has_decoration(var.self, DecorationLocation))
	{
		auto location_meta_itr = meta.location_meta.find(get_decoration(var.self, DecorationLocation));
		if (location_meta_itr != end(meta.location_meta))
			location_meta = &location_meta_itr->second;
	}

	bool pad_fragment_output = has_decoration(var.self, DecorationLocation) &&
	                           msl_options.pad_fragment_output_components &&
	                           get_entry_point().model == ExecutionModelFragment && storage == StorageClassOutput;

	// Check if we need to pad fragment output to match a certain number of components.
	if (location_meta)
	{
		start_component = get_decoration(var.self, DecorationComponent);
		uint32_t num_components = location_meta->num_components;
		if (pad_fragment_output)
		{
			uint32_t locn = get_decoration(var.self, DecorationLocation);
			num_components = std::max(num_components, get_target_components_for_fragment_location(locn));
		}

		if (location_meta->ib_index != ~0u)
		{
			// We have already declared the variable. Just emit an early-declared variable and fixup as needed.
			entry_func.add_local_variable(var.self);
			vars_needing_early_declaration.push_back(var.self);

			if (var.storage == StorageClassInput)
			{
				uint32_t ib_index = location_meta->ib_index;
				entry_func.fixup_hooks_in.push_back([=, &var]() {
					statement(to_name(var.self), " = ", ib_var_ref, ".", to_member_name(ib_type, ib_index),
					          vector_swizzle(type_components, start_component), ";");
				});
			}
			else
			{
				uint32_t ib_index = location_meta->ib_index;
				entry_func.fixup_hooks_out.push_back([=, &var]() {
					statement(ib_var_ref, ".", to_member_name(ib_type, ib_index),
					          vector_swizzle(type_components, start_component), " = ", to_name(var.self), ";");
				});
			}
			return;
		}
		else
		{
			location_meta->ib_index = uint32_t(ib_type.member_types.size());
			type_id = build_extended_vector_type(type_id, num_components);
			if (var.storage == StorageClassInput)
				padded_input = true;
			else
				padded_output = true;
		}
	}
	else if (pad_fragment_output)
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

	if (padded_output || padded_input)
	{
		entry_func.add_local_variable(var.self);
		vars_needing_early_declaration.push_back(var.self);

		if (padded_output)
		{
			entry_func.fixup_hooks_out.push_back([=, &var]() {
				statement(qual_var_name, vector_swizzle(type_components, start_component), " = ", to_name(var.self),
				          ";");
			});
		}
		else
		{
			entry_func.fixup_hooks_in.push_back([=, &var]() {
				statement(to_name(var.self), " = ", qual_var_name, vector_swizzle(type_components, start_component),
				          ";");
			});
		}
	}
	else if (!meta.strip_array)
		ir.meta[var.self].decoration.qualified_alias = qual_var_name;

	if (var.storage == StorageClassOutput && var.initializer != ID(0))
	{
		if (padded_output || padded_input)
		{
			entry_func.fixup_hooks_in.push_back(
			    [=, &var]() { statement(to_name(var.self), " = ", to_expression(var.initializer), ";"); });
		}
		else
		{
			entry_func.fixup_hooks_in.push_back(
			    [=, &var]() { statement(qual_var_name, " = ", to_expression(var.initializer), ";"); });
		}
	}

	// Copy the variable location from the original variable to the member
	if (get_decoration_bitset(var.self).get(DecorationLocation))
	{
		uint32_t locn = get_decoration(var.self, DecorationLocation);
		if (storage == StorageClassInput && (get_execution_model() == ExecutionModelVertex || is_tessellation_shader()))
		{
			type_id = ensure_correct_attribute_type(var.basetype, locn,
			                                        location_meta ? location_meta->num_components : type.vecsize);

			if (!location_meta)
				var.basetype = type_id;

			type_id = get_pointee_type_id(type_id);
			if (meta.strip_array && is_array(get<SPIRType>(type_id)))
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

	if (!location_meta)
	{
		if (get_decoration_bitset(var.self).get(DecorationComponent))
		{
			uint32_t component = get_decoration(var.self, DecorationComponent);
			set_member_decoration(ib_type.self, ib_mbr_idx, DecorationComponent, component);
		}
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

	// If we have location meta, there is no unique OrigID. We won't need it, since we flatten/unflatten
	// the variable to stack anyways here.
	if (!location_meta)
		set_extended_member_decoration(ib_type.self, ib_mbr_idx, SPIRVCrossDecorationInterfaceOrigID, var.self);
}

void CompilerMSL::add_composite_variable_to_interface_block(StorageClass storage, const string &ib_var_ref,
                                                            SPIRType &ib_type, SPIRVariable &var,
                                                            InterfaceBlockMeta &meta)
{
	auto &entry_func = get<SPIRFunction>(ir.default_entry_point);
	auto &var_type = meta.strip_array ? get_variable_element_type(var) : get_variable_data_type(var);
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

	bool flatten_from_ib_var = false;
	string flatten_from_ib_mbr_name;

	if (storage == StorageClassOutput && is_builtin && builtin == BuiltInClipDistance)
	{
		// Also declare [[clip_distance]] attribute here.
		uint32_t clip_array_mbr_idx = uint32_t(ib_type.member_types.size());
		ib_type.member_types.push_back(get_variable_data_type_id(var));
		set_member_decoration(ib_type.self, clip_array_mbr_idx, DecorationBuiltIn, BuiltInClipDistance);

		flatten_from_ib_mbr_name = builtin_to_glsl(BuiltInClipDistance, StorageClassOutput);
		set_member_name(ib_type.self, clip_array_mbr_idx, flatten_from_ib_mbr_name);

		// When we flatten, we flatten directly from the "out" struct,
		// not from a function variable.
		flatten_from_ib_var = true;

		if (!msl_options.enable_clip_distance_user_varying)
			return;
	}
	else if (!meta.strip_array)
	{
		// Only flatten/unflatten IO composites for non-tessellation cases where arrays are not stripped.
		entry_func.add_local_variable(var.self);
		// We need to declare the variable early and at entry-point scope.
		vars_needing_early_declaration.push_back(var.self);
	}

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
		else if (is_builtin && builtin == BuiltInClipDistance)
		{
			// Declare the ClipDistance as [[user(clipN)]].
			set_member_decoration(ib_type.self, ib_mbr_idx, DecorationBuiltIn, BuiltInClipDistance);
			set_member_decoration(ib_type.self, ib_mbr_idx, DecorationLocation, i);
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

		// Only flatten/unflatten IO composites for non-tessellation cases where arrays are not stripped.
		if (!meta.strip_array)
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
					else if (flatten_from_ib_var)
						statement(ib_var_ref, ".", mbr_name, " = ", ib_var_ref, ".", flatten_from_ib_mbr_name, "[", i,
						          "];");
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
                                                                   uint32_t mbr_idx, InterfaceBlockMeta &meta)
{
	auto &entry_func = get<SPIRFunction>(ir.default_entry_point);
	auto &var_type = meta.strip_array ? get_variable_element_type(var) : get_variable_data_type(var);

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

	bool flatten_from_ib_var = false;
	string flatten_from_ib_mbr_name;

	if (storage == StorageClassOutput && is_builtin && builtin == BuiltInClipDistance)
	{
		// Also declare [[clip_distance]] attribute here.
		uint32_t clip_array_mbr_idx = uint32_t(ib_type.member_types.size());
		ib_type.member_types.push_back(mbr_type_id);
		set_member_decoration(ib_type.self, clip_array_mbr_idx, DecorationBuiltIn, BuiltInClipDistance);

		flatten_from_ib_mbr_name = builtin_to_glsl(BuiltInClipDistance, StorageClassOutput);
		set_member_name(ib_type.self, clip_array_mbr_idx, flatten_from_ib_mbr_name);

		// When we flatten, we flatten directly from the "out" struct,
		// not from a function variable.
		flatten_from_ib_var = true;

		if (!msl_options.enable_clip_distance_user_varying)
			return;
	}

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
			uint32_t locn = get_accumulated_member_location(var, mbr_idx, meta.strip_array) + i;
			set_member_decoration(ib_type.self, ib_mbr_idx, DecorationLocation, locn);
			mark_location_as_used_by_shader(locn, storage);
		}
		else if (is_builtin && is_tessellation_shader() && vtx_attrs_by_builtin.count(builtin))
		{
			uint32_t locn = vtx_attrs_by_builtin[builtin].location + i;
			set_member_decoration(ib_type.self, ib_mbr_idx, DecorationLocation, locn);
			mark_location_as_used_by_shader(locn, storage);
		}
		else if (is_builtin && builtin == BuiltInClipDistance)
		{
			// Declare the ClipDistance as [[user(clipN)]].
			set_member_decoration(ib_type.self, ib_mbr_idx, DecorationBuiltIn, BuiltInClipDistance);
			set_member_decoration(ib_type.self, ib_mbr_idx, DecorationLocation, i);
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
		if (!meta.strip_array)
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
					if (flatten_from_ib_var)
					{
						statement(ib_var_ref, ".", mbr_name, " = ", ib_var_ref, ".", flatten_from_ib_mbr_name, "[", i,
						          "];");
					}
					else
					{
						statement(ib_var_ref, ".", mbr_name, " = ", to_name(var.self), ".",
						          to_member_name(var_type, mbr_idx), "[", i, "];");
					}
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
                                                               InterfaceBlockMeta &meta)
{
	auto &var_type = meta.strip_array ? get_variable_element_type(var) : get_variable_data_type(var);
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

	if (is_builtin && !meta.strip_array)
	{
		// For the builtin gl_PerVertex, we cannot treat it as a block anyways,
		// so redirect to qualified name.
		set_member_qualified_name(var_type.self, mbr_idx, qual_var_name);
	}
	else if (!meta.strip_array)
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
		uint32_t locn = get_accumulated_member_location(var, mbr_idx, meta.strip_array);
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
                                                  SPIRVariable &var, InterfaceBlockMeta &meta)
{
	auto &entry_func = get<SPIRFunction>(ir.default_entry_point);
	// Tessellation control I/O variables and tessellation evaluation per-point inputs are
	// usually declared as arrays. In these cases, we want to add the element type to the
	// interface block, since in Metal it's the interface block itself which is arrayed.
	auto &var_type = meta.strip_array ? get_variable_element_type(var) : get_variable_data_type(var);
	bool is_builtin = is_builtin_variable(var);
	auto builtin = BuiltIn(get_decoration(var.self, DecorationBuiltIn));

	if (var_type.basetype == SPIRType::Struct)
	{
		if (!is_builtin_type(var_type) && (!capture_output_to_buffer || storage == StorageClassInput) &&
		    !meta.strip_array)
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
			add_plain_variable_to_interface_block(storage, ib_var_ref, ib_type, var, meta);
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
					bool is_composite_type = is_matrix(mbr_type) || is_array(mbr_type);
					bool attribute_load_store =
					    storage == StorageClassInput && get_execution_model() != ExecutionModelFragment;
					bool storage_is_stage_io = storage == StorageClassInput || storage == StorageClassOutput;

					// ClipDistance always needs to be declared as user attributes.
					if (builtin == BuiltInClipDistance)
						is_builtin = false;

					if ((!is_builtin || attribute_load_store) && storage_is_stage_io && is_composite_type)
					{
						add_composite_member_variable_to_interface_block(storage, ib_var_ref, ib_type, var, mbr_idx,
						                                                 meta);
					}
					else
					{
						add_plain_member_variable_to_interface_block(storage, ib_var_ref, ib_type, var, mbr_idx, meta);
					}
				}
			}
		}
	}
	else if (get_execution_model() == ExecutionModelTessellationEvaluation && storage == StorageClassInput &&
	         !meta.strip_array && is_builtin && (builtin == BuiltInTessLevelOuter || builtin == BuiltInTessLevelInner))
	{
		add_tess_level_input_to_interface_block(ib_var_ref, ib_type, var);
	}
	else if (var_type.basetype == SPIRType::Boolean || var_type.basetype == SPIRType::Char ||
	         type_is_integral(var_type) || type_is_floating_point(var_type) || var_type.basetype == SPIRType::Boolean)
	{
		if (!is_builtin || has_active_builtin(builtin, storage))
		{
			bool is_composite_type = is_matrix(var_type) || is_array(var_type);
			bool storage_is_stage_io =
			    storage == StorageClassInput || (storage == StorageClassOutput && !capture_output_to_buffer);
			bool attribute_load_store = storage == StorageClassInput && get_execution_model() != ExecutionModelFragment;

			// ClipDistance always needs to be declared as user attributes.
			if (builtin == BuiltInClipDistance)
				is_builtin = false;

			// MSL does not allow matrices or arrays in input or output variables, so need to handle it specially.
			if ((!is_builtin || attribute_load_store) && storage_is_stage_io && is_composite_type)
			{
				add_composite_variable_to_interface_block(storage, ib_var_ref, ib_type, var, meta);
			}
			else
			{
				add_plain_variable_to_interface_block(storage, ib_var_ref, ib_type, var, meta);
			}
		}
	}
}

// Fix up the mapping of variables to interface member indices, which is used to compile access chains
// for per-vertex variables in a tessellation control shader.
void CompilerMSL::fix_up_interface_member_indices(StorageClass storage, uint32_t ib_type_id)
{
	// Only needed for tessellation shaders.
	// Need to redirect interface indices back to variables themselves.
	// For structs, each member of the struct need a separate instance.
	if (get_execution_model() != ExecutionModelTessellationControl &&
	    !(get_execution_model() == ExecutionModelTessellationEvaluation && storage == StorageClassInput))
		return;

	auto mbr_cnt = uint32_t(ir.meta[ib_type_id].members.size());
	for (uint32_t i = 0; i < mbr_cnt; i++)
	{
		uint32_t var_id = get_extended_member_decoration(ib_type_id, i, SPIRVCrossDecorationInterfaceOrigID);
		if (!var_id)
			continue;
		auto &var = get<SPIRVariable>(var_id);

		auto &type = get_variable_element_type(var);
		if (storage == StorageClassInput && type.basetype == SPIRType::Struct)
		{
			uint32_t mbr_idx = get_extended_member_decoration(ib_type_id, i, SPIRVCrossDecorationInterfaceMemberIndex);

			// Only set the lowest InterfaceMemberIndex for each variable member.
			// IB struct members will be emitted in-order w.r.t. interface member index.
			if (!has_extended_member_decoration(var_id, mbr_idx, SPIRVCrossDecorationInterfaceMemberIndex))
				set_extended_member_decoration(var_id, mbr_idx, SPIRVCrossDecorationInterfaceMemberIndex, i);
		}
		else
		{
			// Only set the lowest InterfaceMemberIndex for each variable.
			// IB struct members will be emitted in-order w.r.t. interface member index.
			if (!has_extended_decoration(var_id, SPIRVCrossDecorationInterfaceMemberIndex))
				set_extended_decoration(var_id, SPIRVCrossDecorationInterfaceMemberIndex, i);
		}
	}
}

// Add an interface structure for the type of storage, which is either StorageClassInput or StorageClassOutput.
// Returns the ID of the newly added variable, or zero if no variable was added.
uint32_t CompilerMSL::add_interface_block(StorageClass storage, bool patch)
{
	// Accumulate the variables that should appear in the interface struct.
	SmallVector<SPIRVariable *> vars;
	bool incl_builtins = storage == StorageClassOutput || is_tessellation_shader();
	bool has_seen_barycentric = false;

	InterfaceBlockMeta meta;

	// Varying interfaces between stages which use "user()" attribute can be dealt with
	// without explicit packing and unpacking of components. For any variables which link against the runtime
	// in some way (vertex attributes, fragment output, etc), we'll need to deal with it somehow.
	bool pack_components =
	    (storage == StorageClassInput && get_execution_model() == ExecutionModelVertex) ||
	    (storage == StorageClassOutput && get_execution_model() == ExecutionModelFragment) ||
	    (storage == StorageClassOutput && get_execution_model() == ExecutionModelVertex && capture_output_to_buffer);

	ir.for_each_typed_id<SPIRVariable>([&](uint32_t var_id, SPIRVariable &var) {
		if (var.storage != storage)
			return;

		auto &type = this->get<SPIRType>(var.basetype);

		bool is_builtin = is_builtin_variable(var);
		auto bi_type = BuiltIn(get_decoration(var_id, DecorationBuiltIn));
		uint32_t location = get_decoration(var_id, DecorationLocation);

		// These builtins are part of the stage in/out structs.
		bool is_interface_block_builtin =
		    (bi_type == BuiltInPosition || bi_type == BuiltInPointSize || bi_type == BuiltInClipDistance ||
		     bi_type == BuiltInCullDistance || bi_type == BuiltInLayer || bi_type == BuiltInViewportIndex ||
		     bi_type == BuiltInBaryCoordNV || bi_type == BuiltInBaryCoordNoPerspNV || bi_type == BuiltInFragDepth ||
		     bi_type == BuiltInFragStencilRefEXT || bi_type == BuiltInSampleMask) ||
		    (get_execution_model() == ExecutionModelTessellationEvaluation &&
		     (bi_type == BuiltInTessLevelOuter || bi_type == BuiltInTessLevelInner));

		bool is_active = interface_variable_exists_in_entry_point(var.self);
		if (is_builtin && is_active)
		{
			// Only emit the builtin if it's active in this entry point. Interface variable list might lie.
			is_active = has_active_builtin(bi_type, storage);
		}

		bool filter_patch_decoration = (has_decoration(var_id, DecorationPatch) || is_patch_block(type)) == patch;

		bool hidden = is_hidden_variable(var, incl_builtins);

		// ClipDistance is never hidden, we need to emulate it when used as an input.
		if (bi_type == BuiltInClipDistance)
			hidden = false;

		// It's not enough to simply avoid marking fragment outputs if the pipeline won't
		// accept them. We can't put them in the struct at all, or otherwise the compiler
		// complains that the outputs weren't explicitly marked.
		if (get_execution_model() == ExecutionModelFragment && storage == StorageClassOutput && !patch &&
			((is_builtin && ((bi_type == BuiltInFragDepth && !msl_options.enable_frag_depth_builtin) ||
		                    (bi_type == BuiltInFragStencilRefEXT && !msl_options.enable_frag_stencil_ref_builtin))) ||
		     (!is_builtin && !(msl_options.enable_frag_output_mask & (1 << location)))))
		{
			hidden = true;
			disabled_frag_outputs.push_back(var_id);
			// If a builtin, force it to have the proper name.
			if (is_builtin)
				set_name(var_id, builtin_to_glsl(bi_type, StorageClassFunction));
		}

		// Barycentric inputs must be emitted in stage-in, because they can have interpolation arguments.
		if (is_active && (bi_type == BuiltInBaryCoordNV || bi_type == BuiltInBaryCoordNoPerspNV))
		{
			if (has_seen_barycentric)
				SPIRV_CROSS_THROW("Cannot declare both BaryCoordNV and BaryCoordNoPerspNV in same shader in MSL.");
			has_seen_barycentric = true;
			hidden = false;
		}

		if (is_active && !hidden && type.pointer && filter_patch_decoration &&
		    (!is_builtin || is_interface_block_builtin))
		{
			vars.push_back(&var);

			if (!is_builtin)
			{
				// Need to deal specially with DecorationComponent.
				// Multiple variables can alias the same Location, and try to make sure each location is declared only once.
				// We will swizzle data in and out to make this work.
				// We only need to consider plain variables here, not composites.
				// This is only relevant for vertex inputs and fragment outputs.
				// Technically tessellation as well, but it is too complicated to support.
				uint32_t component = get_decoration(var_id, DecorationComponent);
				if (component != 0)
				{
					if (is_tessellation_shader())
						SPIRV_CROSS_THROW("Component decoration is not supported in tessellation shaders.");
					else if (pack_components)
					{
						auto &location_meta = meta.location_meta[location];
						location_meta.num_components = std::max(location_meta.num_components, component + type.vecsize);
					}
				}
			}
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
						// We zero-base the InstanceID & VertexID variables for HLSL emulation elsewhere, so don't do it twice
						if (msl_options.enable_base_index_zero)
						{
							statement("device ", to_name(ir.default_entry_point), "_", ib_var_ref, "& ", ib_var_ref,
							          " = ", output_buffer_var_name, "[", to_expression(builtin_instance_idx_id),
							          " * spvIndirectParams[0] + ", to_expression(builtin_vertex_idx_id), "];");
						}
						else
						{
							statement("device ", to_name(ir.default_entry_point), "_", ib_var_ref, "& ", ib_var_ref,
							          " = ", output_buffer_var_name, "[(", to_expression(builtin_instance_idx_id),
							          " - ", to_expression(builtin_base_instance_id), ") * spvIndirectParams[0] + ",
							          to_expression(builtin_vertex_idx_id), " - ",
							          to_expression(builtin_base_vertex_id), "];");
						}
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

	for (auto *p_var : vars)
	{
		bool strip_array =
		    (get_execution_model() == ExecutionModelTessellationControl ||
		     (get_execution_model() == ExecutionModelTessellationEvaluation && storage == StorageClassInput)) &&
		    !patch;

		meta.strip_array = strip_array;
		add_variable_to_interface_block(storage, ib_var_ref, ib_type, *p_var, meta);
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
	    ((builtin == BuiltInLayer || builtin == BuiltInViewportIndex || builtin == BuiltInFragStencilRefEXT) &&
	     type.basetype != SPIRType::UInt))
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
uint32_t CompilerMSL::ensure_correct_attribute_type(uint32_t type_id, uint32_t location, uint32_t num_components)
{
	auto &type = get<SPIRType>(type_id);

	auto p_va = vtx_attrs_by_location.find(location);
	if (p_va == end(vtx_attrs_by_location))
	{
		if (num_components != 0 && type.vecsize != num_components)
			return build_extended_vector_type(type_id, num_components);
		else
			return type_id;
	}

	switch (p_va->second.format)
	{
	case MSL_VERTEX_FORMAT_UINT8:
	{
		switch (type.basetype)
		{
		case SPIRType::UByte:
		case SPIRType::UShort:
		case SPIRType::UInt:
			if (num_components != 0 && type.vecsize != num_components)
				return build_extended_vector_type(type_id, num_components);
			else
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
		if (num_components != 0)
			base_type.vecsize = num_components;

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
			if (num_components != 0 && type.vecsize != num_components)
				return build_extended_vector_type(type_id, num_components);
			else
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
		if (num_components != 0)
			base_type.vecsize = num_components;

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
		if (num_components != 0 && type.vecsize != num_components)
			type_id = build_extended_vector_type(type_id, num_components);
		break;
	}

	return type_id;
}

void CompilerMSL::mark_struct_members_packed(const SPIRType &type)
{
	set_extended_decoration(type.self, SPIRVCrossDecorationPhysicalTypePacked);

	// Problem case! Struct needs to be placed at an awkward alignment.
	// Mark every member of the child struct as packed.
	uint32_t mbr_cnt = uint32_t(type.member_types.size());
	for (uint32_t i = 0; i < mbr_cnt; i++)
	{
		auto &mbr_type = get<SPIRType>(type.member_types[i]);
		if (mbr_type.basetype == SPIRType::Struct)
		{
			// Recursively mark structs as packed.
			auto *struct_type = &mbr_type;
			while (!struct_type->array.empty())
				struct_type = &get<SPIRType>(struct_type->parent_type);
			mark_struct_members_packed(*struct_type);
		}
		else if (!is_scalar(mbr_type))
			set_extended_member_decoration(type.self, i, SPIRVCrossDecorationPhysicalTypePacked);
	}
}

void CompilerMSL::mark_scalar_layout_structs(const SPIRType &type)
{
	uint32_t mbr_cnt = uint32_t(type.member_types.size());
	for (uint32_t i = 0; i < mbr_cnt; i++)
	{
		auto &mbr_type = get<SPIRType>(type.member_types[i]);
		if (mbr_type.basetype == SPIRType::Struct)
		{
			auto *struct_type = &mbr_type;
			while (!struct_type->array.empty())
				struct_type = &get<SPIRType>(struct_type->parent_type);

			if (has_extended_decoration(struct_type->self, SPIRVCrossDecorationPhysicalTypePacked))
				continue;

			uint32_t msl_alignment = get_declared_struct_member_alignment_msl(type, i);
			uint32_t msl_size = get_declared_struct_member_size_msl(type, i);
			uint32_t spirv_offset = type_struct_member_offset(type, i);
			uint32_t spirv_offset_next;
			if (i + 1 < mbr_cnt)
				spirv_offset_next = type_struct_member_offset(type, i + 1);
			else
				spirv_offset_next = spirv_offset + msl_size;

			// Both are complicated cases. In scalar layout, a struct of float3 might just consume 12 bytes,
			// and the next member will be placed at offset 12.
			bool struct_is_misaligned = (spirv_offset % msl_alignment) != 0;
			bool struct_is_too_large = spirv_offset + msl_size > spirv_offset_next;
			uint32_t array_stride = 0;
			bool struct_needs_explicit_padding = false;

			// Verify that if a struct is used as an array that ArrayStride matches the effective size of the struct.
			if (!mbr_type.array.empty())
			{
				array_stride = type_struct_member_array_stride(type, i);
				uint32_t dimensions = uint32_t(mbr_type.array.size() - 1);
				for (uint32_t dim = 0; dim < dimensions; dim++)
				{
					uint32_t array_size = to_array_size_literal(mbr_type, dim);
					array_stride /= max(array_size, 1u);
				}

				// Set expected struct size based on ArrayStride.
				struct_needs_explicit_padding = true;

				// If struct size is larger than array stride, we might be able to fit, if we tightly pack.
				if (get_declared_struct_size_msl(*struct_type) > array_stride)
					struct_is_too_large = true;
			}

			if (struct_is_misaligned || struct_is_too_large)
				mark_struct_members_packed(*struct_type);
			mark_scalar_layout_structs(*struct_type);

			if (struct_needs_explicit_padding)
			{
				msl_size = get_declared_struct_size_msl(*struct_type, true, true);
				if (array_stride < msl_size)
				{
					SPIRV_CROSS_THROW("Cannot express an array stride smaller than size of struct type.");
				}
				else
				{
					if (has_extended_decoration(struct_type->self, SPIRVCrossDecorationPaddingTarget))
					{
						if (array_stride !=
						    get_extended_decoration(struct_type->self, SPIRVCrossDecorationPaddingTarget))
							SPIRV_CROSS_THROW(
							    "A struct is used with different array strides. Cannot express this in MSL.");
					}
					else
						set_extended_decoration(struct_type->self, SPIRVCrossDecorationPaddingTarget, array_stride);
				}
			}
		}
	}
}

// Sort the members of the struct type by offset, and pack and then pad members where needed
// to align MSL members with SPIR-V offsets. The struct members are iterated twice. Packing
// occurs first, followed by padding, because packing a member reduces both its size and its
// natural alignment, possibly requiring a padding member to be added ahead of it.
void CompilerMSL::align_struct(SPIRType &ib_type, unordered_set<uint32_t> &aligned_structs)
{
	// We align structs recursively, so stop any redundant work.
	ID &ib_type_id = ib_type.self;
	if (aligned_structs.count(ib_type_id))
		return;
	aligned_structs.insert(ib_type_id);

	// Sort the members of the interface structure by their offset.
	// They should already be sorted per SPIR-V spec anyway.
	MemberSorter member_sorter(ib_type, ir.meta[ib_type_id], MemberSorter::Offset);
	member_sorter.sort();

	auto mbr_cnt = uint32_t(ib_type.member_types.size());

	for (uint32_t mbr_idx = 0; mbr_idx < mbr_cnt; mbr_idx++)
	{
		// Pack any dependent struct types before we pack a parent struct.
		auto &mbr_type = get<SPIRType>(ib_type.member_types[mbr_idx]);
		if (mbr_type.basetype == SPIRType::Struct)
			align_struct(mbr_type, aligned_structs);
	}

	// Test the alignment of each member, and if a member should be closer to the previous
	// member than the default spacing expects, it is likely that the previous member is in
	// a packed format. If so, and the previous member is packable, pack it.
	// For example ... this applies to any 3-element vector that is followed by a scalar.
	uint32_t msl_offset = 0;
	for (uint32_t mbr_idx = 0; mbr_idx < mbr_cnt; mbr_idx++)
	{
		// This checks the member in isolation, if the member needs some kind of type remapping to conform to SPIR-V
		// offsets, array strides and matrix strides.
		ensure_member_packing_rules_msl(ib_type, mbr_idx);

		// Align current offset to the current member's default alignment. If the member was packed, it will observe
		// the updated alignment here.
		uint32_t msl_align_mask = get_declared_struct_member_alignment_msl(ib_type, mbr_idx) - 1;
		uint32_t aligned_msl_offset = (msl_offset + msl_align_mask) & ~msl_align_mask;

		// Fetch the member offset as declared in the SPIRV.
		uint32_t spirv_mbr_offset = get_member_decoration(ib_type_id, mbr_idx, DecorationOffset);
		if (spirv_mbr_offset > aligned_msl_offset)
		{
			// Since MSL and SPIR-V have slightly different struct member alignment and
			// size rules, we'll pad to standard C-packing rules with a char[] array. If the member is farther
			// away than C-packing, expects, add an inert padding member before the the member.
			uint32_t padding_bytes = spirv_mbr_offset - aligned_msl_offset;
			set_extended_member_decoration(ib_type_id, mbr_idx, SPIRVCrossDecorationPaddingTarget, padding_bytes);

			// Re-align as a sanity check that aligning post-padding matches up.
			msl_offset += padding_bytes;
			aligned_msl_offset = (msl_offset + msl_align_mask) & ~msl_align_mask;
		}
		else if (spirv_mbr_offset < aligned_msl_offset)
		{
			// This should not happen, but deal with unexpected scenarios.
			// It *might* happen if a sub-struct has a larger alignment requirement in MSL than SPIR-V.
			SPIRV_CROSS_THROW("Cannot represent buffer block correctly in MSL.");
		}

		assert(aligned_msl_offset == spirv_mbr_offset);

		// Increment the current offset to be positioned immediately after the current member.
		// Don't do this for the last member since it can be unsized, and it is not relevant for padding purposes here.
		if (mbr_idx + 1 < mbr_cnt)
			msl_offset = aligned_msl_offset + get_declared_struct_member_size_msl(ib_type, mbr_idx);
	}
}

bool CompilerMSL::validate_member_packing_rules_msl(const SPIRType &type, uint32_t index) const
{
	auto &mbr_type = get<SPIRType>(type.member_types[index]);
	uint32_t spirv_offset = get_member_decoration(type.self, index, DecorationOffset);

	if (index + 1 < type.member_types.size())
	{
		// First, we will check offsets. If SPIR-V offset + MSL size > SPIR-V offset of next member,
		// we *must* perform some kind of remapping, no way getting around it.
		// We can always pad after this member if necessary, so that case is fine.
		uint32_t spirv_offset_next = get_member_decoration(type.self, index + 1, DecorationOffset);
		assert(spirv_offset_next >= spirv_offset);
		uint32_t maximum_size = spirv_offset_next - spirv_offset;
		uint32_t msl_mbr_size = get_declared_struct_member_size_msl(type, index);
		if (msl_mbr_size > maximum_size)
			return false;
	}

	if (!mbr_type.array.empty())
	{
		// If we have an array type, array stride must match exactly with SPIR-V.
		uint32_t spirv_array_stride = type_struct_member_array_stride(type, index);
		uint32_t msl_array_stride = get_declared_struct_member_array_stride_msl(type, index);
		if (spirv_array_stride != msl_array_stride)
			return false;
	}

	if (is_matrix(mbr_type))
	{
		// Need to check MatrixStride as well.
		uint32_t spirv_matrix_stride = type_struct_member_matrix_stride(type, index);
		uint32_t msl_matrix_stride = get_declared_struct_member_matrix_stride_msl(type, index);
		if (spirv_matrix_stride != msl_matrix_stride)
			return false;
	}

	// Now, we check alignment.
	uint32_t msl_alignment = get_declared_struct_member_alignment_msl(type, index);
	if ((spirv_offset % msl_alignment) != 0)
		return false;

	// We're in the clear.
	return true;
}

// Here we need to verify that the member type we declare conforms to Offset, ArrayStride or MatrixStride restrictions.
// If there is a mismatch, we need to emit remapped types, either normal types, or "packed_X" types.
// In odd cases we need to emit packed and remapped types, for e.g. weird matrices or arrays with weird array strides.
void CompilerMSL::ensure_member_packing_rules_msl(SPIRType &ib_type, uint32_t index)
{
	if (validate_member_packing_rules_msl(ib_type, index))
		return;

	// We failed validation.
	// This case will be nightmare-ish to deal with. This could possibly happen if struct alignment does not quite
	// match up with what we want. Scalar block layout comes to mind here where we might have to work around the rule
	// that struct alignment == max alignment of all members and struct size depends on this alignment.
	auto &mbr_type = get<SPIRType>(ib_type.member_types[index]);
	if (mbr_type.basetype == SPIRType::Struct)
		SPIRV_CROSS_THROW("Cannot perform any repacking for structs when it is used as a member of another struct.");

	// Perform remapping here.
	set_extended_member_decoration(ib_type.self, index, SPIRVCrossDecorationPhysicalTypePacked);

	// Try validating again, now with packed.
	if (validate_member_packing_rules_msl(ib_type, index))
		return;

	// We're in deep trouble, and we need to create a new PhysicalType which matches up with what we expect.
	// A lot of work goes here ...
	// We will need remapping on Load and Store to translate the types between Logical and Physical.

	// First, we check if we have small vector std140 array.
	// We detect this if we have an array of vectors, and array stride is greater than number of elements.
	if (!mbr_type.array.empty() && !is_matrix(mbr_type))
	{
		uint32_t array_stride = type_struct_member_array_stride(ib_type, index);

		// Hack off array-of-arrays until we find the array stride per element we must have to make it work.
		uint32_t dimensions = uint32_t(mbr_type.array.size() - 1);
		for (uint32_t dim = 0; dim < dimensions; dim++)
			array_stride /= max(to_array_size_literal(mbr_type, dim), 1u);

		uint32_t elems_per_stride = array_stride / (mbr_type.width / 8);

		if (elems_per_stride == 3)
			SPIRV_CROSS_THROW("Cannot use ArrayStride of 3 elements in remapping scenarios.");
		else if (elems_per_stride > 4)
			SPIRV_CROSS_THROW("Cannot represent vectors with more than 4 elements in MSL.");

		auto physical_type = mbr_type;
		physical_type.vecsize = elems_per_stride;
		physical_type.parent_type = 0;
		uint32_t type_id = ir.increase_bound_by(1);
		set<SPIRType>(type_id, physical_type);
		set_extended_member_decoration(ib_type.self, index, SPIRVCrossDecorationPhysicalTypeID, type_id);
		set_decoration(type_id, DecorationArrayStride, array_stride);

		// Remove packed_ for vectors of size 1, 2 and 4.
		if (has_extended_decoration(ib_type.self, SPIRVCrossDecorationPhysicalTypePacked))
			SPIRV_CROSS_THROW("Unable to remove packed decoration as entire struct must be fully packed. Do not mix "
			                  "scalar and std140 layout rules.");
		else
			unset_extended_member_decoration(ib_type.self, index, SPIRVCrossDecorationPhysicalTypePacked);
	}
	else if (is_matrix(mbr_type))
	{
		// MatrixStride might be std140-esque.
		uint32_t matrix_stride = type_struct_member_matrix_stride(ib_type, index);

		uint32_t elems_per_stride = matrix_stride / (mbr_type.width / 8);

		if (elems_per_stride == 3)
			SPIRV_CROSS_THROW("Cannot use ArrayStride of 3 elements in remapping scenarios.");
		else if (elems_per_stride > 4)
			SPIRV_CROSS_THROW("Cannot represent vectors with more than 4 elements in MSL.");

		bool row_major = has_member_decoration(ib_type.self, index, DecorationRowMajor);

		auto physical_type = mbr_type;
		physical_type.parent_type = 0;
		if (row_major)
			physical_type.columns = elems_per_stride;
		else
			physical_type.vecsize = elems_per_stride;
		uint32_t type_id = ir.increase_bound_by(1);
		set<SPIRType>(type_id, physical_type);
		set_extended_member_decoration(ib_type.self, index, SPIRVCrossDecorationPhysicalTypeID, type_id);

		// Remove packed_ for vectors of size 1, 2 and 4.
		if (has_extended_decoration(ib_type.self, SPIRVCrossDecorationPhysicalTypePacked))
			SPIRV_CROSS_THROW("Unable to remove packed decoration as entire struct must be fully packed. Do not mix "
			                  "scalar and std140 layout rules.");
		else
			unset_extended_member_decoration(ib_type.self, index, SPIRVCrossDecorationPhysicalTypePacked);
	}

	// This better validate now, or we must fail gracefully.
	if (!validate_member_packing_rules_msl(ib_type, index))
		SPIRV_CROSS_THROW("Found a buffer packing case which we cannot represent in MSL.");
}

void CompilerMSL::emit_store_statement(uint32_t lhs_expression, uint32_t rhs_expression)
{
	auto &type = expression_type(rhs_expression);

	bool lhs_remapped_type = has_extended_decoration(lhs_expression, SPIRVCrossDecorationPhysicalTypeID);
	bool lhs_packed_type = has_extended_decoration(lhs_expression, SPIRVCrossDecorationPhysicalTypePacked);
	auto *lhs_e = maybe_get<SPIRExpression>(lhs_expression);
	auto *rhs_e = maybe_get<SPIRExpression>(rhs_expression);

	bool transpose = lhs_e && lhs_e->need_transpose;

	// No physical type remapping, and no packed type, so can just emit a store directly.
	if (!lhs_remapped_type && !lhs_packed_type)
	{
		// We might not be dealing with remapped physical types or packed types,
		// but we might be doing a clean store to a row-major matrix.
		// In this case, we just flip transpose states, and emit the store, a transpose must be in the RHS expression, if any.
		if (is_matrix(type) && lhs_e && lhs_e->need_transpose)
		{
			if (!rhs_e)
				SPIRV_CROSS_THROW("Need to transpose right-side expression of a store to row-major matrix, but it is "
				                  "not a SPIRExpression.");
			lhs_e->need_transpose = false;

			if (rhs_e && rhs_e->need_transpose)
			{
				// Direct copy, but might need to unpack RHS.
				// Skip the transpose, as we will transpose when writing to LHS and transpose(transpose(T)) == T.
				rhs_e->need_transpose = false;
				statement(to_expression(lhs_expression), " = ", to_unpacked_row_major_matrix_expression(rhs_expression),
				          ";");
				rhs_e->need_transpose = true;
			}
			else
				statement(to_expression(lhs_expression), " = transpose(", to_unpacked_expression(rhs_expression), ");");

			lhs_e->need_transpose = true;
			register_write(lhs_expression);
		}
		else if (lhs_e && lhs_e->need_transpose)
		{
			lhs_e->need_transpose = false;

			// Storing a column to a row-major matrix. Unroll the write.
			for (uint32_t c = 0; c < type.vecsize; c++)
			{
				auto lhs_expr = to_dereferenced_expression(lhs_expression);
				auto column_index = lhs_expr.find_last_of('[');
				if (column_index != string::npos)
				{
					statement(lhs_expr.insert(column_index, join('[', c, ']')), " = ",
					          to_extract_component_expression(rhs_expression, c), ";");
				}
			}
			lhs_e->need_transpose = true;
			register_write(lhs_expression);
		}
		else
			CompilerGLSL::emit_store_statement(lhs_expression, rhs_expression);
	}
	else if (!lhs_remapped_type && !is_matrix(type) && !transpose)
	{
		// Even if the target type is packed, we can directly store to it. We cannot store to packed matrices directly,
		// since they are declared as array of vectors instead, and we need the fallback path below.
		CompilerGLSL::emit_store_statement(lhs_expression, rhs_expression);
	}
	else
	{
		// Special handling when storing to a remapped physical type.
		// This is mostly to deal with std140 padded matrices or vectors.

		TypeID physical_type_id = lhs_remapped_type ?
		                              ID(get_extended_decoration(lhs_expression, SPIRVCrossDecorationPhysicalTypeID)) :
		                              type.self;

		auto &physical_type = get<SPIRType>(physical_type_id);

		static const char *swizzle_lut[] = {
			".x",
			".xy",
			".xyz",
			"",
		};

		if (is_matrix(type))
		{
			// Packed matrices are stored as arrays of packed vectors, so we need
			// to assign the vectors one at a time.
			// For row-major matrices, we need to transpose the *right-hand* side,
			// not the left-hand side.

			// Lots of cases to cover here ...

			bool rhs_transpose = rhs_e && rhs_e->need_transpose;

			// We're dealing with transpose manually.
			if (rhs_transpose)
				rhs_e->need_transpose = false;

			if (transpose)
			{
				// We're dealing with transpose manually.
				lhs_e->need_transpose = false;

				const char *store_swiz = "";
				if (physical_type.columns != type.columns)
					store_swiz = swizzle_lut[type.columns - 1];

				if (rhs_transpose)
				{
					// If RHS is also transposed, we can just copy row by row.
					for (uint32_t i = 0; i < type.vecsize; i++)
					{
						statement(to_enclosed_expression(lhs_expression), "[", i, "]", store_swiz, " = ",
						          to_unpacked_row_major_matrix_expression(rhs_expression), "[", i, "];");
					}
				}
				else
				{
					auto vector_type = expression_type(rhs_expression);
					vector_type.vecsize = vector_type.columns;
					vector_type.columns = 1;

					// Transpose on the fly. Emitting a lot of full transpose() ops and extracting lanes seems very bad,
					// so pick out individual components instead.
					for (uint32_t i = 0; i < type.vecsize; i++)
					{
						string rhs_row = type_to_glsl_constructor(vector_type) + "(";
						for (uint32_t j = 0; j < vector_type.vecsize; j++)
						{
							rhs_row += join(to_enclosed_unpacked_expression(rhs_expression), "[", j, "][", i, "]");
							if (j + 1 < vector_type.vecsize)
								rhs_row += ", ";
						}
						rhs_row += ")";

						statement(to_enclosed_expression(lhs_expression), "[", i, "]", store_swiz, " = ", rhs_row, ";");
					}
				}

				// We're dealing with transpose manually.
				lhs_e->need_transpose = true;
			}
			else
			{
				const char *store_swiz = "";
				if (physical_type.vecsize != type.vecsize)
					store_swiz = swizzle_lut[type.vecsize - 1];

				if (rhs_transpose)
				{
					auto vector_type = expression_type(rhs_expression);
					vector_type.columns = 1;

					// Transpose on the fly. Emitting a lot of full transpose() ops and extracting lanes seems very bad,
					// so pick out individual components instead.
					for (uint32_t i = 0; i < type.columns; i++)
					{
						string rhs_row = type_to_glsl_constructor(vector_type) + "(";
						for (uint32_t j = 0; j < vector_type.vecsize; j++)
						{
							// Need to explicitly unpack expression since we've mucked with transpose state.
							auto unpacked_expr = to_unpacked_row_major_matrix_expression(rhs_expression);
							rhs_row += join(unpacked_expr, "[", j, "][", i, "]");
							if (j + 1 < vector_type.vecsize)
								rhs_row += ", ";
						}
						rhs_row += ")";

						statement(to_enclosed_expression(lhs_expression), "[", i, "]", store_swiz, " = ", rhs_row, ";");
					}
				}
				else
				{
					// Copy column-by-column.
					for (uint32_t i = 0; i < type.columns; i++)
					{
						statement(to_enclosed_expression(lhs_expression), "[", i, "]", store_swiz, " = ",
						          to_enclosed_unpacked_expression(rhs_expression), "[", i, "];");
					}
				}
			}

			// We're dealing with transpose manually.
			if (rhs_transpose)
				rhs_e->need_transpose = true;
		}
		else if (transpose)
		{
			lhs_e->need_transpose = false;

			// Storing a column to a row-major matrix. Unroll the write.
			for (uint32_t c = 0; c < type.vecsize; c++)
			{
				auto lhs_expr = to_enclosed_expression(lhs_expression);
				auto column_index = lhs_expr.find_last_of('[');
				if (column_index != string::npos)
				{
					statement(lhs_expr.insert(column_index, join('[', c, ']')), " = ",
					          to_extract_component_expression(rhs_expression, c), ";");
				}
			}

			lhs_e->need_transpose = true;
		}
		else if ((is_matrix(physical_type) || is_array(physical_type)) && physical_type.vecsize > type.vecsize)
		{
			assert(type.vecsize >= 1 && type.vecsize <= 3);

			// If we have packed types, we cannot use swizzled stores.
			// We could technically unroll the store for each element if needed.
			// When remapping to a std140 physical type, we always get float4,
			// and the packed decoration should always be removed.
			assert(!lhs_packed_type);

			string lhs = to_dereferenced_expression(lhs_expression);
			string rhs = to_pointer_expression(rhs_expression);

			// Unpack the expression so we can store to it with a float or float2.
			// It's still an l-value, so it's fine. Most other unpacking of expressions turn them into r-values instead.
			lhs = enclose_expression(lhs) + swizzle_lut[type.vecsize - 1];
			if (!optimize_read_modify_write(expression_type(rhs_expression), lhs, rhs))
				statement(lhs, " = ", rhs, ";");
		}
		else if (!is_matrix(type))
		{
			string lhs = to_dereferenced_expression(lhs_expression);
			string rhs = to_pointer_expression(rhs_expression);
			if (!optimize_read_modify_write(expression_type(rhs_expression), lhs, rhs))
				statement(lhs, " = ", rhs, ";");
		}

		register_write(lhs_expression);
	}
}

static bool expression_ends_with(const string &expr_str, const std::string &ending)
{
	if (expr_str.length() >= ending.length())
		return (expr_str.compare(expr_str.length() - ending.length(), ending.length(), ending) == 0);
	else
		return false;
}

// Converts the format of the current expression from packed to unpacked,
// by wrapping the expression in a constructor of the appropriate type.
// Also, handle special physical ID remapping scenarios, similar to emit_store_statement().
string CompilerMSL::unpack_expression_type(string expr_str, const SPIRType &type, uint32_t physical_type_id,
                                           bool packed, bool row_major)
{
	// Trivial case, nothing to do.
	if (physical_type_id == 0 && !packed)
		return expr_str;

	const SPIRType *physical_type = nullptr;
	if (physical_type_id)
		physical_type = &get<SPIRType>(physical_type_id);

	static const char *swizzle_lut[] = {
		".x",
		".xy",
		".xyz",
	};

	if (physical_type && is_vector(*physical_type) && is_array(*physical_type) &&
	    physical_type->vecsize > type.vecsize && !expression_ends_with(expr_str, swizzle_lut[type.vecsize - 1]))
	{
		// std140 array cases for vectors.
		assert(type.vecsize >= 1 && type.vecsize <= 3);
		return enclose_expression(expr_str) + swizzle_lut[type.vecsize - 1];
	}
	else if (physical_type && is_matrix(*physical_type) && is_vector(type) && physical_type->vecsize > type.vecsize)
	{
		// Extract column from padded matrix.
		assert(type.vecsize >= 1 && type.vecsize <= 3);
		return enclose_expression(expr_str) + swizzle_lut[type.vecsize - 1];
	}
	else if (is_matrix(type))
	{
		// Packed matrices are stored as arrays of packed vectors. Unfortunately,
		// we can't just pass the array straight to the matrix constructor. We have to
		// pass each vector individually, so that they can be unpacked to normal vectors.
		if (!physical_type)
			physical_type = &type;

		uint32_t vecsize = type.vecsize;
		uint32_t columns = type.columns;
		if (row_major)
			swap(vecsize, columns);

		uint32_t physical_vecsize = row_major ? physical_type->columns : physical_type->vecsize;

		const char *base_type = type.width == 16 ? "half" : "float";
		string unpack_expr = join(base_type, columns, "x", vecsize, "(");

		const char *load_swiz = "";

		if (physical_vecsize != vecsize)
			load_swiz = swizzle_lut[vecsize - 1];

		for (uint32_t i = 0; i < columns; i++)
		{
			if (i > 0)
				unpack_expr += ", ";

			if (packed)
				unpack_expr += join(base_type, physical_vecsize, "(", expr_str, "[", i, "]", ")", load_swiz);
			else
				unpack_expr += join(expr_str, "[", i, "]", load_swiz);
		}

		unpack_expr += ")";
		return unpack_expr;
	}
	else
	{
		return join(type_to_glsl(type), "(", expr_str, ")");
	}
}

// Emits the file header info
void CompilerMSL::emit_header()
{
	// This particular line can be overridden during compilation, so make it a flag and not a pragma line.
	if (suppress_missing_prototypes)
		statement("#pragma clang diagnostic ignored \"-Wmissing-prototypes\"");

	// Disable warning about missing braces for array<T> template to make arrays a value type
	if (spv_function_implementations.count(SPVFuncImplUnsafeArray) != 0)
		statement("#pragma clang diagnostic ignored \"-Wmissing-braces\"");

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

// Template struct like spvUnsafeArray<> need to be declared *before* any resources are declared
void CompilerMSL::emit_custom_templates()
{
	for (const auto &spv_func : spv_function_implementations)
	{
		switch (spv_func)
		{
		case SPVFuncImplUnsafeArray:
			statement("template<typename T, size_t Num>");
			statement("struct spvUnsafeArray");
			begin_scope();
			statement("T elements[Num ? Num : 1];");
			statement("");
			statement("thread T& operator [] (size_t pos) thread");
			begin_scope();
			statement("return elements[pos];");
			end_scope();
			statement("constexpr const thread T& operator [] (size_t pos) const thread");
			begin_scope();
			statement("return elements[pos];");
			end_scope();
			statement("");
			statement("device T& operator [] (size_t pos) device");
			begin_scope();
			statement("return elements[pos];");
			end_scope();
			statement("constexpr const device T& operator [] (size_t pos) const device");
			begin_scope();
			statement("return elements[pos];");
			end_scope();
			statement("");
			statement("constexpr const constant T& operator [] (size_t pos) const constant");
			begin_scope();
			statement("return elements[pos];");
			end_scope();
			statement("");
			statement("threadgroup T& operator [] (size_t pos) threadgroup");
			begin_scope();
			statement("return elements[pos];");
			end_scope();
			statement("constexpr const threadgroup T& operator [] (size_t pos) const threadgroup");
			begin_scope();
			statement("return elements[pos];");
			end_scope();
			end_scope_decl();
			statement("");
			break;

		default:
			break;
		}
	}
}

// Emits any needed custom function bodies.
// Metal helper functions must be static force-inline, i.e. static inline __attribute__((always_inline))
// otherwise they will cause problems when linked together in a single Metallib.
void CompilerMSL::emit_custom_functions()
{
	for (uint32_t i = SPVFuncImplArrayCopyMultidimMax; i >= 2; i--)
		if (spv_function_implementations.count(static_cast<SPVFuncImpl>(SPVFuncImplArrayCopyMultidimBase + i)))
			spv_function_implementations.insert(static_cast<SPVFuncImpl>(SPVFuncImplArrayCopyMultidimBase + i - 1));

	if (spv_function_implementations.count(SPVFuncImplDynamicImageSampler))
	{
		// Unfortunately, this one needs a lot of the other functions to compile OK.
		if (!msl_options.supports_msl_version(2))
			SPIRV_CROSS_THROW(
			    "spvDynamicImageSampler requires default-constructible texture objects, which require MSL 2.0.");
		spv_function_implementations.insert(SPVFuncImplForwardArgs);
		spv_function_implementations.insert(SPVFuncImplTextureSwizzle);
		if (msl_options.swizzle_texture_samples)
			spv_function_implementations.insert(SPVFuncImplGatherSwizzle);
		for (uint32_t i = SPVFuncImplChromaReconstructNearest2Plane;
		     i <= SPVFuncImplChromaReconstructLinear420XMidpointYMidpoint3Plane; i++)
			spv_function_implementations.insert(static_cast<SPVFuncImpl>(i));
		spv_function_implementations.insert(SPVFuncImplExpandITUFullRange);
		spv_function_implementations.insert(SPVFuncImplExpandITUNarrowRange);
		spv_function_implementations.insert(SPVFuncImplConvertYCbCrBT709);
		spv_function_implementations.insert(SPVFuncImplConvertYCbCrBT601);
		spv_function_implementations.insert(SPVFuncImplConvertYCbCrBT2020);
	}

	for (uint32_t i = SPVFuncImplChromaReconstructNearest2Plane;
	     i <= SPVFuncImplChromaReconstructLinear420XMidpointYMidpoint3Plane; i++)
		if (spv_function_implementations.count(static_cast<SPVFuncImpl>(i)))
			spv_function_implementations.insert(SPVFuncImplForwardArgs);

	if (spv_function_implementations.count(SPVFuncImplTextureSwizzle) ||
	    spv_function_implementations.count(SPVFuncImplGatherSwizzle) ||
	    spv_function_implementations.count(SPVFuncImplGatherCompareSwizzle))
	{
		spv_function_implementations.insert(SPVFuncImplForwardArgs);
		spv_function_implementations.insert(SPVFuncImplGetSwizzle);
	}

	for (const auto &spv_func : spv_function_implementations)
	{
		switch (spv_func)
		{
		case SPVFuncImplMod:
			statement("// Implementation of the GLSL mod() function, which is slightly different than Metal fmod()");
			statement("template<typename Tx, typename Ty>");
			statement("inline Tx mod(Tx x, Ty y)");
			begin_scope();
			statement("return x - y * floor(x / y);");
			end_scope();
			statement("");
			break;

		case SPVFuncImplRadians:
			statement("// Implementation of the GLSL radians() function");
			statement("template<typename T>");
			statement("inline T radians(T d)");
			begin_scope();
			statement("return d * T(0.01745329251);");
			end_scope();
			statement("");
			break;

		case SPVFuncImplDegrees:
			statement("// Implementation of the GLSL degrees() function");
			statement("template<typename T>");
			statement("inline T degrees(T r)");
			begin_scope();
			statement("return r * T(57.2957795131);");
			end_scope();
			statement("");
			break;

		case SPVFuncImplFindILsb:
			statement("// Implementation of the GLSL findLSB() function");
			statement("template<typename T>");
			statement("inline T spvFindLSB(T x)");
			begin_scope();
			statement("return select(ctz(x), T(-1), x == T(0));");
			end_scope();
			statement("");
			break;

		case SPVFuncImplFindUMsb:
			statement("// Implementation of the unsigned GLSL findMSB() function");
			statement("template<typename T>");
			statement("inline T spvFindUMSB(T x)");
			begin_scope();
			statement("return select(clz(T(0)) - (clz(x) + T(1)), T(-1), x == T(0));");
			end_scope();
			statement("");
			break;

		case SPVFuncImplFindSMsb:
			statement("// Implementation of the signed GLSL findMSB() function");
			statement("template<typename T>");
			statement("inline T spvFindSMSB(T x)");
			begin_scope();
			statement("T v = select(x, T(-1) - x, x < T(0));");
			statement("return select(clz(T(0)) - (clz(v) + T(1)), T(-1), v == T(0));");
			end_scope();
			statement("");
			break;

		case SPVFuncImplSSign:
			statement("// Implementation of the GLSL sign() function for integer types");
			statement("template<typename T, typename E = typename enable_if<is_integral<T>::value>::type>");
			statement("inline T sign(T x)");
			begin_scope();
			statement("return select(select(select(x, T(0), x == T(0)), T(1), x > T(0)), T(-1), x < T(0));");
			end_scope();
			statement("");
			break;

		case SPVFuncImplArrayCopy:
		case SPVFuncImplArrayOfArrayCopy2Dim:
		case SPVFuncImplArrayOfArrayCopy3Dim:
		case SPVFuncImplArrayOfArrayCopy4Dim:
		case SPVFuncImplArrayOfArrayCopy5Dim:
		case SPVFuncImplArrayOfArrayCopy6Dim:
		{
			// Unfortunately we cannot template on the address space, so combinatorial explosion it is.
			static const char *function_name_tags[] = {
				"FromConstantToStack",    "FromConstantToThreadGroup", "FromStackToStack",
				"FromStackToThreadGroup", "FromThreadGroupToStack",    "FromThreadGroupToThreadGroup",
			};

			static const char *src_address_space[] = {
				"constant", "constant", "thread const", "thread const", "threadgroup const", "threadgroup const",
			};

			static const char *dst_address_space[] = {
				"thread", "threadgroup", "thread", "threadgroup", "thread", "threadgroup",
			};

			for (uint32_t variant = 0; variant < 6; variant++)
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

				statement("inline void spvArrayCopy", function_name_tags[variant], dimensions, "(",
				          dst_address_space[variant], " T (&dst)", array_arg, ", ", src_address_space[variant],
				          " T (&src)", array_arg, ")");

				begin_scope();
				statement("for (uint i = 0; i < A; i++)");
				begin_scope();

				if (dimensions == 1)
					statement("dst[i] = src[i];");
				else
					statement("spvArrayCopy", function_name_tags[variant], dimensions - 1, "(dst[i], src[i]);");
				end_scope();
				end_scope();
				statement("");
			}
			break;
		}

		// Support for Metal 2.1's new texture_buffer type.
		case SPVFuncImplTexelBufferCoords:
		{
			if (msl_options.texel_buffer_texture_width > 0)
			{
				string tex_width_str = convert_to_string(msl_options.texel_buffer_texture_width);
				statement("// Returns 2D texture coords corresponding to 1D texel buffer coords");
				statement(force_inline);
				statement("uint2 spvTexelBufferCoord(uint tc)");
				begin_scope();
				statement(join("return uint2(tc % ", tex_width_str, ", tc / ", tex_width_str, ");"));
				end_scope();
				statement("");
			}
			else
			{
				statement("// Returns 2D texture coords corresponding to 1D texel buffer coords");
				statement(
				    "#define spvTexelBufferCoord(tc, tex) uint2((tc) % (tex).get_width(), (tc) / (tex).get_width())");
				statement("");
			}
			break;
		}

		// Emulate texture2D atomic operations
		case SPVFuncImplImage2DAtomicCoords:
		{
			statement("// Returns buffer coords corresponding to 2D texture coords for emulating 2D texture atomics");
			statement("#define spvImage2DAtomicCoord(tc, tex) (((tex).get_width() * (tc).x) + (tc).y)");
			statement("");
			break;
		}

		// "fadd" intrinsic support
		case SPVFuncImplFAdd:
			statement("template<typename T>");
			statement("T spvFAdd(T l, T r)");
			begin_scope();
			statement("return fma(T(1), l, r);");
			end_scope();
			statement("");
			break;

		// "fmul' intrinsic support
		case SPVFuncImplFMul:
			statement("template<typename T>");
			statement("T spvFMul(T l, T r)");
			begin_scope();
			statement("return fma(l, r, T(0));");
			end_scope();
			statement("");

			statement("template<typename T, int Cols, int Rows>");
			statement("vec<T, Cols> spvFMulVectorMatrix(vec<T, Rows> v, matrix<T, Cols, Rows> m)");
			begin_scope();
			statement("vec<T, Cols> res = vec<T, Cols>(0);");
			statement("for (uint i = Rows; i > 0; --i)");
			begin_scope();
			statement("vec<T, Cols> tmp(0);");
			statement("for (uint j = 0; j < Cols; ++j)");
			begin_scope();
			statement("tmp[j] = m[j][i - 1];");
			end_scope();
			statement("res = fma(tmp, vec<T, Cols>(v[i - 1]), res);");
			end_scope();
			statement("return res;");
			end_scope();
			statement("");

			statement("template<typename T, int Cols, int Rows>");
			statement("vec<T, Rows> spvFMulMatrixVector(matrix<T, Cols, Rows> m, vec<T, Cols> v)");
			begin_scope();
			statement("vec<T, Rows> res = vec<T, Rows>(0);");
			statement("for (uint i = Cols; i > 0; --i)");
			begin_scope();
			statement("res = fma(m[i - 1], vec<T, Rows>(v[i - 1]), res);");
			end_scope();
			statement("return res;");
			end_scope();
			statement("");

			statement("template<typename T, int LCols, int LRows, int RCols, int RRows>");
			statement(
			    "matrix<T, RCols, LRows> spvFMulMatrixMatrix(matrix<T, LCols, LRows> l, matrix<T, RCols, RRows> r)");
			begin_scope();
			statement("matrix<T, RCols, LRows> res;");
			statement("for (uint i = 0; i < RCols; i++)");
			begin_scope();
			statement("vec<T, RCols> tmp(0);");
			statement("for (uint j = 0; j < LCols; j++)");
			begin_scope();
			statement("tmp = fma(vec<T, RCols>(r[i][j]), l[j], tmp);");
			end_scope();
			statement("res[i] = tmp;");
			end_scope();
			statement("return res;");
			end_scope();
			statement("");
			break;

		// Emulate texturecube_array with texture2d_array for iOS where this type is not available
		case SPVFuncImplCubemapTo2DArrayFace:
			statement(force_inline);
			statement("float3 spvCubemapTo2DArrayFace(float3 P)");
			begin_scope();
			statement("float3 Coords = abs(P.xyz);");
			statement("float CubeFace = 0;");
			statement("float ProjectionAxis = 0;");
			statement("float u = 0;");
			statement("float v = 0;");
			statement("if (Coords.x >= Coords.y && Coords.x >= Coords.z)");
			begin_scope();
			statement("CubeFace = P.x >= 0 ? 0 : 1;");
			statement("ProjectionAxis = Coords.x;");
			statement("u = P.x >= 0 ? -P.z : P.z;");
			statement("v = -P.y;");
			end_scope();
			statement("else if (Coords.y >= Coords.x && Coords.y >= Coords.z)");
			begin_scope();
			statement("CubeFace = P.y >= 0 ? 2 : 3;");
			statement("ProjectionAxis = Coords.y;");
			statement("u = P.x;");
			statement("v = P.y >= 0 ? P.z : -P.z;");
			end_scope();
			statement("else");
			begin_scope();
			statement("CubeFace = P.z >= 0 ? 4 : 5;");
			statement("ProjectionAxis = Coords.z;");
			statement("u = P.z >= 0 ? P.x : -P.x;");
			statement("v = -P.y;");
			end_scope();
			statement("u = 0.5 * (u/ProjectionAxis + 1);");
			statement("v = 0.5 * (v/ProjectionAxis + 1);");
			statement("return float3(u, v, CubeFace);");
			end_scope();
			statement("");
			break;

		case SPVFuncImplInverse4x4:
			statement("// Returns the determinant of a 2x2 matrix.");
			statement(force_inline);
			statement("float spvDet2x2(float a1, float a2, float b1, float b2)");
			begin_scope();
			statement("return a1 * b2 - b1 * a2;");
			end_scope();
			statement("");

			statement("// Returns the determinant of a 3x3 matrix.");
			statement(force_inline);
			statement("float spvDet3x3(float a1, float a2, float a3, float b1, float b2, float b3, float c1, "
			          "float c2, float c3)");
			begin_scope();
			statement("return a1 * spvDet2x2(b2, b3, c2, c3) - b1 * spvDet2x2(a2, a3, c2, c3) + c1 * spvDet2x2(a2, a3, "
			          "b2, b3);");
			end_scope();
			statement("");
			statement("// Returns the inverse of a matrix, by using the algorithm of calculating the classical");
			statement("// adjoint and dividing by the determinant. The contents of the matrix are changed.");
			statement(force_inline);
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
				statement(force_inline);
				statement("float spvDet2x2(float a1, float a2, float b1, float b2)");
				begin_scope();
				statement("return a1 * b2 - b1 * a2;");
				end_scope();
				statement("");
			}

			statement("// Returns the inverse of a matrix, by using the algorithm of calculating the classical");
			statement("// adjoint and dividing by the determinant. The contents of the matrix are changed.");
			statement(force_inline);
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
			statement(force_inline);
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

		case SPVFuncImplForwardArgs:
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
			break;

		case SPVFuncImplGetSwizzle:
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
			break;

		case SPVFuncImplTextureSwizzle:
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
			break;

		case SPVFuncImplGatherSwizzle:
			statement("// Wrapper function that swizzles texture gathers.");
			statement("template<typename T, template<typename, access = access::sample, typename = void> class Tex, "
			          "typename... Ts>");
			statement("inline vec<T, 4> spvGatherSwizzle(const thread Tex<T>& t, sampler s, "
			          "uint sw, component c, Ts... params) METAL_CONST_ARG(c)");
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
			break;

		case SPVFuncImplGatherCompareSwizzle:
			statement("// Wrapper function that swizzles depth texture gathers.");
			statement("template<typename T, template<typename, access = access::sample, typename = void> class Tex, "
			          "typename... Ts>");
			statement("inline vec<T, 4> spvGatherCompareSwizzle(const thread Tex<T>& t, sampler "
			          "s, uint sw, Ts... params) ");
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

		case SPVFuncImplReflectScalar:
			// Metal does not support scalar versions of these functions.
			statement("template<typename T>");
			statement("inline T spvReflect(T i, T n)");
			begin_scope();
			statement("return i - T(2) * i * n * n;");
			end_scope();
			statement("");
			break;

		case SPVFuncImplRefractScalar:
			// Metal does not support scalar versions of these functions.
			statement("template<typename T>");
			statement("inline T spvRefract(T i, T n, T eta)");
			begin_scope();
			statement("T NoI = n * i;");
			statement("T NoI2 = NoI * NoI;");
			statement("T k = T(1) - eta * eta * (T(1) - NoI2);");
			statement("if (k < T(0))");
			begin_scope();
			statement("return T(0);");
			end_scope();
			statement("else");
			begin_scope();
			statement("return eta * i - (eta * NoI + sqrt(k)) * n;");
			end_scope();
			end_scope();
			statement("");
			break;

		case SPVFuncImplFaceForwardScalar:
			// Metal does not support scalar versions of these functions.
			statement("template<typename T>");
			statement("inline T spvFaceForward(T n, T i, T nref)");
			begin_scope();
			statement("return i * nref < T(0) ? n : -n;");
			end_scope();
			statement("");
			break;

		case SPVFuncImplChromaReconstructNearest2Plane:
			statement("template<typename T, typename... LodOptions>");
			statement("inline vec<T, 4> spvChromaReconstructNearest(texture2d<T> plane0, texture2d<T> plane1, sampler "
			          "samp, float2 coord, LodOptions... options)");
			begin_scope();
			statement("vec<T, 4> ycbcr = vec<T, 4>(0, 0, 0, 1);");
			statement("ycbcr.g = plane0.sample(samp, coord, spvForward<LodOptions>(options)...).r;");
			statement("ycbcr.br = plane1.sample(samp, coord, spvForward<LodOptions>(options)...).rg;");
			statement("return ycbcr;");
			end_scope();
			statement("");
			break;

		case SPVFuncImplChromaReconstructNearest3Plane:
			statement("template<typename T, typename... LodOptions>");
			statement("inline vec<T, 4> spvChromaReconstructNearest(texture2d<T> plane0, texture2d<T> plane1, "
			          "texture2d<T> plane2, sampler samp, float2 coord, LodOptions... options)");
			begin_scope();
			statement("vec<T, 4> ycbcr = vec<T, 4>(0, 0, 0, 1);");
			statement("ycbcr.g = plane0.sample(samp, coord, spvForward<LodOptions>(options)...).r;");
			statement("ycbcr.b = plane1.sample(samp, coord, spvForward<LodOptions>(options)...).r;");
			statement("ycbcr.r = plane2.sample(samp, coord, spvForward<LodOptions>(options)...).r;");
			statement("return ycbcr;");
			end_scope();
			statement("");
			break;

		case SPVFuncImplChromaReconstructLinear422CositedEven2Plane:
			statement("template<typename T, typename... LodOptions>");
			statement("inline vec<T, 4> spvChromaReconstructLinear422CositedEven(texture2d<T> plane0, texture2d<T> "
			          "plane1, sampler samp, float2 coord, LodOptions... options)");
			begin_scope();
			statement("vec<T, 4> ycbcr = vec<T, 4>(0, 0, 0, 1);");
			statement("ycbcr.g = plane0.sample(samp, coord, spvForward<LodOptions>(options)...).r;");
			statement("if (fract(coord.x * plane1.get_width()) != 0.0)");
			begin_scope();
			statement("ycbcr.br = vec<T, 2>(mix(plane1.sample(samp, coord, spvForward<LodOptions>(options)...), "
			          "plane1.sample(samp, coord, spvForward<LodOptions>(options)..., int2(1, 0)), 0.5).rg);");
			end_scope();
			statement("else");
			begin_scope();
			statement("ycbcr.br = plane1.sample(samp, coord, spvForward<LodOptions>(options)...).rg;");
			end_scope();
			statement("return ycbcr;");
			end_scope();
			statement("");
			break;

		case SPVFuncImplChromaReconstructLinear422CositedEven3Plane:
			statement("template<typename T, typename... LodOptions>");
			statement("inline vec<T, 4> spvChromaReconstructLinear422CositedEven(texture2d<T> plane0, texture2d<T> "
			          "plane1, texture2d<T> plane2, sampler samp, float2 coord, LodOptions... options)");
			begin_scope();
			statement("vec<T, 4> ycbcr = vec<T, 4>(0, 0, 0, 1);");
			statement("ycbcr.g = plane0.sample(samp, coord, spvForward<LodOptions>(options)...).r;");
			statement("if (fract(coord.x * plane1.get_width()) != 0.0)");
			begin_scope();
			statement("ycbcr.b = T(mix(plane1.sample(samp, coord, spvForward<LodOptions>(options)...), "
			          "plane1.sample(samp, coord, spvForward<LodOptions>(options)..., int2(1, 0)), 0.5).r);");
			statement("ycbcr.r = T(mix(plane2.sample(samp, coord, spvForward<LodOptions>(options)...), "
			          "plane2.sample(samp, coord, spvForward<LodOptions>(options)..., int2(1, 0)), 0.5).r);");
			end_scope();
			statement("else");
			begin_scope();
			statement("ycbcr.b = plane1.sample(samp, coord, spvForward<LodOptions>(options)...).r;");
			statement("ycbcr.r = plane2.sample(samp, coord, spvForward<LodOptions>(options)...).r;");
			end_scope();
			statement("return ycbcr;");
			end_scope();
			statement("");
			break;

		case SPVFuncImplChromaReconstructLinear422Midpoint2Plane:
			statement("template<typename T, typename... LodOptions>");
			statement("inline vec<T, 4> spvChromaReconstructLinear422Midpoint(texture2d<T> plane0, texture2d<T> "
			          "plane1, sampler samp, float2 coord, LodOptions... options)");
			begin_scope();
			statement("vec<T, 4> ycbcr = vec<T, 4>(0, 0, 0, 1);");
			statement("ycbcr.g = plane0.sample(samp, coord, spvForward<LodOptions>(options)...).r;");
			statement("int2 offs = int2(fract(coord.x * plane1.get_width()) != 0.0 ? 1 : -1, 0);");
			statement("ycbcr.br = vec<T, 2>(mix(plane1.sample(samp, coord, spvForward<LodOptions>(options)...), "
			          "plane1.sample(samp, coord, spvForward<LodOptions>(options)..., offs), 0.25).rg);");
			statement("return ycbcr;");
			end_scope();
			statement("");
			break;

		case SPVFuncImplChromaReconstructLinear422Midpoint3Plane:
			statement("template<typename T, typename... LodOptions>");
			statement("inline vec<T, 4> spvChromaReconstructLinear422Midpoint(texture2d<T> plane0, texture2d<T> "
			          "plane1, texture2d<T> plane2, sampler samp, float2 coord, LodOptions... options)");
			begin_scope();
			statement("vec<T, 4> ycbcr = vec<T, 4>(0, 0, 0, 1);");
			statement("ycbcr.g = plane0.sample(samp, coord, spvForward<LodOptions>(options)...).r;");
			statement("int2 offs = int2(fract(coord.x * plane1.get_width()) != 0.0 ? 1 : -1, 0);");
			statement("ycbcr.b = T(mix(plane1.sample(samp, coord, spvForward<LodOptions>(options)...), "
			          "plane1.sample(samp, coord, spvForward<LodOptions>(options)..., offs), 0.25).r);");
			statement("ycbcr.r = T(mix(plane2.sample(samp, coord, spvForward<LodOptions>(options)...), "
			          "plane2.sample(samp, coord, spvForward<LodOptions>(options)..., offs), 0.25).r);");
			statement("return ycbcr;");
			end_scope();
			statement("");
			break;

		case SPVFuncImplChromaReconstructLinear420XCositedEvenYCositedEven2Plane:
			statement("template<typename T, typename... LodOptions>");
			statement("inline vec<T, 4> spvChromaReconstructLinear420XCositedEvenYCositedEven(texture2d<T> plane0, "
			          "texture2d<T> plane1, sampler samp, float2 coord, LodOptions... options)");
			begin_scope();
			statement("vec<T, 4> ycbcr = vec<T, 4>(0, 0, 0, 1);");
			statement("ycbcr.g = plane0.sample(samp, coord, spvForward<LodOptions>(options)...).r;");
			statement("float2 ab = fract(round(coord * float2(plane0.get_width(), plane0.get_height())) * 0.5);");
			statement("ycbcr.br = vec<T, 2>(mix(mix(plane1.sample(samp, coord, spvForward<LodOptions>(options)...), "
			          "plane1.sample(samp, coord, spvForward<LodOptions>(options)..., int2(1, 0)), ab.x), "
			          "mix(plane1.sample(samp, coord, spvForward<LodOptions>(options)..., int2(0, 1)), "
			          "plane1.sample(samp, coord, spvForward<LodOptions>(options)..., int2(1, 1)), ab.x), ab.y).rg);");
			statement("return ycbcr;");
			end_scope();
			statement("");
			break;

		case SPVFuncImplChromaReconstructLinear420XCositedEvenYCositedEven3Plane:
			statement("template<typename T, typename... LodOptions>");
			statement("inline vec<T, 4> spvChromaReconstructLinear420XCositedEvenYCositedEven(texture2d<T> plane0, "
			          "texture2d<T> plane1, texture2d<T> plane2, sampler samp, float2 coord, LodOptions... options)");
			begin_scope();
			statement("vec<T, 4> ycbcr = vec<T, 4>(0, 0, 0, 1);");
			statement("ycbcr.g = plane0.sample(samp, coord, spvForward<LodOptions>(options)...).r;");
			statement("float2 ab = fract(round(coord * float2(plane0.get_width(), plane0.get_height())) * 0.5);");
			statement("ycbcr.b = T(mix(mix(plane1.sample(samp, coord, spvForward<LodOptions>(options)...), "
			          "plane1.sample(samp, coord, spvForward<LodOptions>(options)..., int2(1, 0)), ab.x), "
			          "mix(plane1.sample(samp, coord, spvForward<LodOptions>(options)..., int2(0, 1)), "
			          "plane1.sample(samp, coord, spvForward<LodOptions>(options)..., int2(1, 1)), ab.x), ab.y).r);");
			statement("ycbcr.r = T(mix(mix(plane2.sample(samp, coord, spvForward<LodOptions>(options)...), "
			          "plane2.sample(samp, coord, spvForward<LodOptions>(options)..., int2(1, 0)), ab.x), "
			          "mix(plane2.sample(samp, coord, spvForward<LodOptions>(options)..., int2(0, 1)), "
			          "plane2.sample(samp, coord, spvForward<LodOptions>(options)..., int2(1, 1)), ab.x), ab.y).r);");
			statement("return ycbcr;");
			end_scope();
			statement("");
			break;

		case SPVFuncImplChromaReconstructLinear420XMidpointYCositedEven2Plane:
			statement("template<typename T, typename... LodOptions>");
			statement("inline vec<T, 4> spvChromaReconstructLinear420XMidpointYCositedEven(texture2d<T> plane0, "
			          "texture2d<T> plane1, sampler samp, float2 coord, LodOptions... options)");
			begin_scope();
			statement("vec<T, 4> ycbcr = vec<T, 4>(0, 0, 0, 1);");
			statement("ycbcr.g = plane0.sample(samp, coord, spvForward<LodOptions>(options)...).r;");
			statement("float2 ab = fract((round(coord * float2(plane0.get_width(), plane0.get_height())) - float2(0.5, "
			          "0)) * 0.5);");
			statement("ycbcr.br = vec<T, 2>(mix(mix(plane1.sample(samp, coord, spvForward<LodOptions>(options)...), "
			          "plane1.sample(samp, coord, spvForward<LodOptions>(options)..., int2(1, 0)), ab.x), "
			          "mix(plane1.sample(samp, coord, spvForward<LodOptions>(options)..., int2(0, 1)), "
			          "plane1.sample(samp, coord, spvForward<LodOptions>(options)..., int2(1, 1)), ab.x), ab.y).rg);");
			statement("return ycbcr;");
			end_scope();
			statement("");
			break;

		case SPVFuncImplChromaReconstructLinear420XMidpointYCositedEven3Plane:
			statement("template<typename T, typename... LodOptions>");
			statement("inline vec<T, 4> spvChromaReconstructLinear420XMidpointYCositedEven(texture2d<T> plane0, "
			          "texture2d<T> plane1, texture2d<T> plane2, sampler samp, float2 coord, LodOptions... options)");
			begin_scope();
			statement("vec<T, 4> ycbcr = vec<T, 4>(0, 0, 0, 1);");
			statement("ycbcr.g = plane0.sample(samp, coord, spvForward<LodOptions>(options)...).r;");
			statement("float2 ab = fract((round(coord * float2(plane0.get_width(), plane0.get_height())) - float2(0.5, "
			          "0)) * 0.5);");
			statement("ycbcr.b = T(mix(mix(plane1.sample(samp, coord, spvForward<LodOptions>(options)...), "
			          "plane1.sample(samp, coord, spvForward<LodOptions>(options)..., int2(1, 0)), ab.x), "
			          "mix(plane1.sample(samp, coord, spvForward<LodOptions>(options)..., int2(0, 1)), "
			          "plane1.sample(samp, coord, spvForward<LodOptions>(options)..., int2(1, 1)), ab.x), ab.y).r);");
			statement("ycbcr.r = T(mix(mix(plane2.sample(samp, coord, spvForward<LodOptions>(options)...), "
			          "plane2.sample(samp, coord, spvForward<LodOptions>(options)..., int2(1, 0)), ab.x), "
			          "mix(plane2.sample(samp, coord, spvForward<LodOptions>(options)..., int2(0, 1)), "
			          "plane2.sample(samp, coord, spvForward<LodOptions>(options)..., int2(1, 1)), ab.x), ab.y).r);");
			statement("return ycbcr;");
			end_scope();
			statement("");
			break;

		case SPVFuncImplChromaReconstructLinear420XCositedEvenYMidpoint2Plane:
			statement("template<typename T, typename... LodOptions>");
			statement("inline vec<T, 4> spvChromaReconstructLinear420XCositedEvenYMidpoint(texture2d<T> plane0, "
			          "texture2d<T> plane1, sampler samp, float2 coord, LodOptions... options)");
			begin_scope();
			statement("vec<T, 4> ycbcr = vec<T, 4>(0, 0, 0, 1);");
			statement("ycbcr.g = plane0.sample(samp, coord, spvForward<LodOptions>(options)...).r;");
			statement("float2 ab = fract((round(coord * float2(plane0.get_width(), plane0.get_height())) - float2(0, "
			          "0.5)) * 0.5);");
			statement("ycbcr.br = vec<T, 2>(mix(mix(plane1.sample(samp, coord, spvForward<LodOptions>(options)...), "
			          "plane1.sample(samp, coord, spvForward<LodOptions>(options)..., int2(1, 0)), ab.x), "
			          "mix(plane1.sample(samp, coord, spvForward<LodOptions>(options)..., int2(0, 1)), "
			          "plane1.sample(samp, coord, spvForward<LodOptions>(options)..., int2(1, 1)), ab.x), ab.y).rg);");
			statement("return ycbcr;");
			end_scope();
			statement("");
			break;

		case SPVFuncImplChromaReconstructLinear420XCositedEvenYMidpoint3Plane:
			statement("template<typename T, typename... LodOptions>");
			statement("inline vec<T, 4> spvChromaReconstructLinear420XCositedEvenYMidpoint(texture2d<T> plane0, "
			          "texture2d<T> plane1, texture2d<T> plane2, sampler samp, float2 coord, LodOptions... options)");
			begin_scope();
			statement("vec<T, 4> ycbcr = vec<T, 4>(0, 0, 0, 1);");
			statement("ycbcr.g = plane0.sample(samp, coord, spvForward<LodOptions>(options)...).r;");
			statement("float2 ab = fract((round(coord * float2(plane0.get_width(), plane0.get_height())) - float2(0, "
			          "0.5)) * 0.5);");
			statement("ycbcr.b = T(mix(mix(plane1.sample(samp, coord, spvForward<LodOptions>(options)...), "
			          "plane1.sample(samp, coord, spvForward<LodOptions>(options)..., int2(1, 0)), ab.x), "
			          "mix(plane1.sample(samp, coord, spvForward<LodOptions>(options)..., int2(0, 1)), "
			          "plane1.sample(samp, coord, spvForward<LodOptions>(options)..., int2(1, 1)), ab.x), ab.y).r);");
			statement("ycbcr.r = T(mix(mix(plane2.sample(samp, coord, spvForward<LodOptions>(options)...), "
			          "plane2.sample(samp, coord, spvForward<LodOptions>(options)..., int2(1, 0)), ab.x), "
			          "mix(plane2.sample(samp, coord, spvForward<LodOptions>(options)..., int2(0, 1)), "
			          "plane2.sample(samp, coord, spvForward<LodOptions>(options)..., int2(1, 1)), ab.x), ab.y).r);");
			statement("return ycbcr;");
			end_scope();
			statement("");
			break;

		case SPVFuncImplChromaReconstructLinear420XMidpointYMidpoint2Plane:
			statement("template<typename T, typename... LodOptions>");
			statement("inline vec<T, 4> spvChromaReconstructLinear420XMidpointYMidpoint(texture2d<T> plane0, "
			          "texture2d<T> plane1, sampler samp, float2 coord, LodOptions... options)");
			begin_scope();
			statement("vec<T, 4> ycbcr = vec<T, 4>(0, 0, 0, 1);");
			statement("ycbcr.g = plane0.sample(samp, coord, spvForward<LodOptions>(options)...).r;");
			statement("float2 ab = fract((round(coord * float2(plane0.get_width(), plane0.get_height())) - float2(0.5, "
			          "0.5)) * 0.5);");
			statement("ycbcr.br = vec<T, 2>(mix(mix(plane1.sample(samp, coord, spvForward<LodOptions>(options)...), "
			          "plane1.sample(samp, coord, spvForward<LodOptions>(options)..., int2(1, 0)), ab.x), "
			          "mix(plane1.sample(samp, coord, spvForward<LodOptions>(options)..., int2(0, 1)), "
			          "plane1.sample(samp, coord, spvForward<LodOptions>(options)..., int2(1, 1)), ab.x), ab.y).rg);");
			statement("return ycbcr;");
			end_scope();
			statement("");
			break;

		case SPVFuncImplChromaReconstructLinear420XMidpointYMidpoint3Plane:
			statement("template<typename T, typename... LodOptions>");
			statement("inline vec<T, 4> spvChromaReconstructLinear420XMidpointYMidpoint(texture2d<T> plane0, "
			          "texture2d<T> plane1, texture2d<T> plane2, sampler samp, float2 coord, LodOptions... options)");
			begin_scope();
			statement("vec<T, 4> ycbcr = vec<T, 4>(0, 0, 0, 1);");
			statement("ycbcr.g = plane0.sample(samp, coord, spvForward<LodOptions>(options)...).r;");
			statement("float2 ab = fract((round(coord * float2(plane0.get_width(), plane0.get_height())) - float2(0.5, "
			          "0.5)) * 0.5);");
			statement("ycbcr.b = T(mix(mix(plane1.sample(samp, coord, spvForward<LodOptions>(options)...), "
			          "plane1.sample(samp, coord, spvForward<LodOptions>(options)..., int2(1, 0)), ab.x), "
			          "mix(plane1.sample(samp, coord, spvForward<LodOptions>(options)..., int2(0, 1)), "
			          "plane1.sample(samp, coord, spvForward<LodOptions>(options)..., int2(1, 1)), ab.x), ab.y).r);");
			statement("ycbcr.r = T(mix(mix(plane2.sample(samp, coord, spvForward<LodOptions>(options)...), "
			          "plane2.sample(samp, coord, spvForward<LodOptions>(options)..., int2(1, 0)), ab.x), "
			          "mix(plane2.sample(samp, coord, spvForward<LodOptions>(options)..., int2(0, 1)), "
			          "plane2.sample(samp, coord, spvForward<LodOptions>(options)..., int2(1, 1)), ab.x), ab.y).r);");
			statement("return ycbcr;");
			end_scope();
			statement("");
			break;

		case SPVFuncImplExpandITUFullRange:
			statement("template<typename T>");
			statement("inline vec<T, 4> spvExpandITUFullRange(vec<T, 4> ycbcr, int n)");
			begin_scope();
			statement("ycbcr.br -= exp2(T(n-1))/(exp2(T(n))-1);");
			statement("return ycbcr;");
			end_scope();
			statement("");
			break;

		case SPVFuncImplExpandITUNarrowRange:
			statement("template<typename T>");
			statement("inline vec<T, 4> spvExpandITUNarrowRange(vec<T, 4> ycbcr, int n)");
			begin_scope();
			statement("ycbcr.g = (ycbcr.g * (exp2(T(n)) - 1) - ldexp(T(16), n - 8))/ldexp(T(219), n - 8);");
			statement("ycbcr.br = (ycbcr.br * (exp2(T(n)) - 1) - ldexp(T(128), n - 8))/ldexp(T(224), n - 8);");
			statement("return ycbcr;");
			end_scope();
			statement("");
			break;

		case SPVFuncImplConvertYCbCrBT709:
			statement("// cf. Khronos Data Format Specification, section 15.1.1");
			statement("constant float3x3 spvBT709Factors = {{1, 1, 1}, {0, -0.13397432/0.7152, 1.8556}, {1.5748, "
			          "-0.33480248/0.7152, 0}};");
			statement("");
			statement("template<typename T>");
			statement("inline vec<T, 4> spvConvertYCbCrBT709(vec<T, 4> ycbcr)");
			begin_scope();
			statement("vec<T, 4> rgba;");
			statement("rgba.rgb = vec<T, 3>(spvBT709Factors * ycbcr.gbr);");
			statement("rgba.a = ycbcr.a;");
			statement("return rgba;");
			end_scope();
			statement("");
			break;

		case SPVFuncImplConvertYCbCrBT601:
			statement("// cf. Khronos Data Format Specification, section 15.1.2");
			statement("constant float3x3 spvBT601Factors = {{1, 1, 1}, {0, -0.202008/0.587, 1.772}, {1.402, "
			          "-0.419198/0.587, 0}};");
			statement("");
			statement("template<typename T>");
			statement("inline vec<T, 4> spvConvertYCbCrBT601(vec<T, 4> ycbcr)");
			begin_scope();
			statement("vec<T, 4> rgba;");
			statement("rgba.rgb = vec<T, 3>(spvBT601Factors * ycbcr.gbr);");
			statement("rgba.a = ycbcr.a;");
			statement("return rgba;");
			end_scope();
			statement("");
			break;

		case SPVFuncImplConvertYCbCrBT2020:
			statement("// cf. Khronos Data Format Specification, section 15.1.3");
			statement("constant float3x3 spvBT2020Factors = {{1, 1, 1}, {0, -0.11156702/0.6780, 1.8814}, {1.4746, "
			          "-0.38737742/0.6780, 0}};");
			statement("");
			statement("template<typename T>");
			statement("inline vec<T, 4> spvConvertYCbCrBT2020(vec<T, 4> ycbcr)");
			begin_scope();
			statement("vec<T, 4> rgba;");
			statement("rgba.rgb = vec<T, 3>(spvBT2020Factors * ycbcr.gbr);");
			statement("rgba.a = ycbcr.a;");
			statement("return rgba;");
			end_scope();
			statement("");
			break;

		case SPVFuncImplDynamicImageSampler:
			statement("enum class spvFormatResolution");
			begin_scope();
			statement("_444 = 0,");
			statement("_422,");
			statement("_420");
			end_scope_decl();
			statement("");
			statement("enum class spvChromaFilter");
			begin_scope();
			statement("nearest = 0,");
			statement("linear");
			end_scope_decl();
			statement("");
			statement("enum class spvXChromaLocation");
			begin_scope();
			statement("cosited_even = 0,");
			statement("midpoint");
			end_scope_decl();
			statement("");
			statement("enum class spvYChromaLocation");
			begin_scope();
			statement("cosited_even = 0,");
			statement("midpoint");
			end_scope_decl();
			statement("");
			statement("enum class spvYCbCrModelConversion");
			begin_scope();
			statement("rgb_identity = 0,");
			statement("ycbcr_identity,");
			statement("ycbcr_bt_709,");
			statement("ycbcr_bt_601,");
			statement("ycbcr_bt_2020");
			end_scope_decl();
			statement("");
			statement("enum class spvYCbCrRange");
			begin_scope();
			statement("itu_full = 0,");
			statement("itu_narrow");
			end_scope_decl();
			statement("");
			statement("struct spvComponentBits");
			begin_scope();
			statement("constexpr explicit spvComponentBits(int v) thread : value(v) {}");
			statement("uchar value : 6;");
			end_scope_decl();
			statement("// A class corresponding to metal::sampler which holds sampler");
			statement("// Y'CbCr conversion info.");
			statement("struct spvYCbCrSampler");
			begin_scope();
			statement("constexpr spvYCbCrSampler() thread : val(build()) {}");
			statement("template<typename... Ts>");
			statement("constexpr spvYCbCrSampler(Ts... t) thread : val(build(t...)) {}");
			statement("constexpr spvYCbCrSampler(const thread spvYCbCrSampler& s) thread = default;");
			statement("");
			statement("spvFormatResolution get_resolution() const thread");
			begin_scope();
			statement("return spvFormatResolution((val & resolution_mask) >> resolution_base);");
			end_scope();
			statement("spvChromaFilter get_chroma_filter() const thread");
			begin_scope();
			statement("return spvChromaFilter((val & chroma_filter_mask) >> chroma_filter_base);");
			end_scope();
			statement("spvXChromaLocation get_x_chroma_offset() const thread");
			begin_scope();
			statement("return spvXChromaLocation((val & x_chroma_off_mask) >> x_chroma_off_base);");
			end_scope();
			statement("spvYChromaLocation get_y_chroma_offset() const thread");
			begin_scope();
			statement("return spvYChromaLocation((val & y_chroma_off_mask) >> y_chroma_off_base);");
			end_scope();
			statement("spvYCbCrModelConversion get_ycbcr_model() const thread");
			begin_scope();
			statement("return spvYCbCrModelConversion((val & ycbcr_model_mask) >> ycbcr_model_base);");
			end_scope();
			statement("spvYCbCrRange get_ycbcr_range() const thread");
			begin_scope();
			statement("return spvYCbCrRange((val & ycbcr_range_mask) >> ycbcr_range_base);");
			end_scope();
			statement("int get_bpc() const thread { return (val & bpc_mask) >> bpc_base; }");
			statement("");
			statement("private:");
			statement("ushort val;");
			statement("");
			statement("constexpr static constant ushort resolution_bits = 2;");
			statement("constexpr static constant ushort chroma_filter_bits = 2;");
			statement("constexpr static constant ushort x_chroma_off_bit = 1;");
			statement("constexpr static constant ushort y_chroma_off_bit = 1;");
			statement("constexpr static constant ushort ycbcr_model_bits = 3;");
			statement("constexpr static constant ushort ycbcr_range_bit = 1;");
			statement("constexpr static constant ushort bpc_bits = 6;");
			statement("");
			statement("constexpr static constant ushort resolution_base = 0;");
			statement("constexpr static constant ushort chroma_filter_base = 2;");
			statement("constexpr static constant ushort x_chroma_off_base = 4;");
			statement("constexpr static constant ushort y_chroma_off_base = 5;");
			statement("constexpr static constant ushort ycbcr_model_base = 6;");
			statement("constexpr static constant ushort ycbcr_range_base = 9;");
			statement("constexpr static constant ushort bpc_base = 10;");
			statement("");
			statement(
			    "constexpr static constant ushort resolution_mask = ((1 << resolution_bits) - 1) << resolution_base;");
			statement("constexpr static constant ushort chroma_filter_mask = ((1 << chroma_filter_bits) - 1) << "
			          "chroma_filter_base;");
			statement("constexpr static constant ushort x_chroma_off_mask = ((1 << x_chroma_off_bit) - 1) << "
			          "x_chroma_off_base;");
			statement("constexpr static constant ushort y_chroma_off_mask = ((1 << y_chroma_off_bit) - 1) << "
			          "y_chroma_off_base;");
			statement("constexpr static constant ushort ycbcr_model_mask = ((1 << ycbcr_model_bits) - 1) << "
			          "ycbcr_model_base;");
			statement("constexpr static constant ushort ycbcr_range_mask = ((1 << ycbcr_range_bit) - 1) << "
			          "ycbcr_range_base;");
			statement("constexpr static constant ushort bpc_mask = ((1 << bpc_bits) - 1) << bpc_base;");
			statement("");
			statement("static constexpr ushort build()");
			begin_scope();
			statement("return 0;");
			end_scope();
			statement("");
			statement("template<typename... Ts>");
			statement("static constexpr ushort build(spvFormatResolution res, Ts... t)");
			begin_scope();
			statement("return (ushort(res) << resolution_base) | (build(t...) & ~resolution_mask);");
			end_scope();
			statement("");
			statement("template<typename... Ts>");
			statement("static constexpr ushort build(spvChromaFilter filt, Ts... t)");
			begin_scope();
			statement("return (ushort(filt) << chroma_filter_base) | (build(t...) & ~chroma_filter_mask);");
			end_scope();
			statement("");
			statement("template<typename... Ts>");
			statement("static constexpr ushort build(spvXChromaLocation loc, Ts... t)");
			begin_scope();
			statement("return (ushort(loc) << x_chroma_off_base) | (build(t...) & ~x_chroma_off_mask);");
			end_scope();
			statement("");
			statement("template<typename... Ts>");
			statement("static constexpr ushort build(spvYChromaLocation loc, Ts... t)");
			begin_scope();
			statement("return (ushort(loc) << y_chroma_off_base) | (build(t...) & ~y_chroma_off_mask);");
			end_scope();
			statement("");
			statement("template<typename... Ts>");
			statement("static constexpr ushort build(spvYCbCrModelConversion model, Ts... t)");
			begin_scope();
			statement("return (ushort(model) << ycbcr_model_base) | (build(t...) & ~ycbcr_model_mask);");
			end_scope();
			statement("");
			statement("template<typename... Ts>");
			statement("static constexpr ushort build(spvYCbCrRange range, Ts... t)");
			begin_scope();
			statement("return (ushort(range) << ycbcr_range_base) | (build(t...) & ~ycbcr_range_mask);");
			end_scope();
			statement("");
			statement("template<typename... Ts>");
			statement("static constexpr ushort build(spvComponentBits bpc, Ts... t)");
			begin_scope();
			statement("return (ushort(bpc.value) << bpc_base) | (build(t...) & ~bpc_mask);");
			end_scope();
			end_scope_decl();
			statement("");
			statement("// A class which can hold up to three textures and a sampler, including");
			statement("// Y'CbCr conversion info, used to pass combined image-samplers");
			statement("// dynamically to functions.");
			statement("template<typename T>");
			statement("struct spvDynamicImageSampler");
			begin_scope();
			statement("texture2d<T> plane0;");
			statement("texture2d<T> plane1;");
			statement("texture2d<T> plane2;");
			statement("sampler samp;");
			statement("spvYCbCrSampler ycbcr_samp;");
			statement("uint swizzle = 0;");
			statement("");
			if (msl_options.swizzle_texture_samples)
			{
				statement("constexpr spvDynamicImageSampler(texture2d<T> tex, sampler samp, uint sw) thread :");
				statement("    plane0(tex), samp(samp), swizzle(sw) {}");
			}
			else
			{
				statement("constexpr spvDynamicImageSampler(texture2d<T> tex, sampler samp) thread :");
				statement("    plane0(tex), samp(samp) {}");
			}
			statement("constexpr spvDynamicImageSampler(texture2d<T> tex, sampler samp, spvYCbCrSampler ycbcr_samp, "
			          "uint sw) thread :");
			statement("    plane0(tex), samp(samp), ycbcr_samp(ycbcr_samp), swizzle(sw) {}");
			statement("constexpr spvDynamicImageSampler(texture2d<T> plane0, texture2d<T> plane1,");
			statement("                                 sampler samp, spvYCbCrSampler ycbcr_samp, uint sw) thread :");
			statement("    plane0(plane0), plane1(plane1), samp(samp), ycbcr_samp(ycbcr_samp), swizzle(sw) {}");
			statement(
			    "constexpr spvDynamicImageSampler(texture2d<T> plane0, texture2d<T> plane1, texture2d<T> plane2,");
			statement("                                 sampler samp, spvYCbCrSampler ycbcr_samp, uint sw) thread :");
			statement("    plane0(plane0), plane1(plane1), plane2(plane2), samp(samp), ycbcr_samp(ycbcr_samp), "
			          "swizzle(sw) {}");
			statement("");
			// XXX This is really hard to follow... I've left comments to make it a bit easier.
			statement("template<typename... LodOptions>");
			statement("vec<T, 4> do_sample(float2 coord, LodOptions... options) const thread");
			begin_scope();
			statement("if (!is_null_texture(plane1))");
			begin_scope();
			statement("if (ycbcr_samp.get_resolution() == spvFormatResolution::_444 ||");
			statement("    ycbcr_samp.get_chroma_filter() == spvChromaFilter::nearest)");
			begin_scope();
			statement("if (!is_null_texture(plane2))");
			statement("    return spvChromaReconstructNearest(plane0, plane1, plane2, samp, coord,");
			statement("                                       spvForward<LodOptions>(options)...);");
			statement(
			    "return spvChromaReconstructNearest(plane0, plane1, samp, coord, spvForward<LodOptions>(options)...);");
			end_scope(); // if (resolution == 422 || chroma_filter == nearest)
			statement("switch (ycbcr_samp.get_resolution())");
			begin_scope();
			statement("case spvFormatResolution::_444: break;");
			statement("case spvFormatResolution::_422:");
			begin_scope();
			statement("switch (ycbcr_samp.get_x_chroma_offset())");
			begin_scope();
			statement("case spvXChromaLocation::cosited_even:");
			statement("    if (!is_null_texture(plane2))");
			statement("        return spvChromaReconstructLinear422CositedEven(");
			statement("            plane0, plane1, plane2, samp,");
			statement("            coord, spvForward<LodOptions>(options)...);");
			statement("    return spvChromaReconstructLinear422CositedEven(");
			statement("        plane0, plane1, samp, coord,");
			statement("        spvForward<LodOptions>(options)...);");
			statement("case spvXChromaLocation::midpoint:");
			statement("    if (!is_null_texture(plane2))");
			statement("        return spvChromaReconstructLinear422Midpoint(");
			statement("            plane0, plane1, plane2, samp,");
			statement("            coord, spvForward<LodOptions>(options)...);");
			statement("    return spvChromaReconstructLinear422Midpoint(");
			statement("        plane0, plane1, samp, coord,");
			statement("        spvForward<LodOptions>(options)...);");
			end_scope(); // switch (x_chroma_offset)
			end_scope(); // case 422:
			statement("case spvFormatResolution::_420:");
			begin_scope();
			statement("switch (ycbcr_samp.get_x_chroma_offset())");
			begin_scope();
			statement("case spvXChromaLocation::cosited_even:");
			begin_scope();
			statement("switch (ycbcr_samp.get_y_chroma_offset())");
			begin_scope();
			statement("case spvYChromaLocation::cosited_even:");
			statement("    if (!is_null_texture(plane2))");
			statement("        return spvChromaReconstructLinear420XCositedEvenYCositedEven(");
			statement("            plane0, plane1, plane2, samp,");
			statement("            coord, spvForward<LodOptions>(options)...);");
			statement("    return spvChromaReconstructLinear420XCositedEvenYCositedEven(");
			statement("        plane0, plane1, samp, coord,");
			statement("        spvForward<LodOptions>(options)...);");
			statement("case spvYChromaLocation::midpoint:");
			statement("    if (!is_null_texture(plane2))");
			statement("        return spvChromaReconstructLinear420XCositedEvenYMidpoint(");
			statement("            plane0, plane1, plane2, samp,");
			statement("            coord, spvForward<LodOptions>(options)...);");
			statement("    return spvChromaReconstructLinear420XCositedEvenYMidpoint(");
			statement("        plane0, plane1, samp, coord,");
			statement("        spvForward<LodOptions>(options)...);");
			end_scope(); // switch (y_chroma_offset)
			end_scope(); // case x::cosited_even:
			statement("case spvXChromaLocation::midpoint:");
			begin_scope();
			statement("switch (ycbcr_samp.get_y_chroma_offset())");
			begin_scope();
			statement("case spvYChromaLocation::cosited_even:");
			statement("    if (!is_null_texture(plane2))");
			statement("        return spvChromaReconstructLinear420XMidpointYCositedEven(");
			statement("            plane0, plane1, plane2, samp,");
			statement("            coord, spvForward<LodOptions>(options)...);");
			statement("    return spvChromaReconstructLinear420XMidpointYCositedEven(");
			statement("        plane0, plane1, samp, coord,");
			statement("        spvForward<LodOptions>(options)...);");
			statement("case spvYChromaLocation::midpoint:");
			statement("    if (!is_null_texture(plane2))");
			statement("        return spvChromaReconstructLinear420XMidpointYMidpoint(");
			statement("            plane0, plane1, plane2, samp,");
			statement("            coord, spvForward<LodOptions>(options)...);");
			statement("    return spvChromaReconstructLinear420XMidpointYMidpoint(");
			statement("        plane0, plane1, samp, coord,");
			statement("        spvForward<LodOptions>(options)...);");
			end_scope(); // switch (y_chroma_offset)
			end_scope(); // case x::midpoint
			end_scope(); // switch (x_chroma_offset)
			end_scope(); // case 420:
			end_scope(); // switch (resolution)
			end_scope(); // if (multiplanar)
			statement("return plane0.sample(samp, coord, spvForward<LodOptions>(options)...);");
			end_scope(); // do_sample()
			statement("template <typename... LodOptions>");
			statement("vec<T, 4> sample(float2 coord, LodOptions... options) const thread");
			begin_scope();
			statement(
			    "vec<T, 4> s = spvTextureSwizzle(do_sample(coord, spvForward<LodOptions>(options)...), swizzle);");
			statement("if (ycbcr_samp.get_ycbcr_model() == spvYCbCrModelConversion::rgb_identity)");
			statement("    return s;");
			statement("");
			statement("switch (ycbcr_samp.get_ycbcr_range())");
			begin_scope();
			statement("case spvYCbCrRange::itu_full:");
			statement("    s = spvExpandITUFullRange(s, ycbcr_samp.get_bpc());");
			statement("    break;");
			statement("case spvYCbCrRange::itu_narrow:");
			statement("    s = spvExpandITUNarrowRange(s, ycbcr_samp.get_bpc());");
			statement("    break;");
			end_scope();
			statement("");
			statement("switch (ycbcr_samp.get_ycbcr_model())");
			begin_scope();
			statement("case spvYCbCrModelConversion::rgb_identity:"); // Silence Clang warning
			statement("case spvYCbCrModelConversion::ycbcr_identity:");
			statement("    return s;");
			statement("case spvYCbCrModelConversion::ycbcr_bt_709:");
			statement("    return spvConvertYCbCrBT709(s);");
			statement("case spvYCbCrModelConversion::ycbcr_bt_601:");
			statement("    return spvConvertYCbCrBT601(s);");
			statement("case spvYCbCrModelConversion::ycbcr_bt_2020:");
			statement("    return spvConvertYCbCrBT2020(s);");
			end_scope();
			end_scope();
			statement("");
			// Sampler Y'CbCr conversion forbids offsets.
			statement("vec<T, 4> sample(float2 coord, int2 offset) const thread");
			begin_scope();
			if (msl_options.swizzle_texture_samples)
				statement("return spvTextureSwizzle(plane0.sample(samp, coord, offset), swizzle);");
			else
				statement("return plane0.sample(samp, coord, offset);");
			end_scope();
			statement("template<typename lod_options>");
			statement("vec<T, 4> sample(float2 coord, lod_options options, int2 offset) const thread");
			begin_scope();
			if (msl_options.swizzle_texture_samples)
				statement("return spvTextureSwizzle(plane0.sample(samp, coord, options, offset), swizzle);");
			else
				statement("return plane0.sample(samp, coord, options, offset);");
			end_scope();
			statement("#if __HAVE_MIN_LOD_CLAMP__");
			statement("vec<T, 4> sample(float2 coord, bias b, min_lod_clamp min_lod, int2 offset) const thread");
			begin_scope();
			statement("return plane0.sample(samp, coord, b, min_lod, offset);");
			end_scope();
			statement(
			    "vec<T, 4> sample(float2 coord, gradient2d grad, min_lod_clamp min_lod, int2 offset) const thread");
			begin_scope();
			statement("return plane0.sample(samp, coord, grad, min_lod, offset);");
			end_scope();
			statement("#endif");
			statement("");
			// Y'CbCr conversion forbids all operations but sampling.
			statement("vec<T, 4> read(uint2 coord, uint lod = 0) const thread");
			begin_scope();
			statement("return plane0.read(coord, lod);");
			end_scope();
			statement("");
			statement("vec<T, 4> gather(float2 coord, int2 offset = int2(0), component c = component::x) const thread");
			begin_scope();
			if (msl_options.swizzle_texture_samples)
				statement("return spvGatherSwizzle(plane0, samp, swizzle, c, coord, offset);");
			else
				statement("return plane0.gather(samp, coord, offset, c);");
			end_scope();
			end_scope_decl();
			statement("");

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
	bool fully_inlined = ir.ids_for_type[TypeFunction].size() == 1;

	// MSL cannot declare arrays inline (except when declaring a variable), so we must move them out to
	// global constants directly, so we are able to use constants as variable expressions.
	bool emitted = false;

	ir.for_each_typed_id<SPIRConstant>([&](uint32_t, SPIRConstant &c) {
		if (c.specialization)
			return;

		auto &type = this->get<SPIRType>(c.constant_type);
		// Constant arrays of non-primitive types (i.e. matrices) won't link properly into Metal libraries.
		// FIXME: However, hoisting constants to main() means we need to pass down constant arrays to leaf functions if they are used there.
		// If there are multiple functions in the module, drop this case to avoid breaking use cases which do not need to
		// link into Metal libraries. This is hacky.
		if (!type.array.empty() && (!fully_inlined || is_scalar(type) || is_vector(type)))
		{
			auto name = to_name(c.self);
			statement("constant ", variable_decl(type, name), " = ", constant_expression(c), ";");
			emitted = true;
		}
	});

	if (emitted)
		statement("");
}

// Constant arrays of non-primitive types (i.e. matrices) won't link properly into Metal libraries
void CompilerMSL::declare_complex_constant_arrays()
{
	// If we do not have a fully inlined module, we did not opt in to
	// declaring constant arrays of complex types. See CompilerMSL::declare_constant_arrays().
	bool fully_inlined = ir.ids_for_type[TypeFunction].size() == 1;
	if (!fully_inlined)
		return;

	// MSL cannot declare arrays inline (except when declaring a variable), so we must move them out to
	// global constants directly, so we are able to use constants as variable expressions.
	bool emitted = false;

	ir.for_each_typed_id<SPIRConstant>([&](uint32_t, SPIRConstant &c) {
		if (c.specialization)
			return;

		auto &type = this->get<SPIRType>(c.constant_type);
		if (!type.array.empty() && !(is_scalar(type) || is_vector(type)))
		{
			auto name = to_name(c.self);
			statement("", variable_decl(type, name), " = ", constant_expression(c), ";");
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
	ID workgroup_size_id = get_work_group_size_specialization_constants(wg_x, wg_y, wg_z);
	bool emitted = false;

	unordered_set<uint32_t> declared_structs;
	unordered_set<uint32_t> aligned_structs;

	// First, we need to deal with scalar block layout.
	// It is possible that a struct may have to be placed at an alignment which does not match the innate alignment of the struct itself.
	// In that case, if such a case exists for a struct, we must force that all elements of the struct become packed_ types.
	// This makes the struct alignment as small as physically possible.
	// When we actually align the struct later, we can insert padding as necessary to make the packed members behave like normally aligned types.
	ir.for_each_typed_id<SPIRType>([&](uint32_t type_id, const SPIRType &type) {
		if (type.basetype == SPIRType::Struct &&
		    has_extended_decoration(type_id, SPIRVCrossDecorationBufferBlockRepacked))
			mark_scalar_layout_structs(type);
	});

	// Very particular use of the soft loop lock.
	// align_struct may need to create custom types on the fly, but we don't care about
	// these types for purpose of iterating over them in ir.ids_for_type and friends.
	auto loop_lock = ir.create_loop_soft_lock();

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
			TypeID type_id = type.self;

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

				if (has_extended_decoration(type_id, SPIRVCrossDecorationBufferBlockRepacked))
					align_struct(type, aligned_structs);

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

bool CompilerMSL::emit_tessellation_io_load(uint32_t result_type_id, uint32_t id, uint32_t ptr)
{
	auto &ptr_type = expression_type(ptr);
	auto &result_type = get<SPIRType>(result_type_id);
	if (ptr_type.storage != StorageClassInput && ptr_type.storage != StorageClassOutput)
		return false;
	if (ptr_type.storage == StorageClassOutput && get_execution_model() == ExecutionModelTessellationEvaluation)
		return false;

	bool flat_data_type = is_matrix(result_type) || is_array(result_type) || result_type.basetype == SPIRType::Struct;
	if (!flat_data_type)
		return false;

	if (has_decoration(ptr, DecorationPatch))
		return false;

	// Now, we must unflatten a composite type and take care of interleaving array access with gl_in/gl_out.
	// Lots of painful code duplication since we *really* should not unroll these kinds of loads in entry point fixup
	// unless we're forced to do this when the code is emitting inoptimal OpLoads.
	string expr;

	uint32_t interface_index = get_extended_decoration(ptr, SPIRVCrossDecorationInterfaceMemberIndex);
	auto *var = maybe_get_backing_variable(ptr);
	bool ptr_is_io_variable = ir.ids[ptr].get_type() == TypeVariable;

	const auto &iface_type = expression_type(stage_in_ptr_var_id);

	if (result_type.array.size() > 2)
	{
		SPIRV_CROSS_THROW("Cannot load tessellation IO variables with more than 2 dimensions.");
	}
	else if (result_type.array.size() == 2)
	{
		if (!ptr_is_io_variable)
			SPIRV_CROSS_THROW("Loading an array-of-array must be loaded directly from an IO variable.");
		if (interface_index == uint32_t(-1))
			SPIRV_CROSS_THROW("Interface index is unknown. Cannot continue.");
		if (result_type.basetype == SPIRType::Struct || is_matrix(result_type))
			SPIRV_CROSS_THROW("Cannot load array-of-array of composite type in tessellation IO.");

		expr += type_to_glsl(result_type) + "({ ";
		uint32_t num_control_points = to_array_size_literal(result_type, 1);
		uint32_t base_interface_index = interface_index;

		auto &sub_type = get<SPIRType>(result_type.parent_type);

		for (uint32_t i = 0; i < num_control_points; i++)
		{
			expr += type_to_glsl(sub_type) + "({ ";
			interface_index = base_interface_index;
			uint32_t array_size = to_array_size_literal(result_type, 0);
			for (uint32_t j = 0; j < array_size; j++, interface_index++)
			{
				const uint32_t indices[2] = { i, interface_index };

				AccessChainMeta meta;
				expr += access_chain_internal(stage_in_ptr_var_id, indices, 2,
				                              ACCESS_CHAIN_INDEX_IS_LITERAL_BIT | ACCESS_CHAIN_PTR_CHAIN_BIT, &meta);

				if (j + 1 < array_size)
					expr += ", ";
			}
			expr += " })";
			if (i + 1 < num_control_points)
				expr += ", ";
		}
		expr += " })";
	}
	else if (result_type.basetype == SPIRType::Struct)
	{
		bool is_array_of_struct = is_array(result_type);
		if (is_array_of_struct && !ptr_is_io_variable)
			SPIRV_CROSS_THROW("Loading array of struct from IO variable must come directly from IO variable.");

		uint32_t num_control_points = 1;
		if (is_array_of_struct)
		{
			num_control_points = to_array_size_literal(result_type, 0);
			expr += type_to_glsl(result_type) + "({ ";
		}

		auto &struct_type = is_array_of_struct ? get<SPIRType>(result_type.parent_type) : result_type;
		assert(struct_type.array.empty());

		for (uint32_t i = 0; i < num_control_points; i++)
		{
			expr += type_to_glsl(struct_type) + "{ ";
			for (uint32_t j = 0; j < uint32_t(struct_type.member_types.size()); j++)
			{
				// The base interface index is stored per variable for structs.
				if (var)
				{
					interface_index =
					    get_extended_member_decoration(var->self, j, SPIRVCrossDecorationInterfaceMemberIndex);
				}

				if (interface_index == uint32_t(-1))
					SPIRV_CROSS_THROW("Interface index is unknown. Cannot continue.");

				const auto &mbr_type = get<SPIRType>(struct_type.member_types[j]);
				if (is_matrix(mbr_type))
				{
					expr += type_to_glsl(mbr_type) + "(";
					for (uint32_t k = 0; k < mbr_type.columns; k++, interface_index++)
					{
						if (is_array_of_struct)
						{
							const uint32_t indices[2] = { i, interface_index };
							AccessChainMeta meta;
							expr += access_chain_internal(
							    stage_in_ptr_var_id, indices, 2,
							    ACCESS_CHAIN_INDEX_IS_LITERAL_BIT | ACCESS_CHAIN_PTR_CHAIN_BIT, &meta);
						}
						else
							expr += to_expression(ptr) + "." + to_member_name(iface_type, interface_index);

						if (k + 1 < mbr_type.columns)
							expr += ", ";
					}
					expr += ")";
				}
				else if (is_array(mbr_type))
				{
					expr += type_to_glsl(mbr_type) + "({ ";
					uint32_t array_size = to_array_size_literal(mbr_type, 0);
					for (uint32_t k = 0; k < array_size; k++, interface_index++)
					{
						if (is_array_of_struct)
						{
							const uint32_t indices[2] = { i, interface_index };
							AccessChainMeta meta;
							expr += access_chain_internal(
							    stage_in_ptr_var_id, indices, 2,
							    ACCESS_CHAIN_INDEX_IS_LITERAL_BIT | ACCESS_CHAIN_PTR_CHAIN_BIT, &meta);
						}
						else
							expr += to_expression(ptr) + "." + to_member_name(iface_type, interface_index);

						if (k + 1 < array_size)
							expr += ", ";
					}
					expr += " })";
				}
				else
				{
					if (is_array_of_struct)
					{
						const uint32_t indices[2] = { i, interface_index };
						AccessChainMeta meta;
						expr += access_chain_internal(stage_in_ptr_var_id, indices, 2,
						                              ACCESS_CHAIN_INDEX_IS_LITERAL_BIT | ACCESS_CHAIN_PTR_CHAIN_BIT,
						                              &meta);
					}
					else
						expr += to_expression(ptr) + "." + to_member_name(iface_type, interface_index);
				}

				if (j + 1 < struct_type.member_types.size())
					expr += ", ";
			}
			expr += " }";
			if (i + 1 < num_control_points)
				expr += ", ";
		}
		if (is_array_of_struct)
			expr += " })";
	}
	else if (is_matrix(result_type))
	{
		bool is_array_of_matrix = is_array(result_type);
		if (is_array_of_matrix && !ptr_is_io_variable)
			SPIRV_CROSS_THROW("Loading array of matrix from IO variable must come directly from IO variable.");
		if (interface_index == uint32_t(-1))
			SPIRV_CROSS_THROW("Interface index is unknown. Cannot continue.");

		if (is_array_of_matrix)
		{
			// Loading a matrix from each control point.
			uint32_t base_interface_index = interface_index;
			uint32_t num_control_points = to_array_size_literal(result_type, 0);
			expr += type_to_glsl(result_type) + "({ ";

			auto &matrix_type = get_variable_element_type(get<SPIRVariable>(ptr));

			for (uint32_t i = 0; i < num_control_points; i++)
			{
				interface_index = base_interface_index;
				expr += type_to_glsl(matrix_type) + "(";
				for (uint32_t j = 0; j < result_type.columns; j++, interface_index++)
				{
					const uint32_t indices[2] = { i, interface_index };

					AccessChainMeta meta;
					expr +=
					    access_chain_internal(stage_in_ptr_var_id, indices, 2,
					                          ACCESS_CHAIN_INDEX_IS_LITERAL_BIT | ACCESS_CHAIN_PTR_CHAIN_BIT, &meta);
					if (j + 1 < result_type.columns)
						expr += ", ";
				}
				expr += ")";
				if (i + 1 < num_control_points)
					expr += ", ";
			}

			expr += " })";
		}
		else
		{
			expr += type_to_glsl(result_type) + "(";
			for (uint32_t i = 0; i < result_type.columns; i++, interface_index++)
			{
				expr += to_expression(ptr) + "." + to_member_name(iface_type, interface_index);
				if (i + 1 < result_type.columns)
					expr += ", ";
			}
			expr += ")";
		}
	}
	else if (ptr_is_io_variable)
	{
		assert(is_array(result_type));
		assert(result_type.array.size() == 1);
		if (interface_index == uint32_t(-1))
			SPIRV_CROSS_THROW("Interface index is unknown. Cannot continue.");

		// We're loading an array directly from a global variable.
		// This means we're loading one member from each control point.
		expr += type_to_glsl(result_type) + "({ ";
		uint32_t num_control_points = to_array_size_literal(result_type, 0);

		for (uint32_t i = 0; i < num_control_points; i++)
		{
			const uint32_t indices[2] = { i, interface_index };

			AccessChainMeta meta;
			expr += access_chain_internal(stage_in_ptr_var_id, indices, 2,
			                              ACCESS_CHAIN_INDEX_IS_LITERAL_BIT | ACCESS_CHAIN_PTR_CHAIN_BIT, &meta);

			if (i + 1 < num_control_points)
				expr += ", ";
		}
		expr += " })";
	}
	else
	{
		// We're loading an array from a concrete control point.
		assert(is_array(result_type));
		assert(result_type.array.size() == 1);
		if (interface_index == uint32_t(-1))
			SPIRV_CROSS_THROW("Interface index is unknown. Cannot continue.");

		expr += type_to_glsl(result_type) + "({ ";
		uint32_t array_size = to_array_size_literal(result_type, 0);
		for (uint32_t i = 0; i < array_size; i++, interface_index++)
		{
			expr += to_expression(ptr) + "." + to_member_name(iface_type, interface_index);
			if (i + 1 < array_size)
				expr += ", ";
		}
		expr += " })";
	}

	emit_op(result_type_id, id, expr, false);
	register_read(id, ptr, false);
	return true;
}

bool CompilerMSL::emit_tessellation_access_chain(const uint32_t *ops, uint32_t length)
{
	// If this is a per-vertex output, remap it to the I/O array buffer.

	// Any object which did not go through IO flattening shenanigans will go there instead.
	// We will unflatten on-demand instead as needed, but not all possible cases can be supported, especially with arrays.

	auto *var = maybe_get_backing_variable(ops[2]);
	bool patch = false;
	bool flat_data = false;
	bool ptr_is_chain = false;

	if (var)
	{
		patch = has_decoration(ops[2], DecorationPatch) || is_patch_block(get_variable_data_type(*var));

		// Should match strip_array in add_interface_block.
		flat_data = var->storage == StorageClassInput ||
		            (var->storage == StorageClassOutput && get_execution_model() == ExecutionModelTessellationControl);

		// We might have a chained access chain, where
		// we first take the access chain to the control point, and then we chain into a member or something similar.
		// In this case, we need to skip gl_in/gl_out remapping.
		ptr_is_chain = var->self != ID(ops[2]);
	}

	BuiltIn bi_type = BuiltIn(get_decoration(ops[2], DecorationBuiltIn));
	if (var && flat_data && !patch &&
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

		uint32_t first_non_array_index = ptr_is_chain ? 3 : 4;
		VariableID stage_var_id = var->storage == StorageClassInput ? stage_in_ptr_var_id : stage_out_ptr_var_id;
		VariableID ptr = ptr_is_chain ? VariableID(ops[2]) : stage_var_id;
		if (!ptr_is_chain)
		{
			// Index into gl_in/gl_out with first array index.
			indices.push_back(ops[3]);
		}

		auto &result_ptr_type = get<SPIRType>(ops[0]);

		uint32_t const_mbr_id = next_id++;
		uint32_t index = get_extended_decoration(var->self, SPIRVCrossDecorationInterfaceMemberIndex);
		if (var->storage == StorageClassInput || has_decoration(get_variable_element_type(*var).self, DecorationBlock))
		{
			uint32_t i = first_non_array_index;
			auto *type = &get_variable_element_type(*var);
			if (index == uint32_t(-1) && length >= (first_non_array_index + 1))
			{
				// Maybe this is a struct type in the input class, in which case
				// we put it as a decoration on the corresponding member.
				index = get_extended_member_decoration(var->self, get_constant(ops[first_non_array_index]).scalar(),
				                                       SPIRVCrossDecorationInterfaceMemberIndex);
				assert(index != uint32_t(-1));
				i++;
				type = &get<SPIRType>(type->member_types[get_constant(ops[first_non_array_index]).scalar()]);
			}

			// In this case, we're poking into flattened structures and arrays, so now we have to
			// combine the following indices. If we encounter a non-constant index,
			// we're hosed.
			for (; i < length; ++i)
			{
				if (!is_array(*type) && !is_matrix(*type) && type->basetype != SPIRType::Struct)
					break;

				auto *c = maybe_get<SPIRConstant>(ops[i]);
				if (!c || c->specialization)
					SPIRV_CROSS_THROW("Trying to dynamically index into an array interface variable in tessellation. "
					                  "This is currently unsupported.");

				// We're in flattened space, so just increment the member index into IO block.
				// We can only do this once in the current implementation, so either:
				// Struct, Matrix or 1-dimensional array for a control point.
				index += c->scalar();

				if (type->parent_type)
					type = &get<SPIRType>(type->parent_type);
				else if (type->basetype == SPIRType::Struct)
					type = &get<SPIRType>(type->member_types[c->scalar()]);
			}

			if (is_matrix(result_ptr_type) || is_array(result_ptr_type) || result_ptr_type.basetype == SPIRType::Struct)
			{
				// We're not going to emit the actual member name, we let any further OpLoad take care of that.
				// Tag the access chain with the member index we're referencing.
				set_extended_decoration(ops[1], SPIRVCrossDecorationInterfaceMemberIndex, index);
			}
			else
			{
				// Access the appropriate member of gl_in/gl_out.
				set<SPIRConstant>(const_mbr_id, type_id, index, false);
				indices.push_back(const_mbr_id);

				// Append any straggling access chain indices.
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
		string e;

		if (!ptr_is_chain)
		{
			// This is the start of an access chain, use ptr_chain to index into control point array.
			e = access_chain(ptr, indices.data(), uint32_t(indices.size()), result_ptr_type, &meta, true);
		}
		else
		{
			// If we're accessing a struct, we need to use member indices which are based on the IO block,
			// not actual struct type, so we have to use a split access chain here where
			// first path resolves the control point index, i.e. gl_in[index], and second half deals with
			// looking up flattened member name.

			// However, it is possible that we partially accessed a struct,
			// by taking pointer to member inside the control-point array.
			// For this case, we fall back to a natural access chain since we have already dealt with remapping struct members.
			// One way to check this here is if we have 2 implied read expressions.
			// First one is the gl_in/gl_out struct itself, then an index into that array.
			// If we have traversed further, we use a normal access chain formulation.
			auto *ptr_expr = maybe_get<SPIRExpression>(ptr);
			if (ptr_expr && ptr_expr->implied_read_expressions.size() == 2)
			{
				e = join(to_expression(ptr),
				         access_chain_internal(stage_var_id, indices.data(), uint32_t(indices.size()),
				                               ACCESS_CHAIN_CHAIN_ONLY_BIT, &meta));
			}
			else
			{
				e = access_chain_internal(ptr, indices.data(), uint32_t(indices.size()), 0, &meta);
			}
		}

		auto &expr = set<SPIRExpression>(ops[1], move(e), ops[0], should_forward(ops[2]));
		expr.loaded_from = var->self;
		expr.need_transpose = meta.need_transpose;
		expr.access_chain = true;

		// Mark the result as being packed if necessary.
		if (meta.storage_is_packed)
			set_extended_decoration(ops[1], SPIRVCrossDecorationPhysicalTypePacked);
		if (meta.storage_physical_type != 0)
			set_extended_decoration(ops[1], SPIRVCrossDecorationPhysicalTypeID, meta.storage_physical_type);
		if (meta.storage_is_invariant)
			set_decoration(ops[1], DecorationInvariant);

		// If we have some expression dependencies in our access chain, this access chain is technically a forwarded
		// temporary which could be subject to invalidation.
		// Need to assume we're forwarded while calling inherit_expression_depdendencies.
		forwarded_temporaries.insert(ops[1]);
		// The access chain itself is never forced to a temporary, but its dependencies might.
		suppressed_usage_tracking.insert(ops[1]);

		for (uint32_t i = 2; i < length; i++)
		{
			inherit_expression_dependencies(ops[1], ops[i]);
			add_implied_read_expression(expr, ops[i]);
		}

		// If we have no dependencies after all, i.e., all indices in the access chain are immutable temporaries,
		// we're not forwarded after all.
		if (expr.expression_dependencies.empty())
			forwarded_temporaries.erase(ops[1]);

		return true;
	}

	// If this is the inner tessellation level, and we're tessellating triangles,
	// drop the last index. It isn't an array in this case, so we can't have an
	// array reference here. We need to make this ID a variable instead of an
	// expression so we don't try to dereference it as a variable pointer.
	// Don't do this if the index is a constant 1, though. We need to drop stores
	// to that one.
	auto *m = ir.find_meta(var ? var->self : ID(0));
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
	case OpLoad:
	{
		uint32_t id = ops[1];
		uint32_t ptr = ops[2];
		if (is_tessellation_shader())
		{
			if (!emit_tessellation_io_load(ops[0], id, ptr))
				CompilerGLSL::emit_instruction(instruction);
		}
		else
		{
			// Sample mask input for Metal is not an array
			if (BuiltIn(get_decoration(ptr, DecorationBuiltIn)) == BuiltInSampleMask)
				set_decoration(id, DecorationBuiltIn, BuiltInSampleMask);
			CompilerGLSL::emit_instruction(instruction);
		}
		break;
	}

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
	{
		emit_bitfield_insert_op(ops[0], ops[1], ops[2], ops[3], ops[4], ops[5], "insert_bits", SPIRType::UInt);
		break;
	}

	case OpBitFieldSExtract:
	{
		emit_trinary_func_op_bitextract(ops[0], ops[1], ops[2], ops[3], ops[4], "extract_bits", int_type, int_type,
		                                SPIRType::UInt, SPIRType::UInt);
		break;
	}

	case OpBitFieldUExtract:
	{
		emit_trinary_func_op_bitextract(ops[0], ops[1], ops[2], ops[3], ops[4], "extract_bits", uint_type, uint_type,
		                                SPIRType::UInt, SPIRType::UInt);
		break;
	}

	case OpBitReverse:
		// BitReverse does not have issues with sign since result type must match input type.
		MSL_UFOP(reverse_bits);
		break;

	case OpBitCount:
	{
		auto basetype = expression_type(ops[2]).basetype;
		emit_unary_func_op_cast(ops[0], ops[1], ops[2], "popcount", basetype, basetype);
		break;
	}

	case OpFRem:
		MSL_BFOP(fmod);
		break;

	case OpFMul:
		if (msl_options.invariant_float_math)
			MSL_BFOP(spvFMul);
		else
			MSL_BOP(*);
		break;

	case OpFAdd:
		if (msl_options.invariant_float_math)
			MSL_BFOP(spvFAdd);
		else
			MSL_BOP(+);
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

	// Emulate texture2D atomic operations
	case OpImageTexelPointer:
	{
		// When using the pointer, we need to know which variable it is actually loaded from.
		auto *var = maybe_get_backing_variable(ops[2]);
		if (var && atomic_image_vars.count(var->self))
		{
			uint32_t result_type = ops[0];
			uint32_t id = ops[1];

			std::string coord = to_expression(ops[3]);
			auto &type = expression_type(ops[2]);
			if (type.image.dim == Dim2D)
			{
				coord = join("spvImage2DAtomicCoord(", coord, ", ", to_expression(ops[2]), ")");
			}

			auto &e = set<SPIRExpression>(id, join(to_expression(ops[2]), "_atomic[", coord, "]"), result_type, true);
			e.loaded_from = var ? var->self : ID(0);
			inherit_expression_dependencies(id, ops[3]);
		}
		else
		{
			uint32_t result_type = ops[0];
			uint32_t id = ops[1];
			auto &e =
			    set<SPIRExpression>(id, join(to_expression(ops[2]), ", ", to_expression(ops[3])), result_type, true);

			// When using the pointer, we need to know which variable it is actually loaded from.
			e.loaded_from = var ? var->self : ID(0);
			inherit_expression_dependencies(id, ops[3]);
		}
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

		statement(join(to_expression(img_id), ".write(",
		               remap_swizzle(store_type, texel_type.vecsize, to_expression(texel_id)), ", ",
		               to_function_args(img_id, img_type, true, false, false, coord_id, 0, 0, 0, 0, lod, 0, 0, 0, 0, 0,
		                                0, &forward),
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
		{
			expr += ", " + img_exp + ".get_array_size()";
			if (img_dim == DimCube && msl_options.emulate_cube_array)
				expr += " / 6";
		}

		expr += ")";

		emit_op(rslt_type_id, id, expr, should_forward(img_id));

		break;
	}

	case OpImageQueryLod:
	{
		if (!msl_options.supports_msl_version(2, 2))
			SPIRV_CROSS_THROW("ImageQueryLod is only supported on MSL 2.2 and up.");
		uint32_t result_type = ops[0];
		uint32_t id = ops[1];
		uint32_t image_id = ops[2];
		uint32_t coord_id = ops[3];
		emit_uninitialized_temporary_expression(result_type, id);

		auto sampler_expr = to_sampler_expression(image_id);
		auto *combined = maybe_get<SPIRCombinedImageSampler>(image_id);
		auto image_expr = combined ? to_expression(combined->image) : to_expression(image_id);

		// TODO: It is unclear if calculcate_clamped_lod also conditionally rounds
		// the reported LOD based on the sampler. NEAREST miplevel should
		// round the LOD, but LINEAR miplevel should not round.
		// Let's hope this does not become an issue ...
		statement(to_expression(id), ".x = ", image_expr, ".calculate_clamped_lod(", sampler_expr, ", ",
		          to_expression(coord_id), ");");
		statement(to_expression(id), ".y = ", image_expr, ".calculate_unclamped_lod(", sampler_expr, ", ",
		          to_expression(coord_id), ");");
		register_control_dependent_expression(id);
		break;
	}

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
			auto *var = maybe_get_backing_variable(ops[2]);
			SPIRExpression *e;
			if (var && has_extended_decoration(var->self, SPIRVCrossDecorationDynamicImageSampler))
				e = &emit_op(result_type, id, join(to_expression(ops[2]), ".plane0"), true, true);
			else
				e = &emit_op(result_type, id, to_expression(ops[2]), true, true);
			if (var)
				e->loaded_from = var->self;
		}
		break;
	}

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

	case OpOuterProduct:
	{
		uint32_t result_type = ops[0];
		uint32_t id = ops[1];
		uint32_t a = ops[2];
		uint32_t b = ops[3];

		auto &type = get<SPIRType>(result_type);
		string expr = type_to_glsl_constructor(type);
		expr += "(";
		for (uint32_t col = 0; col < type.columns; col++)
		{
			expr += to_enclosed_expression(a);
			expr += " * ";
			expr += to_extract_component_expression(b, col);
			if (col + 1 < type.columns)
				expr += ", ";
		}
		expr += ")";
		emit_op(result_type, id, expr, should_forward(a) && should_forward(b));
		inherit_expression_dependencies(id, a);
		inherit_expression_dependencies(id, b);
		break;
	}

	case OpVectorTimesMatrix:
	case OpMatrixTimesVector:
	{
		if (!msl_options.invariant_float_math)
		{
			CompilerGLSL::emit_instruction(instruction);
			break;
		}

		// If the matrix needs transpose, just flip the multiply order.
		auto *e = maybe_get<SPIRExpression>(ops[opcode == OpMatrixTimesVector ? 2 : 3]);
		if (e && e->need_transpose)
		{
			e->need_transpose = false;
			string expr;

			if (opcode == OpMatrixTimesVector)
			{
				expr = join("spvFMulVectorMatrix(", to_enclosed_unpacked_expression(ops[3]), ", ",
				            to_unpacked_row_major_matrix_expression(ops[2]), ")");
			}
			else
			{
				expr = join("spvFMulMatrixVector(", to_unpacked_row_major_matrix_expression(ops[3]), ", ",
				            to_enclosed_unpacked_expression(ops[2]), ")");
			}

			bool forward = should_forward(ops[2]) && should_forward(ops[3]);
			emit_op(ops[0], ops[1], expr, forward);
			e->need_transpose = true;
			inherit_expression_dependencies(ops[1], ops[2]);
			inherit_expression_dependencies(ops[1], ops[3]);
		}
		else
		{
			if (opcode == OpMatrixTimesVector)
				MSL_BFOP(spvFMulMatrixVector);
			else
				MSL_BFOP(spvFMulVectorMatrix);
		}
		break;
	}

	case OpMatrixTimesMatrix:
	{
		if (!msl_options.invariant_float_math)
		{
			CompilerGLSL::emit_instruction(instruction);
			break;
		}

		auto *a = maybe_get<SPIRExpression>(ops[2]);
		auto *b = maybe_get<SPIRExpression>(ops[3]);

		// If both matrices need transpose, we can multiply in flipped order and tag the expression as transposed.
		// a^T * b^T = (b * a)^T.
		if (a && b && a->need_transpose && b->need_transpose)
		{
			a->need_transpose = false;
			b->need_transpose = false;

			auto expr =
			    join("spvFMulMatrixMatrix(", enclose_expression(to_unpacked_row_major_matrix_expression(ops[3])), ", ",
			         enclose_expression(to_unpacked_row_major_matrix_expression(ops[2])), ")");

			bool forward = should_forward(ops[2]) && should_forward(ops[3]);
			auto &e = emit_op(ops[0], ops[1], expr, forward);
			e.need_transpose = true;
			a->need_transpose = true;
			b->need_transpose = true;
			inherit_expression_dependencies(ops[1], ops[2]);
			inherit_expression_dependencies(ops[1], ops[3]);
		}
		else
			MSL_BFOP(spvFMulMatrixMatrix);

		break;
	}

	case OpIAddCarry:
	case OpISubBorrow:
	{
		uint32_t result_type = ops[0];
		uint32_t result_id = ops[1];
		uint32_t op0 = ops[2];
		uint32_t op1 = ops[3];
		auto &type = get<SPIRType>(result_type);
		emit_uninitialized_temporary_expression(result_type, result_id);

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
		auto &type = get<SPIRType>(result_type);
		emit_uninitialized_temporary_expression(result_type, result_id);

		statement(to_expression(result_id), ".", to_member_name(type, 0), " = ", to_enclosed_expression(op0), " * ",
		          to_enclosed_expression(op1), ";");
		statement(to_expression(result_id), ".", to_member_name(type, 1), " = mulhi(", to_expression(op0), ", ",
		          to_expression(op1), ");");
		break;
	}

	case OpArrayLength:
	{
		auto &type = expression_type(ops[2]);
		uint32_t offset = type_struct_member_offset(type, ops[3]);
		uint32_t stride = type_struct_member_array_stride(type, ops[3]);

		auto expr = join("(", to_buffer_size_expression(ops[2]), " - ", offset, ") / ", stride);
		emit_op(ops[0], ops[1], expr, true);
		break;
	}

	// SPV_INTEL_shader_integer_functions2
	case OpUCountLeadingZerosINTEL:
		MSL_UFOP(clz);
		break;

	case OpUCountTrailingZerosINTEL:
		MSL_UFOP(ctz);
		break;

	case OpAbsISubINTEL:
	case OpAbsUSubINTEL:
		MSL_BFOP(absdiff);
		break;

	case OpIAddSatINTEL:
	case OpUAddSatINTEL:
		MSL_BFOP(addsat);
		break;

	case OpIAverageINTEL:
	case OpUAverageINTEL:
		MSL_BFOP(hadd);
		break;

	case OpIAverageRoundedINTEL:
	case OpUAverageRoundedINTEL:
		MSL_BFOP(rhadd);
		break;

	case OpISubSatINTEL:
	case OpUSubSatINTEL:
		MSL_BFOP(subsat);
		break;

	case OpIMul32x16INTEL:
	{
		uint32_t result_type = ops[0];
		uint32_t id = ops[1];
		uint32_t a = ops[2], b = ops[3];
		bool forward = should_forward(a) && should_forward(b);
		emit_op(result_type, id, join("int(short(", to_expression(a), ")) * int(short(", to_expression(b), "))"),
		        forward);
		inherit_expression_dependencies(id, a);
		inherit_expression_dependencies(id, b);
		break;
	}

	case OpUMul32x16INTEL:
	{
		uint32_t result_type = ops[0];
		uint32_t id = ops[1];
		uint32_t a = ops[2], b = ops[3];
		bool forward = should_forward(a) && should_forward(b);
		emit_op(result_type, id, join("uint(ushort(", to_expression(a), ")) * uint(ushort(", to_expression(b), "))"),
		        forward);
		inherit_expression_dependencies(id, a);
		inherit_expression_dependencies(id, b);
		break;
	}

	case OpIsHelperInvocationEXT:
		if (msl_options.is_ios())
			SPIRV_CROSS_THROW("simd_is_helper_thread() is only supported on macOS.");
		else if (msl_options.is_macos() && !msl_options.supports_msl_version(2, 1))
			SPIRV_CROSS_THROW("simd_is_helper_thread() requires version 2.1 on macOS.");
		emit_op(ops[0], ops[1], "simd_is_helper_thread()", false);
		break;

	case OpBeginInvocationInterlockEXT:
	case OpEndInvocationInterlockEXT:
		if (!msl_options.supports_msl_version(2, 0))
			SPIRV_CROSS_THROW("Raster order groups require MSL 2.0.");
		break; // Nothing to do in the body

	default:
		CompilerGLSL::emit_instruction(instruction);
		break;
	}

	previous_instruction_opcode = opcode;
}

void CompilerMSL::emit_texture_op(const Instruction &i)
{
	if (msl_options.is_ios() && msl_options.ios_use_framebuffer_fetch_subpasses)
	{
		auto *ops = stream(i);

		uint32_t result_type_id = ops[0];
		uint32_t id = ops[1];
		uint32_t img = ops[2];

		auto &type = expression_type(img);
		auto &imgtype = get<SPIRType>(type.self);

		// Use Metal's native frame-buffer fetch API for subpass inputs.
		if (imgtype.image.dim == DimSubpassData)
		{
			// Subpass inputs cannot be invalidated,
			// so just forward the expression directly.
			string expr = to_expression(img);
			emit_op(result_type_id, id, expr, true);
			return;
		}
	}

	// Fallback to default implementation
	CompilerGLSL::emit_texture_op(i);
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

		// Fix tessellation patch function processing
		if (get_execution_model() == ExecutionModelTessellationControl ||
		    (mem_sem & (MemorySemanticsSubgroupMemoryMask | MemorySemanticsWorkgroupMemoryMask)))
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
		    (mem_sem & (MemorySemanticsSubgroupMemoryMask | MemorySemanticsWorkgroupMemoryMask)))
			bar_stmt += "mem_flags::mem_device_and_threadgroup";
		else if (mem_sem & (MemorySemanticsUniformMemoryMask | MemorySemanticsCrossWorkgroupMemoryMask))
			bar_stmt += "mem_flags::mem_device";
		else if (mem_sem & (MemorySemanticsSubgroupMemoryMask | MemorySemanticsWorkgroupMemoryMask))
			bar_stmt += "mem_flags::mem_threadgroup";
		else if (mem_sem & MemorySemanticsImageMemoryMask)
			bar_stmt += "mem_flags::mem_texture";
		else
			bar_stmt += "mem_flags::mem_none";
	}

	bar_stmt += ");";

	statement(bar_stmt);

	assert(current_emitting_block);
	flush_control_dependent_expressions(current_emitting_block->self);
	flush_all_active_variables();
}

void CompilerMSL::emit_array_copy(const string &lhs, uint32_t rhs_id, StorageClass lhs_storage,
                                  StorageClass rhs_storage)
{
	// Allow Metal to use the array<T> template to make arrays a value type.
	// This, however, cannot be used for threadgroup address specifiers, so consider the custom array copy as fallback.
	bool lhs_thread = (lhs_storage == StorageClassOutput || lhs_storage == StorageClassFunction ||
	                   lhs_storage == StorageClassGeneric || lhs_storage == StorageClassPrivate);
	bool rhs_thread = (rhs_storage == StorageClassInput || rhs_storage == StorageClassFunction ||
	                   rhs_storage == StorageClassGeneric || rhs_storage == StorageClassPrivate);

	// If threadgroup storage qualifiers are *not* used:
	// Avoid spvCopy* wrapper functions; Otherwise, spvUnsafeArray<> template cannot be used with that storage qualifier.
	if (lhs_thread && rhs_thread && !using_builtin_array())
	{
		statement(lhs, " = ", to_expression(rhs_id), ";");
	}
	else
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
			add_spv_func_and_recompile(func);
		}
		else
			add_spv_func_and_recompile(SPVFuncImplArrayCopy);

		const char *tag = nullptr;
		if (lhs_thread && is_constant)
			tag = "FromConstantToStack";
		else if (lhs_storage == StorageClassWorkgroup && is_constant)
			tag = "FromConstantToThreadGroup";
		else if (lhs_thread && rhs_thread)
			tag = "FromStackToStack";
		else if (lhs_storage == StorageClassWorkgroup && rhs_thread)
			tag = "FromStackToThreadGroup";
		else if (lhs_thread && rhs_storage == StorageClassWorkgroup)
			tag = "FromThreadGroupToStack";
		else if (lhs_storage == StorageClassWorkgroup && rhs_storage == StorageClassWorkgroup)
			tag = "FromThreadGroupToThreadGroup";
		else
			SPIRV_CROSS_THROW("Unknown storage class used for copying arrays.");

		// Pass internal array of spvUnsafeArray<> into wrapper functions
		if (lhs_thread && !msl_options.force_native_arrays)
			statement("spvArrayCopy", tag, type.array.size(), "(", lhs, ".elements, ", to_expression(rhs_id), ");");
		else if (rhs_thread && !msl_options.force_native_arrays)
			statement("spvArrayCopy", tag, type.array.size(), "(", lhs, ", ", to_expression(rhs_id), ".elements);");
		else
			statement("spvArrayCopy", tag, type.array.size(), "(", lhs, ", ", to_expression(rhs_id), ");");
	}
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

	emit_array_copy(to_expression(id_lhs), id_rhs, get_backing_variable_storage(id_lhs),
	                get_backing_variable_storage(id_rhs));
	register_write(id_lhs);

	return true;
}

// Emits one of the atomic functions. In MSL, the atomic functions operate on pointers
void CompilerMSL::emit_atomic_func_op(uint32_t result_type, uint32_t result_id, const char *op, uint32_t mem_order_1,
                                      uint32_t mem_order_2, bool has_mem_order_2, uint32_t obj, uint32_t op1,
                                      bool op1_is_pointer, bool op1_is_literal, uint32_t op2)
{
	string exp = string(op) + "(";

	auto &type = get_pointee_type(expression_type(obj));
	exp += "(";
	auto *var = maybe_get_backing_variable(obj);
	if (!var)
		SPIRV_CROSS_THROW("No backing variable for atomic operation.");

	// Emulate texture2D atomic operations
	const auto &res_type = get<SPIRType>(var->basetype);
	if (res_type.storage == StorageClassUniformConstant && res_type.basetype == SPIRType::Image)
	{
		exp += "device";
	}
	else
	{
		exp += get_argument_address_space(*var);
	}

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
		emit_uninitialized_temporary_expression(result_type, result_id);
		statement("do");
		begin_scope();
		statement(to_name(result_id), " = ", to_expression(op1), ";");
		end_scope_decl(join("while (!", exp, " && ", to_name(result_id), " == ", to_enclosed_expression(op1), ")"));
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

	case GLSLstd450FindILsb:
	{
		// In this template version of findLSB, we return T.
		auto basetype = expression_type(args[0]).basetype;
		emit_unary_func_op_cast(result_type, id, args[0], "spvFindLSB", basetype, basetype);
		break;
	}

	case GLSLstd450FindSMsb:
		emit_unary_func_op_cast(result_type, id, args[0], "spvFindSMSB", int_type, int_type);
		break;

	case GLSLstd450FindUMsb:
		emit_unary_func_op_cast(result_type, id, args[0], "spvFindUMSB", uint_type, uint_type);
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

	case GLSLstd450Distance:
		// MSL does not support scalar versions here.
		if (expression_type(args[0]).vecsize == 1)
		{
			// Equivalent to length(a - b) -> abs(a - b).
			emit_op(result_type, id,
			        join("abs(", to_unpacked_expression(args[0]), " - ", to_unpacked_expression(args[1]), ")"),
			        should_forward(args[0]) && should_forward(args[1]));
			inherit_expression_dependencies(id, args[0]);
			inherit_expression_dependencies(id, args[1]);
		}
		else
			CompilerGLSL::emit_glsl_op(result_type, id, eop, args, count);
		break;

	case GLSLstd450Length:
		// MSL does not support scalar versions here.
		if (expression_type(args[0]).vecsize == 1)
		{
			// Equivalent to abs().
			emit_unary_func_op(result_type, id, args[0], "abs");
		}
		else
			CompilerGLSL::emit_glsl_op(result_type, id, eop, args, count);
		break;

	case GLSLstd450Normalize:
		// MSL does not support scalar versions here.
		if (expression_type(args[0]).vecsize == 1)
		{
			// Returns -1 or 1 for valid input, sign() does the job.
			emit_unary_func_op(result_type, id, args[0], "sign");
		}
		else
			CompilerGLSL::emit_glsl_op(result_type, id, eop, args, count);
		break;

	case GLSLstd450Reflect:
		if (get<SPIRType>(result_type).vecsize == 1)
			emit_binary_func_op(result_type, id, args[0], args[1], "spvReflect");
		else
			CompilerGLSL::emit_glsl_op(result_type, id, eop, args, count);
		break;

	case GLSLstd450Refract:
		if (get<SPIRType>(result_type).vecsize == 1)
			emit_trinary_func_op(result_type, id, args[0], args[1], args[2], "spvRefract");
		else
			CompilerGLSL::emit_glsl_op(result_type, id, eop, args, count);
		break;

	case GLSLstd450FaceForward:
		if (get<SPIRType>(result_type).vecsize == 1)
			emit_trinary_func_op(result_type, id, args[0], args[1], args[2], "spvFaceForward");
		else
			CompilerGLSL::emit_glsl_op(result_type, id, eop, args, count);
		break;

	case GLSLstd450Modf:
	case GLSLstd450Frexp:
	{
		// Special case. If the variable is a scalar access chain, we cannot use it directly. We have to emit a temporary.
		auto *ptr = maybe_get<SPIRExpression>(args[1]);
		if (ptr && ptr->access_chain && is_scalar(expression_type(args[1])))
		{
			register_call_out_argument(args[1]);
			forced_temporaries.insert(id);

			// Need to create temporaries and copy over to access chain after.
			// We cannot directly take the reference of a vector swizzle in MSL, even if it's scalar ...
			uint32_t &tmp_id = extra_sub_expressions[id];
			if (!tmp_id)
				tmp_id = ir.increase_bound_by(1);

			uint32_t tmp_type_id = get_pointee_type_id(ptr->expression_type);
			emit_uninitialized_temporary_expression(tmp_type_id, tmp_id);
			emit_binary_func_op(result_type, id, args[0], tmp_id, eop == GLSLstd450Modf ? "modf" : "frexp");
			statement(to_expression(args[1]), " = ", to_expression(tmp_id), ";");
		}
		else
			CompilerGLSL::emit_glsl_op(result_type, id, eop, args, count);
		break;
	}

	default:
		CompilerGLSL::emit_glsl_op(result_type, id, eop, args, count);
		break;
	}
}

void CompilerMSL::emit_spv_amd_shader_trinary_minmax_op(uint32_t result_type, uint32_t id, uint32_t eop,
                                                        const uint32_t *args, uint32_t count)
{
	enum AMDShaderTrinaryMinMax
	{
		FMin3AMD = 1,
		UMin3AMD = 2,
		SMin3AMD = 3,
		FMax3AMD = 4,
		UMax3AMD = 5,
		SMax3AMD = 6,
		FMid3AMD = 7,
		UMid3AMD = 8,
		SMid3AMD = 9
	};

	if (!msl_options.supports_msl_version(2, 1))
		SPIRV_CROSS_THROW("Trinary min/max functions require MSL 2.1.");

	auto op = static_cast<AMDShaderTrinaryMinMax>(eop);

	switch (op)
	{
	case FMid3AMD:
	case UMid3AMD:
	case SMid3AMD:
		emit_trinary_func_op(result_type, id, args[0], args[1], args[2], "median3");
		break;
	default:
		CompilerGLSL::emit_spv_amd_shader_trinary_minmax_op(result_type, id, eop, args, count);
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

	processing_entry_point = func.self == ir.default_entry_point;

	// Metal helper functions must be static force-inline otherwise they will cause problems when linked together in a single Metallib.
	if (!processing_entry_point)
		statement(force_inline);

	auto &type = get<SPIRType>(func.return_type);

	if (!type.array.empty() && msl_options.force_native_arrays)
	{
		// We cannot return native arrays in MSL, so "return" through an out variable.
		decl += "void";
	}
	else
	{
		decl += func_type_decl(type);
	}

	decl += " ";
	decl += to_name(func.self);
	decl += "(";

	if (!type.array.empty() && msl_options.force_native_arrays)
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
			ID &initializer = ed_var.initializer;
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

		bool is_dynamic_img_sampler = has_extended_decoration(arg.id, SPIRVCrossDecorationDynamicImageSampler);

		auto &arg_type = get<SPIRType>(arg.type);
		if (arg_type.basetype == SPIRType::SampledImage && !is_dynamic_img_sampler)
		{
			// Manufacture automatic plane args for multiplanar texture
			uint32_t planes = 1;
			if (auto *constexpr_sampler = find_constexpr_sampler(name_id))
				if (constexpr_sampler->ycbcr_conversion_enable)
					planes = constexpr_sampler->planes;
			for (uint32_t i = 1; i < planes; i++)
				decl += join(", ", argument_decl(arg), plane_name_suffix, i);

			// Manufacture automatic sampler arg for SampledImage texture
			if (arg_type.image.dim != DimBuffer)
				decl += join(", thread const ", sampler_type(arg_type), " ", to_sampler_expression(arg.id));
		}

		// Manufacture automatic swizzle arg.
		if (msl_options.swizzle_texture_samples && has_sampled_images && is_sampled_image_type(arg_type) &&
		    !is_dynamic_img_sampler)
		{
			bool arg_is_array = !arg_type.array.empty();
			decl += join(", constant uint", arg_is_array ? "* " : "& ", to_swizzle_expression(arg.id));
		}

		if (buffers_requiring_array_length.count(name_id))
		{
			bool arg_is_array = !arg_type.array.empty();
			decl += join(", constant uint", arg_is_array ? "* " : "& ", to_buffer_size_expression(name_id));
		}

		if (&arg != &func.arguments.back())
			decl += ", ";
	}

	decl += ")";
	statement(decl);
}

static bool needs_chroma_reconstruction(const MSLConstexprSampler *constexpr_sampler)
{
	// For now, only multiplanar images need explicit reconstruction. GBGR and BGRG images
	// use implicit reconstruction.
	return constexpr_sampler && constexpr_sampler->ycbcr_conversion_enable && constexpr_sampler->planes > 1;
}

// Returns the texture sampling function string for the specified image and sampling characteristics.
string CompilerMSL::to_function_name(VariableID img, const SPIRType &imgtype, bool is_fetch, bool is_gather, bool, bool,
                                     bool, bool, bool has_dref, uint32_t, uint32_t)
{
	const MSLConstexprSampler *constexpr_sampler = nullptr;
	bool is_dynamic_img_sampler = false;
	if (auto *var = maybe_get_backing_variable(img))
	{
		constexpr_sampler = find_constexpr_sampler(var->basevariable ? var->basevariable : VariableID(var->self));
		is_dynamic_img_sampler = has_extended_decoration(var->self, SPIRVCrossDecorationDynamicImageSampler);
	}

	// Special-case gather. We have to alter the component being looked up
	// in the swizzle case.
	if (msl_options.swizzle_texture_samples && is_gather && !is_dynamic_img_sampler &&
	    (!constexpr_sampler || !constexpr_sampler->ycbcr_conversion_enable))
	{
		add_spv_func_and_recompile(imgtype.image.depth ? SPVFuncImplGatherCompareSwizzle : SPVFuncImplGatherSwizzle);
		return imgtype.image.depth ? "spvGatherCompareSwizzle" : "spvGatherSwizzle";
	}

	auto *combined = maybe_get<SPIRCombinedImageSampler>(img);

	// Texture reference
	string fname;
	if (needs_chroma_reconstruction(constexpr_sampler) && !is_dynamic_img_sampler)
	{
		if (constexpr_sampler->planes != 2 && constexpr_sampler->planes != 3)
			SPIRV_CROSS_THROW("Unhandled number of color image planes!");
		// 444 images aren't downsampled, so we don't need to do linear filtering.
		if (constexpr_sampler->resolution == MSL_FORMAT_RESOLUTION_444 ||
		    constexpr_sampler->chroma_filter == MSL_SAMPLER_FILTER_NEAREST)
		{
			if (constexpr_sampler->planes == 2)
				add_spv_func_and_recompile(SPVFuncImplChromaReconstructNearest2Plane);
			else
				add_spv_func_and_recompile(SPVFuncImplChromaReconstructNearest3Plane);
			fname = "spvChromaReconstructNearest";
		}
		else // Linear with a downsampled format
		{
			fname = "spvChromaReconstructLinear";
			switch (constexpr_sampler->resolution)
			{
			case MSL_FORMAT_RESOLUTION_444:
				assert(false);
				break; // not reached
			case MSL_FORMAT_RESOLUTION_422:
				switch (constexpr_sampler->x_chroma_offset)
				{
				case MSL_CHROMA_LOCATION_COSITED_EVEN:
					if (constexpr_sampler->planes == 2)
						add_spv_func_and_recompile(SPVFuncImplChromaReconstructLinear422CositedEven2Plane);
					else
						add_spv_func_and_recompile(SPVFuncImplChromaReconstructLinear422CositedEven3Plane);
					fname += "422CositedEven";
					break;
				case MSL_CHROMA_LOCATION_MIDPOINT:
					if (constexpr_sampler->planes == 2)
						add_spv_func_and_recompile(SPVFuncImplChromaReconstructLinear422Midpoint2Plane);
					else
						add_spv_func_and_recompile(SPVFuncImplChromaReconstructLinear422Midpoint3Plane);
					fname += "422Midpoint";
					break;
				default:
					SPIRV_CROSS_THROW("Invalid chroma location.");
				}
				break;
			case MSL_FORMAT_RESOLUTION_420:
				fname += "420";
				switch (constexpr_sampler->x_chroma_offset)
				{
				case MSL_CHROMA_LOCATION_COSITED_EVEN:
					switch (constexpr_sampler->y_chroma_offset)
					{
					case MSL_CHROMA_LOCATION_COSITED_EVEN:
						if (constexpr_sampler->planes == 2)
							add_spv_func_and_recompile(
							    SPVFuncImplChromaReconstructLinear420XCositedEvenYCositedEven2Plane);
						else
							add_spv_func_and_recompile(
							    SPVFuncImplChromaReconstructLinear420XCositedEvenYCositedEven3Plane);
						fname += "XCositedEvenYCositedEven";
						break;
					case MSL_CHROMA_LOCATION_MIDPOINT:
						if (constexpr_sampler->planes == 2)
							add_spv_func_and_recompile(
							    SPVFuncImplChromaReconstructLinear420XCositedEvenYMidpoint2Plane);
						else
							add_spv_func_and_recompile(
							    SPVFuncImplChromaReconstructLinear420XCositedEvenYMidpoint3Plane);
						fname += "XCositedEvenYMidpoint";
						break;
					default:
						SPIRV_CROSS_THROW("Invalid Y chroma location.");
					}
					break;
				case MSL_CHROMA_LOCATION_MIDPOINT:
					switch (constexpr_sampler->y_chroma_offset)
					{
					case MSL_CHROMA_LOCATION_COSITED_EVEN:
						if (constexpr_sampler->planes == 2)
							add_spv_func_and_recompile(
							    SPVFuncImplChromaReconstructLinear420XMidpointYCositedEven2Plane);
						else
							add_spv_func_and_recompile(
							    SPVFuncImplChromaReconstructLinear420XMidpointYCositedEven3Plane);
						fname += "XMidpointYCositedEven";
						break;
					case MSL_CHROMA_LOCATION_MIDPOINT:
						if (constexpr_sampler->planes == 2)
							add_spv_func_and_recompile(SPVFuncImplChromaReconstructLinear420XMidpointYMidpoint2Plane);
						else
							add_spv_func_and_recompile(SPVFuncImplChromaReconstructLinear420XMidpointYMidpoint3Plane);
						fname += "XMidpointYMidpoint";
						break;
					default:
						SPIRV_CROSS_THROW("Invalid Y chroma location.");
					}
					break;
				default:
					SPIRV_CROSS_THROW("Invalid X chroma location.");
				}
				break;
			default:
				SPIRV_CROSS_THROW("Invalid format resolution.");
			}
		}
	}
	else
	{
		fname = to_expression(combined ? combined->image : img) + ".";

		// Texture function and sampler
		if (is_fetch)
			fname += "read";
		else if (is_gather)
			fname += "gather";
		else
			fname += "sample";

		if (has_dref)
			fname += "_compare";
	}

	return fname;
}

string CompilerMSL::convert_to_f32(const string &expr, uint32_t components)
{
	SPIRType t;
	t.basetype = SPIRType::Float;
	t.vecsize = components;
	t.columns = 1;
	return join(type_to_glsl_constructor(t), "(", expr, ")");
}

static inline bool sampling_type_needs_f32_conversion(const SPIRType &type)
{
	// Double is not supported to begin with, but doesn't hurt to check for completion.
	return type.basetype == SPIRType::Half || type.basetype == SPIRType::Double;
}

// Returns the function args for a texture sampling function for the specified image and sampling characteristics.
string CompilerMSL::to_function_args(VariableID img, const SPIRType &imgtype, bool is_fetch, bool is_gather,
                                     bool is_proj, uint32_t coord, uint32_t, uint32_t dref, uint32_t grad_x,
                                     uint32_t grad_y, uint32_t lod, uint32_t coffset, uint32_t offset, uint32_t bias,
                                     uint32_t comp, uint32_t sample, uint32_t minlod, bool *p_forward)
{
	const MSLConstexprSampler *constexpr_sampler = nullptr;
	bool is_dynamic_img_sampler = false;
	if (auto *var = maybe_get_backing_variable(img))
	{
		constexpr_sampler = find_constexpr_sampler(var->basevariable ? var->basevariable : VariableID(var->self));
		is_dynamic_img_sampler = has_extended_decoration(var->self, SPIRVCrossDecorationDynamicImageSampler);
	}

	string farg_str;
	bool forward = true;

	if (!is_dynamic_img_sampler)
	{
		// Texture reference (for some cases)
		if (needs_chroma_reconstruction(constexpr_sampler))
		{
			// Multiplanar images need two or three textures.
			farg_str += to_expression(img);
			for (uint32_t i = 1; i < constexpr_sampler->planes; i++)
				farg_str += join(", ", to_expression(img), plane_name_suffix, i);
		}
		else if ((!constexpr_sampler || !constexpr_sampler->ycbcr_conversion_enable) &&
		         msl_options.swizzle_texture_samples && is_gather)
		{
			auto *combined = maybe_get<SPIRCombinedImageSampler>(img);
			farg_str += to_expression(combined ? combined->image : img);
		}

		// Sampler reference
		if (!is_fetch)
		{
			if (!farg_str.empty())
				farg_str += ", ";
			farg_str += to_sampler_expression(img);
		}

		if ((!constexpr_sampler || !constexpr_sampler->ycbcr_conversion_enable) &&
		    msl_options.swizzle_texture_samples && is_gather)
		{
			// Add the swizzle constant from the swizzle buffer.
			farg_str += ", " + to_swizzle_expression(img);
			used_swizzle_buffer = true;
		}

		// Swizzled gather puts the component before the other args, to allow template
		// deduction to work.
		if (comp && msl_options.swizzle_texture_samples)
		{
			forward = should_forward(comp);
			farg_str += ", " + to_component_argument(comp);
		}
	}

	// Texture coordinates
	forward = forward && should_forward(coord);
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
		else if (sampling_type_needs_f32_conversion(coord_type))
			tex_coords = convert_to_f32(tex_coords, 1);

		if (msl_options.texture_1D_as_2D)
		{
			if (is_fetch)
				tex_coords = "uint2(" + tex_coords + ", 0)";
			else
				tex_coords = "float2(" + tex_coords + ", 0.5)";
		}

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
			// Support for Metal 2.1's new texture_buffer type.
			if (is_fetch)
			{
				if (msl_options.texel_buffer_texture_width > 0)
				{
					tex_coords = "spvTexelBufferCoord(" + round_fp_tex_coords(tex_coords, coord_is_fp) + ")";
				}
				else
				{
					tex_coords = "spvTexelBufferCoord(" + round_fp_tex_coords(tex_coords, coord_is_fp) + ", " +
					             to_expression(img) + ")";
				}
			}
		}

		alt_coord_component = 1;
		break;

	case DimSubpassData:
		// If we're using Metal's native frame-buffer fetch API for subpass inputs,
		// this path will not be hit.
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
		else if (sampling_type_needs_f32_conversion(coord_type))
			tex_coords = convert_to_f32(tex_coords, 2);

		alt_coord_component = 2;
		break;

	case Dim3D:
		if (coord_type.vecsize > 3)
			tex_coords = enclose_expression(tex_coords) + ".xyz";

		if (is_fetch)
			tex_coords = "uint3(" + round_fp_tex_coords(tex_coords, coord_is_fp) + ")";
		else if (sampling_type_needs_f32_conversion(coord_type))
			tex_coords = convert_to_f32(tex_coords, 3);

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

		if (sampling_type_needs_f32_conversion(coord_type))
			tex_coords = convert_to_f32(tex_coords, 3);

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
	{
		if (sampling_type_needs_f32_conversion(coord_type))
			tex_coords += " / " + convert_to_f32(to_extract_component_expression(coord, alt_coord_component), 1);
		else
			tex_coords += " / " + to_extract_component_expression(coord, alt_coord_component);
	}

	if (!farg_str.empty())
		farg_str += ", ";

	if (imgtype.image.dim == DimCube && imgtype.image.arrayed && msl_options.emulate_cube_array)
	{
		farg_str += "spvCubemapTo2DArrayFace(" + tex_coords + ").xy";

		if (is_cube_fetch)
			farg_str += ", uint(" + to_extract_component_expression(coord, 2) + ")";
		else
			farg_str += ", uint(spvCubemapTo2DArrayFace(" + tex_coords + ").z) + (uint(" +
			            round_fp_tex_coords(to_extract_component_expression(coord, alt_coord_component), coord_is_fp) +
			            ") * 6u)";

		add_spv_func_and_recompile(SPVFuncImplCubemapTo2DArrayFace);
	}
	else
	{
		farg_str += tex_coords;

		// If fetch from cube, add face explicitly
		if (is_cube_fetch)
		{
			// Special case for cube arrays, face and layer are packed in one dimension.
			if (imgtype.image.arrayed)
				farg_str += ", uint(" + to_extract_component_expression(coord, 2) + ") % 6u";
			else
				farg_str +=
				    ", uint(" + round_fp_tex_coords(to_extract_component_expression(coord, 2), coord_is_fp) + ")";
		}

		// If array, use alt coord
		if (imgtype.image.arrayed)
		{
			// Special case for cube arrays, face and layer are packed in one dimension.
			if (imgtype.image.dim == DimCube && is_fetch)
				farg_str += ", uint(" + to_extract_component_expression(coord, 2) + ") / 6u";
			else
				farg_str +=
				    ", uint(" +
				    round_fp_tex_coords(to_extract_component_expression(coord, alt_coord_component), coord_is_fp) + ")";
		}
	}

	// Depth compare reference value
	if (dref)
	{
		forward = forward && should_forward(dref);
		farg_str += ", ";

		auto &dref_type = expression_type(dref);

		string dref_expr;
		if (is_proj)
			dref_expr =
			    join(to_enclosed_expression(dref), " / ", to_extract_component_expression(coord, alt_coord_component));
		else
			dref_expr = to_expression(dref);

		if (sampling_type_needs_f32_conversion(dref_type))
			dref_expr = convert_to_f32(dref_expr, 1);

		farg_str += dref_expr;

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
	if (bias && (imgtype.image.dim != Dim1D || msl_options.texture_1D_as_2D))
	{
		forward = forward && should_forward(bias);
		farg_str += ", bias(" + to_expression(bias) + ")";
	}

	// Metal does not support LOD for 1D textures.
	if (lod && (imgtype.image.dim != Dim1D || msl_options.texture_1D_as_2D))
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
	else if (is_fetch && !lod && (imgtype.image.dim != Dim1D || msl_options.texture_1D_as_2D) &&
	         imgtype.image.dim != DimBuffer && !imgtype.image.ms && imgtype.image.sampled != 2)
	{
		// Lod argument is optional in OpImageFetch, but we require a LOD value, pick 0 as the default.
		// Check for sampled type as well, because is_fetch is also used for OpImageRead in MSL.
		farg_str += ", 0";
	}

	// Metal does not support LOD for 1D textures.
	if ((grad_x || grad_y) && (imgtype.image.dim != Dim1D || msl_options.texture_1D_as_2D))
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
			if (imgtype.image.arrayed && msl_options.emulate_cube_array)
				grad_opt = "2d";
			else
				grad_opt = "cube";
			break;
		default:
			grad_opt = "unsupported_gradient_dimension";
			break;
		}
		farg_str += ", gradient" + grad_opt + "(" + to_expression(grad_x) + ", " + to_expression(grad_y) + ")";
	}

	if (minlod)
	{
		if (msl_options.is_macos())
		{
			if (!msl_options.supports_msl_version(2, 2))
				SPIRV_CROSS_THROW("min_lod_clamp() is only supported in MSL 2.2+ and up on macOS.");
		}
		else if (msl_options.is_ios())
			SPIRV_CROSS_THROW("min_lod_clamp() is not supported on iOS.");

		forward = forward && should_forward(minlod);
		farg_str += ", min_lod_clamp(" + to_expression(minlod) + ")";
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

		if (!msl_options.swizzle_texture_samples || is_dynamic_img_sampler)
		{
			forward = forward && should_forward(comp);
			farg_str += ", " + to_component_argument(comp);
		}
	}

	if (sample)
	{
		forward = forward && should_forward(sample);
		farg_str += ", ";
		farg_str += to_expression(sample);
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

string CompilerMSL::to_texture_op(const Instruction &i, bool *forward, SmallVector<uint32_t> &inherited_expressions)
{
	auto *ops = stream(i);
	uint32_t result_type_id = ops[0];
	uint32_t img = ops[2];
	auto &result_type = get<SPIRType>(result_type_id);
	auto op = static_cast<Op>(i.op);
	bool is_gather = (op == OpImageGather || op == OpImageDrefGather);

	// Bypass pointers because we need the real image struct
	auto &type = expression_type(img);
	auto &imgtype = get<SPIRType>(type.self);

	const MSLConstexprSampler *constexpr_sampler = nullptr;
	bool is_dynamic_img_sampler = false;
	if (auto *var = maybe_get_backing_variable(img))
	{
		constexpr_sampler = find_constexpr_sampler(var->basevariable ? var->basevariable : VariableID(var->self));
		is_dynamic_img_sampler = has_extended_decoration(var->self, SPIRVCrossDecorationDynamicImageSampler);
	}

	string expr;
	if (constexpr_sampler && constexpr_sampler->ycbcr_conversion_enable && !is_dynamic_img_sampler)
	{
		// If this needs sampler Y'CbCr conversion, we need to do some additional
		// processing.
		switch (constexpr_sampler->ycbcr_model)
		{
		case MSL_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_BT_709:
			add_spv_func_and_recompile(SPVFuncImplConvertYCbCrBT709);
			expr += "spvConvertYCbCrBT709(";
			break;
		case MSL_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_BT_601:
			add_spv_func_and_recompile(SPVFuncImplConvertYCbCrBT601);
			expr += "spvConvertYCbCrBT601(";
			break;
		case MSL_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_BT_2020:
			add_spv_func_and_recompile(SPVFuncImplConvertYCbCrBT2020);
			expr += "spvConvertYCbCrBT2020(";
			break;
		default:
			SPIRV_CROSS_THROW("Invalid Y'CbCr model conversion.");
		}

		if (constexpr_sampler->ycbcr_model != MSL_SAMPLER_YCBCR_MODEL_CONVERSION_RGB_IDENTITY)
		{
			switch (constexpr_sampler->ycbcr_range)
			{
			case MSL_SAMPLER_YCBCR_RANGE_ITU_FULL:
				add_spv_func_and_recompile(SPVFuncImplExpandITUFullRange);
				expr += "spvExpandITUFullRange(";
				break;
			case MSL_SAMPLER_YCBCR_RANGE_ITU_NARROW:
				add_spv_func_and_recompile(SPVFuncImplExpandITUNarrowRange);
				expr += "spvExpandITUNarrowRange(";
				break;
			default:
				SPIRV_CROSS_THROW("Invalid Y'CbCr range.");
			}
		}
	}
	else if (msl_options.swizzle_texture_samples && !is_gather && is_sampled_image_type(imgtype) &&
	         !is_dynamic_img_sampler)
	{
		add_spv_func_and_recompile(SPVFuncImplTextureSwizzle);
		expr += "spvTextureSwizzle(";
	}

	string inner_expr = CompilerGLSL::to_texture_op(i, forward, inherited_expressions);

	if (constexpr_sampler && constexpr_sampler->ycbcr_conversion_enable && !is_dynamic_img_sampler)
	{
		if (!constexpr_sampler->swizzle_is_identity())
		{
			static const char swizzle_names[] = "rgba";
			if (!constexpr_sampler->swizzle_has_one_or_zero())
			{
				// If we can, do it inline.
				expr += inner_expr + ".";
				for (uint32_t c = 0; c < 4; c++)
				{
					switch (constexpr_sampler->swizzle[c])
					{
					case MSL_COMPONENT_SWIZZLE_IDENTITY:
						expr += swizzle_names[c];
						break;
					case MSL_COMPONENT_SWIZZLE_R:
					case MSL_COMPONENT_SWIZZLE_G:
					case MSL_COMPONENT_SWIZZLE_B:
					case MSL_COMPONENT_SWIZZLE_A:
						expr += swizzle_names[constexpr_sampler->swizzle[c] - MSL_COMPONENT_SWIZZLE_R];
						break;
					default:
						SPIRV_CROSS_THROW("Invalid component swizzle.");
					}
				}
			}
			else
			{
				// Otherwise, we need to emit a temporary and swizzle that.
				uint32_t temp_id = ir.increase_bound_by(1);
				emit_op(result_type_id, temp_id, inner_expr, false);
				for (auto &inherit : inherited_expressions)
					inherit_expression_dependencies(temp_id, inherit);
				inherited_expressions.clear();
				inherited_expressions.push_back(temp_id);

				switch (op)
				{
				case OpImageSampleDrefImplicitLod:
				case OpImageSampleImplicitLod:
				case OpImageSampleProjImplicitLod:
				case OpImageSampleProjDrefImplicitLod:
					register_control_dependent_expression(temp_id);
					break;

				default:
					break;
				}
				expr += type_to_glsl(result_type) + "(";
				for (uint32_t c = 0; c < 4; c++)
				{
					switch (constexpr_sampler->swizzle[c])
					{
					case MSL_COMPONENT_SWIZZLE_IDENTITY:
						expr += to_expression(temp_id) + "." + swizzle_names[c];
						break;
					case MSL_COMPONENT_SWIZZLE_ZERO:
						expr += "0";
						break;
					case MSL_COMPONENT_SWIZZLE_ONE:
						expr += "1";
						break;
					case MSL_COMPONENT_SWIZZLE_R:
					case MSL_COMPONENT_SWIZZLE_G:
					case MSL_COMPONENT_SWIZZLE_B:
					case MSL_COMPONENT_SWIZZLE_A:
						expr += to_expression(temp_id) + "." +
						        swizzle_names[constexpr_sampler->swizzle[c] - MSL_COMPONENT_SWIZZLE_R];
						break;
					default:
						SPIRV_CROSS_THROW("Invalid component swizzle.");
					}
					if (c < 3)
						expr += ", ";
				}
				expr += ")";
			}
		}
		else
			expr += inner_expr;
		if (constexpr_sampler->ycbcr_model != MSL_SAMPLER_YCBCR_MODEL_CONVERSION_RGB_IDENTITY)
		{
			expr += join(", ", constexpr_sampler->bpc, ")");
			if (constexpr_sampler->ycbcr_model != MSL_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_IDENTITY)
				expr += ")";
		}
	}
	else
	{
		expr += inner_expr;
		if (msl_options.swizzle_texture_samples && !is_gather && is_sampled_image_type(imgtype) &&
		    !is_dynamic_img_sampler)
		{
			// Add the swizzle constant from the swizzle buffer.
			expr += ", " + to_swizzle_expression(img) + ")";
			used_swizzle_buffer = true;
		}
	}

	return expr;
}

static string create_swizzle(MSLComponentSwizzle swizzle)
{
	switch (swizzle)
	{
	case MSL_COMPONENT_SWIZZLE_IDENTITY:
		return "spvSwizzle::none";
	case MSL_COMPONENT_SWIZZLE_ZERO:
		return "spvSwizzle::zero";
	case MSL_COMPONENT_SWIZZLE_ONE:
		return "spvSwizzle::one";
	case MSL_COMPONENT_SWIZZLE_R:
		return "spvSwizzle::red";
	case MSL_COMPONENT_SWIZZLE_G:
		return "spvSwizzle::green";
	case MSL_COMPONENT_SWIZZLE_B:
		return "spvSwizzle::blue";
	case MSL_COMPONENT_SWIZZLE_A:
		return "spvSwizzle::alpha";
	default:
		SPIRV_CROSS_THROW("Invalid component swizzle.");
		return "";
	}
}

// Returns a string representation of the ID, usable as a function arg.
// Manufacture automatic sampler arg for SampledImage texture.
string CompilerMSL::to_func_call_arg(const SPIRFunction::Parameter &arg, uint32_t id)
{
	string arg_str;

	auto &type = expression_type(id);
	bool is_dynamic_img_sampler = has_extended_decoration(arg.id, SPIRVCrossDecorationDynamicImageSampler);
	// If the argument *itself* is a "dynamic" combined-image sampler, then we can just pass that around.
	bool arg_is_dynamic_img_sampler = has_extended_decoration(id, SPIRVCrossDecorationDynamicImageSampler);
	if (is_dynamic_img_sampler && !arg_is_dynamic_img_sampler)
		arg_str = join("spvDynamicImageSampler<", type_to_glsl(get<SPIRType>(type.image.type)), ">(");

	auto *c = maybe_get<SPIRConstant>(id);
	if (msl_options.force_native_arrays && c && !get<SPIRType>(c->constant_type).array.empty())
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
		auto itr = find(begin(constants), end(constants), ID(id));
		if (itr == end(constants))
		{
			force_recompile();
			constants.push_back(id);
		}
	}
	else
		arg_str += CompilerGLSL::to_func_call_arg(arg, id);

	// Need to check the base variable in case we need to apply a qualified alias.
	uint32_t var_id = 0;
	auto *var = maybe_get<SPIRVariable>(id);
	if (var)
		var_id = var->basevariable;

	if (!arg_is_dynamic_img_sampler)
	{
		auto *constexpr_sampler = find_constexpr_sampler(var_id ? var_id : id);
		if (type.basetype == SPIRType::SampledImage)
		{
			// Manufacture automatic plane args for multiplanar texture
			uint32_t planes = 1;
			if (constexpr_sampler && constexpr_sampler->ycbcr_conversion_enable)
			{
				planes = constexpr_sampler->planes;
				// If this parameter isn't aliasing a global, then we need to use
				// the special "dynamic image-sampler" class to pass it--and we need
				// to use it for *every* non-alias parameter, in case a combined
				// image-sampler with a Y'CbCr conversion is passed. Hopefully, this
				// pathological case is so rare that it should never be hit in practice.
				if (!arg.alias_global_variable)
					add_spv_func_and_recompile(SPVFuncImplDynamicImageSampler);
			}
			for (uint32_t i = 1; i < planes; i++)
				arg_str += join(", ", CompilerGLSL::to_func_call_arg(arg, id), plane_name_suffix, i);
			// Manufacture automatic sampler arg if the arg is a SampledImage texture.
			if (type.image.dim != DimBuffer)
				arg_str += ", " + to_sampler_expression(var_id ? var_id : id);

			// Add sampler Y'CbCr conversion info if we have it
			if (is_dynamic_img_sampler && constexpr_sampler && constexpr_sampler->ycbcr_conversion_enable)
			{
				SmallVector<string> samp_args;

				switch (constexpr_sampler->resolution)
				{
				case MSL_FORMAT_RESOLUTION_444:
					// Default
					break;
				case MSL_FORMAT_RESOLUTION_422:
					samp_args.push_back("spvFormatResolution::_422");
					break;
				case MSL_FORMAT_RESOLUTION_420:
					samp_args.push_back("spvFormatResolution::_420");
					break;
				default:
					SPIRV_CROSS_THROW("Invalid format resolution.");
				}

				if (constexpr_sampler->chroma_filter != MSL_SAMPLER_FILTER_NEAREST)
					samp_args.push_back("spvChromaFilter::linear");

				if (constexpr_sampler->x_chroma_offset != MSL_CHROMA_LOCATION_COSITED_EVEN)
					samp_args.push_back("spvXChromaLocation::midpoint");
				if (constexpr_sampler->y_chroma_offset != MSL_CHROMA_LOCATION_COSITED_EVEN)
					samp_args.push_back("spvYChromaLocation::midpoint");
				switch (constexpr_sampler->ycbcr_model)
				{
				case MSL_SAMPLER_YCBCR_MODEL_CONVERSION_RGB_IDENTITY:
					// Default
					break;
				case MSL_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_IDENTITY:
					samp_args.push_back("spvYCbCrModelConversion::ycbcr_identity");
					break;
				case MSL_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_BT_709:
					samp_args.push_back("spvYCbCrModelConversion::ycbcr_bt_709");
					break;
				case MSL_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_BT_601:
					samp_args.push_back("spvYCbCrModelConversion::ycbcr_bt_601");
					break;
				case MSL_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_BT_2020:
					samp_args.push_back("spvYCbCrModelConversion::ycbcr_bt_2020");
					break;
				default:
					SPIRV_CROSS_THROW("Invalid Y'CbCr model conversion.");
				}
				if (constexpr_sampler->ycbcr_range != MSL_SAMPLER_YCBCR_RANGE_ITU_FULL)
					samp_args.push_back("spvYCbCrRange::itu_narrow");
				samp_args.push_back(join("spvComponentBits(", constexpr_sampler->bpc, ")"));
				arg_str += join(", spvYCbCrSampler(", merge(samp_args), ")");
			}
		}

		if (is_dynamic_img_sampler && constexpr_sampler && constexpr_sampler->ycbcr_conversion_enable)
			arg_str += join(", (uint(", create_swizzle(constexpr_sampler->swizzle[3]), ") << 24) | (uint(",
			                create_swizzle(constexpr_sampler->swizzle[2]), ") << 16) | (uint(",
			                create_swizzle(constexpr_sampler->swizzle[1]), ") << 8) | uint(",
			                create_swizzle(constexpr_sampler->swizzle[0]), ")");
		else if (msl_options.swizzle_texture_samples && has_sampled_images && is_sampled_image_type(type))
			arg_str += ", " + to_swizzle_expression(var_id ? var_id : id);

		if (buffers_requiring_array_length.count(var_id))
			arg_str += ", " + to_buffer_size_expression(var_id ? var_id : id);

		if (is_dynamic_img_sampler)
			arg_str += ")";
	}

	// Emulate texture2D atomic operations
	auto *backing_var = maybe_get_backing_variable(var_id);
	if (backing_var && atomic_image_vars.count(backing_var->self))
	{
		arg_str += ", " + to_expression(var_id) + "_atomic";
	}

	return arg_str;
}

// If the ID represents a sampled image that has been assigned a sampler already,
// generate an expression for the sampler, otherwise generate a fake sampler name
// by appending a suffix to the expression constructed from the ID.
string CompilerMSL::to_sampler_expression(uint32_t id)
{
	auto *combined = maybe_get<SPIRCombinedImageSampler>(id);
	auto expr = to_expression(combined ? combined->image : VariableID(id));
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

	auto expr = to_expression(combined ? combined->image : VariableID(id));
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

string CompilerMSL::to_buffer_size_expression(uint32_t id)
{
	auto expr = to_expression(id);
	auto index = expr.find_first_of('[');

	// This is quite crude, but we need to translate the reference name (*spvDescriptorSetN.name) to
	// the pointer expression spvDescriptorSetN.name to make a reasonable expression here.
	// This only happens if we have argument buffers and we are using OpArrayLength on a lone SSBO in that set.
	if (expr.size() >= 3 && expr[0] == '(' && expr[1] == '*')
		expr = address_of_expression(expr);

	// If a buffer is part of an argument buffer translate this to a legal identifier.
	for (auto &c : expr)
		if (c == '.')
			c = '_';

	if (index == string::npos)
		return expr + buffer_size_name_suffix;
	else
	{
		auto buffer_expr = expr.substr(0, index);
		auto array_expr = expr.substr(index);
		return buffer_expr + buffer_size_name_suffix + array_expr;
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
	auto *e = maybe_get<SPIRExpression>(id);
	if (e)
		return e->need_transpose;
	else
		return has_decoration(id, DecorationRowMajor);
}

// Checks whether the member is a row_major matrix that requires conversion before use
bool CompilerMSL::member_is_non_native_row_major_matrix(const SPIRType &type, uint32_t index)
{
	return has_member_decoration(type.self, index, DecorationRowMajor);
}

string CompilerMSL::convert_row_major_matrix(string exp_str, const SPIRType &exp_type, uint32_t physical_type_id,
                                             bool is_packed)
{
	if (!is_matrix(exp_type))
	{
		return CompilerGLSL::convert_row_major_matrix(move(exp_str), exp_type, physical_type_id, is_packed);
	}
	else
	{
		strip_enclosed_expression(exp_str);
		if (physical_type_id != 0 || is_packed)
			exp_str = unpack_expression_type(exp_str, exp_type, physical_type_id, is_packed, true);
		return join("transpose(", exp_str, ")");
	}
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
	if (member_is_remapped_physical_type(type, index))
		member_type_id = get_extended_member_decoration(type.self, index, SPIRVCrossDecorationPhysicalTypeID);
	auto &physical_type = get<SPIRType>(member_type_id);

	// If this member is packed, mark it as so.
	string pack_pfx;

	// Allow Metal to use the array<T> template to make arrays a value type
	uint32_t orig_id = 0;
	if (has_extended_member_decoration(type.self, index, SPIRVCrossDecorationInterfaceOrigID))
		orig_id = get_extended_member_decoration(type.self, index, SPIRVCrossDecorationInterfaceOrigID);

	bool row_major = false;
	if (is_matrix(physical_type))
		row_major = has_member_decoration(type.self, index, DecorationRowMajor);

	SPIRType row_major_physical_type;
	const SPIRType *declared_type = &physical_type;

	// If a struct is being declared with physical layout,
	// do not use array<T> wrappers.
	// This avoids a lot of complicated cases with packed vectors and matrices,
	// and generally we cannot copy full arrays in and out of buffers into Function
	// address space.
	// Array of resources should also be declared as builtin arrays.
	if (has_member_decoration(type.self, index, DecorationOffset))
		is_using_builtin_array = true;
	else if (has_extended_member_decoration(type.self, index, SPIRVCrossDecorationResourceIndexPrimary))
		is_using_builtin_array = true;

	if (member_is_packed_physical_type(type, index))
	{
		// If we're packing a matrix, output an appropriate typedef
		if (physical_type.basetype == SPIRType::Struct)
		{
			SPIRV_CROSS_THROW("Cannot emit a packed struct currently.");
		}
		else if (is_matrix(physical_type))
		{
			uint32_t rows = physical_type.vecsize;
			uint32_t cols = physical_type.columns;
			pack_pfx = "packed_";
			if (row_major)
			{
				// These are stored transposed.
				rows = physical_type.columns;
				cols = physical_type.vecsize;
				pack_pfx = "packed_rm_";
			}
			string base_type = physical_type.width == 16 ? "half" : "float";
			string td_line = "typedef ";
			td_line += "packed_" + base_type + to_string(rows);
			td_line += " " + pack_pfx;
			// Use the actual matrix size here.
			td_line += base_type + to_string(physical_type.columns) + "x" + to_string(physical_type.vecsize);
			td_line += "[" + to_string(cols) + "]";
			td_line += ";";
			add_typedef_line(td_line);
		}
		else
			pack_pfx = "packed_";
	}
	else if (row_major)
	{
		// Need to declare type with flipped vecsize/columns.
		row_major_physical_type = physical_type;
		swap(row_major_physical_type.vecsize, row_major_physical_type.columns);
		declared_type = &row_major_physical_type;
	}

	// Very specifically, image load-store in argument buffers are disallowed on MSL on iOS.
	if (msl_options.is_ios() && physical_type.basetype == SPIRType::Image && physical_type.image.sampled == 2)
	{
		if (!has_decoration(orig_id, DecorationNonWritable))
			SPIRV_CROSS_THROW("Writable images are not allowed in argument buffers on iOS.");
	}

	// Array information is baked into these types.
	string array_type;
	if (physical_type.basetype != SPIRType::Image && physical_type.basetype != SPIRType::Sampler &&
	    physical_type.basetype != SPIRType::SampledImage)
	{
		BuiltIn builtin = BuiltInMax;
		if (is_member_builtin(type, index, &builtin))
			is_using_builtin_array = true;
		array_type = type_to_array_glsl(physical_type);
	}

	auto result = join(pack_pfx, type_to_glsl(*declared_type, orig_id), " ", qualifier, to_member_name(type, index),
	                   member_attribute_qualifier(type, index), array_type, ";");

	is_using_builtin_array = false;
	return result;
}

// Emit a structure member, padding and packing to maintain the correct memeber alignments.
void CompilerMSL::emit_struct_member(const SPIRType &type, uint32_t member_type_id, uint32_t index,
                                     const string &qualifier, uint32_t)
{
	// If this member requires padding to maintain its declared offset, emit a dummy padding member before it.
	if (has_extended_member_decoration(type.self, index, SPIRVCrossDecorationPaddingTarget))
	{
		uint32_t pad_len = get_extended_member_decoration(type.self, index, SPIRVCrossDecorationPaddingTarget);
		statement("char _m", index, "_pad", "[", pad_len, "];");
	}

	// Handle HLSL-style 0-based vertex/instance index.
	builtin_declaration = true;
	statement(to_struct_member(type, member_type_id, index, qualifier));
	builtin_declaration = false;
}

void CompilerMSL::emit_struct_padding_target(const SPIRType &type)
{
	uint32_t struct_size = get_declared_struct_size_msl(type, true, true);
	uint32_t target_size = get_extended_decoration(type.self, SPIRVCrossDecorationPaddingTarget);
	if (target_size < struct_size)
		SPIRV_CROSS_THROW("Cannot pad with negative bytes.");
	else if (target_size > struct_size)
		statement("char _m0_final_padding[", target_size - struct_size, "];");
}

// Return a MSL qualifier for the specified function attribute member
string CompilerMSL::member_attribute_qualifier(const SPIRType &type, uint32_t index)
{
	auto &execution = get_entry_point();

	uint32_t mbr_type_id = type.member_types[index];
	auto &mbr_type = get<SPIRType>(mbr_type_id);

	BuiltIn builtin = BuiltInMax;
	bool is_builtin = is_member_builtin(type, index, &builtin);

	if (has_extended_member_decoration(type.self, index, SPIRVCrossDecorationResourceIndexPrimary))
	{
		string quals = join(
		    " [[id(", get_extended_member_decoration(type.self, index, SPIRVCrossDecorationResourceIndexPrimary), ")");
		if (interlocked_resources.count(
		        get_extended_member_decoration(type.self, index, SPIRVCrossDecorationInterfaceOrigID)))
			quals += ", raster_order_group(0)";
		quals += "]]";
		return quals;
	}

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
				return string(" [[") + builtin_qualifier(builtin) + "]]" + (mbr_type.array.empty() ? "" : " ");

			case BuiltInClipDistance:
				if (has_member_decoration(type.self, index, DecorationLocation))
					return join(" [[user(clip", get_member_decoration(type.self, index, DecorationLocation), ")]]");
				else
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
		string quals;
		if (is_builtin)
		{
			switch (builtin)
			{
			case BuiltInViewIndex:
				if (!msl_options.multiview)
					break;
				/* fallthrough */
			case BuiltInFrontFacing:
			case BuiltInPointCoord:
			case BuiltInFragCoord:
			case BuiltInSampleId:
			case BuiltInSampleMask:
			case BuiltInLayer:
			case BuiltInBaryCoordNV:
			case BuiltInBaryCoordNoPerspNV:
				quals = builtin_qualifier(builtin);
				break;

			case BuiltInClipDistance:
				return join(" [[user(clip", get_member_decoration(type.self, index, DecorationLocation), ")]]");

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
				// For user-defined attributes, this is fine. From Vulkan spec:
				// A user-defined output variable is considered to match an input variable in the subsequent stage if
				// the two variables are declared with the same Location and Component decoration and match in type
				// and decoration, except that interpolation decorations are not required to match. For the purposes
				// of interface matching, variables declared without a Component decoration are considered to have a
				// Component decoration of zero.

				if (comp != k_unknown_component && comp != 0)
					quals = string("user(locn") + convert_to_string(locn) + "_" + convert_to_string(comp) + ")";
				else
					quals = string("user(locn") + convert_to_string(locn) + ")";
			}
		}

		if (builtin == BuiltInBaryCoordNV || builtin == BuiltInBaryCoordNoPerspNV)
		{
			if (has_member_decoration(type.self, index, DecorationFlat) ||
			    has_member_decoration(type.self, index, DecorationCentroid) ||
			    has_member_decoration(type.self, index, DecorationSample) ||
			    has_member_decoration(type.self, index, DecorationNoPerspective))
			{
				// NoPerspective is baked into the builtin type.
				SPIRV_CROSS_THROW(
				    "Flat, Centroid, Sample, NoPerspective decorations are not supported for BaryCoord inputs.");
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
			case BuiltInFragStencilRefEXT:
				// Similar to PointSize, only mark FragStencilRef if there's a stencil buffer.
				// Some shaders may include a FragStencilRef builtin even when used to render
				// without a stencil attachment, and Metal will reject this builtin
				// when compiling the shader into a render pipeline that does not set
				// stencilAttachmentPixelFormat.
				if (!msl_options.enable_frag_stencil_ref_builtin)
					return "";
				if (!msl_options.supports_msl_version(2, 1))
					SPIRV_CROSS_THROW("Stencil export only supported in MSL 2.1 and up.");
				return string(" [[") + builtin_qualifier(builtin) + "]]";

			case BuiltInFragDepth:
				// Ditto FragDepth.
				if (!msl_options.enable_frag_depth_builtin)
					return "";
				/* fallthrough */
			case BuiltInSampleMask:
				return string(" [[") + builtin_qualifier(builtin) + "]]";

			default:
				return "";
			}
		}
		uint32_t locn = get_ordered_member_location(type.self, index);
		// Metal will likely complain about missing color attachments, too.
		if (locn != k_unknown_location && !(msl_options.enable_frag_output_mask & (1 << locn)))
			return "";
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
		entry_type = execution.flags.get(ExecutionModeEarlyFragmentTests) ||
		                     execution.flags.get(ExecutionModePostDepthCoverage) ?
		                 "[[ early_fragment_tests ]] fragment" :
		                 "fragment";
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
	return get_type_address_space(type, argument.self, true);
}

string CompilerMSL::get_type_address_space(const SPIRType &type, uint32_t id, bool argument)
{
	// This can be called for variable pointer contexts as well, so be very careful about which method we choose.
	Bitset flags;
	auto *var = maybe_get<SPIRVariable>(id);
	if (var && type.basetype == SPIRType::Struct &&
	    (has_decoration(type.self, DecorationBlock) || has_decoration(type.self, DecorationBufferBlock)))
		flags = get_buffer_block_flags(id);
	else
		flags = get_decoration_bitset(id);

	const char *addr_space = nullptr;
	switch (type.storage)
	{
	case StorageClassWorkgroup:
		addr_space = "threadgroup";
		break;

	case StorageClassStorageBuffer:
	{
		// For arguments from variable pointers, we use the write count deduction, so
		// we should not assume any constness here. Only for global SSBOs.
		bool readonly = false;
		if (!var || has_decoration(type.self, DecorationBlock))
			readonly = flags.get(DecorationNonWritable);

		addr_space = readonly ? "const device" : "device";
		break;
	}

	case StorageClassUniform:
	case StorageClassUniformConstant:
	case StorageClassPushConstant:
		if (type.basetype == SPIRType::Struct)
		{
			bool ssbo = has_decoration(type.self, DecorationBufferBlock);
			if (ssbo)
				addr_space = flags.get(DecorationNonWritable) ? "const device" : "device";
			else
				addr_space = "constant";
		}
		else if (!argument)
			addr_space = "constant";
		break;

	case StorageClassFunction:
	case StorageClassGeneric:
		break;

	case StorageClassInput:
		if (get_execution_model() == ExecutionModelTessellationControl && var &&
		    var->basevariable == stage_in_ptr_var_id)
			addr_space = "threadgroup";
		break;

	case StorageClassOutput:
		if (capture_output_to_buffer)
			addr_space = "device";
		break;

	default:
		break;
	}

	if (!addr_space)
		// No address space for plain values.
		addr_space = type.pointer || (argument && type.basetype == SPIRType::ControlPointArray) ? "thread" : "";

	return join(flags.get(DecorationVolatile) || flags.get(DecorationCoherent) ? "volatile " : "", addr_space);
}

const char *CompilerMSL::to_restrict(uint32_t id, bool space)
{
	// This can be called for variable pointer contexts as well, so be very careful about which method we choose.
	Bitset flags;
	if (ir.ids[id].get_type() == TypeVariable)
	{
		uint32_t type_id = expression_type_id(id);
		auto &type = expression_type(id);
		if (type.basetype == SPIRType::Struct &&
		    (has_decoration(type_id, DecorationBlock) || has_decoration(type_id, DecorationBufferBlock)))
			flags = get_buffer_block_flags(id);
		else
			flags = get_decoration_bitset(id);
	}
	else
		flags = get_decoration_bitset(id);

	return flags.get(DecorationRestrict) ? (space ? "restrict " : "restrict") : "";
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
	SmallVector<pair<SPIRVariable *, BuiltIn>, 8> active_builtins;
	ir.for_each_typed_id<SPIRVariable>([&](uint32_t var_id, SPIRVariable &var) {
		auto bi_type = BuiltIn(get_decoration(var_id, DecorationBuiltIn));

		// Don't emit SamplePosition as a separate parameter. In the entry
		// point, we get that by calling get_sample_position() on the sample ID.
		if (var.storage == StorageClassInput && is_builtin_variable(var) &&
		    get_variable_data_type(var).basetype != SPIRType::Struct &&
		    get_variable_data_type(var).basetype != SPIRType::ControlPointArray)
		{
			// If the builtin is not part of the active input builtin set, don't emit it.
			// Relevant for multiple entry-point modules which might declare unused builtins.
			if (!active_input_builtins.get(bi_type) || !interface_variable_exists_in_entry_point(var_id))
				return;

			// Remember this variable. We may need to correct its type.
			active_builtins.push_back(make_pair(&var, bi_type));

			// These builtins are emitted specially. If we pass this branch, the builtin directly matches
			// a MSL builtin.
			if (bi_type != BuiltInSamplePosition && bi_type != BuiltInHelperInvocation &&
			    bi_type != BuiltInPatchVertices && bi_type != BuiltInTessLevelInner &&
			    bi_type != BuiltInTessLevelOuter && bi_type != BuiltInPosition && bi_type != BuiltInPointSize &&
			    bi_type != BuiltInClipDistance && bi_type != BuiltInCullDistance && bi_type != BuiltInSubgroupEqMask &&
			    bi_type != BuiltInBaryCoordNV && bi_type != BuiltInBaryCoordNoPerspNV &&
			    bi_type != BuiltInSubgroupGeMask && bi_type != BuiltInSubgroupGtMask &&
			    bi_type != BuiltInSubgroupLeMask && bi_type != BuiltInSubgroupLtMask && bi_type != BuiltInDeviceIndex &&
			    ((get_execution_model() == ExecutionModelFragment && msl_options.multiview) ||
			     bi_type != BuiltInViewIndex) &&
			    (get_execution_model() == ExecutionModelGLCompute ||
			     (get_execution_model() == ExecutionModelFragment && msl_options.supports_msl_version(2, 2)) ||
			     (bi_type != BuiltInSubgroupLocalInvocationId && bi_type != BuiltInSubgroupSize)))
			{
				if (!ep_args.empty())
					ep_args += ", ";

				// Handle HLSL-style 0-based vertex/instance index.
				builtin_declaration = true;
				ep_args += builtin_type_decl(bi_type, var_id) + " " + to_expression(var_id);
				ep_args += " [[" + builtin_qualifier(bi_type);
				if (bi_type == BuiltInSampleMask && get_entry_point().flags.get(ExecutionModePostDepthCoverage))
				{
					if (!msl_options.supports_msl_version(2))
						SPIRV_CROSS_THROW("Post-depth coverage requires Metal 2.0.");
					if (!msl_options.is_ios())
						SPIRV_CROSS_THROW("Post-depth coverage is only supported on iOS.");
					ep_args += ", post_depth_coverage";
				}
				ep_args += "]]";
				builtin_declaration = false;
			}
		}

		if (var.storage == StorageClassInput &&
		    has_extended_decoration(var_id, SPIRVCrossDecorationBuiltInDispatchBase))
		{
			// This is a special implicit builtin, not corresponding to any SPIR-V builtin,
			// which holds the base that was passed to vkCmdDispatchBase(). If it's present,
			// assume we emitted it for a good reason.
			assert(msl_options.supports_msl_version(1, 2));
			if (!ep_args.empty())
				ep_args += ", ";

			ep_args += type_to_glsl(get_variable_data_type(var)) + " " + to_expression(var_id) + " [[grid_origin]]";
		}
	});

	// Correct the types of all encountered active builtins. We couldn't do this before
	// because ensure_correct_builtin_type() may increase the bound, which isn't allowed
	// while iterating over IDs.
	for (auto &var : active_builtins)
		var.first->basetype = ensure_correct_builtin_type(var.first->basetype, var.second);

	// Handle HLSL-style 0-based vertex/instance index.
	if (needs_base_vertex_arg == TriState::Yes)
		ep_args += built_in_func_arg(BuiltInBaseVertex, !ep_args.empty());

	if (needs_base_instance_arg == TriState::Yes)
		ep_args += built_in_func_arg(BuiltInBaseInstance, !ep_args.empty());

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
	Bitset claimed_bindings;

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

		// Check if the argument buffer binding itself has been remapped.
		uint32_t buffer_binding;
		auto itr = resource_bindings.find({ get_entry_point().model, i, kArgumentBufferBinding });
		if (itr != end(resource_bindings))
		{
			buffer_binding = itr->second.first.msl_buffer;
			itr->second.second = true;
		}
		else
		{
			// As a fallback, directly map desc set <-> binding.
			// If that was taken, take the next buffer binding.
			if (claimed_bindings.get(i))
				buffer_binding = next_metal_resource_index_buffer;
			else
				buffer_binding = i;
		}

		claimed_bindings.set(buffer_binding);

		ep_args += get_argument_address_space(var) + " " + type_to_glsl(type) + "& " + to_restrict(id) + to_name(id);
		ep_args += " [[buffer(" + convert_to_string(buffer_binding) + ")]]";

		next_metal_resource_index_buffer = max(next_metal_resource_index_buffer, buffer_binding + 1);
	}

	entry_point_args_discrete_descriptors(ep_args);
	entry_point_args_builtin(ep_args);

	if (!ep_args.empty() && append_comma)
		ep_args += ", ";

	return ep_args;
}

const MSLConstexprSampler *CompilerMSL::find_constexpr_sampler(uint32_t id) const
{
	// Try by ID.
	{
		auto itr = constexpr_samplers_by_id.find(id);
		if (itr != end(constexpr_samplers_by_id))
			return &itr->second;
	}

	// Try by binding.
	{
		uint32_t desc_set = get_decoration(id, DecorationDescriptorSet);
		uint32_t binding = get_decoration(id, DecorationBinding);

		auto itr = constexpr_samplers_by_binding.find({ desc_set, binding });
		if (itr != end(constexpr_samplers_by_binding))
			return &itr->second;
	}

	return nullptr;
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
		uint32_t plane;
		uint32_t secondary_index;
	};

	SmallVector<Resource> resources;

	ir.for_each_typed_id<SPIRVariable>([&](uint32_t var_id, SPIRVariable &var) {
		if ((var.storage == StorageClassUniform || var.storage == StorageClassUniformConstant ||
		     var.storage == StorageClassPushConstant || var.storage == StorageClassStorageBuffer) &&
		    !is_hidden_variable(var))
		{
			auto &type = get_variable_data_type(var);

			// Very specifically, image load-store in argument buffers are disallowed on MSL on iOS.
			// But we won't know when the argument buffer is encoded whether this image will have
			// a NonWritable decoration. So just use discrete arguments for all storage images
			// on iOS.
			if (!(msl_options.is_ios() && type.basetype == SPIRType::Image && type.image.sampled == 2) &&
			    var.storage != StorageClassPushConstant)
			{
				uint32_t desc_set = get_decoration(var_id, DecorationDescriptorSet);
				if (descriptor_set_is_argument_buffer(desc_set))
					return;
			}

			const MSLConstexprSampler *constexpr_sampler = nullptr;
			if (type.basetype == SPIRType::SampledImage || type.basetype == SPIRType::Sampler)
			{
				constexpr_sampler = find_constexpr_sampler(var_id);
				if (constexpr_sampler)
				{
					// Mark this ID as a constexpr sampler for later in case it came from set/bindings.
					constexpr_samplers_by_id[var_id] = *constexpr_sampler;
				}
			}

			// Emulate texture2D atomic operations
			uint32_t secondary_index = 0;
			if (atomic_image_vars.count(var.self))
			{
				secondary_index = get_metal_resource_index(var, SPIRType::AtomicCounter, 0);
			}

			if (type.basetype == SPIRType::SampledImage)
			{
				add_resource_name(var_id);

				uint32_t plane_count = 1;
				if (constexpr_sampler && constexpr_sampler->ycbcr_conversion_enable)
					plane_count = constexpr_sampler->planes;

				for (uint32_t i = 0; i < plane_count; i++)
					resources.push_back({ &var, to_name(var_id), SPIRType::Image,
					                      get_metal_resource_index(var, SPIRType::Image, i), i, secondary_index });

				if (type.image.dim != DimBuffer && !constexpr_sampler)
				{
					resources.push_back({ &var, to_sampler_expression(var_id), SPIRType::Sampler,
					                      get_metal_resource_index(var, SPIRType::Sampler), 0, 0 });
				}
			}
			else if (!constexpr_sampler)
			{
				// constexpr samplers are not declared as resources.
				add_resource_name(var_id);
				resources.push_back({ &var, to_name(var_id), type.basetype,
				                      get_metal_resource_index(var, type.basetype), 0, secondary_index });
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

				// Allow Metal to use the array<T> template to make arrays a value type
				is_using_builtin_array = true;
				buffer_arrays.push_back(var_id);
				for (uint32_t i = 0; i < array_size; ++i)
				{
					if (!ep_args.empty())
						ep_args += ", ";
					ep_args += get_argument_address_space(var) + " " + type_to_glsl(type) + "* " + to_restrict(var_id) +
					           r.name + "_" + convert_to_string(i);
					ep_args += " [[buffer(" + convert_to_string(r.index + i) + ")";
					if (interlocked_resources.count(var_id))
						ep_args += ", raster_order_group(0)";
					ep_args += "]]";
				}
				is_using_builtin_array = false;
			}
			else
			{
				if (!ep_args.empty())
					ep_args += ", ";
				ep_args +=
				    get_argument_address_space(var) + " " + type_to_glsl(type) + "& " + to_restrict(var_id) + r.name;
				ep_args += " [[buffer(" + convert_to_string(r.index) + ")";
				if (interlocked_resources.count(var_id))
					ep_args += ", raster_order_group(0)";
				ep_args += "]]";
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
		{
			if (!ep_args.empty())
				ep_args += ", ";

			// Use Metal's native frame-buffer fetch API for subpass inputs.
			const auto &basetype = get<SPIRType>(var.basetype);
			if (basetype.image.dim != DimSubpassData || !msl_options.is_ios() ||
			    !msl_options.ios_use_framebuffer_fetch_subpasses)
			{
				ep_args += image_type_glsl(type, var_id) + " " + r.name;
				if (r.plane > 0)
					ep_args += join(plane_name_suffix, r.plane);
				ep_args += " [[texture(" + convert_to_string(r.index) + ")";
				if (interlocked_resources.count(var_id))
					ep_args += ", raster_order_group(0)";
				ep_args += "]]";
			}
			else
			{
				ep_args += image_type_glsl(type, var_id) + "4 " + r.name;
				ep_args += " [[color(" + convert_to_string(r.index) + ")]]";
			}

			// Emulate texture2D atomic operations
			if (atomic_image_vars.count(var.self))
			{
				ep_args += ", device atomic_" + type_to_glsl(get<SPIRType>(basetype.image.type), 0);
				ep_args += "* " + r.name + "_atomic";
				ep_args += " [[buffer(" + convert_to_string(r.secondary_index) + ")]]";
			}
			break;
		}
		default:
			if (!ep_args.empty())
				ep_args += ", ";
			if (!type.pointer)
				ep_args += get_type_address_space(get<SPIRType>(var.basetype), var_id) + " " +
				           type_to_glsl(type, var_id) + "& " + r.name;
			else
				ep_args += type_to_glsl(type, var_id) + " " + r.name;
			ep_args += " [[buffer(" + convert_to_string(r.index) + ")";
			if (interlocked_resources.count(var_id))
				ep_args += ", raster_order_group(0)";
			ep_args += "]]";
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
	// Look for sampled images and buffer. Add hooks to set up the swizzle constants or array lengths.
	ir.for_each_typed_id<SPIRVariable>([&](uint32_t, SPIRVariable &var) {
		auto &type = get_variable_data_type(var);
		uint32_t var_id = var.self;
		bool ssbo = has_decoration(type.self, DecorationBufferBlock);

		if (var.storage == StorageClassUniformConstant && !is_hidden_variable(var))
		{
			if (msl_options.swizzle_texture_samples && has_sampled_images && is_sampled_image_type(type))
			{
				auto &entry_func = this->get<SPIRFunction>(ir.default_entry_point);
				entry_func.fixup_hooks_in.push_back([this, &type, &var, var_id]() {
					bool is_array_type = !type.array.empty();

					uint32_t desc_set = get_decoration(var_id, DecorationDescriptorSet);
					if (descriptor_set_is_argument_buffer(desc_set))
					{
						statement("constant uint", is_array_type ? "* " : "& ", to_swizzle_expression(var_id),
						          is_array_type ? " = &" : " = ", to_name(argument_buffer_ids[desc_set]),
						          ".spvSwizzleConstants", "[",
						          convert_to_string(get_metal_resource_index(var, SPIRType::Image)), "];");
					}
					else
					{
						// If we have an array of images, we need to be able to index into it, so take a pointer instead.
						statement("constant uint", is_array_type ? "* " : "& ", to_swizzle_expression(var_id),
						          is_array_type ? " = &" : " = ", to_name(swizzle_buffer_id), "[",
						          convert_to_string(get_metal_resource_index(var, SPIRType::Image)), "];");
					}
				});
			}
		}
		else if ((var.storage == StorageClassStorageBuffer || (var.storage == StorageClassUniform && ssbo)) &&
		         !is_hidden_variable(var))
		{
			if (buffers_requiring_array_length.count(var.self))
			{
				auto &entry_func = this->get<SPIRFunction>(ir.default_entry_point);
				entry_func.fixup_hooks_in.push_back([this, &type, &var, var_id]() {
					bool is_array_type = !type.array.empty();

					uint32_t desc_set = get_decoration(var_id, DecorationDescriptorSet);
					if (descriptor_set_is_argument_buffer(desc_set))
					{
						statement("constant uint", is_array_type ? "* " : "& ", to_buffer_size_expression(var_id),
						          is_array_type ? " = &" : " = ", to_name(argument_buffer_ids[desc_set]),
						          ".spvBufferSizeConstants", "[",
						          convert_to_string(get_metal_resource_index(var, SPIRType::Image)), "];");
					}
					else
					{
						// If we have an array of images, we need to be able to index into it, so take a pointer instead.
						statement("constant uint", is_array_type ? "* " : "& ", to_buffer_size_expression(var_id),
						          is_array_type ? " = &" : " = ", to_name(buffer_size_buffer_id), "[",
						          convert_to_string(get_metal_resource_index(var, type.basetype)), "];");
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
			case BuiltInSubgroupLocalInvocationId:
				// This is natively supported in compute shaders.
				if (get_execution_model() == ExecutionModelGLCompute)
					break;

				// This is natively supported in fragment shaders in MSL 2.2.
				if (get_execution_model() == ExecutionModelFragment && msl_options.supports_msl_version(2, 2))
					break;

				if (msl_options.is_ios())
					SPIRV_CROSS_THROW(
					    "SubgroupLocalInvocationId cannot be used outside of compute shaders before MSL 2.2 on iOS.");

				if (!msl_options.supports_msl_version(2, 1))
					SPIRV_CROSS_THROW(
					    "SubgroupLocalInvocationId cannot be used outside of compute shaders before MSL 2.1.");

				// Shaders other than compute shaders don't support the SIMD-group
				// builtins directly, but we can emulate them using the SIMD-group
				// functions. This might break if some of the subgroup terminated
				// before reaching the entry point.
				entry_func.fixup_hooks_in.push_back([=]() {
					statement(builtin_type_decl(bi_type), " ", to_expression(var_id),
					          " = simd_prefix_exclusive_sum(1);");
				});
				break;
			case BuiltInSubgroupSize:
				// This is natively supported in compute shaders.
				if (get_execution_model() == ExecutionModelGLCompute)
					break;

				// This is natively supported in fragment shaders in MSL 2.2.
				if (get_execution_model() == ExecutionModelFragment && msl_options.supports_msl_version(2, 2))
					break;

				if (msl_options.is_ios())
					SPIRV_CROSS_THROW("SubgroupSize cannot be used outside of compute shaders on iOS.");

				if (!msl_options.supports_msl_version(2, 1))
					SPIRV_CROSS_THROW("SubgroupSize cannot be used outside of compute shaders before Metal 2.1.");

				entry_func.fixup_hooks_in.push_back(
				    [=]() { statement(builtin_type_decl(bi_type), " ", to_expression(var_id), " = simd_sum(1);"); });
				break;
			case BuiltInSubgroupEqMask:
				if (msl_options.is_ios())
					SPIRV_CROSS_THROW("Subgroup ballot functionality is unavailable on iOS.");
				if (!msl_options.supports_msl_version(2, 1))
					SPIRV_CROSS_THROW("Subgroup ballot functionality requires Metal 2.1.");
				entry_func.fixup_hooks_in.push_back([=]() {
					statement(builtin_type_decl(bi_type), " ", to_expression(var_id), " = ",
					          to_expression(builtin_subgroup_invocation_id_id), " > 32 ? uint4(0, (1 << (",
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
			case BuiltInViewIndex:
				if (!msl_options.multiview)
				{
					// According to the Vulkan spec, when not running under a multiview
					// render pass, ViewIndex is 0.
					entry_func.fixup_hooks_in.push_back([=]() {
						statement("const ", builtin_type_decl(bi_type), " ", to_expression(var_id), " = 0;");
					});
				}
				else if (msl_options.view_index_from_device_index)
				{
					// In this case, we take the view index from that of the device we're running on.
					entry_func.fixup_hooks_in.push_back([=]() {
						statement("const ", builtin_type_decl(bi_type), " ", to_expression(var_id), " = ",
						          msl_options.device_index, ";");
					});
					// We actually don't want to set the render_target_array_index here.
					// Since every physical device is rendering a different view,
					// there's no need for layered rendering here.
				}
				else if (get_execution_model() == ExecutionModelFragment)
				{
					// Because we adjusted the view index in the vertex shader, we have to
					// adjust it back here.
					entry_func.fixup_hooks_in.push_back([=]() {
						statement(to_expression(var_id), " += ", to_expression(view_mask_buffer_id), "[0];");
					});
				}
				else if (get_execution_model() == ExecutionModelVertex)
				{
					// Metal provides no special support for multiview, so we smuggle
					// the view index in the instance index.
					entry_func.fixup_hooks_in.push_back([=]() {
						statement(builtin_type_decl(bi_type), " ", to_expression(var_id), " = ",
						          to_expression(view_mask_buffer_id), "[0] + ", to_expression(builtin_instance_idx_id),
						          " % ", to_expression(view_mask_buffer_id), "[1];");
						statement(to_expression(builtin_instance_idx_id), " /= ", to_expression(view_mask_buffer_id),
						          "[1];");
					});
					// In addition to setting the variable itself, we also need to
					// set the render_target_array_index with it on output. We have to
					// offset this by the base view index, because Metal isn't in on
					// our little game here.
					entry_func.fixup_hooks_out.push_back([=]() {
						statement(to_expression(builtin_layer_id), " = ", to_expression(var_id), " - ",
						          to_expression(view_mask_buffer_id), "[0];");
					});
				}
				break;
			case BuiltInDeviceIndex:
				// Metal pipelines belong to the devices which create them, so we'll
				// need to create a MTLPipelineState for every MTLDevice in a grouped
				// VkDevice. We can assume, then, that the device index is constant.
				entry_func.fixup_hooks_in.push_back([=]() {
					statement("const ", builtin_type_decl(bi_type), " ", to_expression(var_id), " = ",
					          msl_options.device_index, ";");
				});
				break;
			case BuiltInWorkgroupId:
				if (!msl_options.dispatch_base || !active_input_builtins.get(BuiltInWorkgroupId))
					break;

				// The vkCmdDispatchBase() command lets the client set the base value
				// of WorkgroupId. Metal has no direct equivalent; we must make this
				// adjustment ourselves.
				entry_func.fixup_hooks_in.push_back([=]() {
					statement(to_expression(var_id), " += ", to_dereferenced_expression(builtin_dispatch_base_id), ";");
				});
				break;
			case BuiltInGlobalInvocationId:
				if (!msl_options.dispatch_base || !active_input_builtins.get(BuiltInGlobalInvocationId))
					break;

				// GlobalInvocationId is defined as LocalInvocationId + WorkgroupId * WorkgroupSize.
				// This needs to be adjusted too.
				entry_func.fixup_hooks_in.push_back([=]() {
					auto &execution = this->get_entry_point();
					uint32_t workgroup_size_id = execution.workgroup_size.constant;
					if (workgroup_size_id)
						statement(to_expression(var_id), " += ", to_dereferenced_expression(builtin_dispatch_base_id),
						          " * ", to_expression(workgroup_size_id), ";");
					else
						statement(to_expression(var_id), " += ", to_dereferenced_expression(builtin_dispatch_base_id),
						          " * uint3(", execution.workgroup_size.x, ", ", execution.workgroup_size.y, ", ",
						          execution.workgroup_size.z, ");");
				});
				break;
			default:
				break;
			}
		}
	});
}

// Returns the Metal index of the resource of the specified type as used by the specified variable.
uint32_t CompilerMSL::get_metal_resource_index(SPIRVariable &var, SPIRType::BaseType basetype, uint32_t plane)
{
	auto &execution = get_entry_point();
	auto &var_dec = ir.meta[var.self].decoration;
	auto &var_type = get<SPIRType>(var.basetype);
	uint32_t var_desc_set = (var.storage == StorageClassPushConstant) ? kPushConstDescSet : var_dec.set;
	uint32_t var_binding = (var.storage == StorageClassPushConstant) ? kPushConstBinding : var_dec.binding;

	// If a matching binding has been specified, find and use it.
	auto itr = resource_bindings.find({ execution.model, var_desc_set, var_binding });

	// Atomic helper buffers for image atomics need to use secondary bindings as well.
	bool use_secondary_binding = (var_type.basetype == SPIRType::SampledImage && basetype == SPIRType::Sampler) ||
	                             basetype == SPIRType::AtomicCounter;

	auto resource_decoration =
	    use_secondary_binding ? SPIRVCrossDecorationResourceIndexSecondary : SPIRVCrossDecorationResourceIndexPrimary;

	if (plane == 1)
		resource_decoration = SPIRVCrossDecorationResourceIndexTertiary;
	if (plane == 2)
		resource_decoration = SPIRVCrossDecorationResourceIndexQuaternary;

	if (itr != end(resource_bindings))
	{
		auto &remap = itr->second;
		remap.second = true;
		switch (basetype)
		{
		case SPIRType::Image:
			set_extended_decoration(var.self, resource_decoration, remap.first.msl_texture + plane);
			return remap.first.msl_texture + plane;
		case SPIRType::Sampler:
			set_extended_decoration(var.self, resource_decoration, remap.first.msl_sampler);
			return remap.first.msl_sampler;
		default:
			set_extended_decoration(var.self, resource_decoration, remap.first.msl_buffer);
			return remap.first.msl_buffer;
		}
	}

	// If we have already allocated an index, keep using it.
	if (has_extended_decoration(var.self, resource_decoration))
		return get_extended_decoration(var.self, resource_decoration);

	// Allow user to enable decoration binding
	if (msl_options.enable_decoration_binding)
	{
		// If there is no explicit mapping of bindings to MSL, use the declared binding.
		if (has_decoration(var.self, DecorationBinding))
		{
			var_binding = get_decoration(var.self, DecorationBinding);
			// Avoid emitting sentinel bindings.
			if (var_binding < 0x80000000u)
				return var_binding;
		}
	}

	// If we did not explicitly remap, allocate bindings on demand.
	// We cannot reliably use Binding decorations since SPIR-V and MSL's binding models are very different.

	bool allocate_argument_buffer_ids = false;

	if (var.storage != StorageClassPushConstant)
		allocate_argument_buffer_ids = descriptor_set_is_argument_buffer(var_desc_set);

	uint32_t binding_stride = 1;
	auto &type = get<SPIRType>(var.basetype);
	for (uint32_t i = 0; i < uint32_t(type.array.size()); i++)
		binding_stride *= to_array_size_literal(type, i);

	assert(binding_stride != 0);

	// If a binding has not been specified, revert to incrementing resource indices.
	uint32_t resource_index;

	if (allocate_argument_buffer_ids)
	{
		// Allocate from a flat ID binding space.
		resource_index = next_metal_resource_ids[var_desc_set];
		next_metal_resource_ids[var_desc_set] += binding_stride;
	}
	else
	{
		// Allocate from plain bindings which are allocated per resource type.
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
	}

	set_extended_decoration(var.self, resource_decoration, resource_index);
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

	// If this is a combined image-sampler for a 2D image with floating-point type,
	// we emitted the 'spvDynamicImageSampler' type, and this is *not* an alias parameter
	// for a global, then we need to emit a "dynamic" combined image-sampler.
	// Unfortunately, this is necessary to properly support passing around
	// combined image-samplers with Y'CbCr conversions on them.
	bool is_dynamic_img_sampler = !arg.alias_global_variable && type.basetype == SPIRType::SampledImage &&
	                              type.image.dim == Dim2D && type_is_floating_point(get<SPIRType>(type.image.type)) &&
	                              spv_function_implementations.count(SPVFuncImplDynamicImageSampler);

	// Allow Metal to use the array<T> template to make arrays a value type
	string address_space = get_argument_address_space(var);
	bool builtin = is_builtin_variable(var);
	is_using_builtin_array = builtin;
	if (address_space == "threadgroup")
		is_using_builtin_array = true;

	if (var.basevariable && (var.basevariable == stage_in_ptr_var_id || var.basevariable == stage_out_ptr_var_id))
		decl += type_to_glsl(type, arg.id);
	else if (builtin)
		decl += builtin_type_decl(static_cast<BuiltIn>(get_decoration(arg.id, DecorationBuiltIn)), arg.id);
	else if ((storage == StorageClassUniform || storage == StorageClassStorageBuffer) && is_array(type))
	{
		is_using_builtin_array = true;
		decl += join(type_to_glsl(type, arg.id), "*");
	}
	else if (is_dynamic_img_sampler)
	{
		decl += join("spvDynamicImageSampler<", type_to_glsl(get<SPIRType>(type.image.type)), ">");
		// Mark the variable so that we can handle passing it to another function.
		set_extended_decoration(arg.id, SPIRVCrossDecorationDynamicImageSampler);
	}
	else
		decl += type_to_glsl(type, arg.id);

	bool opaque_handle = storage == StorageClassUniformConstant;

	if (!builtin && !opaque_handle && !is_pointer &&
	    (storage == StorageClassFunction || storage == StorageClassGeneric))
	{
		// If the argument is a pure value and not an opaque type, we will pass by value.
		if (msl_options.force_native_arrays && is_array(type))
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
			const char *restrict_kw = to_restrict(name_id);
			if (*restrict_kw)
			{
				decl += " ";
				decl += restrict_kw;
			}
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
			uint32_t desc_set = get_decoration(name_id, DecorationDescriptorSet);
			if ((storage == StorageClassUniform || storage == StorageClassStorageBuffer) &&
			    descriptor_set_is_argument_buffer(desc_set))
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

				// If the argument was marked as being in device address space, any pointer to member would
				// be const device, not constant.
				if (argument_buffer_device_storage_mask & (1u << desc_set))
					decl += " const device";
				else
					decl += " constant";
			}
		}

		decl += " (&";
		const char *restrict_kw = to_restrict(name_id);
		if (*restrict_kw)
		{
			decl += " ";
			decl += restrict_kw;
		}
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
		decl += to_restrict(name_id);
		decl += to_expression(name_id);
	}
	else
	{
		if (!address_space.empty())
			decl = join(address_space, " ", decl);
		decl += " ";
		decl += to_expression(name_id);
	}

	// Emulate texture2D atomic operations
	auto *backing_var = maybe_get_backing_variable(name_id);
	if (backing_var && atomic_image_vars.count(backing_var->self))
	{
		decl += ", device atomic_" + type_to_glsl(get<SPIRType>(var_type.image.type), 0);
		decl += "* " + to_expression(name_id) + "_atomic";
	}

	is_using_builtin_array = false;

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
		auto *meta = ir.find_meta(self);
		if (!meta)
			return;

		auto &dec = meta->decoration;
		if (keywords.find(dec.alias) != end(keywords))
			dec.alias += "0";
	});

	ir.for_each_typed_id<SPIRFunction>([&](uint32_t self, SPIRFunction &) {
		auto *meta = ir.find_meta(self);
		if (!meta)
			return;

		auto &dec = meta->decoration;
		if (illegal_func_names.find(dec.alias) != end(illegal_func_names))
			dec.alias += "0";
	});

	ir.for_each_typed_id<SPIRType>([&](uint32_t self, SPIRType &) {
		auto *meta = ir.find_meta(self);
		if (!meta)
			return;

		for (auto &mbr_dec : meta->members)
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
		// Only allow -> dereference for block types. This is so we get expressions like
		// buffer[i]->first_member.second_member, rather than buffer[i]->first->second.
		bool is_block = has_decoration(type.self, DecorationBlock) || has_decoration(type.self, DecorationBufferBlock);

		bool is_buffer_variable =
		    is_block && (var->storage == StorageClassUniform || var->storage == StorageClassStorageBuffer);
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
		const char *restrict_kw;
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
			restrict_kw = to_restrict(id);
			if (*restrict_kw)
			{
				type_name += " ";
				type_name += restrict_kw;
			}
			break;
		}
		return type_name;
	}

	switch (type.basetype)
	{
	case SPIRType::Struct:
		// Need OpName lookup here to get a "sensible" name for a struct.
		// Allow Metal to use the array<T> template to make arrays a value type
		type_name = to_name(type.self);
		break;

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
		if (!msl_options.supports_msl_version(2, 2))
			SPIRV_CROSS_THROW("64-bit integers are only supported in MSL 2.2 and above.");
		type_name = "long";
		break;
	case SPIRType::UInt64:
		if (!msl_options.supports_msl_version(2, 2))
			SPIRV_CROSS_THROW("64-bit integers are only supported in MSL 2.2 and above.");
		type_name = "ulong";
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

	if (type.array.empty() || using_builtin_array())
	{
		return type_name;
	}
	else
	{
		// Allow Metal to use the array<T> template to make arrays a value type
		add_spv_func_and_recompile(SPVFuncImplUnsafeArray);
		string res;
		string sizes;

		for (uint32_t i = 0; i < uint32_t(type.array.size()); i++)
		{
			res += "spvUnsafeArray<";
			sizes += ", ";
			sizes += to_array_size(type, i);
			sizes += ">";
		}

		res += type_name + sizes;
		return res;
	}
}

string CompilerMSL::type_to_array_glsl(const SPIRType &type)
{
	// Allow Metal to use the array<T> template to make arrays a value type
	switch (type.basetype)
	{
	case SPIRType::AtomicCounter:
	case SPIRType::ControlPointArray:
	{
		return CompilerGLSL::type_to_array_glsl(type);
	}
	default:
	{
		if (using_builtin_array())
			return CompilerGLSL::type_to_array_glsl(type);
		else
			return "";
	}
	}
}

// Threadgroup arrays can't have a wrapper type
std::string CompilerMSL::variable_decl(const SPIRVariable &variable)
{
	if (variable.storage == StorageClassWorkgroup)
	{
		is_using_builtin_array = true;
	}
	std::string expr = CompilerGLSL::variable_decl(variable);
	if (variable.storage == StorageClassWorkgroup)
	{
		is_using_builtin_array = false;
	}
	return expr;
}

// GCC workaround of lambdas calling protected funcs
std::string CompilerMSL::variable_decl(const SPIRType &type, const std::string &name, uint32_t id)
{
	return CompilerGLSL::variable_decl(type, name, id);
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
		case Dim2D:
			if (img_type.dim == Dim1D && !msl_options.texture_1D_as_2D)
			{
				// Use a native Metal 1D texture
				img_type_name += "depth1d_unsupported_by_metal";
				break;
			}

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
			if (!msl_options.emulate_cube_array)
				img_type_name += (img_type.arrayed ? "depthcube_array" : "depthcube");
			else
				img_type_name += (img_type.arrayed ? "depth2d_array" : "depthcube");
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
		case Dim1D:
		case Dim2D:
		case DimSubpassData:
			if (img_type.dim == Dim1D && !msl_options.texture_1D_as_2D)
			{
				// Use a native Metal 1D texture
				img_type_name += (img_type.arrayed ? "texture1d_array" : "texture1d");
				break;
			}

			// Use Metal's native frame-buffer fetch API for subpass inputs.
			if (img_type.dim == DimSubpassData && msl_options.is_ios() &&
			    msl_options.ios_use_framebuffer_fetch_subpasses)
			{
				return type_to_glsl(get<SPIRType>(img_type.type));
			}
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
			if (!msl_options.emulate_cube_array)
				img_type_name += (img_type.arrayed ? "texturecube_array" : "texturecube");
			else
				img_type_name += (img_type.arrayed ? "texture2d_array" : "texturecube");
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

	// If we need to do implicit bitcasts, make sure we do it with the correct type.
	uint32_t integer_width = get_integer_width_for_instruction(i);
	auto int_type = to_signed_basetype(integer_width);
	auto uint_type = to_unsigned_basetype(integer_width);

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
			SPIRV_CROSS_THROW("Subgroup ops beyond broadcast and shuffle on macOS require Metal 2.1 and up.");
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

#define MSL_GROUP_OP_CAST(op, msl_op, type) \
case OpGroupNonUniform##op: \
	{ \
		auto operation = static_cast<GroupOperation>(ops[3]); \
		if (operation == GroupOperationReduce) \
			emit_unary_func_op_cast(result_type, id, ops[4], "simd_" #msl_op, type, type); \
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
			emit_unary_func_op_cast(result_type, id, ops[4], "quad_" #msl_op, type, type); \
		} \
		else \
			SPIRV_CROSS_THROW("Invalid group operation."); \
		break; \
	}

	MSL_GROUP_OP(FMin, min)
	MSL_GROUP_OP(FMax, max)
	MSL_GROUP_OP_CAST(SMin, min, int_type)
	MSL_GROUP_OP_CAST(SMax, max, int_type)
	MSL_GROUP_OP_CAST(UMin, min, uint_type)
	MSL_GROUP_OP_CAST(UMax, max, uint_type)
	MSL_GROUP_OP(BitwiseAnd, and)
	MSL_GROUP_OP(BitwiseOr, or)
	MSL_GROUP_OP(BitwiseXor, xor)
	MSL_GROUP_OP(LogicalAnd, and)
	MSL_GROUP_OP(LogicalOr, or)
	MSL_GROUP_OP(LogicalXor, xor)
		// clang-format on
#undef MSL_GROUP_OP
#undef MSL_GROUP_OP_CAST

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

	// Handle HLSL-style 0-based vertex/instance index.
	// Override GLSL compiler strictness
	case BuiltInVertexId:
		ensure_builtin(StorageClassInput, BuiltInVertexId);
		if (msl_options.enable_base_index_zero && msl_options.supports_msl_version(1, 1) &&
		    (msl_options.ios_support_base_vertex_instance || msl_options.is_macos()))
		{
			if (builtin_declaration)
			{
				if (needs_base_vertex_arg != TriState::No)
					needs_base_vertex_arg = TriState::Yes;
				return "gl_VertexID";
			}
			else
			{
				ensure_builtin(StorageClassInput, BuiltInBaseVertex);
				return "(gl_VertexID - gl_BaseVertex)";
			}
		}
		else
		{
			return "gl_VertexID";
		}
	case BuiltInInstanceId:
		ensure_builtin(StorageClassInput, BuiltInInstanceId);
		if (msl_options.enable_base_index_zero && msl_options.supports_msl_version(1, 1) &&
		    (msl_options.ios_support_base_vertex_instance || msl_options.is_macos()))
		{
			if (builtin_declaration)
			{
				if (needs_base_instance_arg != TriState::No)
					needs_base_instance_arg = TriState::Yes;
				return "gl_InstanceID";
			}
			else
			{
				ensure_builtin(StorageClassInput, BuiltInBaseInstance);
				return "(gl_InstanceID - gl_BaseInstance)";
			}
		}
		else
		{
			return "gl_InstanceID";
		}
	case BuiltInVertexIndex:
		ensure_builtin(StorageClassInput, BuiltInVertexIndex);
		if (msl_options.enable_base_index_zero && msl_options.supports_msl_version(1, 1) &&
		    (msl_options.ios_support_base_vertex_instance || msl_options.is_macos()))
		{
			if (builtin_declaration)
			{
				if (needs_base_vertex_arg != TriState::No)
					needs_base_vertex_arg = TriState::Yes;
				return "gl_VertexIndex";
			}
			else
			{
				ensure_builtin(StorageClassInput, BuiltInBaseVertex);
				return "(gl_VertexIndex - gl_BaseVertex)";
			}
		}
		else
		{
			return "gl_VertexIndex";
		}
	case BuiltInInstanceIndex:
		ensure_builtin(StorageClassInput, BuiltInInstanceIndex);
		if (msl_options.enable_base_index_zero && msl_options.supports_msl_version(1, 1) &&
		    (msl_options.ios_support_base_vertex_instance || msl_options.is_macos()))
		{
			if (builtin_declaration)
			{
				if (needs_base_instance_arg != TriState::No)
					needs_base_instance_arg = TriState::Yes;
				return "gl_InstanceIndex";
			}
			else
			{
				ensure_builtin(StorageClassInput, BuiltInBaseInstance);
				return "(gl_InstanceIndex - gl_BaseInstance)";
			}
		}
		else
		{
			return "gl_InstanceIndex";
		}
	case BuiltInBaseVertex:
		if (msl_options.supports_msl_version(1, 1) &&
		    (msl_options.ios_support_base_vertex_instance || msl_options.is_macos()))
		{
			needs_base_vertex_arg = TriState::No;
			return "gl_BaseVertex";
		}
		else
		{
			SPIRV_CROSS_THROW("BaseVertex requires Metal 1.1 and Mac or Apple A9+ hardware.");
		}
	case BuiltInBaseInstance:
		if (msl_options.supports_msl_version(1, 1) &&
		    (msl_options.ios_support_base_vertex_instance || msl_options.is_macos()))
		{
			needs_base_instance_arg = TriState::No;
			return "gl_BaseInstance";
		}
		else
		{
			SPIRV_CROSS_THROW("BaseInstance requires Metal 1.1 and Mac or Apple A9+ hardware.");
		}
	case BuiltInDrawIndex:
		SPIRV_CROSS_THROW("DrawIndex is not supported in MSL.");

	// When used in the entry function, output builtins are qualified with output struct name.
	// Test storage class as NOT Input, as output builtins might be part of generic type.
	// Also don't do this for tessellation control shaders.
	case BuiltInViewportIndex:
		if (!msl_options.supports_msl_version(2, 0))
			SPIRV_CROSS_THROW("ViewportIndex requires Metal 2.0.");
		/* fallthrough */
	case BuiltInFragDepth:
	case BuiltInFragStencilRefEXT:
		if ((builtin == BuiltInFragDepth && !msl_options.enable_frag_depth_builtin) ||
		    (builtin == BuiltInFragStencilRefEXT && !msl_options.enable_frag_stencil_ref_builtin))
			break;
		/* fallthrough */
	case BuiltInPosition:
	case BuiltInPointSize:
	case BuiltInClipDistance:
	case BuiltInCullDistance:
	case BuiltInLayer:
	case BuiltInSampleMask:
		if (get_execution_model() == ExecutionModelTessellationControl)
			break;
		if (storage != StorageClassInput && current_function && (current_function->self == ir.default_entry_point))
			return stage_out_var_name + "." + CompilerGLSL::builtin_to_glsl(builtin, storage);

		break;

	case BuiltInBaryCoordNV:
	case BuiltInBaryCoordNoPerspNV:
		if (storage == StorageClassInput && current_function && (current_function->self == ir.default_entry_point))
			return stage_in_var_name + "." + CompilerGLSL::builtin_to_glsl(builtin, storage);
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
		if (position_invariant)
		{
			if (!msl_options.supports_msl_version(2, 1))
				SPIRV_CROSS_THROW("Invariant position is only supported on MSL 2.1 and up.");
			return "position, invariant";
		}
		else
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
		case ExecutionModelFragment:
			if (msl_options.is_ios())
				SPIRV_CROSS_THROW("PrimitiveId is not supported in fragment on iOS.");
			else if (msl_options.is_macos() && !msl_options.supports_msl_version(2, 2))
				SPIRV_CROSS_THROW("PrimitiveId on macOS requires MSL 2.2.");
			return "primitive_id";
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
	case BuiltInViewIndex:
		if (execution.model != ExecutionModelFragment)
			SPIRV_CROSS_THROW("ViewIndex is handled specially outside fragment shaders.");
		// The ViewIndex was implicitly used in the prior stages to set the render_target_array_index,
		// so we can get it from there.
		return "render_target_array_index";

	// Fragment function out
	case BuiltInFragDepth:
		if (execution.flags.get(ExecutionModeDepthGreater))
			return "depth(greater)";
		else if (execution.flags.get(ExecutionModeDepthLess))
			return "depth(less)";
		else
			return "depth(any)";

	case BuiltInFragStencilRefEXT:
		return "stencil";

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
		if (execution.model == ExecutionModelFragment)
		{
			if (!msl_options.supports_msl_version(2, 2))
				SPIRV_CROSS_THROW("threads_per_simdgroup requires Metal 2.2 in fragment shaders.");
			return "threads_per_simdgroup";
		}
		else
		{
			// thread_execution_width is an alias for threads_per_simdgroup, and it's only available since 1.0,
			// but not in fragment.
			return "thread_execution_width";
		}

	case BuiltInNumSubgroups:
		if (!msl_options.supports_msl_version(2))
			SPIRV_CROSS_THROW("Subgroup builtins require Metal 2.0.");
		return msl_options.is_ios() ? "quadgroups_per_threadgroup" : "simdgroups_per_threadgroup";

	case BuiltInSubgroupId:
		if (!msl_options.supports_msl_version(2))
			SPIRV_CROSS_THROW("Subgroup builtins require Metal 2.0.");
		return msl_options.is_ios() ? "quadgroup_index_in_threadgroup" : "simdgroup_index_in_threadgroup";

	case BuiltInSubgroupLocalInvocationId:
		if (execution.model == ExecutionModelFragment)
		{
			if (!msl_options.supports_msl_version(2, 2))
				SPIRV_CROSS_THROW("thread_index_in_simdgroup requires Metal 2.2 in fragment shaders.");
			return "thread_index_in_simdgroup";
		}
		else
		{
			if (!msl_options.supports_msl_version(2))
				SPIRV_CROSS_THROW("Subgroup builtins require Metal 2.0.");
			return msl_options.is_ios() ? "thread_index_in_quadgroup" : "thread_index_in_simdgroup";
		}

	case BuiltInSubgroupEqMask:
	case BuiltInSubgroupGeMask:
	case BuiltInSubgroupGtMask:
	case BuiltInSubgroupLeMask:
	case BuiltInSubgroupLtMask:
		// Shouldn't be reached.
		SPIRV_CROSS_THROW("Subgroup ballot masks are handled specially in MSL.");

	case BuiltInBaryCoordNV:
		// TODO: AMD barycentrics as well? Seem to have different swizzle and 2 components rather than 3.
		if (msl_options.is_ios())
			SPIRV_CROSS_THROW("Barycentrics not supported on iOS.");
		else if (!msl_options.supports_msl_version(2, 2))
			SPIRV_CROSS_THROW("Barycentrics are only supported in MSL 2.2 and above on macOS.");
		return "barycentric_coord, center_perspective";

	case BuiltInBaryCoordNoPerspNV:
		// TODO: AMD barycentrics as well? Seem to have different swizzle and 2 components rather than 3.
		if (msl_options.is_ios())
			SPIRV_CROSS_THROW("Barycentrics not supported on iOS.");
		else if (!msl_options.supports_msl_version(2, 2))
			SPIRV_CROSS_THROW("Barycentrics are only supported in MSL 2.2 and above on macOS.");
		return "barycentric_coord, center_no_perspective";

	default:
		return "unsupported-built-in";
	}
}

// Returns an MSL string type declaration for a SPIR-V builtin
string CompilerMSL::builtin_type_decl(BuiltIn builtin, uint32_t id)
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
	case BuiltInViewIndex:
		return "uint";

	case BuiltInHelperInvocation:
		return "bool";

	case BuiltInBaryCoordNV:
	case BuiltInBaryCoordNoPerspNV:
		// Use the type as declared, can be 1, 2 or 3 components.
		return type_to_glsl(get_variable_data_type(get<SPIRVariable>(id)));

	// Fragment function out
	case BuiltInFragDepth:
		return "float";

	case BuiltInFragStencilRefEXT:
		return "uint";

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

	case BuiltInDeviceIndex:
		return "int";

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

	// Handle HLSL-style 0-based vertex/instance index.
	builtin_declaration = true;
	bi_arg += builtin_type_decl(builtin);
	bi_arg += " " + builtin_to_glsl(builtin, StorageClassInput);
	bi_arg += " [[" + builtin_qualifier(builtin) + "]]";
	builtin_declaration = false;

	return bi_arg;
}

const SPIRType &CompilerMSL::get_physical_member_type(const SPIRType &type, uint32_t index) const
{
	if (member_is_remapped_physical_type(type, index))
		return get<SPIRType>(get_extended_member_decoration(type.self, index, SPIRVCrossDecorationPhysicalTypeID));
	else
		return get<SPIRType>(type.member_types[index]);
}

uint32_t CompilerMSL::get_declared_type_array_stride_msl(const SPIRType &type, bool is_packed, bool row_major) const
{
	// Array stride in MSL is always size * array_size. sizeof(float3) == 16,
	// unlike GLSL and HLSL where array stride would be 16 and size 12.

	// We could use parent type here and recurse, but that makes creating physical type remappings
	// far more complicated. We'd rather just create the final type, and ignore having to create the entire type
	// hierarchy in order to compute this value, so make a temporary type on the stack.

	auto basic_type = type;
	basic_type.array.clear();
	basic_type.array_size_literal.clear();
	uint32_t value_size = get_declared_type_size_msl(basic_type, is_packed, row_major);

	uint32_t dimensions = uint32_t(type.array.size());
	assert(dimensions > 0);
	dimensions--;

	// Multiply together every dimension, except the last one.
	for (uint32_t dim = 0; dim < dimensions; dim++)
	{
		uint32_t array_size = to_array_size_literal(type, dim);
		value_size *= max(array_size, 1u);
	}

	return value_size;
}

uint32_t CompilerMSL::get_declared_struct_member_array_stride_msl(const SPIRType &type, uint32_t index) const
{
	return get_declared_type_array_stride_msl(get_physical_member_type(type, index),
	                                          member_is_packed_physical_type(type, index),
	                                          has_member_decoration(type.self, index, DecorationRowMajor));
}

uint32_t CompilerMSL::get_declared_type_matrix_stride_msl(const SPIRType &type, bool packed, bool row_major) const
{
	// For packed matrices, we just use the size of the vector type.
	// Otherwise, MatrixStride == alignment, which is the size of the underlying vector type.
	if (packed)
		return (type.width / 8) * (row_major ? type.columns : type.vecsize);
	else
		return get_declared_type_alignment_msl(type, false, row_major);
}

uint32_t CompilerMSL::get_declared_struct_member_matrix_stride_msl(const SPIRType &type, uint32_t index) const
{
	return get_declared_type_matrix_stride_msl(get_physical_member_type(type, index),
	                                           member_is_packed_physical_type(type, index),
	                                           has_member_decoration(type.self, index, DecorationRowMajor));
}

uint32_t CompilerMSL::get_declared_struct_size_msl(const SPIRType &struct_type, bool ignore_alignment,
                                                   bool ignore_padding) const
{
	// If we have a target size, that is the declared size as well.
	if (!ignore_padding && has_extended_decoration(struct_type.self, SPIRVCrossDecorationPaddingTarget))
		return get_extended_decoration(struct_type.self, SPIRVCrossDecorationPaddingTarget);

	if (struct_type.member_types.empty())
		return 0;

	uint32_t mbr_cnt = uint32_t(struct_type.member_types.size());

	// In MSL, a struct's alignment is equal to the maximum alignment of any of its members.
	uint32_t alignment = 1;

	if (!ignore_alignment)
	{
		for (uint32_t i = 0; i < mbr_cnt; i++)
		{
			uint32_t mbr_alignment = get_declared_struct_member_alignment_msl(struct_type, i);
			alignment = max(alignment, mbr_alignment);
		}
	}

	// Last member will always be matched to the final Offset decoration, but size of struct in MSL now depends
	// on physical size in MSL, and the size of the struct itself is then aligned to struct alignment.
	uint32_t spirv_offset = type_struct_member_offset(struct_type, mbr_cnt - 1);
	uint32_t msl_size = spirv_offset + get_declared_struct_member_size_msl(struct_type, mbr_cnt - 1);
	msl_size = (msl_size + alignment - 1) & ~(alignment - 1);
	return msl_size;
}

// Returns the byte size of a struct member.
uint32_t CompilerMSL::get_declared_type_size_msl(const SPIRType &type, bool is_packed, bool row_major) const
{
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
		if (!type.array.empty())
		{
			uint32_t array_size = to_array_size_literal(type);
			return get_declared_type_array_stride_msl(type, is_packed, row_major) * max(array_size, 1u);
		}

		if (type.basetype == SPIRType::Struct)
			return get_declared_struct_size_msl(type);

		if (is_packed)
		{
			return type.vecsize * type.columns * (type.width / 8);
		}
		else
		{
			// An unpacked 3-element vector or matrix column is the same memory size as a 4-element.
			uint32_t vecsize = type.vecsize;
			uint32_t columns = type.columns;

			if (row_major)
				swap(vecsize, columns);

			if (vecsize == 3)
				vecsize = 4;

			return vecsize * columns * (type.width / 8);
		}
	}
	}
}

uint32_t CompilerMSL::get_declared_struct_member_size_msl(const SPIRType &type, uint32_t index) const
{
	return get_declared_type_size_msl(get_physical_member_type(type, index),
	                                  member_is_packed_physical_type(type, index),
	                                  has_member_decoration(type.self, index, DecorationRowMajor));
}

// Returns the byte alignment of a type.
uint32_t CompilerMSL::get_declared_type_alignment_msl(const SPIRType &type, bool is_packed, bool row_major) const
{
	switch (type.basetype)
	{
	case SPIRType::Unknown:
	case SPIRType::Void:
	case SPIRType::AtomicCounter:
	case SPIRType::Image:
	case SPIRType::SampledImage:
	case SPIRType::Sampler:
		SPIRV_CROSS_THROW("Querying alignment of opaque object.");

	case SPIRType::Int64:
		SPIRV_CROSS_THROW("long types are not supported in buffers in MSL.");
	case SPIRType::UInt64:
		SPIRV_CROSS_THROW("ulong types are not supported in buffers in MSL.");
	case SPIRType::Double:
		SPIRV_CROSS_THROW("double types are not supported in buffers in MSL.");

	case SPIRType::Struct:
	{
		// In MSL, a struct's alignment is equal to the maximum alignment of any of its members.
		uint32_t alignment = 1;
		for (uint32_t i = 0; i < type.member_types.size(); i++)
			alignment = max(alignment, uint32_t(get_declared_struct_member_alignment_msl(type, i)));
		return alignment;
	}

	default:
	{
		// Alignment of packed type is the same as the underlying component or column size.
		// Alignment of unpacked type is the same as the vector size.
		// Alignment of 3-elements vector is the same as 4-elements (including packed using column).
		if (is_packed)
		{
			// If we have packed_T and friends, the alignment is always scalar.
			return type.width / 8;
		}
		else
		{
			// This is the general rule for MSL. Size == alignment.
			uint32_t vecsize = row_major ? type.columns : type.vecsize;
			return (type.width / 8) * (vecsize == 3 ? 4 : vecsize);
		}
	}
	}
}

uint32_t CompilerMSL::get_declared_struct_member_alignment_msl(const SPIRType &type, uint32_t index) const
{
	return get_declared_type_alignment_msl(get_physical_member_type(type, index),
	                                       member_is_packed_physical_type(type, index),
	                                       has_member_decoration(type.self, index, DecorationRowMajor));
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

// If a needed custom function wasn't added before, add it and force a recompile.
void CompilerMSL::add_spv_func_and_recompile(SPVFuncImpl spv_func)
{
	if (spv_function_implementations.count(spv_func) == 0)
	{
		spv_function_implementations.insert(spv_func);
		suppress_missing_prototypes = true;
		force_recompile();
	}
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

	// Emulate texture2D atomic operations
	case OpImageTexelPointer:
	{
		auto *var = compiler.maybe_get_backing_variable(args[2]);
		image_pointers[args[1]] = var ? var->self : ID(0);
		break;
	}

	case OpImageWrite:
		uses_resource_write = true;
		break;

	case OpStore:
		check_resource_write(args[0]);
		break;

	// Emulate texture2D atomic operations
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
	{
		uses_atomics = true;
		auto it = image_pointers.find(args[2]);
		if (it != image_pointers.end())
		{
			compiler.atomic_image_vars.insert(it->second);
		}
		check_resource_write(args[2]);
		break;
	}

	case OpAtomicStore:
	{
		uses_atomics = true;
		auto it = image_pointers.find(args[0]);
		if (it != image_pointers.end())
		{
			compiler.atomic_image_vars.insert(it->second);
		}
		check_resource_write(args[0]);
		break;
	}

	case OpAtomicLoad:
	{
		uses_atomics = true;
		auto it = image_pointers.find(args[2]);
		if (it != image_pointers.end())
		{
			compiler.atomic_image_vars.insert(it->second);
		}
		break;
	}

	case OpGroupNonUniformInverseBallot:
		needs_subgroup_invocation_id = true;
		break;

	case OpGroupNonUniformBallotBitCount:
		if (args[3] != GroupOperationReduce)
			needs_subgroup_invocation_id = true;
		break;

	case OpArrayLength:
	{
		auto *var = compiler.maybe_get_backing_variable(args[2]);
		if (var)
			compiler.buffers_requiring_array_length.insert(var->self);
		break;
	}

	case OpInBoundsAccessChain:
	case OpAccessChain:
	case OpPtrAccessChain:
	{
		// OpArrayLength might want to know if taking ArrayLength of an array of SSBOs.
		uint32_t result_type = args[0];
		uint32_t id = args[1];
		uint32_t ptr = args[2];

		compiler.set<SPIRExpression>(id, "", result_type, true);
		compiler.register_read(id, ptr, true);
		compiler.ir.ids[id].set_allow_type_rewrite();
		break;
	}

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

	case OpFAdd:
		if (compiler.msl_options.invariant_float_math)
		{
			return SPVFuncImplFAdd;
		}
		break;

	case OpFMul:
	case OpOuterProduct:
	case OpMatrixTimesVector:
	case OpVectorTimesMatrix:
	case OpMatrixTimesMatrix:
		if (compiler.msl_options.invariant_float_math)
		{
			return SPVFuncImplFMul;
		}
		break;

	case OpTypeArray:
	{
		// Allow Metal to use the array<T> template to make arrays a value type
		return SPVFuncImplUnsafeArray;
	}

	// Emulate texture2D atomic operations
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
	case OpAtomicLoad:
	case OpAtomicStore:
	{
		auto it = image_pointers.find(args[opcode == OpAtomicStore ? 0 : 2]);
		if (it != image_pointers.end())
		{
			uint32_t tid = compiler.get<SPIRVariable>(it->second).basetype;
			if (tid && compiler.get<SPIRType>(tid).image.dim == Dim2D)
				return SPVFuncImplImage2DAtomicCoords;
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
		break;
	}

	case OpExtInst:
	{
		uint32_t extension_set = args[2];
		if (compiler.get<SPIRExtension>(extension_set).ext == SPIRExtension::GLSL)
		{
			auto op_450 = static_cast<GLSLstd450>(args[3]);
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
			case GLSLstd450Reflect:
			{
				auto &type = compiler.get<SPIRType>(args[0]);
				if (type.vecsize == 1)
					return SPVFuncImplReflectScalar;
				break;
			}
			case GLSLstd450Refract:
			{
				auto &type = compiler.get<SPIRType>(args[0]);
				if (type.vecsize == 1)
					return SPVFuncImplRefractScalar;
				break;
			}
			case GLSLstd450FaceForward:
			{
				auto &type = compiler.get<SPIRType>(args[0]);
				if (type.vecsize == 1)
					return SPVFuncImplFaceForwardScalar;
				break;
			}
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
	std::stable_sort(mbr_idxs.begin(), mbr_idxs.end(), *this); // Sort member indices based on sorting aspect

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

void CompilerMSL::remap_constexpr_sampler(VariableID id, const MSLConstexprSampler &sampler)
{
	auto &type = get<SPIRType>(get<SPIRVariable>(id).basetype);
	if (type.basetype != SPIRType::SampledImage && type.basetype != SPIRType::Sampler)
		SPIRV_CROSS_THROW("Can only remap SampledImage and Sampler type.");
	if (!type.array.empty())
		SPIRV_CROSS_THROW("Can not remap array of samplers.");
	constexpr_samplers_by_id[id] = sampler;
}

void CompilerMSL::remap_constexpr_sampler_by_binding(uint32_t desc_set, uint32_t binding,
                                                     const MSLConstexprSampler &sampler)
{
	constexpr_samplers_by_binding[{ desc_set, binding }] = sampler;
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
	case BuiltInFragStencilRefEXT:
	case BuiltInPrimitiveId:
	case BuiltInSubgroupSize:
	case BuiltInSubgroupLocalInvocationId:
	case BuiltInViewIndex:
	case BuiltInVertexIndex:
	case BuiltInInstanceIndex:
	case BuiltInBaseInstance:
	case BuiltInBaseVertex:
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
	case BuiltInFragStencilRefEXT:
	case BuiltInPrimitiveId:
	case BuiltInViewIndex:
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

string CompilerMSL::to_initializer_expression(const SPIRVariable &var)
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

string CompilerMSL::to_zero_initialized_expression(uint32_t)
{
	return "{}";
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
		uint32_t plane;
	};
	SmallVector<Resource> resources_in_set[kMaxArgumentBuffers];
	SmallVector<uint32_t> inline_block_vars;

	bool set_needs_swizzle_buffer[kMaxArgumentBuffers] = {};
	bool set_needs_buffer_sizes[kMaxArgumentBuffers] = {};
	bool needs_buffer_sizes = false;

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

			const MSLConstexprSampler *constexpr_sampler = nullptr;
			if (type.basetype == SPIRType::SampledImage || type.basetype == SPIRType::Sampler)
			{
				constexpr_sampler = find_constexpr_sampler(var_id);
				if (constexpr_sampler)
				{
					// Mark this ID as a constexpr sampler for later in case it came from set/bindings.
					constexpr_samplers_by_id[var_id] = *constexpr_sampler;
				}
			}

			uint32_t binding = get_decoration(var_id, DecorationBinding);
			if (type.basetype == SPIRType::SampledImage)
			{
				add_resource_name(var_id);

				uint32_t plane_count = 1;
				if (constexpr_sampler && constexpr_sampler->ycbcr_conversion_enable)
					plane_count = constexpr_sampler->planes;

				for (uint32_t i = 0; i < plane_count; i++)
				{
					uint32_t image_resource_index = get_metal_resource_index(var, SPIRType::Image, i);
					resources_in_set[desc_set].push_back(
					    { &var, to_name(var_id), SPIRType::Image, image_resource_index, i });
				}

				if (type.image.dim != DimBuffer && !constexpr_sampler)
				{
					uint32_t sampler_resource_index = get_metal_resource_index(var, SPIRType::Sampler);
					resources_in_set[desc_set].push_back(
					    { &var, to_sampler_expression(var_id), SPIRType::Sampler, sampler_resource_index, 0 });
				}
			}
			else if (inline_uniform_blocks.count(SetBindingPair{ desc_set, binding }))
			{
				inline_block_vars.push_back(var_id);
			}
			else if (!constexpr_sampler)
			{
				// constexpr samplers are not declared as resources.
				// Inline uniform blocks are always emitted at the end.
				if (!msl_options.is_ios() || type.basetype != SPIRType::Image || type.image.sampled != 2)
				{
					add_resource_name(var_id);
					resources_in_set[desc_set].push_back(
					    { &var, to_name(var_id), type.basetype, get_metal_resource_index(var, type.basetype), 0 });
				}
			}

			// Check if this descriptor set needs a swizzle buffer.
			if (needs_swizzle_buffer_def && is_sampled_image_type(type))
				set_needs_swizzle_buffer[desc_set] = true;
			else if (buffers_requiring_array_length.count(var_id) != 0)
			{
				set_needs_buffer_sizes[desc_set] = true;
				needs_buffer_sizes = true;
			}
		}
	});

	if (needs_swizzle_buffer_def || needs_buffer_sizes)
	{
		uint32_t uint_ptr_type_id = 0;

		// We might have to add a swizzle buffer resource to the set.
		for (uint32_t desc_set = 0; desc_set < kMaxArgumentBuffers; desc_set++)
		{
			if (!set_needs_swizzle_buffer[desc_set] && !set_needs_buffer_sizes[desc_set])
				continue;

			if (uint_ptr_type_id == 0)
			{
				uint32_t offset = ir.increase_bound_by(2);
				uint32_t type_id = offset;
				uint_ptr_type_id = offset + 1;

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
				set<SPIRType>(uint_ptr_type_id, uint_type_pointer);
				set_decoration(uint_ptr_type_id, DecorationArrayStride, 4);
			}

			if (set_needs_swizzle_buffer[desc_set])
			{
				uint32_t var_id = ir.increase_bound_by(1);
				auto &var = set<SPIRVariable>(var_id, uint_ptr_type_id, StorageClassUniformConstant);
				set_name(var_id, "spvSwizzleConstants");
				set_decoration(var_id, DecorationDescriptorSet, desc_set);
				set_decoration(var_id, DecorationBinding, kSwizzleBufferBinding);
				resources_in_set[desc_set].push_back(
				    { &var, to_name(var_id), SPIRType::UInt, get_metal_resource_index(var, SPIRType::UInt), 0 });
			}

			if (set_needs_buffer_sizes[desc_set])
			{
				uint32_t var_id = ir.increase_bound_by(1);
				auto &var = set<SPIRVariable>(var_id, uint_ptr_type_id, StorageClassUniformConstant);
				set_name(var_id, "spvBufferSizeConstants");
				set_decoration(var_id, DecorationDescriptorSet, desc_set);
				set_decoration(var_id, DecorationBinding, kBufferSizeBufferBinding);
				resources_in_set[desc_set].push_back(
				    { &var, to_name(var_id), SPIRType::UInt, get_metal_resource_index(var, SPIRType::UInt), 0 });
			}
		}
	}

	// Now add inline uniform blocks.
	for (uint32_t var_id : inline_block_vars)
	{
		auto &var = get<SPIRVariable>(var_id);
		uint32_t desc_set = get_decoration(var_id, DecorationDescriptorSet);
		add_resource_name(var_id);
		resources_in_set[desc_set].push_back(
		    { &var, to_name(var_id), SPIRType::Struct, get_metal_resource_index(var, SPIRType::Struct), 0 });
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

		buffer_type.basetype = SPIRType::Struct;

		if ((argument_buffer_device_storage_mask & (1u << desc_set)) != 0)
		{
			buffer_type.storage = StorageClassStorageBuffer;
			// Make sure the argument buffer gets marked as const device.
			set_decoration(next_id, DecorationNonWritable);
			// Need to mark the type as a Block to enable this.
			set_decoration(type_id, DecorationBlock);
		}
		else
			buffer_type.storage = StorageClassUniform;

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
			if (resource.plane > 0)
				mbr_name += join(plane_name_suffix, resource.plane);
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
				uint32_t binding = get_decoration(var.self, DecorationBinding);
				SetBindingPair pair = { desc_set, binding };

				if (resource.basetype == SPIRType::Image || resource.basetype == SPIRType::Sampler ||
				    resource.basetype == SPIRType::SampledImage)
				{
					// Drop pointer information when we emit the resources into a struct.
					buffer_type.member_types.push_back(get_variable_data_type_id(var));
					if (resource.plane == 0)
						set_qualified_name(var.self, join(to_name(buffer_variable_id), ".", mbr_name));
				}
				else if (buffers_requiring_dynamic_offset.count(pair))
				{
					// Don't set the qualified name here; we'll define a variable holding the corrected buffer address later.
					buffer_type.member_types.push_back(var.basetype);
					buffers_requiring_dynamic_offset[pair].second = var.self;
				}
				else if (inline_uniform_blocks.count(pair))
				{
					// Put the buffer block itself into the argument buffer.
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

			set_extended_member_decoration(buffer_type.self, member_index, SPIRVCrossDecorationResourceIndexPrimary,
			                               resource.index);
			set_extended_member_decoration(buffer_type.self, member_index, SPIRVCrossDecorationInterfaceOrigID,
			                               var.self);
			member_index++;
		}
	}
}

void CompilerMSL::activate_argument_buffer_resources()
{
	// For ABI compatibility, force-enable all resources which are part of argument buffers.
	ir.for_each_typed_id<SPIRVariable>([&](uint32_t self, const SPIRVariable &) {
		if (!has_decoration(self, DecorationDescriptorSet))
			return;

		uint32_t desc_set = get_decoration(self, DecorationDescriptorSet);
		if (descriptor_set_is_argument_buffer(desc_set))
			active_interface_variables.insert(self);
	});
}

bool CompilerMSL::using_builtin_array() const
{
	return msl_options.force_native_arrays || is_using_builtin_array;
}
