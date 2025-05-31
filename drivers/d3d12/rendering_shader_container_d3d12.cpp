/**************************************************************************/
/*  rendering_shader_container_d3d12.cpp                                  */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "rendering_shader_container_d3d12.h"

#include "core/templates/sort_array.h"

#include "dxil_hash.h"

#include <zlib.h>

#ifndef _MSC_VER
// Match current version used by MinGW, MSVC and Direct3D 12 headers use 500.
#define __REQUIRED_RPCNDR_H_VERSION__ 475
#endif

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wswitch"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wimplicit-fallthrough"
#elif defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnon-virtual-dtor"
#pragma clang diagnostic ignored "-Wstring-plus-int"
#pragma clang diagnostic ignored "-Wswitch"
#pragma clang diagnostic ignored "-Wmissing-field-initializers"
#pragma clang diagnostic ignored "-Wimplicit-fallthrough"
#endif

#include "d3dx12.h"
#include <dxgi1_6.h>
#define D3D12MA_D3D12_HEADERS_ALREADY_INCLUDED
#include "D3D12MemAlloc.h"

#include <wrl/client.h>

#if defined(_MSC_VER) && defined(MemoryBarrier)
// Annoying define from winnt.h. Reintroduced by some of the headers above.
#undef MemoryBarrier
#endif

// No point in fighting warnings in Mesa.
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4200) // "nonstandard extension used: zero-sized array in struct/union".
#pragma warning(disable : 4806) // "'&': unsafe operation: no value of type 'bool' promoted to type 'uint32_t' can equal the given constant".
#endif

#include "nir_spirv.h"
#include "nir_to_dxil.h"
#include "spirv_to_dxil.h"
extern "C" {
#include "dxil_spirv_nir.h"
}

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#elif defined(__clang__)
#pragma clang diagnostic pop
#endif

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

static D3D12_SHADER_VISIBILITY stages_to_d3d12_visibility(uint32_t p_stages_mask) {
	switch (p_stages_mask) {
		case RenderingDeviceCommons::SHADER_STAGE_VERTEX_BIT:
			return D3D12_SHADER_VISIBILITY_VERTEX;
		case RenderingDeviceCommons::SHADER_STAGE_FRAGMENT_BIT:
			return D3D12_SHADER_VISIBILITY_PIXEL;
		default:
			return D3D12_SHADER_VISIBILITY_ALL;
	}
}

uint32_t RenderingDXIL::patch_specialization_constant(
		RenderingDeviceCommons::PipelineSpecializationConstantType p_type,
		const void *p_value,
		const uint64_t (&p_stages_bit_offsets)[D3D12_BITCODE_OFFSETS_NUM_STAGES],
		HashMap<RenderingDeviceCommons::ShaderStage, Vector<uint8_t>> &r_stages_bytecodes,
		bool p_is_first_patch) {
	uint32_t patch_val = 0;
	switch (p_type) {
		case RenderingDeviceCommons::PIPELINE_SPECIALIZATION_CONSTANT_TYPE_INT: {
			uint32_t int_value = *((const int *)p_value);
			ERR_FAIL_COND_V(int_value & (1 << 31), 0);
			patch_val = int_value;
		} break;
		case RenderingDeviceCommons::PIPELINE_SPECIALIZATION_CONSTANT_TYPE_BOOL: {
			bool bool_value = *((const bool *)p_value);
			patch_val = (uint32_t)bool_value;
		} break;
		case RenderingDeviceCommons::PIPELINE_SPECIALIZATION_CONSTANT_TYPE_FLOAT: {
			uint32_t int_value = *((const int *)p_value);
			ERR_FAIL_COND_V(int_value & (1 << 31), 0);
			patch_val = (int_value >> 1);
		} break;
	}
	// For VBR encoding to encode the number of bits we expect (32), we need to set the MSB unconditionally.
	// However, signed VBR moves the MSB to the LSB, so setting the MSB to 1 wouldn't help. Therefore,
	// the bit we set to 1 is the one at index 30.
	patch_val |= (1 << 30);
	patch_val <<= 1; // What signed VBR does.

	auto tamper_bits = [](uint8_t *p_start, uint64_t p_bit_offset, uint64_t p_tb_value) -> uint64_t {
		uint64_t original = 0;
		uint32_t curr_input_byte = p_bit_offset / 8;
		uint8_t curr_input_bit = p_bit_offset % 8;
		auto get_curr_input_bit = [&]() -> bool {
			return ((p_start[curr_input_byte] >> curr_input_bit) & 1);
		};
		auto move_to_next_input_bit = [&]() {
			if (curr_input_bit == 7) {
				curr_input_bit = 0;
				curr_input_byte++;
			} else {
				curr_input_bit++;
			}
		};
		auto tamper_input_bit = [&](bool p_new_bit) {
			p_start[curr_input_byte] &= ~((uint8_t)1 << curr_input_bit);
			if (p_new_bit) {
				p_start[curr_input_byte] |= (uint8_t)1 << curr_input_bit;
			}
		};
		uint8_t value_bit_idx = 0;
		for (uint32_t i = 0; i < 5; i++) { // 32 bits take 5 full bytes in VBR.
			for (uint32_t j = 0; j < 7; j++) {
				bool input_bit = get_curr_input_bit();
				original |= (uint64_t)(input_bit ? 1 : 0) << value_bit_idx;
				tamper_input_bit((p_tb_value >> value_bit_idx) & 1);
				move_to_next_input_bit();
				value_bit_idx++;
			}
#ifdef DEV_ENABLED
			bool input_bit = get_curr_input_bit();
			DEV_ASSERT((i < 4 && input_bit) || (i == 4 && !input_bit));
#endif
			move_to_next_input_bit();
		}
		return original;
	};
	uint32_t stages_patched_mask = 0;
	for (int stage = 0; stage < RenderingDeviceCommons::SHADER_STAGE_MAX; stage++) {
		if (!r_stages_bytecodes.has((RenderingDeviceCommons::ShaderStage)stage)) {
			continue;
		}

		uint64_t offset = p_stages_bit_offsets[RenderingShaderContainerD3D12::SHADER_STAGES_BIT_OFFSET_INDICES[stage]];
		if (offset == 0) {
			// This constant does not appear at this stage.
			continue;
		}

		Vector<uint8_t> &bytecode = r_stages_bytecodes[(RenderingDeviceCommons::ShaderStage)stage];
#ifdef DEV_ENABLED
		uint64_t orig_patch_val = tamper_bits(bytecode.ptrw(), offset, patch_val);
		// Checking against the value the NIR patch should have set.
		DEV_ASSERT(!p_is_first_patch || ((orig_patch_val >> 1) & GODOT_NIR_SC_SENTINEL_MAGIC_MASK) == GODOT_NIR_SC_SENTINEL_MAGIC);
		uint64_t readback_patch_val = tamper_bits(bytecode.ptrw(), offset, patch_val);
		DEV_ASSERT(readback_patch_val == patch_val);
#else
		tamper_bits(bytecode.ptrw(), offset, patch_val);
#endif

		stages_patched_mask |= (1 << stage);
	}

	return stages_patched_mask;
}

void RenderingDXIL::sign_bytecode(RenderingDeviceCommons::ShaderStage p_stage, Vector<uint8_t> &r_dxil_blob) {
	uint8_t *w = r_dxil_blob.ptrw();
	compute_dxil_hash(w + 20, r_dxil_blob.size() - 20, w + 4);
}

// RenderingShaderContainerD3D12

uint32_t RenderingShaderContainerD3D12::_format() const {
	return 0x43443344;
}

uint32_t RenderingShaderContainerD3D12::_format_version() const {
	return FORMAT_VERSION;
}

uint32_t RenderingShaderContainerD3D12::_from_bytes_reflection_extra_data(const uint8_t *p_bytes) {
	reflection_data_d3d12 = *(const ReflectionDataD3D12 *)(p_bytes);
	return sizeof(ReflectionDataD3D12);
}

uint32_t RenderingShaderContainerD3D12::_from_bytes_reflection_binding_uniform_extra_data_start(const uint8_t *p_bytes) {
	reflection_binding_set_uniforms_data_d3d12.resize(reflection_binding_set_uniforms_data.size());
	return 0;
}

uint32_t RenderingShaderContainerD3D12::_from_bytes_reflection_binding_uniform_extra_data(const uint8_t *p_bytes, uint32_t p_index) {
	reflection_binding_set_uniforms_data_d3d12.ptrw()[p_index] = *(const ReflectionBindingDataD3D12 *)(p_bytes);
	return sizeof(ReflectionBindingDataD3D12);
}

uint32_t RenderingShaderContainerD3D12::_from_bytes_reflection_specialization_extra_data_start(const uint8_t *p_bytes) {
	reflection_specialization_data_d3d12.resize(reflection_specialization_data.size());
	return 0;
}

uint32_t RenderingShaderContainerD3D12::_from_bytes_reflection_specialization_extra_data(const uint8_t *p_bytes, uint32_t p_index) {
	reflection_specialization_data_d3d12.ptrw()[p_index] = *(const ReflectionSpecializationDataD3D12 *)(p_bytes);
	return sizeof(ReflectionSpecializationDataD3D12);
}

uint32_t RenderingShaderContainerD3D12::_from_bytes_footer_extra_data(const uint8_t *p_bytes) {
	ContainerFooterD3D12 footer = *(const ContainerFooterD3D12 *)(p_bytes);
	root_signature_crc = footer.root_signature_crc;
	root_signature_bytes.resize(footer.root_signature_length);
	memcpy(root_signature_bytes.ptrw(), p_bytes + sizeof(ContainerFooterD3D12), root_signature_bytes.size());
	return sizeof(ContainerFooterD3D12) + footer.root_signature_length;
}

uint32_t RenderingShaderContainerD3D12::_to_bytes_reflection_extra_data(uint8_t *p_bytes) const {
	if (p_bytes != nullptr) {
		*(ReflectionDataD3D12 *)(p_bytes) = reflection_data_d3d12;
	}

	return sizeof(ReflectionDataD3D12);
}

uint32_t RenderingShaderContainerD3D12::_to_bytes_reflection_binding_uniform_extra_data(uint8_t *p_bytes, uint32_t p_index) const {
	if (p_bytes != nullptr) {
		*(ReflectionBindingDataD3D12 *)(p_bytes) = reflection_binding_set_uniforms_data_d3d12[p_index];
	}

	return sizeof(ReflectionBindingDataD3D12);
}

uint32_t RenderingShaderContainerD3D12::_to_bytes_reflection_specialization_extra_data(uint8_t *p_bytes, uint32_t p_index) const {
	if (p_bytes != nullptr) {
		*(ReflectionSpecializationDataD3D12 *)(p_bytes) = reflection_specialization_data_d3d12[p_index];
	}

	return sizeof(ReflectionSpecializationDataD3D12);
}

uint32_t RenderingShaderContainerD3D12::_to_bytes_footer_extra_data(uint8_t *p_bytes) const {
	if (p_bytes != nullptr) {
		ContainerFooterD3D12 &footer = *(ContainerFooterD3D12 *)(p_bytes);
		footer.root_signature_length = root_signature_bytes.size();
		footer.root_signature_crc = root_signature_crc;
		memcpy(p_bytes + sizeof(ContainerFooterD3D12), root_signature_bytes.ptr(), root_signature_bytes.size());
	}

	return sizeof(ContainerFooterD3D12) + root_signature_bytes.size();
}

#if NIR_ENABLED
bool RenderingShaderContainerD3D12::_convert_spirv_to_nir(const Vector<RenderingDeviceCommons::ShaderStageSPIRVData> &p_spirv, const nir_shader_compiler_options *p_compiler_options, HashMap<int, nir_shader *> &r_stages_nir_shaders, Vector<RenderingDeviceCommons::ShaderStage> &r_stages, BitField<RenderingDeviceCommons::ShaderStage> &r_stages_processed) {
	r_stages_processed.clear();

	dxil_spirv_runtime_conf dxil_runtime_conf = {};
	dxil_runtime_conf.runtime_data_cbv.base_shader_register = RUNTIME_DATA_REGISTER;
	dxil_runtime_conf.push_constant_cbv.base_shader_register = ROOT_CONSTANT_REGISTER;
	dxil_runtime_conf.zero_based_vertex_instance_id = true;
	dxil_runtime_conf.zero_based_compute_workgroup_id = true;
	dxil_runtime_conf.declared_read_only_images_as_srvs = true;

	// Making this explicit to let maintainers know that in practice this didn't improve performance,
	// probably because data generated by one shader and consumed by another one forces the resource
	// to transition from UAV to SRV, and back, instead of being an UAV all the time.
	// In case someone wants to try, care must be taken so in case of incompatible bindings across stages
	// happen as a result, all the stages are re-translated. That can happen if, for instance, a stage only
	// uses an allegedly writable resource only for reading but the next stage doesn't.
	dxil_runtime_conf.inferred_read_only_images_as_srvs = false;

	// Translate SPIR-V to NIR.
	for (int64_t i = 0; i < p_spirv.size(); i++) {
		RenderingDeviceCommons::ShaderStage stage = p_spirv[i].shader_stage;
		RenderingDeviceCommons::ShaderStage stage_flag = (RenderingDeviceCommons::ShaderStage)(1 << stage);
		r_stages.push_back(stage);
		r_stages_processed.set_flag(stage_flag);

		const char *entry_point = "main";
		static const gl_shader_stage SPIRV_TO_MESA_STAGES[RenderingDeviceCommons::SHADER_STAGE_MAX] = {
			MESA_SHADER_VERTEX, // SHADER_STAGE_VERTEX
			MESA_SHADER_FRAGMENT, // SHADER_STAGE_FRAGMENT
			MESA_SHADER_TESS_CTRL, // SHADER_STAGE_TESSELATION_CONTROL
			MESA_SHADER_TESS_EVAL, // SHADER_STAGE_TESSELATION_EVALUATION
			MESA_SHADER_COMPUTE, // SHADER_STAGE_COMPUTE
		};

		nir_shader *shader = spirv_to_nir(
				(const uint32_t *)(p_spirv[i].spirv.ptr()),
				p_spirv[i].spirv.size() / sizeof(uint32_t),
				nullptr,
				0,
				SPIRV_TO_MESA_STAGES[stage],
				entry_point,
				dxil_spirv_nir_get_spirv_options(),
				p_compiler_options);

		ERR_FAIL_NULL_V_MSG(shader, false, "Shader translation (step 1) at stage " + String(RenderingDeviceCommons::SHADER_STAGE_NAMES[stage]) + " failed.");

#ifdef DEV_ENABLED
		nir_validate_shader(shader, "Validate before feeding NIR to the DXIL compiler");
#endif

		if (stage == RenderingDeviceCommons::SHADER_STAGE_VERTEX) {
			dxil_runtime_conf.yz_flip.y_mask = 0xffff;
			dxil_runtime_conf.yz_flip.mode = DXIL_SPIRV_Y_FLIP_UNCONDITIONAL;
		} else {
			dxil_runtime_conf.yz_flip.y_mask = 0;
			dxil_runtime_conf.yz_flip.mode = DXIL_SPIRV_YZ_FLIP_NONE;
		}

		dxil_spirv_nir_prep(shader);
		bool requires_runtime_data = false;
		dxil_spirv_nir_passes(shader, &dxil_runtime_conf, &requires_runtime_data);

		r_stages_nir_shaders[stage] = shader;
	}

	// Link NIR shaders.
	for (int i = RenderingDeviceCommons::SHADER_STAGE_MAX - 1; i >= 0; i--) {
		if (!r_stages_nir_shaders.has(i)) {
			continue;
		}
		nir_shader *shader = r_stages_nir_shaders[i];
		nir_shader *prev_shader = nullptr;
		for (int j = i - 1; j >= 0; j--) {
			if (r_stages_nir_shaders.has(j)) {
				prev_shader = r_stages_nir_shaders[j];
				break;
			}
		}
		// There is a bug in the Direct3D runtime during creation of a PSO with view instancing. If a fragment
		// shader uses front/back face detection (SV_IsFrontFace), its signature must include the pixel position
		// builtin variable (SV_Position), otherwise an Internal Runtime error will occur.
		if (i == RenderingDeviceCommons::SHADER_STAGE_FRAGMENT) {
			const bool use_front_face =
					nir_find_variable_with_location(shader, nir_var_shader_in, VARYING_SLOT_FACE) ||
					(shader->info.inputs_read & VARYING_BIT_FACE) ||
					nir_find_variable_with_location(shader, nir_var_system_value, SYSTEM_VALUE_FRONT_FACE) ||
					BITSET_TEST(shader->info.system_values_read, SYSTEM_VALUE_FRONT_FACE);
			const bool use_position =
					nir_find_variable_with_location(shader, nir_var_shader_in, VARYING_SLOT_POS) ||
					(shader->info.inputs_read & VARYING_BIT_POS) ||
					nir_find_variable_with_location(shader, nir_var_system_value, SYSTEM_VALUE_FRAG_COORD) ||
					BITSET_TEST(shader->info.system_values_read, SYSTEM_VALUE_FRAG_COORD);
			if (use_front_face && !use_position) {
				nir_variable *const pos = nir_variable_create(shader, nir_var_shader_in, glsl_vec4_type(), "gl_FragCoord");
				pos->data.location = VARYING_SLOT_POS;
				shader->info.inputs_read |= VARYING_BIT_POS;
			}
		}
		if (prev_shader) {
			bool requires_runtime_data = {};
			dxil_spirv_nir_link(shader, prev_shader, &dxil_runtime_conf, &requires_runtime_data);
		}
	}

	return true;
}

struct GodotNirCallbackUserData {
	RenderingShaderContainerD3D12 *container;
	RenderingDeviceCommons::ShaderStage stage;
};

static dxil_shader_model shader_model_d3d_to_dxil(D3D_SHADER_MODEL p_d3d_shader_model) {
	static_assert(SHADER_MODEL_6_0 == 0x60000);
	static_assert(SHADER_MODEL_6_3 == 0x60003);
	static_assert(D3D_SHADER_MODEL_6_0 == 0x60);
	static_assert(D3D_SHADER_MODEL_6_3 == 0x63);
	return (dxil_shader_model)((p_d3d_shader_model >> 4) * 0x10000 + (p_d3d_shader_model & 0xf));
}

bool RenderingShaderContainerD3D12::_convert_nir_to_dxil(const HashMap<int, nir_shader *> &p_stages_nir_shaders, BitField<RenderingDeviceCommons::ShaderStage> p_stages_processed, HashMap<RenderingDeviceCommons::ShaderStage, Vector<uint8_t>> &r_dxil_blobs) {
	// Translate NIR to DXIL.
	for (KeyValue<int, nir_shader *> it : p_stages_nir_shaders) {
		RenderingDeviceCommons::ShaderStage stage = (RenderingDeviceCommons::ShaderStage)(it.key);
		GodotNirCallbackUserData godot_nir_callback_user_data;
		godot_nir_callback_user_data.container = this;
		godot_nir_callback_user_data.stage = stage;

		GodotNirCallbacks godot_nir_callbacks = {};
		godot_nir_callbacks.data = &godot_nir_callback_user_data;
		godot_nir_callbacks.report_resource = _nir_report_resource;
		godot_nir_callbacks.report_sc_bit_offset_fn = _nir_report_sc_bit_offset;
		godot_nir_callbacks.report_bitcode_bit_offset_fn = _nir_report_bitcode_bit_offset;

		nir_to_dxil_options nir_to_dxil_options = {};
		nir_to_dxil_options.environment = DXIL_ENVIRONMENT_VULKAN;
		nir_to_dxil_options.shader_model_max = shader_model_d3d_to_dxil(D3D_SHADER_MODEL(REQUIRED_SHADER_MODEL));
		nir_to_dxil_options.validator_version_max = NO_DXIL_VALIDATION;
		nir_to_dxil_options.godot_nir_callbacks = &godot_nir_callbacks;

		dxil_logger logger = {};
		logger.log = [](void *p_priv, const char *p_msg) {
#ifdef DEBUG_ENABLED
			print_verbose(p_msg);
#endif
		};

		blob dxil_blob = {};
		bool ok = nir_to_dxil(it.value, &nir_to_dxil_options, &logger, &dxil_blob);
		ERR_FAIL_COND_V_MSG(!ok, false, "Shader translation at stage " + String(RenderingDeviceCommons::SHADER_STAGE_NAMES[stage]) + " failed.");

		Vector<uint8_t> blob_copy;
		blob_copy.resize(dxil_blob.size);
		memcpy(blob_copy.ptrw(), dxil_blob.data, dxil_blob.size);
		blob_finish(&dxil_blob);
		r_dxil_blobs.insert(stage, blob_copy);
	}

	return true;
}

bool RenderingShaderContainerD3D12::_convert_spirv_to_dxil(const Vector<RenderingDeviceCommons::ShaderStageSPIRVData> &p_spirv, HashMap<RenderingDeviceCommons::ShaderStage, Vector<uint8_t>> &r_dxil_blobs, Vector<RenderingDeviceCommons::ShaderStage> &r_stages, BitField<RenderingDeviceCommons::ShaderStage> &r_stages_processed) {
	r_dxil_blobs.clear();

	HashMap<int, nir_shader *> stages_nir_shaders;
	auto free_nir_shaders = [&]() {
		for (KeyValue<int, nir_shader *> &E : stages_nir_shaders) {
			ralloc_free(E.value);
		}
		stages_nir_shaders.clear();
	};

	// This structure must live as long as the shaders are alive.
	nir_shader_compiler_options compiler_options = *dxil_get_nir_compiler_options();
	compiler_options.lower_base_vertex = false;

	// This is based on spirv2dxil.c. May need updates when it changes.
	// Also, this has to stay around until after linking.
	if (!_convert_spirv_to_nir(p_spirv, &compiler_options, stages_nir_shaders, r_stages, r_stages_processed)) {
		free_nir_shaders();
		return false;
	}

	if (!_convert_nir_to_dxil(stages_nir_shaders, r_stages_processed, r_dxil_blobs)) {
		free_nir_shaders();
		return false;
	}

	free_nir_shaders();
	return true;
}

bool RenderingShaderContainerD3D12::_generate_root_signature(BitField<RenderingDeviceCommons::ShaderStage> p_stages_processed) {
	// Root (push) constants.
	LocalVector<D3D12_ROOT_PARAMETER1> root_params;
	if (reflection_data_d3d12.dxil_push_constant_stages) {
		CD3DX12_ROOT_PARAMETER1 push_constant;
		push_constant.InitAsConstants(
				reflection_data.push_constant_size / sizeof(uint32_t),
				ROOT_CONSTANT_REGISTER,
				0,
				stages_to_d3d12_visibility(reflection_data_d3d12.dxil_push_constant_stages));

		root_params.push_back(push_constant);
	}

	// NIR-DXIL runtime data.
	if (reflection_data_d3d12.nir_runtime_data_root_param_idx == 1) { // Set above to 1 when discovering runtime data is needed.
		DEV_ASSERT(!reflection_data.is_compute); // Could be supported if needed, but it's pointless as of now.
		reflection_data_d3d12.nir_runtime_data_root_param_idx = root_params.size();
		CD3DX12_ROOT_PARAMETER1 nir_runtime_data;
		nir_runtime_data.InitAsConstants(
				sizeof(dxil_spirv_vertex_runtime_data) / sizeof(uint32_t),
				RUNTIME_DATA_REGISTER,
				0,
				D3D12_SHADER_VISIBILITY_VERTEX);
		root_params.push_back(nir_runtime_data);
	}

	// Descriptor tables (up to two per uniform set, for resources and/or samplers).
	// These have to stay around until serialization!
	struct TraceableDescriptorTable {
		uint32_t stages_mask = {};
		Vector<D3D12_DESCRIPTOR_RANGE1> ranges;
		Vector<RootSignatureLocation *> root_signature_locations;
	};

	uint32_t binding_start = 0;
	Vector<TraceableDescriptorTable> resource_tables_maps;
	Vector<TraceableDescriptorTable> sampler_tables_maps;
	for (uint32_t i = 0; i < reflection_binding_set_uniforms_count.size(); i++) {
		bool first_resource_in_set = true;
		bool first_sampler_in_set = true;
		uint32_t uniform_count = reflection_binding_set_uniforms_count[i];
		for (uint32_t j = 0; j < uniform_count; j++) {
			const ReflectionBindingData &uniform = reflection_binding_set_uniforms_data[binding_start + j];
			ReflectionBindingDataD3D12 &uniform_d3d12 = reflection_binding_set_uniforms_data_d3d12.ptrw()[binding_start + j];
			bool really_used = uniform_d3d12.dxil_stages != 0;
#ifdef DEV_ENABLED
			bool anybody_home = (ResourceClass)(uniform_d3d12.resource_class) != RES_CLASS_INVALID || uniform_d3d12.has_sampler;
			DEV_ASSERT(anybody_home == really_used);
#endif
			if (!really_used) {
				continue; // Existed in SPIR-V; went away in DXIL.
			}

			auto insert_range = [](D3D12_DESCRIPTOR_RANGE_TYPE p_range_type,
										uint32_t p_num_descriptors,
										uint32_t p_dxil_register,
										uint32_t p_dxil_stages_mask,
										RootSignatureLocation *p_root_sig_locations,
										Vector<TraceableDescriptorTable> &r_tables,
										bool &r_first_in_set) {
				if (r_first_in_set) {
					r_tables.resize(r_tables.size() + 1);
					r_first_in_set = false;
				}

				TraceableDescriptorTable &table = r_tables.write[r_tables.size() - 1];
				table.stages_mask |= p_dxil_stages_mask;

				CD3DX12_DESCRIPTOR_RANGE1 range;
				// Due to the aliasing hack for SRV-UAV of different families,
				// we can be causing an unintended change of data (sometimes the validation layers catch it).
				D3D12_DESCRIPTOR_RANGE_FLAGS flags = D3D12_DESCRIPTOR_RANGE_FLAG_NONE;
				if (p_range_type == D3D12_DESCRIPTOR_RANGE_TYPE_SRV || p_range_type == D3D12_DESCRIPTOR_RANGE_TYPE_UAV) {
					flags = D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE;
				} else if (p_range_type == D3D12_DESCRIPTOR_RANGE_TYPE_CBV) {
					flags = D3D12_DESCRIPTOR_RANGE_FLAG_DATA_STATIC_WHILE_SET_AT_EXECUTE;
				}
				range.Init(p_range_type, p_num_descriptors, p_dxil_register, 0, flags);

				table.ranges.push_back(range);
				table.root_signature_locations.push_back(p_root_sig_locations);
			};

			uint32_t num_descriptors = 1;
			D3D12_DESCRIPTOR_RANGE_TYPE resource_range_type = {};
			switch ((ResourceClass)(uniform_d3d12.resource_class)) {
				case RES_CLASS_INVALID: {
					num_descriptors = uniform.length;
					DEV_ASSERT(uniform_d3d12.has_sampler);
				} break;
				case RES_CLASS_CBV: {
					resource_range_type = D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
					DEV_ASSERT(!uniform_d3d12.has_sampler);
				} break;
				case RES_CLASS_SRV: {
					resource_range_type = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
					num_descriptors = MAX(1u, uniform.length); // An unbound R/O buffer is reflected as zero-size.
				} break;
				case RES_CLASS_UAV: {
					resource_range_type = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
					num_descriptors = MAX(1u, uniform.length); // An unbound R/W buffer is reflected as zero-size.
					DEV_ASSERT(!uniform_d3d12.has_sampler);
				} break;
			}

			uint32_t dxil_register = i * GODOT_NIR_DESCRIPTOR_SET_MULTIPLIER + uniform.binding * GODOT_NIR_BINDING_MULTIPLIER;
			if (uniform_d3d12.resource_class != RES_CLASS_INVALID) {
				insert_range(
						resource_range_type,
						num_descriptors,
						dxil_register,
						uniform_d3d12.dxil_stages,
						&uniform_d3d12.root_signature_locations[RS_LOC_TYPE_RESOURCE],
						resource_tables_maps,
						first_resource_in_set);
			}

			if (uniform_d3d12.has_sampler) {
				insert_range(
						D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER,
						num_descriptors,
						dxil_register,
						uniform_d3d12.dxil_stages,
						&uniform_d3d12.root_signature_locations[RS_LOC_TYPE_SAMPLER],
						sampler_tables_maps,
						first_sampler_in_set);
			}
		}

		binding_start += uniform_count;
	}

	auto make_descriptor_tables = [&root_params](const Vector<TraceableDescriptorTable> &p_tables) {
		for (const TraceableDescriptorTable &table : p_tables) {
			D3D12_SHADER_VISIBILITY visibility = stages_to_d3d12_visibility(table.stages_mask);
			DEV_ASSERT(table.ranges.size() == table.root_signature_locations.size());
			for (int i = 0; i < table.ranges.size(); i++) {
				// By now we know very well which root signature location corresponds to the pointed uniform.
				table.root_signature_locations[i]->root_param_index = root_params.size();
				table.root_signature_locations[i]->range_index = i;
			}

			CD3DX12_ROOT_PARAMETER1 root_table;
			root_table.InitAsDescriptorTable(table.ranges.size(), table.ranges.ptr(), visibility);
			root_params.push_back(root_table);
		}
	};

	make_descriptor_tables(resource_tables_maps);
	make_descriptor_tables(sampler_tables_maps);

	CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC root_sig_desc = {};
	D3D12_ROOT_SIGNATURE_FLAGS root_sig_flags =
			D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS |
			D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS |
			D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS |
			D3D12_ROOT_SIGNATURE_FLAG_DENY_AMPLIFICATION_SHADER_ROOT_ACCESS |
			D3D12_ROOT_SIGNATURE_FLAG_DENY_MESH_SHADER_ROOT_ACCESS;

	if (!p_stages_processed.has_flag(RenderingDeviceCommons::SHADER_STAGE_VERTEX_BIT)) {
		root_sig_flags |= D3D12_ROOT_SIGNATURE_FLAG_DENY_VERTEX_SHADER_ROOT_ACCESS;
	}

	if (!p_stages_processed.has_flag(RenderingDeviceCommons::SHADER_STAGE_FRAGMENT_BIT)) {
		root_sig_flags |= D3D12_ROOT_SIGNATURE_FLAG_DENY_PIXEL_SHADER_ROOT_ACCESS;
	}

	if (reflection_data.vertex_input_mask) {
		root_sig_flags |= D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
	}

	root_sig_desc.Init_1_1(root_params.size(), root_params.ptr(), 0, nullptr, root_sig_flags);

	// Create and store the root signature and its CRC32.
	ID3DBlob *error_blob = nullptr;
	ID3DBlob *root_sig_blob = nullptr;
	HRESULT res = D3DX12SerializeVersionedRootSignature(HMODULE(lib_d3d12), &root_sig_desc, D3D_ROOT_SIGNATURE_VERSION_1_1, &root_sig_blob, &error_blob);
	if (SUCCEEDED(res)) {
		root_signature_bytes.resize(root_sig_blob->GetBufferSize());
		memcpy(root_signature_bytes.ptrw(), root_sig_blob->GetBufferPointer(), root_sig_blob->GetBufferSize());

		root_signature_crc = crc32(0, nullptr, 0);
		root_signature_crc = crc32(root_signature_crc, (const Bytef *)root_sig_blob->GetBufferPointer(), root_sig_blob->GetBufferSize());

		return true;
	} else {
		if (root_sig_blob != nullptr) {
			root_sig_blob->Release();
		}

		String error_string;
		if (error_blob != nullptr) {
			error_string = vformat("Serialization of root signature failed with error 0x%08ux and the following message:\n%s", uint32_t(res), String::ascii(Span((char *)error_blob->GetBufferPointer(), error_blob->GetBufferSize())));
			error_blob->Release();
		} else {
			error_string = vformat("Serialization of root signature failed with error 0x%08ux", uint32_t(res));
		}

		ERR_FAIL_V_MSG(false, error_string);
	}
}

void RenderingShaderContainerD3D12::_nir_report_resource(uint32_t p_register, uint32_t p_space, uint32_t p_dxil_type, void *p_data) {
	const GodotNirCallbackUserData &user_data = *(GodotNirCallbackUserData *)p_data;

	// Types based on Mesa's dxil_container.h.
	static const uint32_t DXIL_RES_SAMPLER = 1;
	static const ResourceClass DXIL_TYPE_TO_CLASS[] = {
		RES_CLASS_INVALID, // DXIL_RES_INVALID
		RES_CLASS_INVALID, // DXIL_RES_SAMPLER
		RES_CLASS_CBV, // DXIL_RES_CBV
		RES_CLASS_SRV, // DXIL_RES_SRV_TYPED
		RES_CLASS_SRV, // DXIL_RES_SRV_RAW
		RES_CLASS_SRV, // DXIL_RES_SRV_STRUCTURED
		RES_CLASS_UAV, // DXIL_RES_UAV_TYPED
		RES_CLASS_UAV, // DXIL_RES_UAV_RAW
		RES_CLASS_UAV, // DXIL_RES_UAV_STRUCTURED
		RES_CLASS_INVALID, // DXIL_RES_UAV_STRUCTURED_WITH_COUNTER
	};

	DEV_ASSERT(p_dxil_type < ARRAY_SIZE(DXIL_TYPE_TO_CLASS));
	ResourceClass resource_class = DXIL_TYPE_TO_CLASS[p_dxil_type];

	if (p_register == ROOT_CONSTANT_REGISTER && p_space == 0) {
		DEV_ASSERT(resource_class == RES_CLASS_CBV);
		user_data.container->reflection_data_d3d12.dxil_push_constant_stages |= (1 << user_data.stage);
	} else if (p_register == RUNTIME_DATA_REGISTER && p_space == 0) {
		DEV_ASSERT(resource_class == RES_CLASS_CBV);
		user_data.container->reflection_data_d3d12.nir_runtime_data_root_param_idx = 1; // Temporary, to be determined later.
	} else {
		DEV_ASSERT(p_space == 0);

		uint32_t set = p_register / GODOT_NIR_DESCRIPTOR_SET_MULTIPLIER;
		uint32_t binding = (p_register % GODOT_NIR_DESCRIPTOR_SET_MULTIPLIER) / GODOT_NIR_BINDING_MULTIPLIER;

		DEV_ASSERT(set < (uint32_t)user_data.container->reflection_binding_set_uniforms_count.size());

		uint32_t binding_start = 0;
		for (uint32_t i = 0; i < set; i++) {
			binding_start += user_data.container->reflection_binding_set_uniforms_count[i];
		}

		[[maybe_unused]] bool found = false;
		for (uint32_t i = 0; i < user_data.container->reflection_binding_set_uniforms_count[set]; i++) {
			const ReflectionBindingData &uniform = user_data.container->reflection_binding_set_uniforms_data[binding_start + i];
			ReflectionBindingDataD3D12 &uniform_d3d12 = user_data.container->reflection_binding_set_uniforms_data_d3d12.ptrw()[binding_start + i];
			if (uniform.binding != binding) {
				continue;
			}

			uniform_d3d12.dxil_stages |= (1 << user_data.stage);
			if (resource_class != RES_CLASS_INVALID) {
				DEV_ASSERT(uniform_d3d12.resource_class == (uint32_t)RES_CLASS_INVALID || uniform_d3d12.resource_class == (uint32_t)resource_class);
				uniform_d3d12.resource_class = resource_class;
			} else if (p_dxil_type == DXIL_RES_SAMPLER) {
				uniform_d3d12.has_sampler = (uint32_t)true;
			} else {
				DEV_ASSERT(false && "Unknown resource class.");
			}
			found = true;
		}

		DEV_ASSERT(found);
	}
}

void RenderingShaderContainerD3D12::_nir_report_sc_bit_offset(uint32_t p_sc_id, uint64_t p_bit_offset, void *p_data) {
	const GodotNirCallbackUserData &user_data = *(GodotNirCallbackUserData *)p_data;
	[[maybe_unused]] bool found = false;
	for (int64_t i = 0; i < user_data.container->reflection_specialization_data.size(); i++) {
		const ReflectionSpecializationData &sc = user_data.container->reflection_specialization_data[i];
		ReflectionSpecializationDataD3D12 &sc_d3d12 = user_data.container->reflection_specialization_data_d3d12.ptrw()[i];
		if (sc.constant_id != p_sc_id) {
			continue;
		}

		uint32_t offset_idx = SHADER_STAGES_BIT_OFFSET_INDICES[user_data.stage];
		DEV_ASSERT(sc_d3d12.stages_bit_offsets[offset_idx] == 0);
		sc_d3d12.stages_bit_offsets[offset_idx] = p_bit_offset;
		found = true;
		break;
	}

	DEV_ASSERT(found);
}

void RenderingShaderContainerD3D12::_nir_report_bitcode_bit_offset(uint64_t p_bit_offset, void *p_data) {
	DEV_ASSERT(p_bit_offset % 8 == 0);

	const GodotNirCallbackUserData &user_data = *(GodotNirCallbackUserData *)p_data;
	uint32_t offset_idx = SHADER_STAGES_BIT_OFFSET_INDICES[user_data.stage];
	for (int64_t i = 0; i < user_data.container->reflection_specialization_data.size(); i++) {
		ReflectionSpecializationDataD3D12 &sc_d3d12 = user_data.container->reflection_specialization_data_d3d12.ptrw()[i];
		if (sc_d3d12.stages_bit_offsets[offset_idx] == 0) {
			// This SC has been optimized out from this stage.
			continue;
		}

		sc_d3d12.stages_bit_offsets[offset_idx] += p_bit_offset;
	}
}
#endif

void RenderingShaderContainerD3D12::_set_from_shader_reflection_post(const String &p_shader_name, const RenderingDeviceCommons::ShaderReflection &p_reflection) {
	reflection_binding_set_uniforms_data_d3d12.resize(reflection_binding_set_uniforms_data.size());
	reflection_specialization_data_d3d12.resize(reflection_specialization_data.size());

	// Sort bindings inside each uniform set. This guarantees the root signature will be generated in the correct order.
	SortArray<ReflectionBindingData> sorter;
	uint32_t binding_start = 0;
	for (uint32_t i = 0; i < reflection_binding_set_uniforms_count.size(); i++) {
		uint32_t uniform_count = reflection_binding_set_uniforms_count[i];
		if (uniform_count > 0) {
			sorter.sort(&reflection_binding_set_uniforms_data.ptrw()[binding_start], uniform_count);
			binding_start += uniform_count;
		}
	}
}

bool RenderingShaderContainerD3D12::_set_code_from_spirv(const Vector<RenderingDeviceCommons::ShaderStageSPIRVData> &p_spirv) {
#if NIR_ENABLED
	reflection_data_d3d12.nir_runtime_data_root_param_idx = UINT32_MAX;

	for (int64_t i = 0; i < reflection_specialization_data.size(); i++) {
		DEV_ASSERT(reflection_specialization_data[i].constant_id < (sizeof(reflection_data_d3d12.spirv_specialization_constants_ids_mask) * 8) && "Constant IDs with values above 31 are not supported.");
		reflection_data_d3d12.spirv_specialization_constants_ids_mask |= (1 << reflection_specialization_data[i].constant_id);
	}

	// Translate SPIR-V shaders to DXIL, and collect shader info from the new representation.
	HashMap<RenderingDeviceCommons::ShaderStage, Vector<uint8_t>> dxil_blobs;
	Vector<RenderingDeviceCommons::ShaderStage> stages;
	BitField<RenderingDeviceCommons::ShaderStage> stages_processed = {};
	if (!_convert_spirv_to_dxil(p_spirv, dxil_blobs, stages, stages_processed)) {
		return false;
	}

	// Patch with default values of specialization constants.
	DEV_ASSERT(reflection_specialization_data.size() == reflection_specialization_data_d3d12.size());
	for (int32_t i = 0; i < reflection_specialization_data.size(); i++) {
		const ReflectionSpecializationData &sc = reflection_specialization_data[i];
		const ReflectionSpecializationDataD3D12 &sc_d3d12 = reflection_specialization_data_d3d12[i];
		RenderingDXIL::patch_specialization_constant((RenderingDeviceCommons::PipelineSpecializationConstantType)(sc.type), &sc.int_value, sc_d3d12.stages_bit_offsets, dxil_blobs, true);
	}

	// Sign.
	uint32_t shader_index = 0;
	for (KeyValue<RenderingDeviceCommons::ShaderStage, Vector<uint8_t>> &E : dxil_blobs) {
		RenderingDXIL::sign_bytecode(E.key, E.value);
	}

	// Store compressed DXIL blobs as the shaders.
	shaders.resize(p_spirv.size());
	for (int64_t i = 0; i < shaders.size(); i++) {
		const PackedByteArray &dxil_bytes = dxil_blobs[stages[i]];
		RenderingShaderContainer::Shader &shader = shaders.ptrw()[i];
		uint32_t compressed_size = 0;
		shader.shader_stage = stages[i];
		shader.code_decompressed_size = dxil_bytes.size();
		shader.code_compressed_bytes.resize(dxil_bytes.size());

		bool compressed = compress_code(dxil_bytes.ptr(), dxil_bytes.size(), shader.code_compressed_bytes.ptrw(), &compressed_size, &shader.code_compression_flags);
		ERR_FAIL_COND_V_MSG(!compressed, false, vformat("Failed to compress native code to native for SPIR-V #%d.", shader_index));

		shader.code_compressed_bytes.resize(compressed_size);
	}

	if (!_generate_root_signature(stages_processed)) {
		return false;
	}

	return true;
#else
	ERR_FAIL_V_MSG(false, "Shader compilation is not supported at runtime without NIR.");
#endif
}

RenderingShaderContainerD3D12::RenderingShaderContainerD3D12() {
	// Default empty constructor.
}

RenderingShaderContainerD3D12::RenderingShaderContainerD3D12(void *p_lib_d3d12) {
	lib_d3d12 = p_lib_d3d12;
}

RenderingShaderContainerD3D12::ShaderReflectionD3D12 RenderingShaderContainerD3D12::get_shader_reflection_d3d12() const {
	ShaderReflectionD3D12 reflection;
	reflection.spirv_specialization_constants_ids_mask = reflection_data_d3d12.spirv_specialization_constants_ids_mask;
	reflection.dxil_push_constant_stages = reflection_data_d3d12.dxil_push_constant_stages;
	reflection.nir_runtime_data_root_param_idx = reflection_data_d3d12.nir_runtime_data_root_param_idx;
	reflection.reflection_specialization_data_d3d12 = reflection_specialization_data_d3d12;
	reflection.root_signature_bytes = root_signature_bytes;
	reflection.root_signature_crc = root_signature_crc;

	// Transform data vector into a vector of vectors that's easier to user.
	uint32_t uniform_index = 0;
	reflection.reflection_binding_set_uniforms_d3d12.resize(reflection_binding_set_uniforms_count.size());
	for (int64_t i = 0; i < reflection.reflection_binding_set_uniforms_d3d12.size(); i++) {
		Vector<ReflectionBindingDataD3D12> &uniforms = reflection.reflection_binding_set_uniforms_d3d12.ptrw()[i];
		uniforms.resize(reflection_binding_set_uniforms_count[i]);
		for (int64_t j = 0; j < uniforms.size(); j++) {
			uniforms.ptrw()[j] = reflection_binding_set_uniforms_data_d3d12[uniform_index];
			uniform_index++;
		}
	}

	return reflection;
}

// RenderingShaderContainerFormatD3D12

void RenderingShaderContainerFormatD3D12::set_lib_d3d12(void *p_lib_d3d12) {
	lib_d3d12 = p_lib_d3d12;
}

Ref<RenderingShaderContainer> RenderingShaderContainerFormatD3D12::create_container() const {
	return memnew(RenderingShaderContainerD3D12(lib_d3d12));
}

RenderingDeviceCommons::ShaderLanguageVersion RenderingShaderContainerFormatD3D12::get_shader_language_version() const {
	// NIR-DXIL is Vulkan 1.1-conformant.
	return SHADER_LANGUAGE_VULKAN_VERSION_1_1;
}

RenderingDeviceCommons::ShaderSpirvVersion RenderingShaderContainerFormatD3D12::get_shader_spirv_version() const {
	// The SPIR-V part of Mesa supports 1.6, but:
	// - SPIRV-Reflect won't be able to parse the compute workgroup size.
	// - We want to play it safe with NIR-DXIL.
	return SHADER_SPIRV_VERSION_1_5;
}

RenderingShaderContainerFormatD3D12::RenderingShaderContainerFormatD3D12() {}

RenderingShaderContainerFormatD3D12::~RenderingShaderContainerFormatD3D12() {}
