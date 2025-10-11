/**************************************************************************/
/*  ffx_common.cpp                                                        */
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

#include "ffx_common.h"

#include "servers/rendering/renderer_rd/uniform_set_cache_rd.h"
#include "thirdparty/amd-ffx/ffx_fsr1.h"
#include "thirdparty/amd-ffx/ffx_fsr2.h"

using namespace RendererRD;

RD::TextureType FFXCommonContext::ffx_resource_type_to_rd_texture_type(FfxResourceType p_type) {
	switch (p_type) {
		case FFX_RESOURCE_TYPE_TEXTURE1D:
			return RD::TEXTURE_TYPE_1D;
		case FFX_RESOURCE_TYPE_TEXTURE2D:
			return RD::TEXTURE_TYPE_2D;
		case FFX_RESOURCE_TYPE_TEXTURE3D:
			return RD::TEXTURE_TYPE_3D;
		default:
#ifdef DEV_ENABLED
			ERR_PRINT("Unknown FFX resource type.");
#endif
			return RD::TEXTURE_TYPE_MAX;
	}
}

FfxResourceType FFXCommonContext::rd_texture_type_to_ffx_resource_type(RD::TextureType p_type) {
	switch (p_type) {
		case RD::TEXTURE_TYPE_1D:
			return FFX_RESOURCE_TYPE_TEXTURE1D;
		case RD::TEXTURE_TYPE_2D:
			return FFX_RESOURCE_TYPE_TEXTURE2D;
		case RD::TEXTURE_TYPE_3D:
			return FFX_RESOURCE_TYPE_TEXTURE3D;
		default:
#ifdef DEV_ENABLED
			ERR_PRINT("Unknown FFX resource type.");
#endif
			return FFX_RESOURCE_TYPE_BUFFER;
	}
}

RD::DataFormat FFXCommonContext::ffx_surface_format_to_rd_format(FfxSurfaceFormat p_format) {
	switch (p_format) {
		case FFX_SURFACE_FORMAT_R32G32B32A32_TYPELESS:
			return RD::DATA_FORMAT_R32G32B32A32_SFLOAT;
		case FFX_SURFACE_FORMAT_R32G32B32A32_FLOAT:
			return RD::DATA_FORMAT_R32G32B32A32_SFLOAT;
		case FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT:
			return RD::DATA_FORMAT_R16G16B16A16_SFLOAT;
		case FFX_SURFACE_FORMAT_R32G32_FLOAT:
			return RD::DATA_FORMAT_R32G32_SFLOAT;
		case FFX_SURFACE_FORMAT_R32_UINT:
			return RD::DATA_FORMAT_R32_UINT;
		case FFX_SURFACE_FORMAT_R8G8B8A8_TYPELESS:
			return RD::DATA_FORMAT_R8G8B8A8_UNORM;
		case FFX_SURFACE_FORMAT_R8G8B8A8_UNORM:
			return RD::DATA_FORMAT_R8G8B8A8_UNORM;
		case FFX_SURFACE_FORMAT_R11G11B10_FLOAT:
			return RD::DATA_FORMAT_B10G11R11_UFLOAT_PACK32;
		case FFX_SURFACE_FORMAT_R16G16_FLOAT:
			return RD::DATA_FORMAT_R16G16_SFLOAT;
		case FFX_SURFACE_FORMAT_R16G16_UINT:
			return RD::DATA_FORMAT_R16G16_UINT;
		case FFX_SURFACE_FORMAT_R16_FLOAT:
			return RD::DATA_FORMAT_R16_SFLOAT;
		case FFX_SURFACE_FORMAT_R16_UINT:
			return RD::DATA_FORMAT_R16_UINT;
		case FFX_SURFACE_FORMAT_R16_UNORM:
			return RD::DATA_FORMAT_R16_UNORM;
		case FFX_SURFACE_FORMAT_R16_SNORM:
			return RD::DATA_FORMAT_R16_SNORM;
		case FFX_SURFACE_FORMAT_R8_UNORM:
			return RD::DATA_FORMAT_R8_UNORM;
		case FFX_SURFACE_FORMAT_R8_UINT:
			return RD::DATA_FORMAT_R8_UINT;
		case FFX_SURFACE_FORMAT_R8G8_UNORM:
			return RD::DATA_FORMAT_R8G8_UNORM;
		case FFX_SURFACE_FORMAT_R32_FLOAT:
			return RD::DATA_FORMAT_R32_SFLOAT;
		default:
#ifdef DEV_ENABLED
			ERR_PRINT("Unknown FFX resource type.");
#endif
			return RD::DATA_FORMAT_MAX;
	}
}

FfxSurfaceFormat FFXCommonContext::rd_format_to_ffx_surface_format(RD::DataFormat p_format) {
	switch (p_format) {
		case RD::DATA_FORMAT_R32G32B32A32_SFLOAT:
			return FFX_SURFACE_FORMAT_R32G32B32A32_FLOAT;
		case RD::DATA_FORMAT_R16G16B16A16_SFLOAT:
			return FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT;
		case RD::DATA_FORMAT_R32G32_SFLOAT:
			return FFX_SURFACE_FORMAT_R32G32_FLOAT;
		case RD::DATA_FORMAT_R32_UINT:
			return FFX_SURFACE_FORMAT_R32_UINT;
		case RD::DATA_FORMAT_R8G8B8A8_UNORM:
			return FFX_SURFACE_FORMAT_R8G8B8A8_UNORM;
		case RD::DATA_FORMAT_B10G11R11_UFLOAT_PACK32:
			return FFX_SURFACE_FORMAT_R11G11B10_FLOAT;
		case RD::DATA_FORMAT_R16G16_SFLOAT:
			return FFX_SURFACE_FORMAT_R16G16_FLOAT;
		case RD::DATA_FORMAT_R16G16_UINT:
			return FFX_SURFACE_FORMAT_R16G16_UINT;
		case RD::DATA_FORMAT_R16_SFLOAT:
			return FFX_SURFACE_FORMAT_R16_FLOAT;
		case RD::DATA_FORMAT_R16_UINT:
			return FFX_SURFACE_FORMAT_R16_UINT;
		case RD::DATA_FORMAT_R16_UNORM:
			return FFX_SURFACE_FORMAT_R16_UNORM;
		case RD::DATA_FORMAT_R16_SNORM:
			return FFX_SURFACE_FORMAT_R16_SNORM;
		case RD::DATA_FORMAT_R8_UNORM:
			return FFX_SURFACE_FORMAT_R8_UNORM;
		case RD::DATA_FORMAT_R8_UINT:
			return FFX_SURFACE_FORMAT_R8_UINT;
		case RD::DATA_FORMAT_R8G8_UNORM:
			return FFX_SURFACE_FORMAT_R8G8_UNORM;
		case RD::DATA_FORMAT_R32_SFLOAT:
			return FFX_SURFACE_FORMAT_R32_FLOAT;
		default:
			return FFX_SURFACE_FORMAT_UNKNOWN;
	}
}

static uint32_t ffx_usage_to_rd_usage_flags(uint32_t p_flags) {
	uint32_t ret = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT;

	if (p_flags & FFX_RESOURCE_USAGE_RENDERTARGET) {
		ret |= RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT;
	}

	if (p_flags & FFX_RESOURCE_USAGE_UAV) {
		ret |= RD::TEXTURE_USAGE_STORAGE_BIT;
		ret |= RD::TEXTURE_USAGE_CAN_COPY_FROM_BIT;
		ret |= RD::TEXTURE_USAGE_CAN_COPY_TO_BIT;
	}

	return ret;
}

static FfxVersionNumber get_sdk_version_rd(FfxInterface* backendInterface) {
	return FFX_SDK_MAKE_VERSION(FFX_SDK_VERSION_MAJOR, FFX_SDK_VERSION_MINOR, FFX_SDK_VERSION_PATCH);
}

static FfxErrorCode create_backend_context_rd(FfxInterface *p_backend_interface, FfxEffect p_effect,
	FfxEffectBindlessConfig* p_bindless_config, FfxUInt32* p_effect_context_id) {
	FFXCommonContext::Scratch &scratch = *reinterpret_cast<FFXCommonContext::Scratch *>(p_backend_interface->scratchBuffer);

	if (p_bindless_config) {
		WARN_PRINT_ONCE("Fidelity FX: Bindless resources are not supported in Godot.");
	}

	// Store pointer to the device common to all contexts.
	scratch.device = p_backend_interface->device;
	scratch.staging_constant_buffer = (uint8_t*)memalloc(FFX_CONSTANT_BUFFER_RING_BUFFER_SIZE);
	ERR_FAIL_NULL_V(scratch.staging_constant_buffer, FFX_ERROR_OUT_OF_MEMORY);

	// Create a ring buffer of uniform buffers.
	// FIXME: This could be optimized to be a single memory block if it was possible for RD to create views into a particular memory range of a UBO.
	for (uint32_t i = 0; i < FFX_UBO_RING_BUFFER_SIZE; i++) {
		scratch.ubo_ring_buffer[i] = RD::get_singleton()->uniform_buffer_create(FFX_BUFFER_SIZE);
		ERR_FAIL_COND_V(scratch.ubo_ring_buffer[i].is_null(), FFX_ERROR_BACKEND_API_ERROR);
	}

	switch (p_effect) {
		case FFX_EFFECT_FSR1:
			*p_effect_context_id = FFX_EFFECT_CONTEXT_FSR1;
			break;
		case FFX_EFFECT_FSR2:
			*p_effect_context_id = FFX_EFFECT_CONTEXT_FSR2;
			break;
		case FFX_EFFECT_FSR3UPSCALER:
			*p_effect_context_id = FFX_EFFECT_CONTEXT_FSR3_UPSCALE;
			break;
		case FFX_EFFECT_FRAMEINTERPOLATION:
			*p_effect_context_id = FFX_EFFECT_CONTEXT_FSR3_INTERPOLATE;
			break;
		default:
			ERR_PRINT("Unknown FFX effect.");
			return FFX_ERROR_INVALID_ARGUMENT;
	}

	return FFX_OK;
}

static FfxErrorCode get_device_capabilities_rd(FfxInterface *p_backend_interface, FfxDeviceCapabilities *p_out_device_capabilities) {
	FFXCommonContext::Device &effect_device = *reinterpret_cast<FFXCommonContext::Device *>(p_backend_interface->device);

	*p_out_device_capabilities = effect_device.capabilities;

	return FFX_OK;
}

static FfxErrorCode destroy_backend_context_rd(FfxInterface *p_backend_interface, FfxUInt32 effect_context_id) {
	FFXCommonContext::Scratch &scratch = *reinterpret_cast<FFXCommonContext::Scratch *>(p_backend_interface->scratchBuffer);
	if (scratch.staging_constant_buffer) {
		memfree(scratch.staging_constant_buffer);
	}

	for (uint32_t i = 0; i < FFX_UBO_RING_BUFFER_SIZE; i++) {
		RD::get_singleton()->free_rid(scratch.ubo_ring_buffer[i]);
	}

	return FFX_OK;
}

static FfxErrorCode create_resource_rd(FfxInterface *p_backend_interface, const FfxCreateResourceDescription *p_create_resource_description, FfxUInt32 effect_context_id, FfxResourceInternal *p_out_resource) {
	// FSR2's base implementation won't issue a call to create a heap type that isn't just default on its own,
	// so we can safely ignore it as RD does not expose this concept.
	ERR_FAIL_COND_V(p_create_resource_description->heapType != FFX_HEAP_TYPE_DEFAULT, FFX_ERROR_INVALID_ARGUMENT);

	RenderingDevice *rd = RD::get_singleton();
	FFXCommonContext::Scratch &scratch = *reinterpret_cast<FFXCommonContext::Scratch *>(p_backend_interface->scratchBuffer);
	FfxResourceDescription res_desc = p_create_resource_description->resourceDescription;

	// FSR2's base implementation never requests buffer creation.
	ERR_FAIL_COND_V(res_desc.type != FFX_RESOURCE_TYPE_TEXTURE1D && res_desc.type != FFX_RESOURCE_TYPE_TEXTURE2D && res_desc.type != FFX_RESOURCE_TYPE_TEXTURE3D, FFX_ERROR_INVALID_ARGUMENT);

	if (res_desc.mipCount == 0) {
		// Mipmap count must be derived from the resource's dimensions.
		res_desc.mipCount = uint32_t(1 + std::floor(std::log2(MAX(MAX(res_desc.width, res_desc.height), res_desc.depth))));
	}

#ifdef DEV_ENABLED
	if (p_create_resource_description->initData.type == FFX_RESOURCE_INIT_DATA_TYPE_INVALID) {
		ERR_PRINT("Invalid initial data type. ");
	}
#endif

	Vector<PackedByteArray> initial_data;
	if (p_create_resource_description->initData.type != FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED) {
		PackedByteArray byte_array;
		byte_array.resize(p_create_resource_description->initData.size);
		switch (p_create_resource_description->initData.type) {
			case FFX_RESOURCE_INIT_DATA_TYPE_BUFFER:
				memcpy(byte_array.ptrw(), p_create_resource_description->initData.buffer, p_create_resource_description->initData.size);
				break;
			case FFX_RESOURCE_INIT_DATA_TYPE_VALUE:
				memcpy(byte_array.ptrw(), &p_create_resource_description->initData.value, p_create_resource_description->initData.size);
				break;
		}
		initial_data.push_back(byte_array);
	}

	RD::TextureFormat texture_format;
	texture_format.texture_type = FFXCommonContext::ffx_resource_type_to_rd_texture_type(res_desc.type);
	texture_format.format = FFXCommonContext::ffx_surface_format_to_rd_format(res_desc.format);
	texture_format.usage_bits = ffx_usage_to_rd_usage_flags(p_create_resource_description->resourceDescription.usage);
	texture_format.width = res_desc.width;
	texture_format.height = res_desc.height;
	texture_format.depth = res_desc.depth;
	texture_format.mipmaps = res_desc.mipCount;
	texture_format.is_discardable = true;

	RID texture = rd->texture_create(texture_format, RD::TextureView(), initial_data);
	ERR_FAIL_COND_V(texture.is_null(), FFX_ERROR_BACKEND_API_ERROR);

	rd->set_resource_name(texture, String(p_create_resource_description->name));

	// Add the resource to the storage and use the internal index to reference it.
	p_out_resource->internalIndex = scratch.resources.add(texture, false, p_create_resource_description->id, res_desc);

	return FFX_OK;
}

static FfxErrorCode register_resource_rd(FfxInterface *p_backend_interface, const FfxResource *p_in_resource, FfxUInt32 effect_context_id, FfxResourceInternal *p_out_resource) {
	if (p_in_resource->resource == nullptr) {
		// Null resource case.
		p_out_resource->internalIndex = -1;
		return FFX_OK;
	}

	FFXCommonContext::Scratch &scratch = *reinterpret_cast<FFXCommonContext::Scratch *>(p_backend_interface->scratchBuffer);
	const RID &rid = *reinterpret_cast<const RID *>(p_in_resource->resource);
	ERR_FAIL_COND_V(rid.is_null(), FFX_ERROR_INVALID_ARGUMENT);

	// Add the resource to the storage and use the internal index to reference it.
	p_out_resource->internalIndex = scratch.resources.add(rid, true, FFXCommonContext::RESOURCE_ID_DYNAMIC, p_in_resource->description);

	return FFX_OK;
}

static FfxErrorCode unregister_resources_rd(FfxInterface *p_backend_interface, FfxCommandList p_command_list, FfxUInt32 effect_context_id) {
	FFXCommonContext::Scratch &scratch = *reinterpret_cast<FFXCommonContext::Scratch *>(p_backend_interface->scratchBuffer);
	LocalVector<uint32_t> dynamic_list_copy = scratch.resources.dynamic_list;
	for (uint32_t i : dynamic_list_copy) {
		scratch.resources.remove(i);
	}

	return FFX_OK;
}

static FfxResourceDescription get_resource_description_rd(FfxInterface *p_backend_interface, FfxResourceInternal p_resource) {
	if (p_resource.internalIndex != -1) {
		FFXCommonContext::Scratch &scratch = *reinterpret_cast<FFXCommonContext::Scratch *>(p_backend_interface->scratchBuffer);
		return scratch.resources.descriptions[p_resource.internalIndex];
	} else {
		return {};
	}
}

static FfxErrorCode destroy_resource_rd(FfxInterface *p_backend_interface, FfxResourceInternal p_resource, FfxUInt32 effect_context_id) {
	if (p_resource.internalIndex != -1) {
		FFXCommonContext::Scratch &scratch = *reinterpret_cast<FFXCommonContext::Scratch *>(p_backend_interface->scratchBuffer);
		if (scratch.resources.rids[p_resource.internalIndex].is_valid()) {
			RD::get_singleton()->free_rid(scratch.resources.rids[p_resource.internalIndex]);
			scratch.resources.remove(p_resource.internalIndex);
		}
	}

	return FFX_OK;
}

static FfxErrorCode create_pipeline_rd(FfxInterface *p_backend_interface, FfxEffect p_effect, FfxPass p_pass,  uint32_t p_permutation_options, const FfxPipelineDescription *p_pipeline_description, FfxUInt32 p_effect_context_id, FfxPipelineState *p_out_pipeline) {
	FFXCommonContext::Scratch &scratch = *reinterpret_cast<FFXCommonContext::Scratch *>(p_backend_interface->scratchBuffer);
	FFXCommonContext::Device &device = *reinterpret_cast<FFXCommonContext::Device *>(scratch.device);

	if (p_effect == FFX_EFFECT_FSR1 && p_pass == FFX_FSR1_PASS_EASU_RCAS) {
		// `EASU_RCAS` and `EASU` are basically variants of a same thing and thus shall share the same pipeline
		p_pass = FFX_FSR1_PASS_EASU;
	}

	FFXCommonContext::Pass &effect_pass = device.effect_contexts[p_effect_context_id].passes[p_pass];

	if (effect_pass.pipeline.pipeline_rid.is_null()) {
		// Create pipeline for the device if it hasn't been created yet.
		effect_pass.root_signature.shader_rid = effect_pass.shader->version_get_shader(effect_pass.shader_version, effect_pass.shader_variant);
		ERR_FAIL_COND_V(effect_pass.root_signature.shader_rid.is_null(), FFX_ERROR_BACKEND_API_ERROR);

		effect_pass.pipeline.pipeline_rid = RD::get_singleton()->compute_pipeline_create(effect_pass.root_signature.shader_rid);
		ERR_FAIL_COND_V(effect_pass.pipeline.pipeline_rid.is_null(), FFX_ERROR_BACKEND_API_ERROR);
	}

#ifdef DEV_ENABLED
	memcpy(p_out_pipeline->name, p_pipeline_description->name, sizeof(p_out_pipeline->name));
#endif

	// While this is not their intended use, we use the pipeline and root signature pointers to store the
	// RIDs to the pipeline and shader that RD needs for the compute pipeline.
	p_out_pipeline->pipeline = reinterpret_cast<FfxPipeline>(&effect_pass.pipeline);
	p_out_pipeline->rootSignature = reinterpret_cast<FfxRootSignature>(&effect_pass.root_signature);

	// FSR doesn't use any buffers
	p_out_pipeline->srvBufferCount = 0;
	p_out_pipeline->srvTextureCount = effect_pass.sampled_texture_bindings.size();
	ERR_FAIL_COND_V(p_out_pipeline->srvTextureCount + p_out_pipeline->srvBufferCount > FFX_MAX_NUM_SRVS, FFX_ERROR_OUT_OF_RANGE);
	memcpy(p_out_pipeline->srvTextureBindings, effect_pass.sampled_texture_bindings.ptr(), sizeof(FfxResourceBinding) * p_out_pipeline->srvTextureCount);

	// FSR doesn't use any buffers
	p_out_pipeline->uavBufferCount = 0;
	p_out_pipeline->uavTextureCount = effect_pass.storage_texture_bindings.size();
	ERR_FAIL_COND_V(p_out_pipeline->uavTextureCount + p_out_pipeline->uavBufferCount > FFX_MAX_NUM_UAVS, FFX_ERROR_OUT_OF_RANGE);
	memcpy(p_out_pipeline->uavTextureBindings, effect_pass.storage_texture_bindings.ptr(), sizeof(FfxResourceBinding) * p_out_pipeline->uavTextureCount);

	p_out_pipeline->constCount = effect_pass.uniform_bindings.size();
	ERR_FAIL_COND_V(p_out_pipeline->constCount > FFX_MAX_NUM_CONST_BUFFERS, FFX_ERROR_OUT_OF_RANGE);
	memcpy(p_out_pipeline->constantBufferBindings, effect_pass.uniform_bindings.ptr(), sizeof(FfxResourceBinding) * p_out_pipeline->constCount);

	if (p_effect == FFX_EFFECT_FSR2) {
		bool low_resolution_mvs = (p_pipeline_description->contextFlags & FFX_FSR2_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS) == 0;

		if (p_pass == FFX_FSR2_PASS_ACCUMULATE || p_pass == FFX_FSR2_PASS_ACCUMULATE_SHARPEN) {
			// Change the binding for motion vectors in this particular pass if low resolution MVs are used.
			if (low_resolution_mvs) {
				FfxResourceBinding &binding = p_out_pipeline->srvTextureBindings[2];
				wcscpy_s(binding.name, L"r_dilated_motion_vectors");
			}
		}
	}

	return FFX_OK;
}

static FfxErrorCode destroy_pipeline_rd(FfxInterface *p_backend_interface, FfxPipelineState *p_pipeline, FfxUInt32 p_effect_context_id) {
	// We don't want to destroy pipelines when the FSR2 API deems it necessary as it'll do so whenever the context is destroyed.

	return FFX_OK;
}

static FfxErrorCode schedule_gpu_job_rd(FfxInterface *p_backend_interface, const FfxGpuJobDescription *p_job) {
	ERR_FAIL_NULL_V(p_backend_interface, FFX_ERROR_INVALID_ARGUMENT);
	ERR_FAIL_NULL_V(p_job, FFX_ERROR_INVALID_ARGUMENT);

	FFXCommonContext::Scratch &scratch = *reinterpret_cast<FFXCommonContext::Scratch *>(p_backend_interface->scratchBuffer);
	scratch.gpu_jobs.push_back(*p_job);

	return FFX_OK;
}

static FfxErrorCode execute_gpu_job_clear_float_rd(FFXCommonContext::Scratch &p_scratch, const FfxClearFloatJobDescription &p_job, FfxUInt32 p_effect_context_id) {
	RID resource = p_scratch.resources.rids[p_job.target.internalIndex];
	FfxResourceDescription &desc = p_scratch.resources.descriptions[p_job.target.internalIndex];

	ERR_FAIL_COND_V_MSG(desc.type == FFX_RESOURCE_TYPE_BUFFER, FFX_ERROR_INVALID_ARGUMENT, "Cannot clear a buffer resource.");

	Color color(p_job.color[0], p_job.color[1], p_job.color[2], p_job.color[3]);
	RD::get_singleton()->texture_clear(resource, color, 0, desc.mipCount, 0, 1);

	return FFX_OK;
}

static FfxErrorCode execute_gpu_job_copy_rd(FFXCommonContext::Scratch &p_scratch, const FfxCopyJobDescription &p_job, FfxUInt32 p_effect_context_id) {
	RID src = p_scratch.resources.rids[p_job.src.internalIndex];
	RID dst = p_scratch.resources.rids[p_job.dst.internalIndex];
	FfxResourceDescription &src_desc = p_scratch.resources.descriptions[p_job.src.internalIndex];
	FfxResourceDescription &dst_desc = p_scratch.resources.descriptions[p_job.dst.internalIndex];

	ERR_FAIL_COND_V(src_desc.type == FFX_RESOURCE_TYPE_BUFFER, FFX_ERROR_INVALID_ARGUMENT);
	ERR_FAIL_COND_V(dst_desc.type == FFX_RESOURCE_TYPE_BUFFER, FFX_ERROR_INVALID_ARGUMENT);

	for (uint32_t mip_level = 0; mip_level < src_desc.mipCount; mip_level++) {
		RD::get_singleton()->texture_copy(src, dst, Vector3(0, 0, 0), Vector3(0, 0, 0), Vector3(src_desc.width, src_desc.height, src_desc.depth), mip_level, mip_level, 0, 0);
	}

	return FFX_OK;
}

static FfxErrorCode execute_gpu_job_compute_rd(FFXCommonContext::Scratch &p_scratch, const FfxComputeJobDescription &p_job, FfxUInt32 p_effect_context_id) {
	UniformSetCacheRD *uniform_set_cache = UniformSetCacheRD::get_singleton();
	ERR_FAIL_NULL_V(uniform_set_cache, FFX_ERROR_BACKEND_API_ERROR);

	FFXCommonContext::RootSignature &root_signature = *reinterpret_cast<FFXCommonContext::RootSignature *>(p_job.pipeline.rootSignature);
	ERR_FAIL_COND_V(root_signature.shader_rid.is_null(), FFX_ERROR_INVALID_ARGUMENT);

	FFXCommonContext::Pipeline &backend_pipeline = *reinterpret_cast<FFXCommonContext::Pipeline *>(p_job.pipeline.pipeline);
	ERR_FAIL_COND_V(backend_pipeline.pipeline_rid.is_null(), FFX_ERROR_INVALID_ARGUMENT);

	thread_local LocalVector<RD::Uniform> compute_uniforms;
	compute_uniforms.clear();

	for (uint32_t i = 0; i < p_job.pipeline.srvTextureCount; i++) {
		RID texture_rid = p_scratch.resources.rids[p_job.srvTextures[i].resource.internalIndex];
		RD::Uniform texture_uniform(RD::UNIFORM_TYPE_TEXTURE, p_job.pipeline.srvTextureBindings[i].slotIndex, texture_rid);
		compute_uniforms.push_back(texture_uniform);
	}

	ERR_FAIL_COND_V_MSG(p_job.pipeline.srvBufferCount > 0, FFX_ERROR_BACKEND_API_ERROR, "Since FSR doesn't use buffers, SRV buffers are not supported.");

	for (uint32_t i = 0; i < p_job.pipeline.uavTextureCount; i++) {
		RID image_rid = p_scratch.resources.rids[p_job.uavTextures[i].resource.internalIndex];
		RD::Uniform storage_uniform;
		storage_uniform.uniform_type = RD::UNIFORM_TYPE_IMAGE;
		storage_uniform.binding = p_job.pipeline.uavTextureBindings[i].slotIndex;

		if (p_job.uavTextures[i].mip > 0) {
			LocalVector<RID> &mip_slice_rids = p_scratch.resources.mip_slice_rids[p_job.uavTextures[i].resource.internalIndex];
			if (mip_slice_rids.is_empty()) {
				mip_slice_rids.resize(p_scratch.resources.descriptions[p_job.uavTextures[i].resource.internalIndex].mipCount);
			}

			ERR_FAIL_COND_V(p_job.uavTextures[i].mip >= mip_slice_rids.size(), FFX_ERROR_INVALID_ARGUMENT);

			if (mip_slice_rids[p_job.uavTextures[i].mip].is_null()) {
				mip_slice_rids[p_job.uavTextures[i].mip] = RD::get_singleton()->texture_create_shared_from_slice(RD::TextureView(), image_rid, 0, p_job.uavTextures[i].mip);
			}

			ERR_FAIL_COND_V(mip_slice_rids[p_job.uavTextures[i].mip].is_null(), FFX_ERROR_BACKEND_API_ERROR);

			storage_uniform.append_id(mip_slice_rids[p_job.uavTextures[i].mip]);
		} else {
			storage_uniform.append_id(image_rid);
		}

		compute_uniforms.push_back(storage_uniform);
	}

	ERR_FAIL_COND_V_MSG(p_job.pipeline.uavBufferCount > 0, FFX_ERROR_BACKEND_API_ERROR, "Since FSR doesn't use buffers, UAV buffers are not supported.");

	for (uint32_t i = 0; i < p_job.pipeline.constCount; i++) {
		RID buffer_rid = p_scratch.ubo_ring_buffer[p_scratch.ubo_ring_buffer_index];
		p_scratch.ubo_ring_buffer_index = (p_scratch.ubo_ring_buffer_index + 1) % FFX_UBO_RING_BUFFER_SIZE;

		RD::get_singleton()->buffer_update(buffer_rid, 0, p_job.cbs[i].num32BitEntries * sizeof(uint32_t), p_job.cbs[i].data);

		RD::Uniform buffer_uniform(RD::UNIFORM_TYPE_UNIFORM_BUFFER, p_job.pipeline.constantBufferBindings[i].slotIndex, buffer_rid);
		compute_uniforms.push_back(buffer_uniform);
	}

	FFXCommonContext::Device &device = *reinterpret_cast<FFXCommonContext::Device *>(p_scratch.device);

	if (p_effect_context_id == FFX_EFFECT_CONTEXT_FSR1) {
		RD::Uniform u_linear_clamp_sampler(RD::UniformType::UNIFORM_TYPE_SAMPLER, 1000, device.linear_clamp_sampler);
		compute_uniforms.push_back(u_linear_clamp_sampler);
	} else if (p_effect_context_id == FFX_EFFECT_CONTEXT_FSR2) {
		RD::Uniform u_point_clamp_sampler(RD::UniformType::UNIFORM_TYPE_SAMPLER, 1000, device.point_clamp_sampler);
		RD::Uniform u_linear_clamp_sampler(RD::UniformType::UNIFORM_TYPE_SAMPLER, 1001, device.linear_clamp_sampler);
		compute_uniforms.push_back(u_point_clamp_sampler);
		compute_uniforms.push_back(u_linear_clamp_sampler);
	}

	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();
	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, backend_pipeline.pipeline_rid);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache_vec(root_signature.shader_rid, 0, compute_uniforms), 0);
	RD::get_singleton()->compute_list_dispatch(compute_list, p_job.dimensions[0], p_job.dimensions[1], p_job.dimensions[2]);
	RD::get_singleton()->compute_list_end();

	return FFX_OK;
}

static FfxErrorCode execute_gpu_jobs_rd(FfxInterface *p_backend_interface, FfxCommandList p_command_list, FfxUInt32 p_effect_context_id) {
	ERR_FAIL_NULL_V(p_backend_interface, FFX_ERROR_INVALID_ARGUMENT);

	FFXCommonContext::Scratch &scratch = *reinterpret_cast<FFXCommonContext::Scratch *>(p_backend_interface->scratchBuffer);
	FfxErrorCode error_code;
	for (const FfxGpuJobDescription &job : scratch.gpu_jobs) {
		switch (job.jobType) {
			case FFX_GPU_JOB_CLEAR_FLOAT: {
				error_code = execute_gpu_job_clear_float_rd(scratch, job.clearJobDescriptor, p_effect_context_id);
			} break;
			case FFX_GPU_JOB_COPY: {
				error_code = execute_gpu_job_copy_rd(scratch, job.copyJobDescriptor, p_effect_context_id);
			} break;
			case FFX_GPU_JOB_COMPUTE: {
				error_code = execute_gpu_job_compute_rd(scratch, job.computeJobDescriptor, p_effect_context_id);
			} break;
			default: {
				error_code = FFX_ERROR_INVALID_ARGUMENT;
			} break;
		}

		if (error_code != FFX_OK) {
			scratch.gpu_jobs.clear();
			return error_code;
		}
	}

	scratch.gpu_jobs.clear();

	return FFX_OK;
}

static FfxErrorCode stage_constant_buffer_data_rd(FfxInterface* p_backend_interface, void* p_data, FfxUInt32 p_size, FfxConstantBuffer* p_constant_buffer) {
	ERR_FAIL_NULL_V(p_backend_interface, FFX_ERROR_INVALID_POINTER);
	ERR_FAIL_NULL_V(p_data, FFX_ERROR_INVALID_POINTER);
	ERR_FAIL_NULL_V(p_constant_buffer, FFX_ERROR_INVALID_POINTER);

	FFXCommonContext::Scratch &scratch = *reinterpret_cast<FFXCommonContext::Scratch *>(p_backend_interface->scratchBuffer);
	if (scratch.staging_constant_buffer_base + FFX_ALIGN_UP(p_size, 256) >= FFX_CONSTANT_BUFFER_RING_BUFFER_SIZE) {
		scratch.staging_constant_buffer_base = 0;
	}

	void* dst = scratch.staging_constant_buffer + scratch.staging_constant_buffer_base;
	memcpy(dst, p_data, p_size);

	p_constant_buffer->data = (uint32_t*)dst;
	p_constant_buffer->num32BitEntries = p_size / sizeof(uint32_t);
	scratch.staging_constant_buffer_base += FFX_ALIGN_UP(p_size, 256);

	return FFX_OK;
}

FFXCommonContext *FFXCommonContext::singleton = nullptr;

FfxResource FFXCommonContext::get_resource_rd(RID *p_rid, const wchar_t *p_name) {
	FfxResource res = {};
	if (p_rid->is_null()) {
		return res;
	}

	wcscpy_s(res.name, p_name);

	RD::TextureFormat texture_format = RD::get_singleton()->texture_get_format(*p_rid);
	res.description.type = rd_texture_type_to_ffx_resource_type(texture_format.texture_type);
	res.description.format = rd_format_to_ffx_surface_format(texture_format.format);
	res.description.width = texture_format.width;
	res.description.height = texture_format.height;
	res.description.depth = texture_format.depth;
	res.description.mipCount = texture_format.mipmaps;
	res.description.flags = FFX_RESOURCE_FLAGS_NONE;
	res.resource = reinterpret_cast<void *>(p_rid);

	return res;
}

FFXCommonContext::~FFXCommonContext() {
	RD::get_singleton()->free_rid(device.point_clamp_sampler);
	RD::get_singleton()->free_rid(device.linear_clamp_sampler);
}

void FFXCommonContext::init_device() {
	device.capabilities.maximumSupportedShaderModel = FFX_SHADER_MODEL_6_7;
	device.capabilities.waveLaneCountMin = 32;
	device.capabilities.waveLaneCountMax = 32;
	device.capabilities.fp16Supported = RD::get_singleton()->has_feature(RD::Features::SUPPORTS_HALF_FLOAT);
	device.capabilities.raytracingSupported = false;

	RD::SamplerState state;
	state.mag_filter = RD::SAMPLER_FILTER_NEAREST;
	state.min_filter = RD::SAMPLER_FILTER_NEAREST;
	state.repeat_u = RD::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;
	state.repeat_v = RD::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;
	state.repeat_w = RD::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;
	state.min_lod = -1000.0f;
	state.max_lod = 1000.0f;
	state.anisotropy_max = 1.0;
	device.point_clamp_sampler = RD::get_singleton()->sampler_create(state);
	ERR_FAIL_COND(device.point_clamp_sampler.is_null());

	state.mag_filter = RD::SAMPLER_FILTER_LINEAR;
	state.min_filter = RD::SAMPLER_FILTER_LINEAR;
	device.linear_clamp_sampler = RD::get_singleton()->sampler_create(state);
	ERR_FAIL_COND(device.linear_clamp_sampler.is_null());
}

void FFXCommonContext::create_ffx_interface(FfxInterface *p_interface) {
	p_interface->fpGetSDKVersion = get_sdk_version_rd;
	p_interface->fpCreateBackendContext = create_backend_context_rd;
	p_interface->fpGetDeviceCapabilities = get_device_capabilities_rd;
	p_interface->fpDestroyBackendContext = destroy_backend_context_rd;
	p_interface->fpCreateResource = create_resource_rd;
	p_interface->fpRegisterResource = register_resource_rd;
	p_interface->fpUnregisterResources = unregister_resources_rd;
	p_interface->fpGetResourceDescription = get_resource_description_rd;
	p_interface->fpDestroyResource = destroy_resource_rd;
	p_interface->fpCreatePipeline = create_pipeline_rd;
	p_interface->fpDestroyPipeline = destroy_pipeline_rd;
	p_interface->fpScheduleGpuJob = schedule_gpu_job_rd;
	p_interface->fpExecuteGpuJobs = execute_gpu_jobs_rd;
	p_interface->fpStageConstantBufferDataFunc = stage_constant_buffer_data_rd;
	p_interface->scratchBuffer = &scratch;
	p_interface->scratchBufferSize = sizeof(scratch);

	p_interface->device = &device;
}
