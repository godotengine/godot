/**************************************************************************/
/*  fsr3_upscaler.cpp                                                     */
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

#include "fsr3_upscaler.h"

#include "servers/rendering/renderer_rd/storage_rd/material_storage.h"
#include "servers/rendering/renderer_rd/uniform_set_cache_rd.h"

using namespace RendererRD;

static void fsr3_recv_message(FfxMsgType type, const wchar_t *message) {
#ifdef DEV_ENABLED
	switch (type) {
		case FFX_MESSAGE_TYPE_ERROR:
			ERR_PRINT(message);
			break;
		case FFX_MESSAGE_TYPE_WARNING:
			WARN_PRINT(message);
			break;
	}
#endif
}

FSR3UpscalerContext::~FSR3UpscalerContext() {
	if (generated_reactive_mask.is_valid()) {
		RD::get_singleton()->free_rid(generated_reactive_mask);
	}

	fsr_desc.backendInterface.fpDestroyResource(&fsr_desc.backendInterface, reconstructed_prev_nearest_depth, -1);
	fsr_desc.backendInterface.fpDestroyResource(&fsr_desc.backendInterface, dilated_depth, -1);
	fsr_desc.backendInterface.fpDestroyResource(&fsr_desc.backendInterface, dilated_motion_vectors, -1);

	ffxFsr3UpscalerContextDestroy(&fsr_context);
}

FSR3UpscalerEffect::FSR3UpscalerEffect() {
	FfxDeviceCapabilities capabilities = FFXCommon::get_device_capabilities();

	String general_defines =
			"\n#define FFX_GPU\n"
			"\n#define FFX_GLSL 1\n"
			"\n#define FFX_FSR3UPSCALER_OPTION_LOW_RESOLUTION_MOTION_VECTORS 1\n"
			"\n#define FFX_FSR3UPSCALER_OPTION_HDR_COLOR_INPUT 1\n"
			"\n#define FFX_FSR3UPSCALER_OPTION_INVERTED_DEPTH 1\n"
			"\n#define FFX_FSR3UPSCALER_OPTION_GODOT_REACTIVE_MASK_CLAMP 1\n"
			"\n#define FFX_FSR3UPSCALER_OPTION_GODOT_DERIVE_INVALID_MOTION_VECTORS 1\n";

	Vector<String> modes_single;
	modes_single.push_back("");

	Vector<String> modes_with_fp16;
	modes_with_fp16.push_back("");
	modes_with_fp16.push_back("\n#define FFX_HALF 1\n");

	// Since Godot currently lacks a shader reflection mechanism to persist the name of the bindings in the shader cache and
	// there's also no mechanism to compile the shaders offline, the bindings are created manually by looking at the GLSL
	// files included in FSR2 and mapping the macro bindings (#define FSR2_BIND_*) to their respective implementation names.
	//
	// It is not guaranteed these will remain consistent at all between versions of FSR2, so it'll be necessary to keep these
	// bindings up to date whenever the library is updated. In such cases, it is very likely the validation layer will throw an
	// error if the bindings do not match.

	{
		FFXCommon::Pass &pass = device.passes[FFX_FSR3UPSCALER_PASS_PREPARE_INPUTS];
		pass.shader = &shaders.prepare_inputs;
		pass.shader->initialize(modes_with_fp16, general_defines);
		pass.shader_version = pass.shader->version_create();
		pass.shader_variant = capabilities.fp16Supported ? 1 : 0;

		pass.sampled_texture_bindings = {
			FfxResourceBinding{ 0, 0, 0, L"r_input_motion_vectors" },
			FfxResourceBinding{ 1, 0, 0, L"r_input_depth" },
			FfxResourceBinding{ 2, 0, 0, L"r_input_color_jittered" },
		};

		pass.storage_texture_bindings = {
			FfxResourceBinding{ 3, 0, 0, L"rw_dilated_motion_vectors" },
			FfxResourceBinding{ 4, 0, 0, L"rw_dilated_depth" },
			FfxResourceBinding{ 5, 0, 0, L"rw_reconstructed_previous_nearest_depth" },
			FfxResourceBinding{ 6, 0, 0, L"rw_farthest_depth" },
			FfxResourceBinding{ 7, 0, 0, L"rw_current_luma" },
		};

		pass.uniform_bindings = {
			FfxResourceBinding{ 8, 0, 0, L"cbFSR3Upscaler" }
		};
	}

	{
		FFXCommon::Pass &pass = device.passes[FFX_FSR3UPSCALER_PASS_LUMA_PYRAMID];
		pass.shader = &shaders.luma_pyramid;
		pass.shader->initialize(modes_with_fp16, general_defines);
		pass.shader_version = pass.shader->version_create();
		pass.shader_variant = capabilities.fp16Supported ? 1 : 0;

		pass.sampled_texture_bindings = {
			FfxResourceBinding{ 0, 0, 0, L"r_current_luma" },
			FfxResourceBinding{ 1, 0, 0, L"r_farthest_depth" },
		};

		pass.storage_texture_bindings = {
			FfxResourceBinding{ 2, 0, 0, L"rw_spd_global_atomic" },
			FfxResourceBinding{ 3, 0, 0, L"rw_frame_info" },
			FfxResourceBinding{ 4, 0, 0, L"rw_spd_mip0" },
			FfxResourceBinding{ 5, 0, 0, L"rw_spd_mip1" },
			FfxResourceBinding{ 6, 0, 0, L"rw_spd_mip2" },
			FfxResourceBinding{ 7, 0, 0, L"rw_spd_mip3" },
			FfxResourceBinding{ 8, 0, 0, L"rw_spd_mip4" },
			FfxResourceBinding{ 9, 0, 0, L"rw_spd_mip5" },
			FfxResourceBinding{ 10, 0, 0, L"rw_farthest_depth_mip1" },
		};

		pass.uniform_bindings = {
			FfxResourceBinding{ 11, 0, 0, L"cbFSR3Upscaler" },
			FfxResourceBinding{ 12, 0, 0, L"cbSPD" },
		};
	}

	{
		FFXCommon::Pass &pass = device.passes[FFX_FSR3UPSCALER_PASS_SHADING_CHANGE_PYRAMID];
		pass.shader = &shaders.shading_change_pyramid;
		pass.shader->initialize(modes_with_fp16, general_defines);
		pass.shader_version = pass.shader->version_create();
		pass.shader_variant = capabilities.fp16Supported ? 1 : 0;

		pass.sampled_texture_bindings = {
			FfxResourceBinding{ 0, 0, 0, L"r_current_luma" },
			FfxResourceBinding{ 1, 0, 0, L"r_previous_luma" },
			FfxResourceBinding{ 2, 0, 0, L"r_dilated_motion_vectors" },
			FfxResourceBinding{ 3, 0, 0, L"r_input_exposure" },
		};

		pass.storage_texture_bindings = {
			FfxResourceBinding{ 4, 0, 0, L"rw_spd_global_atomic" },
			FfxResourceBinding{ 5, 0, 0, L"rw_spd_mip0" },
			FfxResourceBinding{ 6, 0, 0, L"rw_spd_mip1" },
			FfxResourceBinding{ 7, 0, 0, L"rw_spd_mip2" },
			FfxResourceBinding{ 8, 0, 0, L"rw_spd_mip3" },
			FfxResourceBinding{ 9, 0, 0, L"rw_spd_mip4" },
			FfxResourceBinding{ 10, 0, 0, L"rw_spd_mip5" },
		};

		pass.uniform_bindings = {
			FfxResourceBinding{ 11, 0, 0, L"cbFSR3Upscaler" },
			FfxResourceBinding{ 12, 0, 0, L"cbSPD" },
		};
	}

	{
		FFXCommon::Pass &pass = device.passes[FFX_FSR3UPSCALER_PASS_SHADING_CHANGE];
		pass.shader = &shaders.shading_change;
		pass.shader->initialize(modes_with_fp16, general_defines);
		pass.shader_version = pass.shader->version_create();
		pass.shader_variant = capabilities.fp16Supported ? 1 : 0;

		pass.sampled_texture_bindings = {
			FfxResourceBinding{ 0, 0, 0, L"r_spd_mips" },
		};

		pass.storage_texture_bindings = {
			FfxResourceBinding{ 1, 0, 0, L"rw_shading_change" },
		};

		pass.uniform_bindings = {
			FfxResourceBinding{ 2, 0, 0, L"cbFSR3Upscaler" },
		};
	}

	{
		FFXCommon::Pass &pass = device.passes[FFX_FSR3UPSCALER_PASS_PREPARE_REACTIVITY];
		pass.shader = &shaders.prepare_reactivity;
		pass.shader->initialize(modes_with_fp16, general_defines);
		pass.shader_version = pass.shader->version_create();
		pass.shader_variant = capabilities.fp16Supported ? 1 : 0;

		pass.sampled_texture_bindings = {
			FfxResourceBinding{ 0, 0, 0, L"r_reconstructed_previous_nearest_depth" },
			FfxResourceBinding{ 1, 0, 0, L"r_dilated_motion_vectors" },
			FfxResourceBinding{ 2, 0, 0, L"r_dilated_depth" },
			FfxResourceBinding{ 3, 0, 0, L"r_reactive_mask" },
			FfxResourceBinding{ 4, 0, 0, L"r_transparency_and_composition_mask" },
			FfxResourceBinding{ 5, 0, 0, L"r_accumulation" },
			FfxResourceBinding{ 6, 0, 0, L"r_shading_change" },
			FfxResourceBinding{ 7, 0, 0, L"r_current_luma" },
			FfxResourceBinding{ 8, 0, 0, L"r_input_exposure" },
		};

		pass.storage_texture_bindings = {
			FfxResourceBinding{ 9, 0, 0, L"rw_dilated_reactive_masks" },
			FfxResourceBinding{ 10, 0, 0, L"rw_new_locks" },
			FfxResourceBinding{ 11, 0, 0, L"rw_accumulation" },
		};

		pass.uniform_bindings = {
			FfxResourceBinding{ 12, 0, 0, L"cbFSR3Upscaler" },
		};
	}

	{
		FFXCommon::Pass &pass = device.passes[FFX_FSR3UPSCALER_PASS_LUMA_INSTABILITY];
		pass.shader = &shaders.luma_instability;
		pass.shader->initialize(modes_with_fp16, general_defines);
		pass.shader_version = pass.shader->version_create();
		pass.shader_variant = capabilities.fp16Supported ? 1 : 0;

		pass.sampled_texture_bindings = {
			FfxResourceBinding{ 0, 0, 0, L"r_input_exposure" },
			FfxResourceBinding{ 1, 0, 0, L"r_dilated_reactive_masks" },
			FfxResourceBinding{ 2, 0, 0, L"r_dilated_motion_vectors" },
			FfxResourceBinding{ 3, 0, 0, L"r_frame_info" },
			FfxResourceBinding{ 4, 0, 0, L"r_luma_history" },
			FfxResourceBinding{ 5, 0, 0, L"r_farthest_depth_mip1" },
			FfxResourceBinding{ 6, 0, 0, L"r_current_luma" },
		};

		pass.storage_texture_bindings = {
			FfxResourceBinding{ 7, 0, 0, L"rw_luma_history" },
			FfxResourceBinding{ 8, 0, 0, L"rw_luma_instability" },
		};

		pass.uniform_bindings = {
			FfxResourceBinding{ 9, 0, 0, L"cbFSR3Upscaler" },
		};
	}

	{
		Vector<String> accumulate_modes_with_fp16;
		accumulate_modes_with_fp16.push_back("\n");
		accumulate_modes_with_fp16.push_back("\n#define FFX_FSR2_OPTION_APPLY_SHARPENING 1\n");
		accumulate_modes_with_fp16.push_back("\n#define FFX_HALF 1\n");
		accumulate_modes_with_fp16.push_back("\n#define FFX_HALF 1\n#define FFX_FSR2_OPTION_APPLY_SHARPENING 1\n");

		// Workaround: Disable FP16 path for the accumulate pass on NVIDIA due to reduced occupancy and high VRAM throughput.
		const bool fp16_path_supported = RD::get_singleton()->get_device_vendor_name() != "NVIDIA";
		FFXCommon::Pass &pass = device.passes[FFX_FSR3UPSCALER_PASS_ACCUMULATE];
		pass.shader = &shaders.accumulate;
		pass.shader->initialize(accumulate_modes_with_fp16, general_defines);
		pass.shader_version = pass.shader->version_create();
		pass.shader_variant = capabilities.fp16Supported && fp16_path_supported ? 2 : 0;

		pass.sampled_texture_bindings = {
			FfxResourceBinding{ 0, 0, 0, L"r_input_exposure" },
			FfxResourceBinding{ 1, 0, 0, L"r_dilated_reactive_masks" },
			FfxResourceBinding{ 2, 0, 0, L"r_input_motion_vectors" },
			FfxResourceBinding{ 3, 0, 0, L"r_internal_upscaled_color" },
			FfxResourceBinding{ 4, 0, 0, L"r_lanczos_lut" },
			FfxResourceBinding{ 5, 0, 0, L"r_farthest_depth_mip1" },
			FfxResourceBinding{ 6, 0, 0, L"r_current_luma" },
			FfxResourceBinding{ 7, 0, 0, L"r_luma_instability" },
			FfxResourceBinding{ 8, 0, 0, L"r_input_color_jittered" },
		};

		pass.storage_texture_bindings = {
			FfxResourceBinding{ 9, 0, 0, L"rw_internal_upscaled_color" },
			FfxResourceBinding{ 10, 0, 0, L"rw_upscaled_output" },
			FfxResourceBinding{ 11, 0, 0, L"rw_new_locks" },
		};

		pass.uniform_bindings = {
			FfxResourceBinding{ 12, 0, 0, L"cbFSR3Upscaler" },
		};

		// Sharpen pass is a clone of the accumulate pass with the sharpening variant.
		FFXCommon::Pass &sharpen_pass = device.passes[FFX_FSR3UPSCALER_PASS_ACCUMULATE_SHARPEN];
		sharpen_pass = pass;
		sharpen_pass.shader_variant = pass.shader_variant + 1;
	}

	{
		FFXCommon::Pass &pass = device.passes[FFX_FSR3UPSCALER_PASS_RCAS];
		pass.shader = &shaders.rcas;
		pass.shader->initialize(modes_single, general_defines);
		pass.shader_version = pass.shader->version_create();

		pass.sampled_texture_bindings = {
			FfxResourceBinding{ 0, 0, 0, L"r_input_exposure" },
			FfxResourceBinding{ 1, 0, 0, L"r_rcas_input" },
		};

		pass.storage_texture_bindings = {
			FfxResourceBinding{ 2, 0, 0, L"rw_upscaled_output" },
		};

		pass.uniform_bindings = {
			FfxResourceBinding{ 3, 0, 0, L"cbFSR3Upscaler" },
			FfxResourceBinding{ 4, 0, 0, L"cbRCAS" },
		};
	}

	{
		FFXCommon::Pass &pass = device.passes[FFX_FSR3UPSCALER_PASS_DEBUG_VIEW];
		pass.shader = &shaders.debug_view;
		pass.shader->initialize(modes_single, general_defines);
		pass.shader_version = pass.shader->version_create();

		pass.sampled_texture_bindings = {
			FfxResourceBinding{ 0, 0, 0, L"r_dilated_reactive_masks" },
			FfxResourceBinding{ 1, 0, 0, L"r_dilated_motion_vectors" },
			FfxResourceBinding{ 2, 0, 0, L"r_dilated_depth" },
			FfxResourceBinding{ 3, 0, 0, L"r_internal_upscaled_color" },
			FfxResourceBinding{ 4, 0, 0, L"r_input_exposure" },
		};

		pass.storage_texture_bindings = {
			FfxResourceBinding{ 5, 0, 0, L"rw_upscaled_output" },
		};

		pass.uniform_bindings = {
			FfxResourceBinding{ 6, 0, 0, L"cbFSR3Upscaler" },
		};
	}

	{
		FFXCommon::Pass &pass = device.passes[FFX_FSR3UPSCALER_PASS_GENERATE_REACTIVE];
		pass.shader = &shaders.autogen_reactive;
		pass.shader->initialize(modes_with_fp16, general_defines);
		pass.shader_version = pass.shader->version_create();
		pass.shader_variant = capabilities.fp16Supported ? 1 : 0;

		pass.sampled_texture_bindings = {
			FfxResourceBinding{ 0, 0, 0, L"r_input_opaque_only" },
			FfxResourceBinding{ 1, 0, 0, L"r_input_color_jittered" },
		};

		pass.storage_texture_bindings = {
			FfxResourceBinding{ 2, 0, 0, L"rw_output_autoreactive" },
			// Though this binding is present in the GLSL source, but the FSR3 CXX side doesn't register it at all.
			// So we must comment it out to avoid runtime errors.
			// FfxResourceBinding{ 3, 0, 0, L"rw_output_autocomposition" },
		};

		pass.uniform_bindings = {
			FfxResourceBinding{ 4, 0, 0, L"cbFSR3Upscaler" },
			FfxResourceBinding{ 5, 0, 0, L"cbGenerateReactive" },
		};
	}

	device.linear_clamp_sampler = FFXCommon::create_clamp_sampler(RD::SAMPLER_FILTER_LINEAR);
	device.point_clamp_sampler = FFXCommon::create_clamp_sampler(RD::SAMPLER_FILTER_NEAREST);
}

FSR3UpscalerEffect::~FSR3UpscalerEffect() {
	RD::get_singleton()->free_rid(device.point_clamp_sampler);
	RD::get_singleton()->free_rid(device.linear_clamp_sampler);

	for (uint32_t i = 0; i < FFX_FSR3UPSCALER_PASS_COUNT; i++) {
		if (i == FFX_FSR3UPSCALER_PASS_TCR_AUTOGENERATE) {
			// These passes are not even created, so no need to be freed
			continue;
		}

		device.passes[i].shader->version_free(device.passes[i].shader_version);
	}
}

FSR3UpscalerContext *FSR3UpscalerEffect::create_context(Size2i p_internal_size, Size2i p_target_size, bool p_autogen_reactive) {
	FSR3UpscalerContext *context = memnew(RendererRD::FSR3UpscalerContext);
	context->fsr_desc.flags = FFX_FSR3UPSCALER_ENABLE_HIGH_DYNAMIC_RANGE | FFX_FSR3UPSCALER_ENABLE_DEPTH_INVERTED;
#ifdef DEV_ENABLED
	context->fsr_desc.flags |= FFX_FSR3UPSCALER_ENABLE_DEBUG_CHECKING;
#endif
	context->fsr_desc.maxRenderSize.width = p_internal_size.x;
	context->fsr_desc.maxRenderSize.height = p_internal_size.y;
	context->fsr_desc.maxUpscaleSize.width = p_target_size.x;
	context->fsr_desc.maxUpscaleSize.height = p_target_size.y;
	context->fsr_desc.fpMessage = fsr3_recv_message;

	FFXCommon::create_ffx_interface(&context->fsr_desc.backendInterface, &context->scratch, &device);
	FfxErrorCode result = ffxFsr3UpscalerContextCreate(&context->fsr_context, &context->fsr_desc);
	if (result == FFX_OK) {
		FfxFsr3UpscalerSharedResourceDescriptions shared_resource_descriptions;
		ffxFsr3UpscalerGetSharedResourceDescriptions(&context->fsr_context, &shared_resource_descriptions);

		// Create shared resources
		result = context->fsr_desc.backendInterface.fpCreateResource(&context->fsr_desc.backendInterface, &shared_resource_descriptions.reconstructedPrevNearestDepth, -1, &context->reconstructed_prev_nearest_depth);
		if (result != FFX_OK) {
			ERR_PRINT("Failed to create FSR3 Upscaler shared resource: reconstructed_prev_nearest_depth.");
			memdelete(context);
			return nullptr;
		}

		result = context->fsr_desc.backendInterface.fpCreateResource(&context->fsr_desc.backendInterface, &shared_resource_descriptions.dilatedDepth, -1, &context->dilated_depth);
		if (result != FFX_OK) {
			ERR_PRINT("Failed to create FSR3 Upscaler shared resource: reconstructed_prev_nearest_depth.");
			memdelete(context);
			return nullptr;
		}

		result = context->fsr_desc.backendInterface.fpCreateResource(&context->fsr_desc.backendInterface, &shared_resource_descriptions.dilatedMotionVectors, -1, &context->dilated_motion_vectors);
		if (result != FFX_OK) {
			ERR_PRINT("Failed to create FSR3 Upscaler shared resource: reconstructed_prev_nearest_depth.");
			memdelete(context);
			return nullptr;
		}

		if (p_autogen_reactive) {
			RD::TextureFormat texture_format;
			texture_format.texture_type = RD::TEXTURE_TYPE_2D;
			texture_format.format = RD::DATA_FORMAT_R8_UNORM;
			texture_format.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_CAN_COPY_FROM_BIT | RD::TEXTURE_USAGE_CAN_COPY_TO_BIT;
			texture_format.width = p_internal_size.width;
			texture_format.height = p_internal_size.height;
			texture_format.depth = 1;
			texture_format.mipmaps = 1;

			context->generated_reactive_mask = RD::get_singleton()->texture_create(texture_format, RD::TextureView());
			ERR_FAIL_COND_V_MSG(context->generated_reactive_mask.is_null(), nullptr, "Failed to create FSR3 Upscaler generated reactive mask texture.");
			RD::get_singleton()->set_resource_name(context->generated_reactive_mask, L"FSR3UPSCALER_GeneratedReactiveMask");
		}

		return context;
	} else {
		memdelete(context);
		return nullptr;
	}
}

void FSR3UpscalerEffect::upscale(const Parameters &p_params) {
	RID color = p_params.color;
	RID depth = p_params.depth;
	RID velocity = p_params.velocity;
	RID reactive = p_params.reactive;
	RID exposure = p_params.exposure;
	RID output = p_params.output;

	FFXCommon::Scratch &scratch = p_params.context->scratch;

	RID opaque_only = p_params.opaque_only;
	RID out_reactive = p_params.context->generated_reactive_mask;
	bool autogen_masks = opaque_only.is_valid() && p_params.context->generated_reactive_mask.is_valid();

	// Optional pass of auto-generating reactive masks from opaque-only color.
	// This may reduce flickering in scenarios where there are massive transparent objects.
	if (autogen_masks) {
		FfxFsr3UpscalerGenerateReactiveDescription generate_desc = {};

		generate_desc.commandList = nullptr;
		generate_desc.colorPreUpscale = FFXCommon::get_resource_rd(&color, L"color");
		generate_desc.colorOpaqueOnly = FFXCommon::get_resource_rd(&opaque_only, L"opaque_only");
		generate_desc.outReactive = FFXCommon::get_resource_rd(&out_reactive, L"generated_reactive_mask");
		generate_desc.binaryValue = 0.9f;
		generate_desc.renderSize.width = p_params.internal_size.width;
		generate_desc.renderSize.height = p_params.internal_size.height;
		generate_desc.cutoffThreshold = 0.2f;
		generate_desc.scale = 1.f;

		FfxErrorCode err = ffxFsr3UpscalerContextGenerateReactiveMask(&p_params.context->fsr_context, &generate_desc);
		if (err != FFX_OK) {
			WARN_PRINT_ONCE("FSR3: Generate reactive mask enabled, but corresponding pass failed.");
			autogen_masks = false;
		}
	}

	FfxFsr3UpscalerDispatchDescription dispatch_desc = {};
	RID reconstructed_prev_nearest_depth = scratch.resources.rids[p_params.context->reconstructed_prev_nearest_depth.internalIndex];
	RID dilated_depth = scratch.resources.rids[p_params.context->dilated_depth.internalIndex];
	RID dilated_motion_vectors = scratch.resources.rids[p_params.context->dilated_motion_vectors.internalIndex];

	dispatch_desc.commandList = nullptr;
	dispatch_desc.color = FFXCommon::get_resource_rd(&color, L"color");
	dispatch_desc.depth = FFXCommon::get_resource_rd(&depth, L"depth");
	dispatch_desc.reconstructedPrevNearestDepth = FFXCommon::get_resource_rd(&reconstructed_prev_nearest_depth, L"reconstructed_prev_nearest_depth");
	dispatch_desc.dilatedDepth = FFXCommon::get_resource_rd(&dilated_depth, L"dilated_depth");
	dispatch_desc.dilatedMotionVectors = FFXCommon::get_resource_rd(&dilated_motion_vectors, L"dilated_motion_vectors");
	dispatch_desc.motionVectors = FFXCommon::get_resource_rd(&velocity, L"velocity");
	dispatch_desc.reactive = FFXCommon::get_resource_rd(autogen_masks ? &out_reactive : &reactive, L"reactive");
	dispatch_desc.exposure = FFXCommon::get_resource_rd(&exposure, L"exposure");
	dispatch_desc.transparencyAndComposition = {};
	dispatch_desc.output = FFXCommon::get_resource_rd(&output, L"output");
	dispatch_desc.jitterOffset.x = p_params.jitter.x;
	dispatch_desc.jitterOffset.y = p_params.jitter.y;
	dispatch_desc.motionVectorScale.x = float(p_params.internal_size.width);
	dispatch_desc.motionVectorScale.y = float(p_params.internal_size.height);
	dispatch_desc.reset = p_params.reset_accumulation;
	dispatch_desc.renderSize.width = p_params.internal_size.width;
	dispatch_desc.renderSize.height = p_params.internal_size.height;
	dispatch_desc.upscaleSize.width = p_params.target_size.width;
	dispatch_desc.upscaleSize.height = p_params.target_size.height;
	dispatch_desc.enableSharpening = (p_params.sharpness > 1e-6f);
	dispatch_desc.sharpness = p_params.sharpness;
	dispatch_desc.frameTimeDelta = p_params.delta_time;
	dispatch_desc.preExposure = 1.0f;
	dispatch_desc.cameraNear = p_params.z_near;
	dispatch_desc.cameraFar = p_params.z_far;
	dispatch_desc.cameraFovAngleVertical = p_params.fovy;
	dispatch_desc.viewSpaceToMetersFactor = 1.0f;

	MaterialStorage::store_camera(p_params.reprojection, dispatch_desc.reprojectionMatrix);

	FfxErrorCode result = ffxFsr3UpscalerContextDispatch(&p_params.context->fsr_context, &dispatch_desc);
	ERR_FAIL_COND(result != FFX_OK);
}
