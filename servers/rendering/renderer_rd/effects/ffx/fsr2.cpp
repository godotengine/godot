/**************************************************************************/
/*  fsr2.cpp                                                              */
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

#include "fsr2.h"

#include "servers/rendering/renderer_rd/storage_rd/material_storage.h"
#include "servers/rendering/renderer_rd/uniform_set_cache_rd.h"

using namespace RendererRD;

FSR2Context::~FSR2Context() {
	ffxFsr2ContextDestroy(&fsr_context);
}

FSR2Effect::FSR2Effect() {
	FfxDeviceCapabilities capabilities = FFXCommon::get_device_capabilities();

	String general_defines =
			"\n#define FFX_GPU\n"
			"\n#define FFX_GLSL 1\n"
			"\n#define FFX_FSR2_OPTION_LOW_RESOLUTION_MOTION_VECTORS 1\n"
			"\n#define FFX_FSR2_OPTION_HDR_COLOR_INPUT 1\n"
			"\n#define FFX_FSR2_OPTION_INVERTED_DEPTH 1\n"
			"\n#define FFX_FSR2_OPTION_GODOT_REACTIVE_MASK_CLAMP 1\n"
			"\n#define FFX_FSR2_OPTION_GODOT_DERIVE_INVALID_MOTION_VECTORS 1\n";

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
		FFXCommon::Pass &pass = device.passes[FFX_FSR2_PASS_DEPTH_CLIP];
		pass.shader = &shaders.depth_clip;
		pass.shader->initialize(modes_with_fp16, general_defines);
		pass.shader_version = pass.shader->version_create();
		pass.shader_variant = capabilities.fp16Supported ? 1 : 0;

		pass.sampled_texture_bindings = {
			FfxResourceBinding{ 0, 0, 0, L"r_reconstructed_previous_nearest_depth" },
			FfxResourceBinding{ 1, 0, 0, L"r_dilated_motion_vectors" },
			FfxResourceBinding{ 2, 0, 0, L"r_dilatedDepth" },
			FfxResourceBinding{ 3, 0, 0, L"r_reactive_mask" },
			FfxResourceBinding{ 4, 0, 0, L"r_transparency_and_composition_mask" },
			// Godot render graph forces one resource to serve only one usage so we have to remove this binding
			// FfxResourceBinding{ 5, 0, 0, L"r_prepared_input_color" },
			FfxResourceBinding{ 6, 0, 0, L"r_previous_dilated_motion_vectors" },
			FfxResourceBinding{ 7, 0, 0, L"r_input_motion_vectors" },
			FfxResourceBinding{ 8, 0, 0, L"r_input_color_jittered" },
			FfxResourceBinding{ 9, 0, 0, L"r_input_depth" },
			FfxResourceBinding{ 10, 0, 0, L"r_input_exposure" }
		};

		pass.storage_texture_bindings = {
			// FSR2_BIND_UAV_DEPTH_CLIP (11) does not point to anything.
			FfxResourceBinding{ 2012, 0, 0, L"rw_dilated_reactive_masks" },
			FfxResourceBinding{ 2013, 0, 0, L"rw_prepared_input_color" }
		};

		pass.uniform_bindings = {
			FfxResourceBinding{ 3000, 0, 0, L"cbFSR2" }
		};
	}

	{
		FFXCommon::Pass &pass = device.passes[FFX_FSR2_PASS_RECONSTRUCT_PREVIOUS_DEPTH];
		pass.shader = &shaders.reconstruct_previous_depth;
		pass.shader->initialize(modes_with_fp16, general_defines);
		pass.shader_version = pass.shader->version_create();
		pass.shader_variant = capabilities.fp16Supported ? 1 : 0;

		pass.sampled_texture_bindings = {
			FfxResourceBinding{ 0, 0, 0, L"r_input_motion_vectors" },
			FfxResourceBinding{ 1, 0, 0, L"r_input_depth" },
			FfxResourceBinding{ 2, 0, 0, L"r_input_color_jittered" },
			FfxResourceBinding{ 3, 0, 0, L"r_input_exposure" },
			FfxResourceBinding{ 4, 0, 0, L"r_luma_history" }
		};

		pass.storage_texture_bindings = {
			FfxResourceBinding{ 2005, 0, 0, L"rw_reconstructed_previous_nearest_depth" },
			FfxResourceBinding{ 2006, 0, 0, L"rw_dilated_motion_vectors" },
			FfxResourceBinding{ 2007, 0, 0, L"rw_dilatedDepth" },
			FfxResourceBinding{ 2008, 0, 0, L"rw_prepared_input_color" },
			FfxResourceBinding{ 2009, 0, 0, L"rw_luma_history" },
			// FSR2_BIND_UAV_LUMA_INSTABILITY (10) does not point to anything.
			FfxResourceBinding{ 2011, 0, 0, L"rw_lock_input_luma" }
		};

		pass.uniform_bindings = {
			FfxResourceBinding{ 3000, 0, 0, L"cbFSR2" }
		};
	}

	{
		FFXCommon::Pass &pass = device.passes[FFX_FSR2_PASS_LOCK];
		pass.shader = &shaders.lock;
		pass.shader->initialize(modes_with_fp16, general_defines);
		pass.shader_version = pass.shader->version_create();
		pass.shader_variant = capabilities.fp16Supported ? 1 : 0;

		pass.sampled_texture_bindings = {
			FfxResourceBinding{ 0, 0, 0, L"r_lock_input_luma" }
		};

		pass.storage_texture_bindings = {
			FfxResourceBinding{ 2001, 0, 0, L"rw_new_locks" },
			FfxResourceBinding{ 2002, 0, 0, L"rw_reconstructed_previous_nearest_depth" }
		};

		pass.uniform_bindings = {
			FfxResourceBinding{ 3000, 0, 0, L"cbFSR2" }
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
		FFXCommon::Pass &pass = device.passes[FFX_FSR2_PASS_ACCUMULATE];
		pass.shader = &shaders.accumulate;
		pass.shader->initialize(accumulate_modes_with_fp16, general_defines);
		pass.shader_version = pass.shader->version_create();
		pass.shader_variant = capabilities.fp16Supported && fp16_path_supported ? 2 : 0;

		pass.sampled_texture_bindings = {
			FfxResourceBinding{ 0, 0, 0, L"r_input_exposure" },
			FfxResourceBinding{ 1, 0, 0, L"r_dilated_reactive_masks" },
			FfxResourceBinding{ 2, 0, 0, L"r_input_motion_vectors" },
			FfxResourceBinding{ 3, 0, 0, L"r_internal_upscaled_color" },
			FfxResourceBinding{ 4, 0, 0, L"r_lock_status" },
			FfxResourceBinding{ 5, 0, 0, L"r_input_depth" },
			FfxResourceBinding{ 6, 0, 0, L"r_prepared_input_color" },
			// FSR2_BIND_SRV_LUMA_INSTABILITY(7) does not point to anything.
			FfxResourceBinding{ 8, 0, 0, L"r_lanczos_lut" },
			FfxResourceBinding{ 9, 0, 0, L"r_upsample_maximum_bias_lut" },
			FfxResourceBinding{ 10, 0, 0, L"r_imgMips" },
			FfxResourceBinding{ 11, 0, 0, L"r_auto_exposure" },
			FfxResourceBinding{ 12, 0, 0, L"r_luma_history" }
		};

		pass.storage_texture_bindings = {
			FfxResourceBinding{ 2013, 0, 0, L"rw_internal_upscaled_color" },
			FfxResourceBinding{ 2014, 0, 0, L"rw_lock_status" },
			FfxResourceBinding{ 2015, 0, 0, L"rw_upscaled_output" },
			FfxResourceBinding{ 2016, 0, 0, L"rw_new_locks" },
			FfxResourceBinding{ 2017, 0, 0, L"rw_luma_history" }
		};

		pass.uniform_bindings = {
			FfxResourceBinding{ 3000, 0, 0, L"cbFSR2" }
		};

		// Sharpen pass is a clone of the accumulate pass with the sharpening variant.
		FFXCommon::Pass &sharpen_pass = device.passes[FFX_FSR2_PASS_ACCUMULATE_SHARPEN];
		sharpen_pass = pass;
		sharpen_pass.shader_variant = pass.shader_variant + 1;
	}

	{
		FFXCommon::Pass &pass = device.passes[FFX_FSR2_PASS_RCAS];
		pass.shader = &shaders.rcas;
		pass.shader->initialize(modes_single, general_defines);
		pass.shader_version = pass.shader->version_create();

		pass.sampled_texture_bindings = {
			FfxResourceBinding{ 0, 0, 0, L"r_input_exposure" },
			FfxResourceBinding{ 1, 0, 0, L"r_rcas_input" }
		};

		pass.storage_texture_bindings = {
			FfxResourceBinding{ 2002, 0, 0, L"rw_upscaled_output" }
		};

		pass.uniform_bindings = {
			FfxResourceBinding{ 3000, 0, 0, L"cbFSR2" },
			FfxResourceBinding{ 3001, 0, 0, L"cbRCAS" }
		};
	}

	{
		FFXCommon::Pass &pass = device.passes[FFX_FSR2_PASS_COMPUTE_LUMINANCE_PYRAMID];
		pass.shader = &shaders.compute_luminance_pyramid;
		pass.shader->initialize(modes_single, general_defines);
		pass.shader_version = pass.shader->version_create();

		pass.sampled_texture_bindings = {
			FfxResourceBinding{ 0, 0, 0, L"r_input_color_jittered" }
		};

		pass.storage_texture_bindings = {
			FfxResourceBinding{ 2001, 0, 0, L"rw_spd_global_atomic" },
			FfxResourceBinding{ 2002, 0, 0, L"rw_img_mip_shading_change" },
			FfxResourceBinding{ 2003, 0, 0, L"rw_img_mip_5" },
			FfxResourceBinding{ 2004, 0, 0, L"rw_auto_exposure" }
		};

		pass.uniform_bindings = {
			FfxResourceBinding{ 3000, 0, 0, L"cbFSR2" },
			FfxResourceBinding{ 3001, 0, 0, L"cbSPD" }
		};
	}

	{
		FFXCommon::Pass &pass = device.passes[FFX_FSR2_PASS_GENERATE_REACTIVE];
		pass.shader = &shaders.autogen_reactive;
		pass.shader->initialize(modes_with_fp16, general_defines);
		pass.shader_version = pass.shader->version_create();
		pass.shader_variant = capabilities.fp16Supported ? 1 : 0;

		pass.sampled_texture_bindings = {
			FfxResourceBinding{ 0, 0, 0, L"r_input_opaque_only" },
			FfxResourceBinding{ 1, 0, 0, L"r_input_color_jittered" },
		};

		pass.storage_texture_bindings = {
			FfxResourceBinding{ 2002, 0, 0, L"rw_output_autoreactive" }
		};

		pass.uniform_bindings = {
			FfxResourceBinding{ 3000, 0, 0, L"cbGenerateReactive" },
			FfxResourceBinding{ 3001, 0, 0, L"cbFSR2" }
		};
	}

	{
		FFXCommon::Pass &pass = device.passes[FFX_FSR2_PASS_TCR_AUTOGENERATE];
		pass.shader = &shaders.tcr_autogen;
		pass.shader->initialize(modes_with_fp16, general_defines);
		pass.shader_version = pass.shader->version_create();
		pass.shader_variant = capabilities.fp16Supported ? 1 : 0;

		pass.sampled_texture_bindings = {
			FfxResourceBinding{ 0, 0, 0, L"r_input_opaque_only" },
			FfxResourceBinding{ 1, 0, 0, L"r_input_color_jittered" },
			FfxResourceBinding{ 2, 0, 0, L"r_input_motion_vectors" },
			FfxResourceBinding{ 3, 0, 0, L"r_input_prev_color_pre_alpha" },
			FfxResourceBinding{ 4, 0, 0, L"r_input_prev_color_post_alpha" },
			FfxResourceBinding{ 5, 0, 0, L"r_reactive_mask" },
			FfxResourceBinding{ 6, 0, 0, L"r_transparency_and_composition_mask" },
			FfxResourceBinding{ 13, 0, 0, L"r_input_depth" },
		};

		pass.storage_texture_bindings = {
			FfxResourceBinding{ 2007, 0, 0, L"rw_output_autoreactive" },
			FfxResourceBinding{ 2008, 0, 0, L"rw_output_autocomposition" },
			FfxResourceBinding{ 2009, 0, 0, L"rw_output_prev_color_pre_alpha" },
			FfxResourceBinding{ 2010, 0, 0, L"rw_output_prev_color_post_alpha" }
		};

		pass.uniform_bindings = {
			FfxResourceBinding{ 3000, 0, 0, L"cbFSR2" },
			FfxResourceBinding{ 3001, 0, 0, L"cbGenerateReactive" }
		};
	}

	device.linear_clamp_sampler = FFXCommon::create_clamp_sampler(RD::SAMPLER_FILTER_LINEAR);
	device.point_clamp_sampler = FFXCommon::create_clamp_sampler(RD::SAMPLER_FILTER_NEAREST);
}

FSR2Effect::~FSR2Effect() {
	RD::get_singleton()->free_rid(device.point_clamp_sampler);
	RD::get_singleton()->free_rid(device.linear_clamp_sampler);

	for (uint32_t i = 0; i < FFX_FSR2_PASS_COUNT; i++) {
		device.passes[i].shader->version_free(device.passes[i].shader_version);
	}
}

FSR2Context *FSR2Effect::create_context(Size2i p_internal_size, Size2i p_target_size) {
	FSR2Context *context = memnew(RendererRD::FSR2Context);
	context->fsr_desc.flags = FFX_FSR2_ENABLE_HIGH_DYNAMIC_RANGE | FFX_FSR2_ENABLE_DEPTH_INVERTED;
	context->fsr_desc.maxRenderSize.width = p_internal_size.x;
	context->fsr_desc.maxRenderSize.height = p_internal_size.y;
	context->fsr_desc.displaySize.width = p_target_size.x;
	context->fsr_desc.displaySize.height = p_target_size.y;

	FFXCommon::create_ffx_interface(&context->fsr_desc.backendInterface, &context->scratch, &device);
	FfxErrorCode result = ffxFsr2ContextCreate(&context->fsr_context, &context->fsr_desc);
	if (result == FFX_OK) {
		return context;
	} else {
		memdelete(context);
		return nullptr;
	}
}

void FSR2Effect::upscale(const Parameters &p_params) {
	// TODO: Transparency & Composition mask is not implemented.
	FfxFsr2DispatchDescription dispatch_desc = {};
	RID color = p_params.color;
	RID depth = p_params.depth;
	RID velocity = p_params.velocity;
	RID reactive = p_params.reactive;
	RID exposure = p_params.exposure;
	RID output = p_params.output;
	dispatch_desc.commandList = nullptr;
	dispatch_desc.color = FFXCommon::get_resource_rd(&color, L"color");
	dispatch_desc.depth = FFXCommon::get_resource_rd(&depth, L"depth");
	dispatch_desc.motionVectors = FFXCommon::get_resource_rd(&velocity, L"velocity");
	dispatch_desc.reactive = FFXCommon::get_resource_rd(&reactive, L"reactive");
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
	dispatch_desc.enableSharpening = (p_params.sharpness > 1e-6f);
	dispatch_desc.sharpness = p_params.sharpness;
	dispatch_desc.frameTimeDelta = p_params.delta_time;
	dispatch_desc.preExposure = 1.0f;
	dispatch_desc.cameraNear = p_params.z_near;
	dispatch_desc.cameraFar = p_params.z_far;
	dispatch_desc.cameraFovAngleVertical = p_params.fovy;
	dispatch_desc.viewSpaceToMetersFactor = 1.0f;
	// FSR2 does provide automatic reactive mask generation, but that requires an opaque only color target,
	// which isn't provided in the current Godot pipeline. So now we just disable it.
	// When Godot adds a deferred renderer, we can re-enable this.
	dispatch_desc.colorOpaqueOnly = {};
	dispatch_desc.enableAutoReactive = false;

	dispatch_desc.autoTcThreshold = 1.0f;
	dispatch_desc.autoTcScale = 1.0f;
	dispatch_desc.autoReactiveScale = 1.0f;
	dispatch_desc.autoReactiveMax = 1.0f;

	RendererRD::MaterialStorage::store_camera(p_params.reprojection, dispatch_desc.reprojectionMatrix);

	FfxErrorCode result = ffxFsr2ContextDispatch(&p_params.context->fsr_context, &dispatch_desc);
	ERR_FAIL_COND(result != FFX_OK);
}
