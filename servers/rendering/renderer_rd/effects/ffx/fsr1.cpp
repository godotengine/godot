/**************************************************************************/
/*  fsr1.cpp                                                              */
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

#include "fsr1.h"

#include "servers/rendering/renderer_rd/storage_rd/material_storage.h"
#include "servers/rendering/renderer_rd/storage_rd/render_scene_buffers_rd.h"

using namespace RendererRD;

FSR1Context::~FSR1Context() {
	ffxFsr1ContextDestroy(&fsr_context);
}

void FSR1Effect::ensure_context(Ref<RenderSceneBuffersRD> p_render_buffers) {
	p_render_buffers->ensure_fsr1(this);
}

FSR1Effect::FSR1Effect() {
	FfxDeviceCapabilities capabilities = FFXCommon::get_device_capabilities();

	String general_defines =
			"\n#define FFX_GPU\n"
			"\n#define FFX_GLSL 1\n";

	Vector<String> modes_with_fp16;
	modes_with_fp16.push_back("");
	modes_with_fp16.push_back("\n#define FFX_HALF 1\n");

	// Since Godot currently lacks a shader reflection mechanism to persist the name of the bindings in the shader cache and
	// there's also no mechanism to compile the shaders offline, the bindings are created manually by looking at the GLSL
	// files included in FSR1 and mapping the macro bindings (#define FSR1_BIND_*) to their respective implementation names.
	//
	// It is not guaranteed these will remain consistent at all between versions of FSR2, so it'll be necessary to keep these
	// bindings up to date whenever the library is updated. In such cases, it is very likely the validation layer will throw an
	// error if the bindings do not match.

	{
		Vector<String> easu_modes_with_fp16;
		easu_modes_with_fp16.push_back("\n");
		easu_modes_with_fp16.push_back("\n#define FFX_FSR1_OPTION_APPLY_RCAS 1\n");
		easu_modes_with_fp16.push_back("\n#define FFX_HALF 1\n");
		easu_modes_with_fp16.push_back("\n#define FFX_HALF 1\n#define FFX_FSR1_OPTION_APPLY_RCAS 1\n");

		FFXCommon::Pass &pass = device.passes[FFX_FSR1_PASS_EASU];
		pass.shader = &shaders.easu;
		pass.shader->initialize(easu_modes_with_fp16, general_defines);
		pass.shader_version = pass.shader->version_create();
		pass.shader_variant = capabilities.fp16Supported ? 2 : 0;

		pass.sampled_texture_bindings = {
			FfxResourceBinding{ 0, 0, 0, L"r_input_color" },
		};

		pass.storage_texture_bindings = {
			FfxResourceBinding{ 2000, 0, 0, L"rw_internal_upscaled_color" },
			FfxResourceBinding{ 2001, 0, 0, L"rw_upscaled_output" }
		};

		pass.uniform_bindings = {
			FfxResourceBinding{ 3000, 0, 0, L"cbFSR1" }
		};

		// EASU RCAS pass is a clone of the EASU pass with the RCAS variant.
		FFXCommon::Pass &easu_rcas_pass = device.passes[FFX_FSR1_PASS_EASU_RCAS];
		easu_rcas_pass = pass;
		easu_rcas_pass.shader_variant = pass.shader_variant + 1;
	}

	{
		FFXCommon::Pass &pass = device.passes[FFX_FSR1_PASS_RCAS];
		pass.shader = &shaders.rcas;
		pass.shader->initialize(modes_with_fp16, general_defines);
		pass.shader_version = pass.shader->version_create();
		pass.shader_variant = capabilities.fp16Supported ? 1 : 0;

		pass.sampled_texture_bindings = {
			FfxResourceBinding{ 0, 0, 0, L"r_internal_upscaled_color" },
		};

		pass.storage_texture_bindings = {
			FfxResourceBinding{ 2000, 0, 0, L"rw_upscaled_output" },
		};

		pass.uniform_bindings = {
			FfxResourceBinding{ 3000, 0, 0, L"cbFSR1" }
		};
	}

	device.linear_clamp_sampler = FFXCommon::create_clamp_sampler(RD::SAMPLER_FILTER_LINEAR);
}

FSR1Effect::~FSR1Effect() {
	RD::get_singleton()->free_rid(device.linear_clamp_sampler);

	for (uint32_t i = 0; i < FFX_FSR1_PASS_COUNT; i++) {
		device.passes[i].shader->version_free(device.passes[i].shader_version);
	}
}

FSR1Context *FSR1Effect::create_context(Size2i p_internal_size, Size2i p_target_size, RD::DataFormat p_output_format) {
	FSR1Context *context = memnew(RendererRD::FSR1Context);
	context->fsr_desc.flags = FFX_FSR1_ENABLE_HIGH_DYNAMIC_RANGE | FFX_FSR1_ENABLE_RCAS;
	context->fsr_desc.maxRenderSize.width = p_internal_size.x;
	context->fsr_desc.maxRenderSize.height = p_internal_size.y;
	context->fsr_desc.displaySize.width = p_target_size.x;
	context->fsr_desc.displaySize.height = p_target_size.y;
	context->fsr_desc.outputFormat = FFXCommon::rd_format_to_ffx_surface_format(p_output_format);

	FFXCommon::create_ffx_interface(&context->fsr_desc.backendInterface, &context->scratch, &device);
	FfxErrorCode result = ffxFsr1ContextCreate(&context->fsr_context, &context->fsr_desc);
	if (result == FFX_OK) {
		return context;
	} else {
		memdelete(context);
		return nullptr;
	}
}

void FSR1Effect::process(Ref<RenderSceneBuffersRD> p_render_buffers, RID p_source_rd_texture, RID p_destination_texture) {
	FSR1Context *fsr1_context = p_render_buffers->get_fsr1_context();

	Size2i internal_size = p_render_buffers->get_internal_size();
	float fsr_upscale_sharpness = p_render_buffers->get_fsr_sharpness();

	FfxFsr1DispatchDescription dispatch_desc = {};
	dispatch_desc.commandList = nullptr;
	dispatch_desc.color = FFXCommon::get_resource_rd(&p_source_rd_texture, L"color");
	dispatch_desc.output = FFXCommon::get_resource_rd(&p_destination_texture, L"output");
	dispatch_desc.renderSize.width = internal_size.width;
	dispatch_desc.renderSize.height = internal_size.height;
	dispatch_desc.enableSharpening = (fsr_upscale_sharpness > 1e-6f);
	dispatch_desc.sharpness = fsr_upscale_sharpness;

	FfxErrorCode result = ffxFsr1ContextDispatch(&fsr1_context->fsr_context, &dispatch_desc);
	ERR_FAIL_COND(result != FFX_OK);
}
