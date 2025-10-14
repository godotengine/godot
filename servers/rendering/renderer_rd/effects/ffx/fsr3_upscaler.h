/**************************************************************************/
/*  fsr3_upscaler.h                                                       */
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

#pragma once

#include "servers/rendering/renderer_rd/shaders/effects/ffx/fsr3upscaler/fsr3upscaler_accumulate_pass.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/effects/ffx/fsr3upscaler/fsr3upscaler_autogen_reactive_pass.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/effects/ffx/fsr3upscaler/fsr3upscaler_debug_view_pass.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/effects/ffx/fsr3upscaler/fsr3upscaler_luma_instability_pass.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/effects/ffx/fsr3upscaler/fsr3upscaler_luma_pyramid_pass.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/effects/ffx/fsr3upscaler/fsr3upscaler_prepare_inputs_pass.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/effects/ffx/fsr3upscaler/fsr3upscaler_prepare_reactivity_pass.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/effects/ffx/fsr3upscaler/fsr3upscaler_rcas_pass.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/effects/ffx/fsr3upscaler/fsr3upscaler_shading_change_pass.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/effects/ffx/fsr3upscaler/fsr3upscaler_shading_change_pyramid_pass.glsl.gen.h"

#include "ffx_common.h"
#include "servers/rendering/rendering_server.h"

#include "thirdparty/amd-ffx/ffx_fsr3upscaler.h"

namespace RendererRD {
class FSR3UpscalerContext {
public:
	FFXCommonContext *ffx_common_context;
	FfxFsr3UpscalerContext fsr_context;
	FfxFsr3UpscalerContextDescription fsr_desc;

	// Output resources from FSR3 Upscaler that are required for frame generation
	FfxResourceInternal reconstructed_prev_nearest_depth;
	FfxResourceInternal dilated_depth;
	FfxResourceInternal dilated_motion_vectors;

	~FSR3UpscalerContext();
};

class FSR3UpscalerEffect {
public:
	struct Parameters {
		FSR3UpscalerContext *context;
		Size2i internal_size;
		Size2i target_size;
		RID color;
		RID depth;
		RID velocity;
		RID reactive;
		RID exposure;
		RID output;
		float z_near = 0.0f;
		float z_far = 0.0f;
		float fovy = 0.0f;
		Vector2 jitter;
		float delta_time = 0.0f;
		float sharpness = 0.0f;
		bool reset_accumulation = false;
		Projection reprojection;
	};

	FSR3UpscalerEffect();
	~FSR3UpscalerEffect();
	FSR3UpscalerContext *create_context(Size2i p_internal_size, Size2i p_target_size);
	void upscale(const Parameters &p_params);

private:
	struct {
		Fsr3UpscalerPrepareInputsPassShaderRD prepare_inputs;
		Fsr3UpscalerLumaPyramidPassShaderRD luma_pyramid;
		Fsr3UpscalerShadingChangePyramidPassShaderRD shading_change_pyramid;
		Fsr3UpscalerShadingChangePassShaderRD shading_change;
		Fsr3UpscalerPrepareReactivityPassShaderRD prepare_reactivity;
		Fsr3UpscalerLumaInstabilityPassShaderRD luma_instability;
		Fsr3UpscalerAccumulatePassShaderRD accumulate;
		Fsr3UpscalerRcasPassShaderRD rcas;
		Fsr3UpscalerDebugViewPassShaderRD debug_view;
		Fsr3UpscalerAutogenReactivePassShaderRD autogen_reactive;
	} shaders;
};
}
