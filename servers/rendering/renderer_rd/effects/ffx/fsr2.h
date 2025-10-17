/**************************************************************************/
/*  fsr2.h                                                                */
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

#include "servers/rendering/renderer_rd/shaders/effects/ffx/fsr2/fsr2_accumulate_pass.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/effects/ffx/fsr2/fsr2_autogen_reactive_pass.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/effects/ffx/fsr2/fsr2_compute_luminance_pyramid_pass.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/effects/ffx/fsr2/fsr2_depth_clip_pass.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/effects/ffx/fsr2/fsr2_lock_pass.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/effects/ffx/fsr2/fsr2_rcas_pass.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/effects/ffx/fsr2/fsr2_reconstruct_previous_depth_pass.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/effects/ffx/fsr2/fsr2_tcr_autogen_pass.glsl.gen.h"
#include "servers/rendering/rendering_server.h"

#include "ffx_common.h"
#include "thirdparty/amd-ffx/ffx_fsr2.h"

namespace RendererRD {
class FSR2Context {
public:
	FFXCommon::Scratch scratch;
	FfxFsr2Context fsr_context;
	FfxFsr2ContextDescription fsr_desc;

	~FSR2Context();
};

class FSR2Effect {
public:
	struct Parameters {
		FSR2Context *context;
		Size2i internal_size;
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

	FSR2Effect();
	~FSR2Effect();
	FSR2Context *create_context(Size2i p_internal_size, Size2i p_target_size);
	void upscale(const Parameters &p_params);

private:
	struct {
		Fsr2DepthClipPassShaderRD depth_clip;
		Fsr2ReconstructPreviousDepthPassShaderRD reconstruct_previous_depth;
		Fsr2LockPassShaderRD lock;
		Fsr2AccumulatePassShaderRD accumulate;
		Fsr2RcasPassShaderRD rcas;
		Fsr2ComputeLuminancePyramidPassShaderRD compute_luminance_pyramid;
		Fsr2AutogenReactivePassShaderRD autogen_reactive;
		Fsr2TcrAutogenPassShaderRD tcr_autogen;
	} shaders;

	FFXCommon::Device device;
};
} //namespace RendererRD
