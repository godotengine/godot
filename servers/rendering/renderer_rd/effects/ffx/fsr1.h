/**************************************************************************/
/*  fsr1.h                                                                */
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

#include "servers/rendering/renderer_rd/shaders/effects/ffx/fsr1/fsr1_easu_pass.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/effects/ffx/fsr1/fsr1_rcas_pass.glsl.gen.h"

#include "ffx_common.h"
#include "servers/rendering/renderer_rd/effects/spatial_upscaler.h"
#include "servers/rendering/rendering_server.h"

#include "thirdparty/amd-ffx/ffx_fsr1.h"

namespace RendererRD {
class FSR1Context {
public:
	FFXCommon::Scratch scratch;
	FfxFsr1Context fsr_context;
	FfxFsr1ContextDescription fsr_desc;

	~FSR1Context();
};

class FSR1Effect : public SpatialUpscaler {
public:
	FSR1Effect();
	~FSR1Effect() override;

	const Span<char> get_label() const final { return "FSR 1.2 Upscale"; }
	void ensure_context(Ref<RenderSceneBuffersRD> p_render_buffers) final;
	void process(Ref<RenderSceneBuffersRD> p_render_buffers, RID p_source_rd_texture, RID p_destination_texture) final;

	FSR1Context *create_context(Size2i p_internal_size, Size2i p_target_size, RD::DataFormat p_output_format);

private:
	struct {
		Fsr1EasuPassShaderRD easu;
		Fsr1RcasPassShaderRD rcas;
	} shaders;

	FFXCommon::Device device;
};
} //namespace RendererRD
