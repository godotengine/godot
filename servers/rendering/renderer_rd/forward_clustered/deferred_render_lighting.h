/**************************************************************************/
/*  deferred_render_lighting.h                                            */
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

#include "servers/rendering/renderer_rd/pipeline_cache_rd.h"
#include "servers/rendering/renderer_rd/shaders/forward_clustered/deferred_renderer_lighting.glsl.gen.h"
#include "servers/rendering/renderer_rd/storage_rd/render_data_rd.h"
#include "servers/rendering/renderer_scene_render.h"

#include "servers/rendering_server.h"

namespace RendererRD {

class DeferredRendererLighting {
private:
	const int SAMPLERS_BINDING_FIRST_INDEX = 14;

	enum DeferredLightingModes {
		DEFERRED_LIGHTING_MODE_DEFAULT,
		DEFERRED_LIGHTING_MODE_MULTIVIEW,
		DEFERRED_LIGHTING_MODE_MAX,
	};

	struct DRPushConstant {
		int32_t width;
		int32_t height;
		uint32_t view;
		float pad;
	};

	struct DeferredLighting {
		DeferredRendererLightingShaderRD compute_shader;
		RID shader_version;
		RID compute_pipelines[DEFERRED_LIGHTING_MODE_MAX];
		RID uniform_set;
	} deferred_lighting;

	RID shadow_sampler;

public:
	void init_shader(uint32_t p_max_directional_lights);
	DeferredRendererLighting();
	~DeferredRendererLighting();

	void apply_lighting(RenderDataRD *p_render_data, bool p_use_directional_shadow_atlas, const RendererRD::MaterialStorage::Samplers &p_samplers, RS::LightProjectorFilter p_light_projection_filter);
};

} // namespace RendererRD
