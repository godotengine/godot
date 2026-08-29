/**************************************************************************/
/*  debug_effects.h                                                       */
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
#include "servers/rendering/renderer_rd/shaders/effects/hddagi_screen_probe_montage.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/effects/motion_vectors.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/effects/shadow_frustum.glsl.gen.h"

namespace RendererRD {

class DebugEffects {
private:
	struct {
		RD::VertexFormatID vertex_format;
		RID vertex_buffer;
		RID vertex_array;

		RID index_buffer;
		RID index_array;

		RID lines_buffer;
		RID lines_array;
	} frustum;

	struct ShadowFrustumPushConstant {
		float mvp[16];
		float color[4];
	};

	enum ShadowFrustumPipelines {
		SFP_TRANSPARENT,
		SFP_WIREFRAME,
		SFP_MAX
	};

	struct {
		ShadowFrustumShaderRD shader;
		RID shader_version;
		PipelineCacheRD pipelines[SFP_MAX];
	} shadow_frustum;

	struct MotionVectorsPushConstant {
		float reprojection_matrix[16];
		float resolution[2];
		uint32_t force_derive_from_depth;
		uint32_t pad;
	};

	struct {
		MotionVectorsShaderRD shader;
		RID shader_version;
		PipelineCacheRD pipeline;
		MotionVectorsPushConstant push_constant;
	} motion_vectors;

	enum HddagiScreenProbeMontageMode {
		HDDAGI_SCREEN_PROBE_MONTAGE_MONO,
		HDDAGI_SCREEN_PROBE_MONTAGE_MULTIVIEW,
		HDDAGI_SCREEN_PROBE_MONTAGE_MAX,
	};

	struct HddagiScreenProbeMontagePushConstant {
		float resolution[2];
		float selected_radiance_scale;
		uint32_t flags;
		uint32_t surface_layer_stride;
		uint32_t surface_history_slot;
		uint32_t hiz_mip_count;
		uint32_t pad;
	};

	struct {
		HddagiScreenProbeMontageShaderRD shader;
		RID shader_version;
		PipelineCacheRD pipelines[HDDAGI_SCREEN_PROBE_MONTAGE_MAX];
	} hddagi_screen_probe_montage;

	void _create_frustum_arrays();

protected:
public:
	enum HddagiScreenProbeMontageFlags {
		HDDAGI_SCREEN_PROBE_MONTAGE_HAS_SELECTED_OUTPUT = 1u << 0,
		HDDAGI_SCREEN_PROBE_MONTAGE_HAS_HIZ = 1u << 1,
		HDDAGI_SCREEN_PROBE_MONTAGE_HAS_NORMAL_ROUGHNESS = 1u << 2,
		HDDAGI_SCREEN_PROBE_MONTAGE_HAS_VELOCITY = 1u << 3,
		HDDAGI_SCREEN_PROBE_MONTAGE_HAS_DIRECTIONAL_ATLAS = 1u << 4,
		HDDAGI_SCREEN_PROBE_MONTAGE_HAS_DIRECTIONAL_ADAPTIVE = 1u << 5,
	};

	DebugEffects();
	~DebugEffects();

	void draw_shadow_frustum(RID p_light, const Projection &p_cam_projection, const Transform3D &p_cam_transform, RID p_dest_fb, const Rect2 p_rect);
	void draw_motion_vectors(RID p_velocity, RID p_depth, RID p_dest_fb, const Projection &p_current_projection, const Transform3D &p_current_transform, const Projection &p_previous_projection, const Transform3D &p_previous_transform, Size2i p_resolution);
	void draw_hddagi_screen_probe_montage(RID p_dest_fb, const Rect2i &p_rect, RID p_resolved_radiance, RID p_selected_radiance, RID p_probe_surface, RID p_trace_debug, RID p_hiz, RID p_normal_roughness, RID p_velocity, RID p_directional_radiance, RID p_directional_filtered, RID p_directional_irradiance, RID p_directional_adaptive_tile_data, RID p_directional_history_age, RID p_directional_adaptive_counter, float p_selected_radiance_scale, uint32_t p_flags, uint32_t p_surface_layer_stride, uint32_t p_surface_history_slot, uint32_t p_hiz_mip_count, bool p_multiview);
};

} // namespace RendererRD
