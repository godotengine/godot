/*************************************************************************/
/*  pipeline_cache_rd.h                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#ifndef PIPELINE_CACHE_RD_H
#define PIPELINE_CACHE_RD_H

#include "core/os/spin_lock.h"
#include "servers/rendering/rendering_device.h"

class PipelineCacheRD {
	SpinLock spin_lock;

	RID shader;
	uint32_t input_mask;

	RD::RenderPrimitive render_primitive;
	RD::PipelineRasterizationState rasterization_state;
	RD::PipelineMultisampleState multisample_state;
	RD::PipelineDepthStencilState depth_stencil_state;
	RD::PipelineColorBlendState blend_state;
	int dynamic_state_flags;
	Vector<RD::PipelineSpecializationConstant> base_specialization_constants;

	struct Version {
		RD::VertexFormatID vertex_id;
		RD::FramebufferFormatID framebuffer_id;
		uint32_t render_pass;
		bool wireframe;
		uint32_t bool_specializations;
		RID pipeline;
	};

	Version *versions;
	uint32_t version_count;

	RID _generate_version(RD::VertexFormatID p_vertex_format_id, RD::FramebufferFormatID p_framebuffer_format_id, bool p_wireframe, uint32_t p_render_pass, uint32_t p_bool_specializations = 0);

	void _clear();

public:
	void setup(RID p_shader, RD::RenderPrimitive p_primitive, const RD::PipelineRasterizationState &p_rasterization_state, RD::PipelineMultisampleState p_multisample, const RD::PipelineDepthStencilState &p_depth_stencil_state, const RD::PipelineColorBlendState &p_blend_state, int p_dynamic_state_flags = 0, const Vector<RD::PipelineSpecializationConstant> &p_base_specialization_constants = Vector<RD::PipelineSpecializationConstant>());
	void update_specialization_constants(const Vector<RD::PipelineSpecializationConstant> &p_base_specialization_constants);
	void update_shader(RID p_shader);

	_FORCE_INLINE_ RID get_render_pipeline(RD::VertexFormatID p_vertex_format_id, RD::FramebufferFormatID p_framebuffer_format_id, bool p_wireframe = false, uint32_t p_render_pass = 0, uint32_t p_bool_specializations = 0) {
#ifdef DEBUG_ENABLED
		ERR_FAIL_COND_V_MSG(shader.is_null(), RID(),
				"Attempted to use an unused shader variant (shader is null),");
#endif

		spin_lock.lock();
		RID result;
		for (uint32_t i = 0; i < version_count; i++) {
			if (versions[i].vertex_id == p_vertex_format_id && versions[i].framebuffer_id == p_framebuffer_format_id && versions[i].wireframe == p_wireframe && versions[i].render_pass == p_render_pass && versions[i].bool_specializations == p_bool_specializations) {
				result = versions[i].pipeline;
				spin_lock.unlock();
				return result;
			}
		}
		result = _generate_version(p_vertex_format_id, p_framebuffer_format_id, p_wireframe, p_render_pass, p_bool_specializations);
		spin_lock.unlock();
		return result;
	}

	_FORCE_INLINE_ uint32_t get_vertex_input_mask() const {
		return input_mask;
	}
	void clear();
	PipelineCacheRD();
	~PipelineCacheRD();
};

#endif // RENDER_PIPELINE_CACHE_RD_H
