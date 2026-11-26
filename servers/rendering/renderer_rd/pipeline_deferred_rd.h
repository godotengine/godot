/**************************************************************************/
/*  pipeline_deferred_rd.h                                                */
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

#include "servers/rendering/rendering_device.h"

// Helper class for automatically deferring compilation of a pipeline to a background task.
// When attempting to retrieve the pipeline with the getter, the caller will automatically
// wait for it to be ready.

class PipelineDeferredRD {
protected:
	struct CreationParameters {
		RID shader;
		RD::FramebufferFormatID framebuffer_format;
		RD::VertexFormatID vertex_format;
		RD::RenderPrimitive render_primitive;
		RD::PipelineRasterizationState rasterization_state;
		RD::PipelineMultisampleState multisample_state;
		RD::PipelineDepthStencilState depth_stencil_state;
		RD::PipelineColorBlendState blend_state;
		BitField<RD::PipelineDynamicStateFlags> dynamic_state_flags;
		uint32_t for_render_pass;
		Vector<RD::PipelineSpecializationConstant> specialization_constants;
		bool is_compute;
	};

	RID pipeline;
	WorkerThreadPool::TaskID task = WorkerThreadPool::INVALID_TASK_ID;

	void _create(const CreationParameters &c) {
		if (c.is_compute) {
			pipeline = RD::get_singleton()->compute_pipeline_create(c.shader, c.specialization_constants);
		} else {
			pipeline = RD::get_singleton()->render_pipeline_create(c.shader, c.framebuffer_format, c.vertex_format, c.render_primitive, c.rasterization_state, c.multisample_state, c.depth_stencil_state, c.blend_state, c.dynamic_state_flags, c.for_render_pass, c.specialization_constants);
		}
	}

	void _start(const CreationParameters &c) {
		free();
		task = WorkerThreadPool::get_singleton()->add_template_task(this, &PipelineDeferredRD::_create, c, true, "PipelineCompilation");
	}

	void _wait() {
		if (task != WorkerThreadPool::INVALID_TASK_ID) {
			WorkerThreadPool::get_singleton()->wait_for_task_completion(task);
			task = WorkerThreadPool::INVALID_TASK_ID;
		}
	}

public:
	PipelineDeferredRD() {
		// Default constructor.
	}

	~PipelineDeferredRD() {
#ifdef DEV_ENABLED
		ERR_FAIL_COND_MSG(pipeline.is_valid(), "'free()' must be called manually before deconstruction and before the corresponding shader is freed.");
#endif
	}

	void create_render_pipeline(RID p_shader, RD::FramebufferFormatID p_framebuffer_format, RD::VertexFormatID p_vertex_format, RD::RenderPrimitive p_render_primitive, const RD::PipelineRasterizationState &p_rasterization_state, const RD::PipelineMultisampleState &p_multisample_state, const RD::PipelineDepthStencilState &p_depth_stencil_state, const RD::PipelineColorBlendState &p_blend_state, BitField<RD::PipelineDynamicStateFlags> p_dynamic_state_flags = 0, uint32_t p_for_render_pass = 0, const Vector<RD::PipelineSpecializationConstant> &p_specialization_constants = Vector<RD::PipelineSpecializationConstant>()) {
		CreationParameters c;
		c.shader = p_shader;
		c.framebuffer_format = p_framebuffer_format;
		c.vertex_format = p_vertex_format;
		c.render_primitive = p_render_primitive;
		c.rasterization_state = p_rasterization_state;
		c.multisample_state = p_multisample_state;
		c.depth_stencil_state = p_depth_stencil_state;
		c.blend_state = p_blend_state;
		c.dynamic_state_flags = p_dynamic_state_flags;
		c.for_render_pass = p_for_render_pass;
		c.specialization_constants = p_specialization_constants;
		c.is_compute = false;
		_start(c);
	}

	void create_compute_pipeline(RID p_shader, const Vector<RD::PipelineSpecializationConstant> &p_specialization_constants = Vector<RD::PipelineSpecializationConstant>()) {
		CreationParameters c = {};
		c.shader = p_shader;
		c.specialization_constants = p_specialization_constants;
		c.is_compute = true;
		_start(c);
	}

	RID get_rid() {
		_wait();
		return pipeline;
	}

	void free() {
		_wait();

		if (pipeline.is_valid()) {
#ifdef DEV_ENABLED
			ERR_FAIL_COND_MSG(!(RD::get_singleton()->render_pipeline_is_valid(pipeline) || RD::get_singleton()->compute_pipeline_is_valid(pipeline)), "`free()` must be called  manually before the dependent shader is freed.");
#endif
			RD::get_singleton()->free_rid(pipeline);
			pipeline = RID();
		}
	}
};
