#include "render_pipeline_cache_rd.h"
#include "core/os/memory.h"

RID RenderPipelineCacheRD::_generate_version(RD::VertexFormatID p_format_id) {
	RID pipeline = RD::get_singleton()->render_pipeline_create(shader, framebuffer_format, p_format_id, render_primitive, rasterization_state, multisample_state, depth_stencil_state, blend_state, dynamic_state_flags);
	ERR_FAIL_COND_V(pipeline.is_null(), RID());
	versions = (Version *)memrealloc(versions, sizeof(Version) * (version_count + 1));
	versions[version_count].format_id = p_format_id;
	versions[version_count].pipeline = pipeline;
	version_count++;
	return pipeline;
}

void RenderPipelineCacheRD::_clear() {

	if (versions) {
		for (uint32_t i = 0; i < version_count; i++) {
			RD::get_singleton()->free(versions[i].pipeline);
		}
		version_count = 0;
		memfree(versions);
		versions = NULL;
	}
}

void RenderPipelineCacheRD::setup(RID p_shader, RD::FramebufferFormatID p_framebuffer_format, RD::RenderPrimitive p_primitive, const RD::PipelineRasterizationState &p_rasterization_state, RD::PipelineMultisampleState p_multisample, const RD::PipelineDepthStencilState &p_depth_stencil_state, const RD::PipelineColorBlendState &p_blend_state, int p_dynamic_state_flags) {
	ERR_FAIL_COND(p_shader.is_null());
	shader = p_shader;
	framebuffer_format = p_framebuffer_format;
	render_primitive = p_primitive;
	rasterization_state = p_rasterization_state;
	multisample_state = p_multisample;
	depth_stencil_state = p_depth_stencil_state;
	blend_state = p_blend_state;
	dynamic_state_flags = p_dynamic_state_flags;
}

void RenderPipelineCacheRD::update_shader(RID p_shader) {
	ERR_FAIL_COND(p_shader.is_null());
	_clear();
	setup(p_shader, framebuffer_format, render_primitive, rasterization_state, multisample_state, depth_stencil_state, blend_state, dynamic_state_flags);
}

RenderPipelineCacheRD::RenderPipelineCacheRD() {
	version_count = 0;
	versions = NULL;
}

RenderPipelineCacheRD::~RenderPipelineCacheRD() {
	_clear();
}
