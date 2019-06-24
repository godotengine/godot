#include "render_pipeline_vertex_format_cache_rd.h"
#include "core/os/memory.h"

RID RenderPipelineVertexFormatCacheRD::_generate_version(RD::VertexFormatID p_format_id, RenderingDevice::TextureSamples p_samples) {

	RD::PipelineMultisampleState multisample_state_version;
	if (p_samples != RD::TEXTURE_SAMPLES_1) {
		multisample_state_version = multisample_state;
		multisample_state_version.sample_count = p_samples;
	}
	RID pipeline = RD::get_singleton()->render_pipeline_create(shader, framebuffer_format, p_format_id, render_primitive, rasterization_state, multisample_state, depth_stencil_state, blend_state, dynamic_state_flags);
	ERR_FAIL_COND_V(pipeline.is_null(), RID());
	versions[p_samples] = (Version *)memrealloc(versions[p_samples], sizeof(Version) * (version_count[p_samples] + 1));
	versions[p_samples][version_count[p_samples]].format_id = p_format_id;
	versions[p_samples][version_count[p_samples]].pipeline = pipeline;
	version_count[p_samples]++;
	return pipeline;
}

void RenderPipelineVertexFormatCacheRD::_clear() {

	for (int v = 0; v < RD::TEXTURE_SAMPLES_MAX; v++) {
		if (versions[v]) {
			for (uint32_t i = 0; i < version_count[v]; i++) {
				//shader may be gone, so this may not be valid
				if (RD::get_singleton()->render_pipeline_is_valid(versions[v][i].pipeline)) {
					RD::get_singleton()->free(versions[v][i].pipeline);
				}
			}
			version_count[v] = 0;
			memfree(versions[v]);
			versions[v] = NULL;
		}
	}
}

void RenderPipelineVertexFormatCacheRD::setup(RID p_shader, RD::FramebufferFormatID p_framebuffer_format, RD::RenderPrimitive p_primitive, const RD::PipelineRasterizationState &p_rasterization_state, RD::PipelineMultisampleState p_multisample, const RD::PipelineDepthStencilState &p_depth_stencil_state, const RD::PipelineColorBlendState &p_blend_state, int p_dynamic_state_flags) {
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

void RenderPipelineVertexFormatCacheRD::update_shader(RID p_shader) {
	ERR_FAIL_COND(p_shader.is_null());
	_clear();
	setup(p_shader, framebuffer_format, render_primitive, rasterization_state, multisample_state, depth_stencil_state, blend_state, dynamic_state_flags);
}

RenderPipelineVertexFormatCacheRD::RenderPipelineVertexFormatCacheRD() {
	for (int i = 0; i < RD::TEXTURE_SAMPLES_MAX; i++) {
		version_count[i] = 0;
		versions[i] = NULL;
	}
}

RenderPipelineVertexFormatCacheRD::~RenderPipelineVertexFormatCacheRD() {
	_clear();
}
