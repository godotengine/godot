#ifndef RENDER_PIPELINE_CACHE_RD_H
#define RENDER_PIPELINE_CACHE_RD_H

#include "servers/visual/rendering_device.h"

class RenderPipelineVertexFormatCacheRD {

	RID shader;

	RD::FramebufferFormatID framebuffer_format;
	RD::RenderPrimitive render_primitive;
	RD::PipelineRasterizationState rasterization_state;
	RD::PipelineMultisampleState multisample_state;
	RD::PipelineDepthStencilState depth_stencil_state;
	RD::PipelineColorBlendState blend_state;
	int dynamic_state_flags;

	struct Version {
		RD::VertexFormatID vertex_id;
		RD::FramebufferFormatID framebuffer_id;
		RID pipeline;
	};

	Version *versions;
	uint32_t version_count;

	RID _generate_version(RD::VertexFormatID p_vertex_format_id, RD::FramebufferFormatID p_framebuffer_format_id);

	void _clear();

public:
	void setup(RID p_shader, RD::RenderPrimitive p_primitive, const RD::PipelineRasterizationState &p_rasterization_state, RD::PipelineMultisampleState p_multisample, const RD::PipelineDepthStencilState &p_depth_stencil_state, const RD::PipelineColorBlendState &p_blend_state, int p_dynamic_state_flags = 0);
	void update_shader(RID p_shader);

	_FORCE_INLINE_ RID get_render_pipeline(RD::VertexFormatID p_vertex_format_id, RD::FramebufferFormatID p_framebuffer_format_id) {
		for (uint32_t i = 0; i < version_count; i++) {
			if (versions[i].vertex_id == p_vertex_format_id && versions[i].framebuffer_id == p_framebuffer_format_id) {
				return versions[i].pipeline;
			}
		}
		return _generate_version(p_vertex_format_id, p_framebuffer_format_id);
	}

	RenderPipelineVertexFormatCacheRD();
	~RenderPipelineVertexFormatCacheRD();
};

#endif // RENDER_PIPELINE_CACHE_RD_H
