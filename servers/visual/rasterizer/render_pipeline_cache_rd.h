#ifndef RENDER_PIPELINE_CACHE_RD_H
#define RENDER_PIPELINE_CACHE_RD_H

#include "servers/visual/rendering_device.h"

class RenderPipelineCacheRD {

	RID shader;

	RD::FramebufferFormatID framebuffer_format;
	RD::RenderPrimitive render_primitive;
	RD::PipelineRasterizationState rasterization_state;
	RD::PipelineMultisampleState multisample_state;
	RD::PipelineDepthStencilState depth_stencil_state;
	RD::PipelineColorBlendState blend_state;
	int dynamic_state_flags;

	struct Version {
		RD::VertexFormatID format_id;
		RID pipeline;
	};

	Version *versions;
	uint32_t version_count;

	RID _generate_version(RD::VertexFormatID p_format_id);

	void _clear();

public:
	void setup(RID p_shader, RD::FramebufferFormatID p_framebuffer_format, RD::RenderPrimitive p_primitive, const RD::PipelineRasterizationState &p_rasterization_state, RD::PipelineMultisampleState p_multisample, const RD::PipelineDepthStencilState &p_depth_stencil_state, const RD::PipelineColorBlendState &p_blend_state, int p_dynamic_state_flags = 0);
	void update_shader(RID p_shader);

	_FORCE_INLINE_ RID get_render_pipeline(RD::VertexFormatID p_format_id) {
		for (uint32_t i = 0; i < version_count; i++) {
			if (versions[i].format_id == p_format_id) {
				return versions[i].pipeline;
			}
		}
		return _generate_version(p_format_id);
	}

	RenderPipelineCacheRD();
	~RenderPipelineCacheRD();
};

#endif // RENDER_PIPELINE_CACHE_RD_H
