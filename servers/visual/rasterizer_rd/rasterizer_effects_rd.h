#ifndef RASTERIZER_EFFECTS_RD_H
#define RASTERIZER_EFFECTS_RD_H

#include "core/math/camera_matrix.h"
#include "render_pipeline_vertex_format_cache_rd.h"
#include "servers/visual/rasterizer_rd/shaders/blur.glsl.gen.h"
#include "servers/visual/rasterizer_rd/shaders/cubemap_roughness.glsl.gen.h"
#include "servers/visual/rasterizer_rd/shaders/sky.glsl.gen.h"

class RasterizerEffectsRD {

	enum BlurMode {
		BLUR_MODE_GAUSSIAN_BLUR,
		BLUR_MODE_GAUSSIAN_GLOW,
		BLUR_MODE_GAUSSIAN_GLOW_AUTO_EXPOSURE,
		BLUR_MODE_DOF_NEAR_LOW,
		BLUR_MODE_DOF_NEAR_MEDIUM,
		BLUR_MODE_DOF_NEAR_HIGH,
		BLUR_MODE_DOF_NEAR_MERGE_LOW,
		BLUR_MODE_DOF_NEAR_MERGE_MEDIUM,
		BLUR_MODE_DOF_NEAR_MERGE_HIGH,
		BLUR_MODE_DOF_FAR_LOW,
		BLUR_MODE_DOF_FAR_MEDIUM,
		BLUR_MODE_DOF_FAR_HIGH,
		BLUR_MODE_SSAO_MERGE,
		BLUR_MODE_SIMPLY_COPY,
		BLUR_MODE_MIPMAP,
		BLUR_MODE_MAX,

	};

	enum {
		BLUR_FLAG_HORIZONTAL = (1 << 0),
		BLUR_FLAG_USE_BLUR_SECTION = (1 << 1),
		BLUR_FLAG_USE_ORTHOGONAL_PROJECTION = (1 << 2),
		BLUR_FLAG_DOF_NEAR_FIRST_TAP = (1 << 3),
		BLUR_FLAG_GLOW_FIRST_PASS = (1 << 4)
	};

	struct BlurPushConstant {
		float section[4];
		float pixel_size[2];
		uint32_t flags;
		uint32_t pad;
		//glow
		float glow_strength;
		float glow_bloom;
		float glow_hdr_threshold;
		float glow_hdr_scale;
		float glow_exposure;
		float glow_white;
		float glow_luminance_cap;
		float glow_auto_exposure_grey;
		//dof
		float dof_begin;
		float dof_end;
		float dof_radius;
		float dof_pad;

		float dof_dir[2];
		float camera_z_far;
		float camera_z_near;

		float ssao_color[4];
	};

	struct Blur {
		BlurPushConstant push_constant;
		BlurShaderRD shader;
		RID shader_version;
		RenderPipelineVertexFormatCacheRD pipelines[BLUR_MODE_MAX];

	} blur;

	enum CubemapRoughnessSource {
		CUBEMAP_ROUGHNESS_SOURCE_PANORAMA,
		CUBEMAP_ROUGHNESS_SOURCE_CUBEMAP,
		CUBEMAP_ROUGHNESS_SOURCE_MAX
	};

	struct CubemapRoughnessPushConstant {
		uint32_t face_id;
		uint32_t sample_count;
		float roughness;
		uint32_t use_direct_write;
	};

	struct CubemapRoughness {

		CubemapRoughnessPushConstant push_constant;
		CubemapRoughnessShaderRD shader;
		RID shader_version;
		RenderPipelineVertexFormatCacheRD pipelines[CUBEMAP_ROUGHNESS_SOURCE_MAX];
	} roughness;

	struct SkyPushConstant {
		float orientation[12];
		float proj[4];
		float multiplier;
		float alpha;
		float depth;
		float pad;
	};

	struct Sky {

		SkyPushConstant push_constant;
		SkyShaderRD shader;
		RID shader_version;
		RenderPipelineVertexFormatCacheRD pipeline;
	} sky;

	RID default_sampler;
	RID index_buffer;
	RID index_array;

	Map<RID, RID> texture_to_uniform_set_cache;

	RID _get_uniform_set_from_texture(RID p_texture);

public:
	void copy(RID p_source_rd_texture, RID p_dest_framebuffer, const Rect2 &p_region);
	void gaussian_blur(RID p_source_rd_texture, RID p_framebuffer_half, RID p_rd_texture_half, RID p_dest_framebuffer, const Vector2 &p_pixel_size, const Rect2 &p_region);
	void cubemap_roughness(RID p_source_rd_texture, bool p_source_is_panorama, RID p_dest_framebuffer, uint32_t p_face_id, uint32_t p_sample_count, float p_roughness);
	void render_panorama(RD::DrawListID p_list, RenderingDevice::FramebufferFormatID p_fb_format, RID p_panorama, const CameraMatrix &p_camera, const Basis &p_orientation, float p_alpha, float p_multipler);
	void make_mipmap(RID p_source_rd_texture, RID p_framebuffer_half, const Vector2 &p_pixel_size);

	RasterizerEffectsRD();
	~RasterizerEffectsRD();
};

#endif // EFFECTS_RD_H
