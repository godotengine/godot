#include "effects_rd.h"

RID EffectsRD::_get_uniform_set_from_texture(RID p_texture) {

	if (texture_to_uniform_set_cache.has(p_texture)) {
		RID uniform_set = texture_to_uniform_set_cache[p_texture];
		if (RD::get_singleton()->uniform_set_is_valid(uniform_set)) {
			return uniform_set;
		}
	}

	Vector<RD::Uniform> uniforms;
	RD::Uniform u;
	u.type = RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE;
	u.binding = 0;
	u.ids.push_back(default_sampler);
	u.ids.push_back(p_texture);
	uniforms.push_back(u);
	//any thing with the same configuration (one texture in binding 0 for set 0), is good
	RID uniform_set = RD::get_singleton()->uniform_set_create(uniforms, blur.shader.version_get_shader(blur.shader_version, 0), 0);

	texture_to_uniform_set_cache[p_texture] = uniform_set;

	return uniform_set;
}

void EffectsRD::copy(RID p_source_rd_texture, RID p_dest_framebuffer, const Rect2 &p_region) {

	zeromem(&blur.push_constant, sizeof(BlurPushConstant));

	if (p_region != Rect2()) {
		blur.push_constant.flags = BLUR_FLAG_USE_BLUR_SECTION;
		blur.push_constant.section[0] = p_region.position.x;
		blur.push_constant.section[1] = p_region.position.y;
		blur.push_constant.section[2] = p_region.size.width;
		blur.push_constant.section[3] = p_region.size.height;
	}

	RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(p_dest_framebuffer, RD::INITIAL_ACTION_KEEP_COLOR, RD::FINAL_ACTION_READ_COLOR_DISCARD_DEPTH);
	RD::get_singleton()->draw_list_bind_render_pipeline(draw_list, blur.pipelines[BLUR_MODE_SIMPLY_COPY].get_render_pipeline(RD::INVALID_ID, RD::get_singleton()->framebuffer_get_format(p_dest_framebuffer)));
	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, _get_uniform_set_from_texture(p_source_rd_texture), 0);
	RD::get_singleton()->draw_list_bind_index_array(draw_list, index_array);
	RD::get_singleton()->draw_list_set_push_constant(draw_list, &blur.push_constant, sizeof(BlurPushConstant));
	RD::get_singleton()->draw_list_draw(draw_list, true);
	RD::get_singleton()->draw_list_end();
}

void EffectsRD::gaussian_blur(RID p_source_rd_texture, RID p_framebuffer_half, RID p_rd_texture_half, RID p_dest_framebuffer, const Vector2 &p_pixel_size, const Rect2 &p_region) {

	zeromem(&blur.push_constant, sizeof(BlurPushConstant));

	uint32_t base_flags = 0;
	if (p_region != Rect2()) {
		base_flags = BLUR_FLAG_USE_BLUR_SECTION;
		blur.push_constant.section[0] = p_region.position.x;
		blur.push_constant.section[1] = p_region.position.y;
		blur.push_constant.section[2] = p_region.size.width;
		blur.push_constant.section[3] = p_region.size.height;
	}

	blur.push_constant.pixel_size[0] = p_pixel_size.x;
	blur.push_constant.pixel_size[1] = p_pixel_size.y;

	//HORIZONTAL
	RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(p_framebuffer_half, RD::INITIAL_ACTION_KEEP_COLOR, RD::FINAL_ACTION_READ_COLOR_DISCARD_DEPTH);
	RD::get_singleton()->draw_list_bind_render_pipeline(draw_list, blur.pipelines[BLUR_MODE_GAUSSIAN_BLUR].get_render_pipeline(RD::INVALID_ID, RD::get_singleton()->framebuffer_get_format(p_framebuffer_half)));
	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, _get_uniform_set_from_texture(p_source_rd_texture), 0);
	RD::get_singleton()->draw_list_bind_index_array(draw_list, index_array);

	blur.push_constant.flags = base_flags | BLUR_FLAG_HORIZONTAL;
	RD::get_singleton()->draw_list_set_push_constant(draw_list, &blur.push_constant, sizeof(BlurPushConstant));

	RD::get_singleton()->draw_list_draw(draw_list, true);
	RD::get_singleton()->draw_list_end();

	//VERTICAL
	draw_list = RD::get_singleton()->draw_list_begin(p_dest_framebuffer, RD::INITIAL_ACTION_KEEP_COLOR, RD::FINAL_ACTION_READ_COLOR_DISCARD_DEPTH);
	RD::get_singleton()->draw_list_bind_render_pipeline(draw_list, blur.pipelines[BLUR_MODE_GAUSSIAN_BLUR].get_render_pipeline(RD::INVALID_ID, RD::get_singleton()->framebuffer_get_format(p_dest_framebuffer)));
	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, _get_uniform_set_from_texture(p_rd_texture_half), 0);
	RD::get_singleton()->draw_list_bind_index_array(draw_list, index_array);

	blur.push_constant.flags = base_flags;
	RD::get_singleton()->draw_list_set_push_constant(draw_list, &blur.push_constant, sizeof(BlurPushConstant));

	RD::get_singleton()->draw_list_draw(draw_list, true);
	RD::get_singleton()->draw_list_end();
}

EffectsRD::EffectsRD() {

	// Initialize blur
	Vector<String> blur_modes;
	blur_modes.push_back("\n#define MODE_GAUSSIAN_BLUR\n");
	blur_modes.push_back("\n#define MODE_GAUSSIAN_GLOW\n");
	blur_modes.push_back("\n#define MODE_GAUSSIAN_GLOW\n#define GLOW_USE_AUTO_EXPOSURE\n");
	blur_modes.push_back("\n#define MODE_DOF_NEAR_BLUR\n#define DOF_QUALITY_LOW\n");
	blur_modes.push_back("\n#define MODE_DOF_NEAR_BLUR\n#define DOF_QUALITY_MEDIUM\n");
	blur_modes.push_back("\n#define MODE_DOF_NEAR_BLUR\n#define DOF_QUALITY_HIGH\n");
	blur_modes.push_back("\n#define MODE_DOF_NEAR_BLUR\n#define DOF_QUALITY_LOW\n#define DOF_NEAR_BLUR_MERGE\n");
	blur_modes.push_back("\n#define MODE_DOF_NEAR_BLUR\n#define DOF_QUALITY_MEDIUM\n#define DOF_NEAR_BLUR_MERGE\n");
	blur_modes.push_back("\n#define MODE_DOF_NEAR_BLUR\n#define DOF_QUALITY_HIGH\n#define DOF_NEAR_BLUR_MERGE\n");
	blur_modes.push_back("\n#define MODE_DOF_FAR_BLUR\n#define DOF_QUALITY_LOW\n");
	blur_modes.push_back("\n#define MODE_DOF_FAR_BLUR\n#define DOF_QUALITY_MEDIUM\n");
	blur_modes.push_back("\n#define MODE_DOF_FAR_BLUR\n#define DOF_QUALITY_HIGH\n");
	blur_modes.push_back("\n#define MODE_SSAO_MERGE\n");
	blur_modes.push_back("\n#define MODE_SIMPLE_COPY\n");

	blur.shader.initialize(blur_modes);
	zeromem(&blur.push_constant, sizeof(BlurPushConstant));
	blur.shader_version = blur.shader.version_create();

	for (int i = 0; i < BLUR_MODE_MAX; i++) {
		blur.pipelines[i].setup(blur.shader.version_get_shader(blur.shader_version, i), RD::RENDER_PRIMITIVE_TRIANGLES, RD::PipelineRasterizationState(), RD::PipelineMultisampleState(), RD::PipelineDepthStencilState(), RD::PipelineColorBlendState::create_disabled(), 0);
	}

	RD::SamplerState sampler;
	sampler.mag_filter = RD::SAMPLER_FILTER_LINEAR;
	sampler.min_filter = RD::SAMPLER_FILTER_LINEAR;
	sampler.max_lod = 0;

	default_sampler = RD::get_singleton()->sampler_create(sampler);

	{ //create index array for copy shaders
		PoolVector<uint8_t> pv;
		pv.resize(6 * 4);
		{
			PoolVector<uint8_t>::Write w = pv.write();
			int *p32 = (int *)w.ptr();
			p32[0] = 0;
			p32[1] = 1;
			p32[2] = 2;
			p32[3] = 0;
			p32[4] = 2;
			p32[5] = 3;
		}
		index_buffer = RD::get_singleton()->index_buffer_create(6, RenderingDevice::INDEX_BUFFER_FORMAT_UINT32, pv);
		index_array = RD::get_singleton()->index_array_create(index_buffer, 0, 6);
	}
}

EffectsRD::~EffectsRD() {
}
