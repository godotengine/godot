#include "rasterizer_scene_rd.h"
#include "core/project_settings.h"

RID RasterizerSceneRD::sky_create() {
	return sky_owner.make_rid(Sky());
}

void RasterizerSceneRD::_sky_invalidate(Sky *p_sky) {
	if (!p_sky->dirty) {
		p_sky->dirty = true;
		p_sky->dirty_list = dirty_sky_list;
		dirty_sky_list = p_sky;
	}
}

void RasterizerSceneRD::sky_set_radiance_size(RID p_sky, int p_radiance_size) {
	Sky *sky = sky_owner.getornull(p_sky);
	ERR_FAIL_COND(!sky);
	ERR_FAIL_COND(p_radiance_size < 32 || p_radiance_size > 2048);
	if (sky->radiance_size == p_radiance_size) {
		return;
	}
	sky->radiance_size = p_radiance_size;
	_sky_invalidate(sky);
	if (sky->radiance.is_valid()) {
		//if size changes, everything must be cleared
		RD::get_singleton()->free(sky->radiance);
		//everything else gets dependency, erase, so just clean it up
		sky->radiance = RID();
		sky->layers.clear();
		sky->radiance_base_cubemap = RID();
	}
}

void RasterizerSceneRD::sky_set_mode(RID p_sky, VS::SkyMode p_mode) {
	Sky *sky = sky_owner.getornull(p_sky);
	ERR_FAIL_COND(!sky);

	if (sky->mode == p_mode) {
		return;
	}

	sky->mode = p_mode;
	_sky_invalidate(sky);
}

void RasterizerSceneRD::sky_set_texture(RID p_sky, RID p_panorama) {

	Sky *sky = sky_owner.getornull(p_sky);
	ERR_FAIL_COND(!sky);

	if (sky->panorama.is_valid()) {
		sky->panorama = RID();
		RD::get_singleton()->free(sky->radiance);
		sky->radiance = RID();
	}

	sky->panorama = p_panorama;

	if (!sky->panorama.is_valid())
		return; //cleared

	_sky_invalidate(sky);
}
void RasterizerSceneRD::_update_dirty_skys() {

	Sky *sky = dirty_sky_list;

	while (sky) {

		//update sky configuration if texture is missing

		if (sky->radiance.is_null()) {
			//recreate radiance and all data
			int mipmaps = Image::get_image_required_mipmaps(sky->radiance_size, sky->radiance_size, Image::FORMAT_RGBAH) + 1;
			if (sky->mode == VS::SKY_MODE_REALTIME) {
				//use less mipmaps
				mipmaps = MIN(8, mipmaps);
			}

			uint32_t w = sky->radiance_size, h = sky->radiance_size;

			if (sky_use_cubemap_array) {
				//array (higher quality, 6 times more memory)
				RD::TextureFormat tf;
				tf.array_layers = roughness_layers * 6;
				tf.format = RD::DATA_FORMAT_R16G16B16A16_SFLOAT;
				tf.type = RD::TEXTURE_TYPE_CUBE_ARRAY;
				tf.mipmaps = mipmaps;
				tf.width = w;
				tf.height = h;
				tf.usage_bits = RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT;

				sky->radiance = RD::get_singleton()->texture_create(tf, RD::TextureView());

				for (int i = 0; i < roughness_layers; i++) {
					Sky::Layer layer;
					uint32_t mmw = w;
					uint32_t mmh = h;
					layer.mipmaps.resize(mipmaps);
					for (int j = 0; j < mipmaps; j++) {
						Sky::Layer::Mipmap &mm = layer.mipmaps.write[j];
						mm.size.width = mmw;
						mm.size.height = mmh;
						for (int k = 0; k < 6; k++) {
							mm.views[k] = RD::get_singleton()->texture_create_shared_from_slice(RD::TextureView(), sky->radiance, i * 6 + k, j);
							Vector<RID> fbtex;
							fbtex.push_back(mm.views[k]);
							mm.framebuffers[k] = RD::get_singleton()->framebuffer_create(fbtex);
						}

						mmw = MAX(1, mmw >> 1);
						mmh = MAX(1, mmh >> 1);
					}

					sky->layers.push_back(layer);
				}

			} else {
				//regular cubemap, lower quality (aliasing, less memory)
				RD::TextureFormat tf;
				tf.array_layers = 6;
				tf.format = RD::DATA_FORMAT_R16G16B16A16_SFLOAT;
				tf.type = RD::TEXTURE_TYPE_CUBE;
				tf.mipmaps = roughness_layers;
				tf.width = w;
				tf.height = h;
				tf.usage_bits = RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT;

				sky->radiance = RD::get_singleton()->texture_create(tf, RD::TextureView());

				Sky::Layer layer;
				uint32_t mmw = w;
				uint32_t mmh = h;
				layer.mipmaps.resize(roughness_layers);
				for (int j = 0; j < roughness_layers; j++) {
					Sky::Layer::Mipmap &mm = layer.mipmaps.write[j];
					mm.size.width = mmw;
					mm.size.height = mmh;
					for (int k = 0; k < 6; k++) {
						mm.views[k] = RD::get_singleton()->texture_create_shared_from_slice(RD::TextureView(), sky->radiance, k, j);
						Vector<RID> fbtex;
						fbtex.push_back(mm.views[k]);
						mm.framebuffers[k] = RD::get_singleton()->framebuffer_create(fbtex);
					}

					mmw = MAX(1, mmw >> 1);
					mmh = MAX(1, mmh >> 1);
				}

				sky->layers.push_back(layer);
			}

			sky->radiance_base_cubemap = RD::get_singleton()->texture_create_shared_from_slice(RD::TextureView(), sky->radiance, 0, 0, RD::TEXTURE_SLICE_CUBEMAP);
		}

		RID panorama_texture = storage->texture_get_rd_texture(sky->panorama);

		if (panorama_texture.is_valid()) {
			//is there a panorama texture?

			if (sky_use_cubemap_array) {

				if (sky->mode == VS::SKY_MODE_QUALITY) {
					//render directly to the layers
					for (int i = 0; i < sky->layers.size(); i++) {
						for (int j = 0; j < 6; j++) {
							storage->get_effects()->cubemap_roughness(panorama_texture, true, sky->layers[i].mipmaps[0].framebuffers[j], j, sky_ggx_samples_quality, float(i) / (sky->layers.size() - 1.0));
						}
					}
				} else if (sky->mode == VS::SKY_MODE_REALTIME) {
					//render to first mipmap
					for (int j = 0; j < 6; j++) {
						storage->get_effects()->cubemap_roughness(panorama_texture, true, sky->layers[0].mipmaps[0].framebuffers[j], j, sky_ggx_samples_realtime, 0.0);
					}
					//do the rest in other mipmaps and use cubemap itself as source
					for (int i = 1; i < roughness_layers; i++) {
						//render using a smaller mipmap, then copy to main layer
						for (int j = 0; j < 6; j++) {
							//storage->get_effects()->cubemap_roughness(sky->radiance_base_cubemap, false, sky->layers[0].mipmaps[i].framebuffers[0], j, sky_ggx_samples_realtime, float(i) / (sky->layers.size() - 1.0));
							storage->get_effects()->cubemap_roughness(panorama_texture, true, sky->layers[0].mipmaps[i].framebuffers[0], j, sky_ggx_samples_realtime, float(i) / (sky->layers.size() - 1.0));
							storage->get_effects()->copy(sky->layers[0].mipmaps[i].views[0], sky->layers[i].mipmaps[0].framebuffers[j], Rect2());
						}
					}
				}

				//generate mipmaps

				for (int i = 0; i < sky->layers.size(); i++) {
					for (int j = 0; j < sky->layers[i].mipmaps.size() - 1; j++) {
						for (int k = 0; k < 6; k++) {
							RID view = sky->layers[i].mipmaps[j].views[k];
							RID fb = sky->layers[i].mipmaps[j + 1].framebuffers[k];
							Vector2 size = sky->layers[i].mipmaps[j].size;
							size = Vector2(1.0 / size.x, 1.0 / size.y);
							storage->get_effects()->make_mipmap(view, fb, size);
						}
					}
				}
			} else {

				if (sky->mode == VS::SKY_MODE_QUALITY) {
					//render directly to the layers
					for (int i = 0; i < sky->layers[0].mipmaps.size(); i++) {
						for (int j = 0; j < 6; j++) {
							storage->get_effects()->cubemap_roughness(panorama_texture, true, sky->layers[0].mipmaps[i].framebuffers[j], j, sky_ggx_samples_quality, float(i) / (sky->layers[0].mipmaps.size() - 1.0));
						}
					}
				} else {

					for (int j = 0; j < 6; j++) {
						storage->get_effects()->cubemap_roughness(panorama_texture, true, sky->layers[0].mipmaps[0].framebuffers[j], j, sky_ggx_samples_realtime, 0);
					}

					for (int i = 1; i < sky->layers[0].mipmaps.size(); i++) {
						for (int j = 0; j < 6; j++) {
							storage->get_effects()->cubemap_roughness(sky->radiance_base_cubemap, false, sky->layers[0].mipmaps[i].framebuffers[j], j, sky_ggx_samples_realtime, float(i) / (sky->layers[0].mipmaps.size() - 1.0));
						}
					}
				}
			}
		}

		Sky *next = sky->dirty_list;
		sky->dirty_list = nullptr;
		sky->dirty = false;
		sky = next;
	}

	dirty_sky_list = nullptr;
}

RID RasterizerSceneRD::sky_get_panorama_texture_rd(RID p_sky) const {

	Sky *sky = sky_owner.getornull(p_sky);
	ERR_FAIL_COND_V(!sky, RID());
	if (sky->panorama.is_null()) {
		return RID();
	}

	return storage->texture_get_rd_texture(sky->panorama, true);
}
RID RasterizerSceneRD::sky_get_radiance_texture_rd(RID p_sky) const {
	Sky *sky = sky_owner.getornull(p_sky);
	ERR_FAIL_COND_V(!sky, RID());

	return sky->radiance;
}

RID RasterizerSceneRD::environment_create() {

	return environment_owner.make_rid(Environent());
}

void RasterizerSceneRD::environment_set_background(RID p_env, VS::EnvironmentBG p_bg) {
	Environent *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND(!env);
	env->background = p_bg;
}
void RasterizerSceneRD::environment_set_sky(RID p_env, RID p_sky) {
	Environent *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND(!env);
	env->sky = p_sky;
}
void RasterizerSceneRD::environment_set_sky_custom_fov(RID p_env, float p_scale) {
	Environent *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND(!env);
	env->sky_custom_fov = p_scale;
}
void RasterizerSceneRD::environment_set_sky_orientation(RID p_env, const Basis &p_orientation) {
	Environent *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND(!env);
	env->sky_orientation = p_orientation;
}
void RasterizerSceneRD::environment_set_bg_color(RID p_env, const Color &p_color) {
	Environent *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND(!env);
	env->bg_color = p_color;
}
void RasterizerSceneRD::environment_set_bg_energy(RID p_env, float p_energy) {
	Environent *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND(!env);
	env->bg_energy = p_energy;
}
void RasterizerSceneRD::environment_set_canvas_max_layer(RID p_env, int p_max_layer) {
	Environent *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND(!env);
	env->canvas_max_layer = p_max_layer;
}
void RasterizerSceneRD::environment_set_ambient_light(RID p_env, const Color &p_color, VS::EnvironmentAmbientSource p_ambient, float p_energy, float p_sky_contribution, VS::EnvironmentReflectionSource p_reflection_source) {
	Environent *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND(!env);
	env->ambient_light = p_color;
	env->ambient_source = p_ambient;
	env->ambient_light_energy = p_energy;
	env->ambient_sky_contribution = p_sky_contribution;
	env->reflection_source = p_reflection_source;
}

VS::EnvironmentBG RasterizerSceneRD::environment_get_background(RID p_env) const {
	Environent *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND_V(!env, VS::ENV_BG_MAX);
	return env->background;
}
RID RasterizerSceneRD::environment_get_sky(RID p_env) const {
	Environent *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND_V(!env, RID());
	return env->sky;
}
float RasterizerSceneRD::environment_get_sky_custom_fov(RID p_env) const {
	Environent *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND_V(!env, 0);
	return env->sky_custom_fov;
}
Basis RasterizerSceneRD::environment_get_sky_orientation(RID p_env) const {
	Environent *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND_V(!env, Basis());
	return env->sky_orientation;
}
Color RasterizerSceneRD::environment_get_bg_color(RID p_env) const {
	Environent *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND_V(!env, Color());
	return env->bg_color;
}
float RasterizerSceneRD::environment_get_bg_energy(RID p_env) const {
	Environent *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND_V(!env, 0);
	return env->bg_energy;
}
int RasterizerSceneRD::environment_get_canvas_max_layer(RID p_env) const {
	Environent *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND_V(!env, 0);
	return env->canvas_max_layer;
}
Color RasterizerSceneRD::environment_get_ambient_light_color(RID p_env) const {
	Environent *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND_V(!env, Color());
	return env->ambient_light;
}
VS::EnvironmentAmbientSource RasterizerSceneRD::environment_get_ambient_light_ambient_source(RID p_env) const {
	Environent *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND_V(!env, VS::ENV_AMBIENT_SOURCE_BG);
	return env->ambient_source;
}
float RasterizerSceneRD::environment_get_ambient_light_ambient_energy(RID p_env) const {
	Environent *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND_V(!env, 0);
	return env->ambient_light_energy;
}
float RasterizerSceneRD::environment_get_ambient_sky_contribution(RID p_env) const {
	Environent *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND_V(!env, 0);
	return env->ambient_sky_contribution;
}
VS::EnvironmentReflectionSource RasterizerSceneRD::environment_get_reflection_source(RID p_env) const {
	Environent *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND_V(!env, VS::ENV_REFLECTION_SOURCE_DISABLED);
	return env->reflection_source;
}

void RasterizerSceneRD::environment_set_tonemap(RID p_env, VS::EnvironmentToneMapper p_tone_mapper, float p_exposure, float p_white, bool p_auto_exposure, float p_min_luminance, float p_max_luminance, float p_auto_exp_speed, float p_auto_exp_scale) {
	Environent *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND(!env);
	env->tone_mapper = p_tone_mapper;
	env->auto_exposure = p_auto_exposure;
	env->white = p_white;
	env->min_luminance = p_min_luminance;
	env->max_luminance = p_max_luminance;
	env->auto_exp_speed = p_auto_exp_speed;
	env->auto_exp_scale = p_auto_exp_scale;
}

VS::EnvironmentToneMapper RasterizerSceneRD::environment_get_tonemapper(RID p_env) const {
	Environent *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND_V(!env, VS::ENV_TONE_MAPPER_LINEAR);
	return env->tone_mapper;
}
float RasterizerSceneRD::environment_get_exposure(RID p_env) const {
	Environent *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND_V(!env, 0);
	return env->exposure;
}
float RasterizerSceneRD::environment_get_white(RID p_env) const {
	Environent *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND_V(!env, 0);
	return env->white;
}
bool RasterizerSceneRD::environment_get_auto_exposure(RID p_env) const {
	Environent *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND_V(!env, false);
	return env->auto_exposure;
}
float RasterizerSceneRD::environment_get_min_luminance(RID p_env) const {
	Environent *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND_V(!env, 0);
	return env->min_luminance;
}
float RasterizerSceneRD::environment_get_max_luminance(RID p_env) const {
	Environent *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND_V(!env, 0);
	return env->max_luminance;
}
float RasterizerSceneRD::environment_get_auto_exposure_scale(RID p_env) const {
	Environent *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND_V(!env, 0);
	return env->auto_exp_scale;
}

float RasterizerSceneRD::environment_get_auto_exposure_speed(RID p_env) const {
	Environent *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND_V(!env, 0);
	return env->auto_exp_speed;
}

bool RasterizerSceneRD::is_environment(RID p_env) const {
	return environment_owner.owns(p_env);
}

RID RasterizerSceneRD::render_buffers_create() {
	RenderBuffers rb;
	rb.data = _create_render_buffer_data();
	return render_buffers_owner.make_rid(rb);
}

void RasterizerSceneRD::render_buffers_configure(RID p_render_buffers, RID p_render_target, int p_width, int p_height, VS::ViewportMSAA p_msaa) {

	RenderBuffers *rb = render_buffers_owner.getornull(p_render_buffers);
	rb->width = p_width;
	rb->height = p_height;
	rb->render_target = p_render_target;
	rb->msaa = p_msaa;
	rb->data->configure(p_render_target, p_width, p_height, p_msaa);
}

int RasterizerSceneRD::get_roughness_layers() const {
	return roughness_layers;
}

bool RasterizerSceneRD::is_using_radiance_cubemap_array() const {
	return sky_use_cubemap_array;
}

void RasterizerSceneRD::render_scene(RID p_render_buffers, const Transform &p_cam_transform, const CameraMatrix &p_cam_projection, bool p_cam_ortogonal, InstanceBase **p_cull_result, int p_cull_count, RID *p_light_cull_result, int p_light_cull_count, RID *p_reflection_probe_cull_result, int p_reflection_probe_cull_count, RID p_environment, RID p_shadow_atlas, RID p_reflection_atlas, RID p_reflection_probe, int p_reflection_probe_pass) {

	RenderBuffers *rb = render_buffers_owner.getornull(p_render_buffers);
	ERR_FAIL_COND(!rb && p_render_buffers.is_valid());

	_render_scene(rb->data, p_cam_transform, p_cam_projection, p_cam_ortogonal, p_cull_result, p_cull_count, p_light_cull_result, p_light_cull_count, p_reflection_probe_cull_result, p_reflection_probe_cull_count, p_environment, p_shadow_atlas, p_reflection_atlas, p_reflection_probe, p_reflection_probe_pass);
}

bool RasterizerSceneRD::free(RID p_rid) {

	if (render_buffers_owner.owns(p_rid)) {
		RenderBuffers *rb = render_buffers_owner.getornull(p_rid);
		memdelete(rb->data);
		render_buffers_owner.free(p_rid);
	} else if (environment_owner.owns(p_rid)) {
		//not much to delete, just free it
		environment_owner.free(p_rid);
	} else if (sky_owner.owns(p_rid)) {
		_update_dirty_skys();
		Sky *sky = sky_owner.getornull(p_rid);
		RD::get_singleton()->free(sky->radiance); //free radiance, everything else gets dependency-erased
		sky_owner.free(p_rid);
	} else {
		return false;
	}

	return true;
}

void RasterizerSceneRD::update() {
	_update_dirty_skys();
}

RasterizerSceneRD::RasterizerSceneRD(RasterizerStorageRD *p_storage) {
	storage = p_storage;

	roughness_layers = GLOBAL_GET("rendering/quality/reflections/roughness_layers");
	sky_ggx_samples_quality = GLOBAL_GET("rendering/quality/reflections/ggx_samples");
	sky_ggx_samples_realtime = GLOBAL_GET("rendering/quality/reflections/ggx_samples_realtime");
	sky_use_cubemap_array = GLOBAL_GET("rendering/quality/reflections/texture_array_reflections");
	sky_use_cubemap_array = false;
}
