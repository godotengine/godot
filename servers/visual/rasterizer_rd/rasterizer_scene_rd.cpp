#include "rasterizer_scene_rd.h"
#include "core/os/os.h"
#include "core/project_settings.h"

void RasterizerSceneRD::_clear_reflection_data(ReflectionData &rd) {

	if (rd.radiance.is_valid()) {
		//if size changes, everything must be cleared
		RD::get_singleton()->free(rd.radiance);
		//everything else gets dependency, erase, so just clean it up
		rd.radiance = RID();
		rd.layers.clear();
		rd.radiance_base_cubemap = RID();
	}
}

void RasterizerSceneRD::_update_reflection_data(ReflectionData &rd, int p_size, bool p_quality) {
	//recreate radiance and all data
	int mipmaps = Image::get_image_required_mipmaps(p_size, p_size, Image::FORMAT_RGBAH) + 1;
	if (!p_quality) {
		//use less mipmaps
		mipmaps = MIN(8, mipmaps);
	}

	uint32_t w = p_size, h = p_size;

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

		rd.radiance = RD::get_singleton()->texture_create(tf, RD::TextureView());

		for (int i = 0; i < roughness_layers; i++) {
			ReflectionData::Layer layer;
			uint32_t mmw = w;
			uint32_t mmh = h;
			layer.mipmaps.resize(mipmaps);
			for (int j = 0; j < mipmaps; j++) {
				ReflectionData::Layer::Mipmap &mm = layer.mipmaps.write[j];
				mm.size.width = mmw;
				mm.size.height = mmh;
				for (int k = 0; k < 6; k++) {
					mm.views[k] = RD::get_singleton()->texture_create_shared_from_slice(RD::TextureView(), rd.radiance, i * 6 + k, j);
					Vector<RID> fbtex;
					fbtex.push_back(mm.views[k]);
					mm.framebuffers[k] = RD::get_singleton()->framebuffer_create(fbtex);
				}

				mmw = MAX(1, mmw >> 1);
				mmh = MAX(1, mmh >> 1);
			}

			rd.layers.push_back(layer);
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

		rd.radiance = RD::get_singleton()->texture_create(tf, RD::TextureView());

		ReflectionData::Layer layer;
		uint32_t mmw = w;
		uint32_t mmh = h;
		layer.mipmaps.resize(roughness_layers);
		for (int j = 0; j < roughness_layers; j++) {
			ReflectionData::Layer::Mipmap &mm = layer.mipmaps.write[j];
			mm.size.width = mmw;
			mm.size.height = mmh;
			for (int k = 0; k < 6; k++) {
				mm.views[k] = RD::get_singleton()->texture_create_shared_from_slice(RD::TextureView(), rd.radiance, k, j);
				Vector<RID> fbtex;
				fbtex.push_back(mm.views[k]);
				mm.framebuffers[k] = RD::get_singleton()->framebuffer_create(fbtex);
			}

			mmw = MAX(1, mmw >> 1);
			mmh = MAX(1, mmh >> 1);
		}

		rd.layers.push_back(layer);
	}

	rd.radiance_base_cubemap = RD::get_singleton()->texture_create_shared_from_slice(RD::TextureView(), rd.radiance, 0, 0, RD::TEXTURE_SLICE_CUBEMAP);
}

void RasterizerSceneRD::_create_reflection_from_panorama(ReflectionData &rd, RID p_panorama, bool p_quality) {

	if (sky_use_cubemap_array) {

		if (p_quality) {
			//render directly to the layers
			for (int i = 0; i < rd.layers.size(); i++) {
				for (int j = 0; j < 6; j++) {
					storage->get_effects()->cubemap_roughness(p_panorama, true, rd.layers[i].mipmaps[0].framebuffers[j], j, sky_ggx_samples_quality, float(i) / (rd.layers.size() - 1.0));
				}
			}
		} else {
			//render to first mipmap
			for (int j = 0; j < 6; j++) {
				storage->get_effects()->cubemap_roughness(p_panorama, true, rd.layers[0].mipmaps[0].framebuffers[j], j, sky_ggx_samples_realtime, 0.0);
			}
			//do the rest in other mipmaps and use cubemap itself as source
			for (int i = 1; i < roughness_layers; i++) {
				//render using a smaller mipmap, then copy to main layer
				for (int j = 0; j < 6; j++) {
					//storage->get_effects()->cubemap_roughness(rd.radiance_base_cubemap, false, rd.layers[0].mipmaps[i].framebuffers[0], j, sky_ggx_samples_realtime, float(i) / (rd.layers.size() - 1.0));
					storage->get_effects()->cubemap_roughness(p_panorama, true, rd.layers[0].mipmaps[i].framebuffers[0], j, sky_ggx_samples_realtime, float(i) / (rd.layers.size() - 1.0));
					storage->get_effects()->region_copy(rd.layers[0].mipmaps[i].views[0], rd.layers[i].mipmaps[0].framebuffers[j], Rect2());
				}
			}
		}
	} else {

		if (p_quality) {
			//render directly to the layers
			for (int i = 0; i < rd.layers[0].mipmaps.size(); i++) {
				for (int j = 0; j < 6; j++) {
					storage->get_effects()->cubemap_roughness(p_panorama, true, rd.layers[0].mipmaps[i].framebuffers[j], j, sky_ggx_samples_quality, float(i) / (rd.layers[0].mipmaps.size() - 1.0));
				}
			}
		} else {

			for (int j = 0; j < 6; j++) {
				storage->get_effects()->cubemap_roughness(p_panorama, true, rd.layers[0].mipmaps[0].framebuffers[j], j, sky_ggx_samples_realtime, 0);
			}

			for (int i = 1; i < rd.layers[0].mipmaps.size(); i++) {
				for (int j = 0; j < 6; j++) {
					storage->get_effects()->cubemap_roughness(rd.radiance_base_cubemap, false, rd.layers[0].mipmaps[i].framebuffers[j], j, sky_ggx_samples_realtime, float(i) / (rd.layers[0].mipmaps.size() - 1.0));
				}
			}
		}
	}
}

void RasterizerSceneRD::_create_reflection_from_base_mipmap(ReflectionData &rd, bool p_quality, int p_cube_side) {

	if (sky_use_cubemap_array) {

		if (p_quality) {
			//render directly to the layers
			for (int i = 1; i < rd.layers.size(); i++) {
				storage->get_effects()->cubemap_roughness(rd.radiance_base_cubemap, false, rd.layers[i].mipmaps[0].framebuffers[p_cube_side], p_cube_side, sky_ggx_samples_quality, float(i) / (rd.layers.size() - 1.0));
			}
		} else {
			//do the rest in other mipmaps and use cubemap itself as source
			for (int i = 1; i < roughness_layers; i++) {
				//render using a smaller mipmap, then copy to main layer
				storage->get_effects()->cubemap_roughness(rd.radiance_base_cubemap, false, rd.layers[0].mipmaps[i].framebuffers[0], p_cube_side, sky_ggx_samples_realtime, float(i) / (rd.layers.size() - 1.0));
				storage->get_effects()->region_copy(rd.layers[0].mipmaps[i].views[0], rd.layers[i].mipmaps[0].framebuffers[p_cube_side], Rect2());
			}
		}
	} else {

		if (p_quality) {
			//render directly to the layers
			for (int i = 1; i < rd.layers[0].mipmaps.size(); i++) {
				storage->get_effects()->cubemap_roughness(rd.radiance_base_cubemap, false, rd.layers[0].mipmaps[i].framebuffers[p_cube_side], p_cube_side, sky_ggx_samples_quality, float(i) / (rd.layers[0].mipmaps.size() - 1.0));
			}
		} else {

			for (int i = 1; i < rd.layers[0].mipmaps.size(); i++) {
				storage->get_effects()->cubemap_roughness(rd.radiance_base_cubemap, false, rd.layers[0].mipmaps[i].framebuffers[p_cube_side], p_cube_side, sky_ggx_samples_realtime, float(i) / (rd.layers[0].mipmaps.size() - 1.0));
			}
		}
	}
}

void RasterizerSceneRD::_update_reflection_mipmaps(ReflectionData &rd, bool p_quality) {

	if (sky_use_cubemap_array) {

		for (int i = 0; i < rd.layers.size(); i++) {
			for (int j = 0; j < rd.layers[i].mipmaps.size() - 1; j++) {
				for (int k = 0; k < 6; k++) {
					RID view = rd.layers[i].mipmaps[j].views[k];
					RID fb = rd.layers[i].mipmaps[j + 1].framebuffers[k];
					Vector2 size = rd.layers[i].mipmaps[j].size;
					size = Vector2(1.0 / size.x, 1.0 / size.y);
					storage->get_effects()->make_mipmap(view, fb, size);
				}
			}
		}
	}
}

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
	_clear_reflection_data(sky->reflection);
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
		_clear_reflection_data(sky->reflection);
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

		if (sky->reflection.radiance.is_null()) {
			_update_reflection_data(sky->reflection, sky->radiance_size, sky->mode == VS::SKY_MODE_QUALITY);
		}

		RID panorama_texture = storage->texture_get_rd_texture(sky->panorama);

		if (panorama_texture.is_valid()) {
			//is there a panorama texture?
			_create_reflection_from_panorama(sky->reflection, panorama_texture, sky->mode == VS::SKY_MODE_QUALITY);
			_update_reflection_mipmaps(sky->reflection, sky->mode == VS::SKY_MODE_QUALITY);
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

	return sky->reflection.radiance;
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

////////////////////////////////////////////////////////////

RID RasterizerSceneRD::reflection_probe_instance_create(RID p_probe) {
	ReflectionProbeInstance rpi;
	rpi.probe = p_probe;
	return reflection_probe_instance_owner.make_rid(rpi);
}

void RasterizerSceneRD::reflection_probe_instance_set_transform(RID p_instance, const Transform &p_transform) {
	ReflectionProbeInstance *rpi = reflection_probe_instance_owner.getornull(p_instance);
	ERR_FAIL_COND(!rpi);

	rpi->transform = p_transform;
	rpi->dirty = true;
}

bool RasterizerSceneRD::reflection_probe_instance_needs_redraw(RID p_instance) {

	ReflectionProbeInstance *rpi = reflection_probe_instance_owner.getornull(p_instance);
	ERR_FAIL_COND_V(!rpi, false);

	if (rpi->rendering) {
		return false;
	}

	if (rpi->dirty) {
		return true;
	}

	if (rpi->current_resolution != storage->reflection_probe_get_resolution(rpi->probe)) {
		return true;
	}

	if (storage->reflection_probe_get_update_mode(rpi->probe) == VS::REFLECTION_PROBE_UPDATE_ALWAYS) {
		return true;
	}

	return false;
}

void RasterizerSceneRD::reflection_probe_instance_begin_render(RID p_instance) {

	ReflectionProbeInstance *rpi = reflection_probe_instance_owner.getornull(p_instance);
	ERR_FAIL_COND(!rpi);
	rpi->rendering = true;
	rpi->processing_side = 0;

	int probe_resolution = storage->reflection_probe_get_resolution(rpi->probe);
	if (rpi->current_resolution != probe_resolution) {
		//need to re-create everything
		_clear_reflection_data(rpi->reflection);
		_update_reflection_data(rpi->reflection, probe_resolution, storage->reflection_probe_get_update_mode(rpi->probe) == VS::REFLECTION_PROBE_UPDATE_ONCE);

		rpi->current_resolution = probe_resolution;

		if (rpi->depth_buffer.is_valid()) {
			RD::get_singleton()->free(rpi->depth_buffer);
		}
		{
			RD::TextureFormat tf;
			tf.format = RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_D24_UNORM_S8_UINT, RD::TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) ? RD::DATA_FORMAT_D24_UNORM_S8_UINT : RD::DATA_FORMAT_D32_SFLOAT_S8_UINT;
			tf.width = probe_resolution;
			tf.height = probe_resolution;
			tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

			rpi->depth_buffer = RD::get_singleton()->texture_create(tf, RD::TextureView());
		}

		for (int i = 0; i < 6; i++) {
			Vector<RID> fb;
			fb.push_back(rpi->reflection.layers[0].mipmaps[0].views[i]);
			fb.push_back(rpi->depth_buffer);
			rpi->render_fb[i] = RD::get_singleton()->framebuffer_create(fb);
		}
	}
}

bool RasterizerSceneRD::reflection_probe_instance_postprocess_step(RID p_instance) {

	ReflectionProbeInstance *rpi = reflection_probe_instance_owner.getornull(p_instance);
	ERR_FAIL_COND_V(!rpi, false);
	ERR_FAIL_COND_V(!rpi->rendering, false);

	_create_reflection_from_base_mipmap(rpi->reflection, storage->reflection_probe_get_update_mode(rpi->probe) == VS::REFLECTION_PROBE_UPDATE_ONCE, rpi->processing_side);

	rpi->processing_side++;

	if (rpi->processing_side == 6) {
		rpi->rendering = false;
		rpi->processing_side = 0;

		return true;
	} else {
		return false;
	}
}

uint32_t RasterizerSceneRD::reflection_probe_instance_get_resolution(RID p_instance) {
	ReflectionProbeInstance *rpi = reflection_probe_instance_owner.getornull(p_instance);
	ERR_FAIL_COND_V(!rpi, 0);

	return rpi->current_resolution;
}

RID RasterizerSceneRD::reflection_probe_instance_get_framebuffer(RID p_instance, int p_index) {
	ReflectionProbeInstance *rpi = reflection_probe_instance_owner.getornull(p_instance);
	ERR_FAIL_COND_V(!rpi, RID());
	ERR_FAIL_INDEX_V(p_index, 6, RID());

	return rpi->render_fb[p_index];
}

///////////////////////////////////////////////////////////

RID RasterizerSceneRD::shadow_atlas_create() {

	return shadow_atlas_owner.make_rid(ShadowAtlas());
}

void RasterizerSceneRD::shadow_atlas_set_size(RID p_atlas, int p_size) {

	ShadowAtlas *shadow_atlas = shadow_atlas_owner.getornull(p_atlas);
	ERR_FAIL_COND(!shadow_atlas);
	ERR_FAIL_COND(p_size < 0);

	p_size = next_power_of_2(p_size);

	if (p_size == shadow_atlas->size)
		return;

	// erasing atlas
	if (shadow_atlas->depth.is_valid()) {
		RD::get_singleton()->free(shadow_atlas->depth);
		shadow_atlas->depth = RID();
		shadow_atlas->fb = RID();
	}
	for (int i = 0; i < 4; i++) {
		//clear subdivisions
		shadow_atlas->quadrants[i].shadows.resize(0);
		shadow_atlas->quadrants[i].shadows.resize(1 << shadow_atlas->quadrants[i].subdivision);
	}

	//erase shadow atlas reference from lights
	for (Map<RID, uint32_t>::Element *E = shadow_atlas->shadow_owners.front(); E; E = E->next()) {
		LightInstance *li = light_instance_owner.getornull(E->key());
		ERR_CONTINUE(!li);
		li->shadow_atlases.erase(p_atlas);
	}

	//clear owners
	shadow_atlas->shadow_owners.clear();

	shadow_atlas->size = p_size;

	if (shadow_atlas->size) {

		RD::TextureFormat tf;
		tf.format = RD::DATA_FORMAT_R32_SFLOAT;
		tf.width = shadow_atlas->size;
		tf.height = shadow_atlas->size;
		tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT;

		shadow_atlas->depth = RD::get_singleton()->texture_create(tf, RD::TextureView());

		Vector<RID> fb;
		fb.push_back(shadow_atlas->depth);
		shadow_atlas->fb = RD::get_singleton()->framebuffer_create(fb);
	}
}

void RasterizerSceneRD::shadow_atlas_set_quadrant_subdivision(RID p_atlas, int p_quadrant, int p_subdivision) {

	ShadowAtlas *shadow_atlas = shadow_atlas_owner.getornull(p_atlas);
	ERR_FAIL_COND(!shadow_atlas);
	ERR_FAIL_INDEX(p_quadrant, 4);
	ERR_FAIL_INDEX(p_subdivision, 16384);

	uint32_t subdiv = next_power_of_2(p_subdivision);
	if (subdiv & 0xaaaaaaaa) { //sqrt(subdiv) must be integer
		subdiv <<= 1;
	}

	subdiv = int(Math::sqrt((float)subdiv));

	//obtain the number that will be x*x

	if (shadow_atlas->quadrants[p_quadrant].subdivision == subdiv)
		return;

	//erase all data from quadrant
	for (int i = 0; i < shadow_atlas->quadrants[p_quadrant].shadows.size(); i++) {

		if (shadow_atlas->quadrants[p_quadrant].shadows[i].owner.is_valid()) {
			shadow_atlas->shadow_owners.erase(shadow_atlas->quadrants[p_quadrant].shadows[i].owner);
			LightInstance *li = light_instance_owner.getornull(shadow_atlas->quadrants[p_quadrant].shadows[i].owner);
			ERR_CONTINUE(!li);
			li->shadow_atlases.erase(p_atlas);
		}
	}

	shadow_atlas->quadrants[p_quadrant].shadows.resize(0);
	shadow_atlas->quadrants[p_quadrant].shadows.resize(subdiv * subdiv);
	shadow_atlas->quadrants[p_quadrant].subdivision = subdiv;

	//cache the smallest subdiv (for faster allocation in light update)

	shadow_atlas->smallest_subdiv = 1 << 30;

	for (int i = 0; i < 4; i++) {
		if (shadow_atlas->quadrants[i].subdivision) {
			shadow_atlas->smallest_subdiv = MIN(shadow_atlas->smallest_subdiv, shadow_atlas->quadrants[i].subdivision);
		}
	}

	if (shadow_atlas->smallest_subdiv == 1 << 30) {
		shadow_atlas->smallest_subdiv = 0;
	}

	//resort the size orders, simple bublesort for 4 elements..

	int swaps = 0;
	do {
		swaps = 0;

		for (int i = 0; i < 3; i++) {
			if (shadow_atlas->quadrants[shadow_atlas->size_order[i]].subdivision < shadow_atlas->quadrants[shadow_atlas->size_order[i + 1]].subdivision) {
				SWAP(shadow_atlas->size_order[i], shadow_atlas->size_order[i + 1]);
				swaps++;
			}
		}
	} while (swaps > 0);
}

bool RasterizerSceneRD::_shadow_atlas_find_shadow(ShadowAtlas *shadow_atlas, int *p_in_quadrants, int p_quadrant_count, int p_current_subdiv, uint64_t p_tick, int &r_quadrant, int &r_shadow) {

	for (int i = p_quadrant_count - 1; i >= 0; i--) {

		int qidx = p_in_quadrants[i];

		if (shadow_atlas->quadrants[qidx].subdivision == (uint32_t)p_current_subdiv) {
			return false;
		}

		//look for an empty space
		int sc = shadow_atlas->quadrants[qidx].shadows.size();
		ShadowAtlas::Quadrant::Shadow *sarr = shadow_atlas->quadrants[qidx].shadows.ptrw();

		int found_free_idx = -1; //found a free one
		int found_used_idx = -1; //found existing one, must steal it
		uint64_t min_pass = 0; // pass of the existing one, try to use the least recently used one (LRU fashion)

		for (int j = 0; j < sc; j++) {
			if (!sarr[j].owner.is_valid()) {
				found_free_idx = j;
				break;
			}

			LightInstance *sli = light_instance_owner.getornull(sarr[j].owner);
			ERR_CONTINUE(!sli);

			if (sli->last_scene_pass != scene_pass) {

				//was just allocated, don't kill it so soon, wait a bit..
				if (p_tick - sarr[j].alloc_tick < shadow_atlas_realloc_tolerance_msec)
					continue;

				if (found_used_idx == -1 || sli->last_scene_pass < min_pass) {
					found_used_idx = j;
					min_pass = sli->last_scene_pass;
				}
			}
		}

		if (found_free_idx == -1 && found_used_idx == -1)
			continue; //nothing found

		if (found_free_idx == -1 && found_used_idx != -1) {
			found_free_idx = found_used_idx;
		}

		r_quadrant = qidx;
		r_shadow = found_free_idx;

		return true;
	}

	return false;
}

bool RasterizerSceneRD::shadow_atlas_update_light(RID p_atlas, RID p_light_intance, float p_coverage, uint64_t p_light_version) {

	ShadowAtlas *shadow_atlas = shadow_atlas_owner.getornull(p_atlas);
	ERR_FAIL_COND_V(!shadow_atlas, false);

	LightInstance *li = light_instance_owner.getornull(p_light_intance);
	ERR_FAIL_COND_V(!li, false);

	if (shadow_atlas->size == 0 || shadow_atlas->smallest_subdiv == 0) {
		return false;
	}

	uint32_t quad_size = shadow_atlas->size >> 1;
	int desired_fit = MIN(quad_size / shadow_atlas->smallest_subdiv, next_power_of_2(quad_size * p_coverage));

	int valid_quadrants[4];
	int valid_quadrant_count = 0;
	int best_size = -1; //best size found
	int best_subdiv = -1; //subdiv for the best size

	//find the quadrants this fits into, and the best possible size it can fit into
	for (int i = 0; i < 4; i++) {
		int q = shadow_atlas->size_order[i];
		int sd = shadow_atlas->quadrants[q].subdivision;
		if (sd == 0)
			continue; //unused

		int max_fit = quad_size / sd;

		if (best_size != -1 && max_fit > best_size)
			break; //too large

		valid_quadrants[valid_quadrant_count++] = q;
		best_subdiv = sd;

		if (max_fit >= desired_fit) {
			best_size = max_fit;
		}
	}

	ERR_FAIL_COND_V(valid_quadrant_count == 0, false);

	uint64_t tick = OS::get_singleton()->get_ticks_msec();

	//see if it already exists

	if (shadow_atlas->shadow_owners.has(p_light_intance)) {
		//it does!
		uint32_t key = shadow_atlas->shadow_owners[p_light_intance];
		uint32_t q = (key >> ShadowAtlas::QUADRANT_SHIFT) & 0x3;
		uint32_t s = key & ShadowAtlas::SHADOW_INDEX_MASK;

		bool should_realloc = shadow_atlas->quadrants[q].subdivision != (uint32_t)best_subdiv && (shadow_atlas->quadrants[q].shadows[s].alloc_tick - tick > shadow_atlas_realloc_tolerance_msec);
		bool should_redraw = shadow_atlas->quadrants[q].shadows[s].version != p_light_version;

		if (!should_realloc) {
			shadow_atlas->quadrants[q].shadows.write[s].version = p_light_version;
			//already existing, see if it should redraw or it's just OK
			return should_redraw;
		}

		int new_quadrant, new_shadow;

		//find a better place
		if (_shadow_atlas_find_shadow(shadow_atlas, valid_quadrants, valid_quadrant_count, shadow_atlas->quadrants[q].subdivision, tick, new_quadrant, new_shadow)) {
			//found a better place!
			ShadowAtlas::Quadrant::Shadow *sh = &shadow_atlas->quadrants[new_quadrant].shadows.write[new_shadow];
			if (sh->owner.is_valid()) {
				//is taken, but is invalid, erasing it
				shadow_atlas->shadow_owners.erase(sh->owner);
				LightInstance *sli = light_instance_owner.getornull(sh->owner);
				sli->shadow_atlases.erase(p_atlas);
			}

			//erase previous
			shadow_atlas->quadrants[q].shadows.write[s].version = 0;
			shadow_atlas->quadrants[q].shadows.write[s].owner = RID();

			sh->owner = p_light_intance;
			sh->alloc_tick = tick;
			sh->version = p_light_version;
			li->shadow_atlases.insert(p_atlas);

			//make new key
			key = new_quadrant << ShadowAtlas::QUADRANT_SHIFT;
			key |= new_shadow;
			//update it in map
			shadow_atlas->shadow_owners[p_light_intance] = key;
			//make it dirty, as it should redraw anyway
			return true;
		}

		//no better place for this shadow found, keep current

		//already existing, see if it should redraw or it's just OK

		shadow_atlas->quadrants[q].shadows.write[s].version = p_light_version;

		return should_redraw;
	}

	int new_quadrant, new_shadow;

	//find a better place
	if (_shadow_atlas_find_shadow(shadow_atlas, valid_quadrants, valid_quadrant_count, -1, tick, new_quadrant, new_shadow)) {
		//found a better place!
		ShadowAtlas::Quadrant::Shadow *sh = &shadow_atlas->quadrants[new_quadrant].shadows.write[new_shadow];
		if (sh->owner.is_valid()) {
			//is taken, but is invalid, erasing it
			shadow_atlas->shadow_owners.erase(sh->owner);
			LightInstance *sli = light_instance_owner.getornull(sh->owner);
			sli->shadow_atlases.erase(p_atlas);
		}

		sh->owner = p_light_intance;
		sh->alloc_tick = tick;
		sh->version = p_light_version;
		li->shadow_atlases.insert(p_atlas);

		//make new key
		uint32_t key = new_quadrant << ShadowAtlas::QUADRANT_SHIFT;
		key |= new_shadow;
		//update it in map
		shadow_atlas->shadow_owners[p_light_intance] = key;
		//make it dirty, as it should redraw anyway

		return true;
	}

	//no place to allocate this light, apologies

	return false;
}

void RasterizerSceneRD::directional_shadow_atlas_set_size(int p_size) {

	p_size = nearest_power_of_2_templated(p_size);

	if (directional_shadow.size == p_size) {
		return;
	}

	if (directional_shadow.depth.is_valid()) {
		RD::get_singleton()->free(directional_shadow.depth);
		directional_shadow.depth = RID();
		directional_shadow.fb = RID();
	}

	if (p_size > 0) {

		RD::TextureFormat tf;
		tf.format = RD::DATA_FORMAT_R32_SFLOAT;
		tf.width = p_size;
		tf.height = p_size;
		tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT;

		directional_shadow.depth = RD::get_singleton()->texture_create(tf, RD::TextureView());
		Vector<RID> fb;
		fb.push_back(directional_shadow.depth);
		directional_shadow.fb = RD::get_singleton()->framebuffer_create(fb);
	}
}

void RasterizerSceneRD::set_directional_shadow_count(int p_count) {

	directional_shadow.light_count = p_count;
	directional_shadow.current_light = 0;
}

int RasterizerSceneRD::get_directional_light_shadow_size(RID p_light_intance) {

	ERR_FAIL_COND_V(directional_shadow.light_count == 0, 0);

	int shadow_size;

	if (directional_shadow.light_count == 1) {
		shadow_size = directional_shadow.size;
	} else {
		shadow_size = directional_shadow.size / 2; //more than 4 not supported anyway
	}

	LightInstance *light_instance = light_instance_owner.getornull(p_light_intance);
	ERR_FAIL_COND_V(!light_instance, 0);

	switch (storage->light_directional_get_shadow_mode(light_instance->light)) {
		case VS::LIGHT_DIRECTIONAL_SHADOW_ORTHOGONAL:
			break; //none
		case VS::LIGHT_DIRECTIONAL_SHADOW_PARALLEL_2_SPLITS:
		case VS::LIGHT_DIRECTIONAL_SHADOW_PARALLEL_4_SPLITS: shadow_size /= 2; break;
	}

	return shadow_size;
}

//////////////////////////////////////////////////

RID RasterizerSceneRD::light_instance_create(RID p_light) {

	RID li = light_instance_owner.make_rid(LightInstance());

	LightInstance *light_instance = light_instance_owner.getornull(li);

	light_instance->self = li;
	light_instance->light = p_light;
	light_instance->light_type = storage->light_get_type(p_light);

	return li;
}

void RasterizerSceneRD::light_instance_set_transform(RID p_light_instance, const Transform &p_transform) {

	LightInstance *light_instance = light_instance_owner.getornull(p_light_instance);
	ERR_FAIL_COND(!light_instance);

	light_instance->transform = p_transform;
}

void RasterizerSceneRD::light_instance_set_shadow_transform(RID p_light_instance, const CameraMatrix &p_projection, const Transform &p_transform, float p_far, float p_split, int p_pass, float p_bias_scale) {

	LightInstance *light_instance = light_instance_owner.getornull(p_light_instance);
	ERR_FAIL_COND(!light_instance);

	if (storage->light_get_type(light_instance->light) != VS::LIGHT_DIRECTIONAL) {
		p_pass = 0;
	}

	ERR_FAIL_INDEX(p_pass, 4);

	light_instance->shadow_transform[p_pass].camera = p_projection;
	light_instance->shadow_transform[p_pass].transform = p_transform;
	light_instance->shadow_transform[p_pass].farplane = p_far;
	light_instance->shadow_transform[p_pass].split = p_split;
	light_instance->shadow_transform[p_pass].bias_scale = p_bias_scale;
}

void RasterizerSceneRD::light_instance_mark_visible(RID p_light_instance) {

	LightInstance *light_instance = light_instance_owner.getornull(p_light_instance);
	ERR_FAIL_COND(!light_instance);

	light_instance->last_scene_pass = scene_pass;
}

RasterizerSceneRD::ShadowCubemap *RasterizerSceneRD::_get_shadow_cubemap(int p_size) {

	if (!shadow_cubemaps.has(p_size)) {

		ShadowCubemap sc;
		{
			RD::TextureFormat tf;
			tf.format = RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_D32_SFLOAT, RD::TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) ? RD::DATA_FORMAT_D32_SFLOAT : RD::DATA_FORMAT_X8_D24_UNORM_PACK32;
			tf.width = p_size;
			tf.height = p_size;
			tf.type = RD::TEXTURE_TYPE_CUBE;
			tf.array_layers = 6;
			tf.usage_bits = RD::TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT;
			sc.cubemap = RD::get_singleton()->texture_create(tf, RD::TextureView());
		}

		for (int i = 0; i < 6; i++) {
			RID side_texture = RD::get_singleton()->texture_create_shared_from_slice(RD::TextureView(), sc.cubemap, i, 0);
			Vector<RID> fbtex;
			fbtex.push_back(side_texture);
			sc.side_fb[i] = RD::get_singleton()->framebuffer_create(fbtex);
		}

		shadow_cubemaps[p_size] = sc;
	}

	return &shadow_cubemaps[p_size];
}

RasterizerSceneRD::ShadowMap *RasterizerSceneRD::_get_shadow_map(const Size2i &p_size) {

	if (!shadow_maps.has(p_size)) {

		ShadowMap sm;
		{
			RD::TextureFormat tf;
			tf.format = RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_D32_SFLOAT, RD::TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) ? RD::DATA_FORMAT_D32_SFLOAT : RD::DATA_FORMAT_X8_D24_UNORM_PACK32;
			tf.width = p_size.width;
			tf.height = p_size.height;
			tf.usage_bits = RD::TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT;

			sm.depth = RD::get_singleton()->texture_create(tf, RD::TextureView());
		}

		Vector<RID> fbtex;
		fbtex.push_back(sm.depth);
		sm.fb = RD::get_singleton()->framebuffer_create(fbtex);

		shadow_maps[p_size] = sm;
	}

	return &shadow_maps[p_size];
}

////////////////////////////////
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

void RasterizerSceneRD::render_scene(RID p_render_buffers, const Transform &p_cam_transform, const CameraMatrix &p_cam_projection, bool p_cam_ortogonal, InstanceBase **p_cull_result, int p_cull_count, RID *p_light_cull_result, int p_light_cull_count, RID *p_reflection_probe_cull_result, int p_reflection_probe_cull_count, RID p_environment, RID p_shadow_atlas, RID p_reflection_probe, int p_reflection_probe_pass) {

	RenderBuffers *rb = render_buffers_owner.getornull(p_render_buffers);
	ERR_FAIL_COND(!rb && p_render_buffers.is_valid());

	_render_scene(rb->data, p_cam_transform, p_cam_projection, p_cam_ortogonal, p_cull_result, p_cull_count, p_light_cull_result, p_light_cull_count, p_reflection_probe_cull_result, p_reflection_probe_cull_count, p_environment, p_shadow_atlas, p_reflection_probe, p_reflection_probe_pass);
}

void RasterizerSceneRD::render_shadow(RID p_light, RID p_shadow_atlas, int p_pass, InstanceBase **p_cull_result, int p_cull_count) {

	LightInstance *light_instance = light_instance_owner.getornull(p_light);
	ERR_FAIL_COND(!light_instance);

	Rect2i atlas_rect;
	RID atlas_fb;
	int atlas_fb_size;

	bool using_dual_paraboloid = false;
	bool using_dual_paraboloid_flip = false;
	float zfar = 0;
	RID render_fb;
	RID render_texture;
	float bias = 0;
	float normal_bias = 0;

	bool render_cubemap = false;
	bool finalize_cubemap = false;

	CameraMatrix light_projection;
	Transform light_transform;

	if (storage->light_get_type(light_instance->light) == VS::LIGHT_DIRECTIONAL) {
		//set pssm stuff
		if (light_instance->last_scene_shadow_pass != scene_pass) {
			//assign rect if unassigned
			light_instance->light_directional_index = directional_shadow.current_light;
			light_instance->last_scene_shadow_pass = scene_pass;
			directional_shadow.current_light++;

			if (directional_shadow.light_count == 1) {
				light_instance->directional_rect = Rect2(0, 0, directional_shadow.size, directional_shadow.size);
			} else if (directional_shadow.light_count == 2) {
				light_instance->directional_rect = Rect2(0, 0, directional_shadow.size, directional_shadow.size / 2);
				if (light_instance->light_directional_index == 1) {
					light_instance->directional_rect.position.x += light_instance->directional_rect.size.x;
				}
			} else { //3 and 4
				light_instance->directional_rect = Rect2(0, 0, directional_shadow.size / 2, directional_shadow.size / 2);
				if (light_instance->light_directional_index & 1) {
					light_instance->directional_rect.position.x += light_instance->directional_rect.size.x;
				}
				if (light_instance->light_directional_index / 2) {
					light_instance->directional_rect.position.y += light_instance->directional_rect.size.y;
				}
			}
		}

		light_projection = light_instance->shadow_transform[p_pass].camera;
		light_transform = light_instance->shadow_transform[p_pass].transform;

		atlas_rect.position.x = light_instance->directional_rect.position.x;
		atlas_rect.position.y = light_instance->directional_rect.position.y;
		atlas_rect.size.width = light_instance->directional_rect.size.x;
		atlas_rect.size.height = light_instance->directional_rect.size.y;

		if (storage->light_directional_get_shadow_mode(light_instance->light) == VS::LIGHT_DIRECTIONAL_SHADOW_PARALLEL_4_SPLITS) {

			atlas_rect.size.width /= 2;
			atlas_rect.size.height /= 2;

			if (p_pass == 1) {
				atlas_rect.position.x += atlas_rect.size.width;
			} else if (p_pass == 2) {
				atlas_rect.position.y += atlas_rect.size.height;
			} else if (p_pass == 3) {
				atlas_rect.position.x += atlas_rect.size.width;
				atlas_rect.position.y += atlas_rect.size.height;
			}

		} else if (storage->light_directional_get_shadow_mode(light_instance->light) == VS::LIGHT_DIRECTIONAL_SHADOW_PARALLEL_2_SPLITS) {

			atlas_rect.size.height /= 2;

			if (p_pass == 0) {

			} else {
				atlas_rect.position.y += atlas_rect.size.height;
			}
		}

		float bias_mult = Math::lerp(1.0f, light_instance->shadow_transform[p_pass].bias_scale, storage->light_get_param(light_instance->light, VS::LIGHT_PARAM_SHADOW_BIAS_SPLIT_SCALE));
		zfar = storage->light_get_param(light_instance->light, VS::LIGHT_PARAM_RANGE);
		bias = storage->light_get_param(light_instance->light, VS::LIGHT_PARAM_SHADOW_BIAS) * bias_mult;
		normal_bias = storage->light_get_param(light_instance->light, VS::LIGHT_PARAM_SHADOW_NORMAL_BIAS) * bias_mult;

		ShadowMap *shadow_map = _get_shadow_map(atlas_rect.size);
		render_fb = shadow_map->fb;
		render_texture = shadow_map->depth;
		atlas_fb = directional_shadow.fb;
		atlas_fb_size = directional_shadow.size;

	} else {
		//set from shadow atlas

		ShadowAtlas *shadow_atlas = shadow_atlas_owner.getornull(p_shadow_atlas);
		ERR_FAIL_COND(!shadow_atlas);
		ERR_FAIL_COND(!shadow_atlas->shadow_owners.has(p_light));

		uint32_t key = shadow_atlas->shadow_owners[p_light];

		uint32_t quadrant = (key >> ShadowAtlas::QUADRANT_SHIFT) & 0x3;
		uint32_t shadow = key & ShadowAtlas::SHADOW_INDEX_MASK;

		ERR_FAIL_INDEX((int)shadow, shadow_atlas->quadrants[quadrant].shadows.size());

		uint32_t quadrant_size = shadow_atlas->size >> 1;

		atlas_rect.position.x = (quadrant & 1) * quadrant_size;
		atlas_rect.position.y = (quadrant >> 1) * quadrant_size;

		uint32_t shadow_size = (quadrant_size / shadow_atlas->quadrants[quadrant].subdivision);
		atlas_rect.position.x += (shadow % shadow_atlas->quadrants[quadrant].subdivision) * shadow_size;
		atlas_rect.position.y += (shadow / shadow_atlas->quadrants[quadrant].subdivision) * shadow_size;

		atlas_rect.size.width = shadow_size;
		atlas_rect.size.height = shadow_size;
		atlas_fb = shadow_atlas->fb;
		atlas_fb_size = shadow_atlas->size;

		zfar = storage->light_get_param(light_instance->light, VS::LIGHT_PARAM_RANGE);
		bias = storage->light_get_param(light_instance->light, VS::LIGHT_PARAM_SHADOW_BIAS);
		normal_bias = storage->light_get_param(light_instance->light, VS::LIGHT_PARAM_SHADOW_NORMAL_BIAS);

		if (storage->light_get_type(light_instance->light) == VS::LIGHT_OMNI) {

			if (storage->light_omni_get_shadow_mode(light_instance->light) == VS::LIGHT_OMNI_SHADOW_CUBE) {

				ShadowCubemap *cubemap = _get_shadow_cubemap(shadow_size / 2);

				render_fb = cubemap->side_fb[p_pass];
				render_texture = cubemap->cubemap;

				light_projection = light_instance->shadow_transform[0].camera;
				light_transform = light_instance->shadow_transform[0].transform;
				render_cubemap = true;
				finalize_cubemap = p_pass == 5;

			} else {

				light_projection = light_instance->shadow_transform[0].camera;
				light_transform = light_instance->shadow_transform[0].transform;

				atlas_rect.size.height /= 2;
				atlas_rect.position.y += p_pass * atlas_rect.size.height;

				using_dual_paraboloid = true;
				using_dual_paraboloid_flip = p_pass == 1;

				ShadowMap *shadow_map = _get_shadow_map(atlas_rect.size);
				render_fb = shadow_map->fb;
				render_texture = shadow_map->depth;
			}

		} else if (storage->light_get_type(light_instance->light) == VS::LIGHT_SPOT) {

			light_projection = light_instance->shadow_transform[0].camera;
			light_transform = light_instance->shadow_transform[0].transform;

			ShadowMap *shadow_map = _get_shadow_map(atlas_rect.size);
			render_fb = shadow_map->fb;
			render_texture = shadow_map->depth;
		}
	}

	if (render_cubemap) {
		//rendering to cubemap
		_render_shadow(render_fb, p_cull_result, p_cull_count, light_projection, light_transform, zfar, 0, 0, false, false);
		if (finalize_cubemap) {
			//reblit
			atlas_rect.size.height /= 2;
			storage->get_effects()->copy_cubemap_to_dp(render_texture, atlas_fb, atlas_rect, light_projection.get_z_near(), light_projection.get_z_far(), bias, false);
			atlas_rect.position.y += atlas_rect.size.height;
			storage->get_effects()->copy_cubemap_to_dp(render_texture, atlas_fb, atlas_rect, light_projection.get_z_near(), light_projection.get_z_far(), bias, true);
		}
	} else {
		//render shadow
		_render_shadow(render_fb, p_cull_result, p_cull_count, light_projection, light_transform, zfar, bias, normal_bias, using_dual_paraboloid, using_dual_paraboloid_flip);

		//copy to atlas
		storage->get_effects()->copy_to_rect(render_texture, atlas_fb, atlas_rect, true);

		//does not work from depth to color
		//RD::get_singleton()->texture_copy(render_texture, atlas_texture, Vector3(0, 0, 0), Vector3(atlas_rect.position.x, atlas_rect.position.y, 0), Vector3(atlas_rect.size.x, atlas_rect.size.y, 1), 0, 0, 0, 0, true);
	}
}

bool RasterizerSceneRD::free(RID p_rid) {

	if (render_buffers_owner.owns(p_rid)) {
		RenderBuffers *rb = render_buffers_owner.getornull(p_rid);
		memdelete(rb->data);
		render_buffers_owner.free(p_rid);
	} else if (environment_owner.owns(p_rid)) {
		//not much to delete, just free it
		environment_owner.free(p_rid);
	} else if (reflection_probe_instance_owner.owns(p_rid)) {
		//not much to delete, just free it
		ReflectionProbeInstance *rpi = reflection_probe_instance_owner.getornull(p_rid);
		_clear_reflection_data(rpi->reflection);
		reflection_probe_instance_owner.free(p_rid);
	} else if (sky_owner.owns(p_rid)) {
		_update_dirty_skys();
		Sky *sky = sky_owner.getornull(p_rid);
		_clear_reflection_data(sky->reflection);
		sky_owner.free(p_rid);
	} else if (light_instance_owner.owns(p_rid)) {

		LightInstance *light_instance = light_instance_owner.getornull(p_rid);

		//remove from shadow atlases..
		for (Set<RID>::Element *E = light_instance->shadow_atlases.front(); E; E = E->next()) {
			ShadowAtlas *shadow_atlas = shadow_atlas_owner.getornull(E->get());
			ERR_CONTINUE(!shadow_atlas->shadow_owners.has(p_rid));
			uint32_t key = shadow_atlas->shadow_owners[p_rid];
			uint32_t q = (key >> ShadowAtlas::QUADRANT_SHIFT) & 0x3;
			uint32_t s = key & ShadowAtlas::SHADOW_INDEX_MASK;

			shadow_atlas->quadrants[q].shadows.write[s].owner = RID();
			shadow_atlas->shadow_owners.erase(p_rid);
		}

		light_instance_owner.free(p_rid);

	} else if (shadow_atlas_owner.owns(p_rid)) {

		shadow_atlas_set_size(p_rid, 0);
		shadow_atlas_owner.free(p_rid);

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

RasterizerSceneRD::~RasterizerSceneRD() {
	directional_shadow_atlas_set_size(0);

	for (Map<Vector2i, ShadowMap>::Element *E = shadow_maps.front(); E; E = E->next()) {
		RD::get_singleton()->free(E->get().depth);
	}
	for (Map<int, ShadowCubemap>::Element *E = shadow_cubemaps.front(); E; E = E->next()) {
		RD::get_singleton()->free(E->get().cubemap);
	}
}
