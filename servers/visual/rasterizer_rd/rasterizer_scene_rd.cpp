/*************************************************************************/
/*  rasterizer_scene_rd.cpp                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "rasterizer_scene_rd.h"

#include "core/os/os.h"
#include "core/project_settings.h"
#include "servers/visual/visual_server_raster.h"

uint64_t RasterizerSceneRD::auto_exposure_counter = 2;

void RasterizerSceneRD::_clear_reflection_data(ReflectionData &rd) {

	rd.layers.clear();
	rd.radiance_base_cubemap = RID();
}

void RasterizerSceneRD::_update_reflection_data(ReflectionData &rd, int p_size, int p_mipmaps, bool p_use_array, RID p_base_cube, int p_base_layer) {
	//recreate radiance and all data

	int mipmaps = p_mipmaps;
	uint32_t w = p_size, h = p_size;

	if (p_use_array) {

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
					mm.views[k] = RD::get_singleton()->texture_create_shared_from_slice(RD::TextureView(), p_base_cube, p_base_layer + i * 6 + k, j);
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
		ReflectionData::Layer layer;
		uint32_t mmw = w;
		uint32_t mmh = h;
		layer.mipmaps.resize(roughness_layers);
		for (int j = 0; j < roughness_layers; j++) {
			ReflectionData::Layer::Mipmap &mm = layer.mipmaps.write[j];
			mm.size.width = mmw;
			mm.size.height = mmh;
			for (int k = 0; k < 6; k++) {
				mm.views[k] = RD::get_singleton()->texture_create_shared_from_slice(RD::TextureView(), p_base_cube, p_base_layer + k, j);
				Vector<RID> fbtex;
				fbtex.push_back(mm.views[k]);
				mm.framebuffers[k] = RD::get_singleton()->framebuffer_create(fbtex);
			}

			mmw = MAX(1, mmw >> 1);
			mmh = MAX(1, mmh >> 1);
		}

		rd.layers.push_back(layer);
	}

	rd.radiance_base_cubemap = RD::get_singleton()->texture_create_shared_from_slice(RD::TextureView(), p_base_cube, p_base_layer, 0, RD::TEXTURE_SLICE_CUBEMAP);
}

void RasterizerSceneRD::_create_reflection_from_panorama(ReflectionData &rd, RID p_panorama, bool p_quality) {

#ifndef _MSC_VER
#warning TODO, should probably use this algorithm instead. Volunteers? - https://www.ppsloan.org/publications/ggx_filtering.pdf	 / https://github.com/dariomanesku/cmft
#endif
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

void RasterizerSceneRD::_create_reflection_from_base_mipmap(ReflectionData &rd, bool p_use_arrays, bool p_quality, int p_cube_side) {

	if (p_use_arrays) {

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
	if (sky->radiance.is_valid()) {
		RD::get_singleton()->free(sky->radiance);
		sky->radiance = RID();
	}
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
		if (sky->radiance.is_valid()) {
			RD::get_singleton()->free(sky->radiance);
			sky->radiance = RID();
		}
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

		if (sky->radiance.is_null()) {
			int mipmaps = Image::get_image_required_mipmaps(sky->radiance_size, sky->radiance_size, Image::FORMAT_RGBAH) + 1;
			if (sky->mode != VS::SKY_MODE_QUALITY) {
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

				_update_reflection_data(sky->reflection, sky->radiance_size, mipmaps, true, sky->radiance, 0);

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

				_update_reflection_data(sky->reflection, sky->radiance_size, mipmaps, false, sky->radiance, 0);
			}
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

	return sky->radiance;
}

RID RasterizerSceneRD::sky_get_radiance_uniform_set_rd(RID p_sky, RID p_shader, int p_set) const {
	Sky *sky = sky_owner.getornull(p_sky);
	ERR_FAIL_COND_V(!sky, RID());

	if (sky->uniform_set.is_null() || !RD::get_singleton()->uniform_set_is_valid(sky->uniform_set)) {

		sky->uniform_set = RID();
		if (sky->radiance.is_valid()) {
			Vector<RD::Uniform> uniforms;
			{
				RD::Uniform u;
				u.type = RD::UNIFORM_TYPE_TEXTURE;
				u.binding = 0;
				u.ids.push_back(sky->radiance);
				uniforms.push_back(u);
			}

			sky->uniform_set = RD::get_singleton()->uniform_set_create(uniforms, p_shader, p_set);
		}
	}

	return sky->uniform_set;
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
void RasterizerSceneRD::environment_set_ambient_light(RID p_env, const Color &p_color, VS::EnvironmentAmbientSource p_ambient, float p_energy, float p_sky_contribution, VS::EnvironmentReflectionSource p_reflection_source, const Color &p_ao_color) {
	Environent *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND(!env);
	env->ambient_light = p_color;
	env->ambient_source = p_ambient;
	env->ambient_light_energy = p_energy;
	env->ambient_sky_contribution = p_sky_contribution;
	env->reflection_source = p_reflection_source;
	env->ao_color = p_ao_color;
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

Color RasterizerSceneRD::environment_get_ao_color(RID p_env) const {
	Environent *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND_V(!env, Color());
	return env->ao_color;
}

void RasterizerSceneRD::environment_set_tonemap(RID p_env, VS::EnvironmentToneMapper p_tone_mapper, float p_exposure, float p_white, bool p_auto_exposure, float p_min_luminance, float p_max_luminance, float p_auto_exp_speed, float p_auto_exp_scale) {
	Environent *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND(!env);
	env->exposure = p_exposure;
	env->tone_mapper = p_tone_mapper;
	if (!env->auto_exposure && p_auto_exposure) {
		env->auto_exposure_version = ++auto_exposure_counter;
	}
	env->auto_exposure = p_auto_exposure;
	env->white = p_white;
	env->min_luminance = p_min_luminance;
	env->max_luminance = p_max_luminance;
	env->auto_exp_speed = p_auto_exp_speed;
	env->auto_exp_scale = p_auto_exp_scale;
}

void RasterizerSceneRD::environment_set_glow(RID p_env, bool p_enable, int p_level_flags, float p_intensity, float p_strength, float p_mix, float p_bloom_threshold, VS::EnvironmentGlowBlendMode p_blend_mode, float p_hdr_bleed_threshold, float p_hdr_bleed_scale, float p_hdr_luminance_cap, bool p_bicubic_upscale) {

	Environent *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND(!env);
	env->glow_enabled = p_enable;
	env->glow_levels = p_level_flags;
	env->glow_intensity = p_intensity;
	env->glow_strength = p_strength;
	env->glow_mix = p_mix;
	env->glow_bloom = p_bloom_threshold;
	env->glow_blend_mode = p_blend_mode;
	env->glow_hdr_bleed_threshold = p_hdr_bleed_threshold;
	env->glow_hdr_bleed_scale = p_hdr_bleed_scale;
	env->glow_hdr_luminance_cap = p_hdr_luminance_cap;
	env->glow_bicubic_upscale = p_bicubic_upscale;
}

void RasterizerSceneRD::environment_set_ssao(RID p_env, bool p_enable, float p_radius, float p_intensity, float p_bias, float p_light_affect, float p_ao_channel_affect, VS::EnvironmentSSAOBlur p_blur, float p_bilateral_sharpness) {

	Environent *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND(!env);

	env->ssao_enabled = p_enable;
	env->ssao_radius = p_radius;
	env->ssao_intensity = p_intensity;
	env->ssao_bias = p_bias;
	env->ssao_direct_light_affect = p_light_affect;
	env->ssao_ao_channel_affect = p_ao_channel_affect;
	env->ssao_blur = p_blur;
}

void RasterizerSceneRD::environment_set_ssao_quality(VS::EnvironmentSSAOQuality p_quality, bool p_half_size) {
	ssao_quality = p_quality;
	ssao_half_size = p_half_size;
}

bool RasterizerSceneRD::environment_is_ssao_enabled(RID p_env) const {

	Environent *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND_V(!env, false);
	return env->ssao_enabled;
}

float RasterizerSceneRD::environment_get_ssao_ao_affect(RID p_env) const {
	Environent *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND_V(!env, false);
	return env->ssao_ao_channel_affect;
}
float RasterizerSceneRD::environment_get_ssao_light_affect(RID p_env) const {
	Environent *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND_V(!env, false);
	return env->ssao_direct_light_affect;
}

bool RasterizerSceneRD::environment_is_ssr_enabled(RID p_env) const {

	Environent *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND_V(!env, false);
	return false;
}

bool RasterizerSceneRD::is_environment(RID p_env) const {
	return environment_owner.owns(p_env);
}

////////////////////////////////////////////////////////////

RID RasterizerSceneRD::reflection_atlas_create() {

	ReflectionAtlas ra;
	ra.count = GLOBAL_GET("rendering/quality/reflection_atlas/reflection_count");
	ra.size = GLOBAL_GET("rendering/quality/reflection_atlas/reflection_size");

	return reflection_atlas_owner.make_rid(ra);
}

void RasterizerSceneRD::reflection_atlas_set_size(RID p_ref_atlas, int p_reflection_size, int p_reflection_count) {

	ReflectionAtlas *ra = reflection_atlas_owner.getornull(p_ref_atlas);
	ERR_FAIL_COND(!ra);

	if (ra->size == p_reflection_size && ra->count == p_reflection_count) {
		return; //no changes
	}

	if (ra->reflection.is_valid()) {
		//clear and invalidate everything
		RD::get_singleton()->free(ra->reflection);
		ra->reflection = RID();

		for (int i = 0; i < ra->reflections.size(); i++) {
			if (ra->reflections[i].owner.is_null()) {
				continue;
			}
			reflection_probe_release_atlas_index(ra->reflections[i].owner);
			//rp->atlasindex clear
		}

		ra->reflections.clear();
	}
}

////////////////////////
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

void RasterizerSceneRD::reflection_probe_release_atlas_index(RID p_instance) {

	ReflectionProbeInstance *rpi = reflection_probe_instance_owner.getornull(p_instance);
	ERR_FAIL_COND(!rpi);

	if (rpi->atlas.is_null()) {
		return; //nothing to release
	}
	ReflectionAtlas *atlas = reflection_atlas_owner.getornull(rpi->atlas);
	ERR_FAIL_COND(!atlas);
	ERR_FAIL_INDEX(rpi->atlas_index, atlas->reflections.size());
	atlas->reflections.write[rpi->atlas_index].owner = RID();
	rpi->atlas_index = -1;
	rpi->atlas = RID();
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

	if (storage->reflection_probe_get_update_mode(rpi->probe) == VS::REFLECTION_PROBE_UPDATE_ALWAYS) {
		return true;
	}

	return rpi->atlas_index == -1;
}

bool RasterizerSceneRD::reflection_probe_instance_has_reflection(RID p_instance) {

	ReflectionProbeInstance *rpi = reflection_probe_instance_owner.getornull(p_instance);
	ERR_FAIL_COND_V(!rpi, false);

	return rpi->atlas.is_valid();
}

bool RasterizerSceneRD::reflection_probe_instance_begin_render(RID p_instance, RID p_reflection_atlas) {

	ReflectionAtlas *atlas = reflection_atlas_owner.getornull(p_reflection_atlas);

	ERR_FAIL_COND_V(!atlas, false);

	if (atlas->reflection.is_null()) {
		{
			//reflection atlas was unused, create:
			RD::TextureFormat tf;
			tf.array_layers = 6 * atlas->count;
			tf.format = RD::DATA_FORMAT_R16G16B16A16_SFLOAT;
			tf.type = RD::TEXTURE_TYPE_CUBE_ARRAY;
			tf.mipmaps = roughness_layers;
			tf.width = atlas->size;
			tf.height = atlas->size;
			tf.usage_bits = RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT;

			atlas->reflection = RD::get_singleton()->texture_create(tf, RD::TextureView());
		}
		{

			RD::TextureFormat tf;
			tf.format = RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_D32_SFLOAT, RD::TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) ? RD::DATA_FORMAT_D32_SFLOAT : RD::DATA_FORMAT_X8_D24_UNORM_PACK32;
			tf.width = atlas->size;
			tf.height = atlas->size;
			tf.usage_bits = RD::TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT;
			atlas->depth_buffer = RD::get_singleton()->texture_create(tf, RD::TextureView());
		}
		atlas->reflections.resize(atlas->count);
		for (int i = 0; i < atlas->count; i++) {
			_update_reflection_data(atlas->reflections.write[i].data, atlas->size, roughness_layers, false, atlas->reflection, i * 6);
			for (int j = 0; j < 6; j++) {
				Vector<RID> fb;
				fb.push_back(atlas->reflections.write[i].data.layers[0].mipmaps[0].views[j]);
				fb.push_back(atlas->depth_buffer);
				atlas->reflections.write[i].fbs[j] = RD::get_singleton()->framebuffer_create(fb);
			}
		}

		Vector<RID> fb;
		fb.push_back(atlas->depth_buffer);
		atlas->depth_fb = RD::get_singleton()->framebuffer_create(fb);
	}

	ReflectionProbeInstance *rpi = reflection_probe_instance_owner.getornull(p_instance);
	ERR_FAIL_COND_V(!rpi, false);

	if (rpi->atlas_index == -1) {
		for (int i = 0; i < atlas->reflections.size(); i++) {
			if (atlas->reflections[i].owner.is_null()) {
				rpi->atlas_index = i;
				break;
			}
		}
		//find the one used last
		if (rpi->atlas_index == -1) {
			//everything is in use, find the one least used via LRU
			uint64_t pass_min = 0;

			for (int i = 0; i < atlas->reflections.size(); i++) {
				ReflectionProbeInstance *rpi2 = reflection_probe_instance_owner.getornull(atlas->reflections[i].owner);
				if (rpi2->last_pass < pass_min) {
					pass_min = rpi2->last_pass;
					rpi->atlas_index = i;
				}
			}
		}
	}

	rpi->atlas = p_reflection_atlas;
	rpi->rendering = true;
	rpi->dirty = false;
	rpi->processing_side = 0;

	return true;
}

bool RasterizerSceneRD::reflection_probe_instance_postprocess_step(RID p_instance) {

	ReflectionProbeInstance *rpi = reflection_probe_instance_owner.getornull(p_instance);
	ERR_FAIL_COND_V(!rpi, false);
	ERR_FAIL_COND_V(!rpi->rendering, false);
	ERR_FAIL_COND_V(rpi->atlas.is_null(), false);

	ReflectionAtlas *atlas = reflection_atlas_owner.getornull(rpi->atlas);
	if (!atlas || rpi->atlas_index == -1) {
		//does not belong to an atlas anymore, cancel (was removed from atlas or atlas changed while rendering)
		rpi->rendering = false;
		return false;
	}

	_create_reflection_from_base_mipmap(atlas->reflections.write[rpi->atlas_index].data, false, storage->reflection_probe_get_update_mode(rpi->probe) == VS::REFLECTION_PROBE_UPDATE_ONCE, rpi->processing_side);

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

	ReflectionAtlas *atlas = reflection_atlas_owner.getornull(rpi->atlas);
	ERR_FAIL_COND_V(!atlas, 0);
	return atlas->size;
}

RID RasterizerSceneRD::reflection_probe_instance_get_framebuffer(RID p_instance, int p_index) {
	ReflectionProbeInstance *rpi = reflection_probe_instance_owner.getornull(p_instance);
	ERR_FAIL_COND_V(!rpi, RID());
	ERR_FAIL_INDEX_V(p_index, 6, RID());

	ReflectionAtlas *atlas = reflection_atlas_owner.getornull(rpi->atlas);
	ERR_FAIL_COND_V(!atlas, RID());
	return atlas->reflections[rpi->atlas_index].fbs[p_index];
}

RID RasterizerSceneRD::reflection_probe_instance_get_depth_framebuffer(RID p_instance, int p_index) {
	ReflectionProbeInstance *rpi = reflection_probe_instance_owner.getornull(p_instance);
	ERR_FAIL_COND_V(!rpi, RID());
	ERR_FAIL_INDEX_V(p_index, 6, RID());

	ReflectionAtlas *atlas = reflection_atlas_owner.getornull(rpi->atlas);
	ERR_FAIL_COND_V(!atlas, RID());
	return atlas->depth_fb;
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
	p_size = MAX(p_size, 1 << roughness_layers);

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

	directional_shadow.size = p_size;

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

	_base_uniforms_changed();
}

void RasterizerSceneRD::set_directional_shadow_count(int p_count) {

	directional_shadow.light_count = p_count;
	directional_shadow.current_light = 0;
}

static Rect2i _get_directional_shadow_rect(int p_size, int p_shadow_count, int p_shadow_index) {

	int split_h = 1;
	int split_v = 1;

	while (split_h * split_v < p_shadow_count) {
		if (split_h == split_v) {
			split_h <<= 1;
		} else {
			split_v <<= 1;
		}
	}

	Rect2i rect(0, 0, p_size, p_size);
	rect.size.width /= split_h;
	rect.size.height /= split_v;

	rect.position.x = rect.size.width * (p_shadow_index % split_h);
	rect.position.y = rect.size.height * (p_shadow_index / split_h);

	return rect;
}

int RasterizerSceneRD::get_directional_light_shadow_size(RID p_light_intance) {

	ERR_FAIL_COND_V(directional_shadow.light_count == 0, 0);

	Rect2i r = _get_directional_shadow_rect(directional_shadow.size, directional_shadow.light_count, 0);

	LightInstance *light_instance = light_instance_owner.getornull(p_light_intance);
	ERR_FAIL_COND_V(!light_instance, 0);

	switch (storage->light_directional_get_shadow_mode(light_instance->light)) {
		case VS::LIGHT_DIRECTIONAL_SHADOW_ORTHOGONAL:
			break; //none
		case VS::LIGHT_DIRECTIONAL_SHADOW_PARALLEL_2_SPLITS: r.size.height /= 2; break;
		case VS::LIGHT_DIRECTIONAL_SHADOW_PARALLEL_4_SPLITS: r.size /= 2; break;
	}

	return MAX(r.size.width, r.size.height);
}

//////////////////////////////////////////////////

RID RasterizerSceneRD::camera_effects_create() {

	return camera_effects_owner.make_rid(CameraEffects());
}

void RasterizerSceneRD::camera_effects_set_dof_blur_quality(VS::DOFBlurQuality p_quality, bool p_use_jitter) {

	dof_blur_quality = p_quality;
	dof_blur_use_jitter = p_use_jitter;
}

void RasterizerSceneRD::camera_effects_set_dof_blur_bokeh_shape(VS::DOFBokehShape p_shape) {

	dof_blur_bokeh_shape = p_shape;
}

void RasterizerSceneRD::camera_effects_set_dof_blur(RID p_camera_effects, bool p_far_enable, float p_far_distance, float p_far_transition, bool p_near_enable, float p_near_distance, float p_near_transition, float p_amount) {
	CameraEffects *camfx = camera_effects_owner.getornull(p_camera_effects);
	ERR_FAIL_COND(!camfx);

	camfx->dof_blur_far_enabled = p_far_enable;
	camfx->dof_blur_far_distance = p_far_distance;
	camfx->dof_blur_far_transition = p_far_transition;

	camfx->dof_blur_near_enabled = p_near_enable;
	camfx->dof_blur_near_distance = p_near_distance;
	camfx->dof_blur_near_transition = p_near_transition;

	camfx->dof_blur_amount = p_amount;
}

void RasterizerSceneRD::camera_effects_set_custom_exposure(RID p_camera_effects, bool p_enable, float p_exposure) {

	CameraEffects *camfx = camera_effects_owner.getornull(p_camera_effects);
	ERR_FAIL_COND(!camfx);

	camfx->override_exposure_enabled = p_enable;
	camfx->override_exposure = p_exposure;
}

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
/////////////////////////////////

RID RasterizerSceneRD::gi_probe_instance_create(RID p_base) {
	//find a free slot
	int index = -1;
	for (int i = 0; i < gi_probe_slots.size(); i++) {
		if (gi_probe_slots[i] == RID()) {
			index = i;
			break;
		}
	}

	ERR_FAIL_COND_V(index == -1, RID());

	GIProbeInstance gi_probe;
	gi_probe.slot = index;
	gi_probe.probe = p_base;
	RID rid = gi_probe_instance_owner.make_rid(gi_probe);
	gi_probe_slots.write[index] = rid;

	return rid;
}

void RasterizerSceneRD::gi_probe_instance_set_transform_to_data(RID p_probe, const Transform &p_xform) {

	GIProbeInstance *gi_probe = gi_probe_instance_owner.getornull(p_probe);
	ERR_FAIL_COND(!gi_probe);

	gi_probe->transform = p_xform;
}

bool RasterizerSceneRD::gi_probe_needs_update(RID p_probe) const {
	GIProbeInstance *gi_probe = gi_probe_instance_owner.getornull(p_probe);
	ERR_FAIL_COND_V(!gi_probe, false);

	//return true;
	return gi_probe->last_probe_version != storage->gi_probe_get_version(gi_probe->probe);
}

void RasterizerSceneRD::gi_probe_update(RID p_probe, bool p_update_light_instances, const Vector<RID> &p_light_instances, int p_dynamic_object_count, InstanceBase **p_dynamic_objects) {

	GIProbeInstance *gi_probe = gi_probe_instance_owner.getornull(p_probe);
	ERR_FAIL_COND(!gi_probe);

	uint32_t data_version = storage->gi_probe_get_data_version(gi_probe->probe);

	// (RE)CREATE IF NEEDED

	if (gi_probe->last_probe_data_version != data_version) {
		//need to re-create everything
		if (gi_probe->texture.is_valid()) {
			RD::get_singleton()->free(gi_probe->texture);
			if (gi_probe_use_anisotropy) {
				RD::get_singleton()->free(gi_probe->anisotropy_r16[0]);
				RD::get_singleton()->free(gi_probe->anisotropy_r16[1]);
			}
			RD::get_singleton()->free(gi_probe->write_buffer);
			gi_probe->mipmaps.clear();
		}

		for (int i = 0; i < gi_probe->dynamic_maps.size(); i++) {
			RD::get_singleton()->free(gi_probe->dynamic_maps[i].texture);
			RD::get_singleton()->free(gi_probe->dynamic_maps[i].depth);
		}

		gi_probe->dynamic_maps.clear();

		Vector3i octree_size = storage->gi_probe_get_octree_size(gi_probe->probe);

		if (octree_size != Vector3i()) {
			//can create a 3D texture
			PoolVector<int> levels = storage->gi_probe_get_level_counts(gi_probe->probe);

			RD::TextureFormat tf;
			tf.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
			tf.width = octree_size.x;
			tf.height = octree_size.y;
			tf.depth = octree_size.z;
			tf.type = RD::TEXTURE_TYPE_3D;
			tf.mipmaps = levels.size();

			tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_CAN_COPY_TO_BIT;

			gi_probe->texture = RD::get_singleton()->texture_create(tf, RD::TextureView());

			RD::get_singleton()->texture_clear(gi_probe->texture, Color(0, 0, 0, 0), 0, levels.size(), 0, 1, false);

			if (gi_probe_use_anisotropy) {
				tf.format = RD::DATA_FORMAT_R16_UINT;
				tf.shareable_formats.push_back(RD::DATA_FORMAT_R16_UINT);
				tf.shareable_formats.push_back(RD::DATA_FORMAT_R5G6B5_UNORM_PACK16);

				//need to create R16 first, else driver does not like the storage bit for compute..
				gi_probe->anisotropy_r16[0] = RD::get_singleton()->texture_create(tf, RD::TextureView());
				gi_probe->anisotropy_r16[1] = RD::get_singleton()->texture_create(tf, RD::TextureView());

				RD::TextureView tv;
				tv.format_override = RD::DATA_FORMAT_R5G6B5_UNORM_PACK16;
				gi_probe->anisotropy[0] = RD::get_singleton()->texture_create_shared(tv, gi_probe->anisotropy_r16[0]);
				gi_probe->anisotropy[1] = RD::get_singleton()->texture_create_shared(tv, gi_probe->anisotropy_r16[1]);

				RD::get_singleton()->texture_clear(gi_probe->anisotropy[0], Color(0, 0, 0, 0), 0, levels.size(), 0, 1, false);
				RD::get_singleton()->texture_clear(gi_probe->anisotropy[1], Color(0, 0, 0, 0), 0, levels.size(), 0, 1, false);
			}

			{
				int total_elements = 0;
				for (int i = 0; i < levels.size(); i++) {
					total_elements += levels[i];
				}

				if (gi_probe_use_anisotropy) {
					total_elements *= 6;
				}

				gi_probe->write_buffer = RD::get_singleton()->storage_buffer_create(total_elements * 16);
			}

			for (int i = 0; i < levels.size(); i++) {
				GIProbeInstance::Mipmap mipmap;
				mipmap.texture = RD::get_singleton()->texture_create_shared_from_slice(RD::TextureView(), gi_probe->texture, 0, i, RD::TEXTURE_SLICE_3D);
				if (gi_probe_use_anisotropy) {
					RD::TextureView tv;
					tv.format_override = RD::DATA_FORMAT_R16_UINT;
					mipmap.anisotropy[0] = RD::get_singleton()->texture_create_shared_from_slice(tv, gi_probe->anisotropy[0], 0, i, RD::TEXTURE_SLICE_3D);
					mipmap.anisotropy[1] = RD::get_singleton()->texture_create_shared_from_slice(tv, gi_probe->anisotropy[1], 0, i, RD::TEXTURE_SLICE_3D);
				}

				mipmap.level = levels.size() - i - 1;
				mipmap.cell_offset = 0;
				for (uint32_t j = 0; j < mipmap.level; j++) {
					mipmap.cell_offset += levels[j];
				}
				mipmap.cell_count = levels[mipmap.level];

				Vector<RD::Uniform> uniforms;
				{
					RD::Uniform u;
					u.type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
					u.binding = 1;
					u.ids.push_back(storage->gi_probe_get_octree_buffer(gi_probe->probe));
					uniforms.push_back(u);
				}
				{
					RD::Uniform u;
					u.type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
					u.binding = 2;
					u.ids.push_back(storage->gi_probe_get_data_buffer(gi_probe->probe));
					uniforms.push_back(u);
				}

				{
					RD::Uniform u;
					u.type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
					u.binding = 4;
					u.ids.push_back(gi_probe->write_buffer);
					uniforms.push_back(u);
				}
				{
					RD::Uniform u;
					u.type = RD::UNIFORM_TYPE_TEXTURE;
					u.binding = 9;
					u.ids.push_back(storage->gi_probe_get_sdf_texture(gi_probe->probe));
					uniforms.push_back(u);
				}
				{
					RD::Uniform u;
					u.type = RD::UNIFORM_TYPE_SAMPLER;
					u.binding = 10;
					u.ids.push_back(storage->sampler_rd_get_default(VS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS, VS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED));
					uniforms.push_back(u);
				}

				{
					Vector<RD::Uniform> copy_uniforms = uniforms;
					if (i == 0) {
						{
							RD::Uniform u;
							u.type = RD::UNIFORM_TYPE_UNIFORM_BUFFER;
							u.binding = 3;
							u.ids.push_back(gi_probe_lights_uniform);
							copy_uniforms.push_back(u);
						}

						mipmap.uniform_set = RD::get_singleton()->uniform_set_create(copy_uniforms, giprobe_lighting_shader_version_shaders[GI_PROBE_SHADER_VERSION_COMPUTE_LIGHT], 0);

						copy_uniforms = uniforms; //restore

						{
							RD::Uniform u;
							u.type = RD::UNIFORM_TYPE_TEXTURE;
							u.binding = 5;
							u.ids.push_back(gi_probe->texture);
							copy_uniforms.push_back(u);
						}

						if (gi_probe_use_anisotropy) {
							{
								RD::Uniform u;
								u.type = RD::UNIFORM_TYPE_TEXTURE;
								u.binding = 7;
								u.ids.push_back(gi_probe->anisotropy[0]);
								copy_uniforms.push_back(u);
							}
							{
								RD::Uniform u;
								u.type = RD::UNIFORM_TYPE_TEXTURE;
								u.binding = 8;
								u.ids.push_back(gi_probe->anisotropy[1]);
								copy_uniforms.push_back(u);
							}
						}

						mipmap.second_bounce_uniform_set = RD::get_singleton()->uniform_set_create(copy_uniforms, giprobe_lighting_shader_version_shaders[GI_PROBE_SHADER_VERSION_COMPUTE_SECOND_BOUNCE], 0);
					} else {
						mipmap.uniform_set = RD::get_singleton()->uniform_set_create(copy_uniforms, giprobe_lighting_shader_version_shaders[GI_PROBE_SHADER_VERSION_COMPUTE_MIPMAP], 0);
					}
				}

				{
					RD::Uniform u;
					u.type = RD::UNIFORM_TYPE_IMAGE;
					u.binding = 5;
					u.ids.push_back(mipmap.texture);
					uniforms.push_back(u);
				}

				if (gi_probe_use_anisotropy) {
					{
						RD::Uniform u;
						u.type = RD::UNIFORM_TYPE_IMAGE;
						u.binding = 6;
						u.ids.push_back(mipmap.anisotropy[0]);
						uniforms.push_back(u);
					}
					{
						RD::Uniform u;
						u.type = RD::UNIFORM_TYPE_IMAGE;
						u.binding = 7;
						u.ids.push_back(mipmap.anisotropy[1]);
						uniforms.push_back(u);
					}
				}

				mipmap.write_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, giprobe_lighting_shader_version_shaders[GI_PROBE_SHADER_VERSION_WRITE_TEXTURE], 0);

				gi_probe->mipmaps.push_back(mipmap);
			}

			{
				uint32_t dynamic_map_size = MAX(MAX(octree_size.x, octree_size.y), octree_size.z);
				uint32_t oversample = nearest_power_of_2_templated(4);
				int mipmap_index = 0;

				while (mipmap_index < gi_probe->mipmaps.size()) {
					GIProbeInstance::DynamicMap dmap;

					if (oversample > 0) {
						dmap.size = dynamic_map_size * (1 << oversample);
						dmap.mipmap = -1;
						oversample--;
					} else {
						dmap.size = dynamic_map_size >> mipmap_index;
						dmap.mipmap = mipmap_index;
						mipmap_index++;
					}

					RD::TextureFormat dtf;
					dtf.width = dmap.size;
					dtf.height = dmap.size;
					dtf.format = RD::DATA_FORMAT_R16G16B16A16_SFLOAT;
					dtf.usage_bits = RD::TEXTURE_USAGE_STORAGE_BIT;

					if (gi_probe->dynamic_maps.size() == 0) {
						dtf.usage_bits |= RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT;
					}
					dmap.texture = RD::get_singleton()->texture_create(dtf, RD::TextureView());

					if (gi_probe->dynamic_maps.size() == 0) {
						//render depth for first one
						dtf.format = RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_D32_SFLOAT, RD::TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) ? RD::DATA_FORMAT_D32_SFLOAT : RD::DATA_FORMAT_X8_D24_UNORM_PACK32;
						dtf.usage_bits = RD::TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
						dmap.fb_depth = RD::get_singleton()->texture_create(dtf, RD::TextureView());
					}

					//just use depth as-is
					dtf.format = RD::DATA_FORMAT_R32_SFLOAT;
					dtf.usage_bits = RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT;

					dmap.depth = RD::get_singleton()->texture_create(dtf, RD::TextureView());

					if (gi_probe->dynamic_maps.size() == 0) {

						dtf.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
						dtf.usage_bits = RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT;
						dmap.albedo = RD::get_singleton()->texture_create(dtf, RD::TextureView());
						dmap.normal = RD::get_singleton()->texture_create(dtf, RD::TextureView());
						dmap.orm = RD::get_singleton()->texture_create(dtf, RD::TextureView());

						Vector<RID> fb;
						fb.push_back(dmap.albedo);
						fb.push_back(dmap.normal);
						fb.push_back(dmap.orm);
						fb.push_back(dmap.texture); //emission
						fb.push_back(dmap.depth);
						fb.push_back(dmap.fb_depth);

						dmap.fb = RD::get_singleton()->framebuffer_create(fb);

						{
							Vector<RD::Uniform> uniforms;
							{
								RD::Uniform u;
								u.type = RD::UNIFORM_TYPE_UNIFORM_BUFFER;
								u.binding = 3;
								u.ids.push_back(gi_probe_lights_uniform);
								uniforms.push_back(u);
							}

							{
								RD::Uniform u;
								u.type = RD::UNIFORM_TYPE_IMAGE;
								u.binding = 5;
								u.ids.push_back(dmap.albedo);
								uniforms.push_back(u);
							}
							{
								RD::Uniform u;
								u.type = RD::UNIFORM_TYPE_IMAGE;
								u.binding = 6;
								u.ids.push_back(dmap.normal);
								uniforms.push_back(u);
							}
							{
								RD::Uniform u;
								u.type = RD::UNIFORM_TYPE_IMAGE;
								u.binding = 7;
								u.ids.push_back(dmap.orm);
								uniforms.push_back(u);
							}
							{
								RD::Uniform u;
								u.type = RD::UNIFORM_TYPE_TEXTURE;
								u.binding = 8;
								u.ids.push_back(dmap.fb_depth);
								uniforms.push_back(u);
							}
							{
								RD::Uniform u;
								u.type = RD::UNIFORM_TYPE_TEXTURE;
								u.binding = 9;
								u.ids.push_back(storage->gi_probe_get_sdf_texture(gi_probe->probe));
								uniforms.push_back(u);
							}
							{
								RD::Uniform u;
								u.type = RD::UNIFORM_TYPE_SAMPLER;
								u.binding = 10;
								u.ids.push_back(storage->sampler_rd_get_default(VS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS, VS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED));
								uniforms.push_back(u);
							}
							{
								RD::Uniform u;
								u.type = RD::UNIFORM_TYPE_IMAGE;
								u.binding = 11;
								u.ids.push_back(dmap.texture);
								uniforms.push_back(u);
							}
							{
								RD::Uniform u;
								u.type = RD::UNIFORM_TYPE_IMAGE;
								u.binding = 12;
								u.ids.push_back(dmap.depth);
								uniforms.push_back(u);
							}

							dmap.uniform_set = RD::get_singleton()->uniform_set_create(uniforms, giprobe_lighting_shader_version_shaders[GI_PROBE_SHADER_VERSION_DYNAMIC_OBJECT_LIGHTING], 0);
						}
					} else {
						bool plot = dmap.mipmap >= 0;
						bool write = dmap.mipmap < (gi_probe->mipmaps.size() - 1);

						Vector<RD::Uniform> uniforms;

						{
							RD::Uniform u;
							u.type = RD::UNIFORM_TYPE_IMAGE;
							u.binding = 5;
							u.ids.push_back(gi_probe->dynamic_maps[gi_probe->dynamic_maps.size() - 1].texture);
							uniforms.push_back(u);
						}
						{
							RD::Uniform u;
							u.type = RD::UNIFORM_TYPE_IMAGE;
							u.binding = 6;
							u.ids.push_back(gi_probe->dynamic_maps[gi_probe->dynamic_maps.size() - 1].depth);
							uniforms.push_back(u);
						}

						if (write) {

							{
								RD::Uniform u;
								u.type = RD::UNIFORM_TYPE_IMAGE;
								u.binding = 7;
								u.ids.push_back(dmap.texture);
								uniforms.push_back(u);
							}
							{
								RD::Uniform u;
								u.type = RD::UNIFORM_TYPE_IMAGE;
								u.binding = 8;
								u.ids.push_back(dmap.depth);
								uniforms.push_back(u);
							}
						}

						{
							RD::Uniform u;
							u.type = RD::UNIFORM_TYPE_TEXTURE;
							u.binding = 9;
							u.ids.push_back(storage->gi_probe_get_sdf_texture(gi_probe->probe));
							uniforms.push_back(u);
						}
						{
							RD::Uniform u;
							u.type = RD::UNIFORM_TYPE_SAMPLER;
							u.binding = 10;
							u.ids.push_back(storage->sampler_rd_get_default(VS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS, VS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED));
							uniforms.push_back(u);
						}

						if (plot) {

							{
								RD::Uniform u;
								u.type = RD::UNIFORM_TYPE_IMAGE;
								u.binding = 11;
								u.ids.push_back(gi_probe->mipmaps[dmap.mipmap].texture);
								uniforms.push_back(u);
							}
							if (gi_probe_is_anisotropic()) {
								{
									RD::Uniform u;
									u.type = RD::UNIFORM_TYPE_IMAGE;
									u.binding = 12;
									u.ids.push_back(gi_probe->mipmaps[dmap.mipmap].anisotropy[0]);
									uniforms.push_back(u);
								}
								{
									RD::Uniform u;
									u.type = RD::UNIFORM_TYPE_IMAGE;
									u.binding = 13;
									u.ids.push_back(gi_probe->mipmaps[dmap.mipmap].anisotropy[1]);
									uniforms.push_back(u);
								}
							}
						}

						dmap.uniform_set = RD::get_singleton()->uniform_set_create(uniforms, giprobe_lighting_shader_version_shaders[(write && plot) ? GI_PROBE_SHADER_VERSION_DYNAMIC_SHRINK_WRITE_PLOT : write ? GI_PROBE_SHADER_VERSION_DYNAMIC_SHRINK_WRITE : GI_PROBE_SHADER_VERSION_DYNAMIC_SHRINK_PLOT], 0);
					}

					gi_probe->dynamic_maps.push_back(dmap);
				}
			}
		}

		gi_probe->last_probe_data_version = data_version;
		p_update_light_instances = true; //just in case

		_base_uniforms_changed();
	}

	// UDPDATE TIME

	if (gi_probe->has_dynamic_object_data) {
		//if it has dynamic object data, it needs to be cleared
		RD::get_singleton()->texture_clear(gi_probe->texture, Color(0, 0, 0, 0), 0, gi_probe->mipmaps.size(), 0, 1, true);
		if (gi_probe_is_anisotropic()) {
			RD::get_singleton()->texture_clear(gi_probe->anisotropy[0], Color(0, 0, 0, 0), 0, gi_probe->mipmaps.size(), 0, 1, true);
			RD::get_singleton()->texture_clear(gi_probe->anisotropy[1], Color(0, 0, 0, 0), 0, gi_probe->mipmaps.size(), 0, 1, true);
		}
	}

	uint32_t light_count = 0;

	if (p_update_light_instances || p_dynamic_object_count > 0) {

		light_count = MIN(gi_probe_max_lights, (uint32_t)p_light_instances.size());

		{
			Transform to_cell = storage->gi_probe_get_to_cell_xform(gi_probe->probe);
			Transform to_probe_xform = (gi_probe->transform * to_cell.affine_inverse()).affine_inverse();
			//update lights

			for (uint32_t i = 0; i < light_count; i++) {
				GIProbeLight &l = gi_probe_lights[i];
				RID light_instance = p_light_instances[i];
				RID light = light_instance_get_base_light(light_instance);

				l.type = storage->light_get_type(light);
				l.attenuation = storage->light_get_param(light, VS::LIGHT_PARAM_ATTENUATION);
				l.energy = storage->light_get_param(light, VS::LIGHT_PARAM_ENERGY) * storage->light_get_param(light, VS::LIGHT_PARAM_INDIRECT_ENERGY);
				l.radius = to_cell.basis.xform(Vector3(storage->light_get_param(light, VS::LIGHT_PARAM_RANGE), 0, 0)).length();
				Color color = storage->light_get_color(light).to_linear();
				l.color[0] = color.r;
				l.color[1] = color.g;
				l.color[2] = color.b;

				l.spot_angle_radians = Math::deg2rad(storage->light_get_param(light, VS::LIGHT_PARAM_SPOT_ANGLE));
				l.spot_attenuation = storage->light_get_param(light, VS::LIGHT_PARAM_SPOT_ATTENUATION);

				Transform xform = light_instance_get_base_transform(light_instance);

				Vector3 pos = to_probe_xform.xform(xform.origin);
				Vector3 dir = to_probe_xform.basis.xform(-xform.basis.get_axis(2)).normalized();

				l.position[0] = pos.x;
				l.position[1] = pos.y;
				l.position[2] = pos.z;

				l.direction[0] = dir.x;
				l.direction[1] = dir.y;
				l.direction[2] = dir.z;

				l.has_shadow = storage->light_has_shadow(light);
			}

			RD::get_singleton()->buffer_update(gi_probe_lights_uniform, 0, sizeof(GIProbeLight) * light_count, gi_probe_lights, true);
		}
	}

	if (gi_probe->has_dynamic_object_data || p_update_light_instances || p_dynamic_object_count) {
		// PROCESS MIPMAPS
		if (gi_probe->mipmaps.size()) {
			//can update mipmaps

			Vector3i probe_size = storage->gi_probe_get_octree_size(gi_probe->probe);

			GIProbePushConstant push_constant;

			push_constant.limits[0] = probe_size.x;
			push_constant.limits[1] = probe_size.y;
			push_constant.limits[2] = probe_size.z;
			push_constant.stack_size = gi_probe->mipmaps.size();
			push_constant.emission_scale = 1.0;
			push_constant.propagation = storage->gi_probe_get_propagation(gi_probe->probe);
			push_constant.dynamic_range = storage->gi_probe_get_dynamic_range(gi_probe->probe);
			push_constant.light_count = light_count;
			push_constant.aniso_strength = storage->gi_probe_get_anisotropy_strength(gi_probe->probe);

			/*		print_line("probe update to version " + itos(gi_probe->last_probe_version));
			print_line("propagation " + rtos(push_constant.propagation));
			print_line("dynrange " + rtos(push_constant.dynamic_range));
	*/
			RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();

			int passes;
			if (p_update_light_instances) {
				passes = storage->gi_probe_is_using_two_bounces(gi_probe->probe) ? 2 : 1;
			} else {
				passes = 1; //only re-blitting is necessary
			}
			int wg_size = 64;
			int wg_limit_x = RD::get_singleton()->limit_get(RD::LIMIT_MAX_COMPUTE_WORKGROUP_COUNT_X);

			for (int pass = 0; pass < passes; pass++) {

				if (p_update_light_instances) {

					for (int i = 0; i < gi_probe->mipmaps.size(); i++) {
						if (i == 0) {
							RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, giprobe_lighting_shader_version_pipelines[pass == 0 ? GI_PROBE_SHADER_VERSION_COMPUTE_LIGHT : GI_PROBE_SHADER_VERSION_COMPUTE_SECOND_BOUNCE]);
						} else if (i == 1) {
							RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, giprobe_lighting_shader_version_pipelines[GI_PROBE_SHADER_VERSION_COMPUTE_MIPMAP]);
						}

						if (pass == 1 || i > 0) {
							RD::get_singleton()->compute_list_add_barrier(compute_list); //wait til previous step is done
						}
						if (pass == 0 || i > 0) {
							RD::get_singleton()->compute_list_bind_uniform_set(compute_list, gi_probe->mipmaps[i].uniform_set, 0);
						} else {
							RD::get_singleton()->compute_list_bind_uniform_set(compute_list, gi_probe->mipmaps[i].second_bounce_uniform_set, 0);
						}

						push_constant.cell_offset = gi_probe->mipmaps[i].cell_offset;
						push_constant.cell_count = gi_probe->mipmaps[i].cell_count;

						int wg_todo = (gi_probe->mipmaps[i].cell_count - 1) / wg_size + 1;
						while (wg_todo) {
							int wg_count = MIN(wg_todo, wg_limit_x);
							RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(GIProbePushConstant));
							RD::get_singleton()->compute_list_dispatch(compute_list, wg_count, 1, 1);
							wg_todo -= wg_count;
							push_constant.cell_offset += wg_count * wg_size;
						}
					}

					RD::get_singleton()->compute_list_add_barrier(compute_list); //wait til previous step is done
				}

				RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, giprobe_lighting_shader_version_pipelines[GI_PROBE_SHADER_VERSION_WRITE_TEXTURE]);

				for (int i = 0; i < gi_probe->mipmaps.size(); i++) {

					RD::get_singleton()->compute_list_bind_uniform_set(compute_list, gi_probe->mipmaps[i].write_uniform_set, 0);

					push_constant.cell_offset = gi_probe->mipmaps[i].cell_offset;
					push_constant.cell_count = gi_probe->mipmaps[i].cell_count;

					int wg_todo = (gi_probe->mipmaps[i].cell_count - 1) / wg_size + 1;
					while (wg_todo) {
						int wg_count = MIN(wg_todo, wg_limit_x);
						RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(GIProbePushConstant));
						RD::get_singleton()->compute_list_dispatch(compute_list, wg_count, 1, 1);
						wg_todo -= wg_count;
						push_constant.cell_offset += wg_count * wg_size;
					}
				}
			}

			RD::get_singleton()->compute_list_end();
		}
	}

	gi_probe->has_dynamic_object_data = false; //clear until dynamic object data is used again

	if (p_dynamic_object_count && gi_probe->dynamic_maps.size()) {

		Vector3i octree_size = storage->gi_probe_get_octree_size(gi_probe->probe);
		int multiplier = gi_probe->dynamic_maps[0].size / MAX(MAX(octree_size.x, octree_size.y), octree_size.z);

		Transform oversample_scale;
		oversample_scale.basis.scale(Vector3(multiplier, multiplier, multiplier));

		Transform to_cell = oversample_scale * storage->gi_probe_get_to_cell_xform(gi_probe->probe);
		Transform to_world_xform = gi_probe->transform * to_cell.affine_inverse();
		Transform to_probe_xform = to_world_xform.affine_inverse();

		AABB probe_aabb(Vector3(), octree_size);

		//this could probably be better parallelized in compute..
		for (int i = 0; i < p_dynamic_object_count; i++) {

			InstanceBase *instance = p_dynamic_objects[i];
			//not used, so clear
			instance->depth_layer = 0;
			instance->depth = 0;

			//transform aabb to giprobe
			AABB aabb = (to_probe_xform * instance->transform).xform(instance->aabb);

			//this needs to wrap to grid resolution to avoid jitter
			//also extend margin a bit just in case
			Vector3i begin = aabb.position - Vector3i(1, 1, 1);
			Vector3i end = aabb.position + aabb.size + Vector3i(1, 1, 1);

			for (int j = 0; j < 3; j++) {
				if ((end[j] - begin[j]) & 1) {
					end[j]++; //for half extents split, it needs to be even
				}
				begin[j] = MAX(begin[j], 0);
				end[j] = MIN(end[j], octree_size[j] * multiplier);
			}

			//aabb = aabb.intersection(probe_aabb); //intersect
			aabb.position = begin;
			aabb.size = end - begin;

			//print_line("aabb: " + aabb);

			for (int j = 0; j < 6; j++) {

				//if (j != 0 && j != 3) {
				//	continue;
				//}
				static const Vector3 render_z[6] = {
					Vector3(1, 0, 0),
					Vector3(0, 1, 0),
					Vector3(0, 0, 1),
					Vector3(-1, 0, 0),
					Vector3(0, -1, 0),
					Vector3(0, 0, -1),
				};
				static const Vector3 render_up[6] = {
					Vector3(0, 1, 0),
					Vector3(0, 0, 1),
					Vector3(0, 1, 0),
					Vector3(0, 1, 0),
					Vector3(0, 0, 1),
					Vector3(0, 1, 0),
				};

				Vector3 render_dir = render_z[j];
				Vector3 up_dir = render_up[j];

				Vector3 center = aabb.position + aabb.size * 0.5;
				Transform xform;
				xform.set_look_at(center - aabb.size * 0.5 * render_dir, center, up_dir);

				Vector3 x_dir = xform.basis.get_axis(0).abs();
				int x_axis = int(Vector3(0, 1, 2).dot(x_dir));
				Vector3 y_dir = xform.basis.get_axis(1).abs();
				int y_axis = int(Vector3(0, 1, 2).dot(y_dir));
				Vector3 z_dir = -xform.basis.get_axis(2);
				int z_axis = int(Vector3(0, 1, 2).dot(z_dir.abs()));

				Rect2i rect(aabb.position[x_axis], aabb.position[y_axis], aabb.size[x_axis], aabb.size[y_axis]);
				bool x_flip = bool(Vector3(1, 1, 1).dot(xform.basis.get_axis(0)) < 0);
				bool y_flip = bool(Vector3(1, 1, 1).dot(xform.basis.get_axis(1)) < 0);
				bool z_flip = bool(Vector3(1, 1, 1).dot(xform.basis.get_axis(2)) > 0);

				CameraMatrix cm;
				cm.set_orthogonal(-rect.size.width / 2, rect.size.width / 2, -rect.size.height / 2, rect.size.height / 2, 0.0001, aabb.size[z_axis]);

				_render_material(to_world_xform * xform, cm, true, &instance, 1, gi_probe->dynamic_maps[0].fb, Rect2i(Vector2i(), rect.size));

				GIProbeDynamicPushConstant push_constant;
				zeromem(&push_constant, sizeof(GIProbeDynamicPushConstant));
				push_constant.limits[0] = octree_size.x;
				push_constant.limits[1] = octree_size.y;
				push_constant.limits[2] = octree_size.z;
				push_constant.light_count = p_light_instances.size();
				push_constant.x_dir[0] = x_dir[0];
				push_constant.x_dir[1] = x_dir[1];
				push_constant.x_dir[2] = x_dir[2];
				push_constant.y_dir[0] = y_dir[0];
				push_constant.y_dir[1] = y_dir[1];
				push_constant.y_dir[2] = y_dir[2];
				push_constant.z_dir[0] = z_dir[0];
				push_constant.z_dir[1] = z_dir[1];
				push_constant.z_dir[2] = z_dir[2];
				push_constant.z_base = xform.origin[z_axis];
				push_constant.z_sign = (z_flip ? -1.0 : 1.0);
				push_constant.pos_multiplier = float(1.0) / multiplier;
				push_constant.dynamic_range = storage->gi_probe_get_dynamic_range(gi_probe->probe);
				push_constant.flip_x = x_flip;
				push_constant.flip_y = y_flip;
				push_constant.rect_pos[0] = rect.position[0];
				push_constant.rect_pos[1] = rect.position[1];
				push_constant.rect_size[0] = rect.size[0];
				push_constant.rect_size[1] = rect.size[1];
				push_constant.prev_rect_ofs[0] = 0;
				push_constant.prev_rect_ofs[1] = 0;
				push_constant.prev_rect_size[0] = 0;
				push_constant.prev_rect_size[1] = 0;
				push_constant.on_mipmap = false;
				push_constant.propagation = storage->gi_probe_get_propagation(gi_probe->probe);
				push_constant.pad[0] = 0;
				push_constant.pad[1] = 0;
				push_constant.pad[2] = 0;

				//process lighting
				RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();
				RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, giprobe_lighting_shader_version_pipelines[GI_PROBE_SHADER_VERSION_DYNAMIC_OBJECT_LIGHTING]);
				RD::get_singleton()->compute_list_bind_uniform_set(compute_list, gi_probe->dynamic_maps[0].uniform_set, 0);
				RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(GIProbeDynamicPushConstant));
				RD::get_singleton()->compute_list_dispatch(compute_list, (rect.size.x - 1) / 8 + 1, (rect.size.y - 1) / 8 + 1, 1);
				//print_line("rect: " + itos(i) + ": " + rect);

				for (int k = 1; k < gi_probe->dynamic_maps.size(); k++) {

					// enlarge the rect if needed so all pixels fit when downscaled,
					// this ensures downsampling is smooth and optimal because no pixels are left behind

					//x
					if (rect.position.x & 1) {
						rect.size.x++;
						push_constant.prev_rect_ofs[0] = 1; //this is used to ensure reading is also optimal
					} else {
						push_constant.prev_rect_ofs[0] = 0;
					}
					if (rect.size.x & 1) {
						rect.size.x++;
					}

					rect.position.x >>= 1;
					rect.size.x = MAX(1, rect.size.x >> 1);

					//y
					if (rect.position.y & 1) {
						rect.size.y++;
						push_constant.prev_rect_ofs[1] = 1;
					} else {
						push_constant.prev_rect_ofs[1] = 0;
					}
					if (rect.size.y & 1) {
						rect.size.y++;
					}

					rect.position.y >>= 1;
					rect.size.y = MAX(1, rect.size.y >> 1);

					//shrink limits to ensure plot does not go outside map
					if (gi_probe->dynamic_maps[k].mipmap > 0) {
						for (int l = 0; l < 3; l++) {
							push_constant.limits[l] = MAX(1, push_constant.limits[l] >> 1);
						}
					}

					//print_line("rect: " + itos(i) + ": " + rect);
					push_constant.rect_pos[0] = rect.position[0];
					push_constant.rect_pos[1] = rect.position[1];
					push_constant.prev_rect_size[0] = push_constant.rect_size[0];
					push_constant.prev_rect_size[1] = push_constant.rect_size[1];
					push_constant.rect_size[0] = rect.size[0];
					push_constant.rect_size[1] = rect.size[1];
					push_constant.on_mipmap = gi_probe->dynamic_maps[k].mipmap > 0;

					RD::get_singleton()->compute_list_add_barrier(compute_list);

					if (gi_probe->dynamic_maps[k].mipmap < 0) {
						RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, giprobe_lighting_shader_version_pipelines[GI_PROBE_SHADER_VERSION_DYNAMIC_SHRINK_WRITE]);
					} else if (k < gi_probe->dynamic_maps.size() - 1) {
						RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, giprobe_lighting_shader_version_pipelines[GI_PROBE_SHADER_VERSION_DYNAMIC_SHRINK_WRITE_PLOT]);
					} else {
						RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, giprobe_lighting_shader_version_pipelines[GI_PROBE_SHADER_VERSION_DYNAMIC_SHRINK_PLOT]);
					}
					RD::get_singleton()->compute_list_bind_uniform_set(compute_list, gi_probe->dynamic_maps[k].uniform_set, 0);
					RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(GIProbeDynamicPushConstant));
					RD::get_singleton()->compute_list_dispatch(compute_list, (rect.size.x - 1) / 8 + 1, (rect.size.y - 1) / 8 + 1, 1);
				}

				RD::get_singleton()->compute_list_end();
			}
		}

		gi_probe->has_dynamic_object_data = true; //clear until dynamic object data is used again
	}

	gi_probe->last_probe_version = storage->gi_probe_get_version(gi_probe->probe);
}

void RasterizerSceneRD::_debug_giprobe(RID p_gi_probe, RD::DrawListID p_draw_list, RID p_framebuffer, const CameraMatrix &p_camera_with_transform, bool p_lighting, bool p_emission, float p_alpha) {
	GIProbeInstance *gi_probe = gi_probe_instance_owner.getornull(p_gi_probe);
	ERR_FAIL_COND(!gi_probe);

	if (gi_probe->mipmaps.size() == 0) {
		return;
	}

	CameraMatrix transform = (p_camera_with_transform * CameraMatrix(gi_probe->transform)) * CameraMatrix(storage->gi_probe_get_to_cell_xform(gi_probe->probe).affine_inverse());

	int level = 0;
	Vector3i octree_size = storage->gi_probe_get_octree_size(gi_probe->probe);

	GIProbeDebugPushConstant push_constant;
	push_constant.alpha = p_alpha;
	push_constant.dynamic_range = storage->gi_probe_get_dynamic_range(gi_probe->probe);
	push_constant.cell_offset = gi_probe->mipmaps[level].cell_offset;
	push_constant.level = level;

	push_constant.bounds[0] = octree_size.x >> level;
	push_constant.bounds[1] = octree_size.y >> level;
	push_constant.bounds[2] = octree_size.z >> level;
	push_constant.pad = 0;

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {

			push_constant.projection[i * 4 + j] = transform.matrix[i][j];
		}
	}

	if (giprobe_debug_uniform_set.is_valid()) {
		RD::get_singleton()->free(giprobe_debug_uniform_set);
	}
	Vector<RD::Uniform> uniforms;
	{
		RD::Uniform u;
		u.type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
		u.binding = 1;
		u.ids.push_back(storage->gi_probe_get_data_buffer(gi_probe->probe));
		uniforms.push_back(u);
	}
	{
		RD::Uniform u;
		u.type = RD::UNIFORM_TYPE_TEXTURE;
		u.binding = 2;
		u.ids.push_back(gi_probe->texture);
		uniforms.push_back(u);
	}
	{
		RD::Uniform u;
		u.type = RD::UNIFORM_TYPE_SAMPLER;
		u.binding = 3;
		u.ids.push_back(storage->sampler_rd_get_default(VS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST, VS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED));
		uniforms.push_back(u);
	}

	if (gi_probe_use_anisotropy) {
		{
			RD::Uniform u;
			u.type = RD::UNIFORM_TYPE_TEXTURE;
			u.binding = 4;
			u.ids.push_back(gi_probe->anisotropy[0]);
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.type = RD::UNIFORM_TYPE_TEXTURE;
			u.binding = 5;
			u.ids.push_back(gi_probe->anisotropy[1]);
			uniforms.push_back(u);
		}
	}

	int cell_count;
	if (!p_emission && p_lighting && gi_probe->has_dynamic_object_data) {
		cell_count = push_constant.bounds[0] * push_constant.bounds[1] * push_constant.bounds[2];
	} else {
		cell_count = gi_probe->mipmaps[level].cell_count;
	}

	giprobe_debug_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, giprobe_debug_shader_version_shaders[0], 0);
	RD::get_singleton()->draw_list_bind_render_pipeline(p_draw_list, giprobe_debug_shader_version_pipelines[p_emission ? GI_PROBE_DEBUG_EMISSION : p_lighting ? (gi_probe->has_dynamic_object_data ? GI_PROBE_DEBUG_LIGHT_FULL : GI_PROBE_DEBUG_LIGHT) : GI_PROBE_DEBUG_COLOR].get_render_pipeline(RD::INVALID_ID, RD::get_singleton()->framebuffer_get_format(p_framebuffer)));
	RD::get_singleton()->draw_list_bind_uniform_set(p_draw_list, giprobe_debug_uniform_set, 0);
	RD::get_singleton()->draw_list_set_push_constant(p_draw_list, &push_constant, sizeof(GIProbeDebugPushConstant));
	RD::get_singleton()->draw_list_draw(p_draw_list, false, cell_count, 36);
}

const Vector<RID> &RasterizerSceneRD::gi_probe_get_slots() const {

	return gi_probe_slots;
}

RasterizerSceneRD::GIProbeQuality RasterizerSceneRD::gi_probe_get_quality() const {
	return gi_probe_quality;
}

////////////////////////////////
RID RasterizerSceneRD::render_buffers_create() {
	RenderBuffers rb;
	rb.data = _create_render_buffer_data();
	return render_buffers_owner.make_rid(rb);
}

void RasterizerSceneRD::_allocate_blur_textures(RenderBuffers *rb) {
	ERR_FAIL_COND(!rb->blur[0].texture.is_null());

	uint32_t mipmaps_required = Image::get_image_required_mipmaps(rb->width, rb->height, Image::FORMAT_RGBAH);

	RD::TextureFormat tf;
	tf.format = RD::DATA_FORMAT_R16G16B16A16_SFLOAT;
	tf.width = rb->width;
	tf.height = rb->height;
	tf.type = RD::TEXTURE_TYPE_2D;
	tf.usage_bits = RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT | RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_COPY_TO_BIT;
	tf.mipmaps = mipmaps_required;

	rb->blur[0].texture = RD::get_singleton()->texture_create(tf, RD::TextureView());
	//the second one is smaller (only used for separatable part of blur)
	tf.width >>= 1;
	tf.height >>= 1;
	tf.mipmaps--;
	rb->blur[1].texture = RD::get_singleton()->texture_create(tf, RD::TextureView());

	int base_width = rb->width;
	int base_height = rb->height;

	for (uint32_t i = 0; i < mipmaps_required; i++) {

		RenderBuffers::Blur::Mipmap mm;
		mm.texture = RD::get_singleton()->texture_create_shared_from_slice(RD::TextureView(), rb->blur[0].texture, 0, i);
		{
			Vector<RID> fbs;
			fbs.push_back(mm.texture);
			mm.framebuffer = RD::get_singleton()->framebuffer_create(fbs);
		}

		mm.width = base_width;
		mm.height = base_height;

		rb->blur[0].mipmaps.push_back(mm);

		if (i > 0) {

			mm.texture = RD::get_singleton()->texture_create_shared_from_slice(RD::TextureView(), rb->blur[1].texture, 0, i - 1);
			{
				Vector<RID> fbs;
				fbs.push_back(mm.texture);
				mm.framebuffer = RD::get_singleton()->framebuffer_create(fbs);
			}

			rb->blur[1].mipmaps.push_back(mm);
		}

		base_width = MAX(1, base_width >> 1);
		base_height = MAX(1, base_height >> 1);
	}
}

void RasterizerSceneRD::_allocate_luminance_textures(RenderBuffers *rb) {
	ERR_FAIL_COND(!rb->luminance.current.is_null());

	int w = rb->width;
	int h = rb->height;

	while (true) {
		w = MAX(w / 8, 1);
		h = MAX(h / 8, 1);

		RD::TextureFormat tf;
		tf.format = RD::DATA_FORMAT_R32_SFLOAT;
		tf.width = w;
		tf.height = h;
		tf.usage_bits = RD::TEXTURE_USAGE_STORAGE_BIT;

		bool final = w == 1 && h == 1;

		if (final) {
			tf.usage_bits |= RD::TEXTURE_USAGE_SAMPLING_BIT;
		}

		RID texture = RD::get_singleton()->texture_create(tf, RD::TextureView());

		rb->luminance.reduce.push_back(texture);

		if (final) {
			rb->luminance.current = RD::get_singleton()->texture_create(tf, RD::TextureView());
			break;
		}
	}
}

void RasterizerSceneRD::_free_render_buffer_data(RenderBuffers *rb) {

	if (rb->texture.is_valid()) {
		RD::get_singleton()->free(rb->texture);
		rb->texture = RID();
	}

	if (rb->depth_texture.is_valid()) {
		RD::get_singleton()->free(rb->depth_texture);
		rb->depth_texture = RID();
	}

	for (int i = 0; i < 2; i++) {
		if (rb->blur[i].texture.is_valid()) {
			RD::get_singleton()->free(rb->blur[i].texture);
			rb->blur[i].texture = RID();
			rb->blur[i].mipmaps.clear();
		}
	}

	for (int i = 0; i < rb->luminance.reduce.size(); i++) {
		RD::get_singleton()->free(rb->luminance.reduce[i]);
	}

	for (int i = 0; i < rb->luminance.reduce.size(); i++) {
		RD::get_singleton()->free(rb->luminance.reduce[i]);
	}
	rb->luminance.reduce.clear();

	if (rb->luminance.current.is_valid()) {
		RD::get_singleton()->free(rb->luminance.current);
		rb->luminance.current = RID();
	}

	if (rb->ssao.ao[0].is_valid()) {
		RD::get_singleton()->free(rb->ssao.depth);
		RD::get_singleton()->free(rb->ssao.ao[0]);
		if (rb->ssao.ao[1].is_valid()) {
			RD::get_singleton()->free(rb->ssao.ao[1]);
		}
		if (rb->ssao.ao_full.is_valid()) {
			RD::get_singleton()->free(rb->ssao.ao_full);
		}

		rb->ssao.depth = RID();
		rb->ssao.ao[0] = RID();
		rb->ssao.ao[1] = RID();
		rb->ssao.ao_full = RID();
		rb->ssao.depth_slices.clear();
	}
}

void RasterizerSceneRD::_process_ssao(RID p_render_buffers, RID p_environment, RID p_normal_buffer, const CameraMatrix &p_projection) {

	RenderBuffers *rb = render_buffers_owner.getornull(p_render_buffers);
	ERR_FAIL_COND(!rb);

	Environent *env = environment_owner.getornull(p_environment);
	ERR_FAIL_COND(!env);

	if (rb->ssao.ao[0].is_valid() && rb->ssao.ao_full.is_valid() != ssao_half_size) {
		RD::get_singleton()->free(rb->ssao.depth);
		RD::get_singleton()->free(rb->ssao.ao[0]);
		if (rb->ssao.ao[1].is_valid()) {
			RD::get_singleton()->free(rb->ssao.ao[1]);
		}
		if (rb->ssao.ao_full.is_valid()) {
			RD::get_singleton()->free(rb->ssao.ao_full);
		}

		rb->ssao.depth = RID();
		rb->ssao.ao[0] = RID();
		rb->ssao.ao[1] = RID();
		rb->ssao.ao_full = RID();
		rb->ssao.depth_slices.clear();
	}

	if (!rb->ssao.ao[0].is_valid()) {
		//allocate depth slices

		{
			RD::TextureFormat tf;
			tf.format = RD::DATA_FORMAT_R32_SFLOAT;
			tf.width = rb->width / 2;
			tf.height = rb->height / 2;
			tf.mipmaps = Image::get_image_required_mipmaps(tf.width, tf.height, Image::FORMAT_RF) + 1;
			tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT;
			rb->ssao.depth = RD::get_singleton()->texture_create(tf, RD::TextureView());
			for (uint32_t i = 0; i < tf.mipmaps; i++) {
				RID slice = RD::get_singleton()->texture_create_shared_from_slice(RD::TextureView(), rb->ssao.depth, 0, i);
				rb->ssao.depth_slices.push_back(slice);
			}
		}

		{
			RD::TextureFormat tf;
			tf.format = RD::DATA_FORMAT_R8_UNORM;
			tf.width = ssao_half_size ? rb->width / 2 : rb->width;
			tf.height = ssao_half_size ? rb->height / 2 : rb->height;
			tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT;
			rb->ssao.ao[0] = RD::get_singleton()->texture_create(tf, RD::TextureView());
			rb->ssao.ao[1] = RD::get_singleton()->texture_create(tf, RD::TextureView());
		}

		if (ssao_half_size) {
			//upsample texture
			RD::TextureFormat tf;
			tf.format = RD::DATA_FORMAT_R8_UNORM;
			tf.width = rb->width;
			tf.height = rb->height;
			tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT;
			rb->ssao.ao_full = RD::get_singleton()->texture_create(tf, RD::TextureView());
		}

		_render_buffers_uniform_set_changed(p_render_buffers);
	}

	storage->get_effects()->generate_ssao(rb->depth_texture, p_normal_buffer, Size2i(rb->width, rb->height), rb->ssao.depth, rb->ssao.depth_slices, rb->ssao.ao[0], rb->ssao.ao_full.is_valid(), rb->ssao.ao[1], rb->ssao.ao_full, env->ssao_intensity, env->ssao_radius, env->ssao_bias, p_projection, ssao_quality, env->ssao_blur, env->ssao_blur_edge_sharpness);
}

void RasterizerSceneRD::_render_buffers_post_process_and_tonemap(RID p_render_buffers, RID p_environment, RID p_camera_effects, const CameraMatrix &p_projection) {

	RenderBuffers *rb = render_buffers_owner.getornull(p_render_buffers);
	ERR_FAIL_COND(!rb);

	Environent *env = environment_owner.getornull(p_environment);
	//glow (if enabled)
	CameraEffects *camfx = camera_effects_owner.getornull(p_camera_effects);

	bool can_use_effects = rb->width >= 8 && rb->height >= 8;

	if (can_use_effects && camfx && (camfx->dof_blur_near_enabled || camfx->dof_blur_far_enabled) && camfx->dof_blur_amount > 0.0) {

		if (rb->blur[0].texture.is_null()) {
			_allocate_blur_textures(rb);
			_render_buffers_uniform_set_changed(p_render_buffers);
		}

		float bokeh_size = camfx->dof_blur_amount * 64.0;
		storage->get_effects()->bokeh_dof(rb->texture, rb->depth_texture, Size2i(rb->width, rb->height), rb->blur[0].mipmaps[0].texture, rb->blur[1].mipmaps[0].texture, rb->blur[0].mipmaps[1].texture, camfx->dof_blur_far_enabled, camfx->dof_blur_far_distance, camfx->dof_blur_far_transition, camfx->dof_blur_near_enabled, camfx->dof_blur_near_distance, camfx->dof_blur_near_transition, bokeh_size, dof_blur_bokeh_shape, dof_blur_quality, dof_blur_use_jitter, p_projection.get_z_near(), p_projection.get_z_far(), p_projection.is_orthogonal());
	}

	if (can_use_effects && env && env->auto_exposure) {

		if (rb->luminance.current.is_null()) {
			_allocate_luminance_textures(rb);
			_render_buffers_uniform_set_changed(p_render_buffers);
		}

		bool set_immediate = env->auto_exposure_version != rb->auto_exposure_version;
		rb->auto_exposure_version = env->auto_exposure_version;

		double step = env->auto_exp_speed * time_step;
		storage->get_effects()->luminance_reduction(rb->texture, Size2i(rb->width, rb->height), rb->luminance.reduce, rb->luminance.current, env->min_luminance, env->max_luminance, step, set_immediate);

		//swap final reduce with prev luminance
		SWAP(rb->luminance.current, rb->luminance.reduce.write[rb->luminance.reduce.size() - 1]);
		VisualServerRaster::redraw_request(); //redraw all the time if auto exposure rendering is on
	}

	int max_glow_level = -1;
	int glow_mask = 0;

	if (can_use_effects && env && env->glow_enabled) {

		/* see that blur textures are allocated */

		if (rb->blur[0].texture.is_null()) {
			_allocate_blur_textures(rb);
			_render_buffers_uniform_set_changed(p_render_buffers);
		}

		for (int i = 0; i < VS::MAX_GLOW_LEVELS; i++) {
			if (env->glow_levels & (1 << i)) {

				if (i >= rb->blur[1].mipmaps.size()) {
					max_glow_level = rb->blur[1].mipmaps.size() - 1;
					glow_mask |= 1 << max_glow_level;

				} else {
					max_glow_level = i;
					glow_mask |= (1 << i);
				}
			}
		}

		for (int i = 0; i < (max_glow_level + 1); i++) {

			int vp_w = rb->blur[1].mipmaps[i].width;
			int vp_h = rb->blur[1].mipmaps[i].height;

			if (i == 0) {
				RID luminance_texture;
				if (env->auto_exposure && rb->luminance.current.is_valid()) {
					luminance_texture = rb->luminance.current;
				}
				storage->get_effects()->gaussian_glow(rb->texture, rb->blur[0].mipmaps[i + 1].framebuffer, rb->blur[0].mipmaps[i + 1].texture, rb->blur[1].mipmaps[i].framebuffer, Vector2(1.0 / vp_w, 1.0 / vp_h), env->glow_strength, true, env->glow_hdr_luminance_cap, env->exposure, env->glow_bloom, env->glow_hdr_bleed_threshold, env->glow_hdr_bleed_scale, luminance_texture, env->auto_exp_scale);
			} else {
				storage->get_effects()->gaussian_glow(rb->blur[1].mipmaps[i - 1].texture, rb->blur[0].mipmaps[i + 1].framebuffer, rb->blur[0].mipmaps[i + 1].texture, rb->blur[1].mipmaps[i].framebuffer, Vector2(1.0 / vp_w, 1.0 / vp_h), env->glow_strength);
			}
		}
	}

	{
		//tonemap
		RasterizerEffectsRD::TonemapSettings tonemap;

		tonemap.color_correction_texture = storage->texture_rd_get_default(RasterizerStorageRD::DEFAULT_RD_TEXTURE_3D_WHITE);

		if (can_use_effects && env && env->auto_exposure && rb->luminance.current.is_valid()) {
			tonemap.use_auto_exposure = true;
			tonemap.exposure_texture = rb->luminance.current;
			tonemap.auto_exposure_grey = env->auto_exp_scale;
		} else {

			tonemap.exposure_texture = storage->texture_rd_get_default(RasterizerStorageRD::DEFAULT_RD_TEXTURE_WHITE);
		}

		if (can_use_effects && env && env->glow_enabled) {
			tonemap.use_glow = true;
			tonemap.glow_mode = RasterizerEffectsRD::TonemapSettings::GlowMode(env->glow_blend_mode);
			tonemap.glow_intensity = env->glow_blend_mode == VS::ENV_GLOW_BLEND_MODE_MIX ? env->glow_mix : env->glow_intensity;
			tonemap.glow_level_flags = glow_mask;
			tonemap.glow_texture_size.x = rb->blur[1].mipmaps[0].width;
			tonemap.glow_texture_size.y = rb->blur[1].mipmaps[0].height;
			tonemap.glow_use_bicubic_upscale = env->glow_bicubic_upscale;
			tonemap.glow_texture = rb->blur[1].texture;
		} else {
			tonemap.glow_texture = storage->texture_rd_get_default(RasterizerStorageRD::DEFAULT_RD_TEXTURE_BLACK);
		}

		if (env) {
			tonemap.tonemap_mode = env->tone_mapper;
			tonemap.white = env->white;
			tonemap.exposure = env->exposure;
		}

		storage->get_effects()->tonemapper(rb->texture, storage->render_target_get_rd_framebuffer(rb->render_target), tonemap);
	}

	storage->render_target_disable_clear_request(rb->render_target);
}

void RasterizerSceneRD::_render_buffers_debug_draw(RID p_render_buffers, RID p_shadow_atlas) {
	RasterizerEffectsRD *effects = storage->get_effects();

	RenderBuffers *rb = render_buffers_owner.getornull(p_render_buffers);
	ERR_FAIL_COND(!rb);

	if (debug_draw == VS::VIEWPORT_DEBUG_DRAW_SHADOW_ATLAS) {
		if (p_shadow_atlas.is_valid()) {
			RID shadow_atlas_texture = shadow_atlas_get_texture(p_shadow_atlas);
			Size2 rtsize = storage->render_target_get_size(rb->render_target);

			effects->copy_to_rect(shadow_atlas_texture, storage->render_target_get_rd_framebuffer(rb->render_target), Rect2(Vector2(), rtsize / 2), false, true);
		}
	}

	if (debug_draw == VS::VIEWPORT_DEBUG_DRAW_DIRECTIONAL_SHADOW_ATLAS) {
		if (directional_shadow_get_texture().is_valid()) {
			RID shadow_atlas_texture = directional_shadow_get_texture();
			Size2 rtsize = storage->render_target_get_size(rb->render_target);

			effects->copy_to_rect(shadow_atlas_texture, storage->render_target_get_rd_framebuffer(rb->render_target), Rect2(Vector2(), rtsize / 2), false, true);
		}
	}

	if (debug_draw == VS::VIEWPORT_DEBUG_DRAW_SCENE_LUMINANCE) {
		if (rb->luminance.current.is_valid()) {
			Size2 rtsize = storage->render_target_get_size(rb->render_target);

			effects->copy_to_rect(rb->luminance.current, storage->render_target_get_rd_framebuffer(rb->render_target), Rect2(Vector2(), rtsize / 8), false, true);
		}
	}

	if (debug_draw == VS::VIEWPORT_DEBUG_DRAW_SSAO && rb->ssao.ao[0].is_valid()) {
		Size2 rtsize = storage->render_target_get_size(rb->render_target);
		RID ao_buf = rb->ssao.ao_full.is_valid() ? rb->ssao.ao_full : rb->ssao.ao[0];
		effects->copy_to_rect(ao_buf, storage->render_target_get_rd_framebuffer(rb->render_target), Rect2(Vector2(), rtsize), false, true);
	}

	if (debug_draw == VS::VIEWPORT_DEBUG_DRAW_ROUGHNESS_LIMITER && _render_buffers_get_roughness_texture(p_render_buffers).is_valid()) {
		Size2 rtsize = storage->render_target_get_size(rb->render_target);
		effects->copy_to_rect(_render_buffers_get_roughness_texture(p_render_buffers), storage->render_target_get_rd_framebuffer(rb->render_target), Rect2(Vector2(), rtsize), false, true);
	}

	if (debug_draw == VS::VIEWPORT_DEBUG_DRAW_NORMAL_BUFFER && _render_buffers_get_normal_texture(p_render_buffers).is_valid()) {
		Size2 rtsize = storage->render_target_get_size(rb->render_target);
		effects->copy_to_rect(_render_buffers_get_normal_texture(p_render_buffers), storage->render_target_get_rd_framebuffer(rb->render_target), Rect2(Vector2(), rtsize));
	}
}

RID RasterizerSceneRD::render_buffers_get_back_buffer_texture(RID p_render_buffers) {

	RenderBuffers *rb = render_buffers_owner.getornull(p_render_buffers);
	ERR_FAIL_COND_V(!rb, RID());
	if (!rb->blur[0].texture.is_valid()) {
		return RID(); //not valid at the moment
	}
	return rb->blur[0].texture;
}

RID RasterizerSceneRD::render_buffers_get_ao_texture(RID p_render_buffers) {
	RenderBuffers *rb = render_buffers_owner.getornull(p_render_buffers);
	ERR_FAIL_COND_V(!rb, RID());

	return rb->ssao.ao_full.is_valid() ? rb->ssao.ao_full : rb->ssao.ao[0];
}

void RasterizerSceneRD::render_buffers_configure(RID p_render_buffers, RID p_render_target, int p_width, int p_height, VS::ViewportMSAA p_msaa) {

	RenderBuffers *rb = render_buffers_owner.getornull(p_render_buffers);
	rb->width = p_width;
	rb->height = p_height;
	rb->render_target = p_render_target;
	rb->msaa = p_msaa;
	_free_render_buffer_data(rb);

	{
		RD::TextureFormat tf;
		tf.format = RD::DATA_FORMAT_R16G16B16A16_SFLOAT;
		tf.width = rb->width;
		tf.height = rb->height;
		tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT;

		rb->texture = RD::get_singleton()->texture_create(tf, RD::TextureView());
	}

	{
		RD::TextureFormat tf;
		tf.format = RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_D24_UNORM_S8_UINT, RD::TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) ? RD::DATA_FORMAT_D24_UNORM_S8_UINT : RD::DATA_FORMAT_D32_SFLOAT_S8_UINT;
		tf.width = p_width;
		tf.height = p_height;
		tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

		rb->depth_texture = RD::get_singleton()->texture_create(tf, RD::TextureView());
	}

	rb->data->configure(rb->texture, rb->depth_texture, p_width, p_height, p_msaa);
	_render_buffers_uniform_set_changed(p_render_buffers);
}

int RasterizerSceneRD::get_roughness_layers() const {
	return roughness_layers;
}

bool RasterizerSceneRD::is_using_radiance_cubemap_array() const {
	return sky_use_cubemap_array;
}

RasterizerSceneRD::RenderBufferData *RasterizerSceneRD::render_buffers_get_data(RID p_render_buffers) {
	RenderBuffers *rb = render_buffers_owner.getornull(p_render_buffers);
	ERR_FAIL_COND_V(!rb, NULL);
	return rb->data;
}

void RasterizerSceneRD::render_scene(RID p_render_buffers, const Transform &p_cam_transform, const CameraMatrix &p_cam_projection, bool p_cam_ortogonal, InstanceBase **p_cull_result, int p_cull_count, RID *p_light_cull_result, int p_light_cull_count, RID *p_reflection_probe_cull_result, int p_reflection_probe_cull_count, RID *p_gi_probe_cull_result, int p_gi_probe_cull_count, RID p_environment, RID p_camera_effects, RID p_shadow_atlas, RID p_reflection_atlas, RID p_reflection_probe, int p_reflection_probe_pass) {

	Color clear_color;
	if (p_render_buffers.is_valid()) {
		RenderBuffers *rb = render_buffers_owner.getornull(p_render_buffers);
		ERR_FAIL_COND(!rb);
		clear_color = storage->render_target_get_clear_request_color(rb->render_target);
	} else {
		clear_color = storage->get_default_clear_color();
	}

	_render_scene(p_render_buffers, p_cam_transform, p_cam_projection, p_cam_ortogonal, p_cull_result, p_cull_count, p_light_cull_result, p_light_cull_count, p_reflection_probe_cull_result, p_reflection_probe_cull_count, p_gi_probe_cull_result, p_gi_probe_cull_count, p_environment, p_camera_effects, p_shadow_atlas, p_reflection_atlas, p_reflection_probe, p_reflection_probe_pass, clear_color);

	if (p_render_buffers.is_valid()) {
		RENDER_TIMESTAMP("Tonemap");

		_render_buffers_post_process_and_tonemap(p_render_buffers, p_environment, p_camera_effects, p_cam_projection);
		_render_buffers_debug_draw(p_render_buffers, p_shadow_atlas);
	}
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
			light_instance->directional_rect = _get_directional_shadow_rect(directional_shadow.size, directional_shadow.light_count, directional_shadow.current_light);
			directional_shadow.current_light++;
			light_instance->last_scene_shadow_pass = scene_pass;
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

		light_instance->shadow_transform[p_pass].atlas_rect = atlas_rect;

		light_instance->shadow_transform[p_pass].atlas_rect.position /= directional_shadow.size;
		light_instance->shadow_transform[p_pass].atlas_rect.size /= directional_shadow.size;

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

void RasterizerSceneRD::render_material(const Transform &p_cam_transform, const CameraMatrix &p_cam_projection, bool p_cam_ortogonal, InstanceBase **p_cull_result, int p_cull_count, RID p_framebuffer, const Rect2i &p_region) {

	_render_material(p_cam_transform, p_cam_projection, p_cam_ortogonal, p_cull_result, p_cull_count, p_framebuffer, p_region);
}

bool RasterizerSceneRD::free(RID p_rid) {

	if (render_buffers_owner.owns(p_rid)) {
		RenderBuffers *rb = render_buffers_owner.getornull(p_rid);
		_free_render_buffer_data(rb);
		memdelete(rb->data);
		render_buffers_owner.free(p_rid);
	} else if (environment_owner.owns(p_rid)) {
		//not much to delete, just free it
		environment_owner.free(p_rid);
	} else if (camera_effects_owner.owns(p_rid)) {
		//not much to delete, just free it
		camera_effects_owner.free(p_rid);
	} else if (reflection_atlas_owner.owns(p_rid)) {
		reflection_atlas_set_size(p_rid, 0, 0);
		reflection_atlas_owner.free(p_rid);
	} else if (reflection_probe_instance_owner.owns(p_rid)) {
		//not much to delete, just free it
		//ReflectionProbeInstance *rpi = reflection_probe_instance_owner.getornull(p_rid);
		reflection_probe_release_atlas_index(p_rid);
		reflection_probe_instance_owner.free(p_rid);
	} else if (gi_probe_instance_owner.owns(p_rid)) {
		GIProbeInstance *gi_probe = gi_probe_instance_owner.getornull(p_rid);
		if (gi_probe->texture.is_valid()) {
			RD::get_singleton()->free(gi_probe->texture);
			RD::get_singleton()->free(gi_probe->write_buffer);
		}
		if (gi_probe->anisotropy[0].is_valid()) {
			RD::get_singleton()->free(gi_probe->anisotropy[0]);
			RD::get_singleton()->free(gi_probe->anisotropy[1]);
		}

		for (int i = 0; i < gi_probe->dynamic_maps.size(); i++) {
			RD::get_singleton()->free(gi_probe->dynamic_maps[i].texture);
			RD::get_singleton()->free(gi_probe->dynamic_maps[i].depth);
		}

		gi_probe_slots.write[gi_probe->slot] = RID();

		gi_probe_instance_owner.free(p_rid);
	} else if (sky_owner.owns(p_rid)) {
		_update_dirty_skys();
		Sky *sky = sky_owner.getornull(p_rid);
		if (sky->radiance.is_valid()) {
			RD::get_singleton()->free(sky->radiance);
			sky->radiance = RID();
		}
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

void RasterizerSceneRD::set_debug_draw_mode(VS::ViewportDebugDraw p_debug_draw) {
	debug_draw = p_debug_draw;
}

void RasterizerSceneRD::update() {
	_update_dirty_skys();
}

void RasterizerSceneRD::set_time(double p_time, double p_step) {
	time_step = p_step;
}

void RasterizerSceneRD::screen_space_roughness_limiter_set_active(bool p_enable, float p_curve) {
	screen_space_roughness_limiter = p_enable;
	screen_space_roughness_limiter_curve = p_curve;
}

bool RasterizerSceneRD::screen_space_roughness_limiter_is_active() const {
	return screen_space_roughness_limiter;
}

float RasterizerSceneRD::screen_space_roughness_limiter_get_curve() const {
	return screen_space_roughness_limiter_curve;
}

RasterizerSceneRD::RasterizerSceneRD(RasterizerStorageRD *p_storage) {
	storage = p_storage;

	roughness_layers = GLOBAL_GET("rendering/quality/reflections/roughness_layers");
	sky_ggx_samples_quality = GLOBAL_GET("rendering/quality/reflections/ggx_samples");
	sky_ggx_samples_realtime = GLOBAL_GET("rendering/quality/reflections/ggx_samples_realtime");
	sky_use_cubemap_array = GLOBAL_GET("rendering/quality/reflections/texture_array_reflections");
	//	sky_use_cubemap_array = false;

	uint32_t textures_per_stage = RD::get_singleton()->limit_get(RD::LIMIT_MAX_TEXTURES_PER_SHADER_STAGE);

	{

		//kinda complicated to compute the amount of slots, we try to use as many as we can

		gi_probe_max_lights = 32;

		gi_probe_lights = memnew_arr(GIProbeLight, gi_probe_max_lights);
		gi_probe_lights_uniform = RD::get_singleton()->uniform_buffer_create(gi_probe_max_lights * sizeof(GIProbeLight));

		gi_probe_use_anisotropy = GLOBAL_GET("rendering/quality/gi_probes/anisotropic");
		gi_probe_quality = GIProbeQuality(CLAMP(int(GLOBAL_GET("rendering/quality/gi_probes/quality")), 0, 2));

		if (textures_per_stage <= 16) {
			gi_probe_slots.resize(2); //thats all you can get
			gi_probe_use_anisotropy = false;
		} else if (textures_per_stage <= 31) {
			gi_probe_slots.resize(4); //thats all you can get, iOS
			gi_probe_use_anisotropy = false;
		} else if (textures_per_stage <= 128) {
			gi_probe_slots.resize(32); //old intel
			gi_probe_use_anisotropy = false;
		} else if (textures_per_stage <= 256) {
			gi_probe_slots.resize(64); //old intel too
			gi_probe_use_anisotropy = false;
		} else {
			if (gi_probe_use_anisotropy) {
				gi_probe_slots.resize(1024 / 3); //needs 3 textures
			} else {
				gi_probe_slots.resize(1024); //modern intel, nvidia, 8192 or greater
			}
		}

		String defines = "\n#define MAX_LIGHTS " + itos(gi_probe_max_lights) + "\n";
		if (gi_probe_use_anisotropy) {
			defines += "\n#define MODE_ANISOTROPIC\n";
		}

		Vector<String> versions;
		versions.push_back("\n#define MODE_COMPUTE_LIGHT\n");
		versions.push_back("\n#define MODE_SECOND_BOUNCE\n");
		versions.push_back("\n#define MODE_UPDATE_MIPMAPS\n");
		versions.push_back("\n#define MODE_WRITE_TEXTURE\n");
		versions.push_back("\n#define MODE_DYNAMIC\n#define MODE_DYNAMIC_LIGHTING\n");
		versions.push_back("\n#define MODE_DYNAMIC\n#define MODE_DYNAMIC_SHRINK\n#define MODE_DYNAMIC_SHRINK_WRITE\n");
		versions.push_back("\n#define MODE_DYNAMIC\n#define MODE_DYNAMIC_SHRINK\n#define MODE_DYNAMIC_SHRINK_PLOT\n");
		versions.push_back("\n#define MODE_DYNAMIC\n#define MODE_DYNAMIC_SHRINK\n#define MODE_DYNAMIC_SHRINK_PLOT\n#define MODE_DYNAMIC_SHRINK_WRITE\n");

		giprobe_shader.initialize(versions, defines);
		giprobe_lighting_shader_version = giprobe_shader.version_create();
		for (int i = 0; i < GI_PROBE_SHADER_VERSION_MAX; i++) {
			giprobe_lighting_shader_version_shaders[i] = giprobe_shader.version_get_shader(giprobe_lighting_shader_version, i);
			giprobe_lighting_shader_version_pipelines[i] = RD::get_singleton()->compute_pipeline_create(giprobe_lighting_shader_version_shaders[i]);
		}
	}

	{

		String defines;
		if (gi_probe_use_anisotropy) {
			defines += "\n#define USE_ANISOTROPY\n";
		}
		Vector<String> versions;
		versions.push_back("\n#define MODE_DEBUG_COLOR\n");
		versions.push_back("\n#define MODE_DEBUG_LIGHT\n");
		versions.push_back("\n#define MODE_DEBUG_EMISSION\n");
		versions.push_back("\n#define MODE_DEBUG_LIGHT\n#define MODE_DEBUG_LIGHT_FULL\n");

		giprobe_debug_shader.initialize(versions, defines);
		giprobe_debug_shader_version = giprobe_debug_shader.version_create();
		for (int i = 0; i < GI_PROBE_DEBUG_MAX; i++) {
			giprobe_debug_shader_version_shaders[i] = giprobe_debug_shader.version_get_shader(giprobe_debug_shader_version, i);

			RD::PipelineRasterizationState rs;
			rs.cull_mode = RD::POLYGON_CULL_FRONT;
			RD::PipelineDepthStencilState ds;
			ds.enable_depth_test = true;
			ds.enable_depth_write = true;
			ds.depth_compare_operator = RD::COMPARE_OP_LESS_OR_EQUAL;

			giprobe_debug_shader_version_pipelines[i].setup(giprobe_debug_shader_version_shaders[i], RD::RENDER_PRIMITIVE_TRIANGLES, rs, RD::PipelineMultisampleState(), ds, RD::PipelineColorBlendState::create_disabled(), 0);
		}
	}

	camera_effects_set_dof_blur_bokeh_shape(VS::DOFBokehShape(int(GLOBAL_GET("rendering/quality/filters/depth_of_field_bokeh_shape"))));
	camera_effects_set_dof_blur_quality(VS::DOFBlurQuality(int(GLOBAL_GET("rendering/quality/filters/depth_of_field_bokeh_quality"))), GLOBAL_GET("rendering/quality/filters/depth_of_field_use_jitter"));
	environment_set_ssao_quality(VS::EnvironmentSSAOQuality(int(GLOBAL_GET("rendering/quality/ssao/quality"))), GLOBAL_GET("rendering/quality/ssao/half_size"));
	screen_space_roughness_limiter = GLOBAL_GET("rendering/quality/filters/screen_space_roughness_limiter");
	screen_space_roughness_limiter_curve = GLOBAL_GET("rendering/quality/filters/screen_space_roughness_limiter_curve");
}

RasterizerSceneRD::~RasterizerSceneRD() {
	for (Map<Vector2i, ShadowMap>::Element *E = shadow_maps.front(); E; E = E->next()) {
		RD::get_singleton()->free(E->get().depth);
	}
	for (Map<int, ShadowCubemap>::Element *E = shadow_cubemaps.front(); E; E = E->next()) {
		RD::get_singleton()->free(E->get().cubemap);
	}

	RD::get_singleton()->free(gi_probe_lights_uniform);
	memdelete_arr(gi_probe_lights);
}
