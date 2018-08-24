/*************************************************************************/
/*  rasterizer_scene_gles2.cpp                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "rasterizer_scene_gles2.h"
#include "math/transform.h"
#include "math_funcs.h"
#include "os/os.h"
#include "project_settings.h"
#include "rasterizer_canvas_gles2.h"
#include "servers/visual/visual_server_raster.h"

#include "vmap.h"

#ifndef GLES_OVER_GL
#define glClearDepth glClearDepthf
#endif

static const GLenum _cube_side_enum[6] = {

	GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
	GL_TEXTURE_CUBE_MAP_POSITIVE_X,
	GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
	GL_TEXTURE_CUBE_MAP_POSITIVE_Y,
	GL_TEXTURE_CUBE_MAP_NEGATIVE_Z,
	GL_TEXTURE_CUBE_MAP_POSITIVE_Z,

};

/* SHADOW ATLAS API */

RID RasterizerSceneGLES2::shadow_atlas_create() {

	ShadowAtlas *shadow_atlas = memnew(ShadowAtlas);
	shadow_atlas->fbo = 0;
	shadow_atlas->depth = 0;
	shadow_atlas->size = 0;
	shadow_atlas->smallest_subdiv = 0;

	for (int i = 0; i < 4; i++) {
		shadow_atlas->size_order[i] = i;
	}

	return shadow_atlas_owner.make_rid(shadow_atlas);
}

void RasterizerSceneGLES2::shadow_atlas_set_size(RID p_atlas, int p_size) {
	ShadowAtlas *shadow_atlas = shadow_atlas_owner.getornull(p_atlas);
	ERR_FAIL_COND(!shadow_atlas);
	ERR_FAIL_COND(p_size < 0);

	p_size = next_power_of_2(p_size);

	if (p_size == shadow_atlas->size)
		return;

	// erase the old atlast
	if (shadow_atlas->fbo) {
		glDeleteTextures(1, &shadow_atlas->depth);
		glDeleteFramebuffers(1, &shadow_atlas->fbo);

		shadow_atlas->fbo = 0;
		shadow_atlas->depth = 0;
	}

	// erase shadow atlast references from lights
	for (Map<RID, uint32_t>::Element *E = shadow_atlas->shadow_owners.front(); E; E = E->next()) {
		LightInstance *li = light_instance_owner.getornull(E->key());
		ERR_CONTINUE(!li);
		li->shadow_atlases.erase(p_atlas);
	}

	shadow_atlas->shadow_owners.clear();

	shadow_atlas->size = p_size;

	if (shadow_atlas->size) {
		glGenFramebuffers(1, &shadow_atlas->fbo);
		glBindFramebuffer(GL_FRAMEBUFFER, shadow_atlas->fbo);

		// create a depth texture
		glActiveTexture(GL_TEXTURE0);
		glGenTextures(1, &shadow_atlas->depth);
		glBindTexture(GL_TEXTURE_2D, shadow_atlas->depth);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, shadow_atlas->size, shadow_atlas->size, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, NULL);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, shadow_atlas->depth, 0);

		glViewport(0, 0, shadow_atlas->size, shadow_atlas->size);

		glDepthMask(GL_TRUE);

		glClearDepth(0.0f);
		glClear(GL_DEPTH_BUFFER_BIT);

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}
}

void RasterizerSceneGLES2::shadow_atlas_set_quadrant_subdivision(RID p_atlas, int p_quadrant, int p_subdivision) {
	ShadowAtlas *shadow_atlas = shadow_atlas_owner.getornull(p_atlas);
	ERR_FAIL_COND(!shadow_atlas);
	ERR_FAIL_INDEX(p_quadrant, 4);
	ERR_FAIL_INDEX(p_subdivision, 16384);

	uint32_t subdiv = next_power_of_2(p_subdivision);
	if (subdiv & 0xaaaaaaaa) { // sqrt(subdiv) must be integer
		subdiv <<= 1;
	}

	subdiv = int(Math::sqrt((float)subdiv));

	if (shadow_atlas->quadrants[p_quadrant].shadows.size() == subdiv)
		return;

	// erase all data from the quadrant
	for (int i = 0; i < shadow_atlas->quadrants[p_quadrant].shadows.size(); i++) {
		if (shadow_atlas->quadrants[p_quadrant].shadows[i].owner.is_valid()) {
			shadow_atlas->shadow_owners.erase(shadow_atlas->quadrants[p_quadrant].shadows[i].owner);

			LightInstance *li = light_instance_owner.getornull(shadow_atlas->quadrants[p_quadrant].shadows[i].owner);
			ERR_CONTINUE(!li);
			li->shadow_atlases.erase(p_atlas);
		}
	}

	shadow_atlas->quadrants[p_quadrant].shadows.resize(0);
	shadow_atlas->quadrants[p_quadrant].shadows.resize(subdiv);
	shadow_atlas->quadrants[p_quadrant].subdivision = subdiv;

	// cache the smallest subdivision for faster allocations

	shadow_atlas->smallest_subdiv = 1 << 30;

	for (int i = 0; i < 4; i++) {
		if (shadow_atlas->quadrants[i].subdivision) {
			shadow_atlas->smallest_subdiv = MIN(shadow_atlas->smallest_subdiv, shadow_atlas->quadrants[i].subdivision);
		}
	}

	if (shadow_atlas->smallest_subdiv == 1 << 30) {
		shadow_atlas->smallest_subdiv = 0;
	}

	// re-sort the quadrants

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

bool RasterizerSceneGLES2::_shadow_atlas_find_shadow(ShadowAtlas *shadow_atlas, int *p_in_quadrants, int p_quadrant_count, int p_current_subdiv, uint64_t p_tick, int &r_quadrant, int &r_shadow) {

	for (int i = p_quadrant_count - 1; i >= 0; i--) {
		int qidx = p_in_quadrants[i];

		if (shadow_atlas->quadrants[qidx].subdivision == (uint32_t)p_current_subdiv) {
			return false;
		}

		// look for an empty space

		int sc = shadow_atlas->quadrants[qidx].shadows.size();

		ShadowAtlas::Quadrant::Shadow *sarr = shadow_atlas->quadrants[qidx].shadows.ptrw();

		int found_free_idx = -1; // found a free one
		int found_used_idx = -1; // found an existing one, must steal it
		uint64_t min_pass = 0; // pass of the existing one, try to use the least recently

		for (int j = 0; j < sc; j++) {
			if (!sarr[j].owner.is_valid()) {
				found_free_idx = j;
				break;
			}

			LightInstance *sli = light_instance_owner.getornull(sarr[j].owner);
			ERR_CONTINUE(!sli);

			if (sli->last_scene_pass != scene_pass) {

				// was just allocated, don't kill it so soon, wait a bit...

				if (p_tick - sarr[j].alloc_tick < shadow_atlas_realloc_tolerance_msec) {
					continue;
				}

				if (found_used_idx == -1 || sli->last_scene_pass < min_pass) {
					found_used_idx = j;
					min_pass = sli->last_scene_pass;
				}
			}
		}

		if (found_free_idx == -1 && found_used_idx == -1) {
			continue; // nothing found
		}

		if (found_free_idx == -1 && found_used_idx != -1) {
			found_free_idx = found_used_idx;
		}

		r_quadrant = qidx;
		r_shadow = found_free_idx;

		return true;
	}

	return false;
}

bool RasterizerSceneGLES2::shadow_atlas_update_light(RID p_atlas, RID p_light_intance, float p_coverage, uint64_t p_light_version) {

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
	int best_size = -1;
	int best_subdiv = -1;

	for (int i = 0; i < 4; i++) {
		int q = shadow_atlas->size_order[i];
		int sd = shadow_atlas->quadrants[q].subdivision;

		if (sd == 0) {
			continue;
		}

		int max_fit = quad_size / sd;

		if (best_size != -1 && max_fit > best_size) {
			break; // what we asked for is bigger than this.
		}

		valid_quadrants[valid_quadrant_count] = q;
		valid_quadrant_count++;

		best_subdiv = sd;

		if (max_fit >= desired_fit) {
			best_size = max_fit;
		}
	}

	ERR_FAIL_COND_V(valid_quadrant_count == 0, false); // no suitable block available

	uint64_t tick = OS::get_singleton()->get_ticks_msec();

	if (shadow_atlas->shadow_owners.has(p_light_intance)) {
		// light was already known!

		uint32_t key = shadow_atlas->shadow_owners[p_light_intance];
		uint32_t q = (key >> ShadowAtlas::QUADRANT_SHIFT) & 0x3;
		uint32_t s = key & ShadowAtlas::SHADOW_INDEX_MASK;

		bool should_realloc = shadow_atlas->quadrants[q].subdivision != (uint32_t)best_subdiv && (shadow_atlas->quadrants[q].shadows[s].alloc_tick - tick > shadow_atlas_realloc_tolerance_msec);

		bool should_redraw = shadow_atlas->quadrants[q].shadows[s].version != p_light_version;

		if (!should_realloc) {
			shadow_atlas->quadrants[q].shadows.write[s].version = p_light_version;
			return should_redraw;
		}

		int new_quadrant;
		int new_shadow;

		// find a better place

		if (_shadow_atlas_find_shadow(shadow_atlas, valid_quadrants, valid_quadrant_count, shadow_atlas->quadrants[q].subdivision, tick, new_quadrant, new_shadow)) {
			// found a better place

			ShadowAtlas::Quadrant::Shadow *sh = &shadow_atlas->quadrants[new_quadrant].shadows.write[new_shadow];
			if (sh->owner.is_valid()) {
				// it is take but invalid, so we can take it

				shadow_atlas->shadow_owners.erase(sh->owner);
				LightInstance *sli = light_instance_owner.get(sh->owner);
				sli->shadow_atlases.erase(p_atlas);
			}

			// erase previous
			shadow_atlas->quadrants[q].shadows.write[s].version = 0;
			shadow_atlas->quadrants[q].shadows.write[s].owner = RID();

			sh->owner = p_light_intance;
			sh->alloc_tick = tick;
			sh->version = p_light_version;
			li->shadow_atlases.insert(p_atlas);

			// make a new key
			key = new_quadrant << ShadowAtlas::QUADRANT_SHIFT;
			key |= new_shadow;

			// update it in the map
			shadow_atlas->shadow_owners[p_light_intance] = key;

			// make it dirty, so we redraw
			return true;
		}

		// no better place found, so we keep the current place

		shadow_atlas->quadrants[q].shadows.write[s].version = p_light_version;

		return should_redraw;
	}

	int new_quadrant;
	int new_shadow;

	if (_shadow_atlas_find_shadow(shadow_atlas, valid_quadrants, valid_quadrant_count, -1, tick, new_quadrant, new_shadow)) {
		// found a better place

		ShadowAtlas::Quadrant::Shadow *sh = &shadow_atlas->quadrants[new_quadrant].shadows.write[new_shadow];
		if (sh->owner.is_valid()) {
			// it is take but invalid, so we can take it

			shadow_atlas->shadow_owners.erase(sh->owner);
			LightInstance *sli = light_instance_owner.get(sh->owner);
			sli->shadow_atlases.erase(p_atlas);
		}

		sh->owner = p_light_intance;
		sh->alloc_tick = tick;
		sh->version = p_light_version;
		li->shadow_atlases.insert(p_atlas);

		// make a new key
		uint32_t key = new_quadrant << ShadowAtlas::QUADRANT_SHIFT;
		key |= new_shadow;

		// update it in the map
		shadow_atlas->shadow_owners[p_light_intance] = key;

		// make it dirty, so we redraw
		return true;
	}

	return false;
}

void RasterizerSceneGLES2::set_directional_shadow_count(int p_count) {
	directional_shadow.light_count = p_count;
	directional_shadow.current_light = 0;
}

int RasterizerSceneGLES2::get_directional_light_shadow_size(RID p_light_intance) {

	ERR_FAIL_COND_V(directional_shadow.light_count == 0, 0);

	int shadow_size;

	if (directional_shadow.light_count == 1) {
		shadow_size = directional_shadow.size;
	} else {
		shadow_size = directional_shadow.size / 2; //more than 4 not supported anyway
	}

	LightInstance *light_instance = light_instance_owner.getornull(p_light_intance);
	ERR_FAIL_COND_V(!light_instance, 0);

	switch (light_instance->light_ptr->directional_shadow_mode) {
		case VS::LIGHT_DIRECTIONAL_SHADOW_ORTHOGONAL:
			break; //none
		case VS::LIGHT_DIRECTIONAL_SHADOW_PARALLEL_2_SPLITS:
		case VS::LIGHT_DIRECTIONAL_SHADOW_PARALLEL_4_SPLITS:
			shadow_size /= 2;
			break;
	}

	return shadow_size;
}
//////////////////////////////////////////////////////

RID RasterizerSceneGLES2::reflection_atlas_create() {
	return RID();
}

void RasterizerSceneGLES2::reflection_atlas_set_size(RID p_ref_atlas, int p_size) {
}

void RasterizerSceneGLES2::reflection_atlas_set_subdivision(RID p_ref_atlas, int p_subdiv) {
}

////////////////////////////////////////////////////

RID RasterizerSceneGLES2::reflection_probe_instance_create(RID p_probe) {
	return RID();
}

void RasterizerSceneGLES2::reflection_probe_instance_set_transform(RID p_instance, const Transform &p_transform) {
}

void RasterizerSceneGLES2::reflection_probe_release_atlas_index(RID p_instance) {
}

bool RasterizerSceneGLES2::reflection_probe_instance_needs_redraw(RID p_instance) {
	return false;
}

bool RasterizerSceneGLES2::reflection_probe_instance_has_reflection(RID p_instance) {
	return false;
}

bool RasterizerSceneGLES2::reflection_probe_instance_begin_render(RID p_instance, RID p_reflection_atlas) {
	return false;
}

bool RasterizerSceneGLES2::reflection_probe_instance_postprocess_step(RID p_instance) {
	return false;
}

/* ENVIRONMENT API */

RID RasterizerSceneGLES2::environment_create() {

	Environment *env = memnew(Environment);

	return environment_owner.make_rid(env);
}

void RasterizerSceneGLES2::environment_set_background(RID p_env, VS::EnvironmentBG p_bg) {

	Environment *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND(!env);
	env->bg_mode = p_bg;
}

void RasterizerSceneGLES2::environment_set_sky(RID p_env, RID p_sky) {
	Environment *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND(!env);

	env->sky = p_sky;
}

void RasterizerSceneGLES2::environment_set_sky_custom_fov(RID p_env, float p_scale) {
	Environment *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND(!env);

	env->sky_custom_fov = p_scale;
}

void RasterizerSceneGLES2::environment_set_bg_color(RID p_env, const Color &p_color) {
	Environment *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND(!env);

	env->bg_color = p_color;
}

void RasterizerSceneGLES2::environment_set_bg_energy(RID p_env, float p_energy) {
	Environment *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND(!env);

	env->bg_energy = p_energy;
}

void RasterizerSceneGLES2::environment_set_canvas_max_layer(RID p_env, int p_max_layer) {
	Environment *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND(!env);

	env->canvas_max_layer = p_max_layer;
}

void RasterizerSceneGLES2::environment_set_ambient_light(RID p_env, const Color &p_color, float p_energy, float p_sky_contribution) {
	Environment *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND(!env);

	env->ambient_color = p_color;
	env->ambient_energy = p_energy;
	env->ambient_sky_contribution = p_sky_contribution;
}

void RasterizerSceneGLES2::environment_set_dof_blur_far(RID p_env, bool p_enable, float p_distance, float p_transition, float p_amount, VS::EnvironmentDOFBlurQuality p_quality) {
	Environment *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND(!env);
}

void RasterizerSceneGLES2::environment_set_dof_blur_near(RID p_env, bool p_enable, float p_distance, float p_transition, float p_amount, VS::EnvironmentDOFBlurQuality p_quality) {
	Environment *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND(!env);
}

void RasterizerSceneGLES2::environment_set_glow(RID p_env, bool p_enable, int p_level_flags, float p_intensity, float p_strength, float p_bloom_threshold, VS::EnvironmentGlowBlendMode p_blend_mode, float p_hdr_bleed_threshold, float p_hdr_bleed_scale, bool p_bicubic_upscale) {
	Environment *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND(!env);
}

void RasterizerSceneGLES2::environment_set_fog(RID p_env, bool p_enable, float p_begin, float p_end, RID p_gradient_texture) {
	Environment *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND(!env);
}

void RasterizerSceneGLES2::environment_set_ssr(RID p_env, bool p_enable, int p_max_steps, float p_fade_in, float p_fade_out, float p_depth_tolerance, bool p_roughness) {
	Environment *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND(!env);
}

void RasterizerSceneGLES2::environment_set_ssao(RID p_env, bool p_enable, float p_radius, float p_intensity, float p_radius2, float p_intensity2, float p_bias, float p_light_affect, float p_ao_channel_affect, const Color &p_color, VS::EnvironmentSSAOQuality p_quality, VisualServer::EnvironmentSSAOBlur p_blur, float p_bilateral_sharpness) {
	Environment *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND(!env);
}

void RasterizerSceneGLES2::environment_set_tonemap(RID p_env, VS::EnvironmentToneMapper p_tone_mapper, float p_exposure, float p_white, bool p_auto_exposure, float p_min_luminance, float p_max_luminance, float p_auto_exp_speed, float p_auto_exp_scale) {
	Environment *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND(!env);
}

void RasterizerSceneGLES2::environment_set_adjustment(RID p_env, bool p_enable, float p_brightness, float p_contrast, float p_saturation, RID p_ramp) {
	Environment *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND(!env);
}

void RasterizerSceneGLES2::environment_set_fog(RID p_env, bool p_enable, const Color &p_color, const Color &p_sun_color, float p_sun_amount) {
	Environment *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND(!env);
}

void RasterizerSceneGLES2::environment_set_fog_depth(RID p_env, bool p_enable, float p_depth_begin, float p_depth_curve, bool p_transmit, float p_transmit_curve) {
	Environment *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND(!env);
}

void RasterizerSceneGLES2::environment_set_fog_height(RID p_env, bool p_enable, float p_min_height, float p_max_height, float p_height_curve) {
	Environment *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND(!env);
}

bool RasterizerSceneGLES2::is_environment(RID p_env) {
	return environment_owner.owns(p_env);
}

VS::EnvironmentBG RasterizerSceneGLES2::environment_get_background(RID p_env) {
	const Environment *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND_V(!env, VS::ENV_BG_MAX);

	return env->bg_mode;
}

int RasterizerSceneGLES2::environment_get_canvas_max_layer(RID p_env) {
	const Environment *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND_V(!env, -1);

	return env->canvas_max_layer;
}

RID RasterizerSceneGLES2::light_instance_create(RID p_light) {

	LightInstance *light_instance = memnew(LightInstance);

	light_instance->last_scene_pass = 0;

	light_instance->light = p_light;
	light_instance->light_ptr = storage->light_owner.getornull(p_light);

	ERR_FAIL_COND_V(!light_instance->light_ptr, RID());

	light_instance->self = light_instance_owner.make_rid(light_instance);

	return light_instance->self;
}

void RasterizerSceneGLES2::light_instance_set_transform(RID p_light_instance, const Transform &p_transform) {

	LightInstance *light_instance = light_instance_owner.getornull(p_light_instance);
	ERR_FAIL_COND(!light_instance);

	light_instance->transform = p_transform;
}

void RasterizerSceneGLES2::light_instance_set_shadow_transform(RID p_light_instance, const CameraMatrix &p_projection, const Transform &p_transform, float p_far, float p_split, int p_pass, float p_bias_scale) {

	LightInstance *light_instance = light_instance_owner.getornull(p_light_instance);
	ERR_FAIL_COND(!light_instance);

	if (light_instance->light_ptr->type != VS::LIGHT_DIRECTIONAL) {
		p_pass = 0;
	}

	ERR_FAIL_INDEX(p_pass, 4);

	light_instance->shadow_transform[p_pass].camera = p_projection;
	light_instance->shadow_transform[p_pass].transform = p_transform;
	light_instance->shadow_transform[p_pass].farplane = p_far;
	light_instance->shadow_transform[p_pass].split = p_split;
	light_instance->shadow_transform[p_pass].bias_scale = p_bias_scale;
}

void RasterizerSceneGLES2::light_instance_mark_visible(RID p_light_instance) {

	LightInstance *light_instance = light_instance_owner.getornull(p_light_instance);
	ERR_FAIL_COND(!light_instance);

	light_instance->last_scene_pass = scene_pass;
}

//////////////////////

RID RasterizerSceneGLES2::gi_probe_instance_create() {

	return RID();
}

void RasterizerSceneGLES2::gi_probe_instance_set_light_data(RID p_probe, RID p_base, RID p_data) {
}
void RasterizerSceneGLES2::gi_probe_instance_set_transform_to_data(RID p_probe, const Transform &p_xform) {
}

void RasterizerSceneGLES2::gi_probe_instance_set_bounds(RID p_probe, const Vector3 &p_bounds) {
}

////////////////////////////
////////////////////////////
////////////////////////////

void RasterizerSceneGLES2::_add_geometry(RasterizerStorageGLES2::Geometry *p_geometry, InstanceBase *p_instance, RasterizerStorageGLES2::GeometryOwner *p_owner, int p_material, bool p_depth_pass, bool p_shadow_pass) {

	RasterizerStorageGLES2::Material *material = NULL;
	RID material_src;

	if (p_instance->material_override.is_valid()) {
		material_src = p_instance->material_override;
	} else if (p_material >= 0) {
		material_src = p_instance->materials[p_material];
	} else {
		material_src = p_geometry->material;
	}

	if (material_src.is_valid()) {
		material = storage->material_owner.getornull(material_src);

		if (!material->shader || !material->shader->valid) {
			material = NULL;
		}
	}

	if (!material) {
		material = storage->material_owner.getptr(default_material);
	}

	ERR_FAIL_COND(!material);

	_add_geometry_with_material(p_geometry, p_instance, p_owner, material, p_depth_pass, p_shadow_pass);

	while (material->next_pass.is_valid()) {
		material = storage->material_owner.getornull(material->next_pass);

		if (!material || !material->shader || !material->shader->valid) {
			break;
		}

		_add_geometry_with_material(p_geometry, p_instance, p_owner, material, p_depth_pass, p_shadow_pass);
	}
}
void RasterizerSceneGLES2::_add_geometry_with_material(RasterizerStorageGLES2::Geometry *p_geometry, InstanceBase *p_instance, RasterizerStorageGLES2::GeometryOwner *p_owner, RasterizerStorageGLES2::Material *p_material, bool p_depth_pass, bool p_shadow_pass) {

	bool has_base_alpha = (p_material->shader->spatial.uses_alpha && !p_material->shader->spatial.uses_alpha_scissor) || p_material->shader->spatial.uses_screen_texture || p_material->shader->spatial.uses_depth_texture;
	bool has_blend_alpha = p_material->shader->spatial.blend_mode != RasterizerStorageGLES2::Shader::Spatial::BLEND_MODE_MIX;
	bool has_alpha = has_base_alpha || has_blend_alpha;

	// TODO add this stuff
	// bool mirror = p_instance->mirror;
	// bool no_cull = false;

	RenderList::Element *e = has_alpha ? render_list.add_alpha_element() : render_list.add_element();

	if (!e) {
		return;
	}

	e->geometry = p_geometry;
	e->material = p_material;
	e->instance = p_instance;
	e->owner = p_owner;
	e->sort_key = 0;

	// TODO check render pass of geometry

	// TODO check directional light flag

	if (p_depth_pass) {
		// if we are in the depth pass we can sort out a few things to improve performance

		if (has_blend_alpha || p_material->shader->spatial.uses_depth_texture || (has_base_alpha && p_material->shader->spatial.depth_draw_mode != RasterizerStorageGLES2::Shader::Spatial::DEPTH_DRAW_ALPHA_PREPASS)) {
			return;
		}

		if (p_material->shader->spatial.uses_alpha_scissor && !p_material->shader->spatial.writes_modelview_or_projection && !p_material->shader->spatial.uses_vertex && !p_material->shader->spatial.uses_discard && p_material->shader->spatial.depth_draw_mode != RasterizerStorageGLES2::Shader::Spatial::DEPTH_DRAW_ALPHA_PREPASS) {

			// shader doesn't use discard or writes a custom vertex position,
			// so we can use a stripped down shader instead

			// TODO twosided and worldcoord stuff

			p_material = storage->material_owner.getptr(default_material_twosided);
		}

		has_alpha = false;
	}

	e->sort_key |= uint64_t(e->geometry->index) << RenderList::SORT_KEY_GEOMETRY_INDEX_SHIFT;
	e->sort_key |= uint64_t(e->instance->base_type) << RenderList::SORT_KEY_GEOMETRY_TYPE_SHIFT;

	if (p_material->shader->spatial.unshaded) {
		e->sort_key |= SORT_KEY_UNSHADED_FLAG;
	}

	if (!p_depth_pass) {
		e->sort_key |= uint64_t(e->material->index) << RenderList::SORT_KEY_MATERIAL_INDEX_SHIFT;

		e->sort_key |= uint64_t(p_material->render_priority + 128) << RenderList::SORT_KEY_PRIORITY_SHIFT;
	} else {
		// TODO
	}

	if (p_material->shader->spatial.uses_time) {
		VisualServerRaster::redraw_request();
	}
}

void RasterizerSceneGLES2::_fill_render_list(InstanceBase **p_cull_result, int p_cull_count, bool p_depth_pass, bool p_shadow_pass) {

	for (int i = 0; i < p_cull_count; i++) {

		InstanceBase *instance = p_cull_result[i];

		switch (instance->base_type) {

			case VS::INSTANCE_MESH: {

				RasterizerStorageGLES2::Mesh *mesh = storage->mesh_owner.getornull(instance->base);
				ERR_CONTINUE(!mesh);

				int num_surfaces = mesh->surfaces.size();

				for (int i = 0; i < num_surfaces; i++) {
					int material_index = instance->materials[i].is_valid() ? i : -1;

					RasterizerStorageGLES2::Surface *surface = mesh->surfaces[i];

					_add_geometry(surface, instance, NULL, material_index, p_depth_pass, p_shadow_pass);
				}

			} break;

			case VS::INSTANCE_MULTIMESH: {
				RasterizerStorageGLES2::MultiMesh *multi_mesh = storage->multimesh_owner.getptr(instance->base);
				ERR_CONTINUE(!multi_mesh);

				if (multi_mesh->size == 0 || multi_mesh->visible_instances == 0)
					continue;

				RasterizerStorageGLES2::Mesh *mesh = storage->mesh_owner.getptr(multi_mesh->mesh);
				if (!mesh)
					continue;

				int ssize = mesh->surfaces.size();

				for (int i = 0; i < ssize; i++) {
					RasterizerStorageGLES2::Surface *s = mesh->surfaces[i];
					_add_geometry(s, instance, multi_mesh, -1, p_depth_pass, p_shadow_pass);
				}
			} break;

			case VS::INSTANCE_IMMEDIATE: {
				RasterizerStorageGLES2::Immediate *im = storage->immediate_owner.getptr(instance->base);
				ERR_CONTINUE(!im);

				_add_geometry(im, instance, NULL, -1, p_depth_pass, p_shadow_pass);

			} break;

			default: {

			} break;
		}
	}
}

static const GLenum gl_primitive[] = {
	GL_POINTS,
	GL_LINES,
	GL_LINE_STRIP,
	GL_LINE_LOOP,
	GL_TRIANGLES,
	GL_TRIANGLE_STRIP,
	GL_TRIANGLE_FAN
};

void RasterizerSceneGLES2::_setup_material(RasterizerStorageGLES2::Material *p_material, bool p_reverse_cull, bool p_alpha_pass, Size2i p_skeleton_tex_size) {

	// material parameters

	state.scene_shader.set_custom_shader(p_material->shader->custom_code_id);

	state.scene_shader.bind();

	if (p_material->shader->spatial.no_depth_test) {
		glDisable(GL_DEPTH_TEST);
	} else {
		glEnable(GL_DEPTH_TEST);
	}

	switch (p_material->shader->spatial.depth_draw_mode) {
		case RasterizerStorageGLES2::Shader::Spatial::DEPTH_DRAW_ALPHA_PREPASS:
		case RasterizerStorageGLES2::Shader::Spatial::DEPTH_DRAW_OPAQUE: {

			glDepthMask(!p_alpha_pass);
		} break;
		case RasterizerStorageGLES2::Shader::Spatial::DEPTH_DRAW_ALWAYS: {
			glDepthMask(GL_TRUE);
		} break;
		case RasterizerStorageGLES2::Shader::Spatial::DEPTH_DRAW_NEVER: {
			glDepthMask(GL_FALSE);
		} break;
	}

	// TODO whyyyyy????
	p_reverse_cull = true;

	switch (p_material->shader->spatial.cull_mode) {
		case RasterizerStorageGLES2::Shader::Spatial::CULL_MODE_DISABLED: {
			glDisable(GL_CULL_FACE);
		} break;

		case RasterizerStorageGLES2::Shader::Spatial::CULL_MODE_BACK: {
			glEnable(GL_CULL_FACE);
			glCullFace(p_reverse_cull ? GL_FRONT : GL_BACK);
		} break;
		case RasterizerStorageGLES2::Shader::Spatial::CULL_MODE_FRONT: {
			glEnable(GL_CULL_FACE);
			glCullFace(p_reverse_cull ? GL_BACK : GL_FRONT);
		} break;
	}

	int tc = p_material->textures.size();
	Pair<StringName, RID> *textures = p_material->textures.ptrw();

	ShaderLanguage::ShaderNode::Uniform::Hint *texture_hints = p_material->shader->texture_hints.ptrw();

	state.scene_shader.set_uniform(SceneShaderGLES2::SKELETON_TEXTURE_SIZE, p_skeleton_tex_size);

	for (int i = 0; i < tc; i++) {

		glActiveTexture(GL_TEXTURE0 + i);

		RasterizerStorageGLES2::Texture *t = storage->texture_owner.getornull(textures[i].second);

		if (!t) {

			switch (texture_hints[i]) {
				case ShaderLanguage::ShaderNode::Uniform::HINT_BLACK_ALBEDO:
				case ShaderLanguage::ShaderNode::Uniform::HINT_BLACK: {
					glBindTexture(GL_TEXTURE_2D, storage->resources.black_tex);
				} break;
				case ShaderLanguage::ShaderNode::Uniform::HINT_ANISO: {
					glBindTexture(GL_TEXTURE_2D, storage->resources.aniso_tex);
				} break;
				case ShaderLanguage::ShaderNode::Uniform::HINT_NORMAL: {
					glBindTexture(GL_TEXTURE_2D, storage->resources.normal_tex);
				} break;
				default: {
					glBindTexture(GL_TEXTURE_2D, storage->resources.white_tex);
				} break;
			}

			continue;
		}

		t = t->get_ptr();

		glBindTexture(t->target, t->tex_id);
	}
	state.scene_shader.use_material((void *)p_material);
}

void RasterizerSceneGLES2::_setup_geometry(RenderList::Element *p_element, RasterizerStorageGLES2::Skeleton *p_skeleton) {

	state.scene_shader.set_conditional(SceneShaderGLES2::USE_SKELETON, p_skeleton != NULL);
	state.scene_shader.set_conditional(SceneShaderGLES2::USE_SKELETON_SOFTWARE, !storage->config.float_texture_supported);
	// state.scene_shader.set_conditional(SceneShaderGLES2::USE_SKELETON_SOFTWARE, true);

	switch (p_element->instance->base_type) {

		case VS::INSTANCE_MESH: {
			RasterizerStorageGLES2::Surface *s = static_cast<RasterizerStorageGLES2::Surface *>(p_element->geometry);

			state.scene_shader.set_conditional(SceneShaderGLES2::USE_INSTANCING, false);
			state.scene_shader.set_conditional(SceneShaderGLES2::ENABLE_COLOR_INTERP, s->attribs[VS::ARRAY_COLOR].enabled);
			state.scene_shader.set_conditional(SceneShaderGLES2::ENABLE_UV_INTERP, s->attribs[VS::ARRAY_TEX_UV].enabled);
			state.scene_shader.set_conditional(SceneShaderGLES2::ENABLE_UV2_INTERP, s->attribs[VS::ARRAY_TEX_UV2].enabled);

		} break;

		case VS::INSTANCE_MULTIMESH: {
			RasterizerStorageGLES2::MultiMesh *multi_mesh = static_cast<RasterizerStorageGLES2::MultiMesh *>(p_element->owner);
			RasterizerStorageGLES2::Surface *s = static_cast<RasterizerStorageGLES2::Surface *>(p_element->geometry);

			state.scene_shader.set_conditional(SceneShaderGLES2::ENABLE_COLOR_INTERP, true);
			state.scene_shader.set_conditional(SceneShaderGLES2::USE_INSTANCING, true);

			state.scene_shader.set_conditional(SceneShaderGLES2::ENABLE_UV_INTERP, s->attribs[VS::ARRAY_TEX_UV].enabled);
			state.scene_shader.set_conditional(SceneShaderGLES2::ENABLE_UV2_INTERP, s->attribs[VS::ARRAY_TEX_UV2].enabled);
		} break;

		case VS::INSTANCE_IMMEDIATE: {
			state.scene_shader.set_conditional(SceneShaderGLES2::USE_INSTANCING, false);
			state.scene_shader.set_conditional(SceneShaderGLES2::ENABLE_COLOR_INTERP, true);
			state.scene_shader.set_conditional(SceneShaderGLES2::ENABLE_UV_INTERP, true);
			state.scene_shader.set_conditional(SceneShaderGLES2::ENABLE_UV2_INTERP, true);
		} break;

		default: {

		} break;
	}

	if (storage->config.float_texture_supported) {
		if (p_skeleton) {
			glActiveTexture(GL_TEXTURE0 + storage->config.max_texture_image_units - 1);
			glBindTexture(GL_TEXTURE_2D, p_skeleton->tex_id);
		}

		return;
	}

	if (p_skeleton) {
		ERR_FAIL_COND(p_skeleton->use_2d);

		PoolVector<float> &transform_buffer = storage->resources.skeleton_transform_cpu_buffer;

		switch (p_element->instance->base_type) {
			case VS::INSTANCE_MESH: {
				RasterizerStorageGLES2::Surface *s = static_cast<RasterizerStorageGLES2::Surface *>(p_element->geometry);

				if (!s->attribs[VS::ARRAY_BONES].enabled || !s->attribs[VS::ARRAY_WEIGHTS].enabled) {
					break; // the whole instance has a skeleton, but this surface is not affected by it.
				}

				// 3 * vec4 per vertex
				if (transform_buffer.size() < s->array_len * 12) {
					transform_buffer.resize(s->array_len * 12);
				}

				const size_t bones_offset = s->attribs[VS::ARRAY_BONES].offset;
				const size_t bones_stride = s->attribs[VS::ARRAY_BONES].stride;
				const size_t bone_weight_offset = s->attribs[VS::ARRAY_WEIGHTS].offset;
				const size_t bone_weight_stride = s->attribs[VS::ARRAY_WEIGHTS].stride;

				{
					PoolVector<float>::Write write = transform_buffer.write();
					float *buffer = write.ptr();

					PoolVector<uint8_t>::Read vertex_array_read = s->data.read();
					const uint8_t *vertex_data = vertex_array_read.ptr();

					for (int i = 0; i < s->array_len; i++) {

						// do magic

						size_t bones[4];
						float bone_weight[4];

						if (s->attribs[VS::ARRAY_BONES].type == GL_UNSIGNED_BYTE) {
							// read as byte
							const uint8_t *bones_ptr = vertex_data + bones_offset + (i * bones_stride);
							bones[0] = bones_ptr[0];
							bones[1] = bones_ptr[1];
							bones[2] = bones_ptr[2];
							bones[3] = bones_ptr[3];
						} else {
							// read as short
							const uint16_t *bones_ptr = (const uint16_t *)(vertex_data + bones_offset + (i * bones_stride));
							bones[0] = bones_ptr[0];
							bones[1] = bones_ptr[1];
							bones[2] = bones_ptr[2];
							bones[3] = bones_ptr[3];
						}

						if (s->attribs[VS::ARRAY_WEIGHTS].type == GL_FLOAT) {
							// read as float
							const float *weight_ptr = (const float *)(vertex_data + bone_weight_offset + (i * bone_weight_stride));
							bone_weight[0] = weight_ptr[0];
							bone_weight[1] = weight_ptr[1];
							bone_weight[2] = weight_ptr[2];
							bone_weight[3] = weight_ptr[3];
						} else {
							// read as half
							const uint16_t *weight_ptr = (const uint16_t *)(vertex_data + bone_weight_offset + (i * bone_weight_stride));
							bone_weight[0] = (weight_ptr[0] / (float)0xFFFF);
							bone_weight[1] = (weight_ptr[1] / (float)0xFFFF);
							bone_weight[2] = (weight_ptr[2] / (float)0xFFFF);
							bone_weight[3] = (weight_ptr[3] / (float)0xFFFF);
						}

						size_t offset = i * 12;

						Transform transform;

						Transform bone_transforms[4] = {
							storage->skeleton_bone_get_transform(p_element->instance->skeleton, bones[0]),
							storage->skeleton_bone_get_transform(p_element->instance->skeleton, bones[1]),
							storage->skeleton_bone_get_transform(p_element->instance->skeleton, bones[2]),
							storage->skeleton_bone_get_transform(p_element->instance->skeleton, bones[3]),
						};

						transform.origin =
								bone_weight[0] * bone_transforms[0].origin +
								bone_weight[1] * bone_transforms[1].origin +
								bone_weight[2] * bone_transforms[2].origin +
								bone_weight[3] * bone_transforms[3].origin;

						transform.basis =
								bone_transforms[0].basis * bone_weight[0] +
								bone_transforms[1].basis * bone_weight[1] +
								bone_transforms[2].basis * bone_weight[2] +
								bone_transforms[3].basis * bone_weight[3];

						float row[3][4] = {
							{ transform.basis[0][0], transform.basis[0][1], transform.basis[0][2], transform.origin[0] },
							{ transform.basis[1][0], transform.basis[1][1], transform.basis[1][2], transform.origin[1] },
							{ transform.basis[2][0], transform.basis[2][1], transform.basis[2][2], transform.origin[2] },
						};

						size_t transform_buffer_offset = i * 12;

						copymem(&buffer[transform_buffer_offset], row, sizeof(row));
					}
				}

				storage->_update_skeleton_transform_buffer(transform_buffer, s->array_len * 12);
			} break;

			default: {

			} break;
		}
	}
}

void RasterizerSceneGLES2::_render_geometry(RenderList::Element *p_element) {

	switch (p_element->instance->base_type) {

		case VS::INSTANCE_MESH: {

			RasterizerStorageGLES2::Surface *s = static_cast<RasterizerStorageGLES2::Surface *>(p_element->geometry);

			// set up

			if (p_element->instance->skeleton.is_valid() && s->attribs[VS::ARRAY_BONES].enabled && s->attribs[VS::ARRAY_WEIGHTS].enabled) {
				glBindBuffer(GL_ARRAY_BUFFER, storage->resources.skeleton_transform_buffer);

				glEnableVertexAttribArray(VS::ARRAY_MAX + 0);
				glEnableVertexAttribArray(VS::ARRAY_MAX + 1);
				glEnableVertexAttribArray(VS::ARRAY_MAX + 2);

				glVertexAttribPointer(VS::ARRAY_MAX + 0, 4, GL_FLOAT, GL_FALSE, sizeof(float) * 12, (const void *)(sizeof(float) * 4 * 0));
				glVertexAttribPointer(VS::ARRAY_MAX + 1, 4, GL_FLOAT, GL_FALSE, sizeof(float) * 12, (const void *)(sizeof(float) * 4 * 1));
				glVertexAttribPointer(VS::ARRAY_MAX + 2, 4, GL_FLOAT, GL_FALSE, sizeof(float) * 12, (const void *)(sizeof(float) * 4 * 2));
			} else {
				// just to make sure
				glDisableVertexAttribArray(VS::ARRAY_MAX + 0);
				glDisableVertexAttribArray(VS::ARRAY_MAX + 1);
				glDisableVertexAttribArray(VS::ARRAY_MAX + 2);

				glVertexAttrib4f(VS::ARRAY_MAX + 0, 1, 0, 0, 0);
				glVertexAttrib4f(VS::ARRAY_MAX + 1, 0, 1, 0, 0);
				glVertexAttrib4f(VS::ARRAY_MAX + 2, 0, 0, 1, 0);
			}

			glBindBuffer(GL_ARRAY_BUFFER, s->vertex_id);

			if (s->index_array_len > 0) {
				glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, s->index_id);
			}

			for (int i = 0; i < VS::ARRAY_MAX - 1; i++) {
				if (s->attribs[i].enabled) {
					glEnableVertexAttribArray(i);
					glVertexAttribPointer(s->attribs[i].index, s->attribs[i].size, s->attribs[i].type, s->attribs[i].normalized, s->attribs[i].stride, (uint8_t *)0 + s->attribs[i].offset);
				} else {
					glDisableVertexAttribArray(i);
				}
			}

			// drawing

			if (s->index_array_len > 0) {
				glDrawElements(gl_primitive[s->primitive], s->index_array_len, (s->array_len >= (1 << 16)) ? GL_UNSIGNED_INT : GL_UNSIGNED_SHORT, 0);
			} else {
				glDrawArrays(gl_primitive[s->primitive], 0, s->array_len);
			}

			// tear down

			for (int i = 0; i < VS::ARRAY_MAX - 1; i++) {
				glDisableVertexAttribArray(i);
			}

			if (s->index_array_len > 0) {
				glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
			}

			if (p_element->instance->skeleton.is_valid() && s->attribs[VS::ARRAY_BONES].enabled && s->attribs[VS::ARRAY_WEIGHTS].enabled) {
				glBindBuffer(GL_ARRAY_BUFFER, storage->resources.skeleton_transform_buffer);

				glDisableVertexAttribArray(VS::ARRAY_MAX + 0);
				glDisableVertexAttribArray(VS::ARRAY_MAX + 1);
				glDisableVertexAttribArray(VS::ARRAY_MAX + 2);
			}

			glBindBuffer(GL_ARRAY_BUFFER, 0);

		} break;

		case VS::INSTANCE_MULTIMESH: {

			RasterizerStorageGLES2::MultiMesh *multi_mesh = static_cast<RasterizerStorageGLES2::MultiMesh *>(p_element->owner);
			RasterizerStorageGLES2::Surface *s = static_cast<RasterizerStorageGLES2::Surface *>(p_element->geometry);

			int amount = MIN(multi_mesh->size, multi_mesh->visible_instances);
			if (amount == -1) {
				amount = multi_mesh->size;
			}

			if (p_element->instance->skeleton.is_valid() && s->attribs[VS::ARRAY_BONES].enabled && s->attribs[VS::ARRAY_WEIGHTS].enabled) {
				glBindBuffer(GL_ARRAY_BUFFER, storage->resources.skeleton_transform_buffer);

				glEnableVertexAttribArray(VS::ARRAY_MAX + 0);
				glEnableVertexAttribArray(VS::ARRAY_MAX + 1);
				glEnableVertexAttribArray(VS::ARRAY_MAX + 2);

				glVertexAttribPointer(VS::ARRAY_MAX + 0, 4, GL_FLOAT, GL_FALSE, sizeof(float) * 12, (const void *)(sizeof(float) * 4 * 0));
				glVertexAttribPointer(VS::ARRAY_MAX + 1, 4, GL_FLOAT, GL_FALSE, sizeof(float) * 12, (const void *)(sizeof(float) * 4 * 1));
				glVertexAttribPointer(VS::ARRAY_MAX + 2, 4, GL_FLOAT, GL_FALSE, sizeof(float) * 12, (const void *)(sizeof(float) * 4 * 2));
			} else {
				// just to make sure
				glDisableVertexAttribArray(VS::ARRAY_MAX + 0);
				glDisableVertexAttribArray(VS::ARRAY_MAX + 1);
				glDisableVertexAttribArray(VS::ARRAY_MAX + 2);

				glVertexAttrib4f(VS::ARRAY_MAX + 0, 1, 0, 0, 0);
				glVertexAttrib4f(VS::ARRAY_MAX + 1, 0, 1, 0, 0);
				glVertexAttrib4f(VS::ARRAY_MAX + 2, 0, 0, 1, 0);
			}

			glBindBuffer(GL_ARRAY_BUFFER, s->vertex_id);

			if (s->index_array_len > 0) {
				glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, s->index_id);
			}

			for (int i = 0; i < VS::ARRAY_MAX - 1; i++) {
				if (s->attribs[i].enabled) {
					glEnableVertexAttribArray(i);
					glVertexAttribPointer(s->attribs[i].index, s->attribs[i].size, s->attribs[i].type, s->attribs[i].normalized, s->attribs[i].stride, (uint8_t *)0 + s->attribs[i].offset);
				} else {
					glDisableVertexAttribArray(i);
				}
			}

			glDisableVertexAttribArray(12); // transform 0
			glDisableVertexAttribArray(13); // transform 1
			glDisableVertexAttribArray(14); // transform 2
			glDisableVertexAttribArray(15); // color
			glDisableVertexAttribArray(8); // custom data

			if (!s->attribs[VS::ARRAY_COLOR].enabled) {
				glDisableVertexAttribArray(VS::ARRAY_COLOR);

				glVertexAttrib4f(VS::ARRAY_COLOR, 1, 1, 1, 1);
			}

			glVertexAttrib4f(15, 1, 1, 1, 1);
			glVertexAttrib4f(8, 0, 0, 0, 0);

			int stride = multi_mesh->color_floats + multi_mesh->custom_data_floats + multi_mesh->xform_floats;

			int color_ofs = multi_mesh->xform_floats;
			int custom_data_ofs = color_ofs + multi_mesh->color_floats;

			// drawing

			for (int i = 0; i < amount; i++) {
				float *buffer = &multi_mesh->data.write[i * stride];

				{
					// inline of multimesh_get_transform since it's such a pain
					// to get a RID from here...
					Transform transform;

					transform.basis.elements[0][0] = buffer[0];
					transform.basis.elements[0][1] = buffer[1];
					transform.basis.elements[0][2] = buffer[2];
					transform.origin.x = buffer[3];
					transform.basis.elements[1][0] = buffer[4];
					transform.basis.elements[1][1] = buffer[5];
					transform.basis.elements[1][2] = buffer[6];
					transform.origin.y = buffer[7];
					transform.basis.elements[2][0] = buffer[8];
					transform.basis.elements[2][1] = buffer[9];
					transform.basis.elements[2][2] = buffer[10];
					transform.origin.z = buffer[11];

					float row[3][4] = {
						{ transform.basis[0][0], transform.basis[0][1], transform.basis[0][2], transform.origin[0] },
						{ transform.basis[1][0], transform.basis[1][1], transform.basis[1][2], transform.origin[1] },
						{ transform.basis[2][0], transform.basis[2][1], transform.basis[2][2], transform.origin[2] },
					};

					glVertexAttrib4fv(12, row[0]);
					glVertexAttrib4fv(13, row[1]);
					glVertexAttrib4fv(14, row[2]);
				}

				if (multi_mesh->color_floats) {
					if (multi_mesh->color_format == VS::MULTIMESH_COLOR_8BIT) {
						uint8_t *color_data = (uint8_t *)(buffer + color_ofs);
						glVertexAttrib4f(15, color_data[0] / 255.0, color_data[1] / 255.0, color_data[2] / 255.0, color_data[3] / 255.0);
					} else {
						glVertexAttrib4fv(15, buffer + color_ofs);
					}
				}

				if (multi_mesh->custom_data_floats) {
					glVertexAttrib4fv(8, buffer + custom_data_ofs);
				}

				if (s->index_array_len > 0) {
					glDrawElements(gl_primitive[s->primitive], s->index_array_len, (s->array_len >= (1 << 16)) ? GL_UNSIGNED_INT : GL_UNSIGNED_SHORT, 0);
				} else {
					glDrawArrays(gl_primitive[s->primitive], 0, s->array_len);
				}
			}

			// tear down

			for (int i = 0; i < VS::ARRAY_MAX - 1; i++) {
				glDisableVertexAttribArray(i);
			}

			if (s->index_array_len > 0) {
				glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
			}

			if (p_element->instance->skeleton.is_valid() && s->attribs[VS::ARRAY_BONES].enabled && s->attribs[VS::ARRAY_WEIGHTS].enabled) {
				glBindBuffer(GL_ARRAY_BUFFER, storage->resources.skeleton_transform_buffer);

				glDisableVertexAttribArray(VS::ARRAY_MAX + 0);
				glDisableVertexAttribArray(VS::ARRAY_MAX + 1);
				glDisableVertexAttribArray(VS::ARRAY_MAX + 2);
			}

			glBindBuffer(GL_ARRAY_BUFFER, 0);
		} break;

		case VS::INSTANCE_IMMEDIATE: {
			const RasterizerStorageGLES2::Immediate *im = static_cast<const RasterizerStorageGLES2::Immediate *>(p_element->geometry);

			if (im->building) {
				return;
			}

			bool restore_tex = false;

			glBindBuffer(GL_ARRAY_BUFFER, state.immediate_buffer);

			for (const List<RasterizerStorageGLES2::Immediate::Chunk>::Element *E = im->chunks.front(); E; E = E->next()) {
				const RasterizerStorageGLES2::Immediate::Chunk &c = E->get();

				if (c.vertices.empty()) {
					continue;
				}

				int vertices = c.vertices.size();

				uint32_t buf_ofs = 0;

				storage->info.render.vertices_count += vertices;

				if (c.texture.is_valid() && storage->texture_owner.owns(c.texture)) {
					RasterizerStorageGLES2::Texture *t = storage->texture_owner.get(c.texture);

					t = t->get_ptr();

					if (t->redraw_if_visible) {
						VisualServerRaster::redraw_request();
					}

#ifdef TOOLS_ENABLED
					if (t->detect_3d) {
						t->detect_3d(t->detect_3d_ud);
					}
#endif
					if (t->render_target) {
						t->render_target->used_in_frame = true;
					}

					glActiveTexture(GL_TEXTURE0);
					glBindTexture(t->target, t->tex_id);
					restore_tex = true;
				} else if (restore_tex) {

					glActiveTexture(GL_TEXTURE0);
					glBindTexture(GL_TEXTURE_2D, state.current_main_tex);
					restore_tex = false;
				}

				if (!c.normals.empty()) {
					glEnableVertexAttribArray(VS::ARRAY_NORMAL);
					glBufferSubData(GL_ARRAY_BUFFER, buf_ofs, sizeof(Vector3) * vertices, c.normals.ptr());
					glVertexAttribPointer(VS::ARRAY_NORMAL, 3, GL_FLOAT, GL_FALSE, sizeof(Vector3), ((uint8_t *)NULL) + buf_ofs);
					buf_ofs += sizeof(Vector3) * vertices;
				} else {
					glDisableVertexAttribArray(VS::ARRAY_NORMAL);
				}

				if (!c.tangents.empty()) {
					glEnableVertexAttribArray(VS::ARRAY_TANGENT);
					glBufferSubData(GL_ARRAY_BUFFER, buf_ofs, sizeof(Plane) * vertices, c.tangents.ptr());
					glVertexAttribPointer(VS::ARRAY_TANGENT, 4, GL_FLOAT, GL_FALSE, sizeof(Plane), ((uint8_t *)NULL) + buf_ofs);
					buf_ofs += sizeof(Plane) * vertices;
				} else {
					glDisableVertexAttribArray(VS::ARRAY_TANGENT);
				}

				if (!c.colors.empty()) {
					glEnableVertexAttribArray(VS::ARRAY_COLOR);
					glBufferSubData(GL_ARRAY_BUFFER, buf_ofs, sizeof(Color) * vertices, c.colors.ptr());
					glVertexAttribPointer(VS::ARRAY_COLOR, 4, GL_FLOAT, GL_FALSE, sizeof(Color), ((uint8_t *)NULL) + buf_ofs);
					buf_ofs += sizeof(Color) * vertices;
				} else {
					glDisableVertexAttribArray(VS::ARRAY_COLOR);
				}

				if (!c.uvs.empty()) {
					glEnableVertexAttribArray(VS::ARRAY_TEX_UV);
					glBufferSubData(GL_ARRAY_BUFFER, buf_ofs, sizeof(Vector2) * vertices, c.uvs.ptr());
					glVertexAttribPointer(VS::ARRAY_TEX_UV, 2, GL_FLOAT, GL_FALSE, sizeof(Vector2), ((uint8_t *)NULL) + buf_ofs);
					buf_ofs += sizeof(Vector2) * vertices;
				} else {
					glDisableVertexAttribArray(VS::ARRAY_TEX_UV);
				}

				if (!c.uv2s.empty()) {
					glEnableVertexAttribArray(VS::ARRAY_TEX_UV2);
					glBufferSubData(GL_ARRAY_BUFFER, buf_ofs, sizeof(Vector2) * vertices, c.uv2s.ptr());
					glVertexAttribPointer(VS::ARRAY_TEX_UV2, 2, GL_FLOAT, GL_FALSE, sizeof(Vector2), ((uint8_t *)NULL) + buf_ofs);
					buf_ofs += sizeof(Vector2) * vertices;
				} else {
					glDisableVertexAttribArray(VS::ARRAY_TEX_UV2);
				}

				glEnableVertexAttribArray(VS::ARRAY_VERTEX);
				glBufferSubData(GL_ARRAY_BUFFER, buf_ofs, sizeof(Vector3) * vertices, c.vertices.ptr());
				glVertexAttribPointer(VS::ARRAY_VERTEX, 3, GL_FLOAT, GL_FALSE, sizeof(Vector3), ((uint8_t *)NULL) + buf_ofs);

				glDrawArrays(gl_primitive[c.primitive], 0, c.vertices.size());
			}

			if (restore_tex) {
				glActiveTexture(GL_TEXTURE0);
				glBindTexture(GL_TEXTURE_2D, state.current_main_tex);
				restore_tex = false;
			}

		} break;
	}
}

void RasterizerSceneGLES2::_render_render_list(RenderList::Element **p_elements, int p_element_count, const RID *p_directional_lights, int p_directional_light_count, const Transform &p_view_transform, const CameraMatrix &p_projection, RID p_shadow_atlas, Environment *p_env, GLuint p_base_env, float p_shadow_bias, float p_shadow_normal_bias, bool p_reverse_cull, bool p_alpha_pass, bool p_shadow, bool p_directional_add) {

	ShadowAtlas *shadow_atlas = shadow_atlas_owner.getornull(p_shadow_atlas);

	Vector2 screen_pixel_size;
	screen_pixel_size.x = 1.0 / storage->frame.current_rt->width;
	screen_pixel_size.y = 1.0 / storage->frame.current_rt->height;

	bool use_radiance_map = false;

	VMap<RID, Vector<RenderList::Element *> > lit_objects;

	for (int i = 0; i < p_element_count; i++) {
		RenderList::Element *e = p_elements[i];

		RasterizerStorageGLES2::Material *material = e->material;

		RasterizerStorageGLES2::Skeleton *skeleton = storage->skeleton_owner.getornull(e->instance->skeleton);

		if (p_base_env) {
			glActiveTexture(GL_TEXTURE0 + storage->config.max_texture_image_units - 2);
			glBindTexture(GL_TEXTURE_CUBE_MAP, p_base_env);
			use_radiance_map = true;
		}
		state.scene_shader.set_conditional(SceneShaderGLES2::USE_RADIANCE_MAP, use_radiance_map);

		if (material->shader->spatial.unshaded) {
			state.scene_shader.set_conditional(SceneShaderGLES2::USE_RADIANCE_MAP, false);
		} else {
			state.scene_shader.set_conditional(SceneShaderGLES2::USE_RADIANCE_MAP, use_radiance_map);
		}

		// opaque pass

		state.scene_shader.set_conditional(SceneShaderGLES2::LIGHT_PASS, false);

		_setup_geometry(e, skeleton);

		_setup_material(material, p_reverse_cull, p_alpha_pass, Size2i(skeleton ? skeleton->size * 3 : 0, 0));

		if (use_radiance_map) {
			state.scene_shader.set_uniform(SceneShaderGLES2::RADIANCE_INVERSE_XFORM, p_view_transform);
		}

		if (p_shadow) {
			state.scene_shader.set_uniform(SceneShaderGLES2::LIGHT_BIAS, p_shadow_bias);
			state.scene_shader.set_uniform(SceneShaderGLES2::LIGHT_NORMAL_BIAS, p_shadow_normal_bias);
		}

		if (p_env) {
			state.scene_shader.set_uniform(SceneShaderGLES2::BG_ENERGY, p_env->bg_energy);
			state.scene_shader.set_uniform(SceneShaderGLES2::AMBIENT_SKY_CONTRIBUTION, p_env->ambient_sky_contribution);
			state.scene_shader.set_uniform(SceneShaderGLES2::AMBIENT_COLOR, p_env->ambient_color);
			state.scene_shader.set_uniform(SceneShaderGLES2::AMBIENT_ENERGY, p_env->ambient_energy);

		} else {
			state.scene_shader.set_uniform(SceneShaderGLES2::BG_ENERGY, 1.0);
			state.scene_shader.set_uniform(SceneShaderGLES2::AMBIENT_SKY_CONTRIBUTION, 1.0);
			state.scene_shader.set_uniform(SceneShaderGLES2::AMBIENT_COLOR, Color(1.0, 1.0, 1.0, 1.0));
			state.scene_shader.set_uniform(SceneShaderGLES2::AMBIENT_ENERGY, 1.0);
		}

		glEnable(GL_BLEND);

		if (p_alpha_pass || p_directional_add) {
			int desired_blend_mode;
			if (p_directional_add) {
				desired_blend_mode = RasterizerStorageGLES2::Shader::Spatial::BLEND_MODE_ADD;
			} else {
				desired_blend_mode = material->shader->spatial.blend_mode;
			}

			switch (desired_blend_mode) {

				case RasterizerStorageGLES2::Shader::Spatial::BLEND_MODE_MIX: {
					glBlendEquation(GL_FUNC_ADD);
					if (storage->frame.current_rt && storage->frame.current_rt->flags[RasterizerStorage::RENDER_TARGET_TRANSPARENT]) {
						glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
					} else {
						glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
					}

				} break;
				case RasterizerStorageGLES2::Shader::Spatial::BLEND_MODE_ADD: {

					glBlendEquation(GL_FUNC_ADD);
					glBlendFunc(p_alpha_pass ? GL_SRC_ALPHA : GL_ONE, GL_ONE);

				} break;
				case RasterizerStorageGLES2::Shader::Spatial::BLEND_MODE_SUB: {

					glBlendEquation(GL_FUNC_REVERSE_SUBTRACT);
					glBlendFunc(GL_SRC_ALPHA, GL_ONE);
				} break;
				case RasterizerStorageGLES2::Shader::Spatial::BLEND_MODE_MUL: {
					glBlendEquation(GL_FUNC_ADD);
					if (storage->frame.current_rt && storage->frame.current_rt->flags[RasterizerStorage::RENDER_TARGET_TRANSPARENT]) {
						glBlendFuncSeparate(GL_DST_COLOR, GL_ZERO, GL_DST_ALPHA, GL_ZERO);
					} else {
						glBlendFuncSeparate(GL_DST_COLOR, GL_ZERO, GL_ZERO, GL_ONE);
					}

				} break;
			}
		} else {
			// no blend mode given - assume mix
			glBlendEquation(GL_FUNC_ADD);
			if (storage->frame.current_rt && storage->frame.current_rt->flags[RasterizerStorage::RENDER_TARGET_TRANSPARENT]) {
				glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
			} else {
				glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
			}
		}

		state.scene_shader.set_uniform(SceneShaderGLES2::CAMERA_MATRIX, p_view_transform.inverse());
		state.scene_shader.set_uniform(SceneShaderGLES2::CAMERA_INVERSE_MATRIX, p_view_transform);
		state.scene_shader.set_uniform(SceneShaderGLES2::PROJECTION_MATRIX, p_projection);
		state.scene_shader.set_uniform(SceneShaderGLES2::PROJECTION_INVERSE_MATRIX, p_projection.inverse());

		state.scene_shader.set_uniform(SceneShaderGLES2::TIME, storage->frame.time[0]);

		state.scene_shader.set_uniform(SceneShaderGLES2::SCREEN_PIXEL_SIZE, screen_pixel_size);
		state.scene_shader.set_uniform(SceneShaderGLES2::NORMAL_MULT, 1.0); // TODO mirror?
		state.scene_shader.set_uniform(SceneShaderGLES2::WORLD_TRANSFORM, e->instance->transform);

		_render_geometry(e);

		if (material->shader->spatial.unshaded)
			continue;

		if (p_shadow)
			continue;

		for (int light = 0; light < e->instance->light_instances.size(); light++) {

			RID light_instance = e->instance->light_instances[light];

			lit_objects[light_instance].push_back(e);
		}
	}

	if (p_shadow) {
		state.scene_shader.set_conditional(SceneShaderGLES2::USE_RADIANCE_MAP, false);
		state.scene_shader.set_conditional(SceneShaderGLES2::LIGHT_USE_PSSM4, false);
		state.scene_shader.set_conditional(SceneShaderGLES2::LIGHT_USE_PSSM2, false);
		state.scene_shader.set_conditional(SceneShaderGLES2::LIGHT_USE_PSSM_BLEND, false);
		return;
	}

	state.scene_shader.set_conditional(SceneShaderGLES2::LIGHT_PASS, true);

	glEnable(GL_BLEND);
	glBlendEquation(GL_FUNC_ADD);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE);

	for (int lo = 0; lo < lit_objects.size(); lo++) {

		RID key = lit_objects.getk(lo);

		LightInstance *light = light_instance_owner.getornull(key);
		RasterizerStorageGLES2::Light *light_ptr = light->light_ptr;

		const Vector<RenderList::Element *> &list = lit_objects.getv(lo);

		for (int i = 0; i < list.size(); i++) {

			RenderList::Element *e = list[i];
			RasterizerStorageGLES2::Material *material = e->material;

			RasterizerStorageGLES2::Skeleton *skeleton = storage->skeleton_owner.getornull(e->instance->skeleton);

			{
				_setup_geometry(e, skeleton);

				_setup_material(material, p_reverse_cull, p_alpha_pass, Size2i(skeleton ? skeleton->size * 3 : 0, 0));
				if (shadow_atlas != NULL) {
					glActiveTexture(GL_TEXTURE0 + storage->config.max_texture_image_units - 4);
					glBindTexture(GL_TEXTURE_2D, shadow_atlas->depth);
				}

				state.scene_shader.set_uniform(SceneShaderGLES2::CAMERA_MATRIX, p_view_transform.inverse());
				state.scene_shader.set_uniform(SceneShaderGLES2::CAMERA_INVERSE_MATRIX, p_view_transform);
				state.scene_shader.set_uniform(SceneShaderGLES2::PROJECTION_MATRIX, p_projection);
				state.scene_shader.set_uniform(SceneShaderGLES2::PROJECTION_INVERSE_MATRIX, p_projection.inverse());

				state.scene_shader.set_uniform(SceneShaderGLES2::TIME, storage->frame.time[0]);

				state.scene_shader.set_uniform(SceneShaderGLES2::SCREEN_PIXEL_SIZE, screen_pixel_size);
				state.scene_shader.set_uniform(SceneShaderGLES2::NORMAL_MULT, 1.0); // TODO mirror?
				state.scene_shader.set_uniform(SceneShaderGLES2::WORLD_TRANSFORM, e->instance->transform);
			}

			switch (light_ptr->type) {
				case VS::LIGHT_OMNI: {

					state.scene_shader.set_uniform(SceneShaderGLES2::LIGHT_TYPE, (int)1);

					Vector3 position = p_view_transform.inverse().xform(light->transform.origin);

					state.scene_shader.set_uniform(SceneShaderGLES2::LIGHT_POSITION, position);

					float range = light_ptr->param[VS::LIGHT_PARAM_RANGE];
					state.scene_shader.set_uniform(SceneShaderGLES2::LIGHT_RANGE, range);

					Color attenuation = Color(0.0, 0.0, 0.0, 0.0);
					attenuation.a = light_ptr->param[VS::LIGHT_PARAM_ATTENUATION];
					state.scene_shader.set_uniform(SceneShaderGLES2::LIGHT_ATTENUATION, attenuation);

					if (light_ptr->shadow && shadow_atlas->shadow_owners.has(light->self)) {

						uint32_t key = shadow_atlas->shadow_owners[light->self];

						uint32_t quadrant = (key >> ShadowAtlas::QUADRANT_SHIFT) & 0x03;
						uint32_t shadow = key & ShadowAtlas::SHADOW_INDEX_MASK;

						ERR_CONTINUE(shadow >= (uint32_t)shadow_atlas->quadrants[quadrant].shadows.size());

						uint32_t atlas_size = shadow_atlas->size;
						uint32_t quadrant_size = atlas_size >> 1;

						uint32_t x = (quadrant & 1) * quadrant_size;
						uint32_t y = (quadrant >> 1) * quadrant_size;

						uint32_t shadow_size = (quadrant_size / shadow_atlas->quadrants[quadrant].subdivision);
						x += (shadow % shadow_atlas->quadrants[quadrant].subdivision) * shadow_size;
						y += (shadow / shadow_atlas->quadrants[quadrant].subdivision) * shadow_size;

						uint32_t width = shadow_size;
						uint32_t height = shadow_size;

						if (light->light_ptr->omni_shadow_detail == VS::LIGHT_OMNI_SHADOW_DETAIL_HORIZONTAL) {
							height /= 2;
						} else {
							width /= 2;
						}

						Transform proj = (p_view_transform.inverse() * light->transform).inverse();

						Color light_clamp;
						light_clamp[0] = float(x) / atlas_size;
						light_clamp[1] = float(y) / atlas_size;
						light_clamp[2] = float(width) / atlas_size;
						light_clamp[3] = float(height) / atlas_size;

						state.scene_shader.set_uniform(SceneShaderGLES2::LIGHT_SHADOW_MATRIX, proj);
						state.scene_shader.set_uniform(SceneShaderGLES2::LIGHT_CLAMP, light_clamp);

						state.scene_shader.set_uniform(SceneShaderGLES2::LIGHT_HAS_SHADOW, 1.0);
					} else {
						state.scene_shader.set_uniform(SceneShaderGLES2::LIGHT_HAS_SHADOW, 0.0);
					}
				} break;

				case VS::LIGHT_SPOT: {
					Vector3 position = p_view_transform.inverse().xform(light->transform.origin);
					state.scene_shader.set_uniform(SceneShaderGLES2::LIGHT_TYPE, (int)2);
					state.scene_shader.set_uniform(SceneShaderGLES2::LIGHT_POSITION, position);

					Vector3 direction = p_view_transform.inverse().basis.xform(light->transform.basis.xform(Vector3(0, 0, -1))).normalized();
					state.scene_shader.set_uniform(SceneShaderGLES2::LIGHT_DIRECTION, direction);
					Color attenuation = Color(0.0, 0.0, 0.0, 0.0);
					attenuation.a = light_ptr->param[VS::LIGHT_PARAM_ATTENUATION];
					float range = light_ptr->param[VS::LIGHT_PARAM_RANGE];
					float spot_attenuation = light_ptr->param[VS::LIGHT_PARAM_SPOT_ATTENUATION];
					float angle = light_ptr->param[VS::LIGHT_PARAM_SPOT_ANGLE];
					angle = Math::cos(Math::deg2rad(angle));
					state.scene_shader.set_uniform(SceneShaderGLES2::LIGHT_ATTENUATION, attenuation);
					state.scene_shader.set_uniform(SceneShaderGLES2::LIGHT_SPOT_ATTENUATION, spot_attenuation);
					state.scene_shader.set_uniform(SceneShaderGLES2::LIGHT_SPOT_RANGE, spot_attenuation);
					state.scene_shader.set_uniform(SceneShaderGLES2::LIGHT_SPOT_ANGLE, angle);
					state.scene_shader.set_uniform(SceneShaderGLES2::LIGHT_RANGE, range);

					if (light->light_ptr->shadow && shadow_atlas && shadow_atlas->shadow_owners.has(light->self)) {
						uint32_t key = shadow_atlas->shadow_owners[light->self];

						uint32_t quadrant = (key >> ShadowAtlas::QUADRANT_SHIFT) & 0x03;
						uint32_t shadow = key & ShadowAtlas::SHADOW_INDEX_MASK;

						ERR_CONTINUE(shadow >= (uint32_t)shadow_atlas->quadrants[quadrant].shadows.size());

						uint32_t atlas_size = shadow_atlas->size;
						uint32_t quadrant_size = atlas_size >> 1;

						uint32_t x = (quadrant & 1) * quadrant_size;
						uint32_t y = (quadrant >> 1) * quadrant_size;

						uint32_t shadow_size = (quadrant_size / shadow_atlas->quadrants[quadrant].subdivision);
						x += (shadow % shadow_atlas->quadrants[quadrant].subdivision) * shadow_size;
						y += (shadow / shadow_atlas->quadrants[quadrant].subdivision) * shadow_size;

						uint32_t width = shadow_size;
						uint32_t height = shadow_size;

						Rect2 rect(float(x) / atlas_size, float(y) / atlas_size, float(width) / atlas_size, float(height) / atlas_size);

						Color light_clamp;
						light_clamp[0] = rect.position.x;
						light_clamp[1] = rect.position.y;
						light_clamp[2] = rect.size.x;
						light_clamp[3] = rect.size.y;

						Transform modelview = (p_view_transform.inverse() * light->transform).inverse();

						CameraMatrix bias;
						bias.set_light_bias();

						CameraMatrix rectm;
						rectm.set_light_atlas_rect(rect);

						CameraMatrix shadow_matrix = rectm * bias * light->shadow_transform[0].camera * modelview;

						state.scene_shader.set_uniform(SceneShaderGLES2::LIGHT_HAS_SHADOW, 1.0);
						state.scene_shader.set_uniform(SceneShaderGLES2::LIGHT_SHADOW_MATRIX, shadow_matrix);
						state.scene_shader.set_uniform(SceneShaderGLES2::LIGHT_CLAMP, light_clamp);

					} else {
						state.scene_shader.set_uniform(SceneShaderGLES2::LIGHT_HAS_SHADOW, 0.0);
					}

				} break;

				default: break;
			}

			float energy = light->light_ptr->param[VS::LIGHT_PARAM_ENERGY];
			float specular = light->light_ptr->param[VS::LIGHT_PARAM_SPECULAR];

			state.scene_shader.set_uniform(SceneShaderGLES2::LIGHT_ENERGY, energy);
			state.scene_shader.set_uniform(SceneShaderGLES2::LIGHT_COLOR, light->light_ptr->color.to_linear());
			state.scene_shader.set_uniform(SceneShaderGLES2::LIGHT_SPECULAR, specular);

			_render_geometry(e);
		}
	}

	for (int dl = 0; dl < p_directional_light_count; dl++) {
		RID light_rid = p_directional_lights[dl];
		LightInstance *light = light_instance_owner.getornull(light_rid);
		RasterizerStorageGLES2::Light *light_ptr = light->light_ptr;

		switch (light_ptr->directional_shadow_mode) {
			case VS::LIGHT_DIRECTIONAL_SHADOW_ORTHOGONAL: {
			} break;
			case VS::LIGHT_DIRECTIONAL_SHADOW_PARALLEL_2_SPLITS: {
				state.scene_shader.set_conditional(SceneShaderGLES2::LIGHT_USE_PSSM2, true);
				state.scene_shader.set_conditional(SceneShaderGLES2::LIGHT_USE_PSSM_BLEND, light_ptr->directional_blend_splits);
			} break;

			case VS::LIGHT_DIRECTIONAL_SHADOW_PARALLEL_4_SPLITS: {
				state.scene_shader.set_conditional(SceneShaderGLES2::LIGHT_USE_PSSM4, true);
				state.scene_shader.set_conditional(SceneShaderGLES2::LIGHT_USE_PSSM_BLEND, light_ptr->directional_blend_splits);
			} break;
			default:
				break;
		}

		for (int i = 0; i < p_element_count; i++) {

			RenderList::Element *e = p_elements[i];
			RasterizerStorageGLES2::Material *material = e->material;
			RasterizerStorageGLES2::Skeleton *skeleton = storage->skeleton_owner.getornull(e->instance->skeleton);

			{
				_setup_material(material, p_reverse_cull, false, Size2i(skeleton ? skeleton->size * 3 : 0, 0));

				if (directional_shadow.depth) {
					glActiveTexture(GL_TEXTURE0 + storage->config.max_texture_image_units - 4); // TODO move into base pass
					glBindTexture(GL_TEXTURE_2D, directional_shadow.depth);
				}

				state.scene_shader.set_uniform(SceneShaderGLES2::CAMERA_MATRIX, p_view_transform.inverse());
				state.scene_shader.set_uniform(SceneShaderGLES2::CAMERA_INVERSE_MATRIX, p_view_transform);
				state.scene_shader.set_uniform(SceneShaderGLES2::PROJECTION_MATRIX, p_projection);
				state.scene_shader.set_uniform(SceneShaderGLES2::PROJECTION_INVERSE_MATRIX, p_projection.inverse());

				state.scene_shader.set_uniform(SceneShaderGLES2::TIME, storage->frame.time[0]);

				state.scene_shader.set_uniform(SceneShaderGLES2::SCREEN_PIXEL_SIZE, screen_pixel_size);
				state.scene_shader.set_uniform(SceneShaderGLES2::NORMAL_MULT, 1.0); // TODO mirror?
				state.scene_shader.set_uniform(SceneShaderGLES2::WORLD_TRANSFORM, e->instance->transform);
			}
			state.scene_shader.set_uniform(SceneShaderGLES2::LIGHT_TYPE, (int)0);
			Vector3 direction = p_view_transform.inverse().basis.xform(light->transform.basis.xform(Vector3(0, 0, -1))).normalized();
			state.scene_shader.set_uniform(SceneShaderGLES2::LIGHT_DIRECTION, direction);

			float energy = light_ptr->param[VS::LIGHT_PARAM_ENERGY];
			float specular = light_ptr->param[VS::LIGHT_PARAM_SPECULAR];

			state.scene_shader.set_uniform(SceneShaderGLES2::LIGHT_ENERGY, energy);
			state.scene_shader.set_uniform(SceneShaderGLES2::LIGHT_SPECULAR, specular);

			float sign = light_ptr->negative ? -1 : 1;

			Color linear_col = light_ptr->color.to_linear();
			Color color;
			for (int c = 0; c < 3; c++)
				color[c] = linear_col[c] * sign * energy * Math_PI;

			color[3] = 0;

			state.scene_shader.set_uniform(SceneShaderGLES2::LIGHT_COLOR, color);

			CameraMatrix matrices[4];

			if (light_ptr->shadow && directional_shadow.depth) {

				int shadow_count = 0;
				Color split_offsets;

				switch (light_ptr->directional_shadow_mode) {
					case VS::LIGHT_DIRECTIONAL_SHADOW_ORTHOGONAL: {
						shadow_count = 1;
					} break;

					case VS::LIGHT_DIRECTIONAL_SHADOW_PARALLEL_2_SPLITS: {
						shadow_count = 2;
					} break;

					case VS::LIGHT_DIRECTIONAL_SHADOW_PARALLEL_4_SPLITS: {
						shadow_count = 4;
					} break;
				}

				for (int k = 0; k < shadow_count; k++) {

					uint32_t x = light->directional_rect.position.x;
					uint32_t y = light->directional_rect.position.y;
					uint32_t width = light->directional_rect.size.x;
					uint32_t height = light->directional_rect.size.y;

					if (light_ptr->directional_shadow_mode == VS::LIGHT_DIRECTIONAL_SHADOW_PARALLEL_4_SPLITS) {

						width /= 2;
						height /= 2;

						if (k == 0) {

						} else if (k == 1) {
							x += width;
						} else if (k == 2) {
							y += height;
						} else if (k == 3) {
							x += width;
							y += height;
						}

					} else if (light_ptr->directional_shadow_mode == VS::LIGHT_DIRECTIONAL_SHADOW_PARALLEL_2_SPLITS) {

						height /= 2;

						if (k == 0) {

						} else {
							y += height;
						}
					}

					split_offsets[k] = light->shadow_transform[k].split;

					Transform modelview = (p_view_transform * light->shadow_transform[k].transform).inverse();

					CameraMatrix bias;
					bias.set_light_bias();
					CameraMatrix rectm;
					Rect2 atlas_rect = Rect2(float(x) / directional_shadow.size, float(y) / directional_shadow.size, float(width) / directional_shadow.size, float(height) / directional_shadow.size);
					rectm.set_light_atlas_rect(atlas_rect);

					CameraMatrix shadow_mtx = rectm * bias * light->shadow_transform[k].camera * modelview;
					matrices[k] = shadow_mtx.inverse();

					Color light_clamp;
					light_clamp[0] = atlas_rect.position.x;
					light_clamp[1] = atlas_rect.position.y;
					light_clamp[2] = atlas_rect.size.x;
					light_clamp[3] = atlas_rect.size.y;

					state.scene_shader.set_uniform(SceneShaderGLES2::LIGHT_HAS_SHADOW, 1.0);
					state.scene_shader.set_uniform(SceneShaderGLES2::LIGHT_CLAMP, light_clamp);
					state.scene_shader.set_uniform(SceneShaderGLES2::LIGHT_SPLIT_OFFSETS, split_offsets);
				}

				state.scene_shader.set_uniform(SceneShaderGLES2::LIGHT_SHADOW_MATRIX1, matrices[0]);
				state.scene_shader.set_uniform(SceneShaderGLES2::LIGHT_SHADOW_MATRIX2, matrices[1]);
				state.scene_shader.set_uniform(SceneShaderGLES2::LIGHT_SHADOW_MATRIX3, matrices[2]);
				state.scene_shader.set_uniform(SceneShaderGLES2::LIGHT_SHADOW_MATRIX4, matrices[3]);
			} else {
				state.scene_shader.set_uniform(SceneShaderGLES2::LIGHT_HAS_SHADOW, 0.0);
			}

			_render_geometry(e);
		}
	}

	state.scene_shader.set_conditional(SceneShaderGLES2::LIGHT_PASS, false);

	state.scene_shader.set_conditional(SceneShaderGLES2::USE_RADIANCE_MAP, false);
	state.scene_shader.set_conditional(SceneShaderGLES2::LIGHT_USE_PSSM4, false);
	state.scene_shader.set_conditional(SceneShaderGLES2::LIGHT_USE_PSSM2, false);
	state.scene_shader.set_conditional(SceneShaderGLES2::LIGHT_USE_PSSM_BLEND, false);
}

void RasterizerSceneGLES2::_draw_sky(RasterizerStorageGLES2::Sky *p_sky, const CameraMatrix &p_projection, const Transform &p_transform, bool p_vflip, float p_custom_fov, float p_energy) {
	ERR_FAIL_COND(!p_sky);

	RasterizerStorageGLES2::Texture *tex = storage->texture_owner.getornull(p_sky->panorama);
	ERR_FAIL_COND(!tex);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(tex->target, tex->tex_id);

	glDepthMask(GL_TRUE);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	glDisable(GL_BLEND);
	glDepthFunc(GL_LEQUAL);
	glColorMask(1, 1, 1, 1);

	// Camera
	CameraMatrix camera;

	if (p_custom_fov) {

		float near_plane = p_projection.get_z_near();
		float far_plane = p_projection.get_z_far();
		float aspect = p_projection.get_aspect();

		camera.set_perspective(p_custom_fov, aspect, near_plane, far_plane);
	} else {
		camera = p_projection;
	}

	float flip_sign = p_vflip ? -1 : 1;

	// If matrix[2][0] or matrix[2][1] we're dealing with an asymmetrical projection matrix. This is the case for stereoscopic rendering (i.e. VR).
	// To ensure the image rendered is perspective correct we need to move some logic into the shader. For this the USE_ASYM_PANO option is introduced.
	// It also means the uv coordinates are ignored in this mode and we don't need our loop.
	bool asymmetrical = ((camera.matrix[2][0] != 0.0) || (camera.matrix[2][1] != 0.0));

	Vector3 vertices[8] = {
		Vector3(-1, -1 * flip_sign, 1),
		Vector3(0, 1, 0),
		Vector3(1, -1 * flip_sign, 1),
		Vector3(1, 1, 0),
		Vector3(1, 1 * flip_sign, 1),
		Vector3(1, 0, 0),
		Vector3(-1, 1 * flip_sign, 1),
		Vector3(0, 0, 0),
	};

	if (!asymmetrical) {
		float vw, vh, zn;
		camera.get_viewport_size(vw, vh);
		zn = p_projection.get_z_near();

		for (int i = 0; i < 4; i++) {
			Vector3 uv = vertices[i * 2 + 1];
			uv.x = (uv.x * 2.0 - 1.0) * vw;
			uv.y = -(uv.y * 2.0 - 1.0) * vh;
			uv.z = -zn;
			vertices[i * 2 + 1] = p_transform.basis.xform(uv).normalized();
			vertices[i * 2 + 1].z = -vertices[i * 2 + 1].z;
		}
	}

	glBindBuffer(GL_ARRAY_BUFFER, state.sky_verts);
	glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(Vector3) * 8, vertices);

	// bind sky vertex array....
	glVertexAttribPointer(VS::ARRAY_VERTEX, 3, GL_FLOAT, GL_FALSE, sizeof(Vector3) * 2, 0);
	glVertexAttribPointer(VS::ARRAY_TEX_UV, 3, GL_FLOAT, GL_FALSE, sizeof(Vector3) * 2, ((uint8_t *)NULL) + sizeof(Vector3));
	glEnableVertexAttribArray(VS::ARRAY_VERTEX);
	glEnableVertexAttribArray(VS::ARRAY_TEX_UV);

	storage->shaders.copy.set_conditional(CopyShaderGLES2::USE_MULTIPLIER, true);
	storage->shaders.copy.set_conditional(CopyShaderGLES2::USE_CUBEMAP, false);
	storage->shaders.copy.set_conditional(CopyShaderGLES2::USE_PANORAMA, true);
	storage->shaders.copy.set_conditional(CopyShaderGLES2::USE_COPY_SECTION, false);
	storage->shaders.copy.set_conditional(CopyShaderGLES2::USE_CUSTOM_ALPHA, false);
	storage->shaders.copy.bind();
	storage->shaders.copy.set_uniform(CopyShaderGLES2::MULTIPLIER, p_energy);

	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

	glDisableVertexAttribArray(VS::ARRAY_VERTEX);
	glDisableVertexAttribArray(VS::ARRAY_TEX_UV);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	storage->shaders.copy.set_conditional(CopyShaderGLES2::USE_MULTIPLIER, false);
	storage->shaders.copy.set_conditional(CopyShaderGLES2::USE_CUBEMAP, false);
}

void RasterizerSceneGLES2::render_scene(const Transform &p_cam_transform, const CameraMatrix &p_cam_projection, bool p_cam_ortogonal, InstanceBase **p_cull_result, int p_cull_count, RID *p_light_cull_result, int p_light_cull_count, RID *p_reflection_probe_cull_result, int p_reflection_probe_cull_count, RID p_environment, RID p_shadow_atlas, RID p_reflection_atlas, RID p_reflection_probe, int p_reflection_probe_pass) {

	glEnable(GL_BLEND);

	GLuint current_fb = storage->frame.current_rt->fbo;
	Environment *env = environment_owner.getornull(p_environment);

	// render list stuff

	render_list.clear();
	_fill_render_list(p_cull_result, p_cull_count, false, false);

	// other stuff

	glBindFramebuffer(GL_FRAMEBUFFER, current_fb);

	glDepthFunc(GL_LEQUAL);
	glDepthMask(GL_TRUE);
	glClearDepth(1.0f);
	glEnable(GL_DEPTH_TEST);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	storage->frame.clear_request = false;

	glVertexAttrib4f(VS::ARRAY_COLOR, 1, 1, 1, 1);

	glBlendEquation(GL_FUNC_ADD);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	// render sky
	RasterizerStorageGLES2::Sky *sky = NULL;
	GLuint env_radiance_tex = 0;
	if (env) {
		switch (env->bg_mode) {

			case VS::ENV_BG_COLOR_SKY:
			case VS::ENV_BG_SKY: {
				sky = storage->sky_owner.getornull(env->sky);

				if (sky) {
					env_radiance_tex = sky->radiance;
				}
			} break;

			default: {
				// FIXME: implement other background modes
			} break;
		}
	}

	if (env && env->bg_mode == VS::ENV_BG_SKY && (!storage->frame.current_rt || !storage->frame.current_rt->flags[RasterizerStorage::RENDER_TARGET_TRANSPARENT])) {

		if (sky && sky->panorama.is_valid()) {
			_draw_sky(sky, p_cam_projection, p_cam_transform, false, env->sky_custom_fov, env->bg_energy);
		}
	}

	Vector<RID> directional_lights;

	for (int i = 0; i < p_light_cull_count; i++) {
		RID light_rid = p_light_cull_result[i];

		LightInstance *light = light_instance_owner.getornull(light_rid);

		if (light->light_ptr->type == VS::LIGHT_DIRECTIONAL) {
			directional_lights.push_back(light_rid);
		}
	}

	// render opaque things first
	render_list.sort_by_key(false);
	_render_render_list(render_list.elements, render_list.element_count, directional_lights.ptr(), directional_lights.size(), p_cam_transform, p_cam_projection, p_shadow_atlas, env, env_radiance_tex, 0.0, 0.0, false, false, false, false);

	// alpha pass

	glBlendEquation(GL_FUNC_ADD);
	glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

	render_list.sort_by_key(true);
	_render_render_list(&render_list.elements[render_list.max_elements - render_list.alpha_element_count], render_list.alpha_element_count, directional_lights.ptr(), directional_lights.size(), p_cam_transform, p_cam_projection, p_shadow_atlas, env, env_radiance_tex, 0.0, 0.0, false, true, false, false);

	glDepthMask(GL_FALSE);
	glDisable(GL_DEPTH_TEST);

	// #define GLES2_SHADOW_ATLAS_DEBUG_VIEW

#ifdef GLES2_SHADOW_ATLAS_DEBUG_VIEW
	ShadowAtlas *shadow_atlas = shadow_atlas_owner.getornull(p_shadow_atlas);
	if (shadow_atlas) {
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, shadow_atlas->depth);

		glViewport(0, 0, storage->frame.current_rt->width / 4, storage->frame.current_rt->height / 4);
		storage->shaders.copy.set_conditional(CopyShaderGLES2::USE_CUBEMAP, false);
		storage->shaders.copy.set_conditional(CopyShaderGLES2::USE_COPY_SECTION, false);
		storage->shaders.copy.set_conditional(CopyShaderGLES2::USE_CUSTOM_ALPHA, false);
		storage->shaders.copy.set_conditional(CopyShaderGLES2::USE_MULTIPLIER, false);
		storage->shaders.copy.set_conditional(CopyShaderGLES2::USE_PANORAMA, false);
		storage->shaders.copy.bind();

		storage->_copy_screen();
	}
#endif
}

void RasterizerSceneGLES2::render_shadow(RID p_light, RID p_shadow_atlas, int p_pass, InstanceBase **p_cull_result, int p_cull_count) {

	LightInstance *light_instance = light_instance_owner.getornull(p_light);
	ERR_FAIL_COND(!light_instance);

	RasterizerStorageGLES2::Light *light = light_instance->light_ptr;
	ERR_FAIL_COND(!light);

	uint32_t x;
	uint32_t y;
	uint32_t width;
	uint32_t height;
	uint32_t vp_height;

	float zfar = 0;
	bool flip_facing = false;
	int custom_vp_size = 0;

	GLuint fbo = 0;

	int current_cubemap = -1;
	float bias = 0;
	float normal_bias = 0;

	CameraMatrix light_projection;
	Transform light_transform;

	// TODO directional light

	if (light->type == VS::LIGHT_DIRECTIONAL) {
		// set pssm stuff

		// TODO set this only when changed

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

		light_projection = light_instance->shadow_transform[p_pass].camera;
		light_transform = light_instance->shadow_transform[p_pass].transform;

		x = light_instance->directional_rect.position.x;
		y = light_instance->directional_rect.position.y;
		width = light_instance->directional_rect.size.width;
		height = light_instance->directional_rect.size.height;

		if (light->directional_shadow_mode == VS::LIGHT_DIRECTIONAL_SHADOW_PARALLEL_4_SPLITS) {

			width /= 2;
			height /= 2;

			if (p_pass == 0) {

			} else if (p_pass == 1) {
				x += width;
			} else if (p_pass == 2) {
				y += height;
			} else if (p_pass == 3) {
				x += width;
				y += height;
			}

		} else if (light->directional_shadow_mode == VS::LIGHT_DIRECTIONAL_SHADOW_PARALLEL_2_SPLITS) {

			height /= 2;

			if (p_pass == 0) {

			} else {
				y += height;
			}
		}

		float bias_mult = Math::lerp(1.0f, light_instance->shadow_transform[p_pass].bias_scale, light->param[VS::LIGHT_PARAM_SHADOW_BIAS_SPLIT_SCALE]);
		zfar = light->param[VS::LIGHT_PARAM_RANGE];
		bias = light->param[VS::LIGHT_PARAM_SHADOW_BIAS] * bias_mult;
		normal_bias = light->param[VS::LIGHT_PARAM_SHADOW_NORMAL_BIAS] * bias_mult;

		fbo = directional_shadow.fbo;
		vp_height = directional_shadow.size;
	} else {
		ShadowAtlas *shadow_atlas = shadow_atlas_owner.getornull(p_shadow_atlas);
		ERR_FAIL_COND(!shadow_atlas);
		ERR_FAIL_COND(!shadow_atlas->shadow_owners.has(p_light));

		fbo = shadow_atlas->fbo;
		vp_height = shadow_atlas->size;

		uint32_t key = shadow_atlas->shadow_owners[p_light];

		uint32_t quadrant = (key >> ShadowAtlas::QUADRANT_SHIFT) & 0x03;
		uint32_t shadow = key & ShadowAtlas::SHADOW_INDEX_MASK;

		ERR_FAIL_INDEX((int)shadow, shadow_atlas->quadrants[quadrant].shadows.size());

		uint32_t quadrant_size = shadow_atlas->size >> 1;

		x = (quadrant & 1) * quadrant_size;
		y = (quadrant >> 1) * quadrant_size;

		uint32_t shadow_size = (quadrant_size / shadow_atlas->quadrants[quadrant].subdivision);
		x += (shadow % shadow_atlas->quadrants[quadrant].subdivision) * shadow_size;
		y += (shadow / shadow_atlas->quadrants[quadrant].subdivision) * shadow_size;

		width = shadow_size;
		height = shadow_size;

		if (light->type == VS::LIGHT_OMNI) {
			// cubemap only
			if (light->omni_shadow_mode == VS::LIGHT_OMNI_SHADOW_CUBE) {
				int cubemap_index = shadow_cubemaps.size() - 1;

				// find an appropriate cubemap to render to
				for (int i = shadow_cubemaps.size() - 1; i >= 0; i--) {
					if (shadow_cubemaps[i].size > shadow_size * 2) {
						break;
					}

					cubemap_index = i;
				}

				fbo = shadow_cubemaps[cubemap_index].fbo[p_pass];
				light_projection = light_instance->shadow_transform[0].camera;
				light_transform = light_instance->shadow_transform[0].transform;

				custom_vp_size = shadow_cubemaps[cubemap_index].size;
				zfar = light->param[VS::LIGHT_PARAM_RANGE];

				current_cubemap = cubemap_index;
			}
		} else {
			light_projection = light_instance->shadow_transform[0].camera;
			light_transform = light_instance->shadow_transform[0].transform;

			flip_facing = false;
			zfar = light->param[VS::LIGHT_PARAM_RANGE];
			bias = light->param[VS::LIGHT_PARAM_SHADOW_BIAS];
			normal_bias = light->param[VS::LIGHT_PARAM_SHADOW_NORMAL_BIAS];
		}
	}

	render_list.clear();

	_fill_render_list(p_cull_result, p_cull_count, true, true);

	render_list.sort_by_depth(false);

	glDisable(GL_BLEND);
	glDisable(GL_DITHER);
	glEnable(GL_DEPTH_TEST);

	glBindFramebuffer(GL_FRAMEBUFFER, fbo);

	glDepthMask(GL_TRUE);
	glColorMask(0, 0, 0, 0);

	if (custom_vp_size) {
		glViewport(0, 0, custom_vp_size, custom_vp_size);
		glScissor(0, 0, custom_vp_size, custom_vp_size);
	} else {
		glViewport(x, y, width, height);
		glScissor(x, y, width, height);
	}

	glEnable(GL_SCISSOR_TEST);
	glClearDepth(1.0f);
	glClear(GL_DEPTH_BUFFER_BIT);
	glDisable(GL_SCISSOR_TEST);

	state.scene_shader.set_conditional(SceneShaderGLES2::RENDER_DEPTH, true);

	_render_render_list(render_list.elements, render_list.element_count, NULL, 0, light_transform, light_projection, RID(), NULL, 0, bias, normal_bias, false, false, true, false);

	state.scene_shader.set_conditional(SceneShaderGLES2::RENDER_DEPTH, false);

	// convert cubemap to dual paraboloid if needed
	if (light->type == VS::LIGHT_OMNI && light->omni_shadow_mode == VS::LIGHT_OMNI_SHADOW_CUBE && p_pass == 5) {
		ShadowAtlas *shadow_atlas = shadow_atlas_owner.getornull(p_shadow_atlas);

		glBindFramebuffer(GL_FRAMEBUFFER, shadow_atlas->fbo);
		state.cube_to_dp_shader.bind();

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_CUBE_MAP, shadow_cubemaps[current_cubemap].cubemap);

		glDisable(GL_CULL_FACE);

		for (int i = 0; i < 2; i++) {
			state.cube_to_dp_shader.set_uniform(CubeToDpShaderGLES2::Z_FLIP, i == 1);
			state.cube_to_dp_shader.set_uniform(CubeToDpShaderGLES2::Z_NEAR, light_projection.get_z_near());
			state.cube_to_dp_shader.set_uniform(CubeToDpShaderGLES2::Z_FAR, light_projection.get_z_far());
			state.cube_to_dp_shader.set_uniform(CubeToDpShaderGLES2::BIAS, light->param[VS::LIGHT_PARAM_SHADOW_BIAS]);

			uint32_t local_width = width;
			uint32_t local_height = height;
			uint32_t local_x = x;
			uint32_t local_y = y;

			if (light->omni_shadow_detail == VS::LIGHT_OMNI_SHADOW_DETAIL_HORIZONTAL) {
				local_height /= 2;
				local_y += i * local_height;
			} else {
				local_width /= 2;
				local_x += i * local_width;
			}

			glViewport(local_x, local_y, local_width, local_height);
			glScissor(local_x, local_y, local_width, local_height);

			glEnable(GL_SCISSOR_TEST);

			glClearDepth(1.0f);

			glClear(GL_DEPTH_BUFFER_BIT);
			glDisable(GL_SCISSOR_TEST);

			glDisable(GL_BLEND);

			storage->_copy_screen();
		}
	}

	glViewport(0, 0, storage->frame.current_rt->width, storage->frame.current_rt->height);
}

void RasterizerSceneGLES2::set_scene_pass(uint64_t p_pass) {
	scene_pass = p_pass;
}

bool RasterizerSceneGLES2::free(RID p_rid) {
	return true;
}

void RasterizerSceneGLES2::set_debug_draw_mode(VS::ViewportDebugDraw p_debug_draw) {
}

void RasterizerSceneGLES2::initialize() {
	state.scene_shader.init();
	state.cube_to_dp_shader.init();

	render_list.init();

	shadow_atlas_realloc_tolerance_msec = 500;

	{
		//default material and shader

		default_shader = storage->shader_create();
		storage->shader_set_code(default_shader, "shader_type spatial;\n");
		default_material = storage->material_create();
		storage->material_set_shader(default_material, default_shader);

		default_shader_twosided = storage->shader_create();
		default_material_twosided = storage->material_create();
		storage->shader_set_code(default_shader_twosided, "shader_type spatial; render_mode cull_disabled;\n");
		storage->material_set_shader(default_material_twosided, default_shader_twosided);
	}

	{
		glGenBuffers(1, &state.sky_verts);
		glBindBuffer(GL_ARRAY_BUFFER, state.sky_verts);
		glBufferData(GL_ARRAY_BUFFER, sizeof(Vector3) * 8, NULL, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	{
		uint32_t immediate_buffer_size = GLOBAL_DEF("rendering/limits/buffers/immediate_buffer_size_kb", 2048);

		glGenBuffers(1, &state.immediate_buffer);
		glBindBuffer(GL_ARRAY_BUFFER, state.immediate_buffer);
		glBufferData(GL_ARRAY_BUFFER, immediate_buffer_size * 1024, NULL, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	// cubemaps for shadows
	{
		int max_shadow_cubemap_sampler_size = 512;

		int cube_size = max_shadow_cubemap_sampler_size;

		glActiveTexture(GL_TEXTURE0);

		while (cube_size >= 32) {

			ShadowCubeMap cube;

			cube.size = cube_size;

			glGenTextures(1, &cube.cubemap);
			glBindTexture(GL_TEXTURE_CUBE_MAP, cube.cubemap);

			for (int i = 0; i < 6; i++) {
				glTexImage2D(_cube_side_enum[i], 0, GL_DEPTH_COMPONENT, cube_size, cube_size, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT, NULL);
			}

			glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
			glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

			glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

			glGenFramebuffers(6, cube.fbo);
			for (int i = 0; i < 6; i++) {

				glBindFramebuffer(GL_FRAMEBUFFER, cube.fbo[i]);
				glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, _cube_side_enum[i], cube.cubemap, 0);
			}

			shadow_cubemaps.push_back(cube);

			cube_size >>= 1;
		}
	}

	{
		// directional shadows

		directional_shadow.light_count = 0;
		directional_shadow.size = next_power_of_2(GLOBAL_GET("rendering/quality/directional_shadow/size"));

		glGenFramebuffers(1, &directional_shadow.fbo);
		glBindFramebuffer(GL_FRAMEBUFFER, directional_shadow.fbo);

		glGenTextures(1, &directional_shadow.depth);
		glBindTexture(GL_TEXTURE_2D, directional_shadow.depth);

		glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, directional_shadow.size, directional_shadow.size, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, NULL);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, directional_shadow.depth, 0);

		GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
		if (status != GL_FRAMEBUFFER_COMPLETE) {
			ERR_PRINT("Directional shadow framebuffer status invalid");
		}
	}
}

void RasterizerSceneGLES2::iteration() {
}

void RasterizerSceneGLES2::finalize() {
}

RasterizerSceneGLES2::RasterizerSceneGLES2() {
}
