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

#include "core/math/math_funcs.h"
#include "core/math/transform.h"
#include "core/os/os.h"
#include "core/project_settings.h"
#include "core/vmap.h"
#include "rasterizer_canvas_gles2.h"
#include "servers/visual/visual_server_raster.h"

#ifndef GLES_OVER_GL
#define glClearDepth glClearDepthf
#endif

#define _DEPTH_COMPONENT24_OES 0x81A6

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

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
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

	RasterizerStorageGLES2::ReflectionProbe *probe = storage->reflection_probe_owner.getornull(p_probe);
	ERR_FAIL_COND_V(!probe, RID());

	ReflectionProbeInstance *rpi = memnew(ReflectionProbeInstance);

	rpi->probe_ptr = probe;
	rpi->self = reflection_probe_instance_owner.make_rid(rpi);
	rpi->probe = p_probe;
	rpi->reflection_atlas_index = -1;
	rpi->render_step = -1;
	rpi->last_pass = 0;
	rpi->current_resolution = 0;
	rpi->dirty = true;

	rpi->last_pass = 0;
	rpi->index = 0;

	for (int i = 0; i < 6; i++) {
		glGenFramebuffers(1, &rpi->fbo[i]);
	}

	glGenFramebuffers(1, &rpi->fbo_blur);
	glGenRenderbuffers(1, &rpi->depth);
	rpi->cubemap = 0;
	//glGenTextures(1, &rpi->cubemap);

	return rpi->self;
}

void RasterizerSceneGLES2::reflection_probe_instance_set_transform(RID p_instance, const Transform &p_transform) {

	ReflectionProbeInstance *rpi = reflection_probe_instance_owner.getornull(p_instance);
	ERR_FAIL_COND(!rpi);
	rpi->transform = p_transform;
}

void RasterizerSceneGLES2::reflection_probe_release_atlas_index(RID p_instance) {
}

bool RasterizerSceneGLES2::reflection_probe_instance_needs_redraw(RID p_instance) {
	const ReflectionProbeInstance *rpi = reflection_probe_instance_owner.getornull(p_instance);
	ERR_FAIL_COND_V(!rpi, false);

	bool need_redraw = rpi->probe_ptr->resolution != rpi->current_resolution || rpi->dirty || rpi->probe_ptr->update_mode == VS::REFLECTION_PROBE_UPDATE_ALWAYS;
	rpi->dirty = false;
	return need_redraw;
}

bool RasterizerSceneGLES2::reflection_probe_instance_has_reflection(RID p_instance) {
	return true;
}

bool RasterizerSceneGLES2::reflection_probe_instance_begin_render(RID p_instance, RID p_reflection_atlas) {

	ReflectionProbeInstance *rpi = reflection_probe_instance_owner.getornull(p_instance);
	ERR_FAIL_COND_V(!rpi, false);

	rpi->render_step = 0;

	if (rpi->probe_ptr->resolution != rpi->current_resolution) {

		//update cubemap if resolution changed
		int size = rpi->probe_ptr->resolution;
		rpi->current_resolution = size;

		GLenum internal_format = GL_RGB;
		GLenum format = GL_RGB;
		GLenum type = GL_UNSIGNED_BYTE;

		glActiveTexture(GL_TEXTURE0);
		if (rpi->cubemap != 0) {
			glDeleteTextures(1, &rpi->cubemap);
		}
		glGenTextures(1, &rpi->cubemap);
		glBindTexture(GL_TEXTURE_CUBE_MAP, rpi->cubemap);
#if 1
		//Mobile hardware (PowerVR specially) prefers this approach, the other one kills the game
		for (int i = 0; i < 6; i++) {
			glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, internal_format, size, size, 0, format, type, NULL);
		}

		glGenerateMipmap(GL_TEXTURE_CUBE_MAP);

		glBindRenderbuffer(GL_RENDERBUFFER, rpi->depth); //resize depth buffer
		glRenderbufferStorage(GL_RENDERBUFFER, _DEPTH_COMPONENT24_OES, size, size);

		for (int i = 0; i < 6; i++) {
			glBindFramebuffer(GL_FRAMEBUFFER, rpi->fbo[i]);
			glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, _cube_side_enum[i], rpi->cubemap, 0);
			glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rpi->depth);
		}

#else
		int lod = 0;

		//the approach below is fatal for powervr

		// Set the initial (empty) mipmaps, all need to be set for this to work in GLES2, even if later wont be used.
		while (size >= 1) {

			for (int i = 0; i < 6; i++) {
				glTexImage2D(_cube_side_enum[i], lod, internal_format, size, size, 0, format, type, NULL);
				if (size == rpi->current_resolution) {
					//adjust framebuffer
					glBindFramebuffer(GL_FRAMEBUFFER, rpi->fbo[i]);
					glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, _cube_side_enum[i], rpi->cubemap, 0);
					glBindRenderbuffer(GL_RENDERBUFFER, rpi->depth);
					glRenderbufferStorage(GL_RENDERBUFFER, _DEPTH_COMPONENT24_OES, size, size);
					glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rpi->depth);

#ifdef DEBUG_ENABLED
					GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
					ERR_CONTINUE(status != GL_FRAMEBUFFER_COMPLETE);
#endif
				}
			}

			lod++;

			size >>= 1;
		}
#endif
		glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	}

	return true;
}

bool RasterizerSceneGLES2::reflection_probe_instance_postprocess_step(RID p_instance) {

	ReflectionProbeInstance *rpi = reflection_probe_instance_owner.getornull(p_instance);
	ERR_FAIL_COND_V(!rpi, false);

	int size = rpi->probe_ptr->resolution;

	{
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
		glDisable(GL_CULL_FACE);
		glDisable(GL_DEPTH_TEST);
		glDisable(GL_SCISSOR_TEST);
		glDisable(GL_BLEND);
		glDepthMask(GL_FALSE);

		for (int i = 0; i < VS::ARRAY_MAX - 1; i++) {
			glDisableVertexAttribArray(i);
		}
	}

	//vdc cache
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, storage->resources.radical_inverse_vdc_cache_tex);

	glBindFramebuffer(GL_FRAMEBUFFER, rpi->fbo_blur);
	// now render to the framebuffer, mipmap level for mipmap level
	int lod = 1;
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_CUBE_MAP, rpi->cubemap);
	glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR); //use linear, no mipmaps so it does not read from what is being written to

	size >>= 1;
	int mipmaps = 6;
	int mm_level = mipmaps - 1;

	storage->shaders.cubemap_filter.set_conditional(CubemapFilterShaderGLES2::USE_SOURCE_PANORAMA, false);
	storage->shaders.cubemap_filter.bind();

	//blur
	while (size >= 1) {

		for (int i = 0; i < 6; i++) {
			glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, _cube_side_enum[i], rpi->cubemap, lod);

			glViewport(0, 0, size, size);
			storage->bind_quad_array();
			storage->shaders.cubemap_filter.set_uniform(CubemapFilterShaderGLES2::FACE_ID, i);
			float roughness = CLAMP(lod / (float)(mipmaps - 1), 0, 1);
			storage->shaders.cubemap_filter.set_uniform(CubemapFilterShaderGLES2::ROUGHNESS, roughness);
			storage->shaders.cubemap_filter.set_uniform(CubemapFilterShaderGLES2::Z_FLIP, false);

			glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
		}

		size >>= 1;

		mm_level--;

		lod++;
	}

	// restore ranges

	glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	return true;
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

	env->fog_enabled = p_enable;
	env->fog_color = p_color;
	env->fog_sun_color = p_sun_color;
	env->fog_sun_amount = p_sun_amount;
}

void RasterizerSceneGLES2::environment_set_fog_depth(RID p_env, bool p_enable, float p_depth_begin, float p_depth_curve, bool p_transmit, float p_transmit_curve) {

	Environment *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND(!env);

	env->fog_depth_enabled = p_enable;
	env->fog_depth_begin = p_depth_begin;
	env->fog_depth_curve = p_depth_curve;
	env->fog_transmit_enabled = p_transmit;
	env->fog_transmit_curve = p_transmit_curve;
}

void RasterizerSceneGLES2::environment_set_fog_height(RID p_env, bool p_enable, float p_min_height, float p_max_height, float p_height_curve) {

	Environment *env = environment_owner.getornull(p_env);
	ERR_FAIL_COND(!env);

	env->fog_height_enabled = p_enable;
	env->fog_height_min = p_min_height;
	env->fog_height_max = p_max_height;
	env->fog_height_curve = p_height_curve;
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

	light_instance->light_index = 0xFFFF;

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

	bool mirror = p_instance->mirror;

	if (p_material->shader->spatial.cull_mode == RasterizerStorageGLES2::Shader::Spatial::CULL_MODE_DISABLED) {
		mirror = false;
	} else if (p_material->shader->spatial.cull_mode == RasterizerStorageGLES2::Shader::Spatial::CULL_MODE_FRONT) {
		mirror = !mirror;
	}

	//if (p_material->shader->spatial.uses_sss) {
	//	state.used_sss = true;
	//}

	if (p_material->shader->spatial.uses_screen_texture) {
		state.used_screen_texture = true;
	}

	if (p_depth_pass) {

		if (has_blend_alpha || p_material->shader->spatial.uses_depth_texture || (has_base_alpha && p_material->shader->spatial.depth_draw_mode != RasterizerStorageGLES2::Shader::Spatial::DEPTH_DRAW_ALPHA_PREPASS))
			return; //bye

		if (!p_material->shader->spatial.uses_alpha_scissor && !p_material->shader->spatial.writes_modelview_or_projection && !p_material->shader->spatial.uses_vertex && !p_material->shader->spatial.uses_discard && p_material->shader->spatial.depth_draw_mode != RasterizerStorageGLES2::Shader::Spatial::DEPTH_DRAW_ALPHA_PREPASS) {
			//shader does not use discard and does not write a vertex position, use generic material
			if (p_instance->cast_shadows == VS::SHADOW_CASTING_SETTING_DOUBLE_SIDED) {
				p_material = storage->material_owner.getptr(!p_shadow_pass && p_material->shader->spatial.uses_world_coordinates ? default_worldcoord_material_twosided : default_material_twosided);
				mirror = false;
			} else {
				p_material = storage->material_owner.getptr(!p_shadow_pass && p_material->shader->spatial.uses_world_coordinates ? default_worldcoord_material : default_material);
			}
		}

		has_alpha = false;
	}

	RenderList::Element *e = has_alpha ? render_list.add_alpha_element() : render_list.add_element();

	if (!e) {
		return;
	}

	e->geometry = p_geometry;
	e->material = p_material;
	e->instance = p_instance;
	e->owner = p_owner;
	e->sort_key = 0;
	e->depth_key = 0;
	e->use_accum = false;
	e->light_index = RenderList::MAX_LIGHTS;
	e->use_accum_ptr = &e->use_accum;
	e->instancing = (e->instance->base_type == VS::INSTANCE_MULTIMESH) ? 1 : 0;

	if (e->geometry->last_pass != render_pass) {
		e->geometry->last_pass = render_pass;
		e->geometry->index = current_geometry_index++;
	}

	e->geometry_index = e->geometry->index;

	if (e->material->last_pass != render_pass) {
		e->material->last_pass = render_pass;
		e->material->index = current_material_index++;

		if (e->material->shader->last_pass != render_pass) {
			e->material->shader->index = current_shader_index++;
		}
	}

	e->material_index = e->material->index;

	e->refprobe_0_index = RenderList::MAX_REFLECTION_PROBES; //refprobe disabled by default
	e->refprobe_1_index = RenderList::MAX_REFLECTION_PROBES; //refprobe disabled by default

	if (!p_depth_pass) {

		e->depth_layer = e->instance->depth_layer;
		e->priority = p_material->render_priority;

		int rpsize = e->instance->reflection_probe_instances.size();
		if (rpsize > 0) {
			bool first = true;
			rpsize = MIN(rpsize, 2); //more than 2 per object are not supported, this keeps it stable

			for (int i = 0; i < rpsize; i++) {
				ReflectionProbeInstance *rpi = reflection_probe_instance_owner.getornull(e->instance->reflection_probe_instances[i]);
				if (rpi->last_pass != render_pass) {
					continue;
				}
				if (first) {
					e->refprobe_0_index = rpi->index;
					first = false;
				} else {
					e->refprobe_1_index = rpi->index;
					break;
				}
			}

			/*	if (e->refprobe_0_index > e->refprobe_1_index) { //if both are valid, swap them to keep order as best as possible
				uint64_t tmp = e->refprobe_0_index;
				e->refprobe_0_index = e->refprobe_1_index;
				e->refprobe_1_index = tmp;
			}*/
		}

		//add directional lights

		if (p_material->shader->spatial.unshaded) {
			e->light_mode = LIGHTMODE_UNSHADED;
		} else {

			bool copy = false;

			for (int i = 0; i < render_directional_lights; i++) {

				if (copy) {
					RenderList::Element *e2 = has_alpha ? render_list.add_alpha_element() : render_list.add_element();
					if (!e2) {
						break;
					}
					*e2 = *e; //this includes accum ptr :)
					e = e2;
				}

				//directional sort key
				e->light_type1 = 0;
				e->light_type2 = 1;
				e->light_index = i;

				copy = true;
			}

			//add omni / spots

			for (int i = 0; i < e->instance->light_instances.size(); i++) {

				LightInstance *li = light_instance_owner.getornull(e->instance->light_instances[i]);

				if (li->light_index >= render_light_instance_count) {
					continue; // too many
				}

				if (copy) {
					RenderList::Element *e2 = has_alpha ? render_list.add_alpha_element() : render_list.add_element();
					if (!e2) {
						break;
					}
					*e2 = *e; //this includes accum ptr :)
					e = e2;
				}

				//directional sort key
				e->light_type1 = 1;
				e->light_type2 = li->light_ptr->type == VisualServer::LIGHT_OMNI ? 0 : 1;
				e->light_index = li->light_index;

				copy = true;
			}

			if (e->instance->lightmap.is_valid()) {
				e->light_mode = LIGHTMODE_LIGHTMAP;
			} else if (!e->instance->lightmap_capture_data.empty()) {
				e->light_mode = LIGHTMODE_LIGHTMAP_CAPTURE;
			} else {
				e->light_mode = LIGHTMODE_NORMAL;
			}
		}
	}

	// do not add anything here, as lights are duplicated elements..

	if (p_material->shader->spatial.uses_time) {
		VisualServerRaster::redraw_request();
	}
}

void RasterizerSceneGLES2::_fill_render_list(InstanceBase **p_cull_result, int p_cull_count, bool p_depth_pass, bool p_shadow_pass) {

	render_pass++;
	current_material_index = 0;
	current_geometry_index = 0;
	current_light_index = 0;
	current_refprobe_index = 0;
	current_shader_index = 0;

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

			default: {}
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

bool RasterizerSceneGLES2::_setup_material(RasterizerStorageGLES2::Material *p_material, bool p_reverse_cull, bool p_alpha_pass, Size2i p_skeleton_tex_size) {

	// material parameters

	state.scene_shader.set_custom_shader(p_material->shader->custom_code_id);

	bool shader_rebind = state.scene_shader.bind();

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

	return shader_rebind;
}

void RasterizerSceneGLES2::_setup_geometry(RenderList::Element *p_element, RasterizerStorageGLES2::Skeleton *p_skeleton) {

	switch (p_element->instance->base_type) {

		case VS::INSTANCE_MESH: {
			RasterizerStorageGLES2::Surface *s = static_cast<RasterizerStorageGLES2::Surface *>(p_element->geometry);

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
					switch (i) {
						case VS::ARRAY_NORMAL: {
							glVertexAttrib4f(VS::ARRAY_NORMAL, 0.0, 0.0, 1, 1);
						} break;
						case VS::ARRAY_COLOR: {
							glVertexAttrib4f(VS::ARRAY_COLOR, 1, 1, 1, 1);

						} break;
						default: {}
					}
				}
			}

			bool clear_skeleton_buffer = !storage->config.float_texture_supported;

			if (p_skeleton) {

				if (storage->config.float_texture_supported) {
					//use float texture workflow
					glActiveTexture(GL_TEXTURE0 + storage->config.max_texture_image_units - 1);
					glBindTexture(GL_TEXTURE_2D, p_skeleton->tex_id);
				} else {
					//use transform buffer workflow
					ERR_FAIL_COND(p_skeleton->use_2d);

					PoolVector<float> &transform_buffer = storage->resources.skeleton_transform_cpu_buffer;

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

					//enable transform buffer and bind it
					glBindBuffer(GL_ARRAY_BUFFER, storage->resources.skeleton_transform_buffer);

					glEnableVertexAttribArray(INSTANCE_BONE_BASE + 0);
					glEnableVertexAttribArray(INSTANCE_BONE_BASE + 1);
					glEnableVertexAttribArray(INSTANCE_BONE_BASE + 2);

					glVertexAttribPointer(INSTANCE_BONE_BASE + 0, 4, GL_FLOAT, GL_FALSE, sizeof(float) * 12, (const void *)(sizeof(float) * 4 * 0));
					glVertexAttribPointer(INSTANCE_BONE_BASE + 1, 4, GL_FLOAT, GL_FALSE, sizeof(float) * 12, (const void *)(sizeof(float) * 4 * 1));
					glVertexAttribPointer(INSTANCE_BONE_BASE + 2, 4, GL_FLOAT, GL_FALSE, sizeof(float) * 12, (const void *)(sizeof(float) * 4 * 2));

					clear_skeleton_buffer = false;
				}
			}

			if (clear_skeleton_buffer) {

				glDisableVertexAttribArray(INSTANCE_BONE_BASE + 0);
				glDisableVertexAttribArray(INSTANCE_BONE_BASE + 1);
				glDisableVertexAttribArray(INSTANCE_BONE_BASE + 2);
			}

		} break;

		case VS::INSTANCE_MULTIMESH: {
			RasterizerStorageGLES2::Surface *s = static_cast<RasterizerStorageGLES2::Surface *>(p_element->geometry);

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
					switch (i) {
						case VS::ARRAY_NORMAL: {
							glVertexAttrib4f(VS::ARRAY_NORMAL, 0.0, 0.0, 1, 1);
						} break;
						case VS::ARRAY_COLOR: {
							glVertexAttrib4f(VS::ARRAY_COLOR, 1, 1, 1, 1);

						} break;
						default: {}
					}
				}
			}

			// prepare multimesh (disable)
			glDisableVertexAttribArray(INSTANCE_ATTRIB_BASE + 0);
			glDisableVertexAttribArray(INSTANCE_ATTRIB_BASE + 1);
			glDisableVertexAttribArray(INSTANCE_ATTRIB_BASE + 2);
			glDisableVertexAttribArray(INSTANCE_ATTRIB_BASE + 3);
			glDisableVertexAttribArray(INSTANCE_ATTRIB_BASE + 4);
			glDisableVertexAttribArray(INSTANCE_BONE_BASE + 0);
			glDisableVertexAttribArray(INSTANCE_BONE_BASE + 1);
			glDisableVertexAttribArray(INSTANCE_BONE_BASE + 2);

		} break;

		case VS::INSTANCE_IMMEDIATE: {
		} break;

		default: {}
	}
}

void RasterizerSceneGLES2::_render_geometry(RenderList::Element *p_element) {

	switch (p_element->instance->base_type) {

		case VS::INSTANCE_MESH: {

			RasterizerStorageGLES2::Surface *s = static_cast<RasterizerStorageGLES2::Surface *>(p_element->geometry);

			// drawing

			if (s->index_array_len > 0) {
				glDrawElements(gl_primitive[s->primitive], s->index_array_len, (s->array_len >= (1 << 16)) ? GL_UNSIGNED_INT : GL_UNSIGNED_SHORT, 0);
			} else {
				glDrawArrays(gl_primitive[s->primitive], 0, s->array_len);
			}
			/*
			if (p_element->instance->skeleton.is_valid() && s->attribs[VS::ARRAY_BONES].enabled && s->attribs[VS::ARRAY_WEIGHTS].enabled) {
				//clean up after skeleton
				glBindBuffer(GL_ARRAY_BUFFER, storage->resources.skeleton_transform_buffer);

				glDisableVertexAttribArray(VS::ARRAY_MAX + 0);
				glDisableVertexAttribArray(VS::ARRAY_MAX + 1);
				glDisableVertexAttribArray(VS::ARRAY_MAX + 2);

				glVertexAttrib4f(VS::ARRAY_MAX + 0, 1, 0, 0, 0);
				glVertexAttrib4f(VS::ARRAY_MAX + 1, 0, 1, 0, 0);
				glVertexAttrib4f(VS::ARRAY_MAX + 2, 0, 0, 1, 0);
			}
*/
		} break;

		case VS::INSTANCE_MULTIMESH: {

			RasterizerStorageGLES2::MultiMesh *multi_mesh = static_cast<RasterizerStorageGLES2::MultiMesh *>(p_element->owner);
			RasterizerStorageGLES2::Surface *s = static_cast<RasterizerStorageGLES2::Surface *>(p_element->geometry);

			int amount = MIN(multi_mesh->size, multi_mesh->visible_instances);

			if (amount == -1) {
				amount = multi_mesh->size;
			}

			int stride = multi_mesh->color_floats + multi_mesh->custom_data_floats + multi_mesh->xform_floats;

			int color_ofs = multi_mesh->xform_floats;
			int custom_data_ofs = color_ofs + multi_mesh->color_floats;

			// drawing

			const float *base_buffer = multi_mesh->data.ptr();

			for (int i = 0; i < amount; i++) {
				const float *buffer = base_buffer + i * stride;

				{

					glVertexAttrib4fv(INSTANCE_ATTRIB_BASE + 0, &buffer[0]);
					glVertexAttrib4fv(INSTANCE_ATTRIB_BASE + 1, &buffer[4]);
					glVertexAttrib4fv(INSTANCE_ATTRIB_BASE + 2, &buffer[8]);
				}

				if (multi_mesh->color_floats) {
					if (multi_mesh->color_format == VS::MULTIMESH_COLOR_8BIT) {
						uint8_t *color_data = (uint8_t *)(buffer + color_ofs);
						glVertexAttrib4f(INSTANCE_ATTRIB_BASE + 3, color_data[0] / 255.0, color_data[1] / 255.0, color_data[2] / 255.0, color_data[3] / 255.0);
					} else {
						glVertexAttrib4fv(INSTANCE_ATTRIB_BASE + 3, buffer + color_ofs);
					}
				}

				if (multi_mesh->custom_data_floats) {
					if (multi_mesh->custom_data_format == VS::MULTIMESH_CUSTOM_DATA_8BIT) {
						uint8_t *custom_data = (uint8_t *)(buffer + custom_data_ofs);
						glVertexAttrib4f(INSTANCE_ATTRIB_BASE + 4, custom_data[0] / 255.0, custom_data[1] / 255.0, custom_data[2] / 255.0, custom_data[3] / 255.0);
					} else {
						glVertexAttrib4fv(INSTANCE_ATTRIB_BASE + 4, buffer + custom_data_ofs);
					}
				}

				if (s->index_array_len > 0) {
					glDrawElements(gl_primitive[s->primitive], s->index_array_len, (s->array_len >= (1 << 16)) ? GL_UNSIGNED_INT : GL_UNSIGNED_SHORT, 0);
				} else {
					glDrawArrays(gl_primitive[s->primitive], 0, s->array_len);
				}
			}

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
		default: {}
	}
}

void RasterizerSceneGLES2::_setup_light_type(LightInstance *p_light, ShadowAtlas *shadow_atlas) {

	//turn off all by default
	state.scene_shader.set_conditional(SceneShaderGLES2::USE_LIGHTING, false);
	state.scene_shader.set_conditional(SceneShaderGLES2::USE_SHADOW, false);
	state.scene_shader.set_conditional(SceneShaderGLES2::SHADOW_MODE_PCF_5, false);
	state.scene_shader.set_conditional(SceneShaderGLES2::SHADOW_MODE_PCF_13, false);
	state.scene_shader.set_conditional(SceneShaderGLES2::LIGHT_MODE_DIRECTIONAL, false);
	state.scene_shader.set_conditional(SceneShaderGLES2::LIGHT_MODE_OMNI, false);
	state.scene_shader.set_conditional(SceneShaderGLES2::LIGHT_MODE_SPOT, false);
	state.scene_shader.set_conditional(SceneShaderGLES2::LIGHT_USE_PSSM2, false);
	state.scene_shader.set_conditional(SceneShaderGLES2::LIGHT_USE_PSSM4, false);
	state.scene_shader.set_conditional(SceneShaderGLES2::LIGHT_USE_PSSM_BLEND, false);
	state.scene_shader.set_conditional(SceneShaderGLES2::USE_SHADOW, false);

	if (!p_light) { //no light, return off
		return;
	}

	//turn on lighting
	state.scene_shader.set_conditional(SceneShaderGLES2::USE_LIGHTING, true);

	switch (p_light->light_ptr->type) {
		case VS::LIGHT_DIRECTIONAL: {

			state.scene_shader.set_conditional(SceneShaderGLES2::LIGHT_MODE_DIRECTIONAL, true);
			switch (p_light->light_ptr->directional_shadow_mode) {
				case VS::LIGHT_DIRECTIONAL_SHADOW_ORTHOGONAL: {
					//no need
				} break;
				case VS::LIGHT_DIRECTIONAL_SHADOW_PARALLEL_2_SPLITS: {
					state.scene_shader.set_conditional(SceneShaderGLES2::LIGHT_USE_PSSM2, true);

				} break;
				case VS::LIGHT_DIRECTIONAL_SHADOW_PARALLEL_4_SPLITS: {
					state.scene_shader.set_conditional(SceneShaderGLES2::LIGHT_USE_PSSM4, true);
				} break;
			}

			state.scene_shader.set_conditional(SceneShaderGLES2::LIGHT_USE_PSSM_BLEND, p_light->light_ptr->directional_blend_splits);
			if (!state.render_no_shadows && p_light->light_ptr->shadow) {
				state.scene_shader.set_conditional(SceneShaderGLES2::USE_SHADOW, true);
				glActiveTexture(GL_TEXTURE0 + storage->config.max_texture_image_units - 3);
				glBindTexture(GL_TEXTURE_2D, directional_shadow.depth);
				state.scene_shader.set_conditional(SceneShaderGLES2::SHADOW_MODE_PCF_5, shadow_filter_mode == SHADOW_FILTER_PCF5);
				state.scene_shader.set_conditional(SceneShaderGLES2::SHADOW_MODE_PCF_13, shadow_filter_mode == SHADOW_FILTER_PCF13);
			}

		} break;
		case VS::LIGHT_OMNI: {

			state.scene_shader.set_conditional(SceneShaderGLES2::LIGHT_MODE_OMNI, true);
			if (!state.render_no_shadows && shadow_atlas && p_light->light_ptr->shadow) {
				state.scene_shader.set_conditional(SceneShaderGLES2::USE_SHADOW, true);
				glActiveTexture(GL_TEXTURE0 + storage->config.max_texture_image_units - 3);
				glBindTexture(GL_TEXTURE_2D, shadow_atlas->depth);
				state.scene_shader.set_conditional(SceneShaderGLES2::SHADOW_MODE_PCF_5, shadow_filter_mode == SHADOW_FILTER_PCF5);
				state.scene_shader.set_conditional(SceneShaderGLES2::SHADOW_MODE_PCF_13, shadow_filter_mode == SHADOW_FILTER_PCF13);
			}
		} break;
		case VS::LIGHT_SPOT: {

			state.scene_shader.set_conditional(SceneShaderGLES2::LIGHT_MODE_SPOT, true);
			if (!state.render_no_shadows && shadow_atlas && p_light->light_ptr->shadow) {
				state.scene_shader.set_conditional(SceneShaderGLES2::USE_SHADOW, true);
				glActiveTexture(GL_TEXTURE0 + storage->config.max_texture_image_units - 3);
				glBindTexture(GL_TEXTURE_2D, shadow_atlas->depth);
				state.scene_shader.set_conditional(SceneShaderGLES2::SHADOW_MODE_PCF_5, shadow_filter_mode == SHADOW_FILTER_PCF5);
				state.scene_shader.set_conditional(SceneShaderGLES2::SHADOW_MODE_PCF_13, shadow_filter_mode == SHADOW_FILTER_PCF13);
			}
		} break;
	}
}

void RasterizerSceneGLES2::_setup_light(LightInstance *light, ShadowAtlas *shadow_atlas, const Transform &p_view_transform) {

	RasterizerStorageGLES2::Light *light_ptr = light->light_ptr;

	//common parameters
	float energy = light_ptr->param[VS::LIGHT_PARAM_ENERGY];
	float specular = light_ptr->param[VS::LIGHT_PARAM_SPECULAR];
	float sign = light_ptr->negative ? -1 : 1;

	state.scene_shader.set_uniform(SceneShaderGLES2::LIGHT_SPECULAR, specular);
	Color color = light_ptr->color * sign * energy * Math_PI;
	state.scene_shader.set_uniform(SceneShaderGLES2::LIGHT_COLOR, color);

	//specific parameters

	switch (light_ptr->type) {
		case VS::LIGHT_DIRECTIONAL: {
			//not using inverse for performance, view should be normalized anyway
			Vector3 direction = p_view_transform.basis.xform_inv(light->transform.basis.xform(Vector3(0, 0, -1))).normalized();
			state.scene_shader.set_uniform(SceneShaderGLES2::LIGHT_DIRECTION, direction);

			CameraMatrix matrices[4];

			if (!state.render_no_shadows && light_ptr->shadow && directional_shadow.depth) {

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

					Transform modelview = (p_view_transform.inverse() * light->shadow_transform[k].transform).affine_inverse();

					CameraMatrix bias;
					bias.set_light_bias();
					CameraMatrix rectm;
					Rect2 atlas_rect = Rect2(float(x) / directional_shadow.size, float(y) / directional_shadow.size, float(width) / directional_shadow.size, float(height) / directional_shadow.size);
					rectm.set_light_atlas_rect(atlas_rect);

					CameraMatrix shadow_mtx = rectm * bias * light->shadow_transform[k].camera * modelview;
					matrices[k] = shadow_mtx;

					/*Color light_clamp;
					light_clamp[0] = atlas_rect.position.x;
					light_clamp[1] = atlas_rect.position.y;
					light_clamp[2] = atlas_rect.size.x;
					light_clamp[3] = atlas_rect.size.y;*/
				}

				//	state.scene_shader.set_uniform(SceneShaderGLES2::LIGHT_CLAMP, light_clamp);
				state.scene_shader.set_uniform(SceneShaderGLES2::SHADOW_PIXEL_SIZE, Size2(1.0 / directional_shadow.size, 1.0 / directional_shadow.size));
				state.scene_shader.set_uniform(SceneShaderGLES2::LIGHT_SPLIT_OFFSETS, split_offsets);
				state.scene_shader.set_uniform(SceneShaderGLES2::LIGHT_SHADOW_MATRIX, matrices[0]);
				state.scene_shader.set_uniform(SceneShaderGLES2::LIGHT_SHADOW_MATRIX2, matrices[1]);
				state.scene_shader.set_uniform(SceneShaderGLES2::LIGHT_SHADOW_MATRIX3, matrices[2]);
				state.scene_shader.set_uniform(SceneShaderGLES2::LIGHT_SHADOW_MATRIX4, matrices[3]);
			}
		} break;
		case VS::LIGHT_OMNI: {

			Vector3 position = p_view_transform.xform_inv(light->transform.origin);

			state.scene_shader.set_uniform(SceneShaderGLES2::LIGHT_POSITION, position);

			float range = light_ptr->param[VS::LIGHT_PARAM_RANGE];
			state.scene_shader.set_uniform(SceneShaderGLES2::LIGHT_RANGE, range);

			float attenuation = light_ptr->param[VS::LIGHT_PARAM_ATTENUATION];
			state.scene_shader.set_uniform(SceneShaderGLES2::LIGHT_ATTENUATION, attenuation);

			if (!state.render_no_shadows && light_ptr->shadow && shadow_atlas && shadow_atlas->shadow_owners.has(light->self)) {

				uint32_t key = shadow_atlas->shadow_owners[light->self];

				uint32_t quadrant = (key >> ShadowAtlas::QUADRANT_SHIFT) & 0x03;
				uint32_t shadow = key & ShadowAtlas::SHADOW_INDEX_MASK;

				ERR_BREAK(shadow >= (uint32_t)shadow_atlas->quadrants[quadrant].shadows.size());

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

				state.scene_shader.set_uniform(SceneShaderGLES2::SHADOW_PIXEL_SIZE, Size2(1.0 / shadow_atlas->size, 1.0 / shadow_atlas->size));
				state.scene_shader.set_uniform(SceneShaderGLES2::LIGHT_SHADOW_MATRIX, proj);
				state.scene_shader.set_uniform(SceneShaderGLES2::LIGHT_CLAMP, light_clamp);
			}
		} break;

		case VS::LIGHT_SPOT: {

			Vector3 position = p_view_transform.xform_inv(light->transform.origin);

			state.scene_shader.set_uniform(SceneShaderGLES2::LIGHT_POSITION, position);

			Vector3 direction = p_view_transform.inverse().basis.xform(light->transform.basis.xform(Vector3(0, 0, -1))).normalized();
			state.scene_shader.set_uniform(SceneShaderGLES2::LIGHT_DIRECTION, direction);
			float attenuation = light_ptr->param[VS::LIGHT_PARAM_ATTENUATION];
			float range = light_ptr->param[VS::LIGHT_PARAM_RANGE];
			float spot_attenuation = light_ptr->param[VS::LIGHT_PARAM_SPOT_ATTENUATION];
			float angle = light_ptr->param[VS::LIGHT_PARAM_SPOT_ANGLE];
			angle = Math::cos(Math::deg2rad(angle));
			state.scene_shader.set_uniform(SceneShaderGLES2::LIGHT_ATTENUATION, attenuation);
			state.scene_shader.set_uniform(SceneShaderGLES2::LIGHT_SPOT_ATTENUATION, spot_attenuation);
			state.scene_shader.set_uniform(SceneShaderGLES2::LIGHT_SPOT_RANGE, spot_attenuation);
			state.scene_shader.set_uniform(SceneShaderGLES2::LIGHT_SPOT_ANGLE, angle);
			state.scene_shader.set_uniform(SceneShaderGLES2::LIGHT_RANGE, range);

			if (!state.render_no_shadows && light->light_ptr->shadow && shadow_atlas && shadow_atlas->shadow_owners.has(light->self)) {
				uint32_t key = shadow_atlas->shadow_owners[light->self];

				uint32_t quadrant = (key >> ShadowAtlas::QUADRANT_SHIFT) & 0x03;
				uint32_t shadow = key & ShadowAtlas::SHADOW_INDEX_MASK;

				ERR_BREAK(shadow >= (uint32_t)shadow_atlas->quadrants[quadrant].shadows.size());

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

				state.scene_shader.set_uniform(SceneShaderGLES2::SHADOW_PIXEL_SIZE, Size2(1.0 / shadow_atlas->size, 1.0 / shadow_atlas->size));
				state.scene_shader.set_uniform(SceneShaderGLES2::LIGHT_SHADOW_MATRIX, shadow_matrix);
				state.scene_shader.set_uniform(SceneShaderGLES2::LIGHT_CLAMP, light_clamp);
			}

		} break;
		default: {}
	}
}

void RasterizerSceneGLES2::_setup_refprobes(ReflectionProbeInstance *p_refprobe1, ReflectionProbeInstance *p_refprobe2, const Transform &p_view_transform, Environment *p_env) {

	if (p_refprobe1) {
		state.scene_shader.set_uniform(SceneShaderGLES2::REFPROBE1_USE_BOX_PROJECT, p_refprobe1->probe_ptr->box_projection);
		state.scene_shader.set_uniform(SceneShaderGLES2::REFPROBE1_BOX_EXTENTS, p_refprobe1->probe_ptr->extents);
		state.scene_shader.set_uniform(SceneShaderGLES2::REFPROBE1_BOX_OFFSET, p_refprobe1->probe_ptr->origin_offset);
		state.scene_shader.set_uniform(SceneShaderGLES2::REFPROBE1_EXTERIOR, !p_refprobe1->probe_ptr->interior);
		state.scene_shader.set_uniform(SceneShaderGLES2::REFPROBE1_INTENSITY, p_refprobe1->probe_ptr->intensity);

		Color ambient;
		if (p_refprobe1->probe_ptr->interior) {
			ambient = p_refprobe1->probe_ptr->interior_ambient * p_refprobe1->probe_ptr->interior_ambient_energy;
			ambient.a = p_refprobe1->probe_ptr->interior_ambient_probe_contrib;
		} else if (p_env) {
			ambient = p_env->ambient_color * p_env->ambient_energy;
			ambient.a = p_env->ambient_sky_contribution;
		}

		state.scene_shader.set_uniform(SceneShaderGLES2::REFPROBE1_AMBIENT, ambient);

		Transform proj = (p_view_transform.inverse() * p_refprobe1->transform).affine_inverse();

		state.scene_shader.set_uniform(SceneShaderGLES2::REFPROBE1_LOCAL_MATRIX, proj);
	}

	if (p_refprobe2) {
		state.scene_shader.set_uniform(SceneShaderGLES2::REFPROBE2_USE_BOX_PROJECT, p_refprobe2->probe_ptr->box_projection);
		state.scene_shader.set_uniform(SceneShaderGLES2::REFPROBE2_BOX_EXTENTS, p_refprobe2->probe_ptr->extents);
		state.scene_shader.set_uniform(SceneShaderGLES2::REFPROBE2_BOX_OFFSET, p_refprobe2->probe_ptr->origin_offset);
		state.scene_shader.set_uniform(SceneShaderGLES2::REFPROBE2_EXTERIOR, !p_refprobe2->probe_ptr->interior);
		state.scene_shader.set_uniform(SceneShaderGLES2::REFPROBE2_INTENSITY, p_refprobe2->probe_ptr->intensity);

		Color ambient;
		if (p_refprobe2->probe_ptr->interior) {
			ambient = p_refprobe2->probe_ptr->interior_ambient * p_refprobe2->probe_ptr->interior_ambient_energy;
			ambient.a = p_refprobe2->probe_ptr->interior_ambient_probe_contrib;
		} else if (p_env) {
			ambient = p_env->ambient_color * p_env->ambient_energy;
			ambient.a = p_env->ambient_sky_contribution;
		}

		state.scene_shader.set_uniform(SceneShaderGLES2::REFPROBE2_AMBIENT, ambient);

		Transform proj = (p_view_transform.inverse() * p_refprobe2->transform).affine_inverse();

		state.scene_shader.set_uniform(SceneShaderGLES2::REFPROBE2_LOCAL_MATRIX, proj);
	}
}

void RasterizerSceneGLES2::_render_render_list(RenderList::Element **p_elements, int p_element_count, const Transform &p_view_transform, const CameraMatrix &p_projection, RID p_shadow_atlas, Environment *p_env, GLuint p_base_env, float p_shadow_bias, float p_shadow_normal_bias, bool p_reverse_cull, bool p_alpha_pass, bool p_shadow) {

	ShadowAtlas *shadow_atlas = shadow_atlas_owner.getornull(p_shadow_atlas);

	Vector2 screen_pixel_size = state.screen_pixel_size;

	bool use_radiance_map = false;
	if (!p_shadow && p_base_env) {
		glActiveTexture(GL_TEXTURE0 + storage->config.max_texture_image_units - 2);
		glBindTexture(GL_TEXTURE_CUBE_MAP, p_base_env);
		use_radiance_map = true;
		state.scene_shader.set_conditional(SceneShaderGLES2::USE_RADIANCE_MAP, true); //since prev unshaded is false, this needs to be true if exists
	}

	bool prev_unshaded = false;
	bool prev_instancing = false;
	state.scene_shader.set_conditional(SceneShaderGLES2::SHADELESS, false);
	RasterizerStorageGLES2::Material *prev_material = NULL;
	RasterizerStorageGLES2::Geometry *prev_geometry = NULL;
	RasterizerStorageGLES2::Skeleton *prev_skeleton = NULL;
	RasterizerStorageGLES2::GeometryOwner *prev_owner = NULL;

	Transform view_transform_inverse = p_view_transform.inverse();
	CameraMatrix projection_inverse = p_projection.inverse();

	bool prev_base_pass = false;
	LightInstance *prev_light = NULL;
	bool prev_vertex_lit = false;
	ReflectionProbeInstance *prev_refprobe_1 = NULL;
	ReflectionProbeInstance *prev_refprobe_2 = NULL;

	int prev_blend_mode = -2; //will always catch the first go

	if (p_alpha_pass) {
		glEnable(GL_BLEND);
	} else {
		glDisable(GL_BLEND);
	}

	float fog_max_distance = 0;
	bool using_fog = false;
	if (p_env && !p_shadow && p_env->fog_enabled && (p_env->fog_depth_enabled || p_env->fog_height_enabled)) {
		state.scene_shader.set_conditional(SceneShaderGLES2::FOG_DEPTH_ENABLED, p_env->fog_depth_enabled);
		state.scene_shader.set_conditional(SceneShaderGLES2::FOG_HEIGHT_ENABLED, p_env->fog_height_enabled);
		fog_max_distance = p_projection.get_z_far();
		using_fog = true;
	}

	RasterizerStorageGLES2::Texture *prev_lightmap = NULL;
	float lightmap_energy = 1.0;
	bool prev_use_lightmap_capture = false;

	for (int i = 0; i < p_element_count; i++) {
		RenderList::Element *e = p_elements[i];

		RasterizerStorageGLES2::Material *material = e->material;

		bool rebind = false;
		bool accum_pass = *e->use_accum_ptr;
		*e->use_accum_ptr = true; //set to accum for next time this is found
		LightInstance *light = NULL;
		ReflectionProbeInstance *refprobe_1 = NULL;
		ReflectionProbeInstance *refprobe_2 = NULL;
		RasterizerStorageGLES2::Texture *lightmap = NULL;
		bool use_lightmap_capture = false;
		bool rebind_light = false;
		bool rebind_reflection = false;
		bool rebind_lightmap = false;

		if (!p_shadow) {

			bool unshaded = material->shader->spatial.unshaded;

			if (unshaded != prev_unshaded) {
				rebind = true;
				if (unshaded) {
					state.scene_shader.set_conditional(SceneShaderGLES2::SHADELESS, true);
					state.scene_shader.set_conditional(SceneShaderGLES2::USE_RADIANCE_MAP, false);
					state.scene_shader.set_conditional(SceneShaderGLES2::USE_LIGHTING, false);
				} else {
					state.scene_shader.set_conditional(SceneShaderGLES2::SHADELESS, false);
					state.scene_shader.set_conditional(SceneShaderGLES2::USE_RADIANCE_MAP, use_radiance_map);
				}

				prev_unshaded = unshaded;
			}

			bool base_pass = !accum_pass && !unshaded; //conditions for a base pass

			if (base_pass != prev_base_pass) {
				state.scene_shader.set_conditional(SceneShaderGLES2::BASE_PASS, base_pass);
				rebind = true;
				prev_base_pass = base_pass;
			}

			if (!unshaded && e->light_index < RenderList::MAX_LIGHTS) {
				light = render_light_instances[e->light_index];
			}

			if (light != prev_light) {

				_setup_light_type(light, shadow_atlas);
				rebind = true;
				rebind_light = true;
			}

			int blend_mode = p_alpha_pass ? material->shader->spatial.blend_mode : -1; // -1 no blend, no mix

			if (accum_pass) { //accum pass force pass
				blend_mode = RasterizerStorageGLES2::Shader::Spatial::BLEND_MODE_ADD;
			}

			if (prev_blend_mode != blend_mode) {

				if (prev_blend_mode == -1 && blend_mode != -1) {
					//does blend
					glEnable(GL_BLEND);
				} else if (blend_mode == -1 && prev_blend_mode != -1) {
					//do not blend
					glDisable(GL_BLEND);
				}

				switch (blend_mode) {
					//-1 not handled because not blend is enabled anyway
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

				prev_blend_mode = blend_mode;
			}

			//condition to enable vertex lighting on this object
			bool vertex_lit = (material->shader->spatial.uses_vertex_lighting || storage->config.force_vertex_shading) && ((!unshaded && light) || using_fog); //fog forces vertex lighting because it still applies even if unshaded or no fog

			if (vertex_lit != prev_vertex_lit) {
				state.scene_shader.set_conditional(SceneShaderGLES2::USE_VERTEX_LIGHTING, vertex_lit);
				prev_vertex_lit = vertex_lit;
			}

			if (!unshaded && !accum_pass && e->refprobe_0_index != RenderList::MAX_REFLECTION_PROBES) {
				ERR_FAIL_INDEX(e->refprobe_0_index, reflection_probe_count);
				refprobe_1 = reflection_probe_instances[e->refprobe_0_index];
			}
			if (!unshaded && !accum_pass && e->refprobe_1_index != RenderList::MAX_REFLECTION_PROBES) {
				ERR_FAIL_INDEX(e->refprobe_1_index, reflection_probe_count);
				refprobe_2 = reflection_probe_instances[e->refprobe_1_index];
			}

			if (refprobe_1 != prev_refprobe_1 || refprobe_2 != prev_refprobe_2) {
				state.scene_shader.set_conditional(SceneShaderGLES2::USE_REFLECTION_PROBE1, refprobe_1 != NULL);
				state.scene_shader.set_conditional(SceneShaderGLES2::USE_REFLECTION_PROBE2, refprobe_2 != NULL);
				if (refprobe_1 != NULL && refprobe_1 != prev_refprobe_1) {
					glActiveTexture(GL_TEXTURE0 + storage->config.max_texture_image_units - 5);
					glBindTexture(GL_TEXTURE_CUBE_MAP, refprobe_1->cubemap);
				}
				if (refprobe_2 != NULL && refprobe_2 != prev_refprobe_2) {
					glActiveTexture(GL_TEXTURE0 + storage->config.max_texture_image_units - 6);
					glBindTexture(GL_TEXTURE_CUBE_MAP, refprobe_2->cubemap);
				}
				rebind = true;
				rebind_reflection = true;
			}

			use_lightmap_capture = !unshaded && !accum_pass && !e->instance->lightmap_capture_data.empty();

			if (use_lightmap_capture != prev_use_lightmap_capture) {

				state.scene_shader.set_conditional(SceneShaderGLES2::USE_LIGHTMAP_CAPTURE, use_lightmap_capture);
				rebind = true;
			}

			if (!unshaded && !accum_pass && e->instance->lightmap.is_valid()) {

				lightmap = storage->texture_owner.getornull(e->instance->lightmap);
				lightmap_energy = 1.0;
				if (lightmap) {
					RasterizerStorageGLES2::LightmapCapture *capture = storage->lightmap_capture_data_owner.getornull(e->instance->lightmap_capture->base);
					if (capture) {
						lightmap_energy = capture->energy;
					}
				}
			}

			if (lightmap != prev_lightmap) {
				state.scene_shader.set_conditional(SceneShaderGLES2::USE_LIGHTMAP, lightmap != NULL);
				if (lightmap != NULL) {
					glActiveTexture(GL_TEXTURE0 + storage->config.max_texture_image_units - 4);
					glBindTexture(GL_TEXTURE_2D, lightmap->tex_id);
				}
				rebind = true;
				rebind_lightmap = true;
			}
		}

		bool instancing = e->instance->base_type == VS::INSTANCE_MULTIMESH;

		if (instancing != prev_instancing) {

			state.scene_shader.set_conditional(SceneShaderGLES2::USE_INSTANCING, instancing);
			rebind = true;
		}

		RasterizerStorageGLES2::Skeleton *skeleton = storage->skeleton_owner.getornull(e->instance->skeleton);

		if (skeleton != prev_skeleton) {

			if (skeleton) {
				state.scene_shader.set_conditional(SceneShaderGLES2::USE_SKELETON, true);
				state.scene_shader.set_conditional(SceneShaderGLES2::USE_SKELETON_SOFTWARE, !storage->config.float_texture_supported);
			} else {
				state.scene_shader.set_conditional(SceneShaderGLES2::USE_SKELETON, false);
				state.scene_shader.set_conditional(SceneShaderGLES2::USE_SKELETON_SOFTWARE, false);
			}

			rebind = true;
		}

		if (e->owner != prev_owner || e->geometry != prev_geometry || skeleton != prev_skeleton) {
			_setup_geometry(e, skeleton);
		}

		bool shader_rebind = false;
		if (rebind || material != prev_material) {
			shader_rebind = _setup_material(material, p_reverse_cull, p_alpha_pass, Size2i(skeleton ? skeleton->size * 3 : 0, 0));
		}

		if (i == 0 || shader_rebind) { //first time must rebind

			if (p_shadow) {
				state.scene_shader.set_uniform(SceneShaderGLES2::LIGHT_BIAS, p_shadow_bias);
				state.scene_shader.set_uniform(SceneShaderGLES2::LIGHT_NORMAL_BIAS, p_shadow_normal_bias);
				if (state.shadow_is_dual_parabolloid) {
					state.scene_shader.set_uniform(SceneShaderGLES2::SHADOW_DUAL_PARABOLOID_RENDER_SIDE, state.dual_parbolloid_direction);
					state.scene_shader.set_uniform(SceneShaderGLES2::SHADOW_DUAL_PARABOLOID_RENDER_ZFAR, state.dual_parbolloid_zfar);
				}
			} else {
				if (use_radiance_map) {
					state.scene_shader.set_uniform(SceneShaderGLES2::RADIANCE_INVERSE_XFORM, p_view_transform);
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

				//rebind all these
				rebind_light = true;
				rebind_reflection = true;
				rebind_lightmap = true;

				if (using_fog) {

					state.scene_shader.set_uniform(SceneShaderGLES2::FOG_COLOR_BASE, p_env->fog_color);
					Color sun_color_amount = p_env->fog_sun_color;
					sun_color_amount.a = p_env->fog_sun_amount;

					state.scene_shader.set_uniform(SceneShaderGLES2::FOG_SUN_COLOR_AMOUNT, sun_color_amount);
					state.scene_shader.set_uniform(SceneShaderGLES2::FOG_TRANSMIT_ENABLED, p_env->fog_transmit_enabled);
					state.scene_shader.set_uniform(SceneShaderGLES2::FOG_TRANSMIT_CURVE, p_env->fog_transmit_curve);

					if (p_env->fog_depth_enabled) {
						state.scene_shader.set_uniform(SceneShaderGLES2::FOG_DEPTH_BEGIN, p_env->fog_depth_begin);
						state.scene_shader.set_uniform(SceneShaderGLES2::FOG_DEPTH_CURVE, p_env->fog_depth_curve);
						state.scene_shader.set_uniform(SceneShaderGLES2::FOG_MAX_DISTANCE, fog_max_distance);
					}

					if (p_env->fog_height_enabled) {
						state.scene_shader.set_uniform(SceneShaderGLES2::FOG_HEIGHT_MIN, p_env->fog_height_min);
						state.scene_shader.set_uniform(SceneShaderGLES2::FOG_HEIGHT_MAX, p_env->fog_height_max);
						state.scene_shader.set_uniform(SceneShaderGLES2::FOG_HEIGHT_MAX, p_env->fog_height_max);
						state.scene_shader.set_uniform(SceneShaderGLES2::FOG_HEIGHT_CURVE, p_env->fog_height_curve);
					}
				}
			}

			state.scene_shader.set_uniform(SceneShaderGLES2::CAMERA_MATRIX, p_view_transform);
			state.scene_shader.set_uniform(SceneShaderGLES2::CAMERA_INVERSE_MATRIX, view_transform_inverse);
			state.scene_shader.set_uniform(SceneShaderGLES2::PROJECTION_MATRIX, p_projection);
			state.scene_shader.set_uniform(SceneShaderGLES2::PROJECTION_INVERSE_MATRIX, projection_inverse);

			state.scene_shader.set_uniform(SceneShaderGLES2::TIME, storage->frame.time[0]);

			state.scene_shader.set_uniform(SceneShaderGLES2::SCREEN_PIXEL_SIZE, screen_pixel_size);
			state.scene_shader.set_uniform(SceneShaderGLES2::NORMAL_MULT, 1.0); // TODO mirror?
		}

		if (rebind_light && light) {
			_setup_light(light, shadow_atlas, p_view_transform);
		}

		if (rebind_reflection && (refprobe_1 || refprobe_2)) {
			_setup_refprobes(refprobe_1, refprobe_2, p_view_transform, p_env);
		}

		if (rebind_lightmap && lightmap) {
			state.scene_shader.set_uniform(SceneShaderGLES2::LIGHTMAP_ENERGY, lightmap_energy);
		}

		state.scene_shader.set_uniform(SceneShaderGLES2::WORLD_TRANSFORM, e->instance->transform);

		if (use_lightmap_capture) { //this is per instance, must be set always if present
			glUniform4fv(state.scene_shader.get_uniform_location(SceneShaderGLES2::LIGHTMAP_CAPTURES), 12, (const GLfloat *)e->instance->lightmap_capture_data.ptr());
			state.scene_shader.set_uniform(SceneShaderGLES2::LIGHTMAP_CAPTURE_SKY, false);
		}

		_render_geometry(e);

		prev_geometry = e->geometry;
		prev_owner = e->owner;
		prev_material = material;
		prev_skeleton = skeleton;
		prev_instancing = instancing;
		prev_light = light;
		prev_refprobe_1 = refprobe_1;
		prev_refprobe_2 = refprobe_2;
		prev_lightmap = lightmap;
		prev_use_lightmap_capture = use_lightmap_capture;
	}

	_setup_light_type(NULL, NULL); //clear light stuff
	state.scene_shader.set_conditional(SceneShaderGLES2::USE_SKELETON, false);
	state.scene_shader.set_conditional(SceneShaderGLES2::SHADELESS, false);
	state.scene_shader.set_conditional(SceneShaderGLES2::BASE_PASS, false);
	state.scene_shader.set_conditional(SceneShaderGLES2::USE_INSTANCING, false);
	state.scene_shader.set_conditional(SceneShaderGLES2::USE_RADIANCE_MAP, false);
	state.scene_shader.set_conditional(SceneShaderGLES2::LIGHT_USE_PSSM4, false);
	state.scene_shader.set_conditional(SceneShaderGLES2::LIGHT_USE_PSSM2, false);
	state.scene_shader.set_conditional(SceneShaderGLES2::LIGHT_USE_PSSM_BLEND, false);
	state.scene_shader.set_conditional(SceneShaderGLES2::USE_VERTEX_LIGHTING, false);
	state.scene_shader.set_conditional(SceneShaderGLES2::USE_REFLECTION_PROBE1, false);
	state.scene_shader.set_conditional(SceneShaderGLES2::USE_REFLECTION_PROBE2, false);
	state.scene_shader.set_conditional(SceneShaderGLES2::USE_LIGHTMAP, false);
	state.scene_shader.set_conditional(SceneShaderGLES2::USE_LIGHTMAP_CAPTURE, false);
	state.scene_shader.set_conditional(SceneShaderGLES2::FOG_DEPTH_ENABLED, false);
	state.scene_shader.set_conditional(SceneShaderGLES2::FOG_HEIGHT_ENABLED, false);
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

	storage->shaders.copy.set_conditional(CopyShaderGLES2::USE_ASYM_PANO, asymmetrical);
	storage->shaders.copy.set_conditional(CopyShaderGLES2::USE_PANORAMA, !asymmetrical);
	storage->shaders.copy.set_conditional(CopyShaderGLES2::USE_MULTIPLIER, true);
	storage->shaders.copy.set_conditional(CopyShaderGLES2::USE_CUBEMAP, false);
	storage->shaders.copy.set_conditional(CopyShaderGLES2::USE_COPY_SECTION, false);
	storage->shaders.copy.set_conditional(CopyShaderGLES2::USE_CUSTOM_ALPHA, false);
	storage->shaders.copy.bind();
	storage->shaders.copy.set_uniform(CopyShaderGLES2::MULTIPLIER, p_energy);
	if (asymmetrical) {
		// pack the bits we need from our projection matrix
		storage->shaders.copy.set_uniform(CopyShaderGLES2::ASYM_PROJ, camera.matrix[2][0], camera.matrix[0][0], camera.matrix[2][1], camera.matrix[1][1]);
		///@TODO I couldn't get mat3 + p_transform.basis to work, that would be better here.
		storage->shaders.copy.set_uniform(CopyShaderGLES2::PANO_TRANSFORM, p_transform);
	}

	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

	glDisableVertexAttribArray(VS::ARRAY_VERTEX);
	glDisableVertexAttribArray(VS::ARRAY_TEX_UV);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	storage->shaders.copy.set_conditional(CopyShaderGLES2::USE_ASYM_PANO, false);
	storage->shaders.copy.set_conditional(CopyShaderGLES2::USE_PANORAMA, false);
	storage->shaders.copy.set_conditional(CopyShaderGLES2::USE_MULTIPLIER, false);
	storage->shaders.copy.set_conditional(CopyShaderGLES2::USE_CUBEMAP, false);
}

void RasterizerSceneGLES2::render_scene(const Transform &p_cam_transform, const CameraMatrix &p_cam_projection, bool p_cam_ortogonal, InstanceBase **p_cull_result, int p_cull_count, RID *p_light_cull_result, int p_light_cull_count, RID *p_reflection_probe_cull_result, int p_reflection_probe_cull_count, RID p_environment, RID p_shadow_atlas, RID p_reflection_atlas, RID p_reflection_probe, int p_reflection_probe_pass) {

	GLuint current_fb = 0;
	Environment *env = NULL;

	int viewport_width, viewport_height;

	if (p_reflection_probe.is_valid()) {
		ReflectionProbeInstance *probe = reflection_probe_instance_owner.getornull(p_reflection_probe);
		ERR_FAIL_COND(!probe);
		state.render_no_shadows = !probe->probe_ptr->enable_shadows;

		if (!probe->probe_ptr->interior) { //use env only if not interior
			env = environment_owner.getornull(p_environment);
		}

		current_fb = probe->fbo[p_reflection_probe_pass];
		state.screen_pixel_size.x = 1.0 / probe->probe_ptr->resolution;
		state.screen_pixel_size.y = 1.0 / probe->probe_ptr->resolution;

		viewport_width = probe->probe_ptr->resolution;
		viewport_height = probe->probe_ptr->resolution;

	} else {
		state.render_no_shadows = false;
		current_fb = storage->frame.current_rt->fbo;
		env = environment_owner.getornull(p_environment);
		state.screen_pixel_size.x = 1.0 / storage->frame.current_rt->width;
		state.screen_pixel_size.y = 1.0 / storage->frame.current_rt->height;
		viewport_width = storage->frame.current_rt->width;
		viewport_height = storage->frame.current_rt->height;
	}
	//push back the directional lights

	if (p_light_cull_count) {
		//harcoded limit of 256 lights
		render_light_instance_count = MIN(RenderList::MAX_LIGHTS, p_light_cull_count);
		render_light_instances = (LightInstance **)alloca(sizeof(LightInstance *) * render_light_instance_count);
		render_directional_lights = 0;

		//doing this because directional lights are at the end, put them at the beginning
		int index = 0;
		for (int i = render_light_instance_count - 1; i >= 0; i--) {
			RID light_rid = p_light_cull_result[i];

			LightInstance *light = light_instance_owner.getornull(light_rid);

			if (light->light_ptr->type == VS::LIGHT_DIRECTIONAL) {
				render_directional_lights++;
				//as goin in reverse, directional lights are always first anyway
			}

			light->light_index = index;
			render_light_instances[index] = light;

			index++;
		}

	} else {
		render_light_instances = NULL;
		render_directional_lights = 0;
		render_light_instance_count = 0;
	}

	if (p_reflection_probe_cull_count) {

		reflection_probe_instances = (ReflectionProbeInstance **)alloca(sizeof(ReflectionProbeInstance *) * p_reflection_probe_cull_count);
		reflection_probe_count = p_reflection_probe_cull_count;
		for (int i = 0; i < p_reflection_probe_cull_count; i++) {
			ReflectionProbeInstance *rpi = reflection_probe_instance_owner.getornull(p_reflection_probe_cull_result[i]);
			ERR_CONTINUE(!rpi);
			rpi->last_pass = render_pass + 1; //will be incremented later
			rpi->index = i;
			reflection_probe_instances[i] = rpi;
		}

	} else {
		reflection_probe_instances = NULL;
		reflection_probe_count = 0;
	}

	// render list stuff

	render_list.clear();
	_fill_render_list(p_cull_result, p_cull_count, false, false);

	// other stuff

	glBindFramebuffer(GL_FRAMEBUFFER, current_fb);
	glViewport(0, 0, viewport_width, viewport_height);

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

	// render opaque things first
	render_list.sort_by_key(false);
	_render_render_list(render_list.elements, render_list.element_count, p_cam_transform, p_cam_projection, p_shadow_atlas, env, env_radiance_tex, 0.0, 0.0, false, false, false);

	// alpha pass

	glBlendEquation(GL_FUNC_ADD);
	glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

	render_list.sort_by_depth(true);

	_render_render_list(&render_list.elements[render_list.max_elements - render_list.alpha_element_count], render_list.alpha_element_count, p_cam_transform, p_cam_projection, p_shadow_atlas, env, env_radiance_tex, 0.0, 0.0, false, true, false);

	glDisable(GL_DEPTH_TEST);

	//#define GLES2_SHADOW_ATLAS_DEBUG_VIEW

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

	//#define GLES2_SHADOW_DIRECTIONAL_DEBUG_VIEW

#ifdef GLES2_SHADOW_DIRECTIONAL_DEBUG_VIEW
	if (true) {
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, directional_shadow.depth);

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

	state.render_no_shadows = false;

	LightInstance *light_instance = light_instance_owner.getornull(p_light);
	ERR_FAIL_COND(!light_instance);

	RasterizerStorageGLES2::Light *light = light_instance->light_ptr;
	ERR_FAIL_COND(!light);

	uint32_t x;
	uint32_t y;
	uint32_t width;
	uint32_t height;

	float zfar = 0;
	bool flip_facing = false;
	int custom_vp_size = 0;
	GLuint fbo = 0;
	state.shadow_is_dual_parabolloid = false;
	state.dual_parbolloid_direction = 0.0;

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
	} else {
		ShadowAtlas *shadow_atlas = shadow_atlas_owner.getornull(p_shadow_atlas);
		ERR_FAIL_COND(!shadow_atlas);
		ERR_FAIL_COND(!shadow_atlas->shadow_owners.has(p_light));

		fbo = shadow_atlas->fbo;

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
			} else {
				//dual parabolloid
				state.shadow_is_dual_parabolloid = true;
				light_projection = light_instance->shadow_transform[0].camera;
				light_transform = light_instance->shadow_transform[0].transform;

				if (light->omni_shadow_detail == VS::LIGHT_OMNI_SHADOW_DETAIL_HORIZONTAL) {

					height /= 2;
					y += p_pass * height;
				} else {
					width /= 2;
					x += p_pass * width;
				}

				state.dual_parbolloid_direction = p_pass == 0 ? 1.0 : -1.0;
				flip_facing = (p_pass == 1);
				zfar = light->param[VS::LIGHT_PARAM_RANGE];
				bias = light->param[VS::LIGHT_PARAM_SHADOW_BIAS];

				state.dual_parbolloid_zfar = zfar;

				state.scene_shader.set_conditional(SceneShaderGLES2::RENDER_DEPTH_DUAL_PARABOLOID, true);
			}

		} else if (light->type == VS::LIGHT_SPOT) {
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

	if (light->reverse_cull) {
		flip_facing = !flip_facing;
	}

	state.scene_shader.set_conditional(SceneShaderGLES2::RENDER_DEPTH, true);

	_render_render_list(render_list.elements, render_list.element_count, light_transform, light_projection, RID(), NULL, 0, bias, normal_bias, flip_facing, false, true);

	state.scene_shader.set_conditional(SceneShaderGLES2::RENDER_DEPTH, false);
	state.scene_shader.set_conditional(SceneShaderGLES2::RENDER_DEPTH_DUAL_PARABOLOID, false);

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
	glColorMask(1, 1, 1, 1);
}

void RasterizerSceneGLES2::set_scene_pass(uint64_t p_pass) {
	scene_pass = p_pass;
}

bool RasterizerSceneGLES2::free(RID p_rid) {

	if (light_instance_owner.owns(p_rid)) {

		LightInstance *light_instance = light_instance_owner.getptr(p_rid);

		//remove from shadow atlases..
		for (Set<RID>::Element *E = light_instance->shadow_atlases.front(); E; E = E->next()) {
			ShadowAtlas *shadow_atlas = shadow_atlas_owner.get(E->get());
			ERR_CONTINUE(!shadow_atlas->shadow_owners.has(p_rid));
			uint32_t key = shadow_atlas->shadow_owners[p_rid];
			uint32_t q = (key >> ShadowAtlas::QUADRANT_SHIFT) & 0x3;
			uint32_t s = key & ShadowAtlas::SHADOW_INDEX_MASK;

			shadow_atlas->quadrants[q].shadows.write[s].owner = RID();
			shadow_atlas->shadow_owners.erase(p_rid);
		}

		light_instance_owner.free(p_rid);
		memdelete(light_instance);

	} else if (shadow_atlas_owner.owns(p_rid)) {

		ShadowAtlas *shadow_atlas = shadow_atlas_owner.get(p_rid);
		shadow_atlas_set_size(p_rid, 0);
		shadow_atlas_owner.free(p_rid);
		memdelete(shadow_atlas);
	} else if (reflection_probe_instance_owner.owns(p_rid)) {

		ReflectionProbeInstance *reflection_instance = reflection_probe_instance_owner.get(p_rid);

		reflection_probe_release_atlas_index(p_rid);
		reflection_probe_instance_owner.free(p_rid);
		memdelete(reflection_instance);

	} else {
		return false;
	}

	return true;
}

void RasterizerSceneGLES2::set_debug_draw_mode(VS::ViewportDebugDraw p_debug_draw) {
}

void RasterizerSceneGLES2::initialize() {
	state.scene_shader.init();
	state.cube_to_dp_shader.init();

	render_list.init();

	render_pass = 1;

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
		default_worldcoord_shader = storage->shader_create();
		storage->shader_set_code(default_worldcoord_shader, "shader_type spatial; render_mode world_vertex_coords;\n");
		default_worldcoord_material = storage->material_create();
		storage->material_set_shader(default_worldcoord_material, default_worldcoord_shader);

		default_worldcoord_shader_twosided = storage->shader_create();
		default_worldcoord_material_twosided = storage->material_create();
		storage->shader_set_code(default_worldcoord_shader_twosided, "shader_type spatial; render_mode cull_disabled,world_vertex_coords;\n");
		storage->material_set_shader(default_worldcoord_material_twosided, default_worldcoord_shader_twosided);
	}

	{
		//default material and shader

		default_overdraw_shader = storage->shader_create();
		storage->shader_set_code(default_overdraw_shader, "shader_type spatial;\nrender_mode blend_add,unshaded;\n void fragment() { ALBEDO=vec3(0.4,0.8,0.8); ALPHA=0.2; }");
		default_overdraw_material = storage->material_create();
		storage->material_set_shader(default_overdraw_material, default_overdraw_shader);
	}

	{
		glGenBuffers(1, &state.sky_verts);
		glBindBuffer(GL_ARRAY_BUFFER, state.sky_verts);
		glBufferData(GL_ARRAY_BUFFER, sizeof(Vector3) * 8, NULL, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	{
		uint32_t immediate_buffer_size = GLOBAL_DEF("rendering/limits/buffers/immediate_buffer_size_kb", 2048);
		ProjectSettings::get_singleton()->set_custom_property_info("rendering/limits/buffers/immediate_buffer_size_kb", PropertyInfo(Variant::INT, "rendering/limits/buffers/immediate_buffer_size_kb", PROPERTY_HINT_RANGE, "0,8192,1,or_greater"));

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
				glTexImage2D(_cube_side_enum[i], 0, GL_DEPTH_COMPONENT, cube_size, cube_size, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, NULL);
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

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, directional_shadow.depth, 0);

		GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
		if (status != GL_FRAMEBUFFER_COMPLETE) {
			ERR_PRINT("Directional shadow framebuffer status invalid");
		}
	}

	shadow_filter_mode = SHADOW_FILTER_NEAREST;
}

void RasterizerSceneGLES2::iteration() {
	shadow_filter_mode = ShadowFilterMode(int(GLOBAL_GET("rendering/quality/shadows/filter_mode")));
}

void RasterizerSceneGLES2::finalize() {
}

RasterizerSceneGLES2::RasterizerSceneGLES2() {
}
