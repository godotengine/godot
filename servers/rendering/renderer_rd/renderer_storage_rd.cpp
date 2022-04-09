/*************************************************************************/
/*  renderer_storage_rd.cpp                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "renderer_storage_rd.h"

#include "core/config/engine.h"
#include "core/config/project_settings.h"
#include "core/io/resource_loader.h"
#include "core/math/math_defs.h"
#include "renderer_compositor_rd.h"
#include "servers/rendering/renderer_rd/storage_rd/canvas_texture_storage.h"
#include "servers/rendering/renderer_rd/storage_rd/decal_atlas_storage.h"
#include "servers/rendering/renderer_rd/storage_rd/mesh_storage.h"
#include "servers/rendering/renderer_rd/storage_rd/texture_storage.h"
#include "servers/rendering/rendering_server_globals.h"
#include "servers/rendering/shader_language.h"

/* CANVAS TEXTURE */

void RendererStorageRD::sampler_rd_configure_custom(float p_mipmap_bias) {
	for (int i = 1; i < RS::CANVAS_ITEM_TEXTURE_FILTER_MAX; i++) {
		for (int j = 1; j < RS::CANVAS_ITEM_TEXTURE_REPEAT_MAX; j++) {
			RD::SamplerState sampler_state;
			switch (i) {
				case RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST: {
					sampler_state.mag_filter = RD::SAMPLER_FILTER_NEAREST;
					sampler_state.min_filter = RD::SAMPLER_FILTER_NEAREST;
					sampler_state.max_lod = 0;
				} break;
				case RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR: {
					sampler_state.mag_filter = RD::SAMPLER_FILTER_LINEAR;
					sampler_state.min_filter = RD::SAMPLER_FILTER_LINEAR;
					sampler_state.max_lod = 0;
				} break;
				case RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST_WITH_MIPMAPS: {
					sampler_state.mag_filter = RD::SAMPLER_FILTER_NEAREST;
					sampler_state.min_filter = RD::SAMPLER_FILTER_NEAREST;
					if (GLOBAL_GET("rendering/textures/default_filters/use_nearest_mipmap_filter")) {
						sampler_state.mip_filter = RD::SAMPLER_FILTER_NEAREST;
					} else {
						sampler_state.mip_filter = RD::SAMPLER_FILTER_LINEAR;
					}
					sampler_state.lod_bias = p_mipmap_bias;
				} break;
				case RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS: {
					sampler_state.mag_filter = RD::SAMPLER_FILTER_LINEAR;
					sampler_state.min_filter = RD::SAMPLER_FILTER_LINEAR;
					if (GLOBAL_GET("rendering/textures/default_filters/use_nearest_mipmap_filter")) {
						sampler_state.mip_filter = RD::SAMPLER_FILTER_NEAREST;
					} else {
						sampler_state.mip_filter = RD::SAMPLER_FILTER_LINEAR;
					}
					sampler_state.lod_bias = p_mipmap_bias;

				} break;
				case RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST_WITH_MIPMAPS_ANISOTROPIC: {
					sampler_state.mag_filter = RD::SAMPLER_FILTER_NEAREST;
					sampler_state.min_filter = RD::SAMPLER_FILTER_NEAREST;
					if (GLOBAL_GET("rendering/textures/default_filters/use_nearest_mipmap_filter")) {
						sampler_state.mip_filter = RD::SAMPLER_FILTER_NEAREST;
					} else {
						sampler_state.mip_filter = RD::SAMPLER_FILTER_LINEAR;
					}
					sampler_state.lod_bias = p_mipmap_bias;
					sampler_state.use_anisotropy = true;
					sampler_state.anisotropy_max = 1 << int(GLOBAL_GET("rendering/textures/default_filters/anisotropic_filtering_level"));
				} break;
				case RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS_ANISOTROPIC: {
					sampler_state.mag_filter = RD::SAMPLER_FILTER_LINEAR;
					sampler_state.min_filter = RD::SAMPLER_FILTER_LINEAR;
					if (GLOBAL_GET("rendering/textures/default_filters/use_nearest_mipmap_filter")) {
						sampler_state.mip_filter = RD::SAMPLER_FILTER_NEAREST;
					} else {
						sampler_state.mip_filter = RD::SAMPLER_FILTER_LINEAR;
					}
					sampler_state.lod_bias = p_mipmap_bias;
					sampler_state.use_anisotropy = true;
					sampler_state.anisotropy_max = 1 << int(GLOBAL_GET("rendering/textures/default_filters/anisotropic_filtering_level"));

				} break;
				default: {
				}
			}
			switch (j) {
				case RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED: {
					sampler_state.repeat_u = RD::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;
					sampler_state.repeat_v = RD::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;
					sampler_state.repeat_w = RD::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;

				} break;
				case RS::CANVAS_ITEM_TEXTURE_REPEAT_ENABLED: {
					sampler_state.repeat_u = RD::SAMPLER_REPEAT_MODE_REPEAT;
					sampler_state.repeat_v = RD::SAMPLER_REPEAT_MODE_REPEAT;
					sampler_state.repeat_w = RD::SAMPLER_REPEAT_MODE_REPEAT;
				} break;
				case RS::CANVAS_ITEM_TEXTURE_REPEAT_MIRROR: {
					sampler_state.repeat_u = RD::SAMPLER_REPEAT_MODE_MIRRORED_REPEAT;
					sampler_state.repeat_v = RD::SAMPLER_REPEAT_MODE_MIRRORED_REPEAT;
					sampler_state.repeat_w = RD::SAMPLER_REPEAT_MODE_MIRRORED_REPEAT;
				} break;
				default: {
				}
			}

			if (custom_rd_samplers[i][j].is_valid()) {
				RD::get_singleton()->free(custom_rd_samplers[i][j]);
			}

			custom_rd_samplers[i][j] = RD::get_singleton()->sampler_create(sampler_state);
		}
	}
}

/* PARTICLES */

RID RendererStorageRD::particles_allocate() {
	return particles_owner.allocate_rid();
}
void RendererStorageRD::particles_initialize(RID p_rid) {
	particles_owner.initialize_rid(p_rid, Particles());
}

void RendererStorageRD::particles_set_mode(RID p_particles, RS::ParticlesMode p_mode) {
	Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND(!particles);
	if (particles->mode == p_mode) {
		return;
	}

	_particles_free_data(particles);

	particles->mode = p_mode;
}

void RendererStorageRD::particles_set_emitting(RID p_particles, bool p_emitting) {
	Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND(!particles);

	particles->emitting = p_emitting;
}

bool RendererStorageRD::particles_get_emitting(RID p_particles) {
	ERR_FAIL_COND_V_MSG(RSG::threaded, false, "This function should never be used with threaded rendering, as it stalls the renderer.");
	Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND_V(!particles, false);

	return particles->emitting;
}

void RendererStorageRD::_particles_free_data(Particles *particles) {
	if (particles->particle_buffer.is_valid()) {
		RD::get_singleton()->free(particles->particle_buffer);
		particles->particle_buffer = RID();
		RD::get_singleton()->free(particles->particle_instance_buffer);
		particles->particle_instance_buffer = RID();
	}

	particles->userdata_count = 0;

	if (particles->frame_params_buffer.is_valid()) {
		RD::get_singleton()->free(particles->frame_params_buffer);
		particles->frame_params_buffer = RID();
	}
	particles->particles_transforms_buffer_uniform_set = RID();

	if (RD::get_singleton()->uniform_set_is_valid(particles->trail_bind_pose_uniform_set)) {
		RD::get_singleton()->free(particles->trail_bind_pose_uniform_set);
	}
	particles->trail_bind_pose_uniform_set = RID();

	if (particles->trail_bind_pose_buffer.is_valid()) {
		RD::get_singleton()->free(particles->trail_bind_pose_buffer);
		particles->trail_bind_pose_buffer = RID();
	}
	if (RD::get_singleton()->uniform_set_is_valid(particles->collision_textures_uniform_set)) {
		RD::get_singleton()->free(particles->collision_textures_uniform_set);
	}
	particles->collision_textures_uniform_set = RID();

	if (particles->particles_sort_buffer.is_valid()) {
		RD::get_singleton()->free(particles->particles_sort_buffer);
		particles->particles_sort_buffer = RID();
		particles->particles_sort_uniform_set = RID();
	}

	if (particles->emission_buffer != nullptr) {
		particles->emission_buffer = nullptr;
		particles->emission_buffer_data.clear();
		RD::get_singleton()->free(particles->emission_storage_buffer);
		particles->emission_storage_buffer = RID();
	}

	if (RD::get_singleton()->uniform_set_is_valid(particles->particles_material_uniform_set)) {
		//will need to be re-created
		RD::get_singleton()->free(particles->particles_material_uniform_set);
	}
	particles->particles_material_uniform_set = RID();
}

void RendererStorageRD::particles_set_amount(RID p_particles, int p_amount) {
	Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND(!particles);

	if (particles->amount == p_amount) {
		return;
	}

	_particles_free_data(particles);

	particles->amount = p_amount;

	particles->prev_ticks = 0;
	particles->phase = 0;
	particles->prev_phase = 0;
	particles->clear = true;

	particles->dependency.changed_notify(DEPENDENCY_CHANGED_PARTICLES);
}

void RendererStorageRD::particles_set_lifetime(RID p_particles, double p_lifetime) {
	Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND(!particles);
	particles->lifetime = p_lifetime;
}

void RendererStorageRD::particles_set_one_shot(RID p_particles, bool p_one_shot) {
	Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND(!particles);
	particles->one_shot = p_one_shot;
}

void RendererStorageRD::particles_set_pre_process_time(RID p_particles, double p_time) {
	Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND(!particles);
	particles->pre_process_time = p_time;
}
void RendererStorageRD::particles_set_explosiveness_ratio(RID p_particles, real_t p_ratio) {
	Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND(!particles);
	particles->explosiveness = p_ratio;
}
void RendererStorageRD::particles_set_randomness_ratio(RID p_particles, real_t p_ratio) {
	Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND(!particles);
	particles->randomness = p_ratio;
}

void RendererStorageRD::particles_set_custom_aabb(RID p_particles, const AABB &p_aabb) {
	Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND(!particles);
	particles->custom_aabb = p_aabb;
	particles->dependency.changed_notify(DEPENDENCY_CHANGED_AABB);
}

void RendererStorageRD::particles_set_speed_scale(RID p_particles, double p_scale) {
	Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND(!particles);

	particles->speed_scale = p_scale;
}
void RendererStorageRD::particles_set_use_local_coordinates(RID p_particles, bool p_enable) {
	Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND(!particles);

	particles->use_local_coords = p_enable;
	particles->dependency.changed_notify(DEPENDENCY_CHANGED_PARTICLES);
}

void RendererStorageRD::particles_set_fixed_fps(RID p_particles, int p_fps) {
	Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND(!particles);

	particles->fixed_fps = p_fps;

	_particles_free_data(particles);

	particles->prev_ticks = 0;
	particles->phase = 0;
	particles->prev_phase = 0;
	particles->clear = true;

	particles->dependency.changed_notify(DEPENDENCY_CHANGED_PARTICLES);
}

void RendererStorageRD::particles_set_interpolate(RID p_particles, bool p_enable) {
	Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND(!particles);

	particles->interpolate = p_enable;
}

void RendererStorageRD::particles_set_fractional_delta(RID p_particles, bool p_enable) {
	Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND(!particles);

	particles->fractional_delta = p_enable;
}

void RendererStorageRD::particles_set_trails(RID p_particles, bool p_enable, double p_length) {
	Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND(!particles);
	ERR_FAIL_COND(p_length < 0.1);
	p_length = MIN(10.0, p_length);

	particles->trails_enabled = p_enable;
	particles->trail_length = p_length;

	_particles_free_data(particles);

	particles->prev_ticks = 0;
	particles->phase = 0;
	particles->prev_phase = 0;
	particles->clear = true;

	particles->dependency.changed_notify(DEPENDENCY_CHANGED_PARTICLES);
}

void RendererStorageRD::particles_set_trail_bind_poses(RID p_particles, const Vector<Transform3D> &p_bind_poses) {
	Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND(!particles);
	if (particles->trail_bind_pose_buffer.is_valid() && particles->trail_bind_poses.size() != p_bind_poses.size()) {
		_particles_free_data(particles);

		particles->prev_ticks = 0;
		particles->phase = 0;
		particles->prev_phase = 0;
		particles->clear = true;
	}
	particles->trail_bind_poses = p_bind_poses;
	particles->trail_bind_poses_dirty = true;

	particles->dependency.changed_notify(DEPENDENCY_CHANGED_PARTICLES);
}

void RendererStorageRD::particles_set_collision_base_size(RID p_particles, real_t p_size) {
	Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND(!particles);

	particles->collision_base_size = p_size;
}

void RendererStorageRD::particles_set_transform_align(RID p_particles, RS::ParticlesTransformAlign p_transform_align) {
	Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND(!particles);

	particles->transform_align = p_transform_align;
}

void RendererStorageRD::particles_set_process_material(RID p_particles, RID p_material) {
	Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND(!particles);

	particles->process_material = p_material;
	particles->dependency.changed_notify(DEPENDENCY_CHANGED_PARTICLES); //the instance buffer may have changed
}

RID RendererStorageRD::particles_get_process_material(RID p_particles) const {
	Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND_V(!particles, RID());

	return particles->process_material;
}

void RendererStorageRD::particles_set_draw_order(RID p_particles, RS::ParticlesDrawOrder p_order) {
	Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND(!particles);

	particles->draw_order = p_order;
}

void RendererStorageRD::particles_set_draw_passes(RID p_particles, int p_passes) {
	Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND(!particles);

	particles->draw_passes.resize(p_passes);
}

void RendererStorageRD::particles_set_draw_pass_mesh(RID p_particles, int p_pass, RID p_mesh) {
	Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND(!particles);
	ERR_FAIL_INDEX(p_pass, particles->draw_passes.size());
	particles->draw_passes.write[p_pass] = p_mesh;
}

void RendererStorageRD::particles_restart(RID p_particles) {
	Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND(!particles);

	particles->restart_request = true;
}

void RendererStorageRD::_particles_allocate_emission_buffer(Particles *particles) {
	ERR_FAIL_COND(particles->emission_buffer != nullptr);

	particles->emission_buffer_data.resize(sizeof(ParticleEmissionBuffer::Data) * particles->amount + sizeof(uint32_t) * 4);
	memset(particles->emission_buffer_data.ptrw(), 0, particles->emission_buffer_data.size());
	particles->emission_buffer = reinterpret_cast<ParticleEmissionBuffer *>(particles->emission_buffer_data.ptrw());
	particles->emission_buffer->particle_max = particles->amount;

	particles->emission_storage_buffer = RD::get_singleton()->storage_buffer_create(particles->emission_buffer_data.size(), particles->emission_buffer_data);

	if (RD::get_singleton()->uniform_set_is_valid(particles->particles_material_uniform_set)) {
		//will need to be re-created
		RD::get_singleton()->free(particles->particles_material_uniform_set);
		particles->particles_material_uniform_set = RID();
	}
}

void RendererStorageRD::particles_set_subemitter(RID p_particles, RID p_subemitter_particles) {
	Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND(!particles);
	ERR_FAIL_COND(p_particles == p_subemitter_particles);

	particles->sub_emitter = p_subemitter_particles;

	if (RD::get_singleton()->uniform_set_is_valid(particles->particles_material_uniform_set)) {
		RD::get_singleton()->free(particles->particles_material_uniform_set);
		particles->particles_material_uniform_set = RID(); //clear and force to re create sub emitting
	}
}

void RendererStorageRD::particles_emit(RID p_particles, const Transform3D &p_transform, const Vector3 &p_velocity, const Color &p_color, const Color &p_custom, uint32_t p_emit_flags) {
	Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND(!particles);
	ERR_FAIL_COND(particles->amount == 0);

	if (particles->emitting) {
		particles->clear = true;
		particles->emitting = false;
	}

	if (particles->emission_buffer == nullptr) {
		_particles_allocate_emission_buffer(particles);
	}

	if (particles->inactive) {
		//in case it was inactive, make active again
		particles->inactive = false;
		particles->inactive_time = 0;
	}

	int32_t idx = particles->emission_buffer->particle_count;
	if (idx < particles->emission_buffer->particle_max) {
		store_transform(p_transform, particles->emission_buffer->data[idx].xform);

		particles->emission_buffer->data[idx].velocity[0] = p_velocity.x;
		particles->emission_buffer->data[idx].velocity[1] = p_velocity.y;
		particles->emission_buffer->data[idx].velocity[2] = p_velocity.z;

		particles->emission_buffer->data[idx].custom[0] = p_custom.r;
		particles->emission_buffer->data[idx].custom[1] = p_custom.g;
		particles->emission_buffer->data[idx].custom[2] = p_custom.b;
		particles->emission_buffer->data[idx].custom[3] = p_custom.a;

		particles->emission_buffer->data[idx].color[0] = p_color.r;
		particles->emission_buffer->data[idx].color[1] = p_color.g;
		particles->emission_buffer->data[idx].color[2] = p_color.b;
		particles->emission_buffer->data[idx].color[3] = p_color.a;

		particles->emission_buffer->data[idx].flags = p_emit_flags;
		particles->emission_buffer->particle_count++;
	}
}

void RendererStorageRD::particles_request_process(RID p_particles) {
	Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND(!particles);

	if (!particles->dirty) {
		particles->dirty = true;
		particles->update_list = particle_update_list;
		particle_update_list = particles;
	}
}

AABB RendererStorageRD::particles_get_current_aabb(RID p_particles) {
	if (RSG::threaded) {
		WARN_PRINT_ONCE("Calling this function with threaded rendering enabled stalls the renderer, use with care.");
	}

	const Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND_V(!particles, AABB());

	int total_amount = particles->amount;
	if (particles->trails_enabled && particles->trail_bind_poses.size() > 1) {
		total_amount *= particles->trail_bind_poses.size();
	}

	Vector<uint8_t> buffer = RD::get_singleton()->buffer_get_data(particles->particle_buffer);
	ERR_FAIL_COND_V(buffer.size() != (int)(total_amount * sizeof(ParticleData)), AABB());

	Transform3D inv = particles->emission_transform.affine_inverse();

	AABB aabb;
	if (buffer.size()) {
		bool first = true;

		const uint8_t *data_ptr = (const uint8_t *)buffer.ptr();
		uint32_t particle_data_size = sizeof(ParticleData) + sizeof(float) * particles->userdata_count;

		for (int i = 0; i < total_amount; i++) {
			const ParticleData &particle_data = *(const ParticleData *)&data_ptr[particle_data_size * i];
			if (particle_data.active) {
				Vector3 pos = Vector3(particle_data.xform[12], particle_data.xform[13], particle_data.xform[14]);
				if (!particles->use_local_coords) {
					pos = inv.xform(pos);
				}
				if (first) {
					aabb.position = pos;
					first = false;
				} else {
					aabb.expand_to(pos);
				}
			}
		}
	}

	float longest_axis_size = 0;
	for (int i = 0; i < particles->draw_passes.size(); i++) {
		if (particles->draw_passes[i].is_valid()) {
			AABB maabb = RendererRD::MeshStorage::get_singleton()->mesh_get_aabb(particles->draw_passes[i], RID());
			longest_axis_size = MAX(maabb.get_longest_axis_size(), longest_axis_size);
		}
	}

	aabb.grow_by(longest_axis_size);

	return aabb;
}

AABB RendererStorageRD::particles_get_aabb(RID p_particles) const {
	const Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND_V(!particles, AABB());

	return particles->custom_aabb;
}

void RendererStorageRD::particles_set_emission_transform(RID p_particles, const Transform3D &p_transform) {
	Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND(!particles);

	particles->emission_transform = p_transform;
}

int RendererStorageRD::particles_get_draw_passes(RID p_particles) const {
	const Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND_V(!particles, 0);

	return particles->draw_passes.size();
}

RID RendererStorageRD::particles_get_draw_pass_mesh(RID p_particles, int p_pass) const {
	const Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND_V(!particles, RID());
	ERR_FAIL_INDEX_V(p_pass, particles->draw_passes.size(), RID());

	return particles->draw_passes[p_pass];
}

void RendererStorageRD::particles_add_collision(RID p_particles, RID p_particles_collision_instance) {
	Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND(!particles);
	particles->collisions.insert(p_particles_collision_instance);
}

void RendererStorageRD::particles_remove_collision(RID p_particles, RID p_particles_collision_instance) {
	Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND(!particles);
	particles->collisions.erase(p_particles_collision_instance);
}

void RendererStorageRD::particles_set_canvas_sdf_collision(RID p_particles, bool p_enable, const Transform2D &p_xform, const Rect2 &p_to_screen, RID p_texture) {
	Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND(!particles);
	particles->has_sdf_collision = p_enable;
	particles->sdf_collision_transform = p_xform;
	particles->sdf_collision_to_screen = p_to_screen;
	particles->sdf_collision_texture = p_texture;
}

void RendererStorageRD::_particles_process(Particles *p_particles, double p_delta) {
	RendererRD::TextureStorage *texture_storage = RendererRD::TextureStorage::get_singleton();
	RendererRD::MaterialStorage *material_storage = RendererRD::MaterialStorage::get_singleton();

	if (p_particles->particles_material_uniform_set.is_null() || !RD::get_singleton()->uniform_set_is_valid(p_particles->particles_material_uniform_set)) {
		Vector<RD::Uniform> uniforms;

		{
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
			u.binding = 0;
			u.append_id(p_particles->frame_params_buffer);
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
			u.binding = 1;
			u.append_id(p_particles->particle_buffer);
			uniforms.push_back(u);
		}

		{
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
			u.binding = 2;
			if (p_particles->emission_storage_buffer.is_valid()) {
				u.append_id(p_particles->emission_storage_buffer);
			} else {
				u.append_id(RendererRD::MeshStorage::get_singleton()->get_default_rd_storage_buffer());
			}
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
			u.binding = 3;
			Particles *sub_emitter = particles_owner.get_or_null(p_particles->sub_emitter);
			if (sub_emitter) {
				if (sub_emitter->emission_buffer == nullptr) { //no emission buffer, allocate emission buffer
					_particles_allocate_emission_buffer(sub_emitter);
				}
				u.append_id(sub_emitter->emission_storage_buffer);
			} else {
				u.append_id(RendererRD::MeshStorage::get_singleton()->get_default_rd_storage_buffer());
			}
			uniforms.push_back(u);
		}

		p_particles->particles_material_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, particles_shader.default_shader_rd, 1);
	}

	double new_phase = Math::fmod((double)p_particles->phase + (p_delta / p_particles->lifetime) * p_particles->speed_scale, 1.0);

	//move back history (if there is any)
	for (uint32_t i = p_particles->frame_history.size() - 1; i > 0; i--) {
		p_particles->frame_history[i] = p_particles->frame_history[i - 1];
	}
	//update current frame
	ParticlesFrameParams &frame_params = p_particles->frame_history[0];

	if (p_particles->clear) {
		p_particles->cycle_number = 0;
		p_particles->random_seed = Math::rand();
	} else if (new_phase < p_particles->phase) {
		if (p_particles->one_shot) {
			p_particles->emitting = false;
		}
		p_particles->cycle_number++;
	}

	frame_params.emitting = p_particles->emitting;
	frame_params.system_phase = new_phase;
	frame_params.prev_system_phase = p_particles->phase;

	p_particles->phase = new_phase;

	frame_params.time = RendererCompositorRD::singleton->get_total_time();
	frame_params.delta = p_delta * p_particles->speed_scale;
	frame_params.random_seed = p_particles->random_seed;
	frame_params.explosiveness = p_particles->explosiveness;
	frame_params.randomness = p_particles->randomness;

	if (p_particles->use_local_coords) {
		store_transform(Transform3D(), frame_params.emission_transform);
	} else {
		store_transform(p_particles->emission_transform, frame_params.emission_transform);
	}

	frame_params.cycle = p_particles->cycle_number;
	frame_params.frame = p_particles->frame_counter++;
	frame_params.pad0 = 0;
	frame_params.pad1 = 0;
	frame_params.pad2 = 0;

	{ //collision and attractors

		frame_params.collider_count = 0;
		frame_params.attractor_count = 0;
		frame_params.particle_size = p_particles->collision_base_size;

		RID collision_3d_textures[ParticlesFrameParams::MAX_3D_TEXTURES];
		RID collision_heightmap_texture;

		Transform3D to_particles;
		if (p_particles->use_local_coords) {
			to_particles = p_particles->emission_transform.affine_inverse();
		}

		if (p_particles->has_sdf_collision && RD::get_singleton()->texture_is_valid(p_particles->sdf_collision_texture)) {
			//2D collision

			Transform2D xform = p_particles->sdf_collision_transform; //will use dotproduct manually so invert beforehand
			Transform2D revert = xform.affine_inverse();
			frame_params.collider_count = 1;
			frame_params.colliders[0].transform[0] = xform.elements[0][0];
			frame_params.colliders[0].transform[1] = xform.elements[0][1];
			frame_params.colliders[0].transform[2] = 0;
			frame_params.colliders[0].transform[3] = xform.elements[2][0];

			frame_params.colliders[0].transform[4] = xform.elements[1][0];
			frame_params.colliders[0].transform[5] = xform.elements[1][1];
			frame_params.colliders[0].transform[6] = 0;
			frame_params.colliders[0].transform[7] = xform.elements[2][1];

			frame_params.colliders[0].transform[8] = revert.elements[0][0];
			frame_params.colliders[0].transform[9] = revert.elements[0][1];
			frame_params.colliders[0].transform[10] = 0;
			frame_params.colliders[0].transform[11] = revert.elements[2][0];

			frame_params.colliders[0].transform[12] = revert.elements[1][0];
			frame_params.colliders[0].transform[13] = revert.elements[1][1];
			frame_params.colliders[0].transform[14] = 0;
			frame_params.colliders[0].transform[15] = revert.elements[2][1];

			frame_params.colliders[0].extents[0] = p_particles->sdf_collision_to_screen.size.x;
			frame_params.colliders[0].extents[1] = p_particles->sdf_collision_to_screen.size.y;
			frame_params.colliders[0].extents[2] = p_particles->sdf_collision_to_screen.position.x;
			frame_params.colliders[0].scale = p_particles->sdf_collision_to_screen.position.y;
			frame_params.colliders[0].texture_index = 0;
			frame_params.colliders[0].type = ParticlesFrameParams::COLLISION_TYPE_2D_SDF;

			collision_heightmap_texture = p_particles->sdf_collision_texture;

			//replace in all other history frames where used because parameters are no longer valid if screen moves
			for (uint32_t i = 1; i < p_particles->frame_history.size(); i++) {
				if (p_particles->frame_history[i].collider_count > 0 && p_particles->frame_history[i].colliders[0].type == ParticlesFrameParams::COLLISION_TYPE_2D_SDF) {
					p_particles->frame_history[i].colliders[0] = frame_params.colliders[0];
				}
			}
		}

		uint32_t collision_3d_textures_used = 0;
		for (const Set<RID>::Element *E = p_particles->collisions.front(); E; E = E->next()) {
			ParticlesCollisionInstance *pci = particles_collision_instance_owner.get_or_null(E->get());
			if (!pci || !pci->active) {
				continue;
			}
			ParticlesCollision *pc = particles_collision_owner.get_or_null(pci->collision);
			ERR_CONTINUE(!pc);

			Transform3D to_collider = pci->transform;
			if (p_particles->use_local_coords) {
				to_collider = to_particles * to_collider;
			}
			Vector3 scale = to_collider.basis.get_scale();
			to_collider.basis.orthonormalize();

			if (pc->type <= RS::PARTICLES_COLLISION_TYPE_VECTOR_FIELD_ATTRACT) {
				//attractor
				if (frame_params.attractor_count >= ParticlesFrameParams::MAX_ATTRACTORS) {
					continue;
				}

				ParticlesFrameParams::Attractor &attr = frame_params.attractors[frame_params.attractor_count];

				store_transform(to_collider, attr.transform);
				attr.strength = pc->attractor_strength;
				attr.attenuation = pc->attractor_attenuation;
				attr.directionality = pc->attractor_directionality;

				switch (pc->type) {
					case RS::PARTICLES_COLLISION_TYPE_SPHERE_ATTRACT: {
						attr.type = ParticlesFrameParams::ATTRACTOR_TYPE_SPHERE;
						float radius = pc->radius;
						radius *= (scale.x + scale.y + scale.z) / 3.0;
						attr.extents[0] = radius;
						attr.extents[1] = radius;
						attr.extents[2] = radius;
					} break;
					case RS::PARTICLES_COLLISION_TYPE_BOX_ATTRACT: {
						attr.type = ParticlesFrameParams::ATTRACTOR_TYPE_BOX;
						Vector3 extents = pc->extents * scale;
						attr.extents[0] = extents.x;
						attr.extents[1] = extents.y;
						attr.extents[2] = extents.z;
					} break;
					case RS::PARTICLES_COLLISION_TYPE_VECTOR_FIELD_ATTRACT: {
						if (collision_3d_textures_used >= ParticlesFrameParams::MAX_3D_TEXTURES) {
							continue;
						}
						attr.type = ParticlesFrameParams::ATTRACTOR_TYPE_VECTOR_FIELD;
						Vector3 extents = pc->extents * scale;
						attr.extents[0] = extents.x;
						attr.extents[1] = extents.y;
						attr.extents[2] = extents.z;
						attr.texture_index = collision_3d_textures_used;

						collision_3d_textures[collision_3d_textures_used] = pc->field_texture;
						collision_3d_textures_used++;
					} break;
					default: {
					}
				}

				frame_params.attractor_count++;
			} else {
				//collider
				if (frame_params.collider_count >= ParticlesFrameParams::MAX_COLLIDERS) {
					continue;
				}

				ParticlesFrameParams::Collider &col = frame_params.colliders[frame_params.collider_count];

				store_transform(to_collider, col.transform);
				switch (pc->type) {
					case RS::PARTICLES_COLLISION_TYPE_SPHERE_COLLIDE: {
						col.type = ParticlesFrameParams::COLLISION_TYPE_SPHERE;
						float radius = pc->radius;
						radius *= (scale.x + scale.y + scale.z) / 3.0;
						col.extents[0] = radius;
						col.extents[1] = radius;
						col.extents[2] = radius;
					} break;
					case RS::PARTICLES_COLLISION_TYPE_BOX_COLLIDE: {
						col.type = ParticlesFrameParams::COLLISION_TYPE_BOX;
						Vector3 extents = pc->extents * scale;
						col.extents[0] = extents.x;
						col.extents[1] = extents.y;
						col.extents[2] = extents.z;
					} break;
					case RS::PARTICLES_COLLISION_TYPE_SDF_COLLIDE: {
						if (collision_3d_textures_used >= ParticlesFrameParams::MAX_3D_TEXTURES) {
							continue;
						}
						col.type = ParticlesFrameParams::COLLISION_TYPE_SDF;
						Vector3 extents = pc->extents * scale;
						col.extents[0] = extents.x;
						col.extents[1] = extents.y;
						col.extents[2] = extents.z;
						col.texture_index = collision_3d_textures_used;
						col.scale = (scale.x + scale.y + scale.z) * 0.333333333333; //non uniform scale non supported

						collision_3d_textures[collision_3d_textures_used] = pc->field_texture;
						collision_3d_textures_used++;
					} break;
					case RS::PARTICLES_COLLISION_TYPE_HEIGHTFIELD_COLLIDE: {
						if (collision_heightmap_texture != RID()) { //already taken
							continue;
						}

						col.type = ParticlesFrameParams::COLLISION_TYPE_HEIGHT_FIELD;
						Vector3 extents = pc->extents * scale;
						col.extents[0] = extents.x;
						col.extents[1] = extents.y;
						col.extents[2] = extents.z;
						collision_heightmap_texture = pc->heightfield_texture;
					} break;
					default: {
					}
				}

				frame_params.collider_count++;
			}
		}

		bool different = false;
		if (collision_3d_textures_used == p_particles->collision_3d_textures_used) {
			for (int i = 0; i < ParticlesFrameParams::MAX_3D_TEXTURES; i++) {
				if (p_particles->collision_3d_textures[i] != collision_3d_textures[i]) {
					different = true;
					break;
				}
			}
		}

		if (collision_heightmap_texture != p_particles->collision_heightmap_texture) {
			different = true;
		}

		bool uniform_set_valid = RD::get_singleton()->uniform_set_is_valid(p_particles->collision_textures_uniform_set);

		if (different || !uniform_set_valid) {
			if (uniform_set_valid) {
				RD::get_singleton()->free(p_particles->collision_textures_uniform_set);
			}

			Vector<RD::Uniform> uniforms;

			{
				RD::Uniform u;
				u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
				u.binding = 0;
				for (uint32_t i = 0; i < ParticlesFrameParams::MAX_3D_TEXTURES; i++) {
					RID rd_tex;
					if (i < collision_3d_textures_used) {
						RendererRD::Texture *t = RendererRD::TextureStorage::get_singleton()->get_texture(collision_3d_textures[i]);
						if (t && t->type == RendererRD::Texture::TYPE_3D) {
							rd_tex = t->rd_texture;
						}
					}

					if (rd_tex == RID()) {
						rd_tex = texture_storage->texture_rd_get_default(RendererRD::DEFAULT_RD_TEXTURE_3D_WHITE);
					}
					u.append_id(rd_tex);
				}
				uniforms.push_back(u);
			}
			{
				RD::Uniform u;
				u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
				u.binding = 1;
				if (collision_heightmap_texture.is_valid()) {
					u.append_id(collision_heightmap_texture);
				} else {
					u.append_id(texture_storage->texture_rd_get_default(RendererRD::DEFAULT_RD_TEXTURE_BLACK));
				}
				uniforms.push_back(u);
			}
			p_particles->collision_textures_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, particles_shader.default_shader_rd, 2);
		}
	}

	ParticlesShader::PushConstant push_constant;

	int process_amount = p_particles->amount;

	if (p_particles->trails_enabled && p_particles->trail_bind_poses.size() > 1) {
		process_amount *= p_particles->trail_bind_poses.size();
	}
	push_constant.clear = p_particles->clear;
	push_constant.total_particles = p_particles->amount;
	push_constant.lifetime = p_particles->lifetime;
	push_constant.trail_size = p_particles->trail_params.size();
	push_constant.use_fractional_delta = p_particles->fractional_delta;
	push_constant.sub_emitter_mode = !p_particles->emitting && p_particles->emission_buffer && (p_particles->emission_buffer->particle_count > 0 || p_particles->force_sub_emit);
	push_constant.trail_pass = false;

	p_particles->force_sub_emit = false; //reset

	Particles *sub_emitter = particles_owner.get_or_null(p_particles->sub_emitter);

	if (sub_emitter && sub_emitter->emission_storage_buffer.is_valid()) {
		//	print_line("updating subemitter buffer");
		int32_t zero[4] = { 0, sub_emitter->amount, 0, 0 };
		RD::get_singleton()->buffer_update(sub_emitter->emission_storage_buffer, 0, sizeof(uint32_t) * 4, zero);
		push_constant.can_emit = true;

		if (sub_emitter->emitting) {
			sub_emitter->emitting = false;
			sub_emitter->clear = true; //will need to clear if it was emitting, sorry
		}
		//make sure the sub emitter processes particles too
		sub_emitter->inactive = false;
		sub_emitter->inactive_time = 0;

		sub_emitter->force_sub_emit = true;

	} else {
		push_constant.can_emit = false;
	}

	if (p_particles->emission_buffer && p_particles->emission_buffer->particle_count) {
		RD::get_singleton()->buffer_update(p_particles->emission_storage_buffer, 0, sizeof(uint32_t) * 4 + sizeof(ParticleEmissionBuffer::Data) * p_particles->emission_buffer->particle_count, p_particles->emission_buffer);
		p_particles->emission_buffer->particle_count = 0;
	}

	p_particles->clear = false;

	if (p_particles->trail_params.size() > 1) {
		//fill the trail params
		for (uint32_t i = 0; i < p_particles->trail_params.size(); i++) {
			uint32_t src_idx = i * p_particles->frame_history.size() / p_particles->trail_params.size();
			p_particles->trail_params[i] = p_particles->frame_history[src_idx];
		}
	} else {
		p_particles->trail_params[0] = p_particles->frame_history[0];
	}

	RD::get_singleton()->buffer_update(p_particles->frame_params_buffer, 0, sizeof(ParticlesFrameParams) * p_particles->trail_params.size(), p_particles->trail_params.ptr());

	ParticlesMaterialData *m = static_cast<ParticlesMaterialData *>(material_storage->material_get_data(p_particles->process_material, RendererRD::SHADER_TYPE_PARTICLES));
	if (!m) {
		m = static_cast<ParticlesMaterialData *>(material_storage->material_get_data(particles_shader.default_material, RendererRD::SHADER_TYPE_PARTICLES));
	}

	ERR_FAIL_COND(!m);

	p_particles->has_collision_cache = m->shader_data->uses_collision;

	//todo should maybe compute all particle systems together?
	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();
	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, m->shader_data->pipeline);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, particles_shader.base_uniform_set, 0);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, p_particles->particles_material_uniform_set, 1);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, p_particles->collision_textures_uniform_set, 2);

	if (m->uniform_set.is_valid() && RD::get_singleton()->uniform_set_is_valid(m->uniform_set)) {
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, m->uniform_set, 3);
	}

	RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(ParticlesShader::PushConstant));

	if (p_particles->trails_enabled && p_particles->trail_bind_poses.size() > 1) {
		//trails requires two passes in order to catch particle starts
		RD::get_singleton()->compute_list_dispatch_threads(compute_list, process_amount / p_particles->trail_bind_poses.size(), 1, 1);

		RD::get_singleton()->compute_list_add_barrier(compute_list);

		push_constant.trail_pass = true;
		RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(ParticlesShader::PushConstant));
		RD::get_singleton()->compute_list_dispatch_threads(compute_list, process_amount - p_particles->amount, 1, 1);
	} else {
		RD::get_singleton()->compute_list_dispatch_threads(compute_list, process_amount, 1, 1);
	}

	RD::get_singleton()->compute_list_end();
}

void RendererStorageRD::particles_set_view_axis(RID p_particles, const Vector3 &p_axis, const Vector3 &p_up_axis) {
	Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND(!particles);

	if (particles->draw_order != RS::PARTICLES_DRAW_ORDER_VIEW_DEPTH && particles->transform_align != RS::PARTICLES_TRANSFORM_ALIGN_Z_BILLBOARD && particles->transform_align != RS::PARTICLES_TRANSFORM_ALIGN_Z_BILLBOARD_Y_TO_VELOCITY) {
		return;
	}

	if (particles->particle_buffer.is_null()) {
		return; //particles have not processed yet
	}

	bool do_sort = particles->draw_order == RS::PARTICLES_DRAW_ORDER_VIEW_DEPTH;

	//copy to sort buffer
	if (do_sort && particles->particles_sort_buffer == RID()) {
		uint32_t size = particles->amount;
		if (size & 1) {
			size++; //make multiple of 16
		}
		size *= sizeof(float) * 2;
		particles->particles_sort_buffer = RD::get_singleton()->storage_buffer_create(size);

		{
			Vector<RD::Uniform> uniforms;

			{
				RD::Uniform u;
				u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
				u.binding = 0;
				u.append_id(particles->particles_sort_buffer);
				uniforms.push_back(u);
			}

			particles->particles_sort_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, particles_shader.copy_shader.version_get_shader(particles_shader.copy_shader_version, ParticlesShader::COPY_MODE_FILL_SORT_BUFFER), 1);
		}
	}

	ParticlesShader::CopyPushConstant copy_push_constant;

	if (particles->trails_enabled && particles->trail_bind_poses.size() > 1) {
		int fixed_fps = 60.0;
		if (particles->fixed_fps > 0) {
			fixed_fps = particles->fixed_fps;
		}

		copy_push_constant.trail_size = particles->trail_bind_poses.size();
		copy_push_constant.trail_total = particles->frame_history.size();
		copy_push_constant.frame_delta = 1.0 / fixed_fps;
	} else {
		copy_push_constant.trail_size = 1;
		copy_push_constant.trail_total = 1;
		copy_push_constant.frame_delta = 0.0;
	}

	copy_push_constant.order_by_lifetime = (particles->draw_order == RS::PARTICLES_DRAW_ORDER_LIFETIME || particles->draw_order == RS::PARTICLES_DRAW_ORDER_REVERSE_LIFETIME);
	copy_push_constant.lifetime_split = MIN(particles->amount * particles->phase, particles->amount - 1);
	copy_push_constant.lifetime_reverse = particles->draw_order == RS::PARTICLES_DRAW_ORDER_REVERSE_LIFETIME;

	copy_push_constant.frame_remainder = particles->interpolate ? particles->frame_remainder : 0.0;
	copy_push_constant.total_particles = particles->amount;
	copy_push_constant.copy_mode_2d = false;

	Vector3 axis = -p_axis; // cameras look to z negative

	if (particles->use_local_coords) {
		axis = particles->emission_transform.basis.xform_inv(axis).normalized();
	}

	copy_push_constant.sort_direction[0] = axis.x;
	copy_push_constant.sort_direction[1] = axis.y;
	copy_push_constant.sort_direction[2] = axis.z;

	copy_push_constant.align_up[0] = p_up_axis.x;
	copy_push_constant.align_up[1] = p_up_axis.y;
	copy_push_constant.align_up[2] = p_up_axis.z;

	copy_push_constant.align_mode = particles->transform_align;

	if (do_sort) {
		RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();

		RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, particles_shader.copy_pipelines[ParticlesShader::COPY_MODE_FILL_SORT_BUFFER + particles->userdata_count * ParticlesShader::COPY_MODE_MAX]);
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, particles->particles_copy_uniform_set, 0);
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, particles->particles_sort_uniform_set, 1);
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, particles->trail_bind_pose_uniform_set, 2);
		RD::get_singleton()->compute_list_set_push_constant(compute_list, &copy_push_constant, sizeof(ParticlesShader::CopyPushConstant));

		RD::get_singleton()->compute_list_dispatch_threads(compute_list, particles->amount, 1, 1);

		RD::get_singleton()->compute_list_end();
		effects->sort_buffer(particles->particles_sort_uniform_set, particles->amount);
	}

	copy_push_constant.total_particles *= copy_push_constant.total_particles;

	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();
	uint32_t copy_pipeline = do_sort ? ParticlesShader::COPY_MODE_FILL_INSTANCES_WITH_SORT_BUFFER : ParticlesShader::COPY_MODE_FILL_INSTANCES;
	copy_pipeline += particles->userdata_count * ParticlesShader::COPY_MODE_MAX;
	copy_push_constant.copy_mode_2d = particles->mode == RS::PARTICLES_MODE_2D ? 1 : 0;
	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, particles_shader.copy_pipelines[copy_pipeline]);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, particles->particles_copy_uniform_set, 0);
	if (do_sort) {
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, particles->particles_sort_uniform_set, 1);
	}
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, particles->trail_bind_pose_uniform_set, 2);

	RD::get_singleton()->compute_list_set_push_constant(compute_list, &copy_push_constant, sizeof(ParticlesShader::CopyPushConstant));

	RD::get_singleton()->compute_list_dispatch_threads(compute_list, copy_push_constant.total_particles, 1, 1);

	RD::get_singleton()->compute_list_end();
}

void RendererStorageRD::_particles_update_buffers(Particles *particles) {
	uint32_t userdata_count = 0;

	const RendererRD::Material *material = RendererRD::MaterialStorage::get_singleton()->get_material(particles->process_material);
	if (material && material->shader && material->shader->data) {
		const ParticlesShaderData *shader_data = static_cast<const ParticlesShaderData *>(material->shader->data);
		userdata_count = shader_data->userdata_count;
	}

	if (userdata_count != particles->userdata_count) {
		// Mismatch userdata, re-create buffers.
		_particles_free_data(particles);
	}

	if (particles->amount > 0 && particles->particle_buffer.is_null()) {
		int total_amount = particles->amount;
		if (particles->trails_enabled && particles->trail_bind_poses.size() > 1) {
			total_amount *= particles->trail_bind_poses.size();
		}

		uint32_t xform_size = particles->mode == RS::PARTICLES_MODE_2D ? 2 : 3;

		particles->particle_buffer = RD::get_singleton()->storage_buffer_create((sizeof(ParticleData) + userdata_count * sizeof(float) * 4) * total_amount);

		particles->userdata_count = userdata_count;

		particles->particle_instance_buffer = RD::get_singleton()->storage_buffer_create(sizeof(float) * 4 * (xform_size + 1 + 1) * total_amount);
		//needs to clear it

		{
			Vector<RD::Uniform> uniforms;

			{
				RD::Uniform u;
				u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
				u.binding = 1;
				u.append_id(particles->particle_buffer);
				uniforms.push_back(u);
			}
			{
				RD::Uniform u;
				u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
				u.binding = 2;
				u.append_id(particles->particle_instance_buffer);
				uniforms.push_back(u);
			}

			particles->particles_copy_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, particles_shader.copy_shader.version_get_shader(particles_shader.copy_shader_version, 0), 0);
		}
	}
}
void RendererStorageRD::update_particles() {
	while (particle_update_list) {
		//use transform feedback to process particles

		Particles *particles = particle_update_list;

		//take and remove
		particle_update_list = particles->update_list;
		particles->update_list = nullptr;
		particles->dirty = false;

		_particles_update_buffers(particles);

		if (particles->restart_request) {
			particles->prev_ticks = 0;
			particles->phase = 0;
			particles->prev_phase = 0;
			particles->clear = true;
			particles->restart_request = false;
		}

		if (particles->inactive && !particles->emitting) {
			//go next
			continue;
		}

		if (particles->emitting) {
			if (particles->inactive) {
				//restart system from scratch
				particles->prev_ticks = 0;
				particles->phase = 0;
				particles->prev_phase = 0;
				particles->clear = true;
			}
			particles->inactive = false;
			particles->inactive_time = 0;
		} else {
			particles->inactive_time += particles->speed_scale * RendererCompositorRD::singleton->get_frame_delta_time();
			if (particles->inactive_time > particles->lifetime * 1.2) {
				particles->inactive = true;
				continue;
			}
		}

#ifndef _MSC_VER
#warning Should use display refresh rate for all this
#endif

		float screen_hz = 60;

		int fixed_fps = 0;
		if (particles->fixed_fps > 0) {
			fixed_fps = particles->fixed_fps;
		} else if (particles->trails_enabled && particles->trail_bind_poses.size() > 1) {
			fixed_fps = screen_hz;
		}
		{
			//update trails
			int history_size = 1;
			int trail_steps = 1;
			if (particles->trails_enabled && particles->trail_bind_poses.size() > 1) {
				history_size = MAX(1, int(particles->trail_length * fixed_fps));
				trail_steps = particles->trail_bind_poses.size();
			}

			if (uint32_t(history_size) != particles->frame_history.size()) {
				particles->frame_history.resize(history_size);
				memset(particles->frame_history.ptr(), 0, sizeof(ParticlesFrameParams) * history_size);
			}

			if (uint32_t(trail_steps) != particles->trail_params.size() || particles->frame_params_buffer.is_null()) {
				particles->trail_params.resize(trail_steps);
				if (particles->frame_params_buffer.is_valid()) {
					RD::get_singleton()->free(particles->frame_params_buffer);
				}
				particles->frame_params_buffer = RD::get_singleton()->storage_buffer_create(sizeof(ParticlesFrameParams) * trail_steps);
			}

			if (particles->trail_bind_poses.size() > 1 && particles->trail_bind_pose_buffer.is_null()) {
				particles->trail_bind_pose_buffer = RD::get_singleton()->storage_buffer_create(sizeof(float) * 16 * particles->trail_bind_poses.size());
				particles->trail_bind_poses_dirty = true;
			}

			if (particles->trail_bind_pose_uniform_set.is_null()) {
				Vector<RD::Uniform> uniforms;
				{
					RD::Uniform u;
					u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
					u.binding = 0;
					if (particles->trail_bind_pose_buffer.is_valid()) {
						u.append_id(particles->trail_bind_pose_buffer);
					} else {
						u.append_id(RendererRD::MeshStorage::get_singleton()->get_default_rd_storage_buffer());
					}
					uniforms.push_back(u);
				}

				particles->trail_bind_pose_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, particles_shader.copy_shader.version_get_shader(particles_shader.copy_shader_version, 0), 2);
			}

			if (particles->trail_bind_pose_buffer.is_valid() && particles->trail_bind_poses_dirty) {
				if (particles_shader.pose_update_buffer.size() < uint32_t(particles->trail_bind_poses.size()) * 16) {
					particles_shader.pose_update_buffer.resize(particles->trail_bind_poses.size() * 16);
				}

				for (int i = 0; i < particles->trail_bind_poses.size(); i++) {
					store_transform(particles->trail_bind_poses[i], &particles_shader.pose_update_buffer[i * 16]);
				}

				RD::get_singleton()->buffer_update(particles->trail_bind_pose_buffer, 0, particles->trail_bind_poses.size() * 16 * sizeof(float), particles_shader.pose_update_buffer.ptr());
			}
		}

		bool zero_time_scale = Engine::get_singleton()->get_time_scale() <= 0.0;

		if (particles->clear && particles->pre_process_time > 0.0) {
			double frame_time;
			if (fixed_fps > 0) {
				frame_time = 1.0 / fixed_fps;
			} else {
				frame_time = 1.0 / 30.0;
			}

			double todo = particles->pre_process_time;

			while (todo >= 0) {
				_particles_process(particles, frame_time);
				todo -= frame_time;
			}
		}

		if (fixed_fps > 0) {
			double frame_time;
			double decr;
			if (zero_time_scale) {
				frame_time = 0.0;
				decr = 1.0 / fixed_fps;
			} else {
				frame_time = 1.0 / fixed_fps;
				decr = frame_time;
			}
			double delta = RendererCompositorRD::singleton->get_frame_delta_time();
			if (delta > 0.1) { //avoid recursive stalls if fps goes below 10
				delta = 0.1;
			} else if (delta <= 0.0) { //unlikely but..
				delta = 0.001;
			}
			double todo = particles->frame_remainder + delta;

			while (todo >= frame_time) {
				_particles_process(particles, frame_time);
				todo -= decr;
			}

			particles->frame_remainder = todo;

		} else {
			if (zero_time_scale) {
				_particles_process(particles, 0.0);
			} else {
				_particles_process(particles, RendererCompositorRD::singleton->get_frame_delta_time());
			}
		}

		//copy particles to instance buffer

		if (particles->draw_order != RS::PARTICLES_DRAW_ORDER_VIEW_DEPTH && particles->transform_align != RS::PARTICLES_TRANSFORM_ALIGN_Z_BILLBOARD && particles->transform_align != RS::PARTICLES_TRANSFORM_ALIGN_Z_BILLBOARD_Y_TO_VELOCITY) {
			//does not need view dependent operation, do copy here
			ParticlesShader::CopyPushConstant copy_push_constant;

			int total_amount = particles->amount;
			if (particles->trails_enabled && particles->trail_bind_poses.size() > 1) {
				total_amount *= particles->trail_bind_poses.size();
			}

			// Affect 2D only.
			if (particles->use_local_coords) {
				// In local mode, particle positions are calculated locally (relative to the node position)
				// and they're also drawn locally.
				// It works as expected, so we just pass an identity transform.
				store_transform(Transform3D(), copy_push_constant.inv_emission_transform);
			} else {
				// In global mode, particle positions are calculated globally (relative to the canvas origin)
				// but they're drawn locally.
				// So, we need to pass the inverse of the emission transform to bring the
				// particles to local coordinates before drawing.
				Transform3D inv = particles->emission_transform.affine_inverse();
				store_transform(inv, copy_push_constant.inv_emission_transform);
			}

			copy_push_constant.total_particles = total_amount;
			copy_push_constant.frame_remainder = particles->interpolate ? particles->frame_remainder : 0.0;
			copy_push_constant.align_mode = particles->transform_align;
			copy_push_constant.align_up[0] = 0;
			copy_push_constant.align_up[1] = 0;
			copy_push_constant.align_up[2] = 0;

			if (particles->trails_enabled && particles->trail_bind_poses.size() > 1) {
				copy_push_constant.trail_size = particles->trail_bind_poses.size();
				copy_push_constant.trail_total = particles->frame_history.size();
				copy_push_constant.frame_delta = 1.0 / fixed_fps;
			} else {
				copy_push_constant.trail_size = 1;
				copy_push_constant.trail_total = 1;
				copy_push_constant.frame_delta = 0.0;
			}

			copy_push_constant.order_by_lifetime = (particles->draw_order == RS::PARTICLES_DRAW_ORDER_LIFETIME || particles->draw_order == RS::PARTICLES_DRAW_ORDER_REVERSE_LIFETIME);
			copy_push_constant.lifetime_split = MIN(particles->amount * particles->phase, particles->amount - 1);
			copy_push_constant.lifetime_reverse = particles->draw_order == RS::PARTICLES_DRAW_ORDER_REVERSE_LIFETIME;

			RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();
			copy_push_constant.copy_mode_2d = particles->mode == RS::PARTICLES_MODE_2D ? 1 : 0;
			RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, particles_shader.copy_pipelines[ParticlesShader::COPY_MODE_FILL_INSTANCES + particles->userdata_count * ParticlesShader::COPY_MODE_MAX]);
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, particles->particles_copy_uniform_set, 0);
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, particles->trail_bind_pose_uniform_set, 2);
			RD::get_singleton()->compute_list_set_push_constant(compute_list, &copy_push_constant, sizeof(ParticlesShader::CopyPushConstant));

			RD::get_singleton()->compute_list_dispatch_threads(compute_list, total_amount, 1, 1);

			RD::get_singleton()->compute_list_end();
		}

		particles->dependency.changed_notify(DEPENDENCY_CHANGED_AABB);
	}
}

bool RendererStorageRD::particles_is_inactive(RID p_particles) const {
	ERR_FAIL_COND_V_MSG(RSG::threaded, false, "This function should never be used with threaded rendering, as it stalls the renderer.");
	const Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND_V(!particles, false);
	return !particles->emitting && particles->inactive;
}

/* SKY SHADER */

void RendererStorageRD::ParticlesShaderData::set_code(const String &p_code) {
	//compile

	code = p_code;
	valid = false;
	ubo_size = 0;
	uniforms.clear();
	uses_collision = false;

	if (code.is_empty()) {
		return; //just invalid, but no error
	}

	ShaderCompiler::GeneratedCode gen_code;
	ShaderCompiler::IdentifierActions actions;
	actions.entry_point_stages["start"] = ShaderCompiler::STAGE_COMPUTE;
	actions.entry_point_stages["process"] = ShaderCompiler::STAGE_COMPUTE;

	/*
	uses_time = false;

	actions.render_mode_flags["use_half_res_pass"] = &uses_half_res;
	actions.render_mode_flags["use_quarter_res_pass"] = &uses_quarter_res;

	actions.usage_flag_pointers["TIME"] = &uses_time;
*/

	actions.usage_flag_pointers["COLLIDED"] = &uses_collision;

	userdata_count = 0;
	for (uint32_t i = 0; i < ParticlesShader::MAX_USERDATAS; i++) {
		userdatas_used[i] = false;
		actions.usage_flag_pointers["USERDATA" + itos(i + 1)] = &userdatas_used[i];
	}

	actions.uniforms = &uniforms;

	Error err = base_singleton->particles_shader.compiler.compile(RS::SHADER_PARTICLES, code, &actions, path, gen_code);
	ERR_FAIL_COND_MSG(err != OK, "Shader compilation failed.");

	if (version.is_null()) {
		version = base_singleton->particles_shader.shader.version_create();
	}

	for (uint32_t i = 0; i < ParticlesShader::MAX_USERDATAS; i++) {
		if (userdatas_used[i]) {
			userdata_count++;
		}
	}

	base_singleton->particles_shader.shader.version_set_compute_code(version, gen_code.code, gen_code.uniforms, gen_code.stage_globals[ShaderCompiler::STAGE_COMPUTE], gen_code.defines);
	ERR_FAIL_COND(!base_singleton->particles_shader.shader.version_is_valid(version));

	ubo_size = gen_code.uniform_total_size;
	ubo_offsets = gen_code.uniform_offsets;
	texture_uniforms = gen_code.texture_uniforms;

	//update pipelines

	pipeline = RD::get_singleton()->compute_pipeline_create(base_singleton->particles_shader.shader.version_get_shader(version, 0));

	valid = true;
}

void RendererStorageRD::ParticlesShaderData::set_default_texture_param(const StringName &p_name, RID p_texture, int p_index) {
	if (!p_texture.is_valid()) {
		if (default_texture_params.has(p_name) && default_texture_params[p_name].has(p_index)) {
			default_texture_params[p_name].erase(p_index);

			if (default_texture_params[p_name].is_empty()) {
				default_texture_params.erase(p_name);
			}
		}
	} else {
		if (!default_texture_params.has(p_name)) {
			default_texture_params[p_name] = Map<int, RID>();
		}
		default_texture_params[p_name][p_index] = p_texture;
	}
}

void RendererStorageRD::ParticlesShaderData::get_param_list(List<PropertyInfo> *p_param_list) const {
	Map<int, StringName> order;

	for (const KeyValue<StringName, ShaderLanguage::ShaderNode::Uniform> &E : uniforms) {
		if (E.value.scope == ShaderLanguage::ShaderNode::Uniform::SCOPE_GLOBAL || E.value.scope == ShaderLanguage::ShaderNode::Uniform::SCOPE_INSTANCE) {
			continue;
		}

		if (E.value.texture_order >= 0) {
			order[E.value.texture_order + 100000] = E.key;
		} else {
			order[E.value.order] = E.key;
		}
	}

	for (const KeyValue<int, StringName> &E : order) {
		PropertyInfo pi = ShaderLanguage::uniform_to_property_info(uniforms[E.value]);
		pi.name = E.value;
		p_param_list->push_back(pi);
	}
}

void RendererStorageRD::ParticlesShaderData::get_instance_param_list(List<RendererMaterialStorage::InstanceShaderParam> *p_param_list) const {
	for (const KeyValue<StringName, ShaderLanguage::ShaderNode::Uniform> &E : uniforms) {
		if (E.value.scope != ShaderLanguage::ShaderNode::Uniform::SCOPE_INSTANCE) {
			continue;
		}

		RendererMaterialStorage::InstanceShaderParam p;
		p.info = ShaderLanguage::uniform_to_property_info(E.value);
		p.info.name = E.key; //supply name
		p.index = E.value.instance_index;
		p.default_value = ShaderLanguage::constant_value_to_variant(E.value.default_value, E.value.type, E.value.array_size, E.value.hint);
		p_param_list->push_back(p);
	}
}

bool RendererStorageRD::ParticlesShaderData::is_param_texture(const StringName &p_param) const {
	if (!uniforms.has(p_param)) {
		return false;
	}

	return uniforms[p_param].texture_order >= 0;
}

bool RendererStorageRD::ParticlesShaderData::is_animated() const {
	return false;
}

bool RendererStorageRD::ParticlesShaderData::casts_shadows() const {
	return false;
}

Variant RendererStorageRD::ParticlesShaderData::get_default_parameter(const StringName &p_parameter) const {
	if (uniforms.has(p_parameter)) {
		ShaderLanguage::ShaderNode::Uniform uniform = uniforms[p_parameter];
		Vector<ShaderLanguage::ConstantNode::Value> default_value = uniform.default_value;
		return ShaderLanguage::constant_value_to_variant(default_value, uniform.type, uniform.array_size, uniform.hint);
	}
	return Variant();
}

RS::ShaderNativeSourceCode RendererStorageRD::ParticlesShaderData::get_native_source_code() const {
	return base_singleton->particles_shader.shader.version_get_native_source_code(version);
}

RendererStorageRD::ParticlesShaderData::ParticlesShaderData() {
	valid = false;
}

RendererStorageRD::ParticlesShaderData::~ParticlesShaderData() {
	//pipeline variants will clear themselves if shader is gone
	if (version.is_valid()) {
		base_singleton->particles_shader.shader.version_free(version);
	}
}

RendererRD::ShaderData *RendererStorageRD::_create_particles_shader_func() {
	ParticlesShaderData *shader_data = memnew(ParticlesShaderData);
	return shader_data;
}

bool RendererStorageRD::ParticlesMaterialData::update_parameters(const Map<StringName, Variant> &p_parameters, bool p_uniform_dirty, bool p_textures_dirty) {
	return update_parameters_uniform_set(p_parameters, p_uniform_dirty, p_textures_dirty, shader_data->uniforms, shader_data->ubo_offsets.ptr(), shader_data->texture_uniforms, shader_data->default_texture_params, shader_data->ubo_size, uniform_set, base_singleton->particles_shader.shader.version_get_shader(shader_data->version, 0), 3);
}

RendererStorageRD::ParticlesMaterialData::~ParticlesMaterialData() {
	free_parameters_uniform_set(uniform_set);
}

RendererRD::MaterialData *RendererStorageRD::_create_particles_material_func(ParticlesShaderData *p_shader) {
	ParticlesMaterialData *material_data = memnew(ParticlesMaterialData);
	material_data->shader_data = p_shader;
	//update will happen later anyway so do nothing.
	return material_data;
}
////////

/* PARTICLES COLLISION API */

RID RendererStorageRD::particles_collision_allocate() {
	return particles_collision_owner.allocate_rid();
}
void RendererStorageRD::particles_collision_initialize(RID p_rid) {
	particles_collision_owner.initialize_rid(p_rid, ParticlesCollision());
}

RID RendererStorageRD::particles_collision_get_heightfield_framebuffer(RID p_particles_collision) const {
	ParticlesCollision *particles_collision = particles_collision_owner.get_or_null(p_particles_collision);
	ERR_FAIL_COND_V(!particles_collision, RID());
	ERR_FAIL_COND_V(particles_collision->type != RS::PARTICLES_COLLISION_TYPE_HEIGHTFIELD_COLLIDE, RID());

	if (particles_collision->heightfield_texture == RID()) {
		//create
		int resolutions[RS::PARTICLES_COLLISION_HEIGHTFIELD_RESOLUTION_MAX] = { 256, 512, 1024, 2048, 4096, 8192 };
		Size2i size;
		if (particles_collision->extents.x > particles_collision->extents.z) {
			size.x = resolutions[particles_collision->heightfield_resolution];
			size.y = int32_t(particles_collision->extents.z / particles_collision->extents.x * size.x);
		} else {
			size.y = resolutions[particles_collision->heightfield_resolution];
			size.x = int32_t(particles_collision->extents.x / particles_collision->extents.z * size.y);
		}

		RD::TextureFormat tf;
		tf.format = RD::DATA_FORMAT_D32_SFLOAT;
		tf.width = size.x;
		tf.height = size.y;
		tf.texture_type = RD::TEXTURE_TYPE_2D;
		tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

		particles_collision->heightfield_texture = RD::get_singleton()->texture_create(tf, RD::TextureView());

		Vector<RID> fb_tex;
		fb_tex.push_back(particles_collision->heightfield_texture);
		particles_collision->heightfield_fb = RD::get_singleton()->framebuffer_create(fb_tex);
		particles_collision->heightfield_fb_size = size;
	}

	return particles_collision->heightfield_fb;
}

void RendererStorageRD::particles_collision_set_collision_type(RID p_particles_collision, RS::ParticlesCollisionType p_type) {
	ParticlesCollision *particles_collision = particles_collision_owner.get_or_null(p_particles_collision);
	ERR_FAIL_COND(!particles_collision);

	if (p_type == particles_collision->type) {
		return;
	}

	if (particles_collision->heightfield_texture.is_valid()) {
		RD::get_singleton()->free(particles_collision->heightfield_texture);
		particles_collision->heightfield_texture = RID();
	}
	particles_collision->type = p_type;
	particles_collision->dependency.changed_notify(DEPENDENCY_CHANGED_AABB);
}

void RendererStorageRD::particles_collision_set_cull_mask(RID p_particles_collision, uint32_t p_cull_mask) {
	ParticlesCollision *particles_collision = particles_collision_owner.get_or_null(p_particles_collision);
	ERR_FAIL_COND(!particles_collision);
	particles_collision->cull_mask = p_cull_mask;
}

void RendererStorageRD::particles_collision_set_sphere_radius(RID p_particles_collision, real_t p_radius) {
	ParticlesCollision *particles_collision = particles_collision_owner.get_or_null(p_particles_collision);
	ERR_FAIL_COND(!particles_collision);

	particles_collision->radius = p_radius;
	particles_collision->dependency.changed_notify(DEPENDENCY_CHANGED_AABB);
}

void RendererStorageRD::particles_collision_set_box_extents(RID p_particles_collision, const Vector3 &p_extents) {
	ParticlesCollision *particles_collision = particles_collision_owner.get_or_null(p_particles_collision);
	ERR_FAIL_COND(!particles_collision);

	particles_collision->extents = p_extents;
	particles_collision->dependency.changed_notify(DEPENDENCY_CHANGED_AABB);
}

void RendererStorageRD::particles_collision_set_attractor_strength(RID p_particles_collision, real_t p_strength) {
	ParticlesCollision *particles_collision = particles_collision_owner.get_or_null(p_particles_collision);
	ERR_FAIL_COND(!particles_collision);

	particles_collision->attractor_strength = p_strength;
}

void RendererStorageRD::particles_collision_set_attractor_directionality(RID p_particles_collision, real_t p_directionality) {
	ParticlesCollision *particles_collision = particles_collision_owner.get_or_null(p_particles_collision);
	ERR_FAIL_COND(!particles_collision);

	particles_collision->attractor_directionality = p_directionality;
}

void RendererStorageRD::particles_collision_set_attractor_attenuation(RID p_particles_collision, real_t p_curve) {
	ParticlesCollision *particles_collision = particles_collision_owner.get_or_null(p_particles_collision);
	ERR_FAIL_COND(!particles_collision);

	particles_collision->attractor_attenuation = p_curve;
}

void RendererStorageRD::particles_collision_set_field_texture(RID p_particles_collision, RID p_texture) {
	ParticlesCollision *particles_collision = particles_collision_owner.get_or_null(p_particles_collision);
	ERR_FAIL_COND(!particles_collision);

	particles_collision->field_texture = p_texture;
}

void RendererStorageRD::particles_collision_height_field_update(RID p_particles_collision) {
	ParticlesCollision *particles_collision = particles_collision_owner.get_or_null(p_particles_collision);
	ERR_FAIL_COND(!particles_collision);
	particles_collision->dependency.changed_notify(DEPENDENCY_CHANGED_AABB);
}

void RendererStorageRD::particles_collision_set_height_field_resolution(RID p_particles_collision, RS::ParticlesCollisionHeightfieldResolution p_resolution) {
	ParticlesCollision *particles_collision = particles_collision_owner.get_or_null(p_particles_collision);
	ERR_FAIL_COND(!particles_collision);
	ERR_FAIL_INDEX(p_resolution, RS::PARTICLES_COLLISION_HEIGHTFIELD_RESOLUTION_MAX);

	if (particles_collision->heightfield_resolution == p_resolution) {
		return;
	}

	particles_collision->heightfield_resolution = p_resolution;

	if (particles_collision->heightfield_texture.is_valid()) {
		RD::get_singleton()->free(particles_collision->heightfield_texture);
		particles_collision->heightfield_texture = RID();
	}
}

AABB RendererStorageRD::particles_collision_get_aabb(RID p_particles_collision) const {
	ParticlesCollision *particles_collision = particles_collision_owner.get_or_null(p_particles_collision);
	ERR_FAIL_COND_V(!particles_collision, AABB());

	switch (particles_collision->type) {
		case RS::PARTICLES_COLLISION_TYPE_SPHERE_ATTRACT:
		case RS::PARTICLES_COLLISION_TYPE_SPHERE_COLLIDE: {
			AABB aabb;
			aabb.position = -Vector3(1, 1, 1) * particles_collision->radius;
			aabb.size = Vector3(2, 2, 2) * particles_collision->radius;
			return aabb;
		}
		default: {
			AABB aabb;
			aabb.position = -particles_collision->extents;
			aabb.size = particles_collision->extents * 2;
			return aabb;
		}
	}

	return AABB();
}

Vector3 RendererStorageRD::particles_collision_get_extents(RID p_particles_collision) const {
	const ParticlesCollision *particles_collision = particles_collision_owner.get_or_null(p_particles_collision);
	ERR_FAIL_COND_V(!particles_collision, Vector3());
	return particles_collision->extents;
}

bool RendererStorageRD::particles_collision_is_heightfield(RID p_particles_collision) const {
	const ParticlesCollision *particles_collision = particles_collision_owner.get_or_null(p_particles_collision);
	ERR_FAIL_COND_V(!particles_collision, false);
	return particles_collision->type == RS::PARTICLES_COLLISION_TYPE_HEIGHTFIELD_COLLIDE;
}

RID RendererStorageRD::particles_collision_instance_create(RID p_collision) {
	ParticlesCollisionInstance pci;
	pci.collision = p_collision;
	return particles_collision_instance_owner.make_rid(pci);
}
void RendererStorageRD::particles_collision_instance_set_transform(RID p_collision_instance, const Transform3D &p_transform) {
	ParticlesCollisionInstance *pci = particles_collision_instance_owner.get_or_null(p_collision_instance);
	ERR_FAIL_COND(!pci);
	pci->transform = p_transform;
}
void RendererStorageRD::particles_collision_instance_set_active(RID p_collision_instance, bool p_active) {
	ParticlesCollisionInstance *pci = particles_collision_instance_owner.get_or_null(p_collision_instance);
	ERR_FAIL_COND(!pci);
	pci->active = p_active;
}

/* FOG VOLUMES */

RID RendererStorageRD::fog_volume_allocate() {
	return fog_volume_owner.allocate_rid();
}
void RendererStorageRD::fog_volume_initialize(RID p_rid) {
	fog_volume_owner.initialize_rid(p_rid, FogVolume());
}

void RendererStorageRD::fog_volume_set_shape(RID p_fog_volume, RS::FogVolumeShape p_shape) {
	FogVolume *fog_volume = fog_volume_owner.get_or_null(p_fog_volume);
	ERR_FAIL_COND(!fog_volume);

	if (p_shape == fog_volume->shape) {
		return;
	}

	fog_volume->shape = p_shape;
	fog_volume->dependency.changed_notify(DEPENDENCY_CHANGED_AABB);
}

void RendererStorageRD::fog_volume_set_extents(RID p_fog_volume, const Vector3 &p_extents) {
	FogVolume *fog_volume = fog_volume_owner.get_or_null(p_fog_volume);
	ERR_FAIL_COND(!fog_volume);

	fog_volume->extents = p_extents;
	fog_volume->dependency.changed_notify(DEPENDENCY_CHANGED_AABB);
}

void RendererStorageRD::fog_volume_set_material(RID p_fog_volume, RID p_material) {
	FogVolume *fog_volume = fog_volume_owner.get_or_null(p_fog_volume);
	ERR_FAIL_COND(!fog_volume);
	fog_volume->material = p_material;
}

RID RendererStorageRD::fog_volume_get_material(RID p_fog_volume) const {
	FogVolume *fog_volume = fog_volume_owner.get_or_null(p_fog_volume);
	ERR_FAIL_COND_V(!fog_volume, RID());

	return fog_volume->material;
}

RS::FogVolumeShape RendererStorageRD::fog_volume_get_shape(RID p_fog_volume) const {
	FogVolume *fog_volume = fog_volume_owner.get_or_null(p_fog_volume);
	ERR_FAIL_COND_V(!fog_volume, RS::FOG_VOLUME_SHAPE_BOX);

	return fog_volume->shape;
}

AABB RendererStorageRD::fog_volume_get_aabb(RID p_fog_volume) const {
	FogVolume *fog_volume = fog_volume_owner.get_or_null(p_fog_volume);
	ERR_FAIL_COND_V(!fog_volume, AABB());

	switch (fog_volume->shape) {
		case RS::FOG_VOLUME_SHAPE_ELLIPSOID:
		case RS::FOG_VOLUME_SHAPE_BOX: {
			AABB aabb;
			aabb.position = -fog_volume->extents;
			aabb.size = fog_volume->extents * 2;
			return aabb;
		}
		default: {
			// Need some size otherwise will get culled
			return AABB(Vector3(-1, -1, -1), Vector3(2, 2, 2));
		}
	}

	return AABB();
}

Vector3 RendererStorageRD::fog_volume_get_extents(RID p_fog_volume) const {
	const FogVolume *fog_volume = fog_volume_owner.get_or_null(p_fog_volume);
	ERR_FAIL_COND_V(!fog_volume, Vector3());
	return fog_volume->extents;
}

/* VISIBILITY NOTIFIER */

RID RendererStorageRD::visibility_notifier_allocate() {
	return visibility_notifier_owner.allocate_rid();
}
void RendererStorageRD::visibility_notifier_initialize(RID p_notifier) {
	visibility_notifier_owner.initialize_rid(p_notifier, VisibilityNotifier());
}
void RendererStorageRD::visibility_notifier_set_aabb(RID p_notifier, const AABB &p_aabb) {
	VisibilityNotifier *vn = visibility_notifier_owner.get_or_null(p_notifier);
	ERR_FAIL_COND(!vn);
	vn->aabb = p_aabb;
	vn->dependency.changed_notify(DEPENDENCY_CHANGED_AABB);
}
void RendererStorageRD::visibility_notifier_set_callbacks(RID p_notifier, const Callable &p_enter_callbable, const Callable &p_exit_callable) {
	VisibilityNotifier *vn = visibility_notifier_owner.get_or_null(p_notifier);
	ERR_FAIL_COND(!vn);
	vn->enter_callback = p_enter_callbable;
	vn->exit_callback = p_exit_callable;
}

AABB RendererStorageRD::visibility_notifier_get_aabb(RID p_notifier) const {
	const VisibilityNotifier *vn = visibility_notifier_owner.get_or_null(p_notifier);
	ERR_FAIL_COND_V(!vn, AABB());
	return vn->aabb;
}
void RendererStorageRD::visibility_notifier_call(RID p_notifier, bool p_enter, bool p_deferred) {
	VisibilityNotifier *vn = visibility_notifier_owner.get_or_null(p_notifier);
	ERR_FAIL_COND(!vn);

	if (p_enter) {
		if (!vn->enter_callback.is_null()) {
			if (p_deferred) {
				vn->enter_callback.call_deferred(nullptr, 0);
			} else {
				Variant r;
				Callable::CallError ce;
				vn->enter_callback.call(nullptr, 0, r, ce);
			}
		}
	} else {
		if (!vn->exit_callback.is_null()) {
			if (p_deferred) {
				vn->exit_callback.call_deferred(nullptr, 0);
			} else {
				Variant r;
				Callable::CallError ce;
				vn->exit_callback.call(nullptr, 0, r, ce);
			}
		}
	}
}

/* LIGHT */

void RendererStorageRD::_light_initialize(RID p_light, RS::LightType p_type) {
	Light light;
	light.type = p_type;

	light.param[RS::LIGHT_PARAM_ENERGY] = 1.0;
	light.param[RS::LIGHT_PARAM_INDIRECT_ENERGY] = 1.0;
	light.param[RS::LIGHT_PARAM_SPECULAR] = 0.5;
	light.param[RS::LIGHT_PARAM_RANGE] = 1.0;
	light.param[RS::LIGHT_PARAM_SIZE] = 0.0;
	light.param[RS::LIGHT_PARAM_ATTENUATION] = 1.0;
	light.param[RS::LIGHT_PARAM_SPOT_ANGLE] = 45;
	light.param[RS::LIGHT_PARAM_SPOT_ATTENUATION] = 1.0;
	light.param[RS::LIGHT_PARAM_SHADOW_MAX_DISTANCE] = 0;
	light.param[RS::LIGHT_PARAM_SHADOW_SPLIT_1_OFFSET] = 0.1;
	light.param[RS::LIGHT_PARAM_SHADOW_SPLIT_2_OFFSET] = 0.3;
	light.param[RS::LIGHT_PARAM_SHADOW_SPLIT_3_OFFSET] = 0.6;
	light.param[RS::LIGHT_PARAM_SHADOW_FADE_START] = 0.8;
	light.param[RS::LIGHT_PARAM_SHADOW_NORMAL_BIAS] = 1.0;
	light.param[RS::LIGHT_PARAM_SHADOW_BIAS] = 0.02;
	light.param[RS::LIGHT_PARAM_SHADOW_BLUR] = 0;
	light.param[RS::LIGHT_PARAM_SHADOW_PANCAKE_SIZE] = 20.0;
	light.param[RS::LIGHT_PARAM_SHADOW_VOLUMETRIC_FOG_FADE] = 0.1;
	light.param[RS::LIGHT_PARAM_TRANSMITTANCE_BIAS] = 0.05;

	light_owner.initialize_rid(p_light, light);
}

RID RendererStorageRD::directional_light_allocate() {
	return light_owner.allocate_rid();
}
void RendererStorageRD::directional_light_initialize(RID p_light) {
	_light_initialize(p_light, RS::LIGHT_DIRECTIONAL);
}

RID RendererStorageRD::omni_light_allocate() {
	return light_owner.allocate_rid();
}
void RendererStorageRD::omni_light_initialize(RID p_light) {
	_light_initialize(p_light, RS::LIGHT_OMNI);
}

RID RendererStorageRD::spot_light_allocate() {
	return light_owner.allocate_rid();
}
void RendererStorageRD::spot_light_initialize(RID p_light) {
	_light_initialize(p_light, RS::LIGHT_SPOT);
}

void RendererStorageRD::light_set_color(RID p_light, const Color &p_color) {
	Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_COND(!light);

	light->color = p_color;
}

void RendererStorageRD::light_set_param(RID p_light, RS::LightParam p_param, float p_value) {
	Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_COND(!light);
	ERR_FAIL_INDEX(p_param, RS::LIGHT_PARAM_MAX);

	if (light->param[p_param] == p_value) {
		return;
	}

	switch (p_param) {
		case RS::LIGHT_PARAM_RANGE:
		case RS::LIGHT_PARAM_SPOT_ANGLE:
		case RS::LIGHT_PARAM_SHADOW_MAX_DISTANCE:
		case RS::LIGHT_PARAM_SHADOW_SPLIT_1_OFFSET:
		case RS::LIGHT_PARAM_SHADOW_SPLIT_2_OFFSET:
		case RS::LIGHT_PARAM_SHADOW_SPLIT_3_OFFSET:
		case RS::LIGHT_PARAM_SHADOW_NORMAL_BIAS:
		case RS::LIGHT_PARAM_SHADOW_PANCAKE_SIZE:
		case RS::LIGHT_PARAM_SHADOW_BIAS: {
			light->version++;
			light->dependency.changed_notify(DEPENDENCY_CHANGED_LIGHT);
		} break;
		case RS::LIGHT_PARAM_SIZE: {
			if ((light->param[p_param] > CMP_EPSILON) != (p_value > CMP_EPSILON)) {
				//changing from no size to size and the opposite
				light->dependency.changed_notify(DEPENDENCY_CHANGED_LIGHT_SOFT_SHADOW_AND_PROJECTOR);
			}
		} break;
		default: {
		}
	}

	light->param[p_param] = p_value;
}

void RendererStorageRD::light_set_shadow(RID p_light, bool p_enabled) {
	Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_COND(!light);
	light->shadow = p_enabled;

	light->version++;
	light->dependency.changed_notify(DEPENDENCY_CHANGED_LIGHT);
}

void RendererStorageRD::light_set_projector(RID p_light, RID p_texture) {
	RendererRD::DecalAtlasStorage *decal_atlas_storage = RendererRD::DecalAtlasStorage::get_singleton();
	Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_COND(!light);

	if (light->projector == p_texture) {
		return;
	}

	if (light->type != RS::LIGHT_DIRECTIONAL && light->projector.is_valid()) {
		decal_atlas_storage->texture_remove_from_decal_atlas(light->projector, light->type == RS::LIGHT_OMNI);
	}

	light->projector = p_texture;

	if (light->type != RS::LIGHT_DIRECTIONAL) {
		if (light->projector.is_valid()) {
			decal_atlas_storage->texture_add_to_decal_atlas(light->projector, light->type == RS::LIGHT_OMNI);
		}
		light->dependency.changed_notify(DEPENDENCY_CHANGED_LIGHT_SOFT_SHADOW_AND_PROJECTOR);
	}
}

void RendererStorageRD::light_set_negative(RID p_light, bool p_enable) {
	Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_COND(!light);

	light->negative = p_enable;
}

void RendererStorageRD::light_set_cull_mask(RID p_light, uint32_t p_mask) {
	Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_COND(!light);

	light->cull_mask = p_mask;

	light->version++;
	light->dependency.changed_notify(DEPENDENCY_CHANGED_LIGHT);
}

void RendererStorageRD::light_set_distance_fade(RID p_light, bool p_enabled, float p_begin, float p_shadow, float p_length) {
	Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_COND(!light);

	light->distance_fade = p_enabled;
	light->distance_fade_begin = p_begin;
	light->distance_fade_shadow = p_shadow;
	light->distance_fade_length = p_length;
}

void RendererStorageRD::light_set_reverse_cull_face_mode(RID p_light, bool p_enabled) {
	Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_COND(!light);

	light->reverse_cull = p_enabled;

	light->version++;
	light->dependency.changed_notify(DEPENDENCY_CHANGED_LIGHT);
}

void RendererStorageRD::light_set_bake_mode(RID p_light, RS::LightBakeMode p_bake_mode) {
	Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_COND(!light);

	light->bake_mode = p_bake_mode;

	light->version++;
	light->dependency.changed_notify(DEPENDENCY_CHANGED_LIGHT);
}

void RendererStorageRD::light_set_max_sdfgi_cascade(RID p_light, uint32_t p_cascade) {
	Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_COND(!light);

	light->max_sdfgi_cascade = p_cascade;

	light->version++;
	light->dependency.changed_notify(DEPENDENCY_CHANGED_LIGHT);
}

void RendererStorageRD::light_omni_set_shadow_mode(RID p_light, RS::LightOmniShadowMode p_mode) {
	Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_COND(!light);

	light->omni_shadow_mode = p_mode;

	light->version++;
	light->dependency.changed_notify(DEPENDENCY_CHANGED_LIGHT);
}

RS::LightOmniShadowMode RendererStorageRD::light_omni_get_shadow_mode(RID p_light) {
	const Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_COND_V(!light, RS::LIGHT_OMNI_SHADOW_CUBE);

	return light->omni_shadow_mode;
}

void RendererStorageRD::light_directional_set_shadow_mode(RID p_light, RS::LightDirectionalShadowMode p_mode) {
	Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_COND(!light);

	light->directional_shadow_mode = p_mode;
	light->version++;
	light->dependency.changed_notify(DEPENDENCY_CHANGED_LIGHT);
}

void RendererStorageRD::light_directional_set_blend_splits(RID p_light, bool p_enable) {
	Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_COND(!light);

	light->directional_blend_splits = p_enable;
	light->version++;
	light->dependency.changed_notify(DEPENDENCY_CHANGED_LIGHT);
}

bool RendererStorageRD::light_directional_get_blend_splits(RID p_light) const {
	const Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_COND_V(!light, false);

	return light->directional_blend_splits;
}

void RendererStorageRD::light_directional_set_sky_mode(RID p_light, RS::LightDirectionalSkyMode p_mode) {
	Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_COND(!light);

	light->directional_sky_mode = p_mode;
}

RS::LightDirectionalSkyMode RendererStorageRD::light_directional_get_sky_mode(RID p_light) const {
	const Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_COND_V(!light, RS::LIGHT_DIRECTIONAL_SKY_MODE_LIGHT_AND_SKY);

	return light->directional_sky_mode;
}

RS::LightDirectionalShadowMode RendererStorageRD::light_directional_get_shadow_mode(RID p_light) {
	const Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_COND_V(!light, RS::LIGHT_DIRECTIONAL_SHADOW_ORTHOGONAL);

	return light->directional_shadow_mode;
}

uint32_t RendererStorageRD::light_get_max_sdfgi_cascade(RID p_light) {
	const Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_COND_V(!light, 0);

	return light->max_sdfgi_cascade;
}

RS::LightBakeMode RendererStorageRD::light_get_bake_mode(RID p_light) {
	const Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_COND_V(!light, RS::LIGHT_BAKE_DISABLED);

	return light->bake_mode;
}

uint64_t RendererStorageRD::light_get_version(RID p_light) const {
	const Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_COND_V(!light, 0);

	return light->version;
}

AABB RendererStorageRD::light_get_aabb(RID p_light) const {
	const Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_COND_V(!light, AABB());

	switch (light->type) {
		case RS::LIGHT_SPOT: {
			float len = light->param[RS::LIGHT_PARAM_RANGE];
			float size = Math::tan(Math::deg2rad(light->param[RS::LIGHT_PARAM_SPOT_ANGLE])) * len;
			return AABB(Vector3(-size, -size, -len), Vector3(size * 2, size * 2, len));
		};
		case RS::LIGHT_OMNI: {
			float r = light->param[RS::LIGHT_PARAM_RANGE];
			return AABB(-Vector3(r, r, r), Vector3(r, r, r) * 2);
		};
		case RS::LIGHT_DIRECTIONAL: {
			return AABB();
		};
	}

	ERR_FAIL_V(AABB());
}

/* REFLECTION PROBE */

RID RendererStorageRD::reflection_probe_allocate() {
	return reflection_probe_owner.allocate_rid();
}
void RendererStorageRD::reflection_probe_initialize(RID p_reflection_probe) {
	reflection_probe_owner.initialize_rid(p_reflection_probe, ReflectionProbe());
}

void RendererStorageRD::reflection_probe_set_update_mode(RID p_probe, RS::ReflectionProbeUpdateMode p_mode) {
	ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND(!reflection_probe);

	reflection_probe->update_mode = p_mode;
	reflection_probe->dependency.changed_notify(DEPENDENCY_CHANGED_REFLECTION_PROBE);
}

void RendererStorageRD::reflection_probe_set_intensity(RID p_probe, float p_intensity) {
	ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND(!reflection_probe);

	reflection_probe->intensity = p_intensity;
}

void RendererStorageRD::reflection_probe_set_ambient_mode(RID p_probe, RS::ReflectionProbeAmbientMode p_mode) {
	ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND(!reflection_probe);

	reflection_probe->ambient_mode = p_mode;
}

void RendererStorageRD::reflection_probe_set_ambient_color(RID p_probe, const Color &p_color) {
	ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND(!reflection_probe);

	reflection_probe->ambient_color = p_color;
}

void RendererStorageRD::reflection_probe_set_ambient_energy(RID p_probe, float p_energy) {
	ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND(!reflection_probe);

	reflection_probe->ambient_color_energy = p_energy;
}

void RendererStorageRD::reflection_probe_set_max_distance(RID p_probe, float p_distance) {
	ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND(!reflection_probe);

	reflection_probe->max_distance = p_distance;

	reflection_probe->dependency.changed_notify(DEPENDENCY_CHANGED_REFLECTION_PROBE);
}

void RendererStorageRD::reflection_probe_set_extents(RID p_probe, const Vector3 &p_extents) {
	ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND(!reflection_probe);

	if (reflection_probe->extents == p_extents) {
		return;
	}
	reflection_probe->extents = p_extents;
	reflection_probe->dependency.changed_notify(DEPENDENCY_CHANGED_REFLECTION_PROBE);
}

void RendererStorageRD::reflection_probe_set_origin_offset(RID p_probe, const Vector3 &p_offset) {
	ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND(!reflection_probe);

	reflection_probe->origin_offset = p_offset;
	reflection_probe->dependency.changed_notify(DEPENDENCY_CHANGED_REFLECTION_PROBE);
}

void RendererStorageRD::reflection_probe_set_as_interior(RID p_probe, bool p_enable) {
	ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND(!reflection_probe);

	reflection_probe->interior = p_enable;
	reflection_probe->dependency.changed_notify(DEPENDENCY_CHANGED_REFLECTION_PROBE);
}

void RendererStorageRD::reflection_probe_set_enable_box_projection(RID p_probe, bool p_enable) {
	ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND(!reflection_probe);

	reflection_probe->box_projection = p_enable;
}

void RendererStorageRD::reflection_probe_set_enable_shadows(RID p_probe, bool p_enable) {
	ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND(!reflection_probe);

	reflection_probe->enable_shadows = p_enable;
	reflection_probe->dependency.changed_notify(DEPENDENCY_CHANGED_REFLECTION_PROBE);
}

void RendererStorageRD::reflection_probe_set_cull_mask(RID p_probe, uint32_t p_layers) {
	ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND(!reflection_probe);

	reflection_probe->cull_mask = p_layers;
	reflection_probe->dependency.changed_notify(DEPENDENCY_CHANGED_REFLECTION_PROBE);
}

void RendererStorageRD::reflection_probe_set_resolution(RID p_probe, int p_resolution) {
	ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND(!reflection_probe);
	ERR_FAIL_COND(p_resolution < 32);

	reflection_probe->resolution = p_resolution;
}

void RendererStorageRD::reflection_probe_set_mesh_lod_threshold(RID p_probe, float p_ratio) {
	ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND(!reflection_probe);

	reflection_probe->mesh_lod_threshold = p_ratio;

	reflection_probe->dependency.changed_notify(DEPENDENCY_CHANGED_REFLECTION_PROBE);
}

AABB RendererStorageRD::reflection_probe_get_aabb(RID p_probe) const {
	const ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND_V(!reflection_probe, AABB());

	AABB aabb;
	aabb.position = -reflection_probe->extents;
	aabb.size = reflection_probe->extents * 2.0;

	return aabb;
}

RS::ReflectionProbeUpdateMode RendererStorageRD::reflection_probe_get_update_mode(RID p_probe) const {
	const ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND_V(!reflection_probe, RS::REFLECTION_PROBE_UPDATE_ALWAYS);

	return reflection_probe->update_mode;
}

uint32_t RendererStorageRD::reflection_probe_get_cull_mask(RID p_probe) const {
	const ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND_V(!reflection_probe, 0);

	return reflection_probe->cull_mask;
}

Vector3 RendererStorageRD::reflection_probe_get_extents(RID p_probe) const {
	const ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND_V(!reflection_probe, Vector3());

	return reflection_probe->extents;
}

Vector3 RendererStorageRD::reflection_probe_get_origin_offset(RID p_probe) const {
	const ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND_V(!reflection_probe, Vector3());

	return reflection_probe->origin_offset;
}

bool RendererStorageRD::reflection_probe_renders_shadows(RID p_probe) const {
	const ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND_V(!reflection_probe, false);

	return reflection_probe->enable_shadows;
}

float RendererStorageRD::reflection_probe_get_origin_max_distance(RID p_probe) const {
	const ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND_V(!reflection_probe, 0);

	return reflection_probe->max_distance;
}

float RendererStorageRD::reflection_probe_get_mesh_lod_threshold(RID p_probe) const {
	const ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND_V(!reflection_probe, 0);

	return reflection_probe->mesh_lod_threshold;
}

int RendererStorageRD::reflection_probe_get_resolution(RID p_probe) const {
	const ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND_V(!reflection_probe, 0);

	return reflection_probe->resolution;
}

float RendererStorageRD::reflection_probe_get_intensity(RID p_probe) const {
	const ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND_V(!reflection_probe, 0);

	return reflection_probe->intensity;
}

bool RendererStorageRD::reflection_probe_is_interior(RID p_probe) const {
	const ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND_V(!reflection_probe, false);

	return reflection_probe->interior;
}

bool RendererStorageRD::reflection_probe_is_box_projection(RID p_probe) const {
	const ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND_V(!reflection_probe, false);

	return reflection_probe->box_projection;
}

RS::ReflectionProbeAmbientMode RendererStorageRD::reflection_probe_get_ambient_mode(RID p_probe) const {
	const ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND_V(!reflection_probe, RS::REFLECTION_PROBE_AMBIENT_DISABLED);
	return reflection_probe->ambient_mode;
}

Color RendererStorageRD::reflection_probe_get_ambient_color(RID p_probe) const {
	const ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND_V(!reflection_probe, Color());

	return reflection_probe->ambient_color;
}
float RendererStorageRD::reflection_probe_get_ambient_color_energy(RID p_probe) const {
	const ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND_V(!reflection_probe, 0);

	return reflection_probe->ambient_color_energy;
}

RID RendererStorageRD::voxel_gi_allocate() {
	return voxel_gi_owner.allocate_rid();
}
void RendererStorageRD::voxel_gi_initialize(RID p_voxel_gi) {
	voxel_gi_owner.initialize_rid(p_voxel_gi, VoxelGI());
}

void RendererStorageRD::voxel_gi_allocate_data(RID p_voxel_gi, const Transform3D &p_to_cell_xform, const AABB &p_aabb, const Vector3i &p_octree_size, const Vector<uint8_t> &p_octree_cells, const Vector<uint8_t> &p_data_cells, const Vector<uint8_t> &p_distance_field, const Vector<int> &p_level_counts) {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND(!voxel_gi);

	if (voxel_gi->octree_buffer.is_valid()) {
		RD::get_singleton()->free(voxel_gi->octree_buffer);
		RD::get_singleton()->free(voxel_gi->data_buffer);
		if (voxel_gi->sdf_texture.is_valid()) {
			RD::get_singleton()->free(voxel_gi->sdf_texture);
		}

		voxel_gi->sdf_texture = RID();
		voxel_gi->octree_buffer = RID();
		voxel_gi->data_buffer = RID();
		voxel_gi->octree_buffer_size = 0;
		voxel_gi->data_buffer_size = 0;
		voxel_gi->cell_count = 0;
	}

	voxel_gi->to_cell_xform = p_to_cell_xform;
	voxel_gi->bounds = p_aabb;
	voxel_gi->octree_size = p_octree_size;
	voxel_gi->level_counts = p_level_counts;

	if (p_octree_cells.size()) {
		ERR_FAIL_COND(p_octree_cells.size() % 32 != 0); //cells size must be a multiple of 32

		uint32_t cell_count = p_octree_cells.size() / 32;

		ERR_FAIL_COND(p_data_cells.size() != (int)cell_count * 16); //see that data size matches

		voxel_gi->cell_count = cell_count;
		voxel_gi->octree_buffer = RD::get_singleton()->storage_buffer_create(p_octree_cells.size(), p_octree_cells);
		voxel_gi->octree_buffer_size = p_octree_cells.size();
		voxel_gi->data_buffer = RD::get_singleton()->storage_buffer_create(p_data_cells.size(), p_data_cells);
		voxel_gi->data_buffer_size = p_data_cells.size();

		if (p_distance_field.size()) {
			RD::TextureFormat tf;
			tf.format = RD::DATA_FORMAT_R8_UNORM;
			tf.width = voxel_gi->octree_size.x;
			tf.height = voxel_gi->octree_size.y;
			tf.depth = voxel_gi->octree_size.z;
			tf.texture_type = RD::TEXTURE_TYPE_3D;
			tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT | RD::TEXTURE_USAGE_CAN_COPY_FROM_BIT;
			Vector<Vector<uint8_t>> s;
			s.push_back(p_distance_field);
			voxel_gi->sdf_texture = RD::get_singleton()->texture_create(tf, RD::TextureView(), s);
		}
#if 0
			{
				RD::TextureFormat tf;
				tf.format = RD::DATA_FORMAT_R8_UNORM;
				tf.width = voxel_gi->octree_size.x;
				tf.height = voxel_gi->octree_size.y;
				tf.depth = voxel_gi->octree_size.z;
				tf.type = RD::TEXTURE_TYPE_3D;
				tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_CAN_COPY_TO_BIT;
				tf.shareable_formats.push_back(RD::DATA_FORMAT_R8_UNORM);
				tf.shareable_formats.push_back(RD::DATA_FORMAT_R8_UINT);
				voxel_gi->sdf_texture = RD::get_singleton()->texture_create(tf, RD::TextureView());
			}
			RID shared_tex;
			{
				RD::TextureView tv;
				tv.format_override = RD::DATA_FORMAT_R8_UINT;
				shared_tex = RD::get_singleton()->texture_create_shared(tv, voxel_gi->sdf_texture);
			}
			//update SDF texture
			Vector<RD::Uniform> uniforms;
			{
				RD::Uniform u;
				u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
				u.binding = 1;
				u.append_id(voxel_gi->octree_buffer);
				uniforms.push_back(u);
			}
			{
				RD::Uniform u;
				u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
				u.binding = 2;
				u.append_id(voxel_gi->data_buffer);
				uniforms.push_back(u);
			}
			{
				RD::Uniform u;
				u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
				u.binding = 3;
				u.append_id(shared_tex);
				uniforms.push_back(u);
			}

			RID uniform_set = RD::get_singleton()->uniform_set_create(uniforms, voxel_gi_sdf_shader_version_shader, 0);

			{
				uint32_t push_constant[4] = { 0, 0, 0, 0 };

				for (int i = 0; i < voxel_gi->level_counts.size() - 1; i++) {
					push_constant[0] += voxel_gi->level_counts[i];
				}
				push_constant[1] = push_constant[0] + voxel_gi->level_counts[voxel_gi->level_counts.size() - 1];

				print_line("offset: " + itos(push_constant[0]));
				print_line("size: " + itos(push_constant[1]));
				//create SDF
				RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();
				RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, voxel_gi_sdf_shader_pipeline);
				RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set, 0);
				RD::get_singleton()->compute_list_set_push_constant(compute_list, push_constant, sizeof(uint32_t) * 4);
				RD::get_singleton()->compute_list_dispatch(compute_list, voxel_gi->octree_size.x / 4, voxel_gi->octree_size.y / 4, voxel_gi->octree_size.z / 4);
				RD::get_singleton()->compute_list_end();
			}

			RD::get_singleton()->free(uniform_set);
			RD::get_singleton()->free(shared_tex);
		}
#endif
	}

	voxel_gi->version++;
	voxel_gi->data_version++;

	voxel_gi->dependency.changed_notify(DEPENDENCY_CHANGED_AABB);
}

AABB RendererStorageRD::voxel_gi_get_bounds(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND_V(!voxel_gi, AABB());

	return voxel_gi->bounds;
}

Vector3i RendererStorageRD::voxel_gi_get_octree_size(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND_V(!voxel_gi, Vector3i());
	return voxel_gi->octree_size;
}

Vector<uint8_t> RendererStorageRD::voxel_gi_get_octree_cells(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND_V(!voxel_gi, Vector<uint8_t>());

	if (voxel_gi->octree_buffer.is_valid()) {
		return RD::get_singleton()->buffer_get_data(voxel_gi->octree_buffer);
	}
	return Vector<uint8_t>();
}

Vector<uint8_t> RendererStorageRD::voxel_gi_get_data_cells(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND_V(!voxel_gi, Vector<uint8_t>());

	if (voxel_gi->data_buffer.is_valid()) {
		return RD::get_singleton()->buffer_get_data(voxel_gi->data_buffer);
	}
	return Vector<uint8_t>();
}

Vector<uint8_t> RendererStorageRD::voxel_gi_get_distance_field(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND_V(!voxel_gi, Vector<uint8_t>());

	if (voxel_gi->data_buffer.is_valid()) {
		return RD::get_singleton()->texture_get_data(voxel_gi->sdf_texture, 0);
	}
	return Vector<uint8_t>();
}

Vector<int> RendererStorageRD::voxel_gi_get_level_counts(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND_V(!voxel_gi, Vector<int>());

	return voxel_gi->level_counts;
}

Transform3D RendererStorageRD::voxel_gi_get_to_cell_xform(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND_V(!voxel_gi, Transform3D());

	return voxel_gi->to_cell_xform;
}

void RendererStorageRD::voxel_gi_set_dynamic_range(RID p_voxel_gi, float p_range) {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND(!voxel_gi);

	voxel_gi->dynamic_range = p_range;
	voxel_gi->version++;
}

float RendererStorageRD::voxel_gi_get_dynamic_range(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND_V(!voxel_gi, 0);

	return voxel_gi->dynamic_range;
}

void RendererStorageRD::voxel_gi_set_propagation(RID p_voxel_gi, float p_range) {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND(!voxel_gi);

	voxel_gi->propagation = p_range;
	voxel_gi->version++;
}

float RendererStorageRD::voxel_gi_get_propagation(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND_V(!voxel_gi, 0);
	return voxel_gi->propagation;
}

void RendererStorageRD::voxel_gi_set_energy(RID p_voxel_gi, float p_energy) {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND(!voxel_gi);

	voxel_gi->energy = p_energy;
}

float RendererStorageRD::voxel_gi_get_energy(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND_V(!voxel_gi, 0);
	return voxel_gi->energy;
}

void RendererStorageRD::voxel_gi_set_bias(RID p_voxel_gi, float p_bias) {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND(!voxel_gi);

	voxel_gi->bias = p_bias;
}

float RendererStorageRD::voxel_gi_get_bias(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND_V(!voxel_gi, 0);
	return voxel_gi->bias;
}

void RendererStorageRD::voxel_gi_set_normal_bias(RID p_voxel_gi, float p_normal_bias) {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND(!voxel_gi);

	voxel_gi->normal_bias = p_normal_bias;
}

float RendererStorageRD::voxel_gi_get_normal_bias(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND_V(!voxel_gi, 0);
	return voxel_gi->normal_bias;
}

void RendererStorageRD::voxel_gi_set_anisotropy_strength(RID p_voxel_gi, float p_strength) {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND(!voxel_gi);

	voxel_gi->anisotropy_strength = p_strength;
}

float RendererStorageRD::voxel_gi_get_anisotropy_strength(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND_V(!voxel_gi, 0);
	return voxel_gi->anisotropy_strength;
}

void RendererStorageRD::voxel_gi_set_interior(RID p_voxel_gi, bool p_enable) {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND(!voxel_gi);

	voxel_gi->interior = p_enable;
}

void RendererStorageRD::voxel_gi_set_use_two_bounces(RID p_voxel_gi, bool p_enable) {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND(!voxel_gi);

	voxel_gi->use_two_bounces = p_enable;
	voxel_gi->version++;
}

bool RendererStorageRD::voxel_gi_is_using_two_bounces(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND_V(!voxel_gi, false);
	return voxel_gi->use_two_bounces;
}

bool RendererStorageRD::voxel_gi_is_interior(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND_V(!voxel_gi, 0);
	return voxel_gi->interior;
}

uint32_t RendererStorageRD::voxel_gi_get_version(RID p_voxel_gi) {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND_V(!voxel_gi, 0);
	return voxel_gi->version;
}

uint32_t RendererStorageRD::voxel_gi_get_data_version(RID p_voxel_gi) {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND_V(!voxel_gi, 0);
	return voxel_gi->data_version;
}

RID RendererStorageRD::voxel_gi_get_octree_buffer(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND_V(!voxel_gi, RID());
	return voxel_gi->octree_buffer;
}

RID RendererStorageRD::voxel_gi_get_data_buffer(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND_V(!voxel_gi, RID());
	return voxel_gi->data_buffer;
}

RID RendererStorageRD::voxel_gi_get_sdf_texture(RID p_voxel_gi) {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND_V(!voxel_gi, RID());

	return voxel_gi->sdf_texture;
}

/* LIGHTMAP API */

RID RendererStorageRD::lightmap_allocate() {
	return lightmap_owner.allocate_rid();
}

void RendererStorageRD::lightmap_initialize(RID p_lightmap) {
	lightmap_owner.initialize_rid(p_lightmap, Lightmap());
}

void RendererStorageRD::lightmap_set_textures(RID p_lightmap, RID p_light, bool p_uses_spherical_haromics) {
	RendererRD::TextureStorage *texture_storage = RendererRD::TextureStorage::get_singleton();

	Lightmap *lm = lightmap_owner.get_or_null(p_lightmap);
	ERR_FAIL_COND(!lm);

	lightmap_array_version++;

	//erase lightmap users
	if (lm->light_texture.is_valid()) {
		RendererRD::Texture *t = RendererRD::TextureStorage::get_singleton()->get_texture(lm->light_texture);
		if (t) {
			t->lightmap_users.erase(p_lightmap);
		}
	}

	RendererRD::Texture *t = RendererRD::TextureStorage::get_singleton()->get_texture(p_light);
	lm->light_texture = p_light;
	lm->uses_spherical_harmonics = p_uses_spherical_haromics;

	RID default_2d_array = texture_storage->texture_rd_get_default(RendererRD::DEFAULT_RD_TEXTURE_2D_ARRAY_WHITE);
	if (!t) {
		if (using_lightmap_array) {
			if (lm->array_index >= 0) {
				lightmap_textures.write[lm->array_index] = default_2d_array;
				lm->array_index = -1;
			}
		}

		return;
	}

	t->lightmap_users.insert(p_lightmap);

	if (using_lightmap_array) {
		if (lm->array_index < 0) {
			//not in array, try to put in array
			for (int i = 0; i < lightmap_textures.size(); i++) {
				if (lightmap_textures[i] == default_2d_array) {
					lm->array_index = i;
					break;
				}
			}
		}
		ERR_FAIL_COND_MSG(lm->array_index < 0, "Maximum amount of lightmaps in use (" + itos(lightmap_textures.size()) + ") has been exceeded, lightmap will nod display properly.");

		lightmap_textures.write[lm->array_index] = t->rd_texture;
	}
}

void RendererStorageRD::lightmap_set_probe_bounds(RID p_lightmap, const AABB &p_bounds) {
	Lightmap *lm = lightmap_owner.get_or_null(p_lightmap);
	ERR_FAIL_COND(!lm);
	lm->bounds = p_bounds;
}

void RendererStorageRD::lightmap_set_probe_interior(RID p_lightmap, bool p_interior) {
	Lightmap *lm = lightmap_owner.get_or_null(p_lightmap);
	ERR_FAIL_COND(!lm);
	lm->interior = p_interior;
}

void RendererStorageRD::lightmap_set_probe_capture_data(RID p_lightmap, const PackedVector3Array &p_points, const PackedColorArray &p_point_sh, const PackedInt32Array &p_tetrahedra, const PackedInt32Array &p_bsp_tree) {
	Lightmap *lm = lightmap_owner.get_or_null(p_lightmap);
	ERR_FAIL_COND(!lm);

	if (p_points.size()) {
		ERR_FAIL_COND(p_points.size() * 9 != p_point_sh.size());
		ERR_FAIL_COND((p_tetrahedra.size() % 4) != 0);
		ERR_FAIL_COND((p_bsp_tree.size() % 6) != 0);
	}

	lm->points = p_points;
	lm->bsp_tree = p_bsp_tree;
	lm->point_sh = p_point_sh;
	lm->tetrahedra = p_tetrahedra;
}

PackedVector3Array RendererStorageRD::lightmap_get_probe_capture_points(RID p_lightmap) const {
	Lightmap *lm = lightmap_owner.get_or_null(p_lightmap);
	ERR_FAIL_COND_V(!lm, PackedVector3Array());

	return lm->points;
}

PackedColorArray RendererStorageRD::lightmap_get_probe_capture_sh(RID p_lightmap) const {
	Lightmap *lm = lightmap_owner.get_or_null(p_lightmap);
	ERR_FAIL_COND_V(!lm, PackedColorArray());
	return lm->point_sh;
}

PackedInt32Array RendererStorageRD::lightmap_get_probe_capture_tetrahedra(RID p_lightmap) const {
	Lightmap *lm = lightmap_owner.get_or_null(p_lightmap);
	ERR_FAIL_COND_V(!lm, PackedInt32Array());
	return lm->tetrahedra;
}

PackedInt32Array RendererStorageRD::lightmap_get_probe_capture_bsp_tree(RID p_lightmap) const {
	Lightmap *lm = lightmap_owner.get_or_null(p_lightmap);
	ERR_FAIL_COND_V(!lm, PackedInt32Array());
	return lm->bsp_tree;
}

void RendererStorageRD::lightmap_set_probe_capture_update_speed(float p_speed) {
	lightmap_probe_capture_update_speed = p_speed;
}

void RendererStorageRD::lightmap_tap_sh_light(RID p_lightmap, const Vector3 &p_point, Color *r_sh) {
	Lightmap *lm = lightmap_owner.get_or_null(p_lightmap);
	ERR_FAIL_COND(!lm);

	for (int i = 0; i < 9; i++) {
		r_sh[i] = Color(0, 0, 0, 0);
	}

	if (!lm->points.size() || !lm->bsp_tree.size() || !lm->tetrahedra.size()) {
		return;
	}

	static_assert(sizeof(Lightmap::BSP) == 24);

	const Lightmap::BSP *bsp = (const Lightmap::BSP *)lm->bsp_tree.ptr();
	int32_t node = 0;
	while (node >= 0) {
		if (Plane(bsp[node].plane[0], bsp[node].plane[1], bsp[node].plane[2], bsp[node].plane[3]).is_point_over(p_point)) {
#ifdef DEBUG_ENABLED
			ERR_FAIL_COND(bsp[node].over >= 0 && bsp[node].over < node);
#endif

			node = bsp[node].over;
		} else {
#ifdef DEBUG_ENABLED
			ERR_FAIL_COND(bsp[node].under >= 0 && bsp[node].under < node);
#endif
			node = bsp[node].under;
		}
	}

	if (node == Lightmap::BSP::EMPTY_LEAF) {
		return; //nothing could be done
	}

	node = ABS(node) - 1;

	uint32_t *tetrahedron = (uint32_t *)&lm->tetrahedra[node * 4];
	Vector3 points[4] = { lm->points[tetrahedron[0]], lm->points[tetrahedron[1]], lm->points[tetrahedron[2]], lm->points[tetrahedron[3]] };
	const Color *sh_colors[4]{ &lm->point_sh[tetrahedron[0] * 9], &lm->point_sh[tetrahedron[1] * 9], &lm->point_sh[tetrahedron[2] * 9], &lm->point_sh[tetrahedron[3] * 9] };
	Color barycentric = Geometry3D::tetrahedron_get_barycentric_coords(points[0], points[1], points[2], points[3], p_point);

	for (int i = 0; i < 4; i++) {
		float c = CLAMP(barycentric[i], 0.0, 1.0);
		for (int j = 0; j < 9; j++) {
			r_sh[j] += sh_colors[i][j] * c;
		}
	}
}

bool RendererStorageRD::lightmap_is_interior(RID p_lightmap) const {
	const Lightmap *lm = lightmap_owner.get_or_null(p_lightmap);
	ERR_FAIL_COND_V(!lm, false);
	return lm->interior;
}

AABB RendererStorageRD::lightmap_get_aabb(RID p_lightmap) const {
	const Lightmap *lm = lightmap_owner.get_or_null(p_lightmap);
	ERR_FAIL_COND_V(!lm, AABB());
	return lm->bounds;
}

/* RENDER TARGET API */

void RendererStorageRD::_clear_render_target(RenderTarget *rt) {
	//free in reverse dependency order
	if (rt->framebuffer.is_valid()) {
		RD::get_singleton()->free(rt->framebuffer);
		rt->framebuffer_uniform_set = RID(); //chain deleted
	}

	if (rt->color.is_valid()) {
		RD::get_singleton()->free(rt->color);
	}

	if (rt->backbuffer.is_valid()) {
		RD::get_singleton()->free(rt->backbuffer);
		rt->backbuffer = RID();
		rt->backbuffer_mipmaps.clear();
		rt->backbuffer_uniform_set = RID(); //chain deleted
	}

	_render_target_clear_sdf(rt);

	rt->framebuffer = RID();
	rt->color = RID();
}

void RendererStorageRD::_update_render_target(RenderTarget *rt) {
	if (rt->texture.is_null()) {
		//create a placeholder until updated
		rt->texture = RendererRD::TextureStorage::get_singleton()->texture_allocate();
		RendererRD::TextureStorage::get_singleton()->texture_2d_placeholder_initialize(rt->texture);
		RendererRD::Texture *tex = RendererRD::TextureStorage::get_singleton()->get_texture(rt->texture);
		tex->is_render_target = true;
	}

	_clear_render_target(rt);

	if (rt->size.width == 0 || rt->size.height == 0) {
		return;
	}
	//until we implement support for HDR monitors (and render target is attached to screen), this is enough.
	rt->color_format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
	rt->color_format_srgb = RD::DATA_FORMAT_R8G8B8A8_SRGB;
	rt->image_format = rt->flags[RENDER_TARGET_TRANSPARENT] ? Image::FORMAT_RGBA8 : Image::FORMAT_RGB8;

	RD::TextureFormat rd_format;
	RD::TextureView rd_view;
	{ //attempt register
		rd_format.format = rt->color_format;
		rd_format.width = rt->size.width;
		rd_format.height = rt->size.height;
		rd_format.depth = 1;
		rd_format.array_layers = rt->view_count; // for stereo we create two (or more) layers, need to see if we can make fallback work like this too if we don't have multiview
		rd_format.mipmaps = 1;
		if (rd_format.array_layers > 1) { // why are we not using rt->texture_type ??
			rd_format.texture_type = RD::TEXTURE_TYPE_2D_ARRAY;
		} else {
			rd_format.texture_type = RD::TEXTURE_TYPE_2D;
		}
		rd_format.samples = RD::TEXTURE_SAMPLES_1;
		rd_format.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT | RD::TEXTURE_USAGE_CAN_COPY_FROM_BIT;
		rd_format.shareable_formats.push_back(rt->color_format);
		rd_format.shareable_formats.push_back(rt->color_format_srgb);
	}

	rt->color = RD::get_singleton()->texture_create(rd_format, rd_view);
	ERR_FAIL_COND(rt->color.is_null());

	Vector<RID> fb_textures;
	fb_textures.push_back(rt->color);
	rt->framebuffer = RD::get_singleton()->framebuffer_create(fb_textures, RenderingDevice::INVALID_ID, rt->view_count);
	if (rt->framebuffer.is_null()) {
		_clear_render_target(rt);
		ERR_FAIL_COND(rt->framebuffer.is_null());
	}

	{ //update texture

		RendererRD::Texture *tex = RendererRD::TextureStorage::get_singleton()->get_texture(rt->texture);

		//free existing textures
		if (RD::get_singleton()->texture_is_valid(tex->rd_texture)) {
			RD::get_singleton()->free(tex->rd_texture);
		}
		if (RD::get_singleton()->texture_is_valid(tex->rd_texture_srgb)) {
			RD::get_singleton()->free(tex->rd_texture_srgb);
		}

		tex->rd_texture = RID();
		tex->rd_texture_srgb = RID();

		//create shared textures to the color buffer,
		//so transparent can be supported
		RD::TextureView view;
		view.format_override = rt->color_format;
		if (!rt->flags[RENDER_TARGET_TRANSPARENT]) {
			view.swizzle_a = RD::TEXTURE_SWIZZLE_ONE;
		}
		tex->rd_texture = RD::get_singleton()->texture_create_shared(view, rt->color);
		if (rt->color_format_srgb != RD::DATA_FORMAT_MAX) {
			view.format_override = rt->color_format_srgb;
			tex->rd_texture_srgb = RD::get_singleton()->texture_create_shared(view, rt->color);
		}
		tex->rd_view = view;
		tex->width = rt->size.width;
		tex->height = rt->size.height;
		tex->width_2d = rt->size.width;
		tex->height_2d = rt->size.height;
		tex->rd_format = rt->color_format;
		tex->rd_format_srgb = rt->color_format_srgb;
		tex->format = rt->image_format;

		Vector<RID> proxies = tex->proxies; //make a copy, since update may change it
		for (int i = 0; i < proxies.size(); i++) {
			RendererRD::TextureStorage::get_singleton()->texture_proxy_update(proxies[i], rt->texture);
		}
	}
}

void RendererStorageRD::_create_render_target_backbuffer(RenderTarget *rt) {
	ERR_FAIL_COND(rt->backbuffer.is_valid());

	uint32_t mipmaps_required = Image::get_image_required_mipmaps(rt->size.width, rt->size.height, Image::FORMAT_RGBA8);
	RD::TextureFormat tf;
	tf.format = rt->color_format;
	tf.width = rt->size.width;
	tf.height = rt->size.height;
	tf.texture_type = RD::TEXTURE_TYPE_2D;
	tf.usage_bits = RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT | RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_COPY_TO_BIT;
	tf.mipmaps = mipmaps_required;

	rt->backbuffer = RD::get_singleton()->texture_create(tf, RD::TextureView());
	RD::get_singleton()->set_resource_name(rt->backbuffer, "Render Target Back Buffer");
	rt->backbuffer_mipmap0 = RD::get_singleton()->texture_create_shared_from_slice(RD::TextureView(), rt->backbuffer, 0, 0);
	RD::get_singleton()->set_resource_name(rt->backbuffer_mipmap0, "Back Buffer slice mipmap 0");

	{
		Vector<RID> fb_tex;
		fb_tex.push_back(rt->backbuffer_mipmap0);
		rt->backbuffer_fb = RD::get_singleton()->framebuffer_create(fb_tex);
	}

	if (rt->framebuffer_uniform_set.is_valid() && RD::get_singleton()->uniform_set_is_valid(rt->framebuffer_uniform_set)) {
		//the new one will require the backbuffer.
		RD::get_singleton()->free(rt->framebuffer_uniform_set);
		rt->framebuffer_uniform_set = RID();
	}
	//create mipmaps
	for (uint32_t i = 1; i < mipmaps_required; i++) {
		RID mipmap = RD::get_singleton()->texture_create_shared_from_slice(RD::TextureView(), rt->backbuffer, 0, i);
		RD::get_singleton()->set_resource_name(mipmap, "Back Buffer slice mip: " + itos(i));

		rt->backbuffer_mipmaps.push_back(mipmap);
	}
}

RID RendererStorageRD::render_target_create() {
	RenderTarget render_target;

	render_target.was_used = false;
	render_target.clear_requested = false;

	for (int i = 0; i < RENDER_TARGET_FLAG_MAX; i++) {
		render_target.flags[i] = false;
	}
	_update_render_target(&render_target);
	return render_target_owner.make_rid(render_target);
}

void RendererStorageRD::render_target_set_position(RID p_render_target, int p_x, int p_y) {
	//unused for this render target
}

void RendererStorageRD::render_target_set_size(RID p_render_target, int p_width, int p_height, uint32_t p_view_count) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND(!rt);
	if (rt->size.x != p_width || rt->size.y != p_height || rt->view_count != p_view_count) {
		rt->size.x = p_width;
		rt->size.y = p_height;
		rt->view_count = p_view_count;
		_update_render_target(rt);
	}
}

RID RendererStorageRD::render_target_get_texture(RID p_render_target) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND_V(!rt, RID());

	return rt->texture;
}

void RendererStorageRD::render_target_set_external_texture(RID p_render_target, unsigned int p_texture_id) {
}

void RendererStorageRD::render_target_set_flag(RID p_render_target, RenderTargetFlags p_flag, bool p_value) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND(!rt);
	rt->flags[p_flag] = p_value;
	_update_render_target(rt);
}

bool RendererStorageRD::render_target_was_used(RID p_render_target) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND_V(!rt, false);
	return rt->was_used;
}

void RendererStorageRD::render_target_set_as_unused(RID p_render_target) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND(!rt);
	rt->was_used = false;
}

Size2 RendererStorageRD::render_target_get_size(RID p_render_target) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND_V(!rt, Size2());

	return rt->size;
}

RID RendererStorageRD::render_target_get_rd_framebuffer(RID p_render_target) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND_V(!rt, RID());

	return rt->framebuffer;
}

RID RendererStorageRD::render_target_get_rd_texture(RID p_render_target) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND_V(!rt, RID());

	return rt->color;
}

RID RendererStorageRD::render_target_get_rd_backbuffer(RID p_render_target) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND_V(!rt, RID());
	return rt->backbuffer;
}

RID RendererStorageRD::render_target_get_rd_backbuffer_framebuffer(RID p_render_target) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND_V(!rt, RID());

	if (!rt->backbuffer.is_valid()) {
		_create_render_target_backbuffer(rt);
	}

	return rt->backbuffer_fb;
}

void RendererStorageRD::render_target_request_clear(RID p_render_target, const Color &p_clear_color) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND(!rt);
	rt->clear_requested = true;
	rt->clear_color = p_clear_color;
}

bool RendererStorageRD::render_target_is_clear_requested(RID p_render_target) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND_V(!rt, false);
	return rt->clear_requested;
}

Color RendererStorageRD::render_target_get_clear_request_color(RID p_render_target) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND_V(!rt, Color());
	return rt->clear_color;
}

void RendererStorageRD::render_target_disable_clear_request(RID p_render_target) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND(!rt);
	rt->clear_requested = false;
}

void RendererStorageRD::render_target_do_clear_request(RID p_render_target) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND(!rt);
	if (!rt->clear_requested) {
		return;
	}
	Vector<Color> clear_colors;
	clear_colors.push_back(rt->clear_color);
	RD::get_singleton()->draw_list_begin(rt->framebuffer, RD::INITIAL_ACTION_CLEAR, RD::FINAL_ACTION_READ, RD::INITIAL_ACTION_KEEP, RD::FINAL_ACTION_DISCARD, clear_colors);
	RD::get_singleton()->draw_list_end();
	rt->clear_requested = false;
}

void RendererStorageRD::render_target_set_sdf_size_and_scale(RID p_render_target, RS::ViewportSDFOversize p_size, RS::ViewportSDFScale p_scale) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND(!rt);
	if (rt->sdf_oversize == p_size && rt->sdf_scale == p_scale) {
		return;
	}

	rt->sdf_oversize = p_size;
	rt->sdf_scale = p_scale;

	_render_target_clear_sdf(rt);
}

Rect2i RendererStorageRD::_render_target_get_sdf_rect(const RenderTarget *rt) const {
	Size2i margin;
	int scale;
	switch (rt->sdf_oversize) {
		case RS::VIEWPORT_SDF_OVERSIZE_100_PERCENT: {
			scale = 100;
		} break;
		case RS::VIEWPORT_SDF_OVERSIZE_120_PERCENT: {
			scale = 120;
		} break;
		case RS::VIEWPORT_SDF_OVERSIZE_150_PERCENT: {
			scale = 150;
		} break;
		case RS::VIEWPORT_SDF_OVERSIZE_200_PERCENT: {
			scale = 200;
		} break;
		default: {
		}
	}

	margin = (rt->size * scale / 100) - rt->size;

	Rect2i r(Vector2i(), rt->size);
	r.position -= margin;
	r.size += margin * 2;

	return r;
}

Rect2i RendererStorageRD::render_target_get_sdf_rect(RID p_render_target) const {
	const RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND_V(!rt, Rect2i());

	return _render_target_get_sdf_rect(rt);
}

void RendererStorageRD::render_target_mark_sdf_enabled(RID p_render_target, bool p_enabled) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND(!rt);

	rt->sdf_enabled = p_enabled;
}

bool RendererStorageRD::render_target_is_sdf_enabled(RID p_render_target) const {
	const RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND_V(!rt, false);

	return rt->sdf_enabled;
}

RID RendererStorageRD::render_target_get_sdf_texture(RID p_render_target) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND_V(!rt, RID());
	if (rt->sdf_buffer_read.is_null()) {
		// no texture, create a dummy one for the 2D uniform set
		RD::TextureFormat tformat;
		tformat.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
		tformat.width = 4;
		tformat.height = 4;
		tformat.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT;
		tformat.texture_type = RD::TEXTURE_TYPE_2D;

		Vector<uint8_t> pv;
		pv.resize(16 * 4);
		memset(pv.ptrw(), 0, 16 * 4);
		Vector<Vector<uint8_t>> vpv;

		rt->sdf_buffer_read = RD::get_singleton()->texture_create(tformat, RD::TextureView(), vpv);
	}

	return rt->sdf_buffer_read;
}

void RendererStorageRD::_render_target_allocate_sdf(RenderTarget *rt) {
	ERR_FAIL_COND(rt->sdf_buffer_write_fb.is_valid());
	if (rt->sdf_buffer_read.is_valid()) {
		RD::get_singleton()->free(rt->sdf_buffer_read);
		rt->sdf_buffer_read = RID();
	}

	Size2i size = _render_target_get_sdf_rect(rt).size;

	RD::TextureFormat tformat;
	tformat.format = RD::DATA_FORMAT_R8_UNORM;
	tformat.width = size.width;
	tformat.height = size.height;
	tformat.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT;
	tformat.texture_type = RD::TEXTURE_TYPE_2D;

	rt->sdf_buffer_write = RD::get_singleton()->texture_create(tformat, RD::TextureView());

	{
		Vector<RID> write_fb;
		write_fb.push_back(rt->sdf_buffer_write);
		rt->sdf_buffer_write_fb = RD::get_singleton()->framebuffer_create(write_fb);
	}

	int scale;
	switch (rt->sdf_scale) {
		case RS::VIEWPORT_SDF_SCALE_100_PERCENT: {
			scale = 100;
		} break;
		case RS::VIEWPORT_SDF_SCALE_50_PERCENT: {
			scale = 50;
		} break;
		case RS::VIEWPORT_SDF_SCALE_25_PERCENT: {
			scale = 25;
		} break;
		default: {
			scale = 100;
		} break;
	}

	rt->process_size = size * scale / 100;
	rt->process_size.x = MAX(rt->process_size.x, 1);
	rt->process_size.y = MAX(rt->process_size.y, 1);

	tformat.format = RD::DATA_FORMAT_R16G16_SINT;
	tformat.width = rt->process_size.width;
	tformat.height = rt->process_size.height;
	tformat.usage_bits = RD::TEXTURE_USAGE_STORAGE_BIT;

	rt->sdf_buffer_process[0] = RD::get_singleton()->texture_create(tformat, RD::TextureView());
	rt->sdf_buffer_process[1] = RD::get_singleton()->texture_create(tformat, RD::TextureView());

	tformat.format = RD::DATA_FORMAT_R16_SNORM;
	tformat.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT;

	rt->sdf_buffer_read = RD::get_singleton()->texture_create(tformat, RD::TextureView());

	{
		Vector<RD::Uniform> uniforms;
		{
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
			u.binding = 1;
			u.append_id(rt->sdf_buffer_write);
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
			u.binding = 2;
			u.append_id(rt->sdf_buffer_read);
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
			u.binding = 3;
			u.append_id(rt->sdf_buffer_process[0]);
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
			u.binding = 4;
			u.append_id(rt->sdf_buffer_process[1]);
			uniforms.push_back(u);
		}

		rt->sdf_buffer_process_uniform_sets[0] = RD::get_singleton()->uniform_set_create(uniforms, rt_sdf.shader.version_get_shader(rt_sdf.shader_version, 0), 0);
		RID aux2 = uniforms.write[2].get_id(0);
		RID aux3 = uniforms.write[3].get_id(0);
		uniforms.write[2].set_id(0, aux3);
		uniforms.write[3].set_id(0, aux2);
		rt->sdf_buffer_process_uniform_sets[1] = RD::get_singleton()->uniform_set_create(uniforms, rt_sdf.shader.version_get_shader(rt_sdf.shader_version, 0), 0);
	}
}

void RendererStorageRD::_render_target_clear_sdf(RenderTarget *rt) {
	if (rt->sdf_buffer_read.is_valid()) {
		RD::get_singleton()->free(rt->sdf_buffer_read);
		rt->sdf_buffer_read = RID();
	}
	if (rt->sdf_buffer_write_fb.is_valid()) {
		RD::get_singleton()->free(rt->sdf_buffer_write);
		RD::get_singleton()->free(rt->sdf_buffer_process[0]);
		RD::get_singleton()->free(rt->sdf_buffer_process[1]);
		rt->sdf_buffer_write = RID();
		rt->sdf_buffer_write_fb = RID();
		rt->sdf_buffer_process[0] = RID();
		rt->sdf_buffer_process[1] = RID();
		rt->sdf_buffer_process_uniform_sets[0] = RID();
		rt->sdf_buffer_process_uniform_sets[1] = RID();
	}
}

RID RendererStorageRD::render_target_get_sdf_framebuffer(RID p_render_target) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND_V(!rt, RID());

	if (rt->sdf_buffer_write_fb.is_null()) {
		_render_target_allocate_sdf(rt);
	}

	return rt->sdf_buffer_write_fb;
}
void RendererStorageRD::render_target_sdf_process(RID p_render_target) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND(!rt);
	ERR_FAIL_COND(rt->sdf_buffer_write_fb.is_null());

	RenderTargetSDF::PushConstant push_constant;

	Rect2i r = _render_target_get_sdf_rect(rt);

	push_constant.size[0] = r.size.width;
	push_constant.size[1] = r.size.height;
	push_constant.stride = 0;
	push_constant.shift = 0;
	push_constant.base_size[0] = r.size.width;
	push_constant.base_size[1] = r.size.height;

	bool shrink = false;

	switch (rt->sdf_scale) {
		case RS::VIEWPORT_SDF_SCALE_50_PERCENT: {
			push_constant.size[0] >>= 1;
			push_constant.size[1] >>= 1;
			push_constant.shift = 1;
			shrink = true;
		} break;
		case RS::VIEWPORT_SDF_SCALE_25_PERCENT: {
			push_constant.size[0] >>= 2;
			push_constant.size[1] >>= 2;
			push_constant.shift = 2;
			shrink = true;
		} break;
		default: {
		};
	}

	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();

	/* Load */

	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, rt_sdf.pipelines[shrink ? RenderTargetSDF::SHADER_LOAD_SHRINK : RenderTargetSDF::SHADER_LOAD]);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, rt->sdf_buffer_process_uniform_sets[1], 0); //fill [0]
	RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(RenderTargetSDF::PushConstant));

	RD::get_singleton()->compute_list_dispatch_threads(compute_list, push_constant.size[0], push_constant.size[1], 1);

	/* Process */

	int stride = nearest_power_of_2_templated(MAX(push_constant.size[0], push_constant.size[1]) / 2);

	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, rt_sdf.pipelines[RenderTargetSDF::SHADER_PROCESS]);

	RD::get_singleton()->compute_list_add_barrier(compute_list);
	bool swap = false;

	//jumpflood
	while (stride > 0) {
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, rt->sdf_buffer_process_uniform_sets[swap ? 1 : 0], 0);
		push_constant.stride = stride;
		RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(RenderTargetSDF::PushConstant));
		RD::get_singleton()->compute_list_dispatch_threads(compute_list, push_constant.size[0], push_constant.size[1], 1);
		stride /= 2;
		swap = !swap;
		RD::get_singleton()->compute_list_add_barrier(compute_list);
	}

	/* Store */

	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, rt_sdf.pipelines[shrink ? RenderTargetSDF::SHADER_STORE_SHRINK : RenderTargetSDF::SHADER_STORE]);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, rt->sdf_buffer_process_uniform_sets[swap ? 1 : 0], 0);
	RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(RenderTargetSDF::PushConstant));
	RD::get_singleton()->compute_list_dispatch_threads(compute_list, push_constant.size[0], push_constant.size[1], 1);

	RD::get_singleton()->compute_list_end();
}

void RendererStorageRD::render_target_copy_to_back_buffer(RID p_render_target, const Rect2i &p_region, bool p_gen_mipmaps) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND(!rt);
	if (!rt->backbuffer.is_valid()) {
		_create_render_target_backbuffer(rt);
	}

	Rect2i region;
	if (p_region == Rect2i()) {
		region.size = rt->size;
	} else {
		region = Rect2i(Size2i(), rt->size).intersection(p_region);
		if (region.size == Size2i()) {
			return; //nothing to do
		}
	}

	//single texture copy for backbuffer
	//RD::get_singleton()->texture_copy(rt->color, rt->backbuffer_mipmap0, Vector3(region.position.x, region.position.y, 0), Vector3(region.position.x, region.position.y, 0), Vector3(region.size.x, region.size.y, 1), 0, 0, 0, 0, true);
	effects->copy_to_rect(rt->color, rt->backbuffer_mipmap0, region, false, false, false, true, true);

	if (!p_gen_mipmaps) {
		return;
	}
	RD::get_singleton()->draw_command_begin_label("Gaussian Blur Mipmaps");
	//then mipmap blur
	RID prev_texture = rt->color; //use color, not backbuffer, as bb has mipmaps.

	for (int i = 0; i < rt->backbuffer_mipmaps.size(); i++) {
		region.position.x >>= 1;
		region.position.y >>= 1;
		region.size.x = MAX(1, region.size.x >> 1);
		region.size.y = MAX(1, region.size.y >> 1);

		RID mipmap = rt->backbuffer_mipmaps[i];
		effects->gaussian_blur(prev_texture, mipmap, region, true);
		prev_texture = mipmap;
	}
	RD::get_singleton()->draw_command_end_label();
}

void RendererStorageRD::render_target_clear_back_buffer(RID p_render_target, const Rect2i &p_region, const Color &p_color) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND(!rt);
	if (!rt->backbuffer.is_valid()) {
		_create_render_target_backbuffer(rt);
	}

	Rect2i region;
	if (p_region == Rect2i()) {
		region.size = rt->size;
	} else {
		region = Rect2i(Size2i(), rt->size).intersection(p_region);
		if (region.size == Size2i()) {
			return; //nothing to do
		}
	}

	//single texture copy for backbuffer
	effects->set_color(rt->backbuffer_mipmap0, p_color, region, true);
}

void RendererStorageRD::render_target_gen_back_buffer_mipmaps(RID p_render_target, const Rect2i &p_region) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND(!rt);
	if (!rt->backbuffer.is_valid()) {
		_create_render_target_backbuffer(rt);
	}

	Rect2i region;
	if (p_region == Rect2i()) {
		region.size = rt->size;
	} else {
		region = Rect2i(Size2i(), rt->size).intersection(p_region);
		if (region.size == Size2i()) {
			return; //nothing to do
		}
	}
	RD::get_singleton()->draw_command_begin_label("Gaussian Blur Mipmaps2");
	//then mipmap blur
	RID prev_texture = rt->backbuffer_mipmap0;

	for (int i = 0; i < rt->backbuffer_mipmaps.size(); i++) {
		region.position.x >>= 1;
		region.position.y >>= 1;
		region.size.x = MAX(1, region.size.x >> 1);
		region.size.y = MAX(1, region.size.y >> 1);

		RID mipmap = rt->backbuffer_mipmaps[i];
		effects->gaussian_blur(prev_texture, mipmap, region, true);
		prev_texture = mipmap;
	}
	RD::get_singleton()->draw_command_end_label();
}

RID RendererStorageRD::render_target_get_framebuffer_uniform_set(RID p_render_target) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND_V(!rt, RID());
	return rt->framebuffer_uniform_set;
}
RID RendererStorageRD::render_target_get_backbuffer_uniform_set(RID p_render_target) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND_V(!rt, RID());
	return rt->backbuffer_uniform_set;
}

void RendererStorageRD::render_target_set_framebuffer_uniform_set(RID p_render_target, RID p_uniform_set) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND(!rt);
	rt->framebuffer_uniform_set = p_uniform_set;
}
void RendererStorageRD::render_target_set_backbuffer_uniform_set(RID p_render_target, RID p_uniform_set) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND(!rt);
	rt->backbuffer_uniform_set = p_uniform_set;
}

void RendererStorageRD::base_update_dependency(RID p_base, DependencyTracker *p_instance) {
	if (RendererRD::MeshStorage::get_singleton()->owns_mesh(p_base)) {
		RendererRD::Mesh *mesh = RendererRD::MeshStorage::get_singleton()->get_mesh(p_base);
		p_instance->update_dependency(&mesh->dependency);
	} else if (RendererRD::MeshStorage::get_singleton()->owns_multimesh(p_base)) {
		RendererRD::MultiMesh *multimesh = RendererRD::MeshStorage::get_singleton()->get_multimesh(p_base);
		p_instance->update_dependency(&multimesh->dependency);
		if (multimesh->mesh.is_valid()) {
			base_update_dependency(multimesh->mesh, p_instance);
		}
	} else if (reflection_probe_owner.owns(p_base)) {
		ReflectionProbe *rp = reflection_probe_owner.get_or_null(p_base);
		p_instance->update_dependency(&rp->dependency);
	} else if (RendererRD::DecalAtlasStorage::get_singleton()->owns_decal(p_base)) {
		RendererRD::Decal *decal = RendererRD::DecalAtlasStorage::get_singleton()->get_decal(p_base);
		p_instance->update_dependency(&decal->dependency);
	} else if (voxel_gi_owner.owns(p_base)) {
		VoxelGI *gip = voxel_gi_owner.get_or_null(p_base);
		p_instance->update_dependency(&gip->dependency);
	} else if (lightmap_owner.owns(p_base)) {
		Lightmap *lm = lightmap_owner.get_or_null(p_base);
		p_instance->update_dependency(&lm->dependency);
	} else if (light_owner.owns(p_base)) {
		Light *l = light_owner.get_or_null(p_base);
		p_instance->update_dependency(&l->dependency);
	} else if (particles_owner.owns(p_base)) {
		Particles *p = particles_owner.get_or_null(p_base);
		p_instance->update_dependency(&p->dependency);
	} else if (particles_collision_owner.owns(p_base)) {
		ParticlesCollision *pc = particles_collision_owner.get_or_null(p_base);
		p_instance->update_dependency(&pc->dependency);
	} else if (fog_volume_owner.owns(p_base)) {
		FogVolume *fv = fog_volume_owner.get_or_null(p_base);
		p_instance->update_dependency(&fv->dependency);
	} else if (visibility_notifier_owner.owns(p_base)) {
		VisibilityNotifier *vn = visibility_notifier_owner.get_or_null(p_base);
		p_instance->update_dependency(&vn->dependency);
	}
}

RS::InstanceType RendererStorageRD::get_base_type(RID p_rid) const {
	if (RendererRD::MeshStorage::get_singleton()->owns_mesh(p_rid)) {
		return RS::INSTANCE_MESH;
	}
	if (RendererRD::MeshStorage::get_singleton()->owns_multimesh(p_rid)) {
		return RS::INSTANCE_MULTIMESH;
	}
	if (reflection_probe_owner.owns(p_rid)) {
		return RS::INSTANCE_REFLECTION_PROBE;
	}
	if (RendererRD::DecalAtlasStorage::get_singleton()->owns_decal(p_rid)) {
		return RS::INSTANCE_DECAL;
	}
	if (voxel_gi_owner.owns(p_rid)) {
		return RS::INSTANCE_VOXEL_GI;
	}
	if (light_owner.owns(p_rid)) {
		return RS::INSTANCE_LIGHT;
	}
	if (lightmap_owner.owns(p_rid)) {
		return RS::INSTANCE_LIGHTMAP;
	}
	if (particles_owner.owns(p_rid)) {
		return RS::INSTANCE_PARTICLES;
	}
	if (particles_collision_owner.owns(p_rid)) {
		return RS::INSTANCE_PARTICLES_COLLISION;
	}
	if (fog_volume_owner.owns(p_rid)) {
		return RS::INSTANCE_FOG_VOLUME;
	}
	if (visibility_notifier_owner.owns(p_rid)) {
		return RS::INSTANCE_VISIBLITY_NOTIFIER;
	}

	return RS::INSTANCE_NONE;
}

void RendererStorageRD::update_dirty_resources() {
	RendererRD::MaterialStorage::get_singleton()->_update_global_variables(); //must do before materials, so it can queue them for update
	RendererRD::MaterialStorage::get_singleton()->_update_queued_materials();
	RendererRD::MeshStorage::get_singleton()->_update_dirty_multimeshes();
	RendererRD::MeshStorage::get_singleton()->_update_dirty_skeletons();
	RendererRD::DecalAtlasStorage::get_singleton()->update_decal_atlas();
}

bool RendererStorageRD::has_os_feature(const String &p_feature) const {
	if (p_feature == "rgtc" && RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_BC5_UNORM_BLOCK, RD::TEXTURE_USAGE_SAMPLING_BIT)) {
		return true;
	}

	if (p_feature == "s3tc" && RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_BC1_RGB_UNORM_BLOCK, RD::TEXTURE_USAGE_SAMPLING_BIT)) {
		return true;
	}

	if (p_feature == "bptc" && RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_BC7_UNORM_BLOCK, RD::TEXTURE_USAGE_SAMPLING_BIT)) {
		return true;
	}

	if ((p_feature == "etc" || p_feature == "etc2") && RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_ETC2_R8G8B8_UNORM_BLOCK, RD::TEXTURE_USAGE_SAMPLING_BIT)) {
		return true;
	}

	return false;
}

bool RendererStorageRD::free(RID p_rid) {
	if (RendererRD::TextureStorage::get_singleton()->owns_texture(p_rid)) {
		RendererRD::TextureStorage::get_singleton()->texture_free(p_rid);
	} else if (RendererRD::CanvasTextureStorage::get_singleton()->owns_canvas_texture(p_rid)) {
		RendererRD::CanvasTextureStorage::get_singleton()->canvas_texture_free(p_rid);
	} else if (RendererRD::MaterialStorage::get_singleton()->owns_shader(p_rid)) {
		RendererRD::MaterialStorage::get_singleton()->shader_free(p_rid);
	} else if (RendererRD::MaterialStorage::get_singleton()->owns_material(p_rid)) {
		RendererRD::MaterialStorage::get_singleton()->material_free(p_rid);
	} else if (RendererRD::MeshStorage::get_singleton()->owns_mesh(p_rid)) {
		RendererRD::MeshStorage::get_singleton()->mesh_free(p_rid);
	} else if (RendererRD::MeshStorage::get_singleton()->owns_mesh_instance(p_rid)) {
		RendererRD::MeshStorage::get_singleton()->mesh_instance_free(p_rid);
	} else if (RendererRD::MeshStorage::get_singleton()->owns_multimesh(p_rid)) {
		RendererRD::MeshStorage::get_singleton()->multimesh_free(p_rid);
	} else if (RendererRD::MeshStorage::get_singleton()->owns_skeleton(p_rid)) {
		RendererRD::MeshStorage::get_singleton()->skeleton_free(p_rid);
	} else if (reflection_probe_owner.owns(p_rid)) {
		ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_rid);
		reflection_probe->dependency.deleted_notify(p_rid);
		reflection_probe_owner.free(p_rid);
	} else if (RendererRD::DecalAtlasStorage::get_singleton()->owns_decal(p_rid)) {
		RendererRD::DecalAtlasStorage::get_singleton()->decal_free(p_rid);
	} else if (voxel_gi_owner.owns(p_rid)) {
		voxel_gi_allocate_data(p_rid, Transform3D(), AABB(), Vector3i(), Vector<uint8_t>(), Vector<uint8_t>(), Vector<uint8_t>(), Vector<int>()); //deallocate
		VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_rid);
		voxel_gi->dependency.deleted_notify(p_rid);
		voxel_gi_owner.free(p_rid);
	} else if (lightmap_owner.owns(p_rid)) {
		lightmap_set_textures(p_rid, RID(), false);
		Lightmap *lightmap = lightmap_owner.get_or_null(p_rid);
		lightmap->dependency.deleted_notify(p_rid);
		lightmap_owner.free(p_rid);

	} else if (light_owner.owns(p_rid)) {
		light_set_projector(p_rid, RID()); //clear projector
		// delete the texture
		Light *light = light_owner.get_or_null(p_rid);
		light->dependency.deleted_notify(p_rid);
		light_owner.free(p_rid);

	} else if (particles_owner.owns(p_rid)) {
		update_particles();
		Particles *particles = particles_owner.get_or_null(p_rid);
		particles->dependency.deleted_notify(p_rid);
		_particles_free_data(particles);
		particles_owner.free(p_rid);
	} else if (particles_collision_owner.owns(p_rid)) {
		ParticlesCollision *particles_collision = particles_collision_owner.get_or_null(p_rid);

		if (particles_collision->heightfield_texture.is_valid()) {
			RD::get_singleton()->free(particles_collision->heightfield_texture);
		}
		particles_collision->dependency.deleted_notify(p_rid);
		particles_collision_owner.free(p_rid);
	} else if (visibility_notifier_owner.owns(p_rid)) {
		VisibilityNotifier *vn = visibility_notifier_owner.get_or_null(p_rid);
		vn->dependency.deleted_notify(p_rid);
		visibility_notifier_owner.free(p_rid);
	} else if (particles_collision_instance_owner.owns(p_rid)) {
		particles_collision_instance_owner.free(p_rid);
	} else if (fog_volume_owner.owns(p_rid)) {
		FogVolume *fog_volume = fog_volume_owner.get_or_null(p_rid);
		fog_volume->dependency.deleted_notify(p_rid);
		fog_volume_owner.free(p_rid);
	} else if (render_target_owner.owns(p_rid)) {
		RenderTarget *rt = render_target_owner.get_or_null(p_rid);

		_clear_render_target(rt);

		if (rt->texture.is_valid()) {
			RendererRD::Texture *tex = RendererRD::TextureStorage::get_singleton()->get_texture(rt->texture);
			tex->is_render_target = false;
			free(rt->texture);
		}

		render_target_owner.free(p_rid);
	} else {
		return false;
	}

	return true;
}

void RendererStorageRD::init_effects(bool p_prefer_raster_effects) {
	effects = memnew(EffectsRD(p_prefer_raster_effects));
}

EffectsRD *RendererStorageRD::get_effects() {
	ERR_FAIL_NULL_V_MSG(effects, nullptr, "Effects haven't been initialised yet.");
	return effects;
}

void RendererStorageRD::capture_timestamps_begin() {
	RD::get_singleton()->capture_timestamp("Frame Begin");
}

void RendererStorageRD::capture_timestamp(const String &p_name) {
	RD::get_singleton()->capture_timestamp(p_name);
}

uint32_t RendererStorageRD::get_captured_timestamps_count() const {
	return RD::get_singleton()->get_captured_timestamps_count();
}

uint64_t RendererStorageRD::get_captured_timestamps_frame() const {
	return RD::get_singleton()->get_captured_timestamps_frame();
}

uint64_t RendererStorageRD::get_captured_timestamp_gpu_time(uint32_t p_index) const {
	return RD::get_singleton()->get_captured_timestamp_gpu_time(p_index);
}

uint64_t RendererStorageRD::get_captured_timestamp_cpu_time(uint32_t p_index) const {
	return RD::get_singleton()->get_captured_timestamp_cpu_time(p_index);
}

String RendererStorageRD::get_captured_timestamp_name(uint32_t p_index) const {
	return RD::get_singleton()->get_captured_timestamp_name(p_index);
}

void RendererStorageRD::update_memory_info() {
	texture_mem_cache = RenderingDevice::get_singleton()->get_memory_usage(RenderingDevice::MEMORY_TEXTURES);
	buffer_mem_cache = RenderingDevice::get_singleton()->get_memory_usage(RenderingDevice::MEMORY_BUFFERS);
	total_mem_cache = RenderingDevice::get_singleton()->get_memory_usage(RenderingDevice::MEMORY_TOTAL);
}
uint64_t RendererStorageRD::get_rendering_info(RS::RenderingInfo p_info) {
	if (p_info == RS::RENDERING_INFO_TEXTURE_MEM_USED) {
		return texture_mem_cache;
	} else if (p_info == RS::RENDERING_INFO_BUFFER_MEM_USED) {
		return buffer_mem_cache;
	} else if (p_info == RS::RENDERING_INFO_VIDEO_MEM_USED) {
		return total_mem_cache;
	}
	return 0;
}

String RendererStorageRD::get_video_adapter_name() const {
	return RenderingDevice::get_singleton()->get_device_name();
}

String RendererStorageRD::get_video_adapter_vendor() const {
	return RenderingDevice::get_singleton()->get_device_vendor_name();
}

RenderingDevice::DeviceType RendererStorageRD::get_video_adapter_type() const {
	return RenderingDevice::get_singleton()->get_device_type();
}

RendererStorageRD *RendererStorageRD::base_singleton = nullptr;

RendererStorageRD::RendererStorageRD() {
	base_singleton = this;

	RendererRD::TextureStorage *texture_storage = RendererRD::TextureStorage::get_singleton();
	RendererRD::MaterialStorage *material_storage = RendererRD::MaterialStorage::get_singleton();

	//default samplers
	for (int i = 1; i < RS::CANVAS_ITEM_TEXTURE_FILTER_MAX; i++) {
		for (int j = 1; j < RS::CANVAS_ITEM_TEXTURE_REPEAT_MAX; j++) {
			RD::SamplerState sampler_state;
			switch (i) {
				case RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST: {
					sampler_state.mag_filter = RD::SAMPLER_FILTER_NEAREST;
					sampler_state.min_filter = RD::SAMPLER_FILTER_NEAREST;
					sampler_state.max_lod = 0;
				} break;
				case RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR: {
					sampler_state.mag_filter = RD::SAMPLER_FILTER_LINEAR;
					sampler_state.min_filter = RD::SAMPLER_FILTER_LINEAR;
					sampler_state.max_lod = 0;
				} break;
				case RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST_WITH_MIPMAPS: {
					sampler_state.mag_filter = RD::SAMPLER_FILTER_NEAREST;
					sampler_state.min_filter = RD::SAMPLER_FILTER_NEAREST;
					if (GLOBAL_GET("rendering/textures/default_filters/use_nearest_mipmap_filter")) {
						sampler_state.mip_filter = RD::SAMPLER_FILTER_NEAREST;
					} else {
						sampler_state.mip_filter = RD::SAMPLER_FILTER_LINEAR;
					}
				} break;
				case RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS: {
					sampler_state.mag_filter = RD::SAMPLER_FILTER_LINEAR;
					sampler_state.min_filter = RD::SAMPLER_FILTER_LINEAR;
					if (GLOBAL_GET("rendering/textures/default_filters/use_nearest_mipmap_filter")) {
						sampler_state.mip_filter = RD::SAMPLER_FILTER_NEAREST;
					} else {
						sampler_state.mip_filter = RD::SAMPLER_FILTER_LINEAR;
					}

				} break;
				case RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST_WITH_MIPMAPS_ANISOTROPIC: {
					sampler_state.mag_filter = RD::SAMPLER_FILTER_NEAREST;
					sampler_state.min_filter = RD::SAMPLER_FILTER_NEAREST;
					if (GLOBAL_GET("rendering/textures/default_filters/use_nearest_mipmap_filter")) {
						sampler_state.mip_filter = RD::SAMPLER_FILTER_NEAREST;
					} else {
						sampler_state.mip_filter = RD::SAMPLER_FILTER_LINEAR;
					}
					sampler_state.use_anisotropy = true;
					sampler_state.anisotropy_max = 1 << int(GLOBAL_GET("rendering/textures/default_filters/anisotropic_filtering_level"));
				} break;
				case RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS_ANISOTROPIC: {
					sampler_state.mag_filter = RD::SAMPLER_FILTER_LINEAR;
					sampler_state.min_filter = RD::SAMPLER_FILTER_LINEAR;
					if (GLOBAL_GET("rendering/textures/default_filters/use_nearest_mipmap_filter")) {
						sampler_state.mip_filter = RD::SAMPLER_FILTER_NEAREST;
					} else {
						sampler_state.mip_filter = RD::SAMPLER_FILTER_LINEAR;
					}
					sampler_state.use_anisotropy = true;
					sampler_state.anisotropy_max = 1 << int(GLOBAL_GET("rendering/textures/default_filters/anisotropic_filtering_level"));

				} break;
				default: {
				}
			}
			switch (j) {
				case RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED: {
					sampler_state.repeat_u = RD::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;
					sampler_state.repeat_v = RD::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;
					sampler_state.repeat_w = RD::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;

				} break;
				case RS::CANVAS_ITEM_TEXTURE_REPEAT_ENABLED: {
					sampler_state.repeat_u = RD::SAMPLER_REPEAT_MODE_REPEAT;
					sampler_state.repeat_v = RD::SAMPLER_REPEAT_MODE_REPEAT;
					sampler_state.repeat_w = RD::SAMPLER_REPEAT_MODE_REPEAT;
				} break;
				case RS::CANVAS_ITEM_TEXTURE_REPEAT_MIRROR: {
					sampler_state.repeat_u = RD::SAMPLER_REPEAT_MODE_MIRRORED_REPEAT;
					sampler_state.repeat_v = RD::SAMPLER_REPEAT_MODE_MIRRORED_REPEAT;
					sampler_state.repeat_w = RD::SAMPLER_REPEAT_MODE_MIRRORED_REPEAT;
				} break;
				default: {
				}
			}

			default_rd_samplers[i][j] = RD::get_singleton()->sampler_create(sampler_state);
		}
	}

	//custom sampler
	sampler_rd_configure_custom(0.0f);

	using_lightmap_array = true; // high end
	if (using_lightmap_array) {
		uint64_t textures_per_stage = RD::get_singleton()->limit_get(RD::LIMIT_MAX_TEXTURES_PER_SHADER_STAGE);

		if (textures_per_stage <= 256) {
			lightmap_textures.resize(32);
		} else {
			lightmap_textures.resize(1024);
		}

		for (int i = 0; i < lightmap_textures.size(); i++) {
			lightmap_textures.write[i] = texture_storage->texture_rd_get_default(RendererRD::DEFAULT_RD_TEXTURE_2D_ARRAY_WHITE);
		}
	}

	lightmap_probe_capture_update_speed = GLOBAL_GET("rendering/lightmapping/probe_capture/update_speed");

	/* Particles */

	{
		// Initialize particles
		Vector<String> particles_modes;
		particles_modes.push_back("");
		particles_shader.shader.initialize(particles_modes, String());
	}
	RendererRD::MaterialStorage::get_singleton()->shader_set_data_request_function(RendererRD::SHADER_TYPE_PARTICLES, _create_particles_shader_funcs);
	RendererRD::MaterialStorage::get_singleton()->material_set_data_request_function(RendererRD::SHADER_TYPE_PARTICLES, _create_particles_material_funcs);

	{
		ShaderCompiler::DefaultIdentifierActions actions;

		actions.renames["COLOR"] = "PARTICLE.color";
		actions.renames["VELOCITY"] = "PARTICLE.velocity";
		//actions.renames["MASS"] = "mass"; ?
		actions.renames["ACTIVE"] = "particle_active";
		actions.renames["RESTART"] = "restart";
		actions.renames["CUSTOM"] = "PARTICLE.custom";
		for (int i = 0; i < ParticlesShader::MAX_USERDATAS; i++) {
			String udname = "USERDATA" + itos(i + 1);
			actions.renames[udname] = "PARTICLE.userdata" + itos(i + 1);
			actions.usage_defines[udname] = "#define USERDATA" + itos(i + 1) + "_USED\n";
		}
		actions.renames["TRANSFORM"] = "PARTICLE.xform";
		actions.renames["TIME"] = "frame_history.data[0].time";
		actions.renames["PI"] = _MKSTR(Math_PI);
		actions.renames["TAU"] = _MKSTR(Math_TAU);
		actions.renames["E"] = _MKSTR(Math_E);
		actions.renames["LIFETIME"] = "params.lifetime";
		actions.renames["DELTA"] = "local_delta";
		actions.renames["NUMBER"] = "particle_number";
		actions.renames["INDEX"] = "index";
		//actions.renames["GRAVITY"] = "current_gravity";
		actions.renames["EMISSION_TRANSFORM"] = "FRAME.emission_transform";
		actions.renames["RANDOM_SEED"] = "FRAME.random_seed";
		actions.renames["FLAG_EMIT_POSITION"] = "EMISSION_FLAG_HAS_POSITION";
		actions.renames["FLAG_EMIT_ROT_SCALE"] = "EMISSION_FLAG_HAS_ROTATION_SCALE";
		actions.renames["FLAG_EMIT_VELOCITY"] = "EMISSION_FLAG_HAS_VELOCITY";
		actions.renames["FLAG_EMIT_COLOR"] = "EMISSION_FLAG_HAS_COLOR";
		actions.renames["FLAG_EMIT_CUSTOM"] = "EMISSION_FLAG_HAS_CUSTOM";
		actions.renames["RESTART_POSITION"] = "restart_position";
		actions.renames["RESTART_ROT_SCALE"] = "restart_rotation_scale";
		actions.renames["RESTART_VELOCITY"] = "restart_velocity";
		actions.renames["RESTART_COLOR"] = "restart_color";
		actions.renames["RESTART_CUSTOM"] = "restart_custom";
		actions.renames["emit_subparticle"] = "emit_subparticle";
		actions.renames["COLLIDED"] = "collided";
		actions.renames["COLLISION_NORMAL"] = "collision_normal";
		actions.renames["COLLISION_DEPTH"] = "collision_depth";
		actions.renames["ATTRACTOR_FORCE"] = "attractor_force";

		actions.render_mode_defines["disable_force"] = "#define DISABLE_FORCE\n";
		actions.render_mode_defines["disable_velocity"] = "#define DISABLE_VELOCITY\n";
		actions.render_mode_defines["keep_data"] = "#define ENABLE_KEEP_DATA\n";
		actions.render_mode_defines["collision_use_scale"] = "#define USE_COLLISON_SCALE\n";

		actions.sampler_array_name = "material_samplers";
		actions.base_texture_binding_index = 1;
		actions.texture_layout_set = 3;
		actions.base_uniform_string = "material.";
		actions.base_varying_index = 10;

		actions.default_filter = ShaderLanguage::FILTER_LINEAR_MIPMAP;
		actions.default_repeat = ShaderLanguage::REPEAT_ENABLE;
		actions.global_buffer_array_variable = "global_variables.data";

		particles_shader.compiler.initialize(actions);
	}

	{
		// default material and shader for particles shader
		particles_shader.default_shader = material_storage->shader_allocate();
		material_storage->shader_initialize(particles_shader.default_shader);
		material_storage->shader_set_code(particles_shader.default_shader, R"(
// Default particles shader.

shader_type particles;

void process() {
	COLOR = vec4(1.0);
}
)");
		particles_shader.default_material = material_storage->material_allocate();
		material_storage->material_initialize(particles_shader.default_material);
		material_storage->material_set_shader(particles_shader.default_material, particles_shader.default_shader);

		ParticlesMaterialData *md = static_cast<ParticlesMaterialData *>(material_storage->material_get_data(particles_shader.default_material, RendererRD::SHADER_TYPE_PARTICLES));
		particles_shader.default_shader_rd = particles_shader.shader.version_get_shader(md->shader_data->version, 0);

		Vector<RD::Uniform> uniforms;

		{
			Vector<RID> ids;
			ids.resize(12);
			RID *ids_ptr = ids.ptrw();
			ids_ptr[0] = sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
			ids_ptr[1] = sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
			ids_ptr[2] = sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST_WITH_MIPMAPS, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
			ids_ptr[3] = sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
			ids_ptr[4] = sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST_WITH_MIPMAPS_ANISOTROPIC, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
			ids_ptr[5] = sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS_ANISOTROPIC, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
			ids_ptr[6] = sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST, RS::CANVAS_ITEM_TEXTURE_REPEAT_ENABLED);
			ids_ptr[7] = sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RS::CANVAS_ITEM_TEXTURE_REPEAT_ENABLED);
			ids_ptr[8] = sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST_WITH_MIPMAPS, RS::CANVAS_ITEM_TEXTURE_REPEAT_ENABLED);
			ids_ptr[9] = sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS, RS::CANVAS_ITEM_TEXTURE_REPEAT_ENABLED);
			ids_ptr[10] = sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST_WITH_MIPMAPS_ANISOTROPIC, RS::CANVAS_ITEM_TEXTURE_REPEAT_ENABLED);
			ids_ptr[11] = sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS_ANISOTROPIC, RS::CANVAS_ITEM_TEXTURE_REPEAT_ENABLED);

			RD::Uniform u(RD::UNIFORM_TYPE_SAMPLER, 1, ids);
			uniforms.push_back(u);
		}

		{
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
			u.binding = 2;
			u.append_id(material_storage->global_variables_get_storage_buffer());
			uniforms.push_back(u);
		}

		particles_shader.base_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, particles_shader.default_shader_rd, 0);
	}

	{
		Vector<String> copy_modes;
		for (int i = 0; i <= ParticlesShader::MAX_USERDATAS; i++) {
			if (i == 0) {
				copy_modes.push_back("\n#define MODE_FILL_INSTANCES\n");
				copy_modes.push_back("\n#define MODE_FILL_SORT_BUFFER\n#define USE_SORT_BUFFER\n");
				copy_modes.push_back("\n#define MODE_FILL_INSTANCES\n#define USE_SORT_BUFFER\n");
			} else {
				copy_modes.push_back("\n#define MODE_FILL_INSTANCES\n#define USERDATA_COUNT " + itos(i) + "\n");
				copy_modes.push_back("\n#define MODE_FILL_SORT_BUFFER\n#define USE_SORT_BUFFER\n#define USERDATA_COUNT " + itos(i) + "\n");
				copy_modes.push_back("\n#define MODE_FILL_INSTANCES\n#define USE_SORT_BUFFER\n#define USERDATA_COUNT " + itos(i) + "\n");
			}
		}

		particles_shader.copy_shader.initialize(copy_modes);

		particles_shader.copy_shader_version = particles_shader.copy_shader.version_create();

		for (int i = 0; i <= ParticlesShader::MAX_USERDATAS; i++) {
			for (int j = 0; j < ParticlesShader::COPY_MODE_MAX; j++) {
				particles_shader.copy_pipelines[i * ParticlesShader::COPY_MODE_MAX + j] = RD::get_singleton()->compute_pipeline_create(particles_shader.copy_shader.version_get_shader(particles_shader.copy_shader_version, i * ParticlesShader::COPY_MODE_MAX + j));
			}
		}
	}

	{
		Vector<String> sdf_modes;
		sdf_modes.push_back("\n#define MODE_LOAD\n");
		sdf_modes.push_back("\n#define MODE_LOAD_SHRINK\n");
		sdf_modes.push_back("\n#define MODE_PROCESS\n");
		sdf_modes.push_back("\n#define MODE_PROCESS_OPTIMIZED\n");
		sdf_modes.push_back("\n#define MODE_STORE\n");
		sdf_modes.push_back("\n#define MODE_STORE_SHRINK\n");

		rt_sdf.shader.initialize(sdf_modes);

		rt_sdf.shader_version = rt_sdf.shader.version_create();

		for (int i = 0; i < RenderTargetSDF::SHADER_MAX; i++) {
			rt_sdf.pipelines[i] = RD::get_singleton()->compute_pipeline_create(rt_sdf.shader.version_get_shader(rt_sdf.shader_version, i));
		}
	}
}

RendererStorageRD::~RendererStorageRD() {
	RendererRD::MaterialStorage *material_storage = RendererRD::MaterialStorage::get_singleton();

	//def samplers
	for (int i = 1; i < RS::CANVAS_ITEM_TEXTURE_FILTER_MAX; i++) {
		for (int j = 1; j < RS::CANVAS_ITEM_TEXTURE_REPEAT_MAX; j++) {
			RD::get_singleton()->free(default_rd_samplers[i][j]);
		}
	}

	//custom samplers
	for (int i = 1; i < RS::CANVAS_ITEM_TEXTURE_FILTER_MAX; i++) {
		for (int j = 0; j < RS::CANVAS_ITEM_TEXTURE_REPEAT_MAX; j++) {
			if (custom_rd_samplers[i][j].is_valid()) {
				RD::get_singleton()->free(custom_rd_samplers[i][j]);
			}
		}
	}

	particles_shader.copy_shader.version_free(particles_shader.copy_shader_version);
	rt_sdf.shader.version_free(rt_sdf.shader_version);

	material_storage->material_free(particles_shader.default_material);
	material_storage->shader_free(particles_shader.default_shader);

	if (effects) {
		memdelete(effects);
		effects = nullptr;
	}
}
