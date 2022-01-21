/*************************************************************************/
/*  gpu_particles_3d.cpp                                                 */
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

#include "gpu_particles_3d.h"

#include "scene/resources/particles_material.h"

AABB GPUParticles3D::get_aabb() const {
	return AABB();
}

Vector<Face3> GPUParticles3D::get_faces(uint32_t p_usage_flags) const {
	return Vector<Face3>();
}

void GPUParticles3D::set_emitting(bool p_emitting) {
	RS::get_singleton()->particles_set_emitting(particles, p_emitting);

	if (p_emitting && one_shot) {
		set_process_internal(true);
	} else if (!p_emitting) {
		set_process_internal(false);
	}
}

void GPUParticles3D::set_amount(int p_amount) {
	ERR_FAIL_COND_MSG(p_amount < 1, "Amount of particles cannot be smaller than 1.");
	amount = p_amount;
	RS::get_singleton()->particles_set_amount(particles, amount);
}

void GPUParticles3D::set_lifetime(double p_lifetime) {
	ERR_FAIL_COND_MSG(p_lifetime <= 0, "Particles lifetime must be greater than 0.");
	lifetime = p_lifetime;
	RS::get_singleton()->particles_set_lifetime(particles, lifetime);
}

void GPUParticles3D::set_one_shot(bool p_one_shot) {
	one_shot = p_one_shot;
	RS::get_singleton()->particles_set_one_shot(particles, one_shot);

	if (is_emitting()) {
		set_process_internal(true);
		if (!one_shot) {
			RenderingServer::get_singleton()->particles_restart(particles);
		}
	}

	if (!one_shot) {
		set_process_internal(false);
	}
}

void GPUParticles3D::set_pre_process_time(double p_time) {
	pre_process_time = p_time;
	RS::get_singleton()->particles_set_pre_process_time(particles, pre_process_time);
}

void GPUParticles3D::set_explosiveness_ratio(real_t p_ratio) {
	explosiveness_ratio = p_ratio;
	RS::get_singleton()->particles_set_explosiveness_ratio(particles, explosiveness_ratio);
}

void GPUParticles3D::set_randomness_ratio(real_t p_ratio) {
	randomness_ratio = p_ratio;
	RS::get_singleton()->particles_set_randomness_ratio(particles, randomness_ratio);
}

void GPUParticles3D::set_visibility_aabb(const AABB &p_aabb) {
	visibility_aabb = p_aabb;
	RS::get_singleton()->particles_set_custom_aabb(particles, visibility_aabb);
	update_gizmos();
}

void GPUParticles3D::set_use_local_coordinates(bool p_enable) {
	local_coords = p_enable;
	RS::get_singleton()->particles_set_use_local_coordinates(particles, local_coords);
}

void GPUParticles3D::set_process_material(const Ref<Material> &p_material) {
	process_material = p_material;
	RID material_rid;
	if (process_material.is_valid()) {
		material_rid = process_material->get_rid();
	}
	RS::get_singleton()->particles_set_process_material(particles, material_rid);

	update_configuration_warnings();
}

void GPUParticles3D::set_speed_scale(double p_scale) {
	speed_scale = p_scale;
	RS::get_singleton()->particles_set_speed_scale(particles, p_scale);
}

void GPUParticles3D::set_collision_base_size(real_t p_size) {
	collision_base_size = p_size;
	RS::get_singleton()->particles_set_collision_base_size(particles, p_size);
}

bool GPUParticles3D::is_emitting() const {
	return RS::get_singleton()->particles_get_emitting(particles);
}

int GPUParticles3D::get_amount() const {
	return amount;
}

double GPUParticles3D::get_lifetime() const {
	return lifetime;
}

bool GPUParticles3D::get_one_shot() const {
	return one_shot;
}

double GPUParticles3D::get_pre_process_time() const {
	return pre_process_time;
}

real_t GPUParticles3D::get_explosiveness_ratio() const {
	return explosiveness_ratio;
}

real_t GPUParticles3D::get_randomness_ratio() const {
	return randomness_ratio;
}

AABB GPUParticles3D::get_visibility_aabb() const {
	return visibility_aabb;
}

bool GPUParticles3D::get_use_local_coordinates() const {
	return local_coords;
}

Ref<Material> GPUParticles3D::get_process_material() const {
	return process_material;
}

double GPUParticles3D::get_speed_scale() const {
	return speed_scale;
}

real_t GPUParticles3D::get_collision_base_size() const {
	return collision_base_size;
}

void GPUParticles3D::set_draw_order(DrawOrder p_order) {
	draw_order = p_order;
	RS::get_singleton()->particles_set_draw_order(particles, RS::ParticlesDrawOrder(p_order));
}

void GPUParticles3D::set_trail_enabled(bool p_enabled) {
	trail_enabled = p_enabled;
	RS::get_singleton()->particles_set_trails(particles, trail_enabled, trail_length);
	update_configuration_warnings();
}

void GPUParticles3D::set_trail_length(double p_seconds) {
	ERR_FAIL_COND(p_seconds < 0.001);
	trail_length = p_seconds;
	RS::get_singleton()->particles_set_trails(particles, trail_enabled, trail_length);
}

bool GPUParticles3D::is_trail_enabled() const {
	return trail_enabled;
}

double GPUParticles3D::get_trail_length() const {
	return trail_length;
}

GPUParticles3D::DrawOrder GPUParticles3D::get_draw_order() const {
	return draw_order;
}

void GPUParticles3D::set_draw_passes(int p_count) {
	ERR_FAIL_COND(p_count < 1);
	for (int i = p_count; i < draw_passes.size(); i++) {
		set_draw_pass_mesh(i, Ref<Mesh>());
	}
	draw_passes.resize(p_count);
	RS::get_singleton()->particles_set_draw_passes(particles, p_count);
	notify_property_list_changed();
}

int GPUParticles3D::get_draw_passes() const {
	return draw_passes.size();
}

void GPUParticles3D::set_draw_pass_mesh(int p_pass, const Ref<Mesh> &p_mesh) {
	ERR_FAIL_INDEX(p_pass, draw_passes.size());

	if (Engine::get_singleton()->is_editor_hint() && draw_passes.write[p_pass].is_valid()) {
		draw_passes.write[p_pass]->disconnect("changed", callable_mp((Node *)this, &Node::update_configuration_warnings));
	}

	draw_passes.write[p_pass] = p_mesh;

	if (Engine::get_singleton()->is_editor_hint() && draw_passes.write[p_pass].is_valid()) {
		draw_passes.write[p_pass]->connect("changed", callable_mp((Node *)this, &Node::update_configuration_warnings), varray(), CONNECT_DEFERRED);
	}

	RID mesh_rid;
	if (p_mesh.is_valid()) {
		mesh_rid = p_mesh->get_rid();
	}

	RS::get_singleton()->particles_set_draw_pass_mesh(particles, p_pass, mesh_rid);

	_skinning_changed();
	update_configuration_warnings();
}

Ref<Mesh> GPUParticles3D::get_draw_pass_mesh(int p_pass) const {
	ERR_FAIL_INDEX_V(p_pass, draw_passes.size(), Ref<Mesh>());

	return draw_passes[p_pass];
}

void GPUParticles3D::set_fixed_fps(int p_count) {
	fixed_fps = p_count;
	RS::get_singleton()->particles_set_fixed_fps(particles, p_count);
}

int GPUParticles3D::get_fixed_fps() const {
	return fixed_fps;
}

void GPUParticles3D::set_fractional_delta(bool p_enable) {
	fractional_delta = p_enable;
	RS::get_singleton()->particles_set_fractional_delta(particles, p_enable);
}

bool GPUParticles3D::get_fractional_delta() const {
	return fractional_delta;
}

void GPUParticles3D::set_interpolate(bool p_enable) {
	interpolate = p_enable;
	RS::get_singleton()->particles_set_interpolate(particles, p_enable);
}

bool GPUParticles3D::get_interpolate() const {
	return interpolate;
}

TypedArray<String> GPUParticles3D::get_configuration_warnings() const {
	TypedArray<String> warnings = Node::get_configuration_warnings();

	if (RenderingServer::get_singleton()->is_low_end()) {
		warnings.push_back(TTR("GPU-based particles are not supported by the OpenGL video driver.\nUse the CPUParticles3D node instead. You can use the \"Convert to CPUParticles3D\" option for this purpose."));
	}

	bool meshes_found = false;
	bool anim_material_found = false;

	for (int i = 0; i < draw_passes.size(); i++) {
		if (draw_passes[i].is_valid()) {
			meshes_found = true;
			for (int j = 0; j < draw_passes[i]->get_surface_count(); j++) {
				anim_material_found = Object::cast_to<ShaderMaterial>(draw_passes[i]->surface_get_material(j).ptr()) != nullptr;
				BaseMaterial3D *spat = Object::cast_to<BaseMaterial3D>(draw_passes[i]->surface_get_material(j).ptr());
				anim_material_found = anim_material_found || (spat && spat->get_billboard_mode() == StandardMaterial3D::BILLBOARD_PARTICLES);
			}
			if (anim_material_found) {
				break;
			}
		}
	}

	anim_material_found = anim_material_found || Object::cast_to<ShaderMaterial>(get_material_override().ptr()) != nullptr;
	{
		BaseMaterial3D *spat = Object::cast_to<BaseMaterial3D>(get_material_override().ptr());
		anim_material_found = anim_material_found || (spat && spat->get_billboard_mode() == BaseMaterial3D::BILLBOARD_PARTICLES);
	}

	if (!meshes_found) {
		warnings.push_back(TTR("Nothing is visible because meshes have not been assigned to draw passes."));
	}

	if (process_material.is_null()) {
		warnings.push_back(TTR("A material to process the particles is not assigned, so no behavior is imprinted."));
	} else {
		const ParticlesMaterial *process = Object::cast_to<ParticlesMaterial>(process_material.ptr());
		if (!anim_material_found && process &&
				(process->get_param_max(ParticlesMaterial::PARAM_ANIM_SPEED) != 0.0 || process->get_param_max(ParticlesMaterial::PARAM_ANIM_OFFSET) != 0.0 ||
						process->get_param_texture(ParticlesMaterial::PARAM_ANIM_SPEED).is_valid() || process->get_param_texture(ParticlesMaterial::PARAM_ANIM_OFFSET).is_valid())) {
			warnings.push_back(TTR("Particles animation requires the usage of a BaseMaterial3D whose Billboard Mode is set to \"Particle Billboard\"."));
		}
	}

	if (trail_enabled) {
		int dp_count = 0;
		bool missing_trails = false;
		bool no_materials = false;

		for (int i = 0; i < draw_passes.size(); i++) {
			Ref<Mesh> draw_pass = draw_passes[i];
			if (draw_pass.is_valid() && draw_pass->get_builtin_bind_pose_count() > 0) {
				dp_count++;
			}

			if (draw_pass.is_valid()) {
				int mats_found = 0;
				for (int j = 0; j < draw_passes[i]->get_surface_count(); j++) {
					BaseMaterial3D *spat = Object::cast_to<BaseMaterial3D>(draw_passes[i]->surface_get_material(j).ptr());
					if (spat) {
						mats_found++;
					}
					if (spat && !spat->get_flag(BaseMaterial3D::FLAG_PARTICLE_TRAILS_MODE)) {
						missing_trails = true;
					}
				}

				if (mats_found != draw_passes[i]->get_surface_count()) {
					no_materials = true;
				}
			}
		}

		BaseMaterial3D *spat = Object::cast_to<BaseMaterial3D>(get_material_override().ptr());
		if (spat) {
			no_materials = false;
		}
		if (spat && !spat->get_flag(BaseMaterial3D::FLAG_PARTICLE_TRAILS_MODE)) {
			missing_trails = true;
		}

		if (dp_count && skin.is_valid()) {
			warnings.push_back(TTR("Using Trail meshes with a skin causes Skin to override Trail poses. Suggest removing the Skin."));
		} else if (dp_count == 0 && skin.is_null()) {
			warnings.push_back(TTR("Trails active, but neither Trail meshes or a Skin were found."));
		} else if (dp_count > 1) {
			warnings.push_back(TTR("Only one Trail mesh is supported. If you want to use more than a single mesh, a Skin is needed (see documentation)."));
		}

		if ((dp_count || !skin.is_null()) && (missing_trails || no_materials)) {
			warnings.push_back(TTR("Trails enabled, but one or more mesh materials are either missing or not set for trails rendering."));
		}
	}

	return warnings;
}

void GPUParticles3D::restart() {
	RenderingServer::get_singleton()->particles_restart(particles);
	RenderingServer::get_singleton()->particles_set_emitting(particles, true);
}

AABB GPUParticles3D::capture_aabb() const {
	return RS::get_singleton()->particles_get_current_aabb(particles);
}

void GPUParticles3D::_validate_property(PropertyInfo &property) const {
	if (property.name.begins_with("draw_pass_")) {
		int index = property.name.get_slicec('_', 2).to_int() - 1;
		if (index >= draw_passes.size()) {
			property.usage = PROPERTY_USAGE_NONE;
			return;
		}
	}

	GeometryInstance3D::_validate_property(property);
}

void GPUParticles3D::emit_particle(const Transform3D &p_transform, const Vector3 &p_velocity, const Color &p_color, const Color &p_custom, uint32_t p_emit_flags) {
	RS::get_singleton()->particles_emit(particles, p_transform, p_velocity, p_color, p_custom, p_emit_flags);
}

void GPUParticles3D::_attach_sub_emitter() {
	Node *n = get_node_or_null(sub_emitter);
	if (n) {
		GPUParticles3D *sen = Object::cast_to<GPUParticles3D>(n);
		if (sen && sen != this) {
			RS::get_singleton()->particles_set_subemitter(particles, sen->particles);
		}
	}
}

void GPUParticles3D::set_sub_emitter(const NodePath &p_path) {
	if (is_inside_tree()) {
		RS::get_singleton()->particles_set_subemitter(particles, RID());
	}

	sub_emitter = p_path;

	if (is_inside_tree() && sub_emitter != NodePath()) {
		_attach_sub_emitter();
	}
}

NodePath GPUParticles3D::get_sub_emitter() const {
	return sub_emitter;
}

void GPUParticles3D::_notification(int p_what) {
	if (p_what == NOTIFICATION_PAUSED || p_what == NOTIFICATION_UNPAUSED) {
		if (can_process()) {
			RS::get_singleton()->particles_set_speed_scale(particles, speed_scale);
		} else {
			RS::get_singleton()->particles_set_speed_scale(particles, 0);
		}
	}

	// Use internal process when emitting and one_shot is on so that when
	// the shot ends the editor can properly update
	if (p_what == NOTIFICATION_INTERNAL_PROCESS) {
		if (one_shot && !is_emitting()) {
			notify_property_list_changed();
			set_process_internal(false);
		}
	}

	if (p_what == NOTIFICATION_ENTER_TREE) {
		if (sub_emitter != NodePath()) {
			_attach_sub_emitter();
		}
	}

	if (p_what == NOTIFICATION_EXIT_TREE) {
		RS::get_singleton()->particles_set_subemitter(particles, RID());
	}

	if (p_what == NOTIFICATION_VISIBILITY_CHANGED) {
		// make sure particles are updated before rendering occurs if they were active before
		if (is_visible_in_tree() && !RS::get_singleton()->particles_is_inactive(particles)) {
			RS::get_singleton()->particles_request_process(particles);
		}
	}
}

void GPUParticles3D::_skinning_changed() {
	Vector<Transform3D> xforms;
	if (skin.is_valid()) {
		xforms.resize(skin->get_bind_count());
		for (int i = 0; i < skin->get_bind_count(); i++) {
			xforms.write[i] = skin->get_bind_pose(i);
		}
	} else {
		for (int i = 0; i < draw_passes.size(); i++) {
			Ref<Mesh> draw_pass = draw_passes[i];
			if (draw_pass.is_valid() && draw_pass->get_builtin_bind_pose_count() > 0) {
				xforms.resize(draw_pass->get_builtin_bind_pose_count());
				for (int j = 0; j < draw_pass->get_builtin_bind_pose_count(); j++) {
					xforms.write[j] = draw_pass->get_builtin_bind_pose(j);
				}
				break;
			}
		}
	}

	RS::get_singleton()->particles_set_trail_bind_poses(particles, xforms);
	update_configuration_warnings();
}

void GPUParticles3D::set_skin(const Ref<Skin> &p_skin) {
	skin = p_skin;
	_skinning_changed();
}

Ref<Skin> GPUParticles3D::get_skin() const {
	return skin;
}

void GPUParticles3D::set_transform_align(TransformAlign p_align) {
	ERR_FAIL_INDEX(uint32_t(p_align), 4);
	transform_align = p_align;
	RS::get_singleton()->particles_set_transform_align(particles, RS::ParticlesTransformAlign(transform_align));
}
GPUParticles3D::TransformAlign GPUParticles3D::get_transform_align() const {
	return transform_align;
}

void GPUParticles3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_emitting", "emitting"), &GPUParticles3D::set_emitting);
	ClassDB::bind_method(D_METHOD("set_amount", "amount"), &GPUParticles3D::set_amount);
	ClassDB::bind_method(D_METHOD("set_lifetime", "secs"), &GPUParticles3D::set_lifetime);
	ClassDB::bind_method(D_METHOD("set_one_shot", "enable"), &GPUParticles3D::set_one_shot);
	ClassDB::bind_method(D_METHOD("set_pre_process_time", "secs"), &GPUParticles3D::set_pre_process_time);
	ClassDB::bind_method(D_METHOD("set_explosiveness_ratio", "ratio"), &GPUParticles3D::set_explosiveness_ratio);
	ClassDB::bind_method(D_METHOD("set_randomness_ratio", "ratio"), &GPUParticles3D::set_randomness_ratio);
	ClassDB::bind_method(D_METHOD("set_visibility_aabb", "aabb"), &GPUParticles3D::set_visibility_aabb);
	ClassDB::bind_method(D_METHOD("set_use_local_coordinates", "enable"), &GPUParticles3D::set_use_local_coordinates);
	ClassDB::bind_method(D_METHOD("set_fixed_fps", "fps"), &GPUParticles3D::set_fixed_fps);
	ClassDB::bind_method(D_METHOD("set_fractional_delta", "enable"), &GPUParticles3D::set_fractional_delta);
	ClassDB::bind_method(D_METHOD("set_interpolate", "enable"), &GPUParticles3D::set_interpolate);
	ClassDB::bind_method(D_METHOD("set_process_material", "material"), &GPUParticles3D::set_process_material);
	ClassDB::bind_method(D_METHOD("set_speed_scale", "scale"), &GPUParticles3D::set_speed_scale);
	ClassDB::bind_method(D_METHOD("set_collision_base_size", "size"), &GPUParticles3D::set_collision_base_size);

	ClassDB::bind_method(D_METHOD("is_emitting"), &GPUParticles3D::is_emitting);
	ClassDB::bind_method(D_METHOD("get_amount"), &GPUParticles3D::get_amount);
	ClassDB::bind_method(D_METHOD("get_lifetime"), &GPUParticles3D::get_lifetime);
	ClassDB::bind_method(D_METHOD("get_one_shot"), &GPUParticles3D::get_one_shot);
	ClassDB::bind_method(D_METHOD("get_pre_process_time"), &GPUParticles3D::get_pre_process_time);
	ClassDB::bind_method(D_METHOD("get_explosiveness_ratio"), &GPUParticles3D::get_explosiveness_ratio);
	ClassDB::bind_method(D_METHOD("get_randomness_ratio"), &GPUParticles3D::get_randomness_ratio);
	ClassDB::bind_method(D_METHOD("get_visibility_aabb"), &GPUParticles3D::get_visibility_aabb);
	ClassDB::bind_method(D_METHOD("get_use_local_coordinates"), &GPUParticles3D::get_use_local_coordinates);
	ClassDB::bind_method(D_METHOD("get_fixed_fps"), &GPUParticles3D::get_fixed_fps);
	ClassDB::bind_method(D_METHOD("get_fractional_delta"), &GPUParticles3D::get_fractional_delta);
	ClassDB::bind_method(D_METHOD("get_interpolate"), &GPUParticles3D::get_interpolate);
	ClassDB::bind_method(D_METHOD("get_process_material"), &GPUParticles3D::get_process_material);
	ClassDB::bind_method(D_METHOD("get_speed_scale"), &GPUParticles3D::get_speed_scale);
	ClassDB::bind_method(D_METHOD("get_collision_base_size"), &GPUParticles3D::get_collision_base_size);

	ClassDB::bind_method(D_METHOD("set_draw_order", "order"), &GPUParticles3D::set_draw_order);

	ClassDB::bind_method(D_METHOD("get_draw_order"), &GPUParticles3D::get_draw_order);

	ClassDB::bind_method(D_METHOD("set_draw_passes", "passes"), &GPUParticles3D::set_draw_passes);
	ClassDB::bind_method(D_METHOD("set_draw_pass_mesh", "pass", "mesh"), &GPUParticles3D::set_draw_pass_mesh);

	ClassDB::bind_method(D_METHOD("get_draw_passes"), &GPUParticles3D::get_draw_passes);
	ClassDB::bind_method(D_METHOD("get_draw_pass_mesh", "pass"), &GPUParticles3D::get_draw_pass_mesh);

	ClassDB::bind_method(D_METHOD("set_skin", "skin"), &GPUParticles3D::set_skin);
	ClassDB::bind_method(D_METHOD("get_skin"), &GPUParticles3D::get_skin);

	ClassDB::bind_method(D_METHOD("restart"), &GPUParticles3D::restart);
	ClassDB::bind_method(D_METHOD("capture_aabb"), &GPUParticles3D::capture_aabb);

	ClassDB::bind_method(D_METHOD("set_sub_emitter", "path"), &GPUParticles3D::set_sub_emitter);
	ClassDB::bind_method(D_METHOD("get_sub_emitter"), &GPUParticles3D::get_sub_emitter);

	ClassDB::bind_method(D_METHOD("emit_particle", "xform", "velocity", "color", "custom", "flags"), &GPUParticles3D::emit_particle);

	ClassDB::bind_method(D_METHOD("set_trail_enabled", "enabled"), &GPUParticles3D::set_trail_enabled);
	ClassDB::bind_method(D_METHOD("set_trail_length", "secs"), &GPUParticles3D::set_trail_length);

	ClassDB::bind_method(D_METHOD("is_trail_enabled"), &GPUParticles3D::is_trail_enabled);
	ClassDB::bind_method(D_METHOD("get_trail_length"), &GPUParticles3D::get_trail_length);

	ClassDB::bind_method(D_METHOD("set_transform_align", "align"), &GPUParticles3D::set_transform_align);
	ClassDB::bind_method(D_METHOD("get_transform_align"), &GPUParticles3D::get_transform_align);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "emitting"), "set_emitting", "is_emitting");
	ADD_PROPERTY_DEFAULT("emitting", true); // Workaround for doctool in headless mode, as dummy rasterizer always returns false.
	ADD_PROPERTY(PropertyInfo(Variant::INT, "amount", PROPERTY_HINT_RANGE, "1,1000000,1,exp"), "set_amount", "get_amount");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "sub_emitter", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "GPUParticles3D"), "set_sub_emitter", "get_sub_emitter");
	ADD_GROUP("Time", "");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "lifetime", PROPERTY_HINT_RANGE, "0.01,600.0,0.01,or_greater,exp"), "set_lifetime", "get_lifetime");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "one_shot"), "set_one_shot", "get_one_shot");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "preprocess", PROPERTY_HINT_RANGE, "0.00,600.0,0.01,exp"), "set_pre_process_time", "get_pre_process_time");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "speed_scale", PROPERTY_HINT_RANGE, "0,64,0.01"), "set_speed_scale", "get_speed_scale");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "explosiveness", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_explosiveness_ratio", "get_explosiveness_ratio");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "randomness", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_randomness_ratio", "get_randomness_ratio");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "fixed_fps", PROPERTY_HINT_RANGE, "0,1000,1"), "set_fixed_fps", "get_fixed_fps");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "interpolate"), "set_interpolate", "get_interpolate");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "fract_delta"), "set_fractional_delta", "get_fractional_delta");
	ADD_GROUP("Collision", "collision_");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "collision_base_size", PROPERTY_HINT_RANGE, "0,128,0.01,or_greater"), "set_collision_base_size", "get_collision_base_size");
	ADD_GROUP("Drawing", "");
	ADD_PROPERTY(PropertyInfo(Variant::AABB, "visibility_aabb"), "set_visibility_aabb", "get_visibility_aabb");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "local_coords"), "set_use_local_coordinates", "get_use_local_coordinates");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "draw_order", PROPERTY_HINT_ENUM, "Index,Lifetime,Reverse Lifetime,View Depth"), "set_draw_order", "get_draw_order");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "transform_align", PROPERTY_HINT_ENUM, "Disabled,ZBillboard,YToVelocity,ZBillboardYToVelocity"), "set_transform_align", "get_transform_align");
	ADD_GROUP("Trails", "trail_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "trail_enabled"), "set_trail_enabled", "is_trail_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "trail_length_secs", PROPERTY_HINT_RANGE, "0.01,10,0.01"), "set_trail_length", "get_trail_length");
	ADD_GROUP("Process Material", "");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "process_material", PROPERTY_HINT_RESOURCE_TYPE, "ShaderMaterial,ParticlesMaterial"), "set_process_material", "get_process_material");
	ADD_GROUP("Draw Passes", "draw_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "draw_passes", PROPERTY_HINT_RANGE, "0," + itos(MAX_DRAW_PASSES) + ",1"), "set_draw_passes", "get_draw_passes");
	for (int i = 0; i < MAX_DRAW_PASSES; i++) {
		ADD_PROPERTYI(PropertyInfo(Variant::OBJECT, "draw_pass_" + itos(i + 1), PROPERTY_HINT_RESOURCE_TYPE, "Mesh"), "set_draw_pass_mesh", "get_draw_pass_mesh", i);
	}
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "draw_skin", PROPERTY_HINT_RESOURCE_TYPE, "Skin"), "set_skin", "get_skin");

	BIND_ENUM_CONSTANT(DRAW_ORDER_INDEX);
	BIND_ENUM_CONSTANT(DRAW_ORDER_LIFETIME);
	BIND_ENUM_CONSTANT(DRAW_ORDER_REVERSE_LIFETIME);
	BIND_ENUM_CONSTANT(DRAW_ORDER_VIEW_DEPTH);

	BIND_ENUM_CONSTANT(EMIT_FLAG_POSITION);
	BIND_ENUM_CONSTANT(EMIT_FLAG_ROTATION_SCALE);
	BIND_ENUM_CONSTANT(EMIT_FLAG_VELOCITY);
	BIND_ENUM_CONSTANT(EMIT_FLAG_COLOR);
	BIND_ENUM_CONSTANT(EMIT_FLAG_CUSTOM);

	BIND_CONSTANT(MAX_DRAW_PASSES);

	BIND_ENUM_CONSTANT(TRANSFORM_ALIGN_DISABLED);
	BIND_ENUM_CONSTANT(TRANSFORM_ALIGN_Z_BILLBOARD);
	BIND_ENUM_CONSTANT(TRANSFORM_ALIGN_Y_TO_VELOCITY);
	BIND_ENUM_CONSTANT(TRANSFORM_ALIGN_Z_BILLBOARD_Y_TO_VELOCITY);
}

GPUParticles3D::GPUParticles3D() {
	particles = RS::get_singleton()->particles_create();
	RS::get_singleton()->particles_set_mode(particles, RS::PARTICLES_MODE_3D);
	set_base(particles);
	one_shot = false; // Needed so that set_emitting doesn't access uninitialized values
	set_emitting(true);
	set_one_shot(false);
	set_amount(8);
	set_lifetime(1);
	set_fixed_fps(30);
	set_fractional_delta(true);
	set_interpolate(true);
	set_pre_process_time(0);
	set_explosiveness_ratio(0);
	set_randomness_ratio(0);
	set_trail_length(0.3);
	set_visibility_aabb(AABB(Vector3(-4, -4, -4), Vector3(8, 8, 8)));
	set_use_local_coordinates(true);
	set_draw_passes(1);
	set_draw_order(DRAW_ORDER_INDEX);
	set_speed_scale(1);
	set_collision_base_size(collision_base_size);
	set_transform_align(TRANSFORM_ALIGN_DISABLED);
}

GPUParticles3D::~GPUParticles3D() {
	RS::get_singleton()->free(particles);
}
