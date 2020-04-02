/*************************************************************************/
/*  gpu_particles_3d.cpp                                                 */
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

#include "gpu_particles_3d.h"

#include "core/os/os.h"
#include "scene/resources/particles_material.h"

#include "servers/rendering_server.h"

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
void GPUParticles3D::set_lifetime(float p_lifetime) {

	ERR_FAIL_COND_MSG(p_lifetime <= 0, "Particles lifetime must be greater than 0.");
	lifetime = p_lifetime;
	RS::get_singleton()->particles_set_lifetime(particles, lifetime);
}

void GPUParticles3D::set_one_shot(bool p_one_shot) {

	one_shot = p_one_shot;
	RS::get_singleton()->particles_set_one_shot(particles, one_shot);

	if (is_emitting()) {

		set_process_internal(true);
		if (!one_shot)
			RenderingServer::get_singleton()->particles_restart(particles);
	}

	if (!one_shot)
		set_process_internal(false);
}

void GPUParticles3D::set_pre_process_time(float p_time) {

	pre_process_time = p_time;
	RS::get_singleton()->particles_set_pre_process_time(particles, pre_process_time);
}
void GPUParticles3D::set_explosiveness_ratio(float p_ratio) {

	explosiveness_ratio = p_ratio;
	RS::get_singleton()->particles_set_explosiveness_ratio(particles, explosiveness_ratio);
}
void GPUParticles3D::set_randomness_ratio(float p_ratio) {

	randomness_ratio = p_ratio;
	RS::get_singleton()->particles_set_randomness_ratio(particles, randomness_ratio);
}
void GPUParticles3D::set_visibility_aabb(const AABB &p_aabb) {

	visibility_aabb = p_aabb;
	RS::get_singleton()->particles_set_custom_aabb(particles, visibility_aabb);
	update_gizmo();
	_change_notify("visibility_aabb");
}
void GPUParticles3D::set_use_local_coordinates(bool p_enable) {

	local_coords = p_enable;
	RS::get_singleton()->particles_set_use_local_coordinates(particles, local_coords);
}
void GPUParticles3D::set_process_material(const Ref<Material> &p_material) {

	process_material = p_material;
	RID material_rid;
	if (process_material.is_valid())
		material_rid = process_material->get_rid();
	RS::get_singleton()->particles_set_process_material(particles, material_rid);

	update_configuration_warning();
}

void GPUParticles3D::set_speed_scale(float p_scale) {

	speed_scale = p_scale;
	RS::get_singleton()->particles_set_speed_scale(particles, p_scale);
}

bool GPUParticles3D::is_emitting() const {

	return RS::get_singleton()->particles_get_emitting(particles);
}
int GPUParticles3D::get_amount() const {

	return amount;
}
float GPUParticles3D::get_lifetime() const {

	return lifetime;
}
bool GPUParticles3D::get_one_shot() const {

	return one_shot;
}

float GPUParticles3D::get_pre_process_time() const {

	return pre_process_time;
}
float GPUParticles3D::get_explosiveness_ratio() const {

	return explosiveness_ratio;
}
float GPUParticles3D::get_randomness_ratio() const {

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

float GPUParticles3D::get_speed_scale() const {

	return speed_scale;
}

void GPUParticles3D::set_draw_order(DrawOrder p_order) {

	draw_order = p_order;
	RS::get_singleton()->particles_set_draw_order(particles, RS::ParticlesDrawOrder(p_order));
}

GPUParticles3D::DrawOrder GPUParticles3D::get_draw_order() const {

	return draw_order;
}

void GPUParticles3D::set_draw_passes(int p_count) {

	ERR_FAIL_COND(p_count < 1);
	draw_passes.resize(p_count);
	RS::get_singleton()->particles_set_draw_passes(particles, p_count);
	_change_notify();
}
int GPUParticles3D::get_draw_passes() const {

	return draw_passes.size();
}

void GPUParticles3D::set_draw_pass_mesh(int p_pass, const Ref<Mesh> &p_mesh) {

	ERR_FAIL_INDEX(p_pass, draw_passes.size());

	draw_passes.write[p_pass] = p_mesh;

	RID mesh_rid;
	if (p_mesh.is_valid())
		mesh_rid = p_mesh->get_rid();

	RS::get_singleton()->particles_set_draw_pass_mesh(particles, p_pass, mesh_rid);

	update_configuration_warning();
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

String GPUParticles3D::get_configuration_warning() const {

	if (RenderingServer::get_singleton()->is_low_end()) {
		return TTR("GPU-based particles are not supported by the GLES2 video driver.\nUse the CPUParticles3D node instead. You can use the \"Convert to CPUParticles3D\" option for this purpose.");
	}

	String warnings;

	bool meshes_found = false;
	bool anim_material_found = false;

	for (int i = 0; i < draw_passes.size(); i++) {
		if (draw_passes[i].is_valid()) {
			meshes_found = true;
			for (int j = 0; j < draw_passes[i]->get_surface_count(); j++) {
				anim_material_found = Object::cast_to<ShaderMaterial>(draw_passes[i]->surface_get_material(j).ptr()) != nullptr;
				StandardMaterial3D *spat = Object::cast_to<StandardMaterial3D>(draw_passes[i]->surface_get_material(j).ptr());
				anim_material_found = anim_material_found || (spat && spat->get_billboard_mode() == StandardMaterial3D::BILLBOARD_PARTICLES);
			}
			if (anim_material_found) break;
		}
	}

	anim_material_found = anim_material_found || Object::cast_to<ShaderMaterial>(get_material_override().ptr()) != nullptr;
	StandardMaterial3D *spat = Object::cast_to<StandardMaterial3D>(get_material_override().ptr());
	anim_material_found = anim_material_found || (spat && spat->get_billboard_mode() == StandardMaterial3D::BILLBOARD_PARTICLES);

	if (!meshes_found) {
		if (warnings != String())
			warnings += "\n";
		warnings += "- " + TTR("Nothing is visible because meshes have not been assigned to draw passes.");
	}

	if (process_material.is_null()) {
		if (warnings != String())
			warnings += "\n";
		warnings += "- " + TTR("A material to process the particles is not assigned, so no behavior is imprinted.");
	} else {
		const ParticlesMaterial *process = Object::cast_to<ParticlesMaterial>(process_material.ptr());
		if (!anim_material_found && process &&
				(process->get_param(ParticlesMaterial::PARAM_ANIM_SPEED) != 0.0 || process->get_param(ParticlesMaterial::PARAM_ANIM_OFFSET) != 0.0 ||
						process->get_param_texture(ParticlesMaterial::PARAM_ANIM_SPEED).is_valid() || process->get_param_texture(ParticlesMaterial::PARAM_ANIM_OFFSET).is_valid())) {
			if (warnings != String())
				warnings += "\n";
			warnings += "- " + TTR("Particles animation requires the usage of a StandardMaterial3D whose Billboard Mode is set to \"Particle Billboard\".");
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
			property.usage = 0;
			return;
		}
	}
}

void GPUParticles3D::_notification(int p_what) {

	if (p_what == NOTIFICATION_PAUSED || p_what == NOTIFICATION_UNPAUSED) {
		if (can_process()) {
			RS::get_singleton()->particles_set_speed_scale(particles, speed_scale);
		} else {

			RS::get_singleton()->particles_set_speed_scale(particles, 0);
		}
	}

	// Use internal process when emitting and one_shot are on so that when
	// the shot ends the editor can properly update
	if (p_what == NOTIFICATION_INTERNAL_PROCESS) {

		if (one_shot && !is_emitting()) {
			_change_notify();
			set_process_internal(false);
		}
	}

	if (p_what == NOTIFICATION_VISIBILITY_CHANGED) {
		// make sure particles are updated before rendering occurs if they were active before
		if (is_visible_in_tree() && !RS::get_singleton()->particles_is_inactive(particles)) {
			RS::get_singleton()->particles_request_process(particles);
		}
	}
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
	ClassDB::bind_method(D_METHOD("set_process_material", "material"), &GPUParticles3D::set_process_material);
	ClassDB::bind_method(D_METHOD("set_speed_scale", "scale"), &GPUParticles3D::set_speed_scale);

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
	ClassDB::bind_method(D_METHOD("get_process_material"), &GPUParticles3D::get_process_material);
	ClassDB::bind_method(D_METHOD("get_speed_scale"), &GPUParticles3D::get_speed_scale);

	ClassDB::bind_method(D_METHOD("set_draw_order", "order"), &GPUParticles3D::set_draw_order);

	ClassDB::bind_method(D_METHOD("get_draw_order"), &GPUParticles3D::get_draw_order);

	ClassDB::bind_method(D_METHOD("set_draw_passes", "passes"), &GPUParticles3D::set_draw_passes);
	ClassDB::bind_method(D_METHOD("set_draw_pass_mesh", "pass", "mesh"), &GPUParticles3D::set_draw_pass_mesh);

	ClassDB::bind_method(D_METHOD("get_draw_passes"), &GPUParticles3D::get_draw_passes);
	ClassDB::bind_method(D_METHOD("get_draw_pass_mesh", "pass"), &GPUParticles3D::get_draw_pass_mesh);

	ClassDB::bind_method(D_METHOD("restart"), &GPUParticles3D::restart);
	ClassDB::bind_method(D_METHOD("capture_aabb"), &GPUParticles3D::capture_aabb);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "emitting"), "set_emitting", "is_emitting");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "amount", PROPERTY_HINT_EXP_RANGE, "1,1000000,1"), "set_amount", "get_amount");
	ADD_GROUP("Time", "");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "lifetime", PROPERTY_HINT_EXP_RANGE, "0.01,600.0,0.01,or_greater"), "set_lifetime", "get_lifetime");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "one_shot"), "set_one_shot", "get_one_shot");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "preprocess", PROPERTY_HINT_EXP_RANGE, "0.00,600.0,0.01"), "set_pre_process_time", "get_pre_process_time");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "speed_scale", PROPERTY_HINT_RANGE, "0,64,0.01"), "set_speed_scale", "get_speed_scale");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "explosiveness", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_explosiveness_ratio", "get_explosiveness_ratio");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "randomness", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_randomness_ratio", "get_randomness_ratio");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "fixed_fps", PROPERTY_HINT_RANGE, "0,1000,1"), "set_fixed_fps", "get_fixed_fps");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "fract_delta"), "set_fractional_delta", "get_fractional_delta");
	ADD_GROUP("Drawing", "");
	ADD_PROPERTY(PropertyInfo(Variant::AABB, "visibility_aabb"), "set_visibility_aabb", "get_visibility_aabb");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "local_coords"), "set_use_local_coordinates", "get_use_local_coordinates");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "draw_order", PROPERTY_HINT_ENUM, "Index,Lifetime,View Depth"), "set_draw_order", "get_draw_order");
	ADD_GROUP("Process Material", "");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "process_material", PROPERTY_HINT_RESOURCE_TYPE, "ShaderMaterial,ParticlesMaterial"), "set_process_material", "get_process_material");
	ADD_GROUP("Draw Passes", "draw_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "draw_passes", PROPERTY_HINT_RANGE, "0," + itos(MAX_DRAW_PASSES) + ",1"), "set_draw_passes", "get_draw_passes");
	for (int i = 0; i < MAX_DRAW_PASSES; i++) {

		ADD_PROPERTYI(PropertyInfo(Variant::OBJECT, "draw_pass_" + itos(i + 1), PROPERTY_HINT_RESOURCE_TYPE, "Mesh"), "set_draw_pass_mesh", "get_draw_pass_mesh", i);
	}

	BIND_ENUM_CONSTANT(DRAW_ORDER_INDEX);
	BIND_ENUM_CONSTANT(DRAW_ORDER_LIFETIME);
	BIND_ENUM_CONSTANT(DRAW_ORDER_VIEW_DEPTH);

	BIND_CONSTANT(MAX_DRAW_PASSES);
}

GPUParticles3D::GPUParticles3D() {

	particles = RS::get_singleton()->particles_create();
	set_base(particles);
	one_shot = false; // Needed so that set_emitting doesn't access uninitialized values
	set_emitting(true);
	set_one_shot(false);
	set_amount(8);
	set_lifetime(1);
	set_fixed_fps(0);
	set_fractional_delta(true);
	set_pre_process_time(0);
	set_explosiveness_ratio(0);
	set_randomness_ratio(0);
	set_visibility_aabb(AABB(Vector3(-4, -4, -4), Vector3(8, 8, 8)));
	set_use_local_coordinates(true);
	set_draw_passes(1);
	set_draw_order(DRAW_ORDER_INDEX);
	set_speed_scale(1);
}

GPUParticles3D::~GPUParticles3D() {

	RS::get_singleton()->free(particles);
}
