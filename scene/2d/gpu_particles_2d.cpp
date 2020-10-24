/*************************************************************************/
/*  gpu_particles_2d.cpp                                                 */
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

#include "gpu_particles_2d.h"

#include "core/os/os.h"
#include "scene/resources/particles_material.h"
#include "scene/scene_string_names.h"

#ifdef TOOLS_ENABLED
#include "core/engine.h"
#endif

void GPUParticles2D::set_emitting(bool p_emitting) {
	RS::get_singleton()->particles_set_emitting(particles, p_emitting);

	if (p_emitting && one_shot) {
		set_process_internal(true);
	} else if (!p_emitting) {
		set_process_internal(false);
	}
}

void GPUParticles2D::set_amount(int p_amount) {
	ERR_FAIL_COND_MSG(p_amount < 1, "Amount of particles cannot be smaller than 1.");
	amount = p_amount;
	RS::get_singleton()->particles_set_amount(particles, amount);
}

void GPUParticles2D::set_lifetime(float p_lifetime) {
	ERR_FAIL_COND_MSG(p_lifetime <= 0, "Particles lifetime must be greater than 0.");
	lifetime = p_lifetime;
	RS::get_singleton()->particles_set_lifetime(particles, lifetime);
}

void GPUParticles2D::set_one_shot(bool p_enable) {
	one_shot = p_enable;
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

void GPUParticles2D::set_pre_process_time(float p_time) {
	pre_process_time = p_time;
	RS::get_singleton()->particles_set_pre_process_time(particles, pre_process_time);
}

void GPUParticles2D::set_explosiveness_ratio(float p_ratio) {
	explosiveness_ratio = p_ratio;
	RS::get_singleton()->particles_set_explosiveness_ratio(particles, explosiveness_ratio);
}

void GPUParticles2D::set_randomness_ratio(float p_ratio) {
	randomness_ratio = p_ratio;
	RS::get_singleton()->particles_set_randomness_ratio(particles, randomness_ratio);
}

void GPUParticles2D::set_visibility_rect(const Rect2 &p_visibility_rect) {
	visibility_rect = p_visibility_rect;
	AABB aabb;
	aabb.position.x = p_visibility_rect.position.x;
	aabb.position.y = p_visibility_rect.position.y;
	aabb.size.x = p_visibility_rect.size.x;
	aabb.size.y = p_visibility_rect.size.y;

	RS::get_singleton()->particles_set_custom_aabb(particles, aabb);

	_change_notify("visibility_rect");
	update();
}

void GPUParticles2D::set_use_local_coordinates(bool p_enable) {
	local_coords = p_enable;
	RS::get_singleton()->particles_set_use_local_coordinates(particles, local_coords);
	set_notify_transform(!p_enable);
	if (!p_enable && is_inside_tree()) {
		_update_particle_emission_transform();
	}
}

void GPUParticles2D::_update_particle_emission_transform() {
	Transform2D xf2d = get_global_transform();
	Transform xf;
	xf.basis.set_axis(0, Vector3(xf2d.get_axis(0).x, xf2d.get_axis(0).y, 0));
	xf.basis.set_axis(1, Vector3(xf2d.get_axis(1).x, xf2d.get_axis(1).y, 0));
	xf.set_origin(Vector3(xf2d.get_origin().x, xf2d.get_origin().y, 0));

	RS::get_singleton()->particles_set_emission_transform(particles, xf);
}

void GPUParticles2D::set_process_material(const Ref<Material> &p_material) {
	process_material = p_material;
	Ref<ParticlesMaterial> pm = p_material;
	if (pm.is_valid() && !pm->get_flag(ParticlesMaterial::FLAG_DISABLE_Z) && pm->get_gravity() == Vector3(0, -9.8, 0)) {
		// Likely a new (3D) material, modify it to match 2D space
		pm->set_flag(ParticlesMaterial::FLAG_DISABLE_Z, true);
		pm->set_gravity(Vector3(0, 98, 0));
	}
	RID material_rid;
	if (process_material.is_valid()) {
		material_rid = process_material->get_rid();
	}
	RS::get_singleton()->particles_set_process_material(particles, material_rid);

	update_configuration_warning();
}

void GPUParticles2D::set_speed_scale(float p_scale) {
	speed_scale = p_scale;
	RS::get_singleton()->particles_set_speed_scale(particles, p_scale);
}

bool GPUParticles2D::is_emitting() const {
	return RS::get_singleton()->particles_get_emitting(particles);
}

int GPUParticles2D::get_amount() const {
	return amount;
}

float GPUParticles2D::get_lifetime() const {
	return lifetime;
}

bool GPUParticles2D::get_one_shot() const {
	return one_shot;
}

float GPUParticles2D::get_pre_process_time() const {
	return pre_process_time;
}

float GPUParticles2D::get_explosiveness_ratio() const {
	return explosiveness_ratio;
}

float GPUParticles2D::get_randomness_ratio() const {
	return randomness_ratio;
}

Rect2 GPUParticles2D::get_visibility_rect() const {
	return visibility_rect;
}

bool GPUParticles2D::get_use_local_coordinates() const {
	return local_coords;
}

Ref<Material> GPUParticles2D::get_process_material() const {
	return process_material;
}

float GPUParticles2D::get_speed_scale() const {
	return speed_scale;
}

void GPUParticles2D::set_draw_order(DrawOrder p_order) {
	draw_order = p_order;
	RS::get_singleton()->particles_set_draw_order(particles, RS::ParticlesDrawOrder(p_order));
}

GPUParticles2D::DrawOrder GPUParticles2D::get_draw_order() const {
	return draw_order;
}

void GPUParticles2D::set_fixed_fps(int p_count) {
	fixed_fps = p_count;
	RS::get_singleton()->particles_set_fixed_fps(particles, p_count);
}

int GPUParticles2D::get_fixed_fps() const {
	return fixed_fps;
}

void GPUParticles2D::set_fractional_delta(bool p_enable) {
	fractional_delta = p_enable;
	RS::get_singleton()->particles_set_fractional_delta(particles, p_enable);
}

bool GPUParticles2D::get_fractional_delta() const {
	return fractional_delta;
}

String GPUParticles2D::get_configuration_warning() const {
	if (RenderingServer::get_singleton()->is_low_end()) {
		return TTR("GPU-based particles are not supported by the GLES2 video driver.\nUse the CPUParticles2D node instead. You can use the \"Convert to CPUParticles2D\" option for this purpose.");
	}

	String warnings = Node2D::get_configuration_warning();

	if (process_material.is_null()) {
		if (warnings != String()) {
			warnings += "\n";
		}
		warnings += "- " + TTR("A material to process the particles is not assigned, so no behavior is imprinted.");
	} else {
		CanvasItemMaterial *mat = Object::cast_to<CanvasItemMaterial>(get_material().ptr());

		if (get_material().is_null() || (mat && !mat->get_particles_animation())) {
			const ParticlesMaterial *process = Object::cast_to<ParticlesMaterial>(process_material.ptr());
			if (process &&
					(process->get_param(ParticlesMaterial::PARAM_ANIM_SPEED) != 0.0 || process->get_param(ParticlesMaterial::PARAM_ANIM_OFFSET) != 0.0 ||
							process->get_param_texture(ParticlesMaterial::PARAM_ANIM_SPEED).is_valid() || process->get_param_texture(ParticlesMaterial::PARAM_ANIM_OFFSET).is_valid())) {
				if (warnings != String()) {
					warnings += "\n";
				}
				warnings += "- " + TTR("Particles2D animation requires the usage of a CanvasItemMaterial with \"Particles Animation\" enabled.");
			}
		}
	}

	return warnings;
}

Rect2 GPUParticles2D::capture_rect() const {
	AABB aabb = RS::get_singleton()->particles_get_current_aabb(particles);
	Rect2 r;
	r.position.x = aabb.position.x;
	r.position.y = aabb.position.y;
	r.size.x = aabb.size.x;
	r.size.y = aabb.size.y;
	return r;
}

void GPUParticles2D::set_texture(const Ref<Texture2D> &p_texture) {
	texture = p_texture;
	update();
}

Ref<Texture2D> GPUParticles2D::get_texture() const {
	return texture;
}

void GPUParticles2D::_validate_property(PropertyInfo &property) const {
}

void GPUParticles2D::restart() {
	RS::get_singleton()->particles_restart(particles);
	RS::get_singleton()->particles_set_emitting(particles, true);
}

void GPUParticles2D::_notification(int p_what) {
	if (p_what == NOTIFICATION_DRAW) {
		RID texture_rid;
		if (texture.is_valid()) {
			texture_rid = texture->get_rid();
		}

		RS::get_singleton()->canvas_item_add_particles(get_canvas_item(), particles, texture_rid);

#ifdef TOOLS_ENABLED
		if (Engine::get_singleton()->is_editor_hint() && (this == get_tree()->get_edited_scene_root() || get_tree()->get_edited_scene_root()->is_a_parent_of(this))) {
			draw_rect(visibility_rect, Color(0, 0.7, 0.9, 0.4), false);
		}
#endif
	}

	if (p_what == NOTIFICATION_PAUSED || p_what == NOTIFICATION_UNPAUSED) {
		if (can_process()) {
			RS::get_singleton()->particles_set_speed_scale(particles, speed_scale);
		} else {
			RS::get_singleton()->particles_set_speed_scale(particles, 0);
		}
	}

	if (p_what == NOTIFICATION_TRANSFORM_CHANGED) {
		_update_particle_emission_transform();
	}

	if (p_what == NOTIFICATION_INTERNAL_PROCESS) {
		if (one_shot && !is_emitting()) {
			_change_notify();
			set_process_internal(false);
		}
	}
}

void GPUParticles2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_emitting", "emitting"), &GPUParticles2D::set_emitting);
	ClassDB::bind_method(D_METHOD("set_amount", "amount"), &GPUParticles2D::set_amount);
	ClassDB::bind_method(D_METHOD("set_lifetime", "secs"), &GPUParticles2D::set_lifetime);
	ClassDB::bind_method(D_METHOD("set_one_shot", "secs"), &GPUParticles2D::set_one_shot);
	ClassDB::bind_method(D_METHOD("set_pre_process_time", "secs"), &GPUParticles2D::set_pre_process_time);
	ClassDB::bind_method(D_METHOD("set_explosiveness_ratio", "ratio"), &GPUParticles2D::set_explosiveness_ratio);
	ClassDB::bind_method(D_METHOD("set_randomness_ratio", "ratio"), &GPUParticles2D::set_randomness_ratio);
	ClassDB::bind_method(D_METHOD("set_visibility_rect", "visibility_rect"), &GPUParticles2D::set_visibility_rect);
	ClassDB::bind_method(D_METHOD("set_use_local_coordinates", "enable"), &GPUParticles2D::set_use_local_coordinates);
	ClassDB::bind_method(D_METHOD("set_fixed_fps", "fps"), &GPUParticles2D::set_fixed_fps);
	ClassDB::bind_method(D_METHOD("set_fractional_delta", "enable"), &GPUParticles2D::set_fractional_delta);
	ClassDB::bind_method(D_METHOD("set_process_material", "material"), &GPUParticles2D::set_process_material);
	ClassDB::bind_method(D_METHOD("set_speed_scale", "scale"), &GPUParticles2D::set_speed_scale);

	ClassDB::bind_method(D_METHOD("is_emitting"), &GPUParticles2D::is_emitting);
	ClassDB::bind_method(D_METHOD("get_amount"), &GPUParticles2D::get_amount);
	ClassDB::bind_method(D_METHOD("get_lifetime"), &GPUParticles2D::get_lifetime);
	ClassDB::bind_method(D_METHOD("get_one_shot"), &GPUParticles2D::get_one_shot);
	ClassDB::bind_method(D_METHOD("get_pre_process_time"), &GPUParticles2D::get_pre_process_time);
	ClassDB::bind_method(D_METHOD("get_explosiveness_ratio"), &GPUParticles2D::get_explosiveness_ratio);
	ClassDB::bind_method(D_METHOD("get_randomness_ratio"), &GPUParticles2D::get_randomness_ratio);
	ClassDB::bind_method(D_METHOD("get_visibility_rect"), &GPUParticles2D::get_visibility_rect);
	ClassDB::bind_method(D_METHOD("get_use_local_coordinates"), &GPUParticles2D::get_use_local_coordinates);
	ClassDB::bind_method(D_METHOD("get_fixed_fps"), &GPUParticles2D::get_fixed_fps);
	ClassDB::bind_method(D_METHOD("get_fractional_delta"), &GPUParticles2D::get_fractional_delta);
	ClassDB::bind_method(D_METHOD("get_process_material"), &GPUParticles2D::get_process_material);
	ClassDB::bind_method(D_METHOD("get_speed_scale"), &GPUParticles2D::get_speed_scale);

	ClassDB::bind_method(D_METHOD("set_draw_order", "order"), &GPUParticles2D::set_draw_order);
	ClassDB::bind_method(D_METHOD("get_draw_order"), &GPUParticles2D::get_draw_order);

	ClassDB::bind_method(D_METHOD("set_texture", "texture"), &GPUParticles2D::set_texture);
	ClassDB::bind_method(D_METHOD("get_texture"), &GPUParticles2D::get_texture);

	ClassDB::bind_method(D_METHOD("capture_rect"), &GPUParticles2D::capture_rect);

	ClassDB::bind_method(D_METHOD("restart"), &GPUParticles2D::restart);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "emitting"), "set_emitting", "is_emitting");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "amount", PROPERTY_HINT_EXP_RANGE, "1,1000000,1"), "set_amount", "get_amount");
	ADD_GROUP("Time", "");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "lifetime", PROPERTY_HINT_RANGE, "0.01,600.0,0.01,or_greater"), "set_lifetime", "get_lifetime");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "one_shot"), "set_one_shot", "get_one_shot");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "preprocess", PROPERTY_HINT_RANGE, "0.00,600.0,0.01"), "set_pre_process_time", "get_pre_process_time");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "speed_scale", PROPERTY_HINT_RANGE, "0,64,0.01"), "set_speed_scale", "get_speed_scale");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "explosiveness", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_explosiveness_ratio", "get_explosiveness_ratio");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "randomness", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_randomness_ratio", "get_randomness_ratio");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "fixed_fps", PROPERTY_HINT_RANGE, "0,1000,1"), "set_fixed_fps", "get_fixed_fps");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "fract_delta"), "set_fractional_delta", "get_fractional_delta");
	ADD_GROUP("Drawing", "");
	ADD_PROPERTY(PropertyInfo(Variant::RECT2, "visibility_rect"), "set_visibility_rect", "get_visibility_rect");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "local_coords"), "set_use_local_coordinates", "get_use_local_coordinates");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "draw_order", PROPERTY_HINT_ENUM, "Index,Lifetime"), "set_draw_order", "get_draw_order");
	ADD_GROUP("Process Material", "process_");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "process_material", PROPERTY_HINT_RESOURCE_TYPE, "ShaderMaterial,ParticlesMaterial"), "set_process_material", "get_process_material");
	ADD_GROUP("Textures", "");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture2D"), "set_texture", "get_texture");

	BIND_ENUM_CONSTANT(DRAW_ORDER_INDEX);
	BIND_ENUM_CONSTANT(DRAW_ORDER_LIFETIME);
}

GPUParticles2D::GPUParticles2D() {
	particles = RS::get_singleton()->particles_create();

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
	set_visibility_rect(Rect2(Vector2(-100, -100), Vector2(200, 200)));
	set_use_local_coordinates(true);
	set_draw_order(DRAW_ORDER_INDEX);
	set_speed_scale(1);
}

GPUParticles2D::~GPUParticles2D() {
	RS::get_singleton()->free(particles);
}
