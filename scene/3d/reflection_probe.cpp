/*************************************************************************/
/*  reflection_probe.cpp                                                 */
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

#include "reflection_probe.h"

void ReflectionProbe::set_intensity(float p_intensity) {
	intensity = p_intensity;
	VS::get_singleton()->reflection_probe_set_intensity(probe, p_intensity);
}

float ReflectionProbe::get_intensity() const {
	return intensity;
}

void ReflectionProbe::set_interior_ambient(Color p_ambient) {
	interior_ambient = p_ambient;
	VS::get_singleton()->reflection_probe_set_interior_ambient(probe, p_ambient);
}

void ReflectionProbe::set_interior_ambient_energy(float p_energy) {
	interior_ambient_energy = p_energy;
	VS::get_singleton()->reflection_probe_set_interior_ambient_energy(probe, p_energy);
}

float ReflectionProbe::get_interior_ambient_energy() const {
	return interior_ambient_energy;
}

Color ReflectionProbe::get_interior_ambient() const {
	return interior_ambient;
}

void ReflectionProbe::set_interior_ambient_probe_contribution(float p_contribution) {
	interior_ambient_probe_contribution = p_contribution;
	VS::get_singleton()->reflection_probe_set_interior_ambient_probe_contribution(probe, p_contribution);
}

float ReflectionProbe::get_interior_ambient_probe_contribution() const {
	return interior_ambient_probe_contribution;
}

void ReflectionProbe::set_max_distance(float p_distance) {
	max_distance = p_distance;
	VS::get_singleton()->reflection_probe_set_max_distance(probe, p_distance);
}
float ReflectionProbe::get_max_distance() const {
	return max_distance;
}

void ReflectionProbe::set_extents(const Vector3 &p_extents) {
	extents = p_extents;

	for (int i = 0; i < 3; i++) {
		if (extents[i] < 0.01) {
			extents[i] = 0.01;
		}

		if (extents[i] - 0.01 < ABS(origin_offset[i])) {
			origin_offset[i] = SGN(origin_offset[i]) * (extents[i] - 0.01);
			_change_notify("origin_offset");
		}
	}

	VS::get_singleton()->reflection_probe_set_extents(probe, extents);
	VS::get_singleton()->reflection_probe_set_origin_offset(probe, origin_offset);
	_change_notify("extents");
	update_gizmo();
}
Vector3 ReflectionProbe::get_extents() const {
	return extents;
}

void ReflectionProbe::set_origin_offset(const Vector3 &p_extents) {
	origin_offset = p_extents;

	for (int i = 0; i < 3; i++) {
		if (extents[i] - 0.01 < ABS(origin_offset[i])) {
			origin_offset[i] = SGN(origin_offset[i]) * (extents[i] - 0.01);
		}
	}
	VS::get_singleton()->reflection_probe_set_extents(probe, extents);
	VS::get_singleton()->reflection_probe_set_origin_offset(probe, origin_offset);

	_change_notify("origin_offset");
	update_gizmo();
}
Vector3 ReflectionProbe::get_origin_offset() const {
	return origin_offset;
}

void ReflectionProbe::set_enable_box_projection(bool p_enable) {
	box_projection = p_enable;
	VS::get_singleton()->reflection_probe_set_enable_box_projection(probe, p_enable);
}
bool ReflectionProbe::is_box_projection_enabled() const {
	return box_projection;
}

void ReflectionProbe::set_as_interior(bool p_enable) {
	interior = p_enable;
	VS::get_singleton()->reflection_probe_set_as_interior(probe, interior);
	_change_notify();
}

bool ReflectionProbe::is_set_as_interior() const {
	return interior;
}

void ReflectionProbe::set_enable_shadows(bool p_enable) {
	enable_shadows = p_enable;
	VS::get_singleton()->reflection_probe_set_enable_shadows(probe, p_enable);
}
bool ReflectionProbe::are_shadows_enabled() const {
	return enable_shadows;
}

void ReflectionProbe::set_cull_mask(uint32_t p_layers) {
	cull_mask = p_layers;
	VS::get_singleton()->reflection_probe_set_cull_mask(probe, p_layers);
}
uint32_t ReflectionProbe::get_cull_mask() const {
	return cull_mask;
}

void ReflectionProbe::set_update_mode(UpdateMode p_mode) {
	update_mode = p_mode;
	VS::get_singleton()->reflection_probe_set_update_mode(probe, VS::ReflectionProbeUpdateMode(p_mode));
}

ReflectionProbe::UpdateMode ReflectionProbe::get_update_mode() const {
	return update_mode;
}

AABB ReflectionProbe::get_aabb() const {
	AABB aabb;
	aabb.position = -origin_offset;
	aabb.size = origin_offset + extents;
	return aabb;
}
PoolVector<Face3> ReflectionProbe::get_faces(uint32_t p_usage_flags) const {
	return PoolVector<Face3>();
}

void ReflectionProbe::_validate_property(PropertyInfo &property) const {
	if (property.name == "interior/ambient_color" || property.name == "interior/ambient_energy" || property.name == "interior/ambient_contrib") {
		if (!interior) {
			property.usage = PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL;
		}
	}
}

void ReflectionProbe::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_intensity", "intensity"), &ReflectionProbe::set_intensity);
	ClassDB::bind_method(D_METHOD("get_intensity"), &ReflectionProbe::get_intensity);

	ClassDB::bind_method(D_METHOD("set_interior_ambient", "ambient"), &ReflectionProbe::set_interior_ambient);
	ClassDB::bind_method(D_METHOD("get_interior_ambient"), &ReflectionProbe::get_interior_ambient);

	ClassDB::bind_method(D_METHOD("set_interior_ambient_energy", "ambient_energy"), &ReflectionProbe::set_interior_ambient_energy);
	ClassDB::bind_method(D_METHOD("get_interior_ambient_energy"), &ReflectionProbe::get_interior_ambient_energy);

	ClassDB::bind_method(D_METHOD("set_interior_ambient_probe_contribution", "ambient_probe_contribution"), &ReflectionProbe::set_interior_ambient_probe_contribution);
	ClassDB::bind_method(D_METHOD("get_interior_ambient_probe_contribution"), &ReflectionProbe::get_interior_ambient_probe_contribution);

	ClassDB::bind_method(D_METHOD("set_max_distance", "max_distance"), &ReflectionProbe::set_max_distance);
	ClassDB::bind_method(D_METHOD("get_max_distance"), &ReflectionProbe::get_max_distance);

	ClassDB::bind_method(D_METHOD("set_extents", "extents"), &ReflectionProbe::set_extents);
	ClassDB::bind_method(D_METHOD("get_extents"), &ReflectionProbe::get_extents);

	ClassDB::bind_method(D_METHOD("set_origin_offset", "origin_offset"), &ReflectionProbe::set_origin_offset);
	ClassDB::bind_method(D_METHOD("get_origin_offset"), &ReflectionProbe::get_origin_offset);

	ClassDB::bind_method(D_METHOD("set_as_interior", "enable"), &ReflectionProbe::set_as_interior);
	ClassDB::bind_method(D_METHOD("is_set_as_interior"), &ReflectionProbe::is_set_as_interior);

	ClassDB::bind_method(D_METHOD("set_enable_box_projection", "enable"), &ReflectionProbe::set_enable_box_projection);
	ClassDB::bind_method(D_METHOD("is_box_projection_enabled"), &ReflectionProbe::is_box_projection_enabled);

	ClassDB::bind_method(D_METHOD("set_enable_shadows", "enable"), &ReflectionProbe::set_enable_shadows);
	ClassDB::bind_method(D_METHOD("are_shadows_enabled"), &ReflectionProbe::are_shadows_enabled);

	ClassDB::bind_method(D_METHOD("set_cull_mask", "layers"), &ReflectionProbe::set_cull_mask);
	ClassDB::bind_method(D_METHOD("get_cull_mask"), &ReflectionProbe::get_cull_mask);

	ClassDB::bind_method(D_METHOD("set_update_mode", "mode"), &ReflectionProbe::set_update_mode);
	ClassDB::bind_method(D_METHOD("get_update_mode"), &ReflectionProbe::get_update_mode);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "update_mode", PROPERTY_HINT_ENUM, "Once (Fast),Always (Slow)"), "set_update_mode", "get_update_mode");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "intensity", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_intensity", "get_intensity");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "max_distance", PROPERTY_HINT_EXP_RANGE, "0,16384,0.1,or_greater"), "set_max_distance", "get_max_distance");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "extents"), "set_extents", "get_extents");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "origin_offset"), "set_origin_offset", "get_origin_offset");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "box_projection"), "set_enable_box_projection", "is_box_projection_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enable_shadows"), "set_enable_shadows", "are_shadows_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "cull_mask", PROPERTY_HINT_LAYERS_3D_RENDER), "set_cull_mask", "get_cull_mask");

	ADD_GROUP("Interior", "interior_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "interior_enable"), "set_as_interior", "is_set_as_interior");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "interior_ambient_color", PROPERTY_HINT_COLOR_NO_ALPHA), "set_interior_ambient", "get_interior_ambient");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "interior_ambient_energy", PROPERTY_HINT_RANGE, "0,16,0.01"), "set_interior_ambient_energy", "get_interior_ambient_energy");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "interior_ambient_contrib", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_interior_ambient_probe_contribution", "get_interior_ambient_probe_contribution");

	BIND_ENUM_CONSTANT(UPDATE_ONCE);
	BIND_ENUM_CONSTANT(UPDATE_ALWAYS);
}

ReflectionProbe::ReflectionProbe() {
	intensity = 1.0;
	interior_ambient = Color(0, 0, 0);
	interior_ambient_probe_contribution = 0;
	interior_ambient_energy = 1.0;
	max_distance = 0;
	extents = Vector3(1, 1, 1);
	origin_offset = Vector3(0, 0, 0);
	box_projection = false;
	interior = false;
	enable_shadows = false;
	cull_mask = (1 << 20) - 1;
	update_mode = UPDATE_ONCE;

	probe = RID_PRIME(VisualServer::get_singleton()->reflection_probe_create());
	VS::get_singleton()->instance_set_base(get_instance(), probe);
	set_disable_scale(true);
}

ReflectionProbe::~ReflectionProbe() {
	VS::get_singleton()->free(probe);
}
