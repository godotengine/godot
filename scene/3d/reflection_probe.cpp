/*************************************************************************/
/*  reflection_probe.cpp                                                 */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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
	RS::get_singleton()->reflection_probe_set_intensity(probe, p_intensity);
}

float ReflectionProbe::get_intensity() const {
	return intensity;
}

void ReflectionProbe::set_ambient_mode(AmbientMode p_mode) {
	ambient_mode = p_mode;
	RS::get_singleton()->reflection_probe_set_ambient_mode(probe, RS::ReflectionProbeAmbientMode(p_mode));
	notify_property_list_changed();
}

ReflectionProbe::AmbientMode ReflectionProbe::get_ambient_mode() const {
	return ambient_mode;
}

void ReflectionProbe::set_ambient_color(Color p_ambient) {
	ambient_color = p_ambient;
	RS::get_singleton()->reflection_probe_set_ambient_color(probe, p_ambient);
}

void ReflectionProbe::set_ambient_color_energy(float p_energy) {
	ambient_color_energy = p_energy;
	RS::get_singleton()->reflection_probe_set_ambient_energy(probe, p_energy);
}

float ReflectionProbe::get_ambient_color_energy() const {
	return ambient_color_energy;
}

Color ReflectionProbe::get_ambient_color() const {
	return ambient_color;
}

void ReflectionProbe::set_max_distance(float p_distance) {
	max_distance = p_distance;
	RS::get_singleton()->reflection_probe_set_max_distance(probe, p_distance);
}

float ReflectionProbe::get_max_distance() const {
	return max_distance;
}

void ReflectionProbe::set_lod_threshold(float p_pixels) {
	lod_threshold = p_pixels;
	RS::get_singleton()->reflection_probe_set_lod_threshold(probe, p_pixels);
}

float ReflectionProbe::get_lod_threshold() const {
	return lod_threshold;
}

void ReflectionProbe::set_extents(const Vector3 &p_extents) {
	extents = p_extents;

	for (int i = 0; i < 3; i++) {
		if (extents[i] < 0.01) {
			extents[i] = 0.01;
		}

		if (extents[i] - 0.01 < ABS(origin_offset[i])) {
			origin_offset[i] = SGN(origin_offset[i]) * (extents[i] - 0.01);
		}
	}

	RS::get_singleton()->reflection_probe_set_extents(probe, extents);
	RS::get_singleton()->reflection_probe_set_origin_offset(probe, origin_offset);

	update_gizmos();
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
	RS::get_singleton()->reflection_probe_set_extents(probe, extents);
	RS::get_singleton()->reflection_probe_set_origin_offset(probe, origin_offset);

	update_gizmos();
}

Vector3 ReflectionProbe::get_origin_offset() const {
	return origin_offset;
}

void ReflectionProbe::set_enable_box_projection(bool p_enable) {
	box_projection = p_enable;
	RS::get_singleton()->reflection_probe_set_enable_box_projection(probe, p_enable);
}

bool ReflectionProbe::is_box_projection_enabled() const {
	return box_projection;
}

void ReflectionProbe::set_as_interior(bool p_enable) {
	interior = p_enable;
	RS::get_singleton()->reflection_probe_set_as_interior(probe, interior);
}

bool ReflectionProbe::is_set_as_interior() const {
	return interior;
}

void ReflectionProbe::set_enable_shadows(bool p_enable) {
	enable_shadows = p_enable;
	RS::get_singleton()->reflection_probe_set_enable_shadows(probe, p_enable);
}

bool ReflectionProbe::are_shadows_enabled() const {
	return enable_shadows;
}

void ReflectionProbe::set_cull_mask(uint32_t p_layers) {
	cull_mask = p_layers;
	RS::get_singleton()->reflection_probe_set_cull_mask(probe, p_layers);
}

uint32_t ReflectionProbe::get_cull_mask() const {
	return cull_mask;
}

void ReflectionProbe::set_update_mode(UpdateMode p_mode) {
	update_mode = p_mode;
	RS::get_singleton()->reflection_probe_set_update_mode(probe, RS::ReflectionProbeUpdateMode(p_mode));
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

Vector<Face3> ReflectionProbe::get_faces(uint32_t p_usage_flags) const {
	return Vector<Face3>();
}

void ReflectionProbe::_validate_property(PropertyInfo &property) const {
	if (property.name == "interior/ambient_color" || property.name == "interior/ambient_color_energy") {
		if (ambient_mode != AMBIENT_COLOR) {
			property.usage = PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL;
		}
	}
	VisualInstance3D::_validate_property(property);
}

void ReflectionProbe::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_intensity", "intensity"), &ReflectionProbe::set_intensity);
	ClassDB::bind_method(D_METHOD("get_intensity"), &ReflectionProbe::get_intensity);

	ClassDB::bind_method(D_METHOD("set_ambient_mode", "ambient"), &ReflectionProbe::set_ambient_mode);
	ClassDB::bind_method(D_METHOD("get_ambient_mode"), &ReflectionProbe::get_ambient_mode);

	ClassDB::bind_method(D_METHOD("set_ambient_color", "ambient"), &ReflectionProbe::set_ambient_color);
	ClassDB::bind_method(D_METHOD("get_ambient_color"), &ReflectionProbe::get_ambient_color);

	ClassDB::bind_method(D_METHOD("set_ambient_color_energy", "ambient_energy"), &ReflectionProbe::set_ambient_color_energy);
	ClassDB::bind_method(D_METHOD("get_ambient_color_energy"), &ReflectionProbe::get_ambient_color_energy);

	ClassDB::bind_method(D_METHOD("set_max_distance", "max_distance"), &ReflectionProbe::set_max_distance);
	ClassDB::bind_method(D_METHOD("get_max_distance"), &ReflectionProbe::get_max_distance);

	ClassDB::bind_method(D_METHOD("set_lod_threshold", "ratio"), &ReflectionProbe::set_lod_threshold);
	ClassDB::bind_method(D_METHOD("get_lod_threshold"), &ReflectionProbe::get_lod_threshold);

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
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "intensity", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_intensity", "get_intensity");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "max_distance", PROPERTY_HINT_RANGE, "0,16384,0.1,or_greater,exp"), "set_max_distance", "get_max_distance");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "extents"), "set_extents", "get_extents");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "origin_offset"), "set_origin_offset", "get_origin_offset");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "box_projection"), "set_enable_box_projection", "is_box_projection_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "interior"), "set_as_interior", "is_set_as_interior");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enable_shadows"), "set_enable_shadows", "are_shadows_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "cull_mask", PROPERTY_HINT_LAYERS_3D_RENDER), "set_cull_mask", "get_cull_mask");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "lod_threshold", PROPERTY_HINT_RANGE, "0,1024,0.1"), "set_lod_threshold", "get_lod_threshold");

	ADD_GROUP("Ambient", "ambient_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "ambient_mode", PROPERTY_HINT_ENUM, "Disabled,Environment,Constant Color"), "set_ambient_mode", "get_ambient_mode");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "ambient_color", PROPERTY_HINT_COLOR_NO_ALPHA), "set_ambient_color", "get_ambient_color");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "ambient_color_energy", PROPERTY_HINT_RANGE, "0,16,0.01"), "set_ambient_color_energy", "get_ambient_color_energy");

	BIND_ENUM_CONSTANT(UPDATE_ONCE);
	BIND_ENUM_CONSTANT(UPDATE_ALWAYS);

	BIND_ENUM_CONSTANT(AMBIENT_DISABLED);
	BIND_ENUM_CONSTANT(AMBIENT_ENVIRONMENT);
	BIND_ENUM_CONSTANT(AMBIENT_COLOR);
}

ReflectionProbe::ReflectionProbe() {
	probe = RenderingServer::get_singleton()->reflection_probe_create();
	RS::get_singleton()->instance_set_base(get_instance(), probe);
	set_disable_scale(true);
}

ReflectionProbe::~ReflectionProbe() {
	RS::get_singleton()->free(probe);
}
