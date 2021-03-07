/*************************************************************************/
/*  decal.cpp                                                            */
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

#include "decal.h"

void Decal::set_extents(const Vector3 &p_extents) {
	extents = p_extents;
	RS::get_singleton()->decal_set_extents(decal, p_extents);
	update_gizmo();
}

Vector3 Decal::get_extents() const {
	return extents;
}

void Decal::set_texture(DecalTexture p_type, const Ref<Texture2D> &p_texture) {
	ERR_FAIL_INDEX(p_type, TEXTURE_MAX);
	textures[p_type] = p_texture;
	RID texture_rid = p_texture.is_valid() ? p_texture->get_rid() : RID();
	RS::get_singleton()->decal_set_texture(decal, RS::DecalTexture(p_type), texture_rid);
}

Ref<Texture2D> Decal::get_texture(DecalTexture p_type) const {
	ERR_FAIL_INDEX_V(p_type, TEXTURE_MAX, Ref<Texture2D>());
	return textures[p_type];
}

void Decal::set_emission_energy(float p_energy) {
	emission_energy = p_energy;
	RS::get_singleton()->decal_set_emission_energy(decal, emission_energy);
}

float Decal::get_emission_energy() const {
	return emission_energy;
}

void Decal::set_albedo_mix(float p_mix) {
	albedo_mix = p_mix;
	RS::get_singleton()->decal_set_albedo_mix(decal, albedo_mix);
}

float Decal::get_albedo_mix() const {
	return albedo_mix;
}

void Decal::set_upper_fade(float p_fade) {
	upper_fade = p_fade;
	RS::get_singleton()->decal_set_fade(decal, upper_fade, lower_fade);
}

float Decal::get_upper_fade() const {
	return upper_fade;
}

void Decal::set_lower_fade(float p_fade) {
	lower_fade = p_fade;
	RS::get_singleton()->decal_set_fade(decal, upper_fade, lower_fade);
}

float Decal::get_lower_fade() const {
	return lower_fade;
}

void Decal::set_normal_fade(float p_fade) {
	normal_fade = p_fade;
	RS::get_singleton()->decal_set_normal_fade(decal, normal_fade);
}

float Decal::get_normal_fade() const {
	return normal_fade;
}

void Decal::set_modulate(Color p_modulate) {
	modulate = p_modulate;
	RS::get_singleton()->decal_set_modulate(decal, p_modulate);
}

Color Decal::get_modulate() const {
	return modulate;
}

void Decal::set_enable_distance_fade(bool p_enable) {
	distance_fade_enabled = p_enable;
	RS::get_singleton()->decal_set_distance_fade(decal, distance_fade_enabled, distance_fade_begin, distance_fade_length);
	notify_property_list_changed();
}

bool Decal::is_distance_fade_enabled() const {
	return distance_fade_enabled;
}

void Decal::set_distance_fade_begin(float p_distance) {
	distance_fade_begin = p_distance;
	RS::get_singleton()->decal_set_distance_fade(decal, distance_fade_enabled, distance_fade_begin, distance_fade_length);
}

float Decal::get_distance_fade_begin() const {
	return distance_fade_begin;
}

void Decal::set_distance_fade_length(float p_length) {
	distance_fade_length = p_length;
	RS::get_singleton()->decal_set_distance_fade(decal, distance_fade_enabled, distance_fade_begin, distance_fade_length);
}

float Decal::get_distance_fade_length() const {
	return distance_fade_length;
}

void Decal::set_cull_mask(uint32_t p_layers) {
	cull_mask = p_layers;
	RS::get_singleton()->decal_set_cull_mask(decal, cull_mask);
}

uint32_t Decal::get_cull_mask() const {
	return cull_mask;
}

AABB Decal::get_aabb() const {
	AABB aabb;
	aabb.position = -extents;
	aabb.size = extents * 2.0;
	return aabb;
}

Vector<Face3> Decal::get_faces(uint32_t p_usage_flags) const {
	return Vector<Face3>();
}

void Decal::_validate_property(PropertyInfo &property) const {
	if (!distance_fade_enabled && (property.name == "distance_fade_begin" || property.name == "distance_fade_length")) {
		property.usage = PROPERTY_USAGE_NOEDITOR;
	}
}

void Decal::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_extents", "extents"), &Decal::set_extents);
	ClassDB::bind_method(D_METHOD("get_extents"), &Decal::get_extents);

	ClassDB::bind_method(D_METHOD("set_texture", "type", "texture"), &Decal::set_texture);
	ClassDB::bind_method(D_METHOD("get_texture", "type"), &Decal::get_texture);

	ClassDB::bind_method(D_METHOD("set_emission_energy", "energy"), &Decal::set_emission_energy);
	ClassDB::bind_method(D_METHOD("get_emission_energy"), &Decal::get_emission_energy);

	ClassDB::bind_method(D_METHOD("set_albedo_mix", "energy"), &Decal::set_albedo_mix);
	ClassDB::bind_method(D_METHOD("get_albedo_mix"), &Decal::get_albedo_mix);

	ClassDB::bind_method(D_METHOD("set_modulate", "color"), &Decal::set_modulate);
	ClassDB::bind_method(D_METHOD("get_modulate"), &Decal::get_modulate);

	ClassDB::bind_method(D_METHOD("set_upper_fade", "fade"), &Decal::set_upper_fade);
	ClassDB::bind_method(D_METHOD("get_upper_fade"), &Decal::get_upper_fade);

	ClassDB::bind_method(D_METHOD("set_lower_fade", "fade"), &Decal::set_lower_fade);
	ClassDB::bind_method(D_METHOD("get_lower_fade"), &Decal::get_lower_fade);

	ClassDB::bind_method(D_METHOD("set_normal_fade", "fade"), &Decal::set_normal_fade);
	ClassDB::bind_method(D_METHOD("get_normal_fade"), &Decal::get_normal_fade);

	ClassDB::bind_method(D_METHOD("set_enable_distance_fade", "enable"), &Decal::set_enable_distance_fade);
	ClassDB::bind_method(D_METHOD("is_distance_fade_enabled"), &Decal::is_distance_fade_enabled);

	ClassDB::bind_method(D_METHOD("set_distance_fade_begin", "distance"), &Decal::set_distance_fade_begin);
	ClassDB::bind_method(D_METHOD("get_distance_fade_begin"), &Decal::get_distance_fade_begin);

	ClassDB::bind_method(D_METHOD("set_distance_fade_length", "distance"), &Decal::set_distance_fade_length);
	ClassDB::bind_method(D_METHOD("get_distance_fade_length"), &Decal::get_distance_fade_length);

	ClassDB::bind_method(D_METHOD("set_cull_mask", "mask"), &Decal::set_cull_mask);
	ClassDB::bind_method(D_METHOD("get_cull_mask"), &Decal::get_cull_mask);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "extents", PROPERTY_HINT_RANGE, "0,1024,0.001,or_greater"), "set_extents", "get_extents");
	ADD_GROUP("Textures", "texture_");
	ADD_PROPERTYI(PropertyInfo(Variant::OBJECT, "texture_albedo", PROPERTY_HINT_RESOURCE_TYPE, "Texture"), "set_texture", "get_texture", TEXTURE_ALBEDO);
	ADD_PROPERTYI(PropertyInfo(Variant::OBJECT, "texture_normal", PROPERTY_HINT_RESOURCE_TYPE, "Texture"), "set_texture", "get_texture", TEXTURE_NORMAL);
	ADD_PROPERTYI(PropertyInfo(Variant::OBJECT, "texture_orm", PROPERTY_HINT_RESOURCE_TYPE, "Texture"), "set_texture", "get_texture", TEXTURE_ORM);
	ADD_PROPERTYI(PropertyInfo(Variant::OBJECT, "texture_emission", PROPERTY_HINT_RESOURCE_TYPE, "Texture"), "set_texture", "get_texture", TEXTURE_EMISSION);
	ADD_GROUP("Parameters", "");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "emission_energy", PROPERTY_HINT_RANGE, "0,128,0.01"), "set_emission_energy", "get_emission_energy");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "modulate"), "set_modulate", "get_modulate");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "albedo_mix", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_albedo_mix", "get_albedo_mix");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "normal_fade", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_normal_fade", "get_normal_fade");
	ADD_GROUP("Vertical Fade", "");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "upper_fade", PROPERTY_HINT_EXP_EASING, "attenuation"), "set_upper_fade", "get_upper_fade");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "lower_fade", PROPERTY_HINT_EXP_EASING, "attenuation"), "set_lower_fade", "get_lower_fade");
	ADD_GROUP("Distance Fade", "distance_fade_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "distance_fade_enabled"), "set_enable_distance_fade", "is_distance_fade_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "distance_fade_begin"), "set_distance_fade_begin", "get_distance_fade_begin");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "distance_fade_length"), "set_distance_fade_length", "get_distance_fade_length");
	ADD_GROUP("Cull Mask", "");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "cull_mask", PROPERTY_HINT_LAYERS_3D_RENDER), "set_cull_mask", "get_cull_mask");

	BIND_ENUM_CONSTANT(TEXTURE_ALBEDO);
	BIND_ENUM_CONSTANT(TEXTURE_NORMAL);
	BIND_ENUM_CONSTANT(TEXTURE_ORM);
	BIND_ENUM_CONSTANT(TEXTURE_EMISSION);
	BIND_ENUM_CONSTANT(TEXTURE_MAX);
}

Decal::Decal() {
	decal = RenderingServer::get_singleton()->decal_create();
	RS::get_singleton()->instance_set_base(get_instance(), decal);
}

Decal::~Decal() {
	RS::get_singleton()->free(decal);
}
