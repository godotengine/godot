/*************************************************************************/
/*  visual_instance_3d.cpp                                               */
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

#include "visual_instance_3d.h"

#include "scene/scene_string_names.h"
#include "servers/rendering_server.h"
#include "skeleton_3d.h"

AABB VisualInstance3D::get_transformed_aabb() const {

	return get_global_transform().xform(get_aabb());
}

void VisualInstance3D::_update_visibility() {

	if (!is_inside_tree())
		return;

	_change_notify("visible");
	RS::get_singleton()->instance_set_visible(get_instance(), is_visible_in_tree());
}

void VisualInstance3D::_notification(int p_what) {

	switch (p_what) {

		case NOTIFICATION_ENTER_WORLD: {

			// CHECK SKELETON => moving skeleton attaching logic to MeshInstance
			/*
			Skeleton *skeleton=Object::cast_to<Skeleton>(get_parent());
			if (skeleton)
				RenderingServer::get_singleton()->instance_attach_skeleton( instance, skeleton->get_skeleton() );
			*/
			ERR_FAIL_COND(get_world().is_null());
			RenderingServer::get_singleton()->instance_set_scenario(instance, get_world()->get_scenario());
			_update_visibility();

		} break;
		case NOTIFICATION_TRANSFORM_CHANGED: {

			Transform gt = get_global_transform();
			RenderingServer::get_singleton()->instance_set_transform(instance, gt);
		} break;
		case NOTIFICATION_EXIT_WORLD: {

			RenderingServer::get_singleton()->instance_set_scenario(instance, RID());
			RenderingServer::get_singleton()->instance_attach_skeleton(instance, RID());
			//RS::get_singleton()->instance_geometry_set_baked_light_sampler(instance, RID() );

		} break;
		case NOTIFICATION_VISIBILITY_CHANGED: {

			_update_visibility();
		} break;
	}
}

RID VisualInstance3D::get_instance() const {

	return instance;
}

RID VisualInstance3D::_get_visual_instance_rid() const {

	return instance;
}

void VisualInstance3D::set_layer_mask(uint32_t p_mask) {

	layers = p_mask;
	RenderingServer::get_singleton()->instance_set_layer_mask(instance, p_mask);
}

uint32_t VisualInstance3D::get_layer_mask() const {

	return layers;
}

void VisualInstance3D::set_layer_mask_bit(int p_layer, bool p_enable) {
	ERR_FAIL_INDEX(p_layer, 32);
	if (p_enable) {
		set_layer_mask(layers | (1 << p_layer));
	} else {
		set_layer_mask(layers & (~(1 << p_layer)));
	}
}

bool VisualInstance3D::get_layer_mask_bit(int p_layer) const {
	ERR_FAIL_INDEX_V(p_layer, 32, false);
	return (layers & (1 << p_layer));
}

void VisualInstance3D::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_get_visual_instance_rid"), &VisualInstance3D::_get_visual_instance_rid);
	ClassDB::bind_method(D_METHOD("set_base", "base"), &VisualInstance3D::set_base);
	ClassDB::bind_method(D_METHOD("get_base"), &VisualInstance3D::get_base);
	ClassDB::bind_method(D_METHOD("get_instance"), &VisualInstance3D::get_instance);
	ClassDB::bind_method(D_METHOD("set_layer_mask", "mask"), &VisualInstance3D::set_layer_mask);
	ClassDB::bind_method(D_METHOD("get_layer_mask"), &VisualInstance3D::get_layer_mask);
	ClassDB::bind_method(D_METHOD("set_layer_mask_bit", "layer", "enabled"), &VisualInstance3D::set_layer_mask_bit);
	ClassDB::bind_method(D_METHOD("get_layer_mask_bit", "layer"), &VisualInstance3D::get_layer_mask_bit);

	ClassDB::bind_method(D_METHOD("get_transformed_aabb"), &VisualInstance3D::get_transformed_aabb);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "layers", PROPERTY_HINT_LAYERS_3D_RENDER), "set_layer_mask", "get_layer_mask");
}

void VisualInstance3D::set_base(const RID &p_base) {

	RenderingServer::get_singleton()->instance_set_base(instance, p_base);
	base = p_base;
}

RID VisualInstance3D::get_base() const {

	return base;
}

VisualInstance3D::VisualInstance3D() {

	instance = RenderingServer::get_singleton()->instance_create();
	RenderingServer::get_singleton()->instance_attach_object_instance_id(instance, get_instance_id());
	layers = 1;
	set_notify_transform(true);
}

VisualInstance3D::~VisualInstance3D() {

	RenderingServer::get_singleton()->free(instance);
}

void GeometryInstance3D::set_material_override(const Ref<Material> &p_material) {

	material_override = p_material;
	RS::get_singleton()->instance_geometry_set_material_override(get_instance(), p_material.is_valid() ? p_material->get_rid() : RID());
}

Ref<Material> GeometryInstance3D::get_material_override() const {

	return material_override;
}

void GeometryInstance3D::set_lod_min_distance(float p_dist) {

	lod_min_distance = p_dist;
	RS::get_singleton()->instance_geometry_set_draw_range(get_instance(), lod_min_distance, lod_max_distance, lod_min_hysteresis, lod_max_hysteresis);
}

float GeometryInstance3D::get_lod_min_distance() const {

	return lod_min_distance;
}

void GeometryInstance3D::set_lod_max_distance(float p_dist) {

	lod_max_distance = p_dist;
	RS::get_singleton()->instance_geometry_set_draw_range(get_instance(), lod_min_distance, lod_max_distance, lod_min_hysteresis, lod_max_hysteresis);
}

float GeometryInstance3D::get_lod_max_distance() const {

	return lod_max_distance;
}

void GeometryInstance3D::set_lod_min_hysteresis(float p_dist) {

	lod_min_hysteresis = p_dist;
	RS::get_singleton()->instance_geometry_set_draw_range(get_instance(), lod_min_distance, lod_max_distance, lod_min_hysteresis, lod_max_hysteresis);
}

float GeometryInstance3D::get_lod_min_hysteresis() const {

	return lod_min_hysteresis;
}

void GeometryInstance3D::set_lod_max_hysteresis(float p_dist) {

	lod_max_hysteresis = p_dist;
	RS::get_singleton()->instance_geometry_set_draw_range(get_instance(), lod_min_distance, lod_max_distance, lod_min_hysteresis, lod_max_hysteresis);
}

float GeometryInstance3D::get_lod_max_hysteresis() const {

	return lod_max_hysteresis;
}

void GeometryInstance3D::_notification(int p_what) {
}

void GeometryInstance3D::set_flag(Flags p_flag, bool p_value) {

	ERR_FAIL_INDEX(p_flag, FLAG_MAX);
	if (flags[p_flag] == p_value)
		return;

	flags[p_flag] = p_value;
	RS::get_singleton()->instance_geometry_set_flag(get_instance(), (RS::InstanceFlags)p_flag, p_value);
}

bool GeometryInstance3D::get_flag(Flags p_flag) const {

	ERR_FAIL_INDEX_V(p_flag, FLAG_MAX, false);

	return flags[p_flag];
}

void GeometryInstance3D::set_cast_shadows_setting(ShadowCastingSetting p_shadow_casting_setting) {

	shadow_casting_setting = p_shadow_casting_setting;

	RS::get_singleton()->instance_geometry_set_cast_shadows_setting(get_instance(), (RS::ShadowCastingSetting)p_shadow_casting_setting);
}

GeometryInstance3D::ShadowCastingSetting GeometryInstance3D::get_cast_shadows_setting() const {

	return shadow_casting_setting;
}

void GeometryInstance3D::set_extra_cull_margin(float p_margin) {

	ERR_FAIL_COND(p_margin < 0);
	extra_cull_margin = p_margin;
	RS::get_singleton()->instance_set_extra_visibility_margin(get_instance(), extra_cull_margin);
}

float GeometryInstance3D::get_extra_cull_margin() const {

	return extra_cull_margin;
}

void GeometryInstance3D::set_custom_aabb(AABB aabb) {

	RS::get_singleton()->instance_set_custom_aabb(get_instance(), aabb);
}

void GeometryInstance3D::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_material_override", "material"), &GeometryInstance3D::set_material_override);
	ClassDB::bind_method(D_METHOD("get_material_override"), &GeometryInstance3D::get_material_override);

	ClassDB::bind_method(D_METHOD("set_flag", "flag", "value"), &GeometryInstance3D::set_flag);
	ClassDB::bind_method(D_METHOD("get_flag", "flag"), &GeometryInstance3D::get_flag);

	ClassDB::bind_method(D_METHOD("set_cast_shadows_setting", "shadow_casting_setting"), &GeometryInstance3D::set_cast_shadows_setting);
	ClassDB::bind_method(D_METHOD("get_cast_shadows_setting"), &GeometryInstance3D::get_cast_shadows_setting);

	ClassDB::bind_method(D_METHOD("set_lod_max_hysteresis", "mode"), &GeometryInstance3D::set_lod_max_hysteresis);
	ClassDB::bind_method(D_METHOD("get_lod_max_hysteresis"), &GeometryInstance3D::get_lod_max_hysteresis);

	ClassDB::bind_method(D_METHOD("set_lod_max_distance", "mode"), &GeometryInstance3D::set_lod_max_distance);
	ClassDB::bind_method(D_METHOD("get_lod_max_distance"), &GeometryInstance3D::get_lod_max_distance);

	ClassDB::bind_method(D_METHOD("set_lod_min_hysteresis", "mode"), &GeometryInstance3D::set_lod_min_hysteresis);
	ClassDB::bind_method(D_METHOD("get_lod_min_hysteresis"), &GeometryInstance3D::get_lod_min_hysteresis);

	ClassDB::bind_method(D_METHOD("set_lod_min_distance", "mode"), &GeometryInstance3D::set_lod_min_distance);
	ClassDB::bind_method(D_METHOD("get_lod_min_distance"), &GeometryInstance3D::get_lod_min_distance);

	ClassDB::bind_method(D_METHOD("set_extra_cull_margin", "margin"), &GeometryInstance3D::set_extra_cull_margin);
	ClassDB::bind_method(D_METHOD("get_extra_cull_margin"), &GeometryInstance3D::get_extra_cull_margin);

	ClassDB::bind_method(D_METHOD("set_custom_aabb", "aabb"), &GeometryInstance3D::set_custom_aabb);

	ClassDB::bind_method(D_METHOD("get_aabb"), &GeometryInstance3D::get_aabb);

	ADD_GROUP("Geometry", "");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "material_override", PROPERTY_HINT_RESOURCE_TYPE, "ShaderMaterial,StandardMaterial3D", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_DEFERRED_SET_RESOURCE), "set_material_override", "get_material_override");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "cast_shadow", PROPERTY_HINT_ENUM, "Off,On,Double-Sided,Shadows Only"), "set_cast_shadows_setting", "get_cast_shadows_setting");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "extra_cull_margin", PROPERTY_HINT_RANGE, "0,16384,0.01"), "set_extra_cull_margin", "get_extra_cull_margin");
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "use_in_baked_light"), "set_flag", "get_flag", FLAG_USE_BAKED_LIGHT);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "use_dynamic_gi"), "set_flag", "get_flag", FLAG_USE_DYNAMIC_GI);

	ADD_GROUP("LOD", "lod_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "lod_min_distance", PROPERTY_HINT_RANGE, "0,32768,0.01"), "set_lod_min_distance", "get_lod_min_distance");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "lod_min_hysteresis", PROPERTY_HINT_RANGE, "0,32768,0.01"), "set_lod_min_hysteresis", "get_lod_min_hysteresis");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "lod_max_distance", PROPERTY_HINT_RANGE, "0,32768,0.01"), "set_lod_max_distance", "get_lod_max_distance");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "lod_max_hysteresis", PROPERTY_HINT_RANGE, "0,32768,0.01"), "set_lod_max_hysteresis", "get_lod_max_hysteresis");

	//ADD_SIGNAL( MethodInfo("visibility_changed"));

	BIND_ENUM_CONSTANT(SHADOW_CASTING_SETTING_OFF);
	BIND_ENUM_CONSTANT(SHADOW_CASTING_SETTING_ON);
	BIND_ENUM_CONSTANT(SHADOW_CASTING_SETTING_DOUBLE_SIDED);
	BIND_ENUM_CONSTANT(SHADOW_CASTING_SETTING_SHADOWS_ONLY);

	BIND_ENUM_CONSTANT(FLAG_USE_BAKED_LIGHT);
	BIND_ENUM_CONSTANT(FLAG_USE_DYNAMIC_GI);
	BIND_ENUM_CONSTANT(FLAG_DRAW_NEXT_FRAME_IF_VISIBLE);
	BIND_ENUM_CONSTANT(FLAG_MAX);
}

GeometryInstance3D::GeometryInstance3D() {
	lod_min_distance = 0;
	lod_max_distance = 0;
	lod_min_hysteresis = 0;
	lod_max_hysteresis = 0;

	for (int i = 0; i < FLAG_MAX; i++) {
		flags[i] = false;
	}

	shadow_casting_setting = SHADOW_CASTING_SETTING_ON;
	extra_cull_margin = 0;
	//RS::get_singleton()->instance_geometry_set_baked_light_texture_index(get_instance(),0);
}
