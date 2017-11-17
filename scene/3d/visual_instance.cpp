/*************************************************************************/
/*  visual_instance.cpp                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "visual_instance.h"

#include "scene/scene_string_names.h"
#include "servers/visual_server.h"
#include "skeleton.h"

AABB VisualInstance::get_transformed_aabb() const {

	return get_global_transform().xform(get_aabb());
}

void VisualInstance::_update_visibility() {

	if (!is_inside_tree())
		return;

	_change_notify("visible");
	VS::get_singleton()->instance_set_visible(get_instance(), is_visible_in_tree());
}

void VisualInstance::_notification(int p_what) {

	switch (p_what) {

		case NOTIFICATION_ENTER_WORLD: {

			// CHECK SKELETON => moving skeleton attaching logic to MeshInstance
			/*
			Skeleton *skeleton=Object::cast_to<Skeleton>(get_parent());
			if (skeleton)
				VisualServer::get_singleton()->instance_attach_skeleton( instance, skeleton->get_skeleton() );
			*/

			VisualServer::get_singleton()->instance_set_scenario(instance, get_world()->get_scenario());
			_update_visibility();

		} break;
		case NOTIFICATION_TRANSFORM_CHANGED: {

			Transform gt = get_global_transform();
			VisualServer::get_singleton()->instance_set_transform(instance, gt);
		} break;
		case NOTIFICATION_EXIT_WORLD: {

			VisualServer::get_singleton()->instance_set_scenario(instance, RID());
			VisualServer::get_singleton()->instance_attach_skeleton(instance, RID());
			//VS::get_singleton()->instance_geometry_set_baked_light_sampler(instance, RID() );

		} break;
		case NOTIFICATION_VISIBILITY_CHANGED: {

			_update_visibility();
		} break;
	}
}

RID VisualInstance::get_instance() const {

	return instance;
}

RID VisualInstance::_get_visual_instance_rid() const {

	return instance;
}

void VisualInstance::set_layer_mask(uint32_t p_mask) {

	layers = p_mask;
	VisualServer::get_singleton()->instance_set_layer_mask(instance, p_mask);
}

uint32_t VisualInstance::get_layer_mask() const {

	return layers;
}

void VisualInstance::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_get_visual_instance_rid"), &VisualInstance::_get_visual_instance_rid);
	ClassDB::bind_method(D_METHOD("set_base", "base"), &VisualInstance::set_base);
	ClassDB::bind_method(D_METHOD("set_layer_mask", "mask"), &VisualInstance::set_layer_mask);
	ClassDB::bind_method(D_METHOD("get_layer_mask"), &VisualInstance::get_layer_mask);

	ClassDB::bind_method(D_METHOD("get_transformed_aabb"), &VisualInstance::get_transformed_aabb);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "layers", PROPERTY_HINT_LAYERS_3D_RENDER), "set_layer_mask", "get_layer_mask");
}

void VisualInstance::set_base(const RID &p_base) {

	VisualServer::get_singleton()->instance_set_base(instance, p_base);
}

VisualInstance::VisualInstance() {

	instance = VisualServer::get_singleton()->instance_create();
	VisualServer::get_singleton()->instance_attach_object_instance_id(instance, get_instance_id());
	layers = 1;
	set_notify_transform(true);
}

VisualInstance::~VisualInstance() {

	VisualServer::get_singleton()->free(instance);
}

void GeometryInstance::set_material_override(const Ref<Material> &p_material) {

	material_override = p_material;
	VS::get_singleton()->instance_geometry_set_material_override(get_instance(), p_material.is_valid() ? p_material->get_rid() : RID());
}

Ref<Material> GeometryInstance::get_material_override() const {

	return material_override;
}

void GeometryInstance::set_lod_min_distance(float p_dist) {

	lod_min_distance = p_dist;
	VS::get_singleton()->instance_geometry_set_draw_range(get_instance(), lod_min_distance, lod_max_distance, lod_min_hysteresis, lod_max_hysteresis);
}

float GeometryInstance::get_lod_min_distance() const {

	return lod_min_distance;
}

void GeometryInstance::set_lod_max_distance(float p_dist) {

	lod_max_distance = p_dist;
	VS::get_singleton()->instance_geometry_set_draw_range(get_instance(), lod_min_distance, lod_max_distance, lod_min_hysteresis, lod_max_hysteresis);
}

float GeometryInstance::get_lod_max_distance() const {

	return lod_max_distance;
}

void GeometryInstance::set_lod_min_hysteresis(float p_dist) {

	lod_min_hysteresis = p_dist;
	VS::get_singleton()->instance_geometry_set_draw_range(get_instance(), lod_min_distance, lod_max_distance, lod_min_hysteresis, lod_max_hysteresis);
}

float GeometryInstance::get_lod_min_hysteresis() const {

	return lod_min_hysteresis;
}

void GeometryInstance::set_lod_max_hysteresis(float p_dist) {

	lod_max_hysteresis = p_dist;
	VS::get_singleton()->instance_geometry_set_draw_range(get_instance(), lod_min_distance, lod_max_distance, lod_min_hysteresis, lod_max_hysteresis);
}

float GeometryInstance::get_lod_max_hysteresis() const {

	return lod_max_hysteresis;
}

void GeometryInstance::_notification(int p_what) {

	if (p_what == NOTIFICATION_ENTER_WORLD) {

		if (flags[FLAG_USE_BAKED_LIGHT]) {
		}

	} else if (p_what == NOTIFICATION_EXIT_WORLD) {

		if (flags[FLAG_USE_BAKED_LIGHT]) {
		}
	}
}

void GeometryInstance::set_flag(Flags p_flag, bool p_value) {

	ERR_FAIL_INDEX(p_flag, FLAG_MAX);
	if (flags[p_flag] == p_value)
		return;

	flags[p_flag] = p_value;
	VS::get_singleton()->instance_geometry_set_flag(get_instance(), (VS::InstanceFlags)p_flag, p_value);
	if (p_flag == FLAG_USE_BAKED_LIGHT) {
	}
}

bool GeometryInstance::get_flag(Flags p_flag) const {

	ERR_FAIL_INDEX_V(p_flag, FLAG_MAX, false);

	return flags[p_flag];
}

void GeometryInstance::set_cast_shadows_setting(ShadowCastingSetting p_shadow_casting_setting) {

	shadow_casting_setting = p_shadow_casting_setting;

	VS::get_singleton()->instance_geometry_set_cast_shadows_setting(get_instance(), (VS::ShadowCastingSetting)p_shadow_casting_setting);
}

GeometryInstance::ShadowCastingSetting GeometryInstance::get_cast_shadows_setting() const {

	return shadow_casting_setting;
}

void GeometryInstance::set_extra_cull_margin(float p_margin) {

	ERR_FAIL_COND(p_margin < 0);
	extra_cull_margin = p_margin;
	VS::get_singleton()->instance_set_extra_visibility_margin(get_instance(), extra_cull_margin);
}

float GeometryInstance::get_extra_cull_margin() const {

	return extra_cull_margin;
}

void GeometryInstance::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_material_override", "material"), &GeometryInstance::set_material_override);
	ClassDB::bind_method(D_METHOD("get_material_override"), &GeometryInstance::get_material_override);

	ClassDB::bind_method(D_METHOD("set_flag", "flag", "value"), &GeometryInstance::set_flag);
	ClassDB::bind_method(D_METHOD("get_flag", "flag"), &GeometryInstance::get_flag);

	ClassDB::bind_method(D_METHOD("set_cast_shadows_setting", "shadow_casting_setting"), &GeometryInstance::set_cast_shadows_setting);
	ClassDB::bind_method(D_METHOD("get_cast_shadows_setting"), &GeometryInstance::get_cast_shadows_setting);

	ClassDB::bind_method(D_METHOD("set_lod_max_hysteresis", "mode"), &GeometryInstance::set_lod_max_hysteresis);
	ClassDB::bind_method(D_METHOD("get_lod_max_hysteresis"), &GeometryInstance::get_lod_max_hysteresis);

	ClassDB::bind_method(D_METHOD("set_lod_max_distance", "mode"), &GeometryInstance::set_lod_max_distance);
	ClassDB::bind_method(D_METHOD("get_lod_max_distance"), &GeometryInstance::get_lod_max_distance);

	ClassDB::bind_method(D_METHOD("set_lod_min_hysteresis", "mode"), &GeometryInstance::set_lod_min_hysteresis);
	ClassDB::bind_method(D_METHOD("get_lod_min_hysteresis"), &GeometryInstance::get_lod_min_hysteresis);

	ClassDB::bind_method(D_METHOD("set_lod_min_distance", "mode"), &GeometryInstance::set_lod_min_distance);
	ClassDB::bind_method(D_METHOD("get_lod_min_distance"), &GeometryInstance::get_lod_min_distance);

	ClassDB::bind_method(D_METHOD("set_extra_cull_margin", "margin"), &GeometryInstance::set_extra_cull_margin);
	ClassDB::bind_method(D_METHOD("get_extra_cull_margin"), &GeometryInstance::get_extra_cull_margin);

	ClassDB::bind_method(D_METHOD("get_aabb"), &GeometryInstance::get_aabb);

	ADD_GROUP("Geometry", "");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "material_override", PROPERTY_HINT_RESOURCE_TYPE, "ShaderMaterial,SpatialMaterial"), "set_material_override", "get_material_override");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "cast_shadow", PROPERTY_HINT_ENUM, "Off,On,Double-Sided,Shadows Only"), "set_cast_shadows_setting", "get_cast_shadows_setting");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "extra_cull_margin", PROPERTY_HINT_RANGE, "0,16384,0"), "set_extra_cull_margin", "get_extra_cull_margin");
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "use_in_baked_light"), "set_flag", "get_flag", FLAG_USE_BAKED_LIGHT);

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
	BIND_ENUM_CONSTANT(FLAG_MAX);
}

GeometryInstance::GeometryInstance() {
	lod_min_distance = 0;
	lod_max_distance = 0;
	lod_min_hysteresis = 0;
	lod_max_hysteresis = 0;

	for (int i = 0; i < FLAG_MAX; i++) {
		flags[i] = false;
	}

	shadow_casting_setting = SHADOW_CASTING_SETTING_ON;
	extra_cull_margin = 0;
	//VS::get_singleton()->instance_geometry_set_baked_light_texture_index(get_instance(),0);
}
