/**************************************************************************/
/*  light.cpp                                                             */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "light.h"

#include "core/engine.h"
#include "core/project_settings.h"
#include "scene/resources/surface_tool.h"

void Light::set_param(Param p_param, float p_value) {
	ERR_FAIL_INDEX(p_param, PARAM_MAX);
	param[p_param] = p_value;

	VS::get_singleton()->light_set_param(light, VS::LightParam(p_param), p_value);

	if (p_param == PARAM_SPOT_ANGLE || p_param == PARAM_RANGE) {
		update_gizmo();

		if (p_param == PARAM_SPOT_ANGLE) {
			_change_notify("spot_angle");
			update_configuration_warning();
		} else if (p_param == PARAM_RANGE) {
			_change_notify("omni_range");
			_change_notify("spot_range");
		}
	}
}

float Light::get_param(Param p_param) const {
	ERR_FAIL_INDEX_V(p_param, PARAM_MAX, 0);
	return param[p_param];
}

void Light::set_shadow(bool p_enable) {
	shadow = p_enable;
	VS::get_singleton()->light_set_shadow(light, p_enable);

	if (type == VisualServer::LIGHT_SPOT) {
		update_configuration_warning();
	}
}
bool Light::has_shadow() const {
	return shadow;
}

void Light::set_negative(bool p_enable) {
	negative = p_enable;
	VS::get_singleton()->light_set_negative(light, p_enable);
}
bool Light::is_negative() const {
	return negative;
}

void Light::set_cull_mask(uint32_t p_cull_mask) {
	cull_mask = p_cull_mask;
	VS::get_singleton()->light_set_cull_mask(light, p_cull_mask);
}
uint32_t Light::get_cull_mask() const {
	return cull_mask;
}

void Light::set_color(const Color &p_color) {
	color = p_color;
	VS::get_singleton()->light_set_color(light, p_color);
	// The gizmo color depends on the light color, so update it.
	update_gizmo();
}
Color Light::get_color() const {
	return color;
}

void Light::set_shadow_color(const Color &p_shadow_color) {
	shadow_color = p_shadow_color;
	VS::get_singleton()->light_set_shadow_color(light, p_shadow_color);
}

Color Light::get_shadow_color() const {
	return shadow_color;
}

void Light::set_shadow_reverse_cull_face(bool p_enable) {
	reverse_cull = p_enable;
	VS::get_singleton()->light_set_reverse_cull_face_mode(light, reverse_cull);
}

bool Light::get_shadow_reverse_cull_face() const {
	return reverse_cull;
}

AABB Light::get_aabb() const {
	if (type == VisualServer::LIGHT_DIRECTIONAL) {
		return AABB(Vector3(-1, -1, -1), Vector3(2, 2, 2));

	} else if (type == VisualServer::LIGHT_OMNI) {
		return AABB(Vector3(-1, -1, -1) * param[PARAM_RANGE], Vector3(2, 2, 2) * param[PARAM_RANGE]);

	} else if (type == VisualServer::LIGHT_SPOT) {
		float len = param[PARAM_RANGE];
		float size = Math::tan(Math::deg2rad(param[PARAM_SPOT_ANGLE])) * len;
		return AABB(Vector3(-size, -size, -len), Vector3(size * 2, size * 2, len));
	}

	return AABB();
}

PoolVector<Face3> Light::get_faces(uint32_t p_usage_flags) const {
	return PoolVector<Face3>();
}

void Light::set_bake_mode(BakeMode p_mode) {
	bake_mode = p_mode;
	VS::get_singleton()->light_set_bake_mode(light, VS::LightBakeMode(bake_mode));
	_change_notify();
}

Light::BakeMode Light::get_bake_mode() const {
	return bake_mode;
}

void Light::owner_changed_notify() {
	// For cases where owner changes _after_ entering tree (as example, editor editing).
	_update_visibility();
}

void Light::_update_visibility() {
	if (!is_inside_tree()) {
		return;
	}

	bool editor_ok = true;

#ifdef TOOLS_ENABLED
	if (editor_only) {
		if (!Engine::get_singleton()->is_editor_hint()) {
			editor_ok = false;
		} else {
			editor_ok = (get_tree()->get_edited_scene_root() && (this == get_tree()->get_edited_scene_root() || get_owner() == get_tree()->get_edited_scene_root()));
		}
	}
#else
	if (editor_only) {
		editor_ok = false;
	}
#endif

	VS::get_singleton()->instance_set_visible(get_instance(), is_visible_in_tree() && editor_ok);

	_change_notify("geometry/visible");
}

void Light::_notification(int p_what) {
	if (p_what == NOTIFICATION_VISIBILITY_CHANGED) {
		_update_visibility();
	}

	if (p_what == NOTIFICATION_ENTER_TREE) {
		_update_visibility();
	}
}

void Light::set_editor_only(bool p_editor_only) {
	editor_only = p_editor_only;
	_update_visibility();
}

bool Light::is_editor_only() const {
	return editor_only;
}

void Light::_validate_property(PropertyInfo &property) const {
	if (VisualServer::get_singleton()->is_low_end() && property.name == "shadow_contact") {
		property.usage = PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL;
	}

	if (bake_mode != BAKE_ALL && property.name == "light_size") {
		property.usage = PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL;
	}
}

void Light::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_editor_only", "editor_only"), &Light::set_editor_only);
	ClassDB::bind_method(D_METHOD("is_editor_only"), &Light::is_editor_only);

	ClassDB::bind_method(D_METHOD("set_param", "param", "value"), &Light::set_param);
	ClassDB::bind_method(D_METHOD("get_param", "param"), &Light::get_param);

	ClassDB::bind_method(D_METHOD("set_shadow", "enabled"), &Light::set_shadow);
	ClassDB::bind_method(D_METHOD("has_shadow"), &Light::has_shadow);

	ClassDB::bind_method(D_METHOD("set_negative", "enabled"), &Light::set_negative);
	ClassDB::bind_method(D_METHOD("is_negative"), &Light::is_negative);

	ClassDB::bind_method(D_METHOD("set_cull_mask", "cull_mask"), &Light::set_cull_mask);
	ClassDB::bind_method(D_METHOD("get_cull_mask"), &Light::get_cull_mask);

	ClassDB::bind_method(D_METHOD("set_color", "color"), &Light::set_color);
	ClassDB::bind_method(D_METHOD("get_color"), &Light::get_color);

	ClassDB::bind_method(D_METHOD("set_shadow_reverse_cull_face", "enable"), &Light::set_shadow_reverse_cull_face);
	ClassDB::bind_method(D_METHOD("get_shadow_reverse_cull_face"), &Light::get_shadow_reverse_cull_face);

	ClassDB::bind_method(D_METHOD("set_shadow_color", "shadow_color"), &Light::set_shadow_color);
	ClassDB::bind_method(D_METHOD("get_shadow_color"), &Light::get_shadow_color);

	ClassDB::bind_method(D_METHOD("set_bake_mode", "bake_mode"), &Light::set_bake_mode);
	ClassDB::bind_method(D_METHOD("get_bake_mode"), &Light::get_bake_mode);

	ADD_GROUP("Light", "light_");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "light_color", PROPERTY_HINT_COLOR_NO_ALPHA), "set_color", "get_color");
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "light_energy", PROPERTY_HINT_RANGE, "0,16,0.001,or_greater"), "set_param", "get_param", PARAM_ENERGY);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "light_indirect_energy", PROPERTY_HINT_RANGE, "0,16,0.001,or_greater"), "set_param", "get_param", PARAM_INDIRECT_ENERGY);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "light_size", PROPERTY_HINT_RANGE, "0,1,0.001,or_greater"), "set_param", "get_param", PARAM_SIZE);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "light_negative"), "set_negative", "is_negative");
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "light_specular", PROPERTY_HINT_RANGE, "0,16,0.001,or_greater"), "set_param", "get_param", PARAM_SPECULAR);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "light_bake_mode", PROPERTY_HINT_ENUM, "Disable,Indirect Only,All (Direct + Indirect)"), "set_bake_mode", "get_bake_mode");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "light_cull_mask", PROPERTY_HINT_LAYERS_3D_RENDER), "set_cull_mask", "get_cull_mask");
	ADD_GROUP("Shadow", "shadow_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "shadow_enabled"), "set_shadow", "has_shadow");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "shadow_color", PROPERTY_HINT_COLOR_NO_ALPHA), "set_shadow_color", "get_shadow_color");
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "shadow_bias", PROPERTY_HINT_RANGE, "-10,10,0.001"), "set_param", "get_param", PARAM_SHADOW_BIAS);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "shadow_contact", PROPERTY_HINT_RANGE, "0,10,0.001"), "set_param", "get_param", PARAM_CONTACT_SHADOW_SIZE);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "shadow_reverse_cull_face"), "set_shadow_reverse_cull_face", "get_shadow_reverse_cull_face");
	ADD_GROUP("Editor", "");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "editor_only"), "set_editor_only", "is_editor_only");
	ADD_GROUP("", "");

	BIND_ENUM_CONSTANT(PARAM_ENERGY);
	BIND_ENUM_CONSTANT(PARAM_INDIRECT_ENERGY);
	BIND_ENUM_CONSTANT(PARAM_SIZE);
	BIND_ENUM_CONSTANT(PARAM_SPECULAR);
	BIND_ENUM_CONSTANT(PARAM_RANGE);
	BIND_ENUM_CONSTANT(PARAM_ATTENUATION);
	BIND_ENUM_CONSTANT(PARAM_SPOT_ANGLE);
	BIND_ENUM_CONSTANT(PARAM_SPOT_ATTENUATION);
	BIND_ENUM_CONSTANT(PARAM_CONTACT_SHADOW_SIZE);
	BIND_ENUM_CONSTANT(PARAM_SHADOW_MAX_DISTANCE);
	BIND_ENUM_CONSTANT(PARAM_SHADOW_SPLIT_1_OFFSET);
	BIND_ENUM_CONSTANT(PARAM_SHADOW_SPLIT_2_OFFSET);
	BIND_ENUM_CONSTANT(PARAM_SHADOW_SPLIT_3_OFFSET);
	BIND_ENUM_CONSTANT(PARAM_SHADOW_NORMAL_BIAS);
	BIND_ENUM_CONSTANT(PARAM_SHADOW_BIAS);
	BIND_ENUM_CONSTANT(PARAM_SHADOW_BIAS_SPLIT_SCALE);
	BIND_ENUM_CONSTANT(PARAM_MAX);

	BIND_ENUM_CONSTANT(BAKE_DISABLED);
	BIND_ENUM_CONSTANT(BAKE_INDIRECT);
	BIND_ENUM_CONSTANT(BAKE_ALL);
}

Light::Light(VisualServer::LightType p_type) {
	type = p_type;
	switch (p_type) {
		case VS::LIGHT_DIRECTIONAL:
			light = RID_PRIME(VisualServer::get_singleton()->directional_light_create());
			break;
		case VS::LIGHT_OMNI:
			light = RID_PRIME(VisualServer::get_singleton()->omni_light_create());
			break;
		case VS::LIGHT_SPOT:
			light = RID_PRIME(VisualServer::get_singleton()->spot_light_create());
			break;
		default: {
		};
	}

	VS::get_singleton()->instance_set_base(get_instance(), light);

	reverse_cull = false;
	bake_mode = BAKE_INDIRECT;

	editor_only = false;
	set_color(Color(1, 1, 1, 1));
	set_shadow(false);
	set_negative(false);
	set_cull_mask(0xFFFFFFFF);

	set_param(PARAM_ENERGY, 1);
	set_param(PARAM_INDIRECT_ENERGY, 1);
	set_param(PARAM_SIZE, 0);
	set_param(PARAM_SPECULAR, 0.5);
	set_param(PARAM_RANGE, 5);
	set_param(PARAM_ATTENUATION, 1);
	set_param(PARAM_SPOT_ANGLE, 45);
	set_param(PARAM_SPOT_ATTENUATION, 1);
	set_param(PARAM_CONTACT_SHADOW_SIZE, 0);
	set_param(PARAM_SHADOW_MAX_DISTANCE, 0);
	set_param(PARAM_SHADOW_SPLIT_1_OFFSET, 0.1);
	set_param(PARAM_SHADOW_SPLIT_2_OFFSET, 0.2);
	set_param(PARAM_SHADOW_SPLIT_3_OFFSET, 0.5);
	set_param(PARAM_SHADOW_NORMAL_BIAS, 0.0);
	set_param(PARAM_SHADOW_BIAS, 0.15);
	set_disable_scale(true);
}

Light::Light() {
	type = VisualServer::LIGHT_DIRECTIONAL;
	ERR_PRINT("Light should not be instanced directly; use the DirectionalLight, OmniLight or SpotLight subtypes instead.");
}

Light::~Light() {
	VS::get_singleton()->instance_set_base(get_instance(), RID());

	if (light.is_valid()) {
		VisualServer::get_singleton()->free(light);
	}
}
/////////////////////////////////////////

void DirectionalLight::set_shadow_mode(ShadowMode p_mode) {
	shadow_mode = p_mode;
	VS::get_singleton()->light_directional_set_shadow_mode(light, VS::LightDirectionalShadowMode(p_mode));
	property_list_changed_notify();
}

DirectionalLight::ShadowMode DirectionalLight::get_shadow_mode() const {
	return shadow_mode;
}

void DirectionalLight::set_shadow_depth_range(ShadowDepthRange p_range) {
	shadow_depth_range = p_range;
	VS::get_singleton()->light_directional_set_shadow_depth_range_mode(light, VS::LightDirectionalShadowDepthRangeMode(p_range));
}

DirectionalLight::ShadowDepthRange DirectionalLight::get_shadow_depth_range() const {
	return shadow_depth_range;
}

void DirectionalLight::set_blend_splits(bool p_enable) {
	blend_splits = p_enable;
	VS::get_singleton()->light_directional_set_blend_splits(light, p_enable);
}

bool DirectionalLight::is_blend_splits_enabled() const {
	return blend_splits;
}

void DirectionalLight::_validate_property(PropertyInfo &property) const {
	if (shadow_mode == SHADOW_ORTHOGONAL && (property.name == "directional_shadow_split_1" || property.name == "directional_shadow_blend_splits" || property.name == "directional_shadow_bias_split_scale")) {
		// Split 2, split blending and bias split scale are only used with the PSSM 2 Splits and PSSM 4 Splits shadow modes.
		property.usage = PROPERTY_USAGE_NOEDITOR;
	}

	if ((shadow_mode == SHADOW_ORTHOGONAL || shadow_mode == SHADOW_PARALLEL_2_SPLITS) && (property.name == "directional_shadow_split_2" || property.name == "directional_shadow_split_3")) {
		// Splits 3 and 4 are only used with the PSSM 4 Splits shadow mode.
		property.usage = PROPERTY_USAGE_NOEDITOR;
	}

	Light::_validate_property(property);
}

void DirectionalLight::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_shadow_mode", "mode"), &DirectionalLight::set_shadow_mode);
	ClassDB::bind_method(D_METHOD("get_shadow_mode"), &DirectionalLight::get_shadow_mode);

	ClassDB::bind_method(D_METHOD("set_shadow_depth_range", "mode"), &DirectionalLight::set_shadow_depth_range);
	ClassDB::bind_method(D_METHOD("get_shadow_depth_range"), &DirectionalLight::get_shadow_depth_range);

	ClassDB::bind_method(D_METHOD("set_blend_splits", "enabled"), &DirectionalLight::set_blend_splits);
	ClassDB::bind_method(D_METHOD("is_blend_splits_enabled"), &DirectionalLight::is_blend_splits_enabled);

	ADD_GROUP("Directional Shadow", "directional_shadow_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "directional_shadow_mode", PROPERTY_HINT_ENUM, "Orthogonal (Fast),PSSM 2 Splits (Average),PSSM 4 Splits (Slow)"), "set_shadow_mode", "get_shadow_mode");
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "directional_shadow_split_1", PROPERTY_HINT_RANGE, "0,1,0.001"), "set_param", "get_param", PARAM_SHADOW_SPLIT_1_OFFSET);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "directional_shadow_split_2", PROPERTY_HINT_RANGE, "0,1,0.001"), "set_param", "get_param", PARAM_SHADOW_SPLIT_2_OFFSET);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "directional_shadow_split_3", PROPERTY_HINT_RANGE, "0,1,0.001"), "set_param", "get_param", PARAM_SHADOW_SPLIT_3_OFFSET);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "directional_shadow_blend_splits"), "set_blend_splits", "is_blend_splits_enabled");
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "directional_shadow_normal_bias", PROPERTY_HINT_RANGE, "0,10,0.001"), "set_param", "get_param", PARAM_SHADOW_NORMAL_BIAS);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "directional_shadow_bias_split_scale", PROPERTY_HINT_RANGE, "0,1,0.001"), "set_param", "get_param", PARAM_SHADOW_BIAS_SPLIT_SCALE);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "directional_shadow_depth_range", PROPERTY_HINT_ENUM, "Stable,Optimized"), "set_shadow_depth_range", "get_shadow_depth_range");
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "directional_shadow_max_distance", PROPERTY_HINT_EXP_RANGE, "0,8192,0.1,or_greater"), "set_param", "get_param", PARAM_SHADOW_MAX_DISTANCE);

	BIND_ENUM_CONSTANT(SHADOW_ORTHOGONAL);
	BIND_ENUM_CONSTANT(SHADOW_PARALLEL_2_SPLITS);
	BIND_ENUM_CONSTANT(SHADOW_PARALLEL_4_SPLITS);

	BIND_ENUM_CONSTANT(SHADOW_DEPTH_RANGE_STABLE);
	BIND_ENUM_CONSTANT(SHADOW_DEPTH_RANGE_OPTIMIZED);
}

DirectionalLight::DirectionalLight() :
		Light(VisualServer::LIGHT_DIRECTIONAL) {
	set_param(PARAM_SHADOW_NORMAL_BIAS, 0.8);
	set_param(PARAM_SHADOW_BIAS, 0.1);
	set_param(PARAM_SHADOW_MAX_DISTANCE, 100);
	set_param(PARAM_SHADOW_BIAS_SPLIT_SCALE, 0.25);
	set_shadow_mode(SHADOW_PARALLEL_4_SPLITS);
	set_shadow_depth_range(SHADOW_DEPTH_RANGE_STABLE);

	blend_splits = false;
}

void OmniLight::set_shadow_mode(ShadowMode p_mode) {
	shadow_mode = p_mode;
	VS::get_singleton()->light_omni_set_shadow_mode(light, VS::LightOmniShadowMode(p_mode));
}

OmniLight::ShadowMode OmniLight::get_shadow_mode() const {
	return shadow_mode;
}

void OmniLight::set_shadow_detail(ShadowDetail p_detail) {
	shadow_detail = p_detail;
	VS::get_singleton()->light_omni_set_shadow_detail(light, VS::LightOmniShadowDetail(p_detail));
}
OmniLight::ShadowDetail OmniLight::get_shadow_detail() const {
	return shadow_detail;
}

void OmniLight::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_shadow_mode", "mode"), &OmniLight::set_shadow_mode);
	ClassDB::bind_method(D_METHOD("get_shadow_mode"), &OmniLight::get_shadow_mode);

	ClassDB::bind_method(D_METHOD("set_shadow_detail", "detail"), &OmniLight::set_shadow_detail);
	ClassDB::bind_method(D_METHOD("get_shadow_detail"), &OmniLight::get_shadow_detail);

	ADD_GROUP("Omni", "omni_");
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "omni_range", PROPERTY_HINT_EXP_RANGE, "0,4096,0.001,or_greater"), "set_param", "get_param", PARAM_RANGE);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "omni_attenuation", PROPERTY_HINT_EXP_EASING, "attenuation"), "set_param", "get_param", PARAM_ATTENUATION);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "omni_shadow_mode", PROPERTY_HINT_ENUM, "Dual Paraboloid,Cube"), "set_shadow_mode", "get_shadow_mode");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "omni_shadow_detail", PROPERTY_HINT_ENUM, "Vertical,Horizontal"), "set_shadow_detail", "get_shadow_detail");

	BIND_ENUM_CONSTANT(SHADOW_DUAL_PARABOLOID);
	BIND_ENUM_CONSTANT(SHADOW_CUBE);

	BIND_ENUM_CONSTANT(SHADOW_DETAIL_VERTICAL);
	BIND_ENUM_CONSTANT(SHADOW_DETAIL_HORIZONTAL);
}

OmniLight::OmniLight() :
		Light(VisualServer::LIGHT_OMNI) {
	set_shadow_mode(SHADOW_CUBE);
	set_shadow_detail(SHADOW_DETAIL_HORIZONTAL);
}

String SpotLight::get_configuration_warning() const {
	String warning = Light::get_configuration_warning();

	if (has_shadow() && get_param(PARAM_SPOT_ANGLE) >= 90.0) {
		if (warning != String()) {
			warning += "\n\n";
		}

		warning += TTR("A SpotLight with an angle wider than 90 degrees cannot cast shadows.");
	}

	return warning;
}

void SpotLight::_bind_methods() {
	ADD_GROUP("Spot", "spot_");
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "spot_range", PROPERTY_HINT_EXP_RANGE, "0,4096,0.001,or_greater"), "set_param", "get_param", PARAM_RANGE);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "spot_attenuation", PROPERTY_HINT_EXP_EASING, "attenuation"), "set_param", "get_param", PARAM_ATTENUATION);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "spot_angle", PROPERTY_HINT_RANGE, "0,180,0.01"), "set_param", "get_param", PARAM_SPOT_ANGLE);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "spot_angle_attenuation", PROPERTY_HINT_EXP_EASING, "attenuation"), "set_param", "get_param", PARAM_SPOT_ATTENUATION);
}
