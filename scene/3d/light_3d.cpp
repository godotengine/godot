/*************************************************************************/
/*  light_3d.cpp                                                         */
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

#include "light_3d.h"

#include "core/config/engine.h"
#include "core/config/project_settings.h"
#include "scene/resources/surface_tool.h"

bool Light3D::_can_gizmo_scale() const {
	return false;
}

void Light3D::set_param(Param p_param, float p_value) {
	ERR_FAIL_INDEX(p_param, PARAM_MAX);
	param[p_param] = p_value;

	RS::get_singleton()->light_set_param(light, RS::LightParam(p_param), p_value);

	if (p_param == PARAM_SPOT_ANGLE || p_param == PARAM_RANGE) {
		update_gizmo();

		if (p_param == PARAM_SPOT_ANGLE) {
			update_configuration_warnings();
		}
	}
}

float Light3D::get_param(Param p_param) const {
	ERR_FAIL_INDEX_V(p_param, PARAM_MAX, 0);
	return param[p_param];
}

void Light3D::set_shadow(bool p_enable) {
	shadow = p_enable;
	RS::get_singleton()->light_set_shadow(light, p_enable);

	if (type == RenderingServer::LIGHT_SPOT || type == RenderingServer::LIGHT_OMNI) {
		update_configuration_warnings();
	}

	notify_property_list_changed();
}

bool Light3D::has_shadow() const {
	return shadow;
}

void Light3D::set_negative(bool p_enable) {
	negative = p_enable;
	RS::get_singleton()->light_set_negative(light, p_enable);
}

bool Light3D::is_negative() const {
	return negative;
}

void Light3D::set_cull_mask(uint32_t p_cull_mask) {
	cull_mask = p_cull_mask;
	RS::get_singleton()->light_set_cull_mask(light, p_cull_mask);
}

uint32_t Light3D::get_cull_mask() const {
	return cull_mask;
}

void Light3D::set_color(const Color &p_color) {
	color = p_color;
	RS::get_singleton()->light_set_color(light, p_color);
	// The gizmo color depends on the light color, so update it.
	update_gizmo();
}

Color Light3D::get_color() const {
	return color;
}

void Light3D::set_shadow_color(const Color &p_shadow_color) {
	shadow_color = p_shadow_color;
	RS::get_singleton()->light_set_shadow_color(light, p_shadow_color);
}

Color Light3D::get_shadow_color() const {
	return shadow_color;
}

void Light3D::set_shadow_reverse_cull_face(bool p_enable) {
	reverse_cull = p_enable;
	RS::get_singleton()->light_set_reverse_cull_face_mode(light, reverse_cull);
}

bool Light3D::get_shadow_reverse_cull_face() const {
	return reverse_cull;
}

AABB Light3D::get_aabb() const {
	if (type == RenderingServer::LIGHT_DIRECTIONAL) {
		return AABB(Vector3(-1, -1, -1), Vector3(2, 2, 2));

	} else if (type == RenderingServer::LIGHT_OMNI) {
		return AABB(Vector3(-1, -1, -1) * param[PARAM_RANGE], Vector3(2, 2, 2) * param[PARAM_RANGE]);

	} else if (type == RenderingServer::LIGHT_SPOT) {
		float len = param[PARAM_RANGE];
		float size = Math::tan(Math::deg2rad(param[PARAM_SPOT_ANGLE])) * len;
		return AABB(Vector3(-size, -size, -len), Vector3(size * 2, size * 2, len));
	}

	return AABB();
}

Vector<Face3> Light3D::get_faces(uint32_t p_usage_flags) const {
	return Vector<Face3>();
}

void Light3D::set_bake_mode(BakeMode p_mode) {
	bake_mode = p_mode;
	RS::get_singleton()->light_set_bake_mode(light, RS::LightBakeMode(p_mode));
}

Light3D::BakeMode Light3D::get_bake_mode() const {
	return bake_mode;
}

void Light3D::set_projector(const Ref<Texture2D> &p_texture) {
	projector = p_texture;
	RID tex_id = projector.is_valid() ? projector->get_rid() : RID();
	RS::get_singleton()->light_set_projector(light, tex_id);
	update_configuration_warnings();
}

Ref<Texture2D> Light3D::get_projector() const {
	return projector;
}

void Light3D::_update_visibility() {
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

	RS::get_singleton()->instance_set_visible(get_instance(), is_visible_in_tree() && editor_ok);
}

void Light3D::_notification(int p_what) {
	if (p_what == NOTIFICATION_VISIBILITY_CHANGED) {
		_update_visibility();
	}

	if (p_what == NOTIFICATION_ENTER_TREE) {
		_update_visibility();
	}
}

void Light3D::set_editor_only(bool p_editor_only) {
	editor_only = p_editor_only;
	_update_visibility();
}

bool Light3D::is_editor_only() const {
	return editor_only;
}

void Light3D::_validate_property(PropertyInfo &property) const {
	if (!shadow && (property.name == "shadow_color" || property.name == "shadow_bias" || property.name == "shadow_normal_bias" || property.name == "shadow_reverse_cull_face" || property.name == "shadow_transmittance_bias" || property.name == "shadow_fog_fade" || property.name == "shadow_blur")) {
		property.usage = PROPERTY_USAGE_NOEDITOR;
	}

	if (get_light_type() == RS::LIGHT_DIRECTIONAL && property.name == "light_size") {
		property.usage = 0;
	}

	if (get_light_type() == RS::LIGHT_DIRECTIONAL && property.name == "light_specular") {
		property.usage = 0;
	}

	if (get_light_type() == RS::LIGHT_DIRECTIONAL && property.name == "light_projector") {
		property.usage = 0;
	}

	if (get_light_type() != RS::LIGHT_DIRECTIONAL && property.name == "light_angular_distance") {
		property.usage = 0;
	}
}

void Light3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_editor_only", "editor_only"), &Light3D::set_editor_only);
	ClassDB::bind_method(D_METHOD("is_editor_only"), &Light3D::is_editor_only);

	ClassDB::bind_method(D_METHOD("set_param", "param", "value"), &Light3D::set_param);
	ClassDB::bind_method(D_METHOD("get_param", "param"), &Light3D::get_param);

	ClassDB::bind_method(D_METHOD("set_shadow", "enabled"), &Light3D::set_shadow);
	ClassDB::bind_method(D_METHOD("has_shadow"), &Light3D::has_shadow);

	ClassDB::bind_method(D_METHOD("set_negative", "enabled"), &Light3D::set_negative);
	ClassDB::bind_method(D_METHOD("is_negative"), &Light3D::is_negative);

	ClassDB::bind_method(D_METHOD("set_cull_mask", "cull_mask"), &Light3D::set_cull_mask);
	ClassDB::bind_method(D_METHOD("get_cull_mask"), &Light3D::get_cull_mask);

	ClassDB::bind_method(D_METHOD("set_color", "color"), &Light3D::set_color);
	ClassDB::bind_method(D_METHOD("get_color"), &Light3D::get_color);

	ClassDB::bind_method(D_METHOD("set_shadow_reverse_cull_face", "enable"), &Light3D::set_shadow_reverse_cull_face);
	ClassDB::bind_method(D_METHOD("get_shadow_reverse_cull_face"), &Light3D::get_shadow_reverse_cull_face);

	ClassDB::bind_method(D_METHOD("set_shadow_color", "shadow_color"), &Light3D::set_shadow_color);
	ClassDB::bind_method(D_METHOD("get_shadow_color"), &Light3D::get_shadow_color);

	ClassDB::bind_method(D_METHOD("set_bake_mode", "bake_mode"), &Light3D::set_bake_mode);
	ClassDB::bind_method(D_METHOD("get_bake_mode"), &Light3D::get_bake_mode);

	ClassDB::bind_method(D_METHOD("set_projector", "projector"), &Light3D::set_projector);
	ClassDB::bind_method(D_METHOD("get_projector"), &Light3D::get_projector);

	ADD_GROUP("Light", "light_");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "light_color", PROPERTY_HINT_COLOR_NO_ALPHA), "set_color", "get_color");
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "light_energy", PROPERTY_HINT_RANGE, "0,16,0.01,or_greater"), "set_param", "get_param", PARAM_ENERGY);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "light_indirect_energy", PROPERTY_HINT_RANGE, "0,16,0.01,or_greater"), "set_param", "get_param", PARAM_INDIRECT_ENERGY);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "light_projector", PROPERTY_HINT_RESOURCE_TYPE, "Texture2D"), "set_projector", "get_projector");
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "light_size", PROPERTY_HINT_RANGE, "0,1,0.01,or_greater"), "set_param", "get_param", PARAM_SIZE);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "light_angular_distance", PROPERTY_HINT_RANGE, "0,90,0.01"), "set_param", "get_param", PARAM_SIZE);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "light_negative"), "set_negative", "is_negative");
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "light_specular", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_param", "get_param", PARAM_SPECULAR);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "light_bake_mode", PROPERTY_HINT_ENUM, "Disabled,Dynamic,Static"), "set_bake_mode", "get_bake_mode");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "light_cull_mask", PROPERTY_HINT_LAYERS_3D_RENDER), "set_cull_mask", "get_cull_mask");
	ADD_GROUP("Shadow", "shadow_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "shadow_enabled"), "set_shadow", "has_shadow");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "shadow_color", PROPERTY_HINT_COLOR_NO_ALPHA), "set_shadow_color", "get_shadow_color");
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "shadow_bias", PROPERTY_HINT_RANGE, "0,10,0.001"), "set_param", "get_param", PARAM_SHADOW_BIAS);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "shadow_normal_bias", PROPERTY_HINT_RANGE, "0,10,0.001"), "set_param", "get_param", PARAM_SHADOW_NORMAL_BIAS);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "shadow_reverse_cull_face"), "set_shadow_reverse_cull_face", "get_shadow_reverse_cull_face");
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "shadow_transmittance_bias", PROPERTY_HINT_RANGE, "-16,16,0.01"), "set_param", "get_param", PARAM_TRANSMITTANCE_BIAS);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "shadow_fog_fade", PROPERTY_HINT_RANGE, "0.01,10,0.01"), "set_param", "get_param", PARAM_SHADOW_VOLUMETRIC_FOG_FADE);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "shadow_blur", PROPERTY_HINT_RANGE, "0.1,8,0.01"), "set_param", "get_param", PARAM_SHADOW_BLUR);
	ADD_GROUP("Editor", "");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "editor_only"), "set_editor_only", "is_editor_only");
	ADD_GROUP("", "");

	BIND_ENUM_CONSTANT(PARAM_ENERGY);
	BIND_ENUM_CONSTANT(PARAM_INDIRECT_ENERGY);
	BIND_ENUM_CONSTANT(PARAM_SPECULAR);
	BIND_ENUM_CONSTANT(PARAM_RANGE);
	BIND_ENUM_CONSTANT(PARAM_SIZE);
	BIND_ENUM_CONSTANT(PARAM_ATTENUATION);
	BIND_ENUM_CONSTANT(PARAM_SPOT_ANGLE);
	BIND_ENUM_CONSTANT(PARAM_SPOT_ATTENUATION);
	BIND_ENUM_CONSTANT(PARAM_SHADOW_MAX_DISTANCE);
	BIND_ENUM_CONSTANT(PARAM_SHADOW_SPLIT_1_OFFSET);
	BIND_ENUM_CONSTANT(PARAM_SHADOW_SPLIT_2_OFFSET);
	BIND_ENUM_CONSTANT(PARAM_SHADOW_SPLIT_3_OFFSET);
	BIND_ENUM_CONSTANT(PARAM_SHADOW_FADE_START);
	BIND_ENUM_CONSTANT(PARAM_SHADOW_NORMAL_BIAS);
	BIND_ENUM_CONSTANT(PARAM_SHADOW_BIAS);
	BIND_ENUM_CONSTANT(PARAM_SHADOW_PANCAKE_SIZE);
	BIND_ENUM_CONSTANT(PARAM_SHADOW_BLUR);
	BIND_ENUM_CONSTANT(PARAM_SHADOW_VOLUMETRIC_FOG_FADE);
	BIND_ENUM_CONSTANT(PARAM_TRANSMITTANCE_BIAS);
	BIND_ENUM_CONSTANT(PARAM_MAX);

	BIND_ENUM_CONSTANT(BAKE_DISABLED);
	BIND_ENUM_CONSTANT(BAKE_DYNAMIC);
	BIND_ENUM_CONSTANT(BAKE_STATIC);
}

Light3D::Light3D(RenderingServer::LightType p_type) {
	type = p_type;
	switch (p_type) {
		case RS::LIGHT_DIRECTIONAL:
			light = RenderingServer::get_singleton()->directional_light_create();
			break;
		case RS::LIGHT_OMNI:
			light = RenderingServer::get_singleton()->omni_light_create();
			break;
		case RS::LIGHT_SPOT:
			light = RenderingServer::get_singleton()->spot_light_create();
			break;
		default: {
		};
	}

	RS::get_singleton()->instance_set_base(get_instance(), light);

	set_color(Color(1, 1, 1, 1));
	set_shadow(false);
	set_negative(false);
	set_cull_mask(0xFFFFFFFF);

	set_param(PARAM_ENERGY, 1);
	set_param(PARAM_INDIRECT_ENERGY, 1);
	set_param(PARAM_SPECULAR, 0.5);
	set_param(PARAM_RANGE, 5);
	set_param(PARAM_SIZE, 0);
	set_param(PARAM_ATTENUATION, 1);
	set_param(PARAM_SPOT_ANGLE, 45);
	set_param(PARAM_SPOT_ATTENUATION, 1);
	set_param(PARAM_SHADOW_MAX_DISTANCE, 0);
	set_param(PARAM_SHADOW_SPLIT_1_OFFSET, 0.1);
	set_param(PARAM_SHADOW_SPLIT_2_OFFSET, 0.2);
	set_param(PARAM_SHADOW_SPLIT_3_OFFSET, 0.5);
	set_param(PARAM_SHADOW_FADE_START, 0.8);
	set_param(PARAM_SHADOW_PANCAKE_SIZE, 20.0);
	set_param(PARAM_SHADOW_BLUR, 1.0);
	set_param(PARAM_SHADOW_BIAS, 0.02);
	set_param(PARAM_SHADOW_NORMAL_BIAS, 1.0);
	set_param(PARAM_TRANSMITTANCE_BIAS, 0.05);
	set_param(PARAM_SHADOW_VOLUMETRIC_FOG_FADE, 0.1);
	set_param(PARAM_SHADOW_FADE_START, 1);
	set_disable_scale(true);
}

Light3D::Light3D() {
	ERR_PRINT("Light3D should not be instanced directly; use the DirectionalLight3D, OmniLight3D or SpotLight3D subtypes instead.");
}

Light3D::~Light3D() {
	RS::get_singleton()->instance_set_base(get_instance(), RID());

	if (light.is_valid()) {
		RenderingServer::get_singleton()->free(light);
	}
}

/////////////////////////////////////////

void DirectionalLight3D::set_shadow_mode(ShadowMode p_mode) {
	shadow_mode = p_mode;
	RS::get_singleton()->light_directional_set_shadow_mode(light, RS::LightDirectionalShadowMode(p_mode));
}

DirectionalLight3D::ShadowMode DirectionalLight3D::get_shadow_mode() const {
	return shadow_mode;
}

void DirectionalLight3D::set_shadow_depth_range(ShadowDepthRange p_range) {
	shadow_depth_range = p_range;
	RS::get_singleton()->light_directional_set_shadow_depth_range_mode(light, RS::LightDirectionalShadowDepthRangeMode(p_range));
}

DirectionalLight3D::ShadowDepthRange DirectionalLight3D::get_shadow_depth_range() const {
	return shadow_depth_range;
}

void DirectionalLight3D::set_blend_splits(bool p_enable) {
	blend_splits = p_enable;
	RS::get_singleton()->light_directional_set_blend_splits(light, p_enable);
}

bool DirectionalLight3D::is_blend_splits_enabled() const {
	return blend_splits;
}

void DirectionalLight3D::set_sky_only(bool p_sky_only) {
	sky_only = p_sky_only;
	RS::get_singleton()->light_directional_set_sky_only(light, p_sky_only);
}

bool DirectionalLight3D::is_sky_only() const {
	return sky_only;
}

void DirectionalLight3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_shadow_mode", "mode"), &DirectionalLight3D::set_shadow_mode);
	ClassDB::bind_method(D_METHOD("get_shadow_mode"), &DirectionalLight3D::get_shadow_mode);

	ClassDB::bind_method(D_METHOD("set_shadow_depth_range", "mode"), &DirectionalLight3D::set_shadow_depth_range);
	ClassDB::bind_method(D_METHOD("get_shadow_depth_range"), &DirectionalLight3D::get_shadow_depth_range);

	ClassDB::bind_method(D_METHOD("set_blend_splits", "enabled"), &DirectionalLight3D::set_blend_splits);
	ClassDB::bind_method(D_METHOD("is_blend_splits_enabled"), &DirectionalLight3D::is_blend_splits_enabled);

	ClassDB::bind_method(D_METHOD("set_sky_only", "priority"), &DirectionalLight3D::set_sky_only);
	ClassDB::bind_method(D_METHOD("is_sky_only"), &DirectionalLight3D::is_sky_only);

	ADD_GROUP("Directional Shadow", "directional_shadow_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "directional_shadow_mode", PROPERTY_HINT_ENUM, "Orthogonal (Fast),PSSM 2 Splits (Average),PSSM 4 Splits (Slow)"), "set_shadow_mode", "get_shadow_mode");
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "directional_shadow_split_1", PROPERTY_HINT_RANGE, "0,1,0.001"), "set_param", "get_param", PARAM_SHADOW_SPLIT_1_OFFSET);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "directional_shadow_split_2", PROPERTY_HINT_RANGE, "0,1,0.001"), "set_param", "get_param", PARAM_SHADOW_SPLIT_2_OFFSET);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "directional_shadow_split_3", PROPERTY_HINT_RANGE, "0,1,0.001"), "set_param", "get_param", PARAM_SHADOW_SPLIT_3_OFFSET);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "directional_shadow_fade_start", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_param", "get_param", PARAM_SHADOW_FADE_START);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "directional_shadow_blend_splits"), "set_blend_splits", "is_blend_splits_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "directional_shadow_depth_range", PROPERTY_HINT_ENUM, "Stable,Optimized"), "set_shadow_depth_range", "get_shadow_depth_range");
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "directional_shadow_max_distance", PROPERTY_HINT_EXP_RANGE, "0,8192,0.1,or_greater"), "set_param", "get_param", PARAM_SHADOW_MAX_DISTANCE);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "directional_shadow_pancake_size", PROPERTY_HINT_EXP_RANGE, "0,1024,0.1,or_greater"), "set_param", "get_param", PARAM_SHADOW_PANCAKE_SIZE);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "use_in_sky_only"), "set_sky_only", "is_sky_only");

	BIND_ENUM_CONSTANT(SHADOW_ORTHOGONAL);
	BIND_ENUM_CONSTANT(SHADOW_PARALLEL_2_SPLITS);
	BIND_ENUM_CONSTANT(SHADOW_PARALLEL_4_SPLITS);

	BIND_ENUM_CONSTANT(SHADOW_DEPTH_RANGE_STABLE);
	BIND_ENUM_CONSTANT(SHADOW_DEPTH_RANGE_OPTIMIZED);
}

DirectionalLight3D::DirectionalLight3D() :
		Light3D(RenderingServer::LIGHT_DIRECTIONAL) {
	set_param(PARAM_SHADOW_MAX_DISTANCE, 100);
	set_param(PARAM_SHADOW_FADE_START, 0.8);
	// Increase the default shadow bias to better suit most scenes.
	// Leave normal bias untouched as it doesn't benefit DirectionalLight3D as much as OmniLight3D.
	set_param(PARAM_SHADOW_BIAS, 0.05);
	set_shadow_mode(SHADOW_PARALLEL_4_SPLITS);
	set_shadow_depth_range(SHADOW_DEPTH_RANGE_STABLE);
	blend_splits = false;
}

void OmniLight3D::set_shadow_mode(ShadowMode p_mode) {
	shadow_mode = p_mode;
	RS::get_singleton()->light_omni_set_shadow_mode(light, RS::LightOmniShadowMode(p_mode));
}

OmniLight3D::ShadowMode OmniLight3D::get_shadow_mode() const {
	return shadow_mode;
}

TypedArray<String> OmniLight3D::get_configuration_warnings() const {
	TypedArray<String> warnings = Node::get_configuration_warnings();

	if (!has_shadow() && get_projector().is_valid()) {
		warnings.push_back(TTR("Projector texture only works with shadows active."));
	}

	return warnings;
}

void OmniLight3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_shadow_mode", "mode"), &OmniLight3D::set_shadow_mode);
	ClassDB::bind_method(D_METHOD("get_shadow_mode"), &OmniLight3D::get_shadow_mode);

	ADD_GROUP("Omni", "omni_");
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "omni_range", PROPERTY_HINT_EXP_RANGE, "0,4096,0.1,or_greater"), "set_param", "get_param", PARAM_RANGE);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "omni_attenuation", PROPERTY_HINT_EXP_EASING, "attenuation"), "set_param", "get_param", PARAM_ATTENUATION);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "omni_shadow_mode", PROPERTY_HINT_ENUM, "Dual Paraboloid,Cube"), "set_shadow_mode", "get_shadow_mode");

	BIND_ENUM_CONSTANT(SHADOW_DUAL_PARABOLOID);
	BIND_ENUM_CONSTANT(SHADOW_CUBE);
}

OmniLight3D::OmniLight3D() :
		Light3D(RenderingServer::LIGHT_OMNI) {
	set_shadow_mode(SHADOW_CUBE);
	// Increase the default shadow biases to better suit most scenes.
	set_param(PARAM_SHADOW_BIAS, 0.1);
	set_param(PARAM_SHADOW_NORMAL_BIAS, 2.0);
}

TypedArray<String> SpotLight3D::get_configuration_warnings() const {
	TypedArray<String> warnings = Node::get_configuration_warnings();

	if (has_shadow() && get_param(PARAM_SPOT_ANGLE) >= 90.0) {
		warnings.push_back(TTR("A SpotLight3D with an angle wider than 90 degrees cannot cast shadows."));
	}

	if (!has_shadow() && get_projector().is_valid()) {
		warnings.push_back(TTR("Projector texture only works with shadows active."));
	}

	return warnings;
}

void SpotLight3D::_bind_methods() {
	ADD_GROUP("Spot", "spot_");
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "spot_range", PROPERTY_HINT_EXP_RANGE, "0,4096,0.1,or_greater"), "set_param", "get_param", PARAM_RANGE);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "spot_attenuation", PROPERTY_HINT_EXP_EASING, "attenuation"), "set_param", "get_param", PARAM_ATTENUATION);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "spot_angle", PROPERTY_HINT_RANGE, "0,180,0.1"), "set_param", "get_param", PARAM_SPOT_ANGLE);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "spot_angle_attenuation", PROPERTY_HINT_EXP_EASING, "attenuation"), "set_param", "get_param", PARAM_SPOT_ATTENUATION);
}
