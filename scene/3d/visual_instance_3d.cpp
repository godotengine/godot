/*************************************************************************/
/*  visual_instance_3d.cpp                                               */
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

#include "visual_instance_3d.h"

#include "scene/scene_string_names.h"

AABB VisualInstance3D::get_transformed_aabb() const {
	return get_global_transform().xform(get_aabb());
}

void VisualInstance3D::_update_visibility() {
	if (!is_inside_tree()) {
		return;
	}

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
			ERR_FAIL_COND(get_world_3d().is_null());
			RenderingServer::get_singleton()->instance_set_scenario(instance, get_world_3d()->get_scenario());
			_update_visibility();

		} break;
		case NOTIFICATION_TRANSFORM_CHANGED: {
			Transform3D gt = get_global_transform();
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

void VisualInstance3D::set_layer_mask_value(int p_layer_number, bool p_value) {
	ERR_FAIL_COND_MSG(p_layer_number < 1, "Render layer number must be between 1 and 20 inclusive.");
	ERR_FAIL_COND_MSG(p_layer_number > 20, "Render layer number must be between 1 and 20 inclusive.");
	uint32_t mask = get_layer_mask();
	if (p_value) {
		mask |= 1 << (p_layer_number - 1);
	} else {
		mask &= ~(1 << (p_layer_number - 1));
	}
	set_layer_mask(mask);
}

bool VisualInstance3D::get_layer_mask_value(int p_layer_number) const {
	ERR_FAIL_COND_V_MSG(p_layer_number < 1, false, "Render layer number must be between 1 and 20 inclusive.");
	ERR_FAIL_COND_V_MSG(p_layer_number > 20, false, "Render layer number must be between 1 and 20 inclusive.");
	return layers & (1 << (p_layer_number - 1));
}

void VisualInstance3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_get_visual_instance_rid"), &VisualInstance3D::_get_visual_instance_rid);
	ClassDB::bind_method(D_METHOD("set_base", "base"), &VisualInstance3D::set_base);
	ClassDB::bind_method(D_METHOD("get_base"), &VisualInstance3D::get_base);
	ClassDB::bind_method(D_METHOD("get_instance"), &VisualInstance3D::get_instance);
	ClassDB::bind_method(D_METHOD("set_layer_mask", "mask"), &VisualInstance3D::set_layer_mask);
	ClassDB::bind_method(D_METHOD("get_layer_mask"), &VisualInstance3D::get_layer_mask);
	ClassDB::bind_method(D_METHOD("set_layer_mask_value", "layer_number", "value"), &VisualInstance3D::set_layer_mask_value);
	ClassDB::bind_method(D_METHOD("get_layer_mask_value", "layer_number"), &VisualInstance3D::get_layer_mask_value);

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

void GeometryInstance3D::set_material_overlay(const Ref<Material> &p_material) {
	material_overlay = p_material;
	RS::get_singleton()->instance_geometry_set_material_overlay(get_instance(), p_material.is_valid() ? p_material->get_rid() : RID());
}

Ref<Material> GeometryInstance3D::get_material_overlay() const {
	return material_overlay;
}

void GeometryInstance3D::set_transparecy(float p_transparency) {
	transparency = CLAMP(p_transparency, 0.0f, 1.0f);
	RS::get_singleton()->instance_geometry_set_transparency(get_instance(), transparency);
}

float GeometryInstance3D::get_transparency() const {
	return transparency;
}

void GeometryInstance3D::set_visibility_range_begin(float p_dist) {
	visibility_range_begin = p_dist;
	RS::get_singleton()->instance_geometry_set_visibility_range(get_instance(), visibility_range_begin, visibility_range_end, visibility_range_begin_margin, visibility_range_end_margin, (RS::VisibilityRangeFadeMode)visibility_range_fade_mode);
	update_configuration_warnings();
}

float GeometryInstance3D::get_visibility_range_begin() const {
	return visibility_range_begin;
}

void GeometryInstance3D::set_visibility_range_end(float p_dist) {
	visibility_range_end = p_dist;
	RS::get_singleton()->instance_geometry_set_visibility_range(get_instance(), visibility_range_begin, visibility_range_end, visibility_range_begin_margin, visibility_range_end_margin, (RS::VisibilityRangeFadeMode)visibility_range_fade_mode);
	update_configuration_warnings();
}

float GeometryInstance3D::get_visibility_range_end() const {
	return visibility_range_end;
}

void GeometryInstance3D::set_visibility_range_begin_margin(float p_dist) {
	visibility_range_begin_margin = p_dist;
	RS::get_singleton()->instance_geometry_set_visibility_range(get_instance(), visibility_range_begin, visibility_range_end, visibility_range_begin_margin, visibility_range_end_margin, (RS::VisibilityRangeFadeMode)visibility_range_fade_mode);
	update_configuration_warnings();
}

float GeometryInstance3D::get_visibility_range_begin_margin() const {
	return visibility_range_begin_margin;
}

void GeometryInstance3D::set_visibility_range_end_margin(float p_dist) {
	visibility_range_end_margin = p_dist;
	RS::get_singleton()->instance_geometry_set_visibility_range(get_instance(), visibility_range_begin, visibility_range_end, visibility_range_begin_margin, visibility_range_end_margin, (RS::VisibilityRangeFadeMode)visibility_range_fade_mode);
	update_configuration_warnings();
}

float GeometryInstance3D::get_visibility_range_end_margin() const {
	return visibility_range_end_margin;
}

void GeometryInstance3D::set_visibility_range_fade_mode(VisibilityRangeFadeMode p_mode) {
	visibility_range_fade_mode = p_mode;
	RS::get_singleton()->instance_geometry_set_visibility_range(get_instance(), visibility_range_begin, visibility_range_end, visibility_range_begin_margin, visibility_range_end_margin, (RS::VisibilityRangeFadeMode)visibility_range_fade_mode);
	update_configuration_warnings();
}

GeometryInstance3D::VisibilityRangeFadeMode GeometryInstance3D::get_visibility_range_fade_mode() const {
	return visibility_range_fade_mode;
}

void GeometryInstance3D::_notification(int p_what) {
}

const StringName *GeometryInstance3D::_instance_uniform_get_remap(const StringName p_name) const {
	StringName *r = instance_uniform_property_remap.getptr(p_name);
	if (!r) {
		String s = p_name;
		if (s.begins_with("shader_params/")) {
			StringName name = s.replace("shader_params/", "");
			instance_uniform_property_remap[p_name] = name;
			return instance_uniform_property_remap.getptr(p_name);
		}

		return nullptr;
	}

	return r;
}

bool GeometryInstance3D::_set(const StringName &p_name, const Variant &p_value) {
	const StringName *r = _instance_uniform_get_remap(p_name);
	if (r) {
		set_shader_instance_uniform(*r, p_value);
		return true;
	}
#ifndef DISABLE_DEPRECATED
	if (p_name == SceneStringNames::get_singleton()->use_in_baked_light && bool(p_value)) {
		set_gi_mode(GI_MODE_STATIC);
		return true;
	}

	if (p_name == SceneStringNames::get_singleton()->use_dynamic_gi && bool(p_value)) {
		set_gi_mode(GI_MODE_DYNAMIC);
		return true;
	}
#endif
	return false;
}

bool GeometryInstance3D::_get(const StringName &p_name, Variant &r_ret) const {
	const StringName *r = _instance_uniform_get_remap(p_name);
	if (r) {
		r_ret = get_shader_instance_uniform(*r);
		return true;
	}

	return false;
}

void GeometryInstance3D::_get_property_list(List<PropertyInfo> *p_list) const {
	List<PropertyInfo> pinfo;
	RS::get_singleton()->instance_geometry_get_shader_parameter_list(get_instance(), &pinfo);
	for (PropertyInfo &pi : pinfo) {
		bool has_def_value = false;
		Variant def_value = RS::get_singleton()->instance_geometry_get_shader_parameter_default_value(get_instance(), pi.name);
		if (def_value.get_type() != Variant::NIL) {
			has_def_value = true;
		}
		if (instance_uniforms.has(pi.name)) {
			pi.usage = PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_STORAGE | (has_def_value ? (PROPERTY_USAGE_CHECKABLE | PROPERTY_USAGE_CHECKED) : PROPERTY_USAGE_NONE);
		} else {
			pi.usage = PROPERTY_USAGE_EDITOR | (has_def_value ? PROPERTY_USAGE_CHECKABLE : PROPERTY_USAGE_NONE); //do not save if not changed
		}

		pi.name = "shader_params/" + pi.name;
		p_list->push_back(pi);
	}
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

void GeometryInstance3D::set_lod_bias(float p_bias) {
	ERR_FAIL_COND(p_bias < 0.0);
	lod_bias = p_bias;
	RS::get_singleton()->instance_geometry_set_lod_bias(get_instance(), lod_bias);
}

float GeometryInstance3D::get_lod_bias() const {
	return lod_bias;
}

void GeometryInstance3D::set_shader_instance_uniform(const StringName &p_uniform, const Variant &p_value) {
	if (p_value.get_type() == Variant::NIL) {
		Variant def_value = RS::get_singleton()->instance_geometry_get_shader_parameter_default_value(get_instance(), p_uniform);
		RS::get_singleton()->instance_geometry_set_shader_parameter(get_instance(), p_uniform, def_value);
		instance_uniforms.erase(p_value);
	} else {
		instance_uniforms[p_uniform] = p_value;
		if (p_value.get_type() == Variant::OBJECT) {
			RID tex_id = p_value;
			RS::get_singleton()->instance_geometry_set_shader_parameter(get_instance(), p_uniform, tex_id);
		} else {
			RS::get_singleton()->instance_geometry_set_shader_parameter(get_instance(), p_uniform, p_value);
		}
	}
}

Variant GeometryInstance3D::get_shader_instance_uniform(const StringName &p_uniform) const {
	return RS::get_singleton()->instance_geometry_get_shader_parameter(get_instance(), p_uniform);
}

void GeometryInstance3D::set_custom_aabb(AABB aabb) {
	RS::get_singleton()->instance_set_custom_aabb(get_instance(), aabb);
}

void GeometryInstance3D::set_lightmap_scale(LightmapScale p_scale) {
	ERR_FAIL_INDEX(p_scale, LIGHTMAP_SCALE_MAX);
	lightmap_scale = p_scale;
}

GeometryInstance3D::LightmapScale GeometryInstance3D::get_lightmap_scale() const {
	return lightmap_scale;
}

void GeometryInstance3D::set_gi_mode(GIMode p_mode) {
	switch (p_mode) {
		case GI_MODE_DISABLED: {
			RS::get_singleton()->instance_geometry_set_flag(get_instance(), RS::INSTANCE_FLAG_USE_BAKED_LIGHT, false);
			RS::get_singleton()->instance_geometry_set_flag(get_instance(), RS::INSTANCE_FLAG_USE_DYNAMIC_GI, false);
		} break;
		case GI_MODE_STATIC: {
			RS::get_singleton()->instance_geometry_set_flag(get_instance(), RS::INSTANCE_FLAG_USE_BAKED_LIGHT, true);
			RS::get_singleton()->instance_geometry_set_flag(get_instance(), RS::INSTANCE_FLAG_USE_DYNAMIC_GI, false);

		} break;
		case GI_MODE_DYNAMIC: {
			RS::get_singleton()->instance_geometry_set_flag(get_instance(), RS::INSTANCE_FLAG_USE_BAKED_LIGHT, false);
			RS::get_singleton()->instance_geometry_set_flag(get_instance(), RS::INSTANCE_FLAG_USE_DYNAMIC_GI, true);
		} break;
	}

	gi_mode = p_mode;
}

GeometryInstance3D::GIMode GeometryInstance3D::get_gi_mode() const {
	return gi_mode;
}

void GeometryInstance3D::set_ignore_occlusion_culling(bool p_enabled) {
	ignore_occlusion_culling = p_enabled;
	RS::get_singleton()->instance_geometry_set_flag(get_instance(), RS::INSTANCE_FLAG_IGNORE_OCCLUSION_CULLING, ignore_occlusion_culling);
}

bool GeometryInstance3D::is_ignoring_occlusion_culling() {
	return ignore_occlusion_culling;
}

TypedArray<String> GeometryInstance3D::get_configuration_warnings() const {
	TypedArray<String> warnings = Node::get_configuration_warnings();

	if (!Math::is_zero_approx(visibility_range_end) && visibility_range_end <= visibility_range_begin) {
		warnings.push_back(TTR("The GeometryInstance3D visibility range's End distance is set to a non-zero value, but is lower than the Begin distance.\nThis means the GeometryInstance3D will never be visible.\nTo resolve this, set the End distance to 0 or to a value greater than the Begin distance."));
	}

	if ((visibility_range_fade_mode == VISIBILITY_RANGE_FADE_SELF || visibility_range_fade_mode == VISIBILITY_RANGE_FADE_DEPENDENCIES) && !Math::is_zero_approx(visibility_range_begin) && Math::is_zero_approx(visibility_range_begin_margin)) {
		warnings.push_back(TTR("The GeometryInstance3D is configured to fade in smoothly over distance, but the fade transition distance is set to 0.\nTo resolve this, increase Visibility Range Begin Margin above 0."));
	}

	if ((visibility_range_fade_mode == VISIBILITY_RANGE_FADE_SELF || visibility_range_fade_mode == VISIBILITY_RANGE_FADE_DEPENDENCIES) && !Math::is_zero_approx(visibility_range_end) && Math::is_zero_approx(visibility_range_end_margin)) {
		warnings.push_back(TTR("The GeometryInstance3D is configured to fade out smoothly over distance, but the fade transition distance is set to 0.\nTo resolve this, increase Visibility Range End Margin above 0."));
	}

	return warnings;
}

void GeometryInstance3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_material_override", "material"), &GeometryInstance3D::set_material_override);
	ClassDB::bind_method(D_METHOD("get_material_override"), &GeometryInstance3D::get_material_override);

	ClassDB::bind_method(D_METHOD("set_material_overlay", "material"), &GeometryInstance3D::set_material_overlay);
	ClassDB::bind_method(D_METHOD("get_material_overlay"), &GeometryInstance3D::get_material_overlay);

	ClassDB::bind_method(D_METHOD("set_cast_shadows_setting", "shadow_casting_setting"), &GeometryInstance3D::set_cast_shadows_setting);
	ClassDB::bind_method(D_METHOD("get_cast_shadows_setting"), &GeometryInstance3D::get_cast_shadows_setting);

	ClassDB::bind_method(D_METHOD("set_lod_bias", "bias"), &GeometryInstance3D::set_lod_bias);
	ClassDB::bind_method(D_METHOD("get_lod_bias"), &GeometryInstance3D::get_lod_bias);

	ClassDB::bind_method(D_METHOD("set_transparency", "transparency"), &GeometryInstance3D::set_transparecy);
	ClassDB::bind_method(D_METHOD("get_transparency"), &GeometryInstance3D::get_transparency);

	ClassDB::bind_method(D_METHOD("set_visibility_range_end_margin", "distance"), &GeometryInstance3D::set_visibility_range_end_margin);
	ClassDB::bind_method(D_METHOD("get_visibility_range_end_margin"), &GeometryInstance3D::get_visibility_range_end_margin);

	ClassDB::bind_method(D_METHOD("set_visibility_range_end", "distance"), &GeometryInstance3D::set_visibility_range_end);
	ClassDB::bind_method(D_METHOD("get_visibility_range_end"), &GeometryInstance3D::get_visibility_range_end);

	ClassDB::bind_method(D_METHOD("set_visibility_range_begin_margin", "distance"), &GeometryInstance3D::set_visibility_range_begin_margin);
	ClassDB::bind_method(D_METHOD("get_visibility_range_begin_margin"), &GeometryInstance3D::get_visibility_range_begin_margin);

	ClassDB::bind_method(D_METHOD("set_visibility_range_begin", "distance"), &GeometryInstance3D::set_visibility_range_begin);
	ClassDB::bind_method(D_METHOD("get_visibility_range_begin"), &GeometryInstance3D::get_visibility_range_begin);

	ClassDB::bind_method(D_METHOD("set_visibility_range_fade_mode", "mode"), &GeometryInstance3D::set_visibility_range_fade_mode);
	ClassDB::bind_method(D_METHOD("get_visibility_range_fade_mode"), &GeometryInstance3D::get_visibility_range_fade_mode);

	ClassDB::bind_method(D_METHOD("set_shader_instance_uniform", "uniform", "value"), &GeometryInstance3D::set_shader_instance_uniform);
	ClassDB::bind_method(D_METHOD("get_shader_instance_uniform", "uniform"), &GeometryInstance3D::get_shader_instance_uniform);

	ClassDB::bind_method(D_METHOD("set_extra_cull_margin", "margin"), &GeometryInstance3D::set_extra_cull_margin);
	ClassDB::bind_method(D_METHOD("get_extra_cull_margin"), &GeometryInstance3D::get_extra_cull_margin);

	ClassDB::bind_method(D_METHOD("set_lightmap_scale", "scale"), &GeometryInstance3D::set_lightmap_scale);
	ClassDB::bind_method(D_METHOD("get_lightmap_scale"), &GeometryInstance3D::get_lightmap_scale);

	ClassDB::bind_method(D_METHOD("set_gi_mode", "mode"), &GeometryInstance3D::set_gi_mode);
	ClassDB::bind_method(D_METHOD("get_gi_mode"), &GeometryInstance3D::get_gi_mode);

	ClassDB::bind_method(D_METHOD("set_ignore_occlusion_culling", "ignore_culling"), &GeometryInstance3D::set_ignore_occlusion_culling);
	ClassDB::bind_method(D_METHOD("is_ignoring_occlusion_culling"), &GeometryInstance3D::is_ignoring_occlusion_culling);

	ClassDB::bind_method(D_METHOD("set_custom_aabb", "aabb"), &GeometryInstance3D::set_custom_aabb);

	ClassDB::bind_method(D_METHOD("get_aabb"), &GeometryInstance3D::get_aabb);

	ADD_GROUP("Geometry", "");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "material_override", PROPERTY_HINT_RESOURCE_TYPE, "BaseMaterial3D,ShaderMaterial", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_DEFERRED_SET_RESOURCE), "set_material_override", "get_material_override");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "material_overlay", PROPERTY_HINT_RESOURCE_TYPE, "BaseMaterial3D,ShaderMaterial", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_DEFERRED_SET_RESOURCE), "set_material_overlay", "get_material_overlay");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "transparency", PROPERTY_HINT_RANGE, "0.0,1.0,0.01"), "set_transparency", "get_transparency");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "cast_shadow", PROPERTY_HINT_ENUM, "Off,On,Double-Sided,Shadows Only"), "set_cast_shadows_setting", "get_cast_shadows_setting");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "extra_cull_margin", PROPERTY_HINT_RANGE, "0,16384,0.01"), "set_extra_cull_margin", "get_extra_cull_margin");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "lod_bias", PROPERTY_HINT_RANGE, "0.001,128,0.001"), "set_lod_bias", "get_lod_bias");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "ignore_occlusion_culling"), "set_ignore_occlusion_culling", "is_ignoring_occlusion_culling");
	ADD_GROUP("Global Illumination", "gi_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "gi_mode", PROPERTY_HINT_ENUM, "Disabled,Static (VoxelGI/SDFGI/LightmapGI),Dynamic (VoxelGI only)"), "set_gi_mode", "get_gi_mode");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "gi_lightmap_scale", PROPERTY_HINT_ENUM, String::utf8("1×,2×,4×,8×")), "set_lightmap_scale", "get_lightmap_scale");

	ADD_GROUP("Visibility Range", "visibility_range_");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "visibility_range_begin", PROPERTY_HINT_RANGE, "0.0,4096.0,0.01"), "set_visibility_range_begin", "get_visibility_range_begin");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "visibility_range_begin_margin", PROPERTY_HINT_RANGE, "0.0,4096.0,0.01"), "set_visibility_range_begin_margin", "get_visibility_range_begin_margin");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "visibility_range_end", PROPERTY_HINT_RANGE, "0.0,4096.0,0.01"), "set_visibility_range_end", "get_visibility_range_end");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "visibility_range_end_margin", PROPERTY_HINT_RANGE, "0.0,4096.0,0.01"), "set_visibility_range_end_margin", "get_visibility_range_end_margin");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "visibility_range_fade_mode", PROPERTY_HINT_ENUM, "Disabled,Self,Dependencies"), "set_visibility_range_fade_mode", "get_visibility_range_fade_mode");
	//ADD_SIGNAL( MethodInfo("visibility_changed"));

	BIND_ENUM_CONSTANT(SHADOW_CASTING_SETTING_OFF);
	BIND_ENUM_CONSTANT(SHADOW_CASTING_SETTING_ON);
	BIND_ENUM_CONSTANT(SHADOW_CASTING_SETTING_DOUBLE_SIDED);
	BIND_ENUM_CONSTANT(SHADOW_CASTING_SETTING_SHADOWS_ONLY);

	BIND_ENUM_CONSTANT(GI_MODE_DISABLED);
	BIND_ENUM_CONSTANT(GI_MODE_STATIC);
	BIND_ENUM_CONSTANT(GI_MODE_DYNAMIC);

	BIND_ENUM_CONSTANT(LIGHTMAP_SCALE_1X);
	BIND_ENUM_CONSTANT(LIGHTMAP_SCALE_2X);
	BIND_ENUM_CONSTANT(LIGHTMAP_SCALE_4X);
	BIND_ENUM_CONSTANT(LIGHTMAP_SCALE_8X);
	BIND_ENUM_CONSTANT(LIGHTMAP_SCALE_MAX);

	BIND_ENUM_CONSTANT(VISIBILITY_RANGE_FADE_DISABLED);
	BIND_ENUM_CONSTANT(VISIBILITY_RANGE_FADE_SELF);
	BIND_ENUM_CONSTANT(VISIBILITY_RANGE_FADE_DEPENDENCIES);
}

GeometryInstance3D::GeometryInstance3D() {
	//RS::get_singleton()->instance_geometry_set_baked_light_texture_index(get_instance(),0);
}
