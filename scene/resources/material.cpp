/*************************************************************************/
/*  material.cpp                                                         */
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

#include "material.h"

#include "core/config/engine.h"
#include "core/version.h"
#include "scene/main/scene_tree.h"
#include "scene/scene_string_names.h"

void Material::set_next_pass(const Ref<Material> &p_pass) {
	for (Ref<Material> pass_child = p_pass; pass_child != nullptr; pass_child = pass_child->get_next_pass()) {
		ERR_FAIL_COND_MSG(pass_child == this, "Can't set as next_pass one of its parents to prevent crashes due to recursive loop.");
	}

	if (next_pass == p_pass) {
		return;
	}

	next_pass = p_pass;
	RID next_pass_rid;
	if (next_pass.is_valid()) {
		next_pass_rid = next_pass->get_rid();
	}
	RS::get_singleton()->material_set_next_pass(material, next_pass_rid);
}

Ref<Material> Material::get_next_pass() const {
	return next_pass;
}

void Material::set_render_priority(int p_priority) {
	ERR_FAIL_COND(p_priority < RENDER_PRIORITY_MIN);
	ERR_FAIL_COND(p_priority > RENDER_PRIORITY_MAX);
	render_priority = p_priority;
	RS::get_singleton()->material_set_render_priority(material, p_priority);
}

int Material::get_render_priority() const {
	return render_priority;
}

RID Material::get_rid() const {
	return material;
}

void Material::_validate_property(PropertyInfo &property) const {
	if (!_can_do_next_pass() && property.name == "next_pass") {
		property.usage = PROPERTY_USAGE_NONE;
	}
	if (!_can_use_render_priority() && property.name == "render_priority") {
		property.usage = PROPERTY_USAGE_NONE;
	}
}

void Material::inspect_native_shader_code() {
	SceneTree *st = Object::cast_to<SceneTree>(OS::get_singleton()->get_main_loop());
	RID shader = get_shader_rid();
	if (st && shader.is_valid()) {
		st->call_group("_native_shader_source_visualizer", "_inspect_shader", shader);
	}
}

void Material::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_next_pass", "next_pass"), &Material::set_next_pass);
	ClassDB::bind_method(D_METHOD("get_next_pass"), &Material::get_next_pass);

	ClassDB::bind_method(D_METHOD("set_render_priority", "priority"), &Material::set_render_priority);
	ClassDB::bind_method(D_METHOD("get_render_priority"), &Material::get_render_priority);

	ClassDB::bind_method(D_METHOD("inspect_native_shader_code"), &Material::inspect_native_shader_code);
	ClassDB::set_method_flags(get_class_static(), _scs_create("inspect_native_shader_code"), METHOD_FLAGS_DEFAULT | METHOD_FLAG_EDITOR);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "render_priority", PROPERTY_HINT_RANGE, itos(RENDER_PRIORITY_MIN) + "," + itos(RENDER_PRIORITY_MAX) + ",1"), "set_render_priority", "get_render_priority");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "next_pass", PROPERTY_HINT_RESOURCE_TYPE, "Material"), "set_next_pass", "get_next_pass");

	BIND_CONSTANT(RENDER_PRIORITY_MAX);
	BIND_CONSTANT(RENDER_PRIORITY_MIN);
}

Material::Material() {
	material = RenderingServer::get_singleton()->material_create();
	render_priority = 0;
}

Material::~Material() {
	RenderingServer::get_singleton()->free(material);
}

///////////////////////////////////

bool ShaderMaterial::_set(const StringName &p_name, const Variant &p_value) {
	if (shader.is_valid()) {
		StringName pr = shader->remap_param(p_name);
		if (!pr) {
			String n = p_name;
			if (n.find("param/") == 0) { //backwards compatibility
				pr = n.substr(6, n.length());
			}
			if (n.find("shader_param/") == 0) { //backwards compatibility
				pr = n.replace_first("shader_param/", "");
			}
		}
		if (pr) {
			set_shader_param(pr, p_value);
			return true;
		}
	}

	return false;
}

bool ShaderMaterial::_get(const StringName &p_name, Variant &r_ret) const {
	if (shader.is_valid()) {
		StringName pr = shader->remap_param(p_name);
		if (!pr) {
			String n = p_name;
			if (n.find("param/") == 0) { //backwards compatibility
				pr = n.substr(6, n.length());
			}
			if (n.find("shader_param/") == 0) { //backwards compatibility
				pr = n.replace_first("shader_param/", "");
			}
		}

		if (pr) {
			const Map<StringName, Variant>::Element *E = param_cache.find(pr);
			if (E) {
				r_ret = E->get();
			} else {
				r_ret = Variant();
			}
			return true;
		}
	}

	return false;
}

void ShaderMaterial::_get_property_list(List<PropertyInfo> *p_list) const {
	if (!shader.is_null()) {
		shader->get_param_list(p_list);
	}
}

bool ShaderMaterial::property_can_revert(const String &p_name) {
	if (shader.is_valid()) {
		StringName pr = shader->remap_param(p_name);
		if (pr) {
			Variant default_value = RenderingServer::get_singleton()->shader_get_param_default(shader->get_rid(), pr);
			Variant current_value;
			_get(p_name, current_value);
			return default_value.get_type() != Variant::NIL && default_value != current_value;
		}
	}
	return false;
}

Variant ShaderMaterial::property_get_revert(const String &p_name) {
	Variant r_ret;
	if (shader.is_valid()) {
		StringName pr = shader->remap_param(p_name);
		if (pr) {
			r_ret = RenderingServer::get_singleton()->shader_get_param_default(shader->get_rid(), pr);
		}
	}
	return r_ret;
}

void ShaderMaterial::set_shader(const Ref<Shader> &p_shader) {
	// Only connect/disconnect the signal when running in the editor.
	// This can be a slow operation, and `notify_property_list_changed()` (which is called by `_shader_changed()`)
	// does nothing in non-editor builds anyway. See GH-34741 for details.
	if (shader.is_valid() && Engine::get_singleton()->is_editor_hint()) {
		shader->disconnect("changed", callable_mp(this, &ShaderMaterial::_shader_changed));
	}

	shader = p_shader;

	RID rid;
	if (shader.is_valid()) {
		rid = shader->get_rid();

		if (Engine::get_singleton()->is_editor_hint()) {
			shader->connect("changed", callable_mp(this, &ShaderMaterial::_shader_changed));
		}
	}

	RS::get_singleton()->material_set_shader(_get_material(), rid);
	notify_property_list_changed(); //properties for shader exposed
	emit_changed();
}

Ref<Shader> ShaderMaterial::get_shader() const {
	return shader;
}

void ShaderMaterial::set_shader_param(const StringName &p_param, const Variant &p_value) {
	if (p_value.get_type() == Variant::NIL) {
		param_cache.erase(p_param);
		RS::get_singleton()->material_set_param(_get_material(), p_param, Variant());
	} else {
		param_cache[p_param] = p_value;
		if (p_value.get_type() == Variant::OBJECT) {
			RID tex_rid = p_value;
			if (tex_rid == RID()) {
				param_cache.erase(p_param);
				RS::get_singleton()->material_set_param(_get_material(), p_param, Variant());
			} else {
				RS::get_singleton()->material_set_param(_get_material(), p_param, tex_rid);
			}
		} else {
			RS::get_singleton()->material_set_param(_get_material(), p_param, p_value);
		}
	}
}

Variant ShaderMaterial::get_shader_param(const StringName &p_param) const {
	if (param_cache.has(p_param)) {
		return param_cache[p_param];
	} else {
		return Variant();
	}
}

void ShaderMaterial::_shader_changed() {
	notify_property_list_changed(); //update all properties
}

void ShaderMaterial::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_shader", "shader"), &ShaderMaterial::set_shader);
	ClassDB::bind_method(D_METHOD("get_shader"), &ShaderMaterial::get_shader);
	ClassDB::bind_method(D_METHOD("set_shader_param", "param", "value"), &ShaderMaterial::set_shader_param);
	ClassDB::bind_method(D_METHOD("get_shader_param", "param"), &ShaderMaterial::get_shader_param);
	ClassDB::bind_method(D_METHOD("property_can_revert", "name"), &ShaderMaterial::property_can_revert);
	ClassDB::bind_method(D_METHOD("property_get_revert", "name"), &ShaderMaterial::property_get_revert);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "shader", PROPERTY_HINT_RESOURCE_TYPE, "Shader"), "set_shader", "get_shader");
}

void ShaderMaterial::get_argument_options(const StringName &p_function, int p_idx, List<String> *r_options) const {
	String f = p_function.operator String();
	if ((f == "get_shader_param" || f == "set_shader_param") && p_idx == 0) {
		if (shader.is_valid()) {
			List<PropertyInfo> pl;
			shader->get_param_list(&pl);
			for (const PropertyInfo &E : pl) {
				r_options->push_back(E.name.replace_first("shader_param/", "").quote());
			}
		}
	}
	Resource::get_argument_options(p_function, p_idx, r_options);
}

bool ShaderMaterial::_can_do_next_pass() const {
	return shader.is_valid() && shader->get_mode() == Shader::MODE_SPATIAL;
}

bool ShaderMaterial::_can_use_render_priority() const {
	return shader.is_valid() && shader->get_mode() == Shader::MODE_SPATIAL;
}

Shader::Mode ShaderMaterial::get_shader_mode() const {
	if (shader.is_valid()) {
		return shader->get_mode();
	} else {
		return Shader::MODE_SPATIAL;
	}
}
RID ShaderMaterial::get_shader_rid() const {
	if (shader.is_valid()) {
		return shader->get_rid();
	} else {
		return RID();
	}
}

ShaderMaterial::ShaderMaterial() {
}

ShaderMaterial::~ShaderMaterial() {
}

/////////////////////////////////

Mutex BaseMaterial3D::material_mutex;
SelfList<BaseMaterial3D>::List *BaseMaterial3D::dirty_materials = nullptr;
Map<BaseMaterial3D::MaterialKey, BaseMaterial3D::ShaderData> BaseMaterial3D::shader_map;
BaseMaterial3D::ShaderNames *BaseMaterial3D::shader_names = nullptr;

void BaseMaterial3D::init_shaders() {
	dirty_materials = memnew(SelfList<BaseMaterial3D>::List);

	shader_names = memnew(ShaderNames);

	shader_names->albedo = "albedo";
	shader_names->specular = "specular";
	shader_names->roughness = "roughness";
	shader_names->metallic = "metallic";
	shader_names->emission = "emission";
	shader_names->emission_energy = "emission_energy";
	shader_names->normal_scale = "normal_scale";
	shader_names->rim = "rim";
	shader_names->rim_tint = "rim_tint";
	shader_names->clearcoat = "clearcoat";
	shader_names->clearcoat_gloss = "clearcoat_gloss";
	shader_names->anisotropy = "anisotropy_ratio";
	shader_names->heightmap_scale = "heightmap_scale";
	shader_names->subsurface_scattering_strength = "subsurface_scattering_strength";
	shader_names->backlight = "backlight";
	shader_names->refraction = "refraction";
	shader_names->point_size = "point_size";
	shader_names->uv1_scale = "uv1_scale";
	shader_names->uv1_offset = "uv1_offset";
	shader_names->uv2_scale = "uv2_scale";
	shader_names->uv2_offset = "uv2_offset";
	shader_names->uv1_blend_sharpness = "uv1_blend_sharpness";
	shader_names->uv2_blend_sharpness = "uv2_blend_sharpness";

	shader_names->particles_anim_h_frames = "particles_anim_h_frames";
	shader_names->particles_anim_v_frames = "particles_anim_v_frames";
	shader_names->particles_anim_loop = "particles_anim_loop";
	shader_names->heightmap_min_layers = "heightmap_min_layers";
	shader_names->heightmap_max_layers = "heightmap_max_layers";
	shader_names->heightmap_flip = "heightmap_flip";

	shader_names->grow = "grow";

	shader_names->ao_light_affect = "ao_light_affect";

	shader_names->proximity_fade_distance = "proximity_fade_distance";
	shader_names->distance_fade_min = "distance_fade_min";
	shader_names->distance_fade_max = "distance_fade_max";

	shader_names->metallic_texture_channel = "metallic_texture_channel";
	shader_names->ao_texture_channel = "ao_texture_channel";
	shader_names->clearcoat_texture_channel = "clearcoat_texture_channel";
	shader_names->rim_texture_channel = "rim_texture_channel";
	shader_names->heightmap_texture_channel = "heightmap_texture_channel";
	shader_names->refraction_texture_channel = "refraction_texture_channel";

	shader_names->transmittance_color = "transmittance_color";
	shader_names->transmittance_depth = "transmittance_depth";
	shader_names->transmittance_boost = "transmittance_boost";

	shader_names->texture_names[TEXTURE_ALBEDO] = "texture_albedo";
	shader_names->texture_names[TEXTURE_METALLIC] = "texture_metallic";
	shader_names->texture_names[TEXTURE_ROUGHNESS] = "texture_roughness";
	shader_names->texture_names[TEXTURE_EMISSION] = "texture_emission";
	shader_names->texture_names[TEXTURE_NORMAL] = "texture_normal";
	shader_names->texture_names[TEXTURE_RIM] = "texture_rim";
	shader_names->texture_names[TEXTURE_CLEARCOAT] = "texture_clearcoat";
	shader_names->texture_names[TEXTURE_FLOWMAP] = "texture_flowmap";
	shader_names->texture_names[TEXTURE_AMBIENT_OCCLUSION] = "texture_ambient_occlusion";
	shader_names->texture_names[TEXTURE_HEIGHTMAP] = "texture_heightmap";
	shader_names->texture_names[TEXTURE_SUBSURFACE_SCATTERING] = "texture_subsurface_scattering";
	shader_names->texture_names[TEXTURE_SUBSURFACE_TRANSMITTANCE] = "texture_subsurface_transmittance";
	shader_names->texture_names[TEXTURE_BACKLIGHT] = "texture_backlight";
	shader_names->texture_names[TEXTURE_REFRACTION] = "texture_refraction";
	shader_names->texture_names[TEXTURE_DETAIL_MASK] = "texture_detail_mask";
	shader_names->texture_names[TEXTURE_DETAIL_ALBEDO] = "texture_detail_albedo";
	shader_names->texture_names[TEXTURE_DETAIL_NORMAL] = "texture_detail_normal";
	shader_names->texture_names[TEXTURE_ORM] = "texture_orm";

	shader_names->alpha_scissor_threshold = "alpha_scissor_threshold";
	shader_names->alpha_hash_scale = "alpha_hash_scale";

	shader_names->alpha_antialiasing_edge = "alpha_antialiasing_edge";
	shader_names->albedo_texture_size = "albedo_texture_size";
}

Ref<StandardMaterial3D> BaseMaterial3D::materials_for_2d[BaseMaterial3D::MAX_MATERIALS_FOR_2D];

void BaseMaterial3D::finish_shaders() {
	for (int i = 0; i < MAX_MATERIALS_FOR_2D; i++) {
		materials_for_2d[i].unref();
	}

	memdelete(dirty_materials);
	dirty_materials = nullptr;

	memdelete(shader_names);
}

void BaseMaterial3D::_update_shader() {
	dirty_materials->remove(&element);

	MaterialKey mk = _compute_key();
	if (mk == current_key) {
		return; //no update required in the end
	}

	if (shader_map.has(current_key)) {
		shader_map[current_key].users--;
		if (shader_map[current_key].users == 0) {
			//deallocate shader, as it's no longer in use
			RS::get_singleton()->free(shader_map[current_key].shader);
			shader_map.erase(current_key);
		}
	}

	current_key = mk;

	if (shader_map.has(mk)) {
		RS::get_singleton()->material_set_shader(_get_material(), shader_map[mk].shader);
		shader_map[mk].users++;
		return;
	}

	String texfilter_str;
	switch (texture_filter) {
		case TEXTURE_FILTER_NEAREST:
			texfilter_str = "filter_nearest";
			break;
		case TEXTURE_FILTER_LINEAR:
			texfilter_str = "filter_linear";
			break;
		case TEXTURE_FILTER_NEAREST_WITH_MIPMAPS:
			texfilter_str = "filter_nearest_mipmap";
			break;
		case TEXTURE_FILTER_LINEAR_WITH_MIPMAPS:
			texfilter_str = "filter_linear_mipmap";
			break;
		case TEXTURE_FILTER_NEAREST_WITH_MIPMAPS_ANISOTROPIC:
			texfilter_str = "filter_nearest_mipmap_aniso";
			break;
		case TEXTURE_FILTER_LINEAR_WITH_MIPMAPS_ANISOTROPIC:
			texfilter_str = "filter_linear_mipmap_aniso";
			break;
		case TEXTURE_FILTER_MAX:
			break; // Internal value, skip.
	}

	if (flags[FLAG_USE_TEXTURE_REPEAT]) {
		texfilter_str += ",repeat_enable";
	} else {
		texfilter_str += ",repeat_disable";
	}

	//must create a shader!

	// Add a comment to describe the shader origin (useful when converting to ShaderMaterial).
	String code = vformat(
			"// NOTE: Shader automatically converted from " VERSION_NAME " " VERSION_FULL_CONFIG "'s %s.\n\n",
			orm ? "ORMMaterial3D" : "StandardMaterial3D");

	code += "shader_type spatial;\nrender_mode ";
	switch (blend_mode) {
		case BLEND_MODE_MIX:
			code += "blend_mix";
			break;
		case BLEND_MODE_ADD:
			code += "blend_add";
			break;
		case BLEND_MODE_SUB:
			code += "blend_sub";
			break;
		case BLEND_MODE_MUL:
			code += "blend_mul";
			break;
		case BLEND_MODE_MAX:
			break; // Internal value, skip.
	}

	DepthDrawMode ddm = depth_draw_mode;
	if (features[FEATURE_REFRACTION]) {
		ddm = DEPTH_DRAW_ALWAYS;
	}

	switch (ddm) {
		case DEPTH_DRAW_OPAQUE_ONLY:
			code += ",depth_draw_opaque";
			break;
		case DEPTH_DRAW_ALWAYS:
			code += ",depth_draw_always";
			break;
		case DEPTH_DRAW_DISABLED:
			code += ",depth_draw_never";
			break;
		case DEPTH_DRAW_MAX:
			break; // Internal value, skip.
	}

	switch (cull_mode) {
		case CULL_BACK:
			code += ",cull_back";
			break;
		case CULL_FRONT:
			code += ",cull_front";
			break;
		case CULL_DISABLED:
			code += ",cull_disabled";
			break;
		case CULL_MAX:
			break; // Internal value, skip.
	}
	switch (diffuse_mode) {
		case DIFFUSE_BURLEY:
			code += ",diffuse_burley";
			break;
		case DIFFUSE_LAMBERT:
			code += ",diffuse_lambert";
			break;
		case DIFFUSE_LAMBERT_WRAP:
			code += ",diffuse_lambert_wrap";
			break;
		case DIFFUSE_TOON:
			code += ",diffuse_toon";
			break;
		case DIFFUSE_MAX:
			break; // Internal value, skip.
	}
	switch (specular_mode) {
		case SPECULAR_SCHLICK_GGX:
			code += ",specular_schlick_ggx";
			break;
		case SPECULAR_BLINN:
			code += ",specular_blinn";
			break;
		case SPECULAR_PHONG:
			code += ",specular_phong";
			break;
		case SPECULAR_TOON:
			code += ",specular_toon";
			break;
		case SPECULAR_DISABLED:
			code += ",specular_disabled";
			break;
		case SPECULAR_MAX:
			break; // Internal value, skip.
	}
	if (features[FEATURE_SUBSURFACE_SCATTERING] && flags[FLAG_SUBSURFACE_MODE_SKIN]) {
		code += ",sss_mode_skin";
	}

	if (shading_mode == SHADING_MODE_UNSHADED) {
		code += ",unshaded";
	}
	if (flags[FLAG_DISABLE_DEPTH_TEST]) {
		code += ",depth_test_disabled";
	}
	if (flags[FLAG_PARTICLE_TRAILS_MODE]) {
		code += ",particle_trails";
	}
	if (shading_mode == SHADING_MODE_PER_VERTEX) {
		code += ",vertex_lighting";
	}
	if (flags[FLAG_DONT_RECEIVE_SHADOWS]) {
		code += ",shadows_disabled";
	}
	if (flags[FLAG_DISABLE_AMBIENT_LIGHT]) {
		code += ",ambient_light_disabled";
	}
	if (flags[FLAG_USE_SHADOW_TO_OPACITY]) {
		code += ",shadow_to_opacity";
	}

	if (transparency == TRANSPARENCY_ALPHA_DEPTH_PRE_PASS) {
		code += ",depth_prepass_alpha";
	}

	// Although its technically possible to do alpha antialiasing without using alpha hash or alpha scissor,
	// it is restricted in the base material because it has no use, and abusing it with regular Alpha blending can
	// saturate the MSAA mask
	if (transparency == TRANSPARENCY_ALPHA_HASH || transparency == TRANSPARENCY_ALPHA_SCISSOR) {
		// alpha antialiasing is only useful in ALPHA_HASH or ALPHA_SCISSOR
		if (alpha_antialiasing_mode == ALPHA_ANTIALIASING_ALPHA_TO_COVERAGE) {
			code += ",alpha_to_coverage";
		} else if (alpha_antialiasing_mode == ALPHA_ANTIALIASING_ALPHA_TO_COVERAGE_AND_TO_ONE) {
			code += ",alpha_to_coverage_and_one";
		}
	}

	code += ";\n";

	code += "uniform vec4 albedo : hint_color;\n";
	code += "uniform sampler2D texture_albedo : hint_albedo," + texfilter_str + ";\n";
	if (grow_enabled) {
		code += "uniform float grow;\n";
	}

	if (proximity_fade_enabled) {
		code += "uniform float proximity_fade_distance;\n";
	}
	if (distance_fade != DISTANCE_FADE_DISABLED) {
		code += "uniform float distance_fade_min;\n";
		code += "uniform float distance_fade_max;\n";
	}

	// alpha scissor is only valid if there is not antialiasing edge
	// alpha hash is valid whenever, but not with alpha scissor
	if (transparency == TRANSPARENCY_ALPHA_SCISSOR) {
		code += "uniform float alpha_scissor_threshold;\n";
	} else if (transparency == TRANSPARENCY_ALPHA_HASH) {
		code += "uniform float alpha_hash_scale;\n";
	}
	// if alpha antialiasing isn't off, add in the edge variable
	if (alpha_antialiasing_mode != ALPHA_ANTIALIASING_OFF &&
			(transparency == TRANSPARENCY_ALPHA_SCISSOR || transparency == TRANSPARENCY_ALPHA_HASH)) {
		code += "uniform float alpha_antialiasing_edge;\n";
		code += "uniform ivec2 albedo_texture_size;\n";
	}

	code += "uniform float point_size : hint_range(0,128);\n";

	//TODO ALL HINTS
	if (!orm) {
		code += "uniform float roughness : hint_range(0,1);\n";
		code += "uniform sampler2D texture_metallic : hint_white," + texfilter_str + ";\n";
		code += "uniform vec4 metallic_texture_channel;\n";
		switch (roughness_texture_channel) {
			case TEXTURE_CHANNEL_RED: {
				code += "uniform sampler2D texture_roughness : hint_roughness_r," + texfilter_str + ";\n";
			} break;
			case TEXTURE_CHANNEL_GREEN: {
				code += "uniform sampler2D texture_roughness : hint_roughness_g," + texfilter_str + ";\n";
			} break;
			case TEXTURE_CHANNEL_BLUE: {
				code += "uniform sampler2D texture_roughness : hint_roughness_b," + texfilter_str + ";\n";
			} break;
			case TEXTURE_CHANNEL_ALPHA: {
				code += "uniform sampler2D texture_roughness : hint_roughness_a," + texfilter_str + ";\n";
			} break;
			case TEXTURE_CHANNEL_GRAYSCALE: {
				code += "uniform sampler2D texture_roughness : hint_roughness_gray," + texfilter_str + ";\n";
			} break;
			case TEXTURE_CHANNEL_MAX:
				break; // Internal value, skip.
		}

		code += "uniform float specular;\n";
		code += "uniform float metallic;\n";
	} else {
		code += "uniform sampler2D texture_orm : hint_roughness_g," + texfilter_str + ";\n";
	}

	if (billboard_mode == BILLBOARD_PARTICLES) {
		code += "uniform int particles_anim_h_frames;\n";
		code += "uniform int particles_anim_v_frames;\n";
		code += "uniform bool particles_anim_loop;\n";
	}

	if (features[FEATURE_EMISSION]) {
		code += "uniform sampler2D texture_emission : hint_black_albedo," + texfilter_str + ";\n";
		code += "uniform vec4 emission : hint_color;\n";
		code += "uniform float emission_energy;\n";
	}

	if (features[FEATURE_REFRACTION]) {
		code += "uniform sampler2D texture_refraction : " + texfilter_str + ";\n";
		code += "uniform float refraction : hint_range(-16,16);\n";
		code += "uniform vec4 refraction_texture_channel;\n";
	}

	if (features[FEATURE_NORMAL_MAPPING]) {
		code += "uniform sampler2D texture_normal : hint_roughness_normal," + texfilter_str + ";\n";
		code += "uniform float normal_scale : hint_range(-16,16);\n";
	}
	if (features[FEATURE_RIM]) {
		code += "uniform float rim : hint_range(0,1);\n";
		code += "uniform float rim_tint : hint_range(0,1);\n";
		code += "uniform sampler2D texture_rim : hint_white," + texfilter_str + ";\n";
	}
	if (features[FEATURE_CLEARCOAT]) {
		code += "uniform float clearcoat : hint_range(0,1);\n";
		code += "uniform float clearcoat_gloss : hint_range(0,1);\n";
		code += "uniform sampler2D texture_clearcoat : hint_white," + texfilter_str + ";\n";
	}
	if (features[FEATURE_ANISOTROPY]) {
		code += "uniform float anisotropy_ratio : hint_range(0,256);\n";
		code += "uniform sampler2D texture_flowmap : hint_aniso," + texfilter_str + ";\n";
	}
	if (features[FEATURE_AMBIENT_OCCLUSION]) {
		code += "uniform sampler2D texture_ambient_occlusion : hint_white, " + texfilter_str + ";\n";
		code += "uniform vec4 ao_texture_channel;\n";
		code += "uniform float ao_light_affect;\n";
	}

	if (features[FEATURE_DETAIL]) {
		code += "uniform sampler2D texture_detail_albedo : hint_albedo," + texfilter_str + ";\n";
		code += "uniform sampler2D texture_detail_normal : hint_normal," + texfilter_str + ";\n";
		code += "uniform sampler2D texture_detail_mask : hint_white," + texfilter_str + ";\n";
	}

	if (features[FEATURE_SUBSURFACE_SCATTERING]) {
		code += "uniform float subsurface_scattering_strength : hint_range(0,1);\n";
		code += "uniform sampler2D texture_subsurface_scattering : hint_white," + texfilter_str + ";\n";
	}

	if (features[FEATURE_SUBSURFACE_TRANSMITTANCE]) {
		code += "uniform vec4 transmittance_color : hint_color;\n";
		code += "uniform float transmittance_depth;\n";
		code += "uniform sampler2D texture_subsurface_transmittance : hint_white," + texfilter_str + ";\n";
		code += "uniform float transmittance_boost;\n";
	}

	if (features[FEATURE_BACKLIGHT]) {
		code += "uniform vec4 backlight : hint_color;\n";
		code += "uniform sampler2D texture_backlight : hint_black," + texfilter_str + ";\n";
	}

	if (features[FEATURE_HEIGHT_MAPPING]) {
		code += "uniform sampler2D texture_heightmap : hint_black," + texfilter_str + ";\n";
		code += "uniform float heightmap_scale;\n";
		code += "uniform int heightmap_min_layers;\n";
		code += "uniform int heightmap_max_layers;\n";
		code += "uniform vec2 heightmap_flip;\n";
	}
	if (flags[FLAG_UV1_USE_TRIPLANAR]) {
		code += "varying vec3 uv1_triplanar_pos;\n";
	}
	if (flags[FLAG_UV2_USE_TRIPLANAR]) {
		code += "varying vec3 uv2_triplanar_pos;\n";
	}
	if (flags[FLAG_UV1_USE_TRIPLANAR]) {
		code += "uniform float uv1_blend_sharpness;\n";
		code += "varying vec3 uv1_power_normal;\n";
	}

	if (flags[FLAG_UV2_USE_TRIPLANAR]) {
		code += "uniform float uv2_blend_sharpness;\n";
		code += "varying vec3 uv2_power_normal;\n";
	}

	code += "uniform vec3 uv1_scale;\n";
	code += "uniform vec3 uv1_offset;\n";
	code += "uniform vec3 uv2_scale;\n";
	code += "uniform vec3 uv2_offset;\n";

	code += "\n\n";

	code += "void vertex() {\n";

	if (flags[FLAG_SRGB_VERTEX_COLOR]) {
		code += "	if (!OUTPUT_IS_SRGB) {\n";
		code += "		COLOR.rgb = mix(pow((COLOR.rgb + vec3(0.055)) * (1.0 / (1.0 + 0.055)), vec3(2.4)), COLOR.rgb * (1.0 / 12.92), lessThan(COLOR.rgb, vec3(0.04045)));\n";
		code += "	}\n";
	}
	if (flags[FLAG_USE_POINT_SIZE]) {
		code += "	POINT_SIZE=point_size;\n";
	}

	if (shading_mode == SHADING_MODE_PER_VERTEX) {
		code += "	ROUGHNESS=roughness;\n";
	}

	if (!flags[FLAG_UV1_USE_TRIPLANAR]) {
		code += "	UV=UV*uv1_scale.xy+uv1_offset.xy;\n";
	}

	switch (billboard_mode) {
		case BILLBOARD_DISABLED: {
		} break;
		case BILLBOARD_ENABLED: {
			code += "	MODELVIEW_MATRIX = INV_CAMERA_MATRIX * mat4(CAMERA_MATRIX[0],CAMERA_MATRIX[1],CAMERA_MATRIX[2],WORLD_MATRIX[3]);\n";

			if (flags[FLAG_BILLBOARD_KEEP_SCALE]) {
				code += "	MODELVIEW_MATRIX = MODELVIEW_MATRIX * mat4(vec4(length(WORLD_MATRIX[0].xyz), 0.0, 0.0, 0.0),vec4(0.0, length(WORLD_MATRIX[1].xyz), 0.0, 0.0),vec4(0.0, 0.0, length(WORLD_MATRIX[2].xyz), 0.0),vec4(0.0, 0.0, 0.0, 1.0));\n";
			}
		} break;
		case BILLBOARD_FIXED_Y: {
			code += "	MODELVIEW_MATRIX = INV_CAMERA_MATRIX * mat4(vec4(normalize(cross(vec3(0.0, 1.0, 0.0), CAMERA_MATRIX[2].xyz)),0.0),vec4(0.0, 1.0, 0.0, 0.0),vec4(normalize(cross(CAMERA_MATRIX[0].xyz, vec3(0.0, 1.0, 0.0))),0.0),WORLD_MATRIX[3]);\n";

			if (flags[FLAG_BILLBOARD_KEEP_SCALE]) {
				code += "	MODELVIEW_MATRIX = MODELVIEW_MATRIX * mat4(vec4(length(WORLD_MATRIX[0].xyz), 0.0, 0.0, 0.0),vec4(0.0, length(WORLD_MATRIX[1].xyz), 0.0, 0.0),vec4(0.0, 0.0, length(WORLD_MATRIX[2].xyz), 0.0),vec4(0.0, 0.0, 0.0, 1.0));\n";
			}
		} break;
		case BILLBOARD_PARTICLES: {
			//make billboard
			code += "	mat4 mat_world = mat4(normalize(CAMERA_MATRIX[0])*length(WORLD_MATRIX[0]),normalize(CAMERA_MATRIX[1])*length(WORLD_MATRIX[0]),normalize(CAMERA_MATRIX[2])*length(WORLD_MATRIX[2]),WORLD_MATRIX[3]);\n";
			//rotate by rotation
			code += "	mat_world = mat_world * mat4( vec4(cos(INSTANCE_CUSTOM.x),-sin(INSTANCE_CUSTOM.x), 0.0, 0.0), vec4(sin(INSTANCE_CUSTOM.x), cos(INSTANCE_CUSTOM.x), 0.0, 0.0),vec4(0.0, 0.0, 1.0, 0.0),vec4(0.0, 0.0, 0.0, 1.0));\n";
			//set modelview
			code += "	MODELVIEW_MATRIX = INV_CAMERA_MATRIX * mat_world;\n";

			//handle animation
			code += "	float h_frames = float(particles_anim_h_frames);\n";
			code += "	float v_frames = float(particles_anim_v_frames);\n";
			code += "	float particle_total_frames = float(particles_anim_h_frames * particles_anim_v_frames);\n";
			code += "	float particle_frame = floor(INSTANCE_CUSTOM.z * float(particle_total_frames));\n";
			code += "	if (!particles_anim_loop) {\n";
			code += "		particle_frame = clamp(particle_frame, 0.0, particle_total_frames - 1.0);\n";
			code += "	} else {\n";
			code += "		particle_frame = mod(particle_frame, particle_total_frames);\n";
			code += "	}";
			code += "	UV /= vec2(h_frames, v_frames);\n";
			code += "	UV += vec2(mod(particle_frame, h_frames) / h_frames, floor((particle_frame + 0.5) / h_frames) / v_frames);\n";
		} break;
		case BILLBOARD_MAX:
			break; // Internal value, skip.
	}

	if (flags[FLAG_FIXED_SIZE]) {
		code += "	if (PROJECTION_MATRIX[3][3] != 0.0) {\n";
		//orthogonal matrix, try to do about the same
		//with viewport size
		code += "		float h = abs(1.0 / (2.0 * PROJECTION_MATRIX[1][1]));\n";
		code += "		float sc = (h * 2.0); //consistent with Y-fov\n";
		code += "		MODELVIEW_MATRIX[0]*=sc;\n";
		code += "		MODELVIEW_MATRIX[1]*=sc;\n";
		code += "		MODELVIEW_MATRIX[2]*=sc;\n";
		code += "	} else {\n";
		//just scale by depth
		code += "		float sc = -(MODELVIEW_MATRIX)[3].z;\n";
		code += "		MODELVIEW_MATRIX[0]*=sc;\n";
		code += "		MODELVIEW_MATRIX[1]*=sc;\n";
		code += "		MODELVIEW_MATRIX[2]*=sc;\n";
		code += "	}\n";
	}

	if (detail_uv == DETAIL_UV_2 && !flags[FLAG_UV2_USE_TRIPLANAR]) {
		code += "	UV2=UV2*uv2_scale.xy+uv2_offset.xy;\n";
	}
	if (flags[FLAG_UV1_USE_TRIPLANAR] || flags[FLAG_UV2_USE_TRIPLANAR]) {
		//generate tangent and binormal in world space
		code += "	TANGENT = vec3(0.0,0.0,-1.0) * abs(NORMAL.x);\n";
		code += "	TANGENT+= vec3(1.0,0.0,0.0) * abs(NORMAL.y);\n";
		code += "	TANGENT+= vec3(1.0,0.0,0.0) * abs(NORMAL.z);\n";
		code += "	TANGENT = normalize(TANGENT);\n";

		code += "	BINORMAL = vec3(0.0,1.0,0.0) * abs(NORMAL.x);\n";
		code += "	BINORMAL+= vec3(0.0,0.0,-1.0) * abs(NORMAL.y);\n";
		code += "	BINORMAL+= vec3(0.0,1.0,0.0) * abs(NORMAL.z);\n";
		code += "	BINORMAL = normalize(BINORMAL);\n";
	}

	if (flags[FLAG_UV1_USE_TRIPLANAR]) {
		if (flags[FLAG_UV1_USE_WORLD_TRIPLANAR]) {
			code += "	uv1_power_normal=pow(abs(mat3(WORLD_MATRIX) * NORMAL),vec3(uv1_blend_sharpness));\n";
			code += "	uv1_triplanar_pos = (WORLD_MATRIX * vec4(VERTEX, 1.0f)).xyz * uv1_scale + uv1_offset;\n";
		} else {
			code += "	uv1_power_normal=pow(abs(NORMAL),vec3(uv1_blend_sharpness));\n";
			code += "	uv1_triplanar_pos = VERTEX * uv1_scale + uv1_offset;\n";
		}
		code += "	uv1_power_normal/=dot(uv1_power_normal,vec3(1.0));\n";
		code += "	uv1_triplanar_pos *= vec3(1.0,-1.0, 1.0);\n";
	}

	if (flags[FLAG_UV2_USE_TRIPLANAR]) {
		if (flags[FLAG_UV2_USE_WORLD_TRIPLANAR]) {
			code += "	uv2_power_normal=pow(abs(mat3(WORLD_MATRIX) * NORMAL), vec3(uv2_blend_sharpness));\n";
			code += "	uv2_triplanar_pos = (WORLD_MATRIX * vec4(VERTEX, 1.0f)).xyz * uv2_scale + uv2_offset;\n";
		} else {
			code += "	uv2_power_normal=pow(abs(NORMAL), vec3(uv2_blend_sharpness));\n";
			code += "	uv2_triplanar_pos = VERTEX * uv2_scale + uv2_offset;\n";
		}
		code += "	uv2_power_normal/=dot(uv2_power_normal,vec3(1.0));\n";
		code += "	uv2_triplanar_pos *= vec3(1.0,-1.0, 1.0);\n";
	}

	if (grow_enabled) {
		code += "	VERTEX+=NORMAL*grow;\n";
	}

	code += "}\n";
	code += "\n\n";
	if (flags[FLAG_UV1_USE_TRIPLANAR] || flags[FLAG_UV2_USE_TRIPLANAR]) {
		code += "vec4 triplanar_texture(sampler2D p_sampler,vec3 p_weights,vec3 p_triplanar_pos) {\n";
		code += "	vec4 samp=vec4(0.0);\n";
		code += "	samp+= texture(p_sampler,p_triplanar_pos.xy) * p_weights.z;\n";
		code += "	samp+= texture(p_sampler,p_triplanar_pos.xz) * p_weights.y;\n";
		code += "	samp+= texture(p_sampler,p_triplanar_pos.zy * vec2(-1.0,1.0)) * p_weights.x;\n";
		code += "	return samp;\n";
		code += "}\n";
	}
	code += "\n\n";
	code += "void fragment() {\n";

	if (!flags[FLAG_UV1_USE_TRIPLANAR]) {
		code += "	vec2 base_uv = UV;\n";
	}

	if ((features[FEATURE_DETAIL] && detail_uv == DETAIL_UV_2) || (features[FEATURE_AMBIENT_OCCLUSION] && flags[FLAG_AO_ON_UV2]) || (features[FEATURE_EMISSION] && flags[FLAG_EMISSION_ON_UV2])) {
		code += "	vec2 base_uv2 = UV2;\n";
	}

	if (features[FEATURE_HEIGHT_MAPPING] && flags[FLAG_UV1_USE_TRIPLANAR]) {
		// Display both resource name and albedo texture name.
		// Materials are often built-in to scenes, so displaying the resource name alone may not be meaningful.
		// On the other hand, albedo textures are almost always external to the scene.
		if (textures[TEXTURE_ALBEDO].is_valid()) {
			WARN_PRINT(vformat("%s (albedo %s): Height mapping is not supported on triplanar materials. Ignoring height mapping in favor of triplanar mapping.", get_path(), textures[TEXTURE_ALBEDO]->get_path()));
		} else if (!get_path().is_empty()) {
			WARN_PRINT(vformat("%s: Height mapping is not supported on triplanar materials. Ignoring height mapping in favor of triplanar mapping.", get_path()));
		} else {
			// Resource wasn't saved yet.
			WARN_PRINT("Height mapping is not supported on triplanar materials. Ignoring height mapping in favor of triplanar mapping.");
		}
	}

	if (!RenderingServer::get_singleton()->is_low_end() && features[FEATURE_HEIGHT_MAPPING] && !flags[FLAG_UV1_USE_TRIPLANAR]) { //heightmap not supported with triplanar
		code += "	{\n";
		code += "		vec3 view_dir = normalize(normalize(-VERTEX)*mat3(TANGENT*heightmap_flip.x,-BINORMAL*heightmap_flip.y,NORMAL));\n"; // binormal is negative due to mikktspace, flip 'unflips' it ;-)

		if (deep_parallax) {
			code += "		float num_layers = mix(float(heightmap_max_layers),float(heightmap_min_layers), abs(dot(vec3(0.0, 0.0, 1.0), view_dir)));\n";
			code += "		float layer_depth = 1.0 / num_layers;\n";
			code += "		float current_layer_depth = 0.0;\n";
			code += "		vec2 P = view_dir.xy * heightmap_scale;\n";
			code += "		vec2 delta = P / num_layers;\n";
			code += "		vec2 ofs = base_uv;\n";
			if (flags[FLAG_INVERT_HEIGHTMAP]) {
				code += "		float depth = texture(texture_heightmap, ofs).r;\n";
			} else {
				code += "		float depth = 1.0 - texture(texture_heightmap, ofs).r;\n";
			}
			code += "		float current_depth = 0.0;\n";
			code += "		while(current_depth < depth) {\n";
			code += "			ofs -= delta;\n";
			if (flags[FLAG_INVERT_HEIGHTMAP]) {
				code += "			depth = texture(texture_heightmap, ofs).r;\n";
			} else {
				code += "			depth = 1.0 - texture(texture_heightmap, ofs).r;\n";
			}
			code += "			current_depth += layer_depth;\n";
			code += "		}\n";
			code += "		vec2 prev_ofs = ofs + delta;\n";
			code += "		float after_depth  = depth - current_depth;\n";
			if (flags[FLAG_INVERT_HEIGHTMAP]) {
				code += "		float before_depth = texture(texture_heightmap, prev_ofs).r - current_depth + layer_depth;\n";
			} else {
				code += "		float before_depth = ( 1.0 - texture(texture_heightmap, prev_ofs).r  ) - current_depth + layer_depth;\n";
			}
			code += "		float weight = after_depth / (after_depth - before_depth);\n";
			code += "		ofs = mix(ofs,prev_ofs,weight);\n";

		} else {
			if (flags[FLAG_INVERT_HEIGHTMAP]) {
				code += "		float depth = texture(texture_heightmap, base_uv).r;\n";
			} else {
				code += "		float depth = 1.0 - texture(texture_heightmap, base_uv).r;\n";
			}
			// Use offset limiting to improve the appearance of non-deep parallax.
			// This reduces the impression of depth, but avoids visible warping in the distance.
			code += "		vec2 ofs = base_uv - view_dir.xy * depth * heightmap_scale;\n";
		}

		code += "		base_uv=ofs;\n";
		if (features[FEATURE_DETAIL] && detail_uv == DETAIL_UV_2) {
			code += "		base_uv2-=ofs;\n";
		}

		code += "	}\n";
	}

	if (flags[FLAG_USE_POINT_SIZE]) {
		code += "	vec4 albedo_tex = texture(texture_albedo,POINT_COORD);\n";
	} else {
		if (flags[FLAG_UV1_USE_TRIPLANAR]) {
			code += "	vec4 albedo_tex = triplanar_texture(texture_albedo,uv1_power_normal,uv1_triplanar_pos);\n";
		} else {
			code += "	vec4 albedo_tex = texture(texture_albedo,base_uv);\n";
		}
	}

	if (flags[FLAG_ALBEDO_TEXTURE_FORCE_SRGB]) {
		code += "	albedo_tex.rgb = mix(pow((albedo_tex.rgb + vec3(0.055)) * (1.0 / (1.0 + 0.055)),vec3(2.4)),albedo_tex.rgb.rgb * (1.0 / 12.92),lessThan(albedo_tex.rgb,vec3(0.04045)));\n";
	}

	if (flags[FLAG_ALBEDO_FROM_VERTEX_COLOR]) {
		code += "	albedo_tex *= COLOR;\n";
	}
	code += "	ALBEDO = albedo.rgb * albedo_tex.rgb;\n";

	if (!orm) {
		if (flags[FLAG_UV1_USE_TRIPLANAR]) {
			code += "	float metallic_tex = dot(triplanar_texture(texture_metallic,uv1_power_normal,uv1_triplanar_pos),metallic_texture_channel);\n";
		} else {
			code += "	float metallic_tex = dot(texture(texture_metallic,base_uv),metallic_texture_channel);\n";
		}
		code += "	METALLIC = metallic_tex * metallic;\n";

		switch (roughness_texture_channel) {
			case TEXTURE_CHANNEL_RED: {
				code += "	vec4 roughness_texture_channel = vec4(1.0,0.0,0.0,0.0);\n";
			} break;
			case TEXTURE_CHANNEL_GREEN: {
				code += "	vec4 roughness_texture_channel = vec4(0.0,1.0,0.0,0.0);\n";
			} break;
			case TEXTURE_CHANNEL_BLUE: {
				code += "	vec4 roughness_texture_channel = vec4(0.0,0.0,1.0,0.0);\n";
			} break;
			case TEXTURE_CHANNEL_ALPHA: {
				code += "	vec4 roughness_texture_channel = vec4(0.0,0.0,0.0,1.0);\n";
			} break;
			case TEXTURE_CHANNEL_GRAYSCALE: {
				code += "	vec4 roughness_texture_channel = vec4(0.333333,0.333333,0.333333,0.0);\n";
			} break;
			case TEXTURE_CHANNEL_MAX:
				break; // Internal value, skip.
		}

		if (flags[FLAG_UV1_USE_TRIPLANAR]) {
			code += "	float roughness_tex = dot(triplanar_texture(texture_roughness,uv1_power_normal,uv1_triplanar_pos),roughness_texture_channel);\n";
		} else {
			code += "	float roughness_tex = dot(texture(texture_roughness,base_uv),roughness_texture_channel);\n";
		}
		code += "	ROUGHNESS = roughness_tex * roughness;\n";
		code += "	SPECULAR = specular;\n";
	} else {
		if (flags[FLAG_UV1_USE_TRIPLANAR]) {
			code += "	vec4 orm_tex = triplanar_texture(texture_orm,uv1_power_normal,uv1_triplanar_pos);\n";
		} else {
			code += "	vec4 orm_tex = texture(texture_orm,base_uv);\n";
		}

		code += "	ROUGHNESS = orm_tex.g;\n";
		code += "	METALLIC = orm_tex.b;\n";
	}

	if (features[FEATURE_NORMAL_MAPPING]) {
		if (flags[FLAG_UV1_USE_TRIPLANAR]) {
			code += "	NORMAL_MAP = triplanar_texture(texture_normal,uv1_power_normal,uv1_triplanar_pos).rgb;\n";
		} else {
			code += "	NORMAL_MAP = texture(texture_normal,base_uv).rgb;\n";
		}
		code += "	NORMAL_MAP_DEPTH = normal_scale;\n";
	}

	if (features[FEATURE_EMISSION]) {
		if (flags[FLAG_EMISSION_ON_UV2]) {
			if (flags[FLAG_UV2_USE_TRIPLANAR]) {
				code += "	vec3 emission_tex = triplanar_texture(texture_emission,uv2_power_normal,uv2_triplanar_pos).rgb;\n";
			} else {
				code += "	vec3 emission_tex = texture(texture_emission,base_uv2).rgb;\n";
			}
		} else {
			if (flags[FLAG_UV1_USE_TRIPLANAR]) {
				code += "	vec3 emission_tex = triplanar_texture(texture_emission,uv1_power_normal,uv1_triplanar_pos).rgb;\n";
			} else {
				code += "	vec3 emission_tex = texture(texture_emission,base_uv).rgb;\n";
			}
		}

		if (emission_op == EMISSION_OP_ADD) {
			code += "	EMISSION = (emission.rgb+emission_tex)*emission_energy;\n";
		} else {
			code += "	EMISSION = (emission.rgb*emission_tex)*emission_energy;\n";
		}
	}

	if (features[FEATURE_REFRACTION]) {
		if (features[FEATURE_NORMAL_MAPPING]) {
			code += "	vec3 unpacked_normal = NORMAL_MAP;\n";
			code += "	unpacked_normal.xy = unpacked_normal.xy * 2.0 - 1.0;\n";
			code += "	unpacked_normal.z = sqrt(max(0.0, 1.0 - dot(unpacked_normal.xy, unpacked_normal.xy)));\n";
			code += "	vec3 ref_normal = normalize( mix(NORMAL,TANGENT * unpacked_normal.x + BINORMAL * unpacked_normal.y + NORMAL * unpacked_normal.z,NORMAL_MAP_DEPTH) );\n";
		} else {
			code += "	vec3 ref_normal = NORMAL;\n";
		}
		if (flags[FLAG_UV1_USE_TRIPLANAR]) {
			code += "	vec2 ref_ofs = SCREEN_UV - ref_normal.xy * dot(triplanar_texture(texture_refraction,uv1_power_normal,uv1_triplanar_pos),refraction_texture_channel) * refraction;\n";
		} else {
			code += "	vec2 ref_ofs = SCREEN_UV - ref_normal.xy * dot(texture(texture_refraction,base_uv),refraction_texture_channel) * refraction;\n";
		}
		code += "	float ref_amount = 1.0 - albedo.a * albedo_tex.a;\n";
		code += "	EMISSION += textureLod(SCREEN_TEXTURE,ref_ofs,ROUGHNESS * 8.0).rgb * ref_amount;\n";
		code += "	ALBEDO *= 1.0 - ref_amount;\n";
		code += "	ALPHA = 1.0;\n";

	} else if (transparency != TRANSPARENCY_DISABLED || flags[FLAG_USE_SHADOW_TO_OPACITY] || (distance_fade == DISTANCE_FADE_PIXEL_ALPHA) || proximity_fade_enabled) {
		code += "	ALPHA *= albedo.a * albedo_tex.a;\n";
	}
	if (transparency == TRANSPARENCY_ALPHA_HASH) {
		code += "	ALPHA_HASH_SCALE = alpha_hash_scale;\n";
	} else if (transparency == TRANSPARENCY_ALPHA_SCISSOR) {
		code += "	ALPHA_SCISSOR_THRESHOLD = alpha_scissor_threshold;\n";
	}
	if (alpha_antialiasing_mode != ALPHA_ANTIALIASING_OFF && (transparency == TRANSPARENCY_ALPHA_HASH || transparency == TRANSPARENCY_ALPHA_SCISSOR)) {
		code += "	ALPHA_ANTIALIASING_EDGE = alpha_antialiasing_edge;\n";
		code += "	ALPHA_TEXTURE_COORDINATE = UV * vec2(albedo_texture_size);\n";
	}

	if (proximity_fade_enabled) {
		code += "	float depth_tex = textureLod(DEPTH_TEXTURE,SCREEN_UV,0.0).r;\n";
		code += "	vec4 world_pos = INV_PROJECTION_MATRIX * vec4(SCREEN_UV*2.0-1.0,depth_tex,1.0);\n";
		code += "	world_pos.xyz/=world_pos.w;\n";
		code += "	ALPHA*=clamp(1.0-smoothstep(world_pos.z+proximity_fade_distance,world_pos.z,VERTEX.z),0.0,1.0);\n";
	}

	if (distance_fade != DISTANCE_FADE_DISABLED) {
		if ((distance_fade == DISTANCE_FADE_OBJECT_DITHER || distance_fade == DISTANCE_FADE_PIXEL_DITHER)) {
			if (!RenderingServer::get_singleton()->is_low_end()) {
				code += "	{\n";
				if (distance_fade == DISTANCE_FADE_OBJECT_DITHER) {
					code += "		float fade_distance = abs((INV_CAMERA_MATRIX * WORLD_MATRIX[3]).z);\n";

				} else {
					code += "		float fade_distance=-VERTEX.z;\n";
				}

				code += "		float fade=clamp(smoothstep(distance_fade_min,distance_fade_max,fade_distance),0.0,1.0);\n";
				code += "		int x = int(FRAGCOORD.x) % 4;\n";
				code += "		int y = int(FRAGCOORD.y) % 4;\n";
				code += "		int index = x + y * 4;\n";
				code += "		float limit = 0.0;\n\n";
				code += "		if (x < 8) {\n";
				code += "			if (index == 0) limit = 0.0625;\n";
				code += "			if (index == 1) limit = 0.5625;\n";
				code += "			if (index == 2) limit = 0.1875;\n";
				code += "			if (index == 3) limit = 0.6875;\n";
				code += "			if (index == 4) limit = 0.8125;\n";
				code += "			if (index == 5) limit = 0.3125;\n";
				code += "			if (index == 6) limit = 0.9375;\n";
				code += "			if (index == 7) limit = 0.4375;\n";
				code += "			if (index == 8) limit = 0.25;\n";
				code += "			if (index == 9) limit = 0.75;\n";
				code += "			if (index == 10) limit = 0.125;\n";
				code += "			if (index == 11) limit = 0.625;\n";
				code += "			if (index == 12) limit = 1.0;\n";
				code += "			if (index == 13) limit = 0.5;\n";
				code += "			if (index == 14) limit = 0.875;\n";
				code += "			if (index == 15) limit = 0.375;\n";
				code += "		}\n\n";
				code += "	if (fade < limit)\n";
				code += "		discard;\n";
				code += "	}\n\n";
			}

		} else {
			code += "	ALPHA*=clamp(smoothstep(distance_fade_min,distance_fade_max,-VERTEX.z),0.0,1.0);\n";
		}
	}

	if (features[FEATURE_RIM]) {
		if (flags[FLAG_UV1_USE_TRIPLANAR]) {
			code += "	vec2 rim_tex = triplanar_texture(texture_rim,uv1_power_normal,uv1_triplanar_pos).xy;\n";
		} else {
			code += "	vec2 rim_tex = texture(texture_rim,base_uv).xy;\n";
		}
		code += "	RIM = rim*rim_tex.x;";
		code += "	RIM_TINT = rim_tint*rim_tex.y;\n";
	}

	if (features[FEATURE_CLEARCOAT]) {
		if (flags[FLAG_UV1_USE_TRIPLANAR]) {
			code += "	vec2 clearcoat_tex = triplanar_texture(texture_clearcoat,uv1_power_normal,uv1_triplanar_pos).xy;\n";
		} else {
			code += "	vec2 clearcoat_tex = texture(texture_clearcoat,base_uv).xy;\n";
		}
		code += "	CLEARCOAT = clearcoat*clearcoat_tex.x;";
		code += "	CLEARCOAT_GLOSS = clearcoat_gloss*clearcoat_tex.y;\n";
	}

	if (features[FEATURE_ANISOTROPY]) {
		if (flags[FLAG_UV1_USE_TRIPLANAR]) {
			code += "	vec3 anisotropy_tex = triplanar_texture(texture_flowmap,uv1_power_normal,uv1_triplanar_pos).rga;\n";
		} else {
			code += "	vec3 anisotropy_tex = texture(texture_flowmap,base_uv).rga;\n";
		}
		code += "	ANISOTROPY = anisotropy_ratio*anisotropy_tex.b;\n";
		code += "	ANISOTROPY_FLOW = anisotropy_tex.rg*2.0-1.0;\n";
	}

	if (features[FEATURE_AMBIENT_OCCLUSION]) {
		if (!orm) {
			if (flags[FLAG_AO_ON_UV2]) {
				if (flags[FLAG_UV2_USE_TRIPLANAR]) {
					code += "	AO = dot(triplanar_texture(texture_ambient_occlusion,uv2_power_normal,uv2_triplanar_pos),ao_texture_channel);\n";
				} else {
					code += "	AO = dot(texture(texture_ambient_occlusion,base_uv2),ao_texture_channel);\n";
				}
			} else {
				if (flags[FLAG_UV1_USE_TRIPLANAR]) {
					code += "	AO = dot(triplanar_texture(texture_ambient_occlusion,uv1_power_normal,uv1_triplanar_pos),ao_texture_channel);\n";
				} else {
					code += "	AO = dot(texture(texture_ambient_occlusion,base_uv),ao_texture_channel);\n";
				}
			}
		} else {
			code += "	AO = orm_tex.r;\n";
		}

		code += "	AO_LIGHT_AFFECT = ao_light_affect;\n";
	}

	if (features[FEATURE_SUBSURFACE_SCATTERING]) {
		if (flags[FLAG_UV1_USE_TRIPLANAR]) {
			code += "	float sss_tex = triplanar_texture(texture_subsurface_scattering,uv1_power_normal,uv1_triplanar_pos).r;\n";
		} else {
			code += "	float sss_tex = texture(texture_subsurface_scattering,base_uv).r;\n";
		}
		code += "	SSS_STRENGTH=subsurface_scattering_strength*sss_tex;\n";
	}

	if (features[FEATURE_SUBSURFACE_TRANSMITTANCE]) {
		if (flags[FLAG_UV1_USE_TRIPLANAR]) {
			code += "	vec4 trans_color_tex = triplanar_texture(texture_subsurface_transmittance,uv1_power_normal,uv1_triplanar_pos);\n";
		} else {
			code += "	vec4 trans_color_tex = texture(texture_subsurface_transmittance,base_uv);\n";
		}
		code += "	SSS_TRANSMITTANCE_COLOR=transmittance_color*trans_color_tex;\n";

		code += "	SSS_TRANSMITTANCE_DEPTH=transmittance_depth;\n";
		code += "	SSS_TRANSMITTANCE_BOOST=transmittance_boost;\n";
	}

	if (features[FEATURE_BACKLIGHT]) {
		if (flags[FLAG_UV1_USE_TRIPLANAR]) {
			code += "	vec3 backlight_tex = triplanar_texture(texture_backlight,uv1_power_normal,uv1_triplanar_pos).rgb;\n";
		} else {
			code += "	vec3 backlight_tex = texture(texture_backlight,base_uv).rgb;\n";
		}
		code += "	BACKLIGHT = (backlight.rgb+backlight_tex);\n";
	}

	if (features[FEATURE_DETAIL]) {
		bool triplanar = (flags[FLAG_UV1_USE_TRIPLANAR] && detail_uv == DETAIL_UV_1) || (flags[FLAG_UV2_USE_TRIPLANAR] && detail_uv == DETAIL_UV_2);

		if (triplanar) {
			String tp_uv = detail_uv == DETAIL_UV_1 ? "uv1" : "uv2";
			code += "	vec4 detail_tex = triplanar_texture(texture_detail_albedo," + tp_uv + "_power_normal," + tp_uv + "_triplanar_pos);\n";
			code += "	vec4 detail_norm_tex = triplanar_texture(texture_detail_normal," + tp_uv + "_power_normal," + tp_uv + "_triplanar_pos);\n";

		} else {
			String det_uv = detail_uv == DETAIL_UV_1 ? "base_uv" : "base_uv2";
			code += "	vec4 detail_tex = texture(texture_detail_albedo," + det_uv + ");\n";
			code += "	vec4 detail_norm_tex = texture(texture_detail_normal," + det_uv + ");\n";
		}

		if (flags[FLAG_UV1_USE_TRIPLANAR]) {
			code += "	vec4 detail_mask_tex = triplanar_texture(texture_detail_mask,uv1_power_normal,uv1_triplanar_pos);\n";
		} else {
			code += "	vec4 detail_mask_tex = texture(texture_detail_mask,base_uv);\n";
		}

		switch (detail_blend_mode) {
			case BLEND_MODE_MIX: {
				code += "	vec3 detail = mix(ALBEDO.rgb,detail_tex.rgb,detail_tex.a);\n";
			} break;
			case BLEND_MODE_ADD: {
				code += "	vec3 detail = mix(ALBEDO.rgb,ALBEDO.rgb+detail_tex.rgb,detail_tex.a);\n";
			} break;
			case BLEND_MODE_SUB: {
				code += "	vec3 detail = mix(ALBEDO.rgb,ALBEDO.rgb-detail_tex.rgb,detail_tex.a);\n";
			} break;
			case BLEND_MODE_MUL: {
				code += "	vec3 detail = mix(ALBEDO.rgb,ALBEDO.rgb*detail_tex.rgb,detail_tex.a);\n";
			} break;
			case BLEND_MODE_MAX:
				break; // Internal value, skip.
		}

		code += "	vec3 detail_norm = mix(NORMAL_MAP,detail_norm_tex.rgb,detail_tex.a);\n";
		code += "	NORMAL_MAP = mix(NORMAL_MAP,detail_norm,detail_mask_tex.r);\n";
		code += "	ALBEDO.rgb = mix(ALBEDO.rgb,detail,detail_mask_tex.r);\n";
	}

	code += "}\n";

	ShaderData shader_data;
	shader_data.shader = RS::get_singleton()->shader_create();
	shader_data.users = 1;

	RS::get_singleton()->shader_set_code(shader_data.shader, code);

	shader_map[mk] = shader_data;

	RS::get_singleton()->material_set_shader(_get_material(), shader_data.shader);
}

void BaseMaterial3D::flush_changes() {
	MutexLock lock(material_mutex);

	while (dirty_materials->first()) {
		dirty_materials->first()->self()->_update_shader();
	}
}

void BaseMaterial3D::_queue_shader_change() {
	MutexLock lock(material_mutex);

	if (is_initialized && !element.in_list()) {
		dirty_materials->add(&element);
	}
}

bool BaseMaterial3D::_is_shader_dirty() const {
	MutexLock lock(material_mutex);

	return element.in_list();
}

void BaseMaterial3D::set_albedo(const Color &p_albedo) {
	albedo = p_albedo;

	RS::get_singleton()->material_set_param(_get_material(), shader_names->albedo, p_albedo);
}

Color BaseMaterial3D::get_albedo() const {
	return albedo;
}

void BaseMaterial3D::set_specular(float p_specular) {
	specular = p_specular;
	RS::get_singleton()->material_set_param(_get_material(), shader_names->specular, p_specular);
}

float BaseMaterial3D::get_specular() const {
	return specular;
}

void BaseMaterial3D::set_roughness(float p_roughness) {
	roughness = p_roughness;
	RS::get_singleton()->material_set_param(_get_material(), shader_names->roughness, p_roughness);
}

float BaseMaterial3D::get_roughness() const {
	return roughness;
}

void BaseMaterial3D::set_metallic(float p_metallic) {
	metallic = p_metallic;
	RS::get_singleton()->material_set_param(_get_material(), shader_names->metallic, p_metallic);
}

float BaseMaterial3D::get_metallic() const {
	return metallic;
}

void BaseMaterial3D::set_emission(const Color &p_emission) {
	emission = p_emission;
	RS::get_singleton()->material_set_param(_get_material(), shader_names->emission, p_emission);
}

Color BaseMaterial3D::get_emission() const {
	return emission;
}

void BaseMaterial3D::set_emission_energy(float p_emission_energy) {
	emission_energy = p_emission_energy;
	RS::get_singleton()->material_set_param(_get_material(), shader_names->emission_energy, p_emission_energy);
}

float BaseMaterial3D::get_emission_energy() const {
	return emission_energy;
}

void BaseMaterial3D::set_normal_scale(float p_normal_scale) {
	normal_scale = p_normal_scale;
	RS::get_singleton()->material_set_param(_get_material(), shader_names->normal_scale, p_normal_scale);
}

float BaseMaterial3D::get_normal_scale() const {
	return normal_scale;
}

void BaseMaterial3D::set_rim(float p_rim) {
	rim = p_rim;
	RS::get_singleton()->material_set_param(_get_material(), shader_names->rim, p_rim);
}

float BaseMaterial3D::get_rim() const {
	return rim;
}

void BaseMaterial3D::set_rim_tint(float p_rim_tint) {
	rim_tint = p_rim_tint;
	RS::get_singleton()->material_set_param(_get_material(), shader_names->rim_tint, p_rim_tint);
}

float BaseMaterial3D::get_rim_tint() const {
	return rim_tint;
}

void BaseMaterial3D::set_ao_light_affect(float p_ao_light_affect) {
	ao_light_affect = p_ao_light_affect;
	RS::get_singleton()->material_set_param(_get_material(), shader_names->ao_light_affect, p_ao_light_affect);
}

float BaseMaterial3D::get_ao_light_affect() const {
	return ao_light_affect;
}

void BaseMaterial3D::set_clearcoat(float p_clearcoat) {
	clearcoat = p_clearcoat;
	RS::get_singleton()->material_set_param(_get_material(), shader_names->clearcoat, p_clearcoat);
}

float BaseMaterial3D::get_clearcoat() const {
	return clearcoat;
}

void BaseMaterial3D::set_clearcoat_gloss(float p_clearcoat_gloss) {
	clearcoat_gloss = p_clearcoat_gloss;
	RS::get_singleton()->material_set_param(_get_material(), shader_names->clearcoat_gloss, p_clearcoat_gloss);
}

float BaseMaterial3D::get_clearcoat_gloss() const {
	return clearcoat_gloss;
}

void BaseMaterial3D::set_anisotropy(float p_anisotropy) {
	anisotropy = p_anisotropy;
	RS::get_singleton()->material_set_param(_get_material(), shader_names->anisotropy, p_anisotropy);
}

float BaseMaterial3D::get_anisotropy() const {
	return anisotropy;
}

void BaseMaterial3D::set_heightmap_scale(float p_heightmap_scale) {
	heightmap_scale = p_heightmap_scale;
	RS::get_singleton()->material_set_param(_get_material(), shader_names->heightmap_scale, p_heightmap_scale);
}

float BaseMaterial3D::get_heightmap_scale() const {
	return heightmap_scale;
}

void BaseMaterial3D::set_subsurface_scattering_strength(float p_subsurface_scattering_strength) {
	subsurface_scattering_strength = p_subsurface_scattering_strength;
	RS::get_singleton()->material_set_param(_get_material(), shader_names->subsurface_scattering_strength, subsurface_scattering_strength);
}

float BaseMaterial3D::get_subsurface_scattering_strength() const {
	return subsurface_scattering_strength;
}

void BaseMaterial3D::set_transmittance_color(const Color &p_color) {
	transmittance_color = p_color;
	RS::get_singleton()->material_set_param(_get_material(), shader_names->transmittance_color, p_color);
}

Color BaseMaterial3D::get_transmittance_color() const {
	return transmittance_color;
}

void BaseMaterial3D::set_transmittance_depth(float p_depth) {
	transmittance_depth = p_depth;
	RS::get_singleton()->material_set_param(_get_material(), shader_names->transmittance_depth, p_depth);
}

float BaseMaterial3D::get_transmittance_depth() const {
	return transmittance_depth;
}

void BaseMaterial3D::set_transmittance_boost(float p_boost) {
	transmittance_boost = p_boost;
	RS::get_singleton()->material_set_param(_get_material(), shader_names->transmittance_boost, p_boost);
}

float BaseMaterial3D::get_transmittance_boost() const {
	return transmittance_boost;
}

void BaseMaterial3D::set_backlight(const Color &p_backlight) {
	backlight = p_backlight;
	RS::get_singleton()->material_set_param(_get_material(), shader_names->backlight, backlight);
}

Color BaseMaterial3D::get_backlight() const {
	return backlight;
}

void BaseMaterial3D::set_refraction(float p_refraction) {
	refraction = p_refraction;
	RS::get_singleton()->material_set_param(_get_material(), shader_names->refraction, refraction);
}

float BaseMaterial3D::get_refraction() const {
	return refraction;
}

void BaseMaterial3D::set_detail_uv(DetailUV p_detail_uv) {
	if (detail_uv == p_detail_uv) {
		return;
	}

	detail_uv = p_detail_uv;
	_queue_shader_change();
}

BaseMaterial3D::DetailUV BaseMaterial3D::get_detail_uv() const {
	return detail_uv;
}

void BaseMaterial3D::set_blend_mode(BlendMode p_mode) {
	if (blend_mode == p_mode) {
		return;
	}

	blend_mode = p_mode;
	_queue_shader_change();
}

BaseMaterial3D::BlendMode BaseMaterial3D::get_blend_mode() const {
	return blend_mode;
}

void BaseMaterial3D::set_detail_blend_mode(BlendMode p_mode) {
	detail_blend_mode = p_mode;
	_queue_shader_change();
}

BaseMaterial3D::BlendMode BaseMaterial3D::get_detail_blend_mode() const {
	return detail_blend_mode;
}

void BaseMaterial3D::set_transparency(Transparency p_transparency) {
	if (transparency == p_transparency) {
		return;
	}

	transparency = p_transparency;
	_queue_shader_change();
	notify_property_list_changed();
}

BaseMaterial3D::Transparency BaseMaterial3D::get_transparency() const {
	return transparency;
}

void BaseMaterial3D::set_alpha_antialiasing(AlphaAntiAliasing p_alpha_aa) {
	if (alpha_antialiasing_mode == p_alpha_aa) {
		return;
	}

	alpha_antialiasing_mode = p_alpha_aa;
	_queue_shader_change();
	notify_property_list_changed();
}

BaseMaterial3D::AlphaAntiAliasing BaseMaterial3D::get_alpha_antialiasing() const {
	return alpha_antialiasing_mode;
}

void BaseMaterial3D::set_shading_mode(ShadingMode p_shading_mode) {
	if (shading_mode == p_shading_mode) {
		return;
	}

	shading_mode = p_shading_mode;
	_queue_shader_change();
	notify_property_list_changed();
}

BaseMaterial3D::ShadingMode BaseMaterial3D::get_shading_mode() const {
	return shading_mode;
}

void BaseMaterial3D::set_depth_draw_mode(DepthDrawMode p_mode) {
	if (depth_draw_mode == p_mode) {
		return;
	}

	depth_draw_mode = p_mode;
	_queue_shader_change();
}

BaseMaterial3D::DepthDrawMode BaseMaterial3D::get_depth_draw_mode() const {
	return depth_draw_mode;
}

void BaseMaterial3D::set_cull_mode(CullMode p_mode) {
	if (cull_mode == p_mode) {
		return;
	}

	cull_mode = p_mode;
	_queue_shader_change();
}

BaseMaterial3D::CullMode BaseMaterial3D::get_cull_mode() const {
	return cull_mode;
}

void BaseMaterial3D::set_diffuse_mode(DiffuseMode p_mode) {
	if (diffuse_mode == p_mode) {
		return;
	}

	diffuse_mode = p_mode;
	_queue_shader_change();
}

BaseMaterial3D::DiffuseMode BaseMaterial3D::get_diffuse_mode() const {
	return diffuse_mode;
}

void BaseMaterial3D::set_specular_mode(SpecularMode p_mode) {
	if (specular_mode == p_mode) {
		return;
	}

	specular_mode = p_mode;
	_queue_shader_change();
}

BaseMaterial3D::SpecularMode BaseMaterial3D::get_specular_mode() const {
	return specular_mode;
}

void BaseMaterial3D::set_flag(Flags p_flag, bool p_enabled) {
	ERR_FAIL_INDEX(p_flag, FLAG_MAX);

	if (flags[p_flag] == p_enabled) {
		return;
	}

	flags[p_flag] = p_enabled;
	if (p_flag == FLAG_USE_SHADOW_TO_OPACITY || p_flag == FLAG_USE_TEXTURE_REPEAT || p_flag == FLAG_SUBSURFACE_MODE_SKIN || p_flag == FLAG_USE_POINT_SIZE) {
		notify_property_list_changed();
	}
	if (p_flag == FLAG_PARTICLE_TRAILS_MODE) {
		update_configuration_warning();
	}
	_queue_shader_change();
}

bool BaseMaterial3D::get_flag(Flags p_flag) const {
	ERR_FAIL_INDEX_V(p_flag, FLAG_MAX, false);
	return flags[p_flag];
}

void BaseMaterial3D::set_feature(Feature p_feature, bool p_enabled) {
	ERR_FAIL_INDEX(p_feature, FEATURE_MAX);
	if (features[p_feature] == p_enabled) {
		return;
	}

	features[p_feature] = p_enabled;
	notify_property_list_changed();
	_queue_shader_change();
}

bool BaseMaterial3D::get_feature(Feature p_feature) const {
	ERR_FAIL_INDEX_V(p_feature, FEATURE_MAX, false);
	return features[p_feature];
}

void BaseMaterial3D::set_texture(TextureParam p_param, const Ref<Texture2D> &p_texture) {
	ERR_FAIL_INDEX(p_param, TEXTURE_MAX);

	if (get_texture(TEXTURE_ROUGHNESS).is_null() && p_texture.is_valid() && p_param == TEXTURE_ROUGHNESS) {
		// If no roughness texture is currently set, automatically set the recommended value
		// for roughness when using a roughness map.
		set_roughness(1.0);
	}

	if (get_texture(TEXTURE_METALLIC).is_null() && p_texture.is_valid() && p_param == TEXTURE_METALLIC) {
		// If no metallic texture is currently set, automatically set the recommended value
		// for metallic when using a metallic map.
		set_metallic(1.0);
	}

	textures[p_param] = p_texture;
	RID rid = p_texture.is_valid() ? p_texture->get_rid() : RID();
	RS::get_singleton()->material_set_param(_get_material(), shader_names->texture_names[p_param], rid);

	if (p_texture.is_valid() && p_param == TEXTURE_ALBEDO) {
		RS::get_singleton()->material_set_param(_get_material(), shader_names->albedo_texture_size,
				Vector2i(p_texture->get_width(), p_texture->get_height()));
	}

	notify_property_list_changed();
	_queue_shader_change();
}

Ref<Texture2D> BaseMaterial3D::get_texture(TextureParam p_param) const {
	ERR_FAIL_INDEX_V(p_param, TEXTURE_MAX, Ref<Texture2D>());
	return textures[p_param];
}

Ref<Texture2D> BaseMaterial3D::get_texture_by_name(StringName p_name) const {
	for (int i = 0; i < (int)BaseMaterial3D::TEXTURE_MAX; i++) {
		TextureParam param = TextureParam(i);
		if (p_name == shader_names->texture_names[param]) {
			return textures[param];
		}
	}
	return Ref<Texture2D>();
}

void BaseMaterial3D::set_texture_filter(TextureFilter p_filter) {
	texture_filter = p_filter;
	_queue_shader_change();
}

BaseMaterial3D::TextureFilter BaseMaterial3D::get_texture_filter() const {
	return texture_filter;
}

void BaseMaterial3D::_validate_feature(const String &text, Feature feature, PropertyInfo &property) const {
	if (property.name.begins_with(text) && property.name != text + "_enabled" && !features[feature]) {
		property.usage = PROPERTY_USAGE_NO_EDITOR;
	}
}

void BaseMaterial3D::_validate_high_end(const String &text, PropertyInfo &property) const {
	if (property.name.begins_with(text)) {
		property.usage |= PROPERTY_USAGE_HIGH_END_GFX;
	}
}

void BaseMaterial3D::_validate_property(PropertyInfo &property) const {
	_validate_feature("normal", FEATURE_NORMAL_MAPPING, property);
	_validate_feature("emission", FEATURE_EMISSION, property);
	_validate_feature("rim", FEATURE_RIM, property);
	_validate_feature("clearcoat", FEATURE_CLEARCOAT, property);
	_validate_feature("anisotropy", FEATURE_ANISOTROPY, property);
	_validate_feature("ao", FEATURE_AMBIENT_OCCLUSION, property);
	_validate_feature("heightmap", FEATURE_HEIGHT_MAPPING, property);
	_validate_feature("subsurf_scatter", FEATURE_SUBSURFACE_SCATTERING, property);
	_validate_feature("backlight", FEATURE_BACKLIGHT, property);
	_validate_feature("refraction", FEATURE_REFRACTION, property);
	_validate_feature("detail", FEATURE_DETAIL, property);

	_validate_high_end("refraction", property);
	_validate_high_end("subsurf_scatter", property);
	_validate_high_end("anisotropy", property);
	_validate_high_end("clearcoat", property);
	_validate_high_end("heightmap", property);

	if (property.name.begins_with("particles_anim_") && billboard_mode != BILLBOARD_PARTICLES) {
		property.usage = PROPERTY_USAGE_NONE;
	}

	if (property.name == "billboard_keep_scale" && billboard_mode == BILLBOARD_DISABLED) {
		property.usage = PROPERTY_USAGE_NO_EDITOR;
	}

	if (property.name == "grow_amount" && !grow_enabled) {
		property.usage = PROPERTY_USAGE_NO_EDITOR;
	}

	if (property.name == "point_size" && !flags[FLAG_USE_POINT_SIZE]) {
		property.usage = PROPERTY_USAGE_NO_EDITOR;
	}

	if (property.name == "proximity_fade_distance" && !proximity_fade_enabled) {
		property.usage = PROPERTY_USAGE_NO_EDITOR;
	}

	if ((property.name == "distance_fade_max_distance" || property.name == "distance_fade_min_distance") && distance_fade == DISTANCE_FADE_DISABLED) {
		property.usage = PROPERTY_USAGE_NO_EDITOR;
	}

	// you can only enable anti-aliasing (in materials) on alpha scissor and alpha hash
	const bool can_select_aa = (transparency == TRANSPARENCY_ALPHA_SCISSOR || transparency == TRANSPARENCY_ALPHA_HASH);
	// alpha anti aliasiasing is only enabled when you can select aa
	const bool alpha_aa_enabled = (alpha_antialiasing_mode != ALPHA_ANTIALIASING_OFF) && can_select_aa;

	// alpha scissor slider isn't needed when alpha antialiasing is enabled
	if (property.name == "alpha_scissor_threshold" && transparency != TRANSPARENCY_ALPHA_SCISSOR) {
		property.usage = PROPERTY_USAGE_NONE;
	}

	// alpha hash scale slider is only needed if transparency is alpha hash
	if (property.name == "alpha_hash_scale" && transparency != TRANSPARENCY_ALPHA_HASH) {
		property.usage = PROPERTY_USAGE_NONE;
	}

	if (property.name == "alpha_antialiasing_mode" && !can_select_aa) {
		property.usage = PROPERTY_USAGE_NONE;
	}

	// we can't choose an antialiasing mode if alpha isn't possible
	if (property.name == "alpha_antialiasing_edge" && !alpha_aa_enabled) {
		property.usage = PROPERTY_USAGE_NONE;
	}

	if (property.name == "blend_mode" && alpha_aa_enabled) {
		property.usage = PROPERTY_USAGE_NONE;
	}

	if ((property.name == "heightmap_min_layers" || property.name == "heightmap_max_layers") && !deep_parallax) {
		property.usage = PROPERTY_USAGE_NONE;
	}

	if (flags[FLAG_SUBSURFACE_MODE_SKIN] && (property.name == "subsurf_scatter_transmittance_color" || property.name == "subsurf_scatter_transmittance_texture")) {
		property.usage = PROPERTY_USAGE_NONE;
	}

	if (orm) {
		if (property.name == "shading_mode") {
			// Vertex not supported in ORM mode, since no individual roughness.
			property.hint_string = "Unshaded,Per-Pixel";
		}
		if (property.name.begins_with("roughness") || property.name.begins_with("metallic") || property.name.begins_with("ao_texture")) {
			property.usage = PROPERTY_USAGE_NONE;
		}

	} else {
		if (property.name == "orm_texture") {
			property.usage = PROPERTY_USAGE_NONE;
		}
	}

	if (shading_mode != SHADING_MODE_PER_PIXEL) {
		if (shading_mode != SHADING_MODE_PER_VERTEX) {
			//these may still work per vertex
			if (property.name.begins_with("ao")) {
				property.usage = PROPERTY_USAGE_NONE;
			}
			if (property.name.begins_with("emission")) {
				property.usage = PROPERTY_USAGE_NONE;
			}

			if (property.name.begins_with("metallic")) {
				property.usage = PROPERTY_USAGE_NONE;
			}
			if (property.name.begins_with("rim")) {
				property.usage = PROPERTY_USAGE_NONE;
			}

			if (property.name.begins_with("roughness")) {
				property.usage = PROPERTY_USAGE_NONE;
			}

			if (property.name.begins_with("subsurf_scatter")) {
				property.usage = PROPERTY_USAGE_NONE;
			}
		}

		//these definitely only need per pixel
		if (property.name.begins_with("anisotropy")) {
			property.usage = PROPERTY_USAGE_NONE;
		}

		if (property.name.begins_with("clearcoat")) {
			property.usage = PROPERTY_USAGE_NONE;
		}

		if (property.name.begins_with("normal")) {
			property.usage = PROPERTY_USAGE_NONE;
		}

		if (property.name.begins_with("backlight")) {
			property.usage = PROPERTY_USAGE_NONE;
		}

		if (property.name.begins_with("transmittance")) {
			property.usage = PROPERTY_USAGE_NONE;
		}
	}
}

void BaseMaterial3D::set_point_size(float p_point_size) {
	point_size = p_point_size;
	RS::get_singleton()->material_set_param(_get_material(), shader_names->point_size, p_point_size);
}

float BaseMaterial3D::get_point_size() const {
	return point_size;
}

void BaseMaterial3D::set_uv1_scale(const Vector3 &p_scale) {
	uv1_scale = p_scale;
	RS::get_singleton()->material_set_param(_get_material(), shader_names->uv1_scale, p_scale);
}

Vector3 BaseMaterial3D::get_uv1_scale() const {
	return uv1_scale;
}

void BaseMaterial3D::set_uv1_offset(const Vector3 &p_offset) {
	uv1_offset = p_offset;
	RS::get_singleton()->material_set_param(_get_material(), shader_names->uv1_offset, p_offset);
}

Vector3 BaseMaterial3D::get_uv1_offset() const {
	return uv1_offset;
}

void BaseMaterial3D::set_uv1_triplanar_blend_sharpness(float p_sharpness) {
	uv1_triplanar_sharpness = p_sharpness;
	RS::get_singleton()->material_set_param(_get_material(), shader_names->uv1_blend_sharpness, p_sharpness);
}

float BaseMaterial3D::get_uv1_triplanar_blend_sharpness() const {
	return uv1_triplanar_sharpness;
}

void BaseMaterial3D::set_uv2_scale(const Vector3 &p_scale) {
	uv2_scale = p_scale;
	RS::get_singleton()->material_set_param(_get_material(), shader_names->uv2_scale, p_scale);
}

Vector3 BaseMaterial3D::get_uv2_scale() const {
	return uv2_scale;
}

void BaseMaterial3D::set_uv2_offset(const Vector3 &p_offset) {
	uv2_offset = p_offset;
	RS::get_singleton()->material_set_param(_get_material(), shader_names->uv2_offset, p_offset);
}

Vector3 BaseMaterial3D::get_uv2_offset() const {
	return uv2_offset;
}

void BaseMaterial3D::set_uv2_triplanar_blend_sharpness(float p_sharpness) {
	uv2_triplanar_sharpness = p_sharpness;
	RS::get_singleton()->material_set_param(_get_material(), shader_names->uv2_blend_sharpness, p_sharpness);
}

float BaseMaterial3D::get_uv2_triplanar_blend_sharpness() const {
	return uv2_triplanar_sharpness;
}

void BaseMaterial3D::set_billboard_mode(BillboardMode p_mode) {
	billboard_mode = p_mode;
	_queue_shader_change();
	notify_property_list_changed();
}

BaseMaterial3D::BillboardMode BaseMaterial3D::get_billboard_mode() const {
	return billboard_mode;
}

void BaseMaterial3D::set_particles_anim_h_frames(int p_frames) {
	particles_anim_h_frames = p_frames;
	RS::get_singleton()->material_set_param(_get_material(), shader_names->particles_anim_h_frames, p_frames);
}

int BaseMaterial3D::get_particles_anim_h_frames() const {
	return particles_anim_h_frames;
}

void BaseMaterial3D::set_particles_anim_v_frames(int p_frames) {
	particles_anim_v_frames = p_frames;
	RS::get_singleton()->material_set_param(_get_material(), shader_names->particles_anim_v_frames, p_frames);
}

int BaseMaterial3D::get_particles_anim_v_frames() const {
	return particles_anim_v_frames;
}

void BaseMaterial3D::set_particles_anim_loop(bool p_loop) {
	particles_anim_loop = p_loop;
	RS::get_singleton()->material_set_param(_get_material(), shader_names->particles_anim_loop, particles_anim_loop);
}

bool BaseMaterial3D::get_particles_anim_loop() const {
	return particles_anim_loop;
}

void BaseMaterial3D::set_heightmap_deep_parallax(bool p_enable) {
	deep_parallax = p_enable;
	_queue_shader_change();
	notify_property_list_changed();
}

bool BaseMaterial3D::is_heightmap_deep_parallax_enabled() const {
	return deep_parallax;
}

void BaseMaterial3D::set_heightmap_deep_parallax_min_layers(int p_layer) {
	deep_parallax_min_layers = p_layer;
	RS::get_singleton()->material_set_param(_get_material(), shader_names->heightmap_min_layers, p_layer);
}

int BaseMaterial3D::get_heightmap_deep_parallax_min_layers() const {
	return deep_parallax_min_layers;
}

void BaseMaterial3D::set_heightmap_deep_parallax_max_layers(int p_layer) {
	deep_parallax_max_layers = p_layer;
	RS::get_singleton()->material_set_param(_get_material(), shader_names->heightmap_max_layers, p_layer);
}

int BaseMaterial3D::get_heightmap_deep_parallax_max_layers() const {
	return deep_parallax_max_layers;
}

void BaseMaterial3D::set_heightmap_deep_parallax_flip_tangent(bool p_flip) {
	heightmap_parallax_flip_tangent = p_flip;
	RS::get_singleton()->material_set_param(_get_material(), shader_names->heightmap_flip, Vector2(heightmap_parallax_flip_tangent ? -1 : 1, heightmap_parallax_flip_binormal ? -1 : 1));
}

bool BaseMaterial3D::get_heightmap_deep_parallax_flip_tangent() const {
	return heightmap_parallax_flip_tangent;
}

void BaseMaterial3D::set_heightmap_deep_parallax_flip_binormal(bool p_flip) {
	heightmap_parallax_flip_binormal = p_flip;
	RS::get_singleton()->material_set_param(_get_material(), shader_names->heightmap_flip, Vector2(heightmap_parallax_flip_tangent ? -1 : 1, heightmap_parallax_flip_binormal ? -1 : 1));
}

bool BaseMaterial3D::get_heightmap_deep_parallax_flip_binormal() const {
	return heightmap_parallax_flip_binormal;
}

void BaseMaterial3D::set_grow_enabled(bool p_enable) {
	grow_enabled = p_enable;
	_queue_shader_change();
	notify_property_list_changed();
}

bool BaseMaterial3D::is_grow_enabled() const {
	return grow_enabled;
}

void BaseMaterial3D::set_alpha_scissor_threshold(float p_threshold) {
	alpha_scissor_threshold = p_threshold;
	RS::get_singleton()->material_set_param(_get_material(), shader_names->alpha_scissor_threshold, p_threshold);
}

float BaseMaterial3D::get_alpha_scissor_threshold() const {
	return alpha_scissor_threshold;
}

void BaseMaterial3D::set_alpha_hash_scale(float p_scale) {
	alpha_hash_scale = p_scale;
	RS::get_singleton()->material_set_param(_get_material(), shader_names->alpha_hash_scale, p_scale);
}

float BaseMaterial3D::get_alpha_hash_scale() const {
	return alpha_hash_scale;
}

void BaseMaterial3D::set_alpha_antialiasing_edge(float p_edge) {
	alpha_antialiasing_edge = p_edge;
	RS::get_singleton()->material_set_param(_get_material(), shader_names->alpha_antialiasing_edge, p_edge);
}

float BaseMaterial3D::get_alpha_antialiasing_edge() const {
	return alpha_antialiasing_edge;
}

void BaseMaterial3D::set_grow(float p_grow) {
	grow = p_grow;
	RS::get_singleton()->material_set_param(_get_material(), shader_names->grow, p_grow);
}

float BaseMaterial3D::get_grow() const {
	return grow;
}

static Plane _get_texture_mask(BaseMaterial3D::TextureChannel p_channel) {
	static const Plane masks[5] = {
		Plane(1, 0, 0, 0),
		Plane(0, 1, 0, 0),
		Plane(0, 0, 1, 0),
		Plane(0, 0, 0, 1),
		Plane(0.3333333, 0.3333333, 0.3333333, 0),
	};

	return masks[p_channel];
}

void BaseMaterial3D::set_metallic_texture_channel(TextureChannel p_channel) {
	ERR_FAIL_INDEX(p_channel, 5);
	metallic_texture_channel = p_channel;
	RS::get_singleton()->material_set_param(_get_material(), shader_names->metallic_texture_channel, _get_texture_mask(p_channel));
}

BaseMaterial3D::TextureChannel BaseMaterial3D::get_metallic_texture_channel() const {
	return metallic_texture_channel;
}

void BaseMaterial3D::set_roughness_texture_channel(TextureChannel p_channel) {
	ERR_FAIL_INDEX(p_channel, 5);
	roughness_texture_channel = p_channel;
	_queue_shader_change();
}

BaseMaterial3D::TextureChannel BaseMaterial3D::get_roughness_texture_channel() const {
	return roughness_texture_channel;
}

void BaseMaterial3D::set_ao_texture_channel(TextureChannel p_channel) {
	ERR_FAIL_INDEX(p_channel, 5);
	ao_texture_channel = p_channel;
	RS::get_singleton()->material_set_param(_get_material(), shader_names->ao_texture_channel, _get_texture_mask(p_channel));
}

BaseMaterial3D::TextureChannel BaseMaterial3D::get_ao_texture_channel() const {
	return ao_texture_channel;
}

void BaseMaterial3D::set_refraction_texture_channel(TextureChannel p_channel) {
	ERR_FAIL_INDEX(p_channel, 5);
	refraction_texture_channel = p_channel;
	RS::get_singleton()->material_set_param(_get_material(), shader_names->refraction_texture_channel, _get_texture_mask(p_channel));
}

BaseMaterial3D::TextureChannel BaseMaterial3D::get_refraction_texture_channel() const {
	return refraction_texture_channel;
}

Ref<Material> BaseMaterial3D::get_material_for_2d(bool p_shaded, bool p_transparent, bool p_double_sided, bool p_cut_alpha, bool p_opaque_prepass, bool p_billboard, bool p_billboard_y, RID *r_shader_rid) {
	int version = 0;
	if (p_shaded) {
		version = 1;
	}
	if (p_transparent) {
		version |= 2;
	}
	if (p_cut_alpha) {
		version |= 4;
	}
	if (p_opaque_prepass) {
		version |= 8;
	}
	if (p_double_sided) {
		version |= 16;
	}
	if (p_billboard) {
		version |= 32;
	}
	if (p_billboard_y) {
		version |= 64;
	}

	if (materials_for_2d[version].is_valid()) {
		if (r_shader_rid) {
			*r_shader_rid = materials_for_2d[version]->get_shader_rid();
		}
		return materials_for_2d[version];
	}

	Ref<StandardMaterial3D> material;
	material.instantiate();

	material->set_shading_mode(p_shaded ? SHADING_MODE_PER_PIXEL : SHADING_MODE_UNSHADED);
	material->set_transparency(p_transparent ? (p_opaque_prepass ? TRANSPARENCY_ALPHA_DEPTH_PRE_PASS : (p_cut_alpha ? TRANSPARENCY_ALPHA_SCISSOR : TRANSPARENCY_ALPHA)) : TRANSPARENCY_DISABLED);
	material->set_cull_mode(p_double_sided ? CULL_DISABLED : CULL_BACK);
	material->set_flag(FLAG_SRGB_VERTEX_COLOR, true);
	material->set_flag(FLAG_ALBEDO_FROM_VERTEX_COLOR, true);
	if (p_billboard || p_billboard_y) {
		material->set_flag(FLAG_BILLBOARD_KEEP_SCALE, true);
		material->set_billboard_mode(p_billboard_y ? BILLBOARD_FIXED_Y : BILLBOARD_ENABLED);
	}

	materials_for_2d[version] = material;

	if (r_shader_rid) {
		*r_shader_rid = materials_for_2d[version]->get_shader_rid();
	}

	return materials_for_2d[version];
}

void BaseMaterial3D::set_on_top_of_alpha() {
	set_transparency(TRANSPARENCY_DISABLED);
	set_render_priority(RENDER_PRIORITY_MAX);
	set_flag(FLAG_DISABLE_DEPTH_TEST, true);
}

void BaseMaterial3D::set_proximity_fade(bool p_enable) {
	proximity_fade_enabled = p_enable;
	_queue_shader_change();
	notify_property_list_changed();
}

bool BaseMaterial3D::is_proximity_fade_enabled() const {
	return proximity_fade_enabled;
}

void BaseMaterial3D::set_proximity_fade_distance(float p_distance) {
	proximity_fade_distance = p_distance;
	RS::get_singleton()->material_set_param(_get_material(), shader_names->proximity_fade_distance, p_distance);
}

float BaseMaterial3D::get_proximity_fade_distance() const {
	return proximity_fade_distance;
}

void BaseMaterial3D::set_distance_fade(DistanceFadeMode p_mode) {
	distance_fade = p_mode;
	_queue_shader_change();
	notify_property_list_changed();
}

BaseMaterial3D::DistanceFadeMode BaseMaterial3D::get_distance_fade() const {
	return distance_fade;
}

void BaseMaterial3D::set_distance_fade_max_distance(float p_distance) {
	distance_fade_max_distance = p_distance;
	RS::get_singleton()->material_set_param(_get_material(), shader_names->distance_fade_max, distance_fade_max_distance);
}

float BaseMaterial3D::get_distance_fade_max_distance() const {
	return distance_fade_max_distance;
}

void BaseMaterial3D::set_distance_fade_min_distance(float p_distance) {
	distance_fade_min_distance = p_distance;
	RS::get_singleton()->material_set_param(_get_material(), shader_names->distance_fade_min, distance_fade_min_distance);
}

float BaseMaterial3D::get_distance_fade_min_distance() const {
	return distance_fade_min_distance;
}

void BaseMaterial3D::set_emission_operator(EmissionOperator p_op) {
	if (emission_op == p_op) {
		return;
	}
	emission_op = p_op;
	_queue_shader_change();
}

BaseMaterial3D::EmissionOperator BaseMaterial3D::get_emission_operator() const {
	return emission_op;
}

RID BaseMaterial3D::get_shader_rid() const {
	MutexLock lock(material_mutex);
	((BaseMaterial3D *)this)->_update_shader();
	ERR_FAIL_COND_V(!shader_map.has(current_key), RID());
	return shader_map[current_key].shader;
}

Shader::Mode BaseMaterial3D::get_shader_mode() const {
	return Shader::MODE_SPATIAL;
}

void BaseMaterial3D::_bind_methods() {
	static_assert(sizeof(MaterialKey) == 16, "MaterialKey should be 16 bytes");

	ClassDB::bind_method(D_METHOD("set_albedo", "albedo"), &BaseMaterial3D::set_albedo);
	ClassDB::bind_method(D_METHOD("get_albedo"), &BaseMaterial3D::get_albedo);

	ClassDB::bind_method(D_METHOD("set_transparency", "transparency"), &BaseMaterial3D::set_transparency);
	ClassDB::bind_method(D_METHOD("get_transparency"), &BaseMaterial3D::get_transparency);

	ClassDB::bind_method(D_METHOD("set_alpha_antialiasing", "alpha_aa"), &BaseMaterial3D::set_alpha_antialiasing);
	ClassDB::bind_method(D_METHOD("get_alpha_antialiasing"), &BaseMaterial3D::get_alpha_antialiasing);

	ClassDB::bind_method(D_METHOD("set_alpha_antialiasing_edge", "edge"), &BaseMaterial3D::set_alpha_antialiasing_edge);
	ClassDB::bind_method(D_METHOD("get_alpha_antialiasing_edge"), &BaseMaterial3D::get_alpha_antialiasing_edge);

	ClassDB::bind_method(D_METHOD("set_shading_mode", "shading_mode"), &BaseMaterial3D::set_shading_mode);
	ClassDB::bind_method(D_METHOD("get_shading_mode"), &BaseMaterial3D::get_shading_mode);

	ClassDB::bind_method(D_METHOD("set_specular", "specular"), &BaseMaterial3D::set_specular);
	ClassDB::bind_method(D_METHOD("get_specular"), &BaseMaterial3D::get_specular);

	ClassDB::bind_method(D_METHOD("set_metallic", "metallic"), &BaseMaterial3D::set_metallic);
	ClassDB::bind_method(D_METHOD("get_metallic"), &BaseMaterial3D::get_metallic);

	ClassDB::bind_method(D_METHOD("set_roughness", "roughness"), &BaseMaterial3D::set_roughness);
	ClassDB::bind_method(D_METHOD("get_roughness"), &BaseMaterial3D::get_roughness);

	ClassDB::bind_method(D_METHOD("set_emission", "emission"), &BaseMaterial3D::set_emission);
	ClassDB::bind_method(D_METHOD("get_emission"), &BaseMaterial3D::get_emission);

	ClassDB::bind_method(D_METHOD("set_emission_energy", "emission_energy"), &BaseMaterial3D::set_emission_energy);
	ClassDB::bind_method(D_METHOD("get_emission_energy"), &BaseMaterial3D::get_emission_energy);

	ClassDB::bind_method(D_METHOD("set_normal_scale", "normal_scale"), &BaseMaterial3D::set_normal_scale);
	ClassDB::bind_method(D_METHOD("get_normal_scale"), &BaseMaterial3D::get_normal_scale);

	ClassDB::bind_method(D_METHOD("set_rim", "rim"), &BaseMaterial3D::set_rim);
	ClassDB::bind_method(D_METHOD("get_rim"), &BaseMaterial3D::get_rim);

	ClassDB::bind_method(D_METHOD("set_rim_tint", "rim_tint"), &BaseMaterial3D::set_rim_tint);
	ClassDB::bind_method(D_METHOD("get_rim_tint"), &BaseMaterial3D::get_rim_tint);

	ClassDB::bind_method(D_METHOD("set_clearcoat", "clearcoat"), &BaseMaterial3D::set_clearcoat);
	ClassDB::bind_method(D_METHOD("get_clearcoat"), &BaseMaterial3D::get_clearcoat);

	ClassDB::bind_method(D_METHOD("set_clearcoat_gloss", "clearcoat_gloss"), &BaseMaterial3D::set_clearcoat_gloss);
	ClassDB::bind_method(D_METHOD("get_clearcoat_gloss"), &BaseMaterial3D::get_clearcoat_gloss);

	ClassDB::bind_method(D_METHOD("set_anisotropy", "anisotropy"), &BaseMaterial3D::set_anisotropy);
	ClassDB::bind_method(D_METHOD("get_anisotropy"), &BaseMaterial3D::get_anisotropy);

	ClassDB::bind_method(D_METHOD("set_heightmap_scale", "heightmap_scale"), &BaseMaterial3D::set_heightmap_scale);
	ClassDB::bind_method(D_METHOD("get_heightmap_scale"), &BaseMaterial3D::get_heightmap_scale);

	ClassDB::bind_method(D_METHOD("set_subsurface_scattering_strength", "strength"), &BaseMaterial3D::set_subsurface_scattering_strength);
	ClassDB::bind_method(D_METHOD("get_subsurface_scattering_strength"), &BaseMaterial3D::get_subsurface_scattering_strength);

	ClassDB::bind_method(D_METHOD("set_transmittance_color", "color"), &BaseMaterial3D::set_transmittance_color);
	ClassDB::bind_method(D_METHOD("get_transmittance_color"), &BaseMaterial3D::get_transmittance_color);

	ClassDB::bind_method(D_METHOD("set_transmittance_depth", "depth"), &BaseMaterial3D::set_transmittance_depth);
	ClassDB::bind_method(D_METHOD("get_transmittance_depth"), &BaseMaterial3D::get_transmittance_depth);

	ClassDB::bind_method(D_METHOD("set_transmittance_boost", "boost"), &BaseMaterial3D::set_transmittance_boost);
	ClassDB::bind_method(D_METHOD("get_transmittance_boost"), &BaseMaterial3D::get_transmittance_boost);

	ClassDB::bind_method(D_METHOD("set_backlight", "backlight"), &BaseMaterial3D::set_backlight);
	ClassDB::bind_method(D_METHOD("get_backlight"), &BaseMaterial3D::get_backlight);

	ClassDB::bind_method(D_METHOD("set_refraction", "refraction"), &BaseMaterial3D::set_refraction);
	ClassDB::bind_method(D_METHOD("get_refraction"), &BaseMaterial3D::get_refraction);

	ClassDB::bind_method(D_METHOD("set_point_size", "point_size"), &BaseMaterial3D::set_point_size);
	ClassDB::bind_method(D_METHOD("get_point_size"), &BaseMaterial3D::get_point_size);

	ClassDB::bind_method(D_METHOD("set_detail_uv", "detail_uv"), &BaseMaterial3D::set_detail_uv);
	ClassDB::bind_method(D_METHOD("get_detail_uv"), &BaseMaterial3D::get_detail_uv);

	ClassDB::bind_method(D_METHOD("set_blend_mode", "blend_mode"), &BaseMaterial3D::set_blend_mode);
	ClassDB::bind_method(D_METHOD("get_blend_mode"), &BaseMaterial3D::get_blend_mode);

	ClassDB::bind_method(D_METHOD("set_depth_draw_mode", "depth_draw_mode"), &BaseMaterial3D::set_depth_draw_mode);
	ClassDB::bind_method(D_METHOD("get_depth_draw_mode"), &BaseMaterial3D::get_depth_draw_mode);

	ClassDB::bind_method(D_METHOD("set_cull_mode", "cull_mode"), &BaseMaterial3D::set_cull_mode);
	ClassDB::bind_method(D_METHOD("get_cull_mode"), &BaseMaterial3D::get_cull_mode);

	ClassDB::bind_method(D_METHOD("set_diffuse_mode", "diffuse_mode"), &BaseMaterial3D::set_diffuse_mode);
	ClassDB::bind_method(D_METHOD("get_diffuse_mode"), &BaseMaterial3D::get_diffuse_mode);

	ClassDB::bind_method(D_METHOD("set_specular_mode", "specular_mode"), &BaseMaterial3D::set_specular_mode);
	ClassDB::bind_method(D_METHOD("get_specular_mode"), &BaseMaterial3D::get_specular_mode);

	ClassDB::bind_method(D_METHOD("set_flag", "flag", "enable"), &BaseMaterial3D::set_flag);
	ClassDB::bind_method(D_METHOD("get_flag", "flag"), &BaseMaterial3D::get_flag);

	ClassDB::bind_method(D_METHOD("set_texture_filter", "mode"), &BaseMaterial3D::set_texture_filter);
	ClassDB::bind_method(D_METHOD("get_texture_filter"), &BaseMaterial3D::get_texture_filter);

	ClassDB::bind_method(D_METHOD("set_feature", "feature", "enable"), &BaseMaterial3D::set_feature);
	ClassDB::bind_method(D_METHOD("get_feature", "feature"), &BaseMaterial3D::get_feature);

	ClassDB::bind_method(D_METHOD("set_texture", "param", "texture"), &BaseMaterial3D::set_texture);
	ClassDB::bind_method(D_METHOD("get_texture", "param"), &BaseMaterial3D::get_texture);

	ClassDB::bind_method(D_METHOD("set_detail_blend_mode", "detail_blend_mode"), &BaseMaterial3D::set_detail_blend_mode);
	ClassDB::bind_method(D_METHOD("get_detail_blend_mode"), &BaseMaterial3D::get_detail_blend_mode);

	ClassDB::bind_method(D_METHOD("set_uv1_scale", "scale"), &BaseMaterial3D::set_uv1_scale);
	ClassDB::bind_method(D_METHOD("get_uv1_scale"), &BaseMaterial3D::get_uv1_scale);

	ClassDB::bind_method(D_METHOD("set_uv1_offset", "offset"), &BaseMaterial3D::set_uv1_offset);
	ClassDB::bind_method(D_METHOD("get_uv1_offset"), &BaseMaterial3D::get_uv1_offset);

	ClassDB::bind_method(D_METHOD("set_uv1_triplanar_blend_sharpness", "sharpness"), &BaseMaterial3D::set_uv1_triplanar_blend_sharpness);
	ClassDB::bind_method(D_METHOD("get_uv1_triplanar_blend_sharpness"), &BaseMaterial3D::get_uv1_triplanar_blend_sharpness);

	ClassDB::bind_method(D_METHOD("set_uv2_scale", "scale"), &BaseMaterial3D::set_uv2_scale);
	ClassDB::bind_method(D_METHOD("get_uv2_scale"), &BaseMaterial3D::get_uv2_scale);

	ClassDB::bind_method(D_METHOD("set_uv2_offset", "offset"), &BaseMaterial3D::set_uv2_offset);
	ClassDB::bind_method(D_METHOD("get_uv2_offset"), &BaseMaterial3D::get_uv2_offset);

	ClassDB::bind_method(D_METHOD("set_uv2_triplanar_blend_sharpness", "sharpness"), &BaseMaterial3D::set_uv2_triplanar_blend_sharpness);
	ClassDB::bind_method(D_METHOD("get_uv2_triplanar_blend_sharpness"), &BaseMaterial3D::get_uv2_triplanar_blend_sharpness);

	ClassDB::bind_method(D_METHOD("set_billboard_mode", "mode"), &BaseMaterial3D::set_billboard_mode);
	ClassDB::bind_method(D_METHOD("get_billboard_mode"), &BaseMaterial3D::get_billboard_mode);

	ClassDB::bind_method(D_METHOD("set_particles_anim_h_frames", "frames"), &BaseMaterial3D::set_particles_anim_h_frames);
	ClassDB::bind_method(D_METHOD("get_particles_anim_h_frames"), &BaseMaterial3D::get_particles_anim_h_frames);

	ClassDB::bind_method(D_METHOD("set_particles_anim_v_frames", "frames"), &BaseMaterial3D::set_particles_anim_v_frames);
	ClassDB::bind_method(D_METHOD("get_particles_anim_v_frames"), &BaseMaterial3D::get_particles_anim_v_frames);

	ClassDB::bind_method(D_METHOD("set_particles_anim_loop", "loop"), &BaseMaterial3D::set_particles_anim_loop);
	ClassDB::bind_method(D_METHOD("get_particles_anim_loop"), &BaseMaterial3D::get_particles_anim_loop);

	ClassDB::bind_method(D_METHOD("set_heightmap_deep_parallax", "enable"), &BaseMaterial3D::set_heightmap_deep_parallax);
	ClassDB::bind_method(D_METHOD("is_heightmap_deep_parallax_enabled"), &BaseMaterial3D::is_heightmap_deep_parallax_enabled);

	ClassDB::bind_method(D_METHOD("set_heightmap_deep_parallax_min_layers", "layer"), &BaseMaterial3D::set_heightmap_deep_parallax_min_layers);
	ClassDB::bind_method(D_METHOD("get_heightmap_deep_parallax_min_layers"), &BaseMaterial3D::get_heightmap_deep_parallax_min_layers);

	ClassDB::bind_method(D_METHOD("set_heightmap_deep_parallax_max_layers", "layer"), &BaseMaterial3D::set_heightmap_deep_parallax_max_layers);
	ClassDB::bind_method(D_METHOD("get_heightmap_deep_parallax_max_layers"), &BaseMaterial3D::get_heightmap_deep_parallax_max_layers);

	ClassDB::bind_method(D_METHOD("set_heightmap_deep_parallax_flip_tangent", "flip"), &BaseMaterial3D::set_heightmap_deep_parallax_flip_tangent);
	ClassDB::bind_method(D_METHOD("get_heightmap_deep_parallax_flip_tangent"), &BaseMaterial3D::get_heightmap_deep_parallax_flip_tangent);

	ClassDB::bind_method(D_METHOD("set_heightmap_deep_parallax_flip_binormal", "flip"), &BaseMaterial3D::set_heightmap_deep_parallax_flip_binormal);
	ClassDB::bind_method(D_METHOD("get_heightmap_deep_parallax_flip_binormal"), &BaseMaterial3D::get_heightmap_deep_parallax_flip_binormal);

	ClassDB::bind_method(D_METHOD("set_grow", "amount"), &BaseMaterial3D::set_grow);
	ClassDB::bind_method(D_METHOD("get_grow"), &BaseMaterial3D::get_grow);

	ClassDB::bind_method(D_METHOD("set_emission_operator", "operator"), &BaseMaterial3D::set_emission_operator);
	ClassDB::bind_method(D_METHOD("get_emission_operator"), &BaseMaterial3D::get_emission_operator);

	ClassDB::bind_method(D_METHOD("set_ao_light_affect", "amount"), &BaseMaterial3D::set_ao_light_affect);
	ClassDB::bind_method(D_METHOD("get_ao_light_affect"), &BaseMaterial3D::get_ao_light_affect);

	ClassDB::bind_method(D_METHOD("set_alpha_scissor_threshold", "threshold"), &BaseMaterial3D::set_alpha_scissor_threshold);
	ClassDB::bind_method(D_METHOD("get_alpha_scissor_threshold"), &BaseMaterial3D::get_alpha_scissor_threshold);

	ClassDB::bind_method(D_METHOD("set_alpha_hash_scale", "threshold"), &BaseMaterial3D::set_alpha_hash_scale);
	ClassDB::bind_method(D_METHOD("get_alpha_hash_scale"), &BaseMaterial3D::get_alpha_hash_scale);

	ClassDB::bind_method(D_METHOD("set_grow_enabled", "enable"), &BaseMaterial3D::set_grow_enabled);
	ClassDB::bind_method(D_METHOD("is_grow_enabled"), &BaseMaterial3D::is_grow_enabled);

	ClassDB::bind_method(D_METHOD("set_metallic_texture_channel", "channel"), &BaseMaterial3D::set_metallic_texture_channel);
	ClassDB::bind_method(D_METHOD("get_metallic_texture_channel"), &BaseMaterial3D::get_metallic_texture_channel);

	ClassDB::bind_method(D_METHOD("set_roughness_texture_channel", "channel"), &BaseMaterial3D::set_roughness_texture_channel);
	ClassDB::bind_method(D_METHOD("get_roughness_texture_channel"), &BaseMaterial3D::get_roughness_texture_channel);

	ClassDB::bind_method(D_METHOD("set_ao_texture_channel", "channel"), &BaseMaterial3D::set_ao_texture_channel);
	ClassDB::bind_method(D_METHOD("get_ao_texture_channel"), &BaseMaterial3D::get_ao_texture_channel);

	ClassDB::bind_method(D_METHOD("set_refraction_texture_channel", "channel"), &BaseMaterial3D::set_refraction_texture_channel);
	ClassDB::bind_method(D_METHOD("get_refraction_texture_channel"), &BaseMaterial3D::get_refraction_texture_channel);

	ClassDB::bind_method(D_METHOD("set_proximity_fade", "enabled"), &BaseMaterial3D::set_proximity_fade);
	ClassDB::bind_method(D_METHOD("is_proximity_fade_enabled"), &BaseMaterial3D::is_proximity_fade_enabled);

	ClassDB::bind_method(D_METHOD("set_proximity_fade_distance", "distance"), &BaseMaterial3D::set_proximity_fade_distance);
	ClassDB::bind_method(D_METHOD("get_proximity_fade_distance"), &BaseMaterial3D::get_proximity_fade_distance);

	ClassDB::bind_method(D_METHOD("set_distance_fade", "mode"), &BaseMaterial3D::set_distance_fade);
	ClassDB::bind_method(D_METHOD("get_distance_fade"), &BaseMaterial3D::get_distance_fade);

	ClassDB::bind_method(D_METHOD("set_distance_fade_max_distance", "distance"), &BaseMaterial3D::set_distance_fade_max_distance);
	ClassDB::bind_method(D_METHOD("get_distance_fade_max_distance"), &BaseMaterial3D::get_distance_fade_max_distance);

	ClassDB::bind_method(D_METHOD("set_distance_fade_min_distance", "distance"), &BaseMaterial3D::set_distance_fade_min_distance);
	ClassDB::bind_method(D_METHOD("get_distance_fade_min_distance"), &BaseMaterial3D::get_distance_fade_min_distance);

	ADD_GROUP("Transparency", "");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "transparency", PROPERTY_HINT_ENUM, "Disabled,Alpha,Alpha Scissor,Alpha Hash,Depth Pre-Pass"), "set_transparency", "get_transparency");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "alpha_scissor_threshold", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_alpha_scissor_threshold", "get_alpha_scissor_threshold");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "alpha_hash_scale", PROPERTY_HINT_RANGE, "0,2,0.01"), "set_alpha_hash_scale", "get_alpha_hash_scale");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "alpha_antialiasing_mode", PROPERTY_HINT_ENUM, "Disabled,Alpha Edge Blend,Alpha Edge Clip"), "set_alpha_antialiasing", "get_alpha_antialiasing");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "alpha_antialiasing_edge", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_alpha_antialiasing_edge", "get_alpha_antialiasing_edge");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "blend_mode", PROPERTY_HINT_ENUM, "Mix,Add,Subtract,Multiply"), "set_blend_mode", "get_blend_mode");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "cull_mode", PROPERTY_HINT_ENUM, "Back,Front,Disabled"), "set_cull_mode", "get_cull_mode");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "depth_draw_mode", PROPERTY_HINT_ENUM, "Opaque Only,Always,Never"), "set_depth_draw_mode", "get_depth_draw_mode");
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "no_depth_test"), "set_flag", "get_flag", FLAG_DISABLE_DEPTH_TEST);

	ADD_GROUP("Shading", "");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "shading_mode", PROPERTY_HINT_ENUM, "Unshaded,Per-Pixel,Per-Vertex"), "set_shading_mode", "get_shading_mode");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "diffuse_mode", PROPERTY_HINT_ENUM, "Burley,Lambert,Lambert Wrap,Toon"), "set_diffuse_mode", "get_diffuse_mode");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "specular_mode", PROPERTY_HINT_ENUM, "SchlickGGX,Blinn,Phong,Toon,Disabled"), "set_specular_mode", "get_specular_mode");
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "disable_ambient_light"), "set_flag", "get_flag", FLAG_DISABLE_AMBIENT_LIGHT);

	ADD_GROUP("Vertex Color", "vertex_color");
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "vertex_color_use_as_albedo"), "set_flag", "get_flag", FLAG_ALBEDO_FROM_VERTEX_COLOR);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "vertex_color_is_srgb"), "set_flag", "get_flag", FLAG_SRGB_VERTEX_COLOR);

	ADD_GROUP("Albedo", "albedo_");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "albedo_color"), "set_albedo", "get_albedo");
	ADD_PROPERTYI(PropertyInfo(Variant::OBJECT, "albedo_texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture2D"), "set_texture", "get_texture", TEXTURE_ALBEDO);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "albedo_tex_force_srgb"), "set_flag", "get_flag", FLAG_ALBEDO_TEXTURE_FORCE_SRGB);

	ADD_GROUP("ORM", "orm_");
	ADD_PROPERTYI(PropertyInfo(Variant::OBJECT, "orm_texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture2D"), "set_texture", "get_texture", TEXTURE_ORM);

	ADD_GROUP("Metallic", "metallic_");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "metallic", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_metallic", "get_metallic");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "metallic_specular", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_specular", "get_specular");
	ADD_PROPERTYI(PropertyInfo(Variant::OBJECT, "metallic_texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture2D"), "set_texture", "get_texture", TEXTURE_METALLIC);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "metallic_texture_channel", PROPERTY_HINT_ENUM, "Red,Green,Blue,Alpha,Gray"), "set_metallic_texture_channel", "get_metallic_texture_channel");

	ADD_GROUP("Roughness", "roughness_");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "roughness", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_roughness", "get_roughness");
	ADD_PROPERTYI(PropertyInfo(Variant::OBJECT, "roughness_texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture2D"), "set_texture", "get_texture", TEXTURE_ROUGHNESS);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "roughness_texture_channel", PROPERTY_HINT_ENUM, "Red,Green,Blue,Alpha,Gray"), "set_roughness_texture_channel", "get_roughness_texture_channel");

	ADD_GROUP("Emission", "emission_");
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "emission_enabled"), "set_feature", "get_feature", FEATURE_EMISSION);
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "emission", PROPERTY_HINT_COLOR_NO_ALPHA), "set_emission", "get_emission");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "emission_energy", PROPERTY_HINT_RANGE, "0,16,0.01,or_greater"), "set_emission_energy", "get_emission_energy");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "emission_operator", PROPERTY_HINT_ENUM, "Add,Multiply"), "set_emission_operator", "get_emission_operator");
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "emission_on_uv2"), "set_flag", "get_flag", FLAG_EMISSION_ON_UV2);
	ADD_PROPERTYI(PropertyInfo(Variant::OBJECT, "emission_texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture2D"), "set_texture", "get_texture", TEXTURE_EMISSION);

	ADD_GROUP("NormalMap", "normal_");
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "normal_enabled"), "set_feature", "get_feature", FEATURE_NORMAL_MAPPING);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "normal_scale", PROPERTY_HINT_RANGE, "-16,16,0.01"), "set_normal_scale", "get_normal_scale");
	ADD_PROPERTYI(PropertyInfo(Variant::OBJECT, "normal_texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture2D"), "set_texture", "get_texture", TEXTURE_NORMAL);

	ADD_GROUP("Rim", "rim_");
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "rim_enabled"), "set_feature", "get_feature", FEATURE_RIM);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "rim", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_rim", "get_rim");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "rim_tint", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_rim_tint", "get_rim_tint");
	ADD_PROPERTYI(PropertyInfo(Variant::OBJECT, "rim_texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture2D"), "set_texture", "get_texture", TEXTURE_RIM);

	ADD_GROUP("Clearcoat", "clearcoat_");
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "clearcoat_enabled"), "set_feature", "get_feature", FEATURE_CLEARCOAT);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "clearcoat", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_clearcoat", "get_clearcoat");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "clearcoat_gloss", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_clearcoat_gloss", "get_clearcoat_gloss");
	ADD_PROPERTYI(PropertyInfo(Variant::OBJECT, "clearcoat_texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture2D"), "set_texture", "get_texture", TEXTURE_CLEARCOAT);

	ADD_GROUP("Anisotropy", "anisotropy_");
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "anisotropy_enabled"), "set_feature", "get_feature", FEATURE_ANISOTROPY);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "anisotropy", PROPERTY_HINT_RANGE, "-1,1,0.01"), "set_anisotropy", "get_anisotropy");
	ADD_PROPERTYI(PropertyInfo(Variant::OBJECT, "anisotropy_flowmap", PROPERTY_HINT_RESOURCE_TYPE, "Texture2D"), "set_texture", "get_texture", TEXTURE_FLOWMAP);

	ADD_GROUP("Ambient Occlusion", "ao_");
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "ao_enabled"), "set_feature", "get_feature", FEATURE_AMBIENT_OCCLUSION);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "ao_light_affect", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_ao_light_affect", "get_ao_light_affect");
	ADD_PROPERTYI(PropertyInfo(Variant::OBJECT, "ao_texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture2D"), "set_texture", "get_texture", TEXTURE_AMBIENT_OCCLUSION);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "ao_on_uv2"), "set_flag", "get_flag", FLAG_AO_ON_UV2);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "ao_texture_channel", PROPERTY_HINT_ENUM, "Red,Green,Blue,Alpha,Gray"), "set_ao_texture_channel", "get_ao_texture_channel");

	ADD_GROUP("Height", "heightmap_");
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "heightmap_enabled"), "set_feature", "get_feature", FEATURE_HEIGHT_MAPPING);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "heightmap_scale", PROPERTY_HINT_RANGE, "-16,16,0.001"), "set_heightmap_scale", "get_heightmap_scale");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "heightmap_deep_parallax"), "set_heightmap_deep_parallax", "is_heightmap_deep_parallax_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "heightmap_min_layers", PROPERTY_HINT_RANGE, "1,64,1"), "set_heightmap_deep_parallax_min_layers", "get_heightmap_deep_parallax_min_layers");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "heightmap_max_layers", PROPERTY_HINT_RANGE, "1,64,1"), "set_heightmap_deep_parallax_max_layers", "get_heightmap_deep_parallax_max_layers");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "heightmap_flip_tangent"), "set_heightmap_deep_parallax_flip_tangent", "get_heightmap_deep_parallax_flip_tangent");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "heightmap_flip_binormal"), "set_heightmap_deep_parallax_flip_binormal", "get_heightmap_deep_parallax_flip_binormal");
	ADD_PROPERTYI(PropertyInfo(Variant::OBJECT, "heightmap_texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture2D"), "set_texture", "get_texture", TEXTURE_HEIGHTMAP);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "heightmap_flip_texture"), "set_flag", "get_flag", FLAG_INVERT_HEIGHTMAP);

	ADD_GROUP("Subsurf Scatter", "subsurf_scatter_");
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "subsurf_scatter_enabled"), "set_feature", "get_feature", FEATURE_SUBSURFACE_SCATTERING);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "subsurf_scatter_strength", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_subsurface_scattering_strength", "get_subsurface_scattering_strength");
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "subsurf_scatter_skin_mode"), "set_flag", "get_flag", FLAG_SUBSURFACE_MODE_SKIN);
	ADD_PROPERTYI(PropertyInfo(Variant::OBJECT, "subsurf_scatter_texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture2D"), "set_texture", "get_texture", TEXTURE_SUBSURFACE_SCATTERING);

	ADD_SUBGROUP("Transmittance", "subsurf_scatter_transmittance_");
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "subsurf_scatter_transmittance_enabled"), "set_feature", "get_feature", FEATURE_SUBSURFACE_TRANSMITTANCE);
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "subsurf_scatter_transmittance_color"), "set_transmittance_color", "get_transmittance_color");
	ADD_PROPERTYI(PropertyInfo(Variant::OBJECT, "subsurf_scatter_transmittance_texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture2D"), "set_texture", "get_texture", TEXTURE_SUBSURFACE_TRANSMITTANCE);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "subsurf_scatter_transmittance_depth", PROPERTY_HINT_RANGE, "0.001,8,0.001,or_greater"), "set_transmittance_depth", "get_transmittance_depth");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "subsurf_scatter_transmittance_boost", PROPERTY_HINT_RANGE, "0.00,1.0,0.01"), "set_transmittance_boost", "get_transmittance_boost");

	ADD_GROUP("Back Lighting", "backlight_");
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "backlight_enabled"), "set_feature", "get_feature", FEATURE_BACKLIGHT);
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "backlight", PROPERTY_HINT_COLOR_NO_ALPHA), "set_backlight", "get_backlight");
	ADD_PROPERTYI(PropertyInfo(Variant::OBJECT, "backlight_texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture2D"), "set_texture", "get_texture", TEXTURE_BACKLIGHT);

	ADD_GROUP("Refraction", "refraction_");
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "refraction_enabled"), "set_feature", "get_feature", FEATURE_REFRACTION);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "refraction_scale", PROPERTY_HINT_RANGE, "-1,1,0.01"), "set_refraction", "get_refraction");
	ADD_PROPERTYI(PropertyInfo(Variant::OBJECT, "refraction_texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture2D"), "set_texture", "get_texture", TEXTURE_REFRACTION);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "refraction_texture_channel", PROPERTY_HINT_ENUM, "Red,Green,Blue,Alpha,Gray"), "set_refraction_texture_channel", "get_refraction_texture_channel");

	ADD_GROUP("Detail", "detail_");
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "detail_enabled"), "set_feature", "get_feature", FEATURE_DETAIL);
	ADD_PROPERTYI(PropertyInfo(Variant::OBJECT, "detail_mask", PROPERTY_HINT_RESOURCE_TYPE, "Texture2D"), "set_texture", "get_texture", TEXTURE_DETAIL_MASK);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "detail_blend_mode", PROPERTY_HINT_ENUM, "Mix,Add,Subtract,Multiply"), "set_detail_blend_mode", "get_detail_blend_mode");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "detail_uv_layer", PROPERTY_HINT_ENUM, "UV1,UV2"), "set_detail_uv", "get_detail_uv");
	ADD_PROPERTYI(PropertyInfo(Variant::OBJECT, "detail_albedo", PROPERTY_HINT_RESOURCE_TYPE, "Texture2D"), "set_texture", "get_texture", TEXTURE_DETAIL_ALBEDO);
	ADD_PROPERTYI(PropertyInfo(Variant::OBJECT, "detail_normal", PROPERTY_HINT_RESOURCE_TYPE, "Texture2D"), "set_texture", "get_texture", TEXTURE_DETAIL_NORMAL);

	ADD_GROUP("UV1", "uv1_");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "uv1_scale"), "set_uv1_scale", "get_uv1_scale");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "uv1_offset"), "set_uv1_offset", "get_uv1_offset");
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "uv1_triplanar"), "set_flag", "get_flag", FLAG_UV1_USE_TRIPLANAR);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "uv1_triplanar_sharpness", PROPERTY_HINT_EXP_EASING), "set_uv1_triplanar_blend_sharpness", "get_uv1_triplanar_blend_sharpness");
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "uv1_world_triplanar"), "set_flag", "get_flag", FLAG_UV1_USE_WORLD_TRIPLANAR);

	ADD_GROUP("UV2", "uv2_");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "uv2_scale"), "set_uv2_scale", "get_uv2_scale");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "uv2_offset"), "set_uv2_offset", "get_uv2_offset");
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "uv2_triplanar"), "set_flag", "get_flag", FLAG_UV2_USE_TRIPLANAR);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "uv2_triplanar_sharpness", PROPERTY_HINT_EXP_EASING), "set_uv2_triplanar_blend_sharpness", "get_uv2_triplanar_blend_sharpness");
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "uv2_world_triplanar"), "set_flag", "get_flag", FLAG_UV2_USE_WORLD_TRIPLANAR);

	ADD_GROUP("Sampling", "texture_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "texture_filter", PROPERTY_HINT_ENUM, "Nearest,Linear,Nearest Mipmap,Linear Mipmap,Nearest Mipmap Aniso.,Linear Mipmap Aniso."), "set_texture_filter", "get_texture_filter");
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "texture_repeat"), "set_flag", "get_flag", FLAG_USE_TEXTURE_REPEAT);

	ADD_GROUP("Shadows", "");
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "disable_receive_shadows"), "set_flag", "get_flag", FLAG_DONT_RECEIVE_SHADOWS);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "shadow_to_opacity"), "set_flag", "get_flag", FLAG_USE_SHADOW_TO_OPACITY);

	ADD_GROUP("Billboard", "billboard_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "billboard_mode", PROPERTY_HINT_ENUM, "Disabled,Enabled,Y-Billboard,Particle Billboard"), "set_billboard_mode", "get_billboard_mode");
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "billboard_keep_scale"), "set_flag", "get_flag", FLAG_BILLBOARD_KEEP_SCALE);

	ADD_GROUP("Particles Anim", "particles_anim_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "particles_anim_h_frames", PROPERTY_HINT_RANGE, "1,128,1"), "set_particles_anim_h_frames", "get_particles_anim_h_frames");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "particles_anim_v_frames", PROPERTY_HINT_RANGE, "1,128,1"), "set_particles_anim_v_frames", "get_particles_anim_v_frames");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "particles_anim_loop"), "set_particles_anim_loop", "get_particles_anim_loop");

	ADD_GROUP("Grow", "grow_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "grow"), "set_grow_enabled", "is_grow_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "grow_amount", PROPERTY_HINT_RANGE, "-16,16,0.001"), "set_grow", "get_grow");
	ADD_GROUP("Transform", "");
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "fixed_size"), "set_flag", "get_flag", FLAG_FIXED_SIZE);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "use_point_size"), "set_flag", "get_flag", FLAG_USE_POINT_SIZE);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "point_size", PROPERTY_HINT_RANGE, "0.1,128,0.1"), "set_point_size", "get_point_size");
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "use_particle_trails"), "set_flag", "get_flag", FLAG_PARTICLE_TRAILS_MODE);
	ADD_GROUP("Proximity Fade", "proximity_fade_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "proximity_fade_enable"), "set_proximity_fade", "is_proximity_fade_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "proximity_fade_distance", PROPERTY_HINT_RANGE, "0,4096,0.01"), "set_proximity_fade_distance", "get_proximity_fade_distance");
	ADD_GROUP("Distance Fade", "distance_fade_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "distance_fade_mode", PROPERTY_HINT_ENUM, "Disabled,PixelAlpha,PixelDither,ObjectDither"), "set_distance_fade", "get_distance_fade");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "distance_fade_min_distance", PROPERTY_HINT_RANGE, "0,4096,0.01"), "set_distance_fade_min_distance", "get_distance_fade_min_distance");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "distance_fade_max_distance", PROPERTY_HINT_RANGE, "0,4096,0.01"), "set_distance_fade_max_distance", "get_distance_fade_max_distance");

	BIND_ENUM_CONSTANT(TEXTURE_ALBEDO);
	BIND_ENUM_CONSTANT(TEXTURE_METALLIC);
	BIND_ENUM_CONSTANT(TEXTURE_ROUGHNESS);
	BIND_ENUM_CONSTANT(TEXTURE_EMISSION);
	BIND_ENUM_CONSTANT(TEXTURE_NORMAL);
	BIND_ENUM_CONSTANT(TEXTURE_RIM);
	BIND_ENUM_CONSTANT(TEXTURE_CLEARCOAT);
	BIND_ENUM_CONSTANT(TEXTURE_FLOWMAP);
	BIND_ENUM_CONSTANT(TEXTURE_AMBIENT_OCCLUSION);
	BIND_ENUM_CONSTANT(TEXTURE_HEIGHTMAP);
	BIND_ENUM_CONSTANT(TEXTURE_SUBSURFACE_SCATTERING);
	BIND_ENUM_CONSTANT(TEXTURE_SUBSURFACE_TRANSMITTANCE);
	BIND_ENUM_CONSTANT(TEXTURE_BACKLIGHT);
	BIND_ENUM_CONSTANT(TEXTURE_REFRACTION);
	BIND_ENUM_CONSTANT(TEXTURE_DETAIL_MASK);
	BIND_ENUM_CONSTANT(TEXTURE_DETAIL_ALBEDO);
	BIND_ENUM_CONSTANT(TEXTURE_DETAIL_NORMAL);
	BIND_ENUM_CONSTANT(TEXTURE_ORM);
	BIND_ENUM_CONSTANT(TEXTURE_MAX);

	BIND_ENUM_CONSTANT(TEXTURE_FILTER_NEAREST);
	BIND_ENUM_CONSTANT(TEXTURE_FILTER_LINEAR);
	BIND_ENUM_CONSTANT(TEXTURE_FILTER_NEAREST_WITH_MIPMAPS);
	BIND_ENUM_CONSTANT(TEXTURE_FILTER_LINEAR_WITH_MIPMAPS);
	BIND_ENUM_CONSTANT(TEXTURE_FILTER_NEAREST_WITH_MIPMAPS_ANISOTROPIC);
	BIND_ENUM_CONSTANT(TEXTURE_FILTER_LINEAR_WITH_MIPMAPS_ANISOTROPIC);
	BIND_ENUM_CONSTANT(TEXTURE_FILTER_MAX);

	BIND_ENUM_CONSTANT(DETAIL_UV_1);
	BIND_ENUM_CONSTANT(DETAIL_UV_2);

	BIND_ENUM_CONSTANT(TRANSPARENCY_DISABLED);
	BIND_ENUM_CONSTANT(TRANSPARENCY_ALPHA);
	BIND_ENUM_CONSTANT(TRANSPARENCY_ALPHA_SCISSOR);
	BIND_ENUM_CONSTANT(TRANSPARENCY_ALPHA_HASH);
	BIND_ENUM_CONSTANT(TRANSPARENCY_ALPHA_DEPTH_PRE_PASS);
	BIND_ENUM_CONSTANT(TRANSPARENCY_MAX);

	BIND_ENUM_CONSTANT(SHADING_MODE_UNSHADED);
	BIND_ENUM_CONSTANT(SHADING_MODE_PER_PIXEL);
	BIND_ENUM_CONSTANT(SHADING_MODE_PER_VERTEX);
	BIND_ENUM_CONSTANT(SHADING_MODE_MAX);

	BIND_ENUM_CONSTANT(FEATURE_EMISSION);
	BIND_ENUM_CONSTANT(FEATURE_NORMAL_MAPPING);
	BIND_ENUM_CONSTANT(FEATURE_RIM);
	BIND_ENUM_CONSTANT(FEATURE_CLEARCOAT);
	BIND_ENUM_CONSTANT(FEATURE_ANISOTROPY);
	BIND_ENUM_CONSTANT(FEATURE_AMBIENT_OCCLUSION);
	BIND_ENUM_CONSTANT(FEATURE_HEIGHT_MAPPING);
	BIND_ENUM_CONSTANT(FEATURE_SUBSURFACE_SCATTERING);
	BIND_ENUM_CONSTANT(FEATURE_SUBSURFACE_TRANSMITTANCE);
	BIND_ENUM_CONSTANT(FEATURE_BACKLIGHT);
	BIND_ENUM_CONSTANT(FEATURE_REFRACTION);
	BIND_ENUM_CONSTANT(FEATURE_DETAIL);
	BIND_ENUM_CONSTANT(FEATURE_MAX);

	BIND_ENUM_CONSTANT(BLEND_MODE_MIX);
	BIND_ENUM_CONSTANT(BLEND_MODE_ADD);
	BIND_ENUM_CONSTANT(BLEND_MODE_SUB);
	BIND_ENUM_CONSTANT(BLEND_MODE_MUL);

	BIND_ENUM_CONSTANT(ALPHA_ANTIALIASING_OFF);
	BIND_ENUM_CONSTANT(ALPHA_ANTIALIASING_ALPHA_TO_COVERAGE);
	BIND_ENUM_CONSTANT(ALPHA_ANTIALIASING_ALPHA_TO_COVERAGE_AND_TO_ONE);

	BIND_ENUM_CONSTANT(DEPTH_DRAW_OPAQUE_ONLY);
	BIND_ENUM_CONSTANT(DEPTH_DRAW_ALWAYS);
	BIND_ENUM_CONSTANT(DEPTH_DRAW_DISABLED);

	BIND_ENUM_CONSTANT(CULL_BACK);
	BIND_ENUM_CONSTANT(CULL_FRONT);
	BIND_ENUM_CONSTANT(CULL_DISABLED);

	BIND_ENUM_CONSTANT(FLAG_DISABLE_DEPTH_TEST);
	BIND_ENUM_CONSTANT(FLAG_ALBEDO_FROM_VERTEX_COLOR);
	BIND_ENUM_CONSTANT(FLAG_SRGB_VERTEX_COLOR);
	BIND_ENUM_CONSTANT(FLAG_USE_POINT_SIZE);
	BIND_ENUM_CONSTANT(FLAG_FIXED_SIZE);
	BIND_ENUM_CONSTANT(FLAG_BILLBOARD_KEEP_SCALE);
	BIND_ENUM_CONSTANT(FLAG_UV1_USE_TRIPLANAR);
	BIND_ENUM_CONSTANT(FLAG_UV2_USE_TRIPLANAR);
	BIND_ENUM_CONSTANT(FLAG_UV1_USE_WORLD_TRIPLANAR);
	BIND_ENUM_CONSTANT(FLAG_UV2_USE_WORLD_TRIPLANAR);
	BIND_ENUM_CONSTANT(FLAG_AO_ON_UV2);
	BIND_ENUM_CONSTANT(FLAG_EMISSION_ON_UV2);
	BIND_ENUM_CONSTANT(FLAG_ALBEDO_TEXTURE_FORCE_SRGB);
	BIND_ENUM_CONSTANT(FLAG_DONT_RECEIVE_SHADOWS);
	BIND_ENUM_CONSTANT(FLAG_DISABLE_AMBIENT_LIGHT);
	BIND_ENUM_CONSTANT(FLAG_USE_SHADOW_TO_OPACITY);
	BIND_ENUM_CONSTANT(FLAG_USE_TEXTURE_REPEAT);
	BIND_ENUM_CONSTANT(FLAG_INVERT_HEIGHTMAP);
	BIND_ENUM_CONSTANT(FLAG_SUBSURFACE_MODE_SKIN);
	BIND_ENUM_CONSTANT(FLAG_PARTICLE_TRAILS_MODE);
	BIND_ENUM_CONSTANT(FLAG_MAX);

	BIND_ENUM_CONSTANT(DIFFUSE_BURLEY);
	BIND_ENUM_CONSTANT(DIFFUSE_LAMBERT);
	BIND_ENUM_CONSTANT(DIFFUSE_LAMBERT_WRAP);
	BIND_ENUM_CONSTANT(DIFFUSE_TOON);

	BIND_ENUM_CONSTANT(SPECULAR_SCHLICK_GGX);
	BIND_ENUM_CONSTANT(SPECULAR_BLINN);
	BIND_ENUM_CONSTANT(SPECULAR_PHONG);
	BIND_ENUM_CONSTANT(SPECULAR_TOON);
	BIND_ENUM_CONSTANT(SPECULAR_DISABLED);

	BIND_ENUM_CONSTANT(BILLBOARD_DISABLED);
	BIND_ENUM_CONSTANT(BILLBOARD_ENABLED);
	BIND_ENUM_CONSTANT(BILLBOARD_FIXED_Y);
	BIND_ENUM_CONSTANT(BILLBOARD_PARTICLES);

	BIND_ENUM_CONSTANT(TEXTURE_CHANNEL_RED);
	BIND_ENUM_CONSTANT(TEXTURE_CHANNEL_GREEN);
	BIND_ENUM_CONSTANT(TEXTURE_CHANNEL_BLUE);
	BIND_ENUM_CONSTANT(TEXTURE_CHANNEL_ALPHA);
	BIND_ENUM_CONSTANT(TEXTURE_CHANNEL_GRAYSCALE);

	BIND_ENUM_CONSTANT(EMISSION_OP_ADD);
	BIND_ENUM_CONSTANT(EMISSION_OP_MULTIPLY);

	BIND_ENUM_CONSTANT(DISTANCE_FADE_DISABLED);
	BIND_ENUM_CONSTANT(DISTANCE_FADE_PIXEL_ALPHA);
	BIND_ENUM_CONSTANT(DISTANCE_FADE_PIXEL_DITHER);
	BIND_ENUM_CONSTANT(DISTANCE_FADE_OBJECT_DITHER);
}

BaseMaterial3D::BaseMaterial3D(bool p_orm) :
		element(this) {
	orm = p_orm;
	// Initialize to the same values as the shader
	set_albedo(Color(1.0, 1.0, 1.0, 1.0));
	set_specular(0.5);
	set_roughness(1.0);
	set_metallic(0.0);
	set_emission(Color(0, 0, 0));
	set_emission_energy(1.0);
	set_normal_scale(1);
	set_rim(1.0);
	set_rim_tint(0.5);
	set_clearcoat(1);
	set_clearcoat_gloss(0.5);
	set_anisotropy(0);
	set_heightmap_scale(0.05);
	set_subsurface_scattering_strength(0);
	set_backlight(Color(0, 0, 0));
	set_transmittance_color(Color(1, 1, 1, 1));
	set_transmittance_depth(0.1);
	set_transmittance_boost(0.0);
	set_refraction(0.05);
	set_point_size(1);
	set_uv1_offset(Vector3(0, 0, 0));
	set_uv1_scale(Vector3(1, 1, 1));
	set_uv1_triplanar_blend_sharpness(1);
	set_uv2_offset(Vector3(0, 0, 0));
	set_uv2_scale(Vector3(1, 1, 1));
	set_uv2_triplanar_blend_sharpness(1);
	set_billboard_mode(BILLBOARD_DISABLED);
	set_particles_anim_h_frames(1);
	set_particles_anim_v_frames(1);
	set_particles_anim_loop(false);

	set_transparency(TRANSPARENCY_DISABLED);
	set_alpha_antialiasing(ALPHA_ANTIALIASING_OFF);
	set_alpha_scissor_threshold(0.05);
	set_alpha_hash_scale(1.0);
	set_alpha_antialiasing_edge(0.3);

	set_proximity_fade_distance(1);
	set_distance_fade_min_distance(0);
	set_distance_fade_max_distance(10);

	set_ao_light_affect(0.0);

	set_metallic_texture_channel(TEXTURE_CHANNEL_RED);
	set_roughness_texture_channel(TEXTURE_CHANNEL_RED);
	set_ao_texture_channel(TEXTURE_CHANNEL_RED);
	set_refraction_texture_channel(TEXTURE_CHANNEL_RED);

	set_grow(0.0);

	set_heightmap_deep_parallax_min_layers(8);
	set_heightmap_deep_parallax_max_layers(32);
	set_heightmap_deep_parallax_flip_tangent(false); //also sets binormal

	flags[FLAG_USE_TEXTURE_REPEAT] = true;

	is_initialized = true;
	_queue_shader_change();
}

BaseMaterial3D::~BaseMaterial3D() {
	MutexLock lock(material_mutex);

	if (shader_map.has(current_key)) {
		shader_map[current_key].users--;
		if (shader_map[current_key].users == 0) {
			//deallocate shader, as it's no longer in use
			RS::get_singleton()->free(shader_map[current_key].shader);
			shader_map.erase(current_key);
		}

		RS::get_singleton()->material_set_shader(_get_material(), RID());
	}
}

//////////////////////

#ifndef DISABLE_DEPRECATED
// Kept for compatibility from 3.x to 4.0.
bool StandardMaterial3D::_set(const StringName &p_name, const Variant &p_value) {
	if (p_name == "flags_transparent") {
		bool transparent = p_value;
		if (transparent) {
			set_transparency(TRANSPARENCY_ALPHA);
		}
		return true;
	} else if (p_name == "flags_unshaded") {
		bool unshaded = p_value;
		if (unshaded) {
			set_shading_mode(SHADING_MODE_UNSHADED);
		}
		return true;
	} else if (p_name == "flags_vertex_lighting") {
		bool vertex_lit = p_value;
		if (vertex_lit && get_shading_mode() != SHADING_MODE_UNSHADED) {
			set_shading_mode(SHADING_MODE_PER_VERTEX);
		}
		return true;
	} else if (p_name == "params_use_alpha_scissor") {
		bool use_scissor = p_value;
		if (use_scissor) {
			set_transparency(TRANSPARENCY_ALPHA_SCISSOR);
		}
		return true;
	} else if (p_name == "params_use_alpha_hash") {
		bool use_hash = p_value;
		if (use_hash) {
			set_transparency(TRANSPARENCY_ALPHA_HASH);
		}
		return true;
	} else if (p_name == "params_depth_draw_mode") {
		int mode = p_value;
		if (mode == 3) {
			set_transparency(TRANSPARENCY_ALPHA_DEPTH_PRE_PASS);
		}
		return true;
	} else if (p_name == "depth_enabled") {
		bool enabled = p_value;
		if (enabled) {
			set_feature(FEATURE_HEIGHT_MAPPING, true);
			set_flag(FLAG_INVERT_HEIGHTMAP, true);
		}
		return true;
	} else {
		static const Pair<const char *, const char *> remaps[] = {
			{ "flags_use_shadow_to_opacity", "shadow_to_opacity" },
			{ "flags_use_shadow_to_opacity", "shadow_to_opacity" },
			{ "flags_no_depth_test", "no_depth_test" },
			{ "flags_use_point_size", "use_point_size" },
			{ "flags_fixed_size", "fixed_Size" },
			{ "flags_albedo_tex_force_srg", "albedo_tex_force_srgb" },
			{ "flags_do_not_receive_shadows", "disable_receive_shadows" },
			{ "flags_disable_ambient_light", "disable_ambient_light" },
			{ "params_diffuse_mode", "diffuse_mode" },
			{ "params_specular_mode", "specular_mode" },
			{ "params_blend_mode", "blend_mode" },
			{ "params_cull_mode", "cull_mode" },
			{ "params_depth_draw_mode", "params_depth_draw_mode" },
			{ "params_point_size", "point_size" },
			{ "params_billboard_mode", "billboard_mode" },
			{ "params_billboard_keep_scale", "billboard_keep_scale" },
			{ "params_grow", "grow" },
			{ "params_grow_amount", "grow_amount" },
			{ "params_alpha_scissor_threshold", "alpha_scissor_threshold" },
			{ "params_alpha_hash_scale", "alpha_hash_scale" },
			{ "params_alpha_antialiasing_edge", "alpha_antialiasing_edge" },

			{ "depth_scale", "heightmap_scale" },
			{ "depth_deep_parallax", "heightmap_deep_parallax" },
			{ "depth_min_layers", "heightmap_min_layers" },
			{ "depth_max_layers", "heightmap_max_layers" },
			{ "depth_flip_tangent", "heightmap_flip_tangent" },
			{ "depth_flip_binormal", "heightmap_flip_binormal" },
			{ "depth_texture", "heightmap_texture" },

			{ nullptr, nullptr },
		};

		int idx = 0;
		while (remaps[idx].first) {
			if (p_name == remaps[idx].first) {
				set(remaps[idx].second, p_value);
				return true;
			}
			idx++;
		}

		print_line("remapped parameter not found: " + String(p_name));
		return true;
	}

	return false;
}
#endif // DISABLE_DEPRECATED
