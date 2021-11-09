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

#include "core/engine.h"
#include "core/project_settings.h"
#include "core/version.h"

#ifdef TOOLS_ENABLED
#include "editor/editor_settings.h"
#endif

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
	VS::get_singleton()->material_set_next_pass(material, next_pass_rid);
}

Ref<Material> Material::get_next_pass() const {
	return next_pass;
}

void Material::set_render_priority(int p_priority) {
	ERR_FAIL_COND(p_priority < RENDER_PRIORITY_MIN);
	ERR_FAIL_COND(p_priority > RENDER_PRIORITY_MAX);
	render_priority = p_priority;
	VS::get_singleton()->material_set_render_priority(material, p_priority);
}

int Material::get_render_priority() const {
	return render_priority;
}

RID Material::get_rid() const {
	return material;
}
void Material::_validate_property(PropertyInfo &property) const {
	if (!_can_do_next_pass() && property.name == "next_pass") {
		property.usage = 0;
	}
}

void Material::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_next_pass", "next_pass"), &Material::set_next_pass);
	ClassDB::bind_method(D_METHOD("get_next_pass"), &Material::get_next_pass);

	ClassDB::bind_method(D_METHOD("set_render_priority", "priority"), &Material::set_render_priority);
	ClassDB::bind_method(D_METHOD("get_render_priority"), &Material::get_render_priority);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "render_priority", PROPERTY_HINT_RANGE, itos(RENDER_PRIORITY_MIN) + "," + itos(RENDER_PRIORITY_MAX) + ",1"), "set_render_priority", "get_render_priority");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "next_pass", PROPERTY_HINT_RESOURCE_TYPE, "Material"), "set_next_pass", "get_next_pass");

	BIND_CONSTANT(RENDER_PRIORITY_MAX);
	BIND_CONSTANT(RENDER_PRIORITY_MIN);
}

Material::Material() {
	material = RID_PRIME(VisualServer::get_singleton()->material_create());
	render_priority = 0;
}

Material::~Material() {
	VisualServer::get_singleton()->free(material);
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
			VisualServer::get_singleton()->material_set_param(_get_material(), pr, p_value);
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
			r_ret = VisualServer::get_singleton()->material_get_param(_get_material(), pr);
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
			Variant default_value = VisualServer::get_singleton()->material_get_param_default(_get_material(), pr);
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
			r_ret = VisualServer::get_singleton()->material_get_param_default(_get_material(), pr);
		}
	}
	return r_ret;
}

void ShaderMaterial::set_shader(const Ref<Shader> &p_shader) {
	// Only connect/disconnect the signal when running in the editor.
	// This can be a slow operation, and `_change_notify()` (which is called by `_shader_changed()`)
	// does nothing in non-editor builds anyway. See GH-34741 for details.
	if (shader.is_valid() && Engine::get_singleton()->is_editor_hint()) {
		shader->disconnect("changed", this, "_shader_changed");
	}

	shader = p_shader;

	RID rid;
	if (shader.is_valid()) {
		rid = shader->get_rid();

		if (Engine::get_singleton()->is_editor_hint()) {
			shader->connect("changed", this, "_shader_changed");
		}
	}

	VS::get_singleton()->material_set_shader(_get_material(), rid);
	_change_notify(); //properties for shader exposed
	emit_changed();
}

Ref<Shader> ShaderMaterial::get_shader() const {
	return shader;
}

void ShaderMaterial::set_shader_param(const StringName &p_param, const Variant &p_value) {
	VS::get_singleton()->material_set_param(_get_material(), p_param, p_value);
}

Variant ShaderMaterial::get_shader_param(const StringName &p_param) const {
	return VS::get_singleton()->material_get_param(_get_material(), p_param);
}

void ShaderMaterial::_shader_changed() {
	_change_notify(); //update all properties
}

void ShaderMaterial::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_shader", "shader"), &ShaderMaterial::set_shader);
	ClassDB::bind_method(D_METHOD("get_shader"), &ShaderMaterial::get_shader);
	ClassDB::bind_method(D_METHOD("set_shader_param", "param", "value"), &ShaderMaterial::set_shader_param);
	ClassDB::bind_method(D_METHOD("get_shader_param", "param"), &ShaderMaterial::get_shader_param);
	ClassDB::bind_method(D_METHOD("_shader_changed"), &ShaderMaterial::_shader_changed);
	ClassDB::bind_method(D_METHOD("property_can_revert", "name"), &ShaderMaterial::property_can_revert);
	ClassDB::bind_method(D_METHOD("property_get_revert", "name"), &ShaderMaterial::property_get_revert);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "shader", PROPERTY_HINT_RESOURCE_TYPE, "Shader"), "set_shader", "get_shader");
}

void ShaderMaterial::get_argument_options(const StringName &p_function, int p_idx, List<String> *r_options) const {
#ifdef TOOLS_ENABLED
	const String quote_style = EDITOR_DEF("text_editor/completion/use_single_quotes", 0) ? "'" : "\"";
#else
	const String quote_style = "\"";
#endif

	String f = p_function.operator String();
	if ((f == "get_shader_param" || f == "set_shader_param") && p_idx == 0) {
		if (shader.is_valid()) {
			List<PropertyInfo> pl;
			shader->get_param_list(&pl);
			for (List<PropertyInfo>::Element *E = pl.front(); E; E = E->next()) {
				r_options->push_back(quote_style + E->get().name.replace_first("shader_param/", "") + quote_style);
			}
		}
	}
	Resource::get_argument_options(p_function, p_idx, r_options);
}

bool ShaderMaterial::_can_do_next_pass() const {
	return shader.is_valid() && shader->get_mode() == Shader::MODE_SPATIAL;
}

Shader::Mode ShaderMaterial::get_shader_mode() const {
	if (shader.is_valid()) {
		return shader->get_mode();
	} else {
		return Shader::MODE_SPATIAL;
	}
}

ShaderMaterial::ShaderMaterial() {
}

ShaderMaterial::~ShaderMaterial() {
}

/////////////////////////////////

Mutex SpatialMaterial::material_mutex;
SelfList<SpatialMaterial>::List *SpatialMaterial::dirty_materials = nullptr;
Map<SpatialMaterial::MaterialKey, SpatialMaterial::ShaderData> SpatialMaterial::shader_map;
SpatialMaterial::ShaderNames *SpatialMaterial::shader_names = nullptr;

void SpatialMaterial::init_shaders() {
	dirty_materials = memnew(SelfList<SpatialMaterial>::List);

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
	shader_names->depth_scale = "depth_scale";
	shader_names->subsurface_scattering_strength = "subsurface_scattering_strength";
	shader_names->transmission = "transmission";
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
	shader_names->depth_min_layers = "depth_min_layers";
	shader_names->depth_max_layers = "depth_max_layers";
	shader_names->depth_flip = "depth_flip";

	shader_names->grow = "grow";

	shader_names->ao_light_affect = "ao_light_affect";

	shader_names->proximity_fade_distance = "proximity_fade_distance";
	shader_names->distance_fade_min = "distance_fade_min";
	shader_names->distance_fade_max = "distance_fade_max";

	shader_names->metallic_texture_channel = "metallic_texture_channel";
	shader_names->roughness_texture_channel = "roughness_texture_channel";
	shader_names->ao_texture_channel = "ao_texture_channel";
	shader_names->clearcoat_texture_channel = "clearcoat_texture_channel";
	shader_names->rim_texture_channel = "rim_texture_channel";
	shader_names->depth_texture_channel = "depth_texture_channel";
	shader_names->refraction_texture_channel = "refraction_texture_channel";
	shader_names->alpha_scissor_threshold = "alpha_scissor_threshold";

	shader_names->texture_names[TEXTURE_ALBEDO] = "texture_albedo";
	shader_names->texture_names[TEXTURE_METALLIC] = "texture_metallic";
	shader_names->texture_names[TEXTURE_ROUGHNESS] = "texture_roughness";
	shader_names->texture_names[TEXTURE_EMISSION] = "texture_emission";
	shader_names->texture_names[TEXTURE_NORMAL] = "texture_normal";
	shader_names->texture_names[TEXTURE_RIM] = "texture_rim";
	shader_names->texture_names[TEXTURE_CLEARCOAT] = "texture_clearcoat";
	shader_names->texture_names[TEXTURE_FLOWMAP] = "texture_flowmap";
	shader_names->texture_names[TEXTURE_AMBIENT_OCCLUSION] = "texture_ambient_occlusion";
	shader_names->texture_names[TEXTURE_DEPTH] = "texture_depth";
	shader_names->texture_names[TEXTURE_SUBSURFACE_SCATTERING] = "texture_subsurface_scattering";
	shader_names->texture_names[TEXTURE_TRANSMISSION] = "texture_transmission";
	shader_names->texture_names[TEXTURE_REFRACTION] = "texture_refraction";
	shader_names->texture_names[TEXTURE_DETAIL_MASK] = "texture_detail_mask";
	shader_names->texture_names[TEXTURE_DETAIL_ALBEDO] = "texture_detail_albedo";
	shader_names->texture_names[TEXTURE_DETAIL_NORMAL] = "texture_detail_normal";
}

Ref<SpatialMaterial> SpatialMaterial::materials_for_2d[SpatialMaterial::MAX_MATERIALS_FOR_2D];

void SpatialMaterial::finish_shaders() {
	for (int i = 0; i < MAX_MATERIALS_FOR_2D; i++) {
		materials_for_2d[i].unref();
	}

	memdelete(dirty_materials);
	dirty_materials = nullptr;

	memdelete(shader_names);
}

void SpatialMaterial::_update_shader() {
	dirty_materials->remove(&element);

	MaterialKey mk = _compute_key();
	if (mk.key == current_key.key) {
		return; //no update required in the end
	}

	if (shader_map.has(current_key)) {
		shader_map[current_key].users--;
		if (shader_map[current_key].users == 0) {
			//deallocate shader, as it's no longer in use
			VS::get_singleton()->free(shader_map[current_key].shader);
			shader_map.erase(current_key);
		}
	}

	current_key = mk;

	if (shader_map.has(mk)) {
		VS::get_singleton()->material_set_shader(_get_material(), shader_map[mk].shader);
		shader_map[mk].users++;
		return;
	}

	//must create a shader!

	// Add a comment to describe the shader origin (useful when converting to ShaderMaterial).
	String code = "// NOTE: Shader automatically converted from " VERSION_NAME " " VERSION_FULL_CONFIG "'s SpatialMaterial.\n\n";

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
		case DEPTH_DRAW_ALPHA_OPAQUE_PREPASS:
			code += ",depth_draw_alpha_prepass";
			break;
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
		case DIFFUSE_OREN_NAYAR:
			code += ",diffuse_oren_nayar";
			break;
		case DIFFUSE_TOON:
			code += ",diffuse_toon";
			break;
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
	}

	if (flags[FLAG_UNSHADED]) {
		code += ",unshaded";
	}
	if (flags[FLAG_DISABLE_DEPTH_TEST]) {
		code += ",depth_test_disable";
	}
	if (flags[FLAG_USE_VERTEX_LIGHTING] || force_vertex_shading) {
		code += ",vertex_lighting";
	}
	if (flags[FLAG_TRIPLANAR_USE_WORLD] && (flags[FLAG_UV1_USE_TRIPLANAR] || flags[FLAG_UV2_USE_TRIPLANAR])) {
		code += ",world_vertex_coords";
	}
	if (flags[FLAG_DONT_RECEIVE_SHADOWS]) {
		code += ",shadows_disabled";
	}
	if (flags[FLAG_DISABLE_AMBIENT_LIGHT]) {
		code += ",ambient_light_disabled";
	}
	if (flags[FLAG_ENSURE_CORRECT_NORMALS]) {
		code += ",ensure_correct_normals";
	}
	if (flags[FLAG_USE_SHADOW_TO_OPACITY]) {
		code += ",shadow_to_opacity";
	}
	code += ";\n";

	code += "uniform vec4 albedo : hint_color;\n";
	code += "uniform sampler2D texture_albedo : hint_albedo;\n";
	code += "uniform float specular;\n";
	code += "uniform float metallic;\n";
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

	if (flags[FLAG_USE_ALPHA_SCISSOR]) {
		code += "uniform float alpha_scissor_threshold;\n";
	}
	code += "uniform float roughness : hint_range(0,1);\n";
	code += "uniform float point_size : hint_range(0,128);\n";

	if (textures[TEXTURE_METALLIC] != nullptr) {
		code += "uniform sampler2D texture_metallic : hint_white;\n";
		code += "uniform vec4 metallic_texture_channel;\n";
	}

	if (textures[TEXTURE_ROUGHNESS] != nullptr) {
		code += "uniform sampler2D texture_roughness : hint_white;\n";
		code += "uniform vec4 roughness_texture_channel;\n";
	}
	if (billboard_mode == BILLBOARD_PARTICLES) {
		code += "uniform int particles_anim_h_frames;\n";
		code += "uniform int particles_anim_v_frames;\n";
		code += "uniform bool particles_anim_loop;\n";
	}

	if (features[FEATURE_EMISSION]) {
		code += "uniform sampler2D texture_emission : hint_black_albedo;\n";
		code += "uniform vec4 emission : hint_color;\n";
		code += "uniform float emission_energy;\n";
	}

	if (features[FEATURE_REFRACTION]) {
		code += "uniform sampler2D texture_refraction;\n";
		code += "uniform float refraction : hint_range(-16,16);\n";
		code += "uniform vec4 refraction_texture_channel;\n";
	}

	if (features[FEATURE_NORMAL_MAPPING]) {
		code += "uniform sampler2D texture_normal : hint_normal;\n";
		code += "uniform float normal_scale : hint_range(-16,16);\n";
	}
	if (features[FEATURE_RIM]) {
		code += "uniform float rim : hint_range(0,1);\n";
		code += "uniform float rim_tint : hint_range(0,1);\n";
		code += "uniform sampler2D texture_rim : hint_white;\n";
	}
	if (features[FEATURE_CLEARCOAT]) {
		code += "uniform float clearcoat : hint_range(0,1);\n";
		code += "uniform float clearcoat_gloss : hint_range(0,1);\n";
		code += "uniform sampler2D texture_clearcoat : hint_white;\n";
	}
	if (features[FEATURE_ANISOTROPY]) {
		code += "uniform float anisotropy_ratio : hint_range(0,256);\n";
		code += "uniform sampler2D texture_flowmap : hint_aniso;\n";
	}
	if (features[FEATURE_AMBIENT_OCCLUSION]) {
		code += "uniform sampler2D texture_ambient_occlusion : hint_white;\n";
		code += "uniform vec4 ao_texture_channel;\n";
		code += "uniform float ao_light_affect;\n";
	}

	if (features[FEATURE_DETAIL]) {
		code += "uniform sampler2D texture_detail_albedo : hint_albedo;\n";
		code += "uniform sampler2D texture_detail_normal : hint_normal;\n";
		code += "uniform sampler2D texture_detail_mask : hint_white;\n";
	}

	if (features[FEATURE_SUBSURACE_SCATTERING]) {
		code += "uniform float subsurface_scattering_strength : hint_range(0,1);\n";
		code += "uniform sampler2D texture_subsurface_scattering : hint_white;\n";
	}

	if (features[FEATURE_TRANSMISSION]) {
		code += "uniform vec4 transmission : hint_color;\n";
		code += "uniform sampler2D texture_transmission : hint_black;\n";
	}

	if (features[FEATURE_DEPTH_MAPPING]) {
		code += "uniform sampler2D texture_depth : hint_black;\n";
		code += "uniform float depth_scale;\n";
		code += "uniform int depth_min_layers;\n";
		code += "uniform int depth_max_layers;\n";
		code += "uniform vec2 depth_flip;\n";
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
		code += "\tif (!OUTPUT_IS_SRGB) {\n";
		code += "\t\tCOLOR.rgb = mix(pow((COLOR.rgb + vec3(0.055)) * (1.0 / (1.0 + 0.055)), vec3(2.4)), COLOR.rgb * (1.0 / 12.92), lessThan(COLOR.rgb, vec3(0.04045)));\n";
		code += "\t}\n";
	}
	if (flags[FLAG_USE_POINT_SIZE]) {
		code += "\tPOINT_SIZE=point_size;\n";
	}

	if (flags[FLAG_USE_VERTEX_LIGHTING] || force_vertex_shading) {
		code += "\tROUGHNESS=roughness;\n";
	}

	if (!flags[FLAG_UV1_USE_TRIPLANAR]) {
		code += "\tUV=UV*uv1_scale.xy+uv1_offset.xy;\n";
	}

	switch (billboard_mode) {
		case BILLBOARD_DISABLED: {
		} break;
		case BILLBOARD_ENABLED: {
			code += "\tMODELVIEW_MATRIX = INV_CAMERA_MATRIX * mat4(CAMERA_MATRIX[0],CAMERA_MATRIX[1],CAMERA_MATRIX[2],WORLD_MATRIX[3]);\n";

			if (flags[FLAG_BILLBOARD_KEEP_SCALE]) {
				code += "\tMODELVIEW_MATRIX = MODELVIEW_MATRIX * mat4(vec4(length(WORLD_MATRIX[0].xyz), 0.0, 0.0, 0.0),vec4(0.0, length(WORLD_MATRIX[1].xyz), 0.0, 0.0),vec4(0.0, 0.0, length(WORLD_MATRIX[2].xyz), 0.0),vec4(0.0, 0.0, 0.0, 1.0));\n";
			}
		} break;
		case BILLBOARD_FIXED_Y: {
			code += "\tMODELVIEW_MATRIX = INV_CAMERA_MATRIX * mat4(vec4(normalize(cross(vec3(0.0, 1.0, 0.0), CAMERA_MATRIX[2].xyz)),0.0),vec4(0.0, 1.0, 0.0, 0.0),vec4(normalize(cross(CAMERA_MATRIX[0].xyz, vec3(0.0, 1.0, 0.0))),0.0),WORLD_MATRIX[3]);\n";

			if (flags[FLAG_BILLBOARD_KEEP_SCALE]) {
				code += "\tMODELVIEW_MATRIX = MODELVIEW_MATRIX * mat4(vec4(length(WORLD_MATRIX[0].xyz), 0.0, 0.0, 0.0),vec4(0.0, length(WORLD_MATRIX[1].xyz), 0.0, 0.0),vec4(0.0, 0.0, length(WORLD_MATRIX[2].xyz), 0.0),vec4(0.0, 0.0, 0.0, 1.0));\n";
			}
		} break;
		case BILLBOARD_PARTICLES: {
			//make billboard
			code += "\tmat4 mat_world = mat4(normalize(CAMERA_MATRIX[0])*length(WORLD_MATRIX[0]),normalize(CAMERA_MATRIX[1])*length(WORLD_MATRIX[0]),normalize(CAMERA_MATRIX[2])*length(WORLD_MATRIX[2]),WORLD_MATRIX[3]);\n";
			//rotate by rotation
			code += "\tmat_world = mat_world * mat4( vec4(cos(INSTANCE_CUSTOM.x),-sin(INSTANCE_CUSTOM.x), 0.0, 0.0), vec4(sin(INSTANCE_CUSTOM.x), cos(INSTANCE_CUSTOM.x), 0.0, 0.0),vec4(0.0, 0.0, 1.0, 0.0),vec4(0.0, 0.0, 0.0, 1.0));\n";
			//set modelview
			code += "\tMODELVIEW_MATRIX = INV_CAMERA_MATRIX * mat_world;\n";

			//handle animation
			code += "\tfloat h_frames = float(particles_anim_h_frames);\n";
			code += "\tfloat v_frames = float(particles_anim_v_frames);\n";
			code += "\tfloat particle_total_frames = float(particles_anim_h_frames * particles_anim_v_frames);\n";
			code += "\tfloat particle_frame = floor(INSTANCE_CUSTOM.z * float(particle_total_frames));\n";
			code += "\tif (!particles_anim_loop) {\n";
			code += "\t\tparticle_frame = clamp(particle_frame, 0.0, particle_total_frames - 1.0);\n";
			code += "\t} else {\n";
			code += "\t\tparticle_frame = mod(particle_frame, particle_total_frames);\n";
			code += "\t}";
			code += "\tUV /= vec2(h_frames, v_frames);\n";
			code += "\tUV += vec2(mod(particle_frame, h_frames) / h_frames, floor((particle_frame + 0.5) / h_frames) / v_frames);\n";
		} break;
	}

	if (flags[FLAG_FIXED_SIZE]) {
		code += "\tif (PROJECTION_MATRIX[3][3] != 0.0) {\n";
		//orthogonal matrix, try to do about the same
		//with viewport size
		code += "\t\tfloat h = abs(1.0 / (2.0 * PROJECTION_MATRIX[1][1]));\n";
		code += "\t\tfloat sc = (h * 2.0); //consistent with Y-fov\n";
		code += "\t\tMODELVIEW_MATRIX[0]*=sc;\n";
		code += "\t\tMODELVIEW_MATRIX[1]*=sc;\n";
		code += "\t\tMODELVIEW_MATRIX[2]*=sc;\n";
		code += "\t} else {\n";
		//just scale by depth
		code += "\t\tfloat sc = -(MODELVIEW_MATRIX)[3].z;\n";
		code += "\t\tMODELVIEW_MATRIX[0]*=sc;\n";
		code += "\t\tMODELVIEW_MATRIX[1]*=sc;\n";
		code += "\t\tMODELVIEW_MATRIX[2]*=sc;\n";
		code += "\t}\n";
	}

	if (detail_uv == DETAIL_UV_2 && !flags[FLAG_UV2_USE_TRIPLANAR]) {
		code += "\tUV2=UV2*uv2_scale.xy+uv2_offset.xy;\n";
	}
	if (flags[FLAG_UV1_USE_TRIPLANAR] || flags[FLAG_UV2_USE_TRIPLANAR]) {
		//generate tangent and binormal in world space
		code += "\tTANGENT = vec3(0.0,0.0,-1.0) * abs(NORMAL.x);\n";
		code += "\tTANGENT+= vec3(1.0,0.0,0.0) * abs(NORMAL.y);\n";
		code += "\tTANGENT+= vec3(1.0,0.0,0.0) * abs(NORMAL.z);\n";
		code += "\tTANGENT = normalize(TANGENT);\n";

		code += "\tBINORMAL = vec3(0.0,1.0,0.0) * abs(NORMAL.x);\n";
		code += "\tBINORMAL+= vec3(0.0,0.0,-1.0) * abs(NORMAL.y);\n";
		code += "\tBINORMAL+= vec3(0.0,1.0,0.0) * abs(NORMAL.z);\n";
		code += "\tBINORMAL = normalize(BINORMAL);\n";
	}

	if (flags[FLAG_UV1_USE_TRIPLANAR]) {
		code += "\tuv1_power_normal=pow(abs(NORMAL),vec3(uv1_blend_sharpness));\n";
		code += "\tuv1_power_normal/=dot(uv1_power_normal,vec3(1.0));\n";
		code += "\tuv1_triplanar_pos = VERTEX * uv1_scale + uv1_offset;\n";
		code += "\tuv1_triplanar_pos *= vec3(1.0,-1.0, 1.0);\n";
	}

	if (flags[FLAG_UV2_USE_TRIPLANAR]) {
		code += "\tuv2_power_normal=pow(abs(NORMAL), vec3(uv2_blend_sharpness));\n";
		code += "\tuv2_power_normal/=dot(uv2_power_normal,vec3(1.0));\n";
		code += "\tuv2_triplanar_pos = VERTEX * uv2_scale + uv2_offset;\n";
		code += "\tuv2_triplanar_pos *= vec3(1.0,-1.0, 1.0);\n";
	}

	if (grow_enabled) {
		code += "\tVERTEX+=NORMAL*grow;\n";
	}

	code += "}\n";
	code += "\n\n";
	if (flags[FLAG_UV1_USE_TRIPLANAR] || flags[FLAG_UV2_USE_TRIPLANAR]) {
		code += "vec4 triplanar_texture(sampler2D p_sampler,vec3 p_weights,vec3 p_triplanar_pos) {\n";
		code += "\tvec4 samp=vec4(0.0);\n";
		code += "\tsamp+= texture(p_sampler,p_triplanar_pos.xy) * p_weights.z;\n";
		code += "\tsamp+= texture(p_sampler,p_triplanar_pos.xz) * p_weights.y;\n";
		code += "\tsamp+= texture(p_sampler,p_triplanar_pos.zy * vec2(-1.0,1.0)) * p_weights.x;\n";
		code += "\treturn samp;\n";
		code += "}\n";
	}
	code += "\n\n";
	code += "void fragment() {\n";

	if (!flags[FLAG_UV1_USE_TRIPLANAR]) {
		code += "\tvec2 base_uv = UV;\n";
	}

	if ((features[FEATURE_DETAIL] && detail_uv == DETAIL_UV_2) || (features[FEATURE_AMBIENT_OCCLUSION] && flags[FLAG_AO_ON_UV2]) || (features[FEATURE_EMISSION] && flags[FLAG_EMISSION_ON_UV2])) {
		code += "\tvec2 base_uv2 = UV2;\n";
	}

	if (!VisualServer::get_singleton()->is_low_end() && features[FEATURE_DEPTH_MAPPING] && !flags[FLAG_UV1_USE_TRIPLANAR]) { //depthmap not supported with triplanar
		code += "\t{\n";
		code += "\t\tvec3 view_dir = normalize(normalize(-VERTEX)*mat3(TANGENT*depth_flip.x,-BINORMAL*depth_flip.y,NORMAL));\n"; // binormal is negative due to mikktspace, flip 'unflips' it ;-)

		if (deep_parallax) {
			code += "\t\tfloat num_layers = mix(float(depth_max_layers),float(depth_min_layers), abs(dot(vec3(0.0, 0.0, 1.0), view_dir)));\n";
			code += "\t\tfloat layer_depth = 1.0 / num_layers;\n";
			code += "\t\tfloat current_layer_depth = 0.0;\n";
			code += "\t\tvec2 P = view_dir.xy * depth_scale;\n";
			code += "\t\tvec2 delta = P / num_layers;\n";
			code += "\t\tvec2 ofs = base_uv;\n";
			code += "\t\tfloat depth = textureLod(texture_depth, ofs, 0.0).r;\n";
			code += "\t\tfloat current_depth = 0.0;\n";
			code += "\t\twhile(current_depth < depth) {\n";
			code += "\t\t\tofs -= delta;\n";
			code += "\t\t\tdepth = textureLod(texture_depth, ofs, 0.0).r;\n";
			code += "\t\t\tcurrent_depth += layer_depth;\n";
			code += "\t\t}\n";
			code += "\t\tvec2 prev_ofs = ofs + delta;\n";
			code += "\t\tfloat after_depth  = depth - current_depth;\n";
			code += "\t\tfloat before_depth = textureLod(texture_depth, prev_ofs, 0.0).r - current_depth + layer_depth;\n";
			code += "\t\tfloat weight = after_depth / (after_depth - before_depth);\n";
			code += "\t\tofs = mix(ofs,prev_ofs,weight);\n";

		} else {
			code += "\t\tfloat depth = texture(texture_depth, base_uv).r;\n";
			// Use offset limiting to improve the appearance of non-deep parallax.
			// This reduces the impression of depth, but avoids visible warping in the distance.
			code += "\t\tvec2 ofs = base_uv - view_dir.xy * depth * depth_scale;\n";
		}

		code += "\t\tbase_uv=ofs;\n";
		if (features[FEATURE_DETAIL] && detail_uv == DETAIL_UV_2) {
			code += "\t\tbase_uv2-=ofs;\n";
		}

		code += "\t}\n";
	}

	if (flags[FLAG_USE_POINT_SIZE]) {
		code += "\tvec4 albedo_tex = texture(texture_albedo,POINT_COORD);\n";
	} else {
		if (flags[FLAG_UV1_USE_TRIPLANAR]) {
			code += "\tvec4 albedo_tex = triplanar_texture(texture_albedo,uv1_power_normal,uv1_triplanar_pos);\n";
		} else {
			code += "\tvec4 albedo_tex = texture(texture_albedo,base_uv);\n";
		}
	}

	if (flags[FLAG_ALBEDO_TEXTURE_FORCE_SRGB]) {
		code += "\talbedo_tex.rgb = mix(pow((albedo_tex.rgb + vec3(0.055)) * (1.0 / (1.0 + 0.055)),vec3(2.4)),albedo_tex.rgb.rgb * (1.0 / 12.92),lessThan(albedo_tex.rgb,vec3(0.04045)));\n";
	}

	if (flags[FLAG_ALBEDO_FROM_VERTEX_COLOR]) {
		code += "\talbedo_tex *= COLOR;\n";
	}
	code += "\tALBEDO = albedo.rgb * albedo_tex.rgb;\n";

	if (textures[TEXTURE_METALLIC] != nullptr) {
		if (flags[FLAG_UV1_USE_TRIPLANAR]) {
			code += "\tfloat metallic_tex = dot(triplanar_texture(texture_metallic,uv1_power_normal,uv1_triplanar_pos),metallic_texture_channel);\n";
		} else {
			code += "\tfloat metallic_tex = dot(texture(texture_metallic,base_uv),metallic_texture_channel);\n";
		}
		code += "\tMETALLIC = metallic_tex * metallic;\n";
	} else {
		code += "\tMETALLIC = metallic;\n";
	}

	if (textures[TEXTURE_ROUGHNESS] != nullptr) {
		if (flags[FLAG_UV1_USE_TRIPLANAR]) {
			code += "\tfloat roughness_tex = dot(triplanar_texture(texture_roughness,uv1_power_normal,uv1_triplanar_pos),roughness_texture_channel);\n";
		} else {
			code += "\tfloat roughness_tex = dot(texture(texture_roughness,base_uv),roughness_texture_channel);\n";
		}
		code += "\tROUGHNESS = roughness_tex * roughness;\n";
	} else {
		code += "\tROUGHNESS = roughness;\n";
	}

	code += "\tSPECULAR = specular;\n";

	if (features[FEATURE_NORMAL_MAPPING]) {
		if (flags[FLAG_UV1_USE_TRIPLANAR]) {
			code += "\tNORMALMAP = triplanar_texture(texture_normal,uv1_power_normal,uv1_triplanar_pos).rgb;\n";
		} else {
			code += "\tNORMALMAP = texture(texture_normal,base_uv).rgb;\n";
		}
		code += "\tNORMALMAP_DEPTH = normal_scale;\n";
	}

	if (features[FEATURE_EMISSION]) {
		if (flags[FLAG_EMISSION_ON_UV2]) {
			if (flags[FLAG_UV2_USE_TRIPLANAR]) {
				code += "\tvec3 emission_tex = triplanar_texture(texture_emission,uv2_power_normal,uv2_triplanar_pos).rgb;\n";
			} else {
				code += "\tvec3 emission_tex = texture(texture_emission,base_uv2).rgb;\n";
			}
		} else {
			if (flags[FLAG_UV1_USE_TRIPLANAR]) {
				code += "\tvec3 emission_tex = triplanar_texture(texture_emission,uv1_power_normal,uv1_triplanar_pos).rgb;\n";
			} else {
				code += "\tvec3 emission_tex = texture(texture_emission,base_uv).rgb;\n";
			}
		}

		if (emission_op == EMISSION_OP_ADD) {
			code += "\tEMISSION = (emission.rgb+emission_tex)*emission_energy;\n";
		} else {
			code += "\tEMISSION = (emission.rgb*emission_tex)*emission_energy;\n";
		}
	}

	if (features[FEATURE_REFRACTION]) {
		if (features[FEATURE_NORMAL_MAPPING]) {
			code += "\tvec3 unpacked_normal = NORMALMAP;\n";
			code += "\tunpacked_normal.xy = unpacked_normal.xy * 2.0 - 1.0;\n";
			code += "\tunpacked_normal.z = sqrt(max(0.0, 1.0 - dot(unpacked_normal.xy, unpacked_normal.xy)));\n";
			code += "\tvec3 ref_normal = normalize( mix(NORMAL,TANGENT * unpacked_normal.x + BINORMAL * unpacked_normal.y + NORMAL * unpacked_normal.z,NORMALMAP_DEPTH) );\n";
		} else {
			code += "\tvec3 ref_normal = NORMAL;\n";
		}
		if (flags[FLAG_UV1_USE_TRIPLANAR]) {
			code += "\tvec2 ref_ofs = SCREEN_UV - ref_normal.xy * dot(triplanar_texture(texture_refraction,uv1_power_normal,uv1_triplanar_pos),refraction_texture_channel) * refraction;\n";
		} else {
			code += "\tvec2 ref_ofs = SCREEN_UV - ref_normal.xy * dot(texture(texture_refraction,base_uv),refraction_texture_channel) * refraction;\n";
		}
		code += "\tfloat ref_amount = 1.0 - albedo.a * albedo_tex.a;\n";
		code += "\tEMISSION += textureLod(SCREEN_TEXTURE,ref_ofs,ROUGHNESS * 8.0).rgb * ref_amount;\n";
		code += "\tALBEDO *= 1.0 - ref_amount;\n";
		code += "\tALPHA = 1.0;\n";

	} else if (features[FEATURE_TRANSPARENT] || flags[FLAG_USE_ALPHA_SCISSOR] || flags[FLAG_USE_SHADOW_TO_OPACITY] || (distance_fade == DISTANCE_FADE_PIXEL_ALPHA) || proximity_fade_enabled) {
		code += "\tALPHA = albedo.a * albedo_tex.a;\n";
	}

	if (proximity_fade_enabled) {
		code += "\tfloat depth_tex = textureLod(DEPTH_TEXTURE,SCREEN_UV,0.0).r;\n";
		code += "\tvec4 world_pos = INV_PROJECTION_MATRIX * vec4(SCREEN_UV*2.0-1.0,depth_tex*2.0-1.0,1.0);\n";
		code += "\tworld_pos.xyz/=world_pos.w;\n";
		code += "\tALPHA*=clamp(1.0-smoothstep(world_pos.z+proximity_fade_distance,world_pos.z,VERTEX.z),0.0,1.0);\n";
	}

	if (distance_fade != DISTANCE_FADE_DISABLED) {
		if ((distance_fade == DISTANCE_FADE_OBJECT_DITHER || distance_fade == DISTANCE_FADE_PIXEL_DITHER)) {
			if (!VisualServer::get_singleton()->is_low_end()) {
				code += "\t{\n";
				if (distance_fade == DISTANCE_FADE_OBJECT_DITHER) {
					code += "\t\tfloat fade_distance = abs((INV_CAMERA_MATRIX * WORLD_MATRIX[3]).z);\n";

				} else {
					code += "\t\tfloat fade_distance=-VERTEX.z;\n";
				}

				code += "\t\tfloat fade=clamp(smoothstep(distance_fade_min,distance_fade_max,fade_distance),0.0,1.0);\n";
				code += "\t\tint x = int(FRAGCOORD.x) % 4;\n";
				code += "\t\tint y = int(FRAGCOORD.y) % 4;\n";
				code += "\t\tint index = x + y * 4;\n";
				code += "\t\tfloat limit = 0.0;\n\n";
				code += "\t\tif (x < 8) {\n";
				code += "\t\t\tif (index == 0) limit = 0.0625;\n";
				code += "\t\t\tif (index == 1) limit = 0.5625;\n";
				code += "\t\t\tif (index == 2) limit = 0.1875;\n";
				code += "\t\t\tif (index == 3) limit = 0.6875;\n";
				code += "\t\t\tif (index == 4) limit = 0.8125;\n";
				code += "\t\t\tif (index == 5) limit = 0.3125;\n";
				code += "\t\t\tif (index == 6) limit = 0.9375;\n";
				code += "\t\t\tif (index == 7) limit = 0.4375;\n";
				code += "\t\t\tif (index == 8) limit = 0.25;\n";
				code += "\t\t\tif (index == 9) limit = 0.75;\n";
				code += "\t\t\tif (index == 10) limit = 0.125;\n";
				code += "\t\t\tif (index == 11) limit = 0.625;\n";
				code += "\t\t\tif (index == 12) limit = 1.0;\n";
				code += "\t\t\tif (index == 13) limit = 0.5;\n";
				code += "\t\t\tif (index == 14) limit = 0.875;\n";
				code += "\t\t\tif (index == 15) limit = 0.375;\n";
				code += "\t\t}\n\n";
				code += "\tif (fade < limit)\n";
				code += "\t\tdiscard;\n";
				code += "\t}\n\n";
			}

		} else {
			code += "\tALPHA*=clamp(smoothstep(distance_fade_min,distance_fade_max,-VERTEX.z),0.0,1.0);\n";
		}
	}

	if (features[FEATURE_RIM]) {
		if (flags[FLAG_UV1_USE_TRIPLANAR]) {
			code += "\tvec2 rim_tex = triplanar_texture(texture_rim,uv1_power_normal,uv1_triplanar_pos).xy;\n";
		} else {
			code += "\tvec2 rim_tex = texture(texture_rim,base_uv).xy;\n";
		}
		code += "\tRIM = rim*rim_tex.x;";
		code += "\tRIM_TINT = rim_tint*rim_tex.y;\n";
	}

	if (features[FEATURE_CLEARCOAT]) {
		if (flags[FLAG_UV1_USE_TRIPLANAR]) {
			code += "\tvec2 clearcoat_tex = triplanar_texture(texture_clearcoat,uv1_power_normal,uv1_triplanar_pos).xy;\n";
		} else {
			code += "\tvec2 clearcoat_tex = texture(texture_clearcoat,base_uv).xy;\n";
		}
		code += "\tCLEARCOAT = clearcoat*clearcoat_tex.x;";
		code += "\tCLEARCOAT_GLOSS = clearcoat_gloss*clearcoat_tex.y;\n";
	}

	if (features[FEATURE_ANISOTROPY]) {
		if (flags[FLAG_UV1_USE_TRIPLANAR]) {
			code += "\tvec3 anisotropy_tex = triplanar_texture(texture_flowmap,uv1_power_normal,uv1_triplanar_pos).rga;\n";
		} else {
			code += "\tvec3 anisotropy_tex = texture(texture_flowmap,base_uv).rga;\n";
		}
		code += "\tANISOTROPY = anisotropy_ratio*anisotropy_tex.b;\n";
		code += "\tANISOTROPY_FLOW = anisotropy_tex.rg*2.0-1.0;\n";
	}

	if (features[FEATURE_AMBIENT_OCCLUSION]) {
		if (flags[FLAG_AO_ON_UV2]) {
			if (flags[FLAG_UV2_USE_TRIPLANAR]) {
				code += "\tAO = dot(triplanar_texture(texture_ambient_occlusion,uv2_power_normal,uv2_triplanar_pos),ao_texture_channel);\n";
			} else {
				code += "\tAO = dot(texture(texture_ambient_occlusion,base_uv2),ao_texture_channel);\n";
			}
		} else {
			if (flags[FLAG_UV1_USE_TRIPLANAR]) {
				code += "\tAO = dot(triplanar_texture(texture_ambient_occlusion,uv1_power_normal,uv1_triplanar_pos),ao_texture_channel);\n";
			} else {
				code += "\tAO = dot(texture(texture_ambient_occlusion,base_uv),ao_texture_channel);\n";
			}
		}

		code += "\tAO_LIGHT_AFFECT = ao_light_affect;\n";
	}

	if (features[FEATURE_SUBSURACE_SCATTERING]) {
		if (flags[FLAG_UV1_USE_TRIPLANAR]) {
			code += "\tfloat sss_tex = triplanar_texture(texture_subsurface_scattering,uv1_power_normal,uv1_triplanar_pos).r;\n";
		} else {
			code += "\tfloat sss_tex = texture(texture_subsurface_scattering,base_uv).r;\n";
		}
		code += "\tSSS_STRENGTH=subsurface_scattering_strength*sss_tex;\n";
	}

	if (features[FEATURE_TRANSMISSION]) {
		if (flags[FLAG_UV1_USE_TRIPLANAR]) {
			code += "\tvec3 transmission_tex = triplanar_texture(texture_transmission,uv1_power_normal,uv1_triplanar_pos).rgb;\n";
		} else {
			code += "\tvec3 transmission_tex = texture(texture_transmission,base_uv).rgb;\n";
		}
		code += "\tTRANSMISSION = (transmission.rgb+transmission_tex);\n";
	}

	if (features[FEATURE_DETAIL]) {
		bool triplanar = (flags[FLAG_UV1_USE_TRIPLANAR] && detail_uv == DETAIL_UV_1) || (flags[FLAG_UV2_USE_TRIPLANAR] && detail_uv == DETAIL_UV_2);

		if (triplanar) {
			String tp_uv = detail_uv == DETAIL_UV_1 ? "uv1" : "uv2";
			code += "\tvec4 detail_tex = triplanar_texture(texture_detail_albedo," + tp_uv + "_power_normal," + tp_uv + "_triplanar_pos);\n";
			code += "\tvec4 detail_norm_tex = triplanar_texture(texture_detail_normal," + tp_uv + "_power_normal," + tp_uv + "_triplanar_pos);\n";

		} else {
			String det_uv = detail_uv == DETAIL_UV_1 ? "base_uv" : "base_uv2";
			code += "\tvec4 detail_tex = texture(texture_detail_albedo," + det_uv + ");\n";
			code += "\tvec4 detail_norm_tex = texture(texture_detail_normal," + det_uv + ");\n";
		}

		if (flags[FLAG_UV1_USE_TRIPLANAR]) {
			code += "\tvec4 detail_mask_tex = triplanar_texture(texture_detail_mask,uv1_power_normal,uv1_triplanar_pos);\n";
		} else {
			code += "\tvec4 detail_mask_tex = texture(texture_detail_mask,base_uv);\n";
		}

		switch (detail_blend_mode) {
			case BLEND_MODE_MIX: {
				code += "\tvec3 detail = mix(ALBEDO.rgb,detail_tex.rgb,detail_tex.a);\n";
			} break;
			case BLEND_MODE_ADD: {
				code += "\tvec3 detail = mix(ALBEDO.rgb,ALBEDO.rgb+detail_tex.rgb,detail_tex.a);\n";
			} break;
			case BLEND_MODE_SUB: {
				code += "\tvec3 detail = mix(ALBEDO.rgb,ALBEDO.rgb-detail_tex.rgb,detail_tex.a);\n";
			} break;
			case BLEND_MODE_MUL: {
				code += "\tvec3 detail = mix(ALBEDO.rgb,ALBEDO.rgb*detail_tex.rgb,detail_tex.a);\n";
			} break;
		}

		code += "\tvec3 detail_norm = mix(NORMALMAP,detail_norm_tex.rgb,detail_tex.a);\n";
		code += "\tNORMALMAP = mix(NORMALMAP,detail_norm,detail_mask_tex.r);\n";
		code += "\tALBEDO.rgb = mix(ALBEDO.rgb,detail,detail_mask_tex.r);\n";
	}

	if (flags[FLAG_USE_ALPHA_SCISSOR]) {
		code += "\tALPHA_SCISSOR=alpha_scissor_threshold;\n";
	}

	code += "}\n";

	String fallback_mode_str;
	switch (async_mode) {
		case ASYNC_MODE_VISIBLE: {
			fallback_mode_str = "async_visible";
		} break;
		case ASYNC_MODE_HIDDEN: {
			fallback_mode_str = "async_hidden";
		} break;
	}
	code = code.replace_first("render_mode ", "render_mode " + fallback_mode_str + ",");

	ShaderData shader_data;
	shader_data.shader = VS::get_singleton()->shader_create();
	shader_data.users = 1;

	VS::get_singleton()->shader_set_code(shader_data.shader, code);

	shader_map[mk] = shader_data;

	VS::get_singleton()->material_set_shader(_get_material(), shader_data.shader);
}

void SpatialMaterial::flush_changes() {
	material_mutex.lock();

	while (dirty_materials->first()) {
		dirty_materials->first()->self()->_update_shader();
	}

	material_mutex.unlock();
}

void SpatialMaterial::_queue_shader_change() {
	material_mutex.lock();

	if (is_initialized && !element.in_list()) {
		dirty_materials->add(&element);
	}

	material_mutex.unlock();
}

bool SpatialMaterial::_is_shader_dirty() const {
	bool dirty = false;

	material_mutex.lock();

	dirty = element.in_list();

	material_mutex.unlock();

	return dirty;
}
void SpatialMaterial::set_albedo(const Color &p_albedo) {
	albedo = p_albedo;

	VS::get_singleton()->material_set_param(_get_material(), shader_names->albedo, p_albedo);
}

Color SpatialMaterial::get_albedo() const {
	return albedo;
}

void SpatialMaterial::set_specular(float p_specular) {
	specular = p_specular;
	VS::get_singleton()->material_set_param(_get_material(), shader_names->specular, p_specular);
}

float SpatialMaterial::get_specular() const {
	return specular;
}

void SpatialMaterial::set_roughness(float p_roughness) {
	roughness = p_roughness;
	VS::get_singleton()->material_set_param(_get_material(), shader_names->roughness, p_roughness);
}

float SpatialMaterial::get_roughness() const {
	return roughness;
}

void SpatialMaterial::set_metallic(float p_metallic) {
	metallic = p_metallic;
	VS::get_singleton()->material_set_param(_get_material(), shader_names->metallic, p_metallic);
}

float SpatialMaterial::get_metallic() const {
	return metallic;
}

void SpatialMaterial::set_emission(const Color &p_emission) {
	emission = p_emission;
	VS::get_singleton()->material_set_param(_get_material(), shader_names->emission, p_emission);
}
Color SpatialMaterial::get_emission() const {
	return emission;
}

void SpatialMaterial::set_emission_energy(float p_emission_energy) {
	emission_energy = p_emission_energy;
	VS::get_singleton()->material_set_param(_get_material(), shader_names->emission_energy, p_emission_energy);
}
float SpatialMaterial::get_emission_energy() const {
	return emission_energy;
}

void SpatialMaterial::set_normal_scale(float p_normal_scale) {
	normal_scale = p_normal_scale;
	VS::get_singleton()->material_set_param(_get_material(), shader_names->normal_scale, p_normal_scale);
}
float SpatialMaterial::get_normal_scale() const {
	return normal_scale;
}

void SpatialMaterial::set_rim(float p_rim) {
	rim = p_rim;
	VS::get_singleton()->material_set_param(_get_material(), shader_names->rim, p_rim);
}
float SpatialMaterial::get_rim() const {
	return rim;
}

void SpatialMaterial::set_rim_tint(float p_rim_tint) {
	rim_tint = p_rim_tint;
	VS::get_singleton()->material_set_param(_get_material(), shader_names->rim_tint, p_rim_tint);
}
float SpatialMaterial::get_rim_tint() const {
	return rim_tint;
}

void SpatialMaterial::set_ao_light_affect(float p_ao_light_affect) {
	ao_light_affect = p_ao_light_affect;
	VS::get_singleton()->material_set_param(_get_material(), shader_names->ao_light_affect, p_ao_light_affect);
}
float SpatialMaterial::get_ao_light_affect() const {
	return ao_light_affect;
}

void SpatialMaterial::set_clearcoat(float p_clearcoat) {
	clearcoat = p_clearcoat;
	VS::get_singleton()->material_set_param(_get_material(), shader_names->clearcoat, p_clearcoat);
}

float SpatialMaterial::get_clearcoat() const {
	return clearcoat;
}

void SpatialMaterial::set_clearcoat_gloss(float p_clearcoat_gloss) {
	clearcoat_gloss = p_clearcoat_gloss;
	VS::get_singleton()->material_set_param(_get_material(), shader_names->clearcoat_gloss, p_clearcoat_gloss);
}

float SpatialMaterial::get_clearcoat_gloss() const {
	return clearcoat_gloss;
}

void SpatialMaterial::set_anisotropy(float p_anisotropy) {
	anisotropy = p_anisotropy;
	VS::get_singleton()->material_set_param(_get_material(), shader_names->anisotropy, p_anisotropy);
}
float SpatialMaterial::get_anisotropy() const {
	return anisotropy;
}

void SpatialMaterial::set_depth_scale(float p_depth_scale) {
	depth_scale = p_depth_scale;
	VS::get_singleton()->material_set_param(_get_material(), shader_names->depth_scale, p_depth_scale);
}

float SpatialMaterial::get_depth_scale() const {
	return depth_scale;
}

void SpatialMaterial::set_subsurface_scattering_strength(float p_subsurface_scattering_strength) {
	subsurface_scattering_strength = p_subsurface_scattering_strength;
	VS::get_singleton()->material_set_param(_get_material(), shader_names->subsurface_scattering_strength, subsurface_scattering_strength);
}

float SpatialMaterial::get_subsurface_scattering_strength() const {
	return subsurface_scattering_strength;
}

void SpatialMaterial::set_transmission(const Color &p_transmission) {
	transmission = p_transmission;
	VS::get_singleton()->material_set_param(_get_material(), shader_names->transmission, transmission);
}

Color SpatialMaterial::get_transmission() const {
	return transmission;
}

void SpatialMaterial::set_refraction(float p_refraction) {
	refraction = p_refraction;
	VS::get_singleton()->material_set_param(_get_material(), shader_names->refraction, refraction);
}

float SpatialMaterial::get_refraction() const {
	return refraction;
}

void SpatialMaterial::set_detail_uv(DetailUV p_detail_uv) {
	if (detail_uv == p_detail_uv) {
		return;
	}

	detail_uv = p_detail_uv;
	_queue_shader_change();
}
SpatialMaterial::DetailUV SpatialMaterial::get_detail_uv() const {
	return detail_uv;
}

void SpatialMaterial::set_blend_mode(BlendMode p_mode) {
	if (blend_mode == p_mode) {
		return;
	}

	blend_mode = p_mode;
	_queue_shader_change();
}
SpatialMaterial::BlendMode SpatialMaterial::get_blend_mode() const {
	return blend_mode;
}

void SpatialMaterial::set_detail_blend_mode(BlendMode p_mode) {
	detail_blend_mode = p_mode;
	_queue_shader_change();
}
SpatialMaterial::BlendMode SpatialMaterial::get_detail_blend_mode() const {
	return detail_blend_mode;
}

void SpatialMaterial::set_depth_draw_mode(DepthDrawMode p_mode) {
	if (depth_draw_mode == p_mode) {
		return;
	}

	depth_draw_mode = p_mode;
	_queue_shader_change();
}
SpatialMaterial::DepthDrawMode SpatialMaterial::get_depth_draw_mode() const {
	return depth_draw_mode;
}

void SpatialMaterial::set_cull_mode(CullMode p_mode) {
	if (cull_mode == p_mode) {
		return;
	}

	cull_mode = p_mode;
	_queue_shader_change();
}
SpatialMaterial::CullMode SpatialMaterial::get_cull_mode() const {
	return cull_mode;
}

void SpatialMaterial::set_diffuse_mode(DiffuseMode p_mode) {
	if (diffuse_mode == p_mode) {
		return;
	}

	diffuse_mode = p_mode;
	_queue_shader_change();
}
SpatialMaterial::DiffuseMode SpatialMaterial::get_diffuse_mode() const {
	return diffuse_mode;
}

void SpatialMaterial::set_specular_mode(SpecularMode p_mode) {
	if (specular_mode == p_mode) {
		return;
	}

	specular_mode = p_mode;
	_queue_shader_change();
}
SpatialMaterial::SpecularMode SpatialMaterial::get_specular_mode() const {
	return specular_mode;
}

void SpatialMaterial::set_flag(Flags p_flag, bool p_enabled) {
	ERR_FAIL_INDEX(p_flag, FLAG_MAX);

	if (flags[p_flag] == p_enabled) {
		return;
	}

	flags[p_flag] = p_enabled;
	if ((p_flag == FLAG_USE_ALPHA_SCISSOR) || (p_flag == FLAG_UNSHADED) || (p_flag == FLAG_USE_SHADOW_TO_OPACITY)) {
		_change_notify();
	}
	_queue_shader_change();
}

bool SpatialMaterial::get_flag(Flags p_flag) const {
	ERR_FAIL_INDEX_V(p_flag, FLAG_MAX, false);
	return flags[p_flag];
}

void SpatialMaterial::set_feature(Feature p_feature, bool p_enabled) {
	ERR_FAIL_INDEX(p_feature, FEATURE_MAX);
	if (features[p_feature] == p_enabled) {
		return;
	}

	features[p_feature] = p_enabled;
	_change_notify();
	_queue_shader_change();
}

bool SpatialMaterial::get_feature(Feature p_feature) const {
	ERR_FAIL_INDEX_V(p_feature, FEATURE_MAX, false);
	return features[p_feature];
}

void SpatialMaterial::set_texture(TextureParam p_param, const Ref<Texture> &p_texture) {
	ERR_FAIL_INDEX(p_param, TEXTURE_MAX);
	textures[p_param] = p_texture;
	RID rid = p_texture.is_valid() ? p_texture->get_rid() : RID();
	VS::get_singleton()->material_set_param(_get_material(), shader_names->texture_names[p_param], rid);
	_change_notify();
	_queue_shader_change();
}

Ref<Texture> SpatialMaterial::get_texture(TextureParam p_param) const {
	ERR_FAIL_INDEX_V(p_param, TEXTURE_MAX, Ref<Texture>());
	return textures[p_param];
}

Ref<Texture> SpatialMaterial::get_texture_by_name(StringName p_name) const {
	for (int i = 0; i < (int)SpatialMaterial::TEXTURE_MAX; i++) {
		TextureParam param = TextureParam(i);
		if (p_name == shader_names->texture_names[param]) {
			return textures[param];
		}
	}
	return Ref<Texture>();
}

void SpatialMaterial::_validate_feature(const String &text, Feature feature, PropertyInfo &property) const {
	if (property.name.begins_with(text) && property.name != text + "_enabled" && !features[feature]) {
		property.usage = 0;
	}
}

void SpatialMaterial::_validate_high_end(const String &text, PropertyInfo &property) const {
	if (property.name.begins_with(text)) {
		property.usage |= PROPERTY_USAGE_HIGH_END_GFX;
	}
}

void SpatialMaterial::_validate_property(PropertyInfo &property) const {
	_validate_feature("normal", FEATURE_NORMAL_MAPPING, property);
	_validate_feature("emission", FEATURE_EMISSION, property);
	_validate_feature("rim", FEATURE_RIM, property);
	_validate_feature("clearcoat", FEATURE_CLEARCOAT, property);
	_validate_feature("anisotropy", FEATURE_ANISOTROPY, property);
	_validate_feature("ao", FEATURE_AMBIENT_OCCLUSION, property);
	_validate_feature("depth", FEATURE_DEPTH_MAPPING, property);
	_validate_feature("subsurf_scatter", FEATURE_SUBSURACE_SCATTERING, property);
	_validate_feature("transmission", FEATURE_TRANSMISSION, property);
	_validate_feature("refraction", FEATURE_REFRACTION, property);
	_validate_feature("detail", FEATURE_DETAIL, property);

	_validate_high_end("subsurf_scatter", property);
	_validate_high_end("depth", property);

	if (property.name.begins_with("particles_anim_") && billboard_mode != BILLBOARD_PARTICLES) {
		property.usage = 0;
	}

	if (property.name == "params_grow_amount" && !grow_enabled) {
		property.usage = 0;
	}

	if (property.name == "proximity_fade_distance" && !proximity_fade_enabled) {
		property.usage = 0;
	}

	if ((property.name == "distance_fade_max_distance" || property.name == "distance_fade_min_distance") && distance_fade == DISTANCE_FADE_DISABLED) {
		property.usage = 0;
	}

	if (property.name == "params_alpha_scissor_threshold" && !flags[FLAG_USE_ALPHA_SCISSOR]) {
		property.usage = 0;
	}

	if ((property.name == "depth_min_layers" || property.name == "depth_max_layers") && !deep_parallax) {
		property.usage = 0;
	}

	if (flags[FLAG_UNSHADED]) {
		if (property.name.begins_with("anisotropy")) {
			property.usage = 0;
		}

		if (property.name.begins_with("ao")) {
			property.usage = 0;
		}

		if (property.name.begins_with("clearcoat")) {
			property.usage = 0;
		}

		if (property.name.begins_with("emission")) {
			property.usage = 0;
		}

		if (property.name.begins_with("metallic")) {
			property.usage = 0;
		}

		if (property.name.begins_with("normal")) {
			property.usage = 0;
		}

		if (property.name.begins_with("rim")) {
			property.usage = 0;
		}

		if (property.name.begins_with("roughness")) {
			property.usage = 0;
		}

		if (property.name.begins_with("subsurf_scatter")) {
			property.usage = 0;
		}

		if (property.name.begins_with("transmission")) {
			property.usage = 0;
		}
	}
}

void SpatialMaterial::set_line_width(float p_line_width) {
	line_width = p_line_width;
	VS::get_singleton()->material_set_line_width(_get_material(), line_width);
}

float SpatialMaterial::get_line_width() const {
	return line_width;
}

void SpatialMaterial::set_point_size(float p_point_size) {
	point_size = p_point_size;
	VS::get_singleton()->material_set_param(_get_material(), shader_names->point_size, p_point_size);
}

float SpatialMaterial::get_point_size() const {
	return point_size;
}

void SpatialMaterial::set_uv1_scale(const Vector3 &p_scale) {
	uv1_scale = p_scale;
	VS::get_singleton()->material_set_param(_get_material(), shader_names->uv1_scale, p_scale);
}

Vector3 SpatialMaterial::get_uv1_scale() const {
	return uv1_scale;
}

void SpatialMaterial::set_uv1_offset(const Vector3 &p_offset) {
	uv1_offset = p_offset;
	VS::get_singleton()->material_set_param(_get_material(), shader_names->uv1_offset, p_offset);
}
Vector3 SpatialMaterial::get_uv1_offset() const {
	return uv1_offset;
}

void SpatialMaterial::set_uv1_triplanar_blend_sharpness(float p_sharpness) {
	uv1_triplanar_sharpness = p_sharpness;
	VS::get_singleton()->material_set_param(_get_material(), shader_names->uv1_blend_sharpness, p_sharpness);
}

float SpatialMaterial::get_uv1_triplanar_blend_sharpness() const {
	return uv1_triplanar_sharpness;
}

void SpatialMaterial::set_uv2_scale(const Vector3 &p_scale) {
	uv2_scale = p_scale;
	VS::get_singleton()->material_set_param(_get_material(), shader_names->uv2_scale, p_scale);
}

Vector3 SpatialMaterial::get_uv2_scale() const {
	return uv2_scale;
}

void SpatialMaterial::set_uv2_offset(const Vector3 &p_offset) {
	uv2_offset = p_offset;
	VS::get_singleton()->material_set_param(_get_material(), shader_names->uv2_offset, p_offset);
}

Vector3 SpatialMaterial::get_uv2_offset() const {
	return uv2_offset;
}

void SpatialMaterial::set_uv2_triplanar_blend_sharpness(float p_sharpness) {
	uv2_triplanar_sharpness = p_sharpness;
	VS::get_singleton()->material_set_param(_get_material(), shader_names->uv2_blend_sharpness, p_sharpness);
}

float SpatialMaterial::get_uv2_triplanar_blend_sharpness() const {
	return uv2_triplanar_sharpness;
}

void SpatialMaterial::set_billboard_mode(BillboardMode p_mode) {
	billboard_mode = p_mode;
	_queue_shader_change();
	_change_notify();
}

SpatialMaterial::BillboardMode SpatialMaterial::get_billboard_mode() const {
	return billboard_mode;
}

void SpatialMaterial::set_particles_anim_h_frames(int p_frames) {
	particles_anim_h_frames = p_frames;
	VS::get_singleton()->material_set_param(_get_material(), shader_names->particles_anim_h_frames, p_frames);
}

int SpatialMaterial::get_particles_anim_h_frames() const {
	return particles_anim_h_frames;
}
void SpatialMaterial::set_particles_anim_v_frames(int p_frames) {
	particles_anim_v_frames = p_frames;
	VS::get_singleton()->material_set_param(_get_material(), shader_names->particles_anim_v_frames, p_frames);
}

int SpatialMaterial::get_particles_anim_v_frames() const {
	return particles_anim_v_frames;
}

void SpatialMaterial::set_particles_anim_loop(bool p_loop) {
	particles_anim_loop = p_loop;
	VS::get_singleton()->material_set_param(_get_material(), shader_names->particles_anim_loop, particles_anim_loop);
}

bool SpatialMaterial::get_particles_anim_loop() const {
	return particles_anim_loop;
}

void SpatialMaterial::set_depth_deep_parallax(bool p_enable) {
	deep_parallax = p_enable;
	_queue_shader_change();
	_change_notify();
}

bool SpatialMaterial::is_depth_deep_parallax_enabled() const {
	return deep_parallax;
}

void SpatialMaterial::set_depth_deep_parallax_min_layers(int p_layer) {
	deep_parallax_min_layers = p_layer;
	VS::get_singleton()->material_set_param(_get_material(), shader_names->depth_min_layers, p_layer);
}
int SpatialMaterial::get_depth_deep_parallax_min_layers() const {
	return deep_parallax_min_layers;
}

void SpatialMaterial::set_depth_deep_parallax_max_layers(int p_layer) {
	deep_parallax_max_layers = p_layer;
	VS::get_singleton()->material_set_param(_get_material(), shader_names->depth_max_layers, p_layer);
}
int SpatialMaterial::get_depth_deep_parallax_max_layers() const {
	return deep_parallax_max_layers;
}

void SpatialMaterial::set_depth_deep_parallax_flip_tangent(bool p_flip) {
	depth_parallax_flip_tangent = p_flip;
	VS::get_singleton()->material_set_param(_get_material(), shader_names->depth_flip, Vector2(depth_parallax_flip_tangent ? -1 : 1, depth_parallax_flip_binormal ? -1 : 1));
}

bool SpatialMaterial::get_depth_deep_parallax_flip_tangent() const {
	return depth_parallax_flip_tangent;
}

void SpatialMaterial::set_depth_deep_parallax_flip_binormal(bool p_flip) {
	depth_parallax_flip_binormal = p_flip;
	VS::get_singleton()->material_set_param(_get_material(), shader_names->depth_flip, Vector2(depth_parallax_flip_tangent ? -1 : 1, depth_parallax_flip_binormal ? -1 : 1));
}

bool SpatialMaterial::get_depth_deep_parallax_flip_binormal() const {
	return depth_parallax_flip_binormal;
}

void SpatialMaterial::set_grow_enabled(bool p_enable) {
	grow_enabled = p_enable;
	_queue_shader_change();
	_change_notify();
}

bool SpatialMaterial::is_grow_enabled() const {
	return grow_enabled;
}

void SpatialMaterial::set_alpha_scissor_threshold(float p_threshold) {
	alpha_scissor_threshold = p_threshold;
	VS::get_singleton()->material_set_param(_get_material(), shader_names->alpha_scissor_threshold, p_threshold);
}

float SpatialMaterial::get_alpha_scissor_threshold() const {
	return alpha_scissor_threshold;
}

void SpatialMaterial::set_grow(float p_grow) {
	grow = p_grow;
	VS::get_singleton()->material_set_param(_get_material(), shader_names->grow, p_grow);
}

float SpatialMaterial::get_grow() const {
	return grow;
}

static Plane _get_texture_mask(SpatialMaterial::TextureChannel p_channel) {
	static const Plane masks[5] = {
		Plane(1, 0, 0, 0),
		Plane(0, 1, 0, 0),
		Plane(0, 0, 1, 0),
		Plane(0, 0, 0, 1),
		Plane(0.3333333, 0.3333333, 0.3333333, 0),
	};

	return masks[p_channel];
}

void SpatialMaterial::set_metallic_texture_channel(TextureChannel p_channel) {
	ERR_FAIL_INDEX(p_channel, 5);
	metallic_texture_channel = p_channel;
	VS::get_singleton()->material_set_param(_get_material(), shader_names->metallic_texture_channel, _get_texture_mask(p_channel));
}

SpatialMaterial::TextureChannel SpatialMaterial::get_metallic_texture_channel() const {
	return metallic_texture_channel;
}

void SpatialMaterial::set_roughness_texture_channel(TextureChannel p_channel) {
	ERR_FAIL_INDEX(p_channel, 5);
	roughness_texture_channel = p_channel;
	VS::get_singleton()->material_set_param(_get_material(), shader_names->roughness_texture_channel, _get_texture_mask(p_channel));
}

SpatialMaterial::TextureChannel SpatialMaterial::get_roughness_texture_channel() const {
	return roughness_texture_channel;
}

void SpatialMaterial::set_ao_texture_channel(TextureChannel p_channel) {
	ERR_FAIL_INDEX(p_channel, 5);
	ao_texture_channel = p_channel;
	VS::get_singleton()->material_set_param(_get_material(), shader_names->ao_texture_channel, _get_texture_mask(p_channel));
}

SpatialMaterial::TextureChannel SpatialMaterial::get_ao_texture_channel() const {
	return ao_texture_channel;
}

void SpatialMaterial::set_refraction_texture_channel(TextureChannel p_channel) {
	ERR_FAIL_INDEX(p_channel, 5);
	refraction_texture_channel = p_channel;
	VS::get_singleton()->material_set_param(_get_material(), shader_names->refraction_texture_channel, _get_texture_mask(p_channel));
}

SpatialMaterial::TextureChannel SpatialMaterial::get_refraction_texture_channel() const {
	return refraction_texture_channel;
}

RID SpatialMaterial::get_material_rid_for_2d(bool p_shaded, bool p_transparent, bool p_double_sided, bool p_cut_alpha, bool p_opaque_prepass, bool p_billboard, bool p_billboard_y) {
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
		return materials_for_2d[version]->get_rid();
	}

	Ref<SpatialMaterial> material;
	material.instance();

	material->set_flag(FLAG_UNSHADED, !p_shaded);
	material->set_feature(FEATURE_TRANSPARENT, p_transparent);
	material->set_cull_mode(p_double_sided ? CULL_DISABLED : CULL_BACK);
	material->set_depth_draw_mode(p_opaque_prepass ? DEPTH_DRAW_ALPHA_OPAQUE_PREPASS : DEPTH_DRAW_OPAQUE_ONLY);
	material->set_flag(FLAG_SRGB_VERTEX_COLOR, true);
	material->set_flag(FLAG_ALBEDO_FROM_VERTEX_COLOR, true);
	material->set_flag(FLAG_USE_ALPHA_SCISSOR, p_cut_alpha);
	if (p_billboard || p_billboard_y) {
		material->set_flag(FLAG_BILLBOARD_KEEP_SCALE, true);
		material->set_billboard_mode(p_billboard_y ? BILLBOARD_FIXED_Y : BILLBOARD_ENABLED);
	}

	materials_for_2d[version] = material;
	// flush before using so we can access the shader right away
	flush_changes();

	return materials_for_2d[version]->get_rid();
}

void SpatialMaterial::set_on_top_of_alpha() {
	set_feature(FEATURE_TRANSPARENT, true);
	set_render_priority(RENDER_PRIORITY_MAX);
	set_flag(FLAG_DISABLE_DEPTH_TEST, true);
}

void SpatialMaterial::set_proximity_fade(bool p_enable) {
	proximity_fade_enabled = p_enable;
	_queue_shader_change();
	_change_notify();
}

bool SpatialMaterial::is_proximity_fade_enabled() const {
	return proximity_fade_enabled;
}

void SpatialMaterial::set_proximity_fade_distance(float p_distance) {
	proximity_fade_distance = p_distance;
	VS::get_singleton()->material_set_param(_get_material(), shader_names->proximity_fade_distance, p_distance);
}
float SpatialMaterial::get_proximity_fade_distance() const {
	return proximity_fade_distance;
}

void SpatialMaterial::set_distance_fade(DistanceFadeMode p_mode) {
	distance_fade = p_mode;
	_queue_shader_change();
	_change_notify();
}
SpatialMaterial::DistanceFadeMode SpatialMaterial::get_distance_fade() const {
	return distance_fade;
}

void SpatialMaterial::set_distance_fade_max_distance(float p_distance) {
	distance_fade_max_distance = p_distance;
	VS::get_singleton()->material_set_param(_get_material(), shader_names->distance_fade_max, distance_fade_max_distance);
}
float SpatialMaterial::get_distance_fade_max_distance() const {
	return distance_fade_max_distance;
}

void SpatialMaterial::set_distance_fade_min_distance(float p_distance) {
	distance_fade_min_distance = p_distance;
	VS::get_singleton()->material_set_param(_get_material(), shader_names->distance_fade_min, distance_fade_min_distance);
}

float SpatialMaterial::get_distance_fade_min_distance() const {
	return distance_fade_min_distance;
}

void SpatialMaterial::set_emission_operator(EmissionOperator p_op) {
	if (emission_op == p_op) {
		return;
	}
	emission_op = p_op;
	_queue_shader_change();
}

SpatialMaterial::EmissionOperator SpatialMaterial::get_emission_operator() const {
	return emission_op;
}

RID SpatialMaterial::get_shader_rid() const {
	ERR_FAIL_COND_V(!shader_map.has(current_key), RID());
	return shader_map[current_key].shader;
}

Shader::Mode SpatialMaterial::get_shader_mode() const {
	return Shader::MODE_SPATIAL;
}

void SpatialMaterial::set_async_mode(AsyncMode p_mode) {
	async_mode = p_mode;
	_queue_shader_change();
	_change_notify();
}

SpatialMaterial::AsyncMode SpatialMaterial::get_async_mode() const {
	return async_mode;
}

void SpatialMaterial::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_albedo", "albedo"), &SpatialMaterial::set_albedo);
	ClassDB::bind_method(D_METHOD("get_albedo"), &SpatialMaterial::get_albedo);

	ClassDB::bind_method(D_METHOD("set_specular", "specular"), &SpatialMaterial::set_specular);
	ClassDB::bind_method(D_METHOD("get_specular"), &SpatialMaterial::get_specular);

	ClassDB::bind_method(D_METHOD("set_metallic", "metallic"), &SpatialMaterial::set_metallic);
	ClassDB::bind_method(D_METHOD("get_metallic"), &SpatialMaterial::get_metallic);

	ClassDB::bind_method(D_METHOD("set_roughness", "roughness"), &SpatialMaterial::set_roughness);
	ClassDB::bind_method(D_METHOD("get_roughness"), &SpatialMaterial::get_roughness);

	ClassDB::bind_method(D_METHOD("set_emission", "emission"), &SpatialMaterial::set_emission);
	ClassDB::bind_method(D_METHOD("get_emission"), &SpatialMaterial::get_emission);

	ClassDB::bind_method(D_METHOD("set_emission_energy", "emission_energy"), &SpatialMaterial::set_emission_energy);
	ClassDB::bind_method(D_METHOD("get_emission_energy"), &SpatialMaterial::get_emission_energy);

	ClassDB::bind_method(D_METHOD("set_normal_scale", "normal_scale"), &SpatialMaterial::set_normal_scale);
	ClassDB::bind_method(D_METHOD("get_normal_scale"), &SpatialMaterial::get_normal_scale);

	ClassDB::bind_method(D_METHOD("set_rim", "rim"), &SpatialMaterial::set_rim);
	ClassDB::bind_method(D_METHOD("get_rim"), &SpatialMaterial::get_rim);

	ClassDB::bind_method(D_METHOD("set_rim_tint", "rim_tint"), &SpatialMaterial::set_rim_tint);
	ClassDB::bind_method(D_METHOD("get_rim_tint"), &SpatialMaterial::get_rim_tint);

	ClassDB::bind_method(D_METHOD("set_clearcoat", "clearcoat"), &SpatialMaterial::set_clearcoat);
	ClassDB::bind_method(D_METHOD("get_clearcoat"), &SpatialMaterial::get_clearcoat);

	ClassDB::bind_method(D_METHOD("set_clearcoat_gloss", "clearcoat_gloss"), &SpatialMaterial::set_clearcoat_gloss);
	ClassDB::bind_method(D_METHOD("get_clearcoat_gloss"), &SpatialMaterial::get_clearcoat_gloss);

	ClassDB::bind_method(D_METHOD("set_anisotropy", "anisotropy"), &SpatialMaterial::set_anisotropy);
	ClassDB::bind_method(D_METHOD("get_anisotropy"), &SpatialMaterial::get_anisotropy);

	ClassDB::bind_method(D_METHOD("set_depth_scale", "depth_scale"), &SpatialMaterial::set_depth_scale);
	ClassDB::bind_method(D_METHOD("get_depth_scale"), &SpatialMaterial::get_depth_scale);

	ClassDB::bind_method(D_METHOD("set_subsurface_scattering_strength", "strength"), &SpatialMaterial::set_subsurface_scattering_strength);
	ClassDB::bind_method(D_METHOD("get_subsurface_scattering_strength"), &SpatialMaterial::get_subsurface_scattering_strength);

	ClassDB::bind_method(D_METHOD("set_transmission", "transmission"), &SpatialMaterial::set_transmission);
	ClassDB::bind_method(D_METHOD("get_transmission"), &SpatialMaterial::get_transmission);

	ClassDB::bind_method(D_METHOD("set_refraction", "refraction"), &SpatialMaterial::set_refraction);
	ClassDB::bind_method(D_METHOD("get_refraction"), &SpatialMaterial::get_refraction);

	ClassDB::bind_method(D_METHOD("set_line_width", "line_width"), &SpatialMaterial::set_line_width);
	ClassDB::bind_method(D_METHOD("get_line_width"), &SpatialMaterial::get_line_width);

	ClassDB::bind_method(D_METHOD("set_point_size", "point_size"), &SpatialMaterial::set_point_size);
	ClassDB::bind_method(D_METHOD("get_point_size"), &SpatialMaterial::get_point_size);

	ClassDB::bind_method(D_METHOD("set_detail_uv", "detail_uv"), &SpatialMaterial::set_detail_uv);
	ClassDB::bind_method(D_METHOD("get_detail_uv"), &SpatialMaterial::get_detail_uv);

	ClassDB::bind_method(D_METHOD("set_blend_mode", "blend_mode"), &SpatialMaterial::set_blend_mode);
	ClassDB::bind_method(D_METHOD("get_blend_mode"), &SpatialMaterial::get_blend_mode);

	ClassDB::bind_method(D_METHOD("set_depth_draw_mode", "depth_draw_mode"), &SpatialMaterial::set_depth_draw_mode);
	ClassDB::bind_method(D_METHOD("get_depth_draw_mode"), &SpatialMaterial::get_depth_draw_mode);

	ClassDB::bind_method(D_METHOD("set_cull_mode", "cull_mode"), &SpatialMaterial::set_cull_mode);
	ClassDB::bind_method(D_METHOD("get_cull_mode"), &SpatialMaterial::get_cull_mode);

	ClassDB::bind_method(D_METHOD("set_diffuse_mode", "diffuse_mode"), &SpatialMaterial::set_diffuse_mode);
	ClassDB::bind_method(D_METHOD("get_diffuse_mode"), &SpatialMaterial::get_diffuse_mode);

	ClassDB::bind_method(D_METHOD("set_specular_mode", "specular_mode"), &SpatialMaterial::set_specular_mode);
	ClassDB::bind_method(D_METHOD("get_specular_mode"), &SpatialMaterial::get_specular_mode);

	ClassDB::bind_method(D_METHOD("set_flag", "flag", "enable"), &SpatialMaterial::set_flag);
	ClassDB::bind_method(D_METHOD("get_flag", "flag"), &SpatialMaterial::get_flag);

	ClassDB::bind_method(D_METHOD("set_feature", "feature", "enable"), &SpatialMaterial::set_feature);
	ClassDB::bind_method(D_METHOD("get_feature", "feature"), &SpatialMaterial::get_feature);

	ClassDB::bind_method(D_METHOD("set_texture", "param", "texture"), &SpatialMaterial::set_texture);
	ClassDB::bind_method(D_METHOD("get_texture", "param"), &SpatialMaterial::get_texture);

	ClassDB::bind_method(D_METHOD("set_detail_blend_mode", "detail_blend_mode"), &SpatialMaterial::set_detail_blend_mode);
	ClassDB::bind_method(D_METHOD("get_detail_blend_mode"), &SpatialMaterial::get_detail_blend_mode);

	ClassDB::bind_method(D_METHOD("set_uv1_scale", "scale"), &SpatialMaterial::set_uv1_scale);
	ClassDB::bind_method(D_METHOD("get_uv1_scale"), &SpatialMaterial::get_uv1_scale);

	ClassDB::bind_method(D_METHOD("set_uv1_offset", "offset"), &SpatialMaterial::set_uv1_offset);
	ClassDB::bind_method(D_METHOD("get_uv1_offset"), &SpatialMaterial::get_uv1_offset);

	ClassDB::bind_method(D_METHOD("set_uv1_triplanar_blend_sharpness", "sharpness"), &SpatialMaterial::set_uv1_triplanar_blend_sharpness);
	ClassDB::bind_method(D_METHOD("get_uv1_triplanar_blend_sharpness"), &SpatialMaterial::get_uv1_triplanar_blend_sharpness);

	ClassDB::bind_method(D_METHOD("set_uv2_scale", "scale"), &SpatialMaterial::set_uv2_scale);
	ClassDB::bind_method(D_METHOD("get_uv2_scale"), &SpatialMaterial::get_uv2_scale);

	ClassDB::bind_method(D_METHOD("set_uv2_offset", "offset"), &SpatialMaterial::set_uv2_offset);
	ClassDB::bind_method(D_METHOD("get_uv2_offset"), &SpatialMaterial::get_uv2_offset);

	ClassDB::bind_method(D_METHOD("set_uv2_triplanar_blend_sharpness", "sharpness"), &SpatialMaterial::set_uv2_triplanar_blend_sharpness);
	ClassDB::bind_method(D_METHOD("get_uv2_triplanar_blend_sharpness"), &SpatialMaterial::get_uv2_triplanar_blend_sharpness);

	ClassDB::bind_method(D_METHOD("set_billboard_mode", "mode"), &SpatialMaterial::set_billboard_mode);
	ClassDB::bind_method(D_METHOD("get_billboard_mode"), &SpatialMaterial::get_billboard_mode);

	ClassDB::bind_method(D_METHOD("set_particles_anim_h_frames", "frames"), &SpatialMaterial::set_particles_anim_h_frames);
	ClassDB::bind_method(D_METHOD("get_particles_anim_h_frames"), &SpatialMaterial::get_particles_anim_h_frames);

	ClassDB::bind_method(D_METHOD("set_particles_anim_v_frames", "frames"), &SpatialMaterial::set_particles_anim_v_frames);
	ClassDB::bind_method(D_METHOD("get_particles_anim_v_frames"), &SpatialMaterial::get_particles_anim_v_frames);

	ClassDB::bind_method(D_METHOD("set_particles_anim_loop", "loop"), &SpatialMaterial::set_particles_anim_loop);
	ClassDB::bind_method(D_METHOD("get_particles_anim_loop"), &SpatialMaterial::get_particles_anim_loop);

	ClassDB::bind_method(D_METHOD("set_depth_deep_parallax", "enable"), &SpatialMaterial::set_depth_deep_parallax);
	ClassDB::bind_method(D_METHOD("is_depth_deep_parallax_enabled"), &SpatialMaterial::is_depth_deep_parallax_enabled);

	ClassDB::bind_method(D_METHOD("set_depth_deep_parallax_min_layers", "layer"), &SpatialMaterial::set_depth_deep_parallax_min_layers);
	ClassDB::bind_method(D_METHOD("get_depth_deep_parallax_min_layers"), &SpatialMaterial::get_depth_deep_parallax_min_layers);

	ClassDB::bind_method(D_METHOD("set_depth_deep_parallax_max_layers", "layer"), &SpatialMaterial::set_depth_deep_parallax_max_layers);
	ClassDB::bind_method(D_METHOD("get_depth_deep_parallax_max_layers"), &SpatialMaterial::get_depth_deep_parallax_max_layers);

	ClassDB::bind_method(D_METHOD("set_depth_deep_parallax_flip_tangent", "flip"), &SpatialMaterial::set_depth_deep_parallax_flip_tangent);
	ClassDB::bind_method(D_METHOD("get_depth_deep_parallax_flip_tangent"), &SpatialMaterial::get_depth_deep_parallax_flip_tangent);

	ClassDB::bind_method(D_METHOD("set_depth_deep_parallax_flip_binormal", "flip"), &SpatialMaterial::set_depth_deep_parallax_flip_binormal);
	ClassDB::bind_method(D_METHOD("get_depth_deep_parallax_flip_binormal"), &SpatialMaterial::get_depth_deep_parallax_flip_binormal);

	ClassDB::bind_method(D_METHOD("set_grow", "amount"), &SpatialMaterial::set_grow);
	ClassDB::bind_method(D_METHOD("get_grow"), &SpatialMaterial::get_grow);

	ClassDB::bind_method(D_METHOD("set_emission_operator", "operator"), &SpatialMaterial::set_emission_operator);
	ClassDB::bind_method(D_METHOD("get_emission_operator"), &SpatialMaterial::get_emission_operator);

	ClassDB::bind_method(D_METHOD("set_ao_light_affect", "amount"), &SpatialMaterial::set_ao_light_affect);
	ClassDB::bind_method(D_METHOD("get_ao_light_affect"), &SpatialMaterial::get_ao_light_affect);

	ClassDB::bind_method(D_METHOD("set_alpha_scissor_threshold", "threshold"), &SpatialMaterial::set_alpha_scissor_threshold);
	ClassDB::bind_method(D_METHOD("get_alpha_scissor_threshold"), &SpatialMaterial::get_alpha_scissor_threshold);

	ClassDB::bind_method(D_METHOD("set_grow_enabled", "enable"), &SpatialMaterial::set_grow_enabled);
	ClassDB::bind_method(D_METHOD("is_grow_enabled"), &SpatialMaterial::is_grow_enabled);

	ClassDB::bind_method(D_METHOD("set_metallic_texture_channel", "channel"), &SpatialMaterial::set_metallic_texture_channel);
	ClassDB::bind_method(D_METHOD("get_metallic_texture_channel"), &SpatialMaterial::get_metallic_texture_channel);

	ClassDB::bind_method(D_METHOD("set_roughness_texture_channel", "channel"), &SpatialMaterial::set_roughness_texture_channel);
	ClassDB::bind_method(D_METHOD("get_roughness_texture_channel"), &SpatialMaterial::get_roughness_texture_channel);

	ClassDB::bind_method(D_METHOD("set_ao_texture_channel", "channel"), &SpatialMaterial::set_ao_texture_channel);
	ClassDB::bind_method(D_METHOD("get_ao_texture_channel"), &SpatialMaterial::get_ao_texture_channel);

	ClassDB::bind_method(D_METHOD("set_refraction_texture_channel", "channel"), &SpatialMaterial::set_refraction_texture_channel);
	ClassDB::bind_method(D_METHOD("get_refraction_texture_channel"), &SpatialMaterial::get_refraction_texture_channel);

	ClassDB::bind_method(D_METHOD("set_proximity_fade", "enabled"), &SpatialMaterial::set_proximity_fade);
	ClassDB::bind_method(D_METHOD("is_proximity_fade_enabled"), &SpatialMaterial::is_proximity_fade_enabled);

	ClassDB::bind_method(D_METHOD("set_proximity_fade_distance", "distance"), &SpatialMaterial::set_proximity_fade_distance);
	ClassDB::bind_method(D_METHOD("get_proximity_fade_distance"), &SpatialMaterial::get_proximity_fade_distance);

	ClassDB::bind_method(D_METHOD("set_distance_fade", "mode"), &SpatialMaterial::set_distance_fade);
	ClassDB::bind_method(D_METHOD("get_distance_fade"), &SpatialMaterial::get_distance_fade);

	ClassDB::bind_method(D_METHOD("set_distance_fade_max_distance", "distance"), &SpatialMaterial::set_distance_fade_max_distance);
	ClassDB::bind_method(D_METHOD("get_distance_fade_max_distance"), &SpatialMaterial::get_distance_fade_max_distance);

	ClassDB::bind_method(D_METHOD("set_distance_fade_min_distance", "distance"), &SpatialMaterial::set_distance_fade_min_distance);
	ClassDB::bind_method(D_METHOD("get_distance_fade_min_distance"), &SpatialMaterial::get_distance_fade_min_distance);

	ClassDB::bind_method(D_METHOD("set_async_mode", "mode"), &SpatialMaterial::set_async_mode);
	ClassDB::bind_method(D_METHOD("get_async_mode"), &SpatialMaterial::get_async_mode);

	ADD_GROUP("Flags", "flags_");
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "flags_transparent"), "set_feature", "get_feature", FEATURE_TRANSPARENT);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "flags_use_shadow_to_opacity"), "set_flag", "get_flag", FLAG_USE_SHADOW_TO_OPACITY);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "flags_unshaded"), "set_flag", "get_flag", FLAG_UNSHADED);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "flags_vertex_lighting"), "set_flag", "get_flag", FLAG_USE_VERTEX_LIGHTING);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "flags_no_depth_test"), "set_flag", "get_flag", FLAG_DISABLE_DEPTH_TEST);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "flags_use_point_size"), "set_flag", "get_flag", FLAG_USE_POINT_SIZE);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "flags_world_triplanar"), "set_flag", "get_flag", FLAG_TRIPLANAR_USE_WORLD);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "flags_fixed_size"), "set_flag", "get_flag", FLAG_FIXED_SIZE);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "flags_albedo_tex_force_srgb"), "set_flag", "get_flag", FLAG_ALBEDO_TEXTURE_FORCE_SRGB);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "flags_do_not_receive_shadows"), "set_flag", "get_flag", FLAG_DONT_RECEIVE_SHADOWS);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "flags_disable_ambient_light"), "set_flag", "get_flag", FLAG_DISABLE_AMBIENT_LIGHT);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "flags_ensure_correct_normals"), "set_flag", "get_flag", FLAG_ENSURE_CORRECT_NORMALS);
	ADD_GROUP("Vertex Color", "vertex_color");
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "vertex_color_use_as_albedo"), "set_flag", "get_flag", FLAG_ALBEDO_FROM_VERTEX_COLOR);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "vertex_color_is_srgb"), "set_flag", "get_flag", FLAG_SRGB_VERTEX_COLOR);

	ADD_GROUP("Parameters", "params_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "params_diffuse_mode", PROPERTY_HINT_ENUM, "Burley,Lambert,Lambert Wrap,Oren Nayar,Toon"), "set_diffuse_mode", "get_diffuse_mode");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "params_specular_mode", PROPERTY_HINT_ENUM, "SchlickGGX,Blinn,Phong,Toon,Disabled"), "set_specular_mode", "get_specular_mode");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "params_blend_mode", PROPERTY_HINT_ENUM, "Mix,Add,Sub,Mul"), "set_blend_mode", "get_blend_mode");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "params_cull_mode", PROPERTY_HINT_ENUM, "Back,Front,Disabled"), "set_cull_mode", "get_cull_mode");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "params_depth_draw_mode", PROPERTY_HINT_ENUM, "Opaque Only,Always,Never,Opaque Pre-Pass"), "set_depth_draw_mode", "get_depth_draw_mode");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "params_line_width", PROPERTY_HINT_RANGE, "0.1,128,0.1"), "set_line_width", "get_line_width");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "params_point_size", PROPERTY_HINT_RANGE, "0.1,128,0.1"), "set_point_size", "get_point_size");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "params_billboard_mode", PROPERTY_HINT_ENUM, "Disabled,Enabled,Y-Billboard,Particle Billboard"), "set_billboard_mode", "get_billboard_mode");
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "params_billboard_keep_scale"), "set_flag", "get_flag", FLAG_BILLBOARD_KEEP_SCALE);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "params_grow"), "set_grow_enabled", "is_grow_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "params_grow_amount", PROPERTY_HINT_RANGE, "-16,16,0.001"), "set_grow", "get_grow");
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "params_use_alpha_scissor"), "set_flag", "get_flag", FLAG_USE_ALPHA_SCISSOR);
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "params_alpha_scissor_threshold", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_alpha_scissor_threshold", "get_alpha_scissor_threshold");
	ADD_GROUP("Particles Anim", "particles_anim_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "particles_anim_h_frames", PROPERTY_HINT_RANGE, "1,128,1"), "set_particles_anim_h_frames", "get_particles_anim_h_frames");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "particles_anim_v_frames", PROPERTY_HINT_RANGE, "1,128,1"), "set_particles_anim_v_frames", "get_particles_anim_v_frames");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "particles_anim_loop"), "set_particles_anim_loop", "get_particles_anim_loop");

	ADD_GROUP("Albedo", "albedo_");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "albedo_color"), "set_albedo", "get_albedo");
	ADD_PROPERTYI(PropertyInfo(Variant::OBJECT, "albedo_texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture"), "set_texture", "get_texture", TEXTURE_ALBEDO);

	ADD_GROUP("Metallic", "metallic_");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "metallic", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_metallic", "get_metallic");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "metallic_specular", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_specular", "get_specular");
	ADD_PROPERTYI(PropertyInfo(Variant::OBJECT, "metallic_texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture"), "set_texture", "get_texture", TEXTURE_METALLIC);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "metallic_texture_channel", PROPERTY_HINT_ENUM, "Red,Green,Blue,Alpha,Gray"), "set_metallic_texture_channel", "get_metallic_texture_channel");

	ADD_GROUP("Roughness", "roughness_");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "roughness", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_roughness", "get_roughness");
	ADD_PROPERTYI(PropertyInfo(Variant::OBJECT, "roughness_texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture"), "set_texture", "get_texture", TEXTURE_ROUGHNESS);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "roughness_texture_channel", PROPERTY_HINT_ENUM, "Red,Green,Blue,Alpha,Gray"), "set_roughness_texture_channel", "get_roughness_texture_channel");

	ADD_GROUP("Emission", "emission_");
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "emission_enabled"), "set_feature", "get_feature", FEATURE_EMISSION);
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "emission", PROPERTY_HINT_COLOR_NO_ALPHA), "set_emission", "get_emission");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "emission_energy", PROPERTY_HINT_RANGE, "0,16,0.01,or_greater"), "set_emission_energy", "get_emission_energy");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "emission_operator", PROPERTY_HINT_ENUM, "Add,Multiply"), "set_emission_operator", "get_emission_operator");
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "emission_on_uv2"), "set_flag", "get_flag", FLAG_EMISSION_ON_UV2);
	ADD_PROPERTYI(PropertyInfo(Variant::OBJECT, "emission_texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture"), "set_texture", "get_texture", TEXTURE_EMISSION);

	ADD_GROUP("NormalMap", "normal_");
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "normal_enabled"), "set_feature", "get_feature", FEATURE_NORMAL_MAPPING);
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "normal_scale", PROPERTY_HINT_RANGE, "-16,16,0.01"), "set_normal_scale", "get_normal_scale");
	ADD_PROPERTYI(PropertyInfo(Variant::OBJECT, "normal_texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture"), "set_texture", "get_texture", TEXTURE_NORMAL);

	ADD_GROUP("Rim", "rim_");
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "rim_enabled"), "set_feature", "get_feature", FEATURE_RIM);
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "rim", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_rim", "get_rim");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "rim_tint", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_rim_tint", "get_rim_tint");
	ADD_PROPERTYI(PropertyInfo(Variant::OBJECT, "rim_texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture"), "set_texture", "get_texture", TEXTURE_RIM);

	ADD_GROUP("Clearcoat", "clearcoat_");
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "clearcoat_enabled"), "set_feature", "get_feature", FEATURE_CLEARCOAT);
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "clearcoat", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_clearcoat", "get_clearcoat");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "clearcoat_gloss", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_clearcoat_gloss", "get_clearcoat_gloss");
	ADD_PROPERTYI(PropertyInfo(Variant::OBJECT, "clearcoat_texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture"), "set_texture", "get_texture", TEXTURE_CLEARCOAT);

	ADD_GROUP("Anisotropy", "anisotropy_");
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "anisotropy_enabled"), "set_feature", "get_feature", FEATURE_ANISOTROPY);
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "anisotropy", PROPERTY_HINT_RANGE, "-1,1,0.01"), "set_anisotropy", "get_anisotropy");
	ADD_PROPERTYI(PropertyInfo(Variant::OBJECT, "anisotropy_flowmap", PROPERTY_HINT_RESOURCE_TYPE, "Texture"), "set_texture", "get_texture", TEXTURE_FLOWMAP);

	ADD_GROUP("Ambient Occlusion", "ao_");
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "ao_enabled"), "set_feature", "get_feature", FEATURE_AMBIENT_OCCLUSION);
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "ao_light_affect", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_ao_light_affect", "get_ao_light_affect");
	ADD_PROPERTYI(PropertyInfo(Variant::OBJECT, "ao_texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture"), "set_texture", "get_texture", TEXTURE_AMBIENT_OCCLUSION);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "ao_on_uv2"), "set_flag", "get_flag", FLAG_AO_ON_UV2);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "ao_texture_channel", PROPERTY_HINT_ENUM, "Red,Green,Blue,Alpha,Gray"), "set_ao_texture_channel", "get_ao_texture_channel");

	ADD_GROUP("Depth", "depth_");
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "depth_enabled"), "set_feature", "get_feature", FEATURE_DEPTH_MAPPING);
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "depth_scale", PROPERTY_HINT_RANGE, "-16,16,0.001"), "set_depth_scale", "get_depth_scale");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "depth_deep_parallax"), "set_depth_deep_parallax", "is_depth_deep_parallax_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "depth_min_layers", PROPERTY_HINT_RANGE, "1,64,1"), "set_depth_deep_parallax_min_layers", "get_depth_deep_parallax_min_layers");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "depth_max_layers", PROPERTY_HINT_RANGE, "1,64,1"), "set_depth_deep_parallax_max_layers", "get_depth_deep_parallax_max_layers");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "depth_flip_tangent"), "set_depth_deep_parallax_flip_tangent", "get_depth_deep_parallax_flip_tangent");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "depth_flip_binormal"), "set_depth_deep_parallax_flip_binormal", "get_depth_deep_parallax_flip_binormal");
	ADD_PROPERTYI(PropertyInfo(Variant::OBJECT, "depth_texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture"), "set_texture", "get_texture", TEXTURE_DEPTH);

	ADD_GROUP("Subsurf Scatter", "subsurf_scatter_");
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "subsurf_scatter_enabled"), "set_feature", "get_feature", FEATURE_SUBSURACE_SCATTERING);
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "subsurf_scatter_strength", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_subsurface_scattering_strength", "get_subsurface_scattering_strength");
	ADD_PROPERTYI(PropertyInfo(Variant::OBJECT, "subsurf_scatter_texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture"), "set_texture", "get_texture", TEXTURE_SUBSURFACE_SCATTERING);

	ADD_GROUP("Transmission", "transmission_");
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "transmission_enabled"), "set_feature", "get_feature", FEATURE_TRANSMISSION);
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "transmission", PROPERTY_HINT_COLOR_NO_ALPHA), "set_transmission", "get_transmission");
	ADD_PROPERTYI(PropertyInfo(Variant::OBJECT, "transmission_texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture"), "set_texture", "get_texture", TEXTURE_TRANSMISSION);

	ADD_GROUP("Refraction", "refraction_");
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "refraction_enabled"), "set_feature", "get_feature", FEATURE_REFRACTION);
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "refraction_scale", PROPERTY_HINT_RANGE, "-1,1,0.01"), "set_refraction", "get_refraction");
	ADD_PROPERTYI(PropertyInfo(Variant::OBJECT, "refraction_texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture"), "set_texture", "get_texture", TEXTURE_REFRACTION);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "refraction_texture_channel", PROPERTY_HINT_ENUM, "Red,Green,Blue,Alpha,Gray"), "set_refraction_texture_channel", "get_refraction_texture_channel");

	ADD_GROUP("Detail", "detail_");
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "detail_enabled"), "set_feature", "get_feature", FEATURE_DETAIL);
	ADD_PROPERTYI(PropertyInfo(Variant::OBJECT, "detail_mask", PROPERTY_HINT_RESOURCE_TYPE, "Texture"), "set_texture", "get_texture", TEXTURE_DETAIL_MASK);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "detail_blend_mode", PROPERTY_HINT_ENUM, "Mix,Add,Sub,Mul"), "set_detail_blend_mode", "get_detail_blend_mode");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "detail_uv_layer", PROPERTY_HINT_ENUM, "UV1,UV2"), "set_detail_uv", "get_detail_uv");
	ADD_PROPERTYI(PropertyInfo(Variant::OBJECT, "detail_albedo", PROPERTY_HINT_RESOURCE_TYPE, "Texture"), "set_texture", "get_texture", TEXTURE_DETAIL_ALBEDO);
	ADD_PROPERTYI(PropertyInfo(Variant::OBJECT, "detail_normal", PROPERTY_HINT_RESOURCE_TYPE, "Texture"), "set_texture", "get_texture", TEXTURE_DETAIL_NORMAL);

	ADD_GROUP("UV1", "uv1_");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "uv1_scale"), "set_uv1_scale", "get_uv1_scale");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "uv1_offset"), "set_uv1_offset", "get_uv1_offset");
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "uv1_triplanar"), "set_flag", "get_flag", FLAG_UV1_USE_TRIPLANAR);
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "uv1_triplanar_sharpness", PROPERTY_HINT_EXP_EASING), "set_uv1_triplanar_blend_sharpness", "get_uv1_triplanar_blend_sharpness");

	ADD_GROUP("UV2", "uv2_");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "uv2_scale"), "set_uv2_scale", "get_uv2_scale");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "uv2_offset"), "set_uv2_offset", "get_uv2_offset");
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "uv2_triplanar"), "set_flag", "get_flag", FLAG_UV2_USE_TRIPLANAR);
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "uv2_triplanar_sharpness", PROPERTY_HINT_EXP_EASING), "set_uv2_triplanar_blend_sharpness", "get_uv2_triplanar_blend_sharpness");

	ADD_GROUP("Proximity Fade", "proximity_fade_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "proximity_fade_enable"), "set_proximity_fade", "is_proximity_fade_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "proximity_fade_distance", PROPERTY_HINT_RANGE, "0,4096,0.01"), "set_proximity_fade_distance", "get_proximity_fade_distance");
	ADD_GROUP("Distance Fade", "distance_fade_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "distance_fade_mode", PROPERTY_HINT_ENUM, "Disabled,PixelAlpha,PixelDither,ObjectDither"), "set_distance_fade", "get_distance_fade");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "distance_fade_min_distance", PROPERTY_HINT_RANGE, "0,4096,0.01"), "set_distance_fade_min_distance", "get_distance_fade_min_distance");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "distance_fade_max_distance", PROPERTY_HINT_RANGE, "0,4096,0.01"), "set_distance_fade_max_distance", "get_distance_fade_max_distance");

	ADD_PROPERTY(PropertyInfo(Variant::INT, "async_mode", PROPERTY_HINT_ENUM, "Visible,Hidden"), "set_async_mode", "get_async_mode");

	BIND_ENUM_CONSTANT(TEXTURE_ALBEDO);
	BIND_ENUM_CONSTANT(TEXTURE_METALLIC);
	BIND_ENUM_CONSTANT(TEXTURE_ROUGHNESS);
	BIND_ENUM_CONSTANT(TEXTURE_EMISSION);
	BIND_ENUM_CONSTANT(TEXTURE_NORMAL);
	BIND_ENUM_CONSTANT(TEXTURE_RIM);
	BIND_ENUM_CONSTANT(TEXTURE_CLEARCOAT);
	BIND_ENUM_CONSTANT(TEXTURE_FLOWMAP);
	BIND_ENUM_CONSTANT(TEXTURE_AMBIENT_OCCLUSION);
	BIND_ENUM_CONSTANT(TEXTURE_DEPTH);
	BIND_ENUM_CONSTANT(TEXTURE_SUBSURFACE_SCATTERING);
	BIND_ENUM_CONSTANT(TEXTURE_TRANSMISSION);
	BIND_ENUM_CONSTANT(TEXTURE_REFRACTION);
	BIND_ENUM_CONSTANT(TEXTURE_DETAIL_MASK);
	BIND_ENUM_CONSTANT(TEXTURE_DETAIL_ALBEDO);
	BIND_ENUM_CONSTANT(TEXTURE_DETAIL_NORMAL);
	BIND_ENUM_CONSTANT(TEXTURE_MAX);

	BIND_ENUM_CONSTANT(DETAIL_UV_1);
	BIND_ENUM_CONSTANT(DETAIL_UV_2);

	BIND_ENUM_CONSTANT(FEATURE_TRANSPARENT);
	BIND_ENUM_CONSTANT(FEATURE_EMISSION);
	BIND_ENUM_CONSTANT(FEATURE_NORMAL_MAPPING);
	BIND_ENUM_CONSTANT(FEATURE_RIM);
	BIND_ENUM_CONSTANT(FEATURE_CLEARCOAT);
	BIND_ENUM_CONSTANT(FEATURE_ANISOTROPY);
	BIND_ENUM_CONSTANT(FEATURE_AMBIENT_OCCLUSION);
	BIND_ENUM_CONSTANT(FEATURE_DEPTH_MAPPING);
	BIND_ENUM_CONSTANT(FEATURE_SUBSURACE_SCATTERING);
	BIND_ENUM_CONSTANT(FEATURE_TRANSMISSION);
	BIND_ENUM_CONSTANT(FEATURE_REFRACTION);
	BIND_ENUM_CONSTANT(FEATURE_DETAIL);
	BIND_ENUM_CONSTANT(FEATURE_MAX);

	BIND_ENUM_CONSTANT(BLEND_MODE_MIX);
	BIND_ENUM_CONSTANT(BLEND_MODE_ADD);
	BIND_ENUM_CONSTANT(BLEND_MODE_SUB);
	BIND_ENUM_CONSTANT(BLEND_MODE_MUL);

	BIND_ENUM_CONSTANT(DEPTH_DRAW_OPAQUE_ONLY);
	BIND_ENUM_CONSTANT(DEPTH_DRAW_ALWAYS);
	BIND_ENUM_CONSTANT(DEPTH_DRAW_DISABLED);
	BIND_ENUM_CONSTANT(DEPTH_DRAW_ALPHA_OPAQUE_PREPASS);

	BIND_ENUM_CONSTANT(CULL_BACK);
	BIND_ENUM_CONSTANT(CULL_FRONT);
	BIND_ENUM_CONSTANT(CULL_DISABLED);

	BIND_ENUM_CONSTANT(FLAG_UNSHADED);
	BIND_ENUM_CONSTANT(FLAG_USE_VERTEX_LIGHTING);
	BIND_ENUM_CONSTANT(FLAG_DISABLE_DEPTH_TEST);
	BIND_ENUM_CONSTANT(FLAG_ALBEDO_FROM_VERTEX_COLOR);
	BIND_ENUM_CONSTANT(FLAG_SRGB_VERTEX_COLOR);
	BIND_ENUM_CONSTANT(FLAG_USE_POINT_SIZE);
	BIND_ENUM_CONSTANT(FLAG_FIXED_SIZE);
	BIND_ENUM_CONSTANT(FLAG_BILLBOARD_KEEP_SCALE);
	BIND_ENUM_CONSTANT(FLAG_UV1_USE_TRIPLANAR);
	BIND_ENUM_CONSTANT(FLAG_UV2_USE_TRIPLANAR);
	BIND_ENUM_CONSTANT(FLAG_AO_ON_UV2);
	BIND_ENUM_CONSTANT(FLAG_EMISSION_ON_UV2);
	BIND_ENUM_CONSTANT(FLAG_USE_ALPHA_SCISSOR);
	BIND_ENUM_CONSTANT(FLAG_TRIPLANAR_USE_WORLD);
	BIND_ENUM_CONSTANT(FLAG_ALBEDO_TEXTURE_FORCE_SRGB);
	BIND_ENUM_CONSTANT(FLAG_DONT_RECEIVE_SHADOWS);
	BIND_ENUM_CONSTANT(FLAG_DISABLE_AMBIENT_LIGHT);
	BIND_ENUM_CONSTANT(FLAG_ENSURE_CORRECT_NORMALS);
	BIND_ENUM_CONSTANT(FLAG_USE_SHADOW_TO_OPACITY);
	BIND_ENUM_CONSTANT(FLAG_MAX);

	BIND_ENUM_CONSTANT(DIFFUSE_BURLEY);
	BIND_ENUM_CONSTANT(DIFFUSE_LAMBERT);
	BIND_ENUM_CONSTANT(DIFFUSE_LAMBERT_WRAP);
	BIND_ENUM_CONSTANT(DIFFUSE_OREN_NAYAR);
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

	BIND_ENUM_CONSTANT(ASYNC_MODE_VISIBLE);
	BIND_ENUM_CONSTANT(ASYNC_MODE_HIDDEN);
}

SpatialMaterial::SpatialMaterial() :
		element(this) {
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
	set_depth_scale(0.05);
	set_subsurface_scattering_strength(0);
	set_transmission(Color(0, 0, 0));
	set_refraction(0.05);
	set_line_width(1);
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
	set_alpha_scissor_threshold(0.98);
	emission_op = EMISSION_OP_ADD;

	proximity_fade_enabled = false;
	distance_fade = DISTANCE_FADE_DISABLED;
	set_proximity_fade_distance(1);
	set_distance_fade_min_distance(0);
	set_distance_fade_max_distance(10);

	set_ao_light_affect(0.0);

	set_metallic_texture_channel(TEXTURE_CHANNEL_RED);
	set_roughness_texture_channel(TEXTURE_CHANNEL_RED);
	set_ao_texture_channel(TEXTURE_CHANNEL_RED);
	set_refraction_texture_channel(TEXTURE_CHANNEL_RED);

	grow_enabled = false;
	set_grow(0.0);

	deep_parallax = false;
	depth_parallax_flip_tangent = false;
	depth_parallax_flip_binormal = false;
	set_depth_deep_parallax_min_layers(8);
	set_depth_deep_parallax_max_layers(32);
	set_depth_deep_parallax_flip_tangent(false); //also sets binormal

	detail_uv = DETAIL_UV_1;
	blend_mode = BLEND_MODE_MIX;
	detail_blend_mode = BLEND_MODE_MIX;
	depth_draw_mode = DEPTH_DRAW_OPAQUE_ONLY;
	cull_mode = CULL_BACK;
	for (int i = 0; i < FLAG_MAX; i++) {
		flags[i] = false;
	}

	force_vertex_shading = GLOBAL_GET("rendering/quality/shading/force_vertex_shading");

	diffuse_mode = DIFFUSE_BURLEY;
	specular_mode = SPECULAR_SCHLICK_GGX;

	async_mode = ASYNC_MODE_VISIBLE;

	for (int i = 0; i < FEATURE_MAX; i++) {
		features[i] = false;
	}

	current_key.key = 0;
	current_key.invalid_key = 1;
	is_initialized = true;
	_queue_shader_change();
}

SpatialMaterial::~SpatialMaterial() {
	material_mutex.lock();

	if (shader_map.has(current_key)) {
		shader_map[current_key].users--;
		if (shader_map[current_key].users == 0) {
			//deallocate shader, as it's no longer in use
			VS::get_singleton()->free(shader_map[current_key].shader);
			shader_map.erase(current_key);
		}

		VS::get_singleton()->material_set_shader(_get_material(), RID());
	}

	material_mutex.unlock();
}
