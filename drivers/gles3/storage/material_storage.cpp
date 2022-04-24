/*************************************************************************/
/*  material_storage.cpp                                                 */
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

#ifdef GLES3_ENABLED

#include "material_storage.h"
#include "config.h"
#include "texture_storage.h"

#include "drivers/gles3/rasterizer_canvas_gles3.h"

using namespace GLES3;

MaterialStorage *MaterialStorage::singleton = nullptr;

MaterialStorage *MaterialStorage::get_singleton() {
	return singleton;
}

MaterialStorage::MaterialStorage() {
	singleton = this;

	shaders.copy.initialize();
	shaders.copy_version = shaders.copy.version_create(); //TODO
	shaders.copy.version_bind_shader(shaders.copy_version, CopyShaderGLES3::MODE_COPY_SECTION);
	//shaders.cubemap_filter.init();
	//bool ggx_hq = GLOBAL_GET("rendering/quality/reflections/high_quality_ggx");
	//shaders.cubemap_filter.set_conditional(CubemapFilterShaderGLES3::LOW_QUALITY, !ggx_hq);
}

MaterialStorage::~MaterialStorage() {
	shaders.copy.version_free(shaders.copy_version);

	singleton = nullptr;
}

/* GLOBAL VARIABLE API */

void MaterialStorage::global_variable_add(const StringName &p_name, RS::GlobalVariableType p_type, const Variant &p_value) {
}

void MaterialStorage::global_variable_remove(const StringName &p_name) {
}

Vector<StringName> MaterialStorage::global_variable_get_list() const {
	return Vector<StringName>();
}

void MaterialStorage::global_variable_set(const StringName &p_name, const Variant &p_value) {
}

void MaterialStorage::global_variable_set_override(const StringName &p_name, const Variant &p_value) {
}

Variant MaterialStorage::global_variable_get(const StringName &p_name) const {
	return Variant();
}

RS::GlobalVariableType MaterialStorage::global_variable_get_type(const StringName &p_name) const {
	return RS::GLOBAL_VAR_TYPE_MAX;
}

void MaterialStorage::global_variables_load_settings(bool p_load_textures) {
}

void MaterialStorage::global_variables_clear() {
}

int32_t MaterialStorage::global_variables_instance_allocate(RID p_instance) {
	return 0;
}

void MaterialStorage::global_variables_instance_free(RID p_instance) {
}

void MaterialStorage::global_variables_instance_update(RID p_instance, int p_index, const Variant &p_value) {
}

/* SHADER API */

void MaterialStorage::_shader_make_dirty(GLES3::Shader *p_shader) {
	if (p_shader->dirty_list.in_list()) {
		return;
	}

	_shader_dirty_list.add(&p_shader->dirty_list);
}

RID MaterialStorage::shader_allocate() {
	GLES3::Shader *shader = memnew(GLES3::Shader);
	shader->mode = RS::SHADER_CANVAS_ITEM;
	//shader->shader = &scene->state.scene_shader;
	RID rid = shader_owner.make_rid(shader);
	_shader_make_dirty(shader);
	shader->self = rid;

	return rid;
}

void MaterialStorage::shader_initialize(RID p_rid) {
	// noop
}

//RID MaterialStorage::shader_create() {
//	GLES3::Shader *shader = memnew(GLES3::Shader);
//	shader->mode = RS::SHADER_SPATIAL;
//	shader->shader = &scene->state.scene_shader;
//	RID rid = shader_owner.make_rid(shader);
//	_shader_make_dirty(shader);
//	shader->self = rid;

//	return rid;
//}

void MaterialStorage::shader_free(RID p_rid) {
	GLES3::Shader *shader = shader_owner.get_or_null(p_rid);

	if (shader->shader && shader->version.is_valid()) {
		shader->shader->version_free(shader->version);
	}

	if (shader->dirty_list.in_list()) {
		_shader_dirty_list.remove(&shader->dirty_list);
	}

	while (shader->materials.first()) {
		GLES3::Material *m = shader->materials.first()->self();

		m->shader = nullptr;
		_material_make_dirty(m);

		shader->materials.remove(shader->materials.first());
	}

	shader_owner.free(p_rid);
	memdelete(shader);
}

void MaterialStorage::shader_set_code(RID p_shader, const String &p_code) {
	GLES3::Shader *shader = shader_owner.get_or_null(p_shader);
	ERR_FAIL_COND(!shader);

	shader->code = p_code;

	String mode_string = ShaderLanguage::get_shader_type(p_code);
	RS::ShaderMode mode;

	if (mode_string == "canvas_item") {
		mode = RS::SHADER_CANVAS_ITEM;
	} else if (mode_string == "particles") {
		mode = RS::SHADER_PARTICLES;
	} else if (mode_string == "sky") {
		mode = RS::SHADER_SKY;
	} else if (mode_string == "spatial") {
		mode = RS::SHADER_SPATIAL;
	} else {
		mode = RS::SHADER_MAX;
		ERR_PRINT("shader type " + mode_string + " not supported in OpenGL renderer");
	}

	if (shader->version.is_valid() && mode != shader->mode) {
		shader->shader->version_free(shader->version);
		shader->version = RID();
	}

	shader->mode = mode;

	// TODO handle all shader types
	if (mode == RS::SHADER_CANVAS_ITEM) {
		shader->shader = &RasterizerCanvasGLES3::get_singleton()->state.canvas_shader;
	} else if (mode == RS::SHADER_SPATIAL) {
		//shader->shader = &scene->state.scene_shader;
	} else if (mode == RS::SHADER_PARTICLES) {
	} else if (mode == RS::SHADER_SKY) {
	} else {
		return;
	}

	if (shader->version.is_null() && shader->shader) {
		shader->version = shader->shader->version_create();
	}

	_shader_make_dirty(shader);
}

String MaterialStorage::shader_get_code(RID p_shader) const {
	const GLES3::Shader *shader = shader_owner.get_or_null(p_shader);
	ERR_FAIL_COND_V(!shader, "");

	return shader->code;
}

void MaterialStorage::shader_get_param_list(RID p_shader, List<PropertyInfo> *p_param_list) const {
	GLES3::Shader *shader = shader_owner.get_or_null(p_shader);
	ERR_FAIL_COND(!shader);

	if (shader->dirty_list.in_list()) {
		_update_shader(shader);
	}

	Map<int, StringName> order;

	for (Map<StringName, ShaderLanguage::ShaderNode::Uniform>::Element *E = shader->uniforms.front(); E; E = E->next()) {
		if (E->get().texture_order >= 0) {
			order[E->get().texture_order + 100000] = E->key();
		} else {
			order[E->get().order] = E->key();
		}
	}

	for (Map<int, StringName>::Element *E = order.front(); E; E = E->next()) {
		PropertyInfo pi;
		ShaderLanguage::ShaderNode::Uniform &u = shader->uniforms[E->get()];

		pi.name = E->get();

		switch (u.type) {
			case ShaderLanguage::TYPE_VOID: {
				pi.type = Variant::NIL;
			} break;

			case ShaderLanguage::TYPE_BOOL: {
				pi.type = Variant::BOOL;
			} break;

			// bool vectors
			case ShaderLanguage::TYPE_BVEC2: {
				pi.type = Variant::INT;
				pi.hint = PROPERTY_HINT_FLAGS;
				pi.hint_string = "x,y";
			} break;
			case ShaderLanguage::TYPE_BVEC3: {
				pi.type = Variant::INT;
				pi.hint = PROPERTY_HINT_FLAGS;
				pi.hint_string = "x,y,z";
			} break;
			case ShaderLanguage::TYPE_BVEC4: {
				pi.type = Variant::INT;
				pi.hint = PROPERTY_HINT_FLAGS;
				pi.hint_string = "x,y,z,w";
			} break;

				// int stuff
			case ShaderLanguage::TYPE_UINT:
			case ShaderLanguage::TYPE_INT: {
				pi.type = Variant::INT;

				if (u.hint == ShaderLanguage::ShaderNode::Uniform::HINT_RANGE) {
					pi.hint = PROPERTY_HINT_RANGE;
					pi.hint_string = rtos(u.hint_range[0]) + "," + rtos(u.hint_range[1]) + "," + rtos(u.hint_range[2]);
				}
			} break;

			case ShaderLanguage::TYPE_IVEC2:
			case ShaderLanguage::TYPE_UVEC2:
			case ShaderLanguage::TYPE_IVEC3:
			case ShaderLanguage::TYPE_UVEC3:
			case ShaderLanguage::TYPE_IVEC4:
			case ShaderLanguage::TYPE_UVEC4: {
				// not sure what this should be in godot 4
				//				pi.type = Variant::POOL_INT_ARRAY;
				pi.type = Variant::PACKED_INT32_ARRAY;
			} break;

			case ShaderLanguage::TYPE_FLOAT: {
				pi.type = Variant::FLOAT;
				if (u.hint == ShaderLanguage::ShaderNode::Uniform::HINT_RANGE) {
					pi.hint = PROPERTY_HINT_RANGE;
					pi.hint_string = rtos(u.hint_range[0]) + "," + rtos(u.hint_range[1]) + "," + rtos(u.hint_range[2]);
				}
			} break;

			case ShaderLanguage::TYPE_VEC2: {
				pi.type = Variant::VECTOR2;
			} break;
			case ShaderLanguage::TYPE_VEC3: {
				pi.type = Variant::VECTOR3;
			} break;

			case ShaderLanguage::TYPE_VEC4: {
				if (u.hint == ShaderLanguage::ShaderNode::Uniform::HINT_COLOR) {
					pi.type = Variant::COLOR;
				} else {
					pi.type = Variant::QUATERNION;
				}
			} break;

			case ShaderLanguage::TYPE_MAT2: {
				pi.type = Variant::TRANSFORM2D;
			} break;

			case ShaderLanguage::TYPE_MAT3: {
				pi.type = Variant::BASIS;
			} break;

			case ShaderLanguage::TYPE_MAT4: {
				pi.type = Variant::TRANSFORM3D;
			} break;

			case ShaderLanguage::TYPE_SAMPLER2D:
				//			case ShaderLanguage::TYPE_SAMPLEREXT:
			case ShaderLanguage::TYPE_ISAMPLER2D:
			case ShaderLanguage::TYPE_USAMPLER2D: {
				pi.type = Variant::OBJECT;
				pi.hint = PROPERTY_HINT_RESOURCE_TYPE;
				pi.hint_string = "Texture";
			} break;

			case ShaderLanguage::TYPE_SAMPLERCUBE: {
				pi.type = Variant::OBJECT;
				pi.hint = PROPERTY_HINT_RESOURCE_TYPE;
				pi.hint_string = "CubeMap";
			} break;

			case ShaderLanguage::TYPE_SAMPLER2DARRAY:
			case ShaderLanguage::TYPE_ISAMPLER2DARRAY:
			case ShaderLanguage::TYPE_USAMPLER2DARRAY:
			case ShaderLanguage::TYPE_SAMPLER3D:
			case ShaderLanguage::TYPE_ISAMPLER3D:
			case ShaderLanguage::TYPE_USAMPLER3D: {
				// Not implemented in OpenGL
			} break;
				// new for godot 4
			case ShaderLanguage::TYPE_SAMPLERCUBEARRAY:
			case ShaderLanguage::TYPE_STRUCT:
			case ShaderLanguage::TYPE_MAX: {
			} break;
		}

		p_param_list->push_back(pi);
	}
}

void MaterialStorage::shader_set_default_texture_param(RID p_shader, const StringName &p_name, RID p_texture, int p_index) {
	GLES3::Shader *shader = shader_owner.get_or_null(p_shader);
	ERR_FAIL_COND(!shader);
	ERR_FAIL_COND(p_texture.is_valid() && !TextureStorage::get_singleton()->owns_texture(p_texture));

	if (!p_texture.is_valid()) {
		if (shader->default_textures.has(p_name) && shader->default_textures[p_name].has(p_index)) {
			shader->default_textures[p_name].erase(p_index);

			if (shader->default_textures[p_name].is_empty()) {
				shader->default_textures.erase(p_name);
			}
		}
	} else {
		if (!shader->default_textures.has(p_name)) {
			shader->default_textures[p_name] = Map<int, RID>();
		}
		shader->default_textures[p_name][p_index] = p_texture;
	}

	_shader_make_dirty(shader);
}

RID MaterialStorage::shader_get_default_texture_param(RID p_shader, const StringName &p_name, int p_index) const {
	const GLES3::Shader *shader = shader_owner.get_or_null(p_shader);
	ERR_FAIL_COND_V(!shader, RID());

	if (shader->default_textures.has(p_name) && shader->default_textures[p_name].has(p_index)) {
		return shader->default_textures[p_name][p_index];
	}

	return RID();
}

void MaterialStorage::_update_shader(GLES3::Shader *p_shader) const {
	_shader_dirty_list.remove(&p_shader->dirty_list);

	p_shader->valid = false;

	p_shader->uniforms.clear();

	if (p_shader->code.is_empty()) {
		return; //just invalid, but no error
	}

	ShaderCompiler::GeneratedCode gen_code;
	ShaderCompiler::IdentifierActions *actions = nullptr;

	switch (p_shader->mode) {
		case RS::SHADER_CANVAS_ITEM: {
			p_shader->canvas_item.light_mode = GLES3::Shader::CanvasItem::LIGHT_MODE_NORMAL;
			p_shader->canvas_item.blend_mode = GLES3::Shader::CanvasItem::BLEND_MODE_MIX;

			p_shader->canvas_item.uses_screen_texture = false;
			p_shader->canvas_item.uses_screen_uv = false;
			p_shader->canvas_item.uses_time = false;
			p_shader->canvas_item.uses_modulate = false;
			p_shader->canvas_item.uses_color = false;
			p_shader->canvas_item.uses_vertex = false;

			p_shader->canvas_item.uses_model_matrix = false;
			p_shader->canvas_item.uses_extra_matrix = false;
			p_shader->canvas_item.uses_projection_matrix = false;
			p_shader->canvas_item.uses_instance_custom = false;

			shaders.actions_canvas.render_mode_values["blend_add"] = Pair<int *, int>(&p_shader->canvas_item.blend_mode, GLES3::Shader::CanvasItem::BLEND_MODE_ADD);
			shaders.actions_canvas.render_mode_values["blend_mix"] = Pair<int *, int>(&p_shader->canvas_item.blend_mode, GLES3::Shader::CanvasItem::BLEND_MODE_MIX);
			shaders.actions_canvas.render_mode_values["blend_sub"] = Pair<int *, int>(&p_shader->canvas_item.blend_mode, GLES3::Shader::CanvasItem::BLEND_MODE_SUB);
			shaders.actions_canvas.render_mode_values["blend_mul"] = Pair<int *, int>(&p_shader->canvas_item.blend_mode, GLES3::Shader::CanvasItem::BLEND_MODE_MUL);
			shaders.actions_canvas.render_mode_values["blend_premul_alpha"] = Pair<int *, int>(&p_shader->canvas_item.blend_mode, GLES3::Shader::CanvasItem::BLEND_MODE_PMALPHA);

			shaders.actions_canvas.render_mode_values["unshaded"] = Pair<int *, int>(&p_shader->canvas_item.light_mode, GLES3::Shader::CanvasItem::LIGHT_MODE_UNSHADED);
			shaders.actions_canvas.render_mode_values["light_only"] = Pair<int *, int>(&p_shader->canvas_item.light_mode, GLES3::Shader::CanvasItem::LIGHT_MODE_LIGHT_ONLY);

			shaders.actions_canvas.usage_flag_pointers["SCREEN_UV"] = &p_shader->canvas_item.uses_screen_uv;
			shaders.actions_canvas.usage_flag_pointers["SCREEN_PIXEL_SIZE"] = &p_shader->canvas_item.uses_screen_uv;
			shaders.actions_canvas.usage_flag_pointers["SCREEN_TEXTURE"] = &p_shader->canvas_item.uses_screen_texture;
			shaders.actions_canvas.usage_flag_pointers["TIME"] = &p_shader->canvas_item.uses_time;
			shaders.actions_canvas.usage_flag_pointers["MODULATE"] = &p_shader->canvas_item.uses_modulate;
			shaders.actions_canvas.usage_flag_pointers["COLOR"] = &p_shader->canvas_item.uses_color;

			shaders.actions_canvas.usage_flag_pointers["VERTEX"] = &p_shader->canvas_item.uses_vertex;

			shaders.actions_canvas.usage_flag_pointers["MODEL_MATRIX"] = &p_shader->canvas_item.uses_model_matrix;
			shaders.actions_canvas.usage_flag_pointers["EXTRA_MATRIX"] = &p_shader->canvas_item.uses_extra_matrix;
			shaders.actions_canvas.usage_flag_pointers["PROJECTION_MATRIX"] = &p_shader->canvas_item.uses_projection_matrix;
			shaders.actions_canvas.usage_flag_pointers["INSTANCE_CUSTOM"] = &p_shader->canvas_item.uses_instance_custom;

			actions = &shaders.actions_canvas;
			actions->uniforms = &p_shader->uniforms;
		} break;

		case RS::SHADER_SPATIAL: {
			// TODO remove once 3D is added back
			return;
			p_shader->spatial.blend_mode = GLES3::Shader::Spatial::BLEND_MODE_MIX;
			p_shader->spatial.depth_draw_mode = GLES3::Shader::Spatial::DEPTH_DRAW_OPAQUE;
			p_shader->spatial.cull_mode = GLES3::Shader::Spatial::CULL_MODE_BACK;
			p_shader->spatial.uses_alpha = false;
			p_shader->spatial.uses_alpha_scissor = false;
			p_shader->spatial.uses_discard = false;
			p_shader->spatial.unshaded = false;
			p_shader->spatial.no_depth_test = false;
			p_shader->spatial.uses_sss = false;
			p_shader->spatial.uses_time = false;
			p_shader->spatial.uses_vertex_lighting = false;
			p_shader->spatial.uses_screen_texture = false;
			p_shader->spatial.uses_depth_texture = false;
			p_shader->spatial.uses_vertex = false;
			p_shader->spatial.uses_tangent = false;
			p_shader->spatial.uses_ensure_correct_normals = false;
			p_shader->spatial.writes_modelview_or_projection = false;
			p_shader->spatial.uses_world_coordinates = false;

			shaders.actions_scene.render_mode_values["blend_add"] = Pair<int *, int>(&p_shader->spatial.blend_mode, GLES3::Shader::Spatial::BLEND_MODE_ADD);
			shaders.actions_scene.render_mode_values["blend_mix"] = Pair<int *, int>(&p_shader->spatial.blend_mode, GLES3::Shader::Spatial::BLEND_MODE_MIX);
			shaders.actions_scene.render_mode_values["blend_sub"] = Pair<int *, int>(&p_shader->spatial.blend_mode, GLES3::Shader::Spatial::BLEND_MODE_SUB);
			shaders.actions_scene.render_mode_values["blend_mul"] = Pair<int *, int>(&p_shader->spatial.blend_mode, GLES3::Shader::Spatial::BLEND_MODE_MUL);

			shaders.actions_scene.render_mode_values["depth_draw_opaque"] = Pair<int *, int>(&p_shader->spatial.depth_draw_mode, GLES3::Shader::Spatial::DEPTH_DRAW_OPAQUE);
			shaders.actions_scene.render_mode_values["depth_draw_always"] = Pair<int *, int>(&p_shader->spatial.depth_draw_mode, GLES3::Shader::Spatial::DEPTH_DRAW_ALWAYS);
			shaders.actions_scene.render_mode_values["depth_draw_never"] = Pair<int *, int>(&p_shader->spatial.depth_draw_mode, GLES3::Shader::Spatial::DEPTH_DRAW_NEVER);
			shaders.actions_scene.render_mode_values["depth_draw_alpha_prepass"] = Pair<int *, int>(&p_shader->spatial.depth_draw_mode, GLES3::Shader::Spatial::DEPTH_DRAW_ALPHA_PREPASS);

			shaders.actions_scene.render_mode_values["cull_front"] = Pair<int *, int>(&p_shader->spatial.cull_mode, GLES3::Shader::Spatial::CULL_MODE_FRONT);
			shaders.actions_scene.render_mode_values["cull_back"] = Pair<int *, int>(&p_shader->spatial.cull_mode, GLES3::Shader::Spatial::CULL_MODE_BACK);
			shaders.actions_scene.render_mode_values["cull_disabled"] = Pair<int *, int>(&p_shader->spatial.cull_mode, GLES3::Shader::Spatial::CULL_MODE_DISABLED);

			shaders.actions_scene.render_mode_flags["unshaded"] = &p_shader->spatial.unshaded;
			shaders.actions_scene.render_mode_flags["depth_test_disable"] = &p_shader->spatial.no_depth_test;

			shaders.actions_scene.render_mode_flags["vertex_lighting"] = &p_shader->spatial.uses_vertex_lighting;

			shaders.actions_scene.render_mode_flags["world_vertex_coords"] = &p_shader->spatial.uses_world_coordinates;

			shaders.actions_scene.render_mode_flags["ensure_correct_normals"] = &p_shader->spatial.uses_ensure_correct_normals;

			shaders.actions_scene.usage_flag_pointers["ALPHA"] = &p_shader->spatial.uses_alpha;
			shaders.actions_scene.usage_flag_pointers["ALPHA_SCISSOR"] = &p_shader->spatial.uses_alpha_scissor;

			shaders.actions_scene.usage_flag_pointers["SSS_STRENGTH"] = &p_shader->spatial.uses_sss;
			shaders.actions_scene.usage_flag_pointers["DISCARD"] = &p_shader->spatial.uses_discard;
			shaders.actions_scene.usage_flag_pointers["SCREEN_TEXTURE"] = &p_shader->spatial.uses_screen_texture;
			shaders.actions_scene.usage_flag_pointers["DEPTH_TEXTURE"] = &p_shader->spatial.uses_depth_texture;
			shaders.actions_scene.usage_flag_pointers["TIME"] = &p_shader->spatial.uses_time;

			// Use of any of these BUILTINS indicate the need for transformed tangents.
			// This is needed to know when to transform tangents in software skinning.
			shaders.actions_scene.usage_flag_pointers["TANGENT"] = &p_shader->spatial.uses_tangent;
			shaders.actions_scene.usage_flag_pointers["NORMALMAP"] = &p_shader->spatial.uses_tangent;

			shaders.actions_scene.write_flag_pointers["MODELVIEW_MATRIX"] = &p_shader->spatial.writes_modelview_or_projection;
			shaders.actions_scene.write_flag_pointers["PROJECTION_MATRIX"] = &p_shader->spatial.writes_modelview_or_projection;
			shaders.actions_scene.write_flag_pointers["VERTEX"] = &p_shader->spatial.uses_vertex;

			actions = &shaders.actions_scene;
			actions->uniforms = &p_shader->uniforms;
		} break;

		default: {
			return;
		} break;
	}

	Error err = shaders.compiler.compile(p_shader->mode, p_shader->code, actions, p_shader->path, gen_code);
	if (err != OK) {
		return;
	}

	Vector<StringName> texture_uniform_names;
	for (int i = 0; i < gen_code.texture_uniforms.size(); i++) {
		texture_uniform_names.push_back(gen_code.texture_uniforms[i].name);
	}

	p_shader->shader->version_set_code(p_shader->version, gen_code.code, gen_code.uniforms, gen_code.stage_globals[ShaderCompiler::STAGE_VERTEX], gen_code.stage_globals[ShaderCompiler::STAGE_FRAGMENT], gen_code.defines, texture_uniform_names);

	p_shader->texture_uniforms = gen_code.texture_uniforms;

	p_shader->uses_vertex_time = gen_code.uses_vertex_time;
	p_shader->uses_fragment_time = gen_code.uses_fragment_time;

	for (SelfList<GLES3::Material> *E = p_shader->materials.first(); E; E = E->next()) {
		_material_make_dirty(E->self());
	}

	p_shader->valid = true;
}

void MaterialStorage::update_dirty_shaders() {
	while (_shader_dirty_list.first()) {
		_update_shader(_shader_dirty_list.first()->self());
	}
}

/* MATERIAL API */

void MaterialStorage::_material_make_dirty(GLES3::Material *p_material) const {
	if (p_material->dirty_list.in_list()) {
		return;
	}

	_material_dirty_list.add(&p_material->dirty_list);
}

void MaterialStorage::_update_material(GLES3::Material *p_material) {
	if (p_material->dirty_list.in_list()) {
		_material_dirty_list.remove(&p_material->dirty_list);
	}

	if (p_material->shader && p_material->shader->dirty_list.in_list()) {
		_update_shader(p_material->shader);
	}

	if (p_material->shader && !p_material->shader->valid) {
		return;
	}

	{
		if (p_material->shader && p_material->shader->mode == RS::SHADER_SPATIAL) {
			bool can_cast_shadow = false;
			bool is_animated = false;

			if (p_material->shader->spatial.blend_mode == GLES3::Shader::Spatial::BLEND_MODE_MIX &&

					(!p_material->shader->spatial.uses_alpha || p_material->shader->spatial.depth_draw_mode == Shader::Spatial::DEPTH_DRAW_ALPHA_PREPASS)) {
				can_cast_shadow = true;
			}

			if (p_material->shader->spatial.uses_discard && p_material->shader->uses_fragment_time) {
				is_animated = true;
			}

			if (p_material->shader->spatial.uses_vertex && p_material->shader->uses_vertex_time) {
				is_animated = true;
			}

			if (can_cast_shadow != p_material->can_cast_shadow_cache || is_animated != p_material->is_animated_cache) {
				p_material->can_cast_shadow_cache = can_cast_shadow;
				p_material->is_animated_cache = is_animated;

				/*
				for (Map<Geometry *, int>::Element *E = p_material->geometry_owners.front(); E; E = E->next()) {
					E->key()->material_changed_notify();
				}

				for (Map<InstanceBaseDependency *, int>::Element *E = p_material->instance_owners.front(); E; E = E->next()) {
					E->key()->base_changed(false, true);
				}
				*/
			}
		}
	}

	// uniforms and other things will be set in the use_material method in ShaderGLES3

	if (p_material->shader && p_material->shader->texture_uniforms.size() > 0) {
		p_material->textures.resize(p_material->shader->texture_uniforms.size());

		for (Map<StringName, ShaderLanguage::ShaderNode::Uniform>::Element *E = p_material->shader->uniforms.front(); E; E = E->next()) {
			if (E->get().texture_order < 0) {
				continue; // not a texture, does not go here
			}

			RID texture;

			Map<StringName, Variant>::Element *V = p_material->params.find(E->key());

			if (V) {
				texture = V->get();
			}

			if (!texture.is_valid()) {
				Map<StringName, Map<int, RID>>::Element *W = p_material->shader->default_textures.find(E->key());

				// TODO: make texture uniform array properly works with GLES3
				if (W && W->get().has(0)) {
					texture = W->get()[0];
				}
			}

			p_material->textures.write[E->get().texture_order] = Pair<StringName, RID>(E->key(), texture);
		}
	} else {
		p_material->textures.clear();
	}
}

RID MaterialStorage::material_allocate() {
	GLES3::Material *material = memnew(GLES3::Material);
	return material_owner.make_rid(material);
}

void MaterialStorage::material_initialize(RID p_rid) {
}

//RID MaterialStorage::material_create() {
//	Material *material = memnew(Material);

//	return material_owner.make_rid(material);
//}

void MaterialStorage::material_free(RID p_rid) {
	GLES3::Material *m = material_owner.get_or_null(p_rid);

	if (m->shader) {
		m->shader->materials.remove(&m->list);
	}

	/*
	for (Map<Geometry *, int>::Element *E = m->geometry_owners.front(); E; E = E->next()) {
		Geometry *g = E->key();
		g->material = RID();
	}

	for (Map<InstanceBaseDependency *, int>::Element *E = m->instance_owners.front(); E; E = E->next()) {
		InstanceBaseDependency *ins = E->key();

		if (ins->material_override == p_rid) {
			ins->material_override = RID();
		}

		for (int i = 0; i < ins->materials.size(); i++) {
			if (ins->materials[i] == p_rid) {
				ins->materials.write[i] = RID();
			}
		}
	}
*/

	material_owner.free(p_rid);
	memdelete(m);
}

void MaterialStorage::material_set_shader(RID p_material, RID p_shader) {
	GLES3::Material *material = material_owner.get_or_null(p_material);
	ERR_FAIL_COND(!material);

	GLES3::Shader *shader = get_shader(p_shader);

	if (material->shader) {
		// if a shader is present, remove the old shader
		material->shader->materials.remove(&material->list);
	}

	material->shader = shader;

	if (shader) {
		shader->materials.add(&material->list);
	}

	_material_make_dirty(material);
}

void MaterialStorage::material_set_param(RID p_material, const StringName &p_param, const Variant &p_value) {
	GLES3::Material *material = material_owner.get_or_null(p_material);
	ERR_FAIL_COND(!material);

	if (p_value.get_type() == Variant::NIL) {
		material->params.erase(p_param);
	} else {
		material->params[p_param] = p_value;
	}

	_material_make_dirty(material);
}

Variant MaterialStorage::material_get_param(RID p_material, const StringName &p_param) const {
	const GLES3::Material *material = material_owner.get_or_null(p_material);
	ERR_FAIL_COND_V(!material, RID());

	if (material->params.has(p_param)) {
		return material->params[p_param];
	}

	return material_get_param_default(p_material, p_param);
}

void MaterialStorage::material_set_next_pass(RID p_material, RID p_next_material) {
	GLES3::Material *material = material_owner.get_or_null(p_material);
	ERR_FAIL_COND(!material);

	material->next_pass = p_next_material;
}

void MaterialStorage::material_set_render_priority(RID p_material, int priority) {
	ERR_FAIL_COND(priority < RS::MATERIAL_RENDER_PRIORITY_MIN);
	ERR_FAIL_COND(priority > RS::MATERIAL_RENDER_PRIORITY_MAX);

	GLES3::Material *material = material_owner.get_or_null(p_material);
	ERR_FAIL_COND(!material);

	material->render_priority = priority;
}

bool MaterialStorage::material_is_animated(RID p_material) {
	GLES3::Material *material = material_owner.get_or_null(p_material);
	ERR_FAIL_COND_V(!material, false);
	if (material->dirty_list.in_list()) {
		_update_material(material);
	}

	bool animated = material->is_animated_cache;
	if (!animated && material->next_pass.is_valid()) {
		animated = material_is_animated(material->next_pass);
	}
	return animated;
}

bool MaterialStorage::material_casts_shadows(RID p_material) {
	GLES3::Material *material = material_owner.get_or_null(p_material);
	ERR_FAIL_COND_V(!material, false);
	if (material->dirty_list.in_list()) {
		_update_material(material);
	}

	bool casts_shadows = material->can_cast_shadow_cache;

	if (!casts_shadows && material->next_pass.is_valid()) {
		casts_shadows = material_casts_shadows(material->next_pass);
	}

	return casts_shadows;
}

Variant MaterialStorage::material_get_param_default(RID p_material, const StringName &p_param) const {
	const GLES3::Material *material = material_owner.get_or_null(p_material);
	ERR_FAIL_COND_V(!material, Variant());

	if (material->shader) {
		if (material->shader->uniforms.has(p_param)) {
			ShaderLanguage::ShaderNode::Uniform uniform = material->shader->uniforms[p_param];
			Vector<ShaderLanguage::ConstantNode::Value> default_value = uniform.default_value;
			return ShaderLanguage::constant_value_to_variant(default_value, uniform.type, uniform.hint);
		}
	}
	return Variant();
}

void MaterialStorage::update_dirty_materials() {
	while (_material_dirty_list.first()) {
		GLES3::Material *material = _material_dirty_list.first()->self();
		_update_material(material);
	}
}

/* are these still used? */
RID MaterialStorage::material_get_shader(RID p_material) const {
	const GLES3::Material *material = material_owner.get_or_null(p_material);
	ERR_FAIL_COND_V(!material, RID());

	if (material->shader) {
		return material->shader->self;
	}

	return RID();
}

void MaterialStorage::material_set_line_width(RID p_material, float p_width) {
	GLES3::Material *material = material_owner.get_or_null(p_material);
	ERR_FAIL_COND(!material);

	material->line_width = p_width;
}

bool MaterialStorage::material_uses_tangents(RID p_material) {
	GLES3::Material *material = material_owner.get_or_null(p_material);
	ERR_FAIL_COND_V(!material, false);

	if (!material->shader) {
		return false;
	}

	if (material->shader->dirty_list.in_list()) {
		_update_shader(material->shader);
	}

	return material->shader->spatial.uses_tangent;
}

bool MaterialStorage::material_uses_ensure_correct_normals(RID p_material) {
	GLES3::Material *material = material_owner.get_or_null(p_material);
	ERR_FAIL_COND_V(!material, false);

	if (!material->shader) {
		return false;
	}

	if (material->shader->dirty_list.in_list()) {
		_update_shader(material->shader);
	}

	return material->shader->spatial.uses_ensure_correct_normals;
}

void MaterialStorage::material_add_instance_owner(RID p_material, RendererStorage::DependencyTracker *p_instance) {
	/*
	GLES3::Material *material = material_owner.get_or_null(p_material);
	ERR_FAIL_COND(!material);

	Map<InstanceBaseDependency *, int>::Element *E = material->instance_owners.find(p_instance);
	if (E) {
		E->get()++;
	} else {
		material->instance_owners[p_instance] = 1;
	}
*/
}

void MaterialStorage::material_remove_instance_owner(RID p_material, RendererStorage::DependencyTracker *p_instance) {
	/*
	GLES3::Material *material = material_owner.get_or_null(p_material);
	ERR_FAIL_COND(!material);

	Map<InstanceBaseDependency *, int>::Element *E = material->instance_owners.find(p_instance);
	ERR_FAIL_COND(!E);

	E->get()--;

	if (E->get() == 0) {
		material->instance_owners.erase(E);
	}
*/
}

/*
void MaterialStorage::_material_add_geometry(RID p_material, Geometry *p_geometry) {
	GLES3::Material *material = material_owner.get_or_null(p_material);
	ERR_FAIL_COND(!material);

	Map<Geometry *, int>::Element *I = material->geometry_owners.find(p_geometry);

	if (I) {
		I->get()++;
	} else {
		material->geometry_owners[p_geometry] = 1;
	}
}

void MaterialStorage::_material_remove_geometry(RID p_material, Geometry *p_geometry) {
	GLES3::Material *material = material_owner.get_or_null(p_material);
	ERR_FAIL_COND(!material);

	Map<Geometry *, int>::Element *I = material->geometry_owners.find(p_geometry);
	ERR_FAIL_COND(!I);

	I->get()--;

	if (I->get() == 0) {
		material->geometry_owners.erase(I);
	}
}
*/

#endif // !GLES3_ENABLED
