/*************************************************************************/
/*  material_storage.h                                                   */
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

#ifndef MATERIAL_STORAGE_GLES3_H
#define MATERIAL_STORAGE_GLES3_H

#ifdef GLES3_ENABLED

#include "core/templates/local_vector.h"
#include "core/templates/rid_owner.h"
#include "core/templates/self_list.h"
#include "servers/rendering/renderer_compositor.h"
#include "servers/rendering/renderer_storage.h"
#include "servers/rendering/shader_compiler.h"
#include "servers/rendering/shader_language.h"
#include "servers/rendering/storage/material_storage.h"

#include "drivers/gles3/shaders/copy.glsl.gen.h"

namespace GLES3 {

/* SHADER Structs */

struct Shaders {
	ShaderCompiler compiler;

	CopyShaderGLES3 copy;
	RID copy_version;
	//CubemapFilterShaderGLES3 cubemap_filter;

	ShaderCompiler::IdentifierActions actions_canvas;
	ShaderCompiler::IdentifierActions actions_scene;
	ShaderCompiler::IdentifierActions actions_particles;
};

struct Material;

struct Shader {
	RID self;

	RS::ShaderMode mode;
	ShaderGLES3 *shader = nullptr;
	String code;
	SelfList<Material>::List materials;

	Map<StringName, ShaderLanguage::ShaderNode::Uniform> uniforms;

	RID version;

	SelfList<Shader> dirty_list;

	Map<StringName, Map<int, RID>> default_textures;

	Vector<ShaderCompiler::GeneratedCode::Texture> texture_uniforms;

	bool valid;

	String path;

	uint32_t index;
	uint64_t last_pass;

	struct CanvasItem {
		enum BlendMode {
			BLEND_MODE_MIX,
			BLEND_MODE_ADD,
			BLEND_MODE_SUB,
			BLEND_MODE_MUL,
			BLEND_MODE_PMALPHA,
		};

		int blend_mode;

		enum LightMode {
			LIGHT_MODE_NORMAL,
			LIGHT_MODE_UNSHADED,
			LIGHT_MODE_LIGHT_ONLY
		};

		int light_mode;

		bool uses_screen_texture;
		bool uses_screen_uv;
		bool uses_time;
		bool uses_modulate;
		bool uses_color;
		bool uses_vertex;

		// all these should disable item joining if used in a custom shader
		bool uses_model_matrix;
		bool uses_extra_matrix;
		bool uses_projection_matrix;
		bool uses_instance_custom;

	} canvas_item;

	struct Spatial {
		enum BlendMode {
			BLEND_MODE_MIX,
			BLEND_MODE_ADD,
			BLEND_MODE_SUB,
			BLEND_MODE_MUL,
		};

		int blend_mode;

		enum DepthDrawMode {
			DEPTH_DRAW_OPAQUE,
			DEPTH_DRAW_ALWAYS,
			DEPTH_DRAW_NEVER,
			DEPTH_DRAW_ALPHA_PREPASS,
		};

		int depth_draw_mode;

		enum CullMode {
			CULL_MODE_FRONT,
			CULL_MODE_BACK,
			CULL_MODE_DISABLED,
		};

		int cull_mode;

		bool uses_alpha;
		bool uses_alpha_scissor;
		bool unshaded;
		bool no_depth_test;
		bool uses_vertex;
		bool uses_discard;
		bool uses_sss;
		bool uses_screen_texture;
		bool uses_depth_texture;
		bool uses_time;
		bool uses_tangent;
		bool uses_ensure_correct_normals;
		bool writes_modelview_or_projection;
		bool uses_vertex_lighting;
		bool uses_world_coordinates;

	} spatial;

	struct Particles {
	} particles;

	bool uses_vertex_time;
	bool uses_fragment_time;

	Shader() :
			dirty_list(this) {
		shader = nullptr;
		valid = false;
		version = RID();
		last_pass = 0;
	}
};

/* MATERIAL Structs */

struct Material {
	RID self;
	Shader *shader = nullptr;
	Map<StringName, Variant> params;
	SelfList<Material> list;
	SelfList<Material> dirty_list;
	Vector<Pair<StringName, RID>> textures;
	float line_width;
	int render_priority;

	RID next_pass;

	uint32_t index;
	uint64_t last_pass;

	//		Map<Geometry *, int> geometry_owners;
	//		Map<InstanceBaseDependency *, int> instance_owners;

	bool can_cast_shadow_cache;
	bool is_animated_cache;

	Material() :
			list(this),
			dirty_list(this) {
		can_cast_shadow_cache = false;
		is_animated_cache = false;
		shader = nullptr;
		line_width = 1.0;
		last_pass = 0;
		render_priority = 0;
	}
};

class MaterialStorage : public RendererMaterialStorage {
private:
	static MaterialStorage *singleton;

	/* SHADER API */

	mutable Shaders shaders;

	mutable RID_PtrOwner<Shader> shader_owner;
	mutable SelfList<Shader>::List _shader_dirty_list;

	/* MATERIAL API */

	mutable SelfList<Material>::List _material_dirty_list;
	mutable RID_PtrOwner<Material> material_owner;

public:
	static MaterialStorage *get_singleton();

	MaterialStorage();
	virtual ~MaterialStorage();

	/* GLOBAL VARIABLE API */

	virtual void global_variable_add(const StringName &p_name, RS::GlobalVariableType p_type, const Variant &p_value) override;
	virtual void global_variable_remove(const StringName &p_name) override;
	virtual Vector<StringName> global_variable_get_list() const override;

	virtual void global_variable_set(const StringName &p_name, const Variant &p_value) override;
	virtual void global_variable_set_override(const StringName &p_name, const Variant &p_value) override;
	virtual Variant global_variable_get(const StringName &p_name) const override;
	virtual RS::GlobalVariableType global_variable_get_type(const StringName &p_name) const override;

	virtual void global_variables_load_settings(bool p_load_textures = true) override;
	virtual void global_variables_clear() override;

	virtual int32_t global_variables_instance_allocate(RID p_instance) override;
	virtual void global_variables_instance_free(RID p_instance) override;
	virtual void global_variables_instance_update(RID p_instance, int p_index, const Variant &p_value) override;

	/* SHADER API */

	Shader *get_shader(RID p_rid) { return shader_owner.get_or_null(p_rid); };
	bool owns_shader(RID p_rid) { return shader_owner.owns(p_rid); };

	void _shader_make_dirty(Shader *p_shader);

	virtual RID shader_allocate() override;
	virtual void shader_initialize(RID p_rid) override;
	virtual void shader_free(RID p_rid) override;

	//RID shader_create() override;

	virtual void shader_set_code(RID p_shader, const String &p_code) override;
	virtual String shader_get_code(RID p_shader) const override;
	virtual void shader_get_param_list(RID p_shader, List<PropertyInfo> *p_param_list) const override;

	virtual void shader_set_default_texture_param(RID p_shader, const StringName &p_name, RID p_texture, int p_index) override;
	virtual RID shader_get_default_texture_param(RID p_shader, const StringName &p_name, int p_index) const override;

	virtual RS::ShaderNativeSourceCode shader_get_native_source_code(RID p_shader) const override { return RS::ShaderNativeSourceCode(); };

	void _update_shader(Shader *p_shader) const;
	void update_dirty_shaders();

	// new
	Variant shader_get_param_default(RID p_material, const StringName &p_param) const override { return Variant(); }

	/* MATERIAL API */

	Material *get_material(RID p_rid) { return material_owner.get_or_null(p_rid); };
	bool owns_material(RID p_rid) { return material_owner.owns(p_rid); };

	void _material_make_dirty(Material *p_material) const;

	//	void _material_add_geometry(RID p_material, Geometry *p_geometry);
	//	void _material_remove_geometry(RID p_material, Geometry *p_geometry);

	void _update_material(Material *p_material);

	// new
	virtual void material_get_instance_shader_parameters(RID p_material, List<InstanceShaderParam> *r_parameters) override {}
	virtual void material_update_dependency(RID p_material, RendererStorage::DependencyTracker *p_instance) override {}

	// old
	virtual RID material_allocate() override;
	virtual void material_initialize(RID p_rid) override;

	virtual void material_free(RID p_rid) override;

	//RID material_create() override;

	virtual void material_set_shader(RID p_material, RID p_shader) override;
	virtual RID material_get_shader(RID p_material) const;

	virtual void material_set_param(RID p_material, const StringName &p_param, const Variant &p_value) override;
	virtual Variant material_get_param(RID p_material, const StringName &p_param) const override;
	virtual Variant material_get_param_default(RID p_material, const StringName &p_param) const;

	void material_set_line_width(RID p_material, float p_width);
	virtual void material_set_next_pass(RID p_material, RID p_next_material) override;

	virtual bool material_is_animated(RID p_material) override;
	virtual bool material_casts_shadows(RID p_material) override;
	bool material_uses_tangents(RID p_material);
	bool material_uses_ensure_correct_normals(RID p_material);

	void material_add_instance_owner(RID p_material, RendererStorage::DependencyTracker *p_instance);
	void material_remove_instance_owner(RID p_material, RendererStorage::DependencyTracker *p_instance);

	void material_set_render_priority(RID p_material, int priority) override;

	void update_dirty_materials();
};

} // namespace GLES3

#endif // GLES3_ENABLED

#endif // !MATERIAL_STORAGE_GLES3_H
