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

#ifndef MATERIAL_STORAGE_RD_H
#define MATERIAL_STORAGE_RD_H

#include "core/templates/local_vector.h"
#include "core/templates/rid_owner.h"
#include "core/templates/self_list.h"
#include "servers/rendering/shader_compiler.h"
#include "servers/rendering/shader_language.h"
#include "servers/rendering/storage/material_storage.h"

namespace RendererRD {

class MaterialStorage;

/* SHADER Structs */

enum ShaderType {
	SHADER_TYPE_2D,
	SHADER_TYPE_3D,
	SHADER_TYPE_PARTICLES,
	SHADER_TYPE_SKY,
	SHADER_TYPE_FOG,
	SHADER_TYPE_MAX
};

struct ShaderData {
	virtual void set_code(const String &p_Code) = 0;
	virtual void set_default_texture_param(const StringName &p_name, RID p_texture, int p_index) = 0;
	virtual void get_param_list(List<PropertyInfo> *p_param_list) const = 0;

	virtual void get_instance_param_list(List<RendererMaterialStorage::InstanceShaderParam> *p_param_list) const = 0;
	virtual bool is_param_texture(const StringName &p_param) const = 0;
	virtual bool is_animated() const = 0;
	virtual bool casts_shadows() const = 0;
	virtual Variant get_default_parameter(const StringName &p_parameter) const = 0;
	virtual RS::ShaderNativeSourceCode get_native_source_code() const { return RS::ShaderNativeSourceCode(); }

	virtual ~ShaderData() {}
};

typedef ShaderData *(*ShaderDataRequestFunction)();

struct Material;

struct Shader {
	ShaderData *data = nullptr;
	String code;
	ShaderType type;
	HashMap<StringName, HashMap<int, RID>> default_texture_parameter;
	RBSet<Material *> owners;
};

/* Material structs */

struct MaterialData {
	void update_uniform_buffer(const HashMap<StringName, ShaderLanguage::ShaderNode::Uniform> &p_uniforms, const uint32_t *p_uniform_offsets, const HashMap<StringName, Variant> &p_parameters, uint8_t *p_buffer, uint32_t p_buffer_size, bool p_use_linear_color);
	void update_textures(const HashMap<StringName, Variant> &p_parameters, const HashMap<StringName, HashMap<int, RID>> &p_default_textures, const Vector<ShaderCompiler::GeneratedCode::Texture> &p_texture_uniforms, RID *p_textures, bool p_use_linear_color);

	virtual void set_render_priority(int p_priority) = 0;
	virtual void set_next_pass(RID p_pass) = 0;
	virtual bool update_parameters(const HashMap<StringName, Variant> &p_parameters, bool p_uniform_dirty, bool p_textures_dirty) = 0;
	virtual ~MaterialData();

	//to be used internally by update_parameters, in the most common configuration of material parameters
	bool update_parameters_uniform_set(const HashMap<StringName, Variant> &p_parameters, bool p_uniform_dirty, bool p_textures_dirty, const HashMap<StringName, ShaderLanguage::ShaderNode::Uniform> &p_uniforms, const uint32_t *p_uniform_offsets, const Vector<ShaderCompiler::GeneratedCode::Texture> &p_texture_uniforms, const HashMap<StringName, HashMap<int, RID>> &p_default_texture_params, uint32_t p_ubo_size, RID &uniform_set, RID p_shader, uint32_t p_shader_uniform_set, uint32_t p_barrier = RD::BARRIER_MASK_ALL);
	void free_parameters_uniform_set(RID p_uniform_set);

private:
	friend class MaterialStorage;
	RID self;
	List<RID>::Element *global_buffer_E = nullptr;
	List<RID>::Element *global_texture_E = nullptr;
	uint64_t global_textures_pass = 0;
	HashMap<StringName, uint64_t> used_global_textures;

	//internally by update_parameters_uniform_set
	Vector<uint8_t> ubo_data;
	RID uniform_buffer;
	Vector<RID> texture_cache;
};

typedef MaterialData *(*MaterialDataRequestFunction)(ShaderData *);

struct Material {
	RID self;
	MaterialData *data = nullptr;
	Shader *shader = nullptr;
	//shortcut to shader data and type
	ShaderType shader_type = SHADER_TYPE_MAX;
	uint32_t shader_id = 0;
	bool uniform_dirty = false;
	bool texture_dirty = false;
	HashMap<StringName, Variant> params;
	int32_t priority = 0;
	RID next_pass;
	SelfList<Material> update_element;

	RendererStorage::Dependency dependency;

	Material() :
			update_element(this) {}
};

/* Global variable structs */
struct GlobalVariables {
	enum {
		BUFFER_DIRTY_REGION_SIZE = 1024
	};
	struct Variable {
		RBSet<RID> texture_materials; // materials using this

		RS::GlobalVariableType type;
		Variant value;
		Variant override;
		int32_t buffer_index; //for vectors
		int32_t buffer_elements; //for vectors
	};

	HashMap<StringName, Variable> variables;

	struct Value {
		float x;
		float y;
		float z;
		float w;
	};

	struct ValueInt {
		int32_t x;
		int32_t y;
		int32_t z;
		int32_t w;
	};

	struct ValueUInt {
		uint32_t x;
		uint32_t y;
		uint32_t z;
		uint32_t w;
	};

	struct ValueUsage {
		uint32_t elements = 0;
	};

	List<RID> materials_using_buffer;
	List<RID> materials_using_texture;

	RID buffer;
	Value *buffer_values = nullptr;
	ValueUsage *buffer_usage = nullptr;
	bool *buffer_dirty_regions = nullptr;
	uint32_t buffer_dirty_region_count = 0;

	uint32_t buffer_size;

	bool must_update_texture_materials = false;
	bool must_update_buffer_materials = false;

	HashMap<RID, int32_t> instance_buffer_pos;
};

class MaterialStorage : public RendererMaterialStorage {
private:
	friend struct MaterialData;
	static MaterialStorage *singleton;

	/* Samplers */

	RID default_rd_samplers[RS::CANVAS_ITEM_TEXTURE_FILTER_MAX][RS::CANVAS_ITEM_TEXTURE_REPEAT_MAX];
	RID custom_rd_samplers[RS::CANVAS_ITEM_TEXTURE_FILTER_MAX][RS::CANVAS_ITEM_TEXTURE_REPEAT_MAX];

	/* Buffers */

	RID quad_index_buffer;
	RID quad_index_array;

	/* GLOBAL VARIABLE API */

	GlobalVariables global_variables;

	int32_t _global_variable_allocate(uint32_t p_elements);
	void _global_variable_store_in_buffer(int32_t p_index, RS::GlobalVariableType p_type, const Variant &p_value);
	void _global_variable_mark_buffer_dirty(int32_t p_index, int32_t p_elements);

	/* SHADER API */

	ShaderDataRequestFunction shader_data_request_func[SHADER_TYPE_MAX];
	mutable RID_Owner<Shader, true> shader_owner;

	/* MATERIAL API */
	MaterialDataRequestFunction material_data_request_func[SHADER_TYPE_MAX];
	mutable RID_Owner<Material, true> material_owner;

	SelfList<Material>::List material_update_list;

	static void _material_uniform_set_erased(void *p_material);

public:
	static MaterialStorage *get_singleton();

	MaterialStorage();
	virtual ~MaterialStorage();

	/* Samplers */

	_FORCE_INLINE_ RID sampler_rd_get_default(RS::CanvasItemTextureFilter p_filter, RS::CanvasItemTextureRepeat p_repeat) {
		return default_rd_samplers[p_filter][p_repeat];
	}
	_FORCE_INLINE_ RID sampler_rd_get_custom(RS::CanvasItemTextureFilter p_filter, RS::CanvasItemTextureRepeat p_repeat) {
		return custom_rd_samplers[p_filter][p_repeat];
	}

	void sampler_rd_configure_custom(float mipmap_bias);

	// void sampler_rd_set_default(float p_mipmap_bias);

	/* Buffers */

	RID get_quad_index_array() { return quad_index_array; }

	/* GLOBAL VARIABLE API */

	void _update_global_variables();

	virtual void global_variable_add(const StringName &p_name, RS::GlobalVariableType p_type, const Variant &p_value) override;
	virtual void global_variable_remove(const StringName &p_name) override;
	virtual Vector<StringName> global_variable_get_list() const override;

	virtual void global_variable_set(const StringName &p_name, const Variant &p_value) override;
	virtual void global_variable_set_override(const StringName &p_name, const Variant &p_value) override;
	virtual Variant global_variable_get(const StringName &p_name) const override;
	virtual RS::GlobalVariableType global_variable_get_type(const StringName &p_name) const override;
	RS::GlobalVariableType global_variable_get_type_internal(const StringName &p_name) const;

	virtual void global_variables_load_settings(bool p_load_textures = true) override;
	virtual void global_variables_clear() override;

	virtual int32_t global_variables_instance_allocate(RID p_instance) override;
	virtual void global_variables_instance_free(RID p_instance) override;
	virtual void global_variables_instance_update(RID p_instance, int p_index, const Variant &p_value) override;

	RID global_variables_get_storage_buffer() const;

	/* SHADER API */

	Shader *get_shader(RID p_rid) { return shader_owner.get_or_null(p_rid); };
	bool owns_shader(RID p_rid) { return shader_owner.owns(p_rid); };

	virtual RID shader_allocate() override;
	virtual void shader_initialize(RID p_shader) override;
	virtual void shader_free(RID p_rid) override;

	virtual void shader_set_code(RID p_shader, const String &p_code) override;
	virtual String shader_get_code(RID p_shader) const override;
	virtual void shader_get_param_list(RID p_shader, List<PropertyInfo> *p_param_list) const override;

	virtual void shader_set_default_texture_param(RID p_shader, const StringName &p_name, RID p_texture, int p_index) override;
	virtual RID shader_get_default_texture_param(RID p_shader, const StringName &p_name, int p_index) const override;
	virtual Variant shader_get_param_default(RID p_shader, const StringName &p_param) const override;
	void shader_set_data_request_function(ShaderType p_shader_type, ShaderDataRequestFunction p_function);

	virtual RS::ShaderNativeSourceCode shader_get_native_source_code(RID p_shader) const override;

	/* MATERIAL API */

	Material *get_material(RID p_rid) { return material_owner.get_or_null(p_rid); };
	bool owns_material(RID p_rid) { return material_owner.owns(p_rid); };

	void _material_queue_update(Material *material, bool p_uniform, bool p_texture);
	void _update_queued_materials();

	virtual RID material_allocate() override;
	virtual void material_initialize(RID p_material) override;
	virtual void material_free(RID p_rid) override;

	virtual void material_set_shader(RID p_material, RID p_shader) override;

	virtual void material_set_param(RID p_material, const StringName &p_param, const Variant &p_value) override;
	virtual Variant material_get_param(RID p_material, const StringName &p_param) const override;

	virtual void material_set_next_pass(RID p_material, RID p_next_material) override;
	virtual void material_set_render_priority(RID p_material, int priority) override;

	virtual bool material_is_animated(RID p_material) override;
	virtual bool material_casts_shadows(RID p_material) override;

	virtual void material_get_instance_shader_parameters(RID p_material, List<InstanceShaderParam> *r_parameters) override;

	virtual void material_update_dependency(RID p_material, RendererStorage::DependencyTracker *p_instance) override;

	void material_set_data_request_function(ShaderType p_shader_type, MaterialDataRequestFunction p_function);
	MaterialDataRequestFunction material_get_data_request_function(ShaderType p_shader_type);

	_FORCE_INLINE_ uint32_t material_get_shader_id(RID p_material) {
		Material *material = material_owner.get_or_null(p_material);
		return material->shader_id;
	}

	_FORCE_INLINE_ MaterialData *material_get_data(RID p_material, ShaderType p_shader_type) {
		Material *material = material_owner.get_or_null(p_material);
		if (!material || material->shader_type != p_shader_type) {
			return nullptr;
		} else {
			return material->data;
		}
	}
};

} // namespace RendererRD

#endif // !MATERIAL_STORAGE_RD_H
