/**************************************************************************/
/*  material_storage.h                                                    */
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

#pragma once

#include "servers/rendering/rendering_server.h"
#include "utilities.h"
#include "core/variant/typed_dictionary.h"

class RendererMaterialStorage {
public:
	virtual ~RendererMaterialStorage() {}

	/* GLOBAL SHADER UNIFORM API */
	virtual void global_shader_parameter_add(const StringName &p_name, RS::GlobalShaderParameterType p_type, const Variant &p_value) = 0;
	virtual void global_shader_parameter_remove(const StringName &p_name) = 0;
	virtual Vector<StringName> global_shader_parameter_get_list() const = 0;

	virtual void global_shader_parameter_set(const StringName &p_name, const Variant &p_value) = 0;
	virtual void global_shader_parameter_set_override(const StringName &p_name, const Variant &p_value) = 0;
	virtual Variant global_shader_parameter_get(const StringName &p_name) const = 0;
	virtual RS::GlobalShaderParameterType global_shader_parameter_get_type(const StringName &p_name) const = 0;

	virtual void global_shader_parameters_load_settings(bool p_load_textures = true) = 0;
	virtual void global_shader_parameters_clear() = 0;

	virtual int32_t global_shader_parameters_instance_allocate(RID p_instance) = 0;
	virtual void global_shader_parameters_instance_free(RID p_instance) = 0;
	virtual void global_shader_parameters_instance_update(RID p_instance, int p_index, const Variant &p_value, int p_flags_count = 0) = 0;

	/* SHADER API */
	virtual RID shader_allocate() = 0;
	virtual void shader_initialize(RID p_rid, bool p_embedded = true) = 0;
	virtual void shader_free(RID p_rid) = 0;

	virtual void shader_set_code(RID p_shader, const String &p_code) = 0;
	virtual void shader_set_path_hint(RID p_shader, const String &p_path) = 0;
	virtual String shader_get_code(RID p_shader) const = 0;
	virtual void get_shader_parameter_list(RID p_shader, List<PropertyInfo> *p_param_list) const = 0;

	virtual void shader_set_default_texture_parameter(RID p_shader, const StringName &p_name, RID p_texture, int p_index) = 0;
	virtual RID shader_get_default_texture_parameter(RID p_shader, const StringName &p_name, int p_index) const = 0;
	virtual Variant shader_get_parameter_default(RID p_material, const StringName &p_param) const = 0;

	virtual RS::ShaderNativeSourceCode shader_get_native_source_code(RID p_shader) const = 0;
	virtual void shader_embedded_set_lock() = 0;
	virtual const HashSet<RID> &shader_embedded_set_get() const = 0;
	virtual void shader_embedded_set_unlock() = 0;

	/* MATERIAL API */

	virtual RID material_allocate() = 0;
	virtual void material_initialize(RID p_rid) = 0;
	virtual void material_free(RID p_rid) = 0;

	virtual void material_set_render_priority(RID p_material, int priority) = 0;
	virtual void material_set_shader(RID p_shader_material, RID p_shader) = 0;

	virtual void material_set_param(RID p_material, const StringName &p_param, const Variant &p_value) = 0;
	virtual Variant material_get_param(RID p_material, const StringName &p_param) const = 0;

	virtual void material_set_buffer(RID p_material, const StringName &p_buffer, const TypedDictionary<StringName, Variant> &p_values) = 0;
	virtual void material_update_buffer(RID p_material, const StringName &p_buffer, const TypedDictionary<StringName, Variant> &p_values) = 0;
	virtual TypedDictionary<StringName, Variant> material_get_buffer(RID p_material, const StringName &p_buffer) const = 0;
	virtual void material_set_buffer_raw(RID p_material, const StringName &p_buffer, const PackedByteArray &p_values) = 0;
	virtual PackedByteArray material_get_buffer_raw(RID p_material, const StringName &p_buffer) const = 0;
	virtual void material_set_buffer_field(RID p_material, const StringName &p_buffer, const StringName &p_field, const Variant &p_value) = 0;
	virtual Variant material_get_buffer_field(RID p_material, const StringName &p_buffer, const StringName &p_field) const = 0;

	virtual void material_set_next_pass(RID p_material, RID p_next_material) = 0;

	virtual bool material_is_animated(RID p_material) = 0;
	virtual bool material_casts_shadows(RID p_material) = 0;
	virtual RS::CullMode material_get_cull_mode(RID p_material) const = 0;

	struct InstanceShaderParam {
		PropertyInfo info;
		int index;
		Variant default_value;
	};

	virtual void material_get_instance_shader_parameters(RID p_material, List<InstanceShaderParam> *r_parameters) = 0;

	virtual void material_update_dependency(RID p_material, DependencyTracker *p_instance) = 0;

	// helper function for buffers
	static void format_buffer_data(PackedByteArray &data, const String &format, const Variant &value, bool empty = false) {
		bool std430 = false;
		if (format == "std430") {
			std430 = true;
		}
		// byte alignment for std140
		constexpr int std140_alignment = 16;
		int alignment = 4;

		// handle structs
		if (value.get_type() == Variant::DICTIONARY) {
			PackedByteArray struct_data;
			for (const KeyValue<Variant, Variant> &val : static_cast<Dictionary>(value)) {
				PackedByteArray struct_member_data;
				format_buffer_data(struct_member_data, format, val.value, empty);
				if (struct_member_data.size() > alignment) {
					alignment = struct_member_data.size();
				}
				struct_data.append_array(struct_member_data);
			}
			// std430 has different requirements for padding for alignment in structs
			PackedByteArray alignment_data;
			if (!std430 && alignment < std140_alignment) {
				alignment = std140_alignment;
			} 
			alignment_data.resize_initialized(alignment - (struct_data.size() % alignment));
			struct_data.append_array(alignment_data);
			
			data.append_array(struct_data);
			return;
		}

		// handle arrays
		if (value.get_type() == Variant::ARRAY) {
			// holds the data representation of the entire array
			PackedByteArray array_data;

			// for each value in the array, get its data form and then align it as specified for the format
			for (const Variant &val : static_cast<Array>(value)) {
				PackedByteArray array_value_data;
				format_buffer_data(array_value_data, format, val, empty);

				// std430 has different requirements for padding for alignment in arrays
				if (!std430) {
					int alignment_gap = (std140_alignment - (array_value_data.size() % std140_alignment)) % std140_alignment;
					PackedByteArray alignment_data;
					alignment_data.resize_initialized(alignment_gap);
					array_value_data.append_array(alignment_data);
				} 
				array_data.append_array(array_value_data);
			}
			data.append_array(array_data);
			return;
		}


		PackedByteArray out;
		int data_size;
		constexpr int max_var_size = 64;
		void *val_out = memalloc(max_var_size);
		// handle normal variables
		switch (value.get_type()) {
			case Variant::FLOAT: {
				out.resize_initialized(4);
				*((float*) val_out) = value;
				data_size = 4;
			} break;
			case Variant::INT: {
				out.resize_initialized(4);
				*((int*) val_out) = value;
				data_size = 4;
			} break;
			case Variant::BOOL: {
				out.resize_initialized(4);
				*((bool*) val_out) = value;
				data_size = 4;
			} break;
			case Variant::VECTOR2: {
				out.resize_initialized(8);
				*((Vector2 *)val_out) = value;
				data_size = 8;
			} break;
			case Variant::VECTOR3: {
				out.resize_initialized(16);
				*((Vector3 *)val_out) = value;
				data_size = 16;
			} break;
			case Variant::VECTOR4: {
				out.resize_initialized(16);
				*((Vector4 *)val_out) = value;
				data_size = 16;
			} break;
			case Variant::VECTOR2I: {
				out.resize_initialized(8);
				*((Vector2i *)val_out) = value;
				data_size = 8;
			} break;
			case Variant::VECTOR3I: {
				out.resize_initialized(16);
				*((Vector3i *)val_out) = value;
				data_size = 16;
			} break;
			case Variant::VECTOR4I: {
				out.resize_initialized(16);
				*((Vector4i *)val_out) = value;
				data_size = 16;
			} break;
			default: {
				// should never be reached
			} return;
		}
		if (!empty) {
			memcpy(out.ptrw(), val_out, data_size);
		} 
		data.append_array(out);
	}
};

