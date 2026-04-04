/**************************************************************************/
/*  importer_mesh.hpp                                                     */
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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#pragma once

#include <godot_cpp/classes/array_mesh.hpp>
#include <godot_cpp/classes/material.hpp>
#include <godot_cpp/classes/mesh.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/resource.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/packed_int32_array.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/typed_array.hpp>
#include <godot_cpp/variant/vector2i.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class ImporterMesh : public Resource {
	GDEXTENSION_CLASS(ImporterMesh, Resource)

public:
	void add_blend_shape(const String &p_name);
	int32_t get_blend_shape_count() const;
	String get_blend_shape_name(int32_t p_blend_shape_idx) const;
	void set_blend_shape_mode(Mesh::BlendShapeMode p_mode);
	Mesh::BlendShapeMode get_blend_shape_mode() const;
	void add_surface(Mesh::PrimitiveType p_primitive, const Array &p_arrays, const TypedArray<Array> &p_blend_shapes = {}, const Dictionary &p_lods = Dictionary(), const Ref<Material> &p_material = nullptr, const String &p_name = String(), uint64_t p_flags = 0);
	int32_t get_surface_count() const;
	Mesh::PrimitiveType get_surface_primitive_type(int32_t p_surface_idx);
	String get_surface_name(int32_t p_surface_idx) const;
	Array get_surface_arrays(int32_t p_surface_idx) const;
	Array get_surface_blend_shape_arrays(int32_t p_surface_idx, int32_t p_blend_shape_idx) const;
	int32_t get_surface_lod_count(int32_t p_surface_idx) const;
	float get_surface_lod_size(int32_t p_surface_idx, int32_t p_lod_idx) const;
	PackedInt32Array get_surface_lod_indices(int32_t p_surface_idx, int32_t p_lod_idx) const;
	Ref<Material> get_surface_material(int32_t p_surface_idx) const;
	uint64_t get_surface_format(int32_t p_surface_idx) const;
	void set_surface_name(int32_t p_surface_idx, const String &p_name);
	void set_surface_material(int32_t p_surface_idx, const Ref<Material> &p_material);
	void generate_lods(float p_normal_merge_angle, float p_normal_split_angle, const Array &p_bone_transform_array);
	Ref<ArrayMesh> get_mesh(const Ref<ArrayMesh> &p_base_mesh = nullptr);
	static Ref<ImporterMesh> from_mesh(const Ref<Mesh> &p_mesh);
	void clear();
	void set_lightmap_size_hint(const Vector2i &p_size);
	Vector2i get_lightmap_size_hint() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Resource::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

