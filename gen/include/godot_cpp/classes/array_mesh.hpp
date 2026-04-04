/**************************************************************************/
/*  array_mesh.hpp                                                        */
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

#include <godot_cpp/classes/global_constants.hpp>
#include <godot_cpp/classes/mesh.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/variant/aabb.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/string_name.hpp>
#include <godot_cpp/variant/typed_array.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class PackedByteArray;
struct Transform3D;

class ArrayMesh : public Mesh {
	GDEXTENSION_CLASS(ArrayMesh, Mesh)

public:
	void add_blend_shape(const StringName &p_name);
	int32_t get_blend_shape_count() const;
	StringName get_blend_shape_name(int32_t p_index) const;
	void set_blend_shape_name(int32_t p_index, const StringName &p_name);
	void clear_blend_shapes();
	void set_blend_shape_mode(Mesh::BlendShapeMode p_mode);
	Mesh::BlendShapeMode get_blend_shape_mode() const;
	void add_surface_from_arrays(Mesh::PrimitiveType p_primitive, const Array &p_arrays, const TypedArray<Array> &p_blend_shapes = Array(), const Dictionary &p_lods = Dictionary(), BitField<Mesh::ArrayFormat> p_flags = (BitField<Mesh::ArrayFormat>)0);
	void clear_surfaces();
	void surface_remove(int32_t p_surf_idx);
	void surface_update_vertex_region(int32_t p_surf_idx, int32_t p_offset, const PackedByteArray &p_data);
	void surface_update_attribute_region(int32_t p_surf_idx, int32_t p_offset, const PackedByteArray &p_data);
	void surface_update_skin_region(int32_t p_surf_idx, int32_t p_offset, const PackedByteArray &p_data);
	int32_t surface_get_array_len(int32_t p_surf_idx) const;
	int32_t surface_get_array_index_len(int32_t p_surf_idx) const;
	BitField<Mesh::ArrayFormat> surface_get_format(int32_t p_surf_idx) const;
	Mesh::PrimitiveType surface_get_primitive_type(int32_t p_surf_idx) const;
	int32_t surface_find_by_name(const String &p_name) const;
	void surface_set_name(int32_t p_surf_idx, const String &p_name);
	String surface_get_name(int32_t p_surf_idx) const;
	void regen_normal_maps();
	Error lightmap_unwrap(const Transform3D &p_transform, float p_texel_size);
	void set_custom_aabb(const AABB &p_aabb);
	AABB get_custom_aabb() const;
	void set_shadow_mesh(const Ref<ArrayMesh> &p_mesh);
	Ref<ArrayMesh> get_shadow_mesh() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Mesh::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

