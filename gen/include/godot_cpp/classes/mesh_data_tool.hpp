/**************************************************************************/
/*  mesh_data_tool.hpp                                                    */
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
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/color.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/packed_int32_array.hpp>
#include <godot_cpp/variant/plane.hpp>
#include <godot_cpp/variant/variant.hpp>
#include <godot_cpp/variant/vector2.hpp>
#include <godot_cpp/variant/vector3.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class ArrayMesh;
class Material;

class MeshDataTool : public RefCounted {
	GDEXTENSION_CLASS(MeshDataTool, RefCounted)

public:
	void clear();
	Error create_from_surface(const Ref<ArrayMesh> &p_mesh, int32_t p_surface);
	Error commit_to_surface(const Ref<ArrayMesh> &p_mesh, uint64_t p_compression_flags = 0);
	uint64_t get_format() const;
	int32_t get_vertex_count() const;
	int32_t get_edge_count() const;
	int32_t get_face_count() const;
	void set_vertex(int32_t p_idx, const Vector3 &p_vertex);
	Vector3 get_vertex(int32_t p_idx) const;
	void set_vertex_normal(int32_t p_idx, const Vector3 &p_normal);
	Vector3 get_vertex_normal(int32_t p_idx) const;
	void set_vertex_tangent(int32_t p_idx, const Plane &p_tangent);
	Plane get_vertex_tangent(int32_t p_idx) const;
	void set_vertex_uv(int32_t p_idx, const Vector2 &p_uv);
	Vector2 get_vertex_uv(int32_t p_idx) const;
	void set_vertex_uv2(int32_t p_idx, const Vector2 &p_uv2);
	Vector2 get_vertex_uv2(int32_t p_idx) const;
	void set_vertex_color(int32_t p_idx, const Color &p_color);
	Color get_vertex_color(int32_t p_idx) const;
	void set_vertex_bones(int32_t p_idx, const PackedInt32Array &p_bones);
	PackedInt32Array get_vertex_bones(int32_t p_idx) const;
	void set_vertex_weights(int32_t p_idx, const PackedFloat32Array &p_weights);
	PackedFloat32Array get_vertex_weights(int32_t p_idx) const;
	void set_vertex_meta(int32_t p_idx, const Variant &p_meta);
	Variant get_vertex_meta(int32_t p_idx) const;
	PackedInt32Array get_vertex_edges(int32_t p_idx) const;
	PackedInt32Array get_vertex_faces(int32_t p_idx) const;
	int32_t get_edge_vertex(int32_t p_idx, int32_t p_vertex) const;
	PackedInt32Array get_edge_faces(int32_t p_idx) const;
	void set_edge_meta(int32_t p_idx, const Variant &p_meta);
	Variant get_edge_meta(int32_t p_idx) const;
	int32_t get_face_vertex(int32_t p_idx, int32_t p_vertex) const;
	int32_t get_face_edge(int32_t p_idx, int32_t p_edge) const;
	void set_face_meta(int32_t p_idx, const Variant &p_meta);
	Variant get_face_meta(int32_t p_idx) const;
	Vector3 get_face_normal(int32_t p_idx) const;
	void set_material(const Ref<Material> &p_material);
	Ref<Material> get_material() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		RefCounted::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

