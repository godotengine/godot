/**************************************************************************/
/*  surface_tool.hpp                                                      */
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
#include <godot_cpp/classes/mesh.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/aabb.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/packed_color_array.hpp>
#include <godot_cpp/variant/packed_int32_array.hpp>
#include <godot_cpp/variant/packed_vector2_array.hpp>
#include <godot_cpp/variant/packed_vector3_array.hpp>
#include <godot_cpp/variant/plane.hpp>
#include <godot_cpp/variant/typed_array.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

struct Color;
class Material;
class PackedFloat32Array;
class String;
struct Transform3D;
struct Vector2;
struct Vector3;

class SurfaceTool : public RefCounted {
	GDEXTENSION_CLASS(SurfaceTool, RefCounted)

public:
	enum CustomFormat {
		CUSTOM_RGBA8_UNORM = 0,
		CUSTOM_RGBA8_SNORM = 1,
		CUSTOM_RG_HALF = 2,
		CUSTOM_RGBA_HALF = 3,
		CUSTOM_R_FLOAT = 4,
		CUSTOM_RG_FLOAT = 5,
		CUSTOM_RGB_FLOAT = 6,
		CUSTOM_RGBA_FLOAT = 7,
		CUSTOM_MAX = 8,
	};

	enum SkinWeightCount {
		SKIN_4_WEIGHTS = 0,
		SKIN_8_WEIGHTS = 1,
	};

	void set_skin_weight_count(SurfaceTool::SkinWeightCount p_count);
	SurfaceTool::SkinWeightCount get_skin_weight_count() const;
	void set_custom_format(int32_t p_channel_index, SurfaceTool::CustomFormat p_format);
	SurfaceTool::CustomFormat get_custom_format(int32_t p_channel_index) const;
	void begin(Mesh::PrimitiveType p_primitive);
	void add_vertex(const Vector3 &p_vertex);
	void set_color(const Color &p_color);
	void set_normal(const Vector3 &p_normal);
	void set_tangent(const Plane &p_tangent);
	void set_uv(const Vector2 &p_uv);
	void set_uv2(const Vector2 &p_uv2);
	void set_bones(const PackedInt32Array &p_bones);
	void set_weights(const PackedFloat32Array &p_weights);
	void set_custom(int32_t p_channel_index, const Color &p_custom_color);
	void set_smooth_group(uint32_t p_index);
	void add_triangle_fan(const PackedVector3Array &p_vertices, const PackedVector2Array &p_uvs = PackedVector2Array(), const PackedColorArray &p_colors = PackedColorArray(), const PackedVector2Array &p_uv2s = PackedVector2Array(), const PackedVector3Array &p_normals = PackedVector3Array(), const TypedArray<Plane> &p_tangents = {});
	void add_index(int32_t p_index);
	void index();
	void deindex();
	void generate_normals(bool p_flip = false);
	void generate_tangents();
	void optimize_indices_for_cache();
	AABB get_aabb() const;
	PackedInt32Array generate_lod(float p_nd_threshold, int32_t p_target_index_count = 3);
	void set_material(const Ref<Material> &p_material);
	Mesh::PrimitiveType get_primitive_type() const;
	void clear();
	void create_from(const Ref<Mesh> &p_existing, int32_t p_surface);
	void create_from_arrays(const Array &p_arrays, Mesh::PrimitiveType p_primitive_type = (Mesh::PrimitiveType)3);
	void create_from_blend_shape(const Ref<Mesh> &p_existing, int32_t p_surface, const String &p_blend_shape);
	void append_from(const Ref<Mesh> &p_existing, int32_t p_surface, const Transform3D &p_transform);
	Ref<ArrayMesh> commit(const Ref<ArrayMesh> &p_existing = nullptr, uint64_t p_flags = 0);
	Array commit_to_arrays();

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		RefCounted::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(SurfaceTool::CustomFormat);
VARIANT_ENUM_CAST(SurfaceTool::SkinWeightCount);

