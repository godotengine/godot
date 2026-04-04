/**************************************************************************/
/*  mesh.hpp                                                              */
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

#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/resource.hpp>
#include <godot_cpp/variant/aabb.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/packed_vector3_array.hpp>
#include <godot_cpp/variant/string_name.hpp>
#include <godot_cpp/variant/typed_array.hpp>
#include <godot_cpp/variant/vector2i.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class ConcavePolygonShape3D;
class ConvexPolygonShape3D;
class Material;
class TriangleMesh;

class Mesh : public Resource {
	GDEXTENSION_CLASS(Mesh, Resource)

public:
	enum PrimitiveType {
		PRIMITIVE_POINTS = 0,
		PRIMITIVE_LINES = 1,
		PRIMITIVE_LINE_STRIP = 2,
		PRIMITIVE_TRIANGLES = 3,
		PRIMITIVE_TRIANGLE_STRIP = 4,
	};

	enum ArrayType {
		ARRAY_VERTEX = 0,
		ARRAY_NORMAL = 1,
		ARRAY_TANGENT = 2,
		ARRAY_COLOR = 3,
		ARRAY_TEX_UV = 4,
		ARRAY_TEX_UV2 = 5,
		ARRAY_CUSTOM0 = 6,
		ARRAY_CUSTOM1 = 7,
		ARRAY_CUSTOM2 = 8,
		ARRAY_CUSTOM3 = 9,
		ARRAY_BONES = 10,
		ARRAY_WEIGHTS = 11,
		ARRAY_INDEX = 12,
		ARRAY_MAX = 13,
	};

	enum ArrayCustomFormat {
		ARRAY_CUSTOM_RGBA8_UNORM = 0,
		ARRAY_CUSTOM_RGBA8_SNORM = 1,
		ARRAY_CUSTOM_RG_HALF = 2,
		ARRAY_CUSTOM_RGBA_HALF = 3,
		ARRAY_CUSTOM_R_FLOAT = 4,
		ARRAY_CUSTOM_RG_FLOAT = 5,
		ARRAY_CUSTOM_RGB_FLOAT = 6,
		ARRAY_CUSTOM_RGBA_FLOAT = 7,
		ARRAY_CUSTOM_MAX = 8,
	};

	enum ArrayFormat : uint64_t {
		ARRAY_FORMAT_VERTEX = 1,
		ARRAY_FORMAT_NORMAL = 2,
		ARRAY_FORMAT_TANGENT = 4,
		ARRAY_FORMAT_COLOR = 8,
		ARRAY_FORMAT_TEX_UV = 16,
		ARRAY_FORMAT_TEX_UV2 = 32,
		ARRAY_FORMAT_CUSTOM0 = 64,
		ARRAY_FORMAT_CUSTOM1 = 128,
		ARRAY_FORMAT_CUSTOM2 = 256,
		ARRAY_FORMAT_CUSTOM3 = 512,
		ARRAY_FORMAT_BONES = 1024,
		ARRAY_FORMAT_WEIGHTS = 2048,
		ARRAY_FORMAT_INDEX = 4096,
		ARRAY_FORMAT_BLEND_SHAPE_MASK = 7,
		ARRAY_FORMAT_CUSTOM_BASE = 13,
		ARRAY_FORMAT_CUSTOM_BITS = 3,
		ARRAY_FORMAT_CUSTOM0_SHIFT = 13,
		ARRAY_FORMAT_CUSTOM1_SHIFT = 16,
		ARRAY_FORMAT_CUSTOM2_SHIFT = 19,
		ARRAY_FORMAT_CUSTOM3_SHIFT = 22,
		ARRAY_FORMAT_CUSTOM_MASK = 7,
		ARRAY_COMPRESS_FLAGS_BASE = 25,
		ARRAY_FLAG_USE_2D_VERTICES = 33554432,
		ARRAY_FLAG_USE_DYNAMIC_UPDATE = 67108864,
		ARRAY_FLAG_USE_8_BONE_WEIGHTS = 134217728,
		ARRAY_FLAG_USES_EMPTY_VERTEX_ARRAY = 268435456,
		ARRAY_FLAG_COMPRESS_ATTRIBUTES = 536870912,
	};

	enum BlendShapeMode {
		BLEND_SHAPE_MODE_NORMALIZED = 0,
		BLEND_SHAPE_MODE_RELATIVE = 1,
	};

	void set_lightmap_size_hint(const Vector2i &p_size);
	Vector2i get_lightmap_size_hint() const;
	AABB get_aabb() const;
	PackedVector3Array get_faces() const;
	int32_t get_surface_count() const;
	Array surface_get_arrays(int32_t p_surf_idx) const;
	TypedArray<Array> surface_get_blend_shape_arrays(int32_t p_surf_idx) const;
	void surface_set_material(int32_t p_surf_idx, const Ref<Material> &p_material);
	Ref<Material> surface_get_material(int32_t p_surf_idx) const;
	Ref<Resource> create_placeholder() const;
	Ref<ConcavePolygonShape3D> create_trimesh_shape() const;
	Ref<ConvexPolygonShape3D> create_convex_shape(bool p_clean = true, bool p_simplify = false) const;
	Ref<Mesh> create_outline(float p_margin) const;
	Ref<TriangleMesh> generate_triangle_mesh() const;
	virtual int32_t _get_surface_count() const;
	virtual int32_t _surface_get_array_len(int32_t p_index) const;
	virtual int32_t _surface_get_array_index_len(int32_t p_index) const;
	virtual Array _surface_get_arrays(int32_t p_index) const;
	virtual TypedArray<Array> _surface_get_blend_shape_arrays(int32_t p_index) const;
	virtual Dictionary _surface_get_lods(int32_t p_index) const;
	virtual uint32_t _surface_get_format(int32_t p_index) const;
	virtual uint32_t _surface_get_primitive_type(int32_t p_index) const;
	virtual void _surface_set_material(int32_t p_index, const Ref<Material> &p_material);
	virtual Ref<Material> _surface_get_material(int32_t p_index) const;
	virtual int32_t _get_blend_shape_count() const;
	virtual StringName _get_blend_shape_name(int32_t p_index) const;
	virtual void _set_blend_shape_name(int32_t p_index, const StringName &p_name);
	virtual AABB _get_aabb() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Resource::register_virtuals<T, B>();
		if constexpr (!std::is_same_v<decltype(&B::_get_surface_count), decltype(&T::_get_surface_count)>) {
			BIND_VIRTUAL_METHOD(T, _get_surface_count, 3905245786);
		}
		if constexpr (!std::is_same_v<decltype(&B::_surface_get_array_len), decltype(&T::_surface_get_array_len)>) {
			BIND_VIRTUAL_METHOD(T, _surface_get_array_len, 923996154);
		}
		if constexpr (!std::is_same_v<decltype(&B::_surface_get_array_index_len), decltype(&T::_surface_get_array_index_len)>) {
			BIND_VIRTUAL_METHOD(T, _surface_get_array_index_len, 923996154);
		}
		if constexpr (!std::is_same_v<decltype(&B::_surface_get_arrays), decltype(&T::_surface_get_arrays)>) {
			BIND_VIRTUAL_METHOD(T, _surface_get_arrays, 663333327);
		}
		if constexpr (!std::is_same_v<decltype(&B::_surface_get_blend_shape_arrays), decltype(&T::_surface_get_blend_shape_arrays)>) {
			BIND_VIRTUAL_METHOD(T, _surface_get_blend_shape_arrays, 663333327);
		}
		if constexpr (!std::is_same_v<decltype(&B::_surface_get_lods), decltype(&T::_surface_get_lods)>) {
			BIND_VIRTUAL_METHOD(T, _surface_get_lods, 3485342025);
		}
		if constexpr (!std::is_same_v<decltype(&B::_surface_get_format), decltype(&T::_surface_get_format)>) {
			BIND_VIRTUAL_METHOD(T, _surface_get_format, 923996154);
		}
		if constexpr (!std::is_same_v<decltype(&B::_surface_get_primitive_type), decltype(&T::_surface_get_primitive_type)>) {
			BIND_VIRTUAL_METHOD(T, _surface_get_primitive_type, 923996154);
		}
		if constexpr (!std::is_same_v<decltype(&B::_surface_set_material), decltype(&T::_surface_set_material)>) {
			BIND_VIRTUAL_METHOD(T, _surface_set_material, 3671737478);
		}
		if constexpr (!std::is_same_v<decltype(&B::_surface_get_material), decltype(&T::_surface_get_material)>) {
			BIND_VIRTUAL_METHOD(T, _surface_get_material, 2897466400);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_blend_shape_count), decltype(&T::_get_blend_shape_count)>) {
			BIND_VIRTUAL_METHOD(T, _get_blend_shape_count, 3905245786);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_blend_shape_name), decltype(&T::_get_blend_shape_name)>) {
			BIND_VIRTUAL_METHOD(T, _get_blend_shape_name, 659327637);
		}
		if constexpr (!std::is_same_v<decltype(&B::_set_blend_shape_name), decltype(&T::_set_blend_shape_name)>) {
			BIND_VIRTUAL_METHOD(T, _set_blend_shape_name, 3780747571);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_aabb), decltype(&T::_get_aabb)>) {
			BIND_VIRTUAL_METHOD(T, _get_aabb, 1068685055);
		}
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(Mesh::PrimitiveType);
VARIANT_ENUM_CAST(Mesh::ArrayType);
VARIANT_ENUM_CAST(Mesh::ArrayCustomFormat);
VARIANT_BITFIELD_CAST(Mesh::ArrayFormat);
VARIANT_ENUM_CAST(Mesh::BlendShapeMode);

