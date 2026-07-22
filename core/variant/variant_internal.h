/**************************************************************************/
/*  variant_internal.h                                                    */
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

#include "core/templates/simple_type.h"
#include "core/variant/type_info.h"
#include "core/variant/variant.h"
#include "core/variant/variant_pools.h"

// For use when you want to access the internal pointer of a Variant directly.
// Use with caution. You need to be sure that the type is correct.

class RefCounted;

template <typename T>
struct GDExtensionPtr;

class VariantInternal {
	friend class Variant;

public:
	// Set type.
	_FORCE_INLINE_ static void set_type(Variant &r_variant, Variant::Type p_type) {
		r_variant.type = p_type;
	}

	_FORCE_INLINE_ static void initialize(Variant *r_variant, Variant::Type p_type) {
		r_variant->clear();
		r_variant->type = p_type;

		switch (p_type) {
			case Variant::STRING:
				init_string(r_variant);
				break;
			case Variant::TRANSFORM2D:
				init_transform2d(r_variant);
				break;
			case Variant::QUATERNION:
				init_quaternion(r_variant);
				break;
			case Variant::AABB:
				init_aabb(r_variant);
				break;
			case Variant::BASIS:
				init_basis(r_variant);
				break;
			case Variant::TRANSFORM3D:
				init_transform3d(r_variant);
				break;
			case Variant::PROJECTION:
				init_projection(r_variant);
				break;
			case Variant::COLOR:
				init_color(r_variant);
				break;
			case Variant::STRING_NAME:
				init_string_name(r_variant);
				break;
			case Variant::NODE_PATH:
				init_node_path(r_variant);
				break;
			case Variant::CALLABLE:
				init_callable(r_variant);
				break;
			case Variant::SIGNAL:
				init_signal(r_variant);
				break;
			case Variant::DICTIONARY:
				init_dictionary(r_variant);
				break;
			case Variant::ARRAY:
				init_array(r_variant);
				break;
			case Variant::PACKED_BYTE_ARRAY:
				init_byte_array(r_variant);
				break;
			case Variant::PACKED_INT32_ARRAY:
				init_int32_array(r_variant);
				break;
			case Variant::PACKED_INT64_ARRAY:
				init_int64_array(r_variant);
				break;
			case Variant::PACKED_FLOAT32_ARRAY:
				init_float32_array(r_variant);
				break;
			case Variant::PACKED_FLOAT64_ARRAY:
				init_float64_array(r_variant);
				break;
			case Variant::PACKED_STRING_ARRAY:
				init_string_array(r_variant);
				break;
			case Variant::PACKED_VECTOR2_ARRAY:
				init_vector2_array(r_variant);
				break;
			case Variant::PACKED_VECTOR3_ARRAY:
				init_vector3_array(r_variant);
				break;
			case Variant::PACKED_COLOR_ARRAY:
				init_color_array(r_variant);
				break;
			case Variant::PACKED_VECTOR4_ARRAY:
				init_vector4_array(r_variant);
				break;
			case Variant::OBJECT:
				init_object(r_variant);
				break;
			default:
				break;
		}
	}

	// Atomic types.
	_FORCE_INLINE_ static bool *get_bool(Variant *p_variant) { return &p_variant->_data._bool; }
	_FORCE_INLINE_ static const bool *get_bool(const Variant *p_variant) { return &p_variant->_data._bool; }
	_FORCE_INLINE_ static int64_t *get_int(Variant *p_variant) { return &p_variant->_data._int; }
	_FORCE_INLINE_ static const int64_t *get_int(const Variant *p_variant) { return &p_variant->_data._int; }
	_FORCE_INLINE_ static double *get_float(Variant *p_variant) { return &p_variant->_data._float; }
	_FORCE_INLINE_ static const double *get_float(const Variant *p_variant) { return &p_variant->_data._float; }
	_FORCE_INLINE_ static String *get_string(Variant *p_variant) { return reinterpret_cast<String *>(p_variant->_data._mem); }
	_FORCE_INLINE_ static const String *get_string(const Variant *p_variant) { return reinterpret_cast<const String *>(p_variant->_data._mem); }

	// Math types.
	_FORCE_INLINE_ static Vector2 *get_vector2(Variant *p_variant) { return reinterpret_cast<Vector2 *>(p_variant->_data._mem); }
	_FORCE_INLINE_ static const Vector2 *get_vector2(const Variant *p_variant) { return reinterpret_cast<const Vector2 *>(p_variant->_data._mem); }
	_FORCE_INLINE_ static Vector2i *get_vector2i(Variant *p_variant) { return reinterpret_cast<Vector2i *>(p_variant->_data._mem); }
	_FORCE_INLINE_ static const Vector2i *get_vector2i(const Variant *p_variant) { return reinterpret_cast<const Vector2i *>(p_variant->_data._mem); }
	_FORCE_INLINE_ static Rect2 *get_rect2(Variant *p_variant) { return reinterpret_cast<Rect2 *>(p_variant->_data._mem); }
	_FORCE_INLINE_ static const Rect2 *get_rect2(const Variant *p_variant) { return reinterpret_cast<const Rect2 *>(p_variant->_data._mem); }
	_FORCE_INLINE_ static Rect2i *get_rect2i(Variant *p_variant) { return reinterpret_cast<Rect2i *>(p_variant->_data._mem); }
	_FORCE_INLINE_ static const Rect2i *get_rect2i(const Variant *p_variant) { return reinterpret_cast<const Rect2i *>(p_variant->_data._mem); }
	_FORCE_INLINE_ static Vector3 *get_vector3(Variant *p_variant) { return reinterpret_cast<Vector3 *>(p_variant->_data._mem); }
	_FORCE_INLINE_ static const Vector3 *get_vector3(const Variant *p_variant) { return reinterpret_cast<const Vector3 *>(p_variant->_data._mem); }
	_FORCE_INLINE_ static Vector3i *get_vector3i(Variant *p_variant) { return reinterpret_cast<Vector3i *>(p_variant->_data._mem); }
	_FORCE_INLINE_ static const Vector3i *get_vector3i(const Variant *p_variant) { return reinterpret_cast<const Vector3i *>(p_variant->_data._mem); }
	_FORCE_INLINE_ static Vector4 *get_vector4(Variant *p_variant) { return reinterpret_cast<Vector4 *>(p_variant->_data._mem); }
	_FORCE_INLINE_ static const Vector4 *get_vector4(const Variant *p_variant) { return reinterpret_cast<const Vector4 *>(p_variant->_data._mem); }
	_FORCE_INLINE_ static Vector4i *get_vector4i(Variant *p_variant) { return reinterpret_cast<Vector4i *>(p_variant->_data._mem); }
	_FORCE_INLINE_ static const Vector4i *get_vector4i(const Variant *p_variant) { return reinterpret_cast<const Vector4i *>(p_variant->_data._mem); }
	_FORCE_INLINE_ static Transform2D *get_transform2d(Variant *p_variant) { return p_variant->_data._transform2d; }
	_FORCE_INLINE_ static const Transform2D *get_transform2d(const Variant *p_variant) { return p_variant->_data._transform2d; }
	_FORCE_INLINE_ static Plane *get_plane(Variant *p_variant) { return reinterpret_cast<Plane *>(p_variant->_data._mem); }
	_FORCE_INLINE_ static const Plane *get_plane(const Variant *p_variant) { return reinterpret_cast<const Plane *>(p_variant->_data._mem); }
	_FORCE_INLINE_ static Quaternion *get_quaternion(Variant *p_variant) { return reinterpret_cast<Quaternion *>(p_variant->_data._mem); }
	_FORCE_INLINE_ static const Quaternion *get_quaternion(const Variant *p_variant) { return reinterpret_cast<const Quaternion *>(p_variant->_data._mem); }
	_FORCE_INLINE_ static ::AABB *get_aabb(Variant *p_variant) { return p_variant->_data._aabb; }
	_FORCE_INLINE_ static const ::AABB *get_aabb(const Variant *p_variant) { return p_variant->_data._aabb; }
	_FORCE_INLINE_ static Basis *get_basis(Variant *p_variant) { return p_variant->_data._basis; }
	_FORCE_INLINE_ static const Basis *get_basis(const Variant *p_variant) { return p_variant->_data._basis; }
	_FORCE_INLINE_ static Transform3D *get_transform(Variant *p_variant) { return p_variant->_data._transform3d; }
	_FORCE_INLINE_ static const Transform3D *get_transform(const Variant *p_variant) { return p_variant->_data._transform3d; }
	_FORCE_INLINE_ static Projection *get_projection(Variant *p_variant) { return p_variant->_data._projection; }
	_FORCE_INLINE_ static const Projection *get_projection(const Variant *p_variant) { return p_variant->_data._projection; }

	// Misc types.
	_FORCE_INLINE_ static Color *get_color(Variant *p_variant) { return reinterpret_cast<Color *>(p_variant->_data._mem); }
	_FORCE_INLINE_ static const Color *get_color(const Variant *p_variant) { return reinterpret_cast<const Color *>(p_variant->_data._mem); }
	_FORCE_INLINE_ static StringName *get_string_name(Variant *p_variant) { return reinterpret_cast<StringName *>(p_variant->_data._mem); }
	_FORCE_INLINE_ static const StringName *get_string_name(const Variant *p_variant) { return reinterpret_cast<const StringName *>(p_variant->_data._mem); }
	_FORCE_INLINE_ static NodePath *get_node_path(Variant *p_variant) { return reinterpret_cast<NodePath *>(p_variant->_data._mem); }
	_FORCE_INLINE_ static const NodePath *get_node_path(const Variant *p_variant) { return reinterpret_cast<const NodePath *>(p_variant->_data._mem); }
	_FORCE_INLINE_ static ::RID *get_rid(Variant *p_variant) { return reinterpret_cast<::RID *>(p_variant->_data._mem); }
	_FORCE_INLINE_ static const ::RID *get_rid(const Variant *p_variant) { return reinterpret_cast<const ::RID *>(p_variant->_data._mem); }
	_FORCE_INLINE_ static Callable *get_callable(Variant *p_variant) { return reinterpret_cast<Callable *>(p_variant->_data._mem); }
	_FORCE_INLINE_ static const Callable *get_callable(const Variant *p_variant) { return reinterpret_cast<const Callable *>(p_variant->_data._mem); }
	_FORCE_INLINE_ static Signal *get_signal(Variant *p_variant) { return reinterpret_cast<Signal *>(p_variant->_data._mem); }
	_FORCE_INLINE_ static const Signal *get_signal(const Variant *p_variant) { return reinterpret_cast<const Signal *>(p_variant->_data._mem); }
	_FORCE_INLINE_ static Dictionary *get_dictionary(Variant *p_variant) { return reinterpret_cast<Dictionary *>(p_variant->_data._mem); }
	_FORCE_INLINE_ static const Dictionary *get_dictionary(const Variant *p_variant) { return reinterpret_cast<const Dictionary *>(p_variant->_data._mem); }
	_FORCE_INLINE_ static Array *get_array(Variant *p_variant) { return reinterpret_cast<Array *>(p_variant->_data._mem); }
	_FORCE_INLINE_ static const Array *get_array(const Variant *p_variant) { return reinterpret_cast<const Array *>(p_variant->_data._mem); }

	// Typed arrays.
	_FORCE_INLINE_ static PackedByteArray *get_byte_array(Variant *p_variant) { return &static_cast<Variant::PackedArrayRef<uint8_t> *>(p_variant->_data.packed_array)->array; }
	_FORCE_INLINE_ static const PackedByteArray *get_byte_array(const Variant *p_variant) { return &static_cast<const Variant::PackedArrayRef<uint8_t> *>(p_variant->_data.packed_array)->array; }
	_FORCE_INLINE_ static PackedInt32Array *get_int32_array(Variant *p_variant) { return &static_cast<Variant::PackedArrayRef<int32_t> *>(p_variant->_data.packed_array)->array; }
	_FORCE_INLINE_ static const PackedInt32Array *get_int32_array(const Variant *p_variant) { return &static_cast<const Variant::PackedArrayRef<int32_t> *>(p_variant->_data.packed_array)->array; }
	_FORCE_INLINE_ static PackedInt64Array *get_int64_array(Variant *p_variant) { return &static_cast<Variant::PackedArrayRef<int64_t> *>(p_variant->_data.packed_array)->array; }
	_FORCE_INLINE_ static const PackedInt64Array *get_int64_array(const Variant *p_variant) { return &static_cast<const Variant::PackedArrayRef<int64_t> *>(p_variant->_data.packed_array)->array; }
	_FORCE_INLINE_ static PackedFloat32Array *get_float32_array(Variant *p_variant) { return &static_cast<Variant::PackedArrayRef<float> *>(p_variant->_data.packed_array)->array; }
	_FORCE_INLINE_ static const PackedFloat32Array *get_float32_array(const Variant *p_variant) { return &static_cast<const Variant::PackedArrayRef<float> *>(p_variant->_data.packed_array)->array; }
	_FORCE_INLINE_ static PackedFloat64Array *get_float64_array(Variant *p_variant) { return &static_cast<Variant::PackedArrayRef<double> *>(p_variant->_data.packed_array)->array; }
	_FORCE_INLINE_ static const PackedFloat64Array *get_float64_array(const Variant *p_variant) { return &static_cast<const Variant::PackedArrayRef<double> *>(p_variant->_data.packed_array)->array; }
	_FORCE_INLINE_ static PackedStringArray *get_string_array(Variant *p_variant) { return &static_cast<Variant::PackedArrayRef<String> *>(p_variant->_data.packed_array)->array; }
	_FORCE_INLINE_ static const PackedStringArray *get_string_array(const Variant *p_variant) { return &static_cast<const Variant::PackedArrayRef<String> *>(p_variant->_data.packed_array)->array; }
	_FORCE_INLINE_ static PackedVector2Array *get_vector2_array(Variant *p_variant) { return &static_cast<Variant::PackedArrayRef<Vector2> *>(p_variant->_data.packed_array)->array; }
	_FORCE_INLINE_ static const PackedVector2Array *get_vector2_array(const Variant *p_variant) { return &static_cast<const Variant::PackedArrayRef<Vector2> *>(p_variant->_data.packed_array)->array; }
	_FORCE_INLINE_ static PackedVector3Array *get_vector3_array(Variant *p_variant) { return &static_cast<Variant::PackedArrayRef<Vector3> *>(p_variant->_data.packed_array)->array; }
	_FORCE_INLINE_ static const PackedVector3Array *get_vector3_array(const Variant *p_variant) { return &static_cast<const Variant::PackedArrayRef<Vector3> *>(p_variant->_data.packed_array)->array; }
	_FORCE_INLINE_ static PackedColorArray *get_color_array(Variant *p_variant) { return &static_cast<Variant::PackedArrayRef<Color> *>(p_variant->_data.packed_array)->array; }
	_FORCE_INLINE_ static const PackedColorArray *get_color_array(const Variant *p_variant) { return &static_cast<const Variant::PackedArrayRef<Color> *>(p_variant->_data.packed_array)->array; }
	_FORCE_INLINE_ static PackedVector4Array *get_vector4_array(Variant *p_variant) { return &static_cast<Variant::PackedArrayRef<Vector4> *>(p_variant->_data.packed_array)->array; }
	_FORCE_INLINE_ static const PackedVector4Array *get_vector4_array(const Variant *p_variant) { return &static_cast<const Variant::PackedArrayRef<Vector4> *>(p_variant->_data.packed_array)->array; }

	_FORCE_INLINE_ static Object **get_object(Variant *p_variant) { return (Object **)&p_variant->_get_obj().obj; }
	_FORCE_INLINE_ static const Object **get_object(const Variant *p_variant) { return (const Object **)&p_variant->_get_obj().obj; }

	_FORCE_INLINE_ static const ObjectID get_object_id(const Variant *p_variant) { return p_variant->_get_obj().id; }

	template <typename T>
	_FORCE_INLINE_ static void init_generic(Variant *r_variant) {
		r_variant->type = GetTypeInfo<T>::VARIANT_TYPE;
	}

	// Should be in the same order as Variant::Type for consistency.
	// Those primitive and vector types don't need an `init_` method:
	// Nil, bool, float, Vector2/i, Rect2/i, Vector3/i, Plane, RID.
	// Object is a special case, handled via `object_reset_data`.
	_FORCE_INLINE_ static void init_string(Variant *r_variant) {
		memnew_placement(r_variant->_data._mem, String);
		r_variant->type = Variant::STRING;
	}
	_FORCE_INLINE_ static void init_transform2d(Variant *r_variant) {
		r_variant->_data._transform2d = VariantPools::alloc<Transform2D>();
		memnew_placement(r_variant->_data._transform2d, Transform2D);
		r_variant->type = Variant::TRANSFORM2D;
	}
	_FORCE_INLINE_ static void init_quaternion(Variant *r_variant) {
		memnew_placement(r_variant->_data._mem, Quaternion);
		r_variant->type = Variant::QUATERNION;
	}
	_FORCE_INLINE_ static void init_aabb(Variant *r_variant) {
		r_variant->_data._aabb = VariantPools::alloc<AABB>();
		memnew_placement(r_variant->_data._aabb, AABB);
		r_variant->type = Variant::AABB;
	}
	_FORCE_INLINE_ static void init_basis(Variant *r_variant) {
		r_variant->_data._basis = VariantPools::alloc<Basis>();
		memnew_placement(r_variant->_data._basis, Basis);
		r_variant->type = Variant::BASIS;
	}
	_FORCE_INLINE_ static void init_transform3d(Variant *r_variant) {
		r_variant->_data._transform3d = VariantPools::alloc<Transform3D>();
		memnew_placement(r_variant->_data._transform3d, Transform3D);
		r_variant->type = Variant::TRANSFORM3D;
	}
	_FORCE_INLINE_ static void init_projection(Variant *r_variant) {
		r_variant->_data._projection = VariantPools::alloc<Projection>();
		memnew_placement(r_variant->_data._projection, Projection);
		r_variant->type = Variant::PROJECTION;
	}
	_FORCE_INLINE_ static void init_color(Variant *r_variant) {
		memnew_placement(r_variant->_data._mem, Color);
		r_variant->type = Variant::COLOR;
	}
	_FORCE_INLINE_ static void init_string_name(Variant *r_variant) {
		memnew_placement(r_variant->_data._mem, StringName);
		r_variant->type = Variant::STRING_NAME;
	}
	_FORCE_INLINE_ static void init_node_path(Variant *r_variant) {
		memnew_placement(r_variant->_data._mem, NodePath);
		r_variant->type = Variant::NODE_PATH;
	}
	_FORCE_INLINE_ static void init_callable(Variant *r_variant) {
		memnew_placement(r_variant->_data._mem, Callable);
		r_variant->type = Variant::CALLABLE;
	}
	_FORCE_INLINE_ static void init_signal(Variant *r_variant) {
		memnew_placement(r_variant->_data._mem, Signal);
		r_variant->type = Variant::SIGNAL;
	}
	_FORCE_INLINE_ static void init_dictionary(Variant *r_variant) {
		memnew_placement(r_variant->_data._mem, Dictionary);
		r_variant->type = Variant::DICTIONARY;
	}
	_FORCE_INLINE_ static void init_array(Variant *r_variant) {
		memnew_placement(r_variant->_data._mem, Array);
		r_variant->type = Variant::ARRAY;
	}
	_FORCE_INLINE_ static void init_byte_array(Variant *r_variant) {
		r_variant->_data.packed_array = Variant::PackedArrayRef<uint8_t>::create(Vector<uint8_t>());
		r_variant->type = Variant::PACKED_BYTE_ARRAY;
	}
	_FORCE_INLINE_ static void init_int32_array(Variant *r_variant) {
		r_variant->_data.packed_array = Variant::PackedArrayRef<int32_t>::create(Vector<int32_t>());
		r_variant->type = Variant::PACKED_INT32_ARRAY;
	}
	_FORCE_INLINE_ static void init_int64_array(Variant *r_variant) {
		r_variant->_data.packed_array = Variant::PackedArrayRef<int64_t>::create(Vector<int64_t>());
		r_variant->type = Variant::PACKED_INT64_ARRAY;
	}
	_FORCE_INLINE_ static void init_float32_array(Variant *r_variant) {
		r_variant->_data.packed_array = Variant::PackedArrayRef<float>::create(Vector<float>());
		r_variant->type = Variant::PACKED_FLOAT32_ARRAY;
	}
	_FORCE_INLINE_ static void init_float64_array(Variant *r_variant) {
		r_variant->_data.packed_array = Variant::PackedArrayRef<double>::create(Vector<double>());
		r_variant->type = Variant::PACKED_FLOAT64_ARRAY;
	}
	_FORCE_INLINE_ static void init_string_array(Variant *r_variant) {
		r_variant->_data.packed_array = Variant::PackedArrayRef<String>::create(Vector<String>());
		r_variant->type = Variant::PACKED_STRING_ARRAY;
	}
	_FORCE_INLINE_ static void init_vector2_array(Variant *r_variant) {
		r_variant->_data.packed_array = Variant::PackedArrayRef<Vector2>::create(Vector<Vector2>());
		r_variant->type = Variant::PACKED_VECTOR2_ARRAY;
	}
	_FORCE_INLINE_ static void init_vector3_array(Variant *r_variant) {
		r_variant->_data.packed_array = Variant::PackedArrayRef<Vector3>::create(Vector<Vector3>());
		r_variant->type = Variant::PACKED_VECTOR3_ARRAY;
	}
	_FORCE_INLINE_ static void init_color_array(Variant *r_variant) {
		r_variant->_data.packed_array = Variant::PackedArrayRef<Color>::create(Vector<Color>());
		r_variant->type = Variant::PACKED_COLOR_ARRAY;
	}
	_FORCE_INLINE_ static void init_vector4_array(Variant *r_variant) {
		r_variant->_data.packed_array = Variant::PackedArrayRef<Vector4>::create(Vector<Vector4>());
		r_variant->type = Variant::PACKED_VECTOR4_ARRAY;
	}
	_FORCE_INLINE_ static void init_object(Variant *r_variant) {
		object_reset_data(r_variant);
		r_variant->type = Variant::OBJECT;
	}

	_FORCE_INLINE_ static void clear(Variant *r_variant) {
		r_variant->clear();
	}

	_FORCE_INLINE_ static void object_assign(Variant *r_variant, const Variant *p_var_obj) {
		r_variant->_get_obj().ref(p_var_obj->_get_obj());
	}

	_FORCE_INLINE_ static void object_assign(Variant *r_variant, Object *p_object) {
		r_variant->_get_obj().ref_pointer(p_object);
	}

	_FORCE_INLINE_ static void object_assign(Variant *r_variant, const Object *p_object) {
		r_variant->_get_obj().ref_pointer(const_cast<Object *>(p_object));
	}

	template <typename T>
	_FORCE_INLINE_ static void object_assign(Variant *r_variant, const Ref<T> &p_ref) {
		r_variant->_get_obj().ref(p_ref);
	}

	// This should be used with extreme caution, as it does not increment the reference count in the
	// case where the object is `RefCounted`. You *must* ensure that the destructor of this
	// `Variant` is never called, and that the assigned object always outlives the `Variant`.
	_FORCE_INLINE_ static void object_assign_without_ref_unsafe(Variant *r_variant, Object *p_object) {
		r_variant->type = Variant::OBJECT;
		Variant::ObjData &obj_data = r_variant->_get_obj();
		obj_data.id = p_object->get_instance_id();
		obj_data.obj = p_object;
	}

	_FORCE_INLINE_ static void object_reset_data(Variant *r_variant) {
		r_variant->_get_obj() = Variant::ObjData();
	}

	_FORCE_INLINE_ static void update_object_id(Variant *r_variant) {
		const Object *o = r_variant->_get_obj().obj;
		if (o) {
			r_variant->_get_obj().id = o->get_instance_id();
		}
	}

	_FORCE_INLINE_ static void *get_opaque_pointer(Variant *p_variant) {
		switch (p_variant->type) {
			case Variant::NIL:
				return nullptr;
			case Variant::BOOL:
				return get_bool(p_variant);
			case Variant::INT:
				return get_int(p_variant);
			case Variant::FLOAT:
				return get_float(p_variant);
			case Variant::STRING:
				return get_string(p_variant);
			case Variant::VECTOR2:
				return get_vector2(p_variant);
			case Variant::VECTOR2I:
				return get_vector2i(p_variant);
			case Variant::VECTOR3:
				return get_vector3(p_variant);
			case Variant::VECTOR3I:
				return get_vector3i(p_variant);
			case Variant::VECTOR4:
				return get_vector4(p_variant);
			case Variant::VECTOR4I:
				return get_vector4i(p_variant);
			case Variant::RECT2:
				return get_rect2(p_variant);
			case Variant::RECT2I:
				return get_rect2i(p_variant);
			case Variant::TRANSFORM3D:
				return get_transform(p_variant);
			case Variant::PROJECTION:
				return get_projection(p_variant);
			case Variant::TRANSFORM2D:
				return get_transform2d(p_variant);
			case Variant::QUATERNION:
				return get_quaternion(p_variant);
			case Variant::PLANE:
				return get_plane(p_variant);
			case Variant::BASIS:
				return get_basis(p_variant);
			case Variant::AABB:
				return get_aabb(p_variant);
			case Variant::COLOR:
				return get_color(p_variant);
			case Variant::STRING_NAME:
				return get_string_name(p_variant);
			case Variant::NODE_PATH:
				return get_node_path(p_variant);
			case Variant::RID:
				return get_rid(p_variant);
			case Variant::CALLABLE:
				return get_callable(p_variant);
			case Variant::SIGNAL:
				return get_signal(p_variant);
			case Variant::DICTIONARY:
				return get_dictionary(p_variant);
			case Variant::ARRAY:
				return get_array(p_variant);
			case Variant::PACKED_BYTE_ARRAY:
				return get_byte_array(p_variant);
			case Variant::PACKED_INT32_ARRAY:
				return get_int32_array(p_variant);
			case Variant::PACKED_INT64_ARRAY:
				return get_int64_array(p_variant);
			case Variant::PACKED_FLOAT32_ARRAY:
				return get_float32_array(p_variant);
			case Variant::PACKED_FLOAT64_ARRAY:
				return get_float64_array(p_variant);
			case Variant::PACKED_STRING_ARRAY:
				return get_string_array(p_variant);
			case Variant::PACKED_VECTOR2_ARRAY:
				return get_vector2_array(p_variant);
			case Variant::PACKED_VECTOR3_ARRAY:
				return get_vector3_array(p_variant);
			case Variant::PACKED_COLOR_ARRAY:
				return get_color_array(p_variant);
			case Variant::PACKED_VECTOR4_ARRAY:
				return get_vector4_array(p_variant);
			case Variant::OBJECT:
				return get_object(p_variant);
			case Variant::VARIANT_MAX:
				ERR_FAIL_V(nullptr);
		}
		ERR_FAIL_V(nullptr);
	}

	_FORCE_INLINE_ static const void *get_opaque_pointer(const Variant *p_variant) {
		switch (p_variant->type) {
			case Variant::NIL:
				return nullptr;
			case Variant::BOOL:
				return get_bool(p_variant);
			case Variant::INT:
				return get_int(p_variant);
			case Variant::FLOAT:
				return get_float(p_variant);
			case Variant::STRING:
				return get_string(p_variant);
			case Variant::VECTOR2:
				return get_vector2(p_variant);
			case Variant::VECTOR2I:
				return get_vector2i(p_variant);
			case Variant::VECTOR3:
				return get_vector3(p_variant);
			case Variant::VECTOR3I:
				return get_vector3i(p_variant);
			case Variant::VECTOR4:
				return get_vector4(p_variant);
			case Variant::VECTOR4I:
				return get_vector4i(p_variant);
			case Variant::RECT2:
				return get_rect2(p_variant);
			case Variant::RECT2I:
				return get_rect2i(p_variant);
			case Variant::TRANSFORM3D:
				return get_transform(p_variant);
			case Variant::PROJECTION:
				return get_projection(p_variant);
			case Variant::TRANSFORM2D:
				return get_transform2d(p_variant);
			case Variant::QUATERNION:
				return get_quaternion(p_variant);
			case Variant::PLANE:
				return get_plane(p_variant);
			case Variant::BASIS:
				return get_basis(p_variant);
			case Variant::AABB:
				return get_aabb(p_variant);
			case Variant::COLOR:
				return get_color(p_variant);
			case Variant::STRING_NAME:
				return get_string_name(p_variant);
			case Variant::NODE_PATH:
				return get_node_path(p_variant);
			case Variant::RID:
				return get_rid(p_variant);
			case Variant::CALLABLE:
				return get_callable(p_variant);
			case Variant::SIGNAL:
				return get_signal(p_variant);
			case Variant::DICTIONARY:
				return get_dictionary(p_variant);
			case Variant::ARRAY:
				return get_array(p_variant);
			case Variant::PACKED_BYTE_ARRAY:
				return get_byte_array(p_variant);
			case Variant::PACKED_INT32_ARRAY:
				return get_int32_array(p_variant);
			case Variant::PACKED_INT64_ARRAY:
				return get_int64_array(p_variant);
			case Variant::PACKED_FLOAT32_ARRAY:
				return get_float32_array(p_variant);
			case Variant::PACKED_FLOAT64_ARRAY:
				return get_float64_array(p_variant);
			case Variant::PACKED_STRING_ARRAY:
				return get_string_array(p_variant);
			case Variant::PACKED_VECTOR2_ARRAY:
				return get_vector2_array(p_variant);
			case Variant::PACKED_VECTOR3_ARRAY:
				return get_vector3_array(p_variant);
			case Variant::PACKED_COLOR_ARRAY:
				return get_color_array(p_variant);
			case Variant::PACKED_VECTOR4_ARRAY:
				return get_vector4_array(p_variant);
			case Variant::OBJECT:
				return get_object(p_variant);
			case Variant::VARIANT_MAX:
				ERR_FAIL_V(nullptr);
		}
		ERR_FAIL_V(nullptr);
	}

	// Used internally in GDExtension and Godot's binding system when converting to Variant
	// from values that may include RequiredParam<T> or RequiredResult<T>.
	template <typename T>
	_FORCE_INLINE_ static Variant make(const T &p_variant) {
		return Variant(p_variant);
	}
	template <typename T>
	_FORCE_INLINE_ static Variant make(const GDExtensionPtr<T> &p_variant) {
		return p_variant.operator Variant();
	}
	template <typename T>
	_FORCE_INLINE_ static Variant make(const RequiredParam<T> &p_variant) {
		return Variant(p_variant._internal_ptr_dont_use());
	}
	template <typename T>
	_FORCE_INLINE_ static Variant make(const RequiredResult<T> &p_variant) {
		return Variant(p_variant._internal_ptr_dont_use());
	}
};

template <typename T, typename = void>
struct VariantInternalAccessor;

template <typename T>
struct VariantInternalAccessor<T, std::enable_if_t<!std::is_same_v<T, GetSimpleTypeT<T>>>> : VariantInternalAccessor<GetSimpleTypeT<T>> {};

template <typename T>
struct _VariantInternalAccessorLocal {
	using declared_when_native_type = void;
	static constexpr bool is_local = true;
	static _FORCE_INLINE_ T &get(Variant *p_variant) { return *reinterpret_cast<T *>(p_variant->_data._mem); }
	static _FORCE_INLINE_ const T &get(const Variant *p_variant) { return *reinterpret_cast<const T *>(p_variant->_data._mem); }
	static _FORCE_INLINE_ void set(Variant *r_variant, T p_value) { *reinterpret_cast<T *>(r_variant->_data._mem) = std::move(p_value); }
};

template <typename T>
struct _VariantInternalAccessorElsewhere {
	using declared_when_native_type = void;
	static _FORCE_INLINE_ T &get(Variant *p_variant) { return *reinterpret_cast<T *>(p_variant->_data._ptr); }
	static _FORCE_INLINE_ const T &get(const Variant *p_variant) { return *reinterpret_cast<const T *>(p_variant->_data._ptr); }
	static _FORCE_INLINE_ void set(Variant *r_variant, T p_value) { *reinterpret_cast<T *>(r_variant->_data._ptr) = std::move(p_value); }
};

template <typename T>
struct _VariantInternalAccessorPackedArrayRef {
	using declared_when_native_type = void;
	static _FORCE_INLINE_ Vector<T> &get(Variant *p_variant) { return static_cast<Variant::PackedArrayRef<T> *>(p_variant->_data.packed_array)->array; }
	static _FORCE_INLINE_ const Vector<T> &get(const Variant *p_variant) { return static_cast<const Variant::PackedArrayRef<T> *>(p_variant->_data.packed_array)->array; }
	static _FORCE_INLINE_ void set(Variant *r_variant, Vector<T> p_value) { static_cast<Variant::PackedArrayRef<T> *>(r_variant->_data.packed_array)->array = std::move(p_value); }
};

template <typename T>
struct VariantInternalAccessor<T *> {
	static _FORCE_INLINE_ T *get(const Variant *p_variant) { return const_cast<T *>(static_cast<const T *>(*VariantInternal::get_object(p_variant))); }
	static _FORCE_INLINE_ void set(Variant *r_variant, const T *p_value) { VariantInternal::object_assign(r_variant, p_value); }
};

template <typename T>
struct VariantInternalAccessor<const T *> {
	static _FORCE_INLINE_ const T *get(const Variant *p_variant) { return static_cast<const T *>(*VariantInternal::get_object(p_variant)); }
	static _FORCE_INLINE_ void set(Variant *r_variant, const T *p_value) { VariantInternal::object_assign(r_variant, p_value); }
};

template <>
struct VariantInternalAccessor<IPAddress> {
	static _FORCE_INLINE_ IPAddress get(const Variant *p_variant) { return IPAddress(*VariantInternal::get_string(p_variant)); }
	static _FORCE_INLINE_ void set(Variant *r_variant, IPAddress p_value) { *VariantInternal::get_string(r_variant) = String(p_value); }
};

template <typename T>
struct VariantInternalAccessor<TypedArray<T>> {
	static _FORCE_INLINE_ TypedArray<T> get(const Variant *p_variant) { return TypedArray<T>(*VariantInternal::get_array(p_variant)); }
	static _FORCE_INLINE_ void set(Variant *r_variant, const TypedArray<T> &p_array) { *VariantInternal::get_array(r_variant) = Array(p_array); }
};

template <typename K, typename V>
struct VariantInternalAccessor<TypedDictionary<K, V>> {
	static _FORCE_INLINE_ TypedDictionary<K, V> get(const Variant *p_variant) { return TypedDictionary<K, V>(*VariantInternal::get_dictionary(p_variant)); }
	static _FORCE_INLINE_ void set(Variant *r_variant, const TypedDictionary<K, V> &p_dictionary) { *VariantInternal::get_dictionary(r_variant) = Dictionary(p_dictionary); }
};

template <>
struct VariantInternalAccessor<Object *> {
	static _FORCE_INLINE_ Object *get(const Variant *p_variant) { return const_cast<Object *>(*VariantInternal::get_object(p_variant)); }
	static _FORCE_INLINE_ void set(Variant *r_variant, const Object *p_value) { VariantInternal::object_assign(r_variant, p_value); }
};

template <typename T>
struct VariantInternalAccessor<Ref<T>> {
	static _FORCE_INLINE_ Ref<T> get(const Variant *p_variant) { return Ref<T>(const_cast<Object *>(*VariantInternal::get_object(p_variant))); }
	static _FORCE_INLINE_ void set(Variant *r_variant, const Ref<T> &p_ref) { VariantInternal::object_assign(r_variant, p_ref); }
};

template <class T>
struct VariantInternalAccessor<RequiredParam<T>> {
	static _FORCE_INLINE_ RequiredParam<T> get(const Variant *p_variant) { return RequiredParam<T>(Object::cast_to<T>(const_cast<Object *>(*VariantInternal::get_object(p_variant)))); }
	static _FORCE_INLINE_ void set(Variant *r_variant, const RequiredParam<T> &p_value) { VariantInternal::object_assign(r_variant, p_value.ptr()); }
};

template <class T>
struct VariantInternalAccessor<const RequiredParam<T> &> {
	static _FORCE_INLINE_ RequiredParam<T> get(const Variant *p_variant) { return RequiredParam<T>(Object::cast_to<T>(*VariantInternal::get_object(p_variant))); }
	static _FORCE_INLINE_ void set(Variant *r_variant, const RequiredParam<T> &p_value) { VariantInternal::object_assign(r_variant, p_value.ptr()); }
};

template <class T>
struct VariantInternalAccessor<RequiredResult<T>> {
	static _FORCE_INLINE_ RequiredResult<T> get(const Variant *p_variant) { return RequiredResult<T>(Object::cast_to<T>(const_cast<Object *>(*VariantInternal::get_object(p_variant)))); }
	static _FORCE_INLINE_ void set(Variant *r_variant, const RequiredResult<T> &p_value) { VariantInternal::object_assign(r_variant, p_value.ptr()); }
};

template <class T>
struct VariantInternalAccessor<const RequiredResult<T> &> {
	static _FORCE_INLINE_ RequiredResult<T> get(const Variant *p_variant) { return RequiredResult<T>(Object::cast_to<T>(*VariantInternal::get_object(p_variant))); }
	static _FORCE_INLINE_ void set(Variant *r_variant, const RequiredResult<T> &p_value) { VariantInternal::object_assign(r_variant, p_value.ptr()); }
};

template <>
struct VariantInternalAccessor<Variant> {
	static _FORCE_INLINE_ Variant &get(Variant *p_variant) { return *p_variant; }
	static _FORCE_INLINE_ const Variant &get(const Variant *p_variant) { return *p_variant; }
	static _FORCE_INLINE_ void set(Variant *r_variant, const Variant &p_value) { *r_variant = p_value; }
};

template <>
struct VariantInternalAccessor<Vector<Variant>> {
	static _FORCE_INLINE_ Vector<Variant> get(const Variant *p_variant) {
		Vector<Variant> ret;
		int s = VariantInternal::get_array(p_variant)->size();
		ret.resize(s);
		for (int i = 0; i < s; i++) {
			ret.write[i] = VariantInternal::get_array(p_variant)->get(i);
		}

		return ret;
	}
	static _FORCE_INLINE_ void set(Variant *r_variant, const Vector<Variant> &p_value) {
		int s = p_value.size();
		VariantInternal::get_array(r_variant)->resize(s);
		for (int i = 0; i < s; i++) {
			VariantInternal::get_array(r_variant)->set(i, p_value[i]);
		}
	}
};

template <>
struct VariantInternalAccessor<bool> : _VariantInternalAccessorLocal<bool> {};

template <>
struct VariantInternalAccessor<int64_t> : _VariantInternalAccessorLocal<int64_t> {};

template <>
struct VariantInternalAccessor<double> : _VariantInternalAccessorLocal<double> {};

template <>
struct VariantInternalAccessor<String> : _VariantInternalAccessorLocal<String> {};

template <>
struct VariantInternalAccessor<Vector2> : _VariantInternalAccessorLocal<Vector2> {};

template <>
struct VariantInternalAccessor<Vector2i> : _VariantInternalAccessorLocal<Vector2i> {};

template <>
struct VariantInternalAccessor<Rect2> : _VariantInternalAccessorLocal<Rect2> {};

template <>
struct VariantInternalAccessor<Rect2i> : _VariantInternalAccessorLocal<Rect2i> {};

template <>
struct VariantInternalAccessor<Vector3> : _VariantInternalAccessorLocal<Vector3> {};

template <>
struct VariantInternalAccessor<Vector3i> : _VariantInternalAccessorLocal<Vector3i> {};

template <>
struct VariantInternalAccessor<Vector4> : _VariantInternalAccessorLocal<Vector4> {};

template <>
struct VariantInternalAccessor<Vector4i> : _VariantInternalAccessorLocal<Vector4i> {};

template <>
struct VariantInternalAccessor<Transform2D> : _VariantInternalAccessorElsewhere<Transform2D> {};

template <>
struct VariantInternalAccessor<Transform3D> : _VariantInternalAccessorElsewhere<Transform3D> {};

template <>
struct VariantInternalAccessor<Projection> : _VariantInternalAccessorElsewhere<Projection> {};

template <>
struct VariantInternalAccessor<Plane> : _VariantInternalAccessorLocal<Plane> {};

template <>
struct VariantInternalAccessor<Quaternion> : _VariantInternalAccessorLocal<Quaternion> {};

template <>
struct VariantInternalAccessor<::AABB> : _VariantInternalAccessorElsewhere<::AABB> {};

template <>
struct VariantInternalAccessor<Basis> : _VariantInternalAccessorElsewhere<Basis> {};

template <>
struct VariantInternalAccessor<Color> : _VariantInternalAccessorLocal<Color> {};

template <>
struct VariantInternalAccessor<StringName> : _VariantInternalAccessorLocal<StringName> {};

template <>
struct VariantInternalAccessor<NodePath> : _VariantInternalAccessorLocal<NodePath> {};

template <>
struct VariantInternalAccessor<::RID> : _VariantInternalAccessorLocal<::RID> {};

// template <>
// struct VariantInternalAccessor<Variant::ObjData> : _VariantInternalAccessorLocal<Variant::ObjData> {};

template <>
struct VariantInternalAccessor<Callable> : _VariantInternalAccessorLocal<Callable> {};

template <>
struct VariantInternalAccessor<Signal> : _VariantInternalAccessorLocal<Signal> {};

template <>
struct VariantInternalAccessor<Dictionary> : _VariantInternalAccessorLocal<Dictionary> {};

template <>
struct VariantInternalAccessor<Array> : _VariantInternalAccessorLocal<Array> {};

template <>
struct VariantInternalAccessor<PackedByteArray> : _VariantInternalAccessorPackedArrayRef<uint8_t> {};

template <>
struct VariantInternalAccessor<PackedInt32Array> : _VariantInternalAccessorPackedArrayRef<int32_t> {};

template <>
struct VariantInternalAccessor<PackedInt64Array> : _VariantInternalAccessorPackedArrayRef<int64_t> {};

template <>
struct VariantInternalAccessor<PackedFloat32Array> : _VariantInternalAccessorPackedArrayRef<float> {};

template <>
struct VariantInternalAccessor<PackedFloat64Array> : _VariantInternalAccessorPackedArrayRef<double> {};

template <>
struct VariantInternalAccessor<PackedStringArray> : _VariantInternalAccessorPackedArrayRef<String> {};

template <>
struct VariantInternalAccessor<PackedVector2Array> : _VariantInternalAccessorPackedArrayRef<Vector2> {};

template <>
struct VariantInternalAccessor<PackedVector3Array> : _VariantInternalAccessorPackedArrayRef<Vector3> {};

template <>
struct VariantInternalAccessor<PackedColorArray> : _VariantInternalAccessorPackedArrayRef<Color> {};

template <>
struct VariantInternalAccessor<PackedVector4Array> : _VariantInternalAccessorPackedArrayRef<Vector4> {};

template <typename T, typename = std::void_t<>>
struct IsVariantType : std::false_type {};

template <typename T>
struct IsVariantType<T, std::void_t<typename VariantInternalAccessor<T>::declared_when_native_type>> : std::true_type {};

template <typename T>
constexpr bool IsVariantTypeT = IsVariantType<T>::value;

template <typename T, typename S>
struct _VariantInternalAccessorConvert {
	static _FORCE_INLINE_ T get(const Variant *p_variant) {
		return T(VariantInternalAccessor<S>::get(p_variant));
	}
	static _FORCE_INLINE_ void set(Variant *r_variant, const T p_value) {
		VariantInternalAccessor<S>::get(r_variant) = S(std::move(p_value));
	}
};

// Integer types.
template <typename T>
struct VariantInternalAccessor<T, std::enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, bool> && !std::is_same_v<T, int64_t>>> : _VariantInternalAccessorConvert<T, int64_t> {};
template <typename T>
struct VariantInternalAccessor<T, std::enable_if_t<std::is_enum_v<T>>> : _VariantInternalAccessorConvert<T, int64_t> {};
template <typename T>
struct VariantInternalAccessor<BitField<T>, std::enable_if_t<std::is_enum_v<T>>> : _VariantInternalAccessorConvert<BitField<T>, int64_t> {};

template <>
struct VariantInternalAccessor<ObjectID> : _VariantInternalAccessorConvert<ObjectID, int64_t> {};

// Float types.
template <>
struct VariantInternalAccessor<float> : _VariantInternalAccessorConvert<float, double> {};

template <typename T, typename = void>
struct VariantInitializer {
	static _FORCE_INLINE_ void init(Variant *r_variant) { VariantInternal::init_generic<T>(r_variant); }
};

template <typename T>
struct VariantInitializer<T, std::enable_if_t<VariantInternalAccessor<T>::is_local>> {
	static _FORCE_INLINE_ void init(Variant *r_variant) {
		memnew_placement(&VariantInternalAccessor<T>::get(r_variant), T());
		VariantInternal::set_type(*r_variant, GetTypeInfo<T>::VARIANT_TYPE);
	}
};

template <>
struct VariantInitializer<Transform2D> {
	static _FORCE_INLINE_ void init(Variant *r_variant) { VariantInternal::init_transform2d(r_variant); }
};

template <>
struct VariantInitializer<AABB> {
	static _FORCE_INLINE_ void init(Variant *r_variant) { VariantInternal::init_aabb(r_variant); }
};

template <>
struct VariantInitializer<Basis> {
	static _FORCE_INLINE_ void init(Variant *r_variant) { VariantInternal::init_basis(r_variant); }
};

template <>
struct VariantInitializer<Transform3D> {
	static _FORCE_INLINE_ void init(Variant *r_variant) { VariantInternal::init_transform3d(r_variant); }
};

template <>
struct VariantInitializer<Projection> {
	static _FORCE_INLINE_ void init(Variant *r_variant) { VariantInternal::init_projection(r_variant); }
};

template <>
struct VariantInitializer<PackedByteArray> {
	static _FORCE_INLINE_ void init(Variant *r_variant) { VariantInternal::init_byte_array(r_variant); }
};

template <>
struct VariantInitializer<PackedInt32Array> {
	static _FORCE_INLINE_ void init(Variant *r_variant) { VariantInternal::init_int32_array(r_variant); }
};

template <>
struct VariantInitializer<PackedInt64Array> {
	static _FORCE_INLINE_ void init(Variant *r_variant) { VariantInternal::init_int64_array(r_variant); }
};

template <>
struct VariantInitializer<PackedFloat32Array> {
	static _FORCE_INLINE_ void init(Variant *r_variant) { VariantInternal::init_float32_array(r_variant); }
};

template <>
struct VariantInitializer<PackedFloat64Array> {
	static _FORCE_INLINE_ void init(Variant *r_variant) { VariantInternal::init_float64_array(r_variant); }
};

template <>
struct VariantInitializer<PackedStringArray> {
	static _FORCE_INLINE_ void init(Variant *r_variant) { VariantInternal::init_string_array(r_variant); }
};

template <>
struct VariantInitializer<PackedVector2Array> {
	static _FORCE_INLINE_ void init(Variant *r_variant) { VariantInternal::init_vector2_array(r_variant); }
};

template <>
struct VariantInitializer<PackedVector3Array> {
	static _FORCE_INLINE_ void init(Variant *r_variant) { VariantInternal::init_vector3_array(r_variant); }
};

template <>
struct VariantInitializer<PackedColorArray> {
	static _FORCE_INLINE_ void init(Variant *r_variant) { VariantInternal::init_color_array(r_variant); }
};

template <>
struct VariantInitializer<PackedVector4Array> {
	static _FORCE_INLINE_ void init(Variant *r_variant) { VariantInternal::init_vector4_array(r_variant); }
};

template <>
struct VariantInitializer<Object *> {
	static _FORCE_INLINE_ void init(Variant *r_variant) { VariantInternal::init_object(r_variant); }
};

/// Note: This struct assumes that the argument type is already of the correct type.
template <typename T, typename = void>
struct VariantDefaultInitializer;

template <typename T>
struct VariantDefaultInitializer<T, std::enable_if_t<IsVariantTypeT<T>>> {
	static _FORCE_INLINE_ void init(Variant *r_variant) {
		VariantInternalAccessor<T>::get(r_variant) = T();
	}
};

template <typename T>
struct VariantTypeChanger {
	static _FORCE_INLINE_ void change(Variant *r_variant) {
		if (r_variant->get_type() != GetTypeInfo<T>::VARIANT_TYPE || GetTypeInfo<T>::VARIANT_TYPE >= Variant::PACKED_BYTE_ARRAY) { //second condition removed by optimizer
			VariantInternal::clear(r_variant);
			VariantInitializer<T>::init(r_variant);
		}
	}
	static _FORCE_INLINE_ void change_and_reset(Variant *r_variant) {
		if (r_variant->get_type() != GetTypeInfo<T>::VARIANT_TYPE || GetTypeInfo<T>::VARIANT_TYPE >= Variant::PACKED_BYTE_ARRAY) { //second condition removed by optimizer
			VariantInternal::clear(r_variant);
			VariantInitializer<T>::init(r_variant);
		} else {
			VariantDefaultInitializer<T>::init(r_variant);
		}
	}
};

template <typename T>
struct VariantTypeAdjust {
	_FORCE_INLINE_ static void adjust(Variant *r_ret) {
		VariantTypeChanger<GetSimpleTypeT<T>>::change(r_ret);
	}
};

template <>
struct VariantTypeAdjust<Variant> {
	_FORCE_INLINE_ static void adjust(Variant *r_ret) {
		// Do nothing for variant.
	}
};

template <>
struct VariantTypeAdjust<Object *> {
	_FORCE_INLINE_ static void adjust(Variant *r_ret) {
		VariantInternal::clear(r_ret);
		*r_ret = (Object *)nullptr;
	}
};

// GDExtension helpers.

template <typename T>
struct VariantTypeConstructor {
	_FORCE_INLINE_ static void variant_from_type(void *r_variant, void *p_value) {
		// r_variant is provided by caller as uninitialized memory
		memnew_placement(r_variant, Variant(*((T *)p_value)));
	}

	_FORCE_INLINE_ static void type_from_variant(void *r_value, void *p_variant) {
		// r_value is provided by caller as uninitialized memory
		memnew_placement(r_value, T(*reinterpret_cast<Variant *>(p_variant)));
	}
};
