/*************************************************************************/
/*  variant_internal.h                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef VARIANT_INTERNAL_H
#define VARIANT_INTERNAL_H

#include "variant.h"

// For use when you want to access the internal pointer of a Variant directly.
// Use with caution. You need to be sure that the type is correct.
class VariantInternal {
public:
	// Set type.
	_FORCE_INLINE_ static void initialize(Variant *v, Variant::Type p_type) { v->type = p_type; }

	// Atomic types.
	_FORCE_INLINE_ static bool *get_bool(Variant *v) { return &v->_data._bool; }
	_FORCE_INLINE_ static const bool *get_bool(const Variant *v) { return &v->_data._bool; }
	_FORCE_INLINE_ static int64_t *get_int(Variant *v) { return &v->_data._int; }
	_FORCE_INLINE_ static const int64_t *get_int(const Variant *v) { return &v->_data._int; }
	_FORCE_INLINE_ static double *get_float(Variant *v) { return &v->_data._float; }
	_FORCE_INLINE_ static const double *get_float(const Variant *v) { return &v->_data._float; }
	_FORCE_INLINE_ static String *get_string(Variant *v) { return reinterpret_cast<String *>(v->_data._mem); }
	_FORCE_INLINE_ static const String *get_string(const Variant *v) { return reinterpret_cast<const String *>(v->_data._mem); }

	// Math types.
	_FORCE_INLINE_ static Vector2 *get_vector2(Variant *v) { return reinterpret_cast<Vector2 *>(v->_data._mem); }
	_FORCE_INLINE_ static const Vector2 *get_vector2(const Variant *v) { return reinterpret_cast<const Vector2 *>(v->_data._mem); }
	_FORCE_INLINE_ static Vector2i *get_vector2i(Variant *v) { return reinterpret_cast<Vector2i *>(v->_data._mem); }
	_FORCE_INLINE_ static const Vector2i *get_vector2i(const Variant *v) { return reinterpret_cast<const Vector2i *>(v->_data._mem); }
	_FORCE_INLINE_ static Rect2 *get_rect2(Variant *v) { return reinterpret_cast<Rect2 *>(v->_data._mem); }
	_FORCE_INLINE_ static const Rect2 *get_rect2(const Variant *v) { return reinterpret_cast<const Rect2 *>(v->_data._mem); }
	_FORCE_INLINE_ static Rect2i *get_rect2i(Variant *v) { return reinterpret_cast<Rect2i *>(v->_data._mem); }
	_FORCE_INLINE_ static const Rect2i *get_rect2i(const Variant *v) { return reinterpret_cast<const Rect2i *>(v->_data._mem); }
	_FORCE_INLINE_ static Vector3 *get_vector3(Variant *v) { return reinterpret_cast<Vector3 *>(v->_data._mem); }
	_FORCE_INLINE_ static const Vector3 *get_vector3(const Variant *v) { return reinterpret_cast<const Vector3 *>(v->_data._mem); }
	_FORCE_INLINE_ static Vector3i *get_vector3i(Variant *v) { return reinterpret_cast<Vector3i *>(v->_data._mem); }
	_FORCE_INLINE_ static const Vector3i *get_vector3i(const Variant *v) { return reinterpret_cast<const Vector3i *>(v->_data._mem); }
	_FORCE_INLINE_ static Transform2D *get_transform2d(Variant *v) { return v->_data._transform2d; }
	_FORCE_INLINE_ static const Transform2D *get_transform2d(const Variant *v) { return v->_data._transform2d; }
	_FORCE_INLINE_ static Plane *get_plane(Variant *v) { return reinterpret_cast<Plane *>(v->_data._mem); }
	_FORCE_INLINE_ static const Plane *get_plane(const Variant *v) { return reinterpret_cast<const Plane *>(v->_data._mem); }
	_FORCE_INLINE_ static Quat *get_quat(Variant *v) { return reinterpret_cast<Quat *>(v->_data._mem); }
	_FORCE_INLINE_ static const Quat *get_quat(const Variant *v) { return reinterpret_cast<const Quat *>(v->_data._mem); }
	_FORCE_INLINE_ static ::AABB *get_aabb(Variant *v) { return v->_data._aabb; }
	_FORCE_INLINE_ static const ::AABB *get_aabb(const Variant *v) { return v->_data._aabb; }
	_FORCE_INLINE_ static Basis *get_basis(Variant *v) { return v->_data._basis; }
	_FORCE_INLINE_ static const Basis *get_basis(const Variant *v) { return v->_data._basis; }
	_FORCE_INLINE_ static Transform *get_transform(Variant *v) { return v->_data._transform; }
	_FORCE_INLINE_ static const Transform *get_transform(const Variant *v) { return v->_data._transform; }

	// Misc types.
	_FORCE_INLINE_ static Color *get_color(Variant *v) { return reinterpret_cast<Color *>(v->_data._mem); }
	_FORCE_INLINE_ static const Color *get_color(const Variant *v) { return reinterpret_cast<const Color *>(v->_data._mem); }
	_FORCE_INLINE_ static StringName *get_string_name(Variant *v) { return reinterpret_cast<StringName *>(v->_data._mem); }
	_FORCE_INLINE_ static const StringName *get_string_name(const Variant *v) { return reinterpret_cast<const StringName *>(v->_data._mem); }
	_FORCE_INLINE_ static NodePath *get_node_path(Variant *v) { return reinterpret_cast<NodePath *>(v->_data._mem); }
	_FORCE_INLINE_ static const NodePath *get_node_path(const Variant *v) { return reinterpret_cast<const NodePath *>(v->_data._mem); }
	_FORCE_INLINE_ static RID *get_rid(Variant *v) { return reinterpret_cast<RID *>(v->_data._mem); }
	_FORCE_INLINE_ static const RID *get_rid(const Variant *v) { return reinterpret_cast<const RID *>(v->_data._mem); }
	_FORCE_INLINE_ static Callable *get_callable(Variant *v) { return reinterpret_cast<Callable *>(v->_data._mem); }
	_FORCE_INLINE_ static const Callable *get_callable(const Variant *v) { return reinterpret_cast<const Callable *>(v->_data._mem); }
	_FORCE_INLINE_ static Signal *get_signal(Variant *v) { return reinterpret_cast<Signal *>(v->_data._mem); }
	_FORCE_INLINE_ static const Signal *get_signal(const Variant *v) { return reinterpret_cast<const Signal *>(v->_data._mem); }
	_FORCE_INLINE_ static Dictionary *get_dictionary(Variant *v) { return reinterpret_cast<Dictionary *>(v->_data._mem); }
	_FORCE_INLINE_ static const Dictionary *get_dictionary(const Variant *v) { return reinterpret_cast<const Dictionary *>(v->_data._mem); }
	_FORCE_INLINE_ static Array *get_array(Variant *v) { return reinterpret_cast<Array *>(v->_data._mem); }
	_FORCE_INLINE_ static const Array *get_array(const Variant *v) { return reinterpret_cast<const Array *>(v->_data._mem); }

	// Typed arrays.
	_FORCE_INLINE_ static PackedByteArray *get_byte_array(Variant *v) { return &static_cast<Variant::PackedArrayRef<uint8_t> *>(v->_data.packed_array)->array; }
	_FORCE_INLINE_ static const PackedByteArray *get_byte_array(const Variant *v) { return &static_cast<const Variant::PackedArrayRef<uint8_t> *>(v->_data.packed_array)->array; }
	_FORCE_INLINE_ static PackedInt32Array *get_int32_array(Variant *v) { return &static_cast<Variant::PackedArrayRef<int32_t> *>(v->_data.packed_array)->array; }
	_FORCE_INLINE_ static const PackedInt32Array *get_int32_array(const Variant *v) { return &static_cast<const Variant::PackedArrayRef<int32_t> *>(v->_data.packed_array)->array; }
	_FORCE_INLINE_ static PackedInt64Array *get_int64_array(Variant *v) { return &static_cast<Variant::PackedArrayRef<int64_t> *>(v->_data.packed_array)->array; }
	_FORCE_INLINE_ static const PackedInt64Array *get_int64_array(const Variant *v) { return &static_cast<const Variant::PackedArrayRef<int64_t> *>(v->_data.packed_array)->array; }
	_FORCE_INLINE_ static PackedFloat32Array *get_float32_array(Variant *v) { return &static_cast<Variant::PackedArrayRef<float> *>(v->_data.packed_array)->array; }
	_FORCE_INLINE_ static const PackedFloat32Array *get_float32_array(const Variant *v) { return &static_cast<const Variant::PackedArrayRef<float> *>(v->_data.packed_array)->array; }
	_FORCE_INLINE_ static PackedFloat64Array *get_float64_array(Variant *v) { return &static_cast<Variant::PackedArrayRef<double> *>(v->_data.packed_array)->array; }
	_FORCE_INLINE_ static const PackedFloat64Array *get_float64_array(const Variant *v) { return &static_cast<const Variant::PackedArrayRef<double> *>(v->_data.packed_array)->array; }
	_FORCE_INLINE_ static PackedStringArray *get_string_array(Variant *v) { return &static_cast<Variant::PackedArrayRef<String> *>(v->_data.packed_array)->array; }
	_FORCE_INLINE_ static const PackedStringArray *get_string_array(const Variant *v) { return &static_cast<const Variant::PackedArrayRef<String> *>(v->_data.packed_array)->array; }
	_FORCE_INLINE_ static PackedVector2Array *get_vector2_array(Variant *v) { return &static_cast<Variant::PackedArrayRef<Vector2> *>(v->_data.packed_array)->array; }
	_FORCE_INLINE_ static const PackedVector2Array *get_vector2_array(const Variant *v) { return &static_cast<const Variant::PackedArrayRef<Vector2> *>(v->_data.packed_array)->array; }
	_FORCE_INLINE_ static PackedVector3Array *get_vector3_array(Variant *v) { return &static_cast<Variant::PackedArrayRef<Vector3> *>(v->_data.packed_array)->array; }
	_FORCE_INLINE_ static const PackedVector3Array *get_vector3_array(const Variant *v) { return &static_cast<const Variant::PackedArrayRef<Vector3> *>(v->_data.packed_array)->array; }
	_FORCE_INLINE_ static PackedColorArray *get_color_array(Variant *v) { return &static_cast<Variant::PackedArrayRef<Color> *>(v->_data.packed_array)->array; }
	_FORCE_INLINE_ static const PackedColorArray *get_color_array(const Variant *v) { return &static_cast<const Variant::PackedArrayRef<Color> *>(v->_data.packed_array)->array; }
};

#endif // VARIANT_INTERNAL_H
