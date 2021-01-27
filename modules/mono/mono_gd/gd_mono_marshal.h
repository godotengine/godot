/*************************************************************************/
/*  gd_mono_marshal.h                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef GDMONOMARSHAL_H
#define GDMONOMARSHAL_H

#include "core/variant/variant.h"

#include "../managed_callable.h"
#include "gd_mono.h"
#include "gd_mono_utils.h"

namespace GDMonoMarshal {

template <typename T>
T unbox(MonoObject *p_obj) {
	return *(T *)mono_object_unbox(p_obj);
}

template <typename T>
T *unbox_addr(MonoObject *p_obj) {
	return (T *)mono_object_unbox(p_obj);
}

#define BOX_DOUBLE(x) mono_value_box(mono_domain_get(), CACHED_CLASS_RAW(double), &x)
#define BOX_FLOAT(x) mono_value_box(mono_domain_get(), CACHED_CLASS_RAW(float), &x)
#define BOX_INT64(x) mono_value_box(mono_domain_get(), CACHED_CLASS_RAW(int64_t), &x)
#define BOX_INT32(x) mono_value_box(mono_domain_get(), CACHED_CLASS_RAW(int32_t), &x)
#define BOX_INT16(x) mono_value_box(mono_domain_get(), CACHED_CLASS_RAW(int16_t), &x)
#define BOX_INT8(x) mono_value_box(mono_domain_get(), CACHED_CLASS_RAW(int8_t), &x)
#define BOX_UINT64(x) mono_value_box(mono_domain_get(), CACHED_CLASS_RAW(uint64_t), &x)
#define BOX_UINT32(x) mono_value_box(mono_domain_get(), CACHED_CLASS_RAW(uint32_t), &x)
#define BOX_UINT16(x) mono_value_box(mono_domain_get(), CACHED_CLASS_RAW(uint16_t), &x)
#define BOX_UINT8(x) mono_value_box(mono_domain_get(), CACHED_CLASS_RAW(uint8_t), &x)
#define BOX_BOOLEAN(x) mono_value_box(mono_domain_get(), CACHED_CLASS_RAW(bool), &x)
#define BOX_PTR(x) mono_value_box(mono_domain_get(), CACHED_CLASS_RAW(IntPtr), x)
#define BOX_ENUM(m_enum_class, x) mono_value_box(mono_domain_get(), m_enum_class, &x)

Variant::Type managed_to_variant_type(const ManagedType &p_type, bool *r_nil_is_variant = nullptr);

bool try_get_array_element_type(const ManagedType &p_array_type, ManagedType &r_elem_type);

// String

_FORCE_INLINE_ String mono_string_to_godot_not_null(MonoString *p_mono_string) {
	char32_t *utf32 = (char32_t *)mono_string_to_utf32(p_mono_string);
	String ret = String(utf32);
	mono_free(utf32);
	return ret;
}

_FORCE_INLINE_ String mono_string_to_godot(MonoString *p_mono_string) {
	if (p_mono_string == nullptr) {
		return String();
	}

	return mono_string_to_godot_not_null(p_mono_string);
}

_FORCE_INLINE_ MonoString *mono_string_from_godot(const String &p_string) {
	return mono_string_from_utf32((mono_unichar4 *)(p_string.get_data()));
}

// Variant

size_t variant_get_managed_unboxed_size(const ManagedType &p_type);
void *variant_to_managed_unboxed(const Variant &p_var, const ManagedType &p_type, void *r_buffer, unsigned int &r_offset);
MonoObject *variant_to_mono_object(const Variant &p_var, const ManagedType &p_type);

MonoObject *variant_to_mono_object(const Variant &p_var);
MonoArray *variant_to_mono_array(const Variant &p_var, GDMonoClass *p_type_class);
MonoObject *variant_to_mono_object_of_class(const Variant &p_var, GDMonoClass *p_type_class);
MonoObject *variant_to_mono_object_of_genericinst(const Variant &p_var, GDMonoClass *p_type_class);
MonoString *variant_to_mono_string(const Variant &p_var);

// These overloads were added to avoid passing a `const Variant *` to the `const Variant &`
// parameter. That would result in the `Variant(bool)` copy constructor being called as
// pointers are implicitly converted to bool. Implicit conversions are f-ing evil.

_FORCE_INLINE_ void *variant_to_managed_unboxed(const Variant *p_var, const ManagedType &p_type, void *r_buffer, unsigned int &r_offset) {
	return variant_to_managed_unboxed(*p_var, p_type, r_buffer, r_offset);
}
_FORCE_INLINE_ MonoObject *variant_to_mono_object(const Variant *p_var, const ManagedType &p_type) {
	return variant_to_mono_object(*p_var, p_type);
}
_FORCE_INLINE_ MonoObject *variant_to_mono_object(const Variant *p_var) {
	return variant_to_mono_object(*p_var);
}
_FORCE_INLINE_ MonoArray *variant_to_mono_array(const Variant *p_var, GDMonoClass *p_type_class) {
	return variant_to_mono_array(*p_var, p_type_class);
}
_FORCE_INLINE_ MonoObject *variant_to_mono_object_of_class(const Variant *p_var, GDMonoClass *p_type_class) {
	return variant_to_mono_object_of_class(*p_var, p_type_class);
}
_FORCE_INLINE_ MonoObject *variant_to_mono_object_of_genericinst(const Variant *p_var, GDMonoClass *p_type_class) {
	return variant_to_mono_object_of_genericinst(*p_var, p_type_class);
}
_FORCE_INLINE_ MonoString *variant_to_mono_string(const Variant *p_var) {
	return variant_to_mono_string(*p_var);
}

Variant mono_object_to_variant(MonoObject *p_obj);
Variant mono_object_to_variant(MonoObject *p_obj, const ManagedType &p_type);
Variant mono_object_to_variant_no_err(MonoObject *p_obj, const ManagedType &p_type);

/// Tries to convert the MonoObject* to Variant and then convert the Variant to String.
/// If the MonoObject* cannot be converted to Variant, then 'ToString()' is called instead.
String mono_object_to_variant_string(MonoObject *p_obj, MonoException **r_exc);

// System.Collections.Generic

MonoObject *Dictionary_to_system_generic_dict(const Dictionary &p_dict, GDMonoClass *p_class, MonoReflectionType *p_key_reftype, MonoReflectionType *p_value_reftype);
Dictionary system_generic_dict_to_Dictionary(MonoObject *p_obj, GDMonoClass *p_class, MonoReflectionType *p_key_reftype, MonoReflectionType *p_value_reftype);

MonoObject *Array_to_system_generic_list(const Array &p_array, GDMonoClass *p_class, MonoReflectionType *p_elem_reftype);
Variant system_generic_list_to_Array_variant(MonoObject *p_obj, GDMonoClass *p_class, MonoReflectionType *p_elem_reftype);

// Array

MonoArray *Array_to_mono_array(const Array &p_array);
MonoArray *Array_to_mono_array(const Array &p_array, MonoClass *p_array_type_class);
Array mono_array_to_Array(MonoArray *p_array);

// PackedInt32Array

MonoArray *PackedInt32Array_to_mono_array(const PackedInt32Array &p_array);
PackedInt32Array mono_array_to_PackedInt32Array(MonoArray *p_array);

// PackedInt64Array

MonoArray *PackedInt64Array_to_mono_array(const PackedInt64Array &p_array);
PackedInt64Array mono_array_to_PackedInt64Array(MonoArray *p_array);

// PackedByteArray

MonoArray *PackedByteArray_to_mono_array(const PackedByteArray &p_array);
PackedByteArray mono_array_to_PackedByteArray(MonoArray *p_array);

// PackedFloat32Array

MonoArray *PackedFloat32Array_to_mono_array(const PackedFloat32Array &p_array);
PackedFloat32Array mono_array_to_PackedFloat32Array(MonoArray *p_array);

// PackedFloat64Array

MonoArray *PackedFloat64Array_to_mono_array(const PackedFloat64Array &p_array);
PackedFloat64Array mono_array_to_PackedFloat64Array(MonoArray *p_array);

// PackedStringArray

MonoArray *PackedStringArray_to_mono_array(const PackedStringArray &p_array);
PackedStringArray mono_array_to_PackedStringArray(MonoArray *p_array);

// PackedColorArray

MonoArray *PackedColorArray_to_mono_array(const PackedColorArray &p_array);
PackedColorArray mono_array_to_PackedColorArray(MonoArray *p_array);

// PackedVector2Array

MonoArray *PackedVector2Array_to_mono_array(const PackedVector2Array &p_array);
PackedVector2Array mono_array_to_PackedVector2Array(MonoArray *p_array);

// PackedVector3Array

MonoArray *PackedVector3Array_to_mono_array(const PackedVector3Array &p_array);
PackedVector3Array mono_array_to_PackedVector3Array(MonoArray *p_array);

#pragma pack(push, 1)

struct M_Callable {
	MonoObject *target;
	MonoObject *method_string_name;
	MonoDelegate *delegate;
};

struct M_SignalInfo {
	MonoObject *owner;
	MonoObject *name_string_name;
};

#pragma pack(pop)

// Callable
Callable managed_to_callable(const M_Callable &p_managed_callable);
M_Callable callable_to_managed(const Callable &p_callable);

// SignalInfo
Signal managed_to_signal_info(const M_SignalInfo &p_managed_signal);
M_SignalInfo signal_info_to_managed(const Signal &p_signal);

// Structures

namespace InteropLayout {

enum {
	MATCHES_int = (sizeof(int32_t) == sizeof(uint32_t)),

	MATCHES_float = (sizeof(float) == sizeof(uint32_t)),

	MATCHES_double = (sizeof(double) == sizeof(uint64_t)),

#ifdef REAL_T_IS_DOUBLE
	MATCHES_real_t = (sizeof(real_t) == sizeof(uint64_t)),
#else
	MATCHES_real_t = (sizeof(real_t) == sizeof(uint32_t)),
#endif

	MATCHES_Vector2 = (MATCHES_real_t && (sizeof(Vector2) == (sizeof(real_t) * 2)) &&
					   offsetof(Vector2, x) == (sizeof(real_t) * 0) &&
					   offsetof(Vector2, y) == (sizeof(real_t) * 1)),

	MATCHES_Vector2i = (MATCHES_int && (sizeof(Vector2i) == (sizeof(int32_t) * 2)) &&
						offsetof(Vector2i, x) == (sizeof(int32_t) * 0) &&
						offsetof(Vector2i, y) == (sizeof(int32_t) * 1)),

	MATCHES_Rect2 = (MATCHES_Vector2 && (sizeof(Rect2) == (sizeof(Vector2) * 2)) &&
					 offsetof(Rect2, position) == (sizeof(Vector2) * 0) &&
					 offsetof(Rect2, size) == (sizeof(Vector2) * 1)),

	MATCHES_Rect2i = (MATCHES_Vector2i && (sizeof(Rect2i) == (sizeof(Vector2i) * 2)) &&
					  offsetof(Rect2i, position) == (sizeof(Vector2i) * 0) &&
					  offsetof(Rect2i, size) == (sizeof(Vector2i) * 1)),

	MATCHES_Transform2D = (MATCHES_Vector2 && (sizeof(Transform2D) == (sizeof(Vector2) * 3))), // No field offset required, it stores an array

	MATCHES_Vector3 = (MATCHES_real_t && (sizeof(Vector3) == (sizeof(real_t) * 3)) &&
					   offsetof(Vector3, x) == (sizeof(real_t) * 0) &&
					   offsetof(Vector3, y) == (sizeof(real_t) * 1) &&
					   offsetof(Vector3, z) == (sizeof(real_t) * 2)),

	MATCHES_Vector3i = (MATCHES_int && (sizeof(Vector3i) == (sizeof(int32_t) * 3)) &&
						offsetof(Vector3i, x) == (sizeof(int32_t) * 0) &&
						offsetof(Vector3i, y) == (sizeof(int32_t) * 1) &&
						offsetof(Vector3i, z) == (sizeof(int32_t) * 2)),

	MATCHES_Basis = (MATCHES_Vector3 && (sizeof(Basis) == (sizeof(Vector3) * 3))), // No field offset required, it stores an array

	MATCHES_Quat = (MATCHES_real_t && (sizeof(Quat) == (sizeof(real_t) * 4)) &&
					offsetof(Quat, x) == (sizeof(real_t) * 0) &&
					offsetof(Quat, y) == (sizeof(real_t) * 1) &&
					offsetof(Quat, z) == (sizeof(real_t) * 2) &&
					offsetof(Quat, w) == (sizeof(real_t) * 3)),

	MATCHES_Transform = (MATCHES_Basis && MATCHES_Vector3 && (sizeof(Transform) == (sizeof(Basis) + sizeof(Vector3))) &&
						 offsetof(Transform, basis) == 0 &&
						 offsetof(Transform, origin) == sizeof(Basis)),

	MATCHES_AABB = (MATCHES_Vector3 && (sizeof(AABB) == (sizeof(Vector3) * 2)) &&
					offsetof(AABB, position) == (sizeof(Vector3) * 0) &&
					offsetof(AABB, size) == (sizeof(Vector3) * 1)),

	MATCHES_Color = (MATCHES_float && (sizeof(Color) == (sizeof(float) * 4)) &&
					 offsetof(Color, r) == (sizeof(float) * 0) &&
					 offsetof(Color, g) == (sizeof(float) * 1) &&
					 offsetof(Color, b) == (sizeof(float) * 2) &&
					 offsetof(Color, a) == (sizeof(float) * 3)),

	MATCHES_Plane = (MATCHES_Vector3 && MATCHES_real_t && (sizeof(Plane) == (sizeof(Vector3) + sizeof(real_t))) &&
					 offsetof(Plane, normal) == 0 &&
					 offsetof(Plane, d) == sizeof(Vector3))
};

// In the future we may force this if we want to ref return these structs
#ifdef GD_MONO_FORCE_INTEROP_STRUCT_COPY
/* clang-format off */
static_assert(MATCHES_Vector2 && MATCHES_Rect2 && MATCHES_Transform2D && MATCHES_Vector3 &&
				MATCHES_Basis && MATCHES_Quat && MATCHES_Transform && MATCHES_AABB && MATCHES_Color &&
				MATCHES_Plane && MATCHES_Vector2i && MATCHES_Rect2i && MATCHES_Vector3i);
/* clang-format on */
#endif
} // namespace InteropLayout

#pragma pack(push, 1)

struct M_Vector2 {
	real_t x, y;

	static _FORCE_INLINE_ Vector2 convert_to(const M_Vector2 &p_from) {
		return Vector2(p_from.x, p_from.y);
	}

	static _FORCE_INLINE_ M_Vector2 convert_from(const Vector2 &p_from) {
		M_Vector2 ret = { p_from.x, p_from.y };
		return ret;
	}
};

struct M_Vector2i {
	int32_t x, y;

	static _FORCE_INLINE_ Vector2i convert_to(const M_Vector2i &p_from) {
		return Vector2i(p_from.x, p_from.y);
	}

	static _FORCE_INLINE_ M_Vector2i convert_from(const Vector2i &p_from) {
		M_Vector2i ret = { p_from.x, p_from.y };
		return ret;
	}
};

struct M_Rect2 {
	M_Vector2 position;
	M_Vector2 size;

	static _FORCE_INLINE_ Rect2 convert_to(const M_Rect2 &p_from) {
		return Rect2(M_Vector2::convert_to(p_from.position),
				M_Vector2::convert_to(p_from.size));
	}

	static _FORCE_INLINE_ M_Rect2 convert_from(const Rect2 &p_from) {
		M_Rect2 ret = { M_Vector2::convert_from(p_from.position), M_Vector2::convert_from(p_from.size) };
		return ret;
	}
};

struct M_Rect2i {
	M_Vector2i position;
	M_Vector2i size;

	static _FORCE_INLINE_ Rect2i convert_to(const M_Rect2i &p_from) {
		return Rect2i(M_Vector2i::convert_to(p_from.position),
				M_Vector2i::convert_to(p_from.size));
	}

	static _FORCE_INLINE_ M_Rect2i convert_from(const Rect2i &p_from) {
		M_Rect2i ret = { M_Vector2i::convert_from(p_from.position), M_Vector2i::convert_from(p_from.size) };
		return ret;
	}
};

struct M_Transform2D {
	M_Vector2 elements[3];

	static _FORCE_INLINE_ Transform2D convert_to(const M_Transform2D &p_from) {
		return Transform2D(p_from.elements[0].x, p_from.elements[0].y,
				p_from.elements[1].x, p_from.elements[1].y,
				p_from.elements[2].x, p_from.elements[2].y);
	}

	static _FORCE_INLINE_ M_Transform2D convert_from(const Transform2D &p_from) {
		M_Transform2D ret = {
			M_Vector2::convert_from(p_from.elements[0]),
			M_Vector2::convert_from(p_from.elements[1]),
			M_Vector2::convert_from(p_from.elements[2])
		};
		return ret;
	}
};

struct M_Vector3 {
	real_t x, y, z;

	static _FORCE_INLINE_ Vector3 convert_to(const M_Vector3 &p_from) {
		return Vector3(p_from.x, p_from.y, p_from.z);
	}

	static _FORCE_INLINE_ M_Vector3 convert_from(const Vector3 &p_from) {
		M_Vector3 ret = { p_from.x, p_from.y, p_from.z };
		return ret;
	}
};

struct M_Vector3i {
	int32_t x, y, z;

	static _FORCE_INLINE_ Vector3i convert_to(const M_Vector3i &p_from) {
		return Vector3i(p_from.x, p_from.y, p_from.z);
	}

	static _FORCE_INLINE_ M_Vector3i convert_from(const Vector3i &p_from) {
		M_Vector3i ret = { p_from.x, p_from.y, p_from.z };
		return ret;
	}
};

struct M_Basis {
	M_Vector3 elements[3];

	static _FORCE_INLINE_ Basis convert_to(const M_Basis &p_from) {
		return Basis(M_Vector3::convert_to(p_from.elements[0]),
				M_Vector3::convert_to(p_from.elements[1]),
				M_Vector3::convert_to(p_from.elements[2]));
	}

	static _FORCE_INLINE_ M_Basis convert_from(const Basis &p_from) {
		M_Basis ret = {
			M_Vector3::convert_from(p_from.elements[0]),
			M_Vector3::convert_from(p_from.elements[1]),
			M_Vector3::convert_from(p_from.elements[2])
		};
		return ret;
	}
};

struct M_Quat {
	real_t x, y, z, w;

	static _FORCE_INLINE_ Quat convert_to(const M_Quat &p_from) {
		return Quat(p_from.x, p_from.y, p_from.z, p_from.w);
	}

	static _FORCE_INLINE_ M_Quat convert_from(const Quat &p_from) {
		M_Quat ret = { p_from.x, p_from.y, p_from.z, p_from.w };
		return ret;
	}
};

struct M_Transform {
	M_Basis basis;
	M_Vector3 origin;

	static _FORCE_INLINE_ Transform convert_to(const M_Transform &p_from) {
		return Transform(M_Basis::convert_to(p_from.basis), M_Vector3::convert_to(p_from.origin));
	}

	static _FORCE_INLINE_ M_Transform convert_from(const Transform &p_from) {
		M_Transform ret = { M_Basis::convert_from(p_from.basis), M_Vector3::convert_from(p_from.origin) };
		return ret;
	}
};

struct M_AABB {
	M_Vector3 position;
	M_Vector3 size;

	static _FORCE_INLINE_ AABB convert_to(const M_AABB &p_from) {
		return AABB(M_Vector3::convert_to(p_from.position), M_Vector3::convert_to(p_from.size));
	}

	static _FORCE_INLINE_ M_AABB convert_from(const AABB &p_from) {
		M_AABB ret = { M_Vector3::convert_from(p_from.position), M_Vector3::convert_from(p_from.size) };
		return ret;
	}
};

struct M_Color {
	float r, g, b, a;

	static _FORCE_INLINE_ Color convert_to(const M_Color &p_from) {
		return Color(p_from.r, p_from.g, p_from.b, p_from.a);
	}

	static _FORCE_INLINE_ M_Color convert_from(const Color &p_from) {
		M_Color ret = { p_from.r, p_from.g, p_from.b, p_from.a };
		return ret;
	}
};

struct M_Plane {
	M_Vector3 normal;
	real_t d;

	static _FORCE_INLINE_ Plane convert_to(const M_Plane &p_from) {
		return Plane(M_Vector3::convert_to(p_from.normal), p_from.d);
	}

	static _FORCE_INLINE_ M_Plane convert_from(const Plane &p_from) {
		M_Plane ret = { M_Vector3::convert_from(p_from.normal), p_from.d };
		return ret;
	}
};

#pragma pack(pop)

#define DECL_TYPE_MARSHAL_TEMPLATES(m_type)                                             \
	template <int>                                                                      \
	_FORCE_INLINE_ m_type marshalled_in_##m_type##_impl(const M_##m_type *p_from);      \
                                                                                        \
	template <>                                                                         \
	_FORCE_INLINE_ m_type marshalled_in_##m_type##_impl<0>(const M_##m_type *p_from) {  \
		return M_##m_type::convert_to(*p_from);                                         \
	}                                                                                   \
                                                                                        \
	template <>                                                                         \
	_FORCE_INLINE_ m_type marshalled_in_##m_type##_impl<1>(const M_##m_type *p_from) {  \
		return *reinterpret_cast<const m_type *>(p_from);                               \
	}                                                                                   \
                                                                                        \
	_FORCE_INLINE_ m_type marshalled_in_##m_type(const M_##m_type *p_from) {            \
		return marshalled_in_##m_type##_impl<InteropLayout::MATCHES_##m_type>(p_from);  \
	}                                                                                   \
                                                                                        \
	template <int>                                                                      \
	_FORCE_INLINE_ M_##m_type marshalled_out_##m_type##_impl(const m_type &p_from);     \
                                                                                        \
	template <>                                                                         \
	_FORCE_INLINE_ M_##m_type marshalled_out_##m_type##_impl<0>(const m_type &p_from) { \
		return M_##m_type::convert_from(p_from);                                        \
	}                                                                                   \
                                                                                        \
	template <>                                                                         \
	_FORCE_INLINE_ M_##m_type marshalled_out_##m_type##_impl<1>(const m_type &p_from) { \
		return *reinterpret_cast<const M_##m_type *>(&p_from);                          \
	}                                                                                   \
                                                                                        \
	_FORCE_INLINE_ M_##m_type marshalled_out_##m_type(const m_type &p_from) {           \
		return marshalled_out_##m_type##_impl<InteropLayout::MATCHES_##m_type>(p_from); \
	}

DECL_TYPE_MARSHAL_TEMPLATES(Vector2)
DECL_TYPE_MARSHAL_TEMPLATES(Vector2i)
DECL_TYPE_MARSHAL_TEMPLATES(Rect2)
DECL_TYPE_MARSHAL_TEMPLATES(Rect2i)
DECL_TYPE_MARSHAL_TEMPLATES(Transform2D)
DECL_TYPE_MARSHAL_TEMPLATES(Vector3)
DECL_TYPE_MARSHAL_TEMPLATES(Vector3i)
DECL_TYPE_MARSHAL_TEMPLATES(Basis)
DECL_TYPE_MARSHAL_TEMPLATES(Quat)
DECL_TYPE_MARSHAL_TEMPLATES(Transform)
DECL_TYPE_MARSHAL_TEMPLATES(AABB)
DECL_TYPE_MARSHAL_TEMPLATES(Color)
DECL_TYPE_MARSHAL_TEMPLATES(Plane)

#define MARSHALLED_IN(m_type, m_from_ptr) (GDMonoMarshal::marshalled_in_##m_type(m_from_ptr))
#define MARSHALLED_OUT(m_type, m_from) (GDMonoMarshal::marshalled_out_##m_type(m_from))
} // namespace GDMonoMarshal

#endif // GDMONOMARSHAL_H
