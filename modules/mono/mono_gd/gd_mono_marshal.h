/*************************************************************************/
/*  gd_mono_marshal.h                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "gd_mono.h"
#include "gd_mono_utils.h"
#include "variant.h"

namespace GDMonoMarshal {

template <typename T>
T unbox(MonoObject *p_obj) {
	return *(T *)mono_object_unbox(p_obj);
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

Variant::Type managed_to_variant_type(const ManagedType &p_type);

// String

String mono_to_utf8_string(MonoString *p_mono_string);
String mono_to_utf16_string(MonoString *p_mono_string);

_FORCE_INLINE_ String mono_string_to_godot(MonoString *p_mono_string) {
	if (sizeof(CharType) == 2)
		return mono_to_utf16_string(p_mono_string);

	return mono_to_utf8_string(p_mono_string);
}

_FORCE_INLINE_ MonoString *mono_from_utf8_string(const String &p_string) {
	return mono_string_new(mono_domain_get(), p_string.utf8().get_data());
}

_FORCE_INLINE_ MonoString *mono_from_utf16_string(const String &p_string) {
	return mono_string_from_utf16((mono_unichar2 *)p_string.c_str());
}

_FORCE_INLINE_ MonoString *mono_string_from_godot(const String &p_string) {
	if (sizeof(CharType) == 2)
		return mono_from_utf16_string(p_string);

	return mono_from_utf8_string(p_string);
}

// Variant

MonoObject *variant_to_mono_object(const Variant *p_var, const ManagedType &p_type);
MonoObject *variant_to_mono_object(const Variant *p_var);

_FORCE_INLINE_ MonoObject *variant_to_mono_object(Variant p_var) {
	return variant_to_mono_object(&p_var);
}

Variant mono_object_to_variant(MonoObject *p_obj);
Variant mono_object_to_variant(MonoObject *p_obj, const ManagedType &p_type);

// Array

MonoArray *Array_to_mono_array(const Array &p_array);
Array mono_array_to_Array(MonoArray *p_array);

// PoolIntArray

MonoArray *PoolIntArray_to_mono_array(const PoolIntArray &p_array);
PoolIntArray mono_array_to_PoolIntArray(MonoArray *p_array);

// PoolByteArray

MonoArray *PoolByteArray_to_mono_array(const PoolByteArray &p_array);
PoolByteArray mono_array_to_PoolByteArray(MonoArray *p_array);

// PoolRealArray

MonoArray *PoolRealArray_to_mono_array(const PoolRealArray &p_array);
PoolRealArray mono_array_to_PoolRealArray(MonoArray *p_array);

// PoolStringArray

MonoArray *PoolStringArray_to_mono_array(const PoolStringArray &p_array);
PoolStringArray mono_array_to_PoolStringArray(MonoArray *p_array);

// PoolColorArray

MonoArray *PoolColorArray_to_mono_array(const PoolColorArray &p_array);
PoolColorArray mono_array_to_PoolColorArray(MonoArray *p_array);

// PoolVector2Array

MonoArray *PoolVector2Array_to_mono_array(const PoolVector2Array &p_array);
PoolVector2Array mono_array_to_PoolVector2Array(MonoArray *p_array);

// PoolVector3Array

MonoArray *PoolVector3Array_to_mono_array(const PoolVector3Array &p_array);
PoolVector3Array mono_array_to_PoolVector3Array(MonoArray *p_array);

// Dictionary

MonoObject *Dictionary_to_mono_object(const Dictionary &p_dict);
Dictionary mono_object_to_Dictionary(MonoObject *p_dict);

#ifdef YOLO_COPY
#define MARSHALLED_OUT(m_t, m_in, m_out) m_t *m_out = (m_t *)&m_in;
#define MARSHALLED_IN(m_t, m_in, m_out) m_t m_out = *reinterpret_cast<m_t *>(m_in);
#else

// Expects m_in to be of type float*

#define MARSHALLED_OUT(m_t, m_in, m_out) MARSHALLED_OUT_##m_t(m_in, m_out)
#define MARSHALLED_IN(m_t, m_in, m_out) MARSHALLED_IN_##m_t(m_in, m_out)

// Vector2

#define MARSHALLED_OUT_Vector2(m_in, m_out) real_t m_out[2] = { m_in.x, m_in.y };
#define MARSHALLED_IN_Vector2(m_in, m_out) Vector2 m_out(m_in[0], m_in[1]);

// Rect2

#define MARSHALLED_OUT_Rect2(m_in, m_out) real_t m_out[4] = { m_in.position.x, m_in.position.y, m_in.size.width, m_in.size.height };
#define MARSHALLED_IN_Rect2(m_in, m_out) Rect2 m_out(m_in[0], m_in[1], m_in[2], m_in[3]);

// Transform2D

#define MARSHALLED_OUT_Transform2D(m_in, m_out) real_t m_out[6] = { m_in[0].x, m_in[0].y, m_in[1].x, m_in[1].y, m_in[2].x, m_in[2].y };
#define MARSHALLED_IN_Transform2D(m_in, m_out) Transform2D m_out(m_in[0], m_in[1], m_in[2], m_in[3], m_in[4], m_in[5]);

// Vector3

#define MARSHALLED_OUT_Vector3(m_in, m_out) real_t m_out[3] = { m_in.x, m_in.y, m_in.z };
#define MARSHALLED_IN_Vector3(m_in, m_out) Vector3 m_out(m_in[0], m_in[1], m_in[2]);

// Basis

#define MARSHALLED_OUT_Basis(m_in, m_out) real_t m_out[9] = { \
	m_in[0].x, m_in[0].y, m_in[0].z,                          \
	m_in[1].x, m_in[1].y, m_in[1].z,                          \
	m_in[2].x, m_in[2].y, m_in[2].z                           \
};
#define MARSHALLED_IN_Basis(m_in, m_out) Basis m_out(m_in[0], m_in[1], m_in[2], m_in[3], m_in[4], m_in[5], m_in[6], m_in[7], m_in[8]);

// Quat

#define MARSHALLED_OUT_Quat(m_in, m_out) real_t m_out[4] = { m_in.x, m_in.y, m_in.z, m_in.w };
#define MARSHALLED_IN_Quat(m_in, m_out) Quat m_out(m_in[0], m_in[1], m_in[2], m_in[3]);

// Transform

#define MARSHALLED_OUT_Transform(m_in, m_out) real_t m_out[12] = { \
	m_in.basis[0].x, m_in.basis[0].y, m_in.basis[0].z,             \
	m_in.basis[1].x, m_in.basis[1].y, m_in.basis[1].z,             \
	m_in.basis[2].x, m_in.basis[2].y, m_in.basis[2].z,             \
	m_in.origin.x, m_in.origin.y, m_in.origin.z                    \
};
#define MARSHALLED_IN_Transform(m_in, m_out) Transform m_out(                                   \
		Basis(m_in[0], m_in[1], m_in[2], m_in[3], m_in[4], m_in[5], m_in[6], m_in[7], m_in[8]), \
		Vector3(m_in[9], m_in[10], m_in[11]));

// AABB

#define MARSHALLED_OUT_AABB(m_in, m_out) real_t m_out[6] = { m_in.position.x, m_in.position.y, m_in.position.z, m_in.size.x, m_in.size.y, m_in.size.z };
#define MARSHALLED_IN_AABB(m_in, m_out) AABB m_out(Vector3(m_in[0], m_in[1], m_in[2]), Vector3(m_in[3], m_in[4], m_in[5]));

// Color

#define MARSHALLED_OUT_Color(m_in, m_out) real_t m_out[4] = { m_in.r, m_in.g, m_in.b, m_in.a };
#define MARSHALLED_IN_Color(m_in, m_out) Color m_out(m_in[0], m_in[1], m_in[2], m_in[3]);

// Plane

#define MARSHALLED_OUT_Plane(m_in, m_out) real_t m_out[4] = { m_in.normal.x, m_in.normal.y, m_in.normal.z, m_in.d };
#define MARSHALLED_IN_Plane(m_in, m_out) Plane m_out(m_in[0], m_in[1], m_in[2], m_in[3]);

#endif

} // namespace GDMonoMarshal

#endif // GDMONOMARSHAL_H
