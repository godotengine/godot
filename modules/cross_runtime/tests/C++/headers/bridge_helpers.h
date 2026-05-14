/**************************************************************************/
/*  bridge_helpers.h                                                      */
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

#ifdef WEB_ENABLED

#include "bridge_api.h"

#include "core/config/engine.h"
#include "core/io/marshalls.h"
#include "core/math/aabb.h"
#include "core/math/basis.h"
#include "core/math/color.h"
#include "core/math/plane.h"
#include "core/math/projection.h"
#include "core/math/quaternion.h"
#include "core/math/rect2.h"
#include "core/math/rect2i.h"
#include "core/math/transform_2d.h"
#include "core/math/transform_3d.h"
#include "core/math/vector2.h"
#include "core/math/vector2i.h"
#include "core/math/vector3.h"
#include "core/math/vector3i.h"
#include "core/math/vector4.h"
#include "core/math/vector4i.h"
#include "core/object/object_id.h"
#include "core/string/node_path.h"
#include "core/string/string_name.h"
#include "core/string/ustring.h"
#include "core/templates/rid.h"
#include "core/variant/array.h"
#include "core/variant/callable.h"
#include "core/variant/dictionary.h"
#include "core/variant/variant.h"

#include <cstdint>
#include <cstdio>

// Updates the current status atomically.
static inline void update_status(volatile uint8_t *offset, int value) {
	__atomic_store_n(reinterpret_cast<volatile int *>(offset), value, __ATOMIC_RELEASE);
}

// Reads raw bytes from offset into a POD
template <typename T>
static inline T reader(const uint8_t *offset) {
	T value{};
	const uint8_t *src = offset;
	uint8_t *dst = reinterpret_cast<uint8_t *>(&value);
	for (size_t i = 0; i < sizeof(T); ++i) {
		dst[i] = src[i];
	}
	return value;
}

// Writes raw bytes from a POD
template <typename T>
static inline void writer(uint8_t *offset, const T &value) {
	uint8_t *dst = offset;
	const uint8_t *src = reinterpret_cast<const uint8_t *>(&value);
	for (size_t i = 0; i < sizeof(T); ++i) {
		dst[i] = src[i];
	}
}

static inline String read_string_from_data(const uint8_t *offset) {
	int len = reader<int>(offset);
	if (len < 0 || len > 4096) {
		return String();
	}

	const uint8_t *src = offset + 4;
	return String::utf8(reinterpret_cast<const char *>(src), len);
}

static inline void write_string_to_data(uint8_t *offset, const String &value) {
	CharString utf8 = value.utf8();
	int len = utf8.length();
	if (len < 0 || len > 4096) {
		writer<int>(offset, 0);
		return;
	}

	writer<int>(offset, len);
	for (int i = 0; i < len; ++i) {
		offset[4 + i] = static_cast<uint8_t>(utf8.ptr()[i]);
	}
}

static inline ObjectID read_object_id(const uint8_t *offset) {
	return ObjectID(reader<uint64_t>(offset));
}

static inline void write_object_id(uint8_t *offset, ObjectID id) {
	writer<uint64_t>(offset, static_cast<uint64_t>(id));
}

static inline int read_int32(const uint8_t *offset) {
	return reader<int>(offset);
}

static inline void write_int32(uint8_t *offset, int value) {
	writer<int>(offset, value);
}

static inline int64_t read_int64(const uint8_t *offset) {
	return reader<int64_t>(offset);
}

static inline void write_int64(uint8_t *offset, int64_t value) {
	writer<int64_t>(offset, value);
}

static inline uint64_t read_uint64(const uint8_t *offset) {
	return reader<uint64_t>(offset);
}

static inline void write_uint64(uint8_t *offset, uint64_t value) {
	writer<uint64_t>(offset, value);
}

static inline float read_float(const uint8_t *offset) {
	return reader<float>(offset);
}

static inline void write_float(uint8_t *offset, float value) {
	writer<float>(offset, value);
}

static inline double read_double(const uint8_t *offset) {
	return reader<double>(offset);
}

static inline void write_double(uint8_t *offset, double value) {
	writer<double>(offset, value);
}

static inline bool read_bool(const uint8_t *offset) {
	return read_int32(offset) != 0;
}

static inline void write_bool(uint8_t *offset, bool value) {
	write_int32(offset, value ? 1 : 0);
}

static inline Vector2 read_vector2(const uint8_t *offset) {
	return reader<Vector2>(offset);
}

static inline void write_vector2(uint8_t *offset, const Vector2 &value) {
	writer<Vector2>(offset, value);
}

static inline Vector2i read_vector2i(const uint8_t *offset) {
	return reader<Vector2i>(offset);
}

static inline void write_vector2i(uint8_t *offset, const Vector2i &value) {
	writer<Vector2i>(offset, value);
}

static inline Rect2 read_rect2(const uint8_t *offset) {
	return reader<Rect2>(offset);
}

static inline void write_rect2(uint8_t *offset, const Rect2 &value) {
	writer<Rect2>(offset, value);
}

static inline Rect2i read_rect2i(const uint8_t *offset) {
	return reader<Rect2i>(offset);
}

static inline void write_rect2i(uint8_t *offset, const Rect2i &value) {
	writer<Rect2i>(offset, value);
}

static inline Vector3 read_vector3(const uint8_t *offset) {
	return reader<Vector3>(offset);
}

static inline void write_vector3(uint8_t *offset, const Vector3 &value) {
	writer<Vector3>(offset, value);
}

static inline Vector3i read_vector3i(const uint8_t *offset) {
	return reader<Vector3i>(offset);
}

static inline void write_vector3i(uint8_t *offset, const Vector3i &value) {
	writer<Vector3i>(offset, value);
}

static inline Transform2D read_transform2d(const uint8_t *offset) {
	return reader<Transform2D>(offset);
}

static inline void write_transform2d(uint8_t *offset, const Transform2D &value) {
	writer<Transform2D>(offset, value);
}

static inline Vector4 read_vector4(const uint8_t *offset) {
	return reader<Vector4>(offset);
}

static inline void write_vector4(uint8_t *offset, const Vector4 &value) {
	writer<Vector4>(offset, value);
}

static inline Vector4i read_vector4i(const uint8_t *offset) {
	return reader<Vector4i>(offset);
}

static inline void write_vector4i(uint8_t *offset, const Vector4i &value) {
	writer<Vector4i>(offset, value);
}

static inline Plane read_plane(const uint8_t *offset) {
	return reader<Plane>(offset);
}

static inline void write_plane(uint8_t *offset, const Plane &value) {
	writer<Plane>(offset, value);
}

static inline Quaternion read_quaternion(const uint8_t *offset) {
	// Godot constructor expects (x, y, z, w).
	return Quaternion(
			read_float(offset + 0),
			read_float(offset + 4),
			read_float(offset + 8),
			read_float(offset + 12));
}

static inline void write_quaternion(uint8_t *offset, const Quaternion &value) {
	write_float(offset + 0, value.x);
	write_float(offset + 4, value.y);
	write_float(offset + 8, value.z);
	write_float(offset + 12, value.w);
}

static inline AABB read_aabb(const uint8_t *offset) {
	return reader<AABB>(offset);
}

static inline void write_aabb(uint8_t *offset, const AABB &value) {
	writer<AABB>(offset, value);
}

static inline Basis read_basis(const uint8_t *offset) {
	return reader<Basis>(offset);
}

static inline void write_basis(uint8_t *offset, const Basis &value) {
	writer<Basis>(offset, value);
}

static inline Transform3D read_transform3d(const uint8_t *offset) {
	return reader<Transform3D>(offset);
}

static inline void write_transform3d(uint8_t *offset, const Transform3D &value) {
	writer<Transform3D>(offset, value);
}

static inline Projection read_projection(const uint8_t *offset) {
	return reader<Projection>(offset);
}

static inline void write_projection(uint8_t *offset, const Projection &value) {
	writer<Projection>(offset, value);
}

static inline Color read_color(const uint8_t *offset) {
	return reader<Color>(offset);
}

static inline void write_color(uint8_t *offset, const Color &value) {
	writer<Color>(offset, value);
}

static inline StringName read_string_name(const uint8_t *offset) {
	return StringName(read_string_from_data(offset));
}

static inline void write_string_name(uint8_t *offset, const StringName &value) {
	write_string_to_data(offset, String(value));
}

static inline NodePath read_node_path(const uint8_t *offset) {
	return NodePath(read_string_from_data(offset));
}

static inline void write_node_path(uint8_t *offset, const NodePath &value) {
	write_string_to_data(offset, String(value));
}

static inline RID read_rid(const uint8_t *offset) {
	return RID::from_uint64(read_uint64(offset));
}

static inline void write_rid(uint8_t *offset, RID value) {
	write_uint64(offset, value.get_id());
}

static inline PackedByteArray read_packed_byte_array(const uint8_t *offset) {
	int len = read_int32(offset);
	if (len < 0) {
		return PackedByteArray();
	}

	PackedByteArray out;
	out.resize(len);
	for (int i = 0; i < len; ++i) {
		out.set(i, offset[4 + i]);
	}
	return out;
}

static inline void write_packed_byte_array(uint8_t *offset, const PackedByteArray &value) {
	const int len = value.size();
	write_int32(offset, len);
	for (int i = 0; i < len; ++i) {
		offset[4 + i] = value[i];
	}
}

static inline PackedInt32Array read_packed_int32_array(const uint8_t *offset) {
	int len = read_int32(offset);
	if (len < 0) {
		return PackedInt32Array();
	}

	PackedInt32Array out;
	out.resize(len);
	for (int i = 0; i < len; ++i) {
		out.set(i, read_int32(offset + 4 + (i * 4)));
	}
	return out;
}

static inline void write_packed_int32_array(uint8_t *offset, const PackedInt32Array &value) {
	const int len = value.size();
	write_int32(offset, len);
	for (int i = 0; i < len; ++i) {
		write_int32(offset + 4 + (i * 4), value[i]);
	}
}

static inline PackedInt64Array read_packed_int64_array(const uint8_t *offset) {
	int len = read_int32(offset);
	if (len < 0) {
		return PackedInt64Array();
	}

	PackedInt64Array out;
	out.resize(len);
	for (int i = 0; i < len; ++i) {
		out.set(i, read_int64(offset + 4 + (i * 8)));
	}
	return out;
}

static inline void write_packed_int64_array(uint8_t *offset, const PackedInt64Array &value) {
	const int len = value.size();
	write_int32(offset, len);
	for (int i = 0; i < len; ++i) {
		write_int64(offset + 4 + (i * 8), value[i]);
	}
}

static inline PackedFloat32Array read_packed_float32_array(const uint8_t *offset) {
	int len = read_int32(offset);
	if (len < 0) {
		return PackedFloat32Array();
	}

	PackedFloat32Array out;
	out.resize(len);
	for (int i = 0; i < len; ++i) {
		out.set(i, read_float(offset + 4 + (i * 4)));
	}
	return out;
}

static inline void write_packed_float32_array(uint8_t *offset, const PackedFloat32Array &value) {
	const int len = value.size();
	write_int32(offset, len);
	for (int i = 0; i < len; ++i) {
		write_float(offset + 4 + (i * 4), value[i]);
	}
}

static inline PackedFloat64Array read_packed_float64_array(const uint8_t *offset) {
	int len = read_int32(offset);
	if (len < 0) {
		return PackedFloat64Array();
	}

	PackedFloat64Array out;
	out.resize(len);
	for (int i = 0; i < len; ++i) {
		out.set(i, read_double(offset + 4 + (i * 8)));
	}
	return out;
}

static inline void write_packed_float64_array(uint8_t *offset, const PackedFloat64Array &value) {
	const int len = value.size();
	write_int32(offset, len);
	for (int i = 0; i < len; ++i) {
		write_double(offset + 4 + (i * 8), value[i]);
	}
}

static inline PackedStringArray read_packed_string_array(const uint8_t *offset) {
	int count = read_int32(offset);
	if (count < 0) {
		return PackedStringArray();
	}

	PackedStringArray out;
	out.resize(count);

	const uint8_t *pos = offset + 4;
	for (int i = 0; i < count; ++i) {
		int str_len = read_int32(pos);
		pos += 4;

		if (str_len < 0 || str_len > 4096) {
			out.set(i, String());
			continue;
		}

		out.set(i, String::utf8(reinterpret_cast<const char *>(pos), str_len));
		pos += str_len;
	}

	return out;
}

static inline void write_packed_string_array(uint8_t *offset, const PackedStringArray &value) {
	const int count = value.size();
	write_int32(offset, count);

	uint8_t *pos = offset + 4;
	for (int i = 0; i < count; ++i) {
		CharString utf8 = value[i].utf8();
		int str_len = utf8.length();

		if (str_len < 0 || str_len > 4096) {
			write_int32(pos, 0);
			pos += 4;
			continue;
		}

		write_int32(pos, str_len);
		pos += 4;

		for (int j = 0; j < str_len; ++j) {
			pos[j] = static_cast<uint8_t>(utf8.ptr()[j]);
		}
		pos += str_len;
	}
}

static inline PackedVector2Array read_packed_vector2_array(const uint8_t *offset) {
	int len = read_int32(offset);
	if (len < 0) {
		return PackedVector2Array();
	}

	PackedVector2Array out;
	out.resize(len);
	for (int i = 0; i < len; ++i) {
		out.set(i, read_vector2(offset + 4 + (i * sizeof(Vector2))));
	}
	return out;
}

static inline void write_packed_vector2_array(uint8_t *offset, const PackedVector2Array &value) {
	const int len = value.size();
	write_int32(offset, len);
	for (int i = 0; i < len; ++i) {
		write_vector2(offset + 4 + (i * sizeof(Vector2)), value[i]);
	}
}

static inline PackedVector3Array read_packed_vector3_array(const uint8_t *offset) {
	int len = read_int32(offset);
	if (len < 0) {
		return PackedVector3Array();
	}

	PackedVector3Array out;
	out.resize(len);
	for (int i = 0; i < len; ++i) {
		out.set(i, read_vector3(offset + 4 + (i * sizeof(Vector3))));
	}
	return out;
}

static inline void write_packed_vector3_array(uint8_t *offset, const PackedVector3Array &value) {
	const int len = value.size();
	write_int32(offset, len);
	for (int i = 0; i < len; ++i) {
		write_vector3(offset + 4 + (i * sizeof(Vector3)), value[i]);
	}
}

static inline PackedColorArray read_packed_color_array(const uint8_t *offset) {
	int len = read_int32(offset);
	if (len < 0) {
		return PackedColorArray();
	}

	PackedColorArray out;
	out.resize(len);
	for (int i = 0; i < len; ++i) {
		out.set(i, read_color(offset + 4 + (i * sizeof(Color))));
	}
	return out;
}

static inline void write_packed_color_array(uint8_t *offset, const PackedColorArray &value) {
	const int len = value.size();
	write_int32(offset, len);
	for (int i = 0; i < len; ++i) {
		write_color(offset + 4 + (i * sizeof(Color)), value[i]);
	}
}

static inline PackedVector4Array read_packed_vector4_array(const uint8_t *offset) {
	int len = read_int32(offset);
	if (len < 0) {
		return PackedVector4Array();
	}

	PackedVector4Array out;
	out.resize(len);
	for (int i = 0; i < len; ++i) {
		out.set(i, read_vector4(offset + 4 + (i * sizeof(Vector4))));
	}
	return out;
}

static inline void write_packed_vector4_array(uint8_t *offset, const PackedVector4Array &value) {
	const int len = value.size();
	write_int32(offset, len);
	for (int i = 0; i < len; ++i) {
		write_vector4(offset + 4 + (i * sizeof(Vector4)), value[i]);
	}
}

// Generic Variant reader.
template <typename T>
static inline T read_variant(const uint8_t *offset) {
	int len = read_int32(offset);
	if (len <= 0) {
		return T();
	}

	Variant v;
	decode_variant(v, offset + 4, len);
	return (T)v;
}

static inline Dictionary read_dictionary(const uint8_t *offset) {
	return read_variant<Dictionary>(offset);
}

static inline Array read_array(const uint8_t *offset) {
	return read_variant<Array>(offset);
}

static inline Callable read_callable(const uint8_t *offset) {
	return read_variant<Callable>(offset);
}

static inline Signal read_signal(const uint8_t *offset) {
	return read_variant<Signal>(offset);
}

static inline void write_variant(uint8_t *offset, const Variant &v) {
	int len = 0;
	encode_variant(v, nullptr, len, false);
	write_int32(offset, len);
	encode_variant(v, offset + 4, len, false);
}

static inline void write_dictionary(uint8_t *offset, const Dictionary &value) {
	write_variant(offset, value);
}

static inline void write_array(uint8_t *offset, const Array &value) {
	write_variant(offset, value);
}

static inline void write_callable(uint8_t *offset, const Callable &value) {
	write_variant(offset, value);
}

static inline void write_signal(uint8_t *offset, const Signal &value) {
	write_variant(offset, value);
}

#endif // WEB_ENABLED
