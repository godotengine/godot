/*
 This is the serialization and deserialization layer
 */
#pragma once

#include "bridge_api.h"

#include "core/config/engine.h"
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
#include "core/object/class_db.h"
#include "core/object/method_bind.h"
#include "core/object/method_bind_common.h"
#include "core/object/object.h"
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
#include <type_traits>

//reads data from offset
template <typename T>
static inline T reader(const volatile uint8_t *data, int offset) {
	T value{};
	const volatile uint8_t *src = data + offset;
	uint8_t *dst = reinterpret_cast<uint8_t *>(&value);
	for (size_t i = 0; i < sizeof(T); ++i) {
		dst[i] = src[i];
	}
	return value;
}

//writes data to offsets
template <typename T>
static inline void writer(volatile uint8_t *data, int offset, const T &value) {
	volatile uint8_t *dst = data + offset;
	const uint8_t *src = reinterpret_cast<const uint8_t *>(&value);
	for (size_t i = 0; i < sizeof(T); ++i) {
		dst[i] = src[i];
	}
}

static inline String read_string_from_data(const volatile uint8_t *data) {
	int len = reader<int>(data, 0);
	if (len < 0 || len > 4096) {
		return String();
	}

	// data + 4 skips the length header
	const uint8_t *src = const_cast<const uint8_t *>(data + 4);

	// Construct directly from the byte array
	return String::utf8((const char *)src, len);
}

//writes strings t the data offsets
static inline void write_string_to_data(volatile uint8_t *data, const String &value) {
	CharString utf8 = value.utf8();
	int len = utf8.length();
	if (len < 0 || len > 4096) {
		writer<int>(data, 0, 0);
		return;
	}

	writer<int>(data, 0, len);
	for (int i = 0; i < len; ++i) {
		data[4 + i] = static_cast<uint8_t>(utf8.ptr()[i]);
	}
}

//reads ObjectID when called
static inline ObjectID read_object_id(const volatile uint8_t *data, int offset) {
	return ObjectID(reader<uint64_t>(data, offset));
}

//writes an ObjectID when called
static inline void write_object_id(volatile uint8_t *data, int offset, ObjectID id) {
	writer<uint64_t>(data, offset, static_cast<uint64_t>(id));
}

//focuses on reading int32
static inline int read_int32(const volatile uint8_t *data, int offset) {
	return reader<int>(data, offset);
}

//focuses on writing int32
static inline void write_int32(volatile uint8_t *data, int offset, int value) {
	writer<int>(data, offset, value);
}

static inline int64_t read_int64(const volatile uint8_t *data, int offset) {
	return reader<int64_t>(data, offset);
}

static inline void write_int64(volatile uint8_t *data, int offset, int64_t value) {
	writer<int64_t>(data, offset, value);
}

static inline uint64_t read_uint64(const volatile uint8_t *data, int offset) {
	return reader<uint64_t>(data, offset);
}

static inline void write_uint64(volatile uint8_t *data, int offset, uint64_t value) {
	writer<uint64_t>(data, offset, value);
}

static inline float read_float(const volatile uint8_t *data, int offset) {
	return reader<float>(data, offset);
}

static inline void write_float(volatile uint8_t *data, int offset, float value) {
	writer<float>(data, offset, value);
}

static inline double read_double(const volatile uint8_t *data, int offset) {
	return reader<double>(data, offset);
}

static inline void write_double(volatile uint8_t *data, int offset, double value) {
	writer<double>(data, offset, value);
}

static inline bool read_bool(const volatile uint8_t *data, int offset) {
	return read_int32(data, offset) != 0;
}

static inline void write_bool(volatile uint8_t *data, int offset, bool value) {
	write_int32(data, offset, value ? 1 : 0);
}

static inline Vector2 read_vector2(const volatile uint8_t *data, int offset) {
	return reader<Vector2>(data, offset);
}

static inline void write_vector2(volatile uint8_t *data, int offset, const Vector2 &value) {
	writer<Vector2>(data, offset, value);
}

static inline Vector2i read_vector2i(const volatile uint8_t *data, int offset) {
	return reader<Vector2i>(data, offset);
}

static inline void write_vector2i(volatile uint8_t *data, int offset, const Vector2i &value) {
	writer<Vector2i>(data, offset, value);
}

static inline Rect2 read_rect2(const volatile uint8_t *data, int offset) {
	return reader<Rect2>(data, offset);
}

static inline void write_rect2(volatile uint8_t *data, int offset, const Rect2 &value) {
	writer<Rect2>(data, offset, value);
}

static inline Rect2i read_rect2i(const volatile uint8_t *data, int offset) {
	return reader<Rect2i>(data, offset);
}

static inline void write_rect2i(volatile uint8_t *data, int offset, const Rect2i &value) {
	writer<Rect2i>(data, offset, value);
}

static inline Vector3 read_vector3(const volatile uint8_t *data, int offset) {
	return reader<Vector3>(data, offset);
}

static inline void write_vector3(volatile uint8_t *data, int offset, const Vector3 &value) {
	writer<Vector3>(data, offset, value);
}

static inline Vector3i read_vector3i(const volatile uint8_t *data, int offset) {
	return reader<Vector3i>(data, offset);
}

static inline void write_vector3i(volatile uint8_t *data, int offset, const Vector3i &value) {
	writer<Vector3i>(data, offset, value);
}

static inline Transform2D read_transform2d(const volatile uint8_t *data, int offset) {
	return reader<Transform2D>(data, offset);
}

static inline void write_transform2d(volatile uint8_t *data, int offset, const Transform2D &value) {
	writer<Transform2D>(data, offset, value);
}

static inline Vector4 read_vector4(const volatile uint8_t *data, int offset) {
	return reader<Vector4>(data, offset);
}

static inline void write_vector4(volatile uint8_t *data, int offset, const Vector4 &value) {
	writer<Vector4>(data, offset, value);
}

static inline Vector4i read_vector4i(const volatile uint8_t *data, int offset) {
	return reader<Vector4i>(data, offset);
}

static inline void write_vector4i(volatile uint8_t *data, int offset, const Vector4i &value) {
	writer<Vector4i>(data, offset, value);
}

static inline Plane read_plane(const volatile uint8_t *data, int offset) {
	return reader<Plane>(data, offset);
}

static inline void write_plane(volatile uint8_t *data, int offset, const Plane &value) {
	writer<Plane>(data, offset, value);
}

static inline Quaternion read_quaternion(const volatile uint8_t *data, int offset) {
	return Quaternion(
			read_float(data, offset + 0), // w
			read_float(data, offset + 4), // x
			read_float(data, offset + 8), // y
			read_float(data, offset + 12) // z

	);
}

static inline void write_quaternion(volatile uint8_t *data, int offset, const Quaternion &value) {
	write_float(data, offset + 0, value.w);
	write_float(data, offset + 4, value.x);
	write_float(data, offset + 8, value.y);
	write_float(data, offset + 12, value.z);
}

static inline AABB read_aabb(const volatile uint8_t *data, int offset) {
	return reader<AABB>(data, offset);
}

static inline void write_aabb(volatile uint8_t *data, int offset, const AABB &value) {
	writer<AABB>(data, offset, value);
}

static inline Basis read_basis(const volatile uint8_t *data, int offset) {
	return reader<Basis>(data, offset);
}

static inline void write_basis(volatile uint8_t *data, int offset, const Basis &value) {
	writer<Basis>(data, offset, value);
}

static inline Transform3D read_transform3d(const volatile uint8_t *data, int offset) {
	return reader<Transform3D>(data, offset);
}

static inline void write_transform3d(volatile uint8_t *data, int offset, const Transform3D &value) {
	writer<Transform3D>(data, offset, value);
}

static inline Projection read_projection(const volatile uint8_t *data, int offset) {
	return reader<Projection>(data, offset);
}

static inline void write_projection(volatile uint8_t *data, int offset, const Projection &value) {
	writer<Projection>(data, offset, value);
}

static inline Color read_color(const volatile uint8_t *data, int offset) {
	return reader<Color>(data, offset);
}

static inline void write_color(volatile uint8_t *data, int offset, const Color &value) {
	writer<Color>(data, offset, value);
}

static inline StringName read_string_name(const volatile uint8_t *data, int offset) {
	return StringName(read_string_from_data(data + offset));
}

static inline void write_string_name(volatile uint8_t *data, int offset, const StringName &value) {
	write_string_to_data(data + offset, String(value));
}

static inline NodePath read_node_path(const volatile uint8_t *data, int offset) {
	return NodePath(read_string_from_data(data + offset));
}

static inline void write_node_path(volatile uint8_t *data, int offset, const NodePath &value) {
	write_string_to_data(data + offset, String(value));
}

static inline RID read_rid(const volatile uint8_t *data, int offset) {
	return RID::from_uint64(read_uint64(data, offset));
}

static inline void write_rid(volatile uint8_t *data, int offset, RID value) {
	write_uint64(data, offset, value.get_id());
}

static inline PackedByteArray read_packed_byte_array(const volatile uint8_t *data, int offset) {
	int len = read_int32(data, offset);
	if (len < 0) {
		return PackedByteArray();
	}

	PackedByteArray out;
	out.resize(len);
	for (int i = 0; i < len; ++i) {
		out.set(i, static_cast<uint8_t>(data[offset + 4 + i]));
	}
	return out;
}

static inline void write_packed_byte_array(volatile uint8_t *data, int offset, const PackedByteArray &value) {
	const int len = value.size();
	write_int32(data, offset, len);
	for (int i = 0; i < len; ++i) {
		data[offset + 4 + i] = value[i];
	}
}

static inline PackedInt32Array read_packed_int32_array(const volatile uint8_t *data, int offset) {
	int len = read_int32(data, offset);
	if (len < 0) {
		return PackedInt32Array();
	}

	PackedInt32Array out;
	out.resize(len);
	for (int i = 0; i < len; ++i) {
		out.set(i, read_int32(data, offset + 4 + (i * 4)));
	}
	return out;
}

static inline void write_packed_int32_array(volatile uint8_t *data, int offset, const PackedInt32Array &value) {
	const int len = value.size();
	write_int32(data, offset, len);
	for (int i = 0; i < len; ++i) {
		write_int32(data, offset + 4 + (i * 4), value[i]);
	}
}

static inline PackedInt64Array read_packed_int64_array(const volatile uint8_t *data, int offset) {
	int len = read_int32(data, offset);
	if (len < 0) {
		return PackedInt64Array();
	}

	PackedInt64Array out;
	out.resize(len);
	for (int i = 0; i < len; ++i) {
		out.set(i, read_int64(data, offset + 4 + (i * 8)));
	}
	return out;
}

static inline void write_packed_int64_array(volatile uint8_t *data, int offset, const PackedInt64Array &value) {
	const int len = value.size();
	write_int32(data, offset, len);
	for (int i = 0; i < len; ++i) {
		write_int64(data, offset + 4 + (i * 8), value[i]);
	}
}

static inline PackedFloat32Array read_packed_float32_array(const volatile uint8_t *data, int offset) {
	int len = read_int32(data, offset);
	if (len < 0) {
		return PackedFloat32Array();
	}

	PackedFloat32Array out;
	out.resize(len);
	for (int i = 0; i < len; ++i) {
		out.set(i, read_float(data, offset + 4 + (i * 4)));
	}
	return out;
}

static inline void write_packed_float32_array(volatile uint8_t *data, int offset, const PackedFloat32Array &value) {
	const int len = value.size();
	write_int32(data, offset, len);
	for (int i = 0; i < len; ++i) {
		write_float(data, offset + 4 + (i * 4), value[i]);
	}
}

static inline PackedFloat64Array read_packed_float64_array(const volatile uint8_t *data, int offset) {
	int len = read_int32(data, offset);
	if (len < 0) {
		return PackedFloat64Array();
	}

	PackedFloat64Array out;
	out.resize(len);
	for (int i = 0; i < len; ++i) {
		out.set(i, read_double(data, offset + 4 + (i * 8)));
	}
	return out;
}

static inline void write_packed_float64_array(volatile uint8_t *data, int offset, const PackedFloat64Array &value) {
	const int len = value.size();
	write_int32(data, offset, len);
	for (int i = 0; i < len; ++i) {
		write_double(data, offset + 4 + (i * 8), value[i]);
	}
}

//Should be looking at this soon enough
static inline PackedStringArray read_packed_string_array(const volatile uint8_t *data, int offset) {
	int len = read_int32(data, offset);
	if (len < 0) {
		return PackedStringArray();
	}

	PackedStringArray out;
	out.resize(len);
	for (int i = 0; i < len; ++i) {
		out.set(i, read_string_from_data(data + offset + 4 + (i * 4)));
	}
	return out;
}

static inline void write_packed_string_array(volatile uint8_t *data, int offset, const PackedStringArray &value) {
	const int len = value.size();
	write_int32(data, offset, len);
	for (int i = 0; i < len; ++i) {
		write_string_to_data(data + offset + 4 + (i * 4), value[i]);
	}
}

static inline PackedVector2Array read_packed_vector2_array(const volatile uint8_t *data, int offset) {
	int len = read_int32(data, offset);
	if (len < 0) {
		return PackedVector2Array();
	}

	PackedVector2Array out;
	out.resize(len);
	for (int i = 0; i < len; ++i) {
		out.set(i, read_vector2(data, offset + 4 + (i * sizeof(Vector2))));
	}
	return out;
}

static inline void write_packed_vector2_array(volatile uint8_t *data, int offset, const PackedVector2Array &value) {
	const int len = value.size();
	write_int32(data, offset, len);
	for (int i = 0; i < len; ++i) {
		write_vector2(data, offset + 4 + (i * sizeof(Vector2)), value[i]);
	}
}

static inline PackedVector3Array read_packed_vector3_array(const volatile uint8_t *data, int offset) {
	int len = read_int32(data, offset);
	if (len < 0) {
		return PackedVector3Array();
	}

	PackedVector3Array out;
	out.resize(len);
	for (int i = 0; i < len; ++i) {
		out.set(i, read_vector3(data, offset + 4 + (i * sizeof(Vector3))));
	}
	return out;
}

static inline void write_packed_vector3_array(volatile uint8_t *data, int offset, const PackedVector3Array &value) {
	const int len = value.size();
	write_int32(data, offset, len);
	for (int i = 0; i < len; ++i) {
		write_vector3(data, offset + 4 + (i * sizeof(Vector3)), value[i]);
	}
}

static inline PackedColorArray read_packed_color_array(const volatile uint8_t *data, int offset) {
	int len = read_int32(data, offset);
	if (len < 0) {
		return PackedColorArray();
	}

	PackedColorArray out;
	out.resize(len);
	for (int i = 0; i < len; ++i) {
		out.set(i, read_color(data, offset + 4 + (i * sizeof(Color))));
	}
	return out;
}

static inline void write_packed_color_array(volatile uint8_t *data, int offset, const PackedColorArray &value) {
	const int len = value.size();
	write_int32(data, offset, len);
	for (int i = 0; i < len; ++i) {
		write_color(data, offset + 4 + (i * sizeof(Color)), value[i]);
	}
}

static inline PackedVector4Array read_packed_vector4_array(const volatile uint8_t *data, int offset) {
	int len = read_int32(data, offset);
	if (len < 0) {
		return PackedVector4Array();
	}

	PackedVector4Array out;
	out.resize(len);
	for (int i = 0; i < len; ++i) {
		out.set(i, read_vector4(data, offset + 4 + (i * sizeof(Vector4))));
	}
	return out;
}

static inline void write_packed_vector4_array(volatile uint8_t *data, int offset, const PackedVector4Array &value) {
	const int len = value.size();
	write_int32(data, offset, len);
	for (int i = 0; i < len; ++i) {
		write_vector4(data, offset + 4 + (i * sizeof(Vector4)), value[i]);
	}
}

//updates the current status - relies on atomicity to prevent races
static inline void update_status(int offset, int value) {
	__atomic_store(reinterpret_cast<int *>(offset), &value, __ATOMIC_RELEASE);
}

//not fully implemented, they are just placeholders
static inline Dictionary read_dictionary(const volatile uint8_t *data, int offset) {
	(void)data;
	(void)offset;
	return Dictionary();
}

static inline Array read_array(const volatile uint8_t *data, int offset) {
	(void)data;
	(void)offset;
	return Array();
}

static inline Callable read_callable(const volatile uint8_t *data, int offset) {
	(void)data;
	(void)offset;
	return Callable();
}

static inline void write_callable(volatile uint8_t *data, int offset, const Callable &value) {
	(void)data;
	(void)offset;
	(void)value;
}
