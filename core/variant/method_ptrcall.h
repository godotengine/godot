/**************************************************************************/
/*  method_ptrcall.h                                                      */
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

#include "core/object/object_id.h"
#include "core/templates/simple_type.h"
#include "core/typedefs.h"
#include "core/variant/variant.h"

namespace Internal {

template <typename T>
struct PtrToArgDirect {
	_FORCE_INLINE_ static const T &convert(const void *p_ptr) {
		return *reinterpret_cast<const T *>(p_ptr);
	}
	typedef T EncodeT;
	_FORCE_INLINE_ static void encode(T p_val, void *p_ptr) {
		*((T *)p_ptr) = p_val;
	}
};

template <typename T, typename S>
struct PtrToArgConvert {
	_FORCE_INLINE_ static T convert(const void *p_ptr) {
		return static_cast<T>(*reinterpret_cast<const S *>(p_ptr));
	}
	typedef S EncodeT;
	_FORCE_INLINE_ static void encode(T p_val, void *p_ptr) {
		*((S *)p_ptr) = static_cast<S>(p_val);
	}
};

template <typename T>
struct PtrToArgByReference {
	_FORCE_INLINE_ static const T &convert(const void *p_ptr) {
		return *reinterpret_cast<const T *>(p_ptr);
	}
	typedef T EncodeT;
	_FORCE_INLINE_ static void encode(const T &p_val, void *p_ptr) {
		*((T *)p_ptr) = p_val;
	}
};

template <typename T, typename TAlt>
struct PtrToArgVectorConvert {
	_FORCE_INLINE_ static Vector<TAlt> convert(const void *p_ptr) {
		const Vector<T> *dvs = reinterpret_cast<const Vector<T> *>(p_ptr);
		Vector<TAlt> ret;
		int len = dvs->size();
		ret.resize(len);
		{
			const T *r = dvs->ptr();
			for (int i = 0; i < len; i++) {
				ret.write[i] = r[i];
			}
		}
		return ret;
	}
	// No EncodeT because direct pointer conversion not possible.
	_FORCE_INLINE_ static void encode(const Vector<TAlt> &p_vec, void *p_ptr) {
		Vector<T> *dv = reinterpret_cast<Vector<T> *>(p_ptr);
		int len = p_vec.size();
		dv->resize(len);
		{
			T *w = dv->ptrw();
			for (int i = 0; i < len; i++) {
				w[i] = p_vec[i];
			}
		}
	}
};

template <typename T>
struct PtrToArgVectorFromArray {
	_FORCE_INLINE_ static Vector<T> convert(const void *p_ptr) {
		const Array *arr = reinterpret_cast<const Array *>(p_ptr);
		Vector<T> ret;
		int len = arr->size();
		ret.resize(len);
		for (int i = 0; i < len; i++) {
			ret.write[i] = (*arr)[i];
		}
		return ret;
	}
	// No EncodeT because direct pointer conversion not possible.
	_FORCE_INLINE_ static void encode(const Vector<T> &p_vec, void *p_ptr) {
		Array *arr = reinterpret_cast<Array *>(p_ptr);
		int len = p_vec.size();
		arr->resize(len);
		for (int i = 0; i < len; i++) {
			(*arr)[i] = p_vec[i];
		}
	}
};

template <typename T>
struct PtrToArgStringConvertByReference {
	_FORCE_INLINE_ static T convert(const void *p_ptr) {
		T s = *reinterpret_cast<const String *>(p_ptr);
		return s;
	}
	// No EncodeT because direct pointer conversion not possible.
	_FORCE_INLINE_ static void encode(const T &p_vec, void *p_ptr) {
		String *arr = reinterpret_cast<String *>(p_ptr);
		*arr = String(p_vec);
	}
};

} //namespace Internal

template <typename T, typename = void>
struct PtrToArg;

template <typename T>
struct PtrToArg<T, std::enable_if_t<!std::is_same_v<T, GetSimpleTypeT<T>>>> : PtrToArg<GetSimpleTypeT<T>> {};

template <>
struct PtrToArg<bool> : Internal::PtrToArgConvert<bool, uint8_t> {};
// Integer types.
template <>
struct PtrToArg<uint8_t> : Internal::PtrToArgConvert<uint8_t, int64_t> {};
template <>
struct PtrToArg<int8_t> : Internal::PtrToArgConvert<int8_t, int64_t> {};
template <>
struct PtrToArg<uint16_t> : Internal::PtrToArgConvert<uint16_t, int64_t> {};
template <>
struct PtrToArg<int16_t> : Internal::PtrToArgConvert<int16_t, int64_t> {};
template <>
struct PtrToArg<uint32_t> : Internal::PtrToArgConvert<uint32_t, int64_t> {};
template <>
struct PtrToArg<int32_t> : Internal::PtrToArgConvert<int32_t, int64_t> {};
template <>
struct PtrToArg<int64_t> : Internal::PtrToArgDirect<int64_t> {};
template <>
struct PtrToArg<uint64_t> : Internal::PtrToArgDirect<uint64_t> {};
// Float types
template <>
struct PtrToArg<float> : Internal::PtrToArgConvert<float, double> {};
template <>
struct PtrToArg<double> : Internal::PtrToArgDirect<double> {};

template <>
struct PtrToArg<String> : Internal::PtrToArgDirect<String> {};
template <>
struct PtrToArg<Vector2> : Internal::PtrToArgDirect<Vector2> {};
template <>
struct PtrToArg<Vector2i> : Internal::PtrToArgDirect<Vector2i> {};
template <>
struct PtrToArg<Rect2> : Internal::PtrToArgDirect<Rect2> {};
template <>
struct PtrToArg<Rect2i> : Internal::PtrToArgDirect<Rect2i> {};
template <>
struct PtrToArg<Vector3> : Internal::PtrToArgByReference<Vector3> {};
template <>
struct PtrToArg<Vector3i> : Internal::PtrToArgByReference<Vector3i> {};
template <>
struct PtrToArg<Vector4> : Internal::PtrToArgByReference<Vector4> {};
template <>
struct PtrToArg<Vector4i> : Internal::PtrToArgByReference<Vector4i> {};
template <>
struct PtrToArg<Transform2D> : Internal::PtrToArgDirect<Transform2D> {};
template <>
struct PtrToArg<Projection> : Internal::PtrToArgDirect<Projection> {};
template <>
struct PtrToArg<Plane> : Internal::PtrToArgByReference<Plane> {};
template <>
struct PtrToArg<Quaternion> : Internal::PtrToArgDirect<Quaternion> {};
template <>
struct PtrToArg<AABB> : Internal::PtrToArgByReference<AABB> {};
template <>
struct PtrToArg<Basis> : Internal::PtrToArgByReference<Basis> {};
template <>
struct PtrToArg<Transform3D> : Internal::PtrToArgByReference<Transform3D> {};
template <>
struct PtrToArg<Color> : Internal::PtrToArgByReference<Color> {};
template <>
struct PtrToArg<StringName> : Internal::PtrToArgDirect<StringName> {};
template <>
struct PtrToArg<NodePath> : Internal::PtrToArgDirect<NodePath> {};
template <>
struct PtrToArg<RID> : Internal::PtrToArgDirect<RID> {};
// Object doesn't need this.
template <>
struct PtrToArg<Callable> : Internal::PtrToArgDirect<Callable> {};
template <>
struct PtrToArg<Signal> : Internal::PtrToArgDirect<Signal> {};
template <>
struct PtrToArg<Dictionary> : Internal::PtrToArgDirect<Dictionary> {};
template <>
struct PtrToArg<Array> : Internal::PtrToArgDirect<Array> {};
template <>
struct PtrToArg<PackedByteArray> : Internal::PtrToArgDirect<PackedByteArray> {};
template <>
struct PtrToArg<PackedInt32Array> : Internal::PtrToArgDirect<PackedInt32Array> {};
template <>
struct PtrToArg<PackedInt64Array> : Internal::PtrToArgDirect<PackedInt64Array> {};
template <>
struct PtrToArg<PackedFloat32Array> : Internal::PtrToArgDirect<PackedFloat32Array> {};
template <>
struct PtrToArg<PackedFloat64Array> : Internal::PtrToArgDirect<PackedFloat64Array> {};
template <>
struct PtrToArg<PackedStringArray> : Internal::PtrToArgDirect<PackedStringArray> {};
template <>
struct PtrToArg<PackedVector2Array> : Internal::PtrToArgDirect<PackedVector2Array> {};
template <>
struct PtrToArg<PackedVector3Array> : Internal::PtrToArgDirect<PackedVector3Array> {};
template <>
struct PtrToArg<PackedColorArray> : Internal::PtrToArgDirect<PackedColorArray> {};
template <>
struct PtrToArg<PackedVector4Array> : Internal::PtrToArgDirect<PackedVector4Array> {};
template <>
struct PtrToArg<Variant> : Internal::PtrToArgByReference<Variant> {};

template <typename T>
struct PtrToArg<T, std::enable_if_t<std::is_enum_v<T>>> : Internal::PtrToArgConvert<T, int64_t> {};
template <typename T>
struct PtrToArg<BitField<T>, std::enable_if_t<std::is_enum_v<T>>> : Internal::PtrToArgConvert<BitField<T>, int64_t> {};

// This is for Object.

template <typename T>
struct PtrToArg<T *> {
	_FORCE_INLINE_ static T *convert(const void *p_ptr) {
		return likely(p_ptr) ? *reinterpret_cast<T *const *>(p_ptr) : nullptr;
	}
	typedef Object *EncodeT;
	_FORCE_INLINE_ static void encode(T *p_var, void *p_ptr) {
		*((T **)p_ptr) = p_var;
	}
};

template <typename T>
struct PtrToArg<const T *> {
	_FORCE_INLINE_ static const T *convert(const void *p_ptr) {
		return likely(p_ptr) ? *reinterpret_cast<T *const *>(p_ptr) : nullptr;
	}
	typedef const Object *EncodeT;
	_FORCE_INLINE_ static void encode(T *p_var, void *p_ptr) {
		*((T **)p_ptr) = p_var;
	}
};

// This is for ObjectID.

template <>
struct PtrToArg<ObjectID> {
	_FORCE_INLINE_ static const ObjectID convert(const void *p_ptr) {
		return ObjectID(*reinterpret_cast<const uint64_t *>(p_ptr));
	}
	typedef uint64_t EncodeT;
	_FORCE_INLINE_ static void encode(const ObjectID &p_val, void *p_ptr) {
		*((uint64_t *)p_ptr) = p_val;
	}
};

// This is for the special cases used by Variant.

template <>
struct PtrToArg<Vector<StringName>> : Internal::PtrToArgVectorConvert<String, StringName> {};

// For stuff that gets converted to Array vectors.

template <>
struct PtrToArg<Vector<Variant>> : Internal::PtrToArgVectorFromArray<Variant> {};
template <>
struct PtrToArg<Vector<RID>> : Internal::PtrToArgVectorFromArray<RID> {};
template <>
struct PtrToArg<Vector<Plane>> : Internal::PtrToArgVectorFromArray<Plane> {};

// Special case for IPAddress.

template <>
struct PtrToArg<IPAddress> : Internal::PtrToArgStringConvertByReference<IPAddress> {};

template <>
struct PtrToArg<Vector<Face3>> {
	_FORCE_INLINE_ static Vector<Face3> convert(const void *p_ptr) {
		const Vector<Vector3> *dvs = reinterpret_cast<const Vector<Vector3> *>(p_ptr);
		Vector<Face3> ret;
		int len = dvs->size() / 3;
		ret.resize(len);
		{
			const Vector3 *r = dvs->ptr();
			Face3 *w = ret.ptrw();
			for (int i = 0; i < len; i++) {
				w[i].vertex[0] = r[i * 3 + 0];
				w[i].vertex[1] = r[i * 3 + 1];
				w[i].vertex[2] = r[i * 3 + 2];
			}
		}
		return ret;
	}
	// No EncodeT because direct pointer conversion not possible.
	_FORCE_INLINE_ static void encode(const Vector<Face3> &p_vec, void *p_ptr) {
		Vector<Vector3> *arr = reinterpret_cast<Vector<Vector3> *>(p_ptr);
		int len = p_vec.size();
		arr->resize(len * 3);
		{
			const Face3 *r = p_vec.ptr();
			Vector3 *w = arr->ptrw();
			for (int i = 0; i < len; i++) {
				w[i * 3 + 0] = r[i].vertex[0];
				w[i * 3 + 1] = r[i].vertex[1];
				w[i * 3 + 2] = r[i].vertex[2];
			}
		}
	}
};
