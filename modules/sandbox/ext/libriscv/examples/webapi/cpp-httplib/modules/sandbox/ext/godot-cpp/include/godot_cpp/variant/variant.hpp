/**************************************************************************/
/*  variant.hpp                                                           */
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

#include <godot_cpp/core/defs.hpp>

#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/builtin_types.hpp>
#include <godot_cpp/variant/variant_size.hpp>

#include <gdextension_interface.h>

#include <array>

namespace godot {

class ObjectID;

class Variant {
	uint8_t opaque[GODOT_CPP_VARIANT_SIZE]{ 0 };

	friend class GDExtensionBinding;
	friend class MethodBind;
	friend class VariantInternal;

	static void init_bindings();

public:
	enum Type {
		NIL,

		// atomic types
		BOOL,
		INT,
		FLOAT,
		STRING,

		// math types
		VECTOR2,
		VECTOR2I,
		RECT2,
		RECT2I,
		VECTOR3,
		VECTOR3I,
		TRANSFORM2D,
		VECTOR4,
		VECTOR4I,
		PLANE,
		QUATERNION,
		AABB,
		BASIS,
		TRANSFORM3D,
		PROJECTION,

		// misc types
		COLOR,
		STRING_NAME,
		NODE_PATH,
		RID,
		OBJECT,
		CALLABLE,
		SIGNAL,
		DICTIONARY,
		ARRAY,

		// typed arrays
		PACKED_BYTE_ARRAY,
		PACKED_INT32_ARRAY,
		PACKED_INT64_ARRAY,
		PACKED_FLOAT32_ARRAY,
		PACKED_FLOAT64_ARRAY,
		PACKED_STRING_ARRAY,
		PACKED_VECTOR2_ARRAY,
		PACKED_VECTOR3_ARRAY,
		PACKED_COLOR_ARRAY,
		PACKED_VECTOR4_ARRAY,

		VARIANT_MAX
	};

	enum Operator {
		// comparison
		OP_EQUAL,
		OP_NOT_EQUAL,
		OP_LESS,
		OP_LESS_EQUAL,
		OP_GREATER,
		OP_GREATER_EQUAL,
		// mathematic
		OP_ADD,
		OP_SUBTRACT,
		OP_MULTIPLY,
		OP_DIVIDE,
		OP_NEGATE,
		OP_POSITIVE,
		OP_MODULE,
		OP_POWER,
		// bitwise
		OP_SHIFT_LEFT,
		OP_SHIFT_RIGHT,
		OP_BIT_AND,
		OP_BIT_OR,
		OP_BIT_XOR,
		OP_BIT_NEGATE,
		// logic
		OP_AND,
		OP_OR,
		OP_XOR,
		OP_NOT,
		// containment
		OP_IN,
		OP_MAX
	};

private:
	static GDExtensionVariantFromTypeConstructorFunc from_type_constructor[VARIANT_MAX];
	static GDExtensionTypeFromVariantConstructorFunc to_type_constructor[VARIANT_MAX];

public:
	_FORCE_INLINE_ GDExtensionVariantPtr _native_ptr() const { return const_cast<uint8_t (*)[GODOT_CPP_VARIANT_SIZE]>(&opaque); }
	Variant();
	Variant(std::nullptr_t n) :
			Variant() {}
	explicit Variant(GDExtensionConstVariantPtr native_ptr);
	Variant(const Variant &other);
	Variant(Variant &&other);
	Variant(bool v);
	Variant(int64_t v);
	Variant(int32_t v) :
			Variant(static_cast<int64_t>(v)) {}
	Variant(int16_t v) :
			Variant(static_cast<int64_t>(v)) {}
	Variant(int8_t v) :
			Variant(static_cast<int64_t>(v)) {}
	Variant(uint64_t v) :
			Variant(static_cast<int64_t>(v)) {}
	Variant(uint32_t v) :
			Variant(static_cast<int64_t>(v)) {}
	Variant(uint16_t v) :
			Variant(static_cast<int64_t>(v)) {}
	Variant(uint8_t v) :
			Variant(static_cast<int64_t>(v)) {}
	Variant(double v);
	Variant(float v) :
			Variant((double)v) {}
	Variant(const String &v);
	Variant(const char *v) :
			Variant(String(v)) {}
	Variant(const char16_t *v) :
			Variant(String(v)) {}
	Variant(const char32_t *v) :
			Variant(String(v)) {}
	Variant(const wchar_t *v) :
			Variant(String(v)) {}
	Variant(const Vector2 &v);
	Variant(const Vector2i &v);
	Variant(const Rect2 &v);
	Variant(const Rect2i &v);
	Variant(const Vector3 &v);
	Variant(const Vector3i &v);
	Variant(const Transform2D &v);
	Variant(const Vector4 &v);
	Variant(const Vector4i &v);
	Variant(const Plane &v);
	Variant(const Quaternion &v);
	Variant(const godot::AABB &v);
	Variant(const Basis &v);
	Variant(const Transform3D &v);
	Variant(const Projection &v);
	Variant(const Color &v);
	Variant(const StringName &v);
	Variant(const NodePath &v);
	Variant(const godot::RID &v);
	Variant(const ObjectID &v);
	Variant(const Object *v);
	Variant(const Callable &v);
	Variant(const Signal &v);
	Variant(const Dictionary &v);
	Variant(const Array &v);
	Variant(const PackedByteArray &v);
	Variant(const PackedInt32Array &v);
	Variant(const PackedInt64Array &v);
	Variant(const PackedFloat32Array &v);
	Variant(const PackedFloat64Array &v);
	Variant(const PackedStringArray &v);
	Variant(const PackedVector2Array &v);
	Variant(const PackedVector3Array &v);
	Variant(const PackedColorArray &v);
	Variant(const PackedVector4Array &v);
	~Variant();

	operator bool() const;
	operator int64_t() const;
	operator int32_t() const;
	operator int16_t() const;
	operator int8_t() const;
	operator uint64_t() const;
	operator uint32_t() const;
	operator uint16_t() const;
	operator uint8_t() const;
	operator double() const;
	operator float() const;
	operator String() const;
	operator Vector2() const;
	operator Vector2i() const;
	operator Rect2() const;
	operator Rect2i() const;
	operator Vector3() const;
	operator Vector3i() const;
	operator Transform2D() const;
	operator Vector4() const;
	operator Vector4i() const;
	operator Plane() const;
	operator Quaternion() const;
	operator godot::AABB() const;
	operator Basis() const;
	operator Transform3D() const;
	operator Projection() const;
	operator Color() const;
	operator StringName() const;
	operator NodePath() const;
	operator godot::RID() const;
	operator ObjectID() const;
	operator Object *() const;
	operator Callable() const;
	operator Signal() const;
	operator Dictionary() const;
	operator Array() const;
	operator PackedByteArray() const;
	operator PackedInt32Array() const;
	operator PackedInt64Array() const;
	operator PackedFloat32Array() const;
	operator PackedFloat64Array() const;
	operator PackedStringArray() const;
	operator PackedVector2Array() const;
	operator PackedVector3Array() const;
	operator PackedColorArray() const;
	operator PackedVector4Array() const;

	Object *get_validated_object() const;

	Variant &operator=(const Variant &other);
	Variant &operator=(Variant &&other);
	bool operator==(const Variant &other) const;
	bool operator!=(const Variant &other) const;
	bool operator<(const Variant &other) const;

	void callp(const StringName &method, const Variant **args, int argcount, Variant &r_ret, GDExtensionCallError &r_error);

	template <typename... Args>
	Variant call(const StringName &method, Args... args) {
		std::array<Variant, sizeof...(args)> vargs = { args... };
		std::array<const Variant *, sizeof...(args)> argptrs;
		for (size_t i = 0; i < vargs.size(); i++) {
			argptrs[i] = &vargs[i];
		}
		Variant result;
		GDExtensionCallError error;
		callp(method, argptrs.data(), argptrs.size(), result, error);
		return result;
	}

	static void callp_static(Variant::Type type, const StringName &method, const Variant **args, int argcount, Variant &r_ret, GDExtensionCallError &r_error);

	template <typename... Args>
	static Variant call_static(Variant::Type type, const StringName &method, Args... args) {
		std::array<Variant, sizeof...(args)> vargs = { args... };
		std::array<const Variant *, sizeof...(args)> argptrs;
		for (size_t i = 0; i < vargs.size(); i++) {
			argptrs[i] = &vargs[i];
		}
		Variant result;
		GDExtensionCallError error;
		callp_static(type, method, argptrs.data(), argptrs.size(), sizeof...(args), result, error);
		return result;
	}

	static void evaluate(const Operator &op, const Variant &a, const Variant &b, Variant &r_ret, bool &r_valid);

	void set(const Variant &key, const Variant &value, bool *r_valid = nullptr);
	void set_named(const StringName &name, const Variant &value, bool &r_valid);
	void set_indexed(int64_t index, const Variant &value, bool &r_valid, bool &r_oob);
	void set_keyed(const Variant &key, const Variant &value, bool &r_valid);
	Variant get(const Variant &key, bool *r_valid = nullptr) const;
	Variant get_named(const StringName &name, bool &r_valid) const;
	Variant get_indexed(int64_t index, bool &r_valid, bool &r_oob) const;
	Variant get_keyed(const Variant &key, bool &r_valid) const;
	bool in(const Variant &index, bool *r_valid = nullptr) const;

	bool iter_init(Variant &r_iter, bool &r_valid) const;
	bool iter_next(Variant &r_iter, bool &r_valid) const;
	Variant iter_get(const Variant &r_iter, bool &r_valid) const;

	Variant::Type get_type() const;
	bool has_method(const StringName &method) const;
	bool has_key(const Variant &key, bool *r_valid = nullptr) const;
	static bool has_member(Variant::Type type, const StringName &member);

	uint32_t hash() const;
	uint32_t recursive_hash(int recursion_count) const;
	bool hash_compare(const Variant &variant) const;
	bool booleanize() const;
	String stringify() const;
	Variant duplicate(bool deep = false) const;

	static String get_type_name(Variant::Type type);
	static bool can_convert(Variant::Type from, Variant::Type to);
	static bool can_convert_strict(Variant::Type from, Variant::Type to);

	void clear();
};

struct VariantHasher {
	static _FORCE_INLINE_ uint32_t hash(const Variant &p_variant) { return p_variant.hash(); }
};

struct VariantComparator {
	static _FORCE_INLINE_ bool compare(const Variant &p_lhs, const Variant &p_rhs) { return p_lhs.hash_compare(p_rhs); }
};

template <typename... VarArgs>
String vformat(const String &p_text, const VarArgs... p_args) {
	Variant args[sizeof...(p_args) + 1] = { p_args..., Variant() }; // +1 makes sure zero sized arrays are also supported.
	Array args_array;
	args_array.resize(sizeof...(p_args));
	for (uint32_t i = 0; i < sizeof...(p_args); i++) {
		args_array[i] = args[i];
	}

	return p_text % args_array;
}

Variant &Array::Iterator::operator*() const {
	return *elem_ptr;
}

Variant *Array::Iterator::operator->() const {
	return elem_ptr;
}

Array::Iterator &Array::Iterator::operator++() {
	elem_ptr++;
	return *this;
}

Array::Iterator &Array::Iterator::operator--() {
	elem_ptr--;
	return *this;
}

const Variant &Array::ConstIterator::operator*() const {
	return *elem_ptr;
}

const Variant *Array::ConstIterator::operator->() const {
	return elem_ptr;
}

Array::ConstIterator &Array::ConstIterator::operator++() {
	elem_ptr++;
	return *this;
}

Array::ConstIterator &Array::ConstIterator::operator--() {
	elem_ptr--;
	return *this;
}

Array::Iterator Array::begin() {
	return Array::Iterator(ptrw());
}
Array::Iterator Array::end() {
	return Array::Iterator(ptrw() + size());
}

Array::ConstIterator Array::begin() const {
	return Array::ConstIterator(ptr());
}
Array::ConstIterator Array::end() const {
	return Array::ConstIterator(ptr() + size());
}

Array::Array(std::initializer_list<Variant> p_init) :
		Array() {
	ERR_FAIL_COND(resize(p_init.size()) != 0);

	size_t i = 0;
	for (const Variant &element : p_init) {
		set(i++, element);
	}
}

#include <godot_cpp/variant/builtin_vararg_methods.hpp>

#ifdef REAL_T_IS_DOUBLE
using PackedRealArray = PackedFloat64Array;
#else
using PackedRealArray = PackedFloat32Array;
#endif // REAL_T_IS_DOUBLE

} // namespace godot
