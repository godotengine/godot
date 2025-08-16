/**************************************************************************/
/*  variant.cpp                                                           */
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

#include <godot_cpp/variant/variant.hpp>

#include <godot_cpp/godot.hpp>

#include <godot_cpp/core/binder_common.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/defs.hpp>
#include <godot_cpp/variant/variant_internal.hpp>

#include <utility>

namespace godot {

GDExtensionVariantFromTypeConstructorFunc Variant::from_type_constructor[Variant::VARIANT_MAX]{};
GDExtensionTypeFromVariantConstructorFunc Variant::to_type_constructor[Variant::VARIANT_MAX]{};

void Variant::init_bindings() {
	// Start from 1 to skip NIL.
	for (int i = 1; i < VARIANT_MAX; i++) {
		from_type_constructor[i] = internal::gdextension_interface_get_variant_from_type_constructor((GDExtensionVariantType)i);
		to_type_constructor[i] = internal::gdextension_interface_get_variant_to_type_constructor((GDExtensionVariantType)i);
	}
	VariantInternal::init_bindings();

	StringName::init_bindings();
	String::init_bindings();
	NodePath::init_bindings();
	RID::init_bindings();
	Callable::init_bindings();
	Signal::init_bindings();
	Dictionary::init_bindings();
	Array::init_bindings();
	PackedByteArray::init_bindings();
	PackedInt32Array::init_bindings();
	PackedInt64Array::init_bindings();
	PackedFloat32Array::init_bindings();
	PackedFloat64Array::init_bindings();
	PackedStringArray::init_bindings();
	PackedVector2Array::init_bindings();
	PackedVector3Array::init_bindings();
	PackedVector4Array::init_bindings();
	PackedColorArray::init_bindings();
}

Variant::Variant() {
	internal::gdextension_interface_variant_new_nil(_native_ptr());
}

Variant::Variant(GDExtensionConstVariantPtr native_ptr) {
	internal::gdextension_interface_variant_new_copy(_native_ptr(), native_ptr);
}

Variant::Variant(const Variant &other) {
	internal::gdextension_interface_variant_new_copy(_native_ptr(), other._native_ptr());
}

Variant::Variant(Variant &&other) {
	std::swap(opaque, other.opaque);
}

Variant::Variant(bool v) {
	GDExtensionBool encoded;
	PtrToArg<bool>::encode(v, &encoded);
	from_type_constructor[BOOL](_native_ptr(), &encoded);
}

Variant::Variant(int64_t v) {
	GDExtensionInt encoded;
	PtrToArg<int64_t>::encode(v, &encoded);
	from_type_constructor[INT](_native_ptr(), &encoded);
}

Variant::Variant(double v) {
	double encoded;
	PtrToArg<double>::encode(v, &encoded);
	from_type_constructor[FLOAT](_native_ptr(), &encoded);
}

Variant::Variant(const String &v) {
	from_type_constructor[STRING](_native_ptr(), v._native_ptr());
}

Variant::Variant(const Vector2 &v) {
	from_type_constructor[VECTOR2](_native_ptr(), (GDExtensionTypePtr)&v);
}

Variant::Variant(const Vector2i &v) {
	from_type_constructor[VECTOR2I](_native_ptr(), (GDExtensionTypePtr)&v);
}

Variant::Variant(const Rect2 &v) {
	from_type_constructor[RECT2](_native_ptr(), (GDExtensionTypePtr)&v);
}

Variant::Variant(const Rect2i &v) {
	from_type_constructor[RECT2I](_native_ptr(), (GDExtensionTypePtr)&v);
}

Variant::Variant(const Vector3 &v) {
	from_type_constructor[VECTOR3](_native_ptr(), (GDExtensionTypePtr)&v);
}

Variant::Variant(const Vector3i &v) {
	from_type_constructor[VECTOR3I](_native_ptr(), (GDExtensionTypePtr)&v);
}

Variant::Variant(const Transform2D &v) {
	from_type_constructor[TRANSFORM2D](_native_ptr(), (GDExtensionTypePtr)&v);
}

Variant::Variant(const Vector4 &v) {
	from_type_constructor[VECTOR4](_native_ptr(), (GDExtensionTypePtr)&v);
}

Variant::Variant(const Vector4i &v) {
	from_type_constructor[VECTOR4I](_native_ptr(), (GDExtensionTypePtr)&v);
}

Variant::Variant(const Plane &v) {
	from_type_constructor[PLANE](_native_ptr(), (GDExtensionTypePtr)&v);
}

Variant::Variant(const Quaternion &v) {
	from_type_constructor[QUATERNION](_native_ptr(), (GDExtensionTypePtr)&v);
}

Variant::Variant(const godot::AABB &v) {
	from_type_constructor[AABB](_native_ptr(), (GDExtensionTypePtr)&v);
}

Variant::Variant(const Basis &v) {
	from_type_constructor[BASIS](_native_ptr(), (GDExtensionTypePtr)&v);
}

Variant::Variant(const Transform3D &v) {
	from_type_constructor[TRANSFORM3D](_native_ptr(), (GDExtensionTypePtr)&v);
}

Variant::Variant(const Projection &v) {
	from_type_constructor[PROJECTION](_native_ptr(), (GDExtensionTypePtr)&v);
}

Variant::Variant(const Color &v) {
	from_type_constructor[COLOR](_native_ptr(), (GDExtensionTypePtr)&v);
}

Variant::Variant(const StringName &v) {
	from_type_constructor[STRING_NAME](_native_ptr(), v._native_ptr());
}

Variant::Variant(const NodePath &v) {
	from_type_constructor[NODE_PATH](_native_ptr(), v._native_ptr());
}

Variant::Variant(const godot::RID &v) {
	from_type_constructor[RID](_native_ptr(), v._native_ptr());
}

Variant::Variant(const Object *v) {
	if (v) {
		from_type_constructor[OBJECT](_native_ptr(), const_cast<GodotObject **>(&v->_owner));
	} else {
		GodotObject *nullobject = nullptr;
		from_type_constructor[OBJECT](_native_ptr(), &nullobject);
	}
}

Variant::Variant(const ObjectID &p_id) :
		Variant(p_id.operator uint64_t()) {
}

Variant::Variant(const Callable &v) {
	from_type_constructor[CALLABLE](_native_ptr(), v._native_ptr());
}

Variant::Variant(const Signal &v) {
	from_type_constructor[SIGNAL](_native_ptr(), v._native_ptr());
}

Variant::Variant(const Dictionary &v) {
	from_type_constructor[DICTIONARY](_native_ptr(), v._native_ptr());
}

Variant::Variant(const Array &v) {
	from_type_constructor[ARRAY](_native_ptr(), v._native_ptr());
}

Variant::Variant(const PackedByteArray &v) {
	from_type_constructor[PACKED_BYTE_ARRAY](_native_ptr(), v._native_ptr());
}

Variant::Variant(const PackedInt32Array &v) {
	from_type_constructor[PACKED_INT32_ARRAY](_native_ptr(), v._native_ptr());
}

Variant::Variant(const PackedInt64Array &v) {
	from_type_constructor[PACKED_INT64_ARRAY](_native_ptr(), v._native_ptr());
}

Variant::Variant(const PackedFloat32Array &v) {
	from_type_constructor[PACKED_FLOAT32_ARRAY](_native_ptr(), v._native_ptr());
}

Variant::Variant(const PackedFloat64Array &v) {
	from_type_constructor[PACKED_FLOAT64_ARRAY](_native_ptr(), v._native_ptr());
}

Variant::Variant(const PackedStringArray &v) {
	from_type_constructor[PACKED_STRING_ARRAY](_native_ptr(), v._native_ptr());
}

Variant::Variant(const PackedVector2Array &v) {
	from_type_constructor[PACKED_VECTOR2_ARRAY](_native_ptr(), v._native_ptr());
}

Variant::Variant(const PackedVector3Array &v) {
	from_type_constructor[PACKED_VECTOR3_ARRAY](_native_ptr(), v._native_ptr());
}

Variant::Variant(const PackedColorArray &v) {
	from_type_constructor[PACKED_COLOR_ARRAY](_native_ptr(), v._native_ptr());
}

Variant::Variant(const PackedVector4Array &v) {
	from_type_constructor[PACKED_VECTOR4_ARRAY](_native_ptr(), v._native_ptr());
}

Variant::~Variant() {
	internal::gdextension_interface_variant_destroy(_native_ptr());
}

Variant::operator bool() const {
	GDExtensionBool result;
	to_type_constructor[BOOL](&result, _native_ptr());
	return PtrToArg<bool>::convert(&result);
}

Variant::operator int64_t() const {
	GDExtensionInt result;
	to_type_constructor[INT](&result, _native_ptr());
	return PtrToArg<int64_t>::convert(&result);
}

Variant::operator int32_t() const {
	return static_cast<int32_t>(operator int64_t());
}

Variant::operator int16_t() const {
	return static_cast<int16_t>(operator int64_t());
}

Variant::operator int8_t() const {
	return static_cast<int8_t>(operator int64_t());
}

Variant::operator uint64_t() const {
	return static_cast<uint64_t>(operator int64_t());
}

Variant::operator uint32_t() const {
	return static_cast<uint32_t>(operator int64_t());
}

Variant::operator uint16_t() const {
	return static_cast<uint16_t>(operator int64_t());
}

Variant::operator uint8_t() const {
	return static_cast<uint8_t>(operator int64_t());
}

Variant::operator double() const {
	double result;
	to_type_constructor[FLOAT](&result, _native_ptr());
	return PtrToArg<double>::convert(&result);
}

Variant::operator float() const {
	return static_cast<float>(operator double());
}

Variant::operator String() const {
	return String(this);
}

Variant::operator Vector2() const {
	// @todo Avoid initializing result before calling constructor (which will initialize it again)
	Vector2 result;
	to_type_constructor[VECTOR2]((GDExtensionTypePtr)&result, _native_ptr());
	return result;
}

Variant::operator Vector2i() const {
	// @todo Avoid initializing result before calling constructor (which will initialize it again)
	Vector2i result;
	to_type_constructor[VECTOR2I]((GDExtensionTypePtr)&result, _native_ptr());
	return result;
}

Variant::operator Rect2() const {
	// @todo Avoid initializing result before calling constructor (which will initialize it again)
	Rect2 result;
	to_type_constructor[RECT2]((GDExtensionTypePtr)&result, _native_ptr());
	return result;
}

Variant::operator Rect2i() const {
	// @todo Avoid initializing result before calling constructor (which will initialize it again)
	Rect2i result;
	to_type_constructor[RECT2I]((GDExtensionTypePtr)&result, _native_ptr());
	return result;
}

Variant::operator Vector3() const {
	// @todo Avoid initializing result before calling constructor (which will initialize it again)
	Vector3 result;
	to_type_constructor[VECTOR3]((GDExtensionTypePtr)&result, _native_ptr());
	return result;
}

Variant::operator Vector3i() const {
	// @todo Avoid initializing result before calling constructor (which will initialize it again)
	Vector3i result;
	to_type_constructor[VECTOR3I]((GDExtensionTypePtr)&result, _native_ptr());
	return result;
}

Variant::operator Transform2D() const {
	// @todo Avoid initializing result before calling constructor (which will initialize it again)
	Transform2D result;
	to_type_constructor[TRANSFORM2D]((GDExtensionTypePtr)&result, _native_ptr());
	return result;
}

Variant::operator Vector4() const {
	// @todo Avoid initializing result before calling constructor (which will initialize it again)
	Vector4 result;
	to_type_constructor[VECTOR4]((GDExtensionTypePtr)&result, _native_ptr());
	return result;
}

Variant::operator Vector4i() const {
	// @todo Avoid initializing result before calling constructor (which will initialize it again)
	Vector4i result;
	to_type_constructor[VECTOR4I]((GDExtensionTypePtr)&result, _native_ptr());
	return result;
}

Variant::operator Plane() const {
	// @todo Avoid initializing result before calling constructor (which will initialize it again)
	Plane result;
	to_type_constructor[PLANE]((GDExtensionTypePtr)&result, _native_ptr());
	return result;
}

Variant::operator Quaternion() const {
	// @todo Avoid initializing result before calling constructor (which will initialize it again)
	Quaternion result;
	to_type_constructor[QUATERNION]((GDExtensionTypePtr)&result, _native_ptr());
	return result;
}

Variant::operator godot::AABB() const {
	// @todo Avoid initializing result before calling constructor (which will initialize it again)
	godot::AABB result;
	to_type_constructor[AABB]((GDExtensionTypePtr)&result, _native_ptr());
	return result;
}

Variant::operator Basis() const {
	// @todo Avoid initializing result before calling constructor (which will initialize it again)
	Basis result;
	to_type_constructor[BASIS]((GDExtensionTypePtr)&result, _native_ptr());
	return result;
}

Variant::operator Transform3D() const {
	// @todo Avoid initializing result before calling constructor (which will initialize it again)
	Transform3D result;
	to_type_constructor[TRANSFORM3D]((GDExtensionTypePtr)&result, _native_ptr());
	return result;
}

Variant::operator Projection() const {
	// @todo Avoid initializing result before calling constructor (which will initialize it again)
	Projection result;
	to_type_constructor[PROJECTION]((GDExtensionTypePtr)&result, _native_ptr());
	return result;
}

Variant::operator Color() const {
	// @todo Avoid initializing result before calling constructor (which will initialize it again)
	Color result;
	to_type_constructor[COLOR]((GDExtensionTypePtr)&result, _native_ptr());
	return result;
}

Variant::operator StringName() const {
	return StringName(this);
}

Variant::operator NodePath() const {
	return NodePath(this);
}

Variant::operator godot::RID() const {
	return godot::RID(this);
}

Variant::operator Object *() const {
	GodotObject *obj;
	to_type_constructor[OBJECT](&obj, _native_ptr());
	if (obj == nullptr) {
		return nullptr;
	}
	return internal::get_object_instance_binding(obj);
}

Variant::operator ObjectID() const {
	if (get_type() == Type::INT) {
		return ObjectID(operator uint64_t());
	} else if (get_type() == Type::OBJECT) {
		return ObjectID(internal::gdextension_interface_variant_get_object_instance_id(_native_ptr()));
	} else {
		return ObjectID();
	}
}

Variant::operator Callable() const {
	return Callable(this);
}

Variant::operator Signal() const {
	return Signal(this);
}

Variant::operator Dictionary() const {
	return Dictionary(this);
}

Variant::operator Array() const {
	return Array(this);
}

Variant::operator PackedByteArray() const {
	return PackedByteArray(this);
}

Variant::operator PackedInt32Array() const {
	return PackedInt32Array(this);
}

Variant::operator PackedInt64Array() const {
	return PackedInt64Array(this);
}

Variant::operator PackedFloat32Array() const {
	return PackedFloat32Array(this);
}

Variant::operator PackedFloat64Array() const {
	return PackedFloat64Array(this);
}

Variant::operator PackedStringArray() const {
	return PackedStringArray(this);
}

Variant::operator PackedVector2Array() const {
	return PackedVector2Array(this);
}

Variant::operator PackedVector3Array() const {
	return PackedVector3Array(this);
}

Variant::operator PackedColorArray() const {
	return PackedColorArray(this);
}

Variant::operator PackedVector4Array() const {
	return PackedVector4Array(this);
}

Object *Variant::get_validated_object() const {
	return ObjectDB::get_instance(operator ObjectID());
}

Variant &Variant::operator=(const Variant &other) {
	clear();
	internal::gdextension_interface_variant_new_copy(_native_ptr(), other._native_ptr());
	return *this;
}

Variant &Variant::operator=(Variant &&other) {
	std::swap(opaque, other.opaque);
	return *this;
}

bool Variant::operator==(const Variant &other) const {
	if (get_type() != other.get_type()) {
		return false;
	}
	bool valid = false;
	Variant result;
	evaluate(OP_EQUAL, *this, other, result, valid);
	return result.operator bool();
}

bool Variant::operator!=(const Variant &other) const {
	if (get_type() != other.get_type()) {
		return true;
	}
	bool valid = false;
	Variant result;
	evaluate(OP_NOT_EQUAL, *this, other, result, valid);
	return result.operator bool();
}

bool Variant::operator<(const Variant &other) const {
	if (get_type() != other.get_type()) {
		return get_type() < other.get_type();
	}
	bool valid = false;
	Variant result;
	evaluate(OP_LESS, *this, other, result, valid);
	return result.operator bool();
}

void Variant::callp(const StringName &method, const Variant **args, int argcount, Variant &r_ret, GDExtensionCallError &r_error) {
	internal::gdextension_interface_variant_call(_native_ptr(), method._native_ptr(), reinterpret_cast<GDExtensionConstVariantPtr *>(args), argcount, r_ret._native_ptr(), &r_error);
}

void Variant::callp_static(Variant::Type type, const StringName &method, const Variant **args, int argcount, Variant &r_ret, GDExtensionCallError &r_error) {
	internal::gdextension_interface_variant_call_static(static_cast<GDExtensionVariantType>(type), method._native_ptr(), reinterpret_cast<GDExtensionConstVariantPtr *>(args), argcount, r_ret._native_ptr(), &r_error);
}

void Variant::evaluate(const Operator &op, const Variant &a, const Variant &b, Variant &r_ret, bool &r_valid) {
	GDExtensionBool valid;
	internal::gdextension_interface_variant_evaluate(static_cast<GDExtensionVariantOperator>(op), a._native_ptr(), b._native_ptr(), r_ret._native_ptr(), &valid);
	r_valid = PtrToArg<bool>::convert(&valid);
}

void Variant::set(const Variant &key, const Variant &value, bool *r_valid) {
	GDExtensionBool valid;
	internal::gdextension_interface_variant_set(_native_ptr(), key._native_ptr(), value._native_ptr(), &valid);
	if (r_valid) {
		*r_valid = PtrToArg<bool>::convert(&valid);
	}
}

void Variant::set_named(const StringName &name, const Variant &value, bool &r_valid) {
	GDExtensionBool valid;
	internal::gdextension_interface_variant_set_named(_native_ptr(), name._native_ptr(), value._native_ptr(), &valid);
	r_valid = PtrToArg<bool>::convert(&valid);
}

void Variant::set_indexed(int64_t index, const Variant &value, bool &r_valid, bool &r_oob) {
	GDExtensionBool valid, oob;
	internal::gdextension_interface_variant_set_indexed(_native_ptr(), index, value._native_ptr(), &valid, &oob);
	r_valid = PtrToArg<bool>::convert(&valid);
	r_oob = PtrToArg<bool>::convert(&oob);
}

void Variant::set_keyed(const Variant &key, const Variant &value, bool &r_valid) {
	GDExtensionBool valid;
	internal::gdextension_interface_variant_set_keyed(_native_ptr(), key._native_ptr(), value._native_ptr(), &valid);
	r_valid = PtrToArg<bool>::convert(&valid);
}

Variant Variant::get(const Variant &key, bool *r_valid) const {
	Variant result;
	GDExtensionBool valid;
	internal::gdextension_interface_variant_get(_native_ptr(), key._native_ptr(), result._native_ptr(), &valid);
	if (r_valid) {
		*r_valid = PtrToArg<bool>::convert(&valid);
	}
	return result;
}

Variant Variant::get_named(const StringName &name, bool &r_valid) const {
	Variant result;
	GDExtensionBool valid;
	internal::gdextension_interface_variant_get_named(_native_ptr(), name._native_ptr(), result._native_ptr(), &valid);
	r_valid = PtrToArg<bool>::convert(&valid);
	return result;
}

Variant Variant::get_indexed(int64_t index, bool &r_valid, bool &r_oob) const {
	Variant result;
	GDExtensionBool valid;
	GDExtensionBool oob;
	internal::gdextension_interface_variant_get_indexed(_native_ptr(), index, result._native_ptr(), &valid, &oob);
	r_valid = PtrToArg<bool>::convert(&valid);
	r_oob = PtrToArg<bool>::convert(&oob);
	return result;
}

Variant Variant::get_keyed(const Variant &key, bool &r_valid) const {
	Variant result;
	GDExtensionBool valid;
	internal::gdextension_interface_variant_get_keyed(_native_ptr(), key._native_ptr(), result._native_ptr(), &valid);
	r_valid = PtrToArg<bool>::convert(&valid);
	return result;
}

bool Variant::in(const Variant &index, bool *r_valid) const {
	Variant result;
	bool valid;
	evaluate(OP_IN, *this, index, result, valid);
	if (r_valid) {
		*r_valid = valid;
	}
	return result.operator bool();
}

bool Variant::iter_init(Variant &r_iter, bool &r_valid) const {
	GDExtensionBool valid;
	GDExtensionBool result = internal::gdextension_interface_variant_iter_init(_native_ptr(), r_iter._native_ptr(), &valid);
	r_valid = PtrToArg<bool>::convert(&valid);
	return PtrToArg<bool>::convert(&result);
}

bool Variant::iter_next(Variant &r_iter, bool &r_valid) const {
	GDExtensionBool valid;
	GDExtensionBool result = internal::gdextension_interface_variant_iter_next(_native_ptr(), r_iter._native_ptr(), &valid);
	r_valid = PtrToArg<bool>::convert(&valid);
	return PtrToArg<bool>::convert(&result);
}

Variant Variant::iter_get(const Variant &r_iter, bool &r_valid) const {
	Variant result;
	GDExtensionBool valid;
	internal::gdextension_interface_variant_iter_get(_native_ptr(), r_iter._native_ptr(), result._native_ptr(), &valid);
	r_valid = PtrToArg<bool>::convert(&valid);
	return result;
}

Variant::Type Variant::get_type() const {
	return static_cast<Variant::Type>(internal::gdextension_interface_variant_get_type(_native_ptr()));
}

bool Variant::has_method(const StringName &method) const {
	GDExtensionBool has = internal::gdextension_interface_variant_has_method(_native_ptr(), method._native_ptr());
	return PtrToArg<bool>::convert(&has);
}

bool Variant::has_key(const Variant &key, bool *r_valid) const {
	GDExtensionBool valid;
	GDExtensionBool has = internal::gdextension_interface_variant_has_key(_native_ptr(), key._native_ptr(), &valid);
	if (r_valid) {
		*r_valid = PtrToArg<bool>::convert(&valid);
	}
	return PtrToArg<bool>::convert(&has);
}

bool Variant::has_member(Variant::Type type, const StringName &member) {
	GDExtensionBool has = internal::gdextension_interface_variant_has_member(static_cast<GDExtensionVariantType>(type), member._native_ptr());
	return PtrToArg<bool>::convert(&has);
}

uint32_t Variant::hash() const {
	GDExtensionInt hash = internal::gdextension_interface_variant_hash(_native_ptr());
	return PtrToArg<uint32_t>::convert(&hash);
}

uint32_t Variant::recursive_hash(int recursion_count) const {
	GDExtensionInt hash = internal::gdextension_interface_variant_recursive_hash(_native_ptr(), recursion_count);
	return PtrToArg<uint32_t>::convert(&hash);
}

bool Variant::hash_compare(const Variant &variant) const {
	GDExtensionBool compare = internal::gdextension_interface_variant_hash_compare(_native_ptr(), variant._native_ptr());
	return PtrToArg<bool>::convert(&compare);
}

bool Variant::booleanize() const {
	GDExtensionBool booleanized = internal::gdextension_interface_variant_booleanize(_native_ptr());
	return PtrToArg<bool>::convert(&booleanized);
}

String Variant::stringify() const {
	String result;
	internal::gdextension_interface_variant_stringify(_native_ptr(), result._native_ptr());
	return result;
}

Variant Variant::duplicate(bool deep) const {
	Variant result;
	GDExtensionBool _deep;
	PtrToArg<bool>::encode(deep, &_deep);
	internal::gdextension_interface_variant_duplicate(_native_ptr(), result._native_ptr(), _deep);
	return result;
}

String Variant::get_type_name(Variant::Type type) {
	String result;
	internal::gdextension_interface_variant_get_type_name(static_cast<GDExtensionVariantType>(type), result._native_ptr());
	return result;
}

bool Variant::can_convert(Variant::Type from, Variant::Type to) {
	GDExtensionBool can = internal::gdextension_interface_variant_can_convert(static_cast<GDExtensionVariantType>(from), static_cast<GDExtensionVariantType>(to));
	return PtrToArg<bool>::convert(&can);
}

bool Variant::can_convert_strict(Variant::Type from, Variant::Type to) {
	GDExtensionBool can = internal::gdextension_interface_variant_can_convert_strict(static_cast<GDExtensionVariantType>(from), static_cast<GDExtensionVariantType>(to));
	return PtrToArg<bool>::convert(&can);
}

void Variant::clear() {
	static const bool needs_deinit[Variant::VARIANT_MAX] = {
		false, // NIL,
		false, // BOOL,
		false, // INT,
		false, // FLOAT,
		true, // STRING,
		false, // VECTOR2,
		false, // VECTOR2I,
		false, // RECT2,
		false, // RECT2I,
		false, // VECTOR3,
		false, // VECTOR3I,
		true, // TRANSFORM2D,
		false, // VECTOR4,
		false, // VECTOR4I,
		false, // PLANE,
		false, // QUATERNION,
		true, // AABB,
		true, // BASIS,
		true, // TRANSFORM3D,
		true, // PROJECTION,

		// misc types
		false, // COLOR,
		true, // STRING_NAME,
		true, // NODE_PATH,
		false, // RID,
		true, // OBJECT,
		true, // CALLABLE,
		true, // SIGNAL,
		true, // DICTIONARY,
		true, // ARRAY,

		// typed arrays
		true, // PACKED_BYTE_ARRAY,
		true, // PACKED_INT32_ARRAY,
		true, // PACKED_INT64_ARRAY,
		true, // PACKED_FLOAT32_ARRAY,
		true, // PACKED_FLOAT64_ARRAY,
		true, // PACKED_STRING_ARRAY,
		true, // PACKED_VECTOR2_ARRAY,
		true, // PACKED_VECTOR3_ARRAY,
		true, // PACKED_COLOR_ARRAY,
	};

	if (unlikely(needs_deinit[get_type()])) { // Make it fast for types that don't need deinit.
		internal::gdextension_interface_variant_destroy(_native_ptr());
	}
	internal::gdextension_interface_variant_new_nil(_native_ptr());
}

} // namespace godot
