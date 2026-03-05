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

#include "variant.h"

#include "core/debugger/engine_debugger.h"
#include "core/io/json.h"
#include "core/io/resource.h"
#include "core/math/math_funcs.h"
#include "core/variant/variant_parser.h"
#include "core/variant/variant_pools.h"

bool Variant::operator==(const Variant &p_variant) const {
	return hash_compare(p_variant);
}

bool Variant::operator<(const Variant &p_variant) const {
	if (type != p_variant.type) { //if types differ, then order by type first
		return type < p_variant.type;
	}
	bool v;
	Variant r;
	evaluate(OP_LESS, *this, p_variant, r, v);
	return r;
}

bool Variant::is_zero() const {
	switch (type) {
		case VariantType::NIL: {
			return true;
		}

		// Atomic types.
		case VariantType::BOOL: {
			return !(_data._bool);
		}
		case VariantType::INT: {
			return _data._int == 0;
		}
		case VariantType::FLOAT: {
			return _data._float == 0;
		}
		case VariantType::STRING: {
			return *reinterpret_cast<const String *>(_data._mem) == String();
		}

		// Math types.
		case VariantType::VECTOR2: {
			return *reinterpret_cast<const Vector2 *>(_data._mem) == Vector2();
		}
		case VariantType::VECTOR2I: {
			return *reinterpret_cast<const Vector2i *>(_data._mem) == Vector2i();
		}
		case VariantType::RECT2: {
			return *reinterpret_cast<const Rect2 *>(_data._mem) == Rect2();
		}
		case VariantType::RECT2I: {
			return *reinterpret_cast<const Rect2i *>(_data._mem) == Rect2i();
		}
		case VariantType::TRANSFORM2D: {
			return *_data._transform2d == Transform2D();
		}
		case VariantType::VECTOR3: {
			return *reinterpret_cast<const Vector3 *>(_data._mem) == Vector3();
		}
		case VariantType::VECTOR3I: {
			return *reinterpret_cast<const Vector3i *>(_data._mem) == Vector3i();
		}
		case VariantType::VECTOR4: {
			return *reinterpret_cast<const Vector4 *>(_data._mem) == Vector4();
		}
		case VariantType::VECTOR4I: {
			return *reinterpret_cast<const Vector4i *>(_data._mem) == Vector4i();
		}
		case VariantType::PLANE: {
			return *reinterpret_cast<const Plane *>(_data._mem) == Plane();
		}
		case VariantType::AABB: {
			return *_data._aabb == ::AABB();
		}
		case VariantType::QUATERNION: {
			return *reinterpret_cast<const Quaternion *>(_data._mem) == Quaternion();
		}
		case VariantType::BASIS: {
			return *_data._basis == Basis();
		}
		case VariantType::TRANSFORM3D: {
			return *_data._transform3d == Transform3D();
		}
		case VariantType::PROJECTION: {
			return *_data._projection == Projection();
		}

		// Miscellaneous types.
		case VariantType::COLOR: {
			return *reinterpret_cast<const Color *>(_data._mem) == Color();
		}
		case VariantType::RID: {
			return *reinterpret_cast<const ::RID *>(_data._mem) == ::RID();
		}
		case VariantType::OBJECT: {
			return get_validated_object() == nullptr;
		}
		case VariantType::CALLABLE: {
			return reinterpret_cast<const Callable *>(_data._mem)->is_null();
		}
		case VariantType::SIGNAL: {
			return reinterpret_cast<const Signal *>(_data._mem)->is_null();
		}
		case VariantType::STRING_NAME: {
			return *reinterpret_cast<const StringName *>(_data._mem) == StringName();
		}
		case VariantType::NODE_PATH: {
			return reinterpret_cast<const NodePath *>(_data._mem)->is_empty();
		}
		case VariantType::DICTIONARY: {
			return reinterpret_cast<const Dictionary *>(_data._mem)->is_empty();
		}
		case VariantType::ARRAY: {
			return reinterpret_cast<const Array *>(_data._mem)->is_empty();
		}

		// Arrays.
		case VariantType::PACKED_BYTE_ARRAY: {
			return PackedArrayRef<uint8_t>::get_array(_data.packed_array).is_empty();
		}
		case VariantType::PACKED_INT32_ARRAY: {
			return PackedArrayRef<int32_t>::get_array(_data.packed_array).is_empty();
		}
		case VariantType::PACKED_INT64_ARRAY: {
			return PackedArrayRef<int64_t>::get_array(_data.packed_array).is_empty();
		}
		case VariantType::PACKED_FLOAT32_ARRAY: {
			return PackedArrayRef<float>::get_array(_data.packed_array).is_empty();
		}
		case VariantType::PACKED_FLOAT64_ARRAY: {
			return PackedArrayRef<double>::get_array(_data.packed_array).is_empty();
		}
		case VariantType::PACKED_STRING_ARRAY: {
			return PackedArrayRef<String>::get_array(_data.packed_array).is_empty();
		}
		case VariantType::PACKED_VECTOR2_ARRAY: {
			return PackedArrayRef<Vector2>::get_array(_data.packed_array).is_empty();
		}
		case VariantType::PACKED_VECTOR3_ARRAY: {
			return PackedArrayRef<Vector3>::get_array(_data.packed_array).is_empty();
		}
		case VariantType::PACKED_COLOR_ARRAY: {
			return PackedArrayRef<Color>::get_array(_data.packed_array).is_empty();
		}
		case VariantType::PACKED_VECTOR4_ARRAY: {
			return PackedArrayRef<Vector4>::get_array(_data.packed_array).is_empty();
		}
		default: {
		}
	}

	return false;
}

bool Variant::is_one() const {
	switch (type) {
		case VariantType::NIL: {
			return true;
		}

		case VariantType::BOOL: {
			return _data._bool;
		}
		case VariantType::INT: {
			return _data._int == 1;
		}
		case VariantType::FLOAT: {
			return _data._float == 1;
		}

		case VariantType::VECTOR2: {
			return *reinterpret_cast<const Vector2 *>(_data._mem) == Vector2(1, 1);
		}
		case VariantType::VECTOR2I: {
			return *reinterpret_cast<const Vector2i *>(_data._mem) == Vector2i(1, 1);
		}
		case VariantType::RECT2: {
			return *reinterpret_cast<const Rect2 *>(_data._mem) == Rect2(1, 1, 1, 1);
		}
		case VariantType::RECT2I: {
			return *reinterpret_cast<const Rect2i *>(_data._mem) == Rect2i(1, 1, 1, 1);
		}
		case VariantType::VECTOR3: {
			return *reinterpret_cast<const Vector3 *>(_data._mem) == Vector3(1, 1, 1);
		}
		case VariantType::VECTOR3I: {
			return *reinterpret_cast<const Vector3i *>(_data._mem) == Vector3i(1, 1, 1);
		}
		case VariantType::VECTOR4: {
			return *reinterpret_cast<const Vector4 *>(_data._mem) == Vector4(1, 1, 1, 1);
		}
		case VariantType::VECTOR4I: {
			return *reinterpret_cast<const Vector4i *>(_data._mem) == Vector4i(1, 1, 1, 1);
		}
		case VariantType::PLANE: {
			return *reinterpret_cast<const Plane *>(_data._mem) == Plane(1, 1, 1, 1);
		}

		case VariantType::COLOR: {
			return *reinterpret_cast<const Color *>(_data._mem) == Color(1, 1, 1, 1);
		}

		default: {
			return !is_zero();
		}
	}
}

bool Variant::is_null() const {
	if (type == VariantType::OBJECT && _get_obj().obj) {
		return false;
	} else {
		return true;
	}
}

void Variant::ObjData::ref(const ObjData &p_from) {
	// Mirrors Ref::ref in refcounted.h
	if (p_from.id == id) {
		return;
	}

	ObjData cleanup_ref = *this;

	*this = p_from;
	if (id.is_ref_counted()) {
		RefCounted *reference = static_cast<RefCounted *>(obj);
		// Assuming reference is not null because id.is_ref_counted() was true.
		if (!reference->reference()) {
			*this = ObjData();
		}
	}

	cleanup_ref.unref();
}

void Variant::ObjData::ref_pointer(Object *p_object) {
	// Mirrors Ref::ref_pointer in refcounted.h
	if (p_object == obj) {
		return;
	}

	ObjData cleanup_ref = *this;

	if (p_object) {
		*this = ObjData{ p_object->get_instance_id(), p_object };
		if (p_object->is_ref_counted()) {
			RefCounted *reference = static_cast<RefCounted *>(p_object);
			if (!reference->init_ref()) {
				*this = ObjData();
			}
		}
	} else {
		*this = ObjData();
	}

	cleanup_ref.unref();
}

void Variant::ObjData::unref() {
	// Mirrors Ref::unref in refcounted.h
	if (id.is_ref_counted()) {
		RefCounted *reference = static_cast<RefCounted *>(obj);
		// Assuming reference is not null because id.is_ref_counted() was true.
		if (reference->unreference()) {
			memdelete(reference);
		}
	}
	*this = ObjData();
}

void Variant::reference(const Variant &p_variant) {
	if (type == VariantType::OBJECT && p_variant.type == VariantType::OBJECT) {
		_get_obj().ref(p_variant._get_obj());
		return;
	}

	clear();

	type = p_variant.type;

	switch (p_variant.type) {
		case VariantType::NIL: {
			// None.
		} break;

		// Atomic types.
		case VariantType::BOOL: {
			_data._bool = p_variant._data._bool;
		} break;
		case VariantType::INT: {
			_data._int = p_variant._data._int;
		} break;
		case VariantType::FLOAT: {
			_data._float = p_variant._data._float;
		} break;
		case VariantType::STRING: {
			memnew_placement(_data._mem, String(*reinterpret_cast<const String *>(p_variant._data._mem)));
		} break;

		// Math types.
		case VariantType::VECTOR2: {
			memnew_placement(_data._mem, Vector2(*reinterpret_cast<const Vector2 *>(p_variant._data._mem)));
		} break;
		case VariantType::VECTOR2I: {
			memnew_placement(_data._mem, Vector2i(*reinterpret_cast<const Vector2i *>(p_variant._data._mem)));
		} break;
		case VariantType::RECT2: {
			memnew_placement(_data._mem, Rect2(*reinterpret_cast<const Rect2 *>(p_variant._data._mem)));
		} break;
		case VariantType::RECT2I: {
			memnew_placement(_data._mem, Rect2i(*reinterpret_cast<const Rect2i *>(p_variant._data._mem)));
		} break;
		case VariantType::TRANSFORM2D: {
			_data._transform2d = VariantPools::alloc<Transform2D>();
			memnew_placement(_data._transform2d, Transform2D(*p_variant._data._transform2d));
		} break;
		case VariantType::VECTOR3: {
			memnew_placement(_data._mem, Vector3(*reinterpret_cast<const Vector3 *>(p_variant._data._mem)));
		} break;
		case VariantType::VECTOR3I: {
			memnew_placement(_data._mem, Vector3i(*reinterpret_cast<const Vector3i *>(p_variant._data._mem)));
		} break;
		case VariantType::VECTOR4: {
			memnew_placement(_data._mem, Vector4(*reinterpret_cast<const Vector4 *>(p_variant._data._mem)));
		} break;
		case VariantType::VECTOR4I: {
			memnew_placement(_data._mem, Vector4i(*reinterpret_cast<const Vector4i *>(p_variant._data._mem)));
		} break;
		case VariantType::PLANE: {
			memnew_placement(_data._mem, Plane(*reinterpret_cast<const Plane *>(p_variant._data._mem)));
		} break;
		case VariantType::AABB: {
			_data._aabb = VariantPools::alloc<::AABB>();
			memnew_placement(_data._aabb, ::AABB(*p_variant._data._aabb));
		} break;
		case VariantType::QUATERNION: {
			memnew_placement(_data._mem, Quaternion(*reinterpret_cast<const Quaternion *>(p_variant._data._mem)));
		} break;
		case VariantType::BASIS: {
			_data._ptr = VariantPools::alloc<Basis>();
			memnew_placement(_data._basis, Basis(*p_variant._data._basis));
		} break;
		case VariantType::TRANSFORM3D: {
			_data._ptr = VariantPools::alloc<Transform3D>();
			memnew_placement(_data._transform3d, Transform3D(*p_variant._data._transform3d));
		} break;
		case VariantType::PROJECTION: {
			_data._ptr = VariantPools::alloc<Projection>();
			memnew_placement(_data._projection, Projection(*p_variant._data._projection));
		} break;

		// Miscellaneous types.
		case VariantType::COLOR: {
			memnew_placement(_data._mem, Color(*reinterpret_cast<const Color *>(p_variant._data._mem)));
		} break;
		case VariantType::RID: {
			memnew_placement(_data._mem, ::RID(*reinterpret_cast<const ::RID *>(p_variant._data._mem)));
		} break;
		case VariantType::OBJECT: {
			memnew_placement(_data._mem, ObjData);
			_get_obj().ref(p_variant._get_obj());
		} break;
		case VariantType::CALLABLE: {
			memnew_placement(_data._mem, Callable(*reinterpret_cast<const Callable *>(p_variant._data._mem)));
		} break;
		case VariantType::SIGNAL: {
			memnew_placement(_data._mem, Signal(*reinterpret_cast<const Signal *>(p_variant._data._mem)));
		} break;
		case VariantType::STRING_NAME: {
			memnew_placement(_data._mem, StringName(*reinterpret_cast<const StringName *>(p_variant._data._mem)));
		} break;
		case VariantType::NODE_PATH: {
			memnew_placement(_data._mem, NodePath(*reinterpret_cast<const NodePath *>(p_variant._data._mem)));
		} break;
		case VariantType::DICTIONARY: {
			memnew_placement(_data._mem, Dictionary(*reinterpret_cast<const Dictionary *>(p_variant._data._mem)));
		} break;
		case VariantType::ARRAY: {
			memnew_placement(_data._mem, Array(*reinterpret_cast<const Array *>(p_variant._data._mem)));
		} break;

		// Arrays.
		case VariantType::PACKED_BYTE_ARRAY: {
			_data.packed_array = static_cast<PackedArrayRef<uint8_t> *>(p_variant._data.packed_array)->reference();
			if (!_data.packed_array) {
				_data.packed_array = PackedArrayRef<uint8_t>::create();
			}
		} break;
		case VariantType::PACKED_INT32_ARRAY: {
			_data.packed_array = static_cast<PackedArrayRef<int32_t> *>(p_variant._data.packed_array)->reference();
			if (!_data.packed_array) {
				_data.packed_array = PackedArrayRef<int32_t>::create();
			}
		} break;
		case VariantType::PACKED_INT64_ARRAY: {
			_data.packed_array = static_cast<PackedArrayRef<int64_t> *>(p_variant._data.packed_array)->reference();
			if (!_data.packed_array) {
				_data.packed_array = PackedArrayRef<int64_t>::create();
			}
		} break;
		case VariantType::PACKED_FLOAT32_ARRAY: {
			_data.packed_array = static_cast<PackedArrayRef<float> *>(p_variant._data.packed_array)->reference();
			if (!_data.packed_array) {
				_data.packed_array = PackedArrayRef<float>::create();
			}
		} break;
		case VariantType::PACKED_FLOAT64_ARRAY: {
			_data.packed_array = static_cast<PackedArrayRef<double> *>(p_variant._data.packed_array)->reference();
			if (!_data.packed_array) {
				_data.packed_array = PackedArrayRef<double>::create();
			}
		} break;
		case VariantType::PACKED_STRING_ARRAY: {
			_data.packed_array = static_cast<PackedArrayRef<String> *>(p_variant._data.packed_array)->reference();
			if (!_data.packed_array) {
				_data.packed_array = PackedArrayRef<String>::create();
			}
		} break;
		case VariantType::PACKED_VECTOR2_ARRAY: {
			_data.packed_array = static_cast<PackedArrayRef<Vector2> *>(p_variant._data.packed_array)->reference();
			if (!_data.packed_array) {
				_data.packed_array = PackedArrayRef<Vector2>::create();
			}
		} break;
		case VariantType::PACKED_VECTOR3_ARRAY: {
			_data.packed_array = static_cast<PackedArrayRef<Vector3> *>(p_variant._data.packed_array)->reference();
			if (!_data.packed_array) {
				_data.packed_array = PackedArrayRef<Vector3>::create();
			}
		} break;
		case VariantType::PACKED_COLOR_ARRAY: {
			_data.packed_array = static_cast<PackedArrayRef<Color> *>(p_variant._data.packed_array)->reference();
			if (!_data.packed_array) {
				_data.packed_array = PackedArrayRef<Color>::create();
			}
		} break;
		case VariantType::PACKED_VECTOR4_ARRAY: {
			_data.packed_array = static_cast<PackedArrayRef<Vector4> *>(p_variant._data.packed_array)->reference();
			if (!_data.packed_array) {
				_data.packed_array = PackedArrayRef<Vector4>::create();
			}
		} break;
		default: {
		}
	}
}

void Variant::zero() {
	switch (type) {
		case VariantType::NIL:
			break;
		case VariantType::BOOL:
			_data._bool = false;
			break;
		case VariantType::INT:
			_data._int = 0;
			break;
		case VariantType::FLOAT:
			_data._float = 0;
			break;

		case VariantType::VECTOR2:
			*reinterpret_cast<Vector2 *>(_data._mem) = Vector2();
			break;
		case VariantType::VECTOR2I:
			*reinterpret_cast<Vector2i *>(_data._mem) = Vector2i();
			break;
		case VariantType::RECT2:
			*reinterpret_cast<Rect2 *>(_data._mem) = Rect2();
			break;
		case VariantType::RECT2I:
			*reinterpret_cast<Rect2i *>(_data._mem) = Rect2i();
			break;
		case VariantType::VECTOR3:
			*reinterpret_cast<Vector3 *>(_data._mem) = Vector3();
			break;
		case VariantType::VECTOR3I:
			*reinterpret_cast<Vector3i *>(_data._mem) = Vector3i();
			break;
		case VariantType::VECTOR4:
			*reinterpret_cast<Vector4 *>(_data._mem) = Vector4();
			break;
		case VariantType::VECTOR4I:
			*reinterpret_cast<Vector4i *>(_data._mem) = Vector4i();
			break;
		case VariantType::PLANE:
			*reinterpret_cast<Plane *>(_data._mem) = Plane();
			break;
		case VariantType::QUATERNION:
			*reinterpret_cast<Quaternion *>(_data._mem) = Quaternion();
			break;

		case VariantType::COLOR:
			*reinterpret_cast<Color *>(_data._mem) = Color();
			break;

		default:
			VariantType::Type prev_type = type;
			clear();
			if (type != prev_type) {
				// clear() changes type to NIL, so it needs to be restored.
				Callable::CallError ce;
				Variant::construct(prev_type, *this, nullptr, 0, ce);
			}
			break;
	}
}

void Variant::_clear_internal() {
	switch (type) {
		case VariantType::STRING: {
			reinterpret_cast<String *>(_data._mem)->~String();
		} break;

		// Math types.
		case VariantType::TRANSFORM2D: {
			if (_data._transform2d) {
				_data._transform2d->~Transform2D();
				VariantPools::free(_data._transform2d);
				_data._transform2d = nullptr;
			}
		} break;
		case VariantType::AABB: {
			if (_data._aabb) {
				_data._aabb->~AABB();
				VariantPools::free(_data._aabb);
				_data._aabb = nullptr;
			}
		} break;
		case VariantType::BASIS: {
			if (_data._basis) {
				_data._basis->~Basis();
				VariantPools::free(_data._basis);
				_data._basis = nullptr;
			}
		} break;
		case VariantType::TRANSFORM3D: {
			if (_data._transform3d) {
				_data._transform3d->~Transform3D();
				VariantPools::free(_data._transform3d);
				_data._transform3d = nullptr;
			}
		} break;
		case VariantType::PROJECTION: {
			if (_data._projection) {
				_data._projection->~Projection();
				VariantPools::free(_data._projection);
				_data._projection = nullptr;
			}
		} break;

		// Miscellaneous types.
		case VariantType::STRING_NAME: {
			reinterpret_cast<StringName *>(_data._mem)->~StringName();
		} break;
		case VariantType::NODE_PATH: {
			reinterpret_cast<NodePath *>(_data._mem)->~NodePath();
		} break;
		case VariantType::OBJECT: {
			_get_obj().unref();
		} break;
		case VariantType::RID: {
			// Not much need probably.
			// HACK: Can't seem to use destructor + scoping operator, so hack.
			typedef ::RID RID_Class;
			reinterpret_cast<RID_Class *>(_data._mem)->~RID_Class();
		} break;
		case VariantType::CALLABLE: {
			reinterpret_cast<Callable *>(_data._mem)->~Callable();
		} break;
		case VariantType::SIGNAL: {
			reinterpret_cast<Signal *>(_data._mem)->~Signal();
		} break;
		case VariantType::DICTIONARY: {
			reinterpret_cast<Dictionary *>(_data._mem)->~Dictionary();
		} break;
		case VariantType::ARRAY: {
			reinterpret_cast<Array *>(_data._mem)->~Array();
		} break;

		// Arrays.
		case VariantType::PACKED_BYTE_ARRAY: {
			PackedArrayRefBase::destroy(_data.packed_array);
		} break;
		case VariantType::PACKED_INT32_ARRAY: {
			PackedArrayRefBase::destroy(_data.packed_array);
		} break;
		case VariantType::PACKED_INT64_ARRAY: {
			PackedArrayRefBase::destroy(_data.packed_array);
		} break;
		case VariantType::PACKED_FLOAT32_ARRAY: {
			PackedArrayRefBase::destroy(_data.packed_array);
		} break;
		case VariantType::PACKED_FLOAT64_ARRAY: {
			PackedArrayRefBase::destroy(_data.packed_array);
		} break;
		case VariantType::PACKED_STRING_ARRAY: {
			PackedArrayRefBase::destroy(_data.packed_array);
		} break;
		case VariantType::PACKED_VECTOR2_ARRAY: {
			PackedArrayRefBase::destroy(_data.packed_array);
		} break;
		case VariantType::PACKED_VECTOR3_ARRAY: {
			PackedArrayRefBase::destroy(_data.packed_array);
		} break;
		case VariantType::PACKED_COLOR_ARRAY: {
			PackedArrayRefBase::destroy(_data.packed_array);
		} break;
		case VariantType::PACKED_VECTOR4_ARRAY: {
			PackedArrayRefBase::destroy(_data.packed_array);
		} break;
		default: {
			// Not needed, there is no point. The following do not allocate memory:
			// VECTOR2, VECTOR3, VECTOR4, RECT2, PLANE, QUATERNION, COLOR.
		}
	}
}

Variant::operator int64_t() const {
	return _to_int<int64_t>();
}

Variant::operator int32_t() const {
	return _to_int<int32_t>();
}

Variant::operator int16_t() const {
	return _to_int<int16_t>();
}

Variant::operator int8_t() const {
	return _to_int<int8_t>();
}

Variant::operator Math::int_alt_t() const {
	return _to_int<Math::int_alt_t>();
}

Variant::operator uint64_t() const {
	return _to_int<uint64_t>();
}

Variant::operator uint32_t() const {
	return _to_int<uint32_t>();
}

Variant::operator uint16_t() const {
	return _to_int<uint16_t>();
}

Variant::operator uint8_t() const {
	return _to_int<uint8_t>();
}

Variant::operator Math::uint_alt_t() const {
	return _to_int<Math::uint_alt_t>();
}

Variant::operator ObjectID() const {
	if (type == VariantType::INT) {
		return ObjectID(_data._int);
	} else if (type == VariantType::OBJECT) {
		return _get_obj().id;
	} else {
		return ObjectID();
	}
}

Variant::operator char32_t() const {
	return operator uint32_t();
}

Variant::operator float() const {
	return _to_float<float>();
}

Variant::operator double() const {
	return _to_float<double>();
}

Variant::operator StringName() const {
	if (type == VariantType::STRING_NAME) {
		return *reinterpret_cast<const StringName *>(_data._mem);
	} else if (type == VariantType::STRING) {
		return *reinterpret_cast<const String *>(_data._mem);
	}

	return StringName();
}

struct _VariantStrPair {
	String key;
	String value;

	bool operator<(const _VariantStrPair &p) const {
		return key < p.key;
	}
};

Variant::operator String() const {
	return stringify(0);
}

String stringify_variant_clean(const Variant &p_variant, int recursion_count) {
	String s = p_variant.stringify(recursion_count);

	// Wrap strings in quotes to avoid ambiguity.
	switch (p_variant.get_type()) {
		case VariantType::STRING: {
			s = s.c_escape().quote();
		} break;
		case VariantType::STRING_NAME: {
			s = "&" + s.c_escape().quote();
		} break;
		case VariantType::NODE_PATH: {
			s = "^" + s.c_escape().quote();
		} break;
		default: {
		} break;
	}

	return s;
}

template <typename T>
String stringify_vector(const T &vec, int recursion_count) {
	String str("[");
	for (int i = 0; i < vec.size(); i++) {
		if (i > 0) {
			str += ", ";
		}

		str += stringify_variant_clean(vec[i], recursion_count);
	}
	str += "]";
	return str;
}

String Variant::stringify(int recursion_count) const {
	switch (type) {
		case VariantType::NIL:
			return "<null>";
		case VariantType::BOOL:
			return _data._bool ? "true" : "false";
		case VariantType::INT:
			return itos(_data._int);
		case VariantType::FLOAT:
			return String::num_real(_data._float, true);
		case VariantType::STRING:
			return *reinterpret_cast<const String *>(_data._mem);
		case VariantType::VECTOR2:
			return String(operator Vector2());
		case VariantType::VECTOR2I:
			return String(operator Vector2i());
		case VariantType::RECT2:
			return String(operator Rect2());
		case VariantType::RECT2I:
			return String(operator Rect2i());
		case VariantType::TRANSFORM2D:
			return String(operator Transform2D());
		case VariantType::VECTOR3:
			return String(operator Vector3());
		case VariantType::VECTOR3I:
			return String(operator Vector3i());
		case VariantType::VECTOR4:
			return String(operator Vector4());
		case VariantType::VECTOR4I:
			return String(operator Vector4i());
		case VariantType::PLANE:
			return String(operator Plane());
		case VariantType::AABB:
			return String(operator ::AABB());
		case VariantType::QUATERNION:
			return String(operator Quaternion());
		case VariantType::BASIS:
			return String(operator Basis());
		case VariantType::TRANSFORM3D:
			return String(operator Transform3D());
		case VariantType::PROJECTION:
			return String(operator Projection());
		case VariantType::STRING_NAME:
			return operator StringName();
		case VariantType::NODE_PATH:
			return String(operator NodePath());
		case VariantType::COLOR:
			return String(operator Color());
		case VariantType::DICTIONARY: {
			ERR_FAIL_COND_V_MSG(recursion_count > MAX_RECURSION, "{ ... }", "Maximum dictionary recursion reached!");
			recursion_count++;

			const Dictionary &d = *reinterpret_cast<const Dictionary *>(_data._mem);

			// Add leading and trailing space to Dictionary printing. This distinguishes it
			// from array printing on fonts that have similar-looking {} and [] characters.
			String str("{ ");

			Vector<_VariantStrPair> pairs;

			for (const KeyValue<Variant, Variant> &kv : d) {
				_VariantStrPair sp;
				sp.key = stringify_variant_clean(kv.key, recursion_count);
				sp.value = stringify_variant_clean(kv.value, recursion_count);

				pairs.push_back(sp);
			}

			for (int i = 0; i < pairs.size(); i++) {
				if (i > 0) {
					str += ", ";
				}
				str += pairs[i].key + ": " + pairs[i].value;
			}
			str += " }";

			return str;
		}
		// Packed arrays cannot contain recursive structures, the recursion_count increment is not needed.
		case VariantType::PACKED_VECTOR2_ARRAY: {
			return stringify_vector(operator PackedVector2Array(), recursion_count);
		}
		case VariantType::PACKED_VECTOR3_ARRAY: {
			return stringify_vector(operator PackedVector3Array(), recursion_count);
		}
		case VariantType::PACKED_COLOR_ARRAY: {
			return stringify_vector(operator PackedColorArray(), recursion_count);
		}
		case VariantType::PACKED_VECTOR4_ARRAY: {
			return stringify_vector(operator PackedVector4Array(), recursion_count);
		}
		case VariantType::PACKED_STRING_ARRAY: {
			return stringify_vector(operator PackedStringArray(), recursion_count);
		}
		case VariantType::PACKED_BYTE_ARRAY: {
			return stringify_vector(operator PackedByteArray(), recursion_count);
		}
		case VariantType::PACKED_INT32_ARRAY: {
			return stringify_vector(operator PackedInt32Array(), recursion_count);
		}
		case VariantType::PACKED_INT64_ARRAY: {
			return stringify_vector(operator PackedInt64Array(), recursion_count);
		}
		case VariantType::PACKED_FLOAT32_ARRAY: {
			return stringify_vector(operator PackedFloat32Array(), recursion_count);
		}
		case VariantType::PACKED_FLOAT64_ARRAY: {
			return stringify_vector(operator PackedFloat64Array(), recursion_count);
		}
		case VariantType::ARRAY: {
			ERR_FAIL_COND_V_MSG(recursion_count > MAX_RECURSION, "[...]", "Maximum array recursion reached!");
			recursion_count++;

			return stringify_vector(operator Array(), recursion_count);
		}
		case VariantType::OBJECT: {
			if (_get_obj().obj) {
				if (!_get_obj().id.is_ref_counted() && ObjectDB::get_instance(_get_obj().id) == nullptr) {
					return "<Freed Object>";
				}

				return _get_obj().obj->to_string();
			} else {
				return "<Object#null>";
			}
		}
		case VariantType::CALLABLE: {
			const Callable &c = *reinterpret_cast<const Callable *>(_data._mem);
			return String(c);
		}
		case VariantType::SIGNAL: {
			const Signal &s = *reinterpret_cast<const Signal *>(_data._mem);
			return String(s);
		}
		case VariantType::RID: {
			const ::RID &s = *reinterpret_cast<const ::RID *>(_data._mem);
			return "RID(" + itos(s.get_id()) + ")";
		}
		default: {
			return "<" + get_type_name(type) + ">";
		}
	}
}

String Variant::to_json_string() const {
	return JSON::stringify(*this);
}

Variant::operator Vector2() const {
	if (type == VariantType::VECTOR2) {
		return *reinterpret_cast<const Vector2 *>(_data._mem);
	} else if (type == VariantType::VECTOR2I) {
		return *reinterpret_cast<const Vector2i *>(_data._mem);
	} else if (type == VariantType::VECTOR3) {
		return Vector2(reinterpret_cast<const Vector3 *>(_data._mem)->x, reinterpret_cast<const Vector3 *>(_data._mem)->y);
	} else if (type == VariantType::VECTOR3I) {
		return Vector2(reinterpret_cast<const Vector3i *>(_data._mem)->x, reinterpret_cast<const Vector3i *>(_data._mem)->y);
	} else if (type == VariantType::VECTOR4) {
		return Vector2(reinterpret_cast<const Vector4 *>(_data._mem)->x, reinterpret_cast<const Vector4 *>(_data._mem)->y);
	} else if (type == VariantType::VECTOR4I) {
		return Vector2(reinterpret_cast<const Vector4i *>(_data._mem)->x, reinterpret_cast<const Vector4i *>(_data._mem)->y);
	} else {
		return Vector2();
	}
}

Variant::operator Vector2i() const {
	if (type == VariantType::VECTOR2I) {
		return *reinterpret_cast<const Vector2i *>(_data._mem);
	} else if (type == VariantType::VECTOR2) {
		return *reinterpret_cast<const Vector2 *>(_data._mem);
	} else if (type == VariantType::VECTOR3) {
		return Vector2(reinterpret_cast<const Vector3 *>(_data._mem)->x, reinterpret_cast<const Vector3 *>(_data._mem)->y);
	} else if (type == VariantType::VECTOR3I) {
		return Vector2(reinterpret_cast<const Vector3i *>(_data._mem)->x, reinterpret_cast<const Vector3i *>(_data._mem)->y);
	} else if (type == VariantType::VECTOR4) {
		return Vector2(reinterpret_cast<const Vector4 *>(_data._mem)->x, reinterpret_cast<const Vector4 *>(_data._mem)->y);
	} else if (type == VariantType::VECTOR4I) {
		return Vector2(reinterpret_cast<const Vector4i *>(_data._mem)->x, reinterpret_cast<const Vector4i *>(_data._mem)->y);
	} else {
		return Vector2i();
	}
}

Variant::operator Rect2() const {
	if (type == VariantType::RECT2) {
		return *reinterpret_cast<const Rect2 *>(_data._mem);
	} else if (type == VariantType::RECT2I) {
		return *reinterpret_cast<const Rect2i *>(_data._mem);
	} else {
		return Rect2();
	}
}

Variant::operator Rect2i() const {
	if (type == VariantType::RECT2I) {
		return *reinterpret_cast<const Rect2i *>(_data._mem);
	} else if (type == VariantType::RECT2) {
		return *reinterpret_cast<const Rect2 *>(_data._mem);
	} else {
		return Rect2i();
	}
}

Variant::operator Vector3() const {
	if (type == VariantType::VECTOR3) {
		return *reinterpret_cast<const Vector3 *>(_data._mem);
	} else if (type == VariantType::VECTOR3I) {
		return *reinterpret_cast<const Vector3i *>(_data._mem);
	} else if (type == VariantType::VECTOR2) {
		return Vector3(reinterpret_cast<const Vector2 *>(_data._mem)->x, reinterpret_cast<const Vector2 *>(_data._mem)->y, 0.0);
	} else if (type == VariantType::VECTOR2I) {
		return Vector3(reinterpret_cast<const Vector2i *>(_data._mem)->x, reinterpret_cast<const Vector2i *>(_data._mem)->y, 0.0);
	} else if (type == VariantType::VECTOR4) {
		return Vector3(reinterpret_cast<const Vector4 *>(_data._mem)->x, reinterpret_cast<const Vector4 *>(_data._mem)->y, reinterpret_cast<const Vector4 *>(_data._mem)->z);
	} else if (type == VariantType::VECTOR4I) {
		return Vector3(reinterpret_cast<const Vector4i *>(_data._mem)->x, reinterpret_cast<const Vector4i *>(_data._mem)->y, reinterpret_cast<const Vector4i *>(_data._mem)->z);
	} else {
		return Vector3();
	}
}

Variant::operator Vector3i() const {
	if (type == VariantType::VECTOR3I) {
		return *reinterpret_cast<const Vector3i *>(_data._mem);
	} else if (type == VariantType::VECTOR3) {
		return *reinterpret_cast<const Vector3 *>(_data._mem);
	} else if (type == VariantType::VECTOR2) {
		return Vector3i(reinterpret_cast<const Vector2 *>(_data._mem)->x, reinterpret_cast<const Vector2 *>(_data._mem)->y, 0.0);
	} else if (type == VariantType::VECTOR2I) {
		return Vector3i(reinterpret_cast<const Vector2i *>(_data._mem)->x, reinterpret_cast<const Vector2i *>(_data._mem)->y, 0.0);
	} else if (type == VariantType::VECTOR4) {
		return Vector3i(reinterpret_cast<const Vector4 *>(_data._mem)->x, reinterpret_cast<const Vector4 *>(_data._mem)->y, reinterpret_cast<const Vector4 *>(_data._mem)->z);
	} else if (type == VariantType::VECTOR4I) {
		return Vector3i(reinterpret_cast<const Vector4i *>(_data._mem)->x, reinterpret_cast<const Vector4i *>(_data._mem)->y, reinterpret_cast<const Vector4i *>(_data._mem)->z);
	} else {
		return Vector3i();
	}
}

Variant::operator Vector4() const {
	if (type == VariantType::VECTOR4) {
		return *reinterpret_cast<const Vector4 *>(_data._mem);
	} else if (type == VariantType::VECTOR4I) {
		return *reinterpret_cast<const Vector4i *>(_data._mem);
	} else if (type == VariantType::VECTOR2) {
		return Vector4(reinterpret_cast<const Vector2 *>(_data._mem)->x, reinterpret_cast<const Vector2 *>(_data._mem)->y, 0.0, 0.0);
	} else if (type == VariantType::VECTOR2I) {
		return Vector4(reinterpret_cast<const Vector2i *>(_data._mem)->x, reinterpret_cast<const Vector2i *>(_data._mem)->y, 0.0, 0.0);
	} else if (type == VariantType::VECTOR3) {
		return Vector4(reinterpret_cast<const Vector3 *>(_data._mem)->x, reinterpret_cast<const Vector3 *>(_data._mem)->y, reinterpret_cast<const Vector3 *>(_data._mem)->z, 0.0);
	} else if (type == VariantType::VECTOR3I) {
		return Vector4(reinterpret_cast<const Vector3i *>(_data._mem)->x, reinterpret_cast<const Vector3i *>(_data._mem)->y, reinterpret_cast<const Vector3i *>(_data._mem)->z, 0.0);
	} else {
		return Vector4();
	}
}

Variant::operator Vector4i() const {
	if (type == VariantType::VECTOR4I) {
		return *reinterpret_cast<const Vector4i *>(_data._mem);
	} else if (type == VariantType::VECTOR4) {
		const Vector4 &v4 = *reinterpret_cast<const Vector4 *>(_data._mem);
		return Vector4i(v4.x, v4.y, v4.z, v4.w);
	} else if (type == VariantType::VECTOR2) {
		return Vector4i(reinterpret_cast<const Vector2 *>(_data._mem)->x, reinterpret_cast<const Vector2 *>(_data._mem)->y, 0.0, 0.0);
	} else if (type == VariantType::VECTOR2I) {
		return Vector4i(reinterpret_cast<const Vector2i *>(_data._mem)->x, reinterpret_cast<const Vector2i *>(_data._mem)->y, 0.0, 0.0);
	} else if (type == VariantType::VECTOR3) {
		return Vector4i(reinterpret_cast<const Vector3 *>(_data._mem)->x, reinterpret_cast<const Vector3 *>(_data._mem)->y, reinterpret_cast<const Vector3 *>(_data._mem)->z, 0.0);
	} else if (type == VariantType::VECTOR3I) {
		return Vector4i(reinterpret_cast<const Vector3i *>(_data._mem)->x, reinterpret_cast<const Vector3i *>(_data._mem)->y, reinterpret_cast<const Vector3i *>(_data._mem)->z, 0.0);
	} else {
		return Vector4i();
	}
}

Variant::operator Plane() const {
	if (type == VariantType::PLANE) {
		return *reinterpret_cast<const Plane *>(_data._mem);
	} else {
		return Plane();
	}
}

Variant::operator ::AABB() const {
	if (type == VariantType::AABB) {
		return *_data._aabb;
	} else {
		return ::AABB();
	}
}

Variant::operator Basis() const {
	if (type == VariantType::BASIS) {
		return *_data._basis;
	} else if (type == VariantType::QUATERNION) {
		return *reinterpret_cast<const Quaternion *>(_data._mem);
	} else if (type == VariantType::TRANSFORM3D) { // unexposed in VariantType::can_convert?
		return _data._transform3d->basis;
	} else {
		return Basis();
	}
}

Variant::operator Quaternion() const {
	if (type == VariantType::QUATERNION) {
		return *reinterpret_cast<const Quaternion *>(_data._mem);
	} else if (type == VariantType::BASIS) {
		return *_data._basis;
	} else if (type == VariantType::TRANSFORM3D) {
		return _data._transform3d->basis;
	} else {
		return Quaternion();
	}
}

Variant::operator Transform3D() const {
	if (type == VariantType::TRANSFORM3D) {
		return *_data._transform3d;
	} else if (type == VariantType::BASIS) {
		return Transform3D(*_data._basis, Vector3());
	} else if (type == VariantType::QUATERNION) {
		return Transform3D(Basis(*reinterpret_cast<const Quaternion *>(_data._mem)), Vector3());
	} else if (type == VariantType::TRANSFORM2D) {
		const Transform2D &t = *_data._transform2d;
		Transform3D m;
		m.basis.rows[0][0] = t.columns[0][0];
		m.basis.rows[1][0] = t.columns[0][1];
		m.basis.rows[0][1] = t.columns[1][0];
		m.basis.rows[1][1] = t.columns[1][1];
		m.origin[0] = t.columns[2][0];
		m.origin[1] = t.columns[2][1];
		return m;
	} else if (type == VariantType::PROJECTION) {
		return *_data._projection;
	} else {
		return Transform3D();
	}
}

Variant::operator Projection() const {
	if (type == VariantType::TRANSFORM3D) {
		return *_data._transform3d;
	} else if (type == VariantType::BASIS) {
		return Transform3D(*_data._basis, Vector3());
	} else if (type == VariantType::QUATERNION) {
		return Transform3D(Basis(*reinterpret_cast<const Quaternion *>(_data._mem)), Vector3());
	} else if (type == VariantType::TRANSFORM2D) {
		const Transform2D &t = *_data._transform2d;
		Transform3D m;
		m.basis.rows[0][0] = t.columns[0][0];
		m.basis.rows[1][0] = t.columns[0][1];
		m.basis.rows[0][1] = t.columns[1][0];
		m.basis.rows[1][1] = t.columns[1][1];
		m.origin[0] = t.columns[2][0];
		m.origin[1] = t.columns[2][1];
		return m;
	} else if (type == VariantType::PROJECTION) {
		return *_data._projection;
	} else {
		return Projection();
	}
}

Variant::operator Transform2D() const {
	if (type == VariantType::TRANSFORM2D) {
		return *_data._transform2d;
	} else if (type == VariantType::TRANSFORM3D) {
		const Transform3D &t = *_data._transform3d;
		Transform2D m;
		m.columns[0][0] = t.basis.rows[0][0];
		m.columns[0][1] = t.basis.rows[1][0];
		m.columns[1][0] = t.basis.rows[0][1];
		m.columns[1][1] = t.basis.rows[1][1];
		m.columns[2][0] = t.origin[0];
		m.columns[2][1] = t.origin[1];
		return m;
	} else {
		return Transform2D();
	}
}

Variant::operator Color() const {
	if (type == VariantType::COLOR) {
		return *reinterpret_cast<const Color *>(_data._mem);
	} else if (type == VariantType::STRING) {
		return Color(operator String());
	} else if (type == VariantType::INT) {
		return Color::hex(operator int());
	} else {
		return Color();
	}
}

Variant::operator NodePath() const {
	if (type == VariantType::NODE_PATH) {
		return *reinterpret_cast<const NodePath *>(_data._mem);
	} else if (type == VariantType::STRING) {
		return NodePath(operator String());
	} else {
		return NodePath();
	}
}

Variant::operator ::RID() const {
	if (type == VariantType::RID) {
		return *reinterpret_cast<const ::RID *>(_data._mem);
	} else if (type == VariantType::OBJECT && _get_obj().obj == nullptr) {
		return ::RID();
	} else if (type == VariantType::OBJECT && _get_obj().obj) {
#ifdef DEBUG_ENABLED
		if (EngineDebugger::is_active()) {
			ERR_FAIL_NULL_V_MSG(ObjectDB::get_instance(_get_obj().id), ::RID(), "Invalid pointer (object was freed).");
		}
#endif
		Callable::CallError ce;
		const Variant ret = _get_obj().obj->callp(CoreStringName(get_rid), nullptr, 0, ce);
		if (ce.error == Callable::CallError::CALL_OK && ret.get_type() == VariantType::RID) {
			return ret;
		}
		return ::RID();
	} else {
		return ::RID();
	}
}

Variant::operator Object *() const {
	if (type == VariantType::OBJECT) {
		return _get_obj().obj;
	} else {
		return nullptr;
	}
}

Object *Variant::get_validated_object_with_check(bool &r_previously_freed) const {
	if (type == VariantType::OBJECT) {
		Object *instance = ObjectDB::get_instance(_get_obj().id);
		r_previously_freed = !instance && _get_obj().id != ObjectID();
		return instance;
	} else {
		r_previously_freed = false;
		return nullptr;
	}
}

Object *Variant::get_validated_object() const {
	if (type == VariantType::OBJECT) {
		return ObjectDB::get_instance(_get_obj().id);
	} else {
		return nullptr;
	}
}

Variant::operator Dictionary() const {
	if (type == VariantType::DICTIONARY) {
		return *reinterpret_cast<const Dictionary *>(_data._mem);
	} else {
		return Dictionary();
	}
}

Variant::operator Callable() const {
	if (type == VariantType::CALLABLE) {
		return *reinterpret_cast<const Callable *>(_data._mem);
	} else {
		return Callable();
	}
}

Variant::operator Signal() const {
	if (type == VariantType::SIGNAL) {
		return *reinterpret_cast<const Signal *>(_data._mem);
	} else {
		return Signal();
	}
}

template <typename DA, typename SA>
inline DA _convert_array(const SA &p_array) {
	DA da;
	da.resize(p_array.size());

	for (int i = 0; i < p_array.size(); i++) {
		da.set(i, Variant(p_array.get(i)));
	}

	return da;
}

template <typename DA>
inline DA _convert_array_from_variant(const Variant &p_variant) {
	switch (p_variant.get_type()) {
		case VariantType::ARRAY: {
			return _convert_array<DA, Array>(p_variant.operator Array());
		}
		case VariantType::PACKED_BYTE_ARRAY: {
			return _convert_array<DA, PackedByteArray>(p_variant.operator PackedByteArray());
		}
		case VariantType::PACKED_INT32_ARRAY: {
			return _convert_array<DA, PackedInt32Array>(p_variant.operator PackedInt32Array());
		}
		case VariantType::PACKED_INT64_ARRAY: {
			return _convert_array<DA, PackedInt64Array>(p_variant.operator PackedInt64Array());
		}
		case VariantType::PACKED_FLOAT32_ARRAY: {
			return _convert_array<DA, PackedFloat32Array>(p_variant.operator PackedFloat32Array());
		}
		case VariantType::PACKED_FLOAT64_ARRAY: {
			return _convert_array<DA, PackedFloat64Array>(p_variant.operator PackedFloat64Array());
		}
		case VariantType::PACKED_STRING_ARRAY: {
			return _convert_array<DA, PackedStringArray>(p_variant.operator PackedStringArray());
		}
		case VariantType::PACKED_VECTOR2_ARRAY: {
			return _convert_array<DA, PackedVector2Array>(p_variant.operator PackedVector2Array());
		}
		case VariantType::PACKED_VECTOR3_ARRAY: {
			return _convert_array<DA, PackedVector3Array>(p_variant.operator PackedVector3Array());
		}
		case VariantType::PACKED_COLOR_ARRAY: {
			return _convert_array<DA, PackedColorArray>(p_variant.operator PackedColorArray());
		}
		case VariantType::PACKED_VECTOR4_ARRAY: {
			return _convert_array<DA, PackedVector4Array>(p_variant.operator PackedVector4Array());
		}
		default: {
			return DA();
		}
	}
}

Variant::operator Array() const {
	if (type == VariantType::ARRAY) {
		return *reinterpret_cast<const Array *>(_data._mem);
	} else {
		return _convert_array_from_variant<Array>(*this);
	}
}

Variant::operator PackedByteArray() const {
	if (type == VariantType::PACKED_BYTE_ARRAY) {
		return static_cast<PackedArrayRef<uint8_t> *>(_data.packed_array)->array;
	} else {
		return _convert_array_from_variant<PackedByteArray>(*this);
	}
}

Variant::operator PackedInt32Array() const {
	if (type == VariantType::PACKED_INT32_ARRAY) {
		return static_cast<PackedArrayRef<int32_t> *>(_data.packed_array)->array;
	} else {
		return _convert_array_from_variant<PackedInt32Array>(*this);
	}
}

Variant::operator PackedInt64Array() const {
	if (type == VariantType::PACKED_INT64_ARRAY) {
		return static_cast<PackedArrayRef<int64_t> *>(_data.packed_array)->array;
	} else {
		return _convert_array_from_variant<PackedInt64Array>(*this);
	}
}

Variant::operator PackedFloat32Array() const {
	if (type == VariantType::PACKED_FLOAT32_ARRAY) {
		return static_cast<PackedArrayRef<float> *>(_data.packed_array)->array;
	} else {
		return _convert_array_from_variant<PackedFloat32Array>(*this);
	}
}

Variant::operator PackedFloat64Array() const {
	if (type == VariantType::PACKED_FLOAT64_ARRAY) {
		return static_cast<PackedArrayRef<double> *>(_data.packed_array)->array;
	} else {
		return _convert_array_from_variant<PackedFloat64Array>(*this);
	}
}

Variant::operator PackedStringArray() const {
	if (type == VariantType::PACKED_STRING_ARRAY) {
		return static_cast<PackedArrayRef<String> *>(_data.packed_array)->array;
	} else {
		return _convert_array_from_variant<PackedStringArray>(*this);
	}
}

Variant::operator PackedVector2Array() const {
	if (type == VariantType::PACKED_VECTOR2_ARRAY) {
		return static_cast<PackedArrayRef<Vector2> *>(_data.packed_array)->array;
	} else {
		return _convert_array_from_variant<PackedVector2Array>(*this);
	}
}

Variant::operator PackedVector3Array() const {
	if (type == VariantType::PACKED_VECTOR3_ARRAY) {
		return static_cast<PackedArrayRef<Vector3> *>(_data.packed_array)->array;
	} else {
		return _convert_array_from_variant<PackedVector3Array>(*this);
	}
}

Variant::operator PackedColorArray() const {
	if (type == VariantType::PACKED_COLOR_ARRAY) {
		return static_cast<PackedArrayRef<Color> *>(_data.packed_array)->array;
	} else {
		return _convert_array_from_variant<PackedColorArray>(*this);
	}
}

Variant::operator PackedVector4Array() const {
	if (type == VariantType::PACKED_VECTOR4_ARRAY) {
		return static_cast<PackedArrayRef<Vector4> *>(_data.packed_array)->array;
	} else {
		return _convert_array_from_variant<PackedVector4Array>(*this);
	}
}

/* helpers */

Variant::operator Vector<::RID>() const {
	Array va = operator Array();
	Vector<::RID> rids;
	rids.resize(va.size());
	for (int i = 0; i < rids.size(); i++) {
		rids.write[i] = va[i];
	}
	return rids;
}

Variant::operator Vector<Plane>() const {
	Array va = operator Array();
	Vector<Plane> planes;
	int va_size = va.size();
	if (va_size == 0) {
		return planes;
	}

	planes.resize(va_size);
	Plane *w = planes.ptrw();

	for (int i = 0; i < va_size; i++) {
		w[i] = va[i];
	}

	return planes;
}

Variant::operator Vector<Face3>() const {
	PackedVector3Array va = operator PackedVector3Array();
	Vector<Face3> faces;
	int va_size = va.size();
	if (va_size == 0) {
		return faces;
	}

	faces.resize(va_size / 3);
	Face3 *w = faces.ptrw();
	const Vector3 *r = va.ptr();

	for (int i = 0; i < va_size; i++) {
		w[i / 3].vertex[i % 3] = r[i];
	}

	return faces;
}

Variant::operator Vector<Variant>() const {
	Array va = operator Array();
	Vector<Variant> variants;
	int va_size = va.size();
	if (va_size == 0) {
		return variants;
	}

	variants.resize(va_size);
	Variant *w = variants.ptrw();
	for (int i = 0; i < va_size; i++) {
		w[i] = va[i];
	}

	return variants;
}

Variant::operator Vector<StringName>() const {
	PackedStringArray from = operator PackedStringArray();
	Vector<StringName> to;
	int len = from.size();
	to.resize(len);
	for (int i = 0; i < len; i++) {
		to.write[i] = from[i];
	}
	return to;
}

Variant::operator IPAddress() const {
	if (type == VariantType::PACKED_FLOAT32_ARRAY || type == VariantType::PACKED_INT32_ARRAY || type == VariantType::PACKED_FLOAT64_ARRAY || type == VariantType::PACKED_INT64_ARRAY || type == VariantType::PACKED_BYTE_ARRAY) {
		Vector<int> addr = operator Vector<int>();
		if (addr.size() == 4) {
			return IPAddress(addr.get(0), addr.get(1), addr.get(2), addr.get(3));
		}
	}

	return IPAddress(operator String());
}

Variant::Variant(bool p_bool) :
		type(VariantType::BOOL) {
	_data._bool = p_bool;
}

Variant::Variant(int64_t p_int64) :
		type(VariantType::INT) {
	_data._int = p_int64;
}

Variant::Variant(int32_t p_int32) :
		type(VariantType::INT) {
	_data._int = p_int32;
}

Variant::Variant(int16_t p_int16) :
		type(VariantType::INT) {
	_data._int = p_int16;
}

Variant::Variant(int8_t p_int8) :
		type(VariantType::INT) {
	_data._int = p_int8;
}

Variant::Variant(Math::int_alt_t p_int_alt) :
		type(VariantType::INT) {
	_data._int = p_int_alt;
}

Variant::Variant(uint64_t p_uint64) :
		type(VariantType::INT) {
	_data._int = int64_t(p_uint64);
}

Variant::Variant(uint32_t p_uint32) :
		type(VariantType::INT) {
	_data._int = int64_t(p_uint32);
}

Variant::Variant(uint16_t p_uint16) :
		type(VariantType::INT) {
	_data._int = int64_t(p_uint16);
}

Variant::Variant(uint8_t p_uint8) :
		type(VariantType::INT) {
	_data._int = int64_t(p_uint8);
}

Variant::Variant(Math::uint_alt_t p_uint_alt) :
		type(VariantType::INT) {
	_data._int = p_uint_alt;
}

Variant::Variant(float p_float) :
		type(VariantType::FLOAT) {
	_data._float = p_float;
}

Variant::Variant(double p_double) :
		type(VariantType::FLOAT) {
	_data._float = p_double;
}

Variant::Variant(const ObjectID &p_id) :
		type(VariantType::INT) {
	_data._int = int64_t(p_id);
}

Variant::Variant(const StringName &p_string) :
		type(VariantType::STRING_NAME) {
	memnew_placement(_data._mem, StringName(p_string));
	static_assert(sizeof(StringName) <= sizeof(_data._mem));
}

Variant::Variant(const String &p_string) :
		type(VariantType::STRING) {
	memnew_placement(_data._mem, String(p_string));
	static_assert(sizeof(String) <= sizeof(_data._mem));
}

Variant::Variant(const char *const p_cstring) :
		type(VariantType::STRING) {
	memnew_placement(_data._mem, String((const char *)p_cstring));
	static_assert(sizeof(String) <= sizeof(_data._mem));
}

Variant::Variant(const char32_t *p_wstring) :
		type(VariantType::STRING) {
	memnew_placement(_data._mem, String(p_wstring));
	static_assert(sizeof(String) <= sizeof(_data._mem));
}

Variant::Variant(const Vector3 &p_vector3) :
		type(VariantType::VECTOR3) {
	memnew_placement(_data._mem, Vector3(p_vector3));
	static_assert(sizeof(Vector3) <= sizeof(_data._mem));
}

Variant::Variant(const Vector3i &p_vector3i) :
		type(VariantType::VECTOR3I) {
	memnew_placement(_data._mem, Vector3i(p_vector3i));
	static_assert(sizeof(Vector3i) <= sizeof(_data._mem));
}

Variant::Variant(const Vector4 &p_vector4) :
		type(VariantType::VECTOR4) {
	memnew_placement(_data._mem, Vector4(p_vector4));
	static_assert(sizeof(Vector4) <= sizeof(_data._mem));
}

Variant::Variant(const Vector4i &p_vector4i) :
		type(VariantType::VECTOR4I) {
	memnew_placement(_data._mem, Vector4i(p_vector4i));
	static_assert(sizeof(Vector4i) <= sizeof(_data._mem));
}

Variant::Variant(const Vector2 &p_vector2) :
		type(VariantType::VECTOR2) {
	memnew_placement(_data._mem, Vector2(p_vector2));
	static_assert(sizeof(Vector2) <= sizeof(_data._mem));
}

Variant::Variant(const Vector2i &p_vector2i) :
		type(VariantType::VECTOR2I) {
	memnew_placement(_data._mem, Vector2i(p_vector2i));
	static_assert(sizeof(Vector2i) <= sizeof(_data._mem));
}

Variant::Variant(const Rect2 &p_rect2) :
		type(VariantType::RECT2) {
	memnew_placement(_data._mem, Rect2(p_rect2));
	static_assert(sizeof(Rect2) <= sizeof(_data._mem));
}

Variant::Variant(const Rect2i &p_rect2i) :
		type(VariantType::RECT2I) {
	memnew_placement(_data._mem, Rect2i(p_rect2i));
	static_assert(sizeof(Rect2i) <= sizeof(_data._mem));
}

Variant::Variant(const Plane &p_plane) :
		type(VariantType::PLANE) {
	memnew_placement(_data._mem, Plane(p_plane));
	static_assert(sizeof(Plane) <= sizeof(_data._mem));
}

Variant::Variant(const ::AABB &p_aabb) :
		type(VariantType::AABB) {
	_data._aabb = VariantPools::alloc<::AABB>();
	memnew_placement(_data._aabb, ::AABB(p_aabb));
}

Variant::Variant(const Basis &p_matrix) :
		type(VariantType::BASIS) {
	_data._basis = VariantPools::alloc<Basis>();
	memnew_placement(_data._basis, Basis(p_matrix));
}

Variant::Variant(const Quaternion &p_quaternion) :
		type(VariantType::QUATERNION) {
	memnew_placement(_data._mem, Quaternion(p_quaternion));
	static_assert(sizeof(Quaternion) <= sizeof(_data._mem));
}

Variant::Variant(const Transform3D &p_transform) :
		type(VariantType::TRANSFORM3D) {
	_data._transform3d = VariantPools::alloc<Transform3D>();
	memnew_placement(_data._transform3d, Transform3D(p_transform));
}

Variant::Variant(const Projection &pp_projection) :
		type(VariantType::PROJECTION) {
	_data._projection = VariantPools::alloc<Projection>();
	memnew_placement(_data._projection, Projection(pp_projection));
}

Variant::Variant(const Transform2D &p_transform) :
		type(VariantType::TRANSFORM2D) {
	_data._transform2d = VariantPools::alloc<Transform2D>();
	memnew_placement(_data._transform2d, Transform2D(p_transform));
}

Variant::Variant(const Color &p_color) :
		type(VariantType::COLOR) {
	memnew_placement(_data._mem, Color(p_color));
	static_assert(sizeof(Color) <= sizeof(_data._mem));
}

Variant::Variant(const NodePath &p_node_path) :
		type(VariantType::NODE_PATH) {
	memnew_placement(_data._mem, NodePath(p_node_path));
	static_assert(sizeof(NodePath) <= sizeof(_data._mem));
}

Variant::Variant(const ::RID &p_rid) :
		type(VariantType::RID) {
	memnew_placement(_data._mem, ::RID(p_rid));
	static_assert(sizeof(::RID) <= sizeof(_data._mem));
}

Variant::Variant(const Object *p_object) :
		type(VariantType::OBJECT) {
	_get_obj() = ObjData();
	_get_obj().ref_pointer(const_cast<Object *>(p_object));
}

Variant::Variant(const Callable &p_callable) :
		type(VariantType::CALLABLE) {
	memnew_placement(_data._mem, Callable(p_callable));
	static_assert(sizeof(Callable) <= sizeof(_data._mem));
}

Variant::Variant(const Signal &p_callable) :
		type(VariantType::SIGNAL) {
	memnew_placement(_data._mem, Signal(p_callable));
	static_assert(sizeof(Signal) <= sizeof(_data._mem));
}

Variant::Variant(const Dictionary &p_dictionary) :
		type(VariantType::DICTIONARY) {
	memnew_placement(_data._mem, Dictionary(p_dictionary));
	static_assert(sizeof(Dictionary) <= sizeof(_data._mem));
}

Variant::Variant(std::initializer_list<Variant> p_init) :
		type(VariantType::ARRAY) {
	memnew_placement(_data._mem, Array(p_init));
}

Variant::Variant(const Array &p_array) :
		type(VariantType::ARRAY) {
	memnew_placement(_data._mem, Array(p_array));
	static_assert(sizeof(Array) <= sizeof(_data._mem));
}

Variant::Variant(const PackedByteArray &p_byte_array) :
		type(VariantType::PACKED_BYTE_ARRAY) {
	_data.packed_array = PackedArrayRef<uint8_t>::create(p_byte_array);
}

Variant::Variant(const PackedInt32Array &p_int32_array) :
		type(VariantType::PACKED_INT32_ARRAY) {
	_data.packed_array = PackedArrayRef<int32_t>::create(p_int32_array);
}

Variant::Variant(const PackedInt64Array &p_int64_array) :
		type(VariantType::PACKED_INT64_ARRAY) {
	_data.packed_array = PackedArrayRef<int64_t>::create(p_int64_array);
}

Variant::Variant(const PackedFloat32Array &p_float32_array) :
		type(VariantType::PACKED_FLOAT32_ARRAY) {
	_data.packed_array = PackedArrayRef<float>::create(p_float32_array);
}

Variant::Variant(const PackedFloat64Array &p_float64_array) :
		type(VariantType::PACKED_FLOAT64_ARRAY) {
	_data.packed_array = PackedArrayRef<double>::create(p_float64_array);
}

Variant::Variant(const PackedStringArray &p_string_array) :
		type(VariantType::PACKED_STRING_ARRAY) {
	_data.packed_array = PackedArrayRef<String>::create(p_string_array);
}

Variant::Variant(const PackedVector2Array &p_vector2_array) :
		type(VariantType::PACKED_VECTOR2_ARRAY) {
	_data.packed_array = PackedArrayRef<Vector2>::create(p_vector2_array);
}

Variant::Variant(const PackedVector3Array &p_vector3_array) :
		type(VariantType::PACKED_VECTOR3_ARRAY) {
	_data.packed_array = PackedArrayRef<Vector3>::create(p_vector3_array);
}

Variant::Variant(const PackedColorArray &p_color_array) :
		type(VariantType::PACKED_COLOR_ARRAY) {
	_data.packed_array = PackedArrayRef<Color>::create(p_color_array);
}

Variant::Variant(const PackedVector4Array &p_vector4_array) :
		type(VariantType::PACKED_VECTOR4_ARRAY) {
	_data.packed_array = PackedArrayRef<Vector4>::create(p_vector4_array);
}

/* helpers */
Variant::Variant(const Vector<::RID> &p_array) :
		type(VariantType::ARRAY) {
	Array *rid_array = memnew_placement(_data._mem, Array);

	rid_array->resize(p_array.size());

	for (int i = 0; i < p_array.size(); i++) {
		rid_array->set(i, Variant(p_array[i]));
	}
}

Variant::Variant(const Vector<Plane> &p_array) :
		type(VariantType::ARRAY) {
	Array *plane_array = memnew_placement(_data._mem, Array);

	plane_array->resize(p_array.size());

	for (int i = 0; i < p_array.size(); i++) {
		plane_array->operator[](i) = Variant(p_array[i]);
	}
}

Variant::Variant(const Vector<Face3> &p_face_array) {
	PackedVector3Array vertices;
	int face_count = p_face_array.size();
	vertices.resize(face_count * 3);

	if (face_count) {
		const Face3 *r = p_face_array.ptr();
		Vector3 *w = vertices.ptrw();

		for (int i = 0; i < face_count; i++) {
			for (int j = 0; j < 3; j++) {
				w[i * 3 + j] = r[i].vertex[j];
			}
		}
	}

	*this = vertices;
}

Variant::Variant(const Vector<Variant> &p_array) {
	Array arr;
	arr.resize(p_array.size());
	for (int i = 0; i < p_array.size(); i++) {
		arr[i] = p_array[i];
	}
	*this = arr;
}

Variant::Variant(const Vector<StringName> &p_array) {
	PackedStringArray v;
	int len = p_array.size();
	v.resize(len);
	for (int i = 0; i < len; i++) {
		v.set(i, p_array[i]);
	}
	*this = v;
}

void Variant::operator=(const Variant &p_variant) {
	if (unlikely(this == &p_variant)) {
		return;
	}

	if (unlikely(type != p_variant.type)) {
		reference(p_variant);
		return;
	}

	switch (p_variant.type) {
		case VariantType::NIL: {
			// none
		} break;

		// atomic types
		case VariantType::BOOL: {
			_data._bool = p_variant._data._bool;
		} break;
		case VariantType::INT: {
			_data._int = p_variant._data._int;
		} break;
		case VariantType::FLOAT: {
			_data._float = p_variant._data._float;
		} break;
		case VariantType::STRING: {
			*reinterpret_cast<String *>(_data._mem) = *reinterpret_cast<const String *>(p_variant._data._mem);
		} break;

		// math types
		case VariantType::VECTOR2: {
			*reinterpret_cast<Vector2 *>(_data._mem) = *reinterpret_cast<const Vector2 *>(p_variant._data._mem);
		} break;
		case VariantType::VECTOR2I: {
			*reinterpret_cast<Vector2i *>(_data._mem) = *reinterpret_cast<const Vector2i *>(p_variant._data._mem);
		} break;
		case VariantType::RECT2: {
			*reinterpret_cast<Rect2 *>(_data._mem) = *reinterpret_cast<const Rect2 *>(p_variant._data._mem);
		} break;
		case VariantType::RECT2I: {
			*reinterpret_cast<Rect2i *>(_data._mem) = *reinterpret_cast<const Rect2i *>(p_variant._data._mem);
		} break;
		case VariantType::TRANSFORM2D: {
			*_data._transform2d = *(p_variant._data._transform2d);
		} break;
		case VariantType::VECTOR3: {
			*reinterpret_cast<Vector3 *>(_data._mem) = *reinterpret_cast<const Vector3 *>(p_variant._data._mem);
		} break;
		case VariantType::VECTOR3I: {
			*reinterpret_cast<Vector3i *>(_data._mem) = *reinterpret_cast<const Vector3i *>(p_variant._data._mem);
		} break;
		case VariantType::VECTOR4: {
			*reinterpret_cast<Vector4 *>(_data._mem) = *reinterpret_cast<const Vector4 *>(p_variant._data._mem);
		} break;
		case VariantType::VECTOR4I: {
			*reinterpret_cast<Vector4i *>(_data._mem) = *reinterpret_cast<const Vector4i *>(p_variant._data._mem);
		} break;
		case VariantType::PLANE: {
			*reinterpret_cast<Plane *>(_data._mem) = *reinterpret_cast<const Plane *>(p_variant._data._mem);
		} break;

		case VariantType::AABB: {
			*_data._aabb = *(p_variant._data._aabb);
		} break;
		case VariantType::QUATERNION: {
			*reinterpret_cast<Quaternion *>(_data._mem) = *reinterpret_cast<const Quaternion *>(p_variant._data._mem);
		} break;
		case VariantType::BASIS: {
			*_data._basis = *(p_variant._data._basis);
		} break;
		case VariantType::TRANSFORM3D: {
			*_data._transform3d = *(p_variant._data._transform3d);
		} break;
		case VariantType::PROJECTION: {
			*_data._projection = *(p_variant._data._projection);
		} break;

		// misc types
		case VariantType::COLOR: {
			*reinterpret_cast<Color *>(_data._mem) = *reinterpret_cast<const Color *>(p_variant._data._mem);
		} break;
		case VariantType::RID: {
			*reinterpret_cast<::RID *>(_data._mem) = *reinterpret_cast<const ::RID *>(p_variant._data._mem);
		} break;
		case VariantType::OBJECT: {
			_get_obj().ref(p_variant._get_obj());
		} break;
		case VariantType::CALLABLE: {
			*reinterpret_cast<Callable *>(_data._mem) = *reinterpret_cast<const Callable *>(p_variant._data._mem);
		} break;
		case VariantType::SIGNAL: {
			*reinterpret_cast<Signal *>(_data._mem) = *reinterpret_cast<const Signal *>(p_variant._data._mem);
		} break;

		case VariantType::STRING_NAME: {
			*reinterpret_cast<StringName *>(_data._mem) = *reinterpret_cast<const StringName *>(p_variant._data._mem);
		} break;
		case VariantType::NODE_PATH: {
			*reinterpret_cast<NodePath *>(_data._mem) = *reinterpret_cast<const NodePath *>(p_variant._data._mem);
		} break;
		case VariantType::DICTIONARY: {
			*reinterpret_cast<Dictionary *>(_data._mem) = *reinterpret_cast<const Dictionary *>(p_variant._data._mem);
		} break;
		case VariantType::ARRAY: {
			*reinterpret_cast<Array *>(_data._mem) = *reinterpret_cast<const Array *>(p_variant._data._mem);
		} break;

		// arrays
		case VariantType::PACKED_BYTE_ARRAY: {
			_data.packed_array = PackedArrayRef<uint8_t>::reference_from(_data.packed_array, p_variant._data.packed_array);
		} break;
		case VariantType::PACKED_INT32_ARRAY: {
			_data.packed_array = PackedArrayRef<int32_t>::reference_from(_data.packed_array, p_variant._data.packed_array);
		} break;
		case VariantType::PACKED_INT64_ARRAY: {
			_data.packed_array = PackedArrayRef<int64_t>::reference_from(_data.packed_array, p_variant._data.packed_array);
		} break;
		case VariantType::PACKED_FLOAT32_ARRAY: {
			_data.packed_array = PackedArrayRef<float>::reference_from(_data.packed_array, p_variant._data.packed_array);
		} break;
		case VariantType::PACKED_FLOAT64_ARRAY: {
			_data.packed_array = PackedArrayRef<double>::reference_from(_data.packed_array, p_variant._data.packed_array);
		} break;
		case VariantType::PACKED_STRING_ARRAY: {
			_data.packed_array = PackedArrayRef<String>::reference_from(_data.packed_array, p_variant._data.packed_array);
		} break;
		case VariantType::PACKED_VECTOR2_ARRAY: {
			_data.packed_array = PackedArrayRef<Vector2>::reference_from(_data.packed_array, p_variant._data.packed_array);
		} break;
		case VariantType::PACKED_VECTOR3_ARRAY: {
			_data.packed_array = PackedArrayRef<Vector3>::reference_from(_data.packed_array, p_variant._data.packed_array);
		} break;
		case VariantType::PACKED_COLOR_ARRAY: {
			_data.packed_array = PackedArrayRef<Color>::reference_from(_data.packed_array, p_variant._data.packed_array);
		} break;
		case VariantType::PACKED_VECTOR4_ARRAY: {
			_data.packed_array = PackedArrayRef<Vector4>::reference_from(_data.packed_array, p_variant._data.packed_array);
		} break;
		default: {
		}
	}
}

Variant::Variant(const IPAddress &p_address) :
		type(VariantType::STRING) {
	memnew_placement(_data._mem, String(p_address));
}

Variant::Variant(const Variant &p_variant) {
	reference(p_variant);
}

uint32_t Variant::hash() const {
	return recursive_hash(0);
}

uint32_t Variant::recursive_hash(int recursion_count) const {
	switch (type) {
		case VariantType::NIL: {
			return 0;
		} break;
		case VariantType::BOOL: {
			return _data._bool ? 1 : 0;
		} break;
		case VariantType::INT: {
			return hash_one_uint64((uint64_t)_data._int);
		} break;
		case VariantType::FLOAT: {
			return hash_murmur3_one_double(_data._float);
		} break;
		case VariantType::STRING: {
			return reinterpret_cast<const String *>(_data._mem)->hash();
		} break;

		// math types
		case VariantType::VECTOR2: {
			return HashMapHasherDefault::hash(*reinterpret_cast<const Vector2 *>(_data._mem));
		} break;
		case VariantType::VECTOR2I: {
			return HashMapHasherDefault::hash(*reinterpret_cast<const Vector2i *>(_data._mem));
		} break;
		case VariantType::RECT2: {
			return HashMapHasherDefault::hash(*reinterpret_cast<const Rect2 *>(_data._mem));
		} break;
		case VariantType::RECT2I: {
			return HashMapHasherDefault::hash(*reinterpret_cast<const Rect2i *>(_data._mem));
		} break;
		case VariantType::TRANSFORM2D: {
			uint32_t h = HASH_MURMUR3_SEED;
			const Transform2D &t = *_data._transform2d;
			h = hash_murmur3_one_real(t[0].x, h);
			h = hash_murmur3_one_real(t[0].y, h);
			h = hash_murmur3_one_real(t[1].x, h);
			h = hash_murmur3_one_real(t[1].y, h);
			h = hash_murmur3_one_real(t[2].x, h);
			h = hash_murmur3_one_real(t[2].y, h);

			return hash_fmix32(h);
		} break;
		case VariantType::VECTOR3: {
			return HashMapHasherDefault::hash(*reinterpret_cast<const Vector3 *>(_data._mem));
		} break;
		case VariantType::VECTOR3I: {
			return HashMapHasherDefault::hash(*reinterpret_cast<const Vector3i *>(_data._mem));
		} break;
		case VariantType::VECTOR4: {
			return HashMapHasherDefault::hash(*reinterpret_cast<const Vector4 *>(_data._mem));
		} break;
		case VariantType::VECTOR4I: {
			return HashMapHasherDefault::hash(*reinterpret_cast<const Vector4i *>(_data._mem));
		} break;
		case VariantType::PLANE: {
			uint32_t h = HASH_MURMUR3_SEED;
			const Plane &p = *reinterpret_cast<const Plane *>(_data._mem);
			h = hash_murmur3_one_real(p.normal.x, h);
			h = hash_murmur3_one_real(p.normal.y, h);
			h = hash_murmur3_one_real(p.normal.z, h);
			h = hash_murmur3_one_real(p.d, h);
			return hash_fmix32(h);
		} break;
		case VariantType::AABB: {
			return HashMapHasherDefault::hash(*_data._aabb);
		} break;
		case VariantType::QUATERNION: {
			uint32_t h = HASH_MURMUR3_SEED;
			const Quaternion &q = *reinterpret_cast<const Quaternion *>(_data._mem);
			h = hash_murmur3_one_real(q.x, h);
			h = hash_murmur3_one_real(q.y, h);
			h = hash_murmur3_one_real(q.z, h);
			h = hash_murmur3_one_real(q.w, h);
			return hash_fmix32(h);
		} break;
		case VariantType::BASIS: {
			uint32_t h = HASH_MURMUR3_SEED;
			const Basis &b = *_data._basis;
			h = hash_murmur3_one_real(b[0].x, h);
			h = hash_murmur3_one_real(b[0].y, h);
			h = hash_murmur3_one_real(b[0].z, h);
			h = hash_murmur3_one_real(b[1].x, h);
			h = hash_murmur3_one_real(b[1].y, h);
			h = hash_murmur3_one_real(b[1].z, h);
			h = hash_murmur3_one_real(b[2].x, h);
			h = hash_murmur3_one_real(b[2].y, h);
			h = hash_murmur3_one_real(b[2].z, h);
			return hash_fmix32(h);
		} break;
		case VariantType::TRANSFORM3D: {
			uint32_t h = HASH_MURMUR3_SEED;
			const Transform3D &t = *_data._transform3d;
			h = hash_murmur3_one_real(t.basis[0].x, h);
			h = hash_murmur3_one_real(t.basis[0].y, h);
			h = hash_murmur3_one_real(t.basis[0].z, h);
			h = hash_murmur3_one_real(t.basis[1].x, h);
			h = hash_murmur3_one_real(t.basis[1].y, h);
			h = hash_murmur3_one_real(t.basis[1].z, h);
			h = hash_murmur3_one_real(t.basis[2].x, h);
			h = hash_murmur3_one_real(t.basis[2].y, h);
			h = hash_murmur3_one_real(t.basis[2].z, h);
			h = hash_murmur3_one_real(t.origin.x, h);
			h = hash_murmur3_one_real(t.origin.y, h);
			h = hash_murmur3_one_real(t.origin.z, h);
			return hash_fmix32(h);
		} break;
		case VariantType::PROJECTION: {
			uint32_t h = HASH_MURMUR3_SEED;
			const Projection &t = *_data._projection;
			h = hash_murmur3_one_real(t.columns[0].x, h);
			h = hash_murmur3_one_real(t.columns[0].y, h);
			h = hash_murmur3_one_real(t.columns[0].z, h);
			h = hash_murmur3_one_real(t.columns[0].w, h);
			h = hash_murmur3_one_real(t.columns[1].x, h);
			h = hash_murmur3_one_real(t.columns[1].y, h);
			h = hash_murmur3_one_real(t.columns[1].z, h);
			h = hash_murmur3_one_real(t.columns[1].w, h);
			h = hash_murmur3_one_real(t.columns[2].x, h);
			h = hash_murmur3_one_real(t.columns[2].y, h);
			h = hash_murmur3_one_real(t.columns[2].z, h);
			h = hash_murmur3_one_real(t.columns[2].w, h);
			h = hash_murmur3_one_real(t.columns[3].x, h);
			h = hash_murmur3_one_real(t.columns[3].y, h);
			h = hash_murmur3_one_real(t.columns[3].z, h);
			h = hash_murmur3_one_real(t.columns[3].w, h);
			return hash_fmix32(h);
		} break;
		// misc types
		case VariantType::COLOR: {
			uint32_t h = HASH_MURMUR3_SEED;
			const Color &c = *reinterpret_cast<const Color *>(_data._mem);
			h = hash_murmur3_one_float(c.r, h);
			h = hash_murmur3_one_float(c.g, h);
			h = hash_murmur3_one_float(c.b, h);
			h = hash_murmur3_one_float(c.a, h);
			return hash_fmix32(h);
		} break;
		case VariantType::RID: {
			return hash_one_uint64(reinterpret_cast<const ::RID *>(_data._mem)->get_id());
		} break;
		case VariantType::OBJECT: {
			return hash_one_uint64(reinterpret_cast<uint64_t>(_get_obj().obj));
		} break;
		case VariantType::STRING_NAME: {
			return reinterpret_cast<const StringName *>(_data._mem)->hash();
		} break;
		case VariantType::NODE_PATH: {
			return reinterpret_cast<const NodePath *>(_data._mem)->hash();
		} break;
		case VariantType::DICTIONARY: {
			return reinterpret_cast<const Dictionary *>(_data._mem)->recursive_hash(recursion_count);

		} break;
		case VariantType::CALLABLE: {
			return reinterpret_cast<const Callable *>(_data._mem)->hash();

		} break;
		case VariantType::SIGNAL: {
			const Signal &s = *reinterpret_cast<const Signal *>(_data._mem);
			uint32_t hash = s.get_name().hash();
			return hash_murmur3_one_64(s.get_object_id(), hash);
		} break;
		case VariantType::ARRAY: {
			const Array &arr = *reinterpret_cast<const Array *>(_data._mem);
			return arr.recursive_hash(recursion_count);

		} break;
		case VariantType::PACKED_BYTE_ARRAY: {
			const PackedByteArray &arr = PackedArrayRef<uint8_t>::get_array(_data.packed_array);
			int len = arr.size();
			if (likely(len)) {
				const uint8_t *r = arr.ptr();
				return hash_murmur3_buffer((uint8_t *)&r[0], len);
			} else {
				return hash_murmur3_one_64(0);
			}

		} break;
		case VariantType::PACKED_INT32_ARRAY: {
			const PackedInt32Array &arr = PackedArrayRef<int32_t>::get_array(_data.packed_array);
			int len = arr.size();
			if (likely(len)) {
				const int32_t *r = arr.ptr();
				return hash_murmur3_buffer((uint8_t *)&r[0], len * sizeof(int32_t));
			} else {
				return hash_murmur3_one_64(0);
			}

		} break;
		case VariantType::PACKED_INT64_ARRAY: {
			const PackedInt64Array &arr = PackedArrayRef<int64_t>::get_array(_data.packed_array);
			int len = arr.size();
			if (likely(len)) {
				const int64_t *r = arr.ptr();
				return hash_murmur3_buffer((uint8_t *)&r[0], len * sizeof(int64_t));
			} else {
				return hash_murmur3_one_64(0);
			}

		} break;
		case VariantType::PACKED_FLOAT32_ARRAY: {
			const PackedFloat32Array &arr = PackedArrayRef<float>::get_array(_data.packed_array);
			int len = arr.size();

			if (likely(len)) {
				const float *r = arr.ptr();
				uint32_t h = HASH_MURMUR3_SEED;
				for (int32_t i = 0; i < len; i++) {
					h = hash_murmur3_one_float(r[i], h);
				}
				return hash_fmix32(h);
			} else {
				return hash_murmur3_one_float(0.0);
			}

		} break;
		case VariantType::PACKED_FLOAT64_ARRAY: {
			const PackedFloat64Array &arr = PackedArrayRef<double>::get_array(_data.packed_array);
			int len = arr.size();

			if (likely(len)) {
				const double *r = arr.ptr();
				uint32_t h = HASH_MURMUR3_SEED;
				for (int32_t i = 0; i < len; i++) {
					h = hash_murmur3_one_double(r[i], h);
				}
				return hash_fmix32(h);
			} else {
				return hash_murmur3_one_double(0.0);
			}

		} break;
		case VariantType::PACKED_STRING_ARRAY: {
			uint32_t hash = HASH_MURMUR3_SEED;
			const PackedStringArray &arr = PackedArrayRef<String>::get_array(_data.packed_array);
			int len = arr.size();

			if (likely(len)) {
				const String *r = arr.ptr();

				for (int i = 0; i < len; i++) {
					hash = hash_murmur3_one_32(r[i].hash(), hash);
				}
				hash = hash_fmix32(hash);
			}

			return hash;
		} break;
		case VariantType::PACKED_VECTOR2_ARRAY: {
			uint32_t hash = HASH_MURMUR3_SEED;
			const PackedVector2Array &arr = PackedArrayRef<Vector2>::get_array(_data.packed_array);
			int len = arr.size();

			if (likely(len)) {
				const Vector2 *r = arr.ptr();

				for (int i = 0; i < len; i++) {
					hash = hash_murmur3_one_real(r[i].x, hash);
					hash = hash_murmur3_one_real(r[i].y, hash);
				}
				hash = hash_fmix32(hash);
			}

			return hash;
		} break;
		case VariantType::PACKED_VECTOR3_ARRAY: {
			uint32_t hash = HASH_MURMUR3_SEED;
			const PackedVector3Array &arr = PackedArrayRef<Vector3>::get_array(_data.packed_array);
			int len = arr.size();

			if (likely(len)) {
				const Vector3 *r = arr.ptr();

				for (int i = 0; i < len; i++) {
					hash = hash_murmur3_one_real(r[i].x, hash);
					hash = hash_murmur3_one_real(r[i].y, hash);
					hash = hash_murmur3_one_real(r[i].z, hash);
				}
				hash = hash_fmix32(hash);
			}

			return hash;
		} break;
		case VariantType::PACKED_COLOR_ARRAY: {
			uint32_t hash = HASH_MURMUR3_SEED;
			const PackedColorArray &arr = PackedArrayRef<Color>::get_array(_data.packed_array);
			int len = arr.size();

			if (likely(len)) {
				const Color *r = arr.ptr();

				for (int i = 0; i < len; i++) {
					hash = hash_murmur3_one_float(r[i].r, hash);
					hash = hash_murmur3_one_float(r[i].g, hash);
					hash = hash_murmur3_one_float(r[i].b, hash);
					hash = hash_murmur3_one_float(r[i].a, hash);
				}
				hash = hash_fmix32(hash);
			}

			return hash;
		} break;
		case VariantType::PACKED_VECTOR4_ARRAY: {
			uint32_t hash = HASH_MURMUR3_SEED;
			const PackedVector4Array &arr = PackedArrayRef<Vector4>::get_array(_data.packed_array);
			int len = arr.size();

			if (likely(len)) {
				const Vector4 *r = arr.ptr();

				for (int i = 0; i < len; i++) {
					hash = hash_murmur3_one_real(r[i].x, hash);
					hash = hash_murmur3_one_real(r[i].y, hash);
					hash = hash_murmur3_one_real(r[i].z, hash);
					hash = hash_murmur3_one_real(r[i].w, hash);
				}
				hash = hash_fmix32(hash);
			}

			return hash;
		} break;
		default: {
		}
	}

	return 0;
}

#define hash_compare_scalar_base(p_lhs, p_rhs, semantic_comparison) \
	(((p_lhs) == (p_rhs)) || (semantic_comparison && Math::is_nan(p_lhs) && Math::is_nan(p_rhs)))

#define hash_compare_scalar(p_lhs, p_rhs) \
	(hash_compare_scalar_base(p_lhs, p_rhs, true))

#define hash_compare_vector2(p_lhs, p_rhs) \
	(p_lhs).is_same(p_rhs)

#define hash_compare_vector3(p_lhs, p_rhs) \
	(p_lhs).is_same(p_rhs)

#define hash_compare_vector4(p_lhs, p_rhs) \
	(p_lhs).is_same(p_rhs)

#define hash_compare_quaternion(p_lhs, p_rhs) \
	(p_lhs).is_same(p_rhs)

#define hash_compare_color(p_lhs, p_rhs) \
	(p_lhs).is_same(p_rhs)

#define hash_compare_packed_array(p_lhs, p_rhs, p_type, p_compare_func) \
	const Vector<p_type> &l = PackedArrayRef<p_type>::get_array(p_lhs); \
	const Vector<p_type> &r = PackedArrayRef<p_type>::get_array(p_rhs); \
\
	if (l.size() != r.size()) \
		return false; \
\
	const p_type *lr = l.ptr(); \
	const p_type *rr = r.ptr(); \
\
	for (int i = 0; i < l.size(); ++i) { \
		if (!p_compare_func((lr[i]), (rr[i]))) \
			return false; \
	} \
\
	return true

bool Variant::hash_compare(const Variant &p_variant, int recursion_count, bool semantic_comparison) const {
	if (type != p_variant.type) {
		return false;
	}

	switch (type) {
		case VariantType::INT: {
			return _data._int == p_variant._data._int;
		} break;

		case VariantType::FLOAT: {
			return hash_compare_scalar_base(_data._float, p_variant._data._float, semantic_comparison);
		} break;

		case VariantType::STRING: {
			return *reinterpret_cast<const String *>(_data._mem) == *reinterpret_cast<const String *>(p_variant._data._mem);
		} break;

		case VariantType::STRING_NAME: {
			return *reinterpret_cast<const StringName *>(_data._mem) == *reinterpret_cast<const StringName *>(p_variant._data._mem);
		} break;

		case VariantType::VECTOR2: {
			const Vector2 *l = reinterpret_cast<const Vector2 *>(_data._mem);
			const Vector2 *r = reinterpret_cast<const Vector2 *>(p_variant._data._mem);

			return hash_compare_vector2(*l, *r);
		} break;
		case VariantType::VECTOR2I: {
			const Vector2i *l = reinterpret_cast<const Vector2i *>(_data._mem);
			const Vector2i *r = reinterpret_cast<const Vector2i *>(p_variant._data._mem);
			return *l == *r;
		} break;

		case VariantType::RECT2: {
			const Rect2 *l = reinterpret_cast<const Rect2 *>(_data._mem);
			const Rect2 *r = reinterpret_cast<const Rect2 *>(p_variant._data._mem);

			return hash_compare_vector2(l->position, r->position) &&
					hash_compare_vector2(l->size, r->size);
		} break;
		case VariantType::RECT2I: {
			const Rect2i *l = reinterpret_cast<const Rect2i *>(_data._mem);
			const Rect2i *r = reinterpret_cast<const Rect2i *>(p_variant._data._mem);

			return *l == *r;
		} break;

		case VariantType::TRANSFORM2D: {
			Transform2D *l = _data._transform2d;
			Transform2D *r = p_variant._data._transform2d;

			return l->is_same(*r);
		} break;

		case VariantType::VECTOR3: {
			const Vector3 *l = reinterpret_cast<const Vector3 *>(_data._mem);
			const Vector3 *r = reinterpret_cast<const Vector3 *>(p_variant._data._mem);

			return hash_compare_vector3(*l, *r);
		} break;
		case VariantType::VECTOR3I: {
			const Vector3i *l = reinterpret_cast<const Vector3i *>(_data._mem);
			const Vector3i *r = reinterpret_cast<const Vector3i *>(p_variant._data._mem);

			return *l == *r;
		} break;
		case VariantType::VECTOR4: {
			const Vector4 *l = reinterpret_cast<const Vector4 *>(_data._mem);
			const Vector4 *r = reinterpret_cast<const Vector4 *>(p_variant._data._mem);

			return hash_compare_vector4(*l, *r);
		} break;
		case VariantType::VECTOR4I: {
			const Vector4i *l = reinterpret_cast<const Vector4i *>(_data._mem);
			const Vector4i *r = reinterpret_cast<const Vector4i *>(p_variant._data._mem);

			return *l == *r;
		} break;

		case VariantType::PLANE: {
			const Plane *l = reinterpret_cast<const Plane *>(_data._mem);
			const Plane *r = reinterpret_cast<const Plane *>(p_variant._data._mem);

			return l->is_same(*r);
		} break;

		case VariantType::AABB: {
			const ::AABB *l = _data._aabb;
			const ::AABB *r = p_variant._data._aabb;

			return l->is_same(*r);
		} break;

		case VariantType::QUATERNION: {
			const Quaternion *l = reinterpret_cast<const Quaternion *>(_data._mem);
			const Quaternion *r = reinterpret_cast<const Quaternion *>(p_variant._data._mem);

			return hash_compare_quaternion(*l, *r);
		} break;

		case VariantType::BASIS: {
			const Basis *l = _data._basis;
			const Basis *r = p_variant._data._basis;

			return l->is_same(*r);
		} break;

		case VariantType::TRANSFORM3D: {
			const Transform3D *l = _data._transform3d;
			const Transform3D *r = p_variant._data._transform3d;

			return l->is_same(*r);
		} break;
		case VariantType::PROJECTION: {
			const Projection *l = _data._projection;
			const Projection *r = p_variant._data._projection;

			return l->is_same(*r);
		} break;

		case VariantType::COLOR: {
			const Color *l = reinterpret_cast<const Color *>(_data._mem);
			const Color *r = reinterpret_cast<const Color *>(p_variant._data._mem);

			return hash_compare_color(*l, *r);
		} break;

		case VariantType::ARRAY: {
			const Array &l = *(reinterpret_cast<const Array *>(_data._mem));
			const Array &r = *(reinterpret_cast<const Array *>(p_variant._data._mem));

			if (!l.recursive_equal(r, recursion_count + 1)) {
				return false;
			}

			return true;
		} break;

		case VariantType::DICTIONARY: {
			const Dictionary &l = *(reinterpret_cast<const Dictionary *>(_data._mem));
			const Dictionary &r = *(reinterpret_cast<const Dictionary *>(p_variant._data._mem));

			if (!l.recursive_equal(r, recursion_count + 1)) {
				return false;
			}

			return true;
		} break;

		// This is for floating point comparisons only.
		case VariantType::PACKED_FLOAT32_ARRAY: {
			hash_compare_packed_array(_data.packed_array, p_variant._data.packed_array, float, hash_compare_scalar);
		} break;

		case VariantType::PACKED_FLOAT64_ARRAY: {
			hash_compare_packed_array(_data.packed_array, p_variant._data.packed_array, double, hash_compare_scalar);
		} break;

		case VariantType::PACKED_VECTOR2_ARRAY: {
			hash_compare_packed_array(_data.packed_array, p_variant._data.packed_array, Vector2, hash_compare_vector2);
		} break;

		case VariantType::PACKED_VECTOR3_ARRAY: {
			hash_compare_packed_array(_data.packed_array, p_variant._data.packed_array, Vector3, hash_compare_vector3);
		} break;

		case VariantType::PACKED_COLOR_ARRAY: {
			hash_compare_packed_array(_data.packed_array, p_variant._data.packed_array, Color, hash_compare_color);
		} break;

		case VariantType::PACKED_VECTOR4_ARRAY: {
			hash_compare_packed_array(_data.packed_array, p_variant._data.packed_array, Vector4, hash_compare_vector4);
		} break;

		default:
			bool v;
			Variant r;
			evaluate(OP_EQUAL, *this, p_variant, r, v);
			return r;
	}
}

bool Variant::identity_compare(const Variant &p_variant) const {
	if (type != p_variant.type) {
		return false;
	}

	switch (type) {
		case VariantType::OBJECT: {
			return _get_obj().id == p_variant._get_obj().id;
		} break;

		case VariantType::DICTIONARY: {
			const Dictionary &l = *(reinterpret_cast<const Dictionary *>(_data._mem));
			const Dictionary &r = *(reinterpret_cast<const Dictionary *>(p_variant._data._mem));
			return l.id() == r.id();
		} break;

		case VariantType::ARRAY: {
			const Array &l = *(reinterpret_cast<const Array *>(_data._mem));
			const Array &r = *(reinterpret_cast<const Array *>(p_variant._data._mem));
			return l.id() == r.id();
		} break;

		case VariantType::PACKED_BYTE_ARRAY:
		case VariantType::PACKED_INT32_ARRAY:
		case VariantType::PACKED_INT64_ARRAY:
		case VariantType::PACKED_FLOAT32_ARRAY:
		case VariantType::PACKED_FLOAT64_ARRAY:
		case VariantType::PACKED_STRING_ARRAY:
		case VariantType::PACKED_VECTOR2_ARRAY:
		case VariantType::PACKED_VECTOR3_ARRAY:
		case VariantType::PACKED_COLOR_ARRAY:
		case VariantType::PACKED_VECTOR4_ARRAY: {
			return _data.packed_array == p_variant._data.packed_array;
		} break;

		default: {
			return hash_compare(p_variant);
		}
	}
}

bool StringLikeVariantComparator::compare(const Variant &p_lhs, const Variant &p_rhs) {
	if (p_lhs.hash_compare(p_rhs)) {
		return true;
	}
	if (p_lhs.get_type() == VariantType::STRING && p_rhs.get_type() == VariantType::STRING_NAME) {
		return *VariantInternal::get_string(&p_lhs) == *VariantInternal::get_string_name(&p_rhs);
	}
	if (p_lhs.get_type() == VariantType::STRING_NAME && p_rhs.get_type() == VariantType::STRING) {
		return *VariantInternal::get_string_name(&p_lhs) == *VariantInternal::get_string(&p_rhs);
	}
	return false;
}

bool StringLikeVariantOrder::compare(const Variant &p_lhs, const Variant &p_rhs) {
	if (p_lhs.get_type() == VariantType::STRING) {
		const String &lhs = *VariantInternal::get_string(&p_lhs);
		if (p_rhs.get_type() == VariantType::STRING) {
			return StringName::AlphCompare::compare(lhs, *VariantInternal::get_string(&p_rhs));
		} else if (p_rhs.get_type() == VariantType::STRING_NAME) {
			return StringName::AlphCompare::compare(lhs, *VariantInternal::get_string_name(&p_rhs));
		}
	} else if (p_lhs.get_type() == VariantType::STRING_NAME) {
		const StringName &lhs = *VariantInternal::get_string_name(&p_lhs);
		if (p_rhs.get_type() == VariantType::STRING) {
			return StringName::AlphCompare::compare(lhs, *VariantInternal::get_string(&p_rhs));
		} else if (p_rhs.get_type() == VariantType::STRING_NAME) {
			return StringName::AlphCompare::compare(lhs, *VariantInternal::get_string_name(&p_rhs));
		}
	}

	return p_lhs < p_rhs;
}

bool Variant::is_ref_counted() const {
	return type == VariantType::OBJECT && _get_obj().id.is_ref_counted();
}

bool Variant::is_shared() const {
	return is_type_shared(type);
}

bool Variant::is_read_only() const {
	switch (type) {
		case VariantType::ARRAY:
			return reinterpret_cast<const Array *>(_data._mem)->is_read_only();
		case VariantType::DICTIONARY:
			return reinterpret_cast<const Dictionary *>(_data._mem)->is_read_only();
		default:
			return false;
	}
}

void Variant::_variant_call_error(const String &p_method, Callable::CallError &error) {
	switch (error.error) {
		case Callable::CallError::CALL_ERROR_INVALID_ARGUMENT: {
			String err = "Invalid type for argument #" + itos(error.argument) + ", expected '" + VariantType::get_type_name(VariantType::Type(error.expected)) + "'.";
			ERR_PRINT(err.utf8().get_data());

		} break;
		case Callable::CallError::CALL_ERROR_INVALID_METHOD: {
			String err = "Invalid method '" + p_method + "' for type '" + VariantType::get_type_name(type) + "'.";
			ERR_PRINT(err.utf8().get_data());
		} break;
		case Callable::CallError::CALL_ERROR_TOO_MANY_ARGUMENTS: {
			String err = "Too many arguments for method '" + p_method + "'";
			ERR_PRINT(err.utf8().get_data());
		} break;
		default: {
		}
	}
}

void Variant::construct_from_string(const String &p_string, Variant &r_value, ObjectConstruct p_obj_construct, void *p_construct_ud) {
	r_value = Variant();
}

String Variant::get_construct_string() const {
	String vars;
	VariantWriter::write_to_string(*this, vars);

	return vars;
}

String Variant::get_call_error_text(const StringName &p_method, const Variant **p_argptrs, int p_argcount, const Callable::CallError &ce) {
	return get_call_error_text(nullptr, p_method, p_argptrs, p_argcount, ce);
}

String Variant::get_call_error_text(Object *p_base, const StringName &p_method, const Variant **p_argptrs, int p_argcount, const Callable::CallError &ce) {
	String err_text;

	if (ce.error == Callable::CallError::CALL_ERROR_INVALID_ARGUMENT) {
		int errorarg = ce.argument;
		if (p_argptrs) {
			err_text = "Cannot convert argument " + itos(errorarg + 1) + " from " + VariantType::get_type_name(p_argptrs[errorarg]->get_type()) + " to " + VariantType::get_type_name(VariantType::Type(ce.expected));
		} else {
			err_text = "Cannot convert argument " + itos(errorarg + 1) + " from [missing argptr, type unknown] to " + VariantType::get_type_name(VariantType::Type(ce.expected));
		}
	} else if (ce.error == Callable::CallError::CALL_ERROR_TOO_MANY_ARGUMENTS) {
		err_text = "Method expected " + itos(ce.expected) + " argument(s), but called with " + itos(p_argcount);
	} else if (ce.error == Callable::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS) {
		err_text = "Method expected " + itos(ce.expected) + " argument(s), but called with " + itos(p_argcount);
	} else if (ce.error == Callable::CallError::CALL_ERROR_INVALID_METHOD) {
		err_text = "Method not found";
	} else if (ce.error == Callable::CallError::CALL_ERROR_INSTANCE_IS_NULL) {
		err_text = "Instance is null";
	} else if (ce.error == Callable::CallError::CALL_ERROR_METHOD_NOT_CONST) {
		err_text = "Method not const in const instance";
	} else if (ce.error == Callable::CallError::CALL_OK) {
		return "Call OK";
	}

	String base_text;
	if (p_base) {
		base_text = p_base->get_class();
		Ref<Resource> script = p_base->get_script();
		if (script.is_valid() && script->get_path().is_resource_file()) {
			base_text += "(" + script->get_path().get_file() + ")";
		}
		base_text += "::";
	}
	return "'" + base_text + String(p_method) + "': " + err_text;
}

String Variant::get_callable_error_text(const Callable &p_callable, const Variant **p_argptrs, int p_argcount, const Callable::CallError &ce) {
	Vector<Variant> binds;
	p_callable.get_bound_arguments_ref(binds);

	int args_unbound = p_callable.get_unbound_arguments_count();

	if (p_argcount - args_unbound < 0) {
		return "Callable unbinds " + itos(args_unbound) + " arguments, but called with " + itos(p_argcount);
	} else {
		Vector<const Variant *> argptrs;
		argptrs.resize(p_argcount - args_unbound + binds.size());
		for (int i = 0; i < p_argcount - args_unbound; i++) {
			argptrs.write[i] = p_argptrs[i];
		}
		for (int i = 0; i < binds.size(); i++) {
			argptrs.write[i + p_argcount - args_unbound] = &binds[i];
		}
		return get_call_error_text(p_callable.get_object(), p_callable.get_method(), (const Variant **)argptrs.ptr(), argptrs.size(), ce);
	}
}

void Variant::register_types() {
	_register_variant_operators();
	_register_variant_methods();
	_register_variant_setters_getters();
	_register_variant_constructors();
	_register_variant_destructors();
	_register_variant_utility_functions();
}
void Variant::unregister_types() {
	_unregister_variant_operators();
	_unregister_variant_methods();
	_unregister_variant_setters_getters();
	_unregister_variant_destructors();
	_unregister_variant_utility_functions();
}
