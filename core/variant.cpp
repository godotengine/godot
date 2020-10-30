/*************************************************************************/
/*  variant.cpp                                                          */
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

#include "variant.h"

#include "core/core_string_names.h"
#include "core/io/marshalls.h"
#include "core/math/math_funcs.h"
#include "core/object_rc.h"
#include "core/print_string.h"
#include "core/resource.h"
#include "core/variant_parser.h"
#include "scene/gui/control.h"
#include "scene/main/node.h"

String Variant::get_type_name(Variant::Type p_type) {

	switch (p_type) {
		case NIL: {

			return "Nil";
		} break;

		// atomic types
		case BOOL: {

			return "bool";
		} break;
		case INT: {

			return "int";

		} break;
		case REAL: {

			return "float";

		} break;
		case STRING: {

			return "String";
		} break;

		// math types
		case VECTOR2: {

			return "Vector2";
		} break;
		case RECT2: {

			return "Rect2";
		} break;
		case TRANSFORM2D: {

			return "Transform2D";
		} break;
		case VECTOR3: {

			return "Vector3";
		} break;
		case PLANE: {

			return "Plane";

		} break;
		/*
			case QUAT: {


			} break;*/
		case AABB: {

			return "AABB";
		} break;
		case QUAT: {

			return "Quat";

		} break;
		case BASIS: {

			return "Basis";

		} break;
		case TRANSFORM: {

			return "Transform";

		} break;

		// misc types
		case COLOR: {

			return "Color";

		} break;
		case _RID: {

			return "RID";
		} break;
		case OBJECT: {

			return "Object";
		} break;
		case NODE_PATH: {

			return "NodePath";

		} break;
		case DICTIONARY: {

			return "Dictionary";

		} break;
		case ARRAY: {

			return "Array";

		} break;

		// arrays
		case POOL_BYTE_ARRAY: {

			return "PoolByteArray";

		} break;
		case POOL_INT_ARRAY: {

			return "PoolIntArray";

		} break;
		case POOL_REAL_ARRAY: {

			return "PoolRealArray";

		} break;
		case POOL_STRING_ARRAY: {

			return "PoolStringArray";
		} break;
		case POOL_VECTOR2_ARRAY: {

			return "PoolVector2Array";

		} break;
		case POOL_VECTOR3_ARRAY: {

			return "PoolVector3Array";

		} break;
		case POOL_COLOR_ARRAY: {

			return "PoolColorArray";

		} break;
		default: {
		}
	}

	return "";
}

bool Variant::can_convert(Variant::Type p_type_from, Variant::Type p_type_to) {

	if (p_type_from == p_type_to)
		return true;
	if (p_type_to == NIL && p_type_from != NIL) //nil can convert to anything
		return true;

	if (p_type_from == NIL) {
		return (p_type_to == OBJECT);
	};

	const Type *valid_types = NULL;
	const Type *invalid_types = NULL;

	switch (p_type_to) {
		case BOOL: {

			static const Type valid[] = {
				INT,
				REAL,
				STRING,
				NIL,
			};

			valid_types = valid;
		} break;
		case INT: {

			static const Type valid[] = {
				BOOL,
				REAL,
				STRING,
				NIL,
			};

			valid_types = valid;

		} break;
		case REAL: {

			static const Type valid[] = {
				BOOL,
				INT,
				STRING,
				NIL,
			};

			valid_types = valid;

		} break;
		case STRING: {

			static const Type invalid[] = {
				OBJECT,
				NIL
			};

			invalid_types = invalid;
		} break;
		case TRANSFORM2D: {

			static const Type valid[] = {
				TRANSFORM,
				NIL
			};

			valid_types = valid;
		} break;
		case QUAT: {

			static const Type valid[] = {
				BASIS,
				NIL
			};

			valid_types = valid;

		} break;
		case BASIS: {

			static const Type valid[] = {
				QUAT,
				VECTOR3,
				NIL
			};

			valid_types = valid;

		} break;
		case TRANSFORM: {

			static const Type valid[] = {
				TRANSFORM2D,
				QUAT,
				BASIS,
				NIL
			};

			valid_types = valid;

		} break;

		case COLOR: {

			static const Type valid[] = {
				STRING,
				INT,
				NIL,
			};

			valid_types = valid;

		} break;

		case _RID: {

			static const Type valid[] = {
				OBJECT,
				NIL
			};

			valid_types = valid;
		} break;
		case OBJECT: {

			static const Type valid[] = {
				NIL
			};

			valid_types = valid;
		} break;
		case NODE_PATH: {

			static const Type valid[] = {
				STRING,
				NIL
			};

			valid_types = valid;
		} break;
		case ARRAY: {

			static const Type valid[] = {
				POOL_BYTE_ARRAY,
				POOL_INT_ARRAY,
				POOL_STRING_ARRAY,
				POOL_REAL_ARRAY,
				POOL_COLOR_ARRAY,
				POOL_VECTOR2_ARRAY,
				POOL_VECTOR3_ARRAY,
				NIL
			};

			valid_types = valid;
		} break;
		// arrays
		case POOL_BYTE_ARRAY: {

			static const Type valid[] = {
				ARRAY,
				NIL
			};

			valid_types = valid;
		} break;
		case POOL_INT_ARRAY: {

			static const Type valid[] = {
				ARRAY,
				NIL
			};
			valid_types = valid;
		} break;
		case POOL_REAL_ARRAY: {

			static const Type valid[] = {
				ARRAY,
				NIL
			};

			valid_types = valid;
		} break;
		case POOL_STRING_ARRAY: {

			static const Type valid[] = {
				ARRAY,
				NIL
			};
			valid_types = valid;
		} break;
		case POOL_VECTOR2_ARRAY: {

			static const Type valid[] = {
				ARRAY,
				NIL
			};
			valid_types = valid;

		} break;
		case POOL_VECTOR3_ARRAY: {

			static const Type valid[] = {
				ARRAY,
				NIL
			};
			valid_types = valid;

		} break;
		case POOL_COLOR_ARRAY: {

			static const Type valid[] = {
				ARRAY,
				NIL
			};

			valid_types = valid;

		} break;
		default: {
		}
	}

	if (valid_types) {

		int i = 0;
		while (valid_types[i] != NIL) {

			if (p_type_from == valid_types[i])
				return true;
			i++;
		}

	} else if (invalid_types) {

		int i = 0;
		while (invalid_types[i] != NIL) {

			if (p_type_from == invalid_types[i])
				return false;
			i++;
		}

		return true;
	}

	return false;
}

bool Variant::can_convert_strict(Variant::Type p_type_from, Variant::Type p_type_to) {

	if (p_type_from == p_type_to)
		return true;
	if (p_type_to == NIL && p_type_from != NIL) //nil can convert to anything
		return true;

	if (p_type_from == NIL) {
		return (p_type_to == OBJECT);
	};

	const Type *valid_types = NULL;

	switch (p_type_to) {
		case BOOL: {

			static const Type valid[] = {
				INT,
				REAL,
				//STRING,
				NIL,
			};

			valid_types = valid;
		} break;
		case INT: {

			static const Type valid[] = {
				BOOL,
				REAL,
				//STRING,
				NIL,
			};

			valid_types = valid;

		} break;
		case REAL: {

			static const Type valid[] = {
				BOOL,
				INT,
				//STRING,
				NIL,
			};

			valid_types = valid;

		} break;
		case STRING: {

			static const Type valid[] = {
				NODE_PATH,
				NIL
			};

			valid_types = valid;
		} break;
		case TRANSFORM2D: {

			static const Type valid[] = {
				TRANSFORM,
				NIL
			};

			valid_types = valid;
		} break;
		case QUAT: {

			static const Type valid[] = {
				BASIS,
				NIL
			};

			valid_types = valid;

		} break;
		case BASIS: {

			static const Type valid[] = {
				QUAT,
				VECTOR3,
				NIL
			};

			valid_types = valid;

		} break;
		case TRANSFORM: {

			static const Type valid[] = {
				TRANSFORM2D,
				QUAT,
				BASIS,
				NIL
			};

			valid_types = valid;

		} break;

		case COLOR: {

			static const Type valid[] = {
				STRING,
				INT,
				NIL,
			};

			valid_types = valid;

		} break;

		case _RID: {

			static const Type valid[] = {
				OBJECT,
				NIL
			};

			valid_types = valid;
		} break;
		case OBJECT: {

			static const Type valid[] = {
				NIL
			};

			valid_types = valid;
		} break;
		case NODE_PATH: {

			static const Type valid[] = {
				STRING,
				NIL
			};

			valid_types = valid;
		} break;
		case ARRAY: {

			static const Type valid[] = {
				POOL_BYTE_ARRAY,
				POOL_INT_ARRAY,
				POOL_STRING_ARRAY,
				POOL_REAL_ARRAY,
				POOL_COLOR_ARRAY,
				POOL_VECTOR2_ARRAY,
				POOL_VECTOR3_ARRAY,
				NIL
			};

			valid_types = valid;
		} break;
		// arrays
		case POOL_BYTE_ARRAY: {

			static const Type valid[] = {
				ARRAY,
				NIL
			};

			valid_types = valid;
		} break;
		case POOL_INT_ARRAY: {

			static const Type valid[] = {
				ARRAY,
				NIL
			};
			valid_types = valid;
		} break;
		case POOL_REAL_ARRAY: {

			static const Type valid[] = {
				ARRAY,
				NIL
			};

			valid_types = valid;
		} break;
		case POOL_STRING_ARRAY: {

			static const Type valid[] = {
				ARRAY,
				NIL
			};
			valid_types = valid;
		} break;
		case POOL_VECTOR2_ARRAY: {

			static const Type valid[] = {
				ARRAY,
				NIL
			};
			valid_types = valid;

		} break;
		case POOL_VECTOR3_ARRAY: {

			static const Type valid[] = {
				ARRAY,
				NIL
			};
			valid_types = valid;

		} break;
		case POOL_COLOR_ARRAY: {

			static const Type valid[] = {
				ARRAY,
				NIL
			};

			valid_types = valid;

		} break;
		default: {
		}
	}

	if (valid_types) {

		int i = 0;
		while (valid_types[i] != NIL) {

			if (p_type_from == valid_types[i])
				return true;
			i++;
		}
	}

	return false;
}

bool Variant::operator==(const Variant &p_variant) const {

	if (type != p_variant.type) //evaluation of operator== needs to be more strict
		return false;
	bool v;
	Variant r;
	evaluate(OP_EQUAL, *this, p_variant, r, v);
	return r;
}

bool Variant::operator!=(const Variant &p_variant) const {

	if (type != p_variant.type) //evaluation of operator== needs to be more strict
		return true;
	bool v;
	Variant r;
	evaluate(OP_NOT_EQUAL, *this, p_variant, r, v);
	return r;
}

bool Variant::operator<(const Variant &p_variant) const {
	if (type != p_variant.type) //if types differ, then order by type first
		return type < p_variant.type;
	bool v;
	Variant r;
	evaluate(OP_LESS, *this, p_variant, r, v);
	return r;
}

bool Variant::is_zero() const {

	switch (type) {
		case NIL: {

			return true;
		} break;

		// atomic types
		case BOOL: {

			return !(_data._bool);
		} break;
		case INT: {

			return _data._int == 0;

		} break;
		case REAL: {

			return _data._real == 0;

		} break;
		case STRING: {

			return *reinterpret_cast<const String *>(_data._mem) == String();

		} break;

		// math types
		case VECTOR2: {

			return *reinterpret_cast<const Vector2 *>(_data._mem) == Vector2();

		} break;
		case RECT2: {

			return *reinterpret_cast<const Rect2 *>(_data._mem) == Rect2();

		} break;
		case TRANSFORM2D: {

			return *_data._transform2d == Transform2D();

		} break;
		case VECTOR3: {

			return *reinterpret_cast<const Vector3 *>(_data._mem) == Vector3();

		} break;
		case PLANE: {

			return *reinterpret_cast<const Plane *>(_data._mem) == Plane();

		} break;
		/*
		case QUAT: {


		} break;*/
		case AABB: {

			return *_data._aabb == ::AABB();
		} break;
		case QUAT: {

			return *reinterpret_cast<const Quat *>(_data._mem) == Quat();

		} break;
		case BASIS: {

			return *_data._basis == Basis();

		} break;
		case TRANSFORM: {

			return *_data._transform == Transform();

		} break;

		// misc types
		case COLOR: {

			return *reinterpret_cast<const Color *>(_data._mem) == Color();

		} break;
		case _RID: {

			return *reinterpret_cast<const RID *>(_data._mem) == RID();
		} break;
		case OBJECT: {

			return _UNSAFE_OBJ_PROXY_PTR(*this) == NULL;
		} break;
		case NODE_PATH: {

			return reinterpret_cast<const NodePath *>(_data._mem)->is_empty();

		} break;
		case DICTIONARY: {

			return reinterpret_cast<const Dictionary *>(_data._mem)->empty();

		} break;
		case ARRAY: {

			return reinterpret_cast<const Array *>(_data._mem)->empty();

		} break;

		// arrays
		case POOL_BYTE_ARRAY: {

			return reinterpret_cast<const PoolVector<uint8_t> *>(_data._mem)->size() == 0;

		} break;
		case POOL_INT_ARRAY: {

			return reinterpret_cast<const PoolVector<int> *>(_data._mem)->size() == 0;

		} break;
		case POOL_REAL_ARRAY: {

			return reinterpret_cast<const PoolVector<real_t> *>(_data._mem)->size() == 0;

		} break;
		case POOL_STRING_ARRAY: {

			return reinterpret_cast<const PoolVector<String> *>(_data._mem)->size() == 0;

		} break;
		case POOL_VECTOR2_ARRAY: {

			return reinterpret_cast<const PoolVector<Vector2> *>(_data._mem)->size() == 0;

		} break;
		case POOL_VECTOR3_ARRAY: {

			return reinterpret_cast<const PoolVector<Vector3> *>(_data._mem)->size() == 0;

		} break;
		case POOL_COLOR_ARRAY: {

			return reinterpret_cast<const PoolVector<Color> *>(_data._mem)->size() == 0;

		} break;
		default: {
		}
	}

	return false;
}

bool Variant::is_one() const {

	switch (type) {
		case NIL: {

			return true;
		} break;

		// atomic types
		case BOOL: {

			return _data._bool;
		} break;
		case INT: {

			return _data._int == 1;

		} break;
		case REAL: {

			return _data._real == 1;

		} break;
		case VECTOR2: {

			return *reinterpret_cast<const Vector2 *>(_data._mem) == Vector2(1, 1);

		} break;
		case RECT2: {

			return *reinterpret_cast<const Rect2 *>(_data._mem) == Rect2(1, 1, 1, 1);

		} break;
		case VECTOR3: {

			return *reinterpret_cast<const Vector3 *>(_data._mem) == Vector3(1, 1, 1);

		} break;
		case PLANE: {

			return *reinterpret_cast<const Plane *>(_data._mem) == Plane(1, 1, 1, 1);

		} break;
		case COLOR: {

			return *reinterpret_cast<const Color *>(_data._mem) == Color(1, 1, 1, 1);

		} break;

		default: {
			return !is_zero();
		}
	}

	return false;
}

void Variant::reference(const Variant &p_variant) {

	switch (type) {
		case NIL:
		case BOOL:
		case INT:
		case REAL:
			break;
		default:
			clear();
	}

	type = p_variant.type;

	switch (p_variant.type) {
		case NIL: {

			// none
		} break;

		// atomic types
		case BOOL: {

			_data._bool = p_variant._data._bool;
		} break;
		case INT: {

			_data._int = p_variant._data._int;
		} break;
		case REAL: {

			_data._real = p_variant._data._real;
		} break;
		case STRING: {

			memnew_placement(_data._mem, String(*reinterpret_cast<const String *>(p_variant._data._mem)));
		} break;

		// math types
		case VECTOR2: {

			memnew_placement(_data._mem, Vector2(*reinterpret_cast<const Vector2 *>(p_variant._data._mem)));
		} break;
		case RECT2: {

			memnew_placement(_data._mem, Rect2(*reinterpret_cast<const Rect2 *>(p_variant._data._mem)));
		} break;
		case TRANSFORM2D: {

			_data._transform2d = memnew(Transform2D(*p_variant._data._transform2d));
		} break;
		case VECTOR3: {

			memnew_placement(_data._mem, Vector3(*reinterpret_cast<const Vector3 *>(p_variant._data._mem)));
		} break;
		case PLANE: {

			memnew_placement(_data._mem, Plane(*reinterpret_cast<const Plane *>(p_variant._data._mem)));
		} break;

		case AABB: {

			_data._aabb = memnew(::AABB(*p_variant._data._aabb));
		} break;
		case QUAT: {

			memnew_placement(_data._mem, Quat(*reinterpret_cast<const Quat *>(p_variant._data._mem)));

		} break;
		case BASIS: {

			_data._basis = memnew(Basis(*p_variant._data._basis));

		} break;
		case TRANSFORM: {

			_data._transform = memnew(Transform(*p_variant._data._transform));
		} break;

		// misc types
		case COLOR: {

			memnew_placement(_data._mem, Color(*reinterpret_cast<const Color *>(p_variant._data._mem)));

		} break;
		case _RID: {

			memnew_placement(_data._mem, RID(*reinterpret_cast<const RID *>(p_variant._data._mem)));
		} break;
		case OBJECT: {

			memnew_placement(_data._mem, ObjData(p_variant._get_obj()));
#ifdef DEBUG_ENABLED
			if (_get_obj().rc) {
				_get_obj().rc->increment();
			}
#endif
		} break;
		case NODE_PATH: {

			memnew_placement(_data._mem, NodePath(*reinterpret_cast<const NodePath *>(p_variant._data._mem)));

		} break;
		case DICTIONARY: {

			memnew_placement(_data._mem, Dictionary(*reinterpret_cast<const Dictionary *>(p_variant._data._mem)));

		} break;
		case ARRAY: {

			memnew_placement(_data._mem, Array(*reinterpret_cast<const Array *>(p_variant._data._mem)));

		} break;

		// arrays
		case POOL_BYTE_ARRAY: {

			memnew_placement(_data._mem, PoolVector<uint8_t>(*reinterpret_cast<const PoolVector<uint8_t> *>(p_variant._data._mem)));

		} break;
		case POOL_INT_ARRAY: {

			memnew_placement(_data._mem, PoolVector<int>(*reinterpret_cast<const PoolVector<int> *>(p_variant._data._mem)));

		} break;
		case POOL_REAL_ARRAY: {

			memnew_placement(_data._mem, PoolVector<real_t>(*reinterpret_cast<const PoolVector<real_t> *>(p_variant._data._mem)));

		} break;
		case POOL_STRING_ARRAY: {

			memnew_placement(_data._mem, PoolVector<String>(*reinterpret_cast<const PoolVector<String> *>(p_variant._data._mem)));

		} break;
		case POOL_VECTOR2_ARRAY: {

			memnew_placement(_data._mem, PoolVector<Vector2>(*reinterpret_cast<const PoolVector<Vector2> *>(p_variant._data._mem)));

		} break;
		case POOL_VECTOR3_ARRAY: {

			memnew_placement(_data._mem, PoolVector<Vector3>(*reinterpret_cast<const PoolVector<Vector3> *>(p_variant._data._mem)));

		} break;
		case POOL_COLOR_ARRAY: {

			memnew_placement(_data._mem, PoolVector<Color>(*reinterpret_cast<const PoolVector<Color> *>(p_variant._data._mem)));

		} break;
		default: {
		}
	}
}

void Variant::zero() {
	switch (type) {
		case NIL: break;
		case BOOL: this->_data._bool = false; break;
		case INT: this->_data._int = 0; break;
		case REAL: this->_data._real = 0; break;
		case VECTOR2: *reinterpret_cast<Vector2 *>(this->_data._mem) = Vector2(); break;
		case RECT2: *reinterpret_cast<Rect2 *>(this->_data._mem) = Rect2(); break;
		case VECTOR3: *reinterpret_cast<Vector3 *>(this->_data._mem) = Vector3(); break;
		case PLANE: *reinterpret_cast<Plane *>(this->_data._mem) = Plane(); break;
		case QUAT: *reinterpret_cast<Quat *>(this->_data._mem) = Quat(); break;
		case COLOR: *reinterpret_cast<Color *>(this->_data._mem) = Color(); break;
		default: this->clear(); break;
	}
}

void Variant::clear() {

	switch (type) {
		case STRING: {

			reinterpret_cast<String *>(_data._mem)->~String();
		} break;
		/*
		// no point, they don't allocate memory
		VECTOR3,
		PLANE,
		QUAT,
		COLOR,
		VECTOR2,
		RECT2
	*/
		case TRANSFORM2D: {

			memdelete(_data._transform2d);
		} break;
		case AABB: {

			memdelete(_data._aabb);
		} break;
		case BASIS: {

			memdelete(_data._basis);
		} break;
		case TRANSFORM: {

			memdelete(_data._transform);
		} break;

		// misc types
		case NODE_PATH: {

			reinterpret_cast<NodePath *>(_data._mem)->~NodePath();
		} break;
		case OBJECT: {

#ifdef DEBUG_ENABLED
			if (likely(_get_obj().rc)) {
				if (unlikely(_get_obj().rc->decrement())) {
					memdelete(_get_obj().rc);
				}
			} else {
				_get_obj().ref.unref();
			}
#else
			_get_obj().obj = NULL;
			_get_obj().ref.unref();
#endif
		} break;
		case _RID: {
			// not much need probably
			reinterpret_cast<RID *>(_data._mem)->~RID();
		} break;
		case DICTIONARY: {

			reinterpret_cast<Dictionary *>(_data._mem)->~Dictionary();
		} break;
		case ARRAY: {

			reinterpret_cast<Array *>(_data._mem)->~Array();
		} break;
		// arrays
		case POOL_BYTE_ARRAY: {

			reinterpret_cast<PoolVector<uint8_t> *>(_data._mem)->~PoolVector<uint8_t>();
		} break;
		case POOL_INT_ARRAY: {

			reinterpret_cast<PoolVector<int> *>(_data._mem)->~PoolVector<int>();
		} break;
		case POOL_REAL_ARRAY: {

			reinterpret_cast<PoolVector<real_t> *>(_data._mem)->~PoolVector<real_t>();
		} break;
		case POOL_STRING_ARRAY: {

			reinterpret_cast<PoolVector<String> *>(_data._mem)->~PoolVector<String>();
		} break;
		case POOL_VECTOR2_ARRAY: {

			reinterpret_cast<PoolVector<Vector2> *>(_data._mem)->~PoolVector<Vector2>();
		} break;
		case POOL_VECTOR3_ARRAY: {

			reinterpret_cast<PoolVector<Vector3> *>(_data._mem)->~PoolVector<Vector3>();
		} break;
		case POOL_COLOR_ARRAY: {

			reinterpret_cast<PoolVector<Color> *>(_data._mem)->~PoolVector<Color>();
		} break;
		default: {
		} /* not needed */
	}

	type = NIL;
}

Variant::operator signed int() const {

	switch (type) {

		case NIL: return 0;
		case BOOL: return _data._bool ? 1 : 0;
		case INT: return _data._int;
		case REAL: return _data._real;
		case STRING: return operator String().to_int();
		default: {

			return 0;
		}
	}
}
Variant::operator unsigned int() const {

	switch (type) {

		case NIL: return 0;
		case BOOL: return _data._bool ? 1 : 0;
		case INT: return _data._int;
		case REAL: return _data._real;
		case STRING: return operator String().to_int();
		default: {

			return 0;
		}
	}
}

Variant::operator int64_t() const {

	switch (type) {

		case NIL: return 0;
		case BOOL: return _data._bool ? 1 : 0;
		case INT: return _data._int;
		case REAL: return _data._real;
		case STRING: return operator String().to_int64();
		default: {

			return 0;
		}
	}
}

/*
Variant::operator long unsigned int() const {

	switch( type ) {

		case NIL: return 0;
		case BOOL: return _data._bool ? 1 : 0;
		case INT: return _data._int;
		case REAL: return _data._real;
		case STRING: return operator String().to_int();
		default: {

			return 0;
		}
	}

	return 0;
};
*/

Variant::operator uint64_t() const {

	switch (type) {

		case NIL: return 0;
		case BOOL: return _data._bool ? 1 : 0;
		case INT: return _data._int;
		case REAL: return _data._real;
		case STRING: return operator String().to_int();
		default: {

			return 0;
		}
	}
}

#ifdef NEED_LONG_INT
Variant::operator signed long() const {

	switch (type) {

		case NIL: return 0;
		case BOOL: return _data._bool ? 1 : 0;
		case INT: return _data._int;
		case REAL: return _data._real;
		case STRING: return operator String().to_int();
		default: {

			return 0;
		}
	}

	return 0;
};

Variant::operator unsigned long() const {

	switch (type) {

		case NIL: return 0;
		case BOOL: return _data._bool ? 1 : 0;
		case INT: return _data._int;
		case REAL: return _data._real;
		case STRING: return operator String().to_int();
		default: {

			return 0;
		}
	}

	return 0;
};
#endif

Variant::operator signed short() const {

	switch (type) {

		case NIL: return 0;
		case BOOL: return _data._bool ? 1 : 0;
		case INT: return _data._int;
		case REAL: return _data._real;
		case STRING: return operator String().to_int();
		default: {

			return 0;
		}
	}
}
Variant::operator unsigned short() const {

	switch (type) {

		case NIL: return 0;
		case BOOL: return _data._bool ? 1 : 0;
		case INT: return _data._int;
		case REAL: return _data._real;
		case STRING: return operator String().to_int();
		default: {

			return 0;
		}
	}
}
Variant::operator signed char() const {

	switch (type) {

		case NIL: return 0;
		case BOOL: return _data._bool ? 1 : 0;
		case INT: return _data._int;
		case REAL: return _data._real;
		case STRING: return operator String().to_int();
		default: {

			return 0;
		}
	}
}
Variant::operator unsigned char() const {

	switch (type) {

		case NIL: return 0;
		case BOOL: return _data._bool ? 1 : 0;
		case INT: return _data._int;
		case REAL: return _data._real;
		case STRING: return operator String().to_int();
		default: {

			return 0;
		}
	}
}

Variant::operator CharType() const {

	return operator unsigned int();
}

Variant::operator float() const {

	switch (type) {

		case NIL: return 0;
		case BOOL: return _data._bool ? 1.0 : 0.0;
		case INT: return (float)_data._int;
		case REAL: return _data._real;
		case STRING: return operator String().to_double();
		default: {

			return 0;
		}
	}
}
Variant::operator double() const {

	switch (type) {

		case NIL: return 0;
		case BOOL: return _data._bool ? 1.0 : 0.0;
		case INT: return (double)_data._int;
		case REAL: return _data._real;
		case STRING: return operator String().to_double();
		default: {

			return 0;
		}
	}
}

Variant::operator StringName() const {

	if (type == NODE_PATH) {
		return reinterpret_cast<const NodePath *>(_data._mem)->get_sname();
	}
	return StringName(operator String());
}

struct _VariantStrPair {

	String key;
	String value;

	bool operator<(const _VariantStrPair &p) const {

		return key < p.key;
	}
};

Variant::operator String() const {
	List<const void *> stack;

	return stringify(stack);
}

String Variant::stringify(List<const void *> &stack) const {
	switch (type) {

		case NIL: return "Null";
		case BOOL: return _data._bool ? "True" : "False";
		case INT: return itos(_data._int);
		case REAL: return rtos(_data._real);
		case STRING: return *reinterpret_cast<const String *>(_data._mem);
		case VECTOR2: return "(" + operator Vector2() + ")";
		case RECT2: return "(" + operator Rect2() + ")";
		case TRANSFORM2D: {

			Transform2D mat32 = operator Transform2D();
			return "(" + Variant(mat32.elements[0]).operator String() + ", " + Variant(mat32.elements[1]).operator String() + ", " + Variant(mat32.elements[2]).operator String() + ")";
		} break;
		case VECTOR3: return "(" + operator Vector3() + ")";
		case PLANE:
			return operator Plane();
		//case QUAT:
		case AABB: return operator ::AABB();
		case QUAT: return "(" + operator Quat() + ")";
		case BASIS: {

			Basis mat3 = operator Basis();

			String mtx("(");
			for (int i = 0; i < 3; i++) {

				if (i != 0)
					mtx += ", ";

				mtx += "(";

				for (int j = 0; j < 3; j++) {

					if (j != 0)
						mtx += ", ";

					mtx += Variant(mat3.elements[i][j]).operator String();
				}

				mtx += ")";
			}

			return mtx + ")";
		} break;
		case TRANSFORM: return operator Transform();
		case NODE_PATH: return operator NodePath();
		case COLOR: return String::num(operator Color().r) + "," + String::num(operator Color().g) + "," + String::num(operator Color().b) + "," + String::num(operator Color().a);
		case DICTIONARY: {

			const Dictionary &d = *reinterpret_cast<const Dictionary *>(_data._mem);
			if (stack.find(d.id())) {
				return "{...}";
			}

			stack.push_back(d.id());

			//const String *K=NULL;
			String str("{");
			List<Variant> keys;
			d.get_key_list(&keys);

			Vector<_VariantStrPair> pairs;

			for (List<Variant>::Element *E = keys.front(); E; E = E->next()) {

				_VariantStrPair sp;
				sp.key = E->get().stringify(stack);
				sp.value = d[E->get()].stringify(stack);

				pairs.push_back(sp);
			}

			pairs.sort();

			for (int i = 0; i < pairs.size(); i++) {
				if (i > 0)
					str += ", ";
				str += pairs[i].key + ":" + pairs[i].value;
			}
			str += "}";

			return str;
		} break;
		case POOL_VECTOR2_ARRAY: {

			PoolVector<Vector2> vec = operator PoolVector<Vector2>();
			String str("[");
			for (int i = 0; i < vec.size(); i++) {

				if (i > 0)
					str += ", ";
				str = str + Variant(vec[i]);
			}
			str += "]";
			return str;
		} break;
		case POOL_VECTOR3_ARRAY: {

			PoolVector<Vector3> vec = operator PoolVector<Vector3>();
			String str("[");
			for (int i = 0; i < vec.size(); i++) {

				if (i > 0)
					str += ", ";
				str = str + Variant(vec[i]);
			}
			str += "]";
			return str;
		} break;
		case POOL_STRING_ARRAY: {

			PoolVector<String> vec = operator PoolVector<String>();
			String str("[");
			for (int i = 0; i < vec.size(); i++) {

				if (i > 0)
					str += ", ";
				str = str + vec[i];
			}
			str += "]";
			return str;
		} break;
		case POOL_INT_ARRAY: {

			PoolVector<int> vec = operator PoolVector<int>();
			String str("[");
			for (int i = 0; i < vec.size(); i++) {

				if (i > 0)
					str += ", ";
				str = str + itos(vec[i]);
			}
			str += "]";
			return str;
		} break;
		case POOL_REAL_ARRAY: {

			PoolVector<real_t> vec = operator PoolVector<real_t>();
			String str("[");
			for (int i = 0; i < vec.size(); i++) {

				if (i > 0)
					str += ", ";
				str = str + rtos(vec[i]);
			}
			str += "]";
			return str;
		} break;
		case ARRAY: {

			Array arr = operator Array();
			if (stack.find(arr.id())) {
				return "[...]";
			}
			stack.push_back(arr.id());

			String str("[");
			for (int i = 0; i < arr.size(); i++) {
				if (i)
					str += ", ";

				str += arr[i].stringify(stack);
			}

			str += "]";
			return str;

		} break;
		case OBJECT: {

			Object *obj = _OBJ_PTR(*this);
			if (obj) {
				if (_get_obj().ref.is_null() && !ObjectDB::get_instance(obj->get_instance_id())) {
					return "[Deleted Object]";
				}

				return obj->to_string();
			} else {
#ifdef DEBUG_ENABLED
				if (ScriptDebugger::get_singleton() && _get_obj().rc && !ObjectDB::get_instance(_get_obj().rc->instance_id)) {
					return "[Deleted Object]";
				}
#endif
				return "[Object:null]";
			}
		} break;
		default: {
			return "[" + get_type_name(type) + "]";
		}
	}

	return "";
}

Variant::operator Vector2() const {

	if (type == VECTOR2)
		return *reinterpret_cast<const Vector2 *>(_data._mem);
	else if (type == VECTOR3)
		return Vector2(reinterpret_cast<const Vector3 *>(_data._mem)->x, reinterpret_cast<const Vector3 *>(_data._mem)->y);
	else
		return Vector2();
}
Variant::operator Rect2() const {

	if (type == RECT2)
		return *reinterpret_cast<const Rect2 *>(_data._mem);
	else
		return Rect2();
}

Variant::operator Vector3() const {

	if (type == VECTOR3)
		return *reinterpret_cast<const Vector3 *>(_data._mem);
	else if (type == VECTOR2)
		return Vector3(reinterpret_cast<const Vector2 *>(_data._mem)->x, reinterpret_cast<const Vector2 *>(_data._mem)->y, 0.0);
	else
		return Vector3();
}
Variant::operator Plane() const {

	if (type == PLANE)
		return *reinterpret_cast<const Plane *>(_data._mem);
	else
		return Plane();
}
Variant::operator ::AABB() const {

	if (type == AABB)
		return *_data._aabb;
	else
		return ::AABB();
}

Variant::operator Basis() const {

	if (type == BASIS)
		return *_data._basis;
	else if (type == QUAT)
		return *reinterpret_cast<const Quat *>(_data._mem);
	else if (type == VECTOR3) {
		return Basis(*reinterpret_cast<const Vector3 *>(_data._mem));
	} else if (type == TRANSFORM) // unexposed in Variant::can_convert?
		return _data._transform->basis;
	else
		return Basis();
}

Variant::operator Quat() const {

	if (type == QUAT)
		return *reinterpret_cast<const Quat *>(_data._mem);
	else if (type == BASIS)
		return *_data._basis;
	else if (type == TRANSFORM)
		return _data._transform->basis;
	else
		return Quat();
}

Variant::operator Transform() const {

	if (type == TRANSFORM)
		return *_data._transform;
	else if (type == BASIS)
		return Transform(*_data._basis, Vector3());
	else if (type == QUAT)
		return Transform(Basis(*reinterpret_cast<const Quat *>(_data._mem)), Vector3());
	else if (type == TRANSFORM2D) {
		const Transform2D &t = *_data._transform2d;
		Transform m;
		m.basis.elements[0][0] = t.elements[0][0];
		m.basis.elements[1][0] = t.elements[0][1];
		m.basis.elements[0][1] = t.elements[1][0];
		m.basis.elements[1][1] = t.elements[1][1];
		m.origin[0] = t.elements[2][0];
		m.origin[1] = t.elements[2][1];
		return m;
	} else
		return Transform();
}

Variant::operator Transform2D() const {

	if (type == TRANSFORM2D) {
		return *_data._transform2d;
	} else if (type == TRANSFORM) {
		const Transform &t = *_data._transform;
		Transform2D m;
		m.elements[0][0] = t.basis.elements[0][0];
		m.elements[0][1] = t.basis.elements[1][0];
		m.elements[1][0] = t.basis.elements[0][1];
		m.elements[1][1] = t.basis.elements[1][1];
		m.elements[2][0] = t.origin[0];
		m.elements[2][1] = t.origin[1];
		return m;
	} else
		return Transform2D();
}

Variant::operator Color() const {

	if (type == COLOR)
		return *reinterpret_cast<const Color *>(_data._mem);
	else if (type == STRING)
		return Color::html(operator String());
	else if (type == INT)
		return Color::hex(operator int());
	else
		return Color();
}

Variant::operator NodePath() const {

	if (type == NODE_PATH)
		return *reinterpret_cast<const NodePath *>(_data._mem);
	else if (type == STRING)
		return NodePath(operator String());
	else
		return NodePath();
}

Variant::operator RefPtr() const {

	if (type == OBJECT)
		return _get_obj().ref;
	else
		return RefPtr();
}

Variant::operator RID() const {

	if (type == _RID) {
		return *reinterpret_cast<const RID *>(_data._mem);
	} else if (type == OBJECT) {
		if (!_get_obj().ref.is_null()) {
			return _get_obj().ref.get_rid();
		} else {
#ifdef DEBUG_ENABLED
			Object *obj = likely(_get_obj().rc) ? _get_obj().rc->get_ptr() : NULL;
			if (unlikely(!obj)) {
				if (ScriptDebugger::get_singleton() && _get_obj().rc && !ObjectDB::get_instance(_get_obj().rc->instance_id)) {
					WARN_PRINT("Attempted get RID on a deleted object.");
				}
				return RID();
			}
#else
			Object *obj = _get_obj().obj;
			if (unlikely(!obj)) {
				return RID();
			}
#endif
			Variant::CallError ce;
			Variant ret = obj->call(CoreStringNames::get_singleton()->get_rid, NULL, 0, ce);
			if (ce.error == Variant::CallError::CALL_OK && ret.get_type() == Variant::_RID) {
				return ret;
			} else {
				return RID();
			}
		}
	} else {
		return RID();
	}
}

Variant::operator Object *() const {

	if (type == OBJECT)
		return _OBJ_PTR(*this);
	else
		return NULL;
}
Variant::operator Node *() const {

	if (type == OBJECT) {
#ifdef DEBUG_ENABLED
		Object *obj = _get_obj().rc ? _get_obj().rc->get_ptr() : NULL;
#else
		Object *obj = _get_obj().obj;
#endif
		return Object::cast_to<Node>(obj);
	}
	return NULL;
}
Variant::operator Control *() const {

	if (type == OBJECT) {
#ifdef DEBUG_ENABLED
		Object *obj = _get_obj().rc ? _get_obj().rc->get_ptr() : NULL;
#else
		Object *obj = _get_obj().obj;
#endif
		return Object::cast_to<Control>(obj);
	}
	return NULL;
}

Variant::operator Dictionary() const {

	if (type == DICTIONARY)
		return *reinterpret_cast<const Dictionary *>(_data._mem);
	else
		return Dictionary();
}

template <class DA, class SA>
inline DA _convert_array(const SA &p_array) {

	DA da;
	da.resize(p_array.size());

	for (int i = 0; i < p_array.size(); i++) {

		da.set(i, Variant(p_array.get(i)));
	}

	return da;
}

template <class DA>
inline DA _convert_array_from_variant(const Variant &p_variant) {

	switch (p_variant.get_type()) {

		case Variant::ARRAY: {
			return _convert_array<DA, Array>(p_variant.operator Array());
		}
		case Variant::POOL_BYTE_ARRAY: {
			return _convert_array<DA, PoolVector<uint8_t> >(p_variant.operator PoolVector<uint8_t>());
		}
		case Variant::POOL_INT_ARRAY: {
			return _convert_array<DA, PoolVector<int> >(p_variant.operator PoolVector<int>());
		}
		case Variant::POOL_REAL_ARRAY: {
			return _convert_array<DA, PoolVector<real_t> >(p_variant.operator PoolVector<real_t>());
		}
		case Variant::POOL_STRING_ARRAY: {
			return _convert_array<DA, PoolVector<String> >(p_variant.operator PoolVector<String>());
		}
		case Variant::POOL_VECTOR2_ARRAY: {
			return _convert_array<DA, PoolVector<Vector2> >(p_variant.operator PoolVector<Vector2>());
		}
		case Variant::POOL_VECTOR3_ARRAY: {
			return _convert_array<DA, PoolVector<Vector3> >(p_variant.operator PoolVector<Vector3>());
		}
		case Variant::POOL_COLOR_ARRAY: {
			return _convert_array<DA, PoolVector<Color> >(p_variant.operator PoolVector<Color>());
		}
		default: {
			return DA();
		}
	}
}

Variant::operator Array() const {

	if (type == ARRAY)
		return *reinterpret_cast<const Array *>(_data._mem);
	else
		return _convert_array_from_variant<Array>(*this);
}

Variant::operator PoolVector<uint8_t>() const {

	if (type == POOL_BYTE_ARRAY)
		return *reinterpret_cast<const PoolVector<uint8_t> *>(_data._mem);
	else
		return _convert_array_from_variant<PoolVector<uint8_t> >(*this);
}
Variant::operator PoolVector<int>() const {

	if (type == POOL_INT_ARRAY)
		return *reinterpret_cast<const PoolVector<int> *>(_data._mem);
	else
		return _convert_array_from_variant<PoolVector<int> >(*this);
}
Variant::operator PoolVector<real_t>() const {

	if (type == POOL_REAL_ARRAY)
		return *reinterpret_cast<const PoolVector<real_t> *>(_data._mem);
	else
		return _convert_array_from_variant<PoolVector<real_t> >(*this);
}

Variant::operator PoolVector<String>() const {

	if (type == POOL_STRING_ARRAY)
		return *reinterpret_cast<const PoolVector<String> *>(_data._mem);
	else
		return _convert_array_from_variant<PoolVector<String> >(*this);
}
Variant::operator PoolVector<Vector3>() const {

	if (type == POOL_VECTOR3_ARRAY)
		return *reinterpret_cast<const PoolVector<Vector3> *>(_data._mem);
	else
		return _convert_array_from_variant<PoolVector<Vector3> >(*this);
}
Variant::operator PoolVector<Vector2>() const {

	if (type == POOL_VECTOR2_ARRAY)
		return *reinterpret_cast<const PoolVector<Vector2> *>(_data._mem);
	else
		return _convert_array_from_variant<PoolVector<Vector2> >(*this);
}

Variant::operator PoolVector<Color>() const {

	if (type == POOL_COLOR_ARRAY)
		return *reinterpret_cast<const PoolVector<Color> *>(_data._mem);
	else
		return _convert_array_from_variant<PoolVector<Color> >(*this);
}

/* helpers */

Variant::operator Vector<RID>() const {

	Array va = operator Array();
	Vector<RID> rids;
	rids.resize(va.size());
	for (int i = 0; i < rids.size(); i++)
		rids.write[i] = va[i];
	return rids;
}

Variant::operator Vector<Vector2>() const {

	PoolVector<Vector2> from = operator PoolVector<Vector2>();
	Vector<Vector2> to;
	int len = from.size();
	if (len == 0)
		return Vector<Vector2>();
	to.resize(len);
	PoolVector<Vector2>::Read r = from.read();
	Vector2 *w = to.ptrw();
	for (int i = 0; i < len; i++) {

		w[i] = r[i];
	}
	return to;
}

Variant::operator PoolVector<Plane>() const {

	Array va = operator Array();
	PoolVector<Plane> planes;
	int va_size = va.size();
	if (va_size == 0)
		return planes;

	planes.resize(va_size);
	PoolVector<Plane>::Write w = planes.write();

	for (int i = 0; i < va_size; i++)
		w[i] = va[i];

	return planes;
}

Variant::operator PoolVector<Face3>() const {

	PoolVector<Vector3> va = operator PoolVector<Vector3>();
	PoolVector<Face3> faces;
	int va_size = va.size();
	if (va_size == 0)
		return faces;

	faces.resize(va_size / 3);
	PoolVector<Face3>::Write w = faces.write();
	PoolVector<Vector3>::Read r = va.read();

	for (int i = 0; i < va_size; i++)
		w[i / 3].vertex[i % 3] = r[i];

	return faces;
}

Variant::operator Vector<Plane>() const {

	Array va = operator Array();
	Vector<Plane> planes;
	int va_size = va.size();
	if (va_size == 0)
		return planes;

	planes.resize(va_size);

	for (int i = 0; i < va_size; i++)
		planes.write[i] = va[i];

	return planes;
}

Variant::operator Vector<Variant>() const {

	Array from = operator Array();
	Vector<Variant> to;
	int len = from.size();
	to.resize(len);
	for (int i = 0; i < len; i++) {

		to.write[i] = from[i];
	}
	return to;
}

Variant::operator Vector<uint8_t>() const {

	PoolVector<uint8_t> from = operator PoolVector<uint8_t>();
	Vector<uint8_t> to;
	int len = from.size();
	to.resize(len);
	for (int i = 0; i < len; i++) {

		to.write[i] = from[i];
	}
	return to;
}
Variant::operator Vector<int>() const {

	PoolVector<int> from = operator PoolVector<int>();
	Vector<int> to;
	int len = from.size();
	to.resize(len);
	for (int i = 0; i < len; i++) {

		to.write[i] = from[i];
	}
	return to;
}
Variant::operator Vector<real_t>() const {

	PoolVector<real_t> from = operator PoolVector<real_t>();
	Vector<real_t> to;
	int len = from.size();
	to.resize(len);
	for (int i = 0; i < len; i++) {

		to.write[i] = from[i];
	}
	return to;
}

Variant::operator Vector<String>() const {

	PoolVector<String> from = operator PoolVector<String>();
	Vector<String> to;
	int len = from.size();
	to.resize(len);
	for (int i = 0; i < len; i++) {

		to.write[i] = from[i];
	}
	return to;
}
Variant::operator Vector<StringName>() const {

	PoolVector<String> from = operator PoolVector<String>();
	Vector<StringName> to;
	int len = from.size();
	to.resize(len);
	for (int i = 0; i < len; i++) {

		to.write[i] = from[i];
	}
	return to;
}

Variant::operator Vector<Vector3>() const {

	PoolVector<Vector3> from = operator PoolVector<Vector3>();
	Vector<Vector3> to;
	int len = from.size();
	if (len == 0)
		return Vector<Vector3>();
	to.resize(len);
	PoolVector<Vector3>::Read r = from.read();
	Vector3 *w = to.ptrw();
	for (int i = 0; i < len; i++) {

		w[i] = r[i];
	}
	return to;
}
Variant::operator Vector<Color>() const {

	PoolVector<Color> from = operator PoolVector<Color>();
	Vector<Color> to;
	int len = from.size();
	if (len == 0)
		return Vector<Color>();
	to.resize(len);
	PoolVector<Color>::Read r = from.read();
	Color *w = to.ptrw();
	for (int i = 0; i < len; i++) {

		w[i] = r[i];
	}
	return to;
}

Variant::operator Margin() const {

	return (Margin) operator int();
}
Variant::operator Orientation() const {

	return (Orientation) operator int();
}

Variant::operator IP_Address() const {

	if (type == POOL_REAL_ARRAY || type == POOL_INT_ARRAY || type == POOL_BYTE_ARRAY) {

		PoolVector<int> addr = operator PoolVector<int>();
		if (addr.size() == 4) {
			return IP_Address(addr.get(0), addr.get(1), addr.get(2), addr.get(3));
		}
	}

	return IP_Address(operator String());
}

Variant::Variant(bool p_bool) {

	type = BOOL;
	_data._bool = p_bool;
}

/*
Variant::Variant(long unsigned int p_long) {

	type=INT;
	_data._int=p_long;
};
*/

Variant::Variant(signed int p_int) {

	type = INT;
	_data._int = p_int;
}
Variant::Variant(unsigned int p_int) {

	type = INT;
	_data._int = p_int;
}

#ifdef NEED_LONG_INT

Variant::Variant(signed long p_int) {

	type = INT;
	_data._int = p_int;
}
Variant::Variant(unsigned long p_int) {

	type = INT;
	_data._int = p_int;
}
#endif

Variant::Variant(int64_t p_int) {

	type = INT;
	_data._int = p_int;
}

Variant::Variant(uint64_t p_int) {

	type = INT;
	_data._int = p_int;
}

Variant::Variant(signed short p_short) {

	type = INT;
	_data._int = p_short;
}
Variant::Variant(unsigned short p_short) {

	type = INT;
	_data._int = p_short;
}
Variant::Variant(signed char p_char) {

	type = INT;
	_data._int = p_char;
}
Variant::Variant(unsigned char p_char) {

	type = INT;
	_data._int = p_char;
}
Variant::Variant(float p_float) {

	type = REAL;
	_data._real = p_float;
}
Variant::Variant(double p_double) {

	type = REAL;
	_data._real = p_double;
}

Variant::Variant(const StringName &p_string) {

	type = STRING;
	memnew_placement(_data._mem, String(p_string.operator String()));
}
Variant::Variant(const String &p_string) {

	type = STRING;
	memnew_placement(_data._mem, String(p_string));
}

Variant::Variant(const char *const p_cstring) {

	type = STRING;
	memnew_placement(_data._mem, String((const char *)p_cstring));
}

Variant::Variant(const CharType *p_wstring) {

	type = STRING;
	memnew_placement(_data._mem, String(p_wstring));
}
Variant::Variant(const Vector3 &p_vector3) {

	type = VECTOR3;
	memnew_placement(_data._mem, Vector3(p_vector3));
}
Variant::Variant(const Vector2 &p_vector2) {

	type = VECTOR2;
	memnew_placement(_data._mem, Vector2(p_vector2));
}
Variant::Variant(const Rect2 &p_rect2) {

	type = RECT2;
	memnew_placement(_data._mem, Rect2(p_rect2));
}

Variant::Variant(const Plane &p_plane) {

	type = PLANE;
	memnew_placement(_data._mem, Plane(p_plane));
}
Variant::Variant(const ::AABB &p_aabb) {

	type = AABB;
	_data._aabb = memnew(::AABB(p_aabb));
}

Variant::Variant(const Basis &p_matrix) {

	type = BASIS;
	_data._basis = memnew(Basis(p_matrix));
}

Variant::Variant(const Quat &p_quat) {

	type = QUAT;
	memnew_placement(_data._mem, Quat(p_quat));
}
Variant::Variant(const Transform &p_transform) {

	type = TRANSFORM;
	_data._transform = memnew(Transform(p_transform));
}

Variant::Variant(const Transform2D &p_transform) {

	type = TRANSFORM2D;
	_data._transform2d = memnew(Transform2D(p_transform));
}
Variant::Variant(const Color &p_color) {

	type = COLOR;
	memnew_placement(_data._mem, Color(p_color));
}

Variant::Variant(const NodePath &p_node_path) {

	type = NODE_PATH;
	memnew_placement(_data._mem, NodePath(p_node_path));
}

Variant::Variant(const RefPtr &p_resource) {

	type = OBJECT;
	memnew_placement(_data._mem, ObjData);
#ifdef DEBUG_ENABLED
	_get_obj().rc = NULL;
#else
	REF *ref = reinterpret_cast<REF *>(p_resource.get_data());
	_get_obj().obj = ref->ptr();
#endif
	_get_obj().ref = p_resource;
}

Variant::Variant(const RID &p_rid) {

	type = _RID;
	memnew_placement(_data._mem, RID(p_rid));
}

Variant::Variant(const Object *p_object) {

	type = OBJECT;
	Object *obj = const_cast<Object *>(p_object);

	memnew_placement(_data._mem, ObjData);
	Reference *ref = Object::cast_to<Reference>(obj);
	if (unlikely(ref)) {
		*reinterpret_cast<Ref<Reference> *>(_get_obj().ref.get_data()) = Ref<Reference>(ref);
#ifdef DEBUG_ENABLED
		_get_obj().rc = NULL;
	} else {
		_get_obj().rc = likely(obj) ? obj->_use_rc() : NULL;
#endif
	}
#if !defined(DEBUG_ENABLED)
	_get_obj().obj = obj;
#endif
}

Variant::Variant(const Dictionary &p_dictionary) {

	type = DICTIONARY;
	memnew_placement(_data._mem, Dictionary(p_dictionary));
}

Variant::Variant(const Array &p_array) {

	type = ARRAY;
	memnew_placement(_data._mem, Array(p_array));
}

Variant::Variant(const PoolVector<Plane> &p_array) {

	type = ARRAY;

	Array *plane_array = memnew_placement(_data._mem, Array);

	plane_array->resize(p_array.size());

	for (int i = 0; i < p_array.size(); i++) {

		plane_array->operator[](i) = Variant(p_array[i]);
	}
}

Variant::Variant(const Vector<Plane> &p_array) {

	type = ARRAY;

	Array *plane_array = memnew_placement(_data._mem, Array);

	plane_array->resize(p_array.size());

	for (int i = 0; i < p_array.size(); i++) {

		plane_array->operator[](i) = Variant(p_array[i]);
	}
}

Variant::Variant(const Vector<RID> &p_array) {

	type = ARRAY;

	Array *rid_array = memnew_placement(_data._mem, Array);

	rid_array->resize(p_array.size());

	for (int i = 0; i < p_array.size(); i++) {

		rid_array->set(i, Variant(p_array[i]));
	}
}

Variant::Variant(const Vector<Vector2> &p_array) {

	type = NIL;
	PoolVector<Vector2> v;
	int len = p_array.size();
	if (len > 0) {
		v.resize(len);
		PoolVector<Vector2>::Write w = v.write();
		const Vector2 *r = p_array.ptr();

		for (int i = 0; i < len; i++)
			w[i] = r[i];
	}
	*this = v;
}

Variant::Variant(const PoolVector<uint8_t> &p_raw_array) {

	type = POOL_BYTE_ARRAY;
	memnew_placement(_data._mem, PoolVector<uint8_t>(p_raw_array));
}
Variant::Variant(const PoolVector<int> &p_int_array) {

	type = POOL_INT_ARRAY;
	memnew_placement(_data._mem, PoolVector<int>(p_int_array));
}
Variant::Variant(const PoolVector<real_t> &p_real_array) {

	type = POOL_REAL_ARRAY;
	memnew_placement(_data._mem, PoolVector<real_t>(p_real_array));
}
Variant::Variant(const PoolVector<String> &p_string_array) {

	type = POOL_STRING_ARRAY;
	memnew_placement(_data._mem, PoolVector<String>(p_string_array));
}
Variant::Variant(const PoolVector<Vector3> &p_vector3_array) {

	type = POOL_VECTOR3_ARRAY;
	memnew_placement(_data._mem, PoolVector<Vector3>(p_vector3_array));
}

Variant::Variant(const PoolVector<Vector2> &p_vector2_array) {

	type = POOL_VECTOR2_ARRAY;
	memnew_placement(_data._mem, PoolVector<Vector2>(p_vector2_array));
}
Variant::Variant(const PoolVector<Color> &p_color_array) {

	type = POOL_COLOR_ARRAY;
	memnew_placement(_data._mem, PoolVector<Color>(p_color_array));
}

Variant::Variant(const PoolVector<Face3> &p_face_array) {

	PoolVector<Vector3> vertices;
	int face_count = p_face_array.size();
	vertices.resize(face_count * 3);

	if (face_count) {
		PoolVector<Face3>::Read r = p_face_array.read();
		PoolVector<Vector3>::Write w = vertices.write();

		for (int i = 0; i < face_count; i++) {

			for (int j = 0; j < 3; j++)
				w[i * 3 + j] = r[i].vertex[j];
		}
	}

	type = NIL;

	*this = vertices;
}

/* helpers */

Variant::Variant(const Vector<Variant> &p_array) {

	type = NIL;
	Array v;
	int len = p_array.size();
	v.resize(len);
	for (int i = 0; i < len; i++)
		v.set(i, p_array[i]);
	*this = v;
}

Variant::Variant(const Vector<uint8_t> &p_array) {

	type = NIL;
	PoolVector<uint8_t> v;
	int len = p_array.size();
	v.resize(len);
	for (int i = 0; i < len; i++)
		v.set(i, p_array[i]);
	*this = v;
}

Variant::Variant(const Vector<int> &p_array) {

	type = NIL;
	PoolVector<int> v;
	int len = p_array.size();
	v.resize(len);
	for (int i = 0; i < len; i++)
		v.set(i, p_array[i]);
	*this = v;
}

Variant::Variant(const Vector<real_t> &p_array) {

	type = NIL;
	PoolVector<real_t> v;
	int len = p_array.size();
	v.resize(len);
	for (int i = 0; i < len; i++)
		v.set(i, p_array[i]);
	*this = v;
}

Variant::Variant(const Vector<String> &p_array) {

	type = NIL;
	PoolVector<String> v;
	int len = p_array.size();
	v.resize(len);
	for (int i = 0; i < len; i++)
		v.set(i, p_array[i]);
	*this = v;
}

Variant::Variant(const Vector<StringName> &p_array) {

	type = NIL;
	PoolVector<String> v;
	int len = p_array.size();
	v.resize(len);
	for (int i = 0; i < len; i++)
		v.set(i, p_array[i]);
	*this = v;
}

Variant::Variant(const Vector<Vector3> &p_array) {

	type = NIL;
	PoolVector<Vector3> v;
	int len = p_array.size();
	if (len > 0) {
		v.resize(len);
		PoolVector<Vector3>::Write w = v.write();
		const Vector3 *r = p_array.ptr();

		for (int i = 0; i < len; i++)
			w[i] = r[i];
	}
	*this = v;
}

Variant::Variant(const Vector<Color> &p_array) {

	type = NIL;
	PoolVector<Color> v;
	int len = p_array.size();
	v.resize(len);
	for (int i = 0; i < len; i++)
		v.set(i, p_array[i]);
	*this = v;
}

void Variant::operator=(const Variant &p_variant) {

	if (unlikely(this == &p_variant))
		return;

	if (unlikely(type != p_variant.type)) {
		reference(p_variant);
		return;
	}

	switch (p_variant.type) {
		case NIL: {

			// none
		} break;

		// atomic types
		case BOOL: {

			_data._bool = p_variant._data._bool;
		} break;
		case INT: {

			_data._int = p_variant._data._int;
		} break;
		case REAL: {

			_data._real = p_variant._data._real;
		} break;
		case STRING: {

			*reinterpret_cast<String *>(_data._mem) = *reinterpret_cast<const String *>(p_variant._data._mem);
		} break;

		// math types
		case VECTOR2: {

			*reinterpret_cast<Vector2 *>(_data._mem) = *reinterpret_cast<const Vector2 *>(p_variant._data._mem);
		} break;
		case RECT2: {

			*reinterpret_cast<Rect2 *>(_data._mem) = *reinterpret_cast<const Rect2 *>(p_variant._data._mem);
		} break;
		case TRANSFORM2D: {

			*_data._transform2d = *(p_variant._data._transform2d);
		} break;
		case VECTOR3: {

			*reinterpret_cast<Vector3 *>(_data._mem) = *reinterpret_cast<const Vector3 *>(p_variant._data._mem);
		} break;
		case PLANE: {

			*reinterpret_cast<Plane *>(_data._mem) = *reinterpret_cast<const Plane *>(p_variant._data._mem);
		} break;

		case AABB: {

			*_data._aabb = *(p_variant._data._aabb);
		} break;
		case QUAT: {

			*reinterpret_cast<Quat *>(_data._mem) = *reinterpret_cast<const Quat *>(p_variant._data._mem);
		} break;
		case BASIS: {

			*_data._basis = *(p_variant._data._basis);
		} break;
		case TRANSFORM: {

			*_data._transform = *(p_variant._data._transform);
		} break;

		// misc types
		case COLOR: {

			*reinterpret_cast<Color *>(_data._mem) = *reinterpret_cast<const Color *>(p_variant._data._mem);
		} break;
		case _RID: {

			*reinterpret_cast<RID *>(_data._mem) = *reinterpret_cast<const RID *>(p_variant._data._mem);
		} break;
		case OBJECT: {

#ifdef DEBUG_ENABLED
			if (likely(_get_obj().rc)) {
				if (unlikely(_get_obj().rc->decrement())) {
					memdelete(_get_obj().rc);
				}
			}
#endif
			*reinterpret_cast<ObjData *>(_data._mem) = p_variant._get_obj();
#ifdef DEBUG_ENABLED
			if (likely(_get_obj().rc)) {
				_get_obj().rc->increment();
			}
#endif
		} break;
		case NODE_PATH: {

			*reinterpret_cast<NodePath *>(_data._mem) = *reinterpret_cast<const NodePath *>(p_variant._data._mem);
		} break;
		case DICTIONARY: {

			*reinterpret_cast<Dictionary *>(_data._mem) = *reinterpret_cast<const Dictionary *>(p_variant._data._mem);
		} break;
		case ARRAY: {

			*reinterpret_cast<Array *>(_data._mem) = *reinterpret_cast<const Array *>(p_variant._data._mem);
		} break;

		// arrays
		case POOL_BYTE_ARRAY: {

			*reinterpret_cast<PoolVector<uint8_t> *>(_data._mem) = *reinterpret_cast<const PoolVector<uint8_t> *>(p_variant._data._mem);
		} break;
		case POOL_INT_ARRAY: {

			*reinterpret_cast<PoolVector<int> *>(_data._mem) = *reinterpret_cast<const PoolVector<int> *>(p_variant._data._mem);
		} break;
		case POOL_REAL_ARRAY: {

			*reinterpret_cast<PoolVector<real_t> *>(_data._mem) = *reinterpret_cast<const PoolVector<real_t> *>(p_variant._data._mem);
		} break;
		case POOL_STRING_ARRAY: {

			*reinterpret_cast<PoolVector<String> *>(_data._mem) = *reinterpret_cast<const PoolVector<String> *>(p_variant._data._mem);
		} break;
		case POOL_VECTOR2_ARRAY: {

			*reinterpret_cast<PoolVector<Vector2> *>(_data._mem) = *reinterpret_cast<const PoolVector<Vector2> *>(p_variant._data._mem);
		} break;
		case POOL_VECTOR3_ARRAY: {

			*reinterpret_cast<PoolVector<Vector3> *>(_data._mem) = *reinterpret_cast<const PoolVector<Vector3> *>(p_variant._data._mem);
		} break;
		case POOL_COLOR_ARRAY: {

			*reinterpret_cast<PoolVector<Color> *>(_data._mem) = *reinterpret_cast<const PoolVector<Color> *>(p_variant._data._mem);
		} break;
		default: {
		}
	}
}

Variant::Variant(const IP_Address &p_address) {

	type = STRING;
	memnew_placement(_data._mem, String(p_address));
}

Variant::Variant(const Variant &p_variant) {

	type = NIL;
	reference(p_variant);
}

/*
Variant::~Variant() {

	clear();
}*/

uint32_t Variant::hash() const {

	switch (type) {
		case NIL: {

			return 0;
		} break;
		case BOOL: {

			return _data._bool ? 1 : 0;
		} break;
		case INT: {

			return _data._int;
		} break;
		case REAL: {

			return hash_djb2_one_float(_data._real);
		} break;
		case STRING: {

			return reinterpret_cast<const String *>(_data._mem)->hash();
		} break;

		// math types
		case VECTOR2: {

			uint32_t hash = hash_djb2_one_float(reinterpret_cast<const Vector2 *>(_data._mem)->x);
			return hash_djb2_one_float(reinterpret_cast<const Vector2 *>(_data._mem)->y, hash);
		} break;
		case RECT2: {

			uint32_t hash = hash_djb2_one_float(reinterpret_cast<const Rect2 *>(_data._mem)->position.x);
			hash = hash_djb2_one_float(reinterpret_cast<const Rect2 *>(_data._mem)->position.y, hash);
			hash = hash_djb2_one_float(reinterpret_cast<const Rect2 *>(_data._mem)->size.x, hash);
			return hash_djb2_one_float(reinterpret_cast<const Rect2 *>(_data._mem)->size.y, hash);
		} break;
		case TRANSFORM2D: {

			uint32_t hash = 5831;
			for (int i = 0; i < 3; i++) {

				for (int j = 0; j < 2; j++) {
					hash = hash_djb2_one_float(_data._transform2d->elements[i][j], hash);
				}
			}

			return hash;
		} break;
		case VECTOR3: {

			uint32_t hash = hash_djb2_one_float(reinterpret_cast<const Vector3 *>(_data._mem)->x);
			hash = hash_djb2_one_float(reinterpret_cast<const Vector3 *>(_data._mem)->y, hash);
			return hash_djb2_one_float(reinterpret_cast<const Vector3 *>(_data._mem)->z, hash);
		} break;
		case PLANE: {

			uint32_t hash = hash_djb2_one_float(reinterpret_cast<const Plane *>(_data._mem)->normal.x);
			hash = hash_djb2_one_float(reinterpret_cast<const Plane *>(_data._mem)->normal.y, hash);
			hash = hash_djb2_one_float(reinterpret_cast<const Plane *>(_data._mem)->normal.z, hash);
			return hash_djb2_one_float(reinterpret_cast<const Plane *>(_data._mem)->d, hash);

		} break;
		/*
			case QUAT: {


			} break;*/
		case AABB: {

			uint32_t hash = 5831;
			for (int i = 0; i < 3; i++) {

				hash = hash_djb2_one_float(_data._aabb->position[i], hash);
				hash = hash_djb2_one_float(_data._aabb->size[i], hash);
			}

			return hash;

		} break;
		case QUAT: {

			uint32_t hash = hash_djb2_one_float(reinterpret_cast<const Quat *>(_data._mem)->x);
			hash = hash_djb2_one_float(reinterpret_cast<const Quat *>(_data._mem)->y, hash);
			hash = hash_djb2_one_float(reinterpret_cast<const Quat *>(_data._mem)->z, hash);
			return hash_djb2_one_float(reinterpret_cast<const Quat *>(_data._mem)->w, hash);

		} break;
		case BASIS: {

			uint32_t hash = 5831;
			for (int i = 0; i < 3; i++) {

				for (int j = 0; j < 3; j++) {
					hash = hash_djb2_one_float(_data._basis->elements[i][j], hash);
				}
			}

			return hash;

		} break;
		case TRANSFORM: {

			uint32_t hash = 5831;
			for (int i = 0; i < 3; i++) {

				for (int j = 0; j < 3; j++) {
					hash = hash_djb2_one_float(_data._transform->basis.elements[i][j], hash);
				}
				hash = hash_djb2_one_float(_data._transform->origin[i], hash);
			}

			return hash;

		} break;

		// misc types
		case COLOR: {

			uint32_t hash = hash_djb2_one_float(reinterpret_cast<const Color *>(_data._mem)->r);
			hash = hash_djb2_one_float(reinterpret_cast<const Color *>(_data._mem)->g, hash);
			hash = hash_djb2_one_float(reinterpret_cast<const Color *>(_data._mem)->b, hash);
			return hash_djb2_one_float(reinterpret_cast<const Color *>(_data._mem)->a, hash);

		} break;
		case _RID: {

			return hash_djb2_one_64(reinterpret_cast<const RID *>(_data._mem)->get_id());
		} break;
		case OBJECT: {

			return hash_djb2_one_64(make_uint64_t(_UNSAFE_OBJ_PROXY_PTR(*this)));
		} break;
		case NODE_PATH: {

			return reinterpret_cast<const NodePath *>(_data._mem)->hash();
		} break;
		case DICTIONARY: {

			return reinterpret_cast<const Dictionary *>(_data._mem)->hash();

		} break;
		case ARRAY: {

			const Array &arr = *reinterpret_cast<const Array *>(_data._mem);
			return arr.hash();

		} break;
		case POOL_BYTE_ARRAY: {

			const PoolVector<uint8_t> &arr = *reinterpret_cast<const PoolVector<uint8_t> *>(_data._mem);
			int len = arr.size();
			if (likely(len)) {
				PoolVector<uint8_t>::Read r = arr.read();
				return hash_djb2_buffer((uint8_t *)&r[0], len);
			} else {
				return hash_djb2_one_64(0);
			}

		} break;
		case POOL_INT_ARRAY: {

			const PoolVector<int> &arr = *reinterpret_cast<const PoolVector<int> *>(_data._mem);
			int len = arr.size();
			if (likely(len)) {
				PoolVector<int>::Read r = arr.read();
				return hash_djb2_buffer((uint8_t *)&r[0], len * sizeof(int));
			} else {
				return hash_djb2_one_64(0);
			}

		} break;
		case POOL_REAL_ARRAY: {

			const PoolVector<real_t> &arr = *reinterpret_cast<const PoolVector<real_t> *>(_data._mem);
			int len = arr.size();

			if (likely(len)) {
				PoolVector<real_t>::Read r = arr.read();
				return hash_djb2_buffer((uint8_t *)&r[0], len * sizeof(real_t));
			} else {
				return hash_djb2_one_float(0.0);
			}

		} break;
		case POOL_STRING_ARRAY: {

			uint32_t hash = 5831;
			const PoolVector<String> &arr = *reinterpret_cast<const PoolVector<String> *>(_data._mem);
			int len = arr.size();

			if (likely(len)) {
				PoolVector<String>::Read r = arr.read();

				for (int i = 0; i < len; i++) {
					hash = hash_djb2_one_32(r[i].hash(), hash);
				}
			}

			return hash;
		} break;
		case POOL_VECTOR2_ARRAY: {

			uint32_t hash = 5831;
			const PoolVector<Vector2> &arr = *reinterpret_cast<const PoolVector<Vector2> *>(_data._mem);
			int len = arr.size();

			if (likely(len)) {
				PoolVector<Vector2>::Read r = arr.read();

				for (int i = 0; i < len; i++) {
					hash = hash_djb2_one_float(r[i].x, hash);
					hash = hash_djb2_one_float(r[i].y, hash);
				}
			}

			return hash;
		} break;
		case POOL_VECTOR3_ARRAY: {

			uint32_t hash = 5831;
			const PoolVector<Vector3> &arr = *reinterpret_cast<const PoolVector<Vector3> *>(_data._mem);
			int len = arr.size();

			if (likely(len)) {
				PoolVector<Vector3>::Read r = arr.read();

				for (int i = 0; i < len; i++) {
					hash = hash_djb2_one_float(r[i].x, hash);
					hash = hash_djb2_one_float(r[i].y, hash);
					hash = hash_djb2_one_float(r[i].z, hash);
				}
			}

			return hash;
		} break;
		case POOL_COLOR_ARRAY: {

			uint32_t hash = 5831;
			const PoolVector<Color> &arr = *reinterpret_cast<const PoolVector<Color> *>(_data._mem);
			int len = arr.size();

			if (likely(len)) {
				PoolVector<Color>::Read r = arr.read();

				for (int i = 0; i < len; i++) {
					hash = hash_djb2_one_float(r[i].r, hash);
					hash = hash_djb2_one_float(r[i].g, hash);
					hash = hash_djb2_one_float(r[i].b, hash);
					hash = hash_djb2_one_float(r[i].a, hash);
				}
			}

			return hash;
		} break;
		default: {
		}
	}

	return 0;
}

#define hash_compare_scalar(p_lhs, p_rhs) \
	((p_lhs) == (p_rhs)) || (Math::is_nan(p_lhs) && Math::is_nan(p_rhs))

#define hash_compare_vector2(p_lhs, p_rhs)         \
	(hash_compare_scalar((p_lhs).x, (p_rhs).x)) && \
			(hash_compare_scalar((p_lhs).y, (p_rhs).y))

#define hash_compare_vector3(p_lhs, p_rhs)                 \
	(hash_compare_scalar((p_lhs).x, (p_rhs).x)) &&         \
			(hash_compare_scalar((p_lhs).y, (p_rhs).y)) && \
			(hash_compare_scalar((p_lhs).z, (p_rhs).z))

#define hash_compare_quat(p_lhs, p_rhs)                    \
	(hash_compare_scalar((p_lhs).x, (p_rhs).x)) &&         \
			(hash_compare_scalar((p_lhs).y, (p_rhs).y)) && \
			(hash_compare_scalar((p_lhs).z, (p_rhs).z)) && \
			(hash_compare_scalar((p_lhs).w, (p_rhs).w))

#define hash_compare_color(p_lhs, p_rhs)                   \
	(hash_compare_scalar((p_lhs).r, (p_rhs).r)) &&         \
			(hash_compare_scalar((p_lhs).g, (p_rhs).g)) && \
			(hash_compare_scalar((p_lhs).b, (p_rhs).b)) && \
			(hash_compare_scalar((p_lhs).a, (p_rhs).a))

#define hash_compare_pool_array(p_lhs, p_rhs, p_type, p_compare_func)                   \
	const PoolVector<p_type> &l = *reinterpret_cast<const PoolVector<p_type> *>(p_lhs); \
	const PoolVector<p_type> &r = *reinterpret_cast<const PoolVector<p_type> *>(p_rhs); \
                                                                                        \
	if (l.size() != r.size())                                                           \
		return false;                                                                   \
                                                                                        \
	PoolVector<p_type>::Read lr = l.read();                                             \
	PoolVector<p_type>::Read rr = r.read();                                             \
                                                                                        \
	for (int i = 0; i < l.size(); ++i) {                                                \
		if (!p_compare_func((lr[i]), (rr[i])))                                          \
			return false;                                                               \
	}                                                                                   \
                                                                                        \
	return true

bool Variant::hash_compare(const Variant &p_variant) const {
	if (type != p_variant.type)
		return false;

	switch (type) {
		case REAL: {
			return hash_compare_scalar(_data._real, p_variant._data._real);
		} break;

		case VECTOR2: {
			const Vector2 *l = reinterpret_cast<const Vector2 *>(_data._mem);
			const Vector2 *r = reinterpret_cast<const Vector2 *>(p_variant._data._mem);

			return hash_compare_vector2(*l, *r);
		} break;

		case RECT2: {
			const Rect2 *l = reinterpret_cast<const Rect2 *>(_data._mem);
			const Rect2 *r = reinterpret_cast<const Rect2 *>(p_variant._data._mem);

			return (hash_compare_vector2(l->position, r->position)) &&
				   (hash_compare_vector2(l->size, r->size));
		} break;

		case TRANSFORM2D: {
			Transform2D *l = _data._transform2d;
			Transform2D *r = p_variant._data._transform2d;

			for (int i = 0; i < 3; i++) {
				if (!(hash_compare_vector2(l->elements[i], r->elements[i])))
					return false;
			}

			return true;
		} break;

		case VECTOR3: {
			const Vector3 *l = reinterpret_cast<const Vector3 *>(_data._mem);
			const Vector3 *r = reinterpret_cast<const Vector3 *>(p_variant._data._mem);

			return hash_compare_vector3(*l, *r);
		} break;

		case PLANE: {
			const Plane *l = reinterpret_cast<const Plane *>(_data._mem);
			const Plane *r = reinterpret_cast<const Plane *>(p_variant._data._mem);

			return (hash_compare_vector3(l->normal, r->normal)) &&
				   (hash_compare_scalar(l->d, r->d));
		} break;

		case AABB: {
			const ::AABB *l = _data._aabb;
			const ::AABB *r = p_variant._data._aabb;

			return (hash_compare_vector3(l->position, r->position) &&
					(hash_compare_vector3(l->size, r->size)));

		} break;

		case QUAT: {
			const Quat *l = reinterpret_cast<const Quat *>(_data._mem);
			const Quat *r = reinterpret_cast<const Quat *>(p_variant._data._mem);

			return hash_compare_quat(*l, *r);
		} break;

		case BASIS: {
			const Basis *l = _data._basis;
			const Basis *r = p_variant._data._basis;

			for (int i = 0; i < 3; i++) {
				if (!(hash_compare_vector3(l->elements[i], r->elements[i])))
					return false;
			}

			return true;
		} break;

		case TRANSFORM: {
			const Transform *l = _data._transform;
			const Transform *r = p_variant._data._transform;

			for (int i = 0; i < 3; i++) {
				if (!(hash_compare_vector3(l->basis.elements[i], r->basis.elements[i])))
					return false;
			}

			return hash_compare_vector3(l->origin, r->origin);
		} break;

		case COLOR: {
			const Color *l = reinterpret_cast<const Color *>(_data._mem);
			const Color *r = reinterpret_cast<const Color *>(p_variant._data._mem);

			return hash_compare_color(*l, *r);
		} break;

		case ARRAY: {
			const Array &l = *(reinterpret_cast<const Array *>(_data._mem));
			const Array &r = *(reinterpret_cast<const Array *>(p_variant._data._mem));

			if (l.size() != r.size())
				return false;

			for (int i = 0; i < l.size(); ++i) {
				if (!l[i].hash_compare(r[i]))
					return false;
			}

			return true;
		} break;

		case POOL_REAL_ARRAY: {
			hash_compare_pool_array(_data._mem, p_variant._data._mem, real_t, hash_compare_scalar);
		} break;

		case POOL_VECTOR2_ARRAY: {
			hash_compare_pool_array(_data._mem, p_variant._data._mem, Vector2, hash_compare_vector2);
		} break;

		case POOL_VECTOR3_ARRAY: {
			hash_compare_pool_array(_data._mem, p_variant._data._mem, Vector3, hash_compare_vector3);
		} break;

		case POOL_COLOR_ARRAY: {
			hash_compare_pool_array(_data._mem, p_variant._data._mem, Color, hash_compare_color);
		} break;

		default:
			bool v;
			Variant r;
			evaluate(OP_EQUAL, *this, p_variant, r, v);
			return r;
	}

	return false;
}

bool Variant::is_ref() const {

	return type == OBJECT && !_get_obj().ref.is_null();
}

Vector<Variant> varray() {

	return Vector<Variant>();
}

Vector<Variant> varray(const Variant &p_arg1) {

	Vector<Variant> v;
	v.push_back(p_arg1);
	return v;
}
Vector<Variant> varray(const Variant &p_arg1, const Variant &p_arg2) {

	Vector<Variant> v;
	v.push_back(p_arg1);
	v.push_back(p_arg2);
	return v;
}
Vector<Variant> varray(const Variant &p_arg1, const Variant &p_arg2, const Variant &p_arg3) {

	Vector<Variant> v;
	v.push_back(p_arg1);
	v.push_back(p_arg2);
	v.push_back(p_arg3);
	return v;
}
Vector<Variant> varray(const Variant &p_arg1, const Variant &p_arg2, const Variant &p_arg3, const Variant &p_arg4) {

	Vector<Variant> v;
	v.push_back(p_arg1);
	v.push_back(p_arg2);
	v.push_back(p_arg3);
	v.push_back(p_arg4);
	return v;
}

Vector<Variant> varray(const Variant &p_arg1, const Variant &p_arg2, const Variant &p_arg3, const Variant &p_arg4, const Variant &p_arg5) {

	Vector<Variant> v;
	v.push_back(p_arg1);
	v.push_back(p_arg2);
	v.push_back(p_arg3);
	v.push_back(p_arg4);
	v.push_back(p_arg5);
	return v;
}

void Variant::static_assign(const Variant &p_variant) {
}

bool Variant::is_shared() const {

	switch (type) {

		case OBJECT: return true;
		case ARRAY: return true;
		case DICTIONARY: return true;
		default: {
		}
	}

	return false;
}

Variant Variant::call(const StringName &p_method, VARIANT_ARG_DECLARE) {
	VARIANT_ARGPTRS;
	int argc = 0;
	for (int i = 0; i < VARIANT_ARG_MAX; i++) {
		if (argptr[i]->get_type() == Variant::NIL)
			break;
		argc++;
	}

	CallError error;

	Variant ret = call(p_method, argptr, argc, error);

	switch (error.error) {

		case CallError::CALL_ERROR_INVALID_ARGUMENT: {

			String err = "Invalid type for argument #" + itos(error.argument) + ", expected '" + Variant::get_type_name(error.expected) + "'.";
			ERR_PRINT(err.utf8().get_data());

		} break;
		case CallError::CALL_ERROR_INVALID_METHOD: {

			String err = "Invalid method '" + p_method + "' for type '" + Variant::get_type_name(type) + "'.";
			ERR_PRINT(err.utf8().get_data());
		} break;
		case CallError::CALL_ERROR_TOO_MANY_ARGUMENTS: {

			String err = "Too many arguments for method '" + p_method + "'";
			ERR_PRINT(err.utf8().get_data());
		} break;
		default: {
		}
	}

	return ret;
}

void Variant::construct_from_string(const String &p_string, Variant &r_value, ObjectConstruct p_obj_construct, void *p_construct_ud) {

	r_value = Variant();
}

String Variant::get_construct_string() const {

	String vars;
	VariantWriter::write_to_string(*this, vars);

	return vars;
}

String Variant::get_call_error_text(Object *p_base, const StringName &p_method, const Variant **p_argptrs, int p_argcount, const Variant::CallError &ce) {

	String err_text;

	if (ce.error == Variant::CallError::CALL_ERROR_INVALID_ARGUMENT) {
		int errorarg = ce.argument;
		if (p_argptrs) {
			err_text = "Cannot convert argument " + itos(errorarg + 1) + " from " + Variant::get_type_name(p_argptrs[errorarg]->get_type()) + " to " + Variant::get_type_name(ce.expected) + ".";
		} else {
			err_text = "Cannot convert argument " + itos(errorarg + 1) + " from [missing argptr, type unknown] to " + Variant::get_type_name(ce.expected) + ".";
		}
	} else if (ce.error == Variant::CallError::CALL_ERROR_TOO_MANY_ARGUMENTS) {
		err_text = "Method expected " + itos(ce.argument) + " arguments, but called with " + itos(p_argcount) + ".";
	} else if (ce.error == Variant::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS) {
		err_text = "Method expected " + itos(ce.argument) + " arguments, but called with " + itos(p_argcount) + ".";
	} else if (ce.error == Variant::CallError::CALL_ERROR_INVALID_METHOD) {
		err_text = "Method not found.";
	} else if (ce.error == Variant::CallError::CALL_ERROR_INSTANCE_IS_NULL) {
		err_text = "Instance is null";
	} else if (ce.error == Variant::CallError::CALL_OK) {
		return "Call OK";
	}

	String class_name = p_base->get_class();
	Ref<Script> script = p_base->get_script();
	if (script.is_valid() && script->get_path().is_resource_file()) {

		class_name += "(" + script->get_path().get_file() + ")";
	}
	return "'" + class_name + "::" + String(p_method) + "': " + err_text;
}

String vformat(const String &p_text, const Variant &p1, const Variant &p2, const Variant &p3, const Variant &p4, const Variant &p5) {

	Array args;
	if (p1.get_type() != Variant::NIL) {

		args.push_back(p1);

		if (p2.get_type() != Variant::NIL) {

			args.push_back(p2);

			if (p3.get_type() != Variant::NIL) {

				args.push_back(p3);

				if (p4.get_type() != Variant::NIL) {

					args.push_back(p4);

					if (p5.get_type() != Variant::NIL) {

						args.push_back(p5);
					}
				}
			}
		}
	}

	bool error = false;
	String fmt = p_text.sprintf(args, &error);

	ERR_FAIL_COND_V_MSG(error, String(), fmt);

	return fmt;
}
