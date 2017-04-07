/*************************************************************************/
/*  variant.h                                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
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
#ifndef VARIANT_H
#define VARIANT_H

/**
	@author Juan Linietsky <reduzio@gmail.com>
*/

#include "array.h"
#include "color.h"
#include "dictionary.h"
#include "dvector.h"
#include "face3.h"
#include "image.h"
#include "io/ip_address.h"
#include "math_2d.h"
#include "matrix3.h"
#include "os/input_event.h"
#include "os/power.h"
#include "path_db.h"
#include "plane.h"
#include "quat.h"
#include "rect3.h"
#include "ref_ptr.h"
#include "rid.h"
#include "simple_type.h"
#include "transform.h"
#include "ustring.h"
#include "vector3.h"

class RefPtr;
class Object;
class Node; // helper
class Control; // helper

struct PropertyInfo;
struct MethodInfo;

typedef PoolVector<uint8_t> PoolByteArray;
typedef PoolVector<int> PoolIntArray;
typedef PoolVector<real_t> PoolRealArray;
typedef PoolVector<String> PoolStringArray;
typedef PoolVector<Vector2> PoolVector2Array;
typedef PoolVector<Vector3> PoolVector3Array;
typedef PoolVector<Color> PoolColorArray;

class Variant {
public:
	enum Type {

		NIL,

		// atomic types
		BOOL,
		INT,
		REAL,
		STRING,

		// math types

		VECTOR2, // 5
		RECT2,
		VECTOR3,
		TRANSFORM2D,
		PLANE,
		QUAT, // 10
		RECT3, //sorry naming convention fail :( not like it's used often
		BASIS,
		TRANSFORM,

		// misc types
		COLOR,
		IMAGE, // 15
		NODE_PATH,
		_RID,
		OBJECT,
		INPUT_EVENT,
		DICTIONARY, // 20
		ARRAY,

		// arrays
		POOL_BYTE_ARRAY,
		POOL_INT_ARRAY,
		POOL_REAL_ARRAY,
		POOL_STRING_ARRAY, // 25
		POOL_VECTOR2_ARRAY,
		POOL_VECTOR3_ARRAY,
		POOL_COLOR_ARRAY,

		VARIANT_MAX

	};

private:
	friend class _VariantCall;
	// Variant takes 20 bytes when real_t is float, and 36 if double
	// it only allocates extra memory for aabb/matrix.

	Type type;

	struct ObjData {

		Object *obj;
		RefPtr ref;
	};

	_FORCE_INLINE_ ObjData &_get_obj();
	_FORCE_INLINE_ const ObjData &_get_obj() const;

	union {

		bool _bool;
		int64_t _int;
		double _real;
		Transform2D *_transform2d;
		Rect3 *_rect3;
		Basis *_basis;
		Transform *_transform;
		RefPtr *_resource;
		InputEvent *_input_event;
		Image *_image;
		void *_ptr; //generic pointer
		uint8_t _mem[sizeof(ObjData) > (sizeof(real_t) * 4) ? sizeof(ObjData) : (sizeof(real_t) * 4)];
	} _data;

	void reference(const Variant &p_variant);
	void clear();

public:
	_FORCE_INLINE_ Type get_type() const { return type; }
	static String get_type_name(Variant::Type p_type);
	static bool can_convert(Type p_type_from, Type p_type_to);
	static bool can_convert_strict(Type p_type_from, Type p_type_to);

	template <class T>
	static Type get_type_for() {

		GetSimpleType<T> t;
		Variant v(t.type);
		Type r = v.get_type();
		return r;
	}

	bool is_ref() const;
	_FORCE_INLINE_ bool is_num() const { return type == INT || type == REAL; };
	_FORCE_INLINE_ bool is_array() const { return type >= ARRAY; };
	bool is_shared() const;
	bool is_zero() const;
	bool is_one() const;

	operator bool() const;
	operator signed int() const;
	operator unsigned int() const; // this is the real one
	operator signed short() const;
	operator unsigned short() const;
	operator signed char() const;
	operator unsigned char() const;
	//operator long unsigned int() const;
	operator int64_t() const;
	operator uint64_t() const;
#ifdef NEED_LONG_INT
	operator signed long() const;
	operator unsigned long() const;
#endif

	operator CharType() const;
	operator float() const;
	operator double() const;
	operator String() const;
	operator StringName() const;
	operator Vector2() const;
	operator Rect2() const;
	operator Vector3() const;
	operator Plane() const;
	operator Rect3() const;
	operator Quat() const;
	operator Basis() const;
	operator Transform() const;
	operator Transform2D() const;

	operator Color() const;
	operator Image() const;
	operator NodePath() const;
	operator RefPtr() const;
	operator RID() const;
	operator InputEvent() const;
	operator Object *() const;
	operator Node *() const;
	operator Control *() const;

	operator Dictionary() const;
	operator Array() const;

	operator PoolVector<uint8_t>() const;
	operator PoolVector<int>() const;
	operator PoolVector<real_t>() const;
	operator PoolVector<String>() const;
	operator PoolVector<Vector3>() const;
	operator PoolVector<Color>() const;
	operator PoolVector<Plane>() const;
	operator PoolVector<Face3>() const;

	operator Vector<Variant>() const;
	operator Vector<uint8_t>() const;
	operator Vector<int>() const;
	operator Vector<real_t>() const;
	operator Vector<String>() const;
	operator Vector<Vector3>() const;
	operator Vector<Color>() const;
	operator Vector<RID>() const;
	operator Vector<Vector2>() const;
	operator PoolVector<Vector2>() const;
	operator Vector<Plane>() const;

	// some core type enums to convert to
	operator Margin() const;
	operator Orientation() const;

	operator IP_Address() const;

	Variant(bool p_bool);
	Variant(signed int p_int); // real one
	Variant(unsigned int p_int);
#ifdef NEED_LONG_INT
	Variant(signed long p_long); // real one
	Variant(unsigned long p_long);
//Variant(long unsigned int p_long);
#endif
	Variant(signed short p_short); // real one
	Variant(unsigned short p_short);
	Variant(signed char p_char); // real one
	Variant(unsigned char p_char);
	Variant(int64_t p_char); // real one
	Variant(uint64_t p_char);
	Variant(float p_float);
	Variant(double p_double);
	Variant(const String &p_string);
	Variant(const StringName &p_string);
	Variant(const char *const p_cstring);
	Variant(const CharType *p_wstring);
	Variant(const Vector2 &p_vector2);
	Variant(const Rect2 &p_rect2);
	Variant(const Vector3 &p_vector3);
	Variant(const Plane &p_plane);
	Variant(const Rect3 &p_aabb);
	Variant(const Quat &p_quat);
	Variant(const Basis &p_transform);
	Variant(const Transform2D &p_transform);
	Variant(const Transform &p_transform);
	Variant(const Color &p_color);
	Variant(const Image &p_image);
	Variant(const NodePath &p_path);
	Variant(const RefPtr &p_resource);
	Variant(const RID &p_rid);
	Variant(const Object *p_object);
	Variant(const InputEvent &p_input_event);
	Variant(const Dictionary &p_dictionary);

	Variant(const Array &p_array);
	Variant(const PoolVector<Plane> &p_array); // helper
	Variant(const PoolVector<uint8_t> &p_raw_array);
	Variant(const PoolVector<int> &p_int_array);
	Variant(const PoolVector<real_t> &p_real_array);
	Variant(const PoolVector<String> &p_string_array);
	Variant(const PoolVector<Vector3> &p_vector3_array);
	Variant(const PoolVector<Color> &p_color_array);
	Variant(const PoolVector<Face3> &p_face_array);

	Variant(const Vector<Variant> &p_array);
	Variant(const Vector<uint8_t> &p_raw_array);
	Variant(const Vector<int> &p_int_array);
	Variant(const Vector<real_t> &p_real_array);
	Variant(const Vector<String> &p_string_array);
	Variant(const Vector<Vector3> &p_vector3_array);
	Variant(const Vector<Color> &p_color_array);
	Variant(const Vector<Plane> &p_array); // helper
	Variant(const Vector<RID> &p_array); // helper
	Variant(const Vector<Vector2> &p_array); // helper
	Variant(const PoolVector<Vector2> &p_array); // helper

	Variant(const IP_Address &p_address);

	enum Operator {

		//comparation
		OP_EQUAL,
		OP_NOT_EQUAL,
		OP_LESS,
		OP_LESS_EQUAL,
		OP_GREATER,
		OP_GREATER_EQUAL,
		//mathematic
		OP_ADD,
		OP_SUBSTRACT,
		OP_MULTIPLY,
		OP_DIVIDE,
		OP_NEGATE,
		OP_POSITIVE,
		OP_MODULE,
		OP_STRING_CONCAT,
		//bitwise
		OP_SHIFT_LEFT,
		OP_SHIFT_RIGHT,
		OP_BIT_AND,
		OP_BIT_OR,
		OP_BIT_XOR,
		OP_BIT_NEGATE,
		//logic
		OP_AND,
		OP_OR,
		OP_XOR,
		OP_NOT,
		//containment
		OP_IN,
		OP_MAX

	};

	static String get_operator_name(Operator p_op);
	static void evaluate(const Operator &p_op, const Variant &p_a, const Variant &p_b, Variant &r_ret, bool &r_valid);
	static _FORCE_INLINE_ Variant evaluate(const Operator &p_op, const Variant &p_a, const Variant &p_b) {

		bool valid = true;
		Variant res;
		evaluate(p_op, p_a, p_b, res, valid);
		return res;
	}

	void zero();
	static void blend(const Variant &a, const Variant &b, float c, Variant &r_dst);
	static void interpolate(const Variant &a, const Variant &b, float c, Variant &r_dst);

	struct CallError {
		enum Error {
			CALL_OK,
			CALL_ERROR_INVALID_METHOD,
			CALL_ERROR_INVALID_ARGUMENT,
			CALL_ERROR_TOO_MANY_ARGUMENTS,
			CALL_ERROR_TOO_FEW_ARGUMENTS,
			CALL_ERROR_INSTANCE_IS_NULL,
		};
		Error error;
		int argument;
		Type expected;
	};

	void call_ptr(const StringName &p_method, const Variant **p_args, int p_argcount, Variant *r_ret, CallError &r_error);
	Variant call(const StringName &p_method, const Variant **p_args, int p_argcount, CallError &r_error);
	Variant call(const StringName &p_method, const Variant &p_arg1 = Variant(), const Variant &p_arg2 = Variant(), const Variant &p_arg3 = Variant(), const Variant &p_arg4 = Variant(), const Variant &p_arg5 = Variant());

	static String get_call_error_text(Object *p_base, const StringName &p_method, const Variant **p_argptrs, int p_argcount, const Variant::CallError &ce);

	static Variant construct(const Variant::Type, const Variant **p_args, int p_argcount, CallError &r_error, bool p_strict = true);

	void get_method_list(List<MethodInfo> *p_list) const;
	bool has_method(const StringName &p_method) const;
	static Vector<Variant::Type> get_method_argument_types(Variant::Type p_type, const StringName &p_method);
	static Vector<Variant> get_method_default_arguments(Variant::Type p_type, const StringName &p_method);
	static Variant::Type get_method_return_type(Variant::Type p_type, const StringName &p_method, bool *r_has_return = NULL);
	static Vector<StringName> get_method_argument_names(Variant::Type p_type, const StringName &p_method);

	void set_named(const StringName &p_index, const Variant &p_value, bool *r_valid = NULL);
	Variant get_named(const StringName &p_index, bool *r_valid = NULL) const;

	void set(const Variant &p_index, const Variant &p_value, bool *r_valid = NULL);
	Variant get(const Variant &p_index, bool *r_valid = NULL) const;
	bool in(const Variant &p_index, bool *r_valid = NULL) const;

	bool iter_init(Variant &r_iter, bool &r_valid) const;
	bool iter_next(Variant &r_iter, bool &r_valid) const;
	Variant iter_get(const Variant &r_iter, bool &r_valid) const;

	void get_property_list(List<PropertyInfo> *p_list) const;

	//argsVariant call()

	bool operator==(const Variant &p_variant) const;
	bool operator!=(const Variant &p_variant) const;
	bool operator<(const Variant &p_variant) const;
	uint32_t hash() const;

	bool hash_compare(const Variant &p_variant) const;
	bool booleanize(bool &valid) const;

	void static_assign(const Variant &p_variant);
	static void get_constructor_list(Variant::Type p_type, List<MethodInfo> *p_list);
	static void get_numeric_constants_for_type(Variant::Type p_type, List<StringName> *p_constants);
	static bool has_numeric_constant(Variant::Type p_type, const StringName &p_value);
	static int get_numeric_constant_value(Variant::Type p_type, const StringName &p_value, bool *r_valid = NULL);

	typedef String (*ObjectDeConstruct)(const Variant &p_object, void *ud);
	typedef void (*ObjectConstruct)(const String &p_text, void *ud, Variant &r_value);

	String get_construct_string() const;
	static void construct_from_string(const String &p_string, Variant &r_value, ObjectConstruct p_obj_construct = NULL, void *p_construct_ud = NULL);

	void operator=(const Variant &p_variant); // only this is enough for all the other types
	Variant(const Variant &p_variant);
	_FORCE_INLINE_ Variant() { type = NIL; }
	_FORCE_INLINE_ ~Variant() {
		if (type != Variant::NIL) clear();
	}
};

//typedef Dictionary Dictionary; no
//typedef Array Array;

Vector<Variant> varray();
Vector<Variant> varray(const Variant &p_arg1);
Vector<Variant> varray(const Variant &p_arg1, const Variant &p_arg2);
Vector<Variant> varray(const Variant &p_arg1, const Variant &p_arg2, const Variant &p_arg3);
Vector<Variant> varray(const Variant &p_arg1, const Variant &p_arg2, const Variant &p_arg3, const Variant &p_arg4);
Vector<Variant> varray(const Variant &p_arg1, const Variant &p_arg2, const Variant &p_arg3, const Variant &p_arg4, const Variant &p_arg5);

struct VariantHasher {

	static _FORCE_INLINE_ uint32_t hash(const Variant &p_variant) { return p_variant.hash(); }
};

struct VariantComparator {

	static _FORCE_INLINE_ bool compare(const Variant &p_lhs, const Variant &p_rhs) { return p_lhs.hash_compare(p_rhs); }
};

Variant::ObjData &Variant::_get_obj() {

	return *reinterpret_cast<ObjData *>(&_data._mem[0]);
}

const Variant::ObjData &Variant::_get_obj() const {

	return *reinterpret_cast<const ObjData *>(&_data._mem[0]);
}

String vformat(const String &p_text, const Variant &p1 = Variant(), const Variant &p2 = Variant(), const Variant &p3 = Variant(), const Variant &p4 = Variant(), const Variant &p5 = Variant());
#endif
