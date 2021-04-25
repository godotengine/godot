/*************************************************************************/
/*  variant_construct.cpp                                                */
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

#include "variant.h"

#include "core/core_string_names.h"
#include "core/crypto/crypto_core.h"
#include "core/debugger/engine_debugger.h"
#include "core/io/compression.h"
#include "core/object/class_db.h"
#include "core/os/os.h"
#include "core/templates/local_vector.h"
#include "core/templates/oa_hash_map.h"

template <class T>
struct PtrConstruct {};

#define MAKE_PTRCONSTRUCT(m_type)                                                  \
	template <>                                                                    \
	struct PtrConstruct<m_type> {                                                  \
		_FORCE_INLINE_ static void construct(const m_type &p_value, void *p_ptr) { \
			memnew_placement(p_ptr, m_type(p_value));                              \
		}                                                                          \
	};

MAKE_PTRCONSTRUCT(bool);
MAKE_PTRCONSTRUCT(int64_t);
MAKE_PTRCONSTRUCT(double);
MAKE_PTRCONSTRUCT(String);
MAKE_PTRCONSTRUCT(Vector2);
MAKE_PTRCONSTRUCT(Vector2i);
MAKE_PTRCONSTRUCT(Rect2);
MAKE_PTRCONSTRUCT(Rect2i);
MAKE_PTRCONSTRUCT(Vector3);
MAKE_PTRCONSTRUCT(Vector3i);
MAKE_PTRCONSTRUCT(Transform2D);
MAKE_PTRCONSTRUCT(Plane);
MAKE_PTRCONSTRUCT(Quat);
MAKE_PTRCONSTRUCT(AABB);
MAKE_PTRCONSTRUCT(Basis);
MAKE_PTRCONSTRUCT(Transform);
MAKE_PTRCONSTRUCT(Color);
MAKE_PTRCONSTRUCT(StringName);
MAKE_PTRCONSTRUCT(NodePath);
MAKE_PTRCONSTRUCT(RID);

template <>
struct PtrConstruct<Object *> {
	_FORCE_INLINE_ static void construct(Object *p_value, void *p_ptr) {
		*((Object **)p_ptr) = p_value;
	}
};

MAKE_PTRCONSTRUCT(Callable);
MAKE_PTRCONSTRUCT(Signal);
MAKE_PTRCONSTRUCT(Dictionary);
MAKE_PTRCONSTRUCT(Array);
MAKE_PTRCONSTRUCT(PackedByteArray);
MAKE_PTRCONSTRUCT(PackedInt32Array);
MAKE_PTRCONSTRUCT(PackedInt64Array);
MAKE_PTRCONSTRUCT(PackedFloat32Array);
MAKE_PTRCONSTRUCT(PackedFloat64Array);
MAKE_PTRCONSTRUCT(PackedStringArray);
MAKE_PTRCONSTRUCT(PackedVector2Array);
MAKE_PTRCONSTRUCT(PackedVector3Array);
MAKE_PTRCONSTRUCT(PackedColorArray);
MAKE_PTRCONSTRUCT(Variant);

template <class T, class... P>
class VariantConstructor {
	template <size_t... Is>
	static _FORCE_INLINE_ void construct_helper(T &base, const Variant **p_args, Callable::CallError &r_error, IndexSequence<Is...>) {
		r_error.error = Callable::CallError::CALL_OK;

#ifdef DEBUG_METHODS_ENABLED
		base = T(VariantCasterAndValidate<P>::cast(p_args, Is, r_error)...);
#else
		base = T(VariantCaster<P>::cast(*p_args[Is])...);
#endif
	}

	template <size_t... Is>
	static _FORCE_INLINE_ void validated_construct_helper(T &base, const Variant **p_args, IndexSequence<Is...>) {
		base = T((*VariantGetInternalPtr<P>::get_ptr(p_args[Is]))...);
	}

	template <size_t... Is>
	static _FORCE_INLINE_ void ptr_construct_helper(void *base, const void **p_args, IndexSequence<Is...>) {
		PtrConstruct<T>::construct(T(PtrToArg<P>::convert(p_args[Is])...), base);
	}

public:
	static void construct(Variant &r_ret, const Variant **p_args, Callable::CallError &r_error) {
		r_error.error = Callable::CallError::CALL_OK;
		VariantTypeChanger<T>::change(&r_ret);
		construct_helper(*VariantGetInternalPtr<T>::get_ptr(&r_ret), p_args, r_error, BuildIndexSequence<sizeof...(P)>{});
	}

	static void validated_construct(Variant *r_ret, const Variant **p_args) {
		VariantTypeChanger<T>::change(r_ret);
		validated_construct_helper(*VariantGetInternalPtr<T>::get_ptr(r_ret), p_args, BuildIndexSequence<sizeof...(P)>{});
	}
	static void ptr_construct(void *base, const void **p_args) {
		ptr_construct_helper(base, p_args, BuildIndexSequence<sizeof...(P)>{});
	}

	static int get_argument_count() {
		return sizeof...(P);
	}

	static Variant::Type get_argument_type(int p_arg) {
		return call_get_argument_type<P...>(p_arg);
	}

	static Variant::Type get_base_type() {
		return GetTypeInfo<T>::VARIANT_TYPE;
	}
};

class VariantConstructorObject {
public:
	static void construct(Variant &r_ret, const Variant **p_args, Callable::CallError &r_error) {
		VariantInternal::clear(&r_ret);
		if (p_args[0]->get_type() == Variant::NIL) {
			VariantInternal::object_assign_null(&r_ret);
			r_error.error = Callable::CallError::CALL_OK;
		} else if (p_args[0]->get_type() == Variant::OBJECT) {
			VariantInternal::object_assign(&r_ret, p_args[0]);
			r_error.error = Callable::CallError::CALL_OK;
		} else {
			r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
			r_error.argument = 0;
			r_error.expected = Variant::OBJECT;
		}
	}

	static void validated_construct(Variant *r_ret, const Variant **p_args) {
		VariantInternal::clear(r_ret);
		VariantInternal::object_assign(r_ret, p_args[0]);
	}
	static void ptr_construct(void *base, const void **p_args) {
		PtrConstruct<Object *>::construct(PtrToArg<Object *>::convert(p_args[0]), base);
	}

	static int get_argument_count() {
		return 1;
	}

	static Variant::Type get_argument_type(int p_arg) {
		return Variant::OBJECT;
	}

	static Variant::Type get_base_type() {
		return Variant::OBJECT;
	}
};

class VariantConstructorNilObject {
public:
	static void construct(Variant &r_ret, const Variant **p_args, Callable::CallError &r_error) {
		if (p_args[0]->get_type() != Variant::NIL) {
			r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
			r_error.argument = 0;
			r_error.expected = Variant::NIL;
		}

		VariantInternal::clear(&r_ret);
		VariantInternal::object_assign_null(&r_ret);
	}

	static void validated_construct(Variant *r_ret, const Variant **p_args) {
		VariantInternal::clear(r_ret);
		VariantInternal::object_assign_null(r_ret);
	}
	static void ptr_construct(void *base, const void **p_args) {
		PtrConstruct<Object *>::construct(nullptr, base);
	}

	static int get_argument_count() {
		return 1;
	}

	static Variant::Type get_argument_type(int p_arg) {
		return Variant::NIL;
	}

	static Variant::Type get_base_type() {
		return Variant::OBJECT;
	}
};

class VariantConstructorCallableArgs {
public:
	static void construct(Variant &r_ret, const Variant **p_args, Callable::CallError &r_error) {
		ObjectID object_id;
		StringName method;

		if (p_args[0]->get_type() == Variant::NIL) {
			// leave as is
		} else if (p_args[0]->get_type() == Variant::OBJECT) {
			object_id = VariantInternal::get_object_id(p_args[0]);
		} else {
			r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
			r_error.argument = 0;
			r_error.expected = Variant::OBJECT;
			return;
		}

		if (p_args[1]->get_type() == Variant::STRING_NAME) {
			method = *VariantGetInternalPtr<StringName>::get_ptr(p_args[1]);
		} else if (p_args[1]->get_type() == Variant::STRING) {
			method = *VariantGetInternalPtr<String>::get_ptr(p_args[1]);
		} else {
			r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
			r_error.argument = 1;
			r_error.expected = Variant::STRING_NAME;
			return;
		}

		VariantTypeChanger<Callable>::change(&r_ret);
		*VariantGetInternalPtr<Callable>::get_ptr(&r_ret) = Callable(object_id, method);
	}

	static void validated_construct(Variant *r_ret, const Variant **p_args) {
		VariantTypeChanger<Callable>::change(r_ret);
		*VariantGetInternalPtr<Callable>::get_ptr(r_ret) = Callable(VariantInternal::get_object_id(p_args[0]), *VariantGetInternalPtr<StringName>::get_ptr(p_args[1]));
	}
	static void ptr_construct(void *base, const void **p_args) {
		PtrConstruct<Callable>::construct(Callable(PtrToArg<Object *>::convert(p_args[0]), PtrToArg<StringName>::convert(p_args[1])), base);
	}

	static int get_argument_count() {
		return 2;
	}

	static Variant::Type get_argument_type(int p_arg) {
		if (p_arg == 0) {
			return Variant::OBJECT;
		} else {
			return Variant::STRING_NAME;
		}
	}

	static Variant::Type get_base_type() {
		return Variant::CALLABLE;
	}
};

class VariantConstructorSignalArgs {
public:
	static void construct(Variant &r_ret, const Variant **p_args, Callable::CallError &r_error) {
		ObjectID object_id;
		StringName method;

		if (p_args[0]->get_type() == Variant::NIL) {
			// leave as is
		} else if (p_args[0]->get_type() == Variant::OBJECT) {
			object_id = VariantInternal::get_object_id(p_args[0]);
		} else {
			r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
			r_error.argument = 0;
			r_error.expected = Variant::OBJECT;
			return;
		}

		if (p_args[1]->get_type() == Variant::STRING_NAME) {
			method = *VariantGetInternalPtr<StringName>::get_ptr(p_args[1]);
		} else if (p_args[1]->get_type() == Variant::STRING) {
			method = *VariantGetInternalPtr<String>::get_ptr(p_args[1]);
		} else {
			r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
			r_error.argument = 1;
			r_error.expected = Variant::STRING_NAME;
			return;
		}

		VariantTypeChanger<Signal>::change(&r_ret);
		*VariantGetInternalPtr<Signal>::get_ptr(&r_ret) = Signal(object_id, method);
	}

	static void validated_construct(Variant *r_ret, const Variant **p_args) {
		VariantTypeChanger<Signal>::change(r_ret);
		*VariantGetInternalPtr<Signal>::get_ptr(r_ret) = Signal(VariantInternal::get_object_id(p_args[0]), *VariantGetInternalPtr<StringName>::get_ptr(p_args[1]));
	}
	static void ptr_construct(void *base, const void **p_args) {
		PtrConstruct<Signal>::construct(Signal(PtrToArg<Object *>::convert(p_args[0]), PtrToArg<StringName>::convert(p_args[1])), base);
	}

	static int get_argument_count() {
		return 2;
	}

	static Variant::Type get_argument_type(int p_arg) {
		if (p_arg == 0) {
			return Variant::OBJECT;
		} else {
			return Variant::STRING_NAME;
		}
	}

	static Variant::Type get_base_type() {
		return Variant::SIGNAL;
	}
};

template <class T>
class VariantConstructorToArray {
public:
	static void construct(Variant &r_ret, const Variant **p_args, Callable::CallError &r_error) {
		if (p_args[0]->get_type() != GetTypeInfo<T>::VARIANT_TYPE) {
			r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
			r_error.argument = 0;
			r_error.expected = GetTypeInfo<T>::VARIANT_TYPE;
			return;
		}

		VariantTypeChanger<Array>::change(&r_ret);
		Array &dst_arr = *VariantGetInternalPtr<Array>::get_ptr(&r_ret);
		const T &src_arr = *VariantGetInternalPtr<T>::get_ptr(p_args[0]);

		int size = src_arr.size();
		dst_arr.resize(size);
		for (int i = 0; i < size; i++) {
			dst_arr[i] = src_arr[i];
		}
	}

	static void validated_construct(Variant *r_ret, const Variant **p_args) {
		VariantTypeChanger<Array>::change(r_ret);
		Array &dst_arr = *VariantGetInternalPtr<Array>::get_ptr(r_ret);
		const T &src_arr = *VariantGetInternalPtr<T>::get_ptr(p_args[0]);

		int size = src_arr.size();
		dst_arr.resize(size);
		for (int i = 0; i < size; i++) {
			dst_arr[i] = src_arr[i];
		}
	}
	static void ptr_construct(void *base, const void **p_args) {
		Array dst_arr;
		T src_arr = PtrToArg<T>::convert(p_args[0]);

		int size = src_arr.size();
		dst_arr.resize(size);
		for (int i = 0; i < size; i++) {
			dst_arr[i] = src_arr[i];
		}

		PtrConstruct<Array>::construct(dst_arr, base);
	}

	static int get_argument_count() {
		return 1;
	}

	static Variant::Type get_argument_type(int p_arg) {
		return GetTypeInfo<T>::VARIANT_TYPE;
	}

	static Variant::Type get_base_type() {
		return Variant::ARRAY;
	}
};

template <class T>
class VariantConstructorFromArray {
public:
	static void construct(Variant &r_ret, const Variant **p_args, Callable::CallError &r_error) {
		if (p_args[0]->get_type() != Variant::ARRAY) {
			r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
			r_error.argument = 0;
			r_error.expected = Variant::ARRAY;
			return;
		}

		VariantTypeChanger<T>::change(&r_ret);
		const Array &src_arr = *VariantGetInternalPtr<Array>::get_ptr(p_args[0]);
		T &dst_arr = *VariantGetInternalPtr<T>::get_ptr(&r_ret);

		int size = src_arr.size();
		dst_arr.resize(size);
		for (int i = 0; i < size; i++) {
			dst_arr.write[i] = src_arr[i];
		}
	}

	static void validated_construct(Variant *r_ret, const Variant **p_args) {
		VariantTypeChanger<T>::change(r_ret);
		const Array &src_arr = *VariantGetInternalPtr<Array>::get_ptr(p_args[0]);
		T &dst_arr = *VariantGetInternalPtr<T>::get_ptr(r_ret);

		int size = src_arr.size();
		dst_arr.resize(size);
		for (int i = 0; i < size; i++) {
			dst_arr.write[i] = src_arr[i];
		}
	}
	static void ptr_construct(void *base, const void **p_args) {
		Array src_arr = PtrToArg<Array>::convert(p_args[0]);
		T dst_arr;

		int size = src_arr.size();
		dst_arr.resize(size);
		for (int i = 0; i < size; i++) {
			dst_arr.write[i] = src_arr[i];
		}

		PtrConstruct<T>::construct(dst_arr, base);
	}

	static int get_argument_count() {
		return 1;
	}

	static Variant::Type get_argument_type(int p_arg) {
		return Variant::ARRAY;
	}

	static Variant::Type get_base_type() {
		return GetTypeInfo<T>::VARIANT_TYPE;
	}
};

class VariantConstructorNil {
public:
	static void construct(Variant &r_ret, const Variant **p_args, Callable::CallError &r_error) {
		if (p_args[0]->get_type() != Variant::NIL) {
			r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
			r_error.argument = 0;
			r_error.expected = Variant::NIL;
			return;
		}

		r_error.error = Callable::CallError::CALL_OK;
		VariantInternal::clear(&r_ret);
	}

	static void validated_construct(Variant *r_ret, const Variant **p_args) {
		VariantInternal::clear(r_ret);
	}
	static void ptr_construct(void *base, const void **p_args) {
		PtrConstruct<Variant>::construct(Variant(), base);
	}

	static int get_argument_count() {
		return 1;
	}

	static Variant::Type get_argument_type(int p_arg) {
		return Variant::NIL;
	}

	static Variant::Type get_base_type() {
		return Variant::NIL;
	}
};

template <class T>
class VariantConstructNoArgs {
public:
	static void construct(Variant &r_ret, const Variant **p_args, Callable::CallError &r_error) {
		VariantTypeChanger<T>::change_and_reset(&r_ret);
		r_error.error = Callable::CallError::CALL_OK;
	}

	static void validated_construct(Variant *r_ret, const Variant **p_args) {
		VariantTypeChanger<T>::change_and_reset(r_ret);
	}
	static void ptr_construct(void *base, const void **p_args) {
		PtrConstruct<T>::construct(T(), base);
	}

	static int get_argument_count() {
		return 0;
	}

	static Variant::Type get_argument_type(int p_arg) {
		return Variant::NIL;
	}

	static Variant::Type get_base_type() {
		return GetTypeInfo<T>::VARIANT_TYPE;
	}
};

class VariantConstructNoArgsNil {
public:
	static void construct(Variant &r_ret, const Variant **p_args, Callable::CallError &r_error) {
		VariantInternal::clear(&r_ret);
		r_error.error = Callable::CallError::CALL_OK;
	}

	static void validated_construct(Variant *r_ret, const Variant **p_args) {
		VariantInternal::clear(r_ret);
	}
	static void ptr_construct(void *base, const void **p_args) {
		ERR_FAIL_MSG("can't ptrcall nil constructor");
	}

	static int get_argument_count() {
		return 0;
	}

	static Variant::Type get_argument_type(int p_arg) {
		return Variant::NIL;
	}

	static Variant::Type get_base_type() {
		return Variant::NIL;
	}
};

class VariantConstructNoArgsObject {
public:
	static void construct(Variant &r_ret, const Variant **p_args, Callable::CallError &r_error) {
		VariantInternal::clear(&r_ret);
		VariantInternal::object_assign_null(&r_ret);
		r_error.error = Callable::CallError::CALL_OK;
	}

	static void validated_construct(Variant *r_ret, const Variant **p_args) {
		VariantInternal::clear(r_ret);
		VariantInternal::object_assign_null(r_ret);
	}
	static void ptr_construct(void *base, const void **p_args) {
		PtrConstruct<Object *>::construct(nullptr, base);
	}

	static int get_argument_count() {
		return 0;
	}

	static Variant::Type get_argument_type(int p_arg) {
		return Variant::NIL;
	}

	static Variant::Type get_base_type() {
		return Variant::OBJECT;
	}
};

struct VariantConstructData {
	void (*construct)(Variant &r_base, const Variant **p_args, Callable::CallError &r_error);
	Variant::ValidatedConstructor validated_construct;
	Variant::PTRConstructor ptr_construct;
	Variant::Type (*get_argument_type)(int);
	int argument_count;
	Vector<String> arg_names;
};

static LocalVector<VariantConstructData> construct_data[Variant::VARIANT_MAX];

template <class T>
static void add_constructor(const Vector<String> &arg_names) {
	ERR_FAIL_COND_MSG(arg_names.size() != T::get_argument_count(), "Argument names size mismatch for " + Variant::get_type_name(T::get_base_type()) + ".");

	VariantConstructData cd;
	cd.construct = T::construct;
	cd.validated_construct = T::validated_construct;
	cd.ptr_construct = T::ptr_construct;
	cd.get_argument_type = T::get_argument_type;
	cd.argument_count = T::get_argument_count();
	cd.arg_names = arg_names;
	construct_data[T::get_base_type()].push_back(cd);
}

void Variant::_register_variant_constructors() {
	add_constructor<VariantConstructNoArgsNil>(sarray());
	add_constructor<VariantConstructorNil>(sarray("from"));

	add_constructor<VariantConstructNoArgs<bool>>(sarray());
	add_constructor<VariantConstructor<bool, bool>>(sarray("from"));
	add_constructor<VariantConstructor<bool, int64_t>>(sarray("from"));
	add_constructor<VariantConstructor<bool, double>>(sarray("from"));

	add_constructor<VariantConstructNoArgs<int64_t>>(sarray());
	add_constructor<VariantConstructor<int64_t, int64_t>>(sarray("from"));
	add_constructor<VariantConstructor<int64_t, double>>(sarray("from"));
	add_constructor<VariantConstructor<int64_t, bool>>(sarray("from"));

	add_constructor<VariantConstructNoArgs<double>>(sarray());
	add_constructor<VariantConstructor<double, double>>(sarray("from"));
	add_constructor<VariantConstructor<double, int64_t>>(sarray("from"));
	add_constructor<VariantConstructor<double, bool>>(sarray("from"));

	add_constructor<VariantConstructNoArgs<String>>(sarray());
	add_constructor<VariantConstructor<String, String>>(sarray("from"));
	add_constructor<VariantConstructor<String, StringName>>(sarray("from"));
	add_constructor<VariantConstructor<String, NodePath>>(sarray("from"));

	add_constructor<VariantConstructNoArgs<Vector2>>(sarray());
	add_constructor<VariantConstructor<Vector2, Vector2>>(sarray("from"));
	add_constructor<VariantConstructor<Vector2, Vector2i>>(sarray("from"));
	add_constructor<VariantConstructor<Vector2, double, double>>(sarray("x", "y"));

	add_constructor<VariantConstructNoArgs<Vector2i>>(sarray());
	add_constructor<VariantConstructor<Vector2i, Vector2i>>(sarray("from"));
	add_constructor<VariantConstructor<Vector2i, Vector2>>(sarray("from"));
	add_constructor<VariantConstructor<Vector2i, int64_t, int64_t>>(sarray("x", "y"));

	add_constructor<VariantConstructNoArgs<Rect2>>(sarray());
	add_constructor<VariantConstructor<Rect2, Rect2>>(sarray("from"));
	add_constructor<VariantConstructor<Rect2, Rect2i>>(sarray("from"));
	add_constructor<VariantConstructor<Rect2, Vector2, Vector2>>(sarray("position", "size"));
	add_constructor<VariantConstructor<Rect2, double, double, double, double>>(sarray("x", "y", "width", "height"));

	add_constructor<VariantConstructNoArgs<Rect2i>>(sarray());
	add_constructor<VariantConstructor<Rect2i, Rect2i>>(sarray("from"));
	add_constructor<VariantConstructor<Rect2i, Rect2>>(sarray("from"));
	add_constructor<VariantConstructor<Rect2i, Vector2i, Vector2i>>(sarray("position", "size"));
	add_constructor<VariantConstructor<Rect2i, int64_t, int64_t, int64_t, int64_t>>(sarray("x", "y", "width", "height"));

	add_constructor<VariantConstructNoArgs<Vector3>>(sarray());
	add_constructor<VariantConstructor<Vector3, Vector3>>(sarray("from"));
	add_constructor<VariantConstructor<Vector3, Vector3i>>(sarray("from"));
	add_constructor<VariantConstructor<Vector3, double, double, double>>(sarray("x", "y", "z"));

	add_constructor<VariantConstructNoArgs<Vector3i>>(sarray());
	add_constructor<VariantConstructor<Vector3i, Vector3i>>(sarray("from"));
	add_constructor<VariantConstructor<Vector3i, Vector3>>(sarray("from"));
	add_constructor<VariantConstructor<Vector3i, int64_t, int64_t, int64_t>>(sarray("x", "y", "z"));

	add_constructor<VariantConstructNoArgs<Transform2D>>(sarray());
	add_constructor<VariantConstructor<Transform2D, Transform2D>>(sarray("from"));
	add_constructor<VariantConstructor<Transform2D, float, Vector2>>(sarray("rotation", "position"));
	add_constructor<VariantConstructor<Transform2D, Vector2, Vector2, Vector2>>(sarray("x_axis", "y_axis", "origin"));

	add_constructor<VariantConstructNoArgs<Plane>>(sarray());
	add_constructor<VariantConstructor<Plane, Plane>>(sarray("from"));
	add_constructor<VariantConstructor<Plane, Vector3, double>>(sarray("normal", "d"));
	add_constructor<VariantConstructor<Plane, Vector3, Vector3>>(sarray("point", "normal"));
	add_constructor<VariantConstructor<Plane, Vector3, Vector3, Vector3>>(sarray("point1", "point2", "point3"));
	add_constructor<VariantConstructor<Plane, double, double, double, double>>(sarray("a", "b", "c", "d"));

	add_constructor<VariantConstructNoArgs<Quat>>(sarray());
	add_constructor<VariantConstructor<Quat, Quat>>(sarray("from"));
	add_constructor<VariantConstructor<Quat, Basis>>(sarray("from"));
	add_constructor<VariantConstructor<Quat, Vector3>>(sarray("euler"));
	add_constructor<VariantConstructor<Quat, Vector3, double>>(sarray("axis", "angle"));
	add_constructor<VariantConstructor<Quat, Vector3, Vector3>>(sarray("arc_from", "arc_to"));
	add_constructor<VariantConstructor<Quat, double, double, double, double>>(sarray("x", "y", "z", "w"));

	add_constructor<VariantConstructNoArgs<::AABB>>(sarray());
	add_constructor<VariantConstructor<::AABB, ::AABB>>(sarray("from"));
	add_constructor<VariantConstructor<::AABB, Vector3, Vector3>>(sarray("position", "size"));

	add_constructor<VariantConstructNoArgs<Basis>>(sarray());
	add_constructor<VariantConstructor<Basis, Basis>>(sarray("from"));
	add_constructor<VariantConstructor<Basis, Quat>>(sarray("from"));
	add_constructor<VariantConstructor<Basis, Vector3>>(sarray("euler"));
	add_constructor<VariantConstructor<Basis, Vector3, double>>(sarray("axis", "phi"));
	add_constructor<VariantConstructor<Basis, Vector3, Vector3, Vector3>>(sarray("x_axis", "y_axis", "z_axis"));

	add_constructor<VariantConstructNoArgs<Transform>>(sarray());
	add_constructor<VariantConstructor<Transform, Transform>>(sarray("from"));
	add_constructor<VariantConstructor<Transform, Basis, Vector3>>(sarray("basis", "origin"));
	add_constructor<VariantConstructor<Transform, Vector3, Vector3, Vector3, Vector3>>(sarray("x_axis", "y_axis", "z_axis", "origin"));

	add_constructor<VariantConstructNoArgs<Color>>(sarray());
	add_constructor<VariantConstructor<Color, Color>>(sarray("from"));
	add_constructor<VariantConstructor<Color, Color, double>>(sarray("from", "alpha"));
	add_constructor<VariantConstructor<Color, double, double, double>>(sarray("r", "g", "b"));
	add_constructor<VariantConstructor<Color, double, double, double, double>>(sarray("r", "g", "b", "a"));
	add_constructor<VariantConstructor<Color, String>>(sarray("code"));
	add_constructor<VariantConstructor<Color, String, double>>(sarray("code", "alpha"));

	add_constructor<VariantConstructNoArgs<StringName>>(sarray());
	add_constructor<VariantConstructor<StringName, StringName>>(sarray("from"));
	add_constructor<VariantConstructor<StringName, String>>(sarray("from"));

	add_constructor<VariantConstructNoArgs<NodePath>>(sarray());
	add_constructor<VariantConstructor<NodePath, NodePath>>(sarray("from"));
	add_constructor<VariantConstructor<NodePath, String>>(sarray("from"));

	add_constructor<VariantConstructNoArgs<::RID>>(sarray());
	add_constructor<VariantConstructor<::RID, ::RID>>(sarray("from"));

	add_constructor<VariantConstructNoArgsObject>(sarray());
	add_constructor<VariantConstructorObject>(sarray("from"));
	add_constructor<VariantConstructorNilObject>(sarray("from"));

	add_constructor<VariantConstructNoArgs<Callable>>(sarray());
	add_constructor<VariantConstructor<Callable, Callable>>(sarray("from"));
	add_constructor<VariantConstructorCallableArgs>(sarray("object", "method"));

	add_constructor<VariantConstructNoArgs<Signal>>(sarray());
	add_constructor<VariantConstructor<Signal, Signal>>(sarray("from"));
	add_constructor<VariantConstructorSignalArgs>(sarray("object", "signal"));

	add_constructor<VariantConstructNoArgs<Dictionary>>(sarray());
	add_constructor<VariantConstructor<Dictionary, Dictionary>>(sarray("from"));

	add_constructor<VariantConstructNoArgs<Array>>(sarray());
	add_constructor<VariantConstructor<Array, Array>>(sarray("from"));
	add_constructor<VariantConstructorToArray<PackedByteArray>>(sarray("from"));
	add_constructor<VariantConstructorToArray<PackedInt32Array>>(sarray("from"));
	add_constructor<VariantConstructorToArray<PackedInt64Array>>(sarray("from"));
	add_constructor<VariantConstructorToArray<PackedFloat32Array>>(sarray("from"));
	add_constructor<VariantConstructorToArray<PackedFloat64Array>>(sarray("from"));
	add_constructor<VariantConstructorToArray<PackedStringArray>>(sarray("from"));
	add_constructor<VariantConstructorToArray<PackedVector2Array>>(sarray("from"));
	add_constructor<VariantConstructorToArray<PackedVector3Array>>(sarray("from"));
	add_constructor<VariantConstructorToArray<PackedColorArray>>(sarray("from"));

	add_constructor<VariantConstructNoArgs<PackedByteArray>>(sarray());
	add_constructor<VariantConstructor<PackedByteArray, PackedByteArray>>(sarray("from"));
	add_constructor<VariantConstructorFromArray<PackedByteArray>>(sarray("from"));

	add_constructor<VariantConstructNoArgs<PackedInt32Array>>(sarray());
	add_constructor<VariantConstructor<PackedInt32Array, PackedInt32Array>>(sarray("from"));
	add_constructor<VariantConstructorFromArray<PackedInt32Array>>(sarray("from"));

	add_constructor<VariantConstructNoArgs<PackedInt64Array>>(sarray());
	add_constructor<VariantConstructor<PackedInt64Array, PackedInt64Array>>(sarray("from"));
	add_constructor<VariantConstructorFromArray<PackedInt64Array>>(sarray("from"));

	add_constructor<VariantConstructNoArgs<PackedFloat32Array>>(sarray());
	add_constructor<VariantConstructor<PackedFloat32Array, PackedFloat32Array>>(sarray("from"));
	add_constructor<VariantConstructorFromArray<PackedFloat32Array>>(sarray("from"));

	add_constructor<VariantConstructNoArgs<PackedFloat64Array>>(sarray());
	add_constructor<VariantConstructor<PackedFloat64Array, PackedFloat64Array>>(sarray("from"));
	add_constructor<VariantConstructorFromArray<PackedFloat64Array>>(sarray("from"));

	add_constructor<VariantConstructNoArgs<PackedStringArray>>(sarray());
	add_constructor<VariantConstructor<PackedStringArray, PackedStringArray>>(sarray("from"));
	add_constructor<VariantConstructorFromArray<PackedStringArray>>(sarray("from"));

	add_constructor<VariantConstructNoArgs<PackedVector2Array>>(sarray());
	add_constructor<VariantConstructor<PackedVector2Array, PackedVector2Array>>(sarray("from"));
	add_constructor<VariantConstructorFromArray<PackedVector2Array>>(sarray("from"));

	add_constructor<VariantConstructNoArgs<PackedVector3Array>>(sarray());
	add_constructor<VariantConstructor<PackedVector3Array, PackedVector3Array>>(sarray("from"));
	add_constructor<VariantConstructorFromArray<PackedVector3Array>>(sarray("from"));

	add_constructor<VariantConstructNoArgs<PackedColorArray>>(sarray());
	add_constructor<VariantConstructor<PackedColorArray, PackedColorArray>>(sarray("from"));
	add_constructor<VariantConstructorFromArray<PackedColorArray>>(sarray("from"));
}

void Variant::_unregister_variant_constructors() {
	for (int i = 0; i < Variant::VARIANT_MAX; i++) {
		construct_data[i].clear();
	}
}

void Variant::construct(Variant::Type p_type, Variant &base, const Variant **p_args, int p_argcount, Callable::CallError &r_error) {
	ERR_FAIL_INDEX(p_type, Variant::VARIANT_MAX);
	uint32_t s = construct_data[p_type].size();
	for (uint32_t i = 0; i < s; i++) {
		int argc = construct_data[p_type][i].argument_count;
		if (argc != p_argcount) {
			continue;
		}
		bool args_match = true;
		for (int j = 0; j < argc; j++) {
			if (!Variant::can_convert_strict(p_args[j]->get_type(), construct_data[p_type][i].get_argument_type(j))) {
				args_match = false;
				break;
			}
		}

		if (!args_match) {
			continue;
		}

		construct_data[p_type][i].construct(base, p_args, r_error);
		return;
	}

	r_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;
}

int Variant::get_constructor_count(Variant::Type p_type) {
	ERR_FAIL_INDEX_V(p_type, Variant::VARIANT_MAX, -1);
	return construct_data[p_type].size();
}

Variant::ValidatedConstructor Variant::get_validated_constructor(Variant::Type p_type, int p_constructor) {
	ERR_FAIL_INDEX_V(p_type, Variant::VARIANT_MAX, nullptr);
	ERR_FAIL_INDEX_V(p_constructor, (int)construct_data[p_type].size(), nullptr);
	return construct_data[p_type][p_constructor].validated_construct;
}

Variant::PTRConstructor Variant::get_ptr_constructor(Variant::Type p_type, int p_constructor) {
	ERR_FAIL_INDEX_V(p_type, Variant::VARIANT_MAX, nullptr);
	ERR_FAIL_INDEX_V(p_constructor, (int)construct_data[p_type].size(), nullptr);
	return construct_data[p_type][p_constructor].ptr_construct;
}

int Variant::get_constructor_argument_count(Variant::Type p_type, int p_constructor) {
	ERR_FAIL_INDEX_V(p_type, Variant::VARIANT_MAX, -1);
	ERR_FAIL_INDEX_V(p_constructor, (int)construct_data[p_type].size(), -1);
	return construct_data[p_type][p_constructor].argument_count;
}

Variant::Type Variant::get_constructor_argument_type(Variant::Type p_type, int p_constructor, int p_argument) {
	ERR_FAIL_INDEX_V(p_type, Variant::VARIANT_MAX, Variant::VARIANT_MAX);
	ERR_FAIL_INDEX_V(p_constructor, (int)construct_data[p_type].size(), Variant::VARIANT_MAX);
	return construct_data[p_type][p_constructor].get_argument_type(p_argument);
}

String Variant::get_constructor_argument_name(Variant::Type p_type, int p_constructor, int p_argument) {
	ERR_FAIL_INDEX_V(p_type, Variant::VARIANT_MAX, String());
	ERR_FAIL_INDEX_V(p_constructor, (int)construct_data[p_type].size(), String());
	return construct_data[p_type][p_constructor].arg_names[p_argument];
}

void VariantInternal::object_assign(Variant *v, const Object *o) {
	if (o) {
		if (o->is_reference()) {
			Reference *reference = const_cast<Reference *>(static_cast<const Reference *>(o));
			if (!reference->init_ref()) {
				v->_get_obj().obj = nullptr;
				v->_get_obj().id = ObjectID();
				return;
			}
		}

		v->_get_obj().obj = const_cast<Object *>(o);
		v->_get_obj().id = o->get_instance_id();
	} else {
		v->_get_obj().obj = nullptr;
		v->_get_obj().id = ObjectID();
	}
}

void Variant::get_constructor_list(Type p_type, List<MethodInfo> *r_list) {
	ERR_FAIL_INDEX(p_type, Variant::VARIANT_MAX);

	MethodInfo mi;
	mi.return_val.type = p_type;
	mi.name = get_type_name(p_type);

	for (int i = 0; i < get_constructor_count(p_type); i++) {
		int ac = get_constructor_argument_count(p_type, i);
		mi.arguments.clear();
		for (int j = 0; j < ac; j++) {
			PropertyInfo arg;
			arg.name = get_constructor_argument_name(p_type, i, j);
			arg.type = get_constructor_argument_type(p_type, i, j);
			mi.arguments.push_back(arg);
		}
		r_list->push_back(mi);
	}
}
