/*************************************************************************/
/*  variant_construct.h                                                  */
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

#ifndef VARIANT_CONSTRUCT_H
#define VARIANT_CONSTRUCT_H

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
MAKE_PTRCONSTRUCT(Quaternion);
MAKE_PTRCONSTRUCT(AABB);
MAKE_PTRCONSTRUCT(Basis);
MAKE_PTRCONSTRUCT(Transform3D);
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

	static inline void validated_construct(Variant *r_ret, const Variant **p_args) {
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

	static inline void validated_construct(Variant *r_ret, const Variant **p_args) {
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

	static inline void validated_construct(Variant *r_ret, const Variant **p_args) {
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

	static inline void validated_construct(Variant *r_ret, const Variant **p_args) {
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

	static inline void validated_construct(Variant *r_ret, const Variant **p_args) {
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

		r_ret = Array();
		Array &dst_arr = *VariantGetInternalPtr<Array>::get_ptr(&r_ret);
		const T &src_arr = *VariantGetInternalPtr<T>::get_ptr(p_args[0]);

		int size = src_arr.size();
		dst_arr.resize(size);
		for (int i = 0; i < size; i++) {
			dst_arr[i] = src_arr[i];
		}
	}

	static inline void validated_construct(Variant *r_ret, const Variant **p_args) {
		*r_ret = Array();
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

	static inline void validated_construct(Variant *r_ret, const Variant **p_args) {
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

	static inline void validated_construct(Variant *r_ret, const Variant **p_args) {
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

	static inline void validated_construct(Variant *r_ret, const Variant **p_args) {
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

	static inline void validated_construct(Variant *r_ret, const Variant **p_args) {
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

	static inline void validated_construct(Variant *r_ret, const Variant **p_args) {
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

#endif // VARIANT_CONSTRUCT_H
