/*************************************************************************/
/*  method_bind.h                                                        */
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

#ifndef METHOD_BIND_H
#define METHOD_BIND_H

#include "core/variant/binder_common.h"

enum MethodFlags {
	METHOD_FLAG_NORMAL = 1,
	METHOD_FLAG_EDITOR = 2,
	METHOD_FLAG_NOSCRIPT = 4,
	METHOD_FLAG_CONST = 8,
	METHOD_FLAG_REVERSE = 16, // used for events
	METHOD_FLAG_VIRTUAL = 32,
	METHOD_FLAG_FROM_SCRIPT = 64,
	METHOD_FLAG_VARARG = 128,
	METHOD_FLAG_STATIC = 256,
	METHOD_FLAGS_DEFAULT = METHOD_FLAG_NORMAL,
};

VARIANT_ENUM_CAST(MethodFlags)

// some helpers

class MethodBind {
	int method_id;
	uint32_t hint_flags = METHOD_FLAGS_DEFAULT;
	StringName name;
	StringName instance_class;
	Vector<Variant> default_arguments;
	int default_argument_count = 0;
	int argument_count = 0;

	bool _const = false;
	bool _returns = false;

protected:
#ifdef DEBUG_METHODS_ENABLED
	Variant::Type *argument_types = nullptr;
	Vector<StringName> arg_names;
#endif
	void _set_const(bool p_const);
	void _set_returns(bool p_returns);
#ifdef DEBUG_METHODS_ENABLED
	virtual Variant::Type _gen_argument_type(int p_arg) const = 0;
	virtual PropertyInfo _gen_argument_type_info(int p_arg) const = 0;
	void _generate_argument_types(int p_count);

#endif
	void set_argument_count(int p_count) { argument_count = p_count; }

public:
	_FORCE_INLINE_ const Vector<Variant> &get_default_arguments() const { return default_arguments; }
	_FORCE_INLINE_ int get_default_argument_count() const { return default_argument_count; }

	_FORCE_INLINE_ Variant has_default_argument(int p_arg) const {
		int idx = p_arg - (argument_count - default_arguments.size());

		if (idx < 0 || idx >= default_arguments.size()) {
			return false;
		} else {
			return true;
		}
	}

	_FORCE_INLINE_ Variant get_default_argument(int p_arg) const {
		int idx = p_arg - (argument_count - default_arguments.size());

		if (idx < 0 || idx >= default_arguments.size()) {
			return Variant();
		} else {
			return default_arguments[idx];
		}
	}

#ifdef DEBUG_METHODS_ENABLED
	_FORCE_INLINE_ Variant::Type get_argument_type(int p_argument) const {
		ERR_FAIL_COND_V(p_argument < -1 || p_argument > argument_count, Variant::NIL);
		return argument_types[p_argument + 1];
	}

	PropertyInfo get_argument_info(int p_argument) const;
	PropertyInfo get_return_info() const;

	void set_argument_names(const Vector<StringName> &p_names); // Set by ClassDB, can't be inferred otherwise.
	Vector<StringName> get_argument_names() const;

	virtual GodotTypeInfo::Metadata get_argument_meta(int p_arg) const = 0;
#endif

	void set_hint_flags(uint32_t p_hint) { hint_flags = p_hint; }
	uint32_t get_hint_flags() const { return hint_flags | (is_const() ? METHOD_FLAG_CONST : 0) | (is_vararg() ? METHOD_FLAG_VARARG : 0); }
	_FORCE_INLINE_ StringName get_instance_class() const { return instance_class; }
	_FORCE_INLINE_ void set_instance_class(const StringName &p_class) { instance_class = p_class; }

	_FORCE_INLINE_ int get_argument_count() const { return argument_count; };

	virtual Variant call(Object *p_object, const Variant **p_args, int p_arg_count, Callable::CallError &r_error) = 0;
	virtual void ptrcall(Object *p_object, const void **p_args, void *r_ret) = 0;

	StringName get_name() const;
	void set_name(const StringName &p_name);
	_FORCE_INLINE_ int get_method_id() const { return method_id; }
	_FORCE_INLINE_ bool is_const() const { return _const; }
	_FORCE_INLINE_ bool has_return() const { return _returns; }
	virtual bool is_vararg() const { return false; }

	void set_default_arguments(const Vector<Variant> &p_defargs);

	MethodBind();
	virtual ~MethodBind();
};

template <class T>
class MethodBindVarArg : public MethodBind {
public:
	typedef Variant (T::*NativeCall)(const Variant **, int, Callable::CallError &);

protected:
	NativeCall call_method = nullptr;
#ifdef DEBUG_METHODS_ENABLED
	MethodInfo arguments;
#endif

public:
#ifdef DEBUG_METHODS_ENABLED
	virtual PropertyInfo _gen_argument_type_info(int p_arg) const {
		if (p_arg < 0) {
			return arguments.return_val;
		} else if (p_arg < arguments.arguments.size()) {
			return arguments.arguments[p_arg];
		} else {
			return PropertyInfo(Variant::NIL, "arg_" + itos(p_arg), PROPERTY_HINT_NONE, String(), PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_NIL_IS_VARIANT);
		}
	}

	virtual Variant::Type _gen_argument_type(int p_arg) const {
		return _gen_argument_type_info(p_arg).type;
	}

	virtual GodotTypeInfo::Metadata get_argument_meta(int) const {
		return GodotTypeInfo::METADATA_NONE;
	}
#else
	virtual Variant::Type _gen_argument_type(int p_arg) const {
		return Variant::NIL;
	}
#endif

	virtual Variant call(Object *p_object, const Variant **p_args, int p_arg_count, Callable::CallError &r_error) {
		T *instance = static_cast<T *>(p_object);
		return (instance->*call_method)(p_args, p_arg_count, r_error);
	}

	void set_method_info(const MethodInfo &p_info, bool p_return_nil_is_variant) {
		set_argument_count(p_info.arguments.size());
#ifdef DEBUG_METHODS_ENABLED
		Variant::Type *at = memnew_arr(Variant::Type, p_info.arguments.size() + 1);
		at[0] = p_info.return_val.type;
		if (p_info.arguments.size()) {
			Vector<StringName> names;
			names.resize(p_info.arguments.size());
			for (int i = 0; i < p_info.arguments.size(); i++) {
				at[i + 1] = p_info.arguments[i].type;
				names.write[i] = p_info.arguments[i].name;
			}

			set_argument_names(names);
		}
		argument_types = at;
		arguments = p_info;
		if (p_return_nil_is_variant) {
			arguments.return_val.usage |= PROPERTY_USAGE_NIL_IS_VARIANT;
		}
#endif
	}

	virtual void ptrcall(Object *p_object, const void **p_args, void *r_ret) {
		ERR_FAIL(); // Can't call.
	}

	void set_method(NativeCall p_method) { call_method = p_method; }
	virtual bool is_const() const { return false; }

	virtual bool is_vararg() const { return true; }

	MethodBindVarArg() {
		_set_returns(true);
	}
};

template <class T>
MethodBind *create_vararg_method_bind(Variant (T::*p_method)(const Variant **, int, Callable::CallError &), const MethodInfo &p_info, bool p_return_nil_is_variant) {
	MethodBindVarArg<T> *a = memnew((MethodBindVarArg<T>));
	a->set_method(p_method);
	a->set_method_info(p_info, p_return_nil_is_variant);
	a->set_instance_class(T::get_class_static());
	return a;
}

/**** VARIADIC TEMPLATES ****/

#ifndef TYPED_METHOD_BIND
class __UnexistingClass;
#define MB_T __UnexistingClass
#else
#define MB_T T
#endif

// no return, not const
#ifdef TYPED_METHOD_BIND
template <class T, class... P>
#else
template <class... P>
#endif
class MethodBindT : public MethodBind {
	void (MB_T::*method)(P...);

protected:
#ifdef DEBUG_METHODS_ENABLED
// GCC raises warnings in the case P = {} as the comparison is always false...
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wlogical-op"
#endif
	virtual Variant::Type _gen_argument_type(int p_arg) const {
		if (p_arg >= 0 && p_arg < (int)sizeof...(P)) {
			return call_get_argument_type<P...>(p_arg);
		} else {
			return Variant::NIL;
		}
	}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

	virtual PropertyInfo _gen_argument_type_info(int p_arg) const {
		PropertyInfo pi;
		call_get_argument_type_info<P...>(p_arg, pi);
		return pi;
	}
#endif

public:
#ifdef DEBUG_METHODS_ENABLED
	virtual GodotTypeInfo::Metadata get_argument_meta(int p_arg) const {
		return call_get_argument_metadata<P...>(p_arg);
	}

#endif
	virtual Variant call(Object *p_object, const Variant **p_args, int p_arg_count, Callable::CallError &r_error) {
#ifdef TYPED_METHOD_BIND
		call_with_variant_args_dv(static_cast<T *>(p_object), method, p_args, p_arg_count, r_error, get_default_arguments());
#else
		call_with_variant_args_dv((MB_T *)(p_object), method, p_args, p_arg_count, r_error, get_default_arguments());
#endif
		return Variant();
	}

	virtual void ptrcall(Object *p_object, const void **p_args, void *r_ret) {
#ifdef TYPED_METHOD_BIND
		call_with_ptr_args<T, P...>(static_cast<T *>(p_object), method, p_args);
#else
		call_with_ptr_args<MB_T, P...>((MB_T *)(p_object), method, p_args);
#endif
	}

	MethodBindT(void (MB_T::*p_method)(P...)) {
		method = p_method;
#ifdef DEBUG_METHODS_ENABLED
		_generate_argument_types(sizeof...(P));
#endif
		set_argument_count(sizeof...(P));
	}
};

template <class T, class... P>
MethodBind *create_method_bind(void (T::*p_method)(P...)) {
#ifdef TYPED_METHOD_BIND
	MethodBind *a = memnew((MethodBindT<T, P...>)(p_method));
#else
	MethodBind *a = memnew((MethodBindT<P...>)(reinterpret_cast<void (MB_T::*)(P...)>(p_method)));
#endif
	a->set_instance_class(T::get_class_static());
	return a;
}

// no return, not const

#ifdef TYPED_METHOD_BIND
template <class T, class... P>
#else
template <class... P>
#endif
class MethodBindTC : public MethodBind {
	void (MB_T::*method)(P...) const;

protected:
#ifdef DEBUG_METHODS_ENABLED
// GCC raises warnings in the case P = {} as the comparison is always false...
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wlogical-op"
#endif
	virtual Variant::Type _gen_argument_type(int p_arg) const {
		if (p_arg >= 0 && p_arg < (int)sizeof...(P)) {
			return call_get_argument_type<P...>(p_arg);
		} else {
			return Variant::NIL;
		}
	}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

	virtual PropertyInfo _gen_argument_type_info(int p_arg) const {
		PropertyInfo pi;
		call_get_argument_type_info<P...>(p_arg, pi);
		return pi;
	}
#endif

public:
#ifdef DEBUG_METHODS_ENABLED
	virtual GodotTypeInfo::Metadata get_argument_meta(int p_arg) const {
		return call_get_argument_metadata<P...>(p_arg);
	}

#endif
	virtual Variant call(Object *p_object, const Variant **p_args, int p_arg_count, Callable::CallError &r_error) {
#ifdef TYPED_METHOD_BIND
		call_with_variant_argsc_dv(static_cast<T *>(p_object), method, p_args, p_arg_count, r_error, get_default_arguments());
#else
		call_with_variant_argsc_dv((MB_T *)(p_object), method, p_args, p_arg_count, r_error, get_default_arguments());
#endif
		return Variant();
	}

	virtual void ptrcall(Object *p_object, const void **p_args, void *r_ret) {
#ifdef TYPED_METHOD_BIND
		call_with_ptr_argsc<T, P...>(static_cast<T *>(p_object), method, p_args);
#else
		call_with_ptr_argsc<MB_T, P...>((MB_T *)(p_object), method, p_args);
#endif
	}

	MethodBindTC(void (MB_T::*p_method)(P...) const) {
		method = p_method;
		_set_const(true);
#ifdef DEBUG_METHODS_ENABLED
		_generate_argument_types(sizeof...(P));
#endif
		set_argument_count(sizeof...(P));
	}
};

template <class T, class... P>
MethodBind *create_method_bind(void (T::*p_method)(P...) const) {
#ifdef TYPED_METHOD_BIND
	MethodBind *a = memnew((MethodBindTC<T, P...>)(p_method));
#else
	MethodBind *a = memnew((MethodBindTC<P...>)(reinterpret_cast<void (MB_T::*)(P...) const>(p_method)));
#endif
	a->set_instance_class(T::get_class_static());
	return a;
}

// return, not const

#ifdef TYPED_METHOD_BIND
template <class T, class R, class... P>
#else
template <class R, class... P>
#endif
class MethodBindTR : public MethodBind {
	R(MB_T::*method)
	(P...);

protected:
#ifdef DEBUG_METHODS_ENABLED
// GCC raises warnings in the case P = {} as the comparison is always false...
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wlogical-op"
#endif
	virtual Variant::Type _gen_argument_type(int p_arg) const {
		if (p_arg >= 0 && p_arg < (int)sizeof...(P)) {
			return call_get_argument_type<P...>(p_arg);
		} else {
			return GetTypeInfo<R>::VARIANT_TYPE;
		}
	}

	virtual PropertyInfo _gen_argument_type_info(int p_arg) const {
		if (p_arg >= 0 && p_arg < (int)sizeof...(P)) {
			PropertyInfo pi;
			call_get_argument_type_info<P...>(p_arg, pi);
			return pi;
		} else {
			return GetTypeInfo<R>::get_class_info();
		}
	}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
#endif

public:
#ifdef DEBUG_METHODS_ENABLED
	virtual GodotTypeInfo::Metadata get_argument_meta(int p_arg) const {
		if (p_arg >= 0) {
			return call_get_argument_metadata<P...>(p_arg);
		} else {
			return GetTypeInfo<R>::METADATA;
		}
	}
#endif

	virtual Variant call(Object *p_object, const Variant **p_args, int p_arg_count, Callable::CallError &r_error) {
		Variant ret;
#ifdef TYPED_METHOD_BIND
		call_with_variant_args_ret_dv(static_cast<T *>(p_object), method, p_args, p_arg_count, ret, r_error, get_default_arguments());
#else
		call_with_variant_args_ret_dv((MB_T *)p_object, method, p_args, p_arg_count, ret, r_error, get_default_arguments());
#endif
		return ret;
	}

	virtual void ptrcall(Object *p_object, const void **p_args, void *r_ret) {
#ifdef TYPED_METHOD_BIND
		call_with_ptr_args_ret<T, R, P...>(static_cast<T *>(p_object), method, p_args, r_ret);
#else
		call_with_ptr_args_ret<MB_T, R, P...>((MB_T *)(p_object), method, p_args, r_ret);
#endif
	}

	MethodBindTR(R (MB_T::*p_method)(P...)) {
		method = p_method;
		_set_returns(true);
#ifdef DEBUG_METHODS_ENABLED
		_generate_argument_types(sizeof...(P));
#endif
		set_argument_count(sizeof...(P));
	}
};

template <class T, class R, class... P>
MethodBind *create_method_bind(R (T::*p_method)(P...)) {
#ifdef TYPED_METHOD_BIND
	MethodBind *a = memnew((MethodBindTR<T, R, P...>)(p_method));
#else
	MethodBind *a = memnew((MethodBindTR<R, P...>)(reinterpret_cast<R (MB_T::*)(P...)>(p_method)));
#endif

	a->set_instance_class(T::get_class_static());
	return a;
}

// return, const

#ifdef TYPED_METHOD_BIND
template <class T, class R, class... P>
#else
template <class R, class... P>
#endif
class MethodBindTRC : public MethodBind {
	R(MB_T::*method)
	(P...) const;

protected:
#ifdef DEBUG_METHODS_ENABLED
// GCC raises warnings in the case P = {} as the comparison is always false...
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wlogical-op"
#endif
	virtual Variant::Type _gen_argument_type(int p_arg) const {
		if (p_arg >= 0 && p_arg < (int)sizeof...(P)) {
			return call_get_argument_type<P...>(p_arg);
		} else {
			return GetTypeInfo<R>::VARIANT_TYPE;
		}
	}

	virtual PropertyInfo _gen_argument_type_info(int p_arg) const {
		if (p_arg >= 0 && p_arg < (int)sizeof...(P)) {
			PropertyInfo pi;
			call_get_argument_type_info<P...>(p_arg, pi);
			return pi;
		} else {
			return GetTypeInfo<R>::get_class_info();
		}
	}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
#endif

public:
#ifdef DEBUG_METHODS_ENABLED
	virtual GodotTypeInfo::Metadata get_argument_meta(int p_arg) const {
		if (p_arg >= 0) {
			return call_get_argument_metadata<P...>(p_arg);
		} else {
			return GetTypeInfo<R>::METADATA;
		}
	}
#endif

	virtual Variant call(Object *p_object, const Variant **p_args, int p_arg_count, Callable::CallError &r_error) {
		Variant ret;
#ifdef TYPED_METHOD_BIND
		call_with_variant_args_retc_dv(static_cast<T *>(p_object), method, p_args, p_arg_count, ret, r_error, get_default_arguments());
#else
		call_with_variant_args_retc_dv((MB_T *)(p_object), method, p_args, p_arg_count, ret, r_error, get_default_arguments());
#endif
		return ret;
	}

	virtual void ptrcall(Object *p_object, const void **p_args, void *r_ret) {
#ifdef TYPED_METHOD_BIND
		call_with_ptr_args_retc<T, R, P...>(static_cast<T *>(p_object), method, p_args, r_ret);
#else
		call_with_ptr_args_retc<MB_T, R, P...>((MB_T *)(p_object), method, p_args, r_ret);
#endif
	}

	MethodBindTRC(R (MB_T::*p_method)(P...) const) {
		method = p_method;
		_set_returns(true);
		_set_const(true);
#ifdef DEBUG_METHODS_ENABLED
		_generate_argument_types(sizeof...(P));
#endif
		set_argument_count(sizeof...(P));
	}
};

template <class T, class R, class... P>
MethodBind *create_method_bind(R (T::*p_method)(P...) const) {
#ifdef TYPED_METHOD_BIND
	MethodBind *a = memnew((MethodBindTRC<T, R, P...>)(p_method));
#else
	MethodBind *a = memnew((MethodBindTRC<R, P...>)(reinterpret_cast<R (MB_T::*)(P...) const>(p_method)));
#endif
	a->set_instance_class(T::get_class_static());
	return a;
}

#endif // METHOD_BIND_H
