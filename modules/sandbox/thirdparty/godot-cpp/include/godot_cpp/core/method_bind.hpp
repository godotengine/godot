/**************************************************************************/
/*  method_bind.hpp                                                       */
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

#include <godot_cpp/core/binder_common.hpp>
#include <godot_cpp/core/type_info.hpp>

#include <godot_cpp/core/memory.hpp>

#include <gdextension_interface.h>

#include <godot_cpp/classes/global_constants.hpp>

#include <string>
#include <vector>

#include <iostream>

namespace godot {

class MethodBind {
	uint32_t hint_flags = METHOD_FLAGS_DEFAULT;
	StringName name;
	StringName instance_class;
	int argument_count = 0;

	bool _static = false;
	bool _const = false;
	bool _returns = false;
	bool _vararg = false;

	std::vector<StringName> argument_names;
	GDExtensionVariantType *argument_types = nullptr;
	std::vector<Variant> default_arguments;

protected:
	void _set_const(bool p_const);
	void _set_static(bool p_static);
	void _set_returns(bool p_returns);
	void _set_vararg(bool p_vararg);
	virtual GDExtensionVariantType gen_argument_type(int p_arg) const = 0;
	virtual PropertyInfo gen_argument_type_info(int p_arg) const = 0;
	void _generate_argument_types(int p_count);

	void set_argument_count(int p_count) { argument_count = p_count; }

public:
	_FORCE_INLINE_ const std::vector<Variant> &get_default_arguments() const { return default_arguments; }
	_FORCE_INLINE_ int get_default_argument_count() const { return (int)default_arguments.size(); }

	_FORCE_INLINE_ Variant has_default_argument(int p_arg) const {
		const int num_default_args = (int)(default_arguments.size());
		const int idx = p_arg - (argument_count - num_default_args);

		if (idx < 0 || idx >= num_default_args) {
			return false;
		} else {
			return true;
		}
	}
	_FORCE_INLINE_ Variant get_default_argument(int p_arg) const {
		const int num_default_args = (int)(default_arguments.size());
		const int idx = p_arg - (argument_count - num_default_args);

		if (idx < 0 || idx >= num_default_args) {
			return Variant();
		} else {
			return default_arguments[idx];
		}
	}

	_FORCE_INLINE_ GDExtensionVariantType get_argument_type(int p_argument) const {
		ERR_FAIL_COND_V(p_argument < -1 || p_argument > argument_count, GDEXTENSION_VARIANT_TYPE_NIL);
		return argument_types[p_argument + 1];
	}

	PropertyInfo get_argument_info(int p_argument) const;

	std::vector<PropertyInfo> get_arguments_info_list() const {
		std::vector<PropertyInfo> vec;
		// First element is return value
		vec.reserve(argument_count + 1);
		for (int i = 0; i < argument_count + 1; i++) {
			vec.push_back(get_argument_info(i - 1));
		}
		return vec;
	}

	void set_argument_names(const std::vector<StringName> &p_names);
	std::vector<StringName> get_argument_names() const;

	virtual GDExtensionClassMethodArgumentMetadata get_argument_metadata(int p_argument) const = 0;

	_FORCE_INLINE_ void set_hint_flags(uint32_t p_hint_flags) { hint_flags = p_hint_flags; }
	_FORCE_INLINE_ uint32_t get_hint_flags() const { return hint_flags | (is_const() ? GDEXTENSION_METHOD_FLAG_CONST : 0) | (is_vararg() ? GDEXTENSION_METHOD_FLAG_VARARG : 0) | (is_static() ? GDEXTENSION_METHOD_FLAG_STATIC : 0); }
	_FORCE_INLINE_ StringName get_instance_class() const { return instance_class; }
	_FORCE_INLINE_ void set_instance_class(StringName p_class) { instance_class = p_class; }

	_FORCE_INLINE_ int get_argument_count() const { return argument_count; }

	virtual Variant call(GDExtensionClassInstancePtr p_instance, const GDExtensionConstVariantPtr *p_args, GDExtensionInt p_argument_count, GDExtensionCallError &r_error) const = 0;
	virtual void ptrcall(GDExtensionClassInstancePtr p_instance, const GDExtensionConstTypePtr *p_args, GDExtensionTypePtr r_return) const = 0;

	StringName get_name() const;
	void set_name(const StringName &p_name);
	_FORCE_INLINE_ bool is_const() const { return _const; }
	_FORCE_INLINE_ bool is_static() const { return _static; }
	_FORCE_INLINE_ bool is_vararg() const { return _vararg; }
	_FORCE_INLINE_ bool has_return() const { return _returns; }

	void set_default_arguments(const std::vector<Variant> &p_default_arguments) { default_arguments = p_default_arguments; }

	std::vector<GDExtensionClassMethodArgumentMetadata> get_arguments_metadata_list() const {
		std::vector<GDExtensionClassMethodArgumentMetadata> vec;
		// First element is return value
		vec.reserve(argument_count + 1);
		for (int i = 0; i < argument_count + 1; i++) {
			vec.push_back(get_argument_metadata(i - 1));
		}
		return vec;
	}

	static void bind_call(void *p_method_userdata, GDExtensionClassInstancePtr p_instance, const GDExtensionConstVariantPtr *p_args, GDExtensionInt p_argument_count, GDExtensionVariantPtr r_return, GDExtensionCallError *r_error);
	static void bind_ptrcall(void *p_method_userdata, GDExtensionClassInstancePtr p_instance, const GDExtensionConstTypePtr *p_args, GDExtensionTypePtr r_return);

	virtual ~MethodBind();
};

template <typename Derived, typename T, typename R, bool should_returns>
class MethodBindVarArgBase : public MethodBind {
protected:
	R (T::*method)(const Variant **, GDExtensionInt, GDExtensionCallError &);
	std::vector<PropertyInfo> arguments;

public:
	virtual PropertyInfo gen_argument_type_info(int p_arg) const {
		if (p_arg < 0) {
			return _gen_return_type_info();
		} else if ((size_t)(p_arg) < arguments.size()) {
			return arguments[p_arg];
		} else {
			return make_property_info(Variant::Type::NIL, "vararg", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_NIL_IS_VARIANT);
		}
	}

	virtual GDExtensionVariantType gen_argument_type(int p_arg) const {
		return static_cast<GDExtensionVariantType>(gen_argument_type_info(p_arg).type);
	}

	virtual GDExtensionClassMethodArgumentMetadata get_argument_metadata(int) const {
		return GDEXTENSION_METHOD_ARGUMENT_METADATA_NONE;
	}

	virtual void ptrcall(GDExtensionClassInstancePtr p_instance, const GDExtensionConstTypePtr *p_args, GDExtensionTypePtr r_return) const {
		ERR_FAIL(); // Can't call.
	}

	MethodBindVarArgBase(
			R (T::*p_method)(const Variant **, GDExtensionInt, GDExtensionCallError &),
			const MethodInfo &p_method_info,
			bool p_return_nil_is_variant) :
			method(p_method) {
		_set_vararg(true);
		_set_const(true);
		set_argument_count(p_method_info.arguments.size());
		if (p_method_info.arguments.size()) {
			arguments = p_method_info.arguments;

			std::vector<StringName> names;
			names.reserve(p_method_info.arguments.size());
			for (size_t i = 0; i < p_method_info.arguments.size(); i++) {
				names.push_back(p_method_info.arguments[i].name);
			}
			set_argument_names(names);
		}

		_generate_argument_types((int)p_method_info.arguments.size());
		_set_returns(should_returns);
	}

	~MethodBindVarArgBase() {}

private:
	PropertyInfo _gen_return_type_info() const {
		return reinterpret_cast<const Derived *>(this)->_gen_return_type_info_impl();
	}
};

template <typename T>
class MethodBindVarArgT : public MethodBindVarArgBase<MethodBindVarArgT<T>, T, void, false> {
	friend class MethodBindVarArgBase<MethodBindVarArgT<T>, T, void, false>;

public:
	virtual Variant call(GDExtensionClassInstancePtr p_instance, const GDExtensionConstVariantPtr *p_args, GDExtensionInt p_argument_count, GDExtensionCallError &r_error) const {
		(static_cast<T *>(p_instance)->*MethodBindVarArgBase<MethodBindVarArgT<T>, T, void, false>::method)((const Variant **)p_args, p_argument_count, r_error);
		return {};
	}

	MethodBindVarArgT(
			void (T::*p_method)(const Variant **, GDExtensionInt, GDExtensionCallError &),
			const MethodInfo &p_method_info,
			bool p_return_nil_is_variant) :
			MethodBindVarArgBase<MethodBindVarArgT<T>, T, void, false>(p_method, p_method_info, p_return_nil_is_variant) {
	}

private:
	PropertyInfo _gen_return_type_info_impl() const {
		return {};
	}
};

template <typename T>
MethodBind *create_vararg_method_bind(void (T::*p_method)(const Variant **, GDExtensionInt, GDExtensionCallError &), const MethodInfo &p_info, bool p_return_nil_is_variant) {
	MethodBind *a = memnew((MethodBindVarArgT<T>)(p_method, p_info, p_return_nil_is_variant));
	a->set_instance_class(T::get_class_static());
	return a;
}

template <typename T, typename R>
class MethodBindVarArgTR : public MethodBindVarArgBase<MethodBindVarArgTR<T, R>, T, R, true> {
	friend class MethodBindVarArgBase<MethodBindVarArgTR<T, R>, T, R, true>;

public:
	virtual Variant call(GDExtensionClassInstancePtr p_instance, const GDExtensionConstVariantPtr *p_args, GDExtensionInt p_argument_count, GDExtensionCallError &r_error) const {
		return (static_cast<T *>(p_instance)->*MethodBindVarArgBase<MethodBindVarArgTR<T, R>, T, R, true>::method)((const Variant **)p_args, p_argument_count, r_error);
	}

	MethodBindVarArgTR(
			R (T::*p_method)(const Variant **, GDExtensionInt, GDExtensionCallError &),
			const MethodInfo &p_info,
			bool p_return_nil_is_variant) :
			MethodBindVarArgBase<MethodBindVarArgTR<T, R>, T, R, true>(p_method, p_info, p_return_nil_is_variant) {
	}

private:
	PropertyInfo _gen_return_type_info_impl() const {
		return GetTypeInfo<R>::get_class_info();
	}
};

template <typename T, typename R>
MethodBind *create_vararg_method_bind(R (T::*p_method)(const Variant **, GDExtensionInt, GDExtensionCallError &), const MethodInfo &p_info, bool p_return_nil_is_variant) {
	MethodBind *a = memnew((MethodBindVarArgTR<T, R>)(p_method, p_info, p_return_nil_is_variant));
	a->set_instance_class(T::get_class_static());
	return a;
}

#ifndef TYPED_METHOD_BIND
class _gde_UnexistingClass;
#define MB_T _gde_UnexistingClass
#else
#define MB_T T
#endif

// No return, not const.

#ifdef TYPED_METHOD_BIND
template <typename T, typename... P>
#else
template <typename... P>
#endif // TYPED_METHOD_BIND
class MethodBindT : public MethodBind {
	void (MB_T::*method)(P...);

protected:
// GCC raises warnings in the case P = {} as the comparison is always false...
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wlogical-op"
#endif
	virtual GDExtensionVariantType gen_argument_type(int p_arg) const {
		if (p_arg >= 0 && p_arg < (int)sizeof...(P)) {
			return call_get_argument_type<P...>(p_arg);
		} else {
			return GDEXTENSION_VARIANT_TYPE_NIL;
		}
	}

	virtual PropertyInfo gen_argument_type_info(int p_arg) const {
		PropertyInfo pi;
		if (p_arg >= 0 && p_arg < (int)sizeof...(P)) {
			call_get_argument_type_info<P...>(p_arg, pi);
		} else {
			pi = PropertyInfo();
		}
		return pi;
	}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

public:
	virtual GDExtensionClassMethodArgumentMetadata get_argument_metadata(int p_argument) const {
		return call_get_argument_metadata<P...>(p_argument);
	}

	virtual Variant call(GDExtensionClassInstancePtr p_instance, const GDExtensionConstVariantPtr *p_args, GDExtensionInt p_argument_count, GDExtensionCallError &r_error) const {
#ifdef TYPED_METHOD_BIND
		call_with_variant_args_dv(static_cast<T *>(p_instance), method, p_args, (int)p_argument_count, r_error, get_default_arguments());
#else
		call_with_variant_args_dv(reinterpret_cast<MB_T *>(p_instance), method, p_args, p_argument_count, r_error, get_default_arguments());
#endif
		return Variant();
	}
	virtual void ptrcall(GDExtensionClassInstancePtr p_instance, const GDExtensionConstTypePtr *p_args, GDExtensionTypePtr r_ret) const {
#ifdef TYPED_METHOD_BIND
		call_with_ptr_args<T, P...>(static_cast<T *>(p_instance), method, p_args, nullptr);
#else
		call_with_ptr_args<MB_T, P...>(reinterpret_cast<MB_T *>(p_instance), method, p_args, nullptr);
#endif // TYPED_METHOD_BIND
	}

	MethodBindT(void (MB_T::*p_method)(P...)) {
		method = p_method;
		_generate_argument_types(sizeof...(P));
		set_argument_count(sizeof...(P));
	}
};

template <typename T, typename... P>
MethodBind *create_method_bind(void (T::*p_method)(P...)) {
#ifdef TYPED_METHOD_BIND
	MethodBind *a = memnew((MethodBindT<T, P...>)(p_method));
#else
	MethodBind *a = memnew((MethodBindT<P...>)(reinterpret_cast<void (MB_T::*)(P...)>(p_method)));
#endif // TYPED_METHOD_BIND
	a->set_instance_class(T::get_class_static());
	return a;
}

// No return, const.

#ifdef TYPED_METHOD_BIND
template <typename T, typename... P>
#else
template <typename... P>
#endif // TYPED_METHOD_BIND
class MethodBindTC : public MethodBind {
	void (MB_T::*method)(P...) const;

protected:
// GCC raises warnings in the case P = {} as the comparison is always false...
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wlogical-op"
#endif
	virtual GDExtensionVariantType gen_argument_type(int p_arg) const {
		if (p_arg >= 0 && p_arg < (int)sizeof...(P)) {
			return call_get_argument_type<P...>(p_arg);
		} else {
			return GDEXTENSION_VARIANT_TYPE_NIL;
		}
	}

	virtual PropertyInfo gen_argument_type_info(int p_arg) const {
		PropertyInfo pi;
		if (p_arg >= 0 && p_arg < (int)sizeof...(P)) {
			call_get_argument_type_info<P...>(p_arg, pi);
		} else {
			pi = PropertyInfo();
		}
		return pi;
	}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

public:
	virtual GDExtensionClassMethodArgumentMetadata get_argument_metadata(int p_argument) const {
		return call_get_argument_metadata<P...>(p_argument);
	}

	virtual Variant call(GDExtensionClassInstancePtr p_instance, const GDExtensionConstVariantPtr *p_args, GDExtensionInt p_argument_count, GDExtensionCallError &r_error) const {
#ifdef TYPED_METHOD_BIND
		call_with_variant_argsc_dv(static_cast<T *>(p_instance), method, p_args, (int)p_argument_count, r_error, get_default_arguments());
#else
		call_with_variant_argsc_dv(reinterpret_cast<MB_T *>(p_instance), method, p_args, p_argument_count, r_error, get_default_arguments());
#endif
		return Variant();
	}
	virtual void ptrcall(GDExtensionClassInstancePtr p_instance, const GDExtensionConstTypePtr *p_args, GDExtensionTypePtr r_ret) const {
#ifdef TYPED_METHOD_BIND
		call_with_ptr_args<T, P...>(static_cast<T *>(p_instance), method, p_args, nullptr);
#else
		call_with_ptr_args<MB_T, P...>(reinterpret_cast<MB_T *>(p_instance), method, p_args, nullptr);
#endif // TYPED_METHOD_BIND
	}

	MethodBindTC(void (MB_T::*p_method)(P...) const) {
		method = p_method;
		_generate_argument_types(sizeof...(P));
		set_argument_count(sizeof...(P));
		_set_const(true);
	}
};

template <typename T, typename... P>
MethodBind *create_method_bind(void (T::*p_method)(P...) const) {
#ifdef TYPED_METHOD_BIND
	MethodBind *a = memnew((MethodBindTC<T, P...>)(p_method));
#else
	MethodBind *a = memnew((MethodBindTC<P...>)(reinterpret_cast<void (MB_T::*)(P...) const>(p_method)));
#endif // TYPED_METHOD_BIND
	a->set_instance_class(T::get_class_static());
	return a;
}

// Return, not const.

#ifdef TYPED_METHOD_BIND
template <typename T, typename R, typename... P>
#else
template <typename R, typename... P>
#endif // TYPED_METHOD_BIND
class MethodBindTR : public MethodBind {
	R (MB_T::*method)(P...);

protected:
// GCC raises warnings in the case P = {} as the comparison is always false...
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wlogical-op"
#endif
	virtual GDExtensionVariantType gen_argument_type(int p_arg) const {
		if (p_arg >= 0 && p_arg < (int)sizeof...(P)) {
			return call_get_argument_type<P...>(p_arg);
		} else {
			return GDExtensionVariantType(GetTypeInfo<R>::VARIANT_TYPE);
		}
	}

	virtual PropertyInfo gen_argument_type_info(int p_arg) const {
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

public:
	virtual GDExtensionClassMethodArgumentMetadata get_argument_metadata(int p_argument) const {
		if (p_argument >= 0) {
			return call_get_argument_metadata<P...>(p_argument);
		} else {
			return GetTypeInfo<R>::METADATA;
		}
	}

	virtual Variant call(GDExtensionClassInstancePtr p_instance, const GDExtensionConstVariantPtr *p_args, GDExtensionInt p_argument_count, GDExtensionCallError &r_error) const {
		Variant ret;
#ifdef TYPED_METHOD_BIND
		call_with_variant_args_ret_dv(static_cast<T *>(p_instance), method, p_args, (int)p_argument_count, ret, r_error, get_default_arguments());
#else
		call_with_variant_args_ret_dv((MB_T *)p_instance, method, p_args, p_argument_count, ret, r_error, get_default_arguments());
#endif
		return ret;
	}
	virtual void ptrcall(GDExtensionClassInstancePtr p_instance, const GDExtensionConstTypePtr *p_args, GDExtensionTypePtr r_ret) const {
#ifdef TYPED_METHOD_BIND
		call_with_ptr_args<T, R, P...>(static_cast<T *>(p_instance), method, p_args, r_ret);
#else
		call_with_ptr_args<MB_T, R, P...>(reinterpret_cast<MB_T *>(p_instance), method, p_args, r_ret);
#endif // TYPED_METHOD_BIND
	}

	MethodBindTR(R (MB_T::*p_method)(P...)) {
		method = p_method;
		_generate_argument_types(sizeof...(P));
		set_argument_count(sizeof...(P));
		_set_returns(true);
	}
};

template <typename T, typename R, typename... P>
MethodBind *create_method_bind(R (T::*p_method)(P...)) {
#ifdef TYPED_METHOD_BIND
	MethodBind *a = memnew((MethodBindTR<T, R, P...>)(p_method));
#else
	MethodBind *a = memnew((MethodBindTR<R, P...>)(reinterpret_cast<R (MB_T::*)(P...)>(p_method)));
#endif // TYPED_METHOD_BIND
	a->set_instance_class(T::get_class_static());
	return a;
}

// Return, const.

#ifdef TYPED_METHOD_BIND
template <typename T, typename R, typename... P>
#else
template <typename R, typename... P>
#endif // TYPED_METHOD_BIND
class MethodBindTRC : public MethodBind {
	R (MB_T::*method)(P...) const;

protected:
// GCC raises warnings in the case P = {} as the comparison is always false...
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wlogical-op"
#endif
	virtual GDExtensionVariantType gen_argument_type(int p_arg) const {
		if (p_arg >= 0 && p_arg < (int)sizeof...(P)) {
			return call_get_argument_type<P...>(p_arg);
		} else {
			return GDExtensionVariantType(GetTypeInfo<R>::VARIANT_TYPE);
		}
	}

	virtual PropertyInfo gen_argument_type_info(int p_arg) const {
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

public:
	virtual GDExtensionClassMethodArgumentMetadata get_argument_metadata(int p_argument) const {
		if (p_argument >= 0) {
			return call_get_argument_metadata<P...>(p_argument);
		} else {
			return GetTypeInfo<R>::METADATA;
		}
	}

	virtual Variant call(GDExtensionClassInstancePtr p_instance, const GDExtensionConstVariantPtr *p_args, GDExtensionInt p_argument_count, GDExtensionCallError &r_error) const {
		Variant ret;
#ifdef TYPED_METHOD_BIND
		call_with_variant_args_retc_dv(static_cast<T *>(p_instance), method, p_args, (int)p_argument_count, ret, r_error, get_default_arguments());
#else
		call_with_variant_args_retc_dv((MB_T *)p_instance, method, p_args, p_argument_count, ret, r_error, get_default_arguments());
#endif
		return ret;
	}
	virtual void ptrcall(GDExtensionClassInstancePtr p_instance, const GDExtensionConstTypePtr *p_args, GDExtensionTypePtr r_ret) const {
#ifdef TYPED_METHOD_BIND
		call_with_ptr_args<T, R, P...>(static_cast<T *>(p_instance), method, p_args, r_ret);
#else
		call_with_ptr_args<MB_T, R, P...>(reinterpret_cast<MB_T *>(p_instance), method, p_args, r_ret);
#endif // TYPED_METHOD_BIND
	}

	MethodBindTRC(R (MB_T::*p_method)(P...) const) {
		method = p_method;
		_generate_argument_types(sizeof...(P));
		set_argument_count(sizeof...(P));
		_set_returns(true);
		_set_const(true);
	}
};

template <typename T, typename R, typename... P>
MethodBind *create_method_bind(R (T::*p_method)(P...) const) {
#ifdef TYPED_METHOD_BIND
	MethodBind *a = memnew((MethodBindTRC<T, R, P...>)(p_method));
#else
	MethodBind *a = memnew((MethodBindTRC<R, P...>)(reinterpret_cast<R (MB_T::*)(P...) const>(p_method)));
#endif // TYPED_METHOD_BIND
	a->set_instance_class(T::get_class_static());
	return a;
}

// STATIC BINDS

// no return

template <typename... P>
class MethodBindTS : public MethodBind {
	void (*function)(P...);

protected:
// GCC raises warnings in the case P = {} as the comparison is always false...
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wlogical-op"
#endif
	virtual GDExtensionVariantType gen_argument_type(int p_arg) const {
		if (p_arg >= 0 && p_arg < (int)sizeof...(P)) {
			return call_get_argument_type<P...>(p_arg);
		} else {
			return GDEXTENSION_VARIANT_TYPE_NIL;
		}
	}

	virtual PropertyInfo gen_argument_type_info(int p_arg) const {
		PropertyInfo pi;
		if (p_arg >= 0 && p_arg < (int)sizeof...(P)) {
			call_get_argument_type_info<P...>(p_arg, pi);
		} else {
			pi = PropertyInfo();
		}
		return pi;
	}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

public:
	virtual GDExtensionClassMethodArgumentMetadata get_argument_metadata(int p_arg) const {
		return call_get_argument_metadata<P...>(p_arg);
	}

	virtual Variant call(GDExtensionClassInstancePtr p_object, const GDExtensionConstVariantPtr *p_args, GDExtensionInt p_arg_count, GDExtensionCallError &r_error) const {
		(void)p_object; // unused
		call_with_variant_args_static_dv(function, p_args, p_arg_count, r_error, get_default_arguments());
		return Variant();
	}

	virtual void ptrcall(GDExtensionClassInstancePtr p_object, const GDExtensionConstTypePtr *p_args, GDExtensionTypePtr r_ret) const {
		(void)p_object;
		(void)r_ret;
		call_with_ptr_args_static_method(function, p_args);
	}

	MethodBindTS(void (*p_function)(P...)) {
		function = p_function;
		_generate_argument_types(sizeof...(P));
		set_argument_count(sizeof...(P));
		_set_static(true);
	}
};

template <typename... P>
MethodBind *create_static_method_bind(void (*p_method)(P...)) {
	MethodBind *a = memnew((MethodBindTS<P...>)(p_method));
	return a;
}

// return

template <typename R, typename... P>
class MethodBindTRS : public MethodBind {
	R (*function)(P...);

protected:
// GCC raises warnings in the case P = {} as the comparison is always false...
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wlogical-op"
#endif
	virtual GDExtensionVariantType gen_argument_type(int p_arg) const {
		if (p_arg >= 0 && p_arg < (int)sizeof...(P)) {
			return call_get_argument_type<P...>(p_arg);
		} else {
			return GDExtensionVariantType(GetTypeInfo<R>::VARIANT_TYPE);
		}
	}

	virtual PropertyInfo gen_argument_type_info(int p_arg) const {
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

public:
	virtual GDExtensionClassMethodArgumentMetadata get_argument_metadata(int p_arg) const {
		if (p_arg >= 0) {
			return call_get_argument_metadata<P...>(p_arg);
		} else {
			return GetTypeInfo<R>::METADATA;
		}
	}

	virtual Variant call(GDExtensionClassInstancePtr p_object, const GDExtensionConstVariantPtr *p_args, GDExtensionInt p_arg_count, GDExtensionCallError &r_error) const {
		Variant ret;
		call_with_variant_args_static_ret_dv(function, p_args, p_arg_count, ret, r_error, get_default_arguments());
		return ret;
	}

	virtual void ptrcall(GDExtensionClassInstancePtr p_object, const GDExtensionConstTypePtr *p_args, GDExtensionTypePtr r_ret) const {
		(void)p_object;
		call_with_ptr_args_static_method_ret(function, p_args, r_ret);
	}

	MethodBindTRS(R (*p_function)(P...)) {
		function = p_function;
		_generate_argument_types(sizeof...(P));
		set_argument_count(sizeof...(P));
		_set_static(true);
		_set_returns(true);
	}
};

template <typename R, typename... P>
MethodBind *create_static_method_bind(R (*p_method)(P...)) {
	MethodBind *a = memnew((MethodBindTRS<R, P...>)(p_method));
	return a;
}

} // namespace godot
