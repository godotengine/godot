/**************************************************************************/
/*  method_bind_common.h                                                  */
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

#include "core/object/method_bind.h"
#include "core/variant/binder_common.h"

VARIANT_BITFIELD_CAST(MethodFlags)

/**** VARIADIC TEMPLATES ****/

#ifndef TYPED_METHOD_BIND
class __UnexistingClass;
#define MB_T __UnexistingClass
#else
#define MB_T T
#endif

// no return, not const
#ifdef TYPED_METHOD_BIND
template <typename T, typename... P>
#else
template <typename... P>
#endif
class MethodBindT : public MethodBind {
	void (MB_T::*method)(P...);

protected:
	virtual Variant::Type _gen_argument_type(int p_arg) const override {
		if (p_arg >= 0 && p_arg < (int)sizeof...(P)) {
			return call_get_argument_type<P...>(p_arg);
		} else {
			return Variant::NIL;
		}
	}

	virtual PropertyInfo _gen_argument_type_info(int p_arg) const override {
		PropertyInfo pi;
		call_get_argument_type_info<P...>(p_arg, pi);
		return pi;
	}

public:
#ifdef DEBUG_ENABLED
	virtual GodotTypeInfo::Metadata get_argument_meta(int p_arg) const override {
		return call_get_argument_metadata<P...>(p_arg);
	}

#endif // DEBUG_ENABLED
	virtual Variant call(Object *p_object, const Variant **p_args, int p_arg_count, Callable::CallError &r_error) const override {
#ifdef TOOLS_ENABLED
		ERR_FAIL_COND_V_MSG(p_object && p_object->is_extension_placeholder() && p_object->get_class_name() == get_instance_class(), Variant(), vformat("Cannot call method bind '%s' on placeholder instance.", MethodBind::get_name()));
#endif
#ifdef TYPED_METHOD_BIND
		call_with_variant_args_dv(static_cast<T *>(p_object), method, p_args, p_arg_count, r_error, get_default_arguments());
#else
		call_with_variant_args_dv(reinterpret_cast<MB_T *>(p_object), method, p_args, p_arg_count, r_error, get_default_arguments());
#endif
		return Variant();
	}

	virtual void validated_call(Object *p_object, const Variant **p_args, Variant *r_ret) const override {
#ifdef TOOLS_ENABLED
		ERR_FAIL_COND_MSG(p_object && p_object->is_extension_placeholder() && p_object->get_class_name() == get_instance_class(), vformat("Cannot call method bind '%s' on placeholder instance.", MethodBind::get_name()));
#endif
#ifdef TYPED_METHOD_BIND
		call_with_validated_object_instance_args(static_cast<T *>(p_object), method, p_args);
#else
		call_with_validated_object_instance_args(reinterpret_cast<MB_T *>(p_object), method, p_args);
#endif
	}

	virtual void ptrcall(Object *p_object, const void **p_args, void *r_ret) const override {
#ifdef TOOLS_ENABLED
		ERR_FAIL_COND_MSG(p_object && p_object->is_extension_placeholder() && p_object->get_class_name() == get_instance_class(), vformat("Cannot call method bind '%s' on placeholder instance.", MethodBind::get_name()));
#endif
#ifdef TYPED_METHOD_BIND
		call_with_ptr_args<T, P...>(static_cast<T *>(p_object), method, p_args);
#else
		call_with_ptr_args<MB_T, P...>(reinterpret_cast<MB_T *>(p_object), method, p_args);
#endif
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
#endif
	a->set_instance_class(T::get_class_static());
	return a;
}

// no return, const

#ifdef TYPED_METHOD_BIND
template <typename T, typename... P>
#else
template <typename... P>
#endif
class MethodBindTC : public MethodBind {
	void (MB_T::*method)(P...) const;

protected:
	virtual Variant::Type _gen_argument_type(int p_arg) const override {
		if (p_arg >= 0 && p_arg < (int)sizeof...(P)) {
			return call_get_argument_type<P...>(p_arg);
		} else {
			return Variant::NIL;
		}
	}

	virtual PropertyInfo _gen_argument_type_info(int p_arg) const override {
		PropertyInfo pi;
		call_get_argument_type_info<P...>(p_arg, pi);
		return pi;
	}

public:
#ifdef DEBUG_ENABLED
	virtual GodotTypeInfo::Metadata get_argument_meta(int p_arg) const override {
		return call_get_argument_metadata<P...>(p_arg);
	}

#endif // DEBUG_ENABLED
	virtual Variant call(Object *p_object, const Variant **p_args, int p_arg_count, Callable::CallError &r_error) const override {
#ifdef TOOLS_ENABLED
		ERR_FAIL_COND_V_MSG(p_object && p_object->is_extension_placeholder() && p_object->get_class_name() == get_instance_class(), Variant(), vformat("Cannot call method bind '%s' on placeholder instance.", MethodBind::get_name()));
#endif
#ifdef TYPED_METHOD_BIND
		call_with_variant_argsc_dv(static_cast<T *>(p_object), method, p_args, p_arg_count, r_error, get_default_arguments());
#else
		call_with_variant_argsc_dv(reinterpret_cast<MB_T *>(p_object), method, p_args, p_arg_count, r_error, get_default_arguments());
#endif
		return Variant();
	}

	virtual void validated_call(Object *p_object, const Variant **p_args, Variant *r_ret) const override {
#ifdef TOOLS_ENABLED
		ERR_FAIL_COND_MSG(p_object && p_object->is_extension_placeholder() && p_object->get_class_name() == get_instance_class(), vformat("Cannot call method bind '%s' on placeholder instance.", MethodBind::get_name()));
#endif
#ifdef TYPED_METHOD_BIND
		call_with_validated_object_instance_argsc(static_cast<T *>(p_object), method, p_args);
#else
		call_with_validated_object_instance_argsc(reinterpret_cast<MB_T *>(p_object), method, p_args);
#endif
	}

	virtual void ptrcall(Object *p_object, const void **p_args, void *r_ret) const override {
#ifdef TOOLS_ENABLED
		ERR_FAIL_COND_MSG(p_object && p_object->is_extension_placeholder() && p_object->get_class_name() == get_instance_class(), vformat("Cannot call method bind '%s' on placeholder instance.", MethodBind::get_name()));
#endif
#ifdef TYPED_METHOD_BIND
		call_with_ptr_argsc<T, P...>(static_cast<T *>(p_object), method, p_args);
#else
		call_with_ptr_argsc<MB_T, P...>(reinterpret_cast<MB_T *>(p_object), method, p_args);
#endif
	}

	MethodBindTC(void (MB_T::*p_method)(P...) const) {
		method = p_method;
		_set_const(true);
		_generate_argument_types(sizeof...(P));
		set_argument_count(sizeof...(P));
	}
};

template <typename T, typename... P>
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
template <typename T, typename R, typename... P>
#else
template <typename R, typename... P>
#endif
class MethodBindTR : public MethodBind {
	R (MB_T::*method)(P...);

protected:
	virtual Variant::Type _gen_argument_type(int p_arg) const override {
		if (p_arg >= 0 && p_arg < (int)sizeof...(P)) {
			return call_get_argument_type<P...>(p_arg);
		} else {
			return GetTypeInfo<R>::VARIANT_TYPE;
		}
	}

	virtual PropertyInfo _gen_argument_type_info(int p_arg) const override {
		if (p_arg >= 0 && p_arg < (int)sizeof...(P)) {
			PropertyInfo pi;
			call_get_argument_type_info<P...>(p_arg, pi);
			return pi;
		} else {
			return GetTypeInfo<R>::get_class_info();
		}
	}

public:
#ifdef DEBUG_ENABLED
	virtual GodotTypeInfo::Metadata get_argument_meta(int p_arg) const override {
		if (p_arg >= 0) {
			return call_get_argument_metadata<P...>(p_arg);
		} else {
			return GetTypeInfo<R>::METADATA;
		}
	}
#endif // DEBUG_ENABLED

	virtual Variant call(Object *p_object, const Variant **p_args, int p_arg_count, Callable::CallError &r_error) const override {
		Variant ret;
#ifdef TOOLS_ENABLED
		ERR_FAIL_COND_V_MSG(p_object && p_object->is_extension_placeholder() && p_object->get_class_name() == get_instance_class(), ret, vformat("Cannot call method bind '%s' on placeholder instance.", MethodBind::get_name()));
#endif
#ifdef TYPED_METHOD_BIND
		call_with_variant_args_ret_dv(static_cast<T *>(p_object), method, p_args, p_arg_count, ret, r_error, get_default_arguments());
#else
		call_with_variant_args_ret_dv(reinterpret_cast<MB_T *>(p_object), method, p_args, p_arg_count, ret, r_error, get_default_arguments());
#endif
		return ret;
	}

	virtual void validated_call(Object *p_object, const Variant **p_args, Variant *r_ret) const override {
#ifdef TOOLS_ENABLED
		ERR_FAIL_COND_MSG(p_object && p_object->is_extension_placeholder() && p_object->get_class_name() == get_instance_class(), vformat("Cannot call method bind '%s' on placeholder instance.", MethodBind::get_name()));
#endif
#ifdef TYPED_METHOD_BIND
		call_with_validated_object_instance_args_ret(static_cast<T *>(p_object), method, p_args, r_ret);
#else
		call_with_validated_object_instance_args_ret(reinterpret_cast<MB_T *>(p_object), method, p_args, r_ret);
#endif
	}

	virtual void ptrcall(Object *p_object, const void **p_args, void *r_ret) const override {
#ifdef TOOLS_ENABLED
		ERR_FAIL_COND_MSG(p_object && p_object->is_extension_placeholder() && p_object->get_class_name() == get_instance_class(), vformat("Cannot call method bind '%s' on placeholder instance.", MethodBind::get_name()));
#endif
#ifdef TYPED_METHOD_BIND
		call_with_ptr_args_ret<T, R, P...>(static_cast<T *>(p_object), method, p_args, r_ret);
#else
		call_with_ptr_args_ret<MB_T, R, P...>(reinterpret_cast<MB_T *>(p_object), method, p_args, r_ret);
#endif
	}

	MethodBindTR(R (MB_T::*p_method)(P...)) {
		method = p_method;
		_set_returns(true);
		_generate_argument_types(sizeof...(P));
		set_argument_count(sizeof...(P));
	}
};

template <typename T, typename R, typename... P>
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
template <typename T, typename R, typename... P>
#else
template <typename R, typename... P>
#endif
class MethodBindTRC : public MethodBind {
	R (MB_T::*method)(P...) const;

protected:
	virtual Variant::Type _gen_argument_type(int p_arg) const override {
		if (p_arg >= 0 && p_arg < (int)sizeof...(P)) {
			return call_get_argument_type<P...>(p_arg);
		} else {
			return GetTypeInfo<R>::VARIANT_TYPE;
		}
	}

	virtual PropertyInfo _gen_argument_type_info(int p_arg) const override {
		if (p_arg >= 0 && p_arg < (int)sizeof...(P)) {
			PropertyInfo pi;
			call_get_argument_type_info<P...>(p_arg, pi);
			return pi;
		} else {
			return GetTypeInfo<R>::get_class_info();
		}
	}

public:
#ifdef DEBUG_ENABLED
	virtual GodotTypeInfo::Metadata get_argument_meta(int p_arg) const override {
		if (p_arg >= 0) {
			return call_get_argument_metadata<P...>(p_arg);
		} else {
			return GetTypeInfo<R>::METADATA;
		}
	}
#endif // DEBUG_ENABLED

	virtual Variant call(Object *p_object, const Variant **p_args, int p_arg_count, Callable::CallError &r_error) const override {
		Variant ret;
#ifdef TOOLS_ENABLED
		ERR_FAIL_COND_V_MSG(p_object && p_object->is_extension_placeholder() && p_object->get_class_name() == get_instance_class(), ret, vformat("Cannot call method bind '%s' on placeholder instance.", MethodBind::get_name()));
#endif
#ifdef TYPED_METHOD_BIND
		call_with_variant_args_retc_dv(static_cast<T *>(p_object), method, p_args, p_arg_count, ret, r_error, get_default_arguments());
#else
		call_with_variant_args_retc_dv(reinterpret_cast<MB_T *>(p_object), method, p_args, p_arg_count, ret, r_error, get_default_arguments());
#endif
		return ret;
	}

	virtual void validated_call(Object *p_object, const Variant **p_args, Variant *r_ret) const override {
#ifdef TOOLS_ENABLED
		ERR_FAIL_COND_MSG(p_object && p_object->is_extension_placeholder() && p_object->get_class_name() == get_instance_class(), vformat("Cannot call method bind '%s' on placeholder instance.", MethodBind::get_name()));
#endif
#ifdef TYPED_METHOD_BIND
		call_with_validated_object_instance_args_retc(static_cast<T *>(p_object), method, p_args, r_ret);
#else
		call_with_validated_object_instance_args_retc(reinterpret_cast<MB_T *>(p_object), method, p_args, r_ret);
#endif
	}

	virtual void ptrcall(Object *p_object, const void **p_args, void *r_ret) const override {
#ifdef TOOLS_ENABLED
		ERR_FAIL_COND_MSG(p_object && p_object->is_extension_placeholder() && p_object->get_class_name() == get_instance_class(), vformat("Cannot call method bind '%s' on placeholder instance.", MethodBind::get_name()));
#endif
#ifdef TYPED_METHOD_BIND
		call_with_ptr_args_retc<T, R, P...>(static_cast<T *>(p_object), method, p_args, r_ret);
#else
		call_with_ptr_args_retc<MB_T, R, P...>(reinterpret_cast<MB_T *>(p_object), method, p_args, r_ret);
#endif
	}

	MethodBindTRC(R (MB_T::*p_method)(P...) const) {
		method = p_method;
		_set_returns(true);
		_set_const(true);
		_generate_argument_types(sizeof...(P));
		set_argument_count(sizeof...(P));
	}
};

template <typename T, typename R, typename... P>
MethodBind *create_method_bind(R (T::*p_method)(P...) const) {
#ifdef TYPED_METHOD_BIND
	MethodBind *a = memnew((MethodBindTRC<T, R, P...>)(p_method));
#else
	MethodBind *a = memnew((MethodBindTRC<R, P...>)(reinterpret_cast<R (MB_T::*)(P...) const>(p_method)));
#endif
	a->set_instance_class(T::get_class_static());
	return a;
}

/* STATIC BINDS */

// no return

template <typename... P>
class MethodBindTS : public MethodBind {
	void (*function)(P...);

protected:
	virtual Variant::Type _gen_argument_type(int p_arg) const override {
		if (p_arg >= 0 && p_arg < (int)sizeof...(P)) {
			return call_get_argument_type<P...>(p_arg);
		} else {
			return Variant::NIL;
		}
	}

	virtual PropertyInfo _gen_argument_type_info(int p_arg) const override {
		PropertyInfo pi;
		call_get_argument_type_info<P...>(p_arg, pi);
		return pi;
	}

public:
#ifdef DEBUG_ENABLED
	virtual GodotTypeInfo::Metadata get_argument_meta(int p_arg) const override {
		return call_get_argument_metadata<P...>(p_arg);
	}

#endif // DEBUG_ENABLED
	virtual Variant call(Object *p_object, const Variant **p_args, int p_arg_count, Callable::CallError &r_error) const override {
		(void)p_object; // unused
		call_with_variant_args_static_dv(function, p_args, p_arg_count, r_error, get_default_arguments());
		return Variant();
	}

	virtual void validated_call(Object *p_object, const Variant **p_args, Variant *r_ret) const override {
		call_with_validated_variant_args_static_method(function, p_args);
	}

	virtual void ptrcall(Object *p_object, const void **p_args, void *r_ret) const override {
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
	virtual Variant::Type _gen_argument_type(int p_arg) const override {
		if (p_arg >= 0 && p_arg < (int)sizeof...(P)) {
			return call_get_argument_type<P...>(p_arg);
		} else {
			return GetTypeInfo<R>::VARIANT_TYPE;
		}
	}

	virtual PropertyInfo _gen_argument_type_info(int p_arg) const override {
		if (p_arg >= 0 && p_arg < (int)sizeof...(P)) {
			PropertyInfo pi;
			call_get_argument_type_info<P...>(p_arg, pi);
			return pi;
		} else {
			return GetTypeInfo<R>::get_class_info();
		}
	}

public:
#ifdef DEBUG_ENABLED
	virtual GodotTypeInfo::Metadata get_argument_meta(int p_arg) const override {
		if (p_arg >= 0) {
			return call_get_argument_metadata<P...>(p_arg);
		} else {
			return GetTypeInfo<R>::METADATA;
		}
	}

#endif // DEBUG_ENABLED
	virtual Variant call(Object *p_object, const Variant **p_args, int p_arg_count, Callable::CallError &r_error) const override {
		Variant ret;
		call_with_variant_args_static_ret_dv(function, p_args, p_arg_count, ret, r_error, get_default_arguments());
		return ret;
	}

	virtual void validated_call(Object *p_object, const Variant **p_args, Variant *r_ret) const override {
		call_with_validated_variant_args_static_method_ret(function, p_args, r_ret);
	}

	virtual void ptrcall(Object *p_object, const void **p_args, void *r_ret) const override {
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
