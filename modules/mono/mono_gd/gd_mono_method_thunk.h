/**************************************************************************/
/*  gd_mono_method_thunk.h                                                */
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

#ifndef GD_MONO_METHOD_THUNK_H
#define GD_MONO_METHOD_THUNK_H

#include <type_traits>

#include "gd_mono_class.h"
#include "gd_mono_header.h"
#include "gd_mono_marshal.h"
#include "gd_mono_method.h"
#include "gd_mono_utils.h"

#if !defined(JAVASCRIPT_ENABLED) && !defined(IPHONE_ENABLED)
#define HAVE_METHOD_THUNKS
#endif

#ifdef HAVE_METHOD_THUNKS

template <class... ParamTypes>
struct GDMonoMethodThunk {
	typedef void(GD_MONO_STDCALL *M)(ParamTypes... p_args, MonoException **);

	M mono_method_thunk;

public:
	_FORCE_INLINE_ void invoke(ParamTypes... p_args, MonoException **r_exc) {
		GD_MONO_BEGIN_RUNTIME_INVOKE;
		mono_method_thunk(p_args..., r_exc);
		GD_MONO_END_RUNTIME_INVOKE;
	}

	_FORCE_INLINE_ bool is_null() {
		return mono_method_thunk == NULL;
	}

	_FORCE_INLINE_ void nullify() {
		mono_method_thunk = NULL;
	}

	_FORCE_INLINE_ void set_from_method(GDMonoMethod *p_mono_method) {
#ifdef DEBUG_ENABLED
		CRASH_COND(p_mono_method == NULL);
		CRASH_COND(p_mono_method->get_return_type().type_encoding != MONO_TYPE_VOID);

		if (p_mono_method->is_static()) {
			CRASH_COND(p_mono_method->get_parameters_count() != sizeof...(ParamTypes));
		} else {
			CRASH_COND(p_mono_method->get_parameters_count() != (sizeof...(ParamTypes) - 1));
		}
#endif
		mono_method_thunk = (M)mono_method_get_unmanaged_thunk(p_mono_method->get_mono_ptr());
	}

	GDMonoMethodThunk() :
			mono_method_thunk(NULL) {
	}

	explicit GDMonoMethodThunk(GDMonoMethod *p_mono_method) {
		set_from_method(p_mono_method);
	}
};

template <class R, class... ParamTypes>
struct GDMonoMethodThunkR {
	typedef R(GD_MONO_STDCALL *M)(ParamTypes... p_args, MonoException **);

	M mono_method_thunk;

public:
	_FORCE_INLINE_ R invoke(ParamTypes... p_args, MonoException **r_exc) {
		GD_MONO_BEGIN_RUNTIME_INVOKE;
		R r = mono_method_thunk(p_args..., r_exc);
		GD_MONO_END_RUNTIME_INVOKE;
		return r;
	}

	_FORCE_INLINE_ bool is_null() {
		return mono_method_thunk == NULL;
	}

	_FORCE_INLINE_ void nullify() {
		mono_method_thunk = NULL;
	}

	_FORCE_INLINE_ void set_from_method(GDMonoMethod *p_mono_method) {
#ifdef DEBUG_ENABLED
		CRASH_COND(p_mono_method == NULL);
		CRASH_COND(p_mono_method->get_return_type().type_encoding == MONO_TYPE_VOID);

		if (p_mono_method->is_static()) {
			CRASH_COND(p_mono_method->get_parameters_count() != sizeof...(ParamTypes));
		} else {
			CRASH_COND(p_mono_method->get_parameters_count() != (sizeof...(ParamTypes) - 1));
		}
#endif
		mono_method_thunk = (M)mono_method_get_unmanaged_thunk(p_mono_method->get_mono_ptr());
	}

	GDMonoMethodThunkR() :
			mono_method_thunk(NULL) {
	}

	explicit GDMonoMethodThunkR(GDMonoMethod *p_mono_method) {
#ifdef DEBUG_ENABLED
		CRASH_COND(p_mono_method == NULL);
#endif
		mono_method_thunk = (M)mono_method_get_unmanaged_thunk(p_mono_method->get_mono_ptr());
	}
};

#else

template <unsigned int ThunkParamCount, class P1, class... ParamTypes>
struct VariadicInvokeMonoMethodImpl {
	static void invoke(GDMonoMethod *p_mono_method, P1 p_arg1, ParamTypes... p_args, MonoException **r_exc) {
		if (p_mono_method->is_static()) {
			void *args[ThunkParamCount] = { p_arg1, p_args... };
			p_mono_method->invoke_raw(NULL, args, r_exc);
		} else {
			void *args[ThunkParamCount] = { p_args... };
			p_mono_method->invoke_raw((MonoObject *)p_arg1, args, r_exc);
		}
	}
};

template <unsigned int ThunkParamCount, class... ParamTypes>
struct VariadicInvokeMonoMethod {
	static void invoke(GDMonoMethod *p_mono_method, ParamTypes... p_args, MonoException **r_exc) {
		VariadicInvokeMonoMethodImpl<ThunkParamCount, ParamTypes...>::invoke(p_mono_method, p_args..., r_exc);
	}
};

template <>
struct VariadicInvokeMonoMethod<0> {
	static void invoke(GDMonoMethod *p_mono_method, MonoException **r_exc) {
#ifdef DEBUG_ENABLED
		CRASH_COND(!p_mono_method->is_static());
#endif
		p_mono_method->invoke_raw(NULL, NULL, r_exc);
	}
};

template <class P1>
struct VariadicInvokeMonoMethod<1, P1> {
	static void invoke(GDMonoMethod *p_mono_method, P1 p_arg1, MonoException **r_exc) {
		if (p_mono_method->is_static()) {
			void *args[1] = { p_arg1 };
			p_mono_method->invoke_raw(NULL, args, r_exc);
		} else {
			p_mono_method->invoke_raw((MonoObject *)p_arg1, NULL, r_exc);
		}
	}
};

template <class R>
R unbox_if_needed(MonoObject *p_val, const ManagedType &, typename std::enable_if<!std::is_pointer<R>::value>::type * = 0) {
	return GDMonoMarshal::unbox<R>(p_val);
}

template <class R>
R unbox_if_needed(MonoObject *p_val, const ManagedType &p_type, typename std::enable_if<std::is_pointer<R>::value>::type * = 0) {
	if (mono_class_is_valuetype(p_type.type_class->get_mono_ptr())) {
		return GDMonoMarshal::unbox<R>(p_val);
	} else {
		// If it's not a value type, we assume 'R' is a pointer to 'MonoObject' or a compatible type, like 'MonoException'.
		return (R)p_val;
	}
}

template <unsigned int ThunkParamCount, class R, class P1, class... ParamTypes>
struct VariadicInvokeMonoMethodRImpl {
	static R invoke(GDMonoMethod *p_mono_method, P1 p_arg1, ParamTypes... p_args, MonoException **r_exc) {
		if (p_mono_method->is_static()) {
			void *args[ThunkParamCount] = { p_arg1, p_args... };
			MonoObject *r = p_mono_method->invoke_raw(NULL, args, r_exc);
			return unbox_if_needed<R>(r, p_mono_method->get_return_type());
		} else {
			void *args[ThunkParamCount] = { p_args... };
			MonoObject *r = p_mono_method->invoke_raw((MonoObject *)p_arg1, args, r_exc);
			return unbox_if_needed<R>(r, p_mono_method->get_return_type());
		}
	}
};

template <unsigned int ThunkParamCount, class R, class... ParamTypes>
struct VariadicInvokeMonoMethodR {
	static R invoke(GDMonoMethod *p_mono_method, ParamTypes... p_args, MonoException **r_exc) {
		return VariadicInvokeMonoMethodRImpl<ThunkParamCount, R, ParamTypes...>::invoke(p_mono_method, p_args..., r_exc);
	}
};

template <class R>
struct VariadicInvokeMonoMethodR<0, R> {
	static R invoke(GDMonoMethod *p_mono_method, MonoException **r_exc) {
#ifdef DEBUG_ENABLED
		CRASH_COND(!p_mono_method->is_static());
#endif
		MonoObject *r = p_mono_method->invoke_raw(NULL, NULL, r_exc);
		return unbox_if_needed<R>(r, p_mono_method->get_return_type());
	}
};

template <class R, class P1>
struct VariadicInvokeMonoMethodR<1, R, P1> {
	static R invoke(GDMonoMethod *p_mono_method, P1 p_arg1, MonoException **r_exc) {
		if (p_mono_method->is_static()) {
			void *args[1] = { p_arg1 };
			MonoObject *r = p_mono_method->invoke_raw(NULL, args, r_exc);
			return unbox_if_needed<R>(r, p_mono_method->get_return_type());
		} else {
			MonoObject *r = p_mono_method->invoke_raw((MonoObject *)p_arg1, NULL, r_exc);
			return unbox_if_needed<R>(r, p_mono_method->get_return_type());
		}
	}
};

template <class... ParamTypes>
struct GDMonoMethodThunk {
	GDMonoMethod *mono_method;

public:
	_FORCE_INLINE_ void invoke(ParamTypes... p_args, MonoException **r_exc) {
		VariadicInvokeMonoMethod<sizeof...(ParamTypes), ParamTypes...>::invoke(mono_method, p_args..., r_exc);
	}

	_FORCE_INLINE_ bool is_null() {
		return mono_method == NULL;
	}

	_FORCE_INLINE_ void nullify() {
		mono_method = NULL;
	}

	_FORCE_INLINE_ void set_from_method(GDMonoMethod *p_mono_method) {
#ifdef DEBUG_ENABLED
		CRASH_COND(p_mono_method == NULL);
		CRASH_COND(p_mono_method->get_return_type().type_encoding != MONO_TYPE_VOID);

		if (p_mono_method->is_static()) {
			CRASH_COND(p_mono_method->get_parameters_count() != sizeof...(ParamTypes));
		} else {
			CRASH_COND(p_mono_method->get_parameters_count() != (sizeof...(ParamTypes) - 1));
		}
#endif
		mono_method = p_mono_method;
	}

	GDMonoMethodThunk() :
			mono_method(NULL) {
	}

	explicit GDMonoMethodThunk(GDMonoMethod *p_mono_method) {
		set_from_method(p_mono_method);
	}
};

template <class R, class... ParamTypes>
struct GDMonoMethodThunkR {
	GDMonoMethod *mono_method;

public:
	_FORCE_INLINE_ R invoke(ParamTypes... p_args, MonoException **r_exc) {
		return VariadicInvokeMonoMethodR<sizeof...(ParamTypes), R, ParamTypes...>::invoke(mono_method, p_args..., r_exc);
	}

	_FORCE_INLINE_ bool is_null() {
		return mono_method == NULL;
	}

	_FORCE_INLINE_ void nullify() {
		mono_method = NULL;
	}

	_FORCE_INLINE_ void set_from_method(GDMonoMethod *p_mono_method) {
#ifdef DEBUG_ENABLED
		CRASH_COND(p_mono_method == NULL);
		CRASH_COND(p_mono_method->get_return_type().type_encoding == MONO_TYPE_VOID);

		if (p_mono_method->is_static()) {
			CRASH_COND(p_mono_method->get_parameters_count() != sizeof...(ParamTypes));
		} else {
			CRASH_COND(p_mono_method->get_parameters_count() != (sizeof...(ParamTypes) - 1));
		}
#endif
		mono_method = p_mono_method;
	}

	GDMonoMethodThunkR() :
			mono_method(NULL) {
	}

	explicit GDMonoMethodThunkR(GDMonoMethod *p_mono_method) {
		set_from_method(p_mono_method);
	}
};

#endif

#endif // GD_MONO_METHOD_THUNK_H
