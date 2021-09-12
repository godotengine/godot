/*************************************************************************/
/*  gd_mono_method_thunk.h                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef GD_MONO_METHOD_THUNK_H
#define GD_MONO_METHOD_THUNK_H

#include <mono/jit/jit.h>
#include <mono/metadata/attrdefs.h>
#include <type_traits>

#include "core/error/error_macros.h"
#include "gd_mono_utils.h"

#ifdef WIN32
#define GD_MONO_STDCALL __stdcall
#else
#define GD_MONO_STDCALL
#endif

template <class... ParamTypes>
struct GDMonoMethodThunk {
	typedef void(GD_MONO_STDCALL *M)(ParamTypes... p_args, MonoException **);

	M mono_method_thunk = nullptr;

public:
	_FORCE_INLINE_ void invoke(ParamTypes... p_args, MonoException **r_exc) {
		GD_MONO_BEGIN_RUNTIME_INVOKE;
		mono_method_thunk(p_args..., r_exc);
		GD_MONO_END_RUNTIME_INVOKE;
	}

	bool is_null() {
		return mono_method_thunk == nullptr;
	}

	void set_from_method(MonoMethod *p_mono_method) {
#ifdef DEBUG_ENABLED
		CRASH_COND(p_mono_method == nullptr);

		MonoMethodSignature *method_sig = mono_method_signature(p_mono_method);
		MonoType *ret_type = mono_signature_get_return_type(method_sig);
		int ret_type_encoding = ret_type ? mono_type_get_type(ret_type) : MONO_TYPE_VOID;

		CRASH_COND(ret_type_encoding != MONO_TYPE_VOID);

		bool is_static = mono_method_get_flags(p_mono_method, nullptr) & MONO_METHOD_ATTR_STATIC;
		CRASH_COND(!is_static);

		uint32_t parameters_count = mono_signature_get_param_count(method_sig);
		CRASH_COND(parameters_count != sizeof...(ParamTypes));
#endif
		mono_method_thunk = (M)mono_method_get_unmanaged_thunk(p_mono_method);
	}

	GDMonoMethodThunk() {}
};

template <class R, class... ParamTypes>
struct GDMonoMethodThunkR {
	typedef R(GD_MONO_STDCALL *M)(ParamTypes... p_args, MonoException **);

	M mono_method_thunk = nullptr;

public:
	_FORCE_INLINE_ R invoke(ParamTypes... p_args, MonoException **r_exc) {
		GD_MONO_BEGIN_RUNTIME_INVOKE;
		R r = mono_method_thunk(p_args..., r_exc);
		GD_MONO_END_RUNTIME_INVOKE;
		return r;
	}

	bool is_null() {
		return mono_method_thunk == nullptr;
	}

	void set_from_method(MonoMethod *p_mono_method) {
#ifdef DEBUG_ENABLED
		CRASH_COND(p_mono_method == nullptr);

		MonoMethodSignature *method_sig = mono_method_signature(p_mono_method);
		MonoType *ret_type = mono_signature_get_return_type(method_sig);
		int ret_type_encoding = ret_type ? mono_type_get_type(ret_type) : MONO_TYPE_VOID;

		CRASH_COND(ret_type_encoding == MONO_TYPE_VOID);

		bool is_static = mono_method_get_flags(p_mono_method, nullptr) & MONO_METHOD_ATTR_STATIC;
		CRASH_COND(!is_static);

		uint32_t parameters_count = mono_signature_get_param_count(method_sig);
		CRASH_COND(parameters_count != sizeof...(ParamTypes));
#endif
		mono_method_thunk = (M)mono_method_get_unmanaged_thunk(p_mono_method);
	}

	GDMonoMethodThunkR() {}
};

#endif // GD_MONO_METHOD_THUNK_H
