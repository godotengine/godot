/**************************************************************************/
/*  binder_common.h                                                       */
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

#include "core/typedefs.h"
#include "core/variant/callable.h"
#include "core/variant/method_ptrcall.h"
#include "core/variant/type_info.h"
#include "core/variant/variant.h"
#include "core/variant/variant_caster.h"
#include "core/variant/variant_internal.h"

#include <cstdio>
#include <type_traits>
#include <utility>

template <>
struct PtrToArg<char32_t> {
	_FORCE_INLINE_ static char32_t convert(const void *p_ptr) {
		return char32_t(*reinterpret_cast<const int64_t *>(p_ptr));
	}
	typedef int64_t EncodeT;
	_FORCE_INLINE_ static void encode(char32_t p_val, const void *p_ptr) {
		*(int64_t *)p_ptr = p_val;
	}
};

/**** Static Binder Instance ****/

struct BinderNoInstance {};
inline constexpr BinderNoInstance *BINDER_NO_INSTANCE = nullptr;

/**** Method Information ****/

template <typename R, typename... P>
struct BinderSignature {
	using Ret = R;
	static constexpr size_t arg_count = sizeof...(P);
};

template <typename M, bool instance_is_first_arg = false>
struct BinderTraits;

// Represents a non-const class method called on an instance.
template <typename R, typename T, typename... P>
struct BinderTraits<R (T::*)(P...), false> {
	using Class = T;
	using Signature = BinderSignature<R, P...>;
	static constexpr bool is_const = false;
};

// Represents a const class method called on an instance.
template <typename R, typename T, typename... P>
struct BinderTraits<R (T::*)(P...) const, false> {
	using Class = T;
	using Signature = BinderSignature<R, P...>;
	static constexpr bool is_const = true;
};

// Represents a class method called on an instance, where the instance is passed as the first argument.
template <typename R, typename T, typename... P>
struct BinderTraits<R (*)(T *, P...), true> {
	using Class = T;
	using Signature = BinderSignature<R, P...>;
	static constexpr bool is_const = false;
};

// Represents a static method.
template <typename R, typename... P>
struct BinderTraits<R (*)(P...), false> {
	using Signature = BinderSignature<R, P...>;
	static constexpr bool is_const = false;
};

// Convenience template for method signature extraction

template <typename T, typename M>
using MethodSignatureOf = typename BinderTraits<M, (!std::is_member_function_pointer_v<M> && !std::is_same_v<T, BinderNoInstance>)>::Signature;

/**** Argument utilities ****/

inline bool check_method_arg_count(int p_argcount, int p_expected, Callable::CallError &r_error) {
#ifdef DEBUG_ENABLED
	if (p_argcount > p_expected) {
		r_error.error = Callable::CallError::CALL_ERROR_TOO_MANY_ARGUMENTS;
		r_error.expected = p_expected;
		return false;
	}

	if (p_argcount < p_expected) {
		r_error.error = Callable::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;
		r_error.expected = p_expected;
		return false;
	}
#endif // DEBUG_ENABLED
	return true;
}

inline bool fill_default_args(const Variant **p_args, int p_argcount, int p_expected, const Vector<Variant> &p_default_values, const Variant **r_args, Callable::CallError &r_error) {
#ifdef DEBUG_ENABLED
	if (p_argcount > p_expected) {
		r_error.error = Callable::CallError::CALL_ERROR_TOO_MANY_ARGUMENTS;
		r_error.expected = p_expected;
		return false;
	}
#endif // DEBUG_ENABLED

	int32_t missing = p_expected - p_argcount;
	int32_t dvs = p_default_values.size();
#ifdef DEBUG_ENABLED
	if (missing > dvs) {
		r_error.error = Callable::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;
		r_error.expected = p_expected;
		return false;
	}
#endif // DEBUG_ENABLED

	for (int32_t i = 0; i < p_expected; i++) {
		if (i < p_argcount) {
			r_args[i] = p_args[i];
		} else {
			r_args[i] = &p_default_values[i - p_argcount + (dvs - missing)];
		}
	}
	return true;
}

/**** Implementation ****/

// Central method

template <typename R, typename T, typename M, typename... A>
_FORCE_INLINE_ R invoke_method(T *p_instance, M p_method, A &&...p_args) {
	if constexpr (std::is_member_function_pointer_v<M>) {
		return (p_instance->*p_method)(std::forward<A>(p_args)...);
	} else if constexpr (std::is_same_v<T, BinderNoInstance>) {
		return p_method(std::forward<A>(p_args)...);
	} else {
		return p_method(p_instance, std::forward<A>(p_args)...);
	}
}

// Invokers

template <typename P>
using CastVariantArg = decltype(VariantCaster<P>::cast(std::declval<const Variant &>()));

template <typename P>
_FORCE_INLINE_ CastVariantArg<P> cast_variant_args(const Variant **p_args, uint32_t p_index, Callable::CallError &r_error) {
#ifdef DEBUG_ENABLED
	return VariantCasterAndValidate<P>::cast(p_args, p_index, r_error);
#else
	return VariantCaster<P>::cast(*p_args[p_index]);
#endif // DEBUG_ENABLED
}

template <typename T, typename M, typename R, typename... P, size_t... Is>
void invoke_variant_call(T *p_instance, M p_method, BinderSignature<R, P...>, const Variant **p_args, Variant &r_ret, Callable::CallError &r_error, IndexSequence<Is...>) {
	r_error.error = Callable::CallError::CALL_OK;
	if constexpr (std::is_void_v<R>) {
		invoke_method<R>(p_instance, p_method, cast_variant_args<P>(p_args, Is, r_error)...);
	} else {
		r_ret = VariantInternal::make(invoke_method<R>(p_instance, p_method, cast_variant_args<P>(p_args, Is, r_error)...));
	}
}

template <typename T, typename M, typename R, typename... P, size_t... Is>
void invoke_ptr_call(T *p_instance, M p_method, BinderSignature<R, P...>, const void **p_args, void *r_ret, IndexSequence<Is...>) {
	if constexpr (std::is_void_v<R>) {
		invoke_method<R>(p_instance, p_method, PtrToArg<P>::convert(p_args[Is])...);
	} else {
		PtrToArg<R>::encode(invoke_method<R>(p_instance, p_method, PtrToArg<P>::convert(p_args[Is])...), r_ret);
	}
}

template <typename T, typename M, typename R, typename... P, size_t... Is>
void invoke_validated_call(T *p_instance, M p_method, BinderSignature<R, P...>, const Variant **p_args, Variant *r_ret, IndexSequence<Is...>) {
	if constexpr (std::is_void_v<R>) {
		invoke_method<R>(p_instance, p_method, (VariantInternalAccessor<std::decay_t<P>>::get(p_args[Is]))...);
	} else {
		VariantInternalAccessor<std::decay_t<R>>::set(r_ret, invoke_method<R>(p_instance, p_method, (VariantInternalAccessor<std::decay_t<P>>::get(p_args[Is]))...));
	}
}

/**** API Methods ****/

template <typename T, typename M>
void call_with_variant_args(T *p_instance, M p_method, const Variant **p_args, int p_argcount, Variant &r_ret, Callable::CallError &r_error) {
	using Signature = MethodSignatureOf<T, M>;
	if (!check_method_arg_count(p_argcount, (int)Signature::arg_count, r_error)) {
		return;
	}
	invoke_variant_call(p_instance, p_method, Signature{}, p_args, r_ret, r_error, BuildIndexSequence<Signature::arg_count>{});
}

template <typename T, typename M>
void call_with_variant_args_dv(T *p_instance, M p_method, const Variant **p_args, int p_argcount, Variant &r_ret, Callable::CallError &r_error, const Vector<Variant> &p_default_values) {
	using Signature = MethodSignatureOf<T, M>;
	const Variant *args[Signature::arg_count == 0 ? 1 : Signature::arg_count]; //avoid zero sized array
	if (!fill_default_args(p_args, p_argcount, (int)Signature::arg_count, p_default_values, args, r_error)) {
		return;
	}
	invoke_variant_call(p_instance, p_method, Signature{}, args, r_ret, r_error, BuildIndexSequence<Signature::arg_count>{});
}

template <typename T, typename M>
void call_with_ptr_args(T *p_instance, M p_method, const void **p_args, void *r_ret) {
	using Signature = MethodSignatureOf<T, M>;
	invoke_ptr_call(p_instance, p_method, Signature{}, p_args, r_ret, BuildIndexSequence<Signature::arg_count>{});
}

template <typename T, typename M>
void call_with_validated_args(T *p_instance, M p_method, const Variant **p_args, Variant *r_ret) {
	using Signature = MethodSignatureOf<T, M>;
	invoke_validated_call(p_instance, p_method, Signature{}, p_args, r_ret, BuildIndexSequence<Signature::arg_count>{});
}

/**** Argument Info ****/

// GCC raises "parameter 'p_args' set but not used" when P = {},
// it's not clever enough to treat other P values as making this branch valid.
GODOT_GCC_WARNING_PUSH_AND_IGNORE("-Wunused-but-set-parameter")

template <typename Q>
void call_get_argument_type_helper(int p_arg, int &r_index, Variant::Type &r_type) {
	if (p_arg == r_index) {
		r_type = GetTypeInfo<Q>::VARIANT_TYPE;
	}
	r_index++;
}

template <typename... P>
Variant::Type call_get_argument_type(int p_arg) {
	Variant::Type type = Variant::NIL;
	int index = 0;
	// I think rocket science is simpler than modern C++.
	using expand_type = int[];
	expand_type a{ 0, (call_get_argument_type_helper<P>(p_arg, index, type), 0)... };
	(void)a; // Suppress (valid, but unavoidable) -Wunused-variable warning.
	(void)index; // Suppress GCC warning.
	return type;
}

template <typename Q>
void call_get_argument_type_info_helper(int p_arg, int &r_index, PropertyInfo &r_info) {
	if (p_arg == r_index) {
		r_info = GetTypeInfo<Q>::get_class_info();
	}
	r_index++;
}

template <typename... P>
void call_get_argument_type_info(int p_arg, PropertyInfo &r_info) {
	int index = 0;
	// I think rocket science is simpler than modern C++.
	using expand_type = int[];
	expand_type a{ 0, (call_get_argument_type_info_helper<P>(p_arg, index, r_info), 0)... };
	(void)a; // Suppress (valid, but unavoidable) -Wunused-variable warning.
	(void)index; // Suppress GCC warning.
}

#ifdef DEBUG_ENABLED
template <typename Q>
void call_get_argument_metadata_helper(int p_arg, int &r_index, GodotTypeInfo::Metadata &r_metadata) {
	if (p_arg == r_index) {
		r_metadata = GetTypeInfo<Q>::METADATA;
	}
	r_index++;
}

template <typename... P>
GodotTypeInfo::Metadata call_get_argument_metadata(int p_arg) {
	GodotTypeInfo::Metadata md = GodotTypeInfo::METADATA_NONE;

	int index = 0;
	// I think rocket science is simpler than modern C++.
	using expand_type = int[];
	expand_type a{ 0, (call_get_argument_metadata_helper<P>(p_arg, index, md), 0)... };
	(void)a; // Suppress (valid, but unavoidable) -Wunused-variable warning.
	(void)index;
	return md;
}

#endif // DEBUG_ENABLED

GODOT_GCC_WARNING_POP
