/**************************************************************************/
/*  binder_common.hpp                                                     */
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

#include <gdextension_interface.h>

#include <godot_cpp/core/defs.hpp>
#include <godot_cpp/core/method_ptrcall.hpp>
#include <godot_cpp/core/type_info.hpp>

#include <array>
#include <vector>

namespace godot {

#define VARIANT_ENUM_CAST(m_enum)                                      \
	namespace godot {                                                  \
	MAKE_ENUM_TYPE_INFO(m_enum)                                        \
	template <>                                                        \
	struct VariantCaster<m_enum> {                                     \
		static _FORCE_INLINE_ m_enum cast(const Variant &p_variant) {  \
			return (m_enum)p_variant.operator int64_t();               \
		}                                                              \
	};                                                                 \
	template <>                                                        \
	struct PtrToArg<m_enum> {                                          \
		_FORCE_INLINE_ static m_enum convert(const void *p_ptr) {      \
			return m_enum(*reinterpret_cast<const int64_t *>(p_ptr));  \
		}                                                              \
		typedef int64_t EncodeT;                                       \
		_FORCE_INLINE_ static void encode(m_enum p_val, void *p_ptr) { \
			*reinterpret_cast<int64_t *>(p_ptr) = p_val;               \
		}                                                              \
	};                                                                 \
	}

#define VARIANT_BITFIELD_CAST(m_enum)                                            \
	namespace godot {                                                            \
	MAKE_BITFIELD_TYPE_INFO(m_enum)                                              \
	template <>                                                                  \
	struct VariantCaster<BitField<m_enum>> {                                     \
		static _FORCE_INLINE_ BitField<m_enum> cast(const Variant &p_variant) {  \
			return BitField<m_enum>(p_variant.operator int64_t());               \
		}                                                                        \
	};                                                                           \
	template <>                                                                  \
	struct PtrToArg<BitField<m_enum>> {                                          \
		_FORCE_INLINE_ static BitField<m_enum> convert(const void *p_ptr) {      \
			return BitField<m_enum>(*reinterpret_cast<const int64_t *>(p_ptr));  \
		}                                                                        \
		typedef int64_t EncodeT;                                                 \
		_FORCE_INLINE_ static void encode(BitField<m_enum> p_val, void *p_ptr) { \
			*reinterpret_cast<int64_t *>(p_ptr) = p_val;                         \
		}                                                                        \
	};                                                                           \
	}

template <typename T>
struct VariantCaster {
	static _FORCE_INLINE_ T cast(const Variant &p_variant) {
		using TStripped = std::remove_pointer_t<T>;
		if constexpr (std::is_base_of<Object, TStripped>::value) {
			return Object::cast_to<TStripped>(p_variant);
		} else {
			return p_variant;
		}
	}
};

template <typename T>
struct VariantCaster<T &> {
	static _FORCE_INLINE_ T cast(const Variant &p_variant) {
		using TStripped = std::remove_pointer_t<T>;
		if constexpr (std::is_base_of<Object, TStripped>::value) {
			return Object::cast_to<TStripped>(p_variant);
		} else {
			return p_variant;
		}
	}
};

template <typename T>
struct VariantCaster<const T &> {
	static _FORCE_INLINE_ T cast(const Variant &p_variant) {
		using TStripped = std::remove_pointer_t<T>;
		if constexpr (std::is_base_of<Object, TStripped>::value) {
			return Object::cast_to<TStripped>(p_variant);
		} else {
			return p_variant;
		}
	}
};

template <typename T>
struct VariantObjectClassChecker {
	static _FORCE_INLINE_ bool check(const Variant &p_variant) {
		using TStripped = std::remove_pointer_t<T>;
		if constexpr (std::is_base_of<Object, TStripped>::value) {
			Object *obj = p_variant;
			return Object::cast_to<TStripped>(p_variant) || !obj;
		} else {
			return true;
		}
	}
};

template <typename T>
class Ref;

template <typename T>
struct VariantObjectClassChecker<const Ref<T> &> {
	static _FORCE_INLINE_ bool check(const Variant &p_variant) {
		Object *obj = p_variant;
		const Ref<T> node = p_variant;
		return node.ptr() || !obj;
	}
};

template <typename T>
struct VariantCasterAndValidate {
	static _FORCE_INLINE_ T cast(const Variant **p_args, uint32_t p_arg_idx, GDExtensionCallError &r_error) {
		GDExtensionVariantType argtype = GDExtensionVariantType(GetTypeInfo<T>::VARIANT_TYPE);
		if (!internal::gdextension_interface_variant_can_convert_strict(static_cast<GDExtensionVariantType>(p_args[p_arg_idx]->get_type()), argtype) ||
				!VariantObjectClassChecker<T>::check(p_args[p_arg_idx])) {
			r_error.error = GDEXTENSION_CALL_ERROR_INVALID_ARGUMENT;
			r_error.argument = p_arg_idx;
			r_error.expected = argtype;
		}

		return VariantCaster<T>::cast(*p_args[p_arg_idx]);
	}
};

template <typename T>
struct VariantCasterAndValidate<T &> {
	static _FORCE_INLINE_ T cast(const Variant **p_args, uint32_t p_arg_idx, GDExtensionCallError &r_error) {
		GDExtensionVariantType argtype = GDExtensionVariantType(GetTypeInfo<T>::VARIANT_TYPE);
		if (!internal::gdextension_interface_variant_can_convert_strict(static_cast<GDExtensionVariantType>(p_args[p_arg_idx]->get_type()), argtype) ||
				!VariantObjectClassChecker<T>::check(p_args[p_arg_idx])) {
			r_error.error = GDEXTENSION_CALL_ERROR_INVALID_ARGUMENT;
			r_error.argument = p_arg_idx;
			r_error.expected = argtype;
		}

		return VariantCaster<T>::cast(*p_args[p_arg_idx]);
	}
};

template <typename T>
struct VariantCasterAndValidate<const T &> {
	static _FORCE_INLINE_ T cast(const Variant **p_args, uint32_t p_arg_idx, GDExtensionCallError &r_error) {
		GDExtensionVariantType argtype = GDExtensionVariantType(GetTypeInfo<T>::VARIANT_TYPE);
		if (!internal::gdextension_interface_variant_can_convert_strict(static_cast<GDExtensionVariantType>(p_args[p_arg_idx]->get_type()), argtype) ||
				!VariantObjectClassChecker<T>::check(p_args[p_arg_idx])) {
			r_error.error = GDEXTENSION_CALL_ERROR_INVALID_ARGUMENT;
			r_error.argument = p_arg_idx;
			r_error.expected = argtype;
		}

		return VariantCaster<T>::cast(*p_args[p_arg_idx]);
	}
};

template <typename T, typename... P, size_t... Is>
void call_with_ptr_args_helper(T *p_instance, void (T::*p_method)(P...), const GDExtensionConstTypePtr *p_args, IndexSequence<Is...>) {
	(p_instance->*p_method)(PtrToArg<P>::convert(p_args[Is])...);
}

template <typename T, typename... P, size_t... Is>
void call_with_ptr_argsc_helper(T *p_instance, void (T::*p_method)(P...) const, const GDExtensionConstTypePtr *p_args, IndexSequence<Is...>) {
	(p_instance->*p_method)(PtrToArg<P>::convert(p_args[Is])...);
}

template <typename T, typename R, typename... P, size_t... Is>
void call_with_ptr_args_ret_helper(T *p_instance, R (T::*p_method)(P...), const GDExtensionConstTypePtr *p_args, void *r_ret, IndexSequence<Is...>) {
	PtrToArg<R>::encode((p_instance->*p_method)(PtrToArg<P>::convert(p_args[Is])...), r_ret);
}

template <typename T, typename R, typename... P, size_t... Is>
void call_with_ptr_args_retc_helper(T *p_instance, R (T::*p_method)(P...) const, const GDExtensionConstTypePtr *p_args, void *r_ret, IndexSequence<Is...>) {
	PtrToArg<R>::encode((p_instance->*p_method)(PtrToArg<P>::convert(p_args[Is])...), r_ret);
}

template <typename T, typename... P>
void call_with_ptr_args(T *p_instance, void (T::*p_method)(P...), const GDExtensionConstTypePtr *p_args, void * /*ret*/) {
	call_with_ptr_args_helper<T, P...>(p_instance, p_method, p_args, BuildIndexSequence<sizeof...(P)>{});
}

template <typename T, typename... P>
void call_with_ptr_args(T *p_instance, void (T::*p_method)(P...) const, const GDExtensionConstTypePtr *p_args, void * /*ret*/) {
	call_with_ptr_argsc_helper<T, P...>(p_instance, p_method, p_args, BuildIndexSequence<sizeof...(P)>{});
}

template <typename T, typename R, typename... P>
void call_with_ptr_args(T *p_instance, R (T::*p_method)(P...), const GDExtensionConstTypePtr *p_args, void *r_ret) {
	call_with_ptr_args_ret_helper<T, R, P...>(p_instance, p_method, p_args, r_ret, BuildIndexSequence<sizeof...(P)>{});
}

template <typename T, typename R, typename... P>
void call_with_ptr_args(T *p_instance, R (T::*p_method)(P...) const, const GDExtensionConstTypePtr *p_args, void *r_ret) {
	call_with_ptr_args_retc_helper<T, R, P...>(p_instance, p_method, p_args, r_ret, BuildIndexSequence<sizeof...(P)>{});
}

template <typename T, typename... P, size_t... Is>
void call_with_variant_args_helper(T *p_instance, void (T::*p_method)(P...), const Variant **p_args, GDExtensionCallError &r_error, IndexSequence<Is...>) {
	r_error.error = GDEXTENSION_CALL_OK;

#ifdef DEBUG_METHODS_ENABLED
	(p_instance->*p_method)(VariantCasterAndValidate<P>::cast(p_args, Is, r_error)...);
#else
	(p_instance->*p_method)(VariantCaster<P>::cast(*p_args[Is])...);
#endif
	(void)(p_args); // Avoid warning.
}

template <typename T, typename... P, size_t... Is>
void call_with_variant_argsc_helper(T *p_instance, void (T::*p_method)(P...) const, const Variant **p_args, GDExtensionCallError &r_error, IndexSequence<Is...>) {
	r_error.error = GDEXTENSION_CALL_OK;

#ifdef DEBUG_METHODS_ENABLED
	(p_instance->*p_method)(VariantCasterAndValidate<P>::cast(p_args, Is, r_error)...);
#else
	(p_instance->*p_method)(VariantCaster<P>::cast(*p_args[Is])...);
#endif
	(void)(p_args); // Avoid warning.
}

template <typename T, typename R, typename... P, size_t... Is>
void call_with_variant_args_ret_helper(T *p_instance, R (T::*p_method)(P...), const Variant **p_args, Variant &r_ret, GDExtensionCallError &r_error, IndexSequence<Is...>) {
	r_error.error = GDEXTENSION_CALL_OK;

#ifdef DEBUG_METHODS_ENABLED
	r_ret = (p_instance->*p_method)(VariantCasterAndValidate<P>::cast(p_args, Is, r_error)...);
#else
	r_ret = (p_instance->*p_method)(VariantCaster<P>::cast(*p_args[Is])...);
#endif
}

template <typename T, typename R, typename... P, size_t... Is>
void call_with_variant_args_retc_helper(T *p_instance, R (T::*p_method)(P...) const, const Variant **p_args, Variant &r_ret, GDExtensionCallError &r_error, IndexSequence<Is...>) {
	r_error.error = GDEXTENSION_CALL_OK;

#ifdef DEBUG_METHODS_ENABLED
	r_ret = (p_instance->*p_method)(VariantCasterAndValidate<P>::cast(p_args, Is, r_error)...);
#else
	r_ret = (p_instance->*p_method)(VariantCaster<P>::cast(*p_args[Is])...);
#endif
	(void)p_args;
}

template <typename T, typename... P>
void call_with_variant_args(T *p_instance, void (T::*p_method)(P...), const Variant **p_args, int p_argcount, GDExtensionCallError &r_error) {
#ifdef DEBUG_ENABLED
	if ((size_t)p_argcount > sizeof...(P)) {
		r_error.error = GDEXTENSION_CALL_ERROR_TOO_MANY_ARGUMENTS;
		r_error.expected = (int32_t)sizeof...(P);
		return;
	}

	if ((size_t)p_argcount < sizeof...(P)) {
		r_error.error = GDEXTENSION_CALL_ERROR_TOO_FEW_ARGUMENTS;
		r_error.expected = (int32_t)sizeof...(P);
		return;
	}
#endif
	call_with_variant_args_helper<T, P...>(p_instance, p_method, p_args, r_error, BuildIndexSequence<sizeof...(P)>{});
}

template <typename T, typename R, typename... P>
void call_with_variant_args_ret(T *p_instance, R (T::*p_method)(P...), const Variant **p_args, int p_argcount, Variant &r_ret, GDExtensionCallError &r_error) {
#ifdef DEBUG_ENABLED
	if ((size_t)p_argcount > sizeof...(P)) {
		r_error.error = GDEXTENSION_CALL_ERROR_TOO_MANY_ARGUMENTS;
		r_error.expected = (int32_t)sizeof...(P);
		return;
	}

	if ((size_t)p_argcount < sizeof...(P)) {
		r_error.error = GDEXTENSION_CALL_ERROR_TOO_FEW_ARGUMENTS;
		r_error.expected = (int32_t)sizeof...(P);
		return;
	}
#endif
	call_with_variant_args_ret_helper<T, R, P...>(p_instance, p_method, p_args, r_ret, r_error, BuildIndexSequence<sizeof...(P)>{});
}

template <typename T, typename R, typename... P>
void call_with_variant_args_retc(T *p_instance, R (T::*p_method)(P...) const, const Variant **p_args, int p_argcount, Variant &r_ret, GDExtensionCallError &r_error) {
#ifdef DEBUG_ENABLED
	if ((size_t)p_argcount > sizeof...(P)) {
		r_error.error = GDEXTENSION_CALL_ERROR_TOO_MANY_ARGUMENTS;
		r_error.expected = (int32_t)sizeof...(P);
		return;
	}

	if ((size_t)p_argcount < sizeof...(P)) {
		r_error.error = GDEXTENSION_CALL_ERROR_TOO_FEW_ARGUMENTS;
		r_error.expected = (int32_t)sizeof...(P);
		return;
	}
#endif
	call_with_variant_args_retc_helper<T, R, P...>(p_instance, p_method, p_args, r_ret, r_error, BuildIndexSequence<sizeof...(P)>{});
}

template <typename T, typename... P>
void call_with_variant_args_dv(T *p_instance, void (T::*p_method)(P...), const GDExtensionConstVariantPtr *p_args, int p_argcount, GDExtensionCallError &r_error, const std::vector<Variant> &default_values) {
#ifdef DEBUG_ENABLED
	if ((size_t)p_argcount > sizeof...(P)) {
		r_error.error = GDEXTENSION_CALL_ERROR_TOO_MANY_ARGUMENTS;
		r_error.expected = (int32_t)sizeof...(P);
		return;
	}
#endif

	int32_t missing = (int32_t)sizeof...(P) - (int32_t)p_argcount;

	int32_t dvs = (int32_t)default_values.size();
#ifdef DEBUG_ENABLED
	if (missing > dvs) {
		r_error.error = GDEXTENSION_CALL_ERROR_TOO_FEW_ARGUMENTS;
		r_error.expected = (int32_t)sizeof...(P);
		return;
	}
#endif

	Variant args[sizeof...(P) == 0 ? 1 : sizeof...(P)]; // Avoid zero sized array.
	std::array<const Variant *, sizeof...(P)> argsp;
	for (int32_t i = 0; i < (int32_t)sizeof...(P); i++) {
		if (i < p_argcount) {
			args[i] = Variant(p_args[i]);
		} else {
			args[i] = default_values[i - p_argcount + (dvs - missing)];
		}
		argsp[i] = &args[i];
	}

	call_with_variant_args_helper(p_instance, p_method, argsp.data(), r_error, BuildIndexSequence<sizeof...(P)>{});
}

template <typename T, typename... P>
void call_with_variant_argsc_dv(T *p_instance, void (T::*p_method)(P...) const, const GDExtensionConstVariantPtr *p_args, int p_argcount, GDExtensionCallError &r_error, const std::vector<Variant> &default_values) {
#ifdef DEBUG_ENABLED
	if ((size_t)p_argcount > sizeof...(P)) {
		r_error.error = GDEXTENSION_CALL_ERROR_TOO_MANY_ARGUMENTS;
		r_error.expected = (int32_t)sizeof...(P);
		return;
	}
#endif

	int32_t missing = (int32_t)sizeof...(P) - (int32_t)p_argcount;

	int32_t dvs = (int32_t)default_values.size();
#ifdef DEBUG_ENABLED
	if (missing > dvs) {
		r_error.error = GDEXTENSION_CALL_ERROR_TOO_FEW_ARGUMENTS;
		r_error.expected = (int32_t)sizeof...(P);
		return;
	}
#endif

	Variant args[sizeof...(P) == 0 ? 1 : sizeof...(P)]; // Avoid zero sized array.
	std::array<const Variant *, sizeof...(P)> argsp;
	for (int32_t i = 0; i < (int32_t)sizeof...(P); i++) {
		if (i < p_argcount) {
			args[i] = Variant(p_args[i]);
		} else {
			args[i] = default_values[i - p_argcount + (dvs - missing)];
		}
		argsp[i] = &args[i];
	}

	call_with_variant_argsc_helper(p_instance, p_method, argsp.data(), r_error, BuildIndexSequence<sizeof...(P)>{});
}

template <typename T, typename R, typename... P>
void call_with_variant_args_ret_dv(T *p_instance, R (T::*p_method)(P...), const GDExtensionConstVariantPtr *p_args, int p_argcount, Variant &r_ret, GDExtensionCallError &r_error, const std::vector<Variant> &default_values) {
#ifdef DEBUG_ENABLED
	if ((size_t)p_argcount > sizeof...(P)) {
		r_error.error = GDEXTENSION_CALL_ERROR_TOO_MANY_ARGUMENTS;
		r_error.expected = (int32_t)sizeof...(P);
		return;
	}
#endif

	int32_t missing = (int32_t)sizeof...(P) - (int32_t)p_argcount;

	int32_t dvs = (int32_t)default_values.size();
#ifdef DEBUG_ENABLED
	if (missing > dvs) {
		r_error.error = GDEXTENSION_CALL_ERROR_TOO_FEW_ARGUMENTS;
		r_error.expected = (int32_t)sizeof...(P);
		return;
	}
#endif

	Variant args[sizeof...(P) == 0 ? 1 : sizeof...(P)]; // Avoid zero sized array.
	std::array<const Variant *, sizeof...(P)> argsp;
	for (int32_t i = 0; i < (int32_t)sizeof...(P); i++) {
		if (i < p_argcount) {
			args[i] = Variant(p_args[i]);
		} else {
			args[i] = default_values[i - p_argcount + (dvs - missing)];
		}
		argsp[i] = &args[i];
	}

	call_with_variant_args_ret_helper(p_instance, p_method, argsp.data(), r_ret, r_error, BuildIndexSequence<sizeof...(P)>{});
}

template <typename T, typename R, typename... P>
void call_with_variant_args_retc_dv(T *p_instance, R (T::*p_method)(P...) const, const GDExtensionConstVariantPtr *p_args, int p_argcount, Variant &r_ret, GDExtensionCallError &r_error, const std::vector<Variant> &default_values) {
#ifdef DEBUG_ENABLED
	if ((size_t)p_argcount > sizeof...(P)) {
		r_error.error = GDEXTENSION_CALL_ERROR_TOO_MANY_ARGUMENTS;
		r_error.expected = (int32_t)sizeof...(P);
		return;
	}
#endif

	int32_t missing = (int32_t)sizeof...(P) - (int32_t)p_argcount;

	int32_t dvs = (int32_t)default_values.size();
#ifdef DEBUG_ENABLED
	if (missing > dvs) {
		r_error.error = GDEXTENSION_CALL_ERROR_TOO_FEW_ARGUMENTS;
		r_error.expected = (int32_t)sizeof...(P);
		return;
	}
#endif

	Variant args[sizeof...(P) == 0 ? 1 : sizeof...(P)]; // Avoid zero sized array.
	std::array<const Variant *, sizeof...(P)> argsp;
	for (int32_t i = 0; i < (int32_t)sizeof...(P); i++) {
		if (i < p_argcount) {
			args[i] = Variant(p_args[i]);
		} else {
			args[i] = default_values[i - p_argcount + (dvs - missing)];
		}
		argsp[i] = &args[i];
	}

	call_with_variant_args_retc_helper(p_instance, p_method, argsp.data(), r_ret, r_error, BuildIndexSequence<sizeof...(P)>{});
}

// GCC raises "parameter 'p_args' set but not used" when P = {},
// it's not clever enough to treat other P values as making this branch valid.
#if defined(DEBUG_METHODS_ENABLED) && defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-but-set-parameter"
#endif

template <typename Q>
void call_get_argument_type_helper(int p_arg, int &index, GDExtensionVariantType &type) {
	if (p_arg == index) {
		type = GDExtensionVariantType(GetTypeInfo<Q>::VARIANT_TYPE);
	}
	index++;
}

template <typename... P>
GDExtensionVariantType call_get_argument_type(int p_arg) {
	GDExtensionVariantType type = GDEXTENSION_VARIANT_TYPE_NIL;
	int index = 0;
	// I think rocket science is simpler than modern C++.
	using expand_type = int[];
	expand_type a{ 0, (call_get_argument_type_helper<P>(p_arg, index, type), 0)... };
	(void)a; // Suppress (valid, but unavoidable) -Wunused-variable warning.
	(void)index; // Suppress GCC warning.
	return type;
}

template <typename Q>
void call_get_argument_type_info_helper(int p_arg, int &index, PropertyInfo &info) {
	if (p_arg == index) {
		info = GetTypeInfo<Q>::get_class_info();
	}
	index++;
}

template <typename... P>
void call_get_argument_type_info(int p_arg, PropertyInfo &info) {
	int index = 0;
	// I think rocket science is simpler than modern C++.
	using expand_type = int[];
	expand_type a{ 0, (call_get_argument_type_info_helper<P>(p_arg, index, info), 0)... };
	(void)a; // Suppress (valid, but unavoidable) -Wunused-variable warning.
	(void)index; // Suppress GCC warning.
}

template <typename Q>
void call_get_argument_metadata_helper(int p_arg, int &index, GDExtensionClassMethodArgumentMetadata &md) {
	if (p_arg == index) {
		md = GetTypeInfo<Q>::METADATA;
	}
	index++;
}

template <typename... P>
GDExtensionClassMethodArgumentMetadata call_get_argument_metadata(int p_arg) {
	GDExtensionClassMethodArgumentMetadata md = GDEXTENSION_METHOD_ARGUMENT_METADATA_NONE;

	int index = 0;
	// I think rocket science is simpler than modern C++.
	using expand_type = int[];
	expand_type a{ 0, (call_get_argument_metadata_helper<P>(p_arg, index, md), 0)... };
	(void)a; // Suppress (valid, but unavoidable) -Wunused-variable warning.
	(void)index;
	return md;
}

template <typename... P, size_t... Is>
void call_with_variant_args_static(void (*p_method)(P...), const Variant **p_args, GDExtensionCallError &r_error, IndexSequence<Is...>) {
	r_error.error = GDEXTENSION_CALL_OK;

#ifdef DEBUG_METHODS_ENABLED
	(p_method)(VariantCasterAndValidate<P>::cast(p_args, Is, r_error)...);
#else
	(p_method)(VariantCaster<P>::cast(*p_args[Is])...);
#endif
}

template <typename... P>
void call_with_variant_args_static_dv(void (*p_method)(P...), const GDExtensionConstVariantPtr *p_args, int p_argcount, GDExtensionCallError &r_error, const std::vector<Variant> &default_values) {
#ifdef DEBUG_ENABLED
	if ((size_t)p_argcount > sizeof...(P)) {
		r_error.error = GDEXTENSION_CALL_ERROR_TOO_MANY_ARGUMENTS;
		r_error.expected = sizeof...(P);
		return;
	}
#endif

	int32_t missing = (int32_t)sizeof...(P) - (int32_t)p_argcount;

	int32_t dvs = default_values.size();
#ifdef DEBUG_ENABLED
	if (missing > dvs) {
		r_error.error = GDEXTENSION_CALL_ERROR_TOO_FEW_ARGUMENTS;
		r_error.expected = sizeof...(P);
		return;
	}
#endif

	Variant args[sizeof...(P) == 0 ? 1 : sizeof...(P)]; // Avoid zero sized array.
	std::array<const Variant *, sizeof...(P)> argsp;
	for (int32_t i = 0; i < (int32_t)sizeof...(P); i++) {
		if (i < p_argcount) {
			args[i] = Variant(p_args[i]);
		} else {
			args[i] = default_values[i - p_argcount + (dvs - missing)];
		}
		argsp[i] = &args[i];
	}

	call_with_variant_args_static(p_method, argsp.data(), r_error, BuildIndexSequence<sizeof...(P)>{});
}

template <typename... P, size_t... Is>
void call_with_ptr_args_static_method_helper(void (*p_method)(P...), const GDExtensionConstTypePtr *p_args, IndexSequence<Is...>) {
	p_method(PtrToArg<P>::convert(p_args[Is])...);
}

template <typename... P>
void call_with_ptr_args_static_method(void (*p_method)(P...), const GDExtensionConstTypePtr *p_args) {
	call_with_ptr_args_static_method_helper<P...>(p_method, p_args, BuildIndexSequence<sizeof...(P)>{});
}

template <typename R, typename... P>
void call_with_variant_args_static_ret(R (*p_method)(P...), const Variant **p_args, int p_argcount, Variant &r_ret, GDExtensionCallError &r_error) {
#ifdef DEBUG_ENABLED
	if ((size_t)p_argcount > sizeof...(P)) {
		r_error.error = GDEXTENSION_CALL_ERROR_TOO_MANY_ARGUMENTS;
		r_error.expected = (int32_t)sizeof...(P);
		return;
	}

	if ((size_t)p_argcount < sizeof...(P)) {
		r_error.error = GDEXTENSION_CALL_ERROR_TOO_FEW_ARGUMENTS;
		r_error.expected = (int32_t)sizeof...(P);
		return;
	}
#endif
	call_with_variant_args_static_ret<R, P...>(p_method, p_args, r_ret, r_error, BuildIndexSequence<sizeof...(P)>{});
}

template <typename... P>
void call_with_variant_args_static_ret(void (*p_method)(P...), const Variant **p_args, int p_argcount, Variant &r_ret, GDExtensionCallError &r_error) {
#ifdef DEBUG_ENABLED
	if ((size_t)p_argcount > sizeof...(P)) {
		r_error.error = GDEXTENSION_CALL_ERROR_TOO_MANY_ARGUMENTS;
		r_error.expected = (int32_t)sizeof...(P);
		return;
	}

	if ((size_t)p_argcount < sizeof...(P)) {
		r_error.error = GDEXTENSION_CALL_ERROR_TOO_FEW_ARGUMENTS;
		r_error.expected = (int32_t)sizeof...(P);
		return;
	}
#endif
	call_with_variant_args_static<P...>(p_method, p_args, r_error, BuildIndexSequence<sizeof...(P)>{});
}

template <typename R, typename... P, size_t... Is>
void call_with_variant_args_static_ret(R (*p_method)(P...), const Variant **p_args, Variant &r_ret, GDExtensionCallError &r_error, IndexSequence<Is...>) {
	r_error.error = GDEXTENSION_CALL_OK;

#ifdef DEBUG_METHODS_ENABLED
	r_ret = (p_method)(VariantCasterAndValidate<P>::cast(p_args, Is, r_error)...);
#else
	r_ret = (p_method)(VariantCaster<P>::cast(*p_args[Is])...);
#endif
}

template <typename R, typename... P>
void call_with_variant_args_static_ret_dv(R (*p_method)(P...), const GDExtensionConstVariantPtr *p_args, int p_argcount, Variant &r_ret, GDExtensionCallError &r_error, const std::vector<Variant> &default_values) {
#ifdef DEBUG_ENABLED
	if ((size_t)p_argcount > sizeof...(P)) {
		r_error.error = GDEXTENSION_CALL_ERROR_TOO_MANY_ARGUMENTS;
		r_error.expected = sizeof...(P);
		return;
	}
#endif

	int32_t missing = (int32_t)sizeof...(P) - (int32_t)p_argcount;

	int32_t dvs = default_values.size();
#ifdef DEBUG_ENABLED
	if (missing > dvs) {
		r_error.error = GDEXTENSION_CALL_ERROR_TOO_FEW_ARGUMENTS;
		r_error.expected = sizeof...(P);
		return;
	}
#endif

	Variant args[sizeof...(P) == 0 ? 1 : sizeof...(P)]; // Avoid zero sized array.
	std::array<const Variant *, sizeof...(P)> argsp;
	for (int32_t i = 0; i < (int32_t)sizeof...(P); i++) {
		if (i < p_argcount) {
			args[i] = Variant(p_args[i]);
		} else {
			args[i] = default_values[i - p_argcount + (dvs - missing)];
		}
		argsp[i] = &args[i];
	}

	call_with_variant_args_static_ret(p_method, argsp.data(), r_ret, r_error, BuildIndexSequence<sizeof...(P)>{});
}

template <typename R, typename... P, size_t... Is>
void call_with_ptr_args_static_method_ret_helper(R (*p_method)(P...), const GDExtensionConstTypePtr *p_args, void *r_ret, IndexSequence<Is...>) {
	PtrToArg<R>::encode(p_method(PtrToArg<P>::convert(p_args[Is])...), r_ret);
}

template <typename R, typename... P>
void call_with_ptr_args_static_method_ret(R (*p_method)(P...), const GDExtensionConstTypePtr *p_args, void *r_ret) {
	call_with_ptr_args_static_method_ret_helper<R, P...>(p_method, p_args, r_ret, BuildIndexSequence<sizeof...(P)>{});
}

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

} // namespace godot

#include <godot_cpp/classes/global_constants_binds.hpp>
#include <godot_cpp/variant/builtin_binds.hpp>
