/*************************************************************************/
/*  gd_mono_wasm_m2n.h                                                   */
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

#ifndef GD_MONO_WASM_M2N_H
#define GD_MONO_WASM_M2N_H

#ifdef JAVASCRIPT_ENABLED

#include "core/typedefs.h"
#include "core/ustring.h"

#include <mono/metadata/loader.h>
#include <mono/utils/mono-publib.h>
#include <stdexcept>
#include <type_traits>

extern "C" {

struct Mono_InterpMethodArguments {
	size_t ilen;
	void **iargs;
	size_t flen;
	double *fargs;
	void **retval;
	size_t is_float_ret;
	//#ifdef TARGET_WASM
	void *sig;
	//#endif
};
} // extern "C"

namespace GDMonoWasmM2n {

template <size_t... Is>
struct IndexSequence {};

template <size_t N, size_t... Is>
struct BuildIndexSequence : BuildIndexSequence<N - 1, N - 1, Is...> {};

template <size_t... Is>
struct BuildIndexSequence<0, Is...> : IndexSequence<Is...> {};

template <typename T, size_t Size>
struct array {
	T elems[Size];
};

template <typename T>
constexpr char get_m2n_cookie_impl() {
#define M2N_REG_COOKIE(m_type, m_cookie)  \
	if (std::is_same<m_type, T>::value) { \
		return m_cookie;                  \
	}

	M2N_REG_COOKIE(MonoBoolean, 'I');
	M2N_REG_COOKIE(int8_t, 'I');
	M2N_REG_COOKIE(uint8_t, 'I');
	M2N_REG_COOKIE(int16_t, 'I');
	M2N_REG_COOKIE(uint16_t, 'I');
	M2N_REG_COOKIE(int32_t, 'I');
	M2N_REG_COOKIE(uint32_t, 'I');
	M2N_REG_COOKIE(int64_t, 'L');
	M2N_REG_COOKIE(uint64_t, 'L');
	M2N_REG_COOKIE(float, 'F');
	M2N_REG_COOKIE(double, 'D');

	if (std::is_pointer<T>::value) {
		if (sizeof(void *) == 4) {
			return 'I';
		} else {
			return 'L';
		}
	}

	if (std::is_void<T>::value) {
		return 'V';
	}

	return 'X';

#undef M2N_REG_COOKIE
}

template <typename T>
constexpr char get_m2n_cookie() {
	constexpr char cookie = get_m2n_cookie_impl<T>();
	static_assert(cookie != 'X', "Type not supported in internal call signature.");
	return cookie;
}

template <typename... T>
constexpr array<const char, sizeof...(T) + 2> get_m2n_cookies() {
	return array<const char, sizeof...(T) + 2>{ 'V', get_m2n_cookie<T>()..., '\0' };
}

template <typename R, typename... T>
constexpr array<const char, sizeof...(T) + 2> get_m2n_cookies_r() {
	return array<const char, sizeof...(T) + 2>{ get_m2n_cookie<R>(), get_m2n_cookie<T>()..., '\0' };
}

template <typename T>
constexpr size_t calc_m2n_index(size_t &r_int_idx, size_t &r_float_idx) {
	constexpr char cookie = get_m2n_cookie<T>();

	static_assert(cookie == 'I' || cookie == 'L' || cookie == 'F' || cookie == 'D', "Cookie should be I, L, F or D.");

	if (cookie == 'I' || cookie == 'L') {
		size_t ret = r_int_idx;
		r_int_idx += cookie == 'I' ? 1 : 2;
		return ret;
	} else {
		size_t ret = r_float_idx;
		r_float_idx += cookie == 'F' ? 1 : 2;
		return ret;
	}
}

template <typename... P>
constexpr array<size_t, sizeof...(P)> get_indices_for_type() {
	size_t int_idx = 0;
	size_t float_idx = 0;
	(void)int_idx; // Suppress 'unused' warning when parameter count is 0
	(void)float_idx; // Suppress 'unused' warning when parameter count is 0
	return array<size_t, sizeof...(P)>{ calc_m2n_index<P>(int_idx, float_idx)... };
}

constexpr size_t fidx(size_t p_x) {
	if (sizeof(void *) == 4) {
		return p_x * 2;
	} else {
		return p_x;
	}
}

template <typename T, char cookie>
struct m2n_arg_cast_helper;

template <typename T>
struct m2n_arg_cast_helper<T, 'I'> {
	static T cast(Mono_InterpMethodArguments *p_margs, size_t p_idx) {
		return (T)(size_t)p_margs->iargs[p_idx];
	}
};

template <typename T>
struct m2n_arg_cast_helper<T, 'L'> {
	static T cast(Mono_InterpMethodArguments *p_margs, size_t p_idx) {
		static_assert(std::is_same<T, int64_t>::value || std::is_same<T, uint64_t>::value ||
						(sizeof(void *) == 8 && std::is_pointer<T>::value),
				"Invalid type for cookie 'L'.");

		union {
			T l;
			struct {
				int32_t lo;
				int32_t hi;
			} pair;
		} p;

		p.pair.lo = (int32_t)(size_t)p_margs->iargs[p_idx];
		p.pair.hi = (int32_t)(size_t)p_margs->iargs[p_idx + 1];

		return p.l;
	}
};

template <typename T>
struct m2n_arg_cast_helper<T, 'F'> {
	static T cast(Mono_InterpMethodArguments *p_margs, size_t p_idx) {
		return *reinterpret_cast<float *>(&p_margs->fargs[fidx(p_idx)]);
	}
};

template <typename T>
struct m2n_arg_cast_helper<T, 'D'> {
	static T cast(Mono_InterpMethodArguments *p_margs, size_t p_idx) {
		return (T)p_margs->fargs[p_idx];
	}
};

template <typename T>
T m2n_arg_cast(Mono_InterpMethodArguments *p_margs, size_t p_idx) {
	constexpr char cookie = get_m2n_cookie<T>();

	static_assert(cookie == 'I' || cookie == 'L' || cookie == 'F' || cookie == 'D', "Cookie should be I, L, F or D.");

	return m2n_arg_cast_helper<T, cookie>::cast(p_margs, p_idx);
}

template <typename... P, size_t... Is>
void m2n_trampoline_with_idx_seq(void *p_target_func, Mono_InterpMethodArguments *p_margs, IndexSequence<Is...>) {
	constexpr array<size_t, sizeof...(P)> indices = get_indices_for_type<P...>();
	(void)indices; // Suppress 'unused' warning when parameter count is 0
	typedef void (*Func)(P...);
	Func func = (Func)p_target_func;
	func(m2n_arg_cast<P>(p_margs, indices.elems[Is])...);
}

template <typename R, typename... P, size_t... Is>
void m2n_trampoline_with_idx_seq_r(void *p_target_func, Mono_InterpMethodArguments *p_margs, IndexSequence<Is...>) {
	constexpr array<size_t, sizeof...(P)> indices = get_indices_for_type<P...>();
	(void)indices; // Suppress 'unused' warning when parameter count is 0
	typedef R (*Func)(P...);
	Func func = (Func)p_target_func;
	R res = func(m2n_arg_cast<P>(p_margs, indices.elems[Is])...);
	*reinterpret_cast<R *>(p_margs->retval) = res;
}

template <typename... P>
void m2n_trampoline(void *p_target_func, Mono_InterpMethodArguments *p_margs) {
	m2n_trampoline_with_idx_seq<P...>(p_target_func, p_margs, BuildIndexSequence<sizeof...(P)>{});
}

template <typename R, typename... P>
void m2n_trampoline_r(void *p_target_func, Mono_InterpMethodArguments *p_margs) {
	m2n_trampoline_with_idx_seq_r<R, P...>(p_target_func, p_margs, BuildIndexSequence<sizeof...(P)>{});
}

typedef void (*TrampolineFunc)(void *p_target_func, Mono_InterpMethodArguments *p_margs);

void set_trampoline(const char *cookies, TrampolineFunc trampoline_func);

void lazy_initialize();

template <typename... P>
struct ICallTrampolines {
	static constexpr array<const char, sizeof...(P) + 2> cookies = get_m2n_cookies<P...>();

	static void add() {
		lazy_initialize();
		set_trampoline(cookies.elems, &m2n_trampoline<P...>);
	}
};

template <typename... P>
constexpr array<const char, sizeof...(P) + 2> ICallTrampolines<P...>::cookies;

template <typename R, typename... P>
struct ICallTrampolinesR {
	static constexpr array<const char, sizeof...(P) + 2> cookies = get_m2n_cookies_r<R, P...>();

	static void add() {
		lazy_initialize();
		set_trampoline(cookies.elems, &m2n_trampoline_r<R, P...>);
	}
};

template <typename R, typename... P>
constexpr array<const char, sizeof...(P) + 2> ICallTrampolinesR<R, P...>::cookies;

void initialize();
} // namespace GDMonoWasmM2n

#endif

#endif // GD_MONO_WASM_M2N_H
