/**************************************************************************/
/*  thread_local.h                                                        */
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

#ifndef MONO_THREAD_LOCAL_H
#define MONO_THREAD_LOCAL_H

#ifdef HAVE_CXX11_THREAD_LOCAL
#define _THREAD_LOCAL_(m_t) thread_local m_t
#else

#if !defined(__GNUC__) && !defined(_MSC_VER)
#error Platform or compiler not supported
#endif

#if defined(__GNUC__)

#ifdef HAVE_GCC___THREAD
#define _THREAD_LOCAL_(m_t) __thread m_t
#else
#define USE_CUSTOM_THREAD_LOCAL
#endif

#elif defined(_MSC_VER)

#ifdef HAVE_DECLSPEC_THREAD
#define _THREAD_LOCAL_(m_t) __declspec(thread) m_t
#else
#define USE_CUSTOM_THREAD_LOCAL
#endif

#endif // __GNUC__ _MSC_VER

#endif // HAVE_CXX11_THREAD_LOCAL

#ifdef USE_CUSTOM_THREAD_LOCAL
#define _THREAD_LOCAL_(m_t) ThreadLocal<m_t>
#endif

#include "core/typedefs.h"

#ifdef WINDOWS_ENABLED
#define _CALLBACK_FUNC_ __stdcall
#else
#define _CALLBACK_FUNC_
#endif

struct ThreadLocalStorage {
	void *get_value() const;
	void set_value(void *p_value) const;

	void alloc(void(_CALLBACK_FUNC_ *p_destr_callback)(void *));
	void free();

private:
	struct Impl;
	Impl *pimpl;
};

template <typename T>
class ThreadLocal {
	ThreadLocalStorage storage;

	T init_val;

	static void _CALLBACK_FUNC_ destr_callback(void *tls_data) {
		memdelete(static_cast<T *>(tls_data));
	}

	T *_tls_get_value() const {
		void *tls_data = storage.get_value();

		if (tls_data)
			return static_cast<T *>(tls_data);

		T *data = memnew(T(init_val));

		storage.set_value(data);

		return data;
	}

	void _initialize(const T &p_init_val) {
		init_val = p_init_val;
		storage.alloc(&destr_callback);
	}

public:
	ThreadLocal() {
		_initialize(T());
	}

	ThreadLocal(const T &p_init_val) {
		_initialize(p_init_val);
	}

	ThreadLocal(const ThreadLocal &other) {
		_initialize(*other._tls_get_value());
	}

	~ThreadLocal() {
		storage.free();
	}

	_FORCE_INLINE_ T *operator&() const {
		return _tls_get_value();
	}

	_FORCE_INLINE_ operator T &() const {
		return *_tls_get_value();
	}

	_FORCE_INLINE_ ThreadLocal &operator=(const T &val) {
		T *ptr = _tls_get_value();
		*ptr = val;
		return *this;
	}
};

struct FlagScopeGuard {
	FlagScopeGuard(bool &p_flag) :
			flag(p_flag) {
		flag = !flag;
	}

	~FlagScopeGuard() {
		flag = !flag;
	}

private:
	bool &flag;
};

#undef _CALLBACK_FUNC_

#define _TLS_RECURSION_GUARD_V_(m_ret)                    \
	static _THREAD_LOCAL_(bool) _recursion_flag_ = false; \
	if (_recursion_flag_)                                 \
		return m_ret;                                     \
	FlagScopeGuard _recursion_guard_(_recursion_flag_);

#define _TLS_RECURSION_GUARD_                             \
	static _THREAD_LOCAL_(bool) _recursion_flag_ = false; \
	if (_recursion_flag_)                                 \
		return;                                           \
	FlagScopeGuard _recursion_guard_(_recursion_flag_);

#endif // MONO_THREAD_LOCAL_H
