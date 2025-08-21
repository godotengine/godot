/**************************************************************************/
/*  callable.hpp                                                          */
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

#include "variant.hpp"

struct Callable {
	constexpr Callable() {}
	template <typename F>
	Callable(F *f, const Variant &args = Nil);

	/// @brief Create a callable from a function pointer, which always returns a Variant.
	/// @tparam F The function type.
	/// @param f The function pointer.
	/// @param args The arguments to pass to the function.
	template <typename F>
	static Callable Create(F *f, const Variant &args = Nil);

	/// @brief Call the function with the given arguments.
	/// @tparam Args The argument types.
	/// @param args The arguments.
	template <typename... Args>
	Variant operator()(Args &&...args);

	/// @brief Call the function with the given arguments.
	/// @tparam Args The argument types.
	/// @param args The arguments.
	template <typename... Args>
	Variant call(Args &&...args);

	static Callable from_variant_index(unsigned idx) {
		Callable a;
		a.m_idx = idx;
		return a;
	}
	unsigned get_variant_index() const noexcept { return m_idx; }

private:
	unsigned m_idx = INT32_MIN;
};

inline Variant::Variant(const Callable &callable) {
	m_type = CALLABLE;
	v.i = callable.get_variant_index();
}

inline Callable Variant::as_callable() const {
	if (m_type != CALLABLE) {
		api_throw("std::bad_cast", "Failed to cast Variant to Callable", this);
	}
	return Callable::from_variant_index(v.i);
}

inline Variant::operator Callable() const {
	return as_callable();
}

template <typename... Args>
inline Variant Callable::operator()(Args &&...args) {
	return Variant(*this)(std::forward<Args>(args)...);
}

template <typename... Args>
inline Variant Callable::call(Args &&...args) {
	return Variant(*this).call(std::forward<Args>(args)...);
}

template <typename F>
inline Callable Callable::Create(F *f, const Variant &args) {
	unsigned idx = sys_callable_create((void (*)())f, &args, nullptr, 0);

	return Callable::from_variant_index(idx);
}

template <typename F>
inline Callable::Callable(F *f, const Variant &args) :
		m_idx(sys_callable_create((void (*)())f, &args, nullptr, 0)) {
}
