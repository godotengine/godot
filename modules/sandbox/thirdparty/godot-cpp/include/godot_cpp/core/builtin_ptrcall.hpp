/**************************************************************************/
/*  builtin_ptrcall.hpp                                                   */
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
#include <godot_cpp/core/object.hpp>

#include <array>

namespace godot {

namespace internal {

template <typename O, typename... Args>
O *_call_builtin_method_ptr_ret_obj(const GDExtensionPtrBuiltInMethod method, GDExtensionTypePtr base, const Args &...args) {
	GodotObject *ret = nullptr;
	std::array<GDExtensionConstTypePtr, sizeof...(Args)> call_args = { { (GDExtensionConstTypePtr)args... } };
	method(base, call_args.data(), &ret, sizeof...(Args));
	if (ret == nullptr) {
		return nullptr;
	}
	return reinterpret_cast<O *>(internal::get_object_instance_binding(ret));
}

template <typename... Args>
void _call_builtin_constructor(const GDExtensionPtrConstructor constructor, GDExtensionTypePtr base, Args... args) {
	std::array<GDExtensionConstTypePtr, sizeof...(Args)> call_args = { { (GDExtensionConstTypePtr)args... } };
	constructor(base, call_args.data());
}

template <typename T, typename... Args>
T _call_builtin_method_ptr_ret(const GDExtensionPtrBuiltInMethod method, GDExtensionTypePtr base, Args... args) {
	T ret;
	std::array<GDExtensionConstTypePtr, sizeof...(Args)> call_args = { { (GDExtensionConstTypePtr)args... } };
	method(base, call_args.data(), &ret, sizeof...(Args));
	return ret;
}

template <typename... Args>
void _call_builtin_method_ptr_no_ret(const GDExtensionPtrBuiltInMethod method, GDExtensionTypePtr base, Args... args) {
	std::array<GDExtensionConstTypePtr, sizeof...(Args)> call_args = { { (GDExtensionConstTypePtr)args... } };
	method(base, call_args.data(), nullptr, sizeof...(Args));
}

template <typename T>
T _call_builtin_operator_ptr(const GDExtensionPtrOperatorEvaluator op, GDExtensionConstTypePtr left, GDExtensionConstTypePtr right) {
	T ret;
	op(left, right, &ret);
	return ret;
}

template <typename T>
T _call_builtin_ptr_getter(const GDExtensionPtrGetter getter, GDExtensionConstTypePtr base) {
	T ret;
	getter(base, &ret);
	return ret;
}

} // namespace internal

} // namespace godot
