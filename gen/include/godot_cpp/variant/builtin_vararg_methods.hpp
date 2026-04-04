/**************************************************************************/
/*  builtin_vararg_methods.hpp                                            */
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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#pragma once

template <typename... Args>
Variant Callable::call(const Args &...p_args) const {
	std::array<Variant, 0 + sizeof...(Args)> variant_args{{ Variant(p_args)... }};
	std::array<const Variant *, 0 + sizeof...(Args)> call_args;
	for (size_t i = 0; i < variant_args.size(); i++) {
		call_args[i] = &variant_args[i];
	}
	Variant ret;
	_method_bindings.method_call((GDExtensionTypePtr)&opaque, reinterpret_cast<GDExtensionConstTypePtr *>(call_args.data()), &ret, 0 + sizeof...(Args));
	return ret;
}

template <typename... Args>
void Callable::call_deferred(const Args &...p_args) const {
	std::array<Variant, 0 + sizeof...(Args)> variant_args{{ Variant(p_args)... }};
	std::array<const Variant *, 0 + sizeof...(Args)> call_args;
	for (size_t i = 0; i < variant_args.size(); i++) {
		call_args[i] = &variant_args[i];
	}
	_method_bindings.method_call_deferred((GDExtensionTypePtr)&opaque, reinterpret_cast<GDExtensionConstTypePtr *>(call_args.data()), nullptr, 0 + sizeof...(Args));
}

template <typename... Args>
void Callable::rpc(const Args &...p_args) const {
	std::array<Variant, 0 + sizeof...(Args)> variant_args{{ Variant(p_args)... }};
	std::array<const Variant *, 0 + sizeof...(Args)> call_args;
	for (size_t i = 0; i < variant_args.size(); i++) {
		call_args[i] = &variant_args[i];
	}
	_method_bindings.method_rpc((GDExtensionTypePtr)&opaque, reinterpret_cast<GDExtensionConstTypePtr *>(call_args.data()), nullptr, 0 + sizeof...(Args));
}

template <typename... Args>
void Callable::rpc_id(int64_t p_peer_id, const Args &...p_args) const {
	std::array<Variant, 1 + sizeof...(Args)> variant_args{{ Variant(p_peer_id), Variant(p_args)... }};
	std::array<const Variant *, 1 + sizeof...(Args)> call_args;
	for (size_t i = 0; i < variant_args.size(); i++) {
		call_args[i] = &variant_args[i];
	}
	_method_bindings.method_rpc_id((GDExtensionTypePtr)&opaque, reinterpret_cast<GDExtensionConstTypePtr *>(call_args.data()), nullptr, 1 + sizeof...(Args));
}

template <typename... Args>
Callable Callable::bind(const Args &...p_args) const {
	std::array<Variant, 0 + sizeof...(Args)> variant_args{{ Variant(p_args)... }};
	std::array<const Variant *, 0 + sizeof...(Args)> call_args;
	for (size_t i = 0; i < variant_args.size(); i++) {
		call_args[i] = &variant_args[i];
	}
	Callable ret;
	_method_bindings.method_bind((GDExtensionTypePtr)&opaque, reinterpret_cast<GDExtensionConstTypePtr *>(call_args.data()), &ret, 0 + sizeof...(Args));
	return ret;
}

template <typename... Args>
void Signal::emit(const Args &...p_args) const {
	std::array<Variant, 0 + sizeof...(Args)> variant_args{{ Variant(p_args)... }};
	std::array<const Variant *, 0 + sizeof...(Args)> call_args;
	for (size_t i = 0; i < variant_args.size(); i++) {
		call_args[i] = &variant_args[i];
	}
	_method_bindings.method_emit((GDExtensionTypePtr)&opaque, reinterpret_cast<GDExtensionConstTypePtr *>(call_args.data()), nullptr, 0 + sizeof...(Args));
}
