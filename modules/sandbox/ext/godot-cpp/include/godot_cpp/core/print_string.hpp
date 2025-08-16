/**************************************************************************/
/*  print_string.hpp                                                      */
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

#include <godot_cpp/variant/utility_functions.hpp>

namespace godot {
inline void print_error(const Variant &p_variant) {
	UtilityFunctions::printerr(p_variant);
}

inline void print_line(const Variant &p_variant) {
	UtilityFunctions::print(p_variant);
}

inline void print_line_rich(const Variant &p_variant) {
	UtilityFunctions::print_rich(p_variant);
}

template <typename... Args>
void print_error(const Variant &p_variant, Args... p_args) {
	UtilityFunctions::printerr(p_variant, p_args...);
}

template <typename... Args>
void print_line(const Variant &p_variant, Args... p_args) {
	UtilityFunctions::print(p_variant, p_args...);
}

template <typename... Args>
void print_line_rich(const Variant &p_variant, Args... p_args) {
	UtilityFunctions::print_rich(p_variant, p_args...);
}

template <typename... Args>
void print_verbose(const Variant &p_variant, Args... p_args) {
	UtilityFunctions::print_verbose(p_variant, p_args...);
}

bool is_print_verbose_enabled();

} // namespace godot
