/**************************************************************************/
/*  method_bind.cpp                                                       */
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

#include <godot_cpp/core/method_bind.hpp>

namespace godot {

void MethodBind::_set_const(bool p_const) {
	_const = p_const;
}

void MethodBind::_set_static(bool p_static) {
	_static = p_static;
}

void MethodBind::_set_returns(bool p_returns) {
	_returns = p_returns;
}

void MethodBind::_set_vararg(bool p_vararg) {
	_vararg = p_vararg;
}

StringName MethodBind::get_name() const {
	return name;
}

void MethodBind::set_name(const StringName &p_name) {
	name = p_name;
}

void MethodBind::set_argument_names(const std::vector<StringName> &p_names) {
	argument_names = p_names;
}

std::vector<StringName> MethodBind::get_argument_names() const {
	return argument_names;
}

void MethodBind::_generate_argument_types(int p_count) {
	set_argument_count(p_count);

	if (argument_types != nullptr) {
		memdelete_arr(argument_types);
	}

	argument_types = memnew_arr(GDExtensionVariantType, p_count + 1);

	// -1 means return type.
	for (int i = -1; i < p_count; i++) {
		argument_types[i + 1] = gen_argument_type(i);
	}
}

PropertyInfo MethodBind::get_argument_info(int p_argument) const {
	PropertyInfo info = gen_argument_type_info(p_argument);
	if (p_argument >= 0) {
		info.name = p_argument < (int)argument_names.size() ? argument_names[p_argument] : "";
	}
	return info;
}

void MethodBind::bind_call(void *p_method_userdata, GDExtensionClassInstancePtr p_instance, const GDExtensionConstVariantPtr *p_args, GDExtensionInt p_argument_count, GDExtensionVariantPtr r_return, GDExtensionCallError *r_error) {
	const MethodBind *bind = reinterpret_cast<const MethodBind *>(p_method_userdata);
	Variant ret = bind->call(p_instance, p_args, p_argument_count, *r_error);
	// This assumes the return value is an empty Variant, so it doesn't need to call the destructor first.
	// Since only GDExtensionMethodBind calls this from the Godot side, it should always be the case.
	internal::gdextension_interface_variant_new_copy(r_return, ret._native_ptr());
}

void MethodBind::bind_ptrcall(void *p_method_userdata, GDExtensionClassInstancePtr p_instance, const GDExtensionConstTypePtr *p_args, GDExtensionTypePtr r_return) {
	const MethodBind *bind = reinterpret_cast<const MethodBind *>(p_method_userdata);
	bind->ptrcall(p_instance, p_args, r_return);
}

MethodBind::~MethodBind() {
	if (argument_types) {
		memdelete_arr(argument_types);
	}
}

} // namespace godot
