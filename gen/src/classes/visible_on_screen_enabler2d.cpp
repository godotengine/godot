/**************************************************************************/
/*  visible_on_screen_enabler2d.cpp                                       */
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

#include <godot_cpp/classes/visible_on_screen_enabler2d.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

void VisibleOnScreenEnabler2D::set_enable_mode(VisibleOnScreenEnabler2D::EnableMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VisibleOnScreenEnabler2D::get_class_static()._native_ptr(), StringName("set_enable_mode")._native_ptr(), 2961788752);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

VisibleOnScreenEnabler2D::EnableMode VisibleOnScreenEnabler2D::get_enable_mode() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VisibleOnScreenEnabler2D::get_class_static()._native_ptr(), StringName("get_enable_mode")._native_ptr(), 2650445576);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (VisibleOnScreenEnabler2D::EnableMode(0)));
	return (VisibleOnScreenEnabler2D::EnableMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void VisibleOnScreenEnabler2D::set_enable_node_path(const NodePath &p_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VisibleOnScreenEnabler2D::get_class_static()._native_ptr(), StringName("set_enable_node_path")._native_ptr(), 1348162250);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_path);
}

NodePath VisibleOnScreenEnabler2D::get_enable_node_path() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VisibleOnScreenEnabler2D::get_class_static()._native_ptr(), StringName("get_enable_node_path")._native_ptr(), 277076166);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (NodePath()));
	return ::godot::internal::_call_native_mb_ret<NodePath>(_gde_method_bind, _owner);
}

} // namespace godot
