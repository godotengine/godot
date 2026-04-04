/**************************************************************************/
/*  flow_container.cpp                                                    */
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

#include <godot_cpp/classes/flow_container.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

int32_t FlowContainer::get_line_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FlowContainer::get_class_static()._native_ptr(), StringName("get_line_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void FlowContainer::set_alignment(FlowContainer::AlignmentMode p_alignment) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FlowContainer::get_class_static()._native_ptr(), StringName("set_alignment")._native_ptr(), 575250951);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_alignment_encoded;
	PtrToArg<int64_t>::encode(p_alignment, &p_alignment_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_alignment_encoded);
}

FlowContainer::AlignmentMode FlowContainer::get_alignment() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FlowContainer::get_class_static()._native_ptr(), StringName("get_alignment")._native_ptr(), 3749743559);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (FlowContainer::AlignmentMode(0)));
	return (FlowContainer::AlignmentMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void FlowContainer::set_last_wrap_alignment(FlowContainer::LastWrapAlignmentMode p_last_wrap_alignment) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FlowContainer::get_class_static()._native_ptr(), StringName("set_last_wrap_alignment")._native_ptr(), 2899697495);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_last_wrap_alignment_encoded;
	PtrToArg<int64_t>::encode(p_last_wrap_alignment, &p_last_wrap_alignment_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_last_wrap_alignment_encoded);
}

FlowContainer::LastWrapAlignmentMode FlowContainer::get_last_wrap_alignment() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FlowContainer::get_class_static()._native_ptr(), StringName("get_last_wrap_alignment")._native_ptr(), 3743456014);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (FlowContainer::LastWrapAlignmentMode(0)));
	return (FlowContainer::LastWrapAlignmentMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void FlowContainer::set_vertical(bool p_vertical) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FlowContainer::get_class_static()._native_ptr(), StringName("set_vertical")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_vertical_encoded;
	PtrToArg<bool>::encode(p_vertical, &p_vertical_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_vertical_encoded);
}

bool FlowContainer::is_vertical() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FlowContainer::get_class_static()._native_ptr(), StringName("is_vertical")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void FlowContainer::set_reverse_fill(bool p_reverse_fill) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FlowContainer::get_class_static()._native_ptr(), StringName("set_reverse_fill")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_reverse_fill_encoded;
	PtrToArg<bool>::encode(p_reverse_fill, &p_reverse_fill_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_reverse_fill_encoded);
}

bool FlowContainer::is_reverse_fill() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(FlowContainer::get_class_static()._native_ptr(), StringName("is_reverse_fill")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

} // namespace godot
