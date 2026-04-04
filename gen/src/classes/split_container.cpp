/**************************************************************************/
/*  split_container.cpp                                                   */
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

#include <godot_cpp/classes/split_container.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/control.hpp>

namespace godot {

void SplitContainer::set_split_offsets(const PackedInt32Array &p_offsets) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SplitContainer::get_class_static()._native_ptr(), StringName("set_split_offsets")._native_ptr(), 3614634198);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_offsets);
}

PackedInt32Array SplitContainer::get_split_offsets() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SplitContainer::get_class_static()._native_ptr(), StringName("get_split_offsets")._native_ptr(), 1930428628);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedInt32Array()));
	return ::godot::internal::_call_native_mb_ret<PackedInt32Array>(_gde_method_bind, _owner);
}

void SplitContainer::clamp_split_offset(int32_t p_priority_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SplitContainer::get_class_static()._native_ptr(), StringName("clamp_split_offset")._native_ptr(), 1995695955);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_priority_index_encoded;
	PtrToArg<int64_t>::encode(p_priority_index, &p_priority_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_priority_index_encoded);
}

void SplitContainer::set_collapsed(bool p_collapsed) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SplitContainer::get_class_static()._native_ptr(), StringName("set_collapsed")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_collapsed_encoded;
	PtrToArg<bool>::encode(p_collapsed, &p_collapsed_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_collapsed_encoded);
}

bool SplitContainer::is_collapsed() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SplitContainer::get_class_static()._native_ptr(), StringName("is_collapsed")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void SplitContainer::set_dragger_visibility(SplitContainer::DraggerVisibility p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SplitContainer::get_class_static()._native_ptr(), StringName("set_dragger_visibility")._native_ptr(), 1168273952);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

SplitContainer::DraggerVisibility SplitContainer::get_dragger_visibility() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SplitContainer::get_class_static()._native_ptr(), StringName("get_dragger_visibility")._native_ptr(), 967297479);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (SplitContainer::DraggerVisibility(0)));
	return (SplitContainer::DraggerVisibility)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void SplitContainer::set_vertical(bool p_vertical) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SplitContainer::get_class_static()._native_ptr(), StringName("set_vertical")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_vertical_encoded;
	PtrToArg<bool>::encode(p_vertical, &p_vertical_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_vertical_encoded);
}

bool SplitContainer::is_vertical() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SplitContainer::get_class_static()._native_ptr(), StringName("is_vertical")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void SplitContainer::set_dragging_enabled(bool p_dragging_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SplitContainer::get_class_static()._native_ptr(), StringName("set_dragging_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_dragging_enabled_encoded;
	PtrToArg<bool>::encode(p_dragging_enabled, &p_dragging_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_dragging_enabled_encoded);
}

bool SplitContainer::is_dragging_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SplitContainer::get_class_static()._native_ptr(), StringName("is_dragging_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void SplitContainer::set_drag_area_margin_begin(int32_t p_margin) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SplitContainer::get_class_static()._native_ptr(), StringName("set_drag_area_margin_begin")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_margin_encoded;
	PtrToArg<int64_t>::encode(p_margin, &p_margin_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_margin_encoded);
}

int32_t SplitContainer::get_drag_area_margin_begin() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SplitContainer::get_class_static()._native_ptr(), StringName("get_drag_area_margin_begin")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void SplitContainer::set_drag_area_margin_end(int32_t p_margin) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SplitContainer::get_class_static()._native_ptr(), StringName("set_drag_area_margin_end")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_margin_encoded;
	PtrToArg<int64_t>::encode(p_margin, &p_margin_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_margin_encoded);
}

int32_t SplitContainer::get_drag_area_margin_end() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SplitContainer::get_class_static()._native_ptr(), StringName("get_drag_area_margin_end")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void SplitContainer::set_drag_area_offset(int32_t p_offset) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SplitContainer::get_class_static()._native_ptr(), StringName("set_drag_area_offset")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_offset_encoded;
	PtrToArg<int64_t>::encode(p_offset, &p_offset_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_offset_encoded);
}

int32_t SplitContainer::get_drag_area_offset() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SplitContainer::get_class_static()._native_ptr(), StringName("get_drag_area_offset")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void SplitContainer::set_drag_area_highlight_in_editor(bool p_drag_area_highlight_in_editor) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SplitContainer::get_class_static()._native_ptr(), StringName("set_drag_area_highlight_in_editor")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_drag_area_highlight_in_editor_encoded;
	PtrToArg<bool>::encode(p_drag_area_highlight_in_editor, &p_drag_area_highlight_in_editor_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_drag_area_highlight_in_editor_encoded);
}

bool SplitContainer::is_drag_area_highlight_in_editor_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SplitContainer::get_class_static()._native_ptr(), StringName("is_drag_area_highlight_in_editor_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

TypedArray<Control> SplitContainer::get_drag_area_controls() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SplitContainer::get_class_static()._native_ptr(), StringName("get_drag_area_controls")._native_ptr(), 2915620761);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<Control>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<Control>>(_gde_method_bind, _owner);
}

void SplitContainer::set_touch_dragger_enabled(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SplitContainer::get_class_static()._native_ptr(), StringName("set_touch_dragger_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool SplitContainer::is_touch_dragger_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SplitContainer::get_class_static()._native_ptr(), StringName("is_touch_dragger_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

Control *SplitContainer::get_drag_area_control() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SplitContainer::get_class_static()._native_ptr(), StringName("get_drag_area_control")._native_ptr(), 829782337);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<Control>(_gde_method_bind, _owner);
}

void SplitContainer::set_split_offset(int32_t p_offset) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SplitContainer::get_class_static()._native_ptr(), StringName("set_split_offset")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_offset_encoded;
	PtrToArg<int64_t>::encode(p_offset, &p_offset_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_offset_encoded);
}

int32_t SplitContainer::get_split_offset() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SplitContainer::get_class_static()._native_ptr(), StringName("get_split_offset")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

} // namespace godot
