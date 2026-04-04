/**************************************************************************/
/*  window.cpp                                                            */
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

#include <godot_cpp/classes/window.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/font.hpp>
#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/classes/style_box.hpp>
#include <godot_cpp/classes/texture2d.hpp>
#include <godot_cpp/classes/theme.hpp>

namespace godot {

void Window::set_title(const String &p_title) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("set_title")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_title);
}

String Window::get_title() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("get_title")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

void Window::set_initial_position(Window::WindowInitialPosition p_initial_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("set_initial_position")._native_ptr(), 4084468099);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_initial_position_encoded;
	PtrToArg<int64_t>::encode(p_initial_position, &p_initial_position_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_initial_position_encoded);
}

Window::WindowInitialPosition Window::get_initial_position() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("get_initial_position")._native_ptr(), 4294066647);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Window::WindowInitialPosition(0)));
	return (Window::WindowInitialPosition)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Window::set_current_screen(int32_t p_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("set_current_screen")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded);
}

int32_t Window::get_current_screen() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("get_current_screen")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Window::set_position(const Vector2i &p_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("set_position")._native_ptr(), 1130785943);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_position);
}

Vector2i Window::get_position() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("get_position")._native_ptr(), 3690982128);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2i()));
	return ::godot::internal::_call_native_mb_ret<Vector2i>(_gde_method_bind, _owner);
}

void Window::move_to_center() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("move_to_center")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void Window::set_size(const Vector2i &p_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("set_size")._native_ptr(), 1130785943);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_size);
}

Vector2i Window::get_size() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("get_size")._native_ptr(), 3690982128);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2i()));
	return ::godot::internal::_call_native_mb_ret<Vector2i>(_gde_method_bind, _owner);
}

void Window::reset_size() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("reset_size")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

Vector2i Window::get_position_with_decorations() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("get_position_with_decorations")._native_ptr(), 3690982128);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2i()));
	return ::godot::internal::_call_native_mb_ret<Vector2i>(_gde_method_bind, _owner);
}

Vector2i Window::get_size_with_decorations() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("get_size_with_decorations")._native_ptr(), 3690982128);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2i()));
	return ::godot::internal::_call_native_mb_ret<Vector2i>(_gde_method_bind, _owner);
}

void Window::set_max_size(const Vector2i &p_max_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("set_max_size")._native_ptr(), 1130785943);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_max_size);
}

Vector2i Window::get_max_size() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("get_max_size")._native_ptr(), 3690982128);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2i()));
	return ::godot::internal::_call_native_mb_ret<Vector2i>(_gde_method_bind, _owner);
}

void Window::set_min_size(const Vector2i &p_min_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("set_min_size")._native_ptr(), 1130785943);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_min_size);
}

Vector2i Window::get_min_size() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("get_min_size")._native_ptr(), 3690982128);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2i()));
	return ::godot::internal::_call_native_mb_ret<Vector2i>(_gde_method_bind, _owner);
}

void Window::set_mode(Window::Mode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("set_mode")._native_ptr(), 3095236531);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

Window::Mode Window::get_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("get_mode")._native_ptr(), 2566346114);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Window::Mode(0)));
	return (Window::Mode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Window::set_flag(Window::Flags p_flag, bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("set_flag")._native_ptr(), 3426449779);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_flag_encoded;
	PtrToArg<int64_t>::encode(p_flag, &p_flag_encoded);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_flag_encoded, &p_enabled_encoded);
}

bool Window::get_flag(Window::Flags p_flag) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("get_flag")._native_ptr(), 3062752289);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_flag_encoded;
	PtrToArg<int64_t>::encode(p_flag, &p_flag_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_flag_encoded);
}

bool Window::is_maximize_allowed() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("is_maximize_allowed")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Window::request_attention() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("request_attention")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void Window::move_to_foreground() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("move_to_foreground")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void Window::set_visible(bool p_visible) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("set_visible")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_visible_encoded;
	PtrToArg<bool>::encode(p_visible, &p_visible_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_visible_encoded);
}

bool Window::is_visible() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("is_visible")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Window::hide() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("hide")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void Window::show() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("show")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void Window::set_transient(bool p_transient) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("set_transient")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_transient_encoded;
	PtrToArg<bool>::encode(p_transient, &p_transient_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_transient_encoded);
}

bool Window::is_transient() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("is_transient")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Window::set_transient_to_focused(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("set_transient_to_focused")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool Window::is_transient_to_focused() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("is_transient_to_focused")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Window::set_exclusive(bool p_exclusive) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("set_exclusive")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_exclusive_encoded;
	PtrToArg<bool>::encode(p_exclusive, &p_exclusive_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_exclusive_encoded);
}

bool Window::is_exclusive() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("is_exclusive")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Window::set_unparent_when_invisible(bool p_unparent) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("set_unparent_when_invisible")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_unparent_encoded;
	PtrToArg<bool>::encode(p_unparent, &p_unparent_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_unparent_encoded);
}

bool Window::can_draw() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("can_draw")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

bool Window::has_focus() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("has_focus")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Window::grab_focus() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("grab_focus")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void Window::start_drag() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("start_drag")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void Window::start_resize(DisplayServer::WindowResizeEdge p_edge) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("start_resize")._native_ptr(), 122288853);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_edge_encoded;
	PtrToArg<int64_t>::encode(p_edge, &p_edge_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_edge_encoded);
}

void Window::set_ime_active(bool p_active) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("set_ime_active")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_active_encoded;
	PtrToArg<bool>::encode(p_active, &p_active_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_active_encoded);
}

void Window::set_ime_position(const Vector2i &p_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("set_ime_position")._native_ptr(), 1130785943);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_position);
}

bool Window::is_embedded() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("is_embedded")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

Vector2 Window::get_contents_minimum_size() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("get_contents_minimum_size")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

void Window::set_force_native(bool p_force_native) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("set_force_native")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_force_native_encoded;
	PtrToArg<bool>::encode(p_force_native, &p_force_native_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_force_native_encoded);
}

bool Window::get_force_native() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("get_force_native")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Window::set_content_scale_size(const Vector2i &p_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("set_content_scale_size")._native_ptr(), 1130785943);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_size);
}

Vector2i Window::get_content_scale_size() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("get_content_scale_size")._native_ptr(), 3690982128);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2i()));
	return ::godot::internal::_call_native_mb_ret<Vector2i>(_gde_method_bind, _owner);
}

void Window::set_content_scale_mode(Window::ContentScaleMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("set_content_scale_mode")._native_ptr(), 2937716473);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

Window::ContentScaleMode Window::get_content_scale_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("get_content_scale_mode")._native_ptr(), 161585230);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Window::ContentScaleMode(0)));
	return (Window::ContentScaleMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Window::set_content_scale_aspect(Window::ContentScaleAspect p_aspect) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("set_content_scale_aspect")._native_ptr(), 2370399418);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_aspect_encoded;
	PtrToArg<int64_t>::encode(p_aspect, &p_aspect_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_aspect_encoded);
}

Window::ContentScaleAspect Window::get_content_scale_aspect() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("get_content_scale_aspect")._native_ptr(), 4158790715);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Window::ContentScaleAspect(0)));
	return (Window::ContentScaleAspect)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Window::set_content_scale_stretch(Window::ContentScaleStretch p_stretch) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("set_content_scale_stretch")._native_ptr(), 349355940);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_stretch_encoded;
	PtrToArg<int64_t>::encode(p_stretch, &p_stretch_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_stretch_encoded);
}

Window::ContentScaleStretch Window::get_content_scale_stretch() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("get_content_scale_stretch")._native_ptr(), 536857316);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Window::ContentScaleStretch(0)));
	return (Window::ContentScaleStretch)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Window::set_nonclient_area(const Rect2i &p_area) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("set_nonclient_area")._native_ptr(), 1763793166);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_area);
}

Rect2i Window::get_nonclient_area() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("get_nonclient_area")._native_ptr(), 410525958);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Rect2i()));
	return ::godot::internal::_call_native_mb_ret<Rect2i>(_gde_method_bind, _owner);
}

void Window::set_keep_title_visible(bool p_title_visible) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("set_keep_title_visible")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_title_visible_encoded;
	PtrToArg<bool>::encode(p_title_visible, &p_title_visible_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_title_visible_encoded);
}

bool Window::get_keep_title_visible() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("get_keep_title_visible")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Window::set_content_scale_factor(float p_factor) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("set_content_scale_factor")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_factor_encoded;
	PtrToArg<double>::encode(p_factor, &p_factor_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_factor_encoded);
}

float Window::get_content_scale_factor() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("get_content_scale_factor")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Window::set_mouse_passthrough_polygon(const PackedVector2Array &p_polygon) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("set_mouse_passthrough_polygon")._native_ptr(), 1509147220);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_polygon);
}

PackedVector2Array Window::get_mouse_passthrough_polygon() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("get_mouse_passthrough_polygon")._native_ptr(), 2961356807);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedVector2Array()));
	return ::godot::internal::_call_native_mb_ret<PackedVector2Array>(_gde_method_bind, _owner);
}

void Window::set_wrap_controls(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("set_wrap_controls")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool Window::is_wrapping_controls() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("is_wrapping_controls")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Window::child_controls_changed() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("child_controls_changed")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void Window::set_theme(const Ref<Theme> &p_theme) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("set_theme")._native_ptr(), 2326690814);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_theme != nullptr ? &p_theme->_owner : nullptr));
}

Ref<Theme> Window::get_theme() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("get_theme")._native_ptr(), 3846893731);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Theme>()));
	return Ref<Theme>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Theme>(_gde_method_bind, _owner));
}

void Window::set_theme_type_variation(const StringName &p_theme_type) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("set_theme_type_variation")._native_ptr(), 3304788590);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_theme_type);
}

StringName Window::get_theme_type_variation() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("get_theme_type_variation")._native_ptr(), 2002593661);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (StringName()));
	return ::godot::internal::_call_native_mb_ret<StringName>(_gde_method_bind, _owner);
}

void Window::begin_bulk_theme_override() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("begin_bulk_theme_override")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void Window::end_bulk_theme_override() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("end_bulk_theme_override")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void Window::add_theme_icon_override(const StringName &p_name, const Ref<Texture2D> &p_texture) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("add_theme_icon_override")._native_ptr(), 1373065600);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name, (p_texture != nullptr ? &p_texture->_owner : nullptr));
}

void Window::add_theme_stylebox_override(const StringName &p_name, const Ref<StyleBox> &p_stylebox) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("add_theme_stylebox_override")._native_ptr(), 4188838905);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name, (p_stylebox != nullptr ? &p_stylebox->_owner : nullptr));
}

void Window::add_theme_font_override(const StringName &p_name, const Ref<Font> &p_font) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("add_theme_font_override")._native_ptr(), 3518018674);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name, (p_font != nullptr ? &p_font->_owner : nullptr));
}

void Window::add_theme_font_size_override(const StringName &p_name, int32_t p_font_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("add_theme_font_size_override")._native_ptr(), 2415702435);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_font_size_encoded;
	PtrToArg<int64_t>::encode(p_font_size, &p_font_size_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name, &p_font_size_encoded);
}

void Window::add_theme_color_override(const StringName &p_name, const Color &p_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("add_theme_color_override")._native_ptr(), 4260178595);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name, &p_color);
}

void Window::add_theme_constant_override(const StringName &p_name, int32_t p_constant) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("add_theme_constant_override")._native_ptr(), 2415702435);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_constant_encoded;
	PtrToArg<int64_t>::encode(p_constant, &p_constant_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name, &p_constant_encoded);
}

void Window::remove_theme_icon_override(const StringName &p_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("remove_theme_icon_override")._native_ptr(), 3304788590);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name);
}

void Window::remove_theme_stylebox_override(const StringName &p_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("remove_theme_stylebox_override")._native_ptr(), 3304788590);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name);
}

void Window::remove_theme_font_override(const StringName &p_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("remove_theme_font_override")._native_ptr(), 3304788590);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name);
}

void Window::remove_theme_font_size_override(const StringName &p_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("remove_theme_font_size_override")._native_ptr(), 3304788590);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name);
}

void Window::remove_theme_color_override(const StringName &p_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("remove_theme_color_override")._native_ptr(), 3304788590);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name);
}

void Window::remove_theme_constant_override(const StringName &p_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("remove_theme_constant_override")._native_ptr(), 3304788590);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name);
}

Ref<Texture2D> Window::get_theme_icon(const StringName &p_name, const StringName &p_theme_type) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("get_theme_icon")._native_ptr(), 3163973443);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Texture2D>()));
	return Ref<Texture2D>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Texture2D>(_gde_method_bind, _owner, &p_name, &p_theme_type));
}

Ref<StyleBox> Window::get_theme_stylebox(const StringName &p_name, const StringName &p_theme_type) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("get_theme_stylebox")._native_ptr(), 604739069);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<StyleBox>()));
	return Ref<StyleBox>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<StyleBox>(_gde_method_bind, _owner, &p_name, &p_theme_type));
}

Ref<Font> Window::get_theme_font(const StringName &p_name, const StringName &p_theme_type) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("get_theme_font")._native_ptr(), 2826986490);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Font>()));
	return Ref<Font>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Font>(_gde_method_bind, _owner, &p_name, &p_theme_type));
}

int32_t Window::get_theme_font_size(const StringName &p_name, const StringName &p_theme_type) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("get_theme_font_size")._native_ptr(), 1327056374);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_name, &p_theme_type);
}

Color Window::get_theme_color(const StringName &p_name, const StringName &p_theme_type) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("get_theme_color")._native_ptr(), 2798751242);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Color()));
	return ::godot::internal::_call_native_mb_ret<Color>(_gde_method_bind, _owner, &p_name, &p_theme_type);
}

int32_t Window::get_theme_constant(const StringName &p_name, const StringName &p_theme_type) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("get_theme_constant")._native_ptr(), 1327056374);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_name, &p_theme_type);
}

bool Window::has_theme_icon_override(const StringName &p_name) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("has_theme_icon_override")._native_ptr(), 2619796661);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_name);
}

bool Window::has_theme_stylebox_override(const StringName &p_name) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("has_theme_stylebox_override")._native_ptr(), 2619796661);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_name);
}

bool Window::has_theme_font_override(const StringName &p_name) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("has_theme_font_override")._native_ptr(), 2619796661);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_name);
}

bool Window::has_theme_font_size_override(const StringName &p_name) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("has_theme_font_size_override")._native_ptr(), 2619796661);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_name);
}

bool Window::has_theme_color_override(const StringName &p_name) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("has_theme_color_override")._native_ptr(), 2619796661);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_name);
}

bool Window::has_theme_constant_override(const StringName &p_name) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("has_theme_constant_override")._native_ptr(), 2619796661);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_name);
}

bool Window::has_theme_icon(const StringName &p_name, const StringName &p_theme_type) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("has_theme_icon")._native_ptr(), 866386512);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_name, &p_theme_type);
}

bool Window::has_theme_stylebox(const StringName &p_name, const StringName &p_theme_type) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("has_theme_stylebox")._native_ptr(), 866386512);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_name, &p_theme_type);
}

bool Window::has_theme_font(const StringName &p_name, const StringName &p_theme_type) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("has_theme_font")._native_ptr(), 866386512);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_name, &p_theme_type);
}

bool Window::has_theme_font_size(const StringName &p_name, const StringName &p_theme_type) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("has_theme_font_size")._native_ptr(), 866386512);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_name, &p_theme_type);
}

bool Window::has_theme_color(const StringName &p_name, const StringName &p_theme_type) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("has_theme_color")._native_ptr(), 866386512);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_name, &p_theme_type);
}

bool Window::has_theme_constant(const StringName &p_name, const StringName &p_theme_type) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("has_theme_constant")._native_ptr(), 866386512);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_name, &p_theme_type);
}

float Window::get_theme_default_base_scale() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("get_theme_default_base_scale")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

Ref<Font> Window::get_theme_default_font() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("get_theme_default_font")._native_ptr(), 3229501585);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Font>()));
	return Ref<Font>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Font>(_gde_method_bind, _owner));
}

int32_t Window::get_theme_default_font_size() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("get_theme_default_font_size")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

int32_t Window::get_window_id() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("get_window_id")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Window::set_accessibility_name(const String &p_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("set_accessibility_name")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name);
}

String Window::get_accessibility_name() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("get_accessibility_name")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

void Window::set_accessibility_description(const String &p_description) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("set_accessibility_description")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_description);
}

String Window::get_accessibility_description() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("get_accessibility_description")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

Window *Window::get_focused_window() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("get_focused_window")._native_ptr(), 1835468782);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<Window>(_gde_method_bind, nullptr);
}

void Window::set_layout_direction(Window::LayoutDirection p_direction) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("set_layout_direction")._native_ptr(), 3094704184);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_direction_encoded;
	PtrToArg<int64_t>::encode(p_direction, &p_direction_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_direction_encoded);
}

Window::LayoutDirection Window::get_layout_direction() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("get_layout_direction")._native_ptr(), 3909617982);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Window::LayoutDirection(0)));
	return (Window::LayoutDirection)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

bool Window::is_layout_rtl() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("is_layout_rtl")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Window::set_auto_translate(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("set_auto_translate")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool Window::is_auto_translating() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("is_auto_translating")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Window::set_use_font_oversampling(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("set_use_font_oversampling")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool Window::is_using_font_oversampling() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("is_using_font_oversampling")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Window::popup(const Rect2i &p_rect) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("popup")._native_ptr(), 1680304321);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_rect);
}

void Window::popup_on_parent(const Rect2i &p_parent_rect) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("popup_on_parent")._native_ptr(), 1763793166);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_parent_rect);
}

void Window::popup_centered(const Vector2i &p_minsize) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("popup_centered")._native_ptr(), 3447975422);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_minsize);
}

void Window::popup_centered_ratio(float p_ratio) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("popup_centered_ratio")._native_ptr(), 1014814997);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_ratio_encoded;
	PtrToArg<double>::encode(p_ratio, &p_ratio_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_ratio_encoded);
}

void Window::popup_centered_clamped(const Vector2i &p_minsize, float p_fallback_ratio) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("popup_centered_clamped")._native_ptr(), 2613752477);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_fallback_ratio_encoded;
	PtrToArg<double>::encode(p_fallback_ratio, &p_fallback_ratio_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_minsize, &p_fallback_ratio_encoded);
}

void Window::popup_exclusive(Node *p_from_node, const Rect2i &p_rect) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("popup_exclusive")._native_ptr(), 2134721627);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_from_node != nullptr ? &p_from_node->_owner : nullptr), &p_rect);
}

void Window::popup_exclusive_on_parent(Node *p_from_node, const Rect2i &p_parent_rect) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("popup_exclusive_on_parent")._native_ptr(), 2344671043);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_from_node != nullptr ? &p_from_node->_owner : nullptr), &p_parent_rect);
}

void Window::popup_exclusive_centered(Node *p_from_node, const Vector2i &p_minsize) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("popup_exclusive_centered")._native_ptr(), 3357594017);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_from_node != nullptr ? &p_from_node->_owner : nullptr), &p_minsize);
}

void Window::popup_exclusive_centered_ratio(Node *p_from_node, float p_ratio) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("popup_exclusive_centered_ratio")._native_ptr(), 2284776287);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_ratio_encoded;
	PtrToArg<double>::encode(p_ratio, &p_ratio_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_from_node != nullptr ? &p_from_node->_owner : nullptr), &p_ratio_encoded);
}

void Window::popup_exclusive_centered_clamped(Node *p_from_node, const Vector2i &p_minsize, float p_fallback_ratio) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Window::get_class_static()._native_ptr(), StringName("popup_exclusive_centered_clamped")._native_ptr(), 2612708785);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_fallback_ratio_encoded;
	PtrToArg<double>::encode(p_fallback_ratio, &p_fallback_ratio_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_from_node != nullptr ? &p_from_node->_owner : nullptr), &p_minsize, &p_fallback_ratio_encoded);
}

Vector2 Window::_get_contents_minimum_size() const {
	return Vector2();
}

} // namespace godot
