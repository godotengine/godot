/**************************************************************************/
/*  text_edit.cpp                                                         */
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

#include <godot_cpp/classes/text_edit.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/h_scroll_bar.hpp>
#include <godot_cpp/classes/popup_menu.hpp>
#include <godot_cpp/classes/syntax_highlighter.hpp>
#include <godot_cpp/classes/texture2d.hpp>
#include <godot_cpp/classes/v_scroll_bar.hpp>
#include <godot_cpp/variant/callable.hpp>

namespace godot {

bool TextEdit::has_ime_text() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("has_ime_text")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void TextEdit::cancel_ime() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("cancel_ime")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void TextEdit::apply_ime() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("apply_ime")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void TextEdit::set_editable(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_editable")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool TextEdit::is_editable() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("is_editable")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void TextEdit::set_text_direction(Control::TextDirection p_direction) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_text_direction")._native_ptr(), 119160795);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_direction_encoded;
	PtrToArg<int64_t>::encode(p_direction, &p_direction_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_direction_encoded);
}

Control::TextDirection TextEdit::get_text_direction() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_text_direction")._native_ptr(), 797257663);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Control::TextDirection(0)));
	return (Control::TextDirection)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void TextEdit::set_language(const String &p_language) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_language")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_language);
}

String TextEdit::get_language() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_language")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

void TextEdit::set_structured_text_bidi_override(TextServer::StructuredTextParser p_parser) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_structured_text_bidi_override")._native_ptr(), 55961453);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_parser_encoded;
	PtrToArg<int64_t>::encode(p_parser, &p_parser_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_parser_encoded);
}

TextServer::StructuredTextParser TextEdit::get_structured_text_bidi_override() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_structured_text_bidi_override")._native_ptr(), 3385126229);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TextServer::StructuredTextParser(0)));
	return (TextServer::StructuredTextParser)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void TextEdit::set_structured_text_bidi_override_options(const Array &p_args) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_structured_text_bidi_override_options")._native_ptr(), 381264803);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_args);
}

Array TextEdit::get_structured_text_bidi_override_options() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_structured_text_bidi_override_options")._native_ptr(), 3995934104);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Array()));
	return ::godot::internal::_call_native_mb_ret<Array>(_gde_method_bind, _owner);
}

void TextEdit::set_tab_size(int32_t p_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_tab_size")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_size_encoded);
}

int32_t TextEdit::get_tab_size() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_tab_size")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void TextEdit::set_indent_wrapped_lines(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_indent_wrapped_lines")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool TextEdit::is_indent_wrapped_lines() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("is_indent_wrapped_lines")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void TextEdit::set_tab_input_mode(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_tab_input_mode")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool TextEdit::get_tab_input_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_tab_input_mode")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void TextEdit::set_overtype_mode_enabled(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_overtype_mode_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool TextEdit::is_overtype_mode_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("is_overtype_mode_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void TextEdit::set_context_menu_enabled(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_context_menu_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool TextEdit::is_context_menu_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("is_context_menu_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void TextEdit::set_emoji_menu_enabled(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_emoji_menu_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool TextEdit::is_emoji_menu_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("is_emoji_menu_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void TextEdit::set_backspace_deletes_composite_character_enabled(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_backspace_deletes_composite_character_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool TextEdit::is_backspace_deletes_composite_character_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("is_backspace_deletes_composite_character_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void TextEdit::set_shortcut_keys_enabled(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_shortcut_keys_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool TextEdit::is_shortcut_keys_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("is_shortcut_keys_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void TextEdit::set_virtual_keyboard_enabled(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_virtual_keyboard_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool TextEdit::is_virtual_keyboard_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("is_virtual_keyboard_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void TextEdit::set_virtual_keyboard_show_on_focus(bool p_show_on_focus) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_virtual_keyboard_show_on_focus")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_show_on_focus_encoded;
	PtrToArg<bool>::encode(p_show_on_focus, &p_show_on_focus_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_show_on_focus_encoded);
}

bool TextEdit::get_virtual_keyboard_show_on_focus() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_virtual_keyboard_show_on_focus")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void TextEdit::set_middle_mouse_paste_enabled(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_middle_mouse_paste_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool TextEdit::is_middle_mouse_paste_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("is_middle_mouse_paste_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void TextEdit::set_empty_selection_clipboard_enabled(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_empty_selection_clipboard_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool TextEdit::is_empty_selection_clipboard_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("is_empty_selection_clipboard_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void TextEdit::clear() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("clear")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void TextEdit::set_text(const String &p_text) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_text")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_text);
}

String TextEdit::get_text() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_text")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

int32_t TextEdit::get_line_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_line_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void TextEdit::set_placeholder(const String &p_text) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_placeholder")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_text);
}

String TextEdit::get_placeholder() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_placeholder")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

void TextEdit::set_line(int32_t p_line, const String &p_new_text) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_line")._native_ptr(), 501894301);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_line_encoded;
	PtrToArg<int64_t>::encode(p_line, &p_line_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_line_encoded, &p_new_text);
}

String TextEdit::get_line(int32_t p_line) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_line")._native_ptr(), 844755477);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_line_encoded;
	PtrToArg<int64_t>::encode(p_line, &p_line_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_line_encoded);
}

String TextEdit::get_line_with_ime(int32_t p_line) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_line_with_ime")._native_ptr(), 844755477);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_line_encoded;
	PtrToArg<int64_t>::encode(p_line, &p_line_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_line_encoded);
}

int32_t TextEdit::get_line_width(int32_t p_line, int32_t p_wrap_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_line_width")._native_ptr(), 688195400);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_line_encoded;
	PtrToArg<int64_t>::encode(p_line, &p_line_encoded);
	int64_t p_wrap_index_encoded;
	PtrToArg<int64_t>::encode(p_wrap_index, &p_wrap_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_line_encoded, &p_wrap_index_encoded);
}

int32_t TextEdit::get_line_height() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_line_height")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

int32_t TextEdit::get_indent_level(int32_t p_line) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_indent_level")._native_ptr(), 923996154);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_line_encoded;
	PtrToArg<int64_t>::encode(p_line, &p_line_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_line_encoded);
}

int32_t TextEdit::get_first_non_whitespace_column(int32_t p_line) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_first_non_whitespace_column")._native_ptr(), 923996154);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_line_encoded;
	PtrToArg<int64_t>::encode(p_line, &p_line_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_line_encoded);
}

void TextEdit::swap_lines(int32_t p_from_line, int32_t p_to_line) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("swap_lines")._native_ptr(), 3937882851);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_from_line_encoded;
	PtrToArg<int64_t>::encode(p_from_line, &p_from_line_encoded);
	int64_t p_to_line_encoded;
	PtrToArg<int64_t>::encode(p_to_line, &p_to_line_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_from_line_encoded, &p_to_line_encoded);
}

void TextEdit::insert_line_at(int32_t p_line, const String &p_text) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("insert_line_at")._native_ptr(), 501894301);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_line_encoded;
	PtrToArg<int64_t>::encode(p_line, &p_line_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_line_encoded, &p_text);
}

void TextEdit::remove_line_at(int32_t p_line, bool p_move_carets_down) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("remove_line_at")._native_ptr(), 972357352);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_line_encoded;
	PtrToArg<int64_t>::encode(p_line, &p_line_encoded);
	int8_t p_move_carets_down_encoded;
	PtrToArg<bool>::encode(p_move_carets_down, &p_move_carets_down_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_line_encoded, &p_move_carets_down_encoded);
}

void TextEdit::insert_text_at_caret(const String &p_text, int32_t p_caret_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("insert_text_at_caret")._native_ptr(), 2697778442);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_caret_index_encoded;
	PtrToArg<int64_t>::encode(p_caret_index, &p_caret_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_text, &p_caret_index_encoded);
}

void TextEdit::insert_text(const String &p_text, int32_t p_line, int32_t p_column, bool p_before_selection_begin, bool p_before_selection_end) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("insert_text")._native_ptr(), 1881564334);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_line_encoded;
	PtrToArg<int64_t>::encode(p_line, &p_line_encoded);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	int8_t p_before_selection_begin_encoded;
	PtrToArg<bool>::encode(p_before_selection_begin, &p_before_selection_begin_encoded);
	int8_t p_before_selection_end_encoded;
	PtrToArg<bool>::encode(p_before_selection_end, &p_before_selection_end_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_text, &p_line_encoded, &p_column_encoded, &p_before_selection_begin_encoded, &p_before_selection_end_encoded);
}

void TextEdit::remove_text(int32_t p_from_line, int32_t p_from_column, int32_t p_to_line, int32_t p_to_column) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("remove_text")._native_ptr(), 4275841770);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_from_line_encoded;
	PtrToArg<int64_t>::encode(p_from_line, &p_from_line_encoded);
	int64_t p_from_column_encoded;
	PtrToArg<int64_t>::encode(p_from_column, &p_from_column_encoded);
	int64_t p_to_line_encoded;
	PtrToArg<int64_t>::encode(p_to_line, &p_to_line_encoded);
	int64_t p_to_column_encoded;
	PtrToArg<int64_t>::encode(p_to_column, &p_to_column_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_from_line_encoded, &p_from_column_encoded, &p_to_line_encoded, &p_to_column_encoded);
}

int32_t TextEdit::get_last_unhidden_line() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_last_unhidden_line")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

int32_t TextEdit::get_next_visible_line_offset_from(int32_t p_line, int32_t p_visible_amount) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_next_visible_line_offset_from")._native_ptr(), 3175239445);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_line_encoded;
	PtrToArg<int64_t>::encode(p_line, &p_line_encoded);
	int64_t p_visible_amount_encoded;
	PtrToArg<int64_t>::encode(p_visible_amount, &p_visible_amount_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_line_encoded, &p_visible_amount_encoded);
}

Vector2i TextEdit::get_next_visible_line_index_offset_from(int32_t p_line, int32_t p_wrap_index, int32_t p_visible_amount) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_next_visible_line_index_offset_from")._native_ptr(), 3386475622);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2i()));
	int64_t p_line_encoded;
	PtrToArg<int64_t>::encode(p_line, &p_line_encoded);
	int64_t p_wrap_index_encoded;
	PtrToArg<int64_t>::encode(p_wrap_index, &p_wrap_index_encoded);
	int64_t p_visible_amount_encoded;
	PtrToArg<int64_t>::encode(p_visible_amount, &p_visible_amount_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector2i>(_gde_method_bind, _owner, &p_line_encoded, &p_wrap_index_encoded, &p_visible_amount_encoded);
}

void TextEdit::backspace(int32_t p_caret_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("backspace")._native_ptr(), 1025054187);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_caret_index_encoded;
	PtrToArg<int64_t>::encode(p_caret_index, &p_caret_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_caret_index_encoded);
}

void TextEdit::cut(int32_t p_caret_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("cut")._native_ptr(), 1025054187);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_caret_index_encoded;
	PtrToArg<int64_t>::encode(p_caret_index, &p_caret_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_caret_index_encoded);
}

void TextEdit::copy(int32_t p_caret_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("copy")._native_ptr(), 1025054187);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_caret_index_encoded;
	PtrToArg<int64_t>::encode(p_caret_index, &p_caret_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_caret_index_encoded);
}

void TextEdit::paste(int32_t p_caret_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("paste")._native_ptr(), 1025054187);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_caret_index_encoded;
	PtrToArg<int64_t>::encode(p_caret_index, &p_caret_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_caret_index_encoded);
}

void TextEdit::paste_primary_clipboard(int32_t p_caret_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("paste_primary_clipboard")._native_ptr(), 1025054187);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_caret_index_encoded;
	PtrToArg<int64_t>::encode(p_caret_index, &p_caret_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_caret_index_encoded);
}

void TextEdit::start_action(TextEdit::EditAction p_action) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("start_action")._native_ptr(), 2834827583);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_action_encoded;
	PtrToArg<int64_t>::encode(p_action, &p_action_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_action_encoded);
}

void TextEdit::end_action() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("end_action")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void TextEdit::begin_complex_operation() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("begin_complex_operation")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void TextEdit::end_complex_operation() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("end_complex_operation")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

bool TextEdit::has_undo() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("has_undo")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

bool TextEdit::has_redo() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("has_redo")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void TextEdit::undo() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("undo")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void TextEdit::redo() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("redo")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void TextEdit::clear_undo_history() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("clear_undo_history")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void TextEdit::tag_saved_version() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("tag_saved_version")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

uint32_t TextEdit::get_version() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_version")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

uint32_t TextEdit::get_saved_version() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_saved_version")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void TextEdit::set_search_text(const String &p_search_text) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_search_text")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_search_text);
}

void TextEdit::set_search_flags(uint32_t p_flags) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_search_flags")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_flags_encoded;
	PtrToArg<int64_t>::encode(p_flags, &p_flags_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_flags_encoded);
}

Vector2i TextEdit::search(const String &p_text, uint32_t p_flags, int32_t p_from_line, int32_t p_from_column) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("search")._native_ptr(), 1203739136);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2i()));
	int64_t p_flags_encoded;
	PtrToArg<int64_t>::encode(p_flags, &p_flags_encoded);
	int64_t p_from_line_encoded;
	PtrToArg<int64_t>::encode(p_from_line, &p_from_line_encoded);
	int64_t p_from_column_encoded;
	PtrToArg<int64_t>::encode(p_from_column, &p_from_column_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector2i>(_gde_method_bind, _owner, &p_text, &p_flags_encoded, &p_from_line_encoded, &p_from_column_encoded);
}

void TextEdit::set_tooltip_request_func(const Callable &p_callback) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_tooltip_request_func")._native_ptr(), 1611583062);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_callback);
}

Vector2 TextEdit::get_local_mouse_pos() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_local_mouse_pos")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

String TextEdit::get_word_at_pos(const Vector2 &p_position) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_word_at_pos")._native_ptr(), 3674420000);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_position);
}

Vector2i TextEdit::get_line_column_at_pos(const Vector2i &p_position, bool p_clamp_line, bool p_clamp_column) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_line_column_at_pos")._native_ptr(), 3472935744);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2i()));
	int8_t p_clamp_line_encoded;
	PtrToArg<bool>::encode(p_clamp_line, &p_clamp_line_encoded);
	int8_t p_clamp_column_encoded;
	PtrToArg<bool>::encode(p_clamp_column, &p_clamp_column_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector2i>(_gde_method_bind, _owner, &p_position, &p_clamp_line_encoded, &p_clamp_column_encoded);
}

Vector2i TextEdit::get_pos_at_line_column(int32_t p_line, int32_t p_column) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_pos_at_line_column")._native_ptr(), 410388347);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2i()));
	int64_t p_line_encoded;
	PtrToArg<int64_t>::encode(p_line, &p_line_encoded);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector2i>(_gde_method_bind, _owner, &p_line_encoded, &p_column_encoded);
}

Rect2i TextEdit::get_rect_at_line_column(int32_t p_line, int32_t p_column) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_rect_at_line_column")._native_ptr(), 3256618057);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Rect2i()));
	int64_t p_line_encoded;
	PtrToArg<int64_t>::encode(p_line, &p_line_encoded);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	return ::godot::internal::_call_native_mb_ret<Rect2i>(_gde_method_bind, _owner, &p_line_encoded, &p_column_encoded);
}

int32_t TextEdit::get_minimap_line_at_pos(const Vector2i &p_position) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_minimap_line_at_pos")._native_ptr(), 2485466453);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_position);
}

bool TextEdit::is_dragging_cursor() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("is_dragging_cursor")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

bool TextEdit::is_mouse_over_selection(bool p_edges, int32_t p_caret_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("is_mouse_over_selection")._native_ptr(), 1840282309);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int8_t p_edges_encoded;
	PtrToArg<bool>::encode(p_edges, &p_edges_encoded);
	int64_t p_caret_index_encoded;
	PtrToArg<int64_t>::encode(p_caret_index, &p_caret_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_edges_encoded, &p_caret_index_encoded);
}

void TextEdit::set_caret_type(TextEdit::CaretType p_type) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_caret_type")._native_ptr(), 1211596914);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_type_encoded;
	PtrToArg<int64_t>::encode(p_type, &p_type_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_type_encoded);
}

TextEdit::CaretType TextEdit::get_caret_type() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_caret_type")._native_ptr(), 2830252959);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TextEdit::CaretType(0)));
	return (TextEdit::CaretType)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void TextEdit::set_caret_blink_enabled(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_caret_blink_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool TextEdit::is_caret_blink_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("is_caret_blink_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void TextEdit::set_caret_blink_interval(float p_interval) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_caret_blink_interval")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_interval_encoded;
	PtrToArg<double>::encode(p_interval, &p_interval_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_interval_encoded);
}

float TextEdit::get_caret_blink_interval() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_caret_blink_interval")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void TextEdit::set_draw_caret_when_editable_disabled(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_draw_caret_when_editable_disabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool TextEdit::is_drawing_caret_when_editable_disabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("is_drawing_caret_when_editable_disabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void TextEdit::set_move_caret_on_right_click_enabled(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_move_caret_on_right_click_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool TextEdit::is_move_caret_on_right_click_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("is_move_caret_on_right_click_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void TextEdit::set_caret_mid_grapheme_enabled(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_caret_mid_grapheme_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool TextEdit::is_caret_mid_grapheme_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("is_caret_mid_grapheme_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void TextEdit::set_multiple_carets_enabled(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_multiple_carets_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool TextEdit::is_multiple_carets_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("is_multiple_carets_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

int32_t TextEdit::add_caret(int32_t p_line, int32_t p_column) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("add_caret")._native_ptr(), 50157827);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_line_encoded;
	PtrToArg<int64_t>::encode(p_line, &p_line_encoded);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_line_encoded, &p_column_encoded);
}

void TextEdit::remove_caret(int32_t p_caret) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("remove_caret")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_caret_encoded;
	PtrToArg<int64_t>::encode(p_caret, &p_caret_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_caret_encoded);
}

void TextEdit::remove_secondary_carets() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("remove_secondary_carets")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

int32_t TextEdit::get_caret_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_caret_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void TextEdit::add_caret_at_carets(bool p_below) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("add_caret_at_carets")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_below_encoded;
	PtrToArg<bool>::encode(p_below, &p_below_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_below_encoded);
}

PackedInt32Array TextEdit::get_sorted_carets(bool p_include_ignored_carets) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_sorted_carets")._native_ptr(), 2131714034);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedInt32Array()));
	int8_t p_include_ignored_carets_encoded;
	PtrToArg<bool>::encode(p_include_ignored_carets, &p_include_ignored_carets_encoded);
	return ::godot::internal::_call_native_mb_ret<PackedInt32Array>(_gde_method_bind, _owner, &p_include_ignored_carets_encoded);
}

void TextEdit::collapse_carets(int32_t p_from_line, int32_t p_from_column, int32_t p_to_line, int32_t p_to_column, bool p_inclusive) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("collapse_carets")._native_ptr(), 228654177);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_from_line_encoded;
	PtrToArg<int64_t>::encode(p_from_line, &p_from_line_encoded);
	int64_t p_from_column_encoded;
	PtrToArg<int64_t>::encode(p_from_column, &p_from_column_encoded);
	int64_t p_to_line_encoded;
	PtrToArg<int64_t>::encode(p_to_line, &p_to_line_encoded);
	int64_t p_to_column_encoded;
	PtrToArg<int64_t>::encode(p_to_column, &p_to_column_encoded);
	int8_t p_inclusive_encoded;
	PtrToArg<bool>::encode(p_inclusive, &p_inclusive_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_from_line_encoded, &p_from_column_encoded, &p_to_line_encoded, &p_to_column_encoded, &p_inclusive_encoded);
}

void TextEdit::merge_overlapping_carets() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("merge_overlapping_carets")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void TextEdit::begin_multicaret_edit() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("begin_multicaret_edit")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void TextEdit::end_multicaret_edit() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("end_multicaret_edit")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

bool TextEdit::is_in_mulitcaret_edit() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("is_in_mulitcaret_edit")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

bool TextEdit::multicaret_edit_ignore_caret(int32_t p_caret_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("multicaret_edit_ignore_caret")._native_ptr(), 1116898809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_caret_index_encoded;
	PtrToArg<int64_t>::encode(p_caret_index, &p_caret_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_caret_index_encoded);
}

bool TextEdit::is_caret_visible(int32_t p_caret_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("is_caret_visible")._native_ptr(), 1051549951);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_caret_index_encoded;
	PtrToArg<int64_t>::encode(p_caret_index, &p_caret_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_caret_index_encoded);
}

Vector2 TextEdit::get_caret_draw_pos(int32_t p_caret_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_caret_draw_pos")._native_ptr(), 478253731);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	int64_t p_caret_index_encoded;
	PtrToArg<int64_t>::encode(p_caret_index, &p_caret_index_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner, &p_caret_index_encoded);
}

void TextEdit::set_caret_line(int32_t p_line, bool p_adjust_viewport, bool p_can_be_hidden, int32_t p_wrap_index, int32_t p_caret_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_caret_line")._native_ptr(), 1302582944);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_line_encoded;
	PtrToArg<int64_t>::encode(p_line, &p_line_encoded);
	int8_t p_adjust_viewport_encoded;
	PtrToArg<bool>::encode(p_adjust_viewport, &p_adjust_viewport_encoded);
	int8_t p_can_be_hidden_encoded;
	PtrToArg<bool>::encode(p_can_be_hidden, &p_can_be_hidden_encoded);
	int64_t p_wrap_index_encoded;
	PtrToArg<int64_t>::encode(p_wrap_index, &p_wrap_index_encoded);
	int64_t p_caret_index_encoded;
	PtrToArg<int64_t>::encode(p_caret_index, &p_caret_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_line_encoded, &p_adjust_viewport_encoded, &p_can_be_hidden_encoded, &p_wrap_index_encoded, &p_caret_index_encoded);
}

int32_t TextEdit::get_caret_line(int32_t p_caret_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_caret_line")._native_ptr(), 1591665591);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_caret_index_encoded;
	PtrToArg<int64_t>::encode(p_caret_index, &p_caret_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_caret_index_encoded);
}

void TextEdit::set_caret_column(int32_t p_column, bool p_adjust_viewport, int32_t p_caret_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_caret_column")._native_ptr(), 3796796178);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	int8_t p_adjust_viewport_encoded;
	PtrToArg<bool>::encode(p_adjust_viewport, &p_adjust_viewport_encoded);
	int64_t p_caret_index_encoded;
	PtrToArg<int64_t>::encode(p_caret_index, &p_caret_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_column_encoded, &p_adjust_viewport_encoded, &p_caret_index_encoded);
}

int32_t TextEdit::get_caret_column(int32_t p_caret_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_caret_column")._native_ptr(), 1591665591);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_caret_index_encoded;
	PtrToArg<int64_t>::encode(p_caret_index, &p_caret_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_caret_index_encoded);
}

int32_t TextEdit::get_next_composite_character_column(int32_t p_line, int32_t p_column) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_next_composite_character_column")._native_ptr(), 3175239445);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_line_encoded;
	PtrToArg<int64_t>::encode(p_line, &p_line_encoded);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_line_encoded, &p_column_encoded);
}

int32_t TextEdit::get_previous_composite_character_column(int32_t p_line, int32_t p_column) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_previous_composite_character_column")._native_ptr(), 3175239445);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_line_encoded;
	PtrToArg<int64_t>::encode(p_line, &p_line_encoded);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_line_encoded, &p_column_encoded);
}

int32_t TextEdit::get_caret_wrap_index(int32_t p_caret_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_caret_wrap_index")._native_ptr(), 1591665591);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_caret_index_encoded;
	PtrToArg<int64_t>::encode(p_caret_index, &p_caret_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_caret_index_encoded);
}

String TextEdit::get_word_under_caret(int32_t p_caret_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_word_under_caret")._native_ptr(), 3929349208);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_caret_index_encoded;
	PtrToArg<int64_t>::encode(p_caret_index, &p_caret_index_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_caret_index_encoded);
}

void TextEdit::set_use_default_word_separators(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_use_default_word_separators")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool TextEdit::is_default_word_separators_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("is_default_word_separators_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void TextEdit::set_use_custom_word_separators(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_use_custom_word_separators")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool TextEdit::is_custom_word_separators_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("is_custom_word_separators_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void TextEdit::set_custom_word_separators(const String &p_custom_word_separators) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_custom_word_separators")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_custom_word_separators);
}

String TextEdit::get_custom_word_separators() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_custom_word_separators")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

void TextEdit::set_selecting_enabled(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_selecting_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool TextEdit::is_selecting_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("is_selecting_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void TextEdit::set_deselect_on_focus_loss_enabled(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_deselect_on_focus_loss_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool TextEdit::is_deselect_on_focus_loss_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("is_deselect_on_focus_loss_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void TextEdit::set_drag_and_drop_selection_enabled(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_drag_and_drop_selection_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool TextEdit::is_drag_and_drop_selection_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("is_drag_and_drop_selection_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void TextEdit::set_selection_mode(TextEdit::SelectionMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_selection_mode")._native_ptr(), 1658801786);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

TextEdit::SelectionMode TextEdit::get_selection_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_selection_mode")._native_ptr(), 3750106938);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TextEdit::SelectionMode(0)));
	return (TextEdit::SelectionMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void TextEdit::select_all() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("select_all")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void TextEdit::select_word_under_caret(int32_t p_caret_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("select_word_under_caret")._native_ptr(), 1025054187);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_caret_index_encoded;
	PtrToArg<int64_t>::encode(p_caret_index, &p_caret_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_caret_index_encoded);
}

void TextEdit::add_selection_for_next_occurrence() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("add_selection_for_next_occurrence")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void TextEdit::skip_selection_for_next_occurrence() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("skip_selection_for_next_occurrence")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void TextEdit::select(int32_t p_origin_line, int32_t p_origin_column, int32_t p_caret_line, int32_t p_caret_column, int32_t p_caret_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("select")._native_ptr(), 2560984452);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_origin_line_encoded;
	PtrToArg<int64_t>::encode(p_origin_line, &p_origin_line_encoded);
	int64_t p_origin_column_encoded;
	PtrToArg<int64_t>::encode(p_origin_column, &p_origin_column_encoded);
	int64_t p_caret_line_encoded;
	PtrToArg<int64_t>::encode(p_caret_line, &p_caret_line_encoded);
	int64_t p_caret_column_encoded;
	PtrToArg<int64_t>::encode(p_caret_column, &p_caret_column_encoded);
	int64_t p_caret_index_encoded;
	PtrToArg<int64_t>::encode(p_caret_index, &p_caret_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_origin_line_encoded, &p_origin_column_encoded, &p_caret_line_encoded, &p_caret_column_encoded, &p_caret_index_encoded);
}

bool TextEdit::has_selection(int32_t p_caret_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("has_selection")._native_ptr(), 2824505868);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_caret_index_encoded;
	PtrToArg<int64_t>::encode(p_caret_index, &p_caret_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_caret_index_encoded);
}

String TextEdit::get_selected_text(int32_t p_caret_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_selected_text")._native_ptr(), 2309358862);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_caret_index_encoded;
	PtrToArg<int64_t>::encode(p_caret_index, &p_caret_index_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_caret_index_encoded);
}

int32_t TextEdit::get_selection_at_line_column(int32_t p_line, int32_t p_column, bool p_include_edges, bool p_only_selections) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_selection_at_line_column")._native_ptr(), 1810224333);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_line_encoded;
	PtrToArg<int64_t>::encode(p_line, &p_line_encoded);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	int8_t p_include_edges_encoded;
	PtrToArg<bool>::encode(p_include_edges, &p_include_edges_encoded);
	int8_t p_only_selections_encoded;
	PtrToArg<bool>::encode(p_only_selections, &p_only_selections_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_line_encoded, &p_column_encoded, &p_include_edges_encoded, &p_only_selections_encoded);
}

TypedArray<Vector2i> TextEdit::get_line_ranges_from_carets(bool p_only_selections, bool p_merge_adjacent) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_line_ranges_from_carets")._native_ptr(), 2393089247);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<Vector2i>()));
	int8_t p_only_selections_encoded;
	PtrToArg<bool>::encode(p_only_selections, &p_only_selections_encoded);
	int8_t p_merge_adjacent_encoded;
	PtrToArg<bool>::encode(p_merge_adjacent, &p_merge_adjacent_encoded);
	return ::godot::internal::_call_native_mb_ret<TypedArray<Vector2i>>(_gde_method_bind, _owner, &p_only_selections_encoded, &p_merge_adjacent_encoded);
}

int32_t TextEdit::get_selection_origin_line(int32_t p_caret_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_selection_origin_line")._native_ptr(), 1591665591);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_caret_index_encoded;
	PtrToArg<int64_t>::encode(p_caret_index, &p_caret_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_caret_index_encoded);
}

int32_t TextEdit::get_selection_origin_column(int32_t p_caret_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_selection_origin_column")._native_ptr(), 1591665591);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_caret_index_encoded;
	PtrToArg<int64_t>::encode(p_caret_index, &p_caret_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_caret_index_encoded);
}

void TextEdit::set_selection_origin_line(int32_t p_line, bool p_can_be_hidden, int32_t p_wrap_index, int32_t p_caret_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_selection_origin_line")._native_ptr(), 195434140);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_line_encoded;
	PtrToArg<int64_t>::encode(p_line, &p_line_encoded);
	int8_t p_can_be_hidden_encoded;
	PtrToArg<bool>::encode(p_can_be_hidden, &p_can_be_hidden_encoded);
	int64_t p_wrap_index_encoded;
	PtrToArg<int64_t>::encode(p_wrap_index, &p_wrap_index_encoded);
	int64_t p_caret_index_encoded;
	PtrToArg<int64_t>::encode(p_caret_index, &p_caret_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_line_encoded, &p_can_be_hidden_encoded, &p_wrap_index_encoded, &p_caret_index_encoded);
}

void TextEdit::set_selection_origin_column(int32_t p_column, int32_t p_caret_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_selection_origin_column")._native_ptr(), 2230941749);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	int64_t p_caret_index_encoded;
	PtrToArg<int64_t>::encode(p_caret_index, &p_caret_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_column_encoded, &p_caret_index_encoded);
}

int32_t TextEdit::get_selection_from_line(int32_t p_caret_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_selection_from_line")._native_ptr(), 1591665591);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_caret_index_encoded;
	PtrToArg<int64_t>::encode(p_caret_index, &p_caret_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_caret_index_encoded);
}

int32_t TextEdit::get_selection_from_column(int32_t p_caret_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_selection_from_column")._native_ptr(), 1591665591);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_caret_index_encoded;
	PtrToArg<int64_t>::encode(p_caret_index, &p_caret_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_caret_index_encoded);
}

int32_t TextEdit::get_selection_to_line(int32_t p_caret_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_selection_to_line")._native_ptr(), 1591665591);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_caret_index_encoded;
	PtrToArg<int64_t>::encode(p_caret_index, &p_caret_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_caret_index_encoded);
}

int32_t TextEdit::get_selection_to_column(int32_t p_caret_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_selection_to_column")._native_ptr(), 1591665591);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_caret_index_encoded;
	PtrToArg<int64_t>::encode(p_caret_index, &p_caret_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_caret_index_encoded);
}

bool TextEdit::is_caret_after_selection_origin(int32_t p_caret_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("is_caret_after_selection_origin")._native_ptr(), 1051549951);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_caret_index_encoded;
	PtrToArg<int64_t>::encode(p_caret_index, &p_caret_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_caret_index_encoded);
}

void TextEdit::deselect(int32_t p_caret_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("deselect")._native_ptr(), 1025054187);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_caret_index_encoded;
	PtrToArg<int64_t>::encode(p_caret_index, &p_caret_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_caret_index_encoded);
}

void TextEdit::delete_selection(int32_t p_caret_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("delete_selection")._native_ptr(), 1025054187);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_caret_index_encoded;
	PtrToArg<int64_t>::encode(p_caret_index, &p_caret_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_caret_index_encoded);
}

void TextEdit::set_line_wrapping_mode(TextEdit::LineWrappingMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_line_wrapping_mode")._native_ptr(), 2525115309);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

TextEdit::LineWrappingMode TextEdit::get_line_wrapping_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_line_wrapping_mode")._native_ptr(), 3562716114);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TextEdit::LineWrappingMode(0)));
	return (TextEdit::LineWrappingMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void TextEdit::set_autowrap_mode(TextServer::AutowrapMode p_autowrap_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_autowrap_mode")._native_ptr(), 3289138044);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_autowrap_mode_encoded;
	PtrToArg<int64_t>::encode(p_autowrap_mode, &p_autowrap_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_autowrap_mode_encoded);
}

TextServer::AutowrapMode TextEdit::get_autowrap_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_autowrap_mode")._native_ptr(), 1549071663);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TextServer::AutowrapMode(0)));
	return (TextServer::AutowrapMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

bool TextEdit::is_line_wrapped(int32_t p_line) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("is_line_wrapped")._native_ptr(), 1116898809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_line_encoded;
	PtrToArg<int64_t>::encode(p_line, &p_line_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_line_encoded);
}

int32_t TextEdit::get_line_wrap_count(int32_t p_line) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_line_wrap_count")._native_ptr(), 923996154);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_line_encoded;
	PtrToArg<int64_t>::encode(p_line, &p_line_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_line_encoded);
}

int32_t TextEdit::get_line_wrap_index_at_column(int32_t p_line, int32_t p_column) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_line_wrap_index_at_column")._native_ptr(), 3175239445);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_line_encoded;
	PtrToArg<int64_t>::encode(p_line, &p_line_encoded);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_line_encoded, &p_column_encoded);
}

PackedStringArray TextEdit::get_line_wrapped_text(int32_t p_line) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_line_wrapped_text")._native_ptr(), 647634434);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedStringArray()));
	int64_t p_line_encoded;
	PtrToArg<int64_t>::encode(p_line, &p_line_encoded);
	return ::godot::internal::_call_native_mb_ret<PackedStringArray>(_gde_method_bind, _owner, &p_line_encoded);
}

void TextEdit::set_smooth_scroll_enabled(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_smooth_scroll_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool TextEdit::is_smooth_scroll_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("is_smooth_scroll_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

VScrollBar *TextEdit::get_v_scroll_bar() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_v_scroll_bar")._native_ptr(), 3226026593);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<VScrollBar>(_gde_method_bind, _owner);
}

HScrollBar *TextEdit::get_h_scroll_bar() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_h_scroll_bar")._native_ptr(), 3774687988);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<HScrollBar>(_gde_method_bind, _owner);
}

void TextEdit::set_v_scroll(double p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_v_scroll")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_value_encoded;
	PtrToArg<double>::encode(p_value, &p_value_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_value_encoded);
}

double TextEdit::get_v_scroll() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_v_scroll")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void TextEdit::set_h_scroll(int32_t p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_h_scroll")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_value_encoded;
	PtrToArg<int64_t>::encode(p_value, &p_value_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_value_encoded);
}

int32_t TextEdit::get_h_scroll() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_h_scroll")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void TextEdit::set_scroll_past_end_of_file_enabled(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_scroll_past_end_of_file_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool TextEdit::is_scroll_past_end_of_file_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("is_scroll_past_end_of_file_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void TextEdit::set_v_scroll_speed(float p_speed) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_v_scroll_speed")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_speed_encoded;
	PtrToArg<double>::encode(p_speed, &p_speed_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_speed_encoded);
}

float TextEdit::get_v_scroll_speed() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_v_scroll_speed")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void TextEdit::set_fit_content_height_enabled(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_fit_content_height_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool TextEdit::is_fit_content_height_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("is_fit_content_height_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void TextEdit::set_fit_content_width_enabled(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_fit_content_width_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool TextEdit::is_fit_content_width_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("is_fit_content_width_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

double TextEdit::get_scroll_pos_for_line(int32_t p_line, int32_t p_wrap_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_scroll_pos_for_line")._native_ptr(), 3929084198);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_line_encoded;
	PtrToArg<int64_t>::encode(p_line, &p_line_encoded);
	int64_t p_wrap_index_encoded;
	PtrToArg<int64_t>::encode(p_wrap_index, &p_wrap_index_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_line_encoded, &p_wrap_index_encoded);
}

void TextEdit::set_line_as_first_visible(int32_t p_line, int32_t p_wrap_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_line_as_first_visible")._native_ptr(), 2230941749);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_line_encoded;
	PtrToArg<int64_t>::encode(p_line, &p_line_encoded);
	int64_t p_wrap_index_encoded;
	PtrToArg<int64_t>::encode(p_wrap_index, &p_wrap_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_line_encoded, &p_wrap_index_encoded);
}

int32_t TextEdit::get_first_visible_line() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_first_visible_line")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void TextEdit::set_line_as_center_visible(int32_t p_line, int32_t p_wrap_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_line_as_center_visible")._native_ptr(), 2230941749);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_line_encoded;
	PtrToArg<int64_t>::encode(p_line, &p_line_encoded);
	int64_t p_wrap_index_encoded;
	PtrToArg<int64_t>::encode(p_wrap_index, &p_wrap_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_line_encoded, &p_wrap_index_encoded);
}

void TextEdit::set_line_as_last_visible(int32_t p_line, int32_t p_wrap_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_line_as_last_visible")._native_ptr(), 2230941749);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_line_encoded;
	PtrToArg<int64_t>::encode(p_line, &p_line_encoded);
	int64_t p_wrap_index_encoded;
	PtrToArg<int64_t>::encode(p_wrap_index, &p_wrap_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_line_encoded, &p_wrap_index_encoded);
}

int32_t TextEdit::get_last_full_visible_line() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_last_full_visible_line")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

int32_t TextEdit::get_last_full_visible_line_wrap_index() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_last_full_visible_line_wrap_index")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

int32_t TextEdit::get_visible_line_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_visible_line_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

int32_t TextEdit::get_visible_line_count_in_range(int32_t p_from_line, int32_t p_to_line) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_visible_line_count_in_range")._native_ptr(), 3175239445);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_from_line_encoded;
	PtrToArg<int64_t>::encode(p_from_line, &p_from_line_encoded);
	int64_t p_to_line_encoded;
	PtrToArg<int64_t>::encode(p_to_line, &p_to_line_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_from_line_encoded, &p_to_line_encoded);
}

int32_t TextEdit::get_total_visible_line_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_total_visible_line_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void TextEdit::adjust_viewport_to_caret(int32_t p_caret_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("adjust_viewport_to_caret")._native_ptr(), 1995695955);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_caret_index_encoded;
	PtrToArg<int64_t>::encode(p_caret_index, &p_caret_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_caret_index_encoded);
}

void TextEdit::center_viewport_to_caret(int32_t p_caret_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("center_viewport_to_caret")._native_ptr(), 1995695955);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_caret_index_encoded;
	PtrToArg<int64_t>::encode(p_caret_index, &p_caret_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_caret_index_encoded);
}

void TextEdit::set_draw_minimap(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_draw_minimap")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool TextEdit::is_drawing_minimap() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("is_drawing_minimap")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void TextEdit::set_minimap_width(int32_t p_width) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_minimap_width")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_width_encoded;
	PtrToArg<int64_t>::encode(p_width, &p_width_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_width_encoded);
}

int32_t TextEdit::get_minimap_width() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_minimap_width")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

int32_t TextEdit::get_minimap_visible_lines() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_minimap_visible_lines")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void TextEdit::add_gutter(int32_t p_at) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("add_gutter")._native_ptr(), 1025054187);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_at_encoded;
	PtrToArg<int64_t>::encode(p_at, &p_at_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_at_encoded);
}

void TextEdit::remove_gutter(int32_t p_gutter) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("remove_gutter")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_gutter_encoded;
	PtrToArg<int64_t>::encode(p_gutter, &p_gutter_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_gutter_encoded);
}

int32_t TextEdit::get_gutter_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_gutter_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void TextEdit::set_gutter_name(int32_t p_gutter, const String &p_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_gutter_name")._native_ptr(), 501894301);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_gutter_encoded;
	PtrToArg<int64_t>::encode(p_gutter, &p_gutter_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_gutter_encoded, &p_name);
}

String TextEdit::get_gutter_name(int32_t p_gutter) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_gutter_name")._native_ptr(), 844755477);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_gutter_encoded;
	PtrToArg<int64_t>::encode(p_gutter, &p_gutter_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_gutter_encoded);
}

void TextEdit::set_gutter_type(int32_t p_gutter, TextEdit::GutterType p_type) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_gutter_type")._native_ptr(), 1088959071);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_gutter_encoded;
	PtrToArg<int64_t>::encode(p_gutter, &p_gutter_encoded);
	int64_t p_type_encoded;
	PtrToArg<int64_t>::encode(p_type, &p_type_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_gutter_encoded, &p_type_encoded);
}

TextEdit::GutterType TextEdit::get_gutter_type(int32_t p_gutter) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_gutter_type")._native_ptr(), 1159699127);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TextEdit::GutterType(0)));
	int64_t p_gutter_encoded;
	PtrToArg<int64_t>::encode(p_gutter, &p_gutter_encoded);
	return (TextEdit::GutterType)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_gutter_encoded);
}

void TextEdit::set_gutter_width(int32_t p_gutter, int32_t p_width) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_gutter_width")._native_ptr(), 3937882851);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_gutter_encoded;
	PtrToArg<int64_t>::encode(p_gutter, &p_gutter_encoded);
	int64_t p_width_encoded;
	PtrToArg<int64_t>::encode(p_width, &p_width_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_gutter_encoded, &p_width_encoded);
}

int32_t TextEdit::get_gutter_width(int32_t p_gutter) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_gutter_width")._native_ptr(), 923996154);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_gutter_encoded;
	PtrToArg<int64_t>::encode(p_gutter, &p_gutter_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_gutter_encoded);
}

void TextEdit::set_gutter_draw(int32_t p_gutter, bool p_draw) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_gutter_draw")._native_ptr(), 300928843);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_gutter_encoded;
	PtrToArg<int64_t>::encode(p_gutter, &p_gutter_encoded);
	int8_t p_draw_encoded;
	PtrToArg<bool>::encode(p_draw, &p_draw_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_gutter_encoded, &p_draw_encoded);
}

bool TextEdit::is_gutter_drawn(int32_t p_gutter) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("is_gutter_drawn")._native_ptr(), 1116898809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_gutter_encoded;
	PtrToArg<int64_t>::encode(p_gutter, &p_gutter_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_gutter_encoded);
}

void TextEdit::set_gutter_clickable(int32_t p_gutter, bool p_clickable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_gutter_clickable")._native_ptr(), 300928843);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_gutter_encoded;
	PtrToArg<int64_t>::encode(p_gutter, &p_gutter_encoded);
	int8_t p_clickable_encoded;
	PtrToArg<bool>::encode(p_clickable, &p_clickable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_gutter_encoded, &p_clickable_encoded);
}

bool TextEdit::is_gutter_clickable(int32_t p_gutter) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("is_gutter_clickable")._native_ptr(), 1116898809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_gutter_encoded;
	PtrToArg<int64_t>::encode(p_gutter, &p_gutter_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_gutter_encoded);
}

void TextEdit::set_gutter_overwritable(int32_t p_gutter, bool p_overwritable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_gutter_overwritable")._native_ptr(), 300928843);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_gutter_encoded;
	PtrToArg<int64_t>::encode(p_gutter, &p_gutter_encoded);
	int8_t p_overwritable_encoded;
	PtrToArg<bool>::encode(p_overwritable, &p_overwritable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_gutter_encoded, &p_overwritable_encoded);
}

bool TextEdit::is_gutter_overwritable(int32_t p_gutter) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("is_gutter_overwritable")._native_ptr(), 1116898809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_gutter_encoded;
	PtrToArg<int64_t>::encode(p_gutter, &p_gutter_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_gutter_encoded);
}

void TextEdit::merge_gutters(int32_t p_from_line, int32_t p_to_line) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("merge_gutters")._native_ptr(), 3937882851);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_from_line_encoded;
	PtrToArg<int64_t>::encode(p_from_line, &p_from_line_encoded);
	int64_t p_to_line_encoded;
	PtrToArg<int64_t>::encode(p_to_line, &p_to_line_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_from_line_encoded, &p_to_line_encoded);
}

void TextEdit::set_gutter_custom_draw(int32_t p_column, const Callable &p_draw_callback) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_gutter_custom_draw")._native_ptr(), 957362965);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_column_encoded, &p_draw_callback);
}

int32_t TextEdit::get_total_gutter_width() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_total_gutter_width")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void TextEdit::set_line_gutter_metadata(int32_t p_line, int32_t p_gutter, const Variant &p_metadata) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_line_gutter_metadata")._native_ptr(), 2060538656);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_line_encoded;
	PtrToArg<int64_t>::encode(p_line, &p_line_encoded);
	int64_t p_gutter_encoded;
	PtrToArg<int64_t>::encode(p_gutter, &p_gutter_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_line_encoded, &p_gutter_encoded, &p_metadata);
}

Variant TextEdit::get_line_gutter_metadata(int32_t p_line, int32_t p_gutter) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_line_gutter_metadata")._native_ptr(), 678354945);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Variant()));
	int64_t p_line_encoded;
	PtrToArg<int64_t>::encode(p_line, &p_line_encoded);
	int64_t p_gutter_encoded;
	PtrToArg<int64_t>::encode(p_gutter, &p_gutter_encoded);
	return ::godot::internal::_call_native_mb_ret<Variant>(_gde_method_bind, _owner, &p_line_encoded, &p_gutter_encoded);
}

void TextEdit::set_line_gutter_text(int32_t p_line, int32_t p_gutter, const String &p_text) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_line_gutter_text")._native_ptr(), 2285447957);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_line_encoded;
	PtrToArg<int64_t>::encode(p_line, &p_line_encoded);
	int64_t p_gutter_encoded;
	PtrToArg<int64_t>::encode(p_gutter, &p_gutter_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_line_encoded, &p_gutter_encoded, &p_text);
}

String TextEdit::get_line_gutter_text(int32_t p_line, int32_t p_gutter) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_line_gutter_text")._native_ptr(), 1391810591);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_line_encoded;
	PtrToArg<int64_t>::encode(p_line, &p_line_encoded);
	int64_t p_gutter_encoded;
	PtrToArg<int64_t>::encode(p_gutter, &p_gutter_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_line_encoded, &p_gutter_encoded);
}

void TextEdit::set_line_gutter_icon(int32_t p_line, int32_t p_gutter, const Ref<Texture2D> &p_icon) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_line_gutter_icon")._native_ptr(), 176101966);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_line_encoded;
	PtrToArg<int64_t>::encode(p_line, &p_line_encoded);
	int64_t p_gutter_encoded;
	PtrToArg<int64_t>::encode(p_gutter, &p_gutter_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_line_encoded, &p_gutter_encoded, (p_icon != nullptr ? &p_icon->_owner : nullptr));
}

Ref<Texture2D> TextEdit::get_line_gutter_icon(int32_t p_line, int32_t p_gutter) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_line_gutter_icon")._native_ptr(), 2584904275);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Texture2D>()));
	int64_t p_line_encoded;
	PtrToArg<int64_t>::encode(p_line, &p_line_encoded);
	int64_t p_gutter_encoded;
	PtrToArg<int64_t>::encode(p_gutter, &p_gutter_encoded);
	return Ref<Texture2D>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Texture2D>(_gde_method_bind, _owner, &p_line_encoded, &p_gutter_encoded));
}

void TextEdit::set_line_gutter_item_color(int32_t p_line, int32_t p_gutter, const Color &p_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_line_gutter_item_color")._native_ptr(), 3733378741);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_line_encoded;
	PtrToArg<int64_t>::encode(p_line, &p_line_encoded);
	int64_t p_gutter_encoded;
	PtrToArg<int64_t>::encode(p_gutter, &p_gutter_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_line_encoded, &p_gutter_encoded, &p_color);
}

Color TextEdit::get_line_gutter_item_color(int32_t p_line, int32_t p_gutter) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_line_gutter_item_color")._native_ptr(), 2165839948);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Color()));
	int64_t p_line_encoded;
	PtrToArg<int64_t>::encode(p_line, &p_line_encoded);
	int64_t p_gutter_encoded;
	PtrToArg<int64_t>::encode(p_gutter, &p_gutter_encoded);
	return ::godot::internal::_call_native_mb_ret<Color>(_gde_method_bind, _owner, &p_line_encoded, &p_gutter_encoded);
}

void TextEdit::set_line_gutter_clickable(int32_t p_line, int32_t p_gutter, bool p_clickable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_line_gutter_clickable")._native_ptr(), 1383440665);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_line_encoded;
	PtrToArg<int64_t>::encode(p_line, &p_line_encoded);
	int64_t p_gutter_encoded;
	PtrToArg<int64_t>::encode(p_gutter, &p_gutter_encoded);
	int8_t p_clickable_encoded;
	PtrToArg<bool>::encode(p_clickable, &p_clickable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_line_encoded, &p_gutter_encoded, &p_clickable_encoded);
}

bool TextEdit::is_line_gutter_clickable(int32_t p_line, int32_t p_gutter) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("is_line_gutter_clickable")._native_ptr(), 2522259332);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_line_encoded;
	PtrToArg<int64_t>::encode(p_line, &p_line_encoded);
	int64_t p_gutter_encoded;
	PtrToArg<int64_t>::encode(p_gutter, &p_gutter_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_line_encoded, &p_gutter_encoded);
}

void TextEdit::set_line_background_color(int32_t p_line, const Color &p_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_line_background_color")._native_ptr(), 2878471219);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_line_encoded;
	PtrToArg<int64_t>::encode(p_line, &p_line_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_line_encoded, &p_color);
}

Color TextEdit::get_line_background_color(int32_t p_line) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_line_background_color")._native_ptr(), 3457211756);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Color()));
	int64_t p_line_encoded;
	PtrToArg<int64_t>::encode(p_line, &p_line_encoded);
	return ::godot::internal::_call_native_mb_ret<Color>(_gde_method_bind, _owner, &p_line_encoded);
}

void TextEdit::set_syntax_highlighter(const Ref<SyntaxHighlighter> &p_syntax_highlighter) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_syntax_highlighter")._native_ptr(), 2765644541);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_syntax_highlighter != nullptr ? &p_syntax_highlighter->_owner : nullptr));
}

Ref<SyntaxHighlighter> TextEdit::get_syntax_highlighter() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_syntax_highlighter")._native_ptr(), 2721131626);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<SyntaxHighlighter>()));
	return Ref<SyntaxHighlighter>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<SyntaxHighlighter>(_gde_method_bind, _owner));
}

void TextEdit::set_highlight_current_line(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_highlight_current_line")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool TextEdit::is_highlight_current_line_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("is_highlight_current_line_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void TextEdit::set_highlight_all_occurrences(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_highlight_all_occurrences")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool TextEdit::is_highlight_all_occurrences_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("is_highlight_all_occurrences_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

bool TextEdit::get_draw_control_chars() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_draw_control_chars")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void TextEdit::set_draw_control_chars(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_draw_control_chars")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

void TextEdit::set_draw_tabs(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_draw_tabs")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool TextEdit::is_drawing_tabs() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("is_drawing_tabs")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void TextEdit::set_draw_spaces(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("set_draw_spaces")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool TextEdit::is_drawing_spaces() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("is_drawing_spaces")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

PopupMenu *TextEdit::get_menu() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_menu")._native_ptr(), 229722558);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<PopupMenu>(_gde_method_bind, _owner);
}

bool TextEdit::is_menu_visible() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("is_menu_visible")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void TextEdit::menu_option(int32_t p_option) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("menu_option")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_option_encoded;
	PtrToArg<int64_t>::encode(p_option, &p_option_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_option_encoded);
}

void TextEdit::adjust_carets_after_edit(int32_t p_caret, int32_t p_from_line, int32_t p_from_col, int32_t p_to_line, int32_t p_to_col) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("adjust_carets_after_edit")._native_ptr(), 1770277138);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_caret_encoded;
	PtrToArg<int64_t>::encode(p_caret, &p_caret_encoded);
	int64_t p_from_line_encoded;
	PtrToArg<int64_t>::encode(p_from_line, &p_from_line_encoded);
	int64_t p_from_col_encoded;
	PtrToArg<int64_t>::encode(p_from_col, &p_from_col_encoded);
	int64_t p_to_line_encoded;
	PtrToArg<int64_t>::encode(p_to_line, &p_to_line_encoded);
	int64_t p_to_col_encoded;
	PtrToArg<int64_t>::encode(p_to_col, &p_to_col_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_caret_encoded, &p_from_line_encoded, &p_from_col_encoded, &p_to_line_encoded, &p_to_col_encoded);
}

PackedInt32Array TextEdit::get_caret_index_edit_order() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_caret_index_edit_order")._native_ptr(), 969006518);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedInt32Array()));
	return ::godot::internal::_call_native_mb_ret<PackedInt32Array>(_gde_method_bind, _owner);
}

int32_t TextEdit::get_selection_line(int32_t p_caret_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_selection_line")._native_ptr(), 1591665591);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_caret_index_encoded;
	PtrToArg<int64_t>::encode(p_caret_index, &p_caret_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_caret_index_encoded);
}

int32_t TextEdit::get_selection_column(int32_t p_caret_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TextEdit::get_class_static()._native_ptr(), StringName("get_selection_column")._native_ptr(), 1591665591);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_caret_index_encoded;
	PtrToArg<int64_t>::encode(p_caret_index, &p_caret_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_caret_index_encoded);
}

void TextEdit::_handle_unicode_input(int32_t p_unicode_char, int32_t p_caret_index) {}

void TextEdit::_backspace(int32_t p_caret_index) {}

void TextEdit::_cut(int32_t p_caret_index) {}

void TextEdit::_copy(int32_t p_caret_index) {}

void TextEdit::_paste(int32_t p_caret_index) {}

void TextEdit::_paste_primary_clipboard(int32_t p_caret_index) {}

} // namespace godot
