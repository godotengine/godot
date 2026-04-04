/**************************************************************************/
/*  tree.cpp                                                              */
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

#include <godot_cpp/classes/tree.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

void Tree::clear() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("clear")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

TreeItem *Tree::create_item(TreeItem *p_parent, int32_t p_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("create_item")._native_ptr(), 528467046);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret_obj<TreeItem>(_gde_method_bind, _owner, (p_parent != nullptr ? &p_parent->_owner : nullptr), &p_index_encoded);
}

TreeItem *Tree::get_root() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("get_root")._native_ptr(), 1514277247);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<TreeItem>(_gde_method_bind, _owner);
}

void Tree::set_column_custom_minimum_width(int32_t p_column, int32_t p_min_width) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("set_column_custom_minimum_width")._native_ptr(), 3937882851);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	int64_t p_min_width_encoded;
	PtrToArg<int64_t>::encode(p_min_width, &p_min_width_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_column_encoded, &p_min_width_encoded);
}

void Tree::set_column_expand(int32_t p_column, bool p_expand) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("set_column_expand")._native_ptr(), 300928843);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	int8_t p_expand_encoded;
	PtrToArg<bool>::encode(p_expand, &p_expand_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_column_encoded, &p_expand_encoded);
}

void Tree::set_column_expand_ratio(int32_t p_column, int32_t p_ratio) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("set_column_expand_ratio")._native_ptr(), 3937882851);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	int64_t p_ratio_encoded;
	PtrToArg<int64_t>::encode(p_ratio, &p_ratio_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_column_encoded, &p_ratio_encoded);
}

void Tree::set_column_clip_content(int32_t p_column, bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("set_column_clip_content")._native_ptr(), 300928843);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_column_encoded, &p_enable_encoded);
}

bool Tree::is_column_expanding(int32_t p_column) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("is_column_expanding")._native_ptr(), 1116898809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_column_encoded);
}

bool Tree::is_column_clipping_content(int32_t p_column) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("is_column_clipping_content")._native_ptr(), 1116898809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_column_encoded);
}

int32_t Tree::get_column_expand_ratio(int32_t p_column) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("get_column_expand_ratio")._native_ptr(), 923996154);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_column_encoded);
}

int32_t Tree::get_column_width(int32_t p_column) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("get_column_width")._native_ptr(), 923996154);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_column_encoded);
}

void Tree::set_hide_root(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("set_hide_root")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool Tree::is_root_hidden() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("is_root_hidden")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

TreeItem *Tree::get_next_selected(TreeItem *p_from) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("get_next_selected")._native_ptr(), 873446299);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<TreeItem>(_gde_method_bind, _owner, (p_from != nullptr ? &p_from->_owner : nullptr));
}

TreeItem *Tree::get_selected() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("get_selected")._native_ptr(), 1514277247);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<TreeItem>(_gde_method_bind, _owner);
}

void Tree::set_selected(TreeItem *p_item, int32_t p_column) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("set_selected")._native_ptr(), 2662547442);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_item != nullptr ? &p_item->_owner : nullptr), &p_column_encoded);
}

int32_t Tree::get_selected_column() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("get_selected_column")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

int32_t Tree::get_pressed_button() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("get_pressed_button")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Tree::set_select_mode(Tree::SelectMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("set_select_mode")._native_ptr(), 3223887270);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

Tree::SelectMode Tree::get_select_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("get_select_mode")._native_ptr(), 100748571);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Tree::SelectMode(0)));
	return (Tree::SelectMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Tree::deselect_all() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("deselect_all")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void Tree::set_columns(int32_t p_amount) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("set_columns")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_amount_encoded;
	PtrToArg<int64_t>::encode(p_amount, &p_amount_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_amount_encoded);
}

int32_t Tree::get_columns() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("get_columns")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

TreeItem *Tree::get_edited() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("get_edited")._native_ptr(), 1514277247);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<TreeItem>(_gde_method_bind, _owner);
}

int32_t Tree::get_edited_column() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("get_edited_column")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

bool Tree::edit_selected(bool p_force_edit) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("edit_selected")._native_ptr(), 2595650253);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int8_t p_force_edit_encoded;
	PtrToArg<bool>::encode(p_force_edit, &p_force_edit_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_force_edit_encoded);
}

Rect2 Tree::get_custom_popup_rect() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("get_custom_popup_rect")._native_ptr(), 1639390495);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Rect2()));
	return ::godot::internal::_call_native_mb_ret<Rect2>(_gde_method_bind, _owner);
}

Rect2 Tree::get_item_area_rect(TreeItem *p_item, int32_t p_column, int32_t p_button_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("get_item_area_rect")._native_ptr(), 47968679);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Rect2()));
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	int64_t p_button_index_encoded;
	PtrToArg<int64_t>::encode(p_button_index, &p_button_index_encoded);
	return ::godot::internal::_call_native_mb_ret<Rect2>(_gde_method_bind, _owner, (p_item != nullptr ? &p_item->_owner : nullptr), &p_column_encoded, &p_button_index_encoded);
}

TreeItem *Tree::get_item_at_position(const Vector2 &p_position) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("get_item_at_position")._native_ptr(), 4193340126);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<TreeItem>(_gde_method_bind, _owner, &p_position);
}

int32_t Tree::get_column_at_position(const Vector2 &p_position) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("get_column_at_position")._native_ptr(), 3820158470);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_position);
}

int32_t Tree::get_drop_section_at_position(const Vector2 &p_position) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("get_drop_section_at_position")._native_ptr(), 3820158470);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_position);
}

int32_t Tree::get_button_id_at_position(const Vector2 &p_position) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("get_button_id_at_position")._native_ptr(), 3820158470);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_position);
}

void Tree::ensure_cursor_is_visible() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("ensure_cursor_is_visible")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void Tree::set_column_titles_visible(bool p_visible) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("set_column_titles_visible")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_visible_encoded;
	PtrToArg<bool>::encode(p_visible, &p_visible_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_visible_encoded);
}

bool Tree::are_column_titles_visible() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("are_column_titles_visible")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Tree::set_column_title(int32_t p_column, const String &p_title) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("set_column_title")._native_ptr(), 501894301);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_column_encoded, &p_title);
}

String Tree::get_column_title(int32_t p_column) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("get_column_title")._native_ptr(), 844755477);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_column_encoded);
}

void Tree::set_column_title_tooltip_text(int32_t p_column, const String &p_tooltip_text) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("set_column_title_tooltip_text")._native_ptr(), 501894301);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_column_encoded, &p_tooltip_text);
}

String Tree::get_column_title_tooltip_text(int32_t p_column) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("get_column_title_tooltip_text")._native_ptr(), 844755477);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_column_encoded);
}

void Tree::set_column_title_alignment(int32_t p_column, HorizontalAlignment p_title_alignment) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("set_column_title_alignment")._native_ptr(), 3276431499);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	int64_t p_title_alignment_encoded;
	PtrToArg<int64_t>::encode(p_title_alignment, &p_title_alignment_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_column_encoded, &p_title_alignment_encoded);
}

HorizontalAlignment Tree::get_column_title_alignment(int32_t p_column) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("get_column_title_alignment")._native_ptr(), 4171562184);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (HorizontalAlignment(0)));
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	return (HorizontalAlignment)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_column_encoded);
}

void Tree::set_column_title_direction(int32_t p_column, Control::TextDirection p_direction) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("set_column_title_direction")._native_ptr(), 1707680378);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	int64_t p_direction_encoded;
	PtrToArg<int64_t>::encode(p_direction, &p_direction_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_column_encoded, &p_direction_encoded);
}

Control::TextDirection Tree::get_column_title_direction(int32_t p_column) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("get_column_title_direction")._native_ptr(), 4235602388);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Control::TextDirection(0)));
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	return (Control::TextDirection)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_column_encoded);
}

void Tree::set_column_title_language(int32_t p_column, const String &p_language) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("set_column_title_language")._native_ptr(), 501894301);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_column_encoded, &p_language);
}

String Tree::get_column_title_language(int32_t p_column) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("get_column_title_language")._native_ptr(), 844755477);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_column_encoded;
	PtrToArg<int64_t>::encode(p_column, &p_column_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_column_encoded);
}

Vector2 Tree::get_scroll() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("get_scroll")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

void Tree::scroll_to_item(TreeItem *p_item, bool p_center_on_item) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("scroll_to_item")._native_ptr(), 1314737213);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_center_on_item_encoded;
	PtrToArg<bool>::encode(p_center_on_item, &p_center_on_item_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_item != nullptr ? &p_item->_owner : nullptr), &p_center_on_item_encoded);
}

void Tree::set_h_scroll_enabled(bool p_h_scroll) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("set_h_scroll_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_h_scroll_encoded;
	PtrToArg<bool>::encode(p_h_scroll, &p_h_scroll_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_h_scroll_encoded);
}

bool Tree::is_h_scroll_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("is_h_scroll_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Tree::set_v_scroll_enabled(bool p_h_scroll) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("set_v_scroll_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_h_scroll_encoded;
	PtrToArg<bool>::encode(p_h_scroll, &p_h_scroll_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_h_scroll_encoded);
}

bool Tree::is_v_scroll_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("is_v_scroll_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Tree::set_scroll_hint_mode(Tree::ScrollHintMode p_scroll_hint_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("set_scroll_hint_mode")._native_ptr(), 415911924);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_scroll_hint_mode_encoded;
	PtrToArg<int64_t>::encode(p_scroll_hint_mode, &p_scroll_hint_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_scroll_hint_mode_encoded);
}

Tree::ScrollHintMode Tree::get_scroll_hint_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("get_scroll_hint_mode")._native_ptr(), 553087187);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Tree::ScrollHintMode(0)));
	return (Tree::ScrollHintMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Tree::set_tile_scroll_hint(bool p_tile_scroll_hint) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("set_tile_scroll_hint")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_tile_scroll_hint_encoded;
	PtrToArg<bool>::encode(p_tile_scroll_hint, &p_tile_scroll_hint_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_tile_scroll_hint_encoded);
}

bool Tree::is_scroll_hint_tiled() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("is_scroll_hint_tiled")._native_ptr(), 2240911060);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Tree::set_hide_folding(bool p_hide) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("set_hide_folding")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_hide_encoded;
	PtrToArg<bool>::encode(p_hide, &p_hide_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_hide_encoded);
}

bool Tree::is_folding_hidden() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("is_folding_hidden")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Tree::set_enable_recursive_folding(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("set_enable_recursive_folding")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool Tree::is_recursive_folding_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("is_recursive_folding_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Tree::set_enable_drag_unfolding(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("set_enable_drag_unfolding")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool Tree::is_drag_unfolding_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("is_drag_unfolding_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Tree::set_drop_mode_flags(int32_t p_flags) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("set_drop_mode_flags")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_flags_encoded;
	PtrToArg<int64_t>::encode(p_flags, &p_flags_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_flags_encoded);
}

int32_t Tree::get_drop_mode_flags() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("get_drop_mode_flags")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Tree::set_allow_rmb_select(bool p_allow) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("set_allow_rmb_select")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_allow_encoded;
	PtrToArg<bool>::encode(p_allow, &p_allow_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_allow_encoded);
}

bool Tree::get_allow_rmb_select() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("get_allow_rmb_select")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Tree::set_allow_reselect(bool p_allow) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("set_allow_reselect")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_allow_encoded;
	PtrToArg<bool>::encode(p_allow, &p_allow_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_allow_encoded);
}

bool Tree::get_allow_reselect() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("get_allow_reselect")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Tree::set_allow_search(bool p_allow) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("set_allow_search")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_allow_encoded;
	PtrToArg<bool>::encode(p_allow, &p_allow_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_allow_encoded);
}

bool Tree::get_allow_search() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("get_allow_search")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Tree::set_auto_tooltip(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("set_auto_tooltip")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool Tree::is_auto_tooltip_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Tree::get_class_static()._native_ptr(), StringName("is_auto_tooltip_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

} // namespace godot
