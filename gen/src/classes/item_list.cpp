/**************************************************************************/
/*  item_list.cpp                                                         */
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

#include <godot_cpp/classes/item_list.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/h_scroll_bar.hpp>
#include <godot_cpp/classes/v_scroll_bar.hpp>
#include <godot_cpp/variant/vector2.hpp>

namespace godot {

int32_t ItemList::add_item(const String &p_text, const Ref<Texture2D> &p_icon, bool p_selectable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("add_item")._native_ptr(), 359861678);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int8_t p_selectable_encoded;
	PtrToArg<bool>::encode(p_selectable, &p_selectable_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_text, (p_icon != nullptr ? &p_icon->_owner : nullptr), &p_selectable_encoded);
}

int32_t ItemList::add_icon_item(const Ref<Texture2D> &p_icon, bool p_selectable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("add_icon_item")._native_ptr(), 4256579627);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int8_t p_selectable_encoded;
	PtrToArg<bool>::encode(p_selectable, &p_selectable_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, (p_icon != nullptr ? &p_icon->_owner : nullptr), &p_selectable_encoded);
}

void ItemList::set_item_text(int32_t p_idx, const String &p_text) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("set_item_text")._native_ptr(), 501894301);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_idx_encoded, &p_text);
}

String ItemList::get_item_text(int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("get_item_text")._native_ptr(), 844755477);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_idx_encoded);
}

void ItemList::set_item_icon(int32_t p_idx, const Ref<Texture2D> &p_icon) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("set_item_icon")._native_ptr(), 666127730);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_idx_encoded, (p_icon != nullptr ? &p_icon->_owner : nullptr));
}

Ref<Texture2D> ItemList::get_item_icon(int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("get_item_icon")._native_ptr(), 3536238170);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Texture2D>()));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return Ref<Texture2D>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Texture2D>(_gde_method_bind, _owner, &p_idx_encoded));
}

void ItemList::set_item_text_direction(int32_t p_idx, Control::TextDirection p_direction) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("set_item_text_direction")._native_ptr(), 1707680378);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	int64_t p_direction_encoded;
	PtrToArg<int64_t>::encode(p_direction, &p_direction_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_idx_encoded, &p_direction_encoded);
}

Control::TextDirection ItemList::get_item_text_direction(int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("get_item_text_direction")._native_ptr(), 4235602388);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Control::TextDirection(0)));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return (Control::TextDirection)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_idx_encoded);
}

void ItemList::set_item_language(int32_t p_idx, const String &p_language) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("set_item_language")._native_ptr(), 501894301);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_idx_encoded, &p_language);
}

String ItemList::get_item_language(int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("get_item_language")._native_ptr(), 844755477);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_idx_encoded);
}

void ItemList::set_item_auto_translate_mode(int32_t p_idx, Node::AutoTranslateMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("set_item_auto_translate_mode")._native_ptr(), 287402019);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_idx_encoded, &p_mode_encoded);
}

Node::AutoTranslateMode ItemList::get_item_auto_translate_mode(int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("get_item_auto_translate_mode")._native_ptr(), 906302372);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Node::AutoTranslateMode(0)));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return (Node::AutoTranslateMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_idx_encoded);
}

void ItemList::set_item_icon_transposed(int32_t p_idx, bool p_transposed) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("set_item_icon_transposed")._native_ptr(), 300928843);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	int8_t p_transposed_encoded;
	PtrToArg<bool>::encode(p_transposed, &p_transposed_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_idx_encoded, &p_transposed_encoded);
}

bool ItemList::is_item_icon_transposed(int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("is_item_icon_transposed")._native_ptr(), 1116898809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_idx_encoded);
}

void ItemList::set_item_icon_region(int32_t p_idx, const Rect2 &p_rect) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("set_item_icon_region")._native_ptr(), 1356297692);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_idx_encoded, &p_rect);
}

Rect2 ItemList::get_item_icon_region(int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("get_item_icon_region")._native_ptr(), 3327874267);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Rect2()));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<Rect2>(_gde_method_bind, _owner, &p_idx_encoded);
}

void ItemList::set_item_icon_modulate(int32_t p_idx, const Color &p_modulate) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("set_item_icon_modulate")._native_ptr(), 2878471219);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_idx_encoded, &p_modulate);
}

Color ItemList::get_item_icon_modulate(int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("get_item_icon_modulate")._native_ptr(), 3457211756);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Color()));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<Color>(_gde_method_bind, _owner, &p_idx_encoded);
}

void ItemList::set_item_selectable(int32_t p_idx, bool p_selectable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("set_item_selectable")._native_ptr(), 300928843);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	int8_t p_selectable_encoded;
	PtrToArg<bool>::encode(p_selectable, &p_selectable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_idx_encoded, &p_selectable_encoded);
}

bool ItemList::is_item_selectable(int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("is_item_selectable")._native_ptr(), 1116898809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_idx_encoded);
}

void ItemList::set_item_disabled(int32_t p_idx, bool p_disabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("set_item_disabled")._native_ptr(), 300928843);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	int8_t p_disabled_encoded;
	PtrToArg<bool>::encode(p_disabled, &p_disabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_idx_encoded, &p_disabled_encoded);
}

bool ItemList::is_item_disabled(int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("is_item_disabled")._native_ptr(), 1116898809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_idx_encoded);
}

void ItemList::set_item_metadata(int32_t p_idx, const Variant &p_metadata) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("set_item_metadata")._native_ptr(), 2152698145);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_idx_encoded, &p_metadata);
}

Variant ItemList::get_item_metadata(int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("get_item_metadata")._native_ptr(), 4227898402);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Variant()));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<Variant>(_gde_method_bind, _owner, &p_idx_encoded);
}

void ItemList::set_item_custom_bg_color(int32_t p_idx, const Color &p_custom_bg_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("set_item_custom_bg_color")._native_ptr(), 2878471219);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_idx_encoded, &p_custom_bg_color);
}

Color ItemList::get_item_custom_bg_color(int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("get_item_custom_bg_color")._native_ptr(), 3457211756);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Color()));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<Color>(_gde_method_bind, _owner, &p_idx_encoded);
}

void ItemList::set_item_custom_fg_color(int32_t p_idx, const Color &p_custom_fg_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("set_item_custom_fg_color")._native_ptr(), 2878471219);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_idx_encoded, &p_custom_fg_color);
}

Color ItemList::get_item_custom_fg_color(int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("get_item_custom_fg_color")._native_ptr(), 3457211756);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Color()));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<Color>(_gde_method_bind, _owner, &p_idx_encoded);
}

Rect2 ItemList::get_item_rect(int32_t p_idx, bool p_expand) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("get_item_rect")._native_ptr(), 159227807);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Rect2()));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	int8_t p_expand_encoded;
	PtrToArg<bool>::encode(p_expand, &p_expand_encoded);
	return ::godot::internal::_call_native_mb_ret<Rect2>(_gde_method_bind, _owner, &p_idx_encoded, &p_expand_encoded);
}

void ItemList::set_item_tooltip_enabled(int32_t p_idx, bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("set_item_tooltip_enabled")._native_ptr(), 300928843);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_idx_encoded, &p_enable_encoded);
}

bool ItemList::is_item_tooltip_enabled(int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("is_item_tooltip_enabled")._native_ptr(), 1116898809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_idx_encoded);
}

void ItemList::set_item_tooltip(int32_t p_idx, const String &p_tooltip) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("set_item_tooltip")._native_ptr(), 501894301);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_idx_encoded, &p_tooltip);
}

String ItemList::get_item_tooltip(int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("get_item_tooltip")._native_ptr(), 844755477);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_idx_encoded);
}

void ItemList::select(int32_t p_idx, bool p_single) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("select")._native_ptr(), 972357352);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	int8_t p_single_encoded;
	PtrToArg<bool>::encode(p_single, &p_single_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_idx_encoded, &p_single_encoded);
}

void ItemList::deselect(int32_t p_idx) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("deselect")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_idx_encoded);
}

void ItemList::deselect_all() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("deselect_all")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

bool ItemList::is_selected(int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("is_selected")._native_ptr(), 1116898809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_idx_encoded);
}

PackedInt32Array ItemList::get_selected_items() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("get_selected_items")._native_ptr(), 969006518);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedInt32Array()));
	return ::godot::internal::_call_native_mb_ret<PackedInt32Array>(_gde_method_bind, _owner);
}

void ItemList::move_item(int32_t p_from_idx, int32_t p_to_idx) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("move_item")._native_ptr(), 3937882851);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_from_idx_encoded;
	PtrToArg<int64_t>::encode(p_from_idx, &p_from_idx_encoded);
	int64_t p_to_idx_encoded;
	PtrToArg<int64_t>::encode(p_to_idx, &p_to_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_from_idx_encoded, &p_to_idx_encoded);
}

void ItemList::set_item_count(int32_t p_count) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("set_item_count")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_count_encoded;
	PtrToArg<int64_t>::encode(p_count, &p_count_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_count_encoded);
}

int32_t ItemList::get_item_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("get_item_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void ItemList::remove_item(int32_t p_idx) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("remove_item")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_idx_encoded);
}

void ItemList::clear() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("clear")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void ItemList::sort_items_by_text() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("sort_items_by_text")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void ItemList::set_fixed_column_width(int32_t p_width) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("set_fixed_column_width")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_width_encoded;
	PtrToArg<int64_t>::encode(p_width, &p_width_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_width_encoded);
}

int32_t ItemList::get_fixed_column_width() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("get_fixed_column_width")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void ItemList::set_same_column_width(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("set_same_column_width")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool ItemList::is_same_column_width() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("is_same_column_width")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void ItemList::set_max_text_lines(int32_t p_lines) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("set_max_text_lines")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_lines_encoded;
	PtrToArg<int64_t>::encode(p_lines, &p_lines_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_lines_encoded);
}

int32_t ItemList::get_max_text_lines() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("get_max_text_lines")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void ItemList::set_max_columns(int32_t p_amount) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("set_max_columns")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_amount_encoded;
	PtrToArg<int64_t>::encode(p_amount, &p_amount_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_amount_encoded);
}

int32_t ItemList::get_max_columns() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("get_max_columns")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void ItemList::set_select_mode(ItemList::SelectMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("set_select_mode")._native_ptr(), 928267388);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

ItemList::SelectMode ItemList::get_select_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("get_select_mode")._native_ptr(), 1191945842);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (ItemList::SelectMode(0)));
	return (ItemList::SelectMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void ItemList::set_icon_mode(ItemList::IconMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("set_icon_mode")._native_ptr(), 2025053633);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

ItemList::IconMode ItemList::get_icon_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("get_icon_mode")._native_ptr(), 3353929232);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (ItemList::IconMode(0)));
	return (ItemList::IconMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void ItemList::set_fixed_icon_size(const Vector2i &p_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("set_fixed_icon_size")._native_ptr(), 1130785943);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_size);
}

Vector2i ItemList::get_fixed_icon_size() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("get_fixed_icon_size")._native_ptr(), 3690982128);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2i()));
	return ::godot::internal::_call_native_mb_ret<Vector2i>(_gde_method_bind, _owner);
}

void ItemList::set_icon_scale(float p_scale) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("set_icon_scale")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_scale_encoded;
	PtrToArg<double>::encode(p_scale, &p_scale_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_scale_encoded);
}

float ItemList::get_icon_scale() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("get_icon_scale")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void ItemList::set_allow_rmb_select(bool p_allow) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("set_allow_rmb_select")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_allow_encoded;
	PtrToArg<bool>::encode(p_allow, &p_allow_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_allow_encoded);
}

bool ItemList::get_allow_rmb_select() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("get_allow_rmb_select")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void ItemList::set_allow_reselect(bool p_allow) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("set_allow_reselect")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_allow_encoded;
	PtrToArg<bool>::encode(p_allow, &p_allow_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_allow_encoded);
}

bool ItemList::get_allow_reselect() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("get_allow_reselect")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void ItemList::set_allow_search(bool p_allow) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("set_allow_search")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_allow_encoded;
	PtrToArg<bool>::encode(p_allow, &p_allow_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_allow_encoded);
}

bool ItemList::get_allow_search() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("get_allow_search")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void ItemList::set_auto_width(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("set_auto_width")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool ItemList::has_auto_width() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("has_auto_width")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void ItemList::set_auto_height(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("set_auto_height")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool ItemList::has_auto_height() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("has_auto_height")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

bool ItemList::is_anything_selected() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("is_anything_selected")._native_ptr(), 2240911060);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

int32_t ItemList::get_item_at_position(const Vector2 &p_position, bool p_exact) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("get_item_at_position")._native_ptr(), 2300324924);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int8_t p_exact_encoded;
	PtrToArg<bool>::encode(p_exact, &p_exact_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_position, &p_exact_encoded);
}

void ItemList::ensure_current_is_visible() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("ensure_current_is_visible")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

VScrollBar *ItemList::get_v_scroll_bar() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("get_v_scroll_bar")._native_ptr(), 2630340773);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<VScrollBar>(_gde_method_bind, _owner);
}

HScrollBar *ItemList::get_h_scroll_bar() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("get_h_scroll_bar")._native_ptr(), 4004517983);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<HScrollBar>(_gde_method_bind, _owner);
}

void ItemList::set_scroll_hint_mode(ItemList::ScrollHintMode p_scroll_hint_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("set_scroll_hint_mode")._native_ptr(), 2917787337);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_scroll_hint_mode_encoded;
	PtrToArg<int64_t>::encode(p_scroll_hint_mode, &p_scroll_hint_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_scroll_hint_mode_encoded);
}

ItemList::ScrollHintMode ItemList::get_scroll_hint_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("get_scroll_hint_mode")._native_ptr(), 2522227939);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (ItemList::ScrollHintMode(0)));
	return (ItemList::ScrollHintMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void ItemList::set_tile_scroll_hint(bool p_tile_scroll_hint) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("set_tile_scroll_hint")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_tile_scroll_hint_encoded;
	PtrToArg<bool>::encode(p_tile_scroll_hint, &p_tile_scroll_hint_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_tile_scroll_hint_encoded);
}

bool ItemList::is_scroll_hint_tiled() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("is_scroll_hint_tiled")._native_ptr(), 2240911060);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void ItemList::set_text_overrun_behavior(TextServer::OverrunBehavior p_overrun_behavior) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("set_text_overrun_behavior")._native_ptr(), 1008890932);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_overrun_behavior_encoded;
	PtrToArg<int64_t>::encode(p_overrun_behavior, &p_overrun_behavior_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_overrun_behavior_encoded);
}

TextServer::OverrunBehavior ItemList::get_text_overrun_behavior() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("get_text_overrun_behavior")._native_ptr(), 3779142101);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TextServer::OverrunBehavior(0)));
	return (TextServer::OverrunBehavior)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void ItemList::set_wraparound_items(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("set_wraparound_items")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool ItemList::has_wraparound_items() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("has_wraparound_items")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void ItemList::force_update_list_size() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ItemList::get_class_static()._native_ptr(), StringName("force_update_list_size")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

} // namespace godot
