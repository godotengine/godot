/**************************************************************************/
/*  editor_context_menu_plugin.cpp                                        */
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

#include <godot_cpp/classes/editor_context_menu_plugin.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/popup_menu.hpp>
#include <godot_cpp/classes/shortcut.hpp>
#include <godot_cpp/variant/callable.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>
#include <godot_cpp/variant/string.hpp>

namespace godot {

void EditorContextMenuPlugin::add_menu_shortcut(const Ref<Shortcut> &p_shortcut, const Callable &p_callback) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorContextMenuPlugin::get_class_static()._native_ptr(), StringName("add_menu_shortcut")._native_ptr(), 851596305);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_shortcut != nullptr ? &p_shortcut->_owner : nullptr), &p_callback);
}

void EditorContextMenuPlugin::add_context_menu_item(const String &p_name, const Callable &p_callback, const Ref<Texture2D> &p_icon) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorContextMenuPlugin::get_class_static()._native_ptr(), StringName("add_context_menu_item")._native_ptr(), 2748336951);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name, &p_callback, (p_icon != nullptr ? &p_icon->_owner : nullptr));
}

void EditorContextMenuPlugin::add_context_menu_item_from_shortcut(const String &p_name, const Ref<Shortcut> &p_shortcut, const Ref<Texture2D> &p_icon) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorContextMenuPlugin::get_class_static()._native_ptr(), StringName("add_context_menu_item_from_shortcut")._native_ptr(), 3799546916);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name, (p_shortcut != nullptr ? &p_shortcut->_owner : nullptr), (p_icon != nullptr ? &p_icon->_owner : nullptr));
}

void EditorContextMenuPlugin::add_context_submenu_item(const String &p_name, PopupMenu *p_menu, const Ref<Texture2D> &p_icon) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorContextMenuPlugin::get_class_static()._native_ptr(), StringName("add_context_submenu_item")._native_ptr(), 1994674995);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name, (p_menu != nullptr ? &p_menu->_owner : nullptr), (p_icon != nullptr ? &p_icon->_owner : nullptr));
}

void EditorContextMenuPlugin::_popup_menu(const PackedStringArray &p_paths) {}

} // namespace godot
