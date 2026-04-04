/**************************************************************************/
/*  editor_dock.cpp                                                       */
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

#include <godot_cpp/classes/editor_dock.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/config_file.hpp>
#include <godot_cpp/classes/shortcut.hpp>
#include <godot_cpp/classes/texture2d.hpp>

namespace godot {

void EditorDock::open() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorDock::get_class_static()._native_ptr(), StringName("open")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void EditorDock::make_visible() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorDock::get_class_static()._native_ptr(), StringName("make_visible")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void EditorDock::close() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorDock::get_class_static()._native_ptr(), StringName("close")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void EditorDock::set_title(const String &p_title) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorDock::get_class_static()._native_ptr(), StringName("set_title")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_title);
}

String EditorDock::get_title() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorDock::get_class_static()._native_ptr(), StringName("get_title")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

void EditorDock::set_layout_key(const String &p_layout_key) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorDock::get_class_static()._native_ptr(), StringName("set_layout_key")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_layout_key);
}

String EditorDock::get_layout_key() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorDock::get_class_static()._native_ptr(), StringName("get_layout_key")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

void EditorDock::set_global(bool p_global) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorDock::get_class_static()._native_ptr(), StringName("set_global")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_global_encoded;
	PtrToArg<bool>::encode(p_global, &p_global_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_global_encoded);
}

bool EditorDock::is_global() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorDock::get_class_static()._native_ptr(), StringName("is_global")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void EditorDock::set_transient(bool p_transient) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorDock::get_class_static()._native_ptr(), StringName("set_transient")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_transient_encoded;
	PtrToArg<bool>::encode(p_transient, &p_transient_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_transient_encoded);
}

bool EditorDock::is_transient() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorDock::get_class_static()._native_ptr(), StringName("is_transient")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void EditorDock::set_closable(bool p_closable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorDock::get_class_static()._native_ptr(), StringName("set_closable")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_closable_encoded;
	PtrToArg<bool>::encode(p_closable, &p_closable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_closable_encoded);
}

bool EditorDock::is_closable() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorDock::get_class_static()._native_ptr(), StringName("is_closable")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void EditorDock::set_icon_name(const StringName &p_icon_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorDock::get_class_static()._native_ptr(), StringName("set_icon_name")._native_ptr(), 3304788590);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_icon_name);
}

StringName EditorDock::get_icon_name() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorDock::get_class_static()._native_ptr(), StringName("get_icon_name")._native_ptr(), 2002593661);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (StringName()));
	return ::godot::internal::_call_native_mb_ret<StringName>(_gde_method_bind, _owner);
}

void EditorDock::set_dock_icon(const Ref<Texture2D> &p_icon) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorDock::get_class_static()._native_ptr(), StringName("set_dock_icon")._native_ptr(), 4051416890);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_icon != nullptr ? &p_icon->_owner : nullptr));
}

Ref<Texture2D> EditorDock::get_dock_icon() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorDock::get_class_static()._native_ptr(), StringName("get_dock_icon")._native_ptr(), 3635182373);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Texture2D>()));
	return Ref<Texture2D>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Texture2D>(_gde_method_bind, _owner));
}

void EditorDock::set_force_show_icon(bool p_force) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorDock::get_class_static()._native_ptr(), StringName("set_force_show_icon")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_force_encoded;
	PtrToArg<bool>::encode(p_force, &p_force_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_force_encoded);
}

bool EditorDock::get_force_show_icon() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorDock::get_class_static()._native_ptr(), StringName("get_force_show_icon")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void EditorDock::set_title_color(const Color &p_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorDock::get_class_static()._native_ptr(), StringName("set_title_color")._native_ptr(), 2920490490);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_color);
}

Color EditorDock::get_title_color() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorDock::get_class_static()._native_ptr(), StringName("get_title_color")._native_ptr(), 3444240500);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Color()));
	return ::godot::internal::_call_native_mb_ret<Color>(_gde_method_bind, _owner);
}

void EditorDock::set_dock_shortcut(const Ref<Shortcut> &p_shortcut) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorDock::get_class_static()._native_ptr(), StringName("set_dock_shortcut")._native_ptr(), 857163497);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_shortcut != nullptr ? &p_shortcut->_owner : nullptr));
}

Ref<Shortcut> EditorDock::get_dock_shortcut() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorDock::get_class_static()._native_ptr(), StringName("get_dock_shortcut")._native_ptr(), 3415666916);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Shortcut>()));
	return Ref<Shortcut>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Shortcut>(_gde_method_bind, _owner));
}

void EditorDock::set_default_slot(EditorDock::DockSlot p_slot) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorDock::get_class_static()._native_ptr(), StringName("set_default_slot")._native_ptr(), 4142995464);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_slot_encoded;
	PtrToArg<int64_t>::encode(p_slot, &p_slot_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_slot_encoded);
}

EditorDock::DockSlot EditorDock::get_default_slot() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorDock::get_class_static()._native_ptr(), StringName("get_default_slot")._native_ptr(), 3298961740);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (EditorDock::DockSlot(0)));
	return (EditorDock::DockSlot)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void EditorDock::set_available_layouts(BitField<EditorDock::DockLayout> p_layouts) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorDock::get_class_static()._native_ptr(), StringName("set_available_layouts")._native_ptr(), 3440531249);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_layouts);
}

BitField<EditorDock::DockLayout> EditorDock::get_available_layouts() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorDock::get_class_static()._native_ptr(), StringName("get_available_layouts")._native_ptr(), 495015512);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (BitField<EditorDock::DockLayout>(0)));
	return (int64_t)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void EditorDock::_update_layout(int32_t p_layout) {}

void EditorDock::_save_layout_to_config(const Ref<ConfigFile> &p_config, const String &p_section) const {}

void EditorDock::_load_layout_from_config(const Ref<ConfigFile> &p_config, const String &p_section) {}

} // namespace godot
