/**************************************************************************/
/*  native_menu.cpp                                                       */
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

#include <godot_cpp/classes/native_menu.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/texture2d.hpp>
#include <godot_cpp/variant/vector2i.hpp>

namespace godot {

NativeMenu *NativeMenu::singleton = nullptr;

NativeMenu *NativeMenu::get_singleton() {
	if (unlikely(singleton == nullptr)) {
		GDExtensionObjectPtr singleton_obj = ::godot::gdextension_interface::global_get_singleton(NativeMenu::get_class_static()._native_ptr());
#ifdef DEBUG_ENABLED
		ERR_FAIL_NULL_V(singleton_obj, nullptr);
#endif // DEBUG_ENABLED
		singleton = reinterpret_cast<NativeMenu *>(::godot::gdextension_interface::object_get_instance_binding(singleton_obj, ::godot::gdextension_interface::token, &NativeMenu::_gde_binding_callbacks));
#ifdef DEBUG_ENABLED
		ERR_FAIL_NULL_V(singleton, nullptr);
#endif // DEBUG_ENABLED
		if (likely(singleton)) {
			ClassDB::_register_engine_singleton(NativeMenu::get_class_static(), singleton);
		}
	}
	return singleton;
}

NativeMenu::~NativeMenu() {
	if (singleton == this) {
		ClassDB::_unregister_engine_singleton(NativeMenu::get_class_static());
		singleton = nullptr;
	}
}

bool NativeMenu::has_feature(NativeMenu::Feature p_feature) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NativeMenu::get_class_static()._native_ptr(), StringName("has_feature")._native_ptr(), 1708975490);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_feature_encoded;
	PtrToArg<int64_t>::encode(p_feature, &p_feature_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_feature_encoded);
}

bool NativeMenu::has_system_menu(NativeMenu::SystemMenus p_menu_id) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NativeMenu::get_class_static()._native_ptr(), StringName("has_system_menu")._native_ptr(), 718213027);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_menu_id_encoded;
	PtrToArg<int64_t>::encode(p_menu_id, &p_menu_id_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_menu_id_encoded);
}

RID NativeMenu::get_system_menu(NativeMenu::SystemMenus p_menu_id) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NativeMenu::get_class_static()._native_ptr(), StringName("get_system_menu")._native_ptr(), 469707506);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	int64_t p_menu_id_encoded;
	PtrToArg<int64_t>::encode(p_menu_id, &p_menu_id_encoded);
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_menu_id_encoded);
}

String NativeMenu::get_system_menu_name(NativeMenu::SystemMenus p_menu_id) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NativeMenu::get_class_static()._native_ptr(), StringName("get_system_menu_name")._native_ptr(), 1281499290);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_menu_id_encoded;
	PtrToArg<int64_t>::encode(p_menu_id, &p_menu_id_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_menu_id_encoded);
}

String NativeMenu::get_system_menu_text(NativeMenu::SystemMenus p_menu_id) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NativeMenu::get_class_static()._native_ptr(), StringName("get_system_menu_text")._native_ptr(), 1281499290);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_menu_id_encoded;
	PtrToArg<int64_t>::encode(p_menu_id, &p_menu_id_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_menu_id_encoded);
}

void NativeMenu::set_system_menu_text(NativeMenu::SystemMenus p_menu_id, const String &p_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NativeMenu::get_class_static()._native_ptr(), StringName("set_system_menu_text")._native_ptr(), 3925225603);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_menu_id_encoded;
	PtrToArg<int64_t>::encode(p_menu_id, &p_menu_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_menu_id_encoded, &p_name);
}

RID NativeMenu::create_menu() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NativeMenu::get_class_static()._native_ptr(), StringName("create_menu")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

bool NativeMenu::has_menu(const RID &p_rid) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NativeMenu::get_class_static()._native_ptr(), StringName("has_menu")._native_ptr(), 4155700596);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_rid);
}

void NativeMenu::free_menu(const RID &p_rid) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NativeMenu::get_class_static()._native_ptr(), StringName("free_menu")._native_ptr(), 2722037293);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_rid);
}

Vector2 NativeMenu::get_size(const RID &p_rid) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NativeMenu::get_class_static()._native_ptr(), StringName("get_size")._native_ptr(), 2440833711);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner, &p_rid);
}

void NativeMenu::popup(const RID &p_rid, const Vector2i &p_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NativeMenu::get_class_static()._native_ptr(), StringName("popup")._native_ptr(), 2450610377);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_rid, &p_position);
}

void NativeMenu::set_interface_direction(const RID &p_rid, bool p_is_rtl) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NativeMenu::get_class_static()._native_ptr(), StringName("set_interface_direction")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_is_rtl_encoded;
	PtrToArg<bool>::encode(p_is_rtl, &p_is_rtl_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_rid, &p_is_rtl_encoded);
}

void NativeMenu::set_popup_open_callback(const RID &p_rid, const Callable &p_callback) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NativeMenu::get_class_static()._native_ptr(), StringName("set_popup_open_callback")._native_ptr(), 3379118538);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_rid, &p_callback);
}

Callable NativeMenu::get_popup_open_callback(const RID &p_rid) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NativeMenu::get_class_static()._native_ptr(), StringName("get_popup_open_callback")._native_ptr(), 3170603026);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Callable()));
	return ::godot::internal::_call_native_mb_ret<Callable>(_gde_method_bind, _owner, &p_rid);
}

void NativeMenu::set_popup_close_callback(const RID &p_rid, const Callable &p_callback) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NativeMenu::get_class_static()._native_ptr(), StringName("set_popup_close_callback")._native_ptr(), 3379118538);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_rid, &p_callback);
}

Callable NativeMenu::get_popup_close_callback(const RID &p_rid) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NativeMenu::get_class_static()._native_ptr(), StringName("get_popup_close_callback")._native_ptr(), 3170603026);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Callable()));
	return ::godot::internal::_call_native_mb_ret<Callable>(_gde_method_bind, _owner, &p_rid);
}

void NativeMenu::set_minimum_width(const RID &p_rid, float p_width) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NativeMenu::get_class_static()._native_ptr(), StringName("set_minimum_width")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_width_encoded;
	PtrToArg<double>::encode(p_width, &p_width_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_rid, &p_width_encoded);
}

float NativeMenu::get_minimum_width(const RID &p_rid) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NativeMenu::get_class_static()._native_ptr(), StringName("get_minimum_width")._native_ptr(), 866169185);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_rid);
}

bool NativeMenu::is_opened(const RID &p_rid) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NativeMenu::get_class_static()._native_ptr(), StringName("is_opened")._native_ptr(), 4155700596);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_rid);
}

int32_t NativeMenu::add_submenu_item(const RID &p_rid, const String &p_label, const RID &p_submenu_rid, const Variant &p_tag, int32_t p_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NativeMenu::get_class_static()._native_ptr(), StringName("add_submenu_item")._native_ptr(), 1002030223);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_rid, &p_label, &p_submenu_rid, &p_tag, &p_index_encoded);
}

int32_t NativeMenu::add_item(const RID &p_rid, const String &p_label, const Callable &p_callback, const Callable &p_key_callback, const Variant &p_tag, Key p_accelerator, int32_t p_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NativeMenu::get_class_static()._native_ptr(), StringName("add_item")._native_ptr(), 980552939);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_accelerator_encoded;
	PtrToArg<int64_t>::encode(p_accelerator, &p_accelerator_encoded);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_rid, &p_label, &p_callback, &p_key_callback, &p_tag, &p_accelerator_encoded, &p_index_encoded);
}

int32_t NativeMenu::add_check_item(const RID &p_rid, const String &p_label, const Callable &p_callback, const Callable &p_key_callback, const Variant &p_tag, Key p_accelerator, int32_t p_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NativeMenu::get_class_static()._native_ptr(), StringName("add_check_item")._native_ptr(), 980552939);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_accelerator_encoded;
	PtrToArg<int64_t>::encode(p_accelerator, &p_accelerator_encoded);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_rid, &p_label, &p_callback, &p_key_callback, &p_tag, &p_accelerator_encoded, &p_index_encoded);
}

int32_t NativeMenu::add_icon_item(const RID &p_rid, const Ref<Texture2D> &p_icon, const String &p_label, const Callable &p_callback, const Callable &p_key_callback, const Variant &p_tag, Key p_accelerator, int32_t p_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NativeMenu::get_class_static()._native_ptr(), StringName("add_icon_item")._native_ptr(), 1372188274);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_accelerator_encoded;
	PtrToArg<int64_t>::encode(p_accelerator, &p_accelerator_encoded);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_rid, (p_icon != nullptr ? &p_icon->_owner : nullptr), &p_label, &p_callback, &p_key_callback, &p_tag, &p_accelerator_encoded, &p_index_encoded);
}

int32_t NativeMenu::add_icon_check_item(const RID &p_rid, const Ref<Texture2D> &p_icon, const String &p_label, const Callable &p_callback, const Callable &p_key_callback, const Variant &p_tag, Key p_accelerator, int32_t p_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NativeMenu::get_class_static()._native_ptr(), StringName("add_icon_check_item")._native_ptr(), 1372188274);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_accelerator_encoded;
	PtrToArg<int64_t>::encode(p_accelerator, &p_accelerator_encoded);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_rid, (p_icon != nullptr ? &p_icon->_owner : nullptr), &p_label, &p_callback, &p_key_callback, &p_tag, &p_accelerator_encoded, &p_index_encoded);
}

int32_t NativeMenu::add_radio_check_item(const RID &p_rid, const String &p_label, const Callable &p_callback, const Callable &p_key_callback, const Variant &p_tag, Key p_accelerator, int32_t p_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NativeMenu::get_class_static()._native_ptr(), StringName("add_radio_check_item")._native_ptr(), 980552939);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_accelerator_encoded;
	PtrToArg<int64_t>::encode(p_accelerator, &p_accelerator_encoded);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_rid, &p_label, &p_callback, &p_key_callback, &p_tag, &p_accelerator_encoded, &p_index_encoded);
}

int32_t NativeMenu::add_icon_radio_check_item(const RID &p_rid, const Ref<Texture2D> &p_icon, const String &p_label, const Callable &p_callback, const Callable &p_key_callback, const Variant &p_tag, Key p_accelerator, int32_t p_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NativeMenu::get_class_static()._native_ptr(), StringName("add_icon_radio_check_item")._native_ptr(), 1372188274);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_accelerator_encoded;
	PtrToArg<int64_t>::encode(p_accelerator, &p_accelerator_encoded);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_rid, (p_icon != nullptr ? &p_icon->_owner : nullptr), &p_label, &p_callback, &p_key_callback, &p_tag, &p_accelerator_encoded, &p_index_encoded);
}

int32_t NativeMenu::add_multistate_item(const RID &p_rid, const String &p_label, int32_t p_max_states, int32_t p_default_state, const Callable &p_callback, const Callable &p_key_callback, const Variant &p_tag, Key p_accelerator, int32_t p_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NativeMenu::get_class_static()._native_ptr(), StringName("add_multistate_item")._native_ptr(), 2674635658);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_max_states_encoded;
	PtrToArg<int64_t>::encode(p_max_states, &p_max_states_encoded);
	int64_t p_default_state_encoded;
	PtrToArg<int64_t>::encode(p_default_state, &p_default_state_encoded);
	int64_t p_accelerator_encoded;
	PtrToArg<int64_t>::encode(p_accelerator, &p_accelerator_encoded);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_rid, &p_label, &p_max_states_encoded, &p_default_state_encoded, &p_callback, &p_key_callback, &p_tag, &p_accelerator_encoded, &p_index_encoded);
}

int32_t NativeMenu::add_separator(const RID &p_rid, int32_t p_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NativeMenu::get_class_static()._native_ptr(), StringName("add_separator")._native_ptr(), 448810126);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_rid, &p_index_encoded);
}

int32_t NativeMenu::find_item_index_with_text(const RID &p_rid, const String &p_text) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NativeMenu::get_class_static()._native_ptr(), StringName("find_item_index_with_text")._native_ptr(), 1362438794);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_rid, &p_text);
}

int32_t NativeMenu::find_item_index_with_tag(const RID &p_rid, const Variant &p_tag) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NativeMenu::get_class_static()._native_ptr(), StringName("find_item_index_with_tag")._native_ptr(), 1260085030);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_rid, &p_tag);
}

int32_t NativeMenu::find_item_index_with_submenu(const RID &p_rid, const RID &p_submenu_rid) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NativeMenu::get_class_static()._native_ptr(), StringName("find_item_index_with_submenu")._native_ptr(), 893635918);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_rid, &p_submenu_rid);
}

bool NativeMenu::is_item_checked(const RID &p_rid, int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NativeMenu::get_class_static()._native_ptr(), StringName("is_item_checked")._native_ptr(), 3120086654);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_rid, &p_idx_encoded);
}

bool NativeMenu::is_item_checkable(const RID &p_rid, int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NativeMenu::get_class_static()._native_ptr(), StringName("is_item_checkable")._native_ptr(), 3120086654);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_rid, &p_idx_encoded);
}

bool NativeMenu::is_item_radio_checkable(const RID &p_rid, int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NativeMenu::get_class_static()._native_ptr(), StringName("is_item_radio_checkable")._native_ptr(), 3120086654);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_rid, &p_idx_encoded);
}

Callable NativeMenu::get_item_callback(const RID &p_rid, int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NativeMenu::get_class_static()._native_ptr(), StringName("get_item_callback")._native_ptr(), 1639989698);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Callable()));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<Callable>(_gde_method_bind, _owner, &p_rid, &p_idx_encoded);
}

Callable NativeMenu::get_item_key_callback(const RID &p_rid, int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NativeMenu::get_class_static()._native_ptr(), StringName("get_item_key_callback")._native_ptr(), 1639989698);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Callable()));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<Callable>(_gde_method_bind, _owner, &p_rid, &p_idx_encoded);
}

Variant NativeMenu::get_item_tag(const RID &p_rid, int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NativeMenu::get_class_static()._native_ptr(), StringName("get_item_tag")._native_ptr(), 4069510997);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Variant()));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<Variant>(_gde_method_bind, _owner, &p_rid, &p_idx_encoded);
}

String NativeMenu::get_item_text(const RID &p_rid, int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NativeMenu::get_class_static()._native_ptr(), StringName("get_item_text")._native_ptr(), 1464764419);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_rid, &p_idx_encoded);
}

RID NativeMenu::get_item_submenu(const RID &p_rid, int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NativeMenu::get_class_static()._native_ptr(), StringName("get_item_submenu")._native_ptr(), 1066463050);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_rid, &p_idx_encoded);
}

Key NativeMenu::get_item_accelerator(const RID &p_rid, int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NativeMenu::get_class_static()._native_ptr(), StringName("get_item_accelerator")._native_ptr(), 316800700);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Key(0)));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return (Key)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_rid, &p_idx_encoded);
}

bool NativeMenu::is_item_disabled(const RID &p_rid, int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NativeMenu::get_class_static()._native_ptr(), StringName("is_item_disabled")._native_ptr(), 3120086654);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_rid, &p_idx_encoded);
}

bool NativeMenu::is_item_hidden(const RID &p_rid, int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NativeMenu::get_class_static()._native_ptr(), StringName("is_item_hidden")._native_ptr(), 3120086654);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_rid, &p_idx_encoded);
}

String NativeMenu::get_item_tooltip(const RID &p_rid, int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NativeMenu::get_class_static()._native_ptr(), StringName("get_item_tooltip")._native_ptr(), 1464764419);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_rid, &p_idx_encoded);
}

int32_t NativeMenu::get_item_state(const RID &p_rid, int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NativeMenu::get_class_static()._native_ptr(), StringName("get_item_state")._native_ptr(), 1120910005);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_rid, &p_idx_encoded);
}

int32_t NativeMenu::get_item_max_states(const RID &p_rid, int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NativeMenu::get_class_static()._native_ptr(), StringName("get_item_max_states")._native_ptr(), 1120910005);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_rid, &p_idx_encoded);
}

Ref<Texture2D> NativeMenu::get_item_icon(const RID &p_rid, int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NativeMenu::get_class_static()._native_ptr(), StringName("get_item_icon")._native_ptr(), 3391850701);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Texture2D>()));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return Ref<Texture2D>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Texture2D>(_gde_method_bind, _owner, &p_rid, &p_idx_encoded));
}

int32_t NativeMenu::get_item_indentation_level(const RID &p_rid, int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NativeMenu::get_class_static()._native_ptr(), StringName("get_item_indentation_level")._native_ptr(), 1120910005);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_rid, &p_idx_encoded);
}

void NativeMenu::set_item_checked(const RID &p_rid, int32_t p_idx, bool p_checked) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NativeMenu::get_class_static()._native_ptr(), StringName("set_item_checked")._native_ptr(), 2658558584);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	int8_t p_checked_encoded;
	PtrToArg<bool>::encode(p_checked, &p_checked_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_rid, &p_idx_encoded, &p_checked_encoded);
}

void NativeMenu::set_item_checkable(const RID &p_rid, int32_t p_idx, bool p_checkable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NativeMenu::get_class_static()._native_ptr(), StringName("set_item_checkable")._native_ptr(), 2658558584);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	int8_t p_checkable_encoded;
	PtrToArg<bool>::encode(p_checkable, &p_checkable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_rid, &p_idx_encoded, &p_checkable_encoded);
}

void NativeMenu::set_item_radio_checkable(const RID &p_rid, int32_t p_idx, bool p_checkable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NativeMenu::get_class_static()._native_ptr(), StringName("set_item_radio_checkable")._native_ptr(), 2658558584);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	int8_t p_checkable_encoded;
	PtrToArg<bool>::encode(p_checkable, &p_checkable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_rid, &p_idx_encoded, &p_checkable_encoded);
}

void NativeMenu::set_item_callback(const RID &p_rid, int32_t p_idx, const Callable &p_callback) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NativeMenu::get_class_static()._native_ptr(), StringName("set_item_callback")._native_ptr(), 2779810226);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_rid, &p_idx_encoded, &p_callback);
}

void NativeMenu::set_item_hover_callbacks(const RID &p_rid, int32_t p_idx, const Callable &p_callback) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NativeMenu::get_class_static()._native_ptr(), StringName("set_item_hover_callbacks")._native_ptr(), 2779810226);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_rid, &p_idx_encoded, &p_callback);
}

void NativeMenu::set_item_key_callback(const RID &p_rid, int32_t p_idx, const Callable &p_key_callback) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NativeMenu::get_class_static()._native_ptr(), StringName("set_item_key_callback")._native_ptr(), 2779810226);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_rid, &p_idx_encoded, &p_key_callback);
}

void NativeMenu::set_item_tag(const RID &p_rid, int32_t p_idx, const Variant &p_tag) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NativeMenu::get_class_static()._native_ptr(), StringName("set_item_tag")._native_ptr(), 2706844827);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_rid, &p_idx_encoded, &p_tag);
}

void NativeMenu::set_item_text(const RID &p_rid, int32_t p_idx, const String &p_text) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NativeMenu::get_class_static()._native_ptr(), StringName("set_item_text")._native_ptr(), 4153150897);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_rid, &p_idx_encoded, &p_text);
}

void NativeMenu::set_item_submenu(const RID &p_rid, int32_t p_idx, const RID &p_submenu_rid) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NativeMenu::get_class_static()._native_ptr(), StringName("set_item_submenu")._native_ptr(), 2310537182);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_rid, &p_idx_encoded, &p_submenu_rid);
}

void NativeMenu::set_item_accelerator(const RID &p_rid, int32_t p_idx, Key p_keycode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NativeMenu::get_class_static()._native_ptr(), StringName("set_item_accelerator")._native_ptr(), 786300043);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	int64_t p_keycode_encoded;
	PtrToArg<int64_t>::encode(p_keycode, &p_keycode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_rid, &p_idx_encoded, &p_keycode_encoded);
}

void NativeMenu::set_item_disabled(const RID &p_rid, int32_t p_idx, bool p_disabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NativeMenu::get_class_static()._native_ptr(), StringName("set_item_disabled")._native_ptr(), 2658558584);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	int8_t p_disabled_encoded;
	PtrToArg<bool>::encode(p_disabled, &p_disabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_rid, &p_idx_encoded, &p_disabled_encoded);
}

void NativeMenu::set_item_hidden(const RID &p_rid, int32_t p_idx, bool p_hidden) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NativeMenu::get_class_static()._native_ptr(), StringName("set_item_hidden")._native_ptr(), 2658558584);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	int8_t p_hidden_encoded;
	PtrToArg<bool>::encode(p_hidden, &p_hidden_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_rid, &p_idx_encoded, &p_hidden_encoded);
}

void NativeMenu::set_item_tooltip(const RID &p_rid, int32_t p_idx, const String &p_tooltip) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NativeMenu::get_class_static()._native_ptr(), StringName("set_item_tooltip")._native_ptr(), 4153150897);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_rid, &p_idx_encoded, &p_tooltip);
}

void NativeMenu::set_item_state(const RID &p_rid, int32_t p_idx, int32_t p_state) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NativeMenu::get_class_static()._native_ptr(), StringName("set_item_state")._native_ptr(), 4288446313);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	int64_t p_state_encoded;
	PtrToArg<int64_t>::encode(p_state, &p_state_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_rid, &p_idx_encoded, &p_state_encoded);
}

void NativeMenu::set_item_max_states(const RID &p_rid, int32_t p_idx, int32_t p_max_states) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NativeMenu::get_class_static()._native_ptr(), StringName("set_item_max_states")._native_ptr(), 4288446313);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	int64_t p_max_states_encoded;
	PtrToArg<int64_t>::encode(p_max_states, &p_max_states_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_rid, &p_idx_encoded, &p_max_states_encoded);
}

void NativeMenu::set_item_icon(const RID &p_rid, int32_t p_idx, const Ref<Texture2D> &p_icon) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NativeMenu::get_class_static()._native_ptr(), StringName("set_item_icon")._native_ptr(), 1388763257);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_rid, &p_idx_encoded, (p_icon != nullptr ? &p_icon->_owner : nullptr));
}

void NativeMenu::set_item_indentation_level(const RID &p_rid, int32_t p_idx, int32_t p_level) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NativeMenu::get_class_static()._native_ptr(), StringName("set_item_indentation_level")._native_ptr(), 4288446313);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	int64_t p_level_encoded;
	PtrToArg<int64_t>::encode(p_level, &p_level_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_rid, &p_idx_encoded, &p_level_encoded);
}

int32_t NativeMenu::get_item_count(const RID &p_rid) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NativeMenu::get_class_static()._native_ptr(), StringName("get_item_count")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_rid);
}

bool NativeMenu::is_system_menu(const RID &p_rid) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NativeMenu::get_class_static()._native_ptr(), StringName("is_system_menu")._native_ptr(), 4155700596);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_rid);
}

void NativeMenu::remove_item(const RID &p_rid, int32_t p_idx) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NativeMenu::get_class_static()._native_ptr(), StringName("remove_item")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_rid, &p_idx_encoded);
}

void NativeMenu::clear(const RID &p_rid) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NativeMenu::get_class_static()._native_ptr(), StringName("clear")._native_ptr(), 2722037293);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_rid);
}

} // namespace godot
