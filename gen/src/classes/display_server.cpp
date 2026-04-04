/**************************************************************************/
/*  display_server.cpp                                                    */
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

#include <godot_cpp/classes/display_server.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/image.hpp>
#include <godot_cpp/classes/resource.hpp>
#include <godot_cpp/classes/texture2d.hpp>
#include <godot_cpp/variant/packed_vector2_array.hpp>
#include <godot_cpp/variant/transform2d.hpp>

namespace godot {

DisplayServer *DisplayServer::singleton = nullptr;

DisplayServer *DisplayServer::get_singleton() {
	if (unlikely(singleton == nullptr)) {
		GDExtensionObjectPtr singleton_obj = ::godot::gdextension_interface::global_get_singleton(DisplayServer::get_class_static()._native_ptr());
#ifdef DEBUG_ENABLED
		ERR_FAIL_NULL_V(singleton_obj, nullptr);
#endif // DEBUG_ENABLED
		singleton = reinterpret_cast<DisplayServer *>(::godot::gdextension_interface::object_get_instance_binding(singleton_obj, ::godot::gdextension_interface::token, &DisplayServer::_gde_binding_callbacks));
#ifdef DEBUG_ENABLED
		ERR_FAIL_NULL_V(singleton, nullptr);
#endif // DEBUG_ENABLED
		if (likely(singleton)) {
			ClassDB::_register_engine_singleton(DisplayServer::get_class_static(), singleton);
		}
	}
	return singleton;
}

DisplayServer::~DisplayServer() {
	if (singleton == this) {
		ClassDB::_unregister_engine_singleton(DisplayServer::get_class_static());
		singleton = nullptr;
	}
}

bool DisplayServer::has_feature(DisplayServer::Feature p_feature) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("has_feature")._native_ptr(), 334065950);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_feature_encoded;
	PtrToArg<int64_t>::encode(p_feature, &p_feature_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_feature_encoded);
}

String DisplayServer::get_name() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("get_name")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

void DisplayServer::help_set_search_callbacks(const Callable &p_search_callback, const Callable &p_action_callback) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("help_set_search_callbacks")._native_ptr(), 1687350599);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_search_callback, &p_action_callback);
}

void DisplayServer::global_menu_set_popup_callbacks(const String &p_menu_root, const Callable &p_open_callback, const Callable &p_close_callback) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("global_menu_set_popup_callbacks")._native_ptr(), 3893727526);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_menu_root, &p_open_callback, &p_close_callback);
}

int32_t DisplayServer::global_menu_add_submenu_item(const String &p_menu_root, const String &p_label, const String &p_submenu, int32_t p_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("global_menu_add_submenu_item")._native_ptr(), 2828985934);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_menu_root, &p_label, &p_submenu, &p_index_encoded);
}

int32_t DisplayServer::global_menu_add_item(const String &p_menu_root, const String &p_label, const Callable &p_callback, const Callable &p_key_callback, const Variant &p_tag, Key p_accelerator, int32_t p_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("global_menu_add_item")._native_ptr(), 3616842746);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_accelerator_encoded;
	PtrToArg<int64_t>::encode(p_accelerator, &p_accelerator_encoded);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_menu_root, &p_label, &p_callback, &p_key_callback, &p_tag, &p_accelerator_encoded, &p_index_encoded);
}

int32_t DisplayServer::global_menu_add_check_item(const String &p_menu_root, const String &p_label, const Callable &p_callback, const Callable &p_key_callback, const Variant &p_tag, Key p_accelerator, int32_t p_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("global_menu_add_check_item")._native_ptr(), 3616842746);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_accelerator_encoded;
	PtrToArg<int64_t>::encode(p_accelerator, &p_accelerator_encoded);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_menu_root, &p_label, &p_callback, &p_key_callback, &p_tag, &p_accelerator_encoded, &p_index_encoded);
}

int32_t DisplayServer::global_menu_add_icon_item(const String &p_menu_root, const Ref<Texture2D> &p_icon, const String &p_label, const Callable &p_callback, const Callable &p_key_callback, const Variant &p_tag, Key p_accelerator, int32_t p_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("global_menu_add_icon_item")._native_ptr(), 3867083847);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_accelerator_encoded;
	PtrToArg<int64_t>::encode(p_accelerator, &p_accelerator_encoded);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_menu_root, (p_icon != nullptr ? &p_icon->_owner : nullptr), &p_label, &p_callback, &p_key_callback, &p_tag, &p_accelerator_encoded, &p_index_encoded);
}

int32_t DisplayServer::global_menu_add_icon_check_item(const String &p_menu_root, const Ref<Texture2D> &p_icon, const String &p_label, const Callable &p_callback, const Callable &p_key_callback, const Variant &p_tag, Key p_accelerator, int32_t p_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("global_menu_add_icon_check_item")._native_ptr(), 3867083847);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_accelerator_encoded;
	PtrToArg<int64_t>::encode(p_accelerator, &p_accelerator_encoded);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_menu_root, (p_icon != nullptr ? &p_icon->_owner : nullptr), &p_label, &p_callback, &p_key_callback, &p_tag, &p_accelerator_encoded, &p_index_encoded);
}

int32_t DisplayServer::global_menu_add_radio_check_item(const String &p_menu_root, const String &p_label, const Callable &p_callback, const Callable &p_key_callback, const Variant &p_tag, Key p_accelerator, int32_t p_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("global_menu_add_radio_check_item")._native_ptr(), 3616842746);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_accelerator_encoded;
	PtrToArg<int64_t>::encode(p_accelerator, &p_accelerator_encoded);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_menu_root, &p_label, &p_callback, &p_key_callback, &p_tag, &p_accelerator_encoded, &p_index_encoded);
}

int32_t DisplayServer::global_menu_add_icon_radio_check_item(const String &p_menu_root, const Ref<Texture2D> &p_icon, const String &p_label, const Callable &p_callback, const Callable &p_key_callback, const Variant &p_tag, Key p_accelerator, int32_t p_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("global_menu_add_icon_radio_check_item")._native_ptr(), 3867083847);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_accelerator_encoded;
	PtrToArg<int64_t>::encode(p_accelerator, &p_accelerator_encoded);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_menu_root, (p_icon != nullptr ? &p_icon->_owner : nullptr), &p_label, &p_callback, &p_key_callback, &p_tag, &p_accelerator_encoded, &p_index_encoded);
}

int32_t DisplayServer::global_menu_add_multistate_item(const String &p_menu_root, const String &p_label, int32_t p_max_states, int32_t p_default_state, const Callable &p_callback, const Callable &p_key_callback, const Variant &p_tag, Key p_accelerator, int32_t p_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("global_menu_add_multistate_item")._native_ptr(), 3297554655);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_max_states_encoded;
	PtrToArg<int64_t>::encode(p_max_states, &p_max_states_encoded);
	int64_t p_default_state_encoded;
	PtrToArg<int64_t>::encode(p_default_state, &p_default_state_encoded);
	int64_t p_accelerator_encoded;
	PtrToArg<int64_t>::encode(p_accelerator, &p_accelerator_encoded);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_menu_root, &p_label, &p_max_states_encoded, &p_default_state_encoded, &p_callback, &p_key_callback, &p_tag, &p_accelerator_encoded, &p_index_encoded);
}

int32_t DisplayServer::global_menu_add_separator(const String &p_menu_root, int32_t p_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("global_menu_add_separator")._native_ptr(), 3214812433);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_menu_root, &p_index_encoded);
}

int32_t DisplayServer::global_menu_get_item_index_from_text(const String &p_menu_root, const String &p_text) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("global_menu_get_item_index_from_text")._native_ptr(), 2878152881);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_menu_root, &p_text);
}

int32_t DisplayServer::global_menu_get_item_index_from_tag(const String &p_menu_root, const Variant &p_tag) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("global_menu_get_item_index_from_tag")._native_ptr(), 2941063483);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_menu_root, &p_tag);
}

bool DisplayServer::global_menu_is_item_checked(const String &p_menu_root, int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("global_menu_is_item_checked")._native_ptr(), 3511468594);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_menu_root, &p_idx_encoded);
}

bool DisplayServer::global_menu_is_item_checkable(const String &p_menu_root, int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("global_menu_is_item_checkable")._native_ptr(), 3511468594);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_menu_root, &p_idx_encoded);
}

bool DisplayServer::global_menu_is_item_radio_checkable(const String &p_menu_root, int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("global_menu_is_item_radio_checkable")._native_ptr(), 3511468594);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_menu_root, &p_idx_encoded);
}

Callable DisplayServer::global_menu_get_item_callback(const String &p_menu_root, int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("global_menu_get_item_callback")._native_ptr(), 748666903);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Callable()));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<Callable>(_gde_method_bind, _owner, &p_menu_root, &p_idx_encoded);
}

Callable DisplayServer::global_menu_get_item_key_callback(const String &p_menu_root, int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("global_menu_get_item_key_callback")._native_ptr(), 748666903);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Callable()));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<Callable>(_gde_method_bind, _owner, &p_menu_root, &p_idx_encoded);
}

Variant DisplayServer::global_menu_get_item_tag(const String &p_menu_root, int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("global_menu_get_item_tag")._native_ptr(), 330672633);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Variant()));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<Variant>(_gde_method_bind, _owner, &p_menu_root, &p_idx_encoded);
}

String DisplayServer::global_menu_get_item_text(const String &p_menu_root, int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("global_menu_get_item_text")._native_ptr(), 591067909);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_menu_root, &p_idx_encoded);
}

String DisplayServer::global_menu_get_item_submenu(const String &p_menu_root, int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("global_menu_get_item_submenu")._native_ptr(), 591067909);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_menu_root, &p_idx_encoded);
}

Key DisplayServer::global_menu_get_item_accelerator(const String &p_menu_root, int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("global_menu_get_item_accelerator")._native_ptr(), 936065394);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Key(0)));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return (Key)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_menu_root, &p_idx_encoded);
}

bool DisplayServer::global_menu_is_item_disabled(const String &p_menu_root, int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("global_menu_is_item_disabled")._native_ptr(), 3511468594);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_menu_root, &p_idx_encoded);
}

bool DisplayServer::global_menu_is_item_hidden(const String &p_menu_root, int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("global_menu_is_item_hidden")._native_ptr(), 3511468594);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_menu_root, &p_idx_encoded);
}

String DisplayServer::global_menu_get_item_tooltip(const String &p_menu_root, int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("global_menu_get_item_tooltip")._native_ptr(), 591067909);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_menu_root, &p_idx_encoded);
}

int32_t DisplayServer::global_menu_get_item_state(const String &p_menu_root, int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("global_menu_get_item_state")._native_ptr(), 3422818498);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_menu_root, &p_idx_encoded);
}

int32_t DisplayServer::global_menu_get_item_max_states(const String &p_menu_root, int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("global_menu_get_item_max_states")._native_ptr(), 3422818498);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_menu_root, &p_idx_encoded);
}

Ref<Texture2D> DisplayServer::global_menu_get_item_icon(const String &p_menu_root, int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("global_menu_get_item_icon")._native_ptr(), 3591713183);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Texture2D>()));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return Ref<Texture2D>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Texture2D>(_gde_method_bind, _owner, &p_menu_root, &p_idx_encoded));
}

int32_t DisplayServer::global_menu_get_item_indentation_level(const String &p_menu_root, int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("global_menu_get_item_indentation_level")._native_ptr(), 3422818498);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_menu_root, &p_idx_encoded);
}

void DisplayServer::global_menu_set_item_checked(const String &p_menu_root, int32_t p_idx, bool p_checked) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("global_menu_set_item_checked")._native_ptr(), 4108344793);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	int8_t p_checked_encoded;
	PtrToArg<bool>::encode(p_checked, &p_checked_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_menu_root, &p_idx_encoded, &p_checked_encoded);
}

void DisplayServer::global_menu_set_item_checkable(const String &p_menu_root, int32_t p_idx, bool p_checkable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("global_menu_set_item_checkable")._native_ptr(), 4108344793);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	int8_t p_checkable_encoded;
	PtrToArg<bool>::encode(p_checkable, &p_checkable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_menu_root, &p_idx_encoded, &p_checkable_encoded);
}

void DisplayServer::global_menu_set_item_radio_checkable(const String &p_menu_root, int32_t p_idx, bool p_checkable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("global_menu_set_item_radio_checkable")._native_ptr(), 4108344793);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	int8_t p_checkable_encoded;
	PtrToArg<bool>::encode(p_checkable, &p_checkable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_menu_root, &p_idx_encoded, &p_checkable_encoded);
}

void DisplayServer::global_menu_set_item_callback(const String &p_menu_root, int32_t p_idx, const Callable &p_callback) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("global_menu_set_item_callback")._native_ptr(), 3809915389);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_menu_root, &p_idx_encoded, &p_callback);
}

void DisplayServer::global_menu_set_item_hover_callbacks(const String &p_menu_root, int32_t p_idx, const Callable &p_callback) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("global_menu_set_item_hover_callbacks")._native_ptr(), 3809915389);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_menu_root, &p_idx_encoded, &p_callback);
}

void DisplayServer::global_menu_set_item_key_callback(const String &p_menu_root, int32_t p_idx, const Callable &p_key_callback) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("global_menu_set_item_key_callback")._native_ptr(), 3809915389);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_menu_root, &p_idx_encoded, &p_key_callback);
}

void DisplayServer::global_menu_set_item_tag(const String &p_menu_root, int32_t p_idx, const Variant &p_tag) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("global_menu_set_item_tag")._native_ptr(), 453659863);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_menu_root, &p_idx_encoded, &p_tag);
}

void DisplayServer::global_menu_set_item_text(const String &p_menu_root, int32_t p_idx, const String &p_text) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("global_menu_set_item_text")._native_ptr(), 965966136);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_menu_root, &p_idx_encoded, &p_text);
}

void DisplayServer::global_menu_set_item_submenu(const String &p_menu_root, int32_t p_idx, const String &p_submenu) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("global_menu_set_item_submenu")._native_ptr(), 965966136);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_menu_root, &p_idx_encoded, &p_submenu);
}

void DisplayServer::global_menu_set_item_accelerator(const String &p_menu_root, int32_t p_idx, Key p_keycode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("global_menu_set_item_accelerator")._native_ptr(), 566943293);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	int64_t p_keycode_encoded;
	PtrToArg<int64_t>::encode(p_keycode, &p_keycode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_menu_root, &p_idx_encoded, &p_keycode_encoded);
}

void DisplayServer::global_menu_set_item_disabled(const String &p_menu_root, int32_t p_idx, bool p_disabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("global_menu_set_item_disabled")._native_ptr(), 4108344793);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	int8_t p_disabled_encoded;
	PtrToArg<bool>::encode(p_disabled, &p_disabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_menu_root, &p_idx_encoded, &p_disabled_encoded);
}

void DisplayServer::global_menu_set_item_hidden(const String &p_menu_root, int32_t p_idx, bool p_hidden) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("global_menu_set_item_hidden")._native_ptr(), 4108344793);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	int8_t p_hidden_encoded;
	PtrToArg<bool>::encode(p_hidden, &p_hidden_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_menu_root, &p_idx_encoded, &p_hidden_encoded);
}

void DisplayServer::global_menu_set_item_tooltip(const String &p_menu_root, int32_t p_idx, const String &p_tooltip) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("global_menu_set_item_tooltip")._native_ptr(), 965966136);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_menu_root, &p_idx_encoded, &p_tooltip);
}

void DisplayServer::global_menu_set_item_state(const String &p_menu_root, int32_t p_idx, int32_t p_state) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("global_menu_set_item_state")._native_ptr(), 3474840532);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	int64_t p_state_encoded;
	PtrToArg<int64_t>::encode(p_state, &p_state_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_menu_root, &p_idx_encoded, &p_state_encoded);
}

void DisplayServer::global_menu_set_item_max_states(const String &p_menu_root, int32_t p_idx, int32_t p_max_states) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("global_menu_set_item_max_states")._native_ptr(), 3474840532);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	int64_t p_max_states_encoded;
	PtrToArg<int64_t>::encode(p_max_states, &p_max_states_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_menu_root, &p_idx_encoded, &p_max_states_encoded);
}

void DisplayServer::global_menu_set_item_icon(const String &p_menu_root, int32_t p_idx, const Ref<Texture2D> &p_icon) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("global_menu_set_item_icon")._native_ptr(), 3201338066);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_menu_root, &p_idx_encoded, (p_icon != nullptr ? &p_icon->_owner : nullptr));
}

void DisplayServer::global_menu_set_item_indentation_level(const String &p_menu_root, int32_t p_idx, int32_t p_level) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("global_menu_set_item_indentation_level")._native_ptr(), 3474840532);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	int64_t p_level_encoded;
	PtrToArg<int64_t>::encode(p_level, &p_level_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_menu_root, &p_idx_encoded, &p_level_encoded);
}

int32_t DisplayServer::global_menu_get_item_count(const String &p_menu_root) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("global_menu_get_item_count")._native_ptr(), 1321353865);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_menu_root);
}

void DisplayServer::global_menu_remove_item(const String &p_menu_root, int32_t p_idx) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("global_menu_remove_item")._native_ptr(), 2956805083);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_menu_root, &p_idx_encoded);
}

void DisplayServer::global_menu_clear(const String &p_menu_root) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("global_menu_clear")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_menu_root);
}

Dictionary DisplayServer::global_menu_get_system_menu_roots() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("global_menu_get_system_menu_roots")._native_ptr(), 3102165223);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Dictionary()));
	return ::godot::internal::_call_native_mb_ret<Dictionary>(_gde_method_bind, _owner);
}

bool DisplayServer::tts_is_speaking() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("tts_is_speaking")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

bool DisplayServer::tts_is_paused() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("tts_is_paused")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

TypedArray<Dictionary> DisplayServer::tts_get_voices() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("tts_get_voices")._native_ptr(), 3995934104);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<Dictionary>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<Dictionary>>(_gde_method_bind, _owner);
}

PackedStringArray DisplayServer::tts_get_voices_for_language(const String &p_language) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("tts_get_voices_for_language")._native_ptr(), 4291131558);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedStringArray()));
	return ::godot::internal::_call_native_mb_ret<PackedStringArray>(_gde_method_bind, _owner, &p_language);
}

void DisplayServer::tts_speak(const String &p_text, const String &p_voice, int32_t p_volume, float p_pitch, float p_rate, int64_t p_utterance_id, bool p_interrupt) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("tts_speak")._native_ptr(), 903992738);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_volume_encoded;
	PtrToArg<int64_t>::encode(p_volume, &p_volume_encoded);
	double p_pitch_encoded;
	PtrToArg<double>::encode(p_pitch, &p_pitch_encoded);
	double p_rate_encoded;
	PtrToArg<double>::encode(p_rate, &p_rate_encoded);
	int64_t p_utterance_id_encoded;
	PtrToArg<int64_t>::encode(p_utterance_id, &p_utterance_id_encoded);
	int8_t p_interrupt_encoded;
	PtrToArg<bool>::encode(p_interrupt, &p_interrupt_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_text, &p_voice, &p_volume_encoded, &p_pitch_encoded, &p_rate_encoded, &p_utterance_id_encoded, &p_interrupt_encoded);
}

void DisplayServer::tts_pause() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("tts_pause")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void DisplayServer::tts_resume() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("tts_resume")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void DisplayServer::tts_stop() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("tts_stop")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void DisplayServer::tts_set_utterance_callback(DisplayServer::TTSUtteranceEvent p_event, const Callable &p_callable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("tts_set_utterance_callback")._native_ptr(), 109679083);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_event_encoded;
	PtrToArg<int64_t>::encode(p_event, &p_event_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_event_encoded, &p_callable);
}

bool DisplayServer::is_dark_mode_supported() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("is_dark_mode_supported")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

bool DisplayServer::is_dark_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("is_dark_mode")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

Color DisplayServer::get_accent_color() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("get_accent_color")._native_ptr(), 3444240500);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Color()));
	return ::godot::internal::_call_native_mb_ret<Color>(_gde_method_bind, _owner);
}

Color DisplayServer::get_base_color() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("get_base_color")._native_ptr(), 3444240500);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Color()));
	return ::godot::internal::_call_native_mb_ret<Color>(_gde_method_bind, _owner);
}

void DisplayServer::set_system_theme_change_callback(const Callable &p_callable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("set_system_theme_change_callback")._native_ptr(), 1611583062);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_callable);
}

void DisplayServer::mouse_set_mode(DisplayServer::MouseMode p_mouse_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("mouse_set_mode")._native_ptr(), 348288463);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mouse_mode_encoded;
	PtrToArg<int64_t>::encode(p_mouse_mode, &p_mouse_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mouse_mode_encoded);
}

DisplayServer::MouseMode DisplayServer::mouse_get_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("mouse_get_mode")._native_ptr(), 1353961651);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (DisplayServer::MouseMode(0)));
	return (DisplayServer::MouseMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void DisplayServer::warp_mouse(const Vector2i &p_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("warp_mouse")._native_ptr(), 1130785943);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_position);
}

Vector2i DisplayServer::mouse_get_position() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("mouse_get_position")._native_ptr(), 3690982128);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2i()));
	return ::godot::internal::_call_native_mb_ret<Vector2i>(_gde_method_bind, _owner);
}

BitField<MouseButtonMask> DisplayServer::mouse_get_button_state() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("mouse_get_button_state")._native_ptr(), 2512161324);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (BitField<MouseButtonMask>(0)));
	return (int64_t)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void DisplayServer::clipboard_set(const String &p_clipboard) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("clipboard_set")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_clipboard);
}

String DisplayServer::clipboard_get() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("clipboard_get")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

Ref<Image> DisplayServer::clipboard_get_image() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("clipboard_get_image")._native_ptr(), 4190603485);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Image>()));
	return Ref<Image>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Image>(_gde_method_bind, _owner));
}

bool DisplayServer::clipboard_has() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("clipboard_has")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

bool DisplayServer::clipboard_has_image() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("clipboard_has_image")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void DisplayServer::clipboard_set_primary(const String &p_clipboard_primary) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("clipboard_set_primary")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_clipboard_primary);
}

String DisplayServer::clipboard_get_primary() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("clipboard_get_primary")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

TypedArray<Rect2> DisplayServer::get_display_cutouts() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("get_display_cutouts")._native_ptr(), 3995934104);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<Rect2>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<Rect2>>(_gde_method_bind, _owner);
}

Rect2i DisplayServer::get_display_safe_area() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("get_display_safe_area")._native_ptr(), 410525958);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Rect2i()));
	return ::godot::internal::_call_native_mb_ret<Rect2i>(_gde_method_bind, _owner);
}

int32_t DisplayServer::get_screen_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("get_screen_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

int32_t DisplayServer::get_primary_screen() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("get_primary_screen")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

int32_t DisplayServer::get_keyboard_focus_screen() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("get_keyboard_focus_screen")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

int32_t DisplayServer::get_screen_from_rect(const Rect2 &p_rect) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("get_screen_from_rect")._native_ptr(), 741354659);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_rect);
}

Vector2i DisplayServer::screen_get_position(int32_t p_screen) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("screen_get_position")._native_ptr(), 1725937825);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2i()));
	int64_t p_screen_encoded;
	PtrToArg<int64_t>::encode(p_screen, &p_screen_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector2i>(_gde_method_bind, _owner, &p_screen_encoded);
}

Vector2i DisplayServer::screen_get_size(int32_t p_screen) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("screen_get_size")._native_ptr(), 1725937825);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2i()));
	int64_t p_screen_encoded;
	PtrToArg<int64_t>::encode(p_screen, &p_screen_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector2i>(_gde_method_bind, _owner, &p_screen_encoded);
}

Rect2i DisplayServer::screen_get_usable_rect(int32_t p_screen) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("screen_get_usable_rect")._native_ptr(), 2439012528);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Rect2i()));
	int64_t p_screen_encoded;
	PtrToArg<int64_t>::encode(p_screen, &p_screen_encoded);
	return ::godot::internal::_call_native_mb_ret<Rect2i>(_gde_method_bind, _owner, &p_screen_encoded);
}

int32_t DisplayServer::screen_get_dpi(int32_t p_screen) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("screen_get_dpi")._native_ptr(), 181039630);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_screen_encoded;
	PtrToArg<int64_t>::encode(p_screen, &p_screen_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_screen_encoded);
}

float DisplayServer::screen_get_scale(int32_t p_screen) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("screen_get_scale")._native_ptr(), 909105437);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_screen_encoded;
	PtrToArg<int64_t>::encode(p_screen, &p_screen_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_screen_encoded);
}

bool DisplayServer::is_touchscreen_available() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("is_touchscreen_available")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

float DisplayServer::screen_get_max_scale() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("screen_get_max_scale")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

float DisplayServer::screen_get_refresh_rate(int32_t p_screen) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("screen_get_refresh_rate")._native_ptr(), 909105437);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_screen_encoded;
	PtrToArg<int64_t>::encode(p_screen, &p_screen_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_screen_encoded);
}

Color DisplayServer::screen_get_pixel(const Vector2i &p_position) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("screen_get_pixel")._native_ptr(), 1532707496);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Color()));
	return ::godot::internal::_call_native_mb_ret<Color>(_gde_method_bind, _owner, &p_position);
}

Ref<Image> DisplayServer::screen_get_image(int32_t p_screen) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("screen_get_image")._native_ptr(), 3813388802);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Image>()));
	int64_t p_screen_encoded;
	PtrToArg<int64_t>::encode(p_screen, &p_screen_encoded);
	return Ref<Image>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Image>(_gde_method_bind, _owner, &p_screen_encoded));
}

Ref<Image> DisplayServer::screen_get_image_rect(const Rect2i &p_rect) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("screen_get_image_rect")._native_ptr(), 2601441065);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Image>()));
	return Ref<Image>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Image>(_gde_method_bind, _owner, &p_rect));
}

void DisplayServer::screen_set_orientation(DisplayServer::ScreenOrientation p_orientation, int32_t p_screen) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("screen_set_orientation")._native_ptr(), 2211511631);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_orientation_encoded;
	PtrToArg<int64_t>::encode(p_orientation, &p_orientation_encoded);
	int64_t p_screen_encoded;
	PtrToArg<int64_t>::encode(p_screen, &p_screen_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_orientation_encoded, &p_screen_encoded);
}

DisplayServer::ScreenOrientation DisplayServer::screen_get_orientation(int32_t p_screen) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("screen_get_orientation")._native_ptr(), 133818562);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (DisplayServer::ScreenOrientation(0)));
	int64_t p_screen_encoded;
	PtrToArg<int64_t>::encode(p_screen, &p_screen_encoded);
	return (DisplayServer::ScreenOrientation)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_screen_encoded);
}

void DisplayServer::screen_set_keep_on(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("screen_set_keep_on")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool DisplayServer::screen_is_kept_on() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("screen_is_kept_on")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

PackedInt32Array DisplayServer::get_window_list() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("get_window_list")._native_ptr(), 1930428628);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedInt32Array()));
	return ::godot::internal::_call_native_mb_ret<PackedInt32Array>(_gde_method_bind, _owner);
}

int32_t DisplayServer::get_window_at_screen_position(const Vector2i &p_position) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("get_window_at_screen_position")._native_ptr(), 2485466453);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_position);
}

int64_t DisplayServer::window_get_native_handle(DisplayServer::HandleType p_handle_type, int32_t p_window_id) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("window_get_native_handle")._native_ptr(), 1096425680);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_handle_type_encoded;
	PtrToArg<int64_t>::encode(p_handle_type, &p_handle_type_encoded);
	int64_t p_window_id_encoded;
	PtrToArg<int64_t>::encode(p_window_id, &p_window_id_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_handle_type_encoded, &p_window_id_encoded);
}

int32_t DisplayServer::window_get_active_popup() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("window_get_active_popup")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void DisplayServer::window_set_popup_safe_rect(int32_t p_window, const Rect2i &p_rect) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("window_set_popup_safe_rect")._native_ptr(), 3317281434);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_window_encoded;
	PtrToArg<int64_t>::encode(p_window, &p_window_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_window_encoded, &p_rect);
}

Rect2i DisplayServer::window_get_popup_safe_rect(int32_t p_window) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("window_get_popup_safe_rect")._native_ptr(), 2161169500);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Rect2i()));
	int64_t p_window_encoded;
	PtrToArg<int64_t>::encode(p_window, &p_window_encoded);
	return ::godot::internal::_call_native_mb_ret<Rect2i>(_gde_method_bind, _owner, &p_window_encoded);
}

void DisplayServer::window_set_title(const String &p_title, int32_t p_window_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("window_set_title")._native_ptr(), 441246282);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_window_id_encoded;
	PtrToArg<int64_t>::encode(p_window_id, &p_window_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_title, &p_window_id_encoded);
}

Vector2i DisplayServer::window_get_title_size(const String &p_title, int32_t p_window_id) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("window_get_title_size")._native_ptr(), 2925301799);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2i()));
	int64_t p_window_id_encoded;
	PtrToArg<int64_t>::encode(p_window_id, &p_window_id_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector2i>(_gde_method_bind, _owner, &p_title, &p_window_id_encoded);
}

void DisplayServer::window_set_mouse_passthrough(const PackedVector2Array &p_region, int32_t p_window_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("window_set_mouse_passthrough")._native_ptr(), 1993637420);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_window_id_encoded;
	PtrToArg<int64_t>::encode(p_window_id, &p_window_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_region, &p_window_id_encoded);
}

int32_t DisplayServer::window_get_current_screen(int32_t p_window_id) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("window_get_current_screen")._native_ptr(), 1591665591);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_window_id_encoded;
	PtrToArg<int64_t>::encode(p_window_id, &p_window_id_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_window_id_encoded);
}

void DisplayServer::window_set_current_screen(int32_t p_screen, int32_t p_window_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("window_set_current_screen")._native_ptr(), 2230941749);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_screen_encoded;
	PtrToArg<int64_t>::encode(p_screen, &p_screen_encoded);
	int64_t p_window_id_encoded;
	PtrToArg<int64_t>::encode(p_window_id, &p_window_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_screen_encoded, &p_window_id_encoded);
}

Vector2i DisplayServer::window_get_position(int32_t p_window_id) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("window_get_position")._native_ptr(), 763922886);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2i()));
	int64_t p_window_id_encoded;
	PtrToArg<int64_t>::encode(p_window_id, &p_window_id_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector2i>(_gde_method_bind, _owner, &p_window_id_encoded);
}

Vector2i DisplayServer::window_get_position_with_decorations(int32_t p_window_id) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("window_get_position_with_decorations")._native_ptr(), 763922886);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2i()));
	int64_t p_window_id_encoded;
	PtrToArg<int64_t>::encode(p_window_id, &p_window_id_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector2i>(_gde_method_bind, _owner, &p_window_id_encoded);
}

void DisplayServer::window_set_position(const Vector2i &p_position, int32_t p_window_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("window_set_position")._native_ptr(), 2019273902);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_window_id_encoded;
	PtrToArg<int64_t>::encode(p_window_id, &p_window_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_position, &p_window_id_encoded);
}

Vector2i DisplayServer::window_get_size(int32_t p_window_id) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("window_get_size")._native_ptr(), 763922886);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2i()));
	int64_t p_window_id_encoded;
	PtrToArg<int64_t>::encode(p_window_id, &p_window_id_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector2i>(_gde_method_bind, _owner, &p_window_id_encoded);
}

void DisplayServer::window_set_size(const Vector2i &p_size, int32_t p_window_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("window_set_size")._native_ptr(), 2019273902);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_window_id_encoded;
	PtrToArg<int64_t>::encode(p_window_id, &p_window_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_size, &p_window_id_encoded);
}

void DisplayServer::window_set_rect_changed_callback(const Callable &p_callback, int32_t p_window_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("window_set_rect_changed_callback")._native_ptr(), 1091192925);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_window_id_encoded;
	PtrToArg<int64_t>::encode(p_window_id, &p_window_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_callback, &p_window_id_encoded);
}

void DisplayServer::window_set_window_event_callback(const Callable &p_callback, int32_t p_window_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("window_set_window_event_callback")._native_ptr(), 1091192925);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_window_id_encoded;
	PtrToArg<int64_t>::encode(p_window_id, &p_window_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_callback, &p_window_id_encoded);
}

void DisplayServer::window_set_input_event_callback(const Callable &p_callback, int32_t p_window_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("window_set_input_event_callback")._native_ptr(), 1091192925);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_window_id_encoded;
	PtrToArg<int64_t>::encode(p_window_id, &p_window_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_callback, &p_window_id_encoded);
}

void DisplayServer::window_set_input_text_callback(const Callable &p_callback, int32_t p_window_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("window_set_input_text_callback")._native_ptr(), 1091192925);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_window_id_encoded;
	PtrToArg<int64_t>::encode(p_window_id, &p_window_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_callback, &p_window_id_encoded);
}

void DisplayServer::window_set_drop_files_callback(const Callable &p_callback, int32_t p_window_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("window_set_drop_files_callback")._native_ptr(), 1091192925);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_window_id_encoded;
	PtrToArg<int64_t>::encode(p_window_id, &p_window_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_callback, &p_window_id_encoded);
}

uint64_t DisplayServer::window_get_attached_instance_id(int32_t p_window_id) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("window_get_attached_instance_id")._native_ptr(), 1591665591);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_window_id_encoded;
	PtrToArg<int64_t>::encode(p_window_id, &p_window_id_encoded);
	return ::godot::internal::_call_native_mb_ret<uint64_t>(_gde_method_bind, _owner, &p_window_id_encoded);
}

Vector2i DisplayServer::window_get_max_size(int32_t p_window_id) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("window_get_max_size")._native_ptr(), 763922886);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2i()));
	int64_t p_window_id_encoded;
	PtrToArg<int64_t>::encode(p_window_id, &p_window_id_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector2i>(_gde_method_bind, _owner, &p_window_id_encoded);
}

void DisplayServer::window_set_max_size(const Vector2i &p_max_size, int32_t p_window_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("window_set_max_size")._native_ptr(), 2019273902);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_window_id_encoded;
	PtrToArg<int64_t>::encode(p_window_id, &p_window_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_max_size, &p_window_id_encoded);
}

Vector2i DisplayServer::window_get_min_size(int32_t p_window_id) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("window_get_min_size")._native_ptr(), 763922886);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2i()));
	int64_t p_window_id_encoded;
	PtrToArg<int64_t>::encode(p_window_id, &p_window_id_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector2i>(_gde_method_bind, _owner, &p_window_id_encoded);
}

void DisplayServer::window_set_min_size(const Vector2i &p_min_size, int32_t p_window_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("window_set_min_size")._native_ptr(), 2019273902);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_window_id_encoded;
	PtrToArg<int64_t>::encode(p_window_id, &p_window_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_min_size, &p_window_id_encoded);
}

Vector2i DisplayServer::window_get_size_with_decorations(int32_t p_window_id) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("window_get_size_with_decorations")._native_ptr(), 763922886);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2i()));
	int64_t p_window_id_encoded;
	PtrToArg<int64_t>::encode(p_window_id, &p_window_id_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector2i>(_gde_method_bind, _owner, &p_window_id_encoded);
}

DisplayServer::WindowMode DisplayServer::window_get_mode(int32_t p_window_id) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("window_get_mode")._native_ptr(), 2185728461);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (DisplayServer::WindowMode(0)));
	int64_t p_window_id_encoded;
	PtrToArg<int64_t>::encode(p_window_id, &p_window_id_encoded);
	return (DisplayServer::WindowMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_window_id_encoded);
}

void DisplayServer::window_set_mode(DisplayServer::WindowMode p_mode, int32_t p_window_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("window_set_mode")._native_ptr(), 1319965401);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	int64_t p_window_id_encoded;
	PtrToArg<int64_t>::encode(p_window_id, &p_window_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded, &p_window_id_encoded);
}

void DisplayServer::window_set_flag(DisplayServer::WindowFlags p_flag, bool p_enabled, int32_t p_window_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("window_set_flag")._native_ptr(), 254894155);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_flag_encoded;
	PtrToArg<int64_t>::encode(p_flag, &p_flag_encoded);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	int64_t p_window_id_encoded;
	PtrToArg<int64_t>::encode(p_window_id, &p_window_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_flag_encoded, &p_enabled_encoded, &p_window_id_encoded);
}

bool DisplayServer::window_get_flag(DisplayServer::WindowFlags p_flag, int32_t p_window_id) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("window_get_flag")._native_ptr(), 802816991);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_flag_encoded;
	PtrToArg<int64_t>::encode(p_flag, &p_flag_encoded);
	int64_t p_window_id_encoded;
	PtrToArg<int64_t>::encode(p_window_id, &p_window_id_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_flag_encoded, &p_window_id_encoded);
}

void DisplayServer::window_set_window_buttons_offset(const Vector2i &p_offset, int32_t p_window_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("window_set_window_buttons_offset")._native_ptr(), 2019273902);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_window_id_encoded;
	PtrToArg<int64_t>::encode(p_window_id, &p_window_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_offset, &p_window_id_encoded);
}

Vector3i DisplayServer::window_get_safe_title_margins(int32_t p_window_id) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("window_get_safe_title_margins")._native_ptr(), 2295066620);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3i()));
	int64_t p_window_id_encoded;
	PtrToArg<int64_t>::encode(p_window_id, &p_window_id_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector3i>(_gde_method_bind, _owner, &p_window_id_encoded);
}

void DisplayServer::window_request_attention(int32_t p_window_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("window_request_attention")._native_ptr(), 1995695955);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_window_id_encoded;
	PtrToArg<int64_t>::encode(p_window_id, &p_window_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_window_id_encoded);
}

void DisplayServer::window_move_to_foreground(int32_t p_window_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("window_move_to_foreground")._native_ptr(), 1995695955);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_window_id_encoded;
	PtrToArg<int64_t>::encode(p_window_id, &p_window_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_window_id_encoded);
}

bool DisplayServer::window_is_focused(int32_t p_window_id) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("window_is_focused")._native_ptr(), 1051549951);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_window_id_encoded;
	PtrToArg<int64_t>::encode(p_window_id, &p_window_id_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_window_id_encoded);
}

bool DisplayServer::window_can_draw(int32_t p_window_id) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("window_can_draw")._native_ptr(), 1051549951);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_window_id_encoded;
	PtrToArg<int64_t>::encode(p_window_id, &p_window_id_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_window_id_encoded);
}

void DisplayServer::window_set_transient(int32_t p_window_id, int32_t p_parent_window_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("window_set_transient")._native_ptr(), 3937882851);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_window_id_encoded;
	PtrToArg<int64_t>::encode(p_window_id, &p_window_id_encoded);
	int64_t p_parent_window_id_encoded;
	PtrToArg<int64_t>::encode(p_parent_window_id, &p_parent_window_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_window_id_encoded, &p_parent_window_id_encoded);
}

void DisplayServer::window_set_exclusive(int32_t p_window_id, bool p_exclusive) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("window_set_exclusive")._native_ptr(), 300928843);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_window_id_encoded;
	PtrToArg<int64_t>::encode(p_window_id, &p_window_id_encoded);
	int8_t p_exclusive_encoded;
	PtrToArg<bool>::encode(p_exclusive, &p_exclusive_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_window_id_encoded, &p_exclusive_encoded);
}

void DisplayServer::window_set_ime_active(bool p_active, int32_t p_window_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("window_set_ime_active")._native_ptr(), 1661950165);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_active_encoded;
	PtrToArg<bool>::encode(p_active, &p_active_encoded);
	int64_t p_window_id_encoded;
	PtrToArg<int64_t>::encode(p_window_id, &p_window_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_active_encoded, &p_window_id_encoded);
}

void DisplayServer::window_set_ime_position(const Vector2i &p_position, int32_t p_window_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("window_set_ime_position")._native_ptr(), 2019273902);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_window_id_encoded;
	PtrToArg<int64_t>::encode(p_window_id, &p_window_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_position, &p_window_id_encoded);
}

void DisplayServer::window_set_vsync_mode(DisplayServer::VSyncMode p_vsync_mode, int32_t p_window_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("window_set_vsync_mode")._native_ptr(), 2179333492);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_vsync_mode_encoded;
	PtrToArg<int64_t>::encode(p_vsync_mode, &p_vsync_mode_encoded);
	int64_t p_window_id_encoded;
	PtrToArg<int64_t>::encode(p_window_id, &p_window_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_vsync_mode_encoded, &p_window_id_encoded);
}

DisplayServer::VSyncMode DisplayServer::window_get_vsync_mode(int32_t p_window_id) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("window_get_vsync_mode")._native_ptr(), 578873795);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (DisplayServer::VSyncMode(0)));
	int64_t p_window_id_encoded;
	PtrToArg<int64_t>::encode(p_window_id, &p_window_id_encoded);
	return (DisplayServer::VSyncMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_window_id_encoded);
}

bool DisplayServer::window_is_maximize_allowed(int32_t p_window_id) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("window_is_maximize_allowed")._native_ptr(), 1051549951);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_window_id_encoded;
	PtrToArg<int64_t>::encode(p_window_id, &p_window_id_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_window_id_encoded);
}

bool DisplayServer::window_maximize_on_title_dbl_click() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("window_maximize_on_title_dbl_click")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

bool DisplayServer::window_minimize_on_title_dbl_click() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("window_minimize_on_title_dbl_click")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void DisplayServer::window_start_drag(int32_t p_window_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("window_start_drag")._native_ptr(), 1995695955);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_window_id_encoded;
	PtrToArg<int64_t>::encode(p_window_id, &p_window_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_window_id_encoded);
}

void DisplayServer::window_start_resize(DisplayServer::WindowResizeEdge p_edge, int32_t p_window_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("window_start_resize")._native_ptr(), 4009722312);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_edge_encoded;
	PtrToArg<int64_t>::encode(p_edge, &p_edge_encoded);
	int64_t p_window_id_encoded;
	PtrToArg<int64_t>::encode(p_window_id, &p_window_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_edge_encoded, &p_window_id_encoded);
}

void DisplayServer::window_set_color(const Color &p_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("window_set_color")._native_ptr(), 2920490490);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_color);
}

int32_t DisplayServer::accessibility_should_increase_contrast() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_should_increase_contrast")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

int32_t DisplayServer::accessibility_should_reduce_animation() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_should_reduce_animation")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

int32_t DisplayServer::accessibility_should_reduce_transparency() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_should_reduce_transparency")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

int32_t DisplayServer::accessibility_screen_reader_active() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_screen_reader_active")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

RID DisplayServer::accessibility_create_element(int32_t p_window_id, DisplayServer::AccessibilityRole p_role) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_create_element")._native_ptr(), 2968347744);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	int64_t p_window_id_encoded;
	PtrToArg<int64_t>::encode(p_window_id, &p_window_id_encoded);
	int64_t p_role_encoded;
	PtrToArg<int64_t>::encode(p_role, &p_role_encoded);
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_window_id_encoded, &p_role_encoded);
}

RID DisplayServer::accessibility_create_sub_element(const RID &p_parent_rid, DisplayServer::AccessibilityRole p_role, int32_t p_insert_pos) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_create_sub_element")._native_ptr(), 1949948826);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	int64_t p_role_encoded;
	PtrToArg<int64_t>::encode(p_role, &p_role_encoded);
	int64_t p_insert_pos_encoded;
	PtrToArg<int64_t>::encode(p_insert_pos, &p_insert_pos_encoded);
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_parent_rid, &p_role_encoded, &p_insert_pos_encoded);
}

RID DisplayServer::accessibility_create_sub_text_edit_elements(const RID &p_parent_rid, const RID &p_shaped_text, float p_min_height, int32_t p_insert_pos, bool p_is_last_line) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_create_sub_text_edit_elements")._native_ptr(), 2702009895);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	double p_min_height_encoded;
	PtrToArg<double>::encode(p_min_height, &p_min_height_encoded);
	int64_t p_insert_pos_encoded;
	PtrToArg<int64_t>::encode(p_insert_pos, &p_insert_pos_encoded);
	int8_t p_is_last_line_encoded;
	PtrToArg<bool>::encode(p_is_last_line, &p_is_last_line_encoded);
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_parent_rid, &p_shaped_text, &p_min_height_encoded, &p_insert_pos_encoded, &p_is_last_line_encoded);
}

bool DisplayServer::accessibility_has_element(const RID &p_id) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_has_element")._native_ptr(), 4155700596);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_id);
}

void DisplayServer::accessibility_free_element(const RID &p_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_free_element")._native_ptr(), 2722037293);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id);
}

void DisplayServer::accessibility_element_set_meta(const RID &p_id, const Variant &p_meta) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_element_set_meta")._native_ptr(), 3175752987);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id, &p_meta);
}

Variant DisplayServer::accessibility_element_get_meta(const RID &p_id) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_element_get_meta")._native_ptr(), 4171304767);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Variant()));
	return ::godot::internal::_call_native_mb_ret<Variant>(_gde_method_bind, _owner, &p_id);
}

void DisplayServer::accessibility_set_window_rect(int32_t p_window_id, const Rect2 &p_rect_out, const Rect2 &p_rect_in) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_set_window_rect")._native_ptr(), 2386961724);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_window_id_encoded;
	PtrToArg<int64_t>::encode(p_window_id, &p_window_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_window_id_encoded, &p_rect_out, &p_rect_in);
}

void DisplayServer::accessibility_set_window_focused(int32_t p_window_id, bool p_focused) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_set_window_focused")._native_ptr(), 300928843);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_window_id_encoded;
	PtrToArg<int64_t>::encode(p_window_id, &p_window_id_encoded);
	int8_t p_focused_encoded;
	PtrToArg<bool>::encode(p_focused, &p_focused_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_window_id_encoded, &p_focused_encoded);
}

void DisplayServer::accessibility_update_set_focus(const RID &p_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_update_set_focus")._native_ptr(), 2722037293);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id);
}

RID DisplayServer::accessibility_get_window_root(int32_t p_window_id) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_get_window_root")._native_ptr(), 495598643);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	int64_t p_window_id_encoded;
	PtrToArg<int64_t>::encode(p_window_id, &p_window_id_encoded);
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_window_id_encoded);
}

void DisplayServer::accessibility_update_set_role(const RID &p_id, DisplayServer::AccessibilityRole p_role) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_update_set_role")._native_ptr(), 3352768215);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_role_encoded;
	PtrToArg<int64_t>::encode(p_role, &p_role_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id, &p_role_encoded);
}

void DisplayServer::accessibility_update_set_name(const RID &p_id, const String &p_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_update_set_name")._native_ptr(), 2726140452);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id, &p_name);
}

void DisplayServer::accessibility_update_set_extra_info(const RID &p_id, const String &p_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_update_set_extra_info")._native_ptr(), 2726140452);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id, &p_name);
}

void DisplayServer::accessibility_update_set_description(const RID &p_id, const String &p_description) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_update_set_description")._native_ptr(), 2726140452);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id, &p_description);
}

void DisplayServer::accessibility_update_set_value(const RID &p_id, const String &p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_update_set_value")._native_ptr(), 2726140452);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id, &p_value);
}

void DisplayServer::accessibility_update_set_tooltip(const RID &p_id, const String &p_tooltip) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_update_set_tooltip")._native_ptr(), 2726140452);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id, &p_tooltip);
}

void DisplayServer::accessibility_update_set_bounds(const RID &p_id, const Rect2 &p_rect) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_update_set_bounds")._native_ptr(), 1378122625);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id, &p_rect);
}

void DisplayServer::accessibility_update_set_transform(const RID &p_id, const Transform2D &p_transform) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_update_set_transform")._native_ptr(), 1246044741);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id, &p_transform);
}

void DisplayServer::accessibility_update_add_child(const RID &p_id, const RID &p_child_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_update_add_child")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id, &p_child_id);
}

void DisplayServer::accessibility_update_add_related_controls(const RID &p_id, const RID &p_related_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_update_add_related_controls")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id, &p_related_id);
}

void DisplayServer::accessibility_update_add_related_details(const RID &p_id, const RID &p_related_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_update_add_related_details")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id, &p_related_id);
}

void DisplayServer::accessibility_update_add_related_described_by(const RID &p_id, const RID &p_related_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_update_add_related_described_by")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id, &p_related_id);
}

void DisplayServer::accessibility_update_add_related_flow_to(const RID &p_id, const RID &p_related_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_update_add_related_flow_to")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id, &p_related_id);
}

void DisplayServer::accessibility_update_add_related_labeled_by(const RID &p_id, const RID &p_related_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_update_add_related_labeled_by")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id, &p_related_id);
}

void DisplayServer::accessibility_update_add_related_radio_group(const RID &p_id, const RID &p_related_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_update_add_related_radio_group")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id, &p_related_id);
}

void DisplayServer::accessibility_update_set_active_descendant(const RID &p_id, const RID &p_other_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_update_set_active_descendant")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id, &p_other_id);
}

void DisplayServer::accessibility_update_set_next_on_line(const RID &p_id, const RID &p_other_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_update_set_next_on_line")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id, &p_other_id);
}

void DisplayServer::accessibility_update_set_previous_on_line(const RID &p_id, const RID &p_other_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_update_set_previous_on_line")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id, &p_other_id);
}

void DisplayServer::accessibility_update_set_member_of(const RID &p_id, const RID &p_group_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_update_set_member_of")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id, &p_group_id);
}

void DisplayServer::accessibility_update_set_in_page_link_target(const RID &p_id, const RID &p_other_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_update_set_in_page_link_target")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id, &p_other_id);
}

void DisplayServer::accessibility_update_set_error_message(const RID &p_id, const RID &p_other_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_update_set_error_message")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id, &p_other_id);
}

void DisplayServer::accessibility_update_set_live(const RID &p_id, DisplayServer::AccessibilityLiveMode p_live) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_update_set_live")._native_ptr(), 2683302212);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_live_encoded;
	PtrToArg<int64_t>::encode(p_live, &p_live_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id, &p_live_encoded);
}

void DisplayServer::accessibility_update_add_action(const RID &p_id, DisplayServer::AccessibilityAction p_action, const Callable &p_callable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_update_add_action")._native_ptr(), 2898696987);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_action_encoded;
	PtrToArg<int64_t>::encode(p_action, &p_action_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id, &p_action_encoded, &p_callable);
}

void DisplayServer::accessibility_update_add_custom_action(const RID &p_id, int32_t p_action_id, const String &p_action_description) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_update_add_custom_action")._native_ptr(), 4153150897);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_action_id_encoded;
	PtrToArg<int64_t>::encode(p_action_id, &p_action_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id, &p_action_id_encoded, &p_action_description);
}

void DisplayServer::accessibility_update_set_table_row_count(const RID &p_id, int32_t p_count) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_update_set_table_row_count")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_count_encoded;
	PtrToArg<int64_t>::encode(p_count, &p_count_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id, &p_count_encoded);
}

void DisplayServer::accessibility_update_set_table_column_count(const RID &p_id, int32_t p_count) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_update_set_table_column_count")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_count_encoded;
	PtrToArg<int64_t>::encode(p_count, &p_count_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id, &p_count_encoded);
}

void DisplayServer::accessibility_update_set_table_row_index(const RID &p_id, int32_t p_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_update_set_table_row_index")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id, &p_index_encoded);
}

void DisplayServer::accessibility_update_set_table_column_index(const RID &p_id, int32_t p_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_update_set_table_column_index")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id, &p_index_encoded);
}

void DisplayServer::accessibility_update_set_table_cell_position(const RID &p_id, int32_t p_row_index, int32_t p_column_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_update_set_table_cell_position")._native_ptr(), 4288446313);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_row_index_encoded;
	PtrToArg<int64_t>::encode(p_row_index, &p_row_index_encoded);
	int64_t p_column_index_encoded;
	PtrToArg<int64_t>::encode(p_column_index, &p_column_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id, &p_row_index_encoded, &p_column_index_encoded);
}

void DisplayServer::accessibility_update_set_table_cell_span(const RID &p_id, int32_t p_row_span, int32_t p_column_span) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_update_set_table_cell_span")._native_ptr(), 4288446313);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_row_span_encoded;
	PtrToArg<int64_t>::encode(p_row_span, &p_row_span_encoded);
	int64_t p_column_span_encoded;
	PtrToArg<int64_t>::encode(p_column_span, &p_column_span_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id, &p_row_span_encoded, &p_column_span_encoded);
}

void DisplayServer::accessibility_update_set_list_item_count(const RID &p_id, int32_t p_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_update_set_list_item_count")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id, &p_size_encoded);
}

void DisplayServer::accessibility_update_set_list_item_index(const RID &p_id, int32_t p_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_update_set_list_item_index")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id, &p_index_encoded);
}

void DisplayServer::accessibility_update_set_list_item_level(const RID &p_id, int32_t p_level) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_update_set_list_item_level")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_level_encoded;
	PtrToArg<int64_t>::encode(p_level, &p_level_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id, &p_level_encoded);
}

void DisplayServer::accessibility_update_set_list_item_selected(const RID &p_id, bool p_selected) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_update_set_list_item_selected")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_selected_encoded;
	PtrToArg<bool>::encode(p_selected, &p_selected_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id, &p_selected_encoded);
}

void DisplayServer::accessibility_update_set_list_item_expanded(const RID &p_id, bool p_expanded) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_update_set_list_item_expanded")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_expanded_encoded;
	PtrToArg<bool>::encode(p_expanded, &p_expanded_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id, &p_expanded_encoded);
}

void DisplayServer::accessibility_update_set_popup_type(const RID &p_id, DisplayServer::AccessibilityPopupType p_popup) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_update_set_popup_type")._native_ptr(), 2040885448);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_popup_encoded;
	PtrToArg<int64_t>::encode(p_popup, &p_popup_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id, &p_popup_encoded);
}

void DisplayServer::accessibility_update_set_checked(const RID &p_id, bool p_checekd) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_update_set_checked")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_checekd_encoded;
	PtrToArg<bool>::encode(p_checekd, &p_checekd_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id, &p_checekd_encoded);
}

void DisplayServer::accessibility_update_set_num_value(const RID &p_id, double p_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_update_set_num_value")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_position_encoded;
	PtrToArg<double>::encode(p_position, &p_position_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id, &p_position_encoded);
}

void DisplayServer::accessibility_update_set_num_range(const RID &p_id, double p_min, double p_max) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_update_set_num_range")._native_ptr(), 2513314492);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_min_encoded;
	PtrToArg<double>::encode(p_min, &p_min_encoded);
	double p_max_encoded;
	PtrToArg<double>::encode(p_max, &p_max_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id, &p_min_encoded, &p_max_encoded);
}

void DisplayServer::accessibility_update_set_num_step(const RID &p_id, double p_step) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_update_set_num_step")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_step_encoded;
	PtrToArg<double>::encode(p_step, &p_step_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id, &p_step_encoded);
}

void DisplayServer::accessibility_update_set_num_jump(const RID &p_id, double p_jump) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_update_set_num_jump")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_jump_encoded;
	PtrToArg<double>::encode(p_jump, &p_jump_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id, &p_jump_encoded);
}

void DisplayServer::accessibility_update_set_scroll_x(const RID &p_id, double p_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_update_set_scroll_x")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_position_encoded;
	PtrToArg<double>::encode(p_position, &p_position_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id, &p_position_encoded);
}

void DisplayServer::accessibility_update_set_scroll_x_range(const RID &p_id, double p_min, double p_max) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_update_set_scroll_x_range")._native_ptr(), 2513314492);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_min_encoded;
	PtrToArg<double>::encode(p_min, &p_min_encoded);
	double p_max_encoded;
	PtrToArg<double>::encode(p_max, &p_max_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id, &p_min_encoded, &p_max_encoded);
}

void DisplayServer::accessibility_update_set_scroll_y(const RID &p_id, double p_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_update_set_scroll_y")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_position_encoded;
	PtrToArg<double>::encode(p_position, &p_position_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id, &p_position_encoded);
}

void DisplayServer::accessibility_update_set_scroll_y_range(const RID &p_id, double p_min, double p_max) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_update_set_scroll_y_range")._native_ptr(), 2513314492);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_min_encoded;
	PtrToArg<double>::encode(p_min, &p_min_encoded);
	double p_max_encoded;
	PtrToArg<double>::encode(p_max, &p_max_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id, &p_min_encoded, &p_max_encoded);
}

void DisplayServer::accessibility_update_set_text_decorations(const RID &p_id, bool p_underline, bool p_strikethrough, bool p_overline) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_update_set_text_decorations")._native_ptr(), 1672422386);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_underline_encoded;
	PtrToArg<bool>::encode(p_underline, &p_underline_encoded);
	int8_t p_strikethrough_encoded;
	PtrToArg<bool>::encode(p_strikethrough, &p_strikethrough_encoded);
	int8_t p_overline_encoded;
	PtrToArg<bool>::encode(p_overline, &p_overline_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id, &p_underline_encoded, &p_strikethrough_encoded, &p_overline_encoded);
}

void DisplayServer::accessibility_update_set_text_align(const RID &p_id, HorizontalAlignment p_align) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_update_set_text_align")._native_ptr(), 3725995085);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_align_encoded;
	PtrToArg<int64_t>::encode(p_align, &p_align_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id, &p_align_encoded);
}

void DisplayServer::accessibility_update_set_text_selection(const RID &p_id, const RID &p_text_start_id, int32_t p_start_char, const RID &p_text_end_id, int32_t p_end_char) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_update_set_text_selection")._native_ptr(), 3119144029);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_start_char_encoded;
	PtrToArg<int64_t>::encode(p_start_char, &p_start_char_encoded);
	int64_t p_end_char_encoded;
	PtrToArg<int64_t>::encode(p_end_char, &p_end_char_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id, &p_text_start_id, &p_start_char_encoded, &p_text_end_id, &p_end_char_encoded);
}

void DisplayServer::accessibility_update_set_flag(const RID &p_id, DisplayServer::AccessibilityFlags p_flag, bool p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_update_set_flag")._native_ptr(), 3758675396);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_flag_encoded;
	PtrToArg<int64_t>::encode(p_flag, &p_flag_encoded);
	int8_t p_value_encoded;
	PtrToArg<bool>::encode(p_value, &p_value_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id, &p_flag_encoded, &p_value_encoded);
}

void DisplayServer::accessibility_update_set_classname(const RID &p_id, const String &p_classname) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_update_set_classname")._native_ptr(), 2726140452);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id, &p_classname);
}

void DisplayServer::accessibility_update_set_placeholder(const RID &p_id, const String &p_placeholder) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_update_set_placeholder")._native_ptr(), 2726140452);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id, &p_placeholder);
}

void DisplayServer::accessibility_update_set_language(const RID &p_id, const String &p_language) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_update_set_language")._native_ptr(), 2726140452);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id, &p_language);
}

void DisplayServer::accessibility_update_set_text_orientation(const RID &p_id, bool p_vertical) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_update_set_text_orientation")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_vertical_encoded;
	PtrToArg<bool>::encode(p_vertical, &p_vertical_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id, &p_vertical_encoded);
}

void DisplayServer::accessibility_update_set_list_orientation(const RID &p_id, bool p_vertical) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_update_set_list_orientation")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_vertical_encoded;
	PtrToArg<bool>::encode(p_vertical, &p_vertical_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id, &p_vertical_encoded);
}

void DisplayServer::accessibility_update_set_shortcut(const RID &p_id, const String &p_shortcut) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_update_set_shortcut")._native_ptr(), 2726140452);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id, &p_shortcut);
}

void DisplayServer::accessibility_update_set_url(const RID &p_id, const String &p_url) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_update_set_url")._native_ptr(), 2726140452);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id, &p_url);
}

void DisplayServer::accessibility_update_set_role_description(const RID &p_id, const String &p_description) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_update_set_role_description")._native_ptr(), 2726140452);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id, &p_description);
}

void DisplayServer::accessibility_update_set_state_description(const RID &p_id, const String &p_description) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_update_set_state_description")._native_ptr(), 2726140452);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id, &p_description);
}

void DisplayServer::accessibility_update_set_color_value(const RID &p_id, const Color &p_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_update_set_color_value")._native_ptr(), 2948539648);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id, &p_color);
}

void DisplayServer::accessibility_update_set_background_color(const RID &p_id, const Color &p_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_update_set_background_color")._native_ptr(), 2948539648);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id, &p_color);
}

void DisplayServer::accessibility_update_set_foreground_color(const RID &p_id, const Color &p_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("accessibility_update_set_foreground_color")._native_ptr(), 2948539648);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id, &p_color);
}

Vector2i DisplayServer::ime_get_selection() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("ime_get_selection")._native_ptr(), 3690982128);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2i()));
	return ::godot::internal::_call_native_mb_ret<Vector2i>(_gde_method_bind, _owner);
}

String DisplayServer::ime_get_text() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("ime_get_text")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

void DisplayServer::virtual_keyboard_show(const String &p_existing_text, const Rect2 &p_position, DisplayServer::VirtualKeyboardType p_type, int32_t p_max_length, int32_t p_cursor_start, int32_t p_cursor_end) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("virtual_keyboard_show")._native_ptr(), 3042891259);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_type_encoded;
	PtrToArg<int64_t>::encode(p_type, &p_type_encoded);
	int64_t p_max_length_encoded;
	PtrToArg<int64_t>::encode(p_max_length, &p_max_length_encoded);
	int64_t p_cursor_start_encoded;
	PtrToArg<int64_t>::encode(p_cursor_start, &p_cursor_start_encoded);
	int64_t p_cursor_end_encoded;
	PtrToArg<int64_t>::encode(p_cursor_end, &p_cursor_end_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_existing_text, &p_position, &p_type_encoded, &p_max_length_encoded, &p_cursor_start_encoded, &p_cursor_end_encoded);
}

void DisplayServer::virtual_keyboard_hide() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("virtual_keyboard_hide")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

int32_t DisplayServer::virtual_keyboard_get_height() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("virtual_keyboard_get_height")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

bool DisplayServer::has_hardware_keyboard() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("has_hardware_keyboard")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void DisplayServer::set_hardware_keyboard_connection_change_callback(const Callable &p_callable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("set_hardware_keyboard_connection_change_callback")._native_ptr(), 1611583062);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_callable);
}

void DisplayServer::cursor_set_shape(DisplayServer::CursorShape p_shape) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("cursor_set_shape")._native_ptr(), 2026291549);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_shape_encoded;
	PtrToArg<int64_t>::encode(p_shape, &p_shape_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_shape_encoded);
}

DisplayServer::CursorShape DisplayServer::cursor_get_shape() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("cursor_get_shape")._native_ptr(), 1087724927);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (DisplayServer::CursorShape(0)));
	return (DisplayServer::CursorShape)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void DisplayServer::cursor_set_custom_image(const Ref<Resource> &p_cursor, DisplayServer::CursorShape p_shape, const Vector2 &p_hotspot) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("cursor_set_custom_image")._native_ptr(), 1816663697);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_shape_encoded;
	PtrToArg<int64_t>::encode(p_shape, &p_shape_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_cursor != nullptr ? &p_cursor->_owner : nullptr), &p_shape_encoded, &p_hotspot);
}

bool DisplayServer::get_swap_cancel_ok() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("get_swap_cancel_ok")._native_ptr(), 2240911060);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void DisplayServer::enable_for_stealing_focus(int64_t p_process_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("enable_for_stealing_focus")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_process_id_encoded;
	PtrToArg<int64_t>::encode(p_process_id, &p_process_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_process_id_encoded);
}

Error DisplayServer::dialog_show(const String &p_title, const String &p_description, const PackedStringArray &p_buttons, const Callable &p_callback) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("dialog_show")._native_ptr(), 4115553226);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_title, &p_description, &p_buttons, &p_callback);
}

Error DisplayServer::dialog_input_text(const String &p_title, const String &p_description, const String &p_existing_text, const Callable &p_callback) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("dialog_input_text")._native_ptr(), 3088703427);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_title, &p_description, &p_existing_text, &p_callback);
}

Error DisplayServer::file_dialog_show(const String &p_title, const String &p_current_directory, const String &p_filename, bool p_show_hidden, DisplayServer::FileDialogMode p_mode, const PackedStringArray &p_filters, const Callable &p_callback, int32_t p_parent_window_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("file_dialog_show")._native_ptr(), 1386825884);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	int8_t p_show_hidden_encoded;
	PtrToArg<bool>::encode(p_show_hidden, &p_show_hidden_encoded);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	int64_t p_parent_window_id_encoded;
	PtrToArg<int64_t>::encode(p_parent_window_id, &p_parent_window_id_encoded);
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_title, &p_current_directory, &p_filename, &p_show_hidden_encoded, &p_mode_encoded, &p_filters, &p_callback, &p_parent_window_id_encoded);
}

Error DisplayServer::file_dialog_with_options_show(const String &p_title, const String &p_current_directory, const String &p_root, const String &p_filename, bool p_show_hidden, DisplayServer::FileDialogMode p_mode, const PackedStringArray &p_filters, const TypedArray<Dictionary> &p_options, const Callable &p_callback, int32_t p_parent_window_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("file_dialog_with_options_show")._native_ptr(), 1448789813);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	int8_t p_show_hidden_encoded;
	PtrToArg<bool>::encode(p_show_hidden, &p_show_hidden_encoded);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	int64_t p_parent_window_id_encoded;
	PtrToArg<int64_t>::encode(p_parent_window_id, &p_parent_window_id_encoded);
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_title, &p_current_directory, &p_root, &p_filename, &p_show_hidden_encoded, &p_mode_encoded, &p_filters, &p_options, &p_callback, &p_parent_window_id_encoded);
}

void DisplayServer::beep() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("beep")._native_ptr(), 4051624405);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

int32_t DisplayServer::keyboard_get_layout_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("keyboard_get_layout_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

int32_t DisplayServer::keyboard_get_current_layout() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("keyboard_get_current_layout")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void DisplayServer::keyboard_set_current_layout(int32_t p_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("keyboard_set_current_layout")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded);
}

String DisplayServer::keyboard_get_layout_language(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("keyboard_get_layout_language")._native_ptr(), 844755477);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_index_encoded);
}

String DisplayServer::keyboard_get_layout_name(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("keyboard_get_layout_name")._native_ptr(), 844755477);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_index_encoded);
}

Key DisplayServer::keyboard_get_keycode_from_physical(Key p_keycode) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("keyboard_get_keycode_from_physical")._native_ptr(), 3447613187);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Key(0)));
	int64_t p_keycode_encoded;
	PtrToArg<int64_t>::encode(p_keycode, &p_keycode_encoded);
	return (Key)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_keycode_encoded);
}

Key DisplayServer::keyboard_get_label_from_physical(Key p_keycode) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("keyboard_get_label_from_physical")._native_ptr(), 3447613187);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Key(0)));
	int64_t p_keycode_encoded;
	PtrToArg<int64_t>::encode(p_keycode, &p_keycode_encoded);
	return (Key)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_keycode_encoded);
}

void DisplayServer::show_emoji_and_symbol_picker() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("show_emoji_and_symbol_picker")._native_ptr(), 4051624405);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

bool DisplayServer::color_picker(const Callable &p_callback) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("color_picker")._native_ptr(), 151643214);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_callback);
}

void DisplayServer::process_events() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("process_events")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void DisplayServer::force_process_and_drop_events() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("force_process_and_drop_events")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void DisplayServer::set_native_icon(const String &p_filename) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("set_native_icon")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_filename);
}

void DisplayServer::set_icon(const Ref<Image> &p_image) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("set_icon")._native_ptr(), 532598488);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_image != nullptr ? &p_image->_owner : nullptr));
}

int32_t DisplayServer::create_status_indicator(const Ref<Texture2D> &p_icon, const String &p_tooltip, const Callable &p_callback) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("create_status_indicator")._native_ptr(), 1904285171);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, (p_icon != nullptr ? &p_icon->_owner : nullptr), &p_tooltip, &p_callback);
}

void DisplayServer::status_indicator_set_icon(int32_t p_id, const Ref<Texture2D> &p_icon) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("status_indicator_set_icon")._native_ptr(), 666127730);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_id_encoded;
	PtrToArg<int64_t>::encode(p_id, &p_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id_encoded, (p_icon != nullptr ? &p_icon->_owner : nullptr));
}

void DisplayServer::status_indicator_set_tooltip(int32_t p_id, const String &p_tooltip) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("status_indicator_set_tooltip")._native_ptr(), 501894301);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_id_encoded;
	PtrToArg<int64_t>::encode(p_id, &p_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id_encoded, &p_tooltip);
}

void DisplayServer::status_indicator_set_menu(int32_t p_id, const RID &p_menu_rid) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("status_indicator_set_menu")._native_ptr(), 4040184819);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_id_encoded;
	PtrToArg<int64_t>::encode(p_id, &p_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id_encoded, &p_menu_rid);
}

void DisplayServer::status_indicator_set_callback(int32_t p_id, const Callable &p_callback) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("status_indicator_set_callback")._native_ptr(), 957362965);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_id_encoded;
	PtrToArg<int64_t>::encode(p_id, &p_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id_encoded, &p_callback);
}

Rect2 DisplayServer::status_indicator_get_rect(int32_t p_id) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("status_indicator_get_rect")._native_ptr(), 3327874267);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Rect2()));
	int64_t p_id_encoded;
	PtrToArg<int64_t>::encode(p_id, &p_id_encoded);
	return ::godot::internal::_call_native_mb_ret<Rect2>(_gde_method_bind, _owner, &p_id_encoded);
}

void DisplayServer::delete_status_indicator(int32_t p_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("delete_status_indicator")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_id_encoded;
	PtrToArg<int64_t>::encode(p_id, &p_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id_encoded);
}

int32_t DisplayServer::tablet_get_driver_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("tablet_get_driver_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

String DisplayServer::tablet_get_driver_name(int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("tablet_get_driver_name")._native_ptr(), 844755477);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_idx_encoded);
}

String DisplayServer::tablet_get_current_driver() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("tablet_get_current_driver")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

void DisplayServer::tablet_set_current_driver(const String &p_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("tablet_set_current_driver")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name);
}

bool DisplayServer::is_window_transparency_available() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("is_window_transparency_available")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void DisplayServer::register_additional_output(Object *p_object) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("register_additional_output")._native_ptr(), 3975164845);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_object != nullptr ? &p_object->_owner : nullptr));
}

void DisplayServer::unregister_additional_output(Object *p_object) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("unregister_additional_output")._native_ptr(), 3975164845);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_object != nullptr ? &p_object->_owner : nullptr));
}

bool DisplayServer::has_additional_outputs() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(DisplayServer::get_class_static()._native_ptr(), StringName("has_additional_outputs")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

} // namespace godot
