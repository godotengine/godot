/**************************************************************************/
/*  input_map.cpp                                                         */
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

#include <godot_cpp/classes/input_map.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/input_event.hpp>

namespace godot {

InputMap *InputMap::singleton = nullptr;

InputMap *InputMap::get_singleton() {
	if (unlikely(singleton == nullptr)) {
		GDExtensionObjectPtr singleton_obj = ::godot::gdextension_interface::global_get_singleton(InputMap::get_class_static()._native_ptr());
#ifdef DEBUG_ENABLED
		ERR_FAIL_NULL_V(singleton_obj, nullptr);
#endif // DEBUG_ENABLED
		singleton = reinterpret_cast<InputMap *>(::godot::gdextension_interface::object_get_instance_binding(singleton_obj, ::godot::gdextension_interface::token, &InputMap::_gde_binding_callbacks));
#ifdef DEBUG_ENABLED
		ERR_FAIL_NULL_V(singleton, nullptr);
#endif // DEBUG_ENABLED
		if (likely(singleton)) {
			ClassDB::_register_engine_singleton(InputMap::get_class_static(), singleton);
		}
	}
	return singleton;
}

InputMap::~InputMap() {
	if (singleton == this) {
		ClassDB::_unregister_engine_singleton(InputMap::get_class_static());
		singleton = nullptr;
	}
}

bool InputMap::has_action(const StringName &p_action) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(InputMap::get_class_static()._native_ptr(), StringName("has_action")._native_ptr(), 2619796661);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_action);
}

TypedArray<StringName> InputMap::get_actions() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(InputMap::get_class_static()._native_ptr(), StringName("get_actions")._native_ptr(), 2915620761);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<StringName>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<StringName>>(_gde_method_bind, _owner);
}

void InputMap::add_action(const StringName &p_action, float p_deadzone) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(InputMap::get_class_static()._native_ptr(), StringName("add_action")._native_ptr(), 1195233573);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_deadzone_encoded;
	PtrToArg<double>::encode(p_deadzone, &p_deadzone_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_action, &p_deadzone_encoded);
}

void InputMap::erase_action(const StringName &p_action) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(InputMap::get_class_static()._native_ptr(), StringName("erase_action")._native_ptr(), 3304788590);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_action);
}

String InputMap::get_action_description(const StringName &p_action) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(InputMap::get_class_static()._native_ptr(), StringName("get_action_description")._native_ptr(), 957595536);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_action);
}

void InputMap::action_set_deadzone(const StringName &p_action, float p_deadzone) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(InputMap::get_class_static()._native_ptr(), StringName("action_set_deadzone")._native_ptr(), 4135858297);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_deadzone_encoded;
	PtrToArg<double>::encode(p_deadzone, &p_deadzone_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_action, &p_deadzone_encoded);
}

float InputMap::action_get_deadzone(const StringName &p_action) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(InputMap::get_class_static()._native_ptr(), StringName("action_get_deadzone")._native_ptr(), 1391627649);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_action);
}

void InputMap::action_add_event(const StringName &p_action, const Ref<InputEvent> &p_event) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(InputMap::get_class_static()._native_ptr(), StringName("action_add_event")._native_ptr(), 518302593);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_action, (p_event != nullptr ? &p_event->_owner : nullptr));
}

bool InputMap::action_has_event(const StringName &p_action, const Ref<InputEvent> &p_event) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(InputMap::get_class_static()._native_ptr(), StringName("action_has_event")._native_ptr(), 1185871985);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_action, (p_event != nullptr ? &p_event->_owner : nullptr));
}

void InputMap::action_erase_event(const StringName &p_action, const Ref<InputEvent> &p_event) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(InputMap::get_class_static()._native_ptr(), StringName("action_erase_event")._native_ptr(), 518302593);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_action, (p_event != nullptr ? &p_event->_owner : nullptr));
}

void InputMap::action_erase_events(const StringName &p_action) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(InputMap::get_class_static()._native_ptr(), StringName("action_erase_events")._native_ptr(), 3304788590);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_action);
}

TypedArray<Ref<InputEvent>> InputMap::action_get_events(const StringName &p_action) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(InputMap::get_class_static()._native_ptr(), StringName("action_get_events")._native_ptr(), 689397652);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<Ref<InputEvent>>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<Ref<InputEvent>>>(_gde_method_bind, _owner, &p_action);
}

bool InputMap::event_is_action(const Ref<InputEvent> &p_event, const StringName &p_action, bool p_exact_match) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(InputMap::get_class_static()._native_ptr(), StringName("event_is_action")._native_ptr(), 3193353650);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int8_t p_exact_match_encoded;
	PtrToArg<bool>::encode(p_exact_match, &p_exact_match_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, (p_event != nullptr ? &p_event->_owner : nullptr), &p_action, &p_exact_match_encoded);
}

void InputMap::load_from_project_settings() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(InputMap::get_class_static()._native_ptr(), StringName("load_from_project_settings")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

} // namespace godot
